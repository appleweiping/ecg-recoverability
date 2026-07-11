"""Conditional 1-D DDPM reduced-lead ECG reconstructor (GPU).

A real generative reconstructor of the class the ECG literature uses (denoising
diffusion, cf. arXiv:2401.05388), so the fabrication finding is not a straw man.
The denoiser is a 1-D U-Net that predicts the noise added to a full 12-lead window,
conditioned on the observed leads (concatenated as extra input channels with a
binary lead mask). Condition dropout during training enables classifier-free
guidance (CFG), whose scale $w$ is the perception knob: small $w$ -> conditional-mean,
low fabrication; large $w$ -> sharp, realistic, high fabrication.

Sampling supports (a) plain conditional ancestral sampling, (b) CFG, and (c)
measurement replacement (RePaint-style inpainting: overwrite observed leads with the
known values each step) so the reconstruction is data-consistent by construction and
realism cannot be dismissed as ignoring the observation.

Everything here is torch; import lazily so the CPU package stays torch-optional.
"""
from __future__ import annotations

import math

import numpy as np


def _t():
    import torch
    return torch


# --------------------------------------------------------------------------- schedule
def cosine_beta_schedule(T: int, s: float = 0.008):
    torch = _t()
    steps = T + 1
    x = torch.linspace(0, T, steps)
    ac = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    ac = ac / ac[0]
    betas = 1 - (ac[1:] / ac[:-1])
    return torch.clip(betas, 1e-4, 0.999)


# --------------------------------------------------------------------------- U-Net
def _sinusoidal(t, dim):
    torch = _t()
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
    a = t[:, None].float() * freqs[None]
    return torch.cat([torch.sin(a), torch.cos(a)], dim=-1)


def _make_unet(in_ch, out_ch, base=64, tdim=256):
    torch = _t(); nn = torch.nn

    class Block(nn.Module):
        def __init__(self, ci, co):
            super().__init__()
            self.norm1 = nn.GroupNorm(8, ci)
            self.conv1 = nn.Conv1d(ci, co, 5, padding=2)
            self.temb = nn.Linear(tdim, co)
            self.norm2 = nn.GroupNorm(8, co)
            self.conv2 = nn.Conv1d(co, co, 5, padding=2)
            self.skip = nn.Conv1d(ci, co, 1) if ci != co else nn.Identity()

        def forward(self, x, temb):
            h = self.conv1(torch.nn.functional.silu(self.norm1(x)))
            h = h + self.temb(temb)[..., None]
            h = self.conv2(torch.nn.functional.silu(self.norm2(h)))
            return h + self.skip(x)

    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.temb = nn.Sequential(nn.Linear(tdim, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
            self.inp = nn.Conv1d(in_ch, base, 5, padding=2)
            self.d1 = Block(base, base); self.d2 = Block(base, base * 2)
            self.d3 = Block(base * 2, base * 4)
            self.down = nn.AvgPool1d(2)
            self.mid = Block(base * 4, base * 4)
            self.u3 = Block(base * 4 + base * 4, base * 2)
            self.u2 = Block(base * 2 + base * 2, base)
            self.u1 = Block(base + base, base)
            self.up = nn.Upsample(scale_factor=2, mode="nearest")
            self.out = nn.Sequential(nn.GroupNorm(8, base), nn.SiLU(),
                                     nn.Conv1d(base, out_ch, 5, padding=2))
            self.tdim = tdim

        def forward(self, x, t, cond):
            temb = self.temb(_sinusoidal(t, self.tdim))
            h = self.inp(torch.cat([x, cond], dim=1))
            h1 = self.d1(h, temb)
            h2 = self.d2(self.down(h1), temb)
            h3 = self.d3(self.down(h2), temb)
            m = self.mid(h3, temb)
            u = self.u3(torch.cat([m, h3], 1), temb)
            u = self.u2(torch.cat([self.up(u), h2], 1), temb)
            u = self.u1(torch.cat([self.up(u), h1], 1), temb)
            return self.out(u)

    return UNet()


# --------------------------------------------------------------------------- model
class DiffusionReconstructor:
    """Conditional DDPM over 12-lead windows, conditioned on observed leads.

    Window length must be divisible by 4 (two downsamplings).
    """

    def __init__(self, obs_idx, T=200, base=64, device="cuda", cond_dropout=0.1, seed=0):
        torch = _t()
        torch.manual_seed(seed)
        self.obs_idx = list(obs_idx)
        self.device = device
        self.T = T
        self.cond_dropout = cond_dropout
        # input channels: 12 (noisy leads) + 12 (observed condition, unobserved zeroed)
        # + 1 (lead mask broadcast). output: 12 (predicted noise).
        self.net = _make_unet(12 + 12 + 1, 12, base=base).to(device)
        betas = cosine_beta_schedule(T).to(device)
        self.betas = betas
        self.alphas = 1 - betas
        self.acp = torch.cumprod(self.alphas, 0)          # alpha-bar
        mask = torch.zeros(12, device=device)
        mask[self.obs_idx] = 1.0
        self.lead_mask = mask                              # (12,)

    def _cond(self, x0):
        """Build the conditioning tensor from a clean batch x0 (B,12,W)."""
        torch = _t()
        B, _, W = x0.shape
        obs = x0 * self.lead_mask[None, :, None]           # observed leads, rest zero
        maskc = self.lead_mask[None, :, None].expand(B, 12, W).mean(1, keepdim=True)
        return torch.cat([obs, maskc], dim=1)              # (B, 13, W)

    def train(self, X, epochs=40, bs=128, lr=2e-4, log_every=5):
        """X: (N,12,W) float32 numpy (per-window z-scored upstream)."""
        torch = _t()
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        Xt = torch.tensor(X, dtype=torch.float32)
        n = Xt.shape[0]
        for ep in range(epochs):
            perm = torch.randperm(n)
            tot = 0.0
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                x0 = Xt[idx].to(self.device)
                cond = self._cond(x0)
                # classifier-free: randomly drop condition
                if self.cond_dropout > 0:
                    drop = (torch.rand(x0.shape[0], device=self.device) < self.cond_dropout)
                    cond = cond * (~drop)[:, None, None]
                t = torch.randint(0, self.T, (x0.shape[0],), device=self.device)
                noise = torch.randn_like(x0)
                ac = self.acp[t][:, None, None]
                xt = ac.sqrt() * x0 + (1 - ac).sqrt() * noise
                pred = self.net(xt, t, cond)
                loss = ((pred - noise) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item() * x0.shape[0]
            if (ep + 1) % log_every == 0 or ep == 0:
                print(f"  [ddpm] epoch {ep+1}/{epochs} loss={tot/n:.5f}", flush=True)
        return self

    def sample(self, y_full, guidance=1.0, replace=True, steps=None, seed=0):
        """Reconstruct from observed leads.

        y_full: (B,12,W) with observed leads set to their true values (others
        arbitrary; only observed rows are read). guidance: CFG scale w. w=1 is the
        plain conditional model (no guidance term); w>1 sharpens toward the
        conditional mode (more realistic, more fabrication); w=0 is fully
        unconditional. replace: RePaint measurement replacement of observed leads.
        Returns (B,12,W) numpy.
        """
        torch = _t()
        self.net.eval()
        g = torch.Generator(device=self.device).manual_seed(seed)
        yt = torch.tensor(y_full, dtype=torch.float32, device=self.device)
        cond = self._cond(yt)
        zero_cond = torch.zeros_like(cond)
        B, _, W = yt.shape
        x = torch.randn(B, 12, W, device=self.device, generator=g)
        ts = list(range(self.T))[::-1]
        clip = 6.0                                     # x0 clamp (robust-normalized units)
        m = self.lead_mask[None, :, None]
        with torch.no_grad():
            for t in ts:
                tt = torch.full((B,), t, device=self.device, dtype=torch.long)
                eps = self.net(x, tt, cond)
                if guidance != 1.0:                            # w=1 is plain conditional
                    eps_u = self.net(x, tt, zero_cond)
                    eps = eps_u + guidance * (eps - eps_u)     # CFG: eps_u + w*(cond-uncond)
                ac = self.acp[t]
                ac_prev = self.acp[t - 1] if t > 0 else torch.tensor(1.0, device=self.device)
                beta = self.betas[t]
                # predict + clamp x0 (Tweedie), then form the DDPM posterior mean.
                x0 = (x - (1 - ac).sqrt() * eps) / ac.sqrt()
                x0 = torch.clamp(x0, -clip, clip)
                if replace:
                    # anchor observed leads to the (noise-free) measurement
                    x0 = x0 * (1 - m) + yt * m
                coef_x0 = ac_prev.sqrt() * beta / (1 - ac)
                coef_xt = self.alphas[t].sqrt() * (1 - ac_prev) / (1 - ac)
                mean = coef_x0 * x0 + coef_xt * x
                if t > 0:
                    var = beta * (1 - ac_prev) / (1 - ac)
                    mean = mean + var.sqrt() * torch.randn(x.shape, device=self.device, generator=g)
                x = mean
        if replace:
            x = x * (1 - m) + yt * m
        return x.cpu().numpy()
