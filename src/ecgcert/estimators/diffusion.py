"""Arbitrary-mask conditional 1-D DDPM reduced-lead ECG reconstructor (GPU).

A single mask-agnostic generative reconstructor (cf. diffusion ECG inpainting,
arXiv:2401.05388). The denoiser is a 1-D U-Net predicting the noise added to a full
12-lead window, conditioned on (i) the observed-lead values (unobserved leads zeroed)
and (ii) the full 12-channel BINARY observation mask. During training a fresh lead
mask is sampled per example, so ONE model serves any observed subset -- there is no
per-configuration model and no target-lead leakage: scoring configuration S conditions
on S's mask only.

(Corrected from an earlier version whose obs_idx was fixed at init to a single
configuration and whose conditioning used a scalar mean(mask); reusing that model for a
different configuration leaked the originally-observed target leads.)

Sampling supports classifier-free guidance (CFG; scale w is the perception knob) and
RePaint measurement replacement (overwrite observed leads with the known values each
step) so reconstructions are data-consistent by construction.

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


# --------------------------------------------------------------------------- masks
# Deployment configurations the model must handle (observed lead indices, 0..11 in the
# clinical order [I,II,III,aVR,aVL,aVF,V1..V6]); used to seed the training-mask mix.
_DEPLOY_CONFIGS = (
    (0, 1, 6, 8, 10),                 # {I, II, V1, V3, V5}  precordial interpolation
    (0, 1, 2, 3, 4, 5),               # limb-6
    (0, 1, 7),                        # {I, II, V2}
    (0,),                             # Lead I
    (1,),                             # Lead II
)


def _sample_masks(torch, B, device, rng, p_config=0.5):
    """Sample B binary lead masks (B,12). With prob p_config use a deployment config;
    else a random subset of 3-8 leads. Ensures eval configs are in-distribution while
    the model still generalises to arbitrary masks."""
    mask = torch.zeros(B, 12, device=device)
    for b in range(B):
        if rng.random() < p_config:
            obs = _DEPLOY_CONFIGS[rng.integers(len(_DEPLOY_CONFIGS))]
        else:
            k = int(rng.integers(3, 9))
            obs = rng.choice(12, size=k, replace=False)
        mask[b, list(obs)] = 1.0
    return mask


# --------------------------------------------------------------------------- model
class DiffusionReconstructor:
    """Arbitrary-mask conditional DDPM over 12-lead windows.

    Window length must be divisible by 4 (two downsamplings). The model takes NO fixed
    observed set; the observed leads are supplied per call via a mask.
    """

    def __init__(self, T=200, base=64, device="cuda", cond_dropout=0.1, seed=0):
        torch = _t()
        torch.manual_seed(seed)
        self.device = device
        self.T = T
        self.cond_dropout = cond_dropout
        self._mask_rng = np.random.default_rng(seed)
        # input channels: 12 (noisy leads) + 12 (observed values, unobserved zeroed)
        # + 12 (binary observation mask). output: 12 (predicted noise).
        self.net = _make_unet(12 + 12 + 12, 12, base=base).to(device)
        betas = cosine_beta_schedule(T).to(device)
        self.betas = betas
        self.alphas = 1 - betas
        self.acp = torch.cumprod(self.alphas, 0)          # alpha-bar

    def _mask_from_obs(self, obs_idx, B, W):
        torch = _t()
        m = torch.zeros(12, device=self.device)
        m[list(obs_idx)] = 1.0
        return m[None, :, None].expand(B, 12, W)

    def _cond(self, x0, mask_bcw):
        """Conditioning tensor from clean batch x0 (B,12,W) and mask (B,12,W)."""
        torch = _t()
        obs = x0 * mask_bcw                                # observed values, rest zero
        return torch.cat([obs, mask_bcw], dim=1)           # (B, 24, W)

    def train(self, X, epochs=40, bs=64, lr=2e-4, log_every=5):
        """X: (N,12,W) float32 numpy (per-window normalised upstream). A fresh random
        lead mask is sampled per example each step (arbitrary-mask training)."""
        torch = _t()
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        Xt = torch.tensor(X, dtype=torch.float32)
        n, _, W = Xt.shape
        for ep in range(epochs):
            perm = torch.randperm(n)
            tot = 0.0
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                x0 = Xt[idx].to(self.device)
                B = x0.shape[0]
                mask = _sample_masks(torch, B, self.device, self._mask_rng)[:, :, None].expand(B, 12, W)
                cond = self._cond(x0, mask)
                if self.cond_dropout > 0:                  # classifier-free: drop condition
                    drop = (torch.rand(B, device=self.device) < self.cond_dropout)
                    cond = cond * (~drop)[:, None, None]
                t = torch.randint(0, self.T, (B,), device=self.device)
                noise = torch.randn_like(x0)
                ac = self.acp[t][:, None, None]
                xt = ac.sqrt() * x0 + (1 - ac).sqrt() * noise
                pred = self.net(xt, t, cond)
                loss = ((pred - noise) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item() * B
            if (ep + 1) % log_every == 0 or ep == 0:
                print(f"  [ddpm] epoch {ep+1}/{epochs} loss={tot/n:.5f}", flush=True)
        return self

    def sample(self, y_full, obs_idx, guidance=1.0, replace=True, steps=None, seed=0):
        """Reconstruct from the leads in ``obs_idx`` (a tuple/list of indices).

        y_full: (B,12,W) with the observed leads set to their true values (others
        arbitrary; only observed rows are read). The model is conditioned ONLY on the
        given mask -- no other lead can leak in. guidance: CFG scale w (1 = plain
        conditional; w>1 sharpens; 0 = unconditional). replace: RePaint replacement of
        observed leads. Returns (B,12,W) numpy.
        """
        torch = _t()
        self.net.eval()
        g = torch.Generator(device=self.device).manual_seed(seed)
        yt = torch.tensor(y_full, dtype=torch.float32, device=self.device)
        B, _, W = yt.shape
        mask = self._mask_from_obs(obs_idx, B, W)          # (B,12,W)
        cond = self._cond(yt, mask)
        zero_cond = torch.zeros_like(cond)
        x = torch.randn(B, 12, W, device=self.device, generator=g)
        ts = list(range(self.T))[::-1]
        clip = 6.0
        m = mask
        with torch.no_grad():
            for t in ts:
                tt = torch.full((B,), t, device=self.device, dtype=torch.long)
                eps = self.net(x, tt, cond)
                if guidance != 1.0:
                    eps_u = self.net(x, tt, zero_cond)
                    eps = eps_u + guidance * (eps - eps_u)
                ac = self.acp[t]
                ac_prev = self.acp[t - 1] if t > 0 else torch.tensor(1.0, device=self.device)
                beta = self.betas[t]
                x0 = (x - (1 - ac).sqrt() * eps) / ac.sqrt()
                x0 = torch.clamp(x0, -clip, clip)
                if replace:
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
