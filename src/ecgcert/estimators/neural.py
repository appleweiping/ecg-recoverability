"""Trained neural reduced-lead reconstructors (PyTorch, CPU-friendly).

Two networks share one architecture but differ in training objective:

* ``train_mse`` -- a distortion-optimal 1-D CNN (MSE loss).  Like OLS but
  nonlinear; being MSE-optimal it regresses Tier III content to the mean, so it
  does *not* fabricate (low hallucination energy).  This is the honest deep
  baseline: it shows the hallucination is a property of the *objective*, not of
  deep learning.
* ``train_adversarial`` -- the same network with an added adversarial (perceptual)
  term.  Chasing realism on the unrecoverable subspace, it invents non-dipolar
  content, so its hallucination energy is high and uncorrelated with truth --
  exactly the perception-distortion tradeoff, now for a genuinely trained model.

The architecture is a small residual 1-D CNN over a fixed-length window; it maps
the observed leads (as channels) to all 12 leads.
"""
from __future__ import annotations

import numpy as np


def _torch():
    import torch
    return torch


class ReconCNN:
    """Small residual 1-D CNN: (|S| channels, T) -> (12 channels, T)."""

    def __init__(self, n_obs: int, width: int = 48, depth: int = 4, seed: int = 0):
        torch = _torch()
        torch.manual_seed(seed)
        nn = torch.nn
        layers = [nn.Conv1d(n_obs, width, 9, padding=4), nn.ReLU()]
        for _ in range(depth):
            layers += [nn.Conv1d(width, width, 9, padding=4), nn.ReLU()]
        layers += [nn.Conv1d(width, 12, 9, padding=4)]
        self.net = nn.Sequential(*layers)
        self.n_obs = n_obs

    def parameters(self):
        return self.net.parameters()

    def __call__(self, x):
        return self.net(x)

    def predict(self, y_S: np.ndarray) -> np.ndarray:
        """``y_S`` is (|S|, T) numpy -> (12, T) numpy."""
        torch = _torch()
        self.net.eval()
        with torch.no_grad():
            x = torch.tensor(np.asarray(y_S, np.float32))[None]      # (1, |S|, T)
            out = self.net(x)[0].numpy()
        return out


class _Disc:
    """Tiny 1-D patch discriminator for the adversarial variant."""

    def __init__(self, width: int = 32, seed: int = 1):
        torch = _torch(); torch.manual_seed(seed); nn = torch.nn
        self.net = nn.Sequential(
            nn.Conv1d(12, width, 9, stride=2, padding=4), nn.LeakyReLU(0.2),
            nn.Conv1d(width, width, 9, stride=2, padding=4), nn.LeakyReLU(0.2),
            nn.Conv1d(width, 1, 9, padding=4),
        )

    def parameters(self):
        return self.net.parameters()

    def __call__(self, x):
        return self.net(x)


def _windows(db, ids, obs_idx, rate, win, max_records, seed=0):
    """Extract fixed-length windows: returns (X_obs, Y_full) tensors (N, C, win)."""
    rng = np.random.default_rng(seed)
    Xo, Yf = [], []
    for eid in list(ids)[:max_records]:
        try:
            sig = db.signal(int(eid), rate=rate)                     # (T, 12)
        except Exception:
            continue
        T = sig.shape[0]
        if T < win:
            continue
        for _ in range(2):                                            # 2 windows/record
            s = rng.integers(0, T - win + 1)
            w = sig[s:s + win].T                                     # (12, win)
            Yf.append(w.astype(np.float32))
            Xo.append(w[obs_idx].astype(np.float32))
    return np.stack(Xo), np.stack(Yf)


def train_mse(db, train_ids, obs_idx, rate=100, win=256, epochs=8, max_records=400,
              lr=1e-3, seed=0):
    torch = _torch()
    Xo, Yf = _windows(db, train_ids, obs_idx, rate, win, max_records, seed)
    model = ReconCNN(len(obs_idx), seed=seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    X = torch.tensor(Xo); Y = torch.tensor(Yf)
    n = X.shape[0]; bs = 64
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            out = model(X[idx])
            loss = ((out - Y[idx]) ** 2).mean()
            loss.backward(); opt.step()
            tot += float(loss) * len(idx)
        print(f"  [mse] epoch {ep+1}/{epochs} loss={tot/n:.5f}", flush=True)
    return model


def train_adversarial(db, train_ids, obs_idx, rate=100, win=256, epochs=8,
                      max_records=400, lr=1e-3, adv_weight=0.05, seed=0):
    """MSE + adversarial (perceptual) reconstructor: chases realism -> fabricates."""
    torch = _torch(); nn = torch.nn
    Xo, Yf = _windows(db, train_ids, obs_idx, rate, win, max_records, seed)
    model = ReconCNN(len(obs_idx), seed=seed)
    disc = _Disc(seed=seed + 1)
    optG = torch.optim.Adam(model.parameters(), lr=lr)
    optD = torch.optim.Adam(disc.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()
    X = torch.tensor(Xo); Y = torch.tensor(Yf)
    n = X.shape[0]; bs = 64
    for ep in range(epochs):
        perm = torch.randperm(n); totg = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            real = Y[idx]; fake = model(X[idx])
            # discriminator
            optD.zero_grad()
            dr = disc(real); df = disc(fake.detach())
            lossD = bce(dr, torch.ones_like(dr)) + bce(df, torch.zeros_like(df))
            lossD.backward(); optD.step()
            # generator: MSE + adversarial realism
            optG.zero_grad()
            df2 = disc(fake)
            lossG = ((fake - real) ** 2).mean() + adv_weight * bce(df2, torch.ones_like(df2))
            lossG.backward(); optG.step()
            totg += float(lossG) * len(idx)
        print(f"  [adv] epoch {ep+1}/{epochs} G={totg/n:.5f}", flush=True)
    return model
