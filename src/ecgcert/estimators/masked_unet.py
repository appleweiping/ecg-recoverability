"""Native arbitrary-lead-mask U-Net baseline for the locked benchmark panel."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ecgcert.data.common import CANONICAL_LEADS
from ecgcert.estimators.api import (
    Reconstructor,
    ReconstructorConfig,
    TrainManifest,
    load_manifest_signals,
)
from ecgcert.protocol import deep_configuration_panel


def _torch():
    import torch

    return torch


def _build_unet(width: int = 48):
    torch = _torch()
    nn = torch.nn

    class Block(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 7, padding=3),
                nn.GroupNorm(4, out_channels),
                nn.GELU(),
                nn.Conv1d(out_channels, out_channels, 7, padding=3),
                nn.GroupNorm(4, out_channels),
                nn.GELU(),
            )

        def forward(self, value):
            return self.net(value)

    class MaskedUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc1 = Block(24, width)
            self.pool1 = nn.Conv1d(width, width * 2, 4, stride=2, padding=1)
            self.enc2 = Block(width * 2, width * 2)
            self.pool2 = nn.Conv1d(width * 2, width * 4, 4, stride=2, padding=1)
            self.mid = Block(width * 4, width * 4)
            self.up2 = nn.ConvTranspose1d(width * 4, width * 2, 4, stride=2, padding=1)
            self.dec2 = Block(width * 4, width * 2)
            self.up1 = nn.ConvTranspose1d(width * 2, width, 4, stride=2, padding=1)
            self.dec1 = Block(width * 2, width)
            self.out = nn.Conv1d(width, 12, 1)

        def forward(self, value):
            original_length = value.shape[-1]
            pad = (-original_length) % 4
            if pad:
                value = nn.functional.pad(value, (0, pad))
            e1 = self.enc1(value)
            e2 = self.enc2(self.pool1(e1))
            middle = self.mid(self.pool2(e2))
            d2 = self.dec2(torch.cat([self.up2(middle), e2], dim=1))
            d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
            return self.out(d1)[..., :original_length]

    return MaskedUNet()


class MaskedUNetReconstructor(Reconstructor):
    """One model trained across the predeclared 64 whole-lead masks."""

    method_id = "masked_unet"
    preferred_batch_size = 16

    def fit(self, train_manifest: TrainManifest, config: ReconstructorConfig):
        torch = _torch()
        config.validate()
        signals = load_manifest_signals(train_manifest)
        seed = int(config.seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        previous_determinism = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(bool(config.parameters.get("deterministic", True)))
        device = torch.device(config.device)

        max_records = min(int(config.parameters.get("max_records", signals.shape[0])), signals.shape[0])
        normalization_records = min(int(config.parameters.get("normalization_records", 2048)), max_records)
        subset = np.asarray(signals[:normalization_records], dtype=np.float32)
        self.scale = np.percentile(np.abs(subset), 95, axis=(0, 2)).astype(np.float32)
        self.scale = np.clip(self.scale, 0.05, None)

        width = int(config.parameters.get("width", 48))
        self.model = _build_unet(width=width).to(device)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(config.parameters.get("learning_rate", 1e-3)),
            weight_decay=float(config.parameters.get("weight_decay", 1e-4)),
        )
        batch_size = int(config.parameters.get("batch_size", 16))
        epochs = int(config.parameters.get("epochs", 60))
        generator = torch.Generator().manual_seed(seed)
        panel = deep_configuration_panel()
        lead_index = {lead: index for index, lead in enumerate(CANONICAL_LEADS)}

        class Dataset(torch.utils.data.Dataset):
            def __len__(self):
                return max_records

            def __getitem__(self, index):
                return torch.as_tensor(np.array(signals[index], dtype=np.float32, copy=True))

        loader = torch.utils.data.DataLoader(
            Dataset(),
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            num_workers=int(config.parameters.get("num_workers", 0)),
            drop_last=False,
        )
        scale = torch.as_tensor(self.scale, dtype=torch.float32, device=device)[None, :, None]
        mask_rng = np.random.default_rng(seed)
        self.model.train()
        try:
            for _epoch in range(epochs):
                for target in loader:
                    target = target.to(device=device, dtype=torch.float32) / scale
                    batch_masks = np.zeros((target.shape[0], 12), dtype=np.float32)
                    for row in range(target.shape[0]):
                        observed = panel[int(mask_rng.integers(0, len(panel)))]
                        batch_masks[row, [lead_index[lead] for lead in observed]] = 1.0
                    observed_mask = torch.as_tensor(batch_masks, device=device)[:, :, None]
                    observed_mask = observed_mask.expand(-1, -1, target.shape[-1])
                    network_input = torch.cat([target * observed_mask, observed_mask], dim=1)
                    prediction = self.model(network_input)
                    missing = 1.0 - observed_mask
                    loss = (((prediction - target) ** 2) * missing).sum() / missing.sum().clamp_min(1.0)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
        finally:
            torch.use_deterministic_algorithms(previous_determinism)

        output = Path(config.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        self._checkpoint_path = output / "masked_unet.pt"
        torch.save(
            {
                "model": self.model.state_dict(),
                "scale": self.scale,
                "seed": seed,
                "width": width,
                "train_manifest_sha256": train_manifest.signals_sha256,
                "training_record_ids_sha256": train_manifest.record_ids_sha256,
                "training_patient_ids_sha256": train_manifest.patient_ids_sha256,
                "training_inclusion_sha256": train_manifest.training_inclusion_sha256,
                "panel_size": len(panel),
            },
            self._checkpoint_path,
        )
        self.device = device
        self._fitted = True
        return self

    def _predict_missing(self, signal: np.ndarray, observed_mask: np.ndarray) -> np.ndarray:
        return self._predict_missing_batch(
            signal[None, ...], observed_mask[None, ...]
        )[0]

    def _predict_missing_batch(
        self, signals: np.ndarray, observed_masks: np.ndarray
    ) -> np.ndarray:
        torch = _torch()
        self.model.eval()
        with torch.no_grad():
            value = torch.as_tensor(
                signals / self.scale[None, :, None],
                dtype=torch.float32,
                device=self.device,
            )
            mask = torch.as_tensor(
                observed_masks, dtype=torch.float32, device=self.device
            )
            prediction = self.model(torch.cat([value * mask, mask], dim=1))
        return prediction.detach().cpu().numpy() * self.scale[None, :, None]
