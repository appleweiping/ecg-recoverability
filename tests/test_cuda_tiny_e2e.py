"""Actual CUDA contract smoke test for the native arbitrary-mask U-Net."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

from ecgcert.estimators import MaskedUNetReconstructor, ReconstructorConfig, TrainManifest
from ecgcert.estimators.api import sha256_file
from ecgcert.reconstruction import (
    checkpoint_descriptor,
    load_fitted_reconstructor,
    write_bundle_metadata,
)


def _cuda_or_skip():
    required = os.environ.get("ECGCERT_REQUIRE_CUDA_TEST") == "1"
    try:
        import torch
    except ImportError:
        if required:
            pytest.fail("ECGCERT_REQUIRE_CUDA_TEST=1 but PyTorch is not installed")
        pytest.skip("PyTorch is not installed")
    if not torch.cuda.is_available():
        if required:
            pytest.fail("ECGCERT_REQUIRE_CUDA_TEST=1 but CUDA is unavailable")
        pytest.skip("CUDA is unavailable")
    return torch


def _tiny_manifest(tmp_path: Path) -> tuple[np.ndarray, TrainManifest]:
    rng = np.random.default_rng(27)
    latent = rng.normal(size=(2, 3, 16)).astype(np.float32)
    mixing = rng.normal(size=(12, 3)).astype(np.float32)
    signals = np.einsum("lc,nct->nlt", mixing, latent, optimize=True).astype(np.float32)
    signal_path = tmp_path / "tiny_cuda_train.npy"
    np.save(signal_path, signals)
    identity_hash = hashlib.sha256(b"tiny-cuda-patients-v1").hexdigest()
    return signals, TrainManifest(
        dataset="synthetic-cuda-smoke",
        split="train",
        signals_path=str(signal_path),
        signals_sha256=sha256_file(signal_path),
        split_sha256=identity_hash,
        patient_ids_sha256=identity_hash,
        rate_hz=500,
    )


def test_masked_unet_cuda_train_save_load_reconstruct_contract(tmp_path):
    torch = _cuda_or_skip()
    signals, manifest = _tiny_manifest(tmp_path)
    bundle = tmp_path / "cuda-unet-bundle"
    model_dir = bundle / "models" / "seed-27"

    fitted = MaskedUNetReconstructor().fit(
        manifest,
        ReconstructorConfig(
            observed_leads=("I", "II"),
            seed=27,
            output_dir=str(model_dir),
            device="cuda:0",
            parameters={
                "epochs": 1,
                "batch_size": 1,
                "max_records": 2,
                "normalization_records": 2,
                "width": 4,
                "num_workers": 0,
                "deterministic": True,
            },
        ),
    )
    checkpoint = model_dir / "masked_unet.pt"
    assert checkpoint.is_file()
    assert fitted.checkpoint_sha256 == sha256_file(checkpoint)
    assert next(fitted.model.parameters()).is_cuda

    descriptor = checkpoint_descriptor(checkpoint, bundle, seed=27)
    write_bundle_metadata(
        bundle,
        {
            "method": "masked-unet",
            "adapter_class": "ecgcert.estimators.MaskedUNetReconstructor",
            "load_helper": "ecgcert.reconstruction.load_fitted_reconstructor",
            "models": [descriptor],
            "training_config": {"device": "cuda:0", "tiny_smoke": True},
            "tuning_config": {},
        },
    )
    loaded = load_fitted_reconstructor(bundle, "masked-unet", 27, device="cuda:0")
    assert next(loaded.model.parameters()).is_cuda
    assert loaded.checkpoint_sha256 == descriptor["sha256"]

    observed = np.zeros(12, dtype=bool)
    observed[:2] = True
    reconstructed = loaded.reconstruct(signals[0], observed)
    assert reconstructed.shape == signals[0].shape
    assert np.array_equal(reconstructed[observed], signals[0, observed])
    assert np.isfinite(reconstructed[~observed]).all()

    del fitted, loaded
    torch.cuda.empty_cache()
