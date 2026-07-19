"""Real CUDA smoke test against the exact official ImputeECG checkout."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from ecgcert.estimators import ReconstructorConfig, TrainManifest
from ecgcert.estimators.api import sha256_file
from ecgcert.estimators.official import (
    IMPUTE_ECG,
    ImputeECGReconstructor,
    validate_pinned_checkout,
)


LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")


def _official_cuda_source_or_skip() -> Path:
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
    raw_source = os.environ.get("ECGCERT_IMPUTEECG_SOURCE_DIR")
    if not raw_source:
        if required:
            pytest.fail(
                "ECGCERT_REQUIRE_CUDA_TEST=1 but ECGCERT_IMPUTEECG_SOURCE_DIR is unset"
            )
        pytest.skip("exact ImputeECG checkout was not supplied")
    source = Path(raw_source).resolve()
    try:
        validate_pinned_checkout(source, IMPUTE_ECG)
    except Exception as exc:
        if required:
            pytest.fail(f"exact ImputeECG checkout validation failed: {exc}")
        pytest.skip(f"exact ImputeECG checkout validation failed: {exc}")
    return source


def _tiny_training_data(root: Path) -> tuple[np.ndarray, TrainManifest, Path]:
    time = np.linspace(0, 10, 5000, endpoint=False, dtype=np.float32)
    signals = np.empty((4, 12, 5000), dtype=np.float32)
    for record in range(4):
        for lead in range(12):
            signals[record, lead] = (0.12 + 0.015 * lead) * np.sin(
                2 * np.pi * (1.0 + 0.025 * lead) * time + 0.13 * record
            )
    signal_path = root / "training_signals.npy"
    np.save(signal_path, signals)
    official_data = root / "official_data"
    official_data.mkdir()
    ground_truth = np.transpose(signals, (0, 2, 1))
    observed = np.full_like(ground_truth, np.float32(65535.0))
    observed[:, :, :2] = ground_truth[:, :, :2]
    np.save(official_data / "train_data_gt.npy", ground_truth)
    np.save(official_data / "train_data_mask.npy", observed)
    identity = hashlib.sha256(b"imputeecg-real-cuda-tiny-v1").hexdigest()
    manifest = TrainManifest(
        dataset="synthetic-imputeecg-cuda-smoke",
        split="train",
        signals_path=str(signal_path),
        signals_sha256=sha256_file(signal_path),
        split_sha256=identity,
        patient_ids_sha256=identity,
        rate_hz=500,
    )
    return signals, manifest, official_data


def test_exact_imputeecg_cuda_train_load_and_batch_infer_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _official_cuda_source_or_skip()
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    signals, manifest, official_data = _tiny_training_data(tmp_path)
    model_dir = tmp_path / "model"
    model = ImputeECGReconstructor(source).fit(
        manifest,
        ReconstructorConfig(
            observed_leads=("I", "II"),
            seed=27,
            output_dir=str(model_dir),
            device="cuda:0",
            parameters={
                "official_data_path": str(official_data),
                "epochs": 1,
                "batch_size": 2,
                "num_workers": 0,
            },
        ),
    )
    checkpoint = model_dir / "checkpoint-1.pth"
    assert checkpoint.is_file()
    assert model.checkpoint_sha256 == sha256_file(checkpoint)
    assert sorted(path.name for path in model_dir.glob("checkpoint-*.pth")) == [
        checkpoint.name
    ]
    retention = json.loads((model_dir / "checkpoint_retention.v1.json").read_text())
    assert retention["retained"]["sha256"] == model.checkpoint_sha256

    observed_leads = np.zeros((2, len(LEADS)), dtype=bool)
    observed_leads[:, :2] = True
    reconstructed = model.reconstruct_batch(signals[:2], observed_leads)
    assert reconstructed.shape == signals[:2].shape
    assert np.array_equal(reconstructed[:, :2], signals[:2, :2])
    assert np.isfinite(reconstructed[:, 2:]).all()
    assert validate_pinned_checkout(source, IMPUTE_ECG) == IMPUTE_ECG.commit

    import torch

    del model
    torch.cuda.empty_cache()
