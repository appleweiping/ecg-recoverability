"""Real CUDA smoke test against the exact official ECGrecover checkout."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from ecgcert.estimators.official import ECG_RECOVER, validate_pinned_checkout


ROOT = Path(__file__).resolve().parents[1]
BRIDGE = ROOT / "integrations" / "ecgrecover" / "ecgrecover_bridge.py"
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
    raw_source = os.environ.get("ECGCERT_ECGRECOVER_SOURCE_DIR")
    if not raw_source:
        if required:
            pytest.fail(
                "ECGCERT_REQUIRE_CUDA_TEST=1 but ECGCERT_ECGRECOVER_SOURCE_DIR is unset"
            )
        pytest.skip("exact ECGrecover checkout was not supplied")
    source = Path(raw_source).resolve()
    try:
        validate_pinned_checkout(source, ECG_RECOVER)
    except Exception as exc:
        if required:
            pytest.fail(f"exact ECGrecover checkout validation failed: {exc}")
        pytest.skip(f"exact ECGrecover checkout validation failed: {exc}")
    return source


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tiny_dataset(root: Path) -> tuple[Path, Path, np.ndarray, np.ndarray]:
    data = root / "data"
    ecgrecover = data / "ecgrecover"
    imputeecg = data / "imputeecg"
    ecgrecover.mkdir(parents=True)
    imputeecg.mkdir(parents=True)
    time = np.linspace(0, 10, 5000, endpoint=False, dtype=np.float32)
    truth = np.empty((4, 5000, 12), dtype=np.float32)
    for record in range(4):
        for lead in range(12):
            truth[record, :, lead] = (0.15 + 0.02 * lead) * np.sin(
                2 * np.pi * (1.0 + 0.03 * lead) * time + 0.1 * record
            )
    truth_path = imputeecg / "train_data_gt.npy"
    np.save(truth_path, truth)
    records_path = data / "training_records.v3.json"
    records_path.write_text(
        json.dumps(
            {
                "records": [
                    {"record_id": str(index), "patient_id": f"cuda-tiny-{index}"}
                    for index in range(4)
                ]
            }
        ),
        encoding="utf-8",
    )
    (ecgrecover / "dataset.v3.json").write_text(
        json.dumps(
            {
                "task": "official-single-input-lead",
                "lead_order": list(LEADS),
                "input_lead": "I",
                "ground_truth": {
                    "path": "../imputeecg/train_data_gt.npy",
                    "sha256": _sha256(truth_path),
                },
                "record_order": {
                    "path": "../training_records.v3.json",
                    "sha256": _sha256(records_path),
                },
            }
        ),
        encoding="utf-8",
    )
    observed = np.zeros((4, 12, 5000), dtype=np.float64)
    observed[:, 0] = truth[:, :, 0]
    mask = np.zeros_like(observed, dtype=bool)
    mask[:, 0] = True
    input_path = root / "input.npz"
    np.savez_compressed(
        input_path,
        observed_signal=observed,
        observed_mask=mask,
        lead_order=np.asarray(LEADS),
    )
    return ecgrecover, input_path, observed, mask


def test_exact_ecgrecover_cuda_train_and_batch_infer_contract(tmp_path):
    source = _official_cuda_source_or_skip()
    subprocess_environment = os.environ.copy()
    subprocess_environment["PYTHONDONTWRITEBYTECODE"] = "1"
    data_dir, input_path, observed, mask = _tiny_dataset(tmp_path)
    model_dir = tmp_path / "model"
    subprocess.run(
        [
            sys.executable,
            str(BRIDGE),
            "train",
            "--source-dir",
            str(source),
            "--data-dir",
            str(data_dir),
            "--seed",
            "27",
            "--output-dir",
            str(model_dir),
            "--input-lead",
            "I",
            "--device",
            "cuda:0",
            "--epochs",
            "1",
            "--batch-size",
            "4",
        ],
        check=True,
        cwd=ROOT,
        env=subprocess_environment,
    )
    checkpoint = model_dir / "Modeltemp.pth"
    output = tmp_path / "output.npz"
    subprocess.run(
        [
            sys.executable,
            str(BRIDGE),
            "infer",
            "--source-dir",
            str(source),
            "--input",
            str(input_path),
            "--output",
            str(output),
            "--checkpoint",
            str(checkpoint),
            "--device",
            "cuda:0",
            "--micro-batch-size",
            "2",
        ],
        check=True,
        cwd=ROOT,
        env=subprocess_environment,
    )
    # The mandatory CUDA smoke is allowed to write only to its isolated model
    # and output directories, never to the persistent frozen source checkout.
    assert validate_pinned_checkout(source, ECG_RECOVER) == ECG_RECOVER.commit
    metadata = json.loads((model_dir / "bridge_metadata.v1.json").read_text())
    assert metadata["upstream_commit"] == ECG_RECOVER.commit
    assert metadata["checkpoint_sha256"] == _sha256(checkpoint)
    assert metadata["architecture"] == "unmodified tools.LoadModel.load_model"
    assert metadata["loss"].startswith("unmodified tools.LossFunction.loss_function")
    assert metadata["training_loop"] == "unmodified learn.Training.training"
    with np.load(output, allow_pickle=False) as payload:
        reconstructed = payload["reconstruction"]
    assert reconstructed.shape == observed.shape
    assert np.isfinite(reconstructed).all()
    assert np.array_equal(reconstructed[mask], observed[mask])
