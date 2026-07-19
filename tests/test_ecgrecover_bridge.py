import argparse
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from ecgcert.estimators.official import ECG_RECOVER
from ecgcert.official_baselines import sha256_file


ROOT = Path(__file__).resolve().parents[1]
BRIDGE_PATH = ROOT / "integrations" / "ecgrecover" / "ecgrecover_bridge.py"
DESCRIPTOR_PATH = ROOT / "config" / "ecgrecover.integration.v3.json"
UPSTREAM_PATH = ROOT / "config" / "ecgrecover.upstream.v1.json"


def _bridge_module():
    spec = importlib.util.spec_from_file_location("ecgrecover_bridge_tested", BRIDGE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_upstream_import_context_never_writes_bytecode(tmp_path):
    bridge = _bridge_module()
    source = tmp_path / "ECGrecover"
    (source / "learn").mkdir(parents=True)
    (source / "tools").mkdir()
    (source / "learn" / "Training.py").write_text(
        "def training(*args, **kwargs):\n    return None\n", encoding="utf-8"
    )
    (source / "tools" / "LoadModel.py").write_text(
        "def load_model(*args, **kwargs):\n    return None\n", encoding="utf-8"
    )
    (source / "tools" / "LossFunction.py").write_text(
        "def loss_function(*args, **kwargs):\n    return None\n", encoding="utf-8"
    )
    module_names = (
        "learn",
        "learn.Training",
        "tools",
        "tools.LoadModel",
        "tools.LossFunction",
    )
    previous_modules = {
        name: sys.modules.pop(name) for name in module_names if name in sys.modules
    }
    previous_flag = sys.dont_write_bytecode
    try:
        with bridge._upstream_imports(source) as imported:
            assert all(callable(value) for value in imported)
            assert sys.dont_write_bytecode is True
    finally:
        for name in module_names:
            sys.modules.pop(name, None)
        sys.modules.update(previous_modules)

    assert sys.dont_write_bytecode is previous_flag
    assert not list(source.rglob("*.pyc"))
    assert not list(source.rglob("__pycache__"))


def test_ecgrecover_public_pin_and_noassertion_license_are_explicit():
    value = json.loads(UPSTREAM_PATH.read_text(encoding="utf-8"))
    assert value["repository"] == ECG_RECOVER.repository
    assert value["commit"] == ECG_RECOVER.commit
    assert value["root_tree_object"] == ECG_RECOVER.root_tree
    assert value["license"]["spdx"] == "NOASSERTION"
    assert value["license"]["repository_license_file"] is None
    assert value["license"]["redistribution_by_this_repository"] is False
    assert value["vendored"] is False


def test_real_integration_descriptor_binds_bridge_and_discloses_adapter():
    value = json.loads(DESCRIPTOR_PATH.read_text(encoding="utf-8"))
    assert value["upstream_commit"] == ECG_RECOVER.commit
    assert value["upstream_root_tree"] == ECG_RECOVER.root_tree
    assert value["license_spdx"] == "NOASSERTION"
    assert value["redistribution"] is False
    assert value["input_lead"] == "I"
    assert value["model_samples"] == 512
    assert value["inference_records_per_process"] == 128
    assert value["inference_micro_batch_size"] == 64
    assert value["bridge_files"] == [
        {"path": "ecgrecover_bridge.py", "sha256": sha256_file(BRIDGE_PATH)}
    ]
    disclosure = "\n".join(value["adaptation_disclosure"]).lower()
    for required in (
        "u-net",
        "loss",
        "scaling",
        "single-input",
        "bit-for-bit",
        "micro-batch",
    ):
        assert required in disclosure


def test_patient_grouped_internal_split_has_no_overlap():
    bridge = _bridge_module()
    patient_ids = [f"patient-{index // 2}" for index in range(100)]
    train, validation = bridge._patient_split(patient_ids)
    train_patients = {patient_ids[index] for index in train}
    validation_patients = {patient_ids[index] for index in validation}
    assert train_patients
    assert validation_patients
    assert train_patients.isdisjoint(validation_patients)
    assert sorted(np.concatenate([train, validation]).tolist()) == list(range(len(patient_ids)))


def test_adapter_restores_observed_lead_exactly_and_outputs_raw_shape():
    bridge = _bridge_module()
    observed = np.zeros((12, 5000), dtype=np.float64)
    observed[0] = np.linspace(-0.8, 1.2, 5000)
    mask = np.zeros_like(observed, dtype=bool)
    mask[0] = True
    prediction = np.linspace(-1.0, 1.0, 12 * 512, dtype=np.float32).reshape(12, 512)
    restored = bridge._restore_prediction(
        prediction,
        length=5000,
        scale=np.linspace(0.5, 1.6, 12, dtype=np.float32),
        observed=observed,
        mask=mask,
    )
    assert restored.shape == observed.shape
    assert np.isfinite(restored).all()
    assert np.array_equal(restored[mask], observed[mask])


def test_cpu_batch_inference_matches_scalar_and_loads_model_once(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    bridge = _bridge_module()

    class TinyOfficialModel(torch.nn.Module):
        def forward(self, value, _device):
            return value[:, 0] * 0.25

    loads = []

    @contextmanager
    def fake_upstream_imports(_source_dir):
        def load_model():
            loads.append("load")
            return TinyOfficialModel()

        yield load_model, None, None

    monkeypatch.setattr(bridge, "_upstream_imports", fake_upstream_imports)
    checkpoint = tmp_path / "Modeltemp.pth"
    torch.save(TinyOfficialModel().state_dict(), checkpoint)
    metadata = {
        "schema_version": bridge.BRIDGE_SCHEMA_VERSION,
        "upstream_commit": bridge.UPSTREAM_COMMIT,
        "checkpoint_sha256": bridge._sha256_file(checkpoint),
        "model_seed": 27,
        "input_lead": "I",
        "lead_order": list(bridge.CANONICAL_LEADS),
        "scale_mv": np.linspace(0.5, 1.6, 12).tolist(),
    }
    (tmp_path / "bridge_metadata.v1.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    time = np.linspace(0.0, 1.0, 96, endpoint=False)
    observed = np.zeros((3, 12, time.size), dtype=np.float64)
    observed[:, 0] = np.stack(
        [np.sin(2 * np.pi * (index + 1) * time) for index in range(3)]
    )
    mask = np.zeros_like(observed, dtype=bool)
    mask[:, 0] = True

    def run_infer(input_path, output_path, *, micro_batch_size):
        bridge.infer(
            argparse.Namespace(
                checkpoint=checkpoint,
                input=input_path,
                output=output_path,
                source_dir=tmp_path,
                device="cpu",
                micro_batch_size=micro_batch_size,
            )
        )
        with np.load(output_path, allow_pickle=False) as payload:
            return np.asarray(payload["reconstruction"])

    scalar_predictions = []
    for index in range(observed.shape[0]):
        input_path = tmp_path / f"scalar-input-{index}.npz"
        output_path = tmp_path / f"scalar-output-{index}.npz"
        np.savez(
            input_path,
            observed_signal=observed[index],
            observed_mask=mask[index],
            lead_order=np.asarray(bridge.CANONICAL_LEADS),
        )
        scalar_predictions.append(
            run_infer(input_path, output_path, micro_batch_size=1)
        )
    loads.clear()
    batch_input = tmp_path / "batch-input.npz"
    np.savez(
        batch_input,
        observed_signal=observed,
        observed_mask=mask,
        lead_order=np.asarray(bridge.CANONICAL_LEADS),
    )
    batch_prediction = run_infer(
        batch_input, tmp_path / "batch-output.npz", micro_batch_size=2
    )
    assert loads == ["load"]
    np.testing.assert_allclose(
        batch_prediction, np.stack(scalar_predictions), rtol=0.0, atol=1e-12
    )
    assert np.array_equal(batch_prediction[mask], observed[mask])


def test_batch_loader_rejects_any_missing_truth(tmp_path):
    bridge = _bridge_module()
    observed = np.zeros((2, 12, 32), dtype=np.float64)
    mask = np.zeros_like(observed, dtype=bool)
    mask[:, 0] = True
    observed[1, 1, 7] = 0.25
    path = tmp_path / "leaky-input.npz"
    np.savez(
        path,
        observed_signal=observed,
        observed_mask=mask,
        lead_order=np.asarray(bridge.CANONICAL_LEADS),
    )
    with pytest.raises(ValueError, match="unobserved samples"):
        bridge._load_inference_input(path)
