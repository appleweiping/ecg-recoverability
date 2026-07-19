import hashlib
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from ecgcert.estimators import (
    IMPUTE_ECG,
    LowRankConditionalMeanReconstructor,
    MaskedUNetReconstructor,
    ReconstructorConfig,
    RidgeLeadReconstructor,
    TrainManifest,
)
from ecgcert.estimators.api import sha256_file
from ecgcert.estimators.official import ImputeECGReconstructor, _source_import_path


def _manifest(tmp_path: Path, *, n=8, length=32):
    rng = np.random.default_rng(4)
    latent = rng.normal(size=(n, 3, length)).astype(np.float32)
    mixing = rng.normal(size=(12, 3)).astype(np.float32)
    signals = np.einsum("lc,nct->nlt", mixing, latent)
    path = tmp_path / "train.npy"
    np.save(path, signals)
    digest = hashlib.sha256(b"ids").hexdigest()
    return signals, TrainManifest(
        dataset="synthetic",
        split="train",
        signals_path=str(path),
        signals_sha256=sha256_file(path),
        split_sha256=digest,
        patient_ids_sha256=digest,
        rate_hz=500,
    )


@pytest.mark.parametrize("factory", [RidgeLeadReconstructor, LowRankConditionalMeanReconstructor])
def test_linear_reconstructors_preserve_observed_samples(factory, tmp_path):
    signals, manifest = _manifest(tmp_path)
    config = ReconstructorConfig(
        observed_leads=("I", "II", "V2"),
        seed=0,
        output_dir=str(tmp_path / factory.__name__),
        parameters={"rank": 3, "ridge_lambda": 1e-3, "noise_variance": 1e-5},
    )
    model = factory().fit(manifest, config)
    observed = np.zeros(12, dtype=bool)
    observed[[0, 1, 7]] = True
    result = model.reconstruct(signals[0], observed)
    assert result.shape == signals[0].shape
    assert np.array_equal(result[observed], signals[0, observed])
    assert len(model.checkpoint_sha256) == 64


def test_manifest_hash_mismatch_fails_closed(tmp_path):
    _, manifest = _manifest(tmp_path)
    broken = TrainManifest(**{**manifest.__dict__, "signals_sha256": "0" * 64})
    with pytest.raises(ValueError, match="hash mismatch"):
        broken.validate()


def test_official_source_import_never_writes_bytecode(tmp_path):
    source = tmp_path / "official-source"
    source.mkdir()
    (source / "upstream_fixture.py").write_text("VALUE = 7\n", encoding="utf-8")
    previous = sys.dont_write_bytecode
    try:
        with _source_import_path(source):
            module = importlib.import_module("upstream_fixture")
            assert module.VALUE == 7
    finally:
        sys.modules.pop("upstream_fixture", None)

    assert sys.dont_write_bytecode is previous
    assert not (source / "__pycache__").exists()


def test_masked_unet_tiny_train_save_load_contract(tmp_path):
    pytest.importorskip("torch")
    signals, manifest = _manifest(tmp_path, n=4, length=32)
    config = ReconstructorConfig(
        observed_leads=("I", "II"),
        seed=7,
        output_dir=str(tmp_path / "unet"),
        parameters={
            "epochs": 1,
            "batch_size": 2,
            "max_records": 4,
            "normalization_records": 4,
            "width": 8,
            "num_workers": 0,
        },
    )
    model = MaskedUNetReconstructor().fit(manifest, config)
    observed = np.zeros(12, dtype=bool)
    observed[:2] = True
    result = model.reconstruct(signals[0], observed)
    assert np.array_equal(result[:2], signals[0, :2])
    batch = model.reconstruct_batch(signals[:3], np.repeat(observed[None], 3, axis=0))
    scalar = np.stack([model.reconstruct(signal, observed) for signal in signals[:3]])
    np.testing.assert_allclose(batch, scalar, rtol=1e-4, atol=1e-5)
    assert np.array_equal(batch[:, observed], signals[:3, observed])
    assert len(model.checkpoint_sha256) == 64


def test_imputeecg_pin_and_command_are_explicit(tmp_path):
    _, manifest = _manifest(tmp_path, length=5000)
    official_data = tmp_path / "official"
    official_data.mkdir()
    np.save(official_data / "train_data_gt.npy", np.zeros((1, 5000, 12), np.float32))
    np.save(official_data / "train_data_mask.npy", np.zeros((1, 5000, 12), np.float32))
    adapter = ImputeECGReconstructor(tmp_path / "source")
    config = ReconstructorConfig(
        observed_leads=("I",),
        seed=3,
        output_dir=str(tmp_path / "out"),
        device="cuda:0",
        parameters={"official_data_path": str(official_data), "epochs": 5, "num_workers": 8},
    )
    command = adapter.build_train_command(manifest, config)
    assert IMPUTE_ECG.commit == "70accf2f1600066392b14a5f50dbc131a6f13943"
    assert "--seed" in command and command[command.index("--seed") + 1] == "3"
    assert "--epochs" in command and command[command.index("--epochs") + 1] == "5"


@pytest.mark.parametrize("configured_epochs", [90, 100])
def test_imputeecg_selects_exact_configured_epoch_checkpoint(
    tmp_path, monkeypatch, configured_epochs
):
    _, manifest = _manifest(tmp_path, length=5000)
    output = tmp_path / "out"
    output.mkdir()
    (output / "checkpoint-90.pth").write_bytes(b"epoch-90")
    (output / "checkpoint-100.pth").write_bytes(b"epoch-100")
    adapter = ImputeECGReconstructor(tmp_path / "source")
    monkeypatch.setattr(
        "ecgcert.estimators.official.validate_pinned_checkout",
        lambda *_args, **_kwargs: IMPUTE_ECG.commit,
    )
    monkeypatch.setattr(adapter, "build_train_command", lambda *_args: ["fake-train"])
    monkeypatch.setattr("ecgcert.estimators.official.subprocess.run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(adapter, "load", lambda _device: adapter)
    adapter.fit(
        manifest,
        ReconstructorConfig(
            observed_leads=("I",),
            seed=0,
            output_dir=str(output),
            parameters={"epochs": configured_epochs},
        ),
    )
    assert adapter._checkpoint_path == output / f"checkpoint-{configured_epochs}.pth"
    assert sorted(path.name for path in output.glob("checkpoint-*.pth")) == [
        f"checkpoint-{configured_epochs}.pth"
    ]
    retention = json.loads((output / "checkpoint_retention.v1.json").read_text())
    assert retention["retained"]["name"] == f"checkpoint-{configured_epochs}.pth"
    assert retention["retained"]["sha256"] == sha256_file(adapter._checkpoint_path)
    assert retention["removed_total_bytes"] > 0


def test_imputeecg_native_batch_matches_scalar_and_calls_upstream_once():
    torch = pytest.importorskip("torch")
    adapter = ImputeECGReconstructor(Path("."))
    adapter.model = object()
    adapter.device = torch.device("cpu")
    adapter._fitted = True
    calls = []

    def fake_impute(_model, observed, _device, *, sentinel):
        calls.append(tuple(observed.shape))
        value = torch.as_tensor(observed.copy())
        value[value == sentinel] = -0.25
        return value

    adapter._impute = fake_impute
    rng = np.random.default_rng(19)
    signals = rng.normal(size=(3, 12, 5000)).astype(np.float32)
    masks = np.zeros((3, 12), dtype=bool)
    masks[:, :2] = True
    batch = adapter.reconstruct_batch(signals, masks)
    assert calls == [(3, 12, 5000)]
    calls.clear()
    scalar = np.stack(
        [adapter.reconstruct(signal, mask) for signal, mask in zip(signals, masks, strict=True)]
    )
    np.testing.assert_array_equal(batch, scalar)
    assert calls == [(1, 12, 5000)] * 3
    assert np.array_equal(batch[:, :2], signals[:, :2])
