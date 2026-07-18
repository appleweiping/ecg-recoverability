import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from ecgcert.estimators.official import ECG_RECOVER, IMPUTE_ECG
from ecgcert.official_baselines import (
    INTEGRATION_SCHEMA_VERSION,
    MISSING_SENTINEL,
    OfficialTrainingRecord,
    load_ecgrecover_integration,
    materialize_official_arrays,
    sha256_file,
    training_configuration,
)
from experiments.reconstruction_benchmark_v3 import PTBXLManifestV3
from scripts import prepare_official_baselines_v3 as preparation


def _write_integration(tmp_path: Path, source: Path) -> Path:
    bridge_root = tmp_path / "bridge"
    bridge_root.mkdir()
    bridge = bridge_root / "ecgrecover_bridge.py"
    bridge.write_text("# audited fixture bridge\n", encoding="utf-8")
    source_file = source / "main.py"
    payload = {
        "schema_version": INTEGRATION_SCHEMA_VERSION,
        "upstream_commit": ECG_RECOVER.commit,
        "input_lead": "I",
        "native_rate_hz": 500,
        "train_command": [
            "python",
            "{bridge_root}/ecgrecover_bridge.py",
            "train",
            "--source-dir",
            "{source_dir}",
            "--data-dir",
            "{data_dir}",
            "--seed",
            "{seed}",
            "--output-dir",
            "{output_dir}",
        ],
        "inference_command": [
            "python",
            "{bridge_root}/ecgrecover_bridge.py",
            "infer",
            "--source-dir",
            "{source_dir}",
            "--input",
            "{input}",
            "--output",
            "{output}",
            "--checkpoint",
            "{checkpoint}",
        ],
        "checkpoint": "{output_dir}/model.pth",
        "upstream_source_files": [
            {"path": "main.py", "sha256": sha256_file(source_file)}
        ],
        "bridge_root": str(bridge_root),
        "bridge_files": [
            {"path": "ecgrecover_bridge.py", "sha256": sha256_file(bridge)}
        ],
    }
    path = tmp_path / "ecgrecover-integration.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_ecgrecover_integration_is_commit_and_file_hash_bound(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "main.py").write_text("# official fixture\n", encoding="utf-8")
    descriptor = _write_integration(tmp_path, source)
    loaded = load_ecgrecover_integration(descriptor, source_dir=source)
    assert loaded.input_lead == "I"
    assert loaded.native_rate_hz == 500
    assert len(loaded.descriptor_sha256) == 64

    (source / "main.py").write_text("# changed after audit\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_ecgrecover_integration(descriptor, source_dir=source)
    (source / "main.py").write_text("# official fixture\n", encoding="utf-8")

    payload = json.loads(descriptor.read_text(encoding="utf-8"))
    payload["upstream_commit"] = "0" * 40
    descriptor.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="pinned upstream"):
        load_ecgrecover_integration(descriptor, source_dir=source)


def test_ecgrecover_inference_descriptor_cannot_receive_training_truth(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "main.py").write_text("# official fixture\n", encoding="utf-8")
    descriptor = _write_integration(tmp_path, source)
    payload = json.loads(descriptor.read_text(encoding="utf-8"))
    payload["inference_command"].extend(["--truth", "{data_dir}/train_data_gt.npy"])
    descriptor.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="must not receive"):
        load_ecgrecover_integration(descriptor, source_dir=source)


def test_official_arrays_use_patient_stable_panel_masks_and_raw_mv(tmp_path):
    first = np.arange(5000 * 12, dtype=np.float32).reshape(5000, 12) / 1000
    second = first + 0.25
    artifacts = materialize_official_arrays(
        [
            OfficialTrainingRecord("1", "patient-a", first, {"status": "included"}),
            OfficialTrainingRecord("2", "patient-a", second, {"status": "included"}),
        ],
        output_dir=tmp_path / "official-data",
        ecgrecover_input_lead="I",
    )
    gt = np.load(artifacts["imputeecg_ground_truth"]["path"], mmap_mode="r")
    masked = np.load(artifacts["imputeecg_masked_observation"]["path"], mmap_mode="r")
    single = np.load(artifacts["ecgrecover_single_lead_input"]["path"], mmap_mode="r")
    assert gt.shape == masked.shape == (2, 5000, 12)
    assert np.array_equal(gt[0], first)
    configuration = training_configuration("patient-a")
    for index, lead in enumerate(
        ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
    ):
        if lead in configuration:
            assert np.array_equal(masked[:, :, index], gt[:, :, index])
        else:
            assert np.all(masked[:, :, index] == MISSING_SENTINEL)
    assert np.array_equal(single[:, :, 0], gt[:, :, 0])


class _FakeDB:
    def __init__(self, root):
        self.root = root

    def signal_with_audit(self, ecg_id, rate):
        assert rate == 500
        from ecgcert.data.audit import SignalAudit

        signal = np.full((5000, 12), ecg_id / 1000, dtype=np.float32)
        return signal, SignalAudit(
            cohort="PTB-XL",
            record_id=str(ecg_id),
            patient_id=f"p{ecg_id}",
            status="included",
            reason=None,
            requested_rate_hz=500,
            source_rate_hz=500,
            n_samples=5000,
            input_leads=(
                "I",
                "II",
                "III",
                "aVR",
                "aVL",
                "aVF",
                "V1",
                "V2",
                "V3",
                "V4",
                "V5",
                "V6",
            ),
            input_units=("mV",) * 12,
            unit_scales_to_mv=(1.0,) * 12,
        )


def test_end_to_end_preparation_writes_configs_but_never_a_checkpoint(tmp_path, monkeypatch):
    root = tmp_path / "ptbxl"
    root.mkdir()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    records = {
        "1": {"patient_id": "p1", "strat_fold": 1},
        "8": {"patient_id": "p8", "strat_fold": 8},
        "9": {"patient_id": "p9", "strat_fold": 9},
        "10": {"patient_id": "p10", "strat_fold": 10},
    }
    contract = PTBXLManifestV3(
        path=manifest_path,
        root=root,
        records=records,
        split={"train": ("1",), "tune": ("8",), "calibration": ("9",), "test": ("10",)},
        manifest_sha256="a" * 64,
        split_sha256="b" * 64,
    )
    upstreams = tmp_path / "upstreams"
    upstreams.mkdir()
    sources = {}
    for spec in (IMPUTE_ECG, ECG_RECOVER):
        source = upstreams / f"{spec.name}-{spec.commit[:12]}"
        source.mkdir()
        (source / "main.py").write_text("# pinned\n", encoding="utf-8")
        sources[spec.name] = source
    impute_source = sources[IMPUTE_ECG.name]
    (impute_source / "train.py").write_text("# train\n", encoding="utf-8")
    (impute_source / "inference.py").write_text("# infer\n", encoding="utf-8")
    (impute_source / "datasets").mkdir()
    (impute_source / "datasets" / "ptbxl.py").write_text("# data\n", encoding="utf-8")
    descriptor = _write_integration(tmp_path, sources[ECG_RECOVER.name])

    monkeypatch.setattr(preparation, "load_ptbxl_manifest", lambda _: contract)
    monkeypatch.setattr(preparation, "_verify_manifest_files", lambda *args, **kwargs: None)
    monkeypatch.setattr(preparation, "_validate_database_identity", lambda *args: None)
    monkeypatch.setattr(preparation, "PTBXL", _FakeDB)
    monkeypatch.setattr(preparation, "validate_pinned_checkout", lambda source, spec: spec.commit)

    def fake_tree(source):
        source = Path(source)
        entries = tuple(
            {"path": path.relative_to(source).as_posix(), "sha256": sha256_file(path)}
            for path in sorted(item for item in source.rglob("*") if item.is_file())
        )
        return hashlib.sha256(repr(entries).encode()).hexdigest(), entries

    monkeypatch.setattr(preparation, "source_tree_sha256", fake_tree)
    output = tmp_path / "official"
    summary = preparation.run(
        argparse.Namespace(
            manifest=manifest_path,
            upstreams=upstreams,
            ecgrecover_integration=descriptor,
            output_dir=output,
            seeds=(0,),
            max_records=None,
            release=False,
        )
    )
    assert summary["status"] == "complete"
    assert summary["checkpoints_created"] is False
    assert not list(output.rglob("*.pth"))
    impute_config = json.loads((output / "imputeecg.config.v3.json").read_text())
    ecgrecover_config = json.loads((output / "ecgrecover.config.v3.json").read_text())
    combined = json.loads((output / "official-reconstruction-config-v3.json").read_text())
    assert impute_config["commit"] == IMPUTE_ECG.commit
    assert ecgrecover_config["commit"] == ECG_RECOVER.commit
    assert "{input}" in "\n".join(ecgrecover_config["official_inference_bridge"])
    assert set(combined["methods"]) == {"imputeecg", "ecgrecover"}
    assert Path(impute_config["official_data_path"]).is_dir()
    assert all(
        str(output / "data") in value["path"]
        for value in summary["arrays"].values()
        if "path" in value
    )


def test_release_preparation_rejects_subsampling(tmp_path):
    arguments = argparse.Namespace(
        max_records=1,
        release=True,
        seeds=(0, 1, 2, 3, 4),
        output_dir=tmp_path / "not-artifacts",
    )
    with pytest.raises(ValueError, match="max-records is forbidden"):
        preparation.validate_arguments(arguments)
