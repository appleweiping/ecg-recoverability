"""Fail-closed preparation contracts for pinned public reconstruction baselines.

This module does not vendor or rewrite either public model.  It materializes the
frozen PTB-XL training arrays expected by ImputeECG and the canonical single-lead
arrays consumed by an audited ECGrecover integration.  The latter repository has
no stable command-line interface, so an explicit, hashed integration descriptor is
required; silently guessing a command or accepting a checkpoint is forbidden.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Iterable, Mapping, Sequence, Sized

import numpy as np

from ecgcert.data.common import CANONICAL_LEADS
from ecgcert.estimators.official import ECG_RECOVER, UpstreamSpec
from ecgcert.physics.dipolar_subspace import INDEPENDENT_LEADS
from ecgcert.protocol import CONFIG_PANEL_SALT, deep_configuration_panel


SCHEMA_VERSION = "official-baseline-preparation-v3"
CONFIG_SCHEMA_VERSION = "official-reconstruction-config-v3"
INTEGRATION_SCHEMA_VERSION = "ecgrecover-integration-v3"
RELEASE_SEEDS = (0, 1, 2, 3, 4)
MISSING_SENTINEL = np.float32(65535.0)
IMPUTE_ECG_PARAMETERS: Mapping[str, Any] = {
    "epochs": 100,
    "batch_size": 128,
    "mask_ratio": 0.20,
    "missing_sentinel": float(MISSING_SENTINEL),
    "num_workers": 8,
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def pinned_source_dir(upstreams: str | Path, spec: UpstreamSpec) -> Path:
    return Path(upstreams).resolve() / f"{spec.name}-{spec.commit[:12]}"


def source_tree_sha256(source_dir: str | Path) -> tuple[str, tuple[dict[str, str], ...]]:
    """Hash every tracked file in a clean checkout, independent of mtimes."""

    source = Path(source_dir).resolve()
    completed = subprocess.run(
        ["git", "-C", str(source), "ls-files", "-z"],
        check=True,
        capture_output=True,
    )
    names = tuple(
        value.decode("utf-8", errors="strict")
        for value in completed.stdout.split(b"\0")
        if value
    )
    if not names:
        raise ValueError(f"pinned checkout has no tracked files: {source}")
    entries = []
    for relative in sorted(names):
        path = (source / relative).resolve()
        try:
            path.relative_to(source)
        except ValueError as exc:
            raise ValueError(f"tracked path escapes checkout: {relative}") from exc
        if not path.is_file():
            raise FileNotFoundError(path)
        entries.append({"path": relative.replace("\\", "/"), "sha256": sha256_file(path)})
    frozen = tuple(entries)
    return canonical_sha256(frozen), frozen


def _argv(value: Any, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty argv array")
    out = tuple(str(token) for token in value)
    if any(not token or "\x00" in token or "\n" in token or "\r" in token for token in out):
        raise ValueError(f"{label} contains an empty or unsafe token")
    if any(token in {"|", "||", "&&", ";", ">", ">>"} for token in out):
        raise ValueError(f"{label} must not use shell composition")
    return out


def _require_markers(values: Sequence[str], markers: Iterable[str], *, label: str) -> None:
    joined = "\n".join(values)
    missing = sorted(marker for marker in markers if "{" + marker + "}" not in joined)
    if missing:
        raise ValueError(f"{label} is missing placeholders: {missing}")


def _restrict_markers(values: Sequence[str], allowed: set[str], *, label: str) -> None:
    present = set(re.findall(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", "\n".join(values)))
    unknown = sorted(present - allowed)
    if unknown:
        raise ValueError(f"{label} contains unsupported placeholders: {unknown}")


def _validate_hashed_files(
    values: Any,
    *,
    base: Path,
    label: str,
) -> tuple[dict[str, str], ...]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"{label} must contain at least one hashed file")
    out = []
    for raw in values:
        if not isinstance(raw, Mapping) or set(raw) != {"path", "sha256"}:
            raise ValueError(f"{label} entries require exactly path and sha256")
        relative = Path(str(raw["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"{label} path must stay below its declared root: {relative}")
        path = (base / relative).resolve()
        try:
            path.relative_to(base)
        except ValueError as exc:
            raise ValueError(f"{label} path escapes its declared root: {relative}") from exc
        expected = str(raw["sha256"])
        if len(expected) != 64 or not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"{label} hash mismatch: {relative.as_posix()}")
        out.append({"path": relative.as_posix(), "sha256": expected})
    return tuple(out)


@dataclass(frozen=True)
class ECGRecoverIntegration:
    input_lead: str
    native_rate_hz: int
    train_command: tuple[str, ...]
    inference_command: tuple[str, ...]
    checkpoint: str
    upstream_source_files: tuple[dict[str, str], ...]
    bridge_root: Path
    bridge_files: tuple[dict[str, str], ...]
    descriptor_sha256: str


@dataclass(frozen=True)
class OfficialTrainingRecord:
    record_id: str
    patient_id: str
    signal: np.ndarray
    source_audit: Mapping[str, Any]


def load_ecgrecover_integration(
    path: str | Path,
    *,
    source_dir: str | Path,
) -> ECGRecoverIntegration:
    """Load and verify the audited bridge descriptor for the pinned checkout."""

    descriptor_path = Path(path).resolve()
    try:
        value = json.loads(descriptor_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read ECGrecover integration descriptor: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ValueError("ECGrecover integration descriptor must be a JSON object")
    if value.get("schema_version") != INTEGRATION_SCHEMA_VERSION:
        raise ValueError(f"ECGrecover integration must use {INTEGRATION_SCHEMA_VERSION}")
    if value.get("upstream_commit") != ECG_RECOVER.commit:
        raise ValueError("ECGrecover integration is not bound to the pinned upstream commit")
    input_lead = str(value.get("input_lead", ""))
    if input_lead not in INDEPENDENT_LEADS:
        raise ValueError("ECGrecover integration requires one canonical independent input lead")
    native_rate_hz = int(value.get("native_rate_hz", 0))
    if native_rate_hz <= 0:
        raise ValueError("ECGrecover integration must disclose a positive native_rate_hz")
    train = _argv(value.get("train_command"), label="ECGrecover train_command")
    inference = _argv(value.get("inference_command"), label="ECGrecover inference_command")
    _require_markers(
        train,
        ("source_dir", "data_dir", "bridge_root", "seed", "output_dir"),
        label="ECGrecover train_command",
    )
    _require_markers(
        inference,
        ("source_dir", "bridge_root", "input", "output", "checkpoint"),
        label="ECGrecover inference_command",
    )
    _restrict_markers(
        train,
        {"source_dir", "data_dir", "bridge_root", "seed", "output_dir"},
        label="ECGrecover train_command",
    )
    joined_inference = "\n".join(inference).lower()
    if "{data_dir}" in joined_inference or "train_data_gt" in joined_inference:
        raise ValueError("ECGrecover inference command must not receive training or missing truth")
    _restrict_markers(
        inference,
        {"source_dir", "bridge_root", "input", "output", "checkpoint"},
        label="ECGrecover inference_command",
    )
    checkpoint = str(value.get("checkpoint", ""))
    if "{output_dir}" not in checkpoint:
        raise ValueError("ECGrecover checkpoint path must be rooted at {output_dir}")
    _restrict_markers(
        [checkpoint],
        {"source_dir", "output_dir", "seed"},
        label="ECGrecover checkpoint",
    )
    source = Path(source_dir).resolve()
    upstream_files = _validate_hashed_files(
        value.get("upstream_source_files"),
        base=source,
        label="ECGrecover upstream_source_files",
    )
    raw_bridge_root = value.get("bridge_root")
    if not isinstance(raw_bridge_root, str) or not raw_bridge_root:
        raise ValueError("ECGrecover integration must declare bridge_root")
    bridge_root = Path(raw_bridge_root)
    if not bridge_root.is_absolute():
        bridge_root = (descriptor_path.parent / bridge_root).resolve()
    else:
        bridge_root = bridge_root.resolve()
    if not bridge_root.is_dir():
        raise FileNotFoundError(bridge_root)
    bridge_files = _validate_hashed_files(
        value.get("bridge_files"),
        base=bridge_root,
        label="ECGrecover bridge_files",
    )
    return ECGRecoverIntegration(
        input_lead=input_lead,
        native_rate_hz=native_rate_hz,
        train_command=train,
        inference_command=inference,
        checkpoint=checkpoint,
        upstream_source_files=upstream_files,
        bridge_root=bridge_root,
        bridge_files=bridge_files,
        descriptor_sha256=sha256_file(descriptor_path),
    )


def replace_markers(values: Sequence[str], replacements: Mapping[str, str]) -> list[str]:
    out = []
    for raw in values:
        token = str(raw)
        for marker, replacement in replacements.items():
            token = token.replace("{" + marker + "}", replacement)
        out.append(token)
    return out


def training_configuration(patient_id: str) -> tuple[str, ...]:
    """Choose one frozen panel mask per patient without consulting outcomes."""

    panel = deep_configuration_panel()
    digest = hashlib.sha256(
        f"{CONFIG_PANEL_SALT}|imputeecg-train-mask|{patient_id}".encode("utf-8")
    ).digest()
    return panel[int.from_bytes(digest[:8], "big") % len(panel)]


def materialize_official_arrays(
    records: Iterable[OfficialTrainingRecord],
    *,
    output_dir: str | Path,
    ecgrecover_input_lead: str,
    n_records: int | None = None,
) -> dict[str, Any]:
    """Write official arrays in canonical raw-mV order and return their lineage."""

    if n_records is None:
        if not isinstance(records, Sized):
            raise ValueError("n_records is required for a streaming record iterable")
        n_records = len(records)
    if n_records < 1:
        raise ValueError("official training split is empty")
    if ecgrecover_input_lead not in INDEPENDENT_LEADS:
        raise ValueError("ECGrecover input lead must be canonical and independent")
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    impute_root = destination / "imputeecg"
    ecgrecover_root = destination / "ecgrecover"
    impute_root.mkdir()
    ecgrecover_root.mkdir()
    shape = (n_records, 5000, len(CANONICAL_LEADS))
    gt_path = impute_root / "train_data_gt.npy"
    mask_path = impute_root / "train_data_mask.npy"
    single_path = ecgrecover_root / "train_input_single_lead.npy"
    ground_truth = np.lib.format.open_memmap(gt_path, mode="w+", dtype=np.float32, shape=shape)
    masked = np.lib.format.open_memmap(mask_path, mode="w+", dtype=np.float32, shape=shape)
    single = np.lib.format.open_memmap(
        single_path,
        mode="w+",
        dtype=np.float32,
        shape=(n_records, 5000, 1),
    )
    input_index = CANONICAL_LEADS.index(ecgrecover_input_lead)
    record_audit = []
    for index, record in enumerate(records):
        if index >= n_records:
            raise ValueError("record iterable contains more items than n_records")
        record_id = str(record.record_id)
        patient_id = str(record.patient_id)
        signal = np.asarray(record.signal, dtype=np.float32)
        if signal.shape != (5000, len(CANONICAL_LEADS)):
            raise ValueError(f"record {record_id} has shape {signal.shape}, expected (5000, 12)")
        if not np.isfinite(signal).all() or np.any(signal == MISSING_SENTINEL):
            raise ValueError(f"record {record_id} is non-finite or collides with missing sentinel")
        configuration = training_configuration(str(patient_id))
        observed_indices = [CANONICAL_LEADS.index(lead) for lead in configuration]
        observed = np.zeros(len(CANONICAL_LEADS), dtype=bool)
        observed[observed_indices] = True
        ground_truth[index] = signal
        masked[index] = np.where(observed[None, :], signal, MISSING_SENTINEL)
        single[index, :, 0] = signal[:, input_index]
        record_audit.append(
            {
                "index": index,
                "record_id": str(record_id),
                "patient_id": str(patient_id),
                "imputeecg_observed_configuration": list(configuration),
                "source_audit": dict(record.source_audit),
            }
        )
    if len(record_audit) != n_records:
        raise ValueError(
            f"record iterable yielded {len(record_audit)} items, expected {n_records}"
        )
    ground_truth.flush()
    masked.flush()
    single.flush()
    del ground_truth, masked, single
    records_path = destination / "training_records.v3.json"
    records_payload = {
        "schema_version": SCHEMA_VERSION,
        "rate_hz": 500,
        "normalization": "raw_mV",
        "lead_order": list(CANONICAL_LEADS),
        "configuration_panel_salt": CONFIG_PANEL_SALT,
        "configuration_assignment": "patient SHA-256 modulo frozen 64-configuration panel",
        "records": record_audit,
    }
    records_path.write_text(
        json.dumps(records_payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    gt_sha256 = sha256_file(gt_path)
    mask_sha256 = sha256_file(mask_path)
    single_sha256 = sha256_file(single_path)
    records_sha256 = sha256_file(records_path)
    dataset_path = ecgrecover_root / "dataset.v3.json"
    ecgrecover_dataset = {
        "schema_version": SCHEMA_VERSION,
        "task": "official-single-input-lead",
        "rate_hz": 500,
        "normalization": "raw_mV",
        "lead_order": list(CANONICAL_LEADS),
        "input_lead": ecgrecover_input_lead,
        "ground_truth": {
            "path": "../imputeecg/train_data_gt.npy",
            "shape": list(shape),
            "dtype": "float32",
            "sha256": gt_sha256,
        },
        "single_lead_input": {
            "path": "train_input_single_lead.npy",
            "shape": [n_records, 5000, 1],
            "dtype": "float32",
            "sha256": single_sha256,
        },
        "record_order": {
            "path": "../training_records.v3.json",
            "sha256": records_sha256,
        },
    }
    dataset_path.write_text(
        json.dumps(ecgrecover_dataset, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    artifacts = {
        "imputeecg_ground_truth": {
            "path": str(gt_path),
            "shape": list(shape),
            "dtype": "float32",
            "sha256": gt_sha256,
        },
        "imputeecg_masked_observation": {
            "path": str(mask_path),
            "shape": list(shape),
            "dtype": "float32",
            "missing_sentinel": float(MISSING_SENTINEL),
            "sha256": mask_sha256,
        },
        "ecgrecover_single_lead_input": {
            "path": str(single_path),
            "shape": [n_records, 5000, 1],
            "dtype": "float32",
            "input_lead": ecgrecover_input_lead,
            "sha256": single_sha256,
        },
        "ecgrecover_dataset": {
            "path": str(dataset_path),
            "sha256": sha256_file(dataset_path),
        },
        "record_order": {
            "path": str(records_path),
            "sha256": records_sha256,
            "record_ids_sha256": canonical_sha256(
                [item["record_id"] for item in record_audit]
            ),
            "patient_ids_sha256": canonical_sha256(
                [item["patient_id"] for item in record_audit]
            ),
        },
    }
    return artifacts


__all__ = [
    "CONFIG_SCHEMA_VERSION",
    "ECGRecoverIntegration",
    "IMPUTE_ECG_PARAMETERS",
    "INTEGRATION_SCHEMA_VERSION",
    "MISSING_SENTINEL",
    "OfficialTrainingRecord",
    "RELEASE_SEEDS",
    "SCHEMA_VERSION",
    "canonical_sha256",
    "load_ecgrecover_integration",
    "materialize_official_arrays",
    "pinned_source_dir",
    "replace_markers",
    "sha256_file",
    "source_tree_sha256",
    "training_configuration",
]
