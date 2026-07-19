"""One authenticated folds-1--7 training cohort shared by every reconstructor.

The inclusion artifact is intentionally small: it freezes strict delineation
eligibility, ordered record/patient membership, the patient-level audit, and the
folds-1--7 simple predictors.  Signal arrays are materialised by each isolated
training job from the same authenticated source records.  Materialisation never
performs a second eligibility decision: an error for an included record is a hard
failure rather than a method-specific exclusion.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ecgcert import lineage
from ecgcert.data.audit import AuditTrail, SignalAudit
from ecgcert.data.common import CANONICAL_LEADS
from ecgcert.estimators.api import TrainManifest, sha256_file
from ecgcert.reconstruction import (
    EvaluationRecord,
    TRAINING_PREDICTORS_FILENAME,
    TrainingPredictorAccumulator,
    training_predictor_lookup,
)


SCHEMA_VERSION = "ptbxl-training-inclusion-v1"
AUDIT_SCHEMA_VERSION = "ptbxl-training-inclusion-audit-v1"
MANIFEST_FILENAME = "training_inclusion.v1.json"
AUDIT_FILENAME = "training_inclusion_audit.v1.json"
LOCKED_RATE_HZ = 500
LOCKED_SEGMENTS = ("QRS", "ST", "T")
LOCKED_DELINEATOR = "dwt"


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _ordered_ids_sha256(values: Sequence[str]) -> str:
    return lineage.canonical_sha256([str(value) for value in values])


def _patient_ids_sha256(values: Sequence[str]) -> str:
    # This is deliberately an ordered, per-record patient sequence.  Hashing a
    # sorted set would authenticate membership while losing the row alignment
    # between the signal array, record IDs, and repeated patient IDs.
    return lineage.canonical_sha256([str(value) for value in values])


def _signal_content_sha256(signal: np.ndarray) -> str:
    """Hash one canonical float32 signal including its shape and byte order."""

    value = np.ascontiguousarray(np.asarray(signal, dtype="<f4"))
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"dtype": "<f4", "shape": list(value.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _predictor_content(frame: pd.DataFrame) -> list[dict[str, Any]]:
    required = (
        "schema_version",
        "source_partition",
        "segment",
        "configuration",
        "target",
        "target_rms",
        "max_target_observed_correlation",
        "n_training_samples",
    )
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"training predictors lack columns: {sorted(missing)}")
    training_predictor_lookup(frame)
    ordered = frame.sort_values(["segment", "configuration", "target"], kind="mergesort")
    return [
        {
            "schema_version": str(row.schema_version),
            "source_partition": str(row.source_partition),
            "segment": str(row.segment),
            "configuration": str(row.configuration),
            "target": str(row.target),
            "target_rms": float(row.target_rms),
            "max_target_observed_correlation": float(
                row.max_target_observed_correlation
            ),
            "n_training_samples": int(row.n_training_samples),
        }
        for row in ordered.itertuples(index=False)
    ]


def _tuple_audit_fields(value: Mapping[str, Any]) -> dict[str, Any]:
    copied = dict(value)
    for name in (
        "input_leads",
        "input_units",
        "canonical_leads",
        "source_channel_indices",
        "unit_scales_to_mv",
    ):
        copied[name] = tuple(copied.get(name, ()))
    copied["segment_counts"] = dict(copied.get("segment_counts", {}))
    return copied


def _signal_audit_without_segments(value: SignalAudit | Mapping[str, Any]) -> dict[str, Any]:
    raw = asdict(value) if isinstance(value, SignalAudit) else dict(value)
    # JSON round-tripping changes tuples into lists.  Restore the audited tuple
    # fields before comparing a later database read with the frozen record.
    raw = _tuple_audit_fields(raw)
    raw.pop("segment_counts", None)
    return raw


def _safe_artifact(root: Path, descriptor: Any, *, expected_name: str) -> Path:
    if not isinstance(descriptor, Mapping) or not {"path", "sha256"} <= set(descriptor):
        raise ValueError(f"training inclusion lacks authenticated {expected_name}")
    relative = Path(str(descriptor["path"]))
    if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != expected_name:
        raise ValueError(f"training inclusion artifact path is invalid: {relative}")
    path = root / relative
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(path)
    if sha256_file(path) != descriptor["sha256"]:
        raise ValueError(f"training inclusion {expected_name} hash mismatch")
    return path


@dataclass(frozen=True)
class TrainingInclusion:
    path: Path
    inclusion_sha256: str
    source_manifest_sha256: str
    split_sha256: str
    rate_hz: int
    segments: tuple[str, ...]
    delineator: str
    configuration_panel_sha256: str
    requested_record_ids: tuple[str, ...]
    included_record_ids: tuple[str, ...]
    included_patient_ids: tuple[str, ...]
    record_ids_sha256: str
    patient_ids_sha256: str
    signal_sha256_by_record: Mapping[str, str]
    audit: Mapping[str, Any]
    predictors_path: Path
    predictors_sha256: str
    predictor_content_sha256: str

    @property
    def root(self) -> Path:
        return self.path.parent

    def iter_validated_signals(
        self,
        db: Any,
        records: Mapping[str, Mapping[str, Any]],
    ):
        """Yield every frozen member in order, failing on any source drift."""

        audit_by_id = {
            str(value["record_id"]): value
            for value in self.audit["records"]
            if value["status"] == "included"
        }
        common_shape: tuple[int, int] | None = None
        for record_id in self.included_record_ids:
            if record_id not in records:
                raise ValueError(f"included record {record_id} is absent from source manifest")
            expected_patient = str(records[record_id]["patient_id"])
            signal_value, source_audit = db.signal_with_audit(
                int(record_id), rate=self.rate_hz
            )
            signal = np.asarray(signal_value, dtype=np.float32)
            if signal.ndim != 2 or signal.shape[1] != len(CANONICAL_LEADS):
                raise ValueError(
                    f"included record {record_id} has invalid signal shape {signal.shape}"
                )
            if not np.isfinite(signal).all():
                raise ValueError(f"included record {record_id} contains non-finite samples")
            if str(source_audit.patient_id) != expected_patient:
                raise ValueError(f"included record {record_id} changed patient identity")
            frozen = audit_by_id.get(record_id)
            if frozen is None or _signal_audit_without_segments(
                source_audit
            ) != _signal_audit_without_segments(frozen):
                raise ValueError(f"included record {record_id} source audit changed")
            if _signal_content_sha256(signal) != self.signal_sha256_by_record.get(record_id):
                raise ValueError(f"included record {record_id} signal content changed")
            if common_shape is None:
                common_shape = signal.shape
            if signal.shape != common_shape:
                raise ValueError(
                    f"included record {record_id} shape changed from {common_shape} "
                    f"to {signal.shape}"
                )
            yield record_id, expected_patient, signal, frozen

    def materialize_signals(
        self,
        db: Any,
        records: Mapping[str, Mapping[str, Any]],
        destination: str | Path,
    ) -> TrainManifest:
        """Materialise exactly the included records; never re-decide eligibility."""

        path = Path(destination).resolve()
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        array = None
        try:
            for index, (_record_id, _patient_id, signal, _audit) in enumerate(
                self.iter_validated_signals(db, records)
            ):
                if array is None:
                    array = np.lib.format.open_memmap(
                        path,
                        mode="w+",
                        dtype=np.float32,
                        shape=(
                            len(self.included_record_ids),
                            len(CANONICAL_LEADS),
                            signal.shape[0],
                        ),
                    )
                assert array is not None
                array[index] = signal.T
            if array is None:
                raise ValueError("training inclusion contains no included records")
            array.flush()
            del array
            array = None
        except BaseException:
            if array is not None:
                del array
            if path.exists():
                path.unlink()
            raise
        manifest = TrainManifest(
            dataset="PTB-XL",
            split="folds1-7/train",
            signals_path=str(path),
            signals_sha256=sha256_file(path),
            split_sha256=self.split_sha256,
            patient_ids_sha256=self.patient_ids_sha256,
            rate_hz=self.rate_hz,
            normalization="raw_mV",
            record_ids_sha256=self.record_ids_sha256,
            training_inclusion_sha256=self.inclusion_sha256,
        )
        manifest.validate()
        return manifest

    def copy_predictors(self, destination: str | Path) -> Path:
        path = Path(destination).resolve()
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.predictors_path, path)
        if sha256_file(path) != self.predictors_sha256:
            path.unlink(missing_ok=True)
            raise ValueError("copied training predictor hash mismatch")
        return path


def build_training_inclusion(
    *,
    db: Any,
    records: Mapping[str, Mapping[str, Any]],
    record_ids: Sequence[str],
    source_manifest_file_sha256: str,
    source_manifest_sha256: str,
    split_sha256: str,
    rate_hz: int,
    segments: Sequence[str],
    delineator: str,
    configurations: Sequence[Sequence[str]],
    configuration_panel_sha256: str,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Create the sole strict-eligibility decision for reconstruction training."""

    if (
        isinstance(rate_hz, bool)
        or int(rate_hz) != LOCKED_RATE_HZ
        or tuple(segments) != LOCKED_SEGMENTS
        or delineator != LOCKED_DELINEATOR
    ):
        raise ValueError(
            "training inclusion is locked to 500 Hz strict DWT QRS/ST/T"
        )
    requested = tuple(str(value) for value in record_ids)
    if not requested or len(requested) != len(set(requested)):
        raise ValueError("training inclusion requires unique requested records")
    for record_id in requested:
        record = records.get(record_id)
        raw_fold = record.get("strat_fold") if isinstance(record, Mapping) else None
        try:
            fold = int(raw_fold)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"requested record {record_id} lacks a valid strat_fold") from exc
        if isinstance(raw_fold, bool) or fold not in range(1, 8):
            raise ValueError(
                f"requested record {record_id} is fold {fold}, outside folds1-7/train"
            )
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    trail = AuditTrail()
    accumulator = TrainingPredictorAccumulator(tuple(segments))
    included_signal_hashes: dict[str, str] = {}
    common_shape: tuple[int, int] | None = None
    for record_id in requested:
        if record_id not in records:
            raise ValueError(f"requested record {record_id} is absent from source manifest")
        patient_id = str(records[record_id]["patient_id"])
        base_audit = None
        try:
            signal_value, base_audit = db.signal_with_audit(int(record_id), rate=rate_hz)
            signal = np.asarray(signal_value, dtype=np.float32)
            if signal.ndim != 2 or signal.shape[1] != len(CANONICAL_LEADS):
                raise ValueError(f"unexpected PTB-XL signal shape {signal.shape}")
            if not np.isfinite(signal).all():
                raise ValueError("signal contains non-finite samples")
            if str(base_audit.patient_id) != patient_id:
                raise ValueError("source audit patient does not match manifest")
            if common_shape is not None and signal.shape != common_shape:
                raise ValueError(
                    f"signal shape {signal.shape} differs from frozen shape {common_shape}"
                )
            indices = db.segment_indices(
                signal, fs=rate_hz, method=delineator, strict=True
            )
            selected = {
                segment: np.asarray(indices.get(segment, ()), dtype=np.int64)
                for segment in segments
            }
            counts = {segment: int(value.size) for segment, value in selected.items()}
            missing_segments = [segment for segment, count in counts.items() if count < 1]
            if missing_segments:
                raise ValueError(
                    "strict delineation lacks requested windows: "
                    + ",".join(missing_segments)
                )
            accumulator.update(
                EvaluationRecord(
                    patient_id=patient_id,
                    record_id=record_id,
                    signal=signal.T,
                    segment_indices=selected,
                )
            )
            if common_shape is None:
                common_shape = signal.shape
            included_signal_hashes[record_id] = _signal_content_sha256(signal)
            trail.append(SignalAudit(**{**asdict(base_audit), "segment_counts": counts}))
        except Exception as exc:
            values: dict[str, Any] = {
                "cohort": "PTB-XL",
                "record_id": record_id,
                "patient_id": patient_id,
                "requested_rate_hz": rate_hz,
            }
            if base_audit is not None:
                for name in (
                    "source_rate_hz",
                    "n_samples",
                    "input_leads",
                    "input_units",
                    "canonical_leads",
                    "source_channel_indices",
                    "unit_scales_to_mv",
                    "output_unit",
                ):
                    values[name] = getattr(base_audit, name)
            trail.append(
                SignalAudit(
                    **values,
                    status="excluded",
                    reason=f"{type(exc).__name__}: {exc}",
                )
            )
    included_records = tuple(
        value.record_id for value in trail.records if value.status == "included"
    )
    included_patients = tuple(
        str(records[record_id]["patient_id"]) for record_id in included_records
    )
    if not included_records:
        raise RuntimeError("no PTB-XL folds1-7 records passed shared training inclusion")
    predictors = accumulator.finalize(configurations)
    predictor_content = _predictor_content(predictors)
    predictor_path = output / TRAINING_PREDICTORS_FILENAME
    predictors.to_parquet(predictor_path, index=False, compression="zstd")
    audit = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "source_manifest_sha256": source_manifest_sha256,
        "split_sha256": split_sha256,
        "train_role": "folds1-7/train",
        "rate_hz": int(rate_hz),
        "segments": list(segments),
        "delineator": delineator,
        "summary": trail.summary_without_hash(),
        "audit_sha256": trail.sha256(),
        "records": [asdict(value) for value in trail.records],
        "included_signals": [
            {
                "record_id": record_id,
                "patient_id": str(records[record_id]["patient_id"]),
                "sha256": included_signal_hashes[record_id],
            }
            for record_id in included_records
        ],
        "included_signals_sha256": lineage.canonical_sha256(
            [included_signal_hashes[record_id] for record_id in included_records]
        ),
    }
    audit_path = output / AUDIT_FILENAME
    _atomic_json(audit_path, audit)
    requested_patients = [str(records[value]["patient_id"]) for value in requested]
    excluded_records = tuple(
        value.record_id for value in trail.records if value.status == "excluded"
    )
    excluded_patients = [str(records[value]["patient_id"]) for value in excluded_records]
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "cohort": "PTB-XL",
        "source_manifest": {
            "file_sha256": source_manifest_file_sha256,
            "manifest_sha256": source_manifest_sha256,
            "split_sha256": split_sha256,
        },
        "train_role": "folds1-7/train",
        "protocol": {
            "rate_hz": int(rate_hz),
            "segments": list(segments),
            "delineator": delineator,
            "strict_delineation": True,
            "normalization": "raw_mV",
            "lead_order": list(CANONICAL_LEADS),
            "configuration_panel_sha256": configuration_panel_sha256,
            "eligibility_owner": "shared_preprocessing_only",
            "consumer_failure_policy": "fail_closed_no_method_specific_exclusion",
        },
        "requested": {
            "n_records": len(requested),
            "n_patients": len(set(requested_patients)),
            "record_ids_sha256": _ordered_ids_sha256(requested),
            "patient_ids_sha256": _patient_ids_sha256(requested_patients),
        },
        "included": {
            "n_records": len(included_records),
            "n_patients": len(set(included_patients)),
            "record_ids_sha256": _ordered_ids_sha256(included_records),
            "patient_ids_sha256": _patient_ids_sha256(included_patients),
            "signals_sha256": lineage.canonical_sha256(
                [included_signal_hashes[record_id] for record_id in included_records]
            ),
        },
        "excluded": {
            "n_records": len(excluded_records),
            "n_patients": len(set(excluded_patients)),
            "record_ids_sha256": _ordered_ids_sha256(excluded_records),
            "patient_ids_sha256": _patient_ids_sha256(excluded_patients),
        },
        "artifacts": {
            "audit": {
                "path": AUDIT_FILENAME,
                "sha256": sha256_file(audit_path),
            },
            "training_predictors": {
                "path": TRAINING_PREDICTORS_FILENAME,
                "sha256": sha256_file(predictor_path),
                "content_sha256": lineage.canonical_sha256(predictor_content),
                "n_rows": len(predictors),
            },
        },
    }
    manifest["inclusion_sha256"] = lineage.canonical_sha256(manifest)
    _atomic_json(output / MANIFEST_FILENAME, manifest)
    return manifest


def load_training_inclusion(
    path: str | Path,
    *,
    source_manifest_path: str | Path,
    source_manifest_sha256: str,
    split_sha256: str,
    expected_record_ids: Sequence[str],
    expected_records: Mapping[str, Mapping[str, Any]],
    rate_hz: int,
    segments: Sequence[str],
    delineator: str,
    configuration_panel_sha256: str,
) -> TrainingInclusion:
    """Authenticate membership against the complete source train role."""

    if (
        isinstance(rate_hz, bool)
        or int(rate_hz) != LOCKED_RATE_HZ
        or tuple(segments) != LOCKED_SEGMENTS
        or delineator != LOCKED_DELINEATOR
    ):
        raise ValueError(
            "training inclusion is locked to 500 Hz strict DWT QRS/ST/T"
        )
    manifest_path = Path(path).resolve()
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read training inclusion manifest: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ValueError("training inclusion manifest must be a JSON object")
    stored_hash = str(value.get("inclusion_sha256", ""))
    unhashed = {key: item for key, item in value.items() if key != "inclusion_sha256"}
    if len(stored_hash) != 64 or lineage.canonical_sha256(unhashed) != stored_hash:
        raise ValueError("training inclusion semantic hash mismatch")
    source = value.get("source_manifest", {})
    protocol = value.get("protocol", {})
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("status") != "complete"
        or value.get("cohort") != "PTB-XL"
        or value.get("train_role") != "folds1-7/train"
        or not isinstance(source, Mapping)
        or source.get("file_sha256") != sha256_file(source_manifest_path)
        or source.get("manifest_sha256") != source_manifest_sha256
        or source.get("split_sha256") != split_sha256
        or not isinstance(protocol, Mapping)
        or protocol.get("rate_hz") != int(rate_hz)
        or tuple(protocol.get("segments", ())) != tuple(segments)
        or protocol.get("delineator") != delineator
        or protocol.get("strict_delineation") is not True
        or protocol.get("normalization") != "raw_mV"
        or tuple(protocol.get("lead_order", ())) != CANONICAL_LEADS
        or protocol.get("configuration_panel_sha256") != configuration_panel_sha256
        or protocol.get("eligibility_owner") != "shared_preprocessing_only"
        or protocol.get("consumer_failure_policy")
        != "fail_closed_no_method_specific_exclusion"
    ):
        raise ValueError("training inclusion source/protocol contract mismatch")
    expected = tuple(str(item) for item in expected_record_ids)
    for record_id in expected:
        record = expected_records.get(record_id)
        raw_fold = record.get("strat_fold") if isinstance(record, Mapping) else None
        try:
            fold = int(raw_fold)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"expected record {record_id} lacks a valid strat_fold") from exc
        if isinstance(raw_fold, bool) or fold not in range(1, 8):
            raise ValueError(
                f"expected record {record_id} is fold {fold}, outside folds1-7/train"
            )
    root = manifest_path.parent
    artifacts = value.get("artifacts", {})
    if not isinstance(artifacts, Mapping):
        raise ValueError("training inclusion lacks artifact descriptors")
    audit_path = _safe_artifact(root, artifacts.get("audit"), expected_name=AUDIT_FILENAME)
    predictor_descriptor = artifacts.get("training_predictors")
    predictor_path = _safe_artifact(
        root, predictor_descriptor, expected_name=TRAINING_PREDICTORS_FILENAME
    )
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read training inclusion audit: {exc}") from exc
    if not isinstance(audit, Mapping) or not isinstance(audit.get("records"), list):
        raise ValueError("training inclusion audit is malformed")
    records = []
    for raw in audit["records"]:
        if not isinstance(raw, Mapping):
            raise ValueError("training inclusion audit record is malformed")
        records.append(SignalAudit(**_tuple_audit_fields(raw)))
    trail = AuditTrail()
    for record in records:
        trail.append(record)
    if (
        audit.get("schema_version") != AUDIT_SCHEMA_VERSION
        or audit.get("source_manifest_sha256") != source_manifest_sha256
        or audit.get("split_sha256") != split_sha256
        or audit.get("train_role") != "folds1-7/train"
        or audit.get("rate_hz") != int(rate_hz)
        or tuple(audit.get("segments", ())) != tuple(segments)
        or audit.get("delineator") != delineator
        or audit.get("summary") != trail.summary_without_hash()
        or audit.get("audit_sha256") != trail.sha256()
    ):
        raise ValueError("training inclusion audit contract mismatch")
    audit_ids = tuple(record.record_id for record in records)
    if audit_ids != expected:
        raise ValueError("training inclusion audit does not cover the exact ordered train role")
    for record in records:
        source_record = expected_records.get(record.record_id)
        if source_record is None or record.patient_id != str(source_record.get("patient_id", "")):
            raise ValueError("training inclusion audit patient identity mismatch")
    included = tuple(record.record_id for record in records if record.status == "included")
    excluded = tuple(record.record_id for record in records if record.status == "excluded")
    included_patients = tuple(
        str(expected_records[record_id]["patient_id"]) for record_id in included
    )
    requested_patients = tuple(
        str(expected_records[record_id]["patient_id"]) for record_id in expected
    )
    excluded_patients = tuple(
        str(expected_records[record_id]["patient_id"]) for record_id in excluded
    )
    raw_signal_records = audit.get("included_signals")
    if not isinstance(raw_signal_records, list):
        raise ValueError("training inclusion lacks per-record signal hashes")
    signal_records: list[dict[str, str]] = []
    for raw in raw_signal_records:
        if not isinstance(raw, Mapping):
            raise ValueError("training inclusion signal hash row is malformed")
        signal_record = {
            "record_id": str(raw.get("record_id", "")),
            "patient_id": str(raw.get("patient_id", "")),
            "sha256": str(raw.get("sha256", "")),
        }
        if len(signal_record["sha256"]) != 64:
            raise ValueError("training inclusion signal hash is malformed")
        signal_records.append(signal_record)
    if tuple(value["record_id"] for value in signal_records) != included:
        raise ValueError("training inclusion signal hashes changed record order")
    if tuple(value["patient_id"] for value in signal_records) != included_patients:
        raise ValueError("training inclusion signal hashes changed patient order")
    ordered_signal_hashes = [value["sha256"] for value in signal_records]
    signals_sha256 = lineage.canonical_sha256(ordered_signal_hashes)
    if audit.get("included_signals_sha256") != signals_sha256:
        raise ValueError("training inclusion signal hash aggregate mismatch")
    expected_blocks = {
        "requested": (expected, requested_patients),
        "included": (included, included_patients),
        "excluded": (excluded, excluded_patients),
    }
    for name, (ids, patients) in expected_blocks.items():
        block = value.get(name, {})
        if (
            not isinstance(block, Mapping)
            or block.get("n_records") != len(ids)
            or block.get("n_patients") != len(set(patients))
            or block.get("record_ids_sha256") != _ordered_ids_sha256(ids)
            or block.get("patient_ids_sha256") != _patient_ids_sha256(patients)
        ):
            raise ValueError(f"training inclusion {name} membership mismatch")
    included_block = value.get("included", {})
    if (
        not isinstance(included_block, Mapping)
        or included_block.get("signals_sha256") != signals_sha256
    ):
        raise ValueError("training inclusion signal membership mismatch")
    if not included:
        raise ValueError("training inclusion contains no included records")
    predictors = pd.read_parquet(predictor_path)
    content = _predictor_content(predictors)
    if (
        not isinstance(predictor_descriptor, Mapping)
        or predictor_descriptor.get("n_rows") != len(predictors)
        or predictor_descriptor.get("content_sha256")
        != lineage.canonical_sha256(content)
    ):
        raise ValueError("training inclusion predictor content mismatch")
    return TrainingInclusion(
        path=manifest_path,
        inclusion_sha256=stored_hash,
        source_manifest_sha256=source_manifest_sha256,
        split_sha256=split_sha256,
        rate_hz=int(rate_hz),
        segments=tuple(segments),
        delineator=delineator,
        configuration_panel_sha256=configuration_panel_sha256,
        requested_record_ids=expected,
        included_record_ids=included,
        included_patient_ids=included_patients,
        record_ids_sha256=_ordered_ids_sha256(included),
        patient_ids_sha256=_patient_ids_sha256(included_patients),
        signal_sha256_by_record={
            value["record_id"]: value["sha256"] for value in signal_records
        },
        audit=audit,
        predictors_path=predictor_path,
        predictors_sha256=str(predictor_descriptor["sha256"]),
        predictor_content_sha256=str(predictor_descriptor["content_sha256"]),
    )


__all__ = [
    "AUDIT_FILENAME",
    "AUDIT_SCHEMA_VERSION",
    "MANIFEST_FILENAME",
    "SCHEMA_VERSION",
    "TrainingInclusion",
    "build_training_inclusion",
    "load_training_inclusion",
]
