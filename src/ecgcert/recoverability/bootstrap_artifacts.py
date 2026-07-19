"""Fail-closed serialization and replay for patient-bootstrap model banks.

The robust-map producer and downstream evidence validators share this module so
neither side can silently reinterpret sufficient moments or rejected bootstrap
attempts.  Component files may contain several segments; each segment is bound
to one canonical descriptor and both whole-file SHA-256 values are required at
read time.
"""
from __future__ import annotations

import base64
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np

from ecgcert import lineage
from ecgcert.physics import BASIS_VARIANTS, LEADS, BasisVariant
from ecgcert.recoverability.model_bank import (
    PatientBootstrapAttemptLedger,
    PatientBootstrapModelBank,
    PatientClusterSufficientStatistics,
    rebuild_spatial_model_bank,
)


BOOTSTRAP_REPLAY_SCHEMA_VERSION = "robust-recoverability-bootstrap-replay-v1"
BOOTSTRAP_MOMENTS_SCHEMA_VERSION = "robust-recoverability-bootstrap-moments-v1"
BOOTSTRAP_ATTEMPT_SCHEMA_VERSION = "robust-recoverability-bootstrap-attempt-v1"


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _encoded_patient_id(patient_id: Hashable) -> tuple[str, str]:
    if isinstance(patient_id, str):
        return "str", patient_id
    if isinstance(patient_id, UUID):
        return "uuid", str(patient_id)
    if isinstance(patient_id, (bytes, bytearray)):
        return "bytes", base64.b64encode(bytes(patient_id)).decode("ascii")
    if isinstance(patient_id, (bool, np.bool_)):
        raise TypeError("boolean patient ids are not serializable")
    if isinstance(patient_id, (int, np.integer)):
        return "int", str(int(patient_id))
    if isinstance(patient_id, (float, np.floating)):
        value = float(patient_id)
        if not np.isfinite(value):
            raise TypeError("non-finite patient ids are not serializable")
        return "float", repr(value)
    raise TypeError(
        "bootstrap artifact patient ids must be str, UUID, bytes, int, or finite float; "
        f"got {type(patient_id).__name__}"
    )


def _decoded_patient_id(kind: str, value: str) -> Hashable:
    try:
        if kind == "str":
            return value
        if kind == "uuid":
            return UUID(value)
        if kind == "bytes":
            return base64.b64decode(value.encode("ascii"), validate=True)
        if kind == "int":
            return int(value)
        if kind == "float":
            result = float(value)
            if not np.isfinite(result):
                raise ValueError("decoded float is non-finite")
            return result
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid serialized patient id ({kind!r}, {value!r})") from exc
    raise ValueError(f"unsupported serialized patient-id kind {kind!r}")


def sufficient_statistics_sha256(
    statistics: PatientClusterSufficientStatistics,
) -> str:
    """Hash canonical patient identities and exact little-endian moment bytes."""

    hasher = hashlib.sha256()
    identities = [_encoded_patient_id(patient_id) for patient_id in statistics.patient_ids]
    hasher.update(_canonical_json(identities).encode("utf-8"))
    arrays = (
        ("origin", statistics.origin, "<f8"),
        ("counts", statistics.counts, "<i8"),
        ("sums", statistics.sums, "<f8"),
        ("crossproducts", statistics.crossproducts, "<f8"),
    )
    for name, values, dtype in arrays:
        array = np.ascontiguousarray(np.asarray(values, dtype=np.dtype(dtype)))
        hasher.update(name.encode("ascii"))
        hasher.update(_canonical_json(list(array.shape)).encode("ascii"))
        hasher.update(array.tobytes(order="C"))
    return hasher.hexdigest()


@dataclass(frozen=True)
class BootstrapReplayDescriptor:
    """Cross-file identity for one segment's replayable bootstrap evidence."""

    segment: str
    ranks: tuple[int, ...]
    basis_variants: tuple[BasisVariant, ...]
    fit_cohort: str
    seed: int
    n_patients: int
    n_boot: int
    n_attempts: int
    statistics_sha256: str

    @classmethod
    def from_bank(
        cls,
        bank: PatientBootstrapModelBank,
        *,
        segment: str,
    ) -> BootstrapReplayDescriptor:
        return cls(
            segment=segment,
            ranks=tuple(bank.ranks),
            basis_variants=tuple(bank.basis_variants),
            fit_cohort=bank.fit_cohort,
            seed=bank.seed,
            n_patients=bank.statistics.n_patients,
            n_boot=bank.n_boot,
            n_attempts=bank.attempt_ledger.n_attempts,
            statistics_sha256=sufficient_statistics_sha256(bank.statistics),
        )

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": BOOTSTRAP_REPLAY_SCHEMA_VERSION,
            "segment": self.segment,
            "ranks": list(self.ranks),
            "basis_variants": list(self.basis_variants),
            "fit_cohort": self.fit_cohort,
            "seed": self.seed,
            "n_patients": self.n_patients,
            "n_boot": self.n_boot,
            "n_attempts": self.n_attempts,
            "statistics_sha256": self.statistics_sha256,
        }

    @property
    def canonical_json(self) -> str:
        return _canonical_json(self.payload())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()


def _large_list_array(matrix: np.ndarray, *, value_type):
    import pyarrow as pa

    values = np.asarray(matrix)
    if values.ndim != 2:
        raise ValueError("nested artifact values must be a matrix")
    flat = pa.array(values.reshape(-1), type=value_type)
    offsets = pa.array(
        np.arange(values.shape[0] + 1, dtype=np.int64) * values.shape[1],
        type=pa.int64(),
    )
    return pa.LargeListArray.from_arrays(offsets, flat)


def bootstrap_moments_table(
    bank: PatientBootstrapModelBank,
    *,
    segment: str,
):
    """Return one Arrow table containing replayable patient-level moments."""

    import pyarrow as pa

    descriptor = BootstrapReplayDescriptor.from_bank(bank, segment=segment)
    statistics = bank.statistics
    n_patients = statistics.n_patients
    patient_identity = [
        _encoded_patient_id(patient_id) for patient_id in statistics.patient_ids
    ]
    origins = np.broadcast_to(statistics.origin, (n_patients, len(LEADS)))
    return pa.table(
        {
            "schema_version": pa.array(
                [BOOTSTRAP_MOMENTS_SCHEMA_VERSION] * n_patients,
                type=pa.string(),
            ),
            "descriptor_json": pa.array(
                [descriptor.canonical_json] * n_patients,
                type=pa.string(),
            ),
            "descriptor_sha256": pa.array(
                [descriptor.sha256] * n_patients,
                type=pa.string(),
            ),
            "segment": pa.array([segment] * n_patients, type=pa.string()),
            "patient_index": pa.array(np.arange(n_patients), type=pa.int32()),
            "patient_id_kind": pa.array(
                [kind for kind, _ in patient_identity],
                type=pa.string(),
            ),
            "patient_id_value": pa.array(
                [value for _, value in patient_identity],
                type=pa.string(),
            ),
            "origin": _large_list_array(origins, value_type=pa.float64()),
            "count": pa.array(statistics.counts, type=pa.int64()),
            "sums": _large_list_array(statistics.sums, value_type=pa.float64()),
            "crossproducts": _large_list_array(
                statistics.crossproducts.reshape(n_patients, -1),
                value_type=pa.float64(),
            ),
        }
    )


def bootstrap_attempts_table(
    bank: PatientBootstrapModelBank,
    *,
    segment: str,
):
    """Return one Arrow table retaining all accepted and rejected proposals."""

    import pyarrow as pa

    descriptor = BootstrapReplayDescriptor.from_bank(bank, segment=segment)
    ledger = bank.attempt_ledger
    if ledger.multiplicities.dtype == np.dtype(np.uint16):
        multiplicity_type = pa.uint16()
    elif ledger.multiplicities.dtype == np.dtype(np.uint32):
        multiplicity_type = pa.uint32()
    else:  # pragma: no cover - ledger normalization guarantees this.
        raise ValueError("unsupported multiplicity dtype")
    status = np.where(ledger.accepted, "accepted", "rejected_rank_deficient")
    return pa.table(
        {
            "schema_version": pa.array(
                [BOOTSTRAP_ATTEMPT_SCHEMA_VERSION] * ledger.n_attempts,
                type=pa.string(),
            ),
            "descriptor_json": pa.array(
                [descriptor.canonical_json] * ledger.n_attempts,
                type=pa.string(),
            ),
            "descriptor_sha256": pa.array(
                [descriptor.sha256] * ledger.n_attempts,
                type=pa.string(),
            ),
            "segment": pa.array([segment] * ledger.n_attempts, type=pa.string()),
            "attempt_index": pa.array(
                np.arange(ledger.n_attempts),
                type=pa.int32(),
            ),
            "accepted": pa.array(ledger.accepted, type=pa.bool_()),
            "accepted_bootstrap_index": pa.array(
                ledger.accepted_bootstrap_index,
                type=pa.int32(),
            ),
            "status": pa.array(status, type=pa.string()),
            "multiplicities": _large_list_array(
                ledger.multiplicities,
                value_type=multiplicity_type,
            ),
        }
    )


def _verify_sha256(path: Path, expected: str, *, label: str) -> None:
    if (
        not isinstance(expected, str)
        or len(expected) != 64
        or any(character not in "0123456789abcdef" for character in expected)
    ):
        raise ValueError(f"{label} requires a full lowercase SHA-256")
    actual = lineage.artifact_sha256(path)
    if actual != expected:
        raise ValueError(f"{label} artifact SHA-256 mismatch")


def _filtered_component(path: Path, *, segment: str, required_columns: set[str]):
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    table = pq.read_table(path).replace_schema_metadata(None)
    actual_columns = set(table.column_names)
    if actual_columns != required_columns:
        missing = sorted(required_columns - actual_columns)
        extra = sorted(actual_columns - required_columns)
        raise ValueError(
            "bootstrap artifact schema columns must match exactly; "
            f"missing={missing}, extra={extra}"
        )
    segment_column = table.column("segment")
    if segment_column.type != pa.string() or segment_column.null_count:
        raise ValueError("bootstrap artifact segment column must be non-null string")
    filtered = table.filter(pc.equal(segment_column, pa.scalar(segment)))
    if filtered.num_rows < 1:
        raise ValueError(f"bootstrap artifact has no rows for segment {segment!r}")
    return filtered


def _single_string(table, name: str) -> str:
    import pyarrow as pa

    column = table.column(name)
    if column.type != pa.string() or column.null_count:
        raise ValueError(f"{name} must be a non-null string column")
    values = column.to_pylist()
    if len(set(values)) != 1:
        raise ValueError(f"{name} must be constant within a segment")
    return str(values[0])


def _descriptor_payload(table) -> dict[str, Any]:
    descriptor_json = _single_string(table, "descriptor_json")
    descriptor_sha256 = _single_string(table, "descriptor_sha256")
    actual_sha256 = hashlib.sha256(descriptor_json.encode("utf-8")).hexdigest()
    if descriptor_sha256 != actual_sha256:
        raise ValueError("bootstrap descriptor SHA-256 mismatch")
    try:
        payload = json.loads(descriptor_json)
    except json.JSONDecodeError as exc:
        raise ValueError("bootstrap descriptor is not valid JSON") from exc
    if not isinstance(payload, dict) or _canonical_json(payload) != descriptor_json:
        raise ValueError("bootstrap descriptor is not canonical JSON")
    if payload.get("schema_version") != BOOTSTRAP_REPLAY_SCHEMA_VERSION:
        raise ValueError("unsupported bootstrap replay descriptor schema")
    return payload


def _integer_column(table, name: str, *, arrow_type) -> np.ndarray:
    column = table.column(name)
    if column.type != arrow_type or column.null_count:
        raise ValueError(f"{name} must be a non-null {arrow_type} column")
    return np.asarray(column.to_numpy(), dtype=arrow_type.to_pandas_dtype())


def _list_matrix(table, name: str, *, width: int, value_types: tuple) -> np.ndarray:
    import pyarrow as pa

    column = table.column(name).combine_chunks()
    if not pa.types.is_large_list(column.type) or column.null_count:
        raise ValueError(f"{name} must be a non-null large-list column")
    if column.type.value_type not in value_types:
        raise ValueError(f"{name} has unsupported value type {column.type.value_type}")
    offsets = np.asarray(column.offsets.to_numpy(), dtype=np.int64)
    if not np.all(np.diff(offsets) == width):
        raise ValueError(f"every {name} row must contain exactly {width} values")
    start = int(offsets[0])
    length = int(offsets[-1] - offsets[0])
    flat = column.values.slice(start, length).to_numpy(zero_copy_only=False)
    return np.asarray(flat).reshape(table.num_rows, width)


def read_bootstrap_replay_artifacts(
    moments_path: str | Path,
    attempts_path: str | Path,
    *,
    artifact_sha256: Mapping[str, str],
    segment: str,
    ranks: Sequence[int],
    basis_variants: Sequence[BasisVariant],
    fit_cohort: str,
    seed: int,
) -> tuple[
    PatientClusterSufficientStatistics,
    PatientBootstrapAttemptLedger,
    BootstrapReplayDescriptor,
]:
    """Authenticate and decode one segment's sufficient moments and ledger."""

    import pyarrow as pa

    if not isinstance(artifact_sha256, Mapping):
        raise ValueError("artifact_sha256 mapping is required for replay")
    required_hashes = {"bootstrap_moments", "bootstrap_attempts"}
    if not required_hashes <= set(artifact_sha256):
        raise ValueError("artifact_sha256 lacks bootstrap_moments/bootstrap_attempts")
    moments_path = Path(moments_path)
    attempts_path = Path(attempts_path)
    _verify_sha256(
        moments_path,
        artifact_sha256["bootstrap_moments"],
        label="bootstrap moments",
    )
    _verify_sha256(
        attempts_path,
        artifact_sha256["bootstrap_attempts"],
        label="bootstrap attempts",
    )

    moment_columns = {
        "schema_version",
        "descriptor_json",
        "descriptor_sha256",
        "segment",
        "patient_index",
        "patient_id_kind",
        "patient_id_value",
        "origin",
        "count",
        "sums",
        "crossproducts",
    }
    attempt_columns = {
        "schema_version",
        "descriptor_json",
        "descriptor_sha256",
        "segment",
        "attempt_index",
        "accepted",
        "accepted_bootstrap_index",
        "status",
        "multiplicities",
    }
    moments = _filtered_component(
        moments_path,
        segment=segment,
        required_columns=moment_columns,
    )
    attempts = _filtered_component(
        attempts_path,
        segment=segment,
        required_columns=attempt_columns,
    )
    if _single_string(moments, "schema_version") != BOOTSTRAP_MOMENTS_SCHEMA_VERSION:
        raise ValueError("unsupported bootstrap moments schema")
    if _single_string(attempts, "schema_version") != BOOTSTRAP_ATTEMPT_SCHEMA_VERSION:
        raise ValueError("unsupported bootstrap attempt schema")
    moment_descriptor = _descriptor_payload(moments)
    attempt_descriptor = _descriptor_payload(attempts)
    if moment_descriptor != attempt_descriptor:
        raise ValueError("bootstrap moment and attempt descriptors do not match")

    n_patients = moments.num_rows
    patient_index = _integer_column(
        moments,
        "patient_index",
        arrow_type=pa.int32(),
    )
    if not np.array_equal(patient_index, np.arange(n_patients, dtype=np.int32)):
        raise ValueError("patient rows must be in dense patient_index order")
    kinds = moments.column("patient_id_kind")
    values = moments.column("patient_id_value")
    if (
        kinds.type != pa.string()
        or values.type != pa.string()
        or kinds.null_count
        or values.null_count
    ):
        raise ValueError("serialized patient ids must be non-null strings")
    patient_ids = tuple(
        _decoded_patient_id(kind, value)
        for kind, value in zip(kinds.to_pylist(), values.to_pylist())
    )
    origins = _list_matrix(
        moments,
        "origin",
        width=len(LEADS),
        value_types=(pa.float64(),),
    ).astype(float, copy=False)
    if not np.all(origins == origins[0]):
        raise ValueError("origin must be identical in every patient moment row")
    counts = _integer_column(moments, "count", arrow_type=pa.int64())
    sums = _list_matrix(
        moments,
        "sums",
        width=len(LEADS),
        value_types=(pa.float64(),),
    ).astype(float, copy=False)
    crossproducts = _list_matrix(
        moments,
        "crossproducts",
        width=len(LEADS) * len(LEADS),
        value_types=(pa.float64(),),
    ).astype(float, copy=False).reshape(n_patients, len(LEADS), len(LEADS))
    statistics = PatientClusterSufficientStatistics(
        patient_ids=patient_ids,
        origin=origins[0],
        counts=counts,
        sums=sums,
        crossproducts=crossproducts,
    )

    n_attempts = attempts.num_rows
    attempt_index = _integer_column(
        attempts,
        "attempt_index",
        arrow_type=pa.int32(),
    )
    if not np.array_equal(attempt_index, np.arange(n_attempts, dtype=np.int32)):
        raise ValueError("attempt rows must be in dense attempt_index order")
    accepted_column = attempts.column("accepted")
    if accepted_column.type != pa.bool_() or accepted_column.null_count:
        raise ValueError("accepted must be a non-null boolean column")
    accepted = np.asarray(accepted_column.to_numpy(), dtype=np.bool_)
    accepted_bootstrap_index = _integer_column(
        attempts,
        "accepted_bootstrap_index",
        arrow_type=pa.int32(),
    )
    status = attempts.column("status")
    if status.type != pa.string() or status.null_count:
        raise ValueError("attempt status must be a non-null string column")
    expected_status = np.where(accepted, "accepted", "rejected_rank_deficient")
    if status.to_pylist() != expected_status.tolist():
        raise ValueError("attempt status disagrees with accepted flag")
    expected_value_type = (
        pa.uint16() if n_patients <= np.iinfo(np.uint16).max else pa.uint32()
    )
    multiplicities = _list_matrix(
        attempts,
        "multiplicities",
        width=n_patients,
        value_types=(expected_value_type,),
    )
    ledger = PatientBootstrapAttemptLedger(
        multiplicities=multiplicities,
        accepted=accepted,
        accepted_bootstrap_index=accepted_bootstrap_index,
    )

    descriptor = BootstrapReplayDescriptor(
        segment=segment,
        ranks=tuple(int(rank) for rank in ranks),
        basis_variants=tuple(basis_variants),
        fit_cohort=fit_cohort,
        seed=int(seed),
        n_patients=statistics.n_patients,
        n_boot=ledger.n_boot,
        n_attempts=ledger.n_attempts,
        statistics_sha256=sufficient_statistics_sha256(statistics),
    )
    if moment_descriptor != descriptor.payload():
        raise ValueError("bootstrap descriptor does not match decoded replay evidence")
    if descriptor.sha256 != hashlib.sha256(
        _canonical_json(moment_descriptor).encode("utf-8")
    ).hexdigest():  # pragma: no cover - equality above already implies this.
        raise ValueError("bootstrap descriptor digest does not match decoded evidence")
    return statistics, ledger, descriptor


def rebuild_model_bank_from_artifacts(
    moments_path: str | Path,
    attempts_path: str | Path,
    *,
    artifact_sha256: Mapping[str, str],
    segment: str,
    ranks: Sequence[int],
    basis_variants: Sequence[BasisVariant] = BASIS_VARIANTS,
    fit_cohort: str,
    seed: int,
) -> PatientBootstrapModelBank:
    """Read, authenticate, and replay every accepted/rejected model-bank draw."""

    statistics, ledger, _ = read_bootstrap_replay_artifacts(
        moments_path,
        attempts_path,
        artifact_sha256=artifact_sha256,
        segment=segment,
        ranks=ranks,
        basis_variants=basis_variants,
        fit_cohort=fit_cohort,
        seed=seed,
    )
    return rebuild_spatial_model_bank(
        statistics,
        ledger,
        ranks=ranks,
        basis_variants=basis_variants,
        fit_cohort=fit_cohort,
        seed=seed,
        verify_rng_sequence=True,
    )


__all__ = [
    "BOOTSTRAP_ATTEMPT_SCHEMA_VERSION",
    "BOOTSTRAP_MOMENTS_SCHEMA_VERSION",
    "BOOTSTRAP_REPLAY_SCHEMA_VERSION",
    "BootstrapReplayDescriptor",
    "bootstrap_attempts_table",
    "bootstrap_moments_table",
    "read_bootstrap_replay_artifacts",
    "rebuild_model_bank_from_artifacts",
    "sufficient_statistics_sha256",
]
