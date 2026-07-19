"""Patient-level data inclusion and signal-normalisation audit records."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from typing import Any

from ecgcert.data.common import CANONICAL_LEADS


@dataclass(frozen=True)
class SignalAudit:
    cohort: str
    record_id: str
    patient_id: str
    status: str
    reason: str | None
    requested_rate_hz: int
    source_rate_hz: float | None = None
    n_samples: int | None = None
    input_leads: tuple[str, ...] = ()
    input_units: tuple[str, ...] = ()
    canonical_leads: tuple[str, ...] = ()
    source_channel_indices: tuple[int, ...] = ()
    unit_scales_to_mv: tuple[float, ...] = ()
    output_unit: str | None = None
    segment_counts: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.cohort or not self.record_id or not self.patient_id:
            raise ValueError("audit cohort, record_id and patient_id must be non-empty")
        if self.status not in {"included", "excluded"}:
            raise ValueError("status must be 'included' or 'excluded'")
        if self.status == "excluded" and not self.reason:
            raise ValueError("excluded records require a reason")
        if self.status == "included" and self.reason not in (None, ""):
            raise ValueError("included records cannot carry an exclusion reason")
        if isinstance(self.requested_rate_hz, bool) or self.requested_rate_hz <= 0:
            raise ValueError("requested_rate_hz must be positive")
        if self.source_rate_hz is not None and (
            not math.isfinite(float(self.source_rate_hz)) or self.source_rate_hz <= 0
        ):
            raise ValueError("source_rate_hz must be finite and positive")
        if self.n_samples is not None and (
            isinstance(self.n_samples, bool) or self.n_samples < 0
        ):
            raise ValueError("n_samples must be a non-negative integer")
        if self.input_leads or self.input_units:
            if len(self.input_leads) != len(self.input_units):
                raise ValueError("audit input lead/unit lengths disagree")
        if self.canonical_leads or self.source_channel_indices:
            n_outputs = len(self.canonical_leads)
            if (
                not n_outputs
                or len(self.source_channel_indices) != n_outputs
                or len(self.unit_scales_to_mv) != n_outputs
            ):
                raise ValueError("audit channel reorder/unit conversion lengths disagree")
            if self.canonical_leads != CANONICAL_LEADS:
                raise ValueError("audit canonical lead order is not the locked twelve-lead order")
            if any(
                isinstance(index, bool)
                or not isinstance(index, int)
                or index < 0
                or index >= len(self.input_leads)
                for index in self.source_channel_indices
            ):
                raise ValueError("audit source channel index is invalid")
            if len(set(self.source_channel_indices)) != n_outputs:
                raise ValueError("audit channel reorder reuses a source channel")
            if self.output_unit != "mV":
                raise ValueError("canonical ECG audit output unit must be mV")
        if any(
            not math.isfinite(float(scale)) or scale <= 0
            for scale in self.unit_scales_to_mv
        ):
            raise ValueError("unit scales to mV must be finite and positive")
        if any(
            not isinstance(segment, str)
            or not segment
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
            for segment, count in self.segment_counts.items()
        ):
            raise ValueError("segment_counts must contain non-negative integer counts")


class AuditTrail:
    """In-memory audit trail; callers decide where immutable artifacts are written."""

    def __init__(self) -> None:
        self.records: list[SignalAudit] = []

    def append(self, record: SignalAudit) -> None:
        self.records.append(record)

    def summary_without_hash(self) -> dict[str, Any]:
        reasons: dict[str, int] = {}
        for record in self.records:
            if record.status == "excluded":
                key = record.reason or "unknown"
                reasons[key] = reasons.get(key, 0) + 1
        included = sum(record.status == "included" for record in self.records)
        patient_status: dict[str, set[str]] = {}
        for record in self.records:
            patient_status.setdefault(record.patient_id, set()).add(record.status)
        return {
            "n_total": len(self.records),
            "n_included": included,
            "n_excluded": len(self.records) - included,
            "n_patients_total": len(patient_status),
            "n_patients_with_included_records": sum(
                "included" in statuses for statuses in patient_status.values()
            ),
            "n_patients_all_records_excluded": sum(
                statuses == {"excluded"} for statuses in patient_status.values()
            ),
            "exclusion_reasons": dict(sorted(reasons.items())),
        }

    def summary(self) -> dict[str, Any]:
        return {**self.summary_without_hash(), "sha256": self.sha256()}

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary_without_hash(),
            "records": [asdict(record) for record in self.records],
        }

    def sha256(self) -> str:
        payload = json.dumps(
            [asdict(record) for record in self.records],
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
