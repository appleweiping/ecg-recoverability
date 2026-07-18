"""Patient-level data inclusion and signal-normalisation audit records."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from typing import Any


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
    unit_scales_to_mv: tuple[float, ...] = ()
    segment_counts: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in {"included", "excluded"}:
            raise ValueError("status must be 'included' or 'excluded'")
        if self.status == "excluded" and not self.reason:
            raise ValueError("excluded records require a reason")


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
        return {
            "n_total": len(self.records),
            "n_included": included,
            "n_excluded": len(self.records) - included,
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
