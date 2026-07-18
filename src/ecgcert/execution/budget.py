"""Cross-run compute budget accounting with an atomic execution lease."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import secrets
from typing import Any, Mapping

from ecgcert.lineage import canonical_sha256

LEDGER_SCHEMA = "ecg-global-budget-ledger/v1"
SETTLEMENT_SCHEMA = "ecg-run-budget-settlement/v1"
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_ZERO_HASH = "0" * 64


class BudgetError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _validate_usage(value: Mapping[str, Any], *, name: str) -> dict[str, float | int]:
    if not isinstance(value, Mapping) or set(value) != {
        "cpu_core_hours", "gpu_hours", "artifact_bytes",
    }:
        raise BudgetError(f"{name} must contain the exact budget dimensions")
    cpu = value["cpu_core_hours"]
    gpu = value["gpu_hours"]
    artifacts = value["artifact_bytes"]
    if (
        isinstance(cpu, bool) or not isinstance(cpu, (int, float)) or cpu < 0
        or isinstance(gpu, bool) or not isinstance(gpu, (int, float)) or gpu < 0
        or isinstance(artifacts, bool) or not isinstance(artifacts, int) or artifacts < 0
    ):
        raise BudgetError(f"{name} contains an invalid value")
    return {
        "cpu_core_hours": float(cpu),
        "gpu_hours": float(gpu),
        "artifact_bytes": artifacts,
    }


class BudgetLease:
    """Serialize runs and maintain a hash-chained, append-only usage ledger.

    A process crash deliberately leaves the lease directory behind. Operators
    must inspect and reconcile the interrupted run instead of silently stealing
    a potentially live training lease.
    """

    def __init__(
        self,
        *,
        control_root: Path | str,
        run_id: str,
        limits: Mapping[str, Any],
        reserved_gpu_hours: float,
    ) -> None:
        if not _SAFE_ID.fullmatch(run_id):
            raise ValueError("run_id is not a safe identifier")
        self.control_root = Path(control_root).resolve()
        self.run_id = run_id
        self.limits = _validate_usage(limits, name="limits")
        if not isinstance(reserved_gpu_hours, (int, float)) or reserved_gpu_hours < 0:
            raise ValueError("reserved_gpu_hours must be non-negative")
        if reserved_gpu_hours >= self.limits["gpu_hours"]:
            raise ValueError("reserved_gpu_hours must be smaller than the GPU limit")
        self.reserved_gpu_hours = float(reserved_gpu_hours)
        self.lease_dir = self.control_root / ".ecgcert-execution.lease"
        self.owner_path = self.lease_dir / "owner.json"
        self.ledger_path = self.control_root / "budget-ledger.v1.jsonl"
        self._token = ""
        self._reservation: dict[str, Any] | None = None
        self._settled = False

    @staticmethod
    def _event_hash(value: Mapping[str, Any]) -> str:
        payload = dict(value)
        payload.pop("event_sha256", None)
        return canonical_sha256(payload)

    def _read_events(self) -> list[dict[str, Any]]:
        if not self.ledger_path.exists():
            return []
        events: list[dict[str, Any]] = []
        previous = _ZERO_HASH
        try:
            lines = self.ledger_path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise BudgetError(f"cannot read global budget ledger: {exc}") from exc
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                raise BudgetError(f"budget ledger contains an empty line at {line_number}")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BudgetError(f"invalid budget ledger JSON at line {line_number}") from exc
            if not isinstance(value, dict) or value.get("schema_version") != LEDGER_SCHEMA:
                raise BudgetError(f"invalid budget ledger schema at line {line_number}")
            if value.get("previous_event_sha256") != previous:
                raise BudgetError(f"broken budget ledger hash chain at line {line_number}")
            digest = value.get("event_sha256")
            if digest != self._event_hash(value):
                raise BudgetError(f"invalid budget ledger event hash at line {line_number}")
            if value.get("event") not in {"reserved", "settled"}:
                raise BudgetError(f"invalid budget ledger event at line {line_number}")
            if not isinstance(value.get("run_id"), str):
                raise BudgetError(f"invalid budget ledger run_id at line {line_number}")
            previous = digest
            events.append(value)
        return events

    def _append_event(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        events = self._read_events()
        event = {
            "schema_version": LEDGER_SCHEMA,
            **payload,
            "previous_event_sha256": events[-1]["event_sha256"] if events else _ZERO_HASH,
        }
        event["event_sha256"] = self._event_hash(event)
        rendered = json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n"
        with self.ledger_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(rendered)
            stream.flush()
            os.fsync(stream.fileno())
        return event

    @staticmethod
    def _cumulative(events: list[dict[str, Any]]) -> dict[str, float | int]:
        reservations: dict[str, dict[str, Any]] = {}
        settlements: dict[str, dict[str, Any]] = {}
        for event in events:
            run_id = event["run_id"]
            target = reservations if event["event"] == "reserved" else settlements
            if run_id in target:
                raise BudgetError(f"duplicate {event['event']} event for {run_id}")
            target[run_id] = event
        unfinished = sorted(set(reservations) - set(settlements))
        orphaned = sorted(set(settlements) - set(reservations))
        if unfinished or orphaned:
            raise BudgetError(
                f"unreconciled budget ledger: unfinished={unfinished}, orphaned={orphaned}"
            )
        total: dict[str, float | int] = {
            "cpu_core_hours": 0.0,
            "gpu_hours": 0.0,
            "artifact_bytes": 0,
        }
        for event in settlements.values():
            used = _validate_usage(event.get("used", {}), name="settled usage")
            for key in total:
                total[key] += used[key]
        return total

    def acquire(self, planned: Mapping[str, Any]) -> dict[str, Any]:
        planned_usage = _validate_usage(planned, name="planned usage")
        self.control_root.mkdir(parents=True, exist_ok=True)
        try:
            self.lease_dir.mkdir()
        except FileExistsError as exc:
            raise BudgetError(
                f"execution lease is already held: {self.lease_dir}; inspect it, do not steal it"
            ) from exc
        self._token = secrets.token_hex(32)
        owner = {
            "schema_version": "ecg-execution-lease/v1",
            "run_id": self.run_id,
            "pid": os.getpid(),
            "created_at": _utc_now(),
            "token": self._token,
        }
        try:
            self.owner_path.write_text(
                json.dumps(owner, indent=2, sort_keys=True) + "\n", encoding="utf-8",
            )
            events = self._read_events()
            if any(event["run_id"] == self.run_id for event in events):
                raise BudgetError(f"run_id already appears in budget ledger: {self.run_id}")
            prior = self._cumulative(events)
            normal_gpu_limit = self.limits["gpu_hours"] - self.reserved_gpu_hours
            if prior["cpu_core_hours"] + planned_usage["cpu_core_hours"] > self.limits[
                "cpu_core_hours"
            ]:
                raise BudgetError("global planned CPU usage exceeds the frozen budget")
            if prior["gpu_hours"] + planned_usage["gpu_hours"] > normal_gpu_limit:
                raise BudgetError("global planned GPU usage would consume the 100-hour reserve")
            if prior["artifact_bytes"] >= self.limits["artifact_bytes"]:
                raise BudgetError("global artifact budget is already exhausted")
            self._reservation = self._append_event({
                "event": "reserved",
                "run_id": self.run_id,
                "recorded_at": _utc_now(),
                "planned": planned_usage,
                "cumulative_before": prior,
                "normal_gpu_limit": normal_gpu_limit,
            })
            return dict(self._reservation)
        except Exception:
            self._release_exact()
            raise

    def settle(self, used: Mapping[str, Any], *, run_state: str) -> dict[str, Any]:
        if self._reservation is None or self._settled:
            raise BudgetError("budget lease is not acquired or is already settled")
        used_usage = _validate_usage(used, name="actual usage")
        before = _validate_usage(
            self._reservation["cumulative_before"], name="cumulative usage",
        )
        after = {key: before[key] + used_usage[key] for key in before}
        within_limits = all(after[key] <= self.limits[key] for key in after)
        event = self._append_event({
            "event": "settled",
            "run_id": self.run_id,
            "recorded_at": _utc_now(),
            "run_state": run_state,
            "reservation_sha256": self._reservation["event_sha256"],
            "used": used_usage,
            "cumulative_after": after,
            "within_limits": within_limits,
        })
        self._settled = True
        if not within_limits:
            raise BudgetError("actual cumulative usage exceeded the frozen global budget")
        return event

    def _release_exact(self) -> None:
        if not self.lease_dir.exists():
            return
        if self.owner_path.is_file():
            try:
                owner = json.loads(self.owner_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise BudgetError("cannot authenticate execution lease owner") from exc
            if owner.get("token") != self._token or owner.get("run_id") != self.run_id:
                raise BudgetError("execution lease ownership changed; refusing to remove it")
            self.owner_path.unlink()
        self.lease_dir.rmdir()

    def release(self) -> None:
        if self._reservation is not None and not self._settled:
            raise BudgetError("refusing to release an unsettled budget reservation")
        self._release_exact()


def validate_settlement_snapshot(
    value: Mapping[str, Any],
    *,
    expected_run_id: str,
    limits: Mapping[str, Any],
    reserved_gpu_hours: float,
) -> dict[str, Any]:
    """Authenticate the immutable per-run copy of its ledger events."""

    if not isinstance(value, Mapping) or set(value) != {
        "schema_version", "reservation", "settlement",
    }:
        raise BudgetError("budget settlement snapshot has unknown or missing fields")
    if value["schema_version"] != SETTLEMENT_SCHEMA:
        raise BudgetError("budget settlement snapshot schema mismatch")
    reservation = value["reservation"]
    settlement = value["settlement"]
    if not isinstance(reservation, Mapping) or not isinstance(settlement, Mapping):
        raise BudgetError("budget settlement events must be objects")
    for name, event, kind in (
        ("reservation", reservation, "reserved"),
        ("settlement", settlement, "settled"),
    ):
        if event.get("schema_version") != LEDGER_SCHEMA or event.get("event") != kind:
            raise BudgetError(f"{name} event identity mismatch")
        if event.get("run_id") != expected_run_id:
            raise BudgetError(f"{name} run_id mismatch")
        if event.get("event_sha256") != BudgetLease._event_hash(event):
            raise BudgetError(f"{name} event hash mismatch")
    if settlement.get("previous_event_sha256") != reservation.get("event_sha256"):
        raise BudgetError("reservation and settlement are not adjacent in the ledger")
    if settlement.get("reservation_sha256") != reservation.get("event_sha256"):
        raise BudgetError("settlement does not bind the reservation")
    frozen_limits = _validate_usage(limits, name="limits")
    if reservation.get("normal_gpu_limit") != (
        frozen_limits["gpu_hours"] - float(reserved_gpu_hours)
    ):
        raise BudgetError("normal GPU limit does not preserve the declared reserve")
    before = _validate_usage(
        reservation.get("cumulative_before", {}), name="cumulative before",
    )
    planned = _validate_usage(reservation.get("planned", {}), name="planned usage")
    used = _validate_usage(settlement.get("used", {}), name="actual usage")
    after = _validate_usage(
        settlement.get("cumulative_after", {}), name="cumulative after",
    )
    expected_after = {key: before[key] + used[key] for key in before}
    if after != expected_after:
        raise BudgetError("cumulative settlement arithmetic mismatch")
    normal_gpu_limit = frozen_limits["gpu_hours"] - float(reserved_gpu_hours)
    if before["cpu_core_hours"] + planned["cpu_core_hours"] > frozen_limits[
        "cpu_core_hours"
    ]:
        raise BudgetError("reservation exceeds the CPU limit")
    if before["gpu_hours"] + planned["gpu_hours"] > normal_gpu_limit:
        raise BudgetError("reservation consumes the GPU reserve")
    if settlement.get("within_limits") is not True:
        raise BudgetError("settlement is not within frozen limits")
    if any(after[key] > frozen_limits[key] for key in after):
        raise BudgetError("settlement exceeds a frozen limit")
    return {
        "reservation_sha256": reservation["event_sha256"],
        "settlement_sha256": settlement["event_sha256"],
        "used": used,
        "cumulative_after": after,
    }


__all__ = [
    "BudgetError",
    "BudgetLease",
    "LEDGER_SCHEMA",
    "SETTLEMENT_SCHEMA",
    "validate_settlement_snapshot",
]
