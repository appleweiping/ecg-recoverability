import json
import os

import pytest

from ecgcert.execution.budget import (
    BudgetError,
    BudgetLease,
    SETTLEMENT_SCHEMA,
    audit_unsettled_reservation,
    validate_settlement_snapshot,
)


LIMITS = {
    "cpu_core_hours": 20.0,
    "gpu_hours": 10.0,
    "artifact_bytes": 1_000,
}


def _usage(cpu=0.0, gpu=0.0, artifacts=0):
    return {
        "cpu_core_hours": cpu,
        "gpu_hours": gpu,
        "artifact_bytes": artifacts,
    }


def test_budget_lease_serializes_runs_and_writes_valid_settlement(tmp_path):
    first = BudgetLease(
        control_root=tmp_path,
        run_id="run-1",
        limits=LIMITS,
        reserved_gpu_hours=2.0,
    )
    reservation = first.acquire(_usage(cpu=4, gpu=3))
    contender = BudgetLease(
        control_root=tmp_path,
        run_id="run-2",
        limits=LIMITS,
        reserved_gpu_hours=2.0,
    )
    with pytest.raises(BudgetError, match="already held"):
        contender.acquire(_usage(cpu=1))

    settlement = first.settle(_usage(cpu=1.5, gpu=0.5, artifacts=40), run_state="succeeded")
    first.release()
    normalized = validate_settlement_snapshot(
        {
            "schema_version": SETTLEMENT_SCHEMA,
            "reservation": reservation,
            "settlement": settlement,
        },
        expected_run_id="run-1",
        limits=LIMITS,
        reserved_gpu_hours=2.0,
    )
    assert normalized["used"] == _usage(cpu=1.5, gpu=0.5, artifacts=40)
    assert not (tmp_path / ".ecgcert-execution.lease").exists()


def test_budget_ledger_enforces_cumulative_limit_and_gpu_reserve(tmp_path):
    first = BudgetLease(
        control_root=tmp_path,
        run_id="run-1",
        limits=LIMITS,
        reserved_gpu_hours=2.0,
    )
    first.acquire(_usage(cpu=4, gpu=7))
    first.settle(_usage(cpu=3, gpu=7), run_state="failed")
    first.release()

    second = BudgetLease(
        control_root=tmp_path,
        run_id="run-2",
        limits=LIMITS,
        reserved_gpu_hours=2.0,
    )
    with pytest.raises(BudgetError, match="reserve"):
        second.acquire(_usage(cpu=1, gpu=2))


def test_budget_ledger_tampering_fails_closed(tmp_path):
    lease = BudgetLease(
        control_root=tmp_path,
        run_id="run-1",
        limits=LIMITS,
        reserved_gpu_hours=2.0,
    )
    lease.acquire(_usage(cpu=1))
    lease.settle(_usage(cpu=0.5), run_state="succeeded")
    lease.release()
    ledger = tmp_path / "budget-ledger.v1.jsonl"
    lines = ledger.read_text(encoding="utf-8").splitlines()
    value = json.loads(lines[0])
    value["planned"]["cpu_core_hours"] = 19.0
    lines[0] = json.dumps(value)
    ledger.write_text("\n".join(lines) + "\n", encoding="utf-8")

    next_lease = BudgetLease(
        control_root=tmp_path,
        run_id="run-2",
        limits=LIMITS,
        reserved_gpu_hours=2.0,
    )
    with pytest.raises(BudgetError, match="event hash"):
        next_lease.acquire(_usage(cpu=1))


def test_unsettled_recovery_audit_is_exact_and_read_only(tmp_path):
    lease = BudgetLease(
        control_root=tmp_path,
        run_id="crashed-run",
        limits=LIMITS,
        reserved_gpu_hours=2.0,
    )
    planned = _usage(cpu=4, gpu=3, artifacts=1_000)
    reservation = lease.acquire(planned)
    ledger_before = lease.ledger_path.read_bytes()
    owner_before = lease.owner_path.read_bytes()

    report = audit_unsettled_reservation(
        control_root=tmp_path,
        expected_run_id="crashed-run",
        expected_reservation_sha256=reservation["event_sha256"],
        expected_owner_pid=os.getpid(),
        expected_planned=planned,
        limits=LIMITS,
        reserved_gpu_hours=2.0,
    )

    assert report["reservation_sha256"] == reservation["event_sha256"]
    assert report["ledger_event_count"] == 1
    assert len(report["ledger_sha256"]) == 64
    assert report["ledger_size_bytes"] == len(ledger_before)
    assert report["planned"] == planned
    assert lease.ledger_path.read_bytes() == ledger_before
    assert lease.owner_path.read_bytes() == owner_before


def test_unsettled_recovery_audit_refuses_wrong_owner_or_finished_ledger(tmp_path):
    lease = BudgetLease(
        control_root=tmp_path,
        run_id="crashed-run",
        limits=LIMITS,
        reserved_gpu_hours=2.0,
    )
    planned = _usage(cpu=1)
    reservation = lease.acquire(planned)
    with pytest.raises(BudgetError, match="owner identity"):
        audit_unsettled_reservation(
            control_root=tmp_path,
            expected_run_id="crashed-run",
            expected_reservation_sha256=reservation["event_sha256"],
            expected_owner_pid=os.getpid() + 1,
            expected_planned=planned,
            limits=LIMITS,
            reserved_gpu_hours=2.0,
        )

    lease.settle(_usage(), run_state="failed")
    with pytest.raises(BudgetError, match="unfinished reservation"):
        audit_unsettled_reservation(
            control_root=tmp_path,
            expected_run_id="crashed-run",
            expected_reservation_sha256=reservation["event_sha256"],
            expected_owner_pid=os.getpid(),
            expected_planned=planned,
            limits=LIMITS,
            reserved_gpu_hours=2.0,
        )


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), float("-inf")])
def test_budget_contract_rejects_nonfinite_values(tmp_path, nonfinite):
    with pytest.raises((BudgetError, ValueError), match="invalid value|non-negative"):
        BudgetLease(
            control_root=tmp_path,
            run_id="bad-limit",
            limits={**LIMITS, "gpu_hours": nonfinite},
            reserved_gpu_hours=2.0,
        )

    lease = BudgetLease(
        control_root=tmp_path,
        run_id="bad-plan",
        limits=LIMITS,
        reserved_gpu_hours=2.0,
    )
    with pytest.raises(BudgetError, match="invalid value"):
        lease.acquire(_usage(cpu=nonfinite))

    with pytest.raises(ValueError, match="non-negative"):
        BudgetLease(
            control_root=tmp_path,
            run_id="bad-reserve",
            limits=LIMITS,
            reserved_gpu_hours=nonfinite,
        )
