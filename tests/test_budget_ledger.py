import json

import pytest

from ecgcert.execution.budget import (
    BudgetError,
    BudgetLease,
    SETTLEMENT_SCHEMA,
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
