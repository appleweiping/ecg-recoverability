import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from ecgcert.execution import remote_job
from ecgcert.execution.budget import BudgetLease
from ecgcert.execution.remote_job import (
    RemoteJobError,
    attach_job,
    audit_recovery_job,
    job_status,
    launch_detached_job,
    process_identity,
    process_identity_is_alive,
    validate_recovery_audit_report,
)


ROOT = Path(__file__).resolve().parents[1]
SUPERVISOR = ROOT / "scripts" / "remote_job.py"


def test_selected_interpreter_symlink_is_not_resolved_away(tmp_path: Path) -> None:
    interpreter = tmp_path / "python-real"
    interpreter.write_bytes(b"fixture")
    link = tmp_path / "venv-python"
    try:
        link.symlink_to(interpreter)
    except OSError:
        pytest.skip("test host cannot create symlinks")
    selected = remote_job._require_absolute_file(  # noqa: SLF001
        link.absolute(), label="Python executable", preserve_symlink=True
    )
    assert selected == link.absolute()


def _wait_terminal(run_root: Path, run_id: str, timeout: float = 30.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = job_status(run_root=run_root, run_id=run_id)
        if status["state"] in {"finished", "inconsistent", "launch_failed", "orphaned"}:
            return status
        time.sleep(0.05)
    raise AssertionError("detached remote job did not reach a terminal state")


def _launch(
    *,
    repo: Path,
    run_root: Path,
    run_id: str,
    command: list[str],
    resume: bool = False,
) -> dict:
    return launch_detached_job(
        repo=repo,
        run_root=run_root,
        run_id=run_id,
        command=command,
        resume=resume,
        profile="icassp",
        resource=None,
        environment_lock="gpu",
        python_executable=Path(sys.executable),
        supervisor_entry=SUPERVISOR,
        startup_wait_seconds=2.0,
    )


def test_detached_launch_survives_launcher_return_and_reconnects(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    run_root = tmp_path / "runs"
    repo.mkdir()
    run_root.mkdir()
    command = [
        sys.executable,
        "-c",
        "import time; print('detached-start', flush=True); "
        "time.sleep(0.6); print('detached-finish', flush=True)",
    ]

    started = time.monotonic()
    first = _launch(repo=repo, run_root=run_root, run_id="run-detached", command=command)
    assert time.monotonic() - started < 5.0
    assert first["attempt_id"] == "initial"

    # A lost launch response can be retried without creating a second process.
    repeated = _launch(repo=repo, run_root=run_root, run_id="run-detached", command=command)
    assert repeated["attempt_id"] == "initial"

    final = _wait_terminal(run_root, "run-detached")
    assert final["state"] == "finished"
    assert final["dag_exit_code"] == 0
    attached = attach_job(run_root=run_root, run_id="run-detached")
    assert "detached-start" in attached["logs"]["dag_stdout"]
    assert "detached-finish" in attached["logs"]["dag_stdout"]


def test_detached_attempt_outlives_a_separate_launcher_process(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    run_root = tmp_path / "runs"
    marker = tmp_path / "ssh-channel-closed.marker"
    repo.mkdir()
    run_root.mkdir()
    dag_code = (
        "from pathlib import Path; import time; time.sleep(0.8); "
        f"Path({str(marker)!r}).write_text('survived-launcher-exit')"
    )
    launcher_code = "\n".join(
        [
            "from pathlib import Path",
            "import sys",
            "from ecgcert.execution.remote_job import launch_detached_job",
            "launch_detached_job(",
            f"    repo=Path({str(repo)!r}),",
            f"    run_root=Path({str(run_root)!r}),",
            "    run_id='separate-launcher',",
            f"    command=[sys.executable, '-c', {dag_code!r}],",
            "    resume=False, profile='icassp', resource=None,",
            "    environment_lock='gpu',",
            "    python_executable=Path(sys.executable),",
            f"    supervisor_entry=Path({str(SUPERVISOR)!r}),",
            "    startup_wait_seconds=3.0,",
            ")",
        ]
    )

    launcher = subprocess.run(
        [sys.executable, "-c", launcher_code],
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert launcher.returncode == 0, launcher.stderr
    final = _wait_terminal(run_root, "separate-launcher")
    assert final["state"] == "finished"
    assert marker.read_text(encoding="utf-8") == "survived-launcher-exit"


def test_live_duplicate_launch_must_match_the_frozen_logical_job(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    run_root = tmp_path / "runs"
    repo.mkdir()
    run_root.mkdir()
    command = [sys.executable, "-c", "import time; time.sleep(3)"]

    _launch(repo=repo, run_root=run_root, run_id="identity-run", command=command)
    with pytest.raises(RemoteJobError, match="logical job identity"):
        _launch(
            repo=repo,
            run_root=run_root,
            run_id="identity-run",
            command=[sys.executable, "-c", "print('different command')"],
        )
    assert _wait_terminal(run_root, "identity-run")["state"] == "finished"


def test_failed_detached_attempt_can_be_resumed_after_reconnect(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    run_root = tmp_path / "runs"
    run_dir = run_root / "run-resume"
    repo.mkdir()
    run_root.mkdir()
    status_path = run_dir / "status.json"
    marker = tmp_path / "resume.marker"
    command_code = "\n".join(
        [
            "from pathlib import Path",
            "import json, sys",
            f"status = Path({str(status_path)!r})",
            f"marker = Path({str(marker)!r})",
            "status.parent.mkdir(parents=True, exist_ok=True)",
            "if marker.exists():",
            "    status.write_text(json.dumps({'run_id': 'run-resume', "
            "'state': 'succeeded', 'exit_code': 0, 'attempts': [{"
            "'attempt_id': 'initial'}], 'active_attempt': None}))",
            "    print('resumed-success', flush=True)",
            "else:",
            "    marker.write_text('failed-once')",
            "    status.write_text(json.dumps({'run_id': 'run-resume', "
            "'state': 'failed', 'exit_code': 7, 'attempts': [{"
            "'attempt_id': 'initial'}], 'active_attempt': None}))",
            "    print('first-attempt-failed', file=sys.stderr, flush=True)",
            "    raise SystemExit(7)",
        ]
    )
    base_command = [sys.executable, "-c", command_code]
    _launch(
        repo=repo,
        run_root=run_root,
        run_id="run-resume",
        command=base_command,
    )
    failed = _wait_terminal(run_root, "run-resume")
    assert failed["state"] == "finished"
    assert failed["dag_exit_code"] == 7

    resumed = _launch(
        repo=repo,
        run_root=run_root,
        run_id="run-resume",
        command=[*base_command, "--resume"],
        resume=True,
    )
    assert resumed["attempt_id"] == "resume-1"
    final = _wait_terminal(run_root, "run-resume")
    assert final["state"] == "finished"
    assert final["dag_exit_code"] == 0
    assert json.loads(status_path.read_text(encoding="utf-8"))["state"] == "succeeded"
    attached = attach_job(run_root=run_root, run_id="run-resume")
    assert "resumed-success" in attached["logs"]["dag_stdout"]


def _stale_recovery_fixture(tmp_path: Path, *, rebooted: bool) -> tuple[Path, Path]:
    run_root = tmp_path / "runs"
    repo = tmp_path / "repo"
    run_root.mkdir()
    repo.mkdir()
    run_id = "crash-run"
    run_dir = run_root / run_id
    run_dir.mkdir()
    attempt = run_root / ".ecgcert-remote-jobs" / run_id / "initial"
    attempt.mkdir(parents=True)
    identity = process_identity(os.getpid())
    if rebooted:
        identity = {**identity, "boot_id": "prior-boot-fixture"}
    spec = {
        "schema_version": remote_job.JOB_SPEC_SCHEMA,
        "run_id": run_id,
        "attempt_id": "initial",
        "repo": str(repo.resolve()),
        "run_root": str(run_root.resolve()),
        "run_dir": str(run_dir),
        "command": [sys.executable, "scripts/dag_runner.py"],
        "resume": False,
        "profile": "icassp",
        "resource": None,
        "environment_lock": "gpu",
        "python_executable": str(Path(sys.executable).absolute()),
        "supervisor_entry": str(SUPERVISOR.resolve()),
        "created_at": "2026-07-19T00:00:00Z",
    }
    remote_status = {
        "schema_version": remote_job.JOB_STATUS_SCHEMA,
        "run_id": run_id,
        "attempt_id": "initial",
        "state": "running",
        "supervisor": identity,
        "child": identity,
        "started_at": "2026-07-19T00:00:00Z",
        "finished_at": None,
        "dag_exit_code": None,
        "run_status_sha256": None,
        "run_state": None,
        "run_exit_code": None,
    }
    (attempt / "spec.v1.json").write_text(json.dumps(spec), encoding="utf-8")
    (attempt / "status.v1.json").write_text(json.dumps(remote_status), encoding="utf-8")

    limits = {
        "cpu_core_hours": 20.0,
        "gpu_hours": 10.0,
        "artifact_bytes": 1_000,
    }
    planned = {
        "cpu_core_hours": 4.0,
        "gpu_hours": 3.0,
        "artifact_bytes": 1_000,
    }
    lease = BudgetLease(
        control_root=run_root,
        run_id=run_id,
        limits=limits,
        reserved_gpu_hours=2.0,
    )
    reservation = lease.acquire(planned)
    dag_status = {
        "schema_version": 3,
        "run_id": run_id,
        "control_root": str(run_root),
        "state": "running",
        "attempts": [],
        "active_attempt": {
            "ordinal": 0,
            "attempt_id": "initial",
            "budget_run_id": run_id,
            "resumed": False,
            "resume_from_node": "train",
            "selected_nodes": ["train"],
            "started_at": "2026-07-19T00:00:00Z",
            "planned_timeout_upper_bound": planned,
            "global_reservation_sha256": reservation["event_sha256"],
            "log_dir": "logs",
        },
        "budget": {
            "limits": {
                **limits,
                "reserved_gpu_hours_for_rerun": 2.0,
            }
        },
    }
    (run_dir / "status.json").write_text(json.dumps(dag_status), encoding="utf-8")
    return run_root, attempt


def test_reboot_recovery_audit_authenticates_but_never_mutates_or_settles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root, attempt = _stale_recovery_fixture(tmp_path, rebooted=True)
    ledger = run_root / "budget-ledger.v1.jsonl"
    owner = run_root / ".ecgcert-execution.lease" / "owner.json"
    ledger_before = ledger.read_bytes()
    owner_before = owner.read_bytes()
    tree_before = {
        path.relative_to(run_root).as_posix(): path.read_bytes()
        for path in run_root.rglob("*")
        if path.is_file()
    }
    directories_before = sorted(
        path.relative_to(run_root).as_posix() for path in run_root.rglob("*") if path.is_dir()
    )

    monkeypatch.setattr(remote_job, "_pid_is_possibly_alive", lambda _pid: False)
    report = audit_recovery_job(run_root=run_root, run_id="crash-run")

    assert report["status"] == "authenticated-stale-reservation", report["failed_checks"]
    assert all(report["checks"].values())
    assert report["automatic_recovery_permitted"] is False
    assert report["resume_permitted"] is False
    assert report["ledger_mutated"] is False
    assert report["lease_mutated"] is False
    assert len(report["report_sha256"]) == 64
    assert len(report["stale_evidence_sha256"]) == 64
    assert validate_recovery_audit_report(report)["status"] == report["status"]
    assert report["budget"]["ledger_sha256"] == remote_job._sha256_file(ledger)  # noqa: SLF001
    assert ledger.read_bytes() == ledger_before
    assert owner.read_bytes() == owner_before
    assert not (attempt / "recovery-audit.v1.json").exists()
    assert tree_before == {
        path.relative_to(run_root).as_posix(): path.read_bytes()
        for path in run_root.rglob("*")
        if path.is_file()
    }
    assert directories_before == sorted(
        path.relative_to(run_root).as_posix() for path in run_root.rglob("*") if path.is_dir()
    )

    tampered = json.loads(json.dumps(report))
    tampered["stale_evidence"]["dag_child_alive"] = True
    with pytest.raises(RemoteJobError, match="report hash|stale-evidence"):
        validate_recovery_audit_report(tampered)


def test_recovery_audit_refuses_live_recorded_executor(tmp_path: Path) -> None:
    run_root, _attempt = _stale_recovery_fixture(tmp_path, rebooted=False)

    report = audit_recovery_job(run_root=run_root, run_id="crash-run")

    assert report["status"] == "recovery-preconditions-failed"
    assert report["dag_child_alive"] is True
    assert report["supervisor_alive"] is True
    assert "dag_child_identity_dead" in report["failed_checks"]
    assert report["budget"] is None
    assert report["automatic_recovery_permitted"] is False


def test_liveness_probe_treats_permission_and_proc_uncertainty_as_live(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = {"pid": 12345, "boot_id": "boot", "start_ticks": "42"}

    def denied(_pid: int, _signal: int) -> None:
        raise PermissionError

    monkeypatch.setattr(remote_job.os, "kill", denied)
    assert remote_job._posix_pid_is_possibly_alive(12345) is True  # noqa: SLF001

    monkeypatch.setattr(
        remote_job.os,
        "kill",
        lambda _pid, _signal: (_ for _ in ()).throw(ProcessLookupError()),
    )
    assert remote_job._posix_pid_is_possibly_alive(12345) is False  # noqa: SLF001

    monkeypatch.setattr(
        remote_job.os,
        "kill",
        lambda _pid, _signal: (_ for _ in ()).throw(OSError("unknown probe error")),
    )
    assert remote_job._posix_pid_is_possibly_alive(12345) is True  # noqa: SLF001

    monkeypatch.setattr(remote_job, "_pid_is_possibly_alive", lambda _pid: True)
    monkeypatch.setattr(
        remote_job,
        "process_identity",
        lambda _pid: {
            "pid": 12345,
            "boot_id": "unavailable",
            "start_ticks": "unavailable",
        },
    )
    assert process_identity_is_alive(identity) is True

    monkeypatch.setattr(
        remote_job,
        "process_identity",
        lambda _pid: {"pid": 12345, "boot_id": "other", "start_ticks": "99"},
    )
    assert process_identity_is_alive(identity) is False

    assert remote_job._windows_open_error_is_possibly_alive(5) is True  # noqa: SLF001
    assert remote_job._windows_open_error_is_possibly_alive(87) is False  # noqa: SLF001


@pytest.mark.skipif(os.name != "nt", reason="Windows-specific non-destructive PID probe")
def test_windows_status_probe_never_calls_os_kill(monkeypatch: pytest.MonkeyPatch) -> None:
    identity = process_identity(os.getpid())

    def forbidden_kill(_pid: int, _signal: int) -> None:
        raise AssertionError("Windows liveness probes must never call os.kill")

    monkeypatch.setattr(remote_job.os, "kill", forbidden_kill)
    assert process_identity_is_alive(identity) is True


@pytest.mark.skipif(os.name != "posix", reason="POSIX signal-zero semantics")
def test_posix_current_process_identity_is_live() -> None:
    assert process_identity_is_alive(process_identity(os.getpid())) is True


def test_status_and_attach_bind_the_expected_repository_and_python(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    other_repo = tmp_path / "other-repo"
    run_root = tmp_path / "runs"
    repo.mkdir()
    other_repo.mkdir()
    run_root.mkdir()
    other_python = tmp_path / "other-python"
    other_python.write_bytes(b"not the selected interpreter")
    _launch(
        repo=repo,
        run_root=run_root,
        run_id="route-run",
        command=[sys.executable, "-c", "print('route-bound')"],
    )
    _wait_terminal(run_root, "route-run")

    status = job_status(
        run_root=run_root,
        run_id="route-run",
        expected_repo=repo,
        expected_python=Path(sys.executable),
    )
    assert status["repo"] == str(repo.resolve())
    assert len(status["logical_job_sha256"]) == 64
    attached = attach_job(
        run_root=run_root,
        run_id="route-run",
        expected_repo=repo,
        expected_python=Path(sys.executable),
    )
    assert attached["status"]["logical_job_sha256"] == status["logical_job_sha256"]

    with pytest.raises(RemoteJobError, match="another repository"):
        job_status(
            run_root=run_root,
            run_id="route-run",
            expected_repo=other_repo,
            expected_python=Path(sys.executable),
        )
    with pytest.raises(RemoteJobError, match="another Python environment"):
        attach_job(
            run_root=run_root,
            run_id="route-run",
            expected_repo=repo,
            expected_python=other_python,
        )
