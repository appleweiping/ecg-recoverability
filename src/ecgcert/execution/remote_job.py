"""Durable server-side supervision for reconnectable remote DAG runs.

The SSH channel is only a launch/control transport.  A tiny supervisor is
started in a new operating-system session, and it starts the DAG in another
session.  Closing the SSH client therefore cannot deliver a terminal hangup to
either process.  Job metadata and logs live beside (not inside) the immutable
run workspace so reconnecting clients can authenticate the logical run before
reading status.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any, Iterator, Mapping, Sequence

from .budget import (
    RECOVERY_AUDIT_SCHEMA as BUDGET_RECOVERY_AUDIT_SCHEMA,
    BudgetError,
    audit_unsettled_reservation,
)


JOB_SPEC_SCHEMA = "ecgcert-remote-job-spec/v1"
JOB_STATUS_SCHEMA = "ecgcert-remote-job-status/v1"
JOB_CONTROL_SCHEMA = "ecgcert-remote-job-control/v1"
JOB_ATTACH_SCHEMA = "ecgcert-remote-job-attach/v1"
RECOVERY_AUDIT_SCHEMA = "ecgcert-remote-recovery-audit/v1"
JOB_DIRECTORY = ".ecgcert-remote-jobs"
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_TERMINAL_STATES = frozenset({"finished", "inconsistent", "launch_failed"})
_SPEC_FIELDS = frozenset(
    {
        "schema_version",
        "run_id",
        "attempt_id",
        "repo",
        "run_root",
        "run_dir",
        "command",
        "resume",
        "profile",
        "resource",
        "environment_lock",
        "python_executable",
        "supervisor_entry",
        "created_at",
    }
)
_STATUS_FIELDS = frozenset(
    {
        "schema_version",
        "run_id",
        "attempt_id",
        "state",
        "supervisor",
        "child",
        "started_at",
        "finished_at",
        "dag_exit_code",
        "run_status_sha256",
        "run_state",
        "run_exit_code",
    }
)


class RemoteJobError(RuntimeError):
    """Raised when a durable remote launch cannot be authenticated."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_artifact_bytes(path: Path) -> bytes:
    deadline = time.monotonic() + 5.0
    while True:
        try:
            return path.read_bytes()
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.02)


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(_read_artifact_bytes(path))


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    rendered = _canonical_bytes(value)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    temporary.write_bytes(rendered)
    deadline = time.monotonic() + 5.0
    while True:
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            # Windows can briefly deny replacement while a reconnecting status
            # reader or virus scanner has the old file open. POSIX never needs
            # this path, but the retry keeps the control format portable.
            if time.monotonic() >= deadline:
                temporary.unlink(missing_ok=True)
                raise
            time.sleep(0.02)


def _read_json_with_sha256(path: Path) -> tuple[dict[str, Any], str]:
    try:
        raw = _read_artifact_bytes(path)
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RemoteJobError(f"cannot read remote job artifact: {path}") from exc
    if not isinstance(value, dict):
        raise RemoteJobError(f"remote job artifact is not an object: {path}")
    return value, _sha256_bytes(raw)


def _read_json(path: Path) -> dict[str, Any]:
    value, _digest = _read_json_with_sha256(path)
    return value


def _require_safe_run_id(run_id: str) -> str:
    if not isinstance(run_id, str) or not _SAFE_ID.fullmatch(run_id):
        raise RemoteJobError("run_id is not a safe identifier")
    return run_id


def _require_absolute_directory(path: Path | str, *, label: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        raise RemoteJobError(f"{label} must be absolute")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise RemoteJobError(f"{label} does not exist") from exc
    if not resolved.is_dir():
        raise RemoteJobError(f"{label} is not a directory")
    return resolved


def _require_absolute_file(path: Path | str, *, label: str, preserve_symlink: bool = False) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        raise RemoteJobError(f"{label} must be absolute")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise RemoteJobError(f"{label} does not exist") from exc
    if not resolved.is_file():
        raise RemoteJobError(f"{label} is not a file")
    # Re-executing the resolved target of venv/bin/python loses the virtual
    # environment because Python discovers pyvenv.cfg from argv[0]. Keep the
    # explicitly selected interpreter path while still verifying its target.
    return candidate.absolute() if preserve_symlink else resolved


def _job_root(run_root: Path, run_id: str) -> Path:
    control_directory = run_root / JOB_DIRECTORY
    root = control_directory / _require_safe_run_id(run_id)
    if control_directory.is_symlink() or root.is_symlink():
        raise RemoteJobError("remote job control directories must not be symbolic links")
    return root


@contextmanager
def _exclusive_file_lock(path: Path) -> Iterator[None]:
    """Hold an OS-released lock, so a killed launcher cannot leave it stale."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise RemoteJobError("remote job launch lock must not be a symbolic link")
    with path.open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "posix":
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        else:  # pragma: no cover - exercised by the Windows test host only
            import msvcrt

            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def _linux_process_start_ticks(pid: int) -> str:
    stat_path = Path("/proc") / str(pid) / "stat"
    try:
        raw = stat_path.read_text(encoding="utf-8")
        remainder = raw[raw.rfind(")") + 2 :].split()
        return remainder[19]
    except (OSError, IndexError):
        return "unavailable"


def _boot_id() -> str:
    try:
        value = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()
    except OSError:
        return "unavailable"
    return value or "unavailable"


def process_identity(pid: int) -> dict[str, Any]:
    return {
        "pid": int(pid),
        "boot_id": _boot_id(),
        "start_ticks": _linux_process_start_ticks(int(pid)),
    }


def _posix_pid_is_possibly_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as exc:
        if exc.errno == errno.ESRCH:
            return False
        return True
    return True


def _windows_pid_is_possibly_alive(pid: int) -> bool:
    """Probe a Windows PID without ``os.kill(pid, 0)``.

    CPython implements non-console-event ``os.kill`` calls on Windows with
    ``TerminateProcess``; signal zero is therefore not a portable existence
    probe.  Querying the process handle and exit code is read-only.
    """

    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    open_process = kernel32.OpenProcess
    open_process.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    open_process.restype = wintypes.HANDLE
    get_exit_code = kernel32.GetExitCodeProcess
    get_exit_code.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    get_exit_code.restype = wintypes.BOOL
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = [wintypes.HANDLE]
    close_handle.restype = wintypes.BOOL

    process_query_limited_information = 0x1000
    error_invalid_parameter = 87
    still_active = 259
    handle = open_process(process_query_limited_information, False, pid)
    if not handle:
        return _windows_open_error_is_possibly_alive(
            ctypes.get_last_error(), invalid_parameter=error_invalid_parameter
        )
    try:
        exit_code = wintypes.DWORD()
        if not get_exit_code(handle, ctypes.byref(exit_code)):
            return True
        return exit_code.value == still_active
    finally:
        close_handle(handle)


def _windows_open_error_is_possibly_alive(error_code: int, *, invalid_parameter: int = 87) -> bool:
    """Only an invalid PID proves absence; access/inspection errors stay live."""

    return error_code != invalid_parameter


def _pid_is_possibly_alive(pid: int) -> bool:
    if os.name == "nt":
        return _windows_pid_is_possibly_alive(pid)
    return _posix_pid_is_possibly_alive(pid)


def process_identity_is_alive(identity: Mapping[str, Any] | None) -> bool:
    """Return ``False`` only when the recorded process is provably absent.

    Recovery safety is asymmetric: permission failures or temporarily
    unavailable ``/proc`` identity fields must be treated as possibly live.
    A false negative could label an active executor stale, whereas a false
    positive merely defers manual recovery.
    """

    if not isinstance(identity, Mapping):
        return False
    pid = identity.get("pid")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return False
    if not _pid_is_possibly_alive(pid):
        return False
    observed = process_identity(pid)
    expected_boot = identity.get("boot_id")
    expected_ticks = identity.get("start_ticks")
    if (
        expected_boot not in {None, "unavailable"}
        and observed["boot_id"] != "unavailable"
        and observed["boot_id"] != expected_boot
    ):
        return False
    if (
        expected_ticks not in {None, "unavailable"}
        and observed["start_ticks"] != "unavailable"
        and observed["start_ticks"] != expected_ticks
    ):
        return False
    return True


def _attempt_ordinal(name: str) -> int | None:
    if name == "initial":
        return 0
    match = re.fullmatch(r"resume-([1-9][0-9]*)", name)
    return int(match.group(1)) if match else None


def _attempts(job_root: Path) -> list[Path]:
    if not job_root.exists():
        return []
    if job_root.is_symlink() or not job_root.is_dir():
        raise RemoteJobError("remote job root is not a canonical directory")
    values: list[tuple[int, Path]] = []
    for path in job_root.iterdir():
        ordinal = _attempt_ordinal(path.name)
        if ordinal is None:
            continue
        if path.is_symlink() or not path.is_dir():
            raise RemoteJobError("remote job attempt is not a canonical directory")
        values.append((ordinal, path))
    return [path for _ordinal, path in sorted(values, key=lambda value: value[0])]


def _detached_process_options() -> dict[str, Any]:
    if os.name == "posix":
        return {"start_new_session": True}
    creationflags = int(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
    creationflags |= int(getattr(subprocess, "DETACHED_PROCESS", 0))
    return {"creationflags": creationflags}


def _status_path(attempt_dir: Path) -> Path:
    return attempt_dir / "status.v1.json"


def _spec_path(attempt_dir: Path) -> Path:
    return attempt_dir / "spec.v1.json"


def _read_dag_status(run_dir: Path) -> tuple[dict[str, Any] | None, str | None]:
    path = run_dir / "status.json"
    if path.is_symlink():
        raise RemoteJobError("DAG status must not be a symbolic link")
    if not path.is_file():
        return None, None
    return _read_json_with_sha256(path)


def _spec_identity(spec: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: spec.get(key)
        for key in (
            "schema_version",
            "run_id",
            "attempt_id",
            "repo",
            "run_root",
            "run_dir",
            "command",
            "resume",
            "profile",
            "resource",
            "environment_lock",
            "python_executable",
            "supervisor_entry",
        )
    }


def _logical_spec_identity(spec: Mapping[str, Any]) -> dict[str, Any]:
    command = spec.get("command")
    normalized_command = list(command) if isinstance(command, list) else command
    if isinstance(normalized_command, list) and normalized_command[-1:] == ["--resume"]:
        normalized_command = normalized_command[:-1]
    return {
        key: spec.get(key)
        for key in (
            "run_id",
            "repo",
            "run_root",
            "run_dir",
            "profile",
            "resource",
            "environment_lock",
            "python_executable",
            "supervisor_entry",
        )
    } | {"command": normalized_command}


def _validate_attempt_spec(spec: Mapping[str, Any], attempt_dir: Path) -> None:
    if set(spec) != _SPEC_FIELDS or spec.get("schema_version") != JOB_SPEC_SCHEMA:
        raise RemoteJobError("remote job spec has unknown or missing fields")
    if attempt_dir.parent.parent.name != JOB_DIRECTORY:
        raise RemoteJobError("remote job attempt is outside the canonical control directory")
    persistent_root = attempt_dir.parents[2].resolve()
    run_id = attempt_dir.parent.name
    ordinal = _attempt_ordinal(attempt_dir.name)
    command = spec.get("command")
    if (
        not _SAFE_ID.fullmatch(run_id)
        or ordinal is None
        or spec.get("run_id") != run_id
        or spec.get("attempt_id") != attempt_dir.name
        or spec.get("run_root") != str(persistent_root)
        or spec.get("run_dir") != str(persistent_root / run_id)
        or not isinstance(spec.get("resume"), bool)
        or spec.get("resume") != (ordinal != 0)
        or spec.get("profile") not in {"icassp", "extended", "legacy"}
        or spec.get("resource") not in {None, "cpu", "gpu", "paper"}
        or spec.get("environment_lock") not in {"cpu", "gpu"}
        or not isinstance(command, list)
        or not command
        or not all(isinstance(value, str) and value for value in command)
        or (command[-1:] == ["--resume"]) != spec.get("resume")
        or not isinstance(spec.get("created_at"), str)
        or not spec["created_at"]
    ):
        raise RemoteJobError("remote job spec identity is invalid")
    for field in ("repo", "python_executable", "supervisor_entry"):
        value = spec.get(field)
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise RemoteJobError(f"remote job spec {field} is not an absolute path")


def _attempt_status(
    attempt_dir: Path,
    *,
    expected_repo: Path | str | None = None,
    expected_python: Path | str | None = None,
) -> dict[str, Any]:
    spec_path = _spec_path(attempt_dir)
    if spec_path.is_symlink() or not spec_path.is_file():
        raise RemoteJobError("remote job spec is missing or not a regular file")
    spec, spec_sha256 = _read_json_with_sha256(spec_path)
    _validate_attempt_spec(spec, attempt_dir)
    if expected_repo is not None:
        repository = _require_absolute_directory(expected_repo, label="expected repository")
        if spec.get("repo") != str(repository):
            raise RemoteJobError("remote job belongs to another repository")
    if expected_python is not None:
        interpreter = _require_absolute_file(
            expected_python,
            label="expected Python executable",
            preserve_symlink=True,
        )
        if spec.get("python_executable") != str(interpreter):
            raise RemoteJobError("remote job belongs to another Python environment")
    status_path = _status_path(attempt_dir)
    if status_path.is_symlink():
        raise RemoteJobError("remote job status must not be a symbolic link")
    if status_path.is_file():
        status, status_sha256 = _read_json_with_sha256(status_path)
        if (
            frozenset(status) not in {_STATUS_FIELDS, _STATUS_FIELDS | {"error_type"}}
            or status.get("schema_version") != JOB_STATUS_SCHEMA
        ):
            raise RemoteJobError("remote job status schema mismatch")
    else:
        status_sha256 = None
        spawn_path = attempt_dir / "spawn.v1.json"
        if spawn_path.is_symlink():
            raise RemoteJobError("remote job spawn record must not be a symbolic link")
        spawn = _read_json(spawn_path) if spawn_path.is_file() else {}
        if spawn and (
            set(spawn) != {"schema_version", "supervisor", "spawned_at"}
            or spawn.get("schema_version") != "ecgcert-remote-job-spawn/v1"
            or not _valid_process_identity(spawn.get("supervisor"))
            or not isinstance(spawn.get("spawned_at"), str)
            or not spawn["spawned_at"]
        ):
            raise RemoteJobError("remote job spawn record is invalid")
        status = {
            "schema_version": JOB_STATUS_SCHEMA,
            "run_id": spec["run_id"],
            "attempt_id": spec["attempt_id"],
            "state": "starting",
            "supervisor": spawn.get("supervisor"),
            "child": None,
            "started_at": spec["created_at"],
            "finished_at": None,
            "dag_exit_code": None,
            "run_status_sha256": None,
            "run_state": None,
            "run_exit_code": None,
        }
    if status.get("run_id") != spec.get("run_id") or status.get("attempt_id") != spec.get(
        "attempt_id"
    ):
        raise RemoteJobError("remote job status/spec identity mismatch")
    state = status.get("state")
    if state in _TERMINAL_STATES:
        supervisor_alive = False
        child_alive = False
    else:
        supervisor_alive = process_identity_is_alive(status.get("supervisor"))
        child_alive = process_identity_is_alive(status.get("child"))
    effective_state = state
    if state not in _TERMINAL_STATES and not supervisor_alive and not child_alive:
        effective_state = "orphaned"
    return {
        "schema_version": JOB_CONTROL_SCHEMA,
        "run_id": spec["run_id"],
        "attempt_id": spec["attempt_id"],
        "repo": spec["repo"],
        "run_root": spec["run_root"],
        "run_dir": spec["run_dir"],
        "profile": spec["profile"],
        "resource": spec["resource"],
        "environment_lock": spec["environment_lock"],
        "python_executable": spec["python_executable"],
        "logical_job_sha256": _sha256_bytes(_canonical_bytes(_logical_spec_identity(spec))),
        "state": effective_state,
        "recorded_state": state,
        "resume": spec["resume"],
        "supervisor_alive": supervisor_alive,
        "child_alive": child_alive,
        "dag_exit_code": status.get("dag_exit_code"),
        "run_state": status.get("run_state"),
        "run_exit_code": status.get("run_exit_code"),
        "started_at": status.get("started_at"),
        "finished_at": status.get("finished_at"),
        "spec_sha256": spec_sha256,
        "status_sha256": status_sha256,
        "run_status_sha256": status.get("run_status_sha256"),
        "attempt_dir": str(attempt_dir),
    }


def job_status(
    *,
    run_root: Path | str,
    run_id: str,
    expected_repo: Path | str | None = None,
    expected_python: Path | str | None = None,
) -> dict[str, Any]:
    persistent_root = _require_absolute_directory(run_root, label="run root")
    root = _job_root(persistent_root, run_id)
    attempts = _attempts(root)
    if not attempts:
        raise RemoteJobError("no durable remote launch exists for this run")
    return _attempt_status(
        attempts[-1],
        expected_repo=expected_repo,
        expected_python=expected_python,
    )


def _valid_process_identity(value: Any) -> bool:
    if not isinstance(value, Mapping) or set(value) != {
        "pid",
        "boot_id",
        "start_ticks",
    }:
        return False
    pid = value.get("pid")
    return (
        isinstance(pid, int)
        and not isinstance(pid, bool)
        and pid > 0
        and isinstance(value.get("boot_id"), str)
        and bool(value["boot_id"])
        and isinstance(value.get("start_ticks"), str)
        and bool(value["start_ticks"])
    )


def validate_recovery_audit_report(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the self-contained, non-authorizing stale-evidence report."""

    expected_fields = {
        "schema_version",
        "audited_at",
        "run_id",
        "attempt_id",
        "status",
        "checks",
        "failed_checks",
        "repo",
        "run_root",
        "run_dir",
        "python_executable",
        "logical_job_sha256",
        "remote_spec_sha256",
        "remote_status_sha256",
        "dag_status_sha256",
        "dag_recorded_state",
        "supervisor_alive",
        "dag_child_alive",
        "budget",
        "budget_error_code",
        "stale_evidence",
        "stale_evidence_sha256",
        "automatic_recovery_permitted",
        "resume_permitted",
        "ledger_mutated",
        "lease_mutated",
        "blocker",
        "report_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise RemoteJobError("recovery audit report has unknown or missing fields")
    report = dict(value)
    report_sha256 = report.pop("report_sha256")
    if (
        not isinstance(report_sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", report_sha256)
        or report_sha256 != _sha256_bytes(_canonical_bytes(report))
    ):
        raise RemoteJobError("recovery audit report hash is invalid")
    checks = value.get("checks")
    if (
        not isinstance(checks, Mapping)
        or not checks
        or not all(
            isinstance(name, str) and isinstance(passed, bool) for name, passed in checks.items()
        )
    ):
        raise RemoteJobError("recovery audit checks are invalid")
    failed_checks = sorted(name for name, passed in checks.items() if not passed)
    authenticated = not failed_checks
    if value.get("failed_checks") != failed_checks or value.get("status") != (
        "authenticated-stale-reservation" if authenticated else "recovery-preconditions-failed"
    ):
        raise RemoteJobError("recovery audit outcome is inconsistent with its checks")
    if any(
        value.get(field) is not False
        for field in (
            "automatic_recovery_permitted",
            "resume_permitted",
            "ledger_mutated",
            "lease_mutated",
        )
    ):
        raise RemoteJobError("recovery audit must not authorize or claim a mutation")
    evidence = value.get("stale_evidence")
    evidence_fields = {
        "remote_spec_sha256",
        "remote_status_sha256",
        "dag_status_sha256",
        "recorded_supervisor",
        "recorded_dag_child",
        "observed_boot_id",
        "supervisor_alive",
        "dag_child_alive",
        "budget",
    }
    if not isinstance(evidence, Mapping) or set(evidence) != evidence_fields:
        raise RemoteJobError("recovery audit stale evidence is invalid")
    evidence_sha256 = value.get("stale_evidence_sha256")
    if (
        not isinstance(evidence_sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", evidence_sha256)
        or evidence_sha256 != _sha256_bytes(_canonical_bytes(evidence))
    ):
        raise RemoteJobError("recovery audit stale-evidence hash is invalid")
    for name in ("remote_spec_sha256", "remote_status_sha256", "dag_status_sha256"):
        digest = value.get(name)
        if (
            not isinstance(digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
            or evidence.get(name) != digest
        ):
            raise RemoteJobError("recovery audit artifact hash binding is invalid")
    if (
        evidence.get("supervisor_alive") is not value.get("supervisor_alive")
        or evidence.get("dag_child_alive") is not value.get("dag_child_alive")
        or evidence.get("budget") != value.get("budget")
    ):
        raise RemoteJobError("recovery audit evidence does not bind the report")
    budget = value.get("budget")
    if budget is not None and (
        not isinstance(budget, Mapping)
        or budget.get("schema_version") != BUDGET_RECOVERY_AUDIT_SCHEMA
    ):
        raise RemoteJobError("recovery audit budget evidence is invalid")
    if authenticated and (
        value.get("supervisor_alive") is not False
        or value.get("dag_child_alive") is not False
        or budget is None
        or value.get("budget_error_code") is not None
    ):
        raise RemoteJobError("authenticated stale evidence is incomplete")
    return {
        "status": value["status"],
        "report_sha256": report_sha256,
        "stale_evidence_sha256": evidence_sha256,
    }


def audit_recovery_job(
    *,
    run_root: Path | str,
    run_id: str,
    expected_repo: Path | str | None = None,
    expected_python: Path | str | None = None,
) -> dict[str, Any]:
    """Audit an interrupted remote attempt without changing lease or ledger.

    The report proves as much as the retained state permits: exact remote/DAG/
    lease identity, dead recorded executor identities, and a valid unfinished
    reservation.  It always refuses automatic settlement because the runner
    has no durable meter for the active node between its last checkpoint and
    an abrupt process or host death.
    """

    persistent_root = _require_absolute_directory(run_root, label="run root")
    logical_run_id = _require_safe_run_id(run_id)
    root = _job_root(persistent_root, logical_run_id)
    attempts = _attempts(root)
    if not attempts:
        raise RemoteJobError("no durable remote launch exists for this run")
    attempt = attempts[-1]
    spec_path = _spec_path(attempt)
    status_path = _status_path(attempt)
    if spec_path.is_symlink() or not spec_path.is_file():
        raise RemoteJobError("recovery audit requires a regular remote job spec")
    if status_path.is_symlink() or not status_path.is_file():
        raise RemoteJobError("recovery audit requires a recorded remote attempt status")
    spec, remote_spec_sha256 = _read_json_with_sha256(spec_path)
    remote_status, remote_status_sha256 = _read_json_with_sha256(status_path)
    if expected_repo is not None:
        repository = _require_absolute_directory(expected_repo, label="expected repository")
        if spec.get("repo") != str(repository):
            raise RemoteJobError("remote recovery record belongs to another repository")
    if expected_python is not None:
        interpreter = _require_absolute_file(
            expected_python,
            label="expected Python executable",
            preserve_symlink=True,
        )
        if spec.get("python_executable") != str(interpreter):
            raise RemoteJobError("remote recovery record belongs to another Python environment")
    run_dir = persistent_root / logical_run_id
    dag_status, dag_status_sha256 = _read_dag_status(run_dir)
    if dag_status is None or dag_status_sha256 is None:
        raise RemoteJobError("recovery audit requires a recorded DAG status")

    checks: dict[str, bool] = {}
    checks["remote_spec_schema"] = (
        set(spec) == _SPEC_FIELDS and spec.get("schema_version") == JOB_SPEC_SCHEMA
    )
    checks["remote_status_schema"] = (
        frozenset(remote_status) in {_STATUS_FIELDS, _STATUS_FIELDS | {"error_type"}}
        and remote_status.get("schema_version") == JOB_STATUS_SCHEMA
    )
    checks["dag_status_schema"] = dag_status.get("schema_version") == 3
    checks["logical_run_identity"] = (
        spec.get("run_id") == logical_run_id
        and remote_status.get("run_id") == logical_run_id
        and dag_status.get("run_id") == logical_run_id
        and spec.get("run_root") == str(persistent_root)
        and spec.get("run_dir") == str(run_dir)
        and dag_status.get("control_root") == str(persistent_root)
    )
    checks["remote_execution_paths"] = all(
        isinstance(spec.get(field), str) and Path(spec[field]).is_absolute()
        for field in ("repo", "python_executable", "supervisor_entry")
    )
    attempt_id = spec.get("attempt_id")
    checks["remote_attempt_identity"] = (
        isinstance(attempt_id, str)
        and attempt.name == attempt_id
        and remote_status.get("attempt_id") == attempt_id
    )

    supervisor = remote_status.get("supervisor")
    child = remote_status.get("child")
    checks["recorded_process_identities"] = _valid_process_identity(
        supervisor
    ) and _valid_process_identity(child)
    supervisor_alive = (
        process_identity_is_alive(supervisor) if _valid_process_identity(supervisor) else True
    )
    child_alive = process_identity_is_alive(child) if _valid_process_identity(child) else True
    checks["supervisor_identity_dead"] = not supervisor_alive
    checks["dag_child_identity_dead"] = not child_alive
    bound_run_status = remote_status.get("run_status_sha256")
    checks["remote_dag_status_binding"] = (
        bound_run_status == dag_status_sha256
        if isinstance(bound_run_status, str)
        else bound_run_status is None
        and remote_status.get("state") in {"starting", "running", "launch_failed"}
    )

    active_attempt = dag_status.get("active_attempt")
    expected_active_fields = {
        "ordinal",
        "attempt_id",
        "budget_run_id",
        "resumed",
        "resume_from_node",
        "selected_nodes",
        "started_at",
        "planned_timeout_upper_bound",
        "global_reservation_sha256",
        "log_dir",
    }
    ordinal = _attempt_ordinal(attempt.name)
    expected_budget_run_id = (
        logical_run_id
        if ordinal == 0
        else f"{logical_run_id}.resume-{ordinal}"
        if ordinal is not None
        else None
    )
    checks["dag_active_attempt_identity"] = (
        isinstance(active_attempt, dict)
        and set(active_attempt) == expected_active_fields
        and active_attempt.get("ordinal") == ordinal
        and active_attempt.get("attempt_id") == attempt_id
        and active_attempt.get("budget_run_id") == expected_budget_run_id
        and active_attempt.get("resumed") is (ordinal != 0)
        and isinstance(active_attempt.get("selected_nodes"), list)
        and bool(active_attempt["selected_nodes"])
        and active_attempt.get("resume_from_node") == active_attempt["selected_nodes"][0]
    )
    checks["dag_interruption_state"] = dag_status.get("state") in {
        "staging",
        "running",
        "failed",
    }
    checks["attempt_history_ordinal"] = isinstance(
        dag_status.get("attempts"), list
    ) and ordinal == len(dag_status["attempts"])

    budget = dag_status.get("budget")
    budget_limits = budget.get("limits") if isinstance(budget, dict) else None
    limits = None
    reserved_gpu_hours = None
    if isinstance(budget_limits, dict) and set(budget_limits) == {
        "cpu_core_hours",
        "gpu_hours",
        "artifact_bytes",
        "reserved_gpu_hours_for_rerun",
    }:
        limits = {
            key: budget_limits[key] for key in ("cpu_core_hours", "gpu_hours", "artifact_bytes")
        }
        reserved_gpu_hours = budget_limits["reserved_gpu_hours_for_rerun"]
    checks["frozen_budget_contract"] = limits is not None and reserved_gpu_hours is not None

    budget_audit: dict[str, Any] | None = None
    budget_error_code: str | None = None
    if (
        checks["dag_active_attempt_identity"]
        and checks["recorded_process_identities"]
        and checks["supervisor_identity_dead"]
        and checks["dag_child_identity_dead"]
        and checks["frozen_budget_contract"]
        and isinstance(expected_budget_run_id, str)
    ):
        try:
            budget_audit = audit_unsettled_reservation(
                control_root=persistent_root,
                expected_run_id=expected_budget_run_id,
                expected_reservation_sha256=active_attempt["global_reservation_sha256"],
                expected_owner_pid=child["pid"],
                expected_planned=active_attempt["planned_timeout_upper_bound"],
                limits=limits,
                reserved_gpu_hours=reserved_gpu_hours,
            )
        except (BudgetError, OSError, TypeError, ValueError) as exc:
            budget_error_code = type(exc).__name__
    checks["lease_and_unfinished_reservation"] = budget_audit is not None

    authenticated_interruption = all(checks.values())
    stale_evidence: dict[str, Any] = {
        "remote_spec_sha256": remote_spec_sha256,
        "remote_status_sha256": remote_status_sha256,
        "dag_status_sha256": dag_status_sha256,
        "recorded_supervisor": supervisor,
        "recorded_dag_child": child,
        "observed_boot_id": _boot_id(),
        "supervisor_alive": supervisor_alive,
        "dag_child_alive": child_alive,
        "budget": budget_audit,
    }
    stale_evidence_sha256 = _sha256_bytes(_canonical_bytes(stale_evidence))
    report: dict[str, Any] = {
        "schema_version": RECOVERY_AUDIT_SCHEMA,
        "audited_at": _utc_now(),
        "run_id": logical_run_id,
        "attempt_id": attempt.name,
        "status": "authenticated-stale-reservation"
        if authenticated_interruption
        else "recovery-preconditions-failed",
        "checks": checks,
        "failed_checks": sorted(name for name, passed in checks.items() if not passed),
        "repo": spec.get("repo"),
        "run_root": str(persistent_root),
        "run_dir": str(run_dir),
        "python_executable": spec.get("python_executable"),
        "logical_job_sha256": _sha256_bytes(_canonical_bytes(_logical_spec_identity(spec))),
        "remote_spec_sha256": remote_spec_sha256,
        "remote_status_sha256": remote_status_sha256,
        "dag_status_sha256": dag_status_sha256,
        "dag_recorded_state": dag_status.get("state"),
        "supervisor_alive": supervisor_alive,
        "dag_child_alive": child_alive,
        "budget": budget_audit,
        "budget_error_code": budget_error_code,
        "stale_evidence": stale_evidence,
        "stale_evidence_sha256": stale_evidence_sha256,
        "automatic_recovery_permitted": False,
        "resume_permitted": False,
        "ledger_mutated": False,
        "lease_mutated": False,
        "blocker": (
            "active-node CPU/GPU usage is not durably metered across abrupt termination; "
            "settlement would require guessing or substituting a bound for actual usage"
        ),
    }
    report["report_sha256"] = _sha256_bytes(_canonical_bytes(report))
    validate_recovery_audit_report(report)
    return report


def launch_detached_job(
    *,
    repo: Path | str,
    run_root: Path | str,
    run_id: str,
    command: Sequence[str],
    resume: bool,
    profile: str,
    resource: str | None,
    environment_lock: str,
    python_executable: Path | str,
    supervisor_entry: Path | str,
    startup_wait_seconds: float = 3.0,
) -> dict[str, Any]:
    """Launch one idempotent detached attempt and return reconnectable status."""

    repository = _require_absolute_directory(repo, label="repository")
    persistent_root = _require_absolute_directory(run_root, label="run root")
    interpreter = _require_absolute_file(
        python_executable,
        label="Python executable",
        preserve_symlink=True,
    )
    supervisor = _require_absolute_file(supervisor_entry, label="supervisor entry")
    _require_safe_run_id(run_id)
    if persistent_root == repository or repository in persistent_root.parents:
        raise RemoteJobError("run root must be outside the source repository")
    if not isinstance(resume, bool):
        raise RemoteJobError("durable resume flag must be boolean")
    if profile not in {"icassp", "extended", "legacy"}:
        raise RemoteJobError("remote job profile is invalid")
    if resource not in {None, "cpu", "gpu", "paper"}:
        raise RemoteJobError("remote job resource is invalid")
    if environment_lock not in {"cpu", "gpu"}:
        raise RemoteJobError("remote job environment lock is invalid")
    if not command or not all(isinstance(value, str) and value for value in command):
        raise RemoteJobError("DAG command must be a non-empty argv sequence")
    if (list(command)[-1:] == ["--resume"]) != resume:
        raise RemoteJobError("DAG command and durable resume flag do not match")
    root = _job_root(persistent_root, run_id)
    root.mkdir(parents=True, exist_ok=True)
    run_dir = persistent_root / run_id
    requested_logical_identity = _logical_spec_identity(
        {
            "run_id": run_id,
            "repo": str(repository),
            "run_root": str(persistent_root),
            "run_dir": str(run_dir),
            "command": list(command),
            "profile": profile,
            "resource": resource,
            "environment_lock": environment_lock,
            "python_executable": str(interpreter),
            "supervisor_entry": str(supervisor),
        }
    )
    with _exclusive_file_lock(root / "launch.lock"):
        attempts = _attempts(root)
        if attempts:
            latest_spec = _read_json(_spec_path(attempts[-1]))
            _validate_attempt_spec(latest_spec, attempts[-1])
            if _logical_spec_identity(latest_spec) != requested_logical_identity:
                raise RemoteJobError(
                    "existing durable launch conflicts with this logical job identity"
                )
            latest = _attempt_status(attempts[-1])
            if latest["state"] not in _TERMINAL_STATES | {"orphaned"}:
                return latest
        if resume:
            if not attempts:
                raise RemoteJobError("--resume requires an existing durable remote launch")
            dag_status, _digest = _read_dag_status(run_dir)
            dag_attempts = dag_status.get("attempts") if dag_status is not None else None
            if (
                dag_status is None
                or dag_status.get("state") != "failed"
                or dag_status.get("active_attempt") is not None
                or not isinstance(dag_attempts, list)
                or not dag_attempts
            ):
                raise RemoteJobError("--resume requires an existing failed DAG run")
            ordinal = len(dag_attempts)
            attempt_id = f"resume-{ordinal}"
            if (root / attempt_id).exists():
                raise RemoteJobError(
                    "durable resume history is not aligned with the DAG attempt history"
                )
        else:
            attempt_id = "initial"
            existing_initial = root / attempt_id
            if existing_initial.exists():
                prior = _attempt_status(existing_initial)
                existing_spec = _read_json(_spec_path(existing_initial))
                expected_identity = {
                    "schema_version": JOB_SPEC_SCHEMA,
                    "run_id": run_id,
                    "attempt_id": attempt_id,
                    "repo": str(repository),
                    "run_root": str(persistent_root),
                    "run_dir": str(run_dir),
                    "command": list(command),
                    "resume": False,
                    "profile": profile,
                    "resource": resource,
                    "environment_lock": environment_lock,
                    "python_executable": str(interpreter),
                    "supervisor_entry": str(supervisor),
                }
                if _spec_identity(existing_spec) != expected_identity:
                    raise RemoteJobError("existing initial launch conflicts with this command")
                return prior
            if run_dir.exists():
                raise RemoteJobError("new launch refuses an existing DAG run directory")
        attempt_dir = root / attempt_id
        attempt_dir.mkdir(parents=False, exist_ok=False)
        spec = {
            "schema_version": JOB_SPEC_SCHEMA,
            "run_id": run_id,
            "attempt_id": attempt_id,
            "repo": str(repository),
            "run_root": str(persistent_root),
            "run_dir": str(run_dir),
            "command": list(command),
            "resume": bool(resume),
            "profile": profile,
            "resource": resource,
            "environment_lock": environment_lock,
            "python_executable": str(interpreter),
            "supervisor_entry": str(supervisor),
            "created_at": _utc_now(),
        }
        _atomic_json(_spec_path(attempt_dir), spec)
        supervisor_stdout = attempt_dir / "supervisor.stdout.log"
        supervisor_stderr = attempt_dir / "supervisor.stderr.log"
        try:
            with supervisor_stdout.open("ab") as stdout, supervisor_stderr.open("ab") as stderr:
                process = subprocess.Popen(
                    [
                        str(interpreter),
                        str(supervisor),
                        "_supervise",
                        "--attempt-dir",
                        str(attempt_dir),
                    ],
                    cwd=repository,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout,
                    stderr=stderr,
                    close_fds=True,
                    **_detached_process_options(),
                )
            _atomic_json(
                attempt_dir / "spawn.v1.json",
                {
                    "schema_version": "ecgcert-remote-job-spawn/v1",
                    "supervisor": process_identity(process.pid),
                    "spawned_at": _utc_now(),
                },
            )
        except (OSError, ValueError) as exc:
            _atomic_json(
                _status_path(attempt_dir),
                {
                    "schema_version": JOB_STATUS_SCHEMA,
                    "run_id": run_id,
                    "attempt_id": attempt_id,
                    "state": "launch_failed",
                    "supervisor": None,
                    "child": None,
                    "started_at": spec["created_at"],
                    "finished_at": _utc_now(),
                    "dag_exit_code": None,
                    "run_status_sha256": None,
                    "run_state": None,
                    "run_exit_code": None,
                    "error_type": type(exc).__name__,
                },
            )
            raise RemoteJobError("detached supervisor could not be started") from exc

    deadline = time.monotonic() + max(0.0, startup_wait_seconds)
    while time.monotonic() < deadline and not _status_path(attempt_dir).is_file():
        time.sleep(0.05)
    return _attempt_status(attempt_dir)


def supervise_attempt(attempt_dir: Path | str) -> int:
    """Run inside the detached supervisor process until the DAG exits."""

    attempt = _require_absolute_directory(attempt_dir, label="attempt directory")
    spec = _read_json(_spec_path(attempt))
    _validate_attempt_spec(spec, attempt)
    repository = _require_absolute_directory(spec["repo"], label="repository")
    run_dir = Path(spec["run_dir"])
    status: dict[str, Any] = {
        "schema_version": JOB_STATUS_SCHEMA,
        "run_id": spec["run_id"],
        "attempt_id": spec["attempt_id"],
        "state": "starting",
        "supervisor": process_identity(os.getpid()),
        "child": None,
        "started_at": _utc_now(),
        "finished_at": None,
        "dag_exit_code": None,
        "run_status_sha256": None,
        "run_state": None,
        "run_exit_code": None,
    }
    _atomic_json(_status_path(attempt), status)
    dag_stdout = attempt / "dag.stdout.log"
    dag_stderr = attempt / "dag.stderr.log"
    try:
        with dag_stdout.open("ab") as stdout, dag_stderr.open("ab") as stderr:
            child = subprocess.Popen(
                list(spec["command"]),
                cwd=repository,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                close_fds=True,
                **_detached_process_options(),
            )
            status.update(
                {
                    "state": "running",
                    "child": process_identity(child.pid),
                }
            )
            _atomic_json(_status_path(attempt), status)
            exit_code = int(child.wait())
    except (OSError, ValueError) as exc:
        status.update(
            {
                "state": "launch_failed",
                "finished_at": _utc_now(),
                "error_type": type(exc).__name__,
            }
        )
        _atomic_json(_status_path(attempt), status)
        return 70
    dag_status, dag_status_sha256 = _read_dag_status(run_dir)
    run_state = dag_status.get("state") if dag_status is not None else None
    run_exit_code = dag_status.get("exit_code") if dag_status is not None else None
    consistent = dag_status is None or (
        isinstance(run_exit_code, int)
        and not isinstance(run_exit_code, bool)
        and run_exit_code == exit_code
    )
    status.update(
        {
            "state": "finished" if consistent else "inconsistent",
            "finished_at": _utc_now(),
            "dag_exit_code": exit_code,
            "run_status_sha256": dag_status_sha256,
            "run_state": run_state,
            "run_exit_code": run_exit_code,
        }
    )
    _atomic_json(_status_path(attempt), status)
    return exit_code if consistent else 70


def _tail_text(path: Path, *, limit: int) -> str:
    if path.is_symlink():
        raise RemoteJobError("remote job log must not be a symbolic link")
    if not path.is_file():
        return ""
    size = path.stat().st_size
    with path.open("rb") as stream:
        if size > limit:
            stream.seek(size - limit)
        return stream.read().decode("utf-8", "replace")


def attach_job(
    *,
    run_root: Path | str,
    run_id: str,
    tail_bytes: int = 65_536,
    expected_repo: Path | str | None = None,
    expected_python: Path | str | None = None,
) -> dict[str, Any]:
    if (
        isinstance(tail_bytes, bool)
        or not isinstance(tail_bytes, int)
        or not 1 <= tail_bytes <= 1_048_576
    ):
        raise RemoteJobError("tail_bytes must be in [1, 1048576]")
    persistent_root = _require_absolute_directory(run_root, label="run root")
    root = _job_root(persistent_root, run_id)
    attempts = _attempts(root)
    if not attempts:
        raise RemoteJobError("no durable remote launch exists for this run")
    attempt = attempts[-1]
    return {
        "schema_version": JOB_ATTACH_SCHEMA,
        "status": _attempt_status(
            attempt,
            expected_repo=expected_repo,
            expected_python=expected_python,
        ),
        "logs": {
            "dag_stdout": _tail_text(attempt / "dag.stdout.log", limit=int(tail_bytes)),
            "dag_stderr": _tail_text(attempt / "dag.stderr.log", limit=int(tail_bytes)),
            "supervisor_stdout": _tail_text(
                attempt / "supervisor.stdout.log", limit=int(tail_bytes)
            ),
            "supervisor_stderr": _tail_text(
                attempt / "supervisor.stderr.log", limit=int(tail_bytes)
            ),
        },
    }


__all__ = [
    "JOB_ATTACH_SCHEMA",
    "JOB_CONTROL_SCHEMA",
    "JOB_SPEC_SCHEMA",
    "JOB_STATUS_SCHEMA",
    "RECOVERY_AUDIT_SCHEMA",
    "RemoteJobError",
    "attach_job",
    "audit_recovery_job",
    "job_status",
    "launch_detached_job",
    "process_identity",
    "process_identity_is_alive",
    "supervise_attempt",
    "validate_recovery_audit_report",
]
