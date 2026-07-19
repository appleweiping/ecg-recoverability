"""Strict-host-key SSH control for durable, reconnectable remote DAG jobs."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from pathlib import PurePosixPath
import re
import shlex
from typing import Any, Callable

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


@dataclass(frozen=True)
class RemoteRunResult:
    action: str
    exit_code: int
    stdout: str
    stderr: str
    run_dir: str
    status: dict[str, Any]


def strict_ssh_client(*, known_hosts: str) -> Any:
    """Create a client pinned exclusively to one explicit ``known_hosts`` file."""
    if not isinstance(known_hosts, str) or not known_hosts.strip():
        raise ValueError("an explicit nonempty known_hosts file is required")
    known_hosts_path = Path(known_hosts).expanduser().resolve()
    if not known_hosts_path.is_file():
        raise FileNotFoundError(known_hosts_path)
    if known_hosts_path.stat().st_size == 0:
        raise ValueError("the explicit known_hosts file must not be empty")
    try:
        import paramiko
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError("paramiko is required for remote execution") from exc
    client = paramiko.SSHClient()
    client.load_host_keys(str(known_hosts_path))
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    return client


def build_remote_command(
    *,
    repo: str,
    run_root: str,
    run_id: str,
    profile: str,
    resource: str | None,
    environment_lock: str,
    python: str,
    manifest: str = "scripts/experiment_manifest.yaml",
    resume: bool = False,
) -> tuple[str, str]:
    """Build an idempotent server-side *background* launch command."""

    if not _SAFE_ID.fullmatch(run_id):
        raise ValueError("run_id is not a safe identifier")
    if environment_lock not in {"cpu", "gpu"}:
        raise ValueError("environment_lock must be 'cpu' or 'gpu'")
    python_path = PurePosixPath(python)
    if not python_path.is_absolute() or ".." in python_path.parts:
        raise ValueError("remote Python must be an explicit safe absolute POSIX path")
    repo_path = PurePosixPath(repo)
    root_path = PurePosixPath(run_root)
    if (
        not repo_path.is_absolute()
        or not root_path.is_absolute()
        or ".." in repo_path.parts + root_path.parts
    ):
        raise ValueError("remote repo and run_root must be safe absolute POSIX paths")
    run_dir = str(root_path / run_id)
    argv = [
        python,
        "scripts/remote_job.py",
        "launch",
        "--repo",
        str(repo_path),
        "--manifest",
        manifest,
        "--profile",
        profile,
        "--run-root",
        str(root_path),
        "--run-id",
        run_id,
        "--environment-lock",
        environment_lock,
    ]
    if resource:
        argv.extend(["--resource", resource])
    if resume:
        argv.append("--resume")
    command = f"cd {shlex.quote(str(repo_path))} && " + " ".join(shlex.quote(x) for x in argv)
    return command, run_dir


def build_remote_control_command(
    *,
    action: str,
    repo: str,
    run_root: str,
    run_id: str,
    python: str,
    tail_bytes: int = 65_536,
) -> tuple[str, str]:
    """Build a one-shot status/log query that can be called after reconnecting."""

    if action not in {"status", "attach", "recover-audit"}:
        raise ValueError("remote control action must be status, attach, or recover-audit")
    if not _SAFE_ID.fullmatch(run_id):
        raise ValueError("run_id is not a safe identifier")
    python_path = PurePosixPath(python)
    repo_path = PurePosixPath(repo)
    root_path = PurePosixPath(run_root)
    if (
        not python_path.is_absolute()
        or not repo_path.is_absolute()
        or not root_path.is_absolute()
        or ".." in python_path.parts + repo_path.parts + root_path.parts
    ):
        raise ValueError("remote Python, repo and run_root must be safe absolute POSIX paths")
    if (
        isinstance(tail_bytes, bool)
        or not isinstance(tail_bytes, int)
        or not 1 <= tail_bytes <= 1_048_576
    ):
        raise ValueError("tail_bytes must be in [1, 1048576]")
    argv = [
        python,
        "scripts/remote_job.py",
        action,
        "--repo",
        str(repo_path),
        "--run-root",
        str(root_path),
        "--run-id",
        run_id,
    ]
    if action == "attach":
        argv.extend(("--tail-bytes", str(tail_bytes)))
    command = f"cd {shlex.quote(str(repo_path))} && " + " ".join(
        shlex.quote(value) for value in argv
    )
    return command, str(root_path / run_id)


def _control_report(stdout: str, *, action: str, run_id: str) -> dict[str, Any]:
    lines = [line for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"remote {action} returned no JSON control report")
    try:
        report = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"remote {action} returned an invalid JSON control report") from exc
    if not isinstance(report, dict):
        raise RuntimeError(f"remote {action} control report is not an object")
    identity = report.get("status") if action == "attach" else report
    if not isinstance(identity, dict) or identity.get("run_id") != run_id:
        raise RuntimeError(f"remote {action} control report belongs to another run")
    expected_schema = {
        "launch": "ecgcert-remote-job-control/v1",
        "status": "ecgcert-remote-job-control/v1",
        "attach": "ecgcert-remote-job-control/v1",
        "recover-audit": "ecgcert-remote-recovery-audit/v1",
    }[action]
    if identity.get("schema_version") != expected_schema:
        raise RuntimeError(f"remote {action} control report schema is invalid")
    for field in ("repo", "run_root", "run_dir", "python_executable"):
        value = identity.get(field)
        if not isinstance(value, str) or not PurePosixPath(value).is_absolute():
            raise RuntimeError(f"remote {action} control report route identity is invalid")
    logical_job_sha256 = identity.get("logical_job_sha256")
    if not isinstance(logical_job_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", logical_job_sha256
    ):
        raise RuntimeError(f"remote {action} logical job identity is invalid")
    return report


def run_remote(
    *,
    host: str,
    port: int,
    username: str,
    repo: str,
    run_root: str,
    run_id: str,
    profile: str | None,
    environment_lock: str | None,
    remote_python: str,
    resource: str | None = None,
    known_hosts: str | None = None,
    key_path: str | None = None,
    client_factory: Callable[..., Any] = strict_ssh_client,
    resume: bool = False,
    action: str = "launch",
    tail_bytes: int = 65_536,
) -> RemoteRunResult:
    if not host or not username or not 1 <= port <= 65535:
        raise ValueError("host, username and a valid port are required")
    if not known_hosts:
        raise ValueError("an explicit pinned known_hosts file is required")
    if not key_path:
        raise ValueError("an explicit private key path is required")
    known_hosts_path = Path(known_hosts).expanduser().resolve()
    private_key_path = Path(key_path).expanduser().resolve()
    if not known_hosts_path.is_file():
        raise FileNotFoundError(known_hosts_path)
    if not private_key_path.is_file():
        raise FileNotFoundError(private_key_path)
    if action == "launch":
        if profile is None or environment_lock is None:
            raise ValueError("launch requires profile and environment_lock")
        command, run_dir = build_remote_command(
            repo=repo,
            run_root=run_root,
            run_id=run_id,
            profile=profile,
            resource=resource,
            environment_lock=environment_lock,
            python=remote_python,
            resume=resume,
        )
    elif action in {"status", "attach", "recover-audit"}:
        if resume:
            raise ValueError("--resume is accepted only by the launch action")
        command, run_dir = build_remote_control_command(
            action=action,
            repo=repo,
            run_root=run_root,
            run_id=run_id,
            python=remote_python,
            tail_bytes=tail_bytes,
        )
    else:
        raise ValueError("action must be launch, status, attach, or recover-audit")
    client = client_factory(known_hosts=str(known_hosts_path))
    connect_args: dict[str, Any] = {
        "hostname": host,
        "port": port,
        "username": username,
        "key_filename": str(private_key_path),
        "look_for_keys": False,
        "allow_agent": False,
        "timeout": 30,
        "banner_timeout": 30,
        "auth_timeout": 30,
    }
    try:
        client.connect(**connect_args)
        _stdin, stdout, stderr = client.exec_command(command, get_pty=False, timeout=None)
        out = stdout.read().decode("utf-8", "replace")
        err = stderr.read().decode("utf-8", "replace")
        exit_code = stdout.channel.recv_exit_status()
        status = (
            _control_report(out, action=action, run_id=run_id)
            if exit_code == 0
            else {
                "schema_version": "ecgcert-remote-job-control-error/v1",
                "run_id": run_id,
                "state": "control_failed",
                "action": action,
            }
        )
        return RemoteRunResult(
            action=action,
            exit_code=exit_code,
            stdout=out,
            stderr=err,
            run_dir=run_dir,
            status=status,
        )
    finally:
        client.close()
