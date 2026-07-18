"""Strict-host-key SSH adapter for launching the same isolated DAG runner remotely."""
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
    python: str = "python",
    manifest: str = "scripts/experiment_manifest.yaml",
) -> tuple[str, str]:
    if not _SAFE_ID.fullmatch(run_id):
        raise ValueError("run_id is not a safe identifier")
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
        python, "scripts/dag_runner.py", "--manifest", manifest, "--profile", profile,
        "--run-root", str(root_path), "--run-id", run_id,
    ]
    if resource:
        argv.extend(["--resource", resource])
    command = f"cd {shlex.quote(str(repo_path))} && " + " ".join(shlex.quote(x) for x in argv)
    return command, run_dir


def run_remote(
    *,
    host: str,
    port: int,
    username: str,
    repo: str,
    run_root: str,
    run_id: str,
    profile: str,
    resource: str | None = None,
    known_hosts: str | None = None,
    key_path: str | None = None,
    client_factory: Callable[..., Any] = strict_ssh_client,
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
    command, run_dir = build_remote_command(
        repo=repo, run_root=run_root, run_id=run_id, profile=profile, resource=resource,
    )
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
        sftp = client.open_sftp()
        try:
            with sftp.open(f"{run_dir}/status.json", "r") as status_file:
                status = json.loads(status_file.read())
        finally:
            sftp.close()
        if not isinstance(status, dict) or status.get("run_id") != run_id:
            raise RuntimeError("remote status.json is missing or belongs to another run")
        if status.get("exit_code") != exit_code:
            raise RuntimeError("remote process exit code disagrees with status.json")
        return RemoteRunResult(
            exit_code=exit_code, stdout=out, stderr=err, run_dir=run_dir, status=status,
        )
    finally:
        client.close()
