"""Tiny paramiko-based SSH/SFTP helper for the seetacloud GPU box.

Usage:
  python scripts/remote.py run "nvidia-smi"
  python scripts/remote.py put localpath remotepath
  python scripts/remote.py get remotepath localpath
  python scripts/remote.py put_dir localdir remotedir   # recursive upload

Host settings come from env; authentication is key/agent only and host keys are pinned.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ecgcert.execution.remote import strict_ssh_client  # noqa: E402

HOST = os.environ.get("REMOTE_HOST", "")   # set to your GPU host, e.g. an AutoDL instance
PORT = int(os.environ.get("REMOTE_PORT", "22"))
USER = os.environ.get("REMOTE_USER", "root")
KNOWN_HOSTS = os.environ.get("REMOTE_KNOWN_HOSTS")
KEY_PATH = os.environ.get("REMOTE_KEY_PATH")


def client():
    if not HOST:
        raise RuntimeError("REMOTE_HOST is required")
    if not KNOWN_HOSTS:
        raise RuntimeError("REMOTE_KNOWN_HOSTS is required")
    if not KEY_PATH:
        raise RuntimeError("REMOTE_KEY_PATH is required")
    c = strict_ssh_client(known_hosts=KNOWN_HOSTS)
    kwargs = dict(hostname=HOST, port=PORT, username=USER, key_filename=KEY_PATH,
                  look_for_keys=False, allow_agent=False, timeout=30,
                  banner_timeout=30, auth_timeout=30)
    c.connect(**kwargs)
    return c


def run(cmd: str, get_pty: bool = False) -> int:
    c = client()
    stdin, stdout, stderr = c.exec_command(cmd, get_pty=get_pty, timeout=None)
    for line in iter(stdout.readline, ""):
        sys.stdout.write(line)
        sys.stdout.flush()
    err = stderr.read().decode(errors="replace")
    if err.strip():
        sys.stderr.write(err)
    rc = stdout.channel.recv_exit_status()
    c.close()
    return rc


def put(local: str, remote: str) -> None:
    c = client()
    sf = c.open_sftp()
    _mkdirs(sf, str(Path(remote).parent).replace("\\", "/"))
    sf.put(local, remote)
    print(f"put {local} -> {remote} ({os.path.getsize(local)} B)")
    sf.close()
    c.close()


def get(remote: str, local: str) -> None:
    c = client()
    sf = c.open_sftp()
    Path(local).parent.mkdir(parents=True, exist_ok=True)
    sf.get(remote, local)
    print(f"get {remote} -> {local}")
    sf.close()
    c.close()


def _mkdirs(sf, remote_dir: str) -> None:
    parts = [p for p in remote_dir.split("/") if p]
    cur = "/" if remote_dir.startswith("/") else ""
    for p in parts:
        cur = cur + p if cur in ("", "/") else cur + "/" + p
        try:
            sf.stat(cur)
        except IOError:
            sf.mkdir(cur)


def put_dir(localdir: str, remotedir: str) -> None:
    c = client()
    sf = c.open_sftp()
    localdir = str(localdir)
    n = 0
    for root, _dirs, files in os.walk(localdir):
        rel = os.path.relpath(root, localdir).replace("\\", "/")
        rdir = remotedir if rel == "." else f"{remotedir}/{rel}"
        _mkdirs(sf, rdir)
        for f in files:
            lp = os.path.join(root, f)
            sf.put(lp, f"{rdir}/{f}")
            n += 1
    print(f"put_dir {localdir} -> {remotedir}: {n} files")
    sf.close()
    c.close()


if __name__ == "__main__":
    op = sys.argv[1]
    if op == "run":
        sys.exit(run(sys.argv[2], get_pty=("--pty" in sys.argv)))
    elif op == "put":
        put(sys.argv[2], sys.argv[3])
    elif op == "get":
        get(sys.argv[2], sys.argv[3])
    elif op == "put_dir":
        put_dir(sys.argv[2], sys.argv[3])
    else:
        print("unknown op", op)
        sys.exit(2)
