"""Tiny paramiko-based SSH/SFTP helper for the seetacloud GPU box.

Usage:
  python scripts/remote.py run "nvidia-smi"
  python scripts/remote.py put localpath remotepath
  python scripts/remote.py get remotepath localpath
  python scripts/remote.py put_dir localdir remotedir   # recursive upload

Host/password read from env (REMOTE_HOST, REMOTE_PORT, REMOTE_USER, REMOTE_PASS)
with seetacloud defaults. Not committed with a real password (see .gitignore note).
"""
from __future__ import annotations

import os
import sys
import stat
from pathlib import Path

import paramiko

HOST = os.environ.get("REMOTE_HOST", "")   # set to your GPU host, e.g. an AutoDL instance
PORT = int(os.environ.get("REMOTE_PORT", "22"))
USER = os.environ.get("REMOTE_USER", "root")
PASS = os.environ.get("REMOTE_PASS", "")


def client() -> paramiko.SSHClient:
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PASS,
              timeout=30, banner_timeout=30, auth_timeout=30)
    return c


def run(cmd: str, get_pty: bool = False) -> int:
    c = client()
    stdin, stdout, stderr = c.exec_command(cmd, get_pty=get_pty, timeout=None)
    for line in iter(stdout.readline, ""):
        sys.stdout.write(line); sys.stdout.flush()
    err = stderr.read().decode(errors="replace")
    if err.strip():
        sys.stderr.write(err)
    rc = stdout.channel.recv_exit_status()
    c.close()
    return rc


def put(local: str, remote: str) -> None:
    c = client(); sf = c.open_sftp()
    _mkdirs(sf, str(Path(remote).parent).replace("\\", "/"))
    sf.put(local, remote)
    print(f"put {local} -> {remote} ({os.path.getsize(local)} B)")
    sf.close(); c.close()


def get(remote: str, local: str) -> None:
    c = client(); sf = c.open_sftp()
    Path(local).parent.mkdir(parents=True, exist_ok=True)
    sf.get(remote, local)
    print(f"get {remote} -> {local}")
    sf.close(); c.close()


def _mkdirs(sf, remote_dir: str) -> None:
    parts = [p for p in remote_dir.split("/") if p]
    cur = "/" if remote_dir.startswith("/") else ""
    for p in parts:
        cur = cur + p if cur in ("", "/") else cur + "/" + p
        try:
            sf.stat(cur)
        except IOError:
            sf.mkdir(cur)


def put_text(local: str, remote: str) -> None:
    """Upload a (text/binary) file via `cat > remote` over the exec channel.

    Works where SFTP stat/put fails (e.g. the AutoDL /root/autodl-tmp mount).
    """
    c = client()
    with open(local, "rb") as fh:
        data = fh.read()
    stdin, stdout, stderr = c.exec_command(f"cat > {remote}")
    stdin.channel.sendall(data)
    stdin.channel.shutdown_write()
    rc = stdout.channel.recv_exit_status()
    err = stderr.read().decode(errors="replace")
    if err.strip():
        sys.stderr.write(err)
    print(f"put_text {local} -> {remote} ({len(data)} B) rc={rc}")
    c.close()


def put_dir(localdir: str, remotedir: str) -> None:
    c = client(); sf = c.open_sftp()
    localdir = str(localdir); n = 0
    for root, _dirs, files in os.walk(localdir):
        rel = os.path.relpath(root, localdir).replace("\\", "/")
        rdir = remotedir if rel == "." else f"{remotedir}/{rel}"
        _mkdirs(sf, rdir)
        for f in files:
            lp = os.path.join(root, f)
            sf.put(lp, f"{rdir}/{f}")
            n += 1
    print(f"put_dir {localdir} -> {remotedir}: {n} files")
    sf.close(); c.close()


if __name__ == "__main__":
    op = sys.argv[1]
    if op == "run":
        sys.exit(run(sys.argv[2], get_pty=("--pty" in sys.argv)))
    elif op == "put":
        put(sys.argv[2], sys.argv[3])
    elif op == "put_text":
        put_text(sys.argv[2], sys.argv[3])
    elif op == "get":
        get(sys.argv[2], sys.argv[3])
    elif op == "put_dir":
        put_dir(sys.argv[2], sys.argv[3])
    else:
        print("unknown op", op); sys.exit(2)
