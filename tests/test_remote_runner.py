import inspect
import sys
from types import SimpleNamespace

import pytest

from ecgcert.execution import remote


class _Channel:
    def recv_exit_status(self):
        return 0


class _Stream:
    def __init__(self, value=b""):
        self.value = value
        self.channel = _Channel()

    def read(self):
        return self.value


class _Client:
    def __init__(self):
        self.connected = None
        self.command = None
        self.closed = False

    def connect(self, **kwargs):
        self.connected = kwargs

    def exec_command(self, command, **kwargs):
        self.command = command
        return None, _Stream(b"ok\n"), _Stream()

    def open_sftp(self):
        return _SFTP()

    def close(self):
        self.closed = True


class _StatusFile:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self):
        return '{"run_id": "run-1", "state": "succeeded", "exit_code": 0}'


class _SFTP:
    def open(self, path, mode):
        assert path == "/srv/runs/run-1/status.json" and mode == "r"
        return _StatusFile()

    def close(self):
        pass


def test_remote_command_is_isolated_and_has_no_global_process_control():
    command, run_dir = remote.build_remote_command(
        repo="/srv/ecg repo", run_root="/srv/ecg-runs", run_id="run-1",
        profile="icassp", resource="gpu",
    )
    assert "dag_runner.py" in command and "--run-id run-1" in command
    assert run_dir == "/srv/ecg-runs/run-1"
    assert "pkill" not in command.lower() and "sentinel" not in command.lower()
    with pytest.raises(ValueError, match="safe identifier"):
        remote.build_remote_command(
            repo="/srv/repo", run_root="/srv/runs", run_id="x; rm -rf /",
            profile="icassp", resource=None,
        )


def test_remote_run_uses_factory_and_does_not_disable_host_checks(tmp_path):
    client = _Client()
    known_hosts = tmp_path / "known_hosts"
    key = tmp_path / "id_ed25519"
    known_hosts.write_text("fixture")
    key.write_text("fixture")
    result = remote.run_remote(
        host="example.invalid", port=22, username="runner", repo="/srv/repo",
        run_root="/srv/runs", run_id="run-1", profile="icassp",
        known_hosts=str(known_hosts), key_path=str(key),
        client_factory=lambda **_kwargs: client,
    )
    assert result.exit_code == 0 and result.stdout == "ok\n"
    assert result.status["state"] == "succeeded"
    assert client.connected["look_for_keys"] is False
    assert client.connected["allow_agent"] is False
    assert client.connected["key_filename"] == str(key.resolve())
    assert "password" not in client.connected
    assert client.closed
    source = inspect.getsource(remote.strict_ssh_client)
    assert "RejectPolicy" in source
    assert "AutoAdd" not in source
    assert "load_system_host_keys" not in source


def test_strict_client_loads_only_the_explicit_known_hosts(tmp_path, monkeypatch):
    known_hosts = tmp_path / "known_hosts"
    known_hosts.write_text("example.invalid ssh-ed25519 AAAAfixture\n", encoding="utf-8")
    calls = []

    class RejectPolicy:
        pass

    class FakeClient:
        def load_host_keys(self, path):
            calls.append(("explicit", path))

        def load_system_host_keys(self):  # pragma: no cover - must never be reached
            raise AssertionError("system host keys must not be trusted")

        def set_missing_host_key_policy(self, policy):
            calls.append(("policy", policy))

    fake_paramiko = SimpleNamespace(SSHClient=FakeClient, RejectPolicy=RejectPolicy)
    monkeypatch.setitem(sys.modules, "paramiko", fake_paramiko)

    client = remote.strict_ssh_client(known_hosts=str(known_hosts))

    assert isinstance(client, FakeClient)
    assert calls[0] == ("explicit", str(known_hosts.resolve()))
    assert calls[1][0] == "policy"
    assert isinstance(calls[1][1], RejectPolicy)
    assert len(calls) == 2


def test_strict_client_rejects_missing_or_empty_known_hosts(tmp_path):
    with pytest.raises(TypeError):
        remote.strict_ssh_client()
    with pytest.raises(ValueError, match="explicit nonempty"):
        remote.strict_ssh_client(known_hosts="")
    with pytest.raises(FileNotFoundError):
        remote.strict_ssh_client(known_hosts=str(tmp_path / "missing"))
    empty = tmp_path / "known_hosts"
    empty.touch()
    with pytest.raises(ValueError, match="must not be empty"):
        remote.strict_ssh_client(known_hosts=str(empty))
