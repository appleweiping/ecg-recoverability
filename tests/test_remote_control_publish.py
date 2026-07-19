import errno
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest
import yaml

from ecgcert import lineage
from ecgcert.execution.control_publish import (
    ControlPublicationError,
    PUBLICATION_SCHEMA,
    PULL_SCHEMA,
    atomic_publish_local_bytes,
    publish_remote_control,
    pull_remote_run_artifact,
)
from ecgcert.execution.envelope import ResultEnvelope, SCHEMA_VERSION
from ecgcert.execution.late_inputs import (
    CAPTURE_SCHEMA,
    SNAPSHOT_SCHEMA,
    empty_late_control_snapshot_sha256,
    secure_control_path_metadata,
)
from ecgcert.execution.manifest import ExperimentManifest
from ecgcert.execution.runner import declared_path_hashes
from scripts import remote as legacy_remote
from scripts import publish_remote_control as publish_cli
from scripts import pull_remote_run_artifact as pull_cli


RUN_ID = "run-1"
COMMIT = "a" * 40
NODE_ID = "stage5_review"
WORKSPACE = "/srv/runs/run-1/workspace"
TARGET = "artifacts/gates/stage5.approval.v3.json"


def _run_status(
    manifest: ExperimentManifest,
    *,
    python: str,
    commit: str,
    nodes: dict,
    state: str = "running",
    exit_code=None,
):
    normalized_nodes = {node_id: dict(node) for node_id, node in nodes.items()}
    for node in normalized_nodes.values():
        if node.get("state") == "succeeded":
            node.setdefault("started_at", "2026-07-19T00:00:00Z")
            node.setdefault("finished_at", "2026-07-19T00:01:00Z")
            node.setdefault("timed_out", False)
    environment_lock = {
        "lock_name": "gpu",
        "lock_path": "environments/gpu.lock.txt",
        "lock_sha256": "e" * 64,
        "python_executable": python,
        "requirement_count": 1,
        "applicable_requirement_count": 1,
        "checked_requirement_count": 1,
        "mismatches": [],
        "ok": True,
    }
    value = {
        "schema_version": 3,
        "run_id": RUN_ID,
        "control_root": "/srv/runs",
        "profile": "icassp",
        "resource": None,
        "environment_lock": environment_lock,
        "environment_sha256": "d" * 64,
        "python_executable": python,
        "manifest_sha256": manifest.sha256(),
        "commit": commit,
        "source_snapshot_sha256": "f" * 64,
        "mutable_workspace_roots": ["artifacts"],
        "state": state,
        "exit_code": exit_code,
        "nodes": normalized_nodes,
    }
    identity = {
        key: value[key]
        for key in (
            "run_id",
            "control_root",
            "profile",
            "resource",
            "environment_lock",
            "environment_sha256",
            "python_executable",
            "manifest_sha256",
            "commit",
            "source_snapshot_sha256",
            "mutable_workspace_roots",
        )
    }
    value["run_identity_sha256"] = lineage.canonical_sha256(identity)
    return value


def _manifest_payload():
    return {
        "schema_version": 1,
        "nodes": [
            {
                "id": NODE_ID,
                "profile": ["icassp", "extended", "legacy"],
                "command": ["{python}", "wait.py", "--approval", TARGET],
                "resource": {"kind": "paper", "cpus": 1, "memory_gb": 2, "gpus": 0},
                "deps": [],
                "inputs": [],
                "late_control_inputs": [TARGET],
                "outputs": ["artifacts/control/stage5_review"],
                "timeout": 30,
                "seed": 0,
            }
        ],
    }


class LocalSFTP:
    def __init__(self, root: Path):
        self.root = root
        self.closed = False
        self.rename_hook = None
        self.open_hook = None
        self.open_counts = {}

    def _path(self, remote: str | PurePosixPath) -> Path:
        value = PurePosixPath(remote)
        assert value.is_absolute()
        return self.root.joinpath(*value.parts[1:])

    @staticmethod
    def _attr(path: Path, filename: str | None = None):
        value = path.lstat()
        return SimpleNamespace(
            st_mode=value.st_mode,
            st_size=value.st_size,
            st_mtime=int(value.st_mtime),
            filename=filename,
        )

    def lstat(self, remote):
        path = self._path(remote)
        try:
            return self._attr(path)
        except FileNotFoundError as error:
            error.errno = errno.ENOENT
            raise

    def normalize(self, remote):
        return PurePosixPath(remote).as_posix()

    def mkdir(self, remote, mode=0o777):
        path = self._path(remote)
        path.mkdir(mode=mode)

    def open(self, remote, mode):
        key = (PurePosixPath(remote).as_posix(), mode)
        self.open_counts[key] = self.open_counts.get(key, 0) + 1
        if self.open_hook is not None:
            self.open_hook(key[0], mode, self.open_counts[key])
        path = self._path(remote)
        path.parent.mkdir(parents=True, exist_ok=True)
        translated = {"x": "xb", "rb": "rb"}[mode]
        return path.open(translated)

    def listdir_attr(self, remote):
        path = self._path(remote)
        return [self._attr(child, child.name) for child in path.iterdir()]

    def rename(self, source, destination):
        source_path = self._path(source)
        destination_path = self._path(destination)
        if self.rename_hook is not None:
            self.rename_hook(source_path, destination_path)
        if destination_path.exists() or destination_path.is_symlink():
            raise FileExistsError(errno.EEXIST, "destination exists", destination_path)
        os.rename(source_path, destination_path)

    def remove(self, remote):
        self._path(remote).unlink()

    def rmdir(self, remote):
        self._path(remote).rmdir()

    def close(self):
        self.closed = True


class FakeKey:
    @staticmethod
    def asbytes():
        return b"pinned-server-host-key"


class FakeTransport:
    @staticmethod
    def get_remote_server_key():
        return FakeKey()


class FakeClient:
    def __init__(self, sftp: LocalSFTP):
        self.sftp = sftp
        self.connect_args = None
        self.closed = False

    def connect(self, **kwargs):
        self.connect_args = kwargs

    @staticmethod
    def get_transport():
        return FakeTransport()

    def open_sftp(self):
        return self.sftp

    def close(self):
        self.closed = True


def _fixture(tmp_path: Path):
    payload = _manifest_payload()
    manifest_path = tmp_path / "experiment_manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    manifest = ExperimentManifest.from_path(manifest_path)
    remote_root = tmp_path / "remote"
    workspace = remote_root / "srv" / "runs" / RUN_ID / "workspace"
    (workspace / "scripts").mkdir(parents=True)
    (workspace / "scripts" / "experiment_manifest.yaml").write_bytes(manifest_path.read_bytes())
    status = _run_status(
        manifest,
        python="/opt/ecg/bin/python",
        commit=COMMIT,
        nodes={NODE_ID: {"state": "running", "exit_code": None}},
    )
    (workspace.parent / "status.json").write_text(json.dumps(status), encoding="utf-8")
    known_hosts = tmp_path / "known_hosts"
    known_hosts.write_text("[host]:22886 ssh-ed25519 fixture\n", encoding="utf-8")
    key = tmp_path / "id_ed25519"
    key.write_text("fixture-private-key", encoding="utf-8")
    source = tmp_path / "approval.json"
    source.write_bytes(b'{"signed":true}\n')
    sftp = LocalSFTP(remote_root)
    client = FakeClient(sftp)
    return manifest_path, known_hosts, key, source, workspace, sftp, client


def _publish(fixture):
    manifest_path, known_hosts, key, source, _workspace, _sftp, client = fixture
    report = publish_remote_control(
        local_path=source,
        manifest_path=manifest_path,
        node_id=NODE_ID,
        run_id=RUN_ID,
        expected_commit=COMMIT,
        remote_workspace=WORKSPACE,
        host="connect.example.invalid",
        port=22886,
        username="root",
        known_hosts=known_hosts,
        key_path=key,
        client_factory=lambda **kwargs: client,
    )
    return report


def test_remote_control_publish_is_key_only_atomic_and_sha_verified(tmp_path):
    fixture = _fixture(tmp_path)
    report = _publish(fixture)
    _manifest, known_hosts, key, source, workspace, sftp, client = fixture

    assert report["schema_version"] == PUBLICATION_SCHEMA
    assert report["remote_target"] == TARGET
    assert report["source_sha256"] == report["remote_readback_sha256"]
    assert (workspace / TARGET).read_bytes() == source.read_bytes()
    assert not list((workspace / "artifacts" / "gates").glob(".ecgcert-upload-*"))
    assert client.connect_args == {
        "hostname": "connect.example.invalid",
        "port": 22886,
        "username": "root",
        "key_filename": str(key.resolve()),
        "look_for_keys": False,
        "allow_agent": False,
        "timeout": 30,
        "banner_timeout": 30,
        "auth_timeout": 30,
    }
    assert report["transport"]["strict_known_hosts"] is True
    assert report["transport"]["key_only"] is True
    assert report["transport"]["rename_operation"] == "SSH_FXP_RENAME"
    assert report["transport"]["posix_rename_extension_used"] is False
    assert report["transport"]["known_hosts_sha256"]
    assert "key_filename" not in report["transport"]
    assert sftp.closed and client.closed


def test_remote_control_publish_atomically_commits_a_complete_directory(tmp_path):
    fixture = list(_fixture(tmp_path))
    source = fixture[3]
    source.unlink()
    source.mkdir()
    (source / "receipt.v1.json").write_text("{}", encoding="utf-8")
    (source / "stage-05").mkdir()
    (source / "stage-05" / "decision.json").write_text('{"decision":"proceed"}', encoding="utf-8")

    report = _publish(tuple(fixture))
    workspace = fixture[4]
    assert report["source_kind"] == "directory"
    assert (workspace / TARGET / "stage-05" / "decision.json").is_file()
    assert report["source_sha256"] == report["remote_readback_sha256"]


def test_remote_control_publish_refuses_overwrite_and_preserves_existing(tmp_path):
    fixture = _fixture(tmp_path)
    workspace = fixture[4]
    target = workspace / TARGET
    target.parent.mkdir(parents=True)
    target.write_bytes(b"existing")

    with pytest.raises(ControlPublicationError, match="overwrite is forbidden"):
        _publish(fixture)
    assert target.read_bytes() == b"existing"
    assert not list(target.parent.glob(".ecgcert-upload-*"))


def test_remote_control_publish_loses_no_overwrite_race_and_cleans_temp(tmp_path):
    fixture = _fixture(tmp_path)
    workspace, sftp = fixture[4], fixture[5]

    def race(_source: Path, destination: Path):
        destination.write_bytes(b"racer")

    sftp.rename_hook = race
    with pytest.raises(ControlPublicationError, match="non-overwriting"):
        _publish(fixture)
    assert (workspace / TARGET).read_bytes() == b"racer"
    assert not list((workspace / "artifacts" / "gates").glob(".ecgcert-upload-*"))


@pytest.mark.parametrize("tamper", ["manifest", "run", "node"])
def test_remote_control_publish_authenticates_run_manifest_and_node_state(tmp_path, tamper):
    fixture = _fixture(tmp_path)
    manifest_path, _known, _key, _source, workspace, _sftp, _client = fixture
    if tamper == "manifest":
        (workspace / "scripts" / "experiment_manifest.yaml").write_text(
            "different", encoding="utf-8"
        )
        message = "manifest differs"
    else:
        status_path = workspace.parent / "status.json"
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if tamper == "run":
            status["run_id"] = "another-run"
            message = "run identity"
        else:
            status["nodes"][NODE_ID]["state"] = "succeeded"
            message = "run identity/node state"
        status_path.write_text(json.dumps(status), encoding="utf-8")
    with pytest.raises(ControlPublicationError, match=message):
        _publish(fixture)
    assert manifest_path.is_file()


def test_remote_control_publish_rejects_wrong_workspace_and_symlink_source(tmp_path):
    fixture = _fixture(tmp_path)
    manifest, known, key, source, _workspace, _sftp, client = fixture
    with pytest.raises(ControlPublicationError, match="<run_id>/workspace"):
        publish_remote_control(
            local_path=source,
            manifest_path=manifest,
            node_id=NODE_ID,
            run_id=RUN_ID,
            expected_commit=COMMIT,
            remote_workspace="/srv/runs/other/workspace",
            host="host",
            port=22,
            username="root",
            known_hosts=known,
            key_path=key,
            client_factory=lambda **kwargs: client,
        )

    external = tmp_path / "external.json"
    external.write_text("{}", encoding="utf-8")
    source.unlink()
    try:
        source.symlink_to(external)
    except OSError as error:
        pytest.skip(f"symlinks unavailable: {error}")
    with pytest.raises(ControlPublicationError, match="must not be a symlink"):
        _publish(fixture)


def test_legacy_remote_helper_cannot_bypass_control_publication_protocol():
    with pytest.raises(RuntimeError, match="publish_remote_control"):
        legacy_remote._reject_gate_control_target(
            "/srv/runs/run-1/workspace/artifacts/gates/stage5.approval.v3.json"
        )


def _pull_fixture(
    tmp_path: Path,
    *,
    operator_response: bool = False,
    sibling: bool = False,
    late: bool = False,
):
    tmp_path.mkdir(parents=True, exist_ok=True)
    if operator_response:
        node_id = "arc_stage5_forward"
        output = "artifacts/gates/arc-operator-responses/stage-05/operator-response.v2.json"
        artifact = output
    else:
        node_id = "stage5_gate"
        output = "artifacts/control/stage5"
        artifact = "artifacts/control/stage5/decision.v3.json"
    late_path = "artifacts/gates/test.approval.v3.json"
    command = ["{python}", "produce.py"]
    if late:
        command.extend(("--approval", late_path))
    payload = {
        "schema_version": 1,
        "nodes": [
            {
                "id": node_id,
                "profile": ["icassp", "extended", "legacy"],
                "command": command,
                "resource": {"kind": "paper", "cpus": 1, "memory_gb": 2, "gpus": 0},
                "deps": [],
                "inputs": [],
                **({"late_control_inputs": [late_path]} if late else {}),
                "outputs": [output],
                "timeout": 30,
                "seed": 0,
            }
        ],
    }
    manifest_path = tmp_path / "pull-manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    manifest = ExperimentManifest.from_path(manifest_path)
    remote_root = tmp_path / "pull-remote"
    workspace = remote_root / "srv" / "runs" / RUN_ID / "workspace"
    (workspace / "scripts").mkdir(parents=True)
    (workspace / "scripts" / "experiment_manifest.yaml").write_bytes(manifest_path.read_bytes())
    artifact_path = workspace / artifact
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_bytes(b'{"schema_version":"handoff","decision":"PROCEED"}\n')
    if sibling:
        (artifact_path.parent / "sibling.txt").write_text("stable", encoding="utf-8")
        (artifact_path.parent / "nested").mkdir()
        (artifact_path.parent / "nested" / "member.json").write_text(
            "{}", encoding="utf-8"
        )
        (artifact_path.parent / "empty").mkdir()
    output_hashes = declared_path_hashes(workspace, (output,))
    commit = COMMIT
    python = "/opt/ecg/bin/python"
    node = manifest.by_id()[node_id]
    late_hashes = {}
    late_snapshot_sha256 = empty_late_control_snapshot_sha256()
    snapshot_root = workspace.parent / "control-inputs" / node_id
    if late:
        payload_bytes = b'{"signed":true}\n'
        payload_digest = hashlib.sha256(payload_bytes).hexdigest()
        payload_path = snapshot_root / "payload" / "0000"
        payload_path.parent.mkdir(parents=True, exist_ok=True)
        payload_path.write_bytes(payload_bytes)
        record = {
            "schema_version": CAPTURE_SCHEMA,
            "index": 0,
            "source_path": late_path,
            "payload_path": "payload/0000",
            "kind": "file",
            "sha256": payload_digest,
            "bytes": len(payload_bytes),
            "captured_at": "2026-07-19T00:00:00Z",
        }
        record_path = snapshot_root / "records" / "0000.json"
        record_path.parent.mkdir(parents=True, exist_ok=True)
        record_path.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
        late_hashes = {late_path: payload_digest}
        snapshot_manifest = {
            "schema_version": SNAPSHOT_SCHEMA,
            "run_id": RUN_ID,
            "node_id": node_id,
            "inputs": [record],
            "inputs_sha256": late_hashes,
            "inputs_bundle_sha256": lineage.canonical_sha256(late_hashes),
        }
        (snapshot_root / "manifest.v1.json").write_text(
            json.dumps(snapshot_manifest, sort_keys=True), encoding="utf-8"
        )
        late_snapshot_sha256 = secure_control_path_metadata(snapshot_root)[0]
    expected_argv = [python if token == "{python}" else token for token in node.command]
    effective_data = {
        f"late-control:{relative}": digest for relative, digest in late_hashes.items()
    }
    split_material = {
        "seed": 0,
        "argv": expected_argv,
        "inputs": {},
        "deps": [],
    }
    if late:
        split_material["late_control_inputs_sha256"] = late_hashes
        split_material["late_control_snapshot_sha256"] = late_snapshot_sha256
    envelope = ResultEnvelope(
        schema_version=SCHEMA_VERSION,
        run_id=RUN_ID,
        node_id=node_id,
        status="succeeded",
        exit_code=0,
        started_at="2026-07-19T00:00:00Z",
        finished_at="2026-07-19T00:01:00Z",
        commit=commit,
        dirty=False,
        argv=expected_argv,
        config_sha256=(
            node.config_sha256(late_control_inputs_sha256=late_hashes)
            if late
            else node.config_sha256()
        ),
        data_sha256=lineage.canonical_sha256(effective_data),
        split_sha256=lineage.canonical_sha256(split_material),
        env_sha256="d" * 64,
        environment_lock_sha256="e" * 64,
        source_sha256="f" * 64,
        hardware={"cpu_count": 1},
        seed=0,
        upstream_sha256={},
        late_control_inputs_sha256=late_hashes,
        late_control_snapshot_sha256=late_snapshot_sha256,
        checkpoint_sha256={},
        outputs_sha256=output_hashes,
    )
    envelope_path = workspace.parent / "envelopes" / f"{node_id}.json"
    envelope.write(envelope_path)
    status_value = _run_status(
        manifest,
        python=python,
        commit=commit,
        nodes={node_id: {"state": "succeeded", "exit_code": 0}},
    )
    (workspace.parent / "status.json").write_text(json.dumps(status_value), encoding="utf-8")
    known_hosts = tmp_path / "pull-known-hosts"
    known_hosts.write_text("[host]:22886 ssh-ed25519 fixture\n", encoding="utf-8")
    key = tmp_path / "pull-key"
    key.write_text("private-key-fixture", encoding="utf-8")
    sftp = LocalSFTP(remote_root)
    client = FakeClient(sftp)
    destination = tmp_path / "local-handoffs" / artifact_path.name
    return {
        "manifest": manifest_path,
        "node_id": node_id,
        "artifact": artifact,
        "workspace": workspace,
        "artifact_path": artifact_path,
        "known_hosts": known_hosts,
        "key": key,
        "sftp": sftp,
        "client": client,
        "destination": destination,
        "output": output,
        "envelope_path": envelope_path,
        "status_path": workspace.parent / "status.json",
        "snapshot_root": snapshot_root,
    }


def _pull(fixture, *, expected_commit=COMMIT):
    return pull_remote_run_artifact(
        remote_artifact=fixture["artifact"],
        destination=fixture["destination"],
        manifest_path=fixture["manifest"],
        producer_node_id=fixture["node_id"],
        run_id=RUN_ID,
        expected_commit=expected_commit,
        remote_workspace=WORKSPACE,
        host="connect.example.invalid",
        port=22886,
        username="root",
        known_hosts=fixture["known_hosts"],
        key_path=fixture["key"],
        client_factory=lambda **kwargs: fixture["client"],
    )


@pytest.mark.parametrize("operator_response", [False, True])
def test_authenticated_pull_closes_gate_and_arc_reverse_handoffs(tmp_path, operator_response):
    fixture = _pull_fixture(tmp_path, operator_response=operator_response)
    report = _pull(fixture)

    assert report["schema_version"] == PULL_SCHEMA
    assert report["producer_node_id"] == fixture["node_id"]
    assert report["remote_artifact"] == fixture["artifact"]
    assert report["remote_sha256"] == report["local_readback_sha256"]
    assert fixture["destination"].read_bytes() == fixture["artifact_path"].read_bytes()
    assert not list(fixture["destination"].parent.glob(".ecgcert-download-*"))
    assert report["transport"]["strict_known_hosts"] is True
    assert report["transport"]["key_only"] is True
    assert report["transport"]["remote_source_stability_verified"] is True
    assert report["producer_outputs_sha256"][fixture["output"]]


def test_authenticated_pull_refuses_output_tamper_or_undeclared_source(tmp_path):
    fixture = _pull_fixture(tmp_path)
    fixture["artifact_path"].write_bytes(b"tampered-after-envelope")
    with pytest.raises(ControlPublicationError, match="output SHA"):
        _pull(fixture)

    fixture = _pull_fixture(tmp_path / "other")
    fixture["artifact"] = "artifacts/control/another/decision.json"
    with pytest.raises(ControlPublicationError, match="not uniquely covered"):
        _pull(fixture)


def test_authenticated_pull_refuses_local_overwrite(tmp_path):
    fixture = _pull_fixture(tmp_path)
    fixture["destination"].parent.mkdir(parents=True)
    fixture["destination"].write_bytes(b"keep")
    with pytest.raises(ControlPublicationError, match="overwrite is forbidden"):
        _pull(fixture)
    assert fixture["destination"].read_bytes() == b"keep"


def test_authenticated_pull_detects_remote_mutation_during_download(tmp_path):
    fixture = _pull_fixture(tmp_path, operator_response=True)
    remote_source = PurePosixPath(WORKSPACE, fixture["artifact"]).as_posix()

    def mutate_on_post_hash(remote, mode, count):
        # Output-envelope verification, explicit pre-hash, and download are the
        # first three reads. Mutate immediately before the post-download hash.
        if remote == remote_source and mode == "rb" and count == 4:
            fixture["artifact_path"].write_bytes(b"changed-during-transfer")

    fixture["sftp"].open_hook = mutate_on_post_hash
    with pytest.raises(ControlPublicationError, match="changed .*download|changed while hashing"):
        _pull(fixture)
    assert not fixture["destination"].exists()
    assert not list(fixture["destination"].parent.glob(".ecgcert-download-*"))


def test_authenticated_pull_requires_success_status_and_matching_envelope(tmp_path):
    fixture = _pull_fixture(tmp_path)
    status_path = fixture["workspace"].parent / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["nodes"][fixture["node_id"]]["state"] = "running"
    status_path.write_text(json.dumps(status), encoding="utf-8")
    with pytest.raises(ControlPublicationError, match="not succeeded"):
        _pull(fixture)

    fixture = _pull_fixture(tmp_path / "envelope")
    envelope_path = fixture["workspace"].parent / "envelopes" / f"{fixture['node_id']}.json"
    value = json.loads(envelope_path.read_text(encoding="utf-8"))
    value["outputs_sha256"][fixture["output"]] = "0" * 64
    envelope_path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ControlPublicationError, match="output SHA"):
        _pull(fixture)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("started_at", "2026-07-19T00:00:01Z", "identity/config"),
        ("seed", 1, "identity/config"),
        ("config_sha256", "0" * 64, "identity/config"),
        ("data_sha256", "0" * 64, "input-data"),
        ("split_sha256", "0" * 64, "input/split"),
        ("env_sha256", "0" * 64, "identity/config"),
        ("environment_lock_sha256", "0" * 64, "identity/config"),
        ("source_sha256", "0" * 64, "identity/config"),
        ("checkpoint_sha256", {"ghost.ckpt": "0" * 64}, "checkpoint"),
        ("upstream_sha256", {"ghost": "0" * 64}, "identity/config"),
        (
            "late_control_inputs_sha256",
            {TARGET: "0" * 64},
            "late-control",
        ),
    ],
)
def test_authenticated_pull_recomputes_complete_envelope_contract(
    tmp_path, field, replacement, message
):
    fixture = _pull_fixture(tmp_path)
    value = json.loads(fixture["envelope_path"].read_text(encoding="utf-8"))
    value[field] = replacement
    fixture["envelope_path"].write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ControlPublicationError, match=message):
        _pull(fixture)
    assert not fixture["destination"].exists()


def test_authenticated_pull_rejects_invalid_run_identity_hash(tmp_path):
    fixture = _pull_fixture(tmp_path)
    status = json.loads(fixture["status_path"].read_text(encoding="utf-8"))
    status["run_identity_sha256"] = "0" * 64
    fixture["status_path"].write_text(json.dumps(status), encoding="utf-8")
    with pytest.raises(ControlPublicationError, match="identity hash"):
        _pull(fixture)


def test_authenticated_pull_requires_explicit_frozen_commit(tmp_path):
    fixture = _pull_fixture(tmp_path)
    with pytest.raises(ControlPublicationError, match="differs from expected_commit"):
        _pull(fixture, expected_commit="b" * 40)


def test_authenticated_pull_rechecks_entire_output_after_download(tmp_path):
    fixture = _pull_fixture(tmp_path, sibling=True)
    remote_source = PurePosixPath(WORKSPACE, fixture["artifact"]).as_posix()

    def mutate_sibling_after_download(remote, mode, count):
        if remote == remote_source and mode == "rb" and count == 4:
            (fixture["artifact_path"].parent / "sibling.txt").write_text(
                "changed", encoding="utf-8"
            )

    fixture["sftp"].open_hook = mutate_sibling_after_download
    with pytest.raises(ControlPublicationError, match="output SHA|outputs changed"):
        _pull(fixture)
    assert not fixture["destination"].exists()


def test_remote_directory_hash_exactly_matches_runner_algorithm(tmp_path):
    fixture = _pull_fixture(tmp_path, sibling=True)
    report = _pull(fixture)
    expected = declared_path_hashes(fixture["workspace"], (fixture["output"],))
    assert report["producer_outputs_sha256"] == expected


def test_authenticated_pull_recomputes_late_control_snapshot(tmp_path):
    fixture = _pull_fixture(tmp_path / "valid", late=True)
    report = _pull(fixture)
    assert report["remote_sha256"] == report["local_readback_sha256"]

    fixture = _pull_fixture(tmp_path / "tampered", late=True)
    (fixture["snapshot_root"] / "payload" / "0000").write_bytes(b"tampered")
    with pytest.raises(ControlPublicationError, match="snapshot hash|payload changed"):
        _pull(fixture)
    assert not fixture["destination"].exists()


def test_atomic_local_report_publication_never_replaces(tmp_path):
    destination = tmp_path / "reports" / "pull.json"
    published, digest = atomic_publish_local_bytes(
        destination, b'{"ok":true}\n', label="test report"
    )
    assert published == destination.resolve()
    assert digest == lineage.artifact_sha256(destination)
    with pytest.raises(ControlPublicationError, match="overwrite is forbidden"):
        atomic_publish_local_bytes(destination, b"different", label="test report")
    assert destination.read_bytes() == b'{"ok":true}\n'
    assert not list(destination.parent.glob(".ecgcert-local-publish-*"))


def test_atomic_local_report_publication_loses_race_without_overwrite(
    tmp_path, monkeypatch
):
    destination = tmp_path / "report.json"
    real_link = os.link

    def race(source, target, *args, **kwargs):
        Path(target).write_bytes(b"racer")
        return real_link(source, target, *args, **kwargs)

    monkeypatch.setattr("ecgcert.execution.control_publish.os.link", race)
    with pytest.raises(ControlPublicationError, match="appeared concurrently"):
        atomic_publish_local_bytes(destination, b"ours", label="test report")
    assert destination.read_bytes() == b"racer"
    assert not list(tmp_path.glob(".ecgcert-local-publish-*"))


def test_publish_cli_requires_absent_atomic_audit_report(tmp_path, monkeypatch):
    report = tmp_path / "publish-report.json"
    calls = []

    def fake_publish(**kwargs):
        calls.append(kwargs)
        return {"schema_version": PUBLICATION_SCHEMA, "ok": True}

    monkeypatch.setattr(publish_cli, "publish_remote_control", fake_publish)
    arguments = [
        "publish_remote_control.py",
        "--local",
        str(tmp_path / "approval.json"),
        "--node-id",
        NODE_ID,
        "--run-id",
        RUN_ID,
        "--expected-commit",
        COMMIT,
        "--remote-workspace",
        WORKSPACE,
        "--host",
        "host",
        "--port",
        "22",
        "--username",
        "root",
        "--known-hosts",
        str(tmp_path / "known-hosts"),
        "--key",
        str(tmp_path / "key"),
        "--report",
        str(report),
    ]
    monkeypatch.setattr("sys.argv", arguments)
    assert publish_cli.main() == 0
    assert json.loads(report.read_text(encoding="utf-8"))["ok"] is True
    monkeypatch.setattr("sys.argv", arguments)
    assert publish_cli.main() == 2
    assert len(calls) == 1


def test_pull_cli_rejects_destination_report_alias_before_transfer(tmp_path, monkeypatch):
    calls = []

    def fake_pull(**kwargs):
        calls.append(kwargs)
        return {"schema_version": PULL_SCHEMA, "ok": True}

    monkeypatch.setattr(pull_cli, "pull_remote_run_artifact", fake_pull)
    same = tmp_path / "handoff.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "pull_remote_run_artifact.py",
            "--remote-artifact",
            "artifacts/control/stage5/decision.v3.json",
            "--destination",
            str(same),
            "--producer-node-id",
            "stage5_gate",
            "--run-id",
            RUN_ID,
            "--expected-commit",
            COMMIT,
            "--remote-workspace",
            WORKSPACE,
            "--host",
            "host",
            "--port",
            "22",
            "--username",
            "root",
            "--known-hosts",
            str(tmp_path / "known-hosts"),
            "--key",
            str(tmp_path / "key"),
            "--report",
            str(same),
        ],
    )
    assert pull_cli.main() == 2
    assert calls == []
    assert not same.exists()
