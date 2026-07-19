"""Authenticated, atomic publication of one declared control artifact to a run.

This is the only supported local-to-server path for ARC receipts and signed
stage approvals.  The SSH server key is pinned, authentication is key-only,
the destination is derived from the committed DAG declaration, and a complete
file or directory is uploaded under a random sibling name before a
non-overwriting server-side rename.  SHA-256 is recomputed locally, on the
temporary remote artifact, and again after publication.
"""

from __future__ import annotations

from datetime import datetime, timezone
import base64
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any, Callable, Mapping, Sequence
import uuid

from ecgcert import lineage
from .late_inputs import (
    CAPTURE_SCHEMA,
    LateControlInputError,
    SNAPSHOT_SCHEMA,
    empty_late_control_snapshot_sha256,
    secure_control_path_metadata,
)
from .envelope import ResultEnvelope
from .manifest import ExperimentManifest, ExperimentNode, ManifestError
from .remote import strict_ssh_client
from .runner import CHECKPOINT_SUFFIXES, RUN_STATUS_SCHEMA_VERSION


PUBLICATION_SCHEMA = "ecg-remote-control-publication/v1"
PULL_SCHEMA = "ecg-remote-run-artifact-pull/v1"
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_MAX_CONTROL_BYTES = 2 * 1024**3


class ControlPublicationError(RuntimeError):
    """Raised when a remote control publication cannot be proven atomic."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_publish_local_bytes(
    path: Path | str,
    payload: bytes,
    *,
    label: str,
) -> tuple[Path, str]:
    """Publish bytes through a sibling temporary file without replacing a name."""

    if not isinstance(payload, bytes):
        raise ControlPublicationError(f"{label} payload must be bytes")
    requested = Path(path).expanduser()
    if requested.exists() or requested.is_symlink():
        raise ControlPublicationError(f"{label} already exists; overwrite is forbidden")
    requested.parent.mkdir(parents=True, exist_ok=True)
    lexical = Path(os.path.abspath(os.fspath(requested)))
    resolved_parent = requested.parent.resolve(strict=True)
    destination = resolved_parent / requested.name
    if os.path.normcase(os.fspath(destination)) != os.path.normcase(os.fspath(lexical)):
        raise ControlPublicationError(f"{label} parent must not be a link")
    temporary = destination.parent / f".ecgcert-local-publish-{uuid.uuid4().hex}"
    created = False
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination)
            created = True
        except FileExistsError as error:
            raise ControlPublicationError(
                f"{label} appeared concurrently; overwrite is forbidden"
            ) from error
        except OSError as error:
            raise ControlPublicationError(
                f"atomic non-overwriting publication failed for {label}"
            ) from error
        try:
            digest, size, kind = secure_control_path_metadata(destination)
        except (OSError, LateControlInputError, ValueError) as error:
            raise ControlPublicationError(f"cannot verify published {label}") from error
        expected = hashlib.sha256(payload).hexdigest()
        if kind != "file" or size != len(payload) or digest != expected:
            raise ControlPublicationError(f"{label} failed SHA-256 readback")
        created = False
        return destination, digest
    finally:
        temporary.unlink(missing_ok=True)
        if created:
            destination.unlink(missing_ok=True)


def _remote_missing(error: BaseException) -> bool:
    return getattr(error, "errno", None) == 2


def _lstat_or_none(sftp: Any, path: str) -> Any | None:
    try:
        return sftp.lstat(path)
    except OSError as error:
        if _remote_missing(error):
            return None
        raise ControlPublicationError(f"cannot inspect remote path: {path}") from error


def _remote_children(sftp: Any, directory: PurePosixPath) -> list[Any]:
    try:
        children = list(sftp.listdir_attr(directory.as_posix()))
    except OSError as error:
        raise ControlPublicationError(f"cannot enumerate remote directory: {directory}") from error
    names: set[str] = set()
    for child in children:
        name = getattr(child, "filename", None)
        if (
            not isinstance(name, str)
            or name in {"", ".", ".."}
            or "/" in name
            or "\\" in name
            or "\x00" in name
            or name in names
        ):
            raise ControlPublicationError("remote directory has an unsafe/duplicate member")
        names.add(name)
    return sorted(children, key=lambda item: item.filename)


def _require_remote_directory(sftp: Any, path: PurePosixPath) -> None:
    if not path.is_absolute() or ".." in path.parts:
        raise ControlPublicationError("remote directory must be an absolute safe POSIX path")
    current = PurePosixPath("/")
    for part in path.parts[1:]:
        current /= part
        attributes = _lstat_or_none(sftp, current.as_posix())
        if attributes is None:
            raise ControlPublicationError(f"remote directory component does not exist: {current}")
        if stat.S_ISLNK(attributes.st_mode) or not stat.S_ISDIR(attributes.st_mode):
            raise ControlPublicationError(
                f"remote directory component is a link/non-directory: {current}"
            )
    try:
        normalized = PurePosixPath(sftp.normalize(path.as_posix()))
    except OSError as error:
        raise ControlPublicationError("cannot canonicalize remote directory") from error
    if normalized != path:
        raise ControlPublicationError("remote directory resolves through a link or alias")


def _mkdirs_beneath(sftp: Any, root: PurePosixPath, relative: PurePosixPath) -> None:
    current = root
    for part in relative.parts:
        current /= part
        attributes = _lstat_or_none(sftp, current.as_posix())
        if attributes is None:
            try:
                sftp.mkdir(current.as_posix(), mode=0o700)
            except OSError as error:
                raise ControlPublicationError(
                    f"cannot create remote control inbox directory: {current}"
                ) from error
            attributes = _lstat_or_none(sftp, current.as_posix())
        if (
            attributes is None
            or stat.S_ISLNK(attributes.st_mode)
            or not stat.S_ISDIR(attributes.st_mode)
        ):
            raise ControlPublicationError(
                f"remote control inbox parent is a link/non-directory: {current}"
            )


def _require_existing_beneath(sftp: Any, root: PurePosixPath, relative: PurePosixPath) -> None:
    current = root
    for index, part in enumerate(relative.parts):
        current /= part
        attributes = _lstat_or_none(sftp, current.as_posix())
        if attributes is None or stat.S_ISLNK(attributes.st_mode):
            raise ControlPublicationError(f"remote artifact path is missing or linked: {current}")
        if index < len(relative.parts) - 1 and not stat.S_ISDIR(attributes.st_mode):
            raise ControlPublicationError(f"remote artifact parent is not a directory: {current}")


def _remote_read(sftp: Any, path: PurePosixPath, *, maximum: int) -> bytes:
    attributes = _lstat_or_none(sftp, path.as_posix())
    if (
        attributes is None
        or stat.S_ISLNK(attributes.st_mode)
        or not stat.S_ISREG(attributes.st_mode)
        or attributes.st_size > maximum
    ):
        raise ControlPublicationError(f"remote metadata file is missing/unsafe: {path}")
    try:
        with sftp.open(path.as_posix(), "rb") as stream:
            raw = stream.read(maximum + 1)
    except OSError as error:
        raise ControlPublicationError(f"cannot read remote metadata file: {path}") from error
    if len(raw) > maximum:
        raise ControlPublicationError(f"remote metadata file is too large: {path}")
    return raw


def _remote_file_sha256(sftp: Any, path: PurePosixPath) -> tuple[str, int]:
    attributes = _lstat_or_none(sftp, path.as_posix())
    if (
        attributes is None
        or stat.S_ISLNK(attributes.st_mode)
        or not stat.S_ISREG(attributes.st_mode)
    ):
        raise ControlPublicationError(f"remote control member is not a regular file: {path}")
    digest = hashlib.sha256()
    observed = 0
    try:
        with sftp.open(path.as_posix(), "rb") as stream:
            while True:
                block = stream.read(1024 * 1024)
                if not block:
                    break
                observed += len(block)
                if observed > _MAX_CONTROL_BYTES:
                    raise ControlPublicationError("remote control artifact exceeds 2 GiB")
                digest.update(block)
    except OSError as error:
        raise ControlPublicationError(f"cannot hash remote control member: {path}") from error
    after = _lstat_or_none(sftp, path.as_posix())
    if (
        after is None
        or stat.S_ISLNK(after.st_mode)
        or not stat.S_ISREG(after.st_mode)
        or observed != attributes.st_size
        or observed != after.st_size
        or getattr(attributes, "st_mtime", None) != getattr(after, "st_mtime", None)
    ):
        raise ControlPublicationError(f"remote control member changed while hashing: {path}")
    return digest.hexdigest(), observed


def _remote_metadata(sftp: Any, root: PurePosixPath) -> tuple[str, int, str]:
    attributes = _lstat_or_none(sftp, root.as_posix())
    if attributes is None or stat.S_ISLNK(attributes.st_mode):
        raise ControlPublicationError("remote control artifact is missing or a symlink")
    if stat.S_ISREG(attributes.st_mode):
        digest, size = _remote_file_sha256(sftp, root)
        return digest, size, "file"
    if not stat.S_ISDIR(attributes.st_mode):
        raise ControlPublicationError("remote control artifact is a special file")
    entries: list[dict[str, str]] = []
    total = 0

    def walk(directory: PurePosixPath, prefix: PurePosixPath) -> None:
        nonlocal total
        children = _remote_children(sftp, directory)
        for child in children:
            name = child.filename
            relative = prefix / name
            child_path = directory / name
            fresh = _lstat_or_none(sftp, child_path.as_posix())
            if fresh is None or stat.S_ISLNK(fresh.st_mode):
                raise ControlPublicationError(
                    f"remote control directory contains a link/missing member: {relative}"
                )
            if stat.S_ISDIR(fresh.st_mode):
                entries.append({"kind": "directory", "path": relative.as_posix()})
                walk(child_path, relative)
            elif stat.S_ISREG(fresh.st_mode):
                digest, size = _remote_file_sha256(sftp, child_path)
                total += size
                if total > _MAX_CONTROL_BYTES:
                    raise ControlPublicationError("remote control artifact exceeds 2 GiB")
                entries.append({"kind": "file", "path": relative.as_posix(), "sha256": digest})
            else:
                raise ControlPublicationError(
                    f"remote control directory contains a special file: {relative}"
                )

    walk(root, PurePosixPath())
    return lineage.canonical_sha256(entries), total, "directory"


def _write_remote_file(sftp: Any, local: Path, remote: PurePosixPath) -> None:
    try:
        with local.open("rb") as source, sftp.open(remote.as_posix(), "x") as target:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                target.write(block)
            target.flush()
    except OSError as error:
        raise ControlPublicationError(f"cannot upload remote control file: {remote}") from error


def _upload_tree(
    sftp: Any,
    local: Path,
    remote: PurePosixPath,
    *,
    kind: str,
) -> None:
    if kind == "file":
        _write_remote_file(sftp, local, remote)
        return
    try:
        sftp.mkdir(remote.as_posix(), mode=0o700)
    except OSError as error:
        raise ControlPublicationError(
            f"cannot create temporary remote directory: {remote}"
        ) from error
    for child in sorted(os.scandir(local), key=lambda item: item.name):
        if (
            child.name in {"", ".", ".."}
            or "/" in child.name
            or "\\" in child.name
            or "\x00" in child.name
        ):
            raise ControlPublicationError("local control directory has an unsafe member name")
        attributes = child.stat(follow_symlinks=False)
        if child.is_symlink():
            raise ControlPublicationError("local control directory changed into a symlink")
        child_local = Path(child.path)
        child_remote = remote / child.name
        if stat.S_ISDIR(attributes.st_mode):
            _upload_tree(sftp, child_local, child_remote, kind="directory")
        elif stat.S_ISREG(attributes.st_mode):
            _upload_tree(sftp, child_local, child_remote, kind="file")
        else:
            raise ControlPublicationError("local control directory contains a special file")


def _remove_remote_temp(sftp: Any, root: PurePosixPath) -> None:
    attributes = _lstat_or_none(sftp, root.as_posix())
    if attributes is None:
        return
    if stat.S_ISLNK(attributes.st_mode) or stat.S_ISREG(attributes.st_mode):
        try:
            sftp.remove(root.as_posix())
        except OSError:
            pass
        return
    if not stat.S_ISDIR(attributes.st_mode):
        return
    try:
        children = sftp.listdir_attr(root.as_posix())
    except OSError:
        return
    for child in children:
        if isinstance(child.filename, str) and child.filename not in {"", ".", ".."}:
            _remove_remote_temp(sftp, root / child.filename)
    try:
        sftp.rmdir(root.as_posix())
    except OSError:
        pass


def _server_fingerprint(client: Any) -> str:
    try:
        key = client.get_transport().get_remote_server_key()
        raw = key.asbytes()
    except (AttributeError, OSError) as error:
        raise ControlPublicationError("cannot record the authenticated SSH host key") from error
    return "SHA256:" + base64.b64encode(hashlib.sha256(raw).digest()).decode("ascii").rstrip("=")


def _contains_path(parent: str, child: str) -> bool:
    parent_path = PurePosixPath(parent)
    child_path = PurePosixPath(child)
    return parent_path == child_path or parent_path in child_path.parents


def _remote_envelope_artifact_sha256(sftp: Any, root: PurePosixPath) -> str:
    """Reproduce ``runner._path_sha256`` while rejecting all remote links."""

    attributes = _lstat_or_none(sftp, root.as_posix())
    if attributes is None or stat.S_ISLNK(attributes.st_mode):
        raise ControlPublicationError("declared remote output is missing or a link")
    if stat.S_ISREG(attributes.st_mode):
        return _remote_file_sha256(sftp, root)[0]
    if not stat.S_ISDIR(attributes.st_mode):
        raise ControlPublicationError("declared remote output is a special file")
    entries: list[tuple[str, str]] = []

    def walk(directory: PurePosixPath, prefix: PurePosixPath) -> None:
        children = _remote_children(sftp, directory)
        for child in children:
            name = child.filename
            relative = prefix / name
            child_path = directory / name
            fresh = _lstat_or_none(sftp, child_path.as_posix())
            if fresh is None or stat.S_ISLNK(fresh.st_mode):
                raise ControlPublicationError("declared remote output contains a link")
            if stat.S_ISDIR(fresh.st_mode):
                walk(child_path, relative)
            elif stat.S_ISREG(fresh.st_mode):
                entries.append((relative.as_posix(), _remote_file_sha256(sftp, child_path)[0]))
            else:
                raise ControlPublicationError("declared remote output contains a special file")

    walk(root, PurePosixPath())
    return lineage.canonical_sha256(sorted(entries, key=lambda item: item[0]))


def _remote_declared_hashes(
    sftp: Any,
    workspace: PurePosixPath,
    paths: Sequence[str],
) -> dict[str, str]:
    """Reproduce ``runner.declared_path_hashes`` for an exact path set.

    The runner's directory digest is a canonical list of ``(relative file,
    SHA-256)`` pairs.  The remote implementation deliberately rejects links
    and special files, which is stricter than following a link and prevents an
    envelope-covered path from escaping the authenticated run workspace.
    """

    hashes: dict[str, str] = {}
    for relative in paths:
        path = PurePosixPath(relative)
        _require_existing_beneath(sftp, workspace, path)
        hashes[relative] = _remote_envelope_artifact_sha256(sftp, workspace / path)
    return hashes


def _remote_checkpoint_hashes(
    sftp: Any,
    workspace: PurePosixPath,
    paths: Sequence[str],
) -> dict[str, str]:
    """Reproduce ``runner.collect_checkpoint_hashes`` without following links."""

    checkpoints: dict[str, str] = {}

    def walk(
        directory: PurePosixPath,
        *,
        declared: str,
        prefix: PurePosixPath,
    ) -> None:
        children = _remote_children(sftp, directory)
        for child in children:
            name = child.filename
            relative = prefix / name
            child_path = directory / name
            fresh = _lstat_or_none(sftp, child_path.as_posix())
            if fresh is None or stat.S_ISLNK(fresh.st_mode):
                raise ControlPublicationError("remote checkpoint root contains a link")
            if stat.S_ISDIR(fresh.st_mode):
                walk(child_path, declared=declared, prefix=relative)
            elif stat.S_ISREG(fresh.st_mode):
                if relative.suffix.lower() in CHECKPOINT_SUFFIXES:
                    key = f"{declared}/{relative.as_posix()}"
                    checkpoints[key] = _remote_file_sha256(sftp, child_path)[0]
            else:
                raise ControlPublicationError(
                    "remote checkpoint root contains a special file"
                )

    for relative in paths:
        path = PurePosixPath(relative)
        _require_existing_beneath(sftp, workspace, path)
        artifact = workspace / path
        attributes = _lstat_or_none(sftp, artifact.as_posix())
        if attributes is None or stat.S_ISLNK(attributes.st_mode):
            raise ControlPublicationError("remote checkpoint root is missing or linked")
        if stat.S_ISREG(attributes.st_mode):
            if path.suffix.lower() in CHECKPOINT_SUFFIXES:
                checkpoints[relative] = _remote_file_sha256(sftp, artifact)[0]
        elif stat.S_ISDIR(attributes.st_mode):
            walk(artifact, declared=relative, prefix=PurePosixPath())
        else:
            raise ControlPublicationError("remote checkpoint root is a special file")
    return checkpoints


def _remote_json_object(
    sftp: Any,
    path: PurePosixPath,
    *,
    label: str,
    maximum: int = 16 * 1024**2,
) -> tuple[dict[str, Any], bytes]:
    raw = _remote_read(sftp, path, maximum=maximum)
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ControlPublicationError(f"{label} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ControlPublicationError(f"{label} must be a JSON object")
    return value, raw


def _remote_directory_names(sftp: Any, path: PurePosixPath) -> set[str]:
    attributes = _lstat_or_none(sftp, path.as_posix())
    if (
        attributes is None
        or stat.S_ISLNK(attributes.st_mode)
        or not stat.S_ISDIR(attributes.st_mode)
    ):
        raise ControlPublicationError(f"remote inventory root is not a directory: {path}")
    children = _remote_children(sftp, path)
    names = {child.filename for child in children}
    for child in children:
        name = child.filename
        fresh = _lstat_or_none(sftp, (path / name).as_posix())
        if fresh is None or stat.S_ISLNK(fresh.st_mode):
            raise ControlPublicationError("remote inventory contains a missing/linked member")
    return names


def _validate_remote_late_snapshot(
    *,
    sftp: Any,
    workspace: PurePosixPath,
    run_id: str,
    node: ExperimentNode,
    envelope: ResultEnvelope,
) -> None:
    """Recompute the runner-owned late-input snapshot bound by an envelope."""

    snapshot_root = workspace.parent / "control-inputs" / node.id
    if not node.late_control_inputs:
        if _lstat_or_none(sftp, snapshot_root.as_posix()) is not None:
            raise ControlPublicationError("node without late inputs has a remote snapshot")
        if (
            envelope.late_control_inputs_sha256
            or envelope.late_control_snapshot_sha256
            != empty_late_control_snapshot_sha256()
        ):
            raise ControlPublicationError("empty late-control envelope binding is invalid")
        return

    digest, _snapshot_bytes, kind = _remote_metadata(sftp, snapshot_root)
    if kind != "directory" or digest != envelope.late_control_snapshot_sha256:
        raise ControlPublicationError("remote late-control snapshot hash/type is invalid")
    if _remote_directory_names(sftp, snapshot_root) != {
        "manifest.v1.json",
        "payload",
        "records",
    }:
        raise ControlPublicationError("remote late-control snapshot inventory is ambiguous")
    manifest, _raw = _remote_json_object(
        sftp,
        snapshot_root / "manifest.v1.json",
        label="remote late-control snapshot manifest",
    )
    expected_manifest_fields = {
        "schema_version",
        "run_id",
        "node_id",
        "inputs",
        "inputs_sha256",
        "inputs_bundle_sha256",
    }
    if (
        set(manifest) != expected_manifest_fields
        or manifest.get("schema_version") != SNAPSHOT_SCHEMA
        or manifest.get("run_id") != run_id
        or manifest.get("node_id") != node.id
        or not isinstance(manifest.get("inputs"), list)
        or len(manifest["inputs"]) != len(node.late_control_inputs)
    ):
        raise ControlPublicationError("remote late-control snapshot identity is invalid")
    expected_names = {f"{index:04d}" for index in range(len(node.late_control_inputs))}
    if (
        _remote_directory_names(sftp, snapshot_root / "records")
        != {f"{name}.json" for name in expected_names}
        or _remote_directory_names(sftp, snapshot_root / "payload") != expected_names
    ):
        raise ControlPublicationError("remote late-control snapshot members are incomplete")

    hashes: dict[str, str] = {}
    for index, source_path in enumerate(node.late_control_inputs):
        record, _record_raw = _remote_json_object(
            sftp,
            snapshot_root / "records" / f"{index:04d}.json",
            label="remote late-control capture record",
        )
        expected_record_fields = {
            "schema_version",
            "index",
            "source_path",
            "payload_path",
            "kind",
            "sha256",
            "bytes",
            "captured_at",
        }
        if (
            set(record) != expected_record_fields
            or record.get("schema_version") != CAPTURE_SCHEMA
            or record.get("index") != index
            or record.get("source_path") != source_path
            or record.get("payload_path") != f"payload/{index:04d}"
            or record.get("kind") not in {"file", "directory"}
            or not isinstance(record.get("sha256"), str)
            or not _HEX64.fullmatch(record["sha256"])
            or isinstance(record.get("bytes"), bool)
            or not isinstance(record.get("bytes"), int)
            or record["bytes"] < 0
            or not isinstance(record.get("captured_at"), str)
            or not record["captured_at"]
            or manifest["inputs"][index] != record
        ):
            raise ControlPublicationError("remote late-control capture record is invalid")
        payload_digest, payload_bytes, payload_kind = _remote_metadata(
            sftp, snapshot_root / "payload" / f"{index:04d}"
        )
        if (
            payload_digest != record["sha256"]
            or payload_bytes != record["bytes"]
            or payload_kind != record["kind"]
        ):
            raise ControlPublicationError("remote late-control payload changed")
        hashes[source_path] = payload_digest
    if (
        manifest.get("inputs_sha256") != hashes
        or manifest.get("inputs_bundle_sha256") != lineage.canonical_sha256(hashes)
        or envelope.late_control_inputs_sha256 != hashes
    ):
        raise ControlPublicationError("remote late-control hash binding is invalid")


def _validate_remote_envelope_contract(
    *,
    sftp: Any,
    workspace: PurePosixPath,
    run_id: str,
    manifest: ExperimentManifest,
    status: Mapping[str, Any],
    producer: ExperimentNode,
    envelope: ResultEnvelope,
) -> dict[str, str]:
    """Recompute every remotely observable field of a successful envelope."""

    expected_python = status["python_executable"]
    expected_argv = [
        expected_python if token == "{python}" else token for token in producer.command
    ]
    _validate_remote_late_snapshot(
        sftp=sftp,
        workspace=workspace,
        run_id=run_id,
        node=producer,
        envelope=envelope,
    )
    try:
        expected_config = (
            producer.config_sha256(
                late_control_inputs_sha256=envelope.late_control_inputs_sha256
            )
            if producer.late_control_inputs
            else producer.config_sha256()
        )
    except ManifestError as error:
        raise ControlPublicationError("producer late-control config is invalid") from error
    environment_lock = status.get("environment_lock")
    node_status = status.get("nodes", {}).get(producer.id)
    if (
        not isinstance(environment_lock, Mapping)
        or not isinstance(environment_lock.get("lock_sha256"), str)
        or not _HEX64.fullmatch(environment_lock["lock_sha256"])
        or envelope.run_id != run_id
        or envelope.node_id != producer.id
        or envelope.commit != status["commit"]
        or envelope.argv != expected_argv
        or envelope.seed != producer.seed
        or envelope.config_sha256 != expected_config
        or envelope.env_sha256 != status.get("environment_sha256")
        or envelope.environment_lock_sha256 != environment_lock["lock_sha256"]
        or envelope.source_sha256 != status.get("source_snapshot_sha256")
        or not isinstance(node_status, Mapping)
        or node_status.get("state") != "succeeded"
        or node_status.get("exit_code") != 0
        or node_status.get("timed_out") is not False
        or node_status.get("started_at") != envelope.started_at
        or node_status.get("finished_at") != envelope.finished_at
        or set(envelope.outputs_sha256) != set(producer.outputs)
        or set(envelope.upstream_sha256) != set(producer.deps)
    ):
        raise ControlPublicationError("producer envelope identity/config contract is invalid")

    input_hashes = _remote_declared_hashes(sftp, workspace, producer.inputs)
    output_hashes = _remote_declared_hashes(sftp, workspace, producer.outputs)
    if output_hashes != envelope.outputs_sha256:
        raise ControlPublicationError("remote producer output SHA differs from its envelope")
    checkpoints = _remote_checkpoint_hashes(
        sftp, workspace, (*producer.inputs, *producer.outputs)
    )
    if checkpoints != envelope.checkpoint_sha256:
        raise ControlPublicationError("remote producer checkpoint SHA contract is invalid")
    data_inputs = {
        relative: digest
        for relative, digest in input_hashes.items()
        if relative not in checkpoints
    }
    effective_data_inputs = dict(data_inputs)
    effective_data_inputs.update(
        {
            f"late-control:{relative}": digest
            for relative, digest in envelope.late_control_inputs_sha256.items()
        }
    )
    if envelope.data_sha256 != lineage.canonical_sha256(effective_data_inputs):
        raise ControlPublicationError("remote producer input-data SHA contract is invalid")
    split_material: dict[str, Any] = {
        "seed": producer.seed,
        "argv": expected_argv,
        "inputs": input_hashes,
        "deps": list(producer.deps),
    }
    if producer.late_control_inputs:
        split_material["late_control_inputs_sha256"] = (
            envelope.late_control_inputs_sha256
        )
        split_material["late_control_snapshot_sha256"] = (
            envelope.late_control_snapshot_sha256
        )
    if envelope.split_sha256 != lineage.canonical_sha256(split_material):
        raise ControlPublicationError("remote producer input/split SHA contract is invalid")

    expected_upstream: dict[str, str] = {}
    nodes = manifest.by_id()
    for dependency_id in producer.deps:
        dependency_path = workspace.parent / "envelopes" / f"{dependency_id}.json"
        dependency_value, dependency_raw = _remote_json_object(
            sftp, dependency_path, label="remote dependency result envelope"
        )
        try:
            dependency = ResultEnvelope.from_dict(dependency_value)
        except ValueError as error:
            raise ControlPublicationError("remote dependency result envelope is invalid") from error
        if (
            dependency.run_id != run_id
            or dependency.node_id != dependency_id
            or dependency.commit != envelope.commit
            or set(dependency.outputs_sha256) != set(nodes[dependency_id].outputs)
        ):
            raise ControlPublicationError("remote dependency envelope identity is invalid")
        expected_upstream[dependency_id] = hashlib.sha256(dependency_raw).hexdigest()
    if envelope.upstream_sha256 != expected_upstream:
        raise ControlPublicationError("remote producer upstream envelope binding is invalid")
    return output_hashes


def _verify_remote_run(
    *,
    sftp: Any,
    workspace: PurePosixPath,
    run_id: str,
    manifest: ExperimentManifest,
    manifest_file: Path,
) -> Mapping[str, Any]:
    _require_remote_directory(sftp, workspace)
    remote_manifest_raw = _remote_read(
        sftp,
        workspace / "scripts" / "experiment_manifest.yaml",
        maximum=16 * 1024**2,
    )
    if (
        hashlib.sha256(remote_manifest_raw).digest()
        != hashlib.sha256(manifest_file.read_bytes()).digest()
    ):
        raise ControlPublicationError("remote run manifest differs from the local declaration")
    status, _status_raw = _remote_json_object(
        sftp,
        workspace.parent / "status.json",
        label="remote run status",
    )
    profile = status.get("profile")
    resource = status.get("resource")
    environment_lock = status.get("environment_lock")
    mutable_roots = status.get("mutable_workspace_roots")
    try:
        selected_ids = {
            node.id for node in manifest.select(profile, resource)  # type: ignore[arg-type]
        }
    except (ManifestError, TypeError, ValueError) as error:
        raise ControlPublicationError("remote run profile/resource selection is invalid") from error
    nodes = status.get("nodes")
    if (
        status.get("schema_version") != RUN_STATUS_SCHEMA_VERSION
        or status.get("run_id") != run_id
        or status.get("control_root") != workspace.parent.parent.as_posix()
        or status.get("manifest_sha256") != manifest.sha256()
        or not isinstance(status.get("commit"), str)
        or not _HEX40.fullmatch(status["commit"])
        or not isinstance(status.get("python_executable"), str)
        or not status["python_executable"]
        or not PurePosixPath(status["python_executable"]).is_absolute()
        or not isinstance(status.get("environment_sha256"), str)
        or not _HEX64.fullmatch(status["environment_sha256"])
        or not isinstance(status.get("source_snapshot_sha256"), str)
        or not _HEX64.fullmatch(status["source_snapshot_sha256"])
        or not isinstance(environment_lock, Mapping)
        or not isinstance(environment_lock.get("lock_sha256"), str)
        or not _HEX64.fullmatch(environment_lock["lock_sha256"])
        or environment_lock.get("python_executable")
        != status.get("python_executable")
        or environment_lock.get("ok") is not True
        or environment_lock.get("mismatches") != []
        or not isinstance(mutable_roots, list)
        or not all(
            isinstance(value, str)
            and value
            and not PurePosixPath(value).is_absolute()
            and ".." not in PurePosixPath(value).parts
            and "\\" not in value
            and ":" not in value
            for value in mutable_roots
        )
        or not isinstance(nodes, Mapping)
        or set(nodes) != selected_ids
    ):
        raise ControlPublicationError("remote run identity does not match the manifest")
    run_identity = {
        "run_id": run_id,
        "control_root": workspace.parent.parent.as_posix(),
        "profile": profile,
        "resource": resource,
        "environment_lock": dict(environment_lock),
        "environment_sha256": status["environment_sha256"],
        "python_executable": status["python_executable"],
        "manifest_sha256": manifest.sha256(),
        "commit": status["commit"],
        "source_snapshot_sha256": status["source_snapshot_sha256"],
        "mutable_workspace_roots": mutable_roots,
    }
    if status.get("run_identity_sha256") != lineage.canonical_sha256(run_identity):
        raise ControlPublicationError("remote run identity hash is invalid")
    return status


def _safe_remote_workspace(value: str, *, run_id: str) -> PurePosixPath:
    workspace = PurePosixPath(value)
    if (
        not workspace.is_absolute()
        or ".." in workspace.parts
        or workspace.name != "workspace"
        or workspace.parent.name != run_id
    ):
        raise ControlPublicationError(
            "remote workspace must be the specified absolute <run_id>/workspace path"
        )
    return workspace


def publish_remote_control(
    *,
    local_path: Path | str,
    manifest_path: Path | str,
    node_id: str,
    run_id: str,
    expected_commit: str,
    remote_workspace: str,
    host: str,
    port: int,
    username: str,
    known_hosts: Path | str,
    key_path: Path | str,
    client_factory: Callable[..., Any] = strict_ssh_client,
) -> dict[str, Any]:
    """Publish one manifest-declared late control input without overwrite."""

    if not _SAFE_ID.fullmatch(run_id) or not _SAFE_ID.fullmatch(node_id):
        raise ControlPublicationError("run_id and node_id must be safe identifiers")
    if not isinstance(expected_commit, str) or not _HEX40.fullmatch(expected_commit):
        raise ControlPublicationError("expected_commit must be a full lowercase git SHA")
    if not host or not username or not 1 <= port <= 65535:
        raise ControlPublicationError("host, username, and port are required")
    source = Path(local_path).expanduser()
    manifest_file = Path(manifest_path).expanduser().resolve(strict=True)
    known_hosts_file = Path(known_hosts).expanduser().resolve(strict=True)
    private_key = Path(key_path).expanduser().resolve(strict=True)
    if source.is_symlink():
        raise ControlPublicationError("local control artifact must not be a symlink")
    try:
        source = source.resolve(strict=True)
        local_digest, local_bytes, local_kind = secure_control_path_metadata(source)
    except (OSError, LateControlInputError, ValueError) as error:
        raise ControlPublicationError(f"local control artifact is unsafe: {error}") from error
    if local_bytes > _MAX_CONTROL_BYTES:
        raise ControlPublicationError("local control artifact exceeds 2 GiB")
    if not known_hosts_file.is_file() or known_hosts_file.stat().st_size == 0:
        raise ControlPublicationError("known_hosts must be an explicit non-empty file")
    if not private_key.is_file():
        raise ControlPublicationError("private key path must identify a file")
    manifest = ExperimentManifest.from_path(manifest_file)
    try:
        node = manifest.by_id()[node_id]
    except KeyError as error:
        raise ControlPublicationError("node_id is absent from the experiment manifest") from error
    if len(node.late_control_inputs) != 1:
        raise ControlPublicationError(
            "publication node must declare exactly one late control input"
        )
    target_relative = PurePosixPath(node.late_control_inputs[0])
    if (
        target_relative.is_absolute()
        or ".." in target_relative.parts
        or target_relative.parts[:2] != ("artifacts", "gates")
    ):
        raise ControlPublicationError("manifest late control target is unsafe")
    workspace = _safe_remote_workspace(remote_workspace, run_id=run_id)

    client = client_factory(known_hosts=str(known_hosts_file))
    connect_args = {
        "hostname": host,
        "port": port,
        "username": username,
        "key_filename": str(private_key),
        "look_for_keys": False,
        "allow_agent": False,
        "timeout": 30,
        "banner_timeout": 30,
        "auth_timeout": 30,
    }
    temporary: PurePosixPath | None = None
    sftp = None
    try:
        client.connect(**connect_args)
        host_fingerprint = _server_fingerprint(client)
        sftp = client.open_sftp()
        status = _verify_remote_run(
            sftp=sftp,
            workspace=workspace,
            run_id=run_id,
            manifest=manifest,
            manifest_file=manifest_file,
        )
        if status.get("commit") != expected_commit:
            raise ControlPublicationError("remote run commit differs from expected_commit")
        node_status = status.get("nodes", {}).get(node_id) if isinstance(status, Mapping) else None
        if (
            not isinstance(status, Mapping)
            or status.get("state") != "running"
            or status.get("exit_code") is not None
            or not isinstance(node_status, Mapping)
            or node_status.get("state") not in {"pending", "running"}
        ):
            raise ControlPublicationError("remote run identity/node state rejects this control")

        _mkdirs_beneath(sftp, workspace, target_relative.parent)
        _require_remote_directory(sftp, workspace / target_relative.parent)
        target = workspace / target_relative
        if _lstat_or_none(sftp, target.as_posix()) is not None:
            raise ControlPublicationError(
                "remote control target already exists; overwrite is forbidden"
            )
        temporary = target.parent / f".ecgcert-upload-{uuid.uuid4().hex}"
        if _lstat_or_none(sftp, temporary.as_posix()) is not None:
            raise ControlPublicationError("random remote temporary path already exists")
        _upload_tree(sftp, source, temporary, kind=local_kind)
        temporary_digest, temporary_bytes, temporary_kind = _remote_metadata(sftp, temporary)
        try:
            after_digest, after_bytes, after_kind = secure_control_path_metadata(source)
        except (OSError, LateControlInputError, ValueError) as error:
            raise ControlPublicationError("local control artifact changed during upload") from error
        if (after_digest, after_bytes, after_kind) != (local_digest, local_bytes, local_kind) or (
            temporary_digest,
            temporary_bytes,
            temporary_kind,
        ) != (local_digest, local_bytes, local_kind):
            raise ControlPublicationError(
                "temporary remote SHA/size/type differs from local source"
            )
        try:
            # SSH_FXP_RENAME, unlike the OpenSSH posix-rename extension, must
            # fail when the destination exists.  This is the no-overwrite
            # atomic commit point for the complete file or directory.
            sftp.rename(temporary.as_posix(), target.as_posix())
        except OSError as error:
            raise ControlPublicationError(
                "atomic non-overwriting remote publication failed"
            ) from error
        temporary = None
        final_digest, final_bytes, final_kind = _remote_metadata(sftp, target)
        if (final_digest, final_bytes, final_kind) != (
            local_digest,
            local_bytes,
            local_kind,
        ):
            raise ControlPublicationError("published remote control failed SHA readback")
        return {
            "schema_version": PUBLICATION_SCHEMA,
            "published_at": _utc_now(),
            "run_id": run_id,
            "node_id": node_id,
            "commit": status["commit"],
            "expected_commit": expected_commit,
            "run_identity_sha256": status["run_identity_sha256"],
            "manifest_sha256": manifest.sha256(),
            "manifest_file_sha256": _file_sha256(manifest_file),
            "source_sha256": local_digest,
            "source_bytes": local_bytes,
            "source_kind": local_kind,
            "remote_workspace": workspace.as_posix(),
            "remote_target": target_relative.as_posix(),
            "remote_readback_sha256": final_digest,
            "transport": {
                "protocol": "ssh-sftp",
                "host": host,
                "port": port,
                "username": username,
                "strict_known_hosts": True,
                "known_hosts_sha256": _file_sha256(known_hosts_file),
                "key_only": True,
                "look_for_keys": False,
                "allow_agent": False,
                "server_host_key_fingerprint": host_fingerprint,
                "atomic_non_overwrite_rename": True,
                "rename_operation": "SSH_FXP_RENAME",
                "posix_rename_extension_used": False,
                "post_publish_sha256_readback": True,
            },
        }
    except ControlPublicationError:
        raise
    except (OSError, ValueError) as error:
        raise ControlPublicationError(f"remote control publication failed: {error}") from error
    finally:
        if sftp is not None and temporary is not None:
            _remove_remote_temp(sftp, temporary)
        if sftp is not None:
            try:
                sftp.close()
            except OSError:
                pass
        try:
            client.close()
        except OSError:
            pass


def pull_remote_run_artifact(
    *,
    remote_artifact: str,
    destination: Path | str,
    manifest_path: Path | str,
    producer_node_id: str,
    run_id: str,
    expected_commit: str,
    remote_workspace: str,
    host: str,
    port: int,
    username: str,
    known_hosts: Path | str,
    key_path: Path | str,
    client_factory: Callable[..., Any] = strict_ssh_client,
) -> dict[str, Any]:
    """Download one successful, envelope-covered run artifact atomically.

    The critical handoffs are JSON files (stage decisions and ARC operator
    responses), so this deliberately accepts only a regular-file source.  A
    same-directory hard-link commit provides an atomic no-replace operation on
    every supported local platform.
    """

    if not _SAFE_ID.fullmatch(run_id) or not _SAFE_ID.fullmatch(producer_node_id):
        raise ControlPublicationError("run_id and producer_node_id must be safe identifiers")
    if not isinstance(expected_commit, str) or not _HEX40.fullmatch(expected_commit):
        raise ControlPublicationError("expected_commit must be a full lowercase git SHA")
    if not host or not username or not 1 <= port <= 65535:
        raise ControlPublicationError("host, username, and port are required")
    if (
        not isinstance(remote_artifact, str)
        or not remote_artifact
        or "\\" in remote_artifact
        or "\x00" in remote_artifact
        or ":" in remote_artifact
    ):
        raise ControlPublicationError("remote artifact must be a safe relative POSIX path")
    artifact_relative = PurePosixPath(remote_artifact)
    if (
        artifact_relative.is_absolute()
        or ".." in artifact_relative.parts
        or artifact_relative.parts[0:1] != ("artifacts",)
    ):
        raise ControlPublicationError("remote artifact must stay beneath artifacts")
    workspace = _safe_remote_workspace(remote_workspace, run_id=run_id)
    manifest_file = Path(manifest_path).expanduser().resolve(strict=True)
    known_hosts_file = Path(known_hosts).expanduser().resolve(strict=True)
    private_key = Path(key_path).expanduser().resolve(strict=True)
    if not known_hosts_file.is_file() or known_hosts_file.stat().st_size == 0:
        raise ControlPublicationError("known_hosts must be an explicit non-empty file")
    if not private_key.is_file():
        raise ControlPublicationError("private key path must identify a file")
    manifest = ExperimentManifest.from_path(manifest_file)
    try:
        producer = manifest.by_id()[producer_node_id]
    except KeyError as error:
        raise ControlPublicationError("producer node is absent from the manifest") from error
    covering_outputs = [
        output for output in producer.outputs if _contains_path(output, remote_artifact)
    ]
    if len(covering_outputs) != 1:
        raise ControlPublicationError(
            "remote artifact is not uniquely covered by a declared producer output"
        )
    destination_path = Path(destination).expanduser()
    if destination_path.exists() or destination_path.is_symlink():
        raise ControlPublicationError("local destination already exists; overwrite is forbidden")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_parent = destination_path.parent.resolve(strict=True)
    lexical_destination = Path(os.path.abspath(os.fspath(destination_path)))
    destination_path = resolved_parent / destination_path.name
    if os.path.normcase(os.fspath(destination_path)) != os.path.normcase(
        os.fspath(lexical_destination)
    ):
        raise ControlPublicationError("local destination parent must not be a link")
    temporary = destination_path.parent / f".ecgcert-download-{uuid.uuid4().hex}"
    if temporary.exists() or temporary.is_symlink():
        raise ControlPublicationError("random local temporary path already exists")

    client = client_factory(known_hosts=str(known_hosts_file))
    connect_args = {
        "hostname": host,
        "port": port,
        "username": username,
        "key_filename": str(private_key),
        "look_for_keys": False,
        "allow_agent": False,
        "timeout": 30,
        "banner_timeout": 30,
        "auth_timeout": 30,
    }
    sftp = None
    destination_created = False
    try:
        client.connect(**connect_args)
        host_fingerprint = _server_fingerprint(client)
        sftp = client.open_sftp()
        status = _verify_remote_run(
            sftp=sftp,
            workspace=workspace,
            run_id=run_id,
            manifest=manifest,
            manifest_file=manifest_file,
        )
        if status.get("commit") != expected_commit:
            raise ControlPublicationError("remote run commit differs from expected_commit")
        node_status = status.get("nodes", {}).get(producer_node_id)
        if (
            not isinstance(node_status, Mapping)
            or node_status.get("state") != "succeeded"
            or node_status.get("exit_code") != 0
        ):
            raise ControlPublicationError("producer node is not succeeded/0")
        envelope_path = workspace.parent / "envelopes" / f"{producer_node_id}.json"
        envelope_raw = _remote_read(sftp, envelope_path, maximum=16 * 1024**2)
        try:
            envelope_value = json.loads(envelope_raw.decode("utf-8"))
            envelope = ResultEnvelope.from_dict(envelope_value)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise ControlPublicationError("producer result envelope is invalid") from error
        actual_outputs = _validate_remote_envelope_contract(
            sftp=sftp,
            workspace=workspace,
            run_id=run_id,
            manifest=manifest,
            status=status,
            producer=producer,
            envelope=envelope,
        )

        _require_existing_beneath(sftp, workspace, artifact_relative)
        source = workspace / artifact_relative
        before_digest, before_bytes = _remote_file_sha256(sftp, source)
        try:
            with sftp.open(source.as_posix(), "rb") as incoming, temporary.open("xb") as outgoing:
                observed = 0
                while True:
                    block = incoming.read(1024 * 1024)
                    if not block:
                        break
                    observed += len(block)
                    if observed > _MAX_CONTROL_BYTES:
                        raise ControlPublicationError("remote handoff artifact exceeds 2 GiB")
                    outgoing.write(block)
                outgoing.flush()
                os.fsync(outgoing.fileno())
        except OSError as error:
            raise ControlPublicationError("cannot download remote handoff artifact") from error
        try:
            local_digest, local_bytes, local_kind = secure_control_path_metadata(temporary)
        except (OSError, LateControlInputError, ValueError) as error:
            raise ControlPublicationError("downloaded temporary artifact is unsafe") from error
        after_digest, after_bytes = _remote_file_sha256(sftp, source)
        if (
            local_kind != "file"
            or (before_digest, before_bytes) != (after_digest, after_bytes)
            or (local_digest, local_bytes) != (before_digest, before_bytes)
        ):
            raise ControlPublicationError(
                "remote source changed during download or local SHA differs"
            )
        envelope_raw_after = _remote_read(sftp, envelope_path, maximum=16 * 1024**2)
        if envelope_raw_after != envelope_raw:
            raise ControlPublicationError("producer result envelope changed during download")
        status_after = _verify_remote_run(
            sftp=sftp,
            workspace=workspace,
            run_id=run_id,
            manifest=manifest,
            manifest_file=manifest_file,
        )
        node_status_after = status_after.get("nodes", {}).get(producer_node_id)
        if (
            status_after.get("commit") != status.get("commit")
            or status_after.get("run_identity_sha256")
            != status.get("run_identity_sha256")
            or not isinstance(node_status_after, Mapping)
            or node_status_after.get("state") != "succeeded"
            or node_status_after.get("exit_code") != 0
        ):
            raise ControlPublicationError("remote run/producer identity changed during download")
        actual_outputs_after = _validate_remote_envelope_contract(
            sftp=sftp,
            workspace=workspace,
            run_id=run_id,
            manifest=manifest,
            status=status_after,
            producer=producer,
            envelope=envelope,
        )
        if actual_outputs_after != actual_outputs:
            raise ControlPublicationError("remote producer outputs changed during download")
        try:
            # Hard-link creation is an atomic create-if-absent operation.  Both
            # names are in one directory/filesystem; unlinking the temporary
            # name after success cannot change the published bytes.
            os.link(temporary, destination_path)
            destination_created = True
        except FileExistsError as error:
            raise ControlPublicationError(
                "local destination appeared concurrently; overwrite is forbidden"
            ) from error
        except OSError as error:
            raise ControlPublicationError(
                "atomic non-overwriting local publication failed"
            ) from error
        temporary.unlink()
        final_digest, final_bytes, final_kind = secure_control_path_metadata(destination_path)
        if final_kind != "file" or (final_digest, final_bytes) != (before_digest, before_bytes):
            raise ControlPublicationError("local published artifact failed SHA readback")
        destination_created = False
        return {
            "schema_version": PULL_SCHEMA,
            "pulled_at": _utc_now(),
            "run_id": run_id,
            "producer_node_id": producer_node_id,
            "commit": envelope.commit,
            "expected_commit": expected_commit,
            "run_identity_sha256": status_after["run_identity_sha256"],
            "manifest_sha256": manifest.sha256(),
            "manifest_file_sha256": _file_sha256(manifest_file),
            "producer_envelope_sha256": hashlib.sha256(envelope_raw).hexdigest(),
            "producer_outputs_sha256": actual_outputs,
            "remote_artifact": artifact_relative.as_posix(),
            "remote_sha256": before_digest,
            "artifact_bytes": before_bytes,
            "local_destination": str(destination_path),
            "local_readback_sha256": final_digest,
            "transport": {
                "protocol": "ssh-sftp",
                "host": host,
                "port": port,
                "username": username,
                "strict_known_hosts": True,
                "known_hosts_sha256": _file_sha256(known_hosts_file),
                "key_only": True,
                "look_for_keys": False,
                "allow_agent": False,
                "server_host_key_fingerprint": host_fingerprint,
                "remote_source_stability_verified": True,
                "producer_output_contract_reverified_after_download": True,
                "atomic_local_no_overwrite": True,
                "post_publish_sha256_readback": True,
            },
        }
    except ControlPublicationError:
        raise
    except (OSError, ValueError) as error:
        raise ControlPublicationError(f"remote artifact pull failed: {error}") from error
    finally:
        if temporary.exists():
            try:
                temporary.unlink()
            except OSError:
                pass
        if destination_created and destination_path.exists():
            try:
                destination_path.unlink()
            except OSError:
                pass
        if sftp is not None:
            try:
                sftp.close()
            except OSError:
                pass
        try:
            client.close()
        except OSError:
            pass


__all__ = [
    "ControlPublicationError",
    "PUBLICATION_SCHEMA",
    "PULL_SCHEMA",
    "atomic_publish_local_bytes",
    "publish_remote_control",
    "pull_remote_run_artifact",
]
