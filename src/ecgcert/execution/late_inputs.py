"""Atomic capture and verification of late-arriving control inputs.

Ordinary DAG inputs exist before a node starts and are hashed on both sides of
the command.  Human/ARC control artifacts are different: a waiting command is
already running when its input arrives.  This module gives those paths an
explicit, fail-closed protocol:

* the runner writes a declaration policy outside the command workspace;
* the waiting command atomically captures a non-link input and consumes only
  the captured copy;
* the runner verifies the live source did not change, seals the capture into a
  run-owned snapshot, and binds its hashes into the result envelope; and
* resume/release re-hash the sealed snapshot independently.

The fallback in :func:`capture_late_control_input` preserves direct, non-DAG
use of the small waiting utilities.  A runner node with declared late inputs
sets the policy environment variable and will reject a command that does not
perform every declared capture.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
from typing import Any, Mapping, Sequence
import uuid

from ecgcert import lineage


POLICY_ENV = "ECGCERT_LATE_CONTROL_POLICY"
POLICY_SCHEMA = "ecg-late-control-policy/v1"
CAPTURE_SCHEMA = "ecg-late-control-capture/v1"
SNAPSHOT_SCHEMA = "ecg-late-control-snapshot/v1"
_MAX_CONTROL_INPUT_BYTES = 2 * 1024**3


class LateControlInputError(ValueError):
    """Raised when a late control input cannot be captured or authenticated."""


@dataclass(frozen=True)
class LateControlBinding:
    inputs_sha256: dict[str, str]
    snapshot_sha256: str
    artifact_bytes: int


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _safe_relative(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise LateControlInputError(f"{field} must be a non-empty relative path")
    if "\\" in value or "\x00" in value or ":" in value:
        raise LateControlInputError(f"{field} is not a portable relative path")
    parsed = PurePosixPath(value)
    if (
        parsed.is_absolute()
        or parsed.as_posix() == "."
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise LateControlInputError(f"{field} is not a safe relative path")
    return parsed.as_posix()


def _absolute_without_resolution(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _same_path(left: Path, right: Path) -> bool:
    return os.path.normcase(os.fspath(left)) == os.path.normcase(os.fspath(right))


def _require_unlinked_path(path: Path, *, root: Path, field: str) -> os.stat_result:
    """Reject links/junctions in every component and require containment."""

    absolute_root = _absolute_without_resolution(root)
    absolute = _absolute_without_resolution(path)
    try:
        relative = absolute.relative_to(absolute_root)
    except ValueError as error:
        raise LateControlInputError(f"{field} escapes its declared root") from error
    current = absolute_root
    try:
        root_stat = current.lstat()
    except OSError as error:
        raise LateControlInputError(f"cannot inspect {field} root") from error
    if stat.S_ISLNK(root_stat.st_mode):
        raise LateControlInputError(f"{field} root must not be a symlink")
    try:
        resolved_root = current.resolve(strict=True)
    except OSError as error:
        raise LateControlInputError(f"cannot resolve {field} root") from error
    if not _same_path(resolved_root, current):
        raise LateControlInputError(f"{field} root contains a link or junction")
    for part in relative.parts:
        current = current / part
        try:
            current_stat = current.lstat()
        except OSError as error:
            raise LateControlInputError(f"cannot inspect {field}: {relative.as_posix()}") from error
        if stat.S_ISLNK(current_stat.st_mode):
            raise LateControlInputError(f"{field} contains a symlink: {relative.as_posix()}")
        # On Windows a junction/reparse point may not report S_IFLNK.  Resolving
        # each existing component catches that second escape form.
        try:
            resolved = current.resolve(strict=True)
        except OSError as error:
            raise LateControlInputError(f"cannot resolve {field}") from error
        if not _same_path(resolved, current):
            raise LateControlInputError(f"{field} contains a link or junction")
    final_stat = absolute.lstat()
    if not (stat.S_ISREG(final_stat.st_mode) or stat.S_ISDIR(final_stat.st_mode)):
        raise LateControlInputError(f"{field} must be a regular file or directory")
    return final_stat


def _sha256_stream(stream: Any) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def _secure_file_sha256(path: Path) -> tuple[str, int]:
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise LateControlInputError(
            f"cannot open control file without following links: {path}"
        ) from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise LateControlInputError(f"control input is not a regular file: {path}")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            digest = _sha256_stream(stream)
        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
        if identity_before != identity_after:
            raise LateControlInputError(f"control file changed while hashing: {path}")
        return digest, int(after.st_size)
    finally:
        os.close(descriptor)


def _secure_inventory(path: Path) -> tuple[str, int, str]:
    """Hash a link-free file/tree, including empty-directory structure."""

    root = _absolute_without_resolution(path)
    root_stat = _require_unlinked_path(root, root=root.parent, field="control input")
    if stat.S_ISREG(root_stat.st_mode):
        digest, size = _secure_file_sha256(root)
        return digest, size, "file"

    entries: list[dict[str, str]] = []
    total = 0

    def walk(directory: Path, prefix: PurePosixPath) -> None:
        nonlocal total
        try:
            children = sorted(os.scandir(directory), key=lambda item: item.name)
        except OSError as error:
            raise LateControlInputError(
                f"cannot enumerate control directory: {directory}"
            ) from error
        for child in children:
            relative = prefix / child.name
            child_path = Path(child.path)
            child_stat = _require_unlinked_path(
                child_path,
                root=root,
                field=f"control member {relative.as_posix()}",
            )
            if stat.S_ISDIR(child_stat.st_mode):
                entries.append({"kind": "directory", "path": relative.as_posix()})
                walk(child_path, relative)
            elif stat.S_ISREG(child_stat.st_mode):
                digest, size = _secure_file_sha256(child_path)
                total += size
                if total > _MAX_CONTROL_INPUT_BYTES:
                    raise LateControlInputError("late control input exceeds the 2 GiB limit")
                entries.append({"kind": "file", "path": relative.as_posix(), "sha256": digest})
            else:
                raise LateControlInputError(
                    f"control directory contains a special file: {relative}"
                )

    walk(root, PurePosixPath())
    return lineage.canonical_sha256(entries), total, "directory"


def secure_control_path_sha256(path: Path | str) -> str:
    """Public link-rejecting content hash used by transfer and tests."""

    return _secure_inventory(Path(path))[0]


def secure_control_path_metadata(path: Path | str) -> tuple[str, int, str]:
    """Return digest, byte count, and kind for a link-free control artifact."""

    return _secure_inventory(Path(path))


def _copy_regular(source: Path, destination: Path) -> int:
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(source, flags)
    except OSError as error:
        raise LateControlInputError(f"cannot securely open control file: {source}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise LateControlInputError(f"control member is not a regular file: {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with (
            os.fdopen(descriptor, "rb", closefd=False) as incoming,
            destination.open("xb") as outgoing,
        ):
            shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
            outgoing.flush()
            os.fsync(outgoing.fileno())
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise LateControlInputError(f"control file changed while copying: {source}")
        return int(after.st_size)
    finally:
        os.close(descriptor)


def _copy_unlinked_tree(
    source: Path,
    destination: Path,
    *,
    kind: str,
    root: Path | None = None,
) -> int:
    root = source if root is None else root
    if kind == "file":
        return _copy_regular(source, destination)
    destination.mkdir(parents=False, exist_ok=False)
    total = 0
    for child in sorted(os.scandir(source), key=lambda item: item.name):
        child_path = Path(child.path)
        child_stat = _require_unlinked_path(
            child_path,
            root=root,
            field="control directory member",
        )
        target = destination / child.name
        if stat.S_ISDIR(child_stat.st_mode):
            total += _copy_unlinked_tree(
                child_path,
                target,
                kind="directory",
                root=root,
            )
        elif stat.S_ISREG(child_stat.st_mode):
            total += _copy_regular(child_path, target)
        else:
            raise LateControlInputError(f"control directory contains a special file: {child_path}")
        if total > _MAX_CONTROL_INPUT_BYTES:
            raise LateControlInputError("late control input exceeds the 2 GiB limit")
    return total


def write_late_control_policy(
    *,
    path: Path,
    run_id: str,
    node_id: str,
    workspace: Path,
    capture_root: Path,
    inputs: Sequence[str],
) -> None:
    """Write the runner-owned policy consumed by a waiting command."""

    if path.exists() or capture_root.exists():
        raise LateControlInputError("late-control capture staging already exists")
    capture_root.mkdir(parents=True, exist_ok=False)
    value = {
        "schema_version": POLICY_SCHEMA,
        "run_id": run_id,
        "node_id": node_id,
        "workspace": str(_absolute_without_resolution(workspace)),
        "capture_root": str(_absolute_without_resolution(capture_root)),
        "inputs": [
            {"index": index, "source_path": _safe_relative(item, field="input")}
            for index, item in enumerate(inputs)
        ],
    }
    _atomic_json(path, value)


def _load_policy(path: Path) -> dict[str, Any]:
    try:
        if path.is_symlink() or not path.is_file():
            raise LateControlInputError("late-control policy must be a regular non-link file")
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LateControlInputError("cannot read late-control policy") from error
    expected = {
        "schema_version",
        "run_id",
        "node_id",
        "workspace",
        "capture_root",
        "inputs",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise LateControlInputError("late-control policy fields are invalid")
    if value["schema_version"] != POLICY_SCHEMA:
        raise LateControlInputError("late-control policy schema is invalid")
    if not isinstance(value["inputs"], list) or not value["inputs"]:
        raise LateControlInputError("late-control policy has no declared inputs")
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(value["inputs"]):
        if not isinstance(item, dict) or set(item) != {"index", "source_path"}:
            raise LateControlInputError("late-control policy input fields are invalid")
        if item["index"] != index:
            raise LateControlInputError("late-control policy input order is invalid")
        normalized.append(
            {
                "index": index,
                "source_path": _safe_relative(item["source_path"], field="source_path"),
            }
        )
    value["inputs"] = normalized
    return value


def capture_late_control_input(
    source: Path | str,
    *,
    require_policy: bool = False,
) -> Path:
    """Capture a declared live inbox path and return the immutable copy to consume."""

    policy_value = os.environ.get(POLICY_ENV)
    if not policy_value:
        if require_policy:
            raise LateControlInputError(
                "runner-owned late-control capture policy is required"
            )
        return Path(source)
    policy_path = _absolute_without_resolution(Path(policy_value))
    policy = _load_policy(policy_path)
    workspace = _absolute_without_resolution(Path(policy["workspace"]))
    capture_root = _absolute_without_resolution(Path(policy["capture_root"]))
    if not _same_path(policy_path.parent, capture_root):
        raise LateControlInputError("late-control policy is outside its capture root")
    source_path = Path(source)
    absolute_source = _absolute_without_resolution(
        source_path if source_path.is_absolute() else workspace / source_path
    )
    try:
        relative = absolute_source.relative_to(workspace).as_posix()
    except ValueError as error:
        raise LateControlInputError("late control input escapes the run workspace") from error
    declaration = next((item for item in policy["inputs"] if item["source_path"] == relative), None)
    if declaration is None:
        raise LateControlInputError(f"undeclared late control input: {relative}")
    index = int(declaration["index"])
    payload_relative = f"payload/{index:04d}"
    record_path = capture_root / "records" / f"{index:04d}.json"
    destination = capture_root / payload_relative
    if record_path.exists() or destination.exists():
        raise LateControlInputError(f"late control input was captured more than once: {relative}")

    source_stat = _require_unlinked_path(
        absolute_source, root=workspace, field=f"late control input {relative}"
    )
    kind = "file" if stat.S_ISREG(source_stat.st_mode) else "directory"
    before_digest, before_bytes, observed_kind = _secure_inventory(absolute_source)
    if observed_kind != kind or before_bytes > _MAX_CONTROL_INPUT_BYTES:
        raise LateControlInputError("late control input type/size changed before capture")
    temporary = capture_root / f".capture-{index:04d}-{uuid.uuid4().hex}"
    try:
        copied_bytes = _copy_unlinked_tree(absolute_source, temporary, kind=kind)
        after_digest, after_bytes, after_kind = _secure_inventory(absolute_source)
        copied_digest, verified_bytes, copied_kind = _secure_inventory(temporary)
        if (
            before_digest != after_digest
            or before_digest != copied_digest
            or before_bytes != after_bytes
            or before_bytes != copied_bytes
            or before_bytes != verified_bytes
            or kind != after_kind
            or kind != copied_kind
        ):
            raise LateControlInputError("late control input changed during atomic capture")
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, destination)
    except Exception:
        if temporary.is_dir():
            shutil.rmtree(temporary, ignore_errors=True)
        elif temporary.exists():
            temporary.unlink(missing_ok=True)
        raise
    record = {
        "schema_version": CAPTURE_SCHEMA,
        "index": index,
        "source_path": relative,
        "payload_path": payload_relative,
        "kind": kind,
        "sha256": before_digest,
        "bytes": before_bytes,
        "captured_at": _utc_now(),
    }
    _atomic_json(record_path, record)
    return destination


def _load_capture_record(path: Path, *, index: int, source_path: str) -> dict[str, Any]:
    try:
        if path.is_symlink() or not path.is_file():
            raise LateControlInputError("late-control capture record is not a regular file")
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LateControlInputError("cannot read late-control capture record") from error
    expected = {
        "schema_version",
        "index",
        "source_path",
        "payload_path",
        "kind",
        "sha256",
        "bytes",
        "captured_at",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise LateControlInputError("late-control capture record fields are invalid")
    if (
        value["schema_version"] != CAPTURE_SCHEMA
        or value["index"] != index
        or value["source_path"] != source_path
        or value["payload_path"] != f"payload/{index:04d}"
        or value["kind"] not in {"file", "directory"}
        or not isinstance(value["sha256"], str)
        or len(value["sha256"]) != 64
        or isinstance(value["bytes"], bool)
        or not isinstance(value["bytes"], int)
        or value["bytes"] < 0
        or not isinstance(value["captured_at"], str)
        or not value["captured_at"]
    ):
        raise LateControlInputError("late-control capture record values are invalid")
    return value


def empty_late_control_snapshot_sha256() -> str:
    return lineage.canonical_sha256({"schema_version": SNAPSHOT_SCHEMA, "inputs": []})


def finalize_late_control_snapshot(
    *,
    policy_path: Path,
    final_root: Path,
    expected_run_id: str,
    expected_node_id: str,
    expected_inputs: Sequence[str],
) -> LateControlBinding:
    """Validate captures against their live sources and atomically seal them."""

    policy = _load_policy(policy_path)
    if policy["run_id"] != expected_run_id or policy["node_id"] != expected_node_id:
        raise LateControlInputError("late-control policy run/node identity changed")
    declared = [item["source_path"] for item in policy["inputs"]]
    if declared != list(expected_inputs):
        raise LateControlInputError("late-control policy declaration changed")
    capture_root = _absolute_without_resolution(Path(policy["capture_root"]))
    workspace = _absolute_without_resolution(Path(policy["workspace"]))
    if not _same_path(policy_path.parent, capture_root):
        raise LateControlInputError("late-control policy path is not capture-root owned")
    if final_root.exists() or final_root.is_symlink():
        raise LateControlInputError("sealed late-control snapshot already exists")
    inputs: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    total_bytes = 0
    for index, source_path in enumerate(declared):
        record = _load_capture_record(
            capture_root / "records" / f"{index:04d}.json",
            index=index,
            source_path=source_path,
        )
        payload = capture_root / record["payload_path"]
        payload_digest, payload_bytes, payload_kind = _secure_inventory(payload)
        if (
            payload_digest != record["sha256"]
            or payload_bytes != record["bytes"]
            or payload_kind != record["kind"]
        ):
            raise LateControlInputError("captured late-control payload changed")
        live = workspace / source_path
        _require_unlinked_path(live, root=workspace, field=f"live control input {source_path}")
        live_digest, live_bytes, live_kind = _secure_inventory(live)
        if (
            live_digest != payload_digest
            or live_bytes != payload_bytes
            or live_kind != payload_kind
        ):
            raise LateControlInputError(f"late control input changed after capture: {source_path}")
        hashes[source_path] = payload_digest
        total_bytes += payload_bytes
        inputs.append(record)
    records_root = capture_root / "records"
    payload_root = capture_root / "payload"
    expected_names = {f"{index:04d}.json" for index in range(len(declared))}
    if (
        not records_root.is_dir()
        or {item.name for item in records_root.iterdir()} != expected_names
        or not payload_root.is_dir()
        or {item.name for item in payload_root.iterdir()}
        != {f"{index:04d}" for index in range(len(declared))}
    ):
        raise LateControlInputError("late-control capture inventory is incomplete or ambiguous")
    policy_path.unlink()
    manifest = {
        "schema_version": SNAPSHOT_SCHEMA,
        "run_id": expected_run_id,
        "node_id": expected_node_id,
        "inputs": inputs,
        "inputs_sha256": hashes,
        "inputs_bundle_sha256": lineage.canonical_sha256(hashes),
    }
    _atomic_json(capture_root / "manifest.v1.json", manifest)
    if {item.name for item in capture_root.iterdir()} != {
        "manifest.v1.json",
        "payload",
        "records",
    }:
        raise LateControlInputError("late-control snapshot contains undeclared files")
    final_root.parent.mkdir(parents=True, exist_ok=True)
    os.replace(capture_root, final_root)
    snapshot_sha256, snapshot_bytes, snapshot_kind = _secure_inventory(final_root)
    if snapshot_kind != "directory":
        raise LateControlInputError("sealed late-control snapshot is not a directory")
    # Best-effort write protection.  Integrity never relies on permissions:
    # resume and release independently re-hash the complete snapshot.
    if os.name == "posix":
        for member in sorted(final_root.rglob("*"), reverse=True):
            try:
                member.chmod(0o555 if member.is_dir() else 0o444)
            except OSError:
                pass
        try:
            final_root.chmod(0o555)
        except OSError:
            pass
    return LateControlBinding(
        inputs_sha256=hashes,
        snapshot_sha256=snapshot_sha256,
        artifact_bytes=snapshot_bytes,
    )


def validate_late_control_snapshot(
    *,
    snapshot_root: Path,
    expected_run_id: str,
    expected_node_id: str,
    expected_inputs: Sequence[str],
) -> LateControlBinding:
    """Recompute a sealed snapshot binding for resume/release."""

    if not expected_inputs:
        if snapshot_root.exists() or snapshot_root.is_symlink():
            raise LateControlInputError("node without late inputs has a control snapshot")
        return LateControlBinding(
            inputs_sha256={},
            snapshot_sha256=empty_late_control_snapshot_sha256(),
            artifact_bytes=0,
        )
    _require_unlinked_path(
        snapshot_root,
        root=snapshot_root.parent,
        field="sealed late-control snapshot",
    )
    try:
        manifest_path = snapshot_root / "manifest.v1.json"
        manifest_stat = _require_unlinked_path(
            manifest_path,
            root=snapshot_root,
            field="late-control snapshot manifest",
        )
        if not stat.S_ISREG(manifest_stat.st_mode):
            raise LateControlInputError("late-control snapshot manifest is missing")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LateControlInputError("cannot read late-control snapshot manifest") from error
    expected_fields = {
        "schema_version",
        "run_id",
        "node_id",
        "inputs",
        "inputs_sha256",
        "inputs_bundle_sha256",
    }
    if not isinstance(manifest, dict) or set(manifest) != expected_fields:
        raise LateControlInputError("late-control snapshot manifest fields are invalid")
    if (
        manifest["schema_version"] != SNAPSHOT_SCHEMA
        or manifest["run_id"] != expected_run_id
        or manifest["node_id"] != expected_node_id
        or not isinstance(manifest["inputs"], list)
        or len(manifest["inputs"]) != len(expected_inputs)
    ):
        raise LateControlInputError("late-control snapshot identity is invalid")
    hashes: dict[str, str] = {}
    total_payload_bytes = 0
    _require_unlinked_path(
        snapshot_root / "records",
        root=snapshot_root,
        field="late-control snapshot records",
    )
    _require_unlinked_path(
        snapshot_root / "payload",
        root=snapshot_root,
        field="late-control snapshot payload",
    )
    for index, source_path in enumerate(expected_inputs):
        raw_record = manifest["inputs"][index]
        if not isinstance(raw_record, dict):
            raise LateControlInputError("late-control snapshot record is invalid")
        # Reuse the strict record validator via the retained immutable record.
        record_path = snapshot_root / "records" / f"{index:04d}.json"
        record = _load_capture_record(record_path, index=index, source_path=source_path)
        if raw_record != record:
            raise LateControlInputError("snapshot manifest and capture record disagree")
        payload_digest, payload_bytes, payload_kind = _secure_inventory(
            snapshot_root / record["payload_path"]
        )
        if (
            payload_digest != record["sha256"]
            or payload_bytes != record["bytes"]
            or payload_kind != record["kind"]
        ):
            raise LateControlInputError("sealed late-control payload changed")
        hashes[source_path] = payload_digest
        total_payload_bytes += payload_bytes
    if (
        manifest["inputs_sha256"] != hashes
        or manifest["inputs_bundle_sha256"] != lineage.canonical_sha256(hashes)
        or {item.name for item in snapshot_root.iterdir()}
        != {"manifest.v1.json", "payload", "records"}
        or {item.name for item in (snapshot_root / "records").iterdir()}
        != {f"{index:04d}.json" for index in range(len(expected_inputs))}
        or {item.name for item in (snapshot_root / "payload").iterdir()}
        != {f"{index:04d}" for index in range(len(expected_inputs))}
    ):
        raise LateControlInputError("late-control snapshot inventory/hash binding is invalid")
    snapshot_sha256, snapshot_bytes, kind = _secure_inventory(snapshot_root)
    if kind != "directory" or snapshot_bytes < total_payload_bytes:
        raise LateControlInputError("late-control snapshot size/type is invalid")
    return LateControlBinding(
        inputs_sha256=hashes,
        snapshot_sha256=snapshot_sha256,
        artifact_bytes=snapshot_bytes,
    )


__all__ = [
    "CAPTURE_SCHEMA",
    "LateControlBinding",
    "LateControlInputError",
    "POLICY_ENV",
    "POLICY_SCHEMA",
    "SNAPSHOT_SCHEMA",
    "capture_late_control_input",
    "empty_late_control_snapshot_sha256",
    "finalize_late_control_snapshot",
    "secure_control_path_sha256",
    "secure_control_path_metadata",
    "validate_late_control_snapshot",
    "write_late_control_policy",
]
