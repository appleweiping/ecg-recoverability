"""Isolated local DAG execution with structured status and result envelopes."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import time
from typing import Any, Iterable

from ecgcert import lineage
from .budget import (
    BudgetError,
    BudgetLease,
    SETTLEMENT_SCHEMA,
    validate_global_budget_ledger,
    validate_settlement_snapshot,
)
from .environment import (
    EnvironmentLockError,
    LockedEnvironmentReport,
    lock_relative_path,
    require_locked_environment,
)
from .envelope import ResultEnvelope, SCHEMA_VERSION
from .late_inputs import (
    POLICY_ENV,
    LateControlBinding,
    LateControlInputError,
    empty_late_control_snapshot_sha256,
    finalize_late_control_snapshot,
    validate_late_control_snapshot,
    write_late_control_policy,
)
from .manifest import ExperimentManifest, ExperimentNode

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
# Linear reconstructors serialize their fitted state as NumPy archives.  They
# are model checkpoints, not anonymous data artifacts, and must receive the
# same independently recomputed lineage treatment as neural checkpoints.
CHECKPOINT_SUFFIXES = frozenset(
    {".pt", ".pth", ".ckpt", ".safetensors", ".npz"}
)
MAX_GPU_HOURS = 500.0
MAX_CPU_CORE_HOURS = 4_000.0
MAX_ARTIFACT_BYTES = 100 * 1024**3
RESERVED_GPU_HOURS = 100.0
_CONTROLLED_MUTABLE_ROOTS = ("artifacts",)
RUN_STATUS_SCHEMA_VERSION = 3
_ZERO_HASH = "0" * 64
class ExecutionError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _attempt_sha256(value: dict[str, Any]) -> str:
    payload = dict(value)
    payload.pop("attempt_sha256", None)
    return lineage.canonical_sha256(payload)


def _sum_usage(values: Iterable[dict[str, Any]]) -> dict[str, float | int]:
    total: dict[str, float | int] = {
        "cpu_core_hours": 0.0,
        "gpu_hours": 0.0,
        "artifact_bytes": 0,
    }
    for value in values:
        try:
            cpu = value["cpu_core_hours"]
            gpu = value["gpu_hours"]
            artifacts = value["artifact_bytes"]
        except (KeyError, TypeError) as exc:
            raise ExecutionError("attempt usage has missing budget dimensions") from exc
        if (
            isinstance(cpu, bool)
            or not isinstance(cpu, (int, float))
            or cpu < 0
            or isinstance(gpu, bool)
            or not isinstance(gpu, (int, float))
            or gpu < 0
            or isinstance(artifacts, bool)
            or not isinstance(artifacts, int)
            or artifacts < 0
        ):
            raise ExecutionError("attempt usage contains an invalid value")
        total["cpu_core_hours"] += float(cpu)
        total["gpu_hours"] += float(gpu)
        total["artifact_bytes"] += artifacts
    return {
        "cpu_core_hours": round(float(total["cpu_core_hours"]), 6),
        "gpu_hours": round(float(total["gpu_hours"]), 6),
        "artifact_bytes": int(total["artifact_bytes"]),
    }


def validate_global_ledger_binding(
    *,
    status: dict[str, Any],
    limits: dict[str, Any],
    reserved_gpu_hours: float,
    expected_control_root: Path | str | None = None,
) -> dict[str, Any]:
    """Authenticate the status-bound prefix inside the shared budget ledger."""

    control_root_value = status.get("control_root")
    if not isinstance(control_root_value, str) or not control_root_value:
        raise ExecutionError("run status does not bind a global budget control root")
    control_root = Path(control_root_value).resolve()
    if expected_control_root is not None and control_root != Path(
        expected_control_root
    ).resolve():
        raise ExecutionError("run status global budget control root changed")
    budget = status.get("budget")
    binding = budget.get("global_ledger") if isinstance(budget, dict) else None
    expected_fields = {
        "ledger_relative_path",
        "bound_event_count",
        "bound_tail_event_sha256",
        "bound_cumulative_after",
    }
    if not isinstance(binding, dict) or set(binding) != expected_fields:
        raise ExecutionError("run status lacks an exact global budget ledger binding")
    if binding.get("ledger_relative_path") != "budget-ledger.v1.jsonl":
        raise ExecutionError("run status global budget ledger path is invalid")
    try:
        validated = validate_global_budget_ledger(
            control_root / binding["ledger_relative_path"],
            limits=limits,
            reserved_gpu_hours=reserved_gpu_hours,
        )
        bound_count = int(binding["bound_event_count"])
    except (BudgetError, OSError, TypeError, ValueError) as exc:
        raise ExecutionError(f"global budget ledger is invalid: {exc}") from exc
    if bound_count < 2 or bound_count > validated["event_count"]:
        raise ExecutionError("global budget ledger binding has an invalid event count")
    bound_event = validated["events"][bound_count - 1]
    if (
        bound_event.get("event") != "settled"
        or bound_event.get("event_sha256") != binding.get("bound_tail_event_sha256")
        or bound_event.get("cumulative_after") != binding.get("bound_cumulative_after")
    ):
        raise ExecutionError("global budget ledger bound prefix changed")
    return {**validated, "bound_event_count": bound_count}


def validate_attempt_history(
    *,
    run_dir: Path | str,
    status: dict[str, Any],
    limits: dict[str, Any],
    reserved_gpu_hours: float,
    selected_node_ids: Iterable[str] | None = None,
    expected_control_root: Path | str | None = None,
) -> dict[str, Any]:
    """Authenticate every finalized execution attempt and its budget settlement.

    The per-attempt record is hash chained and binds the immutable settlement
    artifact plus all retained logs.  This is intentionally usable both before
    a resume and by the release validator.
    """

    root = Path(run_dir).resolve()
    if "active_attempt" in status:
        raise ExecutionError("run status contains an unfinished active attempt")
    attempts = status.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        raise ExecutionError("run status has no authenticated attempt history")
    ledger = validate_global_ledger_binding(
        status=status,
        limits=limits,
        reserved_gpu_hours=reserved_gpu_hours,
        expected_control_root=expected_control_root,
    )
    ledger_events = ledger["events"]
    bound_event_count = ledger["bound_event_count"]
    previous = _ZERO_HASH
    seen_attempt_ids: set[str] = set()
    seen_budget_ids: set[str] = set()
    normalized_usage: list[dict[str, Any]] = []
    previous_cumulative: dict[str, float | int] | None = None
    run_id = status.get("run_id")
    run_identity = status.get("run_identity_sha256")
    full_node_ids = list(selected_node_ids) if selected_node_ids is not None else None
    completed_count = 0
    for ordinal, attempt in enumerate(attempts):
        if not isinstance(attempt, dict):
            raise ExecutionError(f"attempt {ordinal} is not an object")
        expected_fields = {
            "ordinal",
            "attempt_id",
            "budget_run_id",
            "resumed",
            "resume_from_node",
            "selected_nodes",
            "started_at",
            "finished_at",
            "state",
            "error",
            "planned_timeout_upper_bound",
            "used",
            "log_dir",
            "node_results",
            "settlement",
            "run_identity_sha256",
            "previous_attempt_sha256",
            "attempt_sha256",
        }
        if set(attempt) != expected_fields:
            raise ExecutionError(f"attempt {ordinal} has unknown or missing fields")
        attempt_id = attempt.get("attempt_id")
        budget_run_id = attempt.get("budget_run_id")
        expected_attempt_id = "initial" if ordinal == 0 else f"resume-{ordinal}"
        expected_budget_id = str(run_id) if ordinal == 0 else f"{run_id}.resume-{ordinal}"
        expected_log_dir = "logs" if ordinal == 0 else f"attempts/{expected_attempt_id}/logs"
        expected_settlement_path = (
            "budget-settlement.v1.json"
            if ordinal == 0
            else f"attempts/{expected_attempt_id}/budget-settlement.v1.json"
        )
        if (
            attempt.get("ordinal") != ordinal
            or attempt_id != expected_attempt_id
            or budget_run_id != expected_budget_id
            or attempt_id in seen_attempt_ids
            or budget_run_id in seen_budget_ids
        ):
            raise ExecutionError(f"attempt {ordinal} identity is invalid or duplicated")
        seen_attempt_ids.add(attempt_id)
        seen_budget_ids.add(budget_run_id)
        if attempt.get("resumed") is not (ordinal > 0):
            raise ExecutionError(f"attempt {ordinal} resume flag is invalid")
        if (
            attempt.get("previous_attempt_sha256") != previous
            or attempt.get("run_identity_sha256") != run_identity
            or attempt.get("attempt_sha256") != _attempt_sha256(attempt)
        ):
            raise ExecutionError(f"attempt {ordinal} hash chain or run binding is invalid")
        if attempt.get("state") not in {"succeeded", "failed"}:
            raise ExecutionError(f"attempt {ordinal} is not finalized")
        error = attempt.get("error")
        if not isinstance(error, str) or (
            attempt["state"] == "succeeded" and error
        ) or (attempt["state"] == "failed" and not error):
            raise ExecutionError(f"attempt {ordinal} error record is inconsistent")
        if (
            not isinstance(attempt.get("started_at"), str)
            or not attempt["started_at"]
            or not isinstance(attempt.get("finished_at"), str)
            or not attempt["finished_at"]
            or attempt.get("log_dir") != expected_log_dir
        ):
            raise ExecutionError(f"attempt {ordinal} timing or log directory is invalid")
        if not isinstance(attempt.get("selected_nodes"), list) or not attempt[
            "selected_nodes"
        ]:
            raise ExecutionError(f"attempt {ordinal} has no selected node suffix")
        if attempt.get("resume_from_node") != attempt["selected_nodes"][0]:
            raise ExecutionError(f"attempt {ordinal} resume point is inconsistent")
        if full_node_ids is not None and attempt["selected_nodes"] != full_node_ids[
            completed_count:
        ]:
            raise ExecutionError(f"attempt {ordinal} is not the remaining node suffix")
        settlement = attempt.get("settlement")
        if not isinstance(settlement, dict) or set(settlement) != {
            "path",
            "reservation_sha256",
            "settlement_sha256",
            "artifact_sha256",
            "ledger_event_ordinal",
        }:
            raise ExecutionError(f"attempt {ordinal} settlement record is invalid")
        relative = settlement.get("path")
        if relative != expected_settlement_path:
            raise ExecutionError(f"attempt {ordinal} settlement path is invalid")
        ledger_ordinal = settlement.get("ledger_event_ordinal")
        if (
            isinstance(ledger_ordinal, bool)
            or not isinstance(ledger_ordinal, int)
            or ledger_ordinal < 1
            or ledger_ordinal >= bound_event_count
        ):
            raise ExecutionError(f"attempt {ordinal} ledger event ordinal is invalid")
        ledger_reservation = ledger_events[ledger_ordinal - 1]
        ledger_settlement = ledger_events[ledger_ordinal]
        if (
            ledger_reservation.get("event_sha256")
            != settlement.get("reservation_sha256")
            or ledger_settlement.get("event_sha256")
            != settlement.get("settlement_sha256")
            or ledger_settlement.get("reservation_sha256")
            != ledger_reservation.get("event_sha256")
        ):
            raise ExecutionError(
                f"attempt {ordinal} settlement is absent from the global ledger"
            )
        settlement_path = root / relative
        try:
            if settlement_path.resolve().parent != (
                root if ordinal == 0 else root / "attempts" / expected_attempt_id
            ).resolve():
                raise ExecutionError(f"attempt {ordinal} settlement path escapes its attempt")
            raw_settlement = json.loads(settlement_path.read_text(encoding="utf-8"))
            normalized = validate_settlement_snapshot(
                raw_settlement,
                expected_run_id=expected_budget_id,
                limits=limits,
                reserved_gpu_hours=reserved_gpu_hours,
            )
        except (OSError, json.JSONDecodeError, BudgetError, TypeError, ValueError) as exc:
            raise ExecutionError(f"attempt {ordinal} budget settlement is invalid: {exc}") from exc
        if (
            settlement.get("reservation_sha256") != normalized["reservation_sha256"]
            or settlement.get("settlement_sha256") != normalized["settlement_sha256"]
            or settlement.get("artifact_sha256") != lineage.artifact_sha256(settlement_path)
            or attempt.get("used") != normalized["used"]
            or attempt.get("planned_timeout_upper_bound")
            != raw_settlement["reservation"].get("planned")
            or attempt.get("state")
            != raw_settlement["settlement"].get("run_state")
        ):
            raise ExecutionError(f"attempt {ordinal} settlement hashes or usage mismatch")
        before = raw_settlement["reservation"]["cumulative_before"]
        after = raw_settlement["settlement"]["cumulative_after"]
        if previous_cumulative is not None and any(
            before[key] < previous_cumulative[key] for key in previous_cumulative
        ):
            raise ExecutionError("attempt budget cumulative usage moved backwards")
        previous_cumulative = after
        node_results = attempt.get("node_results")
        if not isinstance(node_results, dict):
            raise ExecutionError(f"attempt {ordinal} node results are invalid")
        selected_nodes = set(attempt["selected_nodes"])
        if not set(node_results) <= selected_nodes:
            raise ExecutionError(f"attempt {ordinal} contains results for unselected nodes")
        executed_ids = [
            node_id for node_id in attempt["selected_nodes"] if node_id in node_results
        ]
        if set(executed_ids) != set(node_results) or executed_ids != attempt[
            "selected_nodes"
        ][: len(executed_ids)]:
            raise ExecutionError(f"attempt {ordinal} executed nodes are not a prefix")
        for execution_index, node_id in enumerate(executed_ids):
            result = node_results[node_id]
            if not isinstance(result, dict) or set(result) != {
                "state",
                "exit_code",
                "stdout",
                "stderr",
                "stdout_sha256",
                "stderr_sha256",
            }:
                raise ExecutionError(
                    f"attempt {ordinal} node result for {node_id} is invalid"
                )
            succeeded = result.get("state") == "succeeded" and result.get("exit_code") == 0
            failed = result.get("state") == "failed" and (
                isinstance(result.get("exit_code"), int)
                and result.get("exit_code") != 0
            )
            if not (succeeded or failed):
                raise ExecutionError(
                    f"attempt {ordinal} node result state is inconsistent for {node_id}"
                )
            if failed and execution_index != len(node_results) - 1:
                raise ExecutionError(f"attempt {ordinal} executed nodes after a failure")
            for stream_name in ("stdout", "stderr"):
                log_relative = result[stream_name]
                expected_log = f"{expected_log_dir}/{node_id}.{stream_name}.log"
                if log_relative != expected_log:
                    raise ExecutionError(
                        f"attempt {ordinal} {node_id} {stream_name} path is invalid"
                    )
                log_path = root / log_relative
                try:
                    actual_log_hash = lineage.artifact_sha256(log_path)
                except OSError as exc:
                    raise ExecutionError(
                        f"attempt {ordinal} {node_id} {stream_name} log is missing"
                    ) from exc
                if actual_log_hash != result[f"{stream_name}_sha256"]:
                    raise ExecutionError(
                        f"attempt {ordinal} {node_id} {stream_name} log changed"
                    )
            if succeeded:
                completed_count += 1
        if attempt["state"] == "succeeded" and (
            executed_ids != attempt["selected_nodes"]
            or any(result["state"] != "succeeded" for result in node_results.values())
        ):
            raise ExecutionError(f"attempt {ordinal} succeeded without all selected nodes")
        normalized_usage.append(normalized["used"])
        previous = attempt["attempt_sha256"]
    total = _sum_usage(normalized_usage)
    if status.get("budget", {}).get("used") != total:
        raise ExecutionError("status cumulative budget usage does not match attempts")
    if attempts[-1].get("state") != status.get("state"):
        raise ExecutionError("latest attempt state does not match logical run state")
    if status.get("state") == "failed" and attempts[-1].get("error") != status.get(
        "error"
    ):
        raise ExecutionError("latest attempt error does not match logical run status")
    if status.get("state") == "succeeded" and status.get("error") not in {None, ""}:
        raise ExecutionError("succeeded logical run retains a failure error")
    return {
        "used": total,
        "last_attempt_sha256": previous,
        "attempt_count": len(attempts),
        "last_cumulative_after": previous_cumulative,
        "completed_node_ids": (
            full_node_ids[:completed_count] if full_node_ids is not None else None
        ),
    }


def _path_sha256(path: Path) -> str:
    if path.is_file():
        return lineage.artifact_sha256(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    entries: list[tuple[str, str]] = []
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        entries.append((item.relative_to(path).as_posix(), lineage.artifact_sha256(item)))
    return lineage.canonical_sha256(sorted(entries, key=lambda item: item[0]))


def _path_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if not path.is_dir():
        raise FileNotFoundError(path)
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _is_within_root(relative: str, root: str) -> bool:
    normalized = root.rstrip("/")
    return relative == normalized or relative.startswith(normalized + "/")


def _validate_mutable_roots(roots: Iterable[str]) -> tuple[str, ...]:
    normalized: set[str] = set()
    for raw in roots:
        path = Path(raw)
        rendered = path.as_posix().rstrip("/")
        if not rendered or path.is_absolute() or ".." in path.parts:
            raise ValueError(f"unsafe mutable workspace root: {raw!r}")
        normalized.add(rendered)
    return tuple(sorted(normalized))


def committed_paths(repo: Path, commit: str) -> tuple[str, ...]:
    """Return the exact file inventory in a commit without consulting the worktree."""

    completed = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", commit],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if completed.returncode:
        raise ExecutionError(f"git ls-tree failed: {completed.stderr.strip()}")
    return tuple(sorted(line.replace("\\", "/") for line in completed.stdout.splitlines()))


def expected_mutable_roots(
    repo: Path,
    commit: str,
    selected: Iterable[ExperimentNode],
) -> tuple[str, ...]:
    """Identify declared outputs and repository-external inputs in a run workspace."""

    nodes = tuple(selected)
    tracked = committed_paths(repo, commit)
    outputs = {output for node in nodes for output in node.outputs}
    roots = {*outputs, *_CONTROLLED_MUTABLE_ROOTS}
    for value in {
        item
        for node in nodes
        for item in (*node.inputs, *node.late_control_inputs)
    }:
        if any(_is_within_root(value, output) for output in outputs):
            continue
        if value in tracked or any(path.startswith(value.rstrip("/") + "/") for path in tracked):
            continue
        roots.add(value)
    return _validate_mutable_roots(roots)


def immutable_workspace_sha256(workspace: Path, mutable_roots: Iterable[str]) -> str:
    """Hash every file/symlink outside the explicitly mutable run roots."""

    roots = _validate_mutable_roots(mutable_roots)

    def excluded(relative: str) -> bool:
        return any(_is_within_root(relative, root) for root in roots)

    entries: list[tuple[str, str]] = []
    for current, directories, filenames in os.walk(workspace, topdown=True, followlinks=False):
        current_path = Path(current)
        retained_directories = []
        for name in sorted(directories):
            path = current_path / name
            relative = path.relative_to(workspace).as_posix()
            if excluded(relative):
                continue
            if path.is_symlink():
                entries.append(
                    (relative, lineage.canonical_sha256({"symlink": os.readlink(path)}))
                )
            else:
                retained_directories.append(name)
        directories[:] = retained_directories
        for name in sorted(filenames):
            path = current_path / name
            relative = path.relative_to(workspace).as_posix()
            if excluded(relative):
                continue
            if path.is_symlink():
                digest = lineage.canonical_sha256({"symlink": os.readlink(path)})
            else:
                digest = lineage.artifact_sha256(path)
            entries.append((relative, digest))
    return lineage.canonical_sha256(sorted(entries, key=lambda item: item[0]))


def committed_immutable_sha256(
    repo: Path,
    commit: str,
    mutable_roots: Iterable[str],
) -> str:
    """Hash the immutable file payload of an exact git commit.

    This provides an independent inventory check at release time: merely deleting
    a committed source file from the run workspace cannot produce a new accepted
    snapshot hash.
    """

    roots = _validate_mutable_roots(mutable_roots)
    archived = subprocess.run(
        ["git", "archive", "--format=tar", commit],
        cwd=repo,
        capture_output=True,
        timeout=60,
        check=False,
    )
    if archived.returncode:
        raise ExecutionError(
            f"git archive failed: {archived.stderr.decode(errors='replace')}"
        )
    entries: list[tuple[str, str]] = []
    with tarfile.open(fileobj=io.BytesIO(archived.stdout), mode="r:") as archive:
        for member in sorted(archive.getmembers(), key=lambda value: value.name):
            relative = member.name.rstrip("/")
            if not relative or any(_is_within_root(relative, root) for root in roots):
                continue
            if member.issym() or member.islnk():
                raise ExecutionError(f"git archive links are not accepted: {member.name}")
            if member.isdir():
                continue
            if not member.isfile():
                raise ExecutionError(
                    f"git archive contains unsupported member: {member.name}"
                )
            stream = archive.extractfile(member)
            if stream is None:  # pragma: no cover - tarfile contract guard
                raise ExecutionError(f"cannot read git archive member: {member.name}")
            entries.append((relative, hashlib.sha256(stream.read()).hexdigest()))
    return lineage.canonical_sha256(sorted(entries, key=lambda item: item[0]))


def declared_path_hashes(workspace: Path, paths: Iterable[str]) -> dict[str, str]:
    """Return current content hashes for an exact declared-path set."""
    hashes: dict[str, str] = {}
    for rel in paths:
        path = workspace / rel
        if not path.exists():
            raise FileNotFoundError(path)
        hashes[rel] = _path_sha256(path)
    return hashes


def collect_checkpoint_hashes(workspace: Path, paths: Iterable[str]) -> dict[str, str]:
    """Hash checkpoint files contained in the declared inputs and outputs."""
    checkpoints: dict[str, str] = {}
    for rel in paths:
        artifact = workspace / rel
        if artifact.is_file() and artifact.suffix.lower() in CHECKPOINT_SUFFIXES:
            checkpoints[rel] = _path_sha256(artifact)
        elif artifact.is_dir():
            for checkpoint in sorted(
                path for path in artifact.rglob("*")
                if path.is_file() and path.suffix.lower() in CHECKPOINT_SUFFIXES
            ):
                key = f"{rel}/{checkpoint.relative_to(artifact).as_posix()}"
                checkpoints[key] = _path_sha256(checkpoint)
    return checkpoints


def _safe_extract(raw_tar: bytes, destination: Path) -> None:
    destination_resolved = destination.resolve()
    with tarfile.open(fileobj=io.BytesIO(raw_tar), mode="r:") as archive:
        for member in archive.getmembers():
            target = (destination / member.name).resolve()
            if target != destination_resolved and destination_resolved not in target.parents:
                raise ExecutionError(f"git archive contains unsafe path: {member.name}")
            if member.issym() or member.islnk():
                raise ExecutionError(f"git archive links are not accepted: {member.name}")
            if not (member.isfile() or member.isdir()):
                raise ExecutionError(f"git archive contains unsupported member: {member.name}")
        # Python 3.11 has no extraction filter argument.  The complete member
        # validation above is the equivalent fail-closed policy for git archives.
        archive.extractall(destination)


class DAGRunner:
    """Execute one manifest profile inside a clean git-archive snapshot."""

    def __init__(
        self,
        *,
        repo: Path | str,
        manifest: ExperimentManifest,
        profile: str,
        run_root: Path | str,
        run_id: str,
        resource: str | None = None,
        environment_lock: str | None = None,
        control_root: Path | str | None = None,
        resume: bool = False,
    ) -> None:
        self.repo = Path(repo).resolve()
        self.manifest = manifest
        self.profile = profile
        self.resource = resource
        if environment_lock is not None:
            lock_relative_path(environment_lock)
        self.environment_lock = environment_lock
        self.run_root = Path(run_root).resolve()
        self.control_root = Path(control_root or self.run_root).resolve()
        self.resume = bool(resume)
        if not _SAFE_ID.fullmatch(run_id):
            raise ValueError("run_id is not a safe identifier")
        self.run_id = run_id
        self.run_dir = self.run_root / run_id
        self.workspace = self.run_dir / "workspace"
        self.status_path = self.run_dir / "status.json"
        self.envelope_dir = self.run_dir / "envelopes"
        self.control_input_dir = self.run_dir / "control-inputs"
        self.log_dir = self.run_dir / "logs"
        self._attempt_index = 0
        self._attempt_id = "initial"
        self._attempt_dir = self.run_dir
        self._budget_run_id = self.run_id
        self._status: dict[str, Any] = {}
        self._cpu_core_hours = 0.0
        self._gpu_hours = 0.0
        self._artifact_bytes = 0
        self._mutable_roots: tuple[str, ...] = ()
        self._source_snapshot_sha256 = ""
        self._environment_report: LockedEnvironmentReport | None = None
        self._environment_sha256 = ""
        self._global_cumulative_before = {
            "cpu_core_hours": 0.0,
            "gpu_hours": 0.0,
            "artifact_bytes": 0,
        }

    @staticmethod
    def _planned_budget(selected: Iterable[ExperimentNode]) -> dict[str, float | int]:
        nodes = list(selected)
        return {
            "cpu_core_hours": sum(node.timeout * node.resource.cpus for node in nodes) / 3600,
            "gpu_hours": sum(node.timeout * node.resource.gpus for node in nodes) / 3600,
            "artifact_bytes": MAX_ARTIFACT_BYTES,
        }

    def _declared_hashes(self, paths: Iterable[str], *, kind: str) -> dict[str, str]:
        """Hash declared workspace paths from disk on every call.

        Deliberately do not cache these values: a node is untrusted with respect
        to the shared run workspace, and a persistent cache would hide an
        in-place edit made after a producer had been validated.
        """
        hashes: dict[str, str] = {}
        for rel in paths:
            try:
                hashes[rel] = declared_path_hashes(self.workspace, (rel,))[rel]
            except FileNotFoundError as exc:
                raise ExecutionError(f"missing declared {kind} {rel}") from exc
            except (OSError, ValueError) as exc:
                raise ExecutionError(f"cannot hash declared {kind} {rel}: {exc}") from exc
        return hashes

    def _checkpoint_hashes(self, paths: Iterable[str]) -> dict[str, str]:
        return collect_checkpoint_hashes(self.workspace, paths)

    def _control_snapshot_root(self, node: ExperimentNode) -> Path:
        return self.control_input_dir / node.id

    def _late_control_binding(self, node: ExperimentNode) -> LateControlBinding:
        """Independently verify a node's sealed late-control snapshot."""

        try:
            return validate_late_control_snapshot(
                snapshot_root=self._control_snapshot_root(node),
                expected_run_id=self.run_id,
                expected_node_id=node.id,
                expected_inputs=node.late_control_inputs,
            )
        except (LateControlInputError, OSError, ValueError) as exc:
            raise ExecutionError(
                f"{node.id}: late-control snapshot verification failed: {exc}"
            ) from exc

    @staticmethod
    def _effective_config_sha256(
        node: ExperimentNode, binding: LateControlBinding
    ) -> str:
        if not node.late_control_inputs:
            return node.config_sha256()
        return node.config_sha256(
            late_control_inputs_sha256=binding.inputs_sha256
        )

    def _prepare_late_control_policy(self, node: ExperimentNode) -> Path | None:
        if not node.late_control_inputs:
            if self._control_snapshot_root(node).exists():
                raise ExecutionError(
                    f"{node.id}: node without late controls has a stale snapshot"
                )
            return None
        final_root = self._control_snapshot_root(node)
        if final_root.exists() or final_root.is_symlink():
            raise ExecutionError(f"{node.id}: late-control snapshot already exists")
        capture_root = self._attempt_dir / "late-control-staging" / node.id
        policy_path = capture_root / "policy.v1.json"
        try:
            write_late_control_policy(
                path=policy_path,
                run_id=self.run_id,
                node_id=node.id,
                workspace=self.workspace,
                capture_root=capture_root,
                inputs=node.late_control_inputs,
            )
        except (LateControlInputError, OSError, ValueError) as exc:
            raise ExecutionError(
                f"{node.id}: cannot prepare late-control capture: {exc}"
            ) from exc
        return policy_path

    def _finalize_late_control_policy(
        self, node: ExperimentNode, policy_path: Path | None
    ) -> LateControlBinding:
        if not node.late_control_inputs:
            return LateControlBinding(
                inputs_sha256={},
                snapshot_sha256=empty_late_control_snapshot_sha256(),
                artifact_bytes=0,
            )
        if policy_path is None:
            raise ExecutionError(f"{node.id}: late-control capture policy is missing")
        try:
            return finalize_late_control_snapshot(
                policy_path=policy_path,
                final_root=self._control_snapshot_root(node),
                expected_run_id=self.run_id,
                expected_node_id=node.id,
                expected_inputs=node.late_control_inputs,
            )
        except (LateControlInputError, OSError, ValueError) as exc:
            raise ExecutionError(
                f"{node.id}: late-control capture failed closed: {exc}"
            ) from exc

    def _upstream_envelope_hashes(
        self,
        node: ExperimentNode,
        commit: str,
    ) -> dict[str, str]:
        """Validate and hash every direct dependency envelope, failing closed."""
        upstream: dict[str, str] = {}
        node_map = self.manifest.by_id()
        for dep in node.deps:
            envelope_path = self.envelope_dir / f"{dep}.json"
            if not envelope_path.is_file():
                raise ExecutionError(
                    f"{node.id}: dependency envelope is missing for {dep}; "
                    "dependency output hashes are not a provenance substitute"
                )
            try:
                dependency = ResultEnvelope.read(envelope_path)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                raise ExecutionError(
                    f"{node.id}: dependency envelope for {dep} is invalid: {exc}"
                ) from exc
            expected_node = node_map[dep]
            late_binding = self._late_control_binding(expected_node)
            if (
                dependency.run_id != self.run_id
                or dependency.node_id != dep
                or dependency.commit != commit
                or dependency.config_sha256
                != self._effective_config_sha256(expected_node, late_binding)
                or dependency.late_control_inputs_sha256
                != late_binding.inputs_sha256
                or dependency.late_control_snapshot_sha256
                != late_binding.snapshot_sha256
            ):
                raise ExecutionError(f"{node.id}: dependency envelope identity mismatch for {dep}")
            if set(dependency.outputs_sha256) != set(expected_node.outputs):
                raise ExecutionError(f"{node.id}: dependency output contract mismatch for {dep}")
            actual_outputs = self._declared_hashes(
                expected_node.outputs, kind=f"dependency output from {dep}",
            )
            if actual_outputs != dependency.outputs_sha256:
                raise ExecutionError(f"{node.id}: dependency outputs changed after {dep}")
            upstream[dep] = lineage.artifact_sha256(envelope_path)
        return upstream

    def _git_state(self, selected: Iterable[ExperimentNode]) -> tuple[str, bool]:
        try:
            commit_run = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=self.repo, capture_output=True,
                text=True, timeout=10, check=False,
            )
            status_run = subprocess.run(
                ["git", "status", "--porcelain"], cwd=self.repo, capture_output=True,
                text=True, timeout=10, check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise ExecutionError(f"cannot inspect git state: {exc}") from exc
        commit = commit_run.stdout.strip()
        if commit_run.returncode or not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise ExecutionError("source repository has no valid HEAD commit")
        if status_run.returncode:
            raise ExecutionError("git status failed")
        generated = {output for node in self.manifest.nodes for output in node.outputs}
        dirty_code: list[str] = []
        for line in status_run.stdout.splitlines():
            path = line[3:].strip().strip('"')
            if " -> " in path:
                path = path.split(" -> ")[-1]
            path = path.replace("\\", "/")
            if path.startswith(".runs/") or path in generated:
                continue
            if any(path.startswith(output.rstrip("/") + "/") for output in generated):
                continue
            dirty_code.append(path)
        return commit, bool(dirty_code)

    def _stage_snapshot(self, commit: str, selected: Iterable[ExperimentNode]) -> None:
        selected = tuple(selected)
        self._mutable_roots = expected_mutable_roots(self.repo, commit, selected)
        self.workspace.mkdir(parents=True)
        archived = subprocess.run(
            ["git", "archive", "--format=tar", commit], cwd=self.repo,
            capture_output=True, timeout=60, check=False,
        )
        if archived.returncode:
            raise ExecutionError(f"git archive failed: {archived.stderr.decode(errors='replace')}")
        _safe_extract(archived.stdout, self.workspace)

        produced = {output for node in selected for output in node.outputs}
        for rel in sorted({path for node in selected for path in node.inputs} - produced):
            target = self.workspace / rel
            if target.exists():
                continue
            source = self.repo / rel
            if not source.exists():
                continue  # checked immediately before the consuming node runs
            try:
                resolved_source = source.resolve(strict=True)
            except OSError as exc:
                raise ExecutionError(f"cannot resolve declared external input {rel}") from exc
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                target.symlink_to(
                    resolved_source,
                    target_is_directory=resolved_source.is_dir(),
                )
            except OSError as exc:
                if resolved_source.is_dir():
                    raise ExecutionError(
                        f"cannot stage external directory input {rel} as a symlink"
                    ) from exc
                else:
                    shutil.copy2(resolved_source, target)
        self._source_snapshot_sha256 = immutable_workspace_sha256(
            self.workspace, self._mutable_roots,
        )
        committed_sha256 = committed_immutable_sha256(
            self.repo, commit, self._mutable_roots,
        )
        if self._source_snapshot_sha256 != committed_sha256:
            raise ExecutionError("staged immutable source does not match the selected commit")

    def _verify_source_snapshot(self, *, node_id: str) -> None:
        if not self._source_snapshot_sha256:
            raise ExecutionError("source snapshot has not been staged")
        actual = immutable_workspace_sha256(self.workspace, self._mutable_roots)
        if actual != self._source_snapshot_sha256:
            raise ExecutionError(
                f"{node_id}: immutable source snapshot changed during the run"
            )

    def _environment_lock_sha256(self, node: ExperimentNode) -> str:
        if self._environment_report is None:
            raise ExecutionError("run-level environment lock was not verified")
        relative = self._environment_report.lock_path
        lock = self.workspace / relative
        if not lock.is_file():
            raise ExecutionError(f"{node.id}: required environment lock is missing: {relative}")
        digest = lineage.artifact_sha256(lock)
        if digest != self._environment_report.lock_sha256:
            raise ExecutionError(f"{node.id}: staged environment lock changed after verification")
        return digest

    def _run_identity_payload(self) -> dict[str, Any]:
        if self._environment_report is None:
            raise ExecutionError("run-level environment lock was not verified")
        return {
            "run_id": self.run_id,
            "control_root": str(self.control_root),
            "profile": self.profile,
            "resource": self.resource,
            "environment_lock": self._environment_report.as_dict(),
            "environment_sha256": self._environment_sha256,
            "python_executable": str(Path(sys.executable).resolve()),
            "manifest_sha256": self.manifest.sha256(),
            "commit": self._status.get("commit"),
            "source_snapshot_sha256": self._source_snapshot_sha256,
            "mutable_workspace_roots": list(self._mutable_roots),
        }

    def _run_identity_sha256(self) -> str:
        return lineage.canonical_sha256(self._run_identity_payload())

    def _verify_succeeded_envelope(
        self,
        node: ExperimentNode,
        commit: str,
    ) -> None:
        """Recompute the complete envelope contract for one resumed prefix node."""

        envelope_path = self.envelope_dir / f"{node.id}.json"
        try:
            envelope = ResultEnvelope.read(envelope_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise ExecutionError(f"{node.id}: invalid completed envelope: {exc}") from exc
        expected_argv = [
            str(Path(sys.executable).resolve()) if token == "{python}" else token
            for token in node.command
        ]
        late_binding = self._late_control_binding(node)
        if (
            envelope.run_id != self.run_id
            or envelope.node_id != node.id
            or envelope.commit != commit
            or envelope.dirty is not False
            or envelope.config_sha256
            != self._effective_config_sha256(node, late_binding)
            or envelope.late_control_inputs_sha256
            != late_binding.inputs_sha256
            or envelope.late_control_snapshot_sha256
            != late_binding.snapshot_sha256
            or envelope.argv != expected_argv
            or envelope.seed != node.seed
            or envelope.env_sha256 != self._environment_sha256
            or envelope.source_sha256 != self._source_snapshot_sha256
            or self._environment_report is None
            or envelope.environment_lock_sha256
            != self._environment_report.lock_sha256
        ):
            raise ExecutionError(f"{node.id}: completed envelope identity mismatch")
        try:
            input_hashes = self._declared_hashes(
                node.inputs, kind=f"completed input for {node.id}"
            )
            output_hashes = self._declared_hashes(
                node.outputs, kind=f"completed output for {node.id}"
            )
        except ExecutionError as exc:
            raise ExecutionError(f"{node.id}: completed artifact verification failed: {exc}") from exc
        if set(envelope.outputs_sha256) != set(node.outputs):
            raise ExecutionError(f"{node.id}: completed output contract mismatch")
        if envelope.outputs_sha256 != output_hashes:
            raise ExecutionError(f"{node.id}: completed outputs changed")
        checkpoints = self._checkpoint_hashes((*node.inputs, *node.outputs))
        if envelope.checkpoint_sha256 != checkpoints:
            raise ExecutionError(f"{node.id}: completed checkpoint hashes changed")
        data_inputs = {
            rel: digest for rel, digest in input_hashes.items() if rel not in checkpoints
        }
        effective_data_inputs = dict(data_inputs)
        effective_data_inputs.update(
            {
                f"late-control:{rel}": digest
                for rel, digest in late_binding.inputs_sha256.items()
            }
        )
        if envelope.data_sha256 != lineage.canonical_sha256(effective_data_inputs):
            raise ExecutionError(f"{node.id}: completed input-data hash changed")
        split_material: dict[str, Any] = {
            "seed": node.seed,
            "argv": expected_argv,
            "inputs": input_hashes,
            "deps": list(node.deps),
        }
        if node.late_control_inputs:
            split_material["late_control_inputs_sha256"] = late_binding.inputs_sha256
            split_material["late_control_snapshot_sha256"] = (
                late_binding.snapshot_sha256
            )
        split_sha256 = lineage.canonical_sha256(split_material)
        if envelope.split_sha256 != split_sha256:
            raise ExecutionError(f"{node.id}: completed input/split hash changed")
        expected_upstream = {
            dep: lineage.artifact_sha256(self.envelope_dir / f"{dep}.json")
            for dep in node.deps
        }
        if envelope.upstream_sha256 != expected_upstream:
            raise ExecutionError(f"{node.id}: completed upstream envelope hashes changed")

    def _load_resume_status(
        self,
        *,
        selected: list[ExperimentNode],
        commit: str,
        limits: dict[str, Any],
    ) -> list[ExperimentNode]:
        """Authenticate an existing logical run before acquiring a new lease."""

        if not self.run_dir.is_dir():
            raise ExecutionError("--resume requires an existing run directory")
        try:
            status = json.loads(self.status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ExecutionError(f"cannot read resume status: {exc}") from exc
        if not isinstance(status, dict) or status.get("schema_version") != RUN_STATUS_SCHEMA_VERSION:
            raise ExecutionError("resume requires the authenticated run status schema")
        self._status = status
        if (
            status.get("run_id") != self.run_id
            or status.get("control_root") != str(self.control_root)
            or status.get("profile") != self.profile
            or status.get("resource") != self.resource
            or status.get("manifest_sha256") != self.manifest.sha256()
            or status.get("commit") != commit
            or status.get("python_executable") != str(Path(sys.executable).resolve())
            or status.get("environment_sha256") != self._environment_sha256
            or self._environment_report is None
            or status.get("environment_lock") != self._environment_report.as_dict()
        ):
            raise ExecutionError("resume run identity does not match the current invocation")
        self._mutable_roots = expected_mutable_roots(self.repo, commit, selected)
        if status.get("mutable_workspace_roots") != list(self._mutable_roots):
            raise ExecutionError("resume mutable workspace roots do not match")
        expected_source = committed_immutable_sha256(
            self.repo, commit, self._mutable_roots
        )
        try:
            actual_source = immutable_workspace_sha256(
                self.workspace, self._mutable_roots
            )
        except OSError as exc:
            raise ExecutionError(f"cannot hash resume workspace source: {exc}") from exc
        if (
            actual_source != expected_source
            or status.get("source_snapshot_sha256") != actual_source
        ):
            raise ExecutionError("resume immutable source snapshot is not authentic")
        self._source_snapshot_sha256 = actual_source
        identity_sha256 = self._run_identity_sha256()
        if status.get("run_identity_sha256") != identity_sha256:
            raise ExecutionError("resume run identity hash is invalid")
        budget = status.get("budget")
        if not isinstance(budget, dict) or budget.get("limits") != {
            **limits,
            "reserved_gpu_hours_for_rerun": RESERVED_GPU_HOURS,
        }:
            raise ExecutionError("resume budget limits do not match the frozen protocol")
        selected_ids = [node.id for node in selected]
        attempt_history = validate_attempt_history(
            run_dir=self.run_dir,
            status=status,
            limits=limits,
            reserved_gpu_hours=RESERVED_GPU_HOURS,
            selected_node_ids=selected_ids,
            expected_control_root=self.control_root,
        )
        nodes_status = status.get("nodes")
        if not isinstance(nodes_status, dict) or set(nodes_status) != set(selected_ids):
            raise ExecutionError("resume node inventory does not match the manifest")
        prefix_count = 0
        found_incomplete = False
        for node in selected:
            node_status = nodes_status.get(node.id)
            if not isinstance(node_status, dict):
                raise ExecutionError(f"resume node status is invalid for {node.id}")
            succeeded = (
                node_status.get("state") == "succeeded"
                and node_status.get("exit_code") == 0
            )
            if succeeded and found_incomplete:
                raise ExecutionError("resume succeeded nodes are not a topological prefix")
            if succeeded:
                prefix_count += 1
            else:
                found_incomplete = True
        if attempt_history["completed_node_ids"] != selected_ids[:prefix_count]:
            raise ExecutionError("resume node status does not match the attempt history")
        if prefix_count == len(selected):
            raise ExecutionError("run is already complete; there is nothing to resume")
        expected_envelopes = {node.id for node in selected[:prefix_count]}
        found_envelopes = {path.stem for path in self.envelope_dir.glob("*.json")}
        if found_envelopes != expected_envelopes:
            raise ExecutionError("resume envelope inventory is not the succeeded prefix")
        expected_control_snapshots = {
            node.id for node in selected[:prefix_count] if node.late_control_inputs
        }
        if self.control_input_dir.exists():
            if self.control_input_dir.is_symlink() or not self.control_input_dir.is_dir():
                raise ExecutionError("resume late-control snapshot root is invalid")
            found_control_snapshots = {
                path.name for path in self.control_input_dir.iterdir()
            }
        else:
            found_control_snapshots = set()
        if found_control_snapshots != expected_control_snapshots:
            raise ExecutionError(
                "resume late-control snapshot inventory is not the succeeded prefix"
            )
        self._verify_source_snapshot(node_id="resume-preflight")
        for node in selected[:prefix_count]:
            self._verify_succeeded_envelope(node, commit)
        self._attempt_index = len(status["attempts"])
        self._attempt_id = f"resume-{self._attempt_index}"
        self._budget_run_id = f"{self.run_id}.resume-{self._attempt_index}"
        self._attempt_dir = self.run_dir / "attempts" / self._attempt_id
        if self._attempt_dir.exists():
            raise ExecutionError("resume attempt directory already exists")
        self.log_dir = self._attempt_dir / "logs"
        return selected[prefix_count:]

    def _write_status(self) -> None:
        _atomic_json(self.status_path, self._status)

    def _begin_attempt(
        self,
        *,
        selected: list[ExperimentNode],
        planned: dict[str, float | int],
        reservation: dict[str, Any],
    ) -> None:
        if self.resume:
            self._attempt_dir.mkdir(parents=True, exist_ok=False)
        self._status["active_attempt"] = {
            "ordinal": self._attempt_index,
            "attempt_id": self._attempt_id,
            "budget_run_id": self._budget_run_id,
            "resumed": self.resume,
            "resume_from_node": selected[0].id,
            "selected_nodes": [node.id for node in selected],
            "started_at": _utc_now(),
            "planned_timeout_upper_bound": planned,
            "global_reservation_sha256": reservation["event_sha256"],
            "log_dir": self.log_dir.relative_to(self.run_dir).as_posix(),
        }
        self._write_status()

    def _finalize_attempt(
        self,
        *,
        selected: list[ExperimentNode],
        planned: dict[str, float | int],
        reservation: dict[str, Any],
        settlement: dict[str, Any],
        settlement_path: Path,
        ledger_event_ordinal: int,
        used: dict[str, float | int],
    ) -> None:
        node_results: dict[str, Any] = {}
        for node in selected:
            value = self._status["nodes"][node.id]
            stdout_relative = value.get("stdout")
            stderr_relative = value.get("stderr")
            if stdout_relative is None and stderr_relative is None:
                continue
            if not isinstance(stdout_relative, str) or not isinstance(stderr_relative, str):
                raise ExecutionError(f"{node.id}: attempt logs are incomplete")
            stdout_path = self.run_dir / stdout_relative
            stderr_path = self.run_dir / stderr_relative
            node_results[node.id] = {
                "state": value.get("state"),
                "exit_code": value.get("exit_code"),
                "stdout": stdout_relative,
                "stderr": stderr_relative,
                "stdout_sha256": lineage.artifact_sha256(stdout_path),
                "stderr_sha256": lineage.artifact_sha256(stderr_path),
            }
        previous = (
            self._status["attempts"][-1]["attempt_sha256"]
            if self._status["attempts"]
            else _ZERO_HASH
        )
        attempt = {
            "ordinal": self._attempt_index,
            "attempt_id": self._attempt_id,
            "budget_run_id": self._budget_run_id,
            "resumed": self.resume,
            "resume_from_node": selected[0].id,
            "selected_nodes": [node.id for node in selected],
            "started_at": self._status["active_attempt"]["started_at"],
            "finished_at": _utc_now(),
            "state": self._status.get("state", "failed"),
            "error": str(self._status.get("error", "")),
            "planned_timeout_upper_bound": planned,
            "used": used,
            "log_dir": self.log_dir.relative_to(self.run_dir).as_posix(),
            "node_results": node_results,
            "settlement": {
                "path": settlement_path.relative_to(self.run_dir).as_posix(),
                "reservation_sha256": reservation["event_sha256"],
                "settlement_sha256": settlement["event_sha256"],
                "artifact_sha256": lineage.artifact_sha256(settlement_path),
                "ledger_event_ordinal": ledger_event_ordinal,
            },
            "run_identity_sha256": self._status["run_identity_sha256"],
            "previous_attempt_sha256": previous,
        }
        attempt["attempt_sha256"] = _attempt_sha256(attempt)
        self._status["attempts"].append(attempt)
        self._status.pop("active_attempt", None)
        prior_usage = [value["used"] for value in self._status["attempts"]]
        self._status["budget"]["used"] = _sum_usage(prior_usage)
        self._write_status()

    @staticmethod
    def _posix_process_group_exists(process_group_id: int) -> bool:
        try:
            os.killpg(process_group_id, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    @classmethod
    def _wait_for_posix_process_group(
        cls, process_group_id: int, *, timeout_seconds: float
    ) -> bool:
        deadline = time.monotonic() + timeout_seconds
        while cls._posix_process_group_exists(process_group_id):
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.05)
        return True

    def _terminate_posix_job(self, process: subprocess.Popen[Any]) -> None:
        # The node is started in a new session, so its leader PID is also its
        # process-group id. Waiting only for the leader is insufficient: it may
        # exit on SIGTERM while a training child ignores the signal and keeps
        # consuming GPU time or mutating outputs.
        process_group_id = process.pid
        if self._posix_process_group_exists(process_group_id):
            try:
                os.killpg(process_group_id, signal.SIGTERM)
            except ProcessLookupError:
                pass
        group_exited = self._wait_for_posix_process_group(
            process_group_id, timeout_seconds=5.0
        )
        if not group_exited:
            try:
                os.killpg(process_group_id, signal.SIGKILL)
            except ProcessLookupError:
                pass
            group_exited = self._wait_for_posix_process_group(
                process_group_id, timeout_seconds=5.0
            )
        if process.poll() is None:
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1)
        if not group_exited:
            raise ExecutionError(
                f"timed-out process group {process_group_id} survived SIGKILL"
            )

    def _terminate_job(self, process: subprocess.Popen[Any]) -> None:
        if os.name == "posix":
            self._terminate_posix_job(process)
        elif process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()

    def _execute_node(self, node: ExperimentNode, commit: str) -> ResultEnvelope:
        self._verify_source_snapshot(node_id=node.id)
        environment_lock_sha256 = self._environment_lock_sha256(node)
        try:
            input_hashes = self._declared_hashes(node.inputs, kind=f"input for {node.id}")
        except ExecutionError as exc:
            raise ExecutionError(f"{node.id}: {exc}") from exc
        upstream = self._upstream_envelope_hashes(node, commit)
        # A committed/staged historical result must never satisfy this run's output contract.
        for rel in node.outputs:
            output = self.workspace / rel
            if output.is_symlink() or output.is_file():
                output.unlink()
            elif output.is_dir():
                shutil.rmtree(output)
        late_policy_path = self._prepare_late_control_policy(node)
        python_executable = str(Path(sys.executable).resolve())
        argv = [python_executable if token == "{python}" else token for token in node.command]
        stdout_path = self.log_dir / f"{node.id}.stdout.log"
        stderr_path = self.log_dir / f"{node.id}.stderr.log"
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update({
            "ECG_COMMIT": commit,
            "ECG_SOURCE_REPO": str(self.repo),
            "ECG_RUN_ID": self.run_id,
            "ECG_NODE_ID": node.id,
            "PYTHONUNBUFFERED": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            # Worker processes are budgeted by the node; nested BLAS/OpenMP
            # parallelism would oversubscribe the 8--10 CPU server allocation.
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "ECGCERT_NUM_WORKERS": str(max(1, node.resource.cpus)),
        })
        if late_policy_path is not None:
            env[POLICY_ENV] = str(late_policy_path)
        else:
            env.pop(POLICY_ENV, None)
        pythonpath = [str(self.workspace / "src"), str(self.workspace / "experiments")]
        if env.get("PYTHONPATH"):
            pythonpath.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(pythonpath)

        started_at = _utc_now()
        started = time.monotonic()
        self._status["nodes"][node.id].update({"state": "running", "started_at": started_at})
        self._write_status()
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            process = subprocess.Popen(
                argv, cwd=self.workspace, env=env, stdout=stdout, stderr=stderr,
                start_new_session=(os.name == "posix"), creationflags=creationflags,
            )
            timed_out = False
            try:
                exit_code = process.wait(timeout=node.timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                self._terminate_job(process)
                exit_code = 124
        finished_at = _utc_now()
        duration = time.monotonic() - started
        self._cpu_core_hours += duration * node.resource.cpus / 3600
        self._gpu_hours += duration * node.resource.gpus / 3600
        self._status["nodes"][node.id].update({
            "state": "failed" if exit_code else "validating",
            "exit_code": exit_code,
            "timed_out": timed_out,
            "finished_at": finished_at,
            "duration_seconds": round(duration, 3),
            "stdout": stdout_path.relative_to(self.run_dir).as_posix(),
            "stderr": stderr_path.relative_to(self.run_dir).as_posix(),
        })
        self._write_status()
        self._verify_source_snapshot(node_id=node.id)
        try:
            input_hashes_after = self._declared_hashes(
                node.inputs, kind=f"input for {node.id} after execution",
            )
        except ExecutionError as exc:
            raise ExecutionError(f"{node.id}: declared input disappeared or became unreadable: {exc}") from exc
        if input_hashes_after != input_hashes:
            changed = sorted(
                rel for rel in set(input_hashes) | set(input_hashes_after)
                if input_hashes.get(rel) != input_hashes_after.get(rel)
            )
            raise ExecutionError(f"{node.id}: command mutated declared inputs in place: {changed}")
        upstream_after = self._upstream_envelope_hashes(node, commit)
        if upstream_after != upstream:
            raise ExecutionError(f"{node.id}: command mutated dependency envelopes")
        if exit_code:
            raise ExecutionError(f"{node.id}: command failed with exit code {exit_code}")
        late_binding = self._finalize_late_control_policy(node, late_policy_path)
        if (
            self._global_cumulative_before["cpu_core_hours"] + self._cpu_core_hours
            > MAX_CPU_CORE_HOURS
            or self._global_cumulative_before["gpu_hours"] + self._gpu_hours
            > MAX_GPU_HOURS - RESERVED_GPU_HOURS
        ):
            raise ExecutionError("global execution exceeded the frozen compute budget")

        missing_outputs = [
            rel for rel in node.outputs if not (self.workspace / rel).exists()
        ]
        if missing_outputs:
            raise ExecutionError(f"{node.id}: missing declared outputs {missing_outputs}")
        try:
            outputs = self._declared_hashes(node.outputs, kind=f"output from {node.id}")
        except ExecutionError as exc:
            raise ExecutionError(f"{node.id}: {exc}") from exc
        self._artifact_bytes += late_binding.artifact_bytes
        self._artifact_bytes += sum(_path_bytes(self.workspace / rel) for rel in node.outputs)
        if (
            self._global_cumulative_before["artifact_bytes"] + self._artifact_bytes
            > MAX_ARTIFACT_BYTES
        ):
            raise ExecutionError("global execution exceeded the 100 GiB artifact budget")
        self._status["budget"]["used"] = {
            "cpu_core_hours": round(self._cpu_core_hours, 6),
            "gpu_hours": round(self._gpu_hours, 6),
            "artifact_bytes": self._artifact_bytes,
        }
        self._write_status()
        checkpoints = self._checkpoint_hashes((*node.inputs, *node.outputs))
        data_inputs = {
            rel: digest for rel, digest in input_hashes.items() if rel not in checkpoints
        }
        effective_data_inputs = dict(data_inputs)
        effective_data_inputs.update(
            {
                f"late-control:{rel}": digest
                for rel, digest in late_binding.inputs_sha256.items()
            }
        )
        split_material: dict[str, Any] = {
            "seed": node.seed,
            "argv": argv,
            "inputs": input_hashes,
            "deps": list(node.deps),
        }
        if node.late_control_inputs:
            split_material["late_control_inputs_sha256"] = late_binding.inputs_sha256
            split_material["late_control_snapshot_sha256"] = (
                late_binding.snapshot_sha256
            )
        split_sha = lineage.canonical_sha256(split_material)
        hardware = lineage.hardware_fingerprint()
        if node.resource.gpus and (
            not hardware.get("gpu") or hardware.get("cuda_runtime") == "unavailable"
        ):
            raise ExecutionError(f"{node.id}: GPU/CUDA inventory is unavailable")
        active_environment_sha256 = lineage.environment_sha256()
        if active_environment_sha256 != self._environment_sha256:
            raise ExecutionError(f"{node.id}: active Python environment changed during the run")
        envelope = ResultEnvelope(
            schema_version=SCHEMA_VERSION,
            run_id=self.run_id,
            node_id=node.id,
            status="succeeded",
            exit_code=0,
            started_at=started_at,
            finished_at=finished_at,
            commit=commit,
            dirty=False,
            argv=argv,
            config_sha256=self._effective_config_sha256(node, late_binding),
            data_sha256=lineage.canonical_sha256(effective_data_inputs),
            split_sha256=split_sha,
            env_sha256=self._environment_sha256,
            environment_lock_sha256=environment_lock_sha256,
            source_sha256=self._source_snapshot_sha256,
            hardware=hardware,
            seed=node.seed,
            upstream_sha256=upstream,
            late_control_inputs_sha256=late_binding.inputs_sha256,
            late_control_snapshot_sha256=late_binding.snapshot_sha256,
            checkpoint_sha256=checkpoints,
            outputs_sha256=outputs,
        )
        envelope.write(self.envelope_dir / f"{node.id}.json")
        return envelope

    def run(self) -> Path:
        selected = self.manifest.select(self.profile, self.resource)
        full_planned = self._planned_budget(selected)
        if (
            full_planned["cpu_core_hours"] > MAX_CPU_CORE_HOURS
            or full_planned["gpu_hours"] > MAX_GPU_HOURS - RESERVED_GPU_HOURS
        ):
            raise ExecutionError(
                f"manifest timeouts exceed frozen compute budget: {full_planned}"
            )
        commit, dirty = self._git_state(selected)
        if dirty:
            raise ExecutionError("refusing to run from a dirty source tree")
        if self.environment_lock is None:
            raise ExecutionError(
                "an explicit run-level environment_lock ('cpu' or 'gpu') is required"
            )
        try:
            self._environment_report = require_locked_environment(
                repo=self.repo,
                lock_name=self.environment_lock,
            )
        except (EnvironmentLockError, OSError, ValueError) as exc:
            raise ExecutionError(f"active environment verification failed: {exc}") from exc
        self._environment_sha256 = lineage.environment_sha256()
        limits = {
            "cpu_core_hours": MAX_CPU_CORE_HOURS,
            "gpu_hours": MAX_GPU_HOURS,
            "artifact_bytes": MAX_ARTIFACT_BYTES,
        }
        if self.resume:
            attempt_nodes = self._load_resume_status(
                selected=selected,
                commit=commit,
                limits=limits,
            )
        else:
            if self.run_dir.exists():
                raise ExecutionError(f"run directory already exists: {self.run_dir}")
            attempt_nodes = selected
        planned = self._planned_budget(attempt_nodes)
        lease = BudgetLease(
            control_root=self.control_root,
            run_id=self._budget_run_id,
            limits=limits,
            reserved_gpu_hours=RESERVED_GPU_HOURS,
        )
        try:
            reservation = lease.acquire(planned)
            self._global_cumulative_before = dict(reservation["cumulative_before"])
        except BudgetError as exc:
            raise ExecutionError(f"global budget/lease check failed: {exc}") from exc
        attempt_started = False
        try:
            if not self.resume:
                self.run_root.mkdir(parents=True, exist_ok=True)
                self.run_dir.mkdir()
                self._status = {
                    "schema_version": RUN_STATUS_SCHEMA_VERSION,
                    "run_id": self.run_id,
                    "control_root": str(self.control_root),
                    "profile": self.profile,
                    "resource": self.resource,
                    "environment_lock": self._environment_report.as_dict(),
                    "environment_sha256": self._environment_sha256,
                    "python_executable": str(Path(sys.executable).resolve()),
                    "manifest_sha256": self.manifest.sha256(),
                    "commit": commit,
                    "state": "staging",
                    "exit_code": None,
                    "started_at": _utc_now(),
                    "finished_at": None,
                    "nodes": {
                        node.id: {"state": "pending", "exit_code": None}
                        for node in selected
                    },
                    "attempts": [],
                    "budget": {
                        "limits": {
                            **limits,
                            "reserved_gpu_hours_for_rerun": RESERVED_GPU_HOURS,
                        },
                        "planned_timeout_upper_bound": full_planned,
                        "used": {
                            "cpu_core_hours": 0.0,
                            "gpu_hours": 0.0,
                            "artifact_bytes": 0,
                        },
                    },
                }
                self._write_status()
                self._stage_snapshot(commit, selected)
                self._status["source_snapshot_sha256"] = self._source_snapshot_sha256
                self._status["mutable_workspace_roots"] = list(self._mutable_roots)
                self._status["run_identity_sha256"] = self._run_identity_sha256()
            else:
                for node in attempt_nodes:
                    self._status["nodes"][node.id] = {
                        "state": "pending",
                        "exit_code": None,
                    }
                self._status.pop("error", None)
                self._status["finished_at"] = None
            self._status["state"] = "running"
            self._status["exit_code"] = None
            self._begin_attempt(
                selected=attempt_nodes,
                planned=planned,
                reservation=reservation,
            )
            attempt_started = True
            try:
                for node in attempt_nodes:
                    try:
                        self._execute_node(node, commit)
                    except Exception:
                        node_status = self._status["nodes"][node.id]
                        node_status["state"] = "failed"
                        if node_status.get("exit_code") is None:
                            node_status["exit_code"] = 1
                        node_status.setdefault("finished_at", _utc_now())
                        self._write_status()
                        raise
                    self._status["nodes"][node.id]["state"] = "succeeded"
                    self._write_status()
                self._status["state"] = "succeeded"
                self._status["exit_code"] = 0
            except Exception as exc:
                self._status["state"] = "failed"
                self._status["exit_code"] = 1
                self._status["error"] = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                self._status["finished_at"] = _utc_now()
                self._write_status()
        finally:
            used = {
                "cpu_core_hours": round(self._cpu_core_hours, 6),
                "gpu_hours": round(self._gpu_hours, 6),
                "artifact_bytes": self._artifact_bytes,
            }
            run_state = self._status.get("state", "failed")
            try:
                settlement = lease.settle(used, run_state=run_state)
                if self._status and self.run_dir.is_dir():
                    try:
                        ledger = validate_global_budget_ledger(
                            lease.ledger_path,
                            limits=limits,
                            reserved_gpu_hours=RESERVED_GPU_HOURS,
                        )
                    except (BudgetError, OSError, TypeError, ValueError) as exc:
                        raise ExecutionError(
                            f"cannot authenticate settled global budget ledger: {exc}"
                        ) from exc
                    if ledger["tail_event_sha256"] != settlement["event_sha256"]:
                        raise ExecutionError(
                            "settlement is not the authenticated global ledger tail"
                        )
                    self._status["budget"]["global_ledger"] = {
                        "ledger_relative_path": "budget-ledger.v1.jsonl",
                        "bound_event_count": ledger["event_count"],
                        "bound_tail_event_sha256": ledger["tail_event_sha256"],
                        "bound_cumulative_after": ledger["cumulative_after"],
                    }
                    settlement_path = self._attempt_dir / "budget-settlement.v1.json"
                    snapshot = {
                        "schema_version": SETTLEMENT_SCHEMA,
                        "reservation": reservation,
                        "settlement": settlement,
                    }
                    _atomic_json(settlement_path, snapshot)
                    if attempt_started:
                        self._finalize_attempt(
                            selected=attempt_nodes,
                            planned=planned,
                            reservation=reservation,
                            settlement=settlement,
                            settlement_path=settlement_path,
                            ledger_event_ordinal=ledger["event_count"] - 1,
                            used=used,
                        )
            finally:
                lease.release()
        return self.run_dir
