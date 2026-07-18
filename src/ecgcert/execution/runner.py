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
from .budget import BudgetError, BudgetLease, SETTLEMENT_SCHEMA
from .envelope import ResultEnvelope, SCHEMA_VERSION
from .manifest import ExperimentManifest, ExperimentNode

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_CHECKPOINT_SUFFIXES = frozenset({".pt", ".pth", ".ckpt", ".safetensors"})
MAX_GPU_HOURS = 500.0
MAX_CPU_CORE_HOURS = 4_000.0
MAX_ARTIFACT_BYTES = 100 * 1024**3
RESERVED_GPU_HOURS = 100.0
_CONTROLLED_MUTABLE_ROOTS = ("artifacts",)
_ENVIRONMENT_LOCKS = {
    "cpu": "environments/cpu.lock.txt",
    "gpu": "environments/gpu.lock.txt",
    "paper": "environments/cpu.lock.txt",
}


class ExecutionError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


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
    for value in {item for node in nodes for item in node.inputs}:
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
        if artifact.is_file() and artifact.suffix.lower() in _CHECKPOINT_SUFFIXES:
            checkpoints[rel] = _path_sha256(artifact)
        elif artifact.is_dir():
            for checkpoint in sorted(
                path for path in artifact.rglob("*")
                if path.is_file() and path.suffix.lower() in _CHECKPOINT_SUFFIXES
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
        control_root: Path | str | None = None,
    ) -> None:
        self.repo = Path(repo).resolve()
        self.manifest = manifest
        self.profile = profile
        self.resource = resource
        self.run_root = Path(run_root).resolve()
        self.control_root = Path(control_root or self.run_root).resolve()
        if not _SAFE_ID.fullmatch(run_id):
            raise ValueError("run_id is not a safe identifier")
        self.run_id = run_id
        self.run_dir = self.run_root / run_id
        self.workspace = self.run_dir / "workspace"
        self.status_path = self.run_dir / "status.json"
        self.envelope_dir = self.run_dir / "envelopes"
        self.log_dir = self.run_dir / "logs"
        self._status: dict[str, Any] = {}
        self._cpu_core_hours = 0.0
        self._gpu_hours = 0.0
        self._artifact_bytes = 0
        self._mutable_roots: tuple[str, ...] = ()
        self._source_snapshot_sha256 = ""
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
            if (
                dependency.run_id != self.run_id
                or dependency.node_id != dep
                or dependency.commit != commit
                or dependency.config_sha256 != expected_node.config_sha256()
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
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                target.symlink_to(source, target_is_directory=source.is_dir())
            except OSError:
                if source.is_dir():
                    shutil.copytree(source, target)
                else:
                    shutil.copy2(source, target)
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
        relative = _ENVIRONMENT_LOCKS[node.resource.kind]
        lock = self.workspace / relative
        if not lock.is_file():
            raise ExecutionError(f"{node.id}: required environment lock is missing: {relative}")
        return lineage.artifact_sha256(lock)

    def _write_status(self) -> None:
        _atomic_json(self.status_path, self._status)

    def _terminate_job(self, process: subprocess.Popen[Any]) -> None:
        if process.poll() is not None:
            return
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=5)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        else:
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
        argv = [sys.executable if token == "{python}" else token for token in node.command]
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
        split_sha = lineage.canonical_sha256({
            "seed": node.seed, "argv": argv, "inputs": input_hashes, "deps": list(node.deps),
        })
        hardware = lineage.hardware_fingerprint()
        if node.resource.gpus and (
            not hardware.get("gpu") or hardware.get("cuda_runtime") == "unavailable"
        ):
            raise ExecutionError(f"{node.id}: GPU/CUDA inventory is unavailable")
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
            config_sha256=node.config_sha256(),
            data_sha256=lineage.canonical_sha256(data_inputs),
            split_sha256=split_sha,
            env_sha256=lineage.environment_sha256(),
            environment_lock_sha256=environment_lock_sha256,
            source_sha256=self._source_snapshot_sha256,
            hardware=hardware,
            seed=node.seed,
            upstream_sha256=upstream,
            checkpoint_sha256=checkpoints,
            outputs_sha256=outputs,
        )
        envelope.write(self.envelope_dir / f"{node.id}.json")
        return envelope

    def run(self) -> Path:
        selected = self.manifest.select(self.profile, self.resource)
        planned = self._planned_budget(selected)
        if (
            planned["cpu_core_hours"] > MAX_CPU_CORE_HOURS
            or planned["gpu_hours"] > MAX_GPU_HOURS - RESERVED_GPU_HOURS
        ):
            raise ExecutionError(f"manifest timeouts exceed frozen compute budget: {planned}")
        commit, dirty = self._git_state(selected)
        if dirty:
            raise ExecutionError("refusing to run from a dirty source tree")
        limits = {
            "cpu_core_hours": MAX_CPU_CORE_HOURS,
            "gpu_hours": MAX_GPU_HOURS,
            "artifact_bytes": MAX_ARTIFACT_BYTES,
        }
        lease = BudgetLease(
            control_root=self.control_root,
            run_id=self.run_id,
            limits=limits,
            reserved_gpu_hours=RESERVED_GPU_HOURS,
        )
        try:
            reservation = lease.acquire(planned)
            self._global_cumulative_before = dict(reservation["cumulative_before"])
        except BudgetError as exc:
            raise ExecutionError(f"global budget/lease check failed: {exc}") from exc
        settlement: dict[str, Any] | None = None
        try:
            self.run_root.mkdir(parents=True, exist_ok=True)
            try:
                self.run_dir.mkdir()
            except FileExistsError as exc:
                raise ExecutionError(f"run directory already exists: {self.run_dir}") from exc
            self._status = {
                "schema_version": 2,
                "run_id": self.run_id,
                "profile": self.profile,
                "resource": self.resource,
                "manifest_sha256": self.manifest.sha256(),
                "commit": commit,
                "state": "staging",
                "exit_code": None,
                "started_at": _utc_now(),
                "finished_at": None,
                "nodes": {
                    node.id: {"state": "pending", "exit_code": None} for node in selected
                },
                "budget": {
                    "limits": {
                        **limits,
                        "reserved_gpu_hours_for_rerun": RESERVED_GPU_HOURS,
                    },
                    "planned_timeout_upper_bound": planned,
                    "used": {
                        "cpu_core_hours": 0.0,
                        "gpu_hours": 0.0,
                        "artifact_bytes": 0,
                    },
                    "global_reservation_sha256": reservation["event_sha256"],
                    "cumulative_before": reservation["cumulative_before"],
                },
            }
            self._write_status()
            try:
                self._stage_snapshot(commit, selected)
                self._status["source_snapshot_sha256"] = self._source_snapshot_sha256
                self._status["mutable_workspace_roots"] = list(self._mutable_roots)
                self._status["state"] = "running"
                self._write_status()
                for node in selected:
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
                    settlement_path = self.run_dir / "budget-settlement.v1.json"
                    snapshot = {
                        "schema_version": SETTLEMENT_SCHEMA,
                        "reservation": reservation,
                        "settlement": settlement,
                    }
                    _atomic_json(settlement_path, snapshot)
                    self._status["budget"]["global_settlement_sha256"] = settlement[
                        "event_sha256"
                    ]
                    self._status["budget"]["settlement_artifact_sha256"] = (
                        lineage.artifact_sha256(settlement_path)
                    )
                    self._write_status()
            finally:
                lease.release()
        return self.run_dir
