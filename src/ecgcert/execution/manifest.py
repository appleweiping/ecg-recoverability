"""Typed experiment-manifest parsing and DAG validation."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
import re
from typing import Any, Iterable, Mapping

import yaml

from ecgcert.lineage import canonical_sha256

ALLOWED_PROFILES = frozenset({"icassp", "extended", "legacy"})
ALLOWED_RESOURCES = frozenset({"cpu", "gpu", "paper"})
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class ManifestError(ValueError):
    pass


def _safe_relative(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{field} entries must be non-empty strings")
    value = value.replace("\\", "/")
    path = PurePosixPath(value)
    # A manifest is executed on both POSIX and Windows hosts.  PurePosixPath
    # intentionally does not recognise Windows drive-qualified paths, and a
    # colon can also name an NTFS alternate data stream.  Reject both forms
    # before joining the value to a workspace root.
    if (
        path.is_absolute()
        or ".." in path.parts
        or value.startswith("~")
        or ":" in value
        or "\x00" in value
        or path.as_posix() == "."
    ):
        raise ManifestError(f"{field} contains unsafe path: {value!r}")
    return path.as_posix()


def _path_contains(parent: str, child: str) -> bool:
    """Return whether two normalized manifest paths have containment."""

    parent_path = PurePosixPath(parent)
    child_path = PurePosixPath(child)
    return parent_path == child_path or parent_path in child_path.parents


def _paths_overlap(left: str, right: str) -> bool:
    return _path_contains(left, right) or _path_contains(right, left)


@dataclass(frozen=True)
class ResourceSpec:
    kind: str
    cpus: int = 1
    memory_gb: int = 4
    gpus: int = 0

    @classmethod
    def parse(cls, value: Any) -> "ResourceSpec":
        if isinstance(value, str):
            value = {"kind": value}
        if not isinstance(value, Mapping):
            raise ManifestError("resource must be a string or object")
        unknown = set(value) - {"kind", "cpus", "memory_gb", "gpus"}
        if unknown:
            raise ManifestError(f"resource has unknown fields: {sorted(unknown)}")
        kind = value.get("kind")
        if kind not in ALLOWED_RESOURCES:
            raise ManifestError(f"resource.kind must be one of {sorted(ALLOWED_RESOURCES)}")
        cpus = value.get("cpus", 1)
        memory_gb = value.get("memory_gb", 4)
        gpus = value.get("gpus", 1 if kind == "gpu" else 0)
        if any(isinstance(x, bool) or not isinstance(x, int) for x in (cpus, memory_gb, gpus)):
            raise ManifestError("resource cpus/memory_gb/gpus must be integers")
        if cpus < 1 or cpus > 10 or memory_gb < 1 or gpus < 0 or (kind != "gpu" and gpus != 0):
            raise ManifestError("resource quantities are invalid")
        return cls(kind=kind, cpus=cpus, memory_gb=memory_gb, gpus=gpus)


@dataclass(frozen=True)
class ExperimentNode:
    id: str
    profile: tuple[str, ...]
    command: tuple[str, ...]
    resource: ResourceSpec
    deps: tuple[str, ...]
    inputs: tuple[str, ...]
    late_control_inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    timeout: int
    seed: int = 0

    @classmethod
    def parse(cls, value: Mapping[str, Any]) -> "ExperimentNode":
        required = {"id", "profile", "command", "resource", "deps", "inputs", "outputs", "timeout"}
        if not isinstance(value, Mapping):
            raise ManifestError("each node must be an object")
        missing = sorted(required - set(value))
        unknown = sorted(set(value) - required - {"seed", "late_control_inputs"})
        if missing:
            raise ManifestError(f"node missing fields: {missing}")
        if unknown:
            raise ManifestError(f"node has unknown fields: {unknown}")
        node_id = value["id"]
        if not isinstance(node_id, str) or not _SAFE_ID.fullmatch(node_id):
            raise ManifestError(f"unsafe node id: {node_id!r}")
        profile = value["profile"]
        if isinstance(profile, str):
            profile = [profile]
        if (
            not isinstance(profile, list)
            or not profile
            or not all(isinstance(x, str) for x in profile)
        ):
            raise ManifestError(f"{node_id}: profile must be a non-empty string/list")
        if not set(profile) <= ALLOWED_PROFILES:
            raise ManifestError(f"{node_id}: unknown profile in {profile}")
        command = value["command"]
        if not isinstance(command, list) or not command or not all(
                isinstance(x, str) and x for x in command):
            raise ManifestError(f"{node_id}: command must be a non-empty string list")
        deps = value["deps"]
        inputs = value["inputs"]
        late_control_inputs = value.get("late_control_inputs", [])
        outputs = value["outputs"]
        if not isinstance(deps, list) or not all(
            isinstance(x, str) and _SAFE_ID.fullmatch(x) for x in deps
        ):
            raise ManifestError(f"{node_id}: deps must contain safe node ids")
        if (
            not isinstance(inputs, list)
            or not isinstance(late_control_inputs, list)
            or not isinstance(outputs, list)
            or not outputs
        ):
            raise ManifestError(f"{node_id}: inputs must be a list and outputs a non-empty list")
        timeout = value["timeout"]
        seed = value.get("seed", 0)
        if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout < 1:
            raise ManifestError(f"{node_id}: timeout must be a positive integer")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ManifestError(f"{node_id}: seed must be an integer")
        safe_inputs = tuple(_safe_relative(x, f"{node_id}.inputs") for x in inputs)
        safe_late_inputs = tuple(
            _safe_relative(x, f"{node_id}.late_control_inputs")
            for x in late_control_inputs
        )
        safe_outputs = tuple(_safe_relative(x, f"{node_id}.outputs") for x in outputs)
        if len(set(safe_late_inputs)) != len(safe_late_inputs):
            raise ManifestError(f"{node_id}: late_control_inputs contains duplicates")
        if any(
            not _path_contains("artifacts/gates", path) for path in safe_late_inputs
        ):
            raise ManifestError(
                f"{node_id}: late_control_inputs must be confined to artifacts/gates"
            )
        if any(path not in command for path in safe_late_inputs):
            raise ManifestError(
                f"{node_id}: every late_control_inputs path must be an exact command token"
            )
        overlap = sorted(
            (input_path, output_path)
            for input_path in (*safe_inputs, *safe_late_inputs)
            for output_path in safe_outputs
            if _paths_overlap(input_path, output_path)
        )
        if overlap:
            raise ManifestError(f"{node_id}: inputs and outputs overlap: {overlap}")
        ordinary_late_overlap = sorted(
            (input_path, late_path)
            for input_path in safe_inputs
            for late_path in safe_late_inputs
            if _paths_overlap(input_path, late_path)
        )
        if ordinary_late_overlap:
            raise ManifestError(
                f"{node_id}: inputs and late_control_inputs overlap: "
                f"{ordinary_late_overlap}"
            )
        nested_late = sorted(
            (left, right)
            for index, left in enumerate(safe_late_inputs)
            for right in safe_late_inputs[index + 1 :]
            if _paths_overlap(left, right)
        )
        if nested_late:
            raise ManifestError(
                f"{node_id}: late_control_inputs overlap each other: {nested_late}"
            )
        return cls(
            id=node_id,
            profile=tuple(dict.fromkeys(profile)),
            command=tuple(command),
            resource=ResourceSpec.parse(value["resource"]),
            deps=tuple(dict.fromkeys(deps)),
            inputs=safe_inputs,
            late_control_inputs=safe_late_inputs,
            outputs=safe_outputs,
            timeout=timeout,
            seed=seed,
        )

    def config_sha256(
        self,
        *,
        late_control_inputs_sha256: Mapping[str, str] | None = None,
    ) -> str:
        """Hash the declaration and, for execution, realized control content."""

        payload: dict[str, Any] = {"node": asdict(self)}
        if late_control_inputs_sha256 is not None:
            if set(late_control_inputs_sha256) != set(self.late_control_inputs):
                raise ManifestError(
                    f"{self.id}: realized late-control input set does not match manifest"
                )
            payload["late_control_inputs_sha256"] = {
                path: late_control_inputs_sha256[path]
                for path in sorted(late_control_inputs_sha256)
            }
        return canonical_sha256(payload)


@dataclass(frozen=True)
class ExperimentManifest:
    schema_version: int
    nodes: tuple[ExperimentNode, ...]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExperimentManifest":
        if not isinstance(value, Mapping) or set(value) != {"schema_version", "nodes"}:
            raise ManifestError("manifest requires exactly schema_version and nodes")
        if value["schema_version"] != 1:
            raise ManifestError("manifest schema_version must be 1")
        if not isinstance(value["nodes"], list) or not value["nodes"]:
            raise ManifestError("manifest.nodes must be a non-empty list")
        manifest = cls(1, tuple(ExperimentNode.parse(node) for node in value["nodes"]))
        manifest.validate()
        return manifest

    @classmethod
    def from_path(cls, path: Path | str) -> "ExperimentManifest":
        try:
            value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise ManifestError(f"cannot read manifest {path}: {exc}") from exc
        return cls.from_dict(value)

    def validate(self) -> None:
        by_id: dict[str, ExperimentNode] = {}
        output_owner: dict[str, str] = {}
        for node in self.nodes:
            if node.id in by_id:
                raise ManifestError(f"duplicate node id: {node.id}")
            by_id[node.id] = node
            for output in node.outputs:
                if output in output_owner:
                    raise ManifestError(
                        f"duplicate output {output!r}: {output_owner[output]} and {node.id}")
                for owned_output, owner in output_owner.items():
                    if _paths_overlap(output, owned_output):
                        raise ManifestError(
                            f"nested outputs are ambiguous: {output!r} ({node.id}) and "
                            f"{owned_output!r} ({owner})"
                        )
                output_owner[output] = node.id
        for node in self.nodes:
            missing = [dep for dep in node.deps if dep not in by_id]
            if missing:
                raise ManifestError(f"{node.id}: missing dependencies {missing}")
            if node.id in node.deps:
                raise ManifestError(f"{node.id}: node cannot depend on itself")
            for dep in node.deps:
                absent_profiles = set(node.profile) - set(by_id[dep].profile)
                if absent_profiles:
                    raise ManifestError(
                        f"{node.id}: dependency {dep} is absent from profiles "
                        f"{sorted(absent_profiles)}"
                    )
            for input_path in node.inputs:
                for output_path, owner in output_owner.items():
                    if owner == node.id or not _paths_overlap(input_path, output_path):
                        continue
                    if owner not in node.deps:
                        raise ManifestError(
                            f"{node.id}: input {input_path!r} overlaps output "
                            f"{output_path!r} produced by {owner}, but that node is "
                            "not a dependency"
                        )
            for late_path in node.late_control_inputs:
                for output_path, owner in output_owner.items():
                    if _paths_overlap(late_path, output_path):
                        raise ManifestError(
                            f"{node.id}: late control input {late_path!r} overlaps DAG "
                            f"output {output_path!r} produced by {owner}; late controls "
                            "must come only from the authenticated external inbox"
                        )
            declared_command_paths = {
                *node.inputs,
                *node.late_control_inputs,
                *node.outputs,
            }
            undeclared_gate_tokens = sorted(
                token
                for token in node.command
                if _path_contains("artifacts/gates", token)
                and token not in declared_command_paths
            )
            if undeclared_gate_tokens:
                raise ManifestError(
                    f"{node.id}: command has undeclared artifacts/gates control paths: "
                    f"{undeclared_gate_tokens}"
                )
        self.topological()
        seen_profiles = {profile for node in self.nodes for profile in node.profile}
        missing_profiles = ALLOWED_PROFILES - seen_profiles
        if missing_profiles:
            raise ManifestError(f"manifest has no nodes for profiles: {sorted(missing_profiles)}")

    def by_id(self) -> dict[str, ExperimentNode]:
        return {node.id: node for node in self.nodes}

    def sha256(self) -> str:
        return canonical_sha256({
            "schema_version": self.schema_version,
            "nodes": [asdict(node) for node in self.nodes],
        })

    def topological(self, nodes: Iterable[ExperimentNode] | None = None) -> list[ExperimentNode]:
        chosen = list(nodes if nodes is not None else self.nodes)
        chosen_ids = {node.id for node in chosen}
        by_id = self.by_id()
        state: dict[str, int] = {}
        ordered: list[ExperimentNode] = []

        def visit(node_id: str) -> None:
            if state.get(node_id) == 1:
                raise ManifestError(f"dependency cycle includes {node_id}")
            if state.get(node_id) == 2 or node_id not in chosen_ids:
                return
            state[node_id] = 1
            for dep in by_id[node_id].deps:
                visit(dep)
            state[node_id] = 2
            ordered.append(by_id[node_id])

        for node in chosen:
            visit(node.id)
        return ordered

    def select(self, profile: str, resource: str | None = None) -> list[ExperimentNode]:
        if profile not in ALLOWED_PROFILES:
            raise ManifestError(f"unknown profile: {profile}")
        if resource is not None and resource not in ALLOWED_RESOURCES:
            raise ManifestError(f"unknown resource: {resource}")
        profile_nodes = [node for node in self.nodes if profile in node.profile]
        targets = [
            node for node in profile_nodes
            if resource is None or node.resource.kind == resource
        ]
        if not targets:
            raise ManifestError(f"profile={profile}, resource={resource} selects no nodes")
        if resource is None:
            return self.topological(targets)
        # A resource filter identifies requested target nodes, not a license to
        # omit their producers.  Isolated runs cannot borrow undeclared outputs
        # from an earlier CPU/GPU phase, so include the complete dependency
        # closure and retain topological order.
        by_id = self.by_id()
        chosen: set[str] = set()

        def include(node_id: str) -> None:
            if node_id in chosen:
                return
            node = by_id[node_id]
            if profile not in node.profile:
                raise ManifestError(f"{node_id} is outside selected profile {profile}")
            chosen.add(node_id)
            for dependency in node.deps:
                include(dependency)

        for node in targets:
            include(node.id)
        return self.topological(node for node in self.nodes if node.id in chosen)
