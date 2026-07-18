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
    if path.is_absolute() or ".." in path.parts or value.startswith("~"):
        raise ManifestError(f"{field} contains unsafe path: {value!r}")
    return path.as_posix()


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
    outputs: tuple[str, ...]
    timeout: int
    seed: int = 0

    @classmethod
    def parse(cls, value: Mapping[str, Any]) -> "ExperimentNode":
        required = {"id", "profile", "command", "resource", "deps", "inputs", "outputs", "timeout"}
        if not isinstance(value, Mapping):
            raise ManifestError("each node must be an object")
        missing = sorted(required - set(value))
        unknown = sorted(set(value) - required - {"seed"})
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
        outputs = value["outputs"]
        if not isinstance(deps, list) or not all(
            isinstance(x, str) and _SAFE_ID.fullmatch(x) for x in deps
        ):
            raise ManifestError(f"{node_id}: deps must contain safe node ids")
        if not isinstance(inputs, list) or not isinstance(outputs, list) or not outputs:
            raise ManifestError(f"{node_id}: inputs must be a list and outputs a non-empty list")
        timeout = value["timeout"]
        seed = value.get("seed", 0)
        if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout < 1:
            raise ManifestError(f"{node_id}: timeout must be a positive integer")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ManifestError(f"{node_id}: seed must be an integer")
        safe_inputs = tuple(_safe_relative(x, f"{node_id}.inputs") for x in inputs)
        safe_outputs = tuple(_safe_relative(x, f"{node_id}.outputs") for x in outputs)
        overlap = sorted(set(safe_inputs) & set(safe_outputs))
        if overlap:
            raise ManifestError(f"{node_id}: inputs and outputs overlap: {overlap}")
        return cls(
            id=node_id,
            profile=tuple(dict.fromkeys(profile)),
            command=tuple(command),
            resource=ResourceSpec.parse(value["resource"]),
            deps=tuple(dict.fromkeys(deps)),
            inputs=safe_inputs,
            outputs=safe_outputs,
            timeout=timeout,
            seed=seed,
        )

    def config_sha256(self) -> str:
        return canonical_sha256(asdict(self))


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
                owner = output_owner.get(input_path)
                if owner and owner not in node.deps:
                    raise ManifestError(
                        f"{node.id}: input {input_path!r} is produced by {owner} "
                        "but that node is not a dependency"
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
