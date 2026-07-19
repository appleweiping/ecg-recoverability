"""Submission-grade result envelopes.

The envelope is a sidecar to scientific outputs. It deliberately has no permissive defaults:
missing/null provenance, a dirty source tree, a nonzero exit, or a missing output hash invalidates
the result.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Any, Mapping

SCHEMA_VERSION = "ecg-result-envelope/v3"
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def _validate_hash(name: str, value: Any) -> None:
    if not isinstance(value, str) or not _HEX64.fullmatch(value):
        raise ValueError(f"{name} must be a full lowercase SHA-256")


def _validate_hash_map(name: str, value: Any, *, nonempty: bool = False) -> None:
    if not isinstance(value, Mapping) or (nonempty and not value):
        raise ValueError(f"{name} must be {'a non-empty ' if nonempty else 'an '}object")
    for key, digest in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{name} contains an empty/non-string key")
        _validate_hash(f"{name}[{key!r}]", digest)


def _reject_nulls(value: Any, path: str = "envelope") -> None:
    if value is None:
        raise ValueError(f"{path} must not be null")
    if isinstance(value, Mapping):
        for key, nested in value.items():
            _reject_nulls(nested, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _reject_nulls(nested, f"{path}[{index}]")


@dataclass(frozen=True)
class ResultEnvelope:
    schema_version: str
    run_id: str
    node_id: str
    status: str
    exit_code: int
    started_at: str
    finished_at: str
    commit: str
    dirty: bool
    argv: list[str]
    config_sha256: str
    data_sha256: str
    split_sha256: str
    env_sha256: str
    environment_lock_sha256: str
    source_sha256: str
    hardware: dict[str, Any]
    seed: int
    upstream_sha256: dict[str, str]
    late_control_inputs_sha256: dict[str, str]
    late_control_snapshot_sha256: str
    checkpoint_sha256: dict[str, str]
    outputs_sha256: dict[str, str]

    def validate(self) -> None:
        _reject_nulls(asdict(self))
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported envelope schema: {self.schema_version!r}")
        for name, value in (("run_id", self.run_id), ("node_id", self.node_id)):
            if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
                raise ValueError(f"{name} is not a safe identifier")
        if self.status != "succeeded" or self.exit_code != 0:
            raise ValueError("a result envelope requires status=succeeded and exit_code=0")
        if not self.started_at or not self.finished_at:
            raise ValueError("started_at and finished_at are required")
        if not _HEX40.fullmatch(self.commit):
            raise ValueError("commit must be a full lowercase 40-character git SHA")
        if self.dirty is not False:
            raise ValueError("dirty must be exactly false")
        if not isinstance(self.argv, list) or not self.argv or not all(
                isinstance(item, str) and item for item in self.argv):
            raise ValueError("argv must be a non-empty string list")
        for name in (
            "config_sha256",
            "data_sha256",
            "split_sha256",
            "env_sha256",
            "environment_lock_sha256",
            "source_sha256",
        ):
            _validate_hash(name, getattr(self, name))
        if not isinstance(self.hardware, Mapping) or not self.hardware:
            raise ValueError("hardware must be a non-empty object")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        _validate_hash_map("upstream_sha256", self.upstream_sha256)
        _validate_hash_map(
            "late_control_inputs_sha256", self.late_control_inputs_sha256
        )
        _validate_hash(
            "late_control_snapshot_sha256", self.late_control_snapshot_sha256
        )
        _validate_hash_map("checkpoint_sha256", self.checkpoint_sha256)
        _validate_hash_map("outputs_sha256", self.outputs_sha256, nonempty=True)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ResultEnvelope":
        if not isinstance(value, Mapping):
            raise ValueError("result envelope must be an object")
        required = set(cls.__dataclass_fields__)
        missing = sorted(key for key in required if key not in value or value[key] is None)
        extra = sorted(set(value) - required)
        if missing:
            raise ValueError(f"result envelope missing/null fields: {missing}")
        if extra:
            raise ValueError(f"result envelope has unknown fields: {extra}")
        envelope = cls(**{key: value[key] for key in required})
        envelope.validate()
        return envelope

    def write(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
        tmp.replace(path)

    @classmethod
    def read(cls, path: Path | str) -> "ResultEnvelope":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
