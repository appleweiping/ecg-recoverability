"""Bind an executing interpreter to one frozen requirements lock.

The DAG resource class describes scheduling (CPU count, GPU count, memory), not
which Python environment happens to launch a node.  A mixed CPU/GPU profile is
therefore executed by one explicitly selected interpreter and one run-level
lock.  This module verifies every applicable exact requirement in that lock
against the active interpreter without invoking ``pip`` or enumerating
unrelated installed packages.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.metadata
from pathlib import Path, PurePosixPath
import sys
from typing import Callable, Mapping

from packaging.markers import default_environment
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name


LOCK_PATHS: Mapping[str, PurePosixPath] = {
    "cpu": PurePosixPath("environments/cpu.lock.txt"),
    "gpu": PurePosixPath("environments/gpu.lock.txt"),
}


class EnvironmentLockError(RuntimeError):
    """Raised when the active interpreter does not satisfy its declared lock."""


@dataclass(frozen=True)
class LockedEnvironmentReport:
    lock_name: str
    lock_path: str
    lock_sha256: str
    python_executable: str
    expected_python_version: str
    actual_python_version: str
    requirement_count: int
    applicable_requirement_count: int
    checked_requirement_count: int
    mismatches: tuple[dict[str, str], ...]

    @property
    def ok(self) -> bool:
        return not self.mismatches

    def as_dict(self) -> dict[str, object]:
        return {
            "lock_name": self.lock_name,
            "lock_path": self.lock_path,
            "lock_sha256": self.lock_sha256,
            "python_executable": self.python_executable,
            "expected_python_version": self.expected_python_version,
            "actual_python_version": self.actual_python_version,
            "requirement_count": self.requirement_count,
            "applicable_requirement_count": self.applicable_requirement_count,
            "checked_requirement_count": self.checked_requirement_count,
            "mismatches": [dict(value) for value in self.mismatches],
            "ok": self.ok,
        }


def lock_relative_path(lock_name: str) -> PurePosixPath:
    try:
        return LOCK_PATHS[lock_name]
    except KeyError as exc:
        raise ValueError(f"unknown environment lock: {lock_name!r}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_locked_requirements(path: Path) -> tuple[Requirement, ...]:
    """Parse the exact requirement lines from a pip-compile hash lock."""

    requirements: list[Requirement] = []
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        stripped = raw.strip()
        if (
            not stripped
            or stripped.startswith("#")
            or stripped.startswith("--")
            or stripped.startswith("\\")
            or stripped.startswith("--hash=")
        ):
            continue
        # Requirement records are the only non-indented records. Hashes are
        # continuation lines, and the final backslash is pip syntax rather
        # than part of the PEP 508 requirement.
        if raw[:1].isspace():
            continue
        candidate = stripped.removesuffix("\\").rstrip()
        if " --hash=" in candidate:
            candidate = candidate.split(" --hash=", 1)[0].rstrip()
        try:
            requirement = Requirement(candidate)
        except InvalidRequirement as exc:
            raise EnvironmentLockError(
                f"invalid requirement at {path}:{line_number}"
            ) from exc
        if requirement.url is not None or len(requirement.specifier) != 1:
            raise EnvironmentLockError(
                f"lock requirement is not one exact package pin at {path}:{line_number}"
            )
        specifier = next(iter(requirement.specifier))
        if specifier.operator not in {"==", "==="} or specifier.version.endswith(".*"):
            raise EnvironmentLockError(
                f"lock requirement is not exact at {path}:{line_number}"
            )
        requirements.append(requirement)
    if not requirements:
        raise EnvironmentLockError(f"environment lock has no exact requirements: {path}")
    names = [canonicalize_name(requirement.name) for requirement in requirements]
    if len(names) != len(set(names)):
        raise EnvironmentLockError(f"environment lock contains duplicate packages: {path}")
    return tuple(requirements)


def verify_locked_environment(
    *,
    repo: Path,
    lock_name: str,
    version_lookup: Callable[[str], str] = importlib.metadata.version,
    marker_environment: Mapping[str, str] | None = None,
) -> LockedEnvironmentReport:
    """Verify applicable lock pins against the currently running interpreter."""

    relative = lock_relative_path(lock_name)
    repository = repo.resolve()
    version_file = repository / ".python-version"
    if not version_file.is_file():
        raise EnvironmentLockError("repository has no frozen .python-version")
    expected_python = version_file.read_text(encoding="utf-8").strip()
    parts = expected_python.split(".")
    if len(parts) != 3 or any(not part.isdigit() for part in parts):
        raise EnvironmentLockError("repository .python-version is not an exact X.Y.Z version")
    actual_python = ".".join(str(value) for value in sys.version_info[:3])
    lock = repository / Path(relative.as_posix())
    if not lock.is_file():
        raise EnvironmentLockError(f"required environment lock is missing: {relative}")
    requirements = parse_locked_requirements(lock)
    marker_values = dict(default_environment())
    if marker_environment is not None:
        marker_values.update({str(key): str(value) for key, value in marker_environment.items()})
    applicable = [
        requirement
        for requirement in requirements
        if requirement.marker is None or requirement.marker.evaluate(marker_values)
    ]
    mismatches: list[dict[str, str]] = []
    if actual_python != expected_python:
        mismatches.append(
            {
                "package": "python",
                "expected": f"=={expected_python}",
                "actual": actual_python,
            }
        )
    checked = 0
    for requirement in applicable:
        try:
            actual = version_lookup(requirement.name)
        except importlib.metadata.PackageNotFoundError:
            mismatches.append(
                {
                    "package": requirement.name,
                    "expected": str(requirement.specifier),
                    "actual": "missing",
                }
            )
            continue
        except Exception as exc:
            raise EnvironmentLockError(
                f"cannot inspect locked package {requirement.name}: {type(exc).__name__}"
            ) from exc
        checked += 1
        if not requirement.specifier.contains(actual, prereleases=True):
            mismatches.append(
                {
                    "package": requirement.name,
                    "expected": str(requirement.specifier),
                    "actual": str(actual),
                }
            )
    return LockedEnvironmentReport(
        lock_name=lock_name,
        lock_path=relative.as_posix(),
        lock_sha256=_sha256(lock),
        python_executable=str(Path(sys.executable).resolve()),
        expected_python_version=expected_python,
        actual_python_version=actual_python,
        requirement_count=len(requirements),
        applicable_requirement_count=len(applicable),
        checked_requirement_count=checked,
        mismatches=tuple(mismatches),
    )


def require_locked_environment(
    *,
    repo: Path,
    lock_name: str,
    version_lookup: Callable[[str], str] = importlib.metadata.version,
    marker_environment: Mapping[str, str] | None = None,
) -> LockedEnvironmentReport:
    report = verify_locked_environment(
        repo=repo,
        lock_name=lock_name,
        version_lookup=version_lookup,
        marker_environment=marker_environment,
    )
    if not report.ok:
        packages = ", ".join(value["package"] for value in report.mismatches[:8])
        suffix = "..." if len(report.mismatches) > 8 else ""
        raise EnvironmentLockError(
            f"active interpreter does not satisfy {report.lock_path}: {packages}{suffix}"
        )
    return report


__all__ = [
    "EnvironmentLockError",
    "LOCK_PATHS",
    "LockedEnvironmentReport",
    "lock_relative_path",
    "parse_locked_requirements",
    "require_locked_environment",
    "verify_locked_environment",
]
