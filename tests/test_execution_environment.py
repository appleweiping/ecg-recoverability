from __future__ import annotations

import importlib.metadata
from pathlib import Path
import sys

import pytest

from ecgcert.execution.environment import (
    EnvironmentLockError,
    lock_relative_path,
    parse_locked_requirements,
    require_locked_environment,
    verify_locked_environment,
)


def _repo(tmp_path: Path, content: str) -> Path:
    repo = tmp_path / "repo"
    lock_root = repo / "environments"
    lock_root.mkdir(parents=True)
    (lock_root / "cpu.lock.txt").write_text(content, encoding="utf-8")
    (lock_root / "gpu.lock.txt").write_text(content, encoding="utf-8")
    (repo / ".python-version").write_text(
        ".".join(str(value) for value in sys.version_info[:3]) + "\n",
        encoding="utf-8",
    )
    return repo


def _lookup(versions: dict[str, str]):
    def lookup(name: str) -> str:
        try:
            return versions[name.lower()]
        except KeyError as exc:
            raise importlib.metadata.PackageNotFoundError(name) from exc

    return lookup


def test_lock_parser_accepts_hash_continuations_inline_hash_and_markers(tmp_path):
    repo = _repo(
        tmp_path,
        "--index-url https://example.invalid/simple\n"
        "alpha==1.2.3 \\\n"
        "    --hash=sha256:" + "a" * 64 + "\n"
        "beta==2.0 ; sys_platform == 'linux' --hash=sha256:" + "b" * 64 + "\n",
    )

    requirements = parse_locked_requirements(repo / "environments" / "gpu.lock.txt")

    assert [value.name for value in requirements] == ["alpha", "beta"]


def test_active_environment_checks_only_applicable_exact_pins(tmp_path):
    repo = _repo(
        tmp_path,
        "alpha==1.2.3\n"
        "linux-only==4.0 ; sys_platform == 'linux'\n"
        "windows-only==5.0 ; sys_platform == 'win32'\n",
    )
    report = verify_locked_environment(
        repo=repo,
        lock_name="gpu",
        version_lookup=_lookup({"alpha": "1.2.3", "linux-only": "4.0"}),
        marker_environment={"sys_platform": "linux"},
    )

    assert report.ok
    assert report.requirement_count == 3
    assert report.applicable_requirement_count == 2
    assert report.checked_requirement_count == 2
    assert report.lock_path == "environments/gpu.lock.txt"
    assert report.actual_python_version == report.expected_python_version


def test_missing_or_wrong_distribution_fails_closed(tmp_path):
    repo = _repo(tmp_path, "alpha==1.2.3\nbeta==2.0\n")
    report = verify_locked_environment(
        repo=repo,
        lock_name="cpu",
        version_lookup=_lookup({"alpha": "9.9"}),
    )
    assert not report.ok
    assert {item["actual"] for item in report.mismatches} == {"9.9", "missing"}
    with pytest.raises(EnvironmentLockError, match="alpha, beta"):
        require_locked_environment(
            repo=repo,
            lock_name="cpu",
            version_lookup=_lookup({"alpha": "9.9"}),
        )


@pytest.mark.parametrize(
    "content",
    [
        "alpha>=1.0\n",
        "alpha==1.*\n",
        "alpha==1.0\nAlpha==1.0\n",
        "--index-url https://example.invalid/simple\n",
    ],
)
def test_non_exact_or_empty_locks_are_rejected(tmp_path, content):
    repo = _repo(tmp_path, content)
    with pytest.raises(EnvironmentLockError):
        parse_locked_requirements(repo / "environments" / "cpu.lock.txt")


def test_lock_name_is_allowlisted():
    assert lock_relative_path("gpu").as_posix() == "environments/gpu.lock.txt"
    with pytest.raises(ValueError, match="unknown environment lock"):
        lock_relative_path("../../outside")


def test_missing_or_mismatched_frozen_python_version_fails_closed(tmp_path):
    repo = _repo(tmp_path, "alpha==1.2.3\n")
    (repo / ".python-version").unlink()
    with pytest.raises(EnvironmentLockError, match="no frozen"):
        verify_locked_environment(
            repo=repo,
            lock_name="gpu",
            version_lookup=_lookup({"alpha": "1.2.3"}),
        )

    (repo / ".python-version").write_text("0.0.1\n", encoding="utf-8")
    report = verify_locked_environment(
        repo=repo,
        lock_name="gpu",
        version_lookup=_lookup({"alpha": "1.2.3"}),
    )
    assert not report.ok
    assert report.mismatches[0]["package"] == "python"


def test_repository_gpu_lock_is_fully_parseable():
    root = Path(__file__).resolve().parents[1]
    requirements = parse_locked_requirements(root / "environments" / "gpu.lock.txt")
    assert len(requirements) >= 60
    assert any(value.name.lower() == "torch" for value in requirements)
