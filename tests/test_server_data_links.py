from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest

from ecgcert.data import staging


def _git(repo: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", *arguments],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )


def _repository(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".gitignore").write_text("/data/\n", encoding="utf-8")
    (repo / "README.md").write_text("fixture\n", encoding="utf-8")
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "add", ".gitignore", "README.md")
    _git(repo, "commit", "-m", "fixture")
    return repo.resolve()


def _data_root(tmp_path: Path, *, cohorts: tuple[str, ...] = ("ptbxl", "chapman", "cpsc2018")) -> Path:
    root = tmp_path / "persistent" / "ecg-data"
    for cohort in cohorts:
        cohort_root = root / cohort
        cohort_root.mkdir(parents=True, exist_ok=True)
        (cohort_root / "sentinel.txt").write_text(cohort, encoding="utf-8")
    return root.resolve()


def _symlink_or_skip(link: Path, target: Path) -> None:
    try:
        os.symlink(target, link, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks are unavailable on this test host: {exc}")


def _require_directory_symlinks(tmp_path: Path, target: Path) -> None:
    probe = tmp_path / "symlink-probe"
    _symlink_or_skip(probe, target)
    probe.unlink()


def test_link_plan_uses_only_three_exact_ignored_paths(tmp_path, monkeypatch):
    repo = _repository(tmp_path)
    data_root = _data_root(tmp_path)
    calls: list[tuple[Path, Path, bool]] = []

    def record_symlink(source, destination, *, target_is_directory):
        calls.append((Path(source), Path(destination), target_is_directory))

    monkeypatch.setattr(staging.os, "symlink", record_symlink)
    monkeypatch.setattr(
        staging,
        "_existing_link_matches",
        lambda link, target: any(
            destination == link and source == target for source, destination, _ in calls
        ),
    )

    report = staging.ensure_server_data_links(repo=repo, data_root=data_root)

    expected = {
        (data_root / "ptbxl", repo / "data" / "ptbxl", True),
        (data_root / "chapman", repo / "data" / "external" / "chapman", True),
        (data_root / "cpsc2018", repo / "data" / "external" / "cpsc2018", True),
    }
    assert set(calls) == expected
    assert {item["state"] for item in report["links"]} == {"created"}
    assert sorted(
        path.relative_to(repo).as_posix() for path in (repo / "data").rglob("*")
    ) == ["data/external"]


def test_exact_links_are_idempotent(tmp_path):
    repo = _repository(tmp_path)
    data_root = _data_root(tmp_path)
    (repo / "data" / "external").mkdir(parents=True)
    _symlink_or_skip(repo / "data" / "ptbxl", data_root / "ptbxl")
    _symlink_or_skip(repo / "data" / "external" / "chapman", data_root / "chapman")
    _symlink_or_skip(repo / "data" / "external" / "cpsc2018", data_root / "cpsc2018")

    before = {
        link: os.readlink(link)
        for link in (
            repo / "data" / "ptbxl",
            repo / "data" / "external" / "chapman",
            repo / "data" / "external" / "cpsc2018",
        )
    }
    report = staging.ensure_server_data_links(repo=repo, data_root=data_root)

    assert {item["state"] for item in report["links"]} == {"existing"}
    assert {link: os.readlink(link) for link in before} == before


@pytest.mark.parametrize("collision_kind", ["file", "directory"])
def test_existing_non_link_is_never_overwritten(tmp_path, collision_kind):
    repo = _repository(tmp_path)
    data_root = _data_root(tmp_path)
    collision = repo / "data" / "ptbxl"
    collision.parent.mkdir()
    if collision_kind == "file":
        collision.write_text("keep me", encoding="utf-8")
    else:
        collision.mkdir()
        (collision / "keep.txt").write_text("keep me", encoding="utf-8")

    with pytest.raises(staging.DataLinkError, match="refusing to overwrite"):
        staging.ensure_server_data_links(repo=repo, data_root=data_root)

    assert collision.exists()
    if collision_kind == "file":
        assert collision.read_text(encoding="utf-8") == "keep me"
    else:
        assert (collision / "keep.txt").read_text(encoding="utf-8") == "keep me"


def test_wrong_or_broken_symlink_is_never_replaced(tmp_path):
    repo = _repository(tmp_path)
    data_root = _data_root(tmp_path)
    (repo / "data").mkdir()
    wrong = repo / "data" / "ptbxl"
    missing = tmp_path / "missing-target"
    _symlink_or_skip(wrong, missing)

    with pytest.raises(staging.DataLinkError, match="refusing to overwrite"):
        staging.ensure_server_data_links(repo=repo, data_root=data_root)

    assert wrong.is_symlink()
    assert Path(os.readlink(wrong)) == missing


def test_missing_source_is_rejected_before_repository_layout_is_created(tmp_path):
    repo = _repository(tmp_path)
    data_root = _data_root(tmp_path, cohorts=("ptbxl", "chapman"))

    with pytest.raises(staging.DataLinkError, match="cpsc2018 target"):
        staging.ensure_server_data_links(repo=repo, data_root=data_root)

    assert not (repo / "data").exists()


def test_mid_transaction_failure_removes_only_new_exact_links_and_empty_parents(
    tmp_path, monkeypatch,
):
    repo = _repository(tmp_path)
    data_root = _data_root(tmp_path)
    data_parent = repo / "data"
    data_parent.mkdir()
    existing = data_parent / "ptbxl"
    _symlink_or_skip(existing, data_root / "ptbxl")
    original_symlink = os.symlink

    def fail_for_cpsc(source, destination, *, target_is_directory):
        if Path(destination).name == "cpsc2018":
            raise OSError("injected creation failure")
        original_symlink(source, destination, target_is_directory=target_is_directory)

    monkeypatch.setattr(staging.os, "symlink", fail_for_cpsc)

    with pytest.raises(staging.DataLinkError, match="cannot create dataset symlink"):
        staging.ensure_server_data_links(repo=repo, data_root=data_root)

    assert existing.is_symlink()
    assert existing.resolve(strict=True) == (data_root / "ptbxl").resolve(strict=True)
    assert not (data_parent / "external").exists()
    assert data_parent.is_dir()


def test_failure_rolls_back_all_layout_created_by_call(tmp_path, monkeypatch):
    repo = _repository(tmp_path)
    data_root = _data_root(tmp_path)
    _require_directory_symlinks(tmp_path, data_root / "ptbxl")
    original_symlink = os.symlink

    def fail_for_chapman(source, destination, *, target_is_directory):
        if Path(destination).name == "chapman":
            raise OSError("injected creation failure")
        original_symlink(source, destination, target_is_directory=target_is_directory)

    monkeypatch.setattr(staging.os, "symlink", fail_for_chapman)

    with pytest.raises(staging.DataLinkError, match="cannot create dataset symlink"):
        staging.ensure_server_data_links(repo=repo, data_root=data_root)

    assert not (repo / "data").exists()


def test_rollback_preserves_created_link_if_its_target_changed(tmp_path, monkeypatch):
    repo = _repository(tmp_path)
    data_root = _data_root(tmp_path)
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    _require_directory_symlinks(tmp_path, replacement)
    original_symlink = os.symlink

    def mutate_then_fail(source, destination, *, target_is_directory):
        destination = Path(destination)
        if destination.name == "chapman":
            first = repo / "data" / "ptbxl"
            first.unlink()
            original_symlink(replacement, first, target_is_directory=True)
            raise OSError("injected creation failure")
        original_symlink(source, destination, target_is_directory=target_is_directory)

    monkeypatch.setattr(staging.os, "symlink", mutate_then_fail)

    with pytest.raises(staging.DataLinkError, match="cannot create dataset symlink"):
        staging.ensure_server_data_links(repo=repo, data_root=data_root)

    changed = repo / "data" / "ptbxl"
    assert changed.is_symlink()
    assert Path(os.readlink(changed)) == replacement
    assert not (repo / "data" / "external").exists()


def test_rollback_preserves_replaced_link_even_with_same_target(tmp_path, monkeypatch):
    repo = _repository(tmp_path)
    data_root = _data_root(tmp_path)
    _require_directory_symlinks(tmp_path, data_root / "ptbxl")
    original_symlink = os.symlink

    def replace_then_fail(source, destination, *, target_is_directory):
        destination = Path(destination)
        if destination.name == "chapman":
            first = repo / "data" / "ptbxl"
            first.unlink()
            original_symlink(data_root / "ptbxl", first, target_is_directory=True)
            raise OSError("injected creation failure")
        original_symlink(source, destination, target_is_directory=target_is_directory)

    monkeypatch.setattr(staging.os, "symlink", replace_then_fail)

    with pytest.raises(staging.DataLinkError, match="cannot create dataset symlink"):
        staging.ensure_server_data_links(repo=repo, data_root=data_root)

    replaced = repo / "data" / "ptbxl"
    assert replaced.is_symlink()
    assert Path(os.readlink(replaced)) == data_root / "ptbxl"


def test_rollback_preserves_new_parent_that_is_no_longer_empty(tmp_path, monkeypatch):
    repo = _repository(tmp_path)
    data_root = _data_root(tmp_path)
    _require_directory_symlinks(tmp_path, data_root / "ptbxl")
    original_symlink = os.symlink

    def add_unrelated_file_then_fail(source, destination, *, target_is_directory):
        destination = Path(destination)
        if destination.name == "chapman":
            (destination.parent / "concurrent.txt").write_text("keep", encoding="utf-8")
            raise OSError("injected creation failure")
        original_symlink(source, destination, target_is_directory=target_is_directory)

    monkeypatch.setattr(staging.os, "symlink", add_unrelated_file_then_fail)

    with pytest.raises(staging.DataLinkError, match="cannot create dataset symlink"):
        staging.ensure_server_data_links(repo=repo, data_root=data_root)

    marker = repo / "data" / "external" / "concurrent.txt"
    assert marker.read_text(encoding="utf-8") == "keep"
    assert not (repo / "data" / "ptbxl").exists()


def test_symlinked_repository_parent_is_rejected(tmp_path):
    repo = _repository(tmp_path)
    data_root = _data_root(tmp_path)
    redirected = tmp_path / "redirected-data"
    redirected.mkdir()
    _symlink_or_skip(repo / "data", redirected)

    with pytest.raises(staging.DataLinkError, match="parent is not a real directory"):
        staging.ensure_server_data_links(repo=repo, data_root=data_root)

    assert not list(redirected.iterdir())


def test_relative_arguments_are_rejected(tmp_path):
    repo = _repository(tmp_path)
    data_root = _data_root(tmp_path)

    with pytest.raises(staging.DataLinkError, match="repo must be an absolute path"):
        staging.ensure_server_data_links(repo=Path("repo"), data_root=data_root)
    with pytest.raises(staging.DataLinkError, match="data root must be an absolute path"):
        staging.ensure_server_data_links(repo=repo, data_root=Path("ecg-data"))
