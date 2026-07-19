from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest

from ecgcert.estimators.official import UpstreamSpec
from ecgcert.execution import upstream_staging


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _repository(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".gitignore").write_text("/upstreams\n", encoding="utf-8")
    (repo / "README.md").write_text("fixture\n", encoding="utf-8")
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "add", ".gitignore", "README.md")
    _git(repo, "commit", "-m", "fixture")
    return repo.resolve()


def _checkout(root: Path, name: str) -> tuple[Path, UpstreamSpec]:
    building = root / f".{name}-building"
    building.mkdir(parents=True)
    _git(building, "init")
    _git(building, "config", "user.email", "test@example.invalid")
    _git(building, "config", "user.name", "Test")
    (building / "payload.txt").write_text(f"{name}\n", encoding="utf-8")
    _git(building, "add", "payload.txt")
    _git(building, "commit", "-m", "fixture")
    commit = _git(building, "rev-parse", "HEAD")
    root_tree = _git(building, "rev-parse", "HEAD^{tree}")
    repository = f"https://example.invalid/{name}.git"
    _git(building, "remote", "add", "origin", repository)
    spec = UpstreamSpec(
        name=name,
        repository=repository,
        commit=commit,
        paper="fixture",
        root_tree=root_tree,
    )
    source = root / f"{name}-{commit[:12]}"
    building.rename(source)
    return source, spec


def _tools(tmp_path: Path) -> tuple[Path, tuple[tuple[Path, UpstreamSpec], ...]]:
    tools = tmp_path / "persistent" / "tools"
    upstreams = tools / "upstreams"
    upstreams.mkdir(parents=True)
    values = (
        _checkout(upstreams, "FixtureOne"),
        _checkout(upstreams, "FixtureTwo"),
    )
    return tools.resolve(), values


def _require_directory_symlinks(tmp_path: Path, target: Path) -> None:
    probe = tmp_path / "symlink-probe"
    try:
        os.symlink(target, probe, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks are unavailable on this test host: {exc}")
    probe.unlink()


def test_upstream_link_is_exact_idempotent_and_lineage_validated(tmp_path):
    repo = _repository(tmp_path)
    tools, checkouts = _tools(tmp_path)
    specs = tuple(spec for _, spec in checkouts)
    _require_directory_symlinks(tmp_path, tools / "upstreams")

    created = upstream_staging.ensure_server_upstream_link(
        repo=repo,
        tools_root=tools,
        specs=specs,
    )
    existing = upstream_staging.ensure_server_upstream_link(
        repo=repo,
        tools_root=tools,
        specs=specs,
    )

    link = repo / "upstreams"
    assert created["state"] == "created"
    assert existing["state"] == "existing"
    assert link.is_symlink()
    assert Path(os.readlink(link)) == (tools / "upstreams").resolve()
    assert link.resolve(strict=True) == (tools / "upstreams").resolve()
    assert {item["commit"] for item in created["checkouts"]} == {
        spec.commit for spec in specs
    }


def test_upstream_link_refuses_existing_path_without_overwriting(tmp_path):
    repo = _repository(tmp_path)
    tools, checkouts = _tools(tmp_path)
    collision = repo / "upstreams"
    collision.mkdir()
    sentinel = collision / "keep.txt"
    sentinel.write_text("keep\n", encoding="utf-8")

    with pytest.raises(upstream_staging.UpstreamLinkError, match="refusing to overwrite"):
        upstream_staging.ensure_server_upstream_link(
            repo=repo,
            tools_root=tools,
            specs=tuple(spec for _, spec in checkouts),
        )

    assert sentinel.read_text(encoding="utf-8") == "keep\n"


def test_upstream_link_refuses_nonmatching_existing_symlink_on_posix(tmp_path):
    repo = _repository(tmp_path)
    tools, checkouts = _tools(tmp_path)
    intended = tools / "upstreams"
    _require_directory_symlinks(tmp_path, intended)
    unrelated = tmp_path / "unrelated-upstreams"
    unrelated.mkdir()
    link = repo / "upstreams"
    os.symlink(unrelated, link, target_is_directory=True)

    with pytest.raises(upstream_staging.UpstreamLinkError, match="refusing to overwrite"):
        upstream_staging.ensure_server_upstream_link(
            repo=repo,
            tools_root=tools,
            specs=tuple(spec for _, spec in checkouts),
        )

    assert link.is_symlink()
    assert Path(os.readlink(link)) == unrelated
    assert link.resolve(strict=True) == unrelated.resolve(strict=True)


def test_upstream_link_rejects_wrong_commit_before_creating_link(tmp_path):
    repo = _repository(tmp_path)
    tools, checkouts = _tools(tmp_path)
    source, _ = checkouts[0]
    (source / "payload.txt").write_text("changed\n", encoding="utf-8")
    _git(source, "add", "payload.txt")
    _git(source, "commit", "-m", "unexpected")

    with pytest.raises(upstream_staging.UpstreamLinkError, match="lineage validation"):
        upstream_staging.ensure_server_upstream_link(
            repo=repo,
            tools_root=tools,
            specs=tuple(spec for _, spec in checkouts),
        )

    assert not os.path.lexists(repo / "upstreams")


def test_upstream_link_rejects_dirty_checkout_before_creating_link(tmp_path):
    repo = _repository(tmp_path)
    tools, checkouts = _tools(tmp_path)
    source, _ = checkouts[1]
    (source / "untracked.txt").write_text("dirty\n", encoding="utf-8")

    with pytest.raises(upstream_staging.UpstreamLinkError, match="lineage validation"):
        upstream_staging.ensure_server_upstream_link(
            repo=repo,
            tools_root=tools,
            specs=tuple(spec for _, spec in checkouts),
        )

    assert not os.path.lexists(repo / "upstreams")
