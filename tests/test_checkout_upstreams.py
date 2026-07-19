from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest

from ecgcert.estimators.official import IMPUTE_ECG, UpstreamSpec, validate_pinned_checkout
from ecgcert.execution import ExperimentManifest
from scripts import checkout_upstreams


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _source_checkout(root: Path, name: str) -> tuple[Path, UpstreamSpec]:
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
    source = root / checkout_upstreams.checkout_name(spec)
    building.rename(source)
    return source, spec


def _sparse_ecgrecover_checkout(root: Path) -> tuple[Path, UpstreamSpec]:
    name = "ECGrecover"
    building = root / f".{name}-partial-building"
    building.mkdir(parents=True)
    _git(building, "init")
    _git(building, "config", "user.email", "test@example.invalid")
    _git(building, "config", "user.name", "Test")
    (building / "main.py").write_text("print('fixture')\n", encoding="utf-8")
    omitted = building / "omitted.bin"
    omitted.write_bytes(b"intentionally absent from sparse source")
    _git(building, "add", "main.py", "omitted.bin")
    _git(building, "commit", "-m", "partial fixture")
    commit = _git(building, "rev-parse", "HEAD")
    root_tree = _git(building, "rev-parse", "HEAD^{tree}")
    repository = "https://example.invalid/ECGrecover.git"
    _git(building, "remote", "add", "origin", repository)
    _git(building, "sparse-checkout", "init", "--no-cone")
    _git(building, "sparse-checkout", "set", "--no-cone", "/main.py")
    _git(building, "config", "remote.origin.promisor", "true")
    _git(building, "config", "remote.origin.partialclonefilter", "blob:none")
    _git(building, "config", "protocol.allow", "never")
    spec = UpstreamSpec(
        name=name,
        repository=repository,
        commit=commit,
        paper="fixture",
        root_tree=root_tree,
    )
    source = root / checkout_upstreams.checkout_name(spec)
    building.rename(source)
    return source, spec


def test_offline_materialization_succeeds_with_network_disabled(tmp_path, monkeypatch):
    source_root = tmp_path / "persistent" / "upstreams"
    source_root.mkdir(parents=True)
    sources_and_specs = [
        _source_checkout(source_root, "FixtureOne"),
        _source_checkout(source_root, "FixtureTwo"),
    ]
    specs = tuple(spec for _, spec in sources_and_specs)
    dangling = subprocess.run(
        ["git", "hash-object", "-w", "--stdin"],
        cwd=sources_and_specs[0][0],
        input="source-only dangling object\n",
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    real_run = subprocess.run

    def reject_network_git(command, *args, **kwargs):
        rendered = [str(value) for value in command]
        assert "fetch" not in rendered, "offline materialization must never fetch"
        if "clone" in rendered:
            clone_index = rendered.index("clone")
            clone_arguments = rendered[clone_index + 1 :]
            assert not any(value.startswith(("http://", "https://", "ssh://")) for value in clone_arguments)
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(checkout_upstreams.subprocess, "run", reject_network_git)
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    destination = tmp_path / "artifacts" / "upstreams"

    targets = checkout_upstreams.checkout_offline(
        specs,
        destination,
        source_root=source_root,
    )

    assert set(targets) == {spec.name for spec in specs}
    for source, spec in sources_and_specs:
        target = destination / checkout_upstreams.checkout_name(spec)
        assert validate_pinned_checkout(target, spec) == spec.commit
        assert _git(target, "config", "--get", "protocol.allow") == "never"
        assert not (target / ".git" / "objects" / "info" / "alternates").exists()
        assert target.resolve() != source.resolve()
        assert (target / "payload.txt").read_text(encoding="utf-8") == f"{spec.name}\n"
    assert subprocess.run(
        ["git", "cat-file", "-e", dangling],
        cwd=destination / checkout_upstreams.checkout_name(specs[0]),
        capture_output=True,
        check=False,
    ).returncode != 0


def test_offline_materialization_accepts_valid_sparse_source(tmp_path):
    source_root = tmp_path / "persistent" / "upstreams"
    source_root.mkdir(parents=True)
    _, spec = _sparse_ecgrecover_checkout(source_root)
    destination = tmp_path / "artifacts" / "upstreams"

    targets = checkout_upstreams.checkout_offline(
        (spec,),
        destination,
        source_root=source_root,
    )

    target = targets[spec.name]
    assert (target / "main.py").is_file()
    assert not (target / "omitted.bin").exists()
    assert _git(target, "config", "--get", "protocol.allow") == "never"
    assert _git(target, "fsck", "--connectivity-only") == ""


def test_offline_materialization_rejects_wrong_source_commit_before_writing(tmp_path):
    source_root = tmp_path / "persistent" / "upstreams"
    source_root.mkdir(parents=True)
    source, spec = _source_checkout(source_root, "WrongCommit")
    (source / "payload.txt").write_text("changed\n", encoding="utf-8")
    _git(source, "add", "payload.txt")
    _git(source, "commit", "-m", "unexpected")
    destination = tmp_path / "artifacts" / "upstreams"

    with pytest.raises(ValueError, match="checkout is"):
        checkout_upstreams.checkout_offline(
            (spec,),
            destination,
            source_root=source_root,
        )

    assert not os.path.lexists(destination)


def test_offline_materialization_rejects_dirty_source_before_writing(tmp_path):
    source_root = tmp_path / "persistent" / "upstreams"
    source_root.mkdir(parents=True)
    source, spec = _source_checkout(source_root, "DirtySource")
    (source / "untracked.txt").write_text("dirty\n", encoding="utf-8")
    destination = tmp_path / "artifacts" / "upstreams"

    with pytest.raises(ValueError, match="checkout is dirty"):
        checkout_upstreams.checkout_offline(
            (spec,),
            destination,
            source_root=source_root,
        )

    assert not os.path.lexists(destination)


def test_offline_materialization_refuses_destination_collision(tmp_path):
    source_root = tmp_path / "persistent" / "upstreams"
    source_root.mkdir(parents=True)
    _, spec = _source_checkout(source_root, "Collision")
    destination = tmp_path / "artifacts" / "upstreams"
    destination.mkdir(parents=True)
    sentinel = destination / "keep.txt"
    sentinel.write_text("keep\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        checkout_upstreams.checkout_offline(
            (spec,),
            destination,
            source_root=source_root,
        )

    assert sentinel.read_text(encoding="utf-8") == "keep\n"


def test_canonical_checkout_node_is_offline_and_binds_direct_source_input():
    root = Path(__file__).resolve().parents[1]
    manifest = ExperimentManifest.from_path(root / "scripts" / "experiment_manifest.yaml")
    node = manifest.by_id()["public_baseline_checkouts"]

    assert node.inputs == ("upstreams",)
    assert "--offline" in node.command
    assert node.command[node.command.index("--source-root") + 1] == "upstreams"
    assert node.command[node.command.index("--destination") + 1] == "artifacts/upstreams"


def test_imputeecg_pin_includes_frozen_root_tree():
    assert IMPUTE_ECG.root_tree == "d30565ea404a6b7f848fe3a9f5cc742655eb0388"
    assert {"train.py", "inference.py", "models/mae.py"} <= set(
        IMPUTE_ECG.required_paths
    )


def test_imputeecg_sparse_checkout_contains_runtime_code_not_demo_payloads():
    paths = checkout_upstreams._sparse_paths(IMPUTE_ECG)

    assert {"/train.py", "/inference.py", "/models/", "/utils/", "/datasets/"} <= set(
        paths
    )
    assert "/pics/" not in paths
    assert "/test_sample.csv" not in paths


def test_ecgrecover_sparse_checkout_excludes_tracked_python_bytecode():
    paths = checkout_upstreams._sparse_paths(checkout_upstreams.ECG_RECOVER)

    assert "/learn/" in paths and "/tools/" in paths
    assert "!/learn/__pycache__/" in paths
    assert "!/tools/__pycache__/" in paths


def test_pinned_checkout_rejects_sparse_source_missing_runtime_path(tmp_path):
    source_root = tmp_path / "persistent" / "upstreams"
    source_root.mkdir(parents=True)
    source, spec = _source_checkout(source_root, "RequiredRuntime")
    strict = UpstreamSpec(
        name=spec.name,
        repository=spec.repository,
        commit=spec.commit,
        paper=spec.paper,
        root_tree=spec.root_tree,
        required_paths=("payload.txt", "missing-runtime.py"),
    )

    with pytest.raises(ValueError, match="lacks required runtime paths"):
        validate_pinned_checkout(source, strict)
