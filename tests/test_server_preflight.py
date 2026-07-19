import csv
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import server_preflight as preflight_cli
from ecgcert.execution import preflight
from ecgcert.execution.preflight import (
    PreflightConfig,
    SCHEMA_VERSION,
    collect_server_preflight,
    failed_preflight_report,
    write_report,
)


def _write_ptbxl(root: Path, count: int) -> None:
    (root / "records100").mkdir(parents=True)
    (root / "records500").mkdir(parents=True)
    rows = []
    for index in range(count):
        low = f"records100/{index:05d}_lr"
        high = f"records500/{index:05d}_hr"
        for stem in (root / low, root / high):
            stem.with_suffix(".hea").write_text("fixture 12 500 5000\n", encoding="utf-8")
            stem.with_suffix(".dat").write_bytes(b"signal")
        rows.append(
            {
                "ecg_id": str(index + 1),
                "patient_id": str(1000 + index),
                "strat_fold": str((index % 10) + 1),
                "filename_lr": low,
                "filename_hr": high,
            }
        )
    with (root / "ptbxl_database.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (root / "scp_statements.csv").write_text("code,diagnostic\nNORM,1\n", encoding="utf-8")


def _write_external(root: Path, count: int) -> None:
    root.mkdir(parents=True)
    for index in range(count):
        stem = root / f"record-{index:04d}"
        stem.with_suffix(".hea").write_text("fixture 12 500 5000\n", encoding="utf-8")
        stem.with_suffix(".mat").write_bytes(b"signal")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[PreflightConfig, dict, dict, dict[Path, str]]:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".python-version").write_text(
        ".".join(str(value) for value in sys.version_info[:3]) + "\n",
        encoding="utf-8",
    )
    (repo / "environments").mkdir()
    cpu_lock = repo / "environments" / "cpu.lock.txt"
    gpu_lock = repo / "environments" / "gpu.lock.txt"
    cpu_lock.write_text("fixture==1.0\n", encoding="utf-8")
    gpu_lock.write_text("fixture==1.0\n", encoding="utf-8")
    _write_ptbxl(repo / "data" / "ptbxl", 10)
    _write_external(repo / "data" / "external" / "chapman", 2)
    _write_external(repo / "data" / "external" / "cpsc2018", 1)
    main_commit = "a" * 40

    tools = tmp_path / "tools"
    upstreams = tools / "upstreams"
    upstreams.mkdir(parents=True)
    impute_commit = "b" * 40
    recover_commit = "c" * 40
    impute = upstreams / f"ImputeECG-{impute_commit[:12]}"
    recover = upstreams / f"ECGrecover-{recover_commit[:12]}"
    impute.mkdir()
    recover.mkdir()
    (impute / "required.py").write_text("# fixture\n", encoding="utf-8")
    (recover / "required.py").write_text("# fixture\n", encoding="utf-8")
    arc = tools / "AutoResearchClaw-v0.5.0"
    arc.mkdir()
    arc_commit = "d" * 40
    acpx_package = tools / "acpx-0.12.0" / "node_modules" / "acpx" / "package.json"
    acpx_package.parent.mkdir(parents=True)
    acpx_package.write_text('{"name":"acpx","version":"0.12.0"}\n', encoding="utf-8")

    config = PreflightConfig(
        repo=repo,
        expected_commit=main_commit,
        storage_root=tmp_path,
        tools_root=tools,
        ptbxl_root=repo / "data" / "ptbxl",
        chapman_root=repo / "data" / "external" / "chapman",
        cpsc2018_root=repo / "data" / "external" / "cpsc2018",
        min_free_bytes=0,
        expected_dataset_records={"ptbxl": 10, "chapman": 2, "cpsc2018": 1},
        expected_lock_sha256={"cpu": _sha256(cpu_lock), "gpu": _sha256(gpu_lock)},
        expected_tool_commits={
            "imputeecg": impute_commit,
            "ecgrecover": recover_commit,
            "autoresearchclaw": arc_commit,
        },
        expected_tool_root_trees={
            "imputeecg": "1" * 40,
            "ecgrecover": "2" * 40,
        },
        expected_tool_origins={
            "imputeecg": "https://example.invalid/ImputeECG.git",
            "ecgrecover": "https://example.invalid/ECGrecover.git",
        },
        expected_tool_required_paths={
            "imputeecg": ("required.py",),
            "ecgrecover": ("required.py",),
        },
        require_staged_links=False,
    )
    system = {
        "os": {"system": "Linux", "release": "fixture", "machine": "x86_64"},
        "python": {
            "implementation": "CPython",
            "version": "3.11.2",
            "version_info": [3, 11, 2],
        },
        "cpu": {"logical_count": 10, "model": "fixture"},
        "ram": {"total_bytes": 64 * 1024**3},
    }
    nvidia = {
        "available": True,
        "executable": "/usr/bin/nvidia-smi",
        "driver_version": "570.00",
        "cuda_version_reported": "12.8",
        "devices": [
            {
                "index": 0,
                "name": "fixture GPU",
                "uuid": "GPU-fixture",
                "driver_version": "570.00",
                "memory_total_mib": 24_576,
            }
        ],
        "nvcc_version": "12.8",
    }
    torch = {
        "installed": True,
        "version": "2.8.0+cu128",
        "compiled_cuda": "12.8",
        "cuda_available": True,
        "device_count": 1,
        "devices": [],
    }
    tex = {
        name: {
            "available": True,
            "path": f"/usr/bin/{name}",
            "sha256": "e" * 64,
            "version": f"fixture {name}",
        }
        for name in ("pdflatex", "bibtex")
    }
    commits = {
        repo.resolve(): main_commit,
        impute.resolve(): impute_commit,
        recover.resolve(): recover_commit,
        arc.resolve(): arc_commit,
    }
    return config, {"system": system, "nvidia": nvidia, "tex": tex}, torch, commits


def test_preflight_is_versioned_clean_and_fails_without_disclosure(tmp_path, monkeypatch):
    config, probes, torch, commits = _fixture(tmp_path)
    dirty_name = "credential-shaped-private-file.txt"

    def fake_run(argv, *, cwd=None, timeout=20):
        assert argv[0] == "git" and cwd is not None and timeout == 20
        resolved = Path(cwd).resolve()
        if argv[1:3] == ["rev-parse", "HEAD"]:
            return subprocess.CompletedProcess(argv, 0, commits[resolved] + "\n", "")
        if argv[1:3] == ["rev-parse", "HEAD^{tree}"]:
            name = "imputeecg" if resolved.name.startswith("ImputeECG-") else "ecgrecover"
            return subprocess.CompletedProcess(
                argv, 0, config.expected_tool_root_trees[name] + "\n", ""
            )
        if argv[1:4] == ["remote", "get-url", "origin"]:
            name = "imputeecg" if resolved.name.startswith("ImputeECG-") else "ecgrecover"
            return subprocess.CompletedProcess(
                argv, 0, config.expected_tool_origins[name] + "\n", ""
            )
        if argv[1:3] == ["status", "--porcelain"]:
            dirty = resolved == config.repo.resolve() and (config.repo / dirty_name).exists()
            stdout = f"?? {dirty_name}\n" if dirty else ""
            return subprocess.CompletedProcess(argv, 0, stdout, "")
        if argv[1:4] == ["config", "--get", "remote.origin.promisor"]:
            assert resolved != config.repo.resolve()
            return subprocess.CompletedProcess(argv, 0, "true\n", "")
        if argv[1:4] == ["config", "--get", "remote.origin.partialclonefilter"]:
            assert resolved != config.repo.resolve()
            return subprocess.CompletedProcess(argv, 0, "blob:none\n", "")
        raise AssertionError(argv)

    monkeypatch.setattr(preflight, "_run", fake_run)
    report = collect_server_preflight(
        config,
        system_probe=lambda: probes["system"],
        nvidia_probe=lambda: probes["nvidia"],
        torch_probe=lambda: torch,
        tex_probe=lambda: probes["tex"],
        installed_version_lookup=lambda name: "1.0" if name == "fixture" else "missing",
    )

    assert report["schema_version"] == SCHEMA_VERSION
    assert report["ok"] is True
    assert report["errors"] == []
    assert report["git"]["commit"] == config.expected_commit
    assert report["git"]["promisor"] == "not-required"
    assert report["git"]["partial_clone_filter"] == "not-required"
    assert report["datasets"]["ptbxl"]["metadata_rows"] == 10
    assert report["datasets"]["chapman"]["signals"] == 2
    assert report["environment_locks"]["gpu"]["sha256"] == (
        config.expected_lock_sha256["gpu"]
    )
    assert report["active_environment"]["ok"] is True
    assert report["paper_toolchain"]["pdflatex"]["available"] is True
    assert "autoresearchclaw" not in report["external_tools"]
    assert report["security"] == {
        "credentials_read": False,
        "environment_enumerated": False,
        "ssh_configuration_read": False,
        "raw_subprocess_stderr_included": False,
        "dirty_paths_included": False,
    }

    def upstream_without_partial_clone(argv, *, cwd=None, timeout=20):
        resolved = Path(cwd).resolve() if cwd is not None else None
        if (
            resolved != config.repo.resolve()
            and argv[1:4]
            in (
                ["config", "--get", "remote.origin.promisor"],
                ["config", "--get", "remote.origin.partialclonefilter"],
            )
        ):
            return subprocess.CompletedProcess(argv, 1, "", "")
        return fake_run(argv, cwd=cwd, timeout=timeout)

    monkeypatch.setattr(preflight, "_run", upstream_without_partial_clone)
    incomplete_upstream = collect_server_preflight(
        config,
        system_probe=lambda: probes["system"],
        nvidia_probe=lambda: probes["nvidia"],
        torch_probe=lambda: torch,
        tex_probe=lambda: probes["tex"],
        installed_version_lookup=lambda name: "1.0" if name == "fixture" else "missing",
    )
    assert incomplete_upstream["ok"] is False
    assert {issue["code"] for issue in incomplete_upstream["errors"]} >= {
        "tool.imputeecg.git",
        "tool.ecgrecover.git",
    }
    monkeypatch.setattr(preflight, "_run", fake_run)

    missing_runtime = (
        config.tools_root
        / "upstreams"
        / f"ImputeECG-{config.expected_tool_commits['imputeecg'][:12]}"
        / "required.py"
    )
    missing_runtime.unlink()
    missing_runtime_report = collect_server_preflight(
        config,
        system_probe=lambda: probes["system"],
        nvidia_probe=lambda: probes["nvidia"],
        torch_probe=lambda: torch,
        tex_probe=lambda: probes["tex"],
        installed_version_lookup=lambda name: "1.0" if name == "fixture" else "missing",
    )
    assert missing_runtime_report["ok"] is False
    assert "tool.imputeecg.required_paths" in {
        issue["code"] for issue in missing_runtime_report["errors"]
    }
    missing_runtime.write_text("# fixture\n", encoding="utf-8")

    secret_value = "never-include-this-environment-value"
    monkeypatch.setenv("PREFLIGHT_TEST_SECRET", secret_value)
    (config.repo / dirty_name).write_text("not a credential\n", encoding="utf-8")
    config.cpsc2018_root.rename(config.cpsc2018_root.with_name("cpsc-moved"))

    report = collect_server_preflight(
        config,
        system_probe=lambda: probes["system"],
        nvidia_probe=lambda: probes["nvidia"],
        torch_probe=lambda: torch,
        tex_probe=lambda: probes["tex"],
        installed_version_lookup=lambda name: "1.0" if name == "fixture" else "missing",
    )
    rendered = json.dumps(report, sort_keys=True)
    codes = {issue["code"] for issue in report["errors"]}

    assert report["ok"] is False
    assert "git.repo.dirty" in codes
    assert "dataset.cpsc2018.missing" in codes
    assert report["git"]["dirty_entry_count"] > 0
    assert dirty_name not in rendered
    assert secret_value not in rendered


def test_report_write_is_atomic_json(tmp_path):
    destination = tmp_path / "reports" / "server-preflight.v1.json"
    rendered = write_report({"schema_version": SCHEMA_VERSION, "ok": False}, destination)

    assert json.loads(rendered)["schema_version"] == SCHEMA_VERSION
    assert json.loads(destination.read_text(encoding="utf-8"))["ok"] is False
    assert not destination.with_name(destination.name + ".tmp").exists()

    fallback = failed_preflight_report("preflight.fixture", "allowlisted failure")
    assert fallback["schema_version"] == SCHEMA_VERSION
    assert fallback["ok"] is False
    assert fallback["errors"] == [
        {"code": "preflight.fixture", "message": "allowlisted failure"}
    ]


def test_repository_links_bind_inventory_roots_to_dag_paths(tmp_path):
    config, _probes, _torch, _commits = _fixture(tmp_path)
    persistent = tmp_path / "persistent-data"
    persistent.mkdir()
    roots = {}
    for name, current in (
        ("ptbxl", config.ptbxl_root),
        ("chapman", config.chapman_root),
        ("cpsc2018", config.cpsc2018_root),
    ):
        target = persistent / name
        current.rename(target)
        roots[name] = target.resolve()
    try:
        os.symlink(roots["ptbxl"], config.repo / "data" / "ptbxl", target_is_directory=True)
        os.symlink(
            roots["chapman"],
            config.repo / "data" / "external" / "chapman",
            target_is_directory=True,
        )
        os.symlink(
            roots["cpsc2018"],
            config.repo / "data" / "external" / "cpsc2018",
            target_is_directory=True,
        )
        os.symlink(
            config.tools_root / "upstreams",
            config.repo / "upstreams",
            target_is_directory=True,
        )
    except OSError as exc:
        pytest.skip(f"directory symlinks are unavailable: {exc}")

    strict = replace(
        config,
        ptbxl_root=roots["ptbxl"],
        chapman_root=roots["chapman"],
        cpsc2018_root=roots["cpsc2018"],
        require_staged_links=True,
    )
    issues = preflight._Issues()
    report = preflight.inspect_repository_links(strict, issues)
    assert report["required"] is True
    assert not issues.errors
    assert all(
        value["matches_expected_target"] and value["absolute_target"]
        for value in report["links"].values()
    )

    mismatch = replace(strict, chapman_root=roots["cpsc2018"])
    mismatch_issues = preflight._Issues()
    preflight.inspect_repository_links(mismatch, mismatch_issues)
    assert "link.chapman.mismatch" in {
        issue["code"] for issue in mismatch_issues.errors
    }


def test_dataset_inventory_rejects_empty_and_ambiguous_files(tmp_path):
    config, _probes, _torch, _commits = _fixture(tmp_path)
    (config.ptbxl_root / "records500" / "00000_hr.dat").write_bytes(b"")
    ptbxl_issues = preflight._Issues()
    preflight.inspect_ptbxl(
        config.ptbxl_root,
        expected_records=10,
        issues=ptbxl_issues,
    )
    assert "dataset.ptbxl.records500_empty" in {
        issue["code"] for issue in ptbxl_issues.errors
    }

    first = config.chapman_root / "record-0000"
    first.with_suffix(".dat").write_bytes(b"second representation")
    first.with_suffix(".hea").write_bytes(b"")
    external_issues = preflight._Issues()
    preflight.inspect_external_wfdb(
        "chapman",
        config.chapman_root,
        expected_records=2,
        issues=external_issues,
    )
    assert {
        "dataset.chapman.ambiguous",
        "dataset.chapman.empty_header",
    } <= {issue["code"] for issue in external_issues.errors}


def test_preflight_report_must_be_outside_frozen_repository(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    outside = tmp_path / "reports" / "preflight.json"
    assert preflight_cli._output_path_error(repo, outside) is None
    assert preflight_cli._output_path_error(repo, None) is None
    assert "outside" in preflight_cli._output_path_error(
        repo,
        repo / "ignored" / "preflight.json",
    )
