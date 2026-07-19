from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from scripts import run_cuda_contract_tests as cuda_contract


def _arguments(tmp_path: Path) -> argparse.Namespace:
    repo = tmp_path / "repo"
    upstreams = tmp_path / "upstreams"
    repo.mkdir()
    upstreams.mkdir()
    return argparse.Namespace(
        repo=repo,
        upstreams=upstreams,
        output_dir=tmp_path / "report",
        timeout_seconds=30,
    )


def test_cuda_contract_runner_binds_sources_hardware_and_logs(tmp_path, monkeypatch):
    arguments = _arguments(tmp_path)
    impute = arguments.upstreams / "impute"
    recover = arguments.upstreams / "recover"
    impute.mkdir()
    recover.mkdir()
    monkeypatch.setattr(
        cuda_contract,
        "_resolve_checkout",
        lambda _root, spec: impute if spec.name == "ImputeECG" else recover,
    )
    monkeypatch.setattr(cuda_contract, "validate_pinned_checkout", lambda *_args: "ok")
    monkeypatch.setattr(
        cuda_contract.lineage,
        "hardware_fingerprint",
        lambda: {"gpu": [{"name": "fixture"}], "cuda_runtime": "12.8"},
    )

    def fake_run(command, **kwargs):
        assert kwargs["env"]["ECGCERT_REQUIRE_CUDA_TEST"] == "1"
        assert kwargs["env"]["ECGCERT_IMPUTEECG_SOURCE_DIR"] == str(impute)
        assert kwargs["env"]["ECGCERT_ECGRECOVER_SOURCE_DIR"] == str(recover)
        assert kwargs["env"]["OMP_NUM_THREADS"] == "1"
        assert kwargs["env"]["ECGCERT_NUM_WORKERS"] == "1"
        assert "--basetemp" in command
        kwargs["stdout"].write(b"3 passed\n")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cuda_contract.subprocess, "run", fake_run)
    report = cuda_contract.run(arguments)
    persisted = json.loads((arguments.output_dir / "report.v1.json").read_text())
    assert report == persisted
    assert report["status"] == "complete"
    assert report["tests"] == list(cuda_contract.TESTS)
    assert report["hardware"]["cuda_runtime"] == "12.8"
    assert report["artifacts"]["stdout"]["sha256"]


def test_cuda_contract_runner_writes_failed_report_before_failing(tmp_path, monkeypatch):
    arguments = _arguments(tmp_path)
    checkout = arguments.upstreams / "checkout"
    checkout.mkdir()
    monkeypatch.setattr(cuda_contract, "_resolve_checkout", lambda *_args: checkout)
    monkeypatch.setattr(cuda_contract, "validate_pinned_checkout", lambda *_args: "ok")
    monkeypatch.setattr(cuda_contract.lineage, "hardware_fingerprint", lambda: {})
    monkeypatch.setattr(
        cuda_contract.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=7),
    )
    with pytest.raises(RuntimeError, match="exit code 7"):
        cuda_contract.run(arguments)
    report = json.loads((arguments.output_dir / "report.v1.json").read_text())
    assert report["status"] == "failed"
    assert report["returncode"] == 7


def test_cuda_contract_timeout_writes_structured_failure(tmp_path, monkeypatch):
    arguments = _arguments(tmp_path)
    checkout = arguments.upstreams / "checkout"
    checkout.mkdir()
    monkeypatch.setattr(cuda_contract, "_resolve_checkout", lambda *_args: checkout)
    monkeypatch.setattr(cuda_contract, "validate_pinned_checkout", lambda *_args: "ok")
    monkeypatch.setattr(cuda_contract.lineage, "hardware_fingerprint", lambda: {})

    def timeout(command, **_kwargs):
        raise subprocess.TimeoutExpired(command, arguments.timeout_seconds)

    monkeypatch.setattr(cuda_contract.subprocess, "run", timeout)
    with pytest.raises(RuntimeError, match="exit code 124"):
        cuda_contract.run(arguments)
    report = json.loads((arguments.output_dir / "report.v1.json").read_text())
    assert report["status"] == "failed"
    assert report["failure_kind"] == "timeout"
    assert report["returncode"] == 124
