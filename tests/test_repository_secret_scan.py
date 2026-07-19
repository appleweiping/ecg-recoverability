from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import pytest

from ecgcert import lineage
from ecgcert.security_scan import scan_repository, validate_secret_scan_report


NOW = datetime(2026, 7, 19, 12, 0, tzinfo=timezone.utc)


def _git(repo: Path, *arguments: str) -> None:
    subprocess.run(["git", *arguments], cwd=repo, check=True, capture_output=True)


def _repository(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.test")
    _git(repo, "config", "user.name", "ECG Test")
    (repo / ".gitignore").write_text("scratchpad_*\n", encoding="utf-8")
    (repo / "study.py").write_text("print('recoverability')\n", encoding="utf-8")
    _git(repo, "add", ".gitignore", "study.py")
    _git(repo, "commit", "-m", "fixture")
    return repo


def test_scan_binds_clean_commit_and_includes_ignored_files(tmp_path: Path) -> None:
    repo = _repository(tmp_path)
    (repo / "scratchpad_notes.txt").write_text("ordinary notes\n", encoding="utf-8")
    output = tmp_path / "scan.json"
    report = scan_repository(repo, output, scanned_at=NOW)
    assert report["status"] == "complete"
    assert report["scope"]["ignored"] is True
    assert report["findings_count"] == 0
    validated = validate_secret_scan_report(
        output,
        expected_commit=report["repository"]["commit"],
        expected_tree=report["repository"]["tree"],
        expected_scanner_sha256=report["scanner"]["sha256"],
    )
    assert validated["all_controls_satisfied"] is True


def test_scan_reports_rule_and_path_but_never_secret_value(tmp_path: Path) -> None:
    repo = _repository(tmp_path)
    fixture_value = "sensitive-fixture-value"
    variable_name = "PASS" + "WORD"
    (repo / "scratchpad_server.env").write_text(
        f"{variable_name}={fixture_value}\n", encoding="utf-8"
    )
    output = tmp_path / "scan.json"
    report = scan_repository(repo, output, scanned_at=NOW)
    serialized = output.read_text(encoding="utf-8")
    assert report["status"] == "failed"
    assert report["findings"] == [
        {"path": "scratchpad_server.env", "rule": "credential_assignment"}
    ]
    assert fixture_value not in serialized
    assert validate_secret_scan_report(output)["all_controls_satisfied"] is False


def test_scan_does_not_treat_dynamic_code_expressions_as_literal_credentials(
    tmp_path: Path,
) -> None:
    repo = _repository(tmp_path)
    (repo / "dynamic.py").write_text(
        "token = token.replace(marker, value)\n"
        "password: bytes | None\n"
        "password=password)\n",
        encoding="utf-8",
    )
    _git(repo, "add", "dynamic.py")
    _git(repo, "commit", "-m", "dynamic expressions")
    output = tmp_path / "scan.json"
    report = scan_repository(repo, output, scanned_at=NOW)
    assert report["status"] == "complete"
    assert report["findings_count"] == 0


def test_scan_refuses_dirty_checkout_and_output_inside_repository(tmp_path: Path) -> None:
    repo = _repository(tmp_path)
    with pytest.raises(ValueError, match="outside the repository"):
        scan_repository(repo, repo / "scan.json", scanned_at=NOW)
    (repo / "untracked.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="clean checkout"):
        scan_repository(repo, tmp_path / "scan.json", scanned_at=NOW)


def test_validation_detects_tampered_finding_count(tmp_path: Path) -> None:
    repo = _repository(tmp_path)
    output = tmp_path / "scan.json"
    scan_repository(repo, output, scanned_at=NOW)
    report = json.loads(output.read_text(encoding="utf-8"))
    report["findings_count"] = 1
    output.write_text(json.dumps(report), encoding="utf-8")
    validated = validate_secret_scan_report(output)
    assert validated["all_controls_satisfied"] is False
    assert validated["report_sha256"] == lineage.artifact_sha256(output)
