from __future__ import annotations

import json
from pathlib import Path

import pytest

from ecgcert.arc_control import ArcControlValidationError, ORDERED_STAGES
from scripts import release


def _write_reports(run_dir: Path) -> None:
    for stage in ORDERED_STAGES:
        path = (
            run_dir
            / "workspace"
            / "artifacts"
            / "control"
            / "arc"
            / f"stage{stage}"
            / "report.v1.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"stage": stage}) + "\n", encoding="utf-8")


def test_release_independently_replays_all_four_arc_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    _write_reports(run_dir)
    observed: list[dict] = []

    def validate(reports):
        observed.extend(reports)
        return reports

    monkeypatch.setattr(release, "validate_arc_control_chain", validate)

    release._validate_arc_gate_chain(run_dir)

    assert [item["stage"] for item in observed] == list(ORDERED_STAGES)


def test_release_fails_closed_when_arc_chain_validator_rejects_splice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    _write_reports(run_dir)

    def reject(_reports):
        raise ArcControlValidationError("run_id/session_id changed between formal gates")

    monkeypatch.setattr(release, "validate_arc_control_chain", reject)

    with pytest.raises(SystemExit, match="four-gate chain failed"):
        release._validate_arc_gate_chain(run_dir)


def test_release_requires_every_formal_arc_gate_report(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_reports(run_dir)
    missing = (
        run_dir
        / "workspace"
        / "artifacts"
        / "control"
        / "arc"
        / "stage15"
        / "report.v1.json"
    )
    missing.unlink()

    with pytest.raises(SystemExit, match="Stage 15 report is missing"):
        release._validate_arc_gate_chain(run_dir)
