import json
import subprocess

import pytest

from scripts import download_external_data


def test_official_s3_is_preferred_and_unsigned(tmp_path, monkeypatch):
    commands = []

    def which(name):
        return "/tools/aws" if name == "aws" else None

    def run(command, *, check):
        commands.append(command)
        assert check is True
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(download_external_data.shutil, "which", which)
    monkeypatch.setattr(download_external_data.subprocess, "run", run)

    download_external_data.download("chapman", tmp_path)

    assert commands == [
        [
            "/tools/aws",
            "s3",
            "sync",
            "--no-sign-request",
            "--only-show-errors",
            "s3://physionet-open/ecg-arrhythmia/1.0.0/",
            str((tmp_path / "chapman").resolve()),
        ]
    ]


def test_ptbxl_uses_the_frozen_official_release(tmp_path, monkeypatch):
    commands = []
    monkeypatch.setattr(
        download_external_data.shutil,
        "which",
        lambda name: "/tools/aws" if name == "aws" else None,
    )
    monkeypatch.setattr(
        download_external_data.subprocess,
        "run",
        lambda command, *, check: commands.append((command, check)),
    )

    download_external_data.download("ptbxl", tmp_path, transport="aws")

    assert commands[0][1] is True
    assert "s3://physionet-open/ptb-xl/1.0.3/" in commands[0][0]
    assert commands[0][0][-1] == str((tmp_path / "ptbxl").resolve())


def test_https_fallback_is_resumable(tmp_path, monkeypatch):
    commands = []

    def which(name):
        return "/tools/wget" if name == "wget" else None

    monkeypatch.setattr(download_external_data.shutil, "which", which)
    monkeypatch.setattr(
        download_external_data.subprocess,
        "run",
        lambda command, *, check: commands.append((command, check)),
    )

    download_external_data.download("cpsc2018", tmp_path, transport="wget")

    command, check = commands[0]
    assert check is True
    assert command[0] == "/tools/wget"
    assert "--continue" in command
    assert command[-1].endswith("/training/cpsc_2018/")


def test_forced_aws_fails_closed_when_cli_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(download_external_data.shutil, "which", lambda _name: None)
    with pytest.raises(RuntimeError, match="aws CLI is unavailable"):
        download_external_data.download("chapman", tmp_path, transport="aws")


def test_main_writes_atomic_complete_status(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        download_external_data,
        "download",
        lambda dataset, destination, *, transport: calls.append(
            (dataset, destination, transport)
        ),
    )
    status = tmp_path / "control" / "download.v1.json"
    assert download_external_data.main(
        [
            "--dataset",
            "all",
            "--destination",
            str(tmp_path / "data"),
            "--transport",
            "wget",
            "--status",
            str(status),
        ]
    ) == 0
    payload = json.loads(status.read_text())
    assert payload["status"] == "complete"
    assert payload["completed_datasets"] == ["ptbxl", "chapman", "cpsc2018"]
    assert payload["current_dataset"] is None
    assert len(payload["downloader_sha256"]) == 64
    assert [item[0] for item in calls] == ["ptbxl", "chapman", "cpsc2018"]


def test_main_persists_failure_and_completed_prefix(tmp_path, monkeypatch):
    def fail_on_chapman(dataset, _destination, *, transport):
        assert transport == "wget"
        if dataset == "chapman":
            raise RuntimeError("fixture failure")

    monkeypatch.setattr(download_external_data, "download", fail_on_chapman)
    status = tmp_path / "download.v1.json"
    with pytest.raises(RuntimeError, match="fixture failure"):
        download_external_data.main(
            [
                "--dataset",
                "all",
                "--destination",
                str(tmp_path / "data"),
                "--transport",
                "wget",
                "--status",
                str(status),
            ]
        )
    payload = json.loads(status.read_text())
    assert payload["status"] == "failed"
    assert payload["completed_datasets"] == ["ptbxl"]
    assert payload["current_dataset"] == "chapman"
    assert payload["error_type"] == "RuntimeError"
