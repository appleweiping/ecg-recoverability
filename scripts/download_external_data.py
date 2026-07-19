"""Resume complete PTB-XL, Chapman and CPSC2018 downloads from PhysioNet.

The script intentionally downloads complete cohorts; main-paper runs reject the
old 350-record Chapman cache.  Run on the persistent server volume, not the 30 GB
root filesystem.  PhysioNet's public S3 bucket is preferred because ``aws s3
sync`` is resumable and parallel; recursive HTTPS remains a disclosed fallback.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Sequence


SOURCES = {
    "ptbxl": {
        "url": "https://physionet.org/files/ptb-xl/1.0.3/",
        "s3": "s3://physionet-open/ptb-xl/1.0.3/",
        "cut_dirs": 2,
        "version": "1.0.3",
    },
    "chapman": {
        "url": "https://physionet.org/files/ecg-arrhythmia/1.0.0/",
        "s3": "s3://physionet-open/ecg-arrhythmia/1.0.0/",
        "cut_dirs": 3,
        "version": "1.0.0",
    },
    "cpsc2018": {
        "url": "https://physionet.org/files/challenge-2020/1.0.2/training/cpsc_2018/",
        "s3": "s3://physionet-open/challenge-2020/1.0.2/training/cpsc_2018/",
        "cut_dirs": 5,
        "version": "challenge-2020/1.0.2",
    },
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _script_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def download(
    dataset: str,
    destination: str | Path,
    *,
    transport: str = "auto",
) -> None:
    if dataset not in SOURCES:
        raise ValueError(dataset)
    if transport not in {"auto", "aws", "wget"}:
        raise ValueError(f"unsupported download transport: {transport}")
    target = Path(destination).resolve() / dataset
    target.mkdir(parents=True, exist_ok=True)
    source = SOURCES[dataset]
    aws = shutil.which("aws")
    if transport in {"auto", "aws"} and aws is not None:
        command = [
            aws,
            "s3",
            "sync",
            "--no-sign-request",
            "--only-show-errors",
            str(source["s3"]),
            str(target),
        ]
        subprocess.run(command, check=True)
        return
    if transport == "aws":
        raise RuntimeError("AWS transport requested but the aws CLI is unavailable")
    wget = shutil.which("wget")
    if wget is None:
        raise RuntimeError(
            "complete external downloads require the aws CLI (preferred) or wget"
        )
    command = [
        wget,
        "--recursive",
        "--no-parent",
        "--no-host-directories",
        f"--cut-dirs={source['cut_dirs']}",
        "--continue",
        "--timestamping",
        "--reject",
        "index.html*",
        "--directory-prefix",
        str(target),
        str(source["url"]),
    ]
    subprocess.run(command, check=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=[*SOURCES, "all"], default="all")
    parser.add_argument(
        "--destination",
        required=True,
        help="persistent data root; dataset subdirectories are created below it",
    )
    parser.add_argument(
        "--transport",
        choices=("auto", "aws", "wget"),
        default="auto",
        help="prefer official public S3; use wget only as a resumable HTTPS fallback",
    )
    parser.add_argument(
        "--status",
        type=Path,
        help="optional atomic structured status file for a long-running server download",
    )
    args = parser.parse_args(argv)
    selected = SOURCES if args.dataset == "all" else (args.dataset,)
    selected = tuple(selected)
    state = {
        "schema_version": "ecg-data-download-status-v1",
        "status": "running",
        "started_at": _utc_now(),
        "finished_at": None,
        "current_dataset": None,
        "completed_datasets": [],
        "selected_datasets": list(selected),
        "destination": str(Path(args.destination).resolve()),
        "transport": args.transport,
        "sources": {
            dataset: {
                "url": SOURCES[dataset]["url"],
                "s3": SOURCES[dataset]["s3"],
                "version": SOURCES[dataset]["version"],
            }
            for dataset in selected
        },
        "command": [sys.executable, str(Path(__file__).resolve()), *(argv or sys.argv[1:])],
        "downloader_sha256": _script_sha256(),
    }
    if args.status is not None:
        _atomic_json(args.status.resolve(), state)
    try:
        for dataset in selected:
            state["current_dataset"] = dataset
            if args.status is not None:
                _atomic_json(args.status.resolve(), state)
            download(dataset, args.destination, transport=args.transport)
            state["completed_datasets"].append(dataset)
        state.update(
            {
                "status": "complete",
                "current_dataset": None,
                "finished_at": _utc_now(),
            }
        )
    except Exception as exc:
        state.update(
            {
                "status": "failed",
                "finished_at": _utc_now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        if args.status is not None:
            _atomic_json(args.status.resolve(), state)
        raise
    if args.status is not None:
        _atomic_json(args.status.resolve(), state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
