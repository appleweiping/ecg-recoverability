"""Resume complete Chapman and CPSC2018 downloads from official PhysioNet paths.

The script intentionally downloads complete cohorts; main-paper runs reject the
old 350-record Chapman cache.  Run on the persistent server volume, not the 30 GB
root filesystem.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess


SOURCES = {
    "chapman": {
        "url": "https://physionet.org/files/ecg-arrhythmia/1.0.0/",
        "cut_dirs": 3,
        "version": "1.0.0",
    },
    "cpsc2018": {
        "url": "https://physionet.org/files/challenge-2020/1.0.2/training/cpsc_2018/",
        "cut_dirs": 5,
        "version": "challenge-2020/1.0.2",
    },
}


def download(dataset: str, destination: str | Path) -> None:
    if dataset not in SOURCES:
        raise ValueError(dataset)
    if shutil.which("wget") is None:
        raise RuntimeError("complete resumable external downloads require wget")
    target = Path(destination).resolve() / dataset
    target.mkdir(parents=True, exist_ok=True)
    source = SOURCES[dataset]
    command = [
        "wget",
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=[*SOURCES, "all"], default="all")
    parser.add_argument(
        "--destination",
        required=True,
        help="persistent data root; dataset subdirectories are created below it",
    )
    args = parser.parse_args()
    selected = SOURCES if args.dataset == "all" else (args.dataset,)
    for dataset in selected:
        download(dataset, args.destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
