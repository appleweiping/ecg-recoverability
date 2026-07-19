"""Freeze the sole folds-1--7 reconstruction-training inclusion decision."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from uuid import uuid4

from ecgcert import lineage
from ecgcert.data import PTBXL
from ecgcert.protocol import (
    PRIMARY_RATE_HZ,
    PRIMARY_SEGMENTS,
    configuration_panel_sha256,
    deep_configuration_panel,
)
from ecgcert.training_inclusion import build_training_inclusion

try:
    from experiments.reconstruction_benchmark_v3 import (
        _validate_database_identity,
        _verify_manifest_files,
        load_ptbxl_manifest,
    )
except ModuleNotFoundError:  # pragma: no cover - direct invocation from scripts/
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from experiments.reconstruction_benchmark_v3 import (  # type: ignore[no-redef]
        _validate_database_identity,
        _verify_manifest_files,
        load_ptbxl_manifest,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--release", action="store_true")
    return parser


def validate_arguments(arguments: argparse.Namespace) -> None:
    if not arguments.release:
        return
    try:
        arguments.output_dir.resolve().relative_to((Path.cwd() / "artifacts").resolve())
    except ValueError as exc:
        raise ValueError("release output must stay below artifacts/") from exc


def run(arguments: argparse.Namespace) -> dict:
    validate_arguments(arguments)
    contract = load_ptbxl_manifest(arguments.manifest, release=arguments.release)
    record_ids = contract.record_ids("train")
    _verify_manifest_files(contract, record_ids, rate=PRIMARY_RATE_HZ)
    db = PTBXL(contract.root)
    _validate_database_identity(db, contract, record_ids)
    output = arguments.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite training inclusion artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.tmp-{uuid4().hex}"
    if staging.exists():
        raise FileExistsError(staging)
    try:
        result = build_training_inclusion(
            db=db,
            records=contract.records,
            record_ids=record_ids,
            source_manifest_file_sha256=lineage.artifact_sha256(
                arguments.manifest.resolve()
            ),
            source_manifest_sha256=contract.manifest_sha256,
            split_sha256=contract.split_sha256,
            rate_hz=PRIMARY_RATE_HZ,
            segments=PRIMARY_SEGMENTS,
            delineator="dwt",
            configurations=deep_configuration_panel(),
            configuration_panel_sha256=configuration_panel_sha256(),
            output_dir=staging,
        )
        os.replace(staging, output)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return result


def main() -> int:
    arguments = build_parser().parse_args()
    try:
        result = run(arguments)
    except Exception as exc:
        raise SystemExit(f"training inclusion preparation failed closed: {exc}") from exc
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
