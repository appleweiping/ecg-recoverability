"""Prepare pinned ImputeECG and ECGrecover inputs/configs without training them.

The command verifies the complete 500 Hz PTB-XL manifest and both pristine
upstream checkouts before writing anything.  It creates no checkpoint.  Because
ECGrecover does not expose a stable package API, ``--ecgrecover-integration`` is
an obligatory, hashed argv descriptor for the thin, separately auditable bridge.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence
from uuid import uuid4

from ecgcert.data import PTBXL
from ecgcert.estimators.official import (
    ECG_RECOVER,
    IMPUTE_ECG,
    validate_pinned_checkout,
)
from ecgcert.official_baselines import (
    CONFIG_SCHEMA_VERSION,
    IMPUTE_ECG_PARAMETERS,
    OfficialTrainingRecord,
    RELEASE_SEEDS,
    SCHEMA_VERSION,
    canonical_sha256,
    load_ecgrecover_integration,
    materialize_official_arrays,
    pinned_source_dir,
    replace_markers,
    sha256_file,
    source_tree_sha256,
)
from ecgcert.protocol import PRIMARY_RATE_HZ, configuration_panel_sha256
from ecgcert.protocol import PRIMARY_SEGMENTS
from ecgcert.training_inclusion import TrainingInclusion, load_training_inclusion

try:
    from experiments.reconstruction_benchmark_v3 import (
        PTBXLManifestV3,
        _validate_database_identity,
        _verify_manifest_files,
        load_ptbxl_manifest,
    )
except ModuleNotFoundError:  # pragma: no cover - direct invocation from scripts/
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from experiments.reconstruction_benchmark_v3 import (  # type: ignore[no-redef]
        PTBXLManifestV3,
        _validate_database_identity,
        _verify_manifest_files,
        load_ptbxl_manifest,
    )


def _parse_seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from exc
    if not seeds or any(seed < 0 for seed in seeds) or len(seeds) != len(set(seeds)):
        raise argparse.ArgumentTypeError("seeds must be unique non-negative integers")
    return seeds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--upstreams", type=Path, required=True)
    parser.add_argument("--ecgrecover-integration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--training-inclusion", type=Path)
    parser.add_argument("--seeds", type=_parse_seeds, default=RELEASE_SEEDS)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--release", action="store_true")
    return parser


def validate_arguments(arguments: argparse.Namespace) -> None:
    if arguments.max_records is not None and arguments.max_records < 1:
        raise ValueError("--max-records must be positive")
    if arguments.release:
        violations = []
        if arguments.max_records is not None:
            violations.append("--max-records is forbidden")
        if tuple(arguments.seeds) != RELEASE_SEEDS:
            violations.append("release seeds must be exactly 0,1,2,3,4")
        if getattr(arguments, "training_inclusion", None) is None:
            violations.append("--training-inclusion is required")
        try:
            arguments.output_dir.resolve().relative_to((Path.cwd() / "artifacts").resolve())
        except ValueError:
            violations.append("--output-dir must be under artifacts/ for release")
        if violations:
            raise ValueError("official baseline release violation: " + "; ".join(violations))


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _method_configs(
    *,
    final_output: Path,
    contract: PTBXLManifestV3,
    arrays: Mapping[str, Any],
    source_dirs: Mapping[str, Path],
    source_trees: Mapping[str, str],
    integration: Any,
    training_inclusion: TrainingInclusion,
    seeds: Sequence[int],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build per-run configs plus the combined no-outcome-tuning config."""

    final_data = final_output / "data"
    impute_data = final_data / "imputeecg"
    ecgrecover_data = final_data / "ecgrecover"
    bridge_replacements = {
        "python": sys.executable,
        "source_dir": str(source_dirs["ecgrecover"]),
        "data_dir": str(ecgrecover_data),
        "bridge_root": str(integration.bridge_root),
    }
    train_command = replace_markers(integration.train_command, bridge_replacements)
    inference_command = replace_markers(integration.inference_command, bridge_replacements)
    checkpoint = replace_markers([integration.checkpoint], bridge_replacements)[0]
    unresolved = ("python", "source_dir", "data_dir", "bridge_root")
    joined_commands = "\n".join(inference_command + train_command)
    for marker in unresolved:
        if "{" + marker + "}" in joined_commands:
            raise ValueError(f"unresolved {{{marker}}} in ECGrecover integration")

    imputeecg = {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "method": "imputeecg",
        "selection": "official_fixed",
        "source_dir": str(source_dirs["imputeecg"]),
        "repository": IMPUTE_ECG.repository,
        "commit": IMPUTE_ECG.commit,
        "source_tree_sha256": source_trees["imputeecg"],
        "manifest_sha256": contract.manifest_sha256,
        "split_sha256": contract.split_sha256,
        "training_inclusion_sha256": training_inclusion.inclusion_sha256,
        "training_record_ids_sha256": training_inclusion.record_ids_sha256,
        "training_patient_ids_sha256": training_inclusion.patient_ids_sha256,
        "train_role": "folds1-7/train",
        "official_data_path": str(impute_data),
        "parameters": dict(IMPUTE_ECG_PARAMETERS),
        "array_sha256": {
            "train_data_gt.npy": arrays["imputeecg_ground_truth"]["sha256"],
            "train_data_mask.npy": arrays["imputeecg_masked_observation"]["sha256"],
            "training_records.v3.json": arrays["record_order"]["sha256"],
        },
        "training_records_path": str(final_data / "training_records.v3.json"),
        "seeds": list(seeds),
    }
    ecgrecover = {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "method": "ecgrecover",
        "selection": "official_fixed",
        "source_dir": str(source_dirs["ecgrecover"]),
        "repository": ECG_RECOVER.repository,
        "commit": ECG_RECOVER.commit,
        "source_tree_sha256": source_trees["ecgrecover"],
        "manifest_sha256": contract.manifest_sha256,
        "split_sha256": contract.split_sha256,
        "training_inclusion_sha256": training_inclusion.inclusion_sha256,
        "training_record_ids_sha256": training_inclusion.record_ids_sha256,
        "training_patient_ids_sha256": training_inclusion.patient_ids_sha256,
        "train_role": "folds1-7/train",
        "input_lead": integration.input_lead,
        "native_rate_hz": integration.native_rate_hz,
        "model_samples": integration.model_samples,
        "inference_records_per_process": integration.inference_records_per_process,
        "inference_micro_batch_size": integration.inference_micro_batch_size,
        "benchmark_rate_hz": PRIMARY_RATE_HZ,
        "official_data_path": str(ecgrecover_data),
        "ground_truth_path": str(impute_data / "train_data_gt.npy"),
        "single_lead_input_path": str(ecgrecover_data / "train_input_single_lead.npy"),
        "official_train_command": train_command,
        "official_inference_bridge": inference_command,
        "checkpoint": checkpoint,
        "integration_descriptor_sha256": integration.descriptor_sha256,
        "upstream_source_files": list(integration.upstream_source_files),
        "bridge_root": str(integration.bridge_root),
        "bridge_files": list(integration.bridge_files),
        "license_spdx": integration.license_spdx,
        "redistribution": integration.redistribution,
        "permission_basis": integration.permission_basis,
        "adaptation_disclosure": list(integration.adaptation_disclosure),
        "array_sha256": {
            "ground_truth": arrays["imputeecg_ground_truth"]["sha256"],
            "single_lead_input": arrays["ecgrecover_single_lead_input"]["sha256"],
            "dataset.v3.json": arrays["ecgrecover_dataset"]["sha256"],
            "training_records.v3.json": arrays["record_order"]["sha256"],
        },
        "training_records_path": str(final_data / "training_records.v3.json"),
        "scope": "published single-input-lead task; never treated as arbitrary-mask parity",
        "seeds": list(seeds),
    }
    combined = {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "status": "complete",
        "manifest_sha256": contract.manifest_sha256,
        "split_sha256": contract.split_sha256,
        "training_inclusion_sha256": training_inclusion.inclusion_sha256,
        "training_record_ids_sha256": training_inclusion.record_ids_sha256,
        "training_patient_ids_sha256": training_inclusion.patient_ids_sha256,
        "configuration_panel_sha256": configuration_panel_sha256(),
        "methods": {
            "imputeecg": {
                "selection": "official_fixed",
                "training_parameters": dict(IMPUTE_ECG_PARAMETERS),
                "method_config": str(final_output / "imputeecg.config.v3.json"),
            },
            "ecgrecover": {
                "selection": "official_fixed",
                "training_parameters": {
                    "published_task": "single-input",
                    "input_lead": integration.input_lead,
                    "native_rate_hz": integration.native_rate_hz,
                    "model_samples": integration.model_samples,
                    "inference_records_per_process": (
                        integration.inference_records_per_process
                    ),
                    "inference_micro_batch_size": integration.inference_micro_batch_size,
                    "benchmark_rate_hz": PRIMARY_RATE_HZ,
                    "adaptation_disclosure": list(integration.adaptation_disclosure),
                },
                "method_config": str(final_output / "ecgrecover.config.v3.json"),
            },
        },
        "checkpoints_created": False,
    }
    return imputeecg, ecgrecover, combined


def _rebase_artifact_paths(
    arrays: Mapping[str, Any],
    *,
    old_root: Path,
    new_root: Path,
) -> dict[str, Any]:
    """Replace staging paths structurally (JSON string replacement is unsafe on Windows)."""

    copied = json.loads(json.dumps(arrays))
    for value in copied.values():
        if not isinstance(value, dict) or "path" not in value:
            continue
        path = Path(str(value["path"])).resolve()
        try:
            relative = path.relative_to(old_root.resolve())
        except ValueError as exc:
            raise ValueError(f"prepared artifact path escapes staging data root: {path}") from exc
        value["path"] = str((new_root / relative).resolve())
    return copied


def _remove_orphan_staging_directories(output: Path) -> None:
    """Clean only UUID staging directories owned by this exact output bundle."""

    parent = output.resolve().parent
    prefix = f".{output.name}.tmp-"
    for path in sorted(parent.glob(f"{prefix}*")):
        resolved = path.resolve()
        if resolved.parent != parent or not resolved.name.startswith(prefix):
            raise ValueError("unsafe official-baseline staging cleanup target")
        if not resolved.is_dir():
            raise ValueError("official-baseline staging residue is not a directory")
        shutil.rmtree(resolved)


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    validate_arguments(arguments)
    output = arguments.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite official baseline artifact: {output}")
    _remove_orphan_staging_directories(output)
    upstreams = arguments.upstreams.resolve()
    if not upstreams.is_dir():
        raise FileNotFoundError(upstreams)

    contract = load_ptbxl_manifest(arguments.manifest, release=arguments.release)
    all_ids = tuple(
        record_id
        for role in ("train", "tune", "calibration", "test")
        for record_id in contract.split[role]
    )
    _verify_manifest_files(contract, all_ids, rate=PRIMARY_RATE_HZ)
    db = PTBXL(contract.root)
    _validate_database_identity(db, contract, all_ids)
    requested_train_ids = contract.record_ids("train", arguments.max_records)
    if arguments.training_inclusion is None:
        raise ValueError("official baseline preparation requires --training-inclusion")
    training_inclusion = load_training_inclusion(
        arguments.training_inclusion,
        source_manifest_path=arguments.manifest,
        source_manifest_sha256=contract.manifest_sha256,
        split_sha256=contract.split_sha256,
        expected_record_ids=requested_train_ids,
        expected_records=contract.records,
        rate_hz=PRIMARY_RATE_HZ,
        segments=PRIMARY_SEGMENTS,
        delineator="dwt",
        configuration_panel_sha256=configuration_panel_sha256(),
    )

    source_dirs = {
        "imputeecg": pinned_source_dir(upstreams, IMPUTE_ECG),
        "ecgrecover": pinned_source_dir(upstreams, ECG_RECOVER),
    }
    source_trees = {}
    source_entries = {}
    for method, spec in (("imputeecg", IMPUTE_ECG), ("ecgrecover", ECG_RECOVER)):
        validate_pinned_checkout(source_dirs[method], spec)
        tree_sha, entries = source_tree_sha256(source_dirs[method])
        source_trees[method] = tree_sha
        source_entries[method] = entries
    required_impute_files = {"train.py", "inference.py", "datasets/ptbxl.py"}
    impute_names = {entry["path"] for entry in source_entries["imputeecg"]}
    if not required_impute_files <= impute_names:
        raise ValueError("pinned ImputeECG checkout lacks its documented train/inference files")

    integration = load_ecgrecover_integration(
        arguments.ecgrecover_integration,
        source_dir=source_dirs["ecgrecover"],
    )
    record_ids = training_inclusion.included_record_ids
    if arguments.release and tuple(requested_train_ids) != tuple(contract.split["train"]):
        raise ValueError("release preparation must contain the complete folds 1-7 train split")
    def training_records():
        for record_id, patient_id, signal, audit in training_inclusion.iter_validated_signals(
            db, contract.records
        ):
            audit_value = {
                key: value
                for key, value in dict(audit).items()
                if value is not None
            }
            yield OfficialTrainingRecord(
                record_id=record_id,
                patient_id=patient_id,
                signal=signal,
                source_audit=audit_value,
            )

    staging = output.parent / f".{output.name}.tmp-{uuid4().hex}"
    if staging.exists():
        raise FileExistsError(staging)
    try:
        staging.mkdir(parents=True)
        arrays = materialize_official_arrays(
            training_records(),
            output_dir=staging / "data",
            ecgrecover_input_lead=integration.input_lead,
            n_records=len(record_ids),
        )
        record_order = arrays.get("record_order", {})
        if (
            not isinstance(record_order, Mapping)
            or record_order.get("record_ids_sha256")
            != training_inclusion.record_ids_sha256
            or record_order.get("patient_ids_sha256")
            != training_inclusion.patient_ids_sha256
        ):
            raise ValueError(
                "official arrays changed the shared ordered training record/patient cohort"
            )
        imputeecg, ecgrecover, combined = _method_configs(
            final_output=output,
            contract=contract,
            arrays=arrays,
            source_dirs=source_dirs,
            source_trees=source_trees,
            integration=integration,
            training_inclusion=training_inclusion,
            seeds=arguments.seeds,
        )
        # Paths inside the array lineage are made final before it is persisted.
        serialized_arrays = _rebase_artifact_paths(
            arrays,
            old_root=staging / "data",
            new_root=output / "data",
        )
        _atomic_json(staging / "imputeecg.config.v3.json", imputeecg)
        _atomic_json(staging / "ecgrecover.config.v3.json", ecgrecover)
        combined["method_config_sha256"] = {
            "imputeecg": sha256_file(staging / "imputeecg.config.v3.json"),
            "ecgrecover": sha256_file(staging / "ecgrecover.config.v3.json"),
        }
        _atomic_json(staging / "official-reconstruction-config-v3.json", combined)
        config_artifacts = {
            name: {
                "path": str(output / filename),
                "sha256": sha256_file(staging / filename),
            }
            for name, filename in {
                "combined": "official-reconstruction-config-v3.json",
                "imputeecg": "imputeecg.config.v3.json",
                "ecgrecover": "ecgrecover.config.v3.json",
            }.items()
        }
        summary = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "manifest": {
                "path": str(arguments.manifest.resolve()),
                "file_sha256": sha256_file(arguments.manifest.resolve()),
                "sha256": contract.manifest_sha256,
                "split_sha256": contract.split_sha256,
                "verified_500hz_records": len(all_ids),
            },
            "train_role": "folds1-7/train",
            "training_inclusion": {
                "path": str(training_inclusion.path),
                "file_sha256": sha256_file(training_inclusion.path),
                "sha256": training_inclusion.inclusion_sha256,
                "record_ids_sha256": training_inclusion.record_ids_sha256,
                "patient_ids_sha256": training_inclusion.patient_ids_sha256,
                "n_requested_records": len(training_inclusion.requested_record_ids),
                "n_included_records": len(training_inclusion.included_record_ids),
                "n_excluded_records": (
                    len(training_inclusion.requested_record_ids)
                    - len(training_inclusion.included_record_ids)
                ),
            },
            "n_train_records": len(record_ids),
            "rate_hz": PRIMARY_RATE_HZ,
            "normalization": "raw_mV",
            "seeds": list(arguments.seeds),
            "sources": {
                "imputeecg": {
                    "repository": IMPUTE_ECG.repository,
                    "commit": IMPUTE_ECG.commit,
                    "source_dir": str(source_dirs["imputeecg"]),
                    "source_tree_sha256": source_trees["imputeecg"],
                },
                "ecgrecover": {
                    "repository": ECG_RECOVER.repository,
                    "commit": ECG_RECOVER.commit,
                    "source_dir": str(source_dirs["ecgrecover"]),
                    "source_tree_sha256": source_trees["ecgrecover"],
                    "integration_descriptor_sha256": integration.descriptor_sha256,
                },
            },
            "arrays": serialized_arrays,
            "configs": config_artifacts,
            "checkpoints_created": False,
            "command": [sys.executable, *sys.argv],
        }
        summary["preparation_sha256"] = canonical_sha256(summary)
        _atomic_json(staging / "preparation-summary.v3.json", summary)
        os.replace(staging, output)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return summary


def main() -> int:
    arguments = build_parser().parse_args()
    try:
        summary = run(arguments)
    except Exception as exc:
        raise SystemExit(f"official baseline preparation failed closed: {exc}") from exc
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
