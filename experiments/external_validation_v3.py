"""External Chapman/CPSC validation under the frozen ICASSP-v3 protocol.

``zero-transfer`` reloads materialized PTB-XL checkpoints and evaluates the
untouched external 20% patient test split.  It contains no fitting code.

``cohort-maps`` is a secondary analysis fitted only on the external 60% patient
train split.  It compares recoverability rankings with the frozen PTB-XL map and
never reads external test signals.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from ecgcert import lineage
from ecgcert.benchmarking import (
    EXPECTED_METHODS,
    compare_recoverability_rankings,
    evaluate_zero_transfer_bundles,
    load_benchmark_bundles,
    path_sha256,
)
from ecgcert.data.audit import AuditTrail, SignalAudit
from ecgcert.data.common import CANONICAL_LEADS
from ecgcert.data.external import ExternalWFDBCohort
from ecgcert.data.manifest import DatasetManifest
from ecgcert.data.ptbxl import PTBXL
from ecgcert.protocol import (
    BOOTSTRAP_REPLICATES,
    PRIMARY_RATE_HZ,
    PRIMARY_SEGMENTS,
    RANK_GRID,
    StudyProtocol,
    all_independent_configurations,
    configuration_panel_sha256,
)
from ecgcert.reconstruction import (
    EvaluationRecord,
    evaluation_records_sha256,
    load_fitted_reconstructor,
    load_training_predictors,
    write_benchmark_artifacts,
)
from ecgcert.recoverability import bootstrap_spatial_model_bank

try:
    from experiments.reconstruction_benchmark_v3 import load_ptbxl_manifest
    from experiments.robust_maps_v3 import SCHEMA_VERSION as MAP_SCHEMA_VERSION
    from experiments.robust_maps_v3 import summarize_model_bank
except ModuleNotFoundError as exc:
    if exc.name != "experiments":
        raise
    from reconstruction_benchmark_v3 import load_ptbxl_manifest
    from robust_maps_v3 import SCHEMA_VERSION as MAP_SCHEMA_VERSION
    from robust_maps_v3 import summarize_model_bank


SCHEMA_VERSION = "external-validation-v3"
WINDOW_SAMPLES = 10 * PRIMARY_RATE_HZ
COHORTS = ("chapman", "cpsc2018")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    if frame.empty:
        raise ValueError(f"refusing to write empty artifact {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        frame.to_parquet(path, index=False, compression="zstd")
    except ImportError as exc:
        raise RuntimeError("external validation requires locked pyarrow") from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_manifest_records(manifest: DatasetManifest, record_ids: Sequence[str]) -> None:
    """Verify only the explicitly authorized partition's materialized files."""

    by_id = {record.record_id: record for record in manifest.records}
    root = Path(manifest.root)
    for record_id in record_ids:
        item = by_id[str(record_id)]
        header = root / item.relative_header
        if not header.is_file() or _sha256_file(header) != item.header_sha256:
            raise ValueError(f"header hash mismatch for {record_id}")
        if not item.signal_file or item.signal_size_bytes is None or not item.signal_sha256:
            raise ValueError(f"manifest lacks complete signal provenance for {record_id}")
        signal = root / item.signal_file
        if not signal.is_file() or signal.stat().st_size != item.signal_size_bytes:
            raise ValueError(f"signal size mismatch for {record_id}")
        if _sha256_file(signal) != item.signal_sha256:
            raise ValueError(f"signal hash mismatch for {record_id}")


def _load_external_records(
    manifest: DatasetManifest,
    record_ids: Sequence[str],
    *,
    delineator: str,
    cohort_factory: Callable[[DatasetManifest], Any] = ExternalWFDBCohort,
) -> tuple[tuple[EvaluationRecord, ...], AuditTrail]:
    """Attempt every authorized record and create the common first-10-second window."""

    cohort = cohort_factory(manifest)
    by_id = {record.record_id: record for record in manifest.records}
    trail = AuditTrail()
    records: list[EvaluationRecord] = []
    for raw_record_id in record_ids:
        record_id = str(raw_record_id)
        item = by_id[record_id]
        try:
            signal, base_audit = cohort.signal_with_audit(record_id, rate=PRIMARY_RATE_HZ)
            signal = np.asarray(signal, dtype=float)
            if signal.ndim != 2 or signal.shape[1] != 12 or not np.isfinite(signal).all():
                raise ValueError(f"invalid canonical signal shape/values {signal.shape}")
            if signal.shape[0] < WINDOW_SAMPLES:
                raise ValueError("record shorter than the frozen 10-second evaluation window")
            window = signal[:WINDOW_SAMPLES]
            segment_indices = PTBXL.segment_indices(
                window,
                fs=PRIMARY_RATE_HZ,
                method=delineator,
            )
            counts = {
                segment: int(segment_indices[segment].size) for segment in PRIMARY_SEGMENTS
            }
            if not any(counts.values()):
                raise ValueError("no delineated QRS/ST/T evaluation samples")
            trail.append(
                SignalAudit(
                    **{
                        **base_audit.__dict__,
                        "n_samples": WINDOW_SAMPLES,
                        "segment_counts": counts,
                    }
                )
            )
            records.append(
                EvaluationRecord(
                    patient_id=str(item.patient_id),
                    record_id=record_id,
                    signal=window.T,
                    segment_indices={
                        segment: np.asarray(segment_indices[segment], dtype=np.int64)
                        for segment in PRIMARY_SEGMENTS
                    },
                )
            )
        except Exception as exc:
            trail.append(
                SignalAudit(
                    cohort=manifest.cohort,
                    record_id=record_id,
                    patient_id=str(item.patient_id),
                    status="excluded",
                    reason=f"{type(exc).__name__}: {exc}",
                    requested_rate_hz=PRIMARY_RATE_HZ,
                )
            )
    if not records:
        raise ValueError("no external records have all QRS/ST/T evaluation windows")
    return tuple(records), trail


def _audit_for_injected_records(
    manifest: DatasetManifest,
    expected_record_ids: Sequence[str],
    records: Sequence[EvaluationRecord],
) -> AuditTrail:
    expected = {str(record_id) for record_id in expected_record_ids}
    observed = {str(record.record_id) for record in records}
    if observed != expected or len(records) != len(expected):
        raise ValueError("injected records must cover the authorized partition exactly once")
    by_id = {record.record_id: record for record in manifest.records}
    trail = AuditTrail()
    for record in records:
        record.validate()
        manifest_record = by_id[str(record.record_id)]
        if str(record.patient_id) != str(manifest_record.patient_id):
            raise ValueError(
                f"injected patient ID disagrees with manifest for {record.record_id}"
            )
        if not any(
            np.asarray(record.segment_indices.get(segment, ())).size
            for segment in PRIMARY_SEGMENTS
        ):
            raise ValueError(f"injected record {record.record_id} has no QRS/ST/T samples")
        trail.append(
            SignalAudit(
                cohort=manifest.cohort,
                record_id=str(record.record_id),
                patient_id=str(manifest_record.patient_id),
                status="included",
                reason=None,
                requested_rate_hz=PRIMARY_RATE_HZ,
                source_rate_hz=PRIMARY_RATE_HZ,
                n_samples=int(record.signal.shape[1]),
                segment_counts={
                    segment: int(np.asarray(record.segment_indices.get(segment, ())).size)
                    for segment in PRIMARY_SEGMENTS
                },
            )
        )
    return trail


def _validate_complete_attempt(
    trail: AuditTrail,
    expected_record_ids: Sequence[str],
) -> None:
    audited = [record.record_id for record in trail.records]
    expected = [str(record_id) for record_id in expected_record_ids]
    if len(audited) != len(expected) or set(audited) != set(expected):
        raise ValueError("audit trail does not cover the complete authorized partition")


def _require_disjoint_output(output: Path, protected_roots: Sequence[Path]) -> None:
    for protected in protected_roots:
        protected = protected.resolve()
        if protected.is_file():
            protected = protected.parent
        if output == protected or protected in output.parents:
            raise ValueError(f"output directory must not modify input artifact {protected}")


def _segment_samples(
    records: Sequence[EvaluationRecord],
    *,
    max_per_record: int,
    seed: int,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    rows = {segment: [] for segment in PRIMARY_SEGMENTS}
    record_ids = {segment: [] for segment in PRIMARY_SEGMENTS}
    patient_ids = {segment: [] for segment in PRIMARY_SEGMENTS}
    for record in records:
        for segment in PRIMARY_SEGMENTS:
            indices = np.asarray(record.segment_indices[segment], dtype=np.int64)
            if indices.size > max_per_record:
                indices = np.sort(rng.choice(indices, max_per_record, replace=False))
            values = np.asarray(record.signal[:, indices].T, dtype=float)
            rows[segment].append(values)
            record_ids[segment].append(
                np.full(values.shape[0], str(record.record_id), dtype=object)
            )
            patient_ids[segment].append(
                np.full(values.shape[0], str(record.patient_id), dtype=object)
            )
    return {
        segment: (
            np.vstack(rows[segment]),
            np.concatenate(record_ids[segment]),
            np.concatenate(patient_ids[segment]),
        )
        for segment in PRIMARY_SEGMENTS
    }


def _is_sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text.lower())


def _is_safe_relative(value: Any) -> bool:
    if not isinstance(value, str) or not value:
        return False
    path = Path(value)
    return bool(path.parts) and not path.is_absolute() and ".." not in path.parts


def _manifest(arguments: argparse.Namespace, override: DatasetManifest | None) -> DatasetManifest:
    manifest = override or DatasetManifest.from_path(arguments.target_manifest)
    if manifest.cohort != arguments.cohort:
        raise ValueError(
            f"target manifest cohort {manifest.cohort!r} does not match {arguments.cohort!r}"
        )
    if not manifest.records:
        raise ValueError("target DatasetManifest is empty")
    record_ids = [str(record.record_id) for record in manifest.records]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("target DatasetManifest contains duplicate record IDs")
    for record in manifest.records:
        if (
            not str(record.record_id)
            or not str(record.patient_id)
            or not _is_safe_relative(record.record_id)
            or not _is_safe_relative(record.relative_header)
            or not _is_sha256(record.header_sha256)
            or not _is_safe_relative(record.signal_file)
            or isinstance(record.signal_size_bytes, bool)
            or not isinstance(record.signal_size_bytes, int)
            or record.signal_size_bytes <= 0
            or not _is_sha256(record.signal_sha256)
        ):
            raise ValueError(
                f"target DatasetManifest has incomplete provenance for {record.record_id}"
            )
    split = manifest.split()
    split.validate()
    if split.calibration:
        raise ValueError("external manifest must not define a calibration partition")
    if override is None and (not split.train or not split.tune or not split.test):
        raise ValueError("external manifest must realize the fixed non-empty 60/20/20 split")
    return manifest


def run_zero_transfer(
    arguments: argparse.Namespace,
    *,
    target_manifest: DatasetManifest | None = None,
    source_manifest_sha256: str | None = None,
    rank_maps_sha256: str | None = None,
    records: Sequence[EvaluationRecord] | None = None,
    model_loader: Callable[..., Any] = load_fitted_reconstructor,
    predictor_loader: Callable[[str | Path], Any] = load_training_predictors,
) -> dict[str, Any]:
    """Execute checkpoint-only external inference; dependency injection is test-only."""

    if getattr(arguments, "release", False):
        validate_arguments(arguments)
        if (
            target_manifest is not None
            or source_manifest_sha256 is not None
            or rank_maps_sha256 is not None
            or records is not None
            or model_loader is not load_fitted_reconstructor
            or predictor_loader is not load_training_predictors
        ):
            raise ValueError("release zero-transfer forbids dependency injection")
    manifest = _manifest(arguments, target_manifest)
    split = manifest.split()
    test_ids = tuple(str(record_id) for record_id in split.test)
    if not test_ids:
        raise ValueError("external manifest has an empty fixed test partition")
    if records is None:
        manifest.verify_files()
        evaluation_records, trail = _load_external_records(
            manifest,
            test_ids,
            delineator=arguments.delineator,
        )
    else:
        evaluation_records = tuple(records)
        trail = _audit_for_injected_records(manifest, test_ids, evaluation_records)
    _validate_complete_attempt(trail, test_ids)

    if source_manifest_sha256 is None:
        source_manifest_sha256 = load_ptbxl_manifest(
            arguments.source_manifest
        ).manifest_sha256
    if rank_maps_sha256 is None:
        rank_maps_sha256 = path_sha256(arguments.rank_maps)
    bundles = load_benchmark_bundles(
        arguments.benchmark,
        source_manifest_sha256=source_manifest_sha256,
        rank_maps_sha256=rank_maps_sha256,
        release=arguments.release,
    )
    output = arguments.output_dir.resolve()
    protected = [bundle.root for bundle in bundles.values()]
    if arguments.rank_maps is not None:
        rank_maps_path = Path(arguments.rank_maps).resolve()
        if rank_maps_path.is_dir():
            protected.append(rank_maps_path)
    _require_disjoint_output(output, protected)
    metrics = evaluate_zero_transfer_bundles(
        bundles,
        evaluation_records,
        cohort=manifest.cohort,
        device=arguments.device,
        model_loader=model_loader,
        predictor_loader=predictor_loader,
    )
    predictor_content_sha256 = metrics.attrs.get(
        "training_predictors_content_sha256"
    )
    if not isinstance(predictor_content_sha256, str) or len(predictor_content_sha256) != 64:
        raise RuntimeError("zero-transfer evaluation lacks predictor content lineage")

    audit = {
        "schema_version": SCHEMA_VERSION,
        "mode": "zero-transfer",
        "cohort": manifest.cohort,
        "partition": "test",
        "rate_hz": PRIMARY_RATE_HZ,
        "segments": list(PRIMARY_SEGMENTS),
        "window_policy": "first contiguous 10 seconds at 500 Hz",
        "requested_record_ids_sha256": lineage.canonical_sha256(sorted(test_ids)),
        "source_manifest_sha256": source_manifest_sha256,
        "target_manifest_sha256": manifest.sha256(),
        "target_split_sha256": split.sha256(),
        "rank_maps_sha256": rank_maps_sha256,
        "training_predictors_content_sha256": predictor_content_sha256,
        "evaluation_records_sha256": evaluation_records_sha256(
            evaluation_records,
            segments=PRIMARY_SEGMENTS,
        ),
        "benchmark_bundles": {
            method: {
                "bundle_sha256": lineage.artifact_sha256(
                    bundle.root / "bundle.v3.json"
                ),
                "training_predictors_sha256": bundle.training_predictors_sha256,
                "seeds": list(bundle.seeds),
                "configurations": [list(config) for config in bundle.configurations],
            }
            for method, bundle in bundles.items()
        },
        "data_audit": trail.to_dict(),
        "no_external_fit": True,
    }
    audit_path = output / "evaluation_audit.json"
    _atomic_json(audit_path, audit)
    summary = {
        "mode": "zero-transfer",
        "cohort": manifest.cohort,
        "partition": "test",
        "claim_scope": "true checkpoint-only external transfer; no external fine-tuning",
        "rate_hz": PRIMARY_RATE_HZ,
        "segments": list(PRIMARY_SEGMENTS),
        "n_test_records_requested": len(test_ids),
        "n_test_records_included": len(evaluation_records),
        "n_test_records_excluded": len(test_ids) - len(evaluation_records),
        "source_manifest_sha256": source_manifest_sha256,
        "target_manifest_sha256": manifest.sha256(),
        "target_split_sha256": split.sha256(),
        "rank_maps_sha256": rank_maps_sha256,
        "training_predictors_content_sha256": predictor_content_sha256,
        "configuration_panel_sha256": configuration_panel_sha256(),
        "common_panel_methods": list(EXPECTED_METHODS[:-1]),
        "ecgrecover_configuration": list(bundles["ecgrecover"].configurations[0]),
        "external_training_or_adaptation": "forbidden_and_not_performed",
        "benchmark_bundles": {
            method: {
                "path": str(bundle.root),
                "bundle_sha256": lineage.artifact_sha256(bundle.root / "bundle.v3.json"),
                "seeds": list(bundle.seeds),
                "n_configurations": len(bundle.configurations),
                "training_predictors_sha256": bundle.training_predictors_sha256,
            }
            for method, bundle in bundles.items()
        },
        "artifacts": {
            "evaluation_audit": {
                "path": audit_path.name,
                "sha256": lineage.artifact_sha256(audit_path),
            }
        },
    }
    return write_benchmark_artifacts(metrics, output, summary=summary)


def _validate_cohort_segment_data(
    segment_data: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    train_record_ids: Sequence[str],
    test_record_ids: Sequence[str],
    patient_by_record: Mapping[str, str],
) -> None:
    train = set(map(str, train_record_ids))
    test = set(map(str, test_record_ids))
    for segment in PRIMARY_SEGMENTS:
        if segment not in segment_data:
            raise ValueError(f"cohort-map data lacks segment {segment}")
        X, record_ids, patient_ids = segment_data[segment]
        X = np.asarray(X)
        record_ids = np.asarray(record_ids, dtype=object)
        patient_ids = np.asarray(patient_ids, dtype=object)
        if (
            X.ndim != 2
            or X.shape[1] != 12
            or not len(X)
            or len(record_ids) != len(X)
            or len(patient_ids) != len(X)
            or not np.isfinite(X).all()
        ):
            raise ValueError(f"invalid cohort-map arrays for {segment}")
        used = set(map(str, record_ids))
        if not used <= train:
            raise ValueError(f"cohort-map {segment} contains records outside the 60% train split")
        if used & test:
            raise ValueError(f"cohort-map {segment} accessed the untouched test split")
        for record_id, patient_id in zip(record_ids, patient_ids, strict=True):
            if str(patient_id) != str(patient_by_record[str(record_id)]):
                raise ValueError(
                    f"cohort-map {segment} patient ID disagrees with manifest for {record_id}"
                )


def run_cohort_maps(
    arguments: argparse.Namespace,
    *,
    target_manifest: DatasetManifest | None = None,
    records: Sequence[EvaluationRecord] | None = None,
    segment_data: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
    configurations: Sequence[Sequence[str]] | None = None,
    ranks: Sequence[int] = RANK_GRID,
) -> dict[str, Any]:
    """Fit secondary maps only on the external 60% patient partition."""

    if getattr(arguments, "release", False):
        validate_arguments(arguments)
        if (
            target_manifest is not None
            or records is not None
            or segment_data is not None
            or configurations is not None
            or tuple(ranks) != RANK_GRID
        ):
            raise ValueError("release cohort maps forbid dependency injection")
    manifest = _manifest(arguments, target_manifest)
    split = manifest.split()
    train_ids = tuple(str(record_id) for record_id in split.train)
    test_ids = tuple(str(record_id) for record_id in split.test)
    if not train_ids or not test_ids:
        raise ValueError("external manifest must have non-empty train and test partitions")
    if segment_data is None:
        if records is None:
            _verify_manifest_records(manifest, train_ids)
            fit_records, trail = _load_external_records(
                manifest,
                train_ids,
                delineator=arguments.delineator,
            )
        else:
            fit_records = tuple(records)
            trail = _audit_for_injected_records(manifest, train_ids, fit_records)
        _validate_complete_attempt(trail, train_ids)
        segment_data = _segment_samples(fit_records, max_per_record=40, seed=arguments.seed)
    else:
        if records is None:
            raise ValueError("injected segment_data requires matching injected train records")
        fit_records = tuple(records)
        trail = _audit_for_injected_records(manifest, train_ids, fit_records)
        _validate_complete_attempt(trail, train_ids)
    _validate_cohort_segment_data(
        segment_data,
        train_record_ids=train_ids,
        test_record_ids=test_ids,
        patient_by_record={
            str(record.record_id): str(record.patient_id) for record in manifest.records
        },
    )

    primary_root = arguments.primary_rank_maps.resolve()
    primary_rank_maps_sha256 = path_sha256(primary_root)
    primary_cells_path = primary_root / "map_cells.parquet"
    primary_summary_path = primary_root / "summary.v3.json"
    if not primary_cells_path.is_file() or not primary_summary_path.is_file():
        raise FileNotFoundError("primary rank-map bundle is incomplete")
    primary_cells = pd.read_parquet(primary_cells_path)
    primary_summary = json.loads(primary_summary_path.read_text(encoding="utf-8"))
    protocol = asdict(StudyProtocol())
    serialized_protocol = json.loads(json.dumps(protocol))
    expected_primary = {
        "schema_version": MAP_SCHEMA_VERSION,
        "status": "complete",
        "cohort": "PTB-XL",
        "population": "all",
        "analysis_mode": "primary",
        "delineator": "dwt",
        "basis_variant": "independent8_lifted",
        "rate_hz": PRIMARY_RATE_HZ,
        "segments": list(PRIMARY_SEGMENTS),
        "ranks": list(RANK_GRID),
        "n_configurations": len(all_independent_configurations()),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_unit": "patient",
        "deep_panel_sha256": configuration_panel_sha256(),
        "protocol": serialized_protocol,
        "protocol_sha256": lineage.canonical_sha256(protocol),
    }
    mismatches = [
        field
        for field, expected in expected_primary.items()
        if primary_summary.get(field) != expected
    ]
    if mismatches:
        raise ValueError(
            f"primary rank-map bundle violates frozen protocol fields: {mismatches}"
        )
    if primary_summary.get("artifacts", {}).get("map_cells") != "map_cells.parquet":
        raise ValueError("primary rank-map summary does not declare map_cells.parquet")
    if int(primary_summary.get("n_map_cells", -1)) != len(primary_cells):
        raise ValueError("primary rank-map row count disagrees with its summary")
    required_primary_columns = {
        "schema_version",
        "segment",
        "configuration",
        "target",
        "target_observed",
        "recoverability_lower",
    }
    missing_primary_columns = required_primary_columns - set(primary_cells.columns)
    if missing_primary_columns:
        raise ValueError(
            f"primary rank-map cells lack columns: {sorted(missing_primary_columns)}"
        )
    primary_keys = ["segment", "configuration", "target"]
    if primary_cells.duplicated(primary_keys).any():
        raise ValueError("primary rank-map cells contain duplicate keys")
    expected_keys = {
        (segment, "+".join(configuration), target)
        for segment in PRIMARY_SEGMENTS
        for configuration in all_independent_configurations()
        for target in CANONICAL_LEADS
    }
    actual_keys = set(
        primary_cells.loc[:, primary_keys].itertuples(index=False, name=None)
    )
    if actual_keys != expected_keys:
        raise ValueError("primary rank-map cells do not cover the frozen full map")
    if set(primary_cells["schema_version"]) != {MAP_SCHEMA_VERSION}:
        raise ValueError("primary rank-map cells have the wrong schema")
    observed_lookup = {
        "+".join(configuration): set(configuration)
        for configuration in all_independent_configurations()
    }
    expected_observed = np.fromiter(
        (
            str(row.target) in observed_lookup[str(row.configuration)]
            for row in primary_cells.itertuples(index=False)
        ),
        dtype=bool,
        count=len(primary_cells),
    )
    if not np.array_equal(
        primary_cells["target_observed"].to_numpy(dtype=bool), expected_observed
    ):
        raise ValueError("primary rank-map observed-target flags are inconsistent")
    recoverability = primary_cells["recoverability_lower"].to_numpy(dtype=float)
    if not np.isfinite(recoverability).all() or np.any(
        (recoverability < 0.0) | (recoverability > 1.0)
    ):
        raise ValueError("primary recoverability scores must be finite in [0,1]")
    observation_variance = float(primary_summary["observation_variance_mv2"])
    if not np.isfinite(observation_variance) or observation_variance <= 0:
        raise ValueError("primary rank map has invalid observation variance")

    configurations = tuple(
        all_independent_configurations() if configurations is None else map(tuple, configurations)
    )
    ranks = tuple(int(rank) for rank in ranks)
    rank_frames = []
    map_frames = []
    rejected = {}
    for segment in PRIMARY_SEGMENTS:
        X, _, patient_ids = segment_data[segment]
        bank = bootstrap_spatial_model_bank(
            X,
            patient_ids,
            ranks=ranks,
            basis_variants=("independent8_lifted",),
            fit_cohort=f"{manifest.cohort}/external-train60/{segment}",
            n_boot=arguments.n_bootstrap,
            seed=arguments.seed + 1000 * PRIMARY_SEGMENTS.index(segment),
        )
        rank_frame, map_frame = summarize_model_bank(
            bank,
            configurations,
            segment=segment,
            observation_variance_mv2=observation_variance,
        )
        for frame in (rank_frame, map_frame):
            frame.insert(1, "cohort", manifest.cohort)
            frame.insert(2, "partition", "train")
        rank_frames.append(rank_frame)
        map_frames.append(map_frame)
        rejected[segment] = {
            "draws": bank.rejected_draws,
            "fraction": bank.rejection_fraction,
        }
    rank_path = pd.concat(rank_frames, ignore_index=True)
    map_cells = pd.concat(map_frames, ignore_index=True)
    agreement = compare_recoverability_rankings(map_cells, primary_cells)
    overall = agreement[
        (agreement["segment"] == "__all__") & (agreement["target"] == "__all__")
    ]
    if overall.empty or not np.isfinite(float(overall.iloc[0]["spearman_rho"])):
        raise ValueError("overall PTB/external recoverability ranking is undefined")

    output = arguments.output_dir.resolve()
    _require_disjoint_output(output, [primary_root])
    if path_sha256(primary_root) != primary_rank_maps_sha256:
        raise RuntimeError("primary rank-map bundle changed during cohort-map analysis")
    _write_parquet(rank_path, output / "rank_path.parquet")
    _write_parquet(map_cells, output / "map_cells.parquet")
    _write_parquet(agreement, output / "ranking_spearman.parquet")
    audit = {
        "schema_version": SCHEMA_VERSION,
        "mode": "cohort-maps",
        "cohort": manifest.cohort,
        "partition": "train",
        "data_audit": trail.to_dict(),
        "test_records_accessed": 0,
        "train_record_ids_sha256": lineage.canonical_sha256(sorted(train_ids)),
        "test_record_ids_sha256_membership_only": lineage.canonical_sha256(sorted(test_ids)),
        "target_manifest_sha256": manifest.sha256(),
        "target_split_sha256": split.sha256(),
        "primary_rank_maps_sha256": primary_rank_maps_sha256,
    }
    _atomic_json(output / "evaluation_audit.json", audit)
    protocol = asdict(StudyProtocol())
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "mode": "cohort-maps",
        "claim_scope": "secondary external-train map; no external test outcomes",
        "cohort": manifest.cohort,
        "partition": "train",
        "rate_hz": PRIMARY_RATE_HZ,
        "segments": list(PRIMARY_SEGMENTS),
        "ranks": list(ranks),
        "n_configurations": len(configurations),
        "bootstrap_replicates": arguments.n_bootstrap,
        "bootstrap_unit": "patient",
        "bootstrap_rank_deficient_draws": rejected,
        "target_manifest_sha256": manifest.sha256(),
        "target_split_sha256": split.sha256(),
        "primary_rank_maps_sha256": primary_rank_maps_sha256,
        "observation_variance_mv2": observation_variance,
        "ranking_metric": "Spearman(recoverability_lower), missing targets only",
        "overall_spearman_rho": float(overall.iloc[0]["spearman_rho"]),
        "test_records_accessed": 0,
        "protocol": protocol,
        "protocol_sha256": lineage.canonical_sha256(protocol),
        "artifacts": {
            "rank_path": "rank_path.parquet",
            "map_cells": "map_cells.parquet",
            "ranking_spearman": "ranking_spearman.parquet",
            "evaluation_audit": "evaluation_audit.json",
        },
    }
    _atomic_json(output / "summary.v3.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("zero-transfer", "cohort-maps"), required=True)
    parser.add_argument("--cohort", choices=COHORTS, required=True)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--target-manifest", type=Path, required=True)
    parser.add_argument("--rank-maps", type=Path)
    parser.add_argument("--primary-rank-maps", type=Path)
    parser.add_argument("--benchmark", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--delineator", choices=("dwt", "peak"), default="dwt")
    parser.add_argument("--n-bootstrap", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--release", action="store_true")
    return parser


def validate_arguments(arguments: argparse.Namespace) -> None:
    violations = []
    if arguments.mode == "zero-transfer":
        if arguments.source_manifest is None:
            violations.append("--source-manifest is required")
        if arguments.rank_maps is None:
            violations.append("--rank-maps is required")
        if len(arguments.benchmark) != len(EXPECTED_METHODS):
            violations.append(f"exactly {len(EXPECTED_METHODS)} --benchmark bundles are required")
        if arguments.primary_rank_maps is not None:
            violations.append("--primary-rank-maps is cohort-maps only")
    else:
        if arguments.primary_rank_maps is None:
            violations.append("--primary-rank-maps is required")
        if arguments.source_manifest is not None or arguments.rank_maps is not None:
            violations.append("cohort-maps cannot consume source/test benchmark inputs")
        if arguments.benchmark:
            violations.append("cohort-maps cannot consume reconstruction benchmarks")
        if arguments.n_bootstrap < 1:
            violations.append("--n-bootstrap must be positive")
    if arguments.release:
        if arguments.delineator != "dwt":
            violations.append("release delineator must be dwt")
        if arguments.mode == "cohort-maps" and arguments.n_bootstrap != BOOTSTRAP_REPLICATES:
            violations.append("release cohort maps require exactly 2000 patient bootstraps")
        try:
            arguments.output_dir.resolve().relative_to((Path.cwd() / "artifacts").resolve())
        except ValueError:
            violations.append("release output must be under artifacts/")
    if violations:
        raise ValueError("external validation protocol violation: " + "; ".join(violations))


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        validate_arguments(arguments)
        if arguments.mode == "zero-transfer":
            summary = run_zero_transfer(arguments)
        else:
            summary = run_cohort_maps(arguments)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"external validation failed closed: {exc}") from exc
    print(
        f"[{arguments.cohort}/{arguments.mode}] status={summary['status']} -> "
        f"{arguments.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
