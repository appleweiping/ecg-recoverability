"""Fit the preregistered reconstruction grid and score it on PTB-XL fold 8.

This is the executable producer consumed by :mod:`tune_reconstructors_v3`.
Every candidate is fitted only on folds 1--7.  Low-rank and ridge candidates
are evaluated directly on the complete frozen fold-8 panel.  Masked U-Net
candidates use fold 8 only as a validation set: each seed is stopped with the
frozen patience rule, its best state is restored, and that state is evaluated
once on the complete patient/configuration/segment/target panel.  The compact
best-state representation avoids materialising the same multi-million-row
patient table at every neural training epoch.

No fold-9 calibration or fold-10 test record is loaded by this entry point.
Official methods are deliberately absent: their hyperparameters are pinned by
their official protocols and are never selected by held-out outcomes.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ecgcert import lineage
from ecgcert.data import PTBXL
from ecgcert.data.common import CANONICAL_LEADS
from ecgcert.estimators import (
    LowRankConditionalMeanReconstructor,
    MaskedUNetReconstructor,
    RidgeLeadReconstructor,
    TrainManifest,
)
from ecgcert.estimators.masked_unet import _build_unet
from ecgcert.protocol import (
    PRIMARY_RATE_HZ,
    PRIMARY_SEGMENTS,
    configuration_panel_sha256,
    deep_configuration_panel,
)
from ecgcert.reconstruction import (
    EvaluationRecord,
    TRAINING_PREDICTORS_FILENAME,
    evaluate_reconstructor,
    evaluation_records_sha256,
    training_predictor_lookup,
)

try:  # package import in tests; sibling import under ``python experiments/...``
    from .reconstruction_benchmark_v3 import (
        _atomic_json,
        _load_evaluation_records,
        _materialize_training_manifest,
        _require_parquet_engine,
        _streaming_training_moments,
        _validate_database_identity,
        _verify_manifest_files,
        load_ptbxl_manifest,
    )
    from .tune_reconstructors_v3 import (
        CANDIDATE_BUNDLE_SCHEMA_VERSION,
        CANDIDATE_SCHEMA_VERSION,
        TRACE_SCHEMA_VERSION,
        TUNING_SEEDS,
        Candidate,
        _validate_early_stop_trace,
        _validate_candidate_metrics,
        _verify_candidate_checkpoints,
        candidate_grid,
    )
except ImportError:  # pragma: no cover - direct CLI import path
    from reconstruction_benchmark_v3 import (
        _atomic_json,
        _load_evaluation_records,
        _materialize_training_manifest,
        _require_parquet_engine,
        _streaming_training_moments,
        _validate_database_identity,
        _verify_manifest_files,
        load_ptbxl_manifest,
    )
    from tune_reconstructors_v3 import (
        CANDIDATE_BUNDLE_SCHEMA_VERSION,
        CANDIDATE_SCHEMA_VERSION,
        TRACE_SCHEMA_VERSION,
        TUNING_SEEDS,
        Candidate,
        _validate_early_stop_trace,
        _validate_candidate_metrics,
        _verify_candidate_checkpoints,
        candidate_grid,
    )


CANDIDATE_METRICS_FILENAME = "candidate_metrics.parquet"
EARLY_STOP_TRACE_FILENAME = "unet_early_stop_trace.parquet"
CANDIDATE_BUNDLE_FILENAME = "candidate_bundle.json"
CANDIDATE_COLUMNS = (
    "schema_version",
    "cohort",
    "train_partition",
    "partition",
    "manifest_sha256",
    "split_sha256",
    "method",
    "candidate_id",
    "patient_id",
    "segment",
    "configuration",
    "target",
    "model_seed",
    "epoch",
    "rmse_mv",
    "log_rmse_mv",
    "checkpoint_path",
    "checkpoint_sha256",
    "observed_integrity",
)
TRACE_COLUMNS = (
    "schema_version",
    "cohort",
    "train_partition",
    "partition",
    "manifest_sha256",
    "split_sha256",
    "fold8_records_sha256",
    "configuration_panel_sha256",
    "candidate_id",
    "model_seed",
    "epoch",
    "monitor_log_rmse_mv",
    "best_so_far_log_rmse_mv",
    "stale_epochs",
    "is_strict_improvement",
    "best_epoch",
    "stopped_epoch",
    "checkpoint_path",
    "checkpoint_sha256",
)


def _parse_csv(value: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("value must contain at least one item")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rate", type=int, default=PRIMARY_RATE_HZ)
    parser.add_argument("--segments", type=_parse_csv, default=PRIMARY_SEGMENTS)
    parser.add_argument("--delineator", choices=("dwt", "peak"), default="dwt")
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--max-configurations", type=int)
    parser.add_argument("--release", action="store_true")
    return parser


def validate_release_arguments(arguments: argparse.Namespace) -> None:
    if arguments.max_records is not None and arguments.max_records < 1:
        raise ValueError("--max-records must be positive")
    if arguments.max_configurations is not None and arguments.max_configurations < 1:
        raise ValueError("--max-configurations must be positive")
    if not arguments.release:
        return
    violations = []
    if arguments.max_records is not None:
        violations.append("--max-records is forbidden")
    if arguments.max_configurations is not None:
        violations.append("--max-configurations is forbidden")
    if arguments.rate != PRIMARY_RATE_HZ:
        violations.append(f"--rate must equal {PRIMARY_RATE_HZ}")
    if tuple(arguments.segments) != PRIMARY_SEGMENTS:
        violations.append(f"--segments must equal {PRIMARY_SEGMENTS}")
    if arguments.delineator != "dwt":
        violations.append("--delineator must be dwt")
    try:
        arguments.output_dir.resolve().relative_to((Path.cwd() / "artifacts").resolve())
    except ValueError:
        violations.append("--output-dir must be under artifacts/ for release")
    if violations:
        raise ValueError("release candidate-grid protocol violation: " + "; ".join(violations))


def _checkpoint_fields(checkpoint: Path, output_dir: Path) -> tuple[str, str]:
    relative = checkpoint.resolve().relative_to(output_dir.resolve()).as_posix()
    return relative, lineage.artifact_sha256(checkpoint)


def candidate_metric_rows(
    metrics: pd.DataFrame,
    *,
    candidate: Candidate,
    model_seed: int,
    epoch: int,
    checkpoint: Path,
    output_dir: Path,
    manifest_sha256: str,
    split_sha256: str,
) -> pd.DataFrame:
    """Convert shared benchmark metrics into the strict fold-8 tuning schema."""

    if metrics.empty:
        raise ValueError(f"{candidate.candidate_id} produced no fold-8 metrics")
    if set(metrics["partition"].astype(str)) != {"fold8/tune"}:
        raise ValueError("candidate metrics may only contain fold8/tune rows")
    if set(metrics["method"].astype(str)) != {candidate.method}:
        raise ValueError("candidate metric method does not match its grid declaration")
    if set(metrics["model_seed"].astype(int)) != {int(model_seed)}:
        raise ValueError("candidate metric seed does not match its fitted model")
    if not np.array_equal(
        metrics["outcome_log_rmse"].to_numpy(dtype=float),
        metrics["log_rmse_mv"].to_numpy(dtype=float),
    ):
        raise ValueError("candidate primary outcome must be raw log(RMSE_mV)")
    checkpoint_path, checkpoint_sha256 = _checkpoint_fields(checkpoint, output_dir)
    output = metrics.loc[
        :, ["patient_id", "segment", "configuration", "target", "rmse_mv", "log_rmse_mv"]
    ].copy()
    output.insert(0, "candidate_id", candidate.candidate_id)
    output.insert(0, "method", candidate.method)
    output.insert(0, "split_sha256", split_sha256)
    output.insert(0, "manifest_sha256", manifest_sha256)
    output.insert(0, "partition", "fold8/tune")
    output.insert(0, "train_partition", "folds1-7/train")
    output.insert(0, "cohort", "PTB-XL")
    output.insert(0, "schema_version", CANDIDATE_SCHEMA_VERSION)
    output["model_seed"] = int(model_seed)
    output["epoch"] = int(epoch)
    output["checkpoint_path"] = checkpoint_path
    output["checkpoint_sha256"] = checkpoint_sha256
    # ``evaluate_reconstructor`` already fails closed if a method changes any
    # observed sample.  This explicit flag makes that successful check auditable.
    output["observed_integrity"] = True
    return output.loc[:, list(CANDIDATE_COLUMNS)]


def _linear_candidate_checkpoint(
    method: str,
    candidate: Candidate,
    output_dir: Path,
    *,
    mean: np.ndarray,
    scatter: np.ndarray,
    sample_count: int,
    covariance: np.ndarray | None = None,
) -> Path:
    checkpoint = output_dir / "checkpoints" / method / candidate.candidate_id / "seed-0.npz"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "mean": np.asarray(mean, dtype=np.float64),
        "scatter": np.asarray(scatter, dtype=np.float64),
        "sample_count": np.asarray([sample_count], dtype=np.int64),
        "candidate_parameters_json": np.asarray(
            [json.dumps(dict(candidate.parameters), sort_keys=True, allow_nan=False)]
        ),
    }
    if covariance is not None:
        payload["covariance"] = np.asarray(covariance, dtype=np.float64)
    np.savez(checkpoint, **payload)
    return checkpoint


def evaluate_linear_candidate_grid(
    method: str,
    *,
    mean: np.ndarray,
    scatter: np.ndarray,
    sample_count: int,
    evaluation_records: Sequence[EvaluationRecord],
    configurations: Sequence[Sequence[str]],
    segments: Sequence[str],
    training_predictors: Mapping[tuple[str, str, str], tuple[float, float]],
    output_dir: Path,
    manifest_sha256: str,
    split_sha256: str,
) -> list[pd.DataFrame]:
    """Fit and evaluate every real low-rank or ridge candidate."""

    if method not in {"lowrank", "ridge"}:
        raise ValueError(f"not a native linear candidate method: {method}")
    frames: list[pd.DataFrame] = []
    for candidate in candidate_grid()[method]:
        covariance = None
        ridge_lambda = None
        if method == "lowrank":
            rank = int(candidate.parameters["rank"])
            noise_variance = float(candidate.parameters["noise_variance"])
            eigenvalues, eigenvectors = np.linalg.eigh(scatter)
            order = np.argsort(eigenvalues)[::-1][:rank]
            basis = eigenvectors[:, order]
            coordinate_variance = np.maximum(eigenvalues[order], 0.0) / (
                sample_count - 1
            )
            covariance = (basis * coordinate_variance[None, :]) @ basis.T
            covariance += noise_variance * np.eye(len(CANONICAL_LEADS))
        else:
            ridge_lambda = float(candidate.parameters["ridge_lambda"])
        checkpoint = _linear_candidate_checkpoint(
            method,
            candidate,
            output_dir,
            mean=mean,
            scatter=scatter,
            sample_count=sample_count,
            covariance=covariance,
        )
        for configuration in configurations:
            observed = np.asarray(
                [CANONICAL_LEADS.index(lead) for lead in configuration], dtype=np.int64
            )
            if method == "lowrank":
                assert covariance is not None
                model = LowRankConditionalMeanReconstructor()
                model.mean = np.asarray(mean, dtype=np.float64)
                model.covariance = covariance
                model.observed = observed
            else:
                assert ridge_lambda is not None
                gram = scatter[np.ix_(observed, observed)] + ridge_lambda * np.eye(
                    observed.size
                )
                model = RidgeLeadReconstructor()
                model.x_mean = np.asarray(mean[observed], dtype=np.float64)
                model.y_mean = np.asarray(mean, dtype=np.float64)
                model.weights = np.linalg.solve(gram, scatter[observed]).T
                model.observed = observed
            model._checkpoint_path = checkpoint
            model._fitted = True
            metrics = evaluate_reconstructor(
                model,
                evaluation_records,
                configuration=configuration,
                method=method,
                model_seed=0,
                segments=segments,
                training_predictors=training_predictors,
                cohort="PTB-XL",
                partition="fold8/tune",
            )
            frames.append(
                candidate_metric_rows(
                    metrics,
                    candidate=candidate,
                    model_seed=0,
                    epoch=0,
                    checkpoint=checkpoint,
                    output_dir=output_dir,
                    manifest_sha256=manifest_sha256,
                    split_sha256=split_sha256,
                )
            )
    return frames


def _patient_balanced_score(frame: pd.DataFrame) -> float:
    patients = frame.groupby("patient_id", sort=True)["log_rmse_mv"].mean()
    if patients.empty or not np.isfinite(patients.to_numpy(dtype=float)).all():
        raise ValueError("fold-8 early-stopping monitor is empty or non-finite")
    return float(patients.mean())


def _early_stop_fold8_score(
    model: MaskedUNetReconstructor,
    records: Sequence[EvaluationRecord],
    configurations: Sequence[Sequence[str]],
    *,
    seed: int,
    epoch: int,
    segments: Sequence[str],
    training_predictors: Mapping[tuple[str, str, str], tuple[float, float]],
) -> float:
    """Patient-balanced fold-8 monitor with a frozen rotating mask assignment.

    A record is evaluated under one deterministically assigned configuration.
    The assignment is fixed across epochs and distributed across the complete
    panel, so early-stopping scores are comparable while requiring one, rather
    than 64, neural forward passes per record.  The restored best model is
    subsequently scored on every configuration.
    """

    frames = []
    del epoch  # the validation mask assignment is intentionally frozen across epochs
    offset = (int(seed) * 17) % len(configurations)
    for index, record in enumerate(records):
        configuration = configurations[(index + offset) % len(configurations)]
        frames.append(
            evaluate_reconstructor(
                model,
                (record,),
                configuration=configuration,
                method="masked-unet",
                model_seed=int(seed),
                segments=segments,
                training_predictors=training_predictors,
                cohort="PTB-XL",
                partition="fold8/tune",
            )
        )
    return _patient_balanced_score(pd.concat(frames, ignore_index=True))


def _masked_unet_runtime(model: Any, scale: np.ndarray, device: Any) -> MaskedUNetReconstructor:
    runtime = MaskedUNetReconstructor()
    runtime.model = model
    runtime.scale = np.asarray(scale, dtype=np.float32)
    runtime.device = device
    runtime._fitted = True
    return runtime


def fit_masked_unet_candidate(
    candidate: Candidate,
    *,
    seed: int,
    train_manifest: TrainManifest,
    evaluation_records: Sequence[EvaluationRecord],
    configurations: Sequence[Sequence[str]],
    segments: Sequence[str],
    training_predictors: Mapping[tuple[str, str, str], tuple[float, float]],
    output_dir: Path,
    manifest_sha256: str,
    split_sha256: str,
    device_name: str,
    normalization_scale: np.ndarray | None = None,
) -> tuple[list[pd.DataFrame], dict[str, Any]]:
    """Train one U-Net seed, restore its patience-selected state, and score it."""

    import torch

    if candidate.method != "masked-unet":
        raise ValueError("fit_masked_unet_candidate requires a masked-unet candidate")
    # ``run`` performs the full multi-gigabyte file hash once before dispatching
    # candidates.  Structural checks remain local without hashing it nine more
    # times (three settings x three seeds).
    train_manifest.validate(verify_file=False)
    signals = np.load(train_manifest.signals_path, mmap_mode="r")
    if signals.ndim != 3 or signals.shape[1] != len(CANONICAL_LEADS):
        raise ValueError(f"masked-unet training signals must have shape (N,12,T): {signals.shape}")
    parameters = dict(candidate.parameters)
    max_records = int(signals.shape[0])
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    previous_determinism = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(bool(parameters["deterministic"]))
    device = torch.device(device_name)

    # Release normalization is fitted once on every folds-1--7 training record
    # and then shared by every candidate/seed.
    if normalization_scale is None:
        subset = np.asarray(signals[:max_records], dtype=np.float32)
        scale = np.percentile(np.abs(subset), 95, axis=(0, 2)).astype(np.float32)
        scale = np.clip(scale, 0.05, None)
        del subset
    else:
        scale = np.asarray(normalization_scale, dtype=np.float32)
        if scale.shape != (len(CANONICAL_LEADS),) or not np.isfinite(scale).all():
            raise ValueError("normalization_scale must be finite with shape (12,)")
        if np.any(scale < 0.05):
            raise ValueError("normalization_scale must be clipped at 0.05 mV")
    width = int(parameters["width"])
    network = _build_unet(width=width).to(device)
    optimizer = torch.optim.AdamW(
        network.parameters(),
        lr=float(parameters["learning_rate"]),
        weight_decay=float(parameters["weight_decay"]),
    )
    generator = torch.Generator().manual_seed(int(seed))
    panel = tuple(tuple(value) for value in deep_configuration_panel())
    lead_index = {lead: index for index, lead in enumerate(CANONICAL_LEADS)}

    class Dataset(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return max_records

        def __getitem__(self, index: int):
            return torch.as_tensor(np.array(signals[index], dtype=np.float32, copy=True))

    loader = torch.utils.data.DataLoader(
        Dataset(),
        batch_size=int(parameters["batch_size"]),
        shuffle=True,
        generator=generator,
        num_workers=int(parameters["num_workers"]),
        drop_last=False,
    )
    scale_tensor = torch.as_tensor(scale, dtype=torch.float32, device=device)[None, :, None]
    mask_rng = np.random.default_rng(int(seed))
    patience = int(parameters["early_stopping_patience"])
    max_epochs = int(parameters["max_epochs"])
    best_score = float("inf")
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    stale_epochs = 0
    history = []
    stopped_epoch = max_epochs
    runtime = _masked_unet_runtime(network, scale, device)
    try:
        for epoch in range(1, max_epochs + 1):
            network.train()
            training_loss_sum = 0.0
            training_batches = 0
            for target in loader:
                target = target.to(device=device, dtype=torch.float32) / scale_tensor
                batch_masks = np.zeros((target.shape[0], 12), dtype=np.float32)
                for row in range(target.shape[0]):
                    observed = panel[int(mask_rng.integers(0, len(panel)))]
                    batch_masks[row, [lead_index[lead] for lead in observed]] = 1.0
                observed_mask = torch.as_tensor(batch_masks, device=device)[:, :, None]
                observed_mask = observed_mask.expand(-1, -1, target.shape[-1])
                network_input = torch.cat([target * observed_mask, observed_mask], dim=1)
                prediction = network(network_input)
                missing = 1.0 - observed_mask
                loss = (((prediction - target) ** 2) * missing).sum() / missing.sum().clamp_min(1.0)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                training_loss_sum += float(loss.detach().cpu())
                training_batches += 1
            monitor = _early_stop_fold8_score(
                runtime,
                evaluation_records,
                configurations,
                seed=int(seed),
                epoch=epoch,
                segments=segments,
                training_predictors=training_predictors,
            )
            improved = bool(monitor < best_score)
            if improved:
                best_score = monitor
                best_epoch = epoch
                stale_epochs = 0
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in network.state_dict().items()
                }
            else:
                stale_epochs += 1
            history.append(
                {
                    "epoch": epoch,
                    "train_masked_mse": training_loss_sum / max(training_batches, 1),
                    "fold8_patient_balanced_log_rmse_mv": monitor,
                    "best_so_far_log_rmse_mv": best_score,
                    "stale_epochs": stale_epochs,
                    "is_strict_improvement": improved,
                }
            )
            if stale_epochs >= patience:
                stopped_epoch = epoch
                break
    finally:
        torch.use_deterministic_algorithms(previous_determinism)
    if best_state is None or best_epoch < 1:
        raise RuntimeError(f"{candidate.candidate_id}/seed-{seed} produced no best state")
    network.load_state_dict(best_state)
    checkpoint = (
        output_dir
        / "checkpoints"
        / "masked-unet"
        / candidate.candidate_id
        / f"seed-{seed}.pt"
    )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": network.state_dict(),
            "scale": scale,
            "seed": int(seed),
            "width": width,
            "train_manifest_sha256": train_manifest.signals_sha256,
            "panel_size": len(panel),
            "candidate_id": candidate.candidate_id,
            "candidate_parameters": parameters,
            "best_epoch": best_epoch,
            "stopped_epoch": stopped_epoch,
            "fold8_monitor_history": history,
        },
        checkpoint,
    )
    frames = []
    for configuration in configurations:
        metrics = evaluate_reconstructor(
            runtime,
            evaluation_records,
            configuration=configuration,
            method="masked-unet",
            model_seed=int(seed),
            segments=segments,
            training_predictors=training_predictors,
            cohort="PTB-XL",
            partition="fold8/tune",
        )
        frames.append(
            candidate_metric_rows(
                metrics,
                candidate=candidate,
                model_seed=int(seed),
                epoch=best_epoch,
                checkpoint=checkpoint,
                output_dir=output_dir,
                manifest_sha256=manifest_sha256,
                split_sha256=split_sha256,
            )
        )
    audit = {
        "candidate_id": candidate.candidate_id,
        "model_seed": int(seed),
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
        "best_monitor_log_rmse_mv": best_score,
        "monitor": (
            "fold8 patient-balanced log(RMSE_mV), one frozen distributed panel mask per record"
        ),
        "history": history,
        "checkpoint_path": checkpoint.resolve().relative_to(output_dir.resolve()).as_posix(),
        "checkpoint_sha256": lineage.artifact_sha256(checkpoint),
    }
    return frames, audit


def early_stop_trace_frame(
    audits: Sequence[Mapping[str, Any]],
    *,
    manifest_sha256: str,
    split_sha256: str,
    fold8_records_sha256: str,
    configuration_panel_sha256: str,
) -> pd.DataFrame:
    """Materialise the independently verifiable patience-8 training trace."""

    rows = []
    for audit in audits:
        for item in audit["history"]:
            rows.append(
                {
                    "schema_version": TRACE_SCHEMA_VERSION,
                    "cohort": "PTB-XL",
                    "train_partition": "folds1-7/train",
                    "partition": "fold8/tune",
                    "manifest_sha256": manifest_sha256,
                    "split_sha256": split_sha256,
                    "fold8_records_sha256": fold8_records_sha256,
                    "configuration_panel_sha256": configuration_panel_sha256,
                    "candidate_id": str(audit["candidate_id"]),
                    "model_seed": int(audit["model_seed"]),
                    "epoch": int(item["epoch"]),
                    "monitor_log_rmse_mv": float(
                        item["fold8_patient_balanced_log_rmse_mv"]
                    ),
                    "best_so_far_log_rmse_mv": float(
                        item["best_so_far_log_rmse_mv"]
                    ),
                    "stale_epochs": int(item["stale_epochs"]),
                    "is_strict_improvement": bool(item["is_strict_improvement"]),
                    "best_epoch": int(audit["best_epoch"]),
                    "stopped_epoch": int(audit["stopped_epoch"]),
                    "checkpoint_path": str(audit["checkpoint_path"]),
                    "checkpoint_sha256": str(audit["checkpoint_sha256"]),
                }
            )
    frame = pd.DataFrame(rows, columns=list(TRACE_COLUMNS))
    if frame.empty:
        raise ValueError("masked-unet early-stop trace is empty")
    return frame


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    validate_release_arguments(arguments)
    _require_parquet_engine()
    manifest = load_ptbxl_manifest(arguments.manifest)
    train_ids = manifest.record_ids("train", arguments.max_records)
    tune_ids = manifest.record_ids("tune", arguments.max_records)
    # Intentionally verify and materialise only training and fold-8 records.
    verification_ids = train_ids + tune_ids
    _verify_manifest_files(manifest, verification_ids, rate=arguments.rate)
    db = PTBXL(manifest.root)
    _validate_database_identity(db, manifest, verification_ids)
    configurations: tuple[tuple[str, ...], ...] = deep_configuration_panel()
    if arguments.max_configurations is not None:
        configurations = configurations[: arguments.max_configurations]
    output_dir = arguments.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluation_records, evaluation_audit = _load_evaluation_records(
        db,
        manifest,
        tune_ids,
        rate=arguments.rate,
        segments=arguments.segments,
        delineator=arguments.delineator,
        partition="fold8/tune",
    )

    frames: list[pd.DataFrame] = []
    unet_audit = []
    with TemporaryDirectory(prefix=".ecgcert-candidates-", dir=output_dir) as temporary:
        train_manifest, training_predictors, training_audit = _materialize_training_manifest(
            db,
            manifest,
            train_ids,
            rate=arguments.rate,
            work_dir=Path(temporary),
            segments=arguments.segments,
            delineator=arguments.delineator,
            configurations=configurations,
        )
        predictor_lookup = training_predictor_lookup(training_predictors)
        predictor_path = output_dir / TRAINING_PREDICTORS_FILENAME
        training_predictors.to_parquet(predictor_path, index=False, compression="zstd")
        mean, scatter, sample_count = _streaming_training_moments(train_manifest)
        training_signals = np.load(train_manifest.signals_path, mmap_mode="r")
        normalization_source = np.asarray(training_signals, dtype=np.float32)
        normalization_scale = np.percentile(
            np.abs(normalization_source), 95, axis=(0, 2)
        ).astype(np.float32)
        normalization_scale = np.clip(normalization_scale, 0.05, None)
        del normalization_source, training_signals
        for method in ("lowrank", "ridge"):
            frames.extend(
                evaluate_linear_candidate_grid(
                    method,
                    mean=mean,
                    scatter=scatter,
                    sample_count=sample_count,
                    evaluation_records=evaluation_records,
                    configurations=configurations,
                    segments=arguments.segments,
                    training_predictors=predictor_lookup,
                    output_dir=output_dir,
                    manifest_sha256=manifest.manifest_sha256,
                    split_sha256=manifest.split_sha256,
                )
            )
        for candidate in candidate_grid()["masked-unet"]:
            for seed in TUNING_SEEDS:
                candidate_frames, audit = fit_masked_unet_candidate(
                    candidate,
                    seed=seed,
                    train_manifest=train_manifest,
                    evaluation_records=evaluation_records,
                    configurations=configurations,
                    segments=arguments.segments,
                    training_predictors=predictor_lookup,
                    output_dir=output_dir,
                    manifest_sha256=manifest.manifest_sha256,
                    split_sha256=manifest.split_sha256,
                    device_name=arguments.device,
                    normalization_scale=normalization_scale,
                )
                frames.extend(candidate_frames)
                unet_audit.append(audit)
        training_signal_sha256 = train_manifest.signals_sha256

    metrics = pd.concat(frames, ignore_index=True)
    fold8_case_sha256 = evaluation_records_sha256(
        evaluation_records, segments=arguments.segments
    )
    panel_sha256 = configuration_panel_sha256(configurations)
    trace = early_stop_trace_frame(
        unet_audit,
        manifest_sha256=manifest.manifest_sha256,
        split_sha256=manifest.split_sha256,
        fold8_records_sha256=fold8_case_sha256,
        configuration_panel_sha256=panel_sha256,
    )
    _validate_candidate_metrics(
        metrics,
        manifest_sha256=manifest.manifest_sha256,
        split_sha256=manifest.split_sha256,
    )
    metrics_path = output_dir / CANDIDATE_METRICS_FILENAME
    metrics.to_parquet(metrics_path, index=False, compression="zstd")
    _verify_candidate_checkpoints(metrics, metrics_path)
    _validate_early_stop_trace(
        trace,
        metrics,
        manifest_sha256=manifest.manifest_sha256,
        split_sha256=manifest.split_sha256,
        fold8_records_sha256=fold8_case_sha256,
        configuration_panel_sha256=panel_sha256,
    )
    trace_path = output_dir / EARLY_STOP_TRACE_FILENAME
    trace.to_parquet(trace_path, index=False, compression="zstd")
    summary = {
        "schema_version": CANDIDATE_BUNDLE_SCHEMA_VERSION,
        "candidate_schema_version": CANDIDATE_SCHEMA_VERSION,
        "early_stop_trace_schema_version": TRACE_SCHEMA_VERSION,
        "status": "complete",
        "train_partition": "folds1-7/train",
        "evaluation_partition": "fold8/tune",
        "holdout_partitions_loaded": [],
        "manifest": {
            "path": str(arguments.manifest.resolve()),
            "sha256": manifest.manifest_sha256,
            "split_sha256": manifest.split_sha256,
        },
        "rate_hz": int(arguments.rate),
        "segments": list(arguments.segments),
        "delineator": arguments.delineator,
        "configuration_panel_sha256": panel_sha256,
        "n_configurations": len(configurations),
        "n_train_records": len(train_ids),
        "n_train_records_included": training_audit["summary"]["n_included"],
        "n_train_records_excluded": training_audit["summary"]["n_excluded"],
        "n_tune_records_requested": len(tune_ids),
        "train_signals_sha256": training_signal_sha256,
        "fold8_records_sha256": fold8_case_sha256,
        "evaluation_audit": evaluation_audit,
        "training_audit": training_audit,
        "candidate_grid": {
            method: [asdict(candidate) for candidate in candidates]
            for method, candidates in candidate_grid().items()
        },
        "masked_unet_early_stopping": unet_audit,
        "release": bool(arguments.release),
        "subsampled": arguments.max_records is not None
        or arguments.max_configurations is not None,
        "artifacts": {
            "candidate_metrics": {
                "path": CANDIDATE_METRICS_FILENAME,
                "sha256": lineage.artifact_sha256(metrics_path),
                "n_rows": int(len(metrics)),
            },
            "early_stop_trace": {
                "path": EARLY_STOP_TRACE_FILENAME,
                "sha256": lineage.artifact_sha256(trace_path),
                "n_rows": int(len(trace)),
            },
            "training_predictors": {
                "path": TRAINING_PREDICTORS_FILENAME,
                "sha256": lineage.artifact_sha256(predictor_path),
            },
        },
    }
    summary_path = output_dir / CANDIDATE_BUNDLE_FILENAME
    _atomic_json(summary_path, summary)
    summary["summary_sha256"] = lineage.artifact_sha256(summary_path)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        summary = run(arguments)
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"reconstruction candidate grid failed closed: {exc}") from exc
    print(
        f"[candidate-grid] {summary['artifacts']['candidate_metrics']['n_rows']} "
        f"real fold-8 rows -> {arguments.output_dir.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
