"""Audited ECGrecover bridge for the frozen single-lead benchmark.

The upstream repository is imported from an exact, clean checkout.  This file
does not copy or alter its U-Net, hybrid MSE/Pearson loss, or training loop.
It only translates the project's raw-mV arrays to the upstream 512-sample
tensor contract and translates predictions back to raw mV.

Inference accepts either one ``(12,T)`` record or a homogeneous
``(N,12,T)`` batch.  The official model and checkpoint are loaded once per
bridge invocation, then the batch is evaluated in configurable GPU
micro-batches.  Per-record deterministic masking is unchanged by batching.

The published code normalizes every target lead with that record's target
minimum and maximum before masking.  That would leak missing-target amplitude
at inference.  The bridge therefore uses one fixed per-lead scale estimated
from folds 1--7 only.  This disclosed normalization adapter is necessary for a
truth-free raw-mV evaluation and is deliberately kept outside the upstream
checkout.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from tempfile import TemporaryDirectory
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
from scipy import signal as scipy_signal


UPSTREAM_COMMIT = "ed49dddf8e5e599b8af702e871a1f66b1d628518"
BRIDGE_SCHEMA_VERSION = "ecgrecover-bridge-v1"
CANONICAL_LEADS = (
    "I",
    "II",
    "III",
    "aVR",
    "aVL",
    "aVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
)
MODEL_SAMPLES = 512
SCALE_QUANTILE = 0.995
SCALE_RECORD_LIMIT = 2048
SCALE_TIME_STRIDE = 5
INTERNAL_VALIDATION_SALT = "ecgrecover-internal-validation-v1"


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _resolve_below(
    root: Path,
    relative: str,
    *,
    label: str,
    boundary: Path | None = None,
) -> Path:
    candidate = (root / relative).resolve()
    allowed = root.resolve() if boundary is None else boundary.resolve()
    try:
        candidate.relative_to(allowed)
    except ValueError as exc:
        raise ValueError(f"{label} escapes its declared root") from exc
    return candidate


def _load_dataset(data_dir: str | Path) -> tuple[np.ndarray, list[str], str]:
    root = Path(data_dir).resolve()
    descriptor_path = root / "dataset.v3.json"
    value = json.loads(descriptor_path.read_text(encoding="utf-8"))
    if value.get("task") != "official-single-input-lead":
        raise ValueError("ECGrecover dataset is not the frozen single-input task")
    if tuple(value.get("lead_order", ())) != CANONICAL_LEADS:
        raise ValueError("ECGrecover dataset has the wrong canonical lead order")
    input_lead = str(value.get("input_lead", ""))
    if input_lead not in CANONICAL_LEADS:
        raise ValueError("ECGrecover dataset has an invalid input lead")
    ground_truth = value.get("ground_truth")
    record_order = value.get("record_order")
    if not isinstance(ground_truth, Mapping) or not isinstance(record_order, Mapping):
        raise ValueError("ECGrecover dataset lacks ground truth or record order")
    truth_path = _resolve_below(
        root,
        str(ground_truth.get("path", "")),
        label="ground truth",
        boundary=root.parent,
    )
    records_path = _resolve_below(
        root,
        str(record_order.get("path", "")),
        label="record order",
        boundary=root.parent,
    )
    if _sha256_file(truth_path) != ground_truth.get("sha256"):
        raise ValueError("ECGrecover ground-truth SHA-256 mismatch")
    if _sha256_file(records_path) != record_order.get("sha256"):
        raise ValueError("ECGrecover record-order SHA-256 mismatch")
    truth = np.load(truth_path, mmap_mode="r", allow_pickle=False)
    if truth.ndim != 3 or truth.shape[1:] != (5000, len(CANONICAL_LEADS)):
        raise ValueError(f"ECGrecover ground truth has invalid shape {truth.shape}")
    records = json.loads(records_path.read_text(encoding="utf-8")).get("records")
    if not isinstance(records, list) or len(records) != truth.shape[0]:
        raise ValueError("ECGrecover record order does not align with ground truth")
    patient_ids = [str(item.get("patient_id", "")) for item in records]
    if any(not patient_id for patient_id in patient_ids):
        raise ValueError("ECGrecover record order contains an empty patient id")
    return truth, patient_ids, input_lead


def _patient_split(patient_ids: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    validation_patients = {
        patient_id
        for patient_id in set(patient_ids)
        if int.from_bytes(
            hashlib.sha256(f"{INTERNAL_VALIDATION_SALT}:{patient_id}".encode()).digest()[:8],
            "big",
        )
        % 10
        == 0
    }
    if not validation_patients:
        ordered = sorted(set(patient_ids))
        if len(ordered) < 2:
            raise ValueError("ECGrecover training requires at least two patients")
        validation_patients = {ordered[-1]}
    train = np.asarray(
        [index for index, patient_id in enumerate(patient_ids) if patient_id not in validation_patients],
        dtype=np.int64,
    )
    validation = np.asarray(
        [index for index, patient_id in enumerate(patient_ids) if patient_id in validation_patients],
        dtype=np.int64,
    )
    if not train.size or not validation.size:
        raise ValueError("ECGrecover patient-grouped internal split is empty")
    return train, validation


def _fixed_training_scale(truth: np.ndarray, train_indices: np.ndarray) -> np.ndarray:
    selected = train_indices[: min(SCALE_RECORD_LIMIT, train_indices.size)]
    sample = np.asarray(truth[selected, ::SCALE_TIME_STRIDE, :], dtype=np.float32)
    scale = np.quantile(np.abs(sample), SCALE_QUANTILE, axis=(0, 1)).astype(np.float32)
    if scale.shape != (len(CANONICAL_LEADS),) or not np.isfinite(scale).all():
        raise ValueError("ECGrecover training scale is invalid")
    return np.clip(scale, np.float32(0.05), None)


def _materialize_masked_split(
    truth: np.ndarray,
    indices: np.ndarray,
    *,
    scale: np.ndarray,
    input_index: int,
    seed: int,
    input_path: Path,
    target_path: Path,
    chunk_size: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    shape = (indices.size, MODEL_SAMPLES, len(CANONICAL_LEADS))
    inputs = np.lib.format.open_memmap(input_path, mode="w+", dtype=np.float32, shape=shape)
    targets = np.lib.format.open_memmap(target_path, mode="w+", dtype=np.float32, shape=shape)
    rng = np.random.default_rng(seed)
    for start in range(0, indices.size, chunk_size):
        stop = min(start + chunk_size, indices.size)
        raw = np.asarray(truth[indices[start:stop]], dtype=np.float32)
        normalized = raw / scale[None, None, :]
        resampled = scipy_signal.resample(normalized, MODEL_SAMPLES, axis=1).astype(np.float32)
        np.clip(resampled, -1.0, 1.0, out=resampled)
        masked = rng.random(resampled.shape, dtype=np.float32)
        masked[:, :, input_index] = resampled[:, :, input_index]
        inputs[start:stop] = masked
        targets[start:stop] = resampled
    inputs.flush()
    targets.flush()
    return inputs, targets


def _close_memmaps(*arrays: np.ndarray) -> None:
    """Release Windows file handles before TemporaryDirectory cleanup."""

    for array in arrays:
        array.flush()
        memory_map = getattr(array, "_mmap", None)
        if memory_map is not None:
            memory_map.close()


@contextmanager
def _upstream_imports(source_dir: str | Path) -> Iterator[tuple[Any, Any, Any]]:
    source = Path(source_dir).resolve()
    required = (
        source / "learn" / "Training.py",
        source / "tools" / "LoadModel.py",
        source / "tools" / "LossFunction.py",
    )
    if not all(path.is_file() for path in required):
        raise FileNotFoundError("ECGrecover checkout lacks its pinned official source files")
    value = str(source)
    previous_dont_write_bytecode = sys.dont_write_bytecode
    # The official repository has historically tracked interpreter-specific
    # bytecode.  Importing its frozen source must never mutate the checkout used
    # by the evidence DAG or make a later clean-tree check fail.
    sys.dont_write_bytecode = True
    sys.path.insert(0, value)
    try:
        from learn.Training import training
        from tools.LoadModel import load_model
        from tools.LossFunction import loss_function

        yield load_model, loss_function, training
    finally:
        try:
            sys.path.remove(value)
        except ValueError:
            pass
        sys.dont_write_bytecode = previous_dont_write_bytecode


def _seed_everything(seed: int) -> Any:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return torch


def _resolve_device(torch: Any, requested: str) -> Any:
    if requested == "auto":
        requested = "cuda:0" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"requested {requested}, but CUDA is unavailable")
    return torch.device(requested)


def train(arguments: argparse.Namespace) -> None:
    output = Path(arguments.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "Modeltemp.pth"
    if checkpoint.exists():
        raise FileExistsError(f"refusing to overwrite ECGrecover checkpoint: {checkpoint}")
    truth, patient_ids, input_lead = _load_dataset(arguments.data_dir)
    if arguments.input_lead != input_lead:
        raise ValueError("bridge input lead does not match the audited dataset")
    train_indices, validation_indices = _patient_split(patient_ids)
    scale = _fixed_training_scale(truth, train_indices)
    input_index = CANONICAL_LEADS.index(input_lead)
    torch = _seed_everything(arguments.seed)
    device = _resolve_device(torch, arguments.device)
    with TemporaryDirectory(prefix=".ecgrecover-adapter-", dir=output) as temporary:
        temporary_root = Path(temporary)
        train_input, train_target = _materialize_masked_split(
            truth,
            train_indices,
            scale=scale,
            input_index=input_index,
            seed=arguments.seed,
            input_path=temporary_root / "train-input.npy",
            target_path=temporary_root / "train-target.npy",
        )
        validation_input, validation_target = _materialize_masked_split(
            truth,
            validation_indices,
            scale=scale,
            input_index=input_index,
            seed=arguments.seed + 1_000_003,
            input_path=temporary_root / "validation-input.npy",
            target_path=temporary_root / "validation-target.npy",
        )
        try:
            with _upstream_imports(arguments.source_dir) as (load_model, loss_function, training):
                model = load_model().to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
                scheduler_kwargs = {"mode": "min", "factor": 0.1, "patience": 5}
                try:
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, verbose=True, **scheduler_kwargs
                    )
                except TypeError:  # PyTorch >= 2.6 removed the deprecated verbose argument.
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, **scheduler_kwargs
                    )
                loss = loss_function()
                training(
                    train_input,
                    train_target,
                    validation_input,
                    validation_target,
                    arguments.epochs,
                    arguments.batch_size,
                    model,
                    device,
                    optimizer,
                    scheduler,
                    loss,
                    str(output) + os.sep,
                )
        finally:
            _close_memmaps(train_input, train_target, validation_input, validation_target)
    if not checkpoint.is_file():
        raise RuntimeError("official ECGrecover training loop produced no Modeltemp.pth")
    metadata = {
        "schema_version": BRIDGE_SCHEMA_VERSION,
        "upstream_commit": UPSTREAM_COMMIT,
        "checkpoint_sha256": _sha256_file(checkpoint),
        "model_seed": arguments.seed,
        "input_lead": input_lead,
        "lead_order": list(CANONICAL_LEADS),
        "model_samples": MODEL_SAMPLES,
        "training_records": int(train_indices.size),
        "internal_validation_records": int(validation_indices.size),
        "internal_validation": "patient-grouped SHA-256 10% split within folds 1-7",
        "internal_validation_salt": INTERNAL_VALIDATION_SALT,
        "training_patient_ids_sha256": hashlib.sha256(
            json.dumps(
                [patient_ids[index] for index in train_indices],
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
        "internal_validation_patient_ids_sha256": hashlib.sha256(
            json.dumps(
                [patient_ids[index] for index in validation_indices],
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
        "scale_mv": scale.astype(float).tolist(),
        "scale_estimator": {
            "source": "folds 1-7 training records only",
            "absolute_quantile": SCALE_QUANTILE,
            "record_limit": SCALE_RECORD_LIMIT,
            "time_stride": SCALE_TIME_STRIDE,
            "minimum_mv": 0.05,
        },
        "adapter_disclosure": (
            "fixed folds-1-7 per-lead scaling replaces upstream per-record target min-max "
            "normalization to prevent missing-target amplitude leakage and permit raw-mV scoring"
        ),
        "inference_protocol": (
            "truth-free scalar-or-batch NPZ; one checkpoint load per invocation; "
            "configurable device micro-batches"
        ),
        "architecture": "unmodified tools.LoadModel.load_model",
        "loss": "unmodified tools.LossFunction.loss_function (MSE - 0.1 Pearson)",
        "training_loop": "unmodified learn.Training.training",
        "epochs": arguments.epochs,
        "batch_size": arguments.batch_size,
        "optimizer": "Adam(lr=0.01)",
    }
    _atomic_json(output / "bridge_metadata.v1.json", metadata)


def _load_inference_input(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Load a truth-free scalar or homogeneous batch bridge payload.

    The boolean return value records whether the caller supplied the legacy
    scalar shape so the output can retain that backward-compatible shape.
    """

    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != {"observed_signal", "observed_mask", "lead_order"}:
            raise ValueError("bridge input has an unexpected field")
        observed = np.asarray(payload["observed_signal"], dtype=np.float64)
        mask = np.asarray(payload["observed_mask"], dtype=bool)
        lead_order = tuple(str(value) for value in payload["lead_order"].tolist())
    scalar_input = observed.ndim == 2
    if scalar_input:
        observed = observed[None, ...]
        mask = mask[None, ...]
    if observed.ndim != 3 or observed.shape[0] < 1 or observed.shape[1] != len(CANONICAL_LEADS):
        raise ValueError("bridge input signal must have shape (12,T) or (N,12,T)")
    if mask.shape != observed.shape or not np.isfinite(observed).all():
        raise ValueError("bridge input mask/values are invalid")
    if np.any(observed[~mask] != 0.0):
        raise ValueError("bridge input contains values for unobserved samples")
    if lead_order != CANONICAL_LEADS:
        raise ValueError("bridge input has the wrong canonical lead order")
    whole_leads = np.all(mask == mask[:, :, :1], axis=(1, 2))
    observed_counts = mask[:, :, 0].sum(axis=1)
    if not np.all(whole_leads) or not np.all(observed_counts == 1):
        raise ValueError("ECGrecover bridge accepts exactly one whole observed lead")
    return observed, mask, scalar_input


def _restore_prediction(
    prediction_512: np.ndarray,
    *,
    length: int,
    scale: np.ndarray,
    observed: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    prediction = scipy_signal.resample(prediction_512, length, axis=1)
    prediction = np.asarray(prediction * scale[:, None], dtype=np.float64)
    prediction[mask] = observed[mask]
    if prediction.shape != observed.shape or not np.isfinite(prediction).all():
        raise ValueError("ECGrecover bridge produced an invalid reconstruction")
    return prediction


def _network_input_for_record(
    observed: np.ndarray,
    *,
    input_index: int,
    scale: np.ndarray,
    model_seed: int,
) -> np.ndarray:
    """Create the exact deterministic masked tensor used by scalar inference."""

    seed_material = np.ascontiguousarray(observed[input_index]).view(np.uint8)
    digest = hashlib.sha256(seed_material).digest()
    noise_seed = int.from_bytes(digest[:8], "big") ^ int(model_seed)
    rng = np.random.default_rng(noise_seed)
    network_input = rng.random(
        (MODEL_SAMPLES, len(CANONICAL_LEADS)), dtype=np.float32
    )
    normalized_observed = observed[input_index] / float(scale[input_index])
    resampled_observed = scipy_signal.resample(normalized_observed, MODEL_SAMPLES)
    network_input[:, input_index] = np.clip(resampled_observed, -1.0, 1.0)
    return network_input


def infer(arguments: argparse.Namespace) -> None:
    checkpoint = Path(arguments.checkpoint).resolve()
    metadata_path = checkpoint.with_name("bridge_metadata.v1.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != BRIDGE_SCHEMA_VERSION:
        raise ValueError("ECGrecover checkpoint has no audited bridge metadata")
    if metadata.get("upstream_commit") != UPSTREAM_COMMIT:
        raise ValueError("ECGrecover checkpoint metadata has the wrong upstream commit")
    if metadata.get("checkpoint_sha256") != _sha256_file(checkpoint):
        raise ValueError("ECGrecover checkpoint SHA-256 mismatch")
    if tuple(metadata.get("lead_order", ())) != CANONICAL_LEADS:
        raise ValueError("ECGrecover checkpoint metadata has the wrong lead order")
    scale = np.asarray(metadata.get("scale_mv"), dtype=np.float32)
    if scale.shape != (len(CANONICAL_LEADS),) or not np.isfinite(scale).all():
        raise ValueError("ECGrecover checkpoint metadata has an invalid scale")
    observed, mask, scalar_input = _load_inference_input(arguments.input)
    input_indices = np.argmax(mask[:, :, 0], axis=1)
    if any(
        CANONICAL_LEADS[int(input_index)] != metadata.get("input_lead")
        for input_index in input_indices
    ):
        raise ValueError("ECGrecover checkpoint is restricted to its audited input lead")
    network_input = np.stack(
        [
            _network_input_for_record(
                record,
                input_index=int(input_index),
                scale=scale,
                model_seed=int(metadata["model_seed"]),
            )
            for record, input_index in zip(observed, input_indices, strict=True)
        ]
    )
    torch = _seed_everything(int(metadata["model_seed"]))
    device = _resolve_device(torch, arguments.device)
    with _upstream_imports(arguments.source_dir) as (load_model, _loss, _training):
        model = load_model().to(device)
        try:
            state = torch.load(checkpoint, map_location=device, weights_only=True)
        except TypeError:  # Official PyTorch 2.2 environment.
            state = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state, strict=True)
        model.eval()
        prediction_chunks = []
        for start in range(0, observed.shape[0], arguments.micro_batch_size):
            stop = min(start + arguments.micro_batch_size, observed.shape[0])
            value = torch.as_tensor(network_input[start:stop], device=device)
            value = torch.transpose(torch.unsqueeze(value, 1), 2, 3)
            with torch.no_grad():
                prediction_chunks.append(model(value, device).detach().cpu().numpy())
        prediction_512 = np.concatenate(prediction_chunks, axis=0)
    reconstruction = np.stack(
        [
            _restore_prediction(
                prediction,
                length=record.shape[1],
                scale=scale,
                observed=record,
                mask=record_mask,
            )
            for prediction, record, record_mask in zip(
                prediction_512, observed, mask, strict=True
            )
        ]
    )
    if scalar_input:
        reconstruction = reconstruction[0]
    output = Path(arguments.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as handle:
        # Compression is intentionally avoided: these files are ephemeral and
        # NPZ deflate time otherwise dominates high-throughput inference.
        np.savez(handle, reconstruction=reconstruction)
    os.replace(temporary, output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--source-dir", type=Path, required=True)
    train_parser.add_argument("--data-dir", type=Path, required=True)
    train_parser.add_argument("--seed", type=int, required=True)
    train_parser.add_argument("--output-dir", type=Path, required=True)
    train_parser.add_argument("--input-lead", choices=CANONICAL_LEADS, default="I")
    train_parser.add_argument("--device", default="cuda:0")
    train_parser.add_argument("--epochs", type=int, default=100)
    train_parser.add_argument("--batch-size", type=int, default=256)
    inference_parser = subparsers.add_parser("infer")
    inference_parser.add_argument("--source-dir", type=Path, required=True)
    inference_parser.add_argument("--input", type=Path, required=True)
    inference_parser.add_argument("--output", type=Path, required=True)
    inference_parser.add_argument("--checkpoint", type=Path, required=True)
    inference_parser.add_argument("--device", default="cuda:0")
    inference_parser.add_argument("--micro-batch-size", type=int, default=64)
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    if arguments.command == "train":
        if arguments.seed < 0 or arguments.epochs < 1 or arguments.batch_size < 1:
            raise ValueError("seed/epochs/batch-size are invalid")
        train(arguments)
    else:
        if arguments.micro_batch_size < 1:
            raise ValueError("micro-batch-size must be positive")
        infer(arguments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
