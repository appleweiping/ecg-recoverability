"""Thin, pinned adapters for published reconstruction implementations.

The upstream repositories are not vendored.  A checkout is accepted only when
its exact commit matches the declared specification; the commit is included in
every result lineage.  This keeps upstream scientific code intact and confines
project-specific behaviour to input/output conversion.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Iterator

import numpy as np

from ecgcert.estimators.api import (
    Reconstructor,
    ReconstructorConfig,
    TrainManifest,
    sha256_file,
)


@dataclass(frozen=True)
class UpstreamSpec:
    name: str
    repository: str
    commit: str
    paper: str
    root_tree: str | None = None
    license_spdx: str = "NOASSERTION"
    required_paths: tuple[str, ...] = ()


IMPUTE_ECG = UpstreamSpec(
    name="ImputeECG",
    repository="https://github.com/PKUDigitalHealth/ImputeECG.git",
    commit="70accf2f1600066392b14a5f50dbc131a6f13943",
    paper="arXiv:2607.05009",
    root_tree="d30565ea404a6b7f848fe3a9f5cc742655eb0388",
    required_paths=(
        "train.py",
        "inference.py",
        "datasets/ptbxl.py",
        "models/mae.py",
        "models/pos_embed.py",
        "utils/misc.py",
    ),
)
ECG_RECOVER = UpstreamSpec(
    name="ECGrecover",
    repository="https://git.ummisco.fr/open/2024-ecg-recover.git",
    commit="ed49dddf8e5e599b8af702e871a1f66b1d628518",
    paper="KDD 2025, DOI:10.1145/3690624.3709405",
    root_tree="980f872d6d25b1291942f4929d1417abec66fe1e",
    license_spdx="NOASSERTION",
    required_paths=(
        "main.py",
        "learn/Training.py",
        "tools/LoadModel.py",
        "tools/LossFunction.py",
        "tools/PreProcesing.py",
    ),
)


def validate_pinned_checkout(source_dir: str | Path, spec: UpstreamSpec) -> str:
    source = Path(source_dir).resolve()
    if not (source / ".git").exists():
        raise ValueError(f"{source} is not a git checkout")
    head = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != spec.commit:
        raise ValueError(f"{spec.name} checkout is {head}, expected {spec.commit}")
    origin = subprocess.run(
        ["git", "-C", str(source), "remote", "get-url", "origin"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if origin.rstrip("/") != spec.repository.rstrip("/"):
        raise ValueError(
            f"{spec.name} checkout origin is {origin!r}, expected {spec.repository!r}"
        )
    if spec.root_tree is not None:
        root_tree = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD^{tree}"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if root_tree != spec.root_tree:
            raise ValueError(
                f"{spec.name} root tree is {root_tree}, expected {spec.root_tree}"
            )
    missing_required = [
        relative
        for relative in spec.required_paths
        if not (source / relative).is_file()
    ]
    if missing_required:
        raise ValueError(
            f"{spec.name} checkout lacks required runtime paths: {missing_required}"
        )
    dirty = subprocess.run(
        ["git", "-C", str(source), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise ValueError(f"{spec.name} checkout is dirty")
    return head


@contextmanager
def _source_import_path(path: Path) -> Iterator[None]:
    value = str(path)
    previous_dont_write_bytecode = sys.dont_write_bytecode
    # Public research repositories occasionally track interpreter-specific
    # ``__pycache__`` files. Importing unchanged scientific source must not
    # mutate its pinned checkout or alter downstream lineage.
    sys.dont_write_bytecode = True
    sys.path.insert(0, value)
    try:
        yield
    finally:
        try:
            sys.path.remove(value)
        except ValueError:
            pass
        sys.dont_write_bytecode = previous_dont_write_bytecode


class ImputeECGReconstructor(Reconstructor):
    method_id = "imputeecg_official"
    upstream_commit = IMPUTE_ECG.commit
    preferred_batch_size = 64

    def __init__(self, source_dir: str | Path, checkpoint: str | Path | None = None):
        super().__init__()
        self.source_dir = Path(source_dir).resolve()
        self._checkpoint_path = Path(checkpoint).resolve() if checkpoint else None
        self.model = None
        self.device = None

    def build_train_command(self, train_manifest: TrainManifest, config: ReconstructorConfig) -> list[str]:
        """Build the unmodified official training entrypoint command."""

        official_data = config.parameters.get("official_data_path")
        if not official_data:
            raise ValueError(
                "ImputeECG requires official_data_path containing train_data_gt.npy and "
                "train_data_mask.npy generated from the locked manifest"
            )
        data_path = Path(str(official_data)).resolve()
        for name in ("train_data_gt.npy", "train_data_mask.npy"):
            if not (data_path / name).is_file():
                raise FileNotFoundError(data_path / name)
        output = Path(config.output_dir).resolve()
        gpu = str(config.parameters.get("gpu", 0))
        return [
            sys.executable,
            str(self.source_dir / "train.py"),
            "--data_path",
            str(data_path),
            "--output_dir",
            str(output),
            "--log_dir",
            str(output),
            "--gpu",
            gpu,
            "--epochs",
            str(int(config.parameters.get("epochs", 100))),
            "--batch_size",
            str(int(config.parameters.get("batch_size", 128))),
            "--mask_ratio",
            str(float(config.parameters.get("mask_ratio", 0.20))),
            "--missing_sentinel",
            "65535",
            "--seed",
            str(int(config.seed)),
            "--num_workers",
            str(int(config.parameters.get("num_workers", 8))),
        ]

    def fit(self, train_manifest: TrainManifest, config: ReconstructorConfig):
        train_manifest.validate()
        config.validate()
        validate_pinned_checkout(self.source_dir, IMPUTE_ECG)
        command = self.build_train_command(train_manifest, config)
        if bool(config.parameters.get("dry_run", False)):
            self._planned_command = command
            return self
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        subprocess.run(command, cwd=self.source_dir, check=True, env=environment)
        configured_epochs = int(config.parameters.get("epochs", 100))
        expected_checkpoint = Path(config.output_dir) / f"checkpoint-{configured_epochs}.pth"
        if not expected_checkpoint.is_file():
            raise RuntimeError(
                "official ImputeECG training did not produce the exact configured-epoch "
                f"checkpoint: {expected_checkpoint}"
            )
        self._retain_final_checkpoint_only(
            Path(config.output_dir),
            expected_checkpoint=expected_checkpoint,
            configured_epochs=configured_epochs,
        )
        self._checkpoint_path = expected_checkpoint
        self.load(config.device)
        return self

    @staticmethod
    def _retain_final_checkpoint_only(
        output_dir: Path,
        *,
        expected_checkpoint: Path,
        configured_epochs: int,
    ) -> None:
        """Discard reproducible optimizer-heavy intermediates after authentication.

        The pinned upstream writes a full model/optimizer/scaler checkpoint every
        ten epochs.  Five release seeds would otherwise consume most of the
        project-wide 100 GB artifact budget.  Training always runs in an isolated
        per-seed output directory; only files matching the upstream checkpoint
        naming contract are considered here, and the exact configured-epoch file
        is retained with a hash-bound audit record.
        """

        root = output_dir.resolve(strict=True)
        final = expected_checkpoint.resolve(strict=True)
        if final.parent != root or final.is_symlink():
            raise RuntimeError("ImputeECG final checkpoint escaped its isolated output directory")
        removed: list[dict[str, int | str]] = []
        for candidate in sorted(root.glob("checkpoint-*.pth")):
            resolved = candidate.resolve(strict=True)
            if resolved.parent != root or candidate.is_symlink():
                raise RuntimeError("unsafe ImputeECG checkpoint path in isolated output directory")
            if resolved == final:
                continue
            removed.append({"name": candidate.name, "size_bytes": candidate.stat().st_size})
            candidate.unlink()
        retained = sorted(path.name for path in root.glob("checkpoint-*.pth"))
        if retained != [final.name]:
            raise RuntimeError(
                "ImputeECG checkpoint retention failed closed; expected only "
                f"{final.name}, got {retained}"
            )
        audit = {
            "schema_version": "imputeecg-checkpoint-retention-v1",
            "configured_epochs": int(configured_epochs),
            "retained": {
                "name": final.name,
                "sha256": sha256_file(final),
                "size_bytes": final.stat().st_size,
            },
            "removed_reproducible_intermediates": removed,
            "removed_total_bytes": sum(int(item["size_bytes"]) for item in removed),
        }
        destination = root / "checkpoint_retention.v1.json"
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(
            json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)

    def load(self, device: str = "cuda:0") -> "ImputeECGReconstructor":
        validate_pinned_checkout(self.source_dir, IMPUTE_ECG)
        if self._checkpoint_path is None or not self._checkpoint_path.is_file():
            raise FileNotFoundError(self._checkpoint_path)
        import torch

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        with _source_import_path(self.source_dir):
            inference = importlib.import_module("inference")
            self.model = inference.load_model(str(self._checkpoint_path), self.device)
            self._impute = inference.impute_ecg
        self._fitted = True
        return self

    def _predict_missing(self, signal: np.ndarray, observed_mask: np.ndarray) -> np.ndarray:
        return self._predict_missing_batch(
            signal[None, ...], observed_mask[None, ...]
        )[0]

    def _predict_missing_batch(
        self, signals: np.ndarray, observed_masks: np.ndarray
    ) -> np.ndarray:
        if signals.shape[2] != 5000:
            raise ValueError("official ImputeECG expects a 10 s, 500 Hz (5000 sample) signal")
        if not np.all(observed_masks == observed_masks[:, :, :1]):
            raise ValueError("main benchmark uses whole-lead masks for ImputeECG")
        observed = signals.copy().astype(np.float32)
        observed[~observed_masks] = np.float32(65535.0)
        output = self._impute(self.model, observed, self.device, sentinel=65535.0)
        return output.detach().cpu().numpy()


class ECGrecoverReconstructor(Reconstructor):
    """Official single-input-lead adapter through an explicit pinned bridge.

    The 2024 repository has no stable Python package API.  A bridge command must
    therefore be supplied by the experiment manifest.  It receives ``.npz``
    input/output paths and runs from the pristine checkout; no architecture or
    loss changes are made silently.
    """

    method_id = "ecgrecover_official_single_lead"
    upstream_commit = ECG_RECOVER.commit

    def __init__(self, source_dir: str | Path, checkpoint: str | Path | None = None):
        super().__init__()
        self.source_dir = Path(source_dir).resolve()
        self._checkpoint_path = Path(checkpoint).resolve() if checkpoint else None
        self.bridge_command: tuple[str, ...] | None = None

    def fit(self, train_manifest: TrainManifest, config: ReconstructorConfig):
        train_manifest.validate()
        config.validate()
        validate_pinned_checkout(self.source_dir, ECG_RECOVER)
        if len(config.observed_leads) != 1:
            raise ValueError("ECGrecover is restricted to its published single-input-lead task")
        command = config.parameters.get("official_train_command")
        bridge = config.parameters.get("official_inference_bridge")
        if not isinstance(command, (list, tuple)) or not command:
            raise ValueError("official_train_command must be declared in the experiment manifest")
        if not isinstance(bridge, (list, tuple)) or not bridge:
            raise ValueError("official_inference_bridge must be declared in the experiment manifest")
        self.bridge_command = tuple(str(value) for value in bridge)
        if bool(config.parameters.get("dry_run", False)):
            self._planned_command = [str(value) for value in command]
            return self
        subprocess.run([str(value) for value in command], cwd=self.source_dir, check=True)
        checkpoint = config.parameters.get("checkpoint")
        if not checkpoint:
            raise ValueError("ECGrecover manifest must name the produced checkpoint")
        self._checkpoint_path = Path(str(checkpoint)).resolve()
        if not self._checkpoint_path.is_file():
            raise FileNotFoundError(self._checkpoint_path)
        self._fitted = True
        return self

    def _predict_missing(self, signal: np.ndarray, observed_mask: np.ndarray) -> np.ndarray:
        raise RuntimeError(
            "ECGrecover batch inference is executed by the experiment DAG bridge, not in-process; "
            "use the bridge's hashed output artifact"
        )

    @property
    def checkpoint_sha256(self) -> str:
        return super().checkpoint_sha256


__all__ = [
    "ECG_RECOVER",
    "IMPUTE_ECG",
    "ECGrecoverReconstructor",
    "ImputeECGReconstructor",
    "UpstreamSpec",
    "sha256_file",
    "validate_pinned_checkout",
]
