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


IMPUTE_ECG = UpstreamSpec(
    name="ImputeECG",
    repository="https://github.com/PKUDigitalHealth/ImputeECG.git",
    commit="70accf2f1600066392b14a5f50dbc131a6f13943",
    paper="arXiv:2607.05009",
)
ECG_RECOVER = UpstreamSpec(
    name="ECGrecover",
    repository="https://git.ummisco.fr/open/2024-ecg-recover.git",
    commit="ed49dddf8e5e599b8af702e871a1f66b1d628518",
    paper="KDD 2025, DOI:10.1145/3690624.3709405",
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
    sys.path.insert(0, value)
    try:
        yield
    finally:
        try:
            sys.path.remove(value)
        except ValueError:
            pass


class ImputeECGReconstructor(Reconstructor):
    method_id = "imputeecg_official"
    upstream_commit = IMPUTE_ECG.commit

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
        subprocess.run(command, cwd=self.source_dir, check=True)
        checkpoints = sorted(Path(config.output_dir).glob("checkpoint-*.pth"))
        if not checkpoints:
            raise RuntimeError("official ImputeECG training produced no checkpoint")
        self._checkpoint_path = checkpoints[-1]
        self.load(config.device)
        return self

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
        if signal.shape[1] != 5000:
            raise ValueError("official ImputeECG expects a 10 s, 500 Hz (5000 sample) signal")
        if not np.all(observed_mask == observed_mask[:, :1]):
            raise ValueError("main benchmark uses whole-lead masks for ImputeECG")
        observed = signal.copy().astype(np.float32)
        observed[~observed_mask] = np.float32(65535.0)
        output = self._impute(self.model, observed[None], self.device, sentinel=65535.0)
        return output.detach().cpu().numpy()[0]


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
