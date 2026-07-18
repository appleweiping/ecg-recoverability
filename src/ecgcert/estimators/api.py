"""Uniform reconstruction API and immutable training manifests."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ecgcert.data.common import CANONICAL_LEADS


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class TrainManifest:
    """Metadata and array location consumed by every reconstructor.

    ``signals_path`` is a NumPy array with shape ``(N,12,T)`` or ``(N,T,12)``.
    Raw arrays remain outside git; hashes make the manifest immutable.
    """

    dataset: str
    split: str
    signals_path: str
    signals_sha256: str
    split_sha256: str
    rate_hz: int
    lead_order: tuple[str, ...] = CANONICAL_LEADS
    normalization: str = "raw_mV"
    patient_ids_sha256: str = ""

    def validate(self, *, verify_file: bool = True) -> None:
        path = Path(self.signals_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        if self.rate_hz <= 0:
            raise ValueError("rate_hz must be positive")
        if tuple(self.lead_order) != CANONICAL_LEADS:
            raise ValueError("training arrays must use the canonical twelve-lead order")
        for name, value in {
            "signals_sha256": self.signals_sha256,
            "split_sha256": self.split_sha256,
            "patient_ids_sha256": self.patient_ids_sha256,
        }.items():
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"{name} must be a full SHA-256 digest")
        if verify_file and sha256_file(path) != self.signals_sha256:
            raise ValueError("training signal hash mismatch")


@dataclass(frozen=True)
class ReconstructorConfig:
    observed_leads: tuple[str, ...]
    seed: int
    output_dir: str
    device: str = "cpu"
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        unknown = set(self.observed_leads) - set(CANONICAL_LEADS)
        if unknown:
            raise ValueError(f"unknown observed leads: {sorted(unknown)}")
        if not self.observed_leads:
            raise ValueError("at least one observed lead is required")
        if len(set(self.observed_leads)) != len(self.observed_leads):
            raise ValueError("observed leads contain duplicates")


def load_manifest_signals(manifest: TrainManifest, *, mmap_mode: str | None = "r") -> np.ndarray:
    manifest.validate()
    signals = np.load(manifest.signals_path, mmap_mode=mmap_mode)
    if signals.ndim != 3:
        raise ValueError(f"expected a 3-D signal array, got {signals.shape}")
    if signals.shape[1] == 12:
        out = signals
    elif signals.shape[2] == 12:
        out = np.transpose(signals, (0, 2, 1))
    else:
        raise ValueError(f"no twelve-lead axis found in {signals.shape}")
    if not np.issubdtype(out.dtype, np.number):
        raise TypeError("training signals must be numeric")
    return out


def normalize_observed_mask(mask: np.ndarray | Sequence[bool], signal_shape: tuple[int, int]) -> np.ndarray:
    """Return a boolean ``(12,T)`` observed mask."""

    array = np.asarray(mask, dtype=bool)
    if array.shape == (12,):
        array = np.repeat(array[:, None], signal_shape[1], axis=1)
    if array.shape != signal_shape:
        raise ValueError(f"observed_mask shape {array.shape} does not match {signal_shape}")
    if not array.any(axis=0).all():
        raise ValueError("every time sample must contain at least one observed lead")
    return array


class Reconstructor(ABC):
    """Base class that enforces observed-sample preservation for every method."""

    method_id: str = "abstract"
    upstream_commit: str = "native"

    def __init__(self) -> None:
        self._checkpoint_path: Path | None = None
        self._fitted = False

    @abstractmethod
    def fit(self, train_manifest: TrainManifest, config: ReconstructorConfig) -> "Reconstructor":
        raise NotImplementedError

    @abstractmethod
    def _predict_missing(self, signal: np.ndarray, observed_mask: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def reconstruct(self, signal: np.ndarray, observed_mask: np.ndarray | Sequence[bool]) -> np.ndarray:
        """Reconstruct a ``(12,T)`` mV signal and copy observed samples exactly."""

        if not self._fitted:
            raise RuntimeError(f"{self.method_id} must be fitted or loaded before reconstruction")
        source = np.asarray(signal, dtype=float)
        if source.ndim != 2 or source.shape[0] != 12:
            raise ValueError(f"signal must have shape (12,T), got {source.shape}")
        mask = normalize_observed_mask(observed_mask, source.shape)
        predicted = np.asarray(self._predict_missing(source, mask), dtype=float)
        if predicted.shape != source.shape:
            raise ValueError(f"reconstructor returned {predicted.shape}, expected {source.shape}")
        if not np.all(np.isfinite(predicted)):
            raise ValueError("reconstructor produced non-finite values")
        return np.where(mask, source, predicted)

    @property
    def checkpoint_sha256(self) -> str:
        if self._checkpoint_path is None or not self._checkpoint_path.is_file():
            raise RuntimeError(f"{self.method_id} has no materialized checkpoint")
        return sha256_file(self._checkpoint_path)
