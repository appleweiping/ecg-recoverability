"""Compatibility helpers for the locked ICASSP 2027 protocol.

New experiments should import constants and split/configuration builders from
``ecgcert.protocol``.  The legacy helpers below remain so historical experiments
can still be reproduced, but their NORM-only 100 Hz protocol is not a main-paper
profile.
"""
from __future__ import annotations

import numpy as np

from ecgcert.protocol import (  # re-export the canonical protocol
    BOOTSTRAP_REPLICATES,
    CONFIG_PANEL_SALT,
    PRIMARY_RATE_HZ,
    PRIMARY_SEGMENTS,
    RANK_GRID,
    StudyProtocol,
    all_independent_configurations,
    configuration_panel_sha256,
    deep_configuration_panel,
    patient_hash_split,
    ptbxl_split,
)

RATE = 100
WINDOW = 1000               # samples (10 s at 100 Hz)
NORMALIZATION = "per-lead 95th-percentile |amp| from train, clipped>=0.05 mV"

__all__ = [
    "BOOTSTRAP_REPLICATES",
    "CONFIG_PANEL_SALT",
    "PRIMARY_RATE_HZ",
    "PRIMARY_SEGMENTS",
    "RANK_GRID",
    "StudyProtocol",
    "all_independent_configurations",
    "configuration_panel_sha256",
    "deep_configuration_panel",
    "patient_hash_split",
    "ptbxl_split",
    "standard_split",
    "fold8_ids",
    "load_windows",
]


def standard_split(db, n_train, n_test, seed=0):
    """Deterministic (train_ids, test_ids). NORM folds 1-7 train, NORM fold-10 test;
    fold 8 is reserved for hyperparameter selection (see :func:`fold8_ids`) so it is
    disjoint from training -- selecting ridge lambda on fold 8 is then leakage-free.

    Single RNG, fixed draw order, so any caller with the same (n_train, n_test, seed)
    gets identical arrays.
    """
    rng = np.random.default_rng(seed)
    tr = rng.permutation(db.ids_with_superclass("NORM", exclusive=False, folds=range(1, 8)))
    f10 = db.meta[db.meta["strat_fold"] == 10].index.to_numpy()
    norm10 = db.ids_with_superclass("NORM", exclusive=False, folds=[10])
    test = rng.permutation(np.intersect1d(f10, norm10))[:n_test]
    return tr[:n_train], test


def fold8_ids(db, cap=None, seed=0):
    """NORM fold-8 records for hyperparameter selection (disjoint from train 1-7... 1-8)."""
    rng = np.random.default_rng(seed + 8)
    ids = rng.permutation(db.ids_with_superclass("NORM", exclusive=False, folds=[8]))
    return ids[:cap] if cap else ids


def load_windows(db, ids, scale=None, rate=RATE, window=WINDOW):
    """Load (N,12,window) records + kept ids + per-lead scale. Deterministic filtering,
    so two callers over the same ordered ids get identical (X, kept)."""
    sigs, kept = [], []
    for eid in list(ids):
        try:
            s = db.signal(int(eid), rate=rate)
        except Exception:
            continue
        if s.shape[0] < window or not np.all(np.isfinite(s[:window])):
            continue
        sigs.append(s[:window].T.astype(np.float32))         # (12, window)
        kept.append(int(eid))
    X = np.stack(sigs)                                        # (N,12,window)
    if scale is None:
        scale = np.percentile(np.abs(X), 95, axis=(0, 2)).astype(np.float32)
        scale = np.clip(scale, 0.05, None)
    return X, np.asarray(kept, dtype=np.int64), scale
