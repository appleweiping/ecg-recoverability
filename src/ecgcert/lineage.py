"""Result lineage / integrity metadata (P0-6).

Every result JSON embeds a ``lineage`` block so a number can be traced to the exact
code + data + protocol that produced it, and so independently-produced artifacts (e.g.
the U-Net JSON and the linear-baseline JSON that ``fair_baselines`` merges) can be
*asserted* to share the same train/test split, targets, seed, segment definition and
normalization before they are combined.

Fields:
  commit       git commit SHA of the code (env ``ECG_COMMIT`` overrides; else git; else "unknown")
  dataset      dataset fingerprint (name + record count + sha256 of the sorted ecg_id list)
  protocol     protocol version string (bump when the evaluation protocol changes)
  seed         RNG seed
  targets      target leads
  segment_def  how P/QRS/ST/T indices are derived
  normalization how signals are scaled before modeling
  train_ids_sha256 / test_ids_sha256   sha256 of the sorted int ID lists
  n_train / n_test
"""
from __future__ import annotations

import hashlib
import os
import subprocess
from typing import Iterable

import numpy as np

PROTOCOL_VERSION = "recoverability-v2-2026-07"
SEGMENT_DEF = "neurokit2-dwt on observed Lead II; P=Pon..Poff, QRS=Ron..Roff, ST=J..Ton, T=Ton..Toff"


def _commit() -> str:
    env = os.environ.get("ECG_COMMIT")
    if env:
        return env.strip()
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:
        pass
    return "unknown"


def ids_sha256(ids: Iterable) -> str:
    arr = np.asarray(sorted(int(x) for x in ids), dtype=np.int64)
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def dataset_fingerprint(db) -> dict:
    """Fingerprint the loaded PTB-XL: name, record count, sha256 of all sorted ecg_ids."""
    try:
        ids = np.asarray(sorted(int(i) for i in db.meta.index), dtype=np.int64)
        return {"name": "PTB-XL", "n_records": int(ids.size),
                "ids_sha256": hashlib.sha256(ids.tobytes()).hexdigest()[:16]}
    except Exception:
        return {"name": "PTB-XL", "n_records": None, "ids_sha256": "unknown"}


def make(db=None, *, seed, targets, normalization, train_ids=None, test_ids=None,
         protocol: str = PROTOCOL_VERSION, extra: dict | None = None) -> dict:
    """Build a lineage block for a result JSON."""
    lin = {
        "commit": _commit(),
        "protocol": protocol,
        "seed": int(seed),
        "targets": list(targets),
        "segment_def": SEGMENT_DEF,
        "normalization": normalization,
        "dataset": dataset_fingerprint(db) if db is not None else None,
        "n_train": (int(len(list(train_ids))) if train_ids is not None else None),
        "n_test": (int(len(list(test_ids))) if test_ids is not None else None),
        "train_ids_sha256": (ids_sha256(train_ids) if train_ids is not None else None),
        "test_ids_sha256": (ids_sha256(test_ids) if test_ids is not None else None),
    }
    if extra:
        lin.update(extra)
    return lin


# Keys that MUST agree between two artifacts before they may be combined.
_MERGE_KEYS = ("dataset", "protocol", "seed", "targets", "segment_def", "normalization",
               "train_ids_sha256", "test_ids_sha256")


def assert_consistent(a: dict, b: dict, keys: Iterable[str] = _MERGE_KEYS,
                      label_a: str = "A", label_b: str = "B") -> None:
    """Raise ValueError if lineage blocks ``a`` and ``b`` disagree on ``keys``.

    Used by fair_baselines before merging the independently-produced U-Net JSON: a
    silent train/test-split or normalization mismatch would make the merged table a lie.
    """
    diffs = []
    for k in keys:
        if a.get(k) != b.get(k):
            diffs.append(f"  {k}: {label_a}={a.get(k)!r} != {label_b}={b.get(k)!r}")
    if diffs:
        raise ValueError(
            f"lineage mismatch between {label_a} and {label_b}; refusing to merge:\n"
            + "\n".join(diffs))
