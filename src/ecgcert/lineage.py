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


def file_sha256(path) -> str:
    """sha256 of a file's bytes (16 hex chars); 'missing' if absent."""
    p = os.fspath(path)
    if not os.path.exists(p):
        return "missing"
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


_GEN_DIRS = ("results/", "paper/auto/")
_GEN_EXT = (".pdf", ".png", ".aux", ".log", ".out", ".synctex.gz", ".bbl", ".blg", ".toc")


def _git_dirty() -> bool:
    """True iff there are uncommitted CODE changes.

    ``git_dirty`` records whether the artifact was produced from committed code. It therefore
    IGNORES the generated outputs a run necessarily writes (``results/``, emitted macros under
    ``paper/auto/``, built PDFs/figures, LaTeX aux); otherwise the first experiment's own output
    would flag every subsequent experiment in the same clean-tree run as dirty. A genuine
    uncommitted change to any tracked source file still returns True.
    """
    try:
        out = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, timeout=5)
        for line in out.stdout.splitlines():
            if not line.strip():
                continue
            path = line[3:].strip().strip('"')
            if " -> " in path:                       # rename: take the destination
                path = path.split(" -> ")[-1]
            if path.startswith(_GEN_DIRS) or path.endswith(_GEN_EXT):
                continue                             # generated artifact, not code
            return True
        return False
    except Exception:
        return False


def _protocol_sha() -> str:
    return hashlib.sha256((PROTOCOL_VERSION + "|" + SEGMENT_DEF).encode()).hexdigest()[:12]


def dataset_fingerprint(db) -> dict:
    """Fingerprint the loaded PTB-XL: name, record count, sha256 of sorted ecg_ids, and a
    hash of the metadata CSV (dataset version/content)."""
    try:
        ids = np.asarray(sorted(int(i) for i in db.meta.index), dtype=np.int64)
        meta_sha = "unknown"
        try:
            meta_sha = file_sha256(db.root / "ptbxl_database.csv")
        except Exception:
            pass
        return {"name": "PTB-XL", "n_records": int(ids.size),
                "ids_sha256": hashlib.sha256(ids.tobytes()).hexdigest()[:16],
                "metadata_sha256": meta_sha}
    except Exception:
        return {"name": "PTB-XL", "n_records": None, "ids_sha256": "unknown"}


def make(db=None, *, seed, targets, normalization, train_ids=None, test_ids=None,
         protocol: str = PROTOCOL_VERSION, script: str | None = None,
         checkpoint: str | None = None, extra: dict | None = None) -> dict:
    """Build a lineage block for a result JSON.

    ``script`` (defaults to the immediate caller's file) is hashed as
    ``experiment_script_sha256``; ``checkpoint`` (a model file) is hashed if given. Also records
    ``git_dirty``, a protocol hash, and the dataset metadata hash.
    """
    if script is None:
        try:
            import inspect
            script = inspect.stack()[1].filename
        except Exception:
            script = None
    lin = {
        "commit": _commit(),
        "git_dirty": _git_dirty(),
        "protocol": protocol,
        "protocol_sha256": _protocol_sha(),
        "experiment_script": (os.path.basename(script) if script else None),
        "experiment_script_sha256": (file_sha256(script) if script else None),
        "seed": int(seed),
        "targets": list(targets),
        "segment_def": SEGMENT_DEF,
        "normalization": normalization,
        "dataset": dataset_fingerprint(db) if db is not None else None,
        "n_train": (int(len(list(train_ids))) if train_ids is not None else None),
        "n_test": (int(len(list(test_ids))) if test_ids is not None else None),
        "train_ids_sha256": (ids_sha256(train_ids) if train_ids is not None else None),
        "test_ids_sha256": (ids_sha256(test_ids) if test_ids is not None else None),
        "checkpoint_sha256": (file_sha256(checkpoint) if checkpoint else None),
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
