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
import importlib.metadata
import json
import os
import platform
import re
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

PROTOCOL_VERSION = "icassp27-robust-map-v3-2026-07"
SEGMENT_DEF = (
    "500 Hz NeuroKit2-DWT on canonical Lead II; QRS=Ron..Roff, "
    "ST=J..Ton, T=Ton..Toff (P supplementary)"
)


def _commit() -> str:
    env = os.environ.get("ECG_COMMIT")
    if env:
        return env.strip()
    try:
        source_repo = os.environ.get("ECG_SOURCE_REPO") or None
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=source_repo,
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:
        pass
    return "unknown"


def ids_sha256(ids: Iterable) -> str:
    values = sorted(str(value) for value in ids)
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()[:16]


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


def artifact_sha256(path: os.PathLike | str) -> str:
    """Return the full SHA-256 for an existing artifact, failing closed if it is absent."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"artifact does not exist or is not a file: {p}")
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha256(value: Any) -> str:
    """Hash JSON-compatible data with a stable encoding."""
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def environment_sha256() -> str:
    """Fingerprint the interpreter and installed distributions without invoking pip."""
    packages = sorted(
        (dist.metadata.get("Name", "").lower(), dist.version)
        for dist in importlib.metadata.distributions()
        if dist.metadata.get("Name")
    )
    return canonical_sha256({"python": sys.version, "packages": packages})


def hardware_fingerprint() -> dict[str, Any]:
    """Capture portable hardware facts; GPU probing is read-only and best effort."""
    memory_bytes = None
    try:
        memory_bytes = int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        pass
    if memory_bytes is None and os.name == "nt":
        try:
            import ctypes

            class _MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("length", ctypes.c_ulong),
                    ("memory_load", ctypes.c_ulong),
                    ("total_physical", ctypes.c_ulonglong),
                    ("available_physical", ctypes.c_ulonglong),
                    ("total_page_file", ctypes.c_ulonglong),
                    ("available_page_file", ctypes.c_ulonglong),
                    ("total_virtual", ctypes.c_ulonglong),
                    ("available_virtual", ctypes.c_ulonglong),
                    ("available_extended_virtual", ctypes.c_ulonglong),
                ]

            status = _MemoryStatus()
            status.length = ctypes.sizeof(_MemoryStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                memory_bytes = int(status.total_physical)
        except (AttributeError, OSError, ValueError):
            pass
    gpu: list[dict[str, str]] = []
    torch_version = "unavailable"
    cuda_runtime = "unavailable"
    cudnn_version: str | int = "unavailable"
    try:
        import torch

        torch_version = str(torch.__version__)
        cuda_runtime = str(torch.version.cuda or "unavailable")
        cudnn = torch.backends.cudnn.version()
        if cudnn is not None:
            cudnn_version = int(cudnn)
    except (ImportError, AttributeError, RuntimeError):
        pass
    cgroup: dict[str, str] = {}
    for name, path in (
        ("cpu_max", Path("/sys/fs/cgroup/cpu.max")),
        ("memory_max", Path("/sys/fs/cgroup/memory.max")),
        ("memory_high", Path("/sys/fs/cgroup/memory.high")),
    ):
        try:
            cgroup[name] = path.read_text(encoding="utf-8").strip()
        except OSError:
            pass
    try:
        probe = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=5, check=False,
        )
        if probe.returncode == 0:
            for line in probe.stdout.splitlines():
                parts = [part.strip() for part in line.split(",")]
                if len(parts) == 5:
                    gpu.append(dict(zip(("index", "name", "uuid", "memory_mib", "driver"), parts)))
    except (FileNotFoundError, subprocess.SubprocessError):
        pass
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "memory_bytes": memory_bytes if memory_bytes is not None else "unavailable",
        "cgroup": cgroup,
        "gpu": gpu,
        "torch": torch_version,
        "cuda_runtime": cuda_runtime,
        "cudnn": cudnn_version,
    }


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
        source_repo = os.environ.get("ECG_SOURCE_REPO") or None
        out = subprocess.run(
            ["git", "status", "--porcelain"], cwd=source_repo,
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode != 0:
            return True
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
        return True


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
         checkpoint: str | None = None, upstream: Mapping[str, str] | None = None,
         extra: dict | None = None) -> dict:
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
    train_ids = list(train_ids) if train_ids is not None else None
    test_ids = list(test_ids) if test_ids is not None else None
    targets = list(targets)
    dataset = dataset_fingerprint(db) if db is not None else {}
    argv = list(sys.argv)
    split = {
        "train_ids_sha256": ids_sha256(train_ids) if train_ids is not None else None,
        "test_ids_sha256": ids_sha256(test_ids) if test_ids is not None else None,
    }
    lin = {
        "commit": _commit(),
        "git_dirty": _git_dirty(),
        "argv": argv,
        "config_sha256": canonical_sha256({
            "protocol": protocol, "targets": targets,
            "normalization": normalization, "argv": argv,
        }),
        "data_sha256": canonical_sha256(dataset),
        "split_sha256": canonical_sha256(split),
        "env_sha256": environment_sha256(),
        "hardware": hardware_fingerprint(),
        "protocol": protocol,
        "protocol_sha256": _protocol_sha(),
        "experiment_script": (os.path.basename(script) if script else None),
        "experiment_script_sha256": (file_sha256(script) if script else None),
        "seed": int(seed),
        "targets": targets,
        "segment_def": SEGMENT_DEF,
        "normalization": normalization,
        "dataset": dataset,
        "n_train": (int(len(train_ids)) if train_ids is not None else None),
        "n_test": (int(len(test_ids)) if test_ids is not None else None),
        "train_ids_sha256": split["train_ids_sha256"],
        "test_ids_sha256": split["test_ids_sha256"],
        "upstream_sha256": dict(upstream or {}),
        "checkpoint_sha256": (artifact_sha256(checkpoint) if checkpoint else {}),
    }
    if extra:
        lin.update(extra)
    return lin


_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_STRICT_REQUIRED = (
    "commit", "git_dirty", "argv", "config_sha256", "data_sha256", "split_sha256",
    "env_sha256", "hardware", "seed", "upstream_sha256", "checkpoint_sha256",
)


def validate_strict_lineage(value: Mapping[str, Any], *, require_checkpoint: bool = False) -> None:
    """Validate submission-grade provenance and reject missing/null/dirty metadata.

    Empty upstream/checkpoint mappings are explicit and valid for nodes that genuinely have no
    such artifacts. Model nodes pass ``require_checkpoint=True``.
    """
    if not isinstance(value, Mapping):
        raise ValueError("lineage must be an object")
    missing = [key for key in _STRICT_REQUIRED if key not in value or value[key] is None]
    if missing:
        raise ValueError(f"lineage missing/null required fields: {missing}")
    if not _HEX40.fullmatch(str(value["commit"])):
        raise ValueError("lineage.commit must be a full 40-character git SHA")
    if value["git_dirty"] is not False:
        raise ValueError("lineage.git_dirty must be exactly false")
    if not isinstance(value["argv"], list) or not value["argv"] or not all(
            isinstance(item, str) and item for item in value["argv"]):
        raise ValueError("lineage.argv must be a non-empty string list")
    for key in ("config_sha256", "data_sha256", "split_sha256", "env_sha256"):
        if not _HEX64.fullmatch(str(value[key])):
            raise ValueError(f"lineage.{key} must be a full SHA-256")
    if not isinstance(value["hardware"], Mapping) or not value["hardware"]:
        raise ValueError("lineage.hardware must be a non-empty object")
    if isinstance(value["seed"], bool) or not isinstance(value["seed"], int):
        raise ValueError("lineage.seed must be an integer")
    if not isinstance(value["upstream_sha256"], Mapping):
        raise ValueError("lineage.upstream_sha256 must be an object")
    for name, digest in value["upstream_sha256"].items():
        if not name or not _HEX64.fullmatch(str(digest)):
            raise ValueError("lineage.upstream_sha256 contains an invalid artifact hash")
    checkpoint = value["checkpoint_sha256"]
    if require_checkpoint and not checkpoint:
        raise ValueError("model lineage requires checkpoint_sha256")
    if isinstance(checkpoint, Mapping):
        for name, digest in checkpoint.items():
            if not name or not _HEX64.fullmatch(str(digest)):
                raise ValueError("lineage.checkpoint_sha256 contains an invalid artifact hash")
    elif checkpoint and not _HEX64.fullmatch(str(checkpoint)):
        raise ValueError("lineage.checkpoint_sha256 must be a full SHA-256 or an object")


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
