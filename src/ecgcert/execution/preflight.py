"""Read-only, fail-closed server readiness inventory.

The preflight deliberately uses an allowlist of probes.  It never reads the
process environment, SSH configuration, home-directory files, remotes, or
credential stores, and it never includes raw subprocess stderr in its report.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
from typing import Any, Callable, Mapping

from ..estimators.official import ECG_RECOVER, IMPUTE_ECG
from .environment import EnvironmentLockError, verify_locked_environment


SCHEMA_VERSION = "ecgcert-server-preflight/v2"
EXPECTED_PYTHON = (3, 11, 2)
EXPECTED_DATASET_RECORDS = {
    "ptbxl": 21_799,
    "chapman": 45_152,
    "cpsc2018": 6_877,
}
EXPECTED_LOCK_SHA256 = {
    "cpu": "bc5534f459af61759abe6e3c640553d266d4d58f73d2cac404990584d7704ed9",
    "gpu": "fbe43187cea8667241409d33e0378f4cf937ffb4804a2f2182acd58d1d0efd2e",
}
EXPECTED_TOOL_COMMITS = {
    "imputeecg": "70accf2f1600066392b14a5f50dbc131a6f13943",
    "ecgrecover": "ed49dddf8e5e599b8af702e871a1f66b1d628518",
    "autoresearchclaw": "e2e23c93b4943fd21cc531deb09850d8fda55357",
}
EXPECTED_TOOL_ROOT_TREES = {
    "imputeecg": "d30565ea404a6b7f848fe3a9f5cc742655eb0388",
    "ecgrecover": "980f872d6d25b1291942f4929d1417abec66fe1e",
}
EXPECTED_TOOL_ORIGINS = {
    "imputeecg": "https://github.com/PKUDigitalHealth/ImputeECG.git",
    "ecgrecover": "https://git.ummisco.fr/open/2024-ecg-recover.git",
}
EXPECTED_TOOL_REQUIRED_PATHS = {
    "imputeecg": IMPUTE_ECG.required_paths,
    "ecgrecover": ECG_RECOVER.required_paths,
}
EXPECTED_ACPX_VERSION = "0.12.0"
EXPECTED_TORCH_VERSION = "2.8.0+cu128"
EXPECTED_TORCH_CUDA = "12.8"


@dataclass(frozen=True)
class PreflightConfig:
    repo: Path
    expected_commit: str | None
    storage_root: Path
    tools_root: Path
    ptbxl_root: Path
    chapman_root: Path
    cpsc2018_root: Path
    min_logical_cpus: int = 10
    min_ram_bytes: int = 56 * 1024**3
    min_gpu_memory_mib: int = 16 * 1024
    min_free_bytes: int = 100 * 1024**3
    expected_python: tuple[int, int, int] = EXPECTED_PYTHON
    expected_dataset_records: Mapping[str, int] = field(
        default_factory=lambda: dict(EXPECTED_DATASET_RECORDS)
    )
    expected_lock_sha256: Mapping[str, str] = field(
        default_factory=lambda: dict(EXPECTED_LOCK_SHA256)
    )
    expected_tool_commits: Mapping[str, str] = field(
        default_factory=lambda: dict(EXPECTED_TOOL_COMMITS)
    )
    expected_tool_root_trees: Mapping[str, str] = field(
        default_factory=lambda: dict(EXPECTED_TOOL_ROOT_TREES)
    )
    expected_tool_origins: Mapping[str, str] = field(
        default_factory=lambda: dict(EXPECTED_TOOL_ORIGINS)
    )
    expected_tool_required_paths: Mapping[str, tuple[str, ...]] = field(
        default_factory=lambda: dict(EXPECTED_TOOL_REQUIRED_PATHS)
    )
    expected_acpx_version: str = EXPECTED_ACPX_VERSION
    expected_torch_version: str = EXPECTED_TORCH_VERSION
    expected_torch_cuda: str = EXPECTED_TORCH_CUDA
    active_environment_lock: str = "gpu"
    require_linux: bool = True
    # Production DAGs consume the ignored repository links, whereas the
    # inventory accepts explicit persistent roots.  Requiring both views to
    # resolve to the same absolute targets prevents a clean preflight of one
    # dataset/upstream tree followed by execution against another.
    require_staged_links: bool = True


class _Issues:
    def __init__(self) -> None:
        self.errors: list[dict[str, str]] = []
        self.warnings: list[dict[str, str]] = []

    def error(self, code: str, message: str) -> None:
        self.errors.append({"code": code, "message": message})

    def warning(self, code: str, message: str) -> None:
        self.warnings.append({"code": code, "message": message})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run(argv: list[str], *, cwd: Path | None = None, timeout: int = 20) -> subprocess.CompletedProcess[str] | None:
    """Run one allowlisted read-only command without leaking its diagnostics."""

    try:
        return subprocess.run(
            argv,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return None


def _memory_total_bytes() -> int:
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    if os.name == "nt":  # pragma: no cover - exercised only on Windows hosts
        import ctypes

        class MemoryStatus(ctypes.Structure):
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

        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.total_physical)
    if hasattr(os, "sysconf"):
        try:
            return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
        except (OSError, ValueError):
            pass
    return 0


def _cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name") and ":" in line:
                return line.split(":", 1)[1].strip()
    return platform.processor().strip() or platform.machine().strip() or "unavailable"


def probe_system() -> dict[str, Any]:
    return {
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "version_info": list(sys.version_info[:3]),
        },
        "cpu": {
            "logical_count": int(os.cpu_count() or 0),
            "model": _cpu_model(),
        },
        "ram": {"total_bytes": _memory_total_bytes()},
    }


def probe_tex_toolchain() -> dict[str, Any]:
    """Inventory the two executables used by the terminal paper DAG nodes."""

    tools: dict[str, Any] = {}
    for name in ("pdflatex", "bibtex"):
        discovered = shutil.which(name)
        entry: dict[str, Any] = {
            "available": False,
            "path": "unavailable",
            "sha256": "unavailable",
            "version": "unavailable",
        }
        if discovered:
            try:
                path = Path(discovered).resolve(strict=True)
                completed = _run([str(path), "--version"], timeout=20)
                lines = (
                    completed.stdout.splitlines()
                    if completed is not None and completed.returncode == 0
                    else []
                )
                version = next((line.strip() for line in lines if line.strip()), "")
                if path.is_file() and version:
                    entry.update(
                        {
                            "available": True,
                            "path": str(path),
                            "sha256": _sha256(path),
                            "version": version[:256],
                        }
                    )
            except OSError:
                pass
        tools[name] = entry
    return tools


def _parse_nvidia_rows(stdout: str) -> list[dict[str, Any]]:
    devices: list[dict[str, Any]] = []
    for row in csv.reader(line for line in stdout.splitlines() if line.strip()):
        if len(row) != 5:
            continue
        try:
            devices.append(
                {
                    "index": int(row[0].strip()),
                    "name": row[1].strip(),
                    "uuid": row[2].strip(),
                    "driver_version": row[3].strip(),
                    "memory_total_mib": int(float(row[4].strip())),
                }
            )
        except ValueError:
            continue
    return devices


def probe_nvidia() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    result: dict[str, Any] = {
        "available": False,
        "executable": executable or "unavailable",
        "driver_version": "unavailable",
        "cuda_version_reported": "unavailable",
        "devices": [],
        "nvcc_version": "unavailable",
    }
    if not executable:
        return result
    query = _run(
        [
            executable,
            "--query-gpu=index,name,uuid,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    header = _run([executable])
    if query is None or query.returncode != 0 or header is None or header.returncode != 0:
        return result
    devices = _parse_nvidia_rows(query.stdout)
    cuda_match = re.search(r"CUDA Version:\s*([0-9.]+)", header.stdout)
    if not devices or cuda_match is None:
        return result
    drivers = sorted({str(device["driver_version"]) for device in devices})
    result.update(
        {
            "available": True,
            "driver_version": drivers[0] if len(drivers) == 1 else ",".join(drivers),
            "cuda_version_reported": cuda_match.group(1),
            "devices": devices,
        }
    )
    nvcc = shutil.which("nvcc")
    if nvcc:
        completed = _run([nvcc, "--version"])
        if completed is not None and completed.returncode == 0:
            match = re.search(r"release\s+([0-9.]+)", completed.stdout)
            if match:
                result["nvcc_version"] = match.group(1)
    return result


def probe_torch_cuda() -> dict[str, Any]:
    result: dict[str, Any] = {
        "installed": False,
        "version": "unavailable",
        "compiled_cuda": "unavailable",
        "cuda_available": False,
        "device_count": 0,
        "devices": [],
    }
    if importlib.util.find_spec("torch") is None:
        return result
    try:
        import torch

        result.update(
            {
                "installed": True,
                "version": str(torch.__version__),
                "compiled_cuda": str(torch.version.cuda or "unavailable"),
                "cuda_available": bool(torch.cuda.is_available()),
                "device_count": int(torch.cuda.device_count()),
            }
        )
        if result["cuda_available"]:
            result["devices"] = [
                {
                    "index": index,
                    "name": str(torch.cuda.get_device_name(index)),
                    "compute_capability": list(torch.cuda.get_device_capability(index)),
                    "memory_total_bytes": int(
                        torch.cuda.get_device_properties(index).total_memory
                    ),
                }
                for index in range(result["device_count"])
            ]
    except Exception:
        # The failure is represented by cuda_available=false and becomes a
        # fail-closed issue.  Exception text is intentionally not disclosed.
        pass
    return result


def inspect_git_checkout(
    path: Path,
    *,
    expected_commit: str | None,
    expected_root_tree: str | None = None,
    expected_origin: str | None = None,
    required_paths: tuple[str, ...] = (),
    require_partial_clone: bool = False,
    issues: _Issues,
    code_prefix: str,
) -> dict[str, Any]:
    resolved = path.resolve()
    result: dict[str, Any] = {
        "path": str(resolved),
        "present": resolved.is_dir(),
        "commit": "unavailable",
        "expected_commit": expected_commit or "unavailable",
        "root_tree": "unavailable",
        "expected_root_tree": expected_root_tree or "unavailable",
        "origin": "unavailable",
        "expected_origin": expected_origin or "unavailable",
        "dirty": True,
        "dirty_entry_count": 0,
        "required_path_count": len(required_paths),
        "materialized_required_path_count": 0,
        "borrows_external_objects": True,
        "promisor": "unavailable",
        "partial_clone_filter": "unavailable",
    }
    if not resolved.is_dir():
        issues.error(f"{code_prefix}.missing", "required Git checkout is missing")
        return result
    if expected_commit is None or not re.fullmatch(r"[0-9a-f]{40}", expected_commit):
        issues.error(
            f"{code_prefix}.expected_commit",
            "an exact 40-character expected commit is required",
        )
    head = _run(["git", "rev-parse", "HEAD"], cwd=resolved)
    status = _run(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=resolved
    )
    tree = (
        _run(["git", "rev-parse", "HEAD^{tree}"], cwd=resolved)
        if expected_root_tree is not None
        else None
    )
    origin = (
        _run(["git", "remote", "get-url", "origin"], cwd=resolved)
        if expected_origin is not None
        else None
    )
    promisor = (
        _run(["git", "config", "--get", "remote.origin.promisor"], cwd=resolved)
        if require_partial_clone
        else None
    )
    partial_filter = (
        _run(
            ["git", "config", "--get", "remote.origin.partialclonefilter"],
            cwd=resolved,
        )
        if require_partial_clone
        else None
    )
    if (
        head is None
        or head.returncode != 0
        or status is None
        or status.returncode != 0
        or (expected_root_tree is not None and (tree is None or tree.returncode != 0))
        or (expected_origin is not None and (origin is None or origin.returncode != 0))
        or (
            require_partial_clone
            and (promisor is None or promisor.returncode != 0)
        )
        or (
            require_partial_clone
            and (partial_filter is None or partial_filter.returncode != 0)
        )
    ):
        issues.error(f"{code_prefix}.git", "Git checkout inventory failed")
        return result
    commit = head.stdout.strip()
    root_tree = tree.stdout.strip() if tree is not None else "unavailable"
    origin_value = origin.stdout.strip() if origin is not None else "unavailable"
    dirty_count = len([line for line in status.stdout.splitlines() if line.strip()])
    promisor_value = (
        promisor.stdout.strip().casefold() if promisor is not None else "not-required"
    )
    partial_filter_value = (
        partial_filter.stdout.strip() if partial_filter is not None else "not-required"
    )
    materialized = sum((resolved / relative).is_file() for relative in required_paths)
    alternates = resolved / ".git" / "objects" / "info" / "alternates"
    borrows_objects = os.path.lexists(alternates)
    result.update(
        {
            "commit": commit,
            "root_tree": root_tree,
            "origin": origin_value,
            "dirty": dirty_count != 0,
            "dirty_entry_count": dirty_count,
            "required_path_count": len(required_paths),
            "materialized_required_path_count": materialized,
            "borrows_external_objects": borrows_objects,
            "promisor": promisor_value,
            "partial_clone_filter": partial_filter_value,
        }
    )
    if expected_commit and commit != expected_commit:
        issues.error(f"{code_prefix}.commit", "Git checkout does not match its frozen commit")
    if expected_root_tree and root_tree != expected_root_tree:
        issues.error(f"{code_prefix}.root_tree", "Git checkout root tree is not frozen")
    if expected_origin and origin_value.rstrip("/") != expected_origin.rstrip("/"):
        issues.error(f"{code_prefix}.origin", "Git checkout origin is not the official repository")
    if dirty_count:
        issues.error(f"{code_prefix}.dirty", "Git checkout is not clean")
    if materialized != len(required_paths):
        issues.error(
            f"{code_prefix}.required_paths",
            "Git checkout lacks one or more frozen runtime source files",
        )
    if borrows_objects:
        issues.error(
            f"{code_prefix}.alternates",
            "Git checkout borrows an external object store",
        )
    if require_partial_clone and (
        promisor_value != "true" or partial_filter_value != "blob:none"
    ):
        issues.error(
            f"{code_prefix}.partial_clone",
            "Git checkout does not retain the frozen promisor/partial-clone policy",
        )
    return result


def _inspect_staged_link(
    *,
    link: Path,
    expected_target: Path,
    repository: Path,
    issues: _Issues,
    code: str,
) -> dict[str, Any]:
    """Authenticate one absolute repository link without following it first."""

    result: dict[str, Any] = {
        "path": str(link),
        "present": os.path.lexists(link),
        "is_symlink": link.is_symlink(),
        "raw_target": "unavailable",
        "resolved_target": "unavailable",
        "expected_target": str(expected_target),
        "absolute_target": False,
        "outside_repository": False,
        "matches_expected_target": False,
    }
    if not result["present"]:
        issues.error(f"{code}.missing", "required repository staging link is missing")
        return result
    if not result["is_symlink"]:
        issues.error(f"{code}.type", "repository staging path is not a symbolic link")
        return result
    try:
        raw_target = Path(os.readlink(link))
        resolved_target = link.resolve(strict=True)
        expected = expected_target.resolve(strict=True)
    except OSError:
        issues.error(f"{code}.resolve", "repository staging link cannot be resolved")
        return result
    absolute = raw_target.is_absolute()
    outside = resolved_target != repository and repository not in resolved_target.parents
    matches = resolved_target == expected
    result.update(
        {
            "raw_target": str(raw_target),
            "resolved_target": str(resolved_target),
            "expected_target": str(expected),
            "absolute_target": absolute,
            "outside_repository": outside,
            "matches_expected_target": matches,
        }
    )
    if not absolute:
        issues.error(f"{code}.relative", "repository staging link target is not absolute")
    if not outside:
        issues.error(f"{code}.inside_repo", "staged persistent target is inside the repository")
    if not matches:
        issues.error(
            f"{code}.mismatch",
            "repository staging link does not match the preflight inventory target",
        )
    return result


def inspect_repository_links(
    config: PreflightConfig,
    issues: _Issues,
) -> dict[str, Any]:
    """Bind repository-visible DAG inputs to the roots inspected below."""

    repository = config.repo.resolve()
    if not config.require_staged_links:
        return {"required": False, "status": "not-required-for-test-fixture"}

    for parent, code in (
        (repository / "data", "link.parent.data"),
        (repository / "data" / "external", "link.parent.external"),
    ):
        if parent.is_symlink() or not parent.is_dir():
            issues.error(code, "repository data link parent is not a real directory")

    links = {
        "ptbxl": _inspect_staged_link(
            link=repository / "data" / "ptbxl",
            expected_target=config.ptbxl_root,
            repository=repository,
            issues=issues,
            code="link.ptbxl",
        ),
        "chapman": _inspect_staged_link(
            link=repository / "data" / "external" / "chapman",
            expected_target=config.chapman_root,
            repository=repository,
            issues=issues,
            code="link.chapman",
        ),
        "cpsc2018": _inspect_staged_link(
            link=repository / "data" / "external" / "cpsc2018",
            expected_target=config.cpsc2018_root,
            repository=repository,
            issues=issues,
            code="link.cpsc2018",
        ),
        "upstreams": _inspect_staged_link(
            link=repository / "upstreams",
            expected_target=config.tools_root / "upstreams",
            repository=repository,
            issues=issues,
            code="link.upstreams",
        ),
    }
    return {"required": True, "links": links}


def _inventory_signal_tree(root: Path, suffixes: tuple[str, ...]) -> dict[str, Any]:
    """Inventory a signal tree in one pass instead of issuing random stats per CSV row."""

    stems = {suffix: set() for suffix in suffixes}
    sizes = {suffix: 0 for suffix in suffixes}
    zero = {suffix: 0 for suffix in suffixes}
    complete = root.is_dir()
    stack = [root] if complete else []
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(Path(entry.path))
                        continue
                    if not entry.is_file(follow_symlinks=False):
                        continue
                    suffix = Path(entry.name).suffix.lower()
                    if suffix not in stems:
                        continue
                    relative = Path(entry.path).relative_to(root).with_suffix("").as_posix()
                    stems[suffix].add(relative)
                    size = entry.stat(follow_symlinks=False).st_size
                    sizes[suffix] += size
                    if size == 0:
                        zero[suffix] += 1
        except OSError:
            complete = False
    return {"stems": stems, "sizes": sizes, "zero": zero, "complete": complete}


def inspect_ptbxl(
    root: Path,
    *,
    expected_records: int,
    issues: _Issues,
) -> dict[str, Any]:
    resolved = root.resolve()
    result: dict[str, Any] = {
        "path": str(resolved),
        "present": resolved.is_dir(),
        "expected_records": expected_records,
        "metadata_rows": 0,
        "unique_ecg_ids": 0,
        "unique_patient_ids": 0,
        "strat_folds": [],
        "records100": {
            "headers": 0,
            "signals": 0,
            "signal_bytes": 0,
            "zero_byte_headers": 0,
            "zero_byte_signals": 0,
        },
        "records500": {
            "headers": 0,
            "signals": 0,
            "signal_bytes": 0,
            "zero_byte_headers": 0,
            "zero_byte_signals": 0,
        },
        "missing_record_files": 0,
        "metadata_sha256": "unavailable",
        "scp_statements_sha256": "unavailable",
        "integrity_level": "structural-count-and-pairing",
    }
    if not resolved.is_dir():
        issues.error("dataset.ptbxl.missing", "PTB-XL root is missing")
        return result
    metadata = resolved / "ptbxl_database.csv"
    statements = resolved / "scp_statements.csv"
    if not metadata.is_file() or not statements.is_file():
        issues.error("dataset.ptbxl.metadata", "PTB-XL metadata files are incomplete")
        return result
    result["metadata_sha256"] = _sha256(metadata)
    result["scp_statements_sha256"] = _sha256(statements)
    required = {"ecg_id", "patient_id", "strat_fold", "filename_lr", "filename_hr"}
    rows: list[dict[str, str]] = []
    try:
        with metadata.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                issues.error("dataset.ptbxl.columns", "PTB-XL metadata columns are incomplete")
                return result
            rows = list(reader)
    except (OSError, csv.Error):
        issues.error("dataset.ptbxl.read", "PTB-XL metadata could not be read")
        return result
    ecg_ids = {row["ecg_id"] for row in rows}
    patients = {row["patient_id"] for row in rows if row["patient_id"]}
    folds: set[int] = set()
    invalid_fold = 0
    inventories = {
        rate: _inventory_signal_tree(resolved / f"records{rate}", (".hea", ".dat"))
        for rate in (100, 500)
    }
    missing = 0
    unsafe_paths = 0
    for row in rows:
        try:
            folds.add(int(row["strat_fold"]))
        except ValueError:
            invalid_fold += 1
        for rate, column in ((100, "filename_lr"), (500, "filename_hr")):
            relative = Path(row[column])
            if relative.is_absolute() or ".." in relative.parts:
                unsafe_paths += 1
                missing += 1
                continue
            normalized = relative.as_posix()
            prefix = f"records{rate}/"
            if not normalized.startswith(prefix):
                unsafe_paths += 1
                missing += 1
                continue
            local_stem = normalized[len(prefix) :]
            inventory = inventories[rate]["stems"]
            if local_stem not in inventory[".hea"] or local_stem not in inventory[".dat"]:
                missing += 1
    result.update(
        {
            "metadata_rows": len(rows),
            "unique_ecg_ids": len(ecg_ids),
            "unique_patient_ids": len(patients),
            "strat_folds": sorted(folds),
            "missing_record_files": missing,
        }
    )
    for rate in (100, 500):
        inventory = inventories[rate]
        result[f"records{rate}"] = {
            "headers": len(inventory["stems"][".hea"]),
            "signals": len(inventory["stems"][".dat"]),
            "signal_bytes": inventory["sizes"][".dat"],
            "zero_byte_headers": inventory["zero"][".hea"],
            "zero_byte_signals": inventory["zero"][".dat"],
        }
        if not inventory["complete"]:
            issues.error(
                f"dataset.ptbxl.records{rate}_scan",
                f"PTB-XL {rate} Hz tree inventory was incomplete",
            )
    if len(rows) != expected_records or len(ecg_ids) != expected_records:
        issues.error("dataset.ptbxl.count", "PTB-XL does not contain the complete frozen cohort")
    if folds != set(range(1, 11)) or invalid_fold:
        issues.error("dataset.ptbxl.folds", "PTB-XL stratified folds are incomplete")
    if not patients:
        issues.error("dataset.ptbxl.patients", "PTB-XL patient identifiers are missing")
    if missing:
        issues.error("dataset.ptbxl.pairs", "PTB-XL has missing header or signal pairs")
    if unsafe_paths:
        issues.error("dataset.ptbxl.paths", "PTB-XL metadata contains unsafe record paths")
    for rate in (100, 500):
        counts = result[f"records{rate}"]
        if counts["headers"] != expected_records or counts["signals"] != expected_records:
            issues.error(
                f"dataset.ptbxl.records{rate}",
                f"PTB-XL {rate} Hz file counts are incomplete",
            )
        if counts["zero_byte_headers"] or counts["zero_byte_signals"]:
            issues.error(
                f"dataset.ptbxl.records{rate}_empty",
                f"PTB-XL {rate} Hz contains empty header or signal files",
            )
    return result


def inspect_external_wfdb(
    name: str,
    root: Path,
    *,
    expected_records: int,
    issues: _Issues,
) -> dict[str, Any]:
    resolved = root.resolve()
    result: dict[str, Any] = {
        "path": str(resolved),
        "present": resolved.is_dir(),
        "expected_records": expected_records,
        "headers": 0,
        "signals": 0,
        "mat_signals": 0,
        "dat_signals": 0,
        "signal_bytes": 0,
        "orphan_headers": 0,
        "orphan_signals": 0,
        "ambiguous_signal_files": 0,
        "zero_byte_headers": 0,
        "zero_byte_signals": 0,
        "patient_unit": "one-record-per-patient",
        "integrity_level": "structural-count-and-pairing",
    }
    prefix = f"dataset.{name}"
    if not resolved.is_dir():
        issues.error(f"{prefix}.missing", f"{name} root is missing")
        return result
    inventory = _inventory_signal_tree(resolved, (".hea", ".mat", ".dat"))
    headers = inventory["stems"][".hea"]
    mat = inventory["stems"][".mat"]
    dat = inventory["stems"][".dat"]
    signals = mat | dat
    orphan = len(headers - signals)
    orphan_signals = len(signals - headers)
    ambiguous = len(mat & dat)
    zero_headers = inventory["zero"][".hea"]
    zero = inventory["zero"][".mat"] + inventory["zero"][".dat"]
    if not inventory["complete"]:
        issues.error(f"{prefix}.scan", f"{name} tree inventory was incomplete")
    result.update(
        {
            "headers": len(headers),
            "signals": len(signals),
            "mat_signals": len(mat),
            "dat_signals": len(dat),
            "signal_bytes": inventory["sizes"][".mat"] + inventory["sizes"][".dat"],
            "orphan_headers": orphan,
            "orphan_signals": orphan_signals,
            "ambiguous_signal_files": ambiguous,
            "zero_byte_headers": zero_headers,
            "zero_byte_signals": zero,
        }
    )
    if len(headers) != expected_records or len(signals) != expected_records:
        issues.error(f"{prefix}.count", f"{name} does not contain the complete frozen cohort")
    if orphan:
        issues.error(f"{prefix}.pairs", f"{name} has headers without signal files")
    if ambiguous:
        issues.error(
            f"{prefix}.ambiguous",
            f"{name} has records with both .mat and .dat signal files",
        )
    if zero_headers:
        issues.error(f"{prefix}.empty_header", f"{name} has empty header files")
    if zero:
        issues.error(f"{prefix}.empty", f"{name} has empty signal files")
    if orphan_signals:
        issues.error(f"{prefix}.orphan_signals", f"{name} has signal files without headers")
    return result


def inspect_locks(config: PreflightConfig, issues: _Issues) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name in ("cpu", "gpu"):
        path = (config.repo / "environments" / f"{name}.lock.txt").resolve()
        expected = config.expected_lock_sha256.get(name, "")
        entry = {
            "path": str(path),
            "present": path.is_file(),
            "sha256": "unavailable",
            "expected_sha256": expected or "unavailable",
            "size_bytes": path.stat().st_size if path.is_file() else 0,
        }
        if not path.is_file():
            issues.error(f"lock.{name}.missing", f"{name} environment lock is missing")
        else:
            entry["sha256"] = _sha256(path)
            if not expected or entry["sha256"] != expected:
                issues.error(f"lock.{name}.hash", f"{name} environment lock hash is not frozen")
            if entry["size_bytes"] == 0:
                issues.error(f"lock.{name}.empty", f"{name} environment lock is empty")
        output[name] = entry
    return output


def inspect_active_environment(
    config: PreflightConfig,
    issues: _Issues,
    *,
    version_lookup: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    """Bind the current interpreter to the one lock used by the whole DAG run."""

    try:
        kwargs: dict[str, Any] = {
            "repo": config.repo,
            "lock_name": config.active_environment_lock,
        }
        if version_lookup is not None:
            kwargs["version_lookup"] = version_lookup
        report = verify_locked_environment(**kwargs)
    except (EnvironmentLockError, OSError, ValueError) as exc:
        issues.error(
            "environment.lock_parse",
            f"active environment lock cannot be verified: {type(exc).__name__}",
        )
        return {
            "lock_name": config.active_environment_lock,
            "python_executable": str(Path(sys.executable).resolve()),
            "ok": False,
            "status": "unavailable",
        }
    value = report.as_dict()
    if not report.ok:
        issues.error(
            "environment.lock_mismatch",
            "active interpreter does not satisfy every applicable frozen package pin",
        )
    expected = config.expected_lock_sha256.get(config.active_environment_lock, "")
    if report.lock_sha256 != expected:
        issues.error(
            "environment.lock_hash",
            "active environment lock does not match the frozen lock hash",
        )
    return value


def inspect_tools(config: PreflightConfig, issues: _Issues) -> dict[str, Any]:
    upstream_root = config.tools_root.resolve() / "upstreams"
    impute_commit = config.expected_tool_commits.get("imputeecg", "")
    recover_commit = config.expected_tool_commits.get("ecgrecover", "")
    output = {
        "tools_root": str(config.tools_root.resolve()),
        "imputeecg": inspect_git_checkout(
            upstream_root / f"ImputeECG-{impute_commit[:12]}",
            expected_commit=impute_commit or None,
            expected_root_tree=config.expected_tool_root_trees.get("imputeecg"),
            expected_origin=config.expected_tool_origins.get("imputeecg"),
            required_paths=tuple(
                config.expected_tool_required_paths.get("imputeecg", ())
            ),
            require_partial_clone=True,
            issues=issues,
            code_prefix="tool.imputeecg",
        ),
        "ecgrecover": inspect_git_checkout(
            upstream_root / f"ECGrecover-{recover_commit[:12]}",
            expected_commit=recover_commit or None,
            expected_root_tree=config.expected_tool_root_trees.get("ecgrecover"),
            expected_origin=config.expected_tool_origins.get("ecgrecover"),
            required_paths=tuple(
                config.expected_tool_required_paths.get("ecgrecover", ())
            ),
            require_partial_clone=True,
            issues=issues,
            code_prefix="tool.ecgrecover",
        ),
    }
    return output


def collect_server_preflight(
    config: PreflightConfig,
    *,
    system_probe: Callable[[], dict[str, Any]] = probe_system,
    nvidia_probe: Callable[[], dict[str, Any]] = probe_nvidia,
    torch_probe: Callable[[], dict[str, Any]] = probe_torch_cuda,
    tex_probe: Callable[[], dict[str, Any]] = probe_tex_toolchain,
    installed_version_lookup: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    issues = _Issues()
    system = system_probe()
    system_name = str(system.get("os", {}).get("system", ""))
    version_info = tuple(system.get("python", {}).get("version_info", ()))
    logical_cpus = int(system.get("cpu", {}).get("logical_count", 0) or 0)
    ram_bytes = int(system.get("ram", {}).get("total_bytes", 0) or 0)
    if config.require_linux and system_name != "Linux":
        issues.error("system.os", "server execution requires Linux")
    if version_info != config.expected_python:
        issues.error("system.python", "Python does not match the frozen interpreter version")
    if logical_cpus < config.min_logical_cpus:
        issues.error("system.cpu", "logical CPU count is below the frozen minimum")
    if ram_bytes < config.min_ram_bytes:
        issues.error("system.ram", "physical RAM is below the frozen minimum")

    storage_path = config.storage_root.resolve()
    storage: dict[str, Any] = {
        "path": str(storage_path),
        "present": storage_path.is_dir(),
        "total_bytes": 0,
        "used_bytes": 0,
        "free_bytes": 0,
        "required_free_bytes": config.min_free_bytes,
    }
    if not storage_path.is_dir():
        issues.error("storage.missing", "persistent storage root is missing")
    else:
        try:
            usage = shutil.disk_usage(storage_path)
            storage.update(
                {
                    "total_bytes": usage.total,
                    "used_bytes": usage.used,
                    "free_bytes": usage.free,
                }
            )
            if usage.free < config.min_free_bytes:
                issues.error("storage.free", "persistent storage free space is insufficient")
        except OSError:
            issues.error("storage.inventory", "persistent storage inventory failed")

    nvidia = nvidia_probe()
    if not nvidia.get("available") or not nvidia.get("devices"):
        issues.error("gpu.nvidia_smi", "nvidia-smi did not return a usable GPU inventory")
    if nvidia.get("driver_version") in {None, "", "unavailable"}:
        issues.error("gpu.driver", "NVIDIA driver version is unavailable")
    if nvidia.get("cuda_version_reported") in {None, "", "unavailable"}:
        issues.error("gpu.cuda", "nvidia-smi CUDA compatibility version is unavailable")
    device_memory = [
        int(device.get("memory_total_mib", 0) or 0)
        for device in nvidia.get("devices", [])
        if isinstance(device, Mapping)
    ]
    if not device_memory or max(device_memory) < config.min_gpu_memory_mib:
        issues.error("gpu.memory", "GPU memory is below the frozen executor minimum")
    if nvidia.get("nvcc_version") in {None, "", "unavailable"}:
        issues.warning("gpu.nvcc", "nvcc is unavailable; the wheel runtime remains authoritative")
    torch_cuda = torch_probe()
    if not torch_cuda.get("installed"):
        issues.error("gpu.torch", "the frozen GPU environment does not provide torch")
    elif not torch_cuda.get("cuda_available") or not torch_cuda.get("device_count"):
        issues.error("gpu.torch_cuda", "torch cannot initialize a CUDA device")
    if torch_cuda.get("compiled_cuda") in {None, "", "unavailable"}:
        issues.error("gpu.torch_runtime", "torch compiled CUDA runtime is unavailable")
    elif str(torch_cuda.get("compiled_cuda")) != config.expected_torch_cuda:
        issues.error("gpu.torch_runtime", "torch does not match the frozen CUDA runtime")
    if str(torch_cuda.get("version")) != config.expected_torch_version:
        issues.error("gpu.torch_version", "torch does not match the frozen package version")

    git = inspect_git_checkout(
        config.repo,
        expected_commit=config.expected_commit,
        issues=issues,
        code_prefix="git.repo",
    )
    repository_links = inspect_repository_links(config, issues)
    datasets = {
        "ptbxl": inspect_ptbxl(
            config.ptbxl_root,
            expected_records=int(config.expected_dataset_records["ptbxl"]),
            issues=issues,
        ),
        "chapman": inspect_external_wfdb(
            "chapman",
            config.chapman_root,
            expected_records=int(config.expected_dataset_records["chapman"]),
            issues=issues,
        ),
        "cpsc2018": inspect_external_wfdb(
            "cpsc2018",
            config.cpsc2018_root,
            expected_records=int(config.expected_dataset_records["cpsc2018"]),
            issues=issues,
        ),
    }
    locks = inspect_locks(config, issues)
    active_environment = inspect_active_environment(
        config,
        issues,
        version_lookup=installed_version_lookup,
    )
    tools = inspect_tools(config, issues)
    tex = tex_probe()
    for executable in ("pdflatex", "bibtex"):
        entry = tex.get(executable) if isinstance(tex, Mapping) else None
        if not isinstance(entry, Mapping) or not entry.get("available"):
            issues.error(
                f"paper.{executable}",
                f"{executable} is unavailable from the server executor",
            )
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
            "+00:00", "Z"
        ),
        "mode": "read-only-allowlisted-inventory",
        "security": {
            "credentials_read": False,
            "environment_enumerated": False,
            "ssh_configuration_read": False,
            "raw_subprocess_stderr_included": False,
            "dirty_paths_included": False,
        },
        "policy": {
            "expected_python": ".".join(map(str, config.expected_python)),
            "require_linux": config.require_linux,
            "min_logical_cpus": config.min_logical_cpus,
            "min_ram_bytes": config.min_ram_bytes,
            "min_free_bytes": config.min_free_bytes,
            "min_gpu_memory_mib": config.min_gpu_memory_mib,
            "expected_torch_version": config.expected_torch_version,
            "expected_torch_cuda": config.expected_torch_cuda,
            "active_environment_lock": config.active_environment_lock,
            "require_staged_links": config.require_staged_links,
            "control_plane": "local ARC/acpx; not a server-executor prerequisite",
            "expected_dataset_records": dict(config.expected_dataset_records),
        },
        "system": system,
        "storage": storage,
        "gpu": {"nvidia_smi": nvidia, "torch_cuda": torch_cuda},
        "git": git,
        "repository_links": repository_links,
        "datasets": datasets,
        "external_tools": tools,
        "environment_locks": locks,
        "active_environment": active_environment,
        "paper_toolchain": tex,
        "errors": sorted(issues.errors, key=lambda item: item["code"]),
        "warnings": sorted(issues.warnings, key=lambda item: item["code"]),
    }
    report["ok"] = not report["errors"]
    return report


def write_report(report: Mapping[str, Any], output: Path | None) -> str:
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if output is not None:
        destination = output.resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(destination)
    return rendered


def failed_preflight_report(code: str, message: str) -> dict[str, Any]:
    """Return a credential-free JSON fallback when collection itself cannot complete."""

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
            "+00:00", "Z"
        ),
        "mode": "read-only-allowlisted-inventory",
        "security": {
            "credentials_read": False,
            "environment_enumerated": False,
            "ssh_configuration_read": False,
            "raw_subprocess_stderr_included": False,
            "dirty_paths_included": False,
        },
        "policy": {"status": "unavailable"},
        "system": {"status": "unavailable"},
        "storage": {"status": "unavailable"},
        "gpu": {"status": "unavailable"},
        "git": {"status": "unavailable"},
        "repository_links": {"status": "unavailable"},
        "datasets": {"status": "unavailable"},
        "external_tools": {"status": "unavailable"},
        "environment_locks": {"status": "unavailable"},
        "active_environment": {"status": "unavailable"},
        "paper_toolchain": {"status": "unavailable"},
        "errors": [{"code": code, "message": message}],
        "warnings": [],
        "ok": False,
    }


__all__ = [
    "EXPECTED_ACPX_VERSION",
    "EXPECTED_DATASET_RECORDS",
    "EXPECTED_LOCK_SHA256",
    "EXPECTED_PYTHON",
    "EXPECTED_TOOL_COMMITS",
    "EXPECTED_TOOL_ORIGINS",
    "EXPECTED_TOOL_ROOT_TREES",
    "EXPECTED_TORCH_CUDA",
    "EXPECTED_TORCH_VERSION",
    "PreflightConfig",
    "SCHEMA_VERSION",
    "collect_server_preflight",
    "failed_preflight_report",
    "inspect_external_wfdb",
    "inspect_ptbxl",
    "inspect_repository_links",
    "probe_nvidia",
    "probe_system",
    "probe_tex_toolchain",
    "probe_torch_cuda",
    "write_report",
]
