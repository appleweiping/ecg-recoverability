"""Run pinned AutoResearchClaw with a fail-closed, real co-pilot bridge.

AutoResearchClaw v0.5.0 reads human decisions from stdin.  A detached process
otherwise receives EOF and records an abort.  This bridge keeps stdin open and
forwards only an explicit, stage-bound operator response.  It never creates an
approval and never passes ``--auto-approve``.

The source configuration is treated as a portable, read-only template.  A
run-specific effective configuration is written under ``RUN_DIR/control``;
all ARC-owned writable paths point outside the repository.  Completed stage
artifacts are checked continuously.  Provider error text, missing/short output,
or decision/health contradictions invalidate the run and terminate only the
process tree launched by this bridge.

  The bridge first publishes an immutable ``arc-stageN-waiting`` receipt.  The
  native experiment DAG validates that pause, builds its scientific gate, and
  obtains one Ed25519 author review.  ``forward_arc_stage_review.py`` then
  writes the deterministic response under
  ``RECEIPT_ROOT/arc-operator-responses/stage-N``::

    {
      "schema_version": "arc-operator-response-v2",
      "stage": 5,
      "run_id": "rc-...",
      "session_id": "...",
      "waiting_sha256": "...",
      "preapproval_checkpoint_sha256": "...",
      "nonce": "...",
      "action": "approve",
      "issued_at": "2026-07-19T12:00:00+08:00",
      "message": "{...canonical signed-review forward payload...}"
    }

Until that file both matches ARC's native ``hitl/waiting.json`` and verifies
against the repository-pinned reviewer key, the official ARC process remains
blocked at the gate.  Only after native acceptance does the bridge publish the
formal handoff receipt.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import signal
import subprocess
import sys
import time
import traceback
from typing import Any
from urllib.parse import urlsplit

import yaml

from ecgcert.arc_control import (
    ACPX_VERSION,
    ACP_PACKAGE_LOCK_SHA256,
    ARC_COMMIT,
    ARC_REPOSITORY,
    ARC_VERSION,
    CLAUDE_ADAPTER_VERSION,
    CODEX_ADAPTER_VERSION,
    OPERATOR_RESPONSE_CONSUMPTION_SCHEMA,
    OPERATOR_RESPONSE_SCHEMA,
    RECEIPT_SCHEMA,
    WAITING_RECEIPT_SCHEMA,
    validate_arc_control_bundle,
    validate_arc_waiting_bundle,
)
from ecgcert.arc_forward import validate_signed_review_response


CHALLENGE_SCHEMA = "arc-operator-challenge-v2"
RESPONSE_SCHEMA = OPERATOR_RESPONSE_SCHEMA
CONSUMPTION_SCHEMA = OPERATOR_RESPONSE_CONSUMPTION_SCHEMA
STATUS_SCHEMA = "arc-copilot-bridge-status-v2"
INVALIDATION_SCHEMA = "arc-run-invalidation-v1"
GATE_HANDOFF_SCHEMA = "arc-gate-handoff-v2"
REQUIRED_GATES = frozenset({5, 9, 15, 20})
ORDERED_GATES = (5, 9, 15, 20)
TEXT_SUFFIXES = frozenset(
    {".csv", ".json", ".jsonl", ".md", ".tex", ".txt", ".yaml", ".yml"}
)
MIN_TEXT_ARTIFACT_BYTES = 32
MAX_SCANNED_ARTIFACT_BYTES = 16 * 1024 * 1024
# ARC's native contracts identify these as the scientific core of the stages
# that must complete before the first human gate.  A provider error can be a
# syntactically valid, non-empty response, so existence alone is not evidence.
CORE_ARTIFACT_MIN_BYTES: dict[int, dict[str, int]] = {
    1: {"goal.md": 256, "hardware_profile.json": 32},
    2: {"problem_tree.md": 256},
    3: {"search_plan.yaml": 128, "sources.json": 64, "queries.json": 64},
    4: {"candidates.jsonl": 64},
    5: {"shortlist.jsonl": 64},
}
PROTOCOL_REQUIRED_PATTERNS: dict[tuple[int, str], tuple[tuple[str, re.Pattern[str]], ...]] = {
    (1, "goal.md"): (
        ("icassp-2027", re.compile(r"ICASSP\s*2027", re.IGNORECASE)),
        ("ptb-xl", re.compile(r"PTB[ -]?XL", re.IGNORECASE)),
        ("chapman", re.compile(r"Chapman", re.IGNORECASE)),
        ("cpsc", re.compile(r"CPSC", re.IGNORECASE)),
        (
            "target-specific",
            re.compile(r"target(?:[ -](?:lead))?[ -]specific", re.IGNORECASE),
        ),
        ("model-conditional", re.compile(r"model[ -]conditional", re.IGNORECASE)),
        (
            "rank-set-2-3-4-5",
            re.compile(
                r"rank(?:s|\s+set)?(?:\s+(?:is|of))?\s*[:=]?\s*`?\{?\s*2\s*,\s*3\s*,\s*4\s*,\s*5\s*\}?`?",
                re.IGNORECASE,
            ),
        ),
        ("folds-1-7", re.compile(r"folds?\s*1\s*[–-]\s*7", re.IGNORECASE)),
        (
            "fold-8",
            re.compile(
                r"(?:fold\s*)?8\s+(?:for\s+)?(?:tun(?:e|es|ing)|validation)",
                re.IGNORECASE,
            ),
        ),
        (
            "fold-9",
            re.compile(r"(?:fold\s*)?9\s+(?:fit|meta[ -]?model)", re.IGNORECASE),
        ),
        (
            "fold-10",
            re.compile(
                r"(?:fold[ -]?10|\b10\s+(?:one[ -]time\s+)?(?:final\s+)?test)",
                re.IGNORECASE,
            ),
        ),
        ("delta-r2", re.compile(r"(?:delta|Δ)\s*R(?:\^?2|²)", re.IGNORECASE)),
        (
            "zero-transfer",
            re.compile(r"zero[ -]transfer", re.IGNORECASE),
        ),
    )
}
PROTOCOL_FORBIDDEN_PATTERNS: dict[
    tuple[int, str], tuple[tuple[str, re.Pattern[str]], ...]
] = {
    (1, "goal.md"): (
        ("model-agnostic", re.compile(r"model[ -]agnostic", re.IGNORECASE)),
        (
            "intrinsic-recoverability",
            re.compile(r"intrinsic(?:ally)?\s+(?:more\s+or\s+less\s+)?recover", re.IGNORECASE),
        ),
        (
            "reconstructor-independent-guarantee",
            re.compile(r"reconstructor[ -]independent", re.IGNORECASE),
        ),
        (
            "certificate",
            re.compile(r"\bcertificat(?:e|ion|es|ed)\b", re.IGNORECASE),
        ),
        (
            "clinical-safety-guarantee",
            re.compile(r"clinical[ -]safety(?:\s+(?:claim|guarantee))?", re.IGNORECASE),
        ),
    )
}
STAGE_ONE_SUCCESS_HEADING = re.compile(
    r"^(?:#{1,6}[ \t]+|[-+*][ \t]+)?(?:[*_]{1,2})?"
    r"Success[ \t]+Criteria(?:[*_]{1,2})?[ \t]*:?[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)
MARKDOWN_HEADING = re.compile(r"^#{1,6}[ \t]+", re.MULTILINE)
STAGE_ONE_GENERATED_LABEL = re.compile(
    r"^(?:[-+*][ \t]+)?(?:[*_]{1,2})?Generated(?:[*_]{1,2})?[ \t]*:?",
    re.IGNORECASE | re.MULTILINE,
)
MARKDOWN_LIST_ITEM = re.compile(r"^[ \t]*(?:[-+*]|\d+[.)])[ \t]+(?P<body>.*)$")
STAGE_15_HARD_GATE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "ptb-xl-fold-10",
        re.compile(
            r"(?=.*(?:PTB[ -]?XL|primary))(?=.*fold[ -]?10)"
            r"(?=.*(?:Delta|Δ)[ \t]*R(?:\^?2|²))"
            r"(?=.*(?:confidence[ -]?interval|\bCI\b).*lower[ -]?bound.*(?:above|greater[ \t]+than)[ \t]+zero)",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "external-zero-transfer",
        re.compile(
            r"(?=.*Chapman)(?=.*zero[ -]transfer)"
            r"(?=.*(?:Delta|Δ)[ \t]*R(?:\^?2|²))"
            r"(?=.*(?:confidence[ -]?interval|\bCI\b).*lower[ -]?bound.*(?:above|greater[ \t]+than)[ \t]+zero)",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "three-of-four-reconstructors",
        re.compile(
            r"(?=.*(?:at[ \t]+least[ \t]+)?(?:three|3)[ \t]+of[ \t]+(?:the[ \t]+)?(?:four|4))"
            r"(?=.*reconstructor)(?=.*positive[ \t]+(?:Delta[ \t]*R(?:\^?2|²)[ \t]+)?point[ -]?estimate)",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
)
UNREGISTERED_STAGE_15_GATE_PATTERN = re.compile(
    r"consistent[ \t]+enough",
    re.IGNORECASE,
)
TO_STAGE_NUMBERS = {
    "TOPIC_INIT": 1,
    "PROBLEM_DECOMPOSE": 2,
    "SEARCH_STRATEGY": 3,
    "LITERATURE_COLLECT": 4,
    "LITERATURE_SCREEN": 5,
    "KNOWLEDGE_EXTRACT": 6,
    "SYNTHESIS": 7,
    "HYPOTHESIS_GEN": 8,
    "EXPERIMENT_DESIGN": 9,
    "CODE_GENERATION": 10,
    "RESOURCE_PLANNING": 11,
    "EXPERIMENT_RUN": 12,
    "ITERATIVE_REFINE": 13,
    "RESULT_ANALYSIS": 14,
    "RESEARCH_DECISION": 15,
    "PAPER_OUTLINE": 16,
    "PAPER_DRAFT": 17,
    "PEER_REVIEW": 18,
    "PAPER_REVISION": 19,
    "QUALITY_GATE": 20,
    "KNOWLEDGE_ARCHIVE": 21,
    "EXPORT_PUBLISH": 22,
    "CITATION_VERIFY": 23,
}
STAGE_NUMBER_NAMES = {number: name for name, number in TO_STAGE_NUMBERS.items()}
PROVIDER_ERROR_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("retry-limit", re.compile(r"exceeded\s+retry\s+limit", re.IGNORECASE)),
    ("http-429", re.compile(r"(?:last\s+status\s*:\s*)?429\s+too\s+many\s+requests", re.IGNORECASE)),
    ("authentication", re.compile(r"authentication\s+(?:is\s+)?required", re.IGNORECASE)),
    ("http-401", re.compile(r"401\s+unauthorized", re.IGNORECASE)),
    ("invalid-api-key", re.compile(r"invalid[_ -]+(?:api[_ -]?)?key", re.IGNORECASE)),
    ("rate-limit", re.compile(r"rate\s+limit(?:ed|\s+exceeded)", re.IGNORECASE)),
    ("service-unavailable", re.compile(r"service\s+unavailable", re.IGNORECASE)),
    ("upstream-error", re.compile(r"upstream\s+(?:connect\s+)?error", re.IGNORECASE)),
    ("internal-server-error", re.compile(r"internal\s+server\s+error", re.IGNORECASE)),
    ("queue-owner-disconnected", re.compile(r"queue\s+owner\s+disconnected", re.IGNORECASE)),
    ("quota-exhausted", re.compile(r"(?:quota|resource)\s+(?:is\s+)?exhausted", re.IGNORECASE)),
    (
        "unsupported-runtime-value",
        re.compile(r"unsupported\s+value.*model_reasoning_effort", re.IGNORECASE | re.DOTALL),
    ),
    ("unsupported-model", re.compile(r"model.{0,120}(?:not|isn't)\s+supported", re.IGNORECASE)),
)


@dataclass(frozen=True)
class Violation:
    """A fail-closed stage validation failure."""

    code: str
    detail: str
    stage: int | None = None
    artifact: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "detail": self.detail,
            "stage": self.stage,
            "artifact": self.artifact,
        }


@dataclass(frozen=True)
class ProcessIdentity:
    """PID plus a creation token, preventing PID-reuse collateral damage."""

    pid: int
    parent_pid: int
    start_token: str


@dataclass
class ProcessTreeTracker:
    """Accumulate the exact descendants observed beneath one launched child."""

    root_pid: int
    identities: dict[int, ProcessIdentity]
    captured_parent: dict[int, int]

    @classmethod
    def start(cls, root_pid: int) -> "ProcessTreeTracker":
        table = _current_process_table()
        root = table.get(root_pid)
        if root is None:
            raise RuntimeError(f"cannot identify launched ARC process {root_pid}")
        tracker = cls(
            root_pid=root_pid,
            identities={root_pid: root},
            captured_parent={root_pid: root.parent_pid},
        )
        tracker.refresh(table)
        return tracker

    def refresh(
        self, table: dict[int, ProcessIdentity] | None = None
    ) -> dict[int, ProcessIdentity]:
        """Capture new descendants while retaining children that later orphan."""

        current = _current_process_table() if table is None else table
        live_tracked = {
            pid
            for pid, identity in self.identities.items()
            if _same_process(current.get(pid), identity)
        }
        changed = True
        while changed:
            changed = False
            for pid, identity in current.items():
                if pid in self.identities or identity.parent_pid not in live_tracked:
                    continue
                self.identities[pid] = identity
                self.captured_parent[pid] = identity.parent_pid
                live_tracked.add(pid)
                changed = True
        return current

    def leaf_first(self, table: dict[int, ProcessIdentity]) -> list[ProcessIdentity]:
        live = {
            pid: identity
            for pid, identity in self.identities.items()
            if _same_process(table.get(pid), identity)
        }

        def depth(pid: int) -> int:
            seen: set[int] = set()
            value = 0
            while pid != self.root_pid and pid not in seen:
                seen.add(pid)
                parent = self.captured_parent.get(pid)
                if parent is None or parent not in self.identities:
                    break
                value += 1
                pid = parent
            return value

        return sorted(live.values(), key=lambda item: (depth(item.pid), item.pid), reverse=True)


@dataclass(frozen=True)
class TerminationReport:
    """Auditable result of exact-identity, leaf-first process cleanup."""

    targeted_leaf_first: tuple[int, ...]
    terminated: tuple[int, ...]
    identity_mismatch_skipped: tuple[int, ...]
    still_alive: tuple[int, ...]

    def as_dict(self) -> dict[str, list[int]]:
        return {
            "targeted_leaf_first": list(self.targeted_leaf_first),
            "terminated": list(self.terminated),
            "identity_mismatch_skipped": list(self.identity_mismatch_skipped),
            "still_alive": list(self.still_alive),
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _exclusive_json(path: Path, value: dict[str, Any]) -> None:
    """Write an additive evidence record without replacing prior evidence."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _git_commit(checkout: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip()


def _git_bytes(project_root: Path, arguments: list[str]) -> bytes:
    result = subprocess.run(
        ["git", *arguments],
        cwd=project_root,
        check=True,
        capture_output=True,
    )
    return result.stdout


def _require_clean_git_worktree(repository: Path, *, label: str) -> None:
    """Reject staged, unstaged, or untracked non-ignored checkout content."""

    status = _git_bytes(
        repository,
        ["status", "--porcelain=v1", "-z", "--untracked-files=all"],
    )
    if status:
        raise ValueError(f"{label} must be a clean Git worktree")


def _require_clean_project_state(state: dict[str, Any]) -> None:
    """Make a dirty bound project a launch error, not merely recorded metadata."""

    if state.get("project_git_dirty") is not False:
        raise ValueError("bound project repository must be a clean Git worktree")


def _project_state(
    project_root: Path,
    *,
    source_config: Path,
    effective_loaded: dict[str, Any],
) -> dict[str, Any]:
    """Fingerprint the bound repository state and referenced prompt inputs."""

    head = _git_bytes(project_root, ["rev-parse", "HEAD"]).decode("ascii").strip()
    tracked_diff = _git_bytes(
        project_root,
        ["diff", "--no-ext-diff", "--binary", "HEAD", "--"],
    )
    untracked_raw = _git_bytes(
        project_root,
        ["ls-files", "--others", "--exclude-standard", "-z"],
    )
    untracked: list[dict[str, str]] = []
    for raw_name in sorted(item for item in untracked_raw.split(b"\0") if item):
        relative = raw_name.decode("utf-8", errors="surrogateescape")
        candidate = project_root / relative
        untracked.append({"path": relative, "sha256": _sha256(candidate)})

    prompts = effective_loaded.get("prompts", {})
    prompt_paths: set[Path] = set()
    if isinstance(prompts, dict):
        custom = prompts.get("custom_file")
        if isinstance(custom, str) and custom:
            prompt_paths.add(Path(custom))
        extras = prompts.get("extra_prompts", {})
        if isinstance(extras, dict):
            prompt_paths.update(
                Path(value)
                for value in extras.values()
                if isinstance(value, str) and value
            )
    bound_files = [source_config, *sorted(prompt_paths, key=lambda path: str(path))]
    record: dict[str, Any] = {
        "project_git_commit": head,
        "project_git_dirty": bool(tracked_diff or untracked),
        "tracked_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
        "untracked_files": untracked,
        "bound_files": [
            {"path": str(path), "sha256": _sha256(path)} for path in bound_files
        ],
    }
    canonical = json.dumps(
        record,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    record["state_sha256"] = hashlib.sha256(canonical).hexdigest()
    return record


def _write_or_verify_project_state(
    path: Path,
    state: dict[str, Any],
    *,
    resume: bool,
) -> None:
    if resume:
        if not path.is_file():
            raise ValueError("--resume requires control/project-state.v1.json")
        observed = _read_json(path)
        if observed != state:
            raise ValueError("bound project/config/protocol inputs drifted since original run")
        return
    _exclusive_json(path, state)


def _linux_process_table() -> dict[int, ProcessIdentity]:
    table: dict[int, ProcessIdentity] = {}
    proc_root = Path("/proc")
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "stat").read_text(encoding="ascii")
            # comm is parenthesized and may itself contain spaces or ')'.
            fields = raw[raw.rfind(")") + 2 :].split()
            pid = int(entry.name)
            table[pid] = ProcessIdentity(
                pid=pid,
                parent_pid=int(fields[1]),
                start_token=fields[19],
            )
        except (FileNotFoundError, PermissionError, IndexError, ValueError):
            continue
    return table


def _windows_process_table() -> dict[int, ProcessIdentity]:
    """Enumerate PID/PPID in-process, without spawning PowerShell each poll."""

    import ctypes
    from ctypes import wintypes

    class ProcessEntry32W(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ctypes.c_size_t),
            ("th32ModuleID", wintypes.DWORD),
            ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", wintypes.LONG),
            ("dwFlags", wintypes.DWORD),
            ("szExeFile", wintypes.WCHAR * 260),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel32.Process32FirstW.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ProcessEntry32W),
    ]
    kernel32.Process32FirstW.restype = wintypes.BOOL
    kernel32.Process32NextW.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ProcessEntry32W),
    ]
    kernel32.Process32NextW.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    snapshot = kernel32.CreateToolhelp32Snapshot(0x00000002, 0)
    invalid_handle = ctypes.c_void_p(-1).value
    if snapshot == invalid_handle:
        raise ctypes.WinError(ctypes.get_last_error())
    table: dict[int, ProcessIdentity] = {}
    try:
        entry = ProcessEntry32W()
        entry.dwSize = ctypes.sizeof(ProcessEntry32W)
        available = kernel32.Process32FirstW(snapshot, ctypes.byref(entry))
        while available:
            pid = int(entry.th32ProcessID)
            token = _windows_start_token(pid)
            if token is not None:
                table[pid] = ProcessIdentity(
                    pid=pid,
                    parent_pid=int(entry.th32ParentProcessID),
                    start_token=token,
                )
            available = kernel32.Process32NextW(snapshot, ctypes.byref(entry))
    finally:
        kernel32.CloseHandle(snapshot)
    if not table:
        raise RuntimeError("failed to obtain Windows process identity table")
    return table


def _windows_start_token(pid: int) -> str | None:
    """Read the kernel creation FILETIME without spawning another process."""

    import ctypes
    from ctypes import wintypes

    process_query_limited_information = 0x1000
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.GetProcessTimes.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
    ]
    kernel32.GetProcessTimes.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
    if not handle:
        return None
    creation = wintypes.FILETIME()
    exit_time = wintypes.FILETIME()
    kernel_time = wintypes.FILETIME()
    user_time = wintypes.FILETIME()
    try:
        ok = kernel32.GetProcessTimes(
            handle,
            ctypes.byref(creation),
            ctypes.byref(exit_time),
            ctypes.byref(kernel_time),
            ctypes.byref(user_time),
        )
        if not ok:
            return None
        value = (creation.dwHighDateTime << 32) | creation.dwLowDateTime
        return str(value)
    finally:
        kernel32.CloseHandle(handle)


def _posix_ps_process_table() -> dict[int, ProcessIdentity]:
    result = subprocess.run(
        ["ps", "-axo", "pid=,ppid=,lstart="],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=15,
    )
    table: dict[int, ProcessIdentity] = {}
    for line in result.stdout.splitlines():
        fields = line.split(maxsplit=2)
        if len(fields) != 3:
            continue
        try:
            pid, parent_pid = int(fields[0]), int(fields[1])
        except ValueError:
            continue
        table[pid] = ProcessIdentity(pid, parent_pid, fields[2])
    return table


def _current_process_table() -> dict[int, ProcessIdentity]:
    if os.name == "nt":
        for attempt in range(5):
            try:
                return _windows_process_table()
            except PermissionError:
                if attempt == 4:
                    raise
                # The Toolhelp snapshot is read-only.  Windows can
                # transiently deny snapshot creation while process state is
                # changing; retry only this query with bounded backoff.
                time.sleep(0.05 * (attempt + 1))
        raise AssertionError("unreachable")
    if Path("/proc").is_dir():
        return _linux_process_table()
    return _posix_ps_process_table()


def _current_process_identity(pid: int) -> ProcessIdentity | None:
    if os.name == "nt":
        token = _windows_start_token(pid)
        return ProcessIdentity(pid, 0, token) if token is not None else None
    if Path("/proc").is_dir():
        try:
            raw = (Path("/proc") / str(pid) / "stat").read_text(encoding="ascii")
            fields = raw[raw.rfind(")") + 2 :].split()
            return ProcessIdentity(pid, int(fields[1]), fields[19])
        except (FileNotFoundError, PermissionError, IndexError, ValueError):
            return None
    return _posix_ps_process_table().get(pid)


def _same_process(
    current: ProcessIdentity | None, captured: ProcessIdentity
) -> bool:
    """Compare stable identity fields; parent PID may change after orphaning."""

    return (
        current is not None
        and current.pid == captured.pid
        and current.start_token == captured.start_token
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arc-checkout", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--acpx-command", type=Path, required=True)
    parser.add_argument("--codex-home", type=Path, required=True)
    parser.add_argument("--acp-model", required=True)
    parser.add_argument(
        "--acp-reasoning-effort",
        choices=("low", "medium", "high", "xhigh"),
        default="high",
    )
    parser.add_argument("--acp-provider-id", default="ecgcert-gateway")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--receipt-root",
        type=Path,
        required=True,
        help=(
            "external control directory receiving waiting/final receipt bundles "
            "and authenticated DAG operator responses"
        ),
    )
    parser.add_argument("--to-stage", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=0.25)
    return parser


def _build_arc_command(
    *,
    python: Path,
    effective_config: Path,
    output: Path,
    to_stage: str,
    resume: bool,
) -> list[str]:
    if resume:
        raise ValueError(
            "official ARC v0.5.0 resume/from-stage creates a new run_id and "
            "HITL session_id; restart from an empty directory instead"
        )
    command = [
        str(python),
        "-m",
        "researchclaw.cli",
        "run",
        "--config",
        str(effective_config),
        "--output",
        str(output),
        "--to-stage",
        to_stage,
    ]
    return command


def _control_timestamp(value: Any, *, field: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field} must be a valid ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _build_operator_challenge(
    *, output: Path, waiting: dict[str, Any], waiting_sha256: str
) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{64}", waiting_sha256):
        raise ValueError("waiting SHA-256 must be a full lowercase digest")
    stage = waiting.get("stage")
    if isinstance(stage, bool) or not isinstance(stage, int) or stage not in REQUIRED_GATES:
        raise ValueError("operator challenges are issued only at registered ARC gates")
    waiting_since = _control_timestamp(waiting.get("since"), field="waiting.since")
    session = _read_json(output / "hitl" / "session.json")
    run_id = session.get("run_id")
    session_id = session.get("session_id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("native HITL session lacks a run_id")
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("native HITL session lacks a session_id")
    checkpoint = _checkpoint_binding_for_gate(output, stage)
    return {
        "schema_version": CHALLENGE_SCHEMA,
        "created_at": _utc_now(),
        "stage": stage,
        "run_id": run_id,
        "session_id": session_id,
        "waiting_sha256": waiting_sha256,
        "waiting_since": waiting_since.isoformat(),
        "expires_at": (waiting_since + timedelta(hours=24)).isoformat(),
        "preapproval_checkpoint_sha256": checkpoint["sha256"],
        "nonce": secrets.token_hex(32),
        "available_actions": list(waiting.get("available_actions", [])),
    }


def _validate_response(
    response: dict[str, Any],
    waiting: dict[str, Any],
    challenge: dict[str, Any],
) -> None:
    expected = {
        "schema_version",
        "stage",
        "run_id",
        "session_id",
        "waiting_sha256",
        "preapproval_checkpoint_sha256",
        "nonce",
        "action",
        "issued_at",
        "message",
    }
    if set(response) != expected:
        raise ValueError("operator response has missing or unexpected fields")
    if response["schema_version"] != RESPONSE_SCHEMA:
        raise ValueError(f"operator response schema must be {RESPONSE_SCHEMA}")
    if response["stage"] != waiting.get("stage") or response["stage"] != challenge.get(
        "stage"
    ):
        raise ValueError("operator response does not match the waiting ARC stage")
    for field in (
        "run_id",
        "session_id",
        "waiting_sha256",
        "preapproval_checkpoint_sha256",
        "nonce",
    ):
        if response[field] != challenge.get(field):
            raise ValueError(f"operator response {field} does not match the active challenge")
    for field in ("waiting_sha256", "preapproval_checkpoint_sha256", "nonce"):
        if not isinstance(response[field], str) or not re.fullmatch(
            r"[0-9a-f]{64}", response[field]
        ):
            raise ValueError(f"operator response {field} must be a full lowercase digest")
    for field in ("run_id", "session_id"):
        if not isinstance(response[field], str) or not response[field].strip():
            raise ValueError(f"operator response {field} must be non-empty")
    if response["action"] not in {"approve", "reject", "abort"}:
        raise ValueError("operator response action must be approve, reject, or abort")
    if not isinstance(response["message"], str):
        raise ValueError("operator response message must be a string")
    issued = _control_timestamp(response["issued_at"], field="operator response issued_at")
    waiting_since = _control_timestamp(waiting.get("since"), field="waiting.since")
    challenge_since = _control_timestamp(
        challenge.get("waiting_since"), field="challenge.waiting_since"
    )
    if waiting_since != challenge_since:
        raise ValueError("active waiting timestamp changed after the challenge was issued")
    if issued < waiting_since or issued > waiting_since + timedelta(hours=24):
        raise ValueError(
            "operator response issued_at lies outside the waiting.since-to-24h window"
        )
    available = waiting.get("available_actions", [])
    if response["action"] not in available:
        raise ValueError("operator response action is unavailable at this ARC pause")


def _response_consumption_path(output: Path, challenge: dict[str, Any]) -> Path:
    return (
        output
        / "control"
        / "operator-response-consumption"
        / f"stage-{int(challenge['stage']):02d}-{challenge['waiting_sha256']}.v1.json"
    )


def _operator_response_snapshot_path(
    output: Path, *, stage: int, response_sha256: str
) -> Path:
    return (
        output
        / "control"
        / "operator-response-snapshots"
        / f"stage-{stage:02d}-{response_sha256}.v2.json"
    )


def _preapproval_checkpoint_snapshot_path(
    output: Path, *, stage: int, checkpoint_sha256: str
) -> Path:
    return (
        output
        / "control"
        / "preapproval-checkpoints"
        / f"stage-{stage:02d}-{checkpoint_sha256}.json"
    )


def _exclusive_snapshot(
    source: Path, destination: Path, *, expected_sha256: str, label: str
) -> Path:
    """Copy immutable evidence once without ever replacing an existing record."""

    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError(f"{label} SHA-256 must be a full lowercase digest")
    resolved = source.resolve(strict=True)
    if source.is_symlink() or not resolved.is_file():
        raise ValueError(f"{label} source must be a non-symlink regular file")
    if _sha256(resolved) != expected_sha256:
        raise ValueError(f"{label} source hash changed before snapshot")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with resolved.open("rb") as reader, destination.open("xb") as writer:
            shutil.copyfileobj(reader, writer, length=1 << 20)
            writer.flush()
            os.fsync(writer.fileno())
    except FileExistsError:
        if destination.is_symlink() or not destination.is_file():
            raise ValueError(f"existing {label} snapshot is not a regular file") from None
        if _sha256(destination) != expected_sha256:
            raise ValueError(f"existing {label} snapshot has conflicting content") from None
    if _sha256(destination) != expected_sha256:
        raise ValueError(f"{label} snapshot hash mismatch")
    if _sha256(resolved) != expected_sha256:
        raise ValueError(f"{label} source changed while it was snapshotted")
    return destination


def _snapshot_operator_response(
    *, output: Path, response_path: Path, stage: int, response_sha256: str
) -> Path:
    destination = _operator_response_snapshot_path(
        output, stage=stage, response_sha256=response_sha256
    )
    return _exclusive_snapshot(
        response_path,
        destination,
        expected_sha256=response_sha256,
        label="operator response",
    )


def _snapshot_preapproval_checkpoint(
    *, output: Path, stage: int, checkpoint_binding: dict[str, Any]
) -> Path:
    checkpoint_sha256 = checkpoint_binding.get("sha256")
    if checkpoint_binding.get("path") != "checkpoint.json" or not isinstance(
        checkpoint_sha256, str
    ):
        raise ValueError("pre-approval checkpoint binding is malformed")
    destination = _preapproval_checkpoint_snapshot_path(
        output,
        stage=stage,
        checkpoint_sha256=checkpoint_sha256,
    )
    return _exclusive_snapshot(
        output / "checkpoint.json",
        destination,
        expected_sha256=checkpoint_sha256,
        label="pre-approval checkpoint",
    )


def _claim_operator_response(
    *,
    output: Path,
    response_path: Path,
    response: dict[str, Any],
    response_sha256: str,
    challenge: dict[str, Any],
) -> Path:
    if not re.fullmatch(r"[0-9a-f]{64}", response_sha256):
        raise ValueError("response SHA-256 must be a full lowercase digest")
    claim_path = _response_consumption_path(output, challenge)
    try:
        response_relative = response_path.resolve().relative_to(output.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError("operator response path escapes the ARC run") from exc
    expected_response_path = _operator_response_snapshot_path(
        output,
        stage=int(challenge["stage"]),
        response_sha256=response_sha256,
    ).relative_to(output).as_posix()
    if response_relative != expected_response_path:
        raise ValueError("operator response must be claimed from its immutable snapshot")
    if (
        response_path.is_symlink()
        or not response_path.is_file()
        or _sha256(response_path) != response_sha256
        or _read_json(response_path) != response
    ):
        raise ValueError("operator response snapshot is missing, changed, or inconsistent")
    checkpoint_snapshot = _preapproval_checkpoint_snapshot_path(
        output,
        stage=int(challenge["stage"]),
        checkpoint_sha256=str(challenge["preapproval_checkpoint_sha256"]),
    )
    if (
        not checkpoint_snapshot.is_file()
        or checkpoint_snapshot.is_symlink()
        or _sha256(checkpoint_snapshot)
        != challenge["preapproval_checkpoint_sha256"]
    ):
        raise ValueError("pre-approval checkpoint snapshot is missing or changed")
    record = {
        "schema_version": CONSUMPTION_SCHEMA,
        "claimed_at": _utc_now(),
        "response_path": response_relative,
        "response_sha256": response_sha256,
        "stage": challenge["stage"],
        "run_id": challenge["run_id"],
        "session_id": challenge["session_id"],
        "waiting_sha256": challenge["waiting_sha256"],
        "preapproval_checkpoint_sha256": challenge[
            "preapproval_checkpoint_sha256"
        ],
        "nonce": challenge["nonce"],
        "response": response,
    }
    try:
        _exclusive_json(claim_path, record)
    except FileExistsError as exc:
        raise ValueError("operator response replay: this ARC pause was already consumed") from exc
    return claim_path


def _retire_operator_response(response_path: Path, expected_sha256: str) -> None:
    """Keep the authenticated DAG output immutable after one-time consumption."""

    if _sha256(response_path) != expected_sha256:
        raise ValueError("operator response changed while it was being consumed")


def _assert_empty_new_output(output: Path, *, resume: bool) -> None:
    if resume:
        if not output.is_dir():
            raise ValueError("--resume requires an existing ARC output directory")
        if (output / "run-invalidation.v1.json").exists():
            raise ValueError("an invalidated ARC run cannot be resumed")
        for required in ("checkpoint.json", "config.yaml"):
            if not (output / required).is_file():
                raise ValueError(f"--resume requires {required}")
        return
    if output.exists():
        if not output.is_dir():
            raise ValueError("new ARC output path exists and is not a directory")
        if any(output.iterdir()):
            raise ValueError("new ARC output directory must be completely empty")


def _assert_external_output(output: Path, project_root: Path) -> None:
    """Keep ARC's mutable control, KB, and run artifacts outside the checkout."""

    resolved_output = output.resolve()
    try:
        resolved_output.relative_to(project_root.resolve())
    except ValueError:
        return
    raise ValueError("ARC output must be outside the project repository")


def _target_stage_number(value: str) -> int:
    normalized = value.strip().upper().replace("-", "_")
    if normalized.isdigit():
        stage = int(normalized)
    else:
        try:
            stage = TO_STAGE_NUMBERS[normalized]
        except KeyError as exc:
            raise ValueError(f"unknown ARC target stage: {value}") from exc
    if stage not in range(1, 24):
        raise ValueError("ARC target stage must be between 1 and 23")
    return stage


def _validate_continuous_target(stage: int) -> None:
    if stage < 20:
        raise ValueError(
            "the single native ARC process must target Stage 20 or later so "
            "Stages 5, 9, 15, and 20 share one run_id/session_id"
        )


def _build_codex_acp_config(
    *,
    base_url: str,
    model: str,
    reasoning_effort: str,
    provider_id: str,
) -> tuple[dict[str, Any], str]:
    """Build a secret-free Codex config for an env-key custom provider."""

    parsed = urlsplit(base_url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("OPENAI_BASE_URL must be an HTTPS URL without userinfo")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", model):
        raise ValueError("ACP model contains unsupported characters")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", provider_id):
        raise ValueError("ACP provider id contains unsupported characters")
    if reasoning_effort not in {"low", "medium", "high", "xhigh"}:
        raise ValueError("unsupported ACP reasoning effort")
    config = {
        "model": model,
        "model_reasoning_effort": reasoning_effort,
        "model_provider": provider_id,
        "model_providers": {
            provider_id: {
                "name": "ECGCert ARC gateway",
                "base_url": base_url.rstrip("/"),
                "env_key": "OPENAI_API_KEY",
                "wire_api": "responses",
            }
        },
        "sandbox_mode": "read-only",
    }
    serialized = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return config, serialized


def _absolute_prompt_path(value: str, project_root: Path) -> str:
    if not value:
        return value
    path = Path(value)
    if not path.is_absolute():
        path = project_root / path
    return str(path.resolve(strict=True))


def _prepare_effective_config(
    *,
    source_config: Path,
    output: Path,
    project_root: Path,
    python: Path,
    acpx_command: Path,
) -> tuple[dict[str, Any], bytes]:
    """Create a run-specific config with every local writable root isolated."""

    loaded = yaml.safe_load(source_config.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("ARC source config must be a YAML object")
    hitl = loaded.get("hitl")
    if not isinstance(hitl, dict) or hitl.get("enabled") is not True:
        raise ValueError("ARC source config must enable HITL")
    if hitl.get("mode") != "co-pilot":
        raise ValueError("ARC source config must use co-pilot HITL mode")
    policies = hitl.get("stage_policies")
    if not isinstance(policies, dict):
        raise ValueError("ARC source config must declare stage_policies")
    approval_stages = {
        int(stage)
        for stage, policy in policies.items()
        if isinstance(policy, dict) and policy.get("require_approval") is True
    }
    if approval_stages != REQUIRED_GATES:
        raise ValueError(
            "ARC source config approval stages must be exactly 5, 9, 15, and 20"
        )
    for stage in sorted(REQUIRED_GATES):
        policy = policies.get(stage, policies.get(str(stage)))
        if not isinstance(policy, dict) or not (
            policy.get("pause_before") is False
            and policy.get("pause_after") is True
            and policy.get("require_approval") is True
        ):
            raise ValueError(
                "ARC gates 5, 9, 15, and 20 must each use exactly one "
                "post-artifact review: pause_before=false, pause_after=true, "
                "require_approval=true"
            )

    knowledge_base = loaded.setdefault("knowledge_base", {})
    if not isinstance(knowledge_base, dict):
        raise ValueError("knowledge_base must be a YAML object")
    knowledge_base["root"] = str((output / "kb").resolve())

    llm = loaded.setdefault("llm", {})
    if not isinstance(llm, dict):
        raise ValueError("llm must be a YAML object")
    acp = llm.setdefault("acp", {})
    if not isinstance(acp, dict):
        raise ValueError("llm.acp must be a YAML object")
    acp["acpx_command"] = str(acpx_command)
    acp["cwd"] = str(project_root)

    experiment = loaded.setdefault("experiment", {})
    if not isinstance(experiment, dict):
        raise ValueError("experiment must be a YAML object")
    if experiment.get("mode") != "sandbox":
        raise ValueError(
            "ARC must use the local sandbox control plane; claim-bearing remote "
            "execution belongs only to the authenticated native DAG"
        )
    sandbox = experiment.setdefault("sandbox", {})
    if not isinstance(sandbox, dict):
        raise ValueError("experiment.sandbox must be a YAML object")
    sandbox["python_path"] = str(python)

    prompts = loaded.setdefault("prompts", {})
    if not isinstance(prompts, dict):
        raise ValueError("prompts must be a YAML object")
    custom_file = prompts.get("custom_file")
    if custom_file:
        if not isinstance(custom_file, str):
            raise ValueError("prompts.custom_file must be a string")
        prompts["custom_file"] = _absolute_prompt_path(custom_file, project_root)
    extras = prompts.get("extra_prompts", {})
    if not isinstance(extras, dict):
        raise ValueError("prompts.extra_prompts must be a YAML object")
    for name, value in list(extras.items()):
        if not isinstance(value, str):
            raise ValueError(f"prompts.extra_prompts.{name} must be a string")
        extras[name] = _absolute_prompt_path(value, project_root)

    session_base = str(acp.get("session_name") or "researchclaw")
    run_suffix = hashlib.sha256(str(output.resolve()).encode("utf-8")).hexdigest()[:12]
    acp["session_name"] = f"{session_base}-{run_suffix}"

    payload = yaml.safe_dump(loaded, sort_keys=False, allow_unicode=True).encode("utf-8")
    return loaded, payload


def _write_or_verify_source_snapshot(path: Path, payload: bytes, *, resume: bool) -> None:
    if resume:
        if not path.is_file():
            raise ValueError("--resume requires control/source-config.yaml")
        if path.read_bytes() != payload:
            raise ValueError("source ARC configuration drifted since the original run")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)


def _write_or_verify_effective_config(path: Path, payload: bytes, *, resume: bool) -> None:
    if resume:
        if not path.is_file():
            raise ValueError("--resume requires control/effective-config.yaml")
        if path.read_bytes() != payload:
            raise ValueError("effective ARC configuration drifted since the original run")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)


def _stage_number(stage_dir: Path) -> int | None:
    match = re.fullmatch(r"stage-(\d+)", stage_dir.name)
    return int(match.group(1)) if match else None


def _load_stage_json(path: Path, *, stage: int, kind: str) -> tuple[dict[str, Any] | None, Violation | None]:
    try:
        return _read_json(path), None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        return None, Violation(
            code=f"malformed-{kind}",
            detail=f"{kind} is not a readable JSON object: {exc}",
            stage=stage,
            artifact=str(path),
        )


def _validate_text_artifact(
    path: Path,
    *,
    stage: int,
    relative_name: str,
    minimum_bytes: int = MIN_TEXT_ARTIFACT_BYTES,
) -> list[Violation]:
    size = path.stat().st_size
    violations: list[Violation] = []
    if size < minimum_bytes:
        violations.append(
            Violation(
                code="artifact-too-short",
                detail=(
                    f"declared text artifact is only {size} bytes; "
                    f"minimum is {minimum_bytes}"
                ),
                stage=stage,
                artifact=relative_name,
            )
        )
    if size > MAX_SCANNED_ARTIFACT_BYTES:
        violations.append(
            Violation(
                code="artifact-too-large-to-validate",
                detail=f"declared text artifact exceeds {MAX_SCANNED_ARTIFACT_BYTES} bytes",
                stage=stage,
                artifact=relative_name,
            )
        )
        return violations
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        violations.append(
            Violation(
                code="artifact-not-utf8",
                detail=str(exc),
                stage=stage,
                artifact=relative_name,
            )
        )
        return violations
    for code, pattern in PROVIDER_ERROR_PATTERNS:
        if pattern.search(text):
            violations.append(
                Violation(
                    code=f"provider-error-{code}",
                    detail="declared artifact contains a provider/runtime error marker",
                    stage=stage,
                    artifact=relative_name,
                )
            )
            break
    semantic_key = (stage, relative_name)
    # Ignore Markdown emphasis/code delimiters while retaining the authored
    # words and numbers that define the frozen protocol.
    semantic_text = re.sub(r"[*_`]", "", text)
    for label, pattern in PROTOCOL_REQUIRED_PATTERNS.get(semantic_key, ()):
        if not pattern.search(semantic_text):
            violations.append(
                Violation(
                    code=f"protocol-required-anchor-missing-{label}",
                    detail="scientific core artifact does not bind a frozen-protocol anchor",
                    stage=stage,
                    artifact=relative_name,
                )
            )
    for label, pattern in PROTOCOL_FORBIDDEN_PATTERNS.get(semantic_key, ()):
        if pattern.search(semantic_text):
            violations.append(
                Violation(
                    code=f"protocol-forbidden-claim-{label}",
                    detail="scientific core artifact contradicts the frozen claim boundary",
                    stage=stage,
                    artifact=relative_name,
                )
            )
    if semantic_key == (1, "goal.md"):
        violations.extend(_validate_stage_one_success_criteria(text, relative_name))
    try:
        if path.suffix.lower() == ".json":
            parsed = json.loads(text)
            if parsed in ({}, []):
                raise ValueError("JSON artifact is an empty object or array")
        elif path.suffix.lower() == ".jsonl":
            records = [line for line in text.splitlines() if line.strip()]
            if not records:
                raise ValueError("JSONL artifact has no records")
            for line_number, line in enumerate(records, start=1):
                parsed = json.loads(line)
                if parsed in ({}, []):
                    raise ValueError(f"JSONL record {line_number} is empty")
        elif path.suffix.lower() in {".yaml", ".yml"}:
            parsed = yaml.safe_load(text)
            if parsed in (None, {}, []):
                raise ValueError("YAML artifact is empty")
    except (json.JSONDecodeError, yaml.YAMLError, ValueError) as exc:
        violations.append(
            Violation(
                code="artifact-parse-error",
                detail=str(exc),
                stage=stage,
                artifact=relative_name,
            )
        )
    return violations


def _markdown_list_items(section: str) -> list[str]:
    """Return top-level Markdown list items with their continuation lines."""

    items: list[list[str]] = []
    current: list[str] | None = None
    for line in section.splitlines():
        match = MARKDOWN_LIST_ITEM.match(line)
        if match:
            if current is not None:
                items.append(current)
            current = [match.group("body").strip()]
        elif current is not None and line.strip():
            current.append(line.strip())
    if current is not None:
        items.append(current)
    return [" ".join(parts) for parts in items]


def _validate_stage_one_success_criteria(
    text: str,
    relative_name: str,
) -> list[Violation]:
    """Enforce the frozen Stage 15 decision rule already at Stage 1.

    There are exactly three hard gates.  Wave-window consistency and other
    sensitivity analyses may inform interpretation, but cannot become a
    fourth condition for PROCEED/PIVOT.
    """

    heading = STAGE_ONE_SUCCESS_HEADING.search(text)
    if heading is None:
        return [
            Violation(
                code="protocol-stage15-gates-missing-section",
                detail="goal.md must contain a Success Criteria section with exactly three hard gates",
                stage=1,
                artifact=relative_name,
            )
        ]
    remainder = text[heading.end() :]
    boundaries = [
        match.start()
        for pattern in (MARKDOWN_HEADING, STAGE_ONE_GENERATED_LABEL)
        if (match := pattern.search(remainder)) is not None
    ]
    section = remainder[: min(boundaries)] if boundaries else remainder
    items = _markdown_list_items(section)
    violations: list[Violation] = []
    if len(items) != len(STAGE_15_HARD_GATE_PATTERNS):
        violations.append(
            Violation(
                code="protocol-stage15-gate-count",
                detail=(
                    "Success Criteria must contain exactly three list items, one for each "
                    f"registered hard gate; found {len(items)}"
                ),
                stage=1,
                artifact=relative_name,
            )
        )

    matched_item_indexes: set[int] = set()
    for label, pattern in STAGE_15_HARD_GATE_PATTERNS:
        matching = [
            index
            for index, item in enumerate(items)
            if pattern.search(re.sub(r"[*_`]", "", item))
        ]
        if len(matching) != 1:
            violations.append(
                Violation(
                    code=f"protocol-stage15-gate-missing-{label}",
                    detail="Success Criteria does not state this registered hard gate exactly once",
                    stage=1,
                    artifact=relative_name,
                )
            )
        matched_item_indexes.update(matching)

    unregistered = [
        item for index, item in enumerate(items) if index not in matched_item_indexes
    ]
    if unregistered or UNREGISTERED_STAGE_15_GATE_PATTERN.search(section):
        violations.append(
            Violation(
                code="protocol-unregistered-stage15-gate",
                detail=(
                    "Success Criteria adds an unregistered publishability condition; "
                    "QRS/ST/T consistency is sensitivity evidence only"
                ),
                stage=1,
                artifact=relative_name,
            )
        )
    if not re.search(r"\bPIVOT\b", section, re.IGNORECASE):
        violations.append(
            Violation(
                code="protocol-stage15-pivot-policy-missing",
                detail="Success Criteria must state the frozen failure-to-PIVOT policy",
                stage=1,
                artifact=relative_name,
            )
        )
    return violations


def _validate_declared_artifact(
    candidate: Path,
    *,
    stage: int,
    relative_name: str,
    minimum_bytes: int,
) -> list[Violation]:
    violations: list[Violation] = []
    if candidate.is_dir():
        files = sorted(path for path in candidate.rglob("*") if path.is_file())
        if not files:
            return [
                Violation(
                    code="artifact-directory-empty",
                    detail="declared output directory contains no regular files",
                    stage=stage,
                    artifact=relative_name,
                )
            ]
        for child in files:
            if child.suffix.lower() in TEXT_SUFFIXES:
                child_name = child.relative_to(candidate.parent).as_posix()
                violations.extend(
                    _validate_text_artifact(
                        child,
                        stage=stage,
                        relative_name=child_name,
                    )
                )
        return violations
    if not candidate.is_file():
        return [
            Violation(
                code="artifact-missing",
                detail="declared output artifact is neither a regular file nor directory",
                stage=stage,
                artifact=relative_name,
            )
        ]
    if candidate.stat().st_size == 0:
        return [
            Violation(
                code="artifact-empty",
                detail="declared output artifact is empty",
                stage=stage,
                artifact=relative_name,
            )
        ]
    if candidate.suffix.lower() in TEXT_SUFFIXES:
        violations.extend(
            _validate_text_artifact(
                candidate,
                stage=stage,
                relative_name=relative_name,
                minimum_bytes=minimum_bytes,
            )
        )
    return violations


def _validate_completed_stage(stage_dir: Path) -> tuple[bool, list[Violation]]:
    """Validate a stage once both native completion records are present.

    Returns ``(ready, violations)``.  ``ready=False`` means ARC has not yet
    emitted both native records; it is not itself an error while the child is
    still running.
    """

    stage = _stage_number(stage_dir)
    if stage is None:
        return False, []
    decision_path = stage_dir / "decision.json"
    health_path = stage_dir / "stage_health.json"
    if not decision_path.is_file() or not health_path.is_file():
        return False, []
    decision, error = _load_stage_json(decision_path, stage=stage, kind="decision")
    if error is not None:
        return True, [error]
    health, error = _load_stage_json(health_path, stage=stage, kind="stage-health")
    if error is not None:
        return True, [error]
    assert decision is not None and health is not None

    violations: list[Violation] = []
    pending_gate = (
        stage in REQUIRED_GATES
        and decision.get("status") == "blocked_approval"
        and decision.get("decision") == "block"
        and health.get("status") == "blocked_approval"
    )
    if not pending_gate and (
        decision.get("status") != "done" or decision.get("decision") != "proceed"
    ):
        violations.append(
            Violation(
                code="decision-not-successful",
                detail="native decision must be status=done and decision=proceed",
                stage=stage,
                artifact=str(decision_path),
            )
        )
    if decision.get("error") not in (None, ""):
        violations.append(
            Violation(
                code="decision-error-present",
                detail="native decision contains a non-null error",
                stage=stage,
                artifact=str(decision_path),
            )
        )
    if not pending_gate and (
        health.get("status") != "done" or health.get("error") not in (None, "")
    ):
        violations.append(
            Violation(
                code="stage-health-not-successful",
                detail="native stage health must be status=done with null error",
                stage=stage,
                artifact=str(health_path),
            )
        )
    for field in ("stage_id", "run_id"):
        if decision.get(field) != health.get(field):
            violations.append(
                Violation(
                    code=f"native-{field}-mismatch",
                    detail=f"decision and stage health disagree on {field}",
                    stage=stage,
                )
            )

    artifacts = decision.get("output_artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        violations.append(
            Violation(
                code="missing-output-artifacts",
                detail="completed decision must declare at least one output artifact",
                stage=stage,
                artifact=str(decision_path),
            )
        )
        return True, violations
    if health.get("artifacts_count") != len(artifacts):
        violations.append(
            Violation(
                code="artifact-count-mismatch",
                detail="stage health artifact count disagrees with the native decision",
                stage=stage,
            )
        )

    artifact_names = {item for item in artifacts if isinstance(item, str)}
    core_rules = CORE_ARTIFACT_MIN_BYTES.get(stage, {})
    for required_name in sorted(set(core_rules) - artifact_names):
        violations.append(
            Violation(
                code="missing-core-artifact",
                detail="native decision omitted a required scientific core artifact",
                stage=stage,
                artifact=required_name,
            )
        )

    stage_root = stage_dir.resolve()
    for artifact in artifacts:
        if not isinstance(artifact, str) or not artifact.strip():
            violations.append(
                Violation(
                    code="invalid-artifact-name",
                    detail="output artifact names must be non-empty strings",
                    stage=stage,
                )
            )
            continue
        candidate = (stage_dir / artifact).resolve()
        try:
            candidate.relative_to(stage_root)
        except ValueError:
            violations.append(
                Violation(
                    code="artifact-path-escape",
                    detail="declared output artifact escapes its stage directory",
                    stage=stage,
                    artifact=artifact,
                )
            )
            continue
        violations.extend(
            _validate_declared_artifact(
                candidate,
                stage=stage,
                relative_name=artifact,
                minimum_bytes=core_rules.get(artifact, MIN_TEXT_ARTIFACT_BYTES),
            )
        )
    if pending_gate and not violations:
        return False, []
    return True, violations


def _scan_completed_stages(output: Path, validated: set[int]) -> list[Violation]:
    violations: list[Violation] = []
    for stage_dir in sorted(output.glob("stage-[0-9][0-9]")):
        stage = _stage_number(stage_dir)
        if stage is None or stage in validated:
            continue
        ready, stage_violations = _validate_completed_stage(stage_dir)
        if not ready:
            continue
        if stage_violations:
            violations.extend(stage_violations)
        else:
            validated.add(stage)
    return violations


def _canonical_record_sha256(value: dict[str, Any]) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_jsonl_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} must contain a JSON object")
        records.append(value)
    return records


def _stage_file_descriptors(output: Path, stage: int) -> list[dict[str, str]]:
    stage_dir = output / f"stage-{stage:02d}"
    stage_root = stage_dir.resolve(strict=True)
    descriptors: list[dict[str, str]] = []
    for candidate in sorted(path for path in stage_dir.rglob("*") if path.is_file()):
        resolved = candidate.resolve(strict=True)
        try:
            resolved.relative_to(stage_root)
        except ValueError as exc:
            raise ValueError(f"gate artifact escapes stage-{stage:02d}: {candidate}") from exc
        descriptors.append(
            {
                "path": candidate.relative_to(output).as_posix(),
                "sha256": _sha256(candidate),
            }
        )
    if not descriptors:
        raise ValueError(f"stage-{stage:02d} has no files to bind")
    return descriptors


def _approval_record(
    records: list[dict[str, Any]],
    stage: int,
    *,
    source: str,
) -> tuple[int, dict[str, Any]]:
    stage_records = [
        (index, record)
        for index, record in enumerate(records)
        if record.get("stage") == stage
    ]
    if len(stage_records) != 1:
        raise ValueError(
            f"{source} must contain exactly one human action for Stage {stage}; "
            f"found {len(stage_records)}"
        )
    index, record = stage_records[0]
    human_input = record.get("human_input")
    if source == "native interventions":
        approved = (
            record.get("type") == "approve"
            and record.get("accepted") is True
            and record.get("pause_reason") == "gate_approval"
            and isinstance(human_input, dict)
            and human_input.get("action") == "approve"
        )
    else:
        approved = record.get("action") == "approve"
    if not approved:
        raise ValueError(f"{source} does not contain an accepted Stage {stage} approval")
    return index, record


def _gate_handoff_path(output: Path, stage: int) -> Path:
    return output / "control" / f"gate-handoff-stage-{stage:02d}.v2.json"


def _run_relative_file(output: Path, value: Any, *, field: str) -> Path:
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or "\x00" in value
    ):
        raise ValueError(f"{field} must be a portable relative path")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"{field} contains an unsafe path component")
    candidate = output.joinpath(*parts)
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(output.resolve())
    except ValueError as exc:
        raise ValueError(f"{field} escapes the ARC run") from exc
    if candidate.is_symlink() or not resolved.is_file():
        raise ValueError(f"{field} must identify a non-symlink file")
    return resolved


def _checkpoint_binding_for_gate(output: Path, stage: int) -> dict[str, Any]:
    checkpoint_path = output / "checkpoint.json"
    checkpoint = _read_json(checkpoint_path)
    if checkpoint.get("last_completed_stage") != stage - 1:
        raise ValueError(
            f"Stage {stage} approval requires checkpoint at Stage {stage - 1}"
        )
    return {
        "path": "checkpoint.json",
        "sha256": _sha256(checkpoint_path),
        "last_completed_stage": stage - 1,
    }


def _build_gate_handoff(
    *,
    output: Path,
    stage: int,
    source_config_snapshot: Path,
    effective_config: Path,
    project_state_snapshot: Path,
) -> dict[str, Any]:
    if stage not in REQUIRED_GATES:
        raise ValueError(f"Stage {stage} is not a registered human gate")
    ready, violations = _validate_completed_stage(output / f"stage-{stage:02d}")
    if ready or violations:
        codes = ",".join(item.code for item in violations) or "not-blocked-approval"
        raise ValueError(f"Stage {stage} is not a clean post-artifact gate: {codes}")

    interventions_path = output / "hitl" / "interventions.jsonl"
    intervention_index, intervention = _approval_record(
        _read_jsonl_records(interventions_path),
        stage,
        source="native interventions",
    )
    events_path = output / "bridge.events.v1.jsonl"
    event_index, event = _approval_record(
        _read_jsonl_records(events_path),
        stage,
        source="bridge events",
    )
    response_sha256 = event.get("response_sha256")
    waiting_sha256 = event.get("waiting_sha256")
    if not isinstance(response_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", response_sha256
    ):
        raise ValueError("bridge approval event lacks a valid response SHA-256")
    if not isinstance(waiting_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", waiting_sha256
    ):
        raise ValueError("bridge approval event lacks a valid waiting SHA-256")

    checkpoint_binding = event.get("checkpoint_before_approval")
    if not isinstance(checkpoint_binding, dict) or (
        checkpoint_binding.get("path") != "checkpoint.json"
        or checkpoint_binding.get("last_completed_stage") != stage - 1
        or not isinstance(checkpoint_binding.get("sha256"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", checkpoint_binding["sha256"])
    ):
        raise ValueError("bridge approval event lacks the bound pre-approval checkpoint")

    decision = _read_json(output / f"stage-{stage:02d}" / "decision.json")
    session = _read_json(output / "hitl" / "session.json")
    native_run_id = decision.get("run_id")
    session_id = session.get("session_id")
    if not isinstance(native_run_id, str) or not native_run_id:
        raise ValueError(f"Stage {stage} decision lacks a native run_id")
    if session.get("run_id") != native_run_id:
        raise ValueError("native gate decision and HITL session run_id differ")
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("HITL session lacks a session_id")
    nonce = event.get("nonce")
    if not isinstance(nonce, str) or not re.fullmatch(r"[0-9a-f]{64}", nonce):
        raise ValueError("bridge approval event lacks the consumed challenge nonce")
    if event.get("run_id") != native_run_id or event.get("session_id") != session_id:
        raise ValueError("bridge approval event run/session binding has changed")
    _control_timestamp(event.get("issued_at"), field="bridge approval issued_at")
    consumption_relative = event.get("consumption_receipt_path")
    consumption_path = _run_relative_file(
        output, consumption_relative, field="consumption_receipt_path"
    )
    consumption_sha256 = event.get("consumption_receipt_sha256")
    if (
        not isinstance(consumption_sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", consumption_sha256)
        or _sha256(consumption_path) != consumption_sha256
    ):
        raise ValueError("bridge approval event consumption receipt hash is invalid")
    consumption = _read_json(consumption_path)
    expected_consumption_keys = {
        "schema_version",
        "claimed_at",
        "response_path",
        "response_sha256",
        "stage",
        "run_id",
        "session_id",
        "waiting_sha256",
        "preapproval_checkpoint_sha256",
        "nonce",
        "response",
    }
    if set(consumption) != expected_consumption_keys or consumption.get(
        "schema_version"
    ) != CONSUMPTION_SCHEMA:
        raise ValueError("operator response consumption receipt schema is invalid")
    if (
        consumption.get("stage") != stage
        or consumption.get("run_id") != native_run_id
        or consumption.get("session_id") != session_id
        or consumption.get("waiting_sha256") != waiting_sha256
        or consumption.get("preapproval_checkpoint_sha256")
        != checkpoint_binding["sha256"]
        or consumption.get("nonce") != nonce
        or consumption.get("response_sha256") != response_sha256
    ):
        raise ValueError("operator response consumption receipt binding is invalid")
    expected_response_snapshot = _operator_response_snapshot_path(
        output,
        stage=stage,
        response_sha256=response_sha256,
    )
    response_snapshot = _run_relative_file(
        output,
        consumption.get("response_path"),
        field="operator response snapshot",
    )
    if (
        response_snapshot != expected_response_snapshot.resolve(strict=True)
        or _sha256(response_snapshot) != response_sha256
    ):
        raise ValueError("operator response snapshot binding is invalid")
    consumed_response = consumption.get("response")
    if not isinstance(consumed_response, dict):
        raise ValueError("operator response consumption receipt lacks the response")
    if (
        consumed_response.get("stage") != stage
        or consumed_response.get("run_id") != native_run_id
        or consumed_response.get("session_id") != session_id
        or consumed_response.get("waiting_sha256") != waiting_sha256
        or consumed_response.get("preapproval_checkpoint_sha256")
        != checkpoint_binding["sha256"]
        or consumed_response.get("nonce") != nonce
        or consumed_response.get("action") != "approve"
    ):
        raise ValueError("consumed operator response does not bind this approval")
    if _read_json(response_snapshot) != consumed_response:
        raise ValueError("consumed operator response differs from its snapshot")
    checkpoint_snapshot = _preapproval_checkpoint_snapshot_path(
        output,
        stage=stage,
        checkpoint_sha256=checkpoint_binding["sha256"],
    )
    if (
        not checkpoint_snapshot.is_file()
        or checkpoint_snapshot.is_symlink()
        or _sha256(checkpoint_snapshot) != checkpoint_binding["sha256"]
        or _read_json(checkpoint_snapshot).get("last_completed_stage") != stage - 1
    ):
        raise ValueError("pre-approval checkpoint snapshot binding is invalid")

    previous_path: Path | None = None
    previous_sha256: str | None = None
    earlier = [gate for gate in ORDERED_GATES if gate < stage]
    if earlier:
        previous_path = _gate_handoff_path(output, earlier[-1])
        if not previous_path.is_file():
            raise ValueError(f"Stage {stage} handoff is missing the prior gate handoff")
        previous_sha256 = _sha256(previous_path)

    return {
        "schema_version": GATE_HANDOFF_SCHEMA,
        "created_at": _utc_now(),
        "stage": stage,
        "stage_name": STAGE_NUMBER_NAMES[stage],
        "native_identity": {
            "run_id": native_run_id,
            "session_id": session_id,
        },
        "next_stage": stage + 1,
        "next_stage_name": STAGE_NUMBER_NAMES[stage + 1],
        "stage_files": _stage_file_descriptors(output, stage),
        "approval": {
            "response_sha256": response_sha256,
            "waiting_sha256": waiting_sha256,
            "run_id": native_run_id,
            "session_id": session_id,
            "nonce": nonce,
            "consumption_receipt_path": consumption_relative,
            "consumption_receipt_sha256": consumption_sha256,
            "native_intervention_ordinal": intervention_index,
            "native_intervention_sha256": _canonical_record_sha256(intervention),
            "bridge_event_ordinal": event_index,
            "bridge_event_sha256": _canonical_record_sha256(event),
        },
        "checkpoint": {
            **checkpoint_binding,
        },
        "lineage": {
            "source_config_snapshot_sha256": _sha256(source_config_snapshot),
            "effective_config_sha256": _sha256(effective_config),
            "project_state_sha256": _sha256(project_state_snapshot),
            "previous_handoff_path": (
                previous_path.relative_to(output).as_posix() if previous_path else None
            ),
            "previous_handoff_sha256": previous_sha256,
        },
    }


def _validate_gate_handoffs(output: Path) -> tuple[list[dict[str, Any]], list[Violation]]:
    paths = sorted((output / "control").glob("gate-handoff-stage-*.v2.json"))
    records: list[dict[str, Any]] = []
    violations: list[Violation] = []
    expected_prefix = list(ORDERED_GATES[: len(paths)])
    observed_stages: list[int] = []
    previous_path: Path | None = None
    expected_native_identity: dict[str, str] | None = None
    for path in paths:
        try:
            record = _read_json(path)
            stage = int(record.get("stage"))
            observed_stages.append(stage)
            if record.get("schema_version") != GATE_HANDOFF_SCHEMA:
                raise ValueError("unsupported gate handoff schema")
            if path != _gate_handoff_path(output, stage):
                raise ValueError("gate handoff filename/stage mismatch")
            if record.get("stage_name") != STAGE_NUMBER_NAMES.get(stage):
                raise ValueError("gate handoff stage name mismatch")
            if record.get("next_stage") != stage + 1 or record.get(
                "next_stage_name"
            ) != STAGE_NUMBER_NAMES.get(stage + 1):
                raise ValueError("gate handoff has an incorrect next-stage entrypoint")
            native_identity = record.get("native_identity")
            if not isinstance(native_identity, dict) or not all(
                isinstance(native_identity.get(field), str)
                and bool(native_identity.get(field))
                for field in ("run_id", "session_id")
            ):
                raise ValueError("gate handoff native run/session identity is missing")
            if expected_native_identity is None:
                expected_native_identity = {
                    "run_id": native_identity["run_id"],
                    "session_id": native_identity["session_id"],
                }
            elif native_identity != expected_native_identity:
                raise ValueError("native run_id/session_id changed between approved gates")
            decision = _read_json(output / f"stage-{stage:02d}" / "decision.json")
            session = _read_json(output / "hitl" / "session.json")
            if decision.get("run_id") != native_identity.get("run_id"):
                raise ValueError("approved gate decision run_id has changed")
            if (
                session.get("run_id") != native_identity.get("run_id")
                or session.get("session_id") != native_identity.get("session_id")
            ):
                raise ValueError("current native HITL run/session identity has changed")
            if record.get("stage_files") != _stage_file_descriptors(output, stage):
                raise ValueError("approved gate artifact set or hash has changed")

            approval = record.get("approval")
            if not isinstance(approval, dict):
                raise ValueError("gate handoff approval binding is missing")
            expected_approval_keys = {
                "response_sha256",
                "waiting_sha256",
                "run_id",
                "session_id",
                "nonce",
                "consumption_receipt_path",
                "consumption_receipt_sha256",
                "native_intervention_ordinal",
                "native_intervention_sha256",
                "bridge_event_ordinal",
                "bridge_event_sha256",
            }
            if set(approval) != expected_approval_keys:
                raise ValueError("gate handoff approval binding fields are incomplete")
            interventions = _read_jsonl_records(output / "hitl" / "interventions.jsonl")
            intervention_index = approval.get("native_intervention_ordinal")
            if not isinstance(intervention_index, int) or intervention_index >= len(
                interventions
            ):
                raise ValueError("bound native approval intervention is missing")
            observed_intervention_index, intervention = _approval_record(
                interventions, stage, source="native interventions"
            )
            if observed_intervention_index != intervention_index:
                raise ValueError("bound native approval intervention ordinal has changed")
            if _canonical_record_sha256(intervention) != approval.get(
                "native_intervention_sha256"
            ):
                raise ValueError("bound native approval intervention has changed")
            events = _read_jsonl_records(output / "bridge.events.v1.jsonl")
            event_index = approval.get("bridge_event_ordinal")
            if not isinstance(event_index, int) or event_index >= len(events):
                raise ValueError("bound bridge approval event is missing")
            observed_event_index, event = _approval_record(
                events, stage, source="bridge events"
            )
            if observed_event_index != event_index:
                raise ValueError("bound bridge approval event ordinal has changed")
            if _canonical_record_sha256(event) != approval.get("bridge_event_sha256"):
                raise ValueError("bound bridge approval event has changed")
            if event.get("response_sha256") != approval.get("response_sha256"):
                raise ValueError("approval response hash binding has changed")
            for key in (
                "waiting_sha256",
                "run_id",
                "session_id",
                "nonce",
                "consumption_receipt_path",
                "consumption_receipt_sha256",
            ):
                if event.get(key) != approval.get(key):
                    raise ValueError(f"approval {key} binding has changed")
            if (
                approval.get("run_id") != native_identity.get("run_id")
                or approval.get("session_id") != native_identity.get("session_id")
            ):
                raise ValueError("approval native run/session binding has changed")
            consumption_path = _run_relative_file(
                output,
                approval.get("consumption_receipt_path"),
                field="approval.consumption_receipt_path",
            )
            if _sha256(consumption_path) != approval.get("consumption_receipt_sha256"):
                raise ValueError("bound operator response consumption receipt has changed")
            if record.get("checkpoint") != event.get("checkpoint_before_approval"):
                raise ValueError("pre-approval checkpoint binding has changed")

            lineage = record.get("lineage")
            if not isinstance(lineage, dict):
                raise ValueError("gate handoff lineage is missing")
            for key, candidate in (
                ("source_config_snapshot_sha256", output / "control" / "source-config.yaml"),
                ("effective_config_sha256", output / "control" / "effective-config.yaml"),
                ("project_state_sha256", output / "control" / "project-state.v1.json"),
            ):
                if not candidate.is_file() or lineage.get(key) != _sha256(candidate):
                    raise ValueError(f"gate handoff {key} binding has changed")
            if previous_path is None:
                if lineage.get("previous_handoff_path") is not None or lineage.get(
                    "previous_handoff_sha256"
                ) is not None:
                    raise ValueError("first gate handoff has an unexpected predecessor")
            else:
                if lineage.get("previous_handoff_path") != previous_path.relative_to(
                    output
                ).as_posix() or lineage.get("previous_handoff_sha256") != _sha256(
                    previous_path
                ):
                    raise ValueError("gate handoff hash chain is broken")
            ready, stage_violations = _validate_completed_stage(
                output / f"stage-{stage:02d}"
            )
            if ready or stage_violations:
                raise ValueError("approved gate is no longer a clean native gate artifact")
            records.append(record)
            previous_path = path
        except (
            OSError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            TypeError,
            ValueError,
        ) as exc:
            violations.append(
                Violation(
                    code="gate-handoff-invalid",
                    detail=str(exc),
                    artifact=str(path),
                )
            )
    if observed_stages != expected_prefix:
        violations.append(
            Violation(
                code="gate-handoff-sequence-invalid",
                detail=(
                    f"gate handoffs must be the ordered prefix {expected_prefix}; "
                    f"found {observed_stages}"
                ),
            )
        )
    if expected_native_identity is not None:
        expected_run_id = expected_native_identity["run_id"]
        for decision_path in sorted(output.glob("stage-[0-9][0-9]/decision.json")):
            try:
                decision = _read_json(decision_path)
                if decision.get("run_id") != expected_run_id:
                    raise ValueError(
                        f"{decision_path.parent.name} changed native run_id"
                    )
            except (
                OSError,
                UnicodeDecodeError,
                json.JSONDecodeError,
                ValueError,
            ) as exc:
                violations.append(
                    Violation(
                        code="native-run-identity-invalid",
                        detail=str(exc),
                        artifact=str(decision_path),
                    )
                )
    return records, violations


def _receipt_bundle_path(receipt_root: Path, stage: int) -> Path:
    if stage not in REQUIRED_GATES:
        raise ValueError(f"unsupported formal receipt stage: {stage}")
    return receipt_root / f"arc-stage{stage}"


def _waiting_bundle_path(receipt_root: Path, stage: int) -> Path:
    if stage not in REQUIRED_GATES:
        raise ValueError(f"unsupported waiting receipt stage: {stage}")
    return receipt_root / f"arc-stage{stage}-waiting"


def _external_operator_response_path(receipt_root: Path, stage: int) -> Path:
    if stage not in REQUIRED_GATES:
        raise ValueError(f"unsupported operator response stage: {stage}")
    return (
        receipt_root
        / "arc-operator-responses"
        / f"stage-{stage:02d}"
        / "operator-response.v2.json"
    )


def _copy_receipt_evidence(
    *,
    source: Path,
    bundle: Path,
    relative: str,
    expected_sha256: str,
) -> Path:
    """Copy one source-run file into a private, not-yet-published bundle."""

    if (
        not relative
        or "\\" in relative
        or "\x00" in relative
        or any(part in {"", ".", ".."} for part in relative.split("/"))
    ):
        raise ValueError("receipt evidence path is not a safe POSIX relative path")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("receipt evidence SHA-256 must be a full lowercase digest")
    resolved_source = source.resolve(strict=True)
    if source.is_symlink() or not resolved_source.is_file():
        raise ValueError("receipt evidence source must be a non-symlink regular file")
    if _sha256(resolved_source) != expected_sha256:
        raise ValueError("receipt evidence source changed before export")
    target = bundle.joinpath(*relative.split("/"))
    target.parent.mkdir(parents=True, exist_ok=True)
    with resolved_source.open("rb") as reader, target.open("xb") as writer:
        shutil.copyfileobj(reader, writer, length=1 << 20)
        writer.flush()
        os.fsync(writer.fileno())
    if _sha256(target) != expected_sha256:
        raise ValueError("exported receipt evidence hash mismatch")
    if _sha256(resolved_source) != expected_sha256:
        raise ValueError("receipt evidence source changed during export")
    return target


def _receipt_descriptor(bundle: Path, relative: str) -> dict[str, str]:
    path = bundle.joinpath(*relative.split("/"))
    return {"path": relative, "sha256": _sha256(path)}


def _previous_exported_report(
    receipt_root: Path, stage: int
) -> dict[str, Any] | None:
    previous: dict[str, Any] | None = None
    for candidate in ORDERED_GATES:
        if candidate >= stage:
            break
        previous = validate_arc_control_bundle(
            _receipt_bundle_path(receipt_root, candidate),
            candidate,
            previous_report=previous,
        )
    return previous


def _canonical_ascii_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _waiting_predecessor_export(
    receipt_root: Path, stage: int
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    previous_report = _previous_exported_report(receipt_root, stage)
    previous_stages = [candidate for candidate in ORDERED_GATES if candidate < stage]
    if not previous_stages:
        return None, previous_report
    if previous_report is None:
        raise ValueError(f"Stage {stage} waiting receipt lacks its prior formal report")
    previous_stage = previous_stages[-1]
    if previous_report.get("stage") != previous_stage:
        raise ValueError("prior formal report stage is inconsistent")
    return (
        {
            "stage": previous_stage,
            "report_sha256": _canonical_ascii_sha256(previous_report),
            "receipt_sha256": previous_report["receipt_sha256"],
            "handoff_sha256": previous_report["gate_handoff"]["sha256"],
            "chain_sha256": previous_report["gate_lineage"]["chain_sha256"],
        },
        previous_report,
    )


def _validate_published_waiting_receipt(
    *, receipt_root: Path, stage: int, challenge: dict[str, Any]
) -> dict[str, Any]:
    previous = _previous_exported_report(receipt_root, stage)
    report = validate_arc_waiting_bundle(
        _waiting_bundle_path(receipt_root, stage),
        stage,
        previous_report=previous,
    )
    if (
        report["run_id"] != challenge.get("run_id")
        or report["session_id"] != challenge.get("session_id")
        or report["waiting"]["sha256"] != challenge.get("waiting_sha256")
        or report["challenge"]["nonce"] != challenge.get("nonce")
        or report["challenge"]["preapproval_checkpoint_sha256"]
        != challenge.get("preapproval_checkpoint_sha256")
    ):
        raise ValueError("published waiting receipt does not bind the active challenge")
    return report


def _export_waiting_receipt(
    *,
    output: Path,
    receipt_root: Path,
    waiting: dict[str, Any],
    challenge: dict[str, Any],
) -> Path:
    """Atomically publish a pre-approval bundle without advancing native ARC."""

    stage = waiting.get("stage")
    if isinstance(stage, bool) or not isinstance(stage, int) or stage not in REQUIRED_GATES:
        raise ValueError("native waiting record is not a registered ARC gate")
    final_bundle = _waiting_bundle_path(receipt_root, stage)
    if final_bundle.exists():
        _validate_published_waiting_receipt(
            receipt_root=receipt_root, stage=stage, challenge=challenge
        )
        return final_bundle
    predecessor, previous_report = _waiting_predecessor_export(receipt_root, stage)
    receipt_root.mkdir(parents=True, exist_ok=True)
    staging = receipt_root / (
        f".arc-stage{stage}-waiting.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    )
    staging.mkdir()
    try:
        stage_dir = f"stage-{stage:02d}"
        decision_source = output / stage_dir / "decision.json"
        decision = _read_json(decision_source)
        outputs = decision.get("output_artifacts")
        if not isinstance(outputs, list) or not outputs:
            raise ValueError("native waiting decision has no output artifacts")
        stage_output_descriptors: list[dict[str, str]] = []
        fixed_sources = {
            "decision": decision_source,
            "stage_health": output / stage_dir / "stage_health.json",
            "session": output / "hitl" / "session.json",
            "waiting": output / "hitl" / "waiting.json",
            "operator_challenge": output / "hitl" / "operator-challenge.v2.json",
        }
        fixed_relatives = {
            "decision": f"{stage_dir}/decision.json",
            "stage_health": f"{stage_dir}/stage_health.json",
            "session": "hitl/session.json",
            "waiting": "hitl/waiting.json",
            "operator_challenge": "hitl/operator-challenge.v2.json",
        }
        for name, source in fixed_sources.items():
            _copy_receipt_evidence(
                source=source,
                bundle=staging,
                relative=fixed_relatives[name],
                expected_sha256=_sha256(source),
            )
        for raw_name in outputs:
            if not isinstance(raw_name, str) or not raw_name:
                raise ValueError("native waiting output artifact name is invalid")
            relative = f"{stage_dir}/{raw_name}"
            source = _run_relative_file(
                output, relative, field="native waiting stage output"
            )
            _copy_receipt_evidence(
                source=source,
                bundle=staging,
                relative=relative,
                expected_sha256=_sha256(source),
            )
            stage_output_descriptors.append(_receipt_descriptor(staging, relative))
        checkpoint_binding = _checkpoint_binding_for_gate(output, stage)
        checkpoint_snapshot = _snapshot_preapproval_checkpoint(
            output=output,
            stage=stage,
            checkpoint_binding=checkpoint_binding,
        )
        _copy_receipt_evidence(
            source=checkpoint_snapshot,
            bundle=staging,
            relative="checkpoint.json",
            expected_sha256=checkpoint_binding["sha256"],
        )
        identity = {
            "run_id": challenge.get("run_id"),
            "session_id": challenge.get("session_id"),
            "stage": stage,
        }
        receipt = {
            "schema_version": WAITING_RECEIPT_SCHEMA,
            "autoresearchclaw": {
                "repository": ARC_REPOSITORY,
                "version": ARC_VERSION,
                "commit": ARC_COMMIT,
            },
            "acp": {
                "acpx_version": ACPX_VERSION,
                "claude_adapter_version": CLAUDE_ADAPTER_VERSION,
                "codex_adapter_version": CODEX_ADAPTER_VERSION,
                "package_lock_sha256": ACP_PACKAGE_LOCK_SHA256,
            },
            "invocation": {"mode": "co-pilot", "auto_approve": False},
            "run": identity,
            "artifacts": {
                **{
                    name: _receipt_descriptor(staging, relative)
                    for name, relative in fixed_relatives.items()
                },
                "checkpoint": _receipt_descriptor(staging, "checkpoint.json"),
                "stage_outputs": stage_output_descriptors,
            },
            "predecessor": predecessor,
        }
        _exclusive_json(staging / "waiting-receipt.v1.json", receipt)
        validate_arc_waiting_bundle(
            staging, stage, previous_report=previous_report
        )
        try:
            staging.rename(final_bundle)
        except FileExistsError:
            _validate_published_waiting_receipt(
                receipt_root=receipt_root, stage=stage, challenge=challenge
            )
            return final_bundle
        _validate_published_waiting_receipt(
            receipt_root=receipt_root, stage=stage, challenge=challenge
        )
        return final_bundle
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _validate_published_receipt(
    *, receipt_root: Path, stage: int, handoff: dict[str, Any]
) -> dict[str, Any]:
    previous = _previous_exported_report(receipt_root, stage)
    report = validate_arc_control_bundle(
        _receipt_bundle_path(receipt_root, stage),
        stage,
        previous_report=previous,
    )
    identity = handoff.get("native_identity")
    published_handoff = (
        _receipt_bundle_path(receipt_root, stage)
        / "control"
        / f"gate-handoff-stage-{stage:02d}.v2.json"
    )
    if not isinstance(identity, dict) or (
        report["run_id"] != identity.get("run_id")
        or report["session_id"] != identity.get("session_id")
        or _read_json(published_handoff) != handoff
        or report["gate_handoff"]["sha256"] != _sha256(published_handoff)
    ):
        raise ValueError("published receipt does not bind the requested gate handoff")
    return report


def _export_gate_receipt(
    *,
    output: Path,
    receipt_root: Path,
    handoff: dict[str, Any],
) -> Path:
    """Publish one self-contained v2 control bundle after a real accepted handoff."""

    stage = handoff.get("stage")
    if isinstance(stage, bool) or not isinstance(stage, int) or stage not in REQUIRED_GATES:
        raise ValueError("gate handoff has an unsupported receipt stage")
    handoff_source = _gate_handoff_path(output, stage)
    if _read_json(handoff_source) != handoff:
        raise ValueError("gate handoff changed before receipt export")
    final_bundle = _receipt_bundle_path(receipt_root, stage)
    if final_bundle.exists():
        _validate_published_receipt(
            receipt_root=receipt_root,
            stage=stage,
            handoff=handoff,
        )
        return final_bundle

    receipt_root.mkdir(parents=True, exist_ok=True)
    staging = receipt_root / (
        f".arc-stage{stage}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    )
    staging.mkdir()
    stage_dir = f"stage-{stage:02d}"
    stage_files = handoff.get("stage_files")
    if not isinstance(stage_files, list) or not stage_files:
        raise ValueError("gate handoff has no stage files to export")
    copied_stage_paths: set[str] = set()
    for descriptor in stage_files:
        if not isinstance(descriptor, dict) or set(descriptor) != {"path", "sha256"}:
            raise ValueError("gate handoff stage descriptor is malformed")
        relative = descriptor["path"]
        source = _run_relative_file(output, relative, field="gate handoff stage file")
        _copy_receipt_evidence(
            source=source,
            bundle=staging,
            relative=relative,
            expected_sha256=descriptor["sha256"],
        )
        copied_stage_paths.add(relative)

    fixed_relatives = {
        "decision": f"{stage_dir}/decision.json",
        "stage_health": f"{stage_dir}/stage_health.json",
        "session": "hitl/session.json",
        "interventions": "hitl/interventions.jsonl",
        "gate_handoff": handoff_source.relative_to(output).as_posix(),
    }
    for key in ("decision", "stage_health"):
        if fixed_relatives[key] not in copied_stage_paths:
            raise ValueError(f"gate handoff omitted {fixed_relatives[key]}")
    for key in ("session", "interventions", "gate_handoff"):
        relative = fixed_relatives[key]
        source = _run_relative_file(output, relative, field=f"receipt {key}")
        _copy_receipt_evidence(
            source=source,
            bundle=staging,
            relative=relative,
            expected_sha256=_sha256(source),
        )

    approval = handoff.get("approval")
    checkpoint = handoff.get("checkpoint")
    if not isinstance(approval, dict) or not isinstance(checkpoint, dict):
        raise ValueError("gate handoff lacks approval/checkpoint evidence")
    consumption_relative = approval.get("consumption_receipt_path")
    consumption = _run_relative_file(
        output,
        consumption_relative,
        field="operator response consumption receipt",
    )
    _copy_receipt_evidence(
        source=consumption,
        bundle=staging,
        relative=consumption_relative,
        expected_sha256=approval.get("consumption_receipt_sha256"),
    )
    consumption_record = _read_json(consumption)
    response_relative = consumption_record.get("response_path")
    response_snapshot = _run_relative_file(
        output, response_relative, field="operator response snapshot"
    )
    _copy_receipt_evidence(
        source=response_snapshot,
        bundle=staging,
        relative=response_relative,
        expected_sha256=approval.get("response_sha256"),
    )
    checkpoint_sha256 = checkpoint.get("sha256")
    checkpoint_snapshot = _preapproval_checkpoint_snapshot_path(
        output,
        stage=stage,
        checkpoint_sha256=checkpoint_sha256,
    )
    _copy_receipt_evidence(
        source=checkpoint_snapshot,
        bundle=staging,
        relative="checkpoint.json",
        expected_sha256=checkpoint_sha256,
    )

    decision = _read_json(output / fixed_relatives["decision"])
    output_artifacts = decision.get("output_artifacts")
    if not isinstance(output_artifacts, list) or not output_artifacts:
        raise ValueError("native decision has no stage outputs for its formal receipt")
    stage_output_descriptors: list[dict[str, str]] = []
    for name in output_artifacts:
        if not isinstance(name, str) or not name:
            raise ValueError("native decision output artifact name is invalid")
        relative = f"{stage_dir}/{name}"
        if relative not in copied_stage_paths:
            raise ValueError("native decision output is absent from the gate handoff")
        stage_output_descriptors.append(_receipt_descriptor(staging, relative))

    identity = handoff.get("native_identity")
    if not isinstance(identity, dict) or not isinstance(identity.get("run_id"), str):
        raise ValueError("gate handoff native identity is malformed")
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "autoresearchclaw": {
            "repository": ARC_REPOSITORY,
            "version": ARC_VERSION,
            "commit": ARC_COMMIT,
        },
        "acp": {
            "acpx_version": ACPX_VERSION,
            "claude_adapter_version": CLAUDE_ADAPTER_VERSION,
            "codex_adapter_version": CODEX_ADAPTER_VERSION,
            "package_lock_sha256": ACP_PACKAGE_LOCK_SHA256,
        },
        "invocation": {"mode": "co-pilot", "auto_approve": False},
        "run": {"run_id": identity["run_id"], "stage": stage},
        "artifacts": {
            "decision": _receipt_descriptor(staging, fixed_relatives["decision"]),
            "stage_health": _receipt_descriptor(
                staging, fixed_relatives["stage_health"]
            ),
            "session": _receipt_descriptor(staging, fixed_relatives["session"]),
            "interventions": _receipt_descriptor(
                staging, fixed_relatives["interventions"]
            ),
            "stage_outputs": stage_output_descriptors,
            "gate_handoff": _receipt_descriptor(
                staging, fixed_relatives["gate_handoff"]
            ),
        },
    }
    _exclusive_json(staging / "receipt.v1.json", receipt)
    previous_report = _previous_exported_report(receipt_root, stage)
    validate_arc_control_bundle(staging, stage, previous_report=previous_report)
    try:
        staging.rename(final_bundle)
    except FileExistsError:
        _validate_published_receipt(
            receipt_root=receipt_root,
            stage=stage,
            handoff=handoff,
        )
        return final_bundle
    _validate_published_receipt(
        receipt_root=receipt_root,
        stage=stage,
        handoff=handoff,
    )
    return final_bundle


def _export_ready_gate_receipts(
    *, output: Path, receipt_root: Path, handoffs: list[dict[str, Any]]
) -> dict[int, Path]:
    exported: dict[int, Path] = {}
    for handoff in handoffs:
        stage = int(handoff["stage"])
        exported[stage] = _export_gate_receipt(
            output=output,
            receipt_root=receipt_root,
            handoff=handoff,
        )
    return exported


def _bind_ready_gate_handoffs(
    *,
    output: Path,
    source_config_snapshot: Path,
    effective_config: Path,
    project_state_snapshot: Path,
) -> tuple[list[dict[str, Any]], list[Violation]]:
    """Bind newly approved gates while the one official process stays alive."""

    records, violations = _validate_gate_handoffs(output)
    if violations:
        return records, violations
    try:
        interventions = _read_jsonl_records(output / "hitl" / "interventions.jsonl")
        events = _read_jsonl_records(output / "bridge.events.v1.jsonl")
        while len(records) < len(ORDERED_GATES):
            stage = ORDERED_GATES[len(records)]
            has_native_action = any(item.get("stage") == stage for item in interventions)
            has_bridge_action = any(item.get("stage") == stage for item in events)
            if not has_native_action or not has_bridge_action:
                later_actions = {
                    int(item["stage"])
                    for item in [*interventions, *events]
                    if isinstance(item.get("stage"), int) and item["stage"] > stage
                }
                if later_actions:
                    raise ValueError(
                        f"observed later gate actions before binding Stage {stage}: "
                        f"{sorted(later_actions)}"
                    )
                break
            record = _build_gate_handoff(
                output=output,
                stage=stage,
                source_config_snapshot=source_config_snapshot,
                effective_config=effective_config,
                project_state_snapshot=project_state_snapshot,
            )
            path = _gate_handoff_path(output, stage)
            _exclusive_json(path, record)
            records.append(record)
        validated_records, validation_errors = _validate_gate_handoffs(output)
        if validation_errors:
            return validated_records, validation_errors
        return validated_records, []
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
    ) as exc:
        return records, [
            Violation(
                code="gate-handoff-bind-failed",
                detail=str(exc),
            )
        ]


def _signal_exact_process(identity: ProcessIdentity, sig: signal.Signals) -> bool:
    """Signal a PID only if its creation token still matches the captured one."""

    current = _current_process_identity(identity.pid)
    if not _same_process(current, identity):
        return False
    try:
        os.kill(identity.pid, sig)
    except ProcessLookupError:
        return False
    return True


def _terminate_process_tree(
    process: subprocess.Popen[str],
    tracker: ProcessTreeTracker,
    *,
    timeout: float = 10.0,
) -> TerminationReport:
    """Terminate captured identities leaf-first, including reparented children.

    ``taskkill /T`` and process-name matching can affect an unrelated process if
    a PID is reused.  This routine snapshots PID + creation time and rechecks
    that identity immediately before every exact-PID signal.
    """

    table = tracker.refresh()
    targets = tracker.leaf_first(table)
    targeted = tuple(identity.pid for identity in targets)
    signaled: set[int] = set()
    mismatched: set[int] = {
        pid
        for pid, identity in tracker.identities.items()
        if table.get(pid) is not None and not _same_process(table.get(pid), identity)
    }
    for identity in targets:
        if _signal_exact_process(identity, signal.SIGTERM):
            signaled.add(identity.pid)
        else:
            current_identity = _current_process_identity(identity.pid)
            if current_identity is not None and not _same_process(
                current_identity, identity
            ):
                mismatched.add(identity.pid)

    deadline = time.monotonic() + max(timeout, 0.0)
    while time.monotonic() < deadline:
        current = _current_process_table()
        if not any(_same_process(current.get(item.pid), item) for item in targets):
            break
        time.sleep(min(0.1, max(deadline - time.monotonic(), 0.0)))

    current = _current_process_table()
    remaining = [item for item in targets if _same_process(current.get(item.pid), item)]
    hard_signal = signal.SIGTERM if os.name == "nt" else signal.SIGKILL
    for identity in remaining:
        if _signal_exact_process(identity, hard_signal):
            signaled.add(identity.pid)
        else:
            current_identity = _current_process_identity(identity.pid)
            if current_identity is not None and not _same_process(
                current_identity, identity
            ):
                mismatched.add(identity.pid)

    try:
        process.wait(timeout=max(timeout, 0.1))
    except subprocess.TimeoutExpired:
        pass
    final_table = _current_process_table()
    still_alive = tuple(
        item.pid for item in targets if _same_process(final_table.get(item.pid), item)
    )
    terminated = tuple(pid for pid in targeted if pid in signaled and pid not in still_alive)
    return TerminationReport(
        targeted_leaf_first=targeted,
        terminated=terminated,
        identity_mismatch_skipped=tuple(sorted(mismatched)),
        still_alive=still_alive,
    )


def _write_invalidation(
    *,
    output: Path,
    violations: list[Violation],
    process_pid: int,
    source_config: Path,
    source_config_snapshot: Path,
    effective_config: Path,
    status_path: Path,
    command: list[str],
    process_cleanup: TerminationReport | None = None,
    project_state_snapshot: Path | None = None,
) -> Path:
    path = output / "run-invalidation.v1.json"
    record = {
        "schema_version": INVALIDATION_SCHEMA,
        "invalidated_at": _utc_now(),
        "authority": "additive-fail-closed-bridge-record",
        "arc_commit": ARC_COMMIT,
        "process_pid": process_pid,
        "command": command,
        "source_config": {
            "path": str(source_config),
            "observed_sha256": _sha256(source_config_snapshot),
            "snapshot_path": str(source_config_snapshot),
            "snapshot_sha256": _sha256(source_config_snapshot),
        },
        "effective_config": {
            "path": str(effective_config),
            "sha256": _sha256(effective_config),
        },
        "project_state": (
            {
                "path": str(project_state_snapshot),
                "sha256": _sha256(project_state_snapshot),
            }
            if project_state_snapshot is not None
            else None
        ),
        "bridge_status_sha256_before_invalidation": (
            _sha256(status_path) if status_path.is_file() else None
        ),
        "violations": [violation.as_dict() for violation in violations],
        "process_cleanup": process_cleanup.as_dict() if process_cleanup else None,
        "resume_permitted": False,
    }
    _exclusive_json(path, record)
    return path


def _final_stage_validation(
    output: Path, validated: set[int], *, expected_stage: int
) -> list[Violation]:
    violations = _scan_completed_stages(output, validated)
    for stage_dir in sorted(output.glob("stage-[0-9][0-9]")):
        stage = _stage_number(stage_dir)
        if stage is None or stage in validated:
            continue
        violations.append(
            Violation(
                code="incomplete-native-stage-record",
                detail=(
                    "ARC exited before the stage had a validated successful "
                    "native completion record"
                ),
                stage=stage,
                artifact=str(stage_dir),
            )
        )
    if expected_stage not in validated:
        violations.append(
            Violation(
                code="target-stage-not-validated",
                detail="ARC exited without a validated target-stage completion",
                stage=expected_stage,
            )
        )
    return violations


def main() -> int:
    args = _parser().parse_args()
    checkout = args.arc_checkout.resolve(strict=True)
    project_root = args.project_root.resolve(strict=True)
    python = args.python.resolve(strict=True)
    acpx_command = args.acpx_command.resolve(strict=True)
    codex_home = args.codex_home.resolve(strict=True)
    codex_config = (codex_home / "config.toml").resolve(strict=True)
    source_config = args.config.resolve(strict=True)
    output = args.output.resolve()
    receipt_root = args.receipt_root.resolve()
    if _git_commit(checkout) != ARC_COMMIT:
        raise SystemExit("official ARC checkout is not at the pinned v0.5.0 commit")
    try:
        _require_clean_git_worktree(checkout, label="official ARC checkout")
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    if args.poll_seconds <= 0:
        raise SystemExit("--poll-seconds must be positive")

    try:
        expected_stage = _target_stage_number(args.to_stage)
        _validate_continuous_target(expected_stage)
        if args.resume:
            raise ValueError(
                "--resume is forbidden: official ARC v0.5.0 changes native "
                "run_id/session_id; use one continuous process from an empty output"
            )
        gateway_url = os.environ.get("OPENAI_BASE_URL", "")
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY is required for the isolated ACP provider")
        codex_acp_config, codex_acp_config_json = _build_codex_acp_config(
            base_url=gateway_url,
            model=args.acp_model,
            reasoning_effort=args.acp_reasoning_effort,
            provider_id=args.acp_provider_id,
        )
        _assert_external_output(output, project_root)
        _assert_external_output(receipt_root, project_root)
        _assert_empty_new_output(output, resume=args.resume)
        source_payload = source_config.read_bytes()
        effective_loaded, effective_payload = _prepare_effective_config(
            source_config=source_config,
            output=output,
            project_root=project_root,
            python=python,
            acpx_command=acpx_command,
        )
        initial_project_state = _project_state(
            project_root,
            source_config=source_config,
            effective_loaded=effective_loaded,
        )
        _require_clean_project_state(initial_project_state)
        if source_config.read_bytes() != source_payload:
            raise ValueError("ARC source configuration changed while being prepared")
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise SystemExit(str(exc)) from exc

    output.mkdir(parents=True, exist_ok=True)
    control_dir = output / "control"
    source_config_snapshot = control_dir / "source-config.yaml"
    effective_config = control_dir / "effective-config.yaml"
    project_state_snapshot = control_dir / "project-state.v1.json"
    try:
        _write_or_verify_source_snapshot(
            source_config_snapshot, source_payload, resume=args.resume
        )
        _write_or_verify_effective_config(
            effective_config, effective_payload, resume=args.resume
        )
        _write_or_verify_project_state(
            project_state_snapshot,
            initial_project_state,
            resume=args.resume,
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    stdout_path = output.parent / f"{output.name}.stdout.log"
    stderr_path = output.parent / f"{output.name}.stderr.log"
    if not args.resume and (stdout_path.exists() or stderr_path.exists()):
        raise SystemExit("new ARC run requires absent sibling stdout/stderr logs")
    status_path = output / "bridge.status.v2.json"
    events_path = output / "bridge.events.v1.jsonl"
    challenge_path = output / "hitl" / "operator-challenge.v2.json"
    waiting_path = output / "hitl" / "waiting.json"

    prior_handoffs: list[dict[str, Any]] = []
    prior_events: list[dict[str, Any]] = []

    command = _build_arc_command(
        python=python,
        effective_config=effective_config,
        output=output,
        to_stage=args.to_stage,
        resume=args.resume,
    )
    if "--auto-approve" in command:
        raise AssertionError("the co-pilot bridge must never auto-approve")

    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = str(checkout) + (
        os.pathsep + existing_pythonpath if existing_pythonpath else ""
    )
    environment["CODEX_HOME"] = str(codex_home)
    environment["CODEX_CONFIG"] = codex_acp_config_json
    environment["MODEL_PROVIDER"] = args.acp_provider_id
    started_at = _utc_now()
    log_mode = "a" if args.resume else "x"
    with stdout_path.open(log_mode, encoding="utf-8") as stdout, stderr_path.open(
        log_mode, encoding="utf-8"
    ) as stderr:
        popen_kwargs: dict[str, Any] = {}
        if os.name != "nt":
            popen_kwargs["start_new_session"] = True
        process = subprocess.Popen(
            command,
            cwd=output,
            env=environment,
            stdin=subprocess.PIPE,
            stdout=stdout,
            stderr=stderr,
            text=True,
            encoding="utf-8",
            **popen_kwargs,
        )
        try:
            process_tracker = ProcessTreeTracker.start(process.pid)
        except (OSError, RuntimeError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
            process.terminate()
            process.wait(timeout=10)
            raise SystemExit(f"cannot establish exact child process tracking: {exc}") from exc
        source_config_hash = hashlib.sha256(source_payload).hexdigest()
        base_status: dict[str, Any] = {
            "schema_version": STATUS_SCHEMA,
            "arc_commit": ARC_COMMIT,
            "auto_approve": False,
            "mode": "co-pilot",
            "command": command,
            "resume_strategy": "single-native-process",
            "resume_from_stage": None,
            "source_config_path": str(source_config),
            "source_config_observed_sha256": source_config_hash,
            "source_config_snapshot_path": str(source_config_snapshot),
            "source_config_snapshot_sha256": _sha256(source_config_snapshot),
            "effective_config_path": str(effective_config),
            "effective_config_sha256": _sha256(effective_config),
            "project_state_path": str(project_state_snapshot),
            "project_state_sha256": _sha256(project_state_snapshot),
            "project_git_commit": initial_project_state["project_git_commit"],
            "project_git_dirty": initial_project_state["project_git_dirty"],
            "project_content_state_sha256": initial_project_state["state_sha256"],
            "knowledge_base_root": effective_loaded["knowledge_base"]["root"],
            "acp_session_name": effective_loaded["llm"]["acp"]["session_name"],
            "acpx_command_path": str(acpx_command),
            "acpx_command_sha256": _sha256(acpx_command),
            "codex_home_path": str(codex_home),
            "codex_config_path": str(codex_config),
            "codex_config_sha256": _sha256(codex_config),
            "codex_acp_effective_config_sha256": hashlib.sha256(
                codex_acp_config_json.encode("utf-8")
            ).hexdigest(),
            "codex_acp_model": codex_acp_config["model"],
            "codex_acp_reasoning_effort": codex_acp_config[
                "model_reasoning_effort"
            ],
            "codex_acp_model_provider": codex_acp_config["model_provider"],
            "codex_acp_provider_base_url_sha256": hashlib.sha256(
                gateway_url.rstrip("/").encode("utf-8")
            ).hexdigest(),
            "codex_acp_api_key_source": "OPENAI_API_KEY environment",
            "child_cwd": str(output),
            "project_read_root": str(project_root),
            "output": str(output),
            "receipt_root": str(receipt_root),
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
            "pid": process.pid,
            "started_at": started_at,
        }
        _atomic_json(status_path, {**base_status, "state": "running"})
        consumed_hashes: set[str] = {
            value
            for event in prior_events
            if isinstance((value := event.get("response_sha256")), str)
        }
        validated_stages: set[int] = {
            int(record["stage"]) for record in prior_handoffs
        }
        current_waiting_hash = ""
        current_challenge: dict[str, Any] | None = None
        invalidated = False
        bridge_violation: Violation | None = None
        cleanup_report: TerminationReport | None = None
        next_project_state_check = 0.0
        latest_gate_handoff: Path | None = None
        exported_receipts: dict[int, Path] = {}
        exported_waiting_receipts: dict[int, Path] = {}
        try:
            while process.poll() is None:
                process_tracker.refresh()
                bound_handoffs, handoff_bind_violations = _bind_ready_gate_handoffs(
                    output=output,
                    source_config_snapshot=source_config_snapshot,
                    effective_config=effective_config,
                    project_state_snapshot=project_state_snapshot,
                )
                if handoff_bind_violations:
                    invalidated = True
                    cleanup_report = _terminate_process_tree(process, process_tracker)
                    _write_invalidation(
                        output=output,
                        violations=handoff_bind_violations,
                        process_pid=process.pid,
                        source_config=source_config,
                        source_config_snapshot=source_config_snapshot,
                        effective_config=effective_config,
                        status_path=status_path,
                        command=command,
                        process_cleanup=cleanup_report,
                        project_state_snapshot=project_state_snapshot,
                    )
                    break
                if bound_handoffs:
                    pending_receipts = [
                        record
                        for record in bound_handoffs
                        if int(record["stage"]) not in exported_receipts
                    ]
                    if pending_receipts:
                        exported_receipts.update(
                            _export_ready_gate_receipts(
                                output=output,
                                receipt_root=receipt_root,
                                handoffs=pending_receipts,
                            )
                        )
                    validated_stages.update(
                        int(record["stage"]) for record in bound_handoffs
                    )
                    latest_stage = int(bound_handoffs[-1]["stage"])
                    latest_gate_handoff = _gate_handoff_path(output, latest_stage)
                    base_status.update(
                        {
                            "native_run_id": bound_handoffs[-1]["native_identity"][
                                "run_id"
                            ],
                            "native_session_id": bound_handoffs[-1][
                                "native_identity"
                            ]["session_id"],
                            "gate_handoff_stage": latest_stage,
                            "gate_handoff_path": str(latest_gate_handoff),
                            "gate_handoff_sha256": _sha256(latest_gate_handoff),
                        }
                    )
                    _atomic_json(
                        status_path,
                        {
                            **base_status,
                            "state": "running",
                            "validated_stages": sorted(validated_stages),
                            "formal_receipts": {
                                str(stage): {
                                    "path": str(path / "receipt.v1.json"),
                                    "sha256": _sha256(path / "receipt.v1.json"),
                                }
                                for stage, path in sorted(exported_receipts.items())
                            },
                        },
                    )
                now = time.monotonic()
                if now >= next_project_state_check:
                    current_project_state = _project_state(
                        project_root,
                        source_config=source_config,
                        effective_loaded=effective_loaded,
                    )
                    if current_project_state != initial_project_state:
                        violations = [
                            Violation(
                                code="project-input-drift",
                                detail=(
                                    "bound repository/config/protocol content changed "
                                    "after this run started"
                                ),
                            )
                        ]
                    else:
                        violations = []
                    _, handoff_violations = _validate_gate_handoffs(output)
                    violations.extend(handoff_violations)
                    for stage, _path in sorted(exported_receipts.items()):
                        handoff = next(
                            record
                            for record in bound_handoffs
                            if int(record["stage"]) == stage
                        )
                        _validate_published_receipt(
                            receipt_root=receipt_root,
                            stage=stage,
                            handoff=handoff,
                        )
                    for stage in sorted(exported_waiting_receipts):
                        waiting_report = validate_arc_waiting_bundle(
                            _waiting_bundle_path(receipt_root, stage),
                            stage,
                            previous_report=_previous_exported_report(
                                receipt_root, stage
                            ),
                        )
                        if (
                            bound_handoffs
                            and waiting_report["run_id"]
                            != bound_handoffs[-1]["native_identity"]["run_id"]
                        ):
                            raise ValueError(
                                "waiting receipt native identity differs from handoff chain"
                            )
                    next_project_state_check = now + 15.0
                else:
                    violations = []
                if violations:
                    invalidated = True
                    cleanup_report = _terminate_process_tree(process, process_tracker)
                    _write_invalidation(
                        output=output,
                        violations=violations,
                        process_pid=process.pid,
                        source_config=source_config,
                        source_config_snapshot=source_config_snapshot,
                        effective_config=effective_config,
                        status_path=status_path,
                        command=command,
                        process_cleanup=cleanup_report,
                        project_state_snapshot=project_state_snapshot,
                    )
                    break
                violations = _scan_completed_stages(output, validated_stages)
                if violations:
                    invalidated = True
                    cleanup_report = _terminate_process_tree(process, process_tracker)
                    _write_invalidation(
                        output=output,
                        violations=violations,
                        process_pid=process.pid,
                        source_config=source_config,
                        source_config_snapshot=source_config_snapshot,
                        effective_config=effective_config,
                        status_path=status_path,
                        command=command,
                        process_cleanup=cleanup_report,
                        project_state_snapshot=project_state_snapshot,
                    )
                    break
                if waiting_path.is_file():
                    waiting_hash = _sha256(waiting_path)
                    waiting = _read_json(waiting_path)
                    if waiting_hash != current_waiting_hash:
                        current_waiting_hash = waiting_hash
                        current_challenge = _build_operator_challenge(
                            output=output,
                            waiting=waiting,
                            waiting_sha256=waiting_hash,
                        )
                        _atomic_json(challenge_path, current_challenge)
                        waiting_stage = waiting.get("stage")
                        if (
                            isinstance(waiting_stage, bool)
                            or not isinstance(waiting_stage, int)
                            or waiting_stage not in REQUIRED_GATES
                        ):
                            raise ValueError(
                                "operator challenges are accepted only at registered gates"
                            )
                        waiting_bundle = _export_waiting_receipt(
                            output=output,
                            receipt_root=receipt_root,
                            waiting=waiting,
                            challenge=current_challenge,
                        )
                        exported_waiting_receipts[waiting_stage] = waiting_bundle
                        response_path = _external_operator_response_path(
                            receipt_root, waiting_stage
                        )
                        _atomic_json(
                            status_path,
                            {
                                **base_status,
                                "state": "waiting",
                                "validated_stages": sorted(validated_stages),
                                "waiting_stage": waiting.get("stage"),
                                "waiting_stage_name": waiting.get("stage_name"),
                                "waiting_reason": waiting.get("reason"),
                                "waiting_sha256": waiting_hash,
                                "operator_challenge_path": str(challenge_path),
                                "operator_challenge_sha256": _sha256(challenge_path),
                                "operator_challenge_nonce": current_challenge["nonce"],
                                "operator_response_path": str(response_path),
                                "waiting_receipt_path": str(
                                    waiting_bundle / "waiting-receipt.v1.json"
                                ),
                                "waiting_receipt_sha256": _sha256(
                                    waiting_bundle / "waiting-receipt.v1.json"
                                ),
                            },
                        )
                    waiting_stage = waiting.get("stage")
                    if (
                        isinstance(waiting_stage, bool)
                        or not isinstance(waiting_stage, int)
                        or waiting_stage not in REQUIRED_GATES
                    ):
                        raise ValueError(
                            "operator responses are accepted only at registered gates"
                        )
                    response_path = _external_operator_response_path(
                        receipt_root, waiting_stage
                    )
                    if response_path.is_file():
                        response_hash = _sha256(response_path)
                        if response_hash in consumed_hashes:
                            # ARC may leave waiting.json in place for one more poll
                            # after stdin is flushed.  The immutable DAG response is
                            # retained as evidence; the exclusive consumption record
                            # proves it was forwarded exactly once.
                            time.sleep(args.poll_seconds)
                            continue
                        if current_challenge is None:
                            raise ValueError("operator response has no active challenge")
                        response = _read_json(response_path)
                        _validate_response(response, waiting, current_challenge)
                        forward_payload = validate_signed_review_response(
                            response,
                            expected_stage=waiting_stage,
                            expected_run_id=current_challenge["run_id"],
                            expected_session_id=current_challenge["session_id"],
                            expected_waiting_sha256=waiting_hash,
                            expected_checkpoint_sha256=current_challenge[
                                "preapproval_checkpoint_sha256"
                            ],
                            expected_nonce=current_challenge["nonce"],
                            reviewer_public_key=project_root
                            / "security"
                            / "reviewer_ed25519.pub",
                        )
                        published_waiting = _validate_published_waiting_receipt(
                            receipt_root=receipt_root,
                            stage=waiting_stage,
                            challenge=current_challenge,
                        )
                        if forward_payload["waiting_report"] != published_waiting:
                            raise ValueError(
                                "signed review is not bound to the published waiting receipt"
                            )
                        key = {
                            "approve": "a\n",
                            "reject": f"r\n{response['message']}\n",
                            "abort": "q\ny\n",
                        }[response["action"]]
                        if process.stdin is None:
                            raise RuntimeError("ARC stdin bridge is unavailable")
                        checkpoint_binding = _checkpoint_binding_for_gate(
                            output,
                            waiting_stage,
                        )
                        if checkpoint_binding["sha256"] != current_challenge.get(
                            "preapproval_checkpoint_sha256"
                        ):
                            raise ValueError(
                                "pre-approval checkpoint changed after challenge issuance"
                            )
                        if _sha256(waiting_path) != waiting_hash:
                            raise ValueError(
                                "native waiting record changed while response was validated"
                            )
                        session = _read_json(output / "hitl" / "session.json")
                        if (
                            session.get("run_id") != current_challenge.get("run_id")
                            or session.get("session_id")
                            != current_challenge.get("session_id")
                        ):
                            raise ValueError(
                                "native run/session changed after challenge issuance"
                            )
                        _snapshot_preapproval_checkpoint(
                            output=output,
                            stage=waiting_stage,
                            checkpoint_binding=checkpoint_binding,
                        )
                        response_snapshot = _snapshot_operator_response(
                            output=output,
                            response_path=response_path,
                            stage=waiting_stage,
                            response_sha256=response_hash,
                        )
                        consumption_path = _claim_operator_response(
                            output=output,
                            response_path=response_snapshot,
                            response=response,
                            response_sha256=response_hash,
                            challenge=current_challenge,
                        )
                        process.stdin.write(key)
                        process.stdin.flush()
                        event = {
                            "timestamp": _utc_now(),
                            "stage": waiting_stage,
                            "run_id": current_challenge["run_id"],
                            "session_id": current_challenge["session_id"],
                            "action": response["action"],
                            "issued_at": response["issued_at"],
                            "nonce": current_challenge["nonce"],
                            "response_sha256": response_hash,
                            "waiting_sha256": waiting_hash,
                            "checkpoint_before_approval": checkpoint_binding,
                            "consumption_receipt_path": consumption_path.relative_to(
                                output
                            ).as_posix(),
                            "consumption_receipt_sha256": _sha256(consumption_path),
                        }
                        with events_path.open("a", encoding="utf-8") as events:
                            events.write(
                                json.dumps(event, sort_keys=True, allow_nan=False)
                                + "\n"
                            )
                            events.flush()
                            os.fsync(events.fileno())
                        consumed_hashes.add(response_hash)
                        _retire_operator_response(response_path, response_hash)
                time.sleep(args.poll_seconds)
        except (
            OSError,
            RuntimeError,
            ValueError,
            json.JSONDecodeError,
            subprocess.SubprocessError,
        ) as exc:
            frames = traceback.extract_tb(exc.__traceback__)
            location = frames[-1] if frames else None
            where = (
                f" at {Path(location.filename).name}:{location.lineno} "
                f"in {location.name}"
                if location is not None
                else ""
            )
            bridge_violation = Violation(
                code="bridge-control-error",
                detail=f"{type(exc).__name__}: {exc}{where}",
            )
            invalidated = True
            cleanup_report = _terminate_process_tree(process, process_tracker)
            if not (output / "run-invalidation.v1.json").exists():
                _write_invalidation(
                    output=output,
                    violations=[bridge_violation],
                    process_pid=process.pid,
                    source_config=source_config,
                    source_config_snapshot=source_config_snapshot,
                    effective_config=effective_config,
                    status_path=status_path,
                    command=command,
                    process_cleanup=cleanup_report,
                    project_state_snapshot=project_state_snapshot,
                )
        finally:
            if process.stdin is not None:
                process.stdin.close()

        exit_code = process.wait()
        if invalidated:
            return 2
        cleanup_report = _terminate_process_tree(process, process_tracker)
        bound_handoffs, final_bind_violations = _bind_ready_gate_handoffs(
            output=output,
            source_config_snapshot=source_config_snapshot,
            effective_config=effective_config,
            project_state_snapshot=project_state_snapshot,
        )
        pending_receipts = [
            record
            for record in bound_handoffs
            if int(record["stage"]) not in exported_receipts
        ]
        if pending_receipts and not final_bind_violations:
            exported_receipts.update(
                _export_ready_gate_receipts(
                    output=output,
                    receipt_root=receipt_root,
                    handoffs=pending_receipts,
                )
            )
        if bound_handoffs:
            validated_stages.update(int(record["stage"]) for record in bound_handoffs)
            latest_stage = int(bound_handoffs[-1]["stage"])
            latest_gate_handoff = _gate_handoff_path(output, latest_stage)
            base_status.update(
                {
                    "native_run_id": bound_handoffs[-1]["native_identity"]["run_id"],
                    "native_session_id": bound_handoffs[-1]["native_identity"][
                        "session_id"
                    ],
                    "gate_handoff_stage": latest_stage,
                    "gate_handoff_path": str(latest_gate_handoff),
                    "gate_handoff_sha256": _sha256(latest_gate_handoff),
                }
            )
        final_violations = _final_stage_validation(
            output, validated_stages, expected_stage=expected_stage
        )
        final_violations.extend(final_bind_violations)
        if exit_code != 0:
            final_violations.append(
                Violation(
                    code="native-process-exit-nonzero",
                    detail=f"official ARC process exited with code {exit_code}",
                )
            )
        if (
            _project_state(
                project_root,
                source_config=source_config,
                effective_loaded=effective_loaded,
            )
            != initial_project_state
        ):
            final_violations.append(
                Violation(
                    code="project-input-drift",
                    detail=(
                        "bound repository/config/protocol content changed after this run started"
                    ),
                )
            )
        if cleanup_report.still_alive:
            final_violations.append(
                Violation(
                    code="process-tree-cleanup-incomplete",
                    detail="one or more exact child identities survived cleanup",
                    artifact=",".join(str(pid) for pid in cleanup_report.still_alive),
                )
            )
        _, final_handoff_violations = _validate_gate_handoffs(output)
        final_violations.extend(final_handoff_violations)
        if final_violations:
            _write_invalidation(
                output=output,
                violations=final_violations,
                process_pid=process.pid,
                source_config=source_config,
                source_config_snapshot=source_config_snapshot,
                effective_config=effective_config,
                status_path=status_path,
                command=command,
                process_cleanup=cleanup_report,
                project_state_snapshot=project_state_snapshot,
            )
            return 2
        _atomic_json(
            status_path,
            {
                **base_status,
                "state": "completed" if exit_code == 0 else "failed",
                "ended_at": _utc_now(),
                "exit_code": exit_code,
                "validated_stages": sorted(validated_stages),
                "formal_receipts": {
                    str(stage): {
                        "path": str(path / "receipt.v1.json"),
                        "sha256": _sha256(path / "receipt.v1.json"),
                    }
                    for stage, path in sorted(exported_receipts.items())
                },
                "process_cleanup": cleanup_report.as_dict(),
                "gate_handoff_stage": (
                    int(bound_handoffs[-1]["stage"]) if bound_handoffs else None
                ),
                "gate_handoff_path": (
                    str(latest_gate_handoff)
                    if latest_gate_handoff is not None
                    else None
                ),
                "gate_handoff_sha256": (
                    _sha256(latest_gate_handoff)
                    if latest_gate_handoff is not None
                    else None
                ),
            },
        )
        return exit_code


if __name__ == "__main__":
    sys.exit(main())
