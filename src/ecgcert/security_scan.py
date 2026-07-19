"""Fail-closed repository secret scanning for the Stage-9 evidence gate.

The scanner deliberately reports only rule names and relative paths.  It never
copies a matched value into an artifact or console output.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
from typing import Any, Iterable, Mapping

from ecgcert import lineage


SCHEMA_VERSION = "ecgcert-repository-secret-scan-v1"
SCANNER_VERSION = "1"
MAX_TEXT_BYTES = 5 * 1024 * 1024
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")

# These patterns are intentionally high precision.  The generic assignment
# rule requires a non-placeholder value of at least eight non-space bytes.
_RULES: dict[str, bytes] = {
    "private_key": rb"-----BEGIN (?:OPENSSH |RSA |EC |DSA )?PRIVATE KEY-----",
    "aws_access_key": rb"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b",
    "github_token": rb"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{40,})\b",
    "openai_api_key": rb"\bsk-[A-Za-z0-9_-]{20,}\b",
    "slack_token": rb"\bxox[baprs]-[A-Za-z0-9-]{20,}\b",
    "credentialed_url": rb"\b[a-z][a-z0-9+.-]{1,15}://[^\s/:@]+:[^\s/@]{8,}@",
    "credential_assignment": (
        rb"(?im)\b(?:password|passwd|pwd|token|api[_-]?key|secret)\b"
        rb"\s*[:=]\s*(?:"
        rb"[\"']([^\"'\r\n]{8,})[\"']"
        rb"|([A-Za-z0-9_./+=:@-]{8,})\s*(?:$|#)"
        rb")"
    ),
}
_PLACEHOLDERS = {
    b"<redacted>",
    b"redacted",
    b"changeme",
    b"placeholder",
    b"example-only",
    b"not-a-secret",
}
_EXCLUDED_PREFIXES = (
    ".git/",
    ".venv/",
    "venv/",
    "data/",
    "upstreams/",
    "artifacts/",
    ".pytest_cache/",
    ".ruff_cache/",
    "build/",
    "dist/",
    "results/cache/",
)


def _git(root: Path, *arguments: str) -> str:
    run = subprocess.run(
        ["git", *arguments],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
        timeout=60,
        check=False,
    )
    if run.returncode:
        raise RuntimeError(f"git {' '.join(arguments)} failed")
    return run.stdout.strip()


def _git_paths(root: Path, *arguments: str) -> list[str]:
    run = subprocess.run(
        ["git", *arguments, "-z"],
        cwd=root,
        capture_output=True,
        timeout=120,
        check=False,
    )
    if run.returncode:
        raise RuntimeError(f"git {' '.join(arguments)} failed")
    return [
        raw.decode("utf-8", errors="surrogateescape")
        for raw in run.stdout.split(b"\0")
        if raw
    ]


def _safe_relative_path(value: str) -> PurePosixPath:
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError("git returned an unsafe repository path")
    return path


def _excluded(relative: PurePosixPath) -> bool:
    value = relative.as_posix()
    return any(value == prefix.rstrip("/") or value.startswith(prefix) for prefix in _EXCLUDED_PREFIXES)


def _patterns_sha256() -> str:
    return lineage.canonical_sha256(
        {name: pattern.decode("ascii") for name, pattern in sorted(_RULES.items())}
    )


def _matches(data: bytes) -> Iterable[str]:
    for name, pattern in _RULES.items():
        match = re.search(pattern, data)
        if match is None:
            continue
        if name == "credential_assignment":
            value = (match.group(1) or match.group(2)).strip().lower()
            if value in _PLACEHOLDERS or value.startswith(b"${") or value.startswith(b"{{"):
                continue
        yield name


def _inventory_sha256(rows: list[Mapping[str, Any]]) -> str:
    return lineage.canonical_sha256(rows)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def scan_repository(
    repository: Path,
    output: Path,
    *,
    scanned_at: datetime | None = None,
) -> dict[str, Any]:
    """Scan tracked, untracked, and repository-owned ignored text files.

    Release scans must run from a clean checkout and write outside that checkout.
    Excluded raw-data, upstream, environment, cache, and artifact directories are
    recorded in the report and are never followed through symlinks.
    """

    root = repository.resolve(strict=True)
    output = output.resolve()
    try:
        output.relative_to(root)
    except ValueError:
        pass
    else:
        raise ValueError("secret-scan output must be outside the repository")
    if _git(root, "rev-parse", "--show-toplevel").replace("\\", "/") != str(root).replace("\\", "/"):
        raise ValueError("repository must be the git worktree root")
    status = _git(root, "status", "--porcelain", "--untracked-files=all")
    if status:
        raise RuntimeError("repository secret scan requires a clean checkout")

    commit = _git(root, "rev-parse", "HEAD")
    tree = _git(root, "rev-parse", "HEAD^{tree}")
    if not _HEX40.fullmatch(commit) or not _HEX40.fullmatch(tree):
        raise RuntimeError("repository commit/tree identity is malformed")

    classified: dict[str, str] = {}
    for classification, arguments in (
        ("tracked", ("ls-files",)),
        ("untracked", ("ls-files", "--others", "--exclude-standard")),
        ("ignored", ("ls-files", "--others", "--ignored", "--exclude-standard")),
    ):
        for value in _git_paths(root, *arguments):
            relative = _safe_relative_path(value)
            classified.setdefault(relative.as_posix(), classification)

    inventory: list[dict[str, Any]] = []
    findings: list[dict[str, str]] = []
    scanned_files = 0
    skipped_binary = 0
    skipped_large = 0
    skipped_symlink = 0
    excluded_files = 0
    for value in sorted(classified):
        relative = PurePosixPath(value)
        classification = classified[value]
        if _excluded(relative):
            excluded_files += 1
            inventory.append({"path": value, "class": classification, "scan": "excluded"})
            continue
        path = root.joinpath(*relative.parts)
        if path.is_symlink():
            skipped_symlink += 1
            inventory.append({"path": value, "class": classification, "scan": "symlink"})
            continue
        if not path.is_file():
            inventory.append({"path": value, "class": classification, "scan": "non-file"})
            continue
        size = path.stat().st_size
        if size > MAX_TEXT_BYTES:
            skipped_large += 1
            inventory.append(
                {"path": value, "class": classification, "scan": "large", "bytes": size}
            )
            continue
        data = path.read_bytes()
        if b"\0" in data[:8192]:
            skipped_binary += 1
            inventory.append(
                {"path": value, "class": classification, "scan": "binary", "bytes": size}
            )
            continue
        scanned_files += 1
        digest = hashlib.sha256(data).hexdigest()
        inventory.append(
            {
                "path": value,
                "class": classification,
                "scan": "text",
                "bytes": size,
                "sha256": digest,
            }
        )
        for rule in _matches(data):
            findings.append({"path": value, "rule": rule})

    findings = sorted(findings, key=lambda row: (row["path"], row["rule"]))
    timestamp = scanned_at or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        raise ValueError("scanned_at must be timezone-aware")
    scanner_path = Path(__file__).resolve(strict=True)
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete" if not findings else "failed",
        "scanned_at": timestamp.astimezone(timezone.utc).isoformat(),
        "repository": {
            "commit": commit,
            "tree": tree,
            "clean": True,
        },
        "scanner": {
            "version": SCANNER_VERSION,
            "module": "src/ecgcert/security_scan.py",
            "sha256": lineage.artifact_sha256(scanner_path),
            "patterns_sha256": _patterns_sha256(),
        },
        "scope": {
            "tracked": True,
            "untracked": True,
            "ignored": True,
            "symlinks_followed": False,
            "excluded_prefixes": list(_EXCLUDED_PREFIXES),
            "max_text_bytes": MAX_TEXT_BYTES,
            "inventory_sha256": _inventory_sha256(inventory),
            "inventory_entries": len(inventory),
            "scanned_files": scanned_files,
            "excluded_files": excluded_files,
            "skipped_binary": skipped_binary,
            "skipped_large": skipped_large,
            "skipped_symlink": skipped_symlink,
        },
        "findings_count": len(findings),
        "findings_sha256": lineage.canonical_sha256(findings),
        # No secret values or matching excerpts are ever persisted.
        "findings": findings,
    }
    _atomic_json(output, report)
    return report


def validate_secret_scan_report(
    report_path: Path,
    *,
    expected_commit: str | None = None,
    expected_tree: str | None = None,
    expected_scanner_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate a release scan and optionally bind it to the executing source."""

    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict) or report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"repository secret scan must use {SCHEMA_VERSION}")
    repository = report.get("repository")
    scanner = report.get("scanner")
    scope = report.get("scope")
    findings = report.get("findings")
    if not all(isinstance(value, dict) for value in (repository, scanner, scope)):
        raise ValueError("repository secret scan metadata is malformed")
    if not isinstance(findings, list):
        raise ValueError("repository secret scan findings must be a list")
    checks = {
        "status_complete": report.get("status") == "complete",
        "clean_checkout": repository.get("clean") is True,
        "commit_bound": bool(_HEX40.fullmatch(str(repository.get("commit")))),
        "tree_bound": bool(_HEX40.fullmatch(str(repository.get("tree")))),
        "scanner_bound": bool(_HEX64.fullmatch(str(scanner.get("sha256")))),
        "patterns_bound": bool(_HEX64.fullmatch(str(scanner.get("patterns_sha256")))),
        "inventory_bound": bool(_HEX64.fullmatch(str(scope.get("inventory_sha256")))),
        "all_repository_classes_scanned": all(
            scope.get(field) is True for field in ("tracked", "untracked", "ignored")
        ),
        "symlinks_not_followed": scope.get("symlinks_followed") is False,
        "no_findings": report.get("findings_count") == 0 and not findings,
        "findings_hash_bound": (
            report.get("findings_sha256") == lineage.canonical_sha256(findings)
        ),
        "nonempty_scan": isinstance(scope.get("scanned_files"), int)
        and scope["scanned_files"] > 0,
    }
    if expected_commit is not None:
        checks["executing_commit_matches"] = repository.get("commit") == expected_commit
    if expected_tree is not None:
        checks["executing_tree_matches"] = repository.get("tree") == expected_tree
    if expected_scanner_sha256 is not None:
        checks["executing_scanner_matches"] = (
            scanner.get("sha256") == expected_scanner_sha256
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "checks": checks,
        "all_controls_satisfied": all(checks.values()),
        "report_sha256": lineage.artifact_sha256(report_path),
        "repository_commit": repository.get("commit"),
        "repository_tree": repository.get("tree"),
        "scanner_sha256": scanner.get("sha256"),
        "patterns_sha256": scanner.get("patterns_sha256"),
        "inventory_sha256": scope.get("inventory_sha256"),
        "scanned_files": scope.get("scanned_files"),
        "findings_count": report.get("findings_count"),
    }
