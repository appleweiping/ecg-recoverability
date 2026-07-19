"""Fail-closed exposure of persistent official upstream checkouts to the DAG."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import stat
import subprocess
from typing import Iterable

from ecgcert.estimators.official import (
    ECG_RECOVER,
    IMPUTE_ECG,
    UpstreamSpec,
    validate_pinned_checkout,
)


class UpstreamLinkError(RuntimeError):
    """Raised when the persistent upstream root cannot be linked safely."""


@dataclass(frozen=True)
class _CreatedLink:
    path: Path
    target: Path
    device: int
    inode: int
    ctime_ns: int


UPSTREAM_SPECS = (IMPUTE_ECG, ECG_RECOVER)
REPOSITORY_PATH = "upstreams"


def _lexists(path: Path) -> bool:
    return os.path.lexists(path)


def _absolute_existing_directory(raw: Path | str, *, label: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        raise UpstreamLinkError(f"{label} must be an absolute path: {path}")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise UpstreamLinkError(f"{label} does not exist or cannot be resolved: {path}") from exc
    if not resolved.is_dir():
        raise UpstreamLinkError(f"{label} is not a directory: {resolved}")
    return resolved


def _git(repo: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["git", *arguments],
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise UpstreamLinkError("cannot inspect repository upstream-link safety") from exc


def _require_repository_root(repo: Path) -> None:
    completed = _git(repo, "rev-parse", "--show-toplevel")
    if completed.returncode:
        raise UpstreamLinkError(f"repository path is not a Git worktree: {repo}")
    try:
        top_level = Path(completed.stdout.strip()).resolve(strict=True)
    except OSError as exc:
        raise UpstreamLinkError("Git returned an invalid worktree root") from exc
    if top_level != repo:
        raise UpstreamLinkError(
            f"repository path must be the absolute Git worktree root: {repo} != {top_level}"
        )


def _require_ignored_and_untracked(repo: Path) -> None:
    ignored = _git(repo, "check-ignore", "--quiet", "--no-index", "--", REPOSITORY_PATH)
    if ignored.returncode:
        raise UpstreamLinkError(f"refusing non-ignored repository path: {REPOSITORY_PATH}")
    tracked = _git(repo, "ls-files", "--", REPOSITORY_PATH, f"{REPOSITORY_PATH}/**")
    if tracked.returncode:
        raise UpstreamLinkError("cannot inspect tracked state for repository upstream path")
    if tracked.stdout.strip():
        raise UpstreamLinkError(f"refusing tracked repository path: {REPOSITORY_PATH}")


def _existing_link_matches(link: Path, target: Path) -> bool:
    if not link.is_symlink():
        return False
    try:
        raw_target = Path(os.readlink(link))
        resolved = link.resolve(strict=True)
    except OSError:
        return False
    return raw_target.is_absolute() and raw_target == target and resolved == target


def _created_link_identity(link: Path, target: Path) -> _CreatedLink | None:
    if not _existing_link_matches(link, target):
        return None
    try:
        status = link.lstat()
    except OSError:
        return None
    if not stat.S_ISLNK(status.st_mode):
        return None
    return _CreatedLink(
        path=link,
        target=target,
        device=status.st_dev,
        inode=status.st_ino,
        ctime_ns=status.st_ctime_ns,
    )


def _same_created_link(created: _CreatedLink) -> bool:
    if not _existing_link_matches(created.path, created.target):
        return False
    try:
        status = created.path.lstat()
    except OSError:
        return False
    return (
        stat.S_ISLNK(status.st_mode)
        and status.st_dev == created.device
        and status.st_ino == created.inode
        and status.st_ctime_ns == created.ctime_ns
    )


def _checkout_inventory(source_root: Path, specs: Iterable[UpstreamSpec]) -> list[dict[str, str]]:
    inventory: list[dict[str, str]] = []
    for spec in specs:
        if spec.root_tree is None:
            raise UpstreamLinkError(f"missing frozen root tree for {spec.name}")
        source = source_root / f"{spec.name}-{spec.commit[:12]}"
        try:
            validate_pinned_checkout(source, spec)
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            raise UpstreamLinkError(
                f"persistent {spec.name} checkout failed frozen lineage validation: {source}"
            ) from exc
        inventory.append(
            {
                "name": spec.name,
                "path": str(source.resolve(strict=True)),
                "repository": spec.repository,
                "commit": spec.commit,
                "root_tree": spec.root_tree,
            }
        )
    return inventory


def ensure_server_upstream_link(
    *,
    repo: Path | str,
    tools_root: Path | str,
    specs: Iterable[UpstreamSpec] = UPSTREAM_SPECS,
) -> dict[str, object]:
    """Link ``repo/upstreams`` to the validated ``tools_root/upstreams``.

    Both roots must be absolute.  No existing non-matching path is replaced,
    and a link created by this invocation is removed only if its exact identity
    is still unchanged when post-creation verification fails.
    """

    repository = _absolute_existing_directory(repo, label="repo")
    persistent_tools = _absolute_existing_directory(tools_root, label="tools root")
    _require_repository_root(repository)
    _require_ignored_and_untracked(repository)

    source_root = _absolute_existing_directory(
        persistent_tools / "upstreams",
        label="persistent upstream root",
    )
    if source_root == repository or repository in source_root.parents:
        raise UpstreamLinkError(
            f"persistent upstream root must be outside the repository: {source_root}"
        )
    inventory = _checkout_inventory(source_root, tuple(specs))

    link = repository / REPOSITORY_PATH
    if _lexists(link):
        if not _existing_link_matches(link, source_root):
            raise UpstreamLinkError(f"refusing to overwrite existing repository path: {link}")
        state = "existing"
    else:
        created: _CreatedLink | None = None
        try:
            os.symlink(source_root, link, target_is_directory=True)
            created = _created_link_identity(link, source_root)
            if created is None or not _existing_link_matches(link, source_root):
                raise UpstreamLinkError(
                    f"created upstream symlink failed verification: {link} -> {source_root}"
                )
        except FileExistsError as exc:
            if _existing_link_matches(link, source_root):
                state = "existing"
            else:
                raise UpstreamLinkError(
                    f"refusing concurrently created repository path: {link}"
                ) from exc
        except Exception as exc:
            rollback_error: OSError | None = None
            if created is not None and _same_created_link(created):
                try:
                    created.path.unlink()
                except OSError as rollback_exc:
                    rollback_error = rollback_exc
            if rollback_error is not None:
                raise UpstreamLinkError(
                    f"{exc}; rollback incomplete for {link}: {rollback_error}"
                ) from exc
            if isinstance(exc, UpstreamLinkError):
                raise
            raise UpstreamLinkError(
                f"cannot create upstream symlink: {link} -> {source_root}"
            ) from exc
        else:
            state = "created"

    return {
        "schema_version": "ecgcert-server-upstream-links/v1",
        "repo": str(repository),
        "tools_root": str(persistent_tools),
        "source_root": str(source_root),
        "repository_path": str(link),
        "state": state,
        "checkouts": inventory,
    }


def render_upstream_link_report(report: dict[str, object]) -> str:
    return json.dumps(report, indent=2, sort_keys=True) + "\n"


__all__ = [
    "UPSTREAM_SPECS",
    "UpstreamLinkError",
    "ensure_server_upstream_link",
    "render_upstream_link_report",
]
