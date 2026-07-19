"""Fail-closed placement of persistent ECG datasets into a repository checkout."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
from typing import Iterable


class DataLinkError(RuntimeError):
    """Raised when persistent data cannot be linked without changing existing paths."""


@dataclass(frozen=True)
class DataLinkSpec:
    cohort: str
    repository_path: PurePosixPath


@dataclass(frozen=True)
class _CreatedDirectory:
    path: Path
    device: int
    inode: int


@dataclass(frozen=True)
class _CreatedLink:
    path: Path
    target: Path
    device: int
    inode: int
    ctime_ns: int


DATA_LINK_SPECS = (
    DataLinkSpec("ptbxl", PurePosixPath("data/ptbxl")),
    DataLinkSpec("chapman", PurePosixPath("data/external/chapman")),
    DataLinkSpec("cpsc2018", PurePosixPath("data/external/cpsc2018")),
)


def _absolute_existing_directory(raw: Path | str, *, label: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        raise DataLinkError(f"{label} must be an absolute path: {path}")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise DataLinkError(f"{label} does not exist or cannot be resolved: {path}") from exc
    if not resolved.is_dir():
        raise DataLinkError(f"{label} is not a directory: {resolved}")
    return resolved


def _git(repo: Path, arguments: Iterable[str]) -> subprocess.CompletedProcess[str]:
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
        raise DataLinkError("cannot inspect repository safety constraints with git") from exc


def _require_repository_root(repo: Path) -> None:
    completed = _git(repo, ("rev-parse", "--show-toplevel"))
    if completed.returncode:
        raise DataLinkError(f"repository path is not a git worktree: {repo}")
    try:
        top_level = Path(completed.stdout.strip()).resolve(strict=True)
    except OSError as exc:
        raise DataLinkError("git returned an invalid worktree root") from exc
    if top_level != repo:
        raise DataLinkError(
            f"repository path must be the absolute git worktree root: {repo} != {top_level}"
        )


def _require_ignored_and_untracked(repo: Path, relative: PurePosixPath) -> None:
    rendered = relative.as_posix()
    ignored = _git(repo, ("check-ignore", "--quiet", "--no-index", "--", rendered))
    if ignored.returncode != 0:
        raise DataLinkError(f"refusing non-ignored repository data path: {rendered}")
    tracked = _git(repo, ("ls-files", "--", rendered, f"{rendered}/**"))
    if tracked.returncode:
        raise DataLinkError(f"cannot inspect tracked state for repository path: {rendered}")
    if tracked.stdout.strip():
        raise DataLinkError(f"refusing tracked repository data path: {rendered}")


def _lexists(path: Path) -> bool:
    return os.path.lexists(path)


def _require_real_directory_or_missing(path: Path, *, repo: Path) -> None:
    if not _lexists(path):
        return
    if path.is_symlink() or not path.is_dir():
        raise DataLinkError(f"repository data parent is not a real directory: {path}")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise DataLinkError(f"repository data parent cannot be resolved: {path}") from exc
    if resolved != repo and repo not in resolved.parents:
        raise DataLinkError(f"repository data parent escapes the worktree: {path}")


def _existing_link_matches(link: Path, target: Path) -> bool:
    if not link.is_symlink():
        return False
    try:
        raw_target = Path(os.readlink(link))
        resolved = link.resolve(strict=True)
    except OSError:
        return False
    return raw_target.is_absolute() and raw_target == target and resolved == target


def _raw_link_target_matches(link: Path, target: Path) -> bool:
    """Check link identity without requiring its persistent target to still exist."""

    if not link.is_symlink():
        return False
    try:
        raw_target = Path(os.readlink(link))
    except OSError:
        return False
    return raw_target.is_absolute() and raw_target == target


def _created_link_identity(link: Path, target: Path) -> _CreatedLink | None:
    if not _raw_link_target_matches(link, target):
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


def _directory_identity(path: Path) -> _CreatedDirectory:
    try:
        status = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise DataLinkError(f"created repository data parent cannot be inspected: {path}") from exc
    if not stat.S_ISDIR(status.st_mode):
        raise DataLinkError(f"created repository data parent is not a directory: {path}")
    return _CreatedDirectory(
        path=path,
        device=status.st_dev,
        inode=status.st_ino,
    )


def _create_parent_if_missing(path: Path, *, repo: Path) -> _CreatedDirectory | None:
    if _lexists(path):
        _require_real_directory_or_missing(path, repo=repo)
        return None
    try:
        path.mkdir()
    except FileExistsError:
        # A concurrent invocation may have created the same parent. It is not ours
        # and therefore must never be removed by this transaction's rollback.
        if not _lexists(path):
            raise DataLinkError(
                f"repository data parent disappeared during concurrent creation: {path}"
            )
        _require_real_directory_or_missing(path, repo=repo)
        return None
    except OSError as exc:
        raise DataLinkError(f"cannot create repository data parent: {path}") from exc
    created = _directory_identity(path)
    try:
        _require_real_directory_or_missing(path, repo=repo)
    except Exception as exc:
        rollback_errors = _rollback_created_layout(links=[], directories=[created])
        if rollback_errors:
            detail = "; ".join(rollback_errors)
            raise DataLinkError(f"{exc}; rollback incomplete: {detail}") from exc
        raise
    return created


def _same_empty_directory(created: _CreatedDirectory) -> bool:
    try:
        status = created.path.stat(follow_symlinks=False)
    except OSError:
        return False
    if (
        not stat.S_ISDIR(status.st_mode)
        or status.st_dev != created.device
        or status.st_ino != created.inode
    ):
        return False
    try:
        next(created.path.iterdir())
    except StopIteration:
        return True
    except OSError:
        return False
    return False


def _same_created_link(created: _CreatedLink) -> bool:
    if not _raw_link_target_matches(created.path, created.target):
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


def _rollback_created_layout(
    *,
    links: list[_CreatedLink],
    directories: list[_CreatedDirectory],
) -> list[str]:
    """Undo only unchanged objects known to have been created by this call."""

    errors: list[str] = []
    for created in reversed(links):
        if not _same_created_link(created):
            continue
        try:
            created.path.unlink()
        except OSError as exc:
            errors.append(f"cannot remove created link {created.path}: {exc}")
    for created in reversed(directories):
        if not _same_empty_directory(created):
            continue
        try:
            created.path.rmdir()
        except OSError as exc:
            errors.append(f"cannot remove created parent {created.path}: {exc}")
    return errors


def ensure_server_data_links(
    *,
    repo: Path | str,
    data_root: Path | str,
) -> dict[str, object]:
    """Create the three exact ignored data links without overwriting any path.

    Both arguments must be absolute. Every source and destination is validated before
    the first directory or symlink is created, so a missing cohort or collision cannot
    leave a partial layout. Existing exact links are accepted as an idempotent no-op.
    """

    repository = _absolute_existing_directory(repo, label="repo")
    persistent_root = _absolute_existing_directory(data_root, label="data root")
    _require_repository_root(repository)

    data_parent = repository / "data"
    external_parent = data_parent / "external"
    _require_real_directory_or_missing(data_parent, repo=repository)
    _require_real_directory_or_missing(external_parent, repo=repository)

    prepared: list[tuple[DataLinkSpec, Path, Path, str]] = []
    for spec in DATA_LINK_SPECS:
        _require_ignored_and_untracked(repository, spec.repository_path)
        source = _absolute_existing_directory(
            persistent_root / spec.cohort,
            label=f"{spec.cohort} target",
        )
        if source == repository or repository in source.parents:
            raise DataLinkError(
                f"{spec.cohort} target must be outside the repository: {source}"
            )
        link = repository.joinpath(*spec.repository_path.parts)
        if _lexists(link):
            if not _existing_link_matches(link, source):
                raise DataLinkError(f"refusing to overwrite existing repository path: {link}")
            state = "existing"
        else:
            state = "missing"
        prepared.append((spec, source, link, state))

    created_directories: list[_CreatedDirectory] = []
    created_links: list[_CreatedLink] = []
    results: list[dict[str, str]] = []
    try:
        for parent in (data_parent, external_parent):
            created = _create_parent_if_missing(parent, repo=repository)
            if created is not None:
                created_directories.append(created)

        for spec, source, link, state in prepared:
            if state == "missing":
                try:
                    os.symlink(source, link, target_is_directory=True)
                except FileExistsError as exc:
                    # Never claim or remove an object created by a racing process.
                    if _existing_link_matches(link, source):
                        state = "existing"
                    else:
                        raise DataLinkError(
                            f"refusing concurrently created repository path: {link}"
                        ) from exc
                except OSError as exc:
                    raise DataLinkError(
                        f"cannot create dataset symlink: {link} -> {source}"
                    ) from exc
                else:
                    created_link = _created_link_identity(link, source)
                    if created_link is not None:
                        created_links.append(created_link)
                    if not _existing_link_matches(link, source):
                        raise DataLinkError(
                            f"created dataset symlink failed verification: {link}"
                        )
                    state = "created"
            results.append(
                {
                    "cohort": spec.cohort,
                    "repository_path": str(link),
                    "target": str(source),
                    "state": state,
                }
            )
    except Exception as exc:
        rollback_errors = _rollback_created_layout(
            links=created_links,
            directories=created_directories,
        )
        if rollback_errors:
            detail = "; ".join(rollback_errors)
            raise DataLinkError(f"{exc}; rollback incomplete: {detail}") from exc
        if isinstance(exc, DataLinkError):
            raise
        raise DataLinkError(f"cannot create repository data links: {exc}") from exc

    return {
        "schema_version": 1,
        "repo": str(repository),
        "data_root": str(persistent_root),
        "links": results,
    }


def render_link_report(report: dict[str, object]) -> str:
    return json.dumps(report, indent=2, sort_keys=True) + "\n"
