"""Materialize clean, exact-commit upstream baseline checkouts.

The canonical evidence DAG uses ``--offline --source-root upstreams``.  That
mode validates persistent source checkouts first, makes local no-hardlink
clones, and never runs ``fetch`` or addresses a network repository.  The
online mode remains only as an explicit bootstrap for the persistent tool
volume before a frozen evidence run starts.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Iterable

from ecgcert.estimators.official import (
    ECG_RECOVER,
    IMPUTE_ECG,
    UpstreamSpec,
    validate_pinned_checkout,
)


SPECS = {"imputeecg": IMPUTE_ECG, "ecgrecover": ECG_RECOVER}
IMPUTEECG_SPARSE_PATHS = (
    "/README.md",
    "/datasets/",
    "/downstream/",
    "/inference.py",
    "/models/",
    "/single_kailun_csv_to_npy_png.py",
    "/train.py",
    "/utils/",
)
ECGRECOVER_SPARSE_PATHS = (
    "/main.py",
    "/readme.md",
    "/environment.yml",
    "/learn/",
    "!/learn/__pycache__/",
    "/tools/",
    "!/tools/__pycache__/",
    "/compute_metrics/",
)


def checkout_name(spec: UpstreamSpec) -> str:
    return f"{spec.name}-{spec.commit[:12]}"


def _lexists(path: Path) -> bool:
    return os.path.lexists(path)


def _git(curl_resolve: str | None, *arguments: str) -> list[str]:
    command = ["git"]
    if curl_resolve:
        command.extend(["-c", f"http.curloptResolve={curl_resolve}"])
    command.extend(arguments)
    return command


def _offline_environment() -> dict[str, str]:
    """Return a Git environment which permits only local-file transport."""

    environment = os.environ.copy()
    environment.update(
        {
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_LFS_SKIP_SMUDGE": "1",
            "GIT_NO_LAZY_FETCH": "1",
            "GIT_CONFIG_COUNT": "2",
            "GIT_CONFIG_KEY_0": "protocol.allow",
            "GIT_CONFIG_VALUE_0": "never",
            "GIT_CONFIG_KEY_1": "protocol.file.allow",
            "GIT_CONFIG_VALUE_1": "always",
        }
    )
    return environment


def _run_offline_git(*arguments: str, cwd: Path | None = None) -> None:
    subprocess.run(
        ["git", *arguments],
        cwd=cwd,
        env=_offline_environment(),
        check=True,
    )


def _offline_git_output(*arguments: str, cwd: Path) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=cwd,
        env=_offline_environment(),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _sparse_paths(spec: UpstreamSpec) -> tuple[str, ...]:
    if spec.name.casefold() == IMPUTE_ECG.name.casefold():
        return IMPUTEECG_SPARSE_PATHS
    if spec.name.casefold() == ECG_RECOVER.name.casefold():
        return ECGRECOVER_SPARSE_PATHS
    return ()


def _configure_sparse_checkout(target: Path, spec: UpstreamSpec) -> None:
    paths = _sparse_paths(spec)
    if not paths:
        return
    _run_offline_git("sparse-checkout", "init", "--no-cone", cwd=target)
    _run_offline_git(
        "sparse-checkout",
        "set",
        "--no-cone",
        *paths,
        cwd=target,
    )


def checkout(
    spec: UpstreamSpec,
    destination: str | Path,
    *,
    curl_resolve: str | None = None,
) -> Path:
    """Bootstrap one persistent checkout from its official network origin."""

    root = Path(destination).resolve()
    target = root / checkout_name(spec)
    root.mkdir(parents=True, exist_ok=True)
    if not _lexists(target):
        subprocess.run(
            _git(
                curl_resolve,
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                spec.repository,
                str(target),
            ),
            check=True,
        )
    if not (target / ".git").is_dir():
        raise ValueError(f"existing target is not a Git checkout: {target}")
    subprocess.run(
        _git(
            curl_resolve,
            "-C",
            str(target),
            "fetch",
            "--filter=blob:none",
            "origin",
            spec.commit,
        ),
        check=True,
    )
    sparse_paths = _sparse_paths(spec)
    if sparse_paths:
        subprocess.run(
            ["git", "-C", str(target), "sparse-checkout", "init", "--no-cone"],
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(target),
                "sparse-checkout",
                "set",
                "--no-cone",
                *sparse_paths,
            ],
            check=True,
        )
    subprocess.run(
        _git(curl_resolve, "-C", str(target), "checkout", "--detach", spec.commit),
        check=True,
    )
    validate_pinned_checkout(target, spec)
    return target


def _validated_sources(
    specs: Iterable[UpstreamSpec],
    source_root: Path,
) -> dict[str, Path]:
    try:
        persistent_root = source_root.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"offline source root does not exist: {source_root}") from exc
    if not persistent_root.is_dir():
        raise ValueError(f"offline source root is not a directory: {persistent_root}")

    sources: dict[str, Path] = {}
    for spec in specs:
        if spec.root_tree is None:
            raise ValueError(f"offline source requires a frozen root tree for {spec.name}")
        source = persistent_root / checkout_name(spec)
        validate_pinned_checkout(source, spec)
        sources[spec.name] = source.resolve(strict=True)
    return sources


def _clone_offline(source: Path, target: Path, spec: UpstreamSpec) -> None:
    if _lexists(target):
        raise FileExistsError(f"refusing to overwrite offline checkout target: {target}")
    _run_offline_git(
        "clone",
        "--local",
        "--no-hardlinks",
        "--no-checkout",
        "--no-tags",
        str(source),
        str(target),
    )
    # A persistent source may be a legitimate promisor checkout whose
    # intentionally omitted, non-runtime blobs cannot be repacked by
    # upload-pack.  A no-hardlink local clone preserves those promises without
    # trying to hydrate them.  Expiring clone-created reflogs and pruning then
    # removes any source-only loose objects, so the evidence checkout contains
    # only objects reachable from its frozen refs plus explicit promises.
    _run_offline_git("config", "remote.origin.promisor", "true", cwd=target)
    _run_offline_git(
        "config", "remote.origin.partialclonefilter", "blob:none", cwd=target
    )
    _run_offline_git("config", "protocol.allow", "never", cwd=target)
    _run_offline_git("reflog", "expire", "--expire=now", "--all", cwd=target)
    _run_offline_git("prune", "--no-progress", "--expire=now", cwd=target)
    _configure_sparse_checkout(target, spec)
    _run_offline_git("checkout", "--detach", spec.commit, cwd=target)
    # Store the canonical public origin only after all object transfer and
    # checkout operations are complete. No hard link can couple the artifact
    # to persistent storage. Retain promisor metadata for intentionally absent
    # non-runtime blobs while a repository-local transport deny makes any lazy
    # network retrieval fail closed.
    _run_offline_git("remote", "set-url", "origin", spec.repository, cwd=target)
    _run_offline_git("fsck", "--connectivity-only", "--no-dangling", cwd=target)
    _validate_offline_clone(target, spec)


def _validate_offline_clone(target: Path, spec: UpstreamSpec) -> None:
    validate_pinned_checkout(target, spec)
    if _lexists(target / ".git" / "objects" / "info" / "alternates"):
        raise ValueError(f"offline checkout borrows an external object store: {target}")
    expected_config = {
        "protocol.allow": "never",
        "remote.origin.promisor": "true",
        "remote.origin.partialclonefilter": "blob:none",
    }
    for key, expected in expected_config.items():
        actual = _offline_git_output("config", "--get", key, cwd=target)
        if actual != expected:
            raise ValueError(
                f"offline checkout {key} is {actual!r}, expected {expected!r}: {target}"
            )


def checkout_offline(
    specs: Iterable[UpstreamSpec],
    destination: str | Path,
    *,
    source_root: str | Path,
) -> dict[str, Path]:
    """Materialize validated local sources without network access.

    The complete destination must be absent.  Every source is validated before
    creating a staging directory.  After every clone passes a second validation,
    an exclusive destination-directory creation reserves the final path without
    replacing even an empty directory created by another process.
    """

    selected = tuple(specs)
    if not selected:
        raise ValueError("at least one upstream specification is required")
    sources = _validated_sources(selected, Path(source_root))
    root = Path(destination).resolve()
    if _lexists(root):
        raise FileExistsError(f"refusing to overwrite offline destination: {root}")
    if root == Path(source_root).resolve() or Path(source_root).resolve() in root.parents:
        raise ValueError("offline destination must not be inside the persistent source root")

    root.parent.mkdir(parents=True, exist_ok=True)
    if not root.parent.is_dir():
        raise ValueError(f"offline destination parent is not a directory: {root.parent}")
    staging = Path(
        tempfile.mkdtemp(prefix=f".{root.name}.offline-", dir=str(root.parent))
    )
    try:
        for spec in selected:
            _clone_offline(
                sources[spec.name],
                staging / checkout_name(spec),
                spec,
            )
        try:
            root.mkdir()
        except FileExistsError as exc:
            raise FileExistsError(
                f"refusing concurrently created destination: {root}"
            ) from exc
        for spec in selected:
            source = staging / checkout_name(spec)
            target = root / checkout_name(spec)
            source.rename(target)
    finally:
        if _lexists(staging):
            shutil.rmtree(staging)

    materialized: dict[str, Path] = {}
    for spec in selected:
        target = root / checkout_name(spec)
        _validate_offline_clone(target, spec)
        materialized[spec.name] = target
    return materialized


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=[*SPECS, "all"], default="all")
    parser.add_argument("--destination", required=True)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="clone exclusively from --source-root with file transport only",
    )
    parser.add_argument(
        "--source-root",
        help="validated persistent checkout root used by --offline",
    )
    parser.add_argument(
        "--curl-resolve",
        help=(
            "optional libcurl resolve entry host:port:address for bootstrap DNS failures; "
            "forbidden in offline mode and never stored in the checkout"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.offline and not args.source_root:
        parser.error("--offline requires --source-root")
    if args.source_root and not args.offline:
        parser.error("--source-root is accepted only with --offline")
    if args.offline and args.curl_resolve:
        parser.error("--curl-resolve is forbidden with --offline")

    names = tuple(SPECS) if args.model == "all" else (args.model,)
    selected = tuple(SPECS[name] for name in names)
    if args.offline:
        targets = checkout_offline(
            selected,
            args.destination,
            source_root=args.source_root,
        )
        for name in names:
            spec = SPECS[name]
            print(f"[{name}] {spec.commit} -> {targets[spec.name]} [offline]")
    else:
        for name in names:
            target = checkout(
                SPECS[name],
                args.destination,
                curl_resolve=args.curl_resolve,
            )
            print(f"[{name}] {SPECS[name].commit} -> {target} [bootstrap]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
