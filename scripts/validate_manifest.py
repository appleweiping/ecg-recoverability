"""Validate and summarize the canonical experiment manifest without running anything."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ecgcert.execution import ExperimentManifest  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest", nargs="?", default="scripts/experiment_manifest.yaml")
    args = ap.parse_args(argv)
    path = Path(args.manifest)
    if not path.is_absolute():
        path = ROOT / path
    manifest = ExperimentManifest.from_path(path)
    print(f"sha256: {manifest.sha256()}")
    for profile in ("icassp", "extended", "legacy"):
        nodes = manifest.select(profile)
        print(f"{profile}: {len(nodes)} nodes: " + ", ".join(node.id for node in nodes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
