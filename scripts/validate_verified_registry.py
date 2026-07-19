"""Fail closed when manuscript citations or result macros escape the registry."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ecgcert.verified_registry import (  # re-exported for existing callers/tests
    CITE_RE,
    RESULT_RE,
    VERIFIED,
    manuscript_keys,
    validate,
)

__all__ = ["CITE_RE", "RESULT_RE", "VERIFIED", "manuscript_keys", "validate"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--manuscript", type=Path, default=root / "paper" / "main_v2.tex")
    parser.add_argument(
        "--registry", type=Path, default=root / "arc_audit" / "verified_registry.v1.json"
    )
    parser.add_argument("--require-verified", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate(args.manuscript, args.registry, require_verified=args.require_verified)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
