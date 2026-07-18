"""Validate one hash-bound official AutoResearchClaw control bundle."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ecgcert.arc_control import (
    SUPPORTED_STAGES,
    ArcControlValidationError,
    validate_arc_control_bundle,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle",
        type=Path,
        required=True,
        help="bundle root containing receipt.v1.json and native ARC run artifacts",
    )
    parser.add_argument(
        "--stage",
        type=int,
        choices=sorted(SUPPORTED_STAGES),
        required=True,
        help="ARC stage that the caller expects (never inferred from the receipt)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="optional new file for the normalized validation report",
    )
    arguments = parser.parse_args()
    try:
        report = validate_arc_control_bundle(arguments.bundle, arguments.stage)
    except ArcControlValidationError as error:
        raise SystemExit(f"ARC control validation failed: {error}") from error

    rendered = (
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        + "\n"
    )
    if arguments.output is not None:
        output = arguments.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            with output.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(rendered)
        except FileExistsError as error:
            raise SystemExit(
                f"ARC control report already exists and is immutable: {output}"
            ) from error
    print(rendered, end="")


if __name__ == "__main__":
    main()
