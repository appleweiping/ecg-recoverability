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
    parser.add_argument(
        "--previous-report",
        type=Path,
        help="required prior formal ARC report for Stages 9, 15, and 20",
    )
    arguments = parser.parse_args()
    if arguments.stage == 5 and arguments.previous_report is not None:
        raise SystemExit("Stage 5 must not specify --previous-report")
    if arguments.stage != 5 and arguments.previous_report is None:
        raise SystemExit(
            f"Stage {arguments.stage} requires --previous-report for formal lineage"
        )
    previous_report = None
    if arguments.previous_report is not None:
        try:
            previous_report = json.loads(
                arguments.previous_report.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise SystemExit(f"cannot read prior ARC control report: {error}") from error
        if not isinstance(previous_report, dict):
            raise SystemExit("prior ARC control report must be a JSON object")
    try:
        report = validate_arc_control_bundle(
            arguments.bundle,
            arguments.stage,
            previous_report=previous_report,
        )
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
