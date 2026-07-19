"""Wait up to 24 hours for one official ARC co-pilot receipt bundle."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import time

from ecgcert.arc_control import (
    SUPPORTED_STAGES,
    ArcControlValidationError,
    validate_arc_control_bundle,
    validate_arc_control_report,
    validate_arc_waiting_bundle,
    validate_arc_waiting_report,
)
from ecgcert.execution.late_inputs import (
    LateControlInputError,
    capture_late_control_input,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--stage", type=int, choices=sorted(SUPPORTED_STAGES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--phase",
        choices=("waiting", "final"),
        default="final",
        help="wait for the pre-approval pause or the post-approval formal handoff",
    )
    parser.add_argument(
        "--previous-report",
        type=Path,
        help="required prior formal ARC report for Stages 9, 15, and 20",
    )
    parser.add_argument("--timeout-hours", type=float, default=24.0)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    arguments = parser.parse_args()
    if not 0 < arguments.timeout_hours <= 24:
        raise SystemExit("--timeout-hours must be in (0, 24]")
    if not 0.1 <= arguments.poll_seconds <= 60:
        raise SystemExit("--poll-seconds must be in [0.1, 60]")
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
    output = arguments.output.resolve()
    deadline = datetime.now(timezone.utc) + timedelta(hours=arguments.timeout_hours)
    receipt_name = (
        "waiting-receipt.v1.json" if arguments.phase == "waiting" else "receipt.v1.json"
    )
    receipt = arguments.bundle / receipt_name
    while not receipt.is_file():
        if datetime.now(timezone.utc) >= deadline:
            raise SystemExit(
                f"timed out waiting for official ARC Stage {arguments.stage} "
                f"{arguments.phase} receipt"
            )
        time.sleep(arguments.poll_seconds)
    try:
        captured_bundle = capture_late_control_input(
            arguments.bundle,
            require_policy=True,
        )
    except LateControlInputError as error:
        raise SystemExit(f"cannot atomically capture ARC control bundle: {error}") from error
    try:
        validator = (
            validate_arc_waiting_bundle
            if arguments.phase == "waiting"
            else validate_arc_control_bundle
        )
        report = validator(
            captured_bundle, arguments.stage, previous_report=previous_report
        )
    except ArcControlValidationError as error:
        raise SystemExit(f"ARC control validation failed: {error}") from error
    report_validator = (
        validate_arc_waiting_report
        if arguments.phase == "waiting"
        else validate_arc_control_report
    )
    if output.exists():
        try:
            existing = json.loads(output.read_text(encoding="utf-8"))
            normalized = report_validator(
                existing,
                arguments.stage,
                previous_report=previous_report,
            )
        except (
            OSError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            ArcControlValidationError,
        ) as error:
            raise SystemExit(
                f"existing ARC {arguments.phase} report is not recoverable: {error}"
            ) from error
        if normalized != report:
            raise SystemExit(
                f"existing ARC {arguments.phase} report conflicts with the receipt bundle"
            )
        print(
            f"recovered official ARC Stage {arguments.stage} "
            f"{arguments.phase} report: {output}"
        )
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(
        report, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False,
    ) + "\n"
    try:
        with output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(rendered)
    except FileExistsError as error:
        try:
            concurrent = json.loads(output.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as read_error:
            raise SystemExit(
                f"ARC control report appeared concurrently but is unreadable: {output}"
            ) from read_error
        if report_validator(
            concurrent, arguments.stage, previous_report=previous_report
        ) != report:
            raise SystemExit(f"conflicting ARC control report appeared: {output}") from error
    print(
        f"validated official ARC Stage {arguments.stage} "
        f"{arguments.phase} receipt: {output}"
    )


if __name__ == "__main__":
    main()
