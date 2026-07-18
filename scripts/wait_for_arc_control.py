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
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--stage", type=int, choices=sorted(SUPPORTED_STAGES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-hours", type=float, default=24.0)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    arguments = parser.parse_args()
    if not 0 < arguments.timeout_hours <= 24:
        raise SystemExit("--timeout-hours must be in (0, 24]")
    if not 0.1 <= arguments.poll_seconds <= 60:
        raise SystemExit("--poll-seconds must be in [0.1, 60]")
    output = arguments.output.resolve()
    if output.exists():
        raise SystemExit(f"ARC control report already exists: {output}")
    deadline = datetime.now(timezone.utc) + timedelta(hours=arguments.timeout_hours)
    receipt = arguments.bundle / "receipt.v1.json"
    while not receipt.is_file():
        if datetime.now(timezone.utc) >= deadline:
            raise SystemExit(
                f"timed out waiting for official ARC Stage {arguments.stage} receipt"
            )
        time.sleep(arguments.poll_seconds)
    try:
        report = validate_arc_control_bundle(arguments.bundle, arguments.stage)
    except ArcControlValidationError as error:
        raise SystemExit(f"ARC control validation failed: {error}") from error
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(
        report, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False,
    ) + "\n"
    try:
        with output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(rendered)
    except FileExistsError as error:
        raise SystemExit(f"ARC control report appeared concurrently: {output}") from error
    print(f"validated official ARC Stage {arguments.stage}: {output}")


if __name__ == "__main__":
    main()
