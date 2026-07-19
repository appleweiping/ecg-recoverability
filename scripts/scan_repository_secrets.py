"""Create the clean-commit repository secret-scan artifact required by Stage 9."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from ecgcert.security_scan import scan_repository


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository",
        type=Path,
        default=Path(
            os.environ.get("ECG_SOURCE_REPO", Path(__file__).resolve().parents[1])
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report = scan_repository(arguments.repository, arguments.output)
    if report["status"] != "complete":
        raise SystemExit(
            "repository secret scan failed; inspect rule names and paths in the report"
        )
    print(
        "Repository secret scan passed: "
        f"{report['scope']['scanned_files']} text files, 0 findings"
    )


if __name__ == "__main__":
    main()
