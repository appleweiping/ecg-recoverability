"""Pause an ARC Stage 5/9/15/20 gate for at most 24 hours, failing closed."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

from ecgcert import lineage
from ecgcert.execution.late_inputs import (
    LateControlInputError,
    capture_late_control_input,
)
from ecgcert.stage_gates import (
    DEFAULT_REVIEWER_PUBLIC_KEY,
    json_artifact_bytes,
    merge_review,
    review_deadline,
    validate_gate,
    validate_review,
)


def _valid_review(
    review: dict,
    gate: dict,
    gate_path: Path,
    public_key_path: Path = DEFAULT_REVIEWER_PUBLIC_KEY,
) -> bool:
    """Compatibility wrapper used by tests and callers that only need a boolean."""
    try:
        validate_review(
            review,
            gate,
            gate_sha256=lineage.artifact_sha256(gate_path),
            public_key_path=public_key_path,
        )
    except (TimeoutError, ValueError):
        return False
    return True


def wait_for_review(
    *,
    gate_path: Path,
    approval_path: Path,
    timeout_hours: float,
    poll_seconds: float,
    public_key_path: Path = DEFAULT_REVIEWER_PUBLIC_KEY,
    require_capture_policy: bool = False,
) -> tuple[dict, dict]:
    """Return the combined decision and mandatory review artifact."""
    raw_gate = gate_path.read_bytes()
    gate = json.loads(raw_gate.decode("utf-8"))
    stage = validate_gate(gate)
    gate_sha256 = hashlib.sha256(raw_gate).hexdigest()
    # A valid approval written before the deadline remains resumable even when the
    # DAG process itself is restarted after that deadline.
    wall_remaining = (
        review_deadline(gate) - datetime.now(timezone.utc)
    ).total_seconds()
    budget_seconds = min(timeout_hours * 3600.0, max(0.0, wall_remaining))
    monotonic_deadline = time.monotonic() + budget_seconds
    while True:
        if approval_path.is_file():
            try:
                captured_approval = capture_late_control_input(
                    approval_path,
                    require_policy=require_capture_policy,
                )
            except LateControlInputError as error:
                raise ValueError(
                    f"cannot atomically capture signed approval: {error}"
                ) from error
            raw_approval = captured_approval.read_bytes()
            candidate = json.loads(raw_approval.decode("utf-8"))
            if raw_approval != json_artifact_bytes(candidate):
                raise ValueError("approval artifact is not in the canonical signed encoding")
            approval_sha256 = hashlib.sha256(raw_approval).hexdigest()
            validate_review(
                candidate,
                gate,
                gate_sha256=gate_sha256,
                public_key_path=public_key_path,
            )
            combined = merge_review(
                gate,
                candidate,
                gate_sha256=gate_sha256,
                approval_sha256=approval_sha256,
                public_key_path=public_key_path,
            )
            return combined, candidate
        remaining = monotonic_deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"Stage-{stage} review was not supplied within 24 hours")
        time.sleep(min(poll_seconds, remaining))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout-hours", type=float, default=24.0)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument(
        "--public-key",
        type=Path,
        default=DEFAULT_REVIEWER_PUBLIC_KEY,
        help="repository-pinned Ed25519 reviewer public key",
    )
    arguments = parser.parse_args()
    if not 0 < arguments.timeout_hours <= 24 or not 0.5 <= arguments.poll_seconds <= 60:
        raise SystemExit("timeout must be in (0,24] hours and poll interval in [0.5,60] seconds")
    repository_root = Path(__file__).resolve().parents[1]
    public_key = arguments.public_key.expanduser().resolve(strict=True)
    if not public_key.is_relative_to(repository_root):
        raise SystemExit("reviewer public key must be pinned inside the repository")
    combined, _ = wait_for_review(
        gate_path=arguments.gate,
        approval_path=arguments.approval,
        timeout_hours=arguments.timeout_hours,
        poll_seconds=arguments.poll_seconds,
        public_key_path=public_key,
        require_capture_policy=True,
    )
    stage = int(combined["stage"])
    final_status = combined["status"]
    output = arguments.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "decision.v3.json"
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(json_artifact_bytes(combined))
    temporary.replace(destination)
    if final_status == "REFINE":
        raise SystemExit(
            f"Stage {stage} requested REFINE; downstream evidence remains blocked"
        )
    if final_status == "PIVOT" and stage != 15:
        raise SystemExit(f"Stage {stage} requested PIVOT; downstream evidence remains blocked")


if __name__ == "__main__":
    main()
