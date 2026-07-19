"""Forward one validated Ed25519 stage review to the active native ARC pause."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ecgcert.arc_forward import build_operator_response
from ecgcert.stage_gates import DEFAULT_REVIEWER_PUBLIC_KEY, json_artifact_bytes


def _load(path: Path) -> tuple[dict, bytes]:
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value, raw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--waiting-report", type=Path, required=True)
    parser.add_argument("--reviewed-gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--public-key", type=Path, default=DEFAULT_REVIEWER_PUBLIC_KEY
    )
    arguments = parser.parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    public_key = arguments.public_key.expanduser().resolve(strict=True)
    if not public_key.is_relative_to(repository_root):
        raise SystemExit("reviewer public key must be pinned inside the repository")
    try:
        waiting, waiting_raw = _load(arguments.waiting_report)
        reviewed, _reviewed_raw = _load(arguments.reviewed_gate)
        response = build_operator_response(
            waiting_report=waiting,
            waiting_report_sha256=hashlib.sha256(waiting_raw).hexdigest(),
            reviewed_gate=reviewed,
            reviewer_public_key=public_key,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise SystemExit(f"cannot forward ARC stage review: {error}") from error
    output = arguments.output.resolve()
    rendered = json_artifact_bytes(response)
    if output.exists():
        try:
            conflicts = output.is_symlink() or output.read_bytes() != rendered
        except OSError as error:
            raise SystemExit(f"cannot verify existing ARC operator response: {error}") from error
        if conflicts:
            raise SystemExit("existing ARC operator response has conflicting content")
        print(f"recovered existing signed Stage {response['stage']} ARC response")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_bytes(rendered)
    temporary.replace(output)
    print(f"forwarded signed Stage {response['stage']} review to native ARC")


if __name__ == "__main__":
    main()
