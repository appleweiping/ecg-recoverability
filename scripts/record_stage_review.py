"""Create a Stage 5/9/15/20 human review bound to immutable gate evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from ecgcert.stage_gates import (
    DECISIONS,
    DEFAULT_REVIEWER_PUBLIC_KEY,
    json_artifact_bytes,
    make_review,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--decision", choices=DECISIONS, required=True)
    parser.add_argument(
        "--private-key",
        type=Path,
        required=True,
        help="Ed25519 private key stored outside the repository",
    )
    parser.add_argument(
        "--public-key",
        type=Path,
        default=DEFAULT_REVIEWER_PUBLIC_KEY,
        help="repository-pinned Ed25519 reviewer public key",
    )
    arguments = parser.parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    public_key = arguments.public_key.expanduser().resolve(strict=True)
    if not public_key.is_relative_to(repository_root):
        raise SystemExit("reviewer public key must be pinned inside the repository")
    passphrase = os.environ.get("ECGCERT_REVIEW_KEY_PASSPHRASE")
    raw_gate = arguments.gate.read_bytes()
    gate = json.loads(raw_gate.decode("utf-8"))
    gate_sha256 = hashlib.sha256(raw_gate).hexdigest()
    try:
        payload = make_review(
            gate,
            gate_sha256=gate_sha256,
            reviewer=arguments.reviewer,
            decision=arguments.decision,
            private_key_path=arguments.private_key,
            public_key_path=public_key,
            repository_root=repository_root,
            private_key_password=(passphrase.encode("utf-8") if passphrase else None),
        )
    except (TimeoutError, ValueError) as error:
        raise SystemExit(str(error)) from error
    if arguments.output.exists():
        raise SystemExit(
            f"review artifact already exists and is immutable: {arguments.output}"
        )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_suffix(arguments.output.suffix + ".tmp")
    temporary.write_bytes(json_artifact_bytes(payload))
    temporary.replace(arguments.output)
    print(
        f"Stage {payload['stage']} review recorded as {arguments.decision} "
        f"by {payload['reviewer']}"
    )


if __name__ == "__main__":
    main()
