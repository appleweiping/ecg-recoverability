"""Pytest config: the STRICT clean-tree release gate (tests/test_release_strict.py) is collected
ONLY when ECG_RELEASE_STRICT=1, so the default and ECG_RELEASE=1 suites stay zero-skip.
"""
import os
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

collect_ignore = []
if not os.environ.get("ECG_RELEASE_STRICT"):
    collect_ignore.append("test_release_strict.py")


def _reviewer_keypair(root):
    private_key = Ed25519PrivateKey.generate()
    root.mkdir(parents=True, exist_ok=True)
    private_path = root / "reviewer_ed25519"
    public_path = root / "reviewer_ed25519.pub"
    private_path.write_bytes(
        private_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    public_path.write_bytes(
        private_key.public_key().public_bytes(
            serialization.Encoding.OpenSSH,
            serialization.PublicFormat.OpenSSH,
        )
        + b" test-reviewer\n"
    )
    return SimpleNamespace(private=private_path, public=public_path)


@pytest.fixture
def reviewer_keys(tmp_path):
    """Ephemeral, repository-external Ed25519 keys for gate-authentication tests."""
    return _reviewer_keypair(tmp_path / "signer")


@pytest.fixture
def wrong_reviewer_keys(tmp_path):
    return _reviewer_keypair(tmp_path / "wrong-signer")
