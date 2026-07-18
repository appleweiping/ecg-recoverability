"""Pytest config: the STRICT clean-tree release gate (tests/test_release_strict.py) is collected
ONLY when ECG_RELEASE_STRICT=1, so the default and ECG_RELEASE=1 suites stay zero-skip.
"""
import os

collect_ignore = []
if not os.environ.get("ECG_RELEASE_STRICT"):
    collect_ignore.append("test_release_strict.py")
