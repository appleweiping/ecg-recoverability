"""STRICT clean-tree release gate (P0-6). Collected ONLY under ECG_RELEASE_STRICT=1 (see
tests/conftest.py), so the default ECG_RELEASE=1 suite stays zero-skip. These tests fail-closed:
they FAIL when a cited artifact was computed on a dirty tree, is missing a checkpoint SHA, or was
produced by a now-stale script. A green run of these certifies the artifacts were regenerated from a
committed, clean checkout with the current code.
"""
import json
from pathlib import Path

from test_release_integrity import CITED_JSON            # reuse the single source of truth

ROOT = Path(__file__).resolve().parent.parent
# JSONs that are model results -> MUST carry a non-null checkpoint SHA
MODEL_JSON = ["neural_baseline", "gpu_deficit_ci", "gpu_oracle_gate", "realism_metrics",
              "fabrication_diffusion", "certificate_floor_diffusion"]
_SCRIPT_DIRS = ["experiments", "scripts", "paper", "."]


def _lineage(n):
    p = ROOT / "results" / f"{n}.json"
    assert p.exists(), f"strict release missing result: {n}.json"
    value = json.loads(p.read_text()).get("lineage")
    assert isinstance(value, dict) and value, f"{n}: missing/null lineage"
    return value


def test_strict_all_lineage_fields_complete():
    from ecgcert import lineage
    failures = []
    for n in CITED_JSON:
        try:
            lineage.validate_strict_lineage(_lineage(n), require_checkpoint=n in MODEL_JSON)
        except (AssertionError, ValueError) as exc:
            failures.append(f"{n}: {exc}")
    assert not failures, "strict lineage failures:\n" + "\n".join(failures)


def test_strict_all_git_dirty_false():
    dirty = [n for n in CITED_JSON if _lineage(n).get("git_dirty")]
    assert not dirty, f"cited artifacts computed on a DIRTY tree; clean-tree rerun required: {dirty}"


def test_strict_model_checkpoint_sha_nonnull():
    missing = [n for n in MODEL_JSON
               if (ROOT / "results" / f"{n}.json").exists() and not _lineage(n).get("checkpoint_sha256")]
    assert not missing, f"model-result JSONs missing checkpoint_sha256: {missing}"


def test_strict_script_hash_matches_current():
    from ecgcert import lineage
    stale = []
    for n in CITED_JSON:
        lin = _lineage(n)
        script, sha = lin.get("experiment_script"), lin.get("experiment_script_sha256")
        if not (script and sha):
            continue
        for d in _SCRIPT_DIRS:
            sp = ROOT / d / script
            if sp.exists():
                if lineage.file_sha256(str(sp)) != sha:
                    stale.append(n)
                break
    assert not stale, f"JSON experiment_script_sha256 != current script hash; rerun: {stale}"
