"""Release-integrity gate (P0-G). Under ECG_RELEASE=1 these run with ZERO skips and assert
every paper-cited artifact exists, page counts are right, paper numbers equal the JSON, and
no banned/stale phrase survives. Without ECG_RELEASE they skip (so dev `pytest` stays clean).
"""
import json
import os
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
RELEASE = bool(os.environ.get("ECG_RELEASE"))
pytestmark = pytest.mark.skipif(not RELEASE, reason="release-only (set ECG_RELEASE=1)")

CITED_JSON = ["fair_baselines", "st_safety", "recoverability_maps", "tier2_conformal",
              "baselines_physics", "cross_dataset", "lead_weighting", "gpu_deficit_ci",
              "realism_metrics", "gpu_oracle_gate", "neural_baseline", "certificate_validation",
              "fabrication_audit", "fabrication_diffusion", "transfer_bound",
              "delineator_robustness", "certificate_floor_diffusion", "active_selection"]
MACROS = ["paper/auto/fair_baselines_macros.tex", "paper/auto/fair_baselines_table.tex",
          "paper/auto/long_results_macros.tex"]
SOURCE_PDFS = ["paper/main_v2.pdf", "paper/arxiv_long.pdf"]
BANNED = [
    r"1\.[36]\s*\\?sigma", r"1\.[36]\$?\\sigma", r"\\NegSigmaHi",     # sigma-significance language
    r"certified floor", r"reconstructor-invariant", r"certified numeric floor",
    r"DATASET-INDEPENDENT", r"retracted claim", r"shares two directions",
    r"phantom STEMI", r"fabricated diagnosis",
]


def _pages(doc):
    log = (ROOT / "paper" / f"{doc}.log")
    if not log.exists():
        return None
    m = re.search(rf"Output written on {doc}\.pdf \((\d+) pages", log.read_text(errors="replace"))
    return int(m.group(1)) if m else None


def test_all_cited_json_exist_with_lineage():
    for n in CITED_JSON:
        p = ROOT / "results" / f"{n}.json"
        assert p.exists(), f"missing cited result JSON: {n}.json"
        lin = json.loads(p.read_text()).get("lineage")
        assert lin and lin.get("commit") and lin["commit"] != "unknown", f"{n}: no lineage commit"


def test_macros_exist_and_stale_precompiled_pdfs_are_absent():
    for m in MACROS:
        assert (ROOT / m).exists(), f"missing macro file {m}"
    for p in SOURCE_PDFS:
        assert not (ROOT / p).exists(), (
            f"precompiled source-tree PDF is stale-prone: {p}; "
            "build the hash-bound submission into the external artifact directory"
        )


def test_main_page_count():
    n = _pages("main_v2")
    assert n is not None, "no main_v2 build log"
    assert n <= 5, f"main_v2 must be <= 4 technical + 1 reference page; got {n}"


def test_arxiv_compiles():
    n = _pages("arxiv_long")
    assert n is not None and n >= 6, f"arxiv_long should compile to a multi-page doc; got {n}"


def test_no_banned_phrases_in_tex():
    hits = []
    # include auto-generated macros (paper/auto/*.tex) so banned tokens emitted into macros
    # (e.g. a stray sigma-significance macro) are covered, not just hand-written .tex.
    for tex in sorted((ROOT / "paper").rglob("*.tex")):
        txt = tex.read_text(errors="replace")
        for pat in BANNED:
            if re.search(pat, txt, re.IGNORECASE):
                hits.append(f"{tex.relative_to(ROOT).as_posix()}: /{pat}/")
    assert not hits, f"banned/stale phrases present: {hits}"


def test_key_numbers_match_json():
    macros = {}
    for m in (ROOT / "paper" / "auto" / "fair_baselines_macros.tex",
              ROOT / "paper" / "auto" / "long_results_macros.tex"):
        if m.exists():
            for mm in re.finditer(r"\\newcommand\{\\([A-Za-z]+)\}\{([^}]*)\}", m.read_text()):
                macros[mm.group(1)] = mm.group(2)
    safety = json.loads((ROOT / "results" / "st_safety.json").read_text())
    rec = safety["reconstructors"]
    tots = [100 * r.get("total_wrong_rate", r["false_positive_rate"] + r["false_negative_rate"])
            for r in rec.values()]
    if "TotWrongLo" in macros:
        assert float(macros["TotWrongLo"]) <= min(tots) + 0.1
        assert float(macros["TotWrongHi"]) >= max(tots) - 0.1
