"""P1-13: paper-number consistency. Every load-bearing number in the auto-generated LaTeX
macros must equal the value in the source result JSON (no drift, no hand-editing). Skips a
check when its JSON is absent (e.g. a partial local checkout), so CI stays green while still
catching mismatches when the artifacts are present.
"""
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
MACROS = ROOT / "paper" / "auto" / "fair_baselines_macros.tex"


def _macros():
    if not MACROS.exists():
        pytest.skip("macros not generated")
    txt = MACROS.read_text()
    d = {}
    for m in re.finditer(r"\\newcommand\{\\([A-Za-z]+)\}\{([^}]*)\}", txt):
        d[m.group(1)] = m.group(2)
    return d


def _json(name):
    p = ROOT / "results" / name
    if not p.exists():
        pytest.skip(f"{name} absent")
    return json.loads(p.read_text())


def test_safety_fpfn_match_json():
    """FP/FN/total macros equal 100 * the st_safety.json rates."""
    d = _macros()
    s = _json("st_safety.json")
    for name, tag in (("dipolar", "Dipolar"), ("ridge", "Ridge"), ("ols", "Ols")):
        r = s["reconstructors"].get(name)
        if r is None:
            continue
        if f"Fp{tag}" in d:
            assert abs(float(d[f"Fp{tag}"]) - 100 * r["false_positive_rate"]) < 0.1
            assert abs(float(d[f"Fn{tag}"]) - 100 * r["false_negative_rate"]) < 0.1
            assert abs(float(d[f"Err{tag}"]) - r["mean_st_error_mv"]) < 0.001


def test_graded_eta_match_json():
    """Normalized eta macros equal st_safety.json certificate values."""
    d = _macros()
    s = _json("st_safety.json")
    word = {"V1": "Vone", "V2": "Vtwo", "V3": "Vthree", "V4": "Vfour", "V5": "Vfive", "V6": "Vsix"}
    for lead, v in s["certificate_ST_precordial"].items():
        key = f"EtaNorm{word[lead]}"
        if key in d and v.get("eta_normalized") is not None:
            assert abs(float(d[key]) - v["eta_normalized"]) < 0.001


def test_baseline_table_matches_json():
    """Fair-baseline QRS macros equal fair_baselines.json rmse values."""
    d = _macros()
    f = _json("fair_baselines.json")

    def rm(cfg, mth):
        return f["configs"].get(cfg, {}).get(mth, {}).get("QRS", {}).get("rmse_mV")

    span = "{I,II,V1,V3,V5}"
    checks = [("FairSpanUnetQRS", rm(span, "unet")), ("FairSpanRidgeQRS", rm(span, "ridge")),
              ("FairLimbUnetQRS", rm("limb-6", "unet"))]
    for key, val in checks:
        if key in d and val is not None:
            assert abs(float(d[key]) - val) < 0.001, f"{key}={d[key]} != json {val}"


def test_total_wrong_range_consistent():
    """TotWrongLo/Hi bracket every reconstructor's total in st_safety.json."""
    d = _macros()
    s = _json("st_safety.json")
    if "TotWrongLo" not in d:
        pytest.skip("safety macros absent")
    tots = [100 * (r.get("total_wrong_rate", r["false_positive_rate"] + r["false_negative_rate"]))
            for r in s["reconstructors"].values()]
    assert float(d["TotWrongLo"]) <= min(tots) + 0.1
    assert float(d["TotWrongHi"]) >= max(tots) - 0.1
