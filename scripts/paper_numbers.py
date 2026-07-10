"""Emit the paper's result macros from results/*.json (reproducibility link).

Run after experiments/run_all.py; paste the printed \\newcommand lines into the
top of paper/main.tex (or diff against them). This closes the loop so no number in
the paper is hand-typed.
"""
from __future__ import annotations

import json
from pathlib import Path

RES = Path(__file__).resolve().parent.parent / "results"


def _load(name):
    p = RES / name
    return json.loads(p.read_text()) if p.exists() else None


def main():
    lines = []
    pc = _load("precheck_dipolarity.json")
    if pc:
        d = pc["dipolar_fraction"]["NORM"]
        lines += [f"\\newcommand{{\\dipQRS}}{{{d['QRS']:.2f}}}",
                  f"\\newcommand{{\\dipST}}{{{d['ST']:.2f}}}",
                  f"\\newcommand{{\\dipP}}{{{d['P']:.2f}}}"]
        k = pc["kappa_QRS"]
        def kap(name):
            return k[name]["kappa"]
        lines += [f"\\newcommand{{\\kapSpan}}{{{kap('3-lead spanning {I,II,V2}'):.1f}}}",
                  f"\\newcommand{{\\kapColl}}{{{kap('3-lead collinear {V1,V2,V3}'):.1f}}}"]
    st = _load("ptbxl_stemi.json")
    if st:
        r = st["reconstructors"]
        lines += [f"\\newcommand{{\\stFab}}{{{r['generative']['fabricated_stemi']}}}",
                  f"\\newcommand{{\\stMask}}{{{r['ols']['masked_stemi']}}}",
                  f"\\newcommand{{\\stDipFab}}{{{r['dipolar']['fabricated_stemi']}}}"]
    cd = _load("cross_device.json")
    if cd:
        lines += [f"\\newcommand{{\\covCS}}{{{cd['coverage_CS_indist']:.2f}}}",
                  f"\\newcommand{{\\covATplain}}{{{cd['coverage_AT_plain']:.2f}}}",
                  f"\\newcommand{{\\covATrecal}}{{{cd['coverage_AT_recalibrated']:.2f}}}"]
    nb = _load("neural_baselines.json")
    if nb:
        sw = nb["sweep"]
        base = next(s for s in sw if s["adv_weight"] == 0.0)
        peak = max(sw, key=lambda s: s["h"])
        lines += [f"\\newcommand{{\\corrMSE}}{{{base['corr']:+.2f}}}",
                  f"\\newcommand{{\\hPeak}}{{{peak['h']:.2f}}}"]
    print("\n".join(lines))


if __name__ == "__main__":
    main()
