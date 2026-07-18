# AutoResearchClaw (ARC) audit — status

**Honest summary: the real ARC pipeline could NOT be executed in this environment, so no
ARC-produced review, verification_report.json, or citation-verification exists. The adversarial
reviews that WERE run are ad-hoc Claude Code agents applying ARC's *methodology*, clearly labelled
below as NOT ARC output.**

## What was attempted (real, reproducible)

- Cloned `aiming-lab/AutoResearchClaw` into a separate audit workspace (NOT vendored into this repo).
  - Pinned commit: `e2e23c93b4943fd21cc531deb09850d8fda55357`  (see `arc_version.txt`)
  - Version: `researchclaw 0.5.0`
- Installed it; CLI works (`researchclaw --help` lists the 23-stage pipeline).
- Configured `config.arc.yaml` with `llm.provider: acp` (the local-agent backend the plan intended)
  and `experiment.mode: simulated` (see `config.arc.yaml`).
- Ran `researchclaw doctor` → **PASS** (`arc_doctor.txt`): config valid, and it even resolves an
  ACP agent at `D:\devtools\claude.CMD`.
- Ran a bounded real pipeline probe: `researchclaw run --mode express --skip-preflight
  --auto-approve --to-stage TOPIC_INIT`.

## What actually happened (the decisive result)

The pipeline **FAILED at Stage 01 (TOPIC_INIT)** on its first LLM call (`pipeline_summary.json`):

```
researchclaw/llm/acp_client.py, _send_prompt
    raise RuntimeError("acpx not found")
RuntimeError: acpx not found
[Stage 01/23 TOPIC_INIT FAILED (0.5s) — acpx not found]
Pipeline complete: 0/1 stages done, 1 failed
```

Root cause: ARC's ACP backend shells out to an **`acpx`** protocol bridge that is **not installed**
here (the `claude`/`opencode`/`codex` names on PATH are shell *functions*, not the executable ARC
needs; `doctor` finds `claude.CMD` but the runtime bridge `acpx` is absent). The alternative
`openai-compatible` backend needs an `OPENAI_API_KEY` that is deliberately not provided. Therefore
**ARC's 23-stage pipeline cannot run in this environment**, and no genuine ARC review/verification/
citation artefacts were produced.

## Consequence for this submission

Per the explicit instruction *"do not claim ARC was used if the real pipeline failed and only ad-hoc
agents ran"*: **we make no claim that ARC reviewed this manuscript.** The substantive adversarial
review that DID happen (and drove concrete fixes) was performed by **Claude Code sub-agents and a
multi-agent workflow** — applying ARC's methodology (independent skeptical peer review, adversarial
verification, claim-vs-evidence checking) but **not ARC itself**. Those findings and the commits that
fixed them are recorded in `reviews.md` and clearly labelled as non-ARC.

## To actually run ARC later (for the record)

Install the `acpx` ACP bridge (or set `llm.provider: openai-compatible` with a real
`OPENAI_API_KEY`), then, on a machine that can reach the LLM backend:
`researchclaw run --from-stage PEER_REVIEW --to-stage CITATION_VERIFY` against the drafted paper to
obtain ARC's own `PEER_REVIEW` / `CITATION_VERIFY` stage artefacts.
