# AutoResearchClaw (ARC) audit — status

**Honest summary: a real, bounded ARC co-pilot probe was executed after `acpx` and its ACP adapters
were installed, but Stage 01 failed before producing an artifact. Therefore no ARC-produced review,
`verification_report.json`, or citation-verification exists.  Earlier adversarial reviews remain
ad-hoc Claude Code agents applying ARC's methodology and are NOT ARC output.**

## What was attempted (real, reproducible)

- Cloned `aiming-lab/AutoResearchClaw` into a separate audit workspace (NOT vendored into this repo).
  - Pinned commit: `e2e23c93b4943fd21cc531deb09850d8fda55357`  (see `arc_version.txt`)
  - Version: `researchclaw 0.5.0`
- Installed it; CLI works (`researchclaw --help` lists the 23-stage pipeline).
- The original probe used `experiment.mode: simulated`. The maintained `config.arc.yaml` has since
  been refactored to the real ECG question, `project.mode: semi-auto`, co-pilot human gates at
  stages 5/9/15/20, and `experiment.mode: ssh_remote`. This configuration change is a control-plane
  preparation only; it does not retroactively make the failed probe or any old artifact real.
- The original `researchclaw doctor` report is preserved in `arc_doctor.txt`. The refactored
  configuration was revalidated against the pinned v0.5.0 code on 2026-07-19: schema **PASS**,
  `experiment_mode: ssh_remote`, and the ACP agent name resolves to
  `D:\devtools\claude.CMD`. Doctor does not exercise the missing `acpx` bridge.
- Installed `acpx` 0.12.0, Claude ACP adapter 0.37.0, and Codex ACP adapter 0.0.44 outside the
  repository. Package integrity is recorded in `acpx_version.txt` and
  `arc_probe_status.v1.json`.
- Ran a fresh bounded probe in real `co-pilot` mode through `TOPIC_INIT`, without `--auto-approve`.
  It created an ACP session and connected the agent, but did not start any remote experiment.

## What actually happened (the decisive result)

The fresh pipeline **FAILED at Stage 01 (TOPIC_INIT)** on its first LLM call:

```
status: failed
decision: retry
duration_sec: 211.24
artifacts_count: 0
error: Queue owner disconnected before prompt completion
```

The ACP session was created and the agent connected, but its model response did not complete before
the bounded run was cancelled. The exact external artifact paths and SHA-256 values are recorded in
`arc_probe_status.v1.json`. The only valid conclusion is that no genuine ARC review, verification,
or citation artifact has been produced.

## Consequence for this submission

Per the explicit instruction *"do not claim ARC was used if the real pipeline failed and only ad-hoc
agents ran"*: **we make no claim that ARC reviewed this manuscript.** The substantive adversarial
review that DID happen (and drove concrete fixes) was performed by **Claude Code sub-agents and a
multi-agent workflow** — applying ARC's methodology (independent skeptical peer review, adversarial
verification, claim-vs-evidence checking) but **not ARC itself**. Those findings and the commits that
fixed them are recorded in `reviews.md` and clearly labelled as non-ARC.

The primary DAG now contains explicit `arc_stage5_control`, `arc_stage9_control`,
`arc_stage15_control`, and `arc_stage20_control` nodes. Each waits at most 24 hours for a bundle
containing a hash-bound official ARC v0.5.0 decision, stage-health record, co-pilot session,
intervention log, and every declared stage output. The current failed Stage 1 probe is not such a
bundle and cannot advance the release. A successful ARC control receipt also does not replace the
separate Ed25519-signed author review.

## To actually run ARC later (for the record)

Resolve the ACP queue-owner disconnect, rerun bounded Stage 1 with the pinned bridge on `PATH`, then
enter the full co-pilot pipeline. Stages 5, 9, 15, and 20 must pause for a named human decision for at most 24 hours; timeout
never implies approval.  Only after the earlier gates and real experiments succeed may ARC run
`PEER_REVIEW` through `CITATION_VERIFY` against the drafted paper.
