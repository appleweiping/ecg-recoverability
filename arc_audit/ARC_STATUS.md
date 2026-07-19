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
- The original probe used `experiment.mode: simulated`; an intermediate configuration then tried
  `ssh_remote`.  That backend was rejected because ARC v0.5.0 disables strict host-key checking.
  The maintained `config.arc.yaml` now uses `project.mode: semi-auto`, co-pilot human gates at
  stages 5/9/15/20, and local `experiment.mode: sandbox`.  ARC is the control plane only; the
  server executes the repository's authenticated native DAG through the separately pinned SSH
  runner.  None of these changes retroactively makes the failed probe or any old artifact real.
- The original `researchclaw doctor` report is preserved in `arc_doctor.txt` as historical evidence.
  The current effective sandbox configuration and pinned ACP executable are instead validated by
  the persistent bridge before it launches the official v0.5.0 process.
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

The primary DAG now uses a two-phase control contract at Stages 5, 9, 15, and 20.  A hash-bound
waiting receipt first proves that the single official ARC v0.5.0 process is still blocked at the
native post-artifact gate.  The native DAG then builds the scientific gate and obtains exactly one
Ed25519-signed author decision.  Only a deterministic translation of that signed decision can be
forwarded to ARC; the post-approval formal receipt is published only after ARC records the native
intervention and bridge handoff.  Stage 9 remains blocked while the long experiments run, and
Stage 15 remains blocked while the paper build is prepared, so the next 24-hour review window
cannot start early.  The current failed Stage 1 probe is neither kind of receipt and cannot advance
the release.

## Formal run still required

Launch one fresh, persistent bridge session from the frozen clean commit and enter the full official
co-pilot pipeline.  The bridge owns the ACP queue for the life of the ARC process, so UI or caller
disconnects do not terminate the session.  Stages 5, 9, 15, and 20 must pause for a named human
decision for at most 24 hours; timeout never implies approval.  Only after the earlier gates and
real experiments succeed may ARC run `PEER_REVIEW` through `CITATION_VERIFY` against the drafted
paper.  Until the four formal receipt bundles exist, the submission must continue to describe the
historical Stage-1 probe as failed and the repaired control path as unexecuted.
