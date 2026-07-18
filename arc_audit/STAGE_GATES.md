# ARC stage-gate register

Each gate has two independent prerequisites: a validated official AutoResearchClaw v0.5.0
co-pilot receipt and an Ed25519-signed author decision over the project gate's exact evidence
bytes. Configuration files, failed probes, unsigned JSON, and ad-hoc agent reviews cannot satisfy
the official receipt prerequisite.

Current state: **THE FRESH STAGE 1 PROBE FAILED; NO STAGE HAS PASSED**.

The repository is configured for AutoResearchClaw v0.5.0 in semi-auto/co-pilot mode. The ACP
runtime now has pinned `acpx` and ACP adapters in an external tools directory. A fresh real co-pilot
probe connected its ACP agent but failed with `Queue owner disconnected before prompt completion`;
it produced zero Stage-01 artifacts and is not research evidence. A later ARC probe must complete
successfully before any ARC review claim may be made. The remote
experiment profile is real (`ssh_remote`), but execution additionally requires verified key-based
SSH access; password fallback is prohibited.

| Gate | ARC stage | Human decision | Minimum evidence | Current status |
|---|---:|---|---|---|
| Literature | 5 | approve / revise | primary-source citation ledger; novelty boundary; pending items explicit | NOT RUN |
| Design freeze | 9 | approve / revise | immutable protocol, split roles, models, endpoints, decision rules, security plan | NOT RUN |
| Research decision | 15 | PROCEED / REFINE / PIVOT | real runs, lineage, leakage checks, patient uncertainty, strong baseline, external validation, claim–evidence matrix | NOT RUN |
| Submission quality | 20 | approve / reject | no placeholders, verified citations, clean build/release, disclosure, adversarial review | NOT RUN |

The hard evidence rules are defined in [`docs/research_protocol.md`](../docs/research_protocol.md).
Timeouts never imply approval. Every decision is an Ed25519-signed artifact bound to the exact
gate SHA-256, normalized gate content, evidence hashes, decision, reviewer, timestamp, and the
repository-pinned reviewer-key fingerprint. The matching private key is kept outside the
repository and is cryptographically separate from the SSH login key. A recorded Stage 15
`PROCEED` releases the prespecified positive claims; a recorded `PIVOT` releases only the
transparent negative-result claim path. Unsigned, expired, altered, or wrong-key reviews block the
DAG.
