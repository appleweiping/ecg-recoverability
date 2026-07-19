# Remote security gate

`remote_status.v2.json` is intentionally fail-closed.  The generated project key is stored outside
the repository; this file contains only its public-key hash.  The project owner explicitly declined
password rotation and accepted that residual risk, while the experiment workflow still forbids
password authentication. The project public key is installed and batch-mode key-only login now
succeeds with `StrictHostKeyChecking=yes` against the pinned historical `known_hosts` entry. The
resulting attestation must still be bound by the signed Stage 9 review. Private keys, passwords, and
credential-bearing commands are never written here.

`repository_secret_scan_passed` is not accepted as a stand-alone assertion.  The canonical DAG
runs `scripts/scan_repository_secrets.py` against the clean source checkout and writes a
repository-external `ecgcert-repository-secret-scan-v1` artifact.  Stage 9 verifies that artifact's
commit, git tree, scanner implementation, pattern set, tracked/untracked/ignored inventory, and
zero-finding result, then checks that it agrees with the status flag.  Symlinks and excluded raw
data, upstream, environment, cache, and artifact roots are listed but never followed.  A match
records only a rule name and repository-relative path, never the matched value.

## Stage-review signing key

`reviewer_ed25519.pub` is the repository-pinned Ed25519 key used to authenticate the mandatory
Stage 5, 9, 15, and 20 decisions. The matching private key must remain outside the repository;
`record_stage_review.py` resolves the key path and refuses any private key beneath the repository
root. A passphrase, when used, is read only from `ECGCERT_REVIEW_KEY_PASSPHRASE`, never from a
command-line option.

The reviewer key is a dedicated approval identity and must never be the SSH login key (or any
other service credential). The private half is stored at a repository-external path whose ACL
grants read access only to the local project owner and full access to the local Administrators and
SYSTEM principals; the path itself is not part of the repository contract. The public key has
OpenSSH fingerprint
`SHA256:huw70yw1YGxZe/skXPpdZn5DiZszVdhw+u4zop2WSPU`; the reviewed JSON records the
equivalent raw-key SHA-256 as
`27971f8477ba1d3ab7cfddbadaffa19142f6a4c1347f2f443430c65f2518ac30`.

Create an immutable approval in the controlled gate inbox with a command of this form (substitute
an absolute, repository-external private-key path):

```text
python scripts/record_stage_review.py --gate <decision.v3.json> --output <stage.approval.v3.json> --reviewer <identity> --decision PROCEED --private-key <external-private-key> --public-key security/reviewer_ed25519.pub
```

The approval signs the stage, decision, reviewer, timestamp, pre-review status, exact canonical
gate SHA-256, frozen evidence hashes, signature algorithm, and raw-public-key fingerprint. The wait
node verifies the Ed25519 signature against the static repository input, then embeds the signature,
approval SHA-256, gate SHA-256, and public-key fingerprint in its reviewed decision. Any changed
decision, evidence, timestamp, approval bytes, gate bytes, or public key fails closed. Public-key
rotation is therefore a reviewed source change and invalidates approvals made with the prior key.

## Publishing a control artifact to the remote run

Do not copy ARC bundles or approvals into the server workspace with ad-hoc `scp`, SFTP, or the
legacy `scripts/remote.py put*` helpers. The supported transport is
`scripts/publish_remote_control.py`. It derives the destination from the selected node's single
`late_control_inputs` declaration, verifies that the remote `<run-id>/workspace` contains the same
manifest and an active pending/running node, and connects with an explicit pinned `known_hosts`
file and one explicit key (`look_for_keys=false`, `allow_agent=false`; there is no password option).

For example, after producing a Stage-5 approval outside the run workspace:

```text
python scripts/publish_remote_control.py --local <stage5.approval.v3.json> --node-id stage5_review --run-id <run-id> --expected-commit <40-hex-frozen-commit> --remote-workspace </absolute/run-root/run-id/workspace> --host <host> --port <port> --username <user> --known-hosts <pinned-known-hosts> --key <ssh-private-key> --report <external-publication-report.json>
```

The tool uploads a file or complete directory to a random sibling temporary path, verifies its
SHA-256/size/type on both ends, and commits it with the non-overwriting SFTP rename operation.
Existing targets are never replaced. It then reads the published target back through the pinned
SSH session and requires the same SHA-256. The local report records the manifest, node, run,
server-host-key fingerprint, and content hashes, but never the private-key path or secret bytes.

At execution time the waiting script atomically captures this live inbox path into
`<run-dir>/control-inputs/<node-id>` and consumes only that captured copy. The runner independently
requires the live source to remain unchanged, rejects links/escapes/special files, and binds the
sealed snapshot in the result envelope, effective configuration hash, data/split hashes, resume
validation, and release validation. Thus transport verification and scientific lineage are both
required; neither substitutes for the Ed25519 approval or ARC bundle validation. Submission DAG
waiting CLIs refuse to consume a control unless the runner-owned capture policy is present; the
policy-free fallback exists only for direct unit/development calls.

The reverse handoff uses the symmetric `scripts/pull_remote_run_artifact.py`; do not copy gate
decisions or operator responses back with ad-hoc `scp`. The requested source must be uniquely
contained by the named producer node's declared output. The tool authenticates the same remote
run/manifest, requires `status.json` to record that producer as `succeeded/0`, validates the v3
result envelope, and independently recomputes its run identity, command/seed/config, ordinary and
late-control input hashes, sealed late-input snapshot, data/split hashes, environment/source lock,
checkpoint inventory, upstream-envelope bindings, and every declared producer output. It hashes
the requested source before and after transfer, revalidates the complete envelope and all producer
outputs after transfer, downloads into a local sibling temporary file, and publishes with an atomic
create-if-absent hard link followed by local SHA-256 readback.

Pull a stage decision for local human review/signing:

```text
python scripts/pull_remote_run_artifact.py --remote-artifact artifacts/control/stage5/decision.v3.json --destination <local-stage5-decision.v3.json> --producer-node-id stage5_gate --run-id <run-id> --expected-commit <40-hex-frozen-commit> --remote-workspace </absolute/run-root/run-id/workspace> --host <host> --port <port> --username <user> --known-hosts <pinned-known-hosts> --key <ssh-private-key> --report <external-pull-report.json>
```

After the server has translated the signed review, pull its exact operator response into the local
ARC bridge receipt root (the Stage number and producer must match):

```text
python scripts/pull_remote_run_artifact.py --remote-artifact artifacts/gates/arc-operator-responses/stage-05/operator-response.v2.json --destination <local-receipt-root>/arc-operator-responses/stage-05/operator-response.v2.json --producer-node-id arc_stage5_forward --run-id <run-id> --expected-commit <40-hex-frozen-commit> --remote-workspace </absolute/run-root/run-id/workspace> --host <host> --port <port> --username <user> --known-hosts <pinned-known-hosts> --key <ssh-private-key> --report <external-pull-report.json>
```

Both destination and report must be absent. Reports use the same sibling-temporary,
create-if-absent, SHA-readback discipline. Neither tool exposes a password option, accepts SSH
agent fallback, overwrites a target, or treats console output as evidence.
