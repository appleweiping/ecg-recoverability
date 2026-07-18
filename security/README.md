# Remote security gate

`remote_status.v1.json` is intentionally fail-closed.  The generated project key is stored outside
the repository; this file contains only its public-key hash.  Stage 9 cannot be approved until the
server owner has rotated the exposed password, installed that public key, verified batch-mode
key-only login against a separately pinned `known_hosts` file, and replaced the pending attestation
with the resulting hashes and reviewer identity.  Private keys, passwords, and credential-bearing
commands are never written here.

## Stage-review signing key

`reviewer_ed25519.pub` is the repository-pinned Ed25519 key used to authenticate the mandatory
Stage 5, 9, 15, and 20 decisions. The matching private key must remain outside the repository;
`record_stage_review.py` resolves the key path and refuses any private key beneath the repository
root. A passphrase, when used, is read only from `ECGCERT_REVIEW_KEY_PASSPHRASE`, never from a
command-line option.

The reviewer key is a dedicated approval identity and must never be the SSH login key (or any
other service credential). The initial private half is stored outside the workspace at
`D:\Project\ecg-recoverability-review-secrets\reviewer_ed25519`; its ACL grants read access only to
the local project owner and full access to the local Administrators and SYSTEM principals. The
public key has OpenSSH fingerprint
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
