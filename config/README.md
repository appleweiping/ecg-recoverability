# Audited external integration inputs

The release DAG consumes `ecgrecover.integration.v3.json`. The descriptor is bound to official
GitLab commit `ed49dddf8e5e599b8af702e871a1f66b1d628518`, its root tree object, every imported
upstream source file, and the project-owned bridge by SHA-256. `ecgrecover.upstream.v1.json`
records the independently verified ref and licensing status.

The upstream repository has no license file and GitLab reports no detected license, so its SPDX
status is `NOASSERTION`; this repository neither vendors nor redistributes it. The project owner
reports author permission, whose author-controlled evidence must be reviewed at ARC Stage 9 before
redistribution. The integration also discloses the fixed folds-1-7 scaling required to remove the
published preprocessing's missing-target amplitude leakage. Preparation rejects missing, unpinned,
unhashed, truth-leaking, or undisclosed integrations. The same descriptor freezes a 128-record
process batch and 64-record device micro-batch; these values are carried into generated configs and
result metadata so throughput settings remain auditable.
