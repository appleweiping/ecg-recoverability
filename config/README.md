# Audited external integration inputs

The release DAG expects `ecgrecover.integration.v3.json` in this directory.  It is deliberately not
present yet: the pinned upstream checkout is currently a blobless partial clone and the five source
files needed to audit its real training/preprocessing/inference interface were unavailable from the
upstream server.  A guessed command would invalidate the baseline.  Once those exact-commit blobs
are available, create the descriptor with `schema_version: ecgrecover-integration-v3`, bind every
called upstream/bridge file by SHA-256, and review it at Stage 9.  The preparation command rejects
missing, unpinned, unhashed, or truth-leaking integrations.
