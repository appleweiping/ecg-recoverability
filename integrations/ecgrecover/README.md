# ECGrecover integration boundary

This directory contains project-owned input/output glue only. The upstream
ECGrecover source remains in a pristine external checkout at commit
`ed49dddf8e5e599b8af702e871a1f66b1d628518`; its U-Net implementation,
MSE/Pearson loss, and training loop are imported without modification.

The bridge makes one disclosed methodological adaptation. Upstream
`tools/PreProcesing.py` normalizes each target lead using that record's target
minimum and maximum before applying the missing-lead mask. That is unsuitable
for truth-free raw-mV evaluation because the missing target's amplitude would
be available to preprocessing. The bridge instead estimates a fixed 99.5th
absolute-value scale per lead from folds 1--7, uses it for training and
inference, and converts predictions back to mV. It also restricts training and
evaluation to the published lead-I single-input task and restores observed lead
I bit-for-bit at the output boundary.

For evaluation throughput, one bridge invocation accepts up to the frozen
128-record process batch and evaluates it through the unchanged upstream model
in 64-record device micro-batches. The model and checkpoint are loaded once per
process batch. Deterministic filler values remain record-derived, so results do
not depend on batch membership or order; scalar input remains supported for
contract verification.

The official repository contains no `LICENSE`, `COPYING`, or `NOTICE` file and
the GitLab project reports no detected license. Therefore its SPDX value is
`NOASSERTION`: public readability is not treated as permission to redistribute.
The project owner reports separate author permission. The corresponding
permission record is author-controlled and must be reviewed at ARC Stage 9
before any upstream source or weights are redistributed; neither is vendored
here.
