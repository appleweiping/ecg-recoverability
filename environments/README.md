# Locked execution environments

The claim-bearing pipeline uses Python 3.11. CPU jobs are limited to 8--10
workers, while BLAS/OpenMP threads are forced to one per worker. The server runs
one training node at a time through the experiment DAG.

`cpu.lock.txt` and `gpu.lock.txt` are complete transitive, hash-checked locks compiled
from `cpu.in` and `gpu.in` for Python 3.11.2. The GPU lock is a self-contained Linux
x86-64/CUDA 12.8 environment, not a partial overlay on the CPU lock. Their current
SHA-256 values are `bc5534f459af61759abe6e3c640553d266d4d58f73d2cac404990584d7704ed9`
and `fbe43187cea8667241409d33e0378f4cf937ffb4804a2f2182acd58d1d0efd2e`.
Each run envelope records the selected lock hash, actual Python environment hash, NVIDIA
driver, GPU UUID and memory, CPU/cgroup limits, commands, seeds, inputs,
checkpoints, and upstream commits. A release rejects a missing or null field.

Install exactly one lock into a fresh Python 3.11.2 environment:

```bash
uv venv --python 3.11.2 .venv
uv pip install --python .venv/bin/python --require-hashes -r environments/gpu.lock.txt
uv pip install --python .venv/bin/python -e . --no-deps
```

CUDA kernels may remain nondeterministic across hardware or driver revisions.
Tests therefore use declared numerical tolerances and five independent neural
seeds; no GPU bitwise-reproducibility claim is made.
