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

Install exactly one lock into a fresh Python 3.11.2 environment. On the experiment server,
`PY311` is the separately hash-recorded exact interpreter and `UV` is pinned uv 0.11.29:

```bash
PY311=/root/autodl-fs/ecg-tools/python/conda-cpython-3.11.2/bin/python
UV=/root/autodl-fs/ecg-tools/bootstrap/uv-0.11.29/uv
"$UV" venv --python "$PY311" .venv
"$UV" pip install --python .venv/bin/python \
  --require-hashes --index-strategy unsafe-best-match \
  -r environments/gpu.lock.txt
"$UV" pip install --python .venv/bin/python -e . --no-deps
```

uv's default `first-index` policy cannot install this lock: the CUDA wheel index
contains some ordinary PyPI names but not the exact frozen versions, and a named
PyTorch index has the same behavior for a requirements file. The narrowly scoped
`unsafe-best-match` exception is permitted only with the complete exact `==` pins
above and `--require-hashes`, so a candidate from either index must still match a
committed digest. Do not use this strategy for an unhashed or floating install.

CUDA kernels may remain nondeterministic across hardware or driver revisions.
Tests therefore use declared numerical tolerances and five independent neural
seeds; no GPU bitwise-reproducibility claim is made.
