# Hierarchical Query Packing Diagnostic V1

Date: 2026-06-16

Status: **Closed before treatment**

## Question

Could vectorizing the Python-side tile query packing materially accelerate the
32.53-second full-cache epoch observed during ADR 0118 without changing model
semantics?

## Measurement

The unchanged `_query_batches` path packed all 9,808 tile queries into 307
batches of 32. Each pass materialized 1,623,838,144 bytes. Five warm-cache
passes on john1 took:

```text
1.110594, 1.007266, 1.037080, 1.003127, 0.998854 seconds
```

Mean packing time was 1.031384 seconds and median was 1.007266 seconds.
ADR 0118's first 29 complete epoch intervals averaged 32.530618 seconds.

## Amdahl Gate

- observed epoch share: **3.1705%**
- infinite packing-speedup ceiling: **1.032743x**

Even eliminating the entire packing cost cannot reach a 5% epoch improvement.
No vectorization treatment, remote replication, or production code change is
authorized. The active training bottleneck remains MLX model execution and
optimization, not Python query assembly.

## Reproduction

```bash
PYTHONPATH=python .venv/bin/python - <<'PY'
import statistics
import time
from pathlib import Path

from cascadia_mlx.full_legal_hierarchical_factor_retrieval import (
    HierarchicalFactorCache,
    _query_batches,
)

cache = HierarchicalFactorCache(
    Path(
        "artifacts/experiments/"
        "full-legal-hierarchical-factor-retrieval-pilot-v1/cache/train"
    )
)
times = []
for repeat in range(5):
    started = time.perf_counter()
    for shard_index, arrays in enumerate(cache.iter_shards()):
        for _values in _query_batches(
            arrays,
            stage="tile",
            batch_size=32,
            shuffle=True,
            seed=2026061648 + (repeat + 1) * 1000 + shard_index,
        ):
            pass
    times.append(time.perf_counter() - started)
print(times, statistics.mean(times), statistics.median(times))
PY
```
