# R0 Spatial-Footprint Screen V1 Invalid Launch 1

Status: invalid before representation timing.

Date: 2026-06-16

The first R0 production graph used immutable release binaries but an incomplete
source bundle. The collector derives its scientific source identity through
`cascadia-provenance`, whose authority includes `CASCADIA_V2_GOAL.txt`,
workspace and Python lock/configuration files, MLX and web sources, V2 crates,
and the declared legacy source roots. Those files were not all present in the
first bundle, so collection inherited whichever partial checkout happened to
exist on each worker.

The failure was detected before any representation timing or promotion:

- john1 train and validation parts reported source BLAKE3
  `5ccc54de0929af5db6d2fe4a369bbe93c92d6a2fffb1443ff7a0809e66cc60a7`;
- john3 train and validation parts reported source BLAKE3
  `c7e40ec0ccbbbd63c5353056bdb121d776434bfe43e34357e17c228163b2aebb`;
- john4 had no complete repository root and both collection commands failed
  before producing a dataset; and
- john2 had not started because the authorized MLX dropout origin still owned
  that host.

All datasets and downstream tasks from that graph are quarantined from
scientific use. The 24 incomplete tasks were administratively cancelled in one
audited queue transition. Completed tasks and the two failed john4 attempts
remain in the queue as historical evidence.

The corrected campaign freezes 311 source files and both release binaries in
bundle
`c4e99c53462e9884c0d9bbbb2220fb70429ae71f6486c7769441e85f1a5750d9`.
Every collector runs through `/usr/bin/env -C` from that bundle's `source/`
tree and invokes the bundled executable by absolute path. Whole-tree fanout
verified the bundle on john1, john2, john3, and john4 before collection.

The first six corrected partitions all report source BLAKE3
`78ec63415e342b4820b89ee5bc7acea32db39af652bf48ba43009e6e7489ae6b`.
They contain 45,040 rows. The two john2 partitions remain queued behind the
dropout origin and will bring the frozen corpus to exactly 50,000 train and
10,000 validation rows.

No result from the invalid launch may contribute to row counts, semantic
coverage, timing, model selection, or gameplay evidence.
