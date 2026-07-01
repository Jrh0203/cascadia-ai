# R2-MAP W5 benchmark and dashboard readmission v1 preregistration

Date: 2026-06-18

Status: frozen validation protocol; no benchmark, protected-domain access,
strength claim, controller transition, or W7 launch is authorized

## Question

Does the current sequential-public-market v1.1 implementation provide a
complete, order-independent focal benchmark and an honest dashboard projection
without weakening identity, score, Pinecone, storage, or phase barriers?

## Source boundary

The accepted packet must bind one immutable John2 source transaction created
after the W1/W4 legality repair. It must include every transitive Rust, Python,
and web source, fixture, manifest, and lockfile exercised below. W0 v1.1 must
name the same game, R2, model, search, runner, evaluator, CLI, and protocol
source identities. A legacy hash, local-only result, or component freeze from
an earlier source revision cannot substitute for this boundary.

## Permitted data

Validation uses deterministic synthetic records and the already-open W0
100-game seed domain only. It does not execute those 100 games unless a
separately registered open checkpoint is available. It must not derive,
accept, print, persist, or open the strength-blinded 20-pair, fixed-development
250-pair, or final 1,000-game seed domains. No result from this packet supports
a strength claim.

## Required behavior

The packet must prove all of the following:

1. Focal aggregation uses only the focal seat and Card A base score with no
   habitat bonuses.
2. Mean, P10, P50, and P90 are emitted for base score; all animals in aggregate
   and Bear, Elk, Salmon, Hawk, and Fox separately; all habitats in aggregate
   and Mountain, Forest, Prairie, Wetland, and River separately; and Pinecones
   earned, independent-draft spend, paid-wipe spend, total spend, remaining,
   and free replacements.
3. Every paid wipe and free replacement is derived from the exact sequential
   public market trace and Pinecone conservation holds.
4. Forward and reverse input order produce byte-identical aggregates.
5. Missing, duplicate, extra, tampered, wrong-host, lease-drifted, and
   identity-drifted records fail closed before publication.
6. The fixed 100-game longitudinal layout assigns even indices to John2, odd
   indices to John3, rotates focal seat by index modulo four, resumes without
   replaying completed games, and rejects unregistered artifacts.
7. The benchmark aggregate and deterministic ledger feed have exact schemas;
   the controller import rejects stale state, nonterminal tasks, duplicate
   binding, path escape, or receipt drift.
8. Python publisher, Rust API reader, and React panel agree on the complete
   telemetry schema. Invalid, oversized, stale, or future-dated projections
   are visible as failures, never silently fresh.
9. The production dashboard remains a single-writer John2 canonical status
   plus one bounded read-only John1 projection. W5 validation must not create a
   second publisher or mutate canonical campaign phase.

## Frozen validation commands

Run from the immutable John2 source transaction with all build, cache, temp,
stdout, and stderr paths under the canonical John2 root:

```text
cargo test -p cascadia-eval --lib
cargo test -p cascadia-api cluster_r2_map --lib
cargo test -p cascadia-cli-v2 r2_map_commands
cargo fmt --all -- --check
cargo clippy -p cascadia-eval -p cascadia-api -p cascadia-cli-v2 --no-deps -- -D warnings
python -m pytest -q -p no:cacheprovider \
  python/tests/test_r2_map_dashboard_status.py \
  python/tests/test_r2_map_campaign_controller.py
python -m ruff check --no-cache \
  python/cascadia_mlx/r2_map_dashboard_status.py \
  python/cascadia_mlx/r2_map_campaign_controller.py \
  python/tests/test_r2_map_dashboard_status.py \
  python/tests/test_r2_map_campaign_controller.py
npm ci --ignore-scripts
npm test -- src/R2MapCampaignPanel.test.tsx src/cluster.test.ts
npm run build
npm run lint
```

The web commands run with `apps/web` as their working directory. The validator
must remove its extracted workspace, pytest base temp, `node_modules`, Cargo
target, and all run-scoped build/cache/temp trees after retaining exact output
hashes and receipts on John2.

## Acceptance evidence

W5 is green only when one immutable report binds:

- source transaction manifest, source tar, and every file identity;
- exact command arrays, exit codes, stdout/stderr hashes, and test counts;
- source/run/read/cleanup/publication receipt locators and hashes;
- schema and distribution assertions above;
- zero swap growth and RSS at most 4 GiB for any gameplay-bearing smoke;
- John2-only durable paths, no SSD access, no John4, and no protected seeds;
- live local and Tailscale dashboard samples across at least two publisher
  intervals, still at revision 0 `contracts-ready`; and
- independent John2 and John3 verification.

Dashboard freshness, the migration Gate 2 pass, or an earlier UI component test
is supporting evidence only. None is sufficient by itself.

## Failure rule

Any missing locator, source drift, schema disagreement, nonzero command,
cleanup residue, projection inconsistency, or protected-domain contact keeps W5
red. Do not repair the failure by weakening the command set or substituting a
local-only result.
