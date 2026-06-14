# Cascadia AI

Cascadia AI is being rebuilt as a local-first research and gameplay platform for
the Cascadia board game. The active objective is defined in
[`CASCADIA_V2_GOAL.txt`](CASCADIA_V2_GOAL.txt).

## Target

The primary strength metric is a mean base score of at least **100.0** in
four-player symmetric `AAAAA` games, excluding habitat bonuses. Final claims
require at least 1,000 held-out games (4,000 seat scores).

## Current Status

The independent v2 rules, scoring, replay, simulation, benchmark, dataset,
MLX training, promotion, batched inference, canonical API, and responsive web
product are operational. The existing crates remain the v1 reference
implementation under [`legacy/`](legacy/README.md); v2 does not depend on
their internals.

- [Current status](docs/v2/STATUS.md)
- [Local setup](docs/v2/SETUP.md)
- [V1 audit](docs/v2/V1_AUDIT.md)
- [V2 architecture](docs/v2/ARCHITECTURE.md)
- [Generated CLI reference](docs/v2/CLI_REFERENCE.md)
- [Benchmark protocol](docs/v2/BENCHMARK_PROTOCOL.md)
- [Rules and scoring](docs/v2/RULES_AND_SCORING.md)
- [Dataset format](docs/v2/DATA_FORMAT.md)
- [Local MLX training](docs/v2/TRAINING.md)
- [Model format and inference](docs/v2/MODEL_FORMAT.md)
- [Local web product](docs/v2/WEB.md)
- [Fresh baselines](docs/v2/BASELINES.md)
- [Measured performance](docs/v2/PERFORMANCE.md)
- [Performance qualification](docs/v2/reports/v2-performance-qualification.md)
- [Search diagnostics](docs/v2/SEARCH_DIAGNOSTICS.md)
- [Current score gap](docs/v2/SCORE_GAP.md)
- [Experiment methodology](experiments/README.md)
- [Troubleshooting](docs/v2/TROUBLESHOOTING.md)
- [Roadmap](docs/v2/ROADMAP.md)
- [Decision records](docs/v2/decisions/)
- [Experiment registry](experiments/registry.toml)

## V2 Quick Start

```bash
make bootstrap
make setup
make mlx-device
make check
make benchmark-smoke
make data-smoke
make train-smoke
make ranking-train-smoke
make terminal-ranking-smoke
make web-test
```

Run a canonical 50-game benchmark:

```bash
cargo run --release -p cascadia-cli-v2 -- benchmark \
  --games 50 --strategy pattern-aware
```

Run the promoted interactive policy:

```bash
make benchmark
```

`pattern-aware-v1-k8-h6-b8-m4` is a non-neural, public-information policy that
uses exact score marginals and runs locally at interactive latency. K8
determinized search remains available through `make lookahead-benchmark` for
research comparisons.

Run the unrestricted research policy:

```bash
make benchmark-research
```

`late-conservative-base-policy-improvement-v1-t5-r8-k8-h6-b8-m4-c90` keeps
interactive play before the final five personal turns, then applies eight
shared terminal samples and a fixed one-sided 90% paired confidence gate. With
canonical redetermination it scored 92.100, +0.520 over pattern-aware with 95%
CI `[+0.260,+0.780]`, but failed its balanced-allocation gate by moving 0.375
points away from non-Bear wildlife. It remains an explicit research control,
not a promoted product tier.

Collect resumable training data and train on the Apple GPU:

```bash
cargo run --release -p cascadia-cli-v2 -- collect \
  --output artifacts/datasets/greedy-train \
  --games 1000 --split train --strategy greedy

cargo run --release -p cascadia-cli-v2 -- collect \
  --output artifacts/datasets/greedy-validation \
  --games 200 --split validation --strategy greedy

uv run cascadia-mlx-train \
  --train-dataset artifacts/datasets/greedy-train \
  --validation-dataset artifacts/datasets/greedy-validation \
  --run-dir artifacts/runs/entity-value-v1
```

Collect grouped search labels and train the MLX action ranker:

```bash
make collect-ranking
make train-ranking
make promote-ranking
make evaluate-ranking
```

Run one complete search-guided policy-iteration cycle from a frozen apprentice:

```bash
make collect-ranking-iteration
make train-ranking-iteration
make promote-ranking-iteration
make evaluate-ranking-iteration
```

The iteration collector follows the MLX apprentice for all four seats while
the frozen H6 search teacher labels every candidate at each visited state.
Manifests bind the data to the exact apprentice `model.json` checksum.

Distill the qualified full-game R8 policy-improvement teacher:

```bash
make collect-terminal-ranking
make train-terminal-ranking
make promote-terminal-ranking
make evaluate-terminal-ranking
```

The teacher is a local research oracle, not a product policy. Its terminal
labels preserve the full public market and all opponent boards, and the MLX
ranker serves the same K8+H6+B8 afterstate frontier at interactive latency.

Reproduce the rejected first full-legal apprentice from the independently
qualified 96.35 teacher:

```bash
make collect-imitation
make train-imitation
make collect-imitation-test
make evaluate-imitation-test
```

Only a passing untouched-test report permits `make promote-imitation` and
`make evaluate-imitation`. The apprentice scores every canonical legal action
through a shared-state MLX request; it is not restricted to the sampled
training frontier. The frozen v1 run passed top-one but failed top-five and
MRR, so it was not promoted and its test split is sealed.

The registered successor collects the teacher's complete scored frontier
instead of a one-hot winner and trains uncertainty-aware action preferences:

```bash
make imitation-evidence-parity
make collect-imitation-evidence
make train-imitation-distribution
make resume-imitation-distribution
```

This experiment uses fresh train and validation split domains. It cannot open
a test or gameplay domain unless all preregistered validation gates pass.

The first distributional run was rejected before test access. Its registered
point-scale successor preserves exact immediate score and learns only
continuation residual:

```bash
make collect-imitation-score-residual-validation
make train-imitation-score-residual
make resume-imitation-score-residual
```

The lossless successor ports the qualified teacher's original sparse NNUE
parameters directly into MLX:

```bash
make legacy-nnue-mlx-port
```

ADR 0055 passed synthetic and 80-state Rust parity with 0.00004197 maximum
real error and 40,569 batch-32 evaluations per second on the Apple GPU. This
authorizes a separately preregistered batched search integration; it is not
yet a gameplay strategy or promotion.

Verify the long-lived Rust/MLX sparse service:

```bash
make legacy-nnue-mlx-service
```

ADR 0056 passed with bit-identical service-versus-direct output and 7,589
end-to-end batch-32 evaluations per second. The target consumes immutable ADR
0055 evidence and does not regenerate it.

Verify the exact packed operation and complete search:

```bash
make legacy-nnue-mlx-exact-service
make legacy-nnue-mlx-rollout-parity
```

ADR 0058's packed CSR operation is bit-identical to Rust and sustains 75,176
batch-32 evaluations per second. ADR 0059 reproduced every R32 and R600 search
result exactly at 1.073x native runtime.

Reproduce the fresh gameplay qualification:

```bash
make legacy-nnue-mlx-gameplay-smoke
make legacy-nnue-mlx-gameplay-confirm
```

The ten-game exact MLX baseline scored 95.800 versus the then-promoted
terminal-search control at 92.275,
with a +3.525 paired gain and 95% CI `[+2.388,+4.662]`. It is a research
control, not the final V2 model.

Reproduce the ADR 0065 rollout-return experiment:

```bash
make rollout-value-smoke
make collect-rollout-value
make train-rollout-value
```

Collection and training are local, checksummed, resumable, and MLX-native.
The frozen run improved held-out return prediction sharply but regressed root
action ordering, so it was rejected before gameplay and its fresh gameplay
seeds remain unopened.

The web product exposes three local tiers: near-instant exact greedy,
interactive pattern-aware, and unrestricted terminal-search research. All use
the canonical Rust rules and public-information boundary.

Run the interactive local product:

```bash
make web-dev
```

Open `http://127.0.0.1:5187`.

## V1 Reference

The older crates remain available under `legacy/crates` for reproduction:

```bash
cargo test --workspace
cargo run --release --bin cascadia-cli -- 100
cargo run --release --bin cascadia-web
```

No historical benchmark result is considered valid until it is independently
reproduced under a documented protocol.
