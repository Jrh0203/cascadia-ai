# Cascadia V2 CLI Reference

Generated from the typed Clap schema. Regenerate with `make cli-docs`.

## `opponent_intent_collect`

The standalone O1 collector creates and validates compact policy-held-out
sequential datasets.

```text
Usage:
  opponent_intent_collect collect --output PATH --split SPLIT
    --first-game-index N --games N --shard-games N --cohort-id ID
    --policy-pool IDS [--required-policy ID] [--resume]
  opponent_intent_collect validate --dataset PATH

SPLIT: train, validation, test, final
IDS: comma-separated random, greedy, pattern-aware, pattern-commitment,
     pattern-competition, pattern-portfolio
```

Collection validates the complete dataset before returning success. Resume is
accepted only when schema, split, range, cohort, source, and executable
provenance match exactly.

## `opponent_intent_policy_corpus_audit`

The standalone O1 corpus auditor validates all five frozen policy cohorts,
measures action and survival support, proves forbidden-field exclusion from
model inputs, and rejects exact model-input overlap.

```text
Usage:
  opponent_intent_policy_corpus_audit \
    --dataset train-part-0=PATH \
    --dataset train-part-1=PATH \
    --dataset validation=PATH \
    --dataset test=PATH \
    --dataset final-stress=PATH \
    --output PATH
```

The audit is deterministic over immutable corpus bytes. Production closeout
runs it on two distinct Macs with the same executable and corpus tree, then
requires exact scientific JSON and BLAKE3 agreement.

## Full-Legal Research Binary

The exact full-legal audit and public-oracle experiments use the separately
feature-gated binary:

```bash
cargo build --release -p cascadia-differential \
  --features legacy-teacher --bin full-legal-audit
```

Run a paired public-oracle shard:

```bash
MCE_LMR=1 MCE_DIVERSE_PREFILTER=1 \
./target/release/full-legal-audit oracle-compare \
  --model-dir artifacts/models/legacy-nnue-v4opp-mlx-v1 \
  --games 4 --first-seed 62000 --worker john1 \
  --output artifacts/experiments/full-legal-public-oracle-v1/john1.json
```

Strictly validate and merge distributed shards:

```bash
./target/release/full-legal-audit oracle-merge \
  --input artifacts/experiments/full-legal-public-oracle-v1/john1.json \
  --input artifacts/experiments/full-legal-public-oracle-v1/john2.json \
  --input artifacts/experiments/full-legal-public-oracle-v1/john3.json \
  --expected-first-seed 62000 --expected-games 12 \
  --output docs/v2/reports/full-legal-public-oracle-v1-pilot12.json \
  --markdown-output docs/v2/reports/full-legal-public-oracle-v1-pilot12.md
```

The merger rejects incomplete coverage, overlapping seeds, dirty service
shutdowns, fallbacks, bootstrapped samples, and executable, model,
configuration, or source-root identity mismatches.

## Top Level

```text
Cascadia AI v2

Usage: cascadia-v2 <COMMAND>

Commands:
  benchmark
          Run the canonical four-player AAAAA base-score benchmark
  compare
          Compare two strategies on the same deterministic game seeds
  pattern-potential-sweep
          Select phase-decayed habitat and Bear structural potential on a frozen grid
  pattern-potential-compare
          Compare production pattern-aware with one registered structural-potential point
  collect
          Collect versioned, checksummed MLX training data from canonical games
  collect-search
          Collect final-score value targets from the confirmed H6 search teacher
  collect-score-to-go
          Collect signed score-to-go targets from the frozen H6 teacher
  collect-counterfactual-value
          Collect repeated public-redetermination terminal returns from H6 states
  collect-counterfactual-advantage
          Collect shared-seed same-decision counterfactual action returns
  validate-dataset
          Verify a dataset manifest, every shard header, size, and checksum
  validate-score-to-go-dataset
          Verify a signed score-to-go dataset and every target identity
  validate-counterfactual-value-dataset
          Verify a counterfactual-value dataset and every retained sample
  validate-counterfactual-advantage-dataset
          Verify a grouped counterfactual-advantage dataset and every raw return
  audit-counterfactual-value-dataset
          Audit counterfactual target stability and projected collection cost
  audit-counterfactual-advantage-dataset
          Audit centered action-advantage stability and projected collection cost
  collect-ranking
          Collect grouped counterfactual action labels from the confirmed search teacher
  collect-terminal-ranking
          Collect terminal R8 action values from the qualified policy-improvement teacher
  collect-conservative-advantage
          Collect paired c90 anchor/challenger targets from promoted strong trajectories
  collect-ranking-iteration
          Collect H6 labels on states visited by a frozen MLX habitat apprentice
  validate-ranking-dataset
          Verify a grouped action-ranking dataset and every shard checksum
  enrich-action-ranking
          Enrich frozen terminal-ranking labels with explicit, replay-verified action deltas
  validate-action-ranking-dataset
          Verify an action-delta ranking dataset and every shard checksum
  validate-conservative-advantage-dataset
          Verify a paired conservative-advantage dataset and every shard checksum
  collect-public-beam-value
          Collect frozen public-redetermination beam values for MLX training
  public-beam-value-probe
          Collect and evaluate the frozen public beam-state value observability probe
  validate-public-beam-value-dataset
          Verify a public beam-state value dataset and every shard checksum
  model-smoke
          Verify the complete Rust-to-MLX batch inference boundary
  model-benchmark
          Benchmark a promoted or in-progress MLX value model
  model-compare
          Compare an MLX value model with a baseline on identical game seeds
  ranking-model-benchmark
          Benchmark an MLX ranker over the confirmed K8+B8 candidate union
  ranking-model-compare
          Compare an MLX ranker against K8 or its search teacher
  habitat-ranking-model-benchmark
          Benchmark an MLX ranker over the matching H6 K+H candidate union
  habitat-ranking-model-compare
          Compare an MLX H6 apprentice with pattern-aware or the frozen H6 teacher
  pattern-ranking-model-compare
          Compare an MLX terminal-label ranker with pattern-aware on identical games
  action-ranking-model-compare
          Compare an explicit action-delta MLX ranker with pattern-aware play
  full-action-imitation-compare
          Compare full-legal MLX imitation against promoted pattern-aware
  public-beam-value-model-smoke
          Smoke-test the public beam-value model through the Rust/MLX boundary
  public-beam-value-model-compare
          Compare the qualified public beam-value policy with promoted strong
  habitat-ranking-model-h2h
          Compare two frozen MLX H6 apprentices on identical games
  ranking-prefilter-compare
          Compare MLX-prefiltered rollout search with immediate-score K8
  ranking-habitat-prefilter-compare
          Compare H6 with an MLX-prefiltered wider habitat candidate frontier
  ranking-habitat-rollout-compare
          Compare H6 greedy rollouts with batched MLX H6-policy rollouts
  ranking-self-rollout-compare
          Compare H6 with an MLX policy only on the acting seat's next rollout turn
  value-leaf-compare
          Compare H6 search with an MLX final-score value model at rollout leaves
  lookahead-benchmark
          Benchmark fair hidden-state lookahead with greedy rollout policies
  lookahead-compare
          Compare fair hidden-state lookahead with a baseline on identical seeds
  lookahead-ablate
          Compare two fair hidden-state lookahead configurations on identical seeds
  lookahead-recall
          Measure top-K candidate recall against a wider search on baseline trajectories
  nature-wipe-compare
          Compare promoted lookahead with fair one-wipe Nature Token planning
  bear-candidate-compare
          Compare promoted lookahead with a Bear-specific candidate union
  habitat-candidate-compare
          Compare promoted lookahead with a habitat-cohesion candidate union
  bear-habitat-candidate-compare
          Compare H6 with a combined habitat- and Bear-aware candidate frontier
  habitat-candidate-ablate
          Compare two habitat-cohesion lookahead configurations on identical seeds
  pattern-blueprint-compare
          Compare H6 with the same root frontier using pattern-aware rollout plies
  perfect-information-oracle-compare
          Measure the K8+H6+B8 frontier with a diagnostic true-hidden-state oracle
  perfect-information-oracle-frontier-compare
          Compare exact-hidden-state base and wildlife-diverse focal frontiers
  perfect-information-focal-beam-compare
          Compare exact one-step W2 with final-turn exact focal beam planning
  perfect-information-focal-frontier-compare
          Compare W2 and W4 wildlife frontiers under the same exact focal beam
  perfect-information-beam-capacity-compare
          Compare width-16 and width-32 exact W2 focal beams
  perfect-information-root-diverse-beam-compare
          Compare W2 with root-only W4 under W2 future focal layers
  perfect-information-portfolio-beam-compare
          Compare scalar and portfolio-preserving exact focal beam retention
  public-focal-beam-compare
          Compare promoted strong with a public redetermined focal-beam teacher
  public-focal-tree-compare
          Compare promoted strong with public open-loop focal tree search
  terminal-policy-improvement-compare
          Compare pattern-aware with full-game one-step policy improvement
  late-terminal-policy-improvement-compare
          Compare pattern-aware with R8 terminal search on only the final personal turns
  late-wildlife-diverse-policy-improvement-compare
          Compare pattern-aware with a wildlife-diverse final-turn R8 frontier
  late-conservative-policy-improvement-compare
          Compare pattern-aware with confidence-gated final-turn R8 improvement
  late-conservative-base-policy-improvement-compare
          Compare pattern-aware with confidence-gated R8 on the original frontier
  late-conservative-wildlife-focused-policy-improvement-compare
          Compare promoted strong with confidence-gated focused-species coverage
  conservative-sample-count-compare
          Compare conservative final-five policies at two supported sample counts
  help
          Print this message or the help of the given subcommand(s)

Options:
  -h, --help     Print help
  -V, --version  Print version
```

## `benchmark`

```text
Run the canonical four-player AAAAA base-score benchmark

Usage: cascadia-v2 benchmark [OPTIONS]

Options:
      --games <GAMES>            [default: 4]
      --first-seed <FIRST_SEED>  [default: 0]
      --strategy <STRATEGY>      [default: random] [possible values: random, greedy, pattern-aware, pattern-commitment, pattern-competition, pattern-portfolio]
      --sequential               
      --output <OUTPUT>          
  -h, --help                     Print help
```

## `compare`

```text
Compare two strategies on the same deterministic game seeds

Usage: cascadia-v2 compare [OPTIONS] --baseline <BASELINE> --treatment <TREATMENT>

Options:
      --games <GAMES>            [default: 20]
      --first-seed <FIRST_SEED>  [default: 0]
      --baseline <BASELINE>      [possible values: random, greedy, pattern-aware, pattern-commitment, pattern-competition, pattern-portfolio]
      --treatment <TREATMENT>    [possible values: random, greedy, pattern-aware, pattern-commitment, pattern-competition, pattern-portfolio]
      --sequential               
      --output <OUTPUT>          
  -h, --help                     Print help
```

## `pattern-potential-sweep`

```text
Select phase-decayed habitat and Bear structural potential on a frozen grid

Usage: cascadia-v2 pattern-potential-sweep [OPTIONS]

Options:
      --games <GAMES>            [default: 32]
      --first-seed <FIRST_SEED>  [default: 31300]
      --output <OUTPUT>          
  -h, --help                     Print help
```

## `pattern-potential-compare`

```text
Compare production pattern-aware with one registered structural-potential point

Usage: cascadia-v2 pattern-potential-compare [OPTIONS] --opportunity-weight <OPPORTUNITY_WEIGHT> --habitat-weight <HABITAT_WEIGHT> --bear-weight <BEAR_WEIGHT>

Options:
      --games <GAMES>                            [default: 50]
      --first-seed <FIRST_SEED>                  [default: 31400]
      --opportunity-weight <OPPORTUNITY_WEIGHT>  
      --habitat-weight <HABITAT_WEIGHT>          
      --bear-weight <BEAR_WEIGHT>                
      --sequential                               
      --output <OUTPUT>                          
  -h, --help                                     Print help
```

## `collect`

```text
Collect versioned, checksummed MLX training data from canonical games

Usage: cascadia-v2 collect [OPTIONS] --output <OUTPUT> --games <GAMES>

Options:
      --output <OUTPUT>
          
      --games <GAMES>
          
      --first-game-index <FIRST_GAME_INDEX>
          [default: 0]
      --split <SPLIT>
          [default: train] [possible values: train, validation, test, final]
      --strategy <STRATEGY>
          [default: greedy] [possible values: random, greedy, pattern-aware, pattern-commitment, pattern-competition, pattern-portfolio]
      --shard-games <SHARD_GAMES>
          [default: 64]
      --resume
          
  -h, --help
          Print help
```

## `collect-search`

```text
Collect final-score value targets from the confirmed H6 search teacher

Usage: cascadia-v2 collect-search [OPTIONS] --output <OUTPUT> --games <GAMES>

Options:
      --output <OUTPUT>
          
      --games <GAMES>
          
      --first-game-index <FIRST_GAME_INDEX>
          [default: 0]
      --split <SPLIT>
          [default: train] [possible values: train, validation, test, final]
      --shard-games <SHARD_GAMES>
          [default: 8]
      --resume
          
      --candidates <CANDIDATES>
          [default: 8]
      --habitat-candidates <HABITAT_CANDIDATES>
          [default: 6]
      --determinizations <DETERMINIZATIONS>
          [default: 4]
      --greedy-plies <GREEDY_PLIES>
          [default: 4]
  -h, --help
          Print help
```

## `collect-score-to-go`

```text
Collect signed score-to-go targets from the frozen H6 teacher

Usage: cascadia-v2 collect-score-to-go [OPTIONS] --output <OUTPUT> --games <GAMES>

Options:
      --output <OUTPUT>
          
      --games <GAMES>
          
      --first-game-index <FIRST_GAME_INDEX>
          [default: 0]
      --split <SPLIT>
          [default: train] [possible values: train, validation, test, final]
      --shard-games <SHARD_GAMES>
          [default: 1]
      --resume
          
  -h, --help
          Print help
```

## `collect-counterfactual-value`

```text
Collect repeated public-redetermination terminal returns from H6 states

Usage: cascadia-v2 collect-counterfactual-value [OPTIONS] --output <OUTPUT> --games <GAMES>

Options:
      --output <OUTPUT>
          
      --games <GAMES>
          
      --first-game-index <FIRST_GAME_INDEX>
          [default: 0]
      --split <SPLIT>
          [default: train] [possible values: train, validation, test, final]
      --samples-per-state <SAMPLES_PER_STATE>
          [default: 16]
      --resume
          
  -h, --help
          Print help
```

## `collect-counterfactual-advantage`

```text
Collect shared-seed same-decision counterfactual action returns

Usage: cascadia-v2 collect-counterfactual-advantage [OPTIONS] --output <OUTPUT> --games <GAMES>

Options:
      --output <OUTPUT>
          
      --games <GAMES>
          
      --first-game-index <FIRST_GAME_INDEX>
          [default: 0]
      --split <SPLIT>
          [default: train] [possible values: train, validation, test, final]
      --groups-per-game <GROUPS_PER_GAME>
          [default: 16]
      --samples-per-candidate <SAMPLES_PER_CANDIDATE>
          [default: 16]
      --candidate-selection <CANDIDATE_SELECTION>
          [default: nearest] [possible values: nearest, stratified]
      --resume
          
  -h, --help
          Print help
```

## `validate-dataset`

```text
Verify a dataset manifest, every shard header, size, and checksum

Usage: cascadia-v2 validate-dataset --dataset <DATASET>

Options:
      --dataset <DATASET>  
  -h, --help               Print help
```

## `validate-score-to-go-dataset`

```text
Verify a signed score-to-go dataset and every target identity

Usage: cascadia-v2 validate-score-to-go-dataset --dataset <DATASET>

Options:
      --dataset <DATASET>  
  -h, --help               Print help
```

## `validate-counterfactual-value-dataset`

```text
Verify a counterfactual-value dataset and every retained sample

Usage: cascadia-v2 validate-counterfactual-value-dataset --dataset <DATASET>

Options:
      --dataset <DATASET>  
  -h, --help               Print help
```

## `validate-counterfactual-advantage-dataset`

```text
Verify a grouped counterfactual-advantage dataset and every raw return

Usage: cascadia-v2 validate-counterfactual-advantage-dataset --dataset <DATASET>

Options:
      --dataset <DATASET>  
  -h, --help               Print help
```

## `audit-counterfactual-value-dataset`

```text
Audit counterfactual target stability and projected collection cost

Usage: cascadia-v2 audit-counterfactual-value-dataset --dataset <DATASET> --output <OUTPUT>

Options:
      --dataset <DATASET>  
      --output <OUTPUT>    
  -h, --help               Print help
```

## `audit-counterfactual-advantage-dataset`

```text
Audit centered action-advantage stability and projected collection cost

Usage: cascadia-v2 audit-counterfactual-advantage-dataset [OPTIONS] --dataset <DATASET> --output <OUTPUT> --markdown-output <MARKDOWN_OUTPUT>

Options:
      --dataset <DATASET>                      
      --output <OUTPUT>                        
      --markdown-output <MARKDOWN_OUTPUT>      
      --estimator-samples <ESTIMATOR_SAMPLES>  [default: 8]
  -h, --help                                   Print help
```

## `collect-ranking`

```text
Collect grouped counterfactual action labels from the confirmed search teacher

Usage: cascadia-v2 collect-ranking [OPTIONS] --output <OUTPUT> --games <GAMES>

Options:
      --output <OUTPUT>
          
      --games <GAMES>
          
      --first-game-index <FIRST_GAME_INDEX>
          [default: 0]
      --split <SPLIT>
          [default: train] [possible values: train, validation, test, final]
      --shard-games <SHARD_GAMES>
          [default: 8]
      --resume
          
      --teacher <TEACHER>
          [default: bear] [possible values: bear, habitat]
      --candidates <CANDIDATES>
          [default: 8]
      --bear-candidates <BEAR_CANDIDATES>
          [default: 8]
      --habitat-candidates <HABITAT_CANDIDATES>
          [default: 6]
      --determinizations <DETERMINIZATIONS>
          [default: 4]
      --greedy-plies <GREEDY_PLIES>
          [default: 4]
  -h, --help
          Print help
```

## `collect-terminal-ranking`

```text
Collect terminal R8 action values from the qualified policy-improvement teacher

Usage: cascadia-v2 collect-terminal-ranking [OPTIONS] --output <OUTPUT> --games <GAMES>

Options:
      --output <OUTPUT>
          
      --games <GAMES>
          
      --first-game-index <FIRST_GAME_INDEX>
          [default: 0]
      --split <SPLIT>
          [default: train] [possible values: train, validation, test, final]
      --shard-games <SHARD_GAMES>
          [default: 1]
      --resume
          
      --determinizations <DETERMINIZATIONS>
          [default: 8]
      --policy-candidates <POLICY_CANDIDATES>
          [default: 8]
      --policy-habitat-candidates <POLICY_HABITAT_CANDIDATES>
          [default: 6]
      --policy-bear-candidates <POLICY_BEAR_CANDIDATES>
          [default: 8]
      --policy-market-draws <POLICY_MARKET_DRAWS>
          [default: 4]
  -h, --help
          Print help
```

## `collect-conservative-advantage`

```text
Collect paired c90 anchor/challenger targets from promoted strong trajectories

Usage: cascadia-v2 collect-conservative-advantage [OPTIONS] --output <OUTPUT> --games <GAMES>

Options:
      --output <OUTPUT>
          
      --games <GAMES>
          
      --first-game-index <FIRST_GAME_INDEX>
          [default: 0]
      --split <SPLIT>
          [default: train] [possible values: train, validation, test, final]
      --shard-games <SHARD_GAMES>
          [default: 1]
      --resume
          
      --terminal-turns <TERMINAL_TURNS>
          [default: 5]
      --policy-candidates <POLICY_CANDIDATES>
          [default: 8]
      --policy-habitat-candidates <POLICY_HABITAT_CANDIDATES>
          [default: 6]
      --policy-bear-candidates <POLICY_BEAR_CANDIDATES>
          [default: 8]
      --policy-market-draws <POLICY_MARKET_DRAWS>
          [default: 4]
  -h, --help
          Print help
```

## `collect-ranking-iteration`

```text
Collect H6 labels on states visited by a frozen MLX habitat apprentice

Usage: cascadia-v2 collect-ranking-iteration [OPTIONS] --output <OUTPUT> --games <GAMES> --model-dir <MODEL_DIR>

Options:
      --output <OUTPUT>
          
      --games <GAMES>
          
      --first-game-index <FIRST_GAME_INDEX>
          [default: 0]
      --split <SPLIT>
          [default: train] [possible values: train, validation, test, final]
      --shard-games <SHARD_GAMES>
          [default: 8]
      --resume
          
      --model-dir <MODEL_DIR>
          
      --server <SERVER>
          [default: .venv/bin/cascadia-mlx-ranking-serve]
      --candidates <CANDIDATES>
          [default: 8]
      --habitat-candidates <HABITAT_CANDIDATES>
          [default: 6]
      --determinizations <DETERMINIZATIONS>
          [default: 4]
      --greedy-plies <GREEDY_PLIES>
          [default: 4]
  -h, --help
          Print help
```

## `validate-ranking-dataset`

```text
Verify a grouped action-ranking dataset and every shard checksum

Usage: cascadia-v2 validate-ranking-dataset --dataset <DATASET>

Options:
      --dataset <DATASET>  
  -h, --help               Print help
```

## `enrich-action-ranking`

```text
Enrich frozen terminal-ranking labels with explicit, replay-verified action deltas

Usage: cascadia-v2 enrich-action-ranking [OPTIONS] --source-dataset <SOURCE_DATASET> --output <OUTPUT>

Options:
      --source-dataset <SOURCE_DATASET>            
      --output <OUTPUT>                            
      --resume                                     
      --policy-market-draws <POLICY_MARKET_DRAWS>  [default: 4]
  -h, --help                                       Print help
```

## `validate-action-ranking-dataset`

```text
Verify an action-delta ranking dataset and every shard checksum

Usage: cascadia-v2 validate-action-ranking-dataset --dataset <DATASET>

Options:
      --dataset <DATASET>  
  -h, --help               Print help
```

## `validate-conservative-advantage-dataset`

```text
Verify a paired conservative-advantage dataset and every shard checksum

Usage: cascadia-v2 validate-conservative-advantage-dataset --dataset <DATASET>

Options:
      --dataset <DATASET>  
  -h, --help               Print help
```

## `collect-public-beam-value`

```text
Collect frozen public-redetermination beam values for MLX training

Usage: cascadia-v2 collect-public-beam-value [OPTIONS] --output <OUTPUT> --games <GAMES>

Options:
      --output <OUTPUT>
          
      --games <GAMES>
          
      --first-game-index <FIRST_GAME_INDEX>
          [default: 0]
      --split <SPLIT>
          [default: train] [possible values: train, validation, test, final]
      --resume
          
  -h, --help
          Print help
```

## `public-beam-value-probe`

```text
Collect and evaluate the frozen public beam-state value observability probe

Usage: cascadia-v2 public-beam-value-probe [OPTIONS] --output <OUTPUT>

Options:
      --output <OUTPUT>                      
      --first-game-index <FIRST_GAME_INDEX>  [default: 40000]
      --games <GAMES>                        [default: 2]
      --resume                               
      --report <REPORT>                      
  -h, --help                                 Print help
```

## `validate-public-beam-value-dataset`

```text
Verify a public beam-state value dataset and every shard checksum

Usage: cascadia-v2 validate-public-beam-value-dataset --dataset <DATASET>

Options:
      --dataset <DATASET>  
  -h, --help               Print help
```

## `model-smoke`

```text
Verify the complete Rust-to-MLX batch inference boundary

Usage: cascadia-v2 model-smoke [OPTIONS]

Options:
      --run-dir <RUN_DIR>      
      --model-dir <MODEL_DIR>  
      --server <SERVER>        [default: .venv/bin/cascadia-mlx-serve]
  -h, --help                   Print help
```

## `model-benchmark`

```text
Benchmark a promoted or in-progress MLX value model

Usage: cascadia-v2 model-benchmark [OPTIONS]

Options:
      --games <GAMES>              [default: 4]
      --first-seed <FIRST_SEED>    [default: 0]
      --run-dir <RUN_DIR>          
      --model-dir <MODEL_DIR>      
      --server <SERVER>            [default: .venv/bin/cascadia-mlx-serve]
      --prefilter-k <PREFILTER_K>  Restrict model ranking to the top K exact immediate-score actions
      --output <OUTPUT>            
  -h, --help                       Print help
```

## `model-compare`

```text
Compare an MLX value model with a baseline on identical game seeds

Usage: cascadia-v2 model-compare [OPTIONS]

Options:
      --games <GAMES>              [default: 20]
      --first-seed <FIRST_SEED>    [default: 0]
      --baseline <BASELINE>        [default: greedy] [possible values: random, greedy, pattern-aware, pattern-commitment, pattern-competition, pattern-portfolio]
      --run-dir <RUN_DIR>          
      --model-dir <MODEL_DIR>      
      --server <SERVER>            [default: .venv/bin/cascadia-mlx-serve]
      --prefilter-k <PREFILTER_K>  Restrict model ranking to the top K exact immediate-score actions
      --output <OUTPUT>            
  -h, --help                       Print help
```

## `ranking-model-benchmark`

```text
Benchmark an MLX ranker over the confirmed K8+B8 candidate union

Usage: cascadia-v2 ranking-model-benchmark [OPTIONS]

Options:
      --games <GAMES>                      [default: 4]
      --first-seed <FIRST_SEED>            [default: 0]
      --run-dir <RUN_DIR>                  
      --model-dir <MODEL_DIR>              
      --server <SERVER>                    [default: .venv/bin/cascadia-mlx-ranking-serve]
      --candidates <CANDIDATES>            [default: 8]
      --bear-candidates <BEAR_CANDIDATES>  [default: 8]
      --output <OUTPUT>                    
  -h, --help                               Print help
```

## `ranking-model-compare`

```text
Compare an MLX ranker against K8 or its search teacher

Usage: cascadia-v2 ranking-model-compare [OPTIONS]

Options:
      --games <GAMES>                      [default: 20]
      --first-seed <FIRST_SEED>            [default: 0]
      --run-dir <RUN_DIR>                  
      --model-dir <MODEL_DIR>              
      --server <SERVER>                    [default: .venv/bin/cascadia-mlx-ranking-serve]
      --baseline <BASELINE>                [default: k8] [possible values: k8, bear-teacher]
      --candidates <CANDIDATES>            [default: 8]
      --bear-candidates <BEAR_CANDIDATES>  [default: 8]
      --output <OUTPUT>                    
  -h, --help                               Print help
```

## `habitat-ranking-model-benchmark`

```text
Benchmark an MLX ranker over the matching H6 K+H candidate union

Usage: cascadia-v2 habitat-ranking-model-benchmark [OPTIONS] --model-dir <MODEL_DIR>

Options:
      --games <GAMES>                            [default: 4]
      --first-seed <FIRST_SEED>                  [default: 0]
      --model-dir <MODEL_DIR>                    
      --server <SERVER>                          [default: .venv/bin/cascadia-mlx-ranking-serve]
      --candidates <CANDIDATES>                  [default: 8]
      --habitat-candidates <HABITAT_CANDIDATES>  [default: 6]
      --output <OUTPUT>                          
  -h, --help                                     Print help
```

## `habitat-ranking-model-compare`

```text
Compare an MLX H6 apprentice with pattern-aware or the frozen H6 teacher

Usage: cascadia-v2 habitat-ranking-model-compare [OPTIONS] --model-dir <MODEL_DIR>

Options:
      --games <GAMES>
          [default: 10]
      --first-seed <FIRST_SEED>
          [default: 0]
      --model-dir <MODEL_DIR>
          
      --server <SERVER>
          [default: .venv/bin/cascadia-mlx-ranking-serve]
      --baseline <BASELINE>
          [default: pattern-aware] [possible values: pattern-aware, habitat-teacher]
      --candidates <CANDIDATES>
          [default: 8]
      --habitat-candidates <HABITAT_CANDIDATES>
          [default: 6]
      --determinizations <DETERMINIZATIONS>
          [default: 4]
      --greedy-plies <GREEDY_PLIES>
          [default: 4]
      --output <OUTPUT>
          
  -h, --help
          Print help
```

## `pattern-ranking-model-compare`

```text
Compare an MLX terminal-label ranker with pattern-aware on identical games

Usage: cascadia-v2 pattern-ranking-model-compare [OPTIONS] --model-dir <MODEL_DIR>

Options:
      --games <GAMES>
          [default: 10]
      --first-seed <FIRST_SEED>
          [default: 0]
      --model-dir <MODEL_DIR>
          
      --server <SERVER>
          [default: .venv/bin/cascadia-mlx-ranking-serve]
      --policy-candidates <POLICY_CANDIDATES>
          [default: 8]
      --policy-habitat-candidates <POLICY_HABITAT_CANDIDATES>
          [default: 6]
      --policy-bear-candidates <POLICY_BEAR_CANDIDATES>
          [default: 8]
      --policy-market-draws <POLICY_MARKET_DRAWS>
          [default: 4]
      --output <OUTPUT>
          
  -h, --help
          Print help
```

## `action-ranking-model-compare`

```text
Compare an explicit action-delta MLX ranker with pattern-aware play

Usage: cascadia-v2 action-ranking-model-compare [OPTIONS]

Options:
      --games <GAMES>
          [default: 10]
      --first-seed <FIRST_SEED>
          [default: 25700]
      --run-dir <RUN_DIR>
          
      --model-dir <MODEL_DIR>
          
      --server <SERVER>
          [default: .venv/bin/cascadia-mlx-action-ranking-serve]
      --policy-candidates <POLICY_CANDIDATES>
          [default: 8]
      --policy-habitat-candidates <POLICY_HABITAT_CANDIDATES>
          [default: 6]
      --policy-bear-candidates <POLICY_BEAR_CANDIDATES>
          [default: 8]
      --policy-market-draws <POLICY_MARKET_DRAWS>
          [default: 4]
      --output <OUTPUT>
          
  -h, --help
          Print help
```

## `full-action-imitation-compare`

```text
Compare full-legal MLX imitation against promoted pattern-aware

Usage: cascadia-v2 full-action-imitation-compare [OPTIONS]

Options:
      --games <GAMES>            [default: 1]
      --first-seed <FIRST_SEED>  [default: 32700]
      --run-dir <RUN_DIR>        
      --model-dir <MODEL_DIR>    
      --server <SERVER>          [default: .venv/bin/cascadia-mlx-imitation-serve]
      --output <OUTPUT>          
  -h, --help                     Print help
```

## `public-beam-value-model-smoke`

```text
Smoke-test the public beam-value model through the Rust/MLX boundary

Usage: cascadia-v2 public-beam-value-model-smoke [OPTIONS]

Options:
      --run-dir <RUN_DIR>      
      --model-dir <MODEL_DIR>  
      --server <SERVER>        [default: .venv/bin/cascadia-mlx-public-beam-value-serve]
  -h, --help                   Print help
```

## `public-beam-value-model-compare`

```text
Compare the qualified public beam-value policy with promoted strong

Usage: cascadia-v2 public-beam-value-model-compare [OPTIONS]

Options:
      --games <GAMES>            [default: 10]
      --first-seed <FIRST_SEED>  [default: 31000]
      --run-dir <RUN_DIR>        
      --model-dir <MODEL_DIR>    
      --server <SERVER>          [default: .venv/bin/cascadia-mlx-public-beam-value-serve]
      --output <OUTPUT>          
  -h, --help                     Print help
```

## `habitat-ranking-model-h2h`

```text
Compare two frozen MLX H6 apprentices on identical games

Usage: cascadia-v2 habitat-ranking-model-h2h [OPTIONS] --baseline-model-dir <BASELINE_MODEL_DIR> --treatment-model-dir <TREATMENT_MODEL_DIR>

Options:
      --games <GAMES>                              [default: 10]
      --first-seed <FIRST_SEED>                    [default: 0]
      --baseline-model-dir <BASELINE_MODEL_DIR>    
      --treatment-model-dir <TREATMENT_MODEL_DIR>  
      --server <SERVER>                            [default: .venv/bin/cascadia-mlx-ranking-serve]
      --candidates <CANDIDATES>                    [default: 8]
      --habitat-candidates <HABITAT_CANDIDATES>    [default: 6]
      --output <OUTPUT>                            
  -h, --help                                       Print help
```

## `ranking-prefilter-compare`

```text
Compare MLX-prefiltered rollout search with immediate-score K8

Usage: cascadia-v2 ranking-prefilter-compare [OPTIONS]

Options:
      --games <GAMES>                                [default: 10]
      --first-seed <FIRST_SEED>                      [default: 0]
      --run-dir <RUN_DIR>                            
      --model-dir <MODEL_DIR>                        
      --server <SERVER>                              [default: .venv/bin/cascadia-mlx-ranking-serve]
      --candidates <CANDIDATES>                      [default: 8]
      --bear-candidates <BEAR_CANDIDATES>            [default: 8]
      --immediate-anchors <IMMEDIATE_ANCHORS>        [default: 0]
      --prefilter-candidates <PREFILTER_CANDIDATES>  [default: 8]
      --determinizations <DETERMINIZATIONS>          [default: 4]
      --greedy-plies <GREEDY_PLIES>                  [default: 4]
      --output <OUTPUT>                              
  -h, --help                                         Print help
```

## `ranking-habitat-prefilter-compare`

```text
Compare H6 with an MLX-prefiltered wider habitat candidate frontier

Usage: cascadia-v2 ranking-habitat-prefilter-compare [OPTIONS]

Options:
      --games <GAMES>
          [default: 10]
      --first-seed <FIRST_SEED>
          [default: 0]
      --run-dir <RUN_DIR>
          
      --model-dir <MODEL_DIR>
          
      --server <SERVER>
          [default: .venv/bin/cascadia-mlx-ranking-serve]
      --baseline-candidates <BASELINE_CANDIDATES>
          [default: 8]
      --baseline-habitat-candidates <BASELINE_HABITAT_CANDIDATES>
          [default: 6]
      --candidates <CANDIDATES>
          [default: 16]
      --habitat-candidates <HABITAT_CANDIDATES>
          [default: 8]
      --immediate-anchors <IMMEDIATE_ANCHORS>
          [default: 8]
      --prefilter-candidates <PREFILTER_CANDIDATES>
          [default: 14]
      --determinizations <DETERMINIZATIONS>
          [default: 4]
      --greedy-plies <GREEDY_PLIES>
          [default: 4]
      --output <OUTPUT>
          
  -h, --help
          Print help
```

## `ranking-habitat-rollout-compare`

```text
Compare H6 greedy rollouts with batched MLX H6-policy rollouts

Usage: cascadia-v2 ranking-habitat-rollout-compare [OPTIONS]

Options:
      --games <GAMES>
          [default: 10]
      --first-seed <FIRST_SEED>
          [default: 0]
      --run-dir <RUN_DIR>
          
      --model-dir <MODEL_DIR>
          
      --server <SERVER>
          [default: .venv/bin/cascadia-mlx-ranking-serve]
      --candidates <CANDIDATES>
          [default: 8]
      --habitat-candidates <HABITAT_CANDIDATES>
          [default: 6]
      --determinizations <DETERMINIZATIONS>
          [default: 4]
      --rollout-plies <ROLLOUT_PLIES>
          [default: 4]
      --rollout-candidates <ROLLOUT_CANDIDATES>
          [default: 8]
      --rollout-habitat-candidates <ROLLOUT_HABITAT_CANDIDATES>
          [default: 6]
      --output <OUTPUT>
          
  -h, --help
          Print help
```

## `ranking-self-rollout-compare`

```text
Compare H6 with an MLX policy only on the acting seat's next rollout turn

Usage: cascadia-v2 ranking-self-rollout-compare [OPTIONS]

Options:
      --games <GAMES>
          [default: 10]
      --first-seed <FIRST_SEED>
          [default: 0]
      --run-dir <RUN_DIR>
          
      --model-dir <MODEL_DIR>
          
      --server <SERVER>
          [default: .venv/bin/cascadia-mlx-ranking-serve]
      --candidates <CANDIDATES>
          [default: 8]
      --habitat-candidates <HABITAT_CANDIDATES>
          [default: 6]
      --determinizations <DETERMINIZATIONS>
          [default: 4]
      --rollout-plies <ROLLOUT_PLIES>
          [default: 4]
      --policy-candidates <POLICY_CANDIDATES>
          [default: 8]
      --policy-habitat-candidates <POLICY_HABITAT_CANDIDATES>
          [default: 6]
      --output <OUTPUT>
          
  -h, --help
          Print help
```

## `value-leaf-compare`

```text
Compare H6 search with an MLX final-score value model at rollout leaves

Usage: cascadia-v2 value-leaf-compare [OPTIONS]

Options:
      --games <GAMES>                            [default: 10]
      --first-seed <FIRST_SEED>                  [default: 0]
      --run-dir <RUN_DIR>                        
      --model-dir <MODEL_DIR>                    
      --server <SERVER>                          [default: .venv/bin/cascadia-mlx-serve]
      --candidates <CANDIDATES>                  [default: 8]
      --habitat-candidates <HABITAT_CANDIDATES>  [default: 6]
      --determinizations <DETERMINIZATIONS>      [default: 4]
      --greedy-plies <GREEDY_PLIES>              [default: 4]
      --output <OUTPUT>                          
  -h, --help                                     Print help
```

## `lookahead-benchmark`

```text
Benchmark fair hidden-state lookahead with greedy rollout policies

Usage: cascadia-v2 lookahead-benchmark [OPTIONS]

Options:
      --games <GAMES>                        [default: 1]
      --first-seed <FIRST_SEED>              [default: 0]
      --candidates <CANDIDATES>              [default: 4]
      --determinizations <DETERMINIZATIONS>  [default: 4]
      --greedy-plies <GREEDY_PLIES>          [default: 4]
      --output <OUTPUT>                      
  -h, --help                                 Print help
```

## `lookahead-compare`

```text
Compare fair hidden-state lookahead with a baseline on identical seeds

Usage: cascadia-v2 lookahead-compare [OPTIONS]

Options:
      --games <GAMES>
          [default: 10]
      --first-seed <FIRST_SEED>
          [default: 0]
      --baseline <BASELINE>
          [default: greedy] [possible values: random, greedy, pattern-aware, pattern-commitment, pattern-competition, pattern-portfolio]
      --candidates <CANDIDATES>
          [default: 4]
      --determinizations <DETERMINIZATIONS>
          [default: 4]
      --greedy-plies <GREEDY_PLIES>
          [default: 4]
      --output <OUTPUT>
          
  -h, --help
          Print help
```

## `lookahead-ablate`

```text
Compare two fair hidden-state lookahead configurations on identical seeds

Usage: cascadia-v2 lookahead-ablate [OPTIONS]

Options:
      --games <GAMES>                                            [default: 10]
      --first-seed <FIRST_SEED>                                  [default: 0]
      --baseline-candidates <BASELINE_CANDIDATES>                [default: 4]
      --baseline-determinizations <BASELINE_DETERMINIZATIONS>    [default: 4]
      --baseline-greedy-plies <BASELINE_GREEDY_PLIES>            [default: 4]
      --treatment-candidates <TREATMENT_CANDIDATES>              [default: 4]
      --treatment-determinizations <TREATMENT_DETERMINIZATIONS>  [default: 4]
      --treatment-greedy-plies <TREATMENT_GREEDY_PLIES>          [default: 4]
      --output <OUTPUT>                                          
  -h, --help                                                     Print help
```

## `lookahead-recall`

```text
Measure top-K candidate recall against a wider search on baseline trajectories

Usage: cascadia-v2 lookahead-recall [OPTIONS]

Options:
      --games <GAMES>                              [default: 5]
      --first-seed <FIRST_SEED>                    [default: 20600]
      --retained-candidates <RETAINED_CANDIDATES>  [default: 4]
      --expanded-candidates <EXPANDED_CANDIDATES>  [default: 8]
      --determinizations <DETERMINIZATIONS>        [default: 4]
      --greedy-plies <GREEDY_PLIES>                [default: 4]
      --output <OUTPUT>                            
  -h, --help                                       Print help
```

## `nature-wipe-compare`

```text
Compare promoted lookahead with fair one-wipe Nature Token planning

Usage: cascadia-v2 nature-wipe-compare [OPTIONS]

Options:
      --games <GAMES>                                        [default: 5]
      --first-seed <FIRST_SEED>                              [default: 0]
      --candidates <CANDIDATES>                              [default: 8]
      --determinizations <DETERMINIZATIONS>                  [default: 4]
      --greedy-plies <GREEDY_PLIES>                          [default: 4]
      --prelude-candidates <PRELUDE_CANDIDATES>              [default: 4]
      --prelude-determinizations <PRELUDE_DETERMINIZATIONS>  [default: 2]
      --prelude-greedy-plies <PRELUDE_GREEDY_PLIES>          [default: 4]
      --output <OUTPUT>                                      
  -h, --help                                                 Print help
```

## `bear-candidate-compare`

```text
Compare promoted lookahead with a Bear-specific candidate union

Usage: cascadia-v2 bear-candidate-compare [OPTIONS]

Options:
      --games <GAMES>
          [default: 10]
      --first-seed <FIRST_SEED>
          [default: 0]
      --baseline-candidates <BASELINE_CANDIDATES>
          Immediate-score candidate count used by the baseline
      --candidates <CANDIDATES>
          [default: 8]
      --bear-candidates <BEAR_CANDIDATES>
          [default: 8]
      --determinizations <DETERMINIZATIONS>
          [default: 4]
      --greedy-plies <GREEDY_PLIES>
          [default: 4]
      --output <OUTPUT>
          
  -h, --help
          Print help
```

## `habitat-candidate-compare`

```text
Compare promoted lookahead with a habitat-cohesion candidate union

Usage: cascadia-v2 habitat-candidate-compare [OPTIONS]

Options:
      --games <GAMES>
          [default: 10]
      --first-seed <FIRST_SEED>
          [default: 0]
      --baseline-candidates <BASELINE_CANDIDATES>
          Immediate-score candidate count used by the baseline
      --candidates <CANDIDATES>
          [default: 8]
      --habitat-candidates <HABITAT_CANDIDATES>
          [default: 8]
      --determinizations <DETERMINIZATIONS>
          [default: 4]
      --greedy-plies <GREEDY_PLIES>
          [default: 4]
      --output <OUTPUT>
          
  -h, --help
          Print help
```

## `bear-habitat-candidate-compare`

```text
Compare H6 with a combined habitat- and Bear-aware candidate frontier

Usage: cascadia-v2 bear-habitat-candidate-compare [OPTIONS]

Options:
      --games <GAMES>                            [default: 10]
      --first-seed <FIRST_SEED>                  [default: 0]
      --candidates <CANDIDATES>                  [default: 8]
      --habitat-candidates <HABITAT_CANDIDATES>  [default: 6]
      --bear-candidates <BEAR_CANDIDATES>        [default: 8]
      --determinizations <DETERMINIZATIONS>      [default: 4]
      --greedy-plies <GREEDY_PLIES>              [default: 4]
      --output <OUTPUT>                          
  -h, --help                                     Print help
```

## `habitat-candidate-ablate`

```text
Compare two habitat-cohesion lookahead configurations on identical seeds

Usage: cascadia-v2 habitat-candidate-ablate [OPTIONS]

Options:
      --games <GAMES>                                                [default: 10]
      --first-seed <FIRST_SEED>                                      [default: 0]
      --baseline-candidates <BASELINE_CANDIDATES>                    [default: 8]
      --baseline-habitat-candidates <BASELINE_HABITAT_CANDIDATES>    [default: 6]
      --baseline-determinizations <BASELINE_DETERMINIZATIONS>        [default: 4]
      --baseline-greedy-plies <BASELINE_GREEDY_PLIES>                [default: 4]
      --treatment-candidates <TREATMENT_CANDIDATES>                  [default: 8]
      --treatment-habitat-candidates <TREATMENT_HABITAT_CANDIDATES>  [default: 6]
      --treatment-determinizations <TREATMENT_DETERMINIZATIONS>      [default: 4]
      --treatment-greedy-plies <TREATMENT_GREEDY_PLIES>              [default: 8]
      --output <OUTPUT>                                              
  -h, --help                                                         Print help
```

## `pattern-blueprint-compare`

```text
Compare H6 with the same root frontier using pattern-aware rollout plies

Usage: cascadia-v2 pattern-blueprint-compare [OPTIONS]

Options:
      --games <GAMES>                                          [default: 10]
      --first-seed <FIRST_SEED>                                [default: 0]
      --candidates <CANDIDATES>                                [default: 8]
      --habitat-candidates <HABITAT_CANDIDATES>                [default: 6]
      --determinizations <DETERMINIZATIONS>                    [default: 4]
      --rollout-plies <ROLLOUT_PLIES>                          [default: 4]
      --policy-candidates <POLICY_CANDIDATES>                  [default: 8]
      --policy-habitat-candidates <POLICY_HABITAT_CANDIDATES>  [default: 6]
      --policy-bear-candidates <POLICY_BEAR_CANDIDATES>        [default: 8]
      --policy-market-draws <POLICY_MARKET_DRAWS>              [default: 4]
      --output <OUTPUT>                                        
  -h, --help                                                   Print help
```

## `perfect-information-oracle-compare`

```text
Measure the K8+H6+B8 frontier with a diagnostic true-hidden-state oracle

Usage: cascadia-v2 perfect-information-oracle-compare [OPTIONS]

Options:
      --games <GAMES>                                          [default: 1]
      --first-seed <FIRST_SEED>                                [default: 0]
      --policy-candidates <POLICY_CANDIDATES>                  [default: 8]
      --policy-habitat-candidates <POLICY_HABITAT_CANDIDATES>  [default: 6]
      --policy-bear-candidates <POLICY_BEAR_CANDIDATES>        [default: 8]
      --policy-market-draws <POLICY_MARKET_DRAWS>              [default: 4]
      --output <OUTPUT>                                        
  -h, --help                                                   Print help
```

## `perfect-information-oracle-frontier-compare`

```text
Compare exact-hidden-state base and wildlife-diverse focal frontiers

Usage: cascadia-v2 perfect-information-oracle-frontier-compare [OPTIONS]

Options:
      --games <GAMES>                                          [default: 1]
      --first-seed <FIRST_SEED>                                [default: 0]
      --policy-candidates <POLICY_CANDIDATES>                  [default: 8]
      --policy-habitat-candidates <POLICY_HABITAT_CANDIDATES>  [default: 6]
      --policy-bear-candidates <POLICY_BEAR_CANDIDATES>        [default: 8]
      --wildlife-candidates <WILDLIFE_CANDIDATES>              [default: 2]
      --policy-market-draws <POLICY_MARKET_DRAWS>              [default: 4]
      --output <OUTPUT>                                        
  -h, --help                                                   Print help
```

## `perfect-information-focal-beam-compare`

```text
Compare exact one-step W2 with final-turn exact focal beam planning

Usage: cascadia-v2 perfect-information-focal-beam-compare [OPTIONS]

Options:
      --games <GAMES>                                          [default: 1]
      --first-seed <FIRST_SEED>                                [default: 0]
      --policy-candidates <POLICY_CANDIDATES>                  [default: 8]
      --policy-habitat-candidates <POLICY_HABITAT_CANDIDATES>  [default: 6]
      --policy-bear-candidates <POLICY_BEAR_CANDIDATES>        [default: 8]
      --wildlife-candidates <WILDLIFE_CANDIDATES>              [default: 2]
      --policy-market-draws <POLICY_MARKET_DRAWS>              [default: 4]
      --beam-width <BEAM_WIDTH>                                [default: 16]
      --terminal-turns <TERMINAL_TURNS>                        [default: 5]
      --output <OUTPUT>                                        
  -h, --help                                                   Print help
```

## `perfect-information-focal-frontier-compare`

```text
Compare W2 and W4 wildlife frontiers under the same exact focal beam

Usage: cascadia-v2 perfect-information-focal-frontier-compare [OPTIONS]

Options:
      --games <GAMES>                                                  [default: 1]
      --first-seed <FIRST_SEED>                                        [default: 0]
      --policy-candidates <POLICY_CANDIDATES>                          [default: 8]
      --policy-habitat-candidates <POLICY_HABITAT_CANDIDATES>          [default: 6]
      --policy-bear-candidates <POLICY_BEAR_CANDIDATES>                [default: 8]
      --baseline-wildlife-candidates <BASELINE_WILDLIFE_CANDIDATES>    [default: 2]
      --treatment-wildlife-candidates <TREATMENT_WILDLIFE_CANDIDATES>  [default: 4]
      --policy-market-draws <POLICY_MARKET_DRAWS>                      [default: 4]
      --beam-width <BEAM_WIDTH>                                        [default: 16]
      --terminal-turns <TERMINAL_TURNS>                                [default: 5]
      --output <OUTPUT>                                                
  -h, --help                                                           Print help
```

## `perfect-information-beam-capacity-compare`

```text
Compare width-16 and width-32 exact W2 focal beams

Usage: cascadia-v2 perfect-information-beam-capacity-compare [OPTIONS]

Options:
      --games <GAMES>                                          [default: 1]
      --first-seed <FIRST_SEED>                                [default: 0]
      --policy-candidates <POLICY_CANDIDATES>                  [default: 8]
      --policy-habitat-candidates <POLICY_HABITAT_CANDIDATES>  [default: 6]
      --policy-bear-candidates <POLICY_BEAR_CANDIDATES>        [default: 8]
      --wildlife-candidates <WILDLIFE_CANDIDATES>              [default: 2]
      --policy-market-draws <POLICY_MARKET_DRAWS>              [default: 4]
      --baseline-beam-width <BASELINE_BEAM_WIDTH>              [default: 16]
      --treatment-beam-width <TREATMENT_BEAM_WIDTH>            [default: 32]
      --terminal-turns <TERMINAL_TURNS>                        [default: 5]
      --output <OUTPUT>                                        
  -h, --help                                                   Print help
```

## `perfect-information-root-diverse-beam-compare`

```text
Compare W2 with root-only W4 under W2 future focal layers

Usage: cascadia-v2 perfect-information-root-diverse-beam-compare [OPTIONS]

Options:
      --games <GAMES>                                                            [default: 1]
      --first-seed <FIRST_SEED>                                                  [default: 0]
      --policy-candidates <POLICY_CANDIDATES>                                    [default: 8]
      --policy-habitat-candidates <POLICY_HABITAT_CANDIDATES>                    [default: 6]
      --policy-bear-candidates <POLICY_BEAR_CANDIDATES>                          [default: 8]
      --baseline-root-wildlife-candidates <BASELINE_ROOT_WILDLIFE_CANDIDATES>    [default: 2]
      --treatment-root-wildlife-candidates <TREATMENT_ROOT_WILDLIFE_CANDIDATES>  [default: 4]
      --future-wildlife-candidates <FUTURE_WILDLIFE_CANDIDATES>                  [default: 2]
      --policy-market-draws <POLICY_MARKET_DRAWS>                                [default: 4]
      --beam-width <BEAM_WIDTH>                                                  [default: 16]
      --terminal-turns <TERMINAL_TURNS>                                          [default: 5]
      --output <OUTPUT>                                                          
  -h, --help                                                                     Print help
```

## `perfect-information-portfolio-beam-compare`

```text
Compare scalar and portfolio-preserving exact focal beam retention

Usage: cascadia-v2 perfect-information-portfolio-beam-compare [OPTIONS]

Options:
      --games <GAMES>                                          [default: 1]
      --first-seed <FIRST_SEED>                                [default: 0]
      --policy-candidates <POLICY_CANDIDATES>                  [default: 8]
      --policy-habitat-candidates <POLICY_HABITAT_CANDIDATES>  [default: 6]
      --policy-bear-candidates <POLICY_BEAR_CANDIDATES>        [default: 8]
      --wildlife-candidates <WILDLIFE_CANDIDATES>              [default: 2]
      --policy-market-draws <POLICY_MARKET_DRAWS>              [default: 4]
      --beam-width <BEAM_WIDTH>                                [default: 16]
      --terminal-turns <TERMINAL_TURNS>                        [default: 5]
      --output <OUTPUT>                                        
  -h, --help                                                   Print help
```

## `public-focal-beam-compare`

```text
Compare promoted strong with a public redetermined focal-beam teacher

Usage: cascadia-v2 public-focal-beam-compare [OPTIONS]

Options:
      --games <GAMES>                                          [default: 1]
      --first-seed <FIRST_SEED>                                [default: 0]
      --terminal-turns <TERMINAL_TURNS>                        [default: 5]
      --determinizations <DETERMINIZATIONS>                    [default: 4]
      --beam-width <BEAM_WIDTH>                                [default: 4]
      --policy-candidates <POLICY_CANDIDATES>                  [default: 8]
      --policy-habitat-candidates <POLICY_HABITAT_CANDIDATES>  [default: 6]
      --policy-bear-candidates <POLICY_BEAR_CANDIDATES>        [default: 8]
      --wildlife-candidates <WILDLIFE_CANDIDATES>              [default: 2]
      --policy-market-draws <POLICY_MARKET_DRAWS>              [default: 4]
      --sequential                                             
      --output <OUTPUT>                                        
  -h, --help                                                   Print help
```

## `public-focal-tree-compare`

```text
Compare promoted strong with public open-loop focal tree search

Usage: cascadia-v2 public-focal-tree-compare [OPTIONS]

Options:
      --games <GAMES>                                          [default: 1]
      --first-seed <FIRST_SEED>                                [default: 0]
      --terminal-turns <TERMINAL_TURNS>                        [default: 5]
      --simulations <SIMULATIONS>                              [default: 128]
      --root-candidates <ROOT_CANDIDATES>                      [default: 16]
      --exploration-milli <EXPLORATION_MILLI>                  [default: 2000]
      --policy-candidates <POLICY_CANDIDATES>                  [default: 8]
      --policy-habitat-candidates <POLICY_HABITAT_CANDIDATES>  [default: 6]
      --policy-bear-candidates <POLICY_BEAR_CANDIDATES>        [default: 8]
      --wildlife-candidates <WILDLIFE_CANDIDATES>              [default: 2]
      --policy-market-draws <POLICY_MARKET_DRAWS>              [default: 4]
      --sequential                                             
      --output <OUTPUT>                                        
  -h, --help                                                   Print help
```

## `terminal-policy-improvement-compare`

```text
Compare pattern-aware with full-game one-step policy improvement

Usage: cascadia-v2 terminal-policy-improvement-compare [OPTIONS]

Options:
      --games <GAMES>                                          [default: 1]
      --first-seed <FIRST_SEED>                                [default: 0]
      --determinizations <DETERMINIZATIONS>                    [default: 2]
      --policy-candidates <POLICY_CANDIDATES>                  [default: 8]
      --policy-habitat-candidates <POLICY_HABITAT_CANDIDATES>  [default: 6]
      --policy-bear-candidates <POLICY_BEAR_CANDIDATES>        [default: 8]
      --policy-market-draws <POLICY_MARKET_DRAWS>              [default: 4]
      --output <OUTPUT>                                        
  -h, --help                                                   Print help
```

## `late-terminal-policy-improvement-compare`

```text
Compare pattern-aware with R8 terminal search on only the final personal turns

Usage: cascadia-v2 late-terminal-policy-improvement-compare [OPTIONS]

Options:
      --games <GAMES>                                          [default: 1]
      --first-seed <FIRST_SEED>                                [default: 0]
      --terminal-turns <TERMINAL_TURNS>                        [default: 4]
      --determinizations <DETERMINIZATIONS>                    [default: 8]
      --policy-candidates <POLICY_CANDIDATES>                  [default: 8]
      --policy-habitat-candidates <POLICY_HABITAT_CANDIDATES>  [default: 6]
      --policy-bear-candidates <POLICY_BEAR_CANDIDATES>        [default: 8]
      --policy-market-draws <POLICY_MARKET_DRAWS>              [default: 4]
      --sequential                                             
      --output <OUTPUT>                                        
  -h, --help                                                   Print help
```

## `late-wildlife-diverse-policy-improvement-compare`

```text
Compare pattern-aware with a wildlife-diverse final-turn R8 frontier

Usage: cascadia-v2 late-wildlife-diverse-policy-improvement-compare [OPTIONS]

Options:
      --games <GAMES>                                          [default: 1]
      --first-seed <FIRST_SEED>                                [default: 0]
      --terminal-turns <TERMINAL_TURNS>                        [default: 5]
      --determinizations <DETERMINIZATIONS>                    [default: 8]
      --policy-candidates <POLICY_CANDIDATES>                  [default: 8]
      --policy-habitat-candidates <POLICY_HABITAT_CANDIDATES>  [default: 6]
      --policy-bear-candidates <POLICY_BEAR_CANDIDATES>        [default: 8]
      --wildlife-candidates <WILDLIFE_CANDIDATES>              [default: 2]
      --policy-market-draws <POLICY_MARKET_DRAWS>              [default: 4]
      --sequential                                             
      --output <OUTPUT>                                        
  -h, --help                                                   Print help
```

## `late-conservative-policy-improvement-compare`

```text
Compare pattern-aware with confidence-gated final-turn R8 improvement

Usage: cascadia-v2 late-conservative-policy-improvement-compare [OPTIONS]

Options:
      --games <GAMES>                                          [default: 1]
      --first-seed <FIRST_SEED>                                [default: 0]
      --terminal-turns <TERMINAL_TURNS>                        [default: 5]
      --policy-candidates <POLICY_CANDIDATES>                  [default: 8]
      --policy-habitat-candidates <POLICY_HABITAT_CANDIDATES>  [default: 6]
      --policy-bear-candidates <POLICY_BEAR_CANDIDATES>        [default: 8]
      --wildlife-candidates <WILDLIFE_CANDIDATES>              [default: 2]
      --policy-market-draws <POLICY_MARKET_DRAWS>              [default: 4]
      --sequential                                             
      --output <OUTPUT>                                        
  -h, --help                                                   Print help
```

## `late-conservative-base-policy-improvement-compare`

```text
Compare pattern-aware with confidence-gated R8 on the original frontier

Usage: cascadia-v2 late-conservative-base-policy-improvement-compare [OPTIONS]

Options:
      --games <GAMES>                                          [default: 1]
      --first-seed <FIRST_SEED>                                [default: 0]
      --terminal-turns <TERMINAL_TURNS>                        [default: 5]
      --policy-candidates <POLICY_CANDIDATES>                  [default: 8]
      --policy-habitat-candidates <POLICY_HABITAT_CANDIDATES>  [default: 6]
      --policy-bear-candidates <POLICY_BEAR_CANDIDATES>        [default: 8]
      --policy-market-draws <POLICY_MARKET_DRAWS>              [default: 4]
      --sequential                                             
      --output <OUTPUT>                                        
  -h, --help                                                   Print help
```

## `late-conservative-wildlife-focused-policy-improvement-compare`

```text
Compare promoted strong with confidence-gated focused-species coverage

Usage: cascadia-v2 late-conservative-wildlife-focused-policy-improvement-compare [OPTIONS] --wildlife <WILDLIFE>

Options:
      --games <GAMES>
          [default: 1]
      --first-seed <FIRST_SEED>
          [default: 0]
      --terminal-turns <TERMINAL_TURNS>
          [default: 5]
      --determinizations <DETERMINIZATIONS>
          [default: 8]
      --policy-candidates <POLICY_CANDIDATES>
          [default: 8]
      --policy-habitat-candidates <POLICY_HABITAT_CANDIDATES>
          [default: 6]
      --policy-bear-candidates <POLICY_BEAR_CANDIDATES>
          [default: 8]
      --wildlife <WILDLIFE>
          [possible values: bear, elk, salmon, hawk, fox]
      --wildlife-candidates <WILDLIFE_CANDIDATES>
          [default: 2]
      --policy-market-draws <POLICY_MARKET_DRAWS>
          [default: 4]
      --sequential
          
      --output <OUTPUT>
          
  -h, --help
          Print help
```

## `conservative-sample-count-compare`

```text
Compare conservative final-five policies at two supported sample counts

Usage: cascadia-v2 conservative-sample-count-compare [OPTIONS]

Options:
      --games <GAMES>                                            [default: 1]
      --first-seed <FIRST_SEED>                                  [default: 0]
      --terminal-turns <TERMINAL_TURNS>                          [default: 5]
      --baseline-determinizations <BASELINE_DETERMINIZATIONS>    [default: 8]
      --treatment-determinizations <TREATMENT_DETERMINIZATIONS>  [default: 32]
      --policy-candidates <POLICY_CANDIDATES>                    [default: 8]
      --policy-habitat-candidates <POLICY_HABITAT_CANDIDATES>    [default: 6]
      --policy-bear-candidates <POLICY_BEAR_CANDIDATES>          [default: 8]
      --policy-market-draws <POLICY_MARKET_DRAWS>                [default: 4]
      --sequential                                               
      --output <OUTPUT>                                          
  -h, --help                                                     Print help
```

## O1 opponent-intent MLX factorial

The research runner is installed as `cascadia-mlx-opponent-intent` and is also
available through:

```bash
.venv/bin/python -m cascadia_mlx.opponent_intent_experiment --help
```

Subcommands:

- `authorize`: bind the two train roles, open validation, corpus
  classification, train-only priors, graph, and initialization;
- `verify-authorization`: rebuild all authorization fields without creating an
  optimizer or run directory;
- `run`: train one primary or replay arm with exact checkpoints and validation
  evidence;
- `classify`: require all eight crossed-host reports, models, and evidence
  files before applying frozen validation gates;
- `evaluate-selected`: leave sealed roles unopened after a null, or compare the
  selected treatment with A0 once on test and then descriptive final stress.

Production runs require exactly two `--train-dataset` arguments. Policy IDs,
physical tile IDs, future actions, and final scores are never accepted as
model inputs.

## R2-MAP deterministic campaign controller

`tools/r2_map_expert_iteration.py` adapts immutable R2-MAP work packets to the
existing `cluster_research_queue.py` and `cluster_experiment_ledger.py`. It is
not a second scheduler. ADR 0195 makes John2's internal APFS root canonical.
Canonical controller commands run on John2 (or through a fixed authenticated
SSH invocation) with this environment:

```bash
export PYTHONDONTWRITEBYTECODE=1
export R2_MAP_CAMPAIGN_ROOT=/Users/john2/cascadia-bench/r2-map-v1
export PYTHONPYCACHEPREFIX=$R2_MAP_CAMPAIGN_ROOT/tmp/pycache
export TMPDIR=$R2_MAP_CAMPAIGN_ROOT/tmp
export CARGO_TARGET_DIR=$R2_MAP_CAMPAIGN_ROOT/build/cargo-target
export UV_CACHE_DIR=$R2_MAP_CAMPAIGN_ROOT/cache/uv
```

Commands:

- `advance-phase`: prove the current receipt barrier, CAS the campaign state,
  create immutable packets, and atomically add their DAG to the existing queue;
- `import-work-receipt`: accept a receipt only from its registered
  `incoming-receipts/john1|john2|john3` directory after the queue task completes;
- `import-benchmark-feed`: verify that a deterministic compact ledger feed is
  uniquely path/SHA-bound by an already imported John1 benchmark-aggregate
  receipt, compare-and-swap the current campaign state, stamp the receipt's
  completion time in memory, and idempotently upsert through the existing
  locked ledger;
- `phase-barrier`: require every packet task and immutable receipt;
- `reconcile-controller`: validate packet/queue identities, expire stale
  claims, repair the ledger projection, and publish dashboard inputs;
- `recover-phase`: reconstruct deterministic packets after a crash between
  state CAS, packet installation, queue update, and ledger projection; and
- `w6-dry-run`: exercise bootstrap, rejection, and promotion transition shapes
  using only an isolated John2 root and synthetic receipts.

`advance-phase` and `recover-phase` require a reviewed JSON command manifest
mapping every operation in the next phase to an argv array. Commands, fixed
host, state hash, phase intent, seed lease, retry ceiling, outputs, and gates
are included in the packet hash. `/usr/bin/true` is rejected outside the
explicit isolated dry-run path.

No operator may edit the queue, ledger, campaign state, packet, or imported
receipt by hand. A stale or partial projection is repaired with
`recover-phase`/`reconcile-controller`; identity drift fails closed.

### Headless goal supervisor

`tools/r2_map_headless_resume.sh` runs Codex from John1 but stores no turn
output locally. Anonymous pipes stream the JSONL event log and stderr directly
to canonical John2 `logs/headless/`; owner/lease, stop check, and terminal
sentinel are remote control objects. Every write uses the
content-addressed remote worker's bounded `put-stream` operation and verified
receipt; the runner lock uses its expiring lease API. The underlying transport
uses batch-authenticated, strict-host-key SSH and never names
`/Volumes/John_1`.

`tools/r2_map_headless_turn.py` owns both anonymous-pipe sinks and the Codex
child for one turn. It drains sink diagnostics concurrently, bounds retained
diagnostics, hashes the bytes copied into each sink, and requires each
authenticated result to match the exact remote path, byte count, mode `0400`,
stream SHA-256, and storage-receipt locator/SHA-256. An early or nonzero sink,
broken pump, oversized diagnostic, malformed receipt, or receipt mismatch
terminates the Codex child and exits `74`; the supervisor never advances the
turn counter. The multiplexer also watches the independent remote-lock
heartbeat PID and terminates Codex if lease renewal stops. Shell process
substitutions are intentionally not used because zsh does not propagate a
process-substitution exit through the foreground command's status or a later
bare `wait`.

```bash
tmux new-session -d -s cascadia-r2-headless \
  'cd /Users/johnherrick/cascadia && exec tools/r2_map_headless_resume.sh'

tmux capture-pane -pt cascadia-r2-headless
```

The pane prints the immutable `SESSION_RUN_ID`. Inspect a completed turn without
a local file using `tools/r2_map_remote_storage.py fetch --relative
logs/headless/SESSION_RUN_ID/turn-NNNN.jsonl`.

Request a clean stop before the next turn by atomically installing
`control/headless-STOP` on John2. Do not create a terminal sentinel merely to
stop the supervisor. The runner renews its content-addressed lease every five
minutes; a crashed runner can be replaced only after that lease expires or an
identity-bound release succeeds.

```bash
printf 'operator stop requested\n' | PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=python \
  .venv/bin/python tools/r2_map_remote_storage.py put-stream \
  --relative control/headless-STOP --max-bytes 1024
```

After the aggregate work receipt is imported, centralize its deterministic
ledger feed without changing the feed or scientific report bytes:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/r2_map_expert_iteration.py \
  import-benchmark-feed \
  --feed /Users/john2/cascadia-bench/r2-map-v1/benchmarks/ROUND/projections/ledger-experiment.json \
  --aggregate-task-id TASK_ID \
  --expected-state-sha256 CURRENT_STATE_SHA256
```

The feed must use zero placeholder times and a distinct completed experiment
ID. Only the controller-authorized John2 canonical import replaces those
placeholders, using the immutable
aggregate receipt's `completed_unix_ms`; it appends the feed and receipt hashes
to the ledger note. A stale state hash, nonterminal queue task, non-John1
aggregate, path escape, duplicate binding, byte tamper, or invalid ledger
object is rejected.

## R2-MAP topology-free serving and benchmark work items

The active distributed path is `tools/r2_map_bacalhau_gate.py`, documented in
[`R2_MAP_EXPERT_ITERATION_RESEARCH_PLAN.md`](R2_MAP_EXPERT_ITERATION_RESEARCH_PLAN.md).
It stages content-addressed inputs, submits one independent `pair-NNNN` job per
registered pair by immutable image digest, imports validated MinIO outputs, and
aggregates in a container. Callers never pass a host, worker root, SSH command,
compatibility list, parity split, or manual lease.

The Rust benchmark path uses the verified exhaustive R2-MAP service directly;
it never prunes actions. Checkpoint-controlled turns use sequential public
market decisions: free-replacement keep/replace, then one stop-or-single-wipe
decision per reveal, then the exhaustive draft. Benchmark selection is argmax
at every stage, and the canonical replay action retains the full ordered
prelude so every paid wipe is independently auditable. No future refill may be
observed before its choice is committed. Optional market actions are derived
from visible wildlife plus per-species public bag counts and must be executable
for every consistent hidden order; partial public legal sets fail closed.
Serving bundle v2 also binds collector, source, and serving-protocol hashes and
rejects stale identities. The v3 live draft response carries only the four
consumed tensors (action score, score-to-go, 11 components, bootstrap logits),
while checkpoint fixed panels retain every training-only auxiliary head.
Canonical bundles, checkpoints, source, and images are prepared on John1.
Workers receive only immutable content-addressed inputs and the John1-built
image digest through Bacalhau. Preparing the canonical frozen model registry:

```bash
cargo run -p cascadia-cli-v2 -- prepare-r2-map-serving-bundle \
  --host john1 \
  --output /Users/johnherrick/cascadia-bench/r2-map-v1/bundles/c0.json \
  --checkpoint /Users/johnherrick/cascadia-bench/r2-map-v1/runs/RUN/checkpoints/CHECKPOINT
```

The command reconstructs every compact identity from the schema-v2 checkpoint
and standalone verification receipt, writes one immutable bundle, then asks
both Rust and Python-compatible bundle validation to verify it. Pass
`--checkpoint` more than once for a candidate plus its frozen historical pool.

Initialize the non-protected W0 fixed-100 longitudinal panel:

```bash
cargo run -p cascadia-cli-v2 -- init-r2-map-longitudinal-open-panel \
  --root /Users/johnherrick/cascadia-bench/r2-map-v1/benchmarks/open-r0 \
  --reference-panel-manifest /Users/johnherrick/cascadia-bench/r2-map-v1/control/w0-preregistration/reference-panel-manifest-v1.1.json \
  --reference-panel-registration /Users/johnherrick/cascadia-bench/r2-map-v1/control/w0-preregistration/registration-v1.1.json \
  --benchmark-id r2-map-open-r0 \
  --focal-checkpoint-id CHECKPOINT \
  --field-manifest-id open-r0-field \
  --historical-checkpoint CHECKPOINT
```

This accepts only the append-only sequential-public-market v1.1 registration
and its registered `open-performance-100` panel. The immutable v1 registration
is retained only as a stale predecessor and is rejected for launch. V1.1
reuses exactly the already-open 100-seed domain; it does not open a protected
domain. The initializer requires the panel's non-protected and
no-strength-claim flags, creates one scheduler-managed `game-NNNN` item per
index, rotates the focal seat by `index mod 4`, and freezes all 100 seeds and
opponent seats before execution. A W3 smoke checkpoint is valid
only for real implementation/performance evidence; its report is labeled
`real-open-checkpoint-performance-only` and cannot support a strength claim.
It is also labeled `open-performance-reference-only`; the registered
reference/optimized two-order comparison remains incomplete until an optimized
mode exists and passes exact parity. This command never fabricates that second
arm. The initializer verifies the registration's exact formatted and canonical
manifest identities, the manifest and selected panel canonical hashes, and the
live SHA-256 of every required game, R2, model, search, evaluator, and service
source binding. A superseded or locally edited registration cannot launch.

Production iteration-longitudinal seeds and historical assignments come from
the controller packet, not the W0 open panel. Install those already frozen
inputs with:

```bash
cargo run -p cascadia-cli-v2 -- init-r2-map-longitudinal-campaign \
  --root /Users/johnherrick/cascadia-bench/r2-map-v1/benchmarks/ROUND \
  --contract CONTROLLER_LONGITUDINAL_CONTRACT.json \
  --historical-field CONTROLLER_HISTORICAL_FIELD.json
```

The contract must declare `expert-iteration-longitudinal`, exactly 100
scheduler-managed games, a frozen pool, and
`strength_claim_authorized=false`. It contains no host, parity, or manual lease.

Inside a Bacalhau execution, run one independent game work item:

```bash
cargo run -p cascadia-cli-v2 -- run-r2-map-longitudinal-work-item \
  --root /input/campaign \
  --work-item game-0000 --bundle /input/r2-run/bundle.json \
  --collector-hash COLLECTOR_BLAKE3 --source-hash SOURCE_BLAKE3 \
  --serving-protocol-hash SERVING_BLAKE3
```

Each completed game becomes one fsync-and-rename receipt. Rerunning validates
and resumes complete receipts without replaying them. A service failure gets
one bounded restart and exact same-request retry. Extra, partial, duplicate,
identity-drifted, or hash-drifted receipts fail closed. Central aggregation:

```bash
cargo run -p cascadia-cli-v2 -- aggregate-r2-map-longitudinal \
  --root /Users/johnherrick/cascadia-bench/r2-map-v1/benchmarks/open-r0 \
  --wall-seconds SECONDS
```

It requires exact 0-99 coverage and reproduces all 100 work-item summaries from
game receipts. In addition to the full report it writes compact, deterministic
projections under `projections/dashboard-benchmark.json` and
`projections/ledger-experiment.json`; only the controller may install those
into canonical John1 dashboard and ledger state.

The 20-pair blinded smoke and fixed-250 gate use pre-provisioned immutable
contract/field files. Use the topology-free controller; infrastructure setup
does not open protected seed values:

```bash
PYTHONPATH=python python3 tools/r2_map_bacalhau_gate.py \
  --stage smoke --image 'REGISTRY/IMAGE@sha256:DIGEST' \
  --gate-directory GATE_INPUT --candidate-freeze CANDIDATE_FREEZE \
  --exact-weights nnue_weights_v4opp_modal_iter3.bin \
  --state-directory BACALHAU_CLIENT_STATE \
  --artifact-directory BACALHAU_ACCEPTED_RESULTS \
  --campaign-directory CANONICAL_SMOKE_CAMPAIGN
```

Candidate/control physical execution order alternates by pair index while the
receipt remains normalized as candidate then control. The smoke projection
contains no score output. The fixed-250 projection carries the same report and
contract/field SHA-256 and BLAKE3 identities consumed by strict promotion.
These commands must be launched only from the controller's authorized phase;
they do not derive, reveal, or accept the untouched final-domain seeds.

## R2-MAP compact replay training

`tools/r2_map_compact_dataset.py` owns the production `.r2sh` data boundary:

- `build-index` sequentially validates shards through the Rust exporter and
  writes a compact game/source index; temporary `.r2map` windows are bounded
  and deleted.
- `validate` rechecks index identity plus every source byte count and BLAKE3.
- `project` proves the requested game count, index, current window, and one
  optional prefetched window fit the 40-GiB run budget. It exits 2 if compact
  storage fails or if the 2-MB/game expanded-corpus counterfactual fits.
- `smoke` proves deterministic repeat, reopen/next-batch parity, streamed
  validation, fixed-panel construction, and zero leftover windows.

On John2, reference/local validation of `python -m cascadia_mlx.r2_map_train` uses
`--compact-index`, `--compact-shard-root`, `--compact-exporter`, and
`--compact-window-dir`. `--maximum-window-bytes` defaults to 1 GiB and
`--maximum-prefetch-windows` is restricted to 0 or 1. The CLI projects 100,000
games before allocating the adapter and fails closed above 40 GiB. This local
path mode requires John2 storage authority and cannot run on John1.

Production John1 MLX training uses the authenticated remote-storage training
path. It fetches the compact index into bounded memory, asks John2 to export
exactly one `.json` plus `.r2map` window beneath a registered ephemeral run,
verifies token-bound range receipts and source identity, and then commits the
run-cleanup CAS before moving to another window. Prefetch is zero. Loss events
are published every 20 steps, and each complete checkpoint is installed as one
immutable John2 transaction before the mutable pointers advance. No dataset
window, checkpoint, loss log, cache, or temporary product is written on John1.

The production entry point is `tools/r2_map_john1_train.py`. It uses
`r2_map_remote_training.py` and the frozen John2 client directly; the local
`r2_map_train` CLI remains a John2-only reference path. Its sole data/packing
input is `--packing-report-relative`. The command independently reopens the W0
source transaction, the post-W7 bootstrap phase barrier, all four controller
packet/receipt pairs, the exact 100,000-game dataset transaction and compact
index, the qualifying packing report publication receipt, and the zero-write
attestation plus its digest-derived direct put receipt. Cap, candidate budget,
exact 12-epoch schedule, index, shard root, and exporter are all derived; there
are no CLI overrides. The command fixes
prefetch to zero, retains each window's ordered open/range/cleanup evidence,
publishes loss-stream updates with SHA-256 CAS at complete checkpoints, installs
each checkpoint as an immutable transaction, reopens every committed object,
publishes exact-panel verification, and only then advances `latest_complete` and
`last_verified`. Resume reopens the pointer, verification receipt, loss prefix,
and every checkpoint object before calling `resume_from_bundle`.

The qualifying pre-training entry point is
`tools/r2_map_john1_packing_sweep.py`. Its only storage locators are the W0
source transaction manifest/commit receipt and the W7 dataset transaction
manifest. The source manifest, v1.1 reference panel, compact exporter, bootstrap
phase barrier, compact index, shard root, and dataset commit receipt are derived
from those immutable authorities. Redundant caller-selected dataset paths and
all caller-selected identity digests are rejected by construction.

SSH compression is off by default. `--ssh-compression` is rejected unless the
caller also supplies `--compression-measurement-sha256`, the digest of a
recorded bulk-window comparison. The result records that digest; compression
cannot be enabled as an undocumented tuning guess.

The legacy six manifest/stream arguments require the explicit
`--allow-reference-expanded-streams` flag and are only for small regression
fixtures. `--maximum-reference-stream-bytes` defaults to 1 GiB, so the flag
cannot authorize a corpus-scale expansion. This is not an authorized bootstrap
data path.
