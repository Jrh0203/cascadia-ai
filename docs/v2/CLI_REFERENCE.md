# Cascadia V2 CLI Reference

Generated from the typed Clap schema. Regenerate with `make cli-docs`.

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
