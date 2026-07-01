# Implementation Status

This file tracks what has moved from plan text into implementation inside
`cascadiav3/`.

## Completed Now

- Self-contained standard-library Python package under `src/cascadiav3`.
- Radius 6 canonical coordinate table with exactly 127 cells.
- Exact overflow coordinate references carrying owner and placement identity.
- Search-root schema validation for legal actions, priors, visits,
  per-action Q labels, selected action, final score vector, score decomposition,
  rank vector, and checksum.
- Replay manifest validation for dry-run fixtures.
- Deterministic tiny search-root fixture with one canonical placement action and
  one exact-overflow placement action.
- Mock CascadiaFormer-Zero-S shape backend for CPU-only tensor contract checks.
- Validation CLI that can write fixture and report artifacts.
- Unit tests for radius-6 indexing, overflow, root-record validation, manifest
  validation, mock model shapes, and the validation CLI.
- Torch GPU smoke module that turns the tiny search-root fixture into tensors,
  runs a small model-shaped forward/backward pass, and writes a JSON report.
- Reusable tiny Torch model module shared by GPU smoke and one-root overfit.
- One-root overfit command that verifies loss decrease and checkpoint
  save/load on the RTX 5090.
- Tiny JSONL replay shard with two validated roots and variable legal action
  counts.
- Torch replay Dataset/DataLoader collate that pads action tensors and emits a
  boolean legal-action mask.
- Replay-batch training smoke that verifies masked loss, loss decrease, and
  checkpoint round-trip.
- Isolated Rust real-root exporter under `real-root-exporter/` that reads the
  canonical Cascadia simulator through path dependencies, emits v3 JSONL, and
  writes a manifest without touching the root workspace metadata.
- Real simulator dry-run root shard generated from 4-player Card A/no-bonus
  games with bounded greedy-ranked legal actions and one greedy terminal
  rollout per retained action.
- Replay trainer accepts an explicit `--replay` path, allowing the same GPU
  smoke to train against synthetic fixtures, real simulator dry-runs, and
  future larger shards.
- CRT-mini action-query merit pilot module that trains a small
  TransformerEncoder against a same-feature MLP and an immediate-score baseline
  on held-out simulator roots.
- Public-token root export sidecar from the Rust real-root exporter with
  `cascadiav3.public_tokens.v1`, player/board/frontier/market/supply tokens,
  directed `adjacent_hex` relations, and bidirectional `same_market_slot`
  relations.
- Public-token action-query merit pilot module that trains a small
  TransformerEncoder against a token-pooled MLP and an immediate-score baseline
  on held-out simulator roots.
- Relation-bias action-query merit pilot module with C-GAB-style learned
  additive attention bias over same-board, hex, market, and action-pointer
  relations.
- Sampled-teacher exporter labels with parallel per-seed generation on `john0`
  CPU, repeated top-k continuation rollouts per retained action, and optional
  `per_action_Q_variance` / `per_action_Q_count` arrays.
- Sampled-teacher truncation accounting with `per_action_truncated_count` for
  resource-exhausted continuation samples.
- Top-K prefilter retention metrics in action-ranking reports: top-2/top-4/
  top-8/top-16/top-24/top-32 recall plus top-K oracle regret.
- `scripts/run_crt_merit_pilot.sh` to sync code to `john0`, generate 400 train
  and 100 validation simulator roots on `john0` CPU, run the RTX 5090 merit
  pilot, and pull fixtures/reports/checkpoints back.
- `scripts/run_crt_public_token_pilot.sh` to sync code to `john0`, generate 400
  train and 100 validation public-token simulator roots on `john0` CPU, run the
  RTX 5090 public-token merit pilot, and pull fixtures/reports/checkpoints
  back.
- `scripts/run_crt_relation_bias_pilot.sh` to sync code to `john0`, reuse or
  regenerate public-token simulator roots on `john0` CPU, run the RTX 5090
  relation-bias merit pilot, and pull fixtures/reports/checkpoints back.
- `scripts/run_crt_sampled_teacher_relation_bias_pilot.sh` to generate sampled
  teacher roots on `john0` CPU, train the relation-bias Transformer on the RTX
  5090, and pull the first green-light v3 Transformer artifacts back.
- `scripts/run_crt_scaled_sampled_teacher_relation_bias_pilot.sh` to scale the
  green-light sampled-teacher relation-bias recipe to 1600 train roots, 400
  validation roots, 8 rollout samples per action, and a wider 4-layer model.
- `scripts/run_crt_wide32_sampled_teacher_relation_bias_pilot.sh` to widen the
  sampled-teacher relation-bias experiment to 32 retained legal actions per
  root and measure top-8/top-16 prefilter usefulness.
- `scripts/run_crt_wide32_r16x2_sampled_teacher_relation_bias_pilot.sh` to
  double the stronger R16 wide-32 data scale, train the standard relation-bias
  Transformer on the RTX 5090, and replay the checkpoint as a serving-shaped
  top-K prefilter.
- Semantic action-conditioned relation-bias module with 28 additional action
  features for local habitat geometry, drafted-species context, opponent
  species counts, public supply, and Card-A wildlife pattern signals.
- Semantic-aware public-token collation and checkpoint replay, while preserving
  backward compatibility for existing 33-feature public-token checkpoints.
- `scripts/run_crt_wide32_r16x2_semantic_relation_bias_pilot.sh` to reuse the
  doubled R16 opening shard and test the 61-feature semantic relation-bias
  model.
- `scripts/run_crt_wide32_r16p20_semantic_relation_bias_pilot.sh` to generate a
  phase-diverse 20-ply semantic shard, train on the RTX 5090, and replay the
  checkpoint as a serving-shaped top-K prefilter.
- `topk-retention` loss mode, which directly trains the teacher top-K action
  set above lower-ranked actions for serving prefilter retention.
- Semantic cross-attention action-query module that encodes public state tokens
  once and lets each legal action query cross-attend to that state.
- Semantic residual-attention action-query module that anchors on the semantic
  token-pooled MLP and adds a scaled public-token cross-attention correction.
- First-class semantic vanilla public-token Transformer module that promotes
  the p80x2 vanilla side-member signal into its own checkpoint family with the
  same 61-feature semantic action tensor and top-k retention objective.
- Semantic action-set Transformer module that pools public tokens into a root
  context token, then self-attends only over the legal action set.
- Semantic species-MoE action-query module with a shared relation-bias encoder
  and per-wildlife residual scoring heads for no-wildlife, bear, elk, salmon,
  hawk, and fox actions.
- Generic prefilter blend evaluator support for relation-bias, cross-attention,
  residual-attention, and action-set checkpoint families.
- Checkpoint-backed prefilter forensic analyzer that replays action-set,
  vanilla, MLP, and immediate sources to explain held-out K=16 miss sets.
- Train-selected source-union prefilter evaluator that tests whether simple
  quota unions over complementary source rankings can close the K=16 recall gap.
- Learned source-gating evaluator that trains only on train-shard
  source-score/rank features, selects by an inner tune split, and evaluates once
  on held-out validation.
- Fixed seed-ensemble prefilter evaluator that reads per-root checkpoint replay
  rows, normalizes each source per root, averages explicitly supplied weights,
  and reports the same top-K serving gate without train-selecting weights.
- Prefilter replay support for semantic vanilla public-token checkpoints as a
  primary model family, not only as `--checkpoint-member vanilla` sidecars.
- Interactive Rust policy-game mode in `real-root-exporter` that streams exact
  simulator roots over JSON, accepts retained action ids from an external
  policy, runs sampled rollout search inside the retained set, and can shadow
  the full-32 search winner for decision-level regret telemetry.
- Python/Torch game-pilot controller that keeps the first-class vanilla
  public-token seed ensemble resident on CUDA, ranks each streamed root,
  returns K16 retained actions to Rust, and writes report, per-decision JSONL,
  and Markdown summary artifacts.
- Compact greedy behavior-cloning tensor shard format in
  `src/cascadiav3/greedy_tensor_shards.py`: streamed JSONL input, versioned
  `float16` `.npz` output, per-root offsets, selected greedy action labels, and
  shard summaries.
- Greedy policy pretrainer support for both JSONL and compact `.npz` corpora via
  `--train-format` and `--val-format`, with `.npz` loader/checkpoint smoke
  validated on the RTX 5090.
- `real-root-exporter --greedy-policy-corpus --out -` stdout streaming mode so
  large greedy corpora can be compacted directly without persisting raw JSONL.
- `real-root-exporter --greedy-policy-tensor-corpus` Rust-native feature
  extraction and `.npz` writing for the canonical greedy pretraining tensors:
  public-token features, semantic public-token action features, offsets, and
  selected action labels.
- Rust-native tensor compression control via
  `--tensor-compression deflate|stored`, with parity checked against the Python
  feature path and `stored` measured at `5.22x` the deflated throughput for
  `17.48x` the disk footprint on the 1,024-game `john0` benchmark.
- `scripts/run_greedy_policy_pretrain.sh` to launch, monitor, stop, and fetch
  greedy behavior-cloning pretraining on `john0`; it defaults to Rust-native
  compact `.npz` shards, exposes `TENSOR_COMPRESSION`, and keeps JSONL only when
  explicitly requested.
- `scripts/run_crt_wide32_r16p20_semantic_retention_pilot.sh` to train the
  semantic relation-bias model with direct top-K retention supervision.
- `scripts/run_crt_wide32_r16p20_semantic_cross_attention_pilot.sh` to train and
  replay the semantic cross-attention action-query model.
- `scripts/run_crt_wide32_r16p20_semantic_residual_attention_pilot.sh` to train
  and replay the semantic residual-attention action-query model.
- `scripts/run_crt_wide32_r16p20_semantic_action_set_pilot.sh` to train and
  replay the semantic action-set Transformer.
- `scripts/run_crt_wide32_r16p20_semantic_species_moe_pilot.sh` to train the
  semantic species-MoE Transformer and same-run baselines on `john0`.
- `scripts/run_crt_wide32_r16p80_semantic_relation_bias_pilot.sh` to generate
  and train on an all-phase 80-ply semantic shard instead of the earlier
  active-tile-count 3-7 slice.
- `scripts/run_crt_wide32_r16p80_semantic_relation_bias_detached.sh` to launch,
  monitor, and fetch the all-phase run as a detached `john0` job so SSH
  disconnects do not kill long CPU generation.
- `scripts/run_crt_wide32_r16p80_semantic_residual_attention_pilot.sh` to train
  the MLP-anchored residual-attention Transformer on the all-phase p80 shard and
  replay it as a serving-shaped prefilter.
- `scripts/run_crt_wide32_r16p80_residual_seed_ensemble_eval.sh` to evaluate
  fixed residual-attention seed ensembles from per-root checkpoint replay rows
  on `john0`.
- `scripts/run_crt_wide32_r16p80x2_semantic_residual_attention_sweep.sh` to
  launch, monitor, stop, and fetch the larger/diverser all-phase
  residual-attention hardening run: 60 train seeds, 16 validation seeds,
  80 plies/seed, 32 actions/root, 16 rollout samples/action, three residual
  seeds, and fixed equal-weight seed ensembling.
- `scripts/run_crt_wide32_r16p80x2_mlp_member_ensemble_eval.sh` to replay
  checkpoint members from the p80x2 residual-attention checkpoints, export
  per-root rankings for `mlp` or `vanilla`, and run fixed seed ensembling
  without retraining.
- `scripts/run_crt_wide32_r16p80x2_semantic_vanilla_public_token_sweep.sh` to
  launch, monitor, stop, and fetch the first-class p80x2 semantic vanilla
  public-token seed sweep and fixed equal-weight seed ensemble.
- `scripts/run_crt_wide32_r16p80x2_vanilla_prefilter_game_pilot.sh` to launch,
  monitor, stop, and fetch the first interactive K16 vanilla-ensemble
  prefilter-search game pilot on `john0`.
- `scripts/run_crt_wide32_r16p80x2_vanilla_prefilter_game_nonshadow20.sh` to
  launch, monitor, stop, and fetch the 20-seed non-shadow K16
  vanilla-ensemble prefilter-search strength/speed run on `john0`.
- `scripts/run_crt_wide32_r16p20_prefilter_forensics.sh` to run held-out K=16
  miss-set forensics on `john0`.
- `scripts/run_crt_wide32_r16p20_source_union_prefilter.sh` to run
  train-selected source-quota union evaluation on `john0`.
- `scripts/run_crt_wide32_r16p20_learned_source_gate.sh` to run the strict
  train/tune/held-out learned source-gate evaluation on `john0`.
- `scripts/run_john0_gpu_smoke.sh` to sync this folder to `john0`, run CPU
  validation, GPU smoke, tiny overfit, synthetic replay training, optional
  real-root replay training, and pull reports/checkpoints back.

## Validation Command

```bash
PYTHONPATH=cascadiav3/src python3 -m unittest discover -s cascadiav3/tests -v
PYTHONPATH=cascadiav3/src python3 -m cascadiav3.validate --write-artifacts
./cascadiav3/scripts/generate_real_roots.sh
PYTHONPATH=cascadiav3/src python3 -m cascadiav3.torch_gpu_smoke
PYTHONPATH=cascadiav3/src python3 -m cascadiav3.torch_train_tiny --steps 300
PYTHONPATH=cascadiav3/src python3 -m cascadiav3.torch_train_replay --steps 400 --batch-size 2
PYTHONPATH=cascadiav3/src python3 -m cascadiav3.torch_train_replay \
  --replay cascadiav3/fixtures/real_roots.jsonl \
  --steps 500 \
  --batch-size 2 \
  --out cascadiav3/reports/real_replay_train.json \
  --checkpoint cascadiav3/checkpoints/real_replay_train.pt
./cascadiav3/scripts/run_crt_merit_pilot.sh
./cascadiav3/scripts/run_crt_public_token_pilot.sh
./cascadiav3/scripts/run_crt_relation_bias_pilot.sh
./cascadiav3/scripts/run_crt_sampled_teacher_relation_bias_pilot.sh
./cascadiav3/scripts/run_crt_scaled_sampled_teacher_relation_bias_pilot.sh
./cascadiav3/scripts/run_crt_wide32_sampled_teacher_relation_bias_pilot.sh
./cascadiav3/scripts/run_crt_wide32_r16x2_sampled_teacher_relation_bias_pilot.sh
./cascadiav3/scripts/run_crt_wide32_r16x2_semantic_relation_bias_pilot.sh
./cascadiav3/scripts/run_crt_wide32_r16p20_semantic_relation_bias_pilot.sh
./cascadiav3/scripts/run_crt_wide32_r16p20_semantic_retention_pilot.sh
./cascadiav3/scripts/run_crt_wide32_r16p20_semantic_cross_attention_pilot.sh
./cascadiav3/scripts/run_crt_wide32_r16p20_semantic_residual_attention_pilot.sh
./cascadiav3/scripts/run_crt_wide32_r16p20_semantic_action_set_pilot.sh
./cascadiav3/scripts/run_crt_wide32_r16p20_semantic_species_moe_pilot.sh
./cascadiav3/scripts/run_crt_wide32_r16p80_semantic_relation_bias_pilot.sh
./cascadiav3/scripts/run_crt_wide32_r16p80_semantic_residual_attention_pilot.sh
./cascadiav3/scripts/run_crt_wide32_r16p80_residual_seed_ensemble_eval.sh
bash cascadiav3/scripts/run_crt_wide32_r16p80x2_semantic_residual_attention_sweep.sh launch
bash cascadiav3/scripts/run_crt_wide32_r16p80x2_semantic_residual_attention_sweep.sh status
bash cascadiav3/scripts/run_crt_wide32_r16p80x2_semantic_residual_attention_sweep.sh stop
bash cascadiav3/scripts/run_crt_wide32_r16p80x2_semantic_residual_attention_sweep.sh fetch
bash cascadiav3/scripts/run_crt_wide32_r16p80x2_mlp_member_ensemble_eval.sh
CHECKPOINT_MEMBER=vanilla bash cascadiav3/scripts/run_crt_wide32_r16p80x2_mlp_member_ensemble_eval.sh
bash cascadiav3/scripts/run_crt_wide32_r16p80x2_semantic_vanilla_public_token_sweep.sh launch
bash cascadiav3/scripts/run_crt_wide32_r16p80x2_semantic_vanilla_public_token_sweep.sh status
bash cascadiav3/scripts/run_crt_wide32_r16p80x2_semantic_vanilla_public_token_sweep.sh stop
bash cascadiav3/scripts/run_crt_wide32_r16p80x2_semantic_vanilla_public_token_sweep.sh fetch
bash cascadiav3/scripts/run_crt_wide32_r16p80x2_vanilla_prefilter_game_pilot.sh launch
bash cascadiav3/scripts/run_crt_wide32_r16p80x2_vanilla_prefilter_game_pilot.sh status
bash cascadiav3/scripts/run_crt_wide32_r16p80x2_vanilla_prefilter_game_pilot.sh stop
bash cascadiav3/scripts/run_crt_wide32_r16p80x2_vanilla_prefilter_game_pilot.sh fetch
bash cascadiav3/scripts/run_crt_wide32_r16p80x2_vanilla_prefilter_game_nonshadow20.sh launch
bash cascadiav3/scripts/run_crt_wide32_r16p80x2_vanilla_prefilter_game_nonshadow20.sh status
bash cascadiav3/scripts/run_crt_wide32_r16p80x2_vanilla_prefilter_game_nonshadow20.sh stop
bash cascadiav3/scripts/run_crt_wide32_r16p80x2_vanilla_prefilter_game_nonshadow20.sh fetch
CORPUS_FORMAT=npz KEEP_JSONL=0 bash cascadiav3/scripts/run_greedy_policy_pretrain.sh launch
bash cascadiav3/scripts/run_greedy_policy_pretrain.sh status
bash cascadiav3/scripts/run_greedy_policy_pretrain.sh stop
bash cascadiav3/scripts/run_greedy_policy_pretrain.sh fetch
./cascadiav3/scripts/run_crt_wide32_r16p20_prefilter_forensics.sh
./cascadiav3/scripts/run_crt_wide32_r16p20_source_union_prefilter.sh
./cascadiav3/scripts/run_crt_wide32_r16p20_learned_source_gate.sh
```

## Generated Artifacts

The validation CLI writes:

- `fixtures/tiny_search_root.json`
- `fixtures/tiny_replay_manifest.json`
- `fixtures/tiny_replay.jsonl`
- `fixtures/tiny_replay_shard_manifest.json`
- `fixtures/real_roots.jsonl` after running `scripts/generate_real_roots.sh`.
- `fixtures/real_roots_manifest.json` after running `scripts/generate_real_roots.sh`.
- `reports/pre_gpu_validation.json`
- `reports/gpu_smoke.json` after running the Torch GPU smoke on a CUDA host.
- `reports/tiny_overfit.json` after running the one-root overfit smoke.
- `checkpoints/tiny_overfit.pt` after checkpoint round-trip verification.
- `reports/tiny_replay_train.json` after running replay-batch training.
- `checkpoints/tiny_replay_train.pt` after checkpoint round-trip verification.
- `reports/real_replay_train.json` after running replay-batch training on the
  real simulator dry-run shard.
- `checkpoints/real_replay_train.pt` after checkpoint round-trip verification
  for the real simulator dry-run shard.
- `fixtures/crt_merit_train.jsonl` and `fixtures/crt_merit_val.jsonl` after
  running `scripts/run_crt_merit_pilot.sh`.
- `reports/crt_merit_pilot.json` and `checkpoints/crt_merit_pilot.pt` after the
  default CRT-mini merit pilot.
- `reports/crt_merit_pilot_lr3e4.json` and
  `checkpoints/crt_merit_pilot_lr3e4.pt` after the lower-learning-rate retry.
- `fixtures/crt_token_merit_train.jsonl` and
  `fixtures/crt_token_merit_val.jsonl` after running
  `scripts/run_crt_public_token_pilot.sh`.
- `reports/crt_public_token_pilot.json` and
  `checkpoints/crt_public_token_pilot.pt` after the public-token merit pilot.
- `reports/crt_public_token_pilot_summary.md` as the human-readable summary of
  the public-token merit pilot.
- `reports/crt_relation_bias_pilot.json` and
  `checkpoints/crt_relation_bias_pilot.pt` after the relation-bias merit pilot.
- `reports/crt_relation_bias_pilot_summary.md` as the human-readable summary of
  the relation-bias merit pilot.
- `fixtures/crt_sampled_teacher_train.jsonl` and
  `fixtures/crt_sampled_teacher_val.jsonl` after running
  `scripts/run_crt_sampled_teacher_relation_bias_pilot.sh`.
- `reports/crt_sampled_teacher_relation_bias_pilot.json` and
  `checkpoints/crt_sampled_teacher_relation_bias_pilot.pt` after the
  sampled-teacher relation-bias merit pilot.
- `reports/crt_sampled_teacher_relation_bias_pilot_summary.md` as the
  human-readable summary of the first green-light v3 Transformer merit pilot.
- `fixtures/crt_scaled_sampled_teacher_train.jsonl` and
  `fixtures/crt_scaled_sampled_teacher_val.jsonl` after running
  `scripts/run_crt_scaled_sampled_teacher_relation_bias_pilot.sh`.
- `reports/crt_scaled_sampled_teacher_relation_bias_pilot.json` and
  `checkpoints/crt_scaled_sampled_teacher_relation_bias_pilot.pt` after the
  scaled sampled-teacher relation-bias merit pilot.
- `reports/crt_scaled_sampled_teacher_relation_bias_pilot_summary.md` as the
  human-readable summary of the scaled prefilter-oriented merit pilot.
- `fixtures/crt_wide32_sampled_teacher_train.jsonl` and
  `fixtures/crt_wide32_sampled_teacher_val.jsonl` after running
  `scripts/run_crt_wide32_sampled_teacher_relation_bias_pilot.sh`.
- `reports/crt_wide32_sampled_teacher_relation_bias_pilot.json` and
  `checkpoints/crt_wide32_sampled_teacher_relation_bias_pilot.pt` after the
  wide-32 sampled-teacher relation-bias merit pilot.
- `reports/crt_wide32_sampled_teacher_relation_bias_pilot_summary.md` as the
  human-readable summary of the wide-action prefilter-oriented merit pilot.
- `reports/crt_wide32_prefilter_eval.json` after replaying the wide-32
  checkpoint as a serving-shaped top-K prefilter.
- `reports/crt_wide32_prefilter_eval_roots.jsonl` with per-root ranked action
  ids and retained action sets for K=4/8/16/24/32.
- `reports/crt_wide32_prefilter_eval_summary.md` as the human-readable summary
  of the serving-shaped prefilter gate.
- `reports/crt_wide32_top16_margin_relation_bias_pilot.json` and
  `checkpoints/crt_wide32_top16_margin_relation_bias_pilot.pt` after the
  top-16-focused margin-loss relation-bias pilot.
- `reports/crt_wide32_top16_margin_prefilter_eval.json` and
  `reports/crt_wide32_top16_margin_prefilter_eval_roots.jsonl` after replaying
  the margin-loss checkpoint as a serving-shaped prefilter.
- `reports/crt_wide32_top16_margin_relation_bias_pilot_summary.md` as the
  human-readable summary of the margin-loss pilot.
- `fixtures/crt_wide32_r16_sampled_teacher_train.jsonl` and
  `fixtures/crt_wide32_r16_sampled_teacher_val.jsonl` after generating fresh
  16-rollout/action wide-32 sampled-teacher roots on `john0` CPU.
- `reports/crt_wide32_r16_sampled_teacher_relation_bias_pilot.json` and
  `checkpoints/crt_wide32_r16_sampled_teacher_relation_bias_pilot.pt` after the
  stronger-label relation-bias pilot.
- `reports/crt_wide32_r16_prefilter_eval.json` and
  `reports/crt_wide32_r16_prefilter_eval_roots.jsonl` after replaying the R16
  checkpoint as a serving-shaped prefilter.
- `reports/crt_wide32_r16_sampled_teacher_relation_bias_pilot_summary.md` as
  the human-readable summary of the R16 stronger-label pilot.
- `reports/crt_wide32_r16_prefilter_blend_eval.json` after train-selected
  relation/vanilla/MLP/immediate score blending on the R16 checkpoint.
- `reports/crt_wide32_r16_prefilter_blend_eval_summary.md` as the
  human-readable summary of the blend calibration experiment.
- `reports/crt_wide32_r16_top16_margin_relation_bias_pilot.json` and
  `checkpoints/crt_wide32_r16_top16_margin_relation_bias_pilot.pt` after
  rerunning the top-16 margin/listwise objective on the stronger R16 labels.
- `reports/crt_wide32_r16_top16_margin_prefilter_eval.json` and
  `reports/crt_wide32_r16_top16_margin_prefilter_eval_roots.jsonl` after
  replaying the R16 margin checkpoint as a serving-shaped prefilter.
- `reports/crt_wide32_r16_top16_margin_relation_bias_pilot_summary.md` as the
  human-readable summary of the R16 margin-loss pilot.
- `fixtures/crt_wide32_r16x2_sampled_teacher_train.jsonl` and
  `fixtures/crt_wide32_r16x2_sampled_teacher_val.jsonl` after generating fresh
  doubled 16-rollout/action wide-32 sampled-teacher roots on `john0` CPU.
- `reports/crt_wide32_r16x2_sampled_teacher_relation_bias_pilot.json` and
  `checkpoints/crt_wide32_r16x2_sampled_teacher_relation_bias_pilot.pt` after
  the doubled-data stronger-label relation-bias pilot.
- `reports/crt_wide32_r16x2_prefilter_eval.json` and
  `reports/crt_wide32_r16x2_prefilter_eval_roots.jsonl` after replaying the
  R16x2 checkpoint as a serving-shaped prefilter.
- `reports/crt_wide32_r16x2_sampled_teacher_relation_bias_pilot_summary.md` as
  the human-readable summary of the R16x2 doubled-data pilot.
- `reports/crt_wide32_r16x2_semantic_relation_bias_pilot.json` and
  `checkpoints/crt_wide32_r16x2_semantic_relation_bias_pilot.pt` after training
  the semantic relation-bias model on the R16x2 opening shard.
- `reports/crt_wide32_r16x2_semantic_prefilter_eval.json` and
  `reports/crt_wide32_r16x2_semantic_prefilter_eval_roots.jsonl` after
  replaying the R16x2 semantic checkpoint as a serving-shaped prefilter.
- `reports/crt_wide32_r16x2_semantic_relation_bias_pilot_summary.md` as the
  human-readable summary of the R16x2 semantic opening-shard pilot.
- `fixtures/crt_wide32_r16p20_semantic_train.jsonl` and
  `fixtures/crt_wide32_r16p20_semantic_val.jsonl` after generating the
  phase-diverse 20-ply semantic shard on `john0` CPU.
- `reports/crt_wide32_r16p20_semantic_relation_bias_pilot.json` and
  `checkpoints/crt_wide32_r16p20_semantic_relation_bias_pilot.pt` after
  training the semantic relation-bias model on the phase-diverse shard.
- `reports/crt_wide32_r16p20_semantic_prefilter_eval.json` and
  `reports/crt_wide32_r16p20_semantic_prefilter_eval_roots.jsonl` after
  replaying the phase-diverse semantic checkpoint as a serving-shaped prefilter.
- `reports/crt_wide32_r16p20_semantic_relation_bias_pilot_summary.md` as the
  human-readable summary of the phase-diverse semantic pilot.
- `fixtures/crt_wide32_r16p80_semantic_train.jsonl` and
  `fixtures/crt_wide32_r16p80_semantic_val.jsonl` after generating the
  all-phase 80-ply semantic shard on `john0` CPU.
- `reports/crt_wide32_r16p80_semantic_relation_bias_pilot.json` and
  `checkpoints/crt_wide32_r16p80_semantic_relation_bias_pilot.pt` after
  training the semantic relation-bias model on the all-phase shard.
- `reports/crt_wide32_r16p80_semantic_prefilter_eval.json` and
  `reports/crt_wide32_r16p80_semantic_prefilter_eval_roots.jsonl` after
  replaying the all-phase semantic checkpoint as a serving-shaped prefilter.
- `reports/crt_wide32_r16p80_semantic_relation_bias_pilot_summary.md` as the
  human-readable summary of the all-phase semantic pilot.
- `logs/r16p80_semantic_relation_bias_job.log` after the detached `john0`
  all-phase run completes.
- `reports/crt_wide32_r16p80_semantic_residual_attention_pilot.json`,
  `reports/crt_wide32_r16p80_semantic_residual_attention_seed31_pilot.json`,
  and `reports/crt_wide32_r16p80_semantic_residual_attention_seed32_pilot.json`
  after training three residual-attention seeds on the all-phase shard.
- `checkpoints/crt_wide32_r16p80_semantic_residual_attention_pilot.pt`,
  `checkpoints/crt_wide32_r16p80_semantic_residual_attention_seed31_pilot.pt`,
  and `checkpoints/crt_wide32_r16p80_semantic_residual_attention_seed32_pilot.pt`
  after checkpoint round-trip verification for the three residual-attention
  seeds.
- `reports/crt_wide32_r16p80_semantic_residual_attention_prefilter_eval.json`,
  `reports/crt_wide32_r16p80_semantic_residual_attention_seed31_prefilter_eval.json`,
  and `reports/crt_wide32_r16p80_semantic_residual_attention_seed32_prefilter_eval.json`
  after replaying each residual-attention seed as a serving-shaped prefilter.
- `reports/crt_wide32_r16p80_residual_seed_ensemble_3x_eval.json` and
  `reports/crt_wide32_r16p80_residual_seed_ensemble_3x_eval_roots.jsonl` after
  fixed equal-weight seed ensembling over the three residual-attention replay
  rows.
- `reports/crt_wide32_r16p80_residual_seed_ensemble_3x_eval_summary.md` and
  `reports/crt_wide32_r16p80_semantic_residual_attention_seed_sweep_summary.md`
  as human-readable summaries of the all-phase residual-attention seed sweep.
- `fixtures/crt_wide32_r16p80x2_semantic_train.jsonl` and
  `fixtures/crt_wide32_r16p80x2_semantic_val.jsonl` after generating the
  larger/diverser all-phase residual-attention hardening shard on `john0` CPU:
  4800 train roots and 1280 validation roots.
- `reports/crt_wide32_r16p80x2_semantic_residual_attention_seed20260640_pilot.json`,
  `reports/crt_wide32_r16p80x2_semantic_residual_attention_seed20260641_pilot.json`,
  and
  `reports/crt_wide32_r16p80x2_semantic_residual_attention_seed20260642_pilot.json`
  after training three residual-attention seeds on the p80x2 shard.
- `checkpoints/crt_wide32_r16p80x2_semantic_residual_attention_seed20260640_pilot.pt`,
  `checkpoints/crt_wide32_r16p80x2_semantic_residual_attention_seed20260641_pilot.pt`,
  and
  `checkpoints/crt_wide32_r16p80x2_semantic_residual_attention_seed20260642_pilot.pt`
  after checkpoint round-trip verification for the three p80x2 residual seeds.
- `reports/crt_wide32_r16p80x2_semantic_residual_attention_seed20260640_prefilter_eval.json`,
  `reports/crt_wide32_r16p80x2_semantic_residual_attention_seed20260641_prefilter_eval.json`,
  and
  `reports/crt_wide32_r16p80x2_semantic_residual_attention_seed20260642_prefilter_eval.json`
  after replaying each p80x2 residual-attention seed as a serving-shaped
  prefilter.
- `reports/crt_wide32_r16p80x2_residual_seed_ensemble_3x_eval.json` and
  `reports/crt_wide32_r16p80x2_residual_seed_ensemble_3x_eval_roots.jsonl`
  after fixed equal-weight seed ensembling over the three p80x2 replay rows.
- `reports/crt_wide32_r16p80x2_residual_seed_ensemble_3x_eval_summary.md` and
  `reports/crt_wide32_r16p80x2_semantic_residual_attention_seed_sweep_summary.md`
  as human-readable summaries of the p80x2 hardening sweep.
- `reports/crt_wide32_r16p80x2_semantic_mlp_member_seed20260640_prefilter_eval.json`,
  `reports/crt_wide32_r16p80x2_semantic_mlp_member_seed20260641_prefilter_eval.json`,
  and
  `reports/crt_wide32_r16p80x2_semantic_mlp_member_seed20260642_prefilter_eval.json`
  after replaying the same-run MLP member from each p80x2 checkpoint.
- `reports/crt_wide32_r16p80x2_mlp_seed_ensemble_3x_eval.json` and
  `reports/crt_wide32_r16p80x2_mlp_seed_ensemble_3x_eval_roots.jsonl` after
  fixed equal-weight ensembling of the three MLP-member replay rows.
- `reports/crt_wide32_r16p80x2_semantic_vanilla_member_seed20260640_prefilter_eval.json`,
  `reports/crt_wide32_r16p80x2_semantic_vanilla_member_seed20260641_prefilter_eval.json`,
  and
  `reports/crt_wide32_r16p80x2_semantic_vanilla_member_seed20260642_prefilter_eval.json`
  after replaying the same-run vanilla public-token Transformer member from
  each p80x2 checkpoint.
- `reports/crt_wide32_r16p80x2_vanilla_seed_ensemble_3x_eval.json` and
  `reports/crt_wide32_r16p80x2_vanilla_seed_ensemble_3x_eval_roots.jsonl` after
  fixed equal-weight ensembling of the three vanilla-member replay rows.
- `reports/crt_wide32_r16p80x2_mlp_seed_ensemble_3x_eval_summary.md` and
  `reports/crt_wide32_r16p80x2_vanilla_seed_ensemble_3x_eval_summary.md` as
  human-readable summaries of the checkpoint-member ensemble probes.
- `reports/crt_wide32_r16p80x2_semantic_vanilla_public_token_seed20260650_pilot.json`,
  `reports/crt_wide32_r16p80x2_semantic_vanilla_public_token_seed20260651_pilot.json`,
  and
  `reports/crt_wide32_r16p80x2_semantic_vanilla_public_token_seed20260652_pilot.json`
  after training three first-class vanilla public-token seeds on the p80x2
  shard.
- `checkpoints/crt_wide32_r16p80x2_semantic_vanilla_public_token_seed20260650_pilot.pt`,
  `checkpoints/crt_wide32_r16p80x2_semantic_vanilla_public_token_seed20260651_pilot.pt`,
  and
  `checkpoints/crt_wide32_r16p80x2_semantic_vanilla_public_token_seed20260652_pilot.pt`
  after checkpoint round-trip verification for the three first-class vanilla
  seeds.
- `reports/crt_wide32_r16p80x2_semantic_vanilla_public_token_seed20260650_prefilter_eval.json`,
  `reports/crt_wide32_r16p80x2_semantic_vanilla_public_token_seed20260651_prefilter_eval.json`,
  and
  `reports/crt_wide32_r16p80x2_semantic_vanilla_public_token_seed20260652_prefilter_eval.json`
  plus corresponding `_roots.jsonl` files after replaying each dedicated
  vanilla checkpoint as a serving-shaped prefilter.
- `reports/crt_wide32_r16p80x2_vanilla_public_token_seed_ensemble_3x_eval.json`
  and
  `reports/crt_wide32_r16p80x2_vanilla_public_token_seed_ensemble_3x_eval_roots.jsonl`
  after fixed equal-weight ensembling of the three first-class vanilla replay
  rows.
- `reports/crt_wide32_r16p80x2_vanilla_public_token_seed_ensemble_3x_eval_summary.md`
  and
  `reports/crt_wide32_r16p80x2_semantic_vanilla_public_token_seed_sweep_summary.md`
  as human-readable summaries of the first-class vanilla sweep.
- `logs/r16p80x2_vanilla_prefilter_game_pilot_job.log` after the detached
  interactive K16 prefilter-search game pilot completes on `john0`.
- `reports/crt_wide32_r16p80x2_vanilla_prefilter_game_pilot.json`,
  `reports/crt_wide32_r16p80x2_vanilla_prefilter_game_pilot_decisions.jsonl`,
  and
  `reports/crt_wide32_r16p80x2_vanilla_prefilter_game_pilot_summary.md` after
  running the first interactive vanilla-ensemble prefilter-search pilot.
- `logs/r16p80x2_vanilla_prefilter_game_nonshadow20_job.log` after the
  detached 20-seed non-shadow K16 prefilter-search run completes on `john0`.
- `reports/crt_wide32_r16p80x2_vanilla_prefilter_game_nonshadow20.json`,
  `reports/crt_wide32_r16p80x2_vanilla_prefilter_game_nonshadow20_decisions.jsonl`,
  and
  `reports/crt_wide32_r16p80x2_vanilla_prefilter_game_nonshadow20_summary.md`
  after running the 20-seed non-shadow vanilla-ensemble prefilter-search
  strength/speed test.
- `reports/crt_wide32_r16p20_semantic_retention_pilot.json` and
  `checkpoints/crt_wide32_r16p20_semantic_retention_pilot.pt` after training the
  phase-diverse semantic relation-bias model with direct top-K retention loss.
- `reports/crt_wide32_r16p20_semantic_retention_prefilter_eval.json` and
  `reports/crt_wide32_r16p20_semantic_retention_prefilter_eval_roots.jsonl`
  after replaying the retention checkpoint as a serving-shaped prefilter.
- `reports/crt_wide32_r16p20_semantic_retention_pilot_summary.md` as the
  human-readable summary of the top-K retention pilot.
- `reports/crt_wide32_r16p20_semantic_cross_attention_pilot.json` and
  `checkpoints/crt_wide32_r16p20_semantic_cross_attention_pilot.pt` after
  training the semantic cross-attention action-query model.
- `reports/crt_wide32_r16p20_semantic_cross_attention_prefilter_eval.json` and
  `reports/crt_wide32_r16p20_semantic_cross_attention_prefilter_eval_roots.jsonl`
  after replaying the cross-attention checkpoint as a serving-shaped prefilter.
- `reports/crt_wide32_r16p20_semantic_cross_attention_pilot_summary.md` as the
  human-readable summary of the cross-attention pilot.
- `reports/crt_wide32_r16p20_semantic_residual_attention_pilot.json` and
  `checkpoints/crt_wide32_r16p20_semantic_residual_attention_pilot.pt` after
  training the semantic residual-attention action-query model.
- `reports/crt_wide32_r16p20_semantic_residual_attention_prefilter_eval.json`
  and
  `reports/crt_wide32_r16p20_semantic_residual_attention_prefilter_eval_roots.jsonl`
  after replaying the residual-attention checkpoint as a serving-shaped
  prefilter.
- `reports/crt_wide32_r16p20_semantic_residual_attention_pilot_summary.md` as
  the human-readable summary of the residual-attention pilot.
- `reports/crt_wide32_r16p20_semantic_residual_attention_blend_eval.json` after
  train-selected residual/vanilla/MLP/immediate score blending on the
  phase-diverse residual-attention checkpoint.
- `reports/crt_wide32_r16p20_semantic_residual_attention_blend_eval_summary.md`
  as the human-readable summary of the residual-attention blend calibration
  experiment.
- `reports/crt_wide32_r16p20_semantic_action_set_pilot.json` and
  `checkpoints/crt_wide32_r16p20_semantic_action_set_pilot.pt` after training
  the semantic action-set Transformer.
- `reports/crt_wide32_r16p20_semantic_action_set_prefilter_eval.json` and
  `reports/crt_wide32_r16p20_semantic_action_set_prefilter_eval_roots.jsonl`
  after replaying the action-set checkpoint as a serving-shaped prefilter.
- `reports/crt_wide32_r16p20_semantic_action_set_pilot_summary.md` as the
  human-readable summary of the action-set pilot.
- `reports/crt_wide32_r16p20_semantic_action_set_blend_eval.json` after
  train-selected action-set/vanilla/MLP/immediate score blending on the
  phase-diverse action-set checkpoint.
- `reports/crt_wide32_r16p20_semantic_action_set_blend_eval_summary.md` as the
  human-readable summary of the action-set blend calibration experiment.
- `reports/crt_wide32_r16p20_semantic_species_moe_pilot.json` and
  `checkpoints/crt_wide32_r16p20_semantic_species_moe_pilot.pt` after training
  the semantic species-MoE Transformer with the standard objective.
- `reports/crt_wide32_r16p20_semantic_species_moe_pilot_summary.md` as the
  human-readable summary of the standard species-MoE pilot.
- `reports/crt_wide32_r16p20_semantic_species_moe_retention_pilot.json` and
  `checkpoints/crt_wide32_r16p20_semantic_species_moe_retention_pilot.pt` after
  training the semantic species-MoE Transformer with direct top-K retention
  supervision.
- `reports/crt_wide32_r16p20_semantic_species_moe_retention_pilot_summary.md`
  as the human-readable summary of the retention species-MoE pilot.
- `reports/crt_wide32_r16p20_semantic_prefilter_forensics.json` after replaying
  the action-set checkpoint sources and explaining K=16 miss sets.
- `reports/crt_wide32_r16p20_semantic_prefilter_forensics_summary.md` as the
  human-readable summary of the miss-set forensics.
- `reports/crt_wide32_r16p20_semantic_source_union_prefilter.json` after
  train-selected source-quota union evaluation.
- `reports/crt_wide32_r16p20_semantic_source_union_prefilter_summary.md` as the
  human-readable summary of the source-union prefilter experiment.
- `reports/crt_wide32_r16p20_semantic_learned_source_gate.json` after the
  train/tune/held-out learned source-gate evaluation over action_set, vanilla,
  MLP, and immediate score/rank features.
- `reports/crt_wide32_r16p20_semantic_learned_source_gate_summary.md` as the
  human-readable summary of the learned source-gate experiment.

These are dry-run CPU/GPU artifacts. They are not training evidence, model
strength evidence, or promotion evidence.

## Latest Serving Prefilter Gate

Gate: smallest K with teacher-best recall >= 0.750 and mean oracle regret
<= 0.250 sampled-teacher points.

Latest hardening run:
`crt-wide32-r16p80x2-semantic-vanilla-public-token-seed-ensemble-3x-v1`.

- Runner:
  `scripts/run_crt_wide32_r16p80x2_semantic_vanilla_public_token_sweep.sh`.
- Inputs:
  `reports/crt_wide32_r16p80x2_semantic_vanilla_public_token_seed20260650_prefilter_eval_roots.jsonl`,
  `reports/crt_wide32_r16p80x2_semantic_vanilla_public_token_seed20260651_prefilter_eval_roots.jsonl`,
  and
  `reports/crt_wide32_r16p80x2_semantic_vanilla_public_token_seed20260652_prefilter_eval_roots.jsonl`.
- Serving eval:
  `reports/crt_wide32_r16p80x2_vanilla_public_token_seed_ensemble_3x_eval.json`.
- Validation set: 1280 roots, 32 retained actions/root,
  `fixtures/crt_wide32_r16p80x2_semantic_val.jsonl`.
- Phase coverage: active tile counts 3-22, 64 validation roots per active tile
  count, plies 0-79 represented.
- Residual seed 20260640 K=16: recall 0.7312, oracle regret 0.1098; fail by
  recall.
- Residual seed 20260641 K=16: recall 0.7453, oracle regret 0.1064; fail by
  recall.
- Residual seed 20260642 K=16: recall 0.7383, oracle regret 0.1149; fail by
  recall.
- Same-run semantic MLP K=16 recalls: 0.7406, 0.7563, and 0.7469.
- Fixed 3-seed residual ensemble K=16: recall 0.7367, oracle regret 0.1084;
  fail by recall.
- Fixed 3-seed residual ensemble K=24: recall 0.8945, oracle regret 0.0328;
  pass.
- Fixed 3-seed MLP-member ensemble K=16: recall 0.7461, oracle regret 0.1048;
  fail by recall.
- Fixed 3-seed MLP-member ensemble K=24: recall 0.9000, oracle regret 0.0337;
  pass.
- Vanilla member single-seed K=16 recalls: 0.7508, 0.7328, and 0.7516.
- Fixed 3-seed vanilla public-token Transformer member ensemble K=16: recall
  0.7570, oracle regret 0.1125; pass.
- Fixed 3-seed vanilla public-token Transformer member ensemble K=24: recall
  0.9062, oracle regret 0.0321; pass.
- Dedicated vanilla seed 20260650 K=16: recall 0.7367, oracle regret 0.1216;
  K=24 pass.
- Dedicated vanilla seed 20260651 K=16: recall 0.7375, oracle regret 0.1297;
  K=24 pass.
- Dedicated vanilla seed 20260652 K=16: recall 0.7656, oracle regret 0.1181;
  K=16 pass.
- Fixed 3-seed first-class vanilla public-token ensemble K=16: recall 0.7672,
  oracle regret 0.1146; pass.
- Fixed 3-seed first-class vanilla public-token ensemble K=24: recall 0.8992,
  oracle regret 0.0368; pass.
- `has_merit=false` remains true for all single residual checkpoints under the
  scalar/top-1 offline merit gate.
- Decision: the p80 K=16 residual-attention pass does not survive the p80x2
  hardening gate, and MLP member ensembling also misses K=16. The dedicated
  vanilla public-token Transformer family does survive the strict p80x2 K=16
  gate and slightly improves K=16 recall over the earlier side-member ensemble
  (`0.7672` vs `0.7570`). The next experiment should be a real search-prefilter
  pilot from the first-class vanilla family, with gameplay score impact judged
  separately from dry-run sampled-teacher recall.
- Important correction: the earlier p20 shard was not all-phase; it covered
  active tile counts 3-7. The p80 shard is the first equal-coverage all-phase
  diagnostic for this semantic recipe.

## Latest Interactive Search Prefilter Pilot

Latest gameplay-shaped integration run:
`crt-wide32-r16p80x2-vanilla-prefilter-game-nonshadow20-v1`.

- Runner:
  `scripts/run_crt_wide32_r16p80x2_vanilla_prefilter_game_nonshadow20.sh`.
- Controller:
  `src/cascadiav3/torch_prefilter_game_pilot.py`.
- Rust bridge:
  `real-root-exporter --interactive-policy-game`.
- Report:
  `reports/crt_wide32_r16p80x2_vanilla_prefilter_game_nonshadow20.json`.
- Per-decision rows:
  `reports/crt_wide32_r16p80x2_vanilla_prefilter_game_nonshadow20_decisions.jsonl`.
- Summary:
  `reports/crt_wide32_r16p80x2_vanilla_prefilter_game_nonshadow20_summary.md`.
- Detached log:
  `logs/r16p80x2_vanilla_prefilter_game_nonshadow20_job.log`.
- Seeds: 2026171000 through 2026171019.
- Game shape: 4-player Card A/no-bonus `research_aaaaa`, 80 decisions/game.
- Candidate surface: 32 greedy-ranked actions/root.
- Prefilter: fixed equal-weight ensemble of first-class vanilla public-token
  seeds 20260650, 20260651, and 20260652, retaining K=16.
- Downstream search: 16 rollout samples/action, top-4 sampled greedy
  continuations.
- Shadow mode: disabled, so prefilter-search measures real retained-set search
  latency but does not measure missed full-search winners.
- Paired full-search baseline: enabled on the same 20 seeds with all 32
  candidates retained; full-search baseline games ran with 4 CPU workers.
- Prefilter-search mean seat score: 95.4625.
- Full-search mean seat score: 96.3500.
- Mean paired delta, prefilter minus full search: -0.8875.
- Median paired delta: -1.25; min/max paired delta: -4.50 / +2.75.
- Prefilter-search p50/p90 seat score: 96.0 / 99.0.
- Full-search p50/p90 seat score: 97.0 / 100.0.
- Decisions: 1600 prefilter-search and 1600 full-search; 3200 per-decision
  rows fetched locally.
- Mean model scoring latency: 0.01237 seconds/decision on the RTX 5090.
- Mean total decision seconds: 2.3558 for K16 prefilter-search versus 4.4617
  for full-32 search.
- Measured speedup: 1.8939x; measured time reduction: 47.20%.
- Remote verification: exporter tests passed, release build succeeded, RTX
  5090 was visible to Torch (`2.11.0+cu128`, CUDA 12.8, driver 591.86), all 22
  Torch-enabled unit/schema tests passed on `john0`, artifacts fetched locally,
  and no simulator/controller process remained after completion.

Decision: the interactive Python/Torch/Rust bridge is healthy and K16 delivers
the expected compute reduction, but this K16 vanilla-ensemble filter is too
lossy for promotion: the larger non-shadow paired run lost 0.8875 mean seat
score versus full-32 search. The next evidence branch should test a wider K24
prefilter or train a stronger retention/search-aware model; do not treat the
4-seed shadow pilot's +1.0 as strength evidence.

Prior safety pilot:
`crt-wide32-r16p80x2-vanilla-prefilter-game-pilot-v1` used 4 paired seeds with
shadow full search enabled. It scored 96.0625 versus 95.0625 for full search,
retained the full-search winner on 77.8125% of decisions, had mean shadow regret
0.1142578125, and estimated 50% non-shadow rollout savings. That run proved the
bridge and provided decision-safety telemetry, but the non-shadow20 result above
is the current gameplay-shaped decision record.

Older gate history:

All-phase semantic relation-bias
`crt-wide32-r16p80-semantic-relation-bias-v1`:

- Semantic relation-bias K=16: recall 0.7125, oracle regret 0.1482; fail by
  recall.
- Semantic relation-bias K=24: recall 0.8563, oracle regret 0.0623; pass.
- Same-run vanilla Transformer K=16: recall 0.7000, oracle regret 0.1598.
- Same-run token-pooled MLP K=16: recall 0.7609, oracle regret 0.0995; pass.
- Immediate-score K=16: recall 0.6562, oracle regret 0.2164.
- Decision: relation-bias does not carry the all-phase K=16 signal; the
  MLP-anchored residual-attention branch above is the current transformer-family
  K=16 direction.

R16p20 semantic action-set and post-hoc combiners:

- Action-set Transformer K=16: recall 0.7133, oracle regret 0.1628; fail by
  recall.
- Action-set Transformer K=24: recall 0.8867, oracle regret 0.0583; pass.
- Same-run vanilla Transformer K=16: recall 0.7300, oracle regret 0.1590.
- Same-run token-pooled MLP K=16: recall 0.7417, oracle regret 0.1456.
- Blend validation K=16: recall 0.7350, oracle regret 0.1489; fail by recall.
- Source-union selected quotas on train: action_set 15, vanilla 1, MLP 0,
  immediate 0; train K=16 recall 0.8004, validation K=16 recall 0.7233.
- Learned gate selected by an inner train/tune split: fit K=16 recall 0.7948,
  tune K=16 recall 0.8229, held-out validation K=16 recall 0.7250 and oracle
  regret 0.1644; fail by recall and worse than the same-run MLP.
- Standard species-MoE: K=16 recall 0.7033, oracle regret 0.1715; fail by
  recall and below same-run MLP K=16 recall 0.7367.
- Retention species-MoE: K=16 recall 0.7133, oracle regret 0.1610; fail by
  recall and below same-run vanilla K=16 recall 0.7417.

Direct-retention follow-up
`crt-wide32-r16p20-semantic-topk-retention-v1` trained the teacher top-16 action
set against the bottom-16 set:

- Relation-bias K=16: recall 0.6967, oracle regret 0.1864; fail by recall.
- Relation-bias K=24: recall 0.8883, oracle regret 0.0493; pass.
- Same-run token-pooled MLP K=16: recall 0.7417, oracle regret 0.1469; near
  miss.
- Decision: retention loss helps slightly but does not validate the transformer
  at K=16.

Phase-diverse semantic relation-bias
`crt-wide32-r16p20-semantic-relation-bias-v1`:

- Relation-bias K=16: recall 0.6867, oracle regret 0.1825; fail by recall.
- Relation-bias K=24: recall 0.8617, oracle regret 0.0688; pass.
- Same-run vanilla Transformer K=16: recall 0.7300, oracle regret 0.1556.
- Same-run token-pooled MLP K=16: recall 0.7150, oracle regret 0.1615.
- Decision: K=24 remains the smallest validated serving width on deeper roots.

Opening-shard semantic follow-up `crt-wide32-r16x2-semantic-relation-bias-v1`
added 28 action-conditioned semantic features and was the first checkpoint to
clear the K=16 gate on the doubled R16 opening shard:

- Validation set: 600 roots, 32 retained actions/root, 4 plies/seed.
- Relation-bias K=16: recall 0.7767, oracle regret 0.1233; pass.
- Relation-bias K=24: recall 0.9433, oracle regret 0.0304; pass.
- Decision: good diagnostic signal, but opening-shard only. The 20-ply
  follow-up above is more representative of midgame candidate filtering.

Standard relation-bias `crt-wide32-r16x2-sampled-teacher-relation-bias-v1`
without semantic features:

- Validation set: 600 roots, 32 retained actions/root, 4 plies/seed.
- Relation-bias K=16: recall 0.7400, oracle regret 0.1475; fail by recall.
- Relation-bias K=24: recall 0.9083, oracle regret 0.0448; pass.
- `has_merit=true` versus immediate, MLP, and same-run vanilla point/regret
  baselines.

Follow-up `crt-wide32-top16-margin-relation-bias-v1` added a weighted
teacher-best pairwise margin and sharper policy loss. It improved point
selection but did not clear the top-16 prefilter gate:

- Top-1 agreement: 0.1000, mean regret 2.2108.
- K=16: recall 0.6833, oracle regret 0.2704; fail.
- K=24: recall 0.8633, oracle regret 0.0800; pass.
- Decision: stronger labels are more likely to help than more shaping on the
  same 8-rollout/action targets.

Stronger-label follow-up `crt-wide32-r16-sampled-teacher-relation-bias-v1`
generated fresh 16-rollout/action wide-32 roots on `john0` CPU and trained the
standard relation-bias recipe:

- Top-1 agreement: 0.1333, mean regret 1.5683.
- K=16: recall 0.7233, oracle regret 0.1552; fail by recall only.
- K=24: recall 0.9167, oracle regret 0.0381; pass.

Train-selected normalized score blending over relation/vanilla/MLP/immediate
passed K=16 on train but failed validation:

- Selected weights: relation 0.5, vanilla 0.3, MLP 0.0, immediate 0.2.
- Train K=16: recall 0.7983, oracle regret 0.1045.
- Validation K=16: recall 0.7133, oracle regret 0.1471.
- Decision: blending is not enough; the remaining top-16 blocker is held-out
  recall.

Rerunning the top-16 margin/listwise objective on the stronger R16 labels also
regressed the prefilter:

- Top-1 agreement: 0.1267, mean regret 1.6456.
- K=16: recall 0.6633, oracle regret 0.2110; fail.
- K=24: recall 0.8567, oracle regret 0.0681; pass, but worse than R16
  standard.
- Decision: the standard R16 objective remains the best tested relation-bias
  prefilter; the next credible path is more diverse data or better structure.

This is still dry-run sampled-teacher evidence, not gameplay strength evidence.

## Latest Verified GPU Run

Ran `scripts/run_john0_gpu_smoke.sh` on `john0` with the RTX 5090 visible to
Torch (`torch 2.11.0+cu128`, CUDA 12.8, driver 591.86).

- Unit/schema suite: 10 tests passed.
- Tiny CUDA forward/backward smoke: pass.
- Tiny one-root overfit: loss 8657.8721 -> 1569.5027, checkpoint round-trip
  pass.
- Synthetic replay train: 2 records, action counts `[2, 3]`, loss 8704.8389
  -> 805.0240, checkpoint round-trip pass.
- Real simulator replay train: 4 records, action counts `[8, 8, 8, 8]`, loss
  7903.4102 -> 294.3968, checkpoint round-trip pass.

## CRT-Mini Merit Pilot

Ran `scripts/run_crt_merit_pilot.sh` on 400 train roots and 100 validation roots
with 16 retained legal actions per root. Labels are still
`canonical_simulator_greedy_rollout_dry_run`; this is a plumbing and first-signal
test, not K32/R600 teacher evidence.

Default run (`lr=1e-3`, 1200 steps):

- immediate-score baseline: top-1 0.07, top-4 0.25, mean regret 4.85.
- MLP baseline: top-1 0.08, top-4 0.27, mean regret 4.50.
- CRT-mini transformer: top-1 0.10, top-4 0.26, mean regret 5.14.
- Decision: `has_merit=false`; checkpoint round-trip pass.

Lower-learning-rate retry (`lr=3e-4`, 3000 steps):

- immediate-score baseline: top-1 0.07, top-4 0.25, mean regret 4.85.
- MLP baseline: top-1 0.08, top-4 0.29, mean regret 4.66.
- CRT-mini transformer: top-1 0.07, top-4 0.21, mean regret 5.15.
- Decision: `has_merit=false`; checkpoint round-trip pass.

Interpretation: the current scalar state/action features and greedy-rollout
dry-run labels do not justify launching full expert iteration. The next serious
step should be the real Rust-owned tokenization/C-GAB bridge and/or stronger
teacher labels, not scaling this scalar pilot.

## Public-Token Merit Pilot

Ran `scripts/run_crt_public_token_pilot.sh` on `john0`. The script builds and
tests the Rust exporter remotely, generates 400 train roots and 100 validation
roots on `john0` CPU, runs unit tests inside the Torch venv, trains on the RTX
5090, and pulls artifacts back.

Run config:

- token feature dimension: 41.
- action feature dimension: 33.
- transformer: 3 layers, hidden size 160, 5 heads, 632802 parameters.
- MLP baseline: token-pooled public features, 115362 parameters.
- Torch: 2.11.0+cu128, CUDA 12.8, driver 591.86.

Result:

- immediate-score baseline: top-1 0.13, top-4 0.27, mean regret 4.93.
- token-pooled MLP: top-1 0.08, top-4 0.33, mean regret 4.77.
- public-token transformer: top-1 0.08, top-4 0.32, mean regret 4.73.
- Decision: `has_merit=false`; regret improved 4.1% versus immediate score but
  top-1 was 5pp worse than immediate score. Checkpoint round-trip passed.

Interpretation: the public-token/export/training path is healthy, but this
dry-run does not justify full expert iteration. The next credible transformer
test needs stronger teacher labels and true relation-aware attention bias, not
only scalarized relation-degree features.

## Relation-Bias Merit Pilot

Ran `scripts/run_crt_relation_bias_pilot.sh` on `john0`. The script builds and
tests the Rust exporter remotely, verifies Python contracts, reuses the
`john0`-generated public-token roots unless regeneration is requested, trains on
the RTX 5090, and pulls artifacts back.

Run config:

- relation vocabulary size: 9.
- token feature dimension: 41.
- action feature dimension: 33.
- relation-bias transformer: 3 layers, hidden size 160, 5 heads, 632937
  parameters.
- same-run vanilla public-token transformer: 632802 parameters.
- same-run token-pooled MLP baseline: 115362 parameters.

Result:

- immediate-score baseline: top-1 0.13, top-4 0.27, mean regret 4.93.
- token-pooled MLP: top-1 0.10, top-4 0.29, mean regret 4.92.
- vanilla public-token transformer: top-1 0.12, top-4 0.30, mean regret 4.93.
- relation-bias transformer: top-1 0.10, top-4 0.33, mean regret 4.69.
- Decision: `has_merit=false`; regret improved 4.9% versus immediate score and
  same-run vanilla transformer, but top-1 was 3pp worse than immediate score.
  Checkpoint round-trip passed.

Interpretation: learned relation bias is directionally useful, but the
greedy-rollout dry-run labels are now the likely bottleneck. The next credible
iteration should improve teacher target quality before scaling model size.

## Sampled-Teacher Relation-Bias Merit Pilot

Ran `scripts/run_crt_sampled_teacher_relation_bias_pilot.sh` on `john0`. The
script builds/tests the Rust exporter remotely, generates sampled teacher roots
on `john0` CPU with per-seed parallelism, verifies Python contracts, trains on
the RTX 5090, and pulls artifacts back.

Run config:

- train roots: 400.
- validation roots: 100.
- retained legal actions per root: 16.
- rollout samples per retained action: 4.
- continuation policy: sampled uniformly from the top-4 greedy-ranked actions.
- relation-bias transformer: 3 layers, hidden size 160, 5 heads, 632937
  parameters.

Result:

- immediate-score baseline: top-1 0.05, top-4 0.26, mean regret 3.2800.
- token-pooled MLP: top-1 0.06, top-4 0.35, mean regret 2.7400.
- vanilla public-token transformer: top-1 0.09, top-4 0.35, mean regret 2.7525.
- relation-bias transformer: top-1 0.12, top-4 0.36, mean regret 2.5975.
- Decision: `has_merit=true`; regret improved 20.8% and top-1 improved 7pp
  versus immediate score, and regret improved 5.6% versus same-run vanilla
  transformer. Checkpoint round-trip passed.

Interpretation: this is the first v3 Transformer green light. It is still
offline teacher-ranking merit rather than gameplay strength, but it justifies
scaling the sampled-teacher/relation-bias loop and testing top-K retention for a
future transformer prefilter.

## Scaled Sampled-Teacher Relation-Bias Merit Pilot

Ran `scripts/run_crt_scaled_sampled_teacher_relation_bias_pilot.sh` on `john0`.
The script builds/tests the Rust exporter remotely, generates scaled sampled
teacher roots on `john0` CPU with per-seed parallelism, verifies Python
contracts, trains on the RTX 5090, and pulls artifacts back.

Run config:

- train roots: 1600.
- validation roots: 400.
- retained legal actions per root: 16.
- rollout samples per retained action: 8.
- continuation policy: sampled uniformly from the top-4 greedy-ranked actions.
- relation-bias transformer: 4 layers, hidden size 256, 8 heads, 2129698
  parameters.
- train rollout samples: 204800; validation rollout samples: 51200.
- truncated rollout samples: 2 train, 0 validation.

Result:

- immediate-score baseline: top-1 0.0800, top-4 0.3025, top-8 0.5300,
  mean regret 2.2441.
- token-pooled MLP: top-1 0.1325, top-4 0.3750, top-8 0.6175,
  mean regret 2.0000.
- vanilla public-token transformer: top-1 0.1325, top-4 0.3975, top-8 0.6100,
  mean regret 1.9831.
- relation-bias transformer: top-1 0.1675, top-4 0.4050, top-8 0.6500,
  mean regret 1.7838.
- Decision: `has_merit=true`; regret improved 20.5% and top-1 improved 8.75pp
  versus immediate score, and regret improved 10.1% versus same-run vanilla
  transformer. Checkpoint round-trip passed.

Prefilter interpretation:

- Relation-bias top-4 oracle regret: 0.7200.
- Relation-bias top-8 oracle regret: 0.2963.
- Relation-bias top-8 recall: 0.6500 over a retained 16-action candidate set.

Interpretation: scaling strengthened the offline result and made relation-bias
the best tested top-8 prefilter. The signal is promising but not yet narrow
enough for production prefiltering. Next work should test stronger teacher
labels and/or larger retained action sets before gameplay claims.

## Wide-32 Sampled-Teacher Relation-Bias Merit Pilot

Ran `scripts/run_crt_wide32_sampled_teacher_relation_bias_pilot.sh` on `john0`.
The script builds/tests the Rust exporter remotely, generates sampled teacher
roots with 32 retained actions per root on `john0` CPU, verifies Python
contracts, trains on the RTX 5090, and pulls artifacts back.

Run config:

- train roots: 1200.
- validation roots: 300.
- retained legal actions per root: 32.
- rollout samples per retained action: 8.
- continuation policy: sampled uniformly from the top-4 greedy-ranked actions.
- relation-bias transformer: 4 layers, hidden size 256, 8 heads, 2129698
  parameters.
- train rollout samples: 307200; validation rollout samples: 76800.
- truncated rollout samples: 1 train, 0 validation.

Result:

- immediate-score baseline: top-1 0.0500, top-4 0.2133, top-8 0.3233,
  top-16 0.6100, mean regret 2.5129.
- token-pooled MLP: top-1 0.0633, top-4 0.2200, top-8 0.4000,
  top-16 0.6400, mean regret 2.3813.
- vanilla public-token transformer: top-1 0.0800, top-4 0.2667,
  top-8 0.4200, top-16 0.6433, mean regret 2.3421.
- relation-bias transformer: top-1 0.0833, top-4 0.2767, top-8 0.4333,
  top-16 0.6767, mean regret 2.2408.
- Decision: `has_merit=true`; regret improved 10.8% versus immediate score and
  4.3% versus same-run vanilla transformer. Checkpoint round-trip passed.

Prefilter interpretation:

- Relation-bias top-8 recall: 0.4333, top-8 oracle regret: 0.5658.
- Relation-bias top-16 recall: 0.6767, top-16 oracle regret: 0.2542.

Interpretation: relation-bias remains the best tested architecture on the
wider candidate surface, but top-8 is too narrow over 32 retained actions.
Top-16/top-24 is the more plausible serving prefilter target unless stronger
teacher labels sharpen top-8 retention.

## Still Before Full Transformer Training

- Full C-GAB construction over full token sets.
- Full CRT-S backend beyond the scalar and public-token mini transformer pilots.
- Search teacher implementation beyond greedy-rollout dry runs.
