# Cascadia v3 Formal Plan Package

Status: formal plan package plus isolated v3 implementation scaffold. This
directory contains the approved planning artifacts for a clean-slate
transformer-based Cascadia v3 effort, plus executable contracts, real simulator
root export, public-token/relation-bias pilots, and `john0` remote GPU runners.

Source artifact:
`/Users/johnherrick/cascadia/docs/v3/LITERATURE_FIRST_TRANSFORMER_GAME_AI_PROPOSAL_2026-06-29.md`

Authority rule: the literature-first proposal is the governing source for this
package. Existing Cascadia repo experiment logs, NNUE campaigns, and radius-7
material are background context only and do not override this transformer plan.

Canonical board fast path: radius 6 hex boards are the default implementation
target for tokenization and schema planning. A radius 6 board contains 127 cells
by `1 + 3r(r + 1)`. Legal states outside that radius are represented exactly by
overflow entities; they are not dropped, clipped, or silently projected. Any
future radius expansion requires a CPU coverage census first.

GPU work has started only as dry-run plumbing, offline merit testing, and
serving-shaped prefilter replay on `john0`. These reports are not gameplay
strength or promotion evidence. The semantic relation-bias run
`crt-wide32-r16x2-semantic-relation-bias-v1` cleared the held-out K=16 serving
gate on the doubled R16 opening shard, but the later p20 shard only covered
active tile counts 3-7. The all-phase p80 follow-up
`crt-wide32-r16p80-semantic-relation-bias-v1` covers active tile counts 3-22 and
did not validate relation-bias at K=16: recall was `0.7125`. The next
MLP-anchored residual-attention branch produced the first all-phase
transformer-family K=16 prefilter pass. Single seeds are still borderline
(`0.7531`, `0.7484`, `0.7500` recall), but the fixed 3-seed residual ensemble
reaches K=16 recall `0.7578` with oracle regret `0.0999`. This is a serving
prefilter signal, not gameplay strength or scalar policy merit. The p80x2
hardening run doubled the all-phase train/validation seed counts and did not
hold K=16 for residual attention: the fixed 3-seed residual ensemble fell to
recall `0.7367` with oracle regret `0.1084`, while K=24 remained safe at recall
`0.8945` and regret `0.0328`. Same-checkpoint MLP member ensembling also missed
K=16 at recall `0.7461`. The simpler same-run vanilla public-token Transformer
member ensemble then passed K=16 at recall `0.7570` with oracle regret
`0.1125`. Isolating that recipe as a first-class checkpoint family also passed:
the dedicated fixed 3-seed vanilla public-token ensemble reaches K=16 recall
`0.7672` with oracle regret `0.1146`, recommended K=16. The first interactive
search-prefilter pilot from that dedicated family now runs complete Rust-owned
games through a Python/Torch policy bridge: on 4 paired seeds, K16 prefilter
search scored `96.0625` mean per seat versus `95.0625` for the full-32 sampled
search baseline, retained the shadow full-search winner on `77.8125%` of
decisions, and estimated `50%` non-shadow rollout savings. The larger
non-shadow 20-seed follow-up measured the actual speed/strength tradeoff:
K16 prefilter-search scored `95.4625` mean per seat versus `96.3500` for
full-32 search, with real mean decision time `2.3558s` versus `4.4617s`
(`1.89x` speedup, `47.2%` time reduction). The bridge is healthy, but K16 is
too lossy on this evidence; the next credible branch is K24 or a stronger
retention/search-aware model rather than promoting K16.

Greedy behavior-cloning pretraining now uses Rust-native compact `.npz` tensor
shards by default rather than persistent JSONL. On `john0`, the Rust-native
1,024-game deflated shard took 1:35.36 and wrote 71,413,962 bytes; the same
format with `TENSOR_COMPRESSION=stored` took 18.28s and wrote 1,248,396,657
bytes. Stored shards are `5.22x` faster and `17.48x` larger on that benchmark.
Use JSONL only for bounded debug/audit shards.

Expert iteration follows the same no-JSONL rule for real training data:
`cascadiav3.expert_tensor_shard.v1` is now an active packed `.npz` schema written
directly by the Rust root exporter and consumed by `torch_train_cascadiaformer`
with `--train-format npz --val-format npz`. Expert JSONL is retained only for
small reconstruction and public-boundary audit fixtures.

The raw all-legal-action expert tensor is intentionally filtered before GPU
training. The current phase0 recovery path keeps top-256 rollout-Q actions per
root, always preserves the selected action, remaps relation edges, and trains on
materialized `relation_tail` shards derived from the filtered sparse shards.
The sparse edge list remains in the file for audit/rebuild, while the
trainer reads the fixed-capacity `uint8` action-row relation cache to avoid
Python per-edge collation and dense square relation tensors. This prevents
training batches from padding to 10k-plus legal actions on rare wide roots and
keeps the GPU input path predictable.

The first relation-tail CascadiaFormer-S run completed successfully on `john0`,
but did not produce a useful direct no-search player. Full validation selected
`best_locked_val` at step 625 over final/SWA. A 20-seed paired complete-game
benchmark of that checkpoint scored below greedy:

- K256 action menu: policy `72.5125`, Q `57.4750`, greedy `87.3375`.
- K32 action menu: policy `82.7625`, Q `77.9250`, greedy `87.3375`.

Reports:

- `cascadiav3/reports/full_v3_phase0_bootstrap_jsonl_top256_tail_b64_full_val.json`
- `cascadiav3/reports/cascadiaformer_best_step625_game20_benchmark.json`
- `cascadiav3/reports/cascadiaformer_best_step625_game20_k32_benchmark.json`
See
[IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) and
[EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).

Corrected K32 greedy-retention training is now a first-class objective rather
than an implied side effect of expert labels. The trainer can optimize and
report `greedy_policy`, `greedy_margin`, `greedy_top1`, and
`mean_greedy_rank` separately from the rollout-selected teacher action. The
paired K32 serving surface uses greedy-ranked legal actions, so
`greedy_action_index=0` is the baseline action the model must first retain
before stronger teacher/search labels can credibly improve it.

Two corrected diagnostics have now run on `john0`:

- `run_cascadiaformer_k32_greedy_retention.sh` reused old rollout/expert
  trajectory tensors. It reached perfect offline greedy retention on those
  held-out roots (`locked_val_greedy_top1=1.0`) but failed full-game evaluation
  badly because the policy entered a different state distribution.
- `run_cascadiaformer_greedy_k32_retention.sh` generated actual greedy
  self-play expert tensors. It reached `locked_val_greedy_top1=0.6780` and a
  100-game no-search complete-game benchmark scored `86.7800` mean seat score
  versus greedy `87.5875`, paired delta `-0.8075`, with `67.3625%` exact
  greedy-action match.

That is enough to say the transformer can operate in the greedy neighborhood,
but it has not fully copied greedy and should not yet be treated as an
improvement policy. Use the greedy-state run as the corrected copy-greedy
baseline:

```bash
bash cascadiav3/scripts/run_cascadiaformer_greedy_k32_retention.sh launch
bash cascadiav3/scripts/run_cascadiaformer_greedy_k32_retention.sh status
bash cascadiav3/scripts/run_cascadiaformer_greedy_k32_retention.sh fetch
```

Reports:

- `cascadiav3/reports/full_v3_greedy_k32_retention_train.json`
- `cascadiav3/reports/full_v3_greedy_k32_retention_runbook.json`
- `cascadiav3/reports/cascadiaformer_greedy_k32_retention_game100_benchmark.json`
- `cascadiav3/reports/cascadiaformer_game_benchmark_summary.md`

Plan files:

- [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md): end-to-end implementation
  plan and source hierarchy.
- [SCHEMA_CONTRACTS.md](SCHEMA_CONTRACTS.md): formal token, action, replay,
  search-root, model-config, and gate schemas.
- [CPU_PRE_GPU_MILESTONES.md](CPU_PRE_GPU_MILESTONES.md): CPU-only milestones
  and exit gates before GPU access.
- [PERFORMANCE_BUDGETS.md](PERFORMANCE_BUDGETS.md): performance-sensitive paths,
  expected scale, budgets, and profiling gates.
- [GPU_HANDOFF_GATE.md](GPU_HANDOFF_GATE.md): stop condition and next GPU-only
  decision package.
- [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md): CPU scaffold now
  implemented inside this folder.

CPU scaffold commands:

```bash
PYTHONPATH=cascadiav3/src python3 -m unittest discover -s cascadiav3/tests -v
PYTHONPATH=cascadiav3/src python3 -m cascadiav3.validate --write-artifacts
PYTHONPATH=cascadiav3/src python3 -m cascadiav3.validate_schema_registry --include-legacy --include-expert
cascadiav3/real-root-exporter/target/release/cascadiav3-real-root-exporter --chance-mcts-dry-run --allow-model-fallback --seed-count 2 --plies-per-seed 2 --out cascadiav3/fixtures/expert_tiny.jsonl --manifest cascadiav3/fixtures/expert_tiny_manifest.json
cascadiav3/real-root-exporter/target/release/cascadiav3-real-root-exporter --validate-expert-reconstruction --in cascadiav3/fixtures/expert_tiny.jsonl --manifest cascadiav3/fixtures/expert_tiny_manifest.json
PYTHONPATH=cascadiav3/src python3 -m cascadiav3.validate_public_boundary --roots cascadiav3/fixtures/expert_tiny.jsonl --deny-hidden-fields
PYTHONPATH=cascadiav3/src python3 -m cascadiav3.validate_hidden_redetermination --seeds 2026063000:2026063010
PYTHONPATH=cascadiav3/src python3 -m cascadiav3.validate_d6_roundtrip --roots cascadiav3/fixtures/expert_tiny.jsonl
PYTHONPATH=cascadiav3/src python3 -m cascadiav3.validate_category_targets --roots cascadiav3/fixtures/expert_tiny.jsonl
PYTHONPATH=cascadiav3/src python3 -m cascadiav3.torch_train_cascadiaformer --model-size tiny --train cascadiav3/fixtures/expert_tiny.jsonl --val cascadiav3/fixtures/expert_tiny.jsonl --steps 200 --batch-size 2 --device cpu --overfit-one-batch
PYTHONPATH=cascadiav3/src python3 -m cascadiav3.torch_gpu_smoke
PYTHONPATH=cascadiav3/src python3 -m cascadiav3.torch_train_tiny --steps 300
PYTHONPATH=cascadiav3/src python3 -m cascadiav3.torch_train_replay --steps 400 --batch-size 2
```

Remote `john0` runner:

```bash
bash cascadiav3/scripts/run_john0_gpu_smoke.sh
bash cascadiav3/scripts/run_crt_sampled_teacher_relation_bias_pilot.sh
bash cascadiav3/scripts/run_crt_scaled_sampled_teacher_relation_bias_pilot.sh
bash cascadiav3/scripts/run_crt_wide32_sampled_teacher_relation_bias_pilot.sh
bash cascadiav3/scripts/run_crt_wide32_prefilter_eval.sh
bash cascadiav3/scripts/run_crt_wide32_top16_margin_relation_bias_pilot.sh
bash cascadiav3/scripts/run_crt_wide32_r16_sampled_teacher_relation_bias_pilot.sh
bash cascadiav3/scripts/run_crt_wide32_r16_prefilter_blend_eval.sh
bash cascadiav3/scripts/run_crt_wide32_r16_top16_margin_relation_bias_pilot.sh
bash cascadiav3/scripts/run_crt_wide32_r16x2_sampled_teacher_relation_bias_pilot.sh
bash cascadiav3/scripts/run_crt_wide32_r16x2_semantic_relation_bias_pilot.sh
bash cascadiav3/scripts/run_crt_wide32_r16p20_semantic_relation_bias_pilot.sh
bash cascadiav3/scripts/run_crt_wide32_r16p20_semantic_retention_pilot.sh
bash cascadiav3/scripts/run_crt_wide32_r16p20_semantic_cross_attention_pilot.sh
bash cascadiav3/scripts/run_crt_wide32_r16p20_semantic_residual_attention_pilot.sh
bash cascadiav3/scripts/run_crt_wide32_r16p20_semantic_action_set_pilot.sh
bash cascadiav3/scripts/run_crt_wide32_r16p20_semantic_species_moe_pilot.sh
bash cascadiav3/scripts/run_crt_wide32_r16p80_semantic_relation_bias_pilot.sh
bash cascadiav3/scripts/run_crt_wide32_r16p80_semantic_relation_bias_detached.sh launch
bash cascadiav3/scripts/run_crt_wide32_r16p80_semantic_relation_bias_detached.sh status
bash cascadiav3/scripts/run_crt_wide32_r16p80_semantic_relation_bias_detached.sh fetch
bash cascadiav3/scripts/run_crt_wide32_r16p80_semantic_residual_attention_pilot.sh
bash cascadiav3/scripts/run_crt_wide32_r16p80_residual_seed_ensemble_eval.sh
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
CORPUS_FORMAT=npz TENSOR_EXPORTER=rust TENSOR_COMPRESSION=stored KEEP_JSONL=0 bash cascadiav3/scripts/run_greedy_policy_pretrain.sh launch
bash cascadiav3/scripts/run_greedy_policy_pretrain.sh status
bash cascadiav3/scripts/run_greedy_policy_pretrain.sh stop
bash cascadiav3/scripts/run_greedy_policy_pretrain.sh fetch
bash cascadiav3/scripts/run_full_v3_training_pipeline.sh launch
bash cascadiav3/scripts/run_full_v3_training_pipeline.sh status
bash cascadiav3/scripts/run_full_v3_training_pipeline.sh stop
bash cascadiav3/scripts/run_full_v3_training_pipeline.sh fetch
bash cascadiav3/scripts/run_cascadiaformer_k32_greedy_retention.sh launch
bash cascadiav3/scripts/run_cascadiaformer_k32_greedy_retention.sh status
bash cascadiav3/scripts/run_cascadiaformer_k32_greedy_retention.sh stop
bash cascadiav3/scripts/run_cascadiaformer_k32_greedy_retention.sh fetch
bash cascadiav3/scripts/run_cascadiaformer_greedy_k32_retention.sh launch
bash cascadiav3/scripts/run_cascadiaformer_greedy_k32_retention.sh status
bash cascadiav3/scripts/run_cascadiaformer_greedy_k32_retention.sh stop
bash cascadiav3/scripts/run_cascadiaformer_greedy_k32_retention.sh fetch
bash cascadiav3/scripts/run_crt_wide32_r16p20_prefilter_forensics.sh
bash cascadiav3/scripts/run_crt_wide32_r16p20_source_union_prefilter.sh
bash cascadiav3/scripts/run_crt_wide32_r16p20_learned_source_gate.sh
```
