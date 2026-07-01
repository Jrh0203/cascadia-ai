# Cascadia Transformer Architecture Proposal

Date: 2026-06-29

## Recommendation

Build a **Cascadia Relational Transformer**, abbreviated here as `CRT-v1`:

- a sparse entity/graph transformer over the public game state;
- an action-query decoder that scores every legal action without flattening the game into a dense board image;
- auxiliary opponent-intent and future-market heads;
- teacher-supervised value/policy training from exhaustive legal actions and K32/R600 rollouts;
- eventual use as both a direct policy and a high-quality prefilter/value model for the existing search.

The bet is not "replace NNUE with a giant transformer because we have a big GPU." The bet is:

> The remaining score gap is cross-turn, market, and opponent allocation. A transformer should model public entities, legal actions, recent action history, and future access jointly.

That is the direction best supported by both the local Cascadia evidence and the game-AI literature.

## Local Evidence To Preserve

The current strongest canonical reference is still around 95.7 to 95.9 base points, not 100. `docs/v2/SCORE_GAP.md` reports the sealed canonical reference at 95.744 over 1,000 held-out games, leaving 4.256 points to the target. The failure mode is not a uniform wildlife deficit; the evidence repeatedly points to cross-turn pattern realization, market competition, and opponent-conditioned availability.

The prior `v4-opp` result is the strongest architectural clue: adding detailed opponent features moved the needle by about +1.33 points and doubled head-to-head win rate in the historical v1/v4-opp setting. Feature signal moved strength. Search-time tweaks mostly did not.

Several v2/v3 experiments sharpen the design:

- `r0-spatial-mlx-tournament-v1` showed compact dense board tensors were value-noninferior but much slower than exact entities. So `CRT-v1` should start from sparse occupied/frontier/component/action entities, not a 441-cell or even 121-cell ViT.
- `opportunity-cross-attention-mlx-tournament-v1` was a null result as a candidate-conditioned search-time adapter: slight global recall signal, no strategic recall gain, and failed latency/RSS gates. So attention should be in the trunk and training objective, not bolted onto the existing scorer as an expensive adapter.
- `o1-opponent-intent-mlx-factorial-v1` passed: recent public action history and next-draft auxiliary supervision improved prediction of opponent tile consumption and market survival. So opponent intent and market survival should be auxiliary heads in the main architecture.
- `p1-relational-hierarchical-pointer-foundation-v1` preregistered exactly the right serving shape: active state once, then draft/prelude, tile-placement, and wildlife-placement pointers. `CRT-v1` should use that hierarchical action language.
- The v3 NNUE system already has the right scientific contracts: replay-authoritative game records, exact public-state boundaries, signed score-to-go labels, teacher root labels preserving candidate means/variance/counts, exhaustive legal action enumeration, D6 augmentation, and strict cross-backend verification. The transformer should inherit those contracts instead of inventing a looser ML pipeline.

## Online Research Summary

The literature points to a hybrid of search-supervised policy/value learning, set/graph attention, and action pointers.

- AlphaZero shows the core recipe for board games: policy/value networks trained by self-play and improved by search, using only game rules rather than handcrafted evaluation [AlphaZero, arXiv 1712.01815](https://arxiv.org/abs/1712.01815).
- MuZero shows learned latent models can support planning when dynamics are unknown [MuZero, arXiv 1911.08265](https://arxiv.org/abs/1911.08265). Cascadia has an exact simulator, so full learned dynamics are not the first priority; learned future-access and opponent-market dynamics are.
- Gumbel AlphaZero/MuZero is relevant because Cascadia has expensive search and variable actions. The paper specifically improves policy learning when few simulations are available [Danihelka et al., ICLR 2022](https://openreview.net/forum?id=bERaNdoegnO).
- Student of Games/ReBeL-style work warns against treating hidden-information games as naive perfect-information trees. Cascadia's hidden stack order is not poker, but public belief and opponent dynamics matter [Student of Games, arXiv 2112.03178](https://arxiv.org/abs/2112.03178), [ReBeL, arXiv 2007.13544](https://arxiv.org/abs/2007.13544).
- Set Transformer is the clean fit for variable unordered entity sets such as occupied cells, frontier cells, market slots, opponent boards, and candidate actions [Set Transformer, arXiv 1810.00825](https://arxiv.org/abs/1810.00825).
- Graphormer and GraphGPS argue that transformers on graphs need explicit structural encodings. Cascadia should encode hex distance, direction, D6 orbit, board membership, seat relation, component membership, and market/action relations rather than hoping vanilla attention discovers them [Graphormer, arXiv 2106.05234](https://arxiv.org/abs/2106.05234), [GraphGPS, arXiv 2205.12454](https://arxiv.org/abs/2205.12454).
- Pointer Networks are directly relevant to Cascadia's variable legal-action vocabulary: draft slot, tile placement, wildlife placement, and candidate selection are pointers into public objects [Pointer Networks, arXiv 1506.03134](https://arxiv.org/abs/1506.03134).
- GTrXL explains why plain transformers can be unstable in RL and why recurrence/history should be treated carefully in partially observable settings [GTrXL, arXiv 1910.06764](https://arxiv.org/abs/1910.06764).
- Decision Transformer is useful as a data-modeling and high-return-trajectory auxiliary, but should not be the primary policy because Cascadia needs exact legal action scoring, public-boundary safety, and search-supervised values [Decision Transformer, arXiv 2106.01345](https://arxiv.org/abs/2106.01345).
- Perceiver IO is useful as a design pattern for flexible input and output queries. It supports arbitrary structured inputs/outputs and scales linearly with input/output size, making it a good fit for action-query decoding if full self-attention over every candidate becomes expensive [Perceiver IO, arXiv 2107.14795](https://arxiv.org/abs/2107.14795).
- NVIDIA lists the RTX 5090 as Blackwell with 32 GB GDDR7, 21,760 CUDA cores, fifth-generation Tensor Cores, and 3,352 AI TOPS [NVIDIA RTX 5090](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/). That is enough for serious medium transformer work, but not enough to waste memory on dense empty board representations.

## Architecture: `CRT-v1`

### Inputs

All inputs are public-state legal. Hidden future stack order is never encoded.

State tokens:

- focal occupied-cell tokens: exact q/r, terrain edges, wildlife, allowed wildlife, keystone, tile age, component ids;
- focal frontier tokens: legal tile destinations with local geometry and prospective component connectivity;
- opponent board tokens: compact occupied/component/motif summaries for all three opponents, seat-relative;
- market tokens: four tile slots, four wildlife slots, three-of-kind state, Nature Token wipe affordances;
- bag/public-supply tokens: only public count/distribution information allowed by the rules and current engine boundary;
- global tokens: turn, focal seat, phase bucket, focal tokens, current exact score, scoring-card id, remaining turns;
- history tokens: last 12 public actions using the `opponent_intent` schema: seat, draft, slots, tile/wildlife identity, placement, wipe/prelude.

Action tokens:

- prelude token: free replacement, paid wipe count, wiped slots, Nature Token cost;
- draft token: paired or independent, tile slot, wildlife slot;
- tile-placement token: destination q/r, rotation, local frontier id;
- wildlife-placement token: destination q/r or no wildlife;
- exact immediate deltas: base score, per-animal score, terrain/habitat deltas, token deltas;
- opportunity deltas: demand/supply, contested denial, access delay, completion class, and v3 opportunity summaries.

The action representation should be hierarchical/pointer-native. The model should score legal complete actions, but the internal language should still know the selected draft, tile point, and wildlife point.

### Encoder

Use a hybrid graph/set transformer:

1. Type-specific token embeddings.
2. Hex structural encodings: relative q/r distance, axial direction, same component, adjacency, D6 orbit, local frontier relation.
3. Seat and board relation encodings: focal/opponent, relative seat order, turns until next access.
4. Market/action relation encodings: action uses tile slot, action uses wildlife slot, action competes with opponent likely demand.
5. Local graph attention or message-passing over real hex edges for 1-2 layers.
6. Global self-attention over compressed entity tokens.

Start with Pre-LN/RMSNorm, SwiGLU MLPs, bf16 training, and no recurrence in the first serving model. Add GTrXL-style memory only after the feed-forward public-history version is stable.

### Action Decoder

Encode the public state once, then score action queries:

```text
state tokens -> state transformer -> latent state
action factor tokens -> action encoder -> action queries
action queries x latent state -> continuation Q / policy logit / uncertainty
```

This is the Perceiver IO/Pointer Network flavor: outputs are legal-action queries, not a fixed output vocabulary.

The direct action value is:

```text
exact_afterstate_base_score + transformer_predicted_score_to_go
```

That matches v3 serving semantics.

### Heads

Primary heads:

- scalar score-to-go mean;
- score-to-go uncertainty or quantiles;
- legal-action policy logits;
- per-category score-to-go decomposition: habitat, Bear, Elk, Salmon, Hawk, Fox, Nature Tokens.

Auxiliary heads:

- opponent next-draft distribution for each of three opponents;
- market slot/species survival to focal next access;
- paid-wipe/free-replacement likelihood;
- pattern portfolio: bear-pair targets, elk-line targets, salmon-run targets, hawk-isolation targets, fox-diversity targets;
- optional latent next-public-state embedding after one opponent cycle.

The per-category and portfolio heads are important. Prior Cascadia experiments repeatedly found "gain Bear, lose allocation" failures. A single scalar objective can hide that until gameplay.

### Initial Sizes

Train three sizes, but only after the tokenizer and targets are byte-stable:

| Model | Layers | Width | Heads | Approx params | Use |
|---|---:|---:|---:|---:|---|
| `CRT-S` | 10-12 | 384 | 8 | 25M-45M | Tokenizer/debug/ablation |
| `CRT-M` | 14-18 | 512 | 8 | 70M-120M | Main RTX 5090 target |
| `CRT-L` | 20-24 | 768 | 12 | 180M-300M | Only if data and offline gates pass |

The 5090's 32 GB memory should comfortably handle `CRT-S` and `CRT-M` with bf16, gradient checkpointing, packed batches, and FlashAttention-style kernels. `CRT-L` is not the first move; data quality and legal-action conditioning are more likely to dominate.

## Training Plan

### Stage 0: Contracts And Tokenization

Create a Rust-owned token exporter that consumes replay-authoritative records and teacher labels:

- `V3GameRecord`;
- `V3TrainingEntry`;
- `V3TeacherRootLabel`;
- existing opponent-intent windows;
- action/frontier/pointer caches from P1/R3 work.

Required tests:

- public-boundary test: no real post-draft refill leakage;
- replay roundtrip: exported state/action hash equals Rust source;
- D6 transform/inverse test for every spatial token and action pointer;
- legal action coverage: every Rust legal action has exactly one tokenized action id;
- hidden-state redetermination test: same public state with different hidden stack order exports identical observations;
- category target consistency: exact score deltas sum to base-total delta.

### Stage 1: Offline Supervised Warm Start

Train `CRT-S` and then `CRT-M` on:

- realized signed score-to-go;
- v3 teacher score-to-go where available;
- full candidate teacher distributions, not only winner labels;
- opponent-intent survival/draft targets;
- category decomposition.

Use loss terms:

```text
L = value_huber
  + teacher_weighted_distributional_policy
  + pairwise_action_margin
  + category_decomposition_loss
  + opponent_intent_aux
  + portfolio_aux
```

Teacher variance and sample count should weight the loss exactly as v3 already does. High-variance search labels should teach a softer distribution, not brittle winner imitation.

### Stage 2: Active Teacher Labeling

Use the current champion/control to collect roots, then prioritize labels by:

- high policy uncertainty;
- high disagreement between NNUE/v3 and transformer;
- Bear/Elk/Hawk allocation sensitivity;
- high market competition;
- late terminal conversion states;
- rare Nature Token prelude cases;
- roots where direct action and K32/R600 disagree.

Label selected roots with exhaustive legal enumeration and K32/R600 or stronger teacher budgets. Preserve all candidate means, variances, counts, and eliminated-candidate statistics.

### Stage 3: Search-Integrated Self-Play

Do not jump straight to "transformer-only champion." Integrate in four increasingly risky modes:

1. `CRT-direct`: enumerate legal actions, choose transformer value.
2. `CRT-prefilter`: transformer top 64 or top 128 feeds existing K32/R600/sequential halving.
3. `CRT-leaf`: existing search uses transformer value as a leaf only after terminal rollout gates pass.
4. `CRT-Gumbel`: use Gumbel-style policy improvement at roots when simulation budget is small.

The likely first real gain is `CRT-prefilter`: better action proposal and market/opponent sensitivity without trusting the new value head as the whole player.

### Stage 4: Optional Decision Transformer

Train an RTG-conditioned trajectory model as a diagnostic and proposal generator:

- condition on desired final score, e.g. 100+;
- model sequences of public states and hierarchical actions;
- generate candidate action priors or rare high-score curricula.

Do not promote it as the main policy unless it proves it can respect exact legal action boundaries and beat the action-query model offline. Decision Transformer is a useful lens on high-return trajectories, not the main Cascadia engine.

## Evaluation Gates

### Offline Gates

Before gameplay:

- legal-action coverage: 100%;
- D6 action equivariance: exact transform/inverse;
- no hidden-boundary leakage;
- score-to-go validation beats v3 NNUE on matched held-out roots;
- top-K candidate retention beats the existing direct prefilter on strategic roots;
- top-one regret improves against the teacher by a predeclared margin;
- category decomposition does not improve Bear by collapsing Elk+Salmon+Hawk+Fox;
- opponent-intent heads reproduce or beat the A2 Brier/NLL improvements on held-out policy pools;
- serving batch latency and memory fit the intended mode.

### Gameplay Gates

Use paired, frozen domains:

- `CRT-direct` must first be nonregressive against the current direct neural control.
- `CRT-prefilter + K32/R600` must beat canonical K32/R600 by at least +0.25 over a pilot before a 250-pair gate.
- Any champion claim needs the existing 1,000-game final style: game-block CIs, exact artifacts, score breakdown, latency, and provenance.
- Reject any model that repeats the known allocation failure: Bear up, total score flat, non-Bear wildlife down.

## Implementation Stack For The RTX 5090

Use the 5090 machine for training, not for changing the Rust scientific source of truth.

Recommended stack:

- Rust exporter in this repo remains authoritative for rules, legal actions, targets, D6, and public-boundary hashes.
- PyTorch on CUDA for transformer training.
- bf16 mixed precision.
- FlashAttention or PyTorch scaled-dot-product attention.
- safetensors checkpoints with JSON manifests binding source revision, tokenizer schema, dataset manifests, training code hash, CUDA/PyTorch identity, seed, optimizer state, and loss weights.
- Python batch inference service for early experiments.
- Later, export to ONNX/TensorRT or a purpose-built CUDA service only after the model earns gameplay relevance.

The 5090 makes it practical to iterate quickly on `CRT-S`/`CRT-M`, run ablations, and train with large action-query batches. It does not remove the need for Rust-owned exactness, held-out domains, or paired promotion.

## First Concrete Experiment

Experiment id:

```text
transformer-crt-s-action-query-v1
```

Question:

Can a small relational action-query transformer beat the current direct neural/action-ranking baselines on held-out teacher regret while preserving allocation?

Scope:

- Tokenize existing public states and complete legal actions.
- Train `CRT-S` on realized score-to-go, teacher candidate distributions, category scores, and opponent-intent auxiliaries.
- Compare against existing direct neural baselines on the same roots.
- No gameplay unless all offline gates pass.

Minimum report:

- tokenizer schema and hashes;
- dataset manifest;
- model manifest;
- exact action coverage;
- D6 proof;
- value RMSE/MAE;
- policy NLL;
- top-1/top-4/top-16/top-64 recall;
- mean top-one regret;
- category error by animal/habitat/token;
- opponent survival Brier/NLL;
- latency and memory on the RTX 5090;
- ablations: no-history, no-opponent, no-category, no-action-delta, no-auxiliary.

If this passes, the next experiment is `CRT-M` with active K32/R600 labels and a `CRT-prefilter` gameplay pilot.

## Rejected First Bets

Do not start with:

- dense 441/121-cell ViT boards: local evidence says exact entities are faster and retain value;
- standalone Decision Transformer policy: useful auxiliary, weak legal-action fit;
- full MuZero dynamics: Cascadia rules are known; learn opponent/future-access beliefs first;
- another search-time cross-attention adapter: already null and too expensive in the local experiment;
- scalar-only value learning: too many prior failures gained one species while breaking allocation.

## Bottom Line

The transformer worth building is not a language-model-shaped novelty. It is a legality-preserving, sparse, opponent-aware, action-query graph/set transformer that reuses Cascadia's exact rules and teacher machinery.

If it works, it should first show up as better candidate selection and better cross-turn allocation under market competition. Then it can become the value head. Only after that should it be trusted as the main player.
