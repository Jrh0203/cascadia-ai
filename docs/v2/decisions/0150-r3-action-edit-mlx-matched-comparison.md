# ADR 0150: R3 Action-Edit MLX Matched Comparison

Status: accepted; experiment complete; compact representations rejected

Date: 2026-06-17

Experiment: `r3-action-edit-mlx-comparison-v1`

Protocol: `r3-action-edit-mlx-matched-comparison-v1`

Research-plan item: R3

Foundation: ADR 0148 and
`docs/v2/reports/r3-action-edit-foundation-v1-result.md`

## Context

ADR 0148 established an exact action-centric representation over one reusable
public state trunk. Across 2,679,459 tested actions, every edit reproduced the
authoritative public successor, exact semantic supply, regenerated global
objects, canonical codec, and D6 action view. Median edit size was 55
structural tokens and maximum size was 70.

That foundation did not test learned decision quality or serving speed. The
next question is deliberately narrower:

> Can an action ranker replace one complete R2 afterstate per candidate with a
> canonical R3 local patch plus exact global edits, without losing held-out
> ranking quality, while materially improving realistic action throughput or
> memory?

The accepted R3 locality result also rules out a patch-only shortcut. Radius 3
covered every directly changed coordinate in only 58.2432% of actions. Every
R3 arm in this experiment therefore retains the same exact global board,
frontier, component, and motif edits. Only local-patch radius changes.

The prior S1 learned comparison is also binding evidence. Candidate-relational
exact supply improved aggregate top-64 recall and retained regret, but regressed
low-supply and independent-draft-winner slices. All four R3 arms therefore
receive the same exact public supply and candidate-relational supply facts, and
those two slices are first-class noninferiority gates.

## Decision

Run one four-arm, iso-model, iso-data MLX comparison:

| Arm | Candidate spatial representation | Host |
|---|---|---|
| `c0-full-r2-afterstate` | Exact canonical active-board R2 afterstate tokens | `john1` |
| `t1-r3-radius3-global` | Radius-3 canonical patch plus exact global edits | `john2` |
| `t2-r3-radius2-global` | Radius-2 crop plus the same exact global edits | `john3` |
| `t3-r3-radius1-global` | Radius-1 crop plus the same exact global edits | `john4` |

The four arms use identical:

- train and validation decisions;
- candidate identities and ordering;
- teacher labels and candidate priors;
- exact semantic-supply facts;
- model module graph and parameter initialization;
- optimizer schedule and candidate sampler;
- parent-state D6 schedule;
- loss coefficients;
- validation scoring and stable tie breaking; and
- performance benchmark shapes.

The arm name is metadata only. It does not alter model construction or create
dormant arm-specific parameters.

## Public Information Boundary

Inputs are restricted to the open complete-action graded-oracle train and
validation datasets plus the accepted exact semantic-supply sidecar.

Allowed:

- public `PositionRecord`;
- public market and public supply;
- complete legal action factors;
- visible staged market after the selected prelude;
- exact public semantic archetype counts;
- selected public archetype and public frontier compatibility;
- observable screen priors;
- public afterstate geometry; and
- R600, R1200, and R4800 teacher labels on the open train/validation corpus.

Forbidden:

- sealed test or final-gameplay data;
- hidden tile-stack order;
- hidden wildlife-bag order;
- excluded tile identities;
- future refill realizations;
- future actions;
- terminal score targets outside the graded labels;
- gameplay result selection; and
- any arm-specific data, warm start, or optimization budget.

## Exact Shared Cache

A standalone Rust exporter owns the MLX sidecar. It replays each source game
from its raw seed, verifies every group against the authoritative state, and
advances with the stored champion action exactly as the source exporter did.

Each decision stores one parent R2 state:

- exact R2 token types, relative seats, and 52-byte payloads;
- exact per-board type counts;
- R2 market, player, and global features;
- public-state and group identities; and
- source candidate offsets.

Each selected action stores:

- source candidate index and action hash;
- canonical R3 transform ID and transformed tile center;
- an exact multiset delta from the canonicalized parent R2 board to the
  canonicalized R2 afterstate;
- the full canonical radius-3 R3 MLX token sequence; and
- exact alignment to the graded-oracle and S1 sidecars.

The control is not a lossy delta model. Python reconstructs the complete
candidate-specific R2 afterstate token multiset by:

1. applying the Rust-authoritative D6 transform to the cached parent board;
2. translating token coordinates into the selected action frame;
3. removing the exact parent token indices recorded by Rust;
4. appending exact added R2 tokens; and
5. checking the reconstructed multiset identity against the exporter proof.

The R3 cache stores radius 3 once. Radius 2 and radius 1 are exact crops of
patch tokens by canonical hex distance. Every non-patch token is byte-identical
across all three R3 arms.

## R3 MLX Token Contract

One token has:

- a token-type code;
- an operation code;
- a fixed 64-byte signed payload; and
- an active mask.

The common Python materialization expands this into one shared 80-wide input:

```text
10 token-type one-hot channels
+ 6 operation one-hot channels
+ 64 normalized payload channels
= 80 features
```

The R3 token stream is lossless for the complete canonical action view and
contains:

- one action-meta token;
- local-patch cells;
- exact component-key catalog entries;
- board additions, removals, and update sides;
- frontier additions, removals, and update sides;
- frontier-to-component touch records;
- component additions, removals, and update sides; and
- motif additions, removals, and update sides.

Updates are encoded as paired before/after tokens with a stable object index.
Frontier component references use an exact per-action catalog containing the
full 32-byte component key. No hash truncation, coordinate clipping, object
cap, or silent payload overflow is allowed.

The control maps exact R2 occupied, frontier, component, and motif tokens onto
the same 80-wide surface. The candidate encoder is therefore identical across
all arms.

## Frozen Train Cohort

The open train split has 560 decisions and 2,135,111 complete legal actions.
Every decision contributes at most 512 actions.

If a decision has at most 512 actions, all are retained. Otherwise selection
is deterministic:

1. retain the stable R4800 winner and played champion;
2. retain every R4800-labeled action;
3. retain every R600-labeled action;
4. retain every sentinel, substantial-top, best-frontier, and
   champion-selected action;
5. add actions in `(screen_rank, action_hash)` order until 256 unique actions
   are present;
6. add up to 128 evenly spaced actions from the remaining R1200-labeled actions
   in `(screen_rank, action_hash)` order; and
7. fill to 512 by ascending BLAKE3 of
   `("r3-mlx-train-cohort-v1", group_id, action_hash)`.

If a priority tier would exceed 512, it is truncated by the same stable
`(screen_rank, action_hash)` order. Production export fails if the mandatory
set from steps 1 through 4 alone exceeds 512.

This policy preserves every high-fidelity R600/R4800 action, the exact winner,
the played action, screen leaders, rank coverage, and a target-independent
hash-stratified tail.

The validation split is never sampled. Every one of its 240 decisions and
860,203 actions is scored exactly once.

## Shared Model

### Parent state

The accepted R2 Perceiver fixed-latent state trunk is reused:

- 60-to-64 token adapter;
- exact 4 by 92 board-local layout;
- four type-summary tokens and one player token per board;
- 16 fixed latents per board;
- one cross-attention block;
- one latent self-attention block;
- one explicit global/market/player-board context block; and
- one 64-wide parent-state summary.

The parent encoder runs exactly once per decision, not once per candidate.

### Candidate spatial edit

Every arm uses:

- 80-to-64 token adapter;
- 8 fixed candidate latents;
- one cross-attention block;
- one latent self-attention block; and
- masked mean-plus-max projection to one 64-wide candidate-spatial summary.

Token sequence length is variable and uncapped in the cache. MLX batches pad
only to the maximum length present in that batch. Padding is all zero and
must be invariant.

### Shared factual context

All arms receive identical:

- 140-wide graded action features;
- 8 observable screen priors;
- visible staged-market entities;
- 83 exact semantic-supply scalars;
- selected semantic-archetype identity;
- 17 public frontier-compatibility features; and
- screen value used as the residual baseline.

World action coordinates and the parent R2 trunk receive the same deterministic
D6 transform. Canonical candidate-spatial tokens are invariant under that
transform.

### Output

Independent candidate scoring combines the parent-state, candidate-spatial,
action, prior, staged-market, exact-supply, and supply-relation summaries.
There is no all-candidate attention or candidate-set pooling in the model.

The model emits:

- one bounded score residual added to screen value; and
- one positive standard-error estimate.

Independent scoring permits realistic groups of up to 16,384 legal actions to
be evaluated in fixed candidate chunks while encoding the parent once.

## Frozen Optimization

| Variable | Value |
|---|---:|
| Seed | `2026061708` |
| Optimizer | AdamW |
| Steps | 3,000 |
| Groups per step | 4 |
| Candidates per sampled group | At most 512 |
| Learning rate | `0.0001` |
| Weight decay | `0.0001` |
| Checkpoint interval | 250 steps |
| Metric interval | 100 steps |
| Validation during training | Fixed 24-group open probe |
| Full validation | Once, after step 3,000 |
| Candidate scoring chunk | 256 |
| Initialization | Fresh and byte-identical across arms |
| Warm start | Prohibited |
| Early stopping | Prohibited |

For each step, three group slots advance through independent deterministic
permutations of all 560 train groups. The fourth slot alternates between:

- the 133 low-supply groups with at most 20 unseen tiles; and
- the 55 groups whose stable R4800 winner is an independent draft.

This schedule is identical across arms. Transform IDs are deterministic
functions of `(seed, optimizer_step, group_slot)` over all 12 D6 elements.

## Shared Objective

The scalar loss preserves the graded-oracle objective:

```text
r1200 uncertainty-weighted Huber
+ 4.0 * r4800 uncertainty-weighted Huber
+ 0.5 * r1200 listwise cross entropy
+ 1.0 * r4800 winner cross entropy
+ 0.1 * standard-error calibration
+ 0.01 * screen-only residual regularization
```

Listwise terms operate on the sampled train cohort and complete validation
group respectively. The stable R4800 winner is guaranteed to be present in
every train group.

No representation reconstruction loss is used in the primary comparison. Rust
already proves exactness, and an auxiliary loss tied to token count could
favor one arm for a reason unrelated to ranking.

## Evaluation

Quality:

- R4800 MAE, RMSE, bias, correlation, calibration slope, and intercept;
- top-1, top-8, top-32, and top-64 stable R4800 winner recall;
- retained R4800 regret at the same widths;
- top-64 95% teacher-confidence-set coverage;
- early, middle, and late phase metrics;
- low-supply metrics;
- independent-draft-winner metrics; and
- a fixed prediction panel with action hashes.

Performance:

- parent encodes per decision, which must equal one;
- action scores per second at 256-action chunks;
- complete-decision latency at observed group widths;
- P50, P95, and P99 latency;
- candidate token counts and padding;
- MLX active, cache, and peak memory;
- process peak RSS and process swap;
- system swap before/after as reported operational evidence; and
- compile, warmup, and steady-state timings separately.

Process peak RSS is measured in a fresh serving worker that reloads the exact
final checkpoint and verified open validation inputs. The worker's lifetime
high-water mark includes runtime startup, lightweight data mappings, checkpoint
load, compile, warmup, and serving, but excludes exhaustive cache preflight,
training, and complete validation. The frozen pre-production correction and
empirical proof are in
`docs/v2/reports/r3-action-edit-mlx-serving-rss-amendment-2026-06-17.md`.

Stable score ties are broken by action hash.

## Promotion Gates

Every arm must pass:

```text
complete validation coverage == 240 decisions and 860,203 actions
finite scores and uncertainties == 100%
parent encodes == validation decisions
process swap == 0
peak MLX active memory <= 4 GiB
peak RSS <= 4 GiB
P99 decision latency <= 250 ms
action throughput >= 20,000 scores/second
```

An R3 arm is quality-noninferior to C0 only if all hold:

```text
R4800 MAE delta <= 0.05
R4800 RMSE delta <= 0.05
top-64 winner-recall delta >= -0.005
top-64 retained-regret delta <= 0.005
low-supply top-64 recall delta >= -0.01
independent-draft-winner top-64 recall delta >= -0.01
top-64 confidence-set coverage >= 0.99
```

An R3 arm is materially more efficient only if at least one holds:

```text
action throughput >= 1.35 * C0
P99 decision latency <= 0.80 * C0
peak active MLX memory <= 0.80 * C0
peak RSS <= 0.80 * C0
```

The selected representation is the fastest quality-noninferior R3 arm by
action throughput. Ties within 1% select the smaller local radius, then lower
P99 latency, then lower active memory.

If no R3 arm is both quality-noninferior and materially more efficient, the
classification is a valid null and no representation advances.

Passing this experiment authorizes only a later paired-gameplay gate. It does
not promote a model, claim a mean score increase, or claim the 100-point goal.

## Cluster Execution

One immutable bundle and one content-addressed cache are built and verified
before optimizer work.

Production assignment:

| Host | Arm |
|---|---|
| `john1` | `c0-full-r2-afterstate` |
| `john2` | `t1-r3-radius3-global` |
| `john3` | `t2-r3-radius2-global` |
| `john4` | `t3-r3-radius1-global` |

Before production, john1 and john4 run the same bounded cache and 10-step
training smoke. Scientific batch identities, candidate counts, initial
parameter hashes, panel action identities, and stable panel ranking must be
byte-identical.

MLX GPU reductions are not bitwise deterministic, including repeated runs on
the same M4 host. The pre-production numerical-parity amendment in
`docs/v2/reports/r3-action-edit-mlx-cross-host-smoke-amendment-2026-06-17.md`
therefore replaces exact floating-point equality with these fail-closed bounds:

```text
loss max absolute drift <= 1e-4
loss max relative drift <= 1e-5
checkpoint parameter max absolute drift <= 1e-4
checkpoint parameter mean absolute drift <= 1e-6
prediction score max absolute drift <= 1e-4
prediction uncertainty max absolute drift <= 1e-5
```

The smoke comparison is invalid unless both complete checkpoints are bound by
their report checksums and every exact-identity check also passes.

The scheduler graph contains cache export, cache fanout, four source/runtime/
MLX preflights, four concurrent arm runs, report collection, forward and
reverse classification, and one order proof. A successful negative
classification exits zero; malformed evidence exits nonzero.

Each preflight performs exhaustive checksum, semantic, source-action, and S1
identity verification in its own process. Production optimizer processes may
reuse that content-addressed proof for lightweight header binding; the
exhaustive path remains the default. Every final serving benchmark runs in a
second fresh process, and the classifier rejects reports without bound
request/result, checkpoint, and open-data proof identities.

## Consequences

1. Do not materialize one duplicated full afterstate cache row per candidate.
2. Do not compare patch-only R3 arms.
3. Do not add arm-specific capacity, losses, training data, or stopping.
4. Do not use the sealed test split or gameplay to select the representation.
5. If a compact R3 arm advances, the next gate is paired gameplay with the
   serving implementation and exact same representation.
6. If all R3 arms fail quality, preserve the exact foundation and investigate
   candidate/state interaction rather than weakening exactness.
7. If all R3 arms fail efficiency, profile token materialization and MLX
   padding before changing model semantics.

## Result

The four-arm production campaign completed on 2026-06-17. Every arm passed
the absolute evidence and serving gates, but no compact R3 arm passed all
quality-noninferiority gates and no compact arm met a material-efficiency
gate. Forward and reverse classifications were byte-identical.

The full R2 afterstate control reached MAE 1.32023, RMSE 1.74231, 72.50%
top-64 recall, 0.09812 regret, and 86,208 fixed-chunk scores/s. Radius one
improved recall to 74.58% but worsened MAE to 1.48856, missed protected
low-supply and independent-draft slices, remained below 99% confidence
coverage, and ran at 56,037 scores/s.

The terminal classification is:

```text
r3_action_edit_mlx_all_treatments_degraded
selected_arm = null
```

Classification ID:
`49260f87006bf9c49f145cd6de89db131ad916a9532d64b21c578201312404ae`.

Order proof:
`09a35dc062792159de3ed3fe599b01d93495d68d69e64e8cbb2fa97d6ff30291`.

The classifier-ineligible four-arm failure-atlas merge is:
`462b3c1935a6eabb2853bd39c816a23fc4ef5630f843d151e0239029065725f4`.
It found that radius one ranked the R4800 winner better than radius three in
113 decisions versus 82 in the opposite direction, while all compact arms
missed the top-64 winner together in 50 decisions. Larger local patches
therefore caused more dilution than rescue, and substantial common-mode
candidate-set error remains.

Radius one may be used only in a separately preregistered rescue study whose
success requires recovery against this full-R2 control; it is not an accepted
representation on its own. See
`docs/v2/reports/r3-action-edit-mlx-comparison-v1-result.md`.
