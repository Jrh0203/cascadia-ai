# ADR 0137: Corrected Historical Mid-Tail Schema

Status: accepted

Date: 2026-06-16

Schema ID: `legacy-mid-v4-fixed-v1`

Research-plan item: F5, Corrected V1 mid-tail closure

## Context

The completed F1 feature-schema activation census classified the production
`mid-features,v4-opp` NNUE layout as follows:

- indices `0..10561` are the live historical v2 base;
- indices `10561..10862` are an accidental 301-column prefix of the full-v3
  per-cell adjacency block;
- indices `10862..11231` are the live 369-column opponent-detail block.

The accidental 301-column block was zero in all 200,000 balanced historical
states. The intended 150 extended tile-terrain counts, 150 extended
tile-wildlife-capacity counts, and overflow-used bit were emitted at full-v3
indices near 44,959 and then removed by the mid-feature bound.

F1's final classification is
`feature_schema_activation_census_complete`, with classification scientific
BLAKE3
`f7f8559431f53a461f9464e14ef4cee2119cf3ddcf0bf4e3dd9126ab8bdd91fb`.

Changing extraction under the existing `mid-features,v4-opp` feature pair
would silently assign new meanings to trained first-layer rows. Keeping the
same 11,231-row width without a schema identifier would be equally ambiguous:
file size alone cannot distinguish the historical and corrected meanings.

## Decision

Introduce the cargo feature and schema:

```text
legacy-mid-v4-fixed-v1
```

The feature implies `mid-features` and `v4-opp`, but selects a new, dense
semantic layout:

| Range | Width | Meaning |
|---|---:|---|
| `0..10561` | 10,561 | Frozen historical v2 base |
| `10561..10930` | 369 | Migrated historical opponent detail |
| `10930..11080` | 150 | Extended tile-bag terrain counts |
| `11080..11230` | 150 | Extended tile-bag wildlife-capacity counts |
| `11230..11231` | 1 | Overflow-used bit |

The corrected schema remains 11,231 rows wide. It does not retain the dead
historical adjacency-prefix capacity.

The opponent rows move from historical range `10862..11231` to corrected
range `10561..10930`. Migration copies every row and neuron exactly. The new
301-row tail is zero-initialized.

## Existing-Build Compatibility

The existing layouts are frozen:

- default/full-v3 remains 45,260 rows;
- `mid-features` remains the historical 10,862-row layout;
- `mid-features,v4-opp` remains the historical 11,231-row champion layout;
- existing `NNUE` save bytes and extraction semantics are unchanged when the
  corrected feature is absent.

The corrected schema is incompatible with representation features that define
another complete or append-only layout:

- `legacy-features`;
- `v5-feat`;
- `czero-feat`;
- `v6-peak`;
- `cards-alt`;
- `cards-alt-v2`;
- `oppmarket-feat`; and
- `az-v2`.

Unsupported combinations fail at compile time.

Network-width features (`small-net` and `large-net`) and math backends remain
orthogonal. Their dimensions are recorded in the corrected weight header.

## Extraction Contract

Under `legacy-mid-v4-fixed-v1`:

1. the general extractor emits only the frozen v2 base below 10,561;
2. it emits opponent detail at 10,561;
3. it emits five extended terrain-count one-hots at 10,930;
4. it emits five extended wildlife-capacity one-hots at 11,080; and
5. it emits the overflow bit at 11,230 when active.

No full-v3 adjacency feature is emitted.

The optimized parent-afterstate context follows the same order. The corrected
tail is copied from the parent context because tile-bag composition and the
turn's overflow-use fact are invariant across candidate afterstates. The
historical context continues to reproduce the historical adjacency-prefix
behavior when the corrected feature is absent.

## Weight Container

Historical checkpoints retain the existing container:

```text
magic: NNUE
head-format version: u32
payload: inferred historical feature width
```

Corrected checkpoints use an explicit container:

| Byte range | Field |
|---|---|
| `0..4` | magic `NNUC` |
| `4..8` | corrected-container version, currently `1` |
| `8..12` | existing NNUE head-format version, `1..4` |
| `12..28` | schema tag `MIDTAIL-CORR-V1\0` |
| `28..32` | feature count |
| `32..36` | hidden-layer-1 width |
| `36..40` | hidden-layer-2 width |
| `40..` | existing NNUE tensor payload |

The loader rejects:

- unknown magic;
- unknown container or head versions;
- a mismatched schema tag;
- mismatched feature or hidden dimensions;
- malformed payload size;
- unread trailing bytes; and
- historical first-layer widths that do not identify a supported layout.

An old binary rejects `NNUC` rather than interpreting corrected rows as
historical rows.

## Deterministic Historical Migration

The corrected loader accepts these recognized historical `NNUE` layouts:

| Historical width | Migration |
|---:|---|
| 5,197 | Copy the legacy prefix; zero the remaining corrected rows |
| 5,566 | Copy 5,197 base rows and remap 369 opponent rows |
| 7,670 | Copy the v1 prefix; zero the remaining corrected rows |
| 10,561 | Copy the v2 base; zero opponent and corrected-tail rows |
| 10,862 | Copy the v2 base; discard the historical 301-row defect |
| 11,231 | Copy the v2 base, discard the defect, and remap 369 opponent rows |

For the 11,231-row champion migration:

```text
source 0..10561      -> destination 0..10561
source 10561..10862  -> discarded
source 10862..11231  -> destination 10561..10930
destination 10930..11231 -> zero
```

All biases, second-layer weights, value heads, policy head, split heads, and
heteroscedastic head are copied without semantic changes.

The explicit migration command is:

```bash
cargo run -p cascadia-ai \
  --example migrate_legacy_mid_v4_weights \
  --features legacy-mid-v4-fixed-v1 \
  -- historical.bin corrected.bin
```

The command accepts only a historical `NNUE` input and writes an `NNUC`
output. Loading a historical file directly in a corrected build applies the
same deterministic migration.

## Verification Contract

Permanent tests cover:

- every layout boundary and total width;
- nonzero activation of both corrected count blocks and the overflow bit;
- all `u8` source count values, including 30-bin saturation;
- every emitted feature remaining below `NUM_FEATURES`;
- opponent-before-tail extraction order;
- exact general-extractor and optimized-context parity across complete games;
- corrected save/load across all value-head versions;
- schema-tag corruption rejection;
- exact champion base-row preservation;
- exact opponent-row remapping;
- corrected-tail zero initialization;
- explicit migration output using the corrected container; and
- the unchanged historical champion layout when the new feature is absent.

## Consequences

The schema defect is closed without changing current player behavior.

The corrected schema is ready as a foundation, not as a strength claim. No
training or gameplay evaluation is authorized by this ADR. Future F5 work must
use the preregistered control and treatment, preserve the exact historical
control extractor, and satisfy the MLX parity gate before distributed neural
training begins.

