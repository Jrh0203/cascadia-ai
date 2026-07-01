# R4 Bounded Far-Quotient Foundation V1 Preregistration

Date: 2026-06-17

Experiment ID: `r4-bounded-far-quotient-foundation-v1`

Contract: ADR 0155

Status: completed; foundation passed

## Question

Can the accepted exact radius-four R4 state retain habitat, wildlife,
frontier, overflow, and opponent distinctions while replacing unbounded far
signature streams with a hard-bounded model view?

## Frozen Arms

| Arm | Host | Wildlife quotient | Frontier quotient | Hard max |
|---|---|---|---|---:|
| `q1-seat-marginal` | john1 | 20 seat/species | 4 seat | 184 |
| `q2-directional` | john2 | 20 seat/species | 24 seat/sector | 204 |
| `q3-affordance` | john3 | 20 seat/species | 20 seat/terrain | 200 |
| `q4-selective-exact` | john4 | 20 residual plus 16 exact | 4 residual plus 24 exact | 224 |

Every arm also includes all 61 near cells, all far habitat components, and all
far wildlife components. The exact `CSR4AM1` sidecar is unchanged.

## Predictions

1. Q1 should be the fastest and smallest but is most exposed to relational
   aliases.
2. Q2 should best preserve directional frontier structure.
3. Q3 should best preserve habitat-growth and bridge affordances.
4. Q4 should preserve the richest rare signatures while remaining hard
   bounded.
5. At least one arm should distinguish all seven registered pairs with P99 at
   most 192 tokens.

Q4 admits only complete orientation-invariant priority tie groups. It never
uses raw sector, edge, local-index, source order, or orientation-sensitive
bytes to cut a group at the 16- or 24-token ceiling.

These are prospective predictions. No threshold changes after the production
reports are opened.

## Exactness

For every record:

- construct the accepted radius-four R4 state;
- round-trip the exact codec and require R2 semantic equality;
- construct the bounded arm twice and require identical bytes;
- mutate targets and require identical exact and bounded bytes;
- transform and inverse through all twelve D6 elements;
- require exact source-bucket accounting; and
- use checked counters and strict bounded-envelope decoding.

Distance histograms use exact bins 0 through 14 and one explicit
`distance >= 15` tail, accompanied by exact minimum, maximum, total mass, and
count-weighted distance sum.

## Size And Runtime Gates

Each arm must satisfy:

```text
max tokens <= 224
P99 tokens <= 192
max active scalars <= 16,384
max padded scalar slots <= 24,576
max canonical bytes <= 65,536
paired view throughput >= 0.90x full R4 HWF
```

Token count, primitive width, bytes, and runtime are independent gates.

## Corpus And Parallelism

All four hosts process the same accepted 60,000-position corpus, one distinct
arm per host. All hosts run the complete adversarial preflight first. The
aggregate rejects missing arms, source-stream mismatch, dataset drift,
duplicate arm reports, and input-order dependence.

## Claim Boundary

Passing can establish a mechanically exact, information-probed,
hard-bounded model view and authorize a matched MLX comparison.

It cannot establish learned ranking quality, latency superiority, gameplay
strength, or progress toward the 100-point mean target.

## Result

The frozen production campaign completed without changing the arms or gates.
Q1, Q2, and Q3 passed; Q4 failed only the P99 token gate at 206 against 192.
The aggregate classification is
`r4_bounded_quotient_foundation_passed`, authorizing the exact R2 plus
Q1/Q2/Q3 matched MLX comparison. Full evidence is in
`r4-bounded-far-quotient-foundation-v1-result.md`.
