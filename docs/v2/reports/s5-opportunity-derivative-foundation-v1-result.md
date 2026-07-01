# S5 Opportunity-Derivative Foundation V1 Result

Date: 2026-06-17

ADR: 0160

Classification: `s5_exact_opportunity_derivatives_promoted`

Outcome: passed

## Result

S5 passed exact replay, score-delta, schema, and normalization gates.

| Metric | Result |
|---|---:|
| Positions | 1,600 |
| Complete actions observed | 2,682,245 |
| Sampled actions | 102,400 |
| Replay checks / failures | 102,400 / 0 |
| Score-delta checks / failures | 102,400 / 0 |
| Feature fields | 154 |
| Nonzero fields | 150 |
| Observations per field | 102,400 |
| Total feature observations | 15,769,600 |
| Raw P99 scale ratio | 1,218x |

The four zero-observation fields were:

- `frontier.degree.0`;
- `motif.elk_line.0`;
- `motif.elk_line.4`; and
- `supply.destination_match.6`.

They remain in the fixed schema rather than being silently removed after
observing production data.

## Interpretation

The action derivative is a factual counterfactual surface, not a weighted
heuristic. It joins immediate score, habitat topology, wildlife opportunity,
future placement, semantic supply, and opponent denial in one exact schema.

The 1,218x raw P99 scale range demonstrates that an unnormalized concatenation
would be poorly conditioned. The emitted per-field contract gives MLX one
stable transform and divisor per feature without using labels.

## Decision

Authorize matched learned ablations:

1. no derivatives;
2. immediate score only;
3. topology and motif;
4. supply and opponent access;
5. all 154 fields.

Promotion requires aggregate and protected-slice R4800 improvement at matched
capacity and latency.

## Provenance

- bundle:
  `3d7f8a563319ba6636a3fb06d3db33fd846f8dbfdd70beeabd3d5a0d509eeee3`
- scientific report:
  `556f12e70728a29ef81ae9250becd6f3452462fbbe6abaa9bf567543fc3b9224`
- host: john4
- seeds: `5,410,000-5,410,019`

## Claim Boundary

This result authorizes MLX experiments; it is not itself a ranking or score
gain.
