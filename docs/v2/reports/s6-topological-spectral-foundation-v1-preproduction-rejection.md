# S6 Topological And Spectral Foundation V1 Preproduction Rejection

Date: 2026-06-17

ADR: 0164

Experiment: `s6-topological-spectral-foundation-v1`

Protocol: `s6-exact-topology-collision-census-v1`

Status: rejected before production

## Result

V1 produced no admissible production evidence. Its excluded calibration did
its job: it found one implementation defect and two protocol defects before
the immutable bundle or four-host production wave existed.

## Calibration 1

One excluded game initially classified as authorized. Inspection showed that:

- every individual S6 family separated zero current-S3 collision pairs; but
- the complete encoding appeared to separate all 470 pairs.

The discrepancy came from `relative_seat` being present only in the complete
encoding hash. Identical boards in different relative seat slots were falsely
counted as novel. The seat field was removed from the scientific novelty
surface and a regression test was added.

This calibration is invalid as scientific evidence and is retained only as a
debugging artifact.

## Corrected Calibration

The corrected release candidate ran ten excluded games on john2:

| Metric | Result |
|---|---:|
| Positions | 800 |
| Board encodings | 3,200 |
| Exact topology checks | 35,200 / 35,200 |
| D6 checks | 9,600 / 9,600 |
| Adversarial checks | 4 / 4 |
| Current-S3 collision pairs | 5,677 |
| Pairs separated by any S6 family | 0 |
| Long-range boards | 913 |
| Boards with geometric holes | 173 |
| Median encoding | 1,681 bytes |
| Median extraction | 469 microseconds |
| P99 extraction under ten-way contention | 3.132 milliseconds |

The frozen V1 classification was `s6_corpus_novelty_futile`.

## Why V1 Does Not Proceed

The result does not show that S6 features are constant or useless. It shows
that the accepted S3 scalar signature had no nontrivial natural collision in
this calibration block. Requiring 128 separated exact collisions therefore
tests collision prevalence rather than feature information or learned value.

The latency miss is also not a serving result. Ten games were executing
feature extraction simultaneously on ten CPU cores, so a per-call P99 includes
cross-thread contention. The one-game calibration had a 0.998 millisecond P99,
confirming that the protocol mixed two different latency domains.

## Decision

- Do not launch V1 production.
- Preserve both calibration reports and collection receipts.
- Keep the corrected seat-independent novelty hash.
- Version the protocol before changing any gate.
- In V2, require feature-family variation on natural boards and measure
  serving cost in a separate isolated single-thread probe.
- Leave predictive utility to the preregistered matched MLX ablation.

## Artifacts

- `artifacts/experiments/s6-topological-spectral-foundation-v1/calibration/calibration-seed-5600000.json`
- `artifacts/experiments/s6-topological-spectral-foundation-v1/calibration/calibration-10g-v2.json`
- `artifacts/experiments/s6-topological-spectral-foundation-v1/calibration/collection.json`
- `artifacts/experiments/s6-topological-spectral-foundation-v1/calibration/collection-v2.json`
