# Relational Substrate MLX Tournament V1 Result

## Verdict

**Valid experiment, no promotion.**

Final classification:
`relational_substrate_mlx_control_failed`

The original classifier was repaired under ADR 0169 so R6 identity is enforced
against each treatment's host-paired C0 replay rather than globally across
different machines. The repaired result is order-invariant and preserves every
frozen model-quality and treatment-serving threshold.

## Control

C0 exact R2:

| Metric | Result | Frozen gate |
|---|---:|---:|
| MAE | 1.4416 | <= 1.42 |
| RMSE | 1.8980 | <= 1.85 |
| Top-64 recall | 72.50% | >= 70% |
| Top-64 regret | 0.1046 | <= 0.12 |
| Low-supply recall | 82.46% | >= 88% |
| Independent-draft recall | 66.67% | >= 76% |
| Coverage | 98.33% | >= 97% |
| Combined P99 | 4,351.9 ms | reported, not a baseline qualification gate |
| Combined throughput | 2,207 actions/s | reported, not a baseline qualification gate |

C0 serving integrity passed. Its MAE, RMSE, low-supply recall, and
independent-draft recall missed the preregistered quality sanity thresholds.

## Treatments

| Arm | RMSE | Top-64 recall | Strategic recall | P99 | RSS | Relative throughput |
|---|---:|---:|---:|---:|---:|---:|
| Q1 R5 quotient-local | 1.9385 | 74.58% | 73.99% | 579.9 ms | 7.69 GB | 7.42x |
| G2 R5+S3 | 2.0264 | 72.50% | 73.19% | 510.5 ms | 8.27 GB | 7.86x |
| D3 R5+S3+S5 | **1.7875** | **81.67%** | **78.01%** | 572.7 ms | 8.33 GB | 7.28x |

### D3

D3 was the only treatment to pass every quality gate:

- RMSE improved by 0.1106;
- MAE improved by 0.0519;
- top-64 winner recall improved by 9.17 percentage points;
- strategic-opportunity recall improved by 5.00 points;
- low-supply recall improved by 5.26 points;
- independent-draft recall improved by 9.52 points;
- top-64 retained regret improved by 0.0423.

It also passed both relative material-efficiency gates. It failed absolute
serving because P99 exceeded 250 ms and RSS exceeded 4 GiB.

### Q1 and G2

Q1 failed coverage, strategic-mean improvement, and regret gates. G2 regressed
MAE, RMSE, regret, coverage, and several strategic checks. Both also failed
absolute P99 and RSS.

## Research Meaning

The relational representation contains useful signal: D3's quality lift is
large and broad, and all compact arms materially reduce latency relative to
exact C0. The current MLX implementation is not production-viable because its
resident memory is roughly double the cap and P99 remains about two times the
target.

The next representation work should preserve D3's component-motif and
opportunity-derivative information while replacing padded per-candidate
materialization with the exact preverified vectorized R2 path and a
memory-bounded serving layout.

No gameplay qualification or progress-to-100 claim is authorized.
