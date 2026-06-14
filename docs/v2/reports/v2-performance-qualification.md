# Cascadia V2 Performance Qualification

Verdict: **PASS**

Profile: `john1-m4-16gb-v1`

| Gate | Actual | Budget | Result |
|---|---:|---:|---|
| `instant-complete-game` | 0.403 seconds/game | <= 0.500 seconds/game | PASS |
| `instant-p90-decision` | 12.428 milliseconds | <= 25.000 milliseconds | PASS |
| `instant-p99-decision` | 29.040 milliseconds | <= 75.000 milliseconds | PASS |
| `interactive-complete-game` | 0.820 seconds/game | <= 1.250 seconds/game | PASS |
| `interactive-p90-decision` | 20.827 milliseconds | <= 50.000 milliseconds | PASS |
| `interactive-p99-decision` | 42.493 milliseconds | <= 100.000 milliseconds | PASS |
| `research-complete-game` | 6.995 seconds/game | <= 10.000 seconds/game | PASS |
| `research-p90-decision` | 362.236 milliseconds | <= 1000.000 milliseconds | PASS |
| `research-p99-decision` | 880.542 milliseconds | <= 3000.000 milliseconds | PASS |
| `mlx-batch32-throughput` | 75176.135 evaluations/second | >= 50000.000 evaluations/second | PASS |
| `mlx-batch32-p99` | 0.698 milliseconds | <= 1.000 milliseconds | PASS |

## Evidence

Budget contract SHA-256: `9691e43548d8907a0af8c68d37de5c1b1449cefc631a4bfc47ba35148058b9fd`

- `product`: `docs/v2/reports/pattern-aware-v1-confirm50.json` (`dcf286a441400d71e03faff8a1f92f1a614900b197c6911a7f1299daab927bb1`)
- `research`: `docs/v2/reports/late-conservative-base-policy-improvement-v1-t5-r8-c90-confirm50.json` (`effaeb0ab95d2dd42f80cbdb0153ae40353dfd44c61b5f03289e507f88298f1d`)
- `mlx`: `docs/v2/reports/legacy-nnue-v4opp-mlx-exact-csr-service-v1.json` (`efebcb919cc3430879840fb93f04329cfadd781966bdd17d81daaf31a7db0bf0`)
