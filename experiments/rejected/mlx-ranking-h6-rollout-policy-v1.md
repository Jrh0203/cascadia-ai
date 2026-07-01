# MLX H6 Rollout Policy V1

## Hypothesis

The H6-trained ranker might be more valuable as the future-move policy inside
search than as a root selector, while batched inference could keep the cost
within an unrestricted local research budget.

## Runtime Gate

The preregistration required one complete K8/H6/R4/D4 game at no more than 120
treatment seconds before any ten-game strength pilot.

The qualified H6 ranker completed the full configuration with:

- Treatment wall time: 196.022 seconds
- Mean decision latency: 2,423 ms
- P90 decision latency: 4,097 ms
- Maximum decision latency: 10,405 ms
- Single-game paired delta: -3.5

## Conclusion

Rejected at the mandatory runtime gate. One MLX batch per branch frontier and
ply is correct but still too expensive at the full H6 breadth and four-sample
horizon. The ten-game pilot was not run. Future learned-search work should use
one leaf-value batch per decision or a materially cheaper distilled policy
representation.
