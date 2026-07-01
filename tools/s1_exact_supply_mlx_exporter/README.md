# S1 Exact-Supply MLX Exporter

This standalone Rust tool builds the content-addressed public-information
sidecar consumed by the ADR 0147 MLX comparison. It deliberately remains
outside the root Cargo workspace so the experiment can use the accepted
`cascadia-data` semantic-supply implementation without editing shared source
or workspace metadata.

The exporter:

- validates the existing open complete-action graded-oracle datasets;
- reconstructs exact semantic supply from public `PositionRecord` rows;
- round-trips canonical `CSSSUP1` bytes through `ExactSemanticSupply`;
- proves parity with the frozen 30-value legacy supply;
- exports exact staged wildlife counts, selected archetype IDs, and
  rotation-aware frontier requirements for every complete legal action;
- records no hidden stack order, excluded-tile identity, future refill, test
  row, or gameplay result; and
- writes raw little-endian tensors plus a content-addressed `cache.json`.

The learned experiment ID is
`exact-semantic-supply-learned-comparison-v1`. Its cache also embeds the frozen
ADR 0143 factual collision: physical pools `[0, 23]` and `[2, 20]` have equal
30-value marginals but semantic archetypes `[26, 72]` and `[24, 74]` induce
different refill laws.

The MLX comparison keeps the 30 marginals as C0. Exact candidate archetype and
frontier facts are exported for binding and replay, but the Python adapter
zeros them for C0 and T1; only T2 may consume that relational signal.

Build and test:

```bash
cargo test --manifest-path tools/s1_exact_supply_mlx_exporter/Cargo.toml
cargo build --release \
  --manifest-path tools/s1_exact_supply_mlx_exporter/Cargo.toml
```

The production cache command is intentionally documented in the ADR 0147
preregistration. A bounded `--max-groups-per-split` mode exists only for local
tooling smokes; its manifest is marked incomplete and production preflight
rejects it.

The reviewed entry point is
`tools/s1_exact_supply_mlx_campaign.py export-cache`. It verifies the immutable
bundle, runs the bundle's exact exporter binary, reloads the content-addressed
cache, and records that neither training nor queue mutation occurred.
