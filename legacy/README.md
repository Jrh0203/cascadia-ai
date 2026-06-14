# Cascadia V1 Reference

This directory contains the superseded v1 implementation and its historical
research material. It is retained only for trusted rule fixtures, differential
tests, UX archaeology, and independently reproduced baseline measurements.

- `crates/` contains the original rules, AI, CLI, and embedded web crates.
- `research/` contains historical scripts and overnight experiment material.
- `docs/` contains historical reports and proposals.

V2 production crates do not import v1. The test-only
`cascadia-differential` crate is the explicit compatibility boundary. Historical
weights and generated outputs remain ignored artifacts and are not promoted
into v2.

Run the retained v1 tests with:

```bash
cargo test -p cascadia-core -p cascadia-ai
```

Build the isolated legacy teacher only when reproducing a registered v1
baseline:

```bash
cargo build --release -p cascadia-differential \
  --features legacy-teacher --bin legacy-teacher
```
