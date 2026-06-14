# Cascadia AI V2

Follow [`AGENTS.md`](AGENTS.md) and the authoritative
[`CASCADIA_V2_GOAL.txt`](CASCADIA_V2_GOAL.txt).

Active production code lives in `crates/cascadia-*`, `python/cascadia_mlx`,
and `apps/web`. Superseded v1 source and historical research live under
[`legacy/`](legacy/README.md) and must not drive v2 conclusions without fresh
reproduction.

Use:

```bash
make setup
make check
make benchmark-smoke
make web-dev
```

All neural training and Apple-specific inference use MLX. Substantive
experiments require a preregistered protocol, disjoint deterministic splits,
checksummed artifacts, and registry updates.
