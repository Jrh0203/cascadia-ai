# Experiments

Experiments are registered before execution in `registry.toml`.

Each entry must define:

- hypothesis,
- baseline and treatment,
- exact benchmark protocol and seed suite,
- primary metric and promotion threshold,
- maximum compute or stopping rule,
- code/config/model/data artifact IDs,
- final status and result artifacts.

Smoke tests may be recorded but cannot promote production behavior. Historical
v1 reports are not copied into this registry.

