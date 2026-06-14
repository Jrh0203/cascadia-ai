# Technical Debt

No known v2 technical debt is currently accepted.

The oversized command and research modules recorded here during ADR 0078 were
resolved without rebuilding the frozen collector:

- `cascadia-cli-v2/src/main.rs` is now typed parsing and dispatch only;
- command families own their data, model, policy, oracle, and report workflows;
- search owns lookahead, MLX value, ranking rollout, prefilter, prediction,
  policy-improvement, and oracle mechanisms in separate modules;
- simulation separates pattern strategies from finite-market opportunity math;
- large inline test suites live in dedicated child modules.

`python/tests/test_v2_source_structure.py` prevents the CLI entrypoint from
exceeding 300 lines and prevents any active v2 Rust production module from
exceeding 1,500 lines. New debt or unavoidable compromises must be documented
here with cause, proper fix, and blast radius before merge.
