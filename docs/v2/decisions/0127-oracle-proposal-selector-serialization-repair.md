# ADR 0127: Oracle-Proposal Selector Serialization Repair

Status: accepted

Date: 2026-06-16

## Context

ADR 0126 launched four preregistered complete-action selector arms. The
`wide-concat` and `pairwise-gated` arms completed all 20 training epochs and
their single validation evaluation, then failed while constructing the
scientific report hash. Phase and subset coverage gates contained NumPy
boolean scalars. Python's standard JSON encoder rejected those values before
`report.json` could be written.

The failure occurred after model fitting and evaluation. It did not alter the
frozen caches, targets, seeds, architectures, checkpoint selection, metrics,
or validation-open count.

## Decision

Normalize NumPy scalar and array values only at JSON serialization boundaries.
The repair applies to atomic reports, JSONL metrics, scientific hashes, and CLI
output. Unsupported objects continue to raise `TypeError`.

Retry failed ADR 0126 arms with their original commands, frozen inputs, seeds,
architectures, epoch budgets, and success gates. Preserve the failed attempts
in the queue ledger. Re-run source identity collection after the repair before
the final classification.

## Consequences

- Scientific computation is unchanged.
- JSON artifacts contain ordinary JSON booleans, numbers, and arrays.
- The original preregistration snapshots remain immutable.
- ADR 0126 records both the preregistered source identity and this explicit
  post-launch implementation repair.
- A regression test covers NumPy booleans, integers, floats, and arrays.
