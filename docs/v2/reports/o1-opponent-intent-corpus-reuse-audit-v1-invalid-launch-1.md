# O1 Corpus Reuse Audit v1: Invalid Launch 1

**Date:** 2026-06-16  
**Bundle:** `a861a5e5ec44fb14451c2749b6a17006696169516af88f09321ab0a72944cfd4`  
**Classification:** Invalid before scientific execution

## What Ran

- Immutable bundle fanout to john2 and john4: passed whole-tree parity.
- Train corpus fanout to john2 and john4: passed whole-tree parity.
- Validation corpus fanout to john2 and john4: passed whole-tree parity.
- john4 primary: exited after 0.24 seconds.
- john2 replay: exited after 0.38 seconds.

Neither run reached a trajectory, state comparison, action reconstruction, or
survival-label calculation.

## Failure

Both hosts reported:

```text
Error: Io(Os { code: 2, kind: NotFound, message: "No such file or directory" })
```

The read-only Rust dataset validator attempted to reopen the absolute
collector-local teacher weights path stored as provenance in `dataset.json`.
That file is intentionally absent on the remote hosts.

## Disposition

This is an infrastructure-invalid launch, not a negative scientific result.
ADR 0183 repairs the portability contract while retaining strict weight
validation for dataset creation and resume. The corrected launch uses a new
immutable bundle and `o1reuse-v2` task IDs.

