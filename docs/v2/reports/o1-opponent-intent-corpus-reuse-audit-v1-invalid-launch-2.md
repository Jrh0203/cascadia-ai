# O1 Corpus Reuse Audit v1: Invalid Launch 2

**Date:** 2026-06-17  
**Bundle:** `2d8403087f1d48b896f61701476a49cbaf19d4fe19c13bf2213fd182e33b7d67`  
**Classification:** Scientific execution complete; replay identity invalid

## What Ran

The john4 primary and john2 replay each reconstructed:

- 80 complete games;
- 6,400 exact sequential positions;
- 409,600 candidate actions;
- all selected actions and state transitions;
- unique tile identity and post-action survival windows;
- zero train/validation overlap on every preregistered key.

Both executions completed successfully and agreed on every measured count and
rate.

## Failure

The fail-closed classifier rejected:

```text
ClassificationError: primary and replay differ in datasets
```

The only dataset difference was the absolute input root
(`/Users/john4/...` versus `/Users/john2/...`). That host-local path was also
included in `scientific_blake3`, producing different digests despite identical
scientific results.

## Disposition

This launch does not authorize corpus reuse because crossed-host scientific
identity did not certify. ADR 0184 moves input roots into execution provenance,
increments the report schema, and namespaces all launch outputs by immutable
bundle ID. The unchanged audit is relaunched with `o1reuse-v3`.
