# ADR 0003: Symmetric Complete-Policy Benchmark

Status: Accepted  
Date: 2026-06-10

## Context

V1 benchmark paths differ in opponent strength, pre-move optimization, seat
aggregation, and scoring target. Results from different paths are not directly
comparable.

## Decision

Use `cascadia-aaaaa-4p-base-v1` as the primary protocol. Every seat executes the
same complete turn policy, including overflow and mulligan decisions. Capture
all four seat scores, exclude habitat bonuses from the primary score, use
separate deterministic RNG streams, and compare strategies on paired seeds.

## Consequences

- A benchmark result has one unambiguous meaning.
- Existing v1 commands are diagnostic until adapted.
- Final validation costs more but supports defensible claims.

