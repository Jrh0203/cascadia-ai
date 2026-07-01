# ADR 0001: Build V2 Without V1 Dependencies

Status: Accepted  
Date: 2026-06-10

## Context

V1 mixes rules, search, neural architectures, experiments, feature flags,
environment configuration, and product orchestration. The worktree contains
valuable tests and behavior but also substantial accidental compatibility
constraints.

## Decision

Build new v2 crates alongside v1 with no dependency from v2 to v1. Use v1 only
through fixtures, replay adapters, and benchmark opponents. After v2 reaches
rules and benchmark parity, move v1 under `legacy/` and promote v2 packages to
canonical names.

## Consequences

- V2 can define transactional actions and typed configuration cleanly.
- Historical models and formats do not constrain v2 APIs.
- During migration, some rule logic exists twice.
- Differential tests must distinguish independently verified behavior from
  merely matching v1.

