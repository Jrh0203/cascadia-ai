//! Test-only boundary for trusted v1/v2 rule comparisons.
//!
//! Production v2 crates do not depend on v1. This empty crate owns only
//! integration tests over small fixtures whose expected outcomes are stated
//! independently from either implementation.

#[cfg(feature = "legacy-teacher")]
pub mod legacy_teacher;
