//! Proof-side machinery for a possible future dynamic-urn coupling.
//!
//! There is deliberately no `GameState` hook here.  Canonical bags are
//! private to `cascadia-game`, wildlife tokens do not yet have stable physical
//! IDs, and silently shadowing either fact would create a second rules engine.
//! This module specifies the admission obligations and exhaustively checks the
//! ideal finite priority-permutation oracle used by the proof sketch.

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const DYNAMIC_URN_PROOF_CONTRACT_ID: &str = "cascadiav3.rival_dynamic_urn_proof_contract.v1";
pub const MAX_EXHAUSTIVE_SMALL_URN_ITEMS: u8 = 8;

/// Conditions that must all be proved after a canonical chance hook exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DynamicUrnObligation {
    CanonicalUniquePhysicalItemIds,
    ExactConservationAcrossDrawWipeReturn,
    FreshEventIndexForEverySequentialDraw,
    ReturnedItemsNeverReuseObservedPriority,
    ConditionalUniformityForEveryReachableInventory,
    ExactHighLowAndActionBranchMarginals,
    PolicyCannotObservePhysicalIdentityOrPriority,
    IndependentWorldReferenceReplication,
}

impl DynamicUrnObligation {
    pub const ALL: [Self; 8] = [
        Self::CanonicalUniquePhysicalItemIds,
        Self::ExactConservationAcrossDrawWipeReturn,
        Self::FreshEventIndexForEverySequentialDraw,
        Self::ReturnedItemsNeverReuseObservedPriority,
        Self::ConditionalUniformityForEveryReachableInventory,
        Self::ExactHighLowAndActionBranchMarginals,
        Self::PolicyCannotObservePhysicalIdentityOrPriority,
        Self::IndependentWorldReferenceReplication,
    ];
}

/// Pre-hook status is fail-closed by construction.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "DynamicUrnAdmissionWire", into = "DynamicUrnAdmissionWire")]
pub struct DynamicUrnAdmissionStatus {
    contract_id: String,
    admitted: bool,
    unresolved: Vec<DynamicUrnObligation>,
    reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct DynamicUrnAdmissionWire {
    contract_id: String,
    admitted: bool,
    unresolved: Vec<DynamicUrnObligation>,
    reason: String,
}

impl DynamicUrnAdmissionStatus {
    pub fn pre_hook() -> Self {
        Self {
            contract_id: DYNAMIC_URN_PROOF_CONTRACT_ID.to_owned(),
            admitted: false,
            unresolved: DynamicUrnObligation::ALL.to_vec(),
            reason: "canonical chance hook and physical item identities are intentionally absent"
                .to_owned(),
        }
    }

    pub fn contract_id(&self) -> &str {
        &self.contract_id
    }

    pub const fn admitted(&self) -> bool {
        self.admitted
    }

    pub fn unresolved(&self) -> &[DynamicUrnObligation] {
        &self.unresolved
    }

    pub fn reason(&self) -> &str {
        &self.reason
    }

    pub fn require_admitted(&self) -> Result<(), DynamicUrnError> {
        if self.admitted && self.unresolved.is_empty() {
            Ok(())
        } else {
            Err(DynamicUrnError::NotAdmitted {
                unresolved: self.unresolved.clone(),
            })
        }
    }
}

impl From<DynamicUrnAdmissionStatus> for DynamicUrnAdmissionWire {
    fn from(value: DynamicUrnAdmissionStatus) -> Self {
        Self {
            contract_id: value.contract_id,
            admitted: value.admitted,
            unresolved: value.unresolved,
            reason: value.reason,
        }
    }
}

impl TryFrom<DynamicUrnAdmissionWire> for DynamicUrnAdmissionStatus {
    type Error = DynamicUrnError;

    fn try_from(value: DynamicUrnAdmissionWire) -> Result<Self, Self::Error> {
        let observed = Self {
            contract_id: value.contract_id,
            admitted: value.admitted,
            unresolved: value.unresolved,
            reason: value.reason,
        };
        if observed == Self::pre_hook() {
            Ok(observed)
        } else {
            Err(DynamicUrnError::UnrecognizedAdmissionRecord)
        }
    }
}

/// Exact enumeration summary for ideal priorities on every nonempty subset.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "SmallUrnOracleWire", into = "SmallUrnOracleWire")]
pub struct SmallUrnOracleReport {
    contract_id: String,
    item_count: u8,
    eligible_subsets_checked: u64,
    restricted_priority_permutations_checked: u64,
    first_draw_cells_checked: u64,
    ordered_two_draw_cells_checked: u64,
    exact: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SmallUrnOracleWire {
    contract_id: String,
    item_count: u8,
    eligible_subsets_checked: u64,
    restricted_priority_permutations_checked: u64,
    first_draw_cells_checked: u64,
    ordered_two_draw_cells_checked: u64,
    exact: bool,
}

impl SmallUrnOracleReport {
    pub fn item_count(&self) -> u8 {
        self.item_count
    }

    pub fn eligible_subsets_checked(&self) -> u64 {
        self.eligible_subsets_checked
    }

    pub fn restricted_priority_permutations_checked(&self) -> u64 {
        self.restricted_priority_permutations_checked
    }

    pub fn first_draw_cells_checked(&self) -> u64 {
        self.first_draw_cells_checked
    }

    pub fn ordered_two_draw_cells_checked(&self) -> u64 {
        self.ordered_two_draw_cells_checked
    }

    pub fn exact(&self) -> bool {
        self.exact
    }
}

impl From<SmallUrnOracleReport> for SmallUrnOracleWire {
    fn from(value: SmallUrnOracleReport) -> Self {
        Self {
            contract_id: value.contract_id,
            item_count: value.item_count,
            eligible_subsets_checked: value.eligible_subsets_checked,
            restricted_priority_permutations_checked: value
                .restricted_priority_permutations_checked,
            first_draw_cells_checked: value.first_draw_cells_checked,
            ordered_two_draw_cells_checked: value.ordered_two_draw_cells_checked,
            exact: value.exact,
        }
    }
}

impl TryFrom<SmallUrnOracleWire> for SmallUrnOracleReport {
    type Error = DynamicUrnError;

    fn try_from(value: SmallUrnOracleWire) -> Result<Self, Self::Error> {
        let observed = Self {
            contract_id: value.contract_id,
            item_count: value.item_count,
            eligible_subsets_checked: value.eligible_subsets_checked,
            restricted_priority_permutations_checked: value
                .restricted_priority_permutations_checked,
            first_draw_cells_checked: value.first_draw_cells_checked,
            ordered_two_draw_cells_checked: value.ordered_two_draw_cells_checked,
            exact: value.exact,
        };
        let expected = verify_small_urn_priority_oracle(observed.item_count)?;
        if observed == expected {
            Ok(observed)
        } else {
            Err(DynamicUrnError::UnrecognizedOracleReport)
        }
    }
}

/// Exhaustively prove uniform first and ordered-second draws for all eligible
/// subsets of `0..item_count` under an ideal uniformly random priority order.
///
/// Enumerating restricted permutations is equivalent to enumerating global
/// priority permutations but avoids repeating every restricted order
/// `(n-k)! * C(n,k)` times.  For a subset of size `k`, each first item must
/// occur `(k-1)!` times and each ordered first/second pair `(k-2)!` times.
pub fn verify_small_urn_priority_oracle(
    item_count: u8,
) -> Result<SmallUrnOracleReport, DynamicUrnError> {
    if item_count == 0 || item_count > MAX_EXHAUSTIVE_SMALL_URN_ITEMS {
        return Err(DynamicUrnError::UnsupportedItemCount {
            item_count,
            maximum: MAX_EXHAUSTIVE_SMALL_URN_ITEMS,
        });
    }

    let mut subsets_checked = 0u64;
    let mut permutations_checked = 0u64;
    let mut first_cells_checked = 0u64;
    let mut pair_cells_checked = 0u64;
    for mask in 1u16..(1u16 << item_count) {
        let members: Vec<u8> = (0..item_count)
            .filter(|item| mask & (1u16 << item) != 0)
            .collect();
        let k = members.len();
        let mut first_counts = [0u32; MAX_EXHAUSTIVE_SMALL_URN_ITEMS as usize];
        let mut pair_counts = [[0u32; MAX_EXHAUSTIVE_SMALL_URN_ITEMS as usize];
            MAX_EXHAUSTIVE_SMALL_URN_ITEMS as usize];
        let mut local_permutations = 0u64;
        for_each_permutation(&members, |permutation| {
            local_permutations += 1;
            first_counts[permutation[0] as usize] += 1;
            if permutation.len() >= 2 {
                pair_counts[permutation[0] as usize][permutation[1] as usize] += 1;
            }
        });
        let expected_permutations = factorial(k as u8);
        if local_permutations != expected_permutations {
            return Err(DynamicUrnError::EnumerationMismatch);
        }
        let expected_first = factorial((k - 1) as u8) as u32;
        for &item in &members {
            if first_counts[item as usize] != expected_first {
                return Err(DynamicUrnError::NonUniformFirstDraw {
                    subset_mask: mask,
                    item,
                    observed: first_counts[item as usize],
                    expected: expected_first,
                });
            }
            first_cells_checked += 1;
        }
        if k >= 2 {
            let expected_pair = factorial((k - 2) as u8) as u32;
            for &first in &members {
                for &second in &members {
                    if first == second {
                        continue;
                    }
                    if pair_counts[first as usize][second as usize] != expected_pair {
                        return Err(DynamicUrnError::NonUniformOrderedPair {
                            subset_mask: mask,
                            first,
                            second,
                            observed: pair_counts[first as usize][second as usize],
                            expected: expected_pair,
                        });
                    }
                    pair_cells_checked += 1;
                }
            }
        }
        subsets_checked += 1;
        permutations_checked += local_permutations;
    }

    Ok(SmallUrnOracleReport {
        contract_id: DYNAMIC_URN_PROOF_CONTRACT_ID.to_owned(),
        item_count,
        eligible_subsets_checked: subsets_checked,
        restricted_priority_permutations_checked: permutations_checked,
        first_draw_cells_checked: first_cells_checked,
        ordered_two_draw_cells_checked: pair_cells_checked,
        exact: true,
    })
}

fn factorial(value: u8) -> u64 {
    (1..=u64::from(value)).product()
}

fn for_each_permutation(values: &[u8], mut visit: impl FnMut(&[u8])) {
    fn recurse(values: &mut [u8], index: usize, visit: &mut impl FnMut(&[u8])) {
        if index == values.len() {
            visit(values);
            return;
        }
        for swap_index in index..values.len() {
            values.swap(index, swap_index);
            recurse(values, index + 1, visit);
            values.swap(index, swap_index);
        }
    }

    let mut owned = values.to_vec();
    recurse(&mut owned, 0, &mut visit);
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum DynamicUrnError {
    #[error(
        "dynamic-urn admission cannot be asserted in v1; only the canonical denied record exists"
    )]
    UnrecognizedAdmissionRecord,
    #[error("dynamic-urn coupling is not admitted; unresolved obligations: {unresolved:?}")]
    NotAdmitted {
        unresolved: Vec<DynamicUrnObligation>,
    },
    #[error("small-urn oracle supports 1..={maximum} items, got {item_count}")]
    UnsupportedItemCount { item_count: u8, maximum: u8 },
    #[error("priority permutation enumerator did not visit the expected factorial count")]
    EnumerationMismatch,
    #[error("small-urn oracle report does not match exhaustive Rust recomputation")]
    UnrecognizedOracleReport,
    #[error(
        "nonuniform first draw for subset {subset_mask:#x}, item {item}: {observed} != {expected}"
    )]
    NonUniformFirstDraw {
        subset_mask: u16,
        item: u8,
        observed: u32,
        expected: u32,
    },
    #[error(
        "nonuniform ordered pair for subset {subset_mask:#x}, ({first},{second}): {observed} != {expected}"
    )]
    NonUniformOrderedPair {
        subset_mask: u16,
        first: u8,
        second: u8,
        observed: u32,
        expected: u32,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pre_hook_coupling_fails_closed() {
        let status = DynamicUrnAdmissionStatus::pre_hook();
        assert!(!status.admitted());
        assert_eq!(status.unresolved(), DynamicUrnObligation::ALL);
        assert!(matches!(
            status.require_admitted(),
            Err(DynamicUrnError::NotAdmitted { .. })
        ));
    }

    #[test]
    fn admitted_status_cannot_be_forged_through_the_wire() {
        let mut value = serde_json::to_value(DynamicUrnAdmissionStatus::pre_hook()).unwrap();
        value["admitted"] = serde_json::json!(true);
        value["unresolved"] = serde_json::json!([]);
        assert!(serde_json::from_value::<DynamicUrnAdmissionStatus>(value).is_err());

        let canonical = serde_json::to_vec(&DynamicUrnAdmissionStatus::pre_hook()).unwrap();
        assert!(serde_json::from_slice::<DynamicUrnAdmissionStatus>(&canonical).is_ok());
    }

    #[test]
    fn every_subset_and_priority_order_through_eight_items_is_exact() {
        let report = verify_small_urn_priority_oracle(8).unwrap();
        assert!(report.exact);
        assert_eq!(report.eligible_subsets_checked, 255);
        // Sum_{k=1}^8 C(8,k) k! = 109,600 restricted orders.
        assert_eq!(report.restricted_priority_permutations_checked, 109_600);
        assert!(report.first_draw_cells_checked > 0);
        assert!(report.ordered_two_draw_cells_checked > report.first_draw_cells_checked);
    }

    #[test]
    fn oracle_rejects_nonexhaustive_sizes() {
        assert!(matches!(
            verify_small_urn_priority_oracle(0),
            Err(DynamicUrnError::UnsupportedItemCount { .. })
        ));
        assert!(matches!(
            verify_small_urn_priority_oracle(9),
            Err(DynamicUrnError::UnsupportedItemCount { .. })
        ));
    }

    #[test]
    fn exact_oracle_report_cannot_be_forged_through_the_wire() {
        let report = verify_small_urn_priority_oracle(4).unwrap();
        let bytes = serde_json::to_vec(&report).unwrap();
        assert_eq!(
            serde_json::from_slice::<SmallUrnOracleReport>(&bytes).unwrap(),
            report
        );

        let mut value = serde_json::to_value(&report).unwrap();
        value["eligible_subsets_checked"] = serde_json::json!(1);
        assert!(serde_json::from_value::<SmallUrnOracleReport>(value).is_err());
    }
}
