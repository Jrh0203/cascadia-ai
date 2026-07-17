//! Typed unilateral ceiling-tomography evidence.
//!
//! The type system keeps an executable witness, a heuristic best found, an
//! exact optimum, and a certified interval from being reported as the same
//! thing.  This module stores evidence; it does not turn proxy CPU policies
//! into high-fidelity incumbent claims.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::Sha256Digest;

pub const TOMOGRAPHY_RESULT_SCHEMA_ID: &str = "cascadiav3.rival_tomography_result.v1";

/// Namespace prefix that every non-proxy (incumbent-measured) policy
/// identity must carry.  The prefix is an explicit declaration, not a
/// heuristic: a result can only enter the
/// [`TomographyEvidenceDomain::IncumbentMeasured`] domain when its input
/// manifest declared a policy identity inside this namespace, and a
/// `cpu_proxy` result can never carry one.  Mislabeling in either direction
/// fails closed at construction and at deserialization.
pub const INCUMBENT_POLICY_NAMESPACE: &str = "incumbent:";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TomographyKind {
    T0OwnBoardRepack,
    T1PublicOneSeatWitness,
    T2LateGameBestResponse,
    T3KnownWorldOneSeatOracle,
    T4ResourceRelaxedBound,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InformationBoundary {
    AcquiredResourcesOnly,
    PublicPolicyOnly,
    PublicPolicyWithIntegratedChance,
    KnownExogenousChanceTape,
    OptimisticResourceSuperset,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TomographyEvidenceDomain {
    /// CPU pattern-policy plumbing only; structurally non-funding.
    CpuProxy,
    /// Measurements over games produced by a declared non-proxy incumbent
    /// policy.  Admission requires the incumbent policy identity to live in
    /// the [`INCUMBENT_POLICY_NAMESPACE`]; proxy-population diagnostics can
    /// therefore never be presented as incumbent measurements.
    IncumbentMeasured,
}

impl TomographyEvidenceDomain {
    /// Fail-closed policy-identity admission for this evidence domain.
    ///
    /// `IncumbentMeasured` requires a non-empty identity inside
    /// [`INCUMBENT_POLICY_NAMESPACE`]; `CpuProxy` rejects identities inside
    /// that namespace so a proxy result cannot masquerade in either
    /// direction.
    pub fn admits_incumbent_policy_id(self, incumbent_policy_id: &str) -> bool {
        let namespaced = incumbent_policy_id
            .strip_prefix(INCUMBENT_POLICY_NAMESPACE)
            .is_some_and(|rest| !rest.trim().is_empty());
        match self {
            Self::CpuProxy => !incumbent_policy_id.starts_with(INCUMBENT_POLICY_NAMESPACE),
            Self::IncumbentMeasured => namespaced,
        }
    }
}

/// Population labeling shared by every tomography result of one harness run:
/// which policy produced the input games, which table population opposed it,
/// and the evidence domain those identities admit.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TomographyPopulation {
    pub incumbent_policy_id: String,
    pub opponent_population_id: String,
    pub evidence_domain: TomographyEvidenceDomain,
}

impl TomographyPopulation {
    pub fn validate(&self) -> Result<(), TomographyError> {
        if self.incumbent_policy_id.trim().is_empty()
            || self.opponent_population_id.trim().is_empty()
        {
            return Err(TomographyError::EmptyIdentity);
        }
        if !self
            .evidence_domain
            .admits_incumbent_policy_id(&self.incumbent_policy_id)
        {
            return Err(TomographyError::EvidenceDomainPolicyMismatch);
        }
        Ok(())
    }
}

impl TomographyKind {
    pub const fn required_information_boundary(self) -> InformationBoundary {
        match self {
            Self::T0OwnBoardRepack => InformationBoundary::AcquiredResourcesOnly,
            Self::T1PublicOneSeatWitness => InformationBoundary::PublicPolicyOnly,
            Self::T2LateGameBestResponse => InformationBoundary::PublicPolicyWithIntegratedChance,
            Self::T3KnownWorldOneSeatOracle => InformationBoundary::KnownExogenousChanceTape,
            Self::T4ResourceRelaxedBound => InformationBoundary::OptimisticResourceSuperset,
        }
    }
}

/// Evidence with explicit epistemic strength.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "grade", rename_all = "snake_case", deny_unknown_fields)]
pub enum TomographyEvidence {
    /// Optimality and value both certified for the registered finite problem.
    Exact {
        score_delta: i32,
        optimality_certificate_sha256: Sha256Digest,
    },
    /// A legal executable value; never an optimality or ceiling claim.
    BestFound {
        score_delta: i32,
        solver_config_sha256: Sha256Digest,
        witness_ledger_sha256: Sha256Digest,
        explored_nodes: u64,
    },
    /// A proof interval.  A point at either endpoint is not implied reachable.
    CertifiedBounds {
        lower_score_delta: i32,
        upper_score_delta: i32,
        bound_certificate_sha256: Sha256Digest,
    },
}

impl TomographyEvidence {
    pub const fn lower_bound(&self) -> i32 {
        match self {
            Self::Exact { score_delta, .. } | Self::BestFound { score_delta, .. } => *score_delta,
            Self::CertifiedBounds {
                lower_score_delta, ..
            } => *lower_score_delta,
        }
    }

    pub const fn upper_bound(&self) -> Option<i32> {
        match self {
            Self::Exact { score_delta, .. } => Some(*score_delta),
            Self::BestFound { .. } => None,
            Self::CertifiedBounds {
                upper_score_delta, ..
            } => Some(*upper_score_delta),
        }
    }

    pub const fn is_exact(&self) -> bool {
        matches!(self, Self::Exact { .. })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "TomographyWire", into = "TomographyWire")]
pub struct TomographyResult {
    schema_id: String,
    kind: TomographyKind,
    information_boundary: InformationBoundary,
    root_id: Sha256Digest,
    source_game_id: String,
    acting_seat: u8,
    incumbent_policy_id: String,
    opponent_population_id: String,
    evidence: TomographyEvidence,
    natural_frequency_weight_numerator: u64,
    natural_frequency_weight_denominator: u64,
    evidence_domain: TomographyEvidenceDomain,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct TomographyWire {
    schema_id: String,
    kind: TomographyKind,
    information_boundary: InformationBoundary,
    root_id: Sha256Digest,
    source_game_id: String,
    acting_seat: u8,
    incumbent_policy_id: String,
    opponent_population_id: String,
    evidence: TomographyEvidence,
    natural_frequency_weight_numerator: u64,
    natural_frequency_weight_denominator: u64,
    evidence_domain: TomographyEvidenceDomain,
}

/// Complete constructor input; validation is centralized in `try_new`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TomographyResultInput {
    pub kind: TomographyKind,
    pub root_id: Sha256Digest,
    pub source_game_id: String,
    pub acting_seat: u8,
    pub incumbent_policy_id: String,
    pub opponent_population_id: String,
    pub evidence: TomographyEvidence,
    pub natural_frequency_weight_numerator: u64,
    pub natural_frequency_weight_denominator: u64,
}

impl TomographyResult {
    pub fn try_new(input: TomographyResultInput) -> Result<Self, TomographyError> {
        Self::try_new_in_domain(input, TomographyEvidenceDomain::CpuProxy)
    }

    /// Constructs a result in an explicit evidence domain.  The domain must
    /// admit the incumbent policy identity
    /// ([`TomographyEvidenceDomain::admits_incumbent_policy_id`]); every
    /// other validation rule of [`Self::try_new`] applies unchanged.
    pub fn try_new_in_domain(
        input: TomographyResultInput,
        evidence_domain: TomographyEvidenceDomain,
    ) -> Result<Self, TomographyError> {
        let result = Self {
            schema_id: TOMOGRAPHY_RESULT_SCHEMA_ID.to_owned(),
            information_boundary: input.kind.required_information_boundary(),
            kind: input.kind,
            root_id: input.root_id,
            source_game_id: input.source_game_id,
            acting_seat: input.acting_seat,
            incumbent_policy_id: input.incumbent_policy_id,
            opponent_population_id: input.opponent_population_id,
            evidence: input.evidence,
            natural_frequency_weight_numerator: input.natural_frequency_weight_numerator,
            natural_frequency_weight_denominator: input.natural_frequency_weight_denominator,
            evidence_domain,
        };
        result.validate()?;
        Ok(result)
    }

    pub fn validate(&self) -> Result<(), TomographyError> {
        if self.schema_id != TOMOGRAPHY_RESULT_SCHEMA_ID {
            return Err(TomographyError::WrongSchema);
        }
        if self.information_boundary != self.kind.required_information_boundary() {
            return Err(TomographyError::InformationBoundaryMismatch);
        }
        if self.acting_seat >= 4 {
            return Err(TomographyError::InvalidSeat(self.acting_seat));
        }
        if self.source_game_id.trim().is_empty()
            || self.incumbent_policy_id.trim().is_empty()
            || self.opponent_population_id.trim().is_empty()
        {
            return Err(TomographyError::EmptyIdentity);
        }
        if !self
            .evidence_domain
            .admits_incumbent_policy_id(&self.incumbent_policy_id)
        {
            return Err(TomographyError::EvidenceDomainPolicyMismatch);
        }
        if self.natural_frequency_weight_denominator == 0
            || self.natural_frequency_weight_numerator > self.natural_frequency_weight_denominator
        {
            return Err(TomographyError::InvalidNaturalFrequencyWeight);
        }
        if let TomographyEvidence::CertifiedBounds {
            lower_score_delta,
            upper_score_delta,
            ..
        } = self.evidence
            && lower_score_delta > upper_score_delta
        {
            return Err(TomographyError::InvertedBounds);
        }
        let allowed = match self.kind {
            TomographyKind::T0OwnBoardRepack => matches!(
                self.evidence,
                TomographyEvidence::Exact { .. } | TomographyEvidence::BestFound { .. }
            ),
            TomographyKind::T1PublicOneSeatWitness => {
                matches!(self.evidence, TomographyEvidence::BestFound { .. })
            }
            TomographyKind::T2LateGameBestResponse | TomographyKind::T3KnownWorldOneSeatOracle => {
                true
            }
            TomographyKind::T4ResourceRelaxedBound => {
                matches!(self.evidence, TomographyEvidence::CertifiedBounds { .. })
            }
        };
        if !allowed {
            return Err(TomographyError::InvalidEvidenceForKind(self.kind));
        }
        Ok(())
    }

    pub fn kind(&self) -> TomographyKind {
        self.kind
    }

    pub fn source_game_id(&self) -> &str {
        &self.source_game_id
    }

    pub fn acting_seat(&self) -> u8 {
        self.acting_seat
    }

    pub fn incumbent_policy_id(&self) -> &str {
        &self.incumbent_policy_id
    }

    pub fn evidence(&self) -> &TomographyEvidence {
        &self.evidence
    }

    pub fn evidence_domain(&self) -> TomographyEvidenceDomain {
        self.evidence_domain
    }

    /// P1 tomography has no admission path to funding evidence.  A later
    /// evidence phase requires a distinct schema and validated capability.
    pub const fn eligible_for_high_fidelity_funding_claim(&self) -> bool {
        false
    }
}

impl From<TomographyResult> for TomographyWire {
    fn from(value: TomographyResult) -> Self {
        Self {
            schema_id: value.schema_id,
            kind: value.kind,
            information_boundary: value.information_boundary,
            root_id: value.root_id,
            source_game_id: value.source_game_id,
            acting_seat: value.acting_seat,
            incumbent_policy_id: value.incumbent_policy_id,
            opponent_population_id: value.opponent_population_id,
            evidence: value.evidence,
            natural_frequency_weight_numerator: value.natural_frequency_weight_numerator,
            natural_frequency_weight_denominator: value.natural_frequency_weight_denominator,
            evidence_domain: value.evidence_domain,
        }
    }
}

impl TryFrom<TomographyWire> for TomographyResult {
    type Error = TomographyError;

    fn try_from(value: TomographyWire) -> Result<Self, Self::Error> {
        let result = Self {
            schema_id: value.schema_id,
            kind: value.kind,
            information_boundary: value.information_boundary,
            root_id: value.root_id,
            source_game_id: value.source_game_id,
            acting_seat: value.acting_seat,
            incumbent_policy_id: value.incumbent_policy_id,
            opponent_population_id: value.opponent_population_id,
            evidence: value.evidence,
            natural_frequency_weight_numerator: value.natural_frequency_weight_numerator,
            natural_frequency_weight_denominator: value.natural_frequency_weight_denominator,
            evidence_domain: value.evidence_domain,
        };
        result.validate()?;
        Ok(result)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum TomographyError {
    #[error("unsupported tomography schema")]
    WrongSchema,
    #[error("tomography information boundary does not match its T0-T4 kind")]
    InformationBoundaryMismatch,
    #[error("acting seat {0} is outside the four-player research table")]
    InvalidSeat(u8),
    #[error("tomography identity fields must be non-empty")]
    EmptyIdentity,
    #[error("natural-frequency weight must satisfy 0 <= numerator <= denominator")]
    InvalidNaturalFrequencyWeight,
    #[error(
        "evidence domain does not admit the incumbent policy identity \
         (incumbent_measured requires the 'incumbent:' namespace; cpu_proxy forbids it)"
    )]
    EvidenceDomainPolicyMismatch,
    #[error("certified lower bound exceeds certified upper bound")]
    InvertedBounds,
    #[error("evidence grade is not valid for {0:?}")]
    InvalidEvidenceForKind(TomographyKind),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(label: &str) -> Sha256Digest {
        Sha256Digest::of_bytes(label.as_bytes())
    }

    fn input(kind: TomographyKind, evidence: TomographyEvidence) -> TomographyResultInput {
        TomographyResultInput {
            kind,
            root_id: digest("root"),
            source_game_id: "fixture-game".to_owned(),
            acting_seat: 2,
            incumbent_policy_id: "proxy-b0".to_owned(),
            opponent_population_id: "three-proxy-b0".to_owned(),
            evidence,
            natural_frequency_weight_numerator: 1,
            natural_frequency_weight_denominator: 1,
        }
    }

    #[test]
    fn t1_proxy_is_an_executable_best_found_not_funding_evidence() {
        let result = TomographyResult::try_new(input(
            TomographyKind::T1PublicOneSeatWitness,
            TomographyEvidence::BestFound {
                score_delta: 3,
                solver_config_sha256: digest("solver"),
                witness_ledger_sha256: digest("ledger"),
                explored_nodes: 91,
            },
        ))
        .unwrap();
        assert_eq!(result.evidence().lower_bound(), 3);
        assert_eq!(result.evidence().upper_bound(), None);
        assert_eq!(result.evidence_domain(), TomographyEvidenceDomain::CpuProxy);
        assert!(!result.eligible_for_high_fidelity_funding_claim());
    }

    #[test]
    fn t4_requires_a_certified_bound() {
        let result = TomographyResult::try_new(input(
            TomographyKind::T4ResourceRelaxedBound,
            TomographyEvidence::BestFound {
                score_delta: 9,
                solver_config_sha256: digest("solver"),
                witness_ledger_sha256: digest("ledger"),
                explored_nodes: 10,
            },
        ));
        assert_eq!(
            result,
            Err(TomographyError::InvalidEvidenceForKind(
                TomographyKind::T4ResourceRelaxedBound
            ))
        );
    }

    #[test]
    fn cpu_proxy_wire_form_is_stable_and_still_parses() {
        let result = TomographyResult::try_new(input(
            TomographyKind::T0OwnBoardRepack,
            TomographyEvidence::BestFound {
                score_delta: 2,
                solver_config_sha256: digest("solver"),
                witness_ledger_sha256: digest("ledger"),
                explored_nodes: 5,
            },
        ))
        .unwrap();
        let value = serde_json::to_value(&result).unwrap();
        assert_eq!(value["evidence_domain"], serde_json::json!("cpu_proxy"));
        let decoded: TomographyResult = serde_json::from_value(value).unwrap();
        assert_eq!(decoded, result);
        assert_eq!(
            decoded.evidence_domain(),
            TomographyEvidenceDomain::CpuProxy
        );
    }

    #[test]
    fn incumbent_measured_requires_the_declared_namespace_and_roundtrips() {
        let mut incumbent = input(
            TomographyKind::T0OwnBoardRepack,
            TomographyEvidence::BestFound {
                score_delta: 4,
                solver_config_sha256: digest("solver"),
                witness_ledger_sha256: digest("ledger"),
                explored_nodes: 9,
            },
        );
        // A proxy-named policy id can never enter the incumbent domain.
        assert_eq!(
            TomographyResult::try_new_in_domain(
                incumbent.clone(),
                TomographyEvidenceDomain::IncumbentMeasured,
            ),
            Err(TomographyError::EvidenceDomainPolicyMismatch)
        );
        // A namespace prefix with no identity behind it also fails closed.
        incumbent.incumbent_policy_id = "incumbent:".to_owned();
        assert_eq!(
            TomographyResult::try_new_in_domain(
                incumbent.clone(),
                TomographyEvidenceDomain::IncumbentMeasured,
            ),
            Err(TomographyError::EvidenceDomainPolicyMismatch)
        );
        incumbent.incumbent_policy_id = "incumbent:b0-serving".to_owned();
        let result = TomographyResult::try_new_in_domain(
            incumbent,
            TomographyEvidenceDomain::IncumbentMeasured,
        )
        .unwrap();
        let value = serde_json::to_value(&result).unwrap();
        assert_eq!(
            value["evidence_domain"],
            serde_json::json!("incumbent_measured")
        );
        let decoded: TomographyResult = serde_json::from_value(value).unwrap();
        assert_eq!(decoded, result);
        assert_eq!(
            decoded.evidence_domain(),
            TomographyEvidenceDomain::IncumbentMeasured
        );
        assert!(!decoded.eligible_for_high_fidelity_funding_claim());
    }

    #[test]
    fn incumbent_namespaced_identity_cannot_pose_as_cpu_proxy() {
        let mut proxy = input(
            TomographyKind::T1PublicOneSeatWitness,
            TomographyEvidence::BestFound {
                score_delta: 1,
                solver_config_sha256: digest("solver"),
                witness_ledger_sha256: digest("ledger"),
                explored_nodes: 3,
            },
        );
        proxy.incumbent_policy_id = "incumbent:b0-serving".to_owned();
        assert_eq!(
            TomographyResult::try_new(proxy),
            Err(TomographyError::EvidenceDomainPolicyMismatch)
        );
    }

    #[test]
    fn domain_mislabeling_fails_closed_at_deserialization() {
        let result = TomographyResult::try_new(input(
            TomographyKind::T0OwnBoardRepack,
            TomographyEvidence::BestFound {
                score_delta: 2,
                solver_config_sha256: digest("solver"),
                witness_ledger_sha256: digest("ledger"),
                explored_nodes: 5,
            },
        ))
        .unwrap();
        let mut value = serde_json::to_value(&result).unwrap();
        // Relabeling a proxy-population result as incumbent-measured must be
        // rejected on read, not silently accepted.
        value["evidence_domain"] = serde_json::json!("incumbent_measured");
        assert!(serde_json::from_value::<TomographyResult>(value.clone()).is_err());
        // Unknown domains are rejected outright.
        value["evidence_domain"] = serde_json::json!("gpu_incumbent");
        assert!(serde_json::from_value::<TomographyResult>(value).is_err());
    }

    #[test]
    fn population_validation_matches_result_validation() {
        let proxy = TomographyPopulation {
            incumbent_policy_id: "proxy-b0".to_owned(),
            opponent_population_id: "proxy-b0:table".to_owned(),
            evidence_domain: TomographyEvidenceDomain::CpuProxy,
        };
        proxy.validate().unwrap();
        let mislabeled = TomographyPopulation {
            evidence_domain: TomographyEvidenceDomain::IncumbentMeasured,
            ..proxy.clone()
        };
        assert_eq!(
            mislabeled.validate(),
            Err(TomographyError::EvidenceDomainPolicyMismatch)
        );
        let incumbent = TomographyPopulation {
            incumbent_policy_id: "incumbent:b0-serving".to_owned(),
            opponent_population_id: "incumbent:b0-serving:table".to_owned(),
            evidence_domain: TomographyEvidenceDomain::IncumbentMeasured,
        };
        incumbent.validate().unwrap();
    }

    #[test]
    fn exact_and_bound_claims_roundtrip_with_their_grade() {
        for evidence in [
            TomographyEvidence::Exact {
                score_delta: 4,
                optimality_certificate_sha256: digest("exact"),
            },
            TomographyEvidence::CertifiedBounds {
                lower_score_delta: 2,
                upper_score_delta: 7,
                bound_certificate_sha256: digest("bounds"),
            },
        ] {
            let result =
                TomographyResult::try_new(input(TomographyKind::T2LateGameBestResponse, evidence))
                    .unwrap();
            let encoded = serde_json::to_vec(&result).unwrap();
            let decoded: TomographyResult = serde_json::from_slice(&encoded).unwrap();
            assert_eq!(decoded, result);
        }
    }
}
