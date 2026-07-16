//! Typed CPU-reference contracts for Cascadia Rival.
//!
//! This crate is a protocol and scientific-identity layer over
//! [`cascadia_game`]. It deliberately owns no duplicate rules or scoring
//! implementation and has no accelerator feature or dependency.

#![forbid(unsafe_code)]

mod action_id;
mod bounds;
mod compiler;
mod digest;
mod dynamic_urn;
mod identity;
mod ledger;
mod menu;
mod observation;
mod policy;
mod rng;
mod ruleset_identity;
mod scenario;
mod terminal;
mod tomography;

pub use action_id::{
    ACTION_CONTENT_ID_VERSION, ActionContentId, ActionIdError,
    CANDIDATE_ACTION_OCCURRENCE_ID_VERSION, CandidateActionOccurrenceId, LEGACY_ACTION_ID_VERSION,
    LegacyActionIdV0, PUBLIC_ROOT_ID_VERSION, PublicRootId, ROOT_ACTION_OCCURRENCE_ID_VERSION,
    ROOT_CHRONOLOGY_VERSION, RootActionOccurrenceId, RootKind,
};
pub use bounds::{
    BOUND_CERTIFICATE_SCHEMA_ID, BoundCertificateError, CertifiedScoreDifferenceBound,
    GLOBAL_RESEARCH_BOUND_AUTHORITY_ID, GlobalBoundDerivation,
};
pub use compiler::{
    CompilerError, DENSE_COMPILER_ID, DenseActionSemantics, DenseSemanticCompiler,
    DenseStateSemantics, ScoreDelta, SemanticCompiler,
};
pub use digest::{DigestParseError, Sha256Digest};
pub use dynamic_urn::{
    DYNAMIC_URN_PROOF_CONTRACT_ID, DynamicUrnAdmissionStatus, DynamicUrnError,
    DynamicUrnObligation, MAX_EXHAUSTIVE_SMALL_URN_ITEMS, SmallUrnOracleReport,
    verify_small_urn_priority_oracle,
};
pub use identity::{
    BkIdentity, CANONICAL_SIMULATOR_ID, FailureBehavior, FailureDisposition, ForbiddenCapabilities,
    FrozenPolicyIdentity, MNextIdentity, NumericalMode, POLICY_IDENTITY_SCHEMA_ID, PiLIdentity,
    PolicyIdentityError, PolicyIdentityFields, Precision, RngContractIdentity, WkIdentity,
};
pub use ledger::{
    LedgerCompletion, LedgerError, RootDecisionRecord, SelectedDecisionKind,
    TRAJECTORY_LEDGER_SCHEMA_ID, TrajectoryLedger, TrajectoryLedgerBuilder, TurnEvidenceKind,
    TurnLedgerRecord, replay_policy_decision_trace,
};
pub use menu::{
    INCUMBENT_MENU_HASH_VERSION, IncumbentCandidateMenu, IncumbentMenuHash, MENU_HASH_VERSION,
    MenuComposer, MenuError, RULES_MENU_HASH_VERSION, RulesDecision, RulesLegalMenu, RulesMenuHash,
};
pub use observation::{
    HonestWorldSampler, ObservationError, PUBLIC_POLICY_OBSERVATION_SCHEMA_ID, PolicyMemoryBank,
    PolicyWorld, PrivateSimState, PublicPolicyObs, SEAT_LOCAL_MEMORY_SCHEMA_ID, SeatIndex,
    SeatLocalMemory,
};
pub use policy::{
    BoxedPolicyError, FrozenPolicy, MenuIndex, PolicyContractError, PolicyDecision,
    require_root_kind,
};
pub use rng::{
    BranchKey, CouplingGroupId, EvaluationBranch, Fidelity, InnerRngCoordinate, OuterPhysicalRng,
    PolicyRng, RNG_CONTRACT_ID, RedeterminationSeed, RivalSeed, RngFactory, SearchRng,
    SourceRootSeed, TieBreakRng,
};
pub use ruleset_identity::{
    LEGACY_RESEARCH_RULESET_ID, RESEARCH_RULESET_SCHEMA_ID, ResearchRulesetIdentity,
    RulesetIdentityError,
};
pub use scenario::{
    INDEPENDENT_SCENARIO_SAMPLER_ID, IndependentScenarioSampler, ScenarioCoordinate, ScenarioError,
    ScenarioMode,
};
pub use terminal::{
    MAX_POLICY_DECISIONS_PER_TURN, MAX_TERMINAL_PAIR_LEDGER_BYTES, MAX_TERMINAL_TRAJECTORY_TURNS,
    PROXY_TERMINAL_PAIR_SCHEMA_ID, PROXY_TERMINAL_TRAJECTORY_SCHEMA_ID, ProxyTerminalPair,
    ProxyTerminalPairRequest, ProxyTerminalTrajectory, TERMINAL_PAIR_VERIFIER_CONTRACT_ID,
    TerminalError, VERIFIED_TERMINAL_PAIR_RECEIPT_SCHEMA_ID, VerifiedTerminalPairReceipt,
    run_proxy_terminal_pair,
};
pub use tomography::{
    InformationBoundary, TOMOGRAPHY_RESULT_SCHEMA_ID, TomographyError, TomographyEvidence,
    TomographyEvidenceDomain, TomographyKind, TomographyResult, TomographyResultInput,
};
