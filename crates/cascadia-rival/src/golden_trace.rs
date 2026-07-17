//! Golden decision traces for the exporter extraction (build-scope WI-3,
//! held at the D1 wall).
//!
//! A golden trace is a versioned, fail-closed record of ONE production
//! serving decision, sufficient to prove behavioral identity of an extracted
//! policy library against the exporter it was extracted from.  This module is
//! the *preparation* half of the work item: the schema, the first-divergence
//! comparator, and the trace-set manifest, exercised purely on synthetic CPU
//! fixtures.  Nothing here reads a production checkpoint, a model bridge, or
//! any `real-root-exporter` source; capture of production traces is a
//! post-D1-boundary activity under its own instruction.
//!
//! ## Digest-versus-verbatim policy
//!
//! Traces must stay small enough to publish per seed while still detecting
//! any behavioral change, so bulky sequences are content-hashed rather than
//! stored verbatim:
//!
//! - the legal menu keeps its action count, cap, first/last action ids, and a
//!   SHA-256 of the full canonical ordered action-id list (late-game legal
//!   menus can exceed several thousand compound actions);
//! - every model-bridge interaction keeps its request row count plus SHA-256
//!   digests of the exact request and response payload bytes, in invocation
//!   order.  Two policies are behaviorally identical at a root only if they
//!   issue byte-identical evaluation requests in the same order and observe
//!   byte-identical responses, so the ordered digest sequence is exactly the
//!   equality that extraction must preserve;
//! - scalar decision facts (state hash, prelude, search configuration, chosen
//!   action) are stored verbatim: they are small and pinpointing their
//!   divergence directly is worth more than one opaque hash.
//!
//! Floats are carried as [`CanonicalF64`]: a finite value serialized as its
//! shortest round-trip decimal string, rejected on parse unless the text
//! reproduces the identical bits.  This keeps trace bytes deterministic
//! across platforms and keeps float equality exact, without hashing raw JSON
//! numbers.

use std::{fmt, hash::Hash, path::Path};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{LedgerError, ResearchRulesetIdentity, RulesetIdentityError, Sha256Digest};

pub const GOLDEN_DECISION_TRACE_SCHEMA_ID: &str = "cascadiav3.rival_golden_decision_trace.v1";
pub const GOLDEN_TRACE_MANIFEST_SCHEMA_ID: &str = "cascadiav3.rival_golden_trace_manifest.v1";

/// Hard cap on recorded bridge interactions per decision.  One serving
/// decision batches leaf evaluations; a run that exceeds this is not a
/// plausible single root decision and the trace fails closed.
pub const MAX_GOLDEN_TRACE_BRIDGE_EXCHANGES: usize = 4096;
/// Hard cap on traces in one manifest; golden sets are a handful of seeds.
pub const MAX_GOLDEN_TRACE_MANIFEST_ENTRIES: usize = 65_536;

/// A finite `f64` whose wire form is the exact shortest round-trip decimal
/// string.  Equality and hashing are bit-exact; non-finite values and any
/// non-canonical spelling (`"0.50"`, `".5"`, `"+1"`, `"NaN"`) are rejected.
#[derive(Clone, Copy, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct CanonicalF64(f64);

impl CanonicalF64 {
    pub fn new(value: f64) -> Result<Self, GoldenTraceError> {
        if !value.is_finite() {
            return Err(GoldenTraceError::NonFiniteFloat(value.to_string()));
        }
        Ok(Self(value))
    }

    pub fn value(self) -> f64 {
        self.0
    }

    /// The canonical wire text (Rust's shortest round-trip `Display` form).
    pub fn canonical_text(self) -> String {
        self.0.to_string()
    }
}

impl fmt::Debug for CanonicalF64 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("CanonicalF64")
            .field(&self.canonical_text())
            .finish()
    }
}

impl fmt::Display for CanonicalF64 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.canonical_text())
    }
}

impl PartialEq for CanonicalF64 {
    fn eq(&self, other: &Self) -> bool {
        self.0.to_bits() == other.0.to_bits()
    }
}

impl Eq for CanonicalF64 {}

impl Hash for CanonicalF64 {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state);
    }
}

impl From<CanonicalF64> for String {
    fn from(value: CanonicalF64) -> Self {
        value.canonical_text()
    }
}

impl TryFrom<String> for CanonicalF64 {
    type Error = GoldenTraceError;

    fn try_from(text: String) -> Result<Self, Self::Error> {
        let value: f64 = text
            .parse()
            .map_err(|_| GoldenTraceError::NonCanonicalFloat(text.clone()))?;
        if !value.is_finite() {
            return Err(GoldenTraceError::NonFiniteFloat(text));
        }
        if value.to_string() != text {
            return Err(GoldenTraceError::NonCanonicalFloat(text));
        }
        Ok(Self(value))
    }
}

/// The identity triple every golden artifact must declare: which rules, which
/// source revision, and which frozen policy produced it.  The policy is
/// hash-pinned (the [`crate::FrozenPolicyIdentity`] SHA-256) instead of
/// duplicated: the complete identity document is published once and the trace
/// binds to it, so the two can never drift apart silently.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GoldenTraceIdentity {
    pub ruleset: ResearchRulesetIdentity,
    pub source_revision: String,
    pub policy_identity_sha256: Sha256Digest,
}

impl GoldenTraceIdentity {
    pub fn validate(&self) -> Result<(), GoldenTraceError> {
        self.ruleset.validate()?;
        if self.source_revision.trim().is_empty() {
            return Err(GoldenTraceError::EmptyField("identity.source_revision"));
        }
        Ok(())
    }
}

/// The free three-of-a-kind prelude decision at this root.  When the wipe is
/// accepted the replacement market is public chance; its revealed content
/// hash is part of the decision's causal record.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum GoldenPreludeRecord {
    Declined,
    Accepted {
        choice: String,
        revealed_market_sha256: Sha256Digest,
    },
}

impl GoldenPreludeRecord {
    fn validate(&self) -> Result<(), GoldenTraceError> {
        match self {
            Self::Declined => Ok(()),
            Self::Accepted { choice, .. } => {
                if choice.trim().is_empty() {
                    return Err(GoldenTraceError::EmptyField("prelude.choice"));
                }
                Ok(())
            }
        }
    }
}

impl fmt::Display for GoldenPreludeRecord {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Declined => formatter.write_str("declined"),
            Self::Accepted {
                choice,
                revealed_market_sha256,
            } => write!(formatter, "accepted({choice}, {revealed_market_sha256})"),
        }
    }
}

/// Digest of the ordered legal action menu offered to the policy.  The full
/// id list is hashed, not stored; the endpoints and count localize cheap
/// divergences before falling back to the opaque list hash.  `menu_cap` is
/// the root menu enumeration cap in force when the menu was composed
/// (`0` = uncapped), and must equal the search configuration's cap.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GoldenMenuDigest {
    pub action_count: u32,
    pub menu_cap: u32,
    pub first_action_id: String,
    pub last_action_id: String,
    /// SHA-256 of the canonical JSON array of the full ordered action-id list.
    pub action_ids_sha256: Sha256Digest,
}

impl GoldenMenuDigest {
    /// Digest a complete ordered action-id list under the given cap.
    pub fn from_ordered_action_ids(
        action_ids: &[String],
        menu_cap: u32,
    ) -> Result<Self, GoldenTraceError> {
        if action_ids.is_empty() {
            return Err(GoldenTraceError::ZeroField("menu.action_count"));
        }
        if action_ids.iter().any(|id| id.trim().is_empty()) {
            return Err(GoldenTraceError::EmptyField("menu.action_id"));
        }
        let action_count = u32::try_from(action_ids.len())
            .map_err(|_| GoldenTraceError::MenuTooLarge(action_ids.len()))?;
        let digest = Sha256Digest::of_bytes(&serde_json::to_vec(action_ids)?);
        let menu = Self {
            action_count,
            menu_cap,
            first_action_id: action_ids[0].clone(),
            last_action_id: action_ids[action_ids.len() - 1].clone(),
            action_ids_sha256: digest,
        };
        menu.validate()?;
        Ok(menu)
    }

    fn validate(&self) -> Result<(), GoldenTraceError> {
        if self.action_count == 0 {
            return Err(GoldenTraceError::ZeroField("menu.action_count"));
        }
        if self.menu_cap != 0 && self.action_count > self.menu_cap {
            return Err(GoldenTraceError::MenuCapExceeded {
                action_count: self.action_count,
                menu_cap: self.menu_cap,
            });
        }
        for (name, value) in [
            ("menu.first_action_id", self.first_action_id.as_str()),
            ("menu.last_action_id", self.last_action_id.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(GoldenTraceError::EmptyField(name));
            }
        }
        if self.action_count == 1 && self.first_action_id != self.last_action_id {
            return Err(GoldenTraceError::SingletonMenuEndpointMismatch);
        }
        Ok(())
    }
}

/// The complete serving search configuration in force for this decision.
/// Field names mirror the production exporter's `gumbel_*` arguments;
/// `root_menu_cap` duplicates the menu digest's cap by design (the config
/// block stays self-contained) and the trace validator enforces agreement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GoldenSearchConfig {
    pub n_simulations: u32,
    pub top_m: u32,
    pub depth_rounds: u32,
    pub determinizations: u32,
    pub market_decision_samples: u32,
    pub exact_endgame_turns: u32,
    pub blend_weight: CanonicalF64,
    pub root_menu_cap: u32,
    pub c_visit: CanonicalF64,
    pub c_scale: CanonicalF64,
    pub seed: u64,
}

impl GoldenSearchConfig {
    fn validate(&self) -> Result<(), GoldenTraceError> {
        for (name, value) in [
            ("search.n_simulations", self.n_simulations),
            ("search.top_m", self.top_m),
            ("search.depth_rounds", self.depth_rounds),
            ("search.determinizations", self.determinizations),
            ("search.market_decision_samples", self.market_decision_samples),
        ] {
            if value == 0 {
                return Err(GoldenTraceError::ZeroField(name));
            }
        }
        if !(0.0..=1.0).contains(&self.blend_weight.value()) {
            return Err(GoldenTraceError::UnitIntervalViolation(
                "search.blend_weight",
            ));
        }
        if self.c_visit.value() < 0.0 {
            return Err(GoldenTraceError::NegativeField("search.c_visit"));
        }
        if self.c_scale.value() <= 0.0 {
            return Err(GoldenTraceError::NonPositiveField("search.c_scale"));
        }
        Ok(())
    }
}

/// One model-bridge interaction, digested: how many evaluation rows the
/// request carried plus SHA-256 of the exact request and response payload
/// bytes.  An extracted policy is bridge-identical exactly when it reproduces
/// this sequence element-for-element.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BridgeExchangeDigest {
    pub request_rows: u32,
    pub request_sha256: Sha256Digest,
    pub response_sha256: Sha256Digest,
}

/// The decision the search returned.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GoldenChosenAction {
    pub action_id: String,
    pub menu_index: u32,
    pub completed_q: CanonicalF64,
    pub improved_policy_mass: CanonicalF64,
}

/// Everything a caller supplies to seal one golden decision trace; the
/// schema id and whole-trace content hash are computed, never supplied.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GoldenDecisionTraceInput {
    pub identity: GoldenTraceIdentity,
    pub public_state_sha256: Sha256Digest,
    pub ply: u32,
    pub seat: u8,
    pub completed_turns: u32,
    pub prelude: GoldenPreludeRecord,
    pub menu: GoldenMenuDigest,
    pub search: GoldenSearchConfig,
    pub bridge_exchanges: Vec<BridgeExchangeDigest>,
    pub chosen: GoldenChosenAction,
}

/// One sealed serving decision (`cascadiav3.rival_golden_decision_trace.v1`).
///
/// The struct is always valid: construction and deserialization both run the
/// full fail-closed validator, including the whole-trace content hash.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "TraceWire", into = "TraceWire")]
pub struct GoldenDecisionTrace {
    schema_id: String,
    identity: GoldenTraceIdentity,
    public_state_sha256: Sha256Digest,
    ply: u32,
    seat: u8,
    completed_turns: u32,
    prelude: GoldenPreludeRecord,
    menu: GoldenMenuDigest,
    search: GoldenSearchConfig,
    bridge_exchanges: Vec<BridgeExchangeDigest>,
    chosen: GoldenChosenAction,
    trace_sha256: Sha256Digest,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct TraceWire {
    schema_id: String,
    identity: GoldenTraceIdentity,
    public_state_sha256: Sha256Digest,
    ply: u32,
    seat: u8,
    completed_turns: u32,
    prelude: GoldenPreludeRecord,
    menu: GoldenMenuDigest,
    search: GoldenSearchConfig,
    bridge_exchanges: Vec<BridgeExchangeDigest>,
    chosen: GoldenChosenAction,
    trace_sha256: Sha256Digest,
}

#[derive(Serialize)]
struct TraceContent<'a> {
    schema_id: &'a str,
    identity: &'a GoldenTraceIdentity,
    public_state_sha256: &'a Sha256Digest,
    ply: u32,
    seat: u8,
    completed_turns: u32,
    prelude: &'a GoldenPreludeRecord,
    menu: &'a GoldenMenuDigest,
    search: &'a GoldenSearchConfig,
    bridge_exchanges: &'a [BridgeExchangeDigest],
    chosen: &'a GoldenChosenAction,
}

impl GoldenDecisionTrace {
    pub fn new(input: GoldenDecisionTraceInput) -> Result<Self, GoldenTraceError> {
        let mut trace = Self {
            schema_id: GOLDEN_DECISION_TRACE_SCHEMA_ID.to_owned(),
            identity: input.identity,
            public_state_sha256: input.public_state_sha256,
            ply: input.ply,
            seat: input.seat,
            completed_turns: input.completed_turns,
            prelude: input.prelude,
            menu: input.menu,
            search: input.search,
            bridge_exchanges: input.bridge_exchanges,
            chosen: input.chosen,
            trace_sha256: Sha256Digest::of_bytes(b""),
        };
        trace.trace_sha256 = trace.recompute_hash()?;
        trace.validate()?;
        Ok(trace)
    }

    pub fn identity(&self) -> &GoldenTraceIdentity {
        &self.identity
    }

    pub fn public_state_sha256(&self) -> &Sha256Digest {
        &self.public_state_sha256
    }

    pub fn ply(&self) -> u32 {
        self.ply
    }

    pub fn seat(&self) -> u8 {
        self.seat
    }

    pub fn completed_turns(&self) -> u32 {
        self.completed_turns
    }

    pub fn prelude(&self) -> &GoldenPreludeRecord {
        &self.prelude
    }

    pub fn menu(&self) -> &GoldenMenuDigest {
        &self.menu
    }

    pub fn search(&self) -> &GoldenSearchConfig {
        &self.search
    }

    pub fn bridge_exchanges(&self) -> &[BridgeExchangeDigest] {
        &self.bridge_exchanges
    }

    pub fn chosen(&self) -> &GoldenChosenAction {
        &self.chosen
    }

    pub fn trace_sha256(&self) -> &Sha256Digest {
        &self.trace_sha256
    }

    pub fn from_json_slice(bytes: &[u8]) -> Result<Self, GoldenTraceError> {
        Ok(serde_json::from_slice(bytes)?)
    }

    pub fn canonical_json_bytes(&self) -> Result<Vec<u8>, GoldenTraceError> {
        self.validate()?;
        Ok(serde_json::to_vec_pretty(self)?)
    }

    /// Durably publish through a same-directory temporary without ever
    /// replacing an existing artifact.
    pub fn write_json_immutable(&self, destination: &Path) -> Result<(), GoldenTraceError> {
        let bytes = self.canonical_json_bytes()?;
        crate::ledger::write_immutable_bytes(destination, &bytes)?;
        Ok(())
    }

    pub fn validate(&self) -> Result<(), GoldenTraceError> {
        if self.schema_id != GOLDEN_DECISION_TRACE_SCHEMA_ID {
            return Err(GoldenTraceError::WrongTraceSchema(self.schema_id.clone()));
        }
        self.identity.validate()?;
        if self.seat >= 4 {
            return Err(GoldenTraceError::SeatOutOfRange(self.seat));
        }
        if self.completed_turns > self.ply {
            return Err(GoldenTraceError::TurnAccounting {
                ply: self.ply,
                completed_turns: self.completed_turns,
            });
        }
        self.prelude.validate()?;
        self.menu.validate()?;
        self.search.validate()?;
        if self.menu.menu_cap != self.search.root_menu_cap {
            return Err(GoldenTraceError::MenuCapDisagreement {
                menu: self.menu.menu_cap,
                search: self.search.root_menu_cap,
            });
        }
        if self.bridge_exchanges.len() > MAX_GOLDEN_TRACE_BRIDGE_EXCHANGES {
            return Err(GoldenTraceError::BridgeExchangeCap {
                count: self.bridge_exchanges.len(),
                cap: MAX_GOLDEN_TRACE_BRIDGE_EXCHANGES,
            });
        }
        for (index, exchange) in self.bridge_exchanges.iter().enumerate() {
            if exchange.request_rows == 0 {
                return Err(GoldenTraceError::ZeroBridgeRequestRows(index));
            }
        }
        if self.chosen.action_id.trim().is_empty() {
            return Err(GoldenTraceError::EmptyField("chosen.action_id"));
        }
        if self.chosen.menu_index >= self.menu.action_count {
            return Err(GoldenTraceError::ChosenIndexOutOfRange {
                index: self.chosen.menu_index,
                action_count: self.menu.action_count,
            });
        }
        if self.chosen.menu_index == 0 && self.chosen.action_id != self.menu.first_action_id {
            return Err(GoldenTraceError::ChosenEndpointMismatch("first"));
        }
        if self.chosen.menu_index == self.menu.action_count - 1
            && self.chosen.action_id != self.menu.last_action_id
        {
            return Err(GoldenTraceError::ChosenEndpointMismatch("last"));
        }
        if !(0.0..=1.0).contains(&self.chosen.improved_policy_mass.value()) {
            return Err(GoldenTraceError::UnitIntervalViolation(
                "chosen.improved_policy_mass",
            ));
        }
        if self.recompute_hash()? != self.trace_sha256 {
            return Err(GoldenTraceError::TraceHashMismatch);
        }
        Ok(())
    }

    fn recompute_hash(&self) -> Result<Sha256Digest, GoldenTraceError> {
        let content = TraceContent {
            schema_id: &self.schema_id,
            identity: &self.identity,
            public_state_sha256: &self.public_state_sha256,
            ply: self.ply,
            seat: self.seat,
            completed_turns: self.completed_turns,
            prelude: &self.prelude,
            menu: &self.menu,
            search: &self.search,
            bridge_exchanges: &self.bridge_exchanges,
            chosen: &self.chosen,
        };
        // Hash a recursively key-sorted JSON value (the crate's canonical
        // encoder convention), never the struct declaration order.
        let value = serde_json::to_value(&content)?;
        Ok(Sha256Digest::of_bytes(&serde_json::to_vec(&value)?))
    }
}

impl From<GoldenDecisionTrace> for TraceWire {
    fn from(value: GoldenDecisionTrace) -> Self {
        Self {
            schema_id: value.schema_id,
            identity: value.identity,
            public_state_sha256: value.public_state_sha256,
            ply: value.ply,
            seat: value.seat,
            completed_turns: value.completed_turns,
            prelude: value.prelude,
            menu: value.menu,
            search: value.search,
            bridge_exchanges: value.bridge_exchanges,
            chosen: value.chosen,
            trace_sha256: value.trace_sha256,
        }
    }
}

impl TryFrom<TraceWire> for GoldenDecisionTrace {
    type Error = GoldenTraceError;

    fn try_from(value: TraceWire) -> Result<Self, Self::Error> {
        let trace = Self {
            schema_id: value.schema_id,
            identity: value.identity,
            public_state_sha256: value.public_state_sha256,
            ply: value.ply,
            seat: value.seat,
            completed_turns: value.completed_turns,
            prelude: value.prelude,
            menu: value.menu,
            search: value.search,
            bridge_exchanges: value.bridge_exchanges,
            chosen: value.chosen,
            trace_sha256: value.trace_sha256,
        };
        trace.validate()?;
        Ok(trace)
    }
}

/// The first field at which a candidate trace diverges from its reference,
/// with both observed values.  Variants are ordered by the comparison
/// sequence: identity, public state, prelude, menu, search configuration,
/// the bridge interaction sequence, then the chosen action.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum TraceDivergence {
    #[error("ruleset identity diverges: reference {reference}, candidate {candidate}")]
    Ruleset { reference: String, candidate: String },
    #[error("source revision diverges: reference {reference}, candidate {candidate}")]
    SourceRevision { reference: String, candidate: String },
    #[error("policy identity diverges: reference {reference}, candidate {candidate}")]
    PolicyIdentitySha256 { reference: String, candidate: String },
    #[error("public state hash diverges: reference {reference}, candidate {candidate}")]
    PublicStateSha256 { reference: String, candidate: String },
    #[error("ply diverges: reference {reference}, candidate {candidate}")]
    Ply { reference: String, candidate: String },
    #[error("acting seat diverges: reference {reference}, candidate {candidate}")]
    Seat { reference: String, candidate: String },
    #[error("completed turns diverge: reference {reference}, candidate {candidate}")]
    CompletedTurns { reference: String, candidate: String },
    #[error("prelude decision diverges: reference {reference}, candidate {candidate}")]
    Prelude { reference: String, candidate: String },
    #[error("menu action count diverges: reference {reference}, candidate {candidate}")]
    MenuActionCount { reference: String, candidate: String },
    #[error("menu cap diverges: reference {reference}, candidate {candidate}")]
    MenuCap { reference: String, candidate: String },
    #[error("first menu action id diverges: reference {reference}, candidate {candidate}")]
    MenuFirstActionId { reference: String, candidate: String },
    #[error("last menu action id diverges: reference {reference}, candidate {candidate}")]
    MenuLastActionId { reference: String, candidate: String },
    #[error("menu action-id list hash diverges: reference {reference}, candidate {candidate}")]
    MenuActionIdsSha256 { reference: String, candidate: String },
    #[error("search n_simulations diverges: reference {reference}, candidate {candidate}")]
    NSimulations { reference: String, candidate: String },
    #[error("search top_m diverges: reference {reference}, candidate {candidate}")]
    TopM { reference: String, candidate: String },
    #[error("search depth_rounds diverges: reference {reference}, candidate {candidate}")]
    DepthRounds { reference: String, candidate: String },
    #[error("search determinizations diverge: reference {reference}, candidate {candidate}")]
    Determinizations { reference: String, candidate: String },
    #[error(
        "search market_decision_samples diverge: reference {reference}, candidate {candidate}"
    )]
    MarketDecisionSamples { reference: String, candidate: String },
    #[error("search exact_endgame_turns diverge: reference {reference}, candidate {candidate}")]
    ExactEndgameTurns { reference: String, candidate: String },
    #[error("search blend_weight diverges: reference {reference}, candidate {candidate}")]
    BlendWeight { reference: String, candidate: String },
    #[error("search c_visit diverges: reference {reference}, candidate {candidate}")]
    CVisit { reference: String, candidate: String },
    #[error("search c_scale diverges: reference {reference}, candidate {candidate}")]
    CScale { reference: String, candidate: String },
    #[error("search seed diverges: reference {reference}, candidate {candidate}")]
    Seed { reference: String, candidate: String },
    #[error(
        "bridge exchange {index} request row count diverges: \
         reference {reference}, candidate {candidate}"
    )]
    BridgeRequestRows {
        index: usize,
        reference: String,
        candidate: String,
    },
    #[error(
        "bridge exchange {index} request digest diverges: \
         reference {reference}, candidate {candidate}"
    )]
    BridgeRequestSha256 {
        index: usize,
        reference: String,
        candidate: String,
    },
    #[error(
        "bridge exchange {index} response digest diverges: \
         reference {reference}, candidate {candidate}"
    )]
    BridgeResponseSha256 {
        index: usize,
        reference: String,
        candidate: String,
    },
    #[error("bridge exchange count diverges: reference {reference}, candidate {candidate}")]
    BridgeExchangeCount { reference: String, candidate: String },
    #[error("chosen menu index diverges: reference {reference}, candidate {candidate}")]
    ChosenMenuIndex { reference: String, candidate: String },
    #[error("chosen action id diverges: reference {reference}, candidate {candidate}")]
    ChosenActionId { reference: String, candidate: String },
    #[error("chosen completed Q diverges: reference {reference}, candidate {candidate}")]
    ChosenCompletedQ { reference: String, candidate: String },
    #[error("chosen improved-policy mass diverges: reference {reference}, candidate {candidate}")]
    ChosenImprovedPolicyMass { reference: String, candidate: String },
    #[error("trace content hash diverges: reference {reference}, candidate {candidate}")]
    TraceSha256 { reference: String, candidate: String },
}

macro_rules! check_field {
    ($variant:ident, $reference:expr, $candidate:expr) => {
        if $reference != $candidate {
            return Err(TraceDivergence::$variant {
                reference: $reference.to_string(),
                candidate: $candidate.to_string(),
            });
        }
    };
}

macro_rules! check_indexed_field {
    ($variant:ident, $index:expr, $reference:expr, $candidate:expr) => {
        if $reference != $candidate {
            return Err(TraceDivergence::$variant {
                index: $index,
                reference: $reference.to_string(),
                candidate: $candidate.to_string(),
            });
        }
    };
}

/// Compare a candidate trace against a reference, pinpointing the FIRST
/// differing field.  Byte-identical traces compare equal; this is the
/// behavioral-identity check the post-D1 extraction must pass on every
/// captured production trace.
///
/// `search.root_menu_cap` carries no dedicated variant: each validated trace
/// pins it equal to `menu.menu_cap`, so a cap divergence always surfaces as
/// [`TraceDivergence::MenuCap`].
pub fn compare_traces(
    reference: &GoldenDecisionTrace,
    candidate: &GoldenDecisionTrace,
) -> Result<(), TraceDivergence> {
    if reference.identity.ruleset != candidate.identity.ruleset {
        return Err(TraceDivergence::Ruleset {
            reference: ruleset_display(&reference.identity.ruleset),
            candidate: ruleset_display(&candidate.identity.ruleset),
        });
    }
    check_field!(
        SourceRevision,
        reference.identity.source_revision,
        candidate.identity.source_revision
    );
    check_field!(
        PolicyIdentitySha256,
        reference.identity.policy_identity_sha256,
        candidate.identity.policy_identity_sha256
    );
    check_field!(
        PublicStateSha256,
        reference.public_state_sha256,
        candidate.public_state_sha256
    );
    check_field!(Ply, reference.ply, candidate.ply);
    check_field!(Seat, reference.seat, candidate.seat);
    check_field!(
        CompletedTurns,
        reference.completed_turns,
        candidate.completed_turns
    );
    check_field!(Prelude, reference.prelude, candidate.prelude);
    check_field!(
        MenuActionCount,
        reference.menu.action_count,
        candidate.menu.action_count
    );
    check_field!(MenuCap, reference.menu.menu_cap, candidate.menu.menu_cap);
    check_field!(
        MenuFirstActionId,
        reference.menu.first_action_id,
        candidate.menu.first_action_id
    );
    check_field!(
        MenuLastActionId,
        reference.menu.last_action_id,
        candidate.menu.last_action_id
    );
    check_field!(
        MenuActionIdsSha256,
        reference.menu.action_ids_sha256,
        candidate.menu.action_ids_sha256
    );
    check_field!(
        NSimulations,
        reference.search.n_simulations,
        candidate.search.n_simulations
    );
    check_field!(TopM, reference.search.top_m, candidate.search.top_m);
    check_field!(
        DepthRounds,
        reference.search.depth_rounds,
        candidate.search.depth_rounds
    );
    check_field!(
        Determinizations,
        reference.search.determinizations,
        candidate.search.determinizations
    );
    check_field!(
        MarketDecisionSamples,
        reference.search.market_decision_samples,
        candidate.search.market_decision_samples
    );
    check_field!(
        ExactEndgameTurns,
        reference.search.exact_endgame_turns,
        candidate.search.exact_endgame_turns
    );
    check_field!(
        BlendWeight,
        reference.search.blend_weight,
        candidate.search.blend_weight
    );
    check_field!(CVisit, reference.search.c_visit, candidate.search.c_visit);
    check_field!(CScale, reference.search.c_scale, candidate.search.c_scale);
    check_field!(Seed, reference.search.seed, candidate.search.seed);
    let shared = reference
        .bridge_exchanges
        .len()
        .min(candidate.bridge_exchanges.len());
    for index in 0..shared {
        let reference_exchange = &reference.bridge_exchanges[index];
        let candidate_exchange = &candidate.bridge_exchanges[index];
        check_indexed_field!(
            BridgeRequestRows,
            index,
            reference_exchange.request_rows,
            candidate_exchange.request_rows
        );
        check_indexed_field!(
            BridgeRequestSha256,
            index,
            reference_exchange.request_sha256,
            candidate_exchange.request_sha256
        );
        check_indexed_field!(
            BridgeResponseSha256,
            index,
            reference_exchange.response_sha256,
            candidate_exchange.response_sha256
        );
    }
    check_field!(
        BridgeExchangeCount,
        reference.bridge_exchanges.len(),
        candidate.bridge_exchanges.len()
    );
    check_field!(
        ChosenMenuIndex,
        reference.chosen.menu_index,
        candidate.chosen.menu_index
    );
    check_field!(
        ChosenActionId,
        reference.chosen.action_id,
        candidate.chosen.action_id
    );
    check_field!(
        ChosenCompletedQ,
        reference.chosen.completed_q,
        candidate.chosen.completed_q
    );
    check_field!(
        ChosenImprovedPolicyMass,
        reference.chosen.improved_policy_mass,
        candidate.chosen.improved_policy_mass
    );
    // Every content field above matched and both traces are validated, so the
    // whole-trace hashes must agree; this is a defensive last resort only.
    check_field!(TraceSha256, reference.trace_sha256, candidate.trace_sha256);
    Ok(())
}

fn ruleset_display(ruleset: &ResearchRulesetIdentity) -> String {
    serde_json::to_string(ruleset)
        .expect("serializing a validated in-memory ruleset identity cannot fail")
}

/// One trace's row in a manifest: its search seed and sealed content hash.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GoldenTraceManifestEntry {
    pub seed: u64,
    pub trace_sha256: Sha256Digest,
}

/// A sealed set of golden traces captured under ONE identity
/// (`cascadiav3.rival_golden_trace_manifest.v1`).  Entries are strictly
/// sorted by seed; a trace declaring any other ruleset, source revision, or
/// policy identity is refused, never mixed in.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "ManifestWire", into = "ManifestWire")]
pub struct GoldenTraceManifest {
    schema_id: String,
    identity: GoldenTraceIdentity,
    trace_count: u32,
    entries: Vec<GoldenTraceManifestEntry>,
    manifest_sha256: Sha256Digest,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ManifestWire {
    schema_id: String,
    identity: GoldenTraceIdentity,
    trace_count: u32,
    entries: Vec<GoldenTraceManifestEntry>,
    manifest_sha256: Sha256Digest,
}

#[derive(Serialize)]
struct ManifestContent<'a> {
    schema_id: &'a str,
    identity: &'a GoldenTraceIdentity,
    trace_count: u32,
    entries: &'a [GoldenTraceManifestEntry],
}

impl GoldenTraceManifest {
    /// Seal a manifest over a set of validated traces.  Every trace must
    /// declare exactly the manifest's identity; seeds must be unique.  Input
    /// order is irrelevant: entries are canonically sorted by seed.
    pub fn from_traces(
        identity: GoldenTraceIdentity,
        traces: &[GoldenDecisionTrace],
    ) -> Result<Self, GoldenTraceError> {
        identity.validate()?;
        if traces.is_empty() {
            return Err(GoldenTraceError::EmptyManifest);
        }
        if traces.len() > MAX_GOLDEN_TRACE_MANIFEST_ENTRIES {
            return Err(GoldenTraceError::ManifestEntryCap {
                count: traces.len(),
                cap: MAX_GOLDEN_TRACE_MANIFEST_ENTRIES,
            });
        }
        let mut entries = Vec::with_capacity(traces.len());
        for trace in traces {
            trace.validate()?;
            let seed = trace.search.seed;
            for (field, matches) in [
                ("ruleset", trace.identity.ruleset == identity.ruleset),
                (
                    "source_revision",
                    trace.identity.source_revision == identity.source_revision,
                ),
                (
                    "policy_identity_sha256",
                    trace.identity.policy_identity_sha256 == identity.policy_identity_sha256,
                ),
            ] {
                if !matches {
                    return Err(GoldenTraceError::ManifestIdentityMismatch { seed, field });
                }
            }
            entries.push(GoldenTraceManifestEntry {
                seed,
                trace_sha256: trace.trace_sha256.clone(),
            });
        }
        entries.sort_by_key(|entry| entry.seed);
        if let Some(duplicate) = entries
            .windows(2)
            .find(|pair| pair[0].seed == pair[1].seed)
        {
            return Err(GoldenTraceError::DuplicateSeed(duplicate[0].seed));
        }
        let trace_count = u32::try_from(entries.len())
            .expect("entry count is bounded by MAX_GOLDEN_TRACE_MANIFEST_ENTRIES");
        let mut manifest = Self {
            schema_id: GOLDEN_TRACE_MANIFEST_SCHEMA_ID.to_owned(),
            identity,
            trace_count,
            entries,
            manifest_sha256: Sha256Digest::of_bytes(b""),
        };
        manifest.manifest_sha256 = manifest.recompute_hash()?;
        manifest.validate()?;
        Ok(manifest)
    }

    pub fn identity(&self) -> &GoldenTraceIdentity {
        &self.identity
    }

    pub fn trace_count(&self) -> u32 {
        self.trace_count
    }

    pub fn entries(&self) -> &[GoldenTraceManifestEntry] {
        &self.entries
    }

    pub fn manifest_sha256(&self) -> &Sha256Digest {
        &self.manifest_sha256
    }

    pub fn from_json_slice(bytes: &[u8]) -> Result<Self, GoldenTraceError> {
        Ok(serde_json::from_slice(bytes)?)
    }

    pub fn canonical_json_bytes(&self) -> Result<Vec<u8>, GoldenTraceError> {
        self.validate()?;
        Ok(serde_json::to_vec_pretty(self)?)
    }

    /// Durably publish through a same-directory temporary without ever
    /// replacing an existing artifact.
    pub fn write_json_immutable(&self, destination: &Path) -> Result<(), GoldenTraceError> {
        let bytes = self.canonical_json_bytes()?;
        crate::ledger::write_immutable_bytes(destination, &bytes)?;
        Ok(())
    }

    pub fn validate(&self) -> Result<(), GoldenTraceError> {
        if self.schema_id != GOLDEN_TRACE_MANIFEST_SCHEMA_ID {
            return Err(GoldenTraceError::WrongManifestSchema(
                self.schema_id.clone(),
            ));
        }
        self.identity.validate()?;
        if self.entries.is_empty() {
            return Err(GoldenTraceError::EmptyManifest);
        }
        if self.entries.len() > MAX_GOLDEN_TRACE_MANIFEST_ENTRIES {
            return Err(GoldenTraceError::ManifestEntryCap {
                count: self.entries.len(),
                cap: MAX_GOLDEN_TRACE_MANIFEST_ENTRIES,
            });
        }
        if usize::try_from(self.trace_count) != Ok(self.entries.len()) {
            return Err(GoldenTraceError::ManifestCountMismatch {
                trace_count: self.trace_count,
                entries: self.entries.len(),
            });
        }
        if !self
            .entries
            .windows(2)
            .all(|pair| pair[0].seed < pair[1].seed)
        {
            return Err(GoldenTraceError::UnsortedManifestEntries);
        }
        if self.recompute_hash()? != self.manifest_sha256 {
            return Err(GoldenTraceError::ManifestHashMismatch);
        }
        Ok(())
    }

    fn recompute_hash(&self) -> Result<Sha256Digest, GoldenTraceError> {
        let content = ManifestContent {
            schema_id: &self.schema_id,
            identity: &self.identity,
            trace_count: self.trace_count,
            entries: &self.entries,
        };
        let value = serde_json::to_value(&content)?;
        Ok(Sha256Digest::of_bytes(&serde_json::to_vec(&value)?))
    }
}

impl From<GoldenTraceManifest> for ManifestWire {
    fn from(value: GoldenTraceManifest) -> Self {
        Self {
            schema_id: value.schema_id,
            identity: value.identity,
            trace_count: value.trace_count,
            entries: value.entries,
            manifest_sha256: value.manifest_sha256,
        }
    }
}

impl TryFrom<ManifestWire> for GoldenTraceManifest {
    type Error = GoldenTraceError;

    fn try_from(value: ManifestWire) -> Result<Self, Self::Error> {
        let manifest = Self {
            schema_id: value.schema_id,
            identity: value.identity,
            trace_count: value.trace_count,
            entries: value.entries,
            manifest_sha256: value.manifest_sha256,
        };
        manifest.validate()?;
        Ok(manifest)
    }
}

#[derive(Debug, Error)]
pub enum GoldenTraceError {
    #[error("unsupported golden decision trace schema: {0}")]
    WrongTraceSchema(String),
    #[error("unsupported golden trace manifest schema: {0}")]
    WrongManifestSchema(String),
    #[error("golden trace field {0} must not be empty")]
    EmptyField(&'static str),
    #[error("golden trace field {0} must be nonzero")]
    ZeroField(&'static str),
    #[error("golden trace float is not finite: {0}")]
    NonFiniteFloat(String),
    #[error("golden trace float is not in canonical shortest round-trip form: {0:?}")]
    NonCanonicalFloat(String),
    #[error("golden trace field {0} must lie in [0, 1]")]
    UnitIntervalViolation(&'static str),
    #[error("golden trace field {0} must not be negative")]
    NegativeField(&'static str),
    #[error("golden trace field {0} must be strictly positive")]
    NonPositiveField(&'static str),
    #[error("acting seat {0} is outside the four-player game")]
    SeatOutOfRange(u8),
    #[error("completed turns {completed_turns} cannot exceed ply {ply}")]
    TurnAccounting { ply: u32, completed_turns: u32 },
    #[error("legal menu of {0} actions cannot be digested")]
    MenuTooLarge(usize),
    #[error("menu digest claims {action_count} actions above its own cap {menu_cap}")]
    MenuCapExceeded { action_count: u32, menu_cap: u32 },
    #[error("menu digest cap {menu} disagrees with search config cap {search}")]
    MenuCapDisagreement { menu: u32, search: u32 },
    #[error("a single-action menu must have identical first and last action ids")]
    SingletonMenuEndpointMismatch,
    #[error("chosen index {index} is outside the {action_count}-action menu")]
    ChosenIndexOutOfRange { index: u32, action_count: u32 },
    #[error("chosen action id disagrees with the recorded {0} menu action id")]
    ChosenEndpointMismatch(&'static str),
    #[error("{count} bridge exchanges exceed the trace cap {cap}")]
    BridgeExchangeCap { count: usize, cap: usize },
    #[error("bridge exchange {0} claims an empty request")]
    ZeroBridgeRequestRows(usize),
    #[error("golden trace content hash mismatch")]
    TraceHashMismatch,
    #[error("a golden trace manifest requires at least one trace")]
    EmptyManifest,
    #[error("{count} manifest entries exceed the cap {cap}")]
    ManifestEntryCap { count: usize, cap: usize },
    #[error("manifest trace_count {trace_count} does not match its {entries} entries")]
    ManifestCountMismatch { trace_count: u32, entries: usize },
    #[error("duplicate trace seed in manifest: {0}")]
    DuplicateSeed(u64),
    #[error("manifest entries must be strictly sorted by unique seed")]
    UnsortedManifestEntries,
    #[error("trace with seed {seed} declares a different {field} than the manifest")]
    ManifestIdentityMismatch { seed: u64, field: &'static str },
    #[error("golden trace manifest content hash mismatch")]
    ManifestHashMismatch,
    #[error(transparent)]
    Ruleset(#[from] RulesetIdentityError),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Ledger(#[from] LedgerError),
}
