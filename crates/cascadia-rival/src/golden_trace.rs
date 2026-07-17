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

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::*;

    fn digest(tag: &str) -> Sha256Digest {
        Sha256Digest::of_bytes(tag.as_bytes())
    }

    fn identity() -> GoldenTraceIdentity {
        GoldenTraceIdentity {
            ruleset: ResearchRulesetIdentity::canonical(),
            source_revision: "0123456789abcdef".to_owned(),
            policy_identity_sha256: digest("policy"),
        }
    }

    fn menu_ids() -> Vec<String> {
        ["alpha", "bravo", "charlie"]
            .map(str::to_owned)
            .into_iter()
            .collect()
    }

    fn f(value: f64) -> CanonicalF64 {
        CanonicalF64::new(value).unwrap()
    }

    fn input() -> GoldenDecisionTraceInput {
        GoldenDecisionTraceInput {
            identity: identity(),
            public_state_sha256: digest("public-state"),
            ply: 42,
            seat: 2,
            completed_turns: 10,
            prelude: GoldenPreludeRecord::Accepted {
                choice: "three_hawks".to_owned(),
                revealed_market_sha256: digest("revealed-market"),
            },
            menu: GoldenMenuDigest::from_ordered_action_ids(&menu_ids(), 256).unwrap(),
            search: GoldenSearchConfig {
                n_simulations: 64,
                top_m: 16,
                depth_rounds: 1,
                determinizations: 4,
                market_decision_samples: 1,
                exact_endgame_turns: 0,
                blend_weight: f(0.5),
                root_menu_cap: 256,
                c_visit: f(50.0),
                c_scale: f(1.0),
                seed: 7,
            },
            bridge_exchanges: vec![
                BridgeExchangeDigest {
                    request_rows: 3,
                    request_sha256: digest("request-0"),
                    response_sha256: digest("response-0"),
                },
                BridgeExchangeDigest {
                    request_rows: 17,
                    request_sha256: digest("request-1"),
                    response_sha256: digest("response-1"),
                },
            ],
            chosen: GoldenChosenAction {
                action_id: "bravo".to_owned(),
                menu_index: 1,
                completed_q: f(61.25),
                improved_policy_mass: f(0.375),
            },
        }
    }

    fn trace() -> GoldenDecisionTrace {
        GoldenDecisionTrace::new(input()).unwrap()
    }

    fn mutated(mutator: impl FnOnce(&mut GoldenDecisionTraceInput)) -> GoldenDecisionTrace {
        let mut input = input();
        mutator(&mut input);
        GoldenDecisionTrace::new(input).unwrap()
    }

    fn temp_destination(tag: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "cascadia-rival-golden-trace-test-{}-{tag}.json",
            std::process::id()
        ))
    }

    #[test]
    fn canonical_f64_locks_the_shortest_round_trip_wire_form() {
        for (value, text) in [
            (0.5, "0.5"),
            (50.0, "50"),
            (-0.0, "-0"),
            (61.25, "61.25"),
            (1.0, "1"),
        ] {
            let canonical = f(value);
            assert_eq!(canonical.canonical_text(), text);
            let json = serde_json::to_string(&canonical).unwrap();
            assert_eq!(json, format!("{text:?}"));
            assert_eq!(
                serde_json::from_str::<CanonicalF64>(&json).unwrap(),
                canonical
            );
        }
        for text in ["0.50", "00.5", ".5", "+0.5", "1.0", "NaN", "inf", "-inf", ""] {
            assert!(
                serde_json::from_str::<CanonicalF64>(&format!("{text:?}")).is_err(),
                "non-canonical float unexpectedly accepted: {text:?}"
            );
        }
        assert!(CanonicalF64::new(f64::NAN).is_err());
        assert!(CanonicalF64::new(f64::INFINITY).is_err());
        assert!(CanonicalF64::new(f64::NEG_INFINITY).is_err());
    }

    #[test]
    fn trace_round_trips_and_revalidates() {
        let original = trace();
        let bytes = original.canonical_json_bytes().unwrap();
        let restored = GoldenDecisionTrace::from_json_slice(&bytes).unwrap();
        assert_eq!(restored, original);
        assert_eq!(restored.trace_sha256(), original.trace_sha256());
        assert_eq!(restored.canonical_json_bytes().unwrap(), bytes);
    }

    #[test]
    fn trace_hash_is_deterministic_and_content_bound() {
        assert_eq!(trace().trace_sha256(), trace().trace_sha256());
        assert_ne!(
            mutated(|input| input.ply = 43).trace_sha256(),
            trace().trace_sha256()
        );
        assert_ne!(
            mutated(|input| input.bridge_exchanges.truncate(1)).trace_sha256(),
            trace().trace_sha256()
        );
    }

    #[test]
    fn every_trace_leaf_field_is_required() {
        let original = serde_json::to_value(trace()).unwrap();
        let mut paths = Vec::new();
        collect_leaf_paths(&original, &mut Vec::new(), &mut paths);
        assert!(paths.len() >= 30, "unexpectedly shallow trace: {}", paths.len());
        for path in paths {
            let mut changed = original.clone();
            remove_at_path(&mut changed, &path);
            assert!(
                serde_json::from_value::<GoldenDecisionTrace>(changed).is_err(),
                "missing field unexpectedly accepted: {}",
                path.join(".")
            );
        }
    }

    #[test]
    fn any_trace_leaf_perturbation_fails_closed() {
        let original = serde_json::to_value(trace()).unwrap();
        let mut paths = Vec::new();
        collect_leaf_paths(&original, &mut Vec::new(), &mut paths);
        for path in paths {
            let mut changed = original.clone();
            perturb_at_path(&mut changed, &path);
            assert!(
                serde_json::from_value::<GoldenDecisionTrace>(changed).is_err(),
                "perturbed field was not hash- or schema-bound: {}",
                path.join(".")
            );
        }
    }

    #[test]
    fn unknown_fields_fail_closed_at_every_structured_layer() {
        for path in [
            vec![],
            vec!["identity"],
            vec!["menu"],
            vec!["search"],
            vec!["bridge_exchanges", "0"],
            vec!["chosen"],
        ] {
            let mut value = serde_json::to_value(trace()).unwrap();
            let mut cursor = &mut value;
            for key in &path {
                cursor = descend(cursor, key);
            }
            cursor["unknown"] = json!(1);
            assert!(
                serde_json::from_value::<GoldenDecisionTrace>(value).is_err(),
                "unknown field unexpectedly accepted at {}",
                path.join(".")
            );
        }
    }

    #[test]
    fn invalid_traces_are_refused_at_construction() {
        assert!(matches!(
            GoldenDecisionTrace::new(GoldenDecisionTraceInput {
                seat: 4,
                ..input()
            }),
            Err(GoldenTraceError::SeatOutOfRange(4))
        ));
        assert!(matches!(
            GoldenDecisionTrace::new(GoldenDecisionTraceInput {
                ply: 5,
                completed_turns: 6,
                ..input()
            }),
            Err(GoldenTraceError::TurnAccounting { .. })
        ));
        let mut cap_disagreement = input();
        cap_disagreement.search.root_menu_cap = 128;
        assert!(matches!(
            GoldenDecisionTrace::new(cap_disagreement),
            Err(GoldenTraceError::MenuCapDisagreement {
                menu: 256,
                search: 128
            })
        ));
        let mut chosen_out_of_range = input();
        chosen_out_of_range.chosen.menu_index = 3;
        assert!(matches!(
            GoldenDecisionTrace::new(chosen_out_of_range),
            Err(GoldenTraceError::ChosenIndexOutOfRange {
                index: 3,
                action_count: 3
            })
        ));
        let mut wrong_endpoint = input();
        wrong_endpoint.chosen.menu_index = 0;
        assert!(matches!(
            GoldenDecisionTrace::new(wrong_endpoint),
            Err(GoldenTraceError::ChosenEndpointMismatch("first"))
        ));
        let mut wrong_last = input();
        wrong_last.chosen.menu_index = 2;
        assert!(matches!(
            GoldenDecisionTrace::new(wrong_last),
            Err(GoldenTraceError::ChosenEndpointMismatch("last"))
        ));
        let mut empty_request = input();
        empty_request.bridge_exchanges[1].request_rows = 0;
        assert!(matches!(
            GoldenDecisionTrace::new(empty_request),
            Err(GoldenTraceError::ZeroBridgeRequestRows(1))
        ));
        let mut over_mass = input();
        over_mass.chosen.improved_policy_mass = f(1.5);
        assert!(matches!(
            GoldenDecisionTrace::new(over_mass),
            Err(GoldenTraceError::UnitIntervalViolation(
                "chosen.improved_policy_mass"
            ))
        ));
        let mut over_blend = input();
        over_blend.search.blend_weight = f(1.5);
        assert!(matches!(
            GoldenDecisionTrace::new(over_blend),
            Err(GoldenTraceError::UnitIntervalViolation("search.blend_weight"))
        ));
        let mut zero_scale = input();
        zero_scale.search.c_scale = f(0.0);
        assert!(matches!(
            GoldenDecisionTrace::new(zero_scale),
            Err(GoldenTraceError::NonPositiveField("search.c_scale"))
        ));
        let mut empty_revision = input();
        empty_revision.identity.source_revision = "  ".to_owned();
        assert!(matches!(
            GoldenDecisionTrace::new(empty_revision),
            Err(GoldenTraceError::EmptyField("identity.source_revision"))
        ));
        assert!(matches!(
            GoldenMenuDigest::from_ordered_action_ids(&[], 0),
            Err(GoldenTraceError::ZeroField("menu.action_count"))
        ));
        assert!(matches!(
            GoldenMenuDigest::from_ordered_action_ids(&menu_ids(), 2),
            Err(GoldenTraceError::MenuCapExceeded {
                action_count: 3,
                menu_cap: 2
            })
        ));
    }

    #[test]
    fn comparator_accepts_identical_traces() {
        assert_eq!(compare_traces(&trace(), &trace()), Ok(()));
        let round_tripped =
            GoldenDecisionTrace::from_json_slice(&trace().canonical_json_bytes().unwrap()).unwrap();
        assert_eq!(compare_traces(&trace(), &round_tripped), Ok(()));
    }

    #[test]
    fn comparator_pinpoints_the_first_divergent_field_in_every_class() {
        let reference = trace();
        let cases: Vec<(GoldenDecisionTrace, TraceDivergence)> = vec![
            (
                mutated(|input| input.identity.source_revision = "fedcba9876543210".to_owned()),
                TraceDivergence::SourceRevision {
                    reference: "0123456789abcdef".to_owned(),
                    candidate: "fedcba9876543210".to_owned(),
                },
            ),
            (
                mutated(|input| input.identity.policy_identity_sha256 = digest("other-policy")),
                TraceDivergence::PolicyIdentitySha256 {
                    reference: digest("policy").to_string(),
                    candidate: digest("other-policy").to_string(),
                },
            ),
            (
                mutated(|input| input.public_state_sha256 = digest("other-state")),
                TraceDivergence::PublicStateSha256 {
                    reference: digest("public-state").to_string(),
                    candidate: digest("other-state").to_string(),
                },
            ),
            (
                mutated(|input| input.ply = 43),
                TraceDivergence::Ply {
                    reference: "42".to_owned(),
                    candidate: "43".to_owned(),
                },
            ),
            (
                mutated(|input| input.seat = 3),
                TraceDivergence::Seat {
                    reference: "2".to_owned(),
                    candidate: "3".to_owned(),
                },
            ),
            (
                mutated(|input| input.completed_turns = 11),
                TraceDivergence::CompletedTurns {
                    reference: "10".to_owned(),
                    candidate: "11".to_owned(),
                },
            ),
            (
                mutated(|input| input.prelude = GoldenPreludeRecord::Declined),
                TraceDivergence::Prelude {
                    reference: format!(
                        "accepted(three_hawks, {})",
                        digest("revealed-market")
                    ),
                    candidate: "declined".to_owned(),
                },
            ),
            (
                mutated(|input| {
                    input.menu = GoldenMenuDigest::from_ordered_action_ids(
                        &["alpha", "bravo", "charlie", "delta"].map(str::to_owned),
                        256,
                    )
                    .unwrap();
                }),
                TraceDivergence::MenuActionCount {
                    reference: "3".to_owned(),
                    candidate: "4".to_owned(),
                },
            ),
            (
                mutated(|input| {
                    input.menu = GoldenMenuDigest::from_ordered_action_ids(&menu_ids(), 300)
                        .unwrap();
                    input.search.root_menu_cap = 300;
                }),
                TraceDivergence::MenuCap {
                    reference: "256".to_owned(),
                    candidate: "300".to_owned(),
                },
            ),
            (
                mutated(|input| {
                    input.menu = GoldenMenuDigest::from_ordered_action_ids(
                        &["zulu", "bravo", "charlie"].map(str::to_owned),
                        256,
                    )
                    .unwrap();
                }),
                TraceDivergence::MenuFirstActionId {
                    reference: "alpha".to_owned(),
                    candidate: "zulu".to_owned(),
                },
            ),
            (
                mutated(|input| {
                    input.menu = GoldenMenuDigest::from_ordered_action_ids(
                        &["alpha", "bravo", "zulu"].map(str::to_owned),
                        256,
                    )
                    .unwrap();
                    input.chosen.action_id = "bravo".to_owned();
                }),
                TraceDivergence::MenuLastActionId {
                    reference: "charlie".to_owned(),
                    candidate: "zulu".to_owned(),
                },
            ),
            (
                mutated(|input| {
                    input.menu = GoldenMenuDigest::from_ordered_action_ids(
                        &["alpha", "xray", "charlie"].map(str::to_owned),
                        256,
                    )
                    .unwrap();
                }),
                TraceDivergence::MenuActionIdsSha256 {
                    reference: GoldenMenuDigest::from_ordered_action_ids(&menu_ids(), 256)
                        .unwrap()
                        .action_ids_sha256
                        .to_string(),
                    candidate: GoldenMenuDigest::from_ordered_action_ids(
                        &["alpha", "xray", "charlie"].map(str::to_owned),
                        256,
                    )
                    .unwrap()
                    .action_ids_sha256
                    .to_string(),
                },
            ),
            (
                mutated(|input| input.search.n_simulations = 128),
                TraceDivergence::NSimulations {
                    reference: "64".to_owned(),
                    candidate: "128".to_owned(),
                },
            ),
            (
                mutated(|input| input.search.top_m = 8),
                TraceDivergence::TopM {
                    reference: "16".to_owned(),
                    candidate: "8".to_owned(),
                },
            ),
            (
                mutated(|input| input.search.depth_rounds = 2),
                TraceDivergence::DepthRounds {
                    reference: "1".to_owned(),
                    candidate: "2".to_owned(),
                },
            ),
            (
                mutated(|input| input.search.determinizations = 8),
                TraceDivergence::Determinizations {
                    reference: "4".to_owned(),
                    candidate: "8".to_owned(),
                },
            ),
            (
                mutated(|input| input.search.market_decision_samples = 2),
                TraceDivergence::MarketDecisionSamples {
                    reference: "1".to_owned(),
                    candidate: "2".to_owned(),
                },
            ),
            (
                mutated(|input| input.search.exact_endgame_turns = 1),
                TraceDivergence::ExactEndgameTurns {
                    reference: "0".to_owned(),
                    candidate: "1".to_owned(),
                },
            ),
            (
                mutated(|input| input.search.blend_weight = f(0.25)),
                TraceDivergence::BlendWeight {
                    reference: "0.5".to_owned(),
                    candidate: "0.25".to_owned(),
                },
            ),
            (
                mutated(|input| input.search.c_visit = f(60.0)),
                TraceDivergence::CVisit {
                    reference: "50".to_owned(),
                    candidate: "60".to_owned(),
                },
            ),
            (
                mutated(|input| input.search.c_scale = f(2.0)),
                TraceDivergence::CScale {
                    reference: "1".to_owned(),
                    candidate: "2".to_owned(),
                },
            ),
            (
                mutated(|input| input.search.seed = 8),
                TraceDivergence::Seed {
                    reference: "7".to_owned(),
                    candidate: "8".to_owned(),
                },
            ),
            (
                mutated(|input| input.bridge_exchanges[0].request_rows = 4),
                TraceDivergence::BridgeRequestRows {
                    index: 0,
                    reference: "3".to_owned(),
                    candidate: "4".to_owned(),
                },
            ),
            (
                mutated(|input| {
                    input.bridge_exchanges[0].request_sha256 = digest("other-request-0");
                }),
                TraceDivergence::BridgeRequestSha256 {
                    index: 0,
                    reference: digest("request-0").to_string(),
                    candidate: digest("other-request-0").to_string(),
                },
            ),
            (
                mutated(|input| {
                    input.bridge_exchanges[1].response_sha256 = digest("other-response-1");
                }),
                TraceDivergence::BridgeResponseSha256 {
                    index: 1,
                    reference: digest("response-1").to_string(),
                    candidate: digest("other-response-1").to_string(),
                },
            ),
            (
                mutated(|input| input.bridge_exchanges.truncate(1)),
                TraceDivergence::BridgeExchangeCount {
                    reference: "2".to_owned(),
                    candidate: "1".to_owned(),
                },
            ),
            (
                mutated(|input| {
                    input.bridge_exchanges.push(BridgeExchangeDigest {
                        request_rows: 5,
                        request_sha256: digest("request-2"),
                        response_sha256: digest("response-2"),
                    });
                }),
                TraceDivergence::BridgeExchangeCount {
                    reference: "2".to_owned(),
                    candidate: "3".to_owned(),
                },
            ),
            (
                mutated(|input| {
                    input.chosen.menu_index = 2;
                    input.chosen.action_id = "charlie".to_owned();
                }),
                TraceDivergence::ChosenMenuIndex {
                    reference: "1".to_owned(),
                    candidate: "2".to_owned(),
                },
            ),
            (
                mutated(|input| input.chosen.action_id = "delta".to_owned()),
                TraceDivergence::ChosenActionId {
                    reference: "bravo".to_owned(),
                    candidate: "delta".to_owned(),
                },
            ),
            (
                mutated(|input| input.chosen.completed_q = f(59.5)),
                TraceDivergence::ChosenCompletedQ {
                    reference: "61.25".to_owned(),
                    candidate: "59.5".to_owned(),
                },
            ),
            (
                mutated(|input| input.chosen.improved_policy_mass = f(0.5)),
                TraceDivergence::ChosenImprovedPolicyMass {
                    reference: "0.375".to_owned(),
                    candidate: "0.5".to_owned(),
                },
            ),
        ];
        for (candidate, expected) in cases {
            assert_eq!(
                compare_traces(&reference, &candidate),
                Err(expected.clone()),
                "wrong divergence for candidate mutation expecting {expected:?}"
            );
            // The comparator is directional in its labels but symmetric in
            // detection: the reverse comparison must also diverge.
            assert!(compare_traces(&candidate, &reference).is_err());
        }
    }

    #[test]
    fn trace_publication_is_immutable() {
        let destination = temp_destination("trace");
        let _ = std::fs::remove_file(&destination);
        let original = trace();
        original.write_json_immutable(&destination).unwrap();
        assert!(matches!(
            original.write_json_immutable(&destination),
            Err(GoldenTraceError::Ledger(LedgerError::ArtifactAlreadyExists(
                _
            )))
        ));
        let bytes = std::fs::read(&destination).unwrap();
        assert_eq!(
            GoldenDecisionTrace::from_json_slice(&bytes).unwrap(),
            original
        );
        std::fs::remove_file(&destination).unwrap();
    }

    #[test]
    fn manifest_round_trips_and_sorts_entries_by_seed() {
        let traces = vec![
            mutated(|input| input.search.seed = 11),
            mutated(|input| input.search.seed = 3),
            trace(),
        ];
        let manifest = GoldenTraceManifest::from_traces(identity(), &traces).unwrap();
        assert_eq!(manifest.trace_count(), 3);
        assert_eq!(
            manifest
                .entries()
                .iter()
                .map(|entry| entry.seed)
                .collect::<Vec<_>>(),
            vec![3, 7, 11]
        );
        assert_eq!(
            manifest.entries()[1].trace_sha256,
            *trace().trace_sha256()
        );
        let bytes = manifest.canonical_json_bytes().unwrap();
        let restored = GoldenTraceManifest::from_json_slice(&bytes).unwrap();
        assert_eq!(restored, manifest);
        assert_eq!(restored.manifest_sha256(), manifest.manifest_sha256());
    }

    #[test]
    fn manifest_refuses_duplicates_empties_and_foreign_identities() {
        assert!(matches!(
            GoldenTraceManifest::from_traces(identity(), &[]),
            Err(GoldenTraceError::EmptyManifest)
        ));
        assert!(matches!(
            GoldenTraceManifest::from_traces(identity(), &[trace(), trace()]),
            Err(GoldenTraceError::DuplicateSeed(7))
        ));
        let foreign_revision =
            mutated(|input| input.identity.source_revision = "fedcba9876543210".to_owned());
        assert!(matches!(
            GoldenTraceManifest::from_traces(identity(), &[trace(), foreign_revision]),
            Err(GoldenTraceError::ManifestIdentityMismatch {
                seed: 7,
                field: "source_revision"
            })
        ));
        let foreign_policy =
            mutated(|input| input.identity.policy_identity_sha256 = digest("other-policy"));
        assert!(matches!(
            GoldenTraceManifest::from_traces(identity(), &[foreign_policy]),
            Err(GoldenTraceError::ManifestIdentityMismatch {
                seed: 7,
                field: "policy_identity_sha256"
            })
        ));
    }

    #[test]
    fn every_manifest_leaf_field_is_required_and_hash_bound() {
        let manifest = GoldenTraceManifest::from_traces(
            identity(),
            &[trace(), mutated(|input| input.search.seed = 9)],
        )
        .unwrap();
        let original = serde_json::to_value(&manifest).unwrap();
        let mut paths = Vec::new();
        collect_leaf_paths(&original, &mut Vec::new(), &mut paths);
        assert!(
            paths.len() >= 10,
            "unexpectedly shallow manifest: {}",
            paths.len()
        );
        for path in &paths {
            let mut changed = original.clone();
            remove_at_path(&mut changed, path);
            assert!(
                serde_json::from_value::<GoldenTraceManifest>(changed).is_err(),
                "missing manifest field unexpectedly accepted: {}",
                path.join(".")
            );
        }
        for path in &paths {
            let mut changed = original.clone();
            perturb_at_path(&mut changed, path);
            assert!(
                serde_json::from_value::<GoldenTraceManifest>(changed).is_err(),
                "perturbed manifest field was not hash- or schema-bound: {}",
                path.join(".")
            );
        }
        let mut unknown = original.clone();
        unknown["unknown"] = json!(1);
        assert!(serde_json::from_value::<GoldenTraceManifest>(unknown).is_err());
        let mut nested_unknown = original;
        nested_unknown["entries"][0]["unknown"] = json!(1);
        assert!(serde_json::from_value::<GoldenTraceManifest>(nested_unknown).is_err());
    }

    #[test]
    fn manifest_publication_is_immutable() {
        let destination = temp_destination("manifest");
        let _ = std::fs::remove_file(&destination);
        let manifest = GoldenTraceManifest::from_traces(identity(), &[trace()]).unwrap();
        manifest.write_json_immutable(&destination).unwrap();
        assert!(matches!(
            manifest.write_json_immutable(&destination),
            Err(GoldenTraceError::Ledger(LedgerError::ArtifactAlreadyExists(
                _
            )))
        ));
        let bytes = std::fs::read(&destination).unwrap();
        assert_eq!(
            GoldenTraceManifest::from_json_slice(&bytes).unwrap(),
            manifest
        );
        std::fs::remove_file(&destination).unwrap();
    }

    #[test]
    fn trace_and_manifest_hashes_are_golden() {
        // Locked wire-identity goldens: any schema, canonicalization, or
        // hashing change to the v1 contracts must be deliberate and visible.
        assert_eq!(
            trace().trace_sha256().as_str(),
            TRACE_GOLDEN_SHA256,
            "golden decision trace v1 wire identity changed"
        );
        let manifest = GoldenTraceManifest::from_traces(identity(), &[trace()]).unwrap();
        assert_eq!(
            manifest.manifest_sha256().as_str(),
            MANIFEST_GOLDEN_SHA256,
            "golden trace manifest v1 wire identity changed"
        );
    }

    const TRACE_GOLDEN_SHA256: &str =
        "sha256:ede7e2032171e3019425209d9df3f5b697ed9b2a0517f8aa05473ad580a12e47";
    const MANIFEST_GOLDEN_SHA256: &str =
        "sha256:0225f457998c6040d2df75013eff5391e16d9938be031900ceada214e3428d01";

    fn collect_leaf_paths(value: &Value, prefix: &mut Vec<String>, out: &mut Vec<Vec<String>>) {
        match value {
            Value::Object(map) => {
                for (key, child) in map {
                    prefix.push(key.clone());
                    collect_leaf_paths(child, prefix, out);
                    prefix.pop();
                }
            }
            Value::Array(items) => {
                for (index, child) in items.iter().enumerate() {
                    prefix.push(index.to_string());
                    collect_leaf_paths(child, prefix, out);
                    prefix.pop();
                }
            }
            _ => out.push(prefix.clone()),
        }
    }

    fn descend<'a>(value: &'a mut Value, key: &str) -> &'a mut Value {
        if value.is_array() {
            let index: usize = key.parse().expect("array path segments are indices");
            &mut value[index]
        } else {
            &mut value[key]
        }
    }

    fn remove_at_path(value: &mut Value, path: &[String]) {
        let (last, parents) = path.split_last().unwrap();
        let mut cursor = value;
        for key in parents {
            cursor = descend(cursor, key);
        }
        if let Some(items) = cursor.as_array_mut() {
            items.remove(last.parse().expect("array path segments are indices"));
        } else {
            cursor.as_object_mut().unwrap().remove(last);
        }
    }

    fn perturb_at_path(value: &mut Value, path: &[String]) {
        let mut cursor = value;
        for key in path {
            cursor = descend(cursor, key);
        }
        *cursor = match cursor {
            Value::String(text) if text.starts_with("sha256:") => {
                Value::String(format!("sha256:{}", "f".repeat(64)))
            }
            Value::String(text) => Value::String(format!("{text}-perturbed")),
            Value::Bool(flag) => Value::Bool(!*flag),
            Value::Number(number) => json!(number.as_i64().unwrap_or_default() + 1),
            _ => unreachable!("golden trace leaves are scalar"),
        };
    }
}
