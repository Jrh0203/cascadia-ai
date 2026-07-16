//! CPU physical-world reference sampling.
//!
//! The only pre-hook mode independently redetermines hidden tile partitions
//! and wildlife order for every branch/fidelity coordinate.  Because the
//! canonical engine still retains its original private seed for future
//! wildlife-return insertion, this mode is conservatively named an
//! independent-*hidden-order* reference.  It is valid for high-fidelity-only
//! plumbing with `beta_cv = 0`; it is not admitted as production Rival-MF
//! coupling or covariance evidence.

use std::fmt;

use cascadia_game::{GameConfig, GameSeed, GameState};
use rand::RngCore;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    CouplingGroupId, EvaluationBranch, Fidelity, MenuComposer, PrivateSimState, PublicRootId,
    RngFactory, RootKind, RulesLegalMenu, SeatIndex, SeatLocalMemory, Sha256Digest,
};

pub const INDEPENDENT_SCENARIO_SAMPLER_ID: &str =
    "cascadiav3.rival_independent_hidden_order_reference.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScenarioMode {
    IndependentHiddenOrderReference,
}

impl ScenarioMode {
    pub const fn required_beta_cv(self) -> f64 {
        match self {
            Self::IndependentHiddenOrderReference => 0.0,
        }
    }

    pub const fn production_multifidelity_eligible(self) -> bool {
        false
    }
}

/// Complete coordinate for one independently keyed physical reference world.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScenarioCoordinate {
    pub panel_id: Sha256Digest,
    pub unit_index: u32,
    pub branch: EvaluationBranch,
    pub fidelity: Fidelity,
}

impl ScenarioCoordinate {
    fn coupling_key(
        &self,
        public_root_id: &PublicRootId,
        sampler_identity_sha256: &Sha256Digest,
    ) -> String {
        #[derive(Serialize)]
        struct Key<'a> {
            sampler_id: &'static str,
            public_root_id: &'a PublicRootId,
            sampler_identity_sha256: &'a Sha256Digest,
            coordinate: &'a ScenarioCoordinate,
        }
        serde_json::to_string(&Key {
            sampler_id: INDEPENDENT_SCENARIO_SAMPLER_ID,
            public_root_id,
            sampler_identity_sha256,
            coordinate: self,
        })
        .expect("serializing a validated scenario coordinate cannot fail")
    }
}

/// Trusted sampler capability.  It is never passed to a policy.
pub struct IndependentScenarioSampler {
    source: GameState,
    initial_memories: [SeatLocalMemory; 4],
    rng_factory: RngFactory,
    public_root_id: PublicRootId,
    source_game_identity_sha256: Sha256Digest,
    sampler_identity_sha256: Sha256Digest,
}

impl IndependentScenarioSampler {
    /// Freeze an already revealed post-prelude draft root.
    ///
    /// `source` is the canonical staged state *after* any free replacement and
    /// paid wipes have resolved. Forced candidates are consequently required
    /// to be root-local actions with an empty `MarketPrelude`; replay rejects
    /// compound pre-prelude actions at this boundary.
    pub fn new(
        source: GameState,
        initial_memories: [SeatLocalMemory; 4],
        rng_factory: RngFactory,
    ) -> Result<Self, ScenarioError> {
        if source.config() != GameConfig::research_aaaaa(4).expect("valid research config") {
            return Err(ScenarioError::UnsupportedRuleset);
        }
        source.validate().map_err(ScenarioError::InvalidSource)?;
        let actor =
            SeatIndex::new(source.current_player() as u8).map_err(ScenarioError::Observation)?;
        let observation = PrivateSimState::new(source.clone())
            .map_err(ScenarioError::Observation)?
            .public_observation(actor, initial_memories[usize::from(actor.get())].clone())
            .map_err(ScenarioError::Observation)?;
        let public_root_id = PublicRootId::new(&observation, RootKind::DraftPolicyRoot);
        let rules_menu = MenuComposer::draft_root(&source, &Default::default())?;
        let source_game_identity_sha256 = Sha256Digest::of_bytes(
            &serde_json::to_vec(&source)
                .expect("serializing a validated source game for identity cannot fail"),
        );
        #[derive(Serialize)]
        struct SamplerIdentity<'a> {
            sampler_id: &'static str,
            mode: ScenarioMode,
            source_game_identity_sha256: &'a Sha256Digest,
            public_root_id: &'a PublicRootId,
            rules_menu_hash: crate::RulesMenuHash,
            initial_memories: &'a [SeatLocalMemory; 4],
            outer_rng_factory_identity_sha256: Sha256Digest,
        }
        let sampler_identity_sha256 = Sha256Digest::of_bytes(
            &serde_json::to_vec(&SamplerIdentity {
                sampler_id: INDEPENDENT_SCENARIO_SAMPLER_ID,
                mode: ScenarioMode::IndependentHiddenOrderReference,
                source_game_identity_sha256: &source_game_identity_sha256,
                public_root_id: &public_root_id,
                rules_menu_hash: rules_menu.hash(),
                initial_memories: &initial_memories,
                outer_rng_factory_identity_sha256: rng_factory.identity_sha256(),
            })
            .expect("serializing a fixed sampler identity cannot fail"),
        );
        Ok(Self {
            public_root_id,
            source,
            initial_memories,
            rng_factory,
            source_game_identity_sha256,
            sampler_identity_sha256,
        })
    }

    pub fn mode(&self) -> ScenarioMode {
        ScenarioMode::IndependentHiddenOrderReference
    }

    pub fn source_game_identity_sha256(&self) -> &Sha256Digest {
        &self.source_game_identity_sha256
    }

    pub fn sampler_identity_sha256(&self) -> &Sha256Digest {
        &self.sampler_identity_sha256
    }

    pub fn public_root_id(&self) -> &PublicRootId {
        &self.public_root_id
    }

    pub fn initial_memories(&self) -> &[SeatLocalMemory; 4] {
        &self.initial_memories
    }

    pub(crate) fn source(&self) -> &GameState {
        &self.source
    }

    pub(crate) fn rules_menu(&self) -> Result<RulesLegalMenu, ScenarioError> {
        Ok(MenuComposer::draft_root(&self.source, &Default::default())?)
    }

    pub(crate) fn sample(&self, coordinate: &ScenarioCoordinate) -> ReferenceWorld {
        let group = CouplingGroupId::from_manifest_key(
            &coordinate.coupling_key(&self.public_root_id, &self.sampler_identity_sha256),
        );
        let mut physical = self.rng_factory.outer_physical(group, 0);
        let mut seed_bytes = [0u8; 32];
        physical.fill_bytes(&mut seed_bytes);
        let redetermination_seed = GameSeed(seed_bytes);
        let mut game = self.source.clone();
        game.redeterminize_hidden(redetermination_seed);
        ReferenceWorld {
            coordinate: coordinate.clone(),
            redetermination_seed,
            game,
        }
    }
}

impl fmt::Debug for IndependentScenarioSampler {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("IndependentScenarioSampler(<opaque source and RNG>)")
    }
}

/// Opaque physical reference world owned only by trusted orchestration.
pub(crate) struct ReferenceWorld {
    coordinate: ScenarioCoordinate,
    redetermination_seed: GameSeed,
    game: GameState,
}

impl ReferenceWorld {
    pub(crate) fn coordinate(&self) -> &ScenarioCoordinate {
        &self.coordinate
    }

    pub(crate) fn public_hash(&self) -> blake3::Hash {
        self.game.public_state().canonical_hash()
    }

    pub(crate) fn canonical_hash(&self) -> blake3::Hash {
        self.game.canonical_hash()
    }

    pub(crate) fn game(&self) -> &GameState {
        &self.game
    }

    pub(crate) fn game_mut(&mut self) -> &mut GameState {
        &mut self.game
    }

    pub(crate) fn redetermination_seed(&self) -> GameSeed {
        self.redetermination_seed
    }
}

impl fmt::Debug for ReferenceWorld {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ReferenceWorld")
            .field("coordinate", &self.coordinate)
            .field("private_state", &"<opaque>")
            .finish()
    }
}

#[derive(Debug, Error)]
pub enum ScenarioError {
    #[error("scenario sampler supports only corrected-rules four-player research AAAAA")]
    UnsupportedRuleset,
    #[error("source game state is invalid: {0}")]
    InvalidSource(&'static str),
    #[error("source public-root construction failed: {0}")]
    Observation(crate::ObservationError),
    #[error(transparent)]
    Menu(#[from] crate::MenuError),
}

#[cfg(test)]
mod tests {
    use crate::RivalSeed;

    use super::*;

    fn coordinate(branch: EvaluationBranch, fidelity: Fidelity) -> ScenarioCoordinate {
        ScenarioCoordinate {
            panel_id: Sha256Digest::of_bytes(b"cpu-panel"),
            unit_index: 3,
            branch,
            fidelity,
        }
    }

    fn empty_memories() -> [SeatLocalMemory; 4] {
        std::array::from_fn(|_| SeatLocalMemory::empty())
    }

    #[test]
    fn coordinate_replay_is_deterministic_and_public_facts_are_invariant() {
        let source = GameState::new(
            GameConfig::research_aaaaa(4).unwrap(),
            GameSeed::from_u64(18),
        )
        .unwrap();
        let public = source.public_state();
        let supply = source.public_supply();
        let sampler = IndependentScenarioSampler::new(
            source,
            empty_memories(),
            RngFactory::new(RivalSeed::from_u64(99)),
        )
        .unwrap();
        let coordinate = coordinate(EvaluationBranch::Incumbent, Fidelity::High);
        let left = sampler.sample(&coordinate);
        let right = sampler.sample(&coordinate);
        assert_eq!(left.canonical_hash(), right.canonical_hash());
        assert_eq!(left.game().public_state(), public);
        assert_eq!(left.game().public_supply(), supply);
        assert_eq!(left.public_hash(), public.canonical_hash());
    }

    #[test]
    fn branch_and_fidelity_coordinates_do_not_reuse_hidden_order() {
        let source = GameState::new(
            GameConfig::research_aaaaa(4).unwrap(),
            GameSeed::from_u64(19),
        )
        .unwrap();
        let sampler = IndependentScenarioSampler::new(
            source,
            empty_memories(),
            RngFactory::new(RivalSeed::from_u64(100)),
        )
        .unwrap();
        let incumbent = sampler.sample(&coordinate(EvaluationBranch::Incumbent, Fidelity::High));
        let challenger =
            sampler.sample(&coordinate(EvaluationBranch::Challenger(0), Fidelity::High));
        let low = sampler.sample(&coordinate(EvaluationBranch::Incumbent, Fidelity::Low));
        assert_ne!(incumbent.canonical_hash(), challenger.canonical_hash());
        assert_ne!(incumbent.canonical_hash(), low.canonical_hash());
        assert_eq!(incumbent.public_hash(), challenger.public_hash());
    }

    #[test]
    fn reference_mode_cannot_be_mislabeled_multifidelity() {
        let mode = ScenarioMode::IndependentHiddenOrderReference;
        assert_eq!(mode.required_beta_cv(), 0.0);
        assert!(!mode.production_multifidelity_eligible());
    }

    #[test]
    fn complete_source_identity_prevents_public_state_key_collisions() {
        let source = GameState::new(
            GameConfig::research_aaaaa(4).unwrap(),
            GameSeed::from_u64(21),
        )
        .unwrap();
        let mut value = serde_json::to_value(&source).unwrap();
        let moved = value["wildlife_bag"].as_array_mut().unwrap().pop().unwrap();
        value["discarded_wildlife"]
            .as_array_mut()
            .unwrap()
            .push(moved);
        let changed: GameState = serde_json::from_value(value).unwrap();
        changed.validate().unwrap();
        assert_eq!(source.public_state(), changed.public_state());
        assert_ne!(source.public_supply(), changed.public_supply());

        let left = IndependentScenarioSampler::new(
            source,
            empty_memories(),
            RngFactory::new(RivalSeed::from_u64(101)),
        )
        .unwrap();
        let right = IndependentScenarioSampler::new(
            changed,
            empty_memories(),
            RngFactory::new(RivalSeed::from_u64(101)),
        )
        .unwrap();
        assert_ne!(
            left.source_game_identity_sha256(),
            right.source_game_identity_sha256()
        );
        assert_ne!(
            left.sampler_identity_sha256(),
            right.sampler_identity_sha256()
        );
        let coordinate = coordinate(EvaluationBranch::Incumbent, Fidelity::High);
        assert_ne!(
            left.sample(&coordinate).redetermination_seed(),
            right.sample(&coordinate).redetermination_seed()
        );
    }

    #[test]
    fn acting_seat_memory_is_bound_into_the_root_and_sampler_identity() {
        let source = GameState::new(
            GameConfig::research_aaaaa(4).unwrap(),
            GameSeed::from_u64(22),
        )
        .unwrap();
        let base = empty_memories();
        let mut changed = base.clone();
        changed[source.current_player()] = SeatLocalMemory::new(vec![7, 9]);
        let left = IndependentScenarioSampler::new(
            source.clone(),
            base,
            RngFactory::new(RivalSeed::from_u64(102)),
        )
        .unwrap();
        let right = IndependentScenarioSampler::new(
            source,
            changed.clone(),
            RngFactory::new(RivalSeed::from_u64(102)),
        )
        .unwrap();
        assert_ne!(left.public_root_id(), right.public_root_id());
        assert_ne!(
            left.sampler_identity_sha256(),
            right.sampler_identity_sha256()
        );
        assert_eq!(right.initial_memories(), &changed);
    }
}
