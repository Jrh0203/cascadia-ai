use std::fmt;

use cascadia_game::{
    GameConfig, GameState, MarketPrelude, PublicGameState, PublicSupply, RuleError, TurnAction,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{MenuComposer, RedeterminationSeed, RulesLegalMenu};

pub const PUBLIC_POLICY_OBSERVATION_SCHEMA_ID: &str =
    "cascadiav3.rival_public_policy_observation.v1";
pub const SEAT_LOCAL_MEMORY_SCHEMA_ID: &str = "cascadiav3.rival_seat_local_memory.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(try_from = "u8", into = "u8")]
pub struct SeatIndex(u8);

impl SeatIndex {
    pub fn new(index: u8) -> Result<Self, ObservationError> {
        if index < 4 {
            Ok(Self(index))
        } else {
            Err(ObservationError::SeatOutOfRange {
                seat: index,
                player_count: 4,
            })
        }
    }

    pub const fn get(self) -> u8 {
        self.0
    }
}

impl TryFrom<u8> for SeatIndex {
    type Error = ObservationError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl From<SeatIndex> for u8 {
    fn from(value: SeatIndex) -> Self {
        value.0
    }
}

/// Versioned, seat-local recurrent policy state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "SeatMemoryWire", into = "SeatMemoryWire")]
pub struct SeatLocalMemory {
    schema_id: String,
    payload: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SeatMemoryWire {
    schema_id: String,
    payload: Vec<u8>,
}

impl SeatLocalMemory {
    pub fn empty() -> Self {
        Self::new(Vec::new())
    }

    pub fn new(payload: Vec<u8>) -> Self {
        Self {
            schema_id: SEAT_LOCAL_MEMORY_SCHEMA_ID.to_owned(),
            payload,
        }
    }

    pub fn payload(&self) -> &[u8] {
        &self.payload
    }
}

impl From<SeatLocalMemory> for SeatMemoryWire {
    fn from(value: SeatLocalMemory) -> Self {
        Self {
            schema_id: value.schema_id,
            payload: value.payload,
        }
    }
}

impl TryFrom<SeatMemoryWire> for SeatLocalMemory {
    type Error = ObservationError;

    fn try_from(value: SeatMemoryWire) -> Result<Self, Self::Error> {
        if value.schema_id != SEAT_LOCAL_MEMORY_SCHEMA_ID {
            return Err(ObservationError::WrongMemorySchema(value.schema_id));
        }
        Ok(Self {
            schema_id: value.schema_id,
            payload: value.payload,
        })
    }
}

/// Independently owned memory for every seat. Replacement cannot alias or
/// mutate another seat's bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolicyMemoryBank {
    seats: Vec<SeatLocalMemory>,
}

impl PolicyMemoryBank {
    pub fn new(player_count: u8) -> Result<Self, ObservationError> {
        if !(1..=4).contains(&player_count) {
            return Err(ObservationError::InvalidPlayerCount(player_count));
        }
        Ok(Self {
            seats: vec![SeatLocalMemory::empty(); usize::from(player_count)],
        })
    }

    /// Restore the four independently owned memories captured at a frozen
    /// rollout root. Terminal continuations must never silently replace a
    /// mid-game policy state with four empty memories.
    pub fn from_four(memories: [SeatLocalMemory; 4]) -> Self {
        Self {
            seats: memories.into(),
        }
    }

    pub fn snapshot_four(&self) -> Result<[SeatLocalMemory; 4], ObservationError> {
        self.seats
            .clone()
            .try_into()
            .map_err(|values: Vec<SeatLocalMemory>| {
                ObservationError::InvalidPlayerCount(values.len() as u8)
            })
    }

    pub fn get(&self, seat: SeatIndex) -> Result<&SeatLocalMemory, ObservationError> {
        self.seats
            .get(usize::from(seat.get()))
            .ok_or(ObservationError::SeatOutOfRange {
                seat: seat.get(),
                player_count: self.seats.len() as u8,
            })
    }

    pub fn replace(
        &mut self,
        seat: SeatIndex,
        memory: SeatLocalMemory,
    ) -> Result<SeatLocalMemory, ObservationError> {
        let player_count = self.seats.len() as u8;
        let target = self.seats.get_mut(usize::from(seat.get())).ok_or(
            ObservationError::SeatOutOfRange {
                seat: seat.get(),
                player_count,
            },
        )?;
        Ok(std::mem::replace(target, memory))
    }

    pub fn len(&self) -> usize {
        self.seats.len()
    }

    pub fn is_empty(&self) -> bool {
        self.seats.is_empty()
    }
}

/// Complete public input available at one policy invocation.
///
/// `PublicSupply` is intentionally stored alongside `PublicGameState`; the
/// latter does not carry the public unseen-supply summary.
///
/// ```compile_fail
/// use cascadia_rival::{PrivateSimState, PublicPolicyObs};
/// fn accepts_only_public(_: &PublicPolicyObs) {}
/// fn illegal_leak(private: &PrivateSimState) {
///     accepts_only_public(private);
/// }
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "PublicObservationWire", into = "PublicObservationWire")]
pub struct PublicPolicyObs {
    schema_id: String,
    state: PublicGameState,
    supply: PublicSupply,
    seat: SeatIndex,
    memory: SeatLocalMemory,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PublicObservationWire {
    schema_id: String,
    state: PublicGameState,
    supply: PublicSupply,
    seat: SeatIndex,
    memory: SeatLocalMemory,
}

impl PublicPolicyObs {
    fn new(
        state: PublicGameState,
        supply: PublicSupply,
        seat: SeatIndex,
        memory: SeatLocalMemory,
    ) -> Result<Self, ObservationError> {
        validate_research_config(state.config())?;
        validate_seat(seat, state.config().player_count)?;
        if state.current_player() != usize::from(seat.get()) {
            return Err(ObservationError::SeatNotToMove {
                seat: seat.get(),
                current_player: state.current_player() as u8,
            });
        }
        Ok(Self {
            schema_id: PUBLIC_POLICY_OBSERVATION_SCHEMA_ID.to_owned(),
            state,
            supply,
            seat,
            memory,
        })
    }

    pub fn state(&self) -> &PublicGameState {
        &self.state
    }

    pub fn supply(&self) -> PublicSupply {
        self.supply
    }

    pub fn seat(&self) -> SeatIndex {
        self.seat
    }

    pub fn memory(&self) -> &SeatLocalMemory {
        &self.memory
    }

    pub fn canonical_bytes(&self) -> Vec<u8> {
        postcard::to_allocvec(self)
            .expect("serializing a validated in-memory public observation cannot fail")
    }

    pub fn canonical_hash(&self) -> blake3::Hash {
        blake3::hash(&self.canonical_bytes())
    }
}

impl From<PublicPolicyObs> for PublicObservationWire {
    fn from(value: PublicPolicyObs) -> Self {
        Self {
            schema_id: value.schema_id,
            state: value.state,
            supply: value.supply,
            seat: value.seat,
            memory: value.memory,
        }
    }
}

impl TryFrom<PublicObservationWire> for PublicPolicyObs {
    type Error = ObservationError;

    fn try_from(value: PublicObservationWire) -> Result<Self, Self::Error> {
        if value.schema_id != PUBLIC_POLICY_OBSERVATION_SCHEMA_ID {
            return Err(ObservationError::WrongObservationSchema(value.schema_id));
        }
        Self::new(value.state, value.supply, value.seat, value.memory)
    }
}

/// Simulator-private canonical state. It has no accessor returning the inner
/// `GameState`; orchestration inside this crate is the only trusted boundary.
pub struct PrivateSimState {
    game: GameState,
}

impl PrivateSimState {
    pub fn new(game: GameState) -> Result<Self, ObservationError> {
        game.validate()
            .map_err(ObservationError::InvalidGameState)?;
        validate_research_config(game.config())?;
        Ok(Self { game })
    }

    pub fn public_observation(
        &self,
        seat: SeatIndex,
        memory: SeatLocalMemory,
    ) -> Result<PublicPolicyObs, ObservationError> {
        PublicPolicyObs::new(
            self.game().public_state(),
            self.game().public_supply(),
            seat,
            memory,
        )
    }

    pub(crate) fn game(&self) -> &GameState {
        &self.game
    }
}

impl fmt::Debug for PrivateSimState {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("PrivateSimState(<opaque>)")
    }
}

/// Opaque capability for sampling rules-executable worlds compatible with the
/// private physical state. Construction immediately canonical-redetermines
/// the merged hidden tile partitions and wildlife order, so original hidden
/// ordering is not retained as a policy-observable artifact.
pub struct HonestWorldSampler {
    private_template: GameState,
    base_seed: RedeterminationSeed,
}

impl HonestWorldSampler {
    pub(crate) fn new(private: &PrivateSimState, base_seed: RedeterminationSeed) -> Self {
        Self {
            private_template: private.game().clone(),
            base_seed,
        }
    }

    /// Deterministically sample one opaque world from the root-local
    /// redetermination stream. The policy chooses only an ordinal; seed bytes,
    /// the physical world, and experimental coordinates remain inaccessible.
    pub fn sample(&self, sample_index: u32) -> PolicyWorld {
        let mut world = self.private_template.clone();
        world.redeterminize_hidden(self.base_seed.for_world_sample(sample_index).game_seed());
        PolicyWorld { game: world }
    }

    pub fn initial_world(&self) -> PolicyWorld {
        self.sample(0)
    }
}

impl fmt::Debug for HonestWorldSampler {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("HonestWorldSampler(<opaque>)")
    }
}

/// One opaque, redetermined, canonical-rules world for policy search.
#[derive(Clone)]
pub struct PolicyWorld {
    game: GameState,
}

impl PolicyWorld {
    pub fn public_observation(
        &self,
        seat: SeatIndex,
        memory: SeatLocalMemory,
    ) -> Result<PublicPolicyObs, ObservationError> {
        PublicPolicyObs::new(
            self.game.public_state(),
            self.game.public_supply(),
            seat,
            memory,
        )
    }

    pub fn rules_menu(
        &self,
        accumulated_prelude: &MarketPrelude,
    ) -> Result<RulesLegalMenu, crate::MenuError> {
        MenuComposer::draft_root(&self.game, accumulated_prelude)
    }

    /// Public observation after exactly the accumulated, already chosen
    /// prelude decisions have resolved. The underlying source world is not
    /// replaced or advanced; this prevents a caller from accidentally
    /// applying the same accumulated prelude twice when the next paid-wipe
    /// decision is recomposed.
    pub fn draft_observation(
        &self,
        accumulated_prelude: &MarketPrelude,
        seat: SeatIndex,
        memory: SeatLocalMemory,
    ) -> Result<PublicPolicyObs, ObservationError> {
        let staged = self
            .game
            .preview_market_prelude(accumulated_prelude)
            .map_err(ObservationError::Rules)?;
        PublicPolicyObs::new(staged.public_state(), staged.public_supply(), seat, memory)
    }

    pub fn transition(&self, action: &TurnAction) -> Result<Self, RuleError> {
        Ok(Self {
            game: self.game().transition(action)?,
        })
    }

    pub fn apply(&mut self, action: &TurnAction) -> Result<(), RuleError> {
        self.game_mut().apply(action)
    }

    pub fn is_game_over(&self) -> bool {
        self.game.is_game_over()
    }

    pub fn current_player(&self) -> SeatIndex {
        SeatIndex(self.game.current_player() as u8)
    }

    pub(crate) fn game(&self) -> &GameState {
        &self.game
    }

    pub(crate) fn game_mut(&mut self) -> &mut GameState {
        &mut self.game
    }
}

impl fmt::Debug for PolicyWorld {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("PolicyWorld(<opaque>)")
    }
}

#[derive(Debug, Error)]
pub enum ObservationError {
    #[error("invalid player count {0}; expected 1 through 4")]
    InvalidPlayerCount(u8),
    #[error("seat {seat} is outside player count {player_count}")]
    SeatOutOfRange { seat: u8, player_count: u8 },
    #[error("seat {seat} is not the current player {current_player}")]
    SeatNotToMove { seat: u8, current_player: u8 },
    #[error("wrong seat-memory schema: {0}")]
    WrongMemorySchema(String),
    #[error("wrong public-observation schema: {0}")]
    WrongObservationSchema(String),
    #[error("invalid canonical simulator state: {0}")]
    InvalidGameState(&'static str),
    #[error("Rival requires the canonical four-player AAAAA no-habitat-bonus configuration")]
    WrongGameConfig,
    #[error(transparent)]
    Rules(#[from] RuleError),
}

fn validate_seat(seat: SeatIndex, player_count: u8) -> Result<(), ObservationError> {
    if seat.get() < player_count {
        Ok(())
    } else {
        Err(ObservationError::SeatOutOfRange {
            seat: seat.get(),
            player_count,
        })
    }
}

fn validate_research_config(config: GameConfig) -> Result<(), ObservationError> {
    let expected = GameConfig::research_aaaaa(4)
        .expect("the canonical four-player research configuration is valid");
    if config == expected {
        Ok(())
    } else {
        Err(ObservationError::WrongGameConfig)
    }
}

#[cfg(test)]
mod tests {
    use cascadia_game::{GameConfig, GameSeed};
    use serde_json::Value;

    use crate::{
        EvaluationBranch, Fidelity, InnerRngCoordinate, PublicRootId, RivalSeed, RngFactory,
        RootKind, Sha256Digest,
    };

    use super::*;

    fn private(seed: u64) -> PrivateSimState {
        PrivateSimState::new(
            GameState::new(
                GameConfig::research_aaaaa(4).unwrap(),
                GameSeed::from_u64(seed),
            )
            .unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn observation_includes_supply_and_only_selected_seat_memory() {
        let private = private(7);
        let observation = private
            .public_observation(SeatIndex::new(0).unwrap(), SeatLocalMemory::new(vec![4, 2]))
            .unwrap();
        assert_eq!(observation.seat().get(), 0);
        assert_eq!(observation.memory().payload(), &[4, 2]);
        assert_eq!(observation.supply(), private.game.public_supply());

        let json = serde_json::to_value(&observation).unwrap();
        let text = serde_json::to_string(&json).unwrap();
        for forbidden in [
            "tile_stack",
            "excluded_tiles",
            "seed",
            "scenario",
            "event_priority",
        ] {
            assert!(!text.contains(forbidden), "public input leaked {forbidden}");
        }
    }

    #[test]
    fn memory_updates_are_strictly_seat_local() {
        let mut bank = PolicyMemoryBank::new(4).unwrap();
        let before = bank.clone();
        bank.replace(
            SeatIndex::new(1).unwrap(),
            SeatLocalMemory::new(vec![9, 8, 7]),
        )
        .unwrap();
        for seat in [0, 2, 3] {
            let seat = SeatIndex::new(seat).unwrap();
            assert_eq!(bank.get(seat).unwrap(), before.get(seat).unwrap());
        }
        assert_ne!(
            bank.get(SeatIndex::new(1).unwrap()).unwrap(),
            before.get(SeatIndex::new(1).unwrap()).unwrap()
        );
    }

    #[test]
    fn hidden_order_and_partition_permutations_have_identical_public_input_and_samples() {
        let original = private(41);
        let mut value = serde_json::to_value(original.game()).unwrap();
        permute_hidden_json(&mut value);
        let permuted_game: GameState = serde_json::from_value(value).unwrap();
        permuted_game.validate().unwrap();
        let permuted = PrivateSimState::new(permuted_game).unwrap();

        let seat = SeatIndex::new(0).unwrap();
        let left_obs = original
            .public_observation(seat, SeatLocalMemory::empty())
            .unwrap();
        let right_obs = permuted
            .public_observation(seat, SeatLocalMemory::empty())
            .unwrap();
        assert_eq!(left_obs, right_obs);

        let root = PublicRootId::new(&left_obs, RootKind::DraftPolicyRoot);
        let rng = RngFactory::new(RivalSeed::from_u64(90));
        let panel = Sha256Digest::of_bytes(b"observation-hidden-order-panel");
        let seed = rng.redetermination(InnerRngCoordinate::new(
            &root,
            &panel,
            EvaluationBranch::Incumbent,
            Fidelity::High,
            seat,
            0,
            0,
        ));
        let left = HonestWorldSampler::new(&original, seed).initial_world();
        let right = HonestWorldSampler::new(&permuted, seed).initial_world();
        assert_eq!(left.game().public_state(), original.game().public_state());
        assert_eq!(left.game().public_supply(), original.game().public_supply());
        assert_eq!(
            left.game().canonical_bytes(),
            right.game().canonical_bytes()
        );
    }

    #[test]
    fn legitimate_public_reveal_changes_the_observation() {
        for seed in 0..1000 {
            let original = private(seed);
            let choices = original.game.free_three_of_a_kind_choices().unwrap();
            if let Some(accept) = choices
                .into_iter()
                .find(|prelude| prelude.replace_three_of_a_kind)
            {
                let revealed = original.game.preview_market_prelude(&accept).unwrap();
                assert_ne!(original.game.public_state(), revealed.public_state());
                assert_ne!(original.game.public_supply(), revealed.public_supply());
                return;
            }
        }
        panic!("fixture search did not find a legal public replacement reveal");
    }

    #[test]
    fn observation_and_memory_deserialization_fail_closed() {
        let observation = private(1)
            .public_observation(SeatIndex::new(0).unwrap(), SeatLocalMemory::empty())
            .unwrap();
        let mut value = serde_json::to_value(observation).unwrap();
        value["scenario_id"] = Value::String("forbidden".to_owned());
        assert!(serde_json::from_value::<PublicPolicyObs>(value).is_err());

        let mut memory = serde_json::to_value(SeatLocalMemory::empty()).unwrap();
        memory["schema_id"] = Value::String("unknown".to_owned());
        assert!(serde_json::from_value::<SeatLocalMemory>(memory).is_err());

        assert_eq!(serde_json::from_str::<SeatIndex>("3").unwrap().get(), 3);
        assert!(serde_json::from_str::<SeatIndex>("4").is_err());
    }

    #[test]
    fn policy_observation_rejects_a_nonacting_seat() {
        let private = private(2);
        assert!(matches!(
            private.public_observation(SeatIndex::new(1).unwrap(), SeatLocalMemory::empty()),
            Err(ObservationError::SeatNotToMove {
                seat: 1,
                current_player: 0
            })
        ));
    }

    #[test]
    fn private_boundary_rejects_deserialized_invalid_state() {
        let valid = private(12);
        let mut value = serde_json::to_value(valid.game()).unwrap();
        value["current_player"] = Value::from(9);
        let invalid: GameState = serde_json::from_value(value).unwrap();
        assert!(PrivateSimState::new(invalid).is_err());
    }

    #[test]
    fn private_boundary_rejects_a_different_game_configuration() {
        let game = GameState::new(
            GameConfig::research_aaaaa(2).unwrap(),
            GameSeed::from_u64(12),
        )
        .unwrap();
        assert!(matches!(
            PrivateSimState::new(game),
            Err(ObservationError::WrongGameConfig)
        ));
    }

    #[test]
    fn draft_observation_stages_prelude_without_advancing_source_world() {
        let private = private(13);
        let seat = SeatIndex::new(0).unwrap();
        let root_obs = private
            .public_observation(seat, SeatLocalMemory::empty())
            .unwrap();
        let root = PublicRootId::new(&root_obs, RootKind::DraftPolicyRoot);
        let factory = RngFactory::new(RivalSeed::from_u64(14));
        let panel = Sha256Digest::of_bytes(b"observation-draft-panel");
        let redetermination = factory.redetermination(InnerRngCoordinate::new(
            &root,
            &panel,
            EvaluationBranch::Incumbent,
            Fidelity::High,
            seat,
            0,
            0,
        ));
        let world = HonestWorldSampler::new(&private, redetermination).initial_world();
        let before = world.game().canonical_bytes();
        let observation = world
            .draft_observation(&MarketPrelude::default(), seat, SeatLocalMemory::empty())
            .unwrap();
        assert_eq!(observation.state(), &world.game().public_state());
        assert_eq!(before, world.game().canonical_bytes());
    }

    #[test]
    fn opaque_world_apply_and_transition_are_canonical_and_identical() {
        let private = private(15);
        let seat = SeatIndex::new(0).unwrap();
        let observation = private
            .public_observation(seat, SeatLocalMemory::empty())
            .unwrap();
        let root = PublicRootId::new(&observation, RootKind::DraftPolicyRoot);
        let factory = RngFactory::new(RivalSeed::from_u64(16));
        let panel = Sha256Digest::of_bytes(b"observation-transition-panel");
        let seed = factory.redetermination(InnerRngCoordinate::new(
            &root,
            &panel,
            EvaluationBranch::Incumbent,
            Fidelity::High,
            seat,
            0,
            0,
        ));
        let world = HonestWorldSampler::new(&private, seed).initial_world();
        let menu = world.rules_menu(&MarketPrelude::default()).unwrap();
        let action = menu
            .draft_action(menu.first_draft_index().unwrap())
            .unwrap()
            .clone();
        let transitioned = world.transition(&action).unwrap();
        let mut applied = world;
        applied.apply(&action).unwrap();
        assert_eq!(
            transitioned.game().canonical_bytes(),
            applied.game().canonical_bytes()
        );
    }

    #[test]
    fn honest_world_sampler_exposes_deterministic_separated_ordinals_only() {
        let private = private(17);
        let seat = SeatIndex::new(0).unwrap();
        let observation = private
            .public_observation(seat, SeatLocalMemory::empty())
            .unwrap();
        let root = PublicRootId::new(&observation, RootKind::DraftPolicyRoot);
        let factory = RngFactory::new(RivalSeed::from_u64(18));
        let panel = Sha256Digest::of_bytes(b"multi-world-policy-panel");
        let base = factory.redetermination(InnerRngCoordinate::new(
            &root,
            &panel,
            EvaluationBranch::Incumbent,
            Fidelity::High,
            seat,
            0,
            0,
        ));
        let sampler = HonestWorldSampler::new(&private, base);
        let zero = sampler.sample(0);
        let one = sampler.sample(1);
        let one_again = sampler.sample(1);
        assert_eq!(
            zero.game().canonical_bytes(),
            sampler.initial_world().game().canonical_bytes()
        );
        assert_eq!(
            one.game().canonical_bytes(),
            one_again.game().canonical_bytes()
        );
        assert_ne!(zero.game().canonical_bytes(), one.game().canonical_bytes());
        for world in [zero, one] {
            assert_eq!(world.game().public_state(), private.game().public_state());
            assert_eq!(world.game().public_supply(), private.game().public_supply());
        }
    }

    fn permute_hidden_json(value: &mut Value) {
        let object = value.as_object_mut().unwrap();
        let mut stack = object
            .remove("tile_stack")
            .unwrap()
            .as_array()
            .unwrap()
            .clone();
        let mut excluded = object
            .remove("excluded_tiles")
            .unwrap()
            .as_array()
            .unwrap()
            .clone();
        let stack_len = stack.len();
        stack.append(&mut excluded);
        stack.reverse();
        let new_excluded = stack.split_off(stack_len);
        object.insert("tile_stack".to_owned(), Value::Array(stack));
        object.insert("excluded_tiles".to_owned(), Value::Array(new_excluded));
        object
            .get_mut("wildlife_bag")
            .unwrap()
            .as_array_mut()
            .unwrap()
            .reverse();
    }
}
