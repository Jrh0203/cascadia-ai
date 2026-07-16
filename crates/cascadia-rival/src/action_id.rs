use std::fmt;

use cascadia_game::{DraftChoice, TurnAction};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    IncumbentCandidateMenu, IncumbentMenuHash, PublicPolicyObs, ResearchRulesetIdentity,
    RulesLegalMenu, RulesMenuHash, Sha256Digest,
};

pub const LEGACY_ACTION_ID_VERSION: &str = "legacy_turn_action_json_sha256.v0";
pub const ACTION_CONTENT_ID_VERSION: &str = "cascadiav3.rival_action_content.v1";
pub const PUBLIC_ROOT_ID_VERSION: &str = "cascadiav3.rival_public_root.v1";
pub const ROOT_ACTION_OCCURRENCE_ID_VERSION: &str = "cascadiav3.rival_root_action_occurrence.v1";
pub const CANDIDATE_ACTION_OCCURRENCE_ID_VERSION: &str =
    "cascadiav3.rival_candidate_action_occurrence.v1";
pub const ROOT_CHRONOLOGY_VERSION: &str = "cascadiav3.rival_root_chronology.v1";

/// Byte-for-byte preservation of the exporter's historical
/// `sha256(serde_json::to_vec(TurnAction))` identity.
#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct LegacyActionIdV0(String);

impl LegacyActionIdV0 {
    pub fn new(action: &TurnAction) -> Result<Self, ActionIdError> {
        let bytes = serde_json::to_vec(action)?;
        Ok(Self(Sha256Digest::of_bytes(&bytes).to_string()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for LegacyActionIdV0 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("LegacyActionIdV0")
            .field(&self.0)
            .finish()
    }
}

impl From<LegacyActionIdV0> for String {
    fn from(value: LegacyActionIdV0) -> Self {
        value.0
    }
}

impl TryFrom<String> for LegacyActionIdV0 {
    type Error = ActionIdError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        value
            .parse::<Sha256Digest>()
            .map_err(|_| ActionIdError::InvalidId(LEGACY_ACTION_ID_VERSION))?;
        Ok(Self(value))
    }
}

macro_rules! namespaced_id {
    ($name:ident, $namespace:expr) => {
        #[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
        #[serde(try_from = "String", into = "String")]
        pub struct $name(String);

        impl $name {
            fn from_digest(digest: Sha256Digest) -> Self {
                Self(format!("{}:{}", $namespace, digest))
            }

            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl fmt::Debug for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter
                    .debug_tuple(stringify!($name))
                    .field(&self.0)
                    .finish()
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str(&self.0)
            }
        }

        impl From<$name> for String {
            fn from(value: $name) -> Self {
                value.0
            }
        }

        impl TryFrom<String> for $name {
            type Error = ActionIdError;

            fn try_from(value: String) -> Result<Self, Self::Error> {
                let prefix = format!("{}:", $namespace);
                let digest = value
                    .strip_prefix(&prefix)
                    .ok_or(ActionIdError::InvalidId($namespace))?;
                digest
                    .parse::<Sha256Digest>()
                    .map_err(|_| ActionIdError::InvalidId($namespace))?;
                Ok(Self(value))
            }
        }
    };
}

namespaced_id!(ActionContentId, ACTION_CONTENT_ID_VERSION);
namespaced_id!(PublicRootId, PUBLIC_ROOT_ID_VERSION);
namespaced_id!(RootActionOccurrenceId, ROOT_ACTION_OCCURRENCE_ID_VERSION);
namespaced_id!(
    CandidateActionOccurrenceId,
    CANDIDATE_ACTION_OCCURRENCE_ID_VERSION
);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RootKind {
    PreludePolicyRoot,
    DraftPolicyRoot,
}

impl ActionContentId {
    pub fn new(action: &TurnAction, ruleset: &ResearchRulesetIdentity) -> Self {
        let payload = ActionContentPayload::new(action, ruleset.clone());
        Self::from_digest(hash_json(&payload))
    }

    pub fn canonical(action: &TurnAction) -> Self {
        Self::new(action, &ResearchRulesetIdentity::canonical())
    }
}

impl PublicRootId {
    pub fn new(observation: &PublicPolicyObs, kind: RootKind) -> Self {
        let payload = PublicRootPayload {
            schema_id: PUBLIC_ROOT_ID_VERSION,
            ruleset: ResearchRulesetIdentity::canonical(),
            acting_seat: observation.seat().get(),
            root_kind: kind,
            chronology_version: ROOT_CHRONOLOGY_VERSION,
            observation,
        };
        Self::from_digest(hash_json(&payload))
    }
}

impl RootActionOccurrenceId {
    pub fn new(
        root: &PublicRootId,
        menu: &RulesLegalMenu,
        action_index: usize,
    ) -> Result<Self, ActionIdError> {
        let action = menu
            .draft_action(action_index)
            .ok_or(ActionIdError::NotDraftAction(action_index))?;
        let payload = RootActionOccurrencePayload {
            schema_id: ROOT_ACTION_OCCURRENCE_ID_VERSION,
            public_root_id: root,
            ordered_menu_hash: menu.hash(),
            action_content_id: ActionContentId::canonical(action),
            action_index: u32::try_from(action_index)
                .map_err(|_| ActionIdError::ActionIndexTooLarge(action_index))?,
        };
        Ok(Self::from_digest(hash_json(&payload)))
    }
}

impl CandidateActionOccurrenceId {
    pub fn new(
        root: &PublicRootId,
        menu: &IncumbentCandidateMenu,
        candidate_index: usize,
    ) -> Result<Self, ActionIdError> {
        let action = menu
            .decisions()
            .get(candidate_index)
            .and_then(|decision| match decision {
                crate::RulesDecision::Draft(action) => Some(action),
                crate::RulesDecision::Prelude(_) | crate::RulesDecision::PaidWipe(_) => None,
            })
            .ok_or(ActionIdError::NotDraftAction(candidate_index))?;
        let rules_index = *menu
            .rules_indices()
            .get(candidate_index)
            .ok_or(ActionIdError::NotDraftAction(candidate_index))?;
        let payload = CandidateActionOccurrencePayload {
            schema_id: CANDIDATE_ACTION_OCCURRENCE_ID_VERSION,
            public_root_id: root,
            ordered_candidate_menu_hash: menu.hash(),
            action_content_id: ActionContentId::canonical(action),
            candidate_index: u32::try_from(candidate_index)
                .map_err(|_| ActionIdError::ActionIndexTooLarge(candidate_index))?,
            rules_index: u32::try_from(rules_index)
                .map_err(|_| ActionIdError::ActionIndexTooLarge(rules_index))?,
        };
        Ok(Self::from_digest(hash_json(&payload)))
    }
}

#[derive(Serialize)]
struct ActionContentPayload {
    schema_id: &'static str,
    ruleset: ResearchRulesetIdentity,
    replace_three_of_a_kind: bool,
    wildlife_wipes: Vec<Vec<u8>>,
    draft: CanonicalDraft,
    tile_q: i8,
    tile_r: i8,
    tile_rotation: u8,
    wildlife: Option<CanonicalCoord>,
}

impl ActionContentPayload {
    fn new(action: &TurnAction, ruleset: ResearchRulesetIdentity) -> Self {
        Self {
            schema_id: ACTION_CONTENT_ID_VERSION,
            ruleset,
            replace_three_of_a_kind: action.replace_three_of_a_kind,
            wildlife_wipes: action
                .wildlife_wipes
                .iter()
                .map(|wipe| wipe.slots.iter().map(|slot| slot.index() as u8).collect())
                .collect(),
            draft: match action.draft {
                DraftChoice::Paired { slot } => CanonicalDraft::Paired {
                    slot: slot.index() as u8,
                },
                DraftChoice::Independent {
                    tile_slot,
                    wildlife_slot,
                } => CanonicalDraft::Independent {
                    tile_slot: tile_slot.index() as u8,
                    wildlife_slot: wildlife_slot.index() as u8,
                },
            },
            tile_q: action.tile.coord.q,
            tile_r: action.tile.coord.r,
            tile_rotation: action.tile.rotation.get(),
            wildlife: action.wildlife.map(|coord| CanonicalCoord {
                q: coord.q,
                r: coord.r,
            }),
        }
    }
}

#[derive(Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum CanonicalDraft {
    Paired { slot: u8 },
    Independent { tile_slot: u8, wildlife_slot: u8 },
}

#[derive(Serialize)]
struct CanonicalCoord {
    q: i8,
    r: i8,
}

#[derive(Serialize)]
struct PublicRootPayload<'a> {
    schema_id: &'static str,
    ruleset: ResearchRulesetIdentity,
    acting_seat: u8,
    root_kind: RootKind,
    chronology_version: &'static str,
    observation: &'a PublicPolicyObs,
}

#[derive(Serialize)]
struct RootActionOccurrencePayload<'a> {
    schema_id: &'static str,
    public_root_id: &'a PublicRootId,
    ordered_menu_hash: RulesMenuHash,
    action_content_id: ActionContentId,
    action_index: u32,
}

#[derive(Serialize)]
struct CandidateActionOccurrencePayload<'a> {
    schema_id: &'static str,
    public_root_id: &'a PublicRootId,
    ordered_candidate_menu_hash: IncumbentMenuHash,
    action_content_id: ActionContentId,
    candidate_index: u32,
    rules_index: u32,
}

fn hash_json<T: Serialize>(value: &T) -> Sha256Digest {
    let bytes = serde_json::to_vec(value)
        .expect("serializing a fixed canonical identity payload cannot fail");
    Sha256Digest::of_bytes(&bytes)
}

#[derive(Debug, Error)]
pub enum ActionIdError {
    #[error("failed to serialize legacy action identity: {0}")]
    Json(#[from] serde_json::Error),
    #[error("invalid {0} wire identity")]
    InvalidId(&'static str),
    #[error("menu item {0} is not a complete draft action")]
    NotDraftAction(usize),
    #[error("menu action index {0} cannot be represented in the v1 occurrence schema")]
    ActionIndexTooLarge(usize),
}

#[cfg(test)]
mod tests {
    use cascadia_game::{HexCoord, MarketSlot, Rotation, TilePlacement, WildlifeWipe};

    use cascadia_game::{GameConfig, GameSeed, GameState};

    use crate::{MenuComposer, PrivateSimState, SeatIndex, SeatLocalMemory};

    use super::*;

    fn action() -> TurnAction {
        TurnAction {
            replace_three_of_a_kind: false,
            wildlife_wipes: vec![WildlifeWipe {
                slots: vec![MarketSlot::ZERO, MarketSlot::TWO],
            }],
            draft: DraftChoice::Independent {
                tile_slot: MarketSlot::ONE,
                wildlife_slot: MarketSlot::THREE,
            },
            tile: TilePlacement {
                coord: HexCoord::new(-2, 3),
                rotation: Rotation::FOUR,
            },
            wildlife: Some(HexCoord::new(-1, 2)),
        }
    }

    #[test]
    fn legacy_id_is_exact_exporter_algorithm() {
        let action = action();
        let expected = Sha256Digest::of_bytes(&serde_json::to_vec(&action).unwrap()).to_string();
        assert_eq!(LegacyActionIdV0::new(&action).unwrap().as_str(), expected);
        assert_eq!(
            expected,
            "sha256:b030f4e5a35e77f6a47996e1588242a001b17714e54b0fd6ed4942c35d992db7"
        );
    }

    #[test]
    fn additive_ids_are_explicitly_namespaced_and_round_trip() {
        let action_id = ActionContentId::canonical(&action());
        assert!(action_id.as_str().starts_with(ACTION_CONTENT_ID_VERSION));
        assert_eq!(
            action_id.as_str(),
            "cascadiav3.rival_action_content.v1:sha256:77752331bbc2c63cac3ce6a967ae66094e5dd713c00eaa8ea5c28292bdfd6fa8"
        );
        let bytes = serde_json::to_vec(&action_id).unwrap();
        assert_eq!(
            serde_json::from_slice::<ActionContentId>(&bytes).unwrap(),
            action_id
        );
        assert!(serde_json::from_slice::<LegacyActionIdV0>(&bytes).is_err());
    }

    #[test]
    fn content_identity_is_root_independent_but_occurrence_is_not() {
        let game = GameState::new(
            GameConfig::research_aaaaa(4).unwrap(),
            GameSeed::from_u64(6),
        )
        .unwrap();
        let private = PrivateSimState::new(game.clone()).unwrap();
        let obs = private
            .public_observation(SeatIndex::new(0).unwrap(), SeatLocalMemory::empty())
            .unwrap();
        let menu = MenuComposer::draft_root(&game, &Default::default()).unwrap();
        let root_a = PublicRootId::new(&obs, RootKind::DraftPolicyRoot);
        let mut memory = SeatLocalMemory::new(vec![1]);
        let obs_b = private
            .public_observation(SeatIndex::new(0).unwrap(), memory.clone())
            .unwrap();
        memory = SeatLocalMemory::new(vec![2]);
        assert_ne!(memory.payload(), obs_b.memory().payload());
        let root_b = PublicRootId::new(&obs_b, RootKind::DraftPolicyRoot);
        assert_ne!(root_a, root_b);

        let index = menu.first_draft_index().unwrap();
        let action = menu.draft_action(index).unwrap();
        let content_a = ActionContentId::canonical(action);
        let content_b = ActionContentId::canonical(action);
        assert_eq!(content_a, content_b);
        assert_ne!(
            RootActionOccurrenceId::new(&root_a, &menu, index).unwrap(),
            RootActionOccurrenceId::new(&root_b, &menu, index).unwrap()
        );

        let candidate = IncumbentCandidateMenu::from_rules_indices(&menu, [index]).unwrap();
        let candidate_occurrence =
            CandidateActionOccurrenceId::new(&root_a, &candidate, 0).unwrap();
        assert!(
            candidate_occurrence
                .as_str()
                .starts_with(CANDIDATE_ACTION_OCCURRENCE_ID_VERSION)
        );
        let bytes = serde_json::to_vec(&candidate_occurrence).unwrap();
        assert!(serde_json::from_slice::<CandidateActionOccurrenceId>(&bytes).is_ok());
        assert!(serde_json::from_slice::<RootActionOccurrenceId>(&bytes).is_err());
    }

    #[test]
    fn serialization_perturbations_change_new_content_id() {
        let original = action();
        let original_id = ActionContentId::canonical(&original);

        let mut changed = original.clone();
        changed.tile.rotation = Rotation::FIVE;
        assert_ne!(ActionContentId::canonical(&changed), original_id);

        let mut changed = original;
        changed.wildlife_wipes[0].slots.reverse();
        assert_ne!(ActionContentId::canonical(&changed), original_id);
    }
}
