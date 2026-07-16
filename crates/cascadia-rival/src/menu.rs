use std::{collections::HashSet, fmt};

use cascadia_game::{GameConfig, GameState, MarketPrelude, RuleError, TurnAction, WildlifeWipe};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{ActionContentId, RootKind, Sha256Digest};

pub const RULES_MENU_HASH_VERSION: &str = "cascadiav3.rival_rules_menu.v1";
pub const INCUMBENT_MENU_HASH_VERSION: &str = "cascadiav3.rival_incumbent_menu.v1";
/// Backward source-level name for the canonical rules-menu identity contract.
pub const MENU_HASH_VERSION: &str = RULES_MENU_HASH_VERSION;

/// One choice at a public policy root.
///
/// Paid wipes are deliberately one-step decisions. After a wipe's public
/// replacement is revealed, trusted orchestration extends the accumulated
/// prelude and recomposes a fresh DraftPolicyRoot menu. Future replacement
/// outcomes are never flattened into the current action space.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RulesDecision {
    Prelude(MarketPrelude),
    PaidWipe(WildlifeWipe),
    Draft(TurnAction),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RulesLegalMenu {
    root_kind: RootKind,
    decisions: Vec<RulesDecision>,
    hash: RulesMenuHash,
}

impl RulesLegalMenu {
    fn new(root_kind: RootKind, decisions: Vec<RulesDecision>) -> Result<Self, MenuError> {
        if decisions.is_empty() {
            return Err(MenuError::EmptyMenu);
        }
        let hash = RulesMenuHash::for_decisions(&decisions);
        Ok(Self {
            root_kind,
            decisions,
            hash,
        })
    }

    pub fn root_kind(&self) -> RootKind {
        self.root_kind
    }

    pub fn decisions(&self) -> &[RulesDecision] {
        &self.decisions
    }

    pub fn decision(&self, index: usize) -> Option<&RulesDecision> {
        self.decisions.get(index)
    }

    pub fn draft_action(&self, index: usize) -> Option<&TurnAction> {
        match self.decision(index)? {
            RulesDecision::Draft(action) => Some(action),
            RulesDecision::Prelude(_) | RulesDecision::PaidWipe(_) => None,
        }
    }

    pub fn first_draft_index(&self) -> Option<usize> {
        self.decisions
            .iter()
            .position(|decision| matches!(decision, RulesDecision::Draft(_)))
    }

    pub fn len(&self) -> usize {
        self.decisions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.decisions.is_empty()
    }

    pub fn hash(&self) -> RulesMenuHash {
        self.hash.clone()
    }
}

/// Policy-specific ordered subset of a [`RulesLegalMenu`]. It is intentionally
/// a distinct type: this is the frozen incumbent candidate generator's menu,
/// not canonical legality.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IncumbentCandidateMenu {
    root_kind: RootKind,
    decisions: Vec<RulesDecision>,
    rules_indices: Vec<usize>,
    hash: IncumbentMenuHash,
}

impl IncumbentCandidateMenu {
    pub fn from_rules_indices(
        rules: &RulesLegalMenu,
        ordered_indices: impl IntoIterator<Item = usize>,
    ) -> Result<Self, MenuError> {
        let rules_indices: Vec<_> = ordered_indices.into_iter().collect();
        if rules_indices.is_empty() {
            return Err(MenuError::EmptyMenu);
        }
        let mut seen = HashSet::with_capacity(rules_indices.len());
        let mut decisions = Vec::with_capacity(rules_indices.len());
        for &index in &rules_indices {
            if !seen.insert(index) {
                return Err(MenuError::DuplicateCandidateIndex(index));
            }
            let decision = rules
                .decision(index)
                .ok_or(MenuError::CandidateIndexOutOfRange(index))?;
            if !matches!(decision, RulesDecision::Draft(_)) {
                return Err(MenuError::NonDraftCandidateIndex(index));
            }
            decisions.push(decision.clone());
        }
        let hash = IncumbentMenuHash::for_candidate(rules.hash(), &rules_indices, &decisions);
        Ok(Self {
            root_kind: rules.root_kind,
            decisions,
            rules_indices,
            hash,
        })
    }

    pub fn root_kind(&self) -> RootKind {
        self.root_kind
    }

    pub fn decisions(&self) -> &[RulesDecision] {
        &self.decisions
    }

    pub fn draft_action(&self, index: usize) -> Option<&TurnAction> {
        match self.decisions.get(index)? {
            RulesDecision::Draft(action) => Some(action),
            RulesDecision::Prelude(_) | RulesDecision::PaidWipe(_) => None,
        }
    }

    pub fn len(&self) -> usize {
        self.decisions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.decisions.is_empty()
    }

    pub fn rules_indices(&self) -> &[usize] {
        &self.rules_indices
    }

    pub fn hash(&self) -> IncumbentMenuHash {
        self.hash.clone()
    }
}

/// Canonical root-specific composer. It delegates all legality, chronology,
/// chance resolution, board transitions, and validation to `cascadia-game`.
pub struct MenuComposer;

impl MenuComposer {
    pub fn prelude_root(source: &GameState) -> Result<RulesLegalMenu, MenuError> {
        validate_source_config(source)?;
        let choices = source.free_three_of_a_kind_choices()?;
        for choice in &choices {
            source.preview_market_prelude(choice).map_err(|source| {
                MenuError::CanonicalPreludeMismatch {
                    prelude: choice.clone(),
                    source,
                }
            })?;
        }
        RulesLegalMenu::new(
            RootKind::PreludePolicyRoot,
            choices.into_iter().map(RulesDecision::Prelude).collect(),
        )
    }

    /// Compose one post-reveal draft decision loop.
    ///
    /// Complete drafts are ordered exactly as `legal_turn_actions`. Current
    /// one-step paid wipes follow them in canonical `legal_wildlife_wipes`
    /// order. No future wipe is enumerated before its public reveal.
    pub fn draft_root(
        source: &GameState,
        accumulated_prelude: &MarketPrelude,
    ) -> Result<RulesLegalMenu, MenuError> {
        validate_source_config(source)?;
        let staged = source.preview_market_prelude(accumulated_prelude)?;
        let actions = source.legal_turn_actions(accumulated_prelude)?;
        let mut decisions = Vec::with_capacity(actions.len() + 15);
        for (index, action) in actions.into_iter().enumerate() {
            if action.prelude() != *accumulated_prelude {
                return Err(MenuError::CanonicalActionPreludeMismatch(index));
            }
            source
                .transition(&action)
                .map_err(|source| MenuError::CanonicalActionTransitionMismatch { index, source })?;
            decisions.push(RulesDecision::Draft(action));
        }

        for wipe in staged.legal_wildlife_wipes() {
            let extended = Self::extend_paid_wipe(source, accumulated_prelude, &wipe)?;
            // The preview is the public reveal boundary and also the canonical
            // proof that the one-step extension is executable.
            source.preview_market_prelude(&extended)?;
            decisions.push(RulesDecision::PaidWipe(wipe));
        }
        RulesLegalMenu::new(RootKind::DraftPolicyRoot, decisions)
    }

    pub fn extend_paid_wipe(
        source: &GameState,
        accumulated_prelude: &MarketPrelude,
        wipe: &WildlifeWipe,
    ) -> Result<MarketPrelude, MenuError> {
        validate_source_config(source)?;
        let staged = source.preview_market_prelude(accumulated_prelude)?;
        if !staged.legal_wildlife_wipes().contains(wipe) {
            return Err(MenuError::IllegalCurrentPaidWipe(wipe.clone()));
        }
        let mut extended = accumulated_prelude.clone();
        extended.wildlife_wipes.push(wipe.clone());
        source.preview_market_prelude(&extended)?;
        Ok(extended)
    }
}

macro_rules! typed_menu_hash {
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
            type Error = MenuError;

            fn try_from(value: String) -> Result<Self, Self::Error> {
                let prefix = format!("{}:", $namespace);
                let digest = value
                    .strip_prefix(&prefix)
                    .ok_or(MenuError::InvalidMenuHash)?;
                digest
                    .parse::<Sha256Digest>()
                    .map_err(|_| MenuError::InvalidMenuHash)?;
                Ok(Self(value))
            }
        }
    };
}

typed_menu_hash!(RulesMenuHash, RULES_MENU_HASH_VERSION);
typed_menu_hash!(IncumbentMenuHash, INCUMBENT_MENU_HASH_VERSION);

impl RulesMenuHash {
    fn for_decisions(decisions: &[RulesDecision]) -> Self {
        let payload = RulesMenuHashPayload {
            schema_id: RULES_MENU_HASH_VERSION,
            ordered_decisions: decisions.iter().map(CanonicalDecision::from).collect(),
        };
        Self::from_digest(hash_menu_payload(&payload))
    }
}

impl IncumbentMenuHash {
    fn for_candidate(
        source_rules_menu_hash: RulesMenuHash,
        rules_indices: &[usize],
        decisions: &[RulesDecision],
    ) -> Self {
        let payload = IncumbentMenuHashPayload {
            schema_id: INCUMBENT_MENU_HASH_VERSION,
            source_rules_menu_hash,
            rules_indices,
            ordered_decisions: decisions.iter().map(CanonicalDecision::from).collect(),
        };
        Self::from_digest(hash_menu_payload(&payload))
    }
}

fn hash_menu_payload(value: &impl Serialize) -> Sha256Digest {
    let value = serde_json::to_value(value)
        .expect("serializing a fixed canonical menu payload cannot fail");
    Sha256Digest::of_bytes(
        &serde_json::to_vec(&value).expect("serializing canonical menu JSON cannot fail"),
    )
}

#[derive(Serialize)]
struct RulesMenuHashPayload {
    schema_id: &'static str,
    ordered_decisions: Vec<CanonicalDecision>,
}

#[derive(Serialize)]
struct IncumbentMenuHashPayload<'a> {
    schema_id: &'static str,
    source_rules_menu_hash: RulesMenuHash,
    rules_indices: &'a [usize],
    ordered_decisions: Vec<CanonicalDecision>,
}

#[derive(Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum CanonicalDecision {
    Prelude {
        replace_three_of_a_kind: bool,
        wildlife_wipes: Vec<Vec<u8>>,
    },
    PaidWipe {
        slots: Vec<u8>,
    },
    Draft {
        action_content_id: ActionContentId,
    },
}

impl From<&RulesDecision> for CanonicalDecision {
    fn from(value: &RulesDecision) -> Self {
        match value {
            RulesDecision::Prelude(prelude) => Self::Prelude {
                replace_three_of_a_kind: prelude.replace_three_of_a_kind,
                wildlife_wipes: canonical_wipes(&prelude.wildlife_wipes),
            },
            RulesDecision::PaidWipe(wipe) => Self::PaidWipe {
                slots: wipe.slots.iter().map(|slot| slot.index() as u8).collect(),
            },
            RulesDecision::Draft(action) => Self::Draft {
                action_content_id: ActionContentId::canonical(action),
            },
        }
    }
}

fn canonical_wipes(wipes: &[WildlifeWipe]) -> Vec<Vec<u8>> {
    wipes
        .iter()
        .map(|wipe| wipe.slots.iter().map(|slot| slot.index() as u8).collect())
        .collect()
}

fn validate_source_config(source: &GameState) -> Result<(), MenuError> {
    source.validate().map_err(MenuError::InvalidSource)?;
    let expected = GameConfig::research_aaaaa(4)
        .expect("the canonical four-player research configuration is valid");
    if source.config() == expected {
        Ok(())
    } else {
        Err(MenuError::WrongGameConfig)
    }
}

#[derive(Debug, Error)]
pub enum MenuError {
    #[error("Rival menu source is not a valid canonical game state: {0}")]
    InvalidSource(&'static str),
    #[error(transparent)]
    Rules(#[from] RuleError),
    #[error("canonical prelude choice failed canonical preview: {prelude:?}: {source}")]
    CanonicalPreludeMismatch {
        prelude: MarketPrelude,
        source: RuleError,
    },
    #[error("canonical action {0} does not preserve the accumulated prelude")]
    CanonicalActionPreludeMismatch(usize),
    #[error("canonical action {index} failed canonical transition: {source}")]
    CanonicalActionTransitionMismatch { index: usize, source: RuleError },
    #[error("paid wipe is not legal at the current revealed draft root: {0:?}")]
    IllegalCurrentPaidWipe(WildlifeWipe),
    #[error("a policy root cannot have an empty menu")]
    EmptyMenu,
    #[error("candidate index {0} is outside the canonical menu")]
    CandidateIndexOutOfRange(usize),
    #[error("candidate index {0} is not a complete post-prelude draft action")]
    NonDraftCandidateIndex(usize),
    #[error("candidate index {0} is duplicated")]
    DuplicateCandidateIndex(usize),
    #[error("invalid namespaced menu hash")]
    InvalidMenuHash,
    #[error("Rival menus require the canonical four-player research configuration")]
    WrongGameConfig,
}

#[cfg(test)]
mod tests {
    use cascadia_game::{GameConfig, GameSeed, MarketSlot};
    use serde_json::Value;

    use super::*;

    fn game(seed: u64) -> GameState {
        GameState::new(
            GameConfig::research_aaaaa(4).unwrap(),
            GameSeed::from_u64(seed),
        )
        .unwrap()
    }

    #[test]
    fn draft_menu_is_exact_canonical_action_order_and_every_action_transitions() {
        let game = game(3);
        let prelude = MarketPrelude::default();
        let canonical = game.legal_turn_actions(&prelude).unwrap();
        let menu = MenuComposer::draft_root(&game, &prelude).unwrap();
        let emitted: Vec<_> = menu
            .decisions()
            .iter()
            .filter_map(|decision| match decision {
                RulesDecision::Draft(action) => Some(action.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(emitted, canonical);
        for action in emitted {
            game.transition(&action).unwrap();
        }
    }

    #[test]
    fn paid_wipes_are_one_step_reveal_decisions_not_future_vectors() {
        let game = with_active_nature_tokens(game(5), 2);
        let initial = MarketPrelude::default();
        let menu = MenuComposer::draft_root(&game, &initial).unwrap();
        let wipe = menu
            .decisions()
            .iter()
            .find_map(|decision| match decision {
                RulesDecision::PaidWipe(wipe) => Some(wipe.clone()),
                _ => None,
            })
            .expect("nature tokens expose one-step paid wipes");
        assert!(menu.decisions().iter().all(|decision| match decision {
            RulesDecision::Draft(action) => action.wildlife_wipes.is_empty(),
            RulesDecision::PaidWipe(_) | RulesDecision::Prelude(_) => true,
        }));

        let extended = MenuComposer::extend_paid_wipe(&game, &initial, &wipe).unwrap();
        assert_eq!(extended.wildlife_wipes.len(), 1);
        let after_reveal = MenuComposer::draft_root(&game, &extended).unwrap();
        assert!(
            after_reveal
                .decisions()
                .iter()
                .all(|decision| match decision {
                    RulesDecision::Draft(action) => action.wildlife_wipes.len() == 1,
                    RulesDecision::PaidWipe(_) => true,
                    RulesDecision::Prelude(_) => false,
                })
        );
        assert!(
            after_reveal
                .decisions()
                .iter()
                .any(|decision| matches!(decision, RulesDecision::PaidWipe(_))),
            "second token permits another decision, but only after first reveal"
        );
    }

    #[test]
    fn prelude_menu_matches_canonical_decline_then_optional_accept() {
        for seed in 0..1000 {
            let game = game(seed);
            let canonical = game.free_three_of_a_kind_choices().unwrap();
            if canonical.len() == 2 {
                let menu = MenuComposer::prelude_root(&game).unwrap();
                let emitted: Vec<_> = menu
                    .decisions()
                    .iter()
                    .map(|decision| match decision {
                        RulesDecision::Prelude(prelude) => prelude.clone(),
                        _ => panic!("wrong decision type"),
                    })
                    .collect();
                assert_eq!(emitted, canonical);
                assert!(!emitted[0].replace_three_of_a_kind);
                assert!(emitted[1].replace_three_of_a_kind);
                return;
            }
        }
        panic!("fixture search did not find a three-of-a-kind market");
    }

    #[test]
    fn rules_and_incumbent_menus_are_distinct_and_order_bound() {
        let game = game(8);
        let rules = MenuComposer::draft_root(&game, &MarketPrelude::default()).unwrap();
        let incumbent = IncumbentCandidateMenu::from_rules_indices(&rules, [2, 0]).unwrap();
        let reversed = IncumbentCandidateMenu::from_rules_indices(&rules, [0, 2]).unwrap();
        assert_ne!(incumbent.hash(), reversed.hash());
        assert_eq!(incumbent.rules_indices(), &[2, 0]);
        let full = IncumbentCandidateMenu::from_rules_indices(&rules, 0..rules.len()).unwrap();
        assert_ne!(full.hash().as_str(), rules.hash().as_str());
        assert!(IncumbentCandidateMenu::from_rules_indices(&rules, [0, 0]).is_err());
    }

    #[test]
    fn incumbent_candidates_reject_one_step_wipes_and_preludes() {
        let token_game = with_active_nature_tokens(game(5), 1);
        let draft_menu = MenuComposer::draft_root(&token_game, &MarketPrelude::default()).unwrap();
        let wipe_index = draft_menu
            .decisions()
            .iter()
            .position(|decision| matches!(decision, RulesDecision::PaidWipe(_)))
            .expect("active nature token exposes a wipe decision");
        assert!(matches!(
            IncumbentCandidateMenu::from_rules_indices(&draft_menu, [wipe_index]),
            Err(MenuError::NonDraftCandidateIndex(index)) if index == wipe_index
        ));

        let prelude_game = (0..1_000)
            .map(game)
            .find(|game| MenuComposer::prelude_root(game).unwrap().len() == 2)
            .expect("fixture search finds a public replacement choice");
        let prelude_menu = MenuComposer::prelude_root(&prelude_game).unwrap();
        assert!(matches!(
            IncumbentCandidateMenu::from_rules_indices(&prelude_menu, [0]),
            Err(MenuError::NonDraftCandidateIndex(0))
        ));
    }

    #[test]
    fn menu_hash_has_strict_namespaced_wire_format() {
        let menu = MenuComposer::draft_root(&game(11), &MarketPrelude::default()).unwrap();
        let serialized = serde_json::to_vec(&menu.hash()).unwrap();
        assert_eq!(
            serde_json::from_slice::<RulesMenuHash>(&serialized).unwrap(),
            menu.hash()
        );
        assert!(serde_json::from_str::<RulesMenuHash>(r#""sha256:00""#).is_err());
        assert!(serde_json::from_slice::<IncumbentMenuHash>(&serialized).is_err());
    }

    fn with_active_nature_tokens(game: GameState, count: u8) -> GameState {
        let active = game.current_player();
        let mut value = serde_json::to_value(game).unwrap();
        value["boards"][active]["nature_tokens"] = Value::from(count);
        let game: GameState = serde_json::from_value(value).unwrap();
        game.validate().unwrap();
        game
    }

    #[test]
    fn an_unregistered_wipe_is_rejected() {
        let game = game(2);
        let wipe = WildlifeWipe {
            slots: vec![MarketSlot::ZERO],
        };
        assert!(MenuComposer::extend_paid_wipe(&game, &MarketPrelude::default(), &wipe).is_err());
    }

    #[test]
    fn a_nonresearch_game_cannot_be_mislabeled_as_a_rival_menu() {
        let game = GameState::new(
            GameConfig::research_aaaaa(2).unwrap(),
            GameSeed::from_u64(3),
        )
        .unwrap();
        assert!(matches!(
            MenuComposer::draft_root(&game, &MarketPrelude::default()),
            Err(MenuError::WrongGameConfig)
        ));
    }

    #[test]
    fn invalid_deserialized_state_fails_before_menu_composition() {
        let mut value = serde_json::to_value(game(13)).unwrap();
        value["current_player"] = serde_json::json!(9);
        let invalid: GameState = serde_json::from_value(value).unwrap();
        assert!(matches!(
            MenuComposer::draft_root(&invalid, &MarketPrelude::default()),
            Err(MenuError::InvalidSource(_))
        ));
        assert!(matches!(
            MenuComposer::prelude_root(&invalid),
            Err(MenuError::InvalidSource(_))
        ));
    }
}
