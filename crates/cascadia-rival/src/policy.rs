use std::{error::Error, fmt};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    FrozenPolicyIdentity, HonestWorldSampler, PolicyRng, PublicPolicyObs, RootKind, RulesLegalMenu,
    SeatLocalMemory,
};

/// Checked index into the exact supplied policy menu.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct MenuIndex(usize);

impl MenuIndex {
    pub fn new(index: usize, menu: &RulesLegalMenu) -> Result<Self, PolicyContractError> {
        if index < menu.len() {
            Ok(Self(index))
        } else {
            Err(PolicyContractError::MenuIndexOutOfRange {
                index,
                menu_len: menu.len(),
            })
        }
    }

    pub const fn get(self) -> usize {
        self.0
    }
}

/// One policy decision and its explicit seat-local next memory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolicyDecision {
    choice: MenuIndex,
    next_memory: SeatLocalMemory,
}

impl PolicyDecision {
    pub fn new(
        index: usize,
        menu: &RulesLegalMenu,
        next_memory: SeatLocalMemory,
    ) -> Result<Self, PolicyContractError> {
        Ok(Self {
            choice: MenuIndex::new(index, menu)?,
            next_memory,
        })
    }

    pub fn choice(&self) -> MenuIndex {
        self.choice
    }

    pub fn next_memory(&self) -> &SeatLocalMemory {
        &self.next_memory
    }

    pub fn into_next_memory(self) -> SeatLocalMemory {
        self.next_memory
    }
}

/// Public-input-only interface for a frozen Rival policy.
///
/// The policy may branch over honest redeterminations through the opaque
/// sampler, but neither its top-level input nor `PolicyWorld` exposes the true
/// simulator state, physical seed, hidden order, or scenario key.
///
/// ```compile_fail
/// use cascadia_game::GameState;
/// use cascadia_rival::PublicPolicyObs;
/// fn policy_kernel(_: &PublicPolicyObs) {}
/// fn forbidden(private: &GameState) {
///     policy_kernel(private);
/// }
/// ```
pub trait FrozenPolicy {
    type Identity: FrozenPolicyIdentity;
    type Error: Error + Send + Sync + 'static;

    fn identity(&self) -> &Self::Identity;

    /// Construct a behaviorally clean policy instance for one table seat.
    /// All recurrent, action-affecting state must flow through
    /// [`SeatLocalMemory`](crate::SeatLocalMemory); mutable implementation
    /// state may only be a semantics-neutral cache. The harness calls this
    /// independently for every seat and branch.
    fn fresh_instance(&self) -> Self;

    fn act(
        &mut self,
        observation: &PublicPolicyObs,
        menu: &RulesLegalMenu,
        worlds: &HonestWorldSampler,
        rng: &mut PolicyRng,
    ) -> Result<PolicyDecision, Self::Error>;
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum PolicyContractError {
    #[error("menu index {index} is outside menu length {menu_len}")]
    MenuIndexOutOfRange { index: usize, menu_len: usize },
    #[error("policy expected {expected:?}, received {actual:?}")]
    WrongRootKind {
        expected: RootKind,
        actual: RootKind,
    },
}

/// Small helper for policy implementations that enforce one root kind.
pub fn require_root_kind(
    menu: &RulesLegalMenu,
    expected: RootKind,
) -> Result<(), PolicyContractError> {
    let actual = menu.root_kind();
    if actual == expected {
        Ok(())
    } else {
        Err(PolicyContractError::WrongRootKind { expected, actual })
    }
}

/// Erases an implementation error while preserving a useful source chain for
/// reference harnesses that execute heterogeneous frozen policies.
#[derive(Debug)]
pub struct BoxedPolicyError(Box<dyn Error + Send + Sync>);

impl BoxedPolicyError {
    pub fn new(error: impl Error + Send + Sync + 'static) -> Self {
        Self(Box::new(error))
    }
}

impl fmt::Display for BoxedPolicyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

impl Error for BoxedPolicyError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(self.0.as_ref())
    }
}

#[cfg(test)]
mod tests {
    use cascadia_game::{GameConfig, GameSeed, GameState, MarketPrelude};

    use crate::{MenuComposer, RulesDecision};

    use super::*;

    #[test]
    fn policy_decision_cannot_name_an_action_outside_supplied_menu() {
        let game = GameState::new(
            GameConfig::research_aaaaa(4).unwrap(),
            GameSeed::from_u64(1),
        )
        .unwrap();
        let menu = MenuComposer::draft_root(&game, &MarketPrelude::default()).unwrap();
        assert!(PolicyDecision::new(menu.len(), &menu, SeatLocalMemory::empty()).is_err());
        let valid = PolicyDecision::new(0, &menu, SeatLocalMemory::new(vec![1])).unwrap();
        assert_eq!(valid.choice().get(), 0);
        assert_eq!(valid.next_memory().payload(), &[1]);
        assert!(matches!(menu.decision(0), Some(RulesDecision::Draft(_))));
    }

    #[test]
    fn root_kind_check_is_explicit() {
        let game = GameState::new(
            GameConfig::research_aaaaa(4).unwrap(),
            GameSeed::from_u64(2),
        )
        .unwrap();
        let draft = MenuComposer::draft_root(&game, &MarketPrelude::default()).unwrap();
        assert!(require_root_kind(&draft, RootKind::DraftPolicyRoot).is_ok());
        assert!(require_root_kind(&draft, RootKind::PreludePolicyRoot).is_err());
    }
}
