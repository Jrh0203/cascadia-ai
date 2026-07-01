use cascadia_core::game::GameState;
use cascadia_core::scoring::ScoreBreakdown;
use cascadia_core::types::{ScoringCardVariant, ScoringCards};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ScoreTarget {
    Base = 0,
    WithHabitatBonus = 1,
}

impl ScoreTarget {
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "base" | "no-bonus" | "no_bonus" => Some(Self::Base),
            "with-bonus" | "with_bonus" | "bonus" | "habitat-bonus" | "habitat_bonus" => {
                Some(Self::WithHabitatBonus)
            }
            _ => None,
        }
    }

    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Base),
            1 => Some(Self::WithHabitatBonus),
            _ => None,
        }
    }

    pub fn as_u8(self) -> u8 {
        self as u8
    }

    pub fn from_env(default: Self) -> Self {
        std::env::var("CASCADIA_SCORE_TARGET")
            .ok()
            .or_else(|| std::env::var("MCE_SCORE_TARGET").ok())
            .and_then(|s| Self::parse(&s))
            .unwrap_or(default)
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Base => "base",
            Self::WithHabitatBonus => "with-bonus",
        }
    }
}

pub fn is_all_a(cards: &ScoringCards) -> bool {
    cards.cards.iter().all(|&c| c == ScoringCardVariant::A)
}

pub fn score_breakdown(game: &mut GameState, player: usize, target: ScoreTarget) -> ScoreBreakdown {
    match target {
        ScoreTarget::Base => ScoreBreakdown::compute(&mut game.boards[player], &game.scoring_cards),
        ScoreTarget::WithHabitatBonus => {
            ScoreBreakdown::compute_with_bonuses(&mut game.boards, &game.scoring_cards, player)
        }
    }
}

#[inline]
pub fn score_total(game: &mut GameState, player: usize, target: ScoreTarget) -> u16 {
    score_breakdown(game, player, target).total
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn parses_score_targets() {
        assert_eq!(ScoreTarget::parse("base"), Some(ScoreTarget::Base));
        assert_eq!(
            ScoreTarget::parse("with-bonus"),
            Some(ScoreTarget::WithHabitatBonus)
        );
        assert_eq!(ScoreTarget::from_u8(1), Some(ScoreTarget::WithHabitatBonus));
        assert_eq!(ScoreTarget::from_u8(9), None);
    }

    #[test]
    fn with_bonus_can_differ_from_base() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let mut game = GameState::new(2, ScoringCards::all_a(), &mut rng);
        game.boards[0].largest_group[0] = 10;
        game.boards[1].largest_group[0] = 1;

        let base = score_total(&mut game.clone(), 0, ScoreTarget::Base);
        let with_bonus = score_total(&mut game, 0, ScoreTarget::WithHabitatBonus);
        assert!(with_bonus > base);
    }

    #[test]
    fn all_a_detection_accepts_configuration_a() {
        assert!(is_all_a(&ScoringCards::all_a()));
    }
}
