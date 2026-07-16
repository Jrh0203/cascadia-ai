//! Rules-aware certified score-difference bounds.
//!
//! Python may consume these certificates but must never derive a Cascadia
//! range itself.  The first certificate is intentionally global and
//! conservative.  Its value is an analytic relaxation of the corrected
//! four-player AAAAA rules, not an observed sample range.

use cascadia_game::{GameConfig, GameState, MAX_BOARD_TILES};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{ResearchRulesetIdentity, Sha256Digest};

pub const BOUND_CERTIFICATE_SCHEMA_ID: &str = "cascadiav3.rival_bound_certificate.v1";
pub const GLOBAL_RESEARCH_BOUND_AUTHORITY_ID: &str =
    "cascadia-rival/rust-global-aaaaa-score-relaxation-v1";

/// Human- and machine-auditable terms in the global proof.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GlobalBoundDerivation {
    pub maximum_board_tiles: u16,
    pub terrain_memberships_per_tile: u16,
    pub maximum_personal_wildlife_placements: u16,
    pub maximum_points_per_wildlife_token: u16,
    pub maximum_nature_tokens_earned: u16,
    pub habitat_bonus_points: u16,
}

impl GlobalBoundDerivation {
    pub const fn terminal_score_upper(self) -> u16 {
        self.maximum_board_tiles * self.terrain_memberships_per_tile
            + self.maximum_personal_wildlife_placements * self.maximum_points_per_wildlife_token
            + self.maximum_nature_tokens_earned
            + self.habitat_bonus_points
    }
}

/// Global range for one terminal own score and for a two-action difference.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "BoundWire", into = "BoundWire")]
pub struct CertifiedScoreDifferenceBound {
    schema_id: String,
    authority_id: String,
    ruleset: ResearchRulesetIdentity,
    scope: String,
    terminal_score_min: i32,
    terminal_score_max: i32,
    score_difference_min: i32,
    score_difference_max: i32,
    score_difference_width: u32,
    derivation: GlobalBoundDerivation,
    certificate_sha256: Sha256Digest,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct BoundWire {
    schema_id: String,
    authority_id: String,
    ruleset: ResearchRulesetIdentity,
    scope: String,
    terminal_score_min: i32,
    terminal_score_max: i32,
    score_difference_min: i32,
    score_difference_max: i32,
    score_difference_width: u32,
    derivation: GlobalBoundDerivation,
    certificate_sha256: Sha256Digest,
}

#[derive(Debug, Serialize)]
struct BoundContent<'a> {
    schema_id: &'a str,
    authority_id: &'a str,
    ruleset: &'a ResearchRulesetIdentity,
    scope: &'a str,
    terminal_score_min: i32,
    terminal_score_max: i32,
    score_difference_min: i32,
    score_difference_max: i32,
    score_difference_width: u32,
    derivation: GlobalBoundDerivation,
}

impl CertifiedScoreDifferenceBound {
    /// Analytic global certificate for corrected-rules four-player AAAAA play.
    ///
    /// Proof sketch:
    ///
    /// * a board has at most 23 tiles and each tile contributes membership to
    ///   at most two terrain graphs, so the sum of five largest habitats is at
    ///   most 46;
    /// * a player can place at most one drafted wildlife on each of 20 turns;
    ///   every A-card species awards at most five points per token (Fox A is
    ///   the tight per-token envelope; Bear, Elk, Salmon, and Hawk are lower);
    /// * at most one Nature token is earned by each placed wildlife, so at
    ///   most 20 can remain; and
    /// * research AAAAA disables habitat bonuses.
    ///
    /// Thus every terminal own score is in `[0, 166]`, every two-action
    /// difference is in `[-166, 166]`, and its Hoeffding range width is 332.
    pub fn global_research_aaaaa() -> Self {
        let derivation = GlobalBoundDerivation {
            maximum_board_tiles: MAX_BOARD_TILES as u16,
            terrain_memberships_per_tile: 2,
            maximum_personal_wildlife_placements: 20,
            maximum_points_per_wildlife_token: 5,
            maximum_nature_tokens_earned: 20,
            habitat_bonus_points: 0,
        };
        let terminal_score_min = 0;
        let terminal_score_max = i32::from(derivation.terminal_score_upper());
        let score_difference_min = -terminal_score_max;
        let score_difference_max = terminal_score_max;
        let score_difference_width = (score_difference_max - score_difference_min) as u32;
        let ruleset = ResearchRulesetIdentity::canonical();
        let schema_id = BOUND_CERTIFICATE_SCHEMA_ID.to_owned();
        let authority_id = GLOBAL_RESEARCH_BOUND_AUTHORITY_ID.to_owned();
        let scope = "global_terminal_own_score_difference".to_owned();
        let content = BoundContent {
            schema_id: &schema_id,
            authority_id: &authority_id,
            ruleset: &ruleset,
            scope: &scope,
            terminal_score_min,
            terminal_score_max,
            score_difference_min,
            score_difference_max,
            score_difference_width,
            derivation,
        };
        // `serde_json::Map` is key-sorted without the `preserve_order`
        // feature.  Hashing the resulting compact value therefore matches
        // Python's `json.dumps(..., sort_keys=True, separators=(",", ":"))`.
        let value = serde_json::to_value(&content)
            .expect("serializing a fixed bound certificate cannot fail");
        let bytes =
            serde_json::to_vec(&value).expect("serializing canonical bound JSON cannot fail");
        Self {
            schema_id,
            authority_id,
            ruleset,
            scope,
            terminal_score_min,
            terminal_score_max,
            score_difference_min,
            score_difference_max,
            score_difference_width,
            derivation,
            certificate_sha256: Sha256Digest::of_bytes(&bytes),
        }
    }

    pub fn for_game(game: &GameState) -> Result<Self, BoundCertificateError> {
        let expected = GameConfig::research_aaaaa(4)
            .expect("canonical research configuration must remain valid");
        if game.config() != expected {
            return Err(BoundCertificateError::UnsupportedGameConfig);
        }
        Ok(Self::global_research_aaaaa())
    }

    pub fn validate(&self) -> Result<(), BoundCertificateError> {
        let expected = Self::global_research_aaaaa();
        if self == &expected {
            Ok(())
        } else {
            Err(BoundCertificateError::NonCanonical)
        }
    }

    pub fn schema_id(&self) -> &str {
        &self.schema_id
    }

    pub fn authority_id(&self) -> &str {
        &self.authority_id
    }

    pub fn terminal_score_range(&self) -> (i32, i32) {
        (self.terminal_score_min, self.terminal_score_max)
    }

    pub fn score_difference_range(&self) -> (i32, i32) {
        (self.score_difference_min, self.score_difference_max)
    }

    pub fn score_difference_width(&self) -> u32 {
        self.score_difference_width
    }

    pub fn derivation(&self) -> GlobalBoundDerivation {
        self.derivation
    }

    pub fn certificate_sha256(&self) -> &Sha256Digest {
        &self.certificate_sha256
    }

    pub fn contains_terminal_score(&self, score: i32) -> bool {
        (self.terminal_score_min..=self.terminal_score_max).contains(&score)
    }

    pub fn contains_score_difference(&self, difference: i32) -> bool {
        (self.score_difference_min..=self.score_difference_max).contains(&difference)
    }
}

impl From<CertifiedScoreDifferenceBound> for BoundWire {
    fn from(value: CertifiedScoreDifferenceBound) -> Self {
        Self {
            schema_id: value.schema_id,
            authority_id: value.authority_id,
            ruleset: value.ruleset,
            scope: value.scope,
            terminal_score_min: value.terminal_score_min,
            terminal_score_max: value.terminal_score_max,
            score_difference_min: value.score_difference_min,
            score_difference_max: value.score_difference_max,
            score_difference_width: value.score_difference_width,
            derivation: value.derivation,
            certificate_sha256: value.certificate_sha256,
        }
    }
}

impl TryFrom<BoundWire> for CertifiedScoreDifferenceBound {
    type Error = BoundCertificateError;

    fn try_from(value: BoundWire) -> Result<Self, Self::Error> {
        let certificate = Self {
            schema_id: value.schema_id,
            authority_id: value.authority_id,
            ruleset: value.ruleset,
            scope: value.scope,
            terminal_score_min: value.terminal_score_min,
            terminal_score_max: value.terminal_score_max,
            score_difference_min: value.score_difference_min,
            score_difference_max: value.score_difference_max,
            score_difference_width: value.score_difference_width,
            derivation: value.derivation,
            certificate_sha256: value.certificate_sha256,
        };
        certificate.validate()?;
        Ok(certificate)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum BoundCertificateError {
    #[error("bound certificate supports only corrected-rules four-player research AAAAA")]
    UnsupportedGameConfig,
    #[error("bound certificate is not the canonical Rust-authored global certificate")]
    NonCanonical,
}

#[cfg(test)]
mod tests {
    use cascadia_game::{GameSeed, MarketPrelude, score_game};

    use super::*;

    #[test]
    fn global_certificate_has_auditable_exact_arithmetic() {
        let certificate = CertifiedScoreDifferenceBound::global_research_aaaaa();
        assert_eq!(certificate.terminal_score_range(), (0, 166));
        assert_eq!(certificate.score_difference_range(), (-166, 166));
        assert_eq!(certificate.score_difference_width(), 332);
        assert_eq!(certificate.derivation().terminal_score_upper(), 166);
        certificate.validate().unwrap();
    }

    #[test]
    fn certificate_roundtrip_is_fail_closed() {
        let certificate = CertifiedScoreDifferenceBound::global_research_aaaaa();
        let mut value = serde_json::to_value(&certificate).unwrap();
        let decoded: CertifiedScoreDifferenceBound = serde_json::from_value(value.clone()).unwrap();
        assert_eq!(decoded, certificate);

        value["score_difference_width"] = serde_json::json!(331);
        assert!(serde_json::from_value::<CertifiedScoreDifferenceBound>(value).is_err());
    }

    #[test]
    fn rust_and_python_share_one_locked_certificate_fixture() {
        let fixture = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../cascadiav3/tests/fixtures/rival/global_bound_certificate_v1.json"
        ));
        let decoded: CertifiedScoreDifferenceBound = serde_json::from_str(fixture).unwrap();
        let canonical = CertifiedScoreDifferenceBound::global_research_aaaaa();
        assert_eq!(decoded, canonical);
        assert_eq!(
            canonical.certificate_sha256().as_str(),
            "sha256:67fa21e1f4e887f73a1f0f4e22397ca23f79ca67972b44e34f94f385734eec64"
        );
    }

    #[test]
    fn reachable_cpu_smoke_never_exceeds_the_certificate() {
        let config = GameConfig::research_aaaaa(4).unwrap();
        for seed in 0..2 {
            let mut game = GameState::new(config, GameSeed::from_u64(seed)).unwrap();
            let certificate = CertifiedScoreDifferenceBound::for_game(&game).unwrap();
            while !game.is_game_over() {
                let action = game
                    .legal_turn_actions(&MarketPrelude::default())
                    .unwrap()
                    .into_iter()
                    .next()
                    .unwrap();
                game.apply(&action).unwrap();
                for score in score_game(&game) {
                    assert!(certificate.contains_terminal_score(i32::from(score.total)));
                }
            }
            let scores = score_game(&game);
            for left in &scores {
                for right in &scores {
                    assert!(
                        certificate.contains_score_difference(
                            i32::from(left.total) - i32::from(right.total)
                        )
                    );
                }
            }
        }
    }

    #[test]
    fn nonresearch_configuration_is_rejected() {
        let game = GameState::new(
            GameConfig::standard(4, cascadia_game::ScoringCards::AAAAA).unwrap(),
            GameSeed::from_u64(1),
        )
        .unwrap();
        assert_eq!(
            CertifiedScoreDifferenceBound::for_game(&game),
            Err(BoundCertificateError::UnsupportedGameConfig)
        );
    }
}
