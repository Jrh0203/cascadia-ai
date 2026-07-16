use cascadia_game::{GameConfig, RULES_SEMANTICS_ID};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::Sha256Digest;

pub const RESEARCH_RULESET_SCHEMA_ID: &str = "cascadiav3.research_ruleset_identity.v1";
pub const LEGACY_RESEARCH_RULESET_ID: &str =
    "cascadia_research_aaaaa_4p_card_a_no_habitat_bonus_rules_2026_07_16";

/// The single structured owner of the corrected four-player research rules.
///
/// Deserialization accepts exactly the canonical tuple. A familiar label by
/// itself is deliberately insufficient scientific identity.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "RulesetWire", into = "RulesetWire")]
pub struct ResearchRulesetIdentity {
    schema_id: String,
    legacy_ruleset_id: String,
    rules_semantics_id: String,
    game_config_sha256: Sha256Digest,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RulesetWire {
    schema_id: String,
    legacy_ruleset_id: String,
    rules_semantics_id: String,
    game_config_sha256: Sha256Digest,
}

impl ResearchRulesetIdentity {
    pub fn canonical() -> Self {
        let config = GameConfig::research_aaaaa(4)
            .expect("the canonical four-player research configuration is valid");
        let config_bytes = serde_json::to_vec(&config)
            .expect("serializing a fixed in-memory game configuration cannot fail");
        Self {
            schema_id: RESEARCH_RULESET_SCHEMA_ID.to_owned(),
            legacy_ruleset_id: LEGACY_RESEARCH_RULESET_ID.to_owned(),
            rules_semantics_id: RULES_SEMANTICS_ID.to_owned(),
            game_config_sha256: Sha256Digest::of_bytes(&config_bytes),
        }
    }

    pub fn schema_id(&self) -> &str {
        &self.schema_id
    }

    pub fn legacy_ruleset_id(&self) -> &str {
        &self.legacy_ruleset_id
    }

    pub fn rules_semantics_id(&self) -> &str {
        &self.rules_semantics_id
    }

    pub fn game_config_sha256(&self) -> &Sha256Digest {
        &self.game_config_sha256
    }

    pub fn validate(&self) -> Result<(), RulesetIdentityError> {
        let expected = Self::canonical();
        for (field, matches) in [
            ("schema_id", self.schema_id == expected.schema_id),
            (
                "legacy_ruleset_id",
                self.legacy_ruleset_id == expected.legacy_ruleset_id,
            ),
            (
                "rules_semantics_id",
                self.rules_semantics_id == expected.rules_semantics_id,
            ),
            (
                "game_config_sha256",
                self.game_config_sha256 == expected.game_config_sha256,
            ),
        ] {
            if !matches {
                return Err(RulesetIdentityError::NonCanonical(field));
            }
        }
        Ok(())
    }
}

impl From<ResearchRulesetIdentity> for RulesetWire {
    fn from(value: ResearchRulesetIdentity) -> Self {
        Self {
            schema_id: value.schema_id,
            legacy_ruleset_id: value.legacy_ruleset_id,
            rules_semantics_id: value.rules_semantics_id,
            game_config_sha256: value.game_config_sha256,
        }
    }
}

impl TryFrom<RulesetWire> for ResearchRulesetIdentity {
    type Error = RulesetIdentityError;

    fn try_from(value: RulesetWire) -> Result<Self, Self::Error> {
        let identity = Self {
            schema_id: value.schema_id,
            legacy_ruleset_id: value.legacy_ruleset_id,
            rules_semantics_id: value.rules_semantics_id,
            game_config_sha256: value.game_config_sha256,
        };
        identity.validate()?;
        Ok(identity)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum RulesetIdentityError {
    #[error("research ruleset field {0} is not canonical")]
    NonCanonical(&'static str),
}

#[cfg(test)]
mod tests {
    use serde_json::Value;

    use super::*;

    #[test]
    fn canonical_identity_binds_all_three_rules_layers() {
        let identity = ResearchRulesetIdentity::canonical();
        assert_eq!(identity.schema_id(), RESEARCH_RULESET_SCHEMA_ID);
        assert_eq!(identity.legacy_ruleset_id(), LEGACY_RESEARCH_RULESET_ID);
        assert_eq!(identity.rules_semantics_id(), RULES_SEMANTICS_ID);
        assert_eq!(
            identity.game_config_sha256().as_str(),
            "sha256:f5b2c782a483db870c50366b33cccde6d9a82a92a571cf9f29c752b750a5c07c"
        );
    }

    #[test]
    fn every_wire_field_is_required_and_unknown_fields_reject() {
        let value = serde_json::to_value(ResearchRulesetIdentity::canonical()).unwrap();
        let object = value.as_object().unwrap();
        for field in object.keys() {
            let mut missing = value.clone();
            missing.as_object_mut().unwrap().remove(field);
            assert!(
                serde_json::from_value::<ResearchRulesetIdentity>(missing).is_err(),
                "missing {field} unexpectedly passed"
            );
        }

        let mut extra = value;
        extra
            .as_object_mut()
            .unwrap()
            .insert("table_total".to_owned(), Value::Bool(false));
        assert!(serde_json::from_value::<ResearchRulesetIdentity>(extra).is_err());
    }

    #[test]
    fn a_familiar_label_cannot_mask_changed_semantics() {
        let mut value = serde_json::to_value(ResearchRulesetIdentity::canonical()).unwrap();
        value["rules_semantics_id"] = Value::String("stale-semantics".to_owned());
        assert!(serde_json::from_value::<ResearchRulesetIdentity>(value).is_err());
    }
}
