use std::{fs, path::Path};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::{Result, V3Error};

pub const V3_CAMPAIGN_STATE_SCHEMA_ID: &str = "cascadia-v3-campaign-state-v1";
pub const V3_CAMPAIGN_ID: &str = "cascadia-v3-radius7-stockfish-nnue-v1";

/// Stable read contract for the checksum-chained Python campaign controller.
/// Unknown operational fields remain forward-compatible; these fields are the
/// scientific safety boundary consumed by Rust workers and validators.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct V3CampaignState {
    pub schema_id: String,
    pub campaign_id: String,
    pub phase: String,
    pub part: u8,
    pub round_index: Option<u8>,
    pub legal_next_transitions: Vec<String>,
    pub phase2_authorized: bool,
    pub protected_seed_values_opened: bool,
    pub scientific_training_started: bool,
    pub john4_compute_authorized: bool,
    pub readiness_sha256: Option<String>,
    pub approved_readiness_sha256: Option<String>,
    pub transition_sequence: Option<u64>,
    pub previous_state_sha256: Option<String>,
    pub state_sha256: Option<String>,
}

impl V3CampaignState {
    /// Load a campaign state only after verifying its canonical SHA-256 chain
    /// member. JSON object maps are serialized in key order, matching the
    /// controller's compact canonical encoding.
    pub fn load_verified(path: &Path) -> Result<Self> {
        let bytes = fs::read(path)?;
        let mut value: Value = serde_json::from_slice(&bytes)?;
        let recorded = value
            .as_object_mut()
            .ok_or_else(|| {
                V3Error::InvalidTraining("V3 campaign state must be a JSON object".to_owned())
            })?
            .remove("state_sha256")
            .and_then(|value| value.as_str().map(str::to_owned))
            .ok_or_else(|| {
                V3Error::InvalidTraining("V3 campaign state has no checksum".to_owned())
            })?;
        let observed = format!("{:x}", Sha256::digest(serde_json::to_vec(&value)?));
        if recorded != observed {
            return Err(V3Error::ChecksumMismatch("V3 campaign state".to_owned()));
        }
        value
            .as_object_mut()
            .expect("object shape was verified above")
            .insert("state_sha256".to_owned(), Value::String(recorded));
        let state: Self = serde_json::from_value(value)?;
        state.validate_boundary()?;
        Ok(state)
    }

    pub fn validate_boundary(&self) -> Result<()> {
        let final_phase = matches!(
            self.phase.as_str(),
            "final_protected_comparison" | "final_all_v3_evaluation" | "complete"
        );
        if self.schema_id != V3_CAMPAIGN_STATE_SCHEMA_ID
            || self.campaign_id != V3_CAMPAIGN_ID
            || !(1..=2).contains(&self.part)
            || self
                .round_index
                .is_some_and(|cycle| !(1..=10).contains(&cycle))
            || self.john4_compute_authorized
            || (self.protected_seed_values_opened && !final_phase)
            || (self.part == 1
                && (self.phase2_authorized
                    || self.protected_seed_values_opened
                    || self.scientific_training_started))
            || (self.part == 2
                && (!self.phase2_authorized || self.approved_readiness_sha256.is_none()))
        {
            return Err(V3Error::InvalidTraining(
                "V3 campaign state violates the approval or protected-domain boundary".to_owned(),
            ));
        }
        for digest in [
            self.readiness_sha256.as_deref(),
            self.approved_readiness_sha256.as_deref(),
            self.previous_state_sha256.as_deref(),
            self.state_sha256.as_deref(),
        ]
        .into_iter()
        .flatten()
        {
            if digest.len() != 64 || !digest.bytes().all(|value| value.is_ascii_hexdigit()) {
                return Err(V3Error::InvalidTraining(
                    "V3 campaign state contains an invalid checksum".to_owned(),
                ));
            }
        }
        Ok(())
    }

    pub fn awaiting_phase2_approval(&self) -> bool {
        self.part == 1
            && self.phase == "awaiting_phase2_approval"
            && !self.phase2_authorized
            && !self.protected_seed_values_opened
            && !self.scientific_training_started
    }
}

#[cfg(test)]
mod tests {
    use std::{fs, time::SystemTime};

    use super::*;

    #[test]
    fn part1_stop_cannot_claim_scientific_authority() {
        let mut state = V3CampaignState {
            schema_id: V3_CAMPAIGN_STATE_SCHEMA_ID.to_owned(),
            campaign_id: V3_CAMPAIGN_ID.to_owned(),
            phase: "awaiting_phase2_approval".to_owned(),
            part: 1,
            round_index: None,
            legal_next_transitions: vec!["authorize_phase2".to_owned()],
            phase2_authorized: false,
            protected_seed_values_opened: false,
            scientific_training_started: false,
            john4_compute_authorized: false,
            readiness_sha256: Some("a".repeat(64)),
            approved_readiness_sha256: None,
            transition_sequence: Some(4),
            previous_state_sha256: Some("b".repeat(64)),
            state_sha256: Some("c".repeat(64)),
        };
        state.validate_boundary().unwrap();
        assert!(state.awaiting_phase2_approval());
        state.scientific_training_started = true;
        assert!(state.validate_boundary().is_err());
    }

    #[test]
    fn verified_loader_rejects_a_mutated_chain_member() {
        let path = std::env::temp_dir().join(format!(
            "cascadia-v3-campaign-state-{}-{}.json",
            std::process::id(),
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut value = serde_json::json!({
            "schema_id": V3_CAMPAIGN_STATE_SCHEMA_ID,
            "campaign_id": V3_CAMPAIGN_ID,
            "phase": "awaiting_phase2_approval",
            "part": 1,
            "round_index": null,
            "legal_next_transitions": ["authorize_phase2"],
            "phase2_authorized": false,
            "protected_seed_values_opened": false,
            "scientific_training_started": false,
            "john4_compute_authorized": false,
            "readiness_sha256": "a".repeat(64),
            "approved_readiness_sha256": null,
            "transition_sequence": 4,
            "previous_state_sha256": "b".repeat(64)
        });
        let checksum = format!("{:x}", Sha256::digest(serde_json::to_vec(&value).unwrap()));
        value
            .as_object_mut()
            .unwrap()
            .insert("state_sha256".to_owned(), Value::String(checksum));
        fs::write(&path, serde_json::to_vec_pretty(&value).unwrap()).unwrap();
        assert!(V3CampaignState::load_verified(&path).is_ok());

        value["phase"] = Value::String("bootstrap_collecting".to_owned());
        fs::write(&path, serde_json::to_vec_pretty(&value).unwrap()).unwrap();
        assert!(V3CampaignState::load_verified(&path).is_err());
        fs::remove_file(path).unwrap();
    }
}
