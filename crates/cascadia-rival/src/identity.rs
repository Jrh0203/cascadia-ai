use serde::{Deserialize, Serialize, de::DeserializeOwned};
use thiserror::Error;

use crate::{ResearchRulesetIdentity, Sha256Digest};

pub const POLICY_IDENTITY_SCHEMA_ID: &str = "cascadiav3.rival_policy_identity.v1";
pub const CANONICAL_SIMULATOR_ID: &str = "cascadia-game-canonical";

/// Complete identity of one frozen policy implementation and its behavior.
///
/// Configurations are hash-pinned instead of duplicated here. This keeps the
/// identity complete when a policy-specific knob set evolves: launch code
/// must bind the canonical bytes of each whole config, not a hand-maintained
/// subset of currently interesting fields.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolicyIdentityFields {
    pub ruleset: ResearchRulesetIdentity,
    pub source_revision: String,
    pub source_digest: Sha256Digest,
    pub executable_sha256: Sha256Digest,
    pub model_manifest_sha256: Sha256Digest,
    pub checkpoint_sha256: Sha256Digest,
    pub weights_sha256: Sha256Digest,
    pub bridge_protocol: String,
    pub tensor_schema: String,
    pub numerical_mode: NumericalMode,
    pub precision: Precision,
    pub gumbel_config_sha256: Sha256Digest,
    pub search_config_sha256: Sha256Digest,
    pub refresh_config_sha256: Sha256Digest,
    pub exact_endgame_config_sha256: Sha256Digest,
    pub action_content_id_version: String,
    pub rules_action_occurrence_id_version: String,
    pub candidate_action_occurrence_id_version: String,
    pub rules_menu_hash_version: String,
    pub incumbent_menu_hash_version: String,
    pub rng_contracts: RngContractIdentity,
    pub public_observation_schema: String,
    pub policy_memory_schema: String,
    pub failure_behavior: FailureBehavior,
    pub compiler_identity: String,
    pub simulator_identity: String,
    pub sampler_identity: String,
    pub candidate_generator_identity: String,
    pub forbidden_capabilities: ForbiddenCapabilities,
}

impl PolicyIdentityFields {
    pub fn validate(&self) -> Result<(), PolicyIdentityError> {
        self.ruleset.validate()?;
        for (name, value) in [
            ("source_revision", self.source_revision.as_str()),
            ("bridge_protocol", self.bridge_protocol.as_str()),
            ("tensor_schema", self.tensor_schema.as_str()),
            (
                "action_content_id_version",
                self.action_content_id_version.as_str(),
            ),
            (
                "rules_action_occurrence_id_version",
                self.rules_action_occurrence_id_version.as_str(),
            ),
            (
                "candidate_action_occurrence_id_version",
                self.candidate_action_occurrence_id_version.as_str(),
            ),
            (
                "rules_menu_hash_version",
                self.rules_menu_hash_version.as_str(),
            ),
            (
                "incumbent_menu_hash_version",
                self.incumbent_menu_hash_version.as_str(),
            ),
            (
                "public_observation_schema",
                self.public_observation_schema.as_str(),
            ),
            ("policy_memory_schema", self.policy_memory_schema.as_str()),
            ("compiler_identity", self.compiler_identity.as_str()),
            ("simulator_identity", self.simulator_identity.as_str()),
            ("sampler_identity", self.sampler_identity.as_str()),
            (
                "candidate_generator_identity",
                self.candidate_generator_identity.as_str(),
            ),
        ] {
            if value.trim().is_empty() {
                return Err(PolicyIdentityError::EmptyField(name));
            }
        }
        self.rng_contracts.validate()?;
        self.forbidden_capabilities.validate()?;
        if self.failure_behavior.fallback != FailureDisposition::Forbidden {
            return Err(PolicyIdentityError::FallbackMustBeForbidden);
        }
        Ok(())
    }

    /// Require exact compatibility with the currently compiled CPU reference
    /// harness. A complete identity may describe another frozen implementation,
    /// but it cannot be executed here under merely familiar nonempty labels.
    pub fn validate_for_cpu_reference_harness(&self) -> Result<(), PolicyIdentityError> {
        self.validate()?;
        if self.numerical_mode != NumericalMode::Deterministic {
            return Err(PolicyIdentityError::IncompatibleHarnessExecution(
                "numerical_mode",
            ));
        }
        if self.precision != Precision::Fp32 {
            return Err(PolicyIdentityError::IncompatibleHarnessExecution(
                "precision",
            ));
        }
        for (field, observed, expected) in [
            (
                "action_content_id_version",
                self.action_content_id_version.as_str(),
                crate::ACTION_CONTENT_ID_VERSION,
            ),
            (
                "rules_action_occurrence_id_version",
                self.rules_action_occurrence_id_version.as_str(),
                crate::ROOT_ACTION_OCCURRENCE_ID_VERSION,
            ),
            (
                "candidate_action_occurrence_id_version",
                self.candidate_action_occurrence_id_version.as_str(),
                crate::CANDIDATE_ACTION_OCCURRENCE_ID_VERSION,
            ),
            (
                "rules_menu_hash_version",
                self.rules_menu_hash_version.as_str(),
                crate::RULES_MENU_HASH_VERSION,
            ),
            (
                "incumbent_menu_hash_version",
                self.incumbent_menu_hash_version.as_str(),
                crate::INCUMBENT_MENU_HASH_VERSION,
            ),
            (
                "public_observation_schema",
                self.public_observation_schema.as_str(),
                crate::PUBLIC_POLICY_OBSERVATION_SCHEMA_ID,
            ),
            (
                "policy_memory_schema",
                self.policy_memory_schema.as_str(),
                crate::SEAT_LOCAL_MEMORY_SCHEMA_ID,
            ),
            (
                "compiler_identity",
                self.compiler_identity.as_str(),
                crate::DENSE_COMPILER_ID,
            ),
            (
                "simulator_identity",
                self.simulator_identity.as_str(),
                CANONICAL_SIMULATOR_ID,
            ),
            (
                "sampler_identity",
                self.sampler_identity.as_str(),
                crate::INDEPENDENT_SCENARIO_SAMPLER_ID,
            ),
        ] {
            if observed != expected {
                return Err(PolicyIdentityError::IncompatibleHarnessContract {
                    field,
                    observed: observed.to_owned(),
                    expected,
                });
            }
        }
        for (field, observed) in [
            (
                "rng_contracts.physical",
                self.rng_contracts.physical.as_str(),
            ),
            ("rng_contracts.policy", self.rng_contracts.policy.as_str()),
            (
                "rng_contracts.redetermination",
                self.rng_contracts.redetermination.as_str(),
            ),
            ("rng_contracts.search", self.rng_contracts.search.as_str()),
            (
                "rng_contracts.tie_break",
                self.rng_contracts.tie_break.as_str(),
            ),
        ] {
            if observed != crate::RNG_CONTRACT_ID {
                return Err(PolicyIdentityError::IncompatibleHarnessContract {
                    field,
                    observed: observed.to_owned(),
                    expected: crate::RNG_CONTRACT_ID,
                });
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NumericalMode {
    Deterministic,
    Tf32Off,
    Tf32On,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Precision {
    Fp32,
    Bf16,
    Fp16,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RngContractIdentity {
    pub physical: String,
    pub policy: String,
    pub redetermination: String,
    pub search: String,
    pub tie_break: String,
}

impl RngContractIdentity {
    fn validate(&self) -> Result<(), PolicyIdentityError> {
        for (name, value) in [
            ("rng_contracts.physical", self.physical.as_str()),
            ("rng_contracts.policy", self.policy.as_str()),
            (
                "rng_contracts.redetermination",
                self.redetermination.as_str(),
            ),
            ("rng_contracts.search", self.search.as_str()),
            ("rng_contracts.tie_break", self.tie_break.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(PolicyIdentityError::EmptyField(name));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FailureDisposition {
    RecordIncompleteNoLabel,
    RejectLaunch,
    Forbidden,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FailureBehavior {
    pub timeout: FailureDisposition,
    pub incomplete_unit: FailureDisposition,
    pub oom: FailureDisposition,
    pub fallback: FailureDisposition,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ForbiddenCapabilities {
    pub table_total_utility: bool,
    pub table_native_q: bool,
    pub true_hidden_peeking: bool,
    pub model_fallback: bool,
}

impl ForbiddenCapabilities {
    fn validate(&self) -> Result<(), PolicyIdentityError> {
        if self.table_total_utility {
            return Err(PolicyIdentityError::ForbiddenCapability(
                "table_total_utility",
            ));
        }
        if self.table_native_q {
            return Err(PolicyIdentityError::ForbiddenCapability("table_native_q"));
        }
        if self.true_hidden_peeking {
            return Err(PolicyIdentityError::ForbiddenCapability(
                "true_hidden_peeking",
            ));
        }
        if self.model_fallback {
            return Err(PolicyIdentityError::ForbiddenCapability("model_fallback"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum PolicyKind {
    #[serde(rename = "B_k")]
    Base,
    #[serde(rename = "pi_L")]
    LowFidelity,
    #[serde(rename = "W_k")]
    Shadow,
    #[serde(rename = "M_(k+1)")]
    Distilled,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PolicyIdentityWire {
    schema_id: String,
    policy_kind: PolicyKind,
    fields: PolicyIdentityFields,
}

pub trait FrozenPolicyIdentity: Clone + Serialize + DeserializeOwned {
    fn fields(&self) -> &PolicyIdentityFields;

    fn identity_sha256(&self) -> Sha256Digest {
        // Hash a recursively key-sorted JSON value, matching Python's
        // `sort_keys=True` canonical encoder. Hashing the Rust struct directly
        // would bind declaration order and silently disagree cross-language.
        let value = serde_json::to_value(self)
            .expect("serializing a validated in-memory policy identity cannot fail");
        let bytes = serde_json::to_vec(&value)
            .expect("serializing canonical policy identity JSON cannot fail");
        Sha256Digest::of_bytes(&bytes)
    }
}

macro_rules! define_policy_identity {
    ($name:ident, $kind:expr) => {
        #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
        #[serde(try_from = "PolicyIdentityWire", into = "PolicyIdentityWire")]
        pub struct $name(PolicyIdentityFields);

        impl $name {
            pub fn new(fields: PolicyIdentityFields) -> Result<Self, PolicyIdentityError> {
                fields.validate()?;
                Ok(Self(fields))
            }

            pub fn fields(&self) -> &PolicyIdentityFields {
                &self.0
            }

            pub fn identity_sha256(&self) -> Sha256Digest {
                <Self as FrozenPolicyIdentity>::identity_sha256(self)
            }
        }

        impl FrozenPolicyIdentity for $name {
            fn fields(&self) -> &PolicyIdentityFields {
                &self.0
            }
        }

        impl TryFrom<PolicyIdentityWire> for $name {
            type Error = PolicyIdentityError;

            fn try_from(value: PolicyIdentityWire) -> Result<Self, Self::Error> {
                if value.schema_id != POLICY_IDENTITY_SCHEMA_ID {
                    return Err(PolicyIdentityError::WrongSchema(value.schema_id));
                }
                if value.policy_kind != $kind {
                    return Err(PolicyIdentityError::WrongPolicyKind);
                }
                Self::new(value.fields)
            }
        }

        impl From<$name> for PolicyIdentityWire {
            fn from(value: $name) -> Self {
                Self {
                    schema_id: POLICY_IDENTITY_SCHEMA_ID.to_owned(),
                    policy_kind: $kind,
                    fields: value.0,
                }
            }
        }
    };
}

define_policy_identity!(BkIdentity, PolicyKind::Base);
define_policy_identity!(PiLIdentity, PolicyKind::LowFidelity);
define_policy_identity!(WkIdentity, PolicyKind::Shadow);
define_policy_identity!(MNextIdentity, PolicyKind::Distilled);

#[derive(Debug, Error)]
pub enum PolicyIdentityError {
    #[error("identity field {0} must not be empty")]
    EmptyField(&'static str),
    #[error("forbidden Rival capability enabled: {0}")]
    ForbiddenCapability(&'static str),
    #[error("model fallback behavior must be explicitly forbidden")]
    FallbackMustBeForbidden,
    #[error("wrong policy identity schema: {0}")]
    WrongSchema(String),
    #[error("policy kind cannot be substituted for this identity type")]
    WrongPolicyKind,
    #[error("CPU harness contract {field} is {observed:?}; expected {expected:?}")]
    IncompatibleHarnessContract {
        field: &'static str,
        observed: String,
        expected: &'static str,
    },
    #[error("CPU reference harness requires deterministic fp32 {0}")]
    IncompatibleHarnessExecution(&'static str),
    #[error(transparent)]
    Ruleset(#[from] crate::RulesetIdentityError),
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::*;

    fn digest(tag: &str) -> Sha256Digest {
        Sha256Digest::of_bytes(tag.as_bytes())
    }

    fn fields() -> PolicyIdentityFields {
        PolicyIdentityFields {
            ruleset: ResearchRulesetIdentity::canonical(),
            source_revision: "0123456789abcdef".to_owned(),
            source_digest: digest("source"),
            executable_sha256: digest("exe"),
            model_manifest_sha256: digest("manifest"),
            checkpoint_sha256: digest("checkpoint"),
            weights_sha256: digest("weights"),
            bridge_protocol: "bridge.v1".to_owned(),
            tensor_schema: "tensor.v4".to_owned(),
            numerical_mode: NumericalMode::Deterministic,
            precision: Precision::Fp32,
            gumbel_config_sha256: digest("gumbel"),
            search_config_sha256: digest("search"),
            refresh_config_sha256: digest("refresh"),
            exact_endgame_config_sha256: digest("endgame"),
            action_content_id_version: crate::ACTION_CONTENT_ID_VERSION.to_owned(),
            rules_action_occurrence_id_version: crate::ROOT_ACTION_OCCURRENCE_ID_VERSION.to_owned(),
            candidate_action_occurrence_id_version: crate::CANDIDATE_ACTION_OCCURRENCE_ID_VERSION
                .to_owned(),
            rules_menu_hash_version: crate::RULES_MENU_HASH_VERSION.to_owned(),
            incumbent_menu_hash_version: crate::INCUMBENT_MENU_HASH_VERSION.to_owned(),
            rng_contracts: RngContractIdentity {
                physical: "physical.v1".to_owned(),
                policy: "policy.v1".to_owned(),
                redetermination: "redetermination.v1".to_owned(),
                search: "search.v1".to_owned(),
                tie_break: "tie.v1".to_owned(),
            },
            public_observation_schema: "public_obs.v1".to_owned(),
            policy_memory_schema: "seat_memory.v1".to_owned(),
            failure_behavior: FailureBehavior {
                timeout: FailureDisposition::RecordIncompleteNoLabel,
                incomplete_unit: FailureDisposition::RecordIncompleteNoLabel,
                oom: FailureDisposition::RecordIncompleteNoLabel,
                fallback: FailureDisposition::Forbidden,
            },
            compiler_identity: "compiler.v1".to_owned(),
            simulator_identity: "simulator.v1".to_owned(),
            sampler_identity: "sampler.v1".to_owned(),
            candidate_generator_identity: "candidate.v1".to_owned(),
            forbidden_capabilities: ForbiddenCapabilities {
                table_total_utility: false,
                table_native_q: false,
                true_hidden_peeking: false,
                model_fallback: false,
            },
        }
    }

    #[test]
    fn policy_types_are_wire_incompatible() {
        let base = BkIdentity::new(fields()).unwrap();
        let bytes = serde_json::to_vec(&base).unwrap();
        assert!(serde_json::from_slice::<BkIdentity>(&bytes).is_ok());
        assert!(serde_json::from_slice::<PiLIdentity>(&bytes).is_err());
        assert!(serde_json::from_slice::<WkIdentity>(&bytes).is_err());
        assert!(serde_json::from_slice::<MNextIdentity>(&bytes).is_err());
    }

    #[test]
    fn forbidden_behavior_is_present_and_false_not_defaulted() {
        let mut value = serde_json::to_value(BkIdentity::new(fields()).unwrap()).unwrap();
        value["fields"]["forbidden_capabilities"]["table_total_utility"] = Value::Bool(true);
        assert!(serde_json::from_value::<BkIdentity>(value).is_err());

        let mut missing = serde_json::to_value(BkIdentity::new(fields()).unwrap()).unwrap();
        missing["fields"]["forbidden_capabilities"]
            .as_object_mut()
            .unwrap()
            .remove("table_total_utility");
        assert!(serde_json::from_value::<BkIdentity>(missing).is_err());
    }

    #[test]
    fn all_bound_leaf_fields_are_required() {
        let original = serde_json::to_value(BkIdentity::new(fields()).unwrap()).unwrap();
        let mut paths = Vec::new();
        collect_leaf_paths(&original, &mut Vec::new(), &mut paths);
        assert!(
            paths.len() >= 40,
            "unexpectedly shallow identity: {}",
            paths.len()
        );
        for path in paths {
            let mut changed = original.clone();
            remove_at_path(&mut changed, &path);
            assert!(
                serde_json::from_value::<BkIdentity>(changed).is_err(),
                "missing field unexpectedly accepted: {}",
                path.join(".")
            );
        }
    }

    #[test]
    fn any_valid_leaf_perturbation_changes_identity_or_rejects() {
        let identity = BkIdentity::new(fields()).unwrap();
        let original_hash = identity.identity_sha256();
        let original = serde_json::to_value(identity).unwrap();
        let mut paths = Vec::new();
        collect_leaf_paths(&original, &mut Vec::new(), &mut paths);
        for path in paths {
            let mut changed = original.clone();
            perturb_at_path(&mut changed, &path);
            if let Ok(identity) = serde_json::from_value::<BkIdentity>(changed) {
                assert_ne!(
                    identity.identity_sha256(),
                    original_hash,
                    "field was not identity-bound: {}",
                    path.join(".")
                );
            }
        }
    }

    #[test]
    fn unknown_fields_fail_closed_at_every_structured_layer() {
        let mut outer = serde_json::to_value(BkIdentity::new(fields()).unwrap()).unwrap();
        outer["unknown"] = json!(1);
        assert!(serde_json::from_value::<BkIdentity>(outer).is_err());

        let mut nested = serde_json::to_value(BkIdentity::new(fields()).unwrap()).unwrap();
        nested["fields"]["rng_contracts"]["unknown"] = json!(1);
        assert!(serde_json::from_value::<BkIdentity>(nested).is_err());
    }

    #[test]
    fn rust_and_python_share_one_policy_identity_and_digest_golden() {
        let fixture = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../cascadiav3/tests/fixtures/rival/policy_identity_bk_v1.json"
        ));
        let identity: BkIdentity = serde_json::from_str(fixture).unwrap();
        assert_eq!(
            identity.identity_sha256().as_str(),
            "sha256:323838bfcbd94446f958f90cc268cef6cfaa9806d58ba429492937727b39fbf1"
        );
    }

    fn collect_leaf_paths(value: &Value, prefix: &mut Vec<String>, out: &mut Vec<Vec<String>>) {
        match value {
            Value::Object(map) => {
                for (key, child) in map {
                    prefix.push(key.clone());
                    collect_leaf_paths(child, prefix, out);
                    prefix.pop();
                }
            }
            _ => out.push(prefix.clone()),
        }
    }

    fn remove_at_path(value: &mut Value, path: &[String]) {
        let (last, parents) = path.split_last().unwrap();
        let mut cursor = value;
        for key in parents {
            cursor = cursor.get_mut(key).unwrap();
        }
        cursor.as_object_mut().unwrap().remove(last);
    }

    fn perturb_at_path(value: &mut Value, path: &[String]) {
        let mut cursor = value;
        for key in path {
            cursor = cursor.get_mut(key).unwrap();
        }
        *cursor = match cursor {
            Value::String(text) if text.starts_with("sha256:") => {
                Value::String(format!("sha256:{}", "f".repeat(64)))
            }
            Value::String(text) => Value::String(format!("{text}-perturbed")),
            Value::Bool(value) => Value::Bool(!*value),
            Value::Number(number) => json!(number.as_i64().unwrap_or_default() + 1),
            _ => unreachable!("identity leaves are scalar"),
        };
    }
}
