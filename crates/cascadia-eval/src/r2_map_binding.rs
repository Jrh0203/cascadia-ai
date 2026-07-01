//! Immutable W0 implementation identity carried by every R2-MAP benchmark.
//!
//! Benchmark workers never accept operator-selected protocol hashes. The
//! initializer derives this object from the append-only W0 v1.1 registration,
//! and every contract, receipt, shard, and report echoes it exactly.

use std::collections::BTreeSet;

use cascadia_data::R2MapProtocolIdentity;
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const R2_MAP_IMPLEMENTATION_BINDING_SCHEMA_ID: &str =
    "cascadia.r2-map.implementation-binding.v1.1";
pub const R2_MAP_IMPLEMENTATION_CONTRACT_REVISION: &str = "sequential-public-market-v1.1";
pub const R2_MAP_OPEN_REFERENCE_SEED_DOMAIN_V1_1: &str = "r2-map-open-reference-performance-100-v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct R2MapImplementationBinding {
    pub schema_id: String,
    pub contract_revision: String,
    pub w0_registration_sha256: String,
    pub reference_manifest_sha256: String,
    pub maximum_width_panel_sha256: String,
    pub replay_pinecone_panel_sha256: String,
    pub source_bundle_sha256: String,
    pub serving_protocol_schema_sha256: String,
    pub market_action_schema_blake3: String,
    pub request_schema_blake3: String,
    pub response_schema_blake3: String,
    pub protocol_fixture_canonical_blake3: String,
    pub protocol_fixture_file_blake3: String,
    pub model_schema_sha256: String,
    pub open_reference_seed_domain_id: String,
    pub protocols: R2MapProtocolIdentity,
}

impl R2MapImplementationBinding {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        w0_registration_sha256: String,
        reference_manifest_sha256: String,
        maximum_width_panel_sha256: String,
        replay_pinecone_panel_sha256: String,
        source_bundle_sha256: String,
        serving_protocol_schema_sha256: String,
        market_action_schema_blake3: String,
        request_schema_blake3: String,
        response_schema_blake3: String,
        protocol_fixture_canonical_blake3: String,
        protocol_fixture_file_blake3: String,
        model_schema_sha256: String,
        open_reference_seed_domain_id: String,
    ) -> Result<Self, R2MapBindingError> {
        let protocols = R2MapProtocolIdentity {
            collector_hash: decode_digest(&replay_pinecone_panel_sha256)?,
            source_hash: decode_digest(&source_bundle_sha256)?,
            serving_protocol_hash: decode_digest(&serving_protocol_schema_sha256)?,
        };
        let binding = Self {
            schema_id: R2_MAP_IMPLEMENTATION_BINDING_SCHEMA_ID.to_owned(),
            contract_revision: R2_MAP_IMPLEMENTATION_CONTRACT_REVISION.to_owned(),
            w0_registration_sha256,
            reference_manifest_sha256,
            maximum_width_panel_sha256,
            replay_pinecone_panel_sha256,
            source_bundle_sha256,
            serving_protocol_schema_sha256,
            market_action_schema_blake3,
            request_schema_blake3,
            response_schema_blake3,
            protocol_fixture_canonical_blake3,
            protocol_fixture_file_blake3,
            model_schema_sha256,
            open_reference_seed_domain_id,
            protocols,
        };
        binding.validate()?;
        Ok(binding)
    }

    pub fn validate(&self) -> Result<(), R2MapBindingError> {
        if self.schema_id != R2_MAP_IMPLEMENTATION_BINDING_SCHEMA_ID
            || self.contract_revision != R2_MAP_IMPLEMENTATION_CONTRACT_REVISION
            || self.open_reference_seed_domain_id != R2_MAP_OPEN_REFERENCE_SEED_DOMAIN_V1_1
        {
            return Err(R2MapBindingError::Contract);
        }
        let digests = [
            &self.w0_registration_sha256,
            &self.reference_manifest_sha256,
            &self.maximum_width_panel_sha256,
            &self.replay_pinecone_panel_sha256,
            &self.source_bundle_sha256,
            &self.serving_protocol_schema_sha256,
            &self.market_action_schema_blake3,
            &self.request_schema_blake3,
            &self.response_schema_blake3,
            &self.protocol_fixture_canonical_blake3,
            &self.protocol_fixture_file_blake3,
            &self.model_schema_sha256,
        ];
        for digest in &digests {
            decode_digest(digest)?;
        }
        if digests.iter().copied().collect::<BTreeSet<_>>().len() != digests.len() {
            return Err(R2MapBindingError::CrossPanelSubstitution);
        }
        let expected = R2MapProtocolIdentity {
            collector_hash: decode_digest(&self.replay_pinecone_panel_sha256)?,
            source_hash: decode_digest(&self.source_bundle_sha256)?,
            serving_protocol_hash: decode_digest(&self.serving_protocol_schema_sha256)?,
        };
        if self.protocols != expected {
            return Err(R2MapBindingError::ProtocolDrift);
        }
        self.protocols
            .validate()
            .map_err(|_| R2MapBindingError::ProtocolDrift)
    }
}

fn decode_digest(value: &str) -> Result<[u8; 32], R2MapBindingError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(R2MapBindingError::Digest);
    }
    let mut decoded = [0u8; 32];
    for (index, output) in decoded.iter_mut().enumerate() {
        *output = u8::from_str_radix(&value[index * 2..index * 2 + 2], 16)
            .map_err(|_| R2MapBindingError::Digest)?;
    }
    if decoded == [0; 32] {
        return Err(R2MapBindingError::Digest);
    }
    Ok(decoded)
}

#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum R2MapBindingError {
    #[error("R2-MAP implementation binding has a malformed or zero 256-bit digest")]
    Digest,
    #[error("R2-MAP implementation binding schema, revision, or seed domain drifted")]
    Contract,
    #[error("R2-MAP protocol hashes do not match the registered panel/source identities")]
    ProtocolDrift,
    #[error("R2-MAP implementation binding aliases identities from distinct contracts")]
    CrossPanelSubstitution,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn binding() -> R2MapImplementationBinding {
        R2MapImplementationBinding::new(
            "11".repeat(32),
            "22".repeat(32),
            "33".repeat(32),
            "44".repeat(32),
            "55".repeat(32),
            "66".repeat(32),
            "88".repeat(32),
            "99".repeat(32),
            "aa".repeat(32),
            "bb".repeat(32),
            "cc".repeat(32),
            "77".repeat(32),
            R2_MAP_OPEN_REFERENCE_SEED_DOMAIN_V1_1.to_owned(),
        )
        .unwrap()
    }

    #[test]
    fn protocol_identity_is_derived_from_registered_fields() {
        let value = binding();
        assert_eq!(value.protocols.collector_hash, [0x44; 32]);
        assert_eq!(value.protocols.source_hash, [0x55; 32]);
        assert_eq!(value.protocols.serving_protocol_hash, [0x66; 32]);
        value.validate().unwrap();
    }

    #[test]
    fn tamper_and_cross_panel_substitution_fail_closed() {
        let mut value = binding();
        value.protocols.collector_hash = value.protocols.source_hash;
        assert_eq!(value.validate(), Err(R2MapBindingError::ProtocolDrift));

        let mut value = binding();
        value.maximum_width_panel_sha256 = value.replay_pinecone_panel_sha256.clone();
        assert_eq!(
            value.validate(),
            Err(R2MapBindingError::CrossPanelSubstitution)
        );

        let mut value = binding();
        value.maximum_width_panel_sha256 = "GG".repeat(32);
        assert_eq!(value.validate(), Err(R2MapBindingError::Digest));
    }
}
