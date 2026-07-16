use std::{fmt, str::FromStr};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

const PREFIX: &str = "sha256:";

/// A lowercase, prefix-qualified SHA-256 digest.
///
/// Scientific manifests use the qualified wire form so a digest can never be
/// silently reinterpreted as another algorithm.
#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct Sha256Digest(String);

impl Sha256Digest {
    pub fn of_bytes(bytes: &[u8]) -> Self {
        let digest = Sha256::digest(bytes);
        Self(format!("{PREFIX}{}", encode_lower_hex(&digest)))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn hex(&self) -> &str {
        &self.0[PREFIX.len()..]
    }
}

impl fmt::Debug for Sha256Digest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("Sha256Digest")
            .field(&self.0)
            .finish()
    }
}

impl fmt::Display for Sha256Digest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl From<Sha256Digest> for String {
    fn from(value: Sha256Digest) -> Self {
        value.0
    }
}

impl TryFrom<String> for Sha256Digest {
    type Error = DigestParseError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        validate_sha256(&value)?;
        Ok(Self(value))
    }
}

impl FromStr for Sha256Digest {
    type Err = DigestParseError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        value.to_owned().try_into()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum DigestParseError {
    #[error("SHA-256 digest must start with 'sha256:'")]
    MissingPrefix,
    #[error("SHA-256 digest must contain exactly 64 hexadecimal digits")]
    WrongLength,
    #[error("SHA-256 digest must use lowercase hexadecimal digits")]
    NonCanonicalHex,
}

fn validate_sha256(value: &str) -> Result<(), DigestParseError> {
    let Some(hex) = value.strip_prefix(PREFIX) else {
        return Err(DigestParseError::MissingPrefix);
    };
    if hex.len() != 64 {
        return Err(DigestParseError::WrongLength);
    }
    if !hex
        .bytes()
        .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(DigestParseError::NonCanonicalHex);
    }
    Ok(())
}

pub(crate) fn encode_lower_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        encoded.push(HEX[(byte >> 4) as usize] as char);
        encoded.push(HEX[(byte & 0x0f) as usize] as char);
    }
    encoded
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn digest_has_locked_wire_format() {
        assert_eq!(
            Sha256Digest::of_bytes(b"abc").as_str(),
            "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn parser_rejects_ambiguous_or_noncanonical_forms() {
        assert_eq!(
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
                .parse::<Sha256Digest>(),
            Err(DigestParseError::MissingPrefix)
        );
        assert_eq!(
            format!("sha256:{}", "A".repeat(64)).parse::<Sha256Digest>(),
            Err(DigestParseError::NonCanonicalHex)
        );
        assert_eq!(
            "sha256:00".parse::<Sha256Digest>(),
            Err(DigestParseError::WrongLength)
        );
    }
}
