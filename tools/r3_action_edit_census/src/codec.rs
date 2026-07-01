use serde::{Serialize, de::DeserializeOwned};

use crate::{
    ActionEdit, PublicStateTrunk, R3Error, Result, SupplySnapshot,
    model::{R3_ACTION_EDIT_SCHEMA_VERSION, R3_STATE_TRUNK_SCHEMA_VERSION},
};

pub const STATE_TRUNK_MAGIC: &[u8; 8] = b"CSR3ST1\0";
pub const ACTION_EDIT_MAGIC: &[u8; 8] = b"CSR3AE1\0";
pub const STATE_TRUNK_SCHEMA_VERSION: u16 = R3_STATE_TRUNK_SCHEMA_VERSION;
pub const ACTION_EDIT_SCHEMA_VERSION: u16 = R3_ACTION_EDIT_SCHEMA_VERSION;

impl PublicStateTrunk {
    pub fn to_packed_bytes(&self) -> Result<Vec<u8>> {
        self.validate()?;
        let sparse = self.sparse.to_packed_bytes()?;
        let supply = postcard::to_allocvec(&self.supply)?;
        let mut bytes = Vec::with_capacity(18 + sparse.len() + supply.len());
        bytes.extend_from_slice(STATE_TRUNK_MAGIC);
        bytes.extend_from_slice(&STATE_TRUNK_SCHEMA_VERSION.to_le_bytes());
        write_blob(&mut bytes, &sparse)?;
        write_blob(&mut bytes, &supply)?;
        Ok(bytes)
    }

    pub fn from_packed_bytes(bytes: &[u8]) -> Result<Self> {
        let mut cursor = Cursor::new(bytes);
        if cursor.take_array::<8>()? != *STATE_TRUNK_MAGIC {
            return Err(R3Error::InvalidPackedMagic);
        }
        let schema = u16::from_le_bytes(cursor.take_array()?);
        if schema != STATE_TRUNK_SCHEMA_VERSION {
            return Err(R3Error::UnsupportedPackedSchema(schema));
        }
        let sparse_bytes = cursor.take_blob()?;
        let supply_bytes = cursor.take_blob()?;
        cursor.finish()?;
        let sparse = r2_sparse_entity_census::SparsePublicState::from_packed_bytes(sparse_bytes)?;
        let supply: SupplySnapshot = decode_postcard_exact(supply_bytes)?;
        let trunk = Self {
            schema_version: schema,
            sparse,
            supply,
        };
        trunk.validate()?;
        if trunk.to_packed_bytes()? != bytes {
            return Err(R3Error::NonCanonicalPacked(
                "state-trunk decode followed by encode changed bytes".to_owned(),
            ));
        }
        Ok(trunk)
    }

    pub fn canonical_hash(&self) -> Result<[u8; 32]> {
        Ok(*blake3::hash(&self.to_packed_bytes()?).as_bytes())
    }
}

impl ActionEdit {
    pub fn to_packed_bytes(&self) -> Result<Vec<u8>> {
        self.validate()?;
        let payload = postcard::to_allocvec(self)?;
        let mut bytes = Vec::with_capacity(14 + payload.len());
        bytes.extend_from_slice(ACTION_EDIT_MAGIC);
        bytes.extend_from_slice(&ACTION_EDIT_SCHEMA_VERSION.to_le_bytes());
        write_blob(&mut bytes, &payload)?;
        Ok(bytes)
    }

    pub fn from_packed_bytes(bytes: &[u8]) -> Result<Self> {
        let mut cursor = Cursor::new(bytes);
        if cursor.take_array::<8>()? != *ACTION_EDIT_MAGIC {
            return Err(R3Error::InvalidPackedMagic);
        }
        let schema = u16::from_le_bytes(cursor.take_array()?);
        if schema != ACTION_EDIT_SCHEMA_VERSION {
            return Err(R3Error::UnsupportedPackedSchema(schema));
        }
        let payload = cursor.take_blob()?;
        cursor.finish()?;
        let edit: Self = decode_postcard_exact(payload)?;
        edit.validate()?;
        if edit.schema_version != schema {
            return Err(R3Error::NonCanonicalPacked(
                "action-edit envelope and payload schemas disagree".to_owned(),
            ));
        }
        if edit.to_packed_bytes()? != bytes {
            return Err(R3Error::NonCanonicalPacked(
                "action-edit decode followed by encode changed bytes".to_owned(),
            ));
        }
        Ok(edit)
    }

    pub fn canonical_hash(&self) -> Result<[u8; 32]> {
        Ok(*blake3::hash(&self.to_packed_bytes()?).as_bytes())
    }
}

fn write_blob(output: &mut Vec<u8>, bytes: &[u8]) -> Result<()> {
    let len = u32::try_from(bytes.len())
        .map_err(|_| R3Error::Invariant("packed blob exceeds u32 length".to_owned()))?;
    output.extend_from_slice(&len.to_le_bytes());
    output.extend_from_slice(bytes);
    Ok(())
}

fn decode_postcard_exact<T: DeserializeOwned + Serialize>(bytes: &[u8]) -> Result<T> {
    let (value, remaining) = postcard::take_from_bytes(bytes)?;
    if !remaining.is_empty() {
        return Err(R3Error::TrailingPackedBytes(remaining.len()));
    }
    if postcard::to_allocvec(&value)? != bytes {
        return Err(R3Error::NonCanonicalPacked(
            "postcard payload is not canonical".to_owned(),
        ));
    }
    Ok(value)
}

struct Cursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Cursor<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn take_array<const N: usize>(&mut self) -> Result<[u8; N]> {
        let end = self
            .offset
            .checked_add(N)
            .ok_or(R3Error::UnexpectedPackedEnd)?;
        let value = self
            .bytes
            .get(self.offset..end)
            .ok_or(R3Error::UnexpectedPackedEnd)?
            .try_into()
            .expect("slice length was checked");
        self.offset = end;
        Ok(value)
    }

    fn take_blob(&mut self) -> Result<&'a [u8]> {
        let len = u32::from_le_bytes(self.take_array()?) as usize;
        let end = self
            .offset
            .checked_add(len)
            .ok_or(R3Error::UnexpectedPackedEnd)?;
        let value = self
            .bytes
            .get(self.offset..end)
            .ok_or(R3Error::UnexpectedPackedEnd)?;
        self.offset = end;
        Ok(value)
    }

    fn finish(self) -> Result<()> {
        let remaining = self.bytes.len() - self.offset;
        if remaining == 0 {
            Ok(())
        } else {
            Err(R3Error::TrailingPackedBytes(remaining))
        }
    }
}
