use std::{collections::BTreeSet, fmt, io::Read};

use serde::de::{
    self, DeserializeOwned, DeserializeSeed, Deserializer, MapAccess, SeqAccess, Visitor,
};
use serde_json::{Map, Number, Value};

pub(crate) fn from_reader<T: DeserializeOwned>(reader: impl Read) -> serde_json::Result<T> {
    let mut deserializer = serde_json::Deserializer::from_reader(reader);
    let value = StrictValueSeed.deserialize(&mut deserializer)?;
    deserializer.end()?;
    serde_json::from_value(value)
}

pub(crate) fn from_str<T: DeserializeOwned>(input: &str) -> serde_json::Result<T> {
    let mut deserializer = serde_json::Deserializer::from_str(input);
    let value = StrictValueSeed.deserialize(&mut deserializer)?;
    deserializer.end()?;
    serde_json::from_value(value)
}

struct StrictValueSeed;

impl<'de> DeserializeSeed<'de> for StrictValueSeed {
    type Value = Value;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(StrictValueVisitor)
    }
}

struct StrictValueVisitor;

impl<'de> Visitor<'de> for StrictValueVisitor {
    type Value = Value;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a JSON value without duplicate object keys")
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E> {
        Ok(Value::Null)
    }

    fn visit_none<E>(self) -> Result<Self::Value, E> {
        Ok(Value::Null)
    }

    fn visit_some<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        StrictValueSeed.deserialize(deserializer)
    }

    fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E> {
        Ok(Value::Bool(value))
    }

    fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E> {
        Ok(Value::Number(Number::from(value)))
    }

    fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E> {
        Ok(Value::Number(Number::from(value)))
    }

    fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Number::from_f64(value)
            .map(Value::Number)
            .ok_or_else(|| E::custom("JSON number is not finite"))
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E> {
        Ok(Value::String(value.to_owned()))
    }

    fn visit_string<E>(self, value: String) -> Result<Self::Value, E> {
        Ok(Value::String(value))
    }

    fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut values = Vec::with_capacity(sequence.size_hint().unwrap_or(0));
        while let Some(value) = sequence.next_element_seed(StrictValueSeed)? {
            values.push(value);
        }
        Ok(Value::Array(values))
    }

    fn visit_map<A>(self, mut object: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut keys = BTreeSet::new();
        let mut values = Map::new();
        while let Some(key) = object.next_key::<String>()? {
            if !keys.insert(key.clone()) {
                return Err(de::Error::custom(format!("duplicate object key `{key}`")));
            }
            values.insert(key, object.next_value_seed(StrictValueSeed)?);
        }
        Ok(Value::Object(values))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_duplicate_keys_recursively() {
        let nested_object = from_str::<Value>(r#"{"outer":{"value":1,"value":2}}"#).unwrap_err();
        assert!(nested_object.to_string().contains("duplicate object key"));

        let nested_array = from_str::<Value>(r#"{"outer":[{"value":1,"value":2}]}"#).unwrap_err();
        assert!(nested_array.to_string().contains("duplicate object key"));
    }

    #[test]
    fn accepts_unique_keys_and_rejects_trailing_json() {
        let value = from_str::<Value>(r#"{"outer":[{"value":1},{"value":2}]}"#).unwrap();
        assert_eq!(value["outer"][1]["value"], 2);
        assert!(from_str::<Value>(r#"{"value":1}{"value":2}"#).is_err());
    }
}
