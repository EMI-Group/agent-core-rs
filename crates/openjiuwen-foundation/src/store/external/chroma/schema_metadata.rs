//! The serialized collection schema inside Chroma collection metadata.
//!
//! Chroma has no typed schema, so both implementations of the reference store
//! keep the whole [`CollectionSchema`] in the collection's metadata. The Python
//! `ChromaVectorStore` (`create_collection`/`get_schema` in
//! `openjiuwen/core/foundation/store/vector/chroma_vector_store.py`) writes the
//! same `CollectionSchema.to_dict()` JSON twice, under the keys `schema` and
//! `fields`, and reads them back in that order. This backend historically used
//! a single key of its own (`openjiuwen:schema` by default), so collections
//! written by one implementation were unreadable by the other.
//!
//! Both directions are now interoperable:
//!
//! - **Write** — `insert` stores the schema in Python's shape (the very
//!   `to_dict()` layout: `{fields: [{name, type, is_primary, ...}],
//!   description, enable_dynamic_field}`, with the Python `VectorDataType`
//!   names such as `"FLOAT_VECTOR"`) under `schema` and `fields`, and *also*
//!   stores this backend's own serde JSON under the configured key, so readers
//!   of either generation keep working.
//! - **Read** — `decode` prefers `schema`, then `fields`, then the configured
//!   (legacy) key, and accepts either shape: the Python `type` key with
//!   upper-case type names, Rust's `dtype` key with snake-case names, numbers
//!   written as strings, and a bare field array.
//!
//! A collection with none of those keys was not created through a
//! `ChromaVectorStore`, and a value none of them parses is corrupt; both are
//! reported as errors naming every accepted key.
use serde_json::{json, Map, Value};

use openjiuwen_common::error::{Error, Result};

use crate::store::vector::{CollectionSchema, FieldSchema, VectorDataType};

/// Metadata key under which the Python store writes the serialized schema.
pub(super) const PYTHON_SCHEMA_KEY: &str = "schema";

/// Metadata key under which the Python store writes a second copy, used as the
/// fallback when `schema` is absent.
pub(super) const PYTHON_FIELDS_KEY: &str = "fields";

/// Metadata key this backend wrote before it understood the Python layout.
pub(super) const LEGACY_SCHEMA_KEY: &str = "openjiuwen:schema";

/// Every key `decode` accepts, in precedence order, for its error messages.
pub(super) fn accepted_keys(legacy_key: &str) -> Vec<String> {
    let mut keys = vec![PYTHON_SCHEMA_KEY.to_string(), PYTHON_FIELDS_KEY.to_string()];
    if !keys.iter().any(|key| key == legacy_key) {
        keys.push(legacy_key.to_string());
    }
    keys
}

/// Stores `schema` in `metadata` under the Python keys and the legacy key.
///
/// # Errors
///
/// Propagates a serialization failure of the schema, which can only happen if a
/// `default_value` is not representable as JSON.
pub(super) fn insert(
    metadata: &mut Map<String, Value>,
    legacy_key: &str,
    schema: &CollectionSchema,
) -> Result<()> {
    let python = python_shape(schema);
    for key in [PYTHON_SCHEMA_KEY, PYTHON_FIELDS_KEY] {
        metadata.insert(
            key.to_string(),
            Value::String(serde_json::to_string(&python).map_err(Error::external)?),
        );
    }
    metadata.insert(
        legacy_key.to_string(),
        Value::String(serde_json::to_string(schema).map_err(Error::external)?),
    );
    Ok(())
}

/// Reads a schema out of a collection's metadata, trying every accepted key.
///
/// # Errors
///
/// [`Error::Unsupported`](openjiuwen_common::error::Error) when the metadata
/// carries none of the accepted keys, and
/// [`Error::Validation`](openjiuwen_common::error::Error) when the value found
/// is not a readable schema.
pub(super) fn decode(
    legacy_key: &str,
    collection: &str,
    metadata: Option<&Map<String, Value>>,
) -> Result<CollectionSchema> {
    let empty = Map::new();
    let metadata = metadata.unwrap_or(&empty);
    for key in accepted_keys(legacy_key) {
        let Some(stored) = metadata.get(&key) else {
            continue;
        };
        return read_value(&key, collection, stored);
    }
    Err(Error::unsupported(format!(
        "collection {collection:?} was not created through ChromaVectorStore: its metadata has \
         neither the Python keys \"{PYTHON_SCHEMA_KEY}\"/\"{PYTHON_FIELDS_KEY}\" nor the \
         legacy {legacy_key:?} entry"
    )))
}

/// Parses one metadata value into a schema, naming the key in every error.
fn read_value(key: &str, collection: &str, stored: &Value) -> Result<CollectionSchema> {
    let unreadable = |detail: String| {
        Error::validation(format!(
            "collection {collection:?} stores an unreadable schema in metadata {key:?} \
             (accepted keys: \"{PYTHON_SCHEMA_KEY}\", \"{PYTHON_FIELDS_KEY}\", and this \
             backend's legacy key): {detail}"
        ))
    };
    let parsed = match stored {
        Value::String(raw) => serde_json::from_str::<Value>(raw)
            .map_err(|err| unreadable(format!("its JSON is invalid: {err}")))?,
        Value::Object(_) | Value::Array(_) => stored.clone(),
        other => {
            return Err(unreadable(format!(
                "expected a JSON string or object, got {other}"
            )))
        }
    };
    let (fields, entries) = match &parsed {
        Value::Array(items) => (Value::Array(items.clone()), Map::new()),
        Value::Object(entries) => (
            entries
                .get("fields")
                .cloned()
                .ok_or_else(|| unreadable("its schema has no `fields` member".to_string()))?,
            entries.clone(),
        ),
        other => return Err(unreadable(format!("expected a schema object, got {other}"))),
    };
    let fields = fields
        .as_array()
        .ok_or_else(|| unreadable(format!("its `fields` member is not an array: {fields}")))?;
    let normalized_fields = fields
        .iter()
        .enumerate()
        .map(|(position, field)| normalize_field(field, position, &unreadable))
        .collect::<Result<Vec<_>>>()?;
    let mut normalized = Map::new();
    normalized.insert("fields".to_string(), Value::Array(normalized_fields));
    for key in ["description", "enable_dynamic_field"] {
        if let Some(value) = entries.get(key) {
            normalized.insert(key.to_string(), value.clone());
        }
    }
    serde_json::from_value::<CollectionSchema>(Value::Object(normalized))
        .map_err(|err| unreadable(err.to_string()))
}

/// Normalizes one Python-shaped field entry into this crate's serde shape.
fn normalize_field<F>(field: &Value, position: usize, unreadable: &F) -> Result<Value>
where
    F: Fn(String) -> Error,
{
    let entries = match field {
        Value::Object(entries) => entries,
        other => {
            return Err(unreadable(format!(
                "field at position {position} is not an object: {other}"
            )))
        }
    };
    let mut normalized = Map::new();
    let name = match entries.get("name").and_then(Value::as_str) {
        Some(name) if !name.is_empty() => name.to_string(),
        _ => {
            return Err(unreadable(format!(
                "field at position {position} has no string `name`"
            )))
        }
    };
    let raw_dtype = entries
        .get("dtype")
        .or_else(|| entries.get("type"))
        .and_then(Value::as_str)
        .ok_or_else(|| unreadable(format!("field {name:?} has no `type`/`dtype`")))?;
    let dtype = normalize_dtype(raw_dtype).ok_or_else(|| {
        unreadable(format!(
            "field {name:?} has the unknown data type {raw_dtype:?}"
        ))
    })?;
    normalized.insert("name".to_string(), Value::String(name.clone()));
    normalized.insert(
        "dtype".to_string(),
        Value::String(snake_case(dtype).to_string()),
    );
    for flag in ["is_primary", "auto_id"] {
        if let Some(value) = entries.get(flag) {
            normalized.insert(flag.to_string(), value.clone());
        }
    }
    for bound in ["max_length", "dim"] {
        if let Some(value) = entries.get(bound).and_then(as_number) {
            normalized.insert(bound.to_string(), value);
        }
    }
    if let Some(default) = entries.get("default_value") {
        normalized.insert("default_value".to_string(), default.clone());
    }
    if let Some(description) = entries.get("description") {
        normalized.insert("description".to_string(), description.clone());
    }
    Ok(Value::Object(normalized))
}

/// Reads a JSON number written either as a number or as a numeric string.
fn as_number(value: &Value) -> Option<Value> {
    match value {
        Value::Number(_) => Some(value.clone()),
        Value::String(text) => text.trim().parse::<u64>().ok().map(|parsed| json!(parsed)),
        _ => None,
    }
}

/// Maps a Python or Rust data-type spelling onto [`VectorDataType`].
///
/// Python names are upper-case (`"FLOAT_VECTOR"`), this crate's names are
/// snake-case (`"float_vector"`), and Python additionally declares the 8- and
/// 16-bit integers this crate has no variant for: they widen to `Int64`, which
/// Chroma metadata (JSON scalars) cannot distinguish from a 64-bit field.
fn normalize_dtype(raw: &str) -> Option<VectorDataType> {
    let folded = raw.trim().to_ascii_uppercase().replace(['-', ' '], "_");
    Some(match folded.as_str() {
        "VARCHAR" | "STRING" | "TEXT" => VectorDataType::Varchar,
        "FLOAT_VECTOR" | "FLOATVECTOR" => VectorDataType::FloatVector,
        "INT64" | "INTEGER" | "BIGINT" | "INT8" | "INT16" => VectorDataType::Int64,
        "INT32" => VectorDataType::Int32,
        "FLOAT" | "FLOAT32" => VectorDataType::Float,
        "DOUBLE" | "FLOAT64" => VectorDataType::Double,
        "BOOL" | "BOOLEAN" => VectorDataType::Bool,
        "JSON" => VectorDataType::Json,
        "ARRAY" => VectorDataType::Array,
        _ => return None,
    })
}

/// The snake-case name of `dtype`, as [`VectorDataType`]'s serde form spells it.
fn snake_case(dtype: VectorDataType) -> &'static str {
    match dtype {
        VectorDataType::Varchar => "varchar",
        VectorDataType::FloatVector => "float_vector",
        VectorDataType::Int64 => "int64",
        VectorDataType::Int32 => "int32",
        VectorDataType::Float => "float",
        VectorDataType::Double => "double",
        VectorDataType::Bool => "bool",
        VectorDataType::Json => "json",
        VectorDataType::Array => "array",
    }
}

/// The Python `VectorDataType` name of `dtype`.
fn python_name(dtype: VectorDataType) -> &'static str {
    match dtype {
        VectorDataType::Varchar => "VARCHAR",
        VectorDataType::FloatVector => "FLOAT_VECTOR",
        VectorDataType::Int64 => "INT64",
        VectorDataType::Int32 => "INT32",
        VectorDataType::Float => "FLOAT",
        VectorDataType::Double => "DOUBLE",
        VectorDataType::Bool => "BOOL",
        VectorDataType::Json => "JSON",
        VectorDataType::Array => "ARRAY",
    }
}

/// Renders `schema` in the Python `CollectionSchema.to_dict()` layout.
fn python_shape(schema: &CollectionSchema) -> Value {
    json!({
        "fields": schema
            .fields
            .iter()
            .map(python_field)
            .collect::<Vec<Value>>(),
        "description": schema.description,
        "enable_dynamic_field": schema.enable_dynamic_field,
    })
}

/// Renders one field in the Python `FieldSchema.to_dict()` layout.
fn python_field(field: &FieldSchema) -> Value {
    let mut entry = Map::new();
    entry.insert("name".to_string(), json!(field.name));
    entry.insert("type".to_string(), json!(python_name(field.dtype)));
    if field.is_primary {
        entry.insert("is_primary".to_string(), json!(true));
    }
    if field.auto_id {
        entry.insert("auto_id".to_string(), json!(true));
    }
    if let Some(max_length) = field.max_length {
        entry.insert("max_length".to_string(), json!(max_length));
    }
    if let Some(dim) = field.dim {
        entry.insert("dim".to_string(), json!(dim));
    }
    if let Some(default) = &field.default_value {
        entry.insert("default_value".to_string(), default.clone());
    }
    Value::Object(entry)
}
