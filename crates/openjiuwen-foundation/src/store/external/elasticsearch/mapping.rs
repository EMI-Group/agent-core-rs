//! Mapping between a [`CollectionSchema`] and Elasticsearch index mappings.
//!
//! `create_index_body` derives the `properties` mapping of a schema and stores
//! the serialized schema under the reserved index-metadata key
//! `_meta.openjiuwen_schema`; `schema_from_index` prefers that marker and falls
//! back to `schema_from_properties`, which rebuilds a schema from
//! `mappings.properties` (that is how an index created outside this adapter is
//! read).
//!
//! Elasticsearch declares many more field types than [`VectorDataType`] has
//! variants, so `field_from_mapping` maps every type a schema field can
//! represent and rejects anything else with a typed
//! [`Error::Validation`](openjiuwen_common::error::Error) naming the field, its
//! dtype and the supported set: a foreign mapping is never silently collapsed
//! onto a default type.
//!
//! # What a mapping cannot carry
//!
//! A [`FieldSchema`] has no `index`/`doc_values`/`analyzer` members, so those
//! mapping options are dropped on both directions; `is_primary` and `auto_id`
//! are only recoverable from the `_meta.openjiuwen_schema` marker, and
//! `field_from_mapping` leaves `is_primary` to its caller, which is the only
//! place that sees the whole `properties` object.

use serde_json::{json, Map, Value};

use openjiuwen_common::error::{Error, Result};

use crate::store::document::ID_FIELD;
use crate::store::vector::{CollectionSchema, FieldSchema, Metric, VectorDataType};

/// Index-metadata key holding the serialized [`CollectionSchema`].
pub(super) const SCHEMA_META_KEY: &str = "openjiuwen_schema";

/// The Elasticsearch field types [`field_from_mapping`] understands, in the
/// order the "unsupported type" error lists them.
///
/// `alias` is deliberately absent: it resolves to another field and has no
/// [`VectorDataType`] of its own.
pub(super) const SUPPORTED_DTYPES: &[&str] = &[
    "keyword",
    "text",
    "constant_keyword",
    "wildcard",
    "ip",
    "binary",
    "date",
    "date_nanos",
    "integer",
    "short",
    "byte",
    "long",
    "unsigned_long",
    "float",
    "half_float",
    "scaled_float",
    "double",
    "boolean",
    "object",
    "nested",
    "flattened",
    "geo_point",
    "geo_shape",
    "dense_vector",
];

/// Maps a [`Metric`] onto the Elasticsearch `dense_vector` similarity name.
pub(crate) fn similarity_for(metric: Metric) -> &'static str {
    match metric {
        Metric::Cosine => "cosine",
        Metric::Dot => "dot_product",
        Metric::Euclidean => "l2_norm",
    }
}

/// The Elasticsearch mapping of one schema field, using `metric` as the
/// similarity of a `dense_vector` field.
///
/// # Errors
///
/// Returns [`Error::Validation`](openjiuwen_common::error::Error) for a vector
/// field without a positive `dim` and for a top-level `ARRAY` field, which an
/// Elasticsearch mapping cannot represent.
pub(crate) fn field_mapping_with(field: &FieldSchema, metric: Metric) -> Result<Value> {
    let mapping = match field.dtype {
        VectorDataType::FloatVector => {
            let dim = field.dim.filter(|dim| *dim > 0).ok_or_else(|| {
                Error::validation(format!(
                    "Elasticsearch mapping of vector field {:?} needs a positive `dim`, got {:?}",
                    field.name, field.dim
                ))
            })?;
            json!({
                "type": "dense_vector",
                "dims": dim,
                "index": true,
                "similarity": similarity_for(metric),
            })
        }
        VectorDataType::Varchar => match field.max_length {
            Some(max_length) => json!({"type": "keyword", "ignore_above": max_length}),
            None => json!({"type": "keyword"}),
        },
        VectorDataType::Int64 => json!({"type": "long"}),
        VectorDataType::Int32 => json!({"type": "integer"}),
        VectorDataType::Float => json!({"type": "float"}),
        VectorDataType::Double => json!({"type": "double"}),
        VectorDataType::Bool => json!({"type": "boolean"}),
        VectorDataType::Json => json!({"type": "object"}),
        VectorDataType::Array => {
            return Err(Error::validation(format!(
                "Elasticsearch mappings cannot represent a top-level ARRAY field; \
                 field {:?} must be a JSON or scalar field",
                field.name
            )));
        }
    };
    Ok(mapping)
}

/// The `properties` mapping of every field of `schema`, keyed by field name.
///
/// # Errors
///
/// Propagates [`field_mapping_with`]'s validation errors. The caller validates
/// the schema itself with [`CollectionSchema::validate`].
pub(crate) fn properties_for(schema: &CollectionSchema, metric: Metric) -> Result<Value> {
    let mut properties = Map::new();
    for field in &schema.fields {
        properties.insert(field.name.clone(), field_mapping_with(field, metric)?);
    }
    Ok(Value::Object(properties))
}

/// The body of the `PUT /{index}` that creates an index for `schema`.
///
/// The mapping holds the derived `properties` plus the serialized schema under
/// `_meta.openjiuwen_schema`, which is what makes [`schema_from_index`] restore
/// the exact [`CollectionSchema`] instead of a reconstructed approximation.
///
/// # Errors
///
/// Propagates [`properties_for`]'s validation errors.
pub(crate) fn create_index_body(schema: &CollectionSchema, metric: Metric) -> Result<Value> {
    let serialized = serde_json::to_string(schema).map_err(Error::external)?;
    let properties = properties_for(schema, metric)?;
    Ok(json!({
        "mappings": {
            "_meta": { (SCHEMA_META_KEY): serialized },
            "properties": properties,
        },
    }))
}

/// Reads a [`CollectionSchema`] back out of a `GET /{index}` response body.
///
/// Both response shapes are accepted: `{"<index>": {"mappings": {..}}}` (the
/// `get index` API) and a bare `{"mappings": {..}}`. The `_meta` marker written
/// by [`create_index_body`] wins; without it the schema is reconstructed from
/// `mappings.properties`. `Ok(None)` means the payload describes no mapping at
/// all (a missing index, or an index without properties).
///
/// # Errors
///
/// Returns [`Error::Validation`](openjiuwen_common::error::Error) when the
/// marker is not a readable schema or a property cannot be represented as a
/// [`FieldSchema`] (see [`field_from_mapping`]).
pub(crate) fn schema_from_index(payload: &Value) -> Result<Option<CollectionSchema>> {
    let Some(mappings) = mappings_of(payload) else {
        return Ok(None);
    };
    if let Some(stored) = mappings
        .get("_meta")
        .and_then(Value::as_object)
        .and_then(|meta| meta.get(SCHEMA_META_KEY))
    {
        let raw = stored.as_str().ok_or_else(|| {
            Error::validation(format!(
                "Elasticsearch index metadata `_meta.{SCHEMA_META_KEY}` must hold the \
                 serialized collection schema as a JSON string, found {stored}"
            ))
        })?;
        let schema = serde_json::from_str(raw).map_err(|err| {
            Error::validation(format!(
                "Elasticsearch index metadata `_meta.{SCHEMA_META_KEY}` holds an \
                 unreadable collection schema: {err}"
            ))
        })?;
        return Ok(Some(schema));
    }
    match mappings.get("properties").and_then(Value::as_object) {
        Some(properties) if !properties.is_empty() => {
            schema_from_properties(&Value::Object(properties.clone())).map(Some)
        }
        _ => Ok(None),
    }
}

/// The `mappings` object of an index-metadata payload, wrapped or bare.
fn mappings_of(payload: &Value) -> Option<&Map<String, Value>> {
    let object = payload.as_object()?;
    if let Some(mappings) = object.get("mappings").and_then(Value::as_object) {
        return Some(mappings);
    }
    let mut names: Vec<&String> = object.keys().collect();
    names.sort();
    names
        .into_iter()
        .find_map(|name| object.get(name)?.get("mappings")?.as_object())
}

/// Reconstructs a [`CollectionSchema`] from bare `mappings.properties`.
///
/// A field named [`ID_FIELD`] becomes the primary key; when the mapping has no
/// such field, a `VARCHAR` primary-key field named `id` is synthesized, so the
/// schema always validates with exactly one primary key.
///
/// # Errors
///
/// Returns [`Error::Validation`](openjiuwen_common::error::Error) when
/// `properties` is not an object, a property cannot be represented as a
/// [`FieldSchema`] (see [`field_from_mapping`]), or the `id` field cannot be a
/// primary key.
pub(crate) fn schema_from_properties(properties: &Value) -> Result<CollectionSchema> {
    let entries = properties.as_object().ok_or_else(|| {
        Error::validation(format!(
            "Elasticsearch index mappings `properties` must be an object, found {properties}"
        ))
    })?;
    let mut fields = Vec::with_capacity(entries.len() + 1);
    let mut has_id = false;
    for (name, mapping) in entries {
        let mut field = field_from_mapping(name, mapping)?;
        if name == ID_FIELD {
            if matches!(
                field.dtype,
                VectorDataType::FloatVector | VectorDataType::Array
            ) {
                return Err(Error::validation(format!(
                    "Elasticsearch index field {ID_FIELD:?} has type {:?} and cannot be the \
                     primary key",
                    field.dtype
                )));
            }
            field.is_primary = true;
            has_id = true;
        }
        fields.push(field);
    }
    if !has_id {
        fields.insert(
            0,
            FieldSchema {
                is_primary: true,
                ..FieldSchema::new(ID_FIELD, VectorDataType::Varchar)
            },
        );
    }
    Ok(CollectionSchema::new(fields))
}

/// Converts one `mappings.properties` entry into a [`FieldSchema`].
///
/// The mapping is exhaustive over [`SUPPORTED_DTYPES`]:
///
/// | Elasticsearch `type` | [`VectorDataType`] |
/// | --- | --- |
/// | `keyword`, `text`, `constant_keyword`, `wildcard` | `Varchar` (`ignore_above` → `max_length`) |
/// | `ip`, `binary`, `date`, `date_nanos` | `Varchar` (all four are strings in `_source`) |
/// | `integer`, `short`, `byte` | `Int32` |
/// | `long`, `unsigned_long` | `Int64` |
/// | `float`, `half_float`, `scaled_float` | `Float` |
/// | `double` | `Double` |
/// | `boolean` | `Bool` |
/// | `object`, `nested`, `flattened`, `geo_point`, `geo_shape` | `Json` |
/// | `dense_vector` | `FloatVector` (with `dims`) |
///
/// `is_primary` is the caller's ([`schema_from_properties`]) decision because
/// only it sees the whole mapping, and a schema field has no `index` member, so
/// `index: false` is not carried over.
///
/// # Errors
///
/// Returns [`Error::Validation`](openjiuwen_common::error::Error) naming the
/// field when the mapping is not an object, has no string `type`, declares a
/// type no [`VectorDataType`] can represent, is an `alias` (it resolves to
/// another field), or is a `dense_vector` without a positive `dims`.
fn field_from_mapping(name: &str, mapping: &Value) -> Result<FieldSchema> {
    let object = mapping.as_object().ok_or_else(|| {
        Error::validation(format!(
            "Elasticsearch index mapping of field {name:?} must be an object with a `type`, \
             found {mapping}"
        ))
    })?;
    let dtype_name = object.get("type").and_then(Value::as_str).ok_or_else(|| {
        Error::validation(format!(
            "Elasticsearch index mapping of field {name:?} has no string `type`, found {mapping}"
        ))
    })?;
    let field = match dtype_name {
        "dense_vector" => {
            let dim = object
                .get("dims")
                .and_then(Value::as_u64)
                .and_then(|dim| usize::try_from(dim).ok())
                .filter(|dim| *dim > 0)
                .ok_or_else(|| {
                    Error::validation(format!(
                        "Elasticsearch index mapping of vector field {name:?} has no positive \
                         `dims`, found {mapping}"
                    ))
                })?;
            FieldSchema {
                dim: Some(dim),
                ..FieldSchema::new(name, VectorDataType::FloatVector)
            }
        }
        "alias" => {
            return Err(Error::validation(
                match object.get("path").and_then(Value::as_str) {
                    Some(path) => format!(
                        "Elasticsearch index field {name:?} is an `alias` of {path:?}; an alias \
                     resolves to another field and cannot be reconstructed as a schema field"
                    ),
                    None => format!(
                        "Elasticsearch index field {name:?} is an `alias` of another field, which \
                     cannot be reconstructed as a schema field"
                    ),
                },
            ));
        }
        "keyword" | "text" | "constant_keyword" | "wildcard" => FieldSchema {
            // `ignore_above` caps what the keyword field indexes, which is the
            // schema's `max_length`; a non-numeric value means "no cap".
            max_length: object
                .get("ignore_above")
                .and_then(Value::as_u64)
                .and_then(|limit| usize::try_from(limit).ok()),
            ..FieldSchema::new(name, VectorDataType::Varchar)
        },
        "ip" | "binary" | "date" | "date_nanos" => FieldSchema::new(name, VectorDataType::Varchar),
        "integer" | "short" | "byte" => FieldSchema::new(name, VectorDataType::Int32),
        "long" | "unsigned_long" => FieldSchema::new(name, VectorDataType::Int64),
        "float" | "half_float" | "scaled_float" => FieldSchema::new(name, VectorDataType::Float),
        "double" => FieldSchema::new(name, VectorDataType::Double),
        "boolean" => FieldSchema::new(name, VectorDataType::Bool),
        "object" | "nested" | "flattened" | "geo_point" | "geo_shape" => {
            FieldSchema::new(name, VectorDataType::Json)
        }
        unsupported => {
            return Err(Error::validation(format!(
                "Elasticsearch index mapping of field {name:?} has type {unsupported:?}, which a \
                 collection field cannot represent; supported types are {}",
                SUPPORTED_DTYPES.join(", ")
            )));
        }
    };
    Ok(field)
}

#[cfg(test)]
mod tests {
    use openjiuwen_common::error::ErrorKind;

    use super::*;

    fn schema() -> CollectionSchema {
        CollectionSchema {
            fields: vec![
                FieldSchema {
                    is_primary: true,
                    auto_id: true,
                    max_length: Some(64),
                    ..FieldSchema::new("id", VectorDataType::Varchar)
                },
                FieldSchema {
                    dim: Some(3),
                    ..FieldSchema::new("embedding", VectorDataType::FloatVector)
                },
                FieldSchema::new("title", VectorDataType::Varchar),
                FieldSchema::new("count", VectorDataType::Int64),
                FieldSchema::new("ratio", VectorDataType::Double),
                FieldSchema::new("payload", VectorDataType::Json),
            ],
            description: Some("docs".to_string()),
            enable_dynamic_field: true,
        }
    }

    #[test]
    fn similarity_names_follow_the_metric() {
        assert_eq!(similarity_for(Metric::Cosine), "cosine");
        assert_eq!(similarity_for(Metric::Dot), "dot_product");
        assert_eq!(similarity_for(Metric::Euclidean), "l2_norm");
    }

    #[test]
    fn field_mapping_covers_every_data_type() {
        let vector = FieldSchema {
            dim: Some(4),
            ..FieldSchema::new("embedding", VectorDataType::FloatVector)
        };
        assert_eq!(
            field_mapping_with(&vector, Metric::Dot).expect("mapping"),
            json!({"type": "dense_vector", "dims": 4, "index": true, "similarity": "dot_product"})
        );
        assert_eq!(
            field_mapping_with(&vector, Metric::Cosine).expect("mapping")["similarity"],
            json!("cosine")
        );

        let varchar = FieldSchema {
            max_length: Some(32),
            ..FieldSchema::new("title", VectorDataType::Varchar)
        };
        assert_eq!(
            field_mapping_with(&varchar, Metric::Cosine).expect("mapping"),
            json!({"type": "keyword", "ignore_above": 32})
        );
        assert_eq!(
            field_mapping_with(
                &FieldSchema::new("title", VectorDataType::Varchar),
                Metric::Cosine,
            )
            .expect("mapping"),
            json!({"type": "keyword"})
        );

        for (dtype, expected) in [
            (VectorDataType::Int64, "long"),
            (VectorDataType::Int32, "integer"),
            (VectorDataType::Float, "float"),
            (VectorDataType::Double, "double"),
            (VectorDataType::Bool, "boolean"),
            (VectorDataType::Json, "object"),
        ] {
            assert_eq!(
                field_mapping_with(&FieldSchema::new("field", dtype), Metric::Cosine)
                    .expect("mapping"),
                json!({"type": expected}),
                "dtype {dtype:?}"
            );
        }

        let array = FieldSchema::new("tags", VectorDataType::Array);
        let error = field_mapping_with(&array, Metric::Cosine).expect_err("ARRAY is rejected");
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("\"tags\""), "message: {error}");
    }

    #[test]
    fn field_mapping_requires_a_positive_dim() {
        for dim in [None, Some(0)] {
            let field = FieldSchema {
                dim,
                ..FieldSchema::new("embedding", VectorDataType::FloatVector)
            };
            let error = field_mapping_with(&field, Metric::Cosine).expect_err("dim is required");
            assert_eq!(error.kind(), ErrorKind::Validation);
            assert!(error.to_string().contains("dim"), "message: {error}");
        }
    }

    #[test]
    fn properties_use_later_fields_on_duplicate_names() {
        let schema = CollectionSchema::new(vec![
            FieldSchema::new("id", VectorDataType::Varchar),
            FieldSchema::new("id", VectorDataType::Int64),
        ]);
        let properties = properties_for(&schema, Metric::Cosine).expect("properties");
        assert_eq!(properties, json!({"id": {"type": "long"}}));
    }

    #[test]
    fn schema_round_trips_through_the_index_metadata() {
        let schema = schema();
        let body = create_index_body(&schema, Metric::Euclidean).expect("body");
        assert_eq!(
            body["mappings"]["properties"]["embedding"],
            json!({"type": "dense_vector", "dims": 3, "index": true, "similarity": "l2_norm"})
        );
        assert_eq!(
            body["mappings"]["properties"]["id"],
            json!({"type": "keyword", "ignore_above": 64})
        );
        assert!(body["mappings"]["_meta"][SCHEMA_META_KEY].is_string());

        // The `get index` shape wraps the mappings in the index name.
        let wrapped = json!({"openjiuwen__docs": body.clone()});
        assert_eq!(
            schema_from_index(&wrapped).expect("schema"),
            Some(schema.clone())
        );
        // A bare `{"mappings": {..}}` body works too.
        assert_eq!(schema_from_index(&body).expect("schema"), Some(schema));
    }

    #[test]
    fn schema_from_index_reports_absent_and_broken_metadata() {
        assert_eq!(schema_from_index(&json!({})).expect("no mappings"), None);
        assert_eq!(
            schema_from_index(&json!({"openjiuwen__docs": {"settings": {}}})).expect("no mappings"),
            None
        );
        assert_eq!(
            schema_from_index(&json!({"mappings": {"properties": {}}})).expect("empty"),
            None
        );

        let error =
            schema_from_index(&json!({"mappings": {"_meta": {SCHEMA_META_KEY: "not json"}}}))
                .expect_err("broken schema");
        assert_eq!(error.kind(), ErrorKind::Validation);

        let error = schema_from_index(&json!({"mappings": {"_meta": {SCHEMA_META_KEY: 7}}}))
            .expect_err("non-string schema");
        assert_eq!(error.kind(), ErrorKind::Validation);
    }

    #[test]
    fn schema_from_properties_reconstructs_and_synthesizes_the_primary_key() {
        let properties = json!({
            "id": {"type": "keyword", "ignore_above": 128},
            "embedding": {"type": "dense_vector", "dims": 2, "similarity": "dot_product"},
            "title": {"type": "text"},
            "count": {"type": "long"},
            "small": {"type": "integer"},
            "weight": {"type": "float"},
            "ratio": {"type": "double"},
            "active": {"type": "boolean"},
            "payload": {"type": "object"},
            "nested": {"type": "nested"},
        });
        let schema = schema_from_properties(&properties).expect("schema");
        let primary = schema.primary_key().expect("primary key");
        assert_eq!(primary.name, "id");
        assert!(primary.is_primary);
        assert_eq!(primary.dtype, VectorDataType::Varchar);
        assert_eq!(primary.max_length, Some(128));
        assert!(!primary.auto_id);
        assert_eq!(schema.fields.len(), 10);
        assert_eq!(
            schema.vector_field("embedding").expect("vector field").dim,
            Some(2)
        );
        assert_eq!(
            schema
                .fields
                .iter()
                .find(|field| field.name == "title")
                .expect("title")
                .dtype,
            VectorDataType::Varchar
        );
        assert_eq!(
            schema
                .fields
                .iter()
                .find(|field| field.name == "nested")
                .expect("nested")
                .dtype,
            VectorDataType::Json
        );

        // Without an `id` property, one is synthesized in front.
        let schema =
            schema_from_properties(&json!({"value": {"type": "keyword"}})).expect("schema");
        assert_eq!(schema.fields.len(), 2);
        assert_eq!(schema.fields[0].name, ID_FIELD);
        assert!(schema.fields[0].is_primary);
        assert_eq!(schema.fields[0].dtype, VectorDataType::Varchar);

        // An `id` field that cannot be a primary key is rejected.
        let error = schema_from_properties(&json!({"id": {"type": "dense_vector", "dims": 2}}))
            .expect_err("vector id");
        assert_eq!(error.kind(), ErrorKind::Validation);

        // Unrepresentable types and untyped properties are rejected.
        let error = schema_from_properties(&json!({"value": {"type": "percolator"}}))
            .expect_err("unknown type");
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("\"value\""), "message: {error}");
        let error = schema_from_properties(&json!({"value": {}})).expect_err("untyped");
        assert_eq!(error.kind(), ErrorKind::Validation);
        let error = schema_from_properties(&json!([])).expect_err("not an object");
        assert_eq!(error.kind(), ErrorKind::Validation);
    }

    #[test]
    fn every_supported_dtype_maps_onto_a_field_schema() {
        for (dtype, expected) in [
            ("keyword", VectorDataType::Varchar),
            ("text", VectorDataType::Varchar),
            ("constant_keyword", VectorDataType::Varchar),
            ("wildcard", VectorDataType::Varchar),
            ("ip", VectorDataType::Varchar),
            ("binary", VectorDataType::Varchar),
            ("date", VectorDataType::Varchar),
            ("date_nanos", VectorDataType::Varchar),
            ("integer", VectorDataType::Int32),
            ("short", VectorDataType::Int32),
            ("byte", VectorDataType::Int32),
            ("long", VectorDataType::Int64),
            ("unsigned_long", VectorDataType::Int64),
            ("float", VectorDataType::Float),
            ("half_float", VectorDataType::Float),
            ("scaled_float", VectorDataType::Float),
            ("double", VectorDataType::Double),
            ("boolean", VectorDataType::Bool),
            ("object", VectorDataType::Json),
            ("nested", VectorDataType::Json),
            ("flattened", VectorDataType::Json),
            ("geo_point", VectorDataType::Json),
            ("geo_shape", VectorDataType::Json),
            ("dense_vector", VectorDataType::FloatVector),
        ] {
            // `field_from_mapping` is exhaustive over `SUPPORTED_DTYPES`; every
            // listed dtype must therefore appear in this table.
            assert!(SUPPORTED_DTYPES.contains(&dtype), "dtype: {dtype}");
            let mapping = match dtype {
                "dense_vector" => json!({"type": dtype, "dims": 8}),
                _ => json!({"type": dtype}),
            };
            let field = field_from_mapping("value", &mapping).expect("mapped dtype");
            assert_eq!(field.name, "value", "dtype: {dtype}");
            assert_eq!(field.dtype, expected, "dtype: {dtype}");
            assert!(!field.is_primary, "dtype: {dtype}");
            assert!(!field.auto_id, "dtype: {dtype}");
        }
        assert_eq!(SUPPORTED_DTYPES.len(), 24);

        // A vector field keeps its dimension.
        let field = field_from_mapping("embedding", &json!({"type": "dense_vector", "dims": 8}))
            .expect("dense_vector");
        assert_eq!(field.dim, Some(8));
    }

    #[test]
    fn keyword_like_mappings_keep_ignore_above_as_max_length() {
        for dtype in ["keyword", "text", "constant_keyword", "wildcard"] {
            let field = field_from_mapping("value", &json!({"type": dtype, "ignore_above": 64}))
                .expect("mapped keyword");
            assert_eq!(field.dtype, VectorDataType::Varchar, "dtype: {dtype}");
            assert_eq!(field.max_length, Some(64), "dtype: {dtype}");
        }
        // A non-numeric `ignore_above` is "no cap" rather than an error.
        let field = field_from_mapping("value", &json!({"type": "keyword", "ignore_above": "64"}))
            .expect("mapped keyword");
        assert_eq!(field.max_length, None);
        // `ip`, `date` and friends are strings, but carry no length cap.
        let field = field_from_mapping("value", &json!({"type": "ip", "ignore_above": 64}))
            .expect("mapped ip");
        assert_eq!(field.max_length, None);
    }

    #[test]
    fn unrepresentable_dtypes_name_the_field_and_list_the_supported_ones() {
        let error = field_from_mapping("value", &json!({"type": "some_future_type"}))
            .expect_err("unknown dtype");
        assert_eq!(error.kind(), ErrorKind::Validation);
        let message = error.to_string();
        assert!(message.contains("\"value\""), "message: {message}");
        assert!(message.contains("some_future_type"), "message: {message}");
        assert!(
            message.contains("supported types are"),
            "message: {message}"
        );
        for dtype in SUPPORTED_DTYPES {
            assert!(
                message.contains(dtype),
                "message is missing {dtype}: {message}"
            );
        }
    }

    #[test]
    fn aliases_and_untyped_mappings_are_rejected() {
        let error = field_from_mapping("alias_field", &json!({"type": "alias", "path": "target"}))
            .expect_err("alias");
        assert_eq!(error.kind(), ErrorKind::Validation);
        let message = error.to_string();
        assert!(message.contains("alias_field"), "message: {message}");
        assert!(message.contains("target"), "message: {message}");
        let error = field_from_mapping("alias_field", &json!({"type": "alias"}))
            .expect_err("alias without a path");
        assert_eq!(error.kind(), ErrorKind::Validation);

        let error = field_from_mapping("value", &json!({"analyzer": "standard"}))
            .expect_err("no type member");
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("\"value\""), "message: {error}");
        let error = field_from_mapping("value", &json!({"type": 7})).expect_err("non-string type");
        assert_eq!(error.kind(), ErrorKind::Validation);
        let error = field_from_mapping("value", &json!("keyword")).expect_err("non-object");
        assert_eq!(error.kind(), ErrorKind::Validation);

        // `dense_vector` still needs a positive dimension.
        for mapping in [
            json!({"type": "dense_vector"}),
            json!({"type": "dense_vector", "dims": 0}),
        ] {
            let error = field_from_mapping("embedding", &mapping).expect_err("missing dimension");
            assert_eq!(error.kind(), ErrorKind::Validation);
            assert!(
                error.to_string().contains("\"embedding\""),
                "message: {error}"
            );
        }
    }
}
