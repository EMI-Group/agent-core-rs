//! Vector storage.
//!
//! Collection schemas (`CollectionSchema`/`FieldSchema`), insert-only document
//! adds, and similarity search whose scores are always normalized to
//! "higher is better". The in-memory index is the reference implementation; the
//! local-filesystem store persists the same index as JSON.
//!
//! The contract mirrors the abstract half of the Python
//! `openjiuwen/core/foundation/store/base_vector_store.py`
//! (`VectorDataType`, `FieldSchema`, `CollectionSchema`, `VectorSearchResult`,
//! `BaseVectorStore`). Vendor backends (Milvus, Chroma, Gauss, pgvector) are out
//! of scope; [`InMemoryVectorStore`] and [`LocalFileVectorStore`] are the
//! reference implementations.
//!
//! # Semantics
//!
//! - A collection is created from a validated [`CollectionSchema`] and its name
//!   is unique inside a store; creating an existing name fails with
//!   [`ErrorKind::Validation`](openjiuwen_common::error::ErrorKind::Validation).
//! - `add_docs` is insert-only: a document whose primary key is already stored
//!   (or repeated inside the batch) fails with
//!   [`ErrorKind::Validation`](openjiuwen_common::error::ErrorKind::Validation),
//!   and a missing or mistyped field fails the same way.
//! - Every operation on an unknown collection fails with
//!   [`ErrorKind::NotFound`](openjiuwen_common::error::ErrorKind::NotFound),
//!   except `delete_collection` (a no-op for a missing collection) and
//!   `get_schema`/`collection_exists` (which report the absence instead).
//! - Search scores are normalized per [`Metric`] so that `1.0` is the most
//!   similar score and `0.0` the least; every returned score is finite and lies
//!   in `[0, 1]`. Ties are broken by primary key, so result order is
//!   deterministic.
//! - The local-filesystem backend accepts only collection names that are safe
//!   single path segments (`[A-Za-z0-9._-]`, not `.`/`..`); the in-memory backend
//!   accepts any non-empty name.

use std::collections::BTreeMap;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use openjiuwen_common::error::{Error, Result};

pub mod file;
pub mod memory;

pub use file::LocalFileVectorStore;
pub use memory::InMemoryVectorStore;

/// Default number of results returned by a search.
pub const DEFAULT_TOP_K: usize = 5;

/// Field data type, using the names of the Python `VectorDataType` enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VectorDataType {
    /// UTF-8 text with an optional maximum length.
    Varchar,
    /// Dense floating-point embedding (`dim` components).
    FloatVector,
    /// 64-bit signed integer.
    Int64,
    /// 32-bit signed integer.
    Int32,
    /// 32-bit floating-point number.
    Float,
    /// 64-bit floating-point number.
    Double,
    /// Boolean.
    Bool,
    /// Any JSON value.
    Json,
    /// JSON array.
    Array,
}

/// Definition of a single field in a collection schema.
///
/// The shape follows the Python `FieldSchema` (Milvus-like), with `max_length`
/// and `dim` optional instead of defaulted: a vector field without `dim` is
/// rejected by [`CollectionSchema::validate`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "snake_case")]
pub struct FieldSchema {
    /// Field name, unique inside its schema.
    pub name: String,
    /// Field data type.
    pub dtype: VectorDataType,
    /// Whether this field is the collection's primary key (at most one).
    pub is_primary: bool,
    /// Whether the store generates a primary key when a document omits it.
    pub auto_id: bool,
    /// Maximum character count of a `Varchar` value.
    pub max_length: Option<usize>,
    /// Vector dimension, required and positive for `FloatVector` fields.
    pub dim: Option<usize>,
    /// Human-readable description.
    pub description: Option<String>,
    /// Value inserted when a document omits this field.
    pub default_value: Option<Value>,
}

impl Default for FieldSchema {
    fn default() -> Self {
        Self {
            name: String::new(),
            dtype: VectorDataType::Varchar,
            is_primary: false,
            auto_id: false,
            max_length: None,
            dim: None,
            description: None,
            default_value: None,
        }
    }
}

impl FieldSchema {
    /// Creates a field definition with the given name and type.
    ///
    /// Every flag is off and every optional parameter is `None`; use the
    /// builder-style setters or struct update syntax to fill them in.
    #[must_use]
    pub fn new(name: impl Into<String>, dtype: VectorDataType) -> Self {
        Self {
            name: name.into(),
            dtype,
            ..Self::default()
        }
    }

    /// Returns `true` when this field is a `FloatVector` field.
    #[must_use]
    pub fn is_vector(&self) -> bool {
        self.dtype == VectorDataType::FloatVector
    }
}

/// Definition of a vector collection.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "snake_case")]
pub struct CollectionSchema {
    /// Field definitions, in declaration order.
    pub fields: Vec<FieldSchema>,
    /// Human-readable description.
    pub description: Option<String>,
    /// Whether documents may carry fields that the schema does not declare.
    pub enable_dynamic_field: bool,
}

impl CollectionSchema {
    /// Creates a schema from a list of field definitions.
    ///
    /// The schema is not validated here; call [`Self::validate`] (which every
    /// store does on `create_collection`) before using it.
    #[must_use]
    pub fn new(fields: Vec<FieldSchema>) -> Self {
        Self {
            fields,
            ..Self::default()
        }
    }

    /// The primary-key field, if the schema declares one.
    #[must_use]
    pub fn primary_key(&self) -> Option<&FieldSchema> {
        self.fields.iter().find(|field| field.is_primary)
    }

    /// The field called `name` when it is a `FloatVector` field.
    #[must_use]
    pub fn vector_field(&self, name: &str) -> Option<&FieldSchema> {
        self.fields
            .iter()
            .find(|field| field.name == name && field.is_vector())
    }

    /// The first `FloatVector` field, used when a query names no vector field.
    #[must_use]
    pub fn default_vector_field(&self) -> Option<&FieldSchema> {
        self.fields.iter().find(|field| field.is_vector())
    }

    /// Every `FloatVector` field, in declaration order.
    #[must_use]
    pub fn vector_fields(&self) -> Vec<&FieldSchema> {
        self.fields
            .iter()
            .filter(|field| field.is_vector())
            .collect()
    }

    /// Validates the schema, mirroring the Python `FieldSchema`/`CollectionSchema`
    /// pydantic validators.
    ///
    /// The schema is valid when
    ///
    /// - exactly one field is the primary key,
    /// - every field name is non-empty and unique,
    /// - at least one `FloatVector` field exists, and every `FloatVector` field
    ///   has a `dim` greater than zero,
    /// - `Varchar` `max_length`, when set, is greater than zero.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::Validation`](openjiuwen_common::error::ErrorKind::Validation)
    /// naming the offending field for every violation.
    pub fn validate(&self) -> Result<()> {
        let mut names: BTreeMap<&str, usize> = BTreeMap::new();
        for field in &self.fields {
            if field.name.is_empty() {
                return Err(Error::validation(
                    "collection schema field name must not be empty",
                ));
            }
            *names.entry(field.name.as_str()).or_default() += 1;
        }
        if let Some((name, _)) = names.iter().find(|(_, count)| **count > 1) {
            return Err(Error::validation(format!(
                "duplicate collection schema field name {name:?}"
            )));
        }

        let primaries: Vec<&FieldSchema> = self
            .fields
            .iter()
            .filter(|field| field.is_primary)
            .collect();
        match primaries.as_slice() {
            [] => {
                return Err(Error::validation(
                    "collection schema must declare exactly one primary key field, found none",
                ));
            }
            [primary] => {
                if primary.is_vector() {
                    return Err(Error::validation(format!(
                        "primary key field {:?} must not be a FLOAT_VECTOR field",
                        primary.name
                    )));
                }
            }
            [first, second, ..] => {
                return Err(Error::validation(format!(
                    "collection schema must declare exactly one primary key field, found {:?} and {:?}",
                    first.name, second.name
                )));
            }
        }

        let vectors = self.vector_fields();
        if vectors.is_empty() {
            return Err(Error::validation(
                "collection schema must declare at least one FLOAT_VECTOR field",
            ));
        }
        for field in vectors {
            match field.dim {
                Some(0) | None => {
                    return Err(Error::validation(format!(
                        "dim of vector field {:?} is invalid, dim={:?}",
                        field.name, field.dim
                    )));
                }
                Some(_) => {}
            }
        }
        for field in &self.fields {
            if field.dtype == VectorDataType::Varchar && field.max_length == Some(0) {
                return Err(Error::validation(format!(
                    "max_length of VARCHAR field {:?} must be greater than zero",
                    field.name
                )));
            }
            if !field.is_vector() && field.dim.is_some() {
                return Err(Error::validation(format!(
                    "dim is only valid for FLOAT_VECTOR fields, field={:?}",
                    field.name
                )));
            }
        }
        Ok(())
    }
}

/// A similarity search request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "snake_case")]
pub struct VectorQuery {
    /// Query vector; its length must match the target vector field's `dim`.
    pub query_vector: Vec<f32>,
    /// Vector field to search, or `None` for the schema's first `FloatVector` field.
    pub vector_field: Option<String>,
    /// Maximum number of results.
    pub top_k: usize,
    /// Scalar filter; `null` (the default) matches every document.
    pub filters: Value,
}

impl Default for VectorQuery {
    fn default() -> Self {
        Self {
            query_vector: Vec::new(),
            vector_field: None,
            top_k: DEFAULT_TOP_K,
            filters: Value::Null,
        }
    }
}

impl VectorQuery {
    /// Creates a query for `query_vector` with the default `top_k` and no filter.
    #[must_use]
    pub fn new(query_vector: Vec<f32>) -> Self {
        Self {
            query_vector,
            ..Self::default()
        }
    }

    /// Sets the vector field to search.
    #[must_use]
    pub fn with_vector_field(mut self, vector_field: impl Into<String>) -> Self {
        self.vector_field = Some(vector_field.into());
        self
    }

    /// Sets the maximum number of results.
    #[must_use]
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }

    /// Sets the scalar filter applied to candidate documents.
    #[must_use]
    pub fn with_filters(mut self, filters: Value) -> Self {
        self.filters = filters;
        self
    }
}

/// One search hit: the normalized score and the stored document fields.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct VectorSearchResult {
    /// Similarity score in `[0, 1]`; higher is more similar.
    pub score: f32,
    /// The stored document, including its primary key.
    pub fields: Map<String, Value>,
}

impl VectorSearchResult {
    /// Creates a hit from a score and the document's stored fields.
    #[must_use]
    pub fn new(score: f32, fields: Map<String, Value>) -> Self {
        Self { score, fields }
    }
}

/// Similarity metric of the shared index.
///
/// Every variant normalizes its raw similarity into `[0, 1]`, so callers never
/// have to know whether a backend's native score is a distance or a similarity:
///
/// - [`Metric::Cosine`] — `(cosine_similarity + 1) / 2`
/// - [`Metric::Dot`] — `(tanh(dot_product) + 1) / 2`
/// - [`Metric::Euclidean`] — `1 / (1 + euclidean_distance)`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Metric {
    /// Cosine similarity, the default metric.
    #[default]
    Cosine,
    /// Dot product of the two vectors.
    Dot,
    /// Euclidean (L2) distance.
    Euclidean,
}

impl Metric {
    /// Scores two vectors, returning a finite value in `[0, 1]`.
    ///
    /// Vectors of different lengths, and vectors containing non-finite
    /// components, score `0.0` instead of producing `NaN`; two empty vectors
    /// score `1.0` (they are identical), and a zero vector scores `0.5` against
    /// a non-zero one (orthogonal by convention).
    #[must_use]
    pub fn score(self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }
        if !a.iter().chain(b).all(|value| value.is_finite()) {
            return 0.0;
        }
        if a.is_empty() {
            // Two empty vectors are identical.
            return 1.0;
        }
        let raw = match self {
            Self::Cosine => {
                let dot: f64 = a
                    .iter()
                    .zip(b)
                    .map(|(x, y)| f64::from(*x) * f64::from(*y))
                    .sum();
                let norm_a: f64 = a
                    .iter()
                    .map(|x| f64::from(*x) * f64::from(*x))
                    .sum::<f64>()
                    .sqrt();
                let norm_b: f64 = b
                    .iter()
                    .map(|y| f64::from(*y) * f64::from(*y))
                    .sum::<f64>()
                    .sqrt();
                if norm_a <= 0.0 || norm_b <= 0.0 {
                    // A zero vector is orthogonal to everything by convention.
                    return 0.5;
                }
                f64::midpoint((dot / (norm_a * norm_b)).clamp(-1.0, 1.0), 1.0)
            }
            Self::Dot => {
                let dot: f64 = a
                    .iter()
                    .zip(b)
                    .map(|(x, y)| f64::from(*x) * f64::from(*y))
                    .sum();
                f64::midpoint(dot.tanh(), 1.0)
            }
            Self::Euclidean => {
                let distance: f64 = a
                    .iter()
                    .zip(b)
                    .map(|(x, y)| {
                        let delta = f64::from(*x) - f64::from(*y);
                        delta * delta
                    })
                    .sum::<f64>()
                    .sqrt();
                1.0 / (1.0 + distance)
            }
        };
        normalize_score(raw)
    }
}

/// Clamps a raw score into `[0, 1]`, mapping non-finite values to `0.0`.
fn normalize_score(raw: f64) -> f32 {
    if raw.is_finite() {
        raw.clamp(0.0, 1.0) as f32
    } else {
        0.0
    }
}

/// Vector collections of JSON documents.
///
/// Implementations are expected to be cheap to share: every method takes
/// `&self`, so a store can be used through an `Arc<dyn VectorStore>`.
///
/// # Errors
///
/// `add_docs`, `search`, `delete_docs_by_ids` and `delete_docs_by_filters`
/// return [`ErrorKind::NotFound`](openjiuwen_common::error::ErrorKind::NotFound)
/// for an unknown collection; `create_collection` and document validation
/// return [`ErrorKind::Validation`](openjiuwen_common::error::ErrorKind::Validation).
#[async_trait]
pub trait VectorStore: Send + Sync {
    /// Creates a collection from a validated schema.
    ///
    /// # Errors
    ///
    /// Returns a validation error when the schema is invalid or a collection of
    /// the same name already exists.
    async fn create_collection(&self, collection: &str, schema: CollectionSchema) -> Result<()>;

    /// Deletes a collection and every document in it. A missing collection is a
    /// no-op.
    async fn delete_collection(&self, collection: &str) -> Result<()>;

    /// Reports whether a collection exists.
    async fn collection_exists(&self, collection: &str) -> Result<bool>;

    /// Returns the schema of a collection, or `None` when it does not exist.
    async fn get_schema(&self, collection: &str) -> Result<Option<CollectionSchema>>;

    /// Inserts documents, returning how many were inserted.
    ///
    /// Documents are validated against the schema, `default_value`s are filled
    /// in, and an `auto_id` primary key is generated when the document omits it.
    ///
    /// # Errors
    ///
    /// Returns a validation error for a duplicate primary key (inside the batch
    /// or already stored), a missing required field, an unknown vector field, or
    /// a value whose type or dimension does not match the schema. Validation
    /// happens before any document is inserted, so a failed call inserts nothing.
    async fn add_docs(&self, collection: &str, docs: Vec<Map<String, Value>>) -> Result<usize>;

    /// Searches for the documents closest to `query.query_vector`.
    ///
    /// Results are ordered by score descending, ties broken by primary key
    /// ascending, and truncated to `query.top_k`.
    async fn search(
        &self,
        collection: &str,
        query: &VectorQuery,
    ) -> Result<Vec<VectorSearchResult>>;

    /// Deletes documents by primary key, returning how many were removed.
    ///
    /// Unknown ids are ignored.
    async fn delete_docs_by_ids(&self, collection: &str, ids: &[String]) -> Result<usize>;

    /// Deletes documents matching `filters`, returning how many were removed.
    ///
    /// The filter uses the shared [`crate::store::filter`] language, so `null`
    /// or `{}` matches (and therefore deletes) every document.
    async fn delete_docs_by_filters(&self, collection: &str, filters: &Value) -> Result<usize>;

    /// Lists every collection name in ascending order.
    async fn list_collection_names(&self) -> Result<Vec<String>>;
}

/// The shared not-found error for a missing collection.
pub(crate) fn collection_not_found(collection: &str) -> Error {
    Error::not_found(format!("vector collection {collection:?} not found"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn schema() -> CollectionSchema {
        CollectionSchema::new(vec![
            FieldSchema {
                name: "id".to_string(),
                dtype: VectorDataType::Varchar,
                is_primary: true,
                ..FieldSchema::default()
            },
            FieldSchema {
                name: "embedding".to_string(),
                dtype: VectorDataType::FloatVector,
                dim: Some(3),
                ..FieldSchema::default()
            },
        ])
    }

    #[test]
    fn field_schema_defaults_are_empty() {
        let field = FieldSchema::default();
        assert_eq!(field.name, "");
        assert_eq!(field.dtype, VectorDataType::Varchar);
        assert!(!field.is_primary);
        assert!(!field.auto_id);
        assert_eq!(field.max_length, None);
        assert_eq!(field.dim, None);
        assert_eq!(field.description, None);
        assert_eq!(field.default_value, None);
        assert!(!field.is_vector());
        assert!(FieldSchema::new("v", VectorDataType::FloatVector).is_vector());
    }

    #[test]
    fn schema_helpers_locate_primary_and_vector_fields() {
        let schema = schema();
        assert_eq!(
            schema.primary_key().map(|field| field.name.as_str()),
            Some("id")
        );
        assert_eq!(
            schema
                .default_vector_field()
                .map(|field| field.name.as_str()),
            Some("embedding")
        );
        assert!(schema.vector_field("embedding").is_some());
        assert!(schema.vector_field("id").is_none());
        assert!(schema.vector_field("missing").is_none());
        assert_eq!(schema.vector_fields().len(), 1);
        assert!(schema.validate().is_ok());
    }

    #[test]
    fn validate_rejects_bad_primary_keys() {
        let no_primary = CollectionSchema::new(vec![FieldSchema {
            name: "embedding".to_string(),
            dtype: VectorDataType::FloatVector,
            dim: Some(2),
            ..FieldSchema::default()
        }]);
        assert_eq!(
            no_primary.validate().unwrap_err().kind(),
            openjiuwen_common::error::ErrorKind::Validation
        );

        let two_primary = CollectionSchema::new(vec![
            FieldSchema {
                name: "a".to_string(),
                is_primary: true,
                ..FieldSchema::default()
            },
            FieldSchema {
                name: "b".to_string(),
                is_primary: true,
                ..FieldSchema::default()
            },
            FieldSchema {
                name: "embedding".to_string(),
                dtype: VectorDataType::FloatVector,
                dim: Some(2),
                ..FieldSchema::default()
            },
        ]);
        assert!(two_primary.validate().is_err());

        let mut vector_primary = schema();
        vector_primary.fields[0].is_primary = false;
        vector_primary.fields[1].is_primary = true;
        assert!(vector_primary.validate().is_err());
    }

    #[test]
    fn validate_rejects_bad_fields() {
        let no_vector = CollectionSchema::new(vec![FieldSchema {
            name: "id".to_string(),
            is_primary: true,
            ..FieldSchema::default()
        }]);
        assert!(no_vector.validate().is_err());

        let zero_dim = CollectionSchema::new(vec![
            FieldSchema {
                name: "id".to_string(),
                is_primary: true,
                ..FieldSchema::default()
            },
            FieldSchema {
                name: "embedding".to_string(),
                dtype: VectorDataType::FloatVector,
                dim: Some(0),
                ..FieldSchema::default()
            },
        ]);
        assert!(zero_dim.validate().is_err());

        let missing_dim = CollectionSchema::new(vec![
            FieldSchema {
                name: "id".to_string(),
                is_primary: true,
                ..FieldSchema::default()
            },
            FieldSchema {
                name: "embedding".to_string(),
                dtype: VectorDataType::FloatVector,
                dim: None,
                ..FieldSchema::default()
            },
        ]);
        assert!(missing_dim.validate().is_err());

        let duplicate_names = CollectionSchema::new(vec![
            FieldSchema {
                name: "id".to_string(),
                is_primary: true,
                ..FieldSchema::default()
            },
            FieldSchema {
                name: "id".to_string(),
                dtype: VectorDataType::FloatVector,
                dim: Some(2),
                ..FieldSchema::default()
            },
        ]);
        assert!(duplicate_names.validate().is_err());

        let empty_name = CollectionSchema::new(vec![
            FieldSchema {
                name: String::new(),
                is_primary: true,
                ..FieldSchema::default()
            },
            FieldSchema {
                name: "embedding".to_string(),
                dtype: VectorDataType::FloatVector,
                dim: Some(2),
                ..FieldSchema::default()
            },
        ]);
        assert!(empty_name.validate().is_err());

        let mut zero_length = schema();
        zero_length.fields[0].max_length = Some(0);
        assert!(zero_length.validate().is_err());

        let mut dim_on_scalar = schema();
        dim_on_scalar.fields[0].dim = Some(4);
        assert!(dim_on_scalar.validate().is_err());
    }

    #[test]
    fn data_types_serialize_snake_case_and_round_trip() {
        let cases = [
            (VectorDataType::Varchar, "\"varchar\""),
            (VectorDataType::FloatVector, "\"float_vector\""),
            (VectorDataType::Int64, "\"int64\""),
            (VectorDataType::Int32, "\"int32\""),
            (VectorDataType::Float, "\"float\""),
            (VectorDataType::Double, "\"double\""),
            (VectorDataType::Bool, "\"bool\""),
            (VectorDataType::Json, "\"json\""),
            (VectorDataType::Array, "\"array\""),
        ];
        for (dtype, json) in cases {
            assert_eq!(serde_json::to_string(&dtype).expect("serialize"), json);
        }

        let field = FieldSchema {
            name: "embedding".to_string(),
            dtype: VectorDataType::FloatVector,
            dim: Some(8),
            description: Some("dense".to_string()),
            ..FieldSchema::default()
        };
        let value = serde_json::to_value(&field).expect("serialize field");
        assert_eq!(value["dtype"], "float_vector");
        assert_eq!(value["name"], "embedding");
        assert_eq!(value["dim"], 8);
        assert_eq!(
            serde_json::from_value::<FieldSchema>(value).expect("round-trip"),
            field
        );
    }

    #[test]
    fn schemas_deserialize_tolerantly_with_defaults() {
        let schema: CollectionSchema = serde_json::from_str(
            r#"{"fields":[{"name":"id","is_primary":true,"extra":1}],
                "unknown":true}"#,
        )
        .expect("tolerant deserialize");
        assert_eq!(schema.fields.len(), 1);
        assert_eq!(schema.fields[0].dtype, VectorDataType::Varchar);
        assert_eq!(schema.fields[0].max_length, None);
        assert!(!schema.enable_dynamic_field);
        assert_eq!(schema.description, None);

        let empty: CollectionSchema =
            serde_json::from_str("{}").expect("an empty schema object is tolerated");
        assert_eq!(empty, CollectionSchema::default());
        assert_eq!(
            serde_json::from_str::<VectorQuery>("{}").expect("empty query"),
            VectorQuery::default()
        );
    }

    #[test]
    fn query_defaults_to_five_results_and_matches_all() {
        let query = VectorQuery::default();
        assert_eq!(query.top_k, DEFAULT_TOP_K);
        assert_eq!(query.top_k, 5);
        assert_eq!(query.vector_field, None);
        assert_eq!(query.filters, Value::Null);
        assert_eq!(query.query_vector, Vec::<f32>::new());

        let query = VectorQuery::new(vec![1.0, 0.0])
            .with_vector_field("embedding")
            .with_top_k(2)
            .with_filters(serde_json::json!({"kind": "a"}));
        assert_eq!(query.top_k, 2);
        assert_eq!(query.vector_field.as_deref(), Some("embedding"));
        assert_eq!(query.filters, serde_json::json!({"kind": "a"}));
    }

    #[test]
    fn metric_scores_are_finite_and_normalized() {
        fn assert_score(actual: f32, expected: f32) {
            assert!(
                (actual - expected).abs() < 1e-6,
                "expected {expected}, got {actual}"
            );
        }

        assert_score(
            Metric::Cosine.score(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]),
            1.0,
        );
        assert_score(Metric::Cosine.score(&[1.0, 0.0], &[-1.0, 0.0]), 0.0);
        assert_score(Metric::Cosine.score(&[1.0, 0.0], &[0.0, 1.0]), 0.5);

        // Zero vectors and mismatched dimensions never produce NaN.
        assert_score(Metric::Cosine.score(&[0.0, 0.0], &[1.0, 1.0]), 0.5);
        assert_score(Metric::Dot.score(&[0.0, 0.0], &[1.0, 1.0]), 0.5);
        assert_score(Metric::Cosine.score(&[1.0], &[1.0, 2.0]), 0.0);
        assert_score(Metric::Cosine.score(&[], &[]), 1.0);
        assert_score(Metric::Euclidean.score(&[], &[]), 1.0);

        for metric in [Metric::Cosine, Metric::Dot, Metric::Euclidean] {
            for (a, b) in [
                (vec![0.0_f32, 0.0], vec![0.0_f32, 0.0]),
                (vec![f32::MAX, f32::MAX], vec![f32::MAX, f32::MIN]),
                (vec![f32::MIN, 1.0], vec![f32::MAX, -1.0]),
            ] {
                let score = metric.score(&a, &b);
                assert!(score.is_finite(), "{metric:?} produced {score}");
                assert!((0.0..=1.0).contains(&score), "{metric:?} produced {score}");
            }
        }
        // Non-finite components are rejected.
        assert_score(Metric::Cosine.score(&[f32::NAN], &[1.0]), 0.0);
        assert_score(Metric::Dot.score(&[1.0], &[f32::INFINITY]), 0.0);
    }

    #[test]
    fn metric_ordering_tracks_similarity() {
        let query = [1.0_f32, 0.0];
        let near = [0.9_f32, 0.1];
        let far = [-1.0_f32, 0.0];
        for metric in [Metric::Cosine, Metric::Dot, Metric::Euclidean] {
            assert!(
                metric.score(&query, &near) > metric.score(&query, &far),
                "{metric:?} must rank the nearer vector higher"
            );
            assert!(
                metric.score(&query, &query) >= metric.score(&query, &near),
                "{metric:?} must score an identical vector highest"
            );
        }
        // Cosine and Euclidean reach the top of the scale for identical vectors;
        // Dot is squashed through `tanh`, so it only approaches it.
        assert!((Metric::Cosine.score(&query, &query) - 1.0).abs() < 1e-6);
        assert!((Metric::Euclidean.score(&query, &query) - 1.0).abs() < 1e-6);
        assert!(Metric::Dot.score(&query, &query) < 1.0);
        assert!(Metric::Dot.score(&query, &query) > Metric::Dot.score(&query, &near));
        // Dot rewards magnitude, Euclidean penalizes distance.
        assert!(Metric::Dot.score(&[2.0], &[2.0]) > Metric::Dot.score(&[0.5], &[0.5]));
        assert!(Metric::Euclidean.score(&[0.0], &[1.0]) > Metric::Euclidean.score(&[0.0], &[5.0]));
    }

    #[test]
    fn search_result_round_trips() {
        let fields = serde_json::json!({"id": "a"});
        let result = VectorSearchResult::new(0.25, fields.as_object().cloned().unwrap_or_default());
        let value = serde_json::to_value(&result).expect("serialize result");
        assert_eq!(value["score"], 0.25);
        assert_eq!(value["fields"]["id"], "a");
        assert_eq!(
            serde_json::from_value::<VectorSearchResult>(value).expect("round-trip"),
            result
        );
    }
}
