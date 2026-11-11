//! In-memory vector store and the shared index engine.
//!
//! [`VectorIndex`] is the reference implementation of the vector contract:
//! schemas, documents, and their embeddings live in [`BTreeMap`]s, so every
//! listing is sorted and every result order is deterministic (score descending,
//! primary key ascending for ties).
//!
//! The local-filesystem backend ([`super::file::LocalFileVectorStore`]) reuses
//! this engine and only adds JSON snapshot persistence, so the two cannot drift.

use std::collections::BTreeMap;

use async_trait::async_trait;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use uuid::Uuid;

use openjiuwen_common::error::{Error, Result};

use super::{
    collection_not_found, CollectionSchema, FieldSchema, Metric, VectorDataType, VectorQuery,
    VectorSearchResult, VectorStore,
};
use crate::store::filter;

/// One stored document: its scalar/JSON fields plus its extracted embeddings.
///
/// `fields` holds every non-vector field; `FLOAT_VECTOR` components live as
/// `f32` in `vectors`, and [`Doc::to_json`] recombines both.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "snake_case")]
pub(crate) struct Doc {
    /// Scalar, JSON, and unknown fields, exactly as stored.
    pub(crate) fields: Map<String, Value>,
    /// Embeddings, keyed by vector field name.
    pub(crate) vectors: BTreeMap<String, Vec<f32>>,
}

impl Doc {
    /// The document as a JSON object, with vector fields restored as arrays.
    #[must_use]
    pub(crate) fn to_json(&self) -> Map<String, Value> {
        let mut fields = self.fields.clone();
        for (name, vector) in &self.vectors {
            fields.insert(
                name.clone(),
                Value::Array(vector.iter().map(|value| Value::from(*value)).collect()),
            );
        }
        fields
    }
}

/// One collection: its schema and its documents, keyed by primary key.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) struct CollectionData {
    /// Validated schema of the collection.
    pub(crate) schema: CollectionSchema,
    /// Documents by primary key.
    pub(crate) docs: BTreeMap<String, Doc>,
}

/// The shared vector index: collections, their schemas, and their documents.
///
/// The index is synchronous (mutations happen under a lock in the owning store).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "snake_case")]
pub(crate) struct VectorIndex {
    collections: BTreeMap<String, CollectionData>,
}

impl VectorIndex {
    /// Creates an empty index.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Creates a collection after validating its schema.
    pub(crate) fn create_collection(
        &mut self,
        collection: &str,
        schema: CollectionSchema,
    ) -> Result<()> {
        schema.validate()?;
        if self.collections.contains_key(collection) {
            return Err(Error::validation(format!(
                "vector collection {collection:?} already exists"
            )));
        }
        self.collections.insert(
            collection.to_string(),
            CollectionData {
                schema,
                docs: BTreeMap::new(),
            },
        );
        Ok(())
    }

    /// Removes a collection; a missing collection is a no-op.
    pub(crate) fn delete_collection(&mut self, collection: &str) {
        self.collections.remove(collection);
    }

    /// Reports whether a collection exists.
    pub(crate) fn collection_exists(&self, collection: &str) -> bool {
        self.collections.contains_key(collection)
    }

    /// The schema of a collection.
    pub(crate) fn schema(&self, collection: &str) -> Option<&CollectionSchema> {
        self.collections.get(collection).map(|data| &data.schema)
    }

    /// Every collection name, ascending.
    pub(crate) fn collection_names(&self) -> Vec<String> {
        self.collections.keys().cloned().collect()
    }

    /// Inserts documents into a collection, returning how many were inserted.
    pub(crate) fn add_docs(
        &mut self,
        collection: &str,
        docs: Vec<Map<String, Value>>,
    ) -> Result<usize> {
        let data = self
            .collections
            .get_mut(collection)
            .ok_or_else(|| collection_not_found(collection))?;
        let prepared = prepare_docs(&data.schema, docs)?;
        for (id, _) in &prepared {
            if data.docs.contains_key(id) {
                return Err(Error::validation(format!(
                    "document with primary key {id:?} already exists in collection {collection:?}"
                )));
            }
        }
        let inserted = prepared.len();
        for (id, doc) in prepared {
            data.docs.insert(id, doc);
        }
        Ok(inserted)
    }

    /// Searches a collection with `metric`.
    pub(crate) fn search(
        &self,
        collection: &str,
        query: &VectorQuery,
        metric: Metric,
    ) -> Result<Vec<VectorSearchResult>> {
        let data = self
            .collections
            .get(collection)
            .ok_or_else(|| collection_not_found(collection))?;
        let field = resolve_vector_field(&data.schema, collection, query)?;
        let dim = field.dim.unwrap_or(0);
        if query.query_vector.len() != dim {
            return Err(Error::validation(format!(
                "query vector dimension {} does not match vector field {:?} dimension {dim} of collection {collection:?}",
                query.query_vector.len(),
                field.name
            )));
        }

        let mut hits: Vec<(f32, String, Map<String, Value>)> = Vec::new();
        for (primary_key, doc) in &data.docs {
            let Some(vector) = doc.vectors.get(&field.name) else {
                // Unreachable for documents written by `add_docs`; a hand-edited
                // snapshot without the vector simply cannot match.
                continue;
            };
            let view = doc.to_json();
            if !filter::matches(&query.filters, &Value::Object(view.clone())) {
                continue;
            }
            hits.push((
                metric.score(&query.query_vector, vector),
                primary_key.clone(),
                view,
            ));
        }
        hits.sort_by(|left, right| {
            right
                .0
                .total_cmp(&left.0)
                .then_with(|| left.1.cmp(&right.1))
        });
        hits.truncate(query.top_k);
        Ok(hits
            .into_iter()
            .map(|(score, _, fields)| VectorSearchResult::new(score, fields))
            .collect())
    }

    /// Deletes documents by primary key, returning how many were removed.
    pub(crate) fn delete_docs_by_ids(&mut self, collection: &str, ids: &[String]) -> Result<usize> {
        let data = self
            .collections
            .get_mut(collection)
            .ok_or_else(|| collection_not_found(collection))?;
        let mut deleted = 0;
        for id in ids {
            if data.docs.remove(id).is_some() {
                deleted += 1;
            }
        }
        Ok(deleted)
    }

    /// Deletes documents matching `filters`, returning how many were removed.
    pub(crate) fn delete_docs_by_filters(
        &mut self,
        collection: &str,
        filters: &Value,
    ) -> Result<usize> {
        let data = self
            .collections
            .get_mut(collection)
            .ok_or_else(|| collection_not_found(collection))?;
        let doomed: Vec<String> = data
            .docs
            .iter()
            .filter(|(_, doc)| filter::matches(filters, &Value::Object(doc.to_json())))
            .map(|(id, _)| id.clone())
            .collect();
        let deleted = doomed.len();
        for id in doomed {
            data.docs.remove(&id);
        }
        Ok(deleted)
    }

    /// Serializes the whole index as a JSON snapshot (used by tests and bulk
    /// export; the file backend persists [`VectorIndex::collection_value`] instead).
    #[allow(dead_code)]
    pub(crate) fn to_value(&self) -> Result<Value> {
        serde_json::to_value(self).map_err(Error::external)
    }

    /// Restores an index from a JSON snapshot, validating every schema.
    #[allow(dead_code)]
    pub(crate) fn from_value(value: &Value) -> Result<Self> {
        let index: Self = serde_json::from_value(value.clone()).map_err(Error::external)?;
        for (collection, data) in &index.collections {
            data.schema.validate().map_err(|error| {
                Error::validation(format!(
                    "vector snapshot collection {collection:?} has an invalid schema: {}",
                    error.message()
                ))
            })?;
        }
        Ok(index)
    }

    /// Serializes one collection, or `None` when it does not exist.
    pub(crate) fn collection_value(&self, collection: &str) -> Result<Option<Value>> {
        match self.collections.get(collection) {
            Some(data) => Ok(Some(serde_json::to_value(data).map_err(Error::external)?)),
            None => Ok(None),
        }
    }

    /// Inserts a collection restored from a snapshot; its schema is validated.
    pub(crate) fn restore_collection(
        &mut self,
        collection: String,
        data: CollectionData,
    ) -> Result<()> {
        data.schema.validate()?;
        self.collections.insert(collection, data);
        Ok(())
    }
}

/// Picks the vector field a query scores against.
fn resolve_vector_field<'a>(
    schema: &'a CollectionSchema,
    collection: &str,
    query: &VectorQuery,
) -> Result<&'a FieldSchema> {
    match &query.vector_field {
        Some(name) => schema.vector_field(name).ok_or_else(|| {
            Error::validation(format!(
                "field {name:?} is not a FLOAT_VECTOR field of collection {collection:?}"
            ))
        }),
        None => schema.default_vector_field().ok_or_else(|| {
            Error::validation(format!(
                "collection {collection:?} declares no FLOAT_VECTOR field"
            ))
        }),
    }
}

/// Validates and converts a batch of raw documents into index entries.
///
/// Returns `(primary key, document)` pairs; nothing is inserted until the
/// whole batch is valid.
fn prepare_docs(
    schema: &CollectionSchema,
    docs: Vec<Map<String, Value>>,
) -> Result<Vec<(String, Doc)>> {
    let primary = schema
        .primary_key()
        .ok_or_else(|| Error::validation("collection schema declares no primary key field"))?;
    let mut prepared: Vec<(String, Doc)> = Vec::with_capacity(docs.len());
    let mut seen: BTreeMap<String, usize> = BTreeMap::new();
    for (position, raw) in docs.into_iter().enumerate() {
        let (id, doc) = prepare_doc(schema, primary, raw, position)?;
        if let Some(first) = seen.insert(id.clone(), position) {
            return Err(Error::validation(format!(
                "duplicate primary key {id:?} in the batch (positions {first} and {position})"
            )));
        }
        prepared.push((id, doc));
    }
    Ok(prepared)
}

/// Validates one document, applies defaults, and extracts its embeddings.
fn prepare_doc(
    schema: &CollectionSchema,
    primary: &FieldSchema,
    raw: Map<String, Value>,
    position: usize,
) -> Result<(String, Doc)> {
    let mut fields = raw;
    for field in &schema.fields {
        if let (false, Some(default)) = (fields.contains_key(&field.name), &field.default_value) {
            fields.insert(field.name.clone(), default.clone());
        }
    }

    let id = match fields.get(&primary.name) {
        Some(Value::String(value)) if !value.is_empty() => value.clone(),
        Some(Value::String(_)) => {
            return Err(Error::validation(format!(
                "document at position {position} has an empty primary key field {:?}",
                primary.name
            )));
        }
        Some(Value::Number(number)) if number.is_i64() || number.is_u64() => number.to_string(),
        Some(other) => {
            return Err(Error::validation(format!(
                "document at position {position} has primary key field {:?} of type {other}; \
                 a string or integer is required",
                primary.name
            )));
        }
        None if primary.auto_id => {
            let generated = Uuid::new_v4().to_string();
            fields.insert(primary.name.clone(), Value::String(generated.clone()));
            generated
        }
        None => {
            return Err(Error::validation(format!(
                "document at position {position} is missing primary key field {:?}",
                primary.name
            )));
        }
    };

    let mut vectors: BTreeMap<String, Vec<f32>> = BTreeMap::new();
    for field in &schema.fields {
        match fields.get(&field.name) {
            Some(value) if field.is_vector() => {
                vectors.insert(field.name.clone(), parse_vector(field, value)?);
            }
            Some(value) => check_value(field, value)?,
            None if field.is_vector() => {
                return Err(Error::validation(format!(
                    "document {id:?} is missing vector field {:?}",
                    field.name
                )));
            }
            None => {}
        }
    }

    for name in vectors.keys() {
        fields.remove(name);
    }
    Ok((id, Doc { fields, vectors }))
}

/// Parses a vector field value, checking its dimension and numeric components.
fn parse_vector(field: &FieldSchema, value: &Value) -> Result<Vec<f32>> {
    let items = value.as_array().ok_or_else(|| {
        Error::validation(format!(
            "vector field {:?} expects an array of numbers, got {value}",
            field.name
        ))
    })?;
    let dim = field.dim.unwrap_or(0);
    if items.len() != dim {
        return Err(Error::validation(format!(
            "vector field {:?} expects {dim} components, got {}",
            field.name,
            items.len()
        )));
    }
    let mut vector = Vec::with_capacity(items.len());
    for item in items {
        let number = item.as_f64().ok_or_else(|| {
            Error::validation(format!(
                "vector field {:?} expects an array of numbers, got component {item}",
                field.name
            ))
        })?;
        let component = number as f32;
        if !component.is_finite() {
            return Err(Error::validation(format!(
                "vector field {:?} has a component outside the f32 range: {item}",
                field.name
            )));
        }
        vector.push(component);
    }
    Ok(vector)
}

/// Type-checks a scalar field value against its schema definition.
fn check_value(field: &FieldSchema, value: &Value) -> Result<()> {
    let mismatch = |expected: &str| {
        Error::validation(format!(
            "field {:?} expects {expected}, got {value}",
            field.name
        ))
    };
    match field.dtype {
        VectorDataType::Varchar => {
            let text = value.as_str().ok_or_else(|| mismatch("a string"))?;
            if let Some(max_length) = field.max_length {
                let length = text.chars().count();
                if length > max_length {
                    return Err(Error::validation(format!(
                        "field {:?} exceeds max_length {max_length}: {length} characters",
                        field.name
                    )));
                }
            }
        }
        VectorDataType::Int64 => {
            value.as_i64().ok_or_else(|| mismatch("an integer"))?;
        }
        VectorDataType::Int32 => {
            let number = value.as_i64().ok_or_else(|| mismatch("a 32-bit integer"))?;
            if i32::try_from(number).is_err() {
                return Err(Error::validation(format!(
                    "field {:?} is out of the INT32 range: {number}",
                    field.name
                )));
            }
        }
        VectorDataType::Float | VectorDataType::Double => {
            if !value.is_number() {
                return Err(mismatch("a number"));
            }
        }
        VectorDataType::Bool => {
            if !value.is_boolean() {
                return Err(mismatch("a boolean"));
            }
        }
        VectorDataType::Array => {
            if !value.is_array() {
                return Err(mismatch("an array"));
            }
        }
        VectorDataType::Json => {}
        VectorDataType::FloatVector => {
            parse_vector(field, value)?;
        }
    }
    Ok(())
}

/// A [`VectorStore`] that keeps collections and documents in process memory.
///
/// The store is the behavioral reference of the vector contract: cosine
/// similarity, insert-only adds, deterministic ordering. Use
/// [`InMemoryVectorStore::search_with_metric`] to search with a non-default
/// [`Metric`].
///
/// ```
/// use openjiuwen_foundation::store::vector::*;
///
/// # #[tokio::main(flavor = "current_thread")]
/// # async fn main() -> openjiuwen_common::error::Result<()> {
/// let store = InMemoryVectorStore::new();
/// let schema = CollectionSchema::new(vec![
///     FieldSchema { is_primary: true, ..FieldSchema::new("id", VectorDataType::Varchar) },
///     FieldSchema { dim: Some(2), ..FieldSchema::new("embedding", VectorDataType::FloatVector) },
/// ]);
/// store.create_collection("docs", schema).await?;
/// let docs = vec![
///     serde_json::json!({"id": "a", "embedding": [1.0, 0.0]}),
///     serde_json::json!({"id": "b", "embedding": [0.0, 1.0]}),
/// ]
/// .into_iter()
/// .filter_map(|doc| doc.as_object().cloned())
/// .collect();
/// assert_eq!(store.add_docs("docs", docs).await?, 2);
/// let hits = store.search("docs", &VectorQuery::new(vec![1.0, 0.0])).await?;
/// assert_eq!(hits[0].fields["id"], "a");
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Default)]
pub struct InMemoryVectorStore {
    index: Mutex<VectorIndex>,
}

impl InMemoryVectorStore {
    /// Creates an empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Searches a collection with an explicit similarity metric.
    ///
    /// This is the metric-aware variant of [`VectorStore::search`] (which uses
    /// [`Metric::Cosine`]); it is synchronous because the index is in memory.
    pub fn search_with_metric(
        &self,
        collection: &str,
        query: &VectorQuery,
        metric: Metric,
    ) -> Result<Vec<VectorSearchResult>> {
        self.index.lock().search(collection, query, metric)
    }
}

#[async_trait]
impl VectorStore for InMemoryVectorStore {
    async fn create_collection(&self, collection: &str, schema: CollectionSchema) -> Result<()> {
        self.index.lock().create_collection(collection, schema)
    }

    async fn delete_collection(&self, collection: &str) -> Result<()> {
        self.index.lock().delete_collection(collection);
        Ok(())
    }

    async fn collection_exists(&self, collection: &str) -> Result<bool> {
        Ok(self.index.lock().collection_exists(collection))
    }

    async fn get_schema(&self, collection: &str) -> Result<Option<CollectionSchema>> {
        Ok(self.index.lock().schema(collection).cloned())
    }

    async fn add_docs(&self, collection: &str, docs: Vec<Map<String, Value>>) -> Result<usize> {
        self.index.lock().add_docs(collection, docs)
    }

    async fn search(
        &self,
        collection: &str,
        query: &VectorQuery,
    ) -> Result<Vec<VectorSearchResult>> {
        self.index.lock().search(collection, query, Metric::Cosine)
    }

    async fn delete_docs_by_ids(&self, collection: &str, ids: &[String]) -> Result<usize> {
        self.index.lock().delete_docs_by_ids(collection, ids)
    }

    async fn delete_docs_by_filters(&self, collection: &str, filters: &Value) -> Result<usize> {
        self.index
            .lock()
            .delete_docs_by_filters(collection, filters)
    }

    async fn list_collection_names(&self) -> Result<Vec<String>> {
        Ok(self.index.lock().collection_names())
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use openjiuwen_common::error::ErrorKind;

    use super::*;

    fn field(name: &str, dtype: VectorDataType) -> FieldSchema {
        FieldSchema {
            name: name.to_string(),
            dtype,
            ..FieldSchema::default()
        }
    }

    fn schema() -> CollectionSchema {
        CollectionSchema::new(vec![
            FieldSchema {
                is_primary: true,
                ..field("id", VectorDataType::Varchar)
            },
            FieldSchema {
                dim: Some(2),
                ..field("embedding", VectorDataType::FloatVector)
            },
            field("kind", VectorDataType::Varchar),
        ])
    }

    /// The JSON object of `value`, for the document helpers below.
    fn obj(value: Value) -> Map<String, Value> {
        value.as_object().cloned().unwrap_or_default()
    }

    fn doc(id: &str, embedding: [f32; 2]) -> Map<String, Value> {
        obj(json!({"id": id, "embedding": [embedding[0], embedding[1]]}))
    }

    fn ids(hits: &[VectorSearchResult]) -> Vec<String> {
        hits.iter()
            .map(|hit| hit.fields["id"].as_str().unwrap_or_default().to_string())
            .collect()
    }

    #[tokio::test]
    async fn create_rejects_invalid_schema_and_duplicate_name() {
        let store = InMemoryVectorStore::new();
        let no_primary = CollectionSchema::new(vec![FieldSchema {
            dim: Some(2),
            ..field("embedding", VectorDataType::FloatVector)
        }]);
        assert_eq!(
            store
                .create_collection("bad", no_primary)
                .await
                .expect_err("no primary key")
                .kind(),
            ErrorKind::Validation
        );
        store
            .create_collection("docs", schema())
            .await
            .expect("create");
        assert_eq!(
            store
                .create_collection("docs", schema())
                .await
                .expect_err("duplicate name")
                .kind(),
            ErrorKind::Validation
        );
        // A rejected create leaves no collection behind.
        assert_eq!(store.list_collection_names().await.expect("list").len(), 1);
    }

    #[tokio::test]
    async fn primary_keys_defaults_and_auto_ids() {
        let store = InMemoryVectorStore::new();
        let auto = CollectionSchema::new(vec![
            FieldSchema {
                is_primary: true,
                auto_id: true,
                ..field("id", VectorDataType::Varchar)
            },
            FieldSchema {
                dim: Some(1),
                ..field("embedding", VectorDataType::FloatVector)
            },
            FieldSchema {
                default_value: Some(json!("unknown")),
                ..field("kind", VectorDataType::Varchar)
            },
        ]);
        store.create_collection("auto", auto).await.expect("create");
        assert_eq!(
            store
                .add_docs("auto", vec![obj(json!({"embedding": [1.0]}))])
                .await
                .expect("add"),
            1
        );
        let hits = store
            .search("auto", &VectorQuery::new(vec![1.0]))
            .await
            .expect("search");
        assert_eq!(hits[0].fields["kind"], "unknown", "default applied");
        let generated = hits[0].fields["id"].as_str().expect("generated id");
        assert_ne!(generated, "");
        assert_eq!(
            store
                .delete_docs_by_ids("auto", &[generated.to_string()])
                .await
                .expect("delete by generated id"),
            1
        );

        // Without `auto_id` a missing or malformed primary key is rejected.
        store
            .create_collection("docs", schema())
            .await
            .expect("create docs");
        let error = store
            .add_docs("docs", vec![obj(json!({"embedding": [1.0, 0.0]}))])
            .await
            .expect_err("missing primary key");
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("primary key"), "{error}");
        for value in [
            json!({"id": true, "embedding": [1.0, 0.0]}),
            json!({"id": "", "embedding": [1.0, 0.0]}),
            json!({"id": 1.5, "embedding": [1.0, 0.0]}),
        ] {
            assert_eq!(
                store
                    .add_docs("docs", vec![obj(value.clone())])
                    .await
                    .expect_err("malformed primary key")
                    .kind(),
                ErrorKind::Validation,
                "{value}"
            );
        }

        // An integer primary key is stored under its string form.
        let int_schema = CollectionSchema::new(vec![
            FieldSchema {
                is_primary: true,
                ..field("id", VectorDataType::Int64)
            },
            FieldSchema {
                dim: Some(1),
                ..field("embedding", VectorDataType::FloatVector)
            },
        ]);
        store
            .create_collection("ints", int_schema)
            .await
            .expect("create ints");
        store
            .add_docs("ints", vec![obj(json!({"id": 7, "embedding": [1.0]}))])
            .await
            .expect("add int id");
        let hits = store
            .search("ints", &VectorQuery::new(vec![1.0]))
            .await
            .expect("search");
        assert_eq!(hits[0].fields["id"], 7);
        assert_eq!(
            store
                .delete_docs_by_ids("ints", &["7".to_string()])
                .await
                .expect("delete by string id"),
            1
        );
    }

    #[tokio::test]
    async fn documents_are_type_checked() {
        let store = InMemoryVectorStore::new();
        let schema = CollectionSchema::new(vec![
            FieldSchema {
                is_primary: true,
                ..field("id", VectorDataType::Varchar)
            },
            FieldSchema {
                dim: Some(2),
                ..field("embedding", VectorDataType::FloatVector)
            },
            field("count", VectorDataType::Int32),
            field("ratio", VectorDataType::Double),
            field("flag", VectorDataType::Bool),
            field("tags", VectorDataType::Array),
            field("meta", VectorDataType::Json),
            FieldSchema {
                max_length: Some(3),
                ..field("label", VectorDataType::Varchar)
            },
        ]);
        store
            .create_collection("docs", schema)
            .await
            .expect("create");

        let bad = [
            json!({"id": "a", "embedding": [1.0, 0.0], "count": 1.5}),
            json!({"id": "a", "embedding": [1.0, 0.0], "count": 9_000_000_000_i64}),
            json!({"id": "a", "embedding": [1.0, 0.0], "ratio": "high"}),
            json!({"id": "a", "embedding": [1.0, 0.0], "flag": 1}),
            json!({"id": "a", "embedding": [1.0, 0.0], "tags": "one"}),
            json!({"id": "a", "embedding": [1.0, 0.0], "label": "toolong"}),
            json!({"id": "a", "embedding": [1.0, 0.0], "label": 4}),
            json!({"id": "a", "embedding": [1.0]}),
            json!({"id": "a", "embedding": [1.0, "two"]}),
            json!({"id": "a", "embedding": "vector"}),
            json!({"id": "a"}),
        ];
        for value in bad {
            assert_eq!(
                store
                    .add_docs("docs", vec![obj(value.clone())])
                    .await
                    .expect_err("invalid document")
                    .kind(),
                ErrorKind::Validation,
                "{value}"
            );
        }

        // A well-typed document is accepted, including unknown fields.
        let good = json!({
            "id": "a",
            "embedding": [1.0, 0.0],
            "count": 7,
            "ratio": 0.5,
            "flag": true,
            "tags": ["x", "y"],
            "meta": {"nested": [1, 2]},
            "label": "ok",
            "unknown": {"kept": true},
        });
        assert_eq!(
            store
                .add_docs("docs", vec![obj(good)])
                .await
                .expect("valid"),
            1
        );
        let hits = store
            .search("docs", &VectorQuery::new(vec![1.0, 0.0]))
            .await
            .expect("search");
        assert_eq!(hits[0].fields["unknown"], json!({"kept": true}));
        assert_eq!(hits[0].fields["embedding"], json!([1.0, 0.0]));
    }

    #[tokio::test]
    async fn search_orders_by_score_then_primary_key() {
        let store = InMemoryVectorStore::new();
        store
            .create_collection("docs", schema())
            .await
            .expect("create");
        store
            .add_docs(
                "docs",
                [
                    doc("a", [1.0, 0.0]),
                    doc("b", [0.0, 1.0]),
                    doc("c", [-1.0, 0.0]),
                    doc("d", [1.0, 1.0]),
                ]
                .to_vec(),
            )
            .await
            .expect("add");
        // Cosine similarity to [1, 0]: a = 1.0, d = 0.8536, b = 0.5, c = 0.0.
        // The default `top_k` is five, which exceeds the four documents.
        let hits = store
            .search("docs", &VectorQuery::new(vec![1.0, 0.0]))
            .await
            .expect("search");
        assert_eq!(ids(&hits), ["a", "d", "b", "c"]);
        assert!(
            (hits[0].score - 1.0).abs() < 1e-6,
            "identical vector is 1.0"
        );
        assert!(hits.windows(2).all(|pair| pair[0].score >= pair[1].score));
        assert!(hits.iter().all(|hit| hit.score.is_finite()));
        assert!(hits.iter().all(|hit| (0.0..=1.0).contains(&hit.score)));

        let hits = store
            .search("docs", &VectorQuery::new(vec![1.0, 0.0]).with_top_k(2))
            .await
            .expect("search");
        assert_eq!(ids(&hits), ["a", "d"]);

        // Equal scores are broken by primary key, ascending.
        store
            .add_docs(
                "docs",
                ["z", "m"]
                    .into_iter()
                    .map(|id| doc(id, [1.0, 0.0]))
                    .collect(),
            )
            .await
            .expect("add ties");
        let hits = store
            .search(
                "docs",
                &VectorQuery::new(vec![1.0, 0.0])
                    .with_top_k(2)
                    .with_filters(json!({"id": {"$in": ["m", "z"]}})),
            )
            .await
            .expect("search ties");
        assert_eq!(ids(&hits), ["m", "z"]);
    }

    #[tokio::test]
    async fn search_validates_vector_field_and_dimension() {
        let store = InMemoryVectorStore::new();
        store
            .create_collection("docs", schema())
            .await
            .expect("create");
        store
            .add_docs("docs", vec![doc("a", [1.0, 0.0])])
            .await
            .expect("add");
        for query in [
            VectorQuery::new(vec![1.0, 0.0, 0.0]),
            VectorQuery::new(vec![1.0, 0.0]).with_vector_field("kind"),
            VectorQuery::new(vec![1.0, 0.0]).with_vector_field("missing"),
        ] {
            assert_eq!(
                store
                    .search("docs", &query)
                    .await
                    .expect_err("invalid query")
                    .kind(),
                ErrorKind::Validation,
                "{query:?}"
            );
        }
    }

    #[tokio::test]
    async fn search_with_metric_supports_every_metric() {
        let store = InMemoryVectorStore::new();
        store
            .create_collection("docs", schema())
            .await
            .expect("create");
        store
            .add_docs(
                "docs",
                vec![
                    obj(json!({"id": "near", "embedding": [1.0, 0.0]})),
                    obj(json!({"id": "far", "embedding": [-1.0, 0.0]})),
                ],
            )
            .await
            .expect("add");
        let query = VectorQuery::new(vec![1.0, 0.0]);
        for metric in [Metric::Cosine, Metric::Dot, Metric::Euclidean] {
            let hits = store
                .search_with_metric("docs", &query, metric)
                .expect("search");
            assert_eq!(ids(&hits), ["near", "far"], "{metric:?}");
            // `Dot` is squashed through `tanh`, so only cosine and euclidean
            // reach 1.0 for the identical vector; every metric keeps the order.
            if metric == Metric::Cosine || metric == Metric::Euclidean {
                assert!((hits[0].score - 1.0).abs() < 1e-6, "{metric:?}");
            } else {
                assert!(hits[0].score > 0.5, "{metric:?}");
            }
            assert!(hits.iter().all(|hit| hit.score.is_finite()));
        }
    }

    #[tokio::test]
    async fn index_snapshot_round_trips_and_validates() {
        let store = InMemoryVectorStore::new();
        store
            .create_collection("docs", schema())
            .await
            .expect("create");
        store
            .add_docs("docs", vec![doc("a", [1.0, 0.0]), doc("b", [0.0, 1.0])])
            .await
            .expect("add");

        let snapshot = store.index.lock().to_value().expect("snapshot");
        let restored = VectorIndex::from_value(&snapshot).expect("restore");
        assert_eq!(restored.collection_names(), ["docs".to_string()]);
        assert_eq!(restored.schema("docs").expect("schema"), &schema());
        let hits = restored
            .search("docs", &VectorQuery::new(vec![1.0, 0.0]), Metric::Cosine)
            .expect("search");
        assert_eq!(ids(&hits), ["a", "b"]);
        assert_eq!(
            restored.collection_value("docs").expect("collection"),
            Some(snapshot["collections"]["docs"].clone())
        );
        assert_eq!(restored.collection_value("missing").expect("missing"), None);

        let broken = json!({"collections": {"docs": {"schema": {"fields": []}, "docs": {}}}});
        assert_eq!(
            VectorIndex::from_value(&broken)
                .expect_err("invalid schema")
                .kind(),
            ErrorKind::Validation
        );
        assert!(VectorIndex::from_value(&json!({"collections": 1})).is_err());
    }
}
