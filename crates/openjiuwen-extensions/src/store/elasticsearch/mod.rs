//! Live Elasticsearch vector backend.
//!
//! Ports `openjiuwen/extensions/store/vector/es_vector_store.py`
//! (`ElasticsearchVectorStore`): a collection is an Elasticsearch index named
//! `"{index_prefix}__{collection}"`, dense vectors are indexed `dense_vector`
//! fields, search is a k-NN query, and documents are added through the bulk
//! API. The transport is plain REST over `reqwest` (already a dependency of this
//! crate), so no Elasticsearch client crate is added and the default build still
//! needs no network.
//!
//! [`ElasticsearchVectorStore::new`] and
//! [`ElasticsearchVectorStore::connect`] validate the configuration without any
//! I/O; every network call happens inside a [`VectorStore`] operation.
//!
//! # Collections and schemas
//!
//! [`CollectionSchema::validate`] runs before anything is sent. The index
//! mapping stores the whole schema in `_meta.openjiuwen_schema`, which is what
//! [`ElasticsearchVectorStore::get_schema`] reads back; an index created by
//! another tool is reconstructed from its `properties` (the `id` property, or
//! else the first non-vector property, becomes the primary key). This replaces
//! Python's `_collection_metadata__` document.
//!
//! Creating an existing index is a validation error (matching the in-memory
//! contract), `delete_collection` treats a missing index as a no-op, and the
//! other operations report a missing collection as
//! [`ErrorKind::NotFound`](openjiuwen_common::error::ErrorKind::NotFound).
//!
//! # Documents and search
//!
//! `add_docs` validates the whole batch locally before sending one `_bulk`
//! request: it fills `default_value`s, mints a UUID for an omitted `auto_id`
//! primary key, rejects duplicate primary keys, and type/dimension-checks every
//! field, so a rejected batch inserts nothing. `search` resolves the vector
//! field (the query's field, else the schema's first `FLOAT_VECTOR` field),
//! checks the query length against its `dim`, sends a `knn` query with the
//! translated filter, then reapplies the kernel ordering (score descending,
//! primary key ascending) and truncates to `top_k`.
//!
//! # Deviations from the in-memory reference
//!
//! - Single node only: no sniffing, no cloud id, and one `_bulk` request per
//!   `add_docs` call (no batching knob); no scroll/PIT/`search_after`, no index
//!   templates or ILM management.
//! - There is no per-collection metric registry ([`CollectionSchema`] carries no
//!   metric field), so every index is created with cosine similarity.
//!   Elasticsearch's native `_score` is `(1 + cos) / 2` for `cosine` and
//!   `1 / (1 + distance)` for `l2_norm`, matching the kernel `Metric`
//!   normalization, while `dot_product` scores `(1 + dot) / 2` instead of the
//!   reference's `(tanh(dot) + 1) / 2`; scores are clamped into `[0, 1]`.
//! - Elasticsearch indexes an array field element-by-element, so array
//!   equality is translated as `terms` membership (a documented approximation).
//!   `$contains`, object equality, equality to `null`, and non-scalar `terms`
//!   candidates have no faithful equivalent and return
//!   [`ErrorKind::Validation`](openjiuwen_common::error::ErrorKind::Validation)
//!   pointing at the in-memory backend; the `client` module documents the exact
//!   filter subset.
//! - The bulk `index` action overwrites a document that already holds the same
//!   id, so "insert-only against previously stored documents" cannot be
//!   enforced (duplicates inside one batch still fail with a validation error).
//! - An array-index filter path (`items[1].kind`) is passed through as an
//!   Elasticsearch field path, which Elasticsearch may not resolve.
//!
//! Python counterpart tests:
//! `tests/unit_tests/extensions/store/test_es_vector_store.py`.
#![allow(clippy::module_name_repetitions)] // `Elasticsearch*` names are fixed by the contract.

use std::collections::BTreeSet;
use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_core::foundation::store::{
    CollectionSchema, FieldSchema, VectorDataType, VectorQuery, VectorSearchResult, VectorStore,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use url::Url;
use uuid::Uuid;

pub(crate) mod client;
pub(crate) mod credentials;
#[cfg(test)]
mod tests;

use client::RestClient;
use credentials::{endpoint_without_userinfo, ElasticsearchCredentials};

/// Default index prefix, matching Python's `index_prefix="agent_vector"`.
const DEFAULT_INDEX_PREFIX: &str = "agent_vector";
/// Default vector dimension documented by the Python backend.
const DEFAULT_DIMENSION: usize = 768;

/// Configuration for an [`ElasticsearchVectorStore`].
///
/// Mirrors the Python constructor's inputs — the endpoint, the index-name
/// prefix, and the vector dimension — and adds the credential fields the kernel
/// document store already accepts, so one backend description authenticates
/// both adapters. The defaults are Python's: an empty URL, the `"agent_vector"`
/// prefix, and `768` dimensions, with no credentials.
///
/// Credentials may come from the URL userinfo (`http://user:pass@host:9200`) or
/// from the explicit [`username`](Self::username) /
/// [`password`](Self::password) fields; when both are present the explicit
/// fields win, field by field. An [`api_key`](Self::api_key) is sent as
/// `Authorization: ApiKey <key>` when no basic-auth pair applies. See the
/// `credentials` module for the exact rules.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, rename_all = "snake_case")]
pub struct ElasticsearchVectorStoreConfig {
    /// Elasticsearch endpoint; it must start with `http://` or `https://`.
    pub url: String,
    /// Prefix of every index this store owns.
    pub index_prefix: String,
    /// Dimension of the stored dense vectors; must be greater than zero.
    pub dimension: usize,
    /// Optional basic-auth user name, overriding the URL userinfo.
    #[serde(default)]
    pub username: Option<String>,
    /// Optional basic-auth password, overriding the URL userinfo.
    #[serde(default)]
    pub password: Option<String>,
    /// Optional API key, sent when no basic-auth pair applies.
    #[serde(default)]
    pub api_key: Option<String>,
}

impl Default for ElasticsearchVectorStoreConfig {
    fn default() -> Self {
        Self {
            url: String::new(),
            index_prefix: DEFAULT_INDEX_PREFIX.to_string(),
            dimension: DEFAULT_DIMENSION,
            username: None,
            password: None,
            api_key: None,
        }
    }
}

/// An Elasticsearch-backed [`VectorStore`].
///
/// Construction validates the configuration and names indices like Python's
/// backend; every operation talks to the configured endpoint over REST.
#[derive(Debug, Clone)]
pub struct ElasticsearchVectorStore {
    config: ElasticsearchVectorStoreConfig,
    rest: RestClient,
}

impl ElasticsearchVectorStore {
    /// Validates `config` and builds a store.
    ///
    /// No network I/O happens here: the endpoint is parsed, credentials are
    /// resolved by `ElasticsearchCredentials::resolve` (URL userinfo plus the
    /// explicit configuration fields), and the endpoint is stripped of userinfo,
    /// query, fragment, and trailing slash.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::Validation`](openjiuwen_common::error::ErrorKind::Validation)
    /// when the URL is empty, does not start with `http://`/`https://`, or is
    /// not a parseable URL, or when `dimension` is zero.
    pub fn new(mut config: ElasticsearchVectorStoreConfig) -> Result<Self> {
        config.url = config.url.trim().to_string();
        if config.url.is_empty() {
            return Err(Error::validation(
                "elasticsearch vector store requires a non-empty `url`",
            ));
        }
        if !(config.url.starts_with("http://") || config.url.starts_with("https://")) {
            return Err(Error::validation(format!(
                "elasticsearch url `{}` must start with http:// or https://",
                config.url
            )));
        }
        config.url = config.url.trim_end_matches('/').to_string();
        if config.dimension == 0 {
            return Err(Error::validation(
                "elasticsearch vector dimension must be greater than zero",
            ));
        }
        let parsed = Url::parse(&config.url).map_err(|error| {
            Error::validation(format!(
                "elasticsearch url `{}` is not a valid URL: {error}",
                config.url
            ))
        })?;
        if parsed.host_str().is_none() {
            return Err(Error::validation(format!(
                "elasticsearch url `{}` must name a host",
                config.url
            )));
        }
        let credentials = ElasticsearchCredentials::resolve(
            &parsed,
            config.username.as_deref(),
            config.password.as_deref(),
            config.api_key.as_deref(),
        );
        let rest = RestClient::new(endpoint_without_userinfo(&parsed), credentials)?;
        Ok(Self { config, rest })
    }

    /// Validates `config` and returns a live REST-backed store.
    ///
    /// The asynchronous signature is kept even though connecting performs no
    /// I/O, so call sites match the other backends' factories.
    ///
    /// # Errors
    ///
    /// Returns the same validation errors as
    /// [`ElasticsearchVectorStore::new`].
    // No I/O at connect time; the factory shape is shared with the other backends.
    #[allow(clippy::unused_async, clippy::unused_async_trait_impl)]
    pub async fn connect(config: ElasticsearchVectorStoreConfig) -> Result<Arc<dyn VectorStore>> {
        let store: Arc<dyn VectorStore> = Arc::new(Self::new(config)?);
        Ok(store)
    }

    /// The validated configuration this store was built from.
    #[must_use]
    pub fn config(&self) -> &ElasticsearchVectorStoreConfig {
        &self.config
    }

    /// The index name for `collection`: `"{index_prefix}__{collection}"`.
    #[must_use]
    pub fn index_name(&self, collection: &str) -> String {
        format!("{}__{collection}", self.config.index_prefix)
    }

    /// The schema of `collection`, or [`ErrorKind::NotFound`] when the index is
    /// missing.
    async fn schema_or_not_found(&self, collection: &str) -> Result<CollectionSchema> {
        self.get_schema(collection)
            .await?
            .ok_or_else(|| Error::not_found(format!("vector collection {collection:?} not found")))
    }
}

#[async_trait]
impl VectorStore for ElasticsearchVectorStore {
    async fn create_collection(&self, collection: &str, schema: CollectionSchema) -> Result<()> {
        schema.validate()?;
        let body = client::mapping_body(&schema, self.config.dimension)?;
        self.rest
            .create_index(collection, &self.index_name(collection), &body)
            .await
    }

    async fn delete_collection(&self, collection: &str) -> Result<()> {
        self.rest
            .delete_index(collection, &self.index_name(collection))
            .await
    }

    async fn collection_exists(&self, collection: &str) -> Result<bool> {
        self.rest.index_exists(&self.index_name(collection)).await
    }

    async fn get_schema(&self, collection: &str) -> Result<Option<CollectionSchema>> {
        let index = self.index_name(collection);
        let Some(mappings) = self.rest.mapping(&index).await? else {
            return Ok(None);
        };
        client::schema_from_mappings(&mappings)
            .map(Some)
            .ok_or_else(|| {
                Error::transport(format!(
                    "elasticsearch index {index:?} has no readable collection mapping"
                ))
            })
    }

    async fn add_docs(&self, collection: &str, docs: Vec<Map<String, Value>>) -> Result<usize> {
        let schema = self.schema_or_not_found(collection).await?;
        let prepared = prepare_docs(&schema, docs)?;
        if prepared.is_empty() {
            return Ok(0);
        }
        let index = self.index_name(collection);
        self.rest.bulk_index(&index, &prepared).await
    }

    async fn search(
        &self,
        collection: &str,
        query: &VectorQuery,
    ) -> Result<Vec<VectorSearchResult>> {
        let schema = self.schema_or_not_found(collection).await?;
        let field = resolve_vector_field(&schema, collection, query)?;
        let dim = field.dim.unwrap_or(0);
        if query.query_vector.len() != dim {
            return Err(Error::validation(format!(
                "query vector dimension {} does not match vector field {:?} dimension {dim} of collection {collection:?}",
                query.query_vector.len(),
                field.name
            )));
        }
        if query.top_k == 0 {
            return Ok(Vec::new());
        }
        let top_k = query.top_k;
        let mut knn = Map::new();
        knn.insert("field".to_string(), Value::String(field.name.clone()));
        knn.insert(
            "query_vector".to_string(),
            Value::Array(
                query
                    .query_vector
                    .iter()
                    .map(|component| Value::from(*component))
                    .collect(),
            ),
        );
        knn.insert("k".to_string(), Value::from(top_k));
        knn.insert(
            "num_candidates".to_string(),
            Value::from(top_k.saturating_mul(10).max(100)),
        );
        if !filter_is_empty(&query.filters) {
            knn.insert(
                "filter".to_string(),
                client::translate_filter(&query.filters)?,
            );
        }
        let body = serde_json::json!({ "knn": knn, "size": top_k });
        let hits = self
            .rest
            .knn_search(&self.index_name(collection), &body)
            .await?;
        let primary = schema.primary_key().map(|field| field.name.clone());
        let primary_name = primary.as_deref();
        let mut results: Vec<VectorSearchResult> = hits
            .into_iter()
            .map(|hit| {
                let mut fields = hit.source;
                fields.remove("_meta");
                let missing = primary_name.filter(|name| !fields.contains_key(*name));
                if let Some(name) = missing {
                    fields.insert(name.to_string(), Value::String(hit.id));
                }
                VectorSearchResult::new(normalize_score(hit.score), fields)
            })
            .collect();
        order_results(&mut results, primary_name, top_k);
        Ok(results)
    }

    async fn delete_docs_by_ids(&self, collection: &str, ids: &[String]) -> Result<usize> {
        let schema = self.schema_or_not_found(collection).await?;
        if ids.is_empty() {
            return Ok(0);
        }
        let primary = schema
            .primary_key()
            .ok_or_else(|| Error::validation("collection schema declares no primary key field"))?;
        let candidates = ids
            .iter()
            .cloned()
            .map(Value::String)
            .collect::<Vec<Value>>();
        let query = serde_json::json!({ "terms": { primary.name.clone(): candidates } });
        self.rest
            .delete_by_query(&self.index_name(collection), &query)
            .await
    }

    async fn delete_docs_by_filters(&self, collection: &str, filters: &Value) -> Result<usize> {
        // Check existence first so an unknown collection is a `NotFound`, exactly
        // like the in-memory reference.
        let _schema = self.schema_or_not_found(collection).await?;
        let query = client::translate_filter(filters)?;
        self.rest
            .delete_by_query(&self.index_name(collection), &query)
            .await
    }

    async fn list_collection_names(&self) -> Result<Vec<String>> {
        self.rest.index_names(&self.config.index_prefix).await
    }
}

/// Whether a filter matches every document and therefore need not be sent.
fn filter_is_empty(filters: &Value) -> bool {
    match filters {
        Value::Null => true,
        Value::Object(entries) => entries.is_empty(),
        _ => false,
    }
}

/// Clamps Elasticsearch's native `_score` into the kernel's `[0, 1]` range.
#[allow(clippy::cast_possible_truncation)] // The value is clamped to `[0, 1]` first.
pub(crate) fn normalize_score(score: f64) -> f32 {
    if score.is_nan() {
        return 0.0;
    }
    score.clamp(0.0, 1.0) as f32
}

/// Applies the kernel result order (score descending, primary key ascending)
/// and truncates to `top_k`.
pub(crate) fn order_results(
    results: &mut Vec<VectorSearchResult>,
    primary: Option<&str>,
    top_k: usize,
) {
    let sort_key = |fields: &Map<String, Value>| -> String {
        primary
            .and_then(|name| fields.get(name))
            .map(value_sort_key)
            .unwrap_or_default()
    };
    results.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| sort_key(&left.fields).cmp(&sort_key(&right.fields)))
    });
    results.truncate(top_k);
}

/// The string form of a primary-key value used to break score ties.
fn value_sort_key(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        Value::Number(number) => number.to_string(),
        Value::Bool(flag) => flag.to_string(),
        other => other.to_string(),
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

/// Validates and converts a batch of raw documents into `(primary key, source)`
/// pairs; nothing is sent until the whole batch is valid.
fn prepare_docs(
    schema: &CollectionSchema,
    docs: Vec<Map<String, Value>>,
) -> Result<Vec<(String, Map<String, Value>)>> {
    let primary = schema
        .primary_key()
        .ok_or_else(|| Error::validation("collection schema declares no primary key field"))?;
    let mut prepared = Vec::with_capacity(docs.len());
    let mut seen: BTreeSet<String> = BTreeSet::new();
    for (position, raw) in docs.into_iter().enumerate() {
        let (id, document) = prepare_doc(schema, primary, raw, position)?;
        if !seen.insert(id.clone()) {
            return Err(Error::validation(format!(
                "duplicate primary key {id:?} in the batch (position {position})"
            )));
        }
        prepared.push((id, document));
    }
    Ok(prepared)
}

/// Validates one document, fills defaults, and mints an `auto_id` primary key.
fn prepare_doc(
    schema: &CollectionSchema,
    primary: &FieldSchema,
    mut document: Map<String, Value>,
    position: usize,
) -> Result<(String, Map<String, Value>)> {
    for field in &schema.fields {
        let default = field
            .default_value
            .as_ref()
            .filter(|_| !document.contains_key(&field.name));
        if let Some(default) = default {
            document.insert(field.name.clone(), default.clone());
        }
    }
    let id = match document.get(&primary.name) {
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
            document.insert(primary.name.clone(), Value::String(generated.clone()));
            generated
        }
        None => {
            return Err(Error::validation(format!(
                "document at position {position} is missing primary key field {:?}",
                primary.name
            )));
        }
    };
    for field in &schema.fields {
        match document.get(&field.name) {
            Some(value) => validate_value(field, value)?,
            None if field.is_vector() => {
                return Err(Error::validation(format!(
                    "document {id:?} is missing vector field {:?}",
                    field.name
                )));
            }
            None => {}
        }
    }
    Ok((id, document))
}

/// Type-checks one field value, mirroring the in-memory reference.
fn validate_value(field: &FieldSchema, value: &Value) -> Result<()> {
    let mismatch = |expected: &str| {
        Error::validation(format!(
            "field {:?} expects {expected}, got {value}",
            field.name
        ))
    };
    match field.dtype {
        VectorDataType::FloatVector => {
            validate_vector(field, value)?;
        }
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
    }
    Ok(())
}

/// Checks one vector field value's shape, dimension, and component range.
fn validate_vector(field: &FieldSchema, value: &Value) -> Result<()> {
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
    for item in items {
        let number = item.as_f64().ok_or_else(|| {
            Error::validation(format!(
                "vector field {:?} expects an array of numbers, got component {item}",
                field.name
            ))
        })?;
        if !number.is_finite() {
            return Err(Error::validation(format!(
                "vector field {:?} has a non-finite component: {item}",
                field.name
            )));
        }
    }
    Ok(())
}
