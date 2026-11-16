//! Elasticsearch vector store over the `dense_vector` field type.
//!
//! [`ElasticsearchVectorStore`] implements the [`VectorStore`] contract of
//! [`crate::store::vector`] on top of an Elasticsearch index: one collection is
//! one index named `{index_prefix}__{collection}`, one document is one index
//! document, and a search is a `knn` query over the collection's vector field.
//!
//! # Wire contract
//!
//! Every request goes through the shared `ElasticsearchClient` of [`super`], so
//! it uses [`openjiuwen_common::http::HttpClient`], carries the configured
//! credentials, and maps a non-2xx response to
//! [`Error::Transport`](openjiuwen_common::error::Error) naming the status code,
//! the URL and an excerpt of the body.
//!
//! - `create_collection` sends `PUT /{index}` with the `properties` mapping
//!   derived from the schema plus the serialized schema under
//!   `_meta.openjiuwen_schema`, so the schema round-trips exactly. A
//!   `FLOAT_VECTOR` field becomes a `dense_vector` mapping whose `dims` is the
//!   field's `dim` and whose `similarity` follows
//!   [`ElasticsearchConfig::metric`](super::ElasticsearchConfig::metric).
//!   Elasticsearch cannot express a top-level `ARRAY` field, so such a schema is
//!   rejected with [`Error::Validation`](openjiuwen_common::error::Error) before
//!   any request is sent.
//! - `add_docs` sends one `POST /{index}/_bulk` request whose NDJSON body holds
//!   two lines per document: a `create` action and the document `_source`.
//!   **The `_id` of a document is its primary-key value**, while `_source`
//!   still carries the primary-key field, so a search result is complete even
//!   when Elasticsearch returns no `_id`.
//! - `search` sends `POST /{index}/_search` with a `knn` clause; the returned
//!   `_score` is used as-is and clamped into `[0, 1]`, because Elasticsearch
//!   reports `cosine` and `dot_product` as a similarity in `[0, 1]` and
//!   `l2_norm` as `1 / (1 + d)` — exactly the "higher is better" convention of
//!   [`VectorSearchResult`]. A filter the service cannot express is left out of
//!   the clause, the candidate window is widened, and the hits are re-checked
//!   locally; that fallback only ever sees the bounded window.
//! - `delete_docs_by_filters` uses `POST /{index}/_delete_by_query` when the
//!   filter translates into query DSL ([`super::query::translate`]); otherwise
//!   it pages `_search` with `match_all` and re-checks every candidate with
//!   [`crate::store::filter::matches`] locally. The fallback can only see that
//!   bounded candidate window, so a filter Elasticsearch cannot express is never
//!   silently dropped but may miss documents beyond the window.
//!
//! # Differences from the reference backend
//!
//! - Unknown document fields are rejected unless the schema sets
//!   `enable_dynamic_field`: Elasticsearch would accept them (its indexes are
//!   dynamic by default), so this adapter is deliberately stricter than the
//!   service, matching the in-memory reference and the Chroma adapter.
//! - Insert-only adds are enforced by the service: a `create` action on an
//!   existing `_id` fails and becomes
//!   [`Error::Validation`](openjiuwen_common::error::Error).
//! - A `null` document value is omitted from the stored document instead of
//!   being type-checked.

use std::collections::{BTreeMap, BTreeSet};

use async_trait::async_trait;
use serde_json::{json, Map, Value};
use uuid::Uuid;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::http::{HttpClient, HttpRequest};

use super::query::translate;
use super::{
    create_index_body, failure, index_names_from, schema_from_index, stable_sort,
    ElasticsearchClient, ElasticsearchConfig, SearchScan, LOG_TARGET, MAX_SCAN_DOCUMENTS,
};
use crate::store::vector::{
    collection_not_found, CollectionSchema, FieldSchema, VectorDataType, VectorQuery,
    VectorSearchResult, VectorStore,
};

/// Content type of an Elasticsearch bulk request body.
const NDJSON_CONTENT_TYPE: &str = "application/x-ndjson";

/// The bulk action used to insert documents, so a duplicate `_id` fails.
const CREATE_ACTION: &str = "create";

/// The bulk action used to delete documents by `_id`.
const DELETE_ACTION: &str = "delete";

/// A [`VectorStore`] backed by an Elasticsearch index.
///
/// The store holds one shared `ElasticsearchClient`, so it is cheap to clone
/// and share behind an `Arc`, and every method takes `&self`.
///
/// # Errors
///
/// [`ElasticsearchVectorStore::new`] rejects an unusable
/// [`ElasticsearchConfig`]; the contract methods return the errors documented on
/// [`VectorStore`], plus the service failures of
/// [`Error::Transport`](openjiuwen_common::error::Error).
#[derive(Debug, Clone)]
pub struct ElasticsearchVectorStore {
    client: ElasticsearchClient,
}

impl ElasticsearchVectorStore {
    /// Builds a store from `config`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`](openjiuwen_common::error::Error) for an empty
    /// endpoint or index prefix and propagates the HTTP client's configuration
    /// errors.
    pub fn new(config: ElasticsearchConfig) -> Result<Self> {
        Ok(Self {
            client: ElasticsearchClient::new(config)?,
        })
    }

    /// Builds a store that sends every request through `http`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`](openjiuwen_common::error::Error) for an empty
    /// endpoint or index prefix.
    pub fn with_client(http: HttpClient, config: ElasticsearchConfig) -> Result<Self> {
        Ok(Self {
            client: ElasticsearchClient::with_client(http, config)?,
        })
    }

    /// The configuration this store was built from.
    #[must_use]
    pub fn config(&self) -> &ElasticsearchConfig {
        self.client.config()
    }

    /// Reads the schema of `collection`, or `None` when its index is missing.
    async fn fetch_schema(&self, collection: &str) -> Result<Option<CollectionSchema>> {
        let index = self.client.index_name(collection)?;
        let request = HttpRequest::get(self.client.url(&self.client.path(&index, "")));
        let response = self.client.send_raw(request).await?;
        match response.status {
            404 => Ok(None),
            200..=299 => {
                let payload: Value = response.json()?;
                schema_from_index(&payload)
            }
            _ => Err(failure(&response, "reading the collection schema")),
        }
    }

    /// Loads the schema of `collection` or reports it as missing.
    async fn require_schema(&self, collection: &str) -> Result<CollectionSchema> {
        self.fetch_schema(collection)
            .await?
            .ok_or_else(|| collection_not_found(collection))
    }

    /// Sends one NDJSON bulk request to `index` and returns the reply body.
    async fn send_bulk(&self, index: &str, body: String, context: &str) -> Result<Value> {
        let request = HttpRequest::post(self.client.url(&self.client.path(index, "/_bulk")))
            .header("content-type", NDJSON_CONTENT_TYPE)
            .text(body);
        let response = self.client.send(request, context).await?;
        response.json()
    }

    /// Sends one `_search` request to `index` and returns the reply body.
    async fn send_search(&self, index: &str, body: &Value) -> Result<Value> {
        let request =
            HttpRequest::post(self.client.url(&self.client.path(index, "/_search"))).json(body)?;
        let response = self.client.send(request, "searching collection").await?;
        response.json()
    }

    /// Collects, page by page, the ids of the documents `filters` matches.
    ///
    /// The service cannot express `filters`, so the scan reads the whole index
    /// with `{"match_all": {}}` through the shared [`SearchScan`] — at most
    /// [`SEARCH_PAGE_SIZE`](super::SEARCH_PAGE_SIZE) documents per request,
    /// sorted by the collection's keyword primary key and continued from the
    /// previous page's `sort` values — and re-checks every hit locally with
    /// [`crate::store::filter::matches`].
    ///
    /// # Errors
    ///
    /// `Error::Validation` when the schema declares no primary key or the scan
    /// stops advancing, and `Error::Unsupported` — after a `tracing::warn!`
    /// naming the cap and the filter — when the collection needs more than
    /// [`MAX_SCAN_DOCUMENTS`] documents, rather than deleting a truncated set.
    async fn collect_matching_ids(
        &self,
        index: &str,
        schema: &CollectionSchema,
        filters: &Value,
    ) -> Result<BTreeSet<String>> {
        let primary = primary_key_field(schema)?;
        tracing::warn!(
            target: LOG_TARGET,
            index = %index,
            filter = %filters,
            cap = MAX_SCAN_DOCUMENTS,
            "a filter Elasticsearch cannot express falls back to a local scan",
        );
        let mut collected: BTreeSet<String> = BTreeSet::new();
        self.client
            .scan(
                SearchScan {
                    index,
                    query: json!({"match_all": {}}),
                    sort: stable_sort(&[], &primary.name),
                    needed: None,
                    filter: filters,
                },
                |page| {
                    for hit in page {
                        if crate::store::filter::matches(filters, &Value::Object(hit.source)) {
                            collected.insert(hit.id);
                        }
                    }
                    Ok(())
                },
            )
            .await?;
        Ok(collected)
    }

    /// Deletes the documents `keys` with one bulk request, counting the removals.
    async fn delete_keys(&self, index: &str, keys: &BTreeSet<String>) -> Result<usize> {
        let body = bulk_delete_body(keys)?;
        let payload = self.send_bulk(index, body, "deleting documents").await?;
        let mut deleted = 0usize;
        for item in bulk_items(&payload, DELETE_ACTION) {
            if let Some(error) = &item.error {
                return Err(Error::transport(format!(
                    "Elasticsearch bulk delete from index {index} failed for document \
                     {:?} with status {}: {}: {}",
                    item.id, item.status, error.kind, error.reason
                )));
            }
            if item.result.as_deref() == Some("deleted") {
                deleted += 1;
            }
        }
        Ok(deleted)
    }
}

#[async_trait]
impl VectorStore for ElasticsearchVectorStore {
    async fn create_collection(&self, collection: &str, schema: CollectionSchema) -> Result<()> {
        schema.validate()?;
        let index = self.client.index_name(collection)?;
        let body = create_index_body(&schema, self.client.config().metric)?;
        let request =
            HttpRequest::put(self.client.url(&self.client.path(&index, ""))).json(&body)?;
        let response = self.client.send_raw(request).await?;
        if response.is_success() {
            return Ok(());
        }
        let text = response.text();
        if response.status == 400
            && (text.contains("resource_already_exists_exception")
                || text.to_lowercase().contains("already exists"))
        {
            return Err(Error::validation(format!(
                "vector collection {collection:?} already exists"
            )));
        }
        Err(failure(&response, "creating collection"))
    }

    async fn delete_collection(&self, collection: &str) -> Result<()> {
        let index = self.client.index_name(collection)?;
        let request = HttpRequest::delete(self.client.url(&self.client.path(&index, "")));
        let response = self.client.send_raw(request).await?;
        if matches!(response.status, 404 | 200..=299) {
            return Ok(());
        }
        Err(failure(&response, "deleting collection"))
    }

    async fn collection_exists(&self, collection: &str) -> Result<bool> {
        let index = self.client.index_name(collection)?;
        let request = HttpRequest::head(self.client.url(&self.client.path(&index, "")));
        let response = self.client.send_raw(request).await?;
        match response.status {
            404 => Ok(false),
            200..=299 => Ok(true),
            _ => Err(failure(&response, "checking collection")),
        }
    }

    async fn get_schema(&self, collection: &str) -> Result<Option<CollectionSchema>> {
        self.fetch_schema(collection).await
    }

    async fn add_docs(&self, collection: &str, docs: Vec<Map<String, Value>>) -> Result<usize> {
        let index = self.client.index_name(collection)?;
        let schema = self.require_schema(collection).await?;
        let prepared = prepare_docs(&schema, docs)?;
        let body = bulk_create_body(&prepared)?;
        let payload = self.send_bulk(&index, body, "adding documents").await?;
        if !payload
            .get("errors")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            return Ok(prepared.len());
        }
        for item in bulk_items(&payload, CREATE_ACTION) {
            match &item.error {
                // A malformed item without an explicit status counts as created.
                None if item.status < 300 => {}
                None => {
                    return Err(Error::transport(format!(
                        "Elasticsearch bulk insert into index {index} failed for document \
                         {:?} with status {}",
                        item.id, item.status
                    )));
                }
                Some(error) => {
                    if item.status == 409
                        || matches!(
                            error.kind.as_str(),
                            "document_already_exists" | "version_conflict_engine_exception"
                        )
                    {
                        return Err(Error::validation(format!(
                            "document with primary key {:?} already exists in vector collection \
                             {collection:?}",
                            item.id
                        )));
                    }
                    return Err(Error::transport(format!(
                        "Elasticsearch bulk insert into index {index} failed for document \
                         {:?} with status {}: {}: {}",
                        item.id, item.status, error.kind, error.reason
                    )));
                }
            }
        }
        Err(Error::transport(format!(
            "Elasticsearch bulk insert into index {index} reported errors without an \
             identifiable failing item"
        )))
    }

    async fn search(
        &self,
        collection: &str,
        query: &VectorQuery,
    ) -> Result<Vec<VectorSearchResult>> {
        if query.top_k == 0 {
            return Ok(Vec::new());
        }
        let index = self.client.index_name(collection)?;
        let schema = self.require_schema(collection).await?;
        let field = resolve_vector_field(&schema, collection, query)?;
        let dim = field.dim.unwrap_or(0);
        if query.query_vector.len() != dim {
            return Err(Error::validation(format!(
                "query vector dimension {} does not match vector field {:?} dimension {dim} \
                 of collection {collection:?}",
                query.query_vector.len(),
                field.name
            )));
        }

        let translated = translate(&query.filters);
        let recheck = translated.is_none();
        // Without a service-side filter the hits are re-checked locally, so the
        // window has to be wider than the requested result count.
        let candidates = query.top_k.saturating_mul(4).max(query.top_k);
        let window = if recheck { candidates } else { query.top_k };
        let mut knn = Map::new();
        knn.insert("field".to_string(), json!(field.name));
        knn.insert("query_vector".to_string(), json!(query.query_vector));
        knn.insert("k".to_string(), json!(window));
        knn.insert("num_candidates".to_string(), json!(candidates));
        if let Some(filter) = translated {
            knn.insert("filter".to_string(), filter);
        }
        let body = json!({"knn": Value::Object(knn), "size": window, "_source": true});
        let payload = self.send_search(&index, &body).await?;
        parse_hits(&payload, &schema, &query.filters, recheck, query.top_k)
    }

    async fn delete_docs_by_ids(&self, collection: &str, ids: &[String]) -> Result<usize> {
        let index = self.client.index_name(collection)?;
        if ids.is_empty() {
            return Ok(0);
        }
        if !self.collection_exists(collection).await? {
            return Err(collection_not_found(collection));
        }
        let keys: BTreeSet<String> = ids.iter().cloned().collect();
        self.delete_keys(&index, &keys).await
    }

    async fn delete_docs_by_filters(&self, collection: &str, filters: &Value) -> Result<usize> {
        let index = self.client.index_name(collection)?;
        if let Some(query) = translate(filters) {
            if !self.collection_exists(collection).await? {
                return Err(collection_not_found(collection));
            }
            let body = json!({"query": query});
            let request = HttpRequest::post(
                self.client
                    .url(&self.client.path(&index, "/_delete_by_query")),
            )
            .json(&body)?;
            let response = self.client.send(request, "deleting documents").await?;
            let payload: Value = response.json()?;
            return Ok(payload
                .get("deleted")
                .and_then(Value::as_u64)
                .and_then(|deleted| usize::try_from(deleted).ok())
                .unwrap_or(0));
        }
        // The local fallback orders the scan by the primary key, so it needs the
        // schema; a missing collection is reported here as not found.
        let schema = self.require_schema(collection).await?;
        let keys = self.collect_matching_ids(&index, &schema, filters).await?;
        if keys.is_empty() {
            return Ok(0);
        }
        self.delete_keys(&index, &keys).await
    }

    async fn list_collection_names(&self) -> Result<Vec<String>> {
        let request = HttpRequest::get(self.client.url(&self.client.indexes_path()));
        let response = self.client.send_raw(request).await?;
        if response.status == 404 {
            return Ok(Vec::new());
        }
        if !response.is_success() {
            return Err(failure(&response, "listing collections"));
        }
        let payload: Value = response.json()?;
        let mut names: Vec<String> = index_names_from(&payload)
            .iter()
            .filter_map(|index| self.client.collection_from_index(index))
            .collect();
        names.sort();
        names.dedup();
        Ok(names)
    }
}

/// One validated document, ready for the bulk request.
struct PreparedDoc {
    /// Primary-key value rendered as a string, used as the Elasticsearch `_id`.
    id: String,
    /// The `_source` object: every stored field, the primary key included.
    source: Map<String, Value>,
}

/// One `items` entry of a bulk reply.
struct BulkItem {
    /// Id of the affected document.
    id: String,
    /// HTTP status Elasticsearch reports for the item.
    status: u64,
    /// Outcome name of a successful item, for example `created` or `deleted`.
    result: Option<String>,
    /// Failure of the item, when it failed.
    error: Option<BulkFailure>,
}

/// The failure of one bulk item.
struct BulkFailure {
    /// Elasticsearch error type, for example `version_conflict_engine_exception`.
    kind: String,
    /// Human-readable reason of the failure.
    reason: String,
}

/// The primary-key field of `schema`.
///
/// Every search and every locally filtered read orders documents by it, so a
/// schema without a primary key cannot be served.
fn primary_key_field(schema: &CollectionSchema) -> Result<&FieldSchema> {
    schema
        .primary_key()
        .ok_or_else(|| Error::validation("collection schema declares no primary key field"))
}

/// The `hits.hits` array of a `_search` response.
fn hits_of(payload: &Value) -> &[Value] {
    payload
        .get("hits")
        .and_then(|hits| hits.get("hits"))
        .and_then(Value::as_array)
        .map_or(&[], Vec::as_slice)
}

/// The action results of a bulk reply, filtered to one bulk action.
fn bulk_items(payload: &Value, action: &str) -> Vec<BulkItem> {
    let items: &[Value] = payload
        .get("items")
        .and_then(Value::as_array)
        .map_or(&[], Vec::as_slice);
    items
        .iter()
        .filter_map(|item| {
            let result = item.get(action)?;
            Some(BulkItem {
                id: result.get("_id").map_or_else(String::new, id_text),
                status: result.get("status").and_then(Value::as_u64).unwrap_or(0),
                result: result
                    .get("result")
                    .and_then(Value::as_str)
                    .map(str::to_string),
                error: result.get("error").map(|error| BulkFailure {
                    kind: error
                        .get("type")
                        .and_then(Value::as_str)
                        .unwrap_or("<no type>")
                        .to_string(),
                    reason: error
                        .get("reason")
                        .and_then(Value::as_str)
                        .unwrap_or("<no reason>")
                        .to_string(),
                }),
            })
        })
        .collect()
}

/// Renders a JSON scalar the way Elasticsearch renders an `_id`.
fn id_text(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        other => other.to_string(),
    }
}

/// Clamps a `_score` into `[0, 1]`; a missing or non-finite score becomes `0`.
fn normalize_score(raw: Option<&Value>) -> f32 {
    match raw.and_then(Value::as_f64) {
        Some(score) if score.is_finite() => score.clamp(0.0, 1.0) as f32,
        _ => 0.0,
    }
}

/// Rebuilds search hits from one `_search` response.
///
/// Hits are filtered locally when `recheck` is set (the service could not
/// express the filter), ordered by score descending and by primary key
/// ascending, and truncated to `top_k`.
fn parse_hits(
    payload: &Value,
    schema: &CollectionSchema,
    filters: &Value,
    recheck: bool,
    top_k: usize,
) -> Result<Vec<VectorSearchResult>> {
    let primary = primary_key_field(schema)?;
    let mut ranked: Vec<(f32, String, Map<String, Value>)> = Vec::new();
    for hit in hits_of(payload) {
        let id = hit.get("_id").map_or_else(String::new, id_text);
        let mut fields = hit
            .get("_source")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();
        if !fields.contains_key(&primary.name) {
            fields.insert(primary.name.clone(), Value::String(id.clone()));
        }
        if recheck && !crate::store::filter::matches(filters, &Value::Object(fields.clone())) {
            continue;
        }
        ranked.push((normalize_score(hit.get("_score")), id, fields));
    }
    ranked.sort_by(|left, right| {
        right
            .0
            .total_cmp(&left.0)
            .then_with(|| left.1.cmp(&right.1))
    });
    ranked.truncate(top_k);
    Ok(ranked
        .into_iter()
        .map(|(score, _, fields)| VectorSearchResult::new(score, fields))
        .collect())
}

/// Builds the NDJSON body of a bulk insert: an action line and a `_source` line
/// per document.
fn bulk_create_body(docs: &[PreparedDoc]) -> Result<String> {
    let mut body = String::new();
    for doc in docs {
        let action = json!({(CREATE_ACTION): {"_id": doc.id}});
        body.push_str(&compact(&action)?);
        body.push('\n');
        body.push_str(&compact(&Value::Object(doc.source.clone()))?);
        body.push('\n');
    }
    Ok(body)
}

/// Builds the NDJSON body of a bulk delete: one action line per key.
fn bulk_delete_body(keys: &BTreeSet<String>) -> Result<String> {
    let mut body = String::new();
    for key in keys {
        let action = json!({(DELETE_ACTION): {"_id": key}});
        body.push_str(&compact(&action)?);
        body.push('\n');
    }
    Ok(body)
}

/// Serializes one NDJSON line.
fn compact(value: &Value) -> Result<String> {
    serde_json::to_string(value).map_err(Error::external)
}

/// Picks the vector field a query scores against.
fn resolve_vector_field<'a>(
    schema: &'a CollectionSchema,
    collection: &str,
    query: &VectorQuery,
) -> Result<&'a FieldSchema> {
    if let Some(name) = &query.vector_field {
        return schema.vector_field(name).ok_or_else(|| {
            Error::validation(format!(
                "field {name:?} is not a FLOAT_VECTOR field of collection {collection:?}"
            ))
        });
    }
    schema.default_vector_field().ok_or_else(|| {
        Error::validation(format!(
            "collection {collection:?} declares no FLOAT_VECTOR field"
        ))
    })
}

/// Validates a batch of raw documents and converts them into bulk entries.
///
/// Nothing is sent when any document is invalid, so a failed call inserts
/// nothing — the same all-or-nothing contract as the in-memory backend.
fn prepare_docs(
    schema: &CollectionSchema,
    docs: Vec<Map<String, Value>>,
) -> Result<Vec<PreparedDoc>> {
    let primary = primary_key_field(schema)?;
    let mut prepared: Vec<PreparedDoc> = Vec::with_capacity(docs.len());
    let mut seen: BTreeMap<String, usize> = BTreeMap::new();
    for (position, raw) in docs.into_iter().enumerate() {
        let doc = prepare_doc(schema, primary, raw, position)?;
        if let Some(first) = seen.insert(doc.id.clone(), position) {
            return Err(Error::validation(format!(
                "duplicate primary key {:?} in the batch (positions {first} and {position})",
                doc.id
            )));
        }
        prepared.push(doc);
    }
    Ok(prepared)
}

/// Validates one document, applies defaults, and builds its `_source` object.
fn prepare_doc(
    schema: &CollectionSchema,
    primary: &FieldSchema,
    raw: Map<String, Value>,
    position: usize,
) -> Result<PreparedDoc> {
    let mut fields = raw;
    for field in &schema.fields {
        if let (false, Some(default)) = (fields.contains_key(&field.name), &field.default_value) {
            fields.insert(field.name.clone(), default.clone());
        }
    }
    let id = resolve_id(primary, &mut fields, position)?;

    let mut source = Map::new();
    for field in &schema.fields {
        let Some(value) = fields.get(&field.name) else {
            if field.is_vector() {
                return Err(Error::validation(format!(
                    "document {id:?} is missing vector field {:?}",
                    field.name
                )));
            }
            continue;
        };
        if value.is_null() {
            continue;
        }
        if field.is_vector() {
            source.insert(field.name.clone(), json!(parse_vector(field, value)?));
        } else {
            check_value(field, value)?;
            source.insert(field.name.clone(), value.clone());
        }
    }
    for (name, value) in &fields {
        if value.is_null() || schema.fields.iter().any(|field| field.name == *name) {
            continue;
        }
        if !schema.enable_dynamic_field {
            return Err(Error::validation(format!(
                "document {id:?} has field {name:?}, which the schema does not declare and \
                 enable_dynamic_field is false"
            )));
        }
        source.insert(name.clone(), value.clone());
    }
    Ok(PreparedDoc { id, source })
}

/// Resolves the primary key of a document, generating one for `auto_id` schemas.
fn resolve_id(
    primary: &FieldSchema,
    fields: &mut Map<String, Value>,
    position: usize,
) -> Result<String> {
    match fields.get(&primary.name).cloned() {
        Some(Value::String(value)) if value.is_empty() => Err(Error::validation(format!(
            "document at position {position} has an empty primary key field {:?}",
            primary.name
        ))),
        Some(Value::String(value)) => Ok(value),
        Some(Value::Number(number)) if number.is_i64() || number.is_u64() => Ok(number.to_string()),
        Some(other) => Err(Error::validation(format!(
            "document at position {position} has primary key field {:?} of type {other}; \
             a string or integer is required",
            primary.name
        ))),
        None if primary.auto_id => {
            let generated = Uuid::new_v4().to_string();
            fields.insert(primary.name.clone(), Value::String(generated.clone()));
            Ok(generated)
        }
        None => Err(Error::validation(format!(
            "document at position {position} is missing primary key field {:?}",
            primary.name
        ))),
    }
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

#[cfg(test)]
mod tests;
