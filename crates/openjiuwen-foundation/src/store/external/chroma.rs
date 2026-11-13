//! Chroma vector store over the Chroma v1 REST API.
//!
//! [`ChromaVectorStore`] implements the [`VectorStore`] contract of
//! [`crate::store::vector`] on top of a Chroma server: collections are created
//! from a [`CollectionSchema`], documents are inserted with one embedding each,
//! and search results carry the same normalized, "higher is better" scores as
//! the in-memory reference backend.
//!
//! # Wire contract
//!
//! Every request goes through [`openjiuwen_common::http::HttpClient`] against
//! `<endpoint>/api/v1/collections` and carries the optional `x-chroma-tenant` /
//! `x-chroma-database` headers and the bearer `api_key`. A non-2xx response
//! becomes an [`Error::Transport`](openjiuwen_common::error::Error) naming the
//! status code, the URL, and an excerpt of the body.
//!
//! # Schema round-trip
//!
//! Chroma has no typed schema, so the store serializes the whole
//! [`CollectionSchema`] into one reserved metadata key
//! ([`ChromaConfig::schema_metadata_key`]) at collection creation and reads it
//! back on every operation. A collection that exists but lacks that key was not
//! created through this store and yields
//! [`Error::Unsupported`](openjiuwen_common::error::Error).
//!
//! # Differences from the reference backend
//!
//! - Chroma stores one embedding per record, so a schema with more than one
//!   `FLOAT_VECTOR` field is rejected.
//! - Chroma metadata holds strings, numbers and booleans only, so `JSON`/`ARRAY`
//!   fields are rejected at `create_collection` and a non-scalar value fails
//!   `add_docs`.
//! - The filter language is translated into Chroma's `where` clause. Constructs
//!   Chroma cannot express (dotted field paths, `$contains`, `$not`, object
//!   literals) fail with [`Error::Unsupported`](openjiuwen_common::error::Error)
//!   instead of silently returning unfiltered rows.
//! - Distances are normalized per [`Metric`]: `cosine`/`ip` use `1 - d / 2`,
//!   `l2` (squared) uses `1 / (1 + d)`, and every score is clamped into `[0, 1]`.
//! - `add_docs` is insert-only where Chroma can enforce it: a duplicate id
//!   rejected by the service (HTTP 400/409) becomes a validation error.

use std::collections::{BTreeMap, BTreeSet};
use std::time::Duration;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use uuid::Uuid;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::http::{HttpClient, HttpClientConfig, HttpRequest, HttpResponse};
use openjiuwen_common::security::SecretString;

use crate::store::vector::{
    collection_not_found, CollectionSchema, FieldSchema, Metric, VectorDataType, VectorQuery,
    VectorSearchResult, VectorStore,
};

/// Default total timeout of one Chroma request.
const fn default_timeout() -> Duration {
    Duration::from_secs(30)
}

/// Default metadata key holding the serialized collection schema.
fn default_schema_metadata_key() -> String {
    "openjiuwen:schema".to_string()
}

/// Number of documents read per page while collecting ids for a filtered delete.
const FILTER_DELETE_PAGE: usize = 500;

/// Maximum number of response-body characters echoed in an error message.
const ERROR_BODY_EXCERPT: usize = 200;

/// Configuration of a [`ChromaVectorStore`].
///
/// The struct is `serde`-deserializable with a default for every field, so a
/// backend can be built straight from a JSON configuration object; the builder
/// setters exist for code that would rather not spell out every field.
///
/// ```
/// use openjiuwen_foundation::store::external::chroma::ChromaConfig;
/// use openjiuwen_foundation::store::vector::Metric;
///
/// let config = ChromaConfig::new("http://localhost:8000")
///     .with_tenant("default_tenant")
///     .with_database("default_database")
///     .with_metric(Metric::Euclidean);
///
/// assert_eq!(config.tenant.as_deref(), Some("default_tenant"));
/// assert_eq!(config.metric, Metric::Euclidean);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, rename_all = "snake_case")]
pub struct ChromaConfig {
    /// Base URL of the Chroma server, for example `http://localhost:8000`.
    pub endpoint: String,
    /// Optional tenant, sent as the `x-chroma-tenant` header.
    pub tenant: Option<String>,
    /// Optional database, sent as the `x-chroma-database` header.
    pub database: Option<String>,
    /// Optional API key, sent as `Authorization: Bearer <key>`.
    pub api_key: Option<SecretString>,
    /// Similarity metric of a created collection's HNSW index.
    pub metric: Metric,
    /// Timeout of a single HTTP request.
    pub timeout: Duration,
    /// Metadata key under which the serialized [`CollectionSchema`] is stored.
    pub schema_metadata_key: String,
}

impl Default for ChromaConfig {
    fn default() -> Self {
        Self {
            endpoint: String::new(),
            tenant: None,
            database: None,
            api_key: None,
            metric: Metric::default(),
            timeout: default_timeout(),
            schema_metadata_key: default_schema_metadata_key(),
        }
    }
}

impl ChromaConfig {
    /// Creates a configuration pointing at `endpoint` and using every default.
    #[must_use]
    pub fn new(endpoint: impl Into<String>) -> Self {
        Self {
            endpoint: endpoint.into(),
            ..Self::default()
        }
    }

    /// Sets the tenant sent with every request.
    #[must_use]
    pub fn with_tenant(mut self, tenant: impl Into<String>) -> Self {
        self.tenant = Some(tenant.into());
        self
    }

    /// Sets the database sent with every request.
    #[must_use]
    pub fn with_database(mut self, database: impl Into<String>) -> Self {
        self.database = Some(database.into());
        self
    }

    /// Sets the API key sent as a bearer token.
    #[must_use]
    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(SecretString::new(api_key));
        self
    }

    /// Sets the similarity metric of created collections.
    #[must_use]
    pub fn with_metric(mut self, metric: Metric) -> Self {
        self.metric = metric;
        self
    }

    /// Sets the timeout of a single request.
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Sets the metadata key that holds the serialized collection schema.
    #[must_use]
    pub fn with_schema_metadata_key(mut self, key: impl Into<String>) -> Self {
        self.schema_metadata_key = key.into();
        self
    }
}

/// A [`VectorStore`] backed by a Chroma server.
///
/// The store holds one HTTP client, so it is cheap to share behind an `Arc` and
/// every method takes `&self`.
///
/// # Errors
///
/// [`ChromaVectorStore::new`] rejects an empty endpoint and any endpoint the
/// HTTP client cannot accept; the contract methods return the errors documented
/// on [`VectorStore`], plus
/// [`Error::Unsupported`](openjiuwen_common::error::Error) for a filter or a
/// collection that Chroma cannot represent.
#[derive(Debug, Clone)]
pub struct ChromaVectorStore {
    client: HttpClient,
    config: ChromaConfig,
    endpoint: String,
}

impl ChromaVectorStore {
    /// Builds a store from `config`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`](openjiuwen_common::error::Error) for an empty
    /// endpoint and propagates the HTTP client's configuration errors.
    pub fn new(config: ChromaConfig) -> Result<Self> {
        let endpoint = config.endpoint.trim().trim_end_matches('/').to_string();
        if endpoint.is_empty() {
            return Err(Error::config("Chroma endpoint must not be empty"));
        }
        let client = HttpClient::new(HttpClientConfig::default().with_timeout(config.timeout))?;
        Ok(Self {
            client,
            config,
            endpoint,
        })
    }

    /// Sends `request` unchanged, without interpreting the status code.
    async fn send_raw(&self, request: HttpRequest) -> Result<HttpResponse> {
        self.client.execute(self.apply(request)).await
    }

    /// Sends `request` and turns any non-2xx response into a transport error.
    async fn send(&self, request: HttpRequest, context: &str) -> Result<HttpResponse> {
        let response = self.send_raw(request).await?;
        if response.is_success() {
            Ok(response)
        } else {
            Err(failure(&response, context))
        }
    }

    /// Decorates a request with the timeout and the configured headers.
    fn apply(&self, request: HttpRequest) -> HttpRequest {
        let mut request = request.timeout(self.config.timeout);
        if let Some(tenant) = &self.config.tenant {
            request = request.header("x-chroma-tenant", tenant.clone());
        }
        if let Some(database) = &self.config.database {
            request = request.header("x-chroma-database", database.clone());
        }
        if let Some(api_key) = &self.config.api_key {
            request = request.bearer(api_key.clone());
        }
        request
    }

    /// The absolute URL of `path` on the configured endpoint.
    fn url(&self, path: &str) -> String {
        format!("{}{path}", self.endpoint)
    }

    /// The API path of a collection, with an optional action suffix.
    fn collection_path(collection: &str, action: &str) -> String {
        format!("/api/v1/collections/{}{action}", encode_segment(collection))
    }

    /// Reads the schema of a collection from its reserved metadata key.
    async fn fetch_schema(&self, collection: &str) -> Result<Option<CollectionSchema>> {
        let path = Self::collection_path(collection, "");
        let response = self.send_raw(HttpRequest::get(self.url(&path))).await?;
        match response.status {
            404 => return Ok(None),
            200..=299 => {}
            _ => return Err(failure(&response, "reading collection")),
        }
        let payload: Value = response.json()?;
        let stored = payload
            .get("metadata")
            .and_then(Value::as_object)
            .and_then(|metadata| metadata.get(&self.config.schema_metadata_key));
        let Some(Value::String(raw)) = stored else {
            return Err(Error::unsupported(format!(
                "collection {collection:?} was not created through ChromaVectorStore: \
                 its metadata has no {:?} entry",
                self.config.schema_metadata_key
            )));
        };
        serde_json::from_str(raw).map(Some).map_err(|err| {
            Error::validation(format!(
                "collection {collection:?} stores an unreadable schema in metadata {:?}: {err}",
                self.config.schema_metadata_key
            ))
        })
    }

    /// Reads the ids matched by `where_clause`, paging while pages come back full.
    async fn read_ids(
        &self,
        collection: &str,
        where_clause: Option<&Value>,
    ) -> Result<Vec<String>> {
        let mut collected: BTreeSet<String> = BTreeSet::new();
        let mut offset = 0usize;
        loop {
            let mut body = json!({"include": [], "limit": FILTER_DELETE_PAGE, "offset": offset});
            if let (Some(clause), Some(entries)) = (where_clause, body.as_object_mut()) {
                entries.insert("where".to_string(), clause.clone());
            }
            let path = Self::collection_path(collection, "/get");
            let response = self
                .send(
                    HttpRequest::post(self.url(&path)).json(&body)?,
                    "reading documents",
                )
                .await?;
            let page = response_ids(&response.json()?);
            let fresh = page
                .iter()
                .filter(|id| collected.insert((*id).clone()))
                .count();
            if page.len() < FILTER_DELETE_PAGE || fresh == 0 {
                return Ok(collected.into_iter().collect());
            }
            offset += page.len();
        }
    }

    /// Deletes the given ids with one `/delete` request.
    async fn delete_ids(&self, collection: &str, ids: &BTreeSet<String>) -> Result<()> {
        let path = Self::collection_path(collection, "/delete");
        self.send(
            HttpRequest::post(self.url(&path)).json(&json!({"ids": ids}))?,
            "deleting documents",
        )
        .await?;
        Ok(())
    }
}

#[async_trait]
impl VectorStore for ChromaVectorStore {
    async fn create_collection(&self, collection: &str, schema: CollectionSchema) -> Result<()> {
        schema.validate()?;
        validate_chroma_schema(&schema)?;
        let mut metadata = Map::new();
        metadata.insert(
            "hnsw:space".to_string(),
            json!(hnsw_space(self.config.metric)),
        );
        metadata.insert(
            self.config.schema_metadata_key.clone(),
            json!(serde_json::to_string(&schema).map_err(Error::external)?),
        );
        let body = json!({"name": collection, "get_or_create": false, "metadata": metadata});
        let response = self
            .send_raw(HttpRequest::post(self.url("/api/v1/collections")).json(&body)?)
            .await?;
        if response.is_success() {
            return Ok(());
        }
        let text = response.text();
        if response.status == 409 || text.to_lowercase().contains("already exist") {
            return Err(Error::validation(format!(
                "vector collection {collection:?} already exists"
            )));
        }
        Err(failure(&response, "creating collection"))
    }

    async fn delete_collection(&self, collection: &str) -> Result<()> {
        let path = Self::collection_path(collection, "");
        let response = self.send_raw(HttpRequest::delete(self.url(&path))).await?;
        if matches!(response.status, 404 | 200..=299) {
            return Ok(());
        }
        Err(failure(&response, "deleting collection"))
    }

    async fn collection_exists(&self, collection: &str) -> Result<bool> {
        let path = Self::collection_path(collection, "");
        let response = self.send_raw(HttpRequest::get(self.url(&path))).await?;
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
        let schema = self
            .fetch_schema(collection)
            .await?
            .ok_or_else(|| collection_not_found(collection))?;
        let prepared = prepare_docs(&schema, docs)?;
        let path = Self::collection_path(collection, "/add");
        let response = self
            .send_raw(HttpRequest::post(self.url(&path)).json(&add_body(&schema, &prepared))?)
            .await?;
        if response.is_success() {
            return Ok(prepared.len());
        }
        let text = response.text();
        if matches!(response.status, 400 | 409) {
            return Err(Error::validation(format!(
                "adding documents to Chroma collection {collection:?} was rejected: {}",
                excerpt(&text)
            )));
        }
        Err(failure(&response, "adding documents"))
    }

    async fn search(
        &self,
        collection: &str,
        query: &VectorQuery,
    ) -> Result<Vec<VectorSearchResult>> {
        if query.top_k == 0 {
            return Ok(Vec::new());
        }
        let schema = self
            .fetch_schema(collection)
            .await?
            .ok_or_else(|| collection_not_found(collection))?;
        let field = resolve_vector_field(&schema, collection, query)?;
        let dim = field.dim.unwrap_or(0);
        if query.query_vector.len() != dim {
            return Err(Error::validation(format!(
                "query vector dimension {} does not match vector field {:?} dimension \
                 {dim} of collection {collection:?}",
                query.query_vector.len(),
                field.name
            )));
        }
        let mut body = Map::new();
        body.insert("query_embeddings".to_string(), json!([&query.query_vector]));
        body.insert("n_results".to_string(), json!(query.top_k));
        body.insert(
            "include".to_string(),
            json!(["metadatas", "documents", "distances"]),
        );
        if let Some(where_clause) = compile_where(&query.filters)? {
            body.insert("where".to_string(), where_clause);
        }
        let path = Self::collection_path(collection, "/query");
        let response = self
            .send(
                HttpRequest::post(self.url(&path)).json(&Value::Object(body))?,
                "searching collection",
            )
            .await?;
        parse_hits(&response.json()?, &schema, self.config.metric, query.top_k)
    }

    async fn delete_docs_by_ids(&self, collection: &str, ids: &[String]) -> Result<usize> {
        self.fetch_schema(collection)
            .await?
            .ok_or_else(|| collection_not_found(collection))?;
        if ids.is_empty() {
            return Ok(0);
        }
        let path = Self::collection_path(collection, "/get");
        let response = self
            .send(
                HttpRequest::post(self.url(&path)).json(&json!({"ids": ids, "include": []}))?,
                "reading documents",
            )
            .await?;
        let existing: BTreeSet<String> = response_ids(&response.json()?).into_iter().collect();
        if existing.is_empty() {
            return Ok(0);
        }
        self.delete_ids(collection, &existing).await?;
        Ok(existing.len())
    }

    async fn delete_docs_by_filters(&self, collection: &str, filters: &Value) -> Result<usize> {
        self.fetch_schema(collection)
            .await?
            .ok_or_else(|| collection_not_found(collection))?;
        let where_clause = compile_where(filters)?;
        let ids: BTreeSet<String> = self
            .read_ids(collection, where_clause.as_ref())
            .await?
            .into_iter()
            .collect();
        if ids.is_empty() {
            return Ok(0);
        }
        let deleted = ids.len();
        self.delete_ids(collection, &ids).await?;
        Ok(deleted)
    }

    async fn list_collection_names(&self) -> Result<Vec<String>> {
        let response = self
            .send(
                HttpRequest::get(self.url("/api/v1/collections")),
                "listing collections",
            )
            .await?;
        let payload: Value = response.json()?;
        let items = payload.as_array().ok_or_else(|| {
            Error::validation(format!(
                "Chroma returned a non-array collection list: {}",
                excerpt(&payload.to_string())
            ))
        })?;
        let mut names: Vec<String> = items
            .iter()
            .filter_map(|item| match item {
                Value::String(name) => Some(name.clone()),
                Value::Object(entry) => entry
                    .get("name")
                    .and_then(Value::as_str)
                    .map(str::to_string),
                _ => None,
            })
            .collect();
        names.sort();
        names.dedup();
        Ok(names)
    }
}

/// One validated document, ready for the `/add` request.
struct PreparedDoc {
    /// Primary key, rendered as a string because Chroma ids are strings.
    id: String,
    /// Embedding of the collection's single vector field.
    embedding: Vec<f64>,
    /// Value of the document (full-text) field, when the schema declares one.
    document: Option<String>,
    /// Scalar metadata entries (primary key included).
    metadata: Map<String, Value>,
}

/// One compiled filter: either "everything matches" or a Chroma `where` clause.
enum Clause {
    /// The condition is a tautology, so no `where` entry is sent.
    Everything,
    /// The condition compiles into this Chroma `where` value.
    Where(Value),
}

/// Builds the `/add` request body from validated documents.
fn add_body(schema: &CollectionSchema, docs: &[PreparedDoc]) -> Value {
    let mut body = json!({
        "ids": docs.iter().map(|doc| doc.id.as_str()).collect::<Vec<_>>(),
        "embeddings": docs.iter().map(|doc| doc.embedding.as_slice()).collect::<Vec<_>>(),
    });
    if document_field(schema).is_some() {
        body["documents"] = json!(docs
            .iter()
            .map(|doc| doc.document.as_deref())
            .collect::<Vec<_>>());
    }
    if docs.iter().any(|doc| !doc.metadata.is_empty()) {
        body["metadatas"] = Value::Array(
            docs.iter()
                .map(|doc| {
                    if doc.metadata.is_empty() {
                        Value::Null
                    } else {
                        Value::Object(doc.metadata.clone())
                    }
                })
                .collect(),
        );
    }
    body
}

/// Maps a [`Metric`] onto the HNSW space name Chroma understands.
fn hnsw_space(metric: Metric) -> &'static str {
    match metric {
        Metric::Cosine => "cosine",
        Metric::Dot => "ip",
        Metric::Euclidean => "l2",
    }
}

/// Normalizes a Chroma distance into a `[0, 1]` similarity score.
fn normalize_distance(metric: Metric, distance: f64) -> f32 {
    let score = match metric {
        Metric::Cosine | Metric::Dot => 1.0 - distance / 2.0,
        Metric::Euclidean => 1.0 / (1.0 + distance),
    };
    if score.is_finite() {
        score.clamp(0.0, 1.0) as f32
    } else {
        0.0
    }
}

/// Builds the "unexpected response" error of a failed request.
fn failure(response: &HttpResponse, context: &str) -> Error {
    Error::transport(format!(
        "Chroma {context} failed with HTTP {} for {}: {}",
        response.status,
        response.url,
        excerpt(&response.text())
    ))
}

/// Shortens a response body for an error message.
fn excerpt(text: &str) -> String {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return "<empty body>".to_string();
    }
    let mut excerpt: String = trimmed.chars().take(ERROR_BODY_EXCERPT).collect();
    if trimmed.chars().count() > ERROR_BODY_EXCERPT {
        excerpt.push('…');
    }
    excerpt
}

/// Percent-encodes one path segment, keeping the RFC 3986 unreserved set.
fn encode_segment(segment: &str) -> String {
    const HEX: &[u8; 16] = b"0123456789ABCDEF";
    let mut encoded = String::with_capacity(segment.len());
    for byte in segment.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_' | b'~') {
            encoded.push(char::from(byte));
        } else {
            encoded.push('%');
            encoded.push(char::from(HEX[usize::from(byte >> 4)]));
            encoded.push(char::from(HEX[usize::from(byte & 0x0F)]));
        }
    }
    encoded
}

/// Rejects schema shapes Chroma cannot represent.
fn validate_chroma_schema(schema: &CollectionSchema) -> Result<()> {
    for field in &schema.fields {
        if matches!(field.dtype, VectorDataType::Json | VectorDataType::Array) {
            return Err(Error::validation(format!(
                "field {:?} has type {:?}, which Chroma metadata cannot store; \
                 use a scalar or VARCHAR field",
                field.name, field.dtype
            )));
        }
    }
    let vectors = schema.vector_fields();
    if vectors.len() > 1 {
        let names: Vec<&str> = vectors.iter().map(|field| field.name.as_str()).collect();
        return Err(Error::validation(format!(
            "Chroma stores one embedding per record, but the schema declares {} \
             FLOAT_VECTOR fields: {names:?}",
            vectors.len()
        )));
    }
    Ok(())
}

/// The field stored as the Chroma `document` (full text): the first `VARCHAR`
/// field that is not the primary key.
fn document_field(schema: &CollectionSchema) -> Option<&FieldSchema> {
    schema
        .fields
        .iter()
        .find(|field| field.dtype == VectorDataType::Varchar && !field.is_primary)
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

/// Validates a batch of raw documents and converts them into `/add` entries.
///
/// Nothing is sent when any document is invalid, so a failed call inserts
/// nothing — the same all-or-nothing contract as the in-memory backend.
fn prepare_docs(
    schema: &CollectionSchema,
    docs: Vec<Map<String, Value>>,
) -> Result<Vec<PreparedDoc>> {
    let primary = schema
        .primary_key()
        .ok_or_else(|| Error::validation("collection schema declares no primary key field"))?;
    let vector = schema
        .default_vector_field()
        .ok_or_else(|| Error::validation("collection schema declares no FLOAT_VECTOR field"))?;
    let document = document_field(schema);
    let mut prepared: Vec<PreparedDoc> = Vec::with_capacity(docs.len());
    let mut seen: BTreeMap<String, usize> = BTreeMap::new();
    for (position, raw) in docs.into_iter().enumerate() {
        let doc = prepare_doc(schema, primary, vector, document, raw, position)?;
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

/// Validates one document, applies defaults, and splits it into Chroma parts.
fn prepare_doc(
    schema: &CollectionSchema,
    primary: &FieldSchema,
    vector: &FieldSchema,
    document: Option<&FieldSchema>,
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
    let missing = || {
        Error::validation(format!(
            "document {id:?} is missing vector field {:?}",
            vector.name
        ))
    };
    let embedding = parse_vector(vector, fields.get(&vector.name).ok_or_else(missing)?)?;

    let mut metadata = Map::new();
    let mut document_text = None;
    for field in &schema.fields {
        let Some(value) = fields.get(&field.name) else {
            continue;
        };
        if field.is_vector() || value.is_null() {
            continue;
        }
        check_value(field, value)?;
        if document.is_some_and(|doc| doc.name == field.name) {
            document_text = value.as_str().map(str::to_string);
            continue;
        }
        insert_metadata(&mut metadata, &field.name, value)?;
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
        insert_metadata(&mut metadata, name, value)?;
    }
    Ok(PreparedDoc {
        id,
        embedding,
        document: document_text,
        metadata,
    })
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
fn parse_vector(field: &FieldSchema, value: &Value) -> Result<Vec<f64>> {
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
    items
        .iter()
        .map(|item| {
            item.as_f64()
                .filter(|number| number.is_finite())
                .ok_or_else(|| {
                    Error::validation(format!(
                        "vector field {:?} expects finite numbers, got component {item}",
                        field.name
                    ))
                })
        })
        .collect()
}

/// Inserts one scalar metadata entry, rejecting non-scalar values.
fn insert_metadata(metadata: &mut Map<String, Value>, name: &str, value: &Value) -> Result<()> {
    if matches!(value, Value::String(_) | Value::Number(_) | Value::Bool(_)) {
        metadata.insert(name.to_string(), value.clone());
        return Ok(());
    }
    Err(Error::validation(format!(
        "field {name:?} has the non-scalar value {value}; Chroma metadata accepts strings, \
         numbers and booleans only"
    )))
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
        VectorDataType::Varchar => match value.as_str() {
            None => Err(mismatch("a string")),
            Some(text) => match field.max_length {
                Some(max_length) if text.chars().count() > max_length => {
                    Err(Error::validation(format!(
                        "field {:?} exceeds max_length {max_length}: {} characters",
                        field.name,
                        text.chars().count()
                    )))
                }
                _ => Ok(()),
            },
        },
        VectorDataType::Int64 => value
            .as_i64()
            .map(|_| ())
            .ok_or_else(|| mismatch("an integer")),
        VectorDataType::Int32 => match value.as_i64() {
            None => Err(mismatch("a 32-bit integer")),
            Some(number) if i32::try_from(number).is_err() => Err(Error::validation(format!(
                "field {:?} is out of the INT32 range: {number}",
                field.name
            ))),
            Some(_) => Ok(()),
        },
        VectorDataType::Float | VectorDataType::Double if value.is_number() => Ok(()),
        VectorDataType::Bool if value.is_boolean() => Ok(()),
        VectorDataType::FloatVector => parse_vector(field, value).map(|_| ()),
        VectorDataType::Json | VectorDataType::Array => Err(Error::validation(format!(
            "field {:?} has type {:?}, which Chroma metadata cannot store",
            field.name, field.dtype
        ))),
        _ if field.dtype == VectorDataType::Bool => Err(mismatch("a boolean")),
        _ => Err(mismatch("a number")),
    }
}

/// Compiles the shared filter language into a Chroma `where` clause.
///
/// `Ok(None)` means "no constraint": `null` and `{}` match everything, and so
/// does an `$and` whose sub-filters are all tautologies. Any construct Chroma
/// cannot express is an `Unsupported` error rather than a dropped filter.
fn compile_where(filters: &Value) -> Result<Option<Value>> {
    match compile_clause(filters)? {
        Clause::Everything => Ok(None),
        Clause::Where(where_clause) => Ok(Some(where_clause)),
    }
}

/// Compiles one filter value, keeping the tautology case distinct.
fn compile_clause(filters: &Value) -> Result<Clause> {
    match filters {
        Value::Null => Ok(Clause::Everything),
        Value::Object(entries) => compile_object(entries),
        other => Err(Error::unsupported(format!(
            "Chroma cannot express the filter {other}; only a JSON object is comparable"
        ))),
    }
}

/// Compiles every entry of one filter object, joining them with `$and`.
fn compile_object(entries: &Map<String, Value>) -> Result<Clause> {
    let mut clauses: Vec<Value> = Vec::new();
    for (key, condition) in entries {
        match key.as_str() {
            "$and" | "$or" => {
                if let Clause::Where(value) = compile_logical(key, condition)? {
                    clauses.push(value);
                }
            }
            "$not" => {
                return Err(Error::unsupported(format!(
                    "Chroma cannot express the `$not` filter {condition}"
                )));
            }
            field => clauses.push(compile_field(field, condition)?),
        }
    }
    if clauses.is_empty() {
        return Ok(Clause::Everything);
    }
    if clauses.len() == 1 {
        let only = clauses.remove(0);
        return Ok(Clause::Where(only));
    }
    Ok(Clause::Where(json!({ "$and": clauses })))
}

/// Compiles a top-level `$and` / `$or` over an array of sub-filters.
fn compile_logical(operator: &str, condition: &Value) -> Result<Clause> {
    let items = condition.as_array().ok_or_else(|| {
        Error::unsupported(format!(
            "Chroma cannot express the `{operator}` filter {condition}: \
             it needs an array of sub-filters"
        ))
    })?;
    let mut clauses: Vec<Value> = Vec::new();
    for item in items {
        match compile_clause(item)? {
            Clause::Where(value) => clauses.push(value),
            Clause::Everything if operator == "$and" => {}
            Clause::Everything => return Ok(Clause::Everything),
        }
    }
    if clauses.is_empty() {
        return if operator == "$and" {
            Ok(Clause::Everything)
        } else {
            Err(Error::unsupported(format!(
                "Chroma cannot express the `$or` filter {condition}: \
                 none of its sub-filters is expressible"
            )))
        };
    }
    if clauses.len() == 1 {
        let only = clauses.remove(0);
        return Ok(Clause::Where(only));
    }
    Ok(Clause::Where(json!({ (operator): clauses })))
}

/// Compiles one `field: condition` pair into a Chroma `where` clause.
fn compile_field(field: &str, condition: &Value) -> Result<Value> {
    if field.contains('.') || field.contains('[') {
        return Err(Error::unsupported(format!(
            "Chroma cannot express the dotted filter path {field:?}; \
             only top-level metadata fields are comparable"
        )));
    }
    match condition {
        Value::Object(entries) if entries.keys().any(|key| key.starts_with('$')) => {
            let mut operators = Map::new();
            let mut equality: Option<Value> = None;
            for (operator, operand) in entries {
                match operator.as_str() {
                    "$eq" | "$ne" | "$gt" | "$gte" | "$lt" | "$lte" | "$in" | "$nin" => {
                        check_operand(operator, field, operand)?;
                        if operator == "$eq" {
                            equality = Some(operand.clone());
                        } else {
                            operators.insert(operator.clone(), operand.clone());
                        }
                    }
                    other => {
                        return Err(Error::unsupported(format!(
                            "Chroma cannot express the filter operator {other:?} \
                             on field {field:?}"
                        )));
                    }
                }
            }
            if let Some(value) = equality {
                if operators.is_empty() {
                    return Ok(json!({ (field): value }));
                }
                operators.insert("$eq".to_string(), value);
            }
            Ok(json!({ (field): Value::Object(operators) }))
        }
        Value::Object(_) => Err(Error::unsupported(format!(
            "Chroma cannot express the object equality filter on field {field:?}; \
             only scalar metadata is comparable"
        ))),
        Value::Array(_) | Value::Null => Err(Error::unsupported(format!(
            "Chroma cannot express the filter {condition} on field {field:?}; \
             only scalar equality and comparison operators are supported"
        ))),
        other => Ok(json!({ (field): other })),
    }
}

/// Rejects operator operands Chroma cannot compare against metadata.
fn check_operand(operator: &str, field: &str, operand: &Value) -> Result<()> {
    let scalar =
        |value: &Value| matches!(value, Value::String(_) | Value::Number(_) | Value::Bool(_));
    if operator == "$in" || operator == "$nin" {
        return match operand.as_array() {
            Some(items) if items.iter().all(scalar) => Ok(()),
            _ => Err(Error::unsupported(format!(
                "Chroma cannot express the `{operator}` filter on field {field:?}: \
                 its operand {operand} must be an array of scalars"
            ))),
        };
    }
    if scalar(operand) {
        return Ok(());
    }
    Err(Error::unsupported(format!(
        "Chroma cannot express the `{operator}` filter on field {field:?}: \
         its operand {operand} is not a scalar"
    )))
}

/// Returns one row of a `/query` or `/get` response, flattened when nested.
fn row<'a>(payload: &'a Value, key: &str) -> &'a [Value] {
    match payload.get(key).and_then(Value::as_array) {
        Some(items) => match items.first() {
            Some(Value::Array(inner)) => inner.as_slice(),
            _ => items.as_slice(),
        },
        None => &[],
    }
}

/// Collects the ids of a `/get` response, accepting flat and nested arrays.
fn response_ids(payload: &Value) -> Vec<String> {
    row(payload, "ids")
        .iter()
        .map(|id| match id {
            Value::String(text) => text.clone(),
            other => other.to_string(),
        })
        .collect()
}

/// Rebuilds search hits from one `/query` response.
fn parse_hits(
    payload: &Value,
    schema: &CollectionSchema,
    metric: Metric,
    top_k: usize,
) -> Result<Vec<VectorSearchResult>> {
    let primary = schema
        .primary_key()
        .ok_or_else(|| Error::validation("collection schema declares no primary key field"))?;
    let document = document_field(schema);
    let (ids, distances) = (row(payload, "ids"), row(payload, "distances"));
    let (metadatas, documents) = (row(payload, "metadatas"), row(payload, "documents"));
    let mut hits: Vec<(f32, String, Map<String, Value>)> = ids
        .iter()
        .enumerate()
        .map(|(position, id)| {
            let id_text = match id {
                Value::String(text) => text.clone(),
                other => other.to_string(),
            };
            let mut fields = match metadatas.get(position) {
                Some(Value::Object(metadata)) => metadata.clone(),
                _ => Map::new(),
            };
            if !fields.contains_key(&primary.name) {
                fields.insert(primary.name.clone(), Value::String(id_text.clone()));
            }
            if let (Some(field), Some(Value::String(text))) = (document, documents.get(position)) {
                fields.insert(field.name.clone(), Value::String(text.clone()));
            }
            let distance = distances
                .get(position)
                .and_then(Value::as_f64)
                .unwrap_or(f64::INFINITY);
            (normalize_distance(metric, distance), id_text, fields)
        })
        .collect();
    hits.sort_by(|left, right| {
        right
            .0
            .total_cmp(&left.0)
            .then_with(|| left.1.cmp(&right.1))
    });
    hits.truncate(top_k);
    Ok(hits
        .into_iter()
        .map(|(score, _, fields)| VectorSearchResult::new(score, fields))
        .collect())
}

#[cfg(test)]
mod tests;
