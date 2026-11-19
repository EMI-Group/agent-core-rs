//! Milvus vector store over the Milvus v2 REST API.
//!
//! [`MilvusVectorStore`] implements the [`VectorStore`] contract of
//! [`crate::store::vector`] on top of a Milvus server, speaking the v2 REST
//! protocol (`/v2/vectordb/...`) through
//! [`openjiuwen_common::http::HttpClient`]. A collection is created from a
//! [`CollectionSchema`] with one `AUTOINDEX` per vector field, documents are
//! inserted as typed rows, and search hits carry the same normalized,
//! "higher is better" scores as the in-memory reference backend.
//!
//! # Wire contract
//!
//! Every response is the v2 envelope `{"code": 0, "data": ...}`. `code != 0`
//! becomes an [`Error::External`](openjiuwen_common::error::Error) naming the
//! code and the service message, except a duplicate-primary-key or
//! "already exist" message (validation) and a missing collection (not found).
//! A non-2xx response without a parsable envelope is a transport error naming
//! the status code, the URL and an excerpt of the body.
//!
//! # Differences from the reference backend
//!
//! - `create_collection` rejects `ARRAY` fields: Milvus needs an element type
//!   that [`VectorDataType`] cannot express.
//! - `add_docs` probes the collection for the batch's primary keys before
//!   inserting (see *Duplicate primary keys* below); the batch is also
//!   validated in full locally, so a failed call inserts nothing.
//! - The filter language is compiled into a Milvus boolean expression. Dotted
//!   field paths, `$contains`, `$not`, object/array literals and `null`
//!   operands are rejected with [`Error::Unsupported`](openjiuwen_common::error::Error),
//!   never dropped.
//! - Scores are normalized per [`Metric`]: `COSINE`/`IP` use `(s + 1) / 2`,
//!   `L2` uses `max(0, (4 - d) / 4)`, and every score is clamped to `[0, 1]`.
//!
//! # Duplicate primary keys
//!
//! Milvus `insert` appends rows: it neither rejects a primary key the
//! collection already stores nor overwrites it, so a duplicate would silently
//! become a second row and the insert would look like any other successful
//! insert. [`VectorStore`] is insert-only (the in-memory and local backends
//! reject a stored primary key with a validation error), so `add_docs` asks
//! the service which of the batch's primary keys already exist — one
//! `entities/query` request with an `in [...]` filter over that batch, whose
//! `limit` is the batch length — and returns
//! [`Error::Validation`](openjiuwen_common::error::Error) naming the colliding
//! keys when any is found, writing nothing.
//!
//! The probe costs one extra round trip per `add_docs` call, and a failed
//! probe fails the insert rather than falling back to the append. Bulk
//! ingestion whose keys cannot collide can turn it off with
//! [`MilvusConfig::reject_duplicate_primary_keys`] set to `false`
//! ([`MilvusConfig::with_reject_duplicate_primary_keys`]), which restores the
//! service behaviour of appending a duplicate row.
//!
//! # Example
//!
//! ```no_run
//! use openjiuwen_foundation::store::external::milvus::{MilvusConfig, MilvusVectorStore};
//!
//! # fn main() -> openjiuwen_common::error::Result<()> {
//! let store = MilvusVectorStore::new(
//!     MilvusConfig::new("http://localhost:19530").with_token("root:Milvus"),
//! )?;
//! let _ = store;
//! # Ok(())
//! # }
//! ```

use std::collections::BTreeMap;
use std::time::Duration;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use uuid::Uuid;

use openjiuwen_common::error::{Error, ErrorKind, Result};
use openjiuwen_common::http::{HttpClient, HttpClientConfig, HttpRequest, HttpResponse};
use openjiuwen_common::security::SecretString;

use crate::store::vector::{
    collection_not_found, CollectionSchema, FieldSchema, Metric, VectorDataType, VectorQuery,
    VectorSearchResult, VectorStore,
};

/// REST paths of the Milvus v2 API used by this backend.
const COLLECTIONS_CREATE: &str = "/v2/vectordb/collections/create";
const COLLECTIONS_DROP: &str = "/v2/vectordb/collections/drop";
const COLLECTIONS_HAS: &str = "/v2/vectordb/collections/has";
const COLLECTIONS_LIST: &str = "/v2/vectordb/collections/list";
const COLLECTIONS_DESCRIBE: &str = "/v2/vectordb/collections/describe";
const ENTITIES_INSERT: &str = "/v2/vectordb/entities/insert";
const ENTITIES_SEARCH: &str = "/v2/vectordb/entities/search";
const ENTITIES_QUERY: &str = "/v2/vectordb/entities/query";
const ENTITIES_DELETE: &str = "/v2/vectordb/entities/delete";

/// Default total timeout of one Milvus request.
const fn default_timeout() -> Duration {
    Duration::from_secs(30)
}

/// Maximum number of response-body characters echoed in an error message.
const ERROR_BODY_EXCERPT: usize = 200;

/// Maximum number of primary keys read before a filtered delete.
///
/// A filtered delete reads the matching primary keys first and then deletes
/// them by id. When the read fills the cap the match set may be larger than what
/// was read, so the delete fails with
/// [`Error::Unsupported`](openjiuwen_common::error::Error) instead of removing
/// the visible window and reporting a completed delete.
const FILTER_DELETE_LIMIT: usize = 16_384;

/// Maximum number of colliding primary keys named in one error message.
const KEY_EXCERPT: usize = 8;

/// Configuration of a [`MilvusVectorStore`].
///
/// The struct is `serde`-deserializable with a default for every field, so a
/// backend can be built straight from a JSON configuration object; the builder
/// setters exist for code that would rather not spell out every field.
///
/// ```
/// use openjiuwen_foundation::store::external::milvus::MilvusConfig;
/// use openjiuwen_foundation::store::vector::Metric;
///
/// let config = MilvusConfig::new("http://localhost:19530")
///     .with_token("root:Milvus")
///     .with_db_name("default")
///     .with_metric(Metric::Euclidean);
///
/// assert_eq!(config.endpoint, "http://localhost:19530");
/// assert_eq!(config.metric, Metric::Euclidean);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, rename_all = "snake_case")]
pub struct MilvusConfig {
    /// Base URL of the Milvus server, for example `http://localhost:19530`.
    pub endpoint: String,
    /// Optional token, sent as `Authorization: Bearer <token>`.
    pub token: Option<SecretString>,
    /// Optional database, sent as the `dbName` request-body field.
    pub db_name: Option<String>,
    /// Similarity metric of the created collections' indexes and of the
    /// score normalization applied to search hits.
    pub metric: Metric,
    /// Timeout of a single HTTP request.
    pub timeout: Duration,
    /// Consistency level of every request (`Bounded` by default).
    pub consistency_level: String,
    /// Whether `add_docs` refuses primary keys the collection already stores
    /// (`true` by default).
    ///
    /// Milvus `insert` appends, so without this probe a duplicate primary key
    /// silently becomes a second row. See the module documentation on
    /// duplicate primary keys.
    pub reject_duplicate_primary_keys: bool,
}

impl Default for MilvusConfig {
    fn default() -> Self {
        Self {
            endpoint: String::new(),
            token: None,
            db_name: None,
            metric: Metric::default(),
            timeout: default_timeout(),
            consistency_level: "Bounded".to_string(),
            reject_duplicate_primary_keys: true,
        }
    }
}

impl MilvusConfig {
    /// Creates a configuration pointing at `endpoint` and using every default.
    #[must_use]
    pub fn new(endpoint: impl Into<String>) -> Self {
        Self {
            endpoint: endpoint.into(),
            ..Self::default()
        }
    }

    /// Sets the token sent as a bearer token.
    #[must_use]
    pub fn with_token(mut self, token: impl Into<String>) -> Self {
        self.token = Some(SecretString::new(token));
        self
    }

    /// Sets the database sent as `dbName`.
    #[must_use]
    pub fn with_db_name(mut self, db_name: impl Into<String>) -> Self {
        self.db_name = Some(db_name.into());
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

    /// Sets the consistency level of every request.
    #[must_use]
    pub fn with_consistency_level(mut self, consistency_level: impl Into<String>) -> Self {
        self.consistency_level = consistency_level.into();
        self
    }

    /// Enables or disables the duplicate primary-key probe of `add_docs`.
    ///
    /// Disabling it restores the append-only service behaviour (a duplicate
    /// primary key becomes a second row) for bulk ingestion where collisions
    /// are impossible by construction.
    #[must_use]
    pub fn with_reject_duplicate_primary_keys(mut self, reject: bool) -> Self {
        self.reject_duplicate_primary_keys = reject;
        self
    }
}

/// A [`VectorStore`] backed by a Milvus server.
///
/// The store holds one HTTP client, so it is cheap to share behind an `Arc` and
/// every method takes `&self`.
///
/// # Errors
///
/// [`MilvusVectorStore::new`] rejects an empty endpoint; the contract methods
/// return the errors documented on [`VectorStore`], plus
/// [`Error::Unsupported`](openjiuwen_common::error::Error) for a filter Milvus
/// cannot express.
#[derive(Debug, Clone)]
pub struct MilvusVectorStore {
    client: HttpClient,
    config: MilvusConfig,
    endpoint: String,
}

impl MilvusVectorStore {
    /// Builds a store from `config`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`](openjiuwen_common::error::Error) for an empty
    /// endpoint and propagates the HTTP client's configuration errors.
    pub fn new(config: MilvusConfig) -> Result<Self> {
        let endpoint = config.endpoint.trim().trim_end_matches('/').to_string();
        if endpoint.is_empty() {
            return Err(Error::config("Milvus endpoint must not be empty"));
        }
        let client = HttpClient::new(HttpClientConfig::default().with_timeout(config.timeout))?;
        Ok(Self {
            client,
            config,
            endpoint,
        })
    }

    /// Returns the configuration this store was built with.
    #[must_use]
    pub fn config(&self) -> &MilvusConfig {
        &self.config
    }

    /// The absolute URL of `path` on the configured endpoint.
    fn url(&self, path: &str) -> String {
        format!("{}{path}", self.endpoint)
    }

    /// Builds a `POST` request carrying `body` and the configured headers.
    fn build(&self, path: &str, body: &Value) -> Result<HttpRequest> {
        let mut body = body.clone();
        if let (Some(db_name), Some(entries)) = (&self.config.db_name, body.as_object_mut()) {
            entries.insert("dbName".to_string(), json!(db_name));
        }
        let mut request = HttpRequest::post(self.url(path))
            .header("Content-Type", "application/json")
            .timeout(self.config.timeout)
            .json(&body)?;
        if let Some(token) = &self.config.token {
            request = request.bearer(token.clone());
        }
        Ok(request)
    }

    /// Sends one v2 request and returns its envelope's `data` member.
    async fn call(&self, path: &str, body: &Value, context: &str) -> Result<Value> {
        let response = self.client.execute(self.build(path, body)?).await?;
        envelope(&response, context)
    }

    /// Reads the schema of an existing collection.
    async fn load_schema(&self, collection: &str) -> Result<CollectionSchema> {
        self.get_schema(collection)
            .await?
            .ok_or_else(|| collection_not_found(collection))
    }

    /// Fails when `collection` already stores one of `rows`' primary keys.
    ///
    /// Milvus `insert` appends rows without consulting the stored ones, so a
    /// duplicate primary key is invisible to the insert itself. This probe
    /// queries the keys of the batch (bounded by the batch length) and turns
    /// the collision into the insert-only contract of [`VectorStore`]: a
    /// validation error naming the colliding keys, with nothing written.
    async fn reject_stored_keys(
        &self,
        collection: &str,
        schema: &CollectionSchema,
        rows: &[Value],
    ) -> Result<()> {
        let Some(primary) = schema.primary_key() else {
            return Ok(());
        };
        let mut keys = Vec::with_capacity(rows.len());
        for row in rows {
            if let Some(value) = row.get(&primary.name) {
                keys.push(literal(value)?);
            }
        }
        if keys.is_empty() {
            return Ok(());
        }
        let body = json!({
            "collectionName": collection,
            "filter": format!("{} in [{}]", primary.name, keys.join(", ")),
            "outputFields": [primary.name],
            "limit": rows.len(),
        });
        let context = format!("checking primary keys of collection {collection:?}");
        let data = self.call(ENTITIES_QUERY, &body, &context).await?;
        let stored: Vec<Value> = data
            .as_array()
            .map(|entries| {
                entries
                    .iter()
                    .filter_map(|entry| entry.get(&primary.name).cloned())
                    .filter(|value| !value.is_null())
                    .collect()
            })
            .unwrap_or_default();
        if stored.is_empty() {
            return Ok(());
        }
        Err(Error::validation(format!(
            "adding documents to collection {collection:?} was rejected: primary key(s) {} are \
             already stored (set `reject_duplicate_primary_keys` to false to append duplicates)",
            summarize(&stored)
        )))
    }
}

/// Joins up to [`KEY_EXCERPT`] primary keys for an error message.
fn summarize(keys: &[Value]) -> String {
    let rendered: Vec<String> = keys
        .iter()
        .take(KEY_EXCERPT)
        .map(|key| match key {
            Value::String(text) => format!("{text:?}"),
            other => other.to_string(),
        })
        .collect();
    if keys.len() > KEY_EXCERPT {
        format!(
            "{}, and {} more",
            rendered.join(", "),
            keys.len() - KEY_EXCERPT
        )
    } else {
        rendered.join(", ")
    }
}

#[async_trait]
impl VectorStore for MilvusVectorStore {
    async fn create_collection(&self, collection: &str, schema: CollectionSchema) -> Result<()> {
        schema.validate()?;
        let body = create_body(
            collection,
            &schema,
            self.config.metric,
            &self.config.consistency_level,
        )?;
        let context = format!("creating collection {collection:?}");
        self.call(COLLECTIONS_CREATE, &body, &context).await?;
        Ok(())
    }

    async fn delete_collection(&self, collection: &str) -> Result<()> {
        let body = json!({"collectionName": collection});
        let context = format!("deleting collection {collection:?}");
        match self.call(COLLECTIONS_DROP, &body, &context).await {
            Ok(_) => Ok(()),
            Err(error) if error.kind() == ErrorKind::NotFound => Ok(()),
            Err(error) => Err(error),
        }
    }

    async fn collection_exists(&self, collection: &str) -> Result<bool> {
        let body = json!({"collectionName": collection});
        let context = format!("checking collection {collection:?}");
        let data = match self.call(COLLECTIONS_HAS, &body, &context).await {
            Ok(data) => data,
            Err(error) if error.kind() == ErrorKind::NotFound => return Ok(false),
            Err(error) => return Err(error),
        };
        match &data {
            Value::Bool(has) => Ok(*has),
            Value::Null => Ok(false),
            Value::Object(entry) => match entry.get("has") {
                None | Some(Value::Null) => Ok(false),
                Some(Value::Bool(has)) => Ok(*has),
                Some(other) => Err(Error::validation(format!(
                    "Milvus returned a non-boolean `has` value: {other}"
                ))),
            },
            other => Err(Error::validation(format!(
                "Milvus returned an unexpected `has` payload: {other}"
            ))),
        }
    }

    async fn get_schema(&self, collection: &str) -> Result<Option<CollectionSchema>> {
        let body = json!({"collectionName": collection});
        let context = format!("describing collection {collection:?}");
        match self.call(COLLECTIONS_DESCRIBE, &body, &context).await {
            Ok(data) => parse_schema(&data).map(Some),
            Err(error) if error.kind() == ErrorKind::NotFound => Ok(None),
            Err(error) => Err(error),
        }
    }

    /// Inserts documents, returning how many the service accepted.
    ///
    /// The batch is validated in full locally (so a failed validation inserts
    /// nothing), and — unless
    /// [`MilvusConfig::reject_duplicate_primary_keys`] is off — the primary
    /// keys already stored in the collection are probed first, so an insert
    /// that would duplicate one fails with a validation error naming the
    /// colliding keys instead of appending a second row.
    async fn add_docs(&self, collection: &str, docs: Vec<Map<String, Value>>) -> Result<usize> {
        let schema = self.load_schema(collection).await?;
        let prepared = prepare_docs(&schema, docs)?;
        let count = prepared.len();
        if self.config.reject_duplicate_primary_keys {
            self.reject_stored_keys(collection, &schema, &prepared)
                .await?;
        }
        let body = json!({"collectionName": collection, "data": prepared});
        let context = format!("inserting documents into collection {collection:?}");
        let data = self.call(ENTITIES_INSERT, &body, &context).await?;
        Ok(count_field(&data, "insertCount").unwrap_or(count))
    }

    async fn search(
        &self,
        collection: &str,
        query: &VectorQuery,
    ) -> Result<Vec<VectorSearchResult>> {
        if query.top_k == 0 {
            return Ok(Vec::new());
        }
        let schema = self.load_schema(collection).await?;
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
        let primary = schema
            .primary_key()
            .map_or("id", |field| field.name.as_str());
        let mut body = json!({
            "collectionName": collection,
            "data": [&query.query_vector],
            "annsField": field.name,
            "limit": query.top_k,
            "outputFields": ["*"],
        });
        if let (Some(expression), Some(entries)) =
            (compile_filter(&query.filters)?, body.as_object_mut())
        {
            entries.insert("filter".to_string(), json!(expression));
        }
        let context = format!("searching collection {collection:?}");
        let data = self.call(ENTITIES_SEARCH, &body, &context).await?;
        parse_hits(&data, primary, self.config.metric, query.top_k)
    }

    async fn delete_docs_by_ids(&self, collection: &str, ids: &[String]) -> Result<usize> {
        if ids.is_empty() {
            self.load_schema(collection).await?;
            return Ok(0);
        }
        let body = json!({"collectionName": collection, "ids": ids});
        let context = format!("deleting documents from collection {collection:?}");
        let data = self.call(ENTITIES_DELETE, &body, &context).await?;
        Ok(count_field(&data, "deleteCount").unwrap_or(ids.len()))
    }

    async fn delete_docs_by_filters(&self, collection: &str, filters: &Value) -> Result<usize> {
        let schema = self.load_schema(collection).await?;
        let primary = schema
            .primary_key()
            .ok_or_else(|| Error::validation("collection schema declares no primary key field"))?;
        let expression = compile_filter(filters)?;
        let mut body = json!({
            "collectionName": collection,
            "outputFields": [primary.name],
            "limit": FILTER_DELETE_LIMIT,
        });
        if let (Some(expression), Some(entries)) = (&expression, body.as_object_mut()) {
            entries.insert("filter".to_string(), json!(expression));
        }
        let context = format!("querying documents in collection {collection:?}");
        let data = self.call(ENTITIES_QUERY, &body, &context).await?;
        let ids: Vec<Value> = data
            .as_array()
            .map(|rows| {
                rows.iter()
                    .filter_map(|row| row.get(&primary.name).cloned())
                    .filter(|value| !value.is_null())
                    .collect()
            })
            .unwrap_or_default();
        if ids.is_empty() {
            // Nothing was read back, so let the service report its own count.
            let mut body = json!({"collectionName": collection});
            if let (Some(expression), Some(entries)) = (&expression, body.as_object_mut()) {
                entries.insert("filter".to_string(), json!(expression));
            }
            let context = format!("deleting documents from collection {collection:?}");
            let data = self.call(ENTITIES_DELETE, &body, &context).await?;
            return Ok(count_field(&data, "deleteCount").unwrap_or(0));
        }
        if ids.len() >= FILTER_DELETE_LIMIT {
            // The query read exactly the cap, so matches beyond it were never
            // seen. Deleting the visible window and reporting its size would be
            // a silent partial delete; fail before any deletion instead, like
            // the Elasticsearch backend does at its own scan cap.
            return Err(Error::unsupported(format!(
                "deleting documents from collection {collection:?} by filter is incomplete: the \
                 filter matched at least {FILTER_DELETE_LIMIT} documents, which is the most one \
                 query can read; narrow the filter so fewer documents match"
            )));
        }
        let removed = ids.len();
        let body = json!({"collectionName": collection, "ids": ids});
        let context = format!("deleting documents from collection {collection:?}");
        self.call(ENTITIES_DELETE, &body, &context).await?;
        Ok(removed)
    }

    async fn list_collection_names(&self) -> Result<Vec<String>> {
        let body = json!({});
        let data = self
            .call(COLLECTIONS_LIST, &body, "listing collections")
            .await?;
        let mut names = collection_names(&data)?;
        names.sort();
        names.dedup();
        Ok(names)
    }
}

/// A Milvus service error (`code != 0`) carried inside
/// [`Error::External`](openjiuwen_common::error::Error).
#[derive(Debug)]
struct ServiceError {
    code: i64,
    message: String,
}

impl std::fmt::Display for ServiceError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "Milvus code {}: {}", self.code, self.message)
    }
}

impl std::error::Error for ServiceError {}

/// Builds the `collections/create` request body.
fn create_body(
    collection: &str,
    schema: &CollectionSchema,
    metric: Metric,
    consistency_level: &str,
) -> Result<Value> {
    let primary = schema
        .primary_key()
        .ok_or_else(|| Error::validation("collection schema declares no primary key field"))?;
    let mut fields = Vec::with_capacity(schema.fields.len());
    for field in &schema.fields {
        fields.push(field_body(field)?);
    }
    let indexes: Vec<Value> = schema
        .vector_fields()
        .into_iter()
        .map(|field| {
            json!({
                "fieldName": field.name,
                "indexName": field.name,
                "metricType": metric_type(metric),
                "params": {"index_type": "AUTOINDEX"},
            })
        })
        .collect();
    let mut body = Map::new();
    body.insert("collectionName".to_string(), json!(collection));
    body.insert(
        "schema".to_string(),
        json!({
            "autoID": primary.auto_id,
            "enableDynamicField": schema.enable_dynamic_field,
            "fields": fields,
        }),
    );
    body.insert("indexParams".to_string(), json!(indexes));
    if !consistency_level.is_empty() {
        body.insert(
            "params".to_string(),
            json!({"consistencyLevel": consistency_level}),
        );
    }
    Ok(Value::Object(body))
}

/// Maps one [`FieldSchema`] onto the Milvus create-schema field object.
fn field_body(field: &FieldSchema) -> Result<Value> {
    let data_type = milvus_type(field.dtype)?;
    let mut body = Map::new();
    body.insert("fieldName".to_string(), json!(field.name));
    body.insert("dataType".to_string(), json!(data_type));
    body.insert("isPrimary".to_string(), json!(field.is_primary));
    if let Some(description) = &field.description {
        body.insert("description".to_string(), json!(description));
    }
    if field.dtype == VectorDataType::Varchar {
        if let Some(max_length) = field.max_length {
            body.insert("maxLength".to_string(), json!(max_length));
        }
    }
    if field.is_vector() {
        if let Some(dim) = field.dim {
            body.insert("dim".to_string(), json!(dim));
        }
    }
    Ok(Value::Object(body))
}

/// The Milvus data type of a schema field.
fn milvus_type(dtype: VectorDataType) -> Result<&'static str> {
    Ok(match dtype {
        VectorDataType::Varchar => "VarChar",
        VectorDataType::FloatVector => "FloatVector",
        VectorDataType::Int64 => "Int64",
        VectorDataType::Int32 => "Int32",
        VectorDataType::Float => "Float",
        VectorDataType::Double => "Double",
        VectorDataType::Bool => "Bool",
        VectorDataType::Json => "JSON",
        VectorDataType::Array => {
            return Err(Error::validation(
                "Milvus ARRAY fields need an element type that VectorDataType cannot express; \
                 use a JSON field instead",
            ));
        }
    })
}

/// The Milvus metric name of a [`Metric`].
fn metric_type(metric: Metric) -> &'static str {
    match metric {
        Metric::Cosine => "COSINE",
        Metric::Dot => "IP",
        Metric::Euclidean => "L2",
    }
}

/// Maps a Milvus `describe` payload onto a [`CollectionSchema`].
fn parse_schema(data: &Value) -> Result<CollectionSchema> {
    let fields = data
        .get("fields")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            Error::validation(format!(
                "Milvus describe returned no field list: {}",
                excerpt(&data.to_string())
            ))
        })?;
    let mut parsed = Vec::with_capacity(fields.len());
    for raw in fields {
        parsed.push(parse_field(raw)?);
    }
    Ok(CollectionSchema {
        fields: parsed,
        description: data
            .get("collectionDescription")
            .or_else(|| data.get("description"))
            .and_then(Value::as_str)
            .filter(|description| !description.is_empty())
            .map(str::to_string),
        enable_dynamic_field: data
            .get("enableDynamicField")
            .and_then(Value::as_bool)
            .unwrap_or(false),
    })
}

/// Maps one Milvus `describe` field onto a [`FieldSchema`].
fn parse_field(raw: &Value) -> Result<FieldSchema> {
    let name = raw
        .get("fieldName")
        .or_else(|| raw.get("name"))
        .and_then(Value::as_str)
        .ok_or_else(|| {
            Error::validation(format!(
                "Milvus describe returned a field without a name: {}",
                excerpt(&raw.to_string())
            ))
        })?;
    let raw_type = raw.get("type").and_then(Value::as_str).ok_or_else(|| {
        Error::validation(format!(
            "Milvus describe returned field {name:?} without a type"
        ))
    })?;
    let params: &[Value] = raw
        .get("params")
        .and_then(Value::as_array)
        .map_or(&[], Vec::as_slice);
    let element = raw.get("elementTypeParams").and_then(Value::as_object);
    Ok(FieldSchema {
        name: name.to_string(),
        dtype: parse_data_type(raw_type)?,
        is_primary: raw
            .get("isPrimary")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        auto_id: raw
            .get("autoID")
            .or_else(|| raw.get("autoId"))
            .and_then(Value::as_bool)
            .unwrap_or(false),
        max_length: param_usize(params, element, "max_length")
            .or_else(|| param_usize(params, element, "maxLength")),
        dim: param_usize(params, element, "dim"),
        description: raw
            .get("description")
            .and_then(Value::as_str)
            .filter(|description| !description.is_empty())
            .map(str::to_string),
        default_value: None,
    })
}

/// Reads one `params`/`elementTypeParams` entry as a `usize`.
fn param_usize(params: &[Value], element: Option<&Map<String, Value>>, key: &str) -> Option<usize> {
    let from_params = params.iter().find_map(|param| {
        let name = param.get("key").and_then(Value::as_str)?;
        if name.eq_ignore_ascii_case(key) {
            parse_usize(param.get("value")?)
        } else {
            None
        }
    });
    from_params.or_else(|| {
        element.and_then(|entries| {
            entries
                .iter()
                .find(|(name, _)| name.eq_ignore_ascii_case(key))
                .and_then(|(_, value)| parse_usize(value))
        })
    })
}

/// Parses a Milvus parameter that may be a number or a numeric string.
fn parse_usize(value: &Value) -> Option<usize> {
    match value {
        Value::Number(number) => number
            .as_u64()
            .and_then(|value| usize::try_from(value).ok()),
        Value::String(text) => text.trim().parse().ok(),
        _ => None,
    }
}

/// Maps a Milvus type name (any case) onto a [`VectorDataType`].
fn parse_data_type(raw: &str) -> Result<VectorDataType> {
    let normalized = raw.to_ascii_lowercase().replace('_', "");
    Ok(match normalized.as_str() {
        "varchar" => VectorDataType::Varchar,
        "floatvector" => VectorDataType::FloatVector,
        "int64" => VectorDataType::Int64,
        "int32" => VectorDataType::Int32,
        "float" => VectorDataType::Float,
        "double" => VectorDataType::Double,
        "bool" | "boolean" => VectorDataType::Bool,
        "json" => VectorDataType::Json,
        "array" => VectorDataType::Array,
        other => {
            return Err(Error::validation(format!(
                "Milvus field type {other:?} has no VectorDataType equivalent"
            )));
        }
    })
}

/// Extracts the collection names of a `collections/list` payload.
fn collection_names(data: &Value) -> Result<Vec<String>> {
    let items = match data {
        Value::Array(items) => items,
        Value::Object(entries) => entries
            .get("collectionNames")
            .or_else(|| entries.get("collections"))
            .and_then(Value::as_array)
            .map_or(&[][..], Vec::as_slice),
        Value::Null => &[],
        other => {
            return Err(Error::validation(format!(
                "Milvus returned an unexpected collection list: {other}"
            )));
        }
    };
    let mut names = Vec::with_capacity(items.len());
    for item in items {
        match item {
            Value::String(name) => names.push(name.clone()),
            Value::Object(entry) => {
                let name = entry
                    .get("collectionName")
                    .or_else(|| entry.get("name"))
                    .and_then(Value::as_str);
                if let Some(name) = name {
                    names.push(name.to_string());
                }
            }
            other => {
                return Err(Error::validation(format!(
                    "Milvus returned a non-string collection name: {other}"
                )));
            }
        }
    }
    Ok(names)
}

/// Resolves the vector field a query scores against.
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

/// Validates a batch of raw documents and converts them into insert rows.
///
/// Nothing is sent when any document is invalid, so a failed call inserts
/// nothing — the same all-or-nothing contract as the in-memory backend.
fn prepare_docs(schema: &CollectionSchema, docs: Vec<Map<String, Value>>) -> Result<Vec<Value>> {
    let primary = schema
        .primary_key()
        .ok_or_else(|| Error::validation("collection schema declares no primary key field"))?;
    let mut prepared = Vec::with_capacity(docs.len());
    let mut seen: BTreeMap<String, usize> = BTreeMap::new();
    for (position, raw) in docs.into_iter().enumerate() {
        let doc = prepare_doc(schema, primary, raw, position)?;
        let id = doc
            .get(&primary.name)
            .map(ToString::to_string)
            .unwrap_or_default();
        if let Some(first) = seen.insert(id.clone(), position) {
            return Err(Error::validation(format!(
                "duplicate primary key {id} in the batch (positions {first} and {position})"
            )));
        }
        prepared.push(Value::Object(doc));
    }
    Ok(prepared)
}

/// Validates one document, applies defaults, and checks every declared field.
fn prepare_doc(
    schema: &CollectionSchema,
    primary: &FieldSchema,
    raw: Map<String, Value>,
    position: usize,
) -> Result<Map<String, Value>> {
    let mut fields = raw;
    for field in &schema.fields {
        if let (false, Some(default)) = (fields.contains_key(&field.name), &field.default_value) {
            fields.insert(field.name.clone(), default.clone());
        }
    }
    let id = resolve_id(primary, &mut fields, position)?;
    for field in &schema.fields {
        match fields.get(&field.name) {
            Some(value) if field.is_vector() => {
                parse_vector(field, value)?;
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
    }
    Ok(fields)
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
fn parse_vector(field: &FieldSchema, value: &Value) -> Result<()> {
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
        let number = item.as_f64().filter(|number| number.is_finite());
        if number.is_none() {
            return Err(Error::validation(format!(
                "vector field {:?} expects finite numbers, got component {item}",
                field.name
            )));
        }
    }
    Ok(())
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
        VectorDataType::FloatVector => parse_vector(field, value)?,
    }
    Ok(())
}

/// Reads one `<operation>Count` member of a response payload.
fn count_field(data: &Value, key: &str) -> Option<usize> {
    data.get(key)
        .and_then(Value::as_u64)
        .and_then(|count| usize::try_from(count).ok())
}

/// Normalizes one native Milvus distance into a score in `[0, 1]`.
///
/// `COSINE` returns a cosine similarity and `IP` a dot product, so both use
/// `(score + 1) / 2`; `L2` returns a distance, so it uses `max(0, (4 - d) / 4)`.
fn normalize_score(metric: Metric, distance: f64) -> f32 {
    if !distance.is_finite() {
        return 0.0;
    }
    let raw = match metric {
        Metric::Cosine | Metric::Dot => f64::midpoint(distance, 1.0),
        Metric::Euclidean => (4.0 - distance) / 4.0,
    };
    if raw.is_finite() {
        raw.clamp(0.0, 1.0) as f32
    } else {
        0.0
    }
}

/// Maps an `entities/search` payload onto normalized, ordered search hits.
fn parse_hits(
    data: &Value,
    primary: &str,
    metric: Metric,
    top_k: usize,
) -> Result<Vec<VectorSearchResult>> {
    let items = data.as_array().map_or(&[][..], Vec::as_slice);
    // The v2 API answers one hit list per query; a flat hit list is tolerated.
    let rows = match items.first() {
        Some(Value::Array(_)) => items
            .first()
            .and_then(Value::as_array)
            .map_or(&[][..], Vec::as_slice),
        _ => items,
    };
    let mut hits = Vec::new();
    for row in rows {
        let object = row.as_object().ok_or_else(|| {
            Error::validation(format!("Milvus returned a non-object search hit: {row}"))
        })?;
        let distance = object
            .get("distance")
            .and_then(Value::as_f64)
            .unwrap_or(0.0);
        let mut fields = object
            .get("entity")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();
        let id = object
            .get("id")
            .cloned()
            .or_else(|| fields.get(primary).cloned());
        if let Some(id) = id {
            fields.insert("id".to_string(), id);
        }
        hits.push(VectorSearchResult::new(
            normalize_score(metric, distance),
            fields,
        ));
    }
    let key = |fields: &Map<String, Value>| {
        fields
            .get(primary)
            .or_else(|| fields.get("id"))
            .map(ToString::to_string)
            .unwrap_or_default()
    };
    hits.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| key(&left.fields).cmp(&key(&right.fields)))
    });
    hits.truncate(top_k);
    Ok(hits)
}

/// Decodes a Milvus v2 envelope and returns its `data` member.
fn envelope(response: &HttpResponse, context: &str) -> Result<Value> {
    let Some(payload) = serde_json::from_slice::<Value>(&response.body)
        .ok()
        .filter(Value::is_object)
    else {
        return Err(failure(response, context));
    };
    let code = payload.get("code").and_then(Value::as_i64);
    let message = payload.get("message").and_then(Value::as_str).unwrap_or("");
    if response.is_success() {
        return match code {
            Some(0) | None => Ok(payload.get("data").cloned().unwrap_or(payload)),
            Some(code) => Err(classify(code, message, context)),
        };
    }
    match code {
        Some(code) if code != 0 => Err(classify(code, message, context)),
        _ => Err(failure(response, context)),
    }
}

/// Classifies a non-zero Milvus service code as the matching [`Error`].
fn classify(code: i64, message: &str, context: &str) -> Error {
    let lowered = message.to_ascii_lowercase();
    let missing = lowered.contains("not found")
        || lowered.contains("not exist")
        || lowered.contains("does not exist")
        || lowered.contains("cannot find")
        || lowered.contains("can't find");
    if missing {
        return Error::not_found(format!("Milvus {context}: {message} (code {code})"));
    }
    let duplicate = lowered.contains("already exist") || lowered.contains("duplicate");
    if duplicate {
        return Error::validation(format!(
            "Milvus {context} was rejected: {message} (code {code})"
        ));
    }
    Error::external(ServiceError {
        code,
        message: format!("{context}: {message}"),
    })
}

/// A transport-level failure: status, URL and a body excerpt.
fn failure(response: &HttpResponse, context: &str) -> Error {
    Error::transport(format!(
        "Milvus {context} failed with HTTP {} for {}: {}",
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

/// A compiled filter clause.
enum Clause {
    /// The clause constrains nothing (`null`, `{}`, or a tautological `$and`).
    Everything,
    /// A Milvus boolean expression.
    Expr(String),
}

/// Compiles the shared filter language into a Milvus boolean expression.
///
/// `Ok(None)` means "no filter": `null` and `{}` match every row. A construct
/// Milvus cannot express is an `Unsupported` error instead of a dropped filter.
fn compile_filter(filters: &Value) -> Result<Option<String>> {
    match compile_clause(filters)? {
        Clause::Everything => Ok(None),
        Clause::Expr(expression) => Ok(Some(expression)),
    }
}

/// Compiles one filter node.
fn compile_clause(filters: &Value) -> Result<Clause> {
    match filters {
        Value::Null => Ok(Clause::Everything),
        Value::Object(entries) if entries.is_empty() => Ok(Clause::Everything),
        Value::Object(entries) => compile_object(entries),
        other => Err(Error::validation(format!(
            "Milvus filters must be a JSON object or null, got {other}"
        ))),
    }
}

/// Compiles the conjunction of every field predicate of one object.
fn compile_object(entries: &Map<String, Value>) -> Result<Clause> {
    let mut parts: Vec<String> = Vec::new();
    for (field, condition) in entries {
        let clause = match field.as_str() {
            "$and" => compile_logical("and", condition)?,
            "$or" => compile_logical("or", condition)?,
            operator if operator.starts_with('$') => {
                return Err(Error::unsupported(format!(
                    "Milvus filters cannot express the {operator:?} operator"
                )));
            }
            field => compile_field(field, condition)?,
        };
        if let Clause::Expr(expression) = clause {
            parts.push(expression);
        }
    }
    Ok(match parts.len() {
        0 => Clause::Everything,
        1 => Clause::Expr(parts.remove(0)),
        _ => Clause::Expr(parts.join(" and ")),
    })
}

/// Compiles an `$and`/`$or` operand list, parenthesizing each compound operand.
fn compile_logical(keyword: &str, condition: &Value) -> Result<Clause> {
    let operands = condition.as_array().ok_or_else(|| {
        Error::validation(format!(
            "Milvus filter operator ${keyword} expects an array, got {condition}"
        ))
    })?;
    let mut parts: Vec<String> = Vec::new();
    let mut any = false;
    for operand in operands {
        match compile_clause(operand)? {
            Clause::Everything => {
                if keyword == "or" {
                    // `true or X` constrains nothing.
                    return Ok(Clause::Everything);
                }
            }
            Clause::Expr(expression) => {
                any = true;
                parts.push(format!("({expression})"));
            }
        }
    }
    if !any {
        return Ok(Clause::Everything);
    }
    let separator = format!(" {keyword} ");
    Ok(Clause::Expr(parts.join(separator.as_str())))
}

/// Compiles the predicate of one field.
fn compile_field(field: &str, condition: &Value) -> Result<Clause> {
    check_field_name(field)?;
    match condition {
        Value::Object(operators)
            if !operators.is_empty() && operators.keys().all(|key| key.starts_with('$')) =>
        {
            let mut parts: Vec<String> = Vec::new();
            for (operator, operand) in operators {
                if let Clause::Expr(expression) = compile_operator(field, operator, operand)? {
                    parts.push(expression);
                }
            }
            Ok(match parts.len() {
                0 => Clause::Everything,
                _ => Clause::Expr(parts.join(" and ")),
            })
        }
        Value::Object(_) => Err(Error::unsupported(format!(
            "Milvus filters cannot express the object literal compared against field {field:?}"
        ))),
        value => Ok(Clause::Expr(format!("{field} == {}", literal(value)?))),
    }
}

/// Rejects field names that a Milvus expression cannot address.
fn check_field_name(field: &str) -> Result<()> {
    if field.contains('.') || field.contains('[') {
        return Err(Error::unsupported(format!(
            "Milvus filters cannot express the dotted field path {field:?}"
        )));
    }
    if field.is_empty()
        || !field
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || character == '_')
    {
        return Err(Error::unsupported(format!(
            "Milvus filters cannot express the field name {field:?}"
        )));
    }
    Ok(())
}

/// Compiles one comparison operator of a field.
fn compile_operator(field: &str, operator: &str, operand: &Value) -> Result<Clause> {
    let comparison = match operator {
        "$eq" => Some("=="),
        "$ne" => Some("!="),
        "$gt" => Some(">"),
        "$gte" => Some(">="),
        "$lt" => Some("<"),
        "$lte" => Some("<="),
        _ => None,
    };
    if let Some(comparison) = comparison {
        return Ok(Clause::Expr(format!(
            "{field} {comparison} {}",
            literal(operand)?
        )));
    }
    match operator {
        "$in" => compile_list(field, "in", operand),
        "$nin" => {
            if operand.as_array().is_some_and(Vec::is_empty) {
                // `not in []` matches everything.
                return Ok(Clause::Everything);
            }
            compile_list(field, "not in", operand)
        }
        other => Err(Error::unsupported(format!(
            "Milvus filters cannot express the {other:?} operator"
        ))),
    }
}

/// Compiles a `$in`/`$nin` membership test.
fn compile_list(field: &str, keyword: &str, operand: &Value) -> Result<Clause> {
    let items = operand.as_array().ok_or_else(|| {
        Error::validation(format!(
            "Milvus filter operator ${keyword} of field {field:?} expects an array, got {operand}"
        ))
    })?;
    if items.is_empty() {
        return Err(Error::unsupported(format!(
            "Milvus filters cannot express the empty membership list of field {field:?}"
        )));
    }
    let mut values = Vec::with_capacity(items.len());
    for item in items {
        values.push(literal(item)?);
    }
    Ok(Clause::Expr(format!(
        "{field} {keyword} [{}]",
        values.join(", ")
    )))
}

/// Renders one literal of a filter expression.
fn literal(value: &Value) -> Result<String> {
    match value {
        Value::String(text) => {
            let escaped = text.replace('\\', "\\\\").replace('"', "\\\"");
            Ok(format!("\"{escaped}\""))
        }
        Value::Number(number) => Ok(number.to_string()),
        Value::Bool(boolean) => Ok(boolean.to_string()),
        Value::Null => Err(Error::unsupported(
            "Milvus filters cannot compare against a null operand",
        )),
        Value::Array(_) => Err(Error::unsupported(
            "Milvus filters cannot express an array literal",
        )),
        Value::Object(_) => Err(Error::unsupported(
            "Milvus filters cannot express an object literal",
        )),
    }
}

#[cfg(test)]
mod tests;
