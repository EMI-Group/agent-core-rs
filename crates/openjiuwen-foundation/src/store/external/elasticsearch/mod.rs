//! Elasticsearch vector and document stores over the Elasticsearch REST API.
//!
//! Two adapters share one REST client:
//!
//! - `ElasticsearchVectorStore` ([`vector`]) implements the
//!   [`VectorStore`](crate::store::vector::VectorStore) contract of
//!   [`crate::store::vector`] on top of an Elasticsearch `dense_vector` index;
//! - `ElasticsearchDocumentStore` ([`document`]) implements the
//!   [`DocumentStore`](crate::store::document::DocumentStore) contract of
//!   [`crate::store::document`] on top of plain index documents.
//!
//! One store-level collection is one Elasticsearch index named
//! `{index_prefix}__{collection}`; [`ElasticsearchConfig::index_prefix`]
//! defaults to `"openjiuwen"`, so both adapters share one naming rule and one
//! identity space. `ElasticsearchClient` owns the HTTP client, the index
//! naming and the authentication headers; both stores build it once and use it
//! for every request.
//!
//! Only the REST transport is used, through
//! [`openjiuwen_common::http::HttpClient`]: no vendor SDK is involved. A non-2xx
//! response becomes an [`Error`] whose kind is `Transport`,
//! naming the status code, the URL and an excerpt of the body (`failure`).
//!
//! # Schema round-trip
//!
//! Elasticsearch mappings cannot carry every member of a
//! [`CollectionSchema`] (the primary-key flag, `auto_id`, `max_length`,
//! `enable_dynamic_field`), so `create_index_body` stores the serialized
//! schema as a JSON string under the reserved index-metadata key
//! `_meta.openjiuwen_schema` in addition to the derived `properties` mapping.
//! `schema_from_index` prefers that marker, which restores the exact schema;
//! an index without it (created by someone else) is reconstructed from
//! `mappings.properties` by `schema_from_properties`, which synthesizes the
//! `id` primary key the mapping cannot declare.

use std::time::Duration;

use base64::Engine as _;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::http::{HttpClient, HttpClientConfig, HttpRequest, HttpResponse};
use openjiuwen_common::security::SecretString;

use crate::store::document::{validate_collection, ID_FIELD};
use crate::store::vector::{CollectionSchema, FieldSchema, Metric, VectorDataType};

pub mod document;
pub mod query;
pub mod vector;

pub use document::ElasticsearchDocumentStore;
pub use vector::ElasticsearchVectorStore;

/// Default total timeout of one Elasticsearch request.
const fn default_timeout() -> Duration {
    Duration::from_secs(30)
}

/// Default prefix of the indexes this adapter creates.
fn default_index_prefix() -> String {
    "openjiuwen".to_string()
}

/// Index-metadata key holding the serialized [`CollectionSchema`].
const SCHEMA_META_KEY: &str = "openjiuwen_schema";

/// Number of documents read per page while collecting ids for a filtered delete.
pub(crate) const FILTER_DELETE_PAGE: usize = 500;

/// Maximum number of response-body characters echoed in an error message.
pub(crate) const ERROR_BODY_EXCERPT: usize = 200;

/// Bounded result window of a document query without an explicit `limit`.
pub(crate) const DEFAULT_QUERY_LIMIT: usize = 10_000;

/// Configuration of an Elasticsearch backend.
///
/// The struct is `serde`-deserializable with a default for every field, so a
/// backend can be built straight from a JSON configuration object; the builder
/// setters exist for code that would rather not spell out every field. The
/// password and the API key are [`SecretString`]s, so they are redacted from
/// `Debug` output and never logged.
///
/// ```
/// use std::time::Duration;
///
/// use openjiuwen_foundation::store::external::elasticsearch::ElasticsearchConfig;
/// use openjiuwen_foundation::store::vector::Metric;
///
/// let config = ElasticsearchConfig::new("http://localhost:9200")
///     .with_index_prefix("myapp")
///     .with_metric(Metric::Euclidean)
///     .with_timeout(Duration::from_secs(5));
///
/// assert_eq!(config.index_prefix, "myapp");
/// assert_eq!(config.metric, Metric::Euclidean);
/// assert_eq!(config.timeout, Duration::from_secs(5));
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, rename_all = "snake_case")]
pub struct ElasticsearchConfig {
    /// Base URL of the Elasticsearch node, for example `http://localhost:9200`.
    pub endpoint: String,
    /// Prefix of every index this backend creates.
    pub index_prefix: String,
    /// Optional user name of HTTP basic authentication.
    pub username: Option<String>,
    /// Optional password of HTTP basic authentication.
    pub password: Option<SecretString>,
    /// Optional API key, sent as `Authorization: ApiKey <key>`.
    pub api_key: Option<SecretString>,
    /// Similarity of the `dense_vector` fields of a created index.
    pub metric: Metric,
    /// Timeout of a single HTTP request.
    pub timeout: Duration,
}

impl Default for ElasticsearchConfig {
    fn default() -> Self {
        Self {
            endpoint: String::new(),
            index_prefix: default_index_prefix(),
            username: None,
            password: None,
            api_key: None,
            metric: Metric::default(),
            timeout: default_timeout(),
        }
    }
}

impl ElasticsearchConfig {
    /// Creates a configuration pointing at `endpoint` and using every default.
    #[must_use]
    pub fn new(endpoint: impl Into<String>) -> Self {
        Self {
            endpoint: endpoint.into(),
            ..Self::default()
        }
    }

    /// Sets the prefix of every index this backend creates.
    #[must_use]
    pub fn with_index_prefix(mut self, index_prefix: impl Into<String>) -> Self {
        self.index_prefix = index_prefix.into();
        self
    }

    /// Sets the user name of HTTP basic authentication.
    #[must_use]
    pub fn with_username(mut self, username: impl Into<String>) -> Self {
        self.username = Some(username.into());
        self
    }

    /// Sets the password of HTTP basic authentication.
    #[must_use]
    pub fn with_password(mut self, password: impl Into<String>) -> Self {
        self.password = Some(SecretString::new(password));
        self
    }

    /// Sets the API key sent as `Authorization: ApiKey <key>`.
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
}

/// The shared REST client of both Elasticsearch adapters.
///
/// The client holds one [`HttpClient`], so it is cheap to clone and every
/// method takes `&self`. It owns the index naming, the authentication headers
/// and the error mapping, and it is the only place that talks HTTP.
///
/// # Errors
///
/// [`ElasticsearchClient::new`] rejects an empty endpoint or index prefix and
/// any endpoint the HTTP client cannot accept; the request helpers return the
/// errors documented on [`Self::send`].
#[derive(Debug, Clone)]
pub(crate) struct ElasticsearchClient {
    http: HttpClient,
    config: ElasticsearchConfig,
    endpoint: String,
}

impl ElasticsearchClient {
    /// Builds a client from `config` with a default-tuned HTTP client.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`](openjiuwen_common::error::Error) for an empty
    /// endpoint or index prefix, and propagates the HTTP client's configuration
    /// errors.
    pub(crate) fn new(config: ElasticsearchConfig) -> Result<Self> {
        let endpoint = checked_endpoint(&config)?;
        let http = HttpClient::new(HttpClientConfig::default().with_timeout(config.timeout))?;
        Ok(Self {
            http,
            config,
            endpoint,
        })
    }

    /// Builds a client that reuses `http`, for callers with their own tuning.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`](openjiuwen_common::error::Error) for an empty
    /// endpoint or index prefix.
    pub(crate) fn with_client(http: HttpClient, config: ElasticsearchConfig) -> Result<Self> {
        let endpoint = checked_endpoint(&config)?;
        Ok(Self {
            http,
            config,
            endpoint,
        })
    }

    /// The configuration this client was built from.
    pub(crate) fn config(&self) -> &ElasticsearchConfig {
        &self.config
    }

    /// The validated index prefix, without surrounding whitespace.
    pub(crate) fn index_prefix(&self) -> &str {
        self.config.index_prefix.trim()
    }

    /// The index name of `collection`, e.g. `openjiuwen__docs`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`](openjiuwen_common::error::Error) when the
    /// collection is not a valid collection name or the derived index name is
    /// not a valid Elasticsearch index name; the message names both.
    pub(crate) fn index_name(&self, collection: &str) -> Result<String> {
        validate_collection(collection)?;
        let index = format!("{}__{collection}", self.index_prefix());
        validate_index_name(&index).map_err(|err| {
            Error::validation(format!("{err} (derived from collection {collection:?})"))
        })?;
        Ok(index)
    }

    /// The collection name of `index`, or `None` when it is not ours.
    pub(crate) fn collection_from_index(&self, index: &str) -> Option<String> {
        let rest = index
            .strip_prefix(self.index_prefix())?
            .strip_prefix("__")?;
        (!rest.is_empty()).then(|| rest.to_string())
    }

    /// The path of one index with a suffix such as `/_search`.
    ///
    /// A method rather than an associated function so call sites read
    /// `client.path(index, "/_search")` next to `client.url(path)`.
    #[allow(clippy::unused_self)]
    pub(crate) fn path(&self, index: &str, suffix: &str) -> String {
        format!("/{}{suffix}", encode_segment(index))
    }

    /// The path listing every index of this backend, wildcard included.
    pub(crate) fn indexes_path(&self) -> String {
        format!("/{}__*", self.index_prefix())
    }

    /// The absolute URL of `path` on the configured endpoint.
    pub(crate) fn url(&self, path: &str) -> String {
        format!("{}{path}", self.endpoint)
    }

    /// Decorates a request with the timeout and the configured credentials.
    ///
    /// Basic authentication wins over an API key when both are configured.
    pub(crate) fn apply(&self, request: HttpRequest) -> HttpRequest {
        let mut request = request.timeout(self.config.timeout);
        if let (Some(username), Some(password)) = (&self.config.username, &self.config.password) {
            let credentials = base64::engine::general_purpose::STANDARD
                .encode(format!("{username}:{}", password.expose_secret()));
            request = request.header("authorization", format!("Basic {credentials}"));
        } else if let Some(api_key) = &self.config.api_key {
            request = request.header(
                "authorization",
                format!("ApiKey {}", api_key.expose_secret()),
            );
        }
        request
    }

    /// Sends `request` unchanged, without interpreting the status code.
    ///
    /// # Errors
    ///
    /// Propagates the transport errors of the HTTP client.
    pub(crate) async fn send_raw(&self, request: HttpRequest) -> Result<HttpResponse> {
        self.http.execute(self.apply(request)).await
    }

    /// Sends `request` and turns any non-2xx response into a transport error.
    ///
    /// # Errors
    ///
    /// Returns the HTTP client's transport errors and, for a non-2xx status,
    /// [`failure`] with `context` naming the operation.
    pub(crate) async fn send(&self, request: HttpRequest, context: &str) -> Result<HttpResponse> {
        let response = self.send_raw(request).await?;
        if response.is_success() {
            Ok(response)
        } else {
            Err(failure(&response, context))
        }
    }
}

/// Trims and checks the endpoint, and rejects an empty index prefix.
fn checked_endpoint(config: &ElasticsearchConfig) -> Result<String> {
    let endpoint = config.endpoint.trim().trim_end_matches('/').to_string();
    if endpoint.is_empty() {
        return Err(Error::config("Elasticsearch endpoint must not be empty"));
    }
    if config.index_prefix.trim().is_empty() {
        return Err(Error::config(
            "Elasticsearch index prefix must not be empty",
        ));
    }
    Ok(endpoint)
}

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
/// marker is not a readable schema or a property has an unsupported type.
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
/// `properties` is not an object, a property has no `type` or an unsupported
/// type, or the `id` field cannot be a primary key.
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
fn field_from_mapping(name: &str, mapping: &Value) -> Result<FieldSchema> {
    let dtype_name = mapping.get("type").and_then(Value::as_str).ok_or_else(|| {
        Error::validation(format!(
            "Elasticsearch index mapping of field {name:?} has no string `type`, found {mapping}"
        ))
    })?;
    let unsupported = || {
        Error::validation(format!(
            "Elasticsearch index mapping of field {name:?} has unsupported type {dtype_name:?}"
        ))
    };
    let mut field = match dtype_name {
        "dense_vector" => {
            let dim = mapping
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
        "keyword" | "text" => {
            let max_length = mapping
                .get("ignore_above")
                .and_then(Value::as_u64)
                .and_then(|limit| usize::try_from(limit).ok());
            FieldSchema {
                max_length,
                ..FieldSchema::new(name, VectorDataType::Varchar)
            }
        }
        "long" => FieldSchema::new(name, VectorDataType::Int64),
        "integer" => FieldSchema::new(name, VectorDataType::Int32),
        "float" => FieldSchema::new(name, VectorDataType::Float),
        "double" => FieldSchema::new(name, VectorDataType::Double),
        "boolean" => FieldSchema::new(name, VectorDataType::Bool),
        "object" | "nested" => FieldSchema::new(name, VectorDataType::Json),
        _ => return Err(unsupported()),
    };
    field.name = name.to_string();
    Ok(field)
}

/// The index names of a `GET /{prefix}__*` listing response, sorted.
///
/// The listing API answers with an object keyed by index name, so its keys are
/// the index names; any non-object payload yields an empty vector.
pub(crate) fn index_names_from(payload: &Value) -> Vec<String> {
    let Some(object) = payload.as_object() else {
        return Vec::new();
    };
    let mut names: Vec<String> = object.keys().cloned().collect();
    names.sort();
    names
}

/// Rejects names Elasticsearch cannot use as an index name.
///
/// # Errors
///
/// Returns [`Error::Validation`](openjiuwen_common::error::Error) naming the
/// offending index and the reason.
pub(crate) fn validate_index_name(name: &str) -> Result<()> {
    if name.is_empty() {
        return Err(Error::validation(
            "invalid Elasticsearch index name ``: must not be empty",
        ));
    }
    if let Some(character) = name.chars().find(|character| {
        matches!(
            character,
            '\\' | '/' | '*' | '?' | '"' | '<' | '>' | '|' | ',' | '#' | ':' | ' '
        )
    }) {
        return Err(Error::validation(format!(
            "invalid Elasticsearch index name `{name}`: must not contain {character:?}"
        )));
    }
    if let Some(character) = name
        .chars()
        .next()
        .filter(|character| matches!(character, '_' | '-' | '+'))
    {
        return Err(Error::validation(format!(
            "invalid Elasticsearch index name `{name}`: must not start with {character:?}"
        )));
    }
    if let Some(character) = name.chars().find(|character| character.is_uppercase()) {
        return Err(Error::validation(format!(
            "invalid Elasticsearch index name `{name}`: must be lowercase, found {character:?}"
        )));
    }
    Ok(())
}

/// Builds the "unexpected response" error of a failed request.
pub(crate) fn failure(response: &HttpResponse, context: &str) -> Error {
    Error::transport(format!(
        "Elasticsearch {context} failed with HTTP {} for {}: {}",
        response.status,
        response.url,
        excerpt(&response.text())
    ))
}

/// Shortens a response body for an error message.
pub(crate) fn excerpt(text: &str) -> String {
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
pub(crate) fn encode_segment(segment: &str) -> String {
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
    fn config_defaults_are_stable() {
        let config = ElasticsearchConfig::default();
        assert_eq!(config.endpoint, "");
        assert_eq!(config.index_prefix, "openjiuwen");
        assert!(config.username.is_none());
        assert!(config.password.is_none());
        assert!(config.api_key.is_none());
        assert_eq!(config.metric, Metric::Cosine);
        assert_eq!(config.timeout, Duration::from_secs(30));
    }

    #[test]
    fn config_serde_fills_missing_members_from_defaults() {
        let config: ElasticsearchConfig =
            serde_json::from_value(json!({"endpoint": "http://es:9200", "metric": "dot"}))
                .expect("deserialize config");
        assert_eq!(config.endpoint, "http://es:9200");
        assert_eq!(config.index_prefix, "openjiuwen");
        assert_eq!(config.metric, Metric::Dot);
        assert_eq!(config.timeout, Duration::from_secs(30));

        let json = serde_json::to_value(ElasticsearchConfig::default()).expect("serialize");
        assert_eq!(json["index_prefix"], json!("openjiuwen"));
        assert_eq!(json["endpoint"], json!(""));
    }

    #[test]
    fn builder_setters_fill_every_member() {
        let config = ElasticsearchConfig::new("http://es:9200")
            .with_index_prefix("app")
            .with_username("elastic")
            .with_password("secret")
            .with_api_key("key-1")
            .with_metric(Metric::Euclidean)
            .with_timeout(Duration::from_secs(5));
        assert_eq!(config.endpoint, "http://es:9200");
        assert_eq!(config.index_prefix, "app");
        assert_eq!(config.username.as_deref(), Some("elastic"));
        assert_eq!(
            config.password.as_ref().map(SecretString::expose_secret),
            Some("secret")
        );
        assert_eq!(
            config.api_key.as_ref().map(SecretString::expose_secret),
            Some("key-1")
        );
        assert_eq!(config.metric, Metric::Euclidean);
        assert_eq!(config.timeout, Duration::from_secs(5));
        assert!(
            !format!("{config:?}").contains("secret"),
            "password is redacted"
        );
    }

    #[test]
    fn client_rejects_empty_endpoint_and_prefix() {
        let error = ElasticsearchClient::new(ElasticsearchConfig::default())
            .expect_err("empty endpoint is rejected");
        assert_eq!(error.kind(), ErrorKind::Config);
        assert!(error.to_string().contains("endpoint must not be empty"));

        let config = ElasticsearchConfig::new("  http://es:9200/// ").with_index_prefix(" ");
        let error = ElasticsearchClient::new(config).expect_err("empty index prefix is rejected");
        assert_eq!(error.kind(), ErrorKind::Config);
        assert!(error.to_string().contains("index prefix"));
    }

    #[test]
    fn client_trims_the_endpoint_and_builds_paths() {
        let client = ElasticsearchClient::new(
            ElasticsearchConfig::new(" http://es:9200// ").with_index_prefix(" app "),
        )
        .expect("build client");
        assert_eq!(
            client.url("/_cluster/health"),
            "http://es:9200/_cluster/health"
        );
        assert_eq!(client.index_prefix(), "app");
        assert_eq!(client.index_name("docs").expect("index name"), "app__docs");
        assert_eq!(
            client.url(&client.path("app__docs", "/_search")),
            "http://es:9200/app__docs/_search"
        );
        assert_eq!(client.indexes_path(), "/app__*");
        assert_eq!(client.config().index_prefix, " app ");
    }

    #[test]
    fn index_name_validates_the_collection_and_the_result() {
        let client =
            ElasticsearchClient::new(ElasticsearchConfig::new("http://es:9200")).expect("client");
        assert_eq!(
            client.index_name("docs").expect("valid"),
            "openjiuwen__docs"
        );
        let error = client.index_name("a/b").expect_err("path separator");
        assert_eq!(error.kind(), ErrorKind::Validation);
        let error = client.index_name("Docs").expect_err("uppercase");
        assert_eq!(error.kind(), ErrorKind::Validation);
        let message = error.to_string();
        assert!(message.contains("openjiuwen__Docs"), "message: {message}");
        assert!(
            message.contains("collection \"Docs\""),
            "message: {message}"
        );
        let error = client.index_name("").expect_err("empty");
        assert_eq!(error.kind(), ErrorKind::Validation);
    }

    #[test]
    fn collection_from_index_strips_the_prefix() {
        let client =
            ElasticsearchClient::new(ElasticsearchConfig::new("http://es:9200")).expect("client");
        assert_eq!(
            client.collection_from_index("openjiuwen__docs"),
            Some("docs".to_string())
        );
        assert_eq!(
            client.collection_from_index("openjiuwen__nested__name"),
            Some("nested__name".to_string())
        );
        assert_eq!(client.collection_from_index("openjiuwen__"), None);
        assert_eq!(client.collection_from_index("other__docs"), None);
        assert_eq!(client.collection_from_index("openjiuwen"), None);
    }

    #[test]
    fn apply_sets_basic_auth_before_the_api_key() {
        let client = ElasticsearchClient::new(
            ElasticsearchConfig::new("http://es:9200")
                .with_username("elastic")
                .with_password("hunter2")
                .with_api_key("key-1"),
        )
        .expect("client");
        let request = client.apply(HttpRequest::get("http://es:9200/_cluster/health"));
        // "elastic:hunter2" in standard base64.
        assert_eq!(
            request.headers().get("authorization").map(String::as_str),
            Some("Basic ZWxhc3RpYzpodW50ZXIy")
        );
    }

    #[test]
    fn apply_falls_back_to_the_api_key() {
        let client = ElasticsearchClient::new(
            ElasticsearchConfig::new("http://es:9200").with_api_key("key-1"),
        )
        .expect("client");
        let request = client.apply(HttpRequest::get("http://es:9200/_cluster/health"));
        assert_eq!(
            request.headers().get("authorization").map(String::as_str),
            Some("ApiKey key-1")
        );

        let anonymous =
            ElasticsearchClient::new(ElasticsearchConfig::new("http://es:9200")).expect("client");
        let request = anonymous.apply(HttpRequest::get("http://es:9200/_cluster/health"));
        assert!(!request.headers().contains_key("authorization"));
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

        // Unknown types and untyped properties are rejected.
        let error =
            schema_from_properties(&json!({"value": {"type": "ip"}})).expect_err("unknown type");
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("\"value\""), "message: {error}");
        let error = schema_from_properties(&json!({"value": {}})).expect_err("untyped");
        assert_eq!(error.kind(), ErrorKind::Validation);
        let error = schema_from_properties(&json!([])).expect_err("not an object");
        assert_eq!(error.kind(), ErrorKind::Validation);
    }

    #[test]
    fn index_names_are_sorted_and_ignore_non_objects() {
        let payload = json!({
            "openjiuwen__b": {"mappings": {}},
            "openjiuwen__a": {"mappings": {}},
        });
        assert_eq!(
            index_names_from(&payload),
            vec!["openjiuwen__a".to_string(), "openjiuwen__b".to_string()]
        );
        assert_eq!(index_names_from(&json!(["a"])), Vec::<String>::new());
        assert_eq!(index_names_from(&json!(null)), Vec::<String>::new());
    }

    #[test]
    fn index_names_accept_lowercase_and_reject_reserved_shapes() {
        for name in [
            "openjiuwen__docs",
            "openjiuwen__docs-1",
            "openjiuwen__docs.v1",
            "openjiuwen__9000",
        ] {
            assert!(validate_index_name(name).is_ok(), "name: {name}");
        }
        for name in [
            "",
            "Openjiuwen__docs",
            "openjiuwen__Docs",
            "openjiuwen__a/b",
            "openjiuwen__a\\b",
            "openjiuwen__a*b",
            "openjiuwen__a?b",
            "openjiuwen__a\"b",
            "openjiuwen__a<b",
            "openjiuwen__a>b",
            "openjiuwen__a|b",
            "openjiuwen__a,b",
            "openjiuwen__a#b",
            "openjiuwen__a:b",
            "openjiuwen__a b",
            "_hidden",
            "-leading",
            "+leading",
        ] {
            let error = validate_index_name(name).expect_err("name must be rejected");
            assert_eq!(error.kind(), ErrorKind::Validation, "name: {name}");
            assert!(
                error
                    .to_string()
                    .contains("invalid Elasticsearch index name"),
                "message: {error}"
            );
        }
    }

    #[test]
    fn excerpt_shortens_blank_and_long_bodies() {
        assert_eq!(excerpt("   "), "<empty body>");
        assert_eq!(excerpt("  boom  "), "boom");
        let long = "x".repeat(ERROR_BODY_EXCERPT + 5);
        let shortened = excerpt(&long);
        assert!(shortened.ends_with('…'));
        assert_eq!(shortened.chars().count(), ERROR_BODY_EXCERPT + 1);
        let exact = "y".repeat(ERROR_BODY_EXCERPT);
        assert_eq!(excerpt(&exact), exact);
    }

    #[test]
    fn encode_segment_keeps_the_unreserved_set() {
        assert_eq!(
            encode_segment("openjiuwen__docs-1.v2"),
            "openjiuwen__docs-1.v2"
        );
        assert_eq!(encode_segment("a b/c"), "a%20b%2Fc");
    }
}
