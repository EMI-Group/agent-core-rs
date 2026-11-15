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
//! [`CollectionSchema`](crate::store::vector::CollectionSchema) (the
//! primary-key flag, `auto_id`, `max_length`, `enable_dynamic_field`), so
//! `create_index_body` stores the serialized schema as a JSON string under the
//! reserved index-metadata key `_meta.openjiuwen_schema` in addition to the
//! derived `properties` mapping. `schema_from_index` prefers that marker, which
//! restores the exact schema; an index without it (created by someone else) is
//! reconstructed from `mappings.properties` by `schema_from_properties`, which
//! synthesizes the `id` primary key the mapping cannot declare.
//!
//! Mapping code — the schema ↔ `properties` conversion and the exhaustive
//! Elasticsearch dtype → [`FieldSchema`](crate::store::vector::FieldSchema)
//! table — lives in the private `mapping` module so this file stays a client
//! plus the shared helpers.

use std::time::Duration;

use base64::Engine as _;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::http::{HttpClient, HttpClientConfig, HttpRequest, HttpResponse};
use openjiuwen_common::security::SecretString;

use crate::store::document::validate_collection;
use crate::store::vector::Metric;

pub mod document;
mod mapping;
pub mod query;
pub mod vector;

pub use document::ElasticsearchDocumentStore;
pub use vector::ElasticsearchVectorStore;

pub(crate) use mapping::{create_index_body, schema_from_index};

/// Default total timeout of one Elasticsearch request.
const fn default_timeout() -> Duration {
    Duration::from_secs(30)
}

/// Default prefix of the indexes this adapter creates.
fn default_index_prefix() -> String {
    "openjiuwen".to_string()
}

/// Number of documents requested per page of a paginated read or a filter scan.
pub(crate) const SEARCH_PAGE_SIZE: usize = 500;

/// Maximum number of response-body characters echoed in an error message.
pub(crate) const ERROR_BODY_EXCERPT: usize = 200;

/// Hard cap on the documents a single document-store read may assemble.
///
/// A read that would need more fails with a typed error instead of returning a
/// silently truncated result; the cap is Elasticsearch's default
/// `index.max_result_window`.
pub(crate) const MAX_SCAN_DOCUMENTS: usize = 10_000;

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
    use serde_json::json;

    use super::*;

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
