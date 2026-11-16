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

use std::collections::BTreeSet;
use std::time::Duration;

use base64::Engine as _;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::http::{HttpClient, HttpClientConfig, HttpRequest, HttpResponse};
use openjiuwen_common::security::SecretString;

use crate::store::document::{validate_collection, SortOrder};
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

/// `tracing` target of the warnings of both adapters.
const LOG_TARGET: &str = "openjiuwen_foundation::store::external::elasticsearch";

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

    /// Runs one [`SearchScan`], handing every page of fresh hits to `on_page`.
    ///
    /// Every request asks for at most [`SEARCH_PAGE_SIZE`] documents sorted by
    /// the scan's `sort` and continues from the `sort` values of the previous
    /// page, so a deep read costs one request per page instead of deep
    /// `from`/`size` paging. Hits repeated across pages are dropped before
    /// `on_page` sees them, and a missing index (404) reads as no documents.
    ///
    /// # Errors
    ///
    /// `Error::Validation` when a full page carries no `sort` values (paging
    /// cannot continue) or repeats documents (the cursor does not advance), and
    /// `Error::Unsupported` — after a `tracing::warn!` naming the cap and the
    /// filter — when the scan would need more than [`MAX_SCAN_DOCUMENTS`]
    /// documents, rather than reporting the matches of a truncated window.
    pub(crate) async fn scan<F>(&self, scan: SearchScan<'_>, mut on_page: F) -> Result<()>
    where
        F: FnMut(Vec<SearchHit>) -> Result<()>,
    {
        let SearchScan {
            index,
            query,
            sort,
            needed,
            filter,
        } = scan;
        let path = self.path(index, "/_search");
        // The read stops one document past the cap, which is what tells "exactly
        // the cap" apart from "more than the cap".
        let stop = needed
            .unwrap_or(usize::MAX)
            .min(MAX_SCAN_DOCUMENTS.saturating_add(1));
        let mut gathered = 0usize;
        let mut seen: BTreeSet<String> = BTreeSet::new();
        let mut cursor: Option<Value> = None;
        while gathered < stop {
            let size = SEARCH_PAGE_SIZE.min(stop - gathered);
            let mut body = json!({
                "query": &query,
                "_source": true,
                "size": size,
                "sort": &sort,
            });
            if let (Some(cursor), Some(object)) = (&cursor, body.as_object_mut()) {
                object.insert("search_after".to_string(), cursor.clone());
            }
            let response = self
                .send_raw(HttpRequest::post(self.url(&path)).json(&body)?)
                .await?;
            match response.status {
                404 => break,
                200..=299 => {}
                _ => return Err(failure(&response, "paging documents")),
            }
            let (page, next) = search_hits(&response.json()?)?;
            let page_len = page.len();
            let fresh: Vec<SearchHit> = page
                .into_iter()
                .filter(|hit| seen.insert(hit.id.clone()))
                .collect();
            let fresh_len = fresh.len();
            if !fresh.is_empty() {
                on_page(fresh)?;
            }
            gathered += fresh_len;
            if page_len < size {
                // A short page is the last one.
                break;
            }
            if gathered >= stop {
                break;
            }
            if fresh_len == 0 {
                return Err(Error::validation(format!(
                    "Elasticsearch paging of index `{index}` did not advance: the page after \
                     `search_after` {cursor:?} repeated documents (filter: {filter})"
                )));
            }
            let Some(next) = next else {
                return Err(Error::validation(format!(
                    "Elasticsearch paging of index `{index}` cannot continue: a full page carries \
                     no `sort` values for `search_after` (filter: {filter})"
                )));
            };
            cursor = Some(next);
        }
        if gathered > MAX_SCAN_DOCUMENTS {
            tracing::warn!(
                target: LOG_TARGET,
                index = %index,
                cap = MAX_SCAN_DOCUMENTS,
                filter = %filter,
                "an Elasticsearch read reached the supported document cap",
            );
            return Err(Error::unsupported(format!(
                "Elasticsearch read of index `{index}` needs more than the supported cap of \
                 {MAX_SCAN_DOCUMENTS} documents for filter {filter}; narrow the filter or pass an \
                 explicit `limit`"
            )));
        }
        Ok(())
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

/// One hit of a `_search` response, reduced to what both adapters need.
pub(crate) struct SearchHit {
    /// The document key, i.e. Elasticsearch's `_id`.
    pub(crate) id: String,
    /// The stored document, i.e. its `_source` object.
    pub(crate) source: Map<String, Value>,
}

/// One `search_after` scan over one index, run by
/// [`ElasticsearchClient::scan`].
///
/// Both adapters page through the shared scan, so the page size, the hard cap
/// and the non-progress guards cannot drift between them.
pub(crate) struct SearchScan<'a> {
    /// Index the scan reads.
    pub(crate) index: &'a str,
    /// Query DSL of every request: a translated filter or `{"match_all":{}}`.
    pub(crate) query: Value,
    /// Total order of the scan; its last criterion must be unique per document.
    pub(crate) sort: Value,
    /// Documents the caller needs; `None` reads until the index is exhausted.
    pub(crate) needed: Option<usize>,
    /// The store-level filter, interpolated into the error messages.
    pub(crate) filter: &'a Value,
}

/// The sort clause of a paged `_search` request: the caller's criteria plus
/// the keyword key field as the stable tiebreaker.
///
/// `search_after` needs a total order that does not shift between requests, so
/// `key_field` closes the clause whenever the caller's criteria do not already
/// name it. Every criterion keeps its direction and places documents missing the
/// field first for [`SortOrder::Asc`] and last for [`SortOrder::Desc`], which is
/// the order [`run_query`](crate::store::document::run_query) produces.
pub(crate) fn stable_sort(order_by: &[(String, SortOrder)], key_field: &str) -> Value {
    let mut criteria: Vec<Value> = order_by
        .iter()
        .map(|(field, order)| {
            let (order, missing) = match order {
                SortOrder::Asc => ("asc", "_first"),
                SortOrder::Desc => ("desc", "_last"),
            };
            let mut criterion = Map::new();
            criterion.insert(
                field.clone(),
                json!({"order": order, "missing": missing, "unmapped_type": "keyword"}),
            );
            Value::Object(criterion)
        })
        .collect();
    if !order_by.iter().any(|(field, _)| field == key_field) {
        criteria.push(
            json!({key_field: {"order": "asc", "missing": "_last", "unmapped_type": "keyword"}}),
        );
    }
    Value::Array(criteria)
}

/// The hits of a `_search` response and the `search_after` cursor of its last
/// hit, in the order the server returned them.
///
/// The cursor is the hit's `sort` value array, which is exactly what the next
/// request has to send back; it is `None` when the server returned no `sort`
/// values (a response that cannot be continued). A hit without an `_id` or
/// without an object `_source` is a malformed response and fails with
/// [`Error::Validation`].
pub(crate) fn search_hits(payload: &Value) -> Result<(Vec<SearchHit>, Option<Value>)> {
    let Some(hits) = payload
        .get("hits")
        .and_then(|hits| hits.get("hits"))
        .and_then(Value::as_array)
    else {
        return Ok((Vec::new(), None));
    };
    let mut page: Vec<SearchHit> = Vec::with_capacity(hits.len());
    for hit in hits {
        let id = hit.get("_id").and_then(Value::as_str).ok_or_else(|| {
            Error::validation(format!(
                "Elasticsearch search hit is missing its `_id`: {}",
                excerpt(&hit.to_string())
            ))
        })?;
        let source = hit
            .get("_source")
            .and_then(Value::as_object)
            .cloned()
            .ok_or_else(|| {
                Error::validation(format!(
                    "Elasticsearch search hit {id:?} has no object `_source`: {}",
                    excerpt(&hit.to_string())
                ))
            })?;
        page.push(SearchHit {
            id: id.to_string(),
            source,
        });
    }
    let cursor = hits
        .last()
        .and_then(|hit| hit.get("sort"))
        .filter(|sort| sort.is_array())
        .cloned();
    Ok((page, cursor))
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
