//! The live `DashScope` text-rerank HTTP transport behind [`AliyunReranker`].
//!
//! Ports the transport of `openjiuwen/core/retrieval/reranker/dashscope_reranker.py`
//! (a `StandardReranker` subclass) onto `reqwest`: one `POST` to
//! `{api_base}/services/rerank/text-rerank/text-rerank` carrying
//! `{"model", "input": {"query", "documents"}, "parameters": {"return_documents":
//! false, "top_n"}}` and an `Authorization: Bearer …` header, then the response's
//! `output.results` mapped back onto the caller's documents.
//!
//! Differences from the Python reranker, all deliberate:
//!
//! - Python reranks `str | Document` lists and returns a `dict[str, float]`;
//!   [`Reranker`] takes `Vec<SearchResult>` (text plus id and metadata) and
//!   returns the reordered results, so scores never have to be rejoined by id.
//! - The `instruct` / `extra_body` knobs and the raw `parameters` pass-through
//!   are not ported; the request body is the fixed schema above.
//! - There is no retry/backoff loop (`max_retries`, `retry_wait`) and no
//!   multimodal ([`MultimodalDocument`]-style) input; documents are sent as
//!   plain strings.
//! - The credential falls back to the `DASHSCOPE_API_KEY` environment variable,
//!   which Python's reranker does not read (it takes the key from
//!   `RerankerConfig.api_key` only).
//! - The HTTP client budget is the fixed [`DEFAULT_TIMEOUT`] (30 s) because
//!   `AliyunRerankerConfig` has no timeout field; Python's `RerankerConfig`
//!   defaults to 10 s.
//!
//! [`MultimodalDocument`]: https://docs.rs/openjiuwen-retrieval

use std::time::Duration;

use async_trait::async_trait;
use openjiuwen_common::config::env;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::security::SecretString;
use openjiuwen_core::retrieval::{Reranker, SearchResult};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use url::Url;

use super::AliyunRerankerConfig;

/// `DashScope` text-rerank service path, appended to the normalized `api_base`.
///
/// Python exposes the same value as `DashscopeReranker.end_point`.
const END_POINT: &str = "/services/rerank/text-rerank/text-rerank";
/// Environment variable consulted when the configuration carries no credential.
const API_KEY_ENV: &str = "DASHSCOPE_API_KEY";
/// Request budget of the shared HTTP client (`RerankerConfig.timeout` is 10 s in
/// Python; this configuration has no timeout field, so the transport fixes it).
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);
/// Maximum number of body characters quoted back in an error message.
const BODY_SNIPPET_LIMIT: usize = 300;

/// The `text-rerank` request body, named after the Python `_request_params` dict.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct RerankRequest {
    /// Reranker model name, taken from [`AliyunRerankerConfig::model_name`].
    pub model: String,
    /// Query and the document texts to score.
    pub input: RerankInput,
    /// Service parameters; `return_documents` is always `false`.
    pub parameters: RerankParameters,
}

/// The `input` object of a [`RerankRequest`].
#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct RerankInput {
    /// The search query.
    pub query: String,
    /// The document texts, one per input [`SearchResult`].
    pub documents: Vec<String>,
}

/// The `parameters` object of a [`RerankRequest`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub(crate) struct RerankParameters {
    /// Always `false`: the response carries scores only, never the documents.
    pub return_documents: bool,
    /// Number of ranked results the service should return (`>= 1`).
    pub top_n: usize,
}

/// One scored document of a [`RerankResponse`].
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub(crate) struct RerankResult {
    /// Position of the document in the request's `input.documents` array.
    pub index: usize,
    /// Relevance of that document to the query.
    pub relevance_score: f64,
}

/// The `output` object of a [`RerankResponse`].
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct RerankOutput {
    /// The ranked documents; absent when the service returned nothing.
    #[serde(default)]
    pub results: Option<Vec<RerankResult>>,
}

/// A `text-rerank` response: results plus the error fields `DashScope` reports.
///
/// Unknown fields are ignored, and `code` tolerates both the string codes the
/// service uses and any numeric or structured variant.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct RerankResponse {
    /// The success payload; `None` on an error body.
    #[serde(default)]
    pub output: Option<RerankOutput>,
    /// Fallback result list for gateways that omit the `output` wrapper
    /// (Python: `response_data.get("output", response_data)["results"]`).
    #[serde(default)]
    pub results: Option<Vec<RerankResult>>,
    /// Service error code, for example `Throttling` or `InvalidApiKey`.
    #[serde(default)]
    pub code: Option<Value>,
    /// Service error message.
    #[serde(default)]
    pub message: Option<String>,
    /// Service request id, useful when reporting a failure.
    #[serde(default)]
    pub request_id: Option<String>,
}

impl RerankResponse {
    /// The ranked documents, preferring `output.results` over a top-level
    /// `results` exactly like Python's `get("output", response_data)["results"]`.
    fn results(&self) -> Option<&[RerankResult]> {
        match &self.output {
            Some(output) => output.results.as_deref(),
            None => self.results.as_deref(),
        }
    }

    /// `code`, `message`, and `request_id` joined into one line, if any were sent.
    fn detail(&self) -> Option<String> {
        let mut parts = Vec::new();
        if let Some(code) = &self.code {
            parts.push(format!("code={}", code_text(code)));
        }
        if let Some(message) = &self.message {
            parts.push(format!("message={message}"));
        }
        if let Some(request_id) = &self.request_id {
            parts.push(format!("request_id={request_id}"));
        }
        (!parts.is_empty()).then(|| parts.join(", "))
    }
}

/// The live `DashScope` transport, shared by every call of one [`AliyunReranker`].
#[derive(Debug, Clone)]
pub(crate) struct DashScopeReranker {
    /// The pooled HTTP client, configured with [`DEFAULT_TIMEOUT`].
    client: reqwest::Client,
    /// The configuration this transport was built from.
    config: AliyunRerankerConfig,
    /// The resolved rerank endpoint (normalized `api_base` + [`END_POINT`]).
    endpoint: Url,
    /// The resolved credential; redacted in `Debug` and never logged.
    api_key: SecretString,
}

impl DashScopeReranker {
    /// Builds the transport and validates its configuration.
    ///
    /// No I/O happens here: the HTTP client is created eagerly and connections
    /// are opened by the first [`Reranker::rerank`] call.
    ///
    /// # Errors
    ///
    /// [`openjiuwen_common::error::ErrorKind::Validation`] when neither
    /// [`AliyunRerankerConfig::api_key`] nor the `DASHSCOPE_API_KEY` environment
    /// variable carries a credential, or when `api_base` is empty, unparseable,
    /// or not `http`/`https`; [`openjiuwen_common::error::ErrorKind::Transport`]
    /// when the HTTP client cannot be built.
    pub(crate) fn new(config: AliyunRerankerConfig) -> Result<Self> {
        let api_key = resolve_api_key(config.api_key.as_ref())?;
        let endpoint = endpoint_url(&config.api_base)?;
        let client = reqwest::Client::builder()
            .timeout(DEFAULT_TIMEOUT)
            .build()
            .map_err(|error| {
                Error::transport(format!(
                    "failed to build the DashScope rerank HTTP client: {error}"
                ))
            })?;
        Ok(Self {
            client,
            config,
            endpoint,
            api_key,
        })
    }

    /// The resolved endpoint, for debug logging (never the credential).
    #[must_use]
    pub(crate) fn endpoint(&self) -> &Url {
        &self.endpoint
    }

    /// Builds the wire request for `documents` and an already resolved limit.
    ///
    /// The wire `top_n` is `effective_top_n` clamped to the document count, with
    /// `0` meaning "no truncation" and therefore rendered as the document count
    /// (`DashScope` rejects `top_n = 0`).
    fn build_request(
        &self,
        query: &str,
        documents: &[SearchResult],
        effective_top_n: usize,
    ) -> RerankRequest {
        RerankRequest {
            model: self.config.model_name.clone(),
            input: RerankInput {
                query: query.to_owned(),
                documents: documents
                    .iter()
                    .map(|document| document.text.clone())
                    .collect(),
            },
            parameters: RerankParameters {
                return_documents: false,
                top_n: wire_top_n(effective_top_n, documents.len()),
            },
        }
    }

    /// Maps a `reqwest` send failure onto the crate error type.
    ///
    /// A timeout names the configured budget; every other failure names the
    /// endpoint and the underlying reason. The credential is never included.
    fn send_error(&self, error: &reqwest::Error) -> Error {
        if error.is_timeout() {
            Error::transport(format!(
                "DashScope rerank request timed out after {}s: {error}",
                DEFAULT_TIMEOUT.as_secs()
            ))
        } else {
            Error::transport(format!(
                "DashScope rerank request to {} failed: {error}",
                self.endpoint
            ))
        }
    }
}

#[async_trait]
impl Reranker for DashScopeReranker {
    /// Reranks `documents` through the `DashScope` `text-rerank` service.
    ///
    /// # Semantics
    ///
    /// - Empty `documents` returns `Ok(vec![])` without any HTTP request.
    /// - The effective limit is the call's `top_n`, else the configured
    ///   [`AliyunRerankerConfig::top_n`], else the document count.
    /// - `Some(0)` and any limit `>= documents.len()` mean "no truncation".
    ///   This differs from the kernel's `sort_and_truncate`, where `Some(0)`
    ///   truncates to an empty result.
    /// - The wire `parameters.top_n` is the effective limit clamped to the
    ///   document count, and the document count when the limit is `0`.
    /// - Each returned `index` addresses an input document: its `id`, `text`,
    ///   and `metadata` are preserved and only `score` is overwritten with the
    ///   `relevance_score`. The results are then sorted by descending score with
    ///   a stable `f32::total_cmp`, so equal scores keep the input order, and
    ///   truncated to a limit strictly between `1` and `documents.len()`.
    /// - A duplicated `index` keeps the last score, mirroring Python's dict
    ///   assignment, and a document missing from the response keeps `score`
    ///   `0.0`, mirroring Python's `dict.fromkeys(doc_ids, 0.0)`.
    ///
    /// # Errors
    ///
    /// [`openjiuwen_common::error::ErrorKind::Validation`] for a `401`/`403`
    /// (the message names the credential source, never its value);
    /// [`openjiuwen_common::error::ErrorKind::Transport`] for every other
    /// non-2xx status, a malformed JSON body, a body without results, an
    /// out-of-range `index`, a non-finite `relevance_score`, and transport or
    /// timeout failures.
    async fn rerank(
        &self,
        query: &str,
        documents: Vec<SearchResult>,
        top_n: Option<usize>,
    ) -> Result<Vec<SearchResult>> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }
        let effective_top_n = top_n.or(self.config.top_n).unwrap_or(documents.len());
        let request = self.build_request(query, &documents, effective_top_n);
        tracing::debug!(
            target: "openjiuwen_extensions::vendor_specific",
            endpoint = %self.endpoint,
            model = %request.model,
            documents = documents.len(),
            top_n = request.parameters.top_n,
            "sending a DashScope rerank request"
        );
        let response = self
            .client
            .post(self.endpoint.clone())
            .header(reqwest::header::CONTENT_TYPE, "application/json")
            .header(
                reqwest::header::AUTHORIZATION,
                format!("Bearer {}", self.api_key.expose_secret()),
            )
            .json(&request)
            .send()
            .await
            .map_err(|error| self.send_error(&error))?;
        let status = response.status();
        let body = response.text().await.map_err(|error| {
            Error::transport(format!(
                "DashScope rerank response body could not be read: {error}"
            ))
        })?;
        if !status.is_success() {
            return Err(status_error(status, &body));
        }
        let parsed: RerankResponse = serde_json::from_str(&body).map_err(|error| {
            Error::transport(format!(
                "DashScope rerank response was not valid JSON ({error}): {}",
                error_detail(&body)
            ))
        })?;
        let results = parsed.results().ok_or_else(|| {
            Error::transport(format!(
                "DashScope rerank response carried no results: {}",
                error_detail(&body)
            ))
        })?;
        map_results(documents, results, effective_top_n)
    }
}

/// Resolves the credential from the configuration, then from the environment.
///
/// A missing, empty, or whitespace-only configured key counts as unset. Errors
/// name both sources and never echo a value.
fn resolve_api_key(configured: Option<&SecretString>) -> Result<SecretString> {
    if let Some(secret) = configured {
        if !secret.expose_secret().trim().is_empty() {
            return Ok(secret.clone());
        }
    }
    if let Some(from_env) = env::var(API_KEY_ENV) {
        return Ok(SecretString::new(from_env));
    }
    Err(Error::validation(format!(
        "DashScope reranker requires a credential: set AliyunRerankerConfig::api_key or the \
         {API_KEY_ENV} environment variable"
    )))
}

/// Normalizes `api_base` and appends [`END_POINT`].
///
/// Mirrors Python's `api_base.removesuffix("/").removesuffix(end_point)`, so a
/// bare host and a host with the full service path (with or without a trailing
/// slash) resolve to the same endpoint.
fn endpoint_url(api_base: &str) -> Result<Url> {
    let trimmed = api_base.trim();
    if trimmed.is_empty() {
        return Err(Error::validation(
            "DashScope reranker requires a non-empty api_base",
        ));
    }
    let without_slash = trimmed.strip_suffix('/').unwrap_or(trimmed);
    let base = without_slash
        .strip_suffix(END_POINT)
        .unwrap_or(without_slash);
    let endpoint = format!("{base}{END_POINT}");
    let url = Url::parse(&endpoint).map_err(|error| {
        Error::validation(format!(
            "DashScope reranker api_base `{trimmed}` is not a valid URL: {error}"
        ))
    })?;
    if !matches!(url.scheme(), "http" | "https") {
        return Err(Error::validation(format!(
            "DashScope reranker api_base `{trimmed}` must use the http or https scheme, found `{}`",
            url.scheme()
        )));
    }
    Ok(url)
}

/// The `parameters.top_n` sent to `DashScope`.
///
/// `0` means "no truncation" and is rendered as the document count because the
/// service rejects `top_n = 0`; any larger limit is clamped to the count.
fn wire_top_n(effective_top_n: usize, document_count: usize) -> usize {
    if effective_top_n == 0 {
        document_count
    } else {
        effective_top_n.min(document_count)
    }
}

/// Maps `results` back onto `documents` by index, sorts, and truncates.
fn map_results(
    documents: Vec<SearchResult>,
    results: &[RerankResult],
    effective_top_n: usize,
) -> Result<Vec<SearchResult>> {
    let count = documents.len();
    let mut relevance = vec![0.0_f32; count];
    for result in results {
        if result.index >= count {
            return Err(Error::transport(format!(
                "DashScope rerank response referenced document index {} but only {count} \
                 documents were sent",
                result.index
            )));
        }
        if !result.relevance_score.is_finite() {
            return Err(Error::transport(format!(
                "DashScope rerank response carried a non-finite relevance_score for document \
                 index {}",
                result.index
            )));
        }
        #[allow(clippy::cast_possible_truncation)]
        // A relevance score is a `[0, 1]` value and the kernel stores `f32`.
        let score = result.relevance_score as f32;
        // Python assigns into a dict, so a duplicated index keeps the last score.
        relevance[result.index] = score;
    }
    let mut scored: Vec<SearchResult> = documents
        .into_iter()
        .zip(relevance)
        .map(|(mut document, score)| {
            document.score = score;
            document
        })
        .collect();
    // Stable, so equal scores keep the input order, like the kernel's helper.
    scored.sort_by(|left, right| right.score.total_cmp(&left.score));
    if effective_top_n > 0 && effective_top_n < scored.len() {
        scored.truncate(effective_top_n);
    }
    Ok(scored)
}

/// Maps a non-2xx status onto the crate error type.
///
/// `401`/`403` become a validation error naming the credential source and the
/// status only: their bodies can echo the key, so none of the body is quoted.
/// Every other status keeps the `DashScope` `code`/`message`/`request_id`.
fn status_error(status: reqwest::StatusCode, body: &str) -> Error {
    let code = status.as_u16();
    if status == reqwest::StatusCode::UNAUTHORIZED || status == reqwest::StatusCode::FORBIDDEN {
        return Error::validation(format!(
            "DashScope rerank request was rejected with HTTP {code}: check \
             AliyunRerankerConfig::api_key or the {API_KEY_ENV} environment variable"
        ));
    }
    Error::transport(format!(
        "DashScope rerank request failed with HTTP {code}: {}",
        error_detail(body)
    ))
}

/// Describes a response body for an error message.
///
/// The `code`/`message`/`request_id` fields win when the body is a `DashScope`
/// error object; otherwise the trimmed raw body is quoted, bounded to
/// [`BODY_SNIPPET_LIMIT`] characters.
fn error_detail(body: &str) -> String {
    if let Ok(response) = serde_json::from_str::<RerankResponse>(body) {
        if let Some(detail) = response.detail() {
            return detail;
        }
    }
    let trimmed = body.trim();
    if trimmed.is_empty() {
        return "the response body was empty".to_owned();
    }
    let mut chars = trimmed.chars();
    let snippet: String = chars.by_ref().take(BODY_SNIPPET_LIMIT).collect();
    if chars.next().is_some() {
        format!("{snippet}…")
    } else {
        snippet
    }
}

/// Renders a JSON `code` value as text: strings verbatim, anything else as JSON.
fn code_text(code: &Value) -> String {
    match code {
        Value::String(text) => text.clone(),
        other => other.to_string(),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]
mod tests {
    use std::sync::{Mutex, MutexGuard, PoisonError};

    use openjiuwen_common::error::ErrorKind;
    use serde_json::json;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::*;

    // Python counterpart: `tests/unit_tests/core/retrieval/` (`DashscopeReranker`).
    /// Credential used by every authenticated test.
    const API_KEY: &str = "sk-dashscope-test-key";

    /// Serialises the tests that mutate [`API_KEY_ENV`].
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    /// Holds [`ENV_LOCK`] and leaves [`API_KEY_ENV`] unset, before and after.
    struct EnvGuard {
        /// The lock guard; dropped last.
        _lock: MutexGuard<'static, ()>,
    }

    impl EnvGuard {
        /// Acquires the lock, recovering it when another test panicked.
        fn new() -> Self {
            let lock = ENV_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
            std::env::remove_var(API_KEY_ENV);
            Self { _lock: lock }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            std::env::remove_var(API_KEY_ENV);
        }
    }

    /// One input document whose text embeds its id.
    fn hit(id: &str) -> SearchResult {
        SearchResult {
            id: id.to_owned(),
            text: format!("text {id}"),
            score: 0.0,
            metadata: serde_json::Map::new(),
        }
    }

    /// A configuration pointed at `server`, with a non-default model.
    fn config_for(server: &MockServer, top_n: Option<usize>) -> AliyunRerankerConfig {
        AliyunRerankerConfig {
            api_key: Some(SecretString::new(API_KEY)),
            api_base: server.uri(),
            model_name: "qwen3-reranker-8b".to_owned(),
            top_n,
        }
    }

    /// A transport aimed at `server` with no configured limit.
    fn transport(server: &MockServer) -> DashScopeReranker {
        DashScopeReranker::new(config_for(server, None)).expect("the transport builds")
    }

    /// Mounts the `Mock` answering every rerank call with `output.results`.
    async fn mock_results(server: &MockServer, results: Value) {
        Mock::given(method("POST"))
            .and(path(END_POINT))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(json!({ "output": { "results": results } })),
            )
            .mount(server)
            .await;
    }

    /// Mounts a `Mock` answering with `status` and a raw string body.
    async fn mock_status(server: &MockServer, status: u16, body: &str) {
        Mock::given(method("POST"))
            .and(path(END_POINT))
            .respond_with(ResponseTemplate::new(status).set_body_string(body))
            .mount(server)
            .await;
    }

    /// The single request `server` received, decoded as JSON.
    async fn received_body(server: &MockServer) -> Value {
        let requests = server
            .received_requests()
            .await
            .expect("the mock records requests");
        assert_eq!(requests.len(), 1, "exactly one request was sent");
        serde_json::from_slice(&requests[0].body).expect("the request body is JSON")
    }

    /// The `parameters.top_n` the single request carried.
    async fn sent_top_n(server: &MockServer) -> usize {
        let body = received_body(server).await;
        usize::try_from(body["parameters"]["top_n"].as_u64().unwrap()).unwrap()
    }

    /// The ids of `results`, in order.
    fn ids(results: &[SearchResult]) -> Vec<&str> {
        results.iter().map(|result| result.id.as_str()).collect()
    }

    #[tokio::test]
    async fn request_carries_the_dashscope_body_and_bearer_token() {
        // ports `DashscopeReranker._request_params` plus `StandardReranker._headers`.
        let server = MockServer::start().await;
        mock_results(
            &server,
            json!([{ "index": 0, "relevance_score": 0.9 }, { "index": 1, "relevance_score": 0.5 }]),
        )
        .await;

        transport(&server)
            .rerank("what is retrieval?", vec![hit("a"), hit("b")], None)
            .await
            .expect("the rerank succeeds");

        let requests = server.received_requests().await.unwrap();
        assert_eq!(requests.len(), 1);
        let request = &requests[0];
        assert_eq!(request.method.as_str(), "POST");
        assert_eq!(request.url.path(), END_POINT);
        assert_eq!(
            request
                .headers
                .get("authorization")
                .unwrap()
                .to_str()
                .unwrap(),
            format!("Bearer {API_KEY}")
        );
        assert_eq!(
            request
                .headers
                .get("content-type")
                .unwrap()
                .to_str()
                .unwrap(),
            "application/json"
        );
        assert_eq!(
            serde_json::from_slice::<Value>(&request.body).unwrap(),
            json!({
                "model": "qwen3-reranker-8b",
                "input": {
                    "query": "what is retrieval?",
                    "documents": ["text a", "text b"],
                },
                "parameters": { "return_documents": false, "top_n": 2 },
            })
        );
    }

    #[tokio::test]
    async fn results_map_back_to_the_input_documents_and_sort_descending() {
        // ports `StandardReranker._parse_response`, then the kernel's stable order.
        let server = MockServer::start().await;
        mock_results(
            &server,
            json!([
                { "index": 2, "relevance_score": 0.5 },
                { "index": 0, "relevance_score": 0.2 },
                { "index": 1, "relevance_score": 0.5 },
                { "index": 3, "relevance_score": 0.9 },
            ]),
        )
        .await;
        let mut documents = vec![hit("a"), hit("b"), hit("c"), hit("d")];
        documents[3].metadata.insert("k".to_owned(), json!("v"));

        let results = transport(&server)
            .rerank("query", documents, None)
            .await
            .expect("the rerank succeeds");

        // `d` scores highest; `b` and `c` tie at 0.5, so their input order holds.
        assert_eq!(ids(&results), ["d", "b", "c", "a"]);
        assert_eq!(results[0].metadata.get("k"), Some(&json!("v")));
        for result in &results {
            assert_eq!(result.text, format!("text {}", result.id));
        }
        assert!((results[1].score - 0.5).abs() < f32::EPSILON);
        assert!((results[3].score - 0.2).abs() < f32::EPSILON);
    }

    #[tokio::test]
    async fn a_duplicated_index_keeps_the_last_score() {
        // Python assigns into a dict, so the later score overwrites the earlier one.
        let server = MockServer::start().await;
        mock_results(
            &server,
            json!([
                { "index": 0, "relevance_score": 0.1 },
                { "index": 0, "relevance_score": 0.9 },
                { "index": 1, "relevance_score": 0.4 },
            ]),
        )
        .await;

        let results = transport(&server)
            .rerank("query", vec![hit("a"), hit("b")], None)
            .await
            .expect("the rerank succeeds");

        assert_eq!(ids(&results), ["a", "b"]);
        assert!((results[0].score - 0.9).abs() < f32::EPSILON);
    }

    #[tokio::test]
    async fn top_n_precedence_sends_and_truncates_the_effective_limit() {
        // The call argument wins over the configuration, which wins over the count.
        let documents = || vec![hit("a"), hit("b"), hit("c"), hit("d")];
        let all_scores = json!([
            { "index": 0, "relevance_score": 0.1 },
            { "index": 1, "relevance_score": 0.9 },
            { "index": 2, "relevance_score": 0.7 },
            { "index": 3, "relevance_score": 0.3 },
        ]);

        let server = MockServer::start().await;
        mock_results(&server, all_scores.clone()).await;
        let results = DashScopeReranker::new(config_for(&server, Some(3)))
            .unwrap()
            .rerank("query", documents(), Some(2))
            .await
            .unwrap();
        assert_eq!(sent_top_n(&server).await, 2);
        assert_eq!(ids(&results), ["b", "c"]);

        let server = MockServer::start().await;
        mock_results(&server, all_scores.clone()).await;
        let results = DashScopeReranker::new(config_for(&server, Some(3)))
            .unwrap()
            .rerank("query", documents(), None)
            .await
            .unwrap();
        assert_eq!(sent_top_n(&server).await, 3);
        assert_eq!(ids(&results), ["b", "c", "d"]);

        let server = MockServer::start().await;
        mock_results(&server, all_scores).await;
        let results = transport(&server)
            .rerank("query", documents(), None)
            .await
            .unwrap();
        assert_eq!(sent_top_n(&server).await, 4);
        assert_eq!(ids(&results), ["b", "c", "d", "a"]);
    }

    #[tokio::test]
    async fn a_zero_or_oversized_limit_means_no_truncation() {
        // `DashScope` rejects `top_n = 0`, and `Some(0)` keeps everything here.
        for limit in [0, 9] {
            let server = MockServer::start().await;
            mock_results(
                &server,
                json!([
                    { "index": 0, "relevance_score": 0.2 },
                    { "index": 1, "relevance_score": 0.8 },
                ]),
            )
            .await;

            let results = transport(&server)
                .rerank("query", vec![hit("a"), hit("b")], Some(limit))
                .await
                .expect("the rerank succeeds");

            assert_eq!(sent_top_n(&server).await, 2, "limit {limit}");
            assert_eq!(ids(&results), ["b", "a"], "limit {limit}");
        }
    }

    #[tokio::test]
    async fn documents_absent_from_the_response_score_zero() {
        // Python's `dict.fromkeys(doc_ids, 0.0)` default.
        let server = MockServer::start().await;
        mock_results(&server, json!([{ "index": 1, "relevance_score": 0.8 }])).await;

        let results = transport(&server)
            .rerank("query", vec![hit("a"), hit("b")], None)
            .await
            .expect("the rerank succeeds");

        assert_eq!(ids(&results), ["b", "a"]);
        assert!(results[1].score.abs() < f32::EPSILON);
    }

    #[tokio::test]
    async fn empty_documents_send_no_request() {
        let server = MockServer::start().await;

        let results = transport(&server)
            .rerank("query", Vec::new(), None)
            .await
            .expect("an empty input succeeds");

        assert!(results.is_empty());
        assert!(
            server.received_requests().await.unwrap().is_empty(),
            "no HTTP request was sent"
        );
    }

    #[tokio::test]
    async fn unauthorized_and_forbidden_bodies_are_never_echoed() {
        for status in [401, 403] {
            let server = MockServer::start().await;
            mock_status(
                &server,
                status,
                &json!({
                    "code": "InvalidApiKey",
                    "message": format!("the key {API_KEY} is invalid"),
                    "request_id": "req-auth",
                })
                .to_string(),
            )
            .await;

            let error = transport(&server)
                .rerank("query", vec![hit("a")], None)
                .await
                .unwrap_err();

            assert_eq!(error.kind(), ErrorKind::Validation);
            let message = error.to_string();
            assert!(message.contains(&status.to_string()), "{message}");
            assert!(
                message.contains("AliyunRerankerConfig::api_key"),
                "{message}"
            );
            assert!(message.contains(API_KEY_ENV), "{message}");
            assert!(!message.contains(API_KEY), "leaked the key: {message}");
        }
    }

    #[tokio::test]
    async fn throttling_and_server_errors_keep_the_dashscope_detail() {
        for status in [429, 500] {
            let server = MockServer::start().await;
            mock_status(
                &server,
                status,
                &json!({
                    "code": "Throttling",
                    "message": "Requests rate limit exceeded",
                    "request_id": "req-throttle",
                })
                .to_string(),
            )
            .await;

            let error = transport(&server)
                .rerank("query", vec![hit("a")], None)
                .await
                .unwrap_err();

            assert_eq!(error.kind(), ErrorKind::Transport);
            let message = error.to_string();
            assert!(message.contains(&status.to_string()), "{message}");
            assert!(message.contains("Throttling"), "{message}");
            assert!(
                message.contains("Requests rate limit exceeded"),
                "{message}"
            );
            assert!(message.contains("req-throttle"), "{message}");
            assert!(!message.contains(API_KEY), "leaked the key: {message}");
        }
    }

    #[tokio::test]
    async fn other_client_errors_map_to_transport_with_the_detail() {
        let server = MockServer::start().await;
        mock_status(
            &server,
            400,
            &json!({ "code": "InvalidParameter", "message": "top_n must be positive" }).to_string(),
        )
        .await;

        let error = transport(&server)
            .rerank("query", vec![hit("a")], None)
            .await
            .unwrap_err();

        assert_eq!(error.kind(), ErrorKind::Transport);
        let message = error.to_string();
        assert!(message.contains("400"), "{message}");
        assert!(message.contains("InvalidParameter"), "{message}");
        assert!(message.contains("top_n must be positive"), "{message}");
    }

    #[tokio::test]
    async fn malformed_or_resultless_bodies_map_to_transport() {
        for body in ["not json at all", "{}", r#"{"output": {}}"#] {
            let server = MockServer::start().await;
            mock_status(&server, 200, body).await;

            let error = transport(&server)
                .rerank("query", vec![hit("a")], None)
                .await
                .unwrap_err();

            assert_eq!(error.kind(), ErrorKind::Transport, "body `{body}`");
            let message = error.to_string();
            assert!(
                message.contains("not valid JSON") || message.contains("no results"),
                "body `{body}`: {message}"
            );
        }
    }

    #[tokio::test]
    async fn an_out_of_range_index_maps_to_transport() {
        let server = MockServer::start().await;
        mock_results(&server, json!([{ "index": 5, "relevance_score": 0.5 }])).await;

        let error = transport(&server)
            .rerank("query", vec![hit("a"), hit("b")], None)
            .await
            .unwrap_err();

        assert_eq!(error.kind(), ErrorKind::Transport);
        let message = error.to_string();
        assert!(message.contains("index 5"), "{message}");
        assert!(message.contains("only 2 documents"), "{message}");
    }

    #[test]
    fn a_non_finite_score_maps_to_transport() {
        let error = map_results(
            vec![hit("a")],
            &[RerankResult {
                index: 0,
                relevance_score: f64::NAN,
            }],
            1,
        )
        .unwrap_err();

        assert_eq!(error.kind(), ErrorKind::Transport);
        assert!(error.message().contains("non-finite"), "{}", error);
        assert!(error.message().contains("index 0"), "{}", error);
    }

    #[test]
    fn endpoint_url_normalizes_like_python() {
        let expected = "https://dashscope.aliyuncs.com/services/rerank/text-rerank/text-rerank";
        for api_base in [
            "https://dashscope.aliyuncs.com",
            "https://dashscope.aliyuncs.com/",
            "  https://dashscope.aliyuncs.com  ",
            "https://dashscope.aliyuncs.com/services/rerank/text-rerank/text-rerank",
            "https://dashscope.aliyuncs.com/services/rerank/text-rerank/text-rerank/",
        ] {
            assert_eq!(
                endpoint_url(api_base).unwrap().as_str(),
                expected,
                "api_base `{api_base}`"
            );
        }

        // A gateway prefix survives, matching httpx's `base_url` merging.
        assert_eq!(
            endpoint_url("https://gw.internal/proxy").unwrap().as_str(),
            "https://gw.internal/proxy/services/rerank/text-rerank/text-rerank"
        );

        for invalid in [
            "",
            "   ",
            "/",
            "ftp://dashscope.aliyuncs.com",
            "dashscope.aliyuncs.com",
        ] {
            let error = endpoint_url(invalid).unwrap_err();
            assert_eq!(error.kind(), ErrorKind::Validation, "api_base `{invalid}`");
            assert!(error.message().contains("api_base"), "{}", error);
        }
    }

    #[test]
    fn api_key_falls_back_to_the_dashscope_environment_variable() {
        let _guard = EnvGuard::new();
        std::env::set_var(API_KEY_ENV, "sk-from-env");

        for configured in [
            None,
            Some(SecretString::new("")),
            Some(SecretString::new("   ")),
        ] {
            let transport = DashScopeReranker::new(AliyunRerankerConfig {
                api_key: configured,
                ..AliyunRerankerConfig::default()
            })
            .expect("the environment supplies the key");
            assert_eq!(transport.api_key.expose_secret(), "sk-from-env");
        }
    }

    #[test]
    fn missing_credentials_name_both_sources() {
        let _guard = EnvGuard::new();

        let error = DashScopeReranker::new(AliyunRerankerConfig::default()).unwrap_err();

        assert_eq!(error.kind(), ErrorKind::Validation);
        let message = error.message();
        assert!(
            message.contains("AliyunRerankerConfig::api_key"),
            "{message}"
        );
        assert!(message.contains(API_KEY_ENV), "{message}");
    }

    #[test]
    fn an_invalid_api_base_is_rejected_before_any_request() {
        for api_base in [String::new(), "   ".to_owned(), "ftp://host".to_owned()] {
            let error = DashScopeReranker::new(AliyunRerankerConfig {
                api_key: Some(SecretString::new(API_KEY)),
                api_base: api_base.clone(),
                ..AliyunRerankerConfig::default()
            })
            .unwrap_err();

            assert_eq!(error.kind(), ErrorKind::Validation, "api_base `{api_base}`");
            assert!(error.message().contains("api_base"), "{}", error);
        }
    }

    #[test]
    fn request_serializes_to_the_dashscope_shape() {
        let request = RerankRequest {
            model: "qwen3-reranker-0.6b".to_owned(),
            input: RerankInput {
                query: "q".to_owned(),
                documents: vec!["a".to_owned(), "b".to_owned()],
            },
            parameters: RerankParameters {
                return_documents: false,
                top_n: 2,
            },
        };

        assert_eq!(
            serde_json::to_value(&request).unwrap(),
            json!({
                "model": "qwen3-reranker-0.6b",
                "input": { "query": "q", "documents": ["a", "b"] },
                "parameters": { "return_documents": false, "top_n": 2 },
            })
        );
    }

    #[test]
    fn response_deserializes_nested_top_level_and_tolerates_unknown_fields() {
        let nested: RerankResponse = serde_json::from_str(
            r#"{"output": {"results": [{"index": 0, "relevance_score": 0.25, "extra": true}]}, "usage": 7}"#,
        )
        .unwrap();
        let results = nested.results().unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].index, 0);
        assert!((results[0].relevance_score - 0.25).abs() < 1e-9);

        let top_level: RerankResponse =
            serde_json::from_str(r#"{"results": [{"index": 1, "relevance_score": 0.5}]}"#).unwrap();
        assert_eq!(top_level.results().unwrap()[0].index, 1);

        // An `output` without `results` does not fall back, like Python's lookup.
        let empty_output: RerankResponse = serde_json::from_str(
            r#"{"output": {}, "results": [{"index": 0, "relevance_score": 1.0}]}"#,
        )
        .unwrap();
        assert!(empty_output.results().is_none());

        // Error bodies keep their code/message/request_id.
        let error: RerankResponse = serde_json::from_str(
            r#"{"code": "Throttling", "message": "slow down", "request_id": "r1"}"#,
        )
        .unwrap();
        assert_eq!(
            error.detail().as_deref(),
            Some("code=Throttling, message=slow down, request_id=r1")
        );
    }

    #[test]
    fn error_detail_is_bounded_and_falls_back_to_the_raw_body() {
        let long = "x".repeat(BODY_SNIPPET_LIMIT + 50);
        let detail = error_detail(&long);
        assert!(detail.ends_with('…'), "{detail}");
        assert_eq!(detail.chars().count(), BODY_SNIPPET_LIMIT + 1);

        assert_eq!(error_detail("  \n "), "the response body was empty");
        assert_eq!(error_detail(r#" {"foo": "bar"} "#), r#"{"foo": "bar"}"#);
    }
}
