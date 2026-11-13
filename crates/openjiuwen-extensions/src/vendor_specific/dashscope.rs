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
mod tests;
