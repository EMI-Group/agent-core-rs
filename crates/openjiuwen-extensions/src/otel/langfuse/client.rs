//! REST ingestion client for `POST {host}/api/public/ingestion`.
//!
//! Langfuse's ingestion API takes a JSON body of the shape
//! `{"batch":[{"id":…,"type":…,"timestamp":…,"body":{…}}, …]}` and answers with
//! `{"successes":[…],"errors":[…]}`. Ingestion is **partial**: a `200`/`207`
//! response can still carry per-event failures, so a successful status alone
//! does not mean the events landed. [`LangfuseClient::ingest`] therefore treats
//! an empty `errors` array as success and turns a non-empty one into an error
//! naming the offending ids, statuses, and messages.
//!
//! Unlike the OTLP exporter, this client is plain async `reqwest`: callers drive
//! it from a tokio runtime (tests use `#[tokio::test]`), and no worker thread is
//! involved.
//!
//! ```
//! use openjiuwen_extensions::otel::langfuse::{
//!     IngestionBatch, IngestionEvent, LangfuseClient, LangfuseConfig, TraceBody,
//! };
//!
//! # async fn example() -> openjiuwen_common::error::Result<()> {
//! let client = LangfuseClient::new(LangfuseConfig::new("pk-lf-public", "sk-lf-secret"))?;
//! let mut batch = IngestionBatch::new();
//! batch.push(IngestionEvent::trace(&TraceBody::new("checkout"))?);
//! assert_eq!(batch.len(), 1);
//! let _ = &client; // `ingest` posts to Langfuse Cloud; this example stays offline.
//! # Ok(())
//! # }
//! ```

use openjiuwen_common::error::{Error, Result};
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use reqwest::StatusCode;
use serde_json::{json, Value};
use tracing::debug;

use super::batch::{IngestionBatch, DEFAULT_MAX_BATCH_SIZE};
use super::events::IngestionEvent;
use super::LangfuseConfig;

/// `tracing` target for ingestion diagnostics.
const LOG_TARGET: &str = "openjiuwen_extensions::otel::langfuse";
/// Per-request timeout of the default client, in seconds.
const DEFAULT_TIMEOUT_SECS: u64 = 30;
/// Maximum number of characters of a response body kept in an error message.
const MAX_BODY_SNIPPET_CHARS: usize = 512;
/// Maximum number of per-event failures named in one error message.
const MAX_REPORTED_FAILURES: usize = 10;

/// What one [`LangfuseClient::ingest`] call sent.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct IngestionSummary {
    /// Number of events accepted by Langfuse (summed over the requests).
    pub events: usize,
    /// Number of HTTP requests issued.
    pub requests: usize,
}

/// Posts ingestion events to a Langfuse host.
///
/// Built from a [`LangfuseConfig`] (validated by [`LangfuseClient::new`]) or
/// from the environment ([`LangfuseClient::from_env`]). Batches larger than the
/// configured maximum are split, and an empty batch never reaches the network.
pub struct LangfuseClient {
    /// Credentials and host.
    config: LangfuseConfig,
    /// HTTP client; replaceable through [`Self::with_http_client`].
    http: reqwest::Client,
    /// Maximum number of events per request; never `0`.
    max_batch_size: usize,
}

impl std::fmt::Debug for LangfuseClient {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("LangfuseClient")
            .field("config", &self.config)
            .field("max_batch_size", &self.max_batch_size)
            .finish_non_exhaustive()
    }
}

impl LangfuseClient {
    /// Builds a client for `config`.
    ///
    /// The default HTTP client uses a 30 s request timeout, so a stalled
    /// ingestion call cannot hang a task forever; pass an explicit
    /// [`reqwest::Client`] with [`Self::with_http_client`] to change it.
    ///
    /// # Errors
    ///
    /// The errors of [`LangfuseConfig::validate`], plus [`Error::config`] when
    /// the HTTP client cannot be built.
    ///
    /// [`Error::config`]: openjiuwen_common::error::Error::config
    pub fn new(config: LangfuseConfig) -> Result<Self> {
        config.validate()?;
        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(DEFAULT_TIMEOUT_SECS))
            .build()
            .map_err(|error| {
                Error::config(format!("failed to build the Langfuse HTTP client: {error}"))
            })?;
        Ok(Self {
            config,
            http,
            max_batch_size: DEFAULT_MAX_BATCH_SIZE,
        })
    }

    /// Builds a client from `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and
    /// `LANGFUSE_HOST`/`LANGFUSE_BASEURL`.
    ///
    /// # Errors
    ///
    /// The errors of [`LangfuseConfig::from_env`] and [`Self::new`].
    pub fn from_env() -> Result<Self> {
        Self::new(LangfuseConfig::from_env()?)
    }

    /// Returns the client with a different HTTP client.
    #[must_use]
    pub fn with_http_client(mut self, client: reqwest::Client) -> Self {
        self.http = client;
        self
    }

    /// Returns the client with a different maximum events-per-request size.
    ///
    /// A size of `0` is treated as `1`, so a request always carries an event.
    #[must_use]
    pub fn with_max_batch_size(mut self, size: usize) -> Self {
        self.max_batch_size = size.max(1);
        self
    }

    /// Reports the credentials and host the client posts to.
    #[must_use]
    pub fn config(&self) -> &LangfuseConfig {
        &self.config
    }

    /// Reports the maximum number of events sent per request.
    #[must_use]
    pub fn max_batch_size(&self) -> usize {
        self.max_batch_size
    }

    /// Posts `batch`, splitting it into requests of at most
    /// [`Self::max_batch_size`] events.
    ///
    /// An empty batch returns [`IngestionSummary::default`] without contacting
    /// the network.
    ///
    /// # Errors
    ///
    /// [`Error::tool`] when a request fails at the transport level, when a
    /// response carries a non-2xx status, when a 2xx body is not JSON, or when
    /// the body reports per-event failures in its `errors` array. The message
    /// always names the status code and the (truncated) response body or the
    /// failing event ids.
    pub async fn ingest(&self, batch: IngestionBatch) -> Result<IngestionSummary> {
        self.ingest_events(batch.into_events()).await
    }

    /// Posts `events`, splitting them into requests of at most
    /// [`Self::max_batch_size`] events.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::ingest`].
    pub async fn ingest_events(&self, events: Vec<IngestionEvent>) -> Result<IngestionSummary> {
        if events.is_empty() {
            return Ok(IngestionSummary::default());
        }
        let mut summary = IngestionSummary::default();
        for chunk in events.chunks(self.max_batch_size) {
            self.post_chunk(chunk).await?;
            summary.events += chunk.len();
            summary.requests += 1;
        }
        debug!(
            target: LOG_TARGET,
            events = summary.events,
            requests = summary.requests,
            "posted an ingestion batch"
        );
        Ok(summary)
    }

    /// Posts one request-sized chunk and validates the response.
    async fn post_chunk(&self, events: &[IngestionEvent]) -> Result<()> {
        let url = self.config.ingestion_url();
        let body = json!({ "batch": events });
        let response = self
            .http
            .post(&url)
            .headers(self.headers()?)
            .json(&body)
            .send()
            .await
            .map_err(|error| {
                Error::tool(format!(
                    "the Langfuse ingestion request to {url} failed: {error}"
                ))
            })?;
        let status = response.status();
        let text = response.text().await.map_err(|error| {
            Error::tool(format!(
                "the Langfuse ingestion response from {url} could not be read: {error}"
            ))
        })?;
        if !status.is_success() {
            return Err(Error::tool(format!(
                "the Langfuse ingestion request to {url} failed with HTTP {status}: {}",
                snippet(&text),
            )));
        }
        check_response(status, &text)
    }

    /// Builds the request headers: JSON body plus the Basic credentials.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] when the base64 credentials are not a valid HTTP
    /// header value, which the preset's own validation already rules out.
    fn headers(&self) -> Result<HeaderMap> {
        let mut headers = HeaderMap::with_capacity(2);
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        let authorization =
            HeaderValue::from_str(&self.config.authorization()).map_err(|error| {
                Error::validation(format!(
                    "the Langfuse Basic credentials are not a valid HTTP header value: {error}"
                ))
            })?;
        headers.insert(AUTHORIZATION, authorization);
        Ok(headers)
    }
}

/// Validates the body of a 2xx ingestion response.
///
/// # Errors
///
/// [`Error::tool`] when the body is not JSON, when `errors` is not an array, or
/// when `errors` names at least one failed event.
fn check_response(status: StatusCode, body: &str) -> Result<()> {
    let parsed: Value = serde_json::from_str(body).map_err(|error| {
        Error::tool(format!(
            "the Langfuse ingestion response returned HTTP {status} with a body that is not JSON \
             ({error}): {}",
            snippet(body),
        ))
    })?;
    match parsed.get("errors") {
        None => Ok(()),
        Some(Value::Array(errors)) if errors.is_empty() => Ok(()),
        Some(Value::Array(errors)) => Err(Error::tool(format!(
            "the Langfuse ingestion response returned HTTP {status} with {} failed event(s): {}",
            errors.len(),
            describe_failures(errors),
        ))),
        Some(other) => Err(Error::tool(format!(
            "the Langfuse ingestion response returned HTTP {status} with a malformed `errors` \
             field (expected an array): {}",
            snippet(&other.to_string()),
        ))),
    }
}

/// Renders the per-event failures Langfuse reported.
///
/// Each entry becomes `id=<id> status=<status> message=<message>`; entries
/// beyond [`MAX_REPORTED_FAILURES`] are counted, not listed.
fn describe_failures(errors: &[Value]) -> String {
    let mut described: Vec<String> = errors
        .iter()
        .take(MAX_REPORTED_FAILURES)
        .map(|entry| {
            let field = |name: &str| {
                entry
                    .get(name)
                    .map_or_else(|| "<unknown>".to_owned(), render_value)
            };
            let message = entry
                .get("message")
                .or_else(|| entry.get("error"))
                .map_or_else(|| "<unknown>".to_owned(), render_value);
            format!(
                "id={} status={} message={message}",
                field("id"),
                field("status"),
            )
        })
        .collect();
    if errors.len() > MAX_REPORTED_FAILURES {
        described.push(format!(
            "… and {} more",
            errors.len() - MAX_REPORTED_FAILURES
        ));
    }
    described.join("; ")
}

/// Renders one JSON scalar (or a compact JSON value) for an error message.
fn render_value(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        other => other.to_string(),
    }
}

/// Keeps at most [`MAX_BODY_SNIPPET_CHARS`] characters of a response body.
///
/// The cut is made on a character boundary, so multi-byte text is never split.
fn snippet(body: &str) -> String {
    let trimmed = body.trim();
    if trimmed.chars().count() <= MAX_BODY_SNIPPET_CHARS {
        return trimmed.to_owned();
    }
    let kept: String = trimmed.chars().take(MAX_BODY_SNIPPET_CHARS).collect();
    format!("{kept}…")
}

#[cfg(test)]
mod tests;
