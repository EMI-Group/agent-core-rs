//! `OpenAI` Responses API transport (`POST {base}/responses`).
//!
//! This is the third model protocol of the crate, beside the OpenAI-compatible
//! chat-completions client ([`crate::llm::openai`]) and the Anthropic Messages
//! client ([`crate::llm::anthropic`]). It ports the account Responses backend of
//! `openjiuwen/core/foundation/llm/utils/responses_utils.py` and
//! `.../utils/responses_transport.py`, plus the `api_mode == responses` branch of
//! `.../model_clients/openai_model_client.py`.
//!
//! Message and body projection live in `request`, response parsing in `response`,
//! and event-stream decoding in `stream`; [`OpenAiResponsesClient`] wires them to
//! the shared HTTP client. Both request modes stream: the endpoint has no
//! non-streaming shape this crate relies on, so [`Model::invoke`] drives the SSE
//! parser and aggregates the chunks, exactly like the Python transport's
//! `create_response`.
//!
//! ```no_run
//! use openjiuwen_foundation::llm::config::{ModelClientConfig, ModelRequestConfig};
//! use openjiuwen_foundation::llm::model::Model;
//! use openjiuwen_foundation::llm::responses::OpenAiResponsesClient;
//! use openjiuwen_foundation::prompt::BaseMessage;
//!
//! # async fn example() -> openjiuwen_common::error::Result<()> {
//! let client = OpenAiResponsesClient::new(
//!     ModelClientConfig::new("openai_account").with_api_key("sk-test"),
//!     ModelRequestConfig::new("gpt-5"),
//! )?;
//! let response = client
//!     .invoke(&[BaseMessage::user("hello")], &ModelRequestConfig::new("gpt-5"))
//!     .await?;
//! let text = response.message().map(|message| message.text().into_owned());
//! assert_eq!(text.as_deref(), Some("hi"));
//! # Ok(())
//! # }
//! ```

mod request;
mod response;
mod stream;

#[cfg(test)]
mod tests;

use std::time::Instant;

use async_stream::try_stream;
use async_trait::async_trait;
use futures::stream::BoxStream;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::http::{HttpClient, HttpRequest};
use serde_json::Value;

use crate::prompt::BaseMessage;

use super::config::{ApiMode, ModelClientConfig, ModelRequestConfig};
use super::model::{ChatResponse, Choice, FinishReason, Model, StreamAccumulator, StreamChunk};
use super::provider::{
    normalize_provider_name, provider_profile, ProviderFlavor, ProviderProfile, RESPONSES_PATH,
};
use super::sse::SseDecoder;
use super::transport::{self, carries_payload, next_chunk, poll_budget, read_error_body};

pub use request::{build_responses_body, convert_messages, convert_tools, normalize_api_base};
pub use response::{parse_responses_response, parse_responses_usage};
pub use stream::ResponsesStreamState;

/// Header the account backend uses to scope a conversation.
const SESSION_HEADER: &str = "session_id";

/// Header the account backend uses to correlate a request with one turn.
const CLIENT_REQUEST_ID_HEADER: &str = "x-client-request-id";

/// The `OpenAI` Responses API client.
///
/// The client owns its resolved configuration, so it never re-reads the
/// environment after [`new`](Self::new). One client is bound to one
/// provider/model pair; [`Model::invoke`] and [`Model::stream`] still take a
/// per-call [`ModelRequestConfig`], whose `model_name` must not be empty.
///
/// [`new`](Self::new) normalizes the provider, fills in environment fallbacks and
/// the provider's default API base (via
/// [`ModelClientConfig::resolve_for`](super::config::ModelClientConfig::resolve_for)),
/// and rejects providers that do not speak this transport. The `openai` provider is
/// accepted when its [`api_mode`](ModelClientConfig::api_mode) is
/// [`ApiMode::Responses`]; `openai_account` always speaks Responses. A pasted
/// `/responses` suffix is stripped from `api_base`.
#[derive(Debug, Clone)]
pub struct OpenAiResponsesClient {
    /// Shared HTTP transport (retries, TLS, proxy, default headers).
    http: HttpClient,
    /// Resolved client configuration (credentials, timeouts, headers).
    client: ModelClientConfig,
    /// Default per-request configuration, used for [`Model::model_name`].
    request: ModelRequestConfig,
    /// Provider profile that supplies the responses path.
    profile: ProviderProfile,
    /// Canonical provider name reported by [`Model::provider`].
    provider: String,
    /// API base with trailing slashes and a pasted endpoint path removed.
    base_url: String,
}

impl OpenAiResponsesClient {
    /// Builds a client for the provider named in `client`.
    ///
    /// A blank provider resolves to `openai`. The provider must exist in the
    /// built-in table and either declare the
    /// [`OpenAiResponses`](ProviderFlavor::OpenAiResponses) flavor or be `openai`
    /// configured with [`ApiMode::Responses`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::Unsupported`] for an unknown provider, [`Error::Config`]
    /// when the provider speaks another protocol, has no API base, or requires a
    /// key that neither the configuration nor the environment supplies, and the
    /// errors of [`HttpClient::new`] for an unusable transport configuration.
    pub fn new(client: ModelClientConfig, request: ModelRequestConfig) -> Result<Self> {
        let provider = if client.provider.trim().is_empty() {
            normalize_provider_name("openai")
        } else {
            client.provider.clone()
        };
        let client = client.resolve_for(&provider)?;

        let profile = provider_profile(&client.provider).ok_or_else(|| {
            Error::config(format!(
                "provider `{}` has no Responses endpoint profile",
                client.provider
            ))
        })?;
        let speaks_responses = profile.flavor == ProviderFlavor::OpenAiResponses
            || (profile.flavor == ProviderFlavor::OpenAiCompatible
                && client.api_mode == Some(ApiMode::Responses));
        if !speaks_responses {
            return Err(Error::config(format!(
                "provider `{}` speaks the {:?} protocol, not the OpenAI Responses API; \
                 set `api_mode: responses` to use the Responses endpoint",
                profile.name, profile.flavor
            )));
        }

        let api_base = client.api_base.clone().ok_or_else(|| {
            Error::config(format!(
                "provider `{}` has no API base; set `api_base` (or the API_BASE \
                 environment variable) to the deployment endpoint",
                profile.name
            ))
        })?;
        let base_url = normalize_api_base(&api_base);
        let http = HttpClient::new(client.http_client_config())?;

        Ok(Self {
            http,
            client,
            request,
            profile,
            provider: profile.name.to_owned(),
            base_url,
        })
    }

    /// Replaces the HTTP transport.
    ///
    /// Useful for tests (custom timeouts and retry backoff) and for embedding an
    /// application-wide transport; the client's own headers, bearer token, and
    /// per-request timeout are still applied on top.
    #[must_use]
    pub fn with_http_client(mut self, http: HttpClient) -> Self {
        self.http = http;
        self
    }

    /// The resolved client configuration (credentials are redacted by `Debug`).
    #[must_use]
    pub fn client_config(&self) -> &ModelClientConfig {
        &self.client
    }

    /// The default per-request configuration.
    #[must_use]
    pub fn request_config(&self) -> &ModelRequestConfig {
        &self.request
    }

    /// The normalized API base URL that endpoint paths are appended to.
    #[must_use]
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// The endpoint path this client appends to [`base_url`](Self::base_url).
    #[must_use]
    pub fn chat_path(&self) -> &'static str {
        RESPONSES_PATH
    }

    /// Builds an unauthenticated request carrying the configured headers, bearer
    /// token, `Accept`/`Content-Type`, per-call session headers, and timeout.
    fn build_request(&self, url: String, request: &ModelRequestConfig) -> HttpRequest {
        let mut builder = HttpRequest::post(url)
            .with_headers(self.client.sanitized_headers())
            .header("Accept", "application/json")
            .header("Content-Type", "application/json");
        if let Some(api_key) = self.client.api_key.as_ref().filter(|key| !key.is_empty()) {
            builder = builder.bearer(api_key.clone());
        }
        if let Some(timeout) = self.client.timeout {
            builder = builder.timeout(timeout);
        }
        if let Some(session) = session_id(request) {
            builder = builder
                .header(SESSION_HEADER, session)
                .header(CLIENT_REQUEST_ID_HEADER, session);
        }
        builder
    }

    /// The URL of the responses endpoint.
    fn endpoint(&self) -> String {
        transport::join_url(&self.base_url, RESPONSES_PATH)
    }
}

/// The per-call session id, when the request carries one.
fn session_id(request: &ModelRequestConfig) -> Option<&str> {
    request
        .extra
        .get(SESSION_HEADER)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|session| !session.is_empty())
}

#[async_trait]
impl Model for OpenAiResponsesClient {
    fn provider(&self) -> &str {
        &self.provider
    }

    fn model_name(&self) -> &str {
        self.request.model_name.as_str()
    }

    /// Invokes the endpoint by streaming and aggregating the events.
    ///
    /// The Responses backend of an `OpenAI` account only answers event streams, so
    /// this is the same path as the Python transport's `create_response`: force
    /// `stream: true`, decode every SSE event, and fold the chunks into one
    /// [`ChatResponse`] with a [`StreamAccumulator`].
    async fn invoke(
        &self,
        messages: &[BaseMessage],
        request: &ModelRequestConfig,
    ) -> Result<ChatResponse> {
        let body = build_responses_body(messages, request, true)?;
        let url = self.endpoint();
        let http_request = self.build_request(url.clone(), request).json(&body)?;

        let response = self.http.execute(http_request).await?;
        if !response.is_success() {
            return Err(error_for_response(response.status, &url, &response.text()));
        }

        let mut decoder = SseDecoder::new();
        let mut state = ResponsesStreamState::new();
        let mut accumulator = StreamAccumulator::new();

        let mut events = decoder.push(response.bytes());
        if let Some(event) = decoder.finish() {
            events.push(event);
        }
        for event in events {
            for chunk in state.handle_data(&event.data)? {
                accumulator.push(&chunk);
            }
        }
        for chunk in state.finish() {
            accumulator.push(&chunk);
        }

        let mut turn = accumulator.finish();
        if !turn.message.tool_calls.is_empty() {
            // A completed response reports `stop`; tool calls are the real reason
            // (Python's `message_from_stream_chunk`).
            turn.finish_reason = Some(FinishReason::ToolCalls);
        }
        if turn.finish_reason.is_none() {
            turn.finish_reason = Some(FinishReason::Stop);
        }

        Ok(ChatResponse {
            id: String::new(),
            model: request.model_name.trim().to_owned(),
            choices: vec![Choice {
                index: 0,
                message: turn.message,
                finish_reason: turn.finish_reason,
            }],
            usage: turn.usage,
        })
    }

    fn stream(
        &self,
        messages: Vec<BaseMessage>,
        request: ModelRequestConfig,
    ) -> BoxStream<'static, Result<StreamChunk>> {
        let http = self.http.clone();
        let base_url = self.base_url.clone();
        let path = self.profile.chat_path;
        let headers = self.client.sanitized_headers();
        let api_key = self.client.api_key.clone();
        let timeout = self.client.timeout;
        let first_chunk_timeout = self.client.stream_first_chunk_timeout;
        let idle_timeout = self.client.stream_idle_timeout;

        Box::pin(try_stream! {
            let body = build_responses_body(&messages, &request, true)?;
            let url = transport::join_url(&base_url, path);
            let mut http_request = HttpRequest::post(url.clone())
                .with_headers(headers)
                .header("Accept", "application/json")
                .header("Content-Type", "application/json");
            if let Some(api_key) = api_key.filter(|key| !key.is_empty()) {
                http_request = http_request.bearer(api_key);
            }
            if let Some(timeout) = timeout {
                http_request = http_request.timeout(timeout);
            }
            if let Some(session) = session_id(&request) {
                http_request = http_request
                    .header(SESSION_HEADER, session)
                    .header(CLIENT_REQUEST_ID_HEADER, session);
            }

            let response = http.stream(http_request.json(&body)?).await?;
            let mut body = response.body;
            if !(200..300).contains(&response.status) {
                let detail = read_error_body(&mut body).await;
                Err::<(), Error>(error_for_response(response.status, &url, &detail))?;
            }

            let mut decoder = SseDecoder::new();
            let mut state = ResponsesStreamState::new();
            let mut content_seen = false;
            let mut finished = false;
            let first_chunk_deadline = first_chunk_timeout.map(|limit| Instant::now() + limit);

            while !finished {
                let budget = poll_budget(content_seen, idle_timeout, first_chunk_deadline);
                let Some(chunk) = next_chunk(&mut body, budget, &url, content_seen).await? else {
                    break;
                };
                for event in decoder.push(&chunk) {
                    for item in state.handle_data(&event.data)? {
                        content_seen |= carries_payload(&item);
                        finished |= matches!(item, StreamChunk::Done);
                        yield item;
                    }
                    if finished {
                        break;
                    }
                }
            }

            if let Some(event) = decoder.finish() {
                for item in state.handle_data(&event.data)? {
                    yield item;
                }
            }
            for item in state.finish() {
                yield item;
            }
        })
    }
}

/// Maps a non-2xx Responses reply to the error kind the SDK promises callers.
///
/// The message follows the Python precedence: `error.message`, then an `error`
/// string, then `detail`, then a summarized body, then the generic
/// `...failed with status N.` sentence. The detail is redacted and truncated, and
/// the status picks the kind (`401`/`403` → [`Error::Config`],
/// `408`/`504` → [`Error::Timeout`], anything else → [`Error::Llm`]).
fn error_for_response(status: u16, url: &str, body: &str) -> Error {
    let detail = provider_message(body);
    let message = detail.unwrap_or_else(|| {
        format!("OpenAI Responses request failed with status {status} for {url}.")
    });

    match status {
        401 | 403 => Error::config(format!("authentication failed (HTTP {status}): {message}")),
        408 | 504 => Error::timeout(format!("request timed out (HTTP {status}): {message}")),
        _ => Error::llm(message),
    }
}

/// Extracts the human-readable provider message from an error body.
///
/// Returns the redacted, truncated detail or `None` when the body carries no
/// message, so the caller can fall back to the status sentence.
fn provider_message(body: &str) -> Option<String> {
    let trimmed = body.trim();
    if trimmed.is_empty() {
        return None;
    }
    let message = serde_json::from_str::<Value>(trimmed)
        .ok()
        .and_then(|value| {
            let object = value.as_object()?;
            if let Some(message) = object
                .get("error")
                .and_then(|error| error.get("message"))
                .and_then(Value::as_str)
                .filter(|message| !message.is_empty())
            {
                return Some(message.to_owned());
            }
            if let Some(error) = object
                .get("error")
                .and_then(Value::as_str)
                .filter(|error| !error.is_empty())
            {
                return Some(error.to_owned());
            }
            object
                .get("detail")
                .and_then(Value::as_str)
                .filter(|detail| !detail.is_empty())
                .map(str::to_owned)
        });
    let message = message.unwrap_or_else(|| trimmed.to_owned());
    Some(transport::redacted_body(&message))
}
