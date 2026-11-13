//! OpenAI-compatible chat-completions transport.
//!
//! Handles the chat-completions wire format shared by `OpenAI`, `Qwen`/`DashScope`,
//! `DeepSeek`, `Moonshot`, `OpenRouter`, `Zhipu`, `MiniMax`, `SiliconFlow`,
//! `Ollama`, LM Studio, `Pangu`, and every other vendor the provider table marks
//! as [`ProviderFlavor::OpenAiCompatible`]. Message and body projection live in
//! `request`, response parsing in `response`, and event-stream decoding in
//! `stream`; [`OpenAiCompatibleClient`] wires them to the shared HTTP client.
//!
//! ```no_run
//! use openjiuwen_foundation::llm::config::{ModelClientConfig, ModelRequestConfig};
//! use openjiuwen_foundation::llm::model::Model;
//! use openjiuwen_foundation::llm::openai::OpenAiCompatibleClient;
//! use openjiuwen_foundation::prompt::BaseMessage;
//!
//! # async fn example() -> openjiuwen_common::error::Result<()> {
//! let client = OpenAiCompatibleClient::new(
//!     ModelClientConfig::new("openai").with_api_key("sk-test"),
//!     ModelRequestConfig::new("gpt-4o"),
//! )?;
//! let response = client
//!     .invoke(&[BaseMessage::user("hello")], &ModelRequestConfig::new("gpt-4o"))
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

use super::config::{ModelClientConfig, ModelRequestConfig};
use super::model::{ChatResponse, Model, StreamChunk};
use super::provider::{normalize_provider_name, provider_profile, ProviderFlavor, ProviderProfile};
use super::sse::SseDecoder;
use super::transport::{self, carries_payload, next_chunk, poll_budget, read_error_body};

pub use request::{
    build_chat_completion_body, convert_message, convert_messages, convert_tool_choice,
    convert_tools, normalize_api_base, replays_reasoning_content,
};
pub use response::{parse_chat_completion, parse_finish_reason, parse_usage};
pub use stream::{parse_stream_chunk, OpenAiStreamState};

/// An OpenAI-compatible chat-completions client.
///
/// The client owns its resolved configuration, so it never re-reads the
/// environment after [`new`](Self::new). One client is bound to one
/// provider/model pair; [`Model::invoke`] and [`Model::stream`] still take a
/// per-call [`ModelRequestConfig`], whose `model_name` must not be empty.
///
/// [`new`](Self::new) normalizes the provider, fills in environment fallbacks and
/// the provider's default API base (via
/// [`ModelClientConfig::resolve_for`](super::config::ModelClientConfig::resolve_for)),
/// rejects providers that do not speak this transport, and strips a pasted
/// `/chat/completions` suffix from `api_base`.
#[derive(Debug, Clone)]
pub struct OpenAiCompatibleClient {
    /// Shared HTTP transport (retries, TLS, proxy, default headers).
    http: HttpClient,
    /// Resolved client configuration (credentials, timeouts, headers).
    client: ModelClientConfig,
    /// Default per-request configuration, used for [`Model::model_name`].
    request: ModelRequestConfig,
    /// Provider profile that supplies the chat-completions path.
    profile: ProviderProfile,
    /// Canonical provider name reported by [`Model::provider`].
    provider: String,
    /// API base with trailing slashes and a pasted endpoint path removed.
    base_url: String,
}

impl OpenAiCompatibleClient {
    /// Builds a client for the provider named in `client`.
    ///
    /// A blank provider resolves to `openai`. The provider must exist in the
    /// built-in table and speak the OpenAI-compatible protocol: asking this
    /// constructor for `anthropic` is a [`Error::Config`] error rather than a
    /// client that fails on the first request.
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
                "provider `{}` has no OpenAI-compatible endpoint profile",
                client.provider
            ))
        })?;
        if profile.flavor != ProviderFlavor::OpenAiCompatible {
            return Err(Error::config(format!(
                "provider `{}` speaks the {:?} protocol, not the OpenAI-compatible \
                 chat-completions API",
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

    /// Builds an unauthenticated request carrying the configured headers,
    /// bearer token, and per-request timeout.
    fn build_request(&self, url: String) -> HttpRequest {
        let mut request = HttpRequest::post(url).with_headers(self.client.sanitized_headers());
        if let Some(api_key) = self.client.api_key.as_ref().filter(|key| !key.is_empty()) {
            request = request.bearer(api_key.clone());
        }
        if let Some(timeout) = self.client.timeout {
            request = request.timeout(timeout);
        }
        request
    }
}

#[async_trait]
impl Model for OpenAiCompatibleClient {
    fn provider(&self) -> &str {
        &self.provider
    }

    fn model_name(&self) -> &str {
        self.request.model_name.as_str()
    }

    async fn invoke(
        &self,
        messages: &[BaseMessage],
        request: &ModelRequestConfig,
    ) -> Result<ChatResponse> {
        let body = build_chat_completion_body(&self.provider, messages, request, false)?;
        let url = transport::join_url(&self.base_url, self.profile.chat_path);
        let http_request = self.build_request(url.clone()).json(&body)?;

        let response = self.http.execute(http_request).await?;
        if !response.is_success() {
            return Err(transport::error_for_status(
                response.status,
                &url,
                &response.text(),
            ));
        }

        let value: Value = serde_json::from_slice(response.bytes()).map_err(|error| {
            Error::llm(format!(
                "decoding the chat completion response from {url}: {error}; body: {}",
                transport::redacted_body(&response.text())
            ))
        })?;
        parse_chat_completion(&value)
            .map_err(|error| error.with_context(format!("parsing the chat completion from {url}")))
    }

    fn stream(
        &self,
        messages: Vec<BaseMessage>,
        request: ModelRequestConfig,
    ) -> BoxStream<'static, Result<StreamChunk>> {
        let http = self.http.clone();
        let provider = self.provider.clone();
        let base_url = self.base_url.clone();
        let chat_path = self.profile.chat_path;
        let headers = self.client.sanitized_headers();
        let api_key = self.client.api_key.clone();
        let timeout = self.client.timeout;
        let first_chunk_timeout = self.client.stream_first_chunk_timeout;
        let idle_timeout = self.client.stream_idle_timeout;

        Box::pin(try_stream! {
            let body = build_chat_completion_body(&provider, &messages, &request, true)?;
            let url = transport::join_url(&base_url, chat_path);
            let mut http_request = HttpRequest::post(url.clone()).with_headers(headers);
            if let Some(api_key) = api_key.filter(|key| !key.is_empty()) {
                http_request = http_request.bearer(api_key);
            }
            if let Some(timeout) = timeout {
                http_request = http_request.timeout(timeout);
            }

            let response = http.stream(http_request.json(&body)?).await?;
            let mut body = response.body;
            if !(200..300).contains(&response.status) {
                let detail = read_error_body(&mut body).await;
                Err::<(), Error>(transport::error_for_status(
                    response.status,
                    &url,
                    &detail,
                ))?;
            }

            let mut decoder = SseDecoder::new();
            let mut state = OpenAiStreamState::new();
            let mut content_seen = false;
            let mut finished = false;
            let first_chunk_deadline = first_chunk_timeout.map(|limit| Instant::now() + limit);

            // Stop reading as soon as the provider terminates the logical stream:
            // a connection that stays open after `[DONE]` must not delay the
            // caller (or trip the idle budget).
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
