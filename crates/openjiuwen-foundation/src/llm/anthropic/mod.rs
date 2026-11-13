//! Anthropic Messages transport.
//!
//! Handles the Anthropic-specific message projection (system hoisting,
//! `tool_use`/`tool_result` blocks, base64/URL images, content-block cache
//! breakpoints) and the event-oriented SSE stream. Request, response, and stream
//! conversion live in the private submodules and are re-exported from here.

mod cache;
mod convert;
mod request;
mod response;
mod stream;
#[cfg(test)]
mod tests;

pub use request::{build_messages_body, normalize_api_base};
pub use response::{parse_anthropic_usage, parse_messages_response};
pub use stream::AnthropicStreamState;

use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use futures::stream::BoxStream;
use futures::StreamExt;
use serde_json::Value;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::http::{HttpClient, HttpRequest, HttpStream};

use crate::llm::config::{ModelClientConfig, ModelRequestConfig};
use crate::llm::model::{ChatResponse, Model, StreamChunk};
use crate::llm::provider::{provider_profile, ProviderFlavor, ANTHROPIC_VERSION};
use crate::llm::sse::SseDecoder;
use crate::prompt::BaseMessage;

use super::transport::{error_for_status, join_url};

/// Provider assumed when the client config leaves it empty.
const DEFAULT_PROVIDER: &str = "anthropic";

/// Maximum number of bytes of a failed streaming response kept for its error.
const MAX_ERROR_BODY_BYTES: usize = 8192;

/// Client for the Anthropic Messages API.
///
/// One client is bound to one endpoint, credential set, and model name. It
/// implements [`Model`], so it plugs into every layer above the foundation
/// exactly like the OpenAI-compatible client, and it is `Clone` so a caller can
/// share it between tasks.
///
/// ```no_run
/// use openjiuwen_foundation::llm::anthropic::AnthropicClient;
/// use openjiuwen_foundation::llm::{Model, ModelClientConfig, ModelRequestConfig};
///
/// # async fn example() -> openjiuwen_common::error::Result<()> {
/// let client = AnthropicClient::new(
///     ModelClientConfig::new("anthropic").with_api_key("sk-ant-test"),
///     ModelRequestConfig::new("claude-sonnet-4"),
/// )?;
/// let response = client
///     .invoke(&[openjiuwen_foundation::prompt::BaseMessage::user("hi")], &ModelRequestConfig::new("claude-sonnet-4"))
///     .await?;
/// assert!(response.message().is_some());
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct AnthropicClient {
    /// Resolved client configuration (canonical provider, credential, endpoint).
    client: ModelClientConfig,
    /// Request defaults used by [`Model::model_name`].
    request: ModelRequestConfig,
    /// Endpoint base with any `/v1`/`/v1/messages` suffix removed.
    base_url: String,
    /// Anthropic messages path (`/v1/messages`).
    chat_path: &'static str,
    /// Shared HTTP transport (retries, TLS, timeouts).
    http: HttpClient,
}

impl AnthropicClient {
    /// Builds a client from a client config and request defaults.
    ///
    /// An empty [`ModelClientConfig::provider`] means `anthropic`. The provider
    /// is resolved against the environment and the built-in table, and must map
    /// to a profile with [`ProviderFlavor::Anthropic`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] when the provider is not built in, does not
    /// speak the Anthropic Messages API, has no API base, or has no API key.
    pub fn new(client: ModelClientConfig, request: ModelRequestConfig) -> Result<Self> {
        let provider = if client.provider.trim().is_empty() {
            DEFAULT_PROVIDER.to_owned()
        } else {
            client.provider.clone()
        };
        let client = client.resolve_for(&provider)?;
        let profile = provider_profile(&client.provider).ok_or_else(|| {
            Error::config(format!(
                "no built-in provider profile for `{}`",
                client.provider
            ))
        })?;
        if profile.flavor != ProviderFlavor::Anthropic {
            return Err(Error::config(format!(
                "provider `{}` does not speak the Anthropic Messages API",
                client.provider
            )));
        }
        let api_base = client.api_base.as_deref().ok_or_else(|| {
            Error::config(format!(
                "provider `{}` has no API base; set `api_base`",
                client.provider
            ))
        })?;
        let base_url = normalize_api_base(api_base);
        let http = HttpClient::new(client.http_client_config())?;

        Ok(Self {
            client,
            request,
            base_url,
            chat_path: profile.chat_path,
            http,
        })
    }

    /// Replaces the shared HTTP transport.
    ///
    /// Useful for tests that inject a client with a custom retry policy or for
    /// applications that pool one transport across providers.
    #[must_use]
    pub fn with_http_client(mut self, http: HttpClient) -> Self {
        self.http = http;
        self
    }

    /// The resolved client configuration.
    #[must_use]
    pub fn client_config(&self) -> &ModelClientConfig {
        &self.client
    }

    /// The request defaults this client was built with.
    #[must_use]
    pub fn request_config(&self) -> &ModelRequestConfig {
        &self.request
    }

    /// The normalized API base; the messages path is appended to it.
    #[must_use]
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Builds one authorized `POST /v1/messages` request for `body`.
    fn build_http_request(&self, body: &Value) -> Result<HttpRequest> {
        let mut request = HttpRequest::post(join_url(&self.base_url, self.chat_path))
            .header("anthropic-version", ANTHROPIC_VERSION)
            .with_headers(self.client.sanitized_headers());

        if let Some(api_key) = &self.client.api_key {
            if !api_key.is_empty() {
                request = request.header("x-api-key", api_key.expose_secret());
            }
        }
        if let Some(timeout) = self.client.timeout {
            request = request.timeout(timeout);
        }

        request.json(body)
    }
}

#[async_trait]
impl Model for AnthropicClient {
    fn provider(&self) -> &str {
        self.client.provider.as_str()
    }

    fn model_name(&self) -> &str {
        self.request.model_name.as_str()
    }

    async fn invoke(
        &self,
        messages: &[BaseMessage],
        request: &ModelRequestConfig,
    ) -> Result<ChatResponse> {
        let body = build_messages_body(messages, request, false)?;
        let http_request = self.build_http_request(&body)?;
        let response = self.http.execute(http_request).await?;

        if !response.is_success() {
            return Err(error_for_status(
                response.status,
                &response.url,
                &response.text(),
            ));
        }

        let value: Value = serde_json::from_slice(&response.body).map_err(|error| {
            Error::llm(format!(
                "decoding the Anthropic response from {}: {error}",
                response.url
            ))
        })?;
        parse_messages_response(&value)
    }

    fn stream(
        &self,
        messages: Vec<BaseMessage>,
        request: ModelRequestConfig,
    ) -> BoxStream<'static, Result<StreamChunk>> {
        let prepared = build_messages_body(&messages, &request, true)
            .and_then(|body| self.build_http_request(&body));
        let http_request = match prepared {
            Ok(http_request) => http_request,
            Err(error) => return Box::pin(futures::stream::once(async move { Err(error) })),
        };

        let http = self.http.clone();
        let first_chunk_timeout = self.client.stream_first_chunk_timeout;
        let idle_timeout = self.client.stream_idle_timeout;

        Box::pin(async_stream::try_stream! {
            let response = http.stream(http_request).await?;
            let status = response.status;
            let url = response.url;
            let mut body = response.body;

            if !(200..300).contains(&status) {
                let detail = read_error_body(&mut body).await?;
                Err(error_for_status(status, &url, &detail))?;
            }

            let mut decoder = SseDecoder::new();
            let mut state = AnthropicStreamState::new();
            let mut first_token_seen = false;

            loop {
                let timeout = if first_token_seen { idle_timeout } else { first_chunk_timeout };
                let Some(bytes) = next_chunk(&mut body, timeout).await? else {
                    break;
                };
                for event in decoder.push(&bytes) {
                    for chunk in state.handle_event(&event)? {
                        if is_first_token(&chunk) {
                            first_token_seen = true;
                        }
                        yield chunk;
                    }
                }
            }

            if let Some(event) = decoder.finish() {
                for chunk in state.handle_event(&event)? {
                    yield chunk;
                }
            }

            for chunk in state.finish() {
                yield chunk;
            }
        })
    }
}

/// Reads one streaming body chunk, enforcing `timeout` when configured.
async fn next_chunk(body: &mut HttpStream, timeout: Option<Duration>) -> Result<Option<Bytes>> {
    match timeout {
        Some(limit) => match tokio::time::timeout(limit, body.next()).await {
            Ok(item) => item.transpose(),
            Err(_) => Err(Error::timeout(format!(
                "the Anthropic stream produced no data within {limit:?}"
            ))),
        },
        None => body.next().await.transpose(),
    }
}

/// Buffers a failed streaming response body for the provider error message.
async fn read_error_body(body: &mut HttpStream) -> Result<String> {
    let mut buffer = Vec::new();
    while let Some(chunk) = body.next().await {
        buffer.extend_from_slice(&chunk?);
        if buffer.len() >= MAX_ERROR_BODY_BYTES {
            break;
        }
    }
    Ok(String::from_utf8_lossy(&buffer).into_owned())
}

/// Returns `true` for a chunk that carries the first piece of model output.
fn is_first_token(chunk: &StreamChunk) -> bool {
    match chunk {
        StreamChunk::Delta {
            content,
            reasoning,
            tool_call,
            ..
        } => {
            content.as_deref().is_some_and(|text| !text.is_empty())
                || reasoning.as_deref().is_some_and(|text| !text.is_empty())
                || tool_call.is_some()
        }
        StreamChunk::Usage(_) | StreamChunk::Done => false,
    }
}
