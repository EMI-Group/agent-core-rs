//! The injected text-completion seam used by the context-evolver pipelines.
//!
//! Ports the client half of Python's
//! `openjiuwen/extensions/context_evolver/service/task_memory_service.py`
//! (`LLMWrapper`, lines 105-264): an `OpenAI`-compatible chat-completions call
//! driven by environment configuration, plus the JSON-reply handling the
//! retrieve/summarize pipelines need.
//!
//! # Why a narrow trait
//!
//! The pipelines only need "system prompt + user prompt → text". The kernel
//! clients in `openjiuwen_foundation::llm` model far more (tool calls, streaming,
//! message history, provider-specific config), and coupling the memory
//! pipelines to them would drag that surface — and every vendor adapter — into
//! this crate. [`MemoryCompletion`] is deliberately one method wide:
//!
//! - a host adapts *any* model by implementing the trait (its own client, an
//!   in-process model, a recorded fixture), so no vendor SDK is required;
//! - the pipelines are unit-testable without a network, using
//!   [`ScriptedCompletion`], a public deterministic test double;
//! - [`OpenAiChatCompletion`] is the bundled HTTP adapter, and because it speaks
//!   the `OpenAI` `/chat/completions` wire format it also drives `DeepSeek`,
//!   vLLM, and any other compatible endpoint by pointing `base_url` at it.
//!
//! # Malformed-output contract
//!
//! Models routinely fence their JSON or wrap it in prose. [`parse_json_reply`]
//! is the single tolerant entry point: strict JSON, then a stripped
//! ```` ```json ````/```` ``` ```` fence, then the first balanced `{...}`/`[...]`
//! run embedded in text. Only when all three fail does it return
//! [`Error::Validation`] naming the pipeline phase and previewing the reply;
//! it never panics on truncated or otherwise odd input.
//!
//! # Example
//!
//! ```
//! use openjiuwen_extensions::context_evolver::completion::{
//!     parse_json_reply, strip_code_fence, ChatCompletionConfig,
//! };
//!
//! assert_eq!(
//!     strip_code_fence("```json\n{\"memories\": []}\n```"),
//!     "{\"memories\": []}"
//! );
//! let reply = parse_json_reply("retrieve", "Sure: {\"memories\": []}").unwrap();
//! assert_eq!(reply.to_string(), "{\"memories\":[]}");
//!
//! let config = ChatCompletionConfig::new("https://example.test/v1/", "test-model")
//!     .with_api_key("sk-secret");
//! assert_eq!(config.base_url(), "https://example.test/v1");
//! assert!(!format!("{config:?}").contains("sk-secret"));
//! ```
use std::collections::VecDeque;
use std::fmt;
use std::sync::{Mutex, MutexGuard, PoisonError};
use std::time::Duration;

use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use serde_json::{json, Value};

/// The `OpenAI`-compatible chat-completions path appended to the base URL.
const CHAT_COMPLETIONS_PATH: &str = "/chat/completions";

/// The base URL used when the environment configures none.
const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

/// The model used when the environment configures none.
const DEFAULT_MODEL: &str = "gpt-4o-mini";

/// The placeholder printed instead of the API key by [`ChatCompletionConfig`]'s
/// `Debug` implementation.
const REDACTED: &str = "<redacted>";

/// Maximum number of response-body characters kept in a transport error.
const MAX_BODY_PREVIEW: usize = 200;

/// Maximum number of reply characters kept in a malformed-JSON error.
const MAX_REPLY_PREVIEW: usize = 120;

/// Injected text-completion backend for the context-evolver pipelines.
///
/// A host adapts any model by implementing this trait; [`OpenAiChatCompletion`]
/// is the bundled `OpenAI`-compatible HTTP adapter and [`ScriptedCompletion`]
/// is a deterministic test double. The trait is object-safe and requires
/// `Send + Sync + Debug` so the service structs that own a completion backend
/// can be boxed, shared across tasks, and derive `Debug`.
#[async_trait]
pub trait MemoryCompletion: Send + Sync + fmt::Debug {
    /// Sends the prompt pair and returns the model's raw text reply.
    ///
    /// # Errors
    ///
    /// Implementations return [`Error::transport`] when the backend cannot be
    /// reached or answers with an unusable payload, and [`Error::timeout`] when
    /// the call exceeds the configured deadline.
    async fn complete(&self, system: &str, user: &str) -> Result<String>;

    /// Sends the prompt pair and parses the reply as JSON (fences tolerated).
    ///
    /// `phase` names the calling pipeline (`"retrieve"`, `"summarize"`, …) and
    /// appears in the error message when the reply is not JSON.
    ///
    /// # Errors
    ///
    /// Propagates [`Self::complete`] failures, and returns [`Error::Validation`]
    /// when the reply cannot be read as JSON (see [`parse_json_reply`]).
    async fn complete_json(&self, phase: &str, system: &str, user: &str) -> Result<Value> {
        let reply = self.complete(system, user).await?;
        parse_json_reply(phase, &reply)
    }
}

/// Configuration for [`OpenAiChatCompletion`].
///
/// The field set mirrors Python's `LLMWrapper` constructor (`model_name`,
/// `api_key`, `base_url`, `temperature`, `max_tokens`) plus a request timeout.
/// `Debug` is implemented manually: the API key is printed as `<redacted>` so a
/// derived `Debug` on a host's service struct can never leak it into a log line.
#[derive(Clone, PartialEq)]
pub struct ChatCompletionConfig {
    /// The API root, for example `https://api.openai.com/v1`; trailing `/` is
    /// trimmed.
    pub base_url: String,
    /// The bearer token, when the endpoint requires one.
    pub api_key: Option<String>,
    /// The provider's model identifier, for example `gpt-4o-mini`.
    pub model: String,
    /// Sampling temperature; defaults to `0.7`.
    pub temperature: f32,
    /// Maximum reply tokens; defaults to `4096` (`max_tokens` on the wire).
    pub max_tokens: u32,
    /// Per-request deadline; defaults to 60 seconds.
    pub timeout: Duration,
}

impl fmt::Debug for ChatCompletionConfig {
    /// Never prints the API key (see the type docs).
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ChatCompletionConfig")
            .field("base_url", &self.base_url)
            .field("api_key", &self.api_key.as_ref().map(|_| REDACTED))
            .field("model", &self.model)
            .field("temperature", &self.temperature)
            .field("max_tokens", &self.max_tokens)
            .field("timeout", &self.timeout)
            .finish()
    }
}

impl ChatCompletionConfig {
    /// Builds a configuration for `base_url` and `model` with the defaults
    /// `temperature = 0.7`, `max_tokens = 4096`, and a 60 second timeout.
    ///
    /// A trailing `/` on `base_url` is trimmed so `{base_url}/chat/completions`
    /// never doubles the separator.
    #[must_use]
    pub fn new(base_url: impl Into<String>, model: impl Into<String>) -> Self {
        let base_url = base_url.into();
        let base_url = base_url.trim().trim_end_matches('/').to_string();
        Self {
            base_url,
            api_key: None,
            model: model.into().trim().to_string(),
            temperature: 0.7,
            max_tokens: 4096,
            timeout: Duration::from_secs(60),
        }
    }

    /// Sets the bearer token.
    #[must_use]
    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Overrides the sampling temperature.
    #[must_use]
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    /// Overrides the maximum reply tokens.
    #[must_use]
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Overrides the per-request deadline.
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// The API root (no trailing `/`).
    #[must_use]
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// The configured bearer token, if any.
    ///
    /// This is the secret itself: never log the returned value.
    #[must_use]
    pub fn api_key(&self) -> Option<&str> {
        self.api_key.as_deref()
    }

    /// The model identifier.
    #[must_use]
    pub fn model(&self) -> &str {
        &self.model
    }

    /// The sampling temperature.
    #[must_use]
    pub fn temperature(&self) -> f32 {
        self.temperature
    }

    /// The maximum reply tokens.
    #[must_use]
    pub fn max_tokens(&self) -> u32 {
        self.max_tokens
    }

    /// The per-request deadline.
    #[must_use]
    pub fn timeout(&self) -> Duration {
        self.timeout
    }

    /// The full `chat/completions` endpoint for this configuration.
    #[must_use]
    pub fn endpoint(&self) -> String {
        format!("{}{CHAT_COMPLETIONS_PATH}", self.base_url)
    }

    /// Reads the configuration through `get` (pure environment lookup).
    ///
    /// Lookup order matches the Python wrapper's environment conventions:
    ///
    /// - base URL: `OPENAI_BASE_URL`, then `API_BASE`, then
    ///   `https://api.openai.com/v1`;
    /// - key: `OPENAI_API_KEY`, then `API_KEY`;
    /// - model: `MODEL_NAME`, then `OPENAI_MODEL`, then `gpt-4o-mini`;
    /// - timeout: `OPENJIUWEN_LLM_TIMEOUT_SECS` as a positive integer number of
    ///   seconds when present and well formed (otherwise the default stands).
    ///
    /// The first non-empty value wins for every key. Taking a closure instead of
    /// reading the process environment makes the precedence unit-testable.
    #[must_use]
    pub fn from_lookup(get: impl Fn(&str) -> Option<String>) -> Self {
        let base_url = first_non_empty(&get, &["OPENAI_BASE_URL", "API_BASE"])
            .unwrap_or_else(|| DEFAULT_BASE_URL.to_string());
        let model = first_non_empty(&get, &["MODEL_NAME", "OPENAI_MODEL"])
            .unwrap_or_else(|| DEFAULT_MODEL.to_string());
        let mut config = Self::new(base_url, model);
        config.api_key = first_non_empty(&get, &["OPENAI_API_KEY", "API_KEY"]);
        if let Some(seconds) = first_non_empty(&get, &["OPENJIUWEN_LLM_TIMEOUT_SECS"])
            .and_then(|value| value.parse::<u64>().ok())
            .filter(|seconds| *seconds > 0)
        {
            config.timeout = Duration::from_secs(seconds);
        }
        config
    }

    /// Reads the configuration from the process environment (see
    /// [`Self::from_lookup`] for the precedence).
    #[must_use]
    pub fn from_env() -> Self {
        Self::from_lookup(|name| std::env::var(name).ok())
    }
}

/// Returns the first non-empty (after trimming) value among `names`.
fn first_non_empty(get: &impl Fn(&str) -> Option<String>, names: &[&str]) -> Option<String> {
    names.iter().find_map(|name| {
        get(name)
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
    })
}

/// The `OpenAI`-compatible HTTP implementation of [`MemoryCompletion`].
///
/// Speaks the `/chat/completions` wire format: `DeepSeek`, vLLM, and any other
/// compatible server work by pointing `base_url` at that server. The bearer
/// token is only ever placed in the `Authorization` header — never in an error
/// message, a log line, or [`fmt::Debug`] output.
#[derive(Clone)]
pub struct OpenAiChatCompletion {
    /// Shared HTTP connection pool; carries the configured timeout.
    http: reqwest::Client,
    /// The endpoint, model, and sampling configuration.
    config: ChatCompletionConfig,
}

impl fmt::Debug for OpenAiChatCompletion {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OpenAiChatCompletion")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl OpenAiChatCompletion {
    /// Builds a client for `config`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when `base_url` or `model` is blank (after
    /// trimming), and `Error::Config` when the HTTP client cannot be built.
    pub fn new(config: ChatCompletionConfig) -> Result<Self> {
        let mut config = config;
        config.base_url = config.base_url.trim().trim_end_matches('/').to_string();
        config.model = config.model.trim().to_string();
        if config.base_url.is_empty() {
            return Err(Error::validation(
                "chat completion base_url must not be empty",
            ));
        }
        if config.model.is_empty() {
            return Err(Error::validation("chat completion model must not be empty"));
        }
        let http = reqwest::Client::builder()
            .timeout(config.timeout)
            .build()
            .map_err(|error| {
                Error::config(format!(
                    "chat completion HTTP client could not be built: {error}"
                ))
            })?;
        Ok(Self { http, config })
    }

    /// The configuration this client uses.
    #[must_use]
    pub fn config(&self) -> &ChatCompletionConfig {
        &self.config
    }
}

#[async_trait]
impl MemoryCompletion for OpenAiChatCompletion {
    async fn complete(&self, system: &str, user: &str) -> Result<String> {
        let body = json!({
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        });
        // Sizes only: prompt and completion text never reach the log.
        tracing::debug!(
            target: "openjiuwen_extensions::context_evolver",
            model = %self.config.model,
            system_chars = system.chars().count(),
            user_chars = user.chars().count(),
            "sending chat completion request"
        );
        let mut request = self.http.post(self.config.endpoint()).json(&body);
        if let Some(api_key) = &self.config.api_key {
            request = request.bearer_auth(api_key);
        }
        let response = request
            .send()
            .await
            .map_err(|error| self.transport_error(&error))?;
        let status = response.status();
        if !status.is_success() {
            let body = response
                .text()
                .await
                .unwrap_or_else(|error| format!("<body could not be read: {error}>"));
            return Err(request_error(status.as_u16(), &body));
        }
        let payload: Value = response.json().await.map_err(|error| {
            Error::transport(format!(
                "chat completion response from {CHAT_COMPLETIONS_PATH} is not JSON: {error}"
            ))
        })?;
        let content = payload
            .pointer("/choices/0/message/content")
            .ok_or_else(|| {
                Error::transport(format!(
                    "chat completion response from {CHAT_COMPLETIONS_PATH} has no \
                     `choices[0].message.content`"
                ))
            })?;
        let reply = reply_text(content)?;
        tracing::debug!(
            target: "openjiuwen_extensions::context_evolver",
            status = status.as_u16(),
            reply_chars = reply.chars().count(),
            "chat completion reply received"
        );
        Ok(reply)
    }
}

impl OpenAiChatCompletion {
    /// Maps a `reqwest` transport failure; timeouts become [`Error::timeout`].
    fn transport_error(&self, error: &reqwest::Error) -> Error {
        if error.is_timeout() {
            Error::timeout(format!(
                "chat completion request to {CHAT_COMPLETIONS_PATH} timed out after {:?}",
                self.config.timeout
            ))
        } else {
            Error::transport(format!(
                "chat completion request to {CHAT_COMPLETIONS_PATH} failed: {error}"
            ))
        }
    }
}

/// Maps a non-2xx status and its (previewed) body onto [`Error::transport`].
fn request_error(status: u16, body: &str) -> Error {
    let preview = preview(body, MAX_BODY_PREVIEW);
    let preview = if preview.is_empty() {
        "no response body".to_string()
    } else {
        preview
    };
    Error::transport(format!(
        "chat completion request to {CHAT_COMPLETIONS_PATH} failed with status {status}: {preview}"
    ))
}

/// Reads `choices[0].message.content` as text.
///
/// Accepts the plain-string form and the content-parts form
/// (`[{"type": "text", "text": "..."}, ...]`), concatenating the `text` fields.
fn reply_text(content: &Value) -> Result<String> {
    match content {
        Value::String(text) => Ok(text.clone()),
        Value::Array(parts) => {
            let mut reply = String::new();
            for part in parts {
                let text = part.get("text").and_then(Value::as_str).ok_or_else(|| {
                    Error::transport(format!(
                        "chat completion response from {CHAT_COMPLETIONS_PATH} has a \
                         `choices[0].message.content` part without a string `text` field"
                    ))
                })?;
                reply.push_str(text);
            }
            Ok(reply)
        }
        _ => Err(Error::transport(format!(
            "chat completion response from {CHAT_COMPLETIONS_PATH} has a \
             `choices[0].message.content` that is neither a string nor an array of text parts"
        ))),
    }
}

/// Trims `value` and caps it at `max` characters, ending with `...` when cut.
fn preview(value: &str, max: usize) -> String {
    let text = value.trim();
    if text.chars().count() <= max {
        return text.to_string();
    }
    let mut truncated: String = text.chars().take(max.saturating_sub(3)).collect();
    truncated.push_str("...");
    truncated
}

/// One recorded [`MemoryCompletion::complete`] call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompletionCall {
    /// The system prompt the pipeline sent.
    pub system: String,
    /// The user prompt the pipeline sent.
    pub user: String,
}

/// The mutable state of a [`ScriptedCompletion`].
#[derive(Debug, Default)]
struct ScriptedState {
    /// Replies waiting to be handed out, in order.
    replies: VecDeque<String>,
    /// Every call received, in order.
    calls: Vec<CompletionCall>,
}

/// A deterministic [`MemoryCompletion`] test double.
///
/// Queued replies are returned in order; once the queue is drained every call
/// fails with [`Error::transport`] (`"scripted completion exhausted"`) so a test
/// that asks for more replies than it queued fails loudly. Calls are recorded so
/// a test can assert the prompts a pipeline built.
#[derive(Debug, Default)]
pub struct ScriptedCompletion {
    /// Queued replies plus the recorded calls.
    state: Mutex<ScriptedState>,
}

impl ScriptedCompletion {
    /// Builds a scripted backend answering with `replies` in order.
    pub fn new<I, S>(replies: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let replies = replies.into_iter().map(Into::into).collect();
        Self {
            state: Mutex::new(ScriptedState {
                replies,
                calls: Vec::new(),
            }),
        }
    }

    /// Builds a scripted backend with no queued replies.
    #[must_use]
    pub fn empty() -> Self {
        Self::default()
    }

    /// Appends one reply to the end of the queue.
    pub fn push(&self, reply: impl Into<String>) {
        self.state().replies.push_back(reply.into());
    }

    /// Every recorded call, in order.
    #[must_use]
    pub fn calls(&self) -> Vec<CompletionCall> {
        self.state().calls.clone()
    }

    /// How many calls have been received.
    #[must_use]
    pub fn call_count(&self) -> usize {
        self.state().calls.len()
    }

    /// How many queued replies are still waiting.
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.state().replies.len()
    }

    /// Locks the state, recovering from a poisoned mutex (never panics).
    fn state(&self) -> MutexGuard<'_, ScriptedState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

#[async_trait]
impl MemoryCompletion for ScriptedCompletion {
    async fn complete(&self, system: &str, user: &str) -> Result<String> {
        let mut state = self.state();
        state.calls.push(CompletionCall {
            system: system.to_string(),
            user: user.to_string(),
        });
        state
            .replies
            .pop_front()
            .ok_or_else(|| Error::transport("scripted completion exhausted"))
    }
}

/// Trims `text` and removes one surrounding ```` ``` ````/```` ```json ```` fence.
///
/// The text is returned unchanged (only trimmed) unless the opening fence is
/// followed by a newline and the trimmed text ends with a closing fence. This is
/// the exact shape every chat model produces when asked for JSON.
#[must_use]
pub fn strip_code_fence(text: &str) -> &str {
    let trimmed = text.trim();
    let Some(rest) = trimmed.strip_prefix("```") else {
        return trimmed;
    };
    // The opening fence is its own line: an info string (`json`) or nothing.
    let Some(newline) = rest.find('\n') else {
        return trimmed;
    };
    let info = rest[..newline].trim();
    if !info.is_empty() && !info.eq_ignore_ascii_case("json") {
        return trimmed;
    }
    let Some(body) = rest[newline + 1..].trim_end().strip_suffix("```") else {
        return trimmed;
    };
    body.trim()
}

/// Parses a model reply as JSON, tolerating the shapes models actually emit.
///
/// Order of attempts:
///
/// 1. strict [`serde_json::from_str`] on the trimmed text;
/// 2. the same after [`strip_code_fence`];
/// 3. the first balanced `{...}` or `[...]` run embedded in the text, tracking
///    string literals and escapes so braces inside strings do not unbalance it.
///
/// # Errors
///
/// Returns [`Error::Validation`] naming `phase` when every attempt fails:
/// `"context_evolver {phase} pipeline: model reply is not JSON: {preview}"`. The
/// preview is capped at 120 characters. Odd input (empty, unterminated fence,
/// truncated object) is a validation error, never a panic.
pub fn parse_json_reply(phase: &str, text: &str) -> Result<Value> {
    let trimmed = text.trim();
    if let Ok(value) = serde_json::from_str::<Value>(trimmed) {
        return Ok(value);
    }
    let stripped = strip_code_fence(text);
    if let Ok(value) = serde_json::from_str::<Value>(stripped) {
        return Ok(value);
    }
    if let Some(run) = balanced_json_run(stripped) {
        if let Ok(value) = serde_json::from_str::<Value>(run) {
            return Ok(value);
        }
    }
    Err(Error::validation(format!(
        "context_evolver {phase} pipeline: model reply is not JSON: {}",
        preview(trimmed, MAX_REPLY_PREVIEW)
    )))
}

/// Returns the first balanced, parseable `{...}` or `[...]` run in `text`.
fn balanced_json_run(text: &str) -> Option<&str> {
    let bytes = text.as_bytes();
    for (start, character) in text.char_indices() {
        if character != '{' && character != '[' {
            continue;
        }
        if let Some(end) = matching_close(bytes, start) {
            let candidate = &text[start..=end];
            if serde_json::from_str::<Value>(candidate).is_ok() {
                return Some(candidate);
            }
        }
    }
    None
}

/// The index of the bracket closing the one at `start`, respecting JSON strings.
fn matching_close(bytes: &[u8], start: usize) -> Option<usize> {
    let mut expected: Vec<u8> = Vec::new();
    let mut in_string = false;
    let mut escaped = false;
    for (index, byte) in bytes.iter().enumerate().skip(start) {
        if in_string {
            if escaped {
                escaped = false;
            } else if *byte == b'\\' {
                escaped = true;
            } else if *byte == b'"' {
                in_string = false;
            }
            continue;
        }
        match *byte {
            b'"' => in_string = true,
            b'{' => expected.push(b'}'),
            b'[' => expected.push(b']'),
            b'}' | b']' => {
                if expected.pop() != Some(*byte) {
                    return None;
                }
                if expected.is_empty() {
                    return Some(index);
                }
            }
            _ => {}
        }
    }
    None
}

#[cfg(test)]
mod tests {
    // Python counterpart: `service/task_memory_service.py` (`LLMWrapper`).
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use openjiuwen_common::error::ErrorKind;

    #[test]
    fn strict_json_is_parsed() {
        let value = parse_json_reply("retrieve", "{\"memories\": [{\"content\": \"x\"}]}").unwrap();
        assert_eq!(value["memories"][0]["content"], json!("x"));
    }

    #[test]
    fn json_fence_is_stripped_and_parsed() {
        let reply = "```json\n{\"memories\": []}\n```";
        assert_eq!(strip_code_fence(reply), "{\"memories\": []}");
        let value = parse_json_reply("retrieve", reply).unwrap();
        assert_eq!(value, json!({"memories": []}));
    }

    #[test]
    fn bare_fence_is_stripped_and_parsed() {
        let reply = "```\n{\"memories\": [1, 2]}\n```";
        assert_eq!(strip_code_fence(reply), "{\"memories\": [1, 2]}");
        let value = parse_json_reply("summarize", reply).unwrap();
        assert_eq!(value, json!({"memories": [1, 2]}));
    }

    #[test]
    fn json_embedded_in_prose_is_extracted() {
        let reply = "Here is the memory: {\"memories\": [{\"content\": \"a } b\"}]} — done.";
        let value = parse_json_reply("retrieve", reply).unwrap();
        assert_eq!(value["memories"][0]["content"], json!("a } b"));
    }

    #[test]
    fn fenced_json_with_trailing_prose_is_extracted() {
        let reply = "```json\n{\"memories\": []}\n```\nThat is all I found.";
        let value = parse_json_reply("retrieve", reply).unwrap();
        assert_eq!(value, json!({"memories": []}));
    }

    #[test]
    fn json_array_reply_is_parsed() {
        let value = parse_json_reply("summarize", "[\n  {\"content\": \"x\"}\n]").unwrap();
        assert_eq!(value[0]["content"], json!("x"));
        let embedded = parse_json_reply("summarize", "items: [{\"content\": \"y\"}]").unwrap();
        assert_eq!(embedded[0]["content"], json!("y"));
    }

    #[test]
    fn plain_text_is_a_validation_error_naming_the_phase() {
        let error = parse_json_reply("summarize", "I could not recall anything.").unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(
            error
                .message()
                .contains("context_evolver summarize pipeline: model reply is not JSON"),
            "{}",
            error.message()
        );
        assert!(error.message().contains("I could not recall"));
    }

    #[test]
    fn empty_reply_is_a_validation_error() {
        for reply in ["", "   ", "\n\t"] {
            let error = parse_json_reply("retrieve", reply).unwrap_err();
            assert_eq!(error.kind(), ErrorKind::Validation, "reply: {reply:?}");
            assert!(error
                .message()
                .contains("context_evolver retrieve pipeline"));
        }
    }

    #[test]
    fn unterminated_fence_is_a_validation_error() {
        for reply in ["```json\n{\"memories\": [", "```json\n", "```"] {
            let error = parse_json_reply("retrieve", reply).unwrap_err();
            assert_eq!(error.kind(), ErrorKind::Validation, "reply: {reply:?}");
        }
        // The fence itself is left alone when it has no closing half.
        assert_eq!(
            strip_code_fence("```json\n{\"a\": 1}"),
            "```json\n{\"a\": 1}"
        );
    }

    #[test]
    fn long_replies_are_previewed_in_the_error() {
        let reply = "not json ".repeat(40);
        let error = parse_json_reply("retrieve", &reply).unwrap_err();
        let message = error.message().to_string();
        assert!(message.ends_with("..."), "{message}");
        let preview_length = message.rsplit(": ").next().unwrap().chars().count();
        assert!(preview_length <= MAX_REPLY_PREVIEW, "{preview_length}");
    }

    #[test]
    fn strip_code_fence_leaves_plain_text_alone() {
        assert_eq!(strip_code_fence("hello"), "hello");
        assert_eq!(strip_code_fence("  hello  "), "hello");
        assert_eq!(strip_code_fence(""), "");
        assert_eq!(strip_code_fence("```"), "```");
        assert_eq!(strip_code_fence("``````"), "``````");
        assert_eq!(
            strip_code_fence("```yaml\nkey: value\n```"),
            "```yaml\nkey: value\n```"
        );
    }

    #[test]
    fn strip_code_fence_accepts_uppercase_json_info_string() {
        assert_eq!(strip_code_fence("```JSON\n{}\n```"), "{}");
        assert_eq!(strip_code_fence("``` json \n{}\n```"), "{}");
    }

    #[test]
    fn strip_code_fence_trims_the_body_and_surrounding_whitespace() {
        assert_eq!(strip_code_fence("\n```json\n  {}\n```\n"), "{}");
    }

    #[tokio::test]
    async fn scripted_completion_returns_replies_in_order() {
        let completion = ScriptedCompletion::new(["first", "second"]);
        assert_eq!(completion.remaining(), 2);
        assert_eq!(completion.complete("sys", "one").await.unwrap(), "first");
        assert_eq!(completion.complete("sys", "two").await.unwrap(), "second");
        assert_eq!(completion.remaining(), 0);
        completion.push("third");
        assert_eq!(completion.complete("sys", "three").await.unwrap(), "third");
    }

    #[tokio::test]
    async fn scripted_completion_records_calls() {
        let completion = ScriptedCompletion::empty();
        completion.push("{}");
        completion
            .complete("system prompt", "user prompt")
            .await
            .unwrap();
        let error = completion.complete("", "").await.unwrap_err();
        assert_eq!(error.message().as_ref(), "scripted completion exhausted");
        assert_eq!(completion.call_count(), 2);
        assert_eq!(
            completion.calls(),
            vec![
                CompletionCall {
                    system: "system prompt".into(),
                    user: "user prompt".into(),
                },
                CompletionCall {
                    system: String::new(),
                    user: String::new(),
                },
            ]
        );
        assert_eq!(completion.remaining(), 0);
    }

    #[tokio::test]
    async fn scripted_completion_exhaustion_is_a_transport_error() {
        let completion = ScriptedCompletion::new(Vec::<String>::new());
        let error = completion.complete("sys", "user").await.unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Transport);
        assert_eq!(error.message().as_ref(), "scripted completion exhausted");
        // A failed call is still recorded, so ordering mistakes stay visible.
        assert_eq!(completion.call_count(), 1);
    }

    #[tokio::test]
    async fn scripted_completion_complete_json_parses_fenced_replies() {
        let completion = ScriptedCompletion::new(["```json\n{\"memories\": []}\n```"]);
        let value = completion
            .complete_json("retrieve", "system", "user")
            .await
            .unwrap();
        assert_eq!(value, json!({"memories": []}));

        let completion = ScriptedCompletion::new(["nope"]);
        let error = completion
            .complete_json("summarize", "system", "user")
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error
            .message()
            .contains("context_evolver summarize pipeline"));
    }

    #[test]
    fn config_from_lookup_prefers_openai_variables() {
        let config = ChatCompletionConfig::from_lookup(|name| match name {
            "OPENAI_BASE_URL" => Some("https://openai.test/v1".into()),
            "API_BASE" => Some("https://other.test/v1".into()),
            "OPENAI_API_KEY" => Some("sk-openai".into()),
            "API_KEY" => Some("sk-other".into()),
            "MODEL_NAME" => Some("gpt-test".into()),
            "OPENAI_MODEL" => Some("other-test".into()),
            "OPENJIUWEN_LLM_TIMEOUT_SECS" => Some("15".into()),
            _ => None,
        });
        assert_eq!(config.base_url(), "https://openai.test/v1");
        assert_eq!(config.api_key(), Some("sk-openai"));
        assert_eq!(config.model(), "gpt-test");
        assert_eq!(config.timeout(), Duration::from_secs(15));
        assert_eq!(config.temperature(), 0.7);
        assert_eq!(config.max_tokens(), 4096);
    }

    #[test]
    fn config_from_lookup_falls_back_to_api_base() {
        let config = ChatCompletionConfig::from_lookup(|name| match name {
            "API_BASE" => Some("https://fallback.test/v1".into()),
            "API_KEY" => Some("sk-fallback".into()),
            "OPENAI_MODEL" => Some("fallback-model".into()),
            _ => None,
        });
        assert_eq!(config.base_url(), "https://fallback.test/v1");
        assert_eq!(config.api_key(), Some("sk-fallback"));
        assert_eq!(config.model(), "fallback-model");
        assert_eq!(config.timeout(), Duration::from_secs(60));
    }

    #[test]
    fn config_from_lookup_defaults_when_nothing_is_set() {
        let config = ChatCompletionConfig::from_lookup(|_| None);
        assert_eq!(config.base_url(), DEFAULT_BASE_URL);
        assert_eq!(config.model(), DEFAULT_MODEL);
        assert_eq!(config.api_key(), None);
        assert_eq!(
            config.endpoint(),
            "https://api.openai.com/v1/chat/completions"
        );
    }

    #[test]
    fn config_from_lookup_ignores_blank_and_malformed_values() {
        let config = ChatCompletionConfig::from_lookup(|name| match name {
            "OPENAI_BASE_URL" => Some("   ".into()),
            "OPENAI_API_KEY" => Some(String::new()),
            "MODEL_NAME" => Some(" gpt-blank-test ".into()),
            "OPENJIUWEN_LLM_TIMEOUT_SECS" => Some("soon".into()),
            _ => None,
        });
        assert_eq!(config.base_url(), DEFAULT_BASE_URL);
        assert_eq!(config.api_key(), None);
        assert_eq!(config.model(), "gpt-blank-test");
        assert_eq!(config.timeout(), Duration::from_secs(60));

        let zero = ChatCompletionConfig::from_lookup(|name| match name {
            "OPENJIUWEN_LLM_TIMEOUT_SECS" => Some("0".into()),
            _ => None,
        });
        assert_eq!(zero.timeout(), Duration::from_secs(60));
    }

    #[test]
    fn config_new_trims_and_defaults() {
        let config = ChatCompletionConfig::new(" https://api.test/v1// ", " model ");
        assert_eq!(config.base_url(), "https://api.test/v1");
        assert_eq!(config.model(), "model");
        assert_eq!(config.temperature(), 0.7);
        assert_eq!(config.max_tokens(), 4096);
        assert_eq!(config.timeout(), Duration::from_secs(60));
        assert_eq!(config.api_key(), None);
        assert_eq!(config.endpoint(), "https://api.test/v1/chat/completions");

        let configured = ChatCompletionConfig::new("https://api.test/v1", "m")
            .with_temperature(0.1)
            .with_max_tokens(32)
            .with_timeout(Duration::from_secs(5))
            .with_api_key("sk-key");
        assert_eq!(configured.temperature(), 0.1);
        assert_eq!(configured.max_tokens(), 32);
        assert_eq!(configured.timeout(), Duration::from_secs(5));
        assert_eq!(configured.api_key(), Some("sk-key"));
    }

    #[test]
    fn config_debug_redacts_the_api_key() {
        let config = ChatCompletionConfig::new("https://api.test/v1", "model")
            .with_api_key("sk-super-secret");
        let debug = format!("{config:?}");
        assert!(!debug.contains("sk-super-secret"), "{debug}");
        assert!(debug.contains(REDACTED), "{debug}");
        assert!(debug.contains("model"), "{debug}");

        let without_key = ChatCompletionConfig::new("https://api.test/v1", "model");
        assert!(format!("{without_key:?}").contains("api_key: None"));
    }

    #[test]
    fn openai_client_rejects_blank_endpoint_configuration() {
        let error = OpenAiChatCompletion::new(ChatCompletionConfig::new("  ", "model"))
            .expect_err("a blank base URL is invalid");
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.message().contains("base_url"));

        let error =
            OpenAiChatCompletion::new(ChatCompletionConfig::new("https://api.test/v1", "   "))
                .expect_err("a blank model is invalid");
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.message().contains("model"));
    }

    #[test]
    fn openai_client_debug_hides_the_key_and_keeps_the_endpoint() {
        let client = OpenAiChatCompletion::new(
            ChatCompletionConfig::new("https://api.test/v1/", "gpt-test").with_api_key("sk-secret"),
        )
        .unwrap();
        assert_eq!(
            client.config().endpoint(),
            "https://api.test/v1/chat/completions"
        );
        assert!(!format!("{client:?}").contains("sk-secret"));
    }

    #[test]
    fn preview_caps_on_character_boundaries() {
        assert_eq!(preview("  short  ", MAX_BODY_PREVIEW), "short");
        let long = "é".repeat(300);
        let previewed = preview(&long, MAX_BODY_PREVIEW);
        assert_eq!(previewed.chars().count(), MAX_BODY_PREVIEW);
        assert!(previewed.ends_with("..."));
    }

    #[test]
    fn request_error_previews_the_body() {
        let body = "x".repeat(500);
        let error = request_error(429, &body);
        assert_eq!(error.kind(), ErrorKind::Transport);
        let message = error.message().to_string();
        assert!(message.contains("status 429"), "{message}");
        assert!(message.contains(CHAT_COMPLETIONS_PATH), "{message}");
        let previewed = message.rsplit(": ").next().unwrap();
        assert_eq!(previewed.chars().count(), MAX_BODY_PREVIEW);

        let empty = request_error(500, "   ");
        assert!(empty.message().contains("no response body"));
    }
}
