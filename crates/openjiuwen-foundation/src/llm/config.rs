//! Serializable model configuration: client transport settings and per-request knobs.
//!
//! Ports `ModelClientConfig` and `ModelRequestConfig` from
//! `openjiuwen/core/foundation/llm/schema/config.py`. Both types round-trip
//! through JSON, tolerate unknown fields, and accept the Python field spellings
//! (`client_provider`, `model`) as aliases so a serialized Python config can be
//! read verbatim.
//!
//! Python's shapes that serde would otherwise reject are decoded by the
//! `decode` submodule, which every affected field names in its
//! `deserialize_with` attribute:
//!
//! - the three timeouts are `float` seconds (`360.0`), a numeric string, or the
//!   `{"secs", "nanos"}` object Rust used to write, and serialize as a number of
//!   seconds;
//! - `stop` is one optional string (or an array) and always serializes as an
//!   array;
//! - `custom_headers`/`headers` values may be strings, numbers, or booleans, and
//!   `null` values are dropped.
//!
//! [`ModelClientConfig::resolve_for`] is the single place that turns a partially
//! specified config into a usable one: it normalizes the provider name, applies
//! the environment fallbacks (`API_KEY`/`OPENAI_API_KEY`/`OPENJIUWEN_API_KEY`,
//! `API_BASE`/`OPENAI_BASE_URL`), falls back to the provider table's default base
//! URL, and validates that the result can actually authenticate. Credentials are
//! held in [`SecretString`], so neither `Debug` nor any error message can print
//! them.
//!
//! ```
//! use std::time::Duration;
//! use openjiuwen_foundation::llm::config::{ModelClientConfig, ModelRequestConfig};
//!
//! let client = ModelClientConfig::new("OpenAI")
//!     .with_api_key("sk-test")
//!     .with_timeout(Duration::from_secs(30));
//! let resolved = client.resolve_for("OpenAI").expect("openai is built in");
//! assert_eq!(resolved.provider, "openai");
//! assert_eq!(resolved.api_base.as_deref(), Some("https://api.openai.com/v1"));
//!
//! let request = ModelRequestConfig::default().with_model_name("gpt-4o");
//! assert_eq!(request.model_name, "gpt-4o");
//! ```

use std::collections::BTreeMap;
use std::time::Duration;

use openjiuwen_common::config::env;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::http::HttpClientConfig;
use openjiuwen_common::security::{sanitize_headers, SecretString};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::provider::{normalize_provider_name, provider_names, provider_profile};

mod decode;

/// Default total request timeout: 360 s (Python `ModelClientConfig.timeout`).
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(360);

/// Default wait for the first streaming chunk: 300 s (Python parity).
const DEFAULT_STREAM_FIRST_CHUNK_TIMEOUT: Duration = Duration::from_secs(300);

/// Default maximum gap between streaming chunks: 60 s (Python parity).
const DEFAULT_STREAM_IDLE_TIMEOUT: Duration = Duration::from_secs(60);

/// Providers served from a local machine, which need no API key.
const KEYLESS_PROVIDERS: &[&str] = &["ollama", "lmstudio"];

/// Returns `true` when `provider` cannot authenticate with an API key.
fn is_keyless(provider: &str) -> bool {
    KEYLESS_PROVIDERS.contains(&provider)
}

/// Returns `true` when a credential is absent or blank.
fn secret_is_missing(secret: Option<&SecretString>) -> bool {
    secret.is_none_or(|secret| secret.expose_secret().trim().is_empty())
}

/// Wire protocol an `OpenAI`-family endpoint speaks.
///
/// Ports the Python `LLMApiMode` enum: `chat_completions` (the default for
/// `OpenAI`-compatible vendors), `responses` (the `OpenAI` Responses API), and
/// `anthropic_messages` (the Anthropic Messages API).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ApiMode {
    /// `POST /chat/completions`, the OpenAI-compatible default.
    ChatCompletions,
    /// The `OpenAI` Responses API, `POST /responses`.
    Responses,
    /// The Anthropic Messages API, `POST /v1/messages`.
    AnthropicMessages,
}

/// How a model client authenticates.
///
/// Ports the Python `LLMAuthMode` enum. Only [`AuthMode::ApiKey`] requires an API
/// key: the other modes authenticate out of band (an OAuth token store, caller
/// supplied headers) or not at all, so [`ModelClientConfig::resolve_for`] does not
/// demand a credential for them.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuthMode {
    /// A bearer API key (`api_key` or the environment), the default.
    #[default]
    ApiKey,
    /// No authentication at all.
    #[serde(rename = "none")]
    NoneAuth,
    /// The caller supplies authentication through [`ModelClientConfig::headers`].
    CustomHeaders,
    /// An `OpenAI` account OAuth access token resolved by `llm::oauth`.
    #[serde(rename = "openai_account_oauth")]
    OpenAiAccountOAuth,
}

/// Client-side configuration of one model connection.
///
/// The type mirrors the Python `ModelClientConfig`: an identity ([`client_id`](Self::client_id)),
/// the provider to talk to, credentials, endpoint, protocol/auth modes,
/// TLS/timeout/retry policy, extra headers, and an `extra` map for vendor-specific
/// settings that the struct does not model.
///
/// `Debug` redacts the API key through [`SecretString`], and JSON serialization
/// keeps it so a configuration round-trips. The Python `custom_headers` field is
/// accepted as an alias of [`headers`](Self::headers).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ModelClientConfig {
    /// Stable client identifier used when the client is registered with a runner.
    pub client_id: String,
    /// Provider name, canonical or vendor spelling; [`resolve_for`](Self::resolve_for) normalizes it.
    #[serde(alias = "client_provider")]
    pub provider: String,
    /// API key; `None` means "resolve from the environment".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key: Option<SecretString>,
    /// Base URL of the provider; `None` means "use the provider default".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_base: Option<String>,
    /// Whether TLS certificates are verified.
    pub verify_ssl: bool,
    /// Total timeout of one request.
    ///
    /// Decoded from Python's `float` seconds (`360.0`), a numeric string, or the
    /// `{"secs", "nanos"}` object; serialized as a number of seconds. `null`
    /// disables the per-request timeout.
    #[serde(
        deserialize_with = "decode::deserialize_duration",
        serialize_with = "decode::serialize_duration"
    )]
    pub timeout: Option<Duration>,
    /// Maximum wait for the first streaming chunk that carries model output.
    ///
    /// Decoded and serialized exactly like [`timeout`](Self::timeout).
    #[serde(
        deserialize_with = "decode::deserialize_duration",
        serialize_with = "decode::serialize_duration"
    )]
    pub stream_first_chunk_timeout: Option<Duration>,
    /// Maximum wait between streaming chunks that carry model output.
    ///
    /// Decoded and serialized exactly like [`timeout`](Self::timeout).
    #[serde(
        deserialize_with = "decode::deserialize_duration",
        serialize_with = "decode::serialize_duration"
    )]
    pub stream_idle_timeout: Option<Duration>,
    /// Number of additional attempts after the first one; `0` disables retries.
    pub max_retries: u32,
    /// Extra request headers, merged into every call.
    ///
    /// The Python field name `custom_headers` is accepted as an alias. Python
    /// types the map `dict[str, Any]`, so numbers and booleans are coerced with
    /// `to_string()` and `null` values are dropped; serialization always writes a
    /// string map.
    #[serde(
        alias = "custom_headers",
        deserialize_with = "decode::deserialize_headers"
    )]
    pub headers: BTreeMap<String, String>,
    /// Wire protocol to speak, or `None` for the provider's default.
    ///
    /// A provider that normally speaks chat completions uses the Responses API
    /// when this is [`ApiMode::Responses`].
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_mode: Option<ApiMode>,
    /// Authentication scheme; defaults to [`AuthMode::ApiKey`].
    pub auth_mode: AuthMode,
    /// Provider endpoint dialect name (Python `endpoint_profile`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub endpoint_profile: Option<String>,
    /// Vendor-specific settings the struct does not model.
    pub extra: Value,
}

impl Default for ModelClientConfig {
    /// A fresh UUID v4 `client_id`, OpenAI-compatible transport defaults, and no
    /// provider, credential, endpoint, or headers.
    fn default() -> Self {
        Self {
            client_id: uuid::Uuid::new_v4().to_string(),
            provider: String::new(),
            api_key: None,
            api_base: None,
            verify_ssl: true,
            timeout: Some(DEFAULT_TIMEOUT),
            stream_first_chunk_timeout: Some(DEFAULT_STREAM_FIRST_CHUNK_TIMEOUT),
            stream_idle_timeout: Some(DEFAULT_STREAM_IDLE_TIMEOUT),
            max_retries: 0,
            headers: BTreeMap::new(),
            api_mode: None,
            auth_mode: AuthMode::ApiKey,
            endpoint_profile: None,
            extra: Value::Null,
        }
    }
}

impl ModelClientConfig {
    /// Creates a config for `provider` with every other field at its default.
    #[must_use]
    pub fn new(provider: impl Into<String>) -> Self {
        Self {
            provider: provider.into(),
            ..Self::default()
        }
    }

    /// Overrides [`client_id`](Self::client_id).
    #[must_use]
    pub fn with_client_id(mut self, client_id: impl Into<String>) -> Self {
        self.client_id = client_id.into();
        self
    }

    /// Sets the API key from a plain string.
    #[must_use]
    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(SecretString::new(api_key));
        self
    }

    /// Sets the API key from an already-wrapped [`SecretString`].
    #[must_use]
    pub fn with_secret(mut self, secret: SecretString) -> Self {
        self.api_key = Some(secret);
        self
    }

    /// Overrides [`api_base`](Self::api_base).
    #[must_use]
    pub fn with_api_base(mut self, api_base: impl Into<String>) -> Self {
        self.api_base = Some(api_base.into());
        self
    }

    /// Overrides the total request timeout.
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Overrides both streaming timeouts; `None` disables the corresponding check.
    #[must_use]
    pub fn with_stream_timeouts(
        mut self,
        first_chunk: Option<Duration>,
        idle: Option<Duration>,
    ) -> Self {
        self.stream_first_chunk_timeout = first_chunk;
        self.stream_idle_timeout = idle;
        self
    }

    /// Overrides the number of retries after the initial attempt.
    #[must_use]
    pub fn with_max_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = max_retries;
        self
    }

    /// Adds one extra request header.
    #[must_use]
    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.insert(name.into(), value.into());
        self
    }

    /// Adds several extra request headers.
    #[must_use]
    pub fn with_headers(mut self, headers: impl IntoIterator<Item = (String, String)>) -> Self {
        self.headers.extend(headers);
        self
    }

    /// Enables or disables TLS certificate verification.
    #[must_use]
    pub fn with_verify_ssl(mut self, verify_ssl: bool) -> Self {
        self.verify_ssl = verify_ssl;
        self
    }

    /// Overrides the wire protocol.
    #[must_use]
    pub fn with_api_mode(mut self, api_mode: ApiMode) -> Self {
        self.api_mode = Some(api_mode);
        self
    }

    /// Overrides the authentication scheme.
    #[must_use]
    pub fn with_auth_mode(mut self, auth_mode: AuthMode) -> Self {
        self.auth_mode = auth_mode;
        self
    }

    /// Overrides the provider endpoint dialect name.
    #[must_use]
    pub fn with_endpoint_profile(mut self, endpoint_profile: impl Into<String>) -> Self {
        self.endpoint_profile = Some(endpoint_profile.into());
        self
    }

    /// Replaces the vendor-specific `extra` map.
    #[must_use]
    pub fn with_extra(mut self, extra: Value) -> Self {
        self.extra = extra;
        self
    }

    /// Builds a config from the environment.
    ///
    /// Reads `MODEL_PROVIDER` (default `"openai"`), the API key
    /// (`OPENJIUWEN_API_KEY` → `API_KEY` → `OPENAI_API_KEY`), and the base URL
    /// (`API_BASE` → `OPENAI_BASE_URL`); every other field keeps its default.
    #[must_use]
    pub fn from_env() -> Self {
        Self {
            provider: env::model_provider(),
            api_key: env::api_key(),
            api_base: env::api_base(),
            ..Self::default()
        }
    }

    /// Resolves `provider` into a ready-to-use configuration.
    ///
    /// The provider name is normalized to its canonical spelling and looked up in
    /// the built-in table. Then, in order:
    ///
    /// - `api_key`: an explicit non-empty value wins, otherwise `env::api_key()`.
    /// - `api_base`: an explicit non-empty value wins, otherwise `env::api_base()`,
    ///   otherwise the provider default (for example
    ///   `https://api.openai.com/v1` for `openai`).
    /// - Headers: the provider's static headers are added only when the caller did
    ///   not already set a header of the same name (case-insensitive).
    /// - Validation: a provider without a built-in base URL (`pangu`) must receive
    ///   one, and every provider except `ollama`/`lmstudio` must receive a key when
    ///   [`auth_mode`](Self::auth_mode) is [`AuthMode::ApiKey`]. The modes that
    ///   authenticate out of band (OAuth, caller headers, none) are exempt, matching
    ///   the Python config validator.
    ///
    /// TLS settings, timeouts, retries, `extra`, and `client_id` pass through
    /// unchanged.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Unsupported`] for a provider that is not built in, and
    /// [`Error::Config`] when no API base or no API key can be determined. The
    /// messages name the environment variable to set and never contain the key.
    pub fn resolve_for(&self, provider: &str) -> Result<Self> {
        let canonical = normalize_provider_name(provider);
        let profile = provider_profile(&canonical).ok_or_else(|| {
            Error::unsupported(format!(
                "unsupported model provider `{provider}`; known providers: {}",
                provider_names().join(", ")
            ))
        })?;

        let mut resolved = self.clone();
        resolved.provider = canonical;

        if secret_is_missing(resolved.api_key.as_ref()) {
            resolved.api_key = env::api_key();
        }

        let explicit_base = resolved
            .api_base
            .as_deref()
            .map(str::trim)
            .filter(|base| !base.is_empty())
            .map(str::to_owned);
        resolved.api_base = explicit_base
            .or_else(env::api_base)
            .or_else(|| profile.default_api_base.map(str::to_owned));

        for (name, value) in profile.extra_headers {
            if !resolved
                .headers
                .keys()
                .any(|existing| existing.eq_ignore_ascii_case(name))
            {
                resolved
                    .headers
                    .insert((*name).to_owned(), (*value).to_owned());
            }
        }

        if resolved.api_base.is_none() {
            return Err(Error::config(format!(
                "provider `{}` has no built-in API base; set the {} environment variable \
                 (or `api_base`) to the deployment endpoint",
                resolved.provider,
                env::API_BASE
            )));
        }

        if secret_is_missing(resolved.api_key.as_ref())
            && resolved.auth_mode == AuthMode::ApiKey
            && !is_keyless(&resolved.provider)
        {
            return Err(Error::config(format!(
                "provider `{}` requires an API key; set {}, {}, or {} (or `api_key`)",
                resolved.provider,
                env::OPENJIUWEN_API_KEY,
                env::API_KEY,
                env::OPENAI_API_KEY
            )));
        }

        Ok(resolved)
    }

    /// The caller's headers with protected names and blank values removed.
    ///
    /// Delegates to [`openjiuwen_common::security::sanitize_headers`], which drops
    /// `host`, `content-length`, `transfer-encoding`, `connection`, and
    /// `authorization`, so a caller cannot smuggle a transport-controlled header
    /// onto the wire.
    #[must_use]
    pub fn sanitized_headers(&self) -> BTreeMap<String, String> {
        sanitize_headers(&self.headers)
    }

    /// Maps this configuration onto the shared HTTP client's settings.
    ///
    /// The timeout falls back to 360 s when unset; headers are passed through
    /// [`sanitized_headers`](Self::sanitized_headers).
    #[must_use]
    pub fn http_client_config(&self) -> HttpClientConfig {
        HttpClientConfig::default()
            .with_timeout(self.timeout.unwrap_or(DEFAULT_TIMEOUT))
            .with_max_retries(self.max_retries)
            .with_verify_ssl(self.verify_ssl)
            .with_headers(self.sanitized_headers())
    }
}

/// Per-request model settings.
///
/// Mirrors the Python `ModelRequestConfig`. Every field defaults, so a partial
/// JSON object (or [`Default::default`]) is a valid request; the Python name
/// `model` is accepted as an alias of [`model_name`](Self::model_name).
///
/// ```
/// use openjiuwen_foundation::llm::config::ModelRequestConfig;
/// use serde_json::json;
///
/// let request = ModelRequestConfig::default()
///     .with_model_name("gpt-4o")
///     .with_temperature(0.2)
///     .with_max_tokens(1024);
/// assert_eq!(request.max_tokens, Some(1024));
///
/// let parsed: ModelRequestConfig =
///     serde_json::from_value(json!({"model": "gpt-4o-mini"})).expect("alias deserializes");
/// assert_eq!(parsed.model_name, "gpt-4o-mini");
///
/// // Python's single-string `stop` and typed reasoning intent decode too.
/// let ported: ModelRequestConfig = serde_json::from_value(json!({
///     "model": "gpt-4o-mini",
///     "stop": "\n",
///     "reasoning": {"mode": "enabled", "effort": "low"},
///     "context_window": 128000,
/// }))
/// .expect("a Python config deserializes");
/// assert_eq!(ported.stop, vec!["\n".to_owned()]);
/// assert_eq!(ported.context_window, Some(128_000));
/// assert!(ported.reasoning.is_some());
/// ```
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ModelRequestConfig {
    /// Provider-facing model name.
    #[serde(alias = "model")]
    pub model_name: String,
    /// Sampling temperature.
    pub temperature: Option<f32>,
    /// Nucleus-sampling probability mass.
    pub top_p: Option<f32>,
    /// Upper bound on generated tokens.
    pub max_tokens: Option<u32>,
    /// Sequences that stop generation.
    ///
    /// Decoded from Python's single string, `null`, or an array of strings and
    /// always serialized as an array.
    #[serde(deserialize_with = "decode::deserialize_stop")]
    pub stop: Vec<String>,
    /// Maximum context window the model supports, in tokens.
    ///
    /// Python's `context_window`: internal context metadata that is decoded and
    /// serialized but never written to a provider request body.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_window: Option<u32>,
    /// Provider-neutral reasoning intent, in the shape
    /// [`ReasoningConfig::from_value`](super::reasoning::ReasoningConfig::from_value)
    /// reads (a boolean, a mode/effort string, or an object with `mode`,
    /// `effort` and/or `budget_tokens`).
    ///
    /// Python's `reasoning` field. The three request builders read it first and
    /// fall back to the legacy `extra["reasoning"]` spelling; an intent is
    /// consumed there, so it never reaches a provider body as a raw key.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<Value>,
    /// Tool definitions in provider wire shape.
    pub tools: Option<Value>,
    /// Tool-selection policy in provider wire shape.
    pub tool_choice: Option<Value>,
    /// Whether the client should stream the response.
    pub stream: bool,
    /// Vendor-specific request settings merged into the body last.
    pub extra: Value,
}

impl ModelRequestConfig {
    /// Creates a request for `model_name` with every other field at its default.
    #[must_use]
    pub fn new(model_name: impl Into<String>) -> Self {
        Self {
            model_name: model_name.into(),
            ..Self::default()
        }
    }

    /// Overrides [`model_name`](Self::model_name).
    #[must_use]
    pub fn with_model_name(mut self, model_name: impl Into<String>) -> Self {
        self.model_name = model_name.into();
        self
    }

    /// Overrides the sampling temperature.
    #[must_use]
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Overrides the nucleus-sampling probability mass.
    #[must_use]
    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p);
        self
    }

    /// Overrides the generated-token limit.
    #[must_use]
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Replaces the tool definitions; an empty list clears the field.
    #[must_use]
    pub fn with_tools(mut self, tools: Vec<Value>) -> Self {
        self.tools = if tools.is_empty() {
            None
        } else {
            Some(Value::Array(tools))
        };
        self
    }

    /// Replaces the tool-selection policy.
    #[must_use]
    pub fn with_tool_choice(mut self, tool_choice: Value) -> Self {
        self.tool_choice = Some(tool_choice);
        self
    }

    /// Replaces the stop sequences.
    #[must_use]
    pub fn with_stop(mut self, stop: impl IntoIterator<Item = String>) -> Self {
        self.stop = stop.into_iter().collect();
        self
    }

    /// Enables or disables streaming.
    #[must_use]
    pub fn with_stream(mut self, stream: bool) -> Self {
        self.stream = stream;
        self
    }

    /// Overrides the model's context-window metadata.
    #[must_use]
    pub fn with_context_window(mut self, context_window: u32) -> Self {
        self.context_window = Some(context_window);
        self
    }

    /// Replaces the provider-neutral reasoning intent.
    #[must_use]
    pub fn with_reasoning(mut self, reasoning: Value) -> Self {
        self.reasoning = Some(reasoning);
        self
    }

    /// Replaces the vendor-specific `extra` map.
    #[must_use]
    pub fn with_extra(mut self, extra: Value) -> Self {
        self.extra = extra;
        self
    }

    /// Builds a request from the environment, reading only `MODEL_NAME`.
    ///
    /// The model name is empty when the variable is unset; every provider client
    /// reports a missing model name in its own validation.
    #[must_use]
    pub fn from_env() -> Self {
        Self {
            model_name: env::model_name().unwrap_or_default(),
            ..Self::default()
        }
    }
}

#[cfg(test)]
mod tests;
