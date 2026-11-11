//! Serializable model configuration: client transport settings and per-request knobs.
//!
//! Ports `ModelClientConfig` and `ModelRequestConfig` from
//! `openjiuwen/core/foundation/llm/schema/config.py`. Both types round-trip
//! through JSON, tolerate unknown fields, and accept the Python field spellings
//! (`client_provider`, `model`) as aliases so a serialized Python config can be
//! read verbatim.
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

/// Client-side configuration of one model connection.
///
/// The type mirrors the Python `ModelClientConfig`: an identity ([`client_id`](Self::client_id)),
/// the provider to talk to, credentials, endpoint, TLS/timeout/retry policy,
/// extra headers, and an `extra` map for vendor-specific settings that the struct
/// does not model.
///
/// `Debug` redacts the API key through [`SecretString`], and JSON serialization
/// keeps it so a configuration round-trips.
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
    pub timeout: Option<Duration>,
    /// Maximum wait for the first streaming chunk that carries model output.
    pub stream_first_chunk_timeout: Option<Duration>,
    /// Maximum wait between streaming chunks that carry model output.
    pub stream_idle_timeout: Option<Duration>,
    /// Number of additional attempts after the first one; `0` disables retries.
    pub max_retries: u32,
    /// Extra request headers, merged into every call.
    pub headers: BTreeMap<String, String>,
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
    ///   one, and every provider except `ollama`/`lmstudio` must receive a key.
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

        if secret_is_missing(resolved.api_key.as_ref()) && !is_keyless(&resolved.provider) {
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
    pub stop: Vec<String>,
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
mod tests {
    use super::*;
    use openjiuwen_common::error::ErrorKind;
    use serde_json::json;
    use std::sync::{Mutex, MutexGuard, OnceLock};

    /// Every variable the resolvers in this module may read.
    const ENV_VARS: &[&str] = &[
        env::OPENJIUWEN_API_KEY,
        env::API_KEY,
        env::OPENAI_API_KEY,
        env::API_BASE,
        env::OPENAI_BASE_URL,
        env::MODEL_PROVIDER,
        env::MODEL_NAME,
    ];

    /// Serializes tests that mutate the process environment.
    fn env_lock() -> MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Clears every variable this module reads and restores it on drop.
    struct EnvGuard {
        _lock: MutexGuard<'static, ()>,
        saved: Vec<(&'static str, Option<String>)>,
    }

    impl EnvGuard {
        fn clear() -> Self {
            let lock = env_lock();
            let saved: Vec<(&'static str, Option<String>)> = ENV_VARS
                .iter()
                .map(|name| (*name, std::env::var(name).ok()))
                .collect();
            for name in ENV_VARS {
                std::env::remove_var(name);
            }
            Self { _lock: lock, saved }
        }

        #[allow(clippy::unused_self)]
        fn set(&self, name: &str, value: &str) {
            std::env::set_var(name, value);
        }

        #[allow(clippy::unused_self)]
        fn remove(&self, name: &str) {
            std::env::remove_var(name);
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            for (name, value) in &self.saved {
                match value {
                    Some(value) => std::env::set_var(name, value),
                    None => std::env::remove_var(name),
                }
            }
        }
    }

    #[test]
    fn defaults_match_the_documented_values() {
        let config = ModelClientConfig::default();

        assert!(uuid::Uuid::parse_str(&config.client_id).is_ok());
        assert_eq!(config.provider, "");
        assert_eq!(config.api_key, None);
        assert_eq!(config.api_base, None);
        assert!(config.verify_ssl);
        assert_eq!(config.timeout, Some(Duration::from_secs(360)));
        assert_eq!(
            config.stream_first_chunk_timeout,
            Some(Duration::from_secs(300))
        );
        assert_eq!(config.stream_idle_timeout, Some(Duration::from_secs(60)));
        assert_eq!(config.max_retries, 0);
        assert!(config.headers.is_empty());
        assert_eq!(config.extra, Value::Null);
    }

    #[test]
    fn client_ids_are_unique_per_default() {
        assert_ne!(
            ModelClientConfig::default().client_id,
            ModelClientConfig::default().client_id
        );
    }

    #[test]
    fn json_round_trip_preserves_every_field() {
        let config = ModelClientConfig::new("anthropic")
            .with_client_id("client-1")
            .with_api_key("sk-ant-test")
            .with_api_base("https://proxy.example/v1")
            .with_timeout(Duration::from_secs(12))
            .with_stream_timeouts(Some(Duration::from_secs(7)), None)
            .with_max_retries(4)
            .with_header("X-Org", "research")
            .with_verify_ssl(false)
            .with_extra(json!({"api_mode": "chat_completions"}));

        let encoded = serde_json::to_string(&config).expect("serializes");
        assert!(encoded.contains("\"provider\":\"anthropic\""));

        let restored: ModelClientConfig = serde_json::from_str(&encoded).expect("round-trips");
        assert_eq!(restored, config);
        assert_eq!(
            restored.api_key.as_ref().map(SecretString::expose_secret),
            Some("sk-ant-test")
        );
    }

    #[test]
    fn deserialization_accepts_python_aliases_and_unknown_fields() {
        let config: ModelClientConfig = serde_json::from_value(json!({
            "client_provider": "OpenAI",
            "api_key": "sk-python",
            "timeout": {"secs": 9, "nanos": 0},
            "custom_headers": {"ignored": "value"},
        }))
        .expect("python config deserializes");

        assert_eq!(config.provider, "OpenAI");
        assert_eq!(config.timeout, Some(Duration::from_secs(9)));
        assert!(uuid::Uuid::parse_str(&config.client_id).is_ok());

        let request: ModelRequestConfig =
            serde_json::from_value(json!({"model": "gpt-4o-mini", "unknown": true}))
                .expect("model alias deserializes");
        assert_eq!(request.model_name, "gpt-4o-mini");
        assert!(!request.stream);
        assert_eq!(request.stop, Vec::<String>::new());
    }

    #[test]
    fn builders_set_every_field() {
        let secret = SecretString::new("sk-secret");
        let config = ModelClientConfig::new("qwen")
            .with_client_id("c-2")
            .with_secret(secret.clone())
            .with_api_base("https://example.test/v1")
            .with_timeout(Duration::from_secs(5))
            .with_stream_timeouts(Some(Duration::from_secs(2)), Some(Duration::from_secs(1)))
            .with_max_retries(2)
            .with_headers(vec![("X-A".to_owned(), "1".to_owned())])
            .with_header("X-B", "2")
            .with_verify_ssl(false)
            .with_extra(json!({"request_extra_body": {"seed": 1}}));

        assert_eq!(config.client_id, "c-2");
        assert_eq!(config.provider, "qwen");
        assert_eq!(config.api_key, Some(secret));
        assert_eq!(config.api_base.as_deref(), Some("https://example.test/v1"));
        assert_eq!(config.timeout, Some(Duration::from_secs(5)));
        assert_eq!(
            config.stream_first_chunk_timeout,
            Some(Duration::from_secs(2))
        );
        assert_eq!(config.stream_idle_timeout, Some(Duration::from_secs(1)));
        assert_eq!(config.max_retries, 2);
        assert_eq!(config.headers.len(), 2);
        assert_eq!(config.headers.get("X-B").map(String::as_str), Some("2"));
        assert!(!config.verify_ssl);
        assert_eq!(config.extra, json!({"request_extra_body": {"seed": 1}}));
    }

    #[test]
    fn request_builders_set_every_field() {
        let request = ModelRequestConfig::new("gpt-4o")
            .with_model_name("gpt-4o-2024")
            .with_temperature(0.1)
            .with_top_p(0.9)
            .with_max_tokens(512)
            .with_tools(vec![json!({"type": "function"})])
            .with_tool_choice(json!("auto"))
            .with_stop(vec!["STOP".to_owned()])
            .with_stream(true)
            .with_extra(json!({"reasoning": {"effort": "low"}}));

        assert_eq!(request.model_name, "gpt-4o-2024");
        assert_eq!(request.temperature, Some(0.1));
        assert_eq!(request.top_p, Some(0.9));
        assert_eq!(request.max_tokens, Some(512));
        assert_eq!(request.tools, Some(json!([{"type": "function"}])));
        assert_eq!(request.tool_choice, Some(json!("auto")));
        assert_eq!(request.stop, vec!["STOP".to_owned()]);
        assert!(request.stream);
        assert_eq!(request.extra, json!({"reasoning": {"effort": "low"}}));
    }

    #[test]
    fn empty_tool_list_clears_the_field() {
        let request = ModelRequestConfig::default().with_tools(Vec::new());
        assert_eq!(request.tools, None);
    }

    #[test]
    fn http_client_config_maps_transport_settings() {
        let config = ModelClientConfig::new("openai")
            .with_timeout(Duration::from_secs(42))
            .with_max_retries(3)
            .with_verify_ssl(false)
            .with_header("X-Custom", "value")
            .with_header("Authorization", "Bearer leaked");

        let http = config.http_client_config();
        assert_eq!(http.timeout, Duration::from_secs(42));
        assert_eq!(http.max_retries, 3);
        assert!(!http.verify_ssl);
        assert_eq!(http.headers.len(), 1);
        assert_eq!(
            http.headers.get("X-Custom").map(String::as_str),
            Some("value")
        );
        assert!(!http.headers.contains_key("Authorization"));
    }

    #[test]
    fn http_client_config_falls_back_to_the_default_timeout() {
        let mut config = ModelClientConfig::new("openai");
        config.timeout = None;
        assert_eq!(
            config.http_client_config().timeout,
            Duration::from_secs(360)
        );
    }

    #[test]
    fn sanitized_headers_drop_protected_names_and_blank_values() {
        let mut config = ModelClientConfig::new("openai");
        for (name, value) in [
            ("Authorization", "Bearer secret"),
            ("Host", "api.example"),
            ("Content-Length", "12"),
            ("Transfer-Encoding", "chunked"),
            ("Connection", "keep-alive"),
            ("X-Blank", "   "),
            ("X-Kept", "kept"),
        ] {
            config.headers.insert(name.to_owned(), value.to_owned());
        }

        let sanitized = config.sanitized_headers();
        assert_eq!(sanitized.len(), 1);
        assert_eq!(sanitized.get("X-Kept").map(String::as_str), Some("kept"));
    }

    #[test]
    fn resolve_for_applies_the_provider_default_base_and_key() {
        let _env = EnvGuard::clear();
        let original = ModelClientConfig::new("OpenAI")
            .with_api_key("sk-explicit")
            .with_timeout(Duration::from_secs(11))
            .with_max_retries(2)
            .with_verify_ssl(false);
        let resolved = original
            .clone()
            .resolve_for("OpenAI")
            .expect("openai resolves");

        assert_eq!(resolved.provider, "openai");
        assert_eq!(
            resolved.api_base.as_deref(),
            Some("https://api.openai.com/v1")
        );
        assert_eq!(
            resolved.api_key.as_ref().map(SecretString::expose_secret),
            Some("sk-explicit")
        );
        assert_eq!(resolved.timeout, Some(Duration::from_secs(11)));
        assert_eq!(
            resolved.stream_first_chunk_timeout,
            Some(Duration::from_secs(300))
        );
        assert_eq!(resolved.stream_idle_timeout, Some(Duration::from_secs(60)));
        assert_eq!(resolved.max_retries, 2);
        assert!(!resolved.verify_ssl);
        assert_eq!(resolved.client_id, original.client_id);
    }

    #[test]
    fn resolve_for_prefers_explicit_base_then_env_then_provider_default() {
        let env = EnvGuard::clear();

        let explicit = ModelClientConfig::new("openai")
            .with_api_key("sk")
            .with_api_base("https://explicit.example/v1")
            .resolve_for("openai")
            .expect("resolves");
        assert_eq!(
            explicit.api_base.as_deref(),
            Some("https://explicit.example/v1")
        );

        env.set(env::API_BASE, "https://env.example/v1");
        let from_env = ModelClientConfig::new("openai")
            .with_api_key("sk")
            .resolve_for("openai")
            .expect("resolves");
        assert_eq!(from_env.api_base.as_deref(), Some("https://env.example/v1"));

        env.remove(env::API_BASE);
        env.remove(env::OPENAI_BASE_URL);
        let from_default = ModelClientConfig::new("deepseek")
            .with_api_key("sk")
            .resolve_for("deepseek")
            .expect("resolves");
        assert_eq!(
            from_default.api_base.as_deref(),
            Some("https://api.deepseek.com/v1")
        );
    }

    #[test]
    fn resolve_for_prefers_the_explicit_key_then_the_environment() {
        let env = EnvGuard::clear();
        env.set(env::OPENJIUWEN_API_KEY, "env-key");

        let explicit = ModelClientConfig::new("openai")
            .with_api_key("explicit-key")
            .resolve_for("openai")
            .expect("resolves");
        assert_eq!(
            explicit.api_key.as_ref().map(SecretString::expose_secret),
            Some("explicit-key")
        );

        let inherited = ModelClientConfig::new("openai")
            .resolve_for("openai")
            .expect("resolves");
        assert_eq!(
            inherited.api_key.as_ref().map(SecretString::expose_secret),
            Some("env-key")
        );
    }

    #[test]
    fn blank_explicit_key_is_treated_as_missing() {
        let env = EnvGuard::clear();
        env.set(env::OPENAI_API_KEY, "env-openai-key");

        let resolved = ModelClientConfig::new("openai")
            .with_api_key("   ")
            .resolve_for("openai")
            .expect("resolves");
        assert_eq!(
            resolved.api_key.as_ref().map(SecretString::expose_secret),
            Some("env-openai-key")
        );
    }

    #[test]
    fn resolve_for_normalizes_vendor_aliases() {
        let _env = EnvGuard::clear();
        let resolved = ModelClientConfig::new("DashScope")
            .with_api_key("sk")
            .resolve_for("DashScope")
            .expect("dashscope resolves");

        assert_eq!(resolved.provider, "qwen");
        assert_eq!(
            resolved.api_base.as_deref(),
            Some("https://dashscope.aliyuncs.com/compatible-mode/v1")
        );
    }

    #[test]
    fn local_providers_do_not_require_a_key() {
        let _env = EnvGuard::clear();
        for (name, base) in [
            ("ollama", "http://localhost:11434/v1"),
            ("lmstudio", "http://localhost:1234/v1"),
        ] {
            let resolved = ModelClientConfig::new(name)
                .resolve_for(name)
                .unwrap_or_else(|error| panic!("{name} must not need a key: {error}"));
            assert_eq!(resolved.provider, name);
            assert_eq!(resolved.api_base.as_deref(), Some(base));
            assert_eq!(resolved.api_key, None);
        }
    }

    #[test]
    fn pangu_without_an_api_base_is_a_config_error() {
        let _env = EnvGuard::clear();
        let error = ModelClientConfig::new("pangu")
            .with_api_key("sk")
            .resolve_for("pangu")
            .expect_err("pangu has no default base");

        assert_eq!(error.kind(), ErrorKind::Config);
        let message = error.to_string();
        assert!(message.contains("pangu"), "message: {message}");
        assert!(message.contains(env::API_BASE), "message: {message}");
    }

    #[test]
    fn missing_api_key_names_the_environment_variables_without_a_secret() {
        let _env = EnvGuard::clear();
        let error = ModelClientConfig::new("openai")
            .resolve_for("openai")
            .expect_err("openai requires a key");

        assert_eq!(error.kind(), ErrorKind::Config);
        let message = error.to_string();
        assert!(message.contains(env::API_KEY), "message: {message}");
        assert!(message.contains(env::OPENAI_API_KEY), "message: {message}");
        assert!(
            message.contains(env::OPENJIUWEN_API_KEY),
            "message: {message}"
        );
        assert!(!message.contains("sk-"), "message: {message}");
    }

    #[test]
    fn unknown_provider_is_unsupported_and_lists_the_builtins() {
        let _env = EnvGuard::clear();
        let error = ModelClientConfig::new("acme")
            .with_api_key("sk")
            .resolve_for("acme")
            .expect_err("acme is not built in");

        assert_eq!(error.kind(), ErrorKind::Unsupported);
        let message = error.to_string();
        assert!(message.contains("acme"), "message: {message}");
        assert!(message.contains("openai"), "message: {message}");
        assert!(message.contains("anthropic"), "message: {message}");
    }

    #[test]
    fn from_env_reads_provider_key_base_and_model_name() {
        let env = EnvGuard::clear();
        env.set(env::MODEL_PROVIDER, "deepseek");
        env.set(env::API_KEY, "env-key");
        env.set(env::OPENAI_BASE_URL, "https://env.example/v1");
        env.set(env::MODEL_NAME, "deepseek-chat");

        let client = ModelClientConfig::from_env();
        assert_eq!(client.provider, "deepseek");
        assert_eq!(
            client.api_key.as_ref().map(SecretString::expose_secret),
            Some("env-key")
        );
        assert_eq!(client.api_base.as_deref(), Some("https://env.example/v1"));
        assert!(uuid::Uuid::parse_str(&client.client_id).is_ok());

        assert_eq!(ModelRequestConfig::from_env().model_name, "deepseek-chat");
    }

    #[test]
    fn from_env_defaults_the_provider_to_openai() {
        let _env = EnvGuard::clear();
        let client = ModelClientConfig::from_env();
        assert_eq!(client.provider, "openai");
        assert_eq!(client.api_key, None);
        assert_eq!(client.api_base, None);
        assert_eq!(ModelRequestConfig::from_env().model_name, "");
    }

    #[test]
    fn debug_output_never_contains_the_api_key() {
        let config = ModelClientConfig::new("openai").with_api_key("sk-super-secret");
        let rendered = format!("{config:?}");
        assert!(!rendered.contains("sk-super-secret"), "debug: {rendered}");
    }
}
