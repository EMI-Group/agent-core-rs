//! Configuration of the `DashScope` media client.
//!
//! [`DashScopeConfig`] is the one place a caller states which `DashScope`
//! deployment to talk to, with which credential, and how long the video task
//! polling is allowed to take. Every field has a documented default, so
//! [`DashScopeConfig::new`] plus a key is enough for the public cloud.
//!
//! [`DashScopeConfig::from_env`] reads the two variable families the reference
//! SDK uses — `DASHSCOPE_*` and the `OPENJIUWEN_`-prefixed `OPENJIUWEN_DASHSCOPE_*`
//! — through [`openjiuwen_common::config::env`], so blank values are treated as
//! unset and never reach the wire. [`DashScopeConfig::from_client_config`] is the
//! bridge for callers that already hold a [`ModelClientConfig`] (the intended
//! caller has `provider == "qwen"`/`"dashscope"`).

use std::time::Duration;

use openjiuwen_common::config::env;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::security::SecretString;
use serde::{Deserialize, Serialize};

use crate::llm::config::ModelClientConfig;

/// Default `DashScope` service endpoint (the vendor's public cloud).
pub const DEFAULT_DASHSCOPE_API_BASE: &str = "https://dashscope.aliyuncs.com";

/// Prefixed API key variable, which wins over [`DASHSCOPE_API_KEY_ENV`].
pub const OPENJIUWEN_DASHSCOPE_API_KEY_ENV: &str = "OPENJIUWEN_DASHSCOPE_API_KEY";

/// Vendor API key variable (`DASHSCOPE_API_KEY`), read when the prefixed form is unset.
pub const DASHSCOPE_API_KEY_ENV: &str = "DASHSCOPE_API_KEY";

/// Prefixed base-URL variable, which wins over [`DASHSCOPE_API_BASE_ENV`].
pub const OPENJIUWEN_DASHSCOPE_API_BASE_ENV: &str = "OPENJIUWEN_DASHSCOPE_API_BASE";

/// Vendor base-URL variable (`DASHSCOPE_API_BASE`), read when the prefixed form is unset.
pub const DASHSCOPE_API_BASE_ENV: &str = "DASHSCOPE_API_BASE";

/// Default budget of one media HTTP request (360 s), the same default
/// [`ModelClientConfig`] uses for a model call.
pub const DEFAULT_MODEL_TIMEOUT: Duration = Duration::from_secs(360);

/// Default delay between two video task polls (2 s).
pub const DEFAULT_POLL_INTERVAL: Duration = Duration::from_secs(2);

/// Default overall budget of one video generation call (10 minutes).
///
/// Video synthesis is asynchronous and routinely takes minutes; ten minutes is
/// the vendor's documented ceiling for a single task.
pub const DEFAULT_TASK_TIMEOUT: Duration = Duration::from_secs(10 * 60);

/// Connection settings of [`DashScopeMediaClient`](super::DashScopeMediaClient).
///
/// `Debug` redacts the key through [`SecretString`], while JSON keeps it so the
/// configuration round-trips (the same contract as
/// [`ModelClientConfig`](crate::llm::config::ModelClientConfig)).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct DashScopeConfig {
    /// Service root, without the API path (for example
    /// `https://dashscope.aliyuncs.com`); a trailing slash is tolerated.
    pub api_base: String,
    /// `DashScope` API key, sent as `Authorization: Bearer <key>`.
    pub api_key: SecretString,
    /// Budget of one HTTP request.
    pub timeout: Duration,
    /// Delay before each video task poll.
    pub poll_interval: Duration,
    /// Overall budget of one video generation call, polling included.
    pub task_timeout: Duration,
}

impl Default for DashScopeConfig {
    /// The public cloud endpoint, an unset (empty) key, a 360 s request budget,
    /// a 2 s poll interval, and a 10 minute task budget.
    fn default() -> Self {
        Self {
            api_base: DEFAULT_DASHSCOPE_API_BASE.to_owned(),
            api_key: SecretString::new(String::new()),
            timeout: DEFAULT_MODEL_TIMEOUT,
            poll_interval: DEFAULT_POLL_INTERVAL,
            task_timeout: DEFAULT_TASK_TIMEOUT,
        }
    }
}

impl DashScopeConfig {
    /// Builds a configuration for the public cloud with the given API key.
    #[must_use]
    pub fn new(api_key: impl Into<SecretString>) -> Self {
        Self {
            api_key: api_key.into(),
            ..Self::default()
        }
    }

    /// Replaces the service root.
    #[must_use]
    pub fn with_api_base(mut self, api_base: impl Into<String>) -> Self {
        self.api_base = api_base.into();
        self
    }

    /// Replaces the per-request timeout.
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Replaces the delay between two video task polls.
    #[must_use]
    pub fn with_poll_interval(mut self, poll_interval: Duration) -> Self {
        self.poll_interval = poll_interval;
        self
    }

    /// Replaces the overall budget of one video generation call.
    #[must_use]
    pub fn with_task_timeout(mut self, task_timeout: Duration) -> Self {
        self.task_timeout = task_timeout;
        self
    }

    /// Reads the key and the endpoint from the environment.
    ///
    /// Precedence, highest first:
    /// `OPENJIUWEN_DASHSCOPE_API_KEY` → `DASHSCOPE_API_KEY` for the key, and
    /// `OPENJIUWEN_DASHSCOPE_API_BASE` → `DASHSCOPE_API_BASE` →
    /// [`DEFAULT_DASHSCOPE_API_BASE`] for the endpoint. Timing knobs keep their
    /// defaults. Blank values are treated as unset.
    ///
    /// # Errors
    ///
    /// Returns [`Error::config`] naming both variables when no key is configured.
    pub fn from_env() -> Result<Self> {
        let api_key = first_env(&[OPENJIUWEN_DASHSCOPE_API_KEY_ENV, DASHSCOPE_API_KEY_ENV])
            .ok_or_else(|| Error::config(no_key_message()))?;
        let api_base = first_env(&[OPENJIUWEN_DASHSCOPE_API_BASE_ENV, DASHSCOPE_API_BASE_ENV])
            .unwrap_or_else(|| DEFAULT_DASHSCOPE_API_BASE.to_owned());
        Ok(Self {
            api_base,
            api_key: SecretString::new(api_key),
            ..Self::default()
        })
    }

    /// Adapts an LLM [`ModelClientConfig`] to this media client.
    ///
    /// The key and the endpoint are taken from `client` when set, otherwise from
    /// the environment (with the precedence of [`DashScopeConfig::from_env`]),
    /// and the request timeout follows `client.timeout`. Polling knobs keep their
    /// defaults.
    ///
    /// The media client is `DashScope`-specific: callers are expected to hold a
    /// client config whose provider is `qwen`/`dashscope` (the Python SDK gates
    /// these calls on `endpoint_profile == "dashscope"`). Passing any other
    /// provider here still talks to `DashScope`, so gate the call at the call site.
    ///
    /// # Errors
    ///
    /// Returns [`Error::config`] when neither the config nor the environment
    /// carries a key.
    pub fn from_client_config(client: &ModelClientConfig) -> Result<Self> {
        let configured_key = client
            .api_key
            .as_ref()
            .map(SecretString::expose_secret)
            .map(str::trim)
            .filter(|key| !key.is_empty())
            .map(SecretString::new);
        let api_key = match configured_key {
            Some(key) => key,
            None => SecretString::new(
                first_env(&[OPENJIUWEN_DASHSCOPE_API_KEY_ENV, DASHSCOPE_API_KEY_ENV])
                    .ok_or_else(|| Error::config(no_key_message()))?,
            ),
        };
        let configured_base = client
            .api_base
            .as_deref()
            .map(str::trim)
            .filter(|base| !base.is_empty())
            .map(str::to_owned);
        Ok(Self {
            api_base: configured_base.unwrap_or_else(|| DEFAULT_DASHSCOPE_API_BASE.to_owned()),
            api_key,
            timeout: client.timeout.unwrap_or(DEFAULT_MODEL_TIMEOUT),
            ..Self::default()
        })
    }
}

/// Returns the first non-blank variable among `names`.
fn first_env(names: &[&str]) -> Option<String> {
    names.iter().find_map(|name| env::var(name))
}

/// The message every missing-key path reports.
fn no_key_message() -> String {
    format!(
        "DashScope API key is required: set {OPENJIUWEN_DASHSCOPE_API_KEY_ENV} \
         or {DASHSCOPE_API_KEY_ENV}"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, MutexGuard, OnceLock};

    /// Every variable the resolvers in this module read.
    const ENV_VARS: &[&str] = &[
        OPENJIUWEN_DASHSCOPE_API_KEY_ENV,
        DASHSCOPE_API_KEY_ENV,
        OPENJIUWEN_DASHSCOPE_API_BASE_ENV,
        DASHSCOPE_API_BASE_ENV,
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

        fn set(name: &str, value: &str) {
            std::env::set_var(name, value);
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
    fn default_targets_the_public_cloud_without_a_key() {
        let config = DashScopeConfig::default();
        assert_eq!(config.api_base, DEFAULT_DASHSCOPE_API_BASE);
        assert!(config.api_key.is_empty());
        assert_eq!(config.timeout, DEFAULT_MODEL_TIMEOUT);
        assert_eq!(config.poll_interval, DEFAULT_POLL_INTERVAL);
        assert_eq!(config.task_timeout, DEFAULT_TASK_TIMEOUT);
    }

    #[test]
    fn builders_replace_every_knob() {
        let config = DashScopeConfig::new("sk-test")
            .with_api_base("https://example.test")
            .with_timeout(Duration::from_secs(7))
            .with_poll_interval(Duration::from_millis(5))
            .with_task_timeout(Duration::from_secs(9));
        assert_eq!(config.api_key.expose_secret(), "sk-test");
        assert_eq!(config.api_base, "https://example.test");
        assert_eq!(config.timeout, Duration::from_secs(7));
        assert_eq!(config.poll_interval, Duration::from_millis(5));
        assert_eq!(config.task_timeout, Duration::from_secs(9));
    }

    #[test]
    fn from_env_prefers_the_openjiuwen_prefixed_variables() {
        let _env = EnvGuard::clear();
        EnvGuard::set(DASHSCOPE_API_KEY_ENV, "unprefixed-key");
        EnvGuard::set(DASHSCOPE_API_BASE_ENV, "https://unprefixed.test");
        EnvGuard::set(OPENJIUWEN_DASHSCOPE_API_KEY_ENV, "prefixed-key");
        EnvGuard::set(OPENJIUWEN_DASHSCOPE_API_BASE_ENV, "https://prefixed.test");

        let config = DashScopeConfig::from_env().expect("a key is configured");
        assert_eq!(config.api_key.expose_secret(), "prefixed-key");
        assert_eq!(config.api_base, "https://prefixed.test");
    }

    #[test]
    fn from_env_falls_back_to_the_unprefixed_variables_and_the_default_base() {
        let _env = EnvGuard::clear();
        EnvGuard::set(DASHSCOPE_API_KEY_ENV, "  vendor-key  ");

        let config = DashScopeConfig::from_env().expect("a key is configured");
        assert_eq!(config.api_key.expose_secret(), "vendor-key");
        assert_eq!(config.api_base, DEFAULT_DASHSCOPE_API_BASE);
        assert_eq!(config.timeout, DEFAULT_MODEL_TIMEOUT);
    }

    #[test]
    fn from_env_without_a_key_is_a_config_error() {
        let _env = EnvGuard::clear();
        EnvGuard::set(DASHSCOPE_API_BASE_ENV, "https://example.test");

        let error = DashScopeConfig::from_env().expect_err("no key is configured");
        assert_eq!(error.kind(), openjiuwen_common::error::ErrorKind::Config);
        assert!(error.to_string().contains(DASHSCOPE_API_KEY_ENV));
        assert!(error.to_string().contains(OPENJIUWEN_DASHSCOPE_API_KEY_ENV));
    }

    #[test]
    fn a_blank_prefixed_key_falls_through_to_the_vendor_variable() {
        let _env = EnvGuard::clear();
        EnvGuard::set(OPENJIUWEN_DASHSCOPE_API_KEY_ENV, "   ");
        EnvGuard::set(DASHSCOPE_API_KEY_ENV, "vendor-key");

        let config = DashScopeConfig::from_env().expect("a key is configured");
        assert_eq!(config.api_key.expose_secret(), "vendor-key");
    }

    #[test]
    fn from_client_config_carries_key_base_and_timeout() {
        let _env = EnvGuard::clear();
        let client = ModelClientConfig::new("qwen")
            .with_api_key("sk-config")
            .with_api_base("https://dashscope.internal")
            .with_timeout(Duration::from_secs(42));

        let config = DashScopeConfig::from_client_config(&client).expect("the config has a key");
        assert_eq!(config.api_key.expose_secret(), "sk-config");
        assert_eq!(config.api_base, "https://dashscope.internal");
        assert_eq!(config.timeout, Duration::from_secs(42));
        assert_eq!(config.poll_interval, DEFAULT_POLL_INTERVAL);
        assert_eq!(config.task_timeout, DEFAULT_TASK_TIMEOUT);
    }

    #[test]
    fn from_client_config_falls_back_to_the_environment() {
        let _env = EnvGuard::clear();
        EnvGuard::set(DASHSCOPE_API_KEY_ENV, "env-key");
        let client = ModelClientConfig::new("qwen");

        let config = DashScopeConfig::from_client_config(&client).expect("the env has a key");
        assert_eq!(config.api_key.expose_secret(), "env-key");
        assert_eq!(config.api_base, DEFAULT_DASHSCOPE_API_BASE);
        assert_eq!(config.timeout, DEFAULT_MODEL_TIMEOUT);
    }

    #[test]
    fn from_client_config_without_a_key_anywhere_is_a_config_error() {
        let _env = EnvGuard::clear();
        let error = DashScopeConfig::from_client_config(&ModelClientConfig::new("qwen"))
            .expect_err("no key is configured");
        assert_eq!(error.kind(), openjiuwen_common::error::ErrorKind::Config);
    }

    #[test]
    fn the_key_is_redacted_in_debug_and_kept_in_json() {
        let config = DashScopeConfig::new("sk-secret").with_api_base("https://example.test");
        let debug = format!("{config:?}");
        assert!(!debug.contains("sk-secret"), "debug: {debug}");

        let json = serde_json::to_value(&config).expect("the config serializes");
        assert_eq!(json["api_key"], "sk-secret");
        let restored: DashScopeConfig =
            serde_json::from_value(json).expect("the config deserializes");
        assert_eq!(restored, config);
    }
}
