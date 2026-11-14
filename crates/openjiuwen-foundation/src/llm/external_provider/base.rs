//! Provider-neutral contracts for external account providers.
//!
//! Ports `openjiuwen/extensions/external_provider/base.py`: the credential
//! surface ([`ExternalAuthProvider`]), the model-discovery surface
//! ([`ProviderModelCatalog`]), and the status payload ([`AuthStatus`]) every
//! external provider reports.
//!
//! Python declares both surfaces as `typing.Protocol`s with `**kwargs`; Rust has
//! no keyword arguments and needs object-safe traits, so:
//!
//! - the auth surface takes one explicit `force_refresh: bool`;
//! - the catalog surface takes a [`ModelCatalogRequest`] that carries the
//!   optional auth manager, an optional pre-resolved access token, and the
//!   refresh flag — the equivalent of Python's
//!   `list_model_ids(auth_manager=…, access_token=…, force_refresh=…)`;
//! - provider construction kwargs become [`ExternalProviderConfig`].
//!
//! Both traits are async and object-safe, so a registry hands back
//! `Arc<dyn ExternalAuthProvider>` / `Arc<dyn ProviderModelCatalog>`.

use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use openjiuwen_common::error::Result;
use openjiuwen_common::security::SecretString;
use serde::{Deserialize, Serialize};

/// Outcome of [`ExternalAuthProvider::status`].
///
/// Mirrors Python's `OpenAIAccountAuthStatus` dataclass: `authenticated`,
/// `auth_path`, `has_refresh_token`, `expires_at`, `needs_refresh`, and the
/// optional `error` code. `expires_at` is serialized as epoch seconds — the
/// Python wire shape — and accepts an RFC 3339 string on input for tolerance.
///
/// `auth_path` is `None` for a manager backed by a custom [`TokenStore`] that
/// has no file (for example `InMemoryTokenStore`); the Python dataclass always
/// carries a path because its only store is the auth file.
///
/// [`TokenStore`]: crate::llm::oauth::TokenStore
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuthStatus {
    /// Whether a usable credential is stored.
    pub authenticated: bool,
    /// Auth-store path carrying the credential, when the manager is file-backed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auth_path: Option<PathBuf>,
    /// Whether a refresh token is stored alongside the access token.
    #[serde(default)]
    pub has_refresh_token: bool,
    /// Access-token expiry, when known.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        with = "epoch_seconds"
    )]
    pub expires_at: Option<DateTime<Utc>>,
    /// Whether the access token expires within the manager's refresh skew.
    #[serde(default)]
    pub needs_refresh: bool,
    /// Provider error code explaining why the credential is unusable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl AuthStatus {
    /// Builds a status for "no usable credential, for `error`".
    ///
    /// The Python codes are `openai_account_auth_missing`,
    /// `openai_account_auth_invalid_store`, and
    /// `openai_account_auth_store_read_failed`; a provider may report its own.
    #[must_use]
    pub fn unauthenticated(auth_path: Option<PathBuf>, error: impl Into<String>) -> Self {
        Self {
            authenticated: false,
            auth_path,
            has_refresh_token: false,
            expires_at: None,
            needs_refresh: false,
            error: Some(error.into()),
        }
    }
}

/// Serializes `Option<DateTime<Utc>>` as epoch seconds (`null` when absent).
mod epoch_seconds {
    use chrono::{DateTime, Utc};
    use serde::de::Error as _;
    use serde::{Deserialize, Deserializer, Serializer};

    /// Writes the timestamp as a (fractional) number of seconds since the epoch.
    pub(super) fn serialize<S>(
        value: &Option<DateTime<Utc>>,
        serializer: S,
    ) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match value {
            Some(timestamp) => serializer.serialize_f64(millis_as_seconds(timestamp)),
            None => serializer.serialize_none(),
        }
    }

    /// Reads epoch seconds, or an RFC 3339 string as a tolerance.
    pub(super) fn deserialize<'de, D>(deserializer: D) -> Result<Option<DateTime<Utc>>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let Some(value) = Option::<serde_json::Value>::deserialize(deserializer)? else {
            return Ok(None);
        };
        match value {
            serde_json::Value::Number(number) => number
                .as_f64()
                .and_then(seconds_to_datetime)
                .map(Some)
                .ok_or_else(|| D::Error::custom("expires_at is out of range")),
            serde_json::Value::String(text) => DateTime::parse_from_rfc3339(text.trim())
                .map(|parsed| Some(parsed.with_timezone(&Utc)))
                .map_err(|error| D::Error::custom(format!("invalid expires_at: {error}"))),
            _ => Err(D::Error::custom(
                "expires_at must be epoch seconds or an RFC 3339 string",
            )),
        }
    }

    /// Returns `timestamp` as fractional epoch seconds.
    fn millis_as_seconds(timestamp: &DateTime<Utc>) -> f64 {
        timestamp.timestamp_millis() as f64 / 1000.0
    }

    /// Converts fractional epoch seconds into a UTC timestamp, rejecting values
    /// chrono cannot represent (which returns `None` rather than panicking).
    fn seconds_to_datetime(seconds: f64) -> Option<DateTime<Utc>> {
        if !seconds.is_finite() {
            return None;
        }
        let millis = seconds * 1000.0;
        if millis < i64::MIN as f64 || millis > i64::MAX as f64 {
            return None;
        }
        DateTime::from_timestamp_millis(millis.trunc() as i64)
    }
}

/// Typed replacement for the Python factory keyword arguments.
///
/// Python factories are classes called with `**kwargs`; Rust passes one
/// serde-defaulted struct instead:
///
/// | Python kwarg | field |
/// | --- | --- |
/// | `auth_path` | [`auth_path`](Self::auth_path) |
/// | `base_url` | [`base_url`](Self::base_url) |
/// | `cache_path` | [`cache_path`](Self::cache_path) |
/// | `timeout_seconds` | [`timeout`](Self::timeout) (seconds) |
/// | `refresh_skew_seconds` | [`refresh_skew`](Self::refresh_skew) (seconds) |
///
/// Every field is optional; `None` means "resolve from the environment and
/// built-in defaults", exactly like an omitted Python kwarg. Unknown keys are
/// ignored on deserialization so a config file written for a newer provider keeps
/// working.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ExternalProviderConfig {
    /// Path of the shared auth store, defaulting to the `llm::oauth` resolution.
    pub auth_path: Option<PathBuf>,
    /// Account API base URL, defaulting to `OPENJIUWEN_OPENAI_ACCOUNT_BASE_URL`
    /// and then the provider default.
    pub base_url: Option<String>,
    /// Model-cache file, defaulting to the auth-store directory.
    pub cache_path: Option<PathBuf>,
    /// Request timeout in seconds.
    pub timeout: Option<f64>,
    /// Refresh skew in seconds (`120` by default).
    pub refresh_skew: Option<u64>,
}

impl ExternalProviderConfig {
    /// Builds a config where every field resolves from the environment.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the auth-store path.
    #[must_use]
    pub fn with_auth_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.auth_path = Some(path.into());
        self
    }

    /// Sets the account API base URL.
    #[must_use]
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    /// Sets the model-cache file.
    #[must_use]
    pub fn with_cache_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.cache_path = Some(path.into());
        self
    }

    /// Sets the request timeout in seconds.
    #[must_use]
    pub fn with_timeout(mut self, seconds: f64) -> Self {
        self.timeout = Some(seconds);
        self
    }

    /// Sets the refresh skew in seconds.
    #[must_use]
    pub fn with_refresh_skew(mut self, seconds: u64) -> Self {
        self.refresh_skew = Some(seconds);
        self
    }

    /// Returns the refresh skew as a [`Duration`], or `fallback` when unset.
    #[must_use]
    pub fn refresh_skew_or(&self, fallback: Duration) -> Duration {
        self.refresh_skew
            .map_or(fallback, Duration::from_secs)
    }
}

/// Arguments to [`ProviderModelCatalog::list_model_ids`].
///
/// Replaces Python's `**kwargs` with typed fields. Precedence matches Python:
/// an explicit [`access_token`](Self::access_token) wins; otherwise the
/// [`auth_manager`](Self::auth_manager) resolves one. A request with neither
/// asks the catalog to answer from its cache or built-in defaults.
///
/// The type is not `serde`-serializable on purpose: it carries a trait object
/// and a bearer token, neither of which belongs in a serialized document.
#[derive(Default)]
pub struct ModelCatalogRequest {
    /// Credential source consulted when no explicit token is supplied.
    pub auth_manager: Option<Arc<dyn ExternalAuthProvider>>,
    /// Pre-resolved bearer token; wins over `auth_manager`.
    pub access_token: Option<SecretString>,
    /// Forwarded to the auth manager as `force_refresh`.
    pub force_refresh: bool,
}

impl ModelCatalogRequest {
    /// Builds a request with no credential and `force_refresh: false`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Supplies the auth manager used to resolve a token.
    #[must_use]
    pub fn with_auth_manager(mut self, manager: Arc<dyn ExternalAuthProvider>) -> Self {
        self.auth_manager = Some(manager);
        self
    }

    /// Supplies a pre-resolved bearer token.
    #[must_use]
    pub fn with_access_token(mut self, token: impl Into<String>) -> Self {
        self.access_token = Some(SecretString::new(token));
        self
    }

    /// Supplies a pre-resolved bearer token already wrapped as a secret.
    #[must_use]
    pub fn with_secret(mut self, token: SecretString) -> Self {
        self.access_token = Some(token);
        self
    }

    /// Requests a forced credential refresh.
    #[must_use]
    pub fn with_force_refresh(mut self) -> Self {
        self.force_refresh = true;
        self
    }
}

impl fmt::Debug for ModelCatalogRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        // `SecretString`'s own `Debug` renders the shared `<REDACTED>`
        // placeholder, so the token can be listed without leaking it.
        formatter
            .debug_struct("ModelCatalogRequest")
            .field("auth_manager", &self.auth_manager.as_ref().map(|_| "<provider>"))
            .field("access_token", &self.access_token)
            .field("force_refresh", &self.force_refresh)
            .finish()
    }
}

/// Token-auth surface shared by external account providers.
///
/// Ports the `ExternalAuthProvider` protocol from
/// `openjiuwen/extensions/external_provider/base.py`. Provider login flows stay
/// provider-specific; this trait covers only the stable operations available
/// once credentials exist.
#[async_trait]
pub trait ExternalAuthProvider: Send + Sync {
    /// Returns the current credential status.
    ///
    /// Implementations report a missing, unreadable, or corrupt credential as an
    /// `authenticated: false` status with an [`AuthStatus::error`] code rather
    /// than an error — Python's `status()` never raises for those cases.
    ///
    /// # Errors
    ///
    /// Only for failures a status cannot express (for example a store backend
    /// that fails with a non-I/O error).
    async fn status(&self) -> Result<AuthStatus>;

    /// Removes the stored credential, returning `true` when one was present.
    ///
    /// # Errors
    ///
    /// Returns the store error when the credential store cannot be read or
    /// written.
    async fn logout(&self) -> Result<bool>;

    /// Returns a usable access token, refreshing it when needed.
    ///
    /// `force_refresh` refreshes even a token that is still valid.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::Error::Config`] when no credential is
    /// stored (sign-in is required) and otherwise the errors of the refresh
    /// endpoint.
    async fn resolve_access_token(&self, force_refresh: bool) -> Result<SecretString>;
}

/// Model-discovery surface shared by external providers.
///
/// Ports the `ProviderModelCatalog` protocol from
/// `openjiuwen/extensions/external_provider/base.py`.
#[async_trait]
pub trait ProviderModelCatalog: Send + Sync {
    /// Returns the model IDs this provider offers.
    ///
    /// # Errors
    ///
    /// Implementations return a typed error only when no fallback exists; the
    /// built-in OpenAI-account catalog degrades to cache and defaults and
    /// therefore answers `Ok`.
    async fn list_model_ids(&self, request: &ModelCatalogRequest) -> Result<Vec<String>>;
}
