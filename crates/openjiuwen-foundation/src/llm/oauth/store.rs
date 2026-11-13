//! OAuth token, device-code, and token-store types for the `OpenAI` account provider.
//!
//! The credentials live in openJiuwen's private auth store: a JSON object with a
//! `providers` map and one entry per provider. This module owns only the
//! `OpenAIAccount` entry; every other provider's entry is preserved verbatim by
//! [`FileTokenStore`]'s read-merge-write.
//!
//! Writes are atomic — a temporary file in the same directory is renamed over the
//! target — and, on unix, the store is restricted to its owner (`0o600`), because
//! it holds bearer and refresh tokens in plaintext (Python parity: the reference
//! implementation uses the same temporary-file/`os.replace`/`chmod` sequence).

use std::fmt;
use std::path::{Path, PathBuf};
use std::time::Duration;

use async_trait::async_trait;
use base64::Engine as _;
use chrono::{DateTime, TimeDelta, Utc};
use openjiuwen_common::config::env;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::security::SecretString;
use parking_lot::Mutex;
use serde::ser::SerializeMap;
use serde::{Deserialize, Serialize, Serializer};
use serde_json::{Map, Value};

/// Provider key this module owns inside the auth store envelope.
pub const PROVIDER_KEY: &str = "OpenAIAccount";

/// Auth mode recorded next to the stored tokens.
pub const AUTH_MODE: &str = "openai_account_oauth";

/// Base URL recorded next to the stored tokens by default.
pub const DEFAULT_BASE_URL: &str = "https://chatgpt.com/backend-api/codex";

/// Environment variable overriding the auth store location.
pub const AUTH_FILE_ENV: &str = "OPENJIUWEN_AUTH_FILE";

/// Access tokens are refreshed once they expire within this window.
pub const REFRESH_SKEW: Duration = Duration::from_secs(120);

/// Token type assumed when a provider omits `token_type`.
const DEFAULT_TOKEN_TYPE: &str = "Bearer";

/// File name of the auth store under the openJiuwen home directory.
const AUTH_FILE_NAME: &str = "auth.json";

/// Directory name of the openJiuwen home used when `OPENJIUWEN_HOME` is unset.
const DEFAULT_HOME_DIR: &str = ".openjiuwen";

/// Highest timestamp accepted, so a bogus value cannot overflow date arithmetic.
///
/// The bound is `9999-12-31T23:59:59Z`, which is chrono's own maximum year.
const MAX_EPOCH_SECONDS: f64 = 253_402_300_799.0;

/// Tokens returned by the `OpenAI` account OAuth endpoints.
///
/// The type is `serde`-serializable in the shape the token endpoint and the auth
/// store use. Deserialization is deliberately tolerant:
///
/// - unknown fields are ignored, and an empty `refresh_token` is treated as
///   absent, matching the reference implementation's normalization;
/// - the expiry is resolved with the reference implementation's precedence: an
///   `expires_at` in either epoch seconds or RFC 3339, then `expires_in` (seconds
///   from now, clamped at zero), then — when a previous token is supplied — that
///   token's `expires_at`, and finally the `exp` claim of a JWT `access_token`
///   (see `parse_token_payload`); a token whose expiry cannot be resolved never
///   expires, so it is never refreshed.
///
/// Serialization always writes the canonical field set — `access_token`,
/// `refresh_token`, `id_token`, `expires_at` (epoch seconds), `token_type`, and
/// `scope` — so a round trip never communicates "expires in" and "expires at"
/// twice.
///
/// `Debug` never prints a credential: every token field is a [`SecretString`],
/// whose `Debug` renders the shared `<REDACTED>` placeholder.
#[derive(Clone, Deserialize)]
#[serde(from = "TokenPayload")]
pub struct OAuthToken {
    /// The bearer token presented to the model endpoint.
    pub access_token: SecretString,
    /// The token used to mint a new access token; absent when the provider
    /// issued none.
    pub refresh_token: Option<SecretString>,
    /// The `OpenID` Connect identity token, when the provider issued one.
    pub id_token: Option<SecretString>,
    /// Absolute expiry, when known.
    pub expires_at: Option<DateTime<Utc>>,
    /// The token type reported by the provider (`Bearer` by default).
    pub token_type: String,
    /// The granted OAuth scopes, when the provider reported them.
    pub scope: Option<String>,
}

impl OAuthToken {
    /// Builds a token for `access_token` with type `Bearer` and no expiry.
    ///
    /// The remaining fields are filled in by the builders or by deserialization.
    pub fn new(access_token: impl Into<String>) -> Self {
        Self {
            access_token: SecretString::new(access_token),
            refresh_token: None,
            id_token: None,
            expires_at: None,
            token_type: DEFAULT_TOKEN_TYPE.to_owned(),
            scope: None,
        }
    }

    /// Sets the refresh token.
    #[must_use]
    pub fn with_refresh_token(mut self, refresh_token: impl Into<String>) -> Self {
        self.refresh_token = Some(SecretString::new(refresh_token));
        self
    }

    /// Sets the identity token.
    #[must_use]
    pub fn with_id_token(mut self, id_token: impl Into<String>) -> Self {
        self.id_token = Some(SecretString::new(id_token));
        self
    }

    /// Sets the absolute expiry.
    #[must_use]
    pub fn with_expires_at(mut self, expires_at: DateTime<Utc>) -> Self {
        self.expires_at = Some(expires_at);
        self
    }

    /// Sets the token type.
    #[must_use]
    pub fn with_token_type(mut self, token_type: impl Into<String>) -> Self {
        self.token_type = token_type.into();
        self
    }

    /// Sets the granted scopes.
    #[must_use]
    pub fn with_scope(mut self, scope: impl Into<String>) -> Self {
        self.scope = Some(scope.into());
        self
    }

    /// Returns the expiry the refresh decision is based on.
    ///
    /// The recorded [`expires_at`](Self::expires_at) wins; when it is absent, the
    /// `exp` claim of a JWT access token is used, so a token read from a store
    /// that never recorded an expiry still refreshes on time (Python parity:
    /// `OpenAIAccountTokens.is_expiring`). A token whose expiry cannot be
    /// resolved yields `None` and therefore never expires.
    #[must_use]
    pub fn effective_expires_at(&self) -> Option<DateTime<Utc>> {
        self.expires_at
            .or_else(|| jwt_exp(self.access_token.expose_secret()))
    }

    /// Returns `true` when the token expires within `skew` (or already has).
    ///
    /// A token with no resolvable expiry returns `false`: `None` means "unknown",
    /// and treating an unknown expiry as expired would make every such token
    /// trigger a refresh round-trip on each call. A `skew` too large for chrono's
    /// duration range is saturated, so it simply counts as expired.
    #[must_use]
    pub fn is_expired(&self, skew: Duration) -> bool {
        let Some(expires_at) = self.effective_expires_at() else {
            return false;
        };
        let skew = TimeDelta::from_std(skew).unwrap_or(TimeDelta::MAX);
        Utc::now()
            .checked_add_signed(skew)
            .is_none_or(|deadline| expires_at <= deadline)
    }
}

impl fmt::Debug for OAuthToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // `SecretString`'s own `Debug` prints `<REDACTED>`, so listing the
        // fields here stays safe while remaining useful for diagnostics.
        f.debug_struct("OAuthToken")
            .field("access_token", &self.access_token)
            .field("refresh_token", &self.refresh_token)
            .field("id_token", &self.id_token)
            .field("expires_at", &self.expires_at)
            .field("token_type", &self.token_type)
            .field("scope", &self.scope)
            .finish()
    }
}

impl Serialize for OAuthToken {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut map = serializer.serialize_map(None)?;
        map.serialize_entry("access_token", &self.access_token)?;
        if let Some(refresh_token) = &self.refresh_token {
            map.serialize_entry("refresh_token", refresh_token)?;
        }
        if let Some(id_token) = &self.id_token {
            map.serialize_entry("id_token", id_token)?;
        }
        if let Some(expires_at) = self.expires_at {
            map.serialize_entry("expires_at", &expires_at.timestamp())?;
        }
        map.serialize_entry("token_type", &self.token_type)?;
        if let Some(scope) = &self.scope {
            map.serialize_entry("scope", scope)?;
        }
        map.end()
    }
}

/// The wire shape [`OAuthToken`] is deserialized from.
#[derive(Deserialize)]
struct TokenPayload {
    access_token: SecretString,
    #[serde(default)]
    refresh_token: Option<SecretString>,
    #[serde(default)]
    id_token: Option<SecretString>,
    #[serde(default)]
    expires_at: Option<Value>,
    #[serde(default)]
    expires_in: Option<Value>,
    #[serde(default)]
    token_type: Option<String>,
    #[serde(default)]
    scope: Option<String>,
}

impl TokenPayload {
    /// Resolves the payload into a token, taking every field it omits from
    /// `previous`.
    ///
    /// The expiry follows the reference implementation's precedence:
    /// `expires_at` → `now + max(expires_in, 0)` → `previous.expires_at` → the
    /// JWT `exp` claim. Only when all four fail is the expiry unknown.
    fn into_token(self, previous: Option<&OAuthToken>) -> OAuthToken {
        let expires_at = self
            .expires_at
            .as_ref()
            .and_then(value_to_timestamp)
            .or_else(|| self.expires_in.as_ref().and_then(expiry_from_now))
            .or_else(|| previous.and_then(|token| token.expires_at))
            .or_else(|| jwt_exp(self.access_token.expose_secret()));
        let token_type = self
            .token_type
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_owned)
            .or_else(|| previous.map(|token| token.token_type.clone()))
            .unwrap_or_else(|| DEFAULT_TOKEN_TYPE.to_owned());
        OAuthToken {
            access_token: self.access_token,
            refresh_token: non_empty(self.refresh_token)
                .or_else(|| previous.and_then(|token| non_empty(token.refresh_token.clone()))),
            id_token: non_empty(self.id_token)
                .or_else(|| previous.and_then(|token| non_empty(token.id_token.clone()))),
            expires_at,
            token_type,
            scope: self
                .scope
                .filter(|scope| !scope.trim().is_empty())
                .or_else(|| previous.and_then(|token| token.scope.clone())),
        }
    }
}

impl From<TokenPayload> for OAuthToken {
    fn from(payload: TokenPayload) -> Self {
        payload.into_token(None)
    }
}

/// Deserializes a token endpoint payload, resolving omitted fields from
/// `previous`.
///
/// [`OAuthToken`]'s own `Deserialize` is this function with no previous token;
/// the refresh path calls it with the token being replaced, so a response that
/// rotates only `access_token` keeps the refresh token, the identity token, the
/// scopes, the token type, and the previous expiry.
///
/// # Errors
///
/// Returns the `serde_json` error when `payload` is not a token response (most
/// commonly a missing `access_token`).
pub(crate) fn parse_token_payload(
    payload: &[u8],
    previous: Option<&OAuthToken>,
) -> std::result::Result<OAuthToken, serde_json::Error> {
    let payload: TokenPayload = serde_json::from_slice(payload)?;
    Ok(payload.into_token(previous))
}

/// Drops an empty secret, treating it as "the provider did not send the field".
fn non_empty(token: Option<SecretString>) -> Option<SecretString> {
    token.filter(|value| !value.is_empty())
}

/// Converts a JSON scalar holding seconds or an RFC 3339 timestamp to an expiry.
fn value_to_timestamp(value: &Value) -> Option<DateTime<Utc>> {
    match value {
        Value::Number(number) => number.as_f64().and_then(timestamp_from_epoch_seconds),
        Value::String(text) => {
            let text = text.trim();
            if let Ok(parsed) = DateTime::parse_from_rfc3339(text) {
                return Some(parsed.with_timezone(&Utc));
            }
            text.parse::<f64>()
                .ok()
                .and_then(timestamp_from_epoch_seconds)
        }
        _ => None,
    }
}

/// Converts epoch seconds to a UTC timestamp, rejecting non-positive values.
fn timestamp_from_epoch_seconds(seconds: f64) -> Option<DateTime<Utc>> {
    if !seconds.is_finite() || seconds <= 0.0 {
        return None;
    }
    let seconds = seconds.min(MAX_EPOCH_SECONDS);
    let whole = seconds.trunc();
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let nanos = ((seconds - whole) * 1_000_000_000.0)
        .round()
        .clamp(0.0, 999_999_999.0) as u32;
    #[allow(clippy::cast_possible_truncation)]
    DateTime::from_timestamp(whole as i64, nanos)
}

/// Converts a JSON scalar holding a lifetime in seconds to an absolute expiry.
fn expiry_from_now(value: &Value) -> Option<DateTime<Utc>> {
    let seconds = value_to_f64(value)?;
    let seconds = if seconds.is_finite() {
        seconds.clamp(0.0, MAX_EPOCH_SECONDS)
    } else {
        0.0
    };
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let delta = TimeDelta::try_seconds(seconds as i64).unwrap_or_else(TimeDelta::zero);
    Utc::now().checked_add_signed(delta)
}

/// Reads a JSON scalar as a number, tolerating numeric strings.
fn value_to_f64(value: &Value) -> Option<f64> {
    match value {
        Value::Number(number) => number.as_f64(),
        Value::String(text) => text.trim().parse::<f64>().ok(),
        _ => None,
    }
}

/// Reads a JSON scalar as a non-negative whole number of seconds.
fn value_to_seconds(value: &Value) -> Option<u64> {
    let seconds = value_to_f64(value)?;
    if seconds.is_finite() {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        Some(seconds.clamp(0.0, MAX_EPOCH_SECONDS) as u64)
    } else {
        Some(0)
    }
}

/// Reads the `exp` claim out of a JWT access token.
///
/// The payload is the second dot-separated segment, base64url-encoded with or
/// without padding; anything that does not decode to a JSON object carrying a
/// positive `exp` yields `None`. The signature is deliberately **not** verified
/// (Python parity: `_jwt_exp`): the claim only decides *when* to refresh, never
/// whether to trust the token, so a forged `exp` can at most cause an extra
/// refresh — and the refreshed token still has to come from the issuer.
fn jwt_exp(access_token: &str) -> Option<DateTime<Utc>> {
    let payload = access_token.split('.').nth(1)?.trim_end_matches('=');
    if payload.is_empty() {
        return None;
    }
    let decoded = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(payload)
        .ok()?;
    let claims: Value = serde_json::from_slice(&decoded).ok()?;
    claims.get("exp").and_then(value_to_timestamp)
}

/// The default polling interval advertised by the device-code endpoint.
const fn default_interval() -> u64 {
    5
}

/// Shortest polling interval accepted from the device-code endpoint.
///
/// The reference implementation clamps the advertised interval with
/// `max(3, interval)`, so a provider that asks to be polled faster still gets at
/// least three seconds.
const MIN_ADVERTISED_INTERVAL: u64 = 3;

/// Deserializes `interval`, falling back to [`default_interval`] for a null value
/// and to [`MIN_ADVERTISED_INTERVAL`] for a smaller one.
fn deserialize_interval<'de, D>(deserializer: D) -> std::result::Result<u64, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = Option::<Value>::deserialize(deserializer)?;
    Ok(value
        .as_ref()
        .and_then(value_to_seconds)
        .unwrap_or_else(default_interval)
        .max(MIN_ADVERTISED_INTERVAL))
}

/// Deserializes a lifetime in seconds, treating a null value as `0` (unknown).
fn deserialize_seconds<'de, D>(deserializer: D) -> std::result::Result<u64, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = Option::<Value>::deserialize(deserializer)?;
    Ok(value.as_ref().and_then(value_to_seconds).unwrap_or(0))
}

/// A pending device authorization started by the device-code endpoint.
///
/// The type is tolerant: a missing `interval` falls back to 5 seconds, an
/// advertised interval below three seconds is raised to three (Python parity: the
/// endpoint is never polled faster than the issuer allows), a missing
/// `expires_in` becomes `0` ("the provider did not say"), and both fields accept
/// a number or a numeric string.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceCode {
    /// The code the user types into the verification page.
    pub user_code: String,
    /// The server-side identifier the polling endpoint expects back.
    pub device_auth_id: String,
    /// The advertised polling interval in seconds, never below three.
    #[serde(
        default = "default_interval",
        deserialize_with = "deserialize_interval"
    )]
    pub interval: u64,
    /// How long the code stays valid, in seconds; `0` means "unspecified".
    #[serde(default, deserialize_with = "deserialize_seconds")]
    pub expires_in: u64,
}

impl DeviceCode {
    /// Builds a device code with the default interval and an unspecified lifetime.
    pub fn new(user_code: impl Into<String>, device_auth_id: impl Into<String>) -> Self {
        Self {
            user_code: user_code.into(),
            device_auth_id: device_auth_id.into(),
            interval: default_interval(),
            expires_in: 0,
        }
    }

    /// Overrides the polling interval.
    ///
    /// Unlike deserialization this applies no floor: it is the seam tests (and a
    /// caller driving the poll itself) use to shorten the wait, while
    /// [`OpenAiOAuthClient::poll_device_token`](super::OpenAiOAuthClient::poll_device_token)
    /// still enforces its own minimum delay.
    #[must_use]
    pub fn with_interval(mut self, interval: u64) -> Self {
        self.interval = interval;
        self
    }

    /// Overrides the advertised lifetime.
    #[must_use]
    pub fn with_expires_in(mut self, expires_in: u64) -> Self {
        self.expires_in = expires_in;
        self
    }
}

/// Persistence seam for the OAuth token.
///
/// [`InMemoryTokenStore`] is the testable default and [`FileTokenStore`] is the
/// production backend; callers may supply any other backend (a keyring, a
/// secret manager) without touching [`OpenAiOAuthClient`](super::OpenAiOAuthClient).
#[async_trait]
pub trait TokenStore: Send + Sync {
    /// Loads the stored token, or `None` when none was ever saved.
    async fn load(&self) -> Result<Option<OAuthToken>>;

    /// Stores `token`, replacing any previous value atomically.
    async fn save(&self, token: &OAuthToken) -> Result<()>;

    /// Removes the stored token, leaving other providers untouched.
    async fn clear(&self) -> Result<()>;
}

/// A [`TokenStore`] that keeps the token in process memory.
///
/// Nothing is written to disk, so this store is the natural choice for tests and
/// for short-lived processes that authenticate per run.
#[derive(Debug, Default)]
pub struct InMemoryTokenStore {
    token: Mutex<Option<OAuthToken>>,
}

impl InMemoryTokenStore {
    /// Builds an empty store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl TokenStore for InMemoryTokenStore {
    async fn load(&self) -> Result<Option<OAuthToken>> {
        Ok(self.token.lock().clone())
    }

    async fn save(&self, token: &OAuthToken) -> Result<()> {
        *self.token.lock() = Some(token.clone());
        Ok(())
    }

    async fn clear(&self) -> Result<()> {
        *self.token.lock() = None;
        Ok(())
    }
}

/// A [`TokenStore`] backed by the openJiuwen auth-store JSON file.
///
/// Only the `providers.OpenAIAccount` entry belongs to this store: other
/// providers' entries, and any unknown keys inside the auth store or inside the
/// `OpenAIAccount` entry, survive a write.
///
/// The file operations are small but blocking, so every method runs them on a
/// blocking worker (`tokio::task::spawn_blocking`) and the async caller never
/// stalls a runtime thread.
#[derive(Debug, Clone)]
pub struct FileTokenStore {
    path: PathBuf,
    base_url: String,
}

impl FileTokenStore {
    /// Builds a store for `path`, recording [`DEFAULT_BASE_URL`] in the envelope.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            base_url: DEFAULT_BASE_URL.to_owned(),
        }
    }

    /// Overrides the `base_url` recorded next to the tokens.
    #[must_use]
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        let base_url = base_url.into();
        base_url
            .trim()
            .trim_end_matches('/')
            .clone_into(&mut self.base_url);
        self
    }

    /// Returns the auth-store path.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Returns the `base_url` recorded next to the tokens.
    #[must_use]
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Resolves the default auth-store path from the environment.
    ///
    /// Precedence, matching the reference implementation:
    /// `$OPENJIUWEN_AUTH_FILE`, then `$OPENJIUWEN_HOME/auth.json`, then
    /// `~/.openjiuwen/auth.json` (`%USERPROFILE%` stands in for `$HOME` on
    /// Windows). A leading `~` is expanded.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] when none of the three sources resolves, so a
    /// caller can never silently fall back to a relative path that would put a
    /// credential file in the working directory.
    pub fn default_path() -> Result<PathBuf> {
        resolve_default_path()
    }
}

/// Resolves the default auth-store path; see [`FileTokenStore::default_path`].
fn resolve_default_path() -> Result<PathBuf> {
    if let Some(auth_file) = env::var(AUTH_FILE_ENV) {
        return Ok(expand_tilde(&auth_file));
    }
    if let Some(home) = env::var(env::OPENJIUWEN_HOME) {
        return Ok(expand_tilde(&home).join(AUTH_FILE_NAME));
    }
    if let Some(home) = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")) {
        let home = PathBuf::from(home);
        if !home.as_os_str().is_empty() {
            return Ok(home.join(DEFAULT_HOME_DIR).join(AUTH_FILE_NAME));
        }
    }
    Err(Error::config(format!(
        "cannot resolve the auth store path: set {AUTH_FILE_ENV}, {} , or HOME",
        env::OPENJIUWEN_HOME
    )))
}

/// Expands a leading `~` using `$HOME` (`%USERPROFILE%` on Windows).
fn expand_tilde(raw: &str) -> PathBuf {
    if raw == "~" || raw.starts_with("~/") {
        if let Some(home) = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")) {
            let home = PathBuf::from(home);
            if !home.as_os_str().is_empty() {
                return home.join(raw.trim_start_matches("~/"));
            }
        }
    }
    PathBuf::from(raw)
}

#[async_trait]
impl TokenStore for FileTokenStore {
    async fn load(&self) -> Result<Option<OAuthToken>> {
        let path = self.path.clone();
        tokio::task::spawn_blocking(move || load_token(&path))
            .await
            .map_err(Error::external)?
    }

    async fn save(&self, token: &OAuthToken) -> Result<()> {
        let path = self.path.clone();
        let base_url = self.base_url.clone();
        let token = token.clone();
        tokio::task::spawn_blocking(move || save_token(&path, &base_url, &token))
            .await
            .map_err(Error::external)?
    }

    async fn clear(&self) -> Result<()> {
        let path = self.path.clone();
        tokio::task::spawn_blocking(move || clear_token(&path))
            .await
            .map_err(Error::external)?
    }
}

/// Wraps an I/O failure with the path and action that failed.
fn store_io_error(action: &str, path: &Path, error: std::io::Error) -> Error {
    Error::external(std::io::Error::new(
        error.kind(),
        format!("{action} {}: {error}", path.display()),
    ))
}

/// Reads and parses the auth-store document, returning `None` when it is absent.
fn read_store(path: &Path) -> Result<Option<Value>> {
    let bytes = match std::fs::read(path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(store_io_error("reading the auth store", path, error)),
    };
    parse_store(path, &bytes).map(Some)
}

/// Parses an auth-store document, requiring a JSON object at the top level.
fn parse_store(path: &Path, bytes: &[u8]) -> Result<Value> {
    let value: Value = serde_json::from_slice(bytes).map_err(|error| {
        Error::validation(format!(
            "auth store {} is not valid JSON: {error}",
            path.display()
        ))
    })?;
    if !value.is_object() {
        return Err(Error::validation(format!(
            "auth store {} must be a JSON object",
            path.display()
        )));
    }
    Ok(value)
}

/// Returns the `providers.OpenAIAccount` entry, if the store has one.
fn provider_state(store: &Value) -> Option<&Value> {
    store.get("providers")?.get(PROVIDER_KEY)
}

/// Loads the token from the auth store at `path`.
fn load_token(path: &Path) -> Result<Option<OAuthToken>> {
    let Some(store) = read_store(path)? else {
        return Ok(None);
    };
    let Some(tokens) = provider_state(&store).and_then(|entry| entry.get("tokens")) else {
        return Ok(None);
    };
    if !tokens.is_object() {
        return Err(Error::validation(format!(
            "auth store {} has a non-object `{PROVIDER_KEY}.tokens` entry",
            path.display()
        )));
    }
    serde_json::from_value::<OAuthToken>(tokens.clone())
        .map(Some)
        .map_err(|error| {
            Error::validation(format!(
                "auth store {} has an unreadable `{PROVIDER_KEY}` token entry: {error}",
                path.display()
            ))
        })
}

/// Merges `token` into the auth store at `path`, preserving every other entry.
fn save_token(path: &Path, base_url: &str, token: &OAuthToken) -> Result<()> {
    let mut store = match read_store(path)? {
        Some(store) => store,
        None => Value::Object(Map::new()),
    };
    let root = store.as_object_mut().ok_or_else(|| {
        Error::validation(format!(
            "auth store {} must be a JSON object",
            path.display()
        ))
    })?;
    let providers = root
        .entry("providers".to_owned())
        .or_insert_with(|| Value::Object(Map::new()))
        .as_object_mut()
        .ok_or_else(|| {
            Error::validation(format!(
                "auth store {} has a non-object `providers` entry",
                path.display()
            ))
        })?;
    let entry = providers
        .entry(PROVIDER_KEY.to_owned())
        .or_insert_with(|| Value::Object(Map::new()))
        .as_object_mut()
        .ok_or_else(|| {
            Error::validation(format!(
                "auth store {} has a non-object `{PROVIDER_KEY}` entry",
                path.display()
            ))
        })?;
    let tokens = serde_json::to_value(token).map_err(|error| {
        Error::validation(format!(
            "serializing the `{PROVIDER_KEY}` token entry: {error}"
        ))
    })?;
    entry.insert("tokens".to_owned(), tokens);
    entry.insert("auth_mode".to_owned(), Value::String(AUTH_MODE.to_owned()));
    entry.insert("base_url".to_owned(), Value::String(base_url.to_owned()));
    write_store(path, &store)
}

/// Removes only the `providers.OpenAIAccount` entry from the auth store.
fn clear_token(path: &Path) -> Result<()> {
    let Some(mut store) = read_store(path)? else {
        return Ok(());
    };
    let removed = store
        .get_mut("providers")
        .and_then(Value::as_object_mut)
        .and_then(|providers| providers.remove(PROVIDER_KEY))
        .is_some();
    if !removed {
        return Ok(());
    }
    write_store(path, &store)
}

/// Serializes `store` and replaces the file at `path` with it atomically.
fn write_store(path: &Path, store: &Value) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).map_err(|error| {
                store_io_error("creating the parent directory of", parent, error)
            })?;
        }
    }
    let temporary = temporary_path(path);
    let mut rendered = serde_json::to_vec_pretty(store)
        .map_err(|error| Error::validation(format!("serializing the auth store: {error}")))?;
    rendered.push(b'\n');
    write_file(&temporary, &rendered)?;
    #[cfg(unix)]
    {
        restrict_permissions(&temporary)?;
    }
    std::fs::rename(&temporary, path)
        .map_err(|error| store_io_error("replacing the auth store", path, error))?;
    #[cfg(unix)]
    {
        restrict_permissions(path)?;
    }
    Ok(())
}

/// Writes `bytes` to `path`, flushing them to the filesystem before returning.
fn write_file(path: &Path, bytes: &[u8]) -> Result<()> {
    use std::io::Write;

    let mut file =
        std::fs::File::create(path).map_err(|error| store_io_error("creating", path, error))?;
    file.write_all(bytes)
        .map_err(|error| store_io_error("writing", path, error))?;
    file.sync_all()
        .map_err(|error| store_io_error("flushing", path, error))
}

/// Returns the sibling temporary path used for the atomic replace.
fn temporary_path(path: &Path) -> PathBuf {
    let name = path.file_name().map_or_else(
        || AUTH_FILE_NAME.to_owned(),
        |name| name.to_string_lossy().into_owned(),
    );
    path.with_file_name(format!("{name}.{}.tmp", std::process::id()))
}

/// Restricts `path` to its owner (`0o600`).
#[cfg(unix)]
fn restrict_permissions(path: &Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;

    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
        .map_err(|error| store_io_error("restricting the permissions of", path, error))
}
