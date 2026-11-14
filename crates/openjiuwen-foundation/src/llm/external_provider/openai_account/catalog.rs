//! `OpenAIAccount` model catalog: live discovery, disk cache, built-in defaults.
//!
//! Ports `openjiuwen/extensions/external_provider/openai_auth/openai_account_models.py`.
//! [`OpenAiAccountModelCatalog::list_model_ids`] tries the live endpoint first,
//! then the on-disk cache, then [`DEFAULT_MODELS`], matching the reference
//! implementation's precedence:
//!
//! 1. resolve a token from the request (an explicit token wins, otherwise the
//!    supplied auth manager resolves — and possibly refreshes — one);
//! 2. `GET {base_url}/models?client_version=1.0.0` with the bearer token;
//! 3. on any discovery failure, read the cache file;
//! 4. with no usable cache, answer with the built-in defaults.
//!
//! Discovery failures and cache I/O failures are logged at `debug` and degrade to
//! the next fallback: a catalog lookup must not fail because the network is down
//! (Python parity — `list_model_ids` swallows
//! `OpenAIAccountModelListError` and `OSError`). The individual operations
//! ([`fetch_models`](OpenAiAccountModelCatalog::fetch_models),
//! [`write_cache`](OpenAiAccountModelCatalog::write_cache)) return typed errors so
//! a direct caller sees what went wrong.
//!
//! The cache document is the reference implementation's shape:
//! `{provider, base_url, client_version, fetched_at, model_ids, payload}`, written
//! through a sibling `tmp` file and an atomic rename.
//! There is no cross-process lock (the same limitation
//! [`crate::llm::oauth`] documents): concurrent writers are last-writer-wins.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::time::Duration;

use chrono::Utc;
use openjiuwen_common::config::env;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::http::{HttpClient, HttpRequest};
use openjiuwen_common::security::SecretString;
use serde_json::{Map, Value};

use crate::llm::oauth::FileTokenStore;

use super::super::base::{ExternalProviderConfig, ModelCatalogRequest, ProviderModelCatalog};
use super::{
    BASE_URL_ENV, DEFAULT_BASE_URL, DEFAULT_MODELS, MODELS_CACHE_FILENAME, MODELS_CACHE_FILE_ENV,
    MODELS_CLIENT_VERSION, PROVIDER_NAME,
};

/// Request timeout used when the configuration does not set one, in seconds.
pub const DEFAULT_TIMEOUT_SECONDS: f64 = 10.0;

/// Shortest timeout the catalog applies, in seconds.
///
/// The reference implementation clamps with `max(5.0, timeout_seconds)`, so a
/// smaller configured value is raised instead of rejected.
const MIN_TIMEOUT_SECONDS: f64 = 5.0;

/// Priority assumed for a model entry that carries none (`10_000`).
const DEFAULT_PRIORITY: i64 = 10_000;

/// Visibility values that hide a model (`{"hide", "hidden"}`).
const HIDDEN_VISIBILITIES: [&str; 2] = ["hide", "hidden"];

/// Models added for forward compatibility, with the models that imply them.
const FORWARD_COMPAT_TEMPLATES: [(&str, &[&str]); 4] = [
    (
        "gpt-5.5",
        &["gpt-5.4", "gpt-5.4-mini", "gpt-5.3-codex", "gpt-5-codex"],
    ),
    ("gpt-5.4-mini", &["gpt-5.3-codex", "gpt-5-codex"]),
    ("gpt-5.4", &["gpt-5.3-codex", "gpt-5-codex"]),
    ("gpt-5.3-codex-spark", &["gpt-5.3-codex"]),
];

/// Clock used for the cache's `fetched_at`.
///
/// A fixed clock is a test seam that makes the cache payload deterministic;
/// production code uses [`Clock::System`].
#[derive(Debug, Clone, Copy)]
enum Clock {
    /// Wall-clock time.
    System,
    /// A fixed number of epoch seconds.
    Fixed(f64),
}

impl Clock {
    /// Returns the current time as fractional epoch seconds.
    fn now_seconds(self) -> f64 {
        match self {
            Self::System => Utc::now().timestamp_millis() as f64 / 1000.0,
            Self::Fixed(seconds) => seconds,
        }
    }
}

/// Fetches and caches the `OpenAIAccount` backend model list.
///
/// Construction resolves the account base URL (config →
/// `$OPENJIUWEN_OPENAI_ACCOUNT_BASE_URL` → [`DEFAULT_BASE_URL`]) and the cache
/// path (config → `$OPENJIUWEN_OPENAI_ACCOUNT_MODELS_CACHE_FILE` → the auth-store
/// directory plus [`MODELS_CACHE_FILENAME`]).
pub struct OpenAiAccountModelCatalog {
    /// Account API base URL, without a trailing slash.
    base_url: String,
    /// Cache file the fallback reads and the live path writes.
    cache_path: PathBuf,
    /// Per-request timeout applied to the model-list call.
    timeout: Duration,
    /// `client_version` query value.
    client_version: String,
    /// HTTP stack shared with the rest of the crate.
    http: HttpClient,
    /// Clock used for `fetched_at`.
    clock: Clock,
}

impl OpenAiAccountModelCatalog {
    /// Builds a catalog from the environment and built-in defaults.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] when no cache path can be resolved and the
    /// errors of [`HttpClient::with_defaults`].
    pub fn new() -> Result<Self> {
        Self::with_config(&ExternalProviderConfig::default())
    }

    /// Builds a catalog from a typed provider configuration.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] when no cache path can be resolved (no
    /// `OPENJIUWEN_OPENAI_ACCOUNT_MODELS_CACHE_FILE`, no auth-store path, and no
    /// home directory), [`Error::Validation`] for a non-finite or non-positive
    /// timeout, and the errors of [`HttpClient::with_defaults`].
    pub fn with_config(config: &ExternalProviderConfig) -> Result<Self> {
        let base_url = resolve_base_url(config.base_url.as_deref());
        let cache_path = match &config.cache_path {
            Some(path) => expand_tilde(path),
            None => default_cache_path()?,
        };
        let timeout = resolve_timeout(config.timeout)?;
        Ok(Self {
            base_url,
            cache_path,
            timeout,
            client_version: MODELS_CLIENT_VERSION.to_owned(),
            http: HttpClient::with_defaults()?,
            clock: Clock::System,
        })
    }

    /// Replaces the HTTP client (retry policy, proxies, TLS).
    #[must_use]
    pub fn with_http_client(mut self, http: HttpClient) -> Self {
        self.http = http;
        self
    }

    /// Overrides the account API base URL.
    #[must_use]
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        let base_url = base_url.into();
        base_url
            .trim()
            .trim_end_matches('/')
            .clone_into(&mut self.base_url);
        self
    }

    /// Overrides the cache file.
    #[must_use]
    pub fn with_cache_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.cache_path = path.into();
        self
    }

    /// Overrides the `client_version` query value.
    #[must_use]
    pub fn with_client_version(mut self, client_version: impl Into<String>) -> Self {
        self.client_version = client_version.into();
        self
    }

    /// Overrides the per-request timeout.
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Pins the cache clock, making `fetched_at` deterministic.
    ///
    /// A test seam; production use keeps the system clock.
    #[must_use]
    pub fn with_fixed_time(mut self, epoch_seconds: f64) -> Self {
        self.clock = Clock::Fixed(epoch_seconds);
        self
    }

    /// Returns the account API base URL.
    #[must_use]
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Returns the cache file path.
    #[must_use]
    pub fn cache_path(&self) -> &Path {
        &self.cache_path
    }

    /// Returns the `client_version` query value.
    #[must_use]
    pub fn client_version(&self) -> &str {
        &self.client_version
    }

    /// Returns the per-request timeout.
    #[must_use]
    pub fn timeout(&self) -> Duration {
        self.timeout
    }

    /// Returns the model IDs, trying live discovery, then the cache, then defaults.
    ///
    /// Never fails for the built-in catalog: every discovery or cache problem
    /// degrades to the next fallback (logged at `debug`).
    ///
    /// # Errors
    ///
    /// Reserved for a future catalog whose fallback chain cannot answer; the
    /// built-in implementation always answers `Ok`.
    pub async fn list_model_ids(&self, request: &ModelCatalogRequest) -> Result<Vec<String>> {
        let token = match &request.access_token {
            Some(token) if !token.expose_secret().trim().is_empty() => Some(token.clone()),
            Some(_) => None,
            None => match &request.auth_manager {
                Some(manager) => match manager.resolve_access_token(request.force_refresh).await {
                    Ok(token) if !token.expose_secret().trim().is_empty() => Some(token),
                    Ok(_) => None,
                    Err(error) => {
                        tracing::debug!(
                            target: "openjiuwen_foundation::llm::external_provider",
                            error = %error,
                            "resolving the OpenAI account model-discovery token failed"
                        );
                        None
                    }
                },
                None => None,
            },
        };

        if let Some(token) = token {
            match self.fetch_models(&token).await {
                Ok((payload, model_ids)) => {
                    if let Err(error) = self.write_cache(&payload, &model_ids).await {
                        // Python swallows the cache-write `OSError` as well: the
                        // live answer is already valid, only persistence failed.
                        tracing::debug!(
                            target: "openjiuwen_foundation::llm::external_provider",
                            error = %error,
                            "writing the OpenAI account model cache failed"
                        );
                    }
                    return Ok(model_ids);
                }
                Err(error) => {
                    tracing::debug!(
                        target: "openjiuwen_foundation::llm::external_provider",
                        error = %error,
                        "OpenAI account model discovery failed; falling back"
                    );
                }
            }
        }

        if let Some(cached) = self.read_cached().await {
            if !cached.is_empty() {
                return Ok(cached);
            }
        }
        Ok(forward_compat_models(
            DEFAULT_MODELS.iter().map(|model| (*model).to_owned()).collect(),
        ))
    }

    /// Fetches the live model list, returning the raw payload and parsed IDs.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Llm`] when the token is empty, the endpoint answers a
    /// non-`200` status, the body is not valid JSON, the payload is not an
    /// object, or no visible model remains; transport failures propagate from
    /// [`HttpClient`].
    pub async fn fetch_models(&self, access_token: &SecretString) -> Result<(Value, Vec<String>)> {
        if access_token.expose_secret().trim().is_empty() {
            return Err(Error::llm(
                "OpenAI account model discovery requires an access token.",
            ));
        }
        let url = format!("{}/models", self.base_url);
        let request = HttpRequest::get(url.clone())
            .query("client_version", self.client_version.clone())
            .header("accept", "application/json")
            .bearer(access_token.clone())
            .timeout(self.timeout);
        let response = self.http.execute(request).await?;
        if response.status != 200 {
            return Err(Error::llm(format!(
                "OpenAI account model discovery failed with status {} for {url}.",
                response.status
            )));
        }
        let payload: Value = serde_json::from_slice(response.bytes()).map_err(|error| {
            Error::llm(format!(
                "OpenAI account model discovery returned invalid JSON from {url}: {error}"
            ))
        })?;
        if !payload.is_object() {
            return Err(Error::llm(format!(
                "OpenAI account model discovery returned an unsupported payload from {url}."
            )));
        }
        let model_ids = parse_model_ids(&payload);
        if model_ids.is_empty() {
            return Err(Error::llm(format!(
                "OpenAI account model discovery returned no visible models from {url}."
            )));
        }
        Ok((payload, model_ids))
    }

    /// Reads the cached model IDs, if the cache file holds any.
    ///
    /// Returns `None` when the file is absent, unreadable, not JSON, or not an
    /// object; an empty cache document yields `Some` of an empty list (Python's
    /// `read_cache_model_ids` returns `[]` for both cases and the caller filters
    /// empties).
    pub async fn read_cached(&self) -> Option<Vec<String>> {
        let bytes = tokio::fs::read(&self.cache_path).await.ok()?;
        let payload: Value = serde_json::from_slice(&bytes).ok()?;
        if !payload.is_object() {
            return None;
        }
        let cached = payload
            .get("model_ids")
            .map(model_ids_from_sequence)
            .unwrap_or_default();
        if !cached.is_empty() {
            return Some(forward_compat_models(cached));
        }
        if let Some(inner) = payload.get("payload").filter(|value| value.is_object()) {
            return Some(parse_model_ids(inner));
        }
        Some(parse_model_ids(&payload))
    }

    /// Writes the cache document through a `tmp` file and an atomic rename.
    ///
    /// Does nothing when `model_ids` is empty (Python parity). The stored
    /// `model_ids` include the forward-compatibility additions, while `payload`
    /// keeps the raw endpoint response.
    ///
    /// # Errors
    ///
    /// Returns [`Error::External`] when the parent directory cannot be created,
    /// the `tmp` file cannot be written, or the rename fails, and
    /// [`Error::Validation`] when the document cannot be serialized.
    pub async fn write_cache(&self, payload: &Value, model_ids: &[String]) -> Result<()> {
        if model_ids.is_empty() {
            return Ok(());
        }
        let document = cache_document(
            &self.base_url,
            &self.client_version,
            self.clock.now_seconds(),
            model_ids,
            payload,
        );
        let mut rendered = serde_json::to_vec_pretty(&document)
            .map_err(|error| Error::validation(format!("serializing the model cache: {error}")))?;
        rendered.push(b'\n');
        if let Some(parent) = self.cache_path.parent() {
            if !parent.as_os_str().is_empty() {
                tokio::fs::create_dir_all(parent).await.map_err(|error| {
                    cache_io_error("creating the parent directory of", self.cache_path.as_path(), error)
                })?;
            }
        }
        let temporary = temporary_path(&self.cache_path);
        tokio::fs::write(&temporary, &rendered)
            .await
            .map_err(|error| cache_io_error("writing", temporary.as_path(), error))?;
        tokio::fs::rename(&temporary, &self.cache_path)
            .await
            .map_err(|error| cache_io_error("replacing", self.cache_path.as_path(), error))
    }
}

#[async_trait::async_trait]
impl ProviderModelCatalog for OpenAiAccountModelCatalog {
    async fn list_model_ids(&self, request: &ModelCatalogRequest) -> Result<Vec<String>> {
        OpenAiAccountModelCatalog::list_model_ids(self, request).await
    }
}

impl std::fmt::Debug for OpenAiAccountModelCatalog {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OpenAiAccountModelCatalog")
            .field("base_url", &self.base_url)
            .field("cache_path", &self.cache_path)
            .field("timeout", &self.timeout)
            .field("client_version", &self.client_version)
            .field("clock", &self.clock)
            .finish_non_exhaustive()
    }
}

/// Resolves the default cache path: `$OPENJIUWEN_OPENAI_ACCOUNT_MODELS_CACHE_FILE`,
/// then the auth-store directory plus [`MODELS_CACHE_FILENAME`].
///
/// # Errors
///
/// Returns [`Error::Config`] when neither the environment variable nor the
/// auth-store path resolves.
pub fn default_cache_path() -> Result<PathBuf> {
    if let Some(raw) = env::var(MODELS_CACHE_FILE_ENV) {
        return Ok(expand_tilde(Path::new(&raw)));
    }
    let auth_path = FileTokenStore::default_path()?;
    let directory = auth_path.parent().unwrap_or_else(|| Path::new(""));
    Ok(directory.join(MODELS_CACHE_FILENAME))
}

/// Extracts visible model IDs from a live or cached payload.
///
/// Applies the reference implementation's rules exactly:
///
/// - entries come from `models` (list, or object keyed by model ID), falling back
///   to `data` in the same two shapes; a non-object entry is skipped;
/// - an entry whose `visibility` is `hide`/`hidden` (case-insensitive, trimmed)
///   is skipped;
/// - the ID is the first non-empty trimmed string among `slug`, `id`, `name`,
///   `model`;
/// - entries are sorted by `(priority, id)` with `priority` defaulting to
///   `10_000` when absent, `null`, boolean, or non-numeric;
/// - duplicates are dropped preserving the sorted order;
/// - forward-compatibility models are appended (see
///   [`forward_compat_models`]).
#[must_use]
pub fn parse_model_ids(payload: &Value) -> Vec<String> {
    let mut sortable: Vec<(i64, String)> = Vec::new();
    for entry in model_entries(payload) {
        let Some(item) = entry.as_object() else {
            continue;
        };
        if is_hidden(item.get("visibility")) {
            continue;
        }
        let Some(model_id) = model_id(item) else {
            continue;
        };
        sortable.push((priority(item.get("priority")), model_id));
    }
    sortable.sort_by(|left, right| left.0.cmp(&right.0).then_with(|| left.1.cmp(&right.1)));

    let mut seen = BTreeSet::new();
    let mut ordered = Vec::new();
    for (_, model_id) in sortable {
        if seen.insert(model_id.clone()) {
            ordered.push(model_id);
        }
    }
    forward_compat_models(ordered)
}

/// Appends the forward-compatibility models the reference implementation adds.
///
/// Deduplicates first, then appends each synthetic model whose template set
/// intersects the visible IDs, in the template table's order. Both
/// [`parse_model_ids`] and the built-in-defaults and cache paths use this, so a
/// model set stored by an older SDK still exposes the newer aliases.
#[must_use]
pub fn forward_compat_models(model_ids: Vec<String>) -> Vec<String> {
    let mut ordered: Vec<String> = Vec::new();
    let mut seen: BTreeSet<String> = BTreeSet::new();
    for model_id in model_ids {
        if seen.insert(model_id.clone()) {
            ordered.push(model_id);
        }
    }
    for (synthetic, templates) in FORWARD_COMPAT_TEMPLATES {
        if seen.contains(synthetic) {
            continue;
        }
        if templates.iter().any(|template| seen.contains(*template)) {
            ordered.push(synthetic.to_owned());
            seen.insert(synthetic.to_owned());
        }
    }
    ordered
}

/// Returns the raw model entries of `payload` as owned JSON objects.
///
/// `models` is consulted first, then `data`; a list is returned verbatim, and an
/// object is turned into one entry per value with the key used as a fallback
/// `slug` (Python's `_dict_model_entry`).
fn model_entries(payload: &Value) -> Vec<Value> {
    for key in ["models", "data"] {
        let Some(value) = payload.get(key) else {
            continue;
        };
        if let Some(items) = value.as_array() {
            return items.clone();
        }
        if let Some(map) = value.as_object() {
            return map
                .iter()
                .map(|(model_id, metadata)| {
                    let mut entry = metadata.as_object().cloned().unwrap_or_default();
                    entry
                        .entry("slug".to_owned())
                        .or_insert_with(|| Value::String(model_id.clone()));
                    Value::Object(entry)
                })
                .collect();
        }
    }
    Vec::new()
}

/// Returns `true` when `visibility` marks the entry hidden.
fn is_hidden(visibility: Option<&Value>) -> bool {
    visibility
        .and_then(Value::as_str)
        .map(|value| value.trim().to_lowercase())
        .is_some_and(|value| HIDDEN_VISIBILITIES.contains(&value.as_str()))
}

/// Returns the model ID an entry advertises, checking `slug`, `id`, `name`, then
/// `model`.
fn model_id(item: &Map<String, Value>) -> Option<String> {
    ["slug", "id", "name", "model"]
        .iter()
        .filter_map(|key| item.get(*key).and_then(Value::as_str))
        .map(str::trim)
        .find(|value| !value.is_empty())
        .map(str::to_owned)
}

/// Returns the entry's sort priority, defaulting to [`DEFAULT_PRIORITY`].
///
/// Python uses `int(value)` and falls back on `(TypeError, ValueError)`; the Rust
/// equivalent accepts integers, truncates floats, parses numeric strings, and
/// treats booleans and everything else as the default.
fn priority(value: Option<&Value>) -> i64 {
    match value {
        Some(Value::Number(number)) => number
            .as_i64()
            .or_else(|| number.as_f64().map(|value| value.trunc() as i64))
            .unwrap_or(DEFAULT_PRIORITY),
        Some(Value::String(text)) => text.trim().parse::<i64>().unwrap_or(DEFAULT_PRIORITY),
        _ => DEFAULT_PRIORITY,
    }
}

/// Parses a cache `model_ids` array: trimmed non-empty strings, deduplicated.
fn model_ids_from_sequence(value: &Value) -> Vec<String> {
    let Some(items) = value.as_array() else {
        return Vec::new();
    };
    let mut seen = BTreeSet::new();
    let mut model_ids = Vec::new();
    for item in items {
        let Some(text) = item.as_str() else {
            continue;
        };
        let model_id = text.trim();
        if model_id.is_empty() || !seen.insert(model_id.to_owned()) {
            continue;
        }
        model_ids.push(model_id.to_owned());
    }
    model_ids
}

/// Builds the cache document.
fn cache_document(
    base_url: &str,
    client_version: &str,
    fetched_at: f64,
    model_ids: &[String],
    payload: &Value,
) -> Value {
    let mut document = Map::new();
    document.insert(
        "provider".to_owned(),
        Value::String(PROVIDER_NAME.to_owned()),
    );
    document.insert("base_url".to_owned(), Value::String(base_url.to_owned()));
    document.insert(
        "client_version".to_owned(),
        Value::String(client_version.to_owned()),
    );
    document.insert("fetched_at".to_owned(), serde_json::json!(fetched_at));
    document.insert(
        "model_ids".to_owned(),
        Value::Array(
            forward_compat_models(model_ids.to_vec())
                .into_iter()
                .map(Value::String)
                .collect(),
        ),
    );
    document.insert("payload".to_owned(), payload.clone());
    Value::Object(document)
}

/// Resolves the account base URL: config, then `$OPENJIUWEN_OPENAI_ACCOUNT_BASE_URL`,
/// then [`DEFAULT_BASE_URL`].
fn resolve_base_url(configured: Option<&str>) -> String {
    configured
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
        .or_else(|| {
            env::var(BASE_URL_ENV)
                .map(|value| value.trim().to_owned())
                .filter(|value| !value.is_empty())
        })
        .unwrap_or_else(|| DEFAULT_BASE_URL.to_owned())
        .trim_end_matches('/')
        .to_owned()
}

/// Resolves the request timeout, applying the reference implementation's 5 s
/// floor.
fn resolve_timeout(configured: Option<f64>) -> Result<Duration> {
    let seconds = configured.unwrap_or(DEFAULT_TIMEOUT_SECONDS);
    if !seconds.is_finite() || seconds <= 0.0 {
        return Err(Error::validation(
            "OpenAI account model-catalog timeout must be a positive number of seconds",
        ));
    }
    let seconds = seconds.max(MIN_TIMEOUT_SECONDS);
    Duration::try_from_secs_f64(seconds).map_err(|error| {
        Error::validation(format!(
            "OpenAI account model-catalog timeout is out of range: {error}"
        ))
    })
}

/// Expands a leading `~` in `path` using `$HOME`/`%USERPROFILE%`.
fn expand_tilde(path: &Path) -> PathBuf {
    let raw = path.to_string_lossy();
    if raw == "~" || raw.starts_with("~/") {
        if let Some(home) = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")) {
            let home = PathBuf::from(home);
            if !home.as_os_str().is_empty() {
                return home.join(raw.trim_start_matches("~/"));
            }
        }
    }
    path.to_path_buf()
}

/// Returns the sibling `tmp` path used for the atomic replace.
fn temporary_path(path: &Path) -> PathBuf {
    let name = path
        .file_name()
        .map_or_else(|| MODELS_CACHE_FILENAME.to_owned(), |name| name.to_string_lossy().into_owned());
    path.with_file_name(format!("{name}.tmp"))
}

/// Wraps a cache I/O failure with the path and action that failed.
fn cache_io_error(action: &str, path: &Path, error: std::io::Error) -> Error {
    Error::external(std::io::Error::new(
        error.kind(),
        format!("{action} {}: {error}", path.display()),
    ))
}
