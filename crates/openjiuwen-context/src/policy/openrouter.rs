//! `OpenRouter` context-window cache and catalogue fetch.
//!
//! Ports `ContextUtils.fetch_openrouter_model_context_window_tokens` and the
//! process-wide `_OPENROUTER_MODEL_CONTEXT_WINDOW_TOKENS` cache from
//! `context/context_utils.py`.
//!
//! `OpenRouter` publishes one catalogue (`https://openrouter.ai/api/v1/models`)
//! whose entries carry a `context_length`; downloading it is the only way to
//! learn the window of a model the built-in table does not know. Because the
//! download is a large cross-region HTTPS request, Python splits it in two:
//!
//! - [`ContextWindowPolicy::fetch`] performs the request (bounded by
//!   `openrouter_request_timeout`) and fills the cache;
//! - [`ContextWindowPolicy::prefetch`] does the same with a caller-supplied
//!   [`HttpClient`] and skips the request when every requested model is already
//!   cached — the shape a background warm-up wants;
//! - `resolve` never fetches: it only reads the cache, so creating a context
//!   can never block on the network.
//!
//! Fetches are opt-in (`enable_openrouter_model_context_window_tokens`); while
//! the flag is off both methods return `Ok(0)` without touching the network.
//!
//! # Process-global state
//!
//! [`global_openrouter_window_cache`] is the one piece of process-global mutable
//! state in this module. It exists so a warm cache is shared by every context
//! and every agent in the process instead of being re-downloaded per context.
//! The cache is a `RwLock` read only for short lookups and written only on
//! insert/clear; no lock is ever held across an `await`. Tests and multi-tenant
//! hosts should inject their own handle through
//! [`ContextWindowPolicy::with_openrouter_cache`].
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{OnceLock, PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::http::{HttpClient, HttpClientConfig, HttpRequest};

use crate::policy::window::ContextWindowPolicy;

/// The public `OpenRouter` model catalogue.
pub const OPENROUTER_MODELS_URL: &str = "https://openrouter.ai/api/v1/models";

/// Default lifetime of a cached context window (6 hours).
pub const DEFAULT_OPENROUTER_CACHE_TTL: Duration = Duration::from_secs(6 * 60 * 60);

/// Default per-request deadline for the catalogue fetch.
pub const DEFAULT_OPENROUTER_REQUEST_TIMEOUT: Duration = Duration::from_secs(3);

/// Error code for a transport, HTTP-status, or deadline failure.
pub const OPENROUTER_UNAVAILABLE_CODE: &str = "openrouter_unavailable";

/// Error code for a payload that is not a usable model catalogue.
pub const OPENROUTER_INVALID_RESPONSE_CODE: &str = "openrouter_invalid_response";

/// Error code for a strict caller asking for a fetch while the policy has the
/// feature disabled.
pub const OPENROUTER_DISABLED_CODE: &str = "openrouter_disabled";

/// Builds the compact transport error for a failed fetch.
pub fn openrouter_unavailable_error(detail: impl std::fmt::Display) -> Error {
    Error::transport(format!("{OPENROUTER_UNAVAILABLE_CODE}: {detail}"))
}

/// Builds the compact validation error for an unusable catalogue payload.
pub fn openrouter_invalid_response_error(detail: impl std::fmt::Display) -> Error {
    Error::validation(format!("{OPENROUTER_INVALID_RESPONSE_CODE}: {detail}"))
}

/// Builds the compact config error for a disabled fetch.
///
/// [`ContextWindowPolicy::fetch`] and [`ContextWindowPolicy::prefetch`] return
/// `Ok(0)` when the feature is off; this helper is for a strict caller that must
/// distinguish "disabled" from "found nothing".
pub fn openrouter_disabled_error() -> Error {
    Error::config(format!(
        "{OPENROUTER_DISABLED_CODE}: enable_openrouter_model_context_window_tokens is false"
    ))
}

/// Normalizes a model id into a cache key: trimmed and lowercased.
fn normalize_key(model: &str) -> Option<String> {
    let model = model.trim();
    if model.is_empty() {
        None
    } else {
        Some(model.to_lowercase())
    }
}

#[derive(Debug, Clone, Copy)]
struct CacheEntry {
    tokens: usize,
    stored_at: Instant,
}

impl CacheEntry {
    fn is_fresh(&self, ttl: Duration) -> bool {
        !ttl.is_zero() && self.stored_at.elapsed() < ttl
    }
}

#[derive(Debug)]
struct CacheState {
    entries: BTreeMap<String, CacheEntry>,
    ttl: Duration,
}

impl CacheState {
    /// Drops expired entries; called by every accessor so `len`/`snapshot` never
    /// report stale values.
    fn purge_expired(&mut self) {
        let ttl = self.ttl;
        self.entries.retain(|_, entry| entry.is_fresh(ttl));
    }
}

/// A TTL-bounded map of model id to context window, shared process-wide.
///
/// Keys are lowercased model ids, so lookups are case-insensitive. A zero TTL
/// makes every entry stale immediately, which is the deterministic way to
/// exercise expiry in tests; [`OpenRouterWindowCache::set_ttl`] restores normal
/// behaviour.
#[derive(Debug)]
pub struct OpenRouterWindowCache {
    state: RwLock<CacheState>,
}

impl Default for OpenRouterWindowCache {
    fn default() -> Self {
        Self::new()
    }
}

impl OpenRouterWindowCache {
    /// Builds an empty cache with [`DEFAULT_OPENROUTER_CACHE_TTL`].
    #[must_use]
    pub fn new() -> Self {
        Self::with_ttl(DEFAULT_OPENROUTER_CACHE_TTL)
    }

    /// Builds an empty cache with an explicit TTL.
    #[must_use]
    pub fn with_ttl(ttl: Duration) -> Self {
        Self {
            state: RwLock::new(CacheState {
                entries: BTreeMap::new(),
                ttl,
            }),
        }
    }

    fn read_state(&self) -> RwLockReadGuard<'_, CacheState> {
        self.state.read().unwrap_or_else(PoisonError::into_inner)
    }

    fn write_state(&self) -> RwLockWriteGuard<'_, CacheState> {
        self.state.write().unwrap_or_else(PoisonError::into_inner)
    }

    /// Inserts one window under a lowercased key.
    ///
    /// Empty model ids and zero token counts are ignored: they cannot describe a
    /// usable window, and storing them would shadow a better value.
    pub fn insert(&self, model: impl AsRef<str>, context_length: usize) {
        if context_length == 0 {
            return;
        }
        let Some(key) = normalize_key(model.as_ref()) else {
            return;
        };
        let mut state = self.write_state();
        state.entries.insert(
            key,
            CacheEntry {
                tokens: context_length,
                stored_at: Instant::now(),
            },
        );
    }

    /// Returns the cached window for `model`, or `None` when absent or stale.
    #[must_use]
    pub fn get(&self, model: &str) -> Option<usize> {
        let key = normalize_key(model)?;
        let mut state = self.write_state();
        let ttl = state.ttl;
        match state
            .entries
            .get(&key)
            .map(|entry| (entry.tokens, entry.is_fresh(ttl)))
        {
            Some((tokens, true)) => Some(tokens),
            Some((_, false)) => {
                state.entries.remove(&key);
                None
            }
            None => None,
        }
    }

    /// Whether `model` has a fresh entry.
    #[must_use]
    pub fn contains(&self, model: &str) -> bool {
        self.get(model).is_some()
    }

    /// Drops every entry.
    pub fn clear(&self) {
        self.write_state().entries.clear();
    }

    /// Number of fresh entries.
    #[must_use]
    pub fn len(&self) -> usize {
        let mut state = self.write_state();
        state.purge_expired();
        state.entries.len()
    }

    /// Whether the cache holds no fresh entry.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Sets the TTL for existing and future entries.
    pub fn set_ttl(&self, ttl: Duration) {
        let mut state = self.write_state();
        state.ttl = ttl;
        state.purge_expired();
    }

    /// The current TTL.
    #[must_use]
    pub fn ttl(&self) -> Duration {
        self.read_state().ttl
    }

    /// A copy of the fresh entries, keyed by lowercased model id.
    #[must_use]
    pub fn snapshot(&self) -> BTreeMap<String, usize> {
        let mut state = self.write_state();
        state.purge_expired();
        state
            .entries
            .iter()
            .map(|(model, entry)| (model.clone(), entry.tokens))
            .collect()
    }
}

/// The process-wide cache shared by every default policy.
///
/// See the module docs for why this global exists and how to avoid it.
#[must_use]
pub fn global_openrouter_window_cache() -> &'static OpenRouterWindowCache {
    static CACHE: OnceLock<OpenRouterWindowCache> = OnceLock::new();
    CACHE.get_or_init(OpenRouterWindowCache::new)
}

/// The model windows parsed out of one `OpenRouter` catalogue payload.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct OpenRouterModelTable {
    /// Lowercased model id (plus every unambiguous provider-less alias) mapped
    /// to its context window in tokens.
    pub windows: BTreeMap<String, usize>,
    /// Number of model records that carried a usable `context_length`; it
    /// differs from [`OpenRouterModelTable::windows`]' length because of the
    /// aliases and because duplicate ids collapse.
    pub models: usize,
}

/// Parses an `OpenRouter` catalogue payload.
///
/// The payload is `{"data": [{"id": "openai/gpt-4o", "context_length": 128000},
/// ...]}`. Records without a string `id` or without a positive integer
/// `context_length` are skipped; a missing/non-array `data` is a typed
/// [`OPENROUTER_INVALID_RESPONSE_CODE`] error because the endpoint answered with
/// something else entirely.
///
/// Python also adds the provider-less form of every id as an alias
/// (`openai/gpt-4o` becomes `gpt-4o`) so a model configured without its vendor
/// prefix still resolves. Ambiguous aliases — the same bare name published by
/// two providers — are dropped rather than guessed, and an alias overwrites a
/// bare record of the same name (Python's `dict.update` order).
pub fn parse_openrouter_models(payload: &Value) -> Result<OpenRouterModelTable> {
    let data = payload.get("data").ok_or_else(|| {
        openrouter_invalid_response_error("response payload must be an object with a data array")
    })?;
    let records = data
        .as_array()
        .ok_or_else(|| openrouter_invalid_response_error("response data must be an array"))?;

    let mut table = OpenRouterModelTable::default();
    let mut aliases: BTreeMap<String, usize> = BTreeMap::new();
    let mut ambiguous: BTreeSet<String> = BTreeSet::new();

    for record in records {
        let Some((model_id, context_length)) = parse_record(record) else {
            continue;
        };
        table.models += 1;
        table.windows.insert(model_id.clone(), context_length);
        let Some((_, alias)) = model_id.split_once('/') else {
            continue;
        };
        if alias.is_empty() {
            continue;
        }
        if aliases.contains_key(alias) {
            ambiguous.insert(alias.to_string());
        } else {
            aliases.insert(alias.to_string(), context_length);
        }
    }

    for (alias, context_length) in aliases {
        if !ambiguous.contains(&alias) {
            table.windows.insert(alias, context_length);
        }
    }
    Ok(table)
}

/// Extracts `(lowercased id, context_length)` from one catalogue record.
fn parse_record(record: &Value) -> Option<(String, usize)> {
    let model_id = record.get("id")?.as_str()?.trim();
    if model_id.is_empty() {
        return None;
    }
    let context_length = record.get("context_length")?;
    let tokens = match context_length {
        Value::Number(number) => number.as_u64()?,
        // Tolerate providers that quote the number.
        Value::String(text) => text.trim().parse::<u64>().ok()?,
        _ => return None,
    };
    if tokens == 0 {
        return None;
    }
    Some((model_id.to_lowercase(), usize::try_from(tokens).ok()?))
}

/// Normalizes the requested model ids into cache keys, dropping duplicates.
fn requested_keys(model_ids: &[impl AsRef<str>]) -> Vec<String> {
    let mut keys: Vec<String> = Vec::new();
    for model in model_ids {
        if let Some(key) = normalize_key(model.as_ref()) {
            if !keys.contains(&key) {
                keys.push(key);
            }
        }
    }
    keys
}

/// Maps a transport failure to the compact `OpenRouter` vocabulary.
fn map_transport_error(error: Error) -> Error {
    match error {
        Error::Timeout(message) => {
            Error::timeout(format!("{OPENROUTER_UNAVAILABLE_CODE}: {message}"))
        }
        other => openrouter_unavailable_error(other),
    }
}

impl ContextWindowPolicy {
    /// Fetches the `OpenRouter` catalogue and warms this policy's cache.
    ///
    /// Returns the number of requested model ids the cache covers afterwards
    /// (`0` when the feature is off, the request list is empty, or none of the
    /// requested ids appears in the catalogue). The whole catalogue is cached,
    /// not just the requested ids, so a later `resolve` for another model is
    /// already warm.
    ///
    /// An empty `model_ids` slice returns `Ok(0)` without a request.
    ///
    /// # Errors
    ///
    /// - [`Error::Transport`] (code [`OPENROUTER_UNAVAILABLE_CODE`]) for a
    ///   connection failure or a non-2xx status;
    /// - [`Error::Timeout`] (same code) when the deadline expires;
    /// - [`Error::Validation`] (code [`OPENROUTER_INVALID_RESPONSE_CODE`]) when
    ///   the payload is not a usable catalogue.
    pub async fn fetch(&self, model_ids: &[impl AsRef<str>]) -> Result<usize> {
        if !self.openrouter_enabled() {
            tracing::debug!("openrouter context-window fetch skipped: policy disabled");
            return Ok(0);
        }
        let requested = requested_keys(model_ids);
        if requested.is_empty() {
            return Ok(0);
        }
        let client = HttpClient::new(
            HttpClientConfig::default().with_timeout(self.openrouter_request_timeout),
        )
        .map_err(|error| openrouter_unavailable_error(format!("building HTTP client: {error}")))?;
        self.fetch_models(&client, &requested).await
    }

    /// Warms the cache with a caller-supplied client, skipping the request when
    /// every requested model is already fresh.
    ///
    /// This is the background-warm-up entry point: the current caller falls back
    /// to the built-in table and a later context picks up the fetched values.
    /// The return value matches [`ContextWindowPolicy::fetch`].
    ///
    /// # Errors
    ///
    /// Same as [`ContextWindowPolicy::fetch`].
    pub async fn prefetch(
        &self,
        model_ids: &[impl AsRef<str>],
        client: &HttpClient,
    ) -> Result<usize> {
        if !self.openrouter_enabled() {
            tracing::debug!("openrouter context-window prefetch skipped: policy disabled");
            return Ok(0);
        }
        let requested = requested_keys(model_ids);
        if requested.is_empty() {
            return Ok(0);
        }
        if requested.iter().all(|key| self.cache().contains(key)) {
            return Ok(0);
        }
        self.fetch_models(client, &requested).await
    }

    /// Performs the request and fills the cache; `requested` is already
    /// normalized.
    async fn fetch_models(&self, client: &HttpClient, requested: &[String]) -> Result<usize> {
        let endpoint = self.endpoint();
        let request = HttpRequest::get(endpoint).timeout(self.openrouter_request_timeout);
        let response = client.execute(request).await.map_err(map_transport_error)?;
        if !response.is_success() {
            return Err(openrouter_unavailable_error(format!(
                "HTTP {} from {endpoint}",
                response.status
            )));
        }
        let payload: Value = response.json().map_err(openrouter_invalid_response_error)?;
        let table = parse_openrouter_models(&payload)?;
        if table.windows.is_empty() {
            return Err(openrouter_invalid_response_error(
                "response does not contain valid model context windows",
            ));
        }
        let cache = self.cache();
        for (model, tokens) in &table.windows {
            cache.insert(model, *tokens);
        }
        let resolved = requested
            .iter()
            .filter(|model| cache.contains(model))
            .count();
        tracing::debug!(
            models = table.models,
            cached = table.windows.len(),
            resolved,
            "fetched openrouter context windows"
        );
        Ok(resolved)
    }
}
