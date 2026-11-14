//! Provider context-window resolution.
//!
//! Ports `ContextUtils.resolve_context_max` from
//! `context/context_utils.py`: the size of the window a provider will accept is
//! resolved through a fixed precedence ladder, and the winning rung is reported
//! so telemetry can tell an operator-supplied budget from a guess.
//!
//! The ladder is (highest priority first):
//!
//! 1. [`ContextWindowPolicy::context_window_tokens`] — a global, caller-imposed
//!    cap;
//! 2. [`ContextWindowPolicy::model_context_window_tokens_override`] — the
//!    window of the currently selected model;
//! 3. [`ContextWindowPolicy::model_context_window_tokens`] — a configured map,
//!    matched case-insensitively;
//! 4. [`MODEL_DEFAULT_CONTEXT_WINDOW_TOKENS`] — the hand-maintained built-in
//!    table;
//! 5. the `OpenRouter` cache ([`ContextWindowPolicy::cache`]) — warm values
//!    fetched by [`ContextWindowPolicy::fetch`]/[`ContextWindowPolicy::prefetch`];
//! 6. [`DEFAULT_CONTEXT_TOKENS`] (200 000).
//!
//! Resolution is synchronous and **offline by construction**: it only reads the
//! cache and never issues a request, so building a context can never wait on
//! the network. Warming the cache is an explicit, separate call.
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use openjiuwen_common::error::{Error, Result};

use crate::config::DEFAULT_CONTEXT_TOKENS;
use crate::policy::openrouter::{
    global_openrouter_window_cache, OpenRouterWindowCache, DEFAULT_OPENROUTER_REQUEST_TIMEOUT,
    OPENROUTER_MODELS_URL,
};

/// Built-in context windows (in tokens) for the model families the SDK knows.
///
/// This is Python's hand-maintained `MODEL_DEFAULT_CONTEXT_WINDOW_TOKENS` table,
/// ported verbatim: model ids keep the spelling the provider publishes, and
/// lookups are case-insensitive (see [`lookup_model_default_window`]).
///
/// Wrong values shift every compression threshold: the compression trigger is a
/// fraction of the resolved window, so an oversized entry silently disables
/// compression (and overflows the provider) while an undersized entry compresses
/// a conversation that still fit. Prefer the configured map or the global cap
/// when a deployment's real limit differs; treat this table as a fallback, not
/// as an authority.
///
/// Entries are grouped by family; each family comment states why its numbers are
/// what they are, because a single stale number there moves every threshold.
pub const MODEL_DEFAULT_CONTEXT_WINDOW_TOKENS: &[(&str, usize)] = &[
    // GLM (Zhipu): the 4.5/4.6/4.7 line settled on 200K-class windows, while
    // `glm-4` and the long-context variants diverge sharply. A wrong value here
    // shifts the compression trigger for every GLM deployment.
    ("glm-5.2", 1_048_576),
    ("glm-5.1", 204_800),
    ("glm-5", 204_800),
    ("glm-5-turbo", 202_752),
    ("glm-4.7", 204_800),
    ("glm-4.7-flash", 202_752),
    ("glm-4.7-flashx", 200_000),
    ("glm-4.6", 204_800),
    ("glm-4.5", 131_072),
    ("glm-4.5-air", 131_072),
    ("glm-4-long", 1_000_000),
    ("glm-4", 128_000),
    ("glm-4-9b-chat-1m", 1_048_576),
    // OpenAI GPT: `gpt-4.1` reports 1 047 576 (not 1M) and the GPT-5 line
    // reports 1 050 000, so rounding these numbers shifts the compression
    // trigger for every GPT deployment.
    ("gpt-5.6-luna", 1_050_000),
    ("gpt-5.6-luna-pro", 1_050_000),
    ("gpt-5.6-sol", 1_050_000),
    ("gpt-5.6-sol-pro", 1_050_000),
    ("gpt-5.6-terra", 1_050_000),
    ("gpt-5.6-terra-pro", 1_050_000),
    ("gpt-5.5", 1_050_000),
    ("gpt-5.5-pro", 1_050_000),
    ("gpt-5.4", 1_050_000),
    ("gpt-5.4-pro", 1_050_000),
    ("gpt-5.4-mini", 400_000),
    ("gpt-5.4-nano", 400_000),
    ("gpt-5.2", 400_000),
    ("gpt-5.1", 400_000),
    ("gpt-5", 400_000),
    ("gpt-5-pro", 400_000),
    ("gpt-5-mini", 400_000),
    ("gpt-5-nano", 400_000),
    ("gpt-4.1", 1_047_576),
    ("gpt-4.1-mini", 1_047_576),
    ("gpt-4.1-nano", 1_047_576),
    ("gpt-4o", 128_000),
    ("gpt-4o-mini", 128_000),
    ("gpt-4-turbo", 128_000),
    ("gpt-3.5-turbo", 16_385),
    // OpenAI o-series: a flat 200K, which is what every reasoning model in the
    // family enforces; a larger entry would let compression slip past the real
    // limit.
    ("o1", 200_000),
    ("o3", 200_000),
    ("o3-mini", 200_000),
    ("o3-pro", 200_000),
    ("o4-mini", 200_000),
    // DeepSeek: `deepseek-chat`/`deepseek-v3` are 128K-class, the later
    // revisions 160K-class and v4 1M — conflating them moves thresholds a long
    // way in either direction.
    ("deepseek-v4-pro", 1_048_576),
    ("deepseek-v4-flash", 1_048_576),
    ("deepseek-v3.2", 163_840),
    ("deepseek-v3.1-terminus", 163_840),
    ("deepseek-r1", 163_840),
    ("deepseek-v3", 128_000),
    ("deepseek-chat", 163_840),
    // Anthropic Claude: 200K for opus 4.5 / haiku, 1M for the newer sonnet and
    // opus revisions. Both spellings (`-4-5` and `4.5`) are listed because
    // providers and users write both.
    ("claude-opus-5", 1_000_000),
    ("claude-opus-4-8", 1_000_000),
    ("claude-opus-4.8", 1_000_000),
    ("claude-opus-4-7", 1_000_000),
    ("claude-opus-4.7", 1_000_000),
    ("claude-opus-4-6", 1_000_000),
    ("claude-opus-4.6", 1_000_000),
    ("claude-opus-4-5", 200_000),
    ("claude-opus-4.5", 200_000),
    ("claude-sonnet-5", 1_000_000),
    ("claude-sonnet-4-6", 1_000_000),
    ("claude-sonnet-4.6", 1_000_000),
    ("claude-sonnet-4-5", 1_000_000),
    ("claude-sonnet-4.5", 1_000_000),
    ("claude-haiku-4-5", 200_000),
    ("claude-haiku-4.5", 200_000),
    // Google Gemini: a flat ~1M window across the 2.5/3.x line; an undersized
    // entry would compress conversations the model still accepts.
    ("gemini-3.6-flash", 1_048_576),
    ("gemini-3.5-flash", 1_048_576),
    ("gemini-3.5-flash-lite", 1_048_576),
    ("gemini-3.1-pro-preview", 1_048_576),
    ("gemini-3.1-flash-lite", 1_048_576),
    ("gemini-3-pro-preview", 1_048_576),
    ("gemini-3-flash-preview", 1_048_576),
    ("gemini-2.5-pro", 1_048_576),
    ("gemini-2.5-flash", 1_048_576),
    ("gemini-2.5-flash-lite", 1_048_576),
    // Meta Llama: `llama-4-scout` genuinely accepts 10M, so it must not be
    // clamped to the 1M of its sibling; every compression threshold follows.
    ("llama-4-maverick", 1_048_576),
    ("llama-4-scout", 10_000_000),
    // Qwen: the 8K `qwen-turbo` sits next to 1M models in the same family, so a
    // single wrong entry changes the trigger by two orders of magnitude.
    ("qwen3.7-max", 1_000_000),
    ("qwen3.7-plus", 1_000_000),
    ("qwen3.7-flash", 1_000_000),
    ("qwen3.6-max-preview", 262_144),
    ("qwen3.6-plus", 1_000_000),
    ("qwen3.6-flash", 1_000_000),
    ("qwen3.5-plus", 1_000_000),
    ("qwen3.5-flash", 1_000_000),
    ("qwen3-max", 262_144),
    ("qwen3-max-thinking", 262_144),
    ("qwen3-coder-plus", 1_000_000),
    ("qwen3-coder-flash", 1_000_000),
    ("qwen3-coder-next", 262_144),
    ("qwen-max", 262_144),
    ("qwen-plus", 1_000_000),
    ("qwen-flash", 1_000_000),
    ("qwen-turbo", 8_192),
    ("qwen-long", 1_000_000),
    // Moonshot Kimi: K2 revisions moved from 128K to 256K and K3 to 1M, so a
    // stale entry compresses too early for the current models.
    ("kimi-k3", 1_048_576),
    ("kimi-k2.7-code", 262_144),
    ("kimi-k2.6", 262_144),
    ("kimi-k2.5", 262_144),
    ("kimi-k2-thinking", 262_144),
    ("kimi-k2-0905", 262_144),
    ("kimi-k2", 131_072),
    // MiniMax: M1 is 1M while M2/M2.x are 200K-class, so mixing them up shifts
    // the compression trigger fivefold.
    ("MiniMax-M3", 1_048_576),
    ("MiniMax-M2.7", 204_800),
    ("MiniMax-M2.7-highspeed", 204_800),
    ("MiniMax-M2.5", 204_800),
    ("MiniMax-M2.5-highspeed", 204_800),
    ("MiniMax-M2.1", 204_800),
    ("MiniMax-M2", 204_800),
    ("MiniMax-M1", 1_000_000),
    // xAI Grok: 2M for `grok-4.20`, 1M for 4.3, 500K for the plain aliases; a
    // wrong value lets a window overflow before compression ever engages.
    ("grok-4.20", 2_000_000),
    ("grok-4.20-multi-agent", 2_000_000),
    ("grok-4.5", 500_000),
    ("grok-4.3", 1_000_000),
    ("grok-4.3-latest", 1_000_000),
    ("grok-latest", 500_000),
];

/// Looks `model` up in [`MODEL_DEFAULT_CONTEXT_WINDOW_TOKENS`].
///
/// The lookup is exact first and then case-insensitive, matching Python's
/// `_get_positive_int_by_model_name`; surrounding whitespace is ignored and an
/// empty model id never matches. `None` means "no built-in entry", not "zero
/// tokens".
#[must_use]
pub fn lookup_model_default_window(model: &str) -> Option<usize> {
    let model = model.trim();
    if model.is_empty() {
        return None;
    }
    if let Some((_, tokens)) = MODEL_DEFAULT_CONTEXT_WINDOW_TOKENS
        .iter()
        .find(|(id, _)| *id == model)
    {
        return Some(*tokens);
    }
    MODEL_DEFAULT_CONTEXT_WINDOW_TOKENS
        .iter()
        .find(|(id, _)| id.eq_ignore_ascii_case(model))
        .map(|(_, tokens)| *tokens)
}

/// Which rung of the resolution ladder supplied [`ResolvedContextWindow::tokens`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContextWindowSource {
    /// The global `context_window_tokens` cap.
    Global,
    /// The selected model's `model_context_window_tokens_override`.
    Override,
    /// A hit in the configured `model_context_window_tokens` map.
    ConfiguredModel,
    /// A hit in the built-in [`MODEL_DEFAULT_CONTEXT_WINDOW_TOKENS`] table.
    #[serde(rename = "builtin")]
    BuiltIn,
    /// A hit in the `OpenRouter` cache.
    #[serde(rename = "openrouter_cache")]
    OpenRouterCache,
    /// The [`DEFAULT_CONTEXT_TOKENS`] fallback.
    Default,
}

impl ContextWindowSource {
    /// The stable `snake_case` name used in telemetry and JSON.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Global => "global",
            Self::Override => "override",
            Self::ConfiguredModel => "configured_model",
            Self::BuiltIn => "builtin",
            Self::OpenRouterCache => "openrouter_cache",
            Self::Default => "default",
        }
    }
}

/// A resolved context window plus the rung that produced it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ResolvedContextWindow {
    /// The usable window size, in tokens; always positive.
    pub tokens: usize,
    /// Where [`ResolvedContextWindow::tokens`] came from.
    pub source: ContextWindowSource,
}

impl Default for ResolvedContextWindow {
    fn default() -> Self {
        Self {
            tokens: DEFAULT_CONTEXT_TOKENS,
            source: ContextWindowSource::Default,
        }
    }
}

impl ResolvedContextWindow {
    /// Builds a resolution result.
    #[must_use]
    pub const fn new(tokens: usize, source: ContextWindowSource) -> Self {
        Self { tokens, source }
    }

    /// Whether the size is the [`DEFAULT_CONTEXT_TOKENS`] guess rather than a
    /// configured or discovered value.
    #[must_use]
    pub const fn is_default(&self) -> bool {
        matches!(self.source, ContextWindowSource::Default)
    }
}

/// Context-window resolution policy.
///
/// Every field is optional and serde-defaulted, so an empty policy resolves
/// every model to [`DEFAULT_CONTEXT_TOKENS`]. The struct is the Rust stand-in for
/// Python's context-window fields on `ContextEngineConfig`; it is kept here so
/// the engine can adopt it without changing [`crate::ContextConfig`].
///
/// `openrouter_request_timeout` accepts either a number of seconds
/// (`"openrouter_request_timeout": 2.5`) or the explicit
/// `{"secs": 2, "nanos": 500000000}` shape, so hand-written JSON and a
/// round-tripped value both load.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ContextWindowPolicy {
    /// Global cap; the highest-priority rung. `None` and `0` are ignored.
    pub context_window_tokens: Option<usize>,
    /// Window of the currently selected model; `None` and `0` are ignored.
    pub model_context_window_tokens_override: Option<usize>,
    /// Explicit per-model windows, matched case-insensitively. Entries with a
    /// zero value are ignored.
    pub model_context_window_tokens: BTreeMap<String, usize>,
    /// Whether [`ContextWindowPolicy::fetch`]/[`ContextWindowPolicy::prefetch`]
    /// may hit the network. Off by default, so the crate stays offline unless a
    /// deployment opts in.
    pub enable_openrouter_model_context_window_tokens: bool,
    /// Per-request deadline for the `OpenRouter` catalogue fetch; must be greater
    /// than zero (checked by [`ContextWindowPolicy::validate`]).
    #[serde(with = "duration_seconds")]
    pub openrouter_request_timeout: Duration,
    /// Catalogue URL override; `None` or empty means [`OPENROUTER_MODELS_URL`].
    pub openrouter_endpoint: Option<String>,
    /// Default model id used when [`ContextWindowPolicy::resolve`] is called
    /// without one.
    pub model_name: Option<String>,
    /// Default provider prefix used when [`ContextWindowPolicy::resolve`] is
    /// called without one; it only participates in `OpenRouter` cache lookups.
    pub model_provider: Option<String>,
    /// Cache handle used for `OpenRouter` lookups.
    ///
    /// `None` uses the process-wide cache
    /// ([`global_openrouter_window_cache`]); tests and multi-tenant hosts can
    /// inject their own through
    /// [`ContextWindowPolicy::with_openrouter_cache`]. The handle is not
    /// serialized.
    #[serde(skip)]
    pub openrouter_cache: Option<Arc<OpenRouterWindowCache>>,
}

impl Default for ContextWindowPolicy {
    fn default() -> Self {
        Self {
            context_window_tokens: None,
            model_context_window_tokens_override: None,
            model_context_window_tokens: BTreeMap::new(),
            enable_openrouter_model_context_window_tokens: false,
            openrouter_request_timeout: DEFAULT_OPENROUTER_REQUEST_TIMEOUT,
            openrouter_endpoint: None,
            model_name: None,
            model_provider: None,
            openrouter_cache: None,
        }
    }
}

impl ContextWindowPolicy {
    /// Builds the default policy: offline, no configured model, 200 000 tokens.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the global cap ([`ContextWindowPolicy::context_window_tokens`]).
    #[must_use]
    pub fn with_global_tokens(mut self, tokens: usize) -> Self {
        self.context_window_tokens = Some(tokens);
        self
    }

    /// Sets the selected-model override.
    #[must_use]
    pub fn with_model_override(mut self, tokens: usize) -> Self {
        self.model_context_window_tokens_override = Some(tokens);
        self
    }

    /// Sets the configured per-model window map.
    #[must_use]
    pub fn with_model_window(mut self, model: impl Into<String>, tokens: usize) -> Self {
        self.model_context_window_tokens
            .insert(model.into(), tokens);
        self
    }

    /// Enables or disables `OpenRouter` catalogue fetches.
    #[must_use]
    pub const fn with_openrouter_enabled(mut self, enabled: bool) -> Self {
        self.enable_openrouter_model_context_window_tokens = enabled;
        self
    }

    /// Sets the `OpenRouter` request deadline.
    #[must_use]
    pub const fn with_openrouter_timeout(mut self, timeout: Duration) -> Self {
        self.openrouter_request_timeout = timeout;
        self
    }

    /// Sets the `OpenRouter` catalogue URL.
    #[must_use]
    pub fn with_openrouter_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.openrouter_endpoint = Some(endpoint.into());
        self
    }

    /// Sets the default model id used when none is passed to
    /// [`ContextWindowPolicy::resolve`].
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model_name = Some(model.into());
        self
    }

    /// Sets the default provider prefix.
    #[must_use]
    pub fn with_provider(mut self, provider: impl Into<String>) -> Self {
        self.model_provider = Some(provider.into());
        self
    }

    /// Injects a cache handle instead of the process-wide one.
    #[must_use]
    pub fn with_openrouter_cache(mut self, cache: Arc<OpenRouterWindowCache>) -> Self {
        self.openrouter_cache = Some(cache);
        self
    }

    /// The cache handle this policy reads: the injected one, or the
    /// process-wide [`global_openrouter_window_cache`].
    #[must_use]
    pub fn cache(&self) -> &OpenRouterWindowCache {
        match &self.openrouter_cache {
            Some(cache) => cache.as_ref(),
            None => global_openrouter_window_cache(),
        }
    }

    /// The catalogue URL this policy would fetch from.
    #[must_use]
    pub fn endpoint(&self) -> &str {
        self.openrouter_endpoint
            .as_deref()
            .map(str::trim)
            .filter(|endpoint| !endpoint.is_empty())
            .unwrap_or(OPENROUTER_MODELS_URL)
    }

    /// Whether `OpenRouter` fetches are enabled.
    #[must_use]
    pub const fn openrouter_enabled(&self) -> bool {
        self.enable_openrouter_model_context_window_tokens
    }

    /// Validates the policy.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] when `openrouter_request_timeout` is zero: a
    /// zero deadline would turn every fetch into an immediate failure rather
    /// than a bounded wait. A zero `context_window_tokens` /
    /// `model_context_window_tokens_override` is *not* an error — it is ignored
    /// by [`ContextWindowPolicy::resolve`], matching Python.
    pub fn validate(&self) -> Result<()> {
        if self.openrouter_request_timeout.is_zero() {
            return Err(Error::config(
                "openrouter_request_timeout must be greater than zero".to_string(),
            ));
        }
        Ok(())
    }

    /// Resolves the context window for `model`/`provider`.
    ///
    /// `None` (or an empty/whitespace value) falls back to
    /// [`ContextWindowPolicy::model_name`] and
    /// [`ContextWindowPolicy::model_provider`]. The ladder and its sources are
    /// documented on this module; the resolution never touches the network.
    #[must_use]
    pub fn resolve(&self, model: Option<&str>, provider: Option<&str>) -> ResolvedContextWindow {
        if let Some(tokens) = positive(self.context_window_tokens) {
            return ResolvedContextWindow::new(tokens, ContextWindowSource::Global);
        }
        if let Some(tokens) = positive(self.model_context_window_tokens_override) {
            return ResolvedContextWindow::new(tokens, ContextWindowSource::Override);
        }
        if let Some(model) = self.effective_model(model) {
            if let Some(tokens) = lookup_configured_window(&self.model_context_window_tokens, model)
            {
                return ResolvedContextWindow::new(tokens, ContextWindowSource::ConfiguredModel);
            }
            if let Some(tokens) = lookup_model_default_window(model) {
                return ResolvedContextWindow::new(tokens, ContextWindowSource::BuiltIn);
            }
            if let Some(tokens) = self.lookup_openrouter(provider, model) {
                return ResolvedContextWindow::new(tokens, ContextWindowSource::OpenRouterCache);
            }
        }
        ResolvedContextWindow::new(DEFAULT_CONTEXT_TOKENS, ContextWindowSource::Default)
    }

    /// The model id `resolve` works with: the argument, else `model_name`.
    #[must_use]
    pub fn effective_model<'a>(&'a self, model: Option<&'a str>) -> Option<&'a str> {
        model
            .or(self.model_name.as_deref())
            .map(str::trim)
            .filter(|model| !model.is_empty())
    }

    /// The provider prefix `resolve` works with: the argument, else
    /// `model_provider`.
    #[must_use]
    pub fn effective_provider<'a>(&'a self, provider: Option<&'a str>) -> Option<&'a str> {
        provider
            .or(self.model_provider.as_deref())
            .map(str::trim)
            .filter(|provider| !provider.is_empty())
    }

    /// Cache lookup for one model: `provider/model` first when a provider is
    /// known and the model id carries no `/`, then the bare model id.
    fn lookup_openrouter(&self, provider: Option<&str>, model: &str) -> Option<usize> {
        let cache = self.cache();
        if let Some(provider) = self.effective_provider(provider) {
            if !model.contains('/') {
                if let Some(tokens) = cache.get(&format!("{provider}/{model}")) {
                    return Some(tokens);
                }
            }
        }
        cache.get(model)
    }
}

/// A positive `Some`; `None` and `0` mean "not configured".
const fn positive(value: Option<usize>) -> Option<usize> {
    match value {
        Some(tokens) if tokens > 0 => Some(tokens),
        _ => None,
    }
}

/// Python's `_get_positive_int_by_model_name`: exact key first, then a
/// case-insensitive scan; zero values never win.
fn lookup_configured_window(windows: &BTreeMap<String, usize>, model: &str) -> Option<usize> {
    if let Some(tokens) = windows.get(model).filter(|tokens| **tokens > 0) {
        return Some(*tokens);
    }
    windows
        .iter()
        .find(|(id, tokens)| **tokens > 0 && id.eq_ignore_ascii_case(model))
        .map(|(_, tokens)| *tokens)
}

/// Serde support for a [`Duration`] expressed in seconds.
///
/// Serialization always writes a number of seconds; deserialization accepts a
/// number (seconds, possibly fractional) or the explicit `{secs, nanos}` object,
/// which is what [`Duration`]'s own serde implementation produces.
mod duration_seconds {
    use std::time::Duration;

    use serde::de::{self, Deserializer};
    use serde::{Deserialize, Serializer};

    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Repr {
        Seconds(f64),
        Detailed {
            secs: u64,
            #[serde(default)]
            nanos: u32,
        },
    }

    pub(super) fn serialize<S: Serializer>(
        duration: &Duration,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        serializer.serialize_f64(duration.as_secs_f64())
    }

    pub(super) fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Duration, D::Error> {
        match Repr::deserialize(deserializer)? {
            Repr::Seconds(seconds) => Duration::try_from_secs_f64(seconds).map_err(|_| {
                de::Error::custom("duration must be a non-negative number of seconds")
            }),
            Repr::Detailed { secs, nanos } => Ok(Duration::from_secs(secs)
                .saturating_add(Duration::from_nanos(u64::from(nanos.min(999_999_999))))),
        }
    }
}
