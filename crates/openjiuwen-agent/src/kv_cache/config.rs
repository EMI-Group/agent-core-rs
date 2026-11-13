//! KV-cache affinity and runtime configuration.
//!
//! Ports `openjiuwen/core/kv_cache/kv_cache_config.py` and
//! `kv_cache_runtime_config.py`. The timeout constants are whole-action budgets
//! (provider call, retries, and retry backoff), not transport timeouts; they
//! mirror Python's float seconds and are converted with
//! [`Duration::from_secs_f64`] where the runtime needs a [`Duration`].
//!
//! Durations serialize as whole milliseconds through the local `serde` helpers
//! below, so configuration JSON stays human-readable without pulling in a new
//! dependency and sub-millisecond precision is intentionally dropped.
use crate::kv_cache::types::KVCacheBackend;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::time::Duration;

/// Actions a KV-cache management call is allowed to attempt.
///
/// Python has no retry loop; the constant is the exported single-attempt
/// contract for range and consumer code.
pub const KVC_MANAGEMENT_MAX_ATTEMPTS: u32 = 1;

/// Per-range (messages/tools) provider action budget, in seconds.
pub const KVC_RANGE_ACTION_TIMEOUT_SECONDS: f64 = 1.5;

/// Session-level prefetch/offload budget, in seconds.
pub const KVC_SESSION_OFFLOAD_PREFETCH_TIMEOUT_SECONDS: f64 = 2.0;

/// Session-level evict budget, in seconds.
pub const KVC_SESSION_EVICT_TIMEOUT_SECONDS: f64 = 3.0;

/// Bounded terminal cleanup budget, in seconds.
pub const KVC_TERMINAL_CLEANUP_TIMEOUT_SECONDS: f64 = 5.0;

/// Default capacity of an in-process prefix cache.
pub const DEFAULT_KV_CACHE_CAPACITY: usize = 256;

/// Converts an `f64` seconds constant into a [`Duration`].
fn seconds(value: f64) -> Duration {
    Duration::from_secs_f64(value)
}

/// Whole milliseconds of `value`, saturating on overflow.
fn millis(value: Duration) -> u64 {
    u64::try_from(value.as_millis()).unwrap_or(u64::MAX)
}

/// `serde` helper that maps an optional [`Duration`] to whole milliseconds.
mod option_duration_millis {
    use super::{millis, Deserializer, Duration, Serializer};
    use serde::Deserialize;

    /// Serializes `value` as whole milliseconds, or `null` when absent.
    // `serde(with = ...)` requires the `&Option<T>` receiver shape.
    #[allow(clippy::ref_option)]
    pub(super) fn serialize<S: Serializer>(
        value: &Option<Duration>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        match value {
            Some(value) => serializer.serialize_some(&millis(*value)),
            None => serializer.serialize_none(),
        }
    }

    /// Deserializes optional whole milliseconds into a [`Duration`].
    pub(super) fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Option<Duration>, D::Error> {
        Ok(Option::<u64>::deserialize(deserializer)?.map(Duration::from_millis))
    }
}

/// Affinity settings for the unified KV-cache hint protocol.
///
/// Affinity is opt-in: [`Self::default`] has [`Self::enabled`] `false`, and the
/// model-call hook then never touches the runtime. Every timeout stays `None`
/// until the host overrides it, so the module constants remain the defaults.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct KVCacheAffinityConfig {
    /// Whether prefix affinity is enabled for model calls.
    pub enabled: bool,
    /// Capacity the host should use when it builds an in-process runtime.
    pub max_entries: usize,
    /// Management attempts per action; Python's single-attempt contract.
    pub max_attempts: u32,
    /// Explicit whole-action override for prefetch/offload budgets.
    #[serde(
        with = "option_duration_millis",
        skip_serializing_if = "Option::is_none"
    )]
    pub action_timeout: Option<Duration>,
    /// Explicit session evict budget.
    #[serde(
        with = "option_duration_millis",
        skip_serializing_if = "Option::is_none"
    )]
    pub evict_timeout: Option<Duration>,
    /// Explicit session prefetch/offload budget.
    #[serde(
        with = "option_duration_millis",
        skip_serializing_if = "Option::is_none"
    )]
    pub offload_prefetch_timeout: Option<Duration>,
    /// Explicit bounded terminal-cleanup budget.
    #[serde(
        with = "option_duration_millis",
        skip_serializing_if = "Option::is_none"
    )]
    pub terminal_cleanup_timeout: Option<Duration>,
}

impl Default for KVCacheAffinityConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            max_entries: DEFAULT_KV_CACHE_CAPACITY,
            max_attempts: KVC_MANAGEMENT_MAX_ATTEMPTS,
            action_timeout: None,
            evict_timeout: None,
            offload_prefetch_timeout: None,
            terminal_cleanup_timeout: None,
        }
    }
}

impl KVCacheAffinityConfig {
    /// Creates a config with affinity enabled and every other default kept.
    #[must_use]
    pub fn enabled() -> Self {
        Self::default().with_enabled(true)
    }

    /// Whether prefix affinity is enabled.
    #[must_use]
    pub const fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Sets whether prefix affinity is enabled.
    #[must_use]
    pub const fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    /// Sets the recommended in-process capacity.
    #[must_use]
    pub const fn with_max_entries(mut self, max_entries: usize) -> Self {
        self.max_entries = max_entries;
        self
    }

    /// Sets the management attempt budget.
    #[must_use]
    pub const fn with_max_attempts(mut self, max_attempts: u32) -> Self {
        self.max_attempts = max_attempts;
        self
    }

    /// Sets the whole-action prefetch/offload override.
    #[must_use]
    pub const fn with_action_timeout(mut self, action_timeout: Duration) -> Self {
        self.action_timeout = Some(action_timeout);
        self
    }

    /// Sets the session evict budget.
    #[must_use]
    pub const fn with_evict_timeout(mut self, evict_timeout: Duration) -> Self {
        self.evict_timeout = Some(evict_timeout);
        self
    }

    /// Sets the session prefetch/offload budget.
    #[must_use]
    pub const fn with_offload_prefetch_timeout(mut self, timeout: Duration) -> Self {
        self.offload_prefetch_timeout = Some(timeout);
        self
    }

    /// Sets the bounded terminal-cleanup budget.
    #[must_use]
    pub const fn with_terminal_cleanup_timeout(mut self, timeout: Duration) -> Self {
        self.terminal_cleanup_timeout = Some(timeout);
        self
    }

    /// Effective prefetch/offload budget: explicit action override, then the
    /// specific prefetch override, then
    /// [`KVC_SESSION_OFFLOAD_PREFETCH_TIMEOUT_SECONDS`].
    #[must_use]
    pub fn effective_offload_prefetch_timeout(&self) -> Duration {
        self.action_timeout
            .or(self.offload_prefetch_timeout)
            .unwrap_or_else(|| seconds(KVC_SESSION_OFFLOAD_PREFETCH_TIMEOUT_SECONDS))
    }

    /// Effective session evict budget: explicit override, then
    /// [`KVC_SESSION_EVICT_TIMEOUT_SECONDS`].
    #[must_use]
    pub fn effective_evict_timeout(&self) -> Duration {
        self.evict_timeout
            .unwrap_or_else(|| seconds(KVC_SESSION_EVICT_TIMEOUT_SECONDS))
    }

    /// Effective terminal-cleanup budget: explicit override, then
    /// [`KVC_TERMINAL_CLEANUP_TIMEOUT_SECONDS`].
    #[must_use]
    pub fn effective_terminal_cleanup_timeout(&self) -> Duration {
        self.terminal_cleanup_timeout
            .unwrap_or_else(|| seconds(KVC_TERMINAL_CLEANUP_TIMEOUT_SECONDS))
    }
}

/// Resolves one whole-action KV-cache timeout.
///
/// Returns `None` when affinity is disabled — a disabled config must not spend
/// a budget on cache management at all. Otherwise the explicit
/// [`KVCacheAffinityConfig::action_timeout`] wins, then the specific
/// prefetch/offload timeout, then
/// [`KVC_SESSION_OFFLOAD_PREFETCH_TIMEOUT_SECONDS`].
#[must_use]
pub fn resolve_kvc_action_timeout(config: &KVCacheAffinityConfig) -> Option<Duration> {
    config
        .is_enabled()
        .then(|| config.effective_offload_prefetch_timeout())
}

/// Process-local runtime configuration.
///
/// [`Self::backend`] selects the runtime implementation (see
/// [`KVCacheBackend::runtime`]), `capacity` bounds the retained prefix entries
/// (`0` disables retention), and `ttl` expires idle entries.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct KVCacheRuntimeConfig {
    /// Backend that serves the runtime.
    pub backend: KVCacheBackend,
    /// Maximum number of retained prefix entries.
    pub capacity: usize,
    /// Optional idle time after which an entry is dropped.
    #[serde(
        with = "option_duration_millis",
        skip_serializing_if = "Option::is_none"
    )]
    pub ttl: Option<Duration>,
    /// Affinity settings carried along with the runtime.
    pub affinity: KVCacheAffinityConfig,
}

impl Default for KVCacheRuntimeConfig {
    fn default() -> Self {
        Self {
            backend: KVCacheBackend::InProcess,
            capacity: DEFAULT_KV_CACHE_CAPACITY,
            ttl: None,
            affinity: KVCacheAffinityConfig::default(),
        }
    }
}

impl KVCacheRuntimeConfig {
    /// Sets the backend.
    #[must_use]
    pub const fn with_backend(mut self, backend: KVCacheBackend) -> Self {
        self.backend = backend;
        self
    }

    /// Sets the retained-entry capacity (`0` disables retention).
    #[must_use]
    pub const fn with_capacity(mut self, capacity: usize) -> Self {
        self.capacity = capacity;
        self
    }

    /// Sets the idle time after which an entry expires.
    #[must_use]
    pub const fn with_ttl(mut self, ttl: Duration) -> Self {
        self.ttl = Some(ttl);
        self
    }

    /// Replaces the affinity settings.
    #[must_use]
    pub fn with_affinity(mut self, affinity: KVCacheAffinityConfig) -> Self {
        self.affinity = affinity;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn affinity_defaults_are_disabled_and_constant_backed() {
        let config = KVCacheAffinityConfig::default();
        assert!(!config.enabled);
        assert!(!config.is_enabled());
        assert_eq!(config.max_entries, DEFAULT_KV_CACHE_CAPACITY);
        assert_eq!(config.max_attempts, KVC_MANAGEMENT_MAX_ATTEMPTS);
        assert!(config.action_timeout.is_none());
        assert_eq!(
            config.effective_offload_prefetch_timeout(),
            Duration::from_secs_f64(2.0)
        );
        assert_eq!(
            config.effective_evict_timeout(),
            Duration::from_secs_f64(3.0)
        );
        assert_eq!(
            config.effective_terminal_cleanup_timeout(),
            Duration::from_secs_f64(5.0)
        );
    }

    #[test]
    fn builders_chain_and_override() {
        let config = KVCacheAffinityConfig::default()
            .with_enabled(true)
            .with_max_entries(16)
            .with_max_attempts(3)
            .with_action_timeout(Duration::from_millis(750))
            .with_evict_timeout(Duration::from_millis(900))
            .with_offload_prefetch_timeout(Duration::from_millis(400))
            .with_terminal_cleanup_timeout(Duration::from_millis(1200));

        assert!(config.is_enabled());
        assert_eq!(config.max_entries, 16);
        assert_eq!(config.max_attempts, 3);
        assert_eq!(config.action_timeout, Some(Duration::from_millis(750)));
        assert_eq!(config.effective_evict_timeout(), Duration::from_millis(900));
        assert_eq!(
            config.effective_terminal_cleanup_timeout(),
            Duration::from_millis(1200)
        );
        // The explicit whole-action override wins over the specific budget.
        assert_eq!(
            config.effective_offload_prefetch_timeout(),
            Duration::from_millis(750)
        );
    }

    #[test]
    fn resolve_action_timeout_is_none_when_disabled() {
        assert!(resolve_kvc_action_timeout(&KVCacheAffinityConfig::default()).is_none());
        assert_eq!(
            resolve_kvc_action_timeout(&KVCacheAffinityConfig::enabled()),
            Some(Duration::from_secs_f64(
                KVC_SESSION_OFFLOAD_PREFETCH_TIMEOUT_SECONDS
            ))
        );
        assert_eq!(
            resolve_kvc_action_timeout(
                &KVCacheAffinityConfig::enabled().with_action_timeout(Duration::from_millis(10))
            ),
            Some(Duration::from_millis(10))
        );
    }

    #[test]
    fn durations_serialize_as_whole_milliseconds() {
        let config = KVCacheAffinityConfig::enabled()
            .with_action_timeout(Duration::from_millis(1500))
            .with_evict_timeout(Duration::from_millis(2));
        let value = serde_json::to_value(&config).unwrap();
        assert_eq!(value["action_timeout"], json!(1500));
        assert_eq!(value["evict_timeout"], json!(2));
        assert!(value.get("terminal_cleanup_timeout").is_none());

        let parsed: KVCacheAffinityConfig = serde_json::from_value(value).unwrap();
        assert_eq!(parsed, config);
    }

    #[test]
    fn runtime_config_round_trips_and_tolerates_unknown_fields() {
        let config = KVCacheRuntimeConfig::default()
            .with_capacity(8)
            .with_ttl(Duration::from_millis(250));
        let value = serde_json::to_value(&config).unwrap();
        assert_eq!(
            value,
            json!({
                "backend": "in_process",
                "capacity": 8,
                "ttl": 250,
                "affinity": serde_json::to_value(KVCacheAffinityConfig::default()).unwrap(),
            })
        );

        let parsed: KVCacheRuntimeConfig = serde_json::from_value(json!({
            "backend": "redis",
            "future_field": true,
        }))
        .unwrap();
        assert_eq!(parsed.backend, KVCacheBackend::Redis);
        assert_eq!(parsed.capacity, DEFAULT_KV_CACHE_CAPACITY);
        assert!(parsed.ttl.is_none());
        assert_eq!(parsed.affinity, KVCacheAffinityConfig::default());
    }
}
