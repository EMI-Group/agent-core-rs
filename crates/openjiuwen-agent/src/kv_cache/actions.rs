//! Provider-facing KV-cache actions: prefetch, offload, and evict.
//!
//! Ports the provider half of `openjiuwen/core/kv_cache/kv_cache_runtime.py`: the
//! three session-level actions a host may ask a model provider to perform on a
//! cached prefix, the request those actions carry, the async
//! [`KvCacheActionProvider`] seam a provider integration implements, and the
//! counters the runtime keeps.
//!
//! The Rust rewrite has no model-client handles (Python dispatches on
//! `model.prefetch_kvc`/`offload_kvc`/`evict_kvc`); a provider is an explicit
//! trait object, and the workflow engine that schedules the actions lives in
//! [`crate::kv_cache::action_runtime`].
use async_trait::async_trait;
use openjiuwen_common::error::Result;
use serde::{Deserialize, Serialize};
use std::fmt;

/// The `target` a session-level action addresses.
pub const KV_CACHE_ACTION_TARGET_SESSION: &str = "session";

/// One provider-side cache action.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KvCacheAction {
    /// Load a cache scope back into the provider so the next request reuses it.
    Prefetch,
    /// Move a cache scope out of the provider's hot storage once it is idle.
    Offload,
    /// Drop a cache scope at the end of its session.
    Evict,
}

impl KvCacheAction {
    /// The action name used on the wire, in logs, and in error messages.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Prefetch => "prefetch",
            Self::Offload => "offload",
            Self::Evict => "evict",
        }
    }
}

impl fmt::Display for KvCacheAction {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// One action request handed to a [`KvCacheActionProvider`].
///
/// The shapes mirror Python's keyword arguments: the cache scope (`cache_id`),
/// the lineage it branches from (`parent_cache_id`), the scope kind
/// (`target`, always [`KV_CACHE_ACTION_TARGET_SESSION`]), and the model the
/// scope belongs to when one is known.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct KvCacheActionRequest {
    /// Cache scope the action applies to.
    pub cache_id: String,
    /// Lineage the scope branches from; equals `cache_id` for a root scope.
    pub parent_cache_id: String,
    /// Scope kind addressed by the action; [`KV_CACHE_ACTION_TARGET_SESSION`].
    pub target: String,
    /// Model the scope was built for, when one is known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

impl Default for KvCacheActionRequest {
    fn default() -> Self {
        Self {
            cache_id: String::new(),
            parent_cache_id: String::new(),
            target: KV_CACHE_ACTION_TARGET_SESSION.to_owned(),
            model: None,
        }
    }
}

impl KvCacheActionRequest {
    /// Creates a session-scoped request for `cache_id` branching from
    /// `parent_cache_id`.
    #[must_use]
    pub fn new(cache_id: impl Into<String>, parent_cache_id: impl Into<String>) -> Self {
        Self {
            cache_id: cache_id.into(),
            parent_cache_id: parent_cache_id.into(),
            ..Self::default()
        }
    }

    /// Replaces the addressed scope kind.
    #[must_use]
    pub fn with_target(mut self, target: impl Into<String>) -> Self {
        self.target = target.into();
        self
    }

    /// Sets the model the scope belongs to.
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }
}

/// Provider integration that performs [`KvCacheAction`]s.
///
/// Every method is best-effort: the runtime treats an error as a failed action
/// and fails open (see [`crate::kv_cache::action_runtime::KvCacheActionRuntime`]),
/// so a provider may report `Ok(false)` for "nothing to do" and an `Err` for a
/// transport or provider failure.
#[async_trait]
pub trait KvCacheActionProvider: Send + Sync {
    /// Loads `request`'s scope back into the provider.
    ///
    /// Returns whether the provider reported the scope as available.
    ///
    /// # Errors
    ///
    /// Returns the provider/transport error.
    async fn prefetch(&self, request: &KvCacheActionRequest) -> Result<bool>;

    /// Moves `request`'s scope out of the provider's hot storage.
    ///
    /// # Errors
    ///
    /// Returns the provider/transport error.
    async fn offload(&self, request: &KvCacheActionRequest) -> Result<bool>;

    /// Drops `request`'s scope.
    ///
    /// # Errors
    ///
    /// Returns the provider/transport error.
    async fn evict(&self, request: &KvCacheActionRequest) -> Result<bool>;

    /// Whether this provider performs any action at all.
    ///
    /// A provider that returns `false` behaves like Python's model without a
    /// `*_kvc` method: actions report `Ok(false)` instead of being attempted.
    fn supports(&self) -> bool {
        true
    }

    /// Dispatches `action` to the matching method.
    ///
    /// # Errors
    ///
    /// Returns the provider/transport error of the dispatched call.
    async fn perform(&self, action: KvCacheAction, request: &KvCacheActionRequest) -> Result<bool> {
        match action {
            KvCacheAction::Prefetch => self.prefetch(request).await,
            KvCacheAction::Offload => self.offload(request).await,
            KvCacheAction::Evict => self.evict(request).await,
        }
    }
}

/// One action a provider received, in call order.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct KvCacheActionRecord {
    /// Action that was requested.
    pub action: KvCacheAction,
    /// Request the action carried.
    pub request: KvCacheActionRequest,
}

impl Default for KvCacheActionRecord {
    fn default() -> Self {
        Self {
            action: KvCacheAction::Prefetch,
            request: KvCacheActionRequest::default(),
        }
    }
}

impl KvCacheActionRecord {
    /// Creates a record of `action` applied to `request`.
    #[must_use]
    pub fn new(action: KvCacheAction, request: KvCacheActionRequest) -> Self {
        Self { action, request }
    }

    /// The cache scope the recorded action addressed.
    #[must_use]
    pub fn cache_id(&self) -> &str {
        self.request.cache_id.as_str()
    }
}

/// Counters an action runtime keeps for the provider actions it ran.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct KvCacheActionStats {
    /// Prefetches the provider reported as successful.
    pub prefetch_succeeded: u64,
    /// Offloads the provider reported as successful.
    pub offload_succeeded: u64,
    /// Evictions the provider reported as successful.
    pub evict_succeeded: u64,
    /// Actions that failed, errored, or timed out.
    pub failures: u64,
    /// Actions the runtime skipped because the state already allowed inference
    /// or the scope had failed open.
    pub skipped: u64,
}

impl KvCacheActionStats {
    /// Successful actions of `action`'s kind.
    #[must_use]
    pub const fn succeeded(&self, action: KvCacheAction) -> u64 {
        match action {
            KvCacheAction::Prefetch => self.prefetch_succeeded,
            KvCacheAction::Offload => self.offload_succeeded,
            KvCacheAction::Evict => self.evict_succeeded,
        }
    }

    /// Bumps the success counter for `action`.
    pub(crate) fn record_success(&mut self, action: KvCacheAction) {
        match action {
            KvCacheAction::Prefetch => {
                self.prefetch_succeeded = self.prefetch_succeeded.saturating_add(1);
            }
            KvCacheAction::Offload => {
                self.offload_succeeded = self.offload_succeeded.saturating_add(1);
            }
            KvCacheAction::Evict => {
                self.evict_succeeded = self.evict_succeeded.saturating_add(1);
            }
        }
    }

    /// Bumps the failure counter.
    pub(crate) fn record_failure(&mut self) {
        self.failures = self.failures.saturating_add(1);
    }

    /// Bumps the skipped counter.
    pub(crate) fn record_skipped(&mut self) {
        self.skipped = self.skipped.saturating_add(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn action_wire_names_round_trip() {
        for (action, name) in [
            (KvCacheAction::Prefetch, "prefetch"),
            (KvCacheAction::Offload, "offload"),
            (KvCacheAction::Evict, "evict"),
        ] {
            assert_eq!(action.as_str(), name);
            assert_eq!(action.to_string(), name);
            assert_eq!(serde_json::to_value(action).unwrap(), json!(name));
            let parsed: KvCacheAction = serde_json::from_value(json!(name)).unwrap();
            assert_eq!(parsed, action);
        }
    }

    #[test]
    fn request_defaults_to_a_session_scope() {
        let request = KvCacheActionRequest::new("cache-a", "root-a");
        assert_eq!(request.cache_id, "cache-a");
        assert_eq!(request.parent_cache_id, "root-a");
        assert_eq!(request.target, KV_CACHE_ACTION_TARGET_SESSION);
        assert!(request.model.is_none());
        assert_eq!(
            serde_json::to_value(&request).unwrap(),
            json!({"cache_id": "cache-a", "parent_cache_id": "root-a", "target": "session"})
        );

        let with_model = request.clone().with_model("gpt-test");
        assert_eq!(with_model.model.as_deref(), Some("gpt-test"));
        let round_trip: KvCacheActionRequest =
            serde_json::from_value(serde_json::to_value(&with_model).unwrap()).unwrap();
        assert_eq!(round_trip, with_model);
        assert_eq!(round_trip.clone().with_target("team").target, "team");
    }

    #[test]
    fn request_tolerates_unknown_and_missing_fields() {
        let parsed: KvCacheActionRequest =
            serde_json::from_value(json!({"cache_id": "c", "future": true})).unwrap();
        assert_eq!(parsed.cache_id, "c");
        assert_eq!(parsed.target, KV_CACHE_ACTION_TARGET_SESSION);
        assert_eq!(parsed, KvCacheActionRequest::new("c", ""));
    }

    #[test]
    fn stats_count_per_action_and_serialize() {
        let mut stats = KvCacheActionStats::default();
        stats.record_success(KvCacheAction::Prefetch);
        stats.record_success(KvCacheAction::Prefetch);
        stats.record_success(KvCacheAction::Offload);
        stats.record_failure();
        stats.record_skipped();

        assert_eq!(stats.succeeded(KvCacheAction::Prefetch), 2);
        assert_eq!(stats.succeeded(KvCacheAction::Offload), 1);
        assert_eq!(stats.succeeded(KvCacheAction::Evict), 0);
        assert_eq!(
            serde_json::to_value(stats).unwrap(),
            json!({
                "prefetch_succeeded": 2,
                "offload_succeeded": 1,
                "evict_succeeded": 0,
                "failures": 1,
                "skipped": 1,
            })
        );
        let parsed: KvCacheActionStats = serde_json::from_value(json!({"failures": 3})).unwrap();
        assert_eq!(parsed.failures, 3);
        assert_eq!(parsed.prefetch_succeeded, 0);
    }

    #[test]
    fn record_keeps_the_action_and_request() {
        let record = KvCacheActionRecord::new(
            KvCacheAction::Offload,
            KvCacheActionRequest::new("cache-a", "root-a"),
        );
        assert_eq!(record.action, KvCacheAction::Offload);
        assert_eq!(record.cache_id(), "cache-a");
        let parsed: KvCacheActionRecord =
            serde_json::from_value(serde_json::to_value(&record).unwrap()).unwrap();
        assert_eq!(parsed, record);
        assert_eq!(
            KvCacheActionRecord::default().action,
            KvCacheAction::Prefetch
        );
    }
}
