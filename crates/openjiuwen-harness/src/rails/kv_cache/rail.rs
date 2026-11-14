//! The KV-cache subagent lifecycle and its harness rail.
//!
//! Ports the provider-facing half of
//! `openjiuwen/harness/kv_cache/kv_cache_subagent_lifecycle.py`: instead of
//! calling `Session.prepare_kvc/suspend_kvc/release_kvc`, the Rust lifecycle
//! keeps one [`KvCacheSubagentSession`] per delegated subagent and drives the
//! [`KvCacheProviderRegistry`] seam with the policy helpers of
//! [`super::policy`].
//!
//! Lifecycle mapping (spec `S_16`): start + sticky → `Prepare`; success + sticky →
//! `Suspend` (resumable); success + one-shot → `Release`; failure/cancel/any
//! type → `Release`; explicit eviction → `Release`.
//!
//! [`KvCacheLifecycleRail`] observes the subagent tools of `crate::subagents`
//! through [`HarnessRail::after_tool_call`], because the runtime itself lives
//! outside this module. Affinity is an optimisation: every error is logged and
//! swallowed, an unknown id is a warning plus `Ok(None)`, and an OFF deployment
//! makes no provider call and stores no state.
use super::policy::{
    affinity_enabled, is_sticky_subagent_type, resolve_sub_session_id,
    resolve_subagent_parent_cache_id, KvCacheAffinityConfig, KvCacheLineage, RuntimeSessionLineage,
};
use super::provider::KvCacheProviderRegistry;
use crate::rails::{rail_priority, HarnessRail, HarnessRailContext};
use async_trait::async_trait;
use openjiuwen_common::error::Result;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::sync::Arc;
/// The provider operation the lifecycle decided to run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KvCacheAction {
    /// Prefetch the cache identity of a freshly started subagent.
    Prepare,
    /// Offload the identity of a resumable (sticky) subagent.
    Suspend,
    /// Evict the identity; the subagent will not resume from it.
    Release,
}
/// The cache lifecycle state of one subagent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KvCacheSubagentState {
    /// Started; a sticky type has been prepared.
    Active,
    /// Finished successfully and suspended, so a later turn may resume it.
    Suspended,
    /// Finished or evicted; the remote identity was released.
    Released,
}
/// The identity of one delegated subagent, as handed to the lifecycle.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvCacheSubagent {
    /// Runtime id of the subagent.
    pub subagent_id: String,
    /// Registered subagent type, which decides stickiness.
    pub subagent_type: String,
    /// Child session id carrying the cache identity.
    pub sub_session_id: String,
    /// Provider-facing identity of the parent cache.
    pub parent_cache_id: String,
}
/// The tracked cache lifecycle of one subagent, including its action history.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvCacheSubagentSession {
    /// Runtime id of the subagent.
    pub subagent_id: String,
    /// Registered subagent type.
    pub subagent_type: String,
    /// Child session id carrying the cache identity.
    pub sub_session_id: String,
    /// Provider-facing identity of the parent cache.
    pub parent_cache_id: String,
    /// Current lifecycle state.
    pub state: KvCacheSubagentState,
    /// Ordered provider operations already performed for this subagent.
    pub actions: Vec<KvCacheAction>,
}
/// Tracks subagent cache identities and drives the provider seam.
///
/// State is per instance — there are no module-level singletons — and every
/// buffer is a `BTreeMap` behind a `parking_lot::Mutex` held only for short
/// read/record/insert sections, never across `.await`.
pub struct KvCacheSubagentLifecycle {
    /// Provider seam every action is forwarded to.
    registry: Arc<dyn KvCacheProviderRegistry>,
    /// Affinity switch; OFF means no state and no provider calls.
    config: KvCacheAffinityConfig,
    /// Parent identity resolver; defaults to [`RuntimeSessionLineage`].
    lineage: Option<Arc<dyn KvCacheLineage>>,
    /// Tracked sessions by subagent id.
    sessions: Mutex<BTreeMap<String, KvCacheSubagentSession>>,
}
impl KvCacheSubagentLifecycle {
    /// Builds a lifecycle over `registry`, with affinity disabled.
    ///
    /// The OFF default mirrors Python's absent-config path: a deployment that
    /// never enables affinity also never creates a child cache identity.
    #[must_use]
    pub fn new(registry: Arc<dyn KvCacheProviderRegistry>) -> Self {
        Self {
            registry,
            config: KvCacheAffinityConfig::default(),
            lineage: Some(Arc::new(RuntimeSessionLineage)),
            sessions: Mutex::new(BTreeMap::new()),
        }
    }
    /// Returns this lifecycle with `config` as its affinity switch.
    #[must_use]
    pub fn with_config(mut self, config: KvCacheAffinityConfig) -> Self {
        self.config = config;
        self
    }
    /// Returns this lifecycle with `lineage` as its parent identity resolver.
    #[must_use]
    pub fn with_lineage(mut self, lineage: Arc<dyn KvCacheLineage>) -> Self {
        self.lineage = Some(lineage);
        self
    }
    /// Whether affinity is enabled.
    #[must_use]
    pub fn affinity_enabled(&self) -> bool {
        affinity_enabled(Some(&self.config))
    }
    /// The provider seam.
    #[must_use]
    pub fn registry(&self) -> &Arc<dyn KvCacheProviderRegistry> {
        &self.registry
    }
    /// A snapshot of every tracked session, ordered by subagent id.
    #[must_use]
    pub fn sessions(&self) -> Vec<KvCacheSubagentSession> {
        self.sessions.lock().values().cloned().collect()
    }
    /// The tracked session of `subagent_id`, when one exists.
    #[must_use]
    pub fn get(&self, subagent_id: &str) -> Option<KvCacheSubagentSession> {
        self.sessions.lock().get(subagent_id).cloned()
    }
    /// Starts tracking a delegation and prepares it when its type is sticky.
    ///
    /// With affinity off this makes no provider call and stores nothing, so an
    /// OFF deployment never even mints a child cache identity. A sticky type is
    /// prepared through the provider seam before it is recorded as `Active`; a
    /// provider failure propagates and leaves no state behind.
    ///
    /// # Errors
    ///
    /// A provider failure from [`KvCacheProviderRegistry::prepare`].
    pub async fn prepare_subagent(
        &self,
        subagent: KvCacheSubagent,
    ) -> Result<Option<KvCacheSubagentSession>> {
        if !self.affinity_enabled() {
            return Ok(None);
        }
        let mut actions = Vec::new();
        if is_sticky_subagent_type(&subagent.subagent_type) {
            self.registry.prepare(&subagent.parent_cache_id).await?;
            actions.push(KvCacheAction::Prepare);
        }
        let session = KvCacheSubagentSession {
            subagent_id: subagent.subagent_id,
            subagent_type: subagent.subagent_type,
            sub_session_id: subagent.sub_session_id,
            parent_cache_id: subagent.parent_cache_id,
            state: KvCacheSubagentState::Active,
            actions,
        };
        self.sessions
            .lock()
            .insert(session.subagent_id.clone(), session.clone());
        Ok(Some(session))
    }
    /// Finishes a delegation: suspend sticky successes, release everything else.
    ///
    /// A sticky type that succeeded is offloaded and marked `Suspended` so a
    /// later turn can resume it; any other outcome releases the identity. An
    /// unknown id is logged and ignored, because cache bookkeeping must never
    /// fail the delegation path.
    ///
    /// # Errors
    ///
    /// A provider failure from [`KvCacheProviderRegistry::suspend`] or
    /// [`KvCacheProviderRegistry::release`].
    pub async fn finish_subagent(
        &self,
        subagent_id: &str,
        succeeded: bool,
    ) -> Result<Option<KvCacheSubagentSession>> {
        if !self.affinity_enabled() {
            return Ok(None);
        }
        let Some(current) = self.get(subagent_id) else {
            tracing::warn!(
                subagent_id,
                "harness kv_cache: finish for an unknown subagent; ignoring"
            );
            return Ok(None);
        };
        let suspend = succeeded && is_sticky_subagent_type(&current.subagent_type);
        let action = if suspend {
            self.registry.suspend(&current.parent_cache_id).await?;
            KvCacheAction::Suspend
        } else {
            self.registry.release(&current.parent_cache_id).await?;
            KvCacheAction::Release
        };
        let state = if suspend {
            KvCacheSubagentState::Suspended
        } else {
            KvCacheSubagentState::Released
        };
        Ok(self.record(subagent_id, state, action))
    }
    /// Evicts a delegation unconditionally, releasing its cache identity.
    ///
    /// Python's `evict_subagent` releases without inspecting the current state,
    /// and the provider seam is idempotent, so an already released id is
    /// released again. An unknown id is logged and ignored.
    ///
    /// # Errors
    ///
    /// A provider failure from [`KvCacheProviderRegistry::release`].
    pub async fn evict_subagent(
        &self,
        subagent_id: &str,
    ) -> Result<Option<KvCacheSubagentSession>> {
        if !self.affinity_enabled() {
            return Ok(None);
        }
        let Some(current) = self.get(subagent_id) else {
            tracing::warn!(
                subagent_id,
                "harness kv_cache: evict for an unknown subagent; ignoring"
            );
            return Ok(None);
        };
        self.registry.release(&current.parent_cache_id).await?;
        Ok(self.record(
            subagent_id,
            KvCacheSubagentState::Released,
            KvCacheAction::Release,
        ))
    }
    /// Releases every session that is not already [`KvCacheSubagentState::Released`].
    ///
    /// Returns a snapshot of all tracked sessions after the sweep. Sessions are
    /// processed one at a time, so a provider failure leaves the earlier ones
    /// marked and propagates the error.
    ///
    /// # Errors
    ///
    /// A provider failure from [`KvCacheProviderRegistry::release`].
    pub async fn release_all(&self) -> Result<Vec<KvCacheSubagentSession>> {
        if !self.affinity_enabled() {
            return Ok(Vec::new());
        }
        let pending: Vec<KvCacheSubagentSession> = self
            .sessions
            .lock()
            .values()
            .filter(|session| session.state != KvCacheSubagentState::Released)
            .cloned()
            .collect();
        for session in pending {
            self.registry.release(&session.parent_cache_id).await?;
            self.record(
                &session.subagent_id,
                KvCacheSubagentState::Released,
                KvCacheAction::Release,
            );
        }
        Ok(self.sessions())
    }
    /// Records `action`/`state` for `subagent_id` and returns the updated session.
    fn record(
        &self,
        subagent_id: &str,
        state: KvCacheSubagentState,
        action: KvCacheAction,
    ) -> Option<KvCacheSubagentSession> {
        let mut sessions = self.sessions.lock();
        let session = sessions.get_mut(subagent_id)?;
        session.state = state;
        session.actions.push(action);
        Some(session.clone())
    }
}
/// Drives the subagent KV-cache lifecycle from the harness tool hooks.
///
/// The rail reads `after_tool_call` payloads of the delegation tools and calls
/// the lifecycle:
///
/// | Tool | Effect |
/// | --- | --- |
/// | `subagent_spawn`, `task_tool`, `task` | [`KvCacheSubagentLifecycle::prepare_subagent`] |
/// | `subagent_wait` | [`KvCacheSubagentLifecycle::finish_subagent`] per terminal result |
/// | `subagent_close` | [`KvCacheSubagentLifecycle::evict_subagent`] |
///
/// A missing payload, a `success: false` envelope, malformed arguments or an
/// unknown tool are skipped with a debug log. Every lifecycle error is logged
/// and swallowed: affinity must never abort a run. With affinity off the rail is
/// completely inert.
pub struct KvCacheLifecycleRail {
    /// The lifecycle this rail drives.
    lifecycle: Arc<KvCacheSubagentLifecycle>,
}
impl KvCacheLifecycleRail {
    /// Builds the rail over `lifecycle`.
    #[must_use]
    pub fn new(lifecycle: Arc<KvCacheSubagentLifecycle>) -> Self {
        Self { lifecycle }
    }
    /// The lifecycle this rail drives.
    #[must_use]
    pub fn lifecycle(&self) -> &Arc<KvCacheSubagentLifecycle> {
        &self.lifecycle
    }
    /// The subagent type named by the payload, falling back to the tool arguments.
    fn subagent_type(data: &Value, arguments: &Value) -> Option<String> {
        non_empty(data.get("subagent_type"))
            .or_else(|| non_empty(arguments.get("subagent_type")))
            .map(str::to_owned)
    }
    /// Handles a successful spawn-like payload.
    async fn on_spawn(&self, ctx: &HarnessRailContext, data: &Value, arguments: &Value) {
        let subagent_id =
            non_empty(data.get("subagent_id")).or_else(|| non_empty(data.get("agent_id")));
        let subagent_type = Self::subagent_type(data, arguments);
        let (Some(subagent_id), Some(subagent_type)) = (subagent_id, subagent_type) else {
            tracing::debug!(
                "harness kv_cache: spawn payload without a subagent id or type; skipping"
            );
            return;
        };
        let parent_session_id = ctx
            .base
            .session
            .as_ref()
            .map_or_else(String::new, |session| session.id().to_owned());
        let parent_cache_id =
            resolve_subagent_parent_cache_id(self.lifecycle.lineage.as_deref(), &parent_session_id);
        let task_id = non_empty(data.get("task_id"))
            .or_else(|| non_empty(arguments.get("task_id")))
            .unwrap_or_default();
        let sub_session_id = non_empty(data.get("sub_session_id")).map_or_else(
            || resolve_sub_session_id(task_id, &parent_session_id, None),
            str::to_owned,
        );
        let subagent = KvCacheSubagent {
            subagent_id: subagent_id.to_owned(),
            subagent_type,
            sub_session_id,
            parent_cache_id,
        };
        if let Err(error) = self.lifecycle.prepare_subagent(subagent).await {
            tracing::warn!(
                subagent_id,
                error = %error,
                "harness kv_cache: preparing a subagent failed; ignoring"
            );
        }
    }
    /// Handles a successful `subagent_wait` payload.
    async fn on_wait(&self, data: &Value) {
        let Some(statuses) = data.get("statuses").and_then(Value::as_object) else {
            tracing::debug!("harness kv_cache: wait payload without statuses; skipping");
            return;
        };
        for (subagent_id, status) in statuses {
            let Some(succeeded) = terminal_success(status.as_str()) else {
                continue;
            };
            if let Err(error) = self.lifecycle.finish_subagent(subagent_id, succeeded).await {
                tracing::warn!(
                    subagent_id,
                    error = %error,
                    "harness kv_cache: finishing a subagent failed; ignoring"
                );
            }
        }
    }
    /// Handles a successful `subagent_close` payload.
    async fn on_close(&self, data: &Value, arguments: &Value) {
        let subagent_id =
            non_empty(data.get("subagent_id")).or_else(|| non_empty(arguments.get("subagent_id")));
        let Some(subagent_id) = subagent_id else {
            tracing::debug!("harness kv_cache: close payload without a subagent id; skipping");
            return;
        };
        if let Err(error) = self.lifecycle.evict_subagent(subagent_id).await {
            tracing::warn!(
                subagent_id,
                error = %error,
                "harness kv_cache: evicting a subagent failed; ignoring"
            );
        }
    }
}
#[async_trait]
impl HarnessRail for KvCacheLifecycleRail {
    fn name(&self) -> &'static str {
        "kv_cache_lifecycle"
    }
    fn priority(&self) -> i32 {
        rail_priority::DEFAULT
    }
    async fn after_tool_call(&self, ctx: &mut HarnessRailContext) -> Result<()> {
        if !self.lifecycle.affinity_enabled() {
            return Ok(());
        }
        let Some(call) = ctx.base.tool_call.as_ref() else {
            return Ok(());
        };
        let Some(result) = ctx.base.tool_result.as_ref() else {
            return Ok(());
        };
        if !result.success {
            return Ok(());
        }
        let Some(data) = result.output.as_ref().and_then(payload_data) else {
            return Ok(());
        };
        let arguments = parse_arguments(&call.arguments);
        match call.name.as_str() {
            "subagent_spawn" | "task_tool" | "task" => {
                self.on_spawn(ctx, data, &arguments).await;
            }
            "subagent_wait" => self.on_wait(data).await,
            "subagent_close" => self.on_close(data, &arguments).await,
            _ => {}
        }
        Ok(())
    }
}
/// Returns the `data` object of a successful [`crate::tools::ToolPayload`].
///
/// A `success: false` envelope, a missing `data`, or a non-object payload is
/// `None`: the rail then skips the frame with a debug log.
fn payload_data(payload: &Value) -> Option<&Value> {
    if payload.get("success").and_then(Value::as_bool) == Some(false) {
        return None;
    }
    payload.get("data").filter(|data| data.is_object())
}
/// Parses a tool call's JSON argument text, tolerating malformed fragments.
fn parse_arguments(arguments: &str) -> Value {
    serde_json::from_str(arguments).unwrap_or(Value::Null)
}
/// Returns `Some(trimmed)` for a non-blank JSON string.
fn non_empty(value: Option<&Value>) -> Option<&str> {
    value
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|text| !text.is_empty())
}
/// Maps a subagent status to whether the turn succeeded.
///
/// `completed` is the only success; `failed`/`errored`/`closed`/`cancelled`/
/// `interrupted` are terminal failures; anything else (still running, unknown)
/// is not ready to finish.
fn terminal_success(status: Option<&str>) -> Option<bool> {
    match status?.trim() {
        "completed" => Some(true),
        "failed" | "errored" | "closed" | "cancelled" | "interrupted" => Some(false),
        _ => None,
    }
}
#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::assert_is_empty,
        clippy::needless_pass_by_value
    )]
    use super::*;
    use openjiuwen_agent::RailEvent;
    use openjiuwen_agent::ToolCallResult;
    use openjiuwen_common::error::{Error, ErrorKind};
    use openjiuwen_foundation::prompt::ToolCall;
    use parking_lot::Mutex;
    use serde_json::json;
    /// A [`KvCacheProviderRegistry`] that records every call and can fail on demand.
    struct RecordingRegistry {
        calls: Mutex<Vec<String>>,
        fail: Mutex<bool>,
        backend: &'static str,
    }
    impl Default for RecordingRegistry {
        fn default() -> Self {
            Self {
                calls: Mutex::new(Vec::new()),
                fail: Mutex::new(false),
                backend: "recording",
            }
        }
    }
    impl RecordingRegistry {
        fn calls(&self) -> Vec<String> {
            self.calls.lock().clone()
        }
        fn set_fail(&self, fail: bool) {
            *self.fail.lock() = fail;
        }
        fn record(&self, operation: &str, cache_id: &str) -> Result<()> {
            self.calls.lock().push(format!("{operation}:{cache_id}"));
            if *self.fail.lock() {
                return Err(Error::config("provider unavailable"));
            }
            Ok(())
        }
    }
    #[async_trait::async_trait]
    impl KvCacheProviderRegistry for RecordingRegistry {
        fn backend(&self) -> &str {
            self.backend
        }
        async fn prepare(&self, cache_id: &str) -> Result<()> {
            self.record("prepare", cache_id)
        }
        async fn suspend(&self, cache_id: &str) -> Result<()> {
            self.record("suspend", cache_id)
        }
        async fn release(&self, cache_id: &str) -> Result<()> {
            self.record("release", cache_id)
        }
    }
    /// A lineage answering with a fixed identity.
    struct FixedLineage(&'static str);
    impl KvCacheLineage for FixedLineage {
        fn resolve_parent_cache_id(&self, _runtime_session_id: &str) -> Result<String> {
            Ok(self.0.to_string())
        }
    }
    fn on_config() -> KvCacheAffinityConfig {
        KvCacheAffinityConfig {
            enable_kv_cache_affinity: true,
        }
    }
    fn lifecycle(registry: Arc<RecordingRegistry>, affinity: bool) -> KvCacheSubagentLifecycle {
        KvCacheSubagentLifecycle::new(registry).with_config(KvCacheAffinityConfig {
            enable_kv_cache_affinity: affinity,
        })
    }
    fn subagent(subagent_type: &str) -> KvCacheSubagent {
        KvCacheSubagent {
            subagent_id: "sub-1".to_string(),
            subagent_type: subagent_type.to_string(),
            sub_session_id: "sub-1".to_string(),
            parent_cache_id: "cache-parent".to_string(),
        }
    }
    #[tokio::test]
    async fn affinity_off_makes_no_calls_and_stores_no_state() {
        let registry = Arc::new(RecordingRegistry::default());
        let lifecycle = lifecycle(Arc::clone(&registry), false);
        assert!(!lifecycle.affinity_enabled());
        assert!(lifecycle
            .prepare_subagent(subagent("verification_agent"))
            .await
            .unwrap()
            .is_none());
        assert!(lifecycle
            .finish_subagent("sub-1", true)
            .await
            .unwrap()
            .is_none());
        assert!(lifecycle.evict_subagent("sub-1").await.unwrap().is_none());
        assert!(lifecycle.release_all().await.unwrap().is_empty());
        assert!(lifecycle.sessions().is_empty());
        assert!(registry.calls().is_empty());
    }
    #[tokio::test]
    async fn affinity_on_sticky_prepares_and_suspends_on_success() {
        let registry = Arc::new(RecordingRegistry::default());
        let lifecycle = lifecycle(Arc::clone(&registry), true)
            .with_lineage(Arc::new(FixedLineage("cache-parent")));
        let prepared = lifecycle
            .prepare_subagent(subagent("verification_agent"))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(prepared.state, KvCacheSubagentState::Active);
        assert_eq!(prepared.actions, vec![KvCacheAction::Prepare]);
        let finished = lifecycle
            .finish_subagent("sub-1", true)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(finished.state, KvCacheSubagentState::Suspended);
        assert_eq!(
            finished.actions,
            vec![KvCacheAction::Prepare, KvCacheAction::Suspend]
        );
        assert_eq!(
            registry.calls(),
            vec!["prepare:cache-parent", "suspend:cache-parent"]
        );
    }
    #[tokio::test]
    async fn affinity_on_sticky_releases_on_failure() {
        let registry = Arc::new(RecordingRegistry::default());
        let lifecycle = lifecycle(Arc::clone(&registry), true);
        lifecycle
            .prepare_subagent(subagent("verification_agent"))
            .await
            .unwrap();
        let finished = lifecycle
            .finish_subagent("sub-1", false)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(finished.state, KvCacheSubagentState::Released);
        assert_eq!(
            finished.actions,
            vec![KvCacheAction::Prepare, KvCacheAction::Release]
        );
        assert_eq!(
            registry.calls(),
            vec!["prepare:cache-parent", "release:cache-parent"]
        );
    }
    #[tokio::test]
    async fn non_sticky_never_prepares_and_releases_on_success() {
        let registry = Arc::new(RecordingRegistry::default());
        let lifecycle = lifecycle(Arc::clone(&registry), true);
        let prepared = lifecycle
            .prepare_subagent(subagent("general-purpose"))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(prepared.state, KvCacheSubagentState::Active);
        assert!(prepared.actions.is_empty());
        let finished = lifecycle
            .finish_subagent("sub-1", true)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(finished.state, KvCacheSubagentState::Released);
        assert_eq!(finished.actions, vec![KvCacheAction::Release]);
        assert_eq!(registry.calls(), vec!["release:cache-parent"]);
    }
    #[tokio::test]
    async fn a_failing_prepare_leaves_no_state() {
        let registry = Arc::new(RecordingRegistry::default());
        registry.set_fail(true);
        let lifecycle = lifecycle(Arc::clone(&registry), true);
        assert!(lifecycle
            .prepare_subagent(subagent("verification_agent"))
            .await
            .is_err());
        assert!(lifecycle.get("sub-1").is_none());
    }
    #[tokio::test]
    async fn evict_releases_repeatedly_and_ignores_unknown_ids() {
        let registry = Arc::new(RecordingRegistry::default());
        let lifecycle = lifecycle(Arc::clone(&registry), true);
        lifecycle
            .prepare_subagent(subagent("verification_agent"))
            .await
            .unwrap();
        let first = lifecycle.evict_subagent("sub-1").await.unwrap().unwrap();
        assert_eq!(first.state, KvCacheSubagentState::Released);
        let second = lifecycle.evict_subagent("sub-1").await.unwrap().unwrap();
        assert_eq!(second.state, KvCacheSubagentState::Released);
        assert_eq!(
            second.actions,
            vec![
                KvCacheAction::Prepare,
                KvCacheAction::Release,
                KvCacheAction::Release
            ]
        );
        assert!(lifecycle.evict_subagent("missing").await.unwrap().is_none());
        assert_eq!(
            registry.calls(),
            vec![
                "prepare:cache-parent",
                "release:cache-parent",
                "release:cache-parent"
            ]
        );
    }
    #[tokio::test]
    async fn release_all_skips_already_released_sessions() {
        let registry = Arc::new(RecordingRegistry::default());
        let lifecycle = lifecycle(Arc::clone(&registry), true);
        for id in ["a", "b"] {
            lifecycle
                .prepare_subagent(KvCacheSubagent {
                    subagent_id: id.to_string(),
                    subagent_type: "general-purpose".to_string(),
                    sub_session_id: id.to_string(),
                    parent_cache_id: format!("cache-{id}"),
                })
                .await
                .unwrap();
        }
        lifecycle.finish_subagent("a", true).await.unwrap();
        let sessions = lifecycle.release_all().await.unwrap();
        assert_eq!(sessions.len(), 2);
        assert!(sessions
            .iter()
            .all(|session| session.state == KvCacheSubagentState::Released));
        assert_eq!(registry.calls(), vec!["release:cache-a", "release:cache-b"]);
        assert_eq!(lifecycle.release_all().await.unwrap().len(), 2);
        assert_eq!(registry.calls().len(), 2);
    }
    #[test]
    fn serde_round_trips_state_actions_and_history() {
        let session = KvCacheSubagentSession {
            subagent_id: "sub-1".to_string(),
            subagent_type: "verification_agent".to_string(),
            sub_session_id: "sub-1".to_string(),
            parent_cache_id: "cache-1".to_string(),
            state: KvCacheSubagentState::Suspended,
            actions: vec![KvCacheAction::Prepare, KvCacheAction::Suspend],
        };
        let encoded = serde_json::to_value(&session).unwrap();
        assert_eq!(encoded["state"], "suspended");
        assert_eq!(encoded["actions"], json!(["prepare", "suspend"]));
        let decoded: KvCacheSubagentSession = serde_json::from_value(json!({
            "subagent_id": "sub-1",
            "subagent_type": "verification_agent",
            "sub_session_id": "sub-1",
            "parent_cache_id": "cache-1",
            "state": "suspended",
            "actions": ["prepare", "suspend"],
            "future_field": true
        }))
        .unwrap();
        assert_eq!(decoded, session);
    }
    /// Builds an `AfterToolCall` context around one call/result pair.
    fn tool_context(name: &str, arguments: Value, payload: Value) -> HarnessRailContext {
        let call = ToolCall {
            name: name.to_string(),
            arguments: arguments.to_string(),
            ..ToolCall::default()
        };
        let mut ctx = HarnessRailContext::for_event(RailEvent::AfterToolCall);
        ctx.base.tool_call = Some(call.clone());
        ctx.base.tool_result = Some(ToolCallResult::success(call, payload, String::new()));
        ctx
    }
    fn rail(registry: Arc<RecordingRegistry>, affinity: bool) -> KvCacheLifecycleRail {
        let lifecycle = lifecycle(Arc::clone(&registry), affinity)
            .with_lineage(Arc::new(FixedLineage("cache-parent")));
        KvCacheLifecycleRail::new(Arc::new(lifecycle))
    }
    #[test]
    fn rail_name_and_priority() {
        let registry = Arc::new(RecordingRegistry::default());
        let rail = rail(registry, true);
        assert_eq!(rail.name(), "kv_cache_lifecycle");
        assert_eq!(rail.priority(), rail_priority::DEFAULT);
        assert!(rail.lifecycle().affinity_enabled());
    }
    #[tokio::test]
    async fn spawn_payload_prepares_the_child() {
        let registry = Arc::new(RecordingRegistry::default());
        let rail = rail(Arc::clone(&registry), true);
        let mut ctx = tool_context(
            "subagent_spawn",
            json!({"subagent_type": "verification_agent"}),
            json!({
                "success": true,
                "data": {"subagent_id": "sub-1", "sub_session_id": "sub-1", "task_id": "t-1"}
            }),
        );
        rail.after_tool_call(&mut ctx).await.unwrap();
        let session = rail.lifecycle().get("sub-1").unwrap();
        assert_eq!(session.subagent_type, "verification_agent");
        assert_eq!(session.sub_session_id, "sub-1");
        assert_eq!(registry.calls(), vec!["prepare:cache-parent"]);
    }
    #[tokio::test]
    async fn task_tool_payload_takes_the_type_from_the_arguments() {
        let registry = Arc::new(RecordingRegistry::default());
        let rail = rail(Arc::clone(&registry), true);
        let mut ctx = tool_context(
            "task_tool",
            json!({"subagent_type": "verification_agent", "task_description": "check"}),
            json!({
                "success": true,
                "data": {"output": "ok", "agent_id": "agent-7", "subagent_type": "verification_agent"}
            }),
        );
        rail.after_tool_call(&mut ctx).await.unwrap();
        let session = rail.lifecycle().get("agent-7").unwrap();
        assert_eq!(session.subagent_type, "verification_agent");
        assert_eq!(registry.calls(), vec!["prepare:cache-parent"]);
    }
    #[tokio::test]
    async fn wait_payload_finishes_every_terminal_result() {
        let registry = Arc::new(RecordingRegistry::default());
        let rail = rail(Arc::clone(&registry), true);
        for id in ["sub-1", "sub-2"] {
            let mut spawn = tool_context(
                "subagent_spawn",
                json!({"subagent_type": "verification_agent"}),
                json!({"success": true, "data": {"subagent_id": id, "sub_session_id": id}}),
            );
            rail.after_tool_call(&mut spawn).await.unwrap();
        }
        let mut wait = tool_context(
            "subagent_wait",
            json!({"subagent_ids": ["sub-1", "sub-2"]}),
            json!({
                "success": true,
                "data": {"statuses": {"sub-1": "completed", "sub-2": "errored", "sub-3": "running"}}
            }),
        );
        rail.after_tool_call(&mut wait).await.unwrap();
        assert_eq!(
            rail.lifecycle().get("sub-1").unwrap().state,
            KvCacheSubagentState::Suspended
        );
        assert_eq!(
            rail.lifecycle().get("sub-2").unwrap().state,
            KvCacheSubagentState::Released
        );
        assert!(rail.lifecycle().get("sub-3").is_none());
        assert_eq!(
            registry.calls(),
            vec![
                "prepare:cache-parent",
                "prepare:cache-parent",
                "suspend:cache-parent",
                "release:cache-parent"
            ]
        );
    }
    #[tokio::test]
    async fn close_payload_evicts_the_child() {
        let registry = Arc::new(RecordingRegistry::default());
        let rail = rail(Arc::clone(&registry), true);
        let mut spawn = tool_context(
            "subagent_spawn",
            json!({"subagent_type": "verification_agent"}),
            json!({"success": true, "data": {"subagent_id": "sub-1", "sub_session_id": "sub-1"}}),
        );
        rail.after_tool_call(&mut spawn).await.unwrap();
        let mut close = tool_context(
            "subagent_close",
            json!({"subagent_id": "sub-1"}),
            json!({"success": true, "data": {"subagent_id": "sub-1", "previous_status": "running"}}),
        );
        rail.after_tool_call(&mut close).await.unwrap();
        assert_eq!(
            rail.lifecycle().get("sub-1").unwrap().state,
            KvCacheSubagentState::Released
        );
        assert_eq!(
            registry.calls(),
            vec!["prepare:cache-parent", "release:cache-parent"]
        );
    }
    #[tokio::test]
    async fn failed_malformed_and_unknown_frames_are_ignored() {
        let registry = Arc::new(RecordingRegistry::default());
        let rail = rail(Arc::clone(&registry), true);
        let frames = [
            tool_context(
                "subagent_spawn",
                json!({"subagent_type": "verification_agent"}),
                json!({"success": false, "data": {"subagent_id": "sub-1"}}),
            ),
            tool_context(
                "subagent_spawn",
                json!({"subagent_type": "verification_agent"}),
                json!({"success": true, "data": {"unexpected": true}}),
            ),
            {
                let call = ToolCall {
                    name: "subagent_spawn".to_string(),
                    arguments: "{not json".to_string(),
                    ..ToolCall::default()
                };
                let mut ctx = HarnessRailContext::for_event(RailEvent::AfterToolCall);
                ctx.base.tool_call = Some(call.clone());
                ctx.base.tool_result = Some(ToolCallResult::success(
                    call,
                    json!({"success": true, "data": {"subagent_id": "sub-2", "subagent_type": "verification_agent"}}),
                    String::new(),
                ));
                ctx
            },
            tool_context(
                "read_file",
                json!({"path": "/tmp/x"}),
                json!({"success": true, "data": {"subagent_id": "sub-3"}}),
            ),
        ];
        for mut ctx in frames {
            rail.after_tool_call(&mut ctx).await.unwrap();
        }
        // The malformed-argument frame still prepares from the payload type.
        assert!(rail.lifecycle().get("sub-1").is_none());
        assert!(rail.lifecycle().get("sub-3").is_none());
        assert_eq!(registry.calls(), vec!["prepare:cache-parent"]);
    }
    #[tokio::test]
    async fn a_provider_error_is_swallowed() {
        let registry = Arc::new(RecordingRegistry::default());
        registry.set_fail(true);
        let rail = rail(Arc::clone(&registry), true);
        let mut ctx = tool_context(
            "subagent_spawn",
            json!({"subagent_type": "verification_agent"}),
            json!({"success": true, "data": {"subagent_id": "sub-1", "sub_session_id": "sub-1"}}),
        );
        assert!(rail.after_tool_call(&mut ctx).await.is_ok());
        assert!(rail.lifecycle().get("sub-1").is_none());
    }
    #[tokio::test]
    async fn the_rail_is_inert_when_affinity_is_off() {
        let registry = Arc::new(RecordingRegistry::default());
        let rail = rail(Arc::clone(&registry), false);
        let mut ctx = tool_context(
            "subagent_spawn",
            json!({"subagent_type": "verification_agent"}),
            json!({"success": true, "data": {"subagent_id": "sub-1", "sub_session_id": "sub-1"}}),
        );
        rail.after_tool_call(&mut ctx).await.unwrap();
        assert!(registry.calls().is_empty());
        assert!(rail.lifecycle().sessions().is_empty());
    }
    #[test]
    fn terminal_status_mapping() {
        assert_eq!(terminal_success(Some("completed")), Some(true));
        for status in ["failed", "errored", "closed", "cancelled", "interrupted"] {
            assert_eq!(terminal_success(Some(status)), Some(false));
        }
        for status in ["running", "pending_init", "not_found", "  "] {
            assert_eq!(terminal_success(Some(status)), None);
        }
        assert_eq!(terminal_success(None), None);
    }
    #[tokio::test]
    async fn unsupported_backend_is_reachable_through_the_registry_seam() {
        let backend = super::super::provider::UnsupportedKvCacheBackend::new("vllm");
        let lifecycle = KvCacheSubagentLifecycle::new(Arc::new(backend)).with_config(on_config());
        assert_eq!(lifecycle.registry().backend(), "vllm");
        assert_eq!(
            lifecycle
                .prepare_subagent(subagent("verification_agent"))
                .await
                .unwrap_err()
                .kind(),
            ErrorKind::Unsupported
        );
    }
}
