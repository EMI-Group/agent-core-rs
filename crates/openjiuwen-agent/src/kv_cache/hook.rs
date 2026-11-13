//! The model-call hook: prefix affinity around one model request.
//!
//! Ports the ReAct-facing half of `openjiuwen/core/single_agent/kv_cache/`
//! (`kv_cache_react_model_call_hook.py`, `kv_cache_child_session.py`) onto the
//! in-process runtime: before a model call the hook asks the runtime which
//! leading messages are still cached, records the request the model is about to
//! send, and later accepts a token estimate for it. It also renders the child
//! lineage kwargs that make a subagent's request branch from its parent instead
//! of starting a fresh cache.
//!
//! Affinity is best-effort: a disabled config performs no runtime work at all,
//! and every recorded prefix is an optimization hint, never a correctness
//! requirement.
use crate::kv_cache::config::KVCacheAffinityConfig;
use crate::kv_cache::metadata::KV_CACHE_AFFINITY_PARENT_SESSION_ID_ENV;
use crate::kv_cache::runtime::prefix_hash_states;
use crate::kv_cache::types::{
    KVCacheEntry, KVCacheIdentity, KVCacheLookup, KVCacheRuntimeProtocol, KVCacheStats,
};
use openjiuwen_common::error::Result;
use openjiuwen_foundation::{BaseMessage, ToolCard};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::Arc;

/// What the host may do with the KV cache on this call.
///
/// `affinity` and `evict` follow [`KVCacheAffinityConfig::enabled`] (the runtime
/// supports per-session eviction). `offload` stays `false`: provider-side
/// prefetch/offload is deferred, so a host must not schedule one.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct KVCacheCallCapabilities {
    /// Whether prefix affinity lookup/record is active.
    pub affinity: bool,
    /// Whether provider-side offload is available (never, today).
    pub offload: bool,
    /// Whether per-session eviction is available.
    pub evict: bool,
}

impl KVCacheCallCapabilities {
    /// Capabilities for a disabled affinity config: nothing is available.
    #[must_use]
    pub const fn none() -> Self {
        Self {
            affinity: false,
            offload: false,
            evict: false,
        }
    }
}

/// Records and matches request prefixes around model calls.
///
/// The hook owns no prefix state itself: everything lives in the
/// [`KVCacheRuntimeProtocol`] instance it was built with, so several agents (or
/// a child session) sharing one runtime see the same cached prefixes.
pub struct KVCacheModelCallHook {
    runtime: Arc<dyn KVCacheRuntimeProtocol>,
    config: KVCacheAffinityConfig,
}

impl KVCacheModelCallHook {
    /// Creates a hook over `runtime` with `config`.
    #[must_use]
    pub fn new(runtime: Arc<dyn KVCacheRuntimeProtocol>, config: KVCacheAffinityConfig) -> Self {
        Self { runtime, config }
    }

    /// The runtime the hook records into.
    #[must_use]
    pub fn runtime(&self) -> Arc<dyn KVCacheRuntimeProtocol> {
        Arc::clone(&self.runtime)
    }

    /// The affinity configuration.
    #[must_use]
    pub const fn config(&self) -> &KVCacheAffinityConfig {
        &self.config
    }

    /// What this hook supports for the configured affinity settings.
    #[must_use]
    pub const fn capabilities(&self) -> KVCacheCallCapabilities {
        KVCacheCallCapabilities {
            affinity: self.config.enabled,
            offload: false,
            evict: self.config.enabled,
        }
    }

    /// Builds the identity of the request described by `messages` and `tools`.
    ///
    /// `parent_session_id` defaults to `session_id`, so a root request is its own
    /// lineage; the prefix hash covers the whole message list (see
    /// [`crate::kv_cache::runtime::stable_prefix_hash`]).
    #[must_use]
    pub fn identity(
        &self,
        session_id: &str,
        parent_session_id: Option<&str>,
        model: &str,
        messages: &[BaseMessage],
        tools: &[ToolCard],
    ) -> KVCacheIdentity {
        KVCacheIdentity {
            session_id: session_id.trim().to_owned(),
            parent_session_id: Some(parent_or_self(session_id, parent_session_id)),
            prefix_hash: crate::kv_cache::runtime::stable_prefix_hash(messages, tools, model),
            model: model.to_owned(),
            message_count: messages.len(),
            tool_count: tools.len(),
        }
    }

    /// Looks up the longest reusable prefix of this request and records it.
    ///
    /// Probe order is longest-first: the full request first (the common case of
    /// a retried or repeated call), then every shorter message prefix, so an
    /// append-only conversation reports the previous turn's prefix length
    /// (`first_changed_index` of the two message lists). The first hit is
    /// re-recorded with its hit counter bumped and the current request is
    /// recorded for the next call.
    ///
    /// A disabled config returns a miss without touching the runtime; so does an
    /// empty session id or an empty message list, because neither names a
    /// reusable prefix.
    ///
    /// # Errors
    ///
    /// Propagates the runtime's lookup/record error. The in-process runtime never
    /// fails, so this is only meaningful for a backend that does.
    pub async fn affinity(
        &self,
        session_id: &str,
        parent_session_id: Option<&str>,
        model: &str,
        messages: &[BaseMessage],
        tools: &[ToolCard],
    ) -> Result<KVCacheLookup> {
        if !self.config.enabled || session_id.trim().is_empty() || messages.is_empty() {
            return Ok(KVCacheLookup::miss());
        }
        let parent = parent_or_self(session_id, parent_session_id);
        let hashes = prefix_hash_states(messages, tools, model);
        for message_count in (1..=messages.len()).rev() {
            let Some(prefix_hash) = hashes.get(message_count).cloned() else {
                continue;
            };
            let identity = KVCacheIdentity {
                session_id: session_id.trim().to_owned(),
                parent_session_id: Some(parent.clone()),
                prefix_hash,
                model: model.to_owned(),
                message_count,
                tool_count: tools.len(),
            };
            let lookup = self.runtime.lookup(&identity).await?;
            let is_hit = lookup.hit;
            let Some(entry) = lookup.entry.filter(|_| is_hit) else {
                continue;
            };
            let reused_prefix_len = entry.prefix_len;
            let updated = entry.bumped();
            self.runtime.record(updated.clone()).await?;
            if message_count < messages.len() {
                self.record_full_request(session_id, &parent, model, messages, tools, &hashes)
                    .await?;
            }
            return Ok(KVCacheLookup::hit(updated, reused_prefix_len));
        }
        self.record_full_request(session_id, &parent, model, messages, tools, &hashes)
            .await?;
        Ok(KVCacheLookup::miss())
    }

    /// Records the token estimate of a prefix after the model call reported its
    /// usage.
    ///
    /// `identity` is the one [`Self::affinity`] recorded (see [`Self::identity`]);
    /// the estimate never regresses, and a disabled config is a no-op.
    ///
    /// # Errors
    ///
    /// Propagates the runtime's record error.
    pub async fn note_usage(
        &self,
        identity: &KVCacheIdentity,
        token_estimate: usize,
    ) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }
        let entry = KVCacheEntry::new(identity.clone(), identity.message_count)
            .with_token_estimate(token_estimate);
        self.runtime.record(entry).await
    }

    /// The runtime's current counters.
    ///
    /// # Errors
    ///
    /// Propagates the runtime's statistics error.
    pub async fn stats(&self) -> Result<KVCacheStats> {
        self.runtime.stats().await
    }

    /// Renders the child-session kwargs that carry `parent_session_id` lineage.
    ///
    /// Mirrors Python's `build_child_session_kwargs`: the child session inherits
    /// the parent's env bag with
    /// [`KV_CACHE_AFFINITY_PARENT_SESSION_ID_ENV`] pointing at the parent cache
    /// id, plus the explicit `parent_session_id`. The runtime handle travels as
    /// an `Arc` host-side and therefore has no JSON representation.
    #[must_use]
    pub fn build_child_session_kwargs(parent_session_id: &str) -> Value {
        let parent = parent_session_id.trim();
        json!({
            "parent_session_id": parent,
            "envs": { KV_CACHE_AFFINITY_PARENT_SESSION_ID_ENV: parent },
        })
    }

    /// Records the full request so the next call can reuse it.
    async fn record_full_request(
        &self,
        session_id: &str,
        parent: &str,
        model: &str,
        messages: &[BaseMessage],
        tools: &[ToolCard],
        hashes: &[String],
    ) -> Result<()> {
        let Some(prefix_hash) = hashes.last().cloned() else {
            return Ok(());
        };
        let identity = KVCacheIdentity {
            session_id: session_id.trim().to_owned(),
            parent_session_id: Some(parent.to_owned()),
            prefix_hash,
            model: model.to_owned(),
            message_count: messages.len(),
            tool_count: tools.len(),
        };
        self.runtime
            .record(KVCacheEntry::new(identity, messages.len()))
            .await
    }
}

/// The explicit parent, or the trimmed session id when it is absent.
fn parent_or_self(session_id: &str, parent_session_id: Option<&str>) -> String {
    match parent_session_id.map(str::trim) {
        Some(parent) if !parent.is_empty() => parent.to_owned(),
        _ => session_id.trim().to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_cache::config::KVCacheAffinityConfig;
    use crate::kv_cache::metadata::first_changed_index;
    use crate::kv_cache::runtime::InProcessKVCacheRuntime;
    use async_trait::async_trait;
    use openjiuwen_common::error::Result;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Runtime that counts calls so a bypass is observable.
    #[derive(Default)]
    struct CountingRuntime {
        calls: AtomicUsize,
    }

    #[async_trait]
    impl KVCacheRuntimeProtocol for CountingRuntime {
        async fn record(&self, _entry: KVCacheEntry) -> Result<()> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        async fn lookup(&self, _identity: &KVCacheIdentity) -> Result<KVCacheLookup> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(KVCacheLookup::miss())
        }

        async fn evict_session(&self, _session_id: &str) -> Result<usize> {
            Ok(0)
        }

        async fn stats(&self) -> Result<KVCacheStats> {
            Ok(KVCacheStats::default())
        }
    }

    fn enabled_hook(runtime: Arc<InProcessKVCacheRuntime>) -> KVCacheModelCallHook {
        KVCacheModelCallHook::new(runtime, KVCacheAffinityConfig::enabled())
    }

    fn tool(name: &str) -> ToolCard {
        ToolCard {
            name: name.to_owned(),
            description: format!("{name} tool"),
            ..ToolCard::default()
        }
    }

    async fn turn_one(hook: &KVCacheModelCallHook, tools: &[ToolCard]) -> Vec<BaseMessage> {
        let messages = vec![
            BaseMessage::system("be helpful"),
            BaseMessage::user("hello"),
        ];
        let lookup = hook
            .affinity("session-a", None, "gpt-test", &messages, tools)
            .await
            .unwrap();
        assert!(!lookup.is_hit());
        messages
    }

    #[tokio::test]
    async fn disabled_config_bypasses_the_runtime() {
        let runtime = Arc::new(CountingRuntime::default());
        let hook = KVCacheModelCallHook::new(runtime.clone(), KVCacheAffinityConfig::default());

        let messages = vec![BaseMessage::user("hello")];
        let lookup = hook
            .affinity("session-a", None, "gpt-test", &messages, &[])
            .await
            .unwrap();
        assert!(!lookup.is_hit());
        assert_eq!(runtime.calls.load(Ordering::SeqCst), 0);

        hook.note_usage(
            &hook.identity("session-a", None, "gpt-test", &messages, &[]),
            10,
        )
        .await
        .unwrap();
        assert_eq!(runtime.calls.load(Ordering::SeqCst), 0);
        assert_eq!(hook.capabilities(), KVCacheCallCapabilities::none());
    }

    #[tokio::test]
    async fn empty_session_or_messages_never_touch_the_runtime() {
        let runtime = Arc::new(CountingRuntime::default());
        let hook = KVCacheModelCallHook::new(runtime.clone(), KVCacheAffinityConfig::enabled());

        assert!(!hook
            .affinity("  ", None, "gpt-test", &[BaseMessage::user("hi")], &[])
            .await
            .unwrap()
            .is_hit());
        assert!(!hook
            .affinity("session-a", None, "gpt-test", &[], &[])
            .await
            .unwrap()
            .is_hit());
        assert_eq!(runtime.calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn second_turn_reuses_the_first_turn_prefix() {
        let runtime = Arc::new(InProcessKVCacheRuntime::new(16));
        let hook = enabled_hook(runtime.clone());
        let tools = vec![tool("search")];
        let first = turn_one(&hook, &tools).await;

        let second = vec![
            BaseMessage::system("be helpful"),
            BaseMessage::user("hello"),
            BaseMessage::assistant("hi"),
            BaseMessage::user("again"),
        ];
        assert_eq!(first_changed_index(&first, &second), first.len());

        let lookup = hook
            .affinity("session-a", None, "gpt-test", &second, &tools)
            .await
            .unwrap();
        assert!(lookup.is_hit());
        assert_eq!(lookup.reused_prefix_len, first.len());
        assert_eq!(
            lookup.reused_prefix_len,
            first_changed_index(&first, &second)
        );
        assert_eq!(lookup.entry.unwrap().identity.message_count, first.len());

        let stats = hook.stats().await.unwrap();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.entries, 2);
    }

    #[tokio::test]
    async fn identical_request_is_a_hit_with_the_full_prefix() {
        let runtime = Arc::new(InProcessKVCacheRuntime::new(8));
        let hook = enabled_hook(runtime);
        let messages = vec![
            BaseMessage::system("be helpful"),
            BaseMessage::user("hello"),
        ];

        assert!(!hook
            .affinity("session-a", None, "gpt-test", &messages, &[])
            .await
            .unwrap()
            .is_hit());
        let lookup = hook
            .affinity("session-a", None, "gpt-test", &messages, &[])
            .await
            .unwrap();
        assert!(lookup.is_hit());
        assert_eq!(lookup.reused_prefix_len, messages.len());
        assert_eq!(lookup.entry.unwrap().hits, 1);
    }

    #[tokio::test]
    async fn changed_message_tool_or_model_misses() {
        let runtime = Arc::new(InProcessKVCacheRuntime::new(8));
        let hook = enabled_hook(runtime);
        let tools = vec![tool("search")];
        let first = turn_one(&hook, &tools).await;

        let edited = vec![
            BaseMessage::system("be helpful"),
            BaseMessage::user("something else"),
        ];
        assert!(!hook
            .affinity("session-a", None, "gpt-test", &edited, &tools)
            .await
            .unwrap()
            .is_hit());
        assert!(!hook
            .affinity("session-a", None, "gpt-test", &first, &[tool("read")])
            .await
            .unwrap()
            .is_hit());
        assert!(!hook
            .affinity("session-a", None, "gpt-other", &first, &tools)
            .await
            .unwrap()
            .is_hit());
    }

    #[tokio::test]
    async fn different_lineage_does_not_share_prefixes() {
        let runtime = Arc::new(InProcessKVCacheRuntime::new(8));
        let hook = enabled_hook(runtime);
        let messages = vec![
            BaseMessage::system("be helpful"),
            BaseMessage::user("hello"),
        ];
        let _ = hook
            .affinity("session-a", Some("root"), "gpt-test", &messages, &[])
            .await
            .unwrap();
        assert!(!hook
            .affinity("session-b", Some("root"), "gpt-test", &messages, &[])
            .await
            .unwrap()
            .is_hit());
    }

    #[tokio::test]
    async fn note_usage_stores_the_token_estimate() {
        let runtime = Arc::new(InProcessKVCacheRuntime::new(8));
        let hook = enabled_hook(runtime.clone());
        let messages = vec![
            BaseMessage::system("be helpful"),
            BaseMessage::user("hello"),
        ];
        let identity = hook.identity("session-a", None, "gpt-test", &messages, &[]);
        assert_eq!(identity.message_count, 2);
        assert_eq!(identity.parent(), Some("session-a"));

        hook.affinity("session-a", None, "gpt-test", &messages, &[])
            .await
            .unwrap();
        hook.note_usage(&identity, 96).await.unwrap();

        let entry = runtime.entries().remove(0);
        assert_eq!(entry.token_estimate, 96);
        assert_eq!(entry.prefix_len, 2);
    }

    #[tokio::test]
    async fn capabilities_follow_the_config() {
        let hook = enabled_hook(Arc::new(InProcessKVCacheRuntime::new(1)));
        assert_eq!(
            hook.capabilities(),
            KVCacheCallCapabilities {
                affinity: true,
                offload: false,
                evict: true,
            }
        );
        assert_eq!(
            KVCacheCallCapabilities::none(),
            KVCacheCallCapabilities::default()
        );
        assert!(hook.config().is_enabled());
    }

    #[test]
    fn child_session_kwargs_carry_the_parent_lineage() {
        assert_eq!(
            KVCacheModelCallHook::build_child_session_kwargs(" parent-cache "),
            json!({
                "parent_session_id": "parent-cache",
                "envs": { KV_CACHE_AFFINITY_PARENT_SESSION_ID_ENV: "parent-cache" },
            })
        );
    }

    #[test]
    fn identity_defaults_parent_and_counts() {
        let hook = enabled_hook(Arc::new(InProcessKVCacheRuntime::new(1)));
        let messages = vec![BaseMessage::system("sys")];
        let identity = hook.identity("session-a", None, "gpt-test", &messages, &[tool("search")]);
        assert_eq!(identity.session_id, "session-a");
        assert_eq!(identity.parent(), Some("session-a"));
        assert_eq!(identity.message_count, 1);
        assert_eq!(identity.tool_count, 1);
        assert_eq!(identity.prefix_hash.len(), 16);

        let explicit = hook.identity("child", Some("root"), "gpt-test", &messages, &[]);
        assert_eq!(explicit.parent(), Some("root"));
    }
}
