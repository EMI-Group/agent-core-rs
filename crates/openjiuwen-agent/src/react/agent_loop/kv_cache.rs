//! Best-effort KV-cache affinity bookkeeping around one model call.
//!
//! The loop touches the cache only through
//! [`KVCacheModelCallHook`](crate::kv_cache::KVCacheModelCallHook): before a
//! call it opens the request identity's scope (a prefetch, when the hook has a
//! provider action runtime attached), admits the inference, and records the
//! request prefix; after the call it ends the lease and records the provider's
//! input tokens. Every step is best effort — a failure is logged with
//! `tracing::debug!` and never reaches the run's output, because the cache is an
//! affinity hint, not a correctness requirement.
//!
//! A run without a hook
//! ([`ReActAgent::kv_cache_hook`](crate::react::ReActAgent::kv_cache_hook) is
//! `None`) does no work at all: [`prepare`] returns `None` immediately and
//! [`finish`] has nothing to end.
use crate::kv_cache::{KVCacheIdentity, KVCacheModelCallHook, KvCacheInferenceLease};
use crate::react::agent_loop::AgentRuntime;
use openjiuwen_foundation::{BaseMessage, ModelRequestConfig};
use openjiuwen_session::Session;
use std::sync::Arc;

/// One model call's KV-cache state: the hook, the request identity, and the
/// admission lease `begin_inference` granted.
pub(super) struct KvCacheCall {
    /// The hook the call is recorded through.
    hook: Arc<KVCacheModelCallHook>,
    /// The identity [`prepare`] used, repeated by [`finish`].
    identity: KVCacheIdentity,
    /// The inference lease, when a provider action runtime admitted one.
    lease: Option<KvCacheInferenceLease>,
}

/// Prepares the request's cache scope, admits the inference, and records the
/// prefix.
///
/// Returns `None` when the run has no hook, so every later step is skipped. The
/// scope is the session's own lineage (`parent_session_id = None` resolves to the
/// session itself); the recorded prefix covers the whole model window
/// (`messages`) and the model-visible tools (`visible_cards`), matching what the
/// request carries.
pub(super) async fn prepare(
    runtime: &AgentRuntime,
    session: &Arc<dyn Session>,
    messages: &[BaseMessage],
    request: &ModelRequestConfig,
) -> Option<KvCacheCall> {
    let hook = Arc::clone(runtime.kv_cache_hook.as_ref()?);
    let session_id = session.id();
    let model = request.model_name.as_str();
    let tools = &runtime.visible_cards;
    let identity = hook.identity(session_id, None, model, messages, tools);
    // `prepare` opens the scope and starts a prefetch when a provider action
    // runtime is attached; a bare hook has nothing to do and answers `false`.
    let _prepared = hook.prepare(&identity).await;
    let lease = hook.begin_inference(&identity).await;
    if let Err(error) = hook
        .affinity(session_id, None, model, messages, tools)
        .await
    {
        tracing::debug!(%error, "kv-cache prefix affinity was not recorded");
    }
    Some(KvCacheCall {
        hook,
        identity,
        lease,
    })
}

/// Ends the call's inference lease and records its input tokens on success.
///
/// `input_tokens` is the provider's report for the call that just finished;
/// it is only recorded when `succeeded` is `true`, so a failed call leaves the
/// previous estimate in place.
pub(super) async fn finish(call: Option<KvCacheCall>, succeeded: bool, input_tokens: Option<u64>) {
    let Some(call) = call else {
        return;
    };
    if let Some(lease) = call.lease.as_ref() {
        call.hook.end_inference(lease, succeeded).await;
    }
    if !succeeded {
        return;
    }
    let Some(tokens) = input_tokens else {
        return;
    };
    let estimate = usize::try_from(tokens).unwrap_or(usize::MAX);
    if let Err(error) = call.hook.note_usage(&call.identity, estimate).await {
        tracing::debug!(%error, "kv-cache token usage was not recorded");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_cache::{InProcessKVCacheRuntime, KVCacheAffinityConfig};

    fn hook(config: KVCacheAffinityConfig) -> Arc<KVCacheModelCallHook> {
        Arc::new(KVCacheModelCallHook::new(
            Arc::new(InProcessKVCacheRuntime::new(4)),
            config,
        ))
    }

    #[tokio::test]
    async fn finish_without_a_call_does_nothing() {
        // A run without a hook must not touch any cache state.
        finish(None, true, Some(3)).await;
        finish(None, false, None).await;
    }

    #[tokio::test]
    async fn finish_records_usage_only_for_a_successful_call() {
        let hook = hook(KVCacheAffinityConfig::enabled());
        let identity = hook.identity("s1", None, "gpt-test", &[BaseMessage::user("hi")], &[]);

        let failed = KvCacheCall {
            hook: Arc::clone(&hook),
            identity: identity.clone(),
            lease: None,
        };
        finish(Some(failed), false, Some(7)).await;
        assert_eq!(hook.stats().await.unwrap().entries, 0);

        let succeeded = KvCacheCall {
            hook: Arc::clone(&hook),
            identity: identity.clone(),
            lease: None,
        };
        finish(Some(succeeded), true, Some(7)).await;
        let entry = hook
            .runtime()
            .lookup(&identity)
            .await
            .unwrap()
            .entry
            .expect("the successful call recorded its identity");
        assert_eq!(entry.token_estimate, 7);
    }
}
