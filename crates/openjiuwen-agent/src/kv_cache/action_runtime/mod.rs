//! Lifecycle state machine that runs provider-side KV-cache actions.
//!
//! Ports the `KVCacheRuntime` executor of
//! `openjiuwen/core/kv_cache/kv_cache_runtime.py` without its control domains and
//! model-client handles: a host registers a cache scope, admits inference
//! ([`KvCacheActionRuntime::begin_inference`]), pauses it
//! ([`KvCacheActionRuntime::suspend`] runs an offload once the scope is idle),
//! resumes it ([`KvCacheActionRuntime::prepare`] runs a prefetch), and drops it
//! ([`KvCacheActionRuntime::release`] runs an evict).
//!
//! # State
//!
//! - A **binding** is one registered cache id with its lineage, model, revision,
//!   and [`KvCacheResidency`]; the lineage index (`parent` -> cache ids) makes a
//!   root action cover every binding below it.
//! - **Admission** per scope blocks new inference while an offload runs
//!   ([`KvCacheAdmission::Blocked`]) and refuses it after release
//!   ([`KvCacheAdmission::Terminal`]); a prefetch reopens the scope as soon as
//!   its provider call starts.
//! - **Fail-open** is the invariant that inference never waits on cache
//!   management: a provider error, timeout, or an action that cannot make
//!   progress marks the scope `fail_open`, reopens admission, and is counted in
//!   [`KvCacheActionStats::failures`] instead of being returned.
//! - Actions of one lineage are serialised by a per-lineage ordering lock in
//!   enqueue order, so a prefetch enqueued after an offload runs after it, and
//!   offload/evict additionally wait for the scope's active inference to reach
//!   zero within the configured budget.
//!
//! Every provider error is absorbed (logged with `tracing::warn!` and counted)
//! because the cache is a best-effort affinity hint: a provider that cannot
//! prefetch must not fail, block, or slow down the model call it describes.

/// Whether new inference may enter a cache scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KvCacheAdmission {
    /// Inference may enter; the scope is resident or its prefetch is running.
    Open,
    /// An offload is in progress or queued; inference must prefetch first.
    Blocked,
    /// The scope was released; no action other than eviction applies.
    Terminal,
}

/// Where a cache scope currently lives.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KvCacheResidency {
    /// The provider holds the scope in hot storage.
    Resident,
    /// The provider moved the scope out of hot storage.
    Offloaded,
    /// Residency has not been observed yet.
    #[default]
    Unknown,
}

/// Snapshot of one registered cache scope.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct KvCacheBinding {
    /// Cache id the scope is registered under.
    pub cache_id: String,
    /// Lineage the scope branches from (equals `cache_id` for a root scope).
    pub parent_cache_id: String,
    /// Model the scope was built for; empty when the caller had none.
    pub model: String,
    /// Registration revision, bumped by every
    /// [`KvCacheActionRuntime::register_binding`] and by a lineage move.
    pub revision: u64,
    /// Where the provider currently holds the scope.
    pub residency: KvCacheResidency,
}

/// An admitted inference on a cache scope.
///
/// The lease records the scope it holds so [`KvCacheActionRuntime::end_inference`]
/// can drop both counters; it is released once, so ending it twice is a no-op.
#[derive(Debug)]
pub struct KvCacheInferenceLease {
    cache_id: String,
    parent_cache_id: String,
    released: AtomicBool,
}

impl KvCacheInferenceLease {
    /// Creates an unreleased lease for the scope `cache_id` inside
    /// `parent_cache_id`.
    fn new(cache_id: String, parent_cache_id: String) -> Self {
        Self {
            cache_id,
            parent_cache_id,
            released: AtomicBool::new(false),
        }
    }

    /// Whether the lease has been ended already.
    #[must_use]
    pub fn is_released(&self) -> bool {
        self.released.load(Ordering::SeqCst)
    }
}

mod engine;

use self::engine::{
    enqueue, identity_mapping, prepare_key, primary_key, wait_outcome, ActionKey, Inner, State,
};
use crate::kv_cache::actions::{KvCacheAction, KvCacheActionProvider, KvCacheActionStats};
use crate::kv_cache::config::KVCacheAffinityConfig;
use crate::kv_cache::types::KVCacheIdentity;
use futures::future::join_all;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::Notify;
use tracing::warn;

/// Executor for provider-side KV-cache actions on session scopes.
///
/// Construction is inert; every lifecycle method registers the identity it is
/// called with (so a host may call them standalone) and is inert as a whole
/// while [`KVCacheAffinityConfig::enabled`] is `false`.
#[derive(Clone)]
pub struct KvCacheActionRuntime {
    inner: Arc<Inner>,
}

impl KvCacheActionRuntime {
    /// Creates a runtime that performs actions through `provider`.
    #[must_use]
    pub fn new(provider: Arc<dyn KvCacheActionProvider>, config: KVCacheAffinityConfig) -> Self {
        Self {
            inner: Arc::new(Inner {
                provider,
                config,
                state: Mutex::new(State::default()),
                stats: Mutex::new(KvCacheActionStats::default()),
                notify: Notify::new(),
                lineages: Mutex::new(HashMap::new()),
                sequence: AtomicU64::new(0),
            }),
        }
    }

    /// The provider this runtime performs actions through.
    #[must_use]
    pub fn provider(&self) -> Arc<dyn KvCacheActionProvider> {
        Arc::clone(&self.inner.provider)
    }

    /// The affinity configuration that supplies the budgets.
    #[must_use]
    pub fn config(&self) -> &KVCacheAffinityConfig {
        &self.inner.config
    }

    /// The action counters.
    // The lifecycle surface is async by contract (it mirrors Python's
    // coroutines), so these methods keep their `async` even without an await.
    #[allow(clippy::unused_async, clippy::unused_async_trait_impl)]
    pub async fn stats(&self) -> KvCacheActionStats {
        *self.inner.stats.lock()
    }

    /// The registered binding of `identity`, when it has one.
    #[must_use]
    pub fn binding(&self, identity: &KVCacheIdentity) -> Option<KvCacheBinding> {
        let (cache_id, _) = identity_mapping(identity)?;
        let state = self.inner.state.lock();
        state.bindings.get(&cache_id).map(|binding| KvCacheBinding {
            cache_id,
            parent_cache_id: binding.parent_cache_id.clone(),
            model: binding.model.clone(),
            revision: binding.revision,
            residency: binding.residency,
        })
    }

    /// The admission of `identity`'s primary scope (binding for a child, root
    /// for a root identity).
    #[must_use]
    pub fn admission(&self, identity: &KVCacheIdentity) -> Option<KvCacheAdmission> {
        let (cache_id, parent_cache_id) = identity_mapping(identity)?;
        let key = primary_key(&cache_id, &parent_cache_id);
        self.inner
            .state
            .lock()
            .action_states
            .get(&key)
            .map(|action| action.admission)
    }

    /// Whether `identity`'s primary scope failed open.
    #[must_use]
    pub fn fail_open(&self, identity: &KVCacheIdentity) -> bool {
        let Some((cache_id, parent_cache_id)) = identity_mapping(identity) else {
            return false;
        };
        let key = primary_key(&cache_id, &parent_cache_id);
        self.inner
            .state
            .lock()
            .action_states
            .get(&key)
            .is_some_and(|action| action.fail_open)
    }

    /// Active inferences on `identity`'s primary scope.
    #[must_use]
    pub fn active_inference(&self, identity: &KVCacheIdentity) -> usize {
        let Some((cache_id, parent_cache_id)) = identity_mapping(identity) else {
            return 0;
        };
        let key = primary_key(&cache_id, &parent_cache_id);
        self.inner
            .state
            .lock()
            .action_states
            .get(&key)
            .map_or(0, |action| action.active_inference)
    }

    /// Registers (or refreshes) `identity`'s binding and returns its snapshot.
    ///
    /// Every call bumps [`KvCacheBinding::revision`], which invalidates the
    /// results of actions that snapshotted the previous revision. Returns `None`
    /// for an invalid identity. Registering is bookkeeping only (no provider call
    /// is scheduled), so it stays available while affinity is disabled.
    #[allow(clippy::unused_async, clippy::unused_async_trait_impl)]
    pub async fn register_binding(&self, identity: &KVCacheIdentity) -> Option<KvCacheBinding> {
        let (cache_id, parent_cache_id) = identity_mapping(identity)?;
        Some(
            self.inner
                .bind(&cache_id, &parent_cache_id, &identity.model, true),
        )
    }

    /// Opens `identity`'s scope for inference, prefetching when needed.
    ///
    /// Returns whether the scope is ready to admit inference. A pending offload
    /// whose provider call has not started is cancelled; a pending offload that
    /// already started is left to finish, and the prefetch is ordered behind it.
    pub async fn prepare(&self, identity: &KVCacheIdentity) -> bool {
        if !self.inner.config.enabled {
            return false;
        }
        let Some((cache_id, parent_cache_id)) = identity_mapping(identity) else {
            return false;
        };
        self.inner
            .bind(&cache_id, &parent_cache_id, &identity.model, false);
        let keys = self.inner.affected_keys(&cache_id, &parent_cache_id);
        if keys.is_empty() {
            return false;
        }
        let mut prepared = false;
        for key in self.inner.prepare_keys(keys) {
            let result = prepare_key(&self.inner, &key).await;
            prepared |= result;
        }
        prepared
    }

    /// Blocks `identity`'s scope and offloads it once its inference is idle.
    ///
    /// Returns whether an offload was enqueued: an already offloaded, idle scope
    /// and a scope that failed open are skipped (counted in
    /// [`KvCacheActionStats::skipped`]).
    // The lifecycle surface is async by contract (it mirrors Python's
    // coroutines), so these methods keep their `async` even without an await.
    #[allow(clippy::unused_async, clippy::unused_async_trait_impl)]
    pub async fn suspend(&self, identity: &KVCacheIdentity) -> bool {
        if !self.inner.config.enabled {
            return false;
        }
        let Some((cache_id, parent_cache_id)) = identity_mapping(identity) else {
            return false;
        };
        self.inner
            .bind(&cache_id, &parent_cache_id, &identity.model, false);
        let mut scheduled = false;
        for key in self.inner.affected_keys(&cache_id, &parent_cache_id) {
            let mut state = self.inner.state.lock();
            let admission = state.action(&key).admission;
            if admission == KvCacheAdmission::Terminal {
                continue;
            }
            if state
                .action_states
                .get(&key)
                .is_some_and(|action| action.fail_open)
            {
                state.action(&key).admission = KvCacheAdmission::Open;
                drop(state);
                self.inner.stats.lock().record_skipped();
                self.inner.notify.notify_waiters();
                continue;
            }
            let bindings = state.bindings_for(&key);
            let already_offloaded = !bindings.is_empty()
                && bindings
                    .iter()
                    .all(|binding| binding.residency == KvCacheResidency::Offloaded);
            let idle = state
                .action_states
                .get(&key)
                .is_some_and(|action| action.pending.is_none());
            if admission == KvCacheAdmission::Blocked && idle && already_offloaded {
                drop(state);
                self.inner.stats.lock().record_skipped();
                continue;
            }
            state.action(&key).admission = KvCacheAdmission::Blocked;
            if enqueue(&self.inner, &mut state, &key, KvCacheAction::Offload).is_some() {
                scheduled = true;
            }
        }
        scheduled
    }

    /// Marks `identity`'s scope terminal, evicts it, and forgets the binding.
    ///
    /// Waits for the eviction bounded by the prefetch and evict budgets; returns
    /// whether any provider eviction reported success.
    pub async fn release(&self, identity: &KVCacheIdentity) -> bool {
        if !self.inner.config.enabled {
            return false;
        }
        let Some((cache_id, parent_cache_id)) = identity_mapping(identity) else {
            return false;
        };
        self.inner
            .bind(&cache_id, &parent_cache_id, &identity.model, false);
        let mut outcomes = Vec::new();
        for key in self.inner.affected_keys(&cache_id, &parent_cache_id) {
            let mut state = self.inner.state.lock();
            state.action(&key).admission = KvCacheAdmission::Terminal;
            if let Some(waiters) = enqueue(&self.inner, &mut state, &key, KvCacheAction::Evict) {
                outcomes.push(waiters.outcome);
            }
        }
        let succeeded = if outcomes.is_empty() {
            false
        } else {
            let budget = self.inner.config.effective_offload_prefetch_timeout()
                + self.inner.config.effective_evict_timeout();
            if let Ok(results) =
                tokio::time::timeout(budget, join_all(outcomes.into_iter().map(wait_outcome))).await
            {
                results.into_iter().any(|value| value == Some(true))
            } else {
                warn!(
                    cache_id = %cache_id,
                    "kv-cache session eviction timed out; forgetting the binding anyway"
                );
                false
            }
        };
        self.inner.forget(&cache_id, &parent_cache_id);
        succeeded
    }

    /// Registers `identity`, waits out any pending offload, and admits an
    /// inference.
    ///
    /// A scope that cannot be prepared is failed open (admission reopened,
    /// [`Self::fail_open`] set) so inference proceeds without affinity rather
    /// than waiting. Returns `None` when affinity is disabled, the identity is
    /// invalid, or the scope is terminal.
    pub async fn begin_inference(
        &self,
        identity: &KVCacheIdentity,
    ) -> Option<KvCacheInferenceLease> {
        if !self.inner.config.enabled {
            return None;
        }
        let binding = self.register_binding(identity).await?;
        let child = ActionKey::Binding(binding.cache_id.clone());
        let root = ActionKey::Root(binding.parent_cache_id.clone());
        for _ in 0..2 {
            let mut blocked: Vec<ActionKey> = Vec::new();
            {
                let mut state = self.inner.state.lock();
                let child_admission = state.action(&child).admission;
                let root_admission = state.action(&root).admission;
                if child_admission == KvCacheAdmission::Terminal
                    || root_admission == KvCacheAdmission::Terminal
                {
                    return None;
                }
                if root_admission == KvCacheAdmission::Blocked {
                    blocked.push(root.clone());
                }
                if child_admission == KvCacheAdmission::Blocked {
                    blocked.push(child.clone());
                }
                if blocked.is_empty() {
                    state.action(&child).active_inference += 1;
                    state.action(&root).active_inference += 1;
                    drop(state);
                    return Some(KvCacheInferenceLease::new(
                        binding.cache_id,
                        binding.parent_cache_id,
                    ));
                }
            }
            let mut prepared = true;
            for key in &blocked {
                let result = prepare_key(&self.inner, key).await;
                prepared &= result;
            }
            if !prepared {
                self.inner.fail_open_keys(&blocked);
            }
        }
        self.inner.fail_open_keys(&[root.clone(), child.clone()]);
        {
            let mut state = self.inner.state.lock();
            state.action(&child).active_inference += 1;
            state.action(&root).active_inference += 1;
        }
        Some(KvCacheInferenceLease::new(
            binding.cache_id,
            binding.parent_cache_id,
        ))
    }

    /// Ends `lease`'s inference, updating the scope's residency.
    ///
    /// Ending a lease twice is a no-op; a successful inference marks the scope
    /// resident, an unsuccessful one marks its residency unknown.
    // The lifecycle surface is async by contract (it mirrors Python's
    // coroutines), so these methods keep their `async` even without an await.
    #[allow(clippy::unused_async, clippy::unused_async_trait_impl)]
    pub async fn end_inference(&self, lease: &KvCacheInferenceLease, succeeded: bool) {
        if lease.released.swap(true, Ordering::SeqCst) {
            return;
        }
        {
            let mut state = self.inner.state.lock();
            for key in [
                ActionKey::Binding(lease.cache_id.clone()),
                ActionKey::Root(lease.parent_cache_id.clone()),
            ] {
                if let Some(action) = state.action_states.get_mut(&key) {
                    action.active_inference = action.active_inference.saturating_sub(1);
                }
            }
            if let Some(binding) = state.bindings.get_mut(&lease.cache_id) {
                binding.residency = if succeeded {
                    KvCacheResidency::Resident
                } else {
                    KvCacheResidency::Unknown
                };
            }
        }
        self.inner.notify.notify_waiters();
    }
}

#[cfg(test)]
mod tests;
