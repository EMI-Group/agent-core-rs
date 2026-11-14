//! Unit tests for the KV-cache action runtime.
//!
//! Driven by a gated/delaying/failing provider so ordering, fail-open, the
//! revision guard, and the budgets are observable without sleeps.

use super::*;
use crate::kv_cache::actions::{
    KvCacheAction, KvCacheActionProvider, KvCacheActionRequest, KvCacheActionStats,
};
use crate::kv_cache::backend::InProcessKvCacheActionProvider;
use crate::kv_cache::config::KVCacheAffinityConfig;
use crate::kv_cache::types::KVCacheIdentity;
use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use parking_lot::Mutex;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Semaphore;

fn config() -> KVCacheAffinityConfig {
    KVCacheAffinityConfig::enabled()
}

fn child_identity() -> KVCacheIdentity {
    KVCacheIdentity::new("session-a", "hash-a", "gpt-test").with_parent("root-a")
}

fn root_identity() -> KVCacheIdentity {
    KVCacheIdentity::new("root-a", "hash-root", "gpt-test")
}

fn residency(
    runtime: &KvCacheActionRuntime,
    identity: &KVCacheIdentity,
) -> Option<KvCacheResidency> {
    runtime.binding(identity).map(|binding| binding.residency)
}

/// Polls a synchronous condition, yielding to the action tasks in between.
async fn wait_until(label: &str, condition: impl Fn() -> bool) {
    tokio::time::timeout(Duration::from_secs(5), async {
        while !condition() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap_or_else(|_| panic!("timed out waiting for {label}"));
}

/// Polls the runtime's counters, yielding to the action tasks in between.
async fn wait_for_stats(
    runtime: &KvCacheActionRuntime,
    label: &str,
    condition: impl Fn(&KvCacheActionStats) -> bool,
) {
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            if condition(&runtime.stats().await) {
                return;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap_or_else(|_| panic!("timed out waiting for {label}"));
}

/// Gives every spawned action a bounded chance to make progress.
async fn settle() {
    for _ in 0..200 {
        tokio::task::yield_now().await;
    }
}

struct GateState {
    gate: Semaphore,
    gated: Mutex<HashSet<KvCacheAction>>,
    calls: Mutex<Vec<(KvCacheAction, KvCacheActionRequest)>>,
    delays: Mutex<HashMap<KvCacheAction, Duration>>,
    errors: Mutex<HashSet<KvCacheAction>>,
}

/// Provider that can gate, delay, or fail one specific action.
struct GateProvider {
    state: Arc<GateState>,
}

impl GateProvider {
    fn shared() -> Arc<Self> {
        Arc::new(Self {
            state: Arc::new(GateState {
                gate: Semaphore::new(0),
                gated: Mutex::new(HashSet::new()),
                calls: Mutex::new(Vec::new()),
                delays: Mutex::new(HashMap::new()),
                errors: Mutex::new(HashSet::new()),
            }),
        })
    }

    /// Makes `action` wait for a permit before returning.
    fn gate(&self, action: KvCacheAction) {
        self.state.gated.lock().insert(action);
    }

    /// Releases one gated provider call.
    fn unblock(&self, action: KvCacheAction) {
        assert!(
            self.state.gated.lock().contains(&action),
            "{action} is not gated"
        );
        self.state.gate.add_permits(1);
    }

    fn delay(&self, action: KvCacheAction, delay: Duration) {
        self.state.delays.lock().insert(action, delay);
    }

    fn fail(&self, action: KvCacheAction) {
        self.state.errors.lock().insert(action);
    }

    fn calls(&self, action: KvCacheAction) -> usize {
        self.state
            .calls
            .lock()
            .iter()
            .filter(|(kind, _)| *kind == action)
            .count()
    }

    async fn invoke(&self, action: KvCacheAction, request: &KvCacheActionRequest) -> Result<bool> {
        self.state.calls.lock().push((action, request.clone()));
        if self.state.gated.lock().contains(&action) {
            if let Ok(permit) = self.state.gate.acquire().await {
                permit.forget();
            }
        }
        let delay = self.state.delays.lock().get(&action).copied();
        if let Some(delay) = delay {
            tokio::time::sleep(delay).await;
        }
        if self.state.errors.lock().contains(&action) {
            return Err(Error::transport(format!("{action} unavailable")));
        }
        Ok(true)
    }
}

#[async_trait]
impl KvCacheActionProvider for GateProvider {
    async fn prefetch(&self, request: &KvCacheActionRequest) -> Result<bool> {
        self.invoke(KvCacheAction::Prefetch, request).await
    }

    async fn offload(&self, request: &KvCacheActionRequest) -> Result<bool> {
        self.invoke(KvCacheAction::Offload, request).await
    }

    async fn evict(&self, request: &KvCacheActionRequest) -> Result<bool> {
        self.invoke(KvCacheAction::Evict, request).await
    }
}

#[tokio::test]
async fn prefetch_starts_without_waiting_for_completion() {
    let provider = GateProvider::shared();
    provider.gate(KvCacheAction::Prefetch);
    let runtime = KvCacheActionRuntime::new(provider.clone(), config());
    let identity = child_identity();

    let prepared = tokio::time::timeout(Duration::from_secs(1), runtime.prepare(&identity))
        .await
        .expect("prepare must return while the provider call is in flight");
    assert!(prepared, "a started prefetch opens the scope");
    wait_until("provider entered", || {
        provider.calls(KvCacheAction::Prefetch) == 1
    })
    .await;
    assert_eq!(
        runtime.stats().await.prefetch_succeeded,
        0,
        "the provider call is still gated"
    );

    provider.unblock(KvCacheAction::Prefetch);
    wait_for_stats(&runtime, "prefetch succeeded", |stats| {
        stats.prefetch_succeeded == 1
    })
    .await;
    assert_eq!(
        residency(&runtime, &identity),
        Some(KvCacheResidency::Resident)
    );
}

#[tokio::test]
async fn offload_waits_until_inference_finishes() {
    let provider = GateProvider::shared();
    let runtime = KvCacheActionRuntime::new(provider.clone(), config());
    let identity = child_identity();

    assert!(runtime.prepare(&identity).await);
    wait_for_stats(&runtime, "prefetch succeeded", |stats| {
        stats.prefetch_succeeded == 1
    })
    .await;

    let lease = runtime.begin_inference(&identity).await.expect("lease");
    assert_eq!(runtime.active_inference(&identity), 1);
    assert!(runtime.suspend(&identity).await);
    settle().await;
    assert_eq!(
        provider.calls(KvCacheAction::Offload),
        0,
        "an offload must wait for the active inference"
    );

    runtime.end_inference(&lease, true).await;
    wait_until("offload observed", || {
        provider.calls(KvCacheAction::Offload) == 1
    })
    .await;
}

#[tokio::test]
async fn suspend_skips_an_already_offloaded_scope() {
    let provider = GateProvider::shared();
    let runtime = KvCacheActionRuntime::new(provider.clone(), config());
    let identity = child_identity();

    assert!(runtime.prepare(&identity).await);
    wait_for_stats(&runtime, "prefetch succeeded", |stats| {
        stats.prefetch_succeeded == 1
    })
    .await;
    assert!(runtime.suspend(&identity).await);
    wait_for_stats(&runtime, "offload succeeded", |stats| {
        stats.offload_succeeded == 1
    })
    .await;
    assert_eq!(
        residency(&runtime, &identity),
        Some(KvCacheResidency::Offloaded)
    );

    assert!(!runtime.suspend(&identity).await);
    assert_eq!(provider.calls(KvCacheAction::Offload), 1);
    assert_eq!(runtime.stats().await.skipped, 1);
}

#[tokio::test]
async fn prepare_cancels_a_queued_offload() {
    let provider = GateProvider::shared();
    let runtime = KvCacheActionRuntime::new(provider.clone(), config());
    let identity = child_identity();

    assert!(runtime.prepare(&identity).await);
    wait_for_stats(&runtime, "prefetch succeeded", |stats| {
        stats.prefetch_succeeded == 1
    })
    .await;
    let lease = runtime.begin_inference(&identity).await.expect("lease");
    assert!(runtime.suspend(&identity).await);
    // The offload cannot start its provider call while the inference runs.
    settle().await;

    assert!(runtime.prepare(&identity).await);
    assert_eq!(runtime.stats().await.skipped, 1);
    runtime.end_inference(&lease, true).await;
    settle().await;
    assert_eq!(
        provider.calls(KvCacheAction::Offload),
        0,
        "the cancelled offload must never reach the provider"
    );
    assert_eq!(runtime.admission(&identity), Some(KvCacheAdmission::Open));
}

#[tokio::test]
async fn a_failing_provider_fails_open() {
    let provider = GateProvider::shared();
    provider.fail(KvCacheAction::Offload);
    let runtime = KvCacheActionRuntime::new(provider.clone(), config());
    let identity = child_identity();

    assert!(runtime.prepare(&identity).await);
    wait_for_stats(&runtime, "prefetch succeeded", |stats| {
        stats.prefetch_succeeded == 1
    })
    .await;
    assert!(runtime.suspend(&identity).await);
    wait_for_stats(&runtime, "offload failure recorded", |stats| {
        stats.failures == 1
    })
    .await;
    wait_until("fail open", || runtime.fail_open(&identity)).await;

    assert!(
        !runtime.suspend(&identity).await,
        "a failed scope is skipped"
    );
    assert_eq!(runtime.stats().await.skipped, 1);
    assert_eq!(runtime.admission(&identity), Some(KvCacheAdmission::Open));

    let lease = runtime
        .begin_inference(&identity)
        .await
        .expect("fail-open lease");
    runtime.end_inference(&lease, false).await;
    assert!(lease.is_released());
    assert_eq!(
        residency(&runtime, &identity),
        Some(KvCacheResidency::Unknown)
    );
}

#[tokio::test]
async fn a_re_registered_binding_keeps_its_residency() {
    let provider = GateProvider::shared();
    provider.gate(KvCacheAction::Offload);
    let runtime = KvCacheActionRuntime::new(provider.clone(), config());
    let identity = child_identity();

    assert!(runtime.prepare(&identity).await);
    wait_for_stats(&runtime, "prefetch succeeded", |stats| {
        stats.prefetch_succeeded == 1
    })
    .await;
    assert_eq!(
        residency(&runtime, &identity),
        Some(KvCacheResidency::Resident)
    );

    assert!(runtime.suspend(&identity).await);
    wait_until("offload entered", || {
        provider.calls(KvCacheAction::Offload) == 1
    })
    .await;
    let binding = runtime.register_binding(&identity).await.expect("binding");
    assert_eq!(binding.revision, 1);

    provider.unblock(KvCacheAction::Offload);
    wait_for_stats(&runtime, "offload finished", |stats| {
        stats.offload_succeeded == 1
    })
    .await;
    assert_eq!(
        residency(&runtime, &identity),
        Some(KvCacheResidency::Resident),
        "a result from the previous revision must not apply"
    );
}

#[tokio::test]
async fn an_action_timeout_is_a_failure_not_a_hang() {
    let provider = GateProvider::shared();
    provider.delay(KvCacheAction::Prefetch, Duration::from_secs(5));
    let runtime = KvCacheActionRuntime::new(
        provider.clone(),
        KVCacheAffinityConfig::enabled().with_action_timeout(Duration::from_millis(5)),
    );
    let identity = child_identity();

    // `prepare` returns as soon as the provider call starts; the call itself
    // is bounded by the configured action budget.
    let prepared = tokio::time::timeout(Duration::from_secs(1), runtime.prepare(&identity))
        .await
        .expect("prepare must not wait for the provider");
    assert!(prepared);
    wait_for_stats(&runtime, "prefetch timeout recorded", |stats| {
        stats.failures == 1
    })
    .await;
    assert_eq!(runtime.stats().await.prefetch_succeeded, 0);
    assert_eq!(
        residency(&runtime, &identity),
        Some(KvCacheResidency::Unknown)
    );
}

#[tokio::test]
async fn in_process_provider_sees_the_action_sequence() {
    let provider = Arc::new(InProcessKvCacheActionProvider::new());
    let mut receiver = provider.subscribe();
    let runtime = KvCacheActionRuntime::new(provider.clone(), config());
    let identity = child_identity();

    assert!(runtime.prepare(&identity).await);
    assert!(runtime.suspend(&identity).await);
    assert!(runtime.release(&identity).await);

    assert_eq!(
        provider.actions(),
        [
            KvCacheAction::Prefetch,
            KvCacheAction::Offload,
            KvCacheAction::Evict,
        ]
    );
    for record in provider.history() {
        assert_eq!(record.request.cache_id, "session-a");
        assert_eq!(record.request.parent_cache_id, "root-a");
        assert_eq!(record.request.target, "session");
        assert_eq!(record.request.model.as_deref(), Some("gpt-test"));
    }

    let mut broadcast = Vec::new();
    for _ in 0..3 {
        let record = tokio::time::timeout(Duration::from_secs(1), receiver.recv())
            .await
            .expect("record within the budget")
            .expect("record");
        broadcast.push(record.action);
    }
    assert_eq!(
        broadcast,
        [
            KvCacheAction::Prefetch,
            KvCacheAction::Offload,
            KvCacheAction::Evict,
        ]
    );
    assert!(
        runtime.binding(&identity).is_none(),
        "release forgets the binding"
    );
}

#[tokio::test]
async fn root_offload_cascades_over_the_lineage() {
    let provider = Arc::new(InProcessKvCacheActionProvider::new());
    let runtime = KvCacheActionRuntime::new(provider.clone(), config());
    let root = root_identity();
    let child = child_identity();
    runtime.register_binding(&root).await.expect("root binding");
    runtime
        .register_binding(&child)
        .await
        .expect("child binding");

    assert!(runtime.prepare(&root).await);
    wait_for_stats(&runtime, "root prefetch succeeded", |stats| {
        stats.prefetch_succeeded == 1
    })
    .await;
    assert_eq!(residency(&runtime, &root), Some(KvCacheResidency::Resident));
    assert_eq!(
        residency(&runtime, &child),
        Some(KvCacheResidency::Resident)
    );

    assert!(runtime.suspend(&root).await);
    wait_for_stats(&runtime, "root offload succeeded", |stats| {
        stats.offload_succeeded == 1
    })
    .await;
    assert_eq!(
        residency(&runtime, &root),
        Some(KvCacheResidency::Offloaded)
    );
    assert_eq!(
        residency(&runtime, &child),
        Some(KvCacheResidency::Offloaded)
    );

    let offload = provider
        .history()
        .into_iter()
        .find(|record| record.action == KvCacheAction::Offload)
        .expect("offload record");
    assert_eq!(offload.request.cache_id, "root-a");
    assert_eq!(offload.request.parent_cache_id, "root-a");
}

#[tokio::test]
async fn ending_a_lease_twice_is_a_no_op() {
    let provider = Arc::new(InProcessKvCacheActionProvider::new());
    let runtime = KvCacheActionRuntime::new(provider.clone(), config());
    let identity = child_identity();

    let lease = runtime.begin_inference(&identity).await.expect("lease");
    assert!(!lease.is_released());
    assert_eq!(runtime.active_inference(&identity), 1);

    runtime.end_inference(&lease, true).await;
    assert!(lease.is_released());
    assert_eq!(runtime.active_inference(&identity), 0);
    runtime.end_inference(&lease, true).await;
    assert_eq!(runtime.active_inference(&identity), 0);
    assert_eq!(
        residency(&runtime, &identity),
        Some(KvCacheResidency::Resident)
    );
    assert_eq!(runtime.stats().await, KvCacheActionStats::default());
    assert!(provider.is_empty());
}

#[tokio::test]
async fn a_disabled_config_makes_every_lifecycle_call_inert() {
    let provider = Arc::new(InProcessKvCacheActionProvider::new());
    let runtime = KvCacheActionRuntime::new(provider.clone(), KVCacheAffinityConfig::default());
    let identity = child_identity();

    assert!(!runtime.prepare(&identity).await);
    assert!(!runtime.suspend(&identity).await);
    assert!(!runtime.release(&identity).await);
    assert!(runtime.begin_inference(&identity).await.is_none());
    settle().await;
    assert!(provider.is_empty());
    assert!(runtime.binding(&identity).is_none());
    assert_eq!(runtime.stats().await, KvCacheActionStats::default());
}

#[tokio::test]
async fn identity_mapping_trims_and_defaults_the_lineage() {
    let provider = Arc::new(InProcessKvCacheActionProvider::new());
    let runtime = KvCacheActionRuntime::new(provider, config());

    assert!(
        !runtime
            .prepare(&KVCacheIdentity::new("  ", "hash", "gpt-test"))
            .await,
        "a blank identity names no cache scope"
    );
    let binding = runtime
        .register_binding(&KVCacheIdentity::new(" session-b ", "hash", "gpt-test"))
        .await
        .expect("binding");
    assert_eq!(binding.cache_id, "session-b");
    assert_eq!(binding.parent_cache_id, "session-b");
    assert_eq!(binding.residency, KvCacheResidency::Unknown);
    assert_eq!(binding.revision, 0);

    let child = runtime
        .register_binding(
            &KVCacheIdentity::new("session-c", "hash", "gpt-test").with_parent(" line "),
        )
        .await
        .expect("binding");
    assert_eq!(child.parent_cache_id, "line");
    // Re-registering moves the binding to a new lineage and bumps revision.
    let moved = runtime
        .register_binding(
            &KVCacheIdentity::new("session-c", "hash", "gpt-test").with_parent("root-a"),
        )
        .await
        .expect("binding");
    assert_eq!(moved.parent_cache_id, "root-a");
    assert_eq!(moved.revision, 1);
}
