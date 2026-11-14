//! Internal state and action tasks of the KV-cache action runtime.
//!
//! Holds the binding/lineage index, the per-scope admission state, the action
//! queue with its `watch` signals, and the spawned task that performs one
//! provider action in per-lineage order.
use super::{KvCacheAdmission, KvCacheBinding, KvCacheResidency};
use crate::kv_cache::actions::{
    KvCacheAction, KvCacheActionProvider, KvCacheActionRequest, KvCacheActionStats,
};
use crate::kv_cache::config::KVCacheAffinityConfig;
use crate::kv_cache::types::KVCacheIdentity;
use parking_lot::Mutex;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{watch, Notify};
use tracing::warn;

/// Scope of one action: one binding, or one lineage root.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(super) enum ActionKey {
    /// One registered cache id.
    Binding(String),
    /// Every binding whose parent is this cache id.
    Root(String),
}

/// Registered binding state.
#[derive(Debug, Clone)]
pub(super) struct BindingState {
    pub(super) parent_cache_id: String,
    pub(super) model: String,
    pub(super) revision: u64,
    pub(super) residency: KvCacheResidency,
}

/// Receivers of the started/outcome signals of one enqueued action.
pub(super) struct ActionWaiters {
    pub(super) started: watch::Receiver<bool>,
    pub(super) outcome: watch::Receiver<Option<bool>>,
}

/// One queued or running action.
pub(super) struct PendingAction {
    pub(super) kind: KvCacheAction,
    pub(super) seq: u64,
    pub(super) started: watch::Receiver<bool>,
    pub(super) outcome: watch::Receiver<Option<bool>>,
    pub(super) provider_call_started: bool,
    pub(super) handle: tokio::task::JoinHandle<()>,
}

impl PendingAction {
    /// Whether the action sent its outcome already.
    pub(super) fn is_finished(&self) -> bool {
        self.outcome.borrow().is_some()
    }

    /// Whether the action's provider call has started.
    ///
    /// The flag is set under the state lock right before the started signal is
    /// sent; the watch carries the same fact to admission waiters.
    pub(super) fn provider_call_started(&self) -> bool {
        self.provider_call_started || *self.started.borrow()
    }
}

/// Per-scope admission state.
pub(super) struct ActionState {
    pub(super) active_inference: usize,
    pub(super) admission: KvCacheAdmission,
    pub(super) pending: Option<PendingAction>,
    pub(super) fail_open: bool,
}

impl Default for ActionState {
    fn default() -> Self {
        Self {
            active_inference: 0,
            admission: KvCacheAdmission::Open,
            pending: None,
            fail_open: false,
        }
    }
}

/// Bindings, the lineage index, and per-scope state.
#[derive(Default)]
pub(super) struct State {
    pub(super) bindings: HashMap<String, BindingState>,
    pub(super) root_index: HashMap<String, HashSet<String>>,
    pub(super) action_states: HashMap<ActionKey, ActionState>,
}

/// Binding values an action works on; the revision guards late results.
#[derive(Debug, Clone)]
pub(super) struct Snapshot {
    pub(super) cache_id: String,
    pub(super) parent_cache_id: String,
    pub(super) model: String,
    pub(super) revision: u64,
    pub(super) residency: KvCacheResidency,
}

impl Snapshot {
    /// Snapshots `cache_id`'s binding.
    pub(super) fn of(cache_id: &str, binding: &BindingState) -> Self {
        Self {
            cache_id: cache_id.to_owned(),
            parent_cache_id: binding.parent_cache_id.clone(),
            model: binding.model.clone(),
            revision: binding.revision,
            residency: binding.residency,
        }
    }
}

impl State {
    /// The state of `key`, created on first use.
    pub(super) fn action(&mut self, key: &ActionKey) -> &mut ActionState {
        self.action_states.entry(key.clone()).or_default()
    }

    /// Whether any scope related to `key` is blocked for inference.
    pub(super) fn related_scope_blocked(&self, key: &ActionKey) -> bool {
        match key {
            ActionKey::Root(parent) => self.root_index.get(parent).is_some_and(|indexed| {
                indexed.iter().any(|cache_id| {
                    self.action_states
                        .get(&ActionKey::Binding(cache_id.clone()))
                        .is_some_and(|state| state.admission == KvCacheAdmission::Blocked)
                })
            }),
            ActionKey::Binding(cache_id) => self.bindings.get(cache_id).is_some_and(|binding| {
                self.action_states
                    .get(&ActionKey::Root(binding.parent_cache_id.clone()))
                    .is_some_and(|state| state.admission == KvCacheAdmission::Blocked)
            }),
        }
    }

    /// Bindings `key` covers, ordered by cache id.
    pub(super) fn bindings_for(&self, key: &ActionKey) -> Vec<Snapshot> {
        let mut snapshots: Vec<Snapshot> = match key {
            ActionKey::Binding(cache_id) => self
                .bindings
                .get(cache_id)
                .map(|binding| vec![Snapshot::of(cache_id, binding)])
                .unwrap_or_default(),
            ActionKey::Root(parent) => self
                .root_index
                .get(parent)
                .map(|indexed| {
                    indexed
                        .iter()
                        .filter_map(|cache_id| {
                            self.bindings
                                .get(cache_id)
                                .map(|binding| Snapshot::of(cache_id, binding))
                        })
                        .collect()
                })
                .unwrap_or_default(),
        };
        snapshots.sort_by(|left, right| left.cache_id.cmp(&right.cache_id));
        snapshots
    }
}

/// Shared runtime state; actions hold it through an `Arc`.
pub(super) struct Inner {
    pub(super) provider: Arc<dyn KvCacheActionProvider>,
    pub(super) config: KVCacheAffinityConfig,
    pub(super) state: Mutex<State>,
    pub(super) stats: Mutex<KvCacheActionStats>,
    pub(super) notify: Notify,
    pub(super) lineages: Mutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>,
    pub(super) sequence: AtomicU64,
}

impl Inner {
    /// Registers or refreshes `cache_id` and returns its snapshot.
    pub(super) fn bind(
        &self,
        cache_id: &str,
        parent_cache_id: &str,
        model: &str,
        bump_revision: bool,
    ) -> KvCacheBinding {
        let mut state = self.state.lock();
        let previous = state.bindings.get(cache_id).map(|binding| {
            (
                binding.parent_cache_id.clone(),
                binding.revision,
                binding.residency,
            )
        });
        let mut dropped_root = None;
        if let Some((previous_root, _, _)) = &previous {
            if previous_root != parent_cache_id {
                let emptied = state
                    .root_index
                    .get_mut(previous_root)
                    .is_some_and(|indexed| {
                        indexed.remove(cache_id);
                        indexed.is_empty()
                    });
                if emptied {
                    state.root_index.remove(previous_root);
                    dropped_root = Some(previous_root.clone());
                }
            }
        }
        if let Some(root) = dropped_root {
            state.action_states.remove(&ActionKey::Root(root));
        }
        let (revision, residency) = match previous {
            Some((_, revision, residency)) => (
                if bump_revision {
                    revision.saturating_add(1)
                } else {
                    revision
                },
                residency,
            ),
            None => (0, KvCacheResidency::Unknown),
        };
        state.bindings.insert(
            cache_id.to_owned(),
            BindingState {
                parent_cache_id: parent_cache_id.to_owned(),
                model: model.to_owned(),
                revision,
                residency,
            },
        );
        state
            .root_index
            .entry(parent_cache_id.to_owned())
            .or_default()
            .insert(cache_id.to_owned());
        state
            .action_states
            .entry(ActionKey::Binding(cache_id.to_owned()))
            .or_default();
        state
            .action_states
            .entry(ActionKey::Root(parent_cache_id.to_owned()))
            .or_default();
        KvCacheBinding {
            cache_id: cache_id.to_owned(),
            parent_cache_id: parent_cache_id.to_owned(),
            model: model.to_owned(),
            revision,
            residency,
        }
    }

    /// Action keys a registered identity covers.
    pub(super) fn affected_keys(&self, cache_id: &str, parent_cache_id: &str) -> Vec<ActionKey> {
        let state = self.state.lock();
        keys_for_identity(&state, cache_id, parent_cache_id)
    }

    /// `keys` with every blocked lineage root of a child scope prepended.
    pub(super) fn prepare_keys(&self, keys: Vec<ActionKey>) -> Vec<ActionKey> {
        let state = self.state.lock();
        let mut roots: Vec<ActionKey> = Vec::new();
        for key in &keys {
            let ActionKey::Binding(cache_id) = key else {
                continue;
            };
            let Some(binding) = state.bindings.get(cache_id) else {
                continue;
            };
            let root_key = ActionKey::Root(binding.parent_cache_id.clone());
            let blocked = state
                .action_states
                .get(&root_key)
                .is_some_and(|action| action.admission == KvCacheAdmission::Blocked);
            if blocked && !roots.contains(&root_key) {
                roots.push(root_key);
            }
        }
        roots.extend(keys);
        roots
    }

    /// Bindings `key` covers at this instant.
    pub(super) fn snapshot_bindings(&self, key: &ActionKey) -> Vec<Snapshot> {
        self.state.lock().bindings_for(key)
    }

    /// The ordering lock of `key`'s lineage.
    pub(super) fn lineage_gate(&self, key: &ActionKey) -> Arc<tokio::sync::Mutex<()>> {
        let lineage = match key {
            ActionKey::Root(parent) => parent.clone(),
            ActionKey::Binding(cache_id) => self.state.lock().bindings.get(cache_id).map_or_else(
                || cache_id.clone(),
                |binding| binding.parent_cache_id.clone(),
            ),
        };
        let mut gates = self.lineages.lock();
        Arc::clone(
            gates
                .entry(lineage)
                .or_insert_with(|| Arc::new(tokio::sync::Mutex::new(()))),
        )
    }

    /// Forgets every binding of the identity (the whole lineage for a root).
    pub(super) fn forget(&self, cache_id: &str, parent_cache_id: &str) {
        {
            let mut state = self.state.lock();
            let binding_keys: Vec<String> = if cache_id == parent_cache_id {
                state
                    .root_index
                    .get(parent_cache_id)
                    .map(|indexed| indexed.iter().cloned().collect())
                    .unwrap_or_default()
            } else if state.bindings.contains_key(cache_id) {
                vec![cache_id.to_owned()]
            } else {
                Vec::new()
            };
            for binding_key in binding_keys {
                remove_binding(&mut state, &binding_key);
            }
            if cache_id == parent_cache_id {
                state
                    .action_states
                    .remove(&ActionKey::Root(parent_cache_id.to_owned()));
            }
        }
        self.lineages.lock().remove(parent_cache_id);
        self.notify.notify_waiters();
    }

    /// Fails `keys` open so inference is never blocked by cache management.
    pub(super) fn fail_open_keys(&self, keys: &[ActionKey]) {
        {
            let mut state = self.state.lock();
            for key in keys {
                let action = state.action(key);
                if action.admission != KvCacheAdmission::Terminal {
                    action.admission = KvCacheAdmission::Open;
                    action.fail_open = true;
                }
            }
        }
        self.notify.notify_waiters();
    }
}

/// The action keys a registered identity covers.
pub(super) fn keys_for_identity(
    state: &State,
    cache_id: &str,
    parent_cache_id: &str,
) -> Vec<ActionKey> {
    if cache_id == parent_cache_id {
        if state.root_index.contains_key(parent_cache_id) {
            vec![ActionKey::Root(parent_cache_id.to_owned())]
        } else {
            Vec::new()
        }
    } else if state.bindings.contains_key(cache_id) {
        vec![ActionKey::Binding(cache_id.to_owned())]
    } else {
        Vec::new()
    }
}

/// The scope whose state describes `identity`.
pub(super) fn primary_key(cache_id: &str, parent_cache_id: &str) -> ActionKey {
    if cache_id == parent_cache_id {
        ActionKey::Root(parent_cache_id.to_owned())
    } else {
        ActionKey::Binding(cache_id.to_owned())
    }
}

/// Maps an identity onto `(cache id, parent cache id)`.
///
/// A missing, blank, or whitespace-only parent falls back to the cache id, so a
/// host can register a root scope with `KVCacheIdentity::new` alone. Returns
/// `None` when the cache id itself is blank.
pub(super) fn identity_mapping(identity: &KVCacheIdentity) -> Option<(String, String)> {
    let cache_id = identity.session_id.trim();
    if cache_id.is_empty() {
        return None;
    }
    let parent_cache_id = identity
        .parent()
        .map(str::trim)
        .filter(|parent| !parent.is_empty())
        .unwrap_or(cache_id);
    Some((cache_id.to_owned(), parent_cache_id.to_owned()))
}

/// Removes one binding, pruning the lineage index and its states when empty.
pub(super) fn remove_binding(state: &mut State, cache_id: &str) {
    let Some(binding) = state.bindings.remove(cache_id) else {
        return;
    };
    let root = binding.parent_cache_id;
    let emptied = state.root_index.get_mut(&root).is_some_and(|indexed| {
        indexed.remove(cache_id);
        indexed.is_empty()
    });
    if emptied {
        state.root_index.remove(&root);
        state.action_states.remove(&ActionKey::Root(root));
    }
    state
        .action_states
        .remove(&ActionKey::Binding(cache_id.to_owned()));
}

/// Opens `key` (and, for a root key, every binding below it) for inference.
pub(super) fn open_scope(state: &mut State, key: &ActionKey) {
    let mut keys = vec![key.clone()];
    if let ActionKey::Root(parent) = key {
        let children: Vec<ActionKey> = state
            .root_index
            .get(parent)
            .map(|indexed| {
                indexed
                    .iter()
                    .map(|cache_id| ActionKey::Binding(cache_id.clone()))
                    .collect()
            })
            .unwrap_or_default();
        keys.extend(children);
    }
    for action_key in keys {
        if let Some(action) = state.action_states.get_mut(&action_key) {
            if action.admission != KvCacheAdmission::Terminal {
                action.admission = KvCacheAdmission::Open;
            }
        }
    }
}

/// Enqueues `kind` for `key`, unless an action of the same kind is still running.
pub(super) fn enqueue(
    inner: &Arc<Inner>,
    state: &mut State,
    key: &ActionKey,
    kind: KvCacheAction,
) -> Option<ActionWaiters> {
    let running = state
        .action_states
        .get(key)
        .and_then(|action| action.pending.as_ref())
        .is_some_and(|pending| pending.kind == kind && !pending.is_finished());
    if running {
        return None;
    }
    let seq = inner
        .sequence
        .fetch_add(1, Ordering::SeqCst)
        .saturating_add(1);
    let (started_sender, started) = watch::channel(false);
    let (outcome_sender, outcome) = watch::channel(None::<bool>);
    let handle = tokio::spawn(run_action(
        Arc::clone(inner),
        key.clone(),
        kind,
        seq,
        started_sender,
        outcome_sender,
    ));
    state.action(key).pending = Some(PendingAction {
        kind,
        seq,
        started: started.clone(),
        outcome: outcome.clone(),
        provider_call_started: false,
        handle,
    });
    Some(ActionWaiters { started, outcome })
}

/// Serialises and runs one action, then publishes its outcome.
pub(super) async fn run_action(
    inner: Arc<Inner>,
    key: ActionKey,
    kind: KvCacheAction,
    seq: u64,
    started: watch::Sender<bool>,
    outcome: watch::Sender<Option<bool>>,
) {
    let gate = inner.lineage_gate(&key);
    let guard = gate.lock().await;

    let mut succeeded = false;
    let mut idle = true;
    if kind != KvCacheAction::Prefetch {
        let budget = if kind == KvCacheAction::Evict {
            inner.config.effective_evict_timeout()
        } else {
            inner.config.effective_offload_prefetch_timeout()
        };
        idle = wait_for_idle(&inner, &key, budget).await;
    }

    let mut snapshots = Vec::new();
    if idle {
        snapshots = inner.snapshot_bindings(&key);
        if snapshots.is_empty() {
            warn!(
                action = kind.as_str(),
                "kv-cache action has no registered binding; failing open"
            );
        } else {
            let request = request_for(&key, &snapshots);
            mark_started(&inner, &key, kind, seq, &started);
            let budget = if kind == KvCacheAction::Evict {
                inner.config.effective_evict_timeout()
            } else {
                inner.config.effective_offload_prefetch_timeout()
            };
            match tokio::time::timeout(budget, inner.provider.perform(kind, &request)).await {
                Ok(Ok(result)) => succeeded = result,
                Ok(Err(error)) => warn!(
                    action = kind.as_str(),
                    cache_id = %request.cache_id,
                    error = %error,
                    "kv-cache provider action failed; failing open"
                ),
                Err(_) => warn!(
                    action = kind.as_str(),
                    cache_id = %request.cache_id,
                    "kv-cache provider action timed out; failing open"
                ),
            }
        }
    } else {
        warn!(
            action = kind.as_str(),
            "kv-cache action could not reach an idle scope; failing open"
        );
    }
    drop(guard);

    record_result(&inner, &key, kind, seq, &snapshots, succeeded);
    let _ = outcome.send(Some(succeeded));
    inner.notify.notify_waiters();
}

/// Marks the action as provider-call-started and opens a prefetched scope.
pub(super) fn mark_started(
    inner: &Arc<Inner>,
    key: &ActionKey,
    kind: KvCacheAction,
    seq: u64,
    started: &watch::Sender<bool>,
) {
    {
        let mut state = inner.state.lock();
        if let Some(action) = state.action_states.get_mut(key) {
            if action
                .pending
                .as_ref()
                .is_some_and(|pending| pending.seq == seq)
            {
                if let Some(pending) = action.pending.as_mut() {
                    pending.provider_call_started = true;
                }
            }
        }
        if kind == KvCacheAction::Prefetch {
            open_scope(&mut state, key);
        }
    }
    // The receiver is kept by the pending action; a dropped one is fine.
    let _ = started.send(true);
    inner.notify.notify_waiters();
}

/// Applies the action's result: residency, fail-open, counters, pending slot.
pub(super) fn record_result(
    inner: &Arc<Inner>,
    key: &ActionKey,
    kind: KvCacheAction,
    seq: u64,
    snapshots: &[Snapshot],
    succeeded: bool,
) {
    {
        let mut state = inner.state.lock();
        if succeeded {
            let target = match kind {
                KvCacheAction::Prefetch => KvCacheResidency::Resident,
                KvCacheAction::Offload => KvCacheResidency::Offloaded,
                KvCacheAction::Evict => KvCacheResidency::Unknown,
            };
            for snapshot in snapshots {
                if let Some(binding) = state.bindings.get_mut(&snapshot.cache_id) {
                    // A binding re-registered while the action ran keeps its
                    // newer state: the result describes the previous revision.
                    if binding.revision == snapshot.revision {
                        binding.residency = target;
                    }
                }
            }
        } else if let Some(action) = state.action_states.get_mut(key) {
            action.fail_open = true;
        }
        if let Some(action) = state.action_states.get_mut(key) {
            if action
                .pending
                .as_ref()
                .is_some_and(|pending| pending.seq == seq)
            {
                action.pending = None;
            }
        }
    }
    let mut stats = inner.stats.lock();
    if succeeded {
        stats.record_success(kind);
    } else {
        stats.record_failure();
    }
}

/// Waits until `key` has no active inference, bounded by `budget`.
pub(super) async fn wait_for_idle(inner: &Arc<Inner>, key: &ActionKey, budget: Duration) -> bool {
    let idle = async {
        loop {
            // Create the waiter before re-checking so a concurrent
            // `notify_waiters` cannot slip between the check and the await.
            let notified = inner.notify.notified();
            let count = {
                let state = inner.state.lock();
                state
                    .action_states
                    .get(key)
                    .map_or(0, |action| action.active_inference)
            };
            if count == 0 {
                return;
            }
            notified.await;
        }
    };
    tokio::time::timeout(budget, idle).await.is_ok()
}

/// Waits until the action's provider call starts or the action finishes.
pub(super) async fn wait_until_started(inner: &Arc<Inner>, waiters: ActionWaiters) -> bool {
    let ActionWaiters {
        mut started,
        mut outcome,
    } = waiters;
    let budget = inner.config.effective_offload_prefetch_timeout();
    let admission = tokio::time::timeout(budget, async {
        loop {
            let started_now = *started.borrow();
            if started_now {
                return true;
            }
            let finished = *outcome.borrow();
            if let Some(succeeded) = finished {
                return succeeded;
            }
            tokio::select! {
                changed = started.changed() => {
                    if changed.is_err() {
                        let last = *outcome.borrow();
                        return last.unwrap_or(false);
                    }
                }
                changed = outcome.changed() => {
                    if changed.is_err() {
                        let last = *outcome.borrow();
                        return last.unwrap_or(false);
                    }
                }
            }
        }
    })
    .await;
    if let Ok(value) = admission {
        value
    } else {
        warn!("kv-cache prefetch admission timed out");
        false
    }
}

/// Awaits the action's outcome.
pub(super) async fn wait_outcome(mut outcome: watch::Receiver<Option<bool>>) -> Option<bool> {
    loop {
        let current = *outcome.borrow();
        if current.is_some() {
            return current;
        }
        if outcome.changed().await.is_err() {
            return *outcome.borrow();
        }
    }
}

/// Builds the provider request of `key` from its first snapshot.
pub(super) fn request_for(key: &ActionKey, snapshots: &[Snapshot]) -> KvCacheActionRequest {
    let (cache_id, parent_cache_id) = match key {
        ActionKey::Root(parent) => (parent.clone(), parent.clone()),
        ActionKey::Binding(cache_id) => (
            cache_id.clone(),
            snapshots.first().map_or_else(
                || cache_id.clone(),
                |binding| binding.parent_cache_id.clone(),
            ),
        ),
    };
    let request = KvCacheActionRequest::new(cache_id, parent_cache_id);
    match snapshots
        .first()
        .map(|binding| binding.model.trim())
        .filter(|model| !model.is_empty())
    {
        Some(model) => request.with_model(model),
        None => request,
    }
}

/// Prepares one scope for inference: cancel a queued offload, else prefetch.
pub(super) async fn prepare_key(inner: &Arc<Inner>, key: &ActionKey) -> bool {
    let mut had_started_offload = false;
    let mut cancelled = false;
    {
        let mut state = inner.state.lock();
        let action = state.action(key);
        if action.admission == KvCacheAdmission::Terminal {
            return false;
        }
        let cancellable = action.pending.as_ref().is_some_and(|pending| {
            pending.kind == KvCacheAction::Offload && !pending.provider_call_started()
        });
        if cancellable {
            if let Some(pending) = action.pending.take() {
                pending.handle.abort();
            }
            action.admission = KvCacheAdmission::Open;
            cancelled = true;
        } else {
            had_started_offload = action
                .pending
                .as_ref()
                .is_some_and(|pending| pending.kind == KvCacheAction::Offload);
        }
    }
    if cancelled {
        inner.stats.lock().record_skipped();
        return true;
    }

    let waiters = {
        let mut state = inner.state.lock();
        if state.action(key).admission == KvCacheAdmission::Terminal {
            return false;
        }
        let pending_offload = state
            .action_states
            .get(key)
            .and_then(|action| action.pending.as_ref())
            .is_some_and(|pending| pending.kind == KvCacheAction::Offload);
        let scope_blocked = state.related_scope_blocked(key);
        let bindings = state.bindings_for(key);
        let needs_prefetch = had_started_offload
            || pending_offload
            || scope_blocked
            || bindings
                .iter()
                .any(|binding| binding.residency != KvCacheResidency::Resident);
        if !needs_prefetch {
            state.action(key).admission = KvCacheAdmission::Open;
            return true;
        }
        // An identical prefetch already in flight leaves admission open and
        // reports "not prepared" instead of queueing a second one.
        let Some(waiters) = enqueue(inner, &mut state, key, KvCacheAction::Prefetch) else {
            state.action(key).admission = KvCacheAdmission::Open;
            return false;
        };
        waiters
    };
    wait_until_started(inner, waiters).await
}
