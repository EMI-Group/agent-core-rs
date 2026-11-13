//! Ordered callback chains with per-entry filters and explicit outcomes.
//!
//! Ports `openjiuwen/core/runner/callback/chain.py` and the `ChainAction`/
//! `ChainContext`/`ChainResult` models from `enums.py`/`models.py`, keeping the
//! part the kernel needs and dropping the operational machinery:
//!
//! * An entry carries a `name`, a `namespace`, a `priority`, an `enabled` toggle,
//!   an optional filter list, and a fatal-vs-isolated error policy.
//! * Dispatch order is priority descending, insertion order as the tiebreak —
//!   the same order [`super::CallbackFramework`] uses, so a chain and a framework
//!   registration of the same handlers fire alike.
//! * [`CallbackVerdict::Abort`] ends the chain and surfaces as the `trigger`
//!   error (Python's `AbortError`); [`CallbackVerdict::Stop`] ends the chain
//!   successfully (`report.aborted = true`, Django's `BREAK`); a plain `Err` from
//!   an entry is isolated into [`CallbackReport::failures`] and dispatch
//!   continues, unless the entry is [`fatal`](ChainEntry::fatal), in which case it
//!   becomes the `trigger` error.
//! * Filters run against a mutable working copy of the context, so a
//!   [`super::ParamModifyFilter`] mutates what every later entry sees.
//! * Retries, rollback handlers, and per-entry timeouts from `chain.py` are not
//!   ported (see the crate-level `## Deferred Python Behavior`).
//!
//! A chain is itself a [`CallbackHandler`], so it nests: add `Arc::new(chain)` to
//! another chain or register it on a [`super::CallbackFramework`].
#![allow(clippy::module_name_repetitions)]
#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::unnecessary_wraps
    )
)]

use parking_lot::RwLock;
use serde::Serialize;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use openjiuwen_common::error::{Error, Result};

use super::context::{CallbackContext, CallbackFailure, CallbackReport, CallbackVerdict};
use super::event::CallbackEvent;
use super::filters::{record_failure, record_success, run_filters, EventFilter, FilterRunOutcome};
use super::handler::{CallbackHandler, CallbackResult};
use crate::callback::LOG_TARGET;

/// The handle [`CallbackChain::add`] returns and [`CallbackChain::remove_id`] takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ChainEntryId(u64);

impl ChainEntryId {
    /// The raw numeric value.
    #[must_use]
    pub const fn as_u64(self) -> u64 {
        self.0
    }
}

impl fmt::Display for ChainEntryId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "chain-entry-{}", self.0)
    }
}

/// One handler in a chain, with its dispatch configuration.
#[derive(Clone)]
pub struct ChainEntry {
    name: String,
    namespace: String,
    priority: i32,
    enabled: bool,
    fatal: bool,
    filters: Vec<Arc<dyn EventFilter>>,
    handler: Arc<dyn CallbackHandler>,
}

impl ChainEntry {
    /// An enabled, isolated entry at priority `0` in namespace `"default"`.
    #[must_use]
    pub fn new(name: impl Into<String>, handler: Arc<dyn CallbackHandler>) -> Self {
        Self {
            name: name.into(),
            namespace: "default".to_owned(),
            priority: 0,
            enabled: true,
            fatal: false,
            filters: Vec::new(),
            handler,
        }
    }

    /// An entry named after its handler.
    #[must_use]
    pub fn from_handler(handler: Arc<dyn CallbackHandler>) -> Self {
        Self::new(handler.name().to_owned(), handler)
    }

    /// Sets the dispatch priority; higher runs first.
    #[must_use]
    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }

    /// Sets the namespace reported to filters and in [`ChainEntryInfo`].
    #[must_use]
    pub fn in_namespace(mut self, namespace: impl Into<String>) -> Self {
        self.namespace = namespace.into();
        self
    }

    /// Adds a filter that runs before this entry's handler.
    #[must_use]
    pub fn when(mut self, filter: impl EventFilter + 'static) -> Self {
        self.filters.push(Arc::new(filter));
        self
    }

    /// Adds an already shared filter.
    #[must_use]
    pub fn with_filter(mut self, filter: Arc<dyn EventFilter>) -> Self {
        self.filters.push(filter);
        self
    }

    /// A failing handler becomes the chain's error instead of a recorded failure.
    #[must_use]
    pub fn fatal(mut self) -> Self {
        self.fatal = true;
        self
    }

    /// Adds the entry disabled (it can be enabled later).
    #[must_use]
    pub fn disabled(mut self) -> Self {
        self.enabled = false;
        self
    }

    /// The entry's name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The entry's handler.
    #[must_use]
    pub fn handler(&self) -> Arc<dyn CallbackHandler> {
        Arc::clone(&self.handler)
    }

    /// The entry's filters.
    #[must_use]
    pub fn filters(&self) -> &[Arc<dyn EventFilter>] {
        &self.filters
    }

    /// Whether a handler error aborts the chain.
    #[must_use]
    pub const fn is_fatal(&self) -> bool {
        self.fatal
    }
}

impl fmt::Debug for ChainEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ChainEntry")
            .field("name", &self.name)
            .field("namespace", &self.namespace)
            .field("priority", &self.priority)
            .field("enabled", &self.enabled)
            .field("fatal", &self.fatal)
            .field(
                "filters",
                &self
                    .filters
                    .iter()
                    .map(EventFilter::name)
                    .collect::<Vec<_>>(),
            )
            .field("handler", &self.handler.name())
            .finish_non_exhaustive()
    }
}

/// A read-only snapshot of one entry's configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChainEntryInfo {
    /// The removal handle.
    pub id: ChainEntryId,
    /// The entry's name.
    pub name: String,
    /// The entry's namespace.
    pub namespace: String,
    /// Dispatch priority; higher runs first.
    pub priority: i32,
    /// Whether dispatch currently runs the entry.
    pub enabled: bool,
    /// Whether a handler error aborts the chain.
    pub fatal: bool,
    /// The guarded handler's name.
    pub handler: String,
    /// The filters' names, in evaluation order.
    pub filters: Vec<String>,
}

/// What happened to one entry during a chain run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EntryOutcome {
    /// The handler ran and returned `Continue`.
    Ran,
    /// The handler ran and returned `Stop`, ending the chain.
    Stopped,
    /// A filter skipped the entry, so the handler never ran.
    Skipped {
        /// The skipping filter's name.
        filter: String,
        /// The filter's reason.
        reason: String,
    },
    /// The handler failed and the failure was recorded; dispatch continued.
    Failed,
}

/// One entry's contribution to a [`ChainReport`], in dispatch order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EntryRecord {
    /// The entry's name.
    pub name: String,
    /// The entry's namespace.
    pub namespace: String,
    /// What happened to it.
    pub outcome: EntryOutcome,
}

/// The detailed result of one chain run.
///
/// [`CallbackChain::trigger`] returns the inner [`CallbackReport`] for symmetry
/// with [`super::CallbackFramework::trigger`]; [`CallbackChain::trigger_detailed`]
/// returns this, which adds the per-entry outcomes.
#[derive(Debug)]
pub struct ChainReport {
    /// The framework-shaped report (`fired`, `failures`, `aborted`).
    pub report: CallbackReport,
    /// One record per entry that was considered, in dispatch order.
    pub entries: Vec<EntryRecord>,
}

impl ChainReport {
    /// The entries a filter skipped.
    #[must_use]
    pub fn skipped(&self) -> Vec<&EntryRecord> {
        self.entries
            .iter()
            .filter(|record| matches!(record.outcome, EntryOutcome::Skipped { .. }))
            .collect()
    }

    /// The entries whose handler ran.
    #[must_use]
    pub fn ran(&self) -> Vec<&EntryRecord> {
        self.entries
            .iter()
            .filter(|record| {
                matches!(
                    record.outcome,
                    EntryOutcome::Ran | EntryOutcome::Stopped | EntryOutcome::Failed
                )
            })
            .collect()
    }
}

#[derive(Clone)]
struct StoredEntry {
    id: ChainEntryId,
    /// Negative values are prepended entries, non-negative ones appended.
    sequence: i64,
    entry: ChainEntry,
}

/// A named, ordered group of handlers.
pub struct CallbackChain {
    name: String,
    entries: RwLock<Vec<StoredEntry>>,
    next_id: AtomicU64,
    append_sequence: AtomicU64,
    prepend_sequence: AtomicU64,
}

impl CallbackChain {
    /// An empty chain called `name`.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            entries: RwLock::new(Vec::new()),
            next_id: AtomicU64::new(0),
            append_sequence: AtomicU64::new(0),
            prepend_sequence: AtomicU64::new(0),
        }
    }

    /// The chain's name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Appends `entry`; returns its handle.
    pub fn add(&self, entry: ChainEntry) -> ChainEntryId {
        self.insert(entry, false)
    }

    /// Appends a handler named after itself.
    pub fn add_handler(&self, handler: Arc<dyn CallbackHandler>) -> ChainEntryId {
        self.add(ChainEntry::from_handler(handler))
    }

    /// Appends a handler under `name`.
    pub fn add_named(
        &self,
        name: impl Into<String>,
        handler: Arc<dyn CallbackHandler>,
    ) -> ChainEntryId {
        self.add(ChainEntry::new(name, handler))
    }

    /// Inserts `entry` ahead of every existing entry of equal or lower priority.
    pub fn prepend(&self, entry: ChainEntry) -> ChainEntryId {
        self.insert(entry, true)
    }

    fn insert(&self, entry: ChainEntry, front: bool) -> ChainEntryId {
        let id = ChainEntryId(self.next_id.fetch_add(1, Ordering::Relaxed));
        // Appended entries take increasing sequences, prepended ones decreasing
        // (negative) sequences, so at equal priority a prepend always dispatches
        // first while both keep insertion order within their own block.
        let sequence = if front {
            let step = self.prepend_sequence.fetch_add(1, Ordering::Relaxed);
            i64::try_from(step)
                .unwrap_or(i64::MAX)
                .wrapping_neg()
                .saturating_sub(1)
        } else {
            i64::try_from(self.append_sequence.fetch_add(1, Ordering::Relaxed)).unwrap_or(i64::MAX)
        };
        self.entries.write().push(StoredEntry {
            id,
            sequence,
            entry,
        });
        id
    }

    /// Removes every entry called `name`; returns how many were removed.
    pub fn remove(&self, name: &str) -> usize {
        let mut entries = self.entries.write();
        let before = entries.len();
        entries.retain(|stored| stored.entry.name != name);
        before - entries.len()
    }

    /// Removes one entry by handle; returns whether it existed.
    pub fn remove_id(&self, id: ChainEntryId) -> bool {
        let mut entries = self.entries.write();
        let before = entries.len();
        entries.retain(|stored| stored.id != id);
        entries.len() != before
    }

    /// Removes every entry.
    pub fn clear(&self) {
        self.entries.write().clear();
    }

    /// How many entries the chain holds (enabled or not).
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.read().len()
    }

    /// Whether the chain holds no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Every entry's name in dispatch order.
    #[must_use]
    pub fn names(&self) -> Vec<String> {
        self.snapshot()
            .into_iter()
            .map(|stored| stored.entry.name)
            .collect()
    }

    /// A configuration snapshot in dispatch order.
    #[must_use]
    pub fn entries(&self) -> Vec<ChainEntryInfo> {
        self.snapshot()
            .into_iter()
            .map(|stored| ChainEntryInfo {
                id: stored.id,
                name: stored.entry.name,
                namespace: stored.entry.namespace,
                priority: stored.entry.priority,
                enabled: stored.entry.enabled,
                fatal: stored.entry.fatal,
                handler: stored.entry.handler.name().to_owned(),
                filters: stored
                    .entry
                    .filters
                    .iter()
                    .map(|filter| filter.name().to_owned())
                    .collect(),
            })
            .collect()
    }

    /// Enables or disables every entry called `name`; returns how many changed.
    pub fn set_enabled(&self, name: &str, enabled: bool) -> usize {
        let mut changed = 0;
        for stored in self.entries.write().iter_mut() {
            if stored.entry.name == name && stored.entry.enabled != enabled {
                stored.entry.enabled = enabled;
                changed += 1;
            }
        }
        changed
    }

    /// Enables or disables one entry; returns whether the handle exists.
    pub fn set_enabled_id(&self, id: ChainEntryId, enabled: bool) -> bool {
        for stored in self.entries.write().iter_mut() {
            if stored.id == id {
                stored.entry.enabled = enabled;
                return true;
            }
        }
        false
    }

    /// Dispatches `event` to the entries.
    ///
    /// The chain works on a clone of `ctx`, so filter mutations stay inside the
    /// chain; see [`Self::trigger_mut`] to observe them.
    ///
    /// # Errors
    ///
    /// Returns the error of the entry that aborted dispatch
    /// ([`CallbackVerdict::Abort`]), of a rejected filter, or of a fatal entry.
    pub async fn trigger(
        &self,
        event: CallbackEvent,
        ctx: &CallbackContext,
    ) -> Result<CallbackReport> {
        let mut working = ctx.clone();
        self.trigger_detailed(event, &mut working)
            .await
            .map(|detailed| detailed.report)
    }

    /// Dispatches `event` in place, so payload mutations survive the call.
    ///
    /// # Errors
    ///
    /// Same as [`Self::trigger`].
    pub async fn trigger_mut(
        &self,
        event: CallbackEvent,
        ctx: &mut CallbackContext,
    ) -> Result<CallbackReport> {
        self.trigger_detailed(event, ctx)
            .await
            .map(|detailed| detailed.report)
    }

    /// Dispatches `event` with `payload` as the context input.
    ///
    /// # Errors
    ///
    /// Returns `Error::Validation` when `payload` cannot be serialized, plus the
    /// errors of [`Self::trigger`].
    pub async fn trigger_payload<T>(
        &self,
        event: CallbackEvent,
        payload: &T,
    ) -> Result<CallbackReport>
    where
        T: Serialize,
    {
        let input = serde_json::to_value(payload)?;
        let ctx = CallbackContext::new(event.clone()).with_input(input);
        self.trigger(event, &ctx).await
    }

    /// Dispatches `event` and reports what happened to every entry.
    ///
    /// # Errors
    ///
    /// Same as [`Self::trigger`].
    pub async fn trigger_detailed(
        &self,
        event: CallbackEvent,
        ctx: &mut CallbackContext,
    ) -> Result<ChainReport> {
        let snapshot = self.snapshot();
        ctx.event = event.clone();
        let mut report = CallbackReport::new(event.clone());
        let mut entries = Vec::with_capacity(snapshot.len());
        for stored in snapshot {
            let record = self
                .run_entry(stored.id, &stored.entry, ctx, &mut report)
                .await?;
            let stopped = record
                .as_ref()
                .is_some_and(|record| record.outcome == EntryOutcome::Stopped);
            if let Some(record) = record {
                entries.push(record);
            }
            if stopped {
                break;
            }
        }
        Ok(ChainReport { report, entries })
    }

    /// Runs one entry, recording its outcome and updating `report`.
    ///
    /// `Ok(None)` means the entry was disabled and never considered; `Err` means
    /// dispatch aborts with that error (a rejected filter, an abort verdict, or a
    /// fatal entry failure).
    async fn run_entry(
        &self,
        id: ChainEntryId,
        entry: &ChainEntry,
        ctx: &mut CallbackContext,
        report: &mut CallbackReport,
    ) -> Result<Option<EntryRecord>> {
        let record = |outcome| {
            Some(EntryRecord {
                name: entry.name.clone(),
                namespace: entry.namespace.clone(),
                outcome,
            })
        };
        if !entry.enabled {
            tracing::debug!(
                target: LOG_TARGET,
                chain = %self.name,
                entry = %entry.name,
                "chain entry disabled; skipping"
            );
            return Ok(None);
        }

        match run_filters(&entry.filters, &entry.name, &entry.namespace, ctx)? {
            FilterRunOutcome::Allow => {}
            FilterRunOutcome::Skip { filter, reason } => {
                tracing::debug!(
                    target: LOG_TARGET,
                    chain = %self.name,
                    entry = %entry.name,
                    filter = %filter,
                    reason = %reason,
                    "chain entry skipped by a filter"
                );
                return Ok(record(EntryOutcome::Skipped { filter, reason }));
            }
            FilterRunOutcome::Reject(error) => {
                report.aborted = true;
                return Err(self.entry_context(entry, error));
            }
        }

        let verdict = entry.handler.handle(ctx).await;
        match verdict {
            Ok(CallbackVerdict::Continue) => {
                record_success(&entry.filters, ctx, &entry.name);
                report.fired += 1;
                Ok(record(EntryOutcome::Ran))
            }
            Ok(CallbackVerdict::Stop) => {
                record_success(&entry.filters, ctx, &entry.name);
                report.fired += 1;
                report.aborted = true;
                Ok(record(EntryOutcome::Stopped))
            }
            Ok(CallbackVerdict::Abort(error)) => {
                record_success(&entry.filters, ctx, &entry.name);
                report.aborted = true;
                tracing::debug!(
                    target: LOG_TARGET,
                    chain = %self.name,
                    entry = %entry.name,
                    error = %error,
                    "chain entry aborted dispatch"
                );
                Err(self.entry_context(entry, error))
            }
            Err(error) => {
                record_failure(&entry.filters, ctx, &entry.name);
                tracing::warn!(
                    target: LOG_TARGET,
                    chain = %self.name,
                    entry = %entry.name,
                    error = %error,
                    fatal = entry.fatal,
                    "chain entry failed"
                );
                if entry.fatal {
                    report.aborted = true;
                    return Err(self.entry_context(entry, error));
                }
                report.failures.push(CallbackFailure {
                    // Chain entries carry their own handles; the framework-shaped
                    // failure record reuses the entry id.
                    id: framework_id(id),
                    handler: entry.name.clone(),
                    error,
                });
                Ok(record(EntryOutcome::Failed))
            }
        }
    }

    /// Names the chain and entry an error came from, preserving its kind.
    fn entry_context(&self, entry: &ChainEntry, error: Error) -> Error {
        error.with_context(format!(
            "callback chain '{}' entry '{}'",
            self.name, entry.name
        ))
    }

    /// Dispatch order: priority descending, insertion order as the tiebreak.
    fn snapshot(&self) -> Vec<StoredEntry> {
        let mut snapshot = self.entries.read().clone();
        snapshot.sort_by(|left, right| {
            right
                .entry
                .priority
                .cmp(&left.entry.priority)
                .then(left.sequence.cmp(&right.sequence))
        });
        snapshot
    }
}

impl fmt::Debug for CallbackChain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CallbackChain")
            .field("name", &self.name)
            .field("names", &self.names())
            .finish_non_exhaustive()
    }
}

/// A chain is a handler, so it nests inside another chain or the framework.
///
/// `Continue` when the chain completed, `Stop` when its own dispatch was stopped,
/// and `Abort(error)` when it aborted or a fatal entry failed.
#[async_trait::async_trait]
impl CallbackHandler for CallbackChain {
    fn name(&self) -> &str {
        &self.name
    }

    async fn handle(&self, ctx: &CallbackContext) -> CallbackResult {
        match self.trigger(ctx.event.clone(), ctx).await {
            Ok(report) if report.aborted => Ok(CallbackVerdict::Stop),
            Ok(_) => Ok(CallbackVerdict::Continue),
            Err(error) => Ok(CallbackVerdict::Abort(error)),
        }
    }
}

/// An `Arc<CallbackChain>` is a handler too, so a shared nested chain can be
/// registered or added without re-wrapping.
#[async_trait::async_trait]
impl CallbackHandler for Arc<CallbackChain> {
    fn name(&self) -> &str {
        (**self).name()
    }

    async fn handle(&self, ctx: &CallbackContext) -> CallbackResult {
        (**self).handle(ctx).await
    }
}

/// [`CallbackFailure`] carries a [`super::CallbackId`]; chain entries are not
/// framework registrations, so their ids are carried through the same newtype.
fn framework_id(id: ChainEntryId) -> super::CallbackId {
    super::CallbackId::from_raw(id.as_u64())
}

impl Default for CallbackChain {
    fn default() -> Self {
        Self::new("chain")
    }
}

/// Chains are shared as `Arc<CallbackChain>`, which coerces to
/// `Arc<dyn CallbackHandler>` because the inner type is a handler.
#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    use super::super::handler::SyncHandler;

    fn recorder(
        order: &Arc<parking_lot::Mutex<Vec<String>>>,
        name: &'static str,
    ) -> Arc<dyn CallbackHandler> {
        let order = Arc::clone(order);
        Arc::new(SyncHandler::new(move |_: &CallbackContext| {
            order.lock().push(name.to_owned());
            Ok(CallbackVerdict::Continue)
        }))
    }

    #[tokio::test]
    async fn an_empty_chain_is_a_clean_no_op() {
        let chain = CallbackChain::new("empty");
        assert!(chain.is_empty());
        assert_eq!(chain.len(), 0);
        assert_eq!(chain.names().len(), 0);
        let report = chain
            .trigger(
                CallbackEvent::AgentStarted,
                &CallbackContext::new(CallbackEvent::AgentStarted),
            )
            .await
            .unwrap();
        assert_eq!(report.fired, 0);
        assert!(report.is_clean());
        assert_eq!(chain.name(), "empty");
        assert!(format!("{chain:?}").contains("empty"));
        assert_eq!(CallbackChain::default().name(), "chain");
    }

    #[tokio::test]
    async fn duplicate_names_are_legal_and_removed_together() {
        let order = Arc::new(parking_lot::Mutex::new(Vec::new()));
        let chain = CallbackChain::new("dupes");
        chain.add_named("same", recorder(&order, "first"));
        chain.add_named("same", recorder(&order, "second"));
        assert_eq!(chain.names(), ["same", "same"]);
        assert_eq!(chain.remove("same"), 2);
        assert!(chain.is_empty());
        assert_eq!(chain.remove("same"), 0);
    }

    #[tokio::test]
    async fn a_chain_as_a_handler_maps_verdicts() {
        let chain = Arc::new(CallbackChain::new("nested"));
        chain.add(ChainEntry::new(
            "stop",
            Arc::new(SyncHandler::new(|_: &CallbackContext| {
                Ok(CallbackVerdict::Stop)
            })),
        ));
        let ctx = CallbackContext::new(CallbackEvent::AgentStarted);
        assert!(matches!(
            chain.handle(&ctx).await.unwrap(),
            CallbackVerdict::Stop
        ));
        assert_eq!(chain.name(), "nested");
    }

    #[tokio::test]
    async fn typed_payload_dispatch_serializes_the_input() {
        let chain = CallbackChain::new("typed");
        let seen = Arc::new(parking_lot::Mutex::new(None));
        let captured = Arc::clone(&seen);
        chain.add(ChainEntry::new(
            "capture",
            Arc::new(SyncHandler::new(move |ctx: &CallbackContext| {
                *captured.lock() = ctx.input.clone();
                Ok(CallbackVerdict::Continue)
            })),
        ));
        let report = chain
            .trigger_payload(CallbackEvent::ToolInvokeInput, &json!({"q": "rust"}))
            .await
            .unwrap();
        assert_eq!(report.fired, 1);
        assert_eq!(*seen.lock(), Some(json!({"q": "rust"})));
    }
}
