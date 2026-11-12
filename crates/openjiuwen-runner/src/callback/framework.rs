//! The registration registry and the dispatch loop.
//!
//! Ports the registration bookkeeping and `trigger` path of
//! `openjiuwen/core/runner/callback/framework.py`, keeping only the parts the
//! kernel needs: priority ordering, `once`, enable/disable, error isolation,
//! and removal by id/event/namespace/tag.
//!
//! The framework owns no tasks and no globals. It is `Arc`-shared from a
//! `Runner`, and every lock is a short synchronous `parking_lot` read
//! or write: [`CallbackFramework::trigger`] snapshots the handler list, releases
//! the lock, and only then awaits handlers.
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
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use openjiuwen_common::error::Result;

use super::context::{CallbackContext, CallbackFailure, CallbackReport, CallbackVerdict};
use super::event::CallbackEvent;
use super::handler::{AsyncHandler, CallbackHandler, CallbackResult, SyncHandler};
use crate::callback::LOG_TARGET;

/// The handle a registration returns and removal takes.
///
/// Identifiers are process-local and monotonic within one
/// [`CallbackFramework`]; they are never reused.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct CallbackId(u64);

impl CallbackId {
    /// Rebuilds an id from its raw value (used by tests and wire round-trips).
    #[must_use]
    pub(crate) const fn from_raw(raw: u64) -> Self {
        Self(raw)
    }

    /// The raw numeric value.
    #[must_use]
    pub const fn as_u64(self) -> u64 {
        self.0
    }
}

impl fmt::Display for CallbackId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "callback-{}", self.0)
    }
}

/// A registration as the registry stores it, minus the handler itself.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CallbackInfo {
    /// The removal handle.
    pub id: CallbackId,
    /// The event this handler subscribes to.
    pub event: CallbackEvent,
    /// The handler's name.
    pub name: String,
    /// Dispatch priority; higher runs first.
    pub priority: i32,
    /// Free-form namespace (`"default"` unless overridden).
    pub namespace: String,
    /// Whether the handler is removed after its first successful run.
    pub once: bool,
    /// Whether dispatch currently runs it.
    pub enabled: bool,
    /// Labels used by [`CallbackFramework::unregister_tag`].
    pub tags: BTreeSet<String>,
    /// Global registration order; the tiebreak between equal priorities.
    pub sequence: u64,
}

/// One handler to register.
///
/// Build it with [`CallbackRegistration::new`] and the fluent setters, then
/// pass it to [`CallbackFramework::register`].
pub struct CallbackRegistration {
    /// The event to subscribe to.
    pub event: CallbackEvent,
    /// The handler itself.
    pub handler: Arc<dyn CallbackHandler>,
    /// Dispatch priority; higher runs first.
    pub priority: i32,
    /// Free-form namespace.
    pub namespace: String,
    /// Labels used by [`CallbackFramework::unregister_tag`].
    pub tags: BTreeSet<String>,
    /// Whether the handler is removed after its first successful run.
    pub once: bool,
}

impl CallbackRegistration {
    /// A registration at priority `0` in namespace `"default"`, not `once`.
    #[must_use]
    pub fn new(event: CallbackEvent, handler: Arc<dyn CallbackHandler>) -> Self {
        Self {
            event,
            handler,
            priority: 0,
            namespace: "default".to_owned(),
            tags: BTreeSet::new(),
            once: false,
        }
    }

    /// Sets the dispatch priority.
    #[must_use]
    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }

    /// Sets the namespace.
    #[must_use]
    pub fn with_namespace(mut self, namespace: impl Into<String>) -> Self {
        self.namespace = namespace.into();
        self
    }

    /// Adds one tag.
    #[must_use]
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.insert(tag.into());
        self
    }

    /// Removes the handler after its first successful run.
    #[must_use]
    pub fn once(mut self) -> Self {
        self.once = true;
        self
    }
}

impl fmt::Debug for CallbackRegistration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CallbackRegistration")
            .field("event", &self.event)
            .field("handler", &self.handler.name())
            .field("priority", &self.priority)
            .field("namespace", &self.namespace)
            .field("tags", &self.tags)
            .field("once", &self.once)
            .finish()
    }
}

/// A stored registration: its public metadata plus the live handler.
#[derive(Clone)]
struct Entry {
    info: CallbackInfo,
    handler: Arc<dyn CallbackHandler>,
}

#[derive(Default)]
struct State {
    registrations: HashMap<CallbackEvent, Vec<Entry>>,
}

/// The priority-ordered event registry.
///
/// Duplicate registration is legal and dispatches twice (Python parity);
/// [`CallbackId`] is the explicit removal handle.
pub struct CallbackFramework {
    state: RwLock<State>,
    next_id: AtomicU64,
    next_sequence: AtomicU64,
}

impl CallbackFramework {
    /// An empty framework.
    #[must_use]
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Registers one handler and returns its removal handle.
    pub fn register(&self, registration: CallbackRegistration) -> CallbackId {
        let id = CallbackId::from_raw(self.next_id.fetch_add(1, Ordering::Relaxed));
        let sequence = self.next_sequence.fetch_add(1, Ordering::Relaxed);
        let info = CallbackInfo {
            id,
            event: registration.event.clone(),
            name: registration.handler.name().to_owned(),
            priority: registration.priority,
            namespace: registration.namespace,
            once: registration.once,
            enabled: true,
            tags: registration.tags,
            sequence,
        };
        let entry = Entry {
            info,
            handler: registration.handler,
        };
        self.state
            .write()
            .registrations
            .entry(registration.event)
            .or_default()
            .push(entry);
        id
    }

    /// Registers a synchronous closure at `priority`.
    pub fn register_sync<F>(&self, event: CallbackEvent, priority: i32, handler: F) -> CallbackId
    where
        F: Fn(&CallbackContext) -> CallbackResult + Send + Sync + 'static,
    {
        self.register(
            CallbackRegistration::new(event, Arc::new(SyncHandler::new(handler)))
                .with_priority(priority),
        )
    }

    /// Registers an async closure at `priority`.
    ///
    /// The closure takes the context by value and returns a boxed future, e.g.
    /// `|ctx| Box::pin(async move { Ok(CallbackVerdict::Continue) })`.
    pub fn register_async<F>(&self, event: CallbackEvent, priority: i32, handler: F) -> CallbackId
    where
        F: Fn(CallbackContext) -> futures::future::BoxFuture<'static, CallbackResult>
            + Send
            + Sync
            + 'static,
    {
        self.register(
            CallbackRegistration::new(event, Arc::new(AsyncHandler::new(handler)))
                .with_priority(priority),
        )
    }

    /// Removes one handler by id; returns whether it existed.
    pub fn unregister(&self, id: CallbackId) -> bool {
        let mut state = self.state.write();
        let mut removed = false;
        for entries in state.registrations.values_mut() {
            let before = entries.len();
            entries.retain(|entry| entry.info.id != id);
            removed |= entries.len() != before;
        }
        if removed {
            state.registrations.retain(|_, entries| !entries.is_empty());
        }
        removed
    }

    /// Removes every handler for `event`; returns how many were removed.
    // The by-value signature is the documented API.
    #[allow(clippy::needless_pass_by_value)]
    pub fn unregister_event(&self, event: CallbackEvent) -> usize {
        self.state
            .write()
            .registrations
            .remove(&event)
            .map_or(0, |entries| entries.len())
    }

    /// Removes every handler in `namespace`; returns how many were removed.
    pub fn unregister_namespace(&self, namespace: &str) -> usize {
        let mut state = self.state.write();
        let mut removed = 0;
        for entries in state.registrations.values_mut() {
            let before = entries.len();
            entries.retain(|entry| entry.info.namespace != namespace);
            removed += before - entries.len();
        }
        state.registrations.retain(|_, entries| !entries.is_empty());
        removed
    }

    /// Removes every handler carrying `tag`; returns how many were removed.
    pub fn unregister_tag(&self, tag: &str) -> usize {
        let mut state = self.state.write();
        let mut removed = 0;
        for entries in state.registrations.values_mut() {
            let before = entries.len();
            entries.retain(|entry| !entry.info.tags.contains(tag));
            removed += before - entries.len();
        }
        state.registrations.retain(|_, entries| !entries.is_empty());
        removed
    }

    /// Removes every registration; returns how many were removed.
    pub fn unregister_all(&self) -> usize {
        let mut state = self.state.write();
        let removed = state.registrations.values().map(Vec::len).sum::<usize>();
        state.registrations.clear();
        removed
    }

    /// Enables or disables one handler; returns whether the id exists.
    ///
    /// A disabled handler is skipped without being reported as fired.
    pub fn set_enabled(&self, id: CallbackId, enabled: bool) -> bool {
        let mut state = self.state.write();
        for entries in state.registrations.values_mut() {
            for entry in entries {
                if entry.info.id == id {
                    entry.info.enabled = enabled;
                    return true;
                }
            }
        }
        false
    }

    /// Whether the handler `id` is enabled, or `None` when the id is unknown.
    #[must_use]
    pub fn enabled(&self, id: CallbackId) -> Option<bool> {
        self.state
            .read()
            .registrations
            .values()
            .flatten()
            .find(|entry| entry.info.id == id)
            .map(|entry| entry.info.enabled)
    }

    /// Whether `event` has a registered handler.
    ///
    /// Python parity: a disabled handler still counts as a subscriber.
    #[allow(clippy::needless_pass_by_value)]
    #[must_use]
    pub fn has_subscribers(&self, event: CallbackEvent) -> bool {
        self.state
            .read()
            .registrations
            .get(&event)
            .is_some_and(|entries| !entries.is_empty())
    }

    /// The handlers for `event` in dispatch order (priority descending,
    /// registration order as the tiebreak), disabled ones included.
    // The by-value signature is the documented API.
    #[allow(clippy::needless_pass_by_value)]
    #[must_use]
    pub fn handlers(&self, event: CallbackEvent) -> Vec<CallbackInfo> {
        let state = self.state.read();
        state
            .registrations
            .get(&event)
            .map_or_else(Vec::new, |entries| {
                sorted(entries)
                    .into_iter()
                    .map(|entry| entry.info)
                    .collect()
            })
    }

    /// Every event with at least one registration, sorted by variant order.
    #[must_use]
    pub fn events(&self) -> Vec<CallbackEvent> {
        let mut events: Vec<CallbackEvent> =
            self.state.read().registrations.keys().cloned().collect();
        events.sort_unstable();
        events
    }

    /// The total number of registrations.
    #[must_use]
    pub fn len(&self) -> usize {
        self.state.read().registrations.values().map(Vec::len).sum()
    }

    /// Whether there are no registrations at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Dispatches `event` to its handlers.
    ///
    /// Handlers run in descending `priority`, ties broken by registration
    /// order. `Ok(Continue)` continues, `Ok(Stop)` ends dispatch with
    /// `report.aborted = true`, `Ok(Abort(error))` ends dispatch and returns
    /// that error, and `Err(error)` is recorded in `report.failures` (and
    /// logged at `warn`) while dispatch continues.
    ///
    /// A `once` handler is removed after a successful (`Continue`/`Stop`) run;
    /// a failed or aborted run does not consume it.
    ///
    /// # Errors
    ///
    /// Returns the [`CallbackVerdict::Abort`] error of the handler that aborted
    /// dispatch. Handler `Err` values are data, never a `trigger` error.
    pub async fn trigger(
        &self,
        event: CallbackEvent,
        ctx: &CallbackContext,
    ) -> Result<CallbackReport> {
        let snapshot = {
            let state = self.state.read();
            match state.registrations.get(&event) {
                // No subscribers: the fast path never touches the context.
                None => return Ok(CallbackReport::new(event)),
                Some(entries) if entries.is_empty() => return Ok(CallbackReport::new(event)),
                Some(entries) => sorted(entries),
            }
        };

        let mut report = CallbackReport::new(event.clone());
        for entry in snapshot {
            if !entry.info.enabled {
                continue;
            }
            match entry.handler.handle(ctx).await {
                Ok(CallbackVerdict::Continue) => {
                    report.fired += 1;
                    if entry.info.once {
                        self.unregister(entry.info.id);
                    }
                }
                Ok(CallbackVerdict::Stop) => {
                    report.fired += 1;
                    report.aborted = true;
                    if entry.info.once {
                        self.unregister(entry.info.id);
                    }
                    break;
                }
                Ok(CallbackVerdict::Abort(error)) => {
                    report.aborted = true;
                    tracing::debug!(
                        target: LOG_TARGET,
                        event = %event.as_str(),
                        handler = %entry.info.name,
                        error = %error,
                        "callback aborted dispatch"
                    );
                    return Err(error);
                }
                Err(error) => {
                    tracing::warn!(
                        target: LOG_TARGET,
                        event = %event.as_str(),
                        handler = %entry.info.name,
                        error = %error,
                        "callback handler failed; dispatch continues"
                    );
                    report.failures.push(CallbackFailure {
                        id: entry.info.id,
                        handler: entry.info.name,
                        error,
                    });
                }
            }
        }
        Ok(report)
    }
}

impl Default for CallbackFramework {
    fn default() -> Self {
        Self {
            state: RwLock::new(State::default()),
            next_id: AtomicU64::new(0),
            next_sequence: AtomicU64::new(0),
        }
    }
}

/// Dispatch order: priority descending, registration order as the tiebreak.
fn sorted(entries: &[Entry]) -> Vec<Entry> {
    let mut snapshot = entries.to_vec();
    snapshot.sort_by(|left, right| {
        right
            .info
            .priority
            .cmp(&left.info.priority)
            .then(left.info.sequence.cmp(&right.info.sequence))
    });
    snapshot
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc as StdArc, Mutex};

    use openjiuwen_common::error::Error;

    fn continue_handler() -> CallbackResult {
        Ok(CallbackVerdict::Continue)
    }

    #[test]
    fn registration_builders_apply_their_defaults_and_overrides() {
        let registration = CallbackRegistration::new(
            CallbackEvent::AgentStarted,
            Arc::new(SyncHandler::new(|_: &CallbackContext| continue_handler())),
        );
        assert_eq!(registration.priority, 0);
        assert_eq!(registration.namespace, "default");
        assert!(!registration.once);
        assert!(registration.tags.is_empty());

        let registration = registration
            .with_priority(7)
            .with_namespace("custom")
            .with_tag("a")
            .with_tag("b")
            .once();
        assert_eq!(registration.priority, 7);
        assert_eq!(registration.namespace, "custom");
        assert_eq!(registration.tags.len(), 2);
        assert!(registration.once);
        assert!(format!("{registration:?}").contains("custom"));
    }

    #[tokio::test]
    async fn handlers_run_by_priority_descending_with_insertion_tiebreak() {
        // ports tests/unit_tests/core/runner/callback/test_framework_triggers.py
        let framework = CallbackFramework::new();
        let order = StdArc::new(Mutex::new(Vec::new()));

        for (name, priority) in [("low", 0), ("high", 10), ("mid_a", 5), ("mid_b", 5)] {
            let order = StdArc::clone(&order);
            let registration = CallbackRegistration::new(
                CallbackEvent::AgentStarted,
                Arc::new(
                    SyncHandler::new(move |_: &CallbackContext| {
                        order.lock().unwrap().push(name);
                        continue_handler()
                    })
                    .with_name(name),
                ),
            )
            .with_priority(priority);
            framework.register(registration);
        }

        let report = framework
            .trigger(
                CallbackEvent::AgentStarted,
                &CallbackContext::new(CallbackEvent::AgentStarted),
            )
            .await
            .unwrap();
        assert_eq!(report.fired, 4);
        assert!(report.is_clean());
        assert_eq!(*order.lock().unwrap(), ["high", "mid_a", "mid_b", "low"]);
    }

    #[tokio::test]
    async fn sync_and_async_handlers_both_dispatch() {
        let framework = CallbackFramework::new();
        let seen = StdArc::new(Mutex::new(Vec::new()));

        let sync_seen = StdArc::clone(&seen);
        framework.register_sync(
            CallbackEvent::AgentStarted,
            0,
            move |_: &CallbackContext| {
                sync_seen.lock().unwrap().push("sync");
                continue_handler()
            },
        );
        let async_seen = StdArc::clone(&seen);
        framework.register_async(CallbackEvent::AgentStarted, 0, move |_ctx| {
            let async_seen = StdArc::clone(&async_seen);
            Box::pin(async move {
                async_seen.lock().unwrap().push("async");
                continue_handler()
            }) as futures::future::BoxFuture<'static, CallbackResult>
        });

        let report = framework
            .trigger(
                CallbackEvent::AgentStarted,
                &CallbackContext::new(CallbackEvent::AgentStarted),
            )
            .await
            .unwrap();
        assert_eq!(report.fired, 2);
        assert_eq!(*seen.lock().unwrap(), ["sync", "async"]);
    }

    #[tokio::test]
    async fn a_failing_handler_is_recorded_and_dispatch_continues() {
        let framework = CallbackFramework::new();
        let ran = StdArc::new(Mutex::new(false));

        framework.register_sync(CallbackEvent::AgentStarted, 0, |_: &CallbackContext| {
            Err(Error::validation("first failed"))
        });
        let later = StdArc::clone(&ran);
        framework.register_sync(
            CallbackEvent::AgentStarted,
            0,
            move |_: &CallbackContext| {
                *later.lock().unwrap() = true;
                continue_handler()
            },
        );

        let report = framework
            .trigger(
                CallbackEvent::AgentStarted,
                &CallbackContext::new(CallbackEvent::AgentStarted),
            )
            .await
            .unwrap();
        assert_eq!(report.fired, 1);
        assert_eq!(report.failures.len(), 1);
        assert_eq!(report.failures[0].handler, "sync_handler");
        assert!(report.failures[0]
            .error
            .to_string()
            .contains("first failed"));
        assert!(!report.is_clean());
        assert!(*ran.lock().unwrap());
    }

    #[tokio::test]
    async fn abort_propagates_its_error_and_stops_dispatch() {
        let framework = CallbackFramework::new();
        let later = StdArc::new(Mutex::new(false));

        framework.register_sync(CallbackEvent::AgentStarted, 0, |_: &CallbackContext| {
            Ok(CallbackVerdict::Abort(Error::Security(
                "blocked".to_owned(),
            )))
        });
        let later_clone = StdArc::clone(&later);
        framework.register_sync(
            CallbackEvent::AgentStarted,
            0,
            move |_: &CallbackContext| {
                *later_clone.lock().unwrap() = true;
                continue_handler()
            },
        );

        let error = framework
            .trigger(
                CallbackEvent::AgentStarted,
                &CallbackContext::new(CallbackEvent::AgentStarted),
            )
            .await
            .unwrap_err();
        assert_eq!(error.kind(), openjiuwen_common::error::ErrorKind::Security);
        assert_eq!(error.to_string(), "security: blocked");
        assert!(!*later.lock().unwrap());
    }

    #[tokio::test]
    async fn stop_ends_dispatch_without_an_error() {
        let framework = CallbackFramework::new();
        let later = StdArc::new(Mutex::new(false));

        framework.register_sync(CallbackEvent::AgentStarted, 10, |_: &CallbackContext| {
            Ok(CallbackVerdict::Stop)
        });
        let later_clone = StdArc::clone(&later);
        framework.register_sync(
            CallbackEvent::AgentStarted,
            0,
            move |_: &CallbackContext| {
                *later_clone.lock().unwrap() = true;
                continue_handler()
            },
        );

        let report = framework
            .trigger(
                CallbackEvent::AgentStarted,
                &CallbackContext::new(CallbackEvent::AgentStarted),
            )
            .await
            .unwrap();
        assert_eq!(report.fired, 1);
        assert!(report.aborted);
        assert!(report.failures.is_empty());
        assert!(!report.is_clean());
        assert!(!*later.lock().unwrap());
    }

    #[tokio::test]
    async fn once_handlers_are_removed_after_a_successful_run() {
        let framework = CallbackFramework::new();
        let count = StdArc::new(Mutex::new(0));
        let count_clone = StdArc::clone(&count);
        let registration = CallbackRegistration::new(
            CallbackEvent::AgentStarted,
            Arc::new(SyncHandler::new(move |_: &CallbackContext| {
                *count_clone.lock().unwrap() += 1;
                continue_handler()
            })),
        )
        .once();
        framework.register(registration);

        let first = framework
            .trigger(
                CallbackEvent::AgentStarted,
                &CallbackContext::new(CallbackEvent::AgentStarted),
            )
            .await
            .unwrap();
        assert_eq!(first.fired, 1);
        assert_eq!(framework.len(), 0);

        let second = framework
            .trigger(
                CallbackEvent::AgentStarted,
                &CallbackContext::new(CallbackEvent::AgentStarted),
            )
            .await
            .unwrap();
        assert_eq!(second.fired, 0);
        assert_eq!(*count.lock().unwrap(), 1);
    }

    #[tokio::test]
    async fn once_handlers_are_consumed_by_stop_but_not_by_failure_or_abort() {
        let framework = CallbackFramework::new();
        framework.register(
            CallbackRegistration::new(
                CallbackEvent::AgentStarted,
                Arc::new(SyncHandler::new(|_: &CallbackContext| {
                    Ok(CallbackVerdict::Stop)
                })),
            )
            .once(),
        );
        framework
            .trigger(
                CallbackEvent::AgentStarted,
                &CallbackContext::new(CallbackEvent::AgentStarted),
            )
            .await
            .unwrap();
        assert_eq!(framework.len(), 0);

        let framework = CallbackFramework::new();
        framework.register(
            CallbackRegistration::new(
                CallbackEvent::AgentStarted,
                Arc::new(SyncHandler::new(|_: &CallbackContext| {
                    Err(Error::validation("nope"))
                })),
            )
            .once(),
        );
        framework
            .trigger(
                CallbackEvent::AgentStarted,
                &CallbackContext::new(CallbackEvent::AgentStarted),
            )
            .await
            .unwrap();
        assert_eq!(
            framework.len(),
            1,
            "a failure does not consume a once handler"
        );

        let framework = CallbackFramework::new();
        framework.register(
            CallbackRegistration::new(
                CallbackEvent::AgentStarted,
                Arc::new(SyncHandler::new(|_: &CallbackContext| {
                    Ok(CallbackVerdict::Abort(Error::Security(
                        "blocked".to_owned(),
                    )))
                })),
            )
            .once(),
        );
        let _ = framework
            .trigger(
                CallbackEvent::AgentStarted,
                &CallbackContext::new(CallbackEvent::AgentStarted),
            )
            .await;
        assert_eq!(
            framework.len(),
            1,
            "an abort does not consume a once handler"
        );
    }

    #[tokio::test]
    async fn disabled_handlers_are_skipped_without_being_fired() {
        let framework = CallbackFramework::new();
        let id = framework.register_sync(CallbackEvent::AgentStarted, 0, |_: &CallbackContext| {
            continue_handler()
        });
        assert!(framework.set_enabled(id, false));
        assert_eq!(framework.enabled(id), Some(false));
        assert_eq!(framework.enabled(CallbackId::from_raw(999)), None);
        assert!(framework.has_subscribers(CallbackEvent::AgentStarted));

        let report = framework
            .trigger(
                CallbackEvent::AgentStarted,
                &CallbackContext::new(CallbackEvent::AgentStarted),
            )
            .await
            .unwrap();
        assert_eq!(report.fired, 0);
        assert!(!framework.handlers(CallbackEvent::AgentStarted)[0].enabled);

        assert!(framework.set_enabled(id, true));
        let report = framework
            .trigger(
                CallbackEvent::AgentStarted,
                &CallbackContext::new(CallbackEvent::AgentStarted),
            )
            .await
            .unwrap();
        assert_eq!(report.fired, 1);
        assert!(!framework.set_enabled(CallbackId::from_raw(999), true));
    }

    #[tokio::test]
    async fn duplicate_registration_dispatches_twice() {
        let framework = CallbackFramework::new();
        framework.register_sync(CallbackEvent::AgentStarted, 0, |_: &CallbackContext| {
            continue_handler()
        });
        framework.register_sync(CallbackEvent::AgentStarted, 0, |_: &CallbackContext| {
            continue_handler()
        });
        let report = framework
            .trigger(
                CallbackEvent::AgentStarted,
                &CallbackContext::new(CallbackEvent::AgentStarted),
            )
            .await
            .unwrap();
        assert_eq!(report.fired, 2);
        assert_eq!(framework.handlers(CallbackEvent::AgentStarted).len(), 2);
    }

    #[tokio::test]
    async fn unregister_by_id_event_namespace_and_tag() {
        let framework = CallbackFramework::new();

        let first =
            framework.register_sync(CallbackEvent::AgentStarted, 0, |_: &CallbackContext| {
                continue_handler()
            });
        framework.register(
            CallbackRegistration::new(
                CallbackEvent::AgentStarted,
                Arc::new(SyncHandler::new(|_: &CallbackContext| continue_handler())),
            )
            .with_namespace("custom"),
        );
        framework.register(
            CallbackRegistration::new(
                CallbackEvent::ToolInvokeInput,
                Arc::new(SyncHandler::new(|_: &CallbackContext| continue_handler())),
            )
            .with_tag("audit"),
        );

        assert!(framework.unregister(first));
        assert!(!framework.unregister(first));
        assert_eq!(framework.unregister_namespace("custom"), 1);
        assert_eq!(framework.unregister_tag("audit"), 1);
        assert_eq!(framework.unregister_all(), 0);
        assert!(framework.is_empty());

        framework.register_sync(CallbackEvent::AgentStarted, 0, |_: &CallbackContext| {
            continue_handler()
        });
        assert_eq!(framework.unregister_event(CallbackEvent::AgentStarted), 1);
        assert_eq!(framework.unregister_event(CallbackEvent::AgentStarted), 0);
    }

    #[tokio::test]
    async fn events_reports_subscribed_events_in_order() {
        let framework = CallbackFramework::new();
        assert_eq!(framework.events(), Vec::new());
        framework.register_sync(CallbackEvent::ToolInvokeInput, 0, |_: &CallbackContext| {
            continue_handler()
        });
        framework.register_sync(CallbackEvent::AgentStarted, 0, |_: &CallbackContext| {
            continue_handler()
        });
        assert_eq!(
            framework.events(),
            vec![CallbackEvent::AgentStarted, CallbackEvent::ToolInvokeInput]
        );
    }

    #[tokio::test]
    async fn triggering_without_subscribers_is_a_clean_no_op() {
        let framework = CallbackFramework::new();
        let report = framework
            .trigger(
                CallbackEvent::AgentStarted,
                &CallbackContext::new(CallbackEvent::AgentStarted).with_component("unread"),
            )
            .await
            .unwrap();
        assert_eq!(report.fired, 0);
        assert!(report.is_clean());
        assert!(!framework.has_subscribers(CallbackEvent::AgentStarted));
    }

    #[tokio::test]
    async fn handlers_are_isolated_per_event() {
        let framework = CallbackFramework::new();
        let ran = StdArc::new(Mutex::new(false));
        let ran_clone = StdArc::clone(&ran);
        framework.register_sync(
            CallbackEvent::ToolInvokeInput,
            0,
            move |_: &CallbackContext| {
                *ran_clone.lock().unwrap() = true;
                continue_handler()
            },
        );
        let report = framework
            .trigger(
                CallbackEvent::AgentStarted,
                &CallbackContext::new(CallbackEvent::AgentStarted),
            )
            .await
            .unwrap();
        assert_eq!(report.fired, 0);
        assert!(!*ran.lock().unwrap());

        let report = framework
            .trigger(
                CallbackEvent::ToolInvokeInput,
                &CallbackContext::new(CallbackEvent::ToolInvokeInput),
            )
            .await
            .unwrap();
        assert_eq!(report.fired, 1);
        assert!(*ran.lock().unwrap());
    }

    #[tokio::test]
    async fn handlers_reports_dispatch_order_and_metadata() {
        let framework = CallbackFramework::new();
        framework.register(
            CallbackRegistration::new(
                CallbackEvent::AgentStarted,
                Arc::new(SyncHandler::new(|_: &CallbackContext| continue_handler()).with_name("b")),
            )
            .with_priority(1)
            .with_namespace("ns")
            .with_tag("t"),
        );
        framework.register(
            CallbackRegistration::new(
                CallbackEvent::AgentStarted,
                Arc::new(SyncHandler::new(|_: &CallbackContext| continue_handler()).with_name("a")),
            )
            .with_priority(5),
        );

        let info = framework.handlers(CallbackEvent::AgentStarted);
        assert_eq!(info.len(), 2);
        assert_eq!(info[0].name, "a");
        assert_eq!(info[1].name, "b");
        assert_eq!(info[1].priority, 1);
        assert_eq!(info[1].namespace, "ns");
        assert!(info[1].tags.contains("t"));
        assert_eq!(info[0].event, CallbackEvent::AgentStarted);
        // `a` was registered second but has the higher priority, so it holds
        // the larger id while running first.
        assert!(info[1].id < info[0].id);
        assert!(info[1].sequence < info[0].sequence);
        assert_eq!(
            info[0].id.to_string(),
            format!("callback-{}", info[0].id.as_u64())
        );
    }
}
