//! Reload handlers: what a consumer does when a watched root changes.
//!
//! A [`ReloadHandler`] is the action half of the watcher. The event half (the
//! [`WatchEvent`]) names a kind, path and change; the handler re-reads whatever
//! state that path feeds and publishes nothing itself. [`ReloadHub`] keeps the
//! handlers in registration order, runs only those interested in an event's
//! kind, one after another, and turns every failure into a
//! [`ReloadFailure`] record so one broken body never blocks the rest.
//!
//! [`FnReloadHandler`] adapts a closure into a handler, which is the seam for
//! callers and tests that need a one-line body.
use std::sync::Arc;

use async_trait::async_trait;
use futures::future::BoxFuture;
use openjiuwen_common::error::Result;

use crate::watcher::event::{ReloadFailure, WatchEvent, WatchTargetKind};

/// A reload body invoked after a watched root changed.
///
/// Implementors re-read the state the event points at and return an error when
/// they could not, so the watcher can publish it as a
/// [`WatchUpdate::ReloadFailed`](crate::watcher::WatchUpdate::ReloadFailed).
#[async_trait]
pub trait ReloadHandler: Send + Sync {
    /// Returns the handler's stable name (`"skills"`, `"instruction"`, …), used
    /// in [`ReloadFailure::handler`] and in logs.
    fn name(&self) -> &str;

    /// Returns the event kinds this handler reacts to.
    ///
    /// An empty slice disables the handler without unregistering it.
    fn target_kinds(&self) -> &[WatchTargetKind];

    /// Re-reads the state behind `event`.
    ///
    /// # Errors
    ///
    /// Returns an error when the state could not be refreshed; the hub records
    /// it and keeps dispatching.
    async fn reload(&self, event: &WatchEvent) -> Result<()>;
}

/// The registered reload handlers of one watcher.
///
/// Handlers run in registration order and each dispatch is scoped to the
/// handlers whose [`ReloadHandler::target_kinds`] contains the event's kind.
/// Every handler runs to completion before the next one starts, so two bodies
/// never race for the same file; a failing body is recorded and skipped past.
#[derive(Default)]
pub struct ReloadHub {
    handlers: Vec<Arc<dyn ReloadHandler>>,
}

impl ReloadHub {
    /// Creates an empty hub.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns whether no handler is registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.handlers.is_empty()
    }

    /// Returns how many handlers are registered.
    #[must_use]
    pub fn len(&self) -> usize {
        self.handlers.len()
    }

    /// Returns the handler names in registration order.
    #[must_use]
    pub fn handler_names(&self) -> Vec<&str> {
        self.handlers.iter().map(|handler| handler.name()).collect()
    }

    /// Returns this hub with `handler` registered last.
    #[must_use]
    pub fn with_handler<H: ReloadHandler + 'static>(mut self, handler: H) -> Self {
        self.add_handler(Arc::new(handler));
        self
    }

    /// Registers `handler` last.
    pub fn add_handler(&mut self, handler: Arc<dyn ReloadHandler>) {
        self.handlers.push(handler);
    }

    /// Runs every interested handler for `event` and collects their failures.
    ///
    /// Only handlers whose [`ReloadHandler::target_kinds`] contains
    /// `event.kind` run. Each failure becomes a reload-failed update; the caller
    /// publishes them as
    /// [`WatchUpdate::ReloadFailed`](crate::watcher::WatchUpdate::ReloadFailed).
    /// A failing handler is logged at `warn` and never stops the others.
    pub async fn dispatch(&self, event: &WatchEvent) -> Vec<ReloadFailure> {
        let mut failures = Vec::new();
        for handler in &self.handlers {
            if !handler.target_kinds().contains(&event.kind) {
                continue;
            }
            let Err(error) = handler.reload(event).await else {
                continue;
            };
            let failure = ReloadFailure {
                sequence: event.sequence,
                handler: handler.name().to_owned(),
                kind: event.kind,
                path: event.path.clone(),
                error_kind: error.kind().as_str().to_owned(),
                message: error.to_string(),
            };
            tracing::warn!(
                handler = %failure.handler,
                sequence = failure.sequence,
                path = %failure.path.display(),
                error = %failure.message,
                "workspace watcher reload handler failed"
            );
            failures.push(failure);
        }
        failures
    }
}

impl std::fmt::Debug for ReloadHub {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReloadHub")
            .field("handlers", &self.handler_names())
            .finish()
    }
}

/// The boxed closure shape a [`FnReloadHandler`] stores.
type BoxedReloadBody =
    Box<dyn for<'a> Fn(&'a WatchEvent) -> BoxFuture<'a, Result<()>> + Send + Sync>;

/// A [`ReloadHandler`] built from a name, a kind list and a closure.
///
/// This is the seam for callers and tests that do not need a named type.
pub struct FnReloadHandler {
    name: String,
    kinds: Vec<WatchTargetKind>,
    body: BoxedReloadBody,
}

impl FnReloadHandler {
    /// Creates a handler named `name` that reacts to `kinds` and runs `body`.
    #[must_use]
    pub fn new<F>(name: impl Into<String>, kinds: Vec<WatchTargetKind>, body: F) -> Self
    where
        F: for<'a> Fn(&'a WatchEvent) -> BoxFuture<'a, Result<()>> + Send + Sync + 'static,
    {
        Self {
            name: name.into(),
            kinds,
            body: Box::new(body),
        }
    }
}

impl std::fmt::Debug for FnReloadHandler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FnReloadHandler")
            .field("name", &self.name)
            .field("kinds", &self.kinds)
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl ReloadHandler for FnReloadHandler {
    fn name(&self) -> &str {
        &self.name
    }

    fn target_kinds(&self) -> &[WatchTargetKind] {
        &self.kinds
    }

    async fn reload(&self, event: &WatchEvent) -> Result<()> {
        let future = (self.body)(event);
        future.await
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use crate::watcher::event::{ChangeKind, WatchTargetKind};
    use openjiuwen_common::error::Error;
    use std::path::PathBuf;
    use std::sync::Mutex;

    /// Builds an event of `kind` with the given sequence number.
    fn event(sequence: u64, kind: WatchTargetKind) -> WatchEvent {
        WatchEvent {
            sequence,
            kind,
            path: PathBuf::from("/w/skills/a.md"),
            change: ChangeKind::Modified,
            previous_path: None,
            label: Some("skills:workspace".to_owned()),
        }
    }

    /// A recorder that appends `name` to a shared log, optionally failing.
    fn recording(
        name: &'static str,
        kinds: Vec<WatchTargetKind>,
        log: Arc<Mutex<Vec<&'static str>>>,
        fail: bool,
    ) -> FnReloadHandler {
        FnReloadHandler::new(name, kinds, move |_event| {
            let log = Arc::clone(&log);
            Box::pin(async move {
                log.lock().unwrap().push(name);
                if fail {
                    Err(Error::sys_op(format!("{name} cannot reload")))
                } else {
                    Ok(())
                }
            })
        })
    }

    #[tokio::test]
    async fn only_matching_kinds_run() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let hub = ReloadHub::new()
            .with_handler(recording(
                "skills",
                vec![WatchTargetKind::Skills],
                Arc::clone(&log),
                false,
            ))
            .with_handler(recording(
                "mcp",
                vec![WatchTargetKind::McpConfig],
                Arc::clone(&log),
                false,
            ));
        assert_eq!(hub.len(), 2);
        assert!(!hub.is_empty());

        let failures = hub.dispatch(&event(1, WatchTargetKind::Skills)).await;
        assert_eq!(failures, Vec::<ReloadFailure>::new());
        assert_eq!(*log.lock().unwrap(), vec!["skills"]);
    }

    #[tokio::test]
    async fn a_handler_with_no_kinds_never_runs() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let hub = ReloadHub::new().with_handler(recording(
            "disabled",
            Vec::new(),
            Arc::clone(&log),
            false,
        ));
        assert_eq!(
            hub.dispatch(&event(1, WatchTargetKind::Skills)).await,
            Vec::<ReloadFailure>::new()
        );
        assert!(log.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn a_failing_handler_does_not_block_the_next() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let hub = ReloadHub::new()
            .with_handler(recording(
                "broken",
                vec![WatchTargetKind::Skills],
                Arc::clone(&log),
                true,
            ))
            .with_handler(recording(
                "working",
                vec![WatchTargetKind::Skills],
                Arc::clone(&log),
                false,
            ));

        let failures = hub.dispatch(&event(7, WatchTargetKind::Skills)).await;
        assert_eq!(failures.len(), 1);
        // Both bodies ran, in registration order.
        assert_eq!(*log.lock().unwrap(), vec!["broken", "working"]);

        let failure = &failures[0];
        assert_eq!(failure.handler, "broken");
        assert_eq!(failure.sequence, 7);
        assert_eq!(failure.kind, WatchTargetKind::Skills);
        assert_eq!(failure.path, PathBuf::from("/w/skills/a.md"));
        assert_eq!(failure.error_kind, "sys_op");
        assert!(failure.message.contains("broken cannot reload"));
    }

    #[tokio::test]
    async fn handler_names_follow_registration_order() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let mut hub = ReloadHub::new();
        assert!(hub.is_empty());
        assert_eq!(hub.handler_names(), Vec::<&str>::new());

        hub.add_handler(Arc::new(recording(
            "first",
            vec![WatchTargetKind::Skills],
            Arc::clone(&log),
            false,
        )));
        hub.add_handler(Arc::new(recording(
            "second",
            vec![WatchTargetKind::InstructionFile],
            Arc::clone(&log),
            false,
        )));
        assert_eq!(hub.handler_names(), vec!["first", "second"]);
        assert_eq!(hub.len(), 2);
        assert_eq!(
            format!("{hub:?}"),
            "ReloadHub { handlers: [\"first\", \"second\"] }"
        );
    }

    #[tokio::test]
    async fn dispatch_without_handlers_is_a_no_op() {
        let hub = ReloadHub::new();
        assert_eq!(
            hub.dispatch(&event(1, WatchTargetKind::Other)).await,
            Vec::<ReloadFailure>::new()
        );
    }
}
