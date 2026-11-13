//! Preparing a run for the hub: what a run path registers, what a host sees,
//! the handle that registers it, and the binding that removes it again.
//!
//! [`ActiveRunSpec`] is the crate-internal description a run path builds;
//! [`ActiveRunSpec::for_execution`] is the constructor the execution paths
//! adopt, and [`bind_run`] is the one call that registers the spec and attaches
//! the run's session. [`RunBinding`] owns the registration: dropping it (or
//! calling [`RunBinding::unregister`]) removes the run idempotently, so an
//! `?`-returned error or a panicking task cannot leak a live entry.
//!
//! The execution paths are wired into this bridge: the agent, workflow, and team
//! run paths (and the team pool's runs) call [`bind_run`] to register their run, and
//! hold the returned [`RunBinding`] so the `?`-returned error and the dropped stream
//! paths both unregister it. Outside this module's own tests [`RunBinding::run_id`]
//! is still unreferenced, which is why the item-level `#[allow(dead_code)]`
//! attributes below stay as they are.
//!
//! The second half of the bridge is session observation: `watch_pending` turns the
//! `__interaction__` frames a run's session publishes into the hub's pending set,
//! and `spawn_cancellations` delivers the hub's cancellation decisions back to that
//! session.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Weak};

use futures::stream::BoxStream;
use futures::StreamExt;
use openjiuwen_common::error::Result;
use openjiuwen_common::utils::id::uuid_v4;
use openjiuwen_session::interaction::{InteractionResponse, INTERACTION_FRAME_TYPE};
use openjiuwen_session::stream::StreamEvent;
use openjiuwen_session::Session;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio_util::sync::CancellationToken;
use tracing::{debug, warn};

use crate::monitor::RunPhase;

use super::hub::{Inner, InteractionHub};
use super::TARGET;

/// What a run path asks the hub to register.
///
/// The run id is optional: [`InteractionHub::register`](super::InteractionHub::register)
/// mints a UUID v4 when it is absent (and treats an empty id as absent). The
/// cancellation token is the run's own token (`RunOptions::cancellation`), so
/// [`InteractionHub::cancel`](super::InteractionHub::cancel) trips the token the
/// run's `tokio::select!` observes. Not serde-serializable: the token is a
/// runtime concern.
#[derive(Debug, Clone)]
pub struct ActiveRunSpec {
    /// Run id to register under; `None` mints a fresh UUID v4.
    pub run_id: Option<String>,
    /// Human-readable label (the agent/workflow/team card name).
    pub label: String,
    /// Team name the run belongs to, when it is a team run.
    pub team_name: Option<String>,
    /// Session the run is bound to, when the session was already resolved.
    pub session_id: Option<String>,
    /// The run's cancellation token; `None` gives the hub its own token.
    pub cancellation: Option<CancellationToken>,
    /// Opaque detail reported with the run's initial monitor event.
    pub detail: Value,
}

impl ActiveRunSpec {
    /// Builds a spec with no run id, team, session, token, or detail.
    #[must_use]
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            run_id: None,
            label: label.into(),
            team_name: None,
            session_id: None,
            cancellation: None,
            detail: Value::Null,
        }
    }

    /// Builds the spec an execution path registers for one invocation.
    ///
    /// Clones the run's cancellation token so the hub and the run path share
    /// cancellation; the run id, labels, session id, and detail are copied.
    #[must_use]
    #[allow(dead_code)] // the execution paths adopt this constructor next
    pub(crate) fn for_execution(
        run_id: &str,
        label: &str,
        team_name: Option<&str>,
        session_id: Option<&str>,
        cancellation: &CancellationToken,
        detail: Value,
    ) -> Self {
        Self {
            run_id: Some(run_id.to_string()),
            label: label.to_string(),
            team_name: team_name.map(str::to_string),
            session_id: session_id.map(str::to_string),
            cancellation: Some(cancellation.clone()),
            detail,
        }
    }
}

/// The host-facing snapshot of one registered run.
///
/// `state` is the hub's own coordination state (it starts at
/// [`RunPhase::Queued`] and `pause`/`resume`/`cancel` advance it), not a query of
/// the run itself. `queued` counts the interaction request ids the hub has
/// observed on the run's session but not yet answered; it is not the same as the
/// inbox depth (the inbox holds inbound human payloads). Serialized with
/// `snake_case` field names; `state` uses the monitor's `snake_case` phase
/// strings.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ActiveRunInfo {
    /// The run id the entry is registered under.
    pub run_id: String,
    /// Human-readable label (the card name).
    pub label: String,
    /// Team name the run belongs to, when it is a team run.
    pub team_name: Option<String>,
    /// Session the run is bound to, once attached.
    pub session_id: Option<String>,
    /// The hub's coordination state.
    pub state: RunPhase,
    /// Whether the hub's interaction gate is closed (paused or cancelled).
    pub gate_closed: bool,
    /// Number of interaction request ids awaiting an answer.
    pub queued: usize,
}

/// A cheap handle to one registered run.
///
/// Returned by [`InteractionHub::register`](super::InteractionHub::register); it
/// carries the (possibly minted) run id and the run's cancellation token, and has
/// no `Drop` behaviour — the entry stays registered until
/// [`InteractionHub::unregister`](super::InteractionHub::unregister) removes it.
#[derive(Debug, Clone)]
pub struct RunHandle {
    /// The registered run id.
    run_id: String,
    /// The run's cancellation token (the registered one, or a fresh one).
    cancellation: CancellationToken,
}

impl RunHandle {
    /// Builds a handle; crate-internal because the hub mints the id.
    pub(super) fn new(run_id: String, cancellation: CancellationToken) -> Self {
        Self {
            run_id,
            cancellation,
        }
    }

    /// Returns the registered run id.
    #[must_use]
    pub fn run_id(&self) -> &str {
        &self.run_id
    }

    /// Returns a clone of the run's cancellation token.
    #[must_use]
    pub fn cancellation(&self) -> CancellationToken {
        self.cancellation.clone()
    }
}

/// A run registered on a hub; dropping it (or calling
/// [`RunBinding::unregister`]) removes the run, idempotently.
///
/// A binding for an absent hub is inert: it carries a run id but registers
/// nothing, so a caller can use it unconditionally.
#[allow(dead_code)] // the execution paths adopt this binding next
pub(crate) struct RunBinding {
    /// The hub the run is registered on; `None` makes the binding inert.
    hub: Option<Arc<InteractionHub>>,
    /// The run id the binding removes.
    run_id: String,
    /// Whether the run is still registered (guards idempotent unregistration).
    active: AtomicBool,
}

#[allow(dead_code)] // the execution paths adopt this binding next
impl RunBinding {
    /// Returns the run id this binding is for.
    pub(crate) fn run_id(&self) -> &str {
        &self.run_id
    }

    /// Removes the run from the hub, at most once.
    ///
    /// A mirror failure (the run was already removed, or the hub dropped it) is
    /// logged with `tracing::warn!` and never surfaced: the binding is teardown
    /// bookkeeping, not control flow.
    pub(crate) fn unregister(&self) {
        if !self.active.swap(false, Ordering::SeqCst) {
            return;
        }
        let Some(hub) = self.hub.as_ref() else {
            return;
        };
        if let Err(error) = hub.unregister(&self.run_id) {
            warn!(
                target: TARGET,
                run_id = %self.run_id,
                %error,
                "run hub unregister failed"
            );
        }
    }
}

impl Drop for RunBinding {
    /// Removes the run, so a dropped future or a panicking task cannot leak it.
    fn drop(&mut self) {
        self.unregister();
    }
}

/// Registers `spec` on `hub` and, when both are present, attaches `session`.
///
/// A `None` hub yields an inert binding whose run id is `spec.run_id` (cloned) or
/// a fresh UUID v4. A session-attach failure is logged and does not undo the
/// registration: the run stays observable, just without an interaction watcher.
#[allow(dead_code)] // the execution paths adopt this bridge next
pub(crate) fn bind_run(
    hub: Option<&Arc<InteractionHub>>,
    spec: ActiveRunSpec,
    session: Option<Arc<dyn Session>>,
) -> RunBinding {
    let Some(hub) = hub else {
        let run_id = spec
            .run_id
            .filter(|id| !id.is_empty())
            .unwrap_or_else(uuid_v4);
        return RunBinding {
            hub: None,
            run_id,
            active: AtomicBool::new(true),
        };
    };

    let handle = hub.register(spec);
    let run_id = handle.run_id().to_string();
    if let Some(session) = session {
        if let Err(error) = hub.attach_session(&run_id, session) {
            warn!(
                target: TARGET,
                run_id = %run_id,
                %error,
                "failed to attach the run session to the hub"
            );
        }
    }
    RunBinding {
        hub: Some(Arc::clone(hub)),
        run_id,
        active: AtomicBool::new(true),
    }
}

// ------------------------------------------------- session plumbing --

///
/// Detached on purpose: [`InteractionHub::cancel`] is synchronous, so the
/// responses are delivered by the runtime after it returns.
pub(super) fn spawn_cancellations(run_id: String, session: Arc<dyn Session>, ids: Vec<String>) {
    let Ok(handle) = tokio::runtime::Handle::try_current() else {
        warn!(
            target: TARGET,
            run_id = %run_id,
            "no tokio runtime: cancellation responses could not be delivered"
        );
        return;
    };
    handle.spawn(async move {
        for id in ids {
            let response = InteractionResponse {
                reason: Some(super::CANCEL_REASON.to_string()),
                ..InteractionResponse::cancel(id)
            };
            if let Err(error) = session.resolve_interaction(response).await {
                warn!(
                    target: TARGET,
                    run_id = %run_id,
                    %error,
                    "failed to deliver a cancellation response"
                );
            }
        }
    });
}

/// Records the request ids of the session's `__interaction__` frames.
///
/// Ends on the watcher token, on end of stream, or on [`StreamEvent::End`]; a
/// stream error is logged at debug and does not end the loop. The `Weak`
/// registry means a dropped or removed hub makes the task a no-op instead of
/// keeping it alive.
pub(super) async fn watch_pending(
    inner: Weak<Mutex<Inner>>,
    run_id: String,
    mut stream: BoxStream<'static, Result<StreamEvent>>,
    token: CancellationToken,
) {
    loop {
        tokio::select! {
            biased;
            () = token.cancelled() => return,
            next = stream.next() => match next {
                // The stream finished (no next frame, or the completion marker).
                None | Some(Ok(StreamEvent::End)) => return,
                Some(Ok(StreamEvent::Item(item))) => {
                    if item.item_type != INTERACTION_FRAME_TYPE {
                        continue;
                    }
                    let Some(request_id) = interaction_request_id(&item.payload) else {
                        continue;
                    };
                    let Some(inner) = inner.upgrade() else {
                        return;
                    };
                    let mut guard = inner.lock();
                    if let Some(entry) = guard.runs.get_mut(&run_id) {
                        entry.pending.insert(request_id);
                    }
                    drop(guard);
                }
                Some(Err(error)) => {
                    debug!(
                        target: TARGET,
                        run_id = %run_id,
                        %error,
                        "session stream error while watching interactions"
                    );
                }
            }
        }
    }
}

/// Reads the request id out of an `__interaction__` frame payload.
///
/// The payload is `{"id": ..., "value": ...}`; the id may sit at the top level or
/// inside `value`. A missing, null, or empty id is ignored.
fn interaction_request_id(payload: &Value) -> Option<String> {
    let nested = payload
        .get("value")
        .and_then(|value| value.get("id"))
        .and_then(Value::as_str);
    payload
        .get("id")
        .and_then(Value::as_str)
        .or(nested)
        .filter(|id| !id.is_empty())
        .map(str::to_string)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use serde_json::json;

    #[test]
    fn for_execution_sets_the_run_id_and_shares_the_token() {
        let token = CancellationToken::new();
        let spec = ActiveRunSpec::for_execution(
            "run-1",
            "assistant",
            Some("crew"),
            Some("s-1"),
            &token,
            json!({"step": 0}),
        );

        assert_eq!(spec.run_id.as_deref(), Some("run-1"));
        assert_eq!(spec.label, "assistant");
        assert_eq!(spec.team_name.as_deref(), Some("crew"));
        assert_eq!(spec.session_id.as_deref(), Some("s-1"));
        assert_eq!(spec.detail, json!({"step": 0}));

        let shared = spec.cancellation.as_ref().expect("token");
        assert!(!shared.is_cancelled());
        token.cancel();
        assert!(shared.is_cancelled(), "the spec shares the run's token");
    }

    #[test]
    fn new_spec_is_empty_apart_from_the_label() {
        let spec = ActiveRunSpec::new("assistant");
        assert!(spec.run_id.is_none());
        assert!(spec.team_name.is_none());
        assert!(spec.session_id.is_none());
        assert!(spec.cancellation.is_none());
        assert_eq!(spec.detail, Value::Null);
    }

    #[test]
    fn a_binding_without_a_hub_is_inert_and_idempotent() {
        let binding = bind_run(None, ActiveRunSpec::new("assistant"), None);
        assert_eq!(binding.run_id().len(), 36, "a fresh UUID v4 was minted");
        binding.unregister();
        binding.unregister();

        let spec = ActiveRunSpec {
            run_id: Some("run-7".to_string()),
            ..ActiveRunSpec::new("assistant")
        };
        let binding = bind_run(None, spec, None);
        assert_eq!(binding.run_id(), "run-7");
    }

    #[test]
    fn dropping_a_binding_removes_the_run() {
        let hub = Arc::new(InteractionHub::new());
        let binding = bind_run(
            Some(&hub),
            ActiveRunSpec {
                run_id: Some("run-1".to_string()),
                ..ActiveRunSpec::new("assistant")
            },
            None,
        );
        assert_eq!(hub.len(), 1);

        drop(binding);
        assert!(hub.is_empty(), "drop unregistered the run");

        // An explicit unregister before the drop is a no-op on the second call.
        let binding = bind_run(
            Some(&hub),
            ActiveRunSpec {
                run_id: Some("run-2".to_string()),
                ..ActiveRunSpec::new("assistant")
            },
            None,
        );
        binding.unregister();
        assert!(hub.is_empty());
        drop(binding);
        assert!(hub.is_empty());
    }
}
