//! The runner's error vocabulary: constructors over
//! [`openjiuwen_common::error::Error`] with the wording this crate promises.
//!
//! There is no runner-local error enum: callers match on
//! [`openjiuwen_common::error::ErrorKind`] exactly as they do for the rest of the
//! workspace. The helpers exist so the wording stays identical across the
//! facade, the execution paths, the queue, and the registries (and so tests can
//! assert one source of truth).

use openjiuwen_common::error::Error;
use std::time::Duration;

/// A resource id that is not registered: `Error::NotFound`.
///
/// `kind` is the human noun (`"agent"`, `"workflow"`, `"team"`, `"tool"`).
/// Python raised an `AttributeError` from `None.invoke` instead; the Rust port
/// fails before any work starts.
pub fn not_found(kind: &str, id: &str) -> Error {
    Error::NotFound(format!("{kind} '{id}' is not registered"))
}

/// A duplicate registration under [`crate::RegisterPolicy::Fail`]:
/// `Error::Validation` (Python `RESOURCE_ADD_ERROR`, keeping the first entry).
pub fn already_registered(kind: &str, id: &str) -> Error {
    Error::Validation(format!("{kind} '{id}' is already registered"))
}

/// An empty or whitespace-only resource id: `Error::Validation`.
pub fn invalid_id(kind: &str) -> Error {
    Error::Validation(format!("{kind} id cannot be empty or whitespace only"))
}

/// A resource id containing the reserved `:` separator: `Error::Validation`.
pub fn reserved_id(kind: &str, id: &str) -> Error {
    Error::Validation(format!("{kind} id '{id}' must not contain ':'"))
}

/// An operation attempted on a component that is not started:
/// `Error::Config`.
pub fn not_started(what: &str) -> Error {
    Error::Config(format!("{what} is not started"))
}

/// A run that exceeded its deadline: `Error::Timeout`.
pub fn run_timeout(operation: &str, timeout: Duration) -> Error {
    Error::Timeout(format!(
        "{operation} timed out after {}ms",
        timeout.as_millis()
    ))
}

/// A message-queue request that exceeded its deadline: `Error::Timeout`.
pub fn request_timeout(topic: &str, timeout: Duration) -> Error {
    Error::Timeout(format!(
        "message queue request on '{topic}' timed out after {}ms",
        timeout.as_millis()
    ))
}

/// A feature this build does not implement: `Error::Unsupported`.
pub fn unsupported(feature: &str, hint: &str) -> Error {
    Error::Unsupported(format!("{feature} is not supported: {hint}"))
}

/// A distributed-configured runner reaching an entry point that would need a
/// distributed transport: `Error::Unsupported`, naming the mode and the reason.
///
/// This is the ONE message every such path reports — [`crate::Runner::start`],
/// the six execution entry points, and pool admission all go through
/// [`crate::config::RunnerConfig::require_local`], which builds it here — so a
/// caller reads the same wording wherever the refusal surfaces. The feature name
/// keeps Python's `drunner` spelling so the deferred area stays greppable.
pub fn distributed_mode() -> Error {
    unsupported(
        "distributed runner mode (`drunner`)",
        "this build ships no distributed transport (Pulsar, ZeroMQ, and A2A \
         reply-topic transports are deferred); run in-process with \
         `distributed_mode = false`",
    )
}

/// A pause or resume of a live run: `Error::Unsupported`, naming what is missing.
///
/// Nothing in this build can suspend an in-flight invocation: the in-process team
/// strategy owns no mid-flight suspension seam and Python's in-loop `InteractGate`
/// is deferred, so a pause that reported `Ok` would have paused nothing. `action`
/// is the refused verb (`"pause"` / `"resume"`) and `run_id` the run it addressed,
/// so the message says exactly which call was refused. The hint names `cancel`,
/// which really does stop a run by tripping its `CancellationToken`.
pub fn no_suspension_seam(action: &str, run_id: &str) -> Error {
    unsupported(
        &format!("`{action}` on live run '{run_id}'"),
        "this build has no mid-flight suspension seam (an in-loop `InteractGate` is \
         deferred), so nothing can suspend or restart the invocation; call `cancel` \
         (it trips the run's `CancellationToken`) when a run must stop",
    )
}

/// A session of the wrong kind for an operation: `Error::Validation`.
pub fn invalid_session(message: impl Into<String>) -> Error {
    Error::Validation(message.into())
}

/// A dreaming lifecycle call on a runner with no dreaming service installed:
/// `Error::Config`, naming the fix.
///
/// Dreaming is installed, not configured, so there is nothing to start, stop, or
/// report until the host hands over a
/// [`DreamService`](crate::dreaming::DreamService). The wording names the setter
/// so a caller reads the fix instead of a bare "not configured".
pub fn no_dreaming_service() -> Error {
    Error::Config(
        "no dreaming service is installed; call `Runner::set_dreaming_service` with a \
         `DreamService` before `start_dreaming`, `stop_dreaming`, or `dreaming_status`"
            .to_owned(),
    )
}

/// A transport-level failure: `Error::Transport`.
pub fn transport(message: impl Into<String>) -> Error {
    Error::Transport(message.into())
}

/// A shutdown that did not complete every component: `Error::Config`.
///
/// `details` is [`StopReport::failure_summary`](crate::StopReport::failure_summary),
/// so the message names each failed component (`queue: …; dreaming: …`) with its
/// own reason.
///
/// The kind is `Config` because a runner shutdown is a lifecycle step whose
/// bounds and collaborators come from [`RunnerConfig`](crate::RunnerConfig) —
/// the queue's `shutdown_timeout`, the pool's `TeamPoolConfig::stop_timeout`, and
/// the dreaming stop bound — and the failure is "the configured shutdown could
/// not be carried out": a grace period that expired, a backend that refused to
/// stop, or a dream loop that ignored its cancellation. The original failure is
/// not rewrapped into a new error kind, because the runner has no error enum of
/// its own: the typed per-component reasons stay readable in the report
/// ([`Runner::stop_with_report`](crate::Runner::stop_with_report)) and in this
/// message. Inventing a `Timeout`/`Unsupported` kind here would misreport a
/// mid-shutdown failure as a deadline or a missing capability.
pub fn shutdown_incomplete(details: impl Into<String>) -> Error {
    Error::Config(format!(
        "runner shutdown did not complete every component: {}; call `Runner::stop_with_report` \
         for the per-component report",
        details.into()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use openjiuwen_common::error::ErrorKind;

    #[test]
    fn not_found_names_the_resource() {
        let error = not_found("agent", "assistant");
        assert_eq!(error.kind(), ErrorKind::NotFound);
        assert_eq!(
            error.to_string(),
            "not_found: agent 'assistant' is not registered"
        );
    }

    #[test]
    fn duplicate_is_a_validation_error() {
        let error = already_registered("workflow", "wf");
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert_eq!(
            error.to_string(),
            "validation: workflow 'wf' is already registered"
        );
    }

    #[test]
    fn timeout_messages_report_milliseconds() {
        let error = run_timeout("run_agent", Duration::from_millis(1500));
        assert_eq!(error.kind(), ErrorKind::Timeout);
        assert_eq!(
            error.to_string(),
            "timeout: run_agent timed out after 1500ms"
        );

        let error = request_timeout("inbox", Duration::from_millis(50));
        assert_eq!(
            error.to_string(),
            "timeout: message queue request on 'inbox' timed out after 50ms"
        );
    }

    #[test]
    fn unsupported_names_the_feature_and_hint() {
        let error = unsupported("pulsar message queue", "use `in_memory`");
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        assert_eq!(
            error.to_string(),
            "unsupported: pulsar message queue is not supported: use `in_memory`"
        );
    }

    #[test]
    fn distributed_mode_names_the_mode_and_the_missing_transports() {
        let error = distributed_mode();
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        let message = error.to_string();
        for expected in [
            "distributed runner mode",
            "Pulsar",
            "ZeroMQ",
            "A2A",
            "deferred",
        ] {
            assert!(
                message.contains(expected),
                "`{message}` should name {expected}"
            );
        }
    }

    #[test]
    fn a_live_run_pause_names_the_missing_seam_and_the_alternative() {
        let error = no_suspension_seam("pause", "run-1");
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        let message = error.to_string();
        for expected in ["`pause`", "run-1", "InteractGate", "cancel", "deferred"] {
            assert!(
                message.contains(expected),
                "`{message}` should name {expected}"
            );
        }
    }

    #[test]
    fn not_started_is_a_config_error() {
        let error = not_started("message queue");
        assert_eq!(error.kind(), ErrorKind::Config);
        assert_eq!(error.to_string(), "config: message queue is not started");
    }

    #[test]
    fn a_missing_dreaming_service_names_the_setter() {
        let error = no_dreaming_service();
        assert_eq!(error.kind(), ErrorKind::Config);
        assert!(
            error.to_string().contains("Runner::set_dreaming_service"),
            "`{error}` should name the fix"
        );
        assert!(!matches!(error, Error::Unsupported(_)));
    }
}
