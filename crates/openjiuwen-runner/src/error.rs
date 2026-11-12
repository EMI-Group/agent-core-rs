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
#[must_use]
pub fn not_found(kind: &str, id: &str) -> Error {
    Error::NotFound(format!("{kind} '{id}' is not registered"))
}

/// A duplicate registration under [`crate::RegisterPolicy::Fail`]:
/// `Error::Validation` (Python `RESOURCE_ADD_ERROR`, keeping the first entry).
#[must_use]
pub fn already_registered(kind: &str, id: &str) -> Error {
    Error::Validation(format!("{kind} '{id}' is already registered"))
}

/// An empty or whitespace-only resource id: `Error::Validation`.
#[must_use]
pub fn invalid_id(kind: &str) -> Error {
    Error::Validation(format!("{kind} id cannot be empty or whitespace only"))
}

/// A resource id containing the reserved `:` separator: `Error::Validation`.
#[must_use]
pub fn reserved_id(kind: &str, id: &str) -> Error {
    Error::Validation(format!("{kind} id '{id}' must not contain ':'"))
}

/// An operation attempted on a component that is not started:
/// `Error::Config`.
#[must_use]
pub fn not_started(what: &str) -> Error {
    Error::Config(format!("{what} is not started"))
}

/// A run that exceeded its deadline: `Error::Timeout`.
#[must_use]
pub fn run_timeout(operation: &str, timeout: Duration) -> Error {
    Error::Timeout(format!(
        "{operation} timed out after {}ms",
        timeout.as_millis()
    ))
}

/// A message-queue request that exceeded its deadline: `Error::Timeout`.
#[must_use]
pub fn request_timeout(topic: &str, timeout: Duration) -> Error {
    Error::Timeout(format!(
        "message queue request on '{topic}' timed out after {}ms",
        timeout.as_millis()
    ))
}

/// A feature this build does not implement: `Error::Unsupported`.
#[must_use]
pub fn unsupported(feature: &str, hint: &str) -> Error {
    Error::Unsupported(format!("{feature} is not supported: {hint}"))
}

/// A session of the wrong kind for an operation: `Error::Validation`.
#[must_use]
pub fn invalid_session(message: impl Into<String>) -> Error {
    Error::Validation(message.into())
}

/// A transport-level failure: `Error::Transport`.
#[must_use]
pub fn transport(message: impl Into<String>) -> Error {
    Error::Transport(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use openjiuwen_common::error::ErrorKind;

    #[test]
    fn not_found_names_the_resource() {
        let error = not_found("agent", "assistant");
        assert_eq!(error.kind(), ErrorKind::NotFound);
        assert_eq!(error.to_string(), "not_found: agent 'assistant' is not registered");
    }

    #[test]
    fn duplicate_is_a_validation_error() {
        let error = already_registered("workflow", "wf");
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert_eq!(error.to_string(), "validation: workflow 'wf' is already registered");
    }

    #[test]
    fn timeout_messages_report_milliseconds() {
        let error = run_timeout("run_agent", Duration::from_millis(1500));
        assert_eq!(error.kind(), ErrorKind::Timeout);
        assert_eq!(error.to_string(), "timeout: run_agent timed out after 1500ms");

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
    fn not_started_is_a_config_error() {
        let error = not_started("message queue");
        assert_eq!(error.kind(), ErrorKind::Config);
        assert_eq!(error.to_string(), "config: message queue is not started");
    }
}
