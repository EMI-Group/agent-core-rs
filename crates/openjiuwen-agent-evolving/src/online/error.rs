//! Typed errors of the online RL task state machine.
//!
//! Every failure is one of the domain errors Python raises from
//! `agent_rl/online/task_registry.py` (`TaskNotFoundError`, `TaskConflictError`,
//! `TurnClosedError`) or from the capture pipeline's validation
//! (`ValueError`), plus the sample-budget error the training runner needs and a
//! carrier for storage failures.
//!
//! [`OnlineError::code`] and [`OnlineError::http_status`] are the wire contract
//! of the RL service: the HTTP layer maps them straight into Python's error
//! envelope `{"error": {"code": ..., "message": ...}}`.

use openjiuwen_common::error::Error;

/// Result type for online-RL state machine operations.
///
/// The module deliberately uses [`OnlineError`] rather than
/// [`openjiuwen_common::error::Error`] so callers keep the machine-readable
/// [`OnlineError::code`]; [`Error`] conversion is one `From` call away.
pub type Result<T, E = OnlineError> = std::result::Result<T, E>;

/// A failure of the RL task/capture state machine.
#[derive(Debug, thiserror::Error)]
pub enum OnlineError {
    /// The requested task does not exist.
    #[error("{0}")]
    TaskNotFound(String),
    /// The requested operation conflicts with durable state.
    #[error("{0}")]
    TaskConflict(String),
    /// A delayed-feedback capture arrived after its turn was closed.
    #[error("{0}")]
    TurnClosed(String),
    /// A request payload is malformed or inconsistent.
    #[error("{0}")]
    InvalidRequest(String),
    /// A training request has fewer samples than it requires.
    #[error("insufficient samples: have {have}, need {need}")]
    InsufficientSamples {
        /// Samples available.
        have: usize,
        /// Samples required.
        need: usize,
    },
    /// The backing store failed; the originating error is kept as the source.
    #[error("store: {0}")]
    Store(#[source] Error),
}

impl OnlineError {
    /// The machine-readable error code of the service envelope.
    #[must_use]
    pub fn code(&self) -> &'static str {
        match self {
            Self::TaskNotFound(_) => "task_not_found",
            Self::TaskConflict(_) => "task_conflict",
            Self::TurnClosed(_) => "turn_closed",
            Self::InvalidRequest(_) => "invalid_request",
            Self::InsufficientSamples { .. } => "insufficient_samples",
            Self::Store(_) => "internal_error",
        }
    }

    /// The HTTP status the service answers with.
    #[must_use]
    pub fn http_status(&self) -> u16 {
        match self {
            Self::TaskNotFound(_) => 404,
            Self::TaskConflict(_) | Self::TurnClosed(_) | Self::InsufficientSamples { .. } => 409,
            Self::InvalidRequest(_) => 400,
            Self::Store(_) => 500,
        }
    }

    /// Wraps a store failure.
    pub(crate) fn store(error: Error) -> Self {
        Self::Store(error)
    }
}

impl From<OnlineError> for Error {
    fn from(error: OnlineError) -> Self {
        match error {
            OnlineError::TaskNotFound(message) => Self::not_found(message),
            OnlineError::InvalidRequest(message) => Self::validation(message),
            // A store failure already is a kernel error; keep it whole so its
            // `source` chain and variant survive the conversion.
            OnlineError::Store(error) => error,
            other => Self::workflow(other.to_string()),
        }
    }
}
