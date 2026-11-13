//! The pool's error vocabulary and its shutdown summary.
//!
//! [`PoolError`] is the pool-local failure classification; it converts into the
//! crate-wide [`openjiuwen_common::error::Error`] at the public boundary, so a
//! caller matches on [`ErrorKind`](openjiuwen_common::error::ErrorKind) exactly
//! as it does for the rest of the workspace. The variants mirror Python's
//! `RunActionKind` rejections (`team_runner.py::_TEAM_REJECT_KINDS`):
//!
//! | `PoolError` | `ErrorKind` | Python |
//! | --- | --- | --- |
//! | `AtCapacity` | `Config` | `REJECT_INCONSISTENT` (no runtime slot) |
//! | `Busy` | `Validation` | `REJECT_RUNNING` |
//! | `NotFound` | `NotFound` | `REJECT_ORPHANED` |
//! | `Closed` | `Config` | `REJECT_INCONSISTENT` (pool shut down) |
//! | `Cancelled` | `Cancelled` | (the run's token fired) |

use openjiuwen_common::error::{Error, ErrorKind};

use super::key::PoolKey;

/// Why a pool operation could not be carried out.
///
/// Every variant names the key (or the run) it refers to, so a host can log the
/// failure without re-deriving the identity.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PoolError {
    /// The pool already holds `capacity` runs in flight.
    #[error("team pool is at capacity ({capacity} concurrent runs)")]
    AtCapacity {
        /// The effective concurrency bound.
        capacity: usize,
    },
    /// A run is already live for the key.
    #[error("team run for '{team_name}' / session '{session_id}' is already active")]
    Busy {
        /// Team name of the live run.
        team_name: String,
        /// Session id of the live run.
        session_id: String,
    },
    /// No live (non-terminal) run is pooled for the key.
    #[error("no active pooled team run for '{team_name}' / session '{session_id}'")]
    NotFound {
        /// Team name that was asked for.
        team_name: String,
        /// Session id that was asked for.
        session_id: String,
    },
    /// The pool was shut down with
    /// [`TeamPool::stop_all`](crate::pool::TeamPool::stop_all).
    #[error("team pool is closed")]
    Closed,
    /// The run's cancellation token fired before the operation could finish.
    #[error("team pool run '{run_id}' was cancelled")]
    Cancelled {
        /// The run id that was cancelled.
        run_id: String,
    },
}

impl PoolError {
    /// The error a key-scoped lookup reports.
    #[must_use]
    pub fn not_found(key: &PoolKey) -> Self {
        Self::NotFound {
            team_name: key.team_name.clone(),
            session_id: key.session_id.clone(),
        }
    }

    /// The error a duplicate admission reports.
    #[must_use]
    pub fn busy(key: &PoolKey) -> Self {
        Self::Busy {
            team_name: key.team_name.clone(),
            session_id: key.session_id.clone(),
        }
    }

    /// The crate-wide category this failure maps to.
    pub const fn kind(&self) -> ErrorKind {
        match self {
            Self::AtCapacity { .. } | Self::Closed => ErrorKind::Config,
            Self::Busy { .. } => ErrorKind::Validation,
            Self::NotFound { .. } => ErrorKind::NotFound,
            Self::Cancelled { .. } => ErrorKind::Cancelled,
        }
    }

    /// Converts into the crate-wide error type, keeping the documented wording.
    pub fn into_error(self) -> Error {
        let message = self.to_string();
        match self {
            Self::AtCapacity { .. } | Self::Closed => Error::config(message),
            Self::Busy { .. } => Error::validation(message),
            Self::NotFound { .. } => Error::not_found(message),
            Self::Cancelled { .. } => Error::cancelled(),
        }
    }
}

impl From<PoolError> for Error {
    /// Converts through [`PoolError::into_error`].
    fn from(error: PoolError) -> Self {
        error.into_error()
    }
}

/// Rebuilds an [`Error`] from its kind and message.
///
/// [`Error`] is not `Clone` (its `External` variant boxes a foreign error), so
/// the pool stores the plain `(kind, message)` pair and rebuilds the error
/// whenever a caller needs one. A kind without a pool-side constructor falls
/// back to [`ErrorKind::Validation`], which is why `Validation` and `External`
/// share one arm.
pub(crate) fn rebuild(kind: ErrorKind, message: impl Into<String>) -> Error {
    let message = message.into();
    match kind {
        ErrorKind::Config => Error::config(message),
        ErrorKind::Validation | ErrorKind::External => Error::validation(message),
        ErrorKind::Llm => Error::llm(message),
        ErrorKind::Tool => Error::tool(message),
        ErrorKind::Workflow => Error::workflow(message),
        ErrorKind::Graph => Error::graph(message),
        ErrorKind::Session => Error::session(message),
        ErrorKind::Memory => Error::memory(message),
        ErrorKind::Retrieval => Error::retrieval(message),
        ErrorKind::SysOp => Error::sys_op(message),
        ErrorKind::Security => Error::security(message),
        ErrorKind::Mcp => Error::mcp(message),
        ErrorKind::Transport => Error::transport(message),
        ErrorKind::Timeout => Error::timeout(message),
        ErrorKind::NotFound => Error::not_found(message),
        ErrorKind::Unsupported => Error::unsupported(message),
        ErrorKind::Cancelled => Error::cancelled(),
    }
}

/// The summary [`TeamPool::stop_all`](crate::pool::TeamPool::stop_all) returns.
///
/// `stopped` counts the runs that were live and are now stopped; `first_error`
/// keeps the first failure (the rest are logged), so a shutdown sweep reports
/// one error rather than losing the others silently.
#[derive(Debug, Default)]
pub struct PoolShutdown {
    /// How many live runs were stopped.
    pub stopped: usize,
    /// The first failure of the sweep, when any.
    pub first_error: Option<Error>,
}

impl PoolShutdown {
    /// `true` when every run stopped without an error.
    #[must_use]
    pub fn is_ok(&self) -> bool {
        self.first_error.is_none()
    }

    /// Returns how many runs were stopped, or the first failure.
    ///
    /// # Errors
    ///
    /// The first failure [`PoolShutdown::first_error`], when the sweep reported
    /// one.
    pub fn into_result(self) -> openjiuwen_common::error::Result<usize> {
        match self.first_error {
            Some(error) => Err(error),
            None => Ok(self.stopped),
        }
    }
}

impl Clone for PoolShutdown {
    /// Clones the failure through `rebuild`, because [`Error`] is not
    /// `Clone`.
    fn clone(&self) -> Self {
        Self {
            stopped: self.stopped,
            first_error: self
                .first_error
                .as_ref()
                .map(|error| rebuild(error.kind(), error.message())),
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn kinds_and_messages_map_as_documented() {
        let key = PoolKey::new("crew", "s-1");
        let cases = [
            (
                PoolError::AtCapacity { capacity: 8 },
                ErrorKind::Config,
                "config: team pool is at capacity (8 concurrent runs)",
            ),
            (
                PoolError::busy(&key),
                ErrorKind::Validation,
                "validation: team run for 'crew' / session 's-1' is already active",
            ),
            (
                PoolError::not_found(&key),
                ErrorKind::NotFound,
                "not_found: no active pooled team run for 'crew' / session 's-1'",
            ),
            (
                PoolError::Closed,
                ErrorKind::Config,
                "config: team pool is closed",
            ),
            (
                PoolError::Cancelled {
                    run_id: "run-1".to_string(),
                },
                ErrorKind::Cancelled,
                "cancelled",
            ),
        ];
        for (pool_error, kind, message) in cases {
            assert_eq!(pool_error.kind(), kind);
            let error = pool_error.clone().into_error();
            assert_eq!(error.kind(), kind);
            assert_eq!(error.to_string(), message);
            let converted: Error = pool_error.into();
            assert_eq!(converted.kind(), kind);
        }
    }

    #[test]
    fn rebuilding_preserves_the_kind_and_the_message() {
        let error = rebuild(ErrorKind::Timeout, "run_agent_team timed out after 5ms");
        assert_eq!(error.kind(), ErrorKind::Timeout);
        assert_eq!(
            error.to_string(),
            "timeout: run_agent_team timed out after 5ms"
        );
        assert_eq!(
            rebuild(ErrorKind::External, "foreign").kind(),
            ErrorKind::Validation,
            "a kind without a constructor falls back to validation"
        );
    }

    #[test]
    fn the_shutdown_summary_clones_and_reports() {
        let summary = PoolShutdown::default();
        assert!(summary.is_ok());
        assert_eq!(summary.clone().into_result().unwrap(), 0);

        let summary = PoolShutdown {
            stopped: 3,
            first_error: None,
        };
        assert!(summary.is_ok());
        assert_eq!(summary.into_result().unwrap(), 3);

        let summary = PoolShutdown {
            stopped: 1,
            first_error: Some(Error::validation("boom")),
        };
        assert!(!summary.is_ok());
        let clone = summary.clone();
        assert_eq!(clone.stopped, 1);
        let error = clone.into_result().unwrap_err();
        assert_eq!(error.to_string(), "validation: boom");
        let debug = format!("{summary:?}");
        assert!(debug.contains("stopped: 1"), "Debug is printable: {debug}");
    }
}
