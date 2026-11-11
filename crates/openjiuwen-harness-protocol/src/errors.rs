//! Typed errors for the harness protocol boundary.
//!
//! The variants mirror the Python exception classes in `openjiuwen.harness_protocol.errors`:
//! `HarnessProtocolError`, `HarnessStateError`, `UnsupportedHarnessCapabilityError`,
//! `CheckpointConflictError`, and the `ValueError`/`TypeError` raised from `__post_init__`.
//! The `Display` prefixes are additive; the payload reproduces the Python wording verbatim so
//! tests and provider adapters can keep matching on substrings.

/// Convenience alias for results returned by this crate.
pub type HarnessResult<T> = Result<T, HarnessError>;

/// Error raised across the third-party harness protocol boundary.
///
/// The payload is the message from the corresponding Python exception (without Python quoting).
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum HarnessError {
    /// An implementation violated a protocol invariant (Python `HarnessProtocolError`).
    #[error("harness protocol error: {0}")]
    Protocol(String),
    /// A command is invalid for the harness's current state (Python `HarnessStateError`).
    #[error("harness state error: {0}")]
    State(String),
    /// The caller requested a capability the harness did not declare
    /// (Python `UnsupportedHarnessCapabilityError`).
    #[error("unsupported harness capability: {0}")]
    UnsupportedCapability(String),
    /// A checkpoint write is stale or violates compare-and-set
    /// (Python `CheckpointConflictError`).
    #[error("checkpoint conflict: {0}")]
    CheckpointConflict(String),
    /// An invalid field value (Python `ValueError`/`TypeError` in `__post_init__`).
    #[error("invalid harness value: {0}")]
    Validation(String),
}

/// Discriminant of a [`HarnessError`], for cheap matching without inspecting the message.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum HarnessErrorKind {
    /// See [`HarnessError::Protocol`].
    Protocol,
    /// See [`HarnessError::State`].
    State,
    /// See [`HarnessError::UnsupportedCapability`].
    UnsupportedCapability,
    /// See [`HarnessError::CheckpointConflict`].
    CheckpointConflict,
    /// See [`HarnessError::Validation`].
    Validation,
}

impl HarnessError {
    /// Builds a [`HarnessError::Protocol`].
    #[must_use]
    pub fn protocol(message: impl Into<String>) -> Self {
        Self::Protocol(message.into())
    }

    /// Builds a [`HarnessError::State`].
    #[must_use]
    pub fn state(message: impl Into<String>) -> Self {
        Self::State(message.into())
    }

    /// Builds a [`HarnessError::UnsupportedCapability`].
    #[must_use]
    pub fn unsupported_capability(message: impl Into<String>) -> Self {
        Self::UnsupportedCapability(message.into())
    }

    /// Builds a [`HarnessError::CheckpointConflict`].
    #[must_use]
    pub fn checkpoint_conflict(message: impl Into<String>) -> Self {
        Self::CheckpointConflict(message.into())
    }

    /// Builds a [`HarnessError::Validation`].
    #[must_use]
    pub fn validation(message: impl Into<String>) -> Self {
        Self::Validation(message.into())
    }

    /// Returns the variant discriminant of this error.
    #[must_use]
    pub fn kind(&self) -> HarnessErrorKind {
        match self {
            Self::Protocol(_) => HarnessErrorKind::Protocol,
            Self::State(_) => HarnessErrorKind::State,
            Self::UnsupportedCapability(_) => HarnessErrorKind::UnsupportedCapability,
            Self::CheckpointConflict(_) => HarnessErrorKind::CheckpointConflict,
            Self::Validation(_) => HarnessErrorKind::Validation,
        }
    }

    /// Returns the inner message only (without the `Display` prefix).
    #[must_use]
    pub fn message(&self) -> &str {
        match self {
            Self::Protocol(message)
            | Self::State(message)
            | Self::UnsupportedCapability(message)
            | Self::CheckpointConflict(message)
            | Self::Validation(message) => message,
        }
    }
}

impl From<serde_json::Error> for HarnessError {
    fn from(error: serde_json::Error) -> Self {
        Self::Protocol(format!("invalid harness JSON: {error}"))
    }
}
