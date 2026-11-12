//! Crate-local error type for failures raised by the multi-agent runtime.
//!
//! [`TeamError`] covers the cases specific to teams (an empty P2P response and a
//! member reporting a failure that is not an [`openjiuwen_common::error::Error`]).
//! It converts into [`openjiuwen_common::error::Error::External`], so the crate's
//! public API keeps returning the kernel error type.

use openjiuwen_common::error::Error;

/// A failure raised by the multi-agent runtime itself.
#[derive(Debug, thiserror::Error)]
pub enum TeamError {
    /// A P2P dispatch produced an empty (`Value::Null`) response.
    #[error("empty response from '{sender}' to '{recipient}'")]
    EmptyResponse {
        /// Id of the agent that sent the request.
        sender: String,
        /// Id of the agent that was expected to answer.
        recipient: String,
    },
    /// A member or strategy failed with a message that is not an
    /// `openjiuwen_common::Error`.
    #[error("team execution error: {0}")]
    Execution(String),
}

impl TeamError {
    /// Builds [`TeamError::EmptyResponse`].
    #[must_use]
    pub fn empty_response(sender: impl Into<String>, recipient: impl Into<String>) -> Self {
        Self::EmptyResponse {
            sender: sender.into(),
            recipient: recipient.into(),
        }
    }

    /// Builds [`TeamError::Execution`].
    #[must_use]
    pub fn execution(message: impl Into<String>) -> Self {
        Self::Execution(message.into())
    }
}

impl From<TeamError> for Error {
    fn from(value: TeamError) -> Self {
        Error::external(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use openjiuwen_common::error::ErrorKind;

    #[test]
    fn empty_response_display_names_both_ends() {
        let error = TeamError::empty_response("alice", "bob");
        assert_eq!(error.to_string(), "empty response from 'alice' to 'bob'");
    }

    #[test]
    fn execution_display_carries_the_message() {
        let error = TeamError::execution("boom");
        assert_eq!(error.to_string(), "team execution error: boom");
    }

    #[test]
    fn converts_to_external_kind() {
        let error: Error = TeamError::execution("member crashed").into();
        assert_eq!(error.kind(), ErrorKind::External);
        assert!(error.message().contains("member crashed"));
        assert!(error.message().contains("team execution error"));
    }

    #[test]
    fn empty_response_conversion_keeps_the_text() {
        let error: Error = TeamError::empty_response("a", "b").into();
        assert_eq!(error.kind(), ErrorKind::External);
        assert!(error.message().contains("empty response from 'a' to 'b'"));
    }
}
