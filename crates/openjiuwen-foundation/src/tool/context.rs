//! Per-invocation tool context: identity, cancellation, session, and deadline.
//!
//! Ports the execution-scoped part of `openjiuwen/core/foundation/tool/base.py`
//! (`invoke(inputs, **kwargs)`), where the keyword arguments carry the session,
//! cancellation, and timeout a tool needs while it runs.
//!
//! [`ToolContext`] is runtime-only: it deliberately does **not** implement
//! `serde`. A [`CancellationToken`] has no wire form, and the Python context is
//! never serialized either — it exists only for the duration of a call, so a
//! JSON representation would be lossy and misleading.

use std::collections::BTreeMap;
use std::time::Duration;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::utils::id::uuid_hex;
use serde_json::Value;
use tokio_util::sync::CancellationToken;

/// The runtime context handed to a tool invocation.
///
/// Every invocation gets a fresh [`ToolContext`] (and therefore a fresh
/// [`ToolContext::invocation_id`]); tools may clone it to move cancellation into
/// spawned work, since all clones share the same [`CancellationToken`].
#[derive(Debug, Clone)]
pub struct ToolContext {
    /// Session the call belongs to, when the caller is session-scoped.
    pub session_id: Option<String>,
    /// Unique identifier of this invocation (Python `uuid.uuid4().hex` form).
    pub invocation_id: String,
    /// Cooperative cancellation signal for this invocation.
    pub cancellation: CancellationToken,
    /// Free-form per-call metadata.
    pub metadata: BTreeMap<String, Value>,
    /// Optional per-call deadline.
    pub timeout: Option<Duration>,
}

impl ToolContext {
    /// Creates a context with a fresh invocation id and no session, metadata, or
    /// timeout.
    #[must_use]
    pub fn new() -> Self {
        Self {
            session_id: None,
            invocation_id: uuid_hex(),
            cancellation: CancellationToken::new(),
            metadata: BTreeMap::new(),
            timeout: None,
        }
    }

    /// Returns this context bound to `session_id`.
    #[must_use]
    pub fn with_session(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }

    /// Returns this context with `metadata`, replacing the previous map.
    #[must_use]
    pub fn with_metadata(mut self, metadata: BTreeMap<String, Value>) -> Self {
        self.metadata = metadata;
        self
    }

    /// Returns this context with one additional metadata entry.
    #[must_use]
    pub fn with_metadata_entry(mut self, key: impl Into<String>, value: Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Returns this context with the given deadline.
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Returns `true` once the cancellation token has been cancelled.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.cancellation.is_cancelled()
    }

    /// Returns [`Error::Cancelled`] when the token has been cancelled.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Cancelled`] if [`ToolContext::cancellation`] is
    /// cancelled; otherwise `Ok(())`.
    pub fn check_cancelled(&self) -> Result<()> {
        if self.is_cancelled() {
            return Err(Error::Cancelled);
        }
        Ok(())
    }
}

impl Default for ToolContext {
    /// Same as [`ToolContext::new`]: a fresh invocation id and no extra state.
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use openjiuwen_common::error::ErrorKind;
    use serde_json::json;

    #[test]
    fn contexts_get_distinct_hex_invocation_ids() {
        let first = ToolContext::new();
        let second = ToolContext::new();

        assert_eq!(first.invocation_id.len(), 32);
        assert!(first
            .invocation_id
            .chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
        assert_ne!(first.invocation_id, second.invocation_id);
        assert_eq!(ToolContext::default().invocation_id.len(), 32);
        assert_eq!(first.session_id, None);
        assert!(first.metadata.is_empty());
        assert_eq!(first.timeout, None);
    }

    #[test]
    fn builders_attach_session_metadata_and_timeout() {
        let mut metadata = BTreeMap::new();
        metadata.insert("trace".to_owned(), json!("abc"));

        let ctx = ToolContext::new()
            .with_session("session-1")
            .with_metadata(metadata)
            .with_metadata_entry("attempt", json!(2))
            .with_timeout(Duration::from_secs(5));

        assert_eq!(ctx.session_id.as_deref(), Some("session-1"));
        assert_eq!(ctx.metadata["trace"], json!("abc"));
        assert_eq!(ctx.metadata["attempt"], json!(2));
        assert_eq!(ctx.timeout, Some(Duration::from_secs(5)));
    }

    #[test]
    fn cancellation_is_shared_between_clones() {
        let ctx = ToolContext::new();
        let clone = ctx.clone();

        assert!(!ctx.is_cancelled());
        assert!(ctx.check_cancelled().is_ok());

        clone.cancellation.cancel();

        assert!(ctx.is_cancelled());
        let error = ctx.check_cancelled().unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Cancelled);
        assert!(matches!(error, Error::Cancelled));
    }
}
