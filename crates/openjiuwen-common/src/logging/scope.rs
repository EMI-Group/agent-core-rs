//! Logging scope: trace/member/session identifiers attached to log records.
//!
//! Port of the `contextvars`-based helpers in Python
//! `openjiuwen/core/common/logging/utils.py`. Python stored the identifiers in
//! `ContextVar`s so every log record picked them up automatically; the Rust SDK
//! replaces them with a tokio task-local [`LogScope`] (for lookups) and a
//! `tracing` span (for structured fields), which gives the same user-visible
//! behavior plus structured output:
//!
//! - [`with_scope`] runs a future with a scope installed for its whole subtree;
//! - [`current`] / [`current_trace_id`] / [`current_member_id`] read it back;
//! - [`LogScope::instrument`] installs the scope *and* opens the matching span.
use super::targets;
use crate::utils::id::uuid_v4;
use std::future::Future;
use std::sync::Arc;

/// Trace id used when no [`LogScope`] is active.
///
/// Python `get_trace_id` default, kept verbatim so log parsers keep working.
pub const DEFAULT_TRACE_ID: &str = "default_trace_id";

/// Correlation identifiers propagated through an invocation tree.
///
/// Every clone of a scope shares the same identifiers; nested invocations create
/// a new scope (typically with a fresh [`LogScope::new`] trace id) and the outer
/// scope is restored when the inner future completes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LogScope {
    /// Trace id shared by every log record of one logical request.
    pub trace_id: String,
    /// Member id inside a multi-agent team (empty for single agents).
    pub member_id: String,
    /// Session the work belongs to.
    pub session_id: Option<String>,
    /// Single invocation (turn) inside the session.
    pub invocation_id: Option<String>,
    /// Agent that produced the record.
    pub agent_id: Option<String>,
    /// Conversation id, when it differs from the session id.
    pub conversation_id: Option<String>,
}

impl LogScope {
    /// Creates a scope with a fresh trace id and an empty member id.
    #[must_use]
    pub fn new() -> Self {
        Self {
            trace_id: uuid_v4(),
            member_id: String::new(),
            session_id: None,
            invocation_id: None,
            agent_id: None,
            conversation_id: None,
        }
    }
    /// Overrides the trace id.
    #[must_use]
    pub fn with_trace_id(mut self, trace_id: impl Into<String>) -> Self {
        self.trace_id = trace_id.into();
        self
    }
    /// Overrides the team member id.
    #[must_use]
    pub fn with_member_id(mut self, member_id: impl Into<String>) -> Self {
        self.member_id = member_id.into();
        self
    }
    /// Sets the session id.
    #[must_use]
    pub fn with_session_id(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }
    /// Sets the invocation id.
    #[must_use]
    pub fn with_invocation_id(mut self, invocation_id: impl Into<String>) -> Self {
        self.invocation_id = Some(invocation_id.into());
        self
    }
    /// Sets the agent id.
    #[must_use]
    pub fn with_agent_id(mut self, agent_id: impl Into<String>) -> Self {
        self.agent_id = Some(agent_id.into());
        self
    }
    /// Sets the conversation id.
    #[must_use]
    pub fn with_conversation_id(mut self, conversation_id: impl Into<String>) -> Self {
        self.conversation_id = Some(conversation_id.into());
        self
    }
    /// Opens a `tracing` span carrying every scope field.
    ///
    /// Absent optional identifiers are recorded as empty strings so the field
    /// set (and therefore downstream log schemas) stays stable.
    #[must_use]
    pub fn span(&self) -> tracing::Span {
        tracing::info_span!(
            target: targets::COMMON,
            "openjiuwen_scope",
            trace_id = %self.trace_id,
            member_id = %self.member_id,
            session_id = self.session_id.as_deref().unwrap_or(""),
            invocation_id = self.invocation_id.as_deref().unwrap_or(""),
            agent_id = self.agent_id.as_deref().unwrap_or(""),
            conversation_id = self.conversation_id.as_deref().unwrap_or(""),
        )
    }
    /// Runs `fut` with the scope active, both as a span and as task-local state.
    ///
    /// The runtime scope is restored when the returned future completes, so an
    /// inner invocation can never leak its identifiers into its parent.
    pub fn instrument<F>(self: Arc<Self>, fut: F) -> impl Future<Output = F::Output>
    where
        F: Future,
    {
        use tracing::Instrument as _;
        let span = self.span();
        async move { with_scope(self, fut.instrument(span)).await }
    }
}

impl Default for LogScope {
    fn default() -> Self {
        Self::new()
    }
}

tokio::task_local! {
    static CURRENT_SCOPE: Arc<LogScope>;
}

/// Runs `fut` with `scope` installed as the current task-local [`LogScope`].
///
/// The scope is visible to every nested `.await` on the same task, and the
/// previous scope is restored when `fut` completes.
pub async fn with_scope<F: Future>(scope: Arc<LogScope>, fut: F) -> F::Output {
    CURRENT_SCOPE.scope(scope, fut).await
}

/// Returns the scope active on the current task, or `None` outside a scope.
#[must_use]
pub fn current() -> Option<Arc<LogScope>> {
    CURRENT_SCOPE.try_with(Arc::clone).ok()
}

/// Returns the active trace id, or [`DEFAULT_TRACE_ID`] outside a scope.
#[must_use]
pub fn current_trace_id() -> String {
    current().map_or_else(
        || DEFAULT_TRACE_ID.to_owned(),
        |scope| scope.trace_id.clone(),
    )
}

/// Returns the active member id, or an empty string outside a scope.
#[must_use]
pub fn current_member_id() -> String {
    current().map_or_else(String::new, |scope| scope.member_id.clone())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    // port of tests/unit_tests/core/common/log/test_logger.py (trace id isolation)
    #[tokio::test]
    async fn defaults_apply_outside_a_scope() {
        assert!(current().is_none());
        assert_eq!(current_trace_id(), DEFAULT_TRACE_ID);
        assert_eq!(current_member_id(), "");
        assert_eq!(DEFAULT_TRACE_ID, "default_trace_id");
    }
    #[tokio::test]
    async fn with_scope_exposes_the_scope_inside_the_future() {
        let scope = Arc::new(
            LogScope::new()
                .with_member_id("teammate-1")
                .with_session_id("session-1"),
        );
        let trace_id = with_scope(Arc::clone(&scope), async {
            assert_eq!(current().unwrap().session_id.as_deref(), Some("session-1"));
            assert_eq!(current_member_id(), "teammate-1");
            current_trace_id()
        })
        .await;
        assert_eq!(trace_id, scope.trace_id);
        assert!(current().is_none(), "scope must not leak past the future");
    }
    #[tokio::test]
    async fn nested_scopes_override_and_restore_the_outer_scope() {
        let outer = Arc::new(LogScope::new().with_trace_id("outer"));
        let inner = Arc::new(LogScope::new().with_trace_id("inner"));
        with_scope(Arc::clone(&outer), async move {
            assert_eq!(current_trace_id(), "outer");
            with_scope(Arc::clone(&inner), async {
                assert_eq!(current_trace_id(), "inner");
            })
            .await;
            assert_eq!(
                current_trace_id(),
                "outer",
                "the outer scope must be restored"
            );
        })
        .await;
    }
    #[tokio::test]
    async fn new_scopes_have_distinct_trace_ids() {
        assert_ne!(LogScope::new().trace_id, LogScope::new().trace_id);
        assert_eq!(LogScope::default().member_id, "");
    }
    #[tokio::test]
    async fn instrument_installs_the_scope_for_the_future() {
        let scope = Arc::new(LogScope::new().with_trace_id("instrumented"));
        let (trace_id, member_id) = scope
            .instrument(async { (current_trace_id(), current_member_id()) })
            .await;
        assert_eq!(trace_id, "instrumented");
        assert_eq!(member_id, "");
        assert!(current().is_none());
    }
    #[test]
    fn span_records_every_field() {
        let scope = LogScope::new()
            .with_trace_id("trace")
            .with_member_id("member")
            .with_session_id("session")
            .with_invocation_id("invocation")
            .with_agent_id("agent")
            .with_conversation_id("conversation");
        // A thread-local subscriber makes the callsite interesting without
        // touching the process-global subscriber.
        tracing::subscriber::with_default(tracing_subscriber::Registry::default(), || {
            let span = scope.span();
            let metadata = span.metadata().unwrap();
            assert_eq!(metadata.name(), "openjiuwen_scope");
            assert_eq!(metadata.target(), targets::COMMON);
            assert!(metadata.is_span());
            let field_names: Vec<&str> =
                metadata.fields().iter().map(|field| field.name()).collect();
            for expected in [
                "trace_id",
                "member_id",
                "session_id",
                "invocation_id",
                "agent_id",
                "conversation_id",
            ] {
                assert!(field_names.contains(&expected), "missing field {expected}");
            }
        });
    }
}
