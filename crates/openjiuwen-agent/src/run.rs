//! Session acquisition and ownership for one agent run.
use openjiuwen_common::error::Result;
use openjiuwen_session::{AgentSession, AgentSessionOptions, Session};
use std::sync::Arc;
/// The session an invocation runs on, plus whether the run owns it.
///
/// A caller-supplied session is borrowed and never closed by the run; a session
/// created by [`resolve_session`] is owned and finished when the run ends.
pub struct AgentRun {
    session: Arc<dyn Session>,
    owned: bool,
}
impl AgentRun {
    /// Wraps `session`, recording whether this run owns it.
    #[must_use]
    pub fn new(session: Arc<dyn Session>, owned: bool) -> Self {
        Self { session, owned }
    }
    /// Returns the run's session.
    #[must_use]
    pub fn session(&self) -> &Arc<dyn Session> {
        &self.session
    }
    /// Returns `true` when the run created the session and must finish it.
    #[must_use]
    pub fn is_owned(&self) -> bool {
        self.owned
    }
    /// Consumes the run and returns its session.
    #[must_use]
    pub fn into_session(self) -> Arc<dyn Session> {
        self.session
    }
    /// Ends an owned session's stream; does nothing for a borrowed session.
    ///
    /// A failing close is logged at debug level: the run's result is already
    /// determined, so closing the stream must not turn it into an error.
    pub async fn finish(self) {
        if !self.owned {
            return;
        }
        if let Err(error) = self.session.end_stream().await {
            tracing::debug!(%error, "failed to close owned agent session stream");
        }
    }
}
/// Selects the session for an invocation.
///
/// A supplied `session` is used as-is (`owned = false`). Otherwise an
/// [`AgentSession`] is created with `conversation_id` (default
/// `"default_session"`), `agent_id`, `agent_name`, and `parent_session_id`, and
/// the caller owns it (`owned = true`).
///
/// # Errors
///
/// The current implementation is infallible; the signature keeps room for
/// session factories that validate configuration.
#[allow(clippy::unused_async)]
pub async fn resolve_session(
    session: Option<Arc<dyn Session>>,
    agent_id: &str,
    agent_name: Option<&str>,
    conversation_id: Option<&str>,
    parent_session_id: Option<&str>,
) -> Result<AgentRun> {
    if let Some(session) = session {
        return Ok(AgentRun::new(session, false));
    }
    let options = AgentSessionOptions {
        session_id: Some(conversation_id.unwrap_or("default_session").to_owned()),
        agent_id: Some(agent_id.to_owned()),
        agent_name: agent_name.map(str::to_owned),
        parent_session_id: parent_session_id.map(str::to_owned),
        ..AgentSessionOptions::default()
    };
    let session: Arc<dyn Session> = Arc::new(AgentSession::with_options(options));
    Ok(AgentRun::new(session, true))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn supplied_session_is_borrowed() {
        let session: Arc<dyn Session> = Arc::new(AgentSession::new("s1"));
        let run = resolve_session(Some(session), "agent", None, Some("ignored"), None)
            .await
            .unwrap();
        assert!(!run.is_owned());
        assert_eq!(run.session().id(), "s1");
        assert_eq!(run.into_session().id(), "s1");
    }
    #[tokio::test]
    async fn missing_session_is_created_and_owned() {
        let run = resolve_session(None, "agent", Some("Agent"), None, None)
            .await
            .unwrap();
        assert!(run.is_owned());
        assert_eq!(run.session().id(), "default_session");
    }
    #[tokio::test]
    async fn conversation_id_names_the_created_session() {
        let run = resolve_session(None, "agent", None, Some("conv-9"), Some("parent-1"))
            .await
            .unwrap();
        assert!(run.is_owned());
        assert_eq!(run.session().id(), "conv-9");
        assert_eq!(run.session().parent_id(), Some("parent-1"));
        run.finish().await;
    }
}
