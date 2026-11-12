//! Team sessions, the lease a strategy run holds, and the team stream driver.
//!
//! Python parity: `openjiuwen/core/multi_agent/team_session.py` (a team run
//! creates one [`AgentTeamSession`], binds it to the runtime so bus deliveries
//! run in its child sessions, and tears it down when the run finishes) and the
//! streaming helpers in `openjiuwen/core/multi_agent/base_team.py`.
//!
//! The Rust split is three small pieces:
//!
//! - [`make_team_session`] builds the session of one invocation, taking its id
//!   from the message's `conversation_id` when present;
//! - [`SessionLease`] records whether the run owns that session (and therefore
//!   must unbind and tear it down) or merely borrows a caller-provided one;
//! - [`team_stream`] drives a strategy run into a stream of wire frames while a
//!   spawned task owns the session and reports the run's outcome.
//!
//! The run task is never cancelled: a consumer that stops reading simply drops
//! the stream, the task still finishes and tears its session down (documented
//! Python parity, where the producer coroutine is independent of the consumer).

use std::sync::Arc;

use async_stream::stream;
use futures::future::BoxFuture;
use futures::stream::BoxStream;
use futures::StreamExt;
use serde_json::Value;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::utils::id::uuid_v4;
use openjiuwen_session::{
    create_agent_team_session, AgentTeamSession, AgentTeamSessionOptions, Session, StreamEvent,
};

use crate::card::TeamCard;
use crate::runtime::TeamRuntime;

/// Returns the non-empty `conversation_id` of `message`, when it has one.
///
/// Python parity: the team run reuses the caller's conversation as the session
/// id when it is a string, and mints a fresh uuid otherwise (an empty string or
/// a non-string value is treated as absent).
fn conversation_id(message: &Value) -> Option<String> {
    message
        .get("conversation_id")
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty())
        .map(ToOwned::to_owned)
}

/// Builds the team session for a team invocation.
///
/// The session id is `message["conversation_id"]` when that is a non-empty
/// string, else a fresh uuid v4. The session is keyed to `card.id`, so child
/// agent sessions are stamped with the team they belong to.
#[must_use]
pub fn make_team_session(card: &TeamCard, message: &Value) -> Arc<AgentTeamSession> {
    let session_id = conversation_id(message).unwrap_or_else(uuid_v4);
    Arc::new(create_agent_team_session(AgentTeamSessionOptions {
        session_id: Some(session_id),
        team_id: Some(card.id.clone()),
        ..AgentTeamSessionOptions::default()
    }))
}

/// Ownership wrapper around the team session a strategy run uses.
///
/// A lease either owns the session it created (in which case it bound the
/// session to the runtime and must unbind and tear it down) or borrows a
/// caller-provided session (in which case it only closes the stream).
pub struct SessionLease {
    /// Runtime the owned session is bound to.
    runtime: Arc<TeamRuntime>,
    /// Session the run writes its frames to.
    session: Arc<AgentTeamSession>,
    /// `true` when this lease created the session.
    owned: bool,
}

impl std::fmt::Debug for SessionLease {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SessionLease")
            .field("session_id", &self.session.id())
            .field("owned", &self.owned)
            .finish_non_exhaustive()
    }
}

impl SessionLease {
    /// Takes the lease for one strategy run.
    ///
    /// A caller-provided `session` is used as-is and left unbound (Python only
    /// binds a session it creates). Without one, a session is built with
    /// [`make_team_session`], bound to `runtime`, and marked owned.
    // `async` matches the Python API and leaves room for asynchronous
    // session construction by future strategies.
    #[allow(clippy::unused_async, clippy::unused_async_trait_impl)]
    pub async fn acquire(
        runtime: &Arc<TeamRuntime>,
        card: &TeamCard,
        message: &Value,
        session: Option<Arc<AgentTeamSession>>,
    ) -> Self {
        if let Some(session) = session {
            return Self {
                runtime: Arc::clone(runtime),
                session,
                owned: false,
            };
        }
        let session = make_team_session(card, message);
        runtime.bind_team_session(Arc::clone(&session));
        Self {
            runtime: Arc::clone(runtime),
            session,
            owned: true,
        }
    }

    /// Returns the leased session.
    #[must_use]
    pub fn session(&self) -> &Arc<AgentTeamSession> {
        &self.session
    }

    /// Returns the id of the leased session.
    #[must_use]
    pub fn session_id(&self) -> &str {
        self.session.id()
    }

    /// Returns `true` when the lease created the session it holds.
    #[must_use]
    pub fn is_standalone(&self) -> bool {
        self.owned
    }

    /// Tears the run down according to the lease's ownership.
    ///
    /// An owned session is unbound from the runtime, its transport topics are
    /// cleaned up, its stream is ended, and its state is committed — in that
    /// order, so no delivery can arrive in a half-torn-down session. A
    /// caller-owned session is only ended: the caller keeps it bound and
    /// commits it itself.
    ///
    /// Every teardown step is attempted even when an earlier one fails, and the
    /// first error is returned at the end. This is a deliberate hardening over
    /// Python's unguarded `finally`: one failing step must not leave the
    /// remaining cleanup (notably the stream end marker) unexecuted.
    ///
    /// # Errors
    ///
    /// Returns the first failure of `end_stream` or `commit`; a caller-owned
    /// session's `end_stream` error is returned directly. Unbinding and topic
    /// cleanup are infallible.
    pub async fn finish(self) -> Result<()> {
        let Self {
            runtime,
            session,
            owned,
        } = self;

        if !owned {
            return session.end_stream().await;
        }

        let session_id = session.id().to_owned();
        runtime.unbind_team_session(&session_id);
        runtime.cleanup_session(&session_id).await;

        let mut first_error: Option<Error> = None;
        if let Err(error) = session.end_stream().await {
            first_error = Some(error);
        }
        // `commit` returns the checkpoint id when a checkpointer is configured;
        // the team stream does not surface it, like Python's teardown.
        if let Err(error) = session.commit().await {
            if first_error.is_none() {
                first_error = Some(error);
            }
        }
        first_error.map_or(Ok(()), Err)
    }
}

/// Drives a strategy run into a team stream.
///
/// The stream yields every frame the run writes to its session as the wire
/// projection of the frame (`{"type": …, "index"?: …, "payload": …}`), in
/// producer order. Subscription happens before the run task is spawned, so no
/// frame can be missed; the run and the teardown both finish before the stream
/// ends, so a consumer that drains it observes a fully torn-down session.
///
/// The spawned task is never cancelled: if the consumer stops early, the task
/// still runs `run` to completion and tears the session down. The `JoinHandle`
/// is dropped deliberately — the oneshot channel already reports completion and
/// the task must outlive an early-dropped consumer.
///
/// A run failure is surfaced after the frames and after the teardown, so a
/// consumer sees the scripted output first and the error last.
///
/// A panic inside `run` is not propagated: the task drops its reporting
/// channel, so the stream ends with
/// `Error::session("team stream task ended unexpectedly")` after the frames
/// already published.
pub fn team_stream<F>(
    runtime: Arc<TeamRuntime>,
    card: TeamCard,
    message: Value,
    session: Option<Arc<AgentTeamSession>>,
    run: F,
) -> BoxStream<'static, Result<Value>>
where
    F: FnOnce(Arc<AgentTeamSession>) -> BoxFuture<'static, Result<()>> + Send + 'static,
{
    Box::pin(stream! {
        let lease = SessionLease::acquire(&runtime, &card, &message, session).await;
        // Subscribe before spawning the producer: a frame written by the very
        // first statement of `run` must not be dropped.
        let mut events = lease.session().subscribe();

        let (tx, rx) = tokio::sync::oneshot::channel::<Result<()>>();
        tokio::spawn(async move {
            let run_result = run(Arc::clone(lease.session())).await;
            let finish_result = lease.finish().await;
            let combined = match run_result {
                Err(error) => Err(error),
                Ok(()) => finish_result,
            };
            let _ = tx.send(combined);
        });

        while let Some(event) = events.next().await {
            match event {
                Ok(StreamEvent::Item(item)) => yield Ok(item.to_wire()),
                Ok(StreamEvent::End) => break,
                Err(error) => {
                    yield Err(error);
                    break;
                }
            }
        }

        match rx.await {
            Ok(Err(error)) => yield Err(error),
            Ok(Ok(())) => {}
            // The sender is dropped without a value only when the run task
            // panicked: report it instead of ending the stream silently.
            Err(_) => {
                tracing::warn!("team stream task ended without reporting");
                yield Err(Error::session("team stream task ended unexpectedly"));
            }
        }
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use serde_json::json;
    use uuid::Uuid;

    use super::*;

    #[test]
    fn conversation_id_requires_a_non_empty_string() {
        assert_eq!(
            conversation_id(&json!({"conversation_id": "abc"})),
            Some("abc".to_owned())
        );
        assert_eq!(conversation_id(&json!({"conversation_id": ""})), None);
        assert_eq!(conversation_id(&json!({"conversation_id": 7})), None);
        assert_eq!(conversation_id(&json!({})), None);
        assert_eq!(conversation_id(&Value::Null), None);
    }

    #[test]
    fn team_session_id_falls_back_to_a_uuid() {
        let card = TeamCard::new("test");
        let session = make_team_session(&card, &json!({}));
        assert_ne!(session.id(), "");
        assert!(Uuid::parse_str(session.id()).is_ok());

        let named = make_team_session(&card, &json!({"conversation_id": "abc"}));
        assert_eq!(named.id(), "abc");
        assert_eq!(named.team_id(), card.id);
    }
}
