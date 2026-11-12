//! Session-persisted [`DeepAgentState`].
//!
//! Ports the persistence half of `openjiuwen/harness/deep_agent.py`: the
//! `DeepAgentState` payload is stored under the session key
//! [`SESSION_STATE_KEY`] and restored through
//! [`DeepAgentState::from_session_dict`]. Decoding is deliberately lenient, so
//! a payload written by an older Python checkpoint never fails an invocation;
//! only a broken session itself surfaces an error.
//!
//! [`DeepAgentState::to_session_dict`]: crate::schema::DeepAgentState::to_session_dict
//! [`DeepAgentState::from_session_dict`]: crate::schema::DeepAgentState::from_session_dict

use openjiuwen_common::error::{Error, Result};
use openjiuwen_session::Session;
use serde_json::{Map, Value};
use tracing::debug;

use crate::schema::DeepAgentState;

/// Session-state key under which the `DeepAgent` state payload is stored.
///
/// Re-exported from [`crate::schema`] so callers have one authoritative string.
pub use crate::schema::SESSION_STATE_KEY;

/// Loads the `DeepAgentState` persisted on `session`.
///
/// A missing key and a non-object payload both degrade to
/// [`DeepAgentState::default`] (the documented decode path is
/// [`DeepAgentState::from_session_dict`]). A failing session read becomes
/// [`Error::session`] naming the key; the payload itself is never logged.
///
/// # Errors
///
/// Returns [`Error::session`] when reading the session state fails.
pub async fn load_state(session: &dyn Session) -> Result<DeepAgentState> {
    let stored = session
        .get_state(SESSION_STATE_KEY)
        .await
        .map_err(|error| {
            Error::session(format!(
                "failed to read session state {SESSION_STATE_KEY:?}: {error}"
            ))
        })?;
    match stored {
        Some(value) if value.is_object() => Ok(DeepAgentState::from_session_dict(Some(&value))),
        Some(_) => {
            debug!(
                key = SESSION_STATE_KEY,
                "session state is not an object; using defaults"
            );
            Ok(DeepAgentState::default())
        }
        None => {
            debug!(
                key = SESSION_STATE_KEY,
                "session state is absent; using defaults"
            );
            Ok(DeepAgentState::default())
        }
    }
}

/// Persists `state` on `session` under [`SESSION_STATE_KEY`].
///
/// # Errors
///
/// Returns [`Error::session`] when the session write fails.
pub async fn save_state(session: &dyn Session, state: &DeepAgentState) -> Result<()> {
    let mut patch = Map::new();
    patch.insert(SESSION_STATE_KEY.to_string(), state.to_session_dict());
    session.update_state(Value::Object(patch)).await
}

/// Clears the persisted `DeepAgentState` (Python `clear_persisted=True`).
///
/// Writes `null` under [`SESSION_STATE_KEY`]; a later [`load_state`] therefore
/// yields [`DeepAgentState::default`].
///
/// # Errors
///
/// Returns [`Error::session`] when the session write fails.
pub async fn clear_state(session: &dyn Session) -> Result<()> {
    let mut patch = Map::new();
    patch.insert(SESSION_STATE_KEY.to_string(), Value::Null);
    session.update_state(Value::Object(patch)).await
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use openjiuwen_session::{AgentSession, Session};
    use serde_json::json;

    use super::{clear_state, load_state, save_state, SESSION_STATE_KEY};
    use crate::schema::{DeepAgentState, PlanModeState, TaskPlan};

    #[tokio::test]
    async fn missing_key_loads_defaults() {
        let session = AgentSession::new("state-missing");
        let state = load_state(&session).await.unwrap();
        assert_eq!(state, DeepAgentState::default());
    }

    #[tokio::test]
    async fn round_trip_keeps_every_persisted_field() {
        let session = AgentSession::new("state-round-trip");
        let state = DeepAgentState {
            iteration: 7,
            task_plan: Some(TaskPlan {
                goal: "ship it".to_string(),
                tasks: Vec::new(),
                current_task_id: None,
            }),
            stop_condition_state: None,
            pending_follow_ups: vec!["later".to_string(), "and later".to_string()],
            plan_mode: PlanModeState {
                mode: "plan".to_string(),
                pre_plan_mode: Some("normal".to_string()),
                plan_slug: Some("my-plan".to_string()),
                prompt_context: None,
            },
        };

        save_state(&session, &state).await.unwrap();
        let loaded = load_state(&session).await.unwrap();
        assert_eq!(loaded, state);
    }

    #[tokio::test]
    async fn non_object_payload_loads_defaults() {
        let session = AgentSession::new("state-non-object");
        session
            .update_state(json!({ (SESSION_STATE_KEY): 5 }))
            .await
            .unwrap();
        let state = load_state(&session).await.unwrap();
        assert_eq!(state, DeepAgentState::default());
    }

    #[tokio::test]
    async fn clear_state_restores_defaults() {
        let session = AgentSession::new("state-clear");
        let state = DeepAgentState {
            iteration: 3,
            pending_follow_ups: vec!["queued".to_string()],
            ..DeepAgentState::default()
        };
        save_state(&session, &state).await.unwrap();
        assert_eq!(load_state(&session).await.unwrap(), state);

        clear_state(&session).await.unwrap();
        assert_eq!(
            load_state(&session).await.unwrap(),
            DeepAgentState::default()
        );
    }
}
