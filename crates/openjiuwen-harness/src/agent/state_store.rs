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
/// Python's `update_dict` REPLACES a mapping value wholesale, so its explicit
/// `None`s survive every save. The Rust session store instead deep-merges
/// objects and reads a `null` patch value as "delete this path"
/// (`openjiuwen_session::state::paths::apply_patch`), which would strip
/// `task_plan` / `stop_condition_state` / the null `plan_mode` keys as soon as
/// the lane already exists. [`save_state`] therefore emulates replace semantics
/// in two writes: first `{deepagent: null}` deletes the lane, then the full
/// [`DeepAgentState::to_session_dict`] is written into the now-absent path, so
/// it is stored verbatim (explicit nulls included).
///
/// The persisted key set is therefore stable and idempotent across repeated
/// saves: `iteration`, `task_plan`, `stop_condition_state`,
/// `pending_follow_ups`, `plan_mode` (with `mode`, `pre_plan_mode`,
/// `plan_slug`, `prompt_context`).
///
/// # Errors
///
/// Returns [`Error::session`] when a session write fails.
pub async fn save_state(session: &dyn Session, state: &DeepAgentState) -> Result<()> {
    session
        .update_state(delete_state_patch())
        .await
        .map_err(|error| {
            Error::session(format!(
                "failed to clear session state {SESSION_STATE_KEY:?} before saving: {error}"
            ))
        })?;
    let mut patch = Map::new();
    patch.insert(SESSION_STATE_KEY.to_string(), state.to_session_dict());
    session.update_state(Value::Object(patch)).await
}

/// Replaces the persisted `deepagent` lane with a verbatim JSON payload.
///
/// This is the resume-side counterpart of [`save_state`]: a host that
/// reconstructed the state itself (for example the `state` JSON of a
/// `persistent_agent::ResumePlan`) can install it without going through
/// [`DeepAgentState`]. Unlike [`save_state`] the payload is stored verbatim —
/// the keys and types are the host's and are never validated against the
/// schema; [`load_state`] still decodes it leniently.
///
/// Replace semantics match [`save_state`] and Python's `update_dict`: the lane
/// is deleted first (`{deepagent: null}`) and the object is then written into
/// the now-absent path. The session store would otherwise deep-merge the two
/// objects, leaving keys of the previous payload behind; deleting first also
/// keeps explicit nulls in `state`.
///
/// # Errors
///
/// Returns [`Error::validation`] when `state` is not a JSON object, and
/// [`Error::session`] when a session write fails.
pub async fn seed_state(session: &dyn Session, state: Value) -> Result<()> {
    if !state.is_object() {
        return Err(Error::validation(
            "seeded deepagent state must be a JSON object",
        ));
    }
    session
        .update_state(delete_state_patch())
        .await
        .map_err(|error| {
            Error::session(format!(
                "failed to clear session state {SESSION_STATE_KEY:?} before seeding: {error}"
            ))
        })?;
    let mut patch = Map::new();
    patch.insert(SESSION_STATE_KEY.to_string(), state);
    session
        .update_state(Value::Object(patch))
        .await
        .map_err(|error| {
            Error::session(format!(
                "failed to seed session state {SESSION_STATE_KEY:?}: {error}"
            ))
        })
}

/// A patch that deletes [`SESSION_STATE_KEY`] (`null` patch value = delete).
fn delete_state_patch() -> Value {
    let mut patch = Map::new();
    patch.insert(SESSION_STATE_KEY.to_string(), Value::Null);
    Value::Object(patch)
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
    session.update_state(delete_state_patch()).await
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use openjiuwen_session::{AgentSession, Session};
    use serde_json::{json, Value};

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

    /// The sorted object keys of `value`.
    fn keys_sorted(value: &Value) -> Vec<&str> {
        let mut keys: Vec<&str> = value
            .as_object()
            .expect("object")
            .keys()
            .map(String::as_str)
            .collect();
        keys.sort_unstable();
        keys
    }

    fn full_state() -> DeepAgentState {
        DeepAgentState {
            iteration: 7,
            task_plan: Some(TaskPlan {
                goal: "ship it".to_string(),
                tasks: Vec::new(),
                current_task_id: None,
            }),
            stop_condition_state: Some(json!({"iteration": 7})),
            pending_follow_ups: vec!["later".to_string()],
            plan_mode: PlanModeState {
                mode: "plan".to_string(),
                pre_plan_mode: Some("normal".to_string()),
                plan_slug: Some("my-plan".to_string()),
                prompt_context: None,
            },
        }
    }

    /// Python `update_dict` replaces the `deepagent` value, so a repeated save
    /// keeps every key including the explicit nulls.
    #[tokio::test]
    async fn repeated_saves_keep_null_keys() {
        let session = AgentSession::new("state-repeated-saves");
        save_state(&session, &full_state()).await.unwrap();
        save_state(&session, &DeepAgentState::default())
            .await
            .unwrap();

        let payload = session
            .get_state(SESSION_STATE_KEY)
            .await
            .unwrap()
            .expect("payload present");
        assert_eq!(
            keys_sorted(&payload),
            vec![
                "iteration",
                "pending_follow_ups",
                "plan_mode",
                "stop_condition_state",
                "task_plan",
            ]
        );
        assert_eq!(payload["iteration"], json!(0));
        assert_eq!(payload["task_plan"], Value::Null);
        assert_eq!(payload["stop_condition_state"], Value::Null);
        assert_eq!(payload["pending_follow_ups"], json!([]));
        let plan_mode = payload["plan_mode"].as_object().expect("plan_mode object");
        assert_eq!(
            keys_sorted(&payload["plan_mode"]),
            vec!["mode", "plan_slug", "pre_plan_mode", "prompt_context"]
        );
        // `PlanModeState::default` keeps Python's asymmetric default: the
        // fresh default has `pre_plan_mode = "normal"`, while `plan_slug` and
        // `prompt_context` are explicit nulls.
        assert_eq!(plan_mode.get("mode"), Some(&json!("normal")));
        assert_eq!(plan_mode.get("pre_plan_mode"), Some(&json!("normal")));
        assert_eq!(plan_mode.get("plan_slug"), Some(&Value::Null));
        assert_eq!(plan_mode.get("prompt_context"), Some(&Value::Null));

        // A third save is idempotent.
        save_state(&session, &DeepAgentState::default())
            .await
            .unwrap();
        let again = session
            .get_state(SESSION_STATE_KEY)
            .await
            .unwrap()
            .expect("payload present");
        assert_eq!(again, payload);
    }

    /// A `PlanModeState` with every optional field unset persists all four
    /// keys as explicit nulls, also after repeated saves.
    #[tokio::test]
    async fn repeated_saves_keep_null_plan_mode_fields() {
        let session = AgentSession::new("state-repeated-plan-mode");
        save_state(&session, &full_state()).await.unwrap();
        let bare = DeepAgentState {
            plan_mode: PlanModeState {
                mode: "normal".to_string(),
                pre_plan_mode: None,
                plan_slug: None,
                prompt_context: None,
            },
            ..DeepAgentState::default()
        };
        save_state(&session, &bare).await.unwrap();

        let payload = session
            .get_state(SESSION_STATE_KEY)
            .await
            .unwrap()
            .expect("payload present");
        assert_eq!(
            keys_sorted(&payload["plan_mode"]),
            vec!["mode", "plan_slug", "pre_plan_mode", "prompt_context"]
        );
        assert_eq!(payload["plan_mode"]["pre_plan_mode"], Value::Null);
        assert_eq!(payload["plan_mode"]["plan_slug"], Value::Null);
        assert_eq!(payload["plan_mode"]["prompt_context"], Value::Null);
    }

    #[tokio::test]
    async fn load_state_round_trips_full_state_after_repeated_saves() {
        let session = AgentSession::new("state-repeated-round-trip");
        save_state(&session, &DeepAgentState::default())
            .await
            .unwrap();
        let state = full_state();
        save_state(&session, &state).await.unwrap();
        assert_eq!(load_state(&session).await.unwrap(), state);
    }

    #[tokio::test]
    async fn load_state_round_trips_default_state() {
        let session = AgentSession::new("state-default-round-trip");
        let state = DeepAgentState::default();
        save_state(&session, &state).await.unwrap();
        assert_eq!(load_state(&session).await.unwrap(), state);
    }
}
