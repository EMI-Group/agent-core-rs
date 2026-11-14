//! The primitive store seam and its in-memory implementation.
//!
//! [`OnlineStore`] is intentionally domain-free: it speaks only string keys,
//! string values and sets of strings. That keeps one implementation of the task
//! state machine valid for every backend — the in-memory store used offline and
//! by tests, and the Redis store a later step adds (which maps `get`/`set` onto
//! `GET`/`SET`, `set_if_absent` onto `SET NX`, and the set operations onto
//! `SADD`/`SREM`/`SMEMBERS`).
//!
//! The key namespace every domain module shares is the `keys` module; the
//! literal layout is documented once, in the [`crate::online`] module docs.

use std::collections::{BTreeMap, BTreeSet};

use async_trait::async_trait;
use openjiuwen_common::error::Result;
use tokio::sync::Mutex;

/// Key namespace of the online RL state machine (`rl:v1`).
pub(crate) mod keys {
    /// Versioned prefix of every key this module writes.
    pub(crate) const PREFIX: &str = "rl:v1";

    /// Serialized `TaskRecord` of one RL task.
    pub(crate) fn task(rl_task_id: &str) -> String {
        format!("{PREFIX}:task:{rl_task_id}")
    }

    /// The task id a session currently captures for.
    pub(crate) fn active_session(agent_session_id: &str) -> String {
        format!("{PREFIX}:active_session:{agent_session_id}")
    }

    /// Recovery index of task ids that may still be active.
    pub(crate) fn active_tasks() -> String {
        format!("{PREFIX}:tasks:active")
    }

    /// Serialized `CaptureMeta` of one capture.
    pub(crate) fn capture(rl_task_id: &str, capture_id: &str) -> String {
        format!("{PREFIX}:task:{rl_task_id}:capture:{capture_id}")
    }

    /// Every capture id of the task.
    pub(crate) fn captures(rl_task_id: &str) -> String {
        format!("{PREFIX}:task:{rl_task_id}:captures")
    }

    /// Captures still waiting for their response.
    pub(crate) fn open_captures(rl_task_id: &str) -> String {
        format!("{PREFIX}:task:{rl_task_id}:captures:open")
    }

    /// Captures whose sample is staged but not yet published.
    pub(crate) fn committed_captures(rl_task_id: &str) -> String {
        format!("{PREFIX}:task:{rl_task_id}:captures:committed")
    }

    /// Captures committed during one agent turn.
    pub(crate) fn turn_captures(rl_task_id: &str, agent_turn_id: &str) -> String {
        format!("{PREFIX}:task:{rl_task_id}:turn:{agent_turn_id}:captures")
    }

    /// The turn currently accepting captures.
    pub(crate) fn current_turn(rl_task_id: &str) -> String {
        format!("{PREFIX}:task:{rl_task_id}:turn:current")
    }

    /// The turn waiting to become current.
    pub(crate) fn turn_transition(rl_task_id: &str) -> String {
        format!("{PREFIX}:task:{rl_task_id}:turn:transition")
    }

    /// Turns whose samples were published.
    pub(crate) fn closed_turns(rl_task_id: &str) -> String {
        format!("{PREFIX}:task:{rl_task_id}:turns:closed")
    }

    /// Publication state of one turn.
    pub(crate) fn turn_publish(rl_task_id: &str, agent_turn_id: &str) -> String {
        format!("{PREFIX}:task:{rl_task_id}:turn:{agent_turn_id}:publish")
    }

    /// A sample staged by a committed capture, awaiting publication.
    pub(crate) fn pending_sample(rl_task_id: &str, capture_id: &str) -> String {
        format!("{PREFIX}:task:{rl_task_id}:pending:{capture_id}")
    }

    /// The terminal reward projection record of the task.
    pub(crate) fn reward(rl_task_id: &str) -> String {
        format!("{PREFIX}:task:{rl_task_id}:reward")
    }
}

/// Primitive string/set store the task state machine runs on.
///
/// Implementations must be linearizable per key; the state machine never relies
/// on multi-key atomicity but its create path uses [`Self::set_if_absent`] to
/// stay correct under concurrent starts.
#[async_trait]
pub trait OnlineStore: Send + Sync {
    /// Reads a string value.
    async fn get(&self, key: &str) -> Result<Option<String>>;
    /// Writes a string value, replacing any previous one.
    async fn set(&self, key: &str, value: &str) -> Result<()>;
    /// Writes a string value only when the key is absent; returns whether it wrote.
    async fn set_if_absent(&self, key: &str, value: &str) -> Result<bool>;
    /// Deletes a key (string or set); returns whether anything was deleted.
    async fn del(&self, key: &str) -> Result<bool>;
    /// Adds a member to a set; returns whether it was new.
    async fn set_add(&self, key: &str, member: &str) -> Result<bool>;
    /// Removes a member from a set; returns whether it existed.
    async fn set_remove(&self, key: &str, member: &str) -> Result<bool>;
    /// Returns every member of a set, sorted so callers see a stable order.
    async fn set_members(&self, key: &str) -> Result<Vec<String>>;

    /// Returns whether `member` belongs to the set at `key`.
    ///
    /// The default reads the whole set; a backend with native membership testing
    /// (Redis `SISMEMBER`) overrides it.
    async fn set_contains(&self, key: &str, member: &str) -> Result<bool> {
        Ok(self
            .set_members(key)
            .await?
            .iter()
            .any(|item| item == member))
    }
}

/// Deterministic in-memory [`OnlineStore`].
///
/// Backed by a `BTreeMap` for values and a `BTreeSet` per set key, so iteration
/// order — and therefore [`OnlineStore::set_members`] — is stable without any
/// extra sorting. One `tokio::sync::Mutex` guards both maps; the guard is never
/// held across an `.await`, so the store cannot deadlock a task that only awaits
/// store operations.
#[derive(Debug, Default)]
pub struct MemoryOnlineStore {
    state: Mutex<MemoryState>,
}

#[derive(Debug, Default)]
struct MemoryState {
    values: BTreeMap<String, String>,
    sets: BTreeMap<String, BTreeSet<String>>,
}

impl MemoryOnlineStore {
    /// Creates an empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Every key holding data, sorted — a test helper.
    pub async fn keys(&self) -> Vec<String> {
        let state = self.state.lock().await;
        let mut keys: BTreeSet<String> = state.values.keys().cloned().collect();
        keys.extend(
            state
                .sets
                .iter()
                .filter(|(_, members)| !members.is_empty())
                .map(|(key, _)| key.clone()),
        );
        keys.into_iter().collect()
    }

    /// A copy of every string value, keyed by key — a test helper.
    pub async fn snapshot(&self) -> BTreeMap<String, String> {
        self.state.lock().await.values.clone()
    }
}

#[async_trait]
impl OnlineStore for MemoryOnlineStore {
    async fn get(&self, key: &str) -> Result<Option<String>> {
        Ok(self.state.lock().await.values.get(key).cloned())
    }

    async fn set(&self, key: &str, value: &str) -> Result<()> {
        self.state
            .lock()
            .await
            .values
            .insert(key.to_owned(), value.to_owned());
        Ok(())
    }

    async fn set_if_absent(&self, key: &str, value: &str) -> Result<bool> {
        let mut state = self.state.lock().await;
        if state.values.contains_key(key) {
            return Ok(false);
        }
        state.values.insert(key.to_owned(), value.to_owned());
        Ok(true)
    }

    async fn del(&self, key: &str) -> Result<bool> {
        let mut state = self.state.lock().await;
        let removed_value = state.values.remove(key).is_some();
        let removed_set = state.sets.remove(key).is_some();
        Ok(removed_value || removed_set)
    }

    async fn set_add(&self, key: &str, member: &str) -> Result<bool> {
        Ok(self
            .state
            .lock()
            .await
            .sets
            .entry(key.to_owned())
            .or_default()
            .insert(member.to_owned()))
    }

    async fn set_remove(&self, key: &str, member: &str) -> Result<bool> {
        let mut state = self.state.lock().await;
        let Some(members) = state.sets.get_mut(key) else {
            return Ok(false);
        };
        let removed = members.remove(member);
        if members.is_empty() {
            state.sets.remove(key);
        }
        Ok(removed)
    }

    async fn set_members(&self, key: &str) -> Result<Vec<String>> {
        Ok(self
            .state
            .lock()
            .await
            .sets
            .get(key)
            .map(|members| members.iter().cloned().collect())
            .unwrap_or_default())
    }
}
