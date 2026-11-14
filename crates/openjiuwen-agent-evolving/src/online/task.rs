//! RL task lifecycle: the immutable task identity, its durable record, and the
//! store-driven [`TaskRegistry`].
//!
//! The registry is a pure state machine over [`OnlineStore`] primitives: it
//! holds no cache, so two registries over the same store see the same state, and
//! the in-memory and Redis stores share every behaviour and every check.
//!
//! # Task state machine
//!
//! | From | Event | To | Notes |
//! |---|---|---|---|
//! | — | `start` (new id, no active task for the session) | `active` | creates the record |
//! | — | `start` (same id, same spec) | unchanged | returns the existing record, `created = false` |
//! | — | `start` (same id, different spec) | unchanged | [`OnlineError::TaskConflict`] |
//! | — | `start` (other id, session task active) | unchanged | returns *that* task, `created = false` |
//! | `active` | `finish` with no open captures | `finalized` | records `finished_at`/`finish_reason` |
//! | `active` | `finish` with open captures | unchanged | [`OnlineError::TaskConflict`] |
//! | `active` | `abort` | `aborted` | drops open captures and unpublished samples |
//! | `finalized`, `aborted` | `finish`/`abort` | unchanged | first reason wins, never an error |
//!
//! Timestamps come from the injected [`Clock`]; this module never reads the wall
//! clock, so records are reproducible in tests.

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use super::error::{OnlineError, Result};
use super::json;
use super::store::{keys, OnlineStore};

/// The timestamp [`FixedClock::default`] reports.
pub const EPOCH_STAMP: &str = "1970-01-01T00:00:00+00:00";

/// Source of the `created_at`/`finished_at` stamps of a [`TaskRecord`].
///
/// The trait exists so the state machine stays deterministic: [`TaskRegistry::new`]
/// installs [`FixedClock`], and a process that wants wall-clock stamps (the
/// service layer) injects its own implementation through
/// [`TaskRegistry::with_clock`].
pub trait Clock: Send + Sync {
    /// The current timestamp, ISO-8601 with an explicit UTC offset.
    fn now(&self) -> String;
}

/// A [`Clock`] that always reports the same stamp.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FixedClock {
    stamp: String,
}

impl FixedClock {
    /// Creates a clock pinned to `stamp`.
    pub fn new(stamp: impl Into<String>) -> Self {
        Self {
            stamp: stamp.into(),
        }
    }
}

impl Default for FixedClock {
    fn default() -> Self {
        Self::new(EPOCH_STAMP)
    }
}

impl Clock for FixedClock {
    fn now(&self) -> String {
        self.stamp.clone()
    }
}

/// Reward timing fixed for one RL task.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RewardMode {
    /// One reward for the whole session, submitted after finalizing.
    #[default]
    Terminal,
    /// A reward per agent turn, published on the turn boundary.
    DelayedFeedback,
}

impl RewardMode {
    /// The Python wire literal of the mode.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Terminal => "terminal",
            Self::DelayedFeedback => "delayed_feedback",
        }
    }
}

/// Persisted lifecycle state of an RL task.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskStatus {
    /// Accepting captures.
    #[default]
    Active,
    /// Closed by a normal stop; its samples may still be published.
    Finalized,
    /// Closed abnormally; unpublished samples were dropped.
    Aborted,
}

impl TaskStatus {
    /// The Python wire literal of the status.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Active => "active",
            Self::Finalized => "finalized",
            Self::Aborted => "aborted",
        }
    }
}

/// Why a task left the active state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    /// The user stopped the session.
    UserStopped,
    /// The service stopped the session.
    ServiceStopped,
    /// A capture failed, so the capture window is unusable.
    CaptureFailed,
    /// The session exceeded its lifetime.
    Timeout,
    /// A previous service process died while the task was active.
    ServiceRestarted,
}

impl FinishReason {
    /// The Python wire literal of the reason.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::UserStopped => "user_stopped",
            Self::ServiceStopped => "service_stopped",
            Self::CaptureFailed => "capture_failed",
            Self::Timeout => "timeout",
            Self::ServiceRestarted => "service_restarted",
        }
    }
}

/// Identity and fixed policy of one agent session's capture window.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct TaskSpec {
    /// Caller-chosen task id (`[A-Za-z0-9_-]{1,64}`).
    pub rl_task_id: String,
    /// Agent session the task captures.
    pub agent_session_id: String,
    /// Model identity used by the samples.
    pub model_id: String,
    /// `LoRA` (policy version) name of the samples.
    pub policy_lora_name: String,
    /// Reward timing of the task.
    pub reward_mode: RewardMode,
    /// Model name the policy server answers with; defaults to `policy_lora_name`.
    #[serde(default)]
    pub policy_model: Option<String>,
}

impl TaskSpec {
    /// The effective policy model (`policy_model`, else `policy_lora_name`).
    #[must_use]
    pub fn policy_model(&self) -> &str {
        self.policy_model
            .as_deref()
            .unwrap_or(self.policy_lora_name.as_str())
    }

    /// Rejects blank identities, an out-of-shape task id, and a blank model.
    pub fn validate(&self) -> Result<()> {
        for (field, value) in [
            ("rl_task_id", &self.rl_task_id),
            ("agent_session_id", &self.agent_session_id),
            ("model_id", &self.model_id),
            ("policy_lora_name", &self.policy_lora_name),
        ] {
            if value.trim().is_empty() {
                return Err(OnlineError::InvalidRequest(format!("{field} is required")));
            }
        }
        if !is_valid_task_id(&self.rl_task_id) {
            return Err(OnlineError::InvalidRequest(
                "rl_task_id must contain 1-64 letters, digits, underscores, or hyphens".to_owned(),
            ));
        }
        if self
            .policy_model
            .as_deref()
            .is_some_and(|model| model.trim().is_empty())
        {
            return Err(OnlineError::InvalidRequest(
                "policy_model is required".to_owned(),
            ));
        }
        Ok(())
    }
}

fn is_valid_task_id(task_id: &str) -> bool {
    !task_id.is_empty()
        && task_id.len() <= 64
        && task_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-')
}

/// Durable public state of an RL task.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default, from = "TaskRecordWire")]
pub struct TaskRecord {
    /// Task id.
    pub rl_task_id: String,
    /// Agent session the task captures.
    pub agent_session_id: String,
    /// Model identity used by the samples.
    pub model_id: String,
    /// `LoRA` (policy version) name of the samples.
    pub policy_lora_name: String,
    /// Reward timing of the task.
    pub reward_mode: RewardMode,
    /// Lifecycle state.
    pub status: TaskStatus,
    /// Creation stamp from the registry's [`Clock`].
    pub created_at: String,
    /// Closing stamp, once the task left `active`.
    #[serde(default)]
    pub finished_at: Option<String>,
    /// Why the task left `active`.
    #[serde(default)]
    pub finish_reason: Option<FinishReason>,
    /// Model name the policy server answers with.
    pub policy_model: String,
}

impl TaskRecord {
    /// Builds the active record of `spec`, stamped with `created_at`.
    pub(crate) fn from_spec(spec: &TaskSpec, created_at: String) -> Self {
        Self {
            rl_task_id: spec.rl_task_id.clone(),
            agent_session_id: spec.agent_session_id.clone(),
            model_id: spec.model_id.clone(),
            policy_lora_name: spec.policy_lora_name.clone(),
            reward_mode: spec.reward_mode,
            status: TaskStatus::Active,
            created_at,
            finished_at: None,
            finish_reason: None,
            policy_model: spec.policy_model().to_owned(),
        }
    }

    /// Whether `spec` describes exactly this task (the idempotent-start check).
    #[must_use]
    pub fn matches_spec(&self, spec: &TaskSpec) -> bool {
        self.rl_task_id == spec.rl_task_id
            && self.agent_session_id == spec.agent_session_id
            && self.model_id == spec.model_id
            && self.policy_lora_name == spec.policy_lora_name
            && self.policy_model == spec.policy_model()
            && self.reward_mode == spec.reward_mode
    }

    /// Deserializes a record from its store representation.
    pub fn from_json(payload: &str) -> Result<Self> {
        json::decode(payload)
    }

    /// Serializes the record into its store representation.
    pub fn to_json(&self) -> Result<String> {
        json::encode(self)
    }
}

/// Deserialization mirror of [`TaskRecord`] with an optional `policy_model`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
struct TaskRecordWire {
    rl_task_id: String,
    agent_session_id: String,
    model_id: String,
    policy_lora_name: String,
    reward_mode: RewardMode,
    status: TaskStatus,
    created_at: String,
    finished_at: Option<String>,
    finish_reason: Option<FinishReason>,
    policy_model: Option<String>,
}

impl Default for TaskRecordWire {
    fn default() -> Self {
        let record = TaskRecord::default();
        Self {
            rl_task_id: record.rl_task_id,
            agent_session_id: record.agent_session_id,
            model_id: record.model_id,
            policy_lora_name: record.policy_lora_name,
            reward_mode: record.reward_mode,
            status: record.status,
            created_at: record.created_at,
            finished_at: record.finished_at,
            finish_reason: record.finish_reason,
            policy_model: None,
        }
    }
}

impl From<TaskRecordWire> for TaskRecord {
    fn from(wire: TaskRecordWire) -> Self {
        let policy_model = wire
            .policy_model
            .filter(|model| !model.is_empty())
            .unwrap_or_else(|| wire.policy_lora_name.clone());
        Self {
            rl_task_id: wire.rl_task_id,
            agent_session_id: wire.agent_session_id,
            model_id: wire.model_id,
            policy_lora_name: wire.policy_lora_name,
            reward_mode: wire.reward_mode,
            status: wire.status,
            created_at: wire.created_at,
            finished_at: wire.finished_at,
            finish_reason: wire.finish_reason,
            policy_model,
        }
    }
}

/// The terminal reward projection record (`rl:v1:task:{id}:reward`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub(crate) struct RewardState {
    pub(crate) reward: f64,
    pub(crate) status: RewardStatus,
    pub(crate) sample_count: usize,
}

impl Default for RewardState {
    fn default() -> Self {
        Self {
            reward: 0.0,
            status: RewardStatus::Projecting,
            sample_count: 0,
        }
    }
}

/// Publication state of a terminal reward projection.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum RewardStatus {
    /// Claimed by one submitter; the samples are being projected.
    #[default]
    Projecting,
    /// Projected onto every sample; replaying the same reward is a no-op.
    Published,
}

/// Ownership of RL tasks, their active-session index, and their capture state.
///
/// The registry is stateless: every method reads and writes through the
/// [`OnlineStore`], so the same instance can be shared behind an `Arc` and a
/// second instance over the same store observes identical state.
pub struct TaskRegistry<S: OnlineStore> {
    store: S,
    clock: Arc<dyn Clock>,
}

impl<S: OnlineStore> TaskRegistry<S> {
    /// Creates a registry over `store` with the deterministic [`FixedClock`].
    pub fn new(store: S) -> Self {
        Self::with_clock(store, Arc::new(FixedClock::default()))
    }

    /// Creates a registry over `store` that stamps records with `clock`.
    pub fn with_clock(store: S, clock: Arc<dyn Clock>) -> Self {
        Self { store, clock }
    }

    /// The backing store.
    #[must_use]
    pub fn store(&self) -> &S {
        &self.store
    }

    /// Creates or adopts the session's active task.
    ///
    /// Returns `created = false` when the id already holds the same spec (an
    /// exact replay) or when the session already captures for another active
    /// task (the caller adopts it, as Python does).
    pub async fn start(&self, spec: &TaskSpec) -> Result<(TaskRecord, bool)> {
        spec.validate()?;
        let task_key = keys::task(&spec.rl_task_id);
        if let Some(payload) = self.read(&task_key).await? {
            let existing = TaskRecord::from_json(&payload)?;
            if existing.matches_spec(spec) {
                return Ok((existing, false));
            }
            return Err(OnlineError::TaskConflict(format!(
                "rl_task_id already exists: {}",
                spec.rl_task_id
            )));
        }

        let active_key = keys::active_session(&spec.agent_session_id);
        if let Some(active_id) = self.read(&active_key).await? {
            if let Some(payload) = self.read(&keys::task(&active_id)).await? {
                let active = TaskRecord::from_json(&payload)?;
                if active.status == TaskStatus::Active {
                    return Ok((active, false));
                }
            }
        }

        let record = TaskRecord::from_spec(spec, self.clock.now());
        if !self.write_if_absent(&task_key, &record.to_json()?).await? {
            // A concurrent start won the create; behave like the replay branch.
            let payload = self
                .read(&task_key)
                .await?
                .ok_or_else(|| missing_task(&spec.rl_task_id))?;
            let existing = TaskRecord::from_json(&payload)?;
            if existing.matches_spec(spec) {
                return Ok((existing, false));
            }
            return Err(OnlineError::TaskConflict(format!(
                "rl_task_id already exists: {}",
                spec.rl_task_id
            )));
        }
        self.write(&active_key, &record.rl_task_id).await?;
        self.insert_member(&keys::active_tasks(), &record.rl_task_id)
            .await?;
        Ok((record, true))
    }

    /// Returns a task by id.
    pub async fn get(&self, rl_task_id: &str) -> Result<TaskRecord> {
        self.load(rl_task_id)
            .await?
            .ok_or_else(|| missing_task(rl_task_id))
    }

    /// Finalizes an active task; replayed calls return the first outcome.
    pub async fn finish(&self, rl_task_id: &str, reason: FinishReason) -> Result<TaskRecord> {
        self.close(rl_task_id, TaskStatus::Finalized, reason).await
    }

    /// Aborts an active task, dropping its open captures and unpublished samples.
    pub async fn abort(&self, rl_task_id: &str, reason: FinishReason) -> Result<TaskRecord> {
        self.close(rl_task_id, TaskStatus::Aborted, reason).await
    }

    /// Returns the session's active task, if it has one.
    pub async fn active_task_for_session(
        &self,
        agent_session_id: &str,
    ) -> Result<Option<TaskRecord>> {
        let Some(active_id) = self.read(&keys::active_session(agent_session_id)).await? else {
            return Ok(None);
        };
        Ok(self
            .load(&active_id)
            .await?
            .filter(|task| task.status == TaskStatus::Active))
    }

    /// Projects one terminal reward onto every committed sample, exactly once.
    ///
    /// Returns the number of samples the reward covers. Replaying the same value
    /// is idempotent; a different value after the projection conflicts. The
    /// stored reward is compared exactly, because it is the very `f64` the first
    /// caller submitted.
    #[allow(clippy::float_cmp)]
    pub async fn record_reward(&self, rl_task_id: &str, reward: f64) -> Result<u64> {
        if !reward.is_finite() || !(0.0..=1.0).contains(&reward) {
            return Err(OnlineError::InvalidRequest(format!(
                "reward must be a finite number between 0 and 1: {reward}"
            )));
        }
        let task = self.get(rl_task_id).await?;
        if task.reward_mode != RewardMode::Terminal {
            return Err(OnlineError::TaskConflict(
                "terminal reward is not valid for delayed_feedback Task".to_owned(),
            ));
        }
        if task.status != TaskStatus::Finalized {
            return Err(OnlineError::TaskConflict(
                "terminal reward requires a finalized Task".to_owned(),
            ));
        }
        let capture_ids = self.members(&keys::committed_captures(rl_task_id)).await?;
        if capture_ids.is_empty() {
            return Err(OnlineError::TaskConflict(
                "terminal reward requires at least one capture".to_owned(),
            ));
        }

        let state_key = keys::reward(rl_task_id);
        let claimed = if let Some(payload) = self.read(&state_key).await? {
            json::decode::<RewardState>(&payload)?
        } else {
            let claimed = RewardState {
                reward,
                status: RewardStatus::Projecting,
                sample_count: capture_ids.len(),
            };
            if self
                .write_if_absent(&state_key, &json::encode(&claimed)?)
                .await?
            {
                claimed
            } else {
                // A concurrent submitter claimed first; adopt its record.
                let payload = self.read(&state_key).await?.ok_or_else(|| {
                    OnlineError::TaskConflict(
                        "RL Task reward claim disappeared while projecting".to_owned(),
                    )
                })?;
                json::decode::<RewardState>(&payload)?
            }
        };
        if claimed.reward != reward {
            return Err(OnlineError::TaskConflict(
                "RL Task already has a different terminal reward".to_owned(),
            ));
        }
        if claimed.status == RewardStatus::Published {
            return Ok(claimed.sample_count as u64);
        }

        let published = RewardState {
            reward,
            status: RewardStatus::Published,
            sample_count: claimed.sample_count,
        };
        self.write(&state_key, &json::encode(&published)?).await?;
        for capture_id in &capture_ids {
            self.remove(&keys::pending_sample(rl_task_id, capture_id))
                .await?;
        }
        Ok(published.sample_count as u64)
    }

    async fn close(
        &self,
        rl_task_id: &str,
        status: TaskStatus,
        reason: FinishReason,
    ) -> Result<TaskRecord> {
        let record = self.get(rl_task_id).await?;
        if record.status != TaskStatus::Active {
            return Ok(record);
        }
        if status == TaskStatus::Finalized
            && !self
                .members(&keys::open_captures(rl_task_id))
                .await?
                .is_empty()
        {
            return Err(OnlineError::TaskConflict(
                "RL Task still has open captures".to_owned(),
            ));
        }

        let finished = TaskRecord {
            status,
            finished_at: Some(self.clock.now()),
            finish_reason: Some(reason),
            ..record
        };
        self.write(&keys::task(rl_task_id), &finished.to_json()?)
            .await?;

        let active_key = keys::active_session(&finished.agent_session_id);
        if self.read(&active_key).await?.as_deref() == Some(rl_task_id) {
            self.remove(&active_key).await?;
        }
        self.remove_member(&keys::active_tasks(), rl_task_id)
            .await?;

        if status == TaskStatus::Aborted {
            for capture_id in self.members(&keys::committed_captures(rl_task_id)).await? {
                self.remove(&keys::pending_sample(rl_task_id, &capture_id))
                    .await?;
            }
            for capture_id in self.members(&keys::open_captures(rl_task_id)).await? {
                self.remove(&keys::capture(rl_task_id, &capture_id)).await?;
                self.remove_member(&keys::captures(rl_task_id), &capture_id)
                    .await?;
            }
            self.remove(&keys::open_captures(rl_task_id)).await?;
        }
        Ok(finished)
    }

    async fn load(&self, rl_task_id: &str) -> Result<Option<TaskRecord>> {
        match self.read(&keys::task(rl_task_id)).await? {
            Some(payload) => Ok(Some(TaskRecord::from_json(&payload)?)),
            None => Ok(None),
        }
    }

    /// Loads a task and rejects it unless it is still active.
    pub(crate) async fn require_active(&self, rl_task_id: &str) -> Result<TaskRecord> {
        let task = self.get(rl_task_id).await?;
        if task.status != TaskStatus::Active {
            return Err(OnlineError::TaskConflict(format!(
                "RL Task is not active: {rl_task_id}"
            )));
        }
        Ok(task)
    }

    pub(crate) async fn read(&self, key: &str) -> Result<Option<String>> {
        self.store.get(key).await.map_err(OnlineError::store)
    }

    pub(crate) async fn write(&self, key: &str, value: &str) -> Result<()> {
        self.store.set(key, value).await.map_err(OnlineError::store)
    }

    pub(crate) async fn write_if_absent(&self, key: &str, value: &str) -> Result<bool> {
        self.store
            .set_if_absent(key, value)
            .await
            .map_err(OnlineError::store)
    }

    pub(crate) async fn remove(&self, key: &str) -> Result<bool> {
        self.store.del(key).await.map_err(OnlineError::store)
    }

    pub(crate) async fn members(&self, key: &str) -> Result<Vec<String>> {
        self.store
            .set_members(key)
            .await
            .map_err(OnlineError::store)
    }

    pub(crate) async fn insert_member(&self, key: &str, member: &str) -> Result<bool> {
        self.store
            .set_add(key, member)
            .await
            .map_err(OnlineError::store)
    }

    pub(crate) async fn remove_member(&self, key: &str, member: &str) -> Result<bool> {
        self.store
            .set_remove(key, member)
            .await
            .map_err(OnlineError::store)
    }

    pub(crate) async fn contains_member(&self, key: &str, member: &str) -> Result<bool> {
        self.store
            .set_contains(key, member)
            .await
            .map_err(OnlineError::store)
    }
}

fn missing_task(rl_task_id: &str) -> OnlineError {
    OnlineError::TaskNotFound(format!("unknown RL Task: {rl_task_id}"))
}
