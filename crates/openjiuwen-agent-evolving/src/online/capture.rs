//! Capture and turn bookkeeping of the RL task state machine.
//!
//! Python keeps these operations on `TaskRegistry` (its `capture_pipeline.py`
//! only orchestrates them), so they are extension methods on
//! [`TaskRegistry`] here as well.
//!
//! # Capture state machine
//!
//! | From | Event | To | Notes |
//! |---|---|---|---|
//! | — | `begin_capture` | `open` | reserves the id for one request fingerprint |
//! | — | `begin_capture` (same id, same turn+fingerprint) | unchanged | returns `None` (replay) |
//! | — | `begin_capture` (same id, other input) | unchanged | [`OnlineError::TaskConflict`] |
//! | `open` | `commit_capture` | `committed` | stages the sample, `true` |
//! | `committed` | `commit_capture` (same response) | unchanged | returns `false` (replay) |
//! | `committed` | `commit_capture` (other response) | unchanged | [`OnlineError::TaskConflict`] |
//! | `open` | `discard_capture` | deleted | idempotent |
//! | `committed` | `discard_capture` | unchanged | [`OnlineError::TaskConflict`] |
//!
//! # Turn publication state machine (`delayed_feedback` only)
//!
//! | Stored state | `claim_turn_publish` | Event | Next stored state |
//! |---|---|---|---|
//! | — | `Claimed` | `claim_turn_publish` | `publishing` |
//! | `publishing` | `Publishing` | `complete_turn_publish` | `published` (turn closed, samples published) |
//! | `publishing` | `Publishing` | `fail_turn_publish` | `failed` (terminal) |
//! | `published` | `Published` | — | terminal |
//! | `failed` | `Failed` | — | terminal |
//! | any | `Waiting` | open captures exist | unchanged, claim refused |
//!
//! A turn becomes `current` when its first capture opens; a capture of a *new*
//! turn request parks that turn in `:turn:transition` and returns a
//! [`TurnTransition`], which the caller resolves by publishing the previous turn
//! and calling [`TaskRegistry::advance_turn`].

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::error::{OnlineError, Result};
use super::json;
use super::store::{keys, OnlineStore};
use super::task::{RewardMode, TaskRegistry, TaskStatus};

/// State of one capture reservation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CaptureStatus {
    /// Reserved by `begin_capture`, waiting for its complete response.
    #[default]
    Open,
    /// The response was validated and its sample staged for publication.
    Committed,
}

/// Publication state of one agent turn.
///
/// `Waiting` and `Claimed` are outcomes of [`TaskRegistry::claim_turn_publish`]
/// only; `Publishing`, `Published` and `Failed` are the values stored under
/// `rl:v1:task:{id}:turn:{turn}:publish`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PublishState {
    /// Open captures still block the turn from being published.
    Waiting,
    /// The caller just claimed the turn and must publish it.
    Claimed,
    /// Another caller owns the publication.
    #[default]
    Publishing,
    /// The turn's samples were published; the turn is closed.
    Published,
    /// Publication failed; the turn is closed for good.
    Failed,
}

impl PublishState {
    /// The value stored under the turn's publish key.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Waiting => "waiting",
            Self::Claimed => "claimed",
            Self::Publishing => "publishing",
            Self::Published => "published",
            Self::Failed => "failed",
        }
    }

    fn parse(raw: &str, agent_turn_id: &str) -> Result<Self> {
        match raw {
            "publishing" => Ok(Self::Publishing),
            "published" => Ok(Self::Published),
            "failed" => Ok(Self::Failed),
            other => Err(OnlineError::TaskConflict(format!(
                "turn {agent_turn_id} has an unknown publish state: {other}"
            ))),
        }
    }
}

/// The turn change a delayed-feedback capture asks the caller to perform.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct TurnTransition {
    /// The turn that must be published before the advance.
    pub previous: String,
    /// The turn that becomes current once `previous` is published.
    pub next: String,
}

/// Metadata of one capture reservation.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct CaptureMeta {
    /// Caller-supplied capture id (the sample id).
    pub capture_id: String,
    /// Agent turn the capture belongs to; required in `delayed_feedback` mode.
    pub agent_turn_id: Option<String>,
    /// Fingerprint of the normalized request, computed by the caller.
    pub request_fingerprint: String,
    /// Fingerprint of the normalized response, set on commit.
    pub response_fingerprint: Option<String>,
    /// Reservation state.
    pub status: CaptureStatus,
}

impl CaptureMeta {
    /// Deserializes metadata from its store representation.
    pub fn from_json(payload: &str) -> Result<Self> {
        json::decode(payload)
    }

    /// Serializes the metadata into its store representation.
    pub fn to_json(&self) -> Result<String> {
        json::encode(self)
    }
}

impl<S: OnlineStore> TaskRegistry<S> {
    /// Reserves a capture id for one request.
    ///
    /// Returns a [`TurnTransition`] when a `delayed_feedback` capture opens a new
    /// turn while another turn is current: the caller must publish the previous
    /// turn and then call [`Self::advance_turn`]. Returns `None` for a fresh
    /// capture of the current turn and for an exact replay.
    pub async fn begin_capture(
        &self,
        rl_task_id: &str,
        capture_id: &str,
        agent_turn_id: Option<&str>,
        request_fingerprint: &str,
    ) -> Result<Option<TurnTransition>> {
        let task = self.require_active(rl_task_id).await?;
        let capture_key = keys::capture(rl_task_id, capture_id);
        if let Some(payload) = self.read(&capture_key).await? {
            let existing = CaptureMeta::from_json(&payload)?;
            if existing.agent_turn_id.as_deref() != agent_turn_id
                || existing.request_fingerprint != request_fingerprint
            {
                return Err(OnlineError::TaskConflict(format!(
                    "capture_id already belongs to different input: {capture_id}"
                )));
            }
            return Ok(None);
        }

        if task.reward_mode == RewardMode::DelayedFeedback {
            let turn_id = agent_turn_id.unwrap_or("");
            if self
                .contains_member(&keys::closed_turns(rl_task_id), turn_id)
                .await?
            {
                return Err(OnlineError::TurnClosed(format!(
                    "turn is already closed: {turn_id}"
                )));
            }
            let publish_key = keys::turn_publish(rl_task_id, turn_id);
            if self.read(&publish_key).await?.is_some() {
                return Err(OnlineError::TurnClosed(format!(
                    "turn is already closing: {turn_id}"
                )));
            }
            let current_turn = self.read(&keys::current_turn(rl_task_id)).await?;
            let next_turn = self.read(&keys::turn_transition(rl_task_id)).await?;
            if let Some(current) = current_turn {
                if current != turn_id {
                    if let Some(next) = next_turn.as_deref() {
                        if next != turn_id {
                            return Err(OnlineError::TaskConflict(format!(
                                "turn {next} is already waiting for {current}"
                            )));
                        }
                    }
                    // Park the new turn and hand the advance back to the caller;
                    // the capture itself is not reserved until the turn is current.
                    if next_turn.is_none() {
                        self.write(&keys::turn_transition(rl_task_id), turn_id)
                            .await?;
                    }
                    return Ok(Some(TurnTransition {
                        previous: current,
                        next: turn_id.to_owned(),
                    }));
                }
                if next_turn.is_some() {
                    return Err(OnlineError::TurnClosed(format!(
                        "turn is closing: {turn_id}"
                    )));
                }
            }
        }

        let meta = CaptureMeta {
            capture_id: capture_id.to_owned(),
            agent_turn_id: agent_turn_id.map(str::to_owned),
            request_fingerprint: request_fingerprint.to_owned(),
            response_fingerprint: None,
            status: CaptureStatus::Open,
        };
        if task.reward_mode == RewardMode::DelayedFeedback {
            self.write_if_absent(&keys::current_turn(rl_task_id), agent_turn_id.unwrap_or(""))
                .await?;
        }
        self.write(&capture_key, &meta.to_json()?).await?;
        self.insert_member(&keys::captures(rl_task_id), capture_id)
            .await?;
        self.insert_member(&keys::open_captures(rl_task_id), capture_id)
            .await?;
        Ok(None)
    }

    /// Commits a capture's response and stages its sample.
    ///
    /// Returns `true` on the first commit and `false` when the same response is
    /// replayed; a different response for a committed capture conflicts.
    pub async fn commit_capture(
        &self,
        rl_task_id: &str,
        capture_id: &str,
        agent_turn_id: Option<&str>,
        response_fingerprint: &str,
        sample: &Value,
    ) -> Result<bool> {
        let task = self.get(rl_task_id).await?;
        let capture_key = keys::capture(rl_task_id, capture_id);
        let payload = self.read(&capture_key).await?.ok_or_else(|| {
            OnlineError::TaskConflict(format!("capture has no matching before: {capture_id}"))
        })?;
        let mut meta = CaptureMeta::from_json(&payload)?;
        if meta.agent_turn_id.as_deref() != agent_turn_id {
            return Err(OnlineError::TaskConflict(format!(
                "capture turn mismatch: {capture_id}"
            )));
        }
        if meta.status == CaptureStatus::Committed {
            if meta.response_fingerprint.as_deref() != Some(response_fingerprint) {
                return Err(OnlineError::TaskConflict(format!(
                    "capture_id already has a different response: {capture_id}"
                )));
            }
            return Ok(false);
        }
        if task.status != TaskStatus::Active {
            return Err(OnlineError::TaskConflict(format!(
                "RL Task is not active: {rl_task_id}"
            )));
        }

        meta.status = CaptureStatus::Committed;
        meta.response_fingerprint = Some(response_fingerprint.to_owned());
        self.write(&capture_key, &meta.to_json()?).await?;
        self.write(
            &keys::pending_sample(rl_task_id, capture_id),
            &json::encode(sample)?,
        )
        .await?;
        self.remove_member(&keys::open_captures(rl_task_id), capture_id)
            .await?;
        self.insert_member(&keys::committed_captures(rl_task_id), capture_id)
            .await?;
        if let Some(turn_id) = agent_turn_id {
            self.insert_member(&keys::turn_captures(rl_task_id, turn_id), capture_id)
                .await?;
        }
        Ok(true)
    }

    /// Releases a capture abandoned before a complete response.
    ///
    /// Idempotent: a missing capture, or a task that is no longer active, is not
    /// an error.
    pub async fn discard_capture(
        &self,
        rl_task_id: &str,
        capture_id: &str,
        agent_turn_id: Option<&str>,
    ) -> Result<()> {
        let task = self.get(rl_task_id).await?;
        let capture_key = keys::capture(rl_task_id, capture_id);
        let Some(payload) = self.read(&capture_key).await? else {
            return Ok(());
        };
        if task.status != TaskStatus::Active {
            return Ok(());
        }
        let meta = CaptureMeta::from_json(&payload)?;
        if meta.agent_turn_id.as_deref() != agent_turn_id {
            return Err(OnlineError::TaskConflict(format!(
                "capture turn mismatch: {capture_id}"
            )));
        }
        if meta.status != CaptureStatus::Open {
            return Err(OnlineError::TaskConflict(format!(
                "capture is already committed: {capture_id}"
            )));
        }
        self.remove(&capture_key).await?;
        self.remove_member(&keys::captures(rl_task_id), capture_id)
            .await?;
        self.remove_member(&keys::open_captures(rl_task_id), capture_id)
            .await?;
        Ok(())
    }

    /// Claims the right to publish `agent_turn_id`'s samples.
    pub async fn claim_turn_publish(
        &self,
        rl_task_id: &str,
        agent_turn_id: &str,
    ) -> Result<PublishState> {
        if !self
            .members(&keys::open_captures(rl_task_id))
            .await?
            .is_empty()
        {
            return Ok(PublishState::Waiting);
        }
        let publish_key = keys::turn_publish(rl_task_id, agent_turn_id);
        if let Some(raw) = self.read(&publish_key).await? {
            return PublishState::parse(&raw, agent_turn_id);
        }
        self.write(&publish_key, PublishState::Publishing.as_str())
            .await?;
        Ok(PublishState::Claimed)
    }

    /// Marks the claimed turn published, closes it, and drops its staged samples.
    pub async fn complete_turn_publish(&self, rl_task_id: &str, agent_turn_id: &str) -> Result<()> {
        self.require_active(rl_task_id).await?;
        let publish_key = keys::turn_publish(rl_task_id, agent_turn_id);
        let state = self.read(&publish_key).await?;
        if state.as_deref() != Some(PublishState::Publishing.as_str()) {
            return Err(OnlineError::TaskConflict(format!(
                "turn is not publishing: {agent_turn_id}"
            )));
        }
        self.write(&publish_key, PublishState::Published.as_str())
            .await?;
        self.insert_member(&keys::closed_turns(rl_task_id), agent_turn_id)
            .await?;
        for capture_id in self
            .members(&keys::turn_captures(rl_task_id, agent_turn_id))
            .await?
        {
            self.remove(&keys::pending_sample(rl_task_id, &capture_id))
                .await?;
        }
        Ok(())
    }

    /// Marks the turn's publication failed; the turn stays closed for good.
    pub async fn fail_turn_publish(&self, rl_task_id: &str, agent_turn_id: &str) -> Result<()> {
        self.write(
            &keys::turn_publish(rl_task_id, agent_turn_id),
            PublishState::Failed.as_str(),
        )
        .await
    }

    /// Makes the parked `next` turn current once `previous` is published.
    pub async fn advance_turn(&self, rl_task_id: &str, previous: &str, next: &str) -> Result<()> {
        let publish = self.read(&keys::turn_publish(rl_task_id, previous)).await?;
        if publish.as_deref() != Some(PublishState::Published.as_str()) {
            return Err(OnlineError::TaskConflict(format!(
                "turn is not published: {previous}"
            )));
        }
        let current_key = keys::current_turn(rl_task_id);
        let current = self.read(&current_key).await?;
        if current.as_deref() == Some(next) {
            return Ok(());
        }
        if current.as_deref() != Some(previous) {
            return Err(OnlineError::TaskConflict(
                "current turn changed during transition".to_owned(),
            ));
        }
        let transition_key = keys::turn_transition(rl_task_id);
        let transition = self.read(&transition_key).await?;
        if transition.as_deref() != Some(next) {
            return Err(OnlineError::TaskConflict(
                "next turn changed during transition".to_owned(),
            ));
        }
        self.write(&current_key, next).await?;
        self.remove(&transition_key).await?;
        Ok(())
    }

    /// Number of captures still waiting for a response.
    pub async fn open_capture_count(&self, rl_task_id: &str) -> Result<usize> {
        Ok(self.members(&keys::open_captures(rl_task_id)).await?.len())
    }

    /// Ids of the committed captures whose samples are not published yet, sorted.
    pub async fn pending_sample_ids(&self, rl_task_id: &str) -> Result<Vec<String>> {
        let mut ids = Vec::new();
        for capture_id in self.members(&keys::committed_captures(rl_task_id)).await? {
            if self
                .read(&keys::pending_sample(rl_task_id, &capture_id))
                .await?
                .is_some()
            {
                ids.push(capture_id);
            }
        }
        Ok(ids)
    }
}
