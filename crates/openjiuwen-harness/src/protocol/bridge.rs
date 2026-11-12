//! Envelope construction, id counters, and checkpoint persistence for the native harness.
//!
//! This module owns the small pieces of mutable protocol state the native
//! harness needs around its event stream:
//!
//! - [`EnvelopeBuilder`] stamps every [`HarnessEvent`] with the host/agent
//!   identity, an optional provider session id, a monotonic `sequence`, and a
//!   Unix-epoch-second [`unix_now`] timestamp;
//! - [`next_turn_id`] and [`next_message_id`] hand out the deterministic
//!   `turn-<n>` / `message-<n>` identifiers used by the protocol layer
//!   (documented deviation from Python's uuid hex, for reproducible tests);
//! - [`NativeCheckpointState`] is the only payload the provider stores in a
//!   [`HarnessCheckpoint`], with a stable schema version and JSON round-trip;
//! - [`CheckpointWriter`] builds, remembers, and best-effort persists those
//!   checkpoints through an optional [`HarnessCheckpointSink`].
//!
//! Checkpoint persistence is deliberately non-fatal: a sink failure is logged
//! at `warn` and keeps the previous storage revision, because a turn must never
//! fail only because the host could not persist progress.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use openjiuwen_harness_protocol::{
    json_object_from_value, json_object_to_value, CheckpointReason, HarnessCheckpoint,
    HarnessCheckpointSink, HarnessError, HarnessEvent, HarnessEventPayload, HarnessResult,
    JsonObject, JsonValue,
};
use serde::{Deserialize, Serialize};

/// Schema version of [`NativeCheckpointState`].
pub(crate) const CHECKPOINT_SCHEMA_VERSION: &str = "1";

/// Plan mode recorded when a checkpoint does not state one.
const DEFAULT_PLAN_MODE: &str = "normal";

/// Provider-private state stored in a native harness checkpoint.
///
/// The host treats the enclosing [`HarnessCheckpoint::data`] as opaque; only
/// the native provider decodes this type. It records everything needed to
/// continue a session: its scope (`session_id`, `agent_id`,
/// `host_session_id`), the kernel iteration cursor, queued follow-up inputs,
/// plan mode, the turn counter, and the active model/effort selection.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct NativeCheckpointState {
    /// Schema version of this payload; always [`CHECKPOINT_SCHEMA_VERSION`].
    pub schema_version: String,
    /// Kernel session id the checkpoint was taken from.
    pub session_id: String,
    /// Agent the checkpoint belongs to.
    pub agent_id: String,
    /// Host session the checkpoint belongs to.
    pub host_session_id: String,
    /// Kernel loop iteration reached when the checkpoint was taken.
    pub iteration: u64,
    /// Follow-up inputs still queued at checkpoint time.
    pub pending_follow_ups: Vec<String>,
    /// Plan mode in effect (`normal` or `plan`).
    pub plan_mode: String,
    /// Number of turns started in the current event cycle.
    pub turn_counter: u64,
    /// Selected model name, when the host chose one.
    pub model: Option<String>,
    /// Selected reasoning effort, when the host chose one.
    pub effort: Option<String>,
}

impl NativeCheckpointState {
    /// Builds an empty checkpoint state for the given scope.
    ///
    /// `schema_version` is set to [`CHECKPOINT_SCHEMA_VERSION`], `plan_mode` to
    /// `normal`, and every counter/collection starts empty.
    pub fn new(
        session_id: impl Into<String>,
        agent_id: impl Into<String>,
        host_session_id: impl Into<String>,
    ) -> Self {
        Self {
            schema_version: CHECKPOINT_SCHEMA_VERSION.to_owned(),
            session_id: session_id.into(),
            agent_id: agent_id.into(),
            host_session_id: host_session_id.into(),
            iteration: 0,
            pending_follow_ups: Vec::new(),
            plan_mode: DEFAULT_PLAN_MODE.to_owned(),
            turn_counter: 0,
            model: None,
            effort: None,
        }
    }

    /// Projects this state to a JSON object for [`HarnessCheckpoint::data`].
    ///
    /// A [`NativeCheckpointState`] is always JSON-serializable, so encoding
    /// cannot fail; the fallback empty object only satisfies the infallible
    /// signature.
    #[must_use]
    pub fn to_json(&self) -> JsonObject {
        json_object_from_value(serde_json::to_value(self).unwrap_or(JsonValue::Null))
            .unwrap_or_default()
    }

    /// Decodes a checkpoint payload and checks its schema version.
    ///
    /// # Errors
    ///
    /// [`HarnessError::Validation`] with `invalid native checkpoint` when
    /// `data` does not match this type, or with
    /// `unsupported native checkpoint version` when the decoded
    /// `schema_version` is not [`CHECKPOINT_SCHEMA_VERSION`].
    pub fn from_json(data: &JsonObject) -> HarnessResult<Self> {
        let state: Self = serde_json::from_value(json_object_to_value(data))
            .map_err(|_| HarnessError::validation("invalid native checkpoint"))?;
        if state.schema_version != CHECKPOINT_SCHEMA_VERSION {
            return Err(HarnessError::validation(
                "unsupported native checkpoint version",
            ));
        }
        Ok(state)
    }
}

/// Builds validated event envelopes with monotonic sequence numbers.
///
/// The builder is owned by the harness's shared state; [`next`](Self::next)
/// increments the sequence by one, stamps [`unix_now`], and applies the
/// current provider session id. The caller chains the correlation ids
/// (`with_turn_id`, `with_item_id`, `with_correlation_id`) and validates the
/// event once before emitting it.
#[derive(Debug, Clone)]
pub(crate) struct EnvelopeBuilder {
    /// Host session shared by every event of the run.
    host_session_id: String,
    /// Agent that produced the events.
    agent_id: String,
    /// Provider session id applied to new envelopes while a session is active.
    provider_session_id: Option<String>,
    /// Sequence of the last emitted event (`0` before the first one).
    sequence: u64,
}

impl EnvelopeBuilder {
    /// Builds a builder for the given host session and agent, sequence `0`.
    pub(crate) fn new(host_session_id: impl Into<String>, agent_id: impl Into<String>) -> Self {
        Self {
            host_session_id: host_session_id.into(),
            agent_id: agent_id.into(),
            provider_session_id: None,
            sequence: 0,
        }
    }

    /// Sets (or clears) the provider session id applied to new envelopes.
    pub(crate) fn set_provider_session_id(&mut self, value: Option<String>) {
        self.provider_session_id = value;
    }

    /// Builds the next envelope for `payload`.
    ///
    /// The returned event carries the incremented sequence, the current
    /// timestamp, the builder's identity, and the provider session id when one
    /// is set; correlation ids are left to the caller.
    pub(crate) fn next(&mut self, payload: HarnessEventPayload) -> HarnessEvent {
        self.sequence = self.sequence.saturating_add(1);
        let mut event = HarnessEvent::new(
            self.sequence,
            unix_now(),
            payload,
            self.host_session_id.clone(),
            self.agent_id.clone(),
        );
        if let Some(provider_session_id) = &self.provider_session_id {
            event = event.with_provider_session_id(provider_session_id.clone());
        }
        event
    }
}

/// Returns the current Unix time in epoch seconds.
///
/// Falls back to `0.0` when the system clock is before the Unix epoch (a
/// clock error `SystemTime` cannot otherwise represent), which keeps event
/// timestamps finite as the protocol requires.
pub(crate) fn unix_now() -> f64 {
    let Ok(duration) = SystemTime::now().duration_since(UNIX_EPOCH) else {
        return 0.0;
    };
    let seconds = duration.as_secs_f64();
    if seconds.is_finite() {
        seconds
    } else {
        0.0
    }
}

/// Returns the next turn id (`turn-1`, `turn-2`, …) from `counter`.
pub(crate) fn next_turn_id(counter: &mut u64) -> String {
    *counter = counter.saturating_add(1);
    format!("turn-{}", *counter)
}

/// Returns the next message id (`message-1`, `message-2`, …) from `counter`.
pub(crate) fn next_message_id(counter: &mut u64) -> String {
    *counter = counter.saturating_add(1);
    format!("message-{}", *counter)
}

/// Builds, tracks, and best-effort persists native harness checkpoints.
///
/// The writer keeps the monotonic checkpoint sequence, the latest successful
/// storage revision (used as the compare-and-set expectation of the next
/// write), and the latest locally built checkpoint for
/// `export_checkpoint`.
#[derive(Debug)]
pub(crate) struct CheckpointWriter {
    /// Provider name written into every checkpoint.
    provider: String,
    /// Sequence of the last built checkpoint.
    sequence: u64,
    /// Storage revision returned by the last successful sink write.
    revision: Option<String>,
    /// Most recently built checkpoint, persisted or not.
    latest: Option<HarnessCheckpoint>,
}

impl CheckpointWriter {
    /// Builds a writer for `provider` with sequence `0` and no revision yet.
    pub(crate) fn new(provider: impl Into<String>) -> Self {
        Self {
            provider: provider.into(),
            sequence: 0,
            revision: None,
            latest: None,
        }
    }

    /// Returns the most recently built checkpoint.
    pub(crate) fn latest(&self) -> Option<&HarnessCheckpoint> {
        self.latest.as_ref()
    }

    /// Returns the sequence of the most recently built checkpoint.
    pub(crate) fn checkpoint_sequence(&self) -> u64 {
        self.sequence
    }

    /// Restores the sequence counter, e.g. from a resumed checkpoint.
    pub(crate) fn reset(&mut self, sequence: u64) {
        self.sequence = sequence;
    }

    /// Builds the next checkpoint for `state` and hands it to `sink`.
    ///
    /// The sequence is incremented before the checkpoint is built, and the
    /// checkpoint carries the current storage revision as its own revision and
    /// as the sink's compare-and-set expectation. Persistence is best effort: a
    /// sink failure is logged at `warn` and keeps the previous revision. The
    /// built checkpoint is always recorded as [`latest`](Self::latest) and
    /// returned.
    pub(crate) async fn emit(
        &mut self,
        sink: Option<&Arc<dyn HarnessCheckpointSink>>,
        state: &NativeCheckpointState,
        provider_session_id: Option<&str>,
        reason: CheckpointReason,
    ) -> Option<HarnessCheckpoint> {
        self.sequence = self.sequence.saturating_add(1);
        let checkpoint = HarnessCheckpoint {
            provider: self.provider.clone(),
            schema_version: CHECKPOINT_SCHEMA_VERSION.to_owned(),
            agent_id: state.agent_id.clone(),
            host_session_id: state.host_session_id.clone(),
            checkpoint_id: format!("checkpoint-{}", uuid::Uuid::new_v4().simple()),
            sequence: self.sequence,
            data: state.to_json(),
            provider_session_id: provider_session_id.map(str::to_owned),
            revision: self.revision.clone(),
        };
        if let Some(sink) = sink {
            match sink
                .save(&checkpoint, reason, self.revision.as_deref())
                .await
            {
                Ok(receipt) => self.revision = Some(receipt.storage_revision),
                Err(error) => {
                    tracing::warn!(%error, "failed to persist native checkpoint");
                }
            }
        }
        self.latest = Some(checkpoint.clone());
        Some(checkpoint)
    }
}

/// Decodes a checkpoint payload after checking the provider schema version.
///
/// The provider and scope checks (`provider`, `agent_id`, `host_session_id`)
/// are the caller's responsibility; this helper only gates the payload schema
/// and then delegates to [`NativeCheckpointState::from_json`].
///
/// # Errors
///
/// [`HarnessError::Validation`] with `unsupported native checkpoint version`
/// for a foreign schema, or the [`NativeCheckpointState::from_json`] error for
/// an undecodable payload.
pub(crate) fn state_from_checkpoint(
    checkpoint: &HarnessCheckpoint,
) -> HarnessResult<NativeCheckpointState> {
    if checkpoint.schema_version != CHECKPOINT_SCHEMA_VERSION {
        return Err(HarnessError::validation(
            "unsupported native checkpoint version",
        ));
    }
    NativeCheckpointState::from_json(&checkpoint.data)
}

#[cfg(test)]
mod tests {
    // Tests assert by panicking on failure, which is the intent of a test.
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::{
        next_message_id, next_turn_id, state_from_checkpoint, unix_now, CheckpointWriter,
        EnvelopeBuilder, NativeCheckpointState, CHECKPOINT_SCHEMA_VERSION,
    };
    use async_trait::async_trait;
    use openjiuwen_harness_protocol::{
        CheckpointReason, CheckpointSaveReceipt, HarnessCheckpoint, HarnessCheckpointSink,
        HarnessError, HarnessErrorKind, HarnessEventPayload, HarnessResult, HarnessState,
        JsonObject, StateChangedEvent,
    };
    use std::sync::{Arc, Mutex};

    /// Sink that records the expected revision of every write.
    struct RecordingSink {
        /// Expected revisions observed, in call order.
        expected: Mutex<Vec<Option<String>>>,
        /// When `true`, every write fails with a checkpoint conflict.
        fail: bool,
    }

    impl RecordingSink {
        /// Builds a sink that records expectations and succeeds.
        fn new() -> Arc<Self> {
            Arc::new(Self {
                expected: Mutex::new(Vec::new()),
                fail: false,
            })
        }
    }

    #[async_trait]
    impl HarnessCheckpointSink for RecordingSink {
        async fn save(
            &self,
            checkpoint: &HarnessCheckpoint,
            _reason: CheckpointReason,
            expected_storage_revision: Option<&str>,
        ) -> HarnessResult<CheckpointSaveReceipt> {
            self.expected
                .lock()
                .expect("lock")
                .push(expected_storage_revision.map(str::to_owned));
            if self.fail {
                return Err(HarnessError::checkpoint_conflict("stale"));
            }
            Ok(CheckpointSaveReceipt {
                checkpoint_id: checkpoint.checkpoint_id.clone(),
                sequence: checkpoint.sequence,
                storage_revision: format!("rev-{}", checkpoint.sequence),
            })
        }
    }

    /// Builds a state-change payload.
    fn payload() -> HarnessEventPayload {
        StateChangedEvent {
            old: HarnessState::Idle,
            new: HarnessState::Running,
        }
        .into()
    }

    /// Builds a sample checkpoint state.
    fn state() -> NativeCheckpointState {
        let mut state = NativeCheckpointState::new("session-1", "agent-1", "host-1");
        state.iteration = 3;
        state.turn_counter = 2;
        state.plan_mode = "plan".to_owned();
        state.pending_follow_ups = vec!["next".to_owned()];
        state.model = Some("model-a".to_owned());
        state.effort = Some("high".to_owned());
        state
    }

    #[test]
    fn envelopes_are_monotonic_and_carry_the_provider_session() {
        let mut builder = EnvelopeBuilder::new("host-1", "agent-1");
        builder.set_provider_session_id(Some("provider-1".to_owned()));
        let first = builder.next(payload());
        let second = builder.next(payload());
        assert_eq!(first.sequence, 1);
        assert_eq!(second.sequence, 2);
        assert_eq!(first.host_session_id, "host-1");
        assert_eq!(first.agent_id, "agent-1");
        assert_eq!(first.provider_session_id.as_deref(), Some("provider-1"));
        assert!(first.timestamp.is_finite() && first.timestamp >= 0.0);
        assert!(second.timestamp.is_finite() && second.timestamp >= 0.0);
        assert_eq!(first.turn_id, None);
        assert_eq!(first.causation_ids, Vec::<String>::new());

        builder.set_provider_session_id(None);
        let third = builder.next(payload());
        assert_eq!(third.sequence, 3);
        assert_eq!(third.provider_session_id, None);
    }

    #[test]
    fn unix_now_is_a_finite_timestamp() {
        let now = unix_now();
        assert!(now.is_finite());
        assert!(now > 0.0);
    }

    #[test]
    fn id_counters_start_at_one() {
        let mut turn_counter = 0;
        let mut message_counter = 7;
        assert_eq!(next_turn_id(&mut turn_counter), "turn-1");
        assert_eq!(next_turn_id(&mut turn_counter), "turn-2");
        assert_eq!(next_message_id(&mut message_counter), "message-8");
        assert_eq!(next_message_id(&mut message_counter), "message-9");
    }

    #[test]
    fn checkpoint_state_round_trips_through_json() {
        let state = state();
        let json = state.to_json();
        assert_eq!(
            json.get("schema_version"),
            Some(&serde_json::json!(CHECKPOINT_SCHEMA_VERSION))
        );
        assert_eq!(json.get("plan_mode"), Some(&serde_json::json!("plan")));
        let decoded = NativeCheckpointState::from_json(&json).expect("round-trip");
        assert_eq!(decoded, state);
        assert_eq!(decoded.iteration, 3);
        assert_eq!(decoded.turn_counter, 2);
        assert_eq!(decoded.pending_follow_ups, vec!["next".to_owned()]);
        assert_eq!(decoded.model.as_deref(), Some("model-a"));
        assert_eq!(decoded.effort.as_deref(), Some("high"));
    }

    #[test]
    fn checkpoint_state_defaults_match_the_schema() {
        let state = NativeCheckpointState::new("s", "a", "h");
        assert_eq!(state.schema_version, CHECKPOINT_SCHEMA_VERSION);
        assert_eq!(state.plan_mode, "normal");
        assert_eq!(state.iteration, 0);
        assert_eq!(state.turn_counter, 0);
        assert_eq!(state.pending_follow_ups, Vec::<String>::new());
        assert!(state.model.is_none());
        assert!(state.effort.is_none());
    }

    #[test]
    fn checkpoint_state_rejects_a_foreign_schema_and_bad_payloads() {
        let mut json = state().to_json();
        json.insert("schema_version".to_owned(), serde_json::json!("2"));
        let error = NativeCheckpointState::from_json(&json).expect_err("foreign schema");
        assert_eq!(error.kind(), HarnessErrorKind::Validation);
        assert_eq!(error.message(), "unsupported native checkpoint version");

        let error = NativeCheckpointState::from_json(&JsonObject::new()).expect_err("empty");
        assert_eq!(error.message(), "invalid native checkpoint");

        let encoded = std::collections::BTreeMap::from([(
            "schema_version".to_owned(),
            serde_json::json!(CHECKPOINT_SCHEMA_VERSION),
        )]);
        let error = NativeCheckpointState::from_json(&encoded).expect_err("missing fields");
        assert_eq!(error.message(), "invalid native checkpoint");
    }

    #[test]
    fn state_from_checkpoint_gates_the_schema_before_decoding() {
        let checkpoint = HarnessCheckpoint {
            provider: "openjiuwen-native".to_owned(),
            schema_version: "9".to_owned(),
            agent_id: "agent-1".to_owned(),
            host_session_id: "host-1".to_owned(),
            checkpoint_id: "checkpoint-1".to_owned(),
            sequence: 1,
            data: JsonObject::new(),
            provider_session_id: None,
            revision: None,
        };
        let error = state_from_checkpoint(&checkpoint).expect_err("foreign schema");
        assert_eq!(error.message(), "unsupported native checkpoint version");

        let valid = HarnessCheckpoint {
            schema_version: CHECKPOINT_SCHEMA_VERSION.to_owned(),
            data: state().to_json(),
            ..checkpoint
        };
        assert_eq!(state_from_checkpoint(&valid).expect("decodes"), state());
    }

    #[tokio::test]
    async fn writer_builds_monotonic_checkpoints_without_a_sink() {
        let mut writer = CheckpointWriter::new("openjiuwen-native");
        assert!(writer.latest().is_none());
        assert_eq!(writer.checkpoint_sequence(), 0);
        let first = writer
            .emit(
                None,
                &state(),
                Some("provider-1"),
                CheckpointReason::SessionActivated,
            )
            .await
            .expect("checkpoint");
        assert_eq!(first.provider, "openjiuwen-native");
        assert_eq!(first.schema_version, CHECKPOINT_SCHEMA_VERSION);
        assert_eq!(first.agent_id, "agent-1");
        assert_eq!(first.host_session_id, "host-1");
        assert_eq!(first.sequence, 1);
        assert_eq!(first.provider_session_id.as_deref(), Some("provider-1"));
        assert!(first.revision.is_none());
        assert!(first.checkpoint_id.starts_with("checkpoint-"));
        assert_eq!(
            NativeCheckpointState::from_json(&first.data).expect("data decodes"),
            state()
        );
        assert_eq!(writer.latest(), Some(&first));

        let second = writer
            .emit(None, &state(), None, CheckpointReason::TurnCompleted)
            .await
            .expect("checkpoint");
        assert_eq!(second.sequence, 2);
        assert_ne!(second.checkpoint_id, first.checkpoint_id);
        assert_eq!(writer.checkpoint_sequence(), 2);

        writer.reset(40);
        let third = writer
            .emit(None, &state(), None, CheckpointReason::StateChanged)
            .await
            .expect("checkpoint");
        assert_eq!(third.sequence, 41);
    }

    #[tokio::test]
    async fn writer_records_the_storage_revision_from_a_successful_sink() {
        let sink = RecordingSink::new();
        let dyn_sink: Arc<dyn HarnessCheckpointSink> = sink.clone();
        let mut writer = CheckpointWriter::new("openjiuwen-native");
        let first = writer
            .emit(
                Some(&dyn_sink),
                &state(),
                Some("provider-1"),
                CheckpointReason::SessionActivated,
            )
            .await
            .expect("checkpoint");
        assert_eq!(first.revision, None, "the first write has no expectation");
        let second = writer
            .emit(
                Some(&dyn_sink),
                &state(),
                None,
                CheckpointReason::TurnCompleted,
            )
            .await
            .expect("checkpoint");
        assert_eq!(second.revision.as_deref(), Some("rev-1"));
        assert_eq!(
            sink.expected.lock().expect("lock").clone(),
            vec![None, Some("rev-1".to_owned())]
        );
    }

    #[tokio::test]
    async fn writer_keeps_the_previous_revision_when_the_sink_fails() {
        let failing: Arc<dyn HarnessCheckpointSink> = Arc::new(RecordingSink {
            expected: Mutex::new(Vec::new()),
            fail: true,
        });
        let mut writer = CheckpointWriter::new("openjiuwen-native");
        let checkpoint = writer
            .emit(Some(&failing), &state(), None, CheckpointReason::Periodic)
            .await
            .expect("checkpoint is built even when persistence fails");
        assert_eq!(checkpoint.sequence, 1);
        assert!(checkpoint.revision.is_none());
        assert_eq!(writer.latest(), Some(&checkpoint));

        let next = writer
            .emit(Some(&failing), &state(), None, CheckpointReason::Periodic)
            .await
            .expect("checkpoint");
        assert_eq!(next.sequence, 2);
        assert!(
            next.revision.is_none(),
            "a failed write must not invent a storage revision"
        );
    }
}
