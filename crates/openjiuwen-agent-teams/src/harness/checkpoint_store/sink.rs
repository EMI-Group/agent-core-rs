//! Harness-protocol sink over a [`MemberCheckpointStore`](super::MemberCheckpointStore).
//!
//! The harness publishes an opaque [`HarnessCheckpoint`] envelope; this sink projects it into a
//! [`MemberCheckpointRecord`] (portable payload plus the envelope itself) and persists it under the
//! member's identity. Storage is last-write-wins: the `reason` a checkpoint was published and the
//! protocol's `expected_storage_revision` are accepted for interface compatibility but do not take
//! part in the write, so a stale expected revision never fails a save.
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;
use tracing::warn;

use openjiuwen_harness_protocol::{
    json_object_to_value, CheckpointReason, CheckpointSaveReceipt, HarnessCheckpoint,
    HarnessCheckpointSink, HarnessError, HarnessResult, HarnessState, JsonObject,
};

use super::store::{MemberCheckpointRecord, MemberCheckpointStore};
use crate::harness::checkpoint::NativeCheckpoint;

/// The state key read from the envelope's opaque payload.
const STATE_KEY: &str = "state";

/// Projects the envelope's `state` value onto a harness state.
///
/// The first half is always a decodable state, so a restore never fails on checkpoint contents; the
/// second half names the raw value when it was not one this build knows (an absent state is not a
/// degradation and yields `None`), which the caller logs and records on the saved record.
fn decode_state(data: &JsonObject) -> (HarnessState, Option<String>) {
    let Some(value) = data.get(STATE_KEY) else {
        return (HarnessState::Idle, None);
    };
    match serde_json::from_value::<HarnessState>(value.clone()) {
        Ok(state) => (state, None),
        Err(_) => (HarnessState::Idle, Some(render_raw_state(value))),
    }
}

/// Renders an undecodable `state` value: a JSON string verbatim, anything else as its JSON text.
fn render_raw_state(value: &Value) -> String {
    match value.as_str() {
        Some(text) => text.to_owned(),
        None => value.to_string(),
    }
}

/// Adapts a [`MemberCheckpointStore`] plus a member identity to the harness-protocol sink SPI.
///
/// `session_id`/`card_id` identify the persisted record (the member's session and agent card), and
/// `cwd` is the workspace the member ran in, carried into the portable payload so a restored
/// member can be checked against its working directory.
pub struct MemberCheckpointSink {
    store: Arc<dyn MemberCheckpointStore>,
    session_id: String,
    card_id: String,
    cwd: Option<String>,
}

impl MemberCheckpointSink {
    /// Builds a sink writing `store` under one member identity.
    #[must_use]
    pub fn new(
        store: Arc<dyn MemberCheckpointStore>,
        session_id: impl Into<String>,
        card_id: impl Into<String>,
    ) -> Self {
        Self {
            store,
            session_id: session_id.into(),
            card_id: card_id.into(),
            cwd: None,
        }
    }

    /// Pins the working directory recorded in every payload.
    #[must_use]
    pub fn with_cwd(mut self, cwd: impl Into<String>) -> Self {
        self.cwd = Some(cwd.into());
        self
    }

    /// Session identity the records are stored under.
    #[must_use]
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Agent-card identity the records are stored under.
    #[must_use]
    pub fn card_id(&self) -> &str {
        &self.card_id
    }
}

#[async_trait]
impl HarnessCheckpointSink for MemberCheckpointSink {
    /// Projects the envelope into a [`MemberCheckpointRecord`] and persists it.
    ///
    /// `_reason` and `_expected_storage_revision` are accepted and ignored: the store is
    /// last-write-wins, not compare-and-set. The payload's state is decoded from
    /// `checkpoint.data["state"]` and its output is the whole `data` object. An absent state is
    /// [`HarnessState::Idle`]; a state this build does not know is also recorded as `Idle` so the
    /// restore stays tolerant, but it is logged with the member and checkpoint identity and its raw
    /// value lands in [`MemberCheckpointRecord::unrecognised_state`] so the degradation is
    /// observable instead of looking like a fresh idle member.
    ///
    /// # Errors
    ///
    /// Returns [`HarnessError::Protocol`] when the store rejects the write, carrying the store
    /// error's message.
    async fn save(
        &self,
        checkpoint: &HarnessCheckpoint,
        _reason: CheckpointReason,
        _expected_storage_revision: Option<&str>,
    ) -> HarnessResult<CheckpointSaveReceipt> {
        let (state, unrecognised_state) = decode_state(&checkpoint.data);
        if let Some(raw) = &unrecognised_state {
            warn!(
                member = %self.session_id,
                card = %self.card_id,
                provider = %checkpoint.provider,
                checkpoint_id = %checkpoint.checkpoint_id,
                state = %raw,
                "checkpoint state is not a state this build knows; recording it as idle"
            );
        }
        let mut payload = NativeCheckpoint::new(self.session_id.clone(), self.card_id.clone())
            .with_state(state)
            .with_outputs(json_object_to_value(&checkpoint.data));
        if let Some(cwd) = &self.cwd {
            payload = payload.with_cwd(cwd.clone());
        }
        let mut record = MemberCheckpointRecord::new(payload).with_protocol(checkpoint.clone());
        if let Some(raw) = unrecognised_state {
            record = record.with_unrecognised_state(raw);
        }
        let storage_revision = self
            .store
            .save(record)
            .await
            .map_err(|error| HarnessError::protocol(error.to_string()))?;
        Ok(CheckpointSaveReceipt {
            checkpoint_id: checkpoint.checkpoint_id.clone(),
            sequence: checkpoint.sequence,
            storage_revision,
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::{decode_state, MemberCheckpointSink, STATE_KEY};
    use crate::harness::checkpoint_store::{
        InMemoryMemberCheckpointStore, MemberCheckpointRecord, MemberCheckpointStore,
    };
    use async_trait::async_trait;
    use openjiuwen_common::error::{Error, Result};
    use openjiuwen_harness_protocol::{
        CheckpointReason, CheckpointSaveReceipt, HarnessCheckpoint, HarnessCheckpointSink,
        HarnessErrorKind, HarnessState, JsonObject,
    };
    use parking_lot::Mutex;
    use serde_json::json;
    use std::sync::Arc;

    /// A store that records what it was asked to persist and answers a fixed revision.
    struct RecordingStore {
        saved: Mutex<Vec<MemberCheckpointRecord>>,
        revision: String,
        failure: Option<String>,
    }

    impl RecordingStore {
        fn new(revision: impl Into<String>) -> Self {
            Self {
                saved: Mutex::new(Vec::new()),
                revision: revision.into(),
                failure: None,
            }
        }

        fn failing(message: impl Into<String>) -> Self {
            Self {
                failure: Some(message.into()),
                ..Self::new("unused")
            }
        }

        fn saved(&self) -> Vec<MemberCheckpointRecord> {
            self.saved.lock().clone()
        }
    }

    #[async_trait]
    impl MemberCheckpointStore for RecordingStore {
        fn name(&self) -> &'static str {
            "recording"
        }

        async fn save(&self, record: MemberCheckpointRecord) -> Result<String> {
            if let Some(message) = &self.failure {
                return Err(Error::validation(message.clone()));
            }
            self.saved.lock().push(record);
            Ok(self.revision.clone())
        }

        async fn load(
            &self,
            _session_id: &str,
            _card_id: &str,
        ) -> Result<Option<MemberCheckpointRecord>> {
            Ok(None)
        }
    }

    fn envelope(data: JsonObject) -> HarnessCheckpoint {
        HarnessCheckpoint {
            provider: "native".to_owned(),
            schema_version: "1".to_owned(),
            agent_id: "dev-1".to_owned(),
            host_session_id: "team:dev-1".to_owned(),
            checkpoint_id: "cp-7".to_owned(),
            sequence: 7,
            data,
            provider_session_id: None,
            revision: None,
        }
    }

    fn payload(state: &str) -> JsonObject {
        let mut data = JsonObject::new();
        data.insert(STATE_KEY.to_owned(), json!(state));
        data.insert("answer".to_owned(), json!("hi"));
        data
    }

    #[tokio::test]
    async fn persisted_record_carries_payload_and_envelope() {
        let store = Arc::new(RecordingStore::new("rev-7"));
        let sink = MemberCheckpointSink::new(store.clone(), "team:dev-1", "card-1")
            .with_cwd("/work/dev-1");
        assert_eq!(sink.session_id(), "team:dev-1");
        assert_eq!(sink.card_id(), "card-1");

        let checkpoint = envelope(payload("paused"));
        let receipt = sink
            .save(&checkpoint, CheckpointReason::TurnCompleted, Some("rev-6"))
            .await
            .unwrap();
        assert_eq!(
            receipt,
            CheckpointSaveReceipt {
                checkpoint_id: "cp-7".to_owned(),
                sequence: 7,
                storage_revision: "rev-7".to_owned(),
            }
        );

        let saved = store.saved();
        assert_eq!(saved.len(), 1);
        let record = &saved[0];
        assert_eq!(record.checkpoint.session_id, "team:dev-1");
        assert_eq!(record.checkpoint.card_id, "card-1");
        assert_eq!(record.checkpoint.cwd.as_deref(), Some("/work/dev-1"));
        assert_eq!(record.checkpoint.state, HarnessState::Paused);
        assert_eq!(record.checkpoint.turn_id, None);
        assert_eq!(
            record.checkpoint.outputs,
            json!({"answer": "hi", "state": "paused"})
        );
        assert_eq!(record.protocol, Some(checkpoint));
        assert_eq!(record.unrecognised_state, None);
        assert!(!record.has_unrecognised_state());
    }

    #[tokio::test]
    async fn absent_state_falls_back_to_idle_and_cwd_stays_unset() {
        let store = Arc::new(RecordingStore::new("rev-1"));
        let sink = MemberCheckpointSink::new(store.clone(), "team:dev-1", "card-1");
        let mut data = JsonObject::new();
        data.insert("answer".to_owned(), json!("hi"));

        sink.save(&envelope(data), CheckpointReason::Periodic, None)
            .await
            .unwrap();

        let saved = store.saved();
        assert_eq!(saved[0].checkpoint.state, HarnessState::Idle);
        assert_eq!(saved[0].checkpoint.cwd, None);
        assert_eq!(saved[0].checkpoint.outputs, json!({"answer": "hi"}));
        assert_eq!(
            saved[0].unrecognised_state, None,
            "an absent state is not a degradation"
        );
    }

    #[tokio::test]
    async fn unreadable_state_falls_back_to_idle_and_names_the_raw_value() {
        let store = Arc::new(RecordingStore::new("rev-1"));
        let sink = MemberCheckpointSink::new(store.clone(), "team:dev-1", "card-1");
        let mut data = JsonObject::new();
        data.insert(STATE_KEY.to_owned(), json!(42));

        sink.save(&envelope(data), CheckpointReason::StateChanged, None)
            .await
            .unwrap();

        let saved = store.saved();
        assert_eq!(saved[0].checkpoint.state, HarnessState::Idle);
        assert!(saved[0].has_unrecognised_state());
        assert_eq!(saved[0].unrecognised_state.as_deref(), Some("42"));
    }

    #[tokio::test]
    async fn unknown_state_string_stays_idle_and_is_flagged() {
        let store = Arc::new(RecordingStore::new("rev-1"));
        let sink = MemberCheckpointSink::new(store.clone(), "team:dev-1", "card-1");

        sink.save(
            &envelope(payload("frozen")),
            CheckpointReason::TurnCompleted,
            None,
        )
        .await
        .unwrap();

        let saved = store.saved();
        assert_eq!(
            saved[0].checkpoint.state,
            HarnessState::Idle,
            "a foreign state must not fail the restore"
        );
        assert_eq!(
            saved[0].unrecognised_state.as_deref(),
            Some("frozen"),
            "the raw value stays observable"
        );
        assert_eq!(
            saved[0].protocol.as_ref().map(|c| c.checkpoint_id.as_str()),
            Some("cp-7")
        );
    }

    #[tokio::test]
    async fn unknown_state_survives_a_store_round_trip() {
        let store = InMemoryMemberCheckpointStore::shared();
        let sink = MemberCheckpointSink::new(store.clone(), "team:dev-1", "card-1");

        sink.save(
            &envelope(payload("hibernating")),
            CheckpointReason::Periodic,
            None,
        )
        .await
        .unwrap();

        let loaded = store.load("team:dev-1", "card-1").await.unwrap().unwrap();
        assert_eq!(loaded.checkpoint.state, HarnessState::Idle);
        assert_eq!(loaded.unrecognised_state.as_deref(), Some("hibernating"));
        assert!(loaded.has_unrecognised_state());
    }

    #[test]
    fn decode_state_reports_known_absent_and_foreign_values() {
        let mut known = JsonObject::new();
        known.insert(STATE_KEY.to_owned(), json!("pausing"));
        assert_eq!(decode_state(&known), (HarnessState::Pausing, None));

        assert_eq!(decode_state(&JsonObject::new()), (HarnessState::Idle, None));

        let mut foreign = JsonObject::new();
        foreign.insert(STATE_KEY.to_owned(), json!({"state": "running"}));
        assert_eq!(
            decode_state(&foreign),
            (
                HarnessState::Idle,
                Some("{\"state\":\"running\"}".to_owned())
            )
        );

        let mut null = JsonObject::new();
        null.insert(STATE_KEY.to_owned(), json!(null));
        assert_eq!(
            decode_state(&null),
            (HarnessState::Idle, Some("null".to_owned()))
        );
    }

    #[test]
    fn record_with_an_unrecognised_state_deserialises() {
        let record = MemberCheckpointRecord::new(
            crate::harness::checkpoint::NativeCheckpoint::new("team:dev-1", "card-1"),
        )
        .with_unrecognised_state("frozen");
        let encoded = serde_json::to_value(&record).unwrap();
        assert_eq!(encoded["unrecognised_state"], json!("frozen"));
        assert_eq!(encoded["checkpoint"]["state"], json!("idle"));
        let decoded: MemberCheckpointRecord = serde_json::from_value(encoded).unwrap();
        assert_eq!(decoded, record);

        let legacy: MemberCheckpointRecord =
            serde_json::from_value(json!({"checkpoint": {"session_id": "s", "card_id": "c"}}))
                .unwrap();
        assert_eq!(legacy.unrecognised_state, None);

        let tolerant: MemberCheckpointRecord = serde_json::from_value(json!({
            "checkpoint": {"session_id": "s", "card_id": "c"},
            "unrecognised_state": "frozen",
            "future_knob": true,
        }))
        .unwrap();
        assert_eq!(tolerant.unrecognised_state.as_deref(), Some("frozen"));
        assert_eq!(
            serde_json::to_value(MemberCheckpointRecord::new(
                crate::harness::checkpoint::NativeCheckpoint::new("s", "c")
            ))
            .unwrap()
            .get("unrecognised_state"),
            None,
            "a healthy record omits the flag entirely"
        );
    }

    #[tokio::test]
    async fn store_failure_becomes_a_protocol_error() {
        let store = Arc::new(RecordingStore::failing("checkpoint disk full"));
        let sink = MemberCheckpointSink::new(store, "team:dev-1", "card-1");

        let error = sink
            .save(
                &envelope(payload("idle")),
                CheckpointReason::ProviderRequested,
                None,
            )
            .await
            .unwrap_err();
        assert_eq!(error.kind(), HarnessErrorKind::Protocol);
        assert!(error.to_string().contains("checkpoint disk full"));
    }

    #[tokio::test]
    async fn sink_round_trips_through_the_memory_store() {
        let store = InMemoryMemberCheckpointStore::shared();
        let sink = MemberCheckpointSink::new(store.clone(), "team:dev-1", "card-1");
        let checkpoint = envelope(payload("running"));

        let receipt = sink
            .save(&checkpoint, CheckpointReason::TurnCompleted, None)
            .await
            .unwrap();
        assert_eq!(receipt.storage_revision, "memory-1");
        let loaded = store.load("team:dev-1", "card-1").await.unwrap().unwrap();
        assert_eq!(loaded.protocol, Some(checkpoint));
        assert_eq!(loaded.checkpoint.state, HarnessState::Running);
        assert_eq!(loaded.unrecognised_state, None);
        assert_eq!(
            loaded.checkpoint.outputs,
            json!({"answer": "hi", "state": "running"})
        );
    }
}
