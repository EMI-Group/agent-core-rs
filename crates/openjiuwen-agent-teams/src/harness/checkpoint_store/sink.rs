//! Harness-protocol sink over a [`MemberCheckpointStore`](super::MemberCheckpointStore).
//!
//! The harness publishes an opaque [`HarnessCheckpoint`] envelope; this sink projects it into a
//! [`MemberCheckpointRecord`] (portable payload plus the envelope itself) and persists it under the
//! member's identity. Storage is last-write-wins: the `reason` a checkpoint was published and the
//! protocol's `expected_storage_revision` are accepted for interface compatibility but do not take
//! part in the write, so a stale expected revision never fails a save.
use std::sync::Arc;

use async_trait::async_trait;

use openjiuwen_harness_protocol::{
    json_object_to_value, CheckpointReason, CheckpointSaveReceipt, HarnessCheckpoint,
    HarnessCheckpointSink, HarnessError, HarnessResult, HarnessState,
};

use super::store::{MemberCheckpointRecord, MemberCheckpointStore};
use crate::harness::checkpoint::NativeCheckpoint;

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
    /// `checkpoint.data["state"]` and falls back to [`HarnessState::Idle`] when absent or
    /// unreadable, and its output is the whole `data` object.
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
        let state = checkpoint
            .data
            .get("state")
            .and_then(|value| serde_json::from_value::<HarnessState>(value.clone()).ok())
            .unwrap_or(HarnessState::Idle);
        let mut payload = NativeCheckpoint::new(self.session_id.clone(), self.card_id.clone())
            .with_state(state)
            .with_outputs(json_object_to_value(&checkpoint.data));
        if let Some(cwd) = &self.cwd {
            payload = payload.with_cwd(cwd.clone());
        }
        let record = MemberCheckpointRecord::new(payload).with_protocol(checkpoint.clone());
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

/// The state key read from the envelope's opaque payload.
#[cfg(test)]
const STATE_KEY: &str = "state";

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::{MemberCheckpointSink, STATE_KEY};
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
    }

    #[tokio::test]
    async fn unreadable_state_falls_back_to_idle() {
        let store = Arc::new(RecordingStore::new("rev-1"));
        let sink = MemberCheckpointSink::new(store.clone(), "team:dev-1", "card-1");
        let mut data = JsonObject::new();
        data.insert(STATE_KEY.to_owned(), json!(42));

        sink.save(&envelope(data), CheckpointReason::StateChanged, None)
            .await
            .unwrap();

        assert_eq!(store.saved()[0].checkpoint.state, HarnessState::Idle);
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
        assert_eq!(
            loaded.checkpoint.outputs,
            json!({"answer": "hi", "state": "running"})
        );
    }
}
