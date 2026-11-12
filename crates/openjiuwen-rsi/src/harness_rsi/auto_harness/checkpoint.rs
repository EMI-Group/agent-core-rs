//! Durable session state for resumable auto-harness sessions.
//!
//! Python's orchestrator keeps results on `self` and relies on a live process;
//! Rust checkpoints the whole session
//! ([`SessionState`]) after every iteration so a run can be restored instead of
//! replayed. The state is the single checkpoint unit — candidates, history,
//! counters, and the stop reason — because every derived value (best score,
//! iteration number) can be recomputed from it.
//!
//! Three backends implement [`CheckpointStore`]:
//!
//! - [`InMemoryCheckpointStore`] — the reference backend for tests and fakes.
//! - [`KvCheckpointStore`] — JSON text under `<prefix><session_id>` in any
//!   `KvStore` (memory, local files, `SQLite`).
//! - [`SessionCheckpointStore`] — a component lane of an `openjiuwen-session`
//!   `Session`, so an RSI session shares the surrounding run's session state.
//!
//! Every failure surfaces as [`Error::session`] rather than being swallowed: a
//! checkpoint that cannot be written would otherwise let a resumed session
//! silently repeat work the caller believes was persisted.

use std::collections::BTreeMap;
use std::sync::{Arc, RwLock};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use openjiuwen_common::error::{Error, Result};
use openjiuwen_core::foundation::store::kv::{KvStore, KvValue};
use openjiuwen_core::session::Session;

use crate::evolution::{Candidate, CandidateId};
use crate::harness_rsi::auto_harness::task::{IterationRecord, StopReason};

/// The complete state of one auto-harness session.
///
/// A full JSON round-trip is a resume: restoring `iteration`, `candidates`, and
/// `history` puts the orchestrator exactly where the previous process stopped.
/// Field names are the persistence contract (`snake_case`), unknown fields are
/// tolerated, and missing fields fall back to [`SessionState::default`] so an
/// older checkpoint still loads.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct SessionState {
    /// Session identifier.
    pub session_id: String,
    /// Number of completed iterations.
    pub iteration: u32,
    /// Whether the session is currently running an iteration.
    pub running: bool,
    /// Incumbent best candidate.
    pub best_candidate: Option<CandidateId>,
    /// Incumbent best score.
    pub best_score: Option<f64>,
    /// Every candidate the session has seen, in proposal order.
    pub candidates: Vec<Candidate>,
    /// One record per completed iteration, oldest first.
    pub history: Vec<IterationRecord>,
    /// Cost charged against the budget so far, in US dollars.
    pub spent_usd: f64,
    /// Why the session stopped; `None` while it can continue.
    pub stop_reason: Option<StopReason>,
}

impl SessionState {
    /// Creates the empty state of a session that has not run an iteration.
    #[must_use]
    pub fn new(session_id: impl Into<String>) -> Self {
        Self {
            session_id: session_id.into(),
            ..Self::default()
        }
    }

    /// Returns whether the persisted session has already stopped.
    ///
    /// A finished session carries a [`SessionState::stop_reason`]; the
    /// orchestrator refuses to run further iterations in that case.
    #[must_use]
    pub fn is_finished(&self) -> bool {
        self.stop_reason.is_some()
    }
}

/// Persists and restores [`SessionState`] by session id.
///
/// # Deviation from the Python surface
///
/// Python keeps its checkpoint on the orchestrator object, so there is no load
/// error to model. The Rust trait returns [`Result<Option<SessionState>>`] from
/// [`CheckpointStore::load`]: `Ok(None)` means "no checkpoint for this session"
/// while `Err(_)` reports a backend failure (I/O, malformed JSON, a poisoned
/// lock), which a real store can always hit and must not disguise as "empty".
///
/// # Errors
///
/// Every method returns [`Error::session`] when the backend cannot complete the
/// operation.
#[async_trait]
pub trait CheckpointStore: Send + Sync {
    /// Writes `state`, replacing any checkpoint stored for its session id.
    ///
    /// # Errors
    ///
    /// Returns [`Error::session`] when serialization or the backend write fails.
    async fn save(&self, state: &SessionState) -> Result<()>;

    /// Reads the checkpoint of `session_id`, or `None` when there is none.
    ///
    /// # Errors
    ///
    /// Returns [`Error::session`] when the backend read fails or the stored
    /// document cannot be deserialized.
    async fn load(&self, session_id: &str) -> Result<Option<SessionState>>;

    /// Removes the checkpoint of `session_id`; a missing checkpoint is not an
    /// error.
    ///
    /// # Errors
    ///
    /// Returns [`Error::session`] when the backend delete fails.
    async fn delete(&self, session_id: &str) -> Result<()>;
}

/// A [`CheckpointStore`] keeping states in process memory.
///
/// Suitable for tests, fakes, and single-process runs; the map is behind an
/// [`RwLock`] that is never held across an `await`. `save` replaces the entry
/// for the session id, `load` clones it, and `delete` is a no-op for an unknown
/// id.
#[derive(Debug, Default)]
pub struct InMemoryCheckpointStore {
    /// Session id → state.
    states: RwLock<BTreeMap<String, SessionState>>,
}

impl InMemoryCheckpointStore {
    /// Creates an empty store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the lock guard over the stored states, mapping poisoning to
    /// [`Error::session`].
    fn states(&self) -> Result<std::sync::RwLockReadGuard<'_, BTreeMap<String, SessionState>>> {
        self.states
            .read()
            .map_err(|_| Error::session("in-memory checkpoint store lock poisoned on read"))
    }
}

#[async_trait]
impl CheckpointStore for InMemoryCheckpointStore {
    async fn save(&self, state: &SessionState) -> Result<()> {
        self.states
            .write()
            .map_err(|_| Error::session("in-memory checkpoint store lock poisoned on write"))?
            .insert(state.session_id.clone(), state.clone());
        Ok(())
    }

    async fn load(&self, session_id: &str) -> Result<Option<SessionState>> {
        Ok(self.states()?.get(session_id).cloned())
    }

    async fn delete(&self, session_id: &str) -> Result<()> {
        self.states
            .write()
            .map_err(|_| Error::session("in-memory checkpoint store lock poisoned on write"))?
            .remove(session_id);
        Ok(())
    }
}

/// The default key prefix used by [`KvCheckpointStore`].
pub const DEFAULT_KEY_PREFIX: &str = "rsi:auto_harness:";

/// A [`CheckpointStore`] storing each state as JSON text in a `KvStore`.
///
/// The document lives under `<prefix><session_id>` (default prefix
/// [`DEFAULT_KEY_PREFIX`]), so all sessions of one store share a namespace that
/// `KvStore::get_by_prefix` can scan.
pub struct KvCheckpointStore {
    /// Backing key-value store.
    store: Arc<dyn KvStore>,
    /// Prefix prepended to every session id.
    key_prefix: String,
}

// `Arc<dyn KvStore>` is not `Debug`, so the store is summarized by its prefix.
impl std::fmt::Debug for KvCheckpointStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KvCheckpointStore")
            .field("key_prefix", &self.key_prefix)
            .finish_non_exhaustive()
    }
}

impl KvCheckpointStore {
    /// Wraps `store`, using [`DEFAULT_KEY_PREFIX`].
    #[must_use]
    pub fn new(store: Arc<dyn KvStore>) -> Self {
        Self::with_key_prefix(store, DEFAULT_KEY_PREFIX)
    }

    /// Wraps `store`, using `prefix` for every key.
    #[must_use]
    pub fn with_key_prefix(store: Arc<dyn KvStore>, prefix: impl Into<String>) -> Self {
        Self {
            store,
            key_prefix: prefix.into(),
        }
    }

    /// Returns the key holding `session_id`'s checkpoint.
    #[must_use]
    pub fn key(&self, session_id: &str) -> String {
        format!("{}{session_id}", self.key_prefix)
    }

    /// Returns the key prefix of this store.
    #[must_use]
    pub fn key_prefix(&self) -> &str {
        &self.key_prefix
    }

    /// Returns the backing store.
    #[must_use]
    pub fn store(&self) -> &Arc<dyn KvStore> {
        &self.store
    }
}

#[async_trait]
impl CheckpointStore for KvCheckpointStore {
    async fn save(&self, state: &SessionState) -> Result<()> {
        let key = self.key(&state.session_id);
        let encoded = serde_json::to_string(state).map_err(|err| {
            Error::session(format!("failed to serialize checkpoint {key}: {err}"))
        })?;
        self.store
            .set(&key, KvValue::Text(encoded))
            .await
            .map_err(|err| Error::session(format!("failed to write checkpoint {key}: {err}")))
    }

    async fn load(&self, session_id: &str) -> Result<Option<SessionState>> {
        let key = self.key(session_id);
        let Some(value) = self
            .store
            .get(&key)
            .await
            .map_err(|err| Error::session(format!("failed to read checkpoint {key}: {err}")))?
        else {
            return Ok(None);
        };
        let text = value.as_str().ok_or_else(|| {
            Error::session(format!("checkpoint {key} holds bytes, not JSON text"))
        })?;
        serde_json::from_str(text)
            .map(Some)
            .map_err(|err| Error::session(format!("failed to parse checkpoint {key}: {err}")))
    }

    async fn delete(&self, session_id: &str) -> Result<()> {
        let key = self.key(session_id);
        self.store
            .delete(&key)
            .await
            .map_err(|err| Error::session(format!("failed to delete checkpoint {key}: {err}")))
    }
}

/// The default component id used by [`SessionCheckpointStore`].
pub const DEFAULT_COMPONENT_ID: &str = "rsi_auto_harness";

/// A [`CheckpointStore`] keeping the state in a `Session` component lane.
///
/// The state is stored under the session's component id (default
/// [`DEFAULT_COMPONENT_ID`]), so an RSI session persists alongside the run that
/// owns it and reappears when the surrounding session is checkpointed or
/// resumed.
pub struct SessionCheckpointStore {
    /// Session holding the component lane.
    session: Arc<dyn Session>,
    /// Component id (a nested path) the state lives under.
    component_id: String,
}

// `Arc<dyn Session>` is not `Debug`, so the store is summarized by its component id.
impl std::fmt::Debug for SessionCheckpointStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SessionCheckpointStore")
            .field("component_id", &self.component_id)
            .finish_non_exhaustive()
    }
}

impl SessionCheckpointStore {
    /// Wraps `session`, using [`DEFAULT_COMPONENT_ID`].
    #[must_use]
    pub fn new(session: Arc<dyn Session>) -> Self {
        Self::with_component_id(session, DEFAULT_COMPONENT_ID)
    }

    /// Wraps `session`, storing the state under `component_id`.
    #[must_use]
    pub fn with_component_id(session: Arc<dyn Session>, component_id: impl Into<String>) -> Self {
        Self {
            session,
            component_id: component_id.into(),
        }
    }

    /// Returns the component id the state is stored under.
    #[must_use]
    pub fn component_id(&self) -> &str {
        &self.component_id
    }

    /// Returns the backing session.
    #[must_use]
    pub fn session(&self) -> &Arc<dyn Session> {
        &self.session
    }
}

#[async_trait]
impl CheckpointStore for SessionCheckpointStore {
    async fn save(&self, state: &SessionState) -> Result<()> {
        let patch = serde_json::to_value(state).map_err(|err| {
            Error::session(format!(
                "failed to serialize checkpoint of session {}: {err}",
                state.session_id
            ))
        })?;
        self.session
            .update_component_state(&self.component_id, patch)
            .await
            .map_err(|err| {
                Error::session(format!(
                    "failed to write component state {}: {err}",
                    self.component_id
                ))
            })
    }

    async fn load(&self, session_id: &str) -> Result<Option<SessionState>> {
        let stored = self
            .session
            .get_component_state(&self.component_id)
            .await
            .map_err(|err| {
                Error::session(format!(
                    "failed to read component state {}: {err}",
                    self.component_id
                ))
            })?;
        match stored {
            None => Ok(None),
            // `delete` writes an empty patch (sessions have no delete API), so an
            // empty component lane means "no checkpoint" rather than a state with
            // every field defaulted.
            Some(Value::Object(map)) if map.is_empty() => Ok(None),
            Some(value) => serde_json::from_value(value).map(Some).map_err(|err| {
                Error::session(format!(
                    "failed to parse component state {} for session {session_id}: {err}",
                    self.component_id
                ))
            }),
        }
    }

    async fn delete(&self, session_id: &str) -> Result<()> {
        // A `Session` has no delete API for component lanes, so the state is
        // overwritten with the empty object; `load` treats that as absent.
        self.session
            .update_component_state(&self.component_id, Value::Object(Map::new()))
            .await
            .map_err(|err| {
                Error::session(format!(
                    "failed to clear component state {} for session {session_id}: {err}",
                    self.component_id
                ))
            })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::Mutex;

    use serde_json::json;

    use super::*;
    use crate::evolution::{CaseResult, EvaluationReport};
    use openjiuwen_common::error::ErrorKind;
    use openjiuwen_core::foundation::store::kv::{
        InMemoryKvStore, KvStore, KvValue, LocalFileKvStore,
    };
    use openjiuwen_core::session::{
        InteractionRequest, InteractionResponse, NoopTraceSink, Session, SessionType, StreamItem,
        TraceSink,
    };

    fn report(score: f64) -> EvaluationReport {
        EvaluationReport::from_results(vec![CaseResult {
            case_id: "case-1".to_owned(),
            passed: true,
            score: Some(score),
            weight: 1.0,
            error: None,
            details: json!({"note": "graded"}),
        }])
    }

    fn state() -> SessionState {
        let mut candidate = Candidate::new("seed/g1/v0", "model: demo");
        candidate.parent = Some(CandidateId::new("seed"));
        candidate.generation = 1;
        candidate.evaluation = Some(report(0.75));
        candidate
            .metadata
            .insert("origin".to_owned(), json!("unit-test"));

        SessionState {
            session_id: "sess-1".to_owned(),
            iteration: 1,
            running: false,
            best_candidate: Some(CandidateId::new("seed/g1/v0")),
            best_score: Some(0.75),
            candidates: vec![Candidate::new("seed", "model: demo-base"), candidate],
            history: vec![IterationRecord {
                iteration: 1,
                parents: vec![CandidateId::new("seed")],
                proposed: 2,
                evaluated: 2,
                valid: 2,
                best_candidate: Some(CandidateId::new("seed/g1/v0")),
                best_score: Some(0.75),
                previous_best_score: None,
                improvement: 0.75,
                accepted: true,
                duration_ms: 5,
                stop: Some(StopReason::Converged),
            }],
            spent_usd: 0.5,
            stop_reason: Some(StopReason::Converged),
        }
    }

    #[test]
    fn new_state_is_empty_and_unfinished() {
        let state = SessionState::new("sess-9");
        assert_eq!(state.session_id, "sess-9");
        assert_eq!(state.iteration, 0);
        assert!(!state.running);
        assert!(state.candidates.is_empty(), "no candidates yet");
        assert!(state.history.is_empty(), "no history yet");
        assert!(!state.is_finished());
        assert_eq!(SessionState::default().session_id, "");
    }

    #[test]
    fn finished_requires_a_stop_reason() {
        let mut state = SessionState::new("sess-9");
        state.running = true;
        assert!(!state.is_finished());
        state.stop_reason = Some(StopReason::MaxIterations);
        assert!(state.is_finished());
    }

    #[test]
    fn state_round_trips_through_json() {
        let state = state();
        let encoded = serde_json::to_string(&state).expect("state serializes");
        let decoded: SessionState = serde_json::from_str(&encoded).expect("state parses");
        assert_eq!(decoded, state);
        assert_eq!(decoded.history.len(), 1);
        assert_eq!(decoded.candidates.len(), 2);
        assert_eq!(decoded.stop_reason, Some(StopReason::Converged));
        assert_eq!(
            decoded.candidates[1]
                .evaluation
                .as_ref()
                .and_then(EvaluationReport::score),
            Some(0.75)
        );
    }

    #[test]
    fn state_tolerates_missing_and_unknown_fields() {
        let decoded: SessionState = serde_json::from_value(json!({
            "session_id": "sess-2",
            "iteration": 4,
            "future_field": {"nested": true}
        }))
        .expect("state tolerates unknown fields");
        assert_eq!(decoded.session_id, "sess-2");
        assert_eq!(decoded.iteration, 4);
        assert!(!decoded.running);
        assert!(decoded.best_candidate.is_none());
        assert_eq!(decoded.spent_usd, 0.0);
    }

    #[tokio::test]
    async fn in_memory_store_saves_loads_and_deletes() {
        let store = InMemoryCheckpointStore::new();
        assert!(store.load("sess-1").await.expect("load works").is_none());

        let state = state();
        store.save(&state).await.expect("save works");
        assert_eq!(
            store.load("sess-1").await.expect("load works"),
            Some(state.clone())
        );

        let mut updated = state.clone();
        updated.iteration = 2;
        store.save(&updated).await.expect("save replaces");
        assert_eq!(
            store.load("sess-1").await.expect("load works"),
            Some(updated)
        );

        store.delete("sess-1").await.expect("delete works");
        assert!(store.load("sess-1").await.expect("load works").is_none());
        store
            .delete("sess-1")
            .await
            .expect("deleting a missing checkpoint is a no-op");
    }

    #[tokio::test]
    async fn kv_store_round_trips_over_in_memory_backend() {
        let kv: Arc<dyn KvStore> = Arc::new(InMemoryKvStore::new());
        let store = KvCheckpointStore::new(kv.clone());

        assert_eq!(store.key("sess-1"), "rsi:auto_harness:sess-1");
        assert_eq!(store.key_prefix(), DEFAULT_KEY_PREFIX);
        assert!(store.load("sess-1").await.expect("load works").is_none());

        let state = state();
        store.save(&state).await.expect("save works");
        assert_eq!(
            store.load("sess-1").await.expect("load works"),
            Some(state.clone())
        );

        let stored = kv
            .get("rsi:auto_harness:sess-1")
            .await
            .expect("raw read works")
            .expect("key exists");
        let text = stored.as_str().expect("checkpoint is text");
        assert!(text.contains("\"session_id\":\"sess-1\""));

        store.delete("sess-1").await.expect("delete works");
        assert!(store.load("sess-1").await.expect("load works").is_none());
        assert!(!kv
            .exists("rsi:auto_harness:sess-1")
            .await
            .expect("exists works"));
    }

    #[tokio::test]
    async fn kv_store_round_trips_over_local_file_backend() {
        let dir = tempfile::tempdir().expect("tempdir");
        let kv: Arc<dyn KvStore> = Arc::new(
            LocalFileKvStore::new(dir.path())
                .await
                .expect("file store opens"),
        );
        let store = KvCheckpointStore::with_key_prefix(kv, "custom:");

        assert_eq!(store.key("sess-2"), "custom:sess-2");
        assert_eq!(store.key_prefix(), "custom:");

        let mut state = state();
        state.session_id = "sess-2".to_owned();
        store.save(&state).await.expect("save works");
        assert_eq!(store.load("sess-2").await.expect("load works"), Some(state));
        store.delete("sess-2").await.expect("delete works");
        assert!(store.load("sess-2").await.expect("load works").is_none());
    }

    #[tokio::test]
    async fn kv_store_maps_malformed_json_to_a_session_error() {
        let kv: Arc<dyn KvStore> = Arc::new(InMemoryKvStore::new());
        let store = KvCheckpointStore::new(kv.clone());
        kv.set("rsi:auto_harness:broken", KvValue::text("{not json"))
            .await
            .expect("raw write works");

        let error = store
            .load("broken")
            .await
            .expect_err("malformed checkpoint is an error");
        assert_eq!(error.kind(), ErrorKind::Session);
        assert!(error.to_string().contains("rsi:auto_harness:broken"));
    }

    #[tokio::test]
    async fn kv_store_rejects_a_byte_valued_checkpoint() {
        let kv: Arc<dyn KvStore> = Arc::new(InMemoryKvStore::new());
        let store = KvCheckpointStore::new(kv.clone());
        kv.set("rsi:auto_harness:bytes", KvValue::bytes(vec![0, 1, 2]))
            .await
            .expect("raw write works");

        let error = store
            .load("bytes")
            .await
            .expect_err("byte checkpoint is an error");
        assert_eq!(error.kind(), ErrorKind::Session);
    }

    /// Minimal in-memory session: a global lane and component lanes, both
    /// replaced wholesale (simpler than the real deep merge, which is not under
    /// test here).
    struct FakeSession {
        id: String,
        global: Mutex<Value>,
        components: Mutex<BTreeMap<String, Value>>,
        sink: NoopTraceSink,
    }

    impl FakeSession {
        fn new(id: &str) -> Self {
            Self {
                id: id.to_owned(),
                global: Mutex::new(Value::Object(Map::new())),
                components: Mutex::new(BTreeMap::new()),
                sink: NoopTraceSink::new(),
            }
        }

        fn set_component(&self, component_id: &str, value: Value) {
            self.components
                .lock()
                .expect("components lock")
                .insert(component_id.to_owned(), value);
        }
    }

    #[async_trait]
    impl Session for FakeSession {
        fn id(&self) -> &str {
            &self.id
        }

        fn parent_id(&self) -> Option<&str> {
            None
        }

        fn session_type(&self) -> SessionType {
            SessionType::Workflow
        }

        async fn get_state(&self, _key: &str) -> Result<Option<Value>> {
            Ok(Some(self.global.lock().expect("global lock").clone()))
        }

        async fn update_state(&self, patch: Value) -> Result<()> {
            *self.global.lock().expect("global lock") = patch;
            Ok(())
        }

        async fn get_component_state(&self, component_id: &str) -> Result<Option<Value>> {
            Ok(self
                .components
                .lock()
                .expect("components lock")
                .get(component_id)
                .cloned())
        }

        async fn update_component_state(&self, component_id: &str, patch: Value) -> Result<()> {
            self.set_component(component_id, patch);
            Ok(())
        }

        async fn write_stream(&self, _item: StreamItem) -> Result<()> {
            Ok(())
        }

        async fn request_interaction(
            &self,
            _request: InteractionRequest,
        ) -> Result<InteractionResponse> {
            Err(Error::unsupported("fake session has no interaction broker"))
        }

        fn trace(&self) -> &dyn TraceSink {
            &self.sink
        }
    }

    #[tokio::test]
    async fn session_store_round_trips_and_uses_the_component_id() {
        let session = Arc::new(FakeSession::new("sess-3"));
        let store = SessionCheckpointStore::new(session.clone());

        assert_eq!(store.component_id(), "rsi_auto_harness");
        assert_eq!(store.component_id(), DEFAULT_COMPONENT_ID);
        assert!(store.load("sess-3").await.expect("load works").is_none());

        let mut state = state();
        state.session_id = "sess-3".to_owned();
        store.save(&state).await.expect("save works");

        assert_eq!(
            session
                .get_component_state("rsi_auto_harness")
                .await
                .expect("read works"),
            Some(serde_json::to_value(&state).expect("state serializes"))
        );
        assert_eq!(store.load("sess-3").await.expect("load works"), Some(state));

        store.delete("sess-3").await.expect("delete works");
        assert!(store.load("sess-3").await.expect("load works").is_none());
    }

    #[tokio::test]
    async fn session_store_honours_a_custom_component_id() {
        let session = Arc::new(FakeSession::new("sess-4"));
        let store = SessionCheckpointStore::with_component_id(session.clone(), "nested.rsi");

        assert_eq!(store.component_id(), "nested.rsi");
        store
            .save(&SessionState::new("sess-4"))
            .await
            .expect("save works");
        assert!(session
            .get_component_state("nested.rsi")
            .await
            .expect("read works")
            .is_some());
        assert!(session
            .get_component_state(DEFAULT_COMPONENT_ID)
            .await
            .expect("read works")
            .is_none());
    }

    #[tokio::test]
    async fn session_store_maps_malformed_state_to_a_session_error() {
        let session = Arc::new(FakeSession::new("sess-5"));
        session.set_component("rsi_auto_harness", json!({"iteration": "not a number"}));
        let store = SessionCheckpointStore::new(session);

        let error = store
            .load("sess-5")
            .await
            .expect_err("malformed component state is an error");
        assert_eq!(error.kind(), ErrorKind::Session);
    }

    #[tokio::test]
    async fn session_store_propagates_session_failures_as_session_errors() {
        /// A session that refuses every component write.
        struct WriteRefusingSession {
            inner: FakeSession,
        }

        #[async_trait]
        impl Session for WriteRefusingSession {
            fn id(&self) -> &str {
                self.inner.id()
            }

            fn parent_id(&self) -> Option<&str> {
                self.inner.parent_id()
            }

            fn session_type(&self) -> SessionType {
                self.inner.session_type()
            }

            async fn get_state(&self, key: &str) -> Result<Option<Value>> {
                self.inner.get_state(key).await
            }

            async fn update_state(&self, patch: Value) -> Result<()> {
                self.inner.update_state(patch).await
            }

            async fn get_component_state(&self, component_id: &str) -> Result<Option<Value>> {
                self.inner.get_component_state(component_id).await
            }

            async fn update_component_state(
                &self,
                _component_id: &str,
                _patch: Value,
            ) -> Result<()> {
                Err(Error::session("store is read-only"))
            }

            async fn write_stream(&self, item: StreamItem) -> Result<()> {
                self.inner.write_stream(item).await
            }

            async fn request_interaction(
                &self,
                request: InteractionRequest,
            ) -> Result<InteractionResponse> {
                self.inner.request_interaction(request).await
            }

            fn trace(&self) -> &dyn TraceSink {
                self.inner.trace()
            }
        }

        let session: Arc<dyn Session> = Arc::new(WriteRefusingSession {
            inner: FakeSession::new("sess-6"),
        });
        let store = SessionCheckpointStore::new(session);

        let error = store
            .save(&SessionState::new("sess-6"))
            .await
            .expect_err("write failure surfaces");
        assert_eq!(error.kind(), ErrorKind::Session);
        assert!(error.to_string().contains("read-only"));
    }
}
