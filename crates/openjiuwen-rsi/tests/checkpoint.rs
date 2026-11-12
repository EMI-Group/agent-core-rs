#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Offline checkpoint/resume integration suite for `openjiuwen-rsi`.
//!
//! These tests drive the public checkpoint surface only: [`SessionState`] as a
//! JSON document, the in-memory and key-value [`CheckpointStore`] backends, and
//! an orchestrator resuming a persisted session.
//!
//! Python parity: `openjiuwen/rsi/harness_rsi/auto_harness` keeps its session on
//! the live orchestrator object and has no reload path, so the Rust persistence
//! contract (`snake_case` JSON, tolerant defaults, `Ok(None)` for "no
//! checkpoint") is verified here instead of against a Python test file.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde_json::json;

use openjiuwen_core::foundation::store::kv::{InMemoryKvStore, KvStore, KvValue, LocalFileKvStore};
use openjiuwen_rsi::{
    AutoHarnessConfig, AutoHarnessOrchestrator, Candidate, CandidateId, CandidateStatus,
    CaseResult, CheckpointStore, ErrorKind, EvaluationReport, HarnessRunOutcome, HarnessRunRequest,
    HarnessRunner, InMemoryCheckpointStore, IterationRecord, KvCheckpointStore, Result,
    SessionState, StopReason,
};

/// Session id shared by the resume tests; the checkpoint key is derived from it.
const RESUME_SESSION_ID: &str = "sess-resume";

/// A [`HarnessRunner`] that answers with queued scores and logs every request.
///
/// The orchestrator only ever sees the resulting [`EvaluationReport`], so a
/// scripted fake is all an offline checkpoint test needs.
struct ScriptedRunner {
    /// Scores returned by successive runs, oldest first.
    outcomes: Mutex<VecDeque<f64>>,
    /// Every request the orchestrator issued, in call order.
    requests: Mutex<Vec<HarnessRunRequest>>,
}

impl ScriptedRunner {
    /// Creates a runner that answers with `scores` in order.
    fn new(scores: Vec<f64>) -> Self {
        Self {
            outcomes: Mutex::new(scores.into_iter().collect()),
            requests: Mutex::new(Vec::new()),
        }
    }

    /// Returns a copy of the logged requests.
    fn requests(&self) -> Vec<HarnessRunRequest> {
        self.requests
            .lock()
            .expect("request log lock")
            .iter()
            .cloned()
            .collect()
    }
}

#[async_trait]
impl HarnessRunner for ScriptedRunner {
    fn name(&self) -> &'static str {
        "scripted-checkpoint"
    }

    async fn run(&self, request: HarnessRunRequest) -> Result<HarnessRunOutcome> {
        let score = {
            let mut outcomes = self.outcomes.lock().expect("outcome queue lock");
            outcomes
                .pop_front()
                .expect("scripted runner needs one score per run")
        };
        self.requests
            .lock()
            .expect("request log lock")
            .push(request);
        Ok(HarnessRunOutcome {
            report: graded_report(score),
            ..HarnessRunOutcome::default()
        })
    }
}

/// Returns a one-case report scoring `score`, so the orchestrator sees a graded
/// candidate (an ungraded report would never become the incumbent).
fn graded_report(score: f64) -> EvaluationReport {
    EvaluationReport::from_results(vec![CaseResult {
        case_id: "case-1".to_owned(),
        passed: true,
        score: Some(score),
        weight: 1.0,
        error: None,
        details: json!({"note": "graded"}),
    }])
}

/// Returns a configuration with the one field `validate()` requires set, and
/// every other value at the Python default.
fn config() -> AutoHarnessConfig {
    AutoHarnessConfig {
        data_dir: PathBuf::from("rsi-checkpoint-tests"),
        ..AutoHarnessConfig::default()
    }
}

/// Builds the rich state used by the round-trip and file-backend tests.
///
/// It mirrors a session mid-run that just converged: two iterations of history,
/// a graded and a selected candidate, a persisted spend, and the terminal
/// reason, so every optional checkpoint field is exercised at once.
fn rich_state() -> SessionState {
    let seed = Candidate {
        status: CandidateStatus::Selected,
        metadata: [("note".to_owned(), json!("seed"))].into_iter().collect(),
        ..Candidate::new("seed", "# rsi auto_harness seed\n")
    };

    let mut graded = Candidate::new(
        "seed/g1/v0",
        "# rsi auto_harness seed\n# rsi variant g1-0\n",
    );
    graded.parent = Some(CandidateId::new("seed"));
    graded.generation = 1;
    "appended rsi variant marker".clone_into(&mut graded.change_summary);
    graded.status = CandidateStatus::Evaluated;
    graded.evaluation = Some(graded_report(0.75));

    SessionState {
        session_id: "sess-rich".to_owned(),
        iteration: 2,
        running: false,
        best_candidate: Some(CandidateId::new("seed/g1/v0")),
        best_score: Some(0.75),
        candidates: vec![seed, graded],
        history: vec![
            IterationRecord {
                iteration: 1,
                parents: vec![CandidateId::new("seed")],
                proposed: 1,
                evaluated: 1,
                valid: 1,
                best_candidate: Some(CandidateId::new("seed/g1/v0")),
                best_score: Some(0.5),
                previous_best_score: None,
                improvement: 0.0,
                accepted: true,
                duration_ms: 7,
                stop: None,
            },
            IterationRecord {
                iteration: 2,
                parents: vec![CandidateId::new("seed/g1/v0")],
                proposed: 1,
                evaluated: 1,
                valid: 1,
                best_candidate: Some(CandidateId::new("seed/g1/v0")),
                best_score: Some(0.75),
                previous_best_score: Some(0.5),
                improvement: 0.25,
                accepted: true,
                duration_ms: 9,
                stop: Some(StopReason::Converged),
            },
        ],
        spent_usd: 0.5,
        stop_reason: Some(StopReason::Converged),
    }
}

// 1 — the checkpoint document itself.
// Python has no persisted state object; the Rust contract is that the whole
// session (candidates, history, counters, stop reason) is one JSON document.
#[test]
fn session_state_round_trips_and_defaults_tolerantly() {
    let state = rich_state();
    let encoded = serde_json::to_string(&state).expect("SessionState serializes");
    assert!(encoded.contains("\"stop_reason\":\"converged\""));
    assert!(encoded.contains("\"best_candidate\":\"seed/g1/v0\""));
    assert!(encoded.contains("\"session_id\":\"sess-rich\""));

    let restored: SessionState = serde_json::from_str(&encoded).expect("SessionState deserializes");
    assert_eq!(restored, state);
    assert_eq!(restored.history.len(), 2);
    assert_eq!(restored.history[1].stop, Some(StopReason::Converged));
    assert_eq!(
        restored.candidates[1]
            .evaluation
            .as_ref()
            .and_then(EvaluationReport::score),
        Some(0.75)
    );

    // A sparse document (older checkpoint) falls back to `Default` per field.
    let sparse: SessionState =
        serde_json::from_str(r#"{"session_id":"s"}"#).expect("sparse state loads");
    assert_eq!(sparse.session_id, "s");
    assert_eq!(sparse.iteration, 0);
    assert!(!sparse.running);
    assert_eq!(sparse.candidates, Vec::<Candidate>::new());
    assert_eq!(sparse.history, Vec::<IterationRecord>::new());
    assert!(sparse.best_candidate.is_none());
    assert!(!sparse.is_finished());

    // Unknown fields (a newer document) are tolerated, not rejected.
    let tolerant: SessionState = serde_json::from_str(
        r#"{"session_id":"s2","iteration":3,"future_field":{"nested":[1,2]}}"#,
    )
    .expect("unknown fields are tolerated");
    assert_eq!(tolerant.session_id, "s2");
    assert_eq!(tolerant.iteration, 3);

    let fresh = SessionState::new("x");
    assert_eq!(fresh.session_id, "x");
    assert_eq!(fresh.iteration, 0);
    assert!(!fresh.is_finished());
    assert!(!fresh.running);
}

// 2 — the reference backend.
// `Ok(None)` is "no checkpoint", never an error: resuming a session that never
// ran must be a no-op rather than a failure.
#[tokio::test]
async fn in_memory_store_saves_replaces_and_deletes() {
    let store = InMemoryCheckpointStore::new();
    assert!(store
        .load("no-such-session")
        .await
        .expect("load of an unknown id is Ok")
        .is_none());

    let state = rich_state();
    store.save(&state).await.expect("save works");
    assert_eq!(
        store.load(&state.session_id).await.expect("load works"),
        Some(state.clone())
    );

    let mut replaced = state.clone();
    replaced.iteration = 5;
    replaced.best_score = Some(0.99);
    replaced.spent_usd = 1.25;
    store.save(&replaced).await.expect("second save works");
    assert_eq!(
        store
            .load(&state.session_id)
            .await
            .expect("load after replace"),
        Some(replaced)
    );

    store.delete(&state.session_id).await.expect("delete works");
    assert!(store
        .load(&state.session_id)
        .await
        .expect("load after delete")
        .is_none());
    // Deleting a missing checkpoint is not an error (Python re-runs bootstrap).
    store
        .delete(&state.session_id)
        .await
        .expect("delete is idempotent");
}

// 3 — the key-value backend shares one namespace keyed by session id.
// Python's auto_harness stores per-session JSON under its data dir; the KV
// store reproduces that as `<prefix><session_id>` holding JSON text.
#[tokio::test]
async fn kv_store_uses_prefixed_json_text_keys() {
    let kv: Arc<dyn KvStore> = Arc::new(InMemoryKvStore::new());
    let store = KvCheckpointStore::new(Arc::clone(&kv));

    assert_eq!(store.key_prefix(), "rsi:auto_harness:");
    assert_eq!(store.key("sess-1"), "rsi:auto_harness:sess-1");
    assert!(store
        .load("sess-1")
        .await
        .expect("load of a missing key is Ok")
        .is_none());

    let state = SessionState::new("sess-1");
    store.save(&state).await.expect("save works");

    let stored = kv
        .get(&store.key("sess-1"))
        .await
        .expect("raw get works")
        .expect("checkpoint key exists");
    assert!(stored.is_text(), "checkpoint must be stored as JSON text");
    let text = stored.as_str().expect("text value");
    assert!(text.contains("\"session_id\""));
    assert!(text.contains("\"sess-1\""));

    assert_eq!(store.load("sess-1").await.expect("load works"), Some(state));

    store.delete("sess-1").await.expect("delete works");
    assert!(kv
        .get(&store.key("sess-1"))
        .await
        .expect("raw get after delete")
        .is_none());
    assert!(store
        .load("sess-1")
        .await
        .expect("load after delete")
        .is_none());

    // A document that cannot be parsed is a backend error, never "no state".
    kv.set(&store.key("sess-1"), KvValue::Text("not json".to_owned()))
        .await
        .expect("raw write works");
    let error = store
        .load("sess-1")
        .await
        .expect_err("malformed checkpoint is an error");
    assert_eq!(error.kind(), ErrorKind::Session);
}

// 4 — durability across processes: a fresh file-backed store sees the document
// written by the previous one (Python resumes by re-reading its run directory).
#[tokio::test]
async fn kv_store_persists_across_file_backend_instances() {
    // The tempdir must outlive both stores; it is dropped at the end of the test.
    let dir = tempfile::tempdir().expect("tempdir");

    let first: Arc<dyn KvStore> = Arc::new(
        LocalFileKvStore::new(dir.path())
            .await
            .expect("first file store opens"),
    );
    let writer = KvCheckpointStore::new(Arc::clone(&first));
    let state = rich_state();
    writer.save(&state).await.expect("save to file store works");

    // Simulate a new process: a second store over the same directory.
    let second: Arc<dyn KvStore> = Arc::new(
        LocalFileKvStore::new(dir.path())
            .await
            .expect("second file store opens"),
    );
    let reader = KvCheckpointStore::new(Arc::clone(&second));
    assert_eq!(
        reader
            .load(&state.session_id)
            .await
            .expect("load from a fresh store works"),
        Some(state.clone())
    );
    assert!(second
        .get(&reader.key("sess-rich"))
        .await
        .expect("raw get works")
        .is_some());

    reader
        .delete(&state.session_id)
        .await
        .expect("delete works");
    assert!(second
        .get(&reader.key("sess-rich"))
        .await
        .expect("raw get after delete")
        .is_none());
    assert!(reader
        .load(&state.session_id)
        .await
        .expect("load after delete")
        .is_none());
    assert!(first
        .get(&writer.key("sess-rich"))
        .await
        .expect("first store observes the delete")
        .is_none());
}

// 5 — resume continues the persisted session instead of replaying it.
// Orchestrator A checkpoints iterations 1 and 2; a *new* orchestrator B with the
// same store and session id restores them and runs iteration 3.
#[tokio::test]
async fn resume_continues_at_the_persisted_iteration() {
    let memory: Arc<InMemoryCheckpointStore> = Arc::new(InMemoryCheckpointStore::new());
    let store: Arc<dyn CheckpointStore> = memory.clone();

    let runner_a = Arc::new(ScriptedRunner::new(vec![0.1, 0.2]));
    let first = AutoHarnessOrchestrator::builder(config())
        .with_runner(runner_a)
        .with_session_id(RESUME_SESSION_ID)
        .with_checkpoint_store(Arc::clone(&store))
        .build()
        .expect("orchestrator A builds");

    let iteration_one = first.run_iteration().await.expect("iteration 1 runs");
    assert_eq!(iteration_one.iteration, 1);
    assert!(
        iteration_one.accepted,
        "first scored candidate becomes incumbent"
    );
    let iteration_two = first.run_iteration().await.expect("iteration 2 runs");
    assert_eq!(iteration_two.iteration, 2);
    assert!(iteration_two.accepted, "0.2 improves on 0.1");
    assert!((iteration_two.improvement - 0.1).abs() < 1e-9);
    // 0.1 >= min_improvement (0.05), so convergence has not fired yet.
    assert!(first.state().stop_reason.is_none(), "session can continue");

    let persisted = memory
        .load(RESUME_SESSION_ID)
        .await
        .expect("checkpoint loads")
        .expect("checkpoint was written");
    assert_eq!(persisted.iteration, 2);
    assert_eq!(persisted.history.len(), 2);
    let persisted_best = persisted.best_score.expect("persisted incumbent score");
    assert!((persisted_best - 0.2).abs() < 1e-9);

    let runner_b = Arc::new(ScriptedRunner::new(vec![0.3]));
    let second = AutoHarnessOrchestrator::builder(config())
        .with_runner(runner_b.clone())
        .with_session_id(RESUME_SESSION_ID)
        .with_checkpoint_store(Arc::clone(&store))
        .build()
        .expect("orchestrator B builds");

    let restored = second
        .resume()
        .await
        .expect("resume reads the store")
        .expect("a checkpoint exists");
    assert_eq!(restored, persisted);
    assert_eq!(restored.iteration, 2);
    assert_eq!(restored.history.len(), 2);
    let restored_best = restored.best_score.expect("restored incumbent score");
    assert!((restored_best - 0.2).abs() < 1e-9);
    assert_eq!(second.state().iteration, 2);

    let iteration_three = second
        .run_iteration()
        .await
        .expect("iteration 3 runs after resume");
    assert_eq!(iteration_three.iteration, 3);
    let previous = iteration_three
        .previous_best_score
        .expect("resumed incumbent is the baseline");
    assert!((previous - 0.2).abs() < 1e-9);
    assert!(iteration_three.accepted);
    assert!((iteration_three.improvement - 0.1).abs() < 1e-9);

    let after = second.state();
    assert_eq!(after.iteration, 3);
    assert_eq!(after.history.len(), 3);
    let after_best = after.best_score.expect("incumbent score after iteration 3");
    assert!((after_best - 0.3).abs() < 1e-9);

    // The harness saw the continued numbering, not a reset counter.
    let calls = runner_b.requests();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].session_id, RESUME_SESSION_ID);
    assert_eq!(calls[0].iteration, 3);

    // The resumed run re-checkpoints, so a third process sees iteration 3.
    let recheckpointed = memory
        .load(RESUME_SESSION_ID)
        .await
        .expect("checkpoint loads")
        .expect("checkpoint exists after continuation");
    assert_eq!(recheckpointed.iteration, 3);
    assert_eq!(recheckpointed.history.len(), 3);
}

// 6 — resuming without a checkpoint leaves the session at its initial state.
#[tokio::test]
async fn resume_without_a_checkpoint_leaves_the_session_untouched() {
    let store: Arc<dyn CheckpointStore> = Arc::new(InMemoryCheckpointStore::new());
    let orchestrator = AutoHarnessOrchestrator::builder(config())
        .with_runner(Arc::new(ScriptedRunner::new(Vec::new())))
        .with_session_id("sess-empty")
        .with_checkpoint_store(store)
        .build()
        .expect("orchestrator builds");

    assert_eq!(
        orchestrator
            .resume()
            .await
            .expect("resume on an empty store"),
        None
    );
    let state = orchestrator.state();
    assert_eq!(state, SessionState::new("sess-empty"));
    assert_eq!(state.iteration, 0);
    assert_eq!(state.candidates, Vec::<Candidate>::new());
    assert_eq!(state.history, Vec::<IterationRecord>::new());
    assert!(!state.is_finished());
    assert!(!orchestrator.is_running());
}
