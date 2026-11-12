#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//! Integration tests for `openjiuwen_agent_evolving::optimizer`.
//!
//! Covers the deterministic candidate seam (`ParameterSpace`, `GridSearchOptimizer`,
//! `BestSoFar`, `select_best`, `LlmInvokePolicy`) and the checkpointing stack
//! (`MemoryCheckpointStore`, `FileCheckpointStore`, `DefaultCheckpointManager`).
//!
//! Everything here is offline and deterministic: the file store is rooted in a
//! `tempfile::tempdir()`, no network or environment is touched, and no test reads a
//! clock or depends on execution order.
use std::collections::BTreeMap;

use openjiuwen_agent_evolving::optimizer::*;
use openjiuwen_common::error::ErrorKind;
use serde_json::json;

/// Candidate ids in proposal order, so chunk assertions stay one line.
fn ids(candidates: &[Candidate]) -> Vec<&str> {
    candidates.iter().map(|c| c.id.as_str()).collect()
}

/// The two-dimension grid shared by the space and optimizer tests.
///
/// `a` has two values, `b` has three, so the product is six points with `b`
/// varying fastest.
fn space() -> ParameterSpace {
    ParameterSpace::new()
        .with_dimension("a", vec![json!(1), json!(2)])
        .with_dimension("b", vec![json!("x"), json!("y"), json!("z")])
}

/// A representative resume payload with every optional field populated.
fn checkpoint() -> EvolveCheckpoint {
    let mut operators_state = BTreeMap::new();
    operators_state.insert("op".to_owned(), json!({"temperature": 0.3}));
    EvolveCheckpoint {
        version: "v1".to_owned(),
        run_id: "run-1".to_owned(),
        step: CheckpointStep { epoch: 2, batch: 1 },
        best: CheckpointBest { best_score: 0.8 },
        seed: Some(7),
        operators_state,
        updater_state: None,
        searcher_state: Some(json!({"cursor": 3})),
        last_metrics: BTreeMap::from([("current_epoch_score".to_owned(), 0.8)]),
    }
}

/// A manager snapshot with every field populated.
fn snapshot() -> CheckpointSnapshot {
    let mut operators_state = BTreeMap::new();
    operators_state.insert("op".to_owned(), json!({"temperature": 0.25}));
    CheckpointSnapshot {
        epoch: 3,
        batch: 2,
        best_score: 0.65,
        current_epoch_score: 0.55,
        seed: Some(42),
        operators_state,
        updater_state: Some(json!({"u": 1})),
        searcher_state: Some(json!({"cursor": 2})),
    }
}

#[test]
fn space_grid_is_cartesian_with_last_dimension_fastest() {
    let space = space();
    assert!(!space.is_empty());
    assert_eq!(space.len(), 6);
    assert_eq!(space.dimensions().len(), 2);

    let grid = space.grid(space.len());
    assert_eq!(grid.len(), 6);
    let expected = [(1, "x"), (1, "y"), (1, "z"), (2, "x"), (2, "y"), (2, "z")];
    for (index, (a, b)) in expected.iter().enumerate() {
        assert_eq!(grid[index].id, format!("cand-{index}"));
        assert_eq!(grid[index].generation, 0);
        assert_eq!(grid[index].score, None);
        assert_eq!(grid[index].parameters["a"], json!(a));
        assert_eq!(grid[index].parameters["b"], json!(b));
    }
}

#[test]
fn space_grid_truncates_and_handles_degenerate_input() {
    let space = space();
    let truncated = space.grid(4);
    assert_eq!(truncated.len(), 4);
    assert_eq!(ids(&truncated), ["cand-0", "cand-1", "cand-2", "cand-3"]);
    assert_eq!(truncated[3].parameters["a"], json!(2));
    assert_eq!(truncated[3].parameters["b"], json!("x"));
    assert_eq!(space.grid(0), Vec::<Candidate>::new());

    let no_dimensions = ParameterSpace::new();
    assert!(no_dimensions.is_empty());
    assert_eq!(no_dimensions.len(), 0);
    assert_eq!(no_dimensions.grid(5).len(), 0);

    let empty_dimension = ParameterSpace::new()
        .with_dimension("a", vec![json!(1)])
        .with_dimension("b", Vec::new());
    assert!(empty_dimension.is_empty());
    assert_eq!(empty_dimension.len(), 0);
    assert_eq!(empty_dimension.grid(5).len(), 0);
}

#[tokio::test]
async fn grid_optimizer_proposes_chunks_then_exhausts() {
    let space = space();
    let grid_ids: Vec<String> = space.grid(space.len()).into_iter().map(|c| c.id).collect();

    let mut optimizer = GridSearchOptimizer::new(space);
    assert_eq!(optimizer.name(), "grid_search");
    assert_eq!(optimizer.remaining(), 6);

    let first = optimizer.propose(2).await.expect("first chunk");
    let second = optimizer.propose(2).await.expect("second chunk");
    assert_eq!(ids(&first), ["cand-0", "cand-1"]);
    assert_eq!(ids(&second), ["cand-2", "cand-3"]);
    assert!(first.iter().all(|c| !second.iter().any(|o| o.id == c.id)));
    assert_eq!(optimizer.remaining(), 2);

    let third = optimizer.propose(2).await.expect("third chunk");
    assert_eq!(ids(&third), grid_ids[4..]);
    assert_eq!(optimizer.remaining(), 0);
    let exhausted = optimizer.propose(2).await.expect("exhausted");
    assert_eq!(exhausted.len(), 0);
    let zero_limit = optimizer.propose(0).await.expect("zero limit");
    assert_eq!(zero_limit.len(), 0);
}

#[tokio::test]
async fn grid_optimizer_keeps_strict_best_and_ignores_ties() {
    let mut optimizer = GridSearchOptimizer::new(space());
    assert!(optimizer.best().is_none());

    let candidates = optimizer.propose(3).await.expect("propose");
    optimizer.observe(&candidates[0], 0.5);
    assert_eq!(optimizer.best().map(|c| c.id.as_str()), Some("cand-0"));
    assert_eq!(optimizer.best().and_then(|c| c.score), Some(0.5));

    // Equal scores keep the earlier candidate (strict improvement only).
    optimizer.observe(&candidates[1], 0.5);
    assert_eq!(optimizer.best().map(|c| c.id.as_str()), Some("cand-0"));

    optimizer.observe(&candidates[1], 0.6);
    assert_eq!(optimizer.best().map(|c| c.id.as_str()), Some("cand-1"));
    assert_eq!(optimizer.best().and_then(|c| c.score), Some(0.6));
}

#[tokio::test]
async fn grid_optimizer_from_points_reuses_explicit_candidates() {
    let points = vec![
        Candidate::new("p-0", BTreeMap::from([("a".to_owned(), json!(1))])),
        Candidate::new("p-1", BTreeMap::from([("a".to_owned(), json!(2))])),
    ];
    let mut optimizer = GridSearchOptimizer::from_points(points);
    assert_eq!(optimizer.name(), "grid_search");
    assert_eq!(optimizer.candidates().len(), 2);
    assert_eq!(optimizer.remaining(), 2);

    let proposed = optimizer.propose(10).await.expect("propose");
    assert_eq!(ids(&proposed), ["p-0", "p-1"]);
    assert_eq!(proposed[1].parameters["a"], json!(2));
    assert!(optimizer.best().is_none());
}

#[test]
fn best_so_far_requires_strict_improvement() {
    let mut best = BestSoFar::new();
    assert!(best.is_improved(0.4));
    assert_eq!(best.score(), None);

    best.consider("first", 0.4);
    assert_eq!(best.score(), Some(0.4));
    assert_eq!(best.candidate_id(), Some("first"));
    assert!(!best.is_improved(0.4));

    // A tie keeps the incumbent (the first candidate to reach the score).
    best.consider("tie", 0.4);
    assert_eq!(best.candidate_id(), Some("first"));

    best.consider("better", 0.9);
    assert_eq!(best.score(), Some(0.9));
    assert_eq!(best.candidate_id(), Some("better"));
}

#[test]
fn select_best_skips_unscored_and_keeps_first_on_ties() {
    let scored = |id: &str, score: Option<f64>| Candidate {
        id: id.to_owned(),
        parameters: BTreeMap::new(),
        score,
        generation: 0,
    };
    let candidates = vec![
        scored("a", None),
        scored("b", Some(0.4)),
        scored("c", Some(0.9)),
        scored("d", Some(0.9)),
    ];
    assert_eq!(select_best(&candidates).map(|c| c.id.as_str()), Some("c"));
    assert_eq!(
        select_best(&candidates[..2]).map(|c| c.id.as_str()),
        Some("b")
    );
    assert!(select_best(&candidates[..1]).is_none());
    assert!(select_best(&[]).is_none());
}

#[test]
fn llm_invoke_policy_defaults_and_round_trip() {
    let policy = LlmInvokePolicy::default();
    assert_eq!(policy.max_attempts, 2);
    assert_eq!(policy.backoff_base_secs, 1.0);
    assert_eq!(policy.attempt_timeout_secs, 60.0);
    assert_eq!(policy.total_budget_secs, 600.0);
    assert!(policy.retry_empty_response);

    let encoded = serde_json::to_value(&policy).expect("serialize");
    assert_eq!(encoded["max_attempts"], json!(2));
    assert_eq!(encoded["backoff_base_secs"], json!(1.0));
    assert_eq!(encoded["retry_empty_response"], json!(true));
    let decoded: LlmInvokePolicy = serde_json::from_value(encoded).expect("deserialize");
    assert_eq!(decoded, policy);
}

#[tokio::test]
async fn memory_store_round_trips_and_reports_misses() {
    let store = MemoryCheckpointStore::new();
    let saved = store
        .save_checkpoint(&checkpoint(), "latest.json")
        .await
        .expect("save");
    assert_eq!(saved, Some("latest.json".to_owned()));

    let loaded = store.load_checkpoint("latest.json").await.expect("load");
    assert_eq!(loaded, Some(checkpoint()));
    assert_eq!(
        store.load_checkpoint("missing.json").await.expect("load"),
        None
    );
    let state = store.load_state_dict("latest.json").await.expect("state");
    assert_eq!(state, Some(json!({"op": {"temperature": 0.3}})));
    assert_eq!(
        store.load_state_dict("missing.json").await.expect("state"),
        None
    );
}

#[tokio::test]
async fn file_store_creates_nested_dirs_and_round_trips() {
    let directory = tempfile::tempdir().expect("tempdir");
    let base = directory.path().join("nested").join("checkpoints");
    let store = FileCheckpointStore::new(base.clone());

    let saved = store
        .save_checkpoint(&checkpoint(), "latest.json")
        .await
        .expect("save")
        .expect("path returned");
    assert!(base.is_dir(), "nested base directory must be created");
    assert!(saved.ends_with("latest.json"));
    assert!(std::path::Path::new(&saved).is_file());

    let loaded = store.load_checkpoint(&saved).await.expect("load");
    let loaded = loaded.expect("checkpoint present");
    assert_eq!(
        serde_json::to_value(&loaded).expect("loaded value"),
        serde_json::to_value(checkpoint()).expect("expected value")
    );
    let state = store.load_state_dict("latest.json").await.expect("state");
    assert_eq!(state, Some(json!({"op": {"temperature": 0.3}})));
    assert_eq!(
        store.load_checkpoint("missing.json").await.expect("load"),
        None
    );
    assert_eq!(
        store.load_state_dict("missing.json").await.expect("state"),
        None
    );
}

#[tokio::test]
async fn file_store_disabled_is_a_noop() {
    let store = FileCheckpointStore::disabled();
    assert!(store.base_dir().is_none());
    let saved = store
        .save_checkpoint(&checkpoint(), "latest.json")
        .await
        .expect("save");
    assert_eq!(saved, None);
    assert_eq!(
        store.load_checkpoint("latest.json").await.expect("load"),
        None
    );
    assert_eq!(
        store.load_state_dict("latest.json").await.expect("state"),
        None
    );
}

#[tokio::test]
async fn file_store_rejects_malformed_json() {
    let directory = tempfile::tempdir().expect("tempdir");
    std::fs::write(directory.path().join("bad.json"), b"{not json").expect("write bad file");
    let store = FileCheckpointStore::new(directory.path());

    let error = store
        .load_checkpoint("bad.json")
        .await
        .expect_err("malformed checkpoint must fail");
    assert_eq!(error.kind(), ErrorKind::Validation);
    let error = store
        .load_state_dict("bad.json")
        .await
        .expect_err("malformed state dict must fail");
    assert_eq!(error.kind(), ErrorKind::Validation);
}

#[tokio::test]
async fn file_store_writes_pretty_unescaped_json() {
    let directory = tempfile::tempdir().expect("tempdir");
    let store = FileCheckpointStore::new(directory.path());
    let mut checkpoint = checkpoint();
    checkpoint.run_id = "run-日本語".to_owned();

    let saved = store
        .save_checkpoint(&checkpoint, "unicode.json")
        .await
        .expect("save");
    let path = saved.expect("path returned");
    let raw = std::fs::read_to_string(&path).expect("read checkpoint");
    assert!(raw.contains("run-日本語"), "raw run id: {raw}");
    assert!(!raw.contains("\\u"), "no escapes: {raw}");
    assert!(raw.contains("\n  "), "pretty printed: {raw}");

    let loaded = store.load_checkpoint("unicode.json").await.expect("load");
    let loaded = loaded.expect("checkpoint present");
    assert_eq!(loaded.run_id, "run-日本語");
    assert_eq!(loaded, checkpoint);
}

#[test]
fn manager_save_policy_matches_python() {
    let manager = DefaultCheckpointManager::new().with_run_id("run-x");
    assert_eq!(manager.run_id(), "run-x");
    assert_eq!(manager.checkpoint_version(), "v1");
    assert_eq!(manager.save_every_n_epochs(), 1);
    assert!(manager.save_on_improve());
    assert!(manager.should_save(0, false));
    assert!(manager.should_save(1, false));

    let every_three = DefaultCheckpointManager::new().with_save_policy(3, false);
    assert_eq!(every_three.save_every_n_epochs(), 3);
    assert!(!every_three.save_on_improve());
    assert!(every_three.should_save(0, false));
    assert!(!every_three.should_save(1, false));
    assert!(!every_three.should_save(2, false));
    assert!(every_three.should_save(3, false));
    // Improvement is ignored when `save_on_improve` is off.
    assert!(!every_three.should_save(1, true));

    let clamped = DefaultCheckpointManager::new().with_save_policy(0, false);
    assert_eq!(clamped.save_every_n_epochs(), 1);
    assert!(clamped.should_save(4, false));

    let on_improve = DefaultCheckpointManager::new().with_save_policy(3, true);
    assert!(on_improve.should_save(1, true));
    assert!(!on_improve.should_save(1, false));
}

#[test]
fn manager_builds_checkpoint_from_snapshot() {
    let manager = DefaultCheckpointManager::new().with_run_id("run-9");
    let snapshot = snapshot();
    let built = manager.build_checkpoint(&snapshot);

    assert_eq!(built.version, "v1");
    assert_eq!(built.run_id, "run-9");
    assert_eq!(built.step, CheckpointStep { epoch: 3, batch: 2 });
    assert_eq!(built.best.best_score, 0.65);
    assert_eq!(built.seed, Some(42));
    assert_eq!(built.operators_state, snapshot.operators_state);
    assert_eq!(built.updater_state, Some(json!({"u": 1})));
    assert_eq!(built.searcher_state, Some(json!({"cursor": 2})));
    assert_eq!(built.last_metrics.len(), 1);
    assert_eq!(built.last_metrics["current_epoch_score"], 0.55);
}

#[test]
fn manager_restores_trainer_state() {
    let manager = DefaultCheckpointManager::new().with_run_id("unrelated");
    let checkpoint = checkpoint();
    let restored = manager.restore(&checkpoint);

    assert_eq!(restored.start_epoch, 2);
    assert_eq!(restored.best_score, 0.8);
    assert_eq!(restored.run_id, "run-1", "run id comes from the checkpoint");
    assert_eq!(restored.operators_state, checkpoint.operators_state);
}
