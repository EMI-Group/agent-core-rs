//! Scenarios 6 and 7: the JSONL run log.
//!
//! The record codec round trips through a `TempWorkspace` file (including a
//! corrupt trailing line), and a prompt search writes exactly one record per
//! completed round.

use std::collections::BTreeMap;
use std::io::Write;

use openjiuwen_test_support::TempWorkspace;

use super::support::*;
use crate::tune::*;

#[test]
fn run_log_round_trips_through_a_jsonl_file() {
    let workspace = TempWorkspace::new().expect("a temporary workspace");
    let path = workspace.path("runs/run.jsonl");
    let mut log = JsonlRunLog::new(&path).expect("a writable run log");

    let full = RunLogRecord::new(
        1,
        PromptCandidate::new("prompt one", "seed").with_id("cand-1"),
    )
    .with_candidate_prompts(vec!["prompt two".to_owned(), "prompt three".to_owned()])
    .with_scores(BTreeMap::from([
        ("cand-1".to_owned(), 0.9),
        ("cand-2".to_owned(), 0.4),
    ]))
    .with_reward(0.9)
    .with_reward_components(BTreeMap::from([("correctness".to_owned(), 1.0)]))
    .with_reward_notes(vec!["correctness gate passed".to_owned()])
    .with_best_score(0.9)
    .with_drift(0.05)
    .with_convergence(ConvergenceState {
        converged: true,
        reason: "target reward reached".to_owned(),
        best: 0.9,
        moving_average: 0.8,
        variance: 0.01,
    })
    .with_convergence_reason("target reward reached")
    .with_adopt(AdoptDecision::new(true, AdoptReason::FirstCandidate))
    .with_selected(true)
    .with_timestamp(NOW)
    .with_run_id("run-1")
    .with_observation("adopted the candidate");
    let minimal = RunLogRecord::new(2, PromptCandidate::new("prompt four", "seed"));
    let third = RunLogRecord::new(3, PromptCandidate::new("prompt five", "seed"))
        .with_reward(0.75)
        .with_best_score(0.9)
        .with_timestamp(NOW + 1.0)
        .with_run_id("run-1");

    for record in [&full, &minimal, &third] {
        log.record(record).expect("the log accepts every record");
    }

    let back = read_run_log(&path).expect("a readable run log");
    assert_eq!(back, vec![full.clone(), minimal.clone(), third.clone()]);
    // Field-level equality of the fully populated record.
    assert_eq!(back[0].iteration, 1);
    assert_eq!(back[0].candidate.candidate_id(), "cand-1");
    assert_eq!(back[0].candidate.prompt(), "prompt one");
    assert_eq!(back[0].candidate_prompts, full.candidate_prompts);
    assert_eq!(back[0].scores, full.scores);
    assert_eq!(back[0].reward_components, full.reward_components);
    assert_eq!(back[0].reward_notes, full.reward_notes);
    assert_eq!(back[0].best_score, full.best_score);
    assert_eq!(back[0].drift, full.drift);
    assert_eq!(
        back[0]
            .convergence
            .as_ref()
            .map(|state| state.reason.as_str()),
        Some("target reward reached")
    );
    assert!(back[0].converged);
    assert_eq!(
        back[0].convergence_reason.as_deref(),
        Some("target reward reached")
    );
    assert_eq!(
        back[0].adopt,
        Some(AdoptDecision::new(true, AdoptReason::FirstCandidate))
    );
    assert_eq!(back[0].timestamp, NOW);
    assert_eq!(back[0].run_id, "run-1");
    assert_eq!(
        back[0].observation.as_deref(),
        Some("adopted the candidate")
    );
    // The minimal record round-trips as the defaults it was built from.
    assert_eq!(back[1].iteration, 2);
    assert!(back[1].adopt.is_none());
    assert!(back[1].convergence.is_none());
    assert!(back[1].scores.is_empty());
    assert_eq!(back[1].timestamp, 0.0);

    // The limited reader keeps the tail of the log; `0` means no limit.
    assert_eq!(
        read_run_log_limited(&path, 2).expect("a readable run log"),
        vec![minimal.clone(), third.clone()]
    );
    assert_eq!(read_run_log_limited(&path, 0).expect("a limit"), back);

    // A corrupt trailing line (a truncated write, a foreign event) is skipped.
    let mut file = std::fs::OpenOptions::new()
        .append(true)
        .open(&path)
        .expect("the log file exists");
    file.write_all(b"{\"iteration\": \"three\"}\n\n")
        .expect("append a corrupt line");
    drop(file);
    assert_eq!(read_run_log(&path).expect("a readable run log"), back);
}

// ------------------------------------------- 7. prompt search emits a log ----

#[tokio::test]
async fn prompt_search_emits_one_run_log_record_per_round() {
    let task = PromptTaskSpec::new("Answer the question accurately.")
        .with_base_prompt("Answer the question.")
        .with_cases(vec![
            PromptTaskCase::from_text("question one", "answer one").expect("a valid task case"),
            PromptTaskCase::from_text("question two", "answer two").expect("a valid task case"),
            PromptTaskCase::from_text("question three", "answer three").expect("a valid task case"),
        ]);

    let (concrete, handle) = shared_log();
    let result = PromptSearchOptimizer::new()
        .with_run_log(handle)
        .with_timestamp(NOW)
        .optimize_task(&task)
        .await
        .expect("a valid prompt-search task");
    let records = recorded(&concrete);

    assert!(result.iterations >= 1, "a search runs at least one round");
    assert_eq!(records.len(), result.iterations, "one record per round");
    assert_eq!(
        records
            .iter()
            .map(|record| record.iteration)
            .collect::<Vec<usize>>(),
        (1..=records.len()).collect::<Vec<usize>>(),
        "round indices ascend from one"
    );
    let last = records.last().expect("at least one round");
    assert_close(last.best_score, result.score);
    assert!(records
        .iter()
        .all(|record| (record.timestamp - NOW).abs() < 1e-12));
    assert!(records.iter().all(|record| record.run_id == result.run_id));
    assert!(records
        .iter()
        .all(|record| !record.candidate.prompt().is_empty()));

    // A second run over the same task writes the same records.
    let (replay_concrete, replay_handle) = shared_log();
    let replay = PromptSearchOptimizer::new()
        .with_run_log(replay_handle)
        .with_timestamp(NOW)
        .optimize_task(&task)
        .await
        .expect("a valid prompt-search task");
    assert_eq!(result, replay);
    assert_eq!(recorded(&replay_concrete), records);
}
