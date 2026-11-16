//! Unit tests for the JSONL run log, its sinks, and the tolerant readers.
use super::*;
use crate::tune::AdoptReason;
use openjiuwen_common::error::ErrorKind;
use openjiuwen_test_support::TempWorkspace;

fn candidate(prompt: &str, rationale: &str) -> PromptCandidate {
    PromptCandidate::new(prompt, rationale)
}

fn convergence() -> ConvergenceState {
    ConvergenceState {
        converged: true,
        reason: "rewards plateaued".to_owned(),
        best: 0.8,
        moving_average: 0.79,
        variance: 0.0001,
    }
}

/// A fully populated round, including an adoption verdict.
fn full_record(iteration: usize) -> RunLogRecord {
    let winner = candidate("Answer {{question}} briefly.", "kept the placeholder");
    RunLogRecord::new(iteration, winner.clone())
        .with_candidate_prompts(vec![
            "Answer {{question}} with steps.".to_owned(),
            "Answer shortly.".to_owned(),
        ])
        .with_score(winner.candidate_id.clone(), 0.8)
        .with_score("deadbeef", 0.55)
        .with_reward(0.8)
        .with_reward_components(BTreeMap::from([
            ("correctness".to_owned(), 0.9),
            ("completeness".to_owned(), 0.7),
        ]))
        .with_reward_notes(vec!["correctness gate passed".to_owned()])
        .with_prior_best_score(Some(0.6))
        .with_best_score(0.8)
        .with_drift(0.12)
        .with_convergence(convergence())
        .with_adopt(AdoptDecision::new(
            true,
            AdoptReason::Improved { gain: 0.2 },
        ))
        .with_selected(true)
        .with_timestamp(1_700_000_000.5)
        .with_run_id("run-1")
        .with_observation("first round")
}

/// A round with every optional field absent.
fn bare_record(iteration: usize) -> RunLogRecord {
    RunLogRecord::new(iteration, candidate("Do the thing.", "seed"))
}

fn log_path(workspace: &TempWorkspace) -> PathBuf {
    workspace.path("runs/prompt_search.jsonl")
}

#[test]
fn full_round_trip_preserves_every_field() {
    let workspace = TempWorkspace::new().unwrap();
    let path = log_path(&workspace);
    let mut log = JsonlRunLog::new(&path).unwrap();
    assert_eq!(log.path(), path.as_path());

    let first = full_record(1);
    let second = bare_record(2);
    log.record(&first).unwrap();
    log.record(&second).unwrap();

    let read = read_run_log(&path).unwrap();
    assert_eq!(read, vec![first.clone(), second.clone()]);
    assert_eq!(
        read[0].adopt,
        Some(AdoptDecision::new(
            true,
            AdoptReason::Improved { gain: 0.2 }
        ))
    );
    assert_eq!(read[0].convergence, Some(convergence()));
    assert!(read[0].converged);
    assert_eq!(
        read[0].convergence_reason.as_deref(),
        Some("rewards plateaued")
    );
    assert_eq!(read[0].scores.len(), 2);
    assert_eq!(read[0].candidate_prompts.len(), 2);
    assert_eq!(read[0].reward_components["correctness"], 0.9);
    assert_eq!(
        read[0].reward_notes,
        vec!["correctness gate passed".to_owned()]
    );
    // The pre-round elite survives too, and the winner beat it.
    assert_eq!(read[0].prior_best_score, Some(0.6));
    assert_eq!(read[0].best_score, 0.8);
    assert!((read[0].timestamp - 1_700_000_000.5).abs() < 1e-9);
    assert_eq!(read[0].run_id, "run-1");
    assert_eq!(read[0].observation.as_deref(), Some("first round"));
    assert!(read[0].selected);
    assert!(read[0].is_improvement());
    assert!(read[0].adopted());
    assert_eq!(read[0].iteration(), 1);
    assert_eq!(read[0].candidate(), &first.candidate);
    assert!((read[0].reward() - 0.8).abs() < 1e-12);

    // The `None` variants survive as `None`.
    assert_eq!(read[1].adopt, None);
    assert_eq!(read[1].convergence, None);
    assert_eq!(read[1].convergence_reason, None);
    assert_eq!(read[1].observation, None);
    assert!(!read[1].converged);
    assert!(!read[1].selected);
    assert!(!read[1].adopted());
    assert_eq!(read[1].scores, BTreeMap::new());
    assert_eq!(read[1].candidate_prompts, Vec::<String>::new());
}

#[test]
fn append_across_instances_keeps_earlier_lines() {
    let workspace = TempWorkspace::new().unwrap();
    let path = log_path(&workspace);

    {
        let mut log = JsonlRunLog::new(&path).unwrap();
        log.record(&bare_record(1)).unwrap();
    }
    {
        let mut log = JsonlRunLog::new(&path).unwrap();
        log.record(&bare_record(2)).unwrap();
        log.record(&bare_record(3)).unwrap();
    }

    let read = read_run_log(&path).unwrap();
    assert_eq!(
        read.iter()
            .map(|record| record.iteration)
            .collect::<Vec<_>>(),
        vec![1, 2, 3]
    );
}

#[test]
fn create_truncates_the_file() {
    let workspace = TempWorkspace::new().unwrap();
    let path = log_path(&workspace);

    let mut log = JsonlRunLog::new(&path).unwrap();
    log.record(&bare_record(1)).unwrap();
    log.record(&bare_record(2)).unwrap();
    assert_eq!(read_run_log(&path).unwrap().len(), 2);

    let mut truncated = JsonlRunLog::create(&path).unwrap();
    assert_eq!(read_run_log(&path).unwrap(), Vec::new());

    truncated.record(&bare_record(7)).unwrap();
    let read = read_run_log(&path).unwrap();
    assert_eq!(read.len(), 1);
    assert_eq!(read[0].iteration, 7);
}

#[test]
fn missing_file_reads_empty() {
    let workspace = TempWorkspace::new().unwrap();
    let path = workspace.path("absent/runs.jsonl");
    assert_eq!(read_run_log(&path).unwrap(), Vec::new());
    assert_eq!(read_run_log_limited(&path, 3).unwrap(), Vec::new());
}

#[test]
fn blank_and_corrupt_lines_are_skipped() {
    let workspace = TempWorkspace::new().unwrap();
    let path = workspace.path("runs/mixed.jsonl");
    let good = bare_record(4);
    let line = good.to_json_line().unwrap();
    let contents =
        format!("\n{line}\n\n   \n{{\"stage\":\"round\"}}\nnot json at all\n[1, 2, 3]\n{line}\n");
    workspace.write("runs/mixed.jsonl", contents).unwrap();

    let read = read_run_log(&path).unwrap();
    // The `{"stage": …}` object is an *acceptable* record — every field
    // defaults — while the plain scalar, the array, and the garbage line are
    // skipped.
    assert_eq!(read.len(), 3);
    assert_eq!(read[0], good);
    assert_eq!(read[1], RunLogRecord::default());
    assert_eq!(read[2], good);
}

#[test]
fn limit_keeps_the_last_records() {
    let workspace = TempWorkspace::new().unwrap();
    let path = log_path(&workspace);
    let mut log = JsonlRunLog::new(&path).unwrap();
    for iteration in 1..=5 {
        log.record(&bare_record(iteration)).unwrap();
    }

    let last_two = read_run_log_limited(&path, 2).unwrap();
    assert_eq!(
        last_two
            .iter()
            .map(|record| record.iteration)
            .collect::<Vec<_>>(),
        vec![4, 5]
    );

    assert_eq!(read_run_log_limited(&path, 10).unwrap().len(), 5);
    assert_eq!(read_run_log_limited(&path, 1).unwrap()[0].iteration, 5);

    // `0` means no limit, and is not clamped to one.
    assert_eq!(read_run_log_limited(&path, 0).unwrap().len(), 5);
    assert_eq!(read_run_log(&path).unwrap().len(), 5);
}

#[test]
fn multiline_and_unicode_prompts_survive() {
    let workspace = TempWorkspace::new().unwrap();
    let path = log_path(&workspace);
    let prompt = "第一行\n第二行\t\"quoted\" \\ backslash 🙂\r\n";
    let record = RunLogRecord::new(1, candidate(prompt, "中文说明\nsecond line"))
        .with_candidate_prompts(vec![format!("prefix {prompt} suffix")])
        .with_observation("emoji ✅ and a newline\n");

    let mut log = JsonlRunLog::new(&path).unwrap();
    log.record(&record).unwrap();

    // One physical line only.
    let text = std::fs::read_to_string(&path).unwrap();
    assert_eq!(text.lines().count(), 1);

    let read = read_run_log(&path).unwrap();
    assert_eq!(read, vec![record]);
    assert_eq!(read[0].candidate.prompt, prompt);
    assert_eq!(read[0].candidate.rationale, "中文说明\nsecond line");
}

#[test]
fn null_run_log_discards_and_in_memory_keeps_order() {
    let mut null = NullRunLog::new();
    assert!(null.record(&bare_record(1)).is_ok());
    assert_eq!(null, NullRunLog);

    let mut memory = InMemoryRunLog::new();
    assert!(memory.is_empty());
    memory.record(&bare_record(1)).unwrap();
    memory.record(&full_record(2)).unwrap();
    assert_eq!(memory.len(), 2);
    assert_eq!(memory.records()[0].iteration, 1);
    assert_eq!(memory.last().unwrap().iteration, 2);
    assert_eq!(memory.into_records().len(), 2);

    let mut memory = InMemoryRunLog::new();
    memory
        .record_all(&[bare_record(1), bare_record(2), bare_record(3)])
        .unwrap();
    let iterations = memory
        .records()
        .iter()
        .map(|record| record.iteration)
        .collect::<Vec<_>>();
    assert_eq!(iterations, vec![1, 2, 3]);
    memory.clear();
    assert!(memory.is_empty());
}

#[test]
fn json_line_is_deterministic() {
    let record = full_record(3);
    let first = record.to_json_line().unwrap();
    let second = record.to_json_line().unwrap();
    assert_eq!(first, second);
    assert!(!first.contains('\n'));
    assert!(first.starts_with('{'));
    assert!(first.ends_with('}'));
    assert_eq!(first, serde_json::to_string(&record).unwrap());

    // Field order follows the struct, not a hash map.
    let keys = [
        "iteration",
        "candidate",
        "candidate_prompts",
        "scores",
        "reward",
        "reward_components",
        "reward_notes",
        "prior_best_score",
        "best_score",
        "drift",
        "convergence",
        "converged",
        "convergence_reason",
        "adopt",
        "timestamp",
        "run_id",
        "observation",
    ];
    let mut cursor = 0;
    for key in keys {
        let needle = format!("\"{key}\":");
        let found = first[cursor..]
            .find(&needle)
            .unwrap_or_else(|| panic!("missing {needle} after {cursor}"));
        cursor += found;
    }

    assert_eq!(RunLogRecord::from_json_line(&first), Some(record));
}

#[test]
fn serde_defaults_from_a_minimal_object() {
    let line = "{\"iteration\": 3, \"candidate\": {\"prompt\": \"P\", \
                \"candidate_id\": \"abc12345\", \"rationale\": \"why\"}, \
                \"unknown_key\": [1, 2]}";
    let record = RunLogRecord::from_json_line(line).unwrap();
    assert_eq!(
        record,
        RunLogRecord::new(3, PromptCandidate::new("P", "why").with_id("abc12345"))
    );
    assert_eq!(record.candidate.candidate_id, "abc12345");
    assert_eq!(record.reward, 0.0);
    assert_eq!(record.timestamp, 0.0);
    assert!(!record.converged);
    assert_eq!(record.adopt, None);
    assert_eq!(record.prior_best_score, None);
    assert!(!record.is_improvement());

    // A line that shares no keys is an all-default record; anything that is
    // not a JSON object is rejected.
    assert_eq!(
        RunLogRecord::from_json_line("{}"),
        Some(RunLogRecord::default())
    );
    assert_eq!(
        RunLogRecord::from_json_line("{\"stage\": \"round\"}"),
        Some(RunLogRecord::default())
    );
    assert_eq!(RunLogRecord::from_json_line("[]"), None);
    assert_eq!(RunLogRecord::from_json_line("[{}]"), None);
    assert_eq!(RunLogRecord::from_json_line("null"), None);
    assert_eq!(RunLogRecord::from_json_line("3"), None);
    assert_eq!(RunLogRecord::from_json_line("\"text\""), None);
    assert_eq!(RunLogRecord::from_json_line("   "), None);
    assert_eq!(
        RunLogRecord::from_json_line("{\"iteration\": \"three\"}"),
        None
    );
    assert_eq!(RunLogRecord::from_json_line("{\"iteration\": -1}"), None);
}

#[test]
fn is_improvement_reports_whether_the_elite_advanced() {
    // The winner beat the elite it started from.
    let advanced = bare_record(2)
        .with_prior_best_score(Some(0.5))
        .with_reward(0.75)
        .with_best_score(0.75);
    assert!(advanced.is_improvement());

    // A round that only matched the elite did not move it.
    let equal = bare_record(2)
        .with_prior_best_score(Some(0.75))
        .with_reward(0.75)
        .with_best_score(0.75)
        .with_selected(true);
    assert!(!equal.is_improvement());

    // A worse round leaves the elite where it was.
    let worse = bare_record(3)
        .with_prior_best_score(Some(0.75))
        .with_reward(0.5)
        .with_best_score(0.75);
    assert!(!worse.is_improvement());

    // Round 1 has no elite to advance: it establishes the first one, which is
    // what `selected` reports.
    let first = bare_record(1)
        .with_prior_best_score(None)
        .with_reward(0.9)
        .with_best_score(0.9)
        .with_selected(true);
    assert!(!first.is_improvement());
    assert!(first.selected);

    // No reported pre-round elite (a hand-built record, a legacy line) makes no
    // claim either way.
    let unknown = RunLogRecord::default()
        .with_reward(0.9)
        .with_best_score(0.9)
        .with_selected(true);
    assert_eq!(unknown.prior_best_score, None);
    assert!(!unknown.is_improvement());
}

#[test]
fn the_wire_carries_the_pre_round_elite_and_loads_legacy_lines() {
    let workspace = TempWorkspace::new().unwrap();
    let path = log_path(&workspace);
    let mut log = JsonlRunLog::new(&path).unwrap();

    let first = bare_record(1).with_reward(0.4).with_best_score(0.4);
    let second = bare_record(2)
        .with_prior_best_score(Some(0.4))
        .with_reward(0.6)
        .with_best_score(0.6)
        .with_selected(true);
    log.record(&first).unwrap();
    log.record(&second).unwrap();

    let read = read_run_log(&path).unwrap();
    assert_eq!(read, vec![first.clone(), second.clone()]);
    assert_eq!(read[0].prior_best_score, None);
    assert!(!read[0].is_improvement());
    assert_eq!(read[1].prior_best_score, Some(0.4));
    assert!(read[1].is_improvement());

    // The field is on the wire, explicitly `null` for a round without an elite.
    let text = std::fs::read_to_string(&path).unwrap();
    assert_eq!(text.matches("\"prior_best_score\":").count(), 2);
    assert!(text.contains("\"prior_best_score\":null"));

    // A line written before the field existed still loads, and it claims no
    // advance because it reports no pre-round elite.
    let legacy = "{\"iteration\": 2, \"candidate\": {\"prompt\": \"P\", \
                  \"candidate_id\": \"abc12345\", \"rationale\": \"why\"}, \
                  \"reward\": 0.8, \"best_score\": 0.8, \"selected\": true, \
                  \"timestamp\": 1700000000.5, \"run_id\": \"run-1\"}";
    let record = RunLogRecord::from_json_line(legacy).expect("a legacy line loads");
    assert_eq!(record.iteration, 2);
    assert_eq!(record.reward, 0.8);
    assert!(record.selected);
    assert_eq!(record.prior_best_score, None);
    assert!(!record.is_improvement());
    // Re-serializing writes the field and the record still round trips.
    let line = record.to_json_line().unwrap();
    assert!(line.contains("\"prior_best_score\":null"));
    assert_eq!(RunLogRecord::from_json_line(&line), Some(record));
}

#[test]
fn parent_directories_are_created() {
    let workspace = TempWorkspace::new().unwrap();
    let path = workspace.path("a/b/c/runs.jsonl");
    assert!(!path.parent().unwrap().exists());

    let mut log = JsonlRunLog::new(&path).unwrap();
    assert!(path.is_file());
    assert_eq!(std::fs::read_to_string(&path).unwrap(), "");

    log.record(&bare_record(1)).unwrap();
    assert_eq!(read_run_log(&path).unwrap().len(), 1);

    // The directory is recreated if it disappeared between records.
    std::fs::remove_dir_all(workspace.path("a")).unwrap();
    log.record(&bare_record(2)).unwrap();
    assert_eq!(read_run_log(&path).unwrap().len(), 1);
}

#[test]
fn blank_paths_are_validation_errors() {
    let error = JsonlRunLog::new("").unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    let error = JsonlRunLog::create("").unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    let error = read_run_log("").unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    let error = read_run_log_limited("", 2).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
}

#[test]
fn builders_and_readers_cover_the_optional_fields() {
    let record = RunLogRecord::default();
    assert_eq!(record.iteration, 0);
    assert!(!record.is_improvement());
    assert!(!record.adopted());

    let record = RunLogRecord::new(2, candidate("bare prompt", ""))
        .with_converged(true)
        .with_convergence_reason("target reward reached")
        .with_convergence(ConvergenceState {
            converged: false,
            ..ConvergenceState::default()
        })
        .with_adopt(AdoptDecision::new(
            false,
            AdoptReason::NoImprovement { gain: -0.1 },
        ));
    // `with_convergence` mirrors the state it is given.
    assert!(!record.converged);
    assert_eq!(
        record.convergence_reason.as_deref(),
        Some("target reward reached")
    );
    assert_eq!(record.convergence.as_ref().unwrap().reason, "");
    assert_eq!(
        record.adopt.unwrap().reason,
        AdoptReason::NoImprovement { gain: -0.1 }
    );
    assert!(!record.adopted());

    let record = full_record(1);
    assert_eq!(
        record.reward_notes,
        vec!["correctness gate passed".to_owned()]
    );
}

#[cfg(unix)]
#[test]
fn write_failures_do_not_fail_the_run() {
    let workspace = TempWorkspace::new().unwrap();
    let path = log_path(&workspace);
    let mut log = JsonlRunLog::new(&path).unwrap();

    // Replace the file with a directory: appending is now an I/O failure.
    std::fs::remove_file(&path).unwrap();
    std::fs::create_dir(&path).unwrap();

    assert!(log.record(&bare_record(1)).is_ok());
    assert!(log.record(&full_record(2)).is_ok());
    assert!(path.is_dir());
}
