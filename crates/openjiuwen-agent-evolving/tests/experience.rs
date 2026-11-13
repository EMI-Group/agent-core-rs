//! Offline tests for the `experience` subsystem: records, archive, extractor
//! and lifecycle.
//!
//! Everything here is deterministic — no clocks, no models, no filesystem — so
//! the assertions are hand-computable from the fixtures in each test.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::assert_is_empty,
    clippy::match_wildcard_for_single_variants
)]

use std::collections::BTreeMap;

use openjiuwen_agent_evolving::dataset::{Case, EvaluatedCase};
use openjiuwen_agent_evolving::experience::{
    jaccard, tokens, AddOutcome, ExperienceArchive, ExperienceExtractor, ExperienceLifecycle,
    ExperienceQuery, ExperienceRecord, ExperienceSection, ExperienceStats, LifecycleConfig,
};
use openjiuwen_agent_evolving::signal::{EvolutionSignal, EvolutionTarget};
use openjiuwen_agent_evolving::trajectory::{
    StepKind, StepStatus, Trajectory, TrajectoryOutcome, TrajectoryStep,
};
use openjiuwen_common::error::Error;
use serde_json::{json, Value};

/// A record with a fixed id, skill and section.
fn record(id: &str, content: &str) -> ExperienceRecord {
    ExperienceRecord::new(id, "skill-a", ExperienceSection::Tool, content).expect("valid record")
}

/// A two-entry string map used for case inputs/labels.
fn string_map(key: &str, value: &str) -> BTreeMap<String, Value> {
    let mut map = BTreeMap::new();
    map.insert(key.to_owned(), Value::String(value.to_owned()));
    map
}

/// A trajectory with the given id and outcome.
fn trajectory(id: &str, outcome: TrajectoryOutcome) -> Trajectory {
    Trajectory::new(id).expect("valid id").with_outcome(outcome)
}

#[test]
fn record_new_rejects_blank_fields() {
    for (id, skill, content) in [
        ("", "skill-a", "content"),
        ("exp-1", "  ", "content"),
        ("exp-1", "skill-a", "\n"),
    ] {
        let error = ExperienceRecord::new(id, skill, ExperienceSection::Body, content)
            .expect_err("blank fields must fail");
        assert!(matches!(error, Error::Validation(_)), "got {error}");
    }
}

#[test]
fn record_validation_bounds_confidence_and_weight() {
    let base = record("exp-1", "raise the timeout before parsing");

    assert!(base.clone().with_confidence(1.5).is_err());
    assert!(base.clone().with_confidence(f64::NAN).is_err());
    assert!(base.clone().with_weight(-0.1).is_err());
    assert!(base.clone().with_weight(f64::INFINITY).is_err());
    assert!(base.clone().with_confidence(0.0).is_ok());
    assert!(base.with_weight(2.5).is_ok());
}

#[test]
fn success_rate_and_score_follow_the_documented_formula() {
    let fresh = record("exp-1", "raise the timeout");
    assert_eq!(fresh.attempts(), 0);
    assert!(fresh.success_rate().abs() < f64::EPSILON);
    assert!(
        (fresh.score() - 0.5).abs() < 1e-12,
        "score {}",
        fresh.score()
    );

    let perfect = fresh.clone().record_success().record_success();
    assert_eq!(perfect.attempts(), 2);
    assert!((perfect.success_rate() - 1.0).abs() < 1e-12);
    assert_eq!(perfect.hits, 2);
    assert!((perfect.score() - 1.5).abs() < 1e-12);

    let mixed = fresh.record_success().record_failure();
    assert_eq!(mixed.attempts(), 2);
    assert!((mixed.success_rate() - 0.5).abs() < 1e-12);
    assert!((mixed.score() - 1.0).abs() < 1e-12);

    let weighted = record("exp-2", "content").with_weight(2.0).unwrap();
    assert!((weighted.score() - 1.0).abs() < 1e-12);
}

#[test]
fn builders_normalise_keywords_and_flags() {
    let built = record("exp-1", "content")
        .with_skill("skill-b")
        .with_section(ExperienceSection::Body)
        .with_keywords(["timeout", " timeout ", "", "pdf"])
        .with_source("evaluator")
        .mark_shared();
    assert_eq!(built.skill_name, "skill-b");
    assert_eq!(built.section, ExperienceSection::Body);
    assert_eq!(built.keywords, vec!["pdf", "timeout"]);
    assert_eq!(built.source, "evaluator");
    assert!(built.shared);
}

#[test]
fn tokens_are_sorted_lowercase_and_stopword_filtered() {
    let extracted = tokens("The Read_File failed, then THE read_file failed again 日本語");
    assert!(extracted.contains("read_file"));
    assert!(extracted.contains("failed"));
    assert!(extracted.contains("日") && extracted.contains("語"));
    assert!(!extracted.contains("the"));
    assert!(!extracted.contains("again"), "five-letter stopword");
    assert_eq!(
        extracted,
        tokens("the read_file failed again 日本語, the read_file")
    );

    let sorted: Vec<&String> = extracted.iter().collect();
    let mut expected = sorted.clone();
    expected.sort();
    assert_eq!(sorted, expected);
}

#[test]
fn jaccard_is_symmetric_and_handles_empty_sets() {
    let left = tokens("timeout while calling search");
    let right = tokens("timeout while calling lookup");
    assert!((jaccard(&left, &right) - jaccard(&right, &left)).abs() < f64::EPSILON);
    assert!(
        (jaccard(&left, &right) - 0.5).abs() < 1e-12,
        "2 shared of 4 tokens"
    );

    let empty = tokens("");
    assert!(jaccard(&empty, &empty).abs() < f64::EPSILON);
    assert!(jaccard(&left, &empty).abs() < f64::EPSILON);
    assert!((jaccard(&left, &left) - 1.0).abs() < f64::EPSILON);
}

#[test]
fn add_inserts_and_stamps_sequence_counters() {
    let mut archive = ExperienceArchive::new();
    assert!(archive.is_empty());
    assert_eq!(archive.seq(), 0);

    let outcome = archive.add(record("exp-1", "raise the timeout")).unwrap();
    assert_eq!(outcome, AddOutcome::Inserted("exp-1".to_owned()));
    assert_eq!(archive.len(), 1);
    assert_eq!(archive.seq(), 1);
    let stored = archive.get("exp-1").expect("stored");
    assert_eq!(stored.created_seq, 1);
    assert_eq!(stored.updated_seq, 1);
}

#[test]
fn add_rejects_invalid_records() {
    let mut archive = ExperienceArchive::new();
    let mut broken = record("exp-1", "content");
    broken.content = String::new();
    let error = archive.add(broken).expect_err("empty content must fail");
    assert!(matches!(error, Error::Validation(_)));
    assert!(archive.is_empty());
    assert_eq!(
        archive.seq(),
        0,
        "a rejected record does not advance the seal"
    );
}

#[test]
fn add_merges_near_duplicates_and_keeps_distinct_records() {
    let mut archive = ExperienceArchive::new().with_dedupe_threshold(0.5);
    let first = record(
        "exp-1",
        "raise the timeout before parsing large pdf reports",
    )
    .with_keywords(["timeout", "pdf"])
    .record_success()
    .record_success();
    archive.add(first).unwrap();

    let duplicate = record("exp-2", "raise the timeout before parsing large pdf report")
        .with_keywords(["parsing", "pdf"])
        .record_failure();
    let outcome = archive.add(duplicate).unwrap();
    match outcome {
        AddOutcome::Merged { into, similarity } => {
            assert_eq!(into, "exp-1");
            assert!(similarity >= 0.5, "similarity {similarity}");
        }
        other => panic!("expected a merge, got {other:?}"),
    }
    assert_eq!(archive.len(), 1);

    let merged = archive.get("exp-1").expect("merged record");
    assert_eq!(
        merged.content, "raise the timeout before parsing large pdf reports",
        "the record with more successes keeps its content"
    );
    assert_eq!(merged.keywords, vec!["parsing", "pdf", "timeout"]);
    assert_eq!(merged.successes, 2);
    assert_eq!(merged.failures, 1);
    assert_eq!(merged.hits, 4, "2 + 1 + one extra hit for the merge");
    assert_eq!(merged.confidence, 1.0);
    assert_eq!(merged.weight, 1.0);
    assert_eq!(merged.updated_seq, 2);

    let distinct = record("exp-3", "always cite the failing span index");
    assert_eq!(
        archive.add(distinct).unwrap(),
        AddOutcome::Inserted("exp-3".to_owned())
    );
    assert_eq!(archive.len(), 2);
}

#[test]
fn add_scopes_dedupe_to_skill_and_section() {
    let content = "raise the timeout before parsing large reports";
    let mut archive = ExperienceArchive::new();

    archive.add(record("exp-1", content)).unwrap();
    archive
        .add(
            ExperienceRecord::new("exp-2", "skill-b", ExperienceSection::Tool, content)
                .expect("valid record"),
        )
        .unwrap();
    archive
        .add(
            ExperienceRecord::new("exp-3", "skill-a", ExperienceSection::Body, content)
                .expect("valid record"),
        )
        .unwrap();

    assert_eq!(
        archive.len(),
        3,
        "identical text in another scope is distinct"
    );
}

#[test]
fn capacity_evicts_the_lowest_scoring_record() {
    let mut archive = ExperienceArchive::new().with_capacity(2);
    archive
        .add(
            record("exp-1", "low value lesson")
                .with_weight(0.2)
                .unwrap(),
        )
        .unwrap();
    archive
        .add(
            record("exp-2", "high value lesson")
                .with_weight(1.0)
                .unwrap(),
        )
        .unwrap();
    archive
        .add(
            record("exp-3", "medium value lesson")
                .with_weight(0.6)
                .unwrap(),
        )
        .unwrap();

    assert_eq!(archive.len(), 2);
    assert!(archive.get("exp-1").is_none(), "lowest score was evicted");
    assert!(archive.get("exp-2").is_some());
    assert!(archive.get("exp-3").is_some());
}

#[test]
fn unbounded_capacity_keeps_everything() {
    let mut archive = ExperienceArchive::new().with_capacity(0);
    for index in 0..20 {
        archive
            .add(record(
                &format!("exp-{index}"),
                &format!("lesson about subject{index}"),
            ))
            .unwrap();
    }
    assert_eq!(archive.len(), 20);
    assert_eq!(archive.capacity(), 0);
}

#[test]
fn query_ranks_by_keyword_text_and_success_rate() {
    let mut archive = ExperienceArchive::new();
    archive
        .add(
            record(
                "exp-1",
                "raise the request timeout before retrying the fetch",
            )
            .with_keywords(["timeout", "retry"]),
        )
        .unwrap();
    archive
        .add(record("exp-2", "raise the request timeout").with_keywords(["timeout"]))
        .unwrap();
    archive
        .add(record("exp-3", "unrelated lesson about charts").with_keywords(["charts"]))
        .unwrap();

    let query = ExperienceQuery::new().with_keywords(["timeout", "retry"]);
    let matches = archive.query(&query);
    assert_eq!(
        matches.len(),
        3,
        "every record passing the filters is ranked"
    );
    assert_eq!(matches[0].record.record_id, "exp-1");
    assert_eq!(
        matches[0].matched_keywords,
        vec!["retry".to_owned(), "timeout".to_owned()]
    );
    assert_eq!(matches[1].record.record_id, "exp-2");
    assert_eq!(matches[1].matched_keywords, vec!["timeout".to_owned()]);
    assert_eq!(matches[2].record.record_id, "exp-3");
    assert!(matches[2].matched_keywords.is_empty());
    assert!(matches[0].score > matches[1].score);
    assert!(matches[1].score > matches[2].score);

    assert_eq!(archive.query(&query), matches, "ranking is deterministic");

    let limited = archive.query(&query.clone().with_limit(1));
    assert_eq!(limited.len(), 1);
    assert_eq!(limited[0].record.record_id, "exp-1");
    assert_eq!(archive.query(&query.with_limit(0)).len(), 3);
}

#[test]
fn query_ties_are_broken_by_record_id() {
    let mut archive = ExperienceArchive::new();
    for (id, content) in [
        ("exp-b", "bravo lesson"),
        ("exp-a", "alpha lesson"),
        ("exp-c", "charlie lesson"),
    ] {
        archive
            .add(record(id, content).with_keywords(["shared"]))
            .unwrap();
    }
    let matches = archive.query(&ExperienceQuery::new().with_keywords(["shared"]));
    let ids: Vec<&str> = matches
        .iter()
        .map(|entry| entry.record.record_id.as_str())
        .collect();
    assert_eq!(ids, vec!["exp-a", "exp-b", "exp-c"]);
}

#[test]
fn query_filters_by_skill_section_and_shared() {
    let mut archive = ExperienceArchive::new();
    archive
        .add(record("exp-1", "tool lesson").with_keywords(["lesson"]))
        .unwrap();
    archive
        .add(
            ExperienceRecord::new("exp-2", "skill-b", ExperienceSection::Body, "body lesson")
                .expect("valid record")
                .with_keywords(["lesson"])
                .mark_shared(),
        )
        .unwrap();

    let by_skill = archive.query(&ExperienceQuery::new().with_skill("skill-b"));
    assert_eq!(by_skill.len(), 1);
    assert_eq!(by_skill[0].record.record_id, "exp-2");

    let by_section = archive.query(&ExperienceQuery::new().with_section(ExperienceSection::Tool));
    assert_eq!(by_section.len(), 1);
    assert_eq!(by_section[0].record.record_id, "exp-1");

    let local_only = archive.query(&ExperienceQuery::new().include_shared(false));
    assert_eq!(local_only.len(), 1);
    assert_eq!(local_only[0].record.record_id, "exp-1");

    let text = archive.query(&ExperienceQuery::new().with_text("body lesson"));
    assert_eq!(text[0].record.record_id, "exp-2");
    assert!(text[0].score > text[1].score);
}

#[test]
fn query_is_filter_only_without_text_or_keywords() {
    let query = ExperienceQuery::new();
    assert!(query.is_filter_only());
    assert!(!query.clone().with_text("hello").is_filter_only());
    assert!(!query.with_keywords(["hello"]).is_filter_only());
}

#[test]
fn top_orders_by_score_then_id() {
    let mut archive = ExperienceArchive::new();
    archive
        .add(
            record("exp-1", "first lesson")
                .record_success()
                .record_success(),
        )
        .unwrap();
    archive.add(record("exp-2", "second lesson")).unwrap();
    archive.add(record("exp-3", "third lesson")).unwrap();

    let top = archive.top(2);
    assert_eq!(top.len(), 2);
    assert_eq!(top[0].record_id, "exp-1");
    assert_eq!(top[1].record_id, "exp-2");
    assert_eq!(archive.top(0).len(), 0);
    assert_eq!(archive.top(10).len(), 3);
}

#[test]
fn by_skill_section_and_shared_preserve_insertion_order() {
    let mut archive = ExperienceArchive::new();
    archive.add(record("exp-1", "first tool lesson")).unwrap();
    archive.add(record("exp-2", "second tool lesson")).unwrap();
    archive
        .add(
            ExperienceRecord::new("exp-3", "skill-a", ExperienceSection::Body, "body lesson")
                .expect("valid record")
                .mark_shared(),
        )
        .unwrap();

    let tool: Vec<&str> = archive
        .by_skill("skill-a")
        .iter()
        .map(|record| record.record_id.as_str())
        .collect();
    assert_eq!(tool, vec!["exp-1", "exp-2", "exp-3"]);

    let sections: Vec<&str> = archive
        .by_section(ExperienceSection::Tool)
        .iter()
        .map(|record| record.record_id.as_str())
        .collect();
    assert_eq!(sections, vec!["exp-1", "exp-2"]);

    let shared: Vec<&str> = archive
        .shared()
        .iter()
        .map(|record| record.record_id.as_str())
        .collect();
    assert_eq!(shared, vec!["exp-3"]);
}

#[test]
fn decay_scales_weights_and_rejects_out_of_range_factors() {
    let mut archive = ExperienceArchive::new();
    archive.add(record("exp-1", "first lesson")).unwrap();
    archive
        .add(record("exp-2", "second lesson").with_weight(0.0).unwrap())
        .unwrap();

    assert_eq!(archive.decay(0.5).unwrap(), 1, "zero weight cannot decay");
    assert!((archive.get("exp-1").unwrap().weight - 0.5).abs() < 1e-12);
    assert_eq!(archive.decay(1.0).unwrap(), 0);

    for factor in [-0.5, 1.5, f64::NAN] {
        assert!(archive.decay(factor).is_err(), "factor {factor} must fail");
    }
}

#[test]
fn prune_below_removes_light_records() {
    let mut archive = ExperienceArchive::new();
    archive
        .add(record("exp-1", "light lesson").with_weight(0.05).unwrap())
        .unwrap();
    archive
        .add(record("exp-2", "heavy lesson").with_weight(1.0).unwrap())
        .unwrap();

    assert_eq!(archive.prune_below(0.1), 1);
    assert!(archive.get("exp-1").is_none());
    assert!(archive.get("exp-2").is_some());
    assert_eq!(archive.prune_below(0.1), 0);
}

#[test]
fn merge_from_dedupes_across_archives() {
    let mut left = ExperienceArchive::new().with_dedupe_threshold(0.5);
    let mut right = ExperienceArchive::new().with_dedupe_threshold(0.5);
    left.add(
        record("exp-1", "raise the timeout before parsing large reports")
            .record_success()
            .record_success(),
    )
    .unwrap();
    right
        .add(record(
            "exp-2",
            "raise the timeout before parsing large report",
        ))
        .unwrap();
    right
        .add(record("exp-3", "cite the failing span index"))
        .unwrap();

    let summary = left.merge_from(&right).unwrap();
    assert_eq!(summary.inserted, 1);
    assert_eq!(summary.merged, 1);
    assert_eq!(left.len(), 2);
    assert_eq!(left.get("exp-1").unwrap().successes, 2);
}

#[test]
fn stats_aggregate_counts_means_and_sections() {
    let mut archive = ExperienceArchive::new();
    archive
        .add(
            record("exp-1", "first lesson")
                .with_confidence(0.8)
                .unwrap()
                .record_success()
                .record_success(),
        )
        .unwrap();
    archive
        .add(
            ExperienceRecord::new("exp-2", "skill-a", ExperienceSection::Body, "body lesson")
                .expect("valid record")
                .with_confidence(0.4)
                .unwrap()
                .mark_shared(),
        )
        .unwrap();

    let stats: ExperienceStats = archive.stats();
    assert_eq!(stats.count, 2);
    assert_eq!(stats.shared, 1);
    assert!((stats.mean_confidence - 0.6).abs() < 1e-12);
    assert!((stats.mean_success_rate - 0.5).abs() < 1e-12);
    assert_eq!(stats.sections.get("tool"), Some(&1));
    assert_eq!(stats.sections.get("body"), Some(&1));

    let empty = ExperienceArchive::new().stats();
    assert_eq!(empty.count, 0);
    assert!(empty.sections.is_empty());
    assert!(empty.mean_confidence.abs() < f64::EPSILON);
}

#[test]
fn archive_round_trips_through_json_with_defaults() {
    let mut archive = ExperienceArchive::new().with_dedupe_threshold(0.7);
    archive
        .add(
            record("exp-1", "raise the timeout")
                .with_keywords(["timeout"])
                .record_success(),
        )
        .unwrap();

    let value = serde_json::to_value(&archive).unwrap();
    assert!(value.get("records").is_some());
    assert_eq!(value["capacity"], json!(512));
    assert_eq!(value["dedupe_threshold"], json!(0.7));
    assert_eq!(value["seq"], json!(1));

    let restored: ExperienceArchive = serde_json::from_value(value).unwrap();
    assert_eq!(restored, archive);
}

#[test]
fn record_serde_ignores_unknown_fields_and_defaults_missing_ones() {
    let payload = json!({
        "record_id": "exp-1",
        "skill_name": "skill-a",
        "section": "tool",
        "content": "raise the timeout",
        "provider_extension": {"nested": true}
    });
    let record: ExperienceRecord = serde_json::from_value(payload).unwrap();
    assert_eq!(record.section, ExperienceSection::Tool);
    assert_eq!(record.source, "experience");
    assert_eq!(record.confidence, 1.0);
    assert_eq!(record.weight, 1.0);
    assert_eq!(record.hits, 0);
    assert!(!record.shared);
    assert!(record.keywords.is_empty());

    let round_tripped: ExperienceRecord =
        serde_json::from_value(serde_json::to_value(&record).unwrap()).unwrap();
    assert_eq!(round_tripped, record);
}

#[test]
fn section_projects_signal_targets() {
    for (target, expected) in [
        (EvolutionTarget::Instruction, ExperienceSection::Description),
        (EvolutionTarget::Description, ExperienceSection::Description),
        (EvolutionTarget::Body, ExperienceSection::Body),
        (EvolutionTarget::Script, ExperienceSection::Script),
        (EvolutionTarget::Tool, ExperienceSection::Tool),
        (EvolutionTarget::Memory, ExperienceSection::Memory),
        (EvolutionTarget::Skill, ExperienceSection::Other),
        (EvolutionTarget::Context, ExperienceSection::Other),
        (EvolutionTarget::Other, ExperienceSection::Other),
    ] {
        assert_eq!(
            ExperienceSection::from(target),
            expected,
            "target {target:?}"
        );
    }
}

#[test]
fn extractor_builds_records_from_signals() {
    let signals = vec![
        EvolutionSignal::new(
            "execution_failure",
            EvolutionTarget::Tool,
            "  the read_file tool timed out  on a 12 MB PDF\n",
        )
        .unwrap()
        .with_skill("pdf-report")
        .with_source("evaluator")
        .with_confidence(0.9)
        .unwrap(),
        EvolutionSignal::new(
            "low_score",
            EvolutionTarget::Body,
            "answer missed the units",
        )
        .unwrap()
        .with_confidence(0.2)
        .unwrap(),
    ];

    let records = ExperienceExtractor::new().from_signals(&signals);
    assert_eq!(records.len(), 1, "the low-confidence signal is dropped");
    let record = &records[0];
    assert_eq!(record.skill_name, "pdf-report");
    assert_eq!(record.section, ExperienceSection::Tool);
    assert_eq!(
        record.content, "the read_file tool timed out on a 12 MB PDF",
        "content is whitespace-normalised and case-preserving"
    );
    assert_eq!(record.source, "evaluator");
    assert!((record.confidence - 0.9).abs() < 1e-12);
    assert!(record.keywords.contains(&"read_file".to_owned()));
    assert!(record.keywords.len() <= 8);
    assert_eq!(
        record.created_seq, 0,
        "the archive assigns sequence numbers"
    );

    let duplicate =
        ExperienceExtractor::new().from_signals(&[signals[0].clone(), signals[0].clone()]);
    assert_eq!(duplicate.len(), 1, "identical evidence yields one record");
    assert_eq!(
        duplicate[0].record_id, record.record_id,
        "ids are content-derived"
    );
}

#[test]
fn extractor_falls_back_to_the_default_skill_name() {
    let signal = EvolutionSignal::new("note", EvolutionTarget::Memory, "cache warmup helps")
        .unwrap()
        .with_source("review");
    let records = ExperienceExtractor::new().from_signals(&[signal]);
    assert_eq!(records.len(), 1);
    assert_eq!(
        records[0].skill_name,
        openjiuwen_agent_evolving::experience::DEFAULT_SKILL_NAME
    );
    assert_eq!(records[0].section, ExperienceSection::Memory);
}

#[test]
fn extractor_respects_max_records() {
    let signals: Vec<EvolutionSignal> = (0..5)
        .map(|index| {
            EvolutionSignal::new(
                "note",
                EvolutionTarget::Body,
                format!("lesson number {index} about caching"),
            )
            .unwrap()
        })
        .collect();
    let records = ExperienceExtractor::new()
        .with_max_records(2)
        .from_signals(&signals);
    assert_eq!(records.len(), 2);
}

#[test]
fn extractor_builds_an_avoid_record_from_a_failure() {
    let mut trajectory = trajectory("traj-1", TrajectoryOutcome::Failure)
        .with_metadata("skill_name", json!("pdf-report"));
    trajectory.push(TrajectoryStep::new(StepKind::Message, "parse the report").with_role("user"));
    trajectory.push(
        TrajectoryStep::new(StepKind::ToolCall, "read_file timed out after 30s")
            .with_tool_name("read_file")
            .with_status(StepStatus::Error),
    );

    let records = ExperienceExtractor::new().from_trajectories(&[trajectory]);
    assert_eq!(records.len(), 1);
    let record = &records[0];
    assert_eq!(record.skill_name, "pdf-report");
    assert_eq!(record.section, ExperienceSection::Tool);
    assert!(record.content.starts_with("avoid: "));
    assert!(record.content.contains("read_file"));
    assert_eq!(record.failures, 1);
    assert_eq!(record.successes, 0);
    assert_eq!(record.hits, 1);
    assert_eq!(record.source, "trajectory");
}

#[test]
fn extractor_uses_the_digest_when_a_failure_has_no_error_step() {
    let mut trajectory = trajectory("traj-2", TrajectoryOutcome::Failure);
    trajectory.push(TrajectoryStep::new(StepKind::Message, "summarise the file"));
    let records = ExperienceExtractor::new().from_trajectories(&[trajectory]);
    assert_eq!(records.len(), 1);
    assert!(records[0].content.starts_with("avoid: #0 message"));
    assert_eq!(records[0].section, ExperienceSection::Other);
}

#[test]
fn extractor_builds_a_tool_order_record_from_a_success() {
    let mut trajectory = trajectory("traj-3", TrajectoryOutcome::Success);
    trajectory.push(TrajectoryStep::new(StepKind::Message, "publish the report").with_role("user"));
    for tool in ["read_file", "extract_tables", "write_file"] {
        trajectory.push(
            TrajectoryStep::new(StepKind::ToolCall, "{}")
                .with_tool_name(tool)
                .with_status(StepStatus::Ok),
        );
    }

    let records = ExperienceExtractor::new().from_trajectories(&[trajectory]);
    assert_eq!(records.len(), 1);
    let record = &records[0];
    assert_eq!(record.section, ExperienceSection::Tool);
    assert_eq!(
        record.content,
        "when doing publish the report, call tools in order: read_file -> extract_tables -> write_file"
    );
    assert_eq!(record.successes, 1);
    assert_eq!(record.hits, 1);
    assert_eq!(record.skill_name, "unknown");
}

#[test]
fn extractor_skips_thin_or_undecided_trajectories() {
    let mut single_tool = trajectory("traj-4", TrajectoryOutcome::Success);
    single_tool.push(TrajectoryStep::new(StepKind::ToolCall, "{}").with_tool_name("read_file"));

    let empty = trajectory("traj-5", TrajectoryOutcome::Failure);

    let mut unknown = trajectory("traj-6", TrajectoryOutcome::Unknown);
    unknown.push(TrajectoryStep::new(StepKind::ToolCall, "{}").with_tool_name("read_file"));

    let records = ExperienceExtractor::new().from_trajectories(&[single_tool, empty, unknown]);
    assert!(records.is_empty());
}

#[test]
fn extractor_builds_records_from_failing_cases() {
    let case = Case::new(string_map("question", "2 + 2?"), string_map("answer", "4"))
        .unwrap()
        .with_case_id("case-1");
    let failing = EvaluatedCase::new(case.clone(), None, 0.2, "wrong units").unwrap();
    let passing = EvaluatedCase::new(case, Some(string_map("answer", "4")), 0.9, "exact").unwrap();

    let records = ExperienceExtractor::new().from_evaluated_cases(&[failing, passing], 0.5);
    assert_eq!(records.len(), 1);
    let record = &records[0];
    assert!(record.content.contains("case-1"));
    assert!(record.content.contains("scored 0.20"));
    assert_eq!(record.source, "evaluated_case");
    assert_eq!(record.failures, 1);
    assert!((record.confidence - 0.8).abs() < 1e-12);
    assert_eq!(record.section, ExperienceSection::Other);
}

#[test]
fn extractor_merge_dedupes_a_batch_deterministically() {
    let extractor = ExperienceExtractor::new().with_min_confidence(0.1);
    let batch = vec![
        record(
            "exp-b",
            "raise the request timeout before parsing large pdf reports",
        ),
        record("exp-a", "cite the failing span index"),
        record(
            "exp-c",
            "before parsing large pdf reports raise the request timeout",
        ),
    ];
    let merged = extractor.merge(batch.clone());
    assert_eq!(merged.len(), 2, "the near-duplicate pair collapsed");
    assert_eq!(
        merged
            .iter()
            .map(|record| record.record_id.as_str())
            .collect::<Vec<_>>(),
        vec!["exp-a", "exp-b"],
        "output is ordered by id, not by input order"
    );

    let mut reversed = batch;
    reversed.reverse();
    let reversed = extractor.merge(reversed);
    assert_eq!(
        reversed.len(),
        2,
        "input order does not change how many survive"
    );
    assert_eq!(
        reversed
            .iter()
            .map(|record| record.record_id.as_str())
            .collect::<Vec<_>>(),
        vec!["exp-a", "exp-c"],
        "the first-seen near-duplicate keeps the id"
    );

    let filtered =
        extractor
            .with_min_confidence(0.9)
            .merge(vec![record("exp-d", "low confidence")
                .with_confidence(0.2)
                .unwrap()]);
    assert!(filtered.is_empty());
}

#[test]
fn lifecycle_promotes_proven_records() {
    let mut archive = ExperienceArchive::new();
    archive
        .add(
            record("exp-1", "proven lesson")
                .record_success()
                .record_success()
                .record_success(),
        )
        .unwrap();
    archive
        .add(record("exp-2", "unproven lesson").record_success())
        .unwrap();

    let report = ExperienceLifecycle::default().apply(&mut archive);
    assert_eq!(report.promoted, vec!["exp-1".to_owned()]);
    assert!(report.pruned.is_empty());
    assert_eq!(report.decayed, 0);
    assert!(archive.get("exp-1").unwrap().shared);
    assert!(!archive.get("exp-2").unwrap().shared);
    assert_eq!(archive.get("exp-1").unwrap().updated_seq, archive.seq());

    assert_eq!(
        ExperienceLifecycle::default().apply(&mut archive).promoted,
        Vec::<String>::new(),
        "an already shared record is not promoted twice"
    );
}

#[test]
fn lifecycle_prunes_stale_and_light_records() {
    let mut archive = ExperienceArchive::new();
    archive
        .add(
            record("exp-1", "stale lesson")
                .record_success()
                .record_success(),
        )
        .unwrap();
    archive
        .add(record("exp-2", "fresh lesson").with_weight(0.05).unwrap())
        .unwrap();
    assert_eq!(archive.seq(), 2);

    let config = LifecycleConfig {
        min_hits: 10,
        promote_success_rate: 0.7,
        stale_seq_gap: 0,
        min_weight: 0.1,
    };
    let report = ExperienceLifecycle::new(config).apply(&mut archive);

    assert_eq!(
        report.pruned,
        vec!["exp-1".to_owned(), "exp-2".to_owned()],
        "exp-1 is stale, exp-2 is below the weight floor"
    );
    assert_eq!(report.decayed, 1, "only the stale record was decayed");
    assert!(report.promoted.is_empty());
    assert!(archive.is_empty());
}

#[test]
fn lifecycle_leaves_healthy_archives_untouched() {
    let mut archive = ExperienceArchive::new();
    archive
        .add(record("exp-1", "healthy lesson").record_success())
        .unwrap();

    let report = ExperienceLifecycle::default().apply(&mut archive);
    assert!(report.promoted.is_empty());
    assert!(report.pruned.is_empty());
    assert_eq!(report.decayed, 0);
    assert_eq!(archive.len(), 1);
}

#[test]
fn lifecycle_rebuild_preserves_dedupe_and_rejects_invalid_input() {
    let lifecycle = ExperienceLifecycle::default();
    let rebuilt = lifecycle
        .rebuild(vec![
            record("exp-1", "raise the timeout before parsing large reports"),
            record("exp-2", "before parsing large reports raise the timeout"),
            record("exp-3", "cite the failing span index"),
        ])
        .expect("valid records");

    assert_eq!(rebuilt.len(), 2, "the near-duplicate pair merged");
    assert_eq!(rebuilt.get("exp-1").unwrap().hits, 1);
    assert!(rebuilt.get("exp-3").is_some());

    let mut broken = record("exp-4", "content");
    broken.confidence = 3.0;
    let error = lifecycle.rebuild(vec![broken]).expect_err("invalid input");
    assert!(matches!(error, Error::Validation(_)));
}
