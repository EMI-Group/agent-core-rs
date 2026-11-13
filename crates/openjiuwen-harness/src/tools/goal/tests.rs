//! The persisted goal domain model parity suite: statuses, assessment, record round-trips.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use chrono::{DateTime, TimeZone, Timelike, Utc};
use serde_json::json;

use openjiuwen_common::error::Error;

use super::record::GoalRecord;
use super::schema::*;

/// A fixed clock for deterministic timestamps.
fn now() -> DateTime<Utc> {
    Utc.with_ymd_and_hms(2026, 1, 2, 3, 4, 5).unwrap()
}

/// A clock one minute later.
fn later() -> DateTime<Utc> {
    Utc.with_ymd_and_hms(2026, 1, 2, 3, 5, 10).unwrap()
}

#[test]
fn statuses_use_python_strings() {
    for status in GoalStatus::ALL {
        assert_eq!(GoalStatus::parse(status.as_str()), Some(status));
    }
    for status in GoalAssessmentStatus::ALL {
        assert_eq!(GoalAssessmentStatus::parse(status.as_str()), Some(status));
    }
    assert_eq!(
        GoalStatus::ALL.map(GoalStatus::as_str),
        ["active", "paused", "completed", "blocked"]
    );
    assert_eq!(
        GoalAssessmentStatus::ALL.map(GoalAssessmentStatus::as_str),
        ["continue", "complete", "blocked"]
    );
}

#[test]
fn unknown_statuses_fall_back() {
    assert_eq!(GoalStatus::parse("achieved"), None);
    assert_eq!(GoalStatus::parse_lenient("achieved"), GoalStatus::Active);
    assert_eq!(
        GoalAssessmentStatus::parse_lenient("done"),
        GoalAssessmentStatus::Continue
    );
    assert_eq!(
        serde_json::from_value::<GoalStatus>(json!("cancelled")).unwrap(),
        GoalStatus::Active
    );
    assert_eq!(
        serde_json::from_value::<GoalStatus>(json!(7)).unwrap(),
        GoalStatus::Active
    );
    assert_eq!(
        serde_json::from_value::<GoalAssessmentStatus>(json!(null)).unwrap(),
        GoalAssessmentStatus::Continue
    );
}

#[test]
fn new_record_is_active_and_timestamped() {
    let record = GoalRecord::new("  ship it  ", now()).with_session_id("s-1");
    assert_eq!(record.objective, "ship it");
    assert_eq!(record.session_id, "s-1");
    assert_eq!(record.status, GoalStatus::Active);
    assert_eq!(record.goal_id.len(), 12);
    assert!(record.goal_id.chars().all(|c| c.is_ascii_hexdigit()));
    let timestamp = now_iso(now());
    assert_eq!(record.created_at, timestamp);
    assert_eq!(record.updated_at, timestamp);
    assert_eq!(record.active_started_at, Some(timestamp));
    assert_eq!(record.revision, 0);
    assert_eq!(record.attempt_count, 0);
    assert_eq!(record.time_used_seconds, 0);
    assert!(record.is_active());
}

#[test]
fn now_iso_matches_python_isoformat() {
    let stamp = now_iso(now());
    assert_eq!(stamp, "2026-01-02T03:04:05.000000+00:00");
    assert_eq!(parse_timestamp(&stamp), Some(now()));
    assert_eq!(
        parse_timestamp("2026-01-02T03:04:05Z"),
        Some(now().with_nanosecond(0).unwrap())
    );
    assert_eq!(parse_timestamp("2026-01-02T03:04:05"), None);
    assert_eq!(parse_timestamp("not a timestamp"), None);
}

#[test]
fn record_round_trips_through_json() {
    let mut record = GoalRecord::new("ship it", now())
        .with_session_id("s-1")
        .with_limits(Some(3), Some(1000));
    record.token_usage.accumulate(10, 5, 2);
    record.record_progress(
        &GoalAssessment::new(
            GoalAssessmentStatus::Blocked,
            "missing key",
            Some("ask the user".to_owned()),
            None,
        ),
        later(),
    );

    let dict = record.to_dict();
    for key in [
        "goal_id",
        "session_id",
        "objective",
        "status",
        "revision",
        "attempt_count",
        "token_usage",
        "max_attempts",
        "token_budget",
        "last_assessment",
        "last_stop_reason",
        "time_used_seconds",
        "active_started_at",
        "created_at",
        "updated_at",
    ] {
        assert!(dict.get(key).is_some(), "to_dict is missing {key}");
    }
    assert_eq!(dict["token_usage"]["total_tokens"], json!(15));
    assert_eq!(dict["last_assessment"]["status"], json!("blocked"));
    assert_eq!(dict["last_stop_reason"], json!("blocked"));
    assert_eq!(dict["active_started_at"], json!(null));

    let parsed = GoalRecord::from_dict(&dict).unwrap();
    assert_eq!(parsed, record);
    assert_eq!(parsed.status, GoalStatus::Blocked);
    assert_eq!(parsed.time_used_seconds, 65);

    let through_serde: GoalRecord = serde_json::from_value(dict).unwrap();
    assert_eq!(through_serde, record);
    assert_eq!(serde_json::to_value(&record).unwrap(), record.to_dict());
}

#[test]
fn from_dict_tolerates_missing_optional_fields() {
    let parsed = GoalRecord::from_dict(&json!({
        "goal_id": "abc123abc123",
        "session_id": "s-1",
        "objective": "ship it",
        "created_at": now_iso(now()),
    }))
    .unwrap();
    assert_eq!(parsed.status, GoalStatus::Active);
    assert_eq!(parsed.revision, 0);
    assert_eq!(parsed.attempt_count, 0);
    assert_eq!(parsed.time_used_seconds, 0);
    assert_eq!(parsed.max_attempts, None);
    assert_eq!(parsed.token_usage, TokenUsage::default());
    assert_eq!(parsed.last_assessment, None);
    // A missing `updated_at` follows `created_at`, and the ACTIVE clock is healed.
    assert_eq!(parsed.updated_at, parsed.created_at);
    assert_eq!(parsed.active_started_at, Some(parsed.created_at.clone()));
    // Missing timestamps heal from `updated_at`; missing unknown keys are ignored.
    let healed = GoalRecord::from_dict(&json!({
        "goal_id": "abcdefabcdef",
        "session_id": "s-1",
        "objective": "ship it",
        "updated_at": now_iso(now()),
        "unknown_key": {"nested": true},
    }))
    .unwrap();
    assert_eq!(healed.created_at, now_iso(now()));
}

#[test]
fn unknown_status_degrades_to_active() {
    let parsed = GoalRecord::from_dict(&json!({
        "goal_id": "abcdefabcdef",
        "session_id": "s-1",
        "objective": "ship it",
        "status": "achieved",
        "created_at": now_iso(now()),
    }))
    .unwrap();
    assert_eq!(parsed.status, GoalStatus::Active);
}

#[test]
fn from_dict_reports_invalid_records() {
    let identity = GoalRecord::from_dict(&json!({"objective": "ship it"})).unwrap_err();
    assert_eq!(identity.message(), "invalid persisted GoalRecord identity");
    assert!(matches!(identity, Error::Validation(_)));

    let empty = GoalRecord::from_dict(&json!({
        "goal_id": "",
        "session_id": "s-1",
        "objective": "ship it",
    }))
    .unwrap_err();
    assert_eq!(empty.message(), "invalid persisted GoalRecord identity");

    let budget = GoalRecord::from_dict(&json!({
        "goal_id": "abc",
        "session_id": "s-1",
        "objective": "ship it",
        "token_budget": 0,
    }))
    .unwrap_err();
    assert_eq!(
        budget.message(),
        "invalid persisted GoalRecord token_budget"
    );

    let attempts = GoalRecord::from_dict(&json!({
        "goal_id": "abc",
        "session_id": "s-1",
        "objective": "ship it",
        "max_attempts": "many",
    }))
    .unwrap_err();
    assert_eq!(
        attempts.message(),
        "invalid persisted GoalRecord max_attempts"
    );

    let created = GoalRecord::from_dict(&json!({
        "goal_id": "abc",
        "session_id": "s-1",
        "objective": "ship it",
        "created_at": "yesterday",
    }))
    .unwrap_err();
    assert_eq!(created.message(), "invalid persisted GoalRecord created_at");

    let inactive = GoalRecord::from_dict(&json!({
        "goal_id": "abc",
        "session_id": "s-1",
        "objective": "ship it",
        "created_at": now_iso(now()),
        "active_started_at": 5,
    }))
    .unwrap_err();
    assert_eq!(
        inactive.message(),
        "invalid persisted GoalRecord active_started_at"
    );

    let usage = GoalRecord::from_dict(&json!({
        "goal_id": "abc",
        "session_id": "s-1",
        "objective": "ship it",
        "created_at": now_iso(now()),
        "time_used_seconds": -1,
    }))
    .unwrap_err();
    assert_eq!(
        usage.message(),
        "invalid persisted GoalRecord time_used_seconds"
    );
}

#[test]
fn assessment_normalizes_like_python() {
    let assessment = GoalAssessment::from_dict(&json!({
        "status": "complete",
        "evidence": "tests pass",
        "remaining_work": "",
        "next_instruction": null,
    }));
    assert_eq!(assessment.status, GoalAssessmentStatus::Complete);
    assert_eq!(assessment.evidence, "tests pass");
    assert_eq!(assessment.remaining_work, None);
    assert_eq!(assessment.next_instruction, None);

    let fallback = GoalAssessment::from_dict(&json!({"status": "done", "evidence": 5}));
    assert_eq!(fallback.status, GoalAssessmentStatus::Continue);
    assert_eq!(fallback.evidence, "5");

    assert_eq!(
        GoalAssessment::from_dict(&json!("nope")),
        GoalAssessment::default()
    );
    assert_eq!(
        assessment.to_dict(),
        json!({
            "status": "complete",
            "evidence": "tests pass",
            "remaining_work": null,
            "next_instruction": null,
        })
    );
}

#[test]
fn record_progress_maps_status_and_clock() {
    let mut record = GoalRecord::new("ship it", now());
    record.start_timing(now());
    record.record_progress(
        &GoalAssessment::new(GoalAssessmentStatus::Continue, "halfway", None, None),
        later(),
    );
    assert_eq!(record.status, GoalStatus::Active);
    assert_eq!(record.time_used_seconds, 65);
    assert_eq!(record.active_started_at, Some(now_iso(later())));

    record.set_status(GoalStatus::Paused, later());
    record.record_progress(
        &GoalAssessment::new(GoalAssessmentStatus::Continue, "still paused", None, None),
        later(),
    );
    assert_eq!(record.status, GoalStatus::Paused);
    assert_eq!(record.active_started_at, None);

    record.record_progress(
        &GoalAssessment::new(GoalAssessmentStatus::Complete, "done", None, None),
        later(),
    );
    assert_eq!(record.status, GoalStatus::Completed);
    assert_eq!(record.last_stop_reason.as_deref(), Some("completed"));
    assert_eq!(record.active_started_at, None);
    assert!(record.status.is_terminal());
    assert!(!record.is_active());

    // A settled goal ignores a later `continue` assessment.
    record.record_progress(
        &GoalAssessment::new(GoalAssessmentStatus::Continue, "again", None, None),
        later(),
    );
    assert_eq!(record.status, GoalStatus::Completed);
    assert_eq!(record.time_used_seconds, 65);
}

#[test]
fn blocked_assessment_settles_the_goal() {
    let mut record = GoalRecord::new("ship it", now());
    record.record_progress(
        &GoalAssessment::new(GoalAssessmentStatus::Blocked, "no key", None, None),
        later(),
    );
    assert_eq!(record.status, GoalStatus::Blocked);
    assert_eq!(record.last_stop_reason.as_deref(), Some("blocked"));
}

#[test]
fn settle_without_an_open_clock_reopens_only_while_active() {
    let mut record = GoalRecord::new("ship it", now());
    record.active_started_at = None;
    record.settle_active_time(true, later());
    assert_eq!(record.active_started_at, Some(now_iso(later())));
    assert_eq!(record.time_used_seconds, 0);

    record.active_started_at = None;
    record.status = GoalStatus::Paused;
    record.settle_active_time(true, later());
    assert_eq!(record.active_started_at, None);
}

#[test]
fn settle_ignores_an_unparsable_clock() {
    let mut record = GoalRecord::new("ship it", now());
    record.active_started_at = Some("yesterday".to_owned());
    record.settle_active_time(true, later());
    assert_eq!(record.time_used_seconds, 0);
    assert_eq!(record.active_started_at, Some(now_iso(later())));
}

#[test]
fn progress_summary_skips_empty_fields() {
    let record = GoalRecord::new("ship it", now());
    assert_eq!(
        record.progress_summary(),
        "objective: ship it\nstatus: active\nattempt_count: 0"
    );

    let mut record = record;
    record.record_progress(
        &GoalAssessment::new(
            GoalAssessmentStatus::Continue,
            "halfway",
            Some("write docs".to_owned()),
            Some("then tag".to_owned()),
        ),
        now(),
    );
    assert_eq!(
            record.progress_summary(),
            "objective: ship it\nstatus: active\nattempt_count: 0\nevidence: halfway\nremaining_work: write docs\nnext_instruction: then tag"
        );
}

#[test]
fn token_usage_accumulates_like_python() {
    let mut usage = TokenUsage::default();
    usage.accumulate(10, 5, 3);
    usage.accumulate(1, 2, 0);
    assert_eq!(usage.input_tokens, 11);
    assert_eq!(usage.output_tokens, 7);
    assert_eq!(usage.cached_input_tokens, 3);
    assert_eq!(usage.total_tokens, 18);
    assert_eq!(TokenUsage::from_dict(&usage.to_dict()), usage);
    assert_eq!(
        TokenUsage::from_dict(&json!({"input_tokens": "4"})).input_tokens,
        4
    );
    assert_eq!(TokenUsage::from_dict(&json!([])), TokenUsage::default());
}
