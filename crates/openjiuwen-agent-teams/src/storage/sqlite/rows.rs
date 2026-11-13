//! Row encoding and decoding for the SQLite team store.
//!
//! The SQLite backend persists each record as one flat row: enums are stored as their wire string
//! (`as_str`), booleans as `INTEGER`, counters and millisecond stamps as `INTEGER`, and the list
//! columns (`reviewers`, `change_notes`) as a JSON array in a `TEXT` column. This module owns those
//! translations plus the single `sqlx::Error` mapping, so `super::SqliteTeamStore` only builds
//! statements.
//!
//! Column names match Python where they can; `desc` is a SQL keyword, so every `description`
//! column carries the record's `desc` field.

use serde::de::DeserializeOwned;
use serde::Serialize;
use serde_json::Value;
use sqlx::sqlite::SqliteRow;
use sqlx::Row;

use openjiuwen_common::error::{Error, Result};

use crate::schema::{
    ExecutionStatus, MemberStatus, ReviewVote, TaskPatch, TaskRecord, TaskRevision, TaskStatus,
    TeamRole, Verdict,
};
use crate::storage::records::{MemberRecord, MessageRecord, TeamInfoRecord};
use crate::storage::stamps::refresh;

/// Context attached to every error the SQLite driver reports.
const SQLITE_CONTEXT: &str = "sqlite team store";

/// Maps a driver error onto the crate error type, keeping the driver error as the source.
pub(super) fn sqlite_error(error: sqlx::Error) -> Error {
    Error::external(error).with_context(SQLITE_CONTEXT)
}

/// Converts an unsigned counter or stamp to the signed integer SQLite stores.
///
/// A value above `i64::MAX` saturates rather than wrapping, so a caller-supplied stamp that large is
/// still stored (and read back) as a sane, ordered value instead of a negative one.
#[must_use]
pub(super) fn to_i64(value: u64) -> i64 {
    i64::try_from(value).unwrap_or(i64::MAX)
}

/// Applies a [`TaskPatch`] to `record` with the in-memory backend's exact semantics.
///
/// A changed title or content bumps [`TaskRecord::revision`] (unless the patch sets it explicitly)
/// and appends the rendered [`TaskRevision`] note; an assignee is cleared before it is set; a
/// caller-supplied `updated_at` wins over the store stamp. The caller writes the whole row
/// afterwards, so this helper only mutates the record.
pub(super) fn apply_task_patch(record: &mut TaskRecord, patch: &TaskPatch) {
    let before = record.clone();
    let mut edited: Vec<&'static str> = Vec::new();
    if let Some(title) = &patch.title {
        if record.title != *title {
            record.title.clone_from(title);
            edited.push("title");
        }
    }
    if let Some(content) = &patch.content {
        if record.content != *content {
            record.content.clone_from(content);
            edited.push("content");
        }
    }
    if let Some(status) = patch.status {
        record.status = status;
    }
    if patch.clear_assignee {
        record.assignee = None;
    } else if let Some(assignee) = &patch.assignee {
        record.assignee = Some(assignee.clone());
    }
    if let Some(reviewers) = &patch.reviewers {
        record.reviewers.clone_from(reviewers);
    }
    if let Some(round) = patch.review_round {
        record.review_round = round;
    }
    if let Some(rounds) = patch.max_review_rounds {
        record.max_review_rounds = rounds;
    }
    if let Some(summary) = &patch.summary {
        record.summary = Some(summary.clone());
    }
    if let Some(note) = &patch.note {
        record.note = Some(note.clone());
    }
    if let Some(parent) = &patch.parent_task_id {
        record.parent_task_id = Some(parent.clone());
    }
    if let Some(observed) = patch.observed_revision {
        record.observed_revision = observed;
    }
    let bumped = patch.revision.is_none() && !edited.is_empty();
    if let Some(revision) = patch.revision {
        record.revision = revision;
    } else if bumped {
        record.revision = record.revision.saturating_add(1);
        let note = match edited.as_slice() {
            ["title"] => "title updated",
            ["content"] => "content updated",
            _ => "title and content updated",
        };
        record.push_change_note(
            TaskRevision {
                revision: record.revision,
                note: note.to_owned(),
                by: None,
            }
            .to_note(),
        );
    }
    if let Some(updated_at) = patch.updated_at {
        record.updated_at = updated_at;
    } else if *record != before {
        record.updated_at = refresh(before.updated_at);
    }
}

/// Serializes a list column (reviewers, change notes) into its JSON `TEXT` form.
pub(super) fn encode_list<T: Serialize>(values: &[T]) -> Result<String> {
    serde_json::to_string(values).map_err(|error| {
        Error::external(error).with_context("sqlite team store: encode a JSON column")
    })
}

/// Serializes an enum column to its wire string, the inverse of [`decode_enum`].
///
/// `serde` already knows the wire name of every schema enum (the `snake_case` rename), so the backend
/// does not repeat those names; a non-string representation is reported rather than stored.
pub(super) fn encode_enum<T: Serialize>(value: T) -> Result<String> {
    match serde_json::to_value(value) {
        Ok(Value::String(raw)) => Ok(raw),
        Ok(other) => Err(Error::config(format!(
            "sqlite team store: enum column encoded as {other}, not a string"
        ))),
        Err(error) => {
            Err(Error::external(error).with_context("sqlite team store: encode an enum column"))
        }
    }
}

/// Decodes an enum column: the wire string round-trips through `serde`'s `snake_case` name.
///
/// Reading through `serde` instead of a hand-written match keeps the persisted values identical to
/// the `as_str` / serde pair the rest of the crate asserts on, and a value written by a newer build
/// is reported instead of silently replaced by the default.
pub(super) fn decode_enum<T: DeserializeOwned>(raw: &str, column: &'static str) -> Result<T> {
    serde_json::from_value(Value::String(raw.to_owned())).map_err(|error| {
        Error::config(format!(
            "sqlite team store: invalid {column} value '{raw}': {error}"
        ))
    })
}

/// Decodes one team row into its record.
pub(super) fn decode_team(row: &SqliteRow) -> Result<TeamInfoRecord> {
    Ok(TeamInfoRecord {
        team_name: text(row, "team_name")?,
        display_name: text(row, "display_name")?,
        leader_member_name: text(row, "leader_member_name")?,
        desc: text(row, "description")?,
        prompt: text(row, "prompt")?,
        dispatch_mode: decode_enum(&text(row, "dispatch_mode")?, "dispatch_mode")?,
        enable_task_verification: boolean(row, "enable_task_verification")?,
        created: unsigned(row, "created")?,
        updated_at: unsigned(row, "updated_at")?,
    })
}

/// Decodes one roster row into its record.
pub(super) fn decode_member(row: &SqliteRow) -> Result<MemberRecord> {
    Ok(MemberRecord {
        team_name: text(row, "team_name")?,
        member_name: text(row, "member_name")?,
        display_name: text(row, "display_name")?,
        desc: text(row, "description")?,
        role: decode_enum::<TeamRole>(&text(row, "role")?, "role")?,
        status: decode_enum::<MemberStatus>(&text(row, "status")?, "status")?,
        execution: decode_enum::<ExecutionStatus>(&text(row, "execution")?, "execution")?,
        mode: decode_enum(&text(row, "mode")?, "mode")?,
        prompt: text(row, "prompt")?,
        model_ref: optional_text(row, "model_ref")?,
        worktree: optional_text(row, "worktree")?,
        updated_at: unsigned(row, "updated_at")?,
    })
}

/// Decodes one task row into its record.
pub(super) fn decode_task(row: &SqliteRow) -> Result<TaskRecord> {
    Ok(TaskRecord {
        team_name: text(row, "team_name")?,
        task_id: text(row, "task_id")?,
        title: text(row, "title")?,
        content: text(row, "content")?,
        status: decode_enum::<TaskStatus>(&text(row, "status")?, "status")?,
        assignee: optional_text(row, "assignee")?,
        reviewers: decode_list(&text(row, "reviewers")?, "reviewers")?,
        review_round: counter(row, "review_round")?,
        max_review_rounds: counter(row, "max_review_rounds")?,
        summary: optional_text(row, "summary")?,
        note: optional_text(row, "note")?,
        parent_task_id: optional_text(row, "parent_task_id")?,
        parent_revision: optional_counter(row, "parent_revision")?,
        created_by: optional_text(row, "created_by")?,
        revision: counter(row, "revision")?,
        observed_revision: counter(row, "observed_revision")?,
        change_notes: decode_list(&text(row, "change_notes")?, "change_notes")?,
        updated_at: unsigned(row, "updated_at")?,
    })
}

/// Decodes one review-vote row into its record.
pub(super) fn decode_vote(row: &SqliteRow) -> Result<ReviewVote> {
    Ok(ReviewVote {
        team_name: text(row, "team_name")?,
        task_id: text(row, "task_id")?,
        round: counter(row, "review_round")?,
        reviewer: text(row, "reviewer")?,
        verdict: decode_enum::<Verdict>(&text(row, "verdict")?, "verdict")?,
    })
}

/// Decodes one mailbox row into its record.
pub(super) fn decode_message(row: &SqliteRow) -> Result<MessageRecord> {
    Ok(MessageRecord {
        team_name: text(row, "team_name")?,
        message_id: text(row, "message_id")?,
        from_member_name: text(row, "from_member_name")?,
        to_member_name: optional_text(row, "to_member_name")?,
        content: text(row, "content")?,
        timestamp: unsigned(row, "timestamp")?,
        broadcast: boolean(row, "broadcast")?,
        protocol: text(row, "protocol")?,
        is_read: boolean(row, "is_read")?,
    })
}

/// Decodes a JSON list column; a value that does not parse is reported, never defaulted.
fn decode_list<T: DeserializeOwned>(raw: &str, column: &'static str) -> Result<Vec<T>> {
    serde_json::from_str(raw).map_err(|error| {
        Error::config(format!("sqlite team store: invalid {column} JSON: {error}"))
    })
}

/// Reads a `NOT NULL` text column.
fn text(row: &SqliteRow, column: &'static str) -> Result<String> {
    row.try_get::<String, _>(column).map_err(sqlite_error)
}

/// Reads a nullable text column.
fn optional_text(row: &SqliteRow, column: &'static str) -> Result<Option<String>> {
    row.try_get::<Option<String>, _>(column)
        .map_err(sqlite_error)
}

/// Reads an `INTEGER` column as the signed value SQLite stores.
fn integer(row: &SqliteRow, column: &'static str) -> Result<i64> {
    row.try_get::<i64, _>(column).map_err(sqlite_error)
}

/// Reads an `INTEGER` column as an unsigned millisecond stamp (`0` for a negative value).
fn unsigned(row: &SqliteRow, column: &'static str) -> Result<u64> {
    Ok(u64::try_from(integer(row, column)?).unwrap_or_default())
}

/// Reads an `INTEGER` column as a `u32` counter (`0` outside the `u32` range).
fn counter(row: &SqliteRow, column: &'static str) -> Result<u32> {
    Ok(u32::try_from(integer(row, column)?).unwrap_or_default())
}

/// Reads a nullable `INTEGER` column as an optional `u32` counter.
fn optional_counter(row: &SqliteRow, column: &'static str) -> Result<Option<u32>> {
    Ok(row
        .try_get::<Option<i64>, _>(column)
        .map_err(sqlite_error)?
        .and_then(|value| u32::try_from(value).ok()))
}

/// Reads an `INTEGER` flag (`0` / `1`) as a boolean.
fn boolean(row: &SqliteRow, column: &'static str) -> Result<bool> {
    row.try_get::<bool, _>(column).map_err(sqlite_error)
}
