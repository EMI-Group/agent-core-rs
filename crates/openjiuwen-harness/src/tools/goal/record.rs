//! The persisted [`GoalRecord`]: its Python JSON keys, validation and mutations.
//!
//! Python parity: `GoalRecord` of `openjiuwen/harness/goal/schema.py`. Every mutation receives
//! `now`, so no helper reads the wall clock; [`GoalRecord::from_dict`] applies Python's persisted
//! record validation and reports [`openjiuwen_common::error::Error::Validation`] with the Python
//! messages. An unknown persisted `status` degrades to `active` with a warning instead of
//! raising, so one bad enum value cannot make the store drop a whole goal.
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use openjiuwen_common::error::{Error, Result};

use super::schema::{
    lenient_text, lenient_u64, now_iso, parse_timestamp, GoalAssessment, GoalAssessmentStatus,
    GoalStatus, TokenUsage, DEFAULT_SESSION_ID,
};
use crate::tools::support::render_fields;

/// The persisted goal of the session owning the store.
///
/// Python parity: `GoalRecord` — the same field names, the same JSON keys and the same
/// validation messages. The record is created by [`GoalRecord::new`], read through
/// [`GoalRecord::from_dict`] and mutated through [`GoalRecord::touch`],
/// [`GoalRecord::start_timing`], [`GoalRecord::settle_active_time`] and
/// [`GoalRecord::record_progress`]; every one of those receives `now`.
///
/// ```no_run
/// use chrono::Utc;
/// use openjiuwen_harness::tools::goal::{GoalAssessment, GoalAssessmentStatus, GoalRecord};
///
/// let now = Utc::now();
/// let mut record = GoalRecord::new("ship it", now).with_session_id("s-1");
/// record.start_timing(now);
/// record.record_progress(
///     &GoalAssessment::new(GoalAssessmentStatus::Complete, "tests pass", None, None),
///     now,
/// );
/// assert_eq!(record.status.as_str(), "completed");
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "Value")]
pub struct GoalRecord {
    /// Identifier of the goal, 12 hex characters like Python's `uuid4().hex[:12]`.
    pub goal_id: String,
    /// Session the goal belongs to.
    pub session_id: String,
    /// The objective in the user's words, trimmed.
    pub objective: String,
    /// Lifecycle status; `active` when absent.
    pub status: GoalStatus,
    /// Generation counter; bumped when a new attempt generation starts.
    pub revision: u64,
    /// Number of attempts started for this goal.
    pub attempt_count: u64,
    /// Cumulative token usage of the attempts.
    pub token_usage: TokenUsage,
    /// Attempt budget, when the host set one.
    pub max_attempts: Option<u64>,
    /// Token budget, when the host set one.
    pub token_budget: Option<u64>,
    /// The most recent assessment, when one was reported.
    pub last_assessment: Option<GoalAssessment>,
    /// Why the goal stopped (`completed`, `blocked`, ...), when it did.
    pub last_stop_reason: Option<String>,
    /// Wall-clock seconds the goal spent in active attempts.
    pub time_used_seconds: u64,
    /// Start of the currently open active segment, when the clock runs.
    pub active_started_at: Option<String>,
    /// Creation timestamp (RFC3339 UTC).
    pub created_at: String,
    /// Last mutation timestamp (RFC3339 UTC).
    pub updated_at: String,
}

impl GoalRecord {
    /// Creates an active record for `objective`, stamped with `now`.
    ///
    /// The objective is trimmed; the caller validates that it is not empty (Python's
    /// `GoalManager.set` does the same before creating the record).
    #[must_use]
    pub fn new(objective: impl Into<String>, now: DateTime<Utc>) -> Self {
        let timestamp = now_iso(now);
        Self {
            goal_id: new_goal_id(),
            session_id: DEFAULT_SESSION_ID.to_owned(),
            objective: objective.into().trim().to_owned(),
            status: GoalStatus::Active,
            revision: 0,
            attempt_count: 0,
            token_usage: TokenUsage::default(),
            max_attempts: None,
            token_budget: None,
            last_assessment: None,
            last_stop_reason: None,
            time_used_seconds: 0,
            active_started_at: Some(timestamp.clone()),
            created_at: timestamp.clone(),
            updated_at: timestamp,
        }
    }

    /// Sets the session id the record belongs to.
    #[must_use]
    pub fn with_session_id(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = session_id.into();
        self
    }

    /// Sets the optional attempt and token budgets.
    #[must_use]
    pub fn with_limits(mut self, max_attempts: Option<u64>, token_budget: Option<u64>) -> Self {
        self.max_attempts = max_attempts;
        self.token_budget = token_budget;
        self
    }

    /// Stamps `updated_at` with `now` without bumping the revision.
    pub fn touch(&mut self, now: DateTime<Utc>) {
        self.updated_at = now_iso(now);
    }

    /// Stamps `updated_at` with `now` and bumps the revision (a new attempt generation).
    pub fn bump_revision(&mut self, now: DateTime<Utc>) {
        self.revision = self.revision.saturating_add(1);
        self.touch(now);
    }

    /// Opens the active-time clock if it is closed, and stamps `updated_at`.
    ///
    /// Python parity: `start_timing` is a no-op while a segment is already open.
    pub fn start_timing(&mut self, now: DateTime<Utc>) {
        let timestamp = now_iso(now);
        if self.active_started_at.is_none() {
            self.active_started_at = Some(timestamp.clone());
        }
        self.updated_at = timestamp;
    }

    /// Closes (or re-opens) the active-time clock and folds the elapsed seconds into
    /// [`Self::time_used_seconds`].
    ///
    /// Python parity: `settle_active_time`. A stored `active_started_at` that does not parse
    /// cannot be measured; it is logged and the elapsed time is treated as zero instead of
    /// aborting the mutation ([`Self::from_dict`] never stores such a value).
    pub fn settle_active_time(&mut self, keep_active: bool, now: DateTime<Utc>) {
        let timestamp = now_iso(now);
        if let Some(started_at) = self.active_started_at.as_deref() {
            if let Some(started) = parse_timestamp(started_at) {
                let elapsed = now.signed_duration_since(started).num_seconds().max(0);
                self.time_used_seconds = self
                    .time_used_seconds
                    .saturating_add(u64::try_from(elapsed).unwrap_or(0));
            } else {
                tracing::warn!(
                    goal_id = %self.goal_id,
                    active_started_at = %started_at,
                    "unparsable goal active_started_at; elapsed time ignored"
                );
            }
            self.active_started_at = if keep_active {
                Some(timestamp.clone())
            } else {
                None
            };
        } else if keep_active && self.status == GoalStatus::Active {
            self.active_started_at = Some(timestamp.clone());
        }
        self.updated_at = timestamp;
    }

    /// Commits one assessment, updating the lifecycle status like Python's
    /// `GoalManager.apply_assessment`.
    ///
    /// - `complete` settles the clock and moves the goal to `completed`
    ///   (`last_stop_reason = "completed"`).
    /// - `blocked` settles the clock and moves the goal to `blocked`
    ///   (`last_stop_reason = "blocked"`).
    /// - `continue` keeps the clock open while the goal is `active`, and closes it while the
    ///   goal is `paused` (the attempt finished during a pause).
    ///
    /// The assessment is always stored as [`Self::last_assessment`] and `updated_at` is stamped;
    /// the revision is deliberately not bumped, matching Python.
    pub fn record_progress(&mut self, assessment: &GoalAssessment, now: DateTime<Utc>) {
        self.last_assessment = Some(assessment.clone());
        match assessment.status {
            GoalAssessmentStatus::Complete => {
                self.settle_active_time(false, now);
                self.status = GoalStatus::Completed;
                self.last_stop_reason = Some("completed".to_owned());
            }
            GoalAssessmentStatus::Blocked => {
                self.settle_active_time(false, now);
                self.status = GoalStatus::Blocked;
                self.last_stop_reason = Some("blocked".to_owned());
            }
            GoalAssessmentStatus::Continue => match self.status {
                GoalStatus::Active => self.settle_active_time(true, now),
                GoalStatus::Paused => self.settle_active_time(false, now),
                GoalStatus::Completed | GoalStatus::Blocked => {}
            },
        }
        self.touch(now);
    }

    /// Sets the lifecycle status and stamps `updated_at`.
    pub fn set_status(&mut self, status: GoalStatus, now: DateTime<Utc>) {
        self.status = status;
        self.touch(now);
    }

    /// Whether the goal is being worked on.
    #[must_use]
    pub const fn is_active(&self) -> bool {
        matches!(self.status, GoalStatus::Active)
    }

    /// A compact progress/next-step summary, one `key: value` line per non-empty field.
    ///
    /// The order is objective, status, attempt count, then the last assessment's evidence,
    /// remaining work and next instruction; absent parts are skipped like Python's
    /// `render_fields`.
    #[must_use]
    pub fn progress_summary(&self) -> String {
        let assessment = self.last_assessment.clone().unwrap_or_default();
        render_fields(&[
            ("objective", json!(self.objective)),
            ("status", json!(self.status.as_str())),
            ("attempt_count", json!(self.attempt_count)),
            ("evidence", json!(assessment.evidence)),
            ("remaining_work", json!(assessment.remaining_work)),
            ("next_instruction", json!(assessment.next_instruction)),
        ])
    }

    /// Renders the record as the Python `to_dict` object, keys in Python order.
    #[must_use]
    pub fn to_dict(&self) -> Value {
        json!({
            "goal_id": self.goal_id,
            "session_id": self.session_id,
            "objective": self.objective,
            "status": self.status.as_str(),
            "revision": self.revision,
            "attempt_count": self.attempt_count,
            "token_usage": self.token_usage.to_dict(),
            "max_attempts": self.max_attempts,
            "token_budget": self.token_budget,
            "last_assessment": self.last_assessment.as_ref().map(GoalAssessment::to_dict),
            "last_stop_reason": self.last_stop_reason,
            "time_used_seconds": self.time_used_seconds,
            "active_started_at": self.active_started_at,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        })
    }

    /// Reads a persisted record with Python's validation rules.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] with Python's message when the identity fields are missing
    /// or empty, a budget is not positive, a timestamp is missing or unparsable, or an integer
    /// field holds a non-integer. An unknown `status` degrades to `active` with a warning.
    pub fn from_dict(data: &Value) -> Result<Self> {
        Self::try_from(data.clone())
    }
}

impl TryFrom<Value> for GoalRecord {
    type Error = Error;

    fn try_from(data: Value) -> Result<Self> {
        let object = data
            .as_object()
            .ok_or_else(|| Error::validation("invalid persisted GoalRecord identity"))?;
        let identity = ["objective", "session_id", "goal_id"];
        let mut parts = Vec::with_capacity(identity.len());
        for field in identity {
            match object.get(field).and_then(Value::as_str) {
                Some(text) if !text.is_empty() => parts.push(text.to_owned()),
                _ => return Err(Error::validation("invalid persisted GoalRecord identity")),
            }
        }
        let Some(objective) = parts.first().cloned() else {
            return Err(Error::validation("invalid persisted GoalRecord identity"));
        };
        let Some(session_id) = parts.get(1).cloned() else {
            return Err(Error::validation("invalid persisted GoalRecord identity"));
        };
        let Some(goal_id) = parts.get(2).cloned() else {
            return Err(Error::validation("invalid persisted GoalRecord identity"));
        };

        let status = match object.get("status").and_then(Value::as_str) {
            None => GoalStatus::Active,
            Some(text) => GoalStatus::parse(text).unwrap_or_else(|| {
                tracing::warn!(
                    goal_id = %goal_id,
                    status = %text,
                    "unknown persisted goal status; falling back to active"
                );
                GoalStatus::Active
            }),
        };

        let token_usage = object
            .get("token_usage")
            .filter(|value| value.is_object())
            .map_or_else(TokenUsage::default, TokenUsage::from_dict);
        let max_attempts = optional_positive_int(object.get("max_attempts"), "max_attempts")?;
        let token_budget = optional_positive_int(object.get("token_budget"), "token_budget")?;
        let last_assessment = object
            .get("last_assessment")
            .filter(|value| value.is_object())
            .map(GoalAssessment::from_dict);
        let last_stop_reason = object.get("last_stop_reason").and_then(lenient_text);
        let time_used_seconds = match object.get("time_used_seconds") {
            None | Some(Value::Null) => 0,
            Some(value) => strict_u64(value, "time_used_seconds")?,
        };

        let created_at = match timestamp_field(object.get("created_at"), "created_at")? {
            Some(timestamp) => Some(timestamp),
            None => timestamp_field(object.get("updated_at"), "updated_at")?,
        };
        let created_at = created_at
            .ok_or_else(|| Error::validation("invalid persisted GoalRecord created_at"))?;
        let updated_at = match timestamp_field(object.get("updated_at"), "updated_at")? {
            Some(timestamp) => timestamp,
            None => created_at.clone(),
        };
        let mut active_started_at =
            timestamp_field(object.get("active_started_at"), "active_started_at")?;
        // ACTIVE records keep a start mark for further accumulation: heal rather than reject
        // when persistence dropped the optional timestamp.
        if status == GoalStatus::Active && active_started_at.is_none() {
            active_started_at = Some(updated_at.clone());
        }

        Ok(Self {
            goal_id,
            session_id,
            objective,
            status,
            revision: optional_non_negative_int(object.get("revision"), "revision")?,
            attempt_count: optional_non_negative_int(object.get("attempt_count"), "attempt_count")?,
            token_usage,
            max_attempts,
            token_budget,
            last_assessment,
            last_stop_reason,
            time_used_seconds,
            active_started_at,
            created_at,
            updated_at,
        })
    }
}

/// Generates the 12-hex-character goal id used by Python (`uuid4().hex[:12]`).
fn new_goal_id() -> String {
    uuid::Uuid::new_v4().simple().to_string()[..12].to_owned()
}

/// Validates one optional timestamp field, returning it verbatim when it parses.
fn timestamp_field(value: Option<&Value>, field: &str) -> Result<Option<String>> {
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(text)) if !text.is_empty() && parse_timestamp(text).is_some() => {
            Ok(Some(text.clone()))
        }
        Some(_) => Err(Error::validation(format!(
            "invalid persisted GoalRecord {field}"
        ))),
    }
}

/// Reads a required non-negative integer field with Python's message.
fn strict_u64(value: &Value, field: &str) -> Result<u64> {
    lenient_u64(value)
        .ok_or_else(|| Error::validation(format!("invalid persisted GoalRecord {field}")))
}

/// Reads an optional field counting from zero, defaulting to zero when absent.
fn optional_non_negative_int(value: Option<&Value>, field: &str) -> Result<u64> {
    match value {
        None | Some(Value::Null) => Ok(0),
        Some(value) => strict_u64(value, field),
    }
}

/// Reads an optional strictly-positive field (Python `_optional_positive_int`).
fn optional_positive_int(value: Option<&Value>, field: &str) -> Result<Option<u64>> {
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(value) => match strict_u64(value, field)? {
            0 => Err(Error::validation(format!(
                "invalid persisted GoalRecord {field}"
            ))),
            parsed => Ok(Some(parsed)),
        },
    }
}
