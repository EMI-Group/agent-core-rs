//! Goal statuses, assessment, token usage and the JSON value helpers.
//!
//! Ports `openjiuwen/harness/goal/schema.py` with the Python field names, the Python JSON keys
//! and the Python validation messages. Two deliberate differences:
//!
//! - Timestamps are RFC3339 UTC strings and every helper receives `now` from its caller, so no
//!   helper reads the wall clock and the tests are deterministic. Python's `_utc_now_iso()` has
//!   no counterpart here; [`now_iso`] is the only formatter.
//! - An unknown persisted `status` degrades to [`GoalStatus::Active`] with a warning instead of
//!   raising, so one bad enum value cannot make the store drop a whole goal. Structural
//!   corruption (missing identity, a non-positive limit, an unparsable timestamp) is still an
//!   `Error::Validation`.
use chrono::{DateTime, SecondsFormat, Utc};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{json, Value};

/// Session id stamped into a record created without an explicit session.
pub const DEFAULT_SESSION_ID: &str = "default";

/// Lifecycle status of a persisted goal.
///
/// Python parity: `GoalStatus` (`active`, `paused`, `completed`, `blocked`). The values are the
/// persisted strings, so they never change with Rust naming.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum GoalStatus {
    /// The goal is being worked on.
    #[default]
    Active,
    /// The goal is suspended; no new attempt is queued.
    Paused,
    /// The goal was reported complete.
    Completed,
    /// The goal cannot progress without outside help.
    Blocked,
}

impl GoalStatus {
    /// Every status, in Python declaration order.
    pub const ALL: [Self; 4] = [Self::Active, Self::Paused, Self::Completed, Self::Blocked];

    /// The persisted string of this status.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Active => "active",
            Self::Paused => "paused",
            Self::Completed => "completed",
            Self::Blocked => "blocked",
        }
    }

    /// Parses the exact persisted string, or `None` for anything else.
    ///
    /// Python parity: `GoalStatus(value)` raises for an unknown value; the caller decides
    /// whether that is an error (`GoalRecord::from_dict` degrades instead).
    #[must_use]
    pub fn parse(value: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|status| status.as_str() == value)
    }

    /// Parses `value`, falling back to [`Self::Active`] when it is unknown.
    ///
    /// Python parity: `data.get("status", GoalStatus.ACTIVE.value)` defaults a *missing*
    /// status to `active`; the unknown-value fallback is this port's tolerance rule.
    #[must_use]
    pub fn parse_lenient(value: &str) -> Self {
        Self::parse(value).unwrap_or_default()
    }

    /// Whether the status is settled, so no further attempt should be scheduled.
    #[must_use]
    pub const fn is_terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Blocked)
    }
}

impl std::fmt::Display for GoalStatus {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl Serialize for GoalStatus {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for GoalStatus {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        Ok(match Value::deserialize(deserializer)? {
            Value::String(text) => Self::parse_lenient(&text),
            // Python would raise on a non-string; a persisted record is kept instead.
            _ => Self::default(),
        })
    }
}

/// Verdict of one goal attempt, as reported by the model.
///
/// Python parity: `GoalAssessmentStatus` (`continue`, `complete`, `blocked`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum GoalAssessmentStatus {
    /// Progress was made and the objective can still be advanced.
    #[default]
    Continue,
    /// The objective is achieved, with auditable evidence.
    Complete,
    /// No meaningful progress is possible without outside help.
    Blocked,
}

impl GoalAssessmentStatus {
    /// Every assessment status, in Python declaration order.
    pub const ALL: [Self; 3] = [Self::Continue, Self::Complete, Self::Blocked];

    /// The persisted string of this status.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Continue => "continue",
            Self::Complete => "complete",
            Self::Blocked => "blocked",
        }
    }

    /// Parses the exact persisted string, or `None` for anything else.
    #[must_use]
    pub fn parse(value: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|status| status.as_str() == value)
    }

    /// Parses `value`, falling back to [`Self::Continue`] when it is unknown.
    ///
    /// Python parity: `GoalAssessment.from_dict` and `SubmitGoalReportTool` both normalize an
    /// unknown status to `continue`.
    #[must_use]
    pub fn parse_lenient(value: &str) -> Self {
        Self::parse(value).unwrap_or_default()
    }

    /// Whether the status settles the goal it assesses.
    #[must_use]
    pub const fn is_settling(self) -> bool {
        !matches!(self, Self::Continue)
    }
}

impl std::fmt::Display for GoalAssessmentStatus {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl Serialize for GoalAssessmentStatus {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for GoalAssessmentStatus {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        Ok(match Value::deserialize(deserializer)? {
            Value::String(text) => Self::parse_lenient(&text),
            _ => Self::default(),
        })
    }
}

/// One structured assessment of a goal attempt.
///
/// Python parity: `GoalAssessment`. `remaining_work` and `next_instruction` are `None` when
/// they are absent, `null` or the empty string, matching Python's `value or None`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct GoalAssessment {
    /// The verdict of the attempt; defaults to `continue` like Python.
    #[serde(default)]
    pub status: GoalAssessmentStatus,
    /// Evidence for the verdict, as the model reported it.
    #[serde(default, deserialize_with = "deserialize_lenient_string")]
    pub evidence: String,
    /// What is still missing, when the model reported anything.
    #[serde(default, deserialize_with = "deserialize_lenient_opt_string")]
    pub remaining_work: Option<String>,
    /// The instruction the next attempt should follow, when the model reported one.
    #[serde(default, deserialize_with = "deserialize_lenient_opt_string")]
    pub next_instruction: Option<String>,
}

impl GoalAssessment {
    /// Builds an assessment from its parts.
    ///
    /// The optional parts are normalized like Python: an empty string becomes `None`.
    #[must_use]
    pub fn new(
        status: GoalAssessmentStatus,
        evidence: impl Into<String>,
        remaining_work: Option<String>,
        next_instruction: Option<String>,
    ) -> Self {
        Self {
            status,
            evidence: evidence.into(),
            remaining_work: remaining_work.filter(|text| !text.is_empty()),
            next_instruction: next_instruction.filter(|text| !text.is_empty()),
        }
    }

    /// Renders the assessment as the Python `to_dict` object (all four keys, `null`s kept).
    #[must_use]
    pub fn to_dict(&self) -> Value {
        json!({
            "status": self.status.as_str(),
            "evidence": self.evidence,
            "remaining_work": self.remaining_work,
            "next_instruction": self.next_instruction,
        })
    }

    /// Reads an assessment, tolerating any shape.
    ///
    /// Python parity: `GoalAssessment.from_dict`. A missing or unknown `status` becomes
    /// `continue`, a non-object input yields the default assessment, and empty optional text
    /// becomes `None`.
    #[must_use]
    pub fn from_dict(data: &Value) -> Self {
        let Some(object) = data.as_object() else {
            return Self::default();
        };
        Self {
            status: object
                .get("status")
                .and_then(lenient_text)
                .map_or(GoalAssessmentStatus::default(), |status| {
                    GoalAssessmentStatus::parse_lenient(&status)
                }),
            evidence: object
                .get("evidence")
                .and_then(lenient_text)
                .unwrap_or_default(),
            remaining_work: object.get("remaining_work").and_then(lenient_text),
            next_instruction: object.get("next_instruction").and_then(lenient_text),
        }
    }
}

/// Cumulative token usage of a goal.
///
/// Python parity: `TokenUsage`. `total_tokens` accumulates input plus output tokens, exactly like
/// [`TokenUsage::accumulate`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenUsage {
    /// Prompt tokens consumed by the goal's attempts.
    #[serde(default, deserialize_with = "deserialize_lenient_u64")]
    pub input_tokens: u64,
    /// Completion tokens produced by the goal's attempts.
    #[serde(default, deserialize_with = "deserialize_lenient_u64")]
    pub output_tokens: u64,
    /// Prompt tokens served from the provider cache.
    #[serde(default, deserialize_with = "deserialize_lenient_u64")]
    pub cached_input_tokens: u64,
    /// Input plus output tokens, as Python keeps it.
    #[serde(default, deserialize_with = "deserialize_lenient_u64")]
    pub total_tokens: u64,
}

impl TokenUsage {
    /// Adds one round of usage, keeping `total_tokens` in step with Python.
    pub fn accumulate(&mut self, input_tokens: u64, output_tokens: u64, cached_input_tokens: u64) {
        self.input_tokens = self.input_tokens.saturating_add(input_tokens);
        self.output_tokens = self.output_tokens.saturating_add(output_tokens);
        self.cached_input_tokens = self.cached_input_tokens.saturating_add(cached_input_tokens);
        self.total_tokens = self
            .total_tokens
            .saturating_add(input_tokens.saturating_add(output_tokens));
    }

    /// Renders the usage as the Python `to_dict` object.
    #[must_use]
    pub fn to_dict(&self) -> Value {
        json!({
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "total_tokens": self.total_tokens,
        })
    }

    /// Reads the usage, treating a non-object or missing counts as zero.
    #[must_use]
    pub fn from_dict(data: &Value) -> Self {
        let Some(object) = data.as_object() else {
            return Self::default();
        };
        Self {
            input_tokens: object
                .get("input_tokens")
                .and_then(lenient_u64)
                .unwrap_or(0),
            output_tokens: object
                .get("output_tokens")
                .and_then(lenient_u64)
                .unwrap_or(0),
            cached_input_tokens: object
                .get("cached_input_tokens")
                .and_then(lenient_u64)
                .unwrap_or(0),
            total_tokens: object
                .get("total_tokens")
                .and_then(lenient_u64)
                .unwrap_or(0),
        }
    }
}

/// Formats `now` as the RFC3339 UTC string persisted in every timestamp field.
///
/// Python parity: `datetime.isoformat()` — microsecond precision and an explicit `+00:00`
/// offset.
#[must_use]
pub fn now_iso(now: DateTime<Utc>) -> String {
    now.to_rfc3339_opts(SecondsFormat::Micros, false)
}

/// Parses an RFC3339 timestamp into UTC, or `None` when it has no parseable offset.
#[must_use]
pub fn parse_timestamp(text: &str) -> Option<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(text)
        .ok()
        .map(|parsed| parsed.with_timezone(&Utc))
}

/// Reads an optional string field, mapping `null`, non-strings and the empty string to `None`.
pub(super) fn lenient_text(value: &Value) -> Option<String> {
    match value {
        Value::Null => None,
        Value::String(text) if text.is_empty() => None,
        Value::String(text) => Some(text.clone()),
        other => Some(other.to_string()),
    }
}

/// Reads an unsigned integer from a JSON number or a numeric string.
pub(super) fn lenient_u64(value: &Value) -> Option<u64> {
    match value {
        Value::Number(number) => number.as_u64(),
        Value::String(text) => text.trim().parse().ok(),
        _ => None,
    }
}

/// Deserializes a string, coercing `null` to the empty string and other JSON scalars to text.
pub(super) fn deserialize_lenient_string<'de, D>(
    deserializer: D,
) -> std::result::Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    Ok(lenient_text(&Value::deserialize(deserializer)?).unwrap_or_default())
}

/// Deserializes an optional string, mapping `null` and the empty string to `None`.
pub(super) fn deserialize_lenient_opt_string<'de, D>(
    deserializer: D,
) -> std::result::Result<Option<String>, D::Error>
where
    D: Deserializer<'de>,
{
    Ok(lenient_text(&Value::deserialize(deserializer)?))
}

/// Deserializes an unsigned integer, falling back to zero for anything unreadable.
fn deserialize_lenient_u64<'de, D>(deserializer: D) -> std::result::Result<u64, D::Error>
where
    D: Deserializer<'de>,
{
    Ok(lenient_u64(&Value::deserialize(deserializer)?).unwrap_or(0))
}
