//! The trajectory record: typed steps, outcome, and a bounded digest.
//!
//! Ports the data model of Python `openjiuwen.agent_evolving.trajectory`
//! (`model.py`, `schema.py`, `serialization.py`) to an idiomatic serde record.
//! Python keeps a canonical OTLP payload and projects spans out of it; this
//! crate keeps a typed step list with the same `snake_case` field names, which
//! is enough for offline evaluation, experience extraction, and prompt
//! injection.
//!
//! No clock is ever read: [`Trajectory::elapsed_ms`] is caller-supplied, so
//! recordings stay deterministic and serializable.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// The excerpt length used when one step is rendered into a digest line.
const DIGEST_STEP_EXCERPT_CHARS: usize = 80;

/// The kind of action one [`TrajectoryStep`] records.
///
/// Python classifies spans by name (`llm`, `tool`, `agent`, ...); the typed
/// kind keeps the same information without the vendor span vocabulary.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StepKind {
    /// A conversation message (user or assistant text).
    Message,
    /// A tool invocation requested by the model.
    ToolCall,
    /// The result a tool returned for a [`StepKind::ToolCall`].
    ToolResult,
    /// Raw model output.
    Model,
    /// A system prompt or system instruction.
    System,
    /// Any other recorded action.
    #[default]
    Other,
}

impl StepKind {
    /// The stable `snake_case` label, used by digests and JSON output.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Message => "message",
            Self::ToolCall => "tool_call",
            Self::ToolResult => "tool_result",
            Self::Model => "model",
            Self::System => "system",
            Self::Other => "other",
        }
    }
}

impl fmt::Display for StepKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Whether one step (or span) succeeded or failed.
///
/// Python infers failure from span status codes; this crate records it
/// explicitly, defaulting to [`StepStatus::Ok`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StepStatus {
    /// The action completed.
    #[default]
    Ok,
    /// The action failed; see [`TrajectoryStep::is_error`].
    Error,
}

impl StepStatus {
    /// The stable `snake_case` label used by JSON output.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Ok => "ok",
            Self::Error => "error",
        }
    }
}

impl fmt::Display for StepStatus {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// The overall result of a recorded run.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrajectoryOutcome {
    /// The run met its objective.
    Success,
    /// The run failed.
    Failure,
    /// The outcome was not decided (the default while recording).
    #[default]
    Unknown,
}

impl TrajectoryOutcome {
    /// The stable `snake_case` label used by JSON output.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Success => "success",
            Self::Failure => "failure",
            Self::Unknown => "unknown",
        }
    }
}

impl fmt::Display for TrajectoryOutcome {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// One recorded action inside a [`Trajectory`].
///
/// The `index` is assigned by [`Trajectory::push`]/[`Trajectory::extend`], so
/// callers building steps by hand start at `0` and stay contiguous. The
/// builder-style `with_*` methods consume and return the step.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TrajectoryStep {
    /// Position inside the trajectory; assigned by the trajectory on insertion.
    pub index: usize,
    /// What the action was.
    pub kind: StepKind,
    /// Speaker or actor role as a wire string (`user`, `assistant`, `tool`, ...).
    pub role: Option<String>,
    /// The recorded text; never `None`, so renderers need no fallback.
    pub content: String,
    /// Tool name for [`StepKind::ToolCall`] steps and, when known, results.
    pub tool_name: Option<String>,
    /// Provider call id linking a tool call to its result.
    pub tool_call_id: Option<String>,
    /// Whether the action succeeded.
    pub status: StepStatus,
    /// Free-form annotations; keys are stable and serialized sorted.
    pub metadata: BTreeMap<String, Value>,
}

impl Default for TrajectoryStep {
    fn default() -> Self {
        Self {
            index: 0,
            kind: StepKind::Other,
            role: None,
            content: String::new(),
            tool_name: None,
            tool_call_id: None,
            status: StepStatus::Ok,
            metadata: BTreeMap::new(),
        }
    }
}

impl TrajectoryStep {
    /// Creates a step of `kind` carrying `content`.
    ///
    /// `index` starts at `0`; the owning trajectory overwrites it when the
    /// step is pushed. `role`, tool fields, and metadata stay unset.
    #[must_use]
    pub fn new(kind: StepKind, content: impl Into<String>) -> Self {
        Self {
            kind,
            content: content.into(),
            ..Self::default()
        }
    }

    /// Sets the speaker/actor role string.
    #[must_use]
    pub fn with_role(mut self, role: impl Into<String>) -> Self {
        self.role = Some(role.into());
        self
    }

    /// Sets the tool name.
    #[must_use]
    pub fn with_tool_name(mut self, tool_name: impl Into<String>) -> Self {
        self.tool_name = Some(tool_name.into());
        self
    }

    /// Sets the provider call id.
    #[must_use]
    pub fn with_tool_call_id(mut self, tool_call_id: impl Into<String>) -> Self {
        self.tool_call_id = Some(tool_call_id.into());
        self
    }

    /// Sets the step status.
    #[must_use]
    pub fn with_status(mut self, status: StepStatus) -> Self {
        self.status = status;
        self
    }

    /// Adds one metadata entry, replacing a previous value for the same key.
    #[must_use]
    pub fn with_metadata(mut self, key: impl Into<String>, value: Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Whether the step records a failure.
    #[must_use]
    pub fn is_error(&self) -> bool {
        self.status == StepStatus::Error
    }

    /// Returns at most `max_chars` characters of [`content`](Self::content).
    ///
    /// Truncation is UTF-8 safe (it counts characters, never bytes) and the
    /// last kept character becomes `…` when the text was cut. A `max_chars`
    /// of `0` yields an empty string.
    #[must_use]
    pub fn excerpt(&self, max_chars: usize) -> String {
        truncate_chars(&self.content, max_chars)
    }
}

/// One recorded run: ordered steps plus identity, outcome, and scoring data.
///
/// Python stores the run as an immutable OTLP payload with resource
/// attributes; this crate stores the typed projection directly. Every field
/// is `serde`-defaulted and unknown JSON fields are ignored, so recordings
/// from older or newer producers still deserialize.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Trajectory {
    /// Stable identifier of the run (Python `openjiuwen.trajectory_id`).
    #[serde(default)]
    pub trajectory_id: String,
    /// Conversation/session identity (Python `gen_ai.conversation.id`).
    #[serde(default)]
    pub session_id: Option<String>,
    /// Evaluation case identity, when the run was produced from a dataset case.
    #[serde(default)]
    pub case_id: Option<String>,
    /// The recorded steps in execution order.
    #[serde(default)]
    pub steps: Vec<TrajectoryStep>,
    /// The overall result of the run.
    #[serde(default)]
    pub outcome: TrajectoryOutcome,
    /// Reward or evaluation score; `None` while unscored.
    #[serde(default)]
    pub score: Option<f64>,
    /// Wall-clock duration in milliseconds; caller-supplied, never measured here.
    #[serde(default)]
    pub elapsed_ms: Option<u64>,
    /// Free-form run annotations; keys are stable and serialized sorted.
    #[serde(default)]
    pub metadata: BTreeMap<String, Value>,
}

impl Trajectory {
    /// Creates an empty trajectory with the given identity.
    ///
    /// The id must be non-empty after trimming; a blank id is rejected with
    /// [`Error::validation`] instead of producing an unaddressable record.
    pub fn new(trajectory_id: impl Into<String>) -> Result<Self> {
        let trajectory_id = trajectory_id.into();
        if trajectory_id.trim().is_empty() {
            return Err(Error::validation(
                "Trajectory.trajectory_id must not be empty",
            ));
        }
        Ok(Self {
            trajectory_id,
            session_id: None,
            case_id: None,
            steps: Vec::new(),
            outcome: TrajectoryOutcome::Unknown,
            score: None,
            elapsed_ms: None,
            metadata: BTreeMap::new(),
        })
    }

    /// Appends one step, assigning its [`index`](TrajectoryStep::index) to the
    /// current step count.
    pub fn push(&mut self, mut step: TrajectoryStep) {
        step.index = self.steps.len();
        self.steps.push(step);
    }

    /// Appends every step in iteration order, assigning contiguous indices.
    pub fn extend<I>(&mut self, steps: I)
    where
        I: IntoIterator<Item = TrajectoryStep>,
    {
        for step in steps {
            self.push(step);
        }
    }

    /// The recorded steps in execution order.
    #[must_use]
    pub fn steps(&self) -> &[TrajectoryStep] {
        &self.steps
    }

    /// Number of recorded steps.
    #[must_use]
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Whether the trajectory has no steps.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Sets the session identity.
    #[must_use]
    pub fn with_session(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }

    /// Sets the evaluation case identity.
    #[must_use]
    pub fn with_case(mut self, case_id: impl Into<String>) -> Self {
        self.case_id = Some(case_id.into());
        self
    }

    /// Sets the overall outcome.
    #[must_use]
    pub fn with_outcome(mut self, outcome: TrajectoryOutcome) -> Self {
        self.outcome = outcome;
        self
    }

    /// Sets the score, rejecting non-finite values (`NaN`, `±∞`).
    ///
    /// Python has no such guard; a non-finite score is unusable for ranking
    /// and would not round-trip through JSON, so it is a validation error.
    pub fn with_score(mut self, score: f64) -> Result<Self> {
        if !score.is_finite() {
            return Err(Error::validation(
                "Trajectory.score must be a finite number",
            ));
        }
        self.score = Some(score);
        Ok(self)
    }

    /// Sets the caller-measured duration in milliseconds.
    #[must_use]
    pub fn with_elapsed_ms(mut self, elapsed_ms: u64) -> Self {
        self.elapsed_ms = Some(elapsed_ms);
        self
    }

    /// Adds one metadata entry, replacing a previous value for the same key.
    #[must_use]
    pub fn with_metadata(mut self, key: impl Into<String>, value: Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Number of steps whose status is [`StepStatus::Error`].
    #[must_use]
    pub fn error_count(&self) -> usize {
        self.steps.iter().filter(|step| step.is_error()).count()
    }

    /// The distinct tool names used, sorted ascending.
    #[must_use]
    pub fn tool_names(&self) -> Vec<&str> {
        self.steps
            .iter()
            .filter_map(|step| step.tool_name.as_deref())
            .filter(|name| !name.is_empty())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }

    /// Whether the recorded outcome is [`TrajectoryOutcome::Success`].
    #[must_use]
    pub fn is_successful(&self) -> bool {
        self.outcome == TrajectoryOutcome::Success
    }

    /// Builds a compact summary bounded to `max_chars` characters.
    ///
    /// The summary renders one `#index kind role: excerpt` line per step
    /// (each excerpt capped at 80 characters) and is then truncated to
    /// `max_chars` with a trailing `…` when cut. The result is deterministic:
    /// it depends only on the recorded values.
    #[must_use]
    pub fn digest(&self, max_chars: usize) -> TrajectoryDigest {
        let rendered = self
            .steps
            .iter()
            .map(render_step_line)
            .collect::<Vec<_>>()
            .join("\n");
        TrajectoryDigest {
            trajectory_id: self.trajectory_id.clone(),
            outcome: self.outcome,
            score: self.score,
            step_count: self.steps.len(),
            tool_names: self.tool_names().into_iter().map(str::to_owned).collect(),
            summary: truncate_chars(&rendered, max_chars),
        }
    }
}

/// A compact, bounded projection of one [`Trajectory`].
///
/// Used where a full recording is too large to embed — prompt context,
/// experience records, and logs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TrajectoryDigest {
    /// Identifier of the summarized trajectory.
    pub trajectory_id: String,
    /// The summarized outcome.
    pub outcome: TrajectoryOutcome,
    /// The score, when one was recorded.
    pub score: Option<f64>,
    /// How many steps the trajectory holds.
    pub step_count: usize,
    /// The distinct tool names used, sorted ascending.
    pub tool_names: Vec<String>,
    /// The bounded rendering of the steps.
    pub summary: String,
}

impl Default for TrajectoryDigest {
    fn default() -> Self {
        Self {
            trajectory_id: String::new(),
            outcome: TrajectoryOutcome::Unknown,
            score: None,
            step_count: 0,
            tool_names: Vec::new(),
            summary: String::new(),
        }
    }
}

/// Renders one step as `#index kind role: excerpt`, without a trailing newline.
pub(crate) fn render_step_line(step: &TrajectoryStep) -> String {
    let role = step.role.as_deref().unwrap_or("-");
    format!(
        "#{} {} {}: {}",
        step.index,
        step.kind.as_str(),
        role,
        step.excerpt(DIGEST_STEP_EXCERPT_CHARS)
    )
}

/// Truncates `text` to at most `max_chars` characters, UTF-8 safely.
///
/// When cut, the last kept character is replaced by `…`, so the result is
/// exactly `max_chars` characters. Works on `char`s, never splits a code point.
pub(crate) fn truncate_chars(text: &str, max_chars: usize) -> String {
    if max_chars == 0 {
        return String::new();
    }
    if text.chars().count() <= max_chars {
        return text.to_owned();
    }
    let keep = max_chars - 1;
    let mut truncated: String = text.chars().take(keep).collect();
    truncated.push('…');
    truncated
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn new_rejects_blank_id() {
        let error = Trajectory::new("  ").expect_err("blank id must fail");
        assert!(matches!(error, Error::Validation(_)));
    }

    #[test]
    fn push_assigns_contiguous_indices() {
        let mut trajectory = Trajectory::new("t-1").expect("valid id");
        trajectory.push(TrajectoryStep::new(StepKind::Message, "hello"));
        trajectory.push(TrajectoryStep::new(StepKind::ToolCall, "{}"));
        let indices: Vec<usize> = trajectory.steps().iter().map(|step| step.index).collect();
        assert_eq!(indices, vec![0, 1]);
    }

    #[test]
    fn excerpt_is_utf8_safe_and_bounded() {
        let step = TrajectoryStep::new(StepKind::Message, "日本語テキスト");
        assert_eq!(step.excerpt(3), "日本…");
        assert_eq!(step.excerpt(10), "日本語テキスト");
        assert_eq!(step.excerpt(0), "");
    }

    #[test]
    fn score_must_be_finite() {
        let trajectory = Trajectory::new("t-1").expect("valid id");
        assert!(trajectory.clone().with_score(f64::NAN).is_err());
        assert!(trajectory.with_score(0.5).is_ok());
    }

    #[test]
    fn digest_is_deterministic_and_bounded() {
        let mut trajectory = Trajectory::new("t-1").expect("valid id");
        trajectory.push(TrajectoryStep::new(StepKind::Message, "hi").with_role("user"));
        let digest = trajectory.digest(1_000);
        assert_eq!(digest.step_count, 1);
        assert_eq!(digest.summary, "#0 message user: hi");
        assert_eq!(digest, trajectory.digest(1_000));
        assert_eq!(trajectory.digest(5).summary.chars().count(), 5);
    }

    #[test]
    fn metadata_and_tools_are_aggregated() {
        let mut trajectory = Trajectory::new("t-1")
            .expect("valid id")
            .with_metadata("source", json!("offline"));
        trajectory.push(
            TrajectoryStep::new(StepKind::ToolCall, "{}")
                .with_tool_name("search")
                .with_status(StepStatus::Error),
        );
        trajectory.push(TrajectoryStep::new(StepKind::ToolCall, "{}").with_tool_name("search"));
        trajectory.push(TrajectoryStep::new(StepKind::ToolCall, "{}").with_tool_name("calc"));
        assert_eq!(trajectory.tool_names(), vec!["calc", "search"]);
        assert_eq!(trajectory.error_count(), 1);
        assert_eq!(trajectory.metadata.get("source"), Some(&json!("offline")));
    }
}
