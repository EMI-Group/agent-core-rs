//! Dream job data model: states, kinds, triggers, jobs, and metrics.
//!
//! Everything here is plain `serde` data. The types are tolerant on JSON input
//! (`default` at the container level, unknown fields ignored) so a persisted job can
//! gain new fields without breaking older readers. Validation is a pure method
//! ([`DreamJob::validate`]) that the repository runs before it persists anything.
//! [`DreamOutcome`](crate::dreaming::DreamOutcome) and
//! [`DreamRunSummary`](crate::dreaming::DreamRunSummary) live in
//! [`outcome`](crate::dreaming) alongside this model.
use chrono::{DateTime, Utc};
use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt;

use crate::record::MemoryScope;

/// Default attempt budget for a newly built [`DreamJob`].
pub const DEFAULT_MAX_ATTEMPTS: u32 = 3;

/// Lifecycle state of a [`DreamJob`].
///
/// The wire and database names are the `snake_case` variant names. Parsing is
/// deliberately strict: [`DreamState::parse`] is the tolerant, case-insensitive entry
/// point, while `serde` deserialization only accepts the canonical names (the SQLite
/// backend owns tolerant parsing of legacy rows).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, Default,
)]
#[serde(rename_all = "snake_case")]
pub enum DreamState {
    /// Queued and eligible to be claimed.
    #[default]
    Pending,
    /// Claimed by a worker; the job is in flight.
    Running,
    /// Finished successfully.
    Succeeded,
    /// Finished unsuccessfully after exhausting its attempts.
    Failed,
    /// Finished without doing any work (nothing to consolidate).
    Skipped,
    /// Finished without acting because the run was cancelled.
    Cancelled,
}

impl DreamState {
    /// Every state, in declaration order.
    pub const ALL: [Self; 6] = [
        Self::Pending,
        Self::Running,
        Self::Succeeded,
        Self::Failed,
        Self::Skipped,
        Self::Cancelled,
    ];

    /// The `snake_case` name used on the wire and as the database discriminator.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Running => "running",
            Self::Succeeded => "succeeded",
            Self::Failed => "failed",
            Self::Skipped => "skipped",
            Self::Cancelled => "cancelled",
        }
    }

    /// Returns `true` when the job no longer needs to run again.
    ///
    /// [`DreamState::Succeeded`], [`DreamState::Failed`], [`DreamState::Skipped`], and
    /// [`DreamState::Cancelled`] are terminal; `Pending` and `Running` are not.
    #[must_use]
    pub const fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Succeeded | Self::Failed | Self::Skipped | Self::Cancelled
        )
    }

    /// Parses a state name case-insensitively, trimming surrounding whitespace.
    ///
    /// Returns `None` for an unknown name; callers that persist rows decide whether an
    /// unknown state is an error or a fallback.
    #[must_use]
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "pending" => Some(Self::Pending),
            "running" => Some(Self::Running),
            "succeeded" => Some(Self::Succeeded),
            "failed" => Some(Self::Failed),
            "skipped" => Some(Self::Skipped),
            "cancelled" => Some(Self::Cancelled),
            _ => None,
        }
    }
}

impl fmt::Display for DreamState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// What a [`DreamJob`] should consolidate.
///
/// Known variants serialize as their canonical `snake_case` name; [`DreamKind::Custom`]
/// serializes as its own string, and any unknown string deserializes into
/// [`DreamKind::Custom`] instead of failing. The accepted aliases are:
/// `consolidate` → [`Self::Consolidation`], `deduplicate`/`dedup` → [`Self::Dedupe`],
/// `summarise` → [`Self::Summarize`], `prune`/`pruning` → [`Self::Forget`], and
/// `graph` → [`Self::GraphExtraction`].
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub enum DreamKind {
    /// Merge related and near-duplicate memories into compact records (the default).
    #[default]
    Consolidation,
    /// Remove exact and near duplicates without otherwise rewriting memories.
    Dedupe,
    /// Collapse long histories into summaries.
    Summarize,
    /// Drop stale or low-value memories.
    Forget,
    /// Extract entities and relations into graph memory.
    GraphExtraction,
    /// An application-defined pass; the string is the pipeline identifier.
    Custom(String),
}

impl DreamKind {
    /// The canonical wire name, or the custom identifier for [`DreamKind::Custom`].
    ///
    /// The return type is `&str` rather than `&'static str` because a custom kind
    /// borrows its own string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        match self {
            Self::Consolidation => "consolidation",
            Self::Dedupe => "dedupe",
            Self::Summarize => "summarize",
            Self::Forget => "forget",
            Self::GraphExtraction => "graph_extraction",
            Self::Custom(kind) => kind,
        }
    }

    /// Parses a kind name, accepting the documented aliases.
    ///
    /// This never fails: an unknown (or empty) string becomes
    /// [`DreamKind::Custom`] holding the trimmed input. Matching is case-insensitive.
    #[must_use]
    pub fn parse(value: &str) -> Self {
        let canonical = value.trim().to_ascii_lowercase();
        match canonical.as_str() {
            "consolidation" | "consolidate" => Self::Consolidation,
            "dedupe" | "deduplicate" | "dedup" => Self::Dedupe,
            "summarize" | "summarise" => Self::Summarize,
            "forget" | "prune" | "pruning" => Self::Forget,
            "graph_extraction" | "graph" => Self::GraphExtraction,
            _ => Self::Custom(value.trim().to_owned()),
        }
    }
}

impl fmt::Display for DreamKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl From<&str> for DreamKind {
    fn from(value: &str) -> Self {
        Self::parse(value)
    }
}

impl From<String> for DreamKind {
    fn from(value: String) -> Self {
        Self::parse(&value)
    }
}

impl Serialize for DreamKind {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for DreamKind {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = String::deserialize(deserializer)?;
        Ok(Self::parse(&raw))
    }
}

/// What asked for a [`DreamJob`] to be enqueued.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, Default,
)]
#[serde(rename_all = "snake_case")]
pub enum DreamTrigger {
    /// An explicit API call (the default).
    #[default]
    Manual,
    /// A periodic or scheduled sweep; the scheduler itself is deferred.
    Schedule,
    /// The scope accumulated enough messages (see
    /// [`DreamConfig::message_threshold`](crate::dreaming::DreamConfig::message_threshold)).
    MessageThreshold,
    /// The session went idle (see
    /// [`DreamConfig::idle_secs`](crate::dreaming::DreamConfig::idle_secs)).
    Idle,
}

impl DreamTrigger {
    /// The `snake_case` name used on the wire and in the database.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Manual => "manual",
            Self::Schedule => "schedule",
            Self::MessageThreshold => "message_threshold",
            Self::Idle => "idle",
        }
    }

    /// Parses a trigger name case-insensitively, accepting `threshold` for
    /// [`DreamTrigger::MessageThreshold`].
    ///
    /// Returns `None` for an unknown name.
    #[must_use]
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "manual" => Some(Self::Manual),
            "schedule" => Some(Self::Schedule),
            "message_threshold" | "threshold" => Some(Self::MessageThreshold),
            "idle" => Some(Self::Idle),
            _ => None,
        }
    }
}

impl fmt::Display for DreamTrigger {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Counters describing what one dream attempt did.
///
/// The fields are deliberately coarse: they answer "how much work happened" without
/// coupling the queue to the details of any particular consolidation pass.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct DreamMetrics {
    /// Memories looked at.
    pub records_scanned: u64,
    /// Memories folded into existing records.
    pub records_merged: u64,
    /// Memories removed.
    pub records_deleted: u64,
    /// Memories written as new records.
    pub records_created: u64,
    /// Approximate prompt tokens reclaimed by the pass.
    pub tokens_saved: u64,
    /// Wall-clock duration of the attempt.
    pub duration_ms: u64,
}

impl DreamMetrics {
    /// Returns `true` when every counter is zero.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.records_scanned == 0
            && self.records_merged == 0
            && self.records_deleted == 0
            && self.records_created == 0
            && self.tokens_saved == 0
            && self.duration_ms == 0
    }

    /// Adds `other` into `self`, saturating instead of wrapping on overflow.
    pub fn accumulate(&mut self, other: &Self) {
        self.records_scanned = self.records_scanned.saturating_add(other.records_scanned);
        self.records_merged = self.records_merged.saturating_add(other.records_merged);
        self.records_deleted = self.records_deleted.saturating_add(other.records_deleted);
        self.records_created = self.records_created.saturating_add(other.records_created);
        self.tokens_saved = self.tokens_saved.saturating_add(other.tokens_saved);
        self.duration_ms = self.duration_ms.saturating_add(other.duration_ms);
    }

    /// Returns the metrics with [`DreamMetrics::duration_ms`] set to `ms`.
    #[must_use]
    pub const fn with_duration_ms(mut self, ms: u64) -> Self {
        self.duration_ms = ms;
        self
    }
}

/// One unit of deferred memory consolidation.
///
/// A job is identified by [`DreamJob::id`], owned by a [`MemoryScope`], and carries the
/// state machine fields a durable queue needs (`state`, `attempts`, `started_at`,
/// `finished_at`, `last_error`). `payload` and `metadata` are free-form JSON objects so
/// callers and future pipelines can attach parameters without changing the schema.
///
/// JSON is `snake_case`, every field is optional on input (missing fields take their
/// [`DreamJob::default`] value), and unknown fields are ignored.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct DreamJob {
    /// Stable job id (a UUID v4 string by default).
    pub id: String,
    /// Scope whose memories this job consolidates.
    pub scope: MemoryScope,
    /// Which consolidation pass to run.
    pub kind: DreamKind,
    /// Current lifecycle state.
    pub state: DreamState,
    /// What enqueued the job.
    pub trigger: DreamTrigger,
    /// Queue priority; higher runs first.
    pub priority: i32,
    /// Creation time.
    pub created_at: DateTime<Utc>,
    /// Last mutation time.
    pub updated_at: DateTime<Utc>,
    /// When the current (or last) attempt started.
    pub started_at: Option<DateTime<Utc>>,
    /// When the job reached a terminal state.
    pub finished_at: Option<DateTime<Utc>>,
    /// Attempts started so far; incremented by every claim.
    pub attempts: u32,
    /// Attempt budget; [`DEFAULT_MAX_ATTEMPTS`] by default.
    pub max_attempts: u32,
    /// Message of the most recent failed attempt, cleared on a successful completion.
    pub last_error: Option<String>,
    /// Counters reported by the last completion.
    pub metrics: DreamMetrics,
    /// Free-form job parameters.
    pub payload: serde_json::Map<String, serde_json::Value>,
    /// Free-form caller annotations.
    pub metadata: serde_json::Map<String, serde_json::Value>,
}

impl Default for DreamJob {
    /// A pending manual consolidation job with a fresh UUID and the current time.
    fn default() -> Self {
        let now = Utc::now();
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            scope: MemoryScope::default(),
            kind: DreamKind::default(),
            state: DreamState::default(),
            trigger: DreamTrigger::default(),
            priority: 0,
            created_at: now,
            updated_at: now,
            started_at: None,
            finished_at: None,
            attempts: 0,
            max_attempts: DEFAULT_MAX_ATTEMPTS,
            last_error: None,
            metrics: DreamMetrics::default(),
            payload: serde_json::Map::new(),
            metadata: serde_json::Map::new(),
        }
    }
}

impl DreamJob {
    /// Builds a pending job for `scope`, stamped with `now` as creation time.
    #[must_use]
    pub fn new(
        scope: MemoryScope,
        kind: DreamKind,
        trigger: DreamTrigger,
        now: DateTime<Utc>,
    ) -> Self {
        Self {
            scope,
            kind,
            trigger,
            created_at: now,
            updated_at: now,
            ..Self::default()
        }
    }

    /// Returns the job with `id` as its identifier.
    #[must_use]
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = id.into();
        self
    }

    /// Returns the job with the given queue priority.
    #[must_use]
    pub const fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }

    /// Returns the job with `max_attempts` as its attempt budget.
    #[must_use]
    pub const fn with_max_attempts(mut self, max_attempts: u32) -> Self {
        self.max_attempts = max_attempts;
        self
    }

    /// Returns the job created at `created_at`.
    ///
    /// `updated_at` is set to the same instant so the job stays within its
    /// "updated never precedes created" invariant.
    #[must_use]
    pub const fn with_created_at(mut self, created_at: DateTime<Utc>) -> Self {
        self.created_at = created_at;
        self.updated_at = created_at;
        self
    }

    /// Returns the job carrying `payload`.
    #[must_use]
    pub fn with_payload(mut self, payload: serde_json::Map<String, serde_json::Value>) -> Self {
        self.payload = payload;
        self
    }

    /// Returns the job carrying `metadata`.
    #[must_use]
    pub fn with_metadata(mut self, metadata: serde_json::Map<String, serde_json::Value>) -> Self {
        self.metadata = metadata;
        self
    }

    /// Returns the job in `state`.
    ///
    /// This only sets the state; callers importing or constructing a job set the
    /// matching timestamps themselves and then call [`DreamJob::validate`].
    #[must_use]
    pub const fn with_state(mut self, state: DreamState) -> Self {
        self.state = state;
        self
    }

    /// Returns `true` when the job's state is terminal.
    #[must_use]
    pub const fn is_terminal(&self) -> bool {
        self.state.is_terminal()
    }

    /// Checks the job invariants a repository relies on.
    ///
    /// The ordering contract is:
    /// - `id` is non-empty and NUL-free, `scope` is valid, and a
    ///   [`DreamKind::Custom`] name is non-empty and NUL-free;
    /// - `max_attempts >= 1` and `attempts <= max_attempts`;
    /// - `created_at <= updated_at`;
    /// - `created_at <= started_at` when present;
    /// - `finished_at` is at least `started_at` (or `created_at` when there is no
    ///   `started_at`);
    /// - a `Running` job has `started_at`, and a terminal job has `finished_at`;
    /// - `last_error` is NUL-free when present.
    ///
    /// A `Pending` job may retain `started_at` from a failed attempt: that timestamp is
    /// the retry marker the repository compares against the caller-supplied clock, so
    /// it is explicitly allowed here. Likewise `last_error` may be set on a requeued
    /// `Pending` job (it describes the failed attempt); it is only required to be
    /// NUL-free and is cleared on a successful completion.
    pub fn validate(&self) -> Result<()> {
        validate_non_empty("dream job `id`", &self.id)?;
        self.scope.validate()?;
        if let DreamKind::Custom(kind) = &self.kind {
            validate_non_empty("dream job custom kind", kind)?;
        }
        if self.max_attempts == 0 {
            return Err(Error::validation(format!(
                "dream job `max_attempts` must be at least 1 (got {})",
                self.max_attempts
            )));
        }
        if self.attempts > self.max_attempts {
            return Err(Error::validation(format!(
                "dream job `attempts` ({}) must not exceed `max_attempts` ({})",
                self.attempts, self.max_attempts
            )));
        }
        if self.updated_at < self.created_at {
            return Err(Error::validation(format!(
                "dream job `updated_at` ({}) must not precede `created_at` ({})",
                self.updated_at, self.created_at
            )));
        }
        if let Some(started_at) = self.started_at {
            if started_at < self.created_at {
                return Err(Error::validation(format!(
                    "dream job `started_at` ({started_at}) must not precede `created_at` ({})",
                    self.created_at
                )));
            }
        }
        if let Some(finished_at) = self.finished_at {
            let anchor = self.started_at.unwrap_or(self.created_at);
            if finished_at < anchor {
                return Err(Error::validation(format!(
                    "dream job `finished_at` ({finished_at}) must not precede `started_at`/`created_at` ({anchor})"
                )));
            }
        }
        if self.state == DreamState::Running && self.started_at.is_none() {
            return Err(Error::validation(
                "dream job in state `running` requires `started_at`",
            ));
        }
        if self.state.is_terminal() && self.finished_at.is_none() {
            return Err(Error::validation(format!(
                "dream job in state `{}` requires `finished_at`",
                self.state
            )));
        }
        if let Some(last_error) = &self.last_error {
            if last_error.contains('\0') {
                return Err(Error::validation(
                    "dream job `last_error` must not contain NUL",
                ));
            }
        }
        Ok(())
    }
}

/// Rejects an empty or NUL-containing identifier, naming `label` and the value.
pub(super) fn validate_non_empty(label: &str, value: &str) -> Result<()> {
    if value.is_empty() {
        return Err(Error::validation(format!("{label} must not be empty")));
    }
    if value.contains('\0') {
        return Err(Error::validation(format!("{label} must not contain NUL")));
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use serde_json::json;

    fn fixed_time() -> DateTime<Utc> {
        DateTime::from_timestamp(1_700_000_000, 0).unwrap()
    }

    fn scope() -> MemoryScope {
        MemoryScope::new("alice", "ws")
    }

    fn job() -> DreamJob {
        DreamJob::new(
            scope(),
            DreamKind::Consolidation,
            DreamTrigger::Manual,
            fixed_time(),
        )
    }

    fn running_job() -> DreamJob {
        DreamJob {
            state: DreamState::Running,
            started_at: Some(fixed_time()),
            attempts: 1,
            ..job()
        }
    }

    fn succeeded_job() -> DreamJob {
        DreamJob {
            state: DreamState::Succeeded,
            started_at: Some(fixed_time()),
            finished_at: Some(fixed_time()),
            attempts: 1,
            ..job()
        }
    }

    fn assert_validation(error: Error, needle: &str) {
        assert!(matches!(error, Error::Validation(_)), "got {error:?}");
        assert!(
            error.to_string().contains(needle),
            "`{error}` should mention `{needle}`"
        );
    }

    #[test]
    fn state_names_parse_display_and_terminality() {
        let names = [
            "pending",
            "running",
            "succeeded",
            "failed",
            "skipped",
            "cancelled",
        ];
        assert_eq!(DreamState::default(), DreamState::Pending);
        for (state, name) in DreamState::ALL.into_iter().zip(names) {
            assert_eq!(state.as_str(), name);
            assert_eq!(state.to_string(), name);
            assert_eq!(
                DreamState::parse(&format!(" {} ", name.to_uppercase())),
                Some(state)
            );
            assert_eq!(
                state.is_terminal(),
                !matches!(state, DreamState::Pending | DreamState::Running)
            );
        }
        assert_eq!(DreamState::parse("nope"), None);
    }

    #[test]
    fn state_serde_round_trip_and_strict_unknown() {
        for state in DreamState::ALL {
            let encoded = serde_json::to_string(&state).unwrap();
            assert_eq!(encoded, format!("\"{}\"", state.as_str()));
            assert_eq!(serde_json::from_str::<DreamState>(&encoded).unwrap(), state);
        }
        // Tolerant parsing of persisted rows belongs to the SQLite backend.
        assert!(serde_json::from_str::<DreamState>("\"weird\"").is_err());
    }

    #[test]
    fn kind_names_aliases_and_tolerance() {
        let cases = [
            ("consolidation", DreamKind::Consolidation),
            ("consolidate", DreamKind::Consolidation),
            ("dedupe", DreamKind::Dedupe),
            ("deduplicate", DreamKind::Dedupe),
            ("dedup", DreamKind::Dedupe),
            ("summarize", DreamKind::Summarize),
            ("summarise", DreamKind::Summarize),
            ("forget", DreamKind::Forget),
            ("prune", DreamKind::Forget),
            ("pruning", DreamKind::Forget),
            ("graph_extraction", DreamKind::GraphExtraction),
            ("graph", DreamKind::GraphExtraction),
        ];
        for (raw, expected) in cases {
            assert_eq!(DreamKind::parse(raw), expected, "raw = {raw}");
            assert_eq!(DreamKind::from(raw), expected);
            assert_eq!(DreamKind::from(raw.to_owned()), expected);
        }
        assert_eq!(DreamKind::default(), DreamKind::Consolidation);
        assert_eq!(
            DreamKind::parse(" Sleeve "),
            DreamKind::Custom("Sleeve".to_owned())
        );
        assert_eq!(DreamKind::Custom("custom".to_owned()).as_str(), "custom");
        assert_eq!(DreamKind::Dedupe.as_str(), "dedupe");
        assert_eq!(DreamKind::Dedupe.to_string(), "dedupe");
    }

    #[test]
    fn kind_serde_is_tolerant_both_ways() {
        assert_eq!(
            serde_json::to_value(DreamKind::GraphExtraction).unwrap(),
            json!("graph_extraction")
        );
        assert_eq!(
            serde_json::to_value(DreamKind::Custom("mine".to_owned())).unwrap(),
            json!("mine")
        );
        assert_eq!(
            serde_json::from_value::<DreamKind>(json!("dedup")).unwrap(),
            DreamKind::Dedupe
        );
        assert_eq!(
            serde_json::from_value::<DreamKind>(json!("never-heard-of-it")).unwrap(),
            DreamKind::Custom("never-heard-of-it".to_owned())
        );
        let custom = DreamKind::Custom("mine".to_owned());
        let restored: DreamKind =
            serde_json::from_value(serde_json::to_value(&custom).unwrap()).unwrap();
        assert_eq!(restored, custom);
    }

    #[test]
    fn trigger_names_aliases_and_serde() {
        for trigger in [
            DreamTrigger::Manual,
            DreamTrigger::Schedule,
            DreamTrigger::MessageThreshold,
            DreamTrigger::Idle,
        ] {
            assert_eq!(DreamTrigger::parse(trigger.as_str()), Some(trigger));
            let encoded = serde_json::to_string(&trigger).unwrap();
            assert_eq!(encoded, format!("\"{}\"", trigger.as_str()));
            assert_eq!(
                serde_json::from_str::<DreamTrigger>(&encoded).unwrap(),
                trigger
            );
        }
        assert_eq!(
            DreamTrigger::parse(" THRESHOLD "),
            Some(DreamTrigger::MessageThreshold)
        );
        assert_eq!(DreamTrigger::parse("weekly"), None);
        assert_eq!(DreamTrigger::default(), DreamTrigger::Manual);
        assert_eq!(DreamTrigger::Idle.to_string(), "idle");
    }

    #[test]
    fn metrics_accumulate_saturate_and_serde() {
        let mut metrics = DreamMetrics::default();
        assert!(metrics.is_empty());
        metrics.accumulate(&DreamMetrics {
            records_scanned: 2,
            records_merged: 3,
            records_deleted: 1,
            records_created: 4,
            tokens_saved: 5,
            duration_ms: 6,
        });
        assert!(!metrics.is_empty());
        assert_eq!(
            metrics,
            DreamMetrics {
                records_scanned: 2,
                records_merged: 3,
                records_deleted: 1,
                records_created: 4,
                tokens_saved: 5,
                duration_ms: 6,
            }
        );
        let mut saturated = DreamMetrics {
            records_scanned: u64::MAX,
            ..DreamMetrics::default()
        };
        saturated.accumulate(&DreamMetrics {
            records_scanned: 1,
            ..DreamMetrics::default()
        });
        assert_eq!(saturated.records_scanned, u64::MAX);
        assert_eq!(metrics.with_duration_ms(9).duration_ms, 9);
        assert_eq!(
            serde_json::from_value::<DreamMetrics>(json!({})).unwrap(),
            DreamMetrics::default()
        );
    }

    #[test]
    fn job_defaults_new_and_builders() {
        let default_job = DreamJob::default();
        assert_ne!(default_job.id, "");
        assert_eq!(default_job.state, DreamState::Pending);
        assert_eq!(default_job.kind, DreamKind::Consolidation);
        assert_eq!(default_job.trigger, DreamTrigger::Manual);
        assert_eq!(default_job.max_attempts, DEFAULT_MAX_ATTEMPTS);
        assert_eq!(default_job.attempts, 0);
        assert_eq!(default_job.created_at, default_job.updated_at);
        assert!(default_job.payload.is_empty() && default_job.metadata.is_empty());
        assert!(default_job.is_terminal().eq(&false));
        assert!(default_job.validate().is_ok());

        let fresh = job();
        assert_eq!(
            (fresh.created_at, fresh.updated_at),
            (fixed_time(), fixed_time())
        );
        assert_eq!(
            (fresh.scope, fresh.kind),
            (scope(), DreamKind::Consolidation)
        );

        let payload = serde_json::Map::from_iter([("key".to_owned(), json!(1))]);
        let metadata = serde_json::Map::from_iter([("tag".to_owned(), json!("x"))]);
        let built = job()
            .with_id("job-1")
            .with_priority(7)
            .with_max_attempts(5)
            .with_created_at(fixed_time() + chrono::Duration::seconds(5))
            .with_payload(payload.clone())
            .with_metadata(metadata.clone())
            .with_state(DreamState::Running);
        assert_eq!(built.id, "job-1");
        assert_eq!((built.priority, built.max_attempts), (7, 5));
        assert_eq!(
            built.created_at,
            fixed_time() + chrono::Duration::seconds(5)
        );
        assert_eq!(built.updated_at, built.created_at);
        assert_eq!(built.payload, payload);
        assert_eq!(built.metadata, metadata);
        assert_eq!(built.state, DreamState::Running);
    }

    #[test]
    fn job_json_tolerates_missing_unknown_and_round_trips_objects() {
        let restored: DreamJob = serde_json::from_value(json!({})).unwrap();
        assert_eq!(restored.state, DreamState::Pending);
        assert_eq!(restored.max_attempts, DEFAULT_MAX_ATTEMPTS);
        assert!(restored.payload.is_empty());
        assert!(restored.validate().is_ok());

        let restored: DreamJob = serde_json::from_value(json!({
            "id": "job-9",
            "state": "pending",
            "kind": "prune",
            "trigger": "message_threshold",
            "priority": -2,
            "payload": {"a": [1, 2]},
            "metadata": {"b": true},
            "something_new": "ignored"
        }))
        .unwrap();
        assert_eq!(restored.id, "job-9");
        assert_eq!(restored.kind, DreamKind::Forget);
        assert_eq!(restored.trigger, DreamTrigger::MessageThreshold);
        assert_eq!(restored.priority, -2);
        assert_eq!(restored.payload.get("a"), Some(&json!([1, 2])));
        assert_eq!(restored.metadata.get("b"), Some(&json!(true)));
        assert!(restored.validate().is_ok());

        let job = running_job().with_metadata(serde_json::Map::from_iter([(
            "note".to_owned(),
            json!("hello"),
        )]));
        let value = serde_json::to_value(&job).unwrap();
        assert_eq!(value["state"], json!("running"));
        assert_eq!(value["kind"], json!("consolidation"));
        assert_eq!(value["trigger"], json!("manual"));
        assert_eq!(value["payload"], json!({}));
        assert_eq!(value["metadata"], json!({"note": "hello"}));
        assert_eq!(value["scope"]["user_id"], json!("alice"));
        assert_eq!(serde_json::from_value::<DreamJob>(value).unwrap(), job);
    }

    #[test]
    fn validate_rejects_bad_ids_scope_and_custom_kind() {
        assert_validation(job().with_id("").validate().unwrap_err(), "`id`");
        assert_validation(job().with_id("a\0b").validate().unwrap_err(), "NUL");
        let bad_scope = DreamJob::new(
            MemoryScope::new("", "ws"),
            DreamKind::Consolidation,
            DreamTrigger::Manual,
            fixed_time(),
        );
        assert_validation(bad_scope.validate().unwrap_err(), "user_id");
        let empty_kind = DreamJob::new(
            scope(),
            DreamKind::Custom(String::new()),
            DreamTrigger::Manual,
            fixed_time(),
        );
        assert_validation(empty_kind.validate().unwrap_err(), "custom kind");
        let nul_kind = DreamJob::new(
            scope(),
            DreamKind::Custom("bad\0kind".to_owned()),
            DreamTrigger::Manual,
            fixed_time(),
        );
        assert_validation(nul_kind.validate().unwrap_err(), "NUL");
    }

    #[test]
    fn validate_rejects_attempt_budget_violations() {
        assert_validation(
            job().with_max_attempts(0).validate().unwrap_err(),
            "max_attempts",
        );
        let exhausted = DreamJob {
            attempts: 4,
            ..job().with_max_attempts(3)
        };
        assert_validation(exhausted.validate().unwrap_err(), "must not exceed");
    }

    #[test]
    fn validate_rejects_bad_timestamp_ordering() {
        let later = fixed_time() + chrono::Duration::seconds(10);
        let earlier = fixed_time() - chrono::Duration::seconds(10);
        let mut bad = job();
        bad.updated_at = earlier;
        assert_validation(bad.validate().unwrap_err(), "`updated_at`");
        let mut bad = job();
        bad.started_at = Some(earlier);
        assert_validation(bad.validate().unwrap_err(), "`started_at`");
        let mut bad = job();
        bad.started_at = Some(later);
        bad.finished_at = Some(fixed_time());
        assert_validation(bad.validate().unwrap_err(), "`finished_at`");
        assert!(job().with_created_at(later).validate().is_ok());
    }

    #[test]
    fn validate_state_requires_matching_timestamps() {
        let mut bad = job();
        bad.state = DreamState::Running;
        assert_validation(bad.validate().unwrap_err(), "requires `started_at`");
        assert!(running_job().validate().is_ok());
        for state in [
            DreamState::Succeeded,
            DreamState::Failed,
            DreamState::Skipped,
            DreamState::Cancelled,
        ] {
            let mut bad = job();
            bad.state = state;
            assert_validation(bad.validate().unwrap_err(), "requires `finished_at`");
        }
        assert!(succeeded_job().validate().is_ok());
    }

    #[test]
    fn pending_job_may_keep_started_at_as_retry_marker() {
        let retry_marker = DreamJob {
            started_at: Some(fixed_time()),
            ..job()
        };
        assert_eq!(retry_marker.state, DreamState::Pending);
        assert!(retry_marker.validate().is_ok());
        let mut bad = running_job();
        bad.last_error = Some("boom\0".to_owned());
        assert_validation(bad.validate().unwrap_err(), "NUL");
    }
}
