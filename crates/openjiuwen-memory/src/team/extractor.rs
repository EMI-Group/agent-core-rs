//! Team memory extraction: a [`TeamMemoryExtractor`] seam with a rule-based default and
//! a model-backed implementation.
//!
//! Extraction is deliberately separate from the team store: an extractor reads a
//! [`TeamTranscript`] and returns [`NewTeamMemory`] create requests, and it never
//! writes anything. The caller decides which records to keep and hands them to
//! [`crate::team::TeamMemory::remember_many`], so the store stays the single source of
//! truth and a failed extraction can never leave a half-written memory behind.
//!
//! # The two implementations
//!
//! [`RuleTeamExtractor`] is the deterministic, offline default. It scans each turn for
//! a fixed keyword inventory and returns at most one record per turn plus one `Task`
//! record per transcript task note, so extraction works with no model, no network, and
//! no clock in the loop.
//!
//! [`LlmTeamExtractor`] asks a [`Model`] for a JSON answer shaped by
//! [`crate::team::prompts::EXTRACTION_SYSTEM_PROMPT`] and parses it with
//! [`parse_team_extraction`]. A provider error, an empty answer, or an unparseable
//! answer consumes one of its `model_retries + 1` attempts; once the budget is spent it
//! falls back to the rule extractor and reports why on [`TeamExtraction::degraded`].
//! The fallback never fails silently: the reason string always names the original
//! failure, and cancellation (`ErrorKind::Cancelled`) is propagated instead of being
//! degraded, so a cancelled call returns no records at all.
//!
//! # Bounds
//!
//! Both implementations bound their output by [`TeamExtractConfig`]: turns shorter than
//! `min_content_chars` are skipped, content is truncated to `max_turn_chars`, the
//! rendered transcript is truncated to `max_transcript_chars`, duplicates (normalized
//! text: lowercased, whitespace collapsed) are dropped, and at most `max_records`
//! records are returned. Every returned record is validated, and a validation failure
//! is surfaced as [`openjiuwen_common::error::Error::Validation`] rather than a silently
//! dropped record.
//!
//! # Difference from the Python reference
//!
//! Python's `extract_team_memories` runs a deep agent whose only write path is file
//! tools over `TEAM_MEMORY.md`: it hands the model a Markdown prompt, lets the model
//! rewrite the file, and parses nothing, dedupes nothing, and applies no thresholds.
//! The Rust port returns typed [`NewTeamMemory`] records instead: the model answer is
//! parsed, sanitized, deduplicated, capped, and validated here, and the caller — not
//! the model — decides what is stored. Consequently the prompt asks for JSON rather
//! than Markdown, and the four Python section tags
//! ([`crate::team::prompts::TEAM_MEMORY_TAGS`]) map onto [`TeamMemoryKind`]s via
//! [`TeamMemoryKind::extractor_tag`].

use crate::record::CallOptions;
use crate::team::model::{
    normalize_tags, NewTeamMemory, TeamMemoryKind, TeamVisibility, MAX_CONTENT_CHARS,
    MAX_SCOPE_ID_CHARS, MAX_TAGS, MAX_TAG_CHARS, MAX_TITLE_CHARS,
};
use crate::team::prompts::{render_transcript, EXTRACTION_SYSTEM_PROMPT};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use openjiuwen_common::error::{Error, ErrorKind, Result};
use openjiuwen_foundation::llm::{Model, ModelRequestConfig};
use openjiuwen_foundation::prompt::BaseMessage;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::BTreeSet;
use std::sync::Arc;

/// Provenance marker written on records the rule extractor produced.
pub const RULE_SOURCE: &str = "rule-extractor";
/// Provenance marker written on records the model extractor produced.
pub const LLM_SOURCE: &str = "llm-extractor";
/// Importance of an extracted `decision` record.
pub const DECISION_IMPORTANCE: f32 = 0.7;
/// Importance of an extracted `lesson`/handoff record.
pub const LESSON_IMPORTANCE: f32 = 0.6;
/// Importance of an extracted `member` record.
pub const MEMBER_IMPORTANCE: f32 = 0.6;
/// Importance of an extracted `task` record.
pub const TASK_IMPORTANCE: f32 = 0.5;
/// Importance of an extracted `context` record.
pub const CONTEXT_IMPORTANCE: f32 = 0.4;
/// Longest generated title, in characters; longer content gets a title.
pub const TITLE_MAX_CHARS: usize = 80;
/// Longest accepted task note in a transcript, in characters.
pub const MAX_TASK_NOTE_CHARS: usize = 16_384;
/// Most task notes accepted in one transcript.
pub const MAX_TASK_NOTES: usize = 256;
/// Longest accepted raw turn content, in characters.
pub const MAX_TURN_CONTENT_CHARS: usize = 65_536;

/// Default [`TeamExtractConfig::max_records`].
pub const DEFAULT_MAX_RECORDS: usize = 8;
/// Default [`TeamExtractConfig::max_turn_chars`].
pub const DEFAULT_MAX_TURN_CHARS: usize = 1000;
/// Default [`TeamExtractConfig::max_transcript_chars`].
pub const DEFAULT_MAX_TRANSCRIPT_CHARS: usize = 8000;
/// Default [`TeamExtractConfig::min_content_chars`].
pub const DEFAULT_MIN_CONTENT_CHARS: usize = 12;
/// Default [`TeamExtractConfig::model_retries`].
pub const DEFAULT_MODEL_RETRIES: u32 = 1;

/// One turn of team collaboration: a message from one member to another or the team.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct TeamTurn {
    /// The member who sent the message.
    pub from_member: String,
    /// The member addressed; `None` means the whole team.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub to_member: Option<String>,
    /// The message text.
    #[serde(default)]
    pub content: String,
    /// When the message was sent (UTC).
    #[serde(default = "epoch")]
    pub at: DateTime<Utc>,
}

/// The UTC epoch, the serde default of [`TeamTurn::at`].
fn epoch() -> DateTime<Utc> {
    DateTime::<Utc>::UNIX_EPOCH
}

impl TeamTurn {
    /// Builds a team-wide turn sent by `from_member` at `at`.
    #[must_use]
    pub fn new(
        from_member: impl Into<String>,
        content: impl Into<String>,
        at: DateTime<Utc>,
    ) -> Self {
        Self {
            from_member: from_member.into(),
            to_member: None,
            content: content.into(),
            at,
        }
    }

    /// Returns the turn addressed to `to_member` instead of the whole team.
    #[must_use]
    pub fn with_to(mut self, to_member: impl Into<String>) -> Self {
        self.to_member = Some(to_member.into());
        self
    }

    /// Validates the sender, the recipient, and the content.
    ///
    /// Members are storage-key components (`from_member` and `to_member` become a
    /// record's `member_id`), so the rules mirror
    /// [`crate::team::model`]: non-blank, no NUL, no `/` or `\`, no control
    /// characters, at most [`MAX_SCOPE_ID_CHARS`] characters. Content must contain a
    /// non-whitespace character and stay within [`MAX_TURN_CONTENT_CHARS`].
    pub fn validate(&self) -> Result<()> {
        validate_component("from_member", &self.from_member)?;
        if let Some(to_member) = &self.to_member {
            validate_component("to_member", to_member)?;
        }
        if self.content.trim().is_empty() {
            return Err(Error::validation(
                "team turn `content` must not be empty".to_owned(),
            ));
        }
        if self.content.contains('\0') {
            return Err(Error::validation(
                "team turn `content` must not contain NUL".to_owned(),
            ));
        }
        let length = self.content.chars().count();
        if length > MAX_TURN_CONTENT_CHARS {
            return Err(Error::validation(format!(
                "team turn `content` must be at most {MAX_TURN_CONTENT_CHARS} characters (got {length})"
            )));
        }
        Ok(())
    }
}

/// Everything one extraction pass reads: the turns of a session plus task notes.
///
/// A transcript is the extractor's entire input; it carries no store handle and no
/// clock, so extraction is a pure function of it plus the configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct TeamTranscript {
    /// The team the collaboration belongs to.
    pub team_id: String,
    /// The coordination session, copied onto every extracted record.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    /// The conversation turns, in any order (renderers sort by timestamp).
    #[serde(default)]
    pub turns: Vec<TeamTurn>,
    /// Task-board notes; the rule extractor emits one `Task` record per note.
    #[serde(default)]
    pub task_notes: Vec<String>,
}

impl TeamTranscript {
    /// Builds an empty transcript for `team_id`.
    #[must_use]
    pub fn new(team_id: impl Into<String>) -> Self {
        Self {
            team_id: team_id.into(),
            session_id: None,
            turns: Vec::new(),
            task_notes: Vec::new(),
        }
    }

    /// Sets the coordination session copied onto extracted records.
    #[must_use]
    pub fn with_session(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }

    /// Appends one turn.
    #[must_use]
    pub fn with_turn(mut self, turn: TeamTurn) -> Self {
        self.turns.push(turn);
        self
    }

    /// Appends one turn in place.
    pub fn push_turn(&mut self, turn: TeamTurn) {
        self.turns.push(turn);
    }

    /// Appends one task note.
    #[must_use]
    pub fn with_task(mut self, note: impl Into<String>) -> Self {
        self.task_notes.push(note.into());
        self
    }

    /// Returns `true` when there is nothing to extract from.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.turns.is_empty() && self.task_notes.is_empty()
    }

    /// Validates the team id, the session id, every turn, and the task notes.
    pub fn validate(&self) -> Result<()> {
        validate_component("team_id", &self.team_id)?;
        if let Some(session_id) = &self.session_id {
            validate_component("session_id", session_id)?;
        }
        for turn in &self.turns {
            turn.validate()?;
        }
        if self.task_notes.len() > MAX_TASK_NOTES {
            return Err(Error::validation(format!(
                "team transcript `task_notes` must contain at most {MAX_TASK_NOTES} entries (got {})",
                self.task_notes.len()
            )));
        }
        for note in &self.task_notes {
            let length = note.chars().count();
            if length > MAX_TASK_NOTE_CHARS {
                return Err(Error::validation(format!(
                    "team transcript task notes must be at most {MAX_TASK_NOTE_CHARS} characters (got {length})"
                )));
            }
        }
        Ok(())
    }
}

/// The result of one extraction pass.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct TeamExtraction {
    /// The create requests, already sanitized, deduplicated, and validated.
    pub records: Vec<NewTeamMemory>,
    /// Why a degraded path produced these records, when one did.
    pub degraded: Option<String>,
}

impl TeamExtraction {
    /// Builds an extraction result from `records`.
    #[must_use]
    pub fn new(records: Vec<NewTeamMemory>, degraded: Option<String>) -> Self {
        Self { records, degraded }
    }

    /// Returns an empty, non-degraded result.
    #[must_use]
    pub fn empty() -> Self {
        Self::default()
    }

    /// Returns `true` when no record was extracted.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Returns the number of extracted records.
    #[must_use]
    pub fn len(&self) -> usize {
        self.records.len()
    }
}

/// The extraction seam: turn a [`TeamTranscript`] into [`NewTeamMemory`] requests.
///
/// Implementations never write and never mutate the transcript; they validate their
/// input, honor [`CallOptions`] cancellation, and bound their output by
/// [`TeamExtractConfig`].
#[async_trait]
pub trait TeamMemoryExtractor: Send + Sync {
    /// The extractor name (`"rule"` or `"llm"`), used in logs and telemetry.
    fn name(&self) -> &str;

    /// Extracts team memories from `transcript`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for an invalid configuration or transcript,
    /// [`Error::Cancelled`] when `options` was cancelled, and the provider's own error
    /// when a call fails in a way the extractor cannot degrade.
    async fn extract(
        &self,
        transcript: &TeamTranscript,
        options: &CallOptions,
    ) -> Result<TeamExtraction>;
}

/// Knobs shared by both extractors.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct TeamExtractConfig {
    /// When `false`, extraction returns an empty result without reading anything.
    pub enabled: bool,
    /// Maximum number of records one pass may return.
    pub max_records: usize,
    /// Maximum characters kept from one turn (or one task note) of content.
    pub max_turn_chars: usize,
    /// Maximum characters of rendered transcript sent to the model.
    pub max_transcript_chars: usize,
    /// Turns shorter than this many characters are skipped by the rule extractor.
    pub min_content_chars: usize,
    /// Extra model attempts after the first one before degrading to rules.
    pub model_retries: u32,
}

impl Default for TeamExtractConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_records: DEFAULT_MAX_RECORDS,
            max_turn_chars: DEFAULT_MAX_TURN_CHARS,
            max_transcript_chars: DEFAULT_MAX_TRANSCRIPT_CHARS,
            min_content_chars: DEFAULT_MIN_CONTENT_CHARS,
            model_retries: DEFAULT_MODEL_RETRIES,
        }
    }
}

impl TeamExtractConfig {
    /// Validates the limits: the three sizes are non-zero and the minimum content
    /// length fits inside a turn.
    pub fn validate(&self) -> Result<()> {
        if self.max_records == 0 {
            return Err(Error::validation(
                "team extract `max_records` must be greater than zero".to_owned(),
            ));
        }
        if self.max_turn_chars == 0 {
            return Err(Error::validation(
                "team extract `max_turn_chars` must be greater than zero".to_owned(),
            ));
        }
        if self.max_transcript_chars == 0 {
            return Err(Error::validation(
                "team extract `max_transcript_chars` must be greater than zero".to_owned(),
            ));
        }
        if self.min_content_chars > self.max_turn_chars {
            return Err(Error::validation(format!(
                "team extract `min_content_chars` ({}) must not exceed `max_turn_chars` ({})",
                self.min_content_chars, self.max_turn_chars
            )));
        }
        Ok(())
    }

    /// The number of characters a record's content may keep.
    ///
    /// `max_turn_chars`, additionally clamped to `MAX_CONTENT_CHARS` so a large
    /// per-turn budget can never produce a content that the record model rejects.
    fn content_budget(&self) -> usize {
        self.max_turn_chars.min(MAX_CONTENT_CHARS)
    }
}

/// One entry of the rule keyword inventory.
struct MarkerRule {
    /// The tag the matched marker category earns.
    category: &'static str,
    /// The kind a match produces.
    kind: TeamMemoryKind,
    /// The record importance a match earns.
    importance: f32,
    /// The lowercase markers; the first rule with any marker wins.
    markers: &'static [&'static str],
}

/// The rule inventory, in priority order; the first match wins.
static RULES: [MarkerRule; 5] = [
    MarkerRule {
        category: "decision",
        kind: TeamMemoryKind::Decision,
        importance: DECISION_IMPORTANCE,
        markers: &[
            "we decided",
            "decided to",
            "we chose",
            "chose to",
            "instead of",
            "决定",
            "选择",
        ],
    },
    MarkerRule {
        category: "lesson",
        kind: TeamMemoryKind::HandoffNote,
        importance: LESSON_IMPORTANCE,
        markers: &[
            "learned",
            "lesson",
            "should have",
            "next time",
            "root cause",
            "教训",
            "经验",
        ],
    },
    MarkerRule {
        category: "member",
        kind: TeamMemoryKind::MemberProfile,
        importance: MEMBER_IMPORTANCE,
        markers: &["is good at", "owns", "responsible for", "擅长", "负责"],
    },
    MarkerRule {
        category: "task",
        kind: TeamMemoryKind::Task,
        importance: TASK_IMPORTANCE,
        markers: &["action item", "todo", "assigned to", "待办"],
    },
    MarkerRule {
        category: "context",
        kind: TeamMemoryKind::SharedFact,
        importance: CONTEXT_IMPORTANCE,
        markers: &["constraint", "deadline", "requirement", "约束", "背景"],
    },
];

/// The deterministic, model-free team memory extractor.
///
/// The inventory is fixed and tried in priority order (`decision`, `lesson`,
/// `member`, `task`, `context`), so one turn yields at most one record and the same
/// transcript always yields the same records. A turn without a marker yields nothing:
/// the rule extractor never invents a memory from unmarked chatter.
#[derive(Debug, Clone)]
pub struct RuleTeamExtractor {
    /// The limits and switch in use.
    config: TeamExtractConfig,
}

impl RuleTeamExtractor {
    /// Builds a rule extractor from `config`.
    #[must_use]
    pub fn new(config: TeamExtractConfig) -> Self {
        Self { config }
    }

    /// Returns the configuration in use.
    #[must_use]
    pub fn config(&self) -> &TeamExtractConfig {
        &self.config
    }

    /// Builds the deterministic records for `transcript`.
    fn extract_records(&self, transcript: &TeamTranscript) -> Result<Vec<NewTeamMemory>> {
        let config = &self.config;
        let session = transcript.session_id.as_deref();
        let mut records: Vec<NewTeamMemory> = Vec::new();
        let mut seen: BTreeSet<String> = BTreeSet::new();
        for turn in &transcript.turns {
            if records.len() >= config.max_records {
                break;
            }
            let content = turn.content.trim();
            if content.chars().count() < config.min_content_chars {
                continue;
            }
            let Some(rule) = classify(content) else {
                continue;
            };
            if !seen.insert(normalize_key(content)) {
                tracing::warn!(
                    target: "openjiuwen_memory::team",
                    from_member = %turn.from_member,
                    "skipping a duplicate team turn"
                );
                continue;
            }
            let tags = rule_tags(rule);
            records.push(build_record(
                &transcript.team_id,
                session,
                Some(turn.from_member.trim()),
                &rule.kind,
                content,
                tags,
                rule.importance,
                RULE_SOURCE,
                config,
            )?);
        }
        for note in &transcript.task_notes {
            if records.len() >= config.max_records {
                break;
            }
            let content = note.trim();
            if content.is_empty() || !seen.insert(normalize_key(content)) {
                continue;
            }
            let tags = vec![
                "task".to_owned(),
                TeamMemoryKind::Task.extractor_tag().to_owned(),
            ];
            records.push(build_record(
                &transcript.team_id,
                session,
                None,
                &TeamMemoryKind::Task,
                content,
                tags,
                TASK_IMPORTANCE,
                RULE_SOURCE,
                config,
            )?);
        }
        Ok(records)
    }
}

#[async_trait]
impl TeamMemoryExtractor for RuleTeamExtractor {
    #[allow(clippy::unnecessary_literal_bound)] // the trait signature ties `&str` to `self`
    fn name(&self) -> &str {
        "rule"
    }

    async fn extract(
        &self,
        transcript: &TeamTranscript,
        options: &CallOptions,
    ) -> Result<TeamExtraction> {
        options.check()?;
        self.config.validate()?;
        transcript.validate()?;
        if !self.config.enabled {
            return Ok(TeamExtraction::empty());
        }
        Ok(TeamExtraction::new(self.extract_records(transcript)?, None))
    }
}

/// The model-backed team memory extractor.
///
/// One [`Model::invoke`] call per attempt with the system prompt plus the rendered
/// transcript; the answer is parsed by [`parse_team_extraction`]. Every failure mode
/// is retried up to `model_retries` times and then degraded to
/// [`RuleTeamExtractor`], with the original failure kept on
/// [`TeamExtraction::degraded`].
pub struct LlmTeamExtractor {
    /// The model that answers the extraction prompt.
    model: Arc<dyn Model>,
    /// The limits and switch in use.
    config: TeamExtractConfig,
    /// Deterministic extractor used once the retry budget is spent.
    fallback: RuleTeamExtractor,
}

impl LlmTeamExtractor {
    /// Builds an extractor around `model` using `config`.
    ///
    /// The fallback is a [`RuleTeamExtractor`] built from the same config;
    /// [`LlmTeamExtractor::with_fallback`] replaces it when a caller tuned the rules
    /// separately.
    #[must_use]
    pub fn new(model: Arc<dyn Model>, config: TeamExtractConfig) -> Self {
        let fallback = RuleTeamExtractor::new(config.clone());
        Self {
            model,
            config,
            fallback,
        }
    }

    /// Returns the extractor with `fallback` as its deterministic backstop.
    #[must_use]
    pub fn with_fallback(mut self, fallback: RuleTeamExtractor) -> Self {
        self.fallback = fallback;
        self
    }

    /// Returns the configuration in use.
    #[must_use]
    pub fn config(&self) -> &TeamExtractConfig {
        &self.config
    }
}

#[async_trait]
impl TeamMemoryExtractor for LlmTeamExtractor {
    #[allow(clippy::unnecessary_literal_bound)] // the trait signature ties `&str` to `self`
    fn name(&self) -> &str {
        "llm"
    }

    async fn extract(
        &self,
        transcript: &TeamTranscript,
        options: &CallOptions,
    ) -> Result<TeamExtraction> {
        options.check()?;
        self.config.validate()?;
        transcript.validate()?;
        if !self.config.enabled {
            return Ok(TeamExtraction::empty());
        }
        let messages = vec![
            BaseMessage::system(EXTRACTION_SYSTEM_PROMPT),
            BaseMessage::user(render_transcript(
                transcript,
                self.config.max_transcript_chars,
            )),
        ];
        let request = ModelRequestConfig::new(self.model.model_name().to_owned());
        let attempts = self.config.model_retries.saturating_add(1);
        // The first failure is kept: it is the root cause the retries started from.
        let mut failure: Option<String> = None;
        for attempt in 1..=attempts {
            match options.run(self.model.invoke(&messages, &request)).await {
                Ok(Ok(response)) => {
                    let raw = response
                        .message()
                        .map(|message| message.text().into_owned())
                        .unwrap_or_default();
                    if let Some(records) =
                        parse_team_extraction(&raw, &transcript.team_id, &self.config)
                    {
                        return Ok(TeamExtraction::new(records, None));
                    }
                    failure.get_or_insert_with(|| {
                        "the model answer had no usable team memories".to_owned()
                    });
                    tracing::warn!(
                        target: "openjiuwen_memory::team",
                        attempt,
                        attempts,
                        "LLM team memory extraction returned unusable output; retrying"
                    );
                }
                Ok(Err(error)) | Err(error) => {
                    if error.kind() == ErrorKind::Cancelled {
                        return Err(error);
                    }
                    failure.get_or_insert_with(|| format!("the model call failed: {error}"));
                    tracing::warn!(
                        target: "openjiuwen_memory::team",
                        attempt,
                        attempts,
                        %error,
                        "LLM team memory extraction call failed; retrying"
                    );
                }
            }
            options.check()?;
        }
        let failure = failure.unwrap_or_else(|| "no attempt was made".to_owned());
        tracing::warn!(
            target: "openjiuwen_memory::team",
            attempts,
            %failure,
            "LLM team memory extraction exhausted its retries; falling back to rules"
        );
        let fallback = self.fallback.extract(transcript, options).await?;
        Ok(TeamExtraction::new(
            fallback.records,
            Some(format!(
                "llm team memory extraction fell back to rules: {failure}"
            )),
        ))
    }
}

/// Parses a model answer into team memory create requests.
///
/// Accepted shapes: a bare array of entries, an object holding a `memories` (also
/// `records`/`items`) array, or a single entry object. Markdown fences and surrounding
/// prose are tolerated. Per entry, `kind` (string, parsed tolerantly) and `content`
/// (required, non-blank) are read, plus the optional `title`, `tags`, `importance`
/// (clamped into `0.0..=1.0`; a non-finite or non-numeric value is ignored),
/// `member_id`, `session_id`, and `visibility`. Unknown keys are ignored.
///
/// An entry with a blank `content` or a non-string `kind` is dropped with a
/// `tracing::warn!`; so is an entry whose sanitized form does not validate. Records
/// are deduplicated by normalized content, capped at
/// [`TeamExtractConfig::max_records`], tagged `llm-extractor`, and returned in answer
/// order. The result is `None` when no record survives.
#[must_use]
pub fn parse_team_extraction(
    raw: &str,
    team_id: &str,
    config: &TeamExtractConfig,
) -> Option<Vec<NewTeamMemory>> {
    let value = payload(raw)?;
    let items: Vec<&Value> = match &value {
        Value::Array(items) => items.iter().collect(),
        Value::Object(map) => {
            let nested = ["memories", "records", "items"]
                .iter()
                .find_map(|key| map.get(*key));
            match nested {
                Some(Value::Array(items)) => items.iter().collect(),
                Some(item @ Value::Object(_)) => vec![item],
                _ if map.contains_key("content") || map.contains_key("text") => vec![&value],
                _ => return None,
            }
        }
        _ => return None,
    };
    let mut records: Vec<NewTeamMemory> = Vec::new();
    let mut seen: BTreeSet<String> = BTreeSet::new();
    for item in items {
        if records.len() >= config.max_records {
            break;
        }
        let Some(record) = parse_memory_entry(item, team_id, config) else {
            continue;
        };
        if !seen.insert(normalize_key(&record.content)) {
            tracing::warn!(
                target: "openjiuwen_memory::team",
                "skipping a duplicate team memory in the model answer"
            );
            continue;
        }
        records.push(record);
    }
    if records.is_empty() {
        None
    } else {
        Some(records)
    }
}

/// Parses one entry of a model answer into a validated create request.
fn parse_memory_entry(
    value: &Value,
    team_id: &str,
    config: &TeamExtractConfig,
) -> Option<NewTeamMemory> {
    let Some(map) = value.as_object() else {
        tracing::warn!(
            target: "openjiuwen_memory::team",
            value = %value,
            "skipping a team memory that is not an object in the model answer"
        );
        return None;
    };
    let Some(raw_content) = string_field(map, &["content", "text"]) else {
        tracing::warn!(
            target: "openjiuwen_memory::team",
            "skipping a team memory without string content in the model answer"
        );
        return None;
    };
    let content = raw_content.trim();
    if content.is_empty() {
        tracing::warn!(
            target: "openjiuwen_memory::team",
            "skipping a team memory with empty content in the model answer"
        );
        return None;
    }
    let kind = match map.get("kind").or_else(|| map.get("type")) {
        Some(Value::String(raw)) => TeamMemoryKind::parse(raw),
        Some(other) => {
            tracing::warn!(
                target: "openjiuwen_memory::team",
                value = %other,
                "skipping a team memory whose kind is not a string"
            );
            return None;
        }
        None => TeamMemoryKind::SharedFact,
    };
    let tags = entry_tags(map, &kind);
    let content = truncate_chars(content, config.content_budget());
    let mut record = NewTeamMemory::new(team_id, content, kind)
        .with_source(LLM_SOURCE)
        .with_visibility(entry_visibility(map.get("visibility")));
    if let Some(importance) = map.get("importance").and_then(clamped_importance) {
        record = record.with_importance(importance);
    }
    if let Some(member_id) = entry_component(map, &["member_id", "member"]) {
        record = record.with_member(member_id);
    }
    if let Some(session_id) = entry_component(map, &["session_id", "session"]) {
        record = record.with_session(session_id);
    }
    if let Some(title) = string_field(map, &["title"])
        .map(str::trim)
        .filter(|title| !title.is_empty())
    {
        record = record.with_title(truncate_chars(title, MAX_TITLE_CHARS));
    }
    for tag in tags {
        record = record.with_tag(tag);
    }
    if let Err(error) = record.validate() {
        tracing::warn!(
            target: "openjiuwen_memory::team",
            %error,
            "skipping an invalid team memory in the model answer"
        );
        return None;
    }
    Some(record)
}

/// Collects and sanitizes an entry's tags, defaulting to the kind's extractor tag.
fn entry_tags(map: &Map<String, Value>, kind: &TeamMemoryKind) -> Vec<String> {
    let mut raw_tags: Vec<String> = Vec::new();
    match map.get("tags") {
        Some(Value::Array(items)) => {
            for item in items {
                if let Some(tag) = item.as_str() {
                    push_tag(&mut raw_tags, tag);
                }
            }
        }
        Some(Value::String(raw)) => {
            for tag in raw.split(',') {
                push_tag(&mut raw_tags, tag);
            }
        }
        _ => {}
    }
    let tags = normalize_tags(raw_tags);
    if tags.is_empty() {
        vec![kind.extractor_tag().to_owned()]
    } else {
        tags.into_iter().take(MAX_TAGS).collect()
    }
}

/// Pushes one trimmed, length-bounded tag.
fn push_tag(tags: &mut Vec<String>, raw: &str) {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return;
    }
    tags.push(truncate_chars(trimmed, MAX_TAG_CHARS));
}

/// Reads a storage-key component (member or session id), ignoring an invalid value.
fn entry_component(map: &Map<String, Value>, keys: &[&str]) -> Option<String> {
    let raw = string_field(map, keys)?.trim();
    if raw.is_empty() {
        return None;
    }
    if let Err(error) = validate_component("id", raw) {
        tracing::warn!(
            target: "openjiuwen_memory::team",
            %error,
            "ignoring an invalid id in the model answer"
        );
        return None;
    }
    Some(raw.to_owned())
}

/// Parses an entry's visibility, falling back to team-wide on an unusable value.
fn entry_visibility(value: Option<&Value>) -> TeamVisibility {
    let Some(value) = value else {
        return TeamVisibility::Team;
    };
    match serde_json::from_value::<TeamVisibility>(value.clone()) {
        Ok(visibility) if visibility.validate().is_ok() => visibility,
        _ => {
            tracing::warn!(
                target: "openjiuwen_memory::team",
                value = %value,
                "ignoring an invalid visibility in the model answer; using team"
            );
            TeamVisibility::Team
        }
    }
}

/// Finds the first rule whose markers appear in `content`.
fn classify(content: &str) -> Option<&'static MarkerRule> {
    let lowered = content.to_lowercase();
    RULES.iter().find(|rule| {
        rule.markers
            .iter()
            .any(|marker| contains_marker(&lowered, marker))
    })
}

/// Returns `true` when `lowered` contains `marker` as a whole word (ASCII markers) or
/// as a plain substring (non-ASCII markers).
///
/// The word-boundary check keeps `owns` from matching inside `downloads`; CJK markers
/// have no word boundaries, so they are matched as substrings.
fn contains_marker(lowered: &str, marker: &str) -> bool {
    if marker.is_empty() {
        return false;
    }
    if !marker.is_ascii() {
        return lowered.contains(marker);
    }
    let mut start = 0;
    while let Some(found) = lowered[start..].find(marker) {
        let begin = start + found;
        let end = begin + marker.len();
        let before_ok = lowered[..begin]
            .chars()
            .next_back()
            .is_none_or(|ch| !is_word_char(ch));
        let after_ok = lowered[end..]
            .chars()
            .next()
            .is_none_or(|ch| !is_word_char(ch));
        if before_ok && after_ok {
            return true;
        }
        start = begin + lowered[begin..].chars().next().map_or(1, char::len_utf8);
    }
    false
}

/// Returns `true` for characters that continue an ASCII word.
fn is_word_char(ch: char) -> bool {
    ch.is_alphanumeric() || ch == '_'
}

/// Builds the tags of a rule match: the marker category plus the kind's extractor tag.
///
/// The tag is listed once when the two agree (`decision` for a decision, `lesson` for
/// a handoff note); `task` additionally carries `context`, its extractor tag.
fn rule_tags(rule: &MarkerRule) -> Vec<String> {
    let mut tags = vec![rule.category.to_owned()];
    let extractor_tag = rule.kind.extractor_tag();
    if !tags.iter().any(|tag| tag == extractor_tag) {
        tags.push(extractor_tag.to_owned());
    }
    tags
}

/// Builds one validated create request.
fn build_record(
    team_id: &str,
    session_id: Option<&str>,
    member_id: Option<&str>,
    kind: &TeamMemoryKind,
    content: &str,
    tags: Vec<String>,
    importance: f32,
    source: &str,
    config: &TeamExtractConfig,
) -> Result<NewTeamMemory> {
    let content = truncate_chars(content.trim(), config.content_budget());
    let mut record = NewTeamMemory::new(team_id, content.clone(), kind.clone())
        .with_importance(importance)
        .with_source(source)
        .with_visibility(TeamVisibility::Team);
    if let Some(member_id) = member_id {
        record = record.with_member(member_id);
    }
    if let Some(session_id) = session_id {
        record = record.with_session(session_id);
    }
    if let Some(title) = summarize_title(&content) {
        record = record.with_title(title);
    }
    for tag in tags {
        record = record.with_tag(tag);
    }
    record.validate()?;
    Ok(record)
}

/// Returns a title for content longer than [`TITLE_MAX_CHARS`], cut at a word boundary.
fn summarize_title(content: &str) -> Option<String> {
    if content.chars().count() <= TITLE_MAX_CHARS {
        return None;
    }
    let head: String = content.chars().take(TITLE_MAX_CHARS).collect();
    let title = match head.rsplit_once(char::is_whitespace) {
        Some((prefix, _)) if !prefix.trim().is_empty() => prefix.trim_end(),
        _ => head.trim_end(),
    };
    if title.trim().is_empty() {
        None
    } else {
        Some(title.to_owned())
    }
}

/// Normalizes text for duplicate detection: lowercased, whitespace collapsed.
fn normalize_key(text: &str) -> String {
    text.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

/// Validates a storage-key component, mirroring the model's rules.
fn validate_component(field: &str, value: &str) -> Result<()> {
    if value.trim().is_empty() {
        return Err(Error::validation(format!(
            "team transcript `{field}` must not be empty"
        )));
    }
    if value.contains('\0') {
        return Err(Error::validation(format!(
            "team transcript `{field}` must not contain NUL (got {value:?})"
        )));
    }
    if value.contains('/') || value.contains('\\') {
        return Err(Error::validation(format!(
            "team transcript `{field}` must not contain a path separator (got {value:?})"
        )));
    }
    if value.chars().any(char::is_control) {
        return Err(Error::validation(format!(
            "team transcript `{field}` must not contain control characters (got {value:?})"
        )));
    }
    let length = value.chars().count();
    if length > MAX_SCOPE_ID_CHARS {
        return Err(Error::validation(format!(
            "team transcript `{field}` must be at most {MAX_SCOPE_ID_CHARS} characters (got {length})"
        )));
    }
    Ok(())
}

/// Truncates `text` to at most `max_chars` characters on a character boundary.
fn truncate_chars(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_owned();
    }
    text.chars().take(max_chars).collect()
}

/// Cuts the JSON payload out of a model answer.
fn payload(raw: &str) -> Option<Value> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    let body = fenced_body(trimmed).unwrap_or(trimmed);
    let slice = json_slice(body)?;
    match serde_json::from_str(slice) {
        Ok(value) => Some(value),
        Err(error) => {
            tracing::warn!(
                target: "openjiuwen_memory::team",
                %error,
                "the team memory answer is not valid JSON"
            );
            None
        }
    }
}

/// Returns the content of the first fenced code block, minus a `json` language tag.
fn fenced_body(raw: &str) -> Option<&str> {
    let open = raw.find("```")?;
    let after_open = raw.get(open + 3..)?;
    let close = after_open.find("```")?;
    Some(strip_language_tag(after_open.get(..close)?))
}

/// Drops a leading `json`/`JSON` tag from a fenced block body.
fn strip_language_tag(body: &str) -> &str {
    let trimmed = body.trim_start();
    let Some(head) = trimmed.get(..4) else {
        return trimmed;
    };
    if !head.eq_ignore_ascii_case("json") {
        return trimmed;
    }
    let rest = trimmed.get(4..).unwrap_or_default();
    let followed_by_whitespace = rest.chars().next().is_none_or(char::is_whitespace);
    if followed_by_whitespace {
        rest.trim_start()
    } else {
        trimmed
    }
}

/// Cuts the outermost JSON payload out of surrounding prose.
fn json_slice(text: &str) -> Option<&str> {
    let object_start = text.find('{');
    let array_start = text.find('[');
    let (start, close) = match (object_start, array_start) {
        (Some(object), Some(array)) if array < object => (array, ']'),
        (Some(object), _) => (object, '}'),
        (None, Some(array)) => (array, ']'),
        (None, None) => return None,
    };
    let end = text.rfind(close)?;
    if end <= start {
        return None;
    }
    text.get(start..=end)
}

/// Returns the first present string field among `keys`, untrimmed.
fn string_field<'a>(map: &'a Map<String, Value>, keys: &[&str]) -> Option<&'a str> {
    keys.iter()
        .find_map(|key| map.get(*key).and_then(Value::as_str))
}

/// Reads an `importance` number, rejecting non-finite values and clamping to `[0, 1]`.
fn clamped_importance(value: &Value) -> Option<f32> {
    let number = value.as_f64()?;
    if !number.is_finite() {
        return None;
    }
    Some(number.clamp(0.0, 1.0) as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::team::model::DEFAULT_IMPORTANCE;
    use openjiuwen_test_support::{ScriptedError, ScriptedModel};
    use tokio_util::sync::CancellationToken;

    fn at(seconds: i64) -> DateTime<Utc> {
        DateTime::from_timestamp(seconds, 0).expect("a valid timestamp")
    }

    fn turn(from: &str, content: &str) -> TeamTurn {
        TeamTurn::new(from, content, at(1_700_000_000))
    }

    fn transcript() -> TeamTranscript {
        TeamTranscript::new("review-team")
            .with_session("s-1")
            .with_turn(turn(
                "alice",
                "We decided to use SQLite instead of Postgres for the store.",
            ))
            .with_turn(turn(
                "bob",
                "I learned that parallel tasks need aligned interfaces.",
            ))
            .with_task("Write the migration")
    }

    async fn rule_records(content: &str, config: TeamExtractConfig) -> TeamExtraction {
        let transcript = TeamTranscript::new("review-team")
            .with_session("s-1")
            .with_turn(turn("alice", content));
        RuleTeamExtractor::new(config)
            .extract(&transcript, &CallOptions::new())
            .await
            .expect("the rule extractor accepts a valid transcript")
    }

    const FENCED_MEMORIES: &str = "Here is the extraction:\n```json\n{\"memories\": [\
        {\"kind\": \"decision\", \"content\": \"Use SQLite instead of Postgres.\", \
        \"title\": \"Storage\", \"tags\": [\"storage\", \"Storage\"], \"importance\": 0.9, \
        \"member_id\": \"alice\", \"session_id\": \"s-1\", \"visibility\": \"team\"},\
        {\"kind\": \"lesson\", \"content\": \"Align interfaces before parallel work.\", \
        \"importance\": 1.7}]}\n```";

    #[tokio::test]
    async fn markers_map_to_kinds_tags_and_importance() {
        let cases = [
            (
                "We decided to use SQLite instead of Postgres.",
                TeamMemoryKind::Decision,
                vec!["decision"],
                DECISION_IMPORTANCE,
            ),
            (
                "I learned that parallel tasks need aligned interfaces.",
                TeamMemoryKind::HandoffNote,
                vec!["lesson"],
                LESSON_IMPORTANCE,
            ),
            (
                "Bob is good at writing SQL.",
                TeamMemoryKind::MemberProfile,
                vec!["member"],
                MEMBER_IMPORTANCE,
            ),
            (
                "Assigned to Alice: write the migration script.",
                TeamMemoryKind::Task,
                vec!["task", "context"],
                TASK_IMPORTANCE,
            ),
            (
                "Constraint: the release must ship before Friday.",
                TeamMemoryKind::SharedFact,
                vec!["context"],
                CONTEXT_IMPORTANCE,
            ),
        ];
        for (content, kind, tags, importance) in cases {
            let extraction = rule_records(content, TeamExtractConfig::default()).await;
            assert_eq!(extraction.len(), 1, "no record for {content}");
            let record = &extraction.records[0];
            assert_eq!(record.kind, kind, "wrong kind for {content}");
            assert_eq!(record.tags, tags, "wrong tags for {content}");
            assert!((record.importance.unwrap_or_default() - importance).abs() < f32::EPSILON);
            assert_eq!(record.member_id.as_deref(), Some("alice"));
            assert_eq!(record.session_id.as_deref(), Some("s-1"));
            assert_eq!(record.source.as_deref(), Some(RULE_SOURCE));
            assert_eq!(record.visibility, TeamVisibility::Team);
            assert_eq!(record.team_id, "review-team");
            assert_eq!(record.content, content);
            record.validate().expect("the record validates");
        }
    }

    #[tokio::test]
    async fn chinese_markers_are_recognized() {
        let cases = [
            ("我们决定使用 SQLite", TeamMemoryKind::Decision),
            ("经验：并行任务要先对齐接口", TeamMemoryKind::HandoffNote),
            ("Alice 负责迁移脚本", TeamMemoryKind::MemberProfile),
            ("待办：编写迁移脚本并补充测试用例", TeamMemoryKind::Task),
            ("约束：必须在周五之前完成发布", TeamMemoryKind::SharedFact),
        ];
        for (content, kind) in cases {
            let extraction = rule_records(content, TeamExtractConfig::default()).await;
            assert_eq!(extraction.len(), 1, "no record for {content}");
            assert_eq!(extraction.records[0].kind, kind, "wrong kind for {content}");
        }
    }

    #[tokio::test]
    async fn marker_free_turns_produce_nothing() {
        let extraction = rule_records(
            "Alice pushed a commit and the tests are green.",
            TeamExtractConfig::default(),
        )
        .await;
        assert!(extraction.is_empty());
        assert!(extraction.degraded.is_none());
    }

    #[tokio::test]
    async fn short_content_is_skipped() {
        let config = TeamExtractConfig::default();
        // Contains a marker but is shorter than `min_content_chars`.
        let extraction = rule_records("we chose a", config).await;
        assert!(extraction.is_empty());
        let config = TeamExtractConfig {
            min_content_chars: 5,
            ..Default::default()
        };
        let extraction = rule_records("we chose a", config).await;
        assert_eq!(extraction.len(), 1);
    }

    #[tokio::test]
    async fn duplicate_turns_are_deduped() {
        let transcript = TeamTranscript::new("review-team")
            .with_turn(turn("alice", "We decided to use SQLite."))
            .with_turn(turn("bob", "  we decided   to use sqlite.  "));
        let extraction = RuleTeamExtractor::new(TeamExtractConfig::default())
            .extract(&transcript, &CallOptions::new())
            .await
            .expect("valid transcript");
        assert_eq!(extraction.len(), 1);
        assert_eq!(extraction.records[0].member_id.as_deref(), Some("alice"));
    }

    #[tokio::test]
    async fn task_notes_become_task_records_after_the_turns() {
        let extraction = RuleTeamExtractor::new(TeamExtractConfig::default())
            .extract(&transcript(), &CallOptions::new())
            .await
            .expect("valid transcript");
        assert_eq!(extraction.len(), 3);
        assert_eq!(extraction.records[0].kind, TeamMemoryKind::Decision);
        assert_eq!(extraction.records[1].kind, TeamMemoryKind::HandoffNote);
        let task = &extraction.records[2];
        assert_eq!(task.kind, TeamMemoryKind::Task);
        assert_eq!(task.content, "Write the migration");
        assert_eq!(task.tags, vec!["task", "context"]);
        assert_eq!(task.member_id, None);
        assert_eq!(task.session_id.as_deref(), Some("s-1"));
        assert_eq!(task.source.as_deref(), Some(RULE_SOURCE));
    }

    #[tokio::test]
    async fn the_record_cap_applies_to_turns_and_notes() {
        let config = TeamExtractConfig {
            max_records: 2,
            ..Default::default()
        };
        let extraction = RuleTeamExtractor::new(config)
            .extract(&transcript(), &CallOptions::new())
            .await
            .expect("valid transcript");
        assert_eq!(extraction.len(), 2);
        assert_eq!(extraction.records[1].kind, TeamMemoryKind::HandoffNote);
    }

    #[tokio::test]
    async fn content_is_truncated_and_long_content_gets_a_title() {
        let long = format!("We decided to {}.", "go ".repeat(200));
        let config = TeamExtractConfig {
            max_turn_chars: 120,
            ..Default::default()
        };
        let extraction = rule_records(&long, config).await;
        let record = &extraction.records[0];
        assert_eq!(record.content.chars().count(), 120);
        let title = record.title.as_deref().expect("long content gets a title");
        assert!(title.chars().count() <= TITLE_MAX_CHARS);
        assert!(record.content.starts_with(title));
        assert!(!title.ends_with(' '));
        record.validate().expect("the record validates");
    }

    #[tokio::test]
    async fn every_rule_record_materializes() {
        let extraction = RuleTeamExtractor::new(TeamExtractConfig::default())
            .extract(&transcript(), &CallOptions::new())
            .await
            .expect("valid transcript");
        for record in &extraction.records {
            record.validate().expect("the create request validates");
            record
                .to_record(at(1_700_000_000), DEFAULT_IMPORTANCE)
                .expect("the record materializes");
        }
    }

    #[tokio::test]
    async fn a_disabled_config_returns_an_empty_result() {
        let config = TeamExtractConfig {
            enabled: false,
            ..Default::default()
        };
        let extraction = RuleTeamExtractor::new(config)
            .extract(&transcript(), &CallOptions::new())
            .await
            .expect("a disabled extractor still succeeds");
        assert!(extraction.is_empty());
        assert!(extraction.degraded.is_none());
    }

    #[tokio::test]
    async fn cancellation_and_invalid_input_are_errors() {
        let cancellation = CancellationToken::new();
        cancellation.cancel();
        let options = CallOptions::new().with_cancellation(cancellation);
        let error = RuleTeamExtractor::new(TeamExtractConfig::default())
            .extract(&transcript(), &options)
            .await
            .expect_err("a cancelled call fails");
        assert!(matches!(error, Error::Cancelled));

        let extractor = RuleTeamExtractor::new(TeamExtractConfig::default());
        let error = extractor
            .extract(
                &TeamTranscript::new("review-team").with_turn(turn("alice", "   ")),
                &CallOptions::new(),
            )
            .await
            .expect_err("a blank turn is invalid");
        assert!(matches!(error, Error::Validation(_)));

        let config = TeamExtractConfig {
            max_records: 0,
            ..Default::default()
        };
        let error = RuleTeamExtractor::new(config)
            .extract(&transcript(), &CallOptions::new())
            .await
            .expect_err("a zero cap is invalid");
        assert!(matches!(error, Error::Validation(_)));
    }

    #[test]
    fn transcript_and_turn_validation() {
        assert!(TeamTranscript::new("").validate().is_err());
        assert!(TeamTranscript::new("a/b").validate().is_err());
        assert!(TeamTranscript::new("review-team")
            .with_session("")
            .validate()
            .is_err());
        assert!(TeamTranscript::new("review-team")
            .with_turn(turn("", "we decided to ship"))
            .validate()
            .is_err());
        assert!(TeamTranscript::new("review-team")
            .with_turn(turn("alice", "   "))
            .validate()
            .is_err());
        assert!(TeamTranscript::new("review-team")
            .with_turn(turn("alice", "we decided to ship").with_to(" "))
            .validate()
            .is_err());
        assert!(TeamTranscript::new("review-team")
            .with_task("x".repeat(MAX_TASK_NOTE_CHARS + 1))
            .validate()
            .is_err());
        assert!(TeamTranscript::new("review-team")
            .with_turn(turn("alice", "we decided to ship").with_to("bob"))
            .with_task("a note")
            .validate()
            .is_ok());
        assert!(TeamTranscript::new("review-team").is_empty());
        assert!(!transcript().is_empty());
    }

    #[test]
    fn config_defaults_and_validation() {
        let config = TeamExtractConfig::default();
        assert!(config.enabled);
        assert_eq!(config.max_records, 8);
        assert_eq!(config.max_turn_chars, 1000);
        assert_eq!(config.max_transcript_chars, 8000);
        assert_eq!(config.min_content_chars, 12);
        assert_eq!(config.model_retries, 1);
        config.validate().expect("the defaults are valid");
        let json = serde_json::to_string(&config).expect("the config is serializable");
        let restored: TeamExtractConfig =
            serde_json::from_str(&json).expect("the config round-trips");
        assert_eq!(restored, config);
        for config in [
            TeamExtractConfig {
                max_turn_chars: 0,
                ..Default::default()
            },
            TeamExtractConfig {
                max_transcript_chars: 0,
                ..Default::default()
            },
            TeamExtractConfig {
                min_content_chars: 2000,
                ..Default::default()
            },
        ] {
            assert!(config.validate().is_err());
        }
    }

    #[test]
    fn parsing_accepts_fenced_and_bare_shapes() {
        let config = TeamExtractConfig::default();
        let records = parse_team_extraction(FENCED_MEMORIES, "review-team", &config)
            .expect("the fenced answer parses");
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].kind, TeamMemoryKind::Decision);
        assert_eq!(records[0].content, "Use SQLite instead of Postgres.");
        assert_eq!(records[0].title.as_deref(), Some("Storage"));
        assert_eq!(records[0].tags, vec!["storage"]);
        assert_eq!(records[0].member_id.as_deref(), Some("alice"));
        assert_eq!(records[0].session_id.as_deref(), Some("s-1"));
        assert_eq!(records[0].source.as_deref(), Some(LLM_SOURCE));
        assert!((records[0].importance.unwrap_or_default() - 0.9).abs() < f32::EPSILON);
        assert_eq!(records[1].kind, TeamMemoryKind::HandoffNote);
        // 1.7 is clamped, never stored as-is.
        assert!((records[1].importance.unwrap_or_default() - 1.0).abs() < f32::EPSILON);
        // Missing tags default to the kind's extractor tag.
        assert_eq!(records[1].tags, vec!["lesson"]);

        let bare_object = parse_team_extraction(
            r#"{"memories": [{"content": "Ship on Friday."}]}"#,
            "review-team",
            &config,
        )
        .expect("a bare object parses");
        assert_eq!(bare_object[0].kind, TeamMemoryKind::SharedFact);
        assert_eq!(bare_object[0].tags, vec!["context"]);

        let bare_array = parse_team_extraction(
            r#"[{"kind": "member", "content": "Bob owns the release process."}]"#,
            "review-team",
            &config,
        )
        .expect("a bare array parses");
        assert_eq!(bare_array[0].kind, TeamMemoryKind::MemberProfile);
    }

    #[test]
    fn parsing_ignores_unknown_keys_and_invalid_values() {
        let config = TeamExtractConfig::default();
        let records = parse_team_extraction(
            r#"{"memories": [
                {"kind": "decision", "content": "Use SQLite.", "unknown": 1, "importance": 0.5},
                {"kind": "lesson", "content": "Align first.", "importance": "high"},
                {"kind": "lesson", "content": "Tag early.", "importance": -3},
                {"kind": "lesson", "content": "Test first.", "tags": ["  ", "CI"],
                 "visibility": {"members": ["alice"]}, "member_id": "a/b",
                 "session_id": "s-9"}
            ]}"#,
            "review-team",
            &config,
        )
        .expect("the answer parses");
        assert_eq!(records.len(), 4);
        assert!((records[0].importance.unwrap_or_default() - 0.5).abs() < f32::EPSILON);
        // A non-numeric importance is ignored, leaving the default fallback.
        assert_eq!(records[1].importance, None);
        // A negative importance is clamped to zero.
        assert!((records[2].importance.unwrap_or_default()).abs() < f32::EPSILON);
        assert_eq!(records[3].tags, vec!["ci"]);
        assert_eq!(records[3].visibility, TeamVisibility::members(["alice"]));
        // An id that is not a storage-key component is dropped, not stored.
        assert_eq!(records[3].member_id, None);
        assert_eq!(records[3].session_id.as_deref(), Some("s-9"));
    }

    #[test]
    fn parsing_drops_unusable_entries_and_dedupes() {
        let config = TeamExtractConfig::default();
        let records = parse_team_extraction(
            r#"[{"kind": "decision", "content": "Use SQLite."},
                {"kind": "decision", "content": "  use   sqlite. "},
                {"kind": "decision", "content": "   "},
                {"kind": 7, "content": "Use Redis."},
                {"content": "Ship on Friday."},
                {"kind": "decision"}]"#,
            "review-team",
            &config,
        )
        .expect("the surviving entries parse");
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].content, "Use SQLite.");
        assert_eq!(records[1].content, "Ship on Friday.");

        assert!(
            parse_team_extraction("[{\"kind\": \"decision\"}]", "review-team", &config).is_none()
        );
        assert!(parse_team_extraction("nothing here", "review-team", &config).is_none());
        assert!(parse_team_extraction("", "review-team", &config).is_none());
        assert!(parse_team_extraction("{\"memories\": []}", "review-team", &config).is_none());
    }

    #[test]
    fn parsing_caps_records_and_truncates_content() {
        let config = TeamExtractConfig {
            max_records: 2,
            max_turn_chars: 10,
            min_content_chars: 1,
            ..Default::default()
        };
        let records = parse_team_extraction(
            r#"[{"content": "one two three four five"},
                {"content": "second memory"},
                {"content": "third memory"}]"#,
            "review-team",
            &config,
        )
        .expect("the answer parses");
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].content, "one two th");
    }

    fn llm_extractor(model: &ScriptedModel, config: TeamExtractConfig) -> LlmTeamExtractor {
        LlmTeamExtractor::new(Arc::new(model.clone()), config)
    }

    #[tokio::test]
    async fn llm_happy_path_returns_parsed_records() {
        let model = ScriptedModel::builder()
            .respond_text(FENCED_MEMORIES)
            .build();
        let handle = model.clone();
        let extractor = llm_extractor(&model, TeamExtractConfig::default());
        let extraction = extractor
            .extract(&transcript(), &CallOptions::new())
            .await
            .expect("the model answers");
        assert_eq!(extraction.len(), 2);
        assert!(extraction.degraded.is_none());
        assert_eq!(extraction.records[0].source.as_deref(), Some(LLM_SOURCE));
        assert_eq!(TeamMemoryExtractor::name(&extractor), "llm");
        assert_eq!(handle.call_count(), 1);
        let call = handle.last_call().expect("the model was called");
        assert_eq!(
            call.messages[0].text().as_ref(),
            EXTRACTION_SYSTEM_PROMPT,
            "the system prompt is sent verbatim"
        );
        let user = call.last_user_text().expect("the user message is sent");
        assert!(user.contains("# Team collaboration record for review-team"));
        assert!(user.contains("We decided to use SQLite"));
        assert!(user.contains("### Write the migration"));
    }

    #[tokio::test]
    async fn llm_unparseable_answer_falls_back_to_rules() {
        let model = ScriptedModel::builder()
            .respond_text("I think the team decided to use SQLite.")
            .build();
        let handle = model.clone();
        let extraction = llm_extractor(&model, TeamExtractConfig::default())
            .extract(&transcript(), &CallOptions::new())
            .await
            .expect("the fallback succeeds");
        assert!(extraction
            .degraded
            .as_deref()
            .is_some_and(|reason| reason.contains("fell back to rules")));
        assert!(extraction
            .degraded
            .as_deref()
            .is_some_and(|reason| reason.contains("no usable team memories")));
        assert_eq!(extraction.len(), 3);
        assert_eq!(extraction.records[0].source.as_deref(), Some(RULE_SOURCE));
        assert_eq!(handle.call_count(), 2);
    }

    #[tokio::test]
    async fn llm_provider_failure_falls_back_with_the_error_text() {
        let model = ScriptedModel::builder()
            .fail(ScriptedError::llm("provider is down"))
            .build();
        let handle = model.clone();
        let extraction = llm_extractor(&model, TeamExtractConfig::default())
            .extract(&transcript(), &CallOptions::new())
            .await
            .expect("the fallback succeeds");
        let reason = extraction
            .degraded
            .as_deref()
            .expect("the degradation is reported");
        assert!(reason.contains("fell back to rules"));
        assert!(reason.contains("provider is down"));
        assert_eq!(extraction.records[0].kind, TeamMemoryKind::Decision);
        assert_eq!(handle.call_count(), 2);
    }

    #[tokio::test]
    async fn llm_retries_can_still_succeed() {
        let model = ScriptedModel::builder()
            .fail(ScriptedError::transport("flaky"))
            .respond_text(FENCED_MEMORIES)
            .build();
        let handle = model.clone();
        let extraction = llm_extractor(&model, TeamExtractConfig::default())
            .extract(&transcript(), &CallOptions::new())
            .await
            .expect("the retry succeeds");
        assert!(extraction.degraded.is_none());
        assert_eq!(extraction.records[0].source.as_deref(), Some(LLM_SOURCE));
        assert_eq!(handle.call_count(), 2);
    }

    #[tokio::test]
    async fn llm_cancellation_returns_early_without_falling_back() {
        let model = ScriptedModel::builder()
            .respond_text(FENCED_MEMORIES)
            .build();
        let handle = model.clone();
        let cancellation = CancellationToken::new();
        cancellation.cancel();
        let options = CallOptions::new().with_cancellation(cancellation);
        let error = llm_extractor(&model, TeamExtractConfig::default())
            .extract(&transcript(), &options)
            .await
            .expect_err("a cancelled call fails");
        assert!(matches!(error, Error::Cancelled));
        assert_eq!(handle.call_count(), 0);
    }

    #[tokio::test]
    async fn llm_disabled_config_skips_the_model() {
        let model = ScriptedModel::builder()
            .respond_text(FENCED_MEMORIES)
            .build();
        let handle = model.clone();
        let config = TeamExtractConfig {
            enabled: false,
            ..Default::default()
        };
        let extraction = llm_extractor(&model, config)
            .extract(&transcript(), &CallOptions::new())
            .await
            .expect("a disabled extractor succeeds");
        assert!(extraction.is_empty());
        assert!(extraction.degraded.is_none());
        assert_eq!(handle.call_count(), 0);
    }

    #[tokio::test]
    async fn llm_validates_config_and_transcript_before_calling() {
        let model = ScriptedModel::builder()
            .respond_text(FENCED_MEMORIES)
            .build();
        let handle = model.clone();
        let config = TeamExtractConfig {
            max_records: 0,
            ..Default::default()
        };
        let error = llm_extractor(&model, config)
            .extract(&transcript(), &CallOptions::new())
            .await
            .expect_err("an invalid config fails");
        assert!(matches!(error, Error::Validation(_)));
        let error = llm_extractor(&model, TeamExtractConfig::default())
            .extract(&TeamTranscript::new(""), &CallOptions::new())
            .await
            .expect_err("an invalid transcript fails");
        assert!(matches!(error, Error::Validation(_)));
        assert_eq!(handle.call_count(), 0);
    }

    #[tokio::test]
    async fn llm_fallback_can_be_replaced() {
        let model = ScriptedModel::builder().build();
        let fallback = RuleTeamExtractor::new(TeamExtractConfig {
            max_records: 1,
            ..Default::default()
        });
        let extractor = llm_extractor(&model, TeamExtractConfig::default()).with_fallback(fallback);
        assert_eq!(extractor.config().max_records, DEFAULT_MAX_RECORDS);
        let extraction = extractor
            .extract(&transcript(), &CallOptions::new())
            .await
            .expect("the fallback succeeds");
        assert_eq!(extraction.len(), 1);
        assert!(extraction.degraded.is_some());
    }

    #[test]
    fn word_boundaries_reject_marker_lookalikes() {
        assert!(contains_marker("we decided to ship", "we decided"));
        assert!(!contains_marker("unwe decided to ship", "we decided"));
        assert!(!contains_marker("alice disowns the plan", "owns"));
        assert!(contains_marker("alice owns the release", "owns"));
        assert!(contains_marker("我们决定发布", "决定"));
    }

    #[test]
    fn titles_are_cut_at_word_boundaries() {
        assert_eq!(summarize_title("short content"), None);
        let long = format!("{} tail", "word ".repeat(30));
        let title = summarize_title(&long).expect("long content gets a title");
        assert!(title.chars().count() <= TITLE_MAX_CHARS);
        assert!(title.starts_with("word word"));
        assert!(!title.ends_with(' '));
    }
}
