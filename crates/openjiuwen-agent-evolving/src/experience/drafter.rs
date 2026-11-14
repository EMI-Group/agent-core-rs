//! The LLM experience draft pipeline: prompt-driven drafts over a deterministic
//! fallback.
//!
//! Ports the portable half of Python `optimizer/skill_call/experience_optimizer.py`
//! and `experience/scorer.py`: a model reads the skill document, the evolution
//! signals, the trajectories and the existing lessons, and answers with a bare
//! JSON array of experiences. The Rust port keeps the same output contract and
//! adds one hard guarantee Python lacks: **a model failure never fails the
//! pipeline**. Any error, unusable answer or explicit format violation falls
//! back to [`ExperienceExtractor`], which derives the same draft from recorded
//! evidence without a model.
//!
//! Two rules make the offline path reproducible: drafts derive from the input
//! order only (no clock, no I/O, no randomness), and every model call is
//! bounded by [`ExperienceDrafter::max_items`] and the prompt builders below.
//!
//! The model is optional ([`ExperienceDrafter::has_model`]); a drafter without
//! one is a pure function of its input.

use std::fmt;
use std::sync::Arc;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_core::foundation::llm::{Model, ModelRequestConfig};
use openjiuwen_core::foundation::prompt::BaseMessage;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::experience::archive::tokens;
use crate::experience::draft::{
    normalize_evolve_draft, normalize_simplify_draft, validate_simplify_record_refs,
    DraftExperience, DraftSection, EvolutionSubject, EvolutionTarget, EvolveDraft, SimplifyAction,
    SimplifyActionKind, SimplifyDraft,
};
use crate::experience::extract::ExperienceExtractor;
use crate::experience::models::{ExperienceRecord, ExperienceSection};
use crate::signal::{truncate_excerpt, EvolutionSignal};
use crate::trajectory::{render_digest, Trajectory};

use crate::experience::prompts::render_template;
pub use crate::experience::prompts::{DRAFT_PROMPT, FORMAT_REPAIR_PROMPT, SIMPLIFY_PROMPT};
/// The no-model default for `ExperienceDrafter::max_items`: Python's two text
/// experiences plus one script experience.
pub const DEFAULT_MAX_ITEMS: usize = 3;

/// The no-model default for `ExperienceDrafter::max_summary_chars`: the
/// protocol [`SUMMARY_LIMIT`](crate::experience::draft::SUMMARY_LIMIT).
pub const DEFAULT_MAX_SUMMARY_CHARS: usize = 160;

/// The no-model default for `ExperienceDrafter::max_skill_chars`, matching
/// Python's `_SKILL_CONTENT_MAX_CHARS`.
pub const DEFAULT_MAX_SKILL_CHARS: usize = 6000;

/// The score at or above which the deterministic simplify path keeps a record.
pub const SIMPLIFY_KEEP_SCORE: f64 = 0.4;

/// How many signals, trajectories and existing records one prompt carries.
const MAX_PROMPT_ITEMS: usize = 12;
/// How many characters one signal excerpt contributes to a prompt.
const PROMPT_SIGNAL_CHARS: usize = 240;
/// How many characters one trajectory digest contributes to a prompt.
const PROMPT_DIGEST_CHARS: usize = 600;
/// How many characters of a rejected reply the repair prompt shows.
const REPAIR_PREVIEW_CHARS: usize = 500;
/// How many keywords a deterministic draft lifts from its content.
const DRAFT_KEYWORD_LIMIT: usize = 8;

/// The borrowed input of one [`ExperienceDrafter::draft`] call.
///
/// This is a view over caller-owned evidence, not a stored record, so it is
/// deliberately not serde-serializable.
#[derive(Debug, Clone, PartialEq)]
pub struct DraftInput<'a> {
    /// The skill the draft is for; must be non-empty.
    pub skill_name: &'a str,
    /// The current `SKILL.md` content, truncated at
    /// `ExperienceDrafter::max_skill_chars` for the prompt.
    pub skill_content: &'a str,
    /// The evolution signals attributed to the skill.
    pub signals: &'a [EvolutionSignal],
    /// The recorded runs that produced the signals.
    pub trajectories: &'a [Trajectory],
    /// The user's requested optimization direction; may be empty.
    pub user_query: &'a str,
    /// Rendered summaries of the skill's existing experiences.
    pub existing: Vec<String>,
}

impl<'a> DraftInput<'a> {
    /// Creates an input with no trajectories, user query or existing records.
    #[must_use]
    pub fn new(
        skill_name: &'a str,
        skill_content: &'a str,
        signals: &'a [EvolutionSignal],
    ) -> Self {
        Self {
            skill_name,
            skill_content,
            signals,
            trajectories: &[],
            user_query: "",
            existing: Vec::new(),
        }
    }

    /// Sets the trajectories.
    #[must_use]
    pub fn with_trajectories(mut self, trajectories: &'a [Trajectory]) -> Self {
        self.trajectories = trajectories;
        self
    }

    /// Sets the user's requested direction.
    #[must_use]
    pub fn with_user_query(mut self, user_query: &'a str) -> Self {
        self.user_query = user_query;
        self
    }

    /// Sets the rendered existing-experience summaries.
    #[must_use]
    pub fn with_existing(mut self, existing: Vec<String>) -> Self {
        self.existing = existing;
        self
    }
}

/// The outcome of parsing one model reply.
///
/// `appended` counts the items the model asked to add and `skipped` the ones it
/// explicitly rejected; an explicit all-`skip` answer is a valid empty draft
/// (`appended == 0`, `skipped > 0`) and is *not* replaced by the deterministic
/// fallback.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct DraftParse {
    /// The normalized draft the reply described.
    pub draft: EvolveDraft,
    /// How many experiences the reply asked to add.
    pub appended: usize,
    /// How many items the reply explicitly skipped.
    pub skipped: usize,
}

/// Drafts evolution experiences, with an optional model and a deterministic
/// fallback.
///
/// With a model installed the prompt path runs first, retries once through
/// [`FORMAT_REPAIR_PROMPT`] when the answer is not parseable, and otherwise
/// falls back to [`ExperienceExtractor`]. Model errors are never propagated.
#[derive(Clone)]
pub struct ExperienceDrafter {
    /// The optional model driving the prompt path; `None` keeps drafting offline.
    model: Option<Arc<dyn Model>>,
    /// The request sent with every model call.
    request: ModelRequestConfig,
    /// The hard cap on experiences one draft carries; `0` drafts nothing.
    max_items: usize,
    /// The summary length the deterministic path truncates to.
    max_summary_chars: usize,
    /// The skill content length the prompt keeps before truncating.
    max_skill_chars: usize,
}

impl fmt::Debug for ExperienceDrafter {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut debug = formatter.debug_struct("ExperienceDrafter");
        debug
            .field("request", &self.request)
            .field("max_items", &self.max_items)
            .field("max_summary_chars", &self.max_summary_chars)
            .field("max_skill_chars", &self.max_skill_chars);
        if let Some(model) = &self.model {
            debug.field("model", &model.model_name());
        }
        debug.finish()
    }
}

impl Default for ExperienceDrafter {
    fn default() -> Self {
        Self {
            model: None,
            request: ModelRequestConfig::default(),
            max_items: DEFAULT_MAX_ITEMS,
            max_summary_chars: DEFAULT_MAX_SUMMARY_CHARS,
            max_skill_chars: DEFAULT_MAX_SKILL_CHARS,
        }
    }
}

impl ExperienceDrafter {
    /// Builds an offline drafter; drafting is deterministic until
    /// [`Self::with_model`] is called.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Installs the model used by the prompt-driven path.
    ///
    /// A failing or unusable answer falls back to the deterministic path, so
    /// injecting a model never makes drafting failable.
    #[must_use]
    pub fn with_model(mut self, model: Arc<dyn Model>, request: ModelRequestConfig) -> Self {
        self.model = Some(model);
        self.request = request;
        self
    }

    /// Sets the hard cap on experiences one draft carries.
    #[must_use]
    pub fn with_max_items(mut self, max_items: usize) -> Self {
        self.max_items = max_items;
        self
    }

    /// Sets the summary length the deterministic path truncates to.
    #[must_use]
    pub fn with_max_summary_chars(mut self, max_summary_chars: usize) -> Self {
        self.max_summary_chars = max_summary_chars;
        self
    }

    /// Sets the skill content length the prompt keeps.
    #[must_use]
    pub fn with_max_skill_chars(mut self, max_skill_chars: usize) -> Self {
        self.max_skill_chars = max_skill_chars;
        self
    }

    /// Whether the prompt-driven path is available.
    #[must_use]
    pub fn has_model(&self) -> bool {
        self.model.is_some()
    }

    /// Drafts experiences for `input`.
    ///
    /// With a model, the prompt path answers first. A parse failure triggers one
    /// [`FORMAT_REPAIR_PROMPT`] retry; a second failure, a model error, or an
    /// answer with neither appended nor skipped items falls back to
    /// [`Self::draft_offline`]. An explicit all-`skip` answer is honoured and
    /// yields an empty draft.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] only when `input.skill_name` is blank; model and
    /// evidence problems never surface as errors.
    pub async fn draft(&self, input: &DraftInput<'_>) -> Result<EvolveDraft> {
        let subject = EvolutionSubject::for_skill(input.skill_name)?;
        if let Some(parse) = self.draft_with_model(&subject, input).await {
            if parse.appended > 0 {
                let mut draft = parse.draft;
                draft.experiences.truncate(self.max_items);
                return Ok(draft);
            }
            if parse.skipped > 0 {
                return Ok(parse.draft);
            }
        }
        Ok(self.draft_offline(&subject, input))
    }

    /// The deterministic draft: one experience per extracted record.
    ///
    /// Records come from [`ExperienceExtractor::from_signals`] (confidence at
    /// least `0.5`) followed by [`ExperienceExtractor::from_trajectories`], and
    /// every experience targets `body` so a reviewer keeps full control over
    /// placement. The summary is the collapsed record content truncated at
    /// `Self::max_summary_chars`, the section follows the record section, and
    /// `reason` names the provenance and confidence the record carried.
    #[must_use]
    pub fn draft_offline(&self, subject: &EvolutionSubject, input: &DraftInput<'_>) -> EvolveDraft {
        let extractor = ExperienceExtractor::new().with_max_records(self.max_items.max(1));
        let mut records = extractor.from_signals(input.signals);
        records.extend(extractor.from_trajectories(input.trajectories));
        let mut experiences = Vec::new();
        for record in records {
            if experiences.len() >= self.max_items {
                break;
            }
            experiences.push(self.experience_from_record(record));
        }
        EvolveDraft::new(subject.clone(), experiences)
    }

    /// Projects one extracted record into a draft experience.
    fn experience_from_record(&self, record: ExperienceRecord) -> DraftExperience {
        let summary = truncate_excerpt(
            &collapse_whitespace(&record.content),
            self.max_summary_chars.max(1),
        );
        DraftExperience {
            summary,
            content: record.content.clone(),
            target: EvolutionTarget::Body,
            section: draft_section(record.section),
            reason: format!(
                "derived from {} evidence with confidence {:.2}",
                record.source, record.confidence
            ),
            script_filename: None,
            script_language: None,
            script_purpose: None,
            source_refs: vec![record.record_id],
        }
    }

    /// Runs the prompt path, returning `None` when it yields no usable answer.
    async fn draft_with_model(
        &self,
        subject: &EvolutionSubject,
        input: &DraftInput<'_>,
    ) -> Option<DraftParse> {
        let model = self.model.as_ref()?;
        let prompt = self.build_draft_prompt(input);
        let first = self.invoke(model, &prompt).await?;
        match parse_draft_text(subject, &first) {
            Ok(parse) => Some(parse),
            Err(error) => {
                tracing::debug!(
                    model = model.model_name(),
                    error = %error,
                    "experience draft answer was unusable, asking for a repair"
                );
                let repair = self.build_repair_prompt(&error.to_string(), &first);
                let second = self.invoke(model, &repair).await?;
                match parse_draft_text(subject, &second) {
                    Ok(parse) => Some(parse),
                    Err(error) => {
                        tracing::debug!(
                            model = model.model_name(),
                            error = %error,
                            "experience draft repair failed, using the deterministic path"
                        );
                        None
                    }
                }
            }
        }
    }

    /// Sends one user message and returns its text; errors become `None`.
    async fn invoke(&self, model: &Arc<dyn Model>, prompt: &str) -> Option<String> {
        let messages = [BaseMessage::user(prompt)];
        match model.invoke(&messages, &self.request).await {
            Ok(response) => {
                let text = response
                    .message()
                    .map_or_else(String::new, |message| message.text().into_owned());
                Some(text)
            }
            Err(error) => {
                tracing::debug!(
                    model = model.model_name(),
                    error = %error,
                    "experience draft model call failed, using the deterministic path"
                );
                None
            }
        }
    }

    /// Renders [`DRAFT_PROMPT`] for `input`.
    #[must_use]
    pub fn build_draft_prompt(&self, input: &DraftInput<'_>) -> String {
        let skill_content = summarize_skill_content(input.skill_content, self.max_skill_chars);
        let signals = render_signals(input.signals);
        let trajectories = render_trajectories(input.trajectories);
        let existing = if input.existing.is_empty() {
            "No existing records".to_owned()
        } else {
            input
                .existing
                .iter()
                .take(MAX_PROMPT_ITEMS)
                .map(|line| format!("- {line}"))
                .collect::<Vec<_>>()
                .join("\n")
        };
        let user_query = if input.user_query.trim().is_empty() {
            "None"
        } else {
            input.user_query.trim()
        };
        render_template(
            DRAFT_PROMPT,
            &[
                ("skill_name", input.skill_name.trim()),
                ("skill_content", &skill_content),
                ("signals", &signals),
                ("trajectories", &trajectories),
                ("existing", &existing),
                ("user_query", user_query),
            ],
        )
    }

    /// Renders [`FORMAT_REPAIR_PROMPT`] for one failed answer.
    #[must_use]
    pub fn build_repair_prompt(&self, error: &str, previous: &str) -> String {
        let preview = truncate_excerpt(previous.trim(), REPAIR_PREVIEW_CHARS);
        render_template(
            FORMAT_REPAIR_PROMPT,
            &[("error", error), ("previous", &preview)],
        )
    }

    /// Renders [`SIMPLIFY_PROMPT`] for `records`.
    #[must_use]
    pub fn build_simplify_prompt(
        &self,
        subject: &EvolutionSubject,
        records: &[ExperienceRecord],
        user_intent: &str,
    ) -> String {
        let rendered = records
            .iter()
            .take(MAX_PROMPT_ITEMS)
            .map(render_record_line)
            .collect::<Vec<_>>()
            .join("\n");
        let intent = if user_intent.trim().is_empty() {
            "None"
        } else {
            user_intent.trim()
        };
        let values: [(&str, &str); 3] = [
            ("skill_name", subject.name.as_str()),
            ("records", &rendered),
            ("user_intent", intent),
        ];
        render_template(SIMPLIFY_PROMPT, &values)
    }

    /// Proposes maintenance actions for `records`.
    ///
    /// With a model, the prompt path answers first; a model error, an unusable
    /// answer, an action naming an unknown record, or an empty action list falls
    /// back to [`Self::simplify_offline`].
    ///
    /// The deterministic rule mirrors Python's first maintenance rule: a record
    /// scored below [`SIMPLIFY_KEEP_SCORE`] that was **never used** is deleted,
    /// and every other record is kept (so a low-scored but used lesson survives).
    ///
    /// # Errors
    ///
    /// Never in practice: the deterministic path is total. The signature stays
    /// `Result` so a future model path can surface typed failures.
    pub async fn simplify(
        &self,
        subject: &EvolutionSubject,
        records: &[ExperienceRecord],
        user_intent: &str,
    ) -> Result<SimplifyDraft> {
        if records.is_empty() {
            return Ok(SimplifyDraft::new(subject.clone(), Vec::new()));
        }
        if let Some(draft) = self
            .simplify_with_model(subject, records, user_intent)
            .await
        {
            return Ok(draft);
        }
        Ok(self.simplify_offline(subject, records))
    }

    /// Runs the simplify prompt path, returning `None` when it is unusable.
    async fn simplify_with_model(
        &self,
        subject: &EvolutionSubject,
        records: &[ExperienceRecord],
        user_intent: &str,
    ) -> Option<SimplifyDraft> {
        let model = self.model.as_ref()?;
        let prompt = self.build_simplify_prompt(subject, records, user_intent);
        let answer = self.invoke(model, &prompt).await?;
        let value = extract_json_value(&answer).ok()?;
        let items = json_items(value, "actions")?;
        if items.is_empty() {
            return None;
        }
        let draft = normalize_simplify_draft(subject.clone(), &items).ok()?;
        let existing = records
            .iter()
            .map(|record| record.record_id.clone())
            .collect::<std::collections::BTreeSet<_>>();
        if validate_simplify_record_refs(&draft, &existing).is_err() {
            tracing::debug!(
                model = model.model_name(),
                "simplify answer referenced unknown records, using the deterministic path"
            );
            return None;
        }
        Some(draft)
    }

    /// The deterministic maintenance pass described on [`Self::simplify`].
    #[must_use]
    pub fn simplify_offline(
        &self,
        subject: &EvolutionSubject,
        records: &[ExperienceRecord],
    ) -> SimplifyDraft {
        let actions = records
            .iter()
            .map(|record| {
                let score = record.score();
                let delete = score < SIMPLIFY_KEEP_SCORE && record.attempts() == 0;
                SimplifyAction {
                    action: if delete {
                        SimplifyActionKind::Delete
                    } else {
                        SimplifyActionKind::Keep
                    },
                    record_id: record.record_id.clone(),
                    reason: if delete {
                        format!("score {score:.2} is below {SIMPLIFY_KEEP_SCORE:.2} and the record was never used")
                    } else {
                        format!("score {score:.2} meets the maintenance keep rule")
                    },
                    new_content: None,
                    merge_remove_ids: Vec::new(),
                }
            })
            .collect();
        SimplifyDraft::new(subject.clone(), actions)
    }
}

/// Parses one model reply into a [`DraftParse`].
///
/// Tolerance is deliberate and bounded: one outer Markdown fence, `//` comments
/// outside `scheme://` URLs, trailing commas, then the outermost `[...]` (or
/// `{...}`) slice. Items with `action: "skip"` (or an unrecognised action) are
/// counted as skipped; the rest are normalized through
/// [`normalize_evolve_draft`].
///
/// # Errors
///
/// [`Error::validation`] when `text` holds no parseable JSON, when the JSON is
/// not an array/object, or when an appended item breaks the draft schema.
pub fn parse_draft_text(subject: &EvolutionSubject, text: &str) -> Result<DraftParse> {
    let items = json_items(extract_json_value(text)?, "experiences")
        .ok_or_else(|| Error::validation("model reply must be a JSON array of experiences"))?;
    let mut appended = Vec::new();
    let mut skipped = 0usize;
    for item in items {
        let Some(object) = item.as_object() else {
            skipped += 1;
            continue;
        };
        let action = object
            .get("action")
            .and_then(Value::as_str)
            .unwrap_or("append")
            .trim()
            .to_ascii_lowercase();
        if action.is_empty() || action == "append" {
            appended.push(item);
        } else {
            skipped += 1;
        }
    }
    if appended.is_empty() {
        return Ok(DraftParse {
            draft: EvolveDraft::new(subject.clone(), Vec::new()),
            appended: 0,
            skipped,
        });
    }
    let draft = normalize_evolve_draft(subject.clone(), &appended)?;
    Ok(DraftParse {
        draft,
        appended: appended.len(),
        skipped,
    })
}

/// Extracts the first JSON value from a model reply.
///
/// The steps mirror Python's `_extract_json_with_error`: parse as-is, parse the
/// repaired text (fence, comments, trailing commas), then parse the outermost
/// `[...]` and finally `{...}` slice of the repaired text.
///
/// # Errors
///
/// [`Error::validation`] when no step produces JSON.
pub fn extract_json_value(text: &str) -> Result<Value> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Err(Error::validation("model reply is empty"));
    }
    if let Some(value) = try_parse(trimmed) {
        return Ok(value);
    }
    let repaired = repair_json_text(trimmed);
    if let Some(value) = try_parse(&repaired) {
        return Ok(value);
    }
    for (open, close) in [('[', ']'), ('{', '}')] {
        let Some(slice) = outermost_slice(&repaired, open, close) else {
            continue;
        };
        if let Some(value) = try_parse(slice) {
            return Ok(value);
        }
        if let Some(value) = try_parse(&repair_json_text(slice)) {
            return Ok(value);
        }
    }
    Err(Error::validation(
        "model reply does not contain a parseable JSON array or object",
    ))
}

/// Projects an extracted JSON value onto its item list.
///
/// An array is used as-is; an object with an array under `key` uses that array;
/// any other object is a one-item list. Non-container values yield `None`.
fn json_items(value: Value, key: &str) -> Option<Vec<Value>> {
    match value {
        Value::Array(items) => Some(items),
        Value::Object(mut map) => match map.remove(key) {
            Some(Value::Array(items)) => Some(items),
            Some(other) => Some(vec![other]),
            None => Some(vec![Value::Object(map)]),
        },
        _ => None,
    }
}

/// Strips one outer Markdown fence, keeping fences inside JSON strings intact.
fn strip_outer_fence(text: &str) -> &str {
    let stripped = text.trim();
    if !stripped.starts_with("```") {
        return stripped;
    }
    let Some(newline) = stripped.find('\n') else {
        return stripped;
    };
    let language = stripped[..newline].trim().trim_start_matches("```").trim();
    if !language.is_empty() && !language.eq_ignore_ascii_case("json") {
        return stripped;
    }
    let inner = stripped[newline + 1..].trim_end();
    match inner.strip_suffix("```") {
        Some(closing) => closing.trim_end(),
        None => inner,
    }
}

/// Applies Python's tolerant repairs: one outer fence, `//` comments outside
/// URLs, and trailing commas.
fn repair_json_text(text: &str) -> String {
    let fenced = strip_outer_fence(text);
    let uncommented = drop_line_comments(fenced);
    drop_trailing_commas(&uncommented).trim().to_owned()
}

/// Drops whole-line `//` comments and comments directly after `,`, `[` or `{`.
fn drop_line_comments(text: &str) -> String {
    let mut kept = String::with_capacity(text.len());
    for (index, line) in text.split('\n').enumerate() {
        if index > 0 {
            kept.push('\n');
        }
        kept.push_str(strip_line_comment(line));
    }
    kept
}

/// Returns `line` without a trailing `//` comment, if it has one.
fn strip_line_comment(line: &str) -> &str {
    let mut search_from = 0usize;
    while let Some(relative) = line[search_from..].find("//") {
        let at = search_from + relative;
        let prefix = line[..at].trim_end();
        if prefix.is_empty()
            || prefix.ends_with(',')
            || prefix.ends_with('[')
            || prefix.ends_with('{')
        {
            return &line[..at];
        }
        search_from = at + 2;
    }
    line
}

/// Drops commas that directly precede `]` or `}`, ignoring commas in strings.
fn drop_trailing_commas(text: &str) -> String {
    let characters: Vec<char> = text.chars().collect();
    let mut output = String::with_capacity(text.len());
    let mut in_string = false;
    let mut escaped = false;
    let mut index = 0usize;
    while index < characters.len() {
        let character = characters[index];
        if in_string {
            output.push(character);
            if escaped {
                escaped = false;
            } else if character == '\\' {
                escaped = true;
            } else if character == '"' {
                in_string = false;
            }
            index += 1;
            continue;
        }
        if character == '"' {
            in_string = true;
            output.push(character);
            index += 1;
            continue;
        }
        if character == ',' {
            let mut lookahead = index + 1;
            while lookahead < characters.len() && characters[lookahead].is_whitespace() {
                lookahead += 1;
            }
            if lookahead < characters.len() && matches!(characters[lookahead], ']' | '}') {
                index += 1;
                continue;
            }
        }
        output.push(character);
        index += 1;
    }
    output
}

/// Returns the first `open` through the last `close` in `text`.
fn outermost_slice(text: &str, open: char, close: char) -> Option<&str> {
    let start = text.find(open)?;
    let end = text.rfind(close)?;
    (end > start).then(|| &text[start..=end])
}

/// Parses `text` as JSON, returning `None` on any syntax error.
fn try_parse(text: &str) -> Option<Value> {
    serde_json::from_str(text).ok()
}

/// Collapses runs of whitespace and trims.
fn collapse_whitespace(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Projects an experience section onto its draft section.
fn draft_section(section: ExperienceSection) -> DraftSection {
    match section {
        ExperienceSection::Description => DraftSection::Instructions,
        ExperienceSection::Script => DraftSection::Scripts,
        ExperienceSection::Body
        | ExperienceSection::Tool
        | ExperienceSection::Memory
        | ExperienceSection::Other => DraftSection::Troubleshooting,
    }
}

/// Renders the signals block of [`DRAFT_PROMPT`].
fn render_signals(signals: &[EvolutionSignal]) -> String {
    if signals.is_empty() {
        return "None".to_owned();
    }
    signals
        .iter()
        .take(MAX_PROMPT_ITEMS)
        .map(|signal| {
            let excerpt =
                truncate_excerpt(&collapse_whitespace(&signal.excerpt), PROMPT_SIGNAL_CHARS);
            format!(
                "- [{}] {} (confidence {:.2}, weight {:.2}, target {})",
                signal.signal_type,
                excerpt,
                signal.confidence,
                signal.weight,
                signal.target().as_str()
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Renders the trajectory block of [`DRAFT_PROMPT`].
fn render_trajectories(trajectories: &[Trajectory]) -> String {
    if trajectories.is_empty() {
        return "None".to_owned();
    }
    trajectories
        .iter()
        .take(MAX_PROMPT_ITEMS)
        .map(|trajectory| {
            format!(
                "- {} ({:?}): {}",
                trajectory.trajectory_id,
                trajectory.outcome,
                render_digest(trajectory, PROMPT_DIGEST_CHARS)
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Renders one record line for [`SIMPLIFY_PROMPT`].
fn render_record_line(record: &ExperienceRecord) -> String {
    format!(
        "- [{}] [{}] score={:.2} hits={} successes={} failures={} content={}",
        record.record_id,
        record.section.as_str(),
        record.score(),
        record.attempts(),
        record.successes,
        record.failures,
        truncate_excerpt(&collapse_whitespace(&record.content), PROMPT_SIGNAL_CHARS)
    )
}

/// Truncates `raw` at `max_chars` and appends an explicit truncation marker.
///
/// A `max_chars` of `0` keeps no content but still reports the original length,
/// so the prompt never silently loses the budget marker.
#[must_use]
pub fn summarize_skill_content(raw: &str, max_chars: usize) -> String {
    if raw.chars().count() <= max_chars {
        return raw.to_owned();
    }
    let kept = truncate_excerpt(raw, max_chars);
    format!(
        "{kept}\n... [truncated, the original SKILL.md is {} characters]",
        raw.chars().count()
    )
}

/// Builds the skill's existing-experience summary lines for a [`DraftInput`].
///
/// The `summary` metadata wins over a content excerpt, so a prompt shows the
/// one-line lesson the archive stored.
#[must_use]
pub fn existing_summaries<'a>(
    records: impl IntoIterator<Item = &'a ExperienceRecord>,
) -> Vec<String> {
    records
        .into_iter()
        .take(MAX_PROMPT_ITEMS)
        .map(|record| {
            let summary = record
                .metadata
                .get("summary")
                .and_then(Value::as_str)
                .map_or_else(
                    || truncate_excerpt(&collapse_whitespace(&record.content), 120),
                    str::to_owned,
                );
            format!(
                "[{}] [{}] {summary}",
                record.record_id,
                record.section.as_str()
            )
        })
        .collect()
}

/// Builds the keywords a persisted deterministic draft carries.
#[must_use]
pub fn draft_keywords(content: &str) -> Vec<String> {
    tokens(content)
        .into_iter()
        .take(DRAFT_KEYWORD_LIMIT)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signal::{EvolutionSignal, EvolutionTarget as SignalTarget};

    fn signal(excerpt: &str) -> EvolutionSignal {
        EvolutionSignal::new("execution_failure", SignalTarget::Body, excerpt)
            .expect("valid signal")
    }

    fn subject() -> EvolutionSubject {
        EvolutionSubject::for_skill("pdf-report").expect("valid subject")
    }

    fn input(signals: &[EvolutionSignal]) -> DraftInput<'_> {
        DraftInput::new("pdf-report", "# PDF report\n\nRead and summarise.", signals)
    }

    #[test]
    fn fenced_commented_and_trailing_comma_json_is_tolerated() {
        let text = "```json\n[\n  {\"summary\": \"s\", \"content\": \"c\", // why\n   \"source_refs\": [\"https://example.test/run\"],},\n]\n```";
        let parse = parse_draft_text(&subject(), text).expect("tolerant parse");
        assert_eq!(parse.appended, 1);
        assert_eq!(parse.skipped, 0);
        assert_eq!(parse.draft.experiences[0].summary, "s");
        assert_eq!(
            parse.draft.experiences[0].source_refs,
            vec!["https://example.test/run"]
        );
    }

    #[test]
    fn extraction_handles_prose_and_objects() {
        let prose =
            "Here you go: [{\"action\": \"append\", \"summary\": \"s\", \"content\": \"c\"}] thanks";
        let value = extract_json_value(prose).expect("outermost array");
        assert!(value.is_array());
        let object = extract_json_value("{\"experiences\": []}").expect("object");
        assert_eq!(json_items(object, "experiences").expect("items").len(), 0);
        assert!(extract_json_value("   ").is_err());
        assert!(extract_json_value("no json here").is_err());
        let comments = "// leading\n[{\"summary\":\"s\",\"content\":\"c\"}]";
        assert!(extract_json_value(comments).is_ok());
    }

    #[test]
    fn skip_items_are_counted_and_not_normalized() {
        let text = "[{\"action\": \"skip\", \"skip_reason\": \"irrelevant\"}, {\"action\": \"append\", \"summary\": \"s\", \"content\": \"c\"}]";
        let parse = parse_draft_text(&subject(), text).expect("valid parse");
        assert_eq!(parse.appended, 1);
        assert_eq!(parse.skipped, 1);
        let all_skip = parse_draft_text(&subject(), "[{\"action\":\"skip\"}]").expect("skip only");
        assert!(all_skip.draft.is_empty());
        assert_eq!(all_skip.skipped, 1);
        assert!(parse_draft_text(&subject(), "[{\"action\":\"append\"}]").is_err());
    }

    #[test]
    fn skill_content_truncation_is_marked() {
        let long = "x".repeat(DEFAULT_MAX_SKILL_CHARS + 10);
        let summary = summarize_skill_content(&long, DEFAULT_MAX_SKILL_CHARS);
        assert!(summary.contains("[truncated, the original SKILL.md is 6010 characters]"));
        assert_eq!(summarize_skill_content("short", 10), "short");
    }

    #[test]
    fn offline_draft_truncates_summaries_and_caps_items() {
        let signals: Vec<EvolutionSignal> = (0..5)
            .map(|index| signal(&format!("lesson number {index} with a long tail")))
            .collect();
        let drafter = ExperienceDrafter::new()
            .with_max_summary_chars(10)
            .with_max_items(2);
        let draft = drafter.draft_offline(&subject(), &input(&signals));
        assert_eq!(draft.experiences.len(), 2);
        for experience in &draft.experiences {
            assert_eq!(experience.summary.chars().count(), 10);
            assert_eq!(
                experience.reason,
                "derived from signal evidence with confidence 1.00"
            );
            assert!(experience.source_refs[0].starts_with("sig-"));
        }
    }

    #[test]
    fn prompt_builders_fill_every_placeholder() {
        let signals = [signal("the parser timed out")];
        let drafter = ExperienceDrafter::new().with_max_skill_chars(20);
        let prompt = drafter.build_draft_prompt(
            &input(&signals)
                .with_user_query("tighten the parser")
                .with_existing(vec!["[ev-1] [Body] old lesson".to_owned()]),
        );
        assert!(!prompt.contains("{skill_name}"));
        assert!(prompt.contains("tighten the parser"));
        assert!(!prompt.contains("No existing records"));
        assert!(prompt.contains(&format!(
            "[truncated, the original SKILL.md is {} characters]",
            "# PDF report\n\nRead and summarise.".chars().count()
        )));
        let repair = drafter.build_repair_prompt("bad json", "[]");
        assert!(repair.contains("## Parse error\nbad json"));
    }
}
