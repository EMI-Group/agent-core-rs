//! Plan stage: decide *what* to extract, from *which* turns, and *why*.
//!
//! The Python `memory_analysis_prompt` asks the model four questions in one shot — is
//! there key information, which variables hold values, what is the summary, and (in
//! `fragment_memory_prompt`) which of the six execution steps apply. This module turns
//! those questions into an explicit [`ExtractionPlan`]: an intent, a topic, the ordered
//! [`PlanOperation`]s, the turn indices worth extracting from, and a short rationale.
//!
//! [`RulePlanStage`] derives the plan deterministically (marker rules plus the shared
//! [`crate::analyzer::rule`] regexes), so planning works offline; `LlmPlanStage` asks a
//! model for the same JSON shape and falls back to the rule plan with a recorded
//! degradation. Nothing here touches storage: the plan only steers later stages.
use crate::analyzer::config::AnalyzerConfig;
use crate::analyzer::fact::{truncate_chars, ConversationTurn, ExtractionInput};
use crate::analyzer::parse::payload;
use crate::analyzer::rule::{compile, matches, parse_assignment};
use crate::analyzer::stage::{ExtractionStage, StageInput, StageOutput};
use crate::recall::keyword::tokenize;
use crate::record::CallOptions;
use async_trait::async_trait;
use openjiuwen_common::error::{ErrorKind, Result};
use openjiuwen_foundation::llm::{Model, ModelRequestConfig};
use openjiuwen_foundation::prompt::BaseMessage;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::{Arc, LazyLock};

/// A correction marker ("actually ...", "no longer ...", "changed to ...").
static CORRECTION: LazyLock<Option<Regex>> = LazyLock::new(|| {
    compile(
        r"(?i)(?:\b(?:actually|no\s+longer|instead|changed\s+to|switched\s+to|correction|rather\s+than)\b|updated\s*:|\bnot\b[^.;。！？]{0,60}\banymore\b)",
    )
});
/// An explicit capture directive ("remember ...", "keep in mind ...").
static DIRECTIVE: LazyLock<Option<Regex>> = LazyLock::new(|| {
    compile(
        r"(?i)\b(?:remember|note\s+that|keep\s+in\s+mind|don'?t\s+forget|memorize|memorise|make\s+a\s+note)\b",
    )
});
/// An edit instruction that has to be resolved against stored records.
static UPDATE: LazyLock<Option<Regex>> = LazyLock::new(|| {
    compile(
        r"(?i)(?:\b(?:update|delete|remove|forget|change|replace|rename)\b|改成|修改|删除|更新|换成|去掉|清除|移除)",
    )
});
/// Third-person and demonstrative pronouns the anaphora step resolves.
static PRONOUN: LazyLock<Option<Regex>> = LazyLock::new(|| {
    compile(r"(?i)(?:\b(?:he|she|it|they|them|him|her|this|that|these|those)\b|他|她|它|这|那)")
});
/// Relative time expressions the time-reasoning step normalizes.
static RELATIVE_TIME: LazyLock<Option<Regex>> = LazyLock::new(|| {
    compile(
        r"(?i)(?:\b(?:today|tomorrow|yesterday|tonight|this\s+(?:morning|afternoon|evening|week|month)|next\s+(?:week|month|year)|last\s+(?:week|month|year|night)|(?:a|two|three|four|five|\d+)\s+(?:day|days|week|weeks|month|months)\s+ago|in\s+(?:a|two|three|four|five|\d+)\s+(?:day|days|week|weeks|month|months))\b|明天|昨天|后天|前天|今天|下周|上周|下个月|上个月|明年|去年)",
    )
});
/// A stated preference.
static PREFERENCE: LazyLock<Option<Regex>> = LazyLock::new(|| {
    compile(
        r"(?i)\b(?:i|we)\s+(?:(?:really|also|always|never|just|do)\s+)*(?:prefer|like|love|enjoy|hate|dislike|avoid|want|need)\b",
    )
});
/// A user-identity statement.
static IDENTITY: LazyLock<Option<Regex>> = LazyLock::new(|| {
    compile(
        r"(?i)(?:\b(?:my|our)\s+(?:name|age|e-?mail|job|role|title|company|employer|team|phone|address|city|country|timezone|birthday|username|location|language|favourite|favorite)\b|\bi(?:'m| am)\s+(?:a|an)\b|\bi\s+(?:work|live)\s+(?:at|for|in)\b)",
    )
});
/// A project/technology statement.
static PROJECT: LazyLock<Option<Regex>> = LazyLock::new(|| {
    compile(
        r"(?i)(?:\b(?:the|our|this|my)\s+(?:project|repo|repository|service|app|application|codebase|library|package|system|backend|frontend|pipeline|stack|tool|api)\s+(?:uses|runs\s+on|depends\s+on|is\s+built\s+with|requires|targets)\b|\b(?:we|i)\s+use\b|\bi(?:'m| am)\s+(?:building|working\s+on|migrating|refactoring)\b)",
    )
});

/// Returns `true` when `text` carries a correction marker.
pub(crate) fn has_correction_marker(text: &str) -> bool {
    matches(CORRECTION.as_ref(), text)
}

/// Returns `true` when `text` carries an explicit capture directive.
pub(crate) fn has_directive_marker(text: &str) -> bool {
    matches(DIRECTIVE.as_ref(), text)
}

/// Returns `true` when `text` carries a stated preference.
pub(crate) fn has_preference_marker(text: &str) -> bool {
    matches(PREFERENCE.as_ref(), text)
}

/// Tokens ignored when deriving a plan topic.
const TOPIC_STOPWORDS: &[&str] = &[
    "this", "that", "these", "those", "with", "from", "have", "has", "will", "would", "should",
    "could", "there", "here", "when", "what", "where", "which", "while", "your", "their", "them",
    "they", "then", "than", "just", "also", "very", "into", "about", "after", "before", "because",
    "please", "remember", "really", "thing", "things", "want", "need", "like", "work",
];

/// Token count from which the plan asks for a conversation summary.
pub const SUMMARY_TOKEN_THRESHOLD: usize = 64;

/// The dominant intent the plan detected in the conversation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanIntent {
    /// A correction/edit of stored memory ("actually ...", "changed to ...").
    Correction,
    /// An explicit capture directive ("remember ...").
    Directive,
    /// A variable assignment (`key = value`, `set key to value`).
    Variable,
    /// A stated preference.
    Preference,
    /// A user-identity statement.
    Identity,
    /// A project/technology statement.
    Project,
    /// A plain declarative statement with no stronger signal.
    General,
    /// Nothing to extract: every turn was blank, too short, or extraction is disabled.
    ///
    /// Deserializing an unknown intent name also lands here, so a stale or misspelled
    /// plan never fails to replay.
    #[default]
    #[serde(other)]
    None,
}

impl PlanIntent {
    /// Returns the stable `snake_case` name used in JSON and rationales.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Correction => "correction",
            Self::Directive => "directive",
            Self::Variable => "variable",
            Self::Preference => "preference",
            Self::Identity => "identity",
            Self::Project => "project",
            Self::General => "general",
        }
    }

    /// Returns `true` when the plan has something to extract.
    #[must_use]
    pub const fn is_actionable(self) -> bool {
        !matches!(self, Self::None)
    }

    /// Tolerant parser for a model answer's intent field.
    #[must_use]
    pub fn parse(raw: &str) -> Option<Self> {
        let normalized = raw.trim().to_ascii_lowercase().replace([' ', '-'], "_");
        Some(match normalized.as_str() {
            "none" | "" => Self::None,
            "correction" | "correct" | "update" | "delete" | "edit" | "conflict" => {
                Self::Correction
            }
            "directive" | "explicit" | "instruct" | "instruction" => Self::Directive,
            "variable" | "variables" => Self::Variable,
            "preference" | "preferences" | "user_preference" => Self::Preference,
            "identity" | "user_profile" | "profile" => Self::Identity,
            "project" | "technology" | "semantic_memory" | "tech_stack" => Self::Project,
            "general" | "conversation" | "episodic_memory" | "other" => Self::General,
            _ => return None,
        })
    }
}

/// One operation the plan asks the pipeline to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanOperation {
    /// Resolve pronouns against the history block before extracting.
    ResolveReferences,
    /// Normalize relative time expressions to absolute dates.
    NormalizeTime,
    /// Extract memory facts from the focused turns.
    ExtractFacts,
    /// Extract variable assignments.
    ExtractVariables,
    /// Produce a short conversation summary.
    ExtractSummary,
    /// Resolve update/delete instructions against stored records.
    HandleDirectives,
}

impl PlanOperation {
    /// Returns the stable `snake_case` name used in JSON and rationales.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ResolveReferences => "resolve_references",
            Self::NormalizeTime => "normalize_time",
            Self::ExtractFacts => "extract_facts",
            Self::ExtractVariables => "extract_variables",
            Self::ExtractSummary => "extract_summary",
            Self::HandleDirectives => "handle_directives",
        }
    }

    /// Tolerant parser for a model answer's operation entries.
    #[must_use]
    pub fn parse(raw: &str) -> Option<Self> {
        let normalized = raw.trim().to_ascii_lowercase().replace([' ', '-'], "_");
        Some(match normalized.as_str() {
            "resolve_references" | "resolve_reference" | "anaphora" | "coreference" => {
                Self::ResolveReferences
            }
            "normalize_time" | "time_reasoning" | "time_normalization" | "normalize_dates" => {
                Self::NormalizeTime
            }
            "extract_facts" | "extract" | "extract_fragment_memory" | "facts" => Self::ExtractFacts,
            "extract_variables" | "variables" | "variable_extraction" => Self::ExtractVariables,
            "extract_summary" | "summary" | "summarize" => Self::ExtractSummary,
            "handle_directives" | "directives" | "handle_updates" | "memory_operations" => {
                Self::HandleDirectives
            }
            _ => return None,
        })
    }
}

/// The plan a plan stage produced.
///
/// `intent` and `focus_turns` describe *what* and *where*; `operations` is the ordered
/// list of steps the pipeline should run; `rationale` records the reasoning so a caller
/// (or a log) can see why the plan looks the way it does. A plan with
/// [`PlanIntent::None`] and no operations means "nothing to extract".
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ExtractionPlan {
    /// The dominant intent.
    pub intent: PlanIntent,
    /// A short content topic (the most frequent salient token), when one exists.
    pub topic: Option<String>,
    /// The ordered operations to run.
    pub operations: Vec<PlanOperation>,
    /// Indices (into `ExtractionInput::turns`) of the turns worth extracting from.
    pub focus_turns: Vec<usize>,
    /// Why this plan was chosen.
    pub rationale: String,
    /// Why a deterministic fallback produced this plan, when one did.
    pub degraded: Option<String>,
}

impl ExtractionPlan {
    /// Builds the empty, non-actionable plan with `rationale` as its explanation.
    #[must_use]
    pub fn empty(rationale: impl Into<String>) -> Self {
        Self {
            rationale: rationale.into(),
            ..Self::default()
        }
    }

    /// Returns `true` when the plan asks for fact extraction.
    #[must_use]
    pub fn extracts_facts(&self) -> bool {
        self.operations.contains(&PlanOperation::ExtractFacts)
    }

    /// Returns `true` when the plan includes `operation`.
    #[must_use]
    pub fn has(&self, operation: PlanOperation) -> bool {
        self.operations.contains(&operation)
    }

    /// Returns `true` when the plan asks for at least one operation.
    #[must_use]
    pub fn is_actionable(&self) -> bool {
        !self.operations.is_empty()
    }

    /// Returns the turns the plan selected, falling back to the whole conversation.
    ///
    /// Out-of-range indices are ignored; an empty selection means "all turns", which
    /// keeps a hand-written or model-supplied plan from silently extracting nothing.
    #[must_use]
    pub fn selected_turns(&self, conversation: &ExtractionInput) -> Vec<usize> {
        if self.focus_turns.is_empty() {
            return (0..conversation.turns.len()).collect();
        }
        let mut selected: Vec<usize> = self
            .focus_turns
            .iter()
            .copied()
            .filter(|index| *index < conversation.turns.len())
            .collect();
        selected.sort_unstable();
        selected.dedup();
        selected
    }
}

/// Deterministic plan stage.
pub struct RulePlanStage {
    /// Effective analyzer settings (limits and the enable flag).
    config: AnalyzerConfig,
}

impl RulePlanStage {
    /// Builds the rule plan stage with `config`.
    #[must_use]
    pub fn new(config: AnalyzerConfig) -> Self {
        Self { config }
    }

    /// Returns the configuration in use.
    #[must_use]
    pub fn config(&self) -> &AnalyzerConfig {
        &self.config
    }

    /// Derives the plan for `input`.
    fn build_plan(&self, input: &StageInput) -> ExtractionPlan {
        if !self.config.enabled {
            return ExtractionPlan::empty("memory extraction is disabled");
        }
        let turns = &input.conversation.turns;
        let focus_turns: Vec<usize> = turns
            .iter()
            .enumerate()
            .filter(|(_, turn)| {
                !turn.is_blank() && turn.text.trim().chars().count() >= self.config.min_message_len
            })
            .map(|(index, _)| index)
            .collect();
        if focus_turns.is_empty() {
            return ExtractionPlan::empty(format!(
                "no turn reaches the {}-character extraction minimum",
                self.config.min_message_len
            ));
        }
        let probe = focus_turns
            .iter()
            .filter_map(|index| turns.get(*index))
            .map(|turn| turn.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        let intent = Self::intent(&probe, turns);
        if !intent.is_actionable() {
            return ExtractionPlan::empty("no memory signal was detected");
        }
        let mut operations = Vec::new();
        if !input.conversation.history.is_empty() && matches(PRONOUN.as_ref(), &probe) {
            operations.push(PlanOperation::ResolveReferences);
        }
        if matches(RELATIVE_TIME.as_ref(), &probe) {
            operations.push(PlanOperation::NormalizeTime);
        }
        operations.push(PlanOperation::ExtractFacts);
        if Self::has_assignment(turns) {
            operations.push(PlanOperation::ExtractVariables);
        }
        if tokenize(&probe).len() >= SUMMARY_TOKEN_THRESHOLD {
            operations.push(PlanOperation::ExtractSummary);
        }
        if matches(CORRECTION.as_ref(), &probe) || matches(UPDATE.as_ref(), &probe) {
            operations.push(PlanOperation::HandleDirectives);
        }
        let topic = Self::topic(&probe, intent);
        let rationale = Self::rationale(
            intent,
            topic.as_deref(),
            &operations,
            focus_turns.len(),
            turns.len(),
        );
        ExtractionPlan {
            intent,
            topic,
            operations,
            focus_turns,
            rationale,
            degraded: None,
        }
    }

    /// Returns the dominant intent of `probe` (first match wins, correction first).
    fn intent(probe: &str, turns: &[ConversationTurn]) -> PlanIntent {
        if matches(CORRECTION.as_ref(), probe) {
            return PlanIntent::Correction;
        }
        if matches(DIRECTIVE.as_ref(), probe) {
            return PlanIntent::Directive;
        }
        if Self::has_assignment(turns) {
            return PlanIntent::Variable;
        }
        if matches(PREFERENCE.as_ref(), probe) {
            return PlanIntent::Preference;
        }
        if matches(IDENTITY.as_ref(), probe) {
            return PlanIntent::Identity;
        }
        if matches(PROJECT.as_ref(), probe) {
            return PlanIntent::Project;
        }
        PlanIntent::General
    }

    /// Returns `true` when any focused line is a variable assignment.
    fn has_assignment(turns: &[ConversationTurn]) -> bool {
        turns.iter().any(|turn| {
            turn.text
                .lines()
                .any(|line| parse_assignment(line.trim()).is_some())
        })
    }

    /// Derives the plan topic: the most frequent salient token, else the intent label.
    fn topic(probe: &str, intent: PlanIntent) -> Option<String> {
        let mut counts: HashMap<String, (usize, usize)> = HashMap::new();
        for (position, token) in tokenize(probe).into_iter().enumerate() {
            if token.chars().count() < 4 || TOPIC_STOPWORDS.contains(&token.as_str()) {
                continue;
            }
            let entry = counts.entry(token).or_insert((0, position));
            entry.0 += 1;
        }
        let best = counts
            .into_iter()
            .max_by(
                |(left_token, (left_count, left_first)),
                 (right_token, (right_count, right_first))| {
                    left_count
                        .cmp(right_count)
                        .then(right_first.cmp(left_first))
                        .then(right_token.cmp(left_token))
                },
            )
            .map(|(token, _)| token);
        best.or_else(|| match intent {
            PlanIntent::Correction
            | PlanIntent::Directive
            | PlanIntent::Variable
            | PlanIntent::Preference
            | PlanIntent::Identity
            | PlanIntent::Project => Some(intent.as_str().to_owned()),
            PlanIntent::General | PlanIntent::None => None,
        })
    }

    /// Renders the deterministic plan rationale.
    fn rationale(
        intent: PlanIntent,
        topic: Option<&str>,
        operations: &[PlanOperation],
        focused: usize,
        total: usize,
    ) -> String {
        let mut rationale = format!(
            "{} intent over {focused} of {total} turn(s)",
            intent.as_str()
        );
        if let Some(topic) = topic {
            rationale.push_str(", topic ");
            rationale.push_str(topic);
        }
        let operations: Vec<&str> = operations
            .iter()
            .map(|operation| operation.as_str())
            .collect();
        rationale.push_str("; operations: ");
        rationale.push_str(&operations.join(", "));
        rationale
    }
}

#[async_trait]
impl ExtractionStage for RulePlanStage {
    #[allow(clippy::unnecessary_literal_bound)] // the trait signature ties `&str` to `self`
    fn name(&self) -> &str {
        "plan"
    }

    async fn run(&self, input: &StageInput, options: &CallOptions) -> Result<StageOutput> {
        options.check()?;
        self.config.validate()?;
        input.conversation.validate()?;
        Ok(StageOutput::of(self.name()).with_plan(self.build_plan(input)))
    }
}

/// The JSON schema a plan answer must follow.
pub const PLAN_JSON_SCHEMA: &str = concat!(
    "{\"intent\":\"preference|identity|project|directive|correction|variable|general|none\",",
    "\"topic\":\"...\",",
    "\"operations\":[\"resolve_references|normalize_time|extract_facts|extract_variables|",
    "extract_summary|handle_directives\"],",
    "\"focus_turns\":[0],\"rationale\":\"...\"}"
);

/// System prompt for [`LlmPlanStage`].
pub const PLAN_SYSTEM_PROMPT: &str = concat!(
    "You plan how to extract long-term memory from a conversation. ",
    "Reply with JSON only, using exactly this shape and no other keys: ",
    "{\"intent\":\"preference|identity|project|directive|correction|variable|general|none\",",
    "\"topic\":\"...\",",
    "\"operations\":[\"resolve_references|normalize_time|extract_facts|extract_variables|",
    "extract_summary|handle_directives\"],",
    "\"focus_turns\":[0],\"rationale\":\"...\"}. ",
    "`intent` is the dominant memory signal: correction when the user edits/negates ",
    "stored memory, directive when they explicitly ask to remember something, variable ",
    "for assignments, else preference/identity/project, else general, and none when the ",
    "conversation holds nothing worth remembering. `operations` lists the steps to run: ",
    "resolve_references when pronouns need the history block, normalize_time when ",
    "relative dates appear, extract_facts for the facts themselves, extract_variables ",
    "for assignments, extract_summary for a long conversation, handle_directives when ",
    "stored memory must be updated or deleted. `focus_turns` holds the 0-based indices ",
    "of the turns worth extracting from, `topic` is a few words, and `rationale` is one ",
    "sentence explaining the plan. Do not add prose, Markdown fences, or explanations."
);

/// Model-backed plan stage with the rule plan as its fallback.
pub struct LlmPlanStage {
    /// The model that answers the planning prompt.
    model: Arc<dyn Model>,
    /// Analyzer settings (limits, retries, prompt bounding).
    config: AnalyzerConfig,
    /// Deterministic plan stage used once the retry budget is spent.
    fallback: RulePlanStage,
}

impl LlmPlanStage {
    /// Builds the stage around `model`.
    #[must_use]
    pub fn new(model: Arc<dyn Model>, config: AnalyzerConfig) -> Self {
        let fallback = RulePlanStage::new(config.clone());
        Self {
            model,
            config,
            fallback,
        }
    }

    /// Returns the stage with `fallback` as its deterministic backstop.
    #[must_use]
    pub fn with_fallback(mut self, fallback: RulePlanStage) -> Self {
        self.fallback = fallback;
        self
    }

    /// Returns the configuration in use.
    #[must_use]
    pub fn config(&self) -> &AnalyzerConfig {
        &self.config
    }

    /// Renders the planning prompt body.
    fn render(&self, input: &StageInput) -> String {
        let conversation = crate::analyzer::prompt::render_conversation(
            &input.conversation.bounded(self.config.max_input_chars),
            self.config.max_input_chars,
        );
        truncate_chars(
            &format!(
                "TURNS\n{}\n\nCONVERSATION\n{conversation}",
                input.conversation.turns.len()
            ),
            self.config.max_input_chars,
        )
    }
}

#[async_trait]
impl ExtractionStage for LlmPlanStage {
    #[allow(clippy::unnecessary_literal_bound)] // the trait signature ties `&str` to `self`
    fn name(&self) -> &str {
        "plan"
    }

    async fn run(&self, input: &StageInput, options: &CallOptions) -> Result<StageOutput> {
        options.check()?;
        self.config.validate()?;
        input.conversation.validate()?;
        if !self.config.enabled {
            return Ok(StageOutput::of(self.name())
                .with_plan(ExtractionPlan::empty("memory extraction is disabled")));
        }
        let messages = vec![
            BaseMessage::system(PLAN_SYSTEM_PROMPT),
            BaseMessage::user(self.render(input)),
        ];
        let request = ModelRequestConfig::new(self.model.model_name().to_owned());
        let attempts = self.config.model_retries.saturating_add(1);
        let mut failure: Option<String> = None;
        for attempt in 1..=attempts {
            match options.run(self.model.invoke(&messages, &request)).await {
                Ok(Ok(response)) => {
                    let raw = response
                        .message()
                        .map(|message| message.text().into_owned())
                        .unwrap_or_default();
                    if let Some(plan) =
                        parse_plan_response(&raw, &self.config, input.conversation.turns.len())
                    {
                        return Ok(StageOutput::of(self.name()).with_plan(plan));
                    }
                    failure
                        .get_or_insert_with(|| "the model answer held no usable plan".to_owned());
                    tracing::warn!(
                        target: "openjiuwen_memory::analyzer",
                        attempt,
                        attempts,
                        "LLM plan stage returned unusable output; retrying"
                    );
                }
                Ok(Err(error)) | Err(error) => {
                    if error.kind() == ErrorKind::Cancelled {
                        return Err(error);
                    }
                    failure.get_or_insert_with(|| format!("the model call failed: {error}"));
                    tracing::warn!(
                        target: "openjiuwen_memory::analyzer",
                        attempt,
                        attempts,
                        %error,
                        "LLM plan stage call failed; retrying"
                    );
                }
            }
            options.check()?;
        }
        let failure = failure.unwrap_or_else(|| "no attempt was made".to_owned());
        tracing::warn!(
            target: "openjiuwen_memory::analyzer",
            attempts,
            %failure,
            "LLM plan stage exhausted its retries; falling back to rules"
        );
        let mut plan = self.fallback.build_plan(input);
        plan.degraded = Some(format!("llm plan fell back to rules: {failure}"));
        Ok(StageOutput::of(self.name())
            .with_plan(plan.clone())
            .degraded(plan.degraded.unwrap_or_default()))
    }
}

/// Parses a plan answer into an [`ExtractionPlan`].
///
/// Accepted shapes: an object holding `intent`, `topic`, `operations`, `focus_turns`,
/// and `rationale` (aliases: `plan`/`steps` for `operations`, `turns`/`indices` for
/// `focus_turns`, `reason`/`why` for `rationale`). Unknown intents become
/// [`PlanIntent::General`], unknown operations are skipped with a `tracing::warn!`,
/// out-of-range turn indices are dropped, and duplicate operations keep their first
/// position. The result is `None` when no operation is usable, so a plan the pipeline
/// cannot execute always degrades to the rule plan.
#[must_use]
pub fn parse_plan_response(
    raw: &str,
    config: &AnalyzerConfig,
    turn_count: usize,
) -> Option<ExtractionPlan> {
    let value = payload(raw)?;
    let map = match &value {
        Value::Object(map) => map,
        _ => return None,
    };
    let intent = map
        .get("intent")
        .or_else(|| map.get("plan_intent"))
        .and_then(Value::as_str)
        .and_then(PlanIntent::parse)
        .unwrap_or(PlanIntent::General);
    let operations = parse_operations(map.get("operations").or_else(|| map.get("steps")))?;
    let mut focus_turns: Vec<usize> = map
        .get("focus_turns")
        .or_else(|| map.get("turns"))
        .or_else(|| map.get("indices"))
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_u64)
                .filter_map(|index| usize::try_from(index).ok())
                .filter(|index| *index < turn_count)
                .collect()
        })
        .unwrap_or_default();
    focus_turns.sort_unstable();
    focus_turns.dedup();
    let rationale = match map
        .get("rationale")
        .or_else(|| map.get("reason"))
        .or_else(|| map.get("why"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|text| !text.is_empty())
    {
        Some(text) => truncate_chars(text, config.max_content_chars),
        None => format!("model plan: {} intent", intent.as_str()),
    };
    let topic = map
        .get("topic")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|topic| !topic.is_empty())
        .map(|topic| truncate_chars(topic, config.max_content_chars));
    Some(ExtractionPlan {
        intent,
        topic,
        operations,
        focus_turns,
        rationale,
        degraded: None,
    })
}

/// Parses and normalizes the operations array of a plan answer.
fn parse_operations(value: Option<&Value>) -> Option<Vec<PlanOperation>> {
    let items = match value? {
        Value::Array(items) => items.clone(),
        Value::String(raw) => raw
            .split(',')
            .map(|item| Value::String(item.trim().to_owned()))
            .collect(),
        _ => return None,
    };
    let mut operations = Vec::new();
    for item in items {
        let raw = match &item {
            Value::String(raw) => raw.as_str(),
            Value::Object(map) => map
                .get("operation")
                .or_else(|| map.get("name"))
                .or_else(|| map.get("step"))
                .and_then(Value::as_str)
                .unwrap_or_default(),
            _ => continue,
        };
        if let Some(operation) = PlanOperation::parse(raw) {
            if !operations.contains(&operation) {
                operations.push(operation);
            }
        } else {
            tracing::warn!(
                target: "openjiuwen_memory::analyzer",
                %raw,
                "skipping unknown plan operation"
            );
        }
    }
    if operations.is_empty() {
        None
    } else {
        Some(operations)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analyzer::stage::StagedResult;
    use openjiuwen_common::error::Error;
    use openjiuwen_test_support::ScriptedModel;
    use tokio_util::sync::CancellationToken;

    fn conversation(turns: &[&str]) -> ExtractionInput {
        let mut input = ExtractionInput::default();
        for turn in turns {
            input.push_user(*turn);
        }
        input
    }

    async fn plan_of(turns: &[&str]) -> ExtractionPlan {
        plan_with_history(turns, &[]).await
    }

    async fn plan_with_history(turns: &[&str], history: &[&str]) -> ExtractionPlan {
        let mut input = conversation(turns);
        for turn in history {
            input.history.push(ConversationTurn::new("user", *turn));
        }
        let config = AnalyzerConfig::default();
        let stage = RulePlanStage::new(config.clone());
        stage
            .run(&StageInput::new(input, config), &CallOptions::new())
            .await
            .unwrap()
            .plan
            .unwrap()
    }

    #[tokio::test]
    async fn rule_plan_selects_operations_across_transcripts() {
        let plan = plan_of(&["I prefer dark mode."]).await;
        assert_eq!(plan.intent, PlanIntent::Preference);
        assert!(plan.extracts_facts());
        assert!(!plan.has(PlanOperation::HandleDirectives));
        assert_eq!(plan.focus_turns, vec![0]);
        assert!(plan.rationale.contains("preference intent"));
        assert!(plan.topic.is_some());

        let plan = plan_of(&["remember that the deploy key rotates on Fridays."]).await;
        assert_eq!(plan.intent, PlanIntent::Directive);
        assert!(plan.has(PlanOperation::ExtractFacts));
        assert!(!plan.has(PlanOperation::ExtractVariables));

        let plan = plan_of(&["Actually the build changed to stable."]).await;
        assert_eq!(plan.intent, PlanIntent::Correction);
        assert!(plan.has(PlanOperation::HandleDirectives));
        assert!(plan.extracts_facts());

        let plan = plan_of(&["agent_timeout = 30"]).await;
        assert_eq!(plan.intent, PlanIntent::Variable);
        assert!(plan.has(PlanOperation::ExtractVariables));

        let plan =
            plan_with_history(&["It runs on postgres."], &["The service handles billing."]).await;
        assert!(plan.has(PlanOperation::ResolveReferences));
        assert!(!plan.has(PlanOperation::NormalizeTime));

        let plan = plan_of(&["We use postgres tomorrow morning."]).await;
        assert_eq!(plan.intent, PlanIntent::Project);
        assert!(plan.has(PlanOperation::NormalizeTime));

        let long = "the service uses postgres and the pipeline has three stages. ".repeat(8);
        let plan = plan_of(&[&long]).await;
        assert!(plan.has(PlanOperation::ExtractSummary), "{plan:?}");

        let plan = plan_of(&["ok"]).await;
        assert_eq!(plan.intent, PlanIntent::None);
        assert!(!plan.is_actionable());
        assert!(plan.rationale.contains("extraction minimum"));
    }

    #[tokio::test]
    async fn disabled_analyzer_plans_nothing() {
        let config = AnalyzerConfig {
            enabled: false,
            ..Default::default()
        };
        let stage = RulePlanStage::new(config.clone());
        let output = stage
            .run(
                &StageInput::new(conversation(&["I prefer dark mode."]), config),
                &CallOptions::new(),
            )
            .await
            .unwrap();
        let plan = output.plan.unwrap();
        assert_eq!(plan.intent, PlanIntent::None);
        assert!(!plan.is_actionable());
        assert!(plan.rationale.contains("disabled"));
    }

    #[test]
    fn plan_serde_round_trip_and_intent_aliases() {
        let plan = ExtractionPlan {
            intent: PlanIntent::Correction,
            topic: Some("build".to_owned()),
            operations: vec![
                PlanOperation::ResolveReferences,
                PlanOperation::ExtractFacts,
            ],
            focus_turns: vec![0, 1],
            rationale: "correction intent".to_owned(),
            degraded: None,
        };
        let json = serde_json::to_value(&plan).unwrap();
        assert_eq!(json["intent"], serde_json::json!("correction"));
        assert_eq!(json["operations"][1], serde_json::json!("extract_facts"));
        let restored: ExtractionPlan = serde_json::from_value(json).unwrap();
        assert_eq!(restored, plan);
        let tolerant: ExtractionPlan = serde_json::from_value(serde_json::json!({
            "intent": "nonsense",
            "operations": ["extract_facts"],
            "unknown": 1
        }))
        .unwrap();
        assert_eq!(tolerant.intent, PlanIntent::None);
        assert!(tolerant.is_actionable(), "the operations still replay");
        assert_eq!(
            PlanIntent::parse("user_profile"),
            Some(PlanIntent::Identity)
        );
        assert_eq!(PlanIntent::parse("delete"), Some(PlanIntent::Correction));
        assert!(PlanIntent::parse("nope").is_none());
        assert_eq!(
            PlanOperation::parse("time reasoning"),
            Some(PlanOperation::NormalizeTime)
        );
        assert_eq!(
            PlanOperation::parse("SUMMARY"),
            Some(PlanOperation::ExtractSummary)
        );
        assert!(PlanOperation::parse("nope").is_none());
    }

    #[test]
    fn selected_turns_falls_back_to_everything() {
        let conversation = conversation(&["one", "two", "three"]);
        let plan = ExtractionPlan {
            focus_turns: vec![2, 0, 9, 0],
            ..ExtractionPlan::default()
        };
        assert_eq!(plan.selected_turns(&conversation), vec![0, 2]);
        assert_eq!(
            ExtractionPlan::default().selected_turns(&conversation),
            vec![0, 1, 2]
        );
    }

    #[tokio::test]
    async fn llm_plan_parses_a_fenced_answer() {
        let model = ScriptedModel::builder()
            .respond_text(
                "```json\n{\"intent\":\"preference\",\"topic\":\"theme\",\
                 \"operations\":[\"extract_facts\",\"normalize_time\"],\
                 \"focus_turns\":[0,7],\"rationale\":\"the user states a theme preference\"}\n```",
            )
            .build();
        let handle = model.clone();
        let config = AnalyzerConfig::default();
        let stage = LlmPlanStage::new(Arc::new(model), config.clone());
        let output = stage
            .run(
                &StageInput::new(conversation(&["I prefer dark mode."]), config),
                &CallOptions::new(),
            )
            .await
            .unwrap();
        assert!(!output.is_degraded());
        let plan = output.plan.unwrap();
        assert_eq!(plan.intent, PlanIntent::Preference);
        assert_eq!(plan.topic.as_deref(), Some("theme"));
        assert_eq!(
            plan.operations,
            vec![PlanOperation::ExtractFacts, PlanOperation::NormalizeTime]
        );
        assert_eq!(plan.focus_turns, vec![0]);
        assert_eq!(plan.rationale, "the user states a theme preference");
        assert_eq!(handle.call_count(), 1);
    }

    #[tokio::test]
    async fn llm_plan_unparseable_output_falls_back_to_rules() {
        let model = ScriptedModel::builder()
            .respond_text("I would extract a preference.")
            .build();
        let handle = model.clone();
        let config = AnalyzerConfig::default();
        let conversation = conversation(&["I prefer dark mode."]);
        let stage = LlmPlanStage::new(Arc::new(model), config.clone());
        let output = stage
            .run(
                &StageInput::new(conversation.clone(), config.clone()),
                &CallOptions::new(),
            )
            .await
            .unwrap();
        let plan = output.plan.clone().unwrap();
        assert!(output.is_degraded());
        assert_eq!(plan.intent, PlanIntent::Preference);
        assert!(plan
            .degraded
            .as_deref()
            .is_some_and(|reason| reason.contains("fell back to rules")));
        let rule_plan = RulePlanStage::new(config.clone())
            .run(&StageInput::new(conversation, config), &CallOptions::new())
            .await
            .unwrap()
            .plan
            .unwrap();
        assert_eq!(plan.intent, rule_plan.intent);
        assert_eq!(plan.operations, rule_plan.operations);
        assert_eq!(handle.call_count(), 4);
    }

    #[tokio::test]
    async fn llm_plan_cancelled_makes_zero_calls() {
        let model = ScriptedModel::builder().build();
        let handle = model.clone();
        let config = AnalyzerConfig::default();
        let stage = LlmPlanStage::new(Arc::new(model), config.clone());
        let token = CancellationToken::new();
        token.cancel();
        let options = CallOptions::new().with_cancellation(token);
        let error = stage
            .run(
                &StageInput::new(conversation(&["a fact"]), config),
                &options,
            )
            .await
            .unwrap_err();
        assert!(matches!(error, Error::Cancelled), "{error:?}");
        assert_eq!(handle.call_count(), 0);
    }

    #[test]
    fn parse_plan_response_tolerates_shapes() {
        let config = AnalyzerConfig::default();
        let plan = parse_plan_response(
            "{\"plan_intent\":\"update\",\"steps\":[{\"operation\":\"extract\"},{\"name\":\"bogus\"}],\
             \"turns\":[0,1,4],\"reason\":\"an edit\"}",
            &config,
            2,
        )
        .unwrap();
        assert_eq!(plan.intent, PlanIntent::Correction);
        assert_eq!(plan.operations, vec![PlanOperation::ExtractFacts]);
        assert_eq!(plan.focus_turns, vec![0, 1]);
        assert_eq!(plan.rationale, "an edit");
        assert!(plan.topic.is_none());

        let plan =
            parse_plan_response("{\"operations\":\"extract_facts, summary\"}", &config, 1).unwrap();
        assert_eq!(
            plan.operations,
            vec![PlanOperation::ExtractFacts, PlanOperation::ExtractSummary]
        );
        assert!(plan.rationale.contains("model plan"));

        assert!(parse_plan_response("no json here", &config, 1).is_none());
        assert!(parse_plan_response("{\"intent\":\"preference\"}", &config, 1).is_none());
        assert!(parse_plan_response("[1, 2]", &config, 1).is_none());
    }

    #[test]
    fn prompts_embed_their_schemas() {
        assert!(PLAN_SYSTEM_PROMPT.contains(PLAN_JSON_SCHEMA));
        let schema: serde_json::Value = serde_json::from_str(PLAN_JSON_SCHEMA).unwrap();
        assert!(schema.get("operations").is_some());
        let _: StagedResult = StagedResult::default();
    }
}
