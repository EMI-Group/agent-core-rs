//! Proactive stage: should this turn be written to memory at all?
//!
//! Python's `fragment_memory_prompt` only emits facts the user asked for, and
//! `generation.py` additionally splits explicit `UPDATE`/`DELETE` instructions from new
//! fragments (`has_explict_instruct`, `instruct_memories`) before anything is stored.
//! This stage keeps that gating decision explicit: it inspects the plan, the extracted
//! facts, and the history, and returns a typed [`ProactiveDecision`] — write or skip, with
//! the signals that drove the choice and the fact indices worth persisting.
//!
//! [`RuleProactiveStage`] is deterministic: explicit capture directives, correction and
//! decision markers, a restated fact (repetition against the history block), or a fact at
//! or above the importance threshold all justify a write. `LlmProactiveStage` asks a model
//! for the same decision and falls back to the rule decision with a recorded degradation.
//! The decision is advisory: [`StagedAnalyzer`](crate::analyzer::StagedAnalyzer) never
//! drops facts because of it unless a caller asks to gate on it.
use crate::analyzer::config::AnalyzerConfig;
use crate::analyzer::fact::{truncate_chars, ExtractedFact};
use crate::analyzer::parse::payload;
use crate::analyzer::rule::compile;
use crate::analyzer::stage::plan::{
    has_correction_marker, has_directive_marker, has_preference_marker,
};
use crate::analyzer::stage::{ExtractionStage, StageInput, StageOutput};
use crate::recall::keyword::jaccard;
use crate::record::CallOptions;
use async_trait::async_trait;
use openjiuwen_common::error::{Error, ErrorKind, Result};
use openjiuwen_foundation::llm::{Model, ModelRequestConfig};
use openjiuwen_foundation::prompt::BaseMessage;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fmt::Write as _;
use std::sync::{Arc, LazyLock};

/// A decision/commitment phrase that makes memory worth writing proactively.
static DECISION: LazyLock<Option<Regex>> = LazyLock::new(|| {
    compile(
        r"(?i)\b(?:let'?s\s+go\s+with|we\s+decided|i\s+decided|decided\s+to|final\s+answer|approved|confirmed|settle\s+on|going\s+with|i'?ll\s+use|we'?ll\s+use|switch\s+to|stick\s+with|决定|确定|就用|采用)\b",
    )
});

/// The knobs of the proactive stage.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ProactiveConfig {
    /// Whether the stage runs at all (`true`); `false` always decides "skip".
    pub enabled: bool,
    /// Facts at or above this importance justify a write on their own (`0.7`).
    pub importance_threshold: f32,
    /// Transcripts shorter than this many characters are never written (`12`).
    pub min_transcript_chars: usize,
    /// Token-overlap ratio from which a fact counts as a repetition (`0.6`).
    pub repetition_threshold: f32,
}

impl Default for ProactiveConfig {
    /// Returns the documented defaults.
    fn default() -> Self {
        Self {
            enabled: true,
            importance_threshold: 0.7,
            min_transcript_chars: 12,
            repetition_threshold: 0.6,
        }
    }
}

impl ProactiveConfig {
    /// Validates the thresholds.
    ///
    /// Non-finite or out-of-`0.0..=1.0` ratios are `Error::Validation` naming the field
    /// and value; `min_transcript_chars == 0` is legal and disables the length gate.
    pub fn validate(&self) -> Result<()> {
        for (field, value) in [
            ("importance_threshold", self.importance_threshold),
            ("repetition_threshold", self.repetition_threshold),
        ] {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(Error::validation(format!(
                    "proactive `{field}` must be within 0.0..=1.0 (got {value})"
                )));
            }
        }
        Ok(())
    }
}

/// One reason the proactive stage decided to write (or considered writing).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProactiveSignal {
    /// The turn asked to remember something.
    ExplicitDirective,
    /// The turn corrects or negates stored memory.
    Correction,
    /// The turn records a decision or commitment.
    Decision,
    /// The turn states a preference.
    Preference,
    /// The turn restates something already present in the history block.
    Repetition,
    /// A fact reached the importance threshold.
    HighImportance,
}

impl ProactiveSignal {
    /// Returns the stable `snake_case` name used in JSON and reason strings.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ExplicitDirective => "explicit_directive",
            Self::Correction => "correction",
            Self::Decision => "decision",
            Self::Preference => "preference",
            Self::Repetition => "repetition",
            Self::HighImportance => "high_importance",
        }
    }

    /// Tolerant parser for a model answer's signal list.
    #[must_use]
    pub fn parse(raw: &str) -> Option<Self> {
        let normalized = raw.trim().to_ascii_lowercase().replace([' ', '-'], "_");
        Some(match normalized.as_str() {
            "explicit_directive" | "explicit" | "directive" | "instruction" => {
                Self::ExplicitDirective
            }
            "correction" | "update" | "delete" | "edit" => Self::Correction,
            "decision" | "commitment" | "choice" => Self::Decision,
            "preference" => Self::Preference,
            "repetition" | "repeat" | "restatement" => Self::Repetition,
            "high_importance" | "importance" | "importance_threshold" => Self::HighImportance,
            _ => return None,
        })
    }

    /// Returns `true` for the signals that make a whole batch writable.
    #[must_use]
    pub const fn promotes_batch(self) -> bool {
        matches!(
            self,
            Self::ExplicitDirective | Self::Correction | Self::Decision | Self::Preference
        )
    }

    /// Returns `true` for the signals that make every candidate fact writable.
    ///
    /// A repeated fact counts: restating something the history already holds means the
    /// user considers it durable, so the whole batch is written rather than only the
    /// facts above the importance threshold.
    pub const fn promotes_batch_write(self) -> bool {
        self.promotes_batch() || matches!(self, Self::Repetition)
    }
}

/// Whether memory should be written for this turn, and why.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ProactiveDecision {
    /// Whether the caller should persist the selected facts.
    pub write: bool,
    /// Why the stage decided so (never empty).
    pub reason: String,
    /// The signals that drove the decision.
    pub signals: Vec<ProactiveSignal>,
    /// Indices of the facts worth persisting.
    pub fact_indexes: Vec<usize>,
    /// The strongest candidate importance in `0.0..=1.0`.
    pub importance: f32,
    /// Why a deterministic fallback made this decision, when one did.
    pub degraded: Option<String>,
}

impl ProactiveDecision {
    /// Builds a "skip" decision with `reason` as its explanation.
    #[must_use]
    pub fn skip(reason: impl Into<String>) -> Self {
        Self {
            write: false,
            reason: reason.into(),
            ..Self::default()
        }
    }

    /// Returns `true` when a deterministic fallback made this decision.
    #[must_use]
    pub fn is_degraded(&self) -> bool {
        self.degraded.is_some()
    }
}

/// Deterministic proactive stage.
pub struct RuleProactiveStage {
    /// Effective analyzer settings (enable flag, content bounds).
    config: AnalyzerConfig,
    /// The stage's own thresholds.
    proactive: ProactiveConfig,
}

impl RuleProactiveStage {
    /// Builds the stage from the analyzer settings and its own thresholds.
    #[must_use]
    pub fn new(config: AnalyzerConfig, proactive: ProactiveConfig) -> Self {
        Self { config, proactive }
    }

    /// Returns the configuration in use.
    #[must_use]
    pub fn config(&self) -> &AnalyzerConfig {
        &self.config
    }

    /// Returns the proactive thresholds in use.
    #[must_use]
    pub fn proactive(&self) -> &ProactiveConfig {
        &self.proactive
    }

    /// Returns the strongest finite importance in `facts`.
    fn max_importance(facts: &[ExtractedFact]) -> f32 {
        facts
            .iter()
            .filter(|fact| fact.importance.is_finite())
            .map(|fact| fact.importance.clamp(0.0, 1.0))
            .fold(0.0_f32, f32::max)
    }

    /// Decides whether and what to write.
    fn decide(&self, input: &StageInput) -> ProactiveDecision {
        let facts = &input.facts;
        if !self.proactive.enabled {
            return ProactiveDecision::skip("proactive memory is disabled");
        }
        if !self.config.enabled {
            return ProactiveDecision::skip("memory extraction is disabled");
        }
        if facts.is_empty() {
            return ProactiveDecision::skip("no candidate facts were extracted");
        }
        let transcript: String = input
            .conversation
            .turns
            .iter()
            .map(|turn| turn.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        if transcript.chars().count() < self.proactive.min_transcript_chars {
            return ProactiveDecision::skip(format!(
                "transcript is shorter than {} characters",
                self.proactive.min_transcript_chars
            ));
        }
        let importance = Self::max_importance(facts);
        let mut signals: Vec<ProactiveSignal> = Vec::new();
        if has_directive_marker(&transcript) {
            signals.push(ProactiveSignal::ExplicitDirective);
        }
        if has_correction_marker(&transcript) {
            signals.push(ProactiveSignal::Correction);
        }
        if crate::analyzer::rule::matches(DECISION.as_ref(), &transcript) {
            signals.push(ProactiveSignal::Decision);
        }
        if has_preference_marker(&transcript) {
            signals.push(ProactiveSignal::Preference);
        }
        if self.is_repetition(input) {
            signals.push(ProactiveSignal::Repetition);
        }
        if importance >= self.proactive.importance_threshold {
            signals.push(ProactiveSignal::HighImportance);
        }
        if signals.is_empty() {
            return ProactiveDecision {
                write: false,
                reason: format!(
                    "no proactive-memory signal; the strongest fact is at importance {importance:.2} below the {:.2} threshold",
                    self.proactive.importance_threshold
                ),
                signals,
                fact_indexes: Vec::new(),
                importance,
                degraded: None,
            };
        }
        let fact_indexes = self.select(facts, &signals);
        let names: Vec<&str> = signals.iter().map(|signal| signal.as_str()).collect();
        ProactiveDecision {
            write: true,
            reason: format!(
                "signals: {}; writing {} of {} fact(s), strongest importance {importance:.2}",
                names.join(", "),
                fact_indexes.len(),
                facts.len()
            ),
            signals,
            fact_indexes,
            importance,
            degraded: None,
        }
    }

    /// Returns the fact indices a write should persist.
    fn select(&self, facts: &[ExtractedFact], signals: &[ProactiveSignal]) -> Vec<usize> {
        let promote = signals.iter().any(|signal| signal.promotes_batch_write());
        facts
            .iter()
            .enumerate()
            .filter(|(_, fact)| {
                promote
                    || (fact.importance.is_finite()
                        && fact.importance >= self.proactive.importance_threshold)
            })
            .map(|(index, _)| index)
            .collect()
    }

    /// Returns `true` when a fact restates something from the history block or the batch.
    fn is_repetition(&self, input: &StageInput) -> bool {
        let threshold = self.proactive.repetition_threshold;
        for (index, fact) in input.facts.iter().enumerate() {
            let repeated = input
                .conversation
                .history
                .iter()
                .any(|turn| jaccard(&fact.content, &turn.text) >= threshold)
                || input.facts.iter().enumerate().any(|(other, candidate)| {
                    other != index && jaccard(&fact.content, &candidate.content) >= threshold
                });
            if repeated {
                return true;
            }
        }
        false
    }
}

#[async_trait]
impl ExtractionStage for RuleProactiveStage {
    #[allow(clippy::unnecessary_literal_bound)] // the trait signature ties `&str` to `self`
    fn name(&self) -> &str {
        "proactive"
    }

    async fn run(&self, input: &StageInput, options: &CallOptions) -> Result<StageOutput> {
        options.check()?;
        self.config.validate()?;
        self.proactive.validate()?;
        Ok(StageOutput::of(self.name()).with_decision(self.decide(input)))
    }
}

/// The JSON schema a proactive decision answer must follow.
pub const PROACTIVE_JSON_SCHEMA: &str = concat!(
    "{\"write\":true,\"reason\":\"...\",",
    "\"signals\":[\"explicit_directive|correction|decision|preference|repetition|high_importance\"],",
    "\"fact_indexes\":[0],\"importance\":0.0}"
);

/// System prompt for [`LlmProactiveStage`].
pub const PROACTIVE_SYSTEM_PROMPT: &str = concat!(
    "You decide whether a conversation turn deserves a proactive long-term memory write. ",
    "Reply with JSON only, using exactly this shape and no other keys: ",
    "{\"write\":true,\"reason\":\"...\",",
    "\"signals\":[\"explicit_directive|correction|decision|preference|repetition|high_importance\"],",
    "\"fact_indexes\":[0],\"importance\":0.0}. ",
    "Set `write` to true only when the turn states a durable preference, a decision or ",
    "commitment, an explicit request to remember something, a correction of stored memory, ",
    "or restates something important; small talk, questions, and one-off commands are not ",
    "memory. `signals` lists the reasons, `fact_indexes` lists the FACT indices worth ",
    "persisting, `importance` is the strongest candidate's importance, and `reason` is one ",
    "sentence in the language of the conversation. Do not add prose, Markdown fences, or ",
    "explanations."
);

/// Model-backed proactive stage with the rule decision as its fallback.
pub struct LlmProactiveStage {
    /// The model that answers the decision prompt.
    model: Arc<dyn Model>,
    /// Analyzer settings (content bounds, retries).
    config: AnalyzerConfig,
    /// Deterministic stage used once the retry budget is spent.
    fallback: RuleProactiveStage,
}

impl LlmProactiveStage {
    /// Builds the stage around `model`.
    #[must_use]
    pub fn new(model: Arc<dyn Model>, config: AnalyzerConfig, proactive: ProactiveConfig) -> Self {
        let fallback = RuleProactiveStage::new(config.clone(), proactive);
        Self {
            model,
            config,
            fallback,
        }
    }

    /// Returns the stage with `fallback` as its deterministic backstop.
    #[must_use]
    pub fn with_fallback(mut self, fallback: RuleProactiveStage) -> Self {
        self.fallback = fallback;
        self
    }

    /// Returns the configuration in use.
    #[must_use]
    pub fn config(&self) -> &AnalyzerConfig {
        &self.config
    }

    /// Renders the decision prompt body.
    fn render(&self, input: &StageInput) -> String {
        let conversation = crate::analyzer::prompt::render_conversation(
            &input.conversation.bounded(self.config.max_input_chars),
            self.config.max_input_chars,
        );
        let mut rendered = format!(
            "PLAN INTENT\n{}\n\nCONVERSATION\n{conversation}\n\nFACTS\n",
            input.plan.intent.as_str()
        );
        for (index, fact) in input.facts.iter().enumerate() {
            let _ = writeln!(rendered, "FACT {index}: {}", fact.content.trim());
        }
        truncate_chars(&rendered, self.config.max_input_chars)
    }
}

#[async_trait]
impl ExtractionStage for LlmProactiveStage {
    #[allow(clippy::unnecessary_literal_bound)] // the trait signature ties `&str` to `self`
    fn name(&self) -> &str {
        "proactive"
    }

    async fn run(&self, input: &StageInput, options: &CallOptions) -> Result<StageOutput> {
        options.check()?;
        self.config.validate()?;
        self.fallback.proactive.validate()?;
        if input.facts.is_empty() || !self.fallback.proactive.enabled {
            return Ok(StageOutput::of(self.name()).with_decision(self.fallback.decide(input)));
        }
        let messages = vec![
            BaseMessage::system(PROACTIVE_SYSTEM_PROMPT),
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
                    if let Some(decision) =
                        parse_proactive_response(&raw, input.facts.len(), &self.config)
                    {
                        return Ok(StageOutput::of(self.name()).with_decision(decision));
                    }
                    failure.get_or_insert_with(|| {
                        "the model answer held no usable decision".to_owned()
                    });
                    tracing::warn!(
                        target: "openjiuwen_memory::analyzer",
                        attempt,
                        attempts,
                        "LLM proactive stage returned unusable output; retrying"
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
                        "LLM proactive stage call failed; retrying"
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
            "LLM proactive stage exhausted its retries; falling back to rules"
        );
        let mut decision = self.fallback.decide(input);
        decision.degraded = Some(format!(
            "llm proactive decision fell back to rules: {failure}"
        ));
        Ok(StageOutput::of(self.name())
            .with_decision(decision.clone())
            .degraded(decision.degraded.unwrap_or_default()))
    }
}

/// Parses a proactive decision answer.
///
/// Accepted shapes: an object with `write` (aliases `should_write`/`remember`), `reason`,
/// `signals`, `fact_indexes`, and `importance`, or the same fields under `decision`.
/// A missing `write` field makes the answer unusable (`None`); unknown signals are
/// skipped with a `tracing::warn!`, out-of-range fact indices are dropped, and the
/// importance is clamped into `0.0..=1.0`.
#[must_use]
pub fn parse_proactive_response(
    raw: &str,
    fact_count: usize,
    config: &AnalyzerConfig,
) -> Option<ProactiveDecision> {
    let value = payload(raw)?;
    let map = match &value {
        Value::Object(map) => map
            .get("decision")
            .and_then(Value::as_object)
            .unwrap_or(map),
        _ => return None,
    };
    let write = map
        .get("write")
        .or_else(|| map.get("should_write"))
        .or_else(|| map.get("remember"))
        .and_then(Value::as_bool)?;
    let mut signals = Vec::new();
    if let Some(items) = map.get("signals").and_then(Value::as_array) {
        for item in items {
            let Some(raw) = item.as_str() else {
                continue;
            };
            if let Some(signal) = ProactiveSignal::parse(raw) {
                if !signals.contains(&signal) {
                    signals.push(signal);
                }
            } else {
                tracing::warn!(
                    target: "openjiuwen_memory::analyzer",
                    %raw,
                    "skipping unknown proactive signal"
                );
            }
        }
    }
    let mut fact_indexes: Vec<usize> = map
        .get("fact_indexes")
        .or_else(|| map.get("facts"))
        .or_else(|| map.get("indices"))
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_u64)
                .filter_map(|index| usize::try_from(index).ok())
                .filter(|index| *index < fact_count)
                .collect()
        })
        .unwrap_or_default();
    fact_indexes.sort_unstable();
    fact_indexes.dedup();
    if write && fact_indexes.is_empty() {
        fact_indexes = (0..fact_count).collect();
    }
    let importance = map
        .get("importance")
        .and_then(Value::as_f64)
        .map_or(0.0, |importance| importance as f32)
        .clamp(0.0, 1.0);
    let reason = match map
        .get("reason")
        .or_else(|| map.get("rationale"))
        .or_else(|| map.get("why"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|text| !text.is_empty())
    {
        Some(text) => truncate_chars(text, config.max_content_chars),
        None if write => "model decided to write memory".to_owned(),
        None => "model decided not to write memory".to_owned(),
    };
    Some(ProactiveDecision {
        write,
        reason,
        signals,
        fact_indexes,
        importance,
        degraded: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analyzer::fact::{ConversationTurn, ExtractedFact, ExtractionInput};
    use crate::analyzer::stage::plan::ExtractionPlan;
    use crate::analyzer::PlanIntent;
    use openjiuwen_common::error::Error;
    use openjiuwen_test_support::ScriptedModel;
    use tokio_util::sync::CancellationToken;

    fn stage() -> RuleProactiveStage {
        RuleProactiveStage::new(AnalyzerConfig::default(), ProactiveConfig::default())
    }

    fn state(transcript: &str, facts: Vec<ExtractedFact>, history: &[&str]) -> StageInput {
        let mut conversation = ExtractionInput::from_text(transcript);
        for turn in history {
            conversation
                .history
                .push(ConversationTurn::new("user", *turn));
        }
        let mut state = StageInput::new(conversation, AnalyzerConfig::default());
        state.facts = facts;
        state.plan = ExtractionPlan {
            intent: PlanIntent::Preference,
            ..ExtractionPlan::default()
        };
        state
    }

    fn preference(importance: f32) -> ExtractedFact {
        ExtractedFact::user_profile("Alice prefers dark mode")
            .with_importance(importance)
            .with_tag("preference")
    }

    #[tokio::test]
    async fn explicit_directive_writes_the_batch() {
        let facts = vec![ExtractedFact::semantic("rotate the deploy key")
            .with_importance(0.9)
            .with_tag("explicit")];
        let decision = stage()
            .run(
                &state("remember to rotate the deploy key", facts, &[]),
                &CallOptions::new(),
            )
            .await
            .unwrap()
            .decision
            .unwrap();
        assert!(decision.write);
        assert!(decision
            .signals
            .contains(&ProactiveSignal::ExplicitDirective));
        assert!(decision.signals.contains(&ProactiveSignal::HighImportance));
        assert_eq!(decision.fact_indexes, vec![0]);
        assert!(decision.reason.contains("explicit_directive"));
        assert!((decision.importance - 0.9).abs() < f32::EPSILON);
        assert!(!decision.is_degraded());
    }

    #[tokio::test]
    async fn plain_smalltalk_skips_with_a_reason() {
        let facts = vec![
            ExtractedFact::semantic("the build pipeline has three stages")
                .with_importance(0.4)
                .with_tag("statement"),
        ];
        let decision = stage()
            .run(
                &state("the build pipeline has three stages", facts, &[]),
                &CallOptions::new(),
            )
            .await
            .unwrap()
            .decision
            .unwrap();
        assert!(!decision.write);
        assert_eq!(decision.signals.len(), 0);
        assert_eq!(decision.fact_indexes.len(), 0);
        assert!(
            decision.reason.contains("below the 0.70 threshold"),
            "{decision:?}"
        );
    }

    #[tokio::test]
    async fn decision_and_preference_markers_count() {
        let facts = vec![preference(0.75)];
        let decision = stage()
            .run(
                &state("let's go with the dark theme", facts.clone(), &[]),
                &CallOptions::new(),
            )
            .await
            .unwrap()
            .decision
            .unwrap();
        assert!(decision.write);
        assert!(decision.signals.contains(&ProactiveSignal::Decision));
        assert_eq!(decision.fact_indexes, vec![0]);

        let decision = stage()
            .run(
                &state("Actually I prefer light mode now", facts, &[]),
                &CallOptions::new(),
            )
            .await
            .unwrap()
            .decision
            .unwrap();
        assert!(decision.signals.contains(&ProactiveSignal::Correction));
        assert!(decision.signals.contains(&ProactiveSignal::Preference));
        assert!(decision.write);
    }

    #[tokio::test]
    async fn repetition_of_a_history_fact_writes() {
        let facts = vec![ExtractedFact::semantic("the deploy key rotates on fridays")
            .with_importance(0.4)
            .with_tag("statement")];
        let decision = stage()
            .run(
                &state(
                    "the deploy key rotates on Fridays",
                    facts,
                    &["the deploy key rotates on fridays"],
                ),
                &CallOptions::new(),
            )
            .await
            .unwrap()
            .decision
            .unwrap();
        assert!(decision.write);
        assert!(decision.signals.contains(&ProactiveSignal::Repetition));
        assert_eq!(decision.fact_indexes, vec![0]);
    }

    #[tokio::test]
    async fn skip_reasons_cover_disabled_short_and_empty() {
        let disabled = RuleProactiveStage::new(
            AnalyzerConfig::default(),
            ProactiveConfig {
                enabled: false,
                ..ProactiveConfig::default()
            },
        );
        let decision = disabled
            .run(
                &state("remember this", vec![preference(0.9)], &[]),
                &CallOptions::new(),
            )
            .await
            .unwrap()
            .decision
            .unwrap();
        assert!(!decision.write);
        assert_eq!(decision.reason, "proactive memory is disabled");

        let decision = stage()
            .run(&state("ok", Vec::new(), &[]), &CallOptions::new())
            .await
            .unwrap()
            .decision
            .unwrap();
        assert!(!decision.write);
        assert_eq!(decision.reason, "no candidate facts were extracted");

        let decision = stage()
            .run(
                &state("hi", vec![preference(0.9)], &[]),
                &CallOptions::new(),
            )
            .await
            .unwrap()
            .decision
            .unwrap();
        assert!(!decision.write);
        assert!(
            decision.reason.contains("shorter than 12 characters"),
            "{decision:?}"
        );
    }

    #[tokio::test]
    async fn llm_proactive_stage_parses_a_decision() {
        let model = ScriptedModel::builder()
            .respond_text(
                "```json\n{\"write\":true,\"reason\":\"a durable preference\",\
                 \"signals\":[\"preference\",\"bogus\"],\"fact_indexes\":[0],\
                 \"importance\":0.8}\n```",
            )
            .build();
        let handle = model.clone();
        let stage = LlmProactiveStage::new(
            Arc::new(model),
            AnalyzerConfig::default(),
            ProactiveConfig::default(),
        );
        let output = stage
            .run(
                &state("I prefer dark mode", vec![preference(0.75)], &[]),
                &CallOptions::new(),
            )
            .await
            .unwrap();
        assert!(!output.is_degraded());
        let decision = output.decision.unwrap();
        assert!(decision.write);
        assert_eq!(decision.reason, "a durable preference");
        assert_eq!(decision.signals, vec![ProactiveSignal::Preference]);
        assert_eq!(decision.fact_indexes, vec![0]);
        assert!((decision.importance - 0.8).abs() < 0.001);
        assert_eq!(handle.call_count(), 1);
    }

    #[tokio::test]
    async fn llm_proactive_unparseable_output_falls_back_to_rules() {
        let model = ScriptedModel::builder()
            .respond_text("yes, remember it")
            .build();
        let handle = model.clone();
        let stage = LlmProactiveStage::new(
            Arc::new(model),
            AnalyzerConfig::default(),
            ProactiveConfig::default(),
        );
        let output = stage
            .run(
                &state("remember the deploy key", vec![preference(0.9)], &[]),
                &CallOptions::new(),
            )
            .await
            .unwrap();
        assert!(output.is_degraded());
        let decision = output.decision.unwrap();
        assert!(decision
            .degraded
            .as_deref()
            .is_some_and(|reason| reason.contains("fell back to rules")));
        assert!(decision.write);
        assert_eq!(handle.call_count(), 4);
    }

    #[tokio::test]
    async fn llm_proactive_cancelled_makes_zero_calls() {
        let model = ScriptedModel::builder().build();
        let handle = model.clone();
        let stage = LlmProactiveStage::new(
            Arc::new(model),
            AnalyzerConfig::default(),
            ProactiveConfig::default(),
        );
        let token = CancellationToken::new();
        token.cancel();
        let error = stage
            .run(
                &state("remember this", vec![preference(0.9)], &[]),
                &CallOptions::new().with_cancellation(token),
            )
            .await
            .unwrap_err();
        assert!(matches!(error, Error::Cancelled), "{error:?}");
        assert_eq!(handle.call_count(), 0);
    }

    #[test]
    fn config_validation_and_decision_serde() {
        let config = ProactiveConfig::default();
        assert!(config.validate().is_ok());
        for invalid in [
            ProactiveConfig {
                importance_threshold: 1.5,
                ..ProactiveConfig::default()
            },
            ProactiveConfig {
                repetition_threshold: f32::NAN,
                ..ProactiveConfig::default()
            },
        ] {
            assert!(matches!(invalid.validate(), Err(Error::Validation(_))));
        }
        let decision = ProactiveDecision {
            write: true,
            reason: "signals".to_owned(),
            signals: vec![ProactiveSignal::HighImportance],
            fact_indexes: vec![0],
            importance: 0.7,
            degraded: None,
        };
        let json = serde_json::to_value(&decision).unwrap();
        assert_eq!(json["signals"][0], serde_json::json!("high_importance"));
        let restored: ProactiveDecision = serde_json::from_value(json).unwrap();
        assert_eq!(restored, decision);
        let tolerant: ProactiveDecision =
            serde_json::from_value(serde_json::json!({"unknown": 1})).unwrap();
        assert!(!tolerant.write);
        assert_eq!(tolerant.reason, "");
        assert_eq!(
            ProactiveSignal::parse("explicit directive"),
            Some(ProactiveSignal::ExplicitDirective)
        );
        assert!(ProactiveSignal::parse("nope").is_none());
        assert!(PROACTIVE_SYSTEM_PROMPT.contains(PROACTIVE_JSON_SCHEMA));
    }

    #[test]
    fn parse_proactive_response_tolerates_shapes() {
        let config = AnalyzerConfig::default();
        assert!(parse_proactive_response("{}", 1, &config).is_none());
        assert!(parse_proactive_response("{\"reason\":\"x\"}", 1, &config).is_none());
        let decision = parse_proactive_response(
            "{\"decision\":{\"should_write\":true,\"rationale\":\"durable\",\
             \"signals\":[\"decision\"],\"indices\":[0,9],\"importance\":2.0}}",
            1,
            &config,
        )
        .unwrap();
        assert!(decision.write);
        assert_eq!(decision.reason, "durable");
        assert_eq!(decision.fact_indexes, vec![0]);
        assert!((decision.importance - 1.0).abs() < f32::EPSILON);

        let decision =
            parse_proactive_response("{\"write\":false,\"signals\":[]}", 2, &config).unwrap();
        assert!(!decision.write);
        assert_eq!(decision.fact_indexes.len(), 0);
        assert_eq!(decision.reason, "model decided not to write memory");

        let decision =
            parse_proactive_response("{\"write\":true,\"signals\":[]}", 2, &config).unwrap();
        assert_eq!(decision.fact_indexes, vec![0, 1]);
    }
}
