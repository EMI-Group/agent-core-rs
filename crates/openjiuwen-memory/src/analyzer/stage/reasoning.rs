//! Reasoning stage: attach a written rationale to every extracted fact.
//!
//! Python's `fragment_memory_prompt` justifies each extraction step inline (anaphora,
//! subject locking, time reasoning, classification) but discards that reasoning once the
//! JSON is parsed. This stage keeps it: every [`ExtractedFact`] gets a short
//! [`FactReasoning`] explaining why it was kept, stored under `metadata["rationale"]`
//! (configurable) together with the plan's intent.
//!
//! [`RuleReasoningStage`] derives the rationale from the fact's tags and importance, so
//! reasoning works offline and is byte-for-byte deterministic; `LlmReasoningStage` asks a
//! model for the same shape and fills anything it cannot answer from the rule rationales,
//! recording the degradation.
use crate::analyzer::config::AnalyzerConfig;
use crate::analyzer::fact::{truncate_chars, ExtractedFact};
use crate::analyzer::parse::payload;
use crate::analyzer::stage::{ExtractionStage, StageInput, StageOutput};
use crate::record::CallOptions;
use async_trait::async_trait;
use openjiuwen_common::error::{Error, ErrorKind, Result};
use openjiuwen_foundation::llm::{Model, ModelRequestConfig};
use openjiuwen_foundation::prompt::BaseMessage;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::Arc;

/// The default metadata key a rationale is stored under.
pub const DEFAULT_REASONING_KEY: &str = "rationale";

/// What a fact's rationale is based on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningBasis {
    /// The turn carried an explicit capture directive.
    Explicit,
    /// The turn stated user-identity information.
    Identity,
    /// The turn stated a preference.
    Preference,
    /// The turn described project/technology knowledge.
    Project,
    /// The turn contained a variable assignment.
    Variable,
    /// The turn contained a plain declarative statement.
    Statement,
    /// The plan selected the fact with no stronger signal.
    Planned,
    /// A model answered the rationale.
    Modeled,
    /// The basis is unknown.
    #[default]
    Unknown,
}

impl ReasoningBasis {
    /// Returns the stable snake_case name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Explicit => "explicit",
            Self::Identity => "identity",
            Self::Preference => "preference",
            Self::Project => "project",
            Self::Variable => "variable",
            Self::Statement => "statement",
            Self::Planned => "planned",
            Self::Modeled => "modeled",
            Self::Unknown => "unknown",
        }
    }

    /// Returns the basis a fact's tags imply, when they imply one.
    #[must_use]
    pub fn from_tags(tags: &[String]) -> Option<Self> {
        for tag in tags {
            let basis = match tag.trim().to_ascii_lowercase().as_str() {
                "explicit" => Self::Explicit,
                "identity" => Self::Identity,
                "preference" => Self::Preference,
                "project" => Self::Project,
                "variable" => Self::Variable,
                "statement" => Self::Statement,
                _ => continue,
            };
            return Some(basis);
        }
        None
    }

    /// Tolerant parser for a model answer's basis field.
    #[must_use]
    pub fn parse(raw: &str) -> Option<Self> {
        let normalized = raw.trim().to_ascii_lowercase().replace([' ', '-'], "_");
        Some(match normalized.as_str() {
            "explicit" | "directive" | "instruction" => Self::Explicit,
            "identity" | "user_profile" | "profile" => Self::Identity,
            "preference" => Self::Preference,
            "project" | "semantic_memory" | "technology" => Self::Project,
            "variable" | "assignment" => Self::Variable,
            "statement" | "fact" | "general" => Self::Statement,
            "planned" | "plan" => Self::Planned,
            "modeled" | "model" => Self::Modeled,
            "unknown" | "" => Self::Unknown,
            _ => return None,
        })
    }
}

/// Why one fact was kept.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct FactReasoning {
    /// Index of the fact in the batch.
    pub fact_index: usize,
    /// One sentence explaining the extraction.
    pub rationale: String,
    /// What the rationale is based on.
    pub basis: ReasoningBasis,
}

impl FactReasoning {
    /// Builds a reasoning entry from a fact index, a rationale, and its basis.
    #[must_use]
    pub fn new(fact_index: usize, rationale: impl Into<String>, basis: ReasoningBasis) -> Self {
        Self {
            fact_index,
            rationale: rationale.into(),
            basis,
        }
    }

    /// Builds an entry with an unknown basis.
    #[must_use]
    pub fn rule(fact_index: usize, rationale: impl Into<String>) -> Self {
        Self::new(fact_index, rationale, ReasoningBasis::Unknown)
    }
}

/// Deterministic reasoning stage.
pub struct RuleReasoningStage {
    /// Effective analyzer settings (content bounds).
    config: AnalyzerConfig,
    /// Metadata key the rationale is stored under.
    key: String,
}

impl RuleReasoningStage {
    /// Builds the rule reasoning stage with the default metadata key.
    #[must_use]
    pub fn new(config: AnalyzerConfig) -> Self {
        Self {
            config,
            key: DEFAULT_REASONING_KEY.to_owned(),
        }
    }

    /// Returns the stage with `key` as its metadata key.
    #[must_use]
    pub fn with_metadata_key(mut self, key: impl Into<String>) -> Self {
        self.key = key.into();
        self
    }

    /// Returns the metadata key in use.
    #[must_use]
    pub fn key(&self) -> &str {
        &self.key
    }

    /// Returns the configuration in use.
    #[must_use]
    pub fn config(&self) -> &AnalyzerConfig {
        &self.config
    }

    /// Builds the rationale list for `facts`.
    fn reason(&self, facts: &[ExtractedFact], plan_intent: &str) -> Vec<FactReasoning> {
        facts
            .iter()
            .enumerate()
            .map(|(index, fact)| {
                let basis = ReasoningBasis::from_tags(&fact.tags).unwrap_or(ReasoningBasis::Planned);
                let rationale = truncate_chars(
                    &self.rationale(fact, basis, plan_intent),
                    self.config.max_content_chars,
                );
                FactReasoning::new(index, rationale, basis)
            })
            .collect()
    }

    /// Renders one fact's deterministic rationale.
    fn rationale(&self, fact: &ExtractedFact, basis: ReasoningBasis, plan_intent: &str) -> String {
        let importance = if fact.importance.is_finite() {
            fact.importance.clamp(0.0, 1.0)
        } else {
            self.config.default_importance
        };
        let subject = match basis {
            ReasoningBasis::Explicit => {
                "explicit capture the user asked to remember".to_owned()
            }
            ReasoningBasis::Identity => "identity statement about the user".to_owned(),
            ReasoningBasis::Preference => "stated preference of the user".to_owned(),
            ReasoningBasis::Project => "project or technology statement".to_owned(),
            ReasoningBasis::Variable => match fact.metadata.get("variable").and_then(Value::as_str) {
                Some(key) => format!("variable assignment for {key}"),
                None => "variable assignment".to_owned(),
            },
            ReasoningBasis::Statement => "declarative statement with a copula".to_owned(),
            ReasoningBasis::Planned | ReasoningBasis::Modeled | ReasoningBasis::Unknown => {
                "selected by the plan's extract_facts operation".to_owned()
            }
        };
        format!(
            "{subject}; {} fact at importance {importance:.2} (plan intent {plan_intent})",
            fact.kind.as_str()
        )
    }

    /// Returns the facts with the rationale metadata attached.
    fn attach(&self, facts: &[ExtractedFact], reasoning: &[FactReasoning], plan_intent: &str) -> Vec<ExtractedFact> {
        let mut updated = facts.to_vec();
        for entry in reasoning {
            if let Some(fact) = updated.get_mut(entry.fact_index) {
                fact.metadata
                    .insert(self.key.clone(), json!(entry.rationale));
                fact.metadata
                    .insert("plan_intent".to_owned(), json!(plan_intent));
            }
        }
        updated
    }
}

#[async_trait]
impl ExtractionStage for RuleReasoningStage {
    #[allow(clippy::unnecessary_literal_bound)] // the trait signature ties `&str` to `self`
    fn name(&self) -> &str {
        "reasoning"
    }

    async fn run(&self, input: &StageInput, options: &CallOptions) -> Result<StageOutput> {
        options.check()?;
        if self.key.trim().is_empty() {
            return Err(Error::validation(
                "reasoning metadata key must not be empty",
            ));
        }
        self.config.validate()?;
        let plan_intent = input.plan.intent.as_str();
        let reasoning = self.reason(&input.facts, plan_intent);
        let facts = self.attach(&input.facts, &reasoning, plan_intent);
        Ok(StageOutput::of(self.name())
            .with_facts(facts)
            .with_reasoning(reasoning))
    }
}

/// The JSON schema a reasoning answer must follow.
pub const REASONING_JSON_SCHEMA: &str = concat!(
    "{\"reasons\":[{\"index\":0,\"rationale\":\"...\",",
    "\"basis\":\"explicit|identity|preference|project|variable|statement|planned\"}]}"
);

/// System prompt for [`LlmReasoningStage`].
pub const REASONING_SYSTEM_PROMPT: &str = concat!(
    "You explain why each extracted memory fact is worth keeping. ",
    "Reply with JSON only, using exactly this shape and no other keys: ",
    "{\"reasons\":[{\"index\":0,\"rationale\":\"...\",",
    "\"basis\":\"explicit|identity|preference|project|variable|statement|planned\"}]}. ",
    "Emit exactly one entry per FACT, keep `index` unchanged, write `rationale` as one ",
    "short sentence in the language of the conversation, and pick `basis` from the ",
    "listed values. Do not invent facts, do not add prose, Markdown fences, or explanations."
);

/// Model-backed reasoning stage with the rule rationales as its fallback.
pub struct LlmReasoningStage {
    /// The model that answers the reasoning prompt.
    model: Arc<dyn Model>,
    /// Analyzer settings (content bounds, retries).
    config: AnalyzerConfig,
    /// Deterministic stage used for unanswered facts and exhausted retries.
    fallback: RuleReasoningStage,
}

impl LlmReasoningStage {
    /// Builds the stage around `model`.
    #[must_use]
    pub fn new(model: Arc<dyn Model>, config: AnalyzerConfig) -> Self {
        let fallback = RuleReasoningStage::new(config.clone());
        Self {
            model,
            config,
            fallback,
        }
    }

    /// Returns the stage with `fallback` as its deterministic backstop.
    #[must_use]
    pub fn with_fallback(mut self, fallback: RuleReasoningStage) -> Self {
        self.fallback = fallback;
        self
    }

    /// Returns the configuration in use.
    #[must_use]
    pub fn config(&self) -> &AnalyzerConfig {
        &self.config
    }

    /// Renders the reasoning prompt body.
    fn render(&self, input: &StageInput) -> String {
        let mut rendered = format!("PLAN INTENT\n{}\n\nFACTS\n", input.plan.intent.as_str());
        for (index, fact) in input.facts.iter().enumerate() {
            rendered.push_str(&format!(
                "FACT {index}: {}: {}\n",
                fact.kind.as_str(),
                fact.content.trim()
            ));
        }
        truncate_chars(&rendered, self.config.max_input_chars)
    }
}

#[async_trait]
impl ExtractionStage for LlmReasoningStage {
    #[allow(clippy::unnecessary_literal_bound)] // the trait signature ties `&str` to `self`
    fn name(&self) -> &str {
        "reasoning"
    }

    async fn run(&self, input: &StageInput, options: &CallOptions) -> Result<StageOutput> {
        options.check()?;
        self.config.validate()?;
        let plan_intent = input.plan.intent.as_str();
        let rule_reasoning = self.fallback.reason(&input.facts, plan_intent);
        if input.facts.is_empty() {
            return Ok(StageOutput::of(self.name()).with_facts(Vec::new()));
        }
        let messages = vec![
            BaseMessage::system(REASONING_SYSTEM_PROMPT),
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
                    if let Some(answered) =
                        parse_reasoning_response(&raw, input.facts.len(), &self.config)
                    {
                        return Ok(self.merge(input, rule_reasoning, answered));
                    }
                    failure
                        .get_or_insert_with(|| "the model answer held no usable rationale".to_owned());
                    tracing::warn!(
                        target: "openjiuwen_memory::analyzer",
                        attempt,
                        attempts,
                        "LLM reasoning stage returned unusable output; retrying"
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
                        "LLM reasoning stage call failed; retrying"
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
            "LLM reasoning stage exhausted its retries; falling back to rules"
        );
        Ok(self.fallback_output(input, rule_reasoning, &failure))
    }
}

impl LlmReasoningStage {
    /// Merges a partial model answer over the rule rationales.
    fn merge(
        &self,
        input: &StageInput,
        rule_reasoning: Vec<FactReasoning>,
        answered: Vec<FactReasoning>,
    ) -> StageOutput {
        let mut answered_count = 0;
        let mut merged = rule_reasoning;
        for entry in answered {
            if let Some(slot) = merged.get_mut(entry.fact_index) {
                answered_count += 1;
                *slot = entry;
            }
        }
        let degraded = if answered_count < merged.len() {
            Some(format!(
                "llm reasoning answered {answered_count} of {} facts; rule rationales filled the rest",
                merged.len()
            ))
        } else {
            None
        };
        let plan_intent = input.plan.intent.as_str();
        let facts = self.fallback.attach(&input.facts, &merged, plan_intent);
        let mut output = StageOutput::of(self.name())
            .with_facts(facts)
            .with_reasoning(merged);
        if let Some(reason) = degraded {
            output = output.degraded(reason);
        }
        output
    }

    /// Builds the full fallback output after the retry budget is spent.
    fn fallback_output(
        &self,
        input: &StageInput,
        rule_reasoning: Vec<FactReasoning>,
        failure: &str,
    ) -> StageOutput {
        let plan_intent = input.plan.intent.as_str();
        let facts = self.fallback.attach(&input.facts, &rule_reasoning, plan_intent);
        StageOutput::of(self.name())
            .with_facts(facts)
            .with_reasoning(rule_reasoning)
            .degraded(format!("llm reasoning fell back to rules: {failure}"))
    }
}

/// Parses a reasoning answer into per-fact entries.
///
/// Accepted shapes: `{"reasons": [...]}`, `{"rationales": [...]}`, or a bare array.
/// Each entry holds `index`/`fact_index` plus `rationale`/`reason`/`why` and an optional
/// `basis`. Out-of-range indices, blank rationales, and unusable entries are skipped; the
/// first entry for an index wins; the result is `None` when nothing usable is left.
#[must_use]
pub fn parse_reasoning_response(
    raw: &str,
    fact_count: usize,
    config: &AnalyzerConfig,
) -> Option<Vec<FactReasoning>> {
    let value = payload(raw)?;
    let items: Vec<&Value> = match &value {
        Value::Array(items) => items.iter().collect(),
        Value::Object(map) => ["reasons", "rationales", "items"]
            .iter()
            .find_map(|key| map.get(*key))
            .and_then(Value::as_array)
            .map(|items| items.iter().collect())?,
        _ => return None,
    };
    let mut reasoning: Vec<FactReasoning> = Vec::new();
    for item in items {
        let Some(map) = item.as_object() else {
            continue;
        };
        let Some(index) = map
            .get("index")
            .or_else(|| map.get("fact_index"))
            .or_else(|| map.get("id"))
            .and_then(Value::as_u64)
            .and_then(|index| usize::try_from(index).ok())
        else {
            continue;
        };
        if index >= fact_count {
            tracing::warn!(
                target: "openjiuwen_memory::analyzer",
                index,
                fact_count,
                "skipping reasoning for an out-of-range fact index"
            );
            continue;
        }
        let rationale = map
            .get("rationale")
            .or_else(|| map.get("reason"))
            .or_else(|| map.get("why"))
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|text| !text.is_empty())
            .map(|text| truncate_chars(text, config.max_content_chars));
        let Some(rationale) = rationale else {
            continue;
        };
        if reasoning.iter().any(|entry| entry.fact_index == index) {
            continue;
        }
        let basis = map
            .get("basis")
            .and_then(Value::as_str)
            .and_then(ReasoningBasis::parse)
            .unwrap_or(ReasoningBasis::Modeled);
        reasoning.push(FactReasoning::new(index, rationale, basis));
    }
    if reasoning.is_empty() {
        None
    } else {
        Some(reasoning)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analyzer::fact::ExtractedFact;
    use crate::analyzer::stage::plan::ExtractionPlan;
    use openjiuwen_common::error::Error;
    use openjiuwen_test_support::ScriptedModel;
    use tokio_util::sync::CancellationToken;

    fn facts_from_builder() -> Vec<ExtractedFact> {
        vec![
            ExtractedFact::user_profile("Alice prefers dark mode")
                .with_importance(0.75)
                .with_tag("preference"),
            ExtractedFact::user_profile("Alice is a developer")
                .with_importance(0.85)
                .with_tag("identity"),
            ExtractedFact::semantic("rotate the deploy key")
                .with_importance(0.9)
                .with_tag("explicit"),
            ExtractedFact::semantic("agent_timeout = 30")
                .with_importance(0.4)
                .with_tag("variable")
                .with_metadata("variable", json!("agent_timeout")),
        ]
    }

    fn state(facts: Vec<ExtractedFact>) -> StageInput {
        let config = AnalyzerConfig::default();
        let mut state = StageInput::new(
            crate::analyzer::fact::ExtractionInput::from_text("I prefer dark mode."),
            config,
        );
        state.facts = facts;
        state.plan = ExtractionPlan {
            intent: crate::analyzer::PlanIntent::Preference,
            ..ExtractionPlan::default()
        };
        state
    }

    #[tokio::test]
    async fn rule_reasoning_is_ordered_and_attached() {
        let config = AnalyzerConfig::default();
        let stage = RuleReasoningStage::new(config);
        let output = stage
            .run(&state(facts_from_builder()), &CallOptions::new())
            .await
            .unwrap();
        let reasoning = output.reasoning.clone();
        assert_eq!(reasoning.len(), 4);
        for (index, entry) in reasoning.iter().enumerate() {
            assert_eq!(entry.fact_index, index);
            assert!(!entry.rationale.trim().is_empty());
        }
        assert_eq!(reasoning[0].basis, ReasoningBasis::Preference);
        assert_eq!(reasoning[1].basis, ReasoningBasis::Identity);
        assert_eq!(reasoning[2].basis, ReasoningBasis::Explicit);
        assert_eq!(reasoning[3].basis, ReasoningBasis::Variable);
        assert!(reasoning[3].rationale.contains("agent_timeout"));
        assert!(reasoning[0].rationale.contains("plan intent preference"));
        let facts = output.facts.unwrap();
        assert_eq!(
            facts[0].metadata.get("rationale"),
            Some(&json!(reasoning[0].rationale))
        );
        assert_eq!(facts[0].metadata.get("plan_intent"), Some(&json!("preference")));
        // Deterministic: a second run over the same input is byte-identical.
        let again = stage
            .run(&state(facts_from_builder()), &CallOptions::new())
            .await
            .unwrap();
        assert_eq!(again.reasoning, reasoning);
    }

    #[tokio::test]
    async fn metadata_key_is_configurable_and_validated() {
        let config = AnalyzerConfig::default();
        let stage = RuleReasoningStage::new(config.clone()).with_metadata_key("why");
        assert_eq!(stage.key(), "why");
        let output = stage
            .run(&state(facts_from_builder()), &CallOptions::new())
            .await
            .unwrap();
        let facts = output.facts.unwrap();
        assert!(facts[0].metadata.contains_key("why"));
        assert!(!facts[0].metadata.contains_key(DEFAULT_REASONING_KEY));

        let stage = RuleReasoningStage::new(config).with_metadata_key("  ");
        let error = stage
            .run(&state(facts_from_builder()), &CallOptions::new())
            .await
            .unwrap_err();
        assert!(matches!(error, Error::Validation(_)), "{error:?}");
    }

    #[tokio::test]
    async fn rule_reasoning_without_facts_is_empty() {
        let stage = RuleReasoningStage::new(AnalyzerConfig::default());
        let output = stage
            .run(&state(Vec::new()), &CallOptions::new())
            .await
            .unwrap();
        assert_eq!(output.reasoning.len(), 0);
        assert_eq!(output.facts.map(|facts| facts.len()), Some(0));
    }

    #[tokio::test]
    async fn llm_reasoning_merges_a_partial_answer() {
        let model = ScriptedModel::builder()
            .respond_text(
                "```json\n{\"reasons\":[{\"index\":0,\"rationale\":\"a durable theme choice\",\
                 \"basis\":\"preference\"}]}\n```",
            )
            .build();
        let handle = model.clone();
        let config = AnalyzerConfig::default();
        let stage = LlmReasoningStage::new(Arc::new(model), config);
        let output = stage
            .run(&state(facts_from_builder()), &CallOptions::new())
            .await
            .unwrap();
        let reasoning = output.reasoning.clone();
        assert_eq!(reasoning.len(), 4);
        assert_eq!(reasoning[0].rationale, "a durable theme choice");
        assert_eq!(reasoning[0].basis, ReasoningBasis::Preference);
        assert_eq!(reasoning[1].basis, ReasoningBasis::Identity);
        assert!(output
            .degraded
            .as_deref()
            .is_some_and(|reason| reason.contains("answered 1 of 4 facts")));
        assert_eq!(handle.call_count(), 1);
    }

    #[tokio::test]
    async fn llm_reasoning_unparseable_output_falls_back_to_rules() {
        let model = ScriptedModel::builder().respond_text("because it matters").build();
        let handle = model.clone();
        let config = AnalyzerConfig::default();
        let stage = LlmReasoningStage::new(Arc::new(model), config);
        let output = stage
            .run(&state(facts_from_builder()), &CallOptions::new())
            .await
            .unwrap();
        assert!(output
            .degraded
            .as_deref()
            .is_some_and(|reason| reason.contains("fell back to rules")));
        assert_eq!(output.reasoning.len(), 4);
        assert!(output.reasoning.iter().all(|entry| entry.basis != ReasoningBasis::Modeled));
        assert_eq!(handle.call_count(), 4);
    }

    #[tokio::test]
    async fn llm_reasoning_cancelled_makes_zero_calls() {
        let model = ScriptedModel::builder().build();
        let handle = model.clone();
        let stage = LlmReasoningStage::new(Arc::new(model), AnalyzerConfig::default());
        let token = CancellationToken::new();
        token.cancel();
        let error = stage
            .run(
                &state(facts_from_builder()),
                &CallOptions::new().with_cancellation(token),
            )
            .await
            .unwrap_err();
        assert!(matches!(error, Error::Cancelled), "{error:?}");
        assert_eq!(handle.call_count(), 0);
    }

    #[test]
    fn parse_reasoning_response_drops_unusable_entries() {
        let config = AnalyzerConfig::default();
        assert!(parse_reasoning_response("{}", 2, &config).is_none());
        assert!(parse_reasoning_response("plain text", 2, &config).is_none());
        let parsed = parse_reasoning_response(
            "{\"rationales\":[{\"fact_index\":1,\"reason\":\"second\",\"basis\":\"identity\"},\
             {\"index\":5,\"rationale\":\"out of range\"},\
             {\"index\":1,\"rationale\":\"duplicate\"},\
             {\"index\":0,\"rationale\":\"  \"}]}",
            2,
            &config,
        )
        .unwrap();
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].fact_index, 1);
        assert_eq!(parsed[0].rationale, "second");
        assert_eq!(parsed[0].basis, ReasoningBasis::Identity);
        let parsed =
            parse_reasoning_response("[{\"index\":0,\"rationale\":\"one\"}]", 1, &config).unwrap();
        assert_eq!(parsed[0].basis, ReasoningBasis::Modeled);
    }

    #[test]
    fn reasoning_types_round_trip_and_prompts_embed_schemas() {
        let entry = FactReasoning::new(2, "because", ReasoningBasis::Project);
        let json = serde_json::to_value(&entry).unwrap();
        assert_eq!(json["basis"], json!("project"));
        let restored: FactReasoning = serde_json::from_value(json).unwrap();
        assert_eq!(restored, entry);
        let tolerant: FactReasoning =
            serde_json::from_value(json!({"unknown": 1, "rationale": "x"})).unwrap();
        assert_eq!(tolerant.fact_index, 0);
        assert_eq!(tolerant.basis, ReasoningBasis::Unknown);
        assert_eq!(ReasoningBasis::from_tags(&["project".to_owned()]), Some(ReasoningBasis::Project));
        assert_eq!(ReasoningBasis::from_tags(&[]), None);
        assert!(REASONING_SYSTEM_PROMPT.contains(REASONING_JSON_SCHEMA));
        let schema: serde_json::Value = serde_json::from_str(REASONING_JSON_SCHEMA).unwrap();
        assert!(schema.get("reasons").is_some());
    }
}
