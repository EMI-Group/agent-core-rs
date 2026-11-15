//! LLM-guided stages with deterministic offline fallbacks.
//!
//! Python's `openjiuwen.symphony` hands five stages to a model: ontology matching
//! between capability outputs and inputs, semantic plan scoring, progressive
//! capability-tree retrieval, fingerprint extraction, and LLM-judged evaluation. This
//! module ports those stages onto the crate's existing one-prompt-in/one-text-out seam
//! — [`CompletionModel`](openjiuwen_retrieval::CompletionModel) from `openjiuwen-retrieval` — and gives every stage a
//! documented deterministic fallback, so the whole crate keeps working offline with no
//! model configured.
//!
//! # The seam
//!
//! [`CompletionModel::complete`](openjiuwen_retrieval::CompletionModel::complete) takes
//! one fully rendered prompt and returns the reply verbatim, so prompt construction and
//! reply parsing stay with each stage. [`StaticCompletionModel`](openjiuwen_retrieval::StaticCompletionModel) replays scripted replies
//! and is what the tests use; [`FoundationCompletionModel`](crate::llm::FoundationCompletionModel) adapts a real
//! `openjiuwen_foundation::llm::Model` (for example an
//! `OpenAiCompatibleClient`) onto the seam.
//!
//! [`complete_json`](crate::llm::complete_json) is the shared call helper: it renders the system/user prompt,
//! invokes the model, extracts the JSON object with
//! [`extract_json_object`](openjiuwen_retrieval::extract_json_object), and turns a reply
//! that carries no parseable object into a soft `None` (after a per-reply
//! `tracing::debug!`). A model *invocation* failure stays an error, so a caller can apply
//! the configured [`ModelConfig`](crate::llm::ModelConfig) fallback.
//!
//! # Fallbacks
//!
//! [`FallbackMode`](crate::llm::FallbackMode) is the single switch: [`FallbackMode::Deterministic`](crate::llm::FallbackMode) (the default)
//! degrades to the stage's offline rule, while [`FallbackMode::Error`](crate::llm::FallbackMode) surfaces the
//! model failure. Every fallback is deterministic and documented on the stage that owns
//! it, and every stage reports where its answer came from:
//!
//! | Module | LLM stage | Deterministic fallback | Provenance marker |
//! | --- | --- | --- | --- |
//! | [`tree`](crate::llm::tree) | progressive tree walk | lexical query-term coverage | `TreeSearchOutcome::provenance` |
//! | [`ontology`](crate::llm::ontology) | `can_feed` validation | the candidate rule's own confidence | `MatchedVerdict::provenance` |
//! | [`scoring`](crate::llm::scoring) | semantic plan scores | [`crate::orchestration::LexicalSemanticScorer`] | `SemanticScores::provenance` |
//! | [`fingerprint`](crate::llm::fingerprint) | semantic profile extraction | a profile derived from the card | `FingerprintOutcome::provenance` |
//! | [`judge`](crate::llm::judge) | LLM-judged metric | a static description/IO rubric | `JudgeVerdict::provenance` |
//!
//! # Provenance
//!
//! Because every stage silently answers from an offline rule when no model is wired, each
//! stage's result carries a `StageProvenance` so a caller can
//! tell a model verdict from a rule: `Model` means a configured model answered and its
//! reply was usable, `DeterministicFallback` means a model was configured but the call
//! failed or the reply was unusable so the documented rule answered, and `NoModel` means
//! no model produced the answer (none is configured, or the request needed none).
//!
//! # Logging policy
//!
//! Every stage emits exactly one record per invocation that had no usable model, naming
//! the stage and the fallback rule it used:
//!
//! - No model configured: exactly one `tracing::debug!`. That is the documented offline
//!   default, so a warning per offline call would be noise; the provenance marker on the
//!   result is the primary signal and log level is left to the caller.
//! - A configured model that failed or answered unusably under
//!   [`FallbackMode::Deterministic`](crate::llm::FallbackMode): exactly one `tracing::warn!`.
//! - [`FallbackMode::Error`](crate::llm::FallbackMode): the typed
//!   `Error::Llm` is returned instead, so nothing is logged as
//!   a fallback (the error itself carries the reason).
//!
//! A stage's own reply-validation diagnostics (a malformed judgement, a missing
//! `semantic_profile`, an unknown candidate id) stay per-item `tracing::warn!`s: they
//! describe the reply, not the fallback decision.
//!
//! # Contracts
//!
//! Every public data type is `serde`-serializable (`snake_case`, tolerant of unknown
//! fields, empty collections by default) and every stage is async. Replies are never
//! trusted: malformed items are dropped and logged, confidence and score values are
//! clamped, and no reply shape can panic. The relation vocabulary is always
//! [`crate::graph::CAN_FEED`], never read from the model.
use std::fmt;
use std::sync::Arc;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::Result;

/// Persistent ontology-match cache.
pub mod cache;
/// Fingerprint extraction: model-authored semantic profiles.
pub mod fingerprint;
/// A [`CompletionModel`] adapter over a foundation `Model`.
pub mod foundation;
/// Model and metadata identity fingerprints used by the cache.
pub mod identity;
/// LLM-judged evaluation metrics.
pub mod judge;
/// LLM-validated `can_feed` ontology matching.
pub mod ontology;
/// LLM-scored semantic relevance for the planner.
pub mod scoring;
/// LLM-guided progressive capability-tree retrieval.
pub mod tree;

pub use openjiuwen_retrieval::{extract_json_object, CompletionModel, StaticCompletionModel};

pub use cache::{cache_key, CachedMatch, MatchCache, CACHE_SCHEMA_VERSION};
pub use fingerprint::{FingerprintConfig, FingerprintExtractor, FingerprintOutcome};
pub use foundation::FoundationCompletionModel;
pub use identity::{
    canonical_json, endpoint_sha256, graph_identity_sha256, sanitize_metadata,
    stable_metadata_sha256, ModelIdentity, IDENTITY_VERSION, MAX_METADATA_DEPTH,
};
pub use judge::{LlmJudgeMetric, DEFAULT_JUDGE_NAME, DEFAULT_RUBRIC};
pub use ontology::{
    MatchOptions, OntologyConfig, OntologyMatch, OntologyMatcher, MATCHER_VERSION,
    MATCH_SCHEMA_VERSION, PROMPT_VERSION,
};
pub use scoring::{DeterministicScorer, LlmSemanticScorer, SemanticScores};
pub use tree::{
    LlmTreeRetriever, RetrieverItem, RetrieverNode, TreeRetriever, TreeRetrieverConfig,
    TreeSearchRequest, TreeSearchResult,
};

/// Chooses what a stage does when the configured model fails or answers unusably.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum FallbackMode {
    /// Fall back to the stage's deterministic offline behaviour (the default).
    #[default]
    Deterministic,
    /// Surface the model failure as an error instead of falling back.
    Error,
}

/// Where a stage's answer came from.
///
/// Every stage in this module answers from its deterministic rule when no model is
/// wired, so the answer alone cannot tell a caller whether a model was involved. Each
/// stage reports this marker on the outcome type it returns (see the module docs for the
/// per-stage location).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum StageProvenance {
    /// A configured model answered and its reply was usable.
    Model,
    /// A model was configured but the call failed or the reply was unusable, so the
    /// stage returned its documented deterministic rule instead.
    DeterministicFallback,
    /// No model produced the answer: none is configured, or the request needed none.
    #[default]
    NoModel,
}

impl StageProvenance {
    /// The `snake_case` wire name (`"model"`, `"deterministic_fallback"`, `"no_model"`).
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Model => "model",
            Self::DeterministicFallback => "deterministic_fallback",
            Self::NoModel => "no_model",
        }
    }

    /// Whether a model produced the answer.
    #[must_use]
    pub fn is_model(self) -> bool {
        matches!(self, Self::Model)
    }

    /// Whether an offline rule produced the answer (both non-model variants).
    #[must_use]
    pub fn is_deterministic(self) -> bool {
        !self.is_model()
    }

    /// Whether this provenance is logged at `warn` rather than `debug`.
    ///
    /// Only [`StageProvenance::DeterministicFallback`] is: a configured model that was
    /// unusable is a degradation worth a warning, while running with no model is the
    /// documented offline default.
    #[must_use]
    pub fn is_warning(self) -> bool {
        matches!(self, Self::DeterministicFallback)
    }
}

impl fmt::Display for StageProvenance {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Records that a stage ran without any model: the documented offline default.
///
/// This is a `debug!` record (not a warning) because a stage with no model is the
/// expected offline path; the structured [`StageProvenance`] on the stage's result is the
/// primary signal and a warning per offline call would make the default noisy.
pub(crate) fn note_no_model(stage: &str, fallback: &str) {
    tracing::debug!(
        stage,
        fallback,
        "llm stage ran without a model; reporting the deterministic fallback"
    );
}

/// Records that a configured model was unusable, so the deterministic rule answered.
///
/// This is the stage's single `warn!`: it names the stage and the fallback rule. The
/// [`FallbackMode::Error`] path returns the typed error instead and does not double-log.
pub(crate) fn note_fallback(stage: &str, fallback: &str, error: &crate::error::Error) {
    tracing::warn!(
        stage,
        fallback,
        %error,
        "llm stage model was unusable; reporting the deterministic fallback"
    );
}

/// The model and sampling settings one stage uses.
///
/// `model: None` keeps the stage on its deterministic path without ever calling a
/// model. `temperature` and `max_tokens` are hints that a model adapter can honour
/// ([`FoundationCompletionModel`] forwards them; [`StaticCompletionModel`] ignores
/// them), so scripts stay reproducible.
#[derive(Clone, Default)]
pub struct ModelConfig {
    /// The completion model; `None` means "use the deterministic path".
    pub model: Option<Arc<dyn CompletionModel>>,
    /// Sampling temperature hint for adapters that can honour it.
    pub temperature: Option<f32>,
    /// Generated-token cap hint for adapters that can honour it.
    pub max_tokens: Option<usize>,
    /// What to do when the model fails or answers unusably.
    pub fallback: FallbackMode,
}

impl ModelConfig {
    /// Builds a configuration with no model and [`FallbackMode::Deterministic`].
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the completion model.
    #[must_use]
    pub fn with_model(mut self, model: Arc<dyn CompletionModel>) -> Self {
        self.model = Some(model);
        self
    }

    /// Sets the sampling temperature hint.
    #[must_use]
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Sets the generated-token cap hint.
    #[must_use]
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Sets the fallback mode.
    #[must_use]
    pub fn with_fallback(mut self, fallback: FallbackMode) -> Self {
        self.fallback = fallback;
        self
    }

    /// The configured model, if any.
    #[must_use]
    pub fn model(&self) -> Option<&dyn CompletionModel> {
        self.model.as_deref()
    }

    /// Whether a model is configured.
    #[must_use]
    pub fn is_configured(&self) -> bool {
        self.model.is_some()
    }
}

impl fmt::Debug for ModelConfig {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ModelConfig")
            .field("model", &self.model().map(CompletionModel::name))
            .field("temperature", &self.temperature)
            .field("max_tokens", &self.max_tokens)
            .field("fallback", &self.fallback)
            .finish()
    }
}

/// Renders the system and user parts into the single prompt a [`CompletionModel`]
/// receives.
///
/// A blank part is dropped entirely (rather than leaving a stray blank line), so a
/// stage without a system instruction sends only its user payload.
#[must_use]
pub fn render_prompt(system: &str, user: &str) -> String {
    let system = system.trim();
    let user = user.trim();
    match (system.is_empty(), user.is_empty()) {
        (true, _) => user.to_string(),
        (false, true) => system.to_string(),
        (false, false) => format!("{system}\n\n{user}"),
    }
}

/// Calls `model` with a rendered system/user prompt and parses the JSON object it
/// returns.
///
/// `model: None` answers `Ok(None)` without calling anything — the stage then takes its
/// deterministic path. A reply with no extractable JSON object (or JSON that is not an
/// object) also answers `Ok(None)`, after a `tracing::debug!`: that is the "soft failure"
/// every Python stage tolerates. A soft reply failure is only a per-reply diagnostic at
/// `debug` level, because the calling stage then emits its own single `warn!`/`debug!`
/// naming itself and the fallback rule it used. A model *invocation* error is returned
/// unchanged so the caller can apply [`ModelConfig::fallback`].
///
/// # Errors
///
/// Returns the [`CompletionModel::complete`]
/// error when the model call itself fails.
pub async fn complete_json(
    model: Option<&dyn CompletionModel>,
    system: &str,
    user: &str,
) -> Result<Option<Value>> {
    let Some(model) = model else {
        return Ok(None);
    };
    let reply = model.complete(&render_prompt(system, user)).await?;
    Ok(parse_json_reply(&reply))
}

/// Parses one model reply into a JSON object, treating anything else as a soft failure.
///
/// Every rejection is a per-reply `debug!` diagnostic, not a warning: the calling stage
/// reports the fallback decision itself, exactly once per invocation.
fn parse_json_reply(reply: &str) -> Option<Value> {
    let Some(object) = extract_json_object(reply) else {
        tracing::debug!("model reply contains no JSON object; treating it as a soft failure");
        return None;
    };
    match serde_json::from_str::<Value>(object) {
        Ok(value) if value.is_object() => Some(value),
        Ok(_) => {
            tracing::debug!("model reply JSON is not an object; treating it as a soft failure");
            None
        }
        Err(error) => {
            tracing::debug!(%error, "model reply JSON could not be parsed; treating it as a soft failure");
            None
        }
    }
}

/// Truncates `text` to at most `limit` characters, on a character boundary.
///
/// Prompt payloads only need a short summary of a card, and truncating on characters
/// (not bytes) keeps multi-byte text intact.
#[must_use]
pub(crate) fn truncate_text(text: &str, limit: usize) -> &str {
    match text.char_indices().nth(limit) {
        Some((end, _)) => &text[..end],
        None => text,
    }
}

/// Deduplicates `items` first-wins, dropping blank entries.
pub(crate) fn dedupe_strings(items: impl IntoIterator<Item = String>) -> Vec<String> {
    let mut seen: Vec<String> = Vec::new();
    for item in items {
        let item = item.trim().to_string();
        if item.is_empty() || seen.contains(&item) {
            continue;
        }
        seen.push(item);
    }
    seen
}

/// Collects the string entries of a JSON array, skipping non-string and blank entries.
pub(crate) fn string_list(value: Option<&Value>) -> Vec<String> {
    let Some(items) = value.and_then(Value::as_array) else {
        return Vec::new();
    };
    dedupe_strings(items.iter().filter_map(Value::as_str).map(str::to_string))
}

/// Reads one non-blank string field out of a JSON object.
pub(crate) fn string_field(value: &Value, field: &str) -> Option<String> {
    let text = value.get(field)?.as_str()?.trim();
    if text.is_empty() {
        None
    } else {
        Some(text.to_string())
    }
}

/// Clamps a score into `[0.0, 1.0]`, mapping non-finite values to `0.0`.
pub(crate) fn clamp_unit(value: f64) -> f64 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn render_prompt_drops_blank_parts() {
        assert_eq!(render_prompt("sys", "user"), "sys\n\nuser");
        assert_eq!(render_prompt("  ", "user"), "user");
        assert_eq!(render_prompt("sys", ""), "sys");
        assert_eq!(render_prompt("", ""), "");
    }

    #[test]
    fn parse_json_reply_accepts_objects_in_prose_and_fences() {
        assert_eq!(
            parse_json_reply("here you go {\"a\": 1} thanks"),
            Some(json!({"a": 1}))
        );
        assert_eq!(
            parse_json_reply("```json\n{\"a\": 2}\n```"),
            Some(json!({"a": 2}))
        );
    }

    #[test]
    fn parse_json_reply_rejects_non_objects_and_missing_json() {
        assert!(parse_json_reply("no json here").is_none());
        assert!(parse_json_reply("[1, 2]").is_none());
        assert!(parse_json_reply("{\"a\": }").is_none());
    }

    #[test]
    fn truncate_text_counts_characters() {
        assert_eq!(truncate_text("abcdef", 3), "abc");
        assert_eq!(truncate_text("abc", 10), "abc");
        assert_eq!(truncate_text("héllo", 2), "hé");
    }

    #[test]
    fn dedupe_strings_is_first_wins_and_drops_blanks() {
        let values = [" b ", "", "a", "b", "a"];
        assert_eq!(
            dedupe_strings(values.into_iter().map(str::to_string)),
            vec!["b".to_string(), "a".to_string()]
        );
    }

    #[test]
    fn clamp_unit_maps_non_finite_to_zero() {
        assert_eq!(clamp_unit(-1.0), 0.0);
        assert_eq!(clamp_unit(2.0), 1.0);
        assert_eq!(clamp_unit(f64::NAN), 0.0);
    }

    #[test]
    fn model_config_debug_never_leaks_the_model_object() {
        let config = ModelConfig::new()
            .with_model(Arc::new(StaticCompletionModel::single("{}")))
            .with_temperature(0.2)
            .with_max_tokens(64);
        let rendered = format!("{config:?}");
        assert!(rendered.contains("static-completion-model"));
        assert!(rendered.contains("Deterministic"));
        assert!(config.is_configured());
    }

    #[test]
    fn stage_provenance_reports_its_wire_name_and_classification() {
        let cases = [
            (StageProvenance::Model, "model", true, false, false),
            (
                StageProvenance::DeterministicFallback,
                "deterministic_fallback",
                false,
                true,
                true,
            ),
            (StageProvenance::NoModel, "no_model", false, true, false),
        ];
        for (provenance, wire, is_model, is_deterministic, is_warning) in cases {
            assert_eq!(provenance.as_str(), wire);
            assert_eq!(provenance.to_string(), wire);
            assert_eq!(provenance.is_model(), is_model, "{wire}");
            assert_eq!(provenance.is_deterministic(), is_deterministic, "{wire}");
            assert_eq!(provenance.is_warning(), is_warning, "{wire}");
        }
    }

    #[test]
    fn stage_provenance_defaults_to_no_model() {
        assert_eq!(StageProvenance::default(), StageProvenance::NoModel);
        assert!(!StageProvenance::default().is_model());
    }

    #[test]
    fn stage_provenance_round_trips_through_serde() {
        for provenance in [
            StageProvenance::Model,
            StageProvenance::DeterministicFallback,
            StageProvenance::NoModel,
        ] {
            let json = serde_json::to_string(&provenance).unwrap();
            assert_eq!(json, format!("\"{}\"", provenance.as_str()));
            assert_eq!(
                serde_json::from_str::<StageProvenance>(&json).unwrap(),
                provenance
            );
        }
        assert_eq!(
            serde_json::from_str::<StageProvenance>("\"model\"").unwrap(),
            StageProvenance::Model
        );
        assert_eq!(
            serde_json::from_str::<StageProvenance>("\"deterministic_fallback\"").unwrap(),
            StageProvenance::DeterministicFallback
        );
        assert_eq!(
            serde_json::from_str::<StageProvenance>("\"no_model\"").unwrap(),
            StageProvenance::NoModel
        );
    }
}
