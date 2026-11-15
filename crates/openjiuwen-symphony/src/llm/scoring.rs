//! LLM-scored semantic relevance for the planner.
//!
//! Python's beam planner asks a model to score every candidate capability against the
//! query. [`LlmSemanticScorer`] is that stage behind the crate's
//! [`SemanticScorer`](crate::orchestration::SemanticScorer) trait, so
//! [`Planner`](crate::orchestration::Planner) can be built with a model-backed scorer
//! without changing anything else about planning.
//!
//! # Request and reply
//!
//! The user payload is `{"query": …, "candidates":[{"candidate_id","name","description"}]}`
//! and the reply must be
//! `{"judgements":[{"candidate_id":"…","score":0.0,"reason":"…"}]}`: each candidate is
//! scored between `0` (useless) and `1` (directly useful). Scores are clamped into
//! `[0, 1]`, a missing or non-numeric score counts as `0.0` (Python behaviour), unknown
//! candidate ids are dropped, and a candidate scored twice keeps its first verdict.
//!
//! # The deterministic fallback
//!
//! With no model configured the scorer delegates to [`DeterministicScorer`] (the
//! crate's [`LexicalSemanticScorer`](crate::orchestration::LexicalSemanticScorer): query
//! term coverage of the retrieval text). A model failure or an unusable reply also falls
//! back, so planning always sees scores; [`FallbackMode::Error`] surfaces the failure
//! instead. Candidates the model did not score are simply absent from the map, which the
//! planner treats as unscored.
//!
//! # Provenance
//!
//! [`score_with_provenance`](crate::llm::scoring::LlmSemanticScorer::score_with_provenance)
//! returns the scores together with a
//! [`StageProvenance`]:
//!
//! | Path | Provenance | Log record |
//! | --- | --- | --- |
//! | no candidates to score | `StageProvenance::NoModel` | none — nothing was asked and nothing was degraded |
//! | no model configured | `StageProvenance::NoModel` | one `tracing::debug!` |
//! | the model answered | `StageProvenance::Model` | none |
//! | the model failed or answered unusably under [`FallbackMode::Deterministic`] | `StageProvenance::DeterministicFallback` | exactly one `tracing::warn!`, naming the stage and the lexical scorer |
//! | the model failed under [`FallbackMode::Error`] | — | the typed `Error::Llm` is returned instead |
//!
//! The [`SemanticScorer`](crate::orchestration::SemanticScorer) implementation delegates to
//! that method and returns only its scores, so the trait's behaviour and scores are
//! unchanged.
use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::error::{Error, Result};
use crate::model::CapabilityDescriptor;
use crate::orchestration::SemanticScorer;

use super::{
    clamp_unit, complete_json, note_fallback, note_no_model, truncate_text, CompletionModel,
    FallbackMode, ModelConfig, StageProvenance,
};

/// The deterministic lexical scorer, re-exported under the name this module's callers
/// use for it.
pub use crate::orchestration::LexicalSemanticScorer as DeterministicScorer;

/// The semantic scores of one planner query plus where they came from.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct SemanticScores {
    /// The score per qualified capability id; candidates the model did not score are absent.
    pub scores: BTreeMap<String, f64>,
    /// Where the scores came from.
    pub provenance: StageProvenance,
}

impl SemanticScores {
    /// Builds a scored answer.
    #[must_use]
    pub fn new(scores: BTreeMap<String, f64>, provenance: StageProvenance) -> Self {
        Self { scores, provenance }
    }

    /// Whether a model produced the scores.
    #[must_use]
    pub fn is_model_backed(&self) -> bool {
        self.provenance.is_model()
    }
}

/// How many characters of a description are sent to the model.
const DESCRIPTION_LIMIT: usize = 240;
/// The system instruction sent with every scoring request.
const SYSTEM_PROMPT: &str = "Score how useful each candidate capability is for fulfilling the query. \
Return JSON only: {\"judgements\":[{\"candidate_id\":\"…\",\"score\":0.0,\"reason\":\"optional short reason\"}]}. \
A score of 0 means the capability is useless for the query; 1 means it is directly useful. \
Scores must be between 0 and 1.";

/// A [`SemanticScorer`] that asks a model for relevance judgements.
///
/// The scorer is cheap to clone: the model is `Arc`-shared through [`ModelConfig`].
#[derive(Debug, Clone, Default)]
pub struct LlmSemanticScorer {
    /// The model and fallback mode in use.
    config: ModelConfig,
}

impl LlmSemanticScorer {
    /// Builds the scorer from `config`.
    #[must_use]
    pub fn new(config: ModelConfig) -> Self {
        Self { config }
    }

    /// Builds the scorer around `model`, with [`FallbackMode::Deterministic`].
    #[must_use]
    pub fn with_model(model: Arc<dyn CompletionModel>) -> Self {
        Self::new(ModelConfig::new().with_model(model))
    }

    /// The configuration in use.
    #[must_use]
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Scores `capabilities` against `query`, reporting where the scores came from.
    ///
    /// An empty `capabilities` list answers empty scores with
    /// [`StageProvenance::NoModel`] and logs nothing: nothing was asked of a model and
    /// nothing was degraded. With no model configured the lexical scorer answers and one
    /// `debug!` is recorded. A configured model that fails or answers unusably falls back
    /// to the lexical scorer with [`StageProvenance::DeterministicFallback`] and exactly
    /// one `warn!` under [`FallbackMode::Deterministic`]; under [`FallbackMode::Error`] the
    /// error is returned unchanged and no fallback is logged.
    ///
    /// # Errors
    ///
    /// Returns the model error when [`FallbackMode::Error`] is configured and the model
    /// call fails, and the lexical scorer's error if that computation fails.
    pub async fn score_with_provenance(
        &self,
        query: &str,
        capabilities: &[CapabilityDescriptor],
    ) -> Result<SemanticScores> {
        let deterministic = DeterministicScorer::new();
        if capabilities.is_empty() {
            return Ok(SemanticScores::new(
                BTreeMap::new(),
                StageProvenance::NoModel,
            ));
        }
        let Some(model) = self.config.model() else {
            note_no_model("semantic scorer", "the lexical scorer");
            let scores = deterministic.score(query, capabilities).await?;
            return Ok(SemanticScores::new(scores, StageProvenance::NoModel));
        };
        match self.score_with_model(model, query, capabilities).await {
            Ok(scores) => Ok(SemanticScores::new(scores, StageProvenance::Model)),
            Err(error) => match self.config.fallback {
                FallbackMode::Deterministic => {
                    note_fallback("semantic scorer", "the lexical scorer", &error);
                    let scores = deterministic.score(query, capabilities).await?;
                    Ok(SemanticScores::new(
                        scores,
                        StageProvenance::DeterministicFallback,
                    ))
                }
                FallbackMode::Error => Err(error),
            },
        }
    }

    /// Scores `capabilities` with the model, or reports why it could not.
    async fn score_with_model(
        &self,
        model: &dyn CompletionModel,
        query: &str,
        capabilities: &[CapabilityDescriptor],
    ) -> Result<BTreeMap<String, f64>> {
        let request = score_request(query, capabilities);
        let user = serde_json::to_string(&request).map_err(|error| {
            Error::llm(format!(
                "semantic scorer: request could not be serialized: {error}"
            ))
        })?;
        let reply = complete_json(Some(model), SYSTEM_PROMPT, &user).await?;
        let Some(value) = reply else {
            return Err(Error::llm(
                "semantic scorer: the model reply contained no JSON object",
            ));
        };
        let Some(judgements) = value.get("judgements").and_then(Value::as_array) else {
            return Err(Error::llm(
                "semantic scorer: the reply JSON has no `judgements` array",
            ));
        };
        let mut scores: BTreeMap<String, f64> = BTreeMap::new();
        for judgement in judgements {
            let Some(candidate_id) = judgement.get("candidate_id").and_then(Value::as_str) else {
                tracing::warn!("semantic scorer: dropping a judgement without a candidate_id");
                continue;
            };
            if !capabilities
                .iter()
                .any(|capability| capability.qualified_id() == candidate_id)
            {
                tracing::warn!(
                    candidate_id,
                    "semantic scorer: dropping a judgement for an unknown candidate"
                );
                continue;
            }
            let score = judgement
                .get("score")
                .and_then(Value::as_f64)
                .map_or(0.0, clamp_unit);
            if scores.contains_key(candidate_id) {
                tracing::warn!(
                    candidate_id,
                    "semantic scorer: keeping the first judgement for a repeated candidate"
                );
                continue;
            }
            scores.insert(candidate_id.to_string(), score);
        }
        Ok(scores)
    }
}

#[async_trait]
impl SemanticScorer for LlmSemanticScorer {
    async fn score(
        &self,
        query: &str,
        capabilities: &[CapabilityDescriptor],
    ) -> Result<BTreeMap<String, f64>> {
        Ok(self
            .score_with_provenance(query, capabilities)
            .await?
            .scores)
    }
}

/// Builds the compact scoring request.
fn score_request(query: &str, capabilities: &[CapabilityDescriptor]) -> Value {
    let candidates: Vec<Value> = capabilities
        .iter()
        .map(|capability| {
            let mut object = Map::new();
            object.insert(
                "candidate_id".to_string(),
                Value::String(capability.qualified_id()),
            );
            let name = capability.name.trim();
            if !name.is_empty() {
                object.insert("name".to_string(), Value::String(name.to_string()));
            }
            let description = truncate_text(&capability.description, DESCRIPTION_LIMIT).trim();
            if !description.is_empty() {
                object.insert(
                    "description".to_string(),
                    Value::String(description.to_string()),
                );
            }
            Value::Object(object)
        })
        .collect();
    let mut root = Map::new();
    root.insert("query".to_string(), Value::String(query.trim().to_string()));
    root.insert("candidates".to_string(), Value::Array(candidates));
    Value::Object(root)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::llm::StaticCompletionModel;

    #[test]
    fn score_request_lists_every_candidate_without_blank_fields() {
        let capabilities = vec![
            CapabilityDescriptor::new("tool", "a", "Extract"),
            CapabilityDescriptor::new("tool", "b", ""),
        ];
        let request = score_request("  summarize  ", &capabilities);
        assert_eq!(request["query"], "summarize");
        let candidates = request["candidates"].as_array().unwrap();
        assert_eq!(candidates.len(), 2);
        assert_eq!(candidates[0]["candidate_id"], "tool:a");
        assert_eq!(candidates[0]["name"], "Extract");
        assert!(candidates[1].get("name").is_none());
    }

    #[tokio::test]
    async fn deterministic_scorer_is_the_lexical_scorer() {
        // The alias must stay a pure re-export: a lexical scorer and the alias share the
        // same behaviour on a blank query (every capability scores 1.0).
        let capabilities = vec![CapabilityDescriptor::new("tool", "a", "Anything")];
        let deterministic = DeterministicScorer::new();
        let scores = deterministic.score("", &capabilities).await.unwrap();
        assert_eq!(scores.get("tool:a"), Some(&1.0));
    }

    fn capabilities() -> Vec<CapabilityDescriptor> {
        vec![
            CapabilityDescriptor::new("tool", "a", "Extract")
                .with_description("Extracts text from a document"),
            CapabilityDescriptor::new("tool", "b", "Summarize")
                .with_description("Summarizes extracted text"),
        ]
    }

    fn scripted(reply: &str) -> StaticCompletionModel {
        StaticCompletionModel::single(reply)
    }

    #[tokio::test]
    async fn no_model_reports_no_model_and_the_lexical_scores() {
        let capabilities = capabilities();
        let expected = DeterministicScorer::new()
            .score("extract text", &capabilities)
            .await
            .unwrap();
        let answer = LlmSemanticScorer::new(ModelConfig::new())
            .score_with_provenance("extract text", &capabilities)
            .await
            .unwrap();
        assert_eq!(answer.provenance, StageProvenance::NoModel);
        assert!(!answer.is_model_backed());
        assert_eq!(answer.scores, expected);
    }

    #[tokio::test]
    async fn an_empty_catalog_reports_no_model_without_a_model_call() {
        let model = Arc::new(scripted("{\"judgements\":[]}"));
        let scorer = LlmSemanticScorer::with_model(model.clone());
        let answer = scorer.score_with_provenance("anything", &[]).await.unwrap();
        assert!(answer.scores.is_empty());
        assert_eq!(answer.provenance, StageProvenance::NoModel);
        assert_eq!(model.call_count(), 0);
    }

    #[tokio::test]
    async fn a_scripted_model_reports_model_and_its_scores() {
        let model = Arc::new(scripted(
            "{\"judgements\":[{\"candidate_id\":\"tool:a\",\"score\":0.9},\
             {\"candidate_id\":\"tool:b\",\"score\":0.25},{\"candidate_id\":\"tool:ghost\"}]}",
        ));
        let scorer = LlmSemanticScorer::with_model(model.clone());
        let answer = scorer
            .score_with_provenance("extract text", &capabilities())
            .await
            .unwrap();
        assert_eq!(answer.provenance, StageProvenance::Model);
        assert!(answer.is_model_backed());
        assert_eq!(answer.scores.get("tool:a"), Some(&0.9));
        assert_eq!(answer.scores.get("tool:b"), Some(&0.25));
        assert!(!answer.scores.contains_key("tool:ghost"));
        assert_eq!(model.call_count(), 1);
    }

    #[tokio::test]
    async fn a_failing_model_falls_back_to_the_lexical_scorer() {
        struct Failing;
        #[async_trait]
        impl CompletionModel for Failing {
            fn name(&self) -> &'static str {
                "failing"
            }
            async fn complete(&self, _prompt: &str) -> Result<String> {
                Err(Error::llm("provider unavailable"))
            }
        }
        let capabilities = capabilities();
        let expected = DeterministicScorer::new()
            .score("extract text", &capabilities)
            .await
            .unwrap();
        let scorer = LlmSemanticScorer::with_model(Arc::new(Failing));
        let answer = scorer
            .score_with_provenance("extract text", &capabilities)
            .await
            .unwrap();
        assert_eq!(answer.provenance, StageProvenance::DeterministicFallback);
        assert!(answer.provenance.is_warning());
        assert_eq!(answer.scores, expected);
    }

    #[tokio::test]
    async fn an_error_mode_failing_model_surfaces_the_error() {
        struct Failing;
        #[async_trait]
        impl CompletionModel for Failing {
            fn name(&self) -> &'static str {
                "failing"
            }
            async fn complete(&self, _prompt: &str) -> Result<String> {
                Err(Error::llm("provider unavailable"))
            }
        }
        let scorer = LlmSemanticScorer::new(
            ModelConfig::new()
                .with_model(Arc::new(Failing))
                .with_fallback(FallbackMode::Error),
        );
        let error = scorer
            .score_with_provenance("extract text", &capabilities())
            .await
            .unwrap_err();
        assert!(
            error.to_string().contains("provider unavailable"),
            "{error}"
        );
        // The trait implementation surfaces the same error.
        assert!(scorer.score("extract text", &capabilities()).await.is_err());
    }

    #[tokio::test]
    async fn semantic_scores_round_trip_and_tolerate_a_legacy_payload() {
        let mut scores = BTreeMap::new();
        scores.insert("tool:a".to_string(), 0.5);
        let answer = SemanticScores::new(scores, StageProvenance::Model);
        let json = serde_json::to_value(&answer).unwrap();
        assert_eq!(json["provenance"], "model");
        assert_eq!(json["scores"]["tool:a"], 0.5);
        assert_eq!(
            serde_json::from_value::<SemanticScores>(json).unwrap(),
            answer
        );

        // A payload without `provenance` default to "no model".
        let legacy = serde_json::from_value::<SemanticScores>(
            serde_json::json!({"scores": {"tool:a": 0.25}, "future_field": true}),
        )
        .unwrap();
        assert_eq!(legacy.provenance, StageProvenance::NoModel);
        assert_eq!(legacy.scores.get("tool:a"), Some(&0.25));

        let empty = serde_json::from_value::<SemanticScores>(serde_json::json!({})).unwrap();
        assert_eq!(empty, SemanticScores::default());
    }
}
