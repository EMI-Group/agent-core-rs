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
use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Map, Value};

use crate::error::{Error, Result};
use crate::model::CapabilityDescriptor;
use crate::orchestration::SemanticScorer;

use super::{clamp_unit, complete_json, truncate_text, CompletionModel, FallbackMode, ModelConfig};

/// The deterministic lexical scorer, re-exported under the name this module's callers
/// use for it.
pub use crate::orchestration::LexicalSemanticScorer as DeterministicScorer;

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
        let deterministic = DeterministicScorer::new();
        if capabilities.is_empty() {
            return Ok(BTreeMap::new());
        }
        let Some(model) = self.config.model() else {
            return deterministic.score(query, capabilities).await;
        };
        match self.score_with_model(model, query, capabilities).await {
            Ok(scores) => Ok(scores),
            Err(error) => match self.config.fallback {
                FallbackMode::Deterministic => {
                    tracing::warn!(
                        %error,
                        "semantic scorer: model scoring failed; using the lexical scorer"
                    );
                    deterministic.score(query, capabilities).await
                }
                FallbackMode::Error => Err(error),
            },
        }
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
}
