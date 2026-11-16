//! Fingerprint extraction: model-authored semantic profiles.
//!
//! Python's `shared/fingerprint/extractor.py` asks a model to describe a capability and
//! turns the answer into the [`SemanticProfile`](crate::model::SemanticProfile) half of its fingerprint. This module
//! ports that stage onto the crate's [`CompletionModel`] seam.
//!
//! # Request and reply
//!
//! The user payload carries the card's identity, its input/output declarations, and its
//! classification/tags. The reply must be
//! `{"description":"…","semantic_profile":{"summary","capabilities","use_cases","limitations","keywords"},"classification":"…","tags":["…"]}`.
//! Only `semantic_profile` is read: the descriptor itself is authoritative for
//! classification and tags, so a model cannot rewrite the card. Lists are deduplicated
//! first-wins and blank entries are dropped, and
//! [`keywords`](crate::model::SemanticProfile) is the card's tags plus the model's keywords.
//!
//! # The deterministic fallback
//!
//! Every field the model leaves out keeps its card-derived value: `summary` is the
//! description (or the name when the description is blank), `capabilities` is the
//! qualified id, `use_cases` and `keywords` are the card's tags, and `limitations` is
//! empty. Under the default [`FallbackMode::Deterministic`], a missing model, a failed
//! call, or an unusable reply therefore always produces a complete profile;
//! [`FallbackMode::Error`] opts into surfacing the call failure or the unusable reply
//! instead.
//!
//! # Provenance
//!
//! [`extract_with_provenance`](crate::llm::fingerprint::FingerprintExtractor::extract_with_provenance)
//! returns the profile together with a [`StageProvenance`]:
//!
//! | Path | Provenance | Log record |
//! | --- | --- | --- |
//! | no model configured | `StageProvenance::NoModel` | one `tracing::debug!` |
//! | the reply carried a usable `semantic_profile` object | `StageProvenance::Model` | none |
//! | the reply carried no usable `semantic_profile` | `StageProvenance::DeterministicFallback` | one `tracing::warn!` naming the stage and the card-derived profile |
//! | the model call failed under [`FallbackMode::Deterministic`] | `StageProvenance::DeterministicFallback` | exactly one `tracing::warn!` naming the stage and the card-derived profile |
//! | the model call failed under [`FallbackMode::Error`] | — | the typed `Error::Llm` is returned instead |
//!
//! Provenance describes where the *profile* came from, not whether every field is
//! model-authored: the overlay keeps the card-derived value of any field the model
//! omitted, so a `Model` profile is always complete.

use std::fmt;
use std::sync::Arc;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::model::{CapabilityDescriptor, CapabilityIo, SemanticProfile};

use super::{
    complete_json, dedupe_strings, note_fallback, note_no_model, string_field, string_list,
    truncate_text, CompletionModel, FallbackMode, StageProvenance,
};

/// How many characters of a description are sent to the model.
const DESCRIPTION_LIMIT: usize = 480;
/// The system instruction sent with every extraction request.
const SYSTEM_PROMPT: &str = "Extract a capability fingerprint. Return JSON only: \
{\"description\":\"…\",\"semantic_profile\":{\"summary\":\"…\",\"capabilities\":[\"…\"],\"use_cases\":[\"…\"],\"limitations\":[\"…\"],\"keywords\":[\"…\"]},\"classification\":\"…\",\"tags\":[\"…\"]}. \
The summary is one or two sentences; keywords are short lowercase terms.";

/// Tuning knobs for a [`FingerprintExtractor`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct FingerprintConfig {
    /// What to do when the model fails or answers unusably.
    pub fallback: FallbackMode,
}

impl FingerprintConfig {
    /// Builds the default configuration ([`FallbackMode::Deterministic`]).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the fallback mode.
    #[must_use]
    pub fn with_fallback(mut self, fallback: FallbackMode) -> Self {
        self.fallback = fallback;
        self
    }
}

/// The profile one extraction produced plus where it came from.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct FingerprintOutcome {
    /// The semantic profile (model-authored, card-derived, or a blend).
    pub profile: SemanticProfile,
    /// Where the profile came from.
    pub provenance: StageProvenance,
}

impl FingerprintOutcome {
    /// Builds an extraction outcome.
    #[must_use]
    pub fn new(profile: SemanticProfile, provenance: StageProvenance) -> Self {
        Self {
            profile,
            provenance,
        }
    }

    /// Whether a model produced the profile.
    #[must_use]
    pub fn is_model_backed(&self) -> bool {
        self.provenance.is_model()
    }
}

/// Extracts [`SemanticProfile`]s, with a model when one is configured.
#[derive(Clone, Default)]
pub struct FingerprintExtractor {
    /// The completion model; `None` keeps the card-derived profile.
    model: Option<Arc<dyn CompletionModel>>,
    /// The fallback behaviour.
    config: FingerprintConfig,
}

impl FingerprintExtractor {
    /// Builds an extractor with no model and [`FallbackMode::Deterministic`].
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

    /// Replaces the configuration.
    #[must_use]
    pub fn with_config(mut self, config: FingerprintConfig) -> Self {
        self.config = config;
        self
    }

    /// The configuration in use.
    #[must_use]
    pub fn config(&self) -> &FingerprintConfig {
        &self.config
    }

    /// Whether a model is configured.
    #[must_use]
    pub fn has_model(&self) -> bool {
        self.model.is_some()
    }

    /// Derives the semantic profile of `descriptor`.
    ///
    /// # Errors
    ///
    /// Returns the model error when [`FallbackMode::Error`] is configured and the model
    /// call fails; under the default [`FallbackMode::Deterministic`] extraction always
    /// succeeds with the card-derived profile.
    pub async fn extract(&self, descriptor: &CapabilityDescriptor) -> Result<SemanticProfile> {
        Ok(self.extract_with_provenance(descriptor).await?.profile)
    }

    /// Derives the semantic profile of `descriptor`, reporting where the profile came
    /// from.
    ///
    /// With no model configured the card-derived profile answers with
    /// [`StageProvenance::NoModel`] and one `debug!` is recorded. A reply carrying a usable
    /// `semantic_profile` object is [`StageProvenance::Model`] (the overlay keeps the
    /// card-derived value of any field the model omitted, so the profile is always
    /// complete). A reply without a usable `semantic_profile`, and a failed model call
    /// under [`FallbackMode::Deterministic`], answer the card-derived profile with
    /// [`StageProvenance::DeterministicFallback`] and exactly one `warn!` naming the stage
    /// and the fallback rule. Under [`FallbackMode::Error`] the failure is returned
    /// unchanged and no fallback is logged.
    ///
    /// # Errors
    ///
    /// Returns the model error when [`FallbackMode::Error`] is configured and the model
    /// call fails (or the reply carries no usable JSON object), and the request
    /// serialization error for a descriptor that cannot be rendered.
    pub async fn extract_with_provenance(
        &self,
        descriptor: &CapabilityDescriptor,
    ) -> Result<FingerprintOutcome> {
        let fallback = fallback_profile(descriptor);
        let Some(model) = self.model.as_deref() else {
            note_no_model("fingerprint extractor", "the card-derived profile");
            return Ok(FingerprintOutcome::new(fallback, StageProvenance::NoModel));
        };
        let failure = match complete_json(Some(model), SYSTEM_PROMPT, &request_json(descriptor)?)
            .await
        {
            Ok(Some(value)) => {
                return Ok(
                    match profile_from_reply(&value, descriptor, fallback.clone()) {
                        Some(profile) => FingerprintOutcome::new(profile, StageProvenance::Model),
                        None => FingerprintOutcome::new(
                            fallback,
                            StageProvenance::DeterministicFallback,
                        ),
                    },
                )
            }
            Ok(None) => crate::error::Error::llm(
                "fingerprint extractor: the model reply contained no usable JSON object",
            ),
            Err(error) => error,
        };
        match self.config.fallback {
            FallbackMode::Deterministic => {
                note_fallback(
                    "fingerprint extractor",
                    "the card-derived profile",
                    &failure,
                );
                Ok(FingerprintOutcome::new(
                    fallback,
                    StageProvenance::DeterministicFallback,
                ))
            }
            FallbackMode::Error => Err(failure),
        }
    }
}

impl fmt::Debug for FingerprintExtractor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("FingerprintExtractor")
            .field("model", &self.model.as_deref().map(CompletionModel::name))
            .field("config", &self.config)
            .finish()
    }
}

/// The profile derived from the card alone.
fn fallback_profile(descriptor: &CapabilityDescriptor) -> SemanticProfile {
    let summary = match descriptor.description.trim() {
        "" => descriptor.name.clone(),
        description => description.to_string(),
    };
    let tags = dedupe_strings(descriptor.tags.iter().cloned());
    SemanticProfile {
        summary,
        capabilities: vec![descriptor.qualified_id()],
        use_cases: tags.clone(),
        limitations: Vec::new(),
        keywords: tags,
    }
}

/// Builds the compact extraction request.
fn request_json(descriptor: &CapabilityDescriptor) -> Result<String> {
    let request = serde_json::json!({
        "capability_id": descriptor.capability_id,
        "capability_type": descriptor.capability_type,
        "name": descriptor.name,
        "description": truncate_text(&descriptor.description, DESCRIPTION_LIMIT),
        "inputs": io_values(&descriptor.inputs),
        "outputs": io_values(&descriptor.outputs),
        "classification": descriptor.classification,
        "tags": descriptor.tags,
    });
    serde_json::to_string(&request).map_err(|error| {
        crate::error::Error::llm(format!(
            "fingerprint extractor: request could not be serialized: {error}"
        ))
    })
}

/// The `{name, type, description}` view of one IO list.
fn io_values(ios: &[CapabilityIo]) -> Vec<serde_json::Value> {
    ios.iter()
        .map(|io| {
            serde_json::json!({
                "name": io.name,
                "type": io.io_type,
                "description": truncate_text(&io.description, DESCRIPTION_LIMIT),
            })
        })
        .collect()
}

/// Overlays the model's `semantic_profile` on the card-derived profile.
///
/// Answers `None` — after the module's single `warn!` — when the reply carries no usable
/// `semantic_profile` object, so the caller can report
/// [`StageProvenance::DeterministicFallback`] rather than claiming a model answer.
fn profile_from_reply(
    value: &serde_json::Value,
    descriptor: &CapabilityDescriptor,
    fallback: SemanticProfile,
) -> Option<SemanticProfile> {
    let Some(profile) = value
        .get("semantic_profile")
        .filter(|profile| profile.is_object())
    else {
        tracing::warn!(
            "fingerprint extractor: the reply JSON has no `semantic_profile` object; using the card-derived profile"
        );
        return None;
    };
    let summary = string_field(profile, "summary").unwrap_or(fallback.summary);
    let capabilities = non_empty(
        string_list(profile.get("capabilities")),
        fallback.capabilities,
    );
    let use_cases = non_empty(string_list(profile.get("use_cases")), fallback.use_cases);
    let limitations = non_empty(
        string_list(profile.get("limitations")),
        fallback.limitations,
    );
    let keywords = dedupe_strings(
        descriptor
            .tags
            .iter()
            .cloned()
            .chain(string_list(profile.get("keywords"))),
    );
    let keywords = non_empty(keywords, fallback.keywords);
    Some(SemanticProfile {
        summary,
        capabilities,
        use_cases,
        limitations,
        keywords,
    })
}

/// Answers `model` when it is non-empty, otherwise `fallback`.
fn non_empty(model: Vec<String>, fallback: Vec<String>) -> Vec<String> {
    if model.is_empty() {
        fallback
    } else {
        model
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
#[allow(clippy::assert_is_empty)]
mod tests {
    use super::*;
    use crate::llm::StaticCompletionModel;

    fn descriptor() -> CapabilityDescriptor {
        CapabilityDescriptor::new("tool", "extract", "Extract")
            .with_description("Extracts text")
            .with_classification("document")
            .with_tags(vec![
                "text".to_string(),
                "  ".to_string(),
                "text".to_string(),
            ])
            .with_inputs(vec![CapabilityIo::new("path", "string")])
    }

    #[test]
    fn fallback_profile_uses_the_description_and_tags() {
        let profile = fallback_profile(&descriptor());
        assert_eq!(profile.summary, "Extracts text");
        assert_eq!(profile.capabilities, vec!["tool:extract".to_string()]);
        assert_eq!(profile.use_cases, vec!["text".to_string()]);
        assert!(profile.limitations.is_empty());
        assert_eq!(profile.keywords, vec!["text".to_string()]);
    }

    #[test]
    fn fallback_profile_uses_the_name_without_a_description() {
        let profile = fallback_profile(&CapabilityDescriptor::new("tool", "a", "Named"));
        assert_eq!(profile.summary, "Named");
    }

    #[test]
    fn profile_from_reply_overlays_only_present_fields() {
        let value = serde_json::json!({
            "semantic_profile": {
                "summary": "  A better summary  ",
                "capabilities": ["tool:extract", "tool:extract", ""],
                "keywords": ["document", "text"]
            }
        });
        let descriptor = descriptor();
        let profile = profile_from_reply(&value, &descriptor, fallback_profile(&descriptor))
            .expect("the reply carries a semantic_profile object");
        assert_eq!(profile.summary, "A better summary");
        assert_eq!(profile.capabilities, vec!["tool:extract".to_string()]);
        // use_cases and limitations keep their card-derived values.
        assert_eq!(profile.use_cases, vec!["text".to_string()]);
        assert!(profile.limitations.is_empty());
        // Tags come first, then the model's keywords, deduplicated.
        assert_eq!(
            profile.keywords,
            vec!["text".to_string(), "document".to_string()]
        );
    }

    #[test]
    fn profile_from_reply_without_a_profile_answers_none() {
        let descriptor = descriptor();
        let value = serde_json::json!({"description": "ignored"});
        assert!(profile_from_reply(&value, &descriptor, fallback_profile(&descriptor)).is_none());
    }

    #[test]
    fn request_json_carries_the_card_without_the_semantic_content() {
        let request = request_json(&descriptor()).unwrap();
        assert!(request.contains("\"capability_type\":\"tool\""));
        assert!(request.contains("\"inputs\""));
        assert!(!request.contains("semantic_content"));
    }

    #[tokio::test]
    async fn no_model_reports_no_model_and_the_card_profile() {
        let descriptor = descriptor();
        let outcome = FingerprintExtractor::new()
            .extract_with_provenance(&descriptor)
            .await
            .unwrap();
        assert_eq!(outcome.provenance, StageProvenance::NoModel);
        assert!(!outcome.is_model_backed());
        assert_eq!(outcome.profile, fallback_profile(&descriptor));
        // `extract` keeps its signature and answers the same profile.
        let profile = FingerprintExtractor::new()
            .extract(&descriptor)
            .await
            .unwrap();
        assert_eq!(profile, outcome.profile);
    }

    #[tokio::test]
    async fn a_scripted_model_reports_model_and_its_profile() {
        let model = Arc::new(StaticCompletionModel::single(
            "{\"semantic_profile\":{\"summary\":\"A model summary\",\"keywords\":[\"document\"]}}",
        ));
        let extractor = FingerprintExtractor::new().with_model(model.clone());
        assert!(extractor.has_model());
        let outcome = extractor
            .extract_with_provenance(&descriptor())
            .await
            .unwrap();
        assert_eq!(outcome.provenance, StageProvenance::Model);
        assert!(outcome.is_model_backed());
        assert_eq!(outcome.profile.summary, "A model summary");
        assert_eq!(model.call_count(), 1);
    }

    #[tokio::test]
    async fn a_reply_without_a_semantic_profile_reports_the_fallback() {
        let model = Arc::new(StaticCompletionModel::single(
            "{\"description\":\"ignored\"}",
        ));
        let descriptor = descriptor();
        let outcome = FingerprintExtractor::new()
            .with_model(model)
            .extract_with_provenance(&descriptor)
            .await
            .unwrap();
        assert_eq!(outcome.provenance, StageProvenance::DeterministicFallback);
        assert!(outcome.provenance.is_warning());
        assert_eq!(outcome.profile, fallback_profile(&descriptor));
    }

    #[tokio::test]
    async fn a_failing_model_falls_back_to_the_card_profile() {
        struct Failing;
        #[async_trait::async_trait]
        impl CompletionModel for Failing {
            fn name(&self) -> &'static str {
                "failing"
            }
            async fn complete(&self, _prompt: &str) -> Result<String> {
                Err(crate::error::Error::llm("provider unavailable"))
            }
        }
        let descriptor = descriptor();
        let outcome = FingerprintExtractor::new()
            .with_model(Arc::new(Failing))
            .extract_with_provenance(&descriptor)
            .await
            .unwrap();
        assert_eq!(outcome.provenance, StageProvenance::DeterministicFallback);
        assert_eq!(outcome.profile, fallback_profile(&descriptor));
    }

    #[tokio::test]
    async fn an_error_mode_failing_model_surfaces_the_error() {
        struct Failing;
        #[async_trait::async_trait]
        impl CompletionModel for Failing {
            fn name(&self) -> &'static str {
                "failing"
            }
            async fn complete(&self, _prompt: &str) -> Result<String> {
                Err(crate::error::Error::llm("provider unavailable"))
            }
        }
        let extractor = FingerprintExtractor::new()
            .with_model(Arc::new(Failing))
            .with_config(FingerprintConfig::new().with_fallback(FallbackMode::Error));
        let error = extractor
            .extract_with_provenance(&descriptor())
            .await
            .unwrap_err();
        assert!(
            error.to_string().contains("provider unavailable"),
            "{error}"
        );
        // The legacy entry point surfaces the same error.
        assert!(extractor.extract(&descriptor()).await.is_err());
    }

    #[tokio::test]
    async fn fingerprint_outcome_round_trips_and_tolerates_a_legacy_payload() {
        let outcome = FingerprintOutcome::new(
            fallback_profile(&descriptor()),
            StageProvenance::DeterministicFallback,
        );
        let json = serde_json::to_value(&outcome).unwrap();
        assert_eq!(json["provenance"], "deterministic_fallback");
        assert_eq!(
            serde_json::from_value::<FingerprintOutcome>(json).unwrap(),
            outcome
        );

        // A payload without `provenance` defaults to "no model".
        let legacy = serde_json::from_value::<FingerprintOutcome>(serde_json::json!({
            "profile": {"summary": "Legacy", "future_field": 1},
            "extra": true
        }))
        .unwrap();
        assert_eq!(legacy.provenance, StageProvenance::NoModel);
        assert_eq!(legacy.profile.summary, "Legacy");
        assert!(legacy.profile.capabilities.is_empty());

        let empty = serde_json::from_value::<FingerprintOutcome>(serde_json::json!({})).unwrap();
        assert_eq!(empty, FingerprintOutcome::default());
    }
}
