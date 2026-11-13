//! Per-scope analyzer overrides.
//!
//! The Python `MemoryScopeConfig` lets one scope carry its own `user_profile_definition`
//! /`semantic_memory_definition`/`episodic_memory_definition`, and the memory manager can
//! resolve a different chat model per scope. This module is the analyzer-side half of that
//! contract: a plain value holding the handles and thresholds one scope overrides, plus
//! [`ScopeAnalyzerOverrides::effective_config`] which folds them onto a base
//! [`AnalyzerConfig`].
//!
//! The analyzer never calls the embedding handle — extraction and conflict classification
//! are text-only — but it carries it so the caller that owns storage resolution (engine or
//! facade) can resolve the model and embedding for one scope together, in one place,
//! without a second lookup.
//!
//! # Wiring still required outside this module
//!
//! The facade/engine must (a) store a `ScopeAnalyzerOverrides` per `scope_id`, (b) build
//! the analyzer/pipeline for a call with [`ScopeAnalyzerOverrides::clone`] applied via
//! `StagedAnalyzer::with_overrides`, and (c) pass the resolved embedding handle to the
//! engine's write path. Until that wiring lands, callers pass the overrides explicitly.
use crate::analyzer::config::{AnalyzerConfig, ClassifierKind, ExtractorKind};
use crate::analyzer::stage::pipeline::StageConfig;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::llm::Model;
use openjiuwen_retrieval::embedding::Embedding;
use std::fmt;
use std::sync::Arc;

/// The model, embedding, kinds, and thresholds one scope overrides.
///
/// Every field is optional: `None` means "inherit the base configuration". The type has a
/// manual [`fmt::Debug`] that reports handle *names* only, so logging an override never
/// leaks a provider client or credentials.
#[derive(Clone, Default)]
pub struct ScopeAnalyzerOverrides {
    /// The model this scope uses for the model-backed stages and extractor.
    model: Option<Arc<dyn Model>>,
    /// The embedding model this scope uses (carried, never called by the analyzer).
    embedding: Option<Arc<dyn Embedding>>,
    /// Extractor kind override.
    extractor: Option<ExtractorKind>,
    /// Classifier kind override.
    classifier: Option<ClassifierKind>,
    /// Minimum fact importance override (facts below it are dropped).
    min_importance: Option<f32>,
    /// Maximum facts per extraction override.
    max_facts: Option<usize>,
    /// Default importance override for facts that carry none.
    default_importance: Option<f32>,
    /// Stage selection override (plan/reasoning/proactive kinds and thresholds).
    stages: Option<StageConfig>,
}

impl ScopeAnalyzerOverrides {
    /// Builds an empty override set (inherit everything).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the overrides with `model` set.
    #[must_use]
    pub fn with_model(mut self, model: Arc<dyn Model>) -> Self {
        self.model = Some(model);
        self
    }

    /// Returns the overrides with `embedding` set.
    #[must_use]
    pub fn with_embedding(mut self, embedding: Arc<dyn Embedding>) -> Self {
        self.embedding = Some(embedding);
        self
    }

    /// Returns the overrides with the extractor kind set.
    #[must_use]
    pub fn with_extractor(mut self, extractor: ExtractorKind) -> Self {
        self.extractor = Some(extractor);
        self
    }

    /// Returns the overrides with the classifier kind set.
    #[must_use]
    pub fn with_classifier(mut self, classifier: ClassifierKind) -> Self {
        self.classifier = Some(classifier);
        self
    }

    /// Returns the overrides with the minimum importance set.
    #[must_use]
    pub fn with_min_importance(mut self, min_importance: f32) -> Self {
        self.min_importance = Some(min_importance);
        self
    }

    /// Returns the overrides with the fact cap set.
    #[must_use]
    pub fn with_max_facts(mut self, max_facts: usize) -> Self {
        self.max_facts = Some(max_facts);
        self
    }

    /// Returns the overrides with the default importance set.
    #[must_use]
    pub fn with_default_importance(mut self, default_importance: f32) -> Self {
        self.default_importance = Some(default_importance);
        self
    }

    /// Returns the overrides with the stage selection set.
    #[must_use]
    pub fn with_stages(mut self, stages: StageConfig) -> Self {
        self.stages = Some(stages);
        self
    }

    /// Returns the model handle, when overridden.
    #[must_use]
    pub fn model(&self) -> Option<&Arc<dyn Model>> {
        self.model.as_ref()
    }

    /// Returns the embedding handle, when overridden.
    #[must_use]
    pub fn embedding(&self) -> Option<&Arc<dyn Embedding>> {
        self.embedding.as_ref()
    }

    /// Returns the extractor kind override.
    #[must_use]
    pub fn extractor(&self) -> Option<ExtractorKind> {
        self.extractor
    }

    /// Returns the classifier kind override.
    #[must_use]
    pub fn classifier(&self) -> Option<ClassifierKind> {
        self.classifier
    }

    /// Returns the minimum importance override.
    #[must_use]
    pub fn min_importance(&self) -> Option<f32> {
        self.min_importance
    }

    /// Returns the fact cap override.
    #[must_use]
    pub fn max_facts(&self) -> Option<usize> {
        self.max_facts
    }

    /// Returns the default importance override.
    #[must_use]
    pub fn default_importance(&self) -> Option<f32> {
        self.default_importance
    }

    /// Returns the stage selection override.
    #[must_use]
    pub fn stages(&self) -> Option<&StageConfig> {
        self.stages.as_ref()
    }

    /// Returns `true` when nothing is overridden.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.model.is_none()
            && self.embedding.is_none()
            && self.extractor.is_none()
            && self.classifier.is_none()
            && self.min_importance.is_none()
            && self.max_facts.is_none()
            && self.default_importance.is_none()
            && self.stages.is_none()
    }

    /// Validates the numeric overrides.
    ///
    /// Non-finite or out-of-`0.0..=1.0` importances and a zero `max_facts` are
    /// `Error::Validation` naming the field and value; a stage override is validated too.
    pub fn validate(&self) -> Result<()> {
        for (field, value) in [
            ("min_importance", self.min_importance),
            ("default_importance", self.default_importance),
        ] {
            if let Some(value) = value {
                if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                    return Err(Error::validation(format!(
                        "scope override `{field}` must be within 0.0..=1.0 (got {value})"
                    )));
                }
            }
        }
        if let Some(max_facts) = self.max_facts {
            if max_facts == 0 {
                return Err(Error::validation(
                    "scope override `max_facts` must be greater than zero (got 0)",
                ));
            }
        }
        if let Some(stages) = &self.stages {
            stages.validate()?;
        }
        Ok(())
    }

    /// Folds the overrides onto `base`.
    ///
    /// Overridden kinds and thresholds replace the base values; everything else is
    /// inherited. The result is not validated here — [`ScopeAnalyzerOverrides::validate`]
    /// covers the override values and the pipeline validates the merged config.
    #[must_use]
    pub fn effective_config(&self, base: &AnalyzerConfig) -> AnalyzerConfig {
        let mut config = base.clone();
        if let Some(extractor) = self.extractor {
            config.extractor = extractor;
        }
        if let Some(classifier) = self.classifier {
            config.classifier = classifier;
        }
        if let Some(min_importance) = self.min_importance {
            config.min_importance = min_importance;
        }
        if let Some(max_facts) = self.max_facts {
            config.max_facts = max_facts;
        }
        if let Some(default_importance) = self.default_importance {
            config.default_importance = default_importance;
        }
        config
    }
}

impl fmt::Debug for ScopeAnalyzerOverrides {
    /// Reports handle names and thresholds only.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let model = self.model.as_ref().map(|model| model.model_name());
        let embedding = self
            .embedding
            .as_ref()
            .map(|embedding| embedding.model_name());
        formatter
            .debug_struct("ScopeAnalyzerOverrides")
            .field("model", &model)
            .field("embedding", &embedding)
            .field("extractor", &self.extractor)
            .field("classifier", &self.classifier)
            .field("min_importance", &self.min_importance)
            .field("max_facts", &self.max_facts)
            .field("default_importance", &self.default_importance)
            .field("stages", &self.stages)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use openjiuwen_retrieval::embedding::HashEmbedding;
    use openjiuwen_test_support::ScriptedModel;

    fn model() -> Arc<dyn Model> {
        Arc::new(ScriptedModel::builder().build())
    }

    #[test]
    fn empty_overrides_inherit_everything() {
        let base = AnalyzerConfig::default();
        let overrides = ScopeAnalyzerOverrides::new();
        assert!(overrides.is_empty());
        assert_eq!(overrides.effective_config(&base), base);
        assert!(overrides.validate().is_ok());
        assert_eq!(
            format!("{overrides:?}"),
            "ScopeAnalyzerOverrides { model: None, embedding: None, extractor: None, \
             classifier: None, min_importance: None, max_facts: None, \
             default_importance: None, stages: None }"
        );
    }

    #[test]
    fn overrides_replace_kinds_and_thresholds() {
        let base = AnalyzerConfig::default();
        let overrides = ScopeAnalyzerOverrides::new()
            .with_model(model())
            .with_embedding(Arc::new(HashEmbedding::new(8)))
            .with_extractor(ExtractorKind::Llm)
            .with_classifier(ClassifierKind::Llm)
            .with_min_importance(0.8)
            .with_max_facts(3)
            .with_default_importance(0.55)
            .with_stages(StageConfig {
                plan: crate::analyzer::stage::StageKind::Llm,
                ..StageConfig::default()
            });
        assert!(!overrides.is_empty());
        assert!(overrides.validate().is_ok());
        let effective = overrides.effective_config(&base);
        assert_eq!(effective.extractor, ExtractorKind::Llm);
        assert_eq!(effective.classifier, ClassifierKind::Llm);
        assert!((effective.min_importance - 0.8).abs() < f32::EPSILON);
        assert_eq!(effective.max_facts, 3);
        assert!((effective.default_importance - 0.55).abs() < f32::EPSILON);
        // The base config is untouched.
        assert_eq!(base.extractor, ExtractorKind::Rule);
        assert!((base.min_importance - 0.0).abs() < f32::EPSILON);
        assert_eq!(overrides.extractor(), Some(ExtractorKind::Llm));
        assert_eq!(overrides.classifier(), Some(ClassifierKind::Llm));
        assert_eq!(overrides.min_importance(), Some(0.8));
        assert_eq!(overrides.max_facts(), Some(3));
        assert_eq!(overrides.default_importance(), Some(0.55));
        assert!(overrides.stages().is_some());
        assert!(overrides.model().is_some());
        assert!(overrides.embedding().is_some());
    }

    #[test]
    fn debug_reports_handle_names_only() {
        let overrides = ScopeAnalyzerOverrides::new()
            .with_model(model())
            .with_embedding(Arc::new(HashEmbedding::new(8)));
        let rendered = format!("{overrides:?}");
        assert!(rendered.contains("model: Some("));
        assert!(rendered.contains("embedding: Some("));
    }

    #[test]
    fn invalid_overrides_are_rejected() {
        for overrides in [
            ScopeAnalyzerOverrides::new().with_min_importance(1.5),
            ScopeAnalyzerOverrides::new().with_min_importance(f32::NAN),
            ScopeAnalyzerOverrides::new().with_default_importance(-0.1),
            ScopeAnalyzerOverrides::new().with_max_facts(0),
            ScopeAnalyzerOverrides::new().with_stages(StageConfig {
                reasoning_key: "  ".to_owned(),
                ..StageConfig::default()
            }),
        ] {
            let error = overrides.validate().unwrap_err();
            assert!(matches!(error, Error::Validation(_)), "{error:?}");
        }
    }
}
