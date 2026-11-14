//! Per-scope analyzer and embedding overrides, persisted beside the scope config.
//!
//! Python's `MemoryScopeConfig` lets one scope carry its own memory definitions, and the
//! Python memory manager resolves a different chat model per scope. The Rust facade keeps
//! that contract in two layers:
//!
//! - [`ScopeAnalyzerOverrideSpec`] is the *serializable document* stored under
//!   `memory_scope_overrides/{scope_id}` — names, kinds, and thresholds only;
//! - [`ScopeAnalyzerOverrides`] is the *runtime value* the analyzer pipeline consumes; it
//!   holds `Arc<dyn Model>`/`Arc<dyn Embedding>` handles and is deliberately not
//!   serializable.
//!
//! [`ScopeAnalyzerOverrideSpec::resolve`] is the bridge: it resolves the document against
//! the handles the facade actually owns ([`ScopeOverrideHandles`]) and fails loudly when a
//! name cannot be satisfied. The engine's embedding model is fixed at construction, so a
//! scope naming a *different* embedding is [`Error::Unsupported`] naming the requested
//! handle — never a silent fallback to the engine's model.
//!
//! The stored document is a *sibling* of the scope config, not part of it:
//! [`LongTermMemory::delete_scope_config`](super::LongTermMemory::delete_scope_config)
//! keeps its signature and its behaviour (it clears `memory_scope_config/{scope_id}` only),
//! and [`LongTermMemory::delete_scope_overrides`] clears this namespace. Deleting one never
//! touches the other.
use std::fmt;
use std::sync::Arc;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::llm::Model;
use openjiuwen_retrieval::embedding::Embedding;
use serde::{Deserialize, Serialize};

use crate::analyzer::{ClassifierKind, ExtractorKind, ScopeAnalyzerOverrides, StageConfig};
use crate::record::{MemoryScope, DEFAULT_ID};

use super::LongTermMemory;

/// The tracing target of facade diagnostics.
const TARGET: &str = "openjiuwen_memory::facade";

/// The serializable per-scope analyzer override document.
///
/// Every field is optional: `None` means "inherit the engine configuration", so an
/// all-`None` spec is exactly [`ScopeAnalyzerOverrideSpec::is_empty`] and changes nothing.
/// JSON is `snake_case`, tolerant of unknown fields, and every absent field falls back to
/// `None`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ScopeAnalyzerOverrideSpec {
    /// Extractor kind this scope uses (`rule`/`llm`).
    pub extractor: Option<ExtractorKind>,
    /// Conflict classifier kind this scope uses (`rule`/`llm`).
    pub classifier: Option<ClassifierKind>,
    /// Facts below this importance are dropped for this scope.
    pub min_importance: Option<f32>,
    /// Maximum facts one extraction returns for this scope.
    pub max_facts: Option<usize>,
    /// Importance assigned to a fact that carries none, for this scope.
    pub default_importance: Option<f32>,
    /// Stage selection and proactive thresholds for this scope.
    pub stages: Option<StageConfig>,
    /// Name of the model this scope asks for (resolved against the facade's handles).
    pub model: Option<String>,
    /// Name of the embedding model this scope asks for.
    pub embedding: Option<String>,
}

impl ScopeAnalyzerOverrideSpec {
    /// Builds an empty spec (inherit everything).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the spec with the extractor kind set.
    #[must_use]
    pub fn with_extractor(mut self, extractor: ExtractorKind) -> Self {
        self.extractor = Some(extractor);
        self
    }

    /// Returns the spec with the classifier kind set.
    #[must_use]
    pub fn with_classifier(mut self, classifier: ClassifierKind) -> Self {
        self.classifier = Some(classifier);
        self
    }

    /// Returns the spec with the minimum importance set.
    #[must_use]
    pub fn with_min_importance(mut self, min_importance: f32) -> Self {
        self.min_importance = Some(min_importance);
        self
    }

    /// Returns the spec with the fact cap set.
    #[must_use]
    pub fn with_max_facts(mut self, max_facts: usize) -> Self {
        self.max_facts = Some(max_facts);
        self
    }

    /// Returns the spec with the default importance set.
    #[must_use]
    pub fn with_default_importance(mut self, default_importance: f32) -> Self {
        self.default_importance = Some(default_importance);
        self
    }

    /// Returns the spec with the stage selection set.
    #[must_use]
    pub fn with_stages(mut self, stages: StageConfig) -> Self {
        self.stages = Some(stages);
        self
    }

    /// Returns the spec with a requested model name set.
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Returns the spec with a requested embedding name set.
    #[must_use]
    pub fn with_embedding(mut self, embedding: impl Into<String>) -> Self {
        self.embedding = Some(embedding.into());
        self
    }

    /// Returns `true` when nothing is overridden.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.extractor.is_none()
            && self.classifier.is_none()
            && self.min_importance.is_none()
            && self.max_facts.is_none()
            && self.default_importance.is_none()
            && self.stages.is_none()
            && self.model.is_none()
            && self.embedding.is_none()
    }

    /// Validates the document's shape, without resolving the handle names.
    ///
    /// The scalar overrides are validated by the analyzer's own
    /// [`ScopeAnalyzerOverrides::validate`] (non-finite or out-of-`0.0..=1.0`
    /// importances, a zero `max_facts`, an invalid stage config), and a blank `model` /
    /// `embedding` name is [`Error::Validation`] too. Whether a name can actually be
    /// satisfied is a *resolution* question, answered per call by
    /// [`ScopeAnalyzerOverrideSpec::resolve`].
    pub fn validate(&self) -> Result<()> {
        self.scalars().validate()?;
        for (field, value) in [
            ("model", self.model.as_deref()),
            ("embedding", self.embedding.as_deref()),
        ] {
            if let Some(value) = value {
                if value.trim().is_empty() {
                    return Err(Error::validation(format!(
                        "scope override `{field}` must not be blank"
                    )));
                }
            }
        }
        Ok(())
    }

    /// Resolves the document against the handles `facade` actually holds.
    ///
    /// The scalar overrides and the stage selection are copied; a named `model` /
    /// `embedding` is resolved to a handle. `model: None`/`embedding: None` means "use the
    /// engine's" and adds no handle.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`] for an invalid document (see
    /// [`ScopeAnalyzerOverrideSpec::validate`]).
    ///
    /// [`Error::Unsupported`] when a named handle cannot be satisfied, naming the
    /// requested handle and the reason: no model is attached, the facade only holds a
    /// differently named model, or a per-scope embedding was requested while the engine's
    /// embedding model is fixed at construction. A name is never silently ignored.
    pub fn resolve(&self, handles: &ScopeOverrideHandles) -> Result<ScopeAnalyzerOverrides> {
        self.validate()?;
        let mut overrides = self.scalars();
        if let Some(requested) = &self.model {
            let available = handles.model.as_ref().ok_or_else(|| {
                Error::unsupported(format!(
                    "scope override requests model `{requested}`, but no model is attached to this \
                     facade; attach one with LongTermMemoryBuilder::model or drop the `model` \
                     override"
                ))
            })?;
            if available.model_name() != requested {
                return Err(Error::unsupported(format!(
                    "scope override requests model `{requested}`, but this facade only holds the \
                     engine model `{}`; a per-scope model must be registered on the facade \
                     (LongTermMemoryBuilder::model), not resolved by name",
                    available.model_name()
                )));
            }
            overrides = overrides.with_model(Arc::clone(available));
        }
        if let Some(requested) = &self.embedding {
            let available = handles.embedding.as_ref().ok_or_else(|| {
                Error::unsupported(format!(
                    "scope override requests embedding `{requested}`, but no embedding model is \
                     attached and the engine's embedding model is fixed at construction \
                     (LongTermMemoryBuilder::embedding)"
                ))
            })?;
            if available.model_name() != requested {
                return Err(Error::unsupported(format!(
                    "scope override requests embedding `{requested}`, but the engine's embedding \
                     model is fixed at construction (`{}`); openjiuwen-memory cannot swap the \
                     embedding per scope",
                    available.model_name()
                )));
            }
            overrides = overrides.with_embedding(Arc::clone(available));
        }
        Ok(overrides)
    }

    /// Returns the scalar overrides (no handles), validated as the resolver does.
    fn scalars(&self) -> ScopeAnalyzerOverrides {
        let mut overrides = ScopeAnalyzerOverrides::new();
        if let Some(extractor) = self.extractor {
            overrides = overrides.with_extractor(extractor);
        }
        if let Some(classifier) = self.classifier {
            overrides = overrides.with_classifier(classifier);
        }
        if let Some(min_importance) = self.min_importance {
            overrides = overrides.with_min_importance(min_importance);
        }
        if let Some(max_facts) = self.max_facts {
            overrides = overrides.with_max_facts(max_facts);
        }
        if let Some(default_importance) = self.default_importance {
            overrides = overrides.with_default_importance(default_importance);
        }
        if let Some(stages) = &self.stages {
            overrides = overrides.with_stages(stages.clone());
        }
        overrides
    }
}

/// The in-memory handles a [`ScopeAnalyzerOverrideSpec`] resolves its names against.
///
/// The facade fills these from the engine it was built with, so a spec can only name what
/// the process actually owns. The manual [`fmt::Debug`] reports handle *names* only, so
/// logging a resolution request never leaks a provider client or credentials.
#[derive(Clone, Default)]
pub struct ScopeOverrideHandles {
    /// The model the facade exposes to per-scope analysis.
    model: Option<Arc<dyn Model>>,
    /// The embedding model the engine was constructed with.
    embedding: Option<Arc<dyn Embedding>>,
}

impl ScopeOverrideHandles {
    /// Builds empty handles (nothing to resolve against).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the handles with `model` set.
    #[must_use]
    pub fn with_model(mut self, model: Arc<dyn Model>) -> Self {
        self.model = Some(model);
        self
    }

    /// Returns the handles with `embedding` set.
    #[must_use]
    pub fn with_embedding(mut self, embedding: Arc<dyn Embedding>) -> Self {
        self.embedding = Some(embedding);
        self
    }

    /// Returns the model handle, when one is held.
    #[must_use]
    pub fn model(&self) -> Option<&Arc<dyn Model>> {
        self.model.as_ref()
    }

    /// Returns the embedding handle, when one is held.
    #[must_use]
    pub fn embedding(&self) -> Option<&Arc<dyn Embedding>> {
        self.embedding.as_ref()
    }

    /// Returns `true` when neither handle is held.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.model.is_none() && self.embedding.is_none()
    }
}

impl fmt::Debug for ScopeOverrideHandles {
    /// Reports handle names only.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let model = self.model.as_deref().map(Model::model_name);
        let embedding = self.embedding.as_deref().map(Embedding::model_name);
        formatter
            .debug_struct("ScopeOverrideHandles")
            .field("model", &model)
            .field("embedding", &embedding)
            .finish()
    }
}

impl LongTermMemory {
    /// Stores the analyzer override document of `scope_id`, replacing an existing one.
    ///
    /// The document is stored as JSON under `memory_scope_overrides/{scope_id}`. Only its
    /// *shape* is validated here; whether a named `model`/`embedding` exists is decided per
    /// call by [`LongTermMemory::resolve_scope_overrides`], so a document written on one
    /// facade and read on another still resolves against the handles of the reading
    /// facade.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`] for an invalid `scope_id` or an invalid document, and the
    /// storage error when the write fails.
    pub async fn set_scope_overrides(
        &self,
        scope_id: &str,
        spec: &ScopeAnalyzerOverrideSpec,
    ) -> Result<()> {
        validate_scope_id(scope_id)?;
        spec.validate()?;
        let value = serde_json::to_value(spec).map_err(|error| {
            Error::external(error).with_context(format!(
                "serializing the memory scope overrides `{scope_id}`"
            ))
        })?;
        self.engine
            .storage()
            .set_scope_overrides(scope_id, value)
            .await
    }

    /// Reads the analyzer override document of `scope_id`, or `None` when it is absent.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`] for an invalid `scope_id` or a stored payload that is not a
    /// [`ScopeAnalyzerOverrideSpec`] — naming the scope, never defaulting silently — and
    /// the storage error when the read fails.
    pub async fn get_scope_overrides(
        &self,
        scope_id: &str,
    ) -> Result<Option<ScopeAnalyzerOverrideSpec>> {
        validate_scope_id(scope_id)?;
        let Some(value) = self.engine.storage().get_scope_overrides(scope_id).await? else {
            return Ok(None);
        };
        let spec = serde_json::from_value(value).map_err(|error| {
            Error::validation(format!(
                "malformed memory scope overrides JSON for `{scope_id}`: {error}"
            ))
        })?;
        Ok(Some(spec))
    }

    /// Deletes the analyzer override document of `scope_id`, returning whether it existed.
    ///
    /// Independent of [`LongTermMemory::delete_scope_config`](super::LongTermMemory::delete_scope_config):
    /// that method keeps its signature and clears `memory_scope_config/{scope_id}` only, so
    /// a scope config and a scope override are removed by their own calls.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`] for an invalid `scope_id` and the storage error when the
    /// delete fails.
    pub async fn delete_scope_overrides(&self, scope_id: &str) -> Result<bool> {
        validate_scope_id(scope_id)?;
        self.engine.storage().delete_scope_overrides(scope_id).await
    }

    /// Resolves the persisted overrides of `scope` against this facade's handles.
    ///
    /// An absent (or empty) document resolves to empty
    /// [`ScopeAnalyzerOverrides`], i.e. the engine configuration unchanged.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`] for an invalid scope or a malformed stored document, and
    /// [`Error::Unsupported`] when a named `model`/`embedding` cannot be satisfied — see
    /// [`ScopeAnalyzerOverrideSpec::resolve`].
    pub async fn resolve_scope_overrides(
        &self,
        scope: &MemoryScope,
    ) -> Result<ScopeAnalyzerOverrides> {
        scope.validate()?;
        let Some(spec) = self.get_scope_overrides(&scope.scope_id).await? else {
            return Ok(ScopeAnalyzerOverrides::new());
        };
        let overrides = spec.resolve(&self.scope_override_handles())?;
        if !overrides.is_empty() {
            tracing::debug!(
                target: TARGET,
                scope_id = scope.scope_id.as_str(),
                "resolved per-scope analyzer overrides"
            );
        }
        Ok(overrides)
    }

    /// The handles a scope override document can name.
    pub(super) fn scope_override_handles(&self) -> ScopeOverrideHandles {
        let mut handles = ScopeOverrideHandles::new();
        if let Some(model) = self.engine.model() {
            handles = handles.with_model(Arc::clone(model));
        }
        if let Some(embedding) = self.engine.embedding() {
            handles = handles.with_embedding(Arc::clone(embedding));
        }
        handles
    }
}

/// Validates a bare `scope_id` before it becomes a storage-key segment.
///
/// The rules are the [`MemoryScope`] rules (non-empty, no `/`, `\` or NUL, at most 128
/// characters), so an invalid id is [`Error::Validation`] rather than a key that collides
/// with a neighbouring scope's namespace.
fn validate_scope_id(scope_id: &str) -> Result<()> {
    MemoryScope::new(DEFAULT_ID, scope_id).validate()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analyzer::{AnalyzerConfig, StageKind};
    use crate::config::MemoryEngineConfig;
    use openjiuwen_retrieval::embedding::HashEmbedding;
    use openjiuwen_test_support::ScriptedModel;
    use serde_json::json;

    fn model() -> Arc<dyn Model> {
        Arc::new(ScriptedModel::builder().build())
    }

    fn handles() -> ScopeOverrideHandles {
        ScopeOverrideHandles::new()
            .with_model(model())
            .with_embedding(Arc::new(HashEmbedding::new(8)))
    }

    fn full_json() -> serde_json::Value {
        json!({
            "extractor": "llm",
            "classifier": "llm",
            "min_importance": 0.4,
            "max_facts": 2,
            "default_importance": 0.6,
            "stages": {"plan": "llm"},
            "model": "scripted-model",
            "embedding": "hash-embedding",
            "a_future_field": {"ignored": true}
        })
    }

    async fn facade() -> Arc<LongTermMemory> {
        LongTermMemory::in_memory(MemoryEngineConfig::default())
            .await
            .expect("in-memory facade")
    }

    fn scope() -> MemoryScope {
        MemoryScope::new("alice", "ws")
    }

    fn spec(max_facts: usize) -> ScopeAnalyzerOverrideSpec {
        ScopeAnalyzerOverrideSpec::new().with_max_facts(max_facts)
    }

    #[test]
    fn the_spec_round_trips_and_tolerates_unknown_fields() {
        let decoded: ScopeAnalyzerOverrideSpec =
            serde_json::from_value(full_json()).expect("decode");
        assert_eq!(decoded.extractor, Some(ExtractorKind::Llm));
        assert_eq!(decoded.classifier, Some(ClassifierKind::Llm));
        assert_eq!(decoded.min_importance, Some(0.4));
        assert_eq!(decoded.max_facts, Some(2));
        assert_eq!(decoded.default_importance, Some(0.6));
        assert_eq!(
            decoded.stages.as_ref().map(|stages| stages.plan),
            Some(StageKind::Llm)
        );
        assert_eq!(decoded.model.as_deref(), Some("scripted-model"));
        assert_eq!(decoded.embedding.as_deref(), Some("hash-embedding"));

        let json = serde_json::to_value(&decoded).expect("encode");
        assert_eq!(json["extractor"], json!("llm"));
        assert_eq!(json["max_facts"], json!(2));
        assert_eq!(json["model"], json!("scripted-model"));
        assert_eq!(json["stages"]["plan"], json!("llm"));
        assert_eq!(json["stages"]["reasoning_key"], json!("rationale"));
        assert!(json.get("a_future_field").is_none());

        let restored: ScopeAnalyzerOverrideSpec = serde_json::from_value(json).expect("re-decode");
        assert_eq!(restored, decoded);

        // Absent fields default and an unknown-only payload decodes to the empty spec.
        let tolerant: ScopeAnalyzerOverrideSpec =
            serde_json::from_value(json!({"unknown": 1})).expect("tolerant");
        assert!(tolerant.is_empty());
        assert_eq!(tolerant, ScopeAnalyzerOverrideSpec::default());
        assert_eq!(ScopeAnalyzerOverrideSpec::new(), tolerant);
    }

    #[test]
    fn builders_and_empty_detection() {
        let spec = ScopeAnalyzerOverrideSpec::new();
        assert!(spec.is_empty());
        assert!(spec.validate().is_ok());
        // Every builder alone makes the document non-empty.
        for populated in [
            spec.clone().with_extractor(ExtractorKind::Llm),
            spec.clone().with_classifier(ClassifierKind::Llm),
            spec.clone().with_min_importance(0.2),
            spec.clone().with_max_facts(3),
            spec.clone().with_default_importance(0.5),
            spec.clone().with_stages(StageConfig::default()),
            spec.clone().with_model("m"),
            spec.clone().with_embedding("e"),
        ] {
            assert!(!populated.is_empty(), "{populated:?}");
            assert!(populated.validate().is_ok(), "{populated:?}");
        }
        let populated = spec
            .with_extractor(ExtractorKind::Llm)
            .with_classifier(ClassifierKind::Llm)
            .with_min_importance(0.2)
            .with_max_facts(3)
            .with_default_importance(0.5)
            .with_stages(StageConfig::default())
            .with_model("m")
            .with_embedding("e");
        assert!(!populated.is_empty());
        assert!(populated.validate().is_ok());
    }

    #[test]
    fn invalid_specs_are_rejected() {
        for spec in [
            ScopeAnalyzerOverrideSpec::new().with_min_importance(1.5),
            ScopeAnalyzerOverrideSpec::new().with_min_importance(f32::NAN),
            ScopeAnalyzerOverrideSpec::new().with_default_importance(-0.1),
            ScopeAnalyzerOverrideSpec::new().with_max_facts(0),
            ScopeAnalyzerOverrideSpec::new().with_model("  "),
            ScopeAnalyzerOverrideSpec::new().with_embedding(""),
            ScopeAnalyzerOverrideSpec::new().with_stages(StageConfig {
                reasoning_key: " ".to_owned(),
                ..StageConfig::default()
            }),
        ] {
            let error = spec.validate().expect_err("invalid spec");
            assert!(matches!(error, Error::Validation(_)), "{error:?}");
            let resolved = spec.resolve(&handles());
            assert!(matches!(resolved, Err(Error::Validation(_))), "{error:?}");
        }
    }

    #[test]
    fn resolve_applies_the_scalar_overrides() {
        let spec = ScopeAnalyzerOverrideSpec {
            extractor: Some(ExtractorKind::Llm),
            classifier: Some(ClassifierKind::Llm),
            min_importance: Some(0.4),
            max_facts: Some(2),
            default_importance: Some(0.6),
            stages: Some(StageConfig {
                plan: StageKind::Llm,
                ..StageConfig::default()
            }),
            model: None,
            embedding: None,
        };
        let overrides = spec.resolve(&handles()).expect("resolve");
        assert!(!overrides.is_empty());
        assert_eq!(overrides.extractor(), Some(ExtractorKind::Llm));
        assert_eq!(overrides.classifier(), Some(ClassifierKind::Llm));
        assert_eq!(overrides.min_importance(), Some(0.4));
        assert_eq!(overrides.max_facts(), Some(2));
        assert_eq!(overrides.default_importance(), Some(0.6));
        assert_eq!(
            overrides.stages().map(|stages| stages.plan),
            Some(StageKind::Llm)
        );
        // `model: None`/`embedding: None` add no handle at all.
        assert!(overrides.model().is_none());
        assert!(overrides.embedding().is_none());
        let effective = overrides.effective_config(&AnalyzerConfig::default());
        assert_eq!(effective.extractor, ExtractorKind::Llm);
        assert_eq!(effective.max_facts, 2);
        // The engine configuration is untouched.
        assert_eq!(AnalyzerConfig::default().extractor, ExtractorKind::Rule);
    }

    #[test]
    fn resolve_accepts_the_handles_it_names() {
        let spec = ScopeAnalyzerOverrideSpec::new()
            .with_model("scripted-model")
            .with_embedding("hash-embedding");
        let resolved = spec.resolve(&handles()).expect("resolve");
        assert_eq!(
            resolved
                .model()
                .map(|model| Model::model_name(model.as_ref())),
            Some("scripted-model")
        );
        assert_eq!(
            resolved
                .embedding()
                .map(|embedding| Embedding::model_name(embedding.as_ref())),
            Some("hash-embedding")
        );
        // An empty handle set resolves the empty spec (nothing is named).
        assert!(ScopeAnalyzerOverrideSpec::new()
            .resolve(&ScopeOverrideHandles::new())
            .expect("empty resolve")
            .is_empty());
    }

    #[test]
    fn unresolvable_names_are_typed_unsupported_errors() {
        let missing = ScopeAnalyzerOverrideSpec::new().with_model("gpt-4o");
        let error = missing
            .resolve(&ScopeOverrideHandles::new())
            .expect_err("no model attached");
        match error {
            Error::Unsupported(reason) => {
                assert!(reason.contains("gpt-4o"), "{reason}");
                assert!(reason.contains("no model is attached"), "{reason}");
            }
            other => panic!("expected Error::Unsupported, got {other:?}"),
        }

        let error = missing
            .resolve(&ScopeOverrideHandles::new().with_model(model()))
            .expect_err("only the engine model is held");
        match error {
            Error::Unsupported(reason) => {
                assert!(reason.contains("gpt-4o"), "{reason}");
                assert!(reason.contains("scripted-model"), "{reason}");
            }
            other => panic!("expected Error::Unsupported, got {other:?}"),
        }

        let embedding = ScopeAnalyzerOverrideSpec::new().with_embedding("text-embedding-3-large");
        let error = embedding
            .resolve(&handles())
            .expect_err("the engine embedding is fixed");
        match error {
            Error::Unsupported(reason) => {
                assert!(reason.contains("text-embedding-3-large"), "{reason}");
                assert!(reason.contains("fixed at construction"), "{reason}");
                assert!(reason.contains("hash-embedding"), "{reason}");
            }
            other => panic!("expected Error::Unsupported, got {other:?}"),
        }

        let error = embedding
            .resolve(&ScopeOverrideHandles::new())
            .expect_err("no embedding attached");
        match error {
            Error::Unsupported(reason) => {
                assert!(reason.contains("text-embedding-3-large"), "{reason}");
                assert!(reason.contains("fixed at construction"), "{reason}");
            }
            other => panic!("expected Error::Unsupported, got {other:?}"),
        }
    }

    #[test]
    fn handle_debug_reports_names_only() {
        let rendered = format!("{:?}", handles());
        assert!(
            rendered.contains("model: Some(\"scripted-model\")"),
            "{rendered}"
        );
        assert!(
            rendered.contains("embedding: Some(\"hash-embedding\")"),
            "{rendered}"
        );
        let empty = ScopeOverrideHandles::new();
        assert!(empty.is_empty());
        assert!(empty.model().is_none());
        assert!(empty.embedding().is_none());
        assert!(!handles().is_empty());
        assert!(handles().model().is_some());
        assert!(handles().embedding().is_some());
    }

    #[tokio::test]
    async fn overrides_round_trip_overwrite_and_delete() {
        let ltm = facade().await;
        assert_eq!(ltm.get_scope_overrides("ws").await.expect("absent"), None);
        ltm.set_scope_overrides("ws", &spec(2)).await.expect("set");
        assert_eq!(
            ltm.get_scope_overrides("ws").await.expect("get"),
            Some(spec(2))
        );
        // An overwrite replaces the document rather than merging it.
        ltm.set_scope_overrides(
            "ws",
            &ScopeAnalyzerOverrideSpec::new().with_extractor(ExtractorKind::Llm),
        )
        .await
        .expect("overwrite");
        assert_eq!(
            ltm.get_scope_overrides("ws").await.expect("get"),
            Some(ScopeAnalyzerOverrideSpec::new().with_extractor(ExtractorKind::Llm))
        );
        assert!(ltm.delete_scope_overrides("ws").await.expect("delete"));
        assert!(!ltm
            .delete_scope_overrides("ws")
            .await
            .expect("delete again"));
        assert_eq!(ltm.get_scope_overrides("ws").await.expect("gone"), None);
    }

    #[tokio::test]
    async fn scopes_are_isolated_and_scope_ids_are_validated() {
        let ltm = facade().await;
        ltm.set_scope_overrides("ws", &spec(1))
            .await
            .expect("set ws");
        ltm.set_scope_overrides("other", &spec(3))
            .await
            .expect("set other");
        assert_eq!(
            ltm.get_scope_overrides("ws").await.expect("ws"),
            Some(spec(1))
        );
        assert_eq!(
            ltm.get_scope_overrides("other").await.expect("other"),
            Some(spec(3))
        );
        assert!(ltm.delete_scope_overrides("ws").await.expect("delete"));
        assert_eq!(
            ltm.get_scope_overrides("other").await.expect("other"),
            Some(spec(3)),
            "deleting one scope leaves its neighbour alone"
        );
        for scope_id in ["", "bad/scope", "nul\0byte"] {
            for error in [
                ltm.set_scope_overrides(scope_id, &spec(1))
                    .await
                    .expect_err("invalid set"),
                ltm.get_scope_overrides(scope_id)
                    .await
                    .expect_err("invalid get"),
                ltm.delete_scope_overrides(scope_id)
                    .await
                    .expect_err("invalid delete"),
            ] {
                assert!(matches!(error, Error::Validation(_)), "{error:?}");
            }
        }
    }

    #[tokio::test]
    async fn deleting_the_scope_config_keeps_the_overrides() {
        let ltm = facade().await;
        ltm.set_scope_config("ws", crate::config::MemoryScopeConfig::default())
            .await
            .expect("scope config");
        ltm.set_scope_overrides("ws", &spec(4))
            .await
            .expect("overrides");
        assert!(ltm.delete_scope_config("ws").await.expect("delete config"));
        assert_eq!(ltm.get_scope_config("ws").await.expect("config gone"), None);
        assert_eq!(
            ltm.get_scope_overrides("ws").await.expect("overrides kept"),
            Some(spec(4)),
            "the two namespaces are independent"
        );
        assert!(ltm.delete_scope_overrides("ws").await.expect("delete"));
    }

    #[tokio::test]
    async fn a_malformed_stored_payload_is_a_validation_error() {
        let ltm = facade().await;
        ltm.engine()
            .storage()
            .set_scope_overrides("ws", json!({"max_facts": "many"}))
            .await
            .expect("store a mis-shaped payload");
        let error = ltm.get_scope_overrides("ws").await.expect_err("malformed");
        match error {
            Error::Validation(reason) => assert!(reason.contains("ws"), "{reason}"),
            other => panic!("expected Error::Validation, got {other:?}"),
        }
        let error = ltm
            .resolve_scope_overrides(&scope())
            .await
            .expect_err("resolution never defaults a malformed document");
        assert!(matches!(error, Error::Validation(_)), "{error:?}");

        // A payload that is valid JSON but not an object is also rejected.
        ltm.engine()
            .storage()
            .set_scope_overrides("ws", json!("oops"))
            .await
            .expect("store a scalar payload");
        let error = ltm.get_scope_overrides("ws").await.expect_err("malformed");
        assert!(matches!(error, Error::Validation(_)), "{error:?}");
    }

    #[tokio::test]
    async fn resolution_reads_the_persisted_document() {
        let ltm = facade().await;
        let overrides = ltm.resolve_scope_overrides(&scope()).await.expect("absent");
        assert!(overrides.is_empty());
        ltm.set_scope_overrides("ws", &spec(2)).await.expect("set");
        let resolved = ltm
            .resolve_scope_overrides(&scope())
            .await
            .expect("resolve");
        assert_eq!(resolved.max_facts(), Some(2));
        // Another scope is unaffected.
        let other = MemoryScope::new("alice", "other");
        assert!(ltm
            .resolve_scope_overrides(&other)
            .await
            .expect("other")
            .is_empty());
        // An invalid scope is a validation error before any read.
        let invalid = MemoryScope::new("", "ws");
        let error = ltm
            .resolve_scope_overrides(&invalid)
            .await
            .expect_err("invalid scope");
        assert!(matches!(error, Error::Validation(_)), "{error:?}");
    }

    #[tokio::test]
    async fn unresolvable_names_fail_at_call_time() {
        // A document naming a model nobody attached stores fine and fails precisely when
        // a call resolves it.
        let ltm = facade().await;
        ltm.set_scope_overrides("ws", &ScopeAnalyzerOverrideSpec::new().with_model("gpt-4o"))
            .await
            .expect("stored, not resolved");
        let error = ltm
            .resolve_scope_overrides(&scope())
            .await
            .expect_err("unresolvable model");
        match error {
            Error::Unsupported(reason) => assert!(reason.contains("gpt-4o"), "{reason}"),
            other => panic!("expected Error::Unsupported, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn overrides_survive_a_local_file_reopen() {
        let directory = tempfile::tempdir().expect("tempdir");
        let opened = LongTermMemory::local_file(MemoryEngineConfig::default(), directory.path())
            .await
            .expect("local file facade");
        opened
            .set_scope_overrides("ws", &spec(3))
            .await
            .expect("set");
        drop(opened);
        let reopened = LongTermMemory::local_file(MemoryEngineConfig::default(), directory.path())
            .await
            .expect("reopen");
        assert_eq!(
            reopened.get_scope_overrides("ws").await.expect("get"),
            Some(spec(3))
        );
        assert!(reopened.delete_scope_overrides("ws").await.expect("delete"));
        assert_eq!(
            reopened.get_scope_overrides("ws").await.expect("gone"),
            None
        );
    }

    #[cfg(feature = "sqlite")]
    mod sqlite {
        use super::*;

        #[tokio::test]
        async fn overrides_survive_a_sqlite_reopen() {
            let directory = tempfile::tempdir().expect("tempdir");
            let path = directory.path().join("memory.db");
            let opened = LongTermMemory::sqlite(MemoryEngineConfig::default(), &path)
                .await
                .expect("sqlite facade");
            opened
                .set_scope_overrides("ws", &spec(5))
                .await
                .expect("set");
            drop(opened);
            let reopened = LongTermMemory::sqlite(MemoryEngineConfig::default(), &path)
                .await
                .expect("reopen");
            assert_eq!(
                reopened.get_scope_overrides("ws").await.expect("get"),
                Some(spec(5))
            );
            assert!(reopened.delete_scope_overrides("ws").await.expect("delete"));
            assert_eq!(
                reopened.get_scope_overrides("ws").await.expect("gone"),
                None
            );
        }
    }
}
