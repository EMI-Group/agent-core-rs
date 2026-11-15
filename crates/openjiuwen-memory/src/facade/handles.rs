//! The named model/embedding handle registry a scope override document resolves against.
//!
//! [`ScopeOverrideHandles`] is the *registry* half of the per-scope override contract:
//! a [`ScopeAnalyzerOverrideSpec`](super::ScopeAnalyzerOverrideSpec) names a `model` or
//! an `embedding` by string, and this type answers "do I hold a handle under that name?".
//!
//! The facade fills the registry from what the process actually owns:
//!
//! - the engine's own model and embedding are registered under their `model_name()`
//!   (so naming the engine model keeps working);
//! - [`LongTermMemoryBuilder::model_named`](super::LongTermMemoryBuilder::model_named)
//!   and [`LongTermMemoryBuilder::embedding_named`](super::LongTermMemoryBuilder::embedding_named)
//!   add further handles under caller-chosen names, which is what makes swapping the
//!   model or the embedding **per scope** possible at all.
//!
//! A name that is registered nowhere is never resolved by falling back to the engine
//! handle: [`ScopeAnalyzerOverrideSpec::resolve`](super::ScopeAnalyzerOverrideSpec::resolve)
//! reports a typed `Error::Unsupported` listing every registered name instead.
//!
//! The manual [`fmt::Debug`] prints handle *names* only, so logging a registry never
//! leaks a provider client or credentials.
use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use openjiuwen_foundation::llm::Model;
use openjiuwen_retrieval::embedding::Embedding;

/// The named handles a [`ScopeAnalyzerOverrideSpec`](super::ScopeAnalyzerOverrideSpec)
/// resolves its names against.
///
/// The registry has no interior mutability: the facade builds one at construction (the
/// engine's handles first, then the caller's named ones) and shares it by reference, so
/// a resolution can never observe a half-registered set.
#[derive(Clone, Default)]
pub struct ScopeOverrideHandles {
    /// The default model handle (the engine's own, when it has one).
    model: Option<Arc<dyn Model>>,
    /// The default embedding handle (the engine's own, when it has one).
    embedding: Option<Arc<dyn Embedding>>,
    /// Every model handle by registered name.
    models: BTreeMap<String, Arc<dyn Model>>,
    /// Every embedding handle by registered name.
    embeddings: BTreeMap<String, Arc<dyn Embedding>>,
}

impl ScopeOverrideHandles {
    /// Builds empty handles (nothing to resolve against).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers `model` under its own [`Model::model_name`] and, when no default model
    /// is held yet, makes it the default handle.
    #[must_use]
    pub fn with_model(mut self, model: Arc<dyn Model>) -> Self {
        self.models
            .insert(model.model_name().to_owned(), Arc::clone(&model));
        if self.model.is_none() {
            self.model = Some(model);
        }
        self
    }

    /// Registers `embedding` under its own [`Embedding::model_name`] and, when no default
    /// embedding is held yet, makes it the default handle.
    #[must_use]
    pub fn with_embedding(mut self, embedding: Arc<dyn Embedding>) -> Self {
        self.embeddings
            .insert(embedding.model_name().to_owned(), Arc::clone(&embedding));
        if self.embedding.is_none() {
            self.embedding = Some(embedding);
        }
        self
    }

    /// Registers `model` under `name`, which need not be its `model_name()`.
    ///
    /// A blank name registers nothing and logs a warning: a name no document can spell
    /// unambiguously is not a resolution target. A duplicate name replaces the earlier
    /// handle (last registration wins).
    #[must_use]
    pub fn with_named_model(mut self, name: impl Into<String>, model: Arc<dyn Model>) -> Self {
        let name = name.into();
        if name.trim().is_empty() {
            tracing::warn!(
                target: "openjiuwen_memory::facade",
                "ignoring a model registered under a blank name"
            );
            return self;
        }
        self.models.insert(name, model);
        self
    }

    /// Registers `embedding` under `name`, which need not be its `model_name()`.
    ///
    /// A blank name registers nothing (with a warning) and a duplicate name replaces the
    /// earlier handle.
    #[must_use]
    pub fn with_named_embedding(
        mut self,
        name: impl Into<String>,
        embedding: Arc<dyn Embedding>,
    ) -> Self {
        let name = name.into();
        if name.trim().is_empty() {
            tracing::warn!(
                target: "openjiuwen_memory::facade",
                "ignoring an embedding registered under a blank name"
            );
            return self;
        }
        self.embeddings.insert(name, embedding);
        self
    }

    /// Returns the default model handle, when one is held.
    ///
    /// This is the engine's own model; a scope that names no model inherits it. It is
    /// **not** a fallback for an unknown name — see [`ScopeOverrideHandles::model_named`].
    #[must_use]
    pub fn model(&self) -> Option<&Arc<dyn Model>> {
        self.model.as_ref()
    }

    /// Returns the default embedding handle, when one is held.
    #[must_use]
    pub fn embedding(&self) -> Option<&Arc<dyn Embedding>> {
        self.embedding.as_ref()
    }

    /// Returns the model registered under `name`, when there is one.
    #[must_use]
    pub fn model_named(&self, name: &str) -> Option<&Arc<dyn Model>> {
        self.models.get(name)
    }

    /// Returns the embedding registered under `name`, when there is one.
    #[must_use]
    pub fn embedding_named(&self, name: &str) -> Option<&Arc<dyn Embedding>> {
        self.embeddings.get(name)
    }

    /// The registered model names, ascending.
    #[must_use]
    pub fn model_names(&self) -> Vec<&str> {
        self.models.keys().map(String::as_str).collect()
    }

    /// The registered embedding names, ascending.
    #[must_use]
    pub fn embedding_names(&self) -> Vec<&str> {
        self.embeddings.keys().map(String::as_str).collect()
    }

    /// Returns `true` when no handle is registered at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.models.is_empty() && self.embeddings.is_empty()
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
            .field("models", &self.model_names())
            .field("embeddings", &self.embedding_names())
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

    fn model_name(handle: &Arc<dyn Model>) -> &str {
        handle.model_name()
    }

    fn named_model(name: &str) -> Arc<dyn Model> {
        Arc::new(ScriptedModel::builder().model_name(name).build())
    }

    fn named_embedding(name: &str, dimensions: usize) -> Arc<dyn Embedding> {
        Arc::new(HashEmbedding::new(dimensions).with_model_name(name))
    }

    #[test]
    fn an_empty_registry_holds_nothing() {
        let handles = ScopeOverrideHandles::new();
        assert!(handles.is_empty());
        assert!(handles.model().is_none());
        assert!(handles.embedding().is_none());
        assert!(handles.model_named("scripted-model").is_none());
        assert!(handles.embedding_named("hash-embedding").is_none());
        assert_eq!(handles.model_names(), Vec::<&str>::new());
        assert_eq!(handles.embedding_names(), Vec::<&str>::new());
        assert_eq!(
            format!("{handles:?}"),
            "ScopeOverrideHandles { model: None, embedding: None, models: [], embeddings: [] }"
        );
    }

    #[test]
    fn the_unnamed_builders_register_under_the_handle_name_and_set_the_default() {
        let handles = ScopeOverrideHandles::new()
            .with_model(model())
            .with_embedding(Arc::new(HashEmbedding::new(8)));
        assert!(!handles.is_empty());
        // The first handle registered is the default.
        assert_eq!(handles.model().map(model_name), Some("scripted-model"));
        assert_eq!(
            handles.embedding().map(|embedding| embedding.model_name()),
            Some("hash-embedding")
        );
        assert_eq!(handles.model_names(), vec!["scripted-model"]);
        assert_eq!(handles.embedding_names(), vec!["hash-embedding"]);
        assert!(handles.model_named("scripted-model").is_some());
        assert!(handles.embedding_named("hash-embedding").is_some());
    }

    #[test]
    fn named_registrations_resolve_by_name_and_never_become_the_default() {
        let handles = ScopeOverrideHandles::new()
            .with_model(model())
            .with_named_model("wide-model", named_model("scripted-wide"))
            .with_embedding(Arc::new(HashEmbedding::new(8)))
            .with_named_embedding("wide", named_embedding("wide", 24));
        assert_eq!(
            handles.model().map(model_name),
            Some("scripted-model"),
            "a named registration never replaces the default handle"
        );
        assert_eq!(
            handles.model_named("wide-model").map(model_name),
            Some("scripted-wide")
        );
        assert_eq!(
            handles
                .embedding_named("wide")
                .map(|embedding| embedding.model_name()),
            Some("wide")
        );
        assert_eq!(
            handles.model_names(),
            vec!["scripted-model", "wide-model"],
            "names are listed ascending"
        );
        assert_eq!(handles.embedding_names(), vec!["hash-embedding", "wide"]);
        // The name is what resolves, not the handle's own model name.
        assert!(handles.model_named("scripted-wide").is_none());
    }

    #[test]
    fn a_duplicate_name_replaces_the_earlier_handle() {
        let handles = ScopeOverrideHandles::new()
            .with_named_model("same", named_model("first"))
            .with_named_model("same", named_model("second"));
        assert_eq!(handles.model_named("same").map(model_name), Some("second"));
        assert_eq!(handles.model_names(), vec!["same"]);
    }

    #[test]
    fn a_blank_name_registers_nothing() {
        let handles = ScopeOverrideHandles::new()
            .with_named_model("  ", model())
            .with_named_embedding("", Arc::new(HashEmbedding::new(8)));
        assert!(handles.is_empty());
        assert_eq!(handles.model_names(), Vec::<&str>::new());
        assert_eq!(handles.embedding_names(), Vec::<&str>::new());
    }

    #[test]
    fn debug_reports_names_only() {
        let rendered = format!(
            "{:?}",
            ScopeOverrideHandles::new()
                .with_model(model())
                .with_embedding(Arc::new(HashEmbedding::new(8)))
                .with_named_embedding("wide", named_embedding("wide", 24))
        );
        assert!(
            rendered.contains("model: Some(\"scripted-model\")"),
            "{rendered}"
        );
        assert!(
            rendered.contains("embedding: Some(\"hash-embedding\")"),
            "{rendered}"
        );
        assert!(
            rendered.contains("embeddings: [\"hash-embedding\", \"wide\"]"),
            "{rendered}"
        );
    }
}
