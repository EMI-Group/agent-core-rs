//! Model factories keyed on an id: [`ModelMgr`].
//!
//! Ports Python `openjiuwen/core/runner/resources_manager/model_manager.py`
//! (`ModelMgr.add_model` / `remove_model` / `get_model`). Python stores a bare
//! `(model_id, ModelProvider)` pair per entry and calls the provider on every
//! `get_model`, so a registration may build a fresh model for each run.
//!
//! [`Registry`] keys, orders, and serializes entries through a
//! [`ResourceCard`], so [`ModelCard`] carries the identity Python keeps as a
//! bare id string: the id is the Python key and the name defaults to it, since
//! Python's model registry has no display name of its own.
//!
//! Not ported: Python's `get_model(model_id, session)` wraps the result in
//! `decorate_model_with_trace(model, session)`. The Rust port returns the
//! provider's model unchanged; per-session tracing belongs to the caller (the
//! session layer's tracer), which keeps this registry free of session state.

use std::sync::Arc;

use openjiuwen_common::error::Result;
use openjiuwen_foundation::llm::Model;
use serde::{Deserialize, Serialize};

use super::registry::{Provider, RegisterPolicy, Registration, Registry, ResourceCard};

/// A factory for models, invoked once per lookup.
///
/// Python parity: `ModelProvider` in
/// `openjiuwen/core/runner/resources_manager/base.py` is "a callable accepting
/// variable arguments and returning a model". The Rust port narrows it to the
/// zero-argument [`Provider`] shape because a factory registered here is already
/// bound to its configuration; Python's flexible call shape was a convenience
/// for `init_model`, which lives in `openjiuwen_foundation::llm`.
///
/// This alias is deliberately **not**
/// [`openjiuwen_foundation::llm::ModelProvider`]: that trait describes how to
/// build a model from a client/request configuration in the model-factory
/// registry, while this is the runner's stored per-id factory. Register a new
/// vendor through the trait; register a ready-to-use model through this alias.
pub type ModelProvider = Provider<Arc<dyn Model>>;

/// The registry identity of one registered model.
///
/// Python keys models on a bare id string (`ModelMgr.add_model(model_id,
/// model)`); there is no card to hold. [`Registry`] needs a
/// [`ResourceCard`], so this card is the Rust identity carrier: [`id`] is
/// Python's `model_id` and [`name`](ModelCard::name) is an optional display
/// name that defaults to the id.
///
/// [`id`]: ModelCard::id
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelCard {
    /// The registry key (Python's `model_id`).
    pub id: String,
    /// A display name; an omitted or empty value falls back to the id.
    ///
    /// Serialization skips an empty name, so a document deserialized without
    /// one round-trips unchanged.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub name: String,
}

impl ModelCard {
    /// Builds a card whose name is the id.
    #[must_use]
    pub fn new(id: impl Into<String>) -> Self {
        let id = id.into();
        Self {
            name: id.clone(),
            id,
        }
    }

    /// Builds a card with an explicit display name.
    ///
    /// An empty `name` is treated as "not given": [`ResourceCard::name`] then
    /// reports the id.
    #[must_use]
    pub fn named(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
        }
    }
}

impl ResourceCard for ModelCard {
    fn id(&self) -> &str {
        &self.id
    }

    fn name(&self) -> &str {
        if self.name.is_empty() {
            &self.id
        } else {
            &self.name
        }
    }
}

/// The registry of model factories a host publishes.
///
/// Python parity: `ModelMgr`. Entries are keyed on [`ModelCard::id`], the id
/// comes from the card (never from the caller's provider), and a lookup invokes
/// the registered factory, so a factory may hand out a fresh model per call.
///
/// Errors name the resource `"model"`, for example
/// `validation: model id cannot be empty or whitespace only`.
///
/// ```
/// use std::sync::Arc;
/// use openjiuwen_runner::resources::model::{ModelCard, ModelMgr};
///
/// let models = ModelMgr::new();
/// assert!(models.is_empty());
/// assert!(models.model_card("gpt").is_none());
/// assert_eq!(ModelCard::new("gpt").name, "gpt");
/// ```
#[derive(Debug)]
pub struct ModelMgr {
    /// Registered models, keyed on [`ModelCard::id`].
    models: Registry<ModelCard, Arc<dyn Model>>,
}

impl ModelMgr {
    /// Builds an empty manager.
    #[must_use]
    pub fn new() -> Self {
        Self {
            models: Registry::with_kind("model"),
        }
    }

    /// Registers the live `model` under `id`, failing on a duplicate id.
    ///
    /// The built-in factory clones the stored `Arc`, so every lookup hands back
    /// that same instance (the tool-like registration shape Python's
    /// `add_model` uses when a caller passes a ready model).
    ///
    /// # Errors
    ///
    /// [`openjiuwen_common::error::Error::Validation`] for an empty or
    /// whitespace-only id and for an id that is already registered.
    pub fn register_model(&self, id: impl Into<String>, model: Arc<dyn Model>) -> Result<()> {
        let provider: ModelProvider = Arc::new(move || Arc::clone(&model));
        self.register_model_with(id, provider, RegisterPolicy::Fail)
            .map(|_| ())
    }

    /// Registers `provider` under `id` with an explicit duplicate policy.
    ///
    /// The card is [`ModelCard::new`], so the display name is the id; use
    /// [`ModelMgr::register_factory`] to carry a custom name.
    ///
    /// # Errors
    ///
    /// [`openjiuwen_common::error::Error::Validation`] for an empty or
    /// whitespace-only id and, under [`RegisterPolicy::Fail`], for a duplicate
    /// id.
    pub fn register_model_with(
        &self,
        id: impl Into<String>,
        provider: ModelProvider,
        policy: RegisterPolicy,
    ) -> Result<Registration> {
        self.register_factory(ModelCard::new(id), provider, policy)
    }

    /// Registers `provider` under `card` with an explicit duplicate policy.
    ///
    /// # Errors
    ///
    /// [`openjiuwen_common::error::Error::Validation`] for an empty or
    /// whitespace-only card id and, under [`RegisterPolicy::Fail`], for a
    /// duplicate id.
    pub fn register_factory(
        &self,
        card: ModelCard,
        provider: ModelProvider,
        policy: RegisterPolicy,
    ) -> Result<Registration> {
        self.models.register(card, provider, policy)
    }

    /// Returns the model registered under `id` by invoking its factory.
    ///
    /// The factory runs after the registry lock is released, so it may itself
    /// consult the registry. A missing id returns `None` and invokes nothing.
    #[must_use]
    pub fn model(&self, id: &str) -> Option<Arc<dyn Model>> {
        self.models.get(id)
    }

    /// Returns the card registered under `id`.
    #[must_use]
    pub fn model_card(&self, id: &str) -> Option<ModelCard> {
        self.models.card(id)
    }

    /// Removes `id` and returns its card; a missing id returns `None`.
    #[must_use]
    pub fn remove_model(&self, id: &str) -> Option<ModelCard> {
        self.models.remove(id)
    }

    /// Returns the registered ids in registration order.
    #[must_use]
    pub fn model_ids(&self) -> Vec<String> {
        self.models.ids()
    }

    /// Returns the registered cards in registration order.
    #[must_use]
    pub fn models(&self) -> Vec<ModelCard> {
        self.models.cards()
    }

    /// Returns the number of registered models.
    #[must_use]
    pub fn len(&self) -> usize {
        self.models.len()
    }

    /// Returns whether no model is registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.models.is_empty()
    }

    /// Returns the mutation counter, incremented on every successful mutation.
    ///
    /// A host caching a projection of this registry compares the revision to
    /// decide whether to rebuild.
    #[must_use]
    pub fn revision(&self) -> u64 {
        self.models.revision()
    }

    /// Removes every model and returns how many were removed.
    ///
    /// A non-empty clear counts as one mutation; clearing an empty registry is a
    /// no-op and leaves the revision untouched.
    pub fn clear(&self) -> usize {
        self.models.clear()
    }

    /// Removes every model and resets the revision counter to zero.
    ///
    /// Python parity: `ResourceMgr.release` empties the model manager too, and
    /// [`ResourceMgr::release`](super::ResourceMgr::release) is the caller once
    /// the manager is wired into it; until then the method is unused in-crate.
    pub(crate) fn reset(&self) -> usize {
        self.models.reset()
    }
}

impl Default for ModelMgr {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use futures::stream::BoxStream;
    use openjiuwen_common::error::{Error, ErrorKind};
    use openjiuwen_foundation::llm::{ChatResponse, ModelRequestConfig, StreamChunk};
    use openjiuwen_foundation::prompt::BaseMessage;

    /// A model that exists only to be registered: invocation is unsupported.
    struct FixedModel {
        provider: &'static str,
        model_name: &'static str,
    }

    impl FixedModel {
        fn new(id: &'static str) -> Self {
            Self {
                provider: "test",
                model_name: id,
            }
        }
    }

    #[async_trait::async_trait]
    impl Model for FixedModel {
        fn provider(&self) -> &str {
            self.provider
        }

        fn model_name(&self) -> &str {
            self.model_name
        }

        async fn invoke(
            &self,
            _messages: &[BaseMessage],
            _request: &ModelRequestConfig,
        ) -> Result<ChatResponse> {
            Err(Error::Unsupported(
                "FixedModel serves the registry tests only".to_owned(),
            ))
        }

        fn stream(
            &self,
            _messages: Vec<BaseMessage>,
            _request: ModelRequestConfig,
        ) -> BoxStream<'static, Result<StreamChunk>> {
            Box::pin(futures::stream::once(async {
                Err(Error::Unsupported(
                    "FixedModel serves the registry tests only".to_owned(),
                ))
            }))
        }
    }

    /// A factory minting a fresh [`FixedModel`] per call and counting the calls.
    fn counting(calls: &Arc<AtomicUsize>, id: &'static str) -> ModelProvider {
        let calls = Arc::clone(calls);
        Arc::new(move || {
            calls.fetch_add(1, Ordering::SeqCst);
            Arc::new(FixedModel::new(id)) as Arc<dyn Model>
        })
    }

    fn fixed(id: &'static str) -> Arc<dyn Model> {
        Arc::new(FixedModel::new(id))
    }

    #[test]
    fn card_round_trips_through_json() {
        let card = ModelCard::named("gpt-4o", "the fast one");
        let json = serde_json::to_value(&card).unwrap();
        assert_eq!(
            json,
            serde_json::json!({"id": "gpt-4o", "name": "the fast one"})
        );
        assert_eq!(serde_json::from_value::<ModelCard>(json).unwrap(), card);

        // An omitted name falls back to the id, and an unknown field is ignored.
        let sparse = serde_json::from_str::<ModelCard>(r#"{"id": "qwen", "extra": 1}"#).unwrap();
        assert_eq!(sparse, ModelCard::named("qwen", ""));
        assert_eq!(sparse.name, "", "the wire value is kept verbatim");
        assert_eq!(ResourceCard::name(&sparse), "qwen");
        assert_eq!(
            serde_json::to_value(&sparse).unwrap(),
            serde_json::json!({"id": "qwen"}),
            "an empty name is not serialized"
        );
    }

    #[test]
    fn constructors_default_the_name_to_the_id() {
        assert_eq!(ModelCard::new("gpt"), ModelCard::named("gpt", "gpt"));
        assert_eq!(ResourceCard::name(&ModelCard::new("gpt")), "gpt");
    }

    #[test]
    fn registered_model_resolves_by_id() {
        let models = ModelMgr::new();
        models.register_model("gpt", fixed("gpt")).unwrap();

        let model = models.model("gpt").expect("registered");
        assert_eq!(model.model_name(), "gpt");
        assert_eq!(models.model_card("gpt"), Some(ModelCard::new("gpt")));
        assert_eq!(models.model_ids(), vec!["gpt".to_owned()]);
        assert_eq!(models.models(), vec![ModelCard::new("gpt")]);
        assert_eq!(models.len(), 1);
        assert!(!models.is_empty());
        assert!(models.model("missing").is_none());
    }

    // ports tests/unit_tests/core/runner/test_resource_manager.py (provider per lookup)
    #[test]
    fn provider_runs_on_every_lookup() {
        let calls = Arc::new(AtomicUsize::new(0));
        let models = ModelMgr::new();
        models
            .register_model_with("gpt", counting(&calls, "gpt"), RegisterPolicy::Fail)
            .unwrap();
        assert_eq!(
            calls.load(Ordering::SeqCst),
            0,
            "registration invokes nothing"
        );

        let first = models.model("gpt").expect("registered");
        let second = models.model("gpt").expect("registered");
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        assert!(
            !Arc::ptr_eq(&first, &second),
            "each lookup gets a fresh model"
        );
        assert_eq!(second.model_name(), "gpt");

        assert!(models.model("missing").is_none());
        assert_eq!(
            calls.load(Ordering::SeqCst),
            2,
            "a missing id invokes nothing"
        );
    }

    #[test]
    fn register_model_hands_back_the_same_instance() {
        let model = fixed("gpt");
        let models = ModelMgr::new();
        models.register_model("gpt", Arc::clone(&model)).unwrap();

        let resolved = models.model("gpt").expect("registered");
        assert!(Arc::ptr_eq(&model, &resolved));
    }

    #[test]
    fn removal_returns_the_card_and_drops_the_lookup() {
        let models = ModelMgr::new();
        models.register_model("gpt", fixed("gpt")).unwrap();
        models.register_model("qwen", fixed("qwen")).unwrap();

        assert_eq!(models.remove_model("gpt"), Some(ModelCard::new("gpt")));
        assert!(models.model("gpt").is_none());
        assert_eq!(models.model_ids(), vec!["qwen".to_owned()]);
        assert!(models.remove_model("gpt").is_none());
    }

    #[test]
    fn fail_policy_keeps_the_first_registration() {
        let models = ModelMgr::new();
        models.register_model("gpt", fixed("first")).unwrap();

        let error = models.register_model("gpt", fixed("second")).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert_eq!(
            error.to_string(),
            "validation: model 'gpt' is already registered"
        );
        assert_eq!(
            models.model("gpt").expect("still there").model_name(),
            "first"
        );
        assert_eq!(models.len(), 1);
    }

    #[test]
    fn keep_keeps_the_original_and_replace_swaps_it() {
        let models = ModelMgr::new();
        models
            .register_model_with("gpt", Arc::new(|| fixed("first")), RegisterPolicy::Fail)
            .unwrap();

        let kept = models
            .register_model_with("gpt", Arc::new(|| fixed("second")), RegisterPolicy::Keep)
            .unwrap();
        assert!(!kept.replaced);
        assert_eq!(kept.name, "gpt", "the surviving card's name is reported");
        assert_eq!(
            models.model("gpt").expect("still there").model_name(),
            "first"
        );

        let replaced = models
            .register_model_with("gpt", Arc::new(|| fixed("second")), RegisterPolicy::Replace)
            .unwrap();
        assert!(replaced.replaced);
        assert_eq!(
            models.model("gpt").expect("still there").model_name(),
            "second"
        );
        assert_eq!(models.len(), 1);
    }

    #[test]
    fn register_factory_carries_a_custom_name() {
        let models = ModelMgr::new();
        let registration = models
            .register_factory(
                ModelCard::named("gpt", "Primary"),
                Arc::new(|| fixed("gpt")),
                RegisterPolicy::Fail,
            )
            .unwrap();

        assert_eq!(registration.id, "gpt");
        assert_eq!(registration.name, "Primary");
        assert!(!registration.replaced);
        assert_eq!(
            models.models(),
            vec![ModelCard::named("gpt", "Primary")],
            "registration order is preserved"
        );
    }

    #[test]
    fn revision_counts_mutations_and_reset_zeroes_it() {
        let models = ModelMgr::new();
        assert_eq!(models.revision(), 0);

        models.register_model("gpt", fixed("gpt")).unwrap();
        assert_eq!(models.revision(), 1);

        models
            .register_model_with("gpt", Arc::new(|| fixed("other")), RegisterPolicy::Keep)
            .unwrap();
        assert_eq!(models.revision(), 1, "Keep is not a mutation");

        models
            .register_model_with("gpt", Arc::new(|| fixed("other")), RegisterPolicy::Replace)
            .unwrap();
        assert_eq!(models.revision(), 2, "Replace counts");

        assert_eq!(models.clear(), 1);
        assert_eq!(models.revision(), 3);
        assert_eq!(models.clear(), 0);
        assert_eq!(
            models.revision(),
            3,
            "clearing an empty registry is a no-op"
        );

        models.register_model("qwen", fixed("qwen")).unwrap();
        assert_eq!(models.reset(), 1);
        assert!(models.is_empty());
        assert_eq!(models.revision(), 0);
    }

    #[test]
    fn empty_and_whitespace_ids_are_rejected() {
        let models = ModelMgr::new();
        for id in ["", "   ", "\t\n"] {
            let error = models.register_model(id, fixed("ghost")).unwrap_err();
            assert_eq!(error.kind(), ErrorKind::Validation);
            assert_eq!(
                error.to_string(),
                "validation: model id cannot be empty or whitespace only"
            );
        }
        assert!(models.is_empty());
        assert_eq!(models.revision(), 0);
        assert_eq!(
            ModelMgr::default().len(),
            0,
            "Default builds an empty manager"
        );
    }
}
