//! Workflow components: the execution trait, its identity card, and ability defaults.
//!
//! Maps `ComponentExecutable`/`WorkflowComponent` from Python's
//! `openjiuwen.core.workflow.components.component` onto a single Rust trait. Python splits
//! execution (`ComponentExecutable`) from graph integration (`ComponentComposable`); in
//! Rust the engine owns graph wiring, so a component only declares what it can do
//! ([`Component::abilities`]) and how to do it ([`Component::invoke`] and friends).
//!
//! An ability that a component does not override fails with
//! [`Error::Unsupported`](openjiuwen_common::error::Error::Unsupported) rendered by
//! [`unsupported_ability`], preserving Python's message shape
//! `Component '<type>' is missing required method: <ability>()`.
//!
//! ```
//! use openjiuwen_workflow::ability::ComponentAbility;
//! use openjiuwen_workflow::component::ComponentCard;
//!
//! let card = ComponentCard::new("llm", "llm")
//!     .with_description("answers a question")
//!     .with_ability(ComponentAbility::Stream);
//! assert_eq!(card.id, "llm");
//! assert_eq!(
//!     card.abilities,
//!     vec![ComponentAbility::Invoke, ComponentAbility::Stream]
//! );
//! ```

use std::sync::Arc;

use futures::stream::BoxStream;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use openjiuwen_common::error::{Error, Result};

use crate::ability::ComponentAbility;
use crate::context::ComponentContext;

/// Identity and capability metadata of a component.
///
/// The card is what the builder records in [`crate::card::NodeSpec`] and what
/// `to_mermaid`/trace metadata read; it carries no runnable state. Field names are
/// `snake_case`, unknown fields are tolerated, and a card round-trips through JSON so a
/// workflow spec can be serialized and rebuilt.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ComponentCard {
    /// Unique component id; defaults to the component name.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Human-readable description.
    pub description: String,
    /// Component type discriminator (`"start"`, `"end"`, `"llm"`, …).
    pub component_type: String,
    /// Abilities the component implements.
    pub abilities: Vec<ComponentAbility>,
    /// Input template/schema metadata.
    pub input_params: Value,
    /// Output schema metadata.
    pub output_params: Value,
}

impl Default for ComponentCard {
    /// An unnamed card that can only be invoked.
    fn default() -> Self {
        Self {
            id: String::new(),
            name: String::new(),
            description: String::new(),
            component_type: String::new(),
            abilities: vec![ComponentAbility::Invoke],
            input_params: Value::Null,
            output_params: Value::Null,
        }
    }
}

impl ComponentCard {
    /// Build a card for `name`; the id defaults to the name.
    #[must_use]
    pub fn new(name: impl Into<String>, component_type: impl Into<String>) -> Self {
        let name = name.into();
        Self {
            id: name.clone(),
            name,
            component_type: component_type.into(),
            ..Self::default()
        }
    }

    /// Replace the component id.
    #[must_use]
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = id.into();
        self
    }

    /// Replace the component name.
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Replace the description.
    #[must_use]
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    /// Replace the ability list.
    ///
    /// Unlike [`ComponentCard::with_ability`] this overwrites the current list, so it
    /// can be used to declare an empty or non-`invoke`-first set.
    #[must_use]
    pub fn with_abilities(mut self, abilities: Vec<ComponentAbility>) -> Self {
        self.abilities = abilities;
        self
    }

    /// Add one ability; an ability already declared is not duplicated.
    #[must_use]
    pub fn with_ability(mut self, ability: ComponentAbility) -> Self {
        if !self.abilities.contains(&ability) {
            self.abilities.push(ability);
        }
        self
    }

    /// Replace the input template/schema metadata.
    #[must_use]
    pub fn with_input_params(mut self, input_params: Value) -> Self {
        self.input_params = input_params;
        self
    }

    /// Replace the output schema metadata.
    #[must_use]
    pub fn with_output_params(mut self, output_params: Value) -> Self {
        self.output_params = output_params;
        self
    }
}

/// A node of a workflow graph.
///
/// A component is driven by the engine in one of four modes, declared through
/// [`Component::abilities`]. Only the mode(s) actually declared need an implementation:
/// the defaults fail with [`unsupported_ability`].
///
/// Implementations are immutable during a run, so one component may serve concurrent
/// runs (`Component: Send + Sync`, and streams are `Send + 'static`). `Component` is
/// implemented for `Arc<T>` where `T: Component + ?Sized`, which makes
/// `Arc<dyn Component>` usable wherever a component is expected.
#[async_trait::async_trait]
pub trait Component: Send + Sync {
    /// The component's identity and capability metadata.
    fn card(&self) -> ComponentCard;

    /// The abilities this component implements; defaults to [`ComponentAbility::Invoke`].
    fn abilities(&self) -> Vec<ComponentAbility> {
        vec![ComponentAbility::Invoke]
    }

    /// The component type discriminator (`"start"`, `"end"`, `"llm"`, …).
    fn component_type(&self) -> &str;

    /// Runs once with batch inputs and returns a batch output.
    ///
    /// # Errors
    ///
    /// The default returns [`Error::Unsupported`]; implementations return their own
    /// execution errors, which the engine wraps in a component-execution error.
    async fn invoke(&self, _inputs: Value, _ctx: &ComponentContext) -> Result<Value> {
        Err(unsupported_ability(
            self.component_type(),
            ComponentAbility::Invoke,
        ))
    }

    /// Consumes batch inputs and yields a stream of frames.
    ///
    /// The stream is never empty: the default yields exactly one
    /// [`Error::Unsupported`].
    fn stream(
        self: Arc<Self>,
        _inputs: Value,
        _ctx: Arc<ComponentContext>,
    ) -> BoxStream<'static, Result<Value>> {
        let error = unsupported_ability(self.component_type(), ComponentAbility::Stream);
        futures::stream::once(async move { Err(error) }).boxed()
    }

    /// Consumes all buffered inputs from upstream producers and returns a batch output.
    ///
    /// # Errors
    ///
    /// The default returns [`Error::Unsupported`].
    async fn collect(&self, _inputs: Value, _ctx: &ComponentContext) -> Result<Value> {
        Err(unsupported_ability(
            self.component_type(),
            ComponentAbility::Collect,
        ))
    }

    /// Consumes a stream of inputs and yields a stream of frames.
    ///
    /// The stream is never empty: the default yields exactly one
    /// [`Error::Unsupported`].
    fn transform(
        self: Arc<Self>,
        _inputs: Value,
        _ctx: Arc<ComponentContext>,
    ) -> BoxStream<'static, Result<Value>> {
        let error = unsupported_ability(self.component_type(), ComponentAbility::Transform);
        futures::stream::once(async move { Err(error) }).boxed()
    }

    /// Chooses the next component ids for a branch; `None` for ordinary components.
    ///
    /// # Errors
    ///
    /// Branch implementations report a malformed condition or an unknown target.
    fn route(&self, _source: &Value) -> Option<Result<Vec<String>>> {
        None
    }

    /// Static facts merged into the component's workflow trace span; `None` adds nothing.
    fn trace_metadata(&self) -> Option<Value> {
        None
    }
}

#[async_trait::async_trait]
impl<T: Component + ?Sized> Component for Arc<T> {
    fn card(&self) -> ComponentCard {
        (**self).card()
    }

    fn abilities(&self) -> Vec<ComponentAbility> {
        (**self).abilities()
    }

    fn component_type(&self) -> &str {
        (**self).component_type()
    }

    async fn invoke(&self, inputs: Value, ctx: &ComponentContext) -> Result<Value> {
        (**self).invoke(inputs, ctx).await
    }

    fn stream(
        self: Arc<Self>,
        inputs: Value,
        ctx: Arc<ComponentContext>,
    ) -> BoxStream<'static, Result<Value>> {
        T::stream((*self).clone(), inputs, ctx)
    }

    async fn collect(&self, inputs: Value, ctx: &ComponentContext) -> Result<Value> {
        (**self).collect(inputs, ctx).await
    }

    fn transform(
        self: Arc<Self>,
        inputs: Value,
        ctx: Arc<ComponentContext>,
    ) -> BoxStream<'static, Result<Value>> {
        T::transform((*self).clone(), inputs, ctx)
    }

    fn route(&self, source: &Value) -> Option<Result<Vec<String>>> {
        (**self).route(source)
    }

    fn trace_metadata(&self) -> Option<Value> {
        (**self).trace_metadata()
    }
}

/// The error a component returns for an ability it does not implement.
///
/// The text keeps Python's shape, for example
/// `Component 'llm' is missing required method: stream()`.
pub fn unsupported_ability(component_type: &str, ability: ComponentAbility) -> Error {
    Error::unsupported(format!(
        "Component '{component_type}' is missing required method: {ability}()"
    ))
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::assert_is_empty,
    clippy::unnecessary_literal_bound
)]
mod tests {
    use std::sync::Arc;

    use futures::StreamExt;
    use openjiuwen_common::error::ErrorKind;
    use openjiuwen_session::{Session, WorkflowSession};
    use serde_json::json;

    use super::*;

    /// A component that implements `invoke` only.
    struct FixedComponent;

    #[async_trait::async_trait]
    impl Component for FixedComponent {
        fn card(&self) -> ComponentCard {
            ComponentCard::new("fixed", "fixed")
        }

        fn component_type(&self) -> &str {
            "fixed"
        }

        async fn invoke(&self, inputs: Value, _ctx: &ComponentContext) -> Result<Value> {
            Ok(json!({"echo": inputs}))
        }
    }

    /// A component that implements nothing beyond the required identity methods.
    struct BareComponent;

    #[async_trait::async_trait]
    impl Component for BareComponent {
        fn card(&self) -> ComponentCard {
            ComponentCard::new("bare", "bare")
        }

        fn component_type(&self) -> &str {
            "bare"
        }
    }

    fn session() -> Arc<dyn Session> {
        Arc::new(WorkflowSession::new("component-tests"))
    }

    fn context() -> ComponentContext {
        ComponentContext::new(session(), "c1", "wf")
    }

    fn describe(component: &impl Component) -> (String, Vec<ComponentAbility>) {
        (component.component_type().to_owned(), component.abilities())
    }

    #[test]
    fn default_card_declares_invoke_only() {
        let card = ComponentCard::default();
        assert_eq!(card.abilities, vec![ComponentAbility::Invoke]);
        assert!(card.id.is_empty());
        assert!(card.name.is_empty());
        assert!(card.component_type.is_empty());
        assert_eq!(card.input_params, Value::Null);
        assert_eq!(card.output_params, Value::Null);
    }

    #[test]
    fn new_derives_the_id_from_the_name() {
        let card = ComponentCard::new("answer", "llm");
        assert_eq!(card.id, "answer");
        assert_eq!(card.name, "answer");
        assert_eq!(card.component_type, "llm");
        assert_eq!(card.abilities, vec![ComponentAbility::Invoke]);
    }

    #[test]
    fn card_builders_are_chainable() {
        let card = ComponentCard::new("n", "t")
            .with_id("id")
            .with_name("name")
            .with_description("desc")
            .with_abilities(vec![ComponentAbility::Collect])
            .with_ability(ComponentAbility::Collect)
            .with_ability(ComponentAbility::Invoke)
            .with_input_params(json!({"a": "${start.a}"}))
            .with_output_params(json!({"b": "string"}));

        assert_eq!(card.id, "id");
        assert_eq!(card.name, "name");
        assert_eq!(card.description, "desc");
        assert_eq!(card.component_type, "t");
        assert_eq!(
            card.abilities,
            vec![ComponentAbility::Collect, ComponentAbility::Invoke]
        );
        assert_eq!(card.input_params, json!({"a": "${start.a}"}));
        assert_eq!(card.output_params, json!({"b": "string"}));
    }

    #[test]
    fn component_card_round_trips_and_tolerates_unknown_fields() {
        let card = ComponentCard::new("llm", "llm")
            .with_description("answers")
            .with_ability(ComponentAbility::Stream)
            .with_input_params(json!({"query": "${start.query}"}));

        let wire = serde_json::to_value(&card).unwrap();
        assert_eq!(wire["component_type"], json!("llm"));
        assert_eq!(wire["abilities"], json!(["invoke", "stream"]));

        let parsed: ComponentCard = serde_json::from_value(json!({
            "id": "llm",
            "name": "llm",
            "description": "answers",
            "component_type": "llm",
            "abilities": ["invoke", "stream"],
            "input_params": {"query": "${start.query}"},
            "output_params": null,
            "future_field": {"ignored": true}
        }))
        .unwrap();
        assert_eq!(parsed, card);

        let sparse: ComponentCard = serde_json::from_value(json!({"id": "x"})).unwrap();
        assert_eq!(sparse.id, "x");
        assert_eq!(sparse.abilities, vec![ComponentAbility::Invoke]);
    }

    #[test]
    fn abilities_default_to_invoke() {
        assert_eq!(
            BareComponent.abilities(),
            vec![ComponentAbility::Invoke],
            "a component that overrides nothing is invoke-only"
        );
        assert_eq!(
            FixedComponent.abilities(),
            vec![ComponentAbility::Invoke],
            "the same default applies to components that implement invoke"
        );
    }

    #[tokio::test]
    async fn default_invoke_is_unsupported() {
        let error = BareComponent
            .invoke(json!({"query": "hi"}), &context())
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        assert_eq!(
            error.message(),
            "Component 'bare' is missing required method: invoke()"
        );
    }

    #[tokio::test]
    async fn default_collect_is_unsupported() {
        let error = BareComponent
            .collect(json!({"query": "hi"}), &context())
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        assert_eq!(
            error.message(),
            "Component 'bare' is missing required method: collect()"
        );
    }

    #[tokio::test]
    async fn default_stream_yields_one_unsupported_error() {
        let component: Arc<dyn Component> = Arc::new(BareComponent);
        let frames: Vec<_> = component
            .stream(json!({}), Arc::new(context()))
            .collect()
            .await;
        assert_eq!(frames.len(), 1);
        let error = frames.into_iter().next().unwrap().unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        assert_eq!(
            error.message(),
            "Component 'bare' is missing required method: stream()"
        );
    }

    #[tokio::test]
    async fn default_transform_yields_one_unsupported_error() {
        let component: Arc<dyn Component> = Arc::new(BareComponent);
        let frames: Vec<_> = component
            .transform(json!({}), Arc::new(context()))
            .collect()
            .await;
        assert_eq!(frames.len(), 1);
        let error = frames.into_iter().next().unwrap().unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        assert_eq!(
            error.message(),
            "Component 'bare' is missing required method: transform()"
        );
    }

    #[test]
    fn default_route_and_trace_metadata_are_empty() {
        assert!(BareComponent.route(&json!({"a": 1})).is_none());
        assert_eq!(BareComponent.trace_metadata(), None);
    }

    #[tokio::test]
    async fn arc_component_forwards_to_the_inner_value() {
        let inner = Arc::new(FixedComponent);
        // `Arc<T>` satisfies `Component` itself, so it can be used generically…
        assert_eq!(
            describe(&inner),
            ("fixed".to_owned(), vec![ComponentAbility::Invoke])
        );
        // …and calls reach the inner implementation.
        let forwarded = Arc::clone(&inner)
            .invoke(json!({"x": 1}), &context())
            .await
            .unwrap();
        assert_eq!(forwarded, json!({"echo": {"x": 1}}));
        assert_eq!(Arc::clone(&inner).card().id, "fixed");
    }

    #[tokio::test]
    async fn arc_dyn_component_forwards_and_keeps_default_abilities() {
        let component: Arc<dyn Component> = Arc::new(FixedComponent);
        assert_eq!(component.component_type(), "fixed");
        assert_eq!(component.abilities(), vec![ComponentAbility::Invoke]);
        assert_eq!(
            component.invoke(json!({"x": 2}), &context()).await.unwrap(),
            json!({"echo": {"x": 2}})
        );
    }
}
