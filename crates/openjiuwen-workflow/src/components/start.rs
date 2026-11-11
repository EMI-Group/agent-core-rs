//! Built-in `start` workflow component.
//!
//! Ports Python's `openjiuwen.core.workflow.components.flow.start_comp.Start`.
//! A start node is a pure pass-through: the engine resolves the component's
//! `inputs_schema` against workflow state and [`StartComponent::invoke`] hands the
//! resolved object back unchanged, so every downstream `${start.field}` reference reads
//! what the caller supplied.
//!
//! ```ignore
//! // Start → Llm → End
//! workflow.set_start_comp("start", StartComponent::new(), ComponentOptions::default());
//! ```

use serde_json::Value;

use openjiuwen_common::error::Result;

use crate::ability::ComponentAbility;
use crate::component::{Component, ComponentCard};
use crate::context::ComponentContext;

/// The `start` built-in component.
///
/// A start node is a pass-through: it owns no state and returns its resolved inputs
/// unchanged, so the workflow's entry data is available as `${start.<field>}`.
#[derive(Debug, Default, Clone)]
pub struct StartComponent;

impl StartComponent {
    /// Creates a start component.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

#[async_trait::async_trait]
impl Component for StartComponent {
    fn card(&self) -> ComponentCard {
        ComponentCard::new("start", "start").with_abilities(self.abilities())
    }

    fn abilities(&self) -> Vec<ComponentAbility> {
        vec![ComponentAbility::Invoke]
    }

    fn component_type(&self) -> &'static str {
        "start"
    }

    async fn invoke(&self, inputs: Value, _ctx: &ComponentContext) -> Result<Value> {
        Ok(inputs)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]
mod tests {
    use std::sync::Arc;

    use openjiuwen_session::{Session, WorkflowSession};
    use serde_json::json;

    use super::*;

    fn context() -> ComponentContext {
        let session: Arc<dyn Session> = Arc::new(WorkflowSession::new("start-tests"));
        ComponentContext::new(session, "start", "wf")
    }

    #[test]
    fn identity_matches_python() {
        let component = StartComponent::new();
        assert_eq!(component.component_type(), "start");
        assert_eq!(component.abilities(), vec![ComponentAbility::Invoke]);
        let card = component.card();
        assert_eq!(card.id, "start");
        assert_eq!(card.name, "start");
        assert_eq!(card.component_type, "start");
        assert_eq!(card.abilities, vec![ComponentAbility::Invoke]);
    }

    #[tokio::test]
    async fn invoke_returns_the_inputs_unchanged() {
        let component = StartComponent::new();
        let inputs = json!({"query": "hello", "nested": {"n": 1}});
        let output = component.invoke(inputs.clone(), &context()).await.unwrap();
        assert_eq!(output, inputs);
    }

    #[tokio::test]
    async fn invoke_passes_null_and_scalars_through() {
        let component = StartComponent::new();
        assert_eq!(
            component.invoke(Value::Null, &context()).await.unwrap(),
            Value::Null
        );
        assert_eq!(
            component.invoke(json!(42), &context()).await.unwrap(),
            json!(42)
        );
    }
}
