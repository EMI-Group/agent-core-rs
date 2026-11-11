//! Built-in `tool` workflow component.
//!
//! Ports `openjiuwen/core/workflow/components/tool/tool_comp.py`. The component
//! adapts any [`Tool`] to the workflow seam: it forwards the resolved inputs, then
//! reports the result in Python's `ToolComponentOutput` shape instead of raising.
//!
//! # Error mapping
//!
//! A tool failure is *captured*, never propagated — this is Python parity:
//! `invoke` always returns a JSON object and the failure lands in `error_code` and
//! `error_message`. The numeric code follows Python's split:
//!
//! - [`ErrorKind::Tool`] and [`ErrorKind::Validation`] (a rejected call or bad
//!   arguments) map to [`COMPONENT_TOOL_EXECUTION_ERROR`] (`102000`);
//! - every other error kind (transport, timeout, cancellation, …) maps to
//!   [`TOOL_EXECUTION_ERROR`] (`182012`).
//!
//! The component never panics: a missing tool is a construction error, and every
//! runtime failure is data.

use std::fmt;
use std::sync::Arc;

use serde_json::{json, Value};

use openjiuwen_common::error::{ErrorKind, Result};
use openjiuwen_foundation::{Tool, ToolContext};

use crate::ability::ComponentAbility;
use crate::component::{Component, ComponentCard};
use crate::context::ComponentContext;
use crate::error::{COMPONENT_TOOL_EXECUTION_ERROR, TOOL_EXECUTION_ERROR};

/// The built-in `tool` component.
///
/// `invoke` calls the wrapped tool and returns
/// `{"error_code": <code>, "error_message": <text>, "data": <value>}`.
pub struct ToolComponent {
    tool: Arc<dyn Tool>,
}

impl fmt::Debug for ToolComponent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let card = self.tool.card();
        f.debug_struct("ToolComponent")
            .field("id", &card.id)
            .field("name", &card.name)
            .finish()
    }
}

impl ToolComponent {
    /// Wraps `tool`.
    #[must_use]
    pub fn new(tool: Arc<dyn Tool>) -> Self {
        Self { tool }
    }

    /// The wrapped tool.
    #[must_use]
    pub fn tool(&self) -> &Arc<dyn Tool> {
        &self.tool
    }

    /// The numeric code reported for a captured tool failure.
    ///
    /// See the module docs for the mapping.
    fn error_code(kind: ErrorKind) -> u32 {
        match kind {
            ErrorKind::Tool | ErrorKind::Validation => COMPONENT_TOOL_EXECUTION_ERROR,
            _ => TOOL_EXECUTION_ERROR,
        }
    }
}

#[async_trait::async_trait]
#[allow(clippy::unnecessary_literal_bound)]
impl Component for ToolComponent {
    fn card(&self) -> ComponentCard {
        let card = self.tool.card();
        ComponentCard::new(card.name.clone(), "tool")
            .with_id(card.id.clone())
            .with_description(card.description.clone())
            .with_input_params(card.input_params.clone())
            .with_abilities(vec![ComponentAbility::Invoke])
    }

    fn abilities(&self) -> Vec<ComponentAbility> {
        vec![ComponentAbility::Invoke]
    }

    fn component_type(&self) -> &str {
        "tool"
    }

    async fn invoke(&self, inputs: Value, ctx: &ComponentContext) -> Result<Value> {
        let tool_ctx = ToolContext {
            session_id: Some(ctx.session().id().to_owned()),
            metadata: ctx.metadata().clone(),
            cancellation: ctx.cancellation().clone(),
            ..ToolContext::default()
        };
        match self.tool.invoke(inputs, &tool_ctx).await {
            Ok(value) => Ok(json!({
                "error_code": 0,
                "error_message": "",
                "data": value,
            })),
            Err(error) => Ok(json!({
                "error_code": Self::error_code(error.kind()),
                "error_message": error.message(),
                "data": Value::Null,
            })),
        }
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::needless_pass_by_value
)]
mod tests {
    use std::sync::Arc;

    use async_trait::async_trait;
    use openjiuwen_common::error::{ErrorKind, Result};
    use openjiuwen_foundation::{Tool, ToolCard, ToolContext};
    use openjiuwen_session::{Session, WorkflowSession};
    use openjiuwen_test_support::ScriptedError;
    use parking_lot::Mutex;
    use serde_json::{json, Value};

    use super::*;

    fn context() -> ComponentContext {
        let session: Arc<dyn Session> = Arc::new(WorkflowSession::new("tool-tests"));
        ComponentContext::new(session, "tool", "wf")
    }

    /// A tool whose behavior and last context are controlled by the test.
    struct FakeTool {
        card: ToolCard,
        result: std::result::Result<Value, ScriptedError>,
        last_ctx: Arc<Mutex<Option<ToolContext>>>,
        last_inputs: Arc<Mutex<Option<Value>>>,
    }

    impl FakeTool {
        fn new(result: std::result::Result<Value, ScriptedError>) -> Self {
            Self {
                card: ToolCard::new("echo", "Echo tool", json!({"type": "object"})),
                result,
                last_ctx: Arc::new(Mutex::new(None)),
                last_inputs: Arc::new(Mutex::new(None)),
            }
        }
    }

    #[async_trait]
    impl Tool for FakeTool {
        fn card(&self) -> &ToolCard {
            &self.card
        }

        async fn invoke(&self, inputs: Value, ctx: &ToolContext) -> Result<Value> {
            *self.last_inputs.lock() = Some(inputs);
            *self.last_ctx.lock() = Some(ctx.clone());
            match &self.result {
                Ok(value) => Ok(value.clone()),
                Err(error) => Err(error.clone().into_error()),
            }
        }
    }

    #[tokio::test]
    async fn success_reports_code_zero_and_the_data() {
        let tool = Arc::new(FakeTool::new(Ok(json!({"echoed": true}))));
        let component = ToolComponent::new(tool);
        let output = component.invoke(json!({"x": 1}), &context()).await.unwrap();
        assert_eq!(
            output,
            json!({"error_code": 0, "error_message": "", "data": {"echoed": true}})
        );
    }

    #[tokio::test]
    async fn tool_errors_are_captured_with_code_102000() {
        let tool = Arc::new(FakeTool::new(Err(ScriptedError::with_kind(
            ErrorKind::Tool,
            "kaboom",
        ))));
        let component = ToolComponent::new(tool);
        let output = component.invoke(json!({}), &context()).await.unwrap();
        assert_eq!(
            output,
            json!({"error_code": 102_000, "error_message": "kaboom", "data": null})
        );
    }

    #[tokio::test]
    async fn validation_errors_use_the_same_component_code() {
        let tool = Arc::new(FakeTool::new(Err(ScriptedError::validation("bad args"))));
        let component = ToolComponent::new(tool);
        let output = component.invoke(json!({}), &context()).await.unwrap();
        assert_eq!(output["error_code"], json!(102_000));
        assert_eq!(output["error_message"], json!("bad args"));
        assert_eq!(output["data"], Value::Null);
    }

    #[tokio::test]
    async fn other_errors_use_the_shared_tool_code() {
        let tool = Arc::new(FakeTool::new(Err(ScriptedError::transport("offline"))));
        let component = ToolComponent::new(tool);
        let output = component.invoke(json!({}), &context()).await.unwrap();
        assert_eq!(output["error_code"], json!(182_012));
        assert_eq!(output["error_message"], json!("offline"));
        assert_eq!(output["data"], Value::Null);
    }

    #[tokio::test]
    async fn inputs_and_session_are_forwarded() {
        let tool = Arc::new(FakeTool::new(Ok(json!(null))));
        let captured = Arc::clone(&tool);
        let component = ToolComponent::new(tool);
        let ctx = context();

        let inputs = json!({"a": [1, 2]});
        component.invoke(inputs.clone(), &ctx).await.unwrap();

        assert_eq!(*captured.last_inputs.lock(), Some(inputs));
        let tool_ctx = captured.last_ctx.lock().clone().expect("context captured");
        assert_eq!(tool_ctx.session_id.as_deref(), Some("tool-tests"));
    }

    #[test]
    fn card_is_derived_from_the_tool_card() {
        let tool = Arc::new(FakeTool::new(Ok(json!(null))));
        let expected = tool.card().clone();
        let component = ToolComponent::new(tool);
        let card = component.card();
        assert_eq!(card.id, expected.id);
        assert_eq!(card.name, "echo");
        assert_eq!(card.description, "Echo tool");
        assert_eq!(card.input_params, json!({"type": "object"}));
        assert_eq!(card.component_type, "tool");
        assert_eq!(component.component_type(), "tool");
        assert_eq!(component.abilities(), vec![ComponentAbility::Invoke]);
    }
}
