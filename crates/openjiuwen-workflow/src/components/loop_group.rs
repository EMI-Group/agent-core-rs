//! The `loop_group` workflow component: bounded iteration with loop variables.
//!
//! Ports Python's `LoopGroup` (`openjiuwen/core/workflow/components/flow/loop/loop_comp.py`)
//! onto this crate's flat body model. Python's `LoopGroup` is a nested `BaseWorkflow`
//! compiled into the loop node; here the body is a flat list of components run
//! sequentially per iteration by the shared driver
//! ([`crate::components::loop_driver`]), and the run is summarized as one wrapper value.
//!
//! Three things differ from the original [`LoopComponent`](crate::components::LoopComponent):
//!
//! - it carries **loop variables** initialized from `loop_variables`, exposed to the body
//!   (and to the closing condition) every iteration, and updated by a
//!   [`LoopSetVariableComponent`](crate::components::LoopSetVariableComponent);
//! - it can be **unbounded** (`loop_type: "until_break"`), in which case only a
//!   [`LoopBreakComponent`](crate::components::LoopBreakComponent) ends it, capped by
//!   `loop_number_max_limit`;
//! - its output reports the loop variables and the break flag besides the round results.
//!
//! ```
//! use openjiuwen_workflow::components::loop_group::{LoopGroupComponent, LoopGroupConfig};
//!
//! // `loop_number_max_limit` is the cap an unbounded group falls back on.
//! let config = LoopGroupConfig::new().with_loop_number(3).with_max_iterations(10);
//! let component = LoopGroupComponent::new(config, Vec::new());
//! assert_eq!(component.config().loop_number, Some(3));
//! assert_eq!(component.config().loop_number_max_limit, 10);
//! ```

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use openjiuwen_common::error::Result;

use crate::ability::ComponentAbility;
use crate::component::{Component, ComponentCard};
use crate::context::ComponentContext;

use crate::components::loop_control::BREAK_COMPONENT_TYPE;
use crate::components::loop_driver::{self, BoundParams, LoopBound, DEFAULT_MAX_ITERATIONS};

/// Static configuration of a [`LoopGroupComponent`].
///
/// Field names are `snake_case`, unknown fields are tolerated, and the type round-trips
/// through JSON so a workflow spec can be serialized and rebuilt. `loop_number_max_limit`
/// also answers to `max_iterations`, and `bool_expression` to `loop_condition`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct LoopGroupConfig {
    /// How the group is bounded; defaults to [`LoopBound::Number`].
    pub loop_type: LoopBound,
    /// The iteration cap for `loop_number` and for an unbounded group.
    #[serde(alias = "max_iterations")]
    pub loop_number_max_limit: u64,
    /// Iteration count for [`LoopBound::Number`].
    pub loop_number: Option<u64>,
    /// Input key (or whole-string `${...}` reference) naming the array for
    /// [`LoopBound::Array`].
    pub loop_array: Option<String>,
    /// Condition expression for [`LoopBound::Condition`], re-checked every iteration.
    #[serde(alias = "loop_condition")]
    pub bool_expression: Option<String>,
    /// Initial loop variables, exposed to the body and updatable from it.
    pub loop_variables: BTreeMap<String, Value>,
}

impl Default for LoopGroupConfig {
    fn default() -> Self {
        Self {
            loop_type: LoopBound::Number,
            loop_number_max_limit: DEFAULT_MAX_ITERATIONS,
            loop_number: None,
            loop_array: None,
            bool_expression: None,
            loop_variables: BTreeMap::new(),
        }
    }
}

impl LoopGroupConfig {
    /// The default configuration of `loop_type`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the bound kind.
    #[must_use]
    pub fn with_loop_type(mut self, loop_type: LoopBound) -> Self {
        self.loop_type = loop_type;
        self
    }

    /// Sets `loop_number`.
    #[must_use]
    pub fn with_loop_number(mut self, loop_number: u64) -> Self {
        self.loop_number = Some(loop_number);
        self
    }

    /// Sets the array key or `${...}` reference used by [`LoopBound::Array`].
    #[must_use]
    pub fn with_loop_array(mut self, loop_array: impl Into<String>) -> Self {
        self.loop_array = Some(loop_array.into());
        self
    }

    /// Sets the condition expression used by [`LoopBound::Condition`].
    #[must_use]
    pub fn with_bool_expression(mut self, expression: impl Into<String>) -> Self {
        self.bool_expression = Some(expression.into());
        self
    }

    /// Sets the alternative `loop_condition` spelling of [`Self::bool_expression`].
    #[must_use]
    pub fn with_loop_condition(self, expression: impl Into<String>) -> Self {
        self.with_bool_expression(expression)
    }

    /// Sets the iteration cap.
    #[must_use]
    pub fn with_max_iterations(mut self, loop_number_max_limit: u64) -> Self {
        self.loop_number_max_limit = loop_number_max_limit;
        self
    }

    /// Replaces the initial loop variables.
    #[must_use]
    pub fn with_loop_variables(mut self, loop_variables: BTreeMap<String, Value>) -> Self {
        self.loop_variables = loop_variables;
        self
    }

    /// Adds one initial loop variable.
    #[must_use]
    pub fn with_variable(mut self, name: impl Into<String>, value: Value) -> Self {
        let _ = self.loop_variables.insert(name.into(), value);
        self
    }
}

/// The `loop_group` component: iterate a flat body under a bound, carrying loop variables.
///
/// The body runs sequentially per iteration, each later component receiving the previous
/// one's output; an empty body echoes the iteration inputs. Every iteration's inputs are
/// the group's resolved inputs overlaid with
/// `{"index", "item", "previous"}` plus the loop variables (both flattened as top-level
/// keys and collected under `"loop_variables"`). The output is the documented container
/// wrapper `{"loop_results", "loop_variables", "loop_count", "broken"}`.
pub struct LoopGroupComponent {
    config: LoopGroupConfig,
    body: Vec<Arc<dyn Component>>,
}

impl fmt::Debug for LoopGroupComponent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LoopGroupComponent")
            .field("config", &self.config)
            .field("body_len", &self.body.len())
            .finish()
    }
}

impl LoopGroupComponent {
    /// Creates a group with `config` and `body`.
    #[must_use]
    pub fn new(config: LoopGroupConfig, body: Vec<Arc<dyn Component>>) -> Self {
        Self { config, body }
    }

    /// The component configuration.
    #[must_use]
    pub fn config(&self) -> &LoopGroupConfig {
        &self.config
    }

    /// The number of body components run per iteration.
    #[must_use]
    pub fn body_len(&self) -> usize {
        self.body.len()
    }

    /// Whether the body holds a [`LoopBreakComponent`](crate::components::LoopBreakComponent).
    ///
    /// The scan looks at the immediate body only; a break inside a nested container ends
    /// that container, which reports `broken` in its own output.
    #[must_use]
    pub fn has_break_component(&self) -> bool {
        self.body
            .iter()
            .any(|component| component.component_type() == BREAK_COMPONENT_TYPE)
    }

    /// The bound parameters of this component's configuration.
    fn bound_params(&self) -> BoundParams<'_> {
        BoundParams::new(
            self.config.loop_number,
            self.config.loop_array.as_deref(),
            self.config.bool_expression.as_deref(),
        )
    }
}

#[async_trait::async_trait]
impl Component for LoopGroupComponent {
    fn card(&self) -> ComponentCard {
        ComponentCard::new("loop_group", "loop_group").with_abilities(self.abilities())
    }

    fn abilities(&self) -> Vec<ComponentAbility> {
        vec![ComponentAbility::Invoke]
    }

    fn component_type(&self) -> &'static str {
        "loop_group"
    }

    /// Rejects a loop that cannot run or cannot terminate, before any run.
    ///
    /// An unbounded group (`until_break`) needs a
    /// [`LoopBreakComponent`](crate::components::LoopBreakComponent) in its body; otherwise
    /// only the iteration cap would end it.
    fn validate_config(&self) -> Result<()> {
        loop_driver::validate_bound(
            self.config.loop_type,
            self.bound_params(),
            self.config.loop_number_max_limit,
        )?;
        if self.config.loop_type.needs_break() && !self.has_break_component() {
            return Err(loop_driver::no_terminator());
        }
        Ok(())
    }

    async fn invoke(&self, inputs: Value, ctx: &ComponentContext) -> Result<Value> {
        let base = loop_driver::object_inputs(inputs, ctx)?;
        let limit = loop_driver::effective_limit(self.config.loop_number_max_limit, ctx);
        let spec = loop_driver::plan(
            self.config.loop_type,
            self.bound_params(),
            &base,
            limit,
            ctx.component_id(),
        )?;
        let variables = loop_driver::resolve_variables(&self.config.loop_variables, &base);
        let run = loop_driver::run_spec(&spec, &self.body, &base, variables, limit, ctx).await?;
        Ok(loop_driver::container_output(&run))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]
mod tests {
    use std::sync::Arc;

    use openjiuwen_common::error::ErrorKind;
    use openjiuwen_session::{Session, WorkflowSession};
    use serde_json::json;

    use super::*;
    use crate::components::{LoopBreakComponent, LoopSetVariableComponent};

    /// Reads the running `total` and increments it, so a group that carries variables
    /// produces `{"total": 1..=n}`.
    #[derive(Debug)]
    struct IncrementComponent;

    #[async_trait::async_trait]
    impl Component for IncrementComponent {
        fn card(&self) -> ComponentCard {
            ComponentCard::new("increment", "increment")
        }

        fn component_type(&self) -> &'static str {
            "increment"
        }

        async fn invoke(&self, inputs: Value, _ctx: &ComponentContext) -> Result<Value> {
            let total = inputs["total"].as_u64().unwrap_or(0) + 1;
            Ok(json!({"total": total}))
        }
    }

    fn context() -> ComponentContext {
        let session: Arc<dyn Session> = Arc::new(WorkflowSession::new("loop-group-tests"));
        ComponentContext::new(session, "loop_group", "wf")
    }

    fn body(break_at: Option<&str>) -> Vec<Arc<dyn Component>> {
        let mut body: Vec<Arc<dyn Component>> = Vec::new();
        if let Some(condition) = break_at {
            body.push(Arc::new(
                LoopBreakComponent::new().with_condition(condition),
            ));
        }
        body.push(Arc::new(IncrementComponent));
        body.push(Arc::new(LoopSetVariableComponent::new(
            [("${loop.total}".to_owned(), json!("${total}"))].into(),
        )));
        body
    }

    fn results(output: &Value) -> &Vec<Value> {
        output["loop_results"].as_array().unwrap()
    }

    #[tokio::test]
    async fn a_number_group_carries_variables_across_iterations() {
        let component =
            LoopGroupComponent::new(LoopGroupConfig::new().with_loop_number(3), body(None));
        let output = component
            .invoke(json!({"seed": 1}), &context())
            .await
            .unwrap();

        assert_eq!(
            results(&output),
            &vec![
                json!({"total": 1}),
                json!({"total": 2}),
                json!({"total": 3})
            ]
        );
        assert_eq!(output["loop_variables"], json!({"total": 3}));
        assert_eq!(output["loop_count"], json!(3));
        assert_eq!(output["broken"], json!(false));
    }

    #[tokio::test]
    async fn initial_variables_template_against_the_inputs() {
        let component = LoopGroupComponent::new(
            LoopGroupConfig::new()
                .with_loop_number(2)
                .with_variable("total", json!("${start.total}")),
            body(None),
        );
        let output = component
            .invoke(json!({"start": {"total": 10}}), &context())
            .await
            .unwrap();
        assert_eq!(
            results(&output),
            &vec![json!({"total": 11}), json!({"total": 12})]
        );
        assert_eq!(output["loop_variables"], json!({"total": 12}));
    }

    #[tokio::test]
    async fn an_array_group_iterates_over_the_referenced_items() {
        let component = LoopGroupComponent::new(
            LoopGroupConfig::new()
                .with_loop_type(LoopBound::Array)
                .with_loop_array("${start.items}"),
            body(None),
        );
        let output = component
            .invoke(json!({"start": {"items": ["a", "b"]}}), &context())
            .await
            .unwrap();
        assert_eq!(output["loop_count"], json!(2));
        assert_eq!(results(&output)[1], json!({"total": 2}));
    }

    #[tokio::test]
    async fn a_condition_group_rechecks_the_condition_every_iteration() {
        let component = LoopGroupComponent::new(
            LoopGroupConfig::new()
                .with_loop_type(LoopBound::Condition)
                .with_bool_expression("${index} < 2"),
            body(None),
        );
        let output = component.invoke(json!({}), &context()).await.unwrap();
        assert_eq!(output["loop_count"], json!(2));

        // A condition reading a loop variable sees the previous iteration's update, so
        // `total` (starting at 0) reaches 2 before the loop stops.
        let component = LoopGroupComponent::new(
            LoopGroupConfig::new()
                .with_loop_type(LoopBound::Condition)
                .with_bool_expression("${total} < 2")
                .with_variable("total", json!(0)),
            body(None),
        );
        let output = component.invoke(json!({}), &context()).await.unwrap();
        assert_eq!(output["loop_count"], json!(2));
        assert_eq!(output["loop_variables"], json!({"total": 2}));
    }

    #[tokio::test]
    async fn an_unbounded_group_stops_on_a_break_component() {
        let component = LoopGroupComponent::new(
            LoopGroupConfig::new()
                .with_loop_type(LoopBound::UntilBreak)
                .with_max_iterations(10),
            body(Some("${index} >= 1")),
        );
        let output = component.invoke(json!({}), &context()).await.unwrap();

        assert_eq!(output["broken"], json!(true));
        assert_eq!(output["loop_count"], json!(1));
        assert_eq!(results(&output), &vec![json!({"total": 1})]);
    }

    #[tokio::test]
    async fn an_unbounded_group_without_a_break_hits_the_limit() {
        let component = LoopGroupComponent::new(
            LoopGroupConfig::new()
                .with_loop_type(LoopBound::UntilBreak)
                .with_max_iterations(2),
            body(None),
        );
        let error = component.invoke(json!({}), &context()).await.unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Workflow);
        assert!(error.message().contains("[101040]"), "{error}");
        assert!(error.message().contains("maximum limit 2"), "{error}");
    }

    #[test]
    fn validate_config_rejects_unrunnable_groups() {
        let missing = LoopGroupComponent::new(LoopGroupConfig::new(), Vec::new());
        let error = missing.validate_config().unwrap_err();
        assert!(error.message().contains("[101030]"), "{error}");
        assert!(error.message().contains("loop_number variable not found"));

        let over_cap = LoopGroupComponent::new(
            LoopGroupConfig::new()
                .with_loop_number(11)
                .with_max_iterations(10),
            Vec::new(),
        );
        let error = over_cap.validate_config().unwrap_err();
        assert!(error
            .message()
            .contains("loop_number exceeds maximum limit 10"));

        let unterminated = LoopGroupComponent::new(
            LoopGroupConfig::new().with_loop_type(LoopBound::UntilBreak),
            body(None),
        );
        let error = unterminated.validate_config().unwrap_err();
        assert!(error.message().contains("cannot terminate"), "{error}");

        let bounded = LoopGroupComponent::new(
            LoopGroupConfig::new().with_loop_type(LoopBound::UntilBreak),
            body(Some("${index} >= 1")),
        );
        bounded.validate_config().unwrap();
        assert!(bounded.has_break_component());
        assert!(
            !LoopGroupComponent::new(LoopGroupConfig::new().with_loop_number(1), body(None))
                .has_break_component()
        );
    }

    #[tokio::test]
    async fn non_object_inputs_are_rejected() {
        let component =
            LoopGroupComponent::new(LoopGroupConfig::new().with_loop_number(1), Vec::new());
        let error = component.invoke(json!(7), &context()).await.unwrap_err();
        assert!(error.message().contains("inputs must be an object"));
    }

    #[test]
    fn config_parses_python_aliases_and_round_trips() {
        let config: LoopGroupConfig = serde_json::from_value(json!({
            "loop_type": "condition",
            "loop_condition": "${index} < 3",
            "max_iterations": 7,
            "loop_variables": {"total": 0},
            "future_field": true,
        }))
        .unwrap();
        assert_eq!(config.loop_type, LoopBound::Condition);
        assert_eq!(config.bool_expression.as_deref(), Some("${index} < 3"));
        assert_eq!(config.loop_number_max_limit, 7);
        assert_eq!(config.loop_variables.get("total"), Some(&json!(0)));

        let wire = serde_json::to_value(&config).unwrap();
        assert_eq!(wire["bool_expression"], json!("${index} < 3"));
        assert_eq!(wire["loop_number_max_limit"], json!(7));
        assert_eq!(
            serde_json::from_value::<LoopGroupConfig>(wire).unwrap(),
            config
        );
        assert_eq!(
            serde_json::from_value::<LoopGroupConfig>(json!({})).unwrap(),
            LoopGroupConfig::default()
        );
        assert_eq!(LoopGroupConfig::default().loop_number_max_limit, 1000);
    }

    #[test]
    fn identity_matches_python() {
        let component = LoopGroupComponent::new(LoopGroupConfig::default(), Vec::new());
        assert_eq!(component.component_type(), "loop_group");
        assert_eq!(component.abilities(), vec![ComponentAbility::Invoke]);
        assert_eq!(component.body_len(), 0);
        assert!(format!("{component:?}").contains("body_len"));
        let card = component.card();
        assert_eq!(card.id, "loop_group");
        assert_eq!(card.abilities, vec![ComponentAbility::Invoke]);
    }
}
