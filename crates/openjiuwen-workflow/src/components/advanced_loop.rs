//! The `advanced_loop` workflow component: per-iteration termination re-evaluation.
//!
//! Ports Python's `AdvancedLoopComponent`
//! (`openjiuwen/core/workflow/components/flow/loop/loop_comp.py`), the scheduler Python's
//! `LoopComponent` builds internally: a `condition → body → post_body → condition` cycle
//! that decides before **every** round whether to continue, so a loop variable updated by
//! the body can end the loop and a [`LoopBreakComponent`](crate::components::LoopBreakComponent)
//! ends it immediately.
//!
//! Where [`LoopGroupComponent`](crate::components::loop_group::LoopGroupComponent) runs one
//! bound kind, this component combines all of them: the iteration cap
//! (`max_iterations`), an optional count (`loop_number`), an optional array
//! (`loop_array`), an optional per-iteration condition (`bool_expression` /
//! `loop_condition`), and the body's break signal. The loop stops as soon as **any** of
//! them ends it, and per-iteration loop-variable updates are applied before the next round
//! (and before the next condition evaluation).
//!
//! ```
//! use openjiuwen_workflow::components::advanced_loop::{
//!     AdvancedLoopComponent, AdvancedLoopConfig,
//! };
//!
//! // A condition-gated loop that is also capped at five rounds.
//! let config = AdvancedLoopConfig::new()
//!     .with_bool_expression("${total} < 3")
//!     .with_max_iterations(5);
//! let component = AdvancedLoopComponent::new(config, Vec::new());
//! assert_eq!(component.config().max_iterations, 5);
//! ```

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use openjiuwen_common::error::Result;

use crate::ability::ComponentAbility;
use crate::component::{Component, ComponentCard};
use crate::condition::Condition;
use crate::context::ComponentContext;
use crate::error::loop_execution_error;

use crate::components::loop_control::BREAK_COMPONENT_TYPE;
use crate::components::loop_driver::{self, BoundParams, LoopBound, DEFAULT_MAX_ITERATIONS};

/// Static configuration of an [`AdvancedLoopComponent`].
///
/// Field names are `snake_case`, unknown fields are tolerated, and the type round-trips
/// through JSON so a workflow spec can be serialized and rebuilt. `max_iterations` also
/// answers to `loop_number_max_limit`, and `bool_expression` to `loop_condition`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct AdvancedLoopConfig {
    /// The hard iteration cap; defaults to Python's `LOOP_NUMBER_MAX_LIMIT` (`1000`).
    #[serde(alias = "loop_number_max_limit")]
    pub max_iterations: u64,
    /// Optional fixed iteration count.
    pub loop_number: Option<u64>,
    /// Optional input key (or whole-string `${...}` reference) naming the array to iterate.
    pub loop_array: Option<String>,
    /// Optional condition expression, re-evaluated before every iteration.
    #[serde(alias = "loop_condition")]
    pub bool_expression: Option<String>,
    /// Initial loop variables, exposed to the body and updatable from it.
    pub loop_variables: BTreeMap<String, Value>,
}

impl Default for AdvancedLoopConfig {
    fn default() -> Self {
        Self {
            max_iterations: DEFAULT_MAX_ITERATIONS,
            loop_number: None,
            loop_array: None,
            bool_expression: None,
            loop_variables: BTreeMap::new(),
        }
    }
}

impl AdvancedLoopConfig {
    /// The default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the hard iteration cap.
    #[must_use]
    pub fn with_max_iterations(mut self, max_iterations: u64) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Sets `loop_number`.
    #[must_use]
    pub fn with_loop_number(mut self, loop_number: u64) -> Self {
        self.loop_number = Some(loop_number);
        self
    }

    /// Sets the array key or `${...}` reference to iterate over.
    #[must_use]
    pub fn with_loop_array(mut self, loop_array: impl Into<String>) -> Self {
        self.loop_array = Some(loop_array.into());
        self
    }

    /// Sets the per-iteration condition expression.
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

    /// How this configuration bounds the loop, in precedence order.
    ///
    /// `loop_number` wins over `loop_array`, which wins over a condition-only loop, which
    /// wins over "until a body component breaks". A `bool_expression` is applied as an
    /// additional per-iteration gate regardless of the count side.
    #[must_use]
    pub fn bound(&self) -> LoopBound {
        if self.loop_number.is_some() {
            LoopBound::Number
        } else if self.loop_array.is_some() {
            LoopBound::Array
        } else if self.bool_expression.is_some() {
            LoopBound::Condition
        } else {
            LoopBound::UntilBreak
        }
    }
}

/// The `advanced_loop` component: a loop that re-evaluates its termination every round.
///
/// The body runs sequentially per iteration, each later component receiving the previous
/// one's output. Every iteration's inputs are the component's resolved inputs overlaid
/// with `{"index", "item", "previous"}` plus the loop variables (flattened and under
/// `"loop_variables"`), and the same inputs are what the condition is evaluated against,
/// so a body-side variable update changes the next termination decision. The output is
/// the documented container wrapper
/// `{"loop_results", "loop_variables", "loop_count", "broken"}`.
pub struct AdvancedLoopComponent {
    config: AdvancedLoopConfig,
    body: Vec<Arc<dyn Component>>,
}

impl fmt::Debug for AdvancedLoopComponent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AdvancedLoopComponent")
            .field("config", &self.config)
            .field("body_len", &self.body.len())
            .finish()
    }
}

impl AdvancedLoopComponent {
    /// Creates a loop with `config` and `body`.
    #[must_use]
    pub fn new(config: AdvancedLoopConfig, body: Vec<Arc<dyn Component>>) -> Self {
        Self { config, body }
    }

    /// The component configuration.
    #[must_use]
    pub fn config(&self) -> &AdvancedLoopConfig {
        &self.config
    }

    /// The number of body components run per iteration.
    #[must_use]
    pub fn body_len(&self) -> usize {
        self.body.len()
    }

    /// Whether the body holds a [`LoopBreakComponent`](crate::components::LoopBreakComponent).
    ///
    /// The scan looks at the immediate body only, like
    /// [`LoopGroupComponent::has_break_component`](crate::components::loop_group::LoopGroupComponent::has_break_component).
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

    /// Whether a per-iteration condition is configured besides the count side.
    fn is_unterminated(&self) -> bool {
        self.config.loop_number.is_none()
            && self.config.loop_array.is_none()
            && self.config.bool_expression.is_none()
    }
}

#[async_trait::async_trait]
impl Component for AdvancedLoopComponent {
    fn card(&self) -> ComponentCard {
        ComponentCard::new("advanced_loop", "advanced_loop").with_abilities(self.abilities())
    }

    fn abilities(&self) -> Vec<ComponentAbility> {
        vec![ComponentAbility::Invoke]
    }

    fn component_type(&self) -> &'static str {
        "advanced_loop"
    }

    /// Rejects a loop that cannot run or cannot terminate, before any run.
    fn validate_config(&self) -> Result<()> {
        loop_driver::validate_bound(
            self.config.bound(),
            self.bound_params(),
            self.config.max_iterations,
        )?;
        if self.is_unterminated() && !self.has_break_component() {
            return Err(loop_driver::no_terminator());
        }
        Ok(())
    }

    async fn invoke(&self, inputs: Value, ctx: &ComponentContext) -> Result<Value> {
        let base = loop_driver::object_inputs(inputs, ctx)?;
        let limit = loop_driver::effective_limit(self.config.max_iterations, ctx);
        let bound = self.config.bound();
        let mut spec =
            loop_driver::plan(bound, self.bound_params(), &base, limit, ctx.component_id())?;
        if bound != LoopBound::Condition {
            if let Some(expression) = self.config.bool_expression.as_deref() {
                spec.condition =
                    Some(Condition::new(expression).map_err(|error| {
                        loop_execution_error(&error.message(), ctx.component_id())
                    })?);
            }
        }
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

    /// Increments the running `total`, so a variable-carrying loop counts rounds.
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
        let session: Arc<dyn Session> = Arc::new(WorkflowSession::new("advanced-loop-tests"));
        ComponentContext::new(session, "advanced_loop", "wf")
    }

    fn body() -> Vec<Arc<dyn Component>> {
        vec![
            Arc::new(IncrementComponent),
            Arc::new(LoopSetVariableComponent::new(
                [("${loop.total}".to_owned(), json!("${total}"))].into(),
            )),
        ]
    }

    fn results(output: &Value) -> &Vec<Value> {
        output["loop_results"].as_array().unwrap()
    }

    #[tokio::test]
    async fn the_condition_ends_the_loop_and_reads_the_updated_variables() {
        let component = AdvancedLoopComponent::new(
            AdvancedLoopConfig::new()
                .with_bool_expression("${total} < 3")
                .with_variable("total", json!(0)),
            body(),
        );
        let output = component.invoke(json!({}), &context()).await.unwrap();
        // `total` starts at 0, so rounds 1..=3 run and `total` reaches 3.
        assert_eq!(output["loop_count"], json!(3));
        assert_eq!(output["loop_variables"], json!({"total": 3}));
        assert_eq!(results(&output).len(), 3);
    }

    #[tokio::test]
    async fn the_iteration_cap_ends_a_condition_that_never_goes_false() {
        let component = AdvancedLoopComponent::new(
            AdvancedLoopConfig::new()
                .with_bool_expression("true")
                .with_max_iterations(2),
            body(),
        );
        let error = component.invoke(json!({}), &context()).await.unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Workflow);
        assert!(error.message().contains("[101040]"), "{error}");
        assert!(error.message().contains("maximum limit 2"), "{error}");
    }

    #[tokio::test]
    async fn a_count_bound_is_gated_by_the_condition() {
        let component = AdvancedLoopComponent::new(
            AdvancedLoopConfig::new()
                .with_loop_number(5)
                .with_bool_expression("${index} < 2"),
            body(),
        );
        let output = component.invoke(json!({}), &context()).await.unwrap();
        assert_eq!(output["loop_count"], json!(2));
        assert_eq!(output["loop_variables"], json!({"total": 2}));
    }

    #[tokio::test]
    async fn an_array_bound_iterates_the_referenced_items() {
        let component =
            AdvancedLoopComponent::new(AdvancedLoopConfig::new().with_loop_array("items"), body());
        let output = component
            .invoke(json!({"items": ["a", "b", "c"]}), &context())
            .await
            .unwrap();
        assert_eq!(output["loop_count"], json!(3));
    }

    #[tokio::test]
    async fn a_break_component_ends_the_loop_without_recording_the_round() {
        let mut body = body();
        body.insert(
            0,
            Arc::new(LoopBreakComponent::new().with_condition("${index} == 2")),
        );
        let component =
            AdvancedLoopComponent::new(AdvancedLoopConfig::new().with_max_iterations(10), body);
        let output = component.invoke(json!({}), &context()).await.unwrap();
        assert_eq!(output["broken"], json!(true));
        assert_eq!(output["loop_count"], json!(2));
        assert_eq!(
            results(&output),
            &vec![json!({"total": 1}), json!({"total": 2})]
        );
    }

    #[test]
    fn validate_config_rejects_unrunnable_loops() {
        let unterminated = AdvancedLoopComponent::new(AdvancedLoopConfig::new(), body());
        let error = unterminated.validate_config().unwrap_err();
        assert!(error.message().contains("[101030]"), "{error}");
        assert!(error.message().contains("cannot terminate"), "{error}");

        let over_cap = AdvancedLoopComponent::new(
            AdvancedLoopConfig::new()
                .with_loop_number(11)
                .with_max_iterations(10),
            body(),
        );
        let error = over_cap.validate_config().unwrap_err();
        assert!(error
            .message()
            .contains("loop_number exceeds maximum limit 10"));

        let bad_condition = AdvancedLoopComponent::new(
            AdvancedLoopConfig::new().with_bool_expression("${total} >"),
            body(),
        );
        let error = bad_condition.validate_config().unwrap_err();
        assert!(error.message().contains("bool_expression is invalid"));

        AdvancedLoopComponent::new(
            AdvancedLoopConfig::new().with_bool_expression("${total} < 3"),
            body(),
        )
        .validate_config()
        .unwrap();

        let mut with_break = body();
        with_break.push(Arc::new(LoopBreakComponent::new()));
        AdvancedLoopComponent::new(AdvancedLoopConfig::new(), with_break)
            .validate_config()
            .unwrap();
    }

    #[test]
    fn bound_precedence_and_config_parsing() {
        assert_eq!(AdvancedLoopConfig::new().bound(), LoopBound::UntilBreak);
        assert_eq!(
            AdvancedLoopConfig::new().with_loop_array("items").bound(),
            LoopBound::Array
        );
        assert_eq!(
            AdvancedLoopConfig::new()
                .with_loop_array("items")
                .with_loop_number(2)
                .bound(),
            LoopBound::Number
        );
        assert_eq!(
            AdvancedLoopConfig::new()
                .with_bool_expression("true")
                .bound(),
            LoopBound::Condition
        );

        let config: AdvancedLoopConfig = serde_json::from_value(json!({
            "loop_number_max_limit": 4,
            "loop_condition": "${index} < 2",
            "loop_variables": {"total": 0},
            "unknown": "tolerated",
        }))
        .unwrap();
        assert_eq!(config.max_iterations, 4);
        assert_eq!(config.bool_expression.as_deref(), Some("${index} < 2"));
        assert_eq!(config.loop_variables.get("total"), Some(&json!(0)));

        let wire = serde_json::to_value(&config).unwrap();
        assert_eq!(wire["max_iterations"], json!(4));
        assert_eq!(
            serde_json::from_value::<AdvancedLoopConfig>(wire).unwrap(),
            config
        );
        assert_eq!(
            serde_json::from_value::<AdvancedLoopConfig>(json!({})).unwrap(),
            AdvancedLoopConfig::default()
        );
        assert_eq!(AdvancedLoopConfig::default().max_iterations, 1000);
    }

    #[test]
    fn identity_matches_python() {
        let component = AdvancedLoopComponent::new(AdvancedLoopConfig::default(), Vec::new());
        assert_eq!(component.component_type(), "advanced_loop");
        assert_eq!(component.abilities(), vec![ComponentAbility::Invoke]);
        assert_eq!(component.body_len(), 0);
        assert!(!component.has_break_component());
        assert!(format!("{component:?}").contains("body_len"));
        let card = component.card();
        assert_eq!(card.id, "advanced_loop");
        assert_eq!(card.abilities, vec![ComponentAbility::Invoke]);
    }
}
