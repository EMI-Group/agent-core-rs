//! Built-in `loop` workflow component.
//!
//! The module is named `loop_` because `loop` is a Rust keyword. It ports Python's
//! `openjiuwen.core.workflow.components.flow.loop.loop_comp.LoopComponent` onto a
//! normalized, in-process shape: instead of Python's nested `LoopGroup` graph the body
//! is a flat list of components run sequentially per iteration, and the collected
//! outputs are wrapped as `{"loop_results": [...]}`.
//!
//! Three iteration plans match Python's loop types:
//!
//! - [`LoopType::Number`]: exactly `loop_number` iterations.
//! - [`LoopType::Array`]: one iteration per element of the array named by `loop_array`
//!   (an input key or a `${...}` reference).
//! - [`LoopType::Condition`]: iterate while `bool_expression` evaluates true, re-checked
//!   before every iteration against that iteration's inputs.
//!
//! Every iteration starts from the component's input object and overlays
//! `{"index": <u64>, "item": <array element or null>, "previous": <last output or null>}`,
//! so both the body and a condition expression can read the loop position. The hard cap
//! is `min(config.max_iterations, session.loop_number_max_limit)`; exceeding it is a
//! `COMPONENT_LOOP_EXECUTION_ERROR` (`101040`).

use std::fmt;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

use openjiuwen_common::error::{Error, Result};

use crate::ability::ComponentAbility;
use crate::component::{Component, ComponentCard};
use crate::condition::Condition;
use crate::context::ComponentContext;
use crate::error::loop_execution_error;
use crate::reference;

/// Python's `LOOP_NUMBER_MAX_LIMIT_DEFAULT`.
const DEFAULT_MAX_ITERATIONS: u64 = 1000;

/// How a [`LoopComponent`] decides how many iterations to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LoopType {
    /// A fixed number of iterations (`loop_number`).
    #[default]
    Number,
    /// One iteration per element of `loop_array`.
    Array,
    /// Iterate while `bool_expression` evaluates true.
    Condition,
}

impl LoopType {
    /// The wire name of this loop type (`"number"`, `"array"`, `"condition"`).
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Number => "number",
            Self::Array => "array",
            Self::Condition => "condition",
        }
    }
}

impl fmt::Display for LoopType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Static configuration of a [`LoopComponent`].
///
/// Field names are `snake_case`, unknown fields are tolerated, and the type round-trips
/// through JSON so a workflow spec can be serialized and rebuilt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct LoopConfig {
    /// How iterations are counted; defaults to [`LoopType::Number`].
    pub loop_type: LoopType,
    /// Iteration count for [`LoopType::Number`].
    pub loop_number: Option<u64>,
    /// Input key (or whole-string `${...}` reference) naming the array for
    /// [`LoopType::Array`].
    pub loop_array: Option<String>,
    /// Condition expression for [`LoopType::Condition`].
    pub bool_expression: Option<String>,
    /// Hard cap on iterations; defaults to Python's `LOOP_NUMBER_MAX_LIMIT` (`1000`).
    pub max_iterations: u64,
}

impl Default for LoopConfig {
    fn default() -> Self {
        Self {
            loop_type: LoopType::Number,
            loop_number: None,
            loop_array: None,
            bool_expression: None,
            max_iterations: DEFAULT_MAX_ITERATIONS,
        }
    }
}

impl LoopConfig {
    /// A configuration for `loop_type` with every type-specific parameter unset.
    #[must_use]
    pub fn new(loop_type: LoopType) -> Self {
        Self {
            loop_type,
            ..Self::default()
        }
    }

    /// Sets `loop_number`.
    #[must_use]
    pub fn with_loop_number(mut self, loop_number: u64) -> Self {
        self.loop_number = Some(loop_number);
        self
    }

    /// Sets the array key or `${...}` reference used by [`LoopType::Array`].
    #[must_use]
    pub fn with_loop_array(mut self, loop_array: impl Into<String>) -> Self {
        self.loop_array = Some(loop_array.into());
        self
    }

    /// Sets the condition expression used by [`LoopType::Condition`].
    #[must_use]
    pub fn with_bool_expression(mut self, expression: impl Into<String>) -> Self {
        self.bool_expression = Some(expression.into());
        self
    }

    /// Sets the hard iteration cap.
    #[must_use]
    pub fn with_max_iterations(mut self, max_iterations: u64) -> Self {
        self.max_iterations = max_iterations;
        self
    }
}

/// The `loop` built-in component.
///
/// The body is a flat list of components run sequentially per iteration: the first
/// receives the iteration inputs, and every later component receives the previous one's
/// output. An empty body echoes the iteration inputs as the iteration output.
pub struct LoopComponent {
    config: LoopConfig,
    body: Vec<Arc<dyn Component>>,
}

impl fmt::Debug for LoopComponent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LoopComponent")
            .field("config", &self.config)
            .field("body_len", &self.body.len())
            .finish()
    }
}

impl LoopComponent {
    /// Creates a loop component with `config` and `body`.
    #[must_use]
    pub fn new(config: LoopConfig, body: Vec<Arc<dyn Component>>) -> Self {
        Self { config, body }
    }

    /// The component configuration.
    #[must_use]
    pub fn config(&self) -> &LoopConfig {
        &self.config
    }

    /// The number of body components run per iteration.
    #[must_use]
    pub fn body_len(&self) -> usize {
        self.body.len()
    }

    /// The effective iteration cap: the tighter of the component and session limits.
    fn effective_limit(&self, ctx: &ComponentContext) -> u64 {
        let Some(config) = ctx.session().config() else {
            return self.config.max_iterations;
        };
        let session_limit = config.read().loop_number_max_limit;
        self.config.max_iterations.min(session_limit)
    }

    /// The error returned when a loop would run more than `limit` iterations.
    fn exceeds_limit(limit: u64, ctx: &ComponentContext) -> Error {
        loop_execution_error(
            &format!("loop_number exceeds maximum limit {limit}"),
            ctx.component_id(),
        )
    }

    /// Builds the iteration plan, validating the type-specific configuration.
    fn iteration_plan(
        &self,
        base: &Value,
        limit: u64,
        ctx: &ComponentContext,
    ) -> Result<IterationPlan> {
        match self.config.loop_type {
            LoopType::Number => {
                let count = self.config.loop_number.ok_or_else(|| {
                    loop_execution_error(
                        "loop_number variable not found or is None",
                        ctx.component_id(),
                    )
                })?;
                if count > limit {
                    return Err(Self::exceeds_limit(limit, ctx));
                }
                Ok(IterationPlan::Count(count))
            }
            LoopType::Array => {
                let spec = self.config.loop_array.as_deref().ok_or_else(|| {
                    loop_execution_error(
                        "loop_array variable not found or is None",
                        ctx.component_id(),
                    )
                })?;
                let Value::Array(items) = resolve_array(base, spec) else {
                    return Err(loop_execution_error(
                        "loop_array is not an array",
                        ctx.component_id(),
                    ));
                };
                let length = u64::try_from(items.len()).unwrap_or(u64::MAX);
                if length > limit {
                    return Err(Self::exceeds_limit(limit, ctx));
                }
                Ok(IterationPlan::Items(items))
            }
            LoopType::Condition => {
                let expression = self.config.bool_expression.as_deref().ok_or_else(|| {
                    loop_execution_error(
                        "bool_expression variable not found or is None",
                        ctx.component_id(),
                    )
                })?;
                let condition = Condition::new(expression)
                    .map_err(|error| loop_execution_error(&error.message(), ctx.component_id()))?;
                Ok(IterationPlan::Condition(condition))
            }
        }
    }

    /// Runs the plan and collects one output per iteration.
    async fn run_plan(
        &self,
        plan: IterationPlan,
        base: &Value,
        limit: u64,
        ctx: &ComponentContext,
    ) -> Result<Vec<Value>> {
        match plan {
            IterationPlan::Count(count) => self.run_count(count, base, ctx).await,
            IterationPlan::Items(items) => self.run_items(&items, base, ctx).await,
            IterationPlan::Condition(condition) => {
                self.run_condition(&condition, base, limit, ctx).await
            }
        }
    }

    /// Runs `count` iterations with a null item.
    async fn run_count(
        &self,
        count: u64,
        base: &Value,
        ctx: &ComponentContext,
    ) -> Result<Vec<Value>> {
        let mut results = Vec::new();
        let mut previous = Value::Null;
        for index in 0..count {
            ctx.check_cancelled()?;
            let inputs = iteration_inputs(base, index, Value::Null, &previous);
            previous = self.run_body(inputs, ctx).await?;
            results.push(previous.clone());
        }
        Ok(results)
    }

    /// Runs one iteration per array element.
    async fn run_items(
        &self,
        items: &[Value],
        base: &Value,
        ctx: &ComponentContext,
    ) -> Result<Vec<Value>> {
        let mut results = Vec::new();
        let mut previous = Value::Null;
        for (index, item) in (0u64..).zip(items) {
            ctx.check_cancelled()?;
            let inputs = iteration_inputs(base, index, item.clone(), &previous);
            previous = self.run_body(inputs, ctx).await?;
            results.push(previous.clone());
        }
        Ok(results)
    }

    /// Runs while the condition holds, failing when it would exceed `limit`.
    async fn run_condition(
        &self,
        condition: &Condition,
        base: &Value,
        limit: u64,
        ctx: &ComponentContext,
    ) -> Result<Vec<Value>> {
        let mut results = Vec::new();
        let mut previous = Value::Null;
        let mut index = 0u64;
        loop {
            ctx.check_cancelled()?;
            let inputs = iteration_inputs(base, index, Value::Null, &previous);
            if !Self::evaluate(condition, &inputs, ctx)? {
                break;
            }
            if index >= limit {
                return Err(Self::exceeds_limit(limit, ctx));
            }
            previous = self.run_body(inputs, ctx).await?;
            results.push(previous.clone());
            index += 1;
        }
        Ok(results)
    }

    /// Evaluates `condition`, mapping condition failures onto the loop error.
    fn evaluate(condition: &Condition, source: &Value, ctx: &ComponentContext) -> Result<bool> {
        condition
            .evaluate(source)
            .map_err(|error| loop_execution_error(&error.message(), ctx.component_id()))
    }

    /// Runs the body sequentially and returns the last component's output.
    async fn run_body(&self, inputs: Value, ctx: &ComponentContext) -> Result<Value> {
        let mut current = inputs;
        for component in &self.body {
            current = component.invoke(current, ctx).await?;
        }
        Ok(current)
    }
}

/// The three iteration plans of a [`LoopComponent`].
enum IterationPlan {
    /// A fixed iteration count.
    Count(u64),
    /// One iteration per element.
    Items(Vec<Value>),
    /// Iterate while the condition holds.
    Condition(Condition),
}

/// Resolves the `loop_array` specification to a JSON value.
///
/// A whole-string reference (`"${items}"`) is resolved against the input object; any
/// other string is looked up as a top-level input key. A missing key resolves to `null`
/// and fails the subsequent array check.
fn resolve_array(base: &Value, spec: &str) -> Value {
    if reference::is_ref_path(spec) {
        return reference::extract_origin_key(spec)
            .and_then(|path| reference::resolve_path(base, path))
            .unwrap_or(Value::Null);
    }
    base.get(spec).cloned().unwrap_or(Value::Null)
}

/// Builds one iteration's inputs: the base object overlaid with the loop variables.
fn iteration_inputs(base: &Value, index: u64, item: Value, previous: &Value) -> Value {
    let mut merged = match base {
        Value::Object(map) => map.clone(),
        _ => Map::new(),
    };
    merged.insert("index".to_owned(), Value::from(index));
    merged.insert("item".to_owned(), item);
    merged.insert("previous".to_owned(), previous.clone());
    Value::Object(merged)
}

#[async_trait::async_trait]
impl Component for LoopComponent {
    fn card(&self) -> ComponentCard {
        ComponentCard::new("loop", "loop").with_abilities(self.abilities())
    }

    fn abilities(&self) -> Vec<ComponentAbility> {
        vec![ComponentAbility::Invoke]
    }

    fn component_type(&self) -> &'static str {
        "loop"
    }

    async fn invoke(&self, inputs: Value, ctx: &ComponentContext) -> Result<Value> {
        let base = match inputs {
            Value::Object(map) => Value::Object(map),
            Value::Null => Value::Object(Map::new()),
            _ => {
                return Err(loop_execution_error(
                    "inputs must be an object",
                    ctx.component_id(),
                ))
            }
        };
        let limit = self.effective_limit(ctx);
        let plan = self.iteration_plan(&base, limit, ctx)?;
        let results = self.run_plan(plan, &base, limit, ctx).await?;
        Ok(json!({"loop_results": results}))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]
mod tests {
    use std::sync::Arc;

    use openjiuwen_common::error::ErrorKind;
    use openjiuwen_session::{Session, WorkflowSession};
    use parking_lot::Mutex;
    use serde_json::{json, Value};

    use super::*;

    /// Records every input it receives and echoes it back.
    struct RecordingComponent {
        seen: Arc<Mutex<Vec<Value>>>,
    }

    impl RecordingComponent {
        fn new() -> (Self, Arc<Mutex<Vec<Value>>>) {
            let seen = Arc::new(Mutex::new(Vec::new()));
            (
                Self {
                    seen: Arc::clone(&seen),
                },
                seen,
            )
        }
    }

    #[async_trait::async_trait]
    impl Component for RecordingComponent {
        fn card(&self) -> ComponentCard {
            ComponentCard::new("recording", "recording")
        }

        fn component_type(&self) -> &'static str {
            "recording"
        }

        async fn invoke(&self, inputs: Value, _ctx: &ComponentContext) -> Result<Value> {
            self.seen.lock().push(inputs.clone());
            Ok(json!({"echo": inputs}))
        }
    }

    fn session() -> Arc<dyn Session> {
        Arc::new(WorkflowSession::new("loop-tests"))
    }

    fn context() -> ComponentContext {
        ComponentContext::new(session(), "loop", "wf")
    }

    fn context_with_limit(limit: u64) -> ComponentContext {
        let session = session();
        if let Some(config) = session.config() {
            config.write().loop_number_max_limit = limit;
        }
        ComponentContext::new(session, "loop", "wf")
    }

    type SeenInputs = Arc<Mutex<Vec<Value>>>;

    fn recording_body() -> (Vec<Arc<dyn Component>>, SeenInputs) {
        let (component, seen) = RecordingComponent::new();
        (vec![Arc::new(component) as Arc<dyn Component>], seen)
    }

    fn results(output: &Value) -> &Vec<Value> {
        output["loop_results"].as_array().unwrap()
    }

    #[tokio::test]
    async fn number_loop_runs_the_configured_number_of_iterations() {
        let (body, seen) = recording_body();
        let component =
            LoopComponent::new(LoopConfig::new(LoopType::Number).with_loop_number(3), body);
        let output = component
            .invoke(json!({"seed": 1}), &context())
            .await
            .unwrap();

        assert_eq!(results(&output).len(), 3);
        let seen = seen.lock();
        assert_eq!(seen.len(), 3);
        for (index, input) in seen.iter().enumerate() {
            assert_eq!(input["index"], json!(index));
            assert_eq!(input["seed"], json!(1));
            assert_eq!(input["item"], Value::Null);
        }
        assert_eq!(seen[0]["previous"], Value::Null);
        assert_eq!(seen[1]["previous"], json!({"echo": seen[0].clone()}));
    }

    #[tokio::test]
    async fn array_loop_supplies_each_item() {
        let (body, seen) = recording_body();
        let component = LoopComponent::new(
            LoopConfig::new(LoopType::Array).with_loop_array("items"),
            body,
        );
        let output = component
            .invoke(json!({"items": ["a", "b"]}), &context())
            .await
            .unwrap();

        assert_eq!(results(&output).len(), 2);
        let seen = seen.lock();
        assert_eq!(seen[0]["item"], json!("a"));
        assert_eq!(seen[1]["item"], json!("b"));
        assert_eq!(seen[1]["index"], json!(1));
    }

    #[tokio::test]
    async fn array_loop_accepts_a_reference_path() {
        let (body, seen) = recording_body();
        let component = LoopComponent::new(
            LoopConfig::new(LoopType::Array).with_loop_array("${payload.items}"),
            body,
        );
        component
            .invoke(json!({"payload": {"items": [10, 20]}}), &context())
            .await
            .unwrap();

        let seen = seen.lock();
        assert_eq!(seen.len(), 2);
        assert_eq!(seen[0]["item"], json!(10));
        assert_eq!(seen[1]["item"], json!(20));
    }

    #[tokio::test]
    async fn array_loop_rejects_a_non_array_value() {
        let component = LoopComponent::new(
            LoopConfig::new(LoopType::Array).with_loop_array("items"),
            Vec::new(),
        );
        let error = component
            .invoke(json!({"items": 3}), &context())
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Workflow);
        assert!(error.message().contains("[101040]"));
        assert!(error.message().contains("loop_array is not an array"));
    }

    #[tokio::test]
    async fn condition_loop_stops_when_the_expression_is_false() {
        let (body, seen) = recording_body();
        let component = LoopComponent::new(
            LoopConfig::new(LoopType::Condition).with_bool_expression("${index} < 3"),
            body,
        );
        let output = component.invoke(json!({}), &context()).await.unwrap();

        assert_eq!(results(&output).len(), 3);
        assert_eq!(seen.lock().len(), 3);
    }

    #[tokio::test]
    async fn condition_loop_fails_when_it_would_exceed_the_limit() {
        let component = LoopComponent::new(
            LoopConfig::new(LoopType::Condition)
                .with_bool_expression("true")
                .with_max_iterations(2),
            Vec::new(),
        );
        let error = component.invoke(json!({}), &context()).await.unwrap_err();
        assert!(error.message().contains("exceeds maximum limit 2"));
    }

    #[tokio::test]
    async fn previous_links_iterations_and_an_empty_body_echoes_the_inputs() {
        let component = LoopComponent::new(
            LoopConfig::new(LoopType::Number).with_loop_number(2),
            Vec::new(),
        );
        let output = component
            .invoke(json!({"seed": "s"}), &context())
            .await
            .unwrap();
        let results = results(&output);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0]["index"], json!(0));
        assert_eq!(results[0]["previous"], Value::Null);
        assert_eq!(results[1]["index"], json!(1));
        assert_eq!(results[1]["previous"], results[0]);
        assert_eq!(results[1]["seed"], json!("s"));
    }

    #[tokio::test]
    async fn body_components_chain_outputs() {
        let (first, seen) = RecordingComponent::new();
        let second = RecordingComponent {
            seen: Arc::clone(&seen),
        };
        let component = LoopComponent::new(
            LoopConfig::new(LoopType::Number).with_loop_number(1),
            vec![Arc::new(first), Arc::new(second)],
        );
        let output = component
            .invoke(json!({"seed": 1}), &context())
            .await
            .unwrap();

        let seen = seen.lock();
        assert_eq!(seen.len(), 2);
        assert_eq!(seen[0]["index"], json!(0));
        assert_eq!(seen[1], json!({"echo": seen[0].clone()}));
        assert_eq!(results(&output), &vec![json!({"echo": seen[1].clone()})]);
    }

    #[tokio::test]
    async fn number_loop_rejects_counts_above_the_limit() {
        let component = LoopComponent::new(
            LoopConfig::new(LoopType::Number).with_loop_number(1001),
            Vec::new(),
        );
        let error = component.invoke(json!({}), &context()).await.unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Workflow);
        assert!(error.message().contains("[101040]"));
        assert!(error.message().contains("exceeds maximum limit"));
    }

    #[tokio::test]
    async fn the_session_limit_tightens_the_component_limit() {
        let component = LoopComponent::new(
            LoopConfig::new(LoopType::Number).with_loop_number(3),
            Vec::new(),
        );
        let error = component
            .invoke(json!({}), &context_with_limit(2))
            .await
            .unwrap_err();
        assert!(error.message().contains("exceeds maximum limit 2"));
    }

    #[tokio::test]
    async fn a_zero_limit_runs_zero_iterations() {
        let component = LoopComponent::new(
            LoopConfig::new(LoopType::Number)
                .with_loop_number(0)
                .with_max_iterations(0),
            Vec::new(),
        );
        let output = component.invoke(json!({}), &context()).await.unwrap();
        assert!(results(&output).is_empty());

        let component = LoopComponent::new(
            LoopConfig::new(LoopType::Array)
                .with_loop_array("items")
                .with_max_iterations(0),
            Vec::new(),
        );
        let output = component
            .invoke(json!({"items": []}), &context())
            .await
            .unwrap();
        assert!(results(&output).is_empty());
    }

    #[tokio::test]
    async fn missing_configuration_is_reported() {
        let error = LoopComponent::new(LoopConfig::new(LoopType::Number), Vec::new())
            .invoke(json!({}), &context())
            .await
            .unwrap_err();
        assert!(error
            .message()
            .contains("loop_number variable not found or is None"));

        let error = LoopComponent::new(LoopConfig::new(LoopType::Array), Vec::new())
            .invoke(json!({}), &context())
            .await
            .unwrap_err();
        assert!(error
            .message()
            .contains("loop_array variable not found or is None"));

        let error = LoopComponent::new(LoopConfig::new(LoopType::Condition), Vec::new())
            .invoke(json!({}), &context())
            .await
            .unwrap_err();
        assert!(error
            .message()
            .contains("bool_expression variable not found or is None"));
    }

    #[tokio::test]
    async fn non_object_inputs_are_rejected() {
        let error = LoopComponent::new(LoopConfig::new(LoopType::Number), Vec::new())
            .invoke(json!(42), &context())
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Workflow);
        assert!(error.message().contains("inputs must be an object"));
    }

    #[tokio::test]
    async fn null_inputs_are_treated_as_an_empty_object() {
        let output = LoopComponent::new(
            LoopConfig::new(LoopType::Number).with_loop_number(1),
            Vec::new(),
        )
        .invoke(Value::Null, &context())
        .await
        .unwrap();
        let results = results(&output);
        assert_eq!(results[0]["index"], json!(0));
    }

    #[test]
    fn loop_type_names_match_python() {
        assert_eq!(LoopType::default(), LoopType::Number);
        assert_eq!(LoopType::Number.as_str(), "number");
        assert_eq!(LoopType::Array.as_str(), "array");
        assert_eq!(LoopType::Condition.as_str(), "condition");
        assert_eq!(LoopType::Array.to_string(), "array");
        assert_eq!(
            serde_json::to_value(LoopType::Condition).unwrap(),
            json!("condition")
        );
    }

    #[test]
    fn config_round_trips_and_tolerates_unknown_fields() {
        let config = LoopConfig::new(LoopType::Array)
            .with_loop_array("${items}")
            .with_max_iterations(5);
        let wire = serde_json::to_value(&config).unwrap();
        assert_eq!(wire["loop_type"], json!("array"));
        assert_eq!(wire["loop_array"], json!("${items}"));
        assert_eq!(wire["max_iterations"], json!(5));

        let parsed: LoopConfig = serde_json::from_value(json!({
            "loop_type": "condition",
            "bool_expression": "${index} < 3",
            "future_field": true
        }))
        .unwrap();
        assert_eq!(parsed.loop_type, LoopType::Condition);
        assert_eq!(parsed.bool_expression.as_deref(), Some("${index} < 3"));
        assert_eq!(parsed.max_iterations, 1000);

        let defaults: LoopConfig = serde_json::from_value(json!({})).unwrap();
        assert_eq!(defaults, LoopConfig::default());
    }

    #[test]
    fn identity_matches_python() {
        let component = LoopComponent::new(LoopConfig::default(), Vec::new());
        assert_eq!(component.component_type(), "loop");
        assert_eq!(component.abilities(), vec![ComponentAbility::Invoke]);
        assert_eq!(component.body_len(), 0);
        assert_eq!(component.config(), &LoopConfig::default());
        let card = component.card();
        assert_eq!(card.id, "loop");
        assert_eq!(card.component_type, "loop");
        assert_eq!(card.abilities, vec![ComponentAbility::Invoke]);
        assert!(format!("{component:?}").contains("body_len"));
    }
}
