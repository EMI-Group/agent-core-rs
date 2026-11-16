//! Shared iteration driver for the loop components.
//!
//! [`LoopComponent`](crate::components::loop_::LoopComponent),
//! [`LoopGroupComponent`](crate::components::loop_group::LoopGroupComponent) and
//! [`AdvancedLoopComponent`](crate::components::advanced_loop::AdvancedLoopComponent) all
//! run a flat body repeatedly, so the iteration plan, the iteration cap, the iteration
//! inputs, and the loop-variable/break folding live here once. [`LoopBound`] is the one
//! public item: the container configurations expose it as their `loop_type` field.
//!
//! Every iteration starts from the component's input object overlaid with
//! `{"index": <u64>, "item": <array element or null>, "previous": <last output or null>}`
//! plus the current loop variables, so both the body and a condition expression can read
//! the loop position (`iteration_inputs`, `overlay_variables`).

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

use openjiuwen_common::error::{Error, Result};

use crate::component::Component;
use crate::condition::Condition;
use crate::context::ComponentContext;
use crate::error::{loop_execution_error, loop_param_invalid};
use crate::reference;

use crate::components::loop_control;

/// Python's `LOOP_NUMBER_MAX_LIMIT_DEFAULT`.
pub(crate) const DEFAULT_MAX_ITERATIONS: u64 = 1000;

/// How a loop container is bounded.
///
/// Python's `LoopType` (`array`, `number`, `always_true`, `expression`) split across two
/// Rust types: [`crate::components::LoopType`] keeps the three bounded kinds the original
/// `loop` component ports, and this enum adds the unbounded kind a container needs to
/// describe "iterate until a [`LoopBreakComponent`](crate::components::LoopBreakComponent)
/// asks me to stop". It is serde `snake_case`; `always_true` is accepted as an alias so a
/// Python-shaped spec parses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LoopBound {
    /// A fixed number of iterations (`loop_number`), capped by the iteration limit.
    #[default]
    Number,
    /// One iteration per element of the array named by `loop_array`.
    Array,
    /// Iterate while the condition (`bool_expression` / `loop_condition`) holds.
    Condition,
    /// Iterate until a body component breaks the loop, capped by the iteration limit.
    ///
    /// A container using it must contain a `LoopBreakComponent`; build-time validation
    /// rejects it otherwise.
    #[serde(alias = "always_true")]
    UntilBreak,
}

impl LoopBound {
    /// The wire name (`"number"`, `"array"`, `"condition"`, `"until_break"`).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Number => "number",
            Self::Array => "array",
            Self::Condition => "condition",
            Self::UntilBreak => "until_break",
        }
    }

    /// Whether the bound alone cannot terminate the loop.
    #[must_use]
    pub const fn needs_break(self) -> bool {
        matches!(self, Self::UntilBreak)
    }
}

impl fmt::Display for LoopBound {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl From<crate::components::LoopType> for LoopBound {
    fn from(loop_type: crate::components::LoopType) -> Self {
        match loop_type {
            crate::components::LoopType::Number => Self::Number,
            crate::components::LoopType::Array => Self::Array,
            crate::components::LoopType::Condition => Self::Condition,
        }
    }
}

/// The bound parameters read from a container configuration.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct BoundParams<'a> {
    /// A fixed iteration count.
    pub loop_number: Option<u64>,
    /// An input key or whole-string `${...}` reference naming the array to iterate.
    pub loop_array: Option<&'a str>,
    /// A condition expression, re-evaluated before every iteration.
    pub bool_expression: Option<&'a str>,
}

impl<'a> BoundParams<'a> {
    /// The parameters of `config`'s loop fields.
    pub(crate) fn new(
        loop_number: Option<u64>,
        loop_array: Option<&'a str>,
        bool_expression: Option<&'a str>,
    ) -> Self {
        Self {
            loop_number,
            loop_array,
            bool_expression,
        }
    }
}

/// The count side of a plan: how many iterations, or over what.
#[derive(Debug, Clone)]
pub(crate) enum IterationPlan {
    /// Exactly this many iterations.
    Count(u64),
    /// One iteration per element.
    Items(Vec<Value>),
    /// Iterate until a condition stops it, a body breaks, or the cap is hit.
    UntilBreak,
}

/// A complete iteration plan: the count side plus the per-iteration condition.
#[derive(Debug, Clone)]
pub(crate) struct LoopSpec {
    /// How the iteration count is bounded.
    pub plan: IterationPlan,
    /// Re-evaluated before every iteration; `None` means "no condition".
    pub condition: Option<Condition>,
}

/// The result of running a plan.
#[derive(Debug, Clone, Default)]
pub(crate) struct LoopRun {
    /// One output per completed iteration.
    pub results: Vec<Value>,
    /// The loop variables after the last completed iteration.
    pub variables: Map<String, Value>,
    /// Whether a body component asked the loop to stop.
    pub broken: bool,
}

impl LoopRun {
    /// The number of completed iterations.
    pub(crate) fn count(&self) -> u64 {
        u64::try_from(self.results.len()).unwrap_or(u64::MAX)
    }
}

/// The effective iteration cap: the tighter of `limit` and the session's limit.
pub(crate) fn effective_limit(limit: u64, ctx: &ComponentContext) -> u64 {
    let Some(config) = ctx.session().config() else {
        return limit;
    };
    let session_limit = config.read().loop_number_max_limit;
    limit.min(session_limit)
}

/// The error returned when a loop would run more than `limit` iterations.
pub(crate) fn exceeds_limit(limit: u64, ctx: &ComponentContext) -> Error {
    loop_execution_error(
        &format!("loop_number exceeds maximum limit {limit}"),
        ctx.component_id(),
    )
}

/// Builds the iteration plan, validating the bound against `base` at run time.
///
/// # Errors
///
/// Returns [`Error::Workflow`] with the loop execution code (`101040`) for a missing or
/// malformed bound, a non-array `loop_array`, a condition that does not parse, or a
/// bound wider than `limit`.
pub(crate) fn plan(
    bound: LoopBound,
    params: BoundParams<'_>,
    base: &Value,
    limit: u64,
    comp: &str,
) -> Result<LoopSpec> {
    let execution = |reason: &str| loop_execution_error(reason, comp);
    match bound {
        LoopBound::Number => {
            let count = params
                .loop_number
                .ok_or_else(|| execution("loop_number variable not found or is None"))?;
            if count > limit {
                return Err(execution(&format!(
                    "loop_number exceeds maximum limit {limit}"
                )));
            }
            Ok(LoopSpec {
                plan: IterationPlan::Count(count),
                condition: None,
            })
        }
        LoopBound::Array => {
            let spec = params
                .loop_array
                .ok_or_else(|| execution("loop_array variable not found or is None"))?;
            let Value::Array(items) = resolve_array(base, spec) else {
                return Err(execution("loop_array is not an array"));
            };
            let length = u64::try_from(items.len()).unwrap_or(u64::MAX);
            if length > limit {
                return Err(execution(&format!(
                    "loop_number exceeds maximum limit {limit}"
                )));
            }
            Ok(LoopSpec {
                plan: IterationPlan::Items(items),
                condition: None,
            })
        }
        LoopBound::Condition => {
            let expression = params
                .bool_expression
                .ok_or_else(|| execution("bool_expression variable not found or is None"))?;
            let condition =
                Condition::new(expression).map_err(|error| execution(&error.message()))?;
            Ok(LoopSpec {
                plan: IterationPlan::UntilBreak,
                condition: Some(condition),
            })
        }
        LoopBound::UntilBreak => Ok(LoopSpec {
            plan: IterationPlan::UntilBreak,
            condition: None,
        }),
    }
}

/// Validates the bound without any run data, for build-time checks.
///
/// # Errors
///
/// Returns [`Error::Workflow`] with the loop parameter code (`101030`) for a missing
/// bound parameter, a zero count or limit, a bound wider than `limit`, or a condition
/// that does not parse.
pub(crate) fn validate_bound(bound: LoopBound, params: BoundParams<'_>, limit: u64) -> Result<()> {
    if limit == 0 {
        return Err(loop_param_invalid(
            "the iteration limit must be greater than 0",
        ));
    }
    match bound {
        LoopBound::Number => {
            let Some(count) = params.loop_number else {
                return Err(loop_param_invalid(
                    "loop_number variable not found or is None",
                ));
            };
            if count == 0 {
                return Err(loop_param_invalid("loop_number must be greater than 0"));
            }
            if count > limit {
                return Err(loop_param_invalid(&format!(
                    "loop_number exceeds maximum limit {limit}"
                )));
            }
        }
        LoopBound::Array => match params.loop_array {
            Some(array) if !array.trim().is_empty() => {}
            _ => {
                return Err(loop_param_invalid(
                    "loop_array variable not found or is None",
                ))
            }
        },
        LoopBound::Condition => {
            let Some(expression) = params.bool_expression else {
                return Err(loop_param_invalid(
                    "bool_expression variable not found or is None",
                ));
            };
            if expression.trim().is_empty() {
                return Err(loop_param_invalid(
                    "bool_expression variable not found or is None",
                ));
            }
            Condition::new(expression).map_err(|error| {
                loop_param_invalid(&format!("bool_expression is invalid, {}", error.message()))
            })?;
        }
        LoopBound::UntilBreak => {}
    }
    Ok(())
}

/// The error every container reports when nothing can terminate its loop.
pub(crate) fn no_terminator() -> Error {
    loop_param_invalid(
        "the loop cannot terminate: it has no loop_number, no loop_array, no loop \
         condition, and no LoopBreakComponent in its body",
    )
}

/// Normalizes a container's invocation inputs to a JSON object.
///
/// A missing (`null`) input is an empty object, matching Python's lenient loop inputs; any
/// other non-object is rejected.
///
/// # Errors
///
/// Returns [`Error::Workflow`] with the loop execution code (`101040`) for a non-object.
pub(crate) fn object_inputs(inputs: Value, ctx: &ComponentContext) -> Result<Value> {
    match inputs {
        Value::Object(map) => Ok(Value::Object(map)),
        Value::Null => Ok(Map::new().into()),
        _ => Err(loop_execution_error(
            "inputs must be an object",
            ctx.component_id(),
        )),
    }
}

/// The documented output wrapper of a loop container.
///
/// `{"loop_results": [<completed round>, …], "loop_variables": {<name>: <value>}, …,
/// "loop_count": <completed rounds>, "broken": <bool>}`; an interrupted round is not
/// recorded, so `loop_count` is always `loop_results.len()`.
pub(crate) fn container_output(run: &LoopRun) -> Value {
    json!({
        "loop_results": run.results,
        "loop_variables": run.variables,
        "loop_count": run.count(),
        "broken": run.broken,
    })
}

/// Resolves a container's initial loop variables against the invocation inputs.
pub(crate) fn resolve_variables(
    variables: &BTreeMap<String, Value>,
    inputs: &Value,
) -> Map<String, Value> {
    if variables.is_empty() {
        return Map::new();
    }
    let template = Value::Object(
        variables
            .iter()
            .map(|(name, value)| (name.clone(), value.clone()))
            .collect(),
    );
    match reference::resolve_schema(&template, inputs) {
        Value::Object(resolved) => resolved,
        _ => Map::new(),
    }
}

/// Runs `spec` over `body`, folding loop variables and honouring break requests.
///
/// # Errors
///
/// Propagates body component failures unchanged and returns the loop execution error
/// (`101040`) when a `UntilBreak` plan reaches `limit` or a condition cannot be
/// evaluated.
pub(crate) async fn run_spec(
    spec: &LoopSpec,
    body: &[Arc<dyn Component>],
    base: &Value,
    variables: Map<String, Value>,
    limit: u64,
    ctx: &ComponentContext,
) -> Result<LoopRun> {
    let count = match &spec.plan {
        IterationPlan::Count(count) => Some(*count),
        _ => None,
    };
    let items = match &spec.plan {
        IterationPlan::Items(items) => Some(items.as_slice()),
        _ => None,
    };
    let mut run = LoopRun {
        variables,
        ..LoopRun::default()
    };
    let mut previous = Value::Null;
    let mut index = 0u64;
    loop {
        ctx.check_cancelled()?;
        let position = usize::try_from(index).unwrap_or(usize::MAX);
        let exhausted = match (count, items) {
            (Some(count), _) => index >= count,
            (None, Some(items)) => position >= items.len(),
            (None, None) => false,
        };
        if exhausted {
            break;
        }
        let item = match items {
            Some(items) => items[position].clone(),
            None => Value::Null,
        };
        let inputs = overlay_variables(
            iteration_inputs(base, index, item, &previous),
            &run.variables,
        );
        if let Some(condition) = &spec.condition {
            if !evaluate(condition, &inputs, ctx)? {
                break;
            }
        }
        if index >= limit {
            return Err(exceeds_limit(limit, ctx));
        }
        let iteration = run_body(body, inputs, ctx).await?;
        run.variables.extend(iteration.variables);
        if iteration.broke {
            run.broken = true;
            break;
        }
        previous = iteration.output.clone();
        run.results.push(iteration.output);
        index += 1;
    }
    Ok(run)
}

/// One iteration's body outcome.
#[derive(Debug, Clone, Default)]
pub(crate) struct BodyRun {
    /// The last body component's output, with the loop markers stripped.
    pub output: Value,
    /// Whether a body component asked the loop to stop.
    pub broke: bool,
    /// Loop-variable assignments folded from the body's markers.
    pub variables: Map<String, Value>,
}

/// Runs the body sequentially, folding markers and stopping on a break request.
///
/// The first component receives the iteration inputs and every later component receives
/// the previous one's output; when a component requests a break the iteration stops there
/// and `BodyRun::output` stays `null`, so the caller can discard the partial round. A
/// control-only component (whose whole output was the loop markers) leaves the data flow
/// untouched.
///
/// # Errors
///
/// Propagates the body component's error unchanged.
pub(crate) async fn run_body(
    body: &[Arc<dyn Component>],
    inputs: Value,
    ctx: &ComponentContext,
) -> Result<BodyRun> {
    let mut run = BodyRun::default();
    let mut current = inputs;
    for component in body {
        let mut output = component.invoke(current.clone(), ctx).await?;
        let marker_only = match &output {
            Value::Object(map) => {
                map.contains_key(loop_control::LOOP_BREAK_MARKER)
                    || map.contains_key(loop_control::LOOP_SET_VARIABLES_MARKER)
            }
            _ => false,
        };
        let markers = loop_control::take_markers(&mut output);
        if let Some(variables) = markers.variables {
            run.variables.extend(variables);
        }
        if markers.broken {
            run.broke = true;
            return Ok(run);
        }
        // A control-only component (a break check or a variable assignment) publishes no
        // value of its own, so it leaves the data flow untouched instead of replacing the
        // previous output with an empty object.
        let empty_after_markers = output.as_object().is_some_and(Map::is_empty);
        if !(marker_only && empty_after_markers) {
            current = output;
        }
    }
    run.output = current;
    Ok(run)
}

/// Evaluates `condition`, mapping condition failures onto the loop execution error.
pub(crate) fn evaluate(
    condition: &Condition,
    source: &Value,
    ctx: &ComponentContext,
) -> Result<bool> {
    condition
        .evaluate(source)
        .map_err(|error| loop_execution_error(&error.message(), ctx.component_id()))
}

/// Resolves the `loop_array` specification to a JSON value.
///
/// A whole-string reference (`"${items}"`) is resolved against the input object; any
/// other string is looked up as a top-level input key. A missing key resolves to `null`
/// and fails the subsequent array check.
pub(crate) fn resolve_array(base: &Value, spec: &str) -> Value {
    if reference::is_ref_path(spec) {
        return reference::extract_origin_key(spec)
            .and_then(|path| reference::resolve_path(base, path))
            .unwrap_or(Value::Null);
    }
    base.get(spec).cloned().unwrap_or(Value::Null)
}

/// Builds one iteration's inputs: the base object overlaid with the loop variables.
fn overlay_variables(inputs: Value, variables: &Map<String, Value>) -> Value {
    if variables.is_empty() {
        return inputs;
    }
    let Value::Object(mut map) = inputs else {
        return inputs;
    };
    let mut scope = Map::new();
    for (name, value) in variables {
        let _ = map.insert(name.clone(), value.clone());
        let _ = scope.insert(name.clone(), value.clone());
    }
    let _ = map.insert("loop_variables".to_owned(), Value::Object(scope));
    Value::Object(map)
}

/// Builds one iteration's inputs: the base object overlaid with the loop position.
pub(crate) fn iteration_inputs(base: &Value, index: u64, item: Value, previous: &Value) -> Value {
    let mut merged = match base {
        Value::Object(map) => map.clone(),
        _ => Map::new(),
    };
    let _ = merged.insert("index".to_owned(), Value::from(index));
    let _ = merged.insert("item".to_owned(), item);
    let _ = merged.insert("previous".to_owned(), previous.clone());
    Value::Object(merged)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]
mod tests {
    use openjiuwen_common::error::ErrorKind;
    use serde_json::json;

    use super::*;

    fn params<'a>(
        loop_number: Option<u64>,
        loop_array: Option<&'a str>,
        expression: Option<&'a str>,
    ) -> BoundParams<'a> {
        BoundParams::new(loop_number, loop_array, expression)
    }

    #[test]
    fn bound_names_and_aliases_match_python() {
        assert_eq!(LoopBound::default(), LoopBound::Number);
        assert_eq!(LoopBound::Number.as_str(), "number");
        assert_eq!(LoopBound::Array.as_str(), "array");
        assert_eq!(LoopBound::Condition.as_str(), "condition");
        assert_eq!(LoopBound::UntilBreak.as_str(), "until_break");
        assert_eq!(LoopBound::UntilBreak.to_string(), "until_break");
        assert!(LoopBound::UntilBreak.needs_break());
        assert!(!LoopBound::Number.needs_break());
        assert_eq!(
            serde_json::to_value(LoopBound::Array).unwrap(),
            json!("array")
        );
        assert_eq!(
            serde_json::from_value::<LoopBound>(json!("until_break")).unwrap(),
            LoopBound::UntilBreak
        );
        assert_eq!(
            serde_json::from_value::<LoopBound>(json!("always_true")).unwrap(),
            LoopBound::UntilBreak
        );
    }

    #[test]
    fn validate_bound_reports_missing_and_out_of_range_parameters() {
        let error = validate_bound(LoopBound::Number, params(None, None, None), 10).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Workflow);
        assert!(error.message().contains("[101030]"), "{error}");
        assert!(error.message().contains("loop_number variable not found"));

        let error = validate_bound(LoopBound::Number, params(Some(0), None, None), 10).unwrap_err();
        assert!(
            error
                .message()
                .contains("loop_number must be greater than 0"),
            "{error}"
        );

        let error =
            validate_bound(LoopBound::Number, params(Some(11), None, None), 10).unwrap_err();
        assert!(error
            .message()
            .contains("loop_number exceeds maximum limit 10"));

        let error = validate_bound(LoopBound::Array, params(None, None, None), 10).unwrap_err();
        assert!(error.message().contains("loop_array variable not found"));
        let error =
            validate_bound(LoopBound::Array, params(None, Some("  "), None), 10).unwrap_err();
        assert!(error.message().contains("loop_array variable not found"));

        let error = validate_bound(LoopBound::Condition, params(None, None, None), 10).unwrap_err();
        assert!(error
            .message()
            .contains("bool_expression variable not found"));

        let error = validate_bound(LoopBound::Condition, params(None, None, Some("${a} >")), 10)
            .unwrap_err();
        assert!(
            error.message().contains("bool_expression is invalid"),
            "{error}"
        );

        let error = validate_bound(LoopBound::UntilBreak, params(None, None, None), 0).unwrap_err();
        assert!(error
            .message()
            .contains("iteration limit must be greater than 0"));

        validate_bound(LoopBound::UntilBreak, params(None, None, None), 10).unwrap();
        validate_bound(LoopBound::Number, params(Some(10), None, None), 10).unwrap();
        validate_bound(LoopBound::Array, params(None, Some("items"), None), 10).unwrap();
        validate_bound(
            LoopBound::Condition,
            params(None, None, Some("${index} < 2")),
            10,
        )
        .unwrap();
    }

    #[test]
    fn plan_builds_every_bound_shape() {
        let spec = plan(
            LoopBound::Number,
            params(Some(2), None, None),
            &json!({}),
            10,
            "c",
        )
        .unwrap();
        assert!(matches!(spec.plan, IterationPlan::Count(2)));
        assert!(spec.condition.is_none());

        let spec = plan(
            LoopBound::Array,
            params(None, Some("${payload.items}"), None),
            &json!({"payload": {"items": [1, 2]}}),
            10,
            "c",
        )
        .unwrap();
        match spec.plan {
            IterationPlan::Items(items) => assert_eq!(items, vec![json!(1), json!(2)]),
            other => panic!("unexpected plan {other:?}"),
        }

        let spec = plan(
            LoopBound::Condition,
            params(None, None, Some("${index} < 3")),
            &json!({}),
            10,
            "c",
        )
        .unwrap();
        assert!(matches!(spec.plan, IterationPlan::UntilBreak));
        assert_eq!(
            spec.condition.as_ref().map(Condition::expression),
            Some("${index} < 3")
        );

        let spec = plan(
            LoopBound::UntilBreak,
            params(None, None, None),
            &json!({}),
            10,
            "c",
        )
        .unwrap();
        assert!(matches!(spec.plan, IterationPlan::UntilBreak));
        assert!(spec.condition.is_none());
    }

    #[test]
    fn plan_reports_run_time_errors_with_the_execution_code() {
        let error = plan(
            LoopBound::Number,
            params(None, None, None),
            &json!({}),
            10,
            "loop_1",
        )
        .unwrap_err();
        assert!(error.message().contains("[101040]"), "{error}");
        assert!(error.message().contains("comp='loop_1'"), "{error}");

        let error = plan(
            LoopBound::Array,
            params(None, Some("items"), None),
            &json!({"items": 3}),
            10,
            "loop_1",
        )
        .unwrap_err();
        assert!(error.message().contains("loop_array is not an array"));

        let error = plan(
            LoopBound::Number,
            params(Some(11), None, None),
            &json!({}),
            10,
            "c",
        )
        .unwrap_err();
        assert!(error.message().contains("exceeds maximum limit 10"));
    }

    #[test]
    fn variables_are_exposed_to_the_body_and_kept_in_a_scope_object() {
        let mut variables = Map::new();
        let _ = variables.insert("count".to_owned(), json!(2));
        let inputs = overlay_variables(json!({"index": 0, "seed": 1}), &variables);
        assert_eq!(
            inputs,
            json!({"index": 0, "seed": 1, "count": 2, "loop_variables": {"count": 2}})
        );
        // Empty variables leave the inputs untouched.
        assert_eq!(
            overlay_variables(json!({"a": 1}), &Map::new()),
            json!({"a": 1})
        );
        // Non-object inputs are passed through.
        assert_eq!(overlay_variables(json!(3), &variables), json!(3));
    }

    #[test]
    fn iteration_inputs_overlay_the_loop_position() {
        assert_eq!(
            iteration_inputs(&json!({"seed": 1}), 2, json!("x"), &json!({"a": 1})),
            json!({"seed": 1, "index": 2, "item": "x", "previous": {"a": 1}})
        );
        assert_eq!(
            iteration_inputs(&json!(null), 0, Value::Null, &Value::Null),
            json!({"index": 0, "item": null, "previous": null})
        );
    }

    #[test]
    fn resolve_variables_templates_against_the_inputs() {
        let mut variables = BTreeMap::new();
        let _ = variables.insert("count".to_owned(), json!("${start.count}"));
        let _ = variables.insert("label".to_owned(), json!("loop"));
        let resolved = resolve_variables(&variables, &json!({"start": {"count": 4}}));
        assert_eq!(resolved.get("count"), Some(&json!(4)));
        assert_eq!(resolved.get("label"), Some(&json!("loop")));
        assert!(resolve_variables(&BTreeMap::new(), &json!({})).is_empty());
    }

    #[test]
    fn no_terminator_names_the_missing_pieces() {
        let error = no_terminator();
        assert_eq!(error.kind(), ErrorKind::Workflow);
        assert!(error.message().contains("[101030]"), "{error}");
        assert!(error.message().contains("cannot terminate"), "{error}");
        assert!(error.message().contains("LoopBreakComponent"), "{error}");
    }
}
