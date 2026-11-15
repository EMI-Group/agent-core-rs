//! Loop-body control components: break requests and loop-variable assignments.
//!
//! Ports Python's `LoopBreakComponent` and `LoopSetVariableComponent`
//! (`openjiuwen/core/workflow/components/flow/loop/loop_comp.py`) onto this crate's flat,
//! in-process loop driver.
//!
//! Python couples both helpers to the loop scheduler: the break component calls a
//! `LoopController` injected at graph build time, and the set-variable component writes
//! into a sibling node's session outputs. This crate has no controller injection and no
//! per-node session outputs inside a loop body, so both components encode their intent as
//! **data**:
//!
//! - [`LoopBreakComponent`] returns `{"__loop_break__": <bool>}` (see
//!   [`LOOP_BREAK_MARKER`]).
//! - [`LoopSetVariableComponent`] returns `{"__loop_set_variables__": {<name>: <value>}}`
//!   (see [`LOOP_SET_VARIABLES_MARKER`]).
//!
//! The shared driver ([`crate::components::loop_driver`]) strips both fields from every
//! body output, folds the assignments into the loop variables, and stops the iteration
//! when a break is requested. Both components are also usable standalone, where the
//! marker simply *is* the component's output.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

use openjiuwen_common::error::Result;

use crate::ability::ComponentAbility;
use crate::component::{Component, ComponentCard};
use crate::condition::Condition;
use crate::context::ComponentContext;
use crate::error::{loop_param_invalid, loop_set_var_param_invalid};
use crate::reference;

/// The output field a [`LoopBreakComponent`] uses to request a break.
///
/// `true` asks the enclosing loop to stop after this body component; `false` records that
/// a break condition was checked and did not hold. The marker is removed from the value
/// the loop driver passes on, so it never reaches downstream body components.
pub const LOOP_BREAK_MARKER: &str = "__loop_break__";

/// The `component_type()` of a [`LoopBreakComponent`].
///
/// A loop container scans its body for it when deciding whether an unbounded loop can
/// terminate.
pub(crate) const BREAK_COMPONENT_TYPE: &str = "loop_break";

/// The output field a [`LoopSetVariableComponent`] uses to publish loop variables.
///
/// The value is a JSON object of `name → resolved value`; the loop driver removes the
/// field and folds the object into the container's loop variables. A non-object value is
/// dropped instead of failing the run.
pub const LOOP_SET_VARIABLES_MARKER: &str = "__loop_set_variables__";

/// The prefix Python writes in a variable key to address the loop scope (`${loop.count}`).
const LOOP_SCOPE_PREFIX: &str = "loop.";

/// What a body component's output asks the loop driver to do.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct BodyMarkers {
    /// A component asked the loop to stop.
    pub broken: bool,
    /// Loop-variable assignments published by a component, if any.
    pub variables: Option<Map<String, Value>>,
}

/// Removes the loop markers from `output` and returns them.
///
/// Only an object can carry markers; any other value is left untouched. A marker is
/// always removed, even when its value has the wrong type, so a malformed marker cannot
/// leak into a downstream body component.
pub(crate) fn take_markers(output: &mut Value) -> BodyMarkers {
    let Value::Object(map) = output else {
        return BodyMarkers::default();
    };
    let broken = map
        .remove(LOOP_BREAK_MARKER)
        .and_then(|value| value.as_bool())
        .unwrap_or(false);
    let variables = match map.remove(LOOP_SET_VARIABLES_MARKER) {
        Some(Value::Object(variables)) => Some(variables),
        _ => None,
    };
    BodyMarkers { broken, variables }
}

/// Static configuration of a [`LoopBreakComponent`].
///
/// Field names are `snake_case`, unknown fields are tolerated, and the type round-trips
/// through JSON so a workflow spec can be serialized and rebuilt.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct LoopBreakConfig {
    /// Optional break condition in the crate's condition DSL; `None` always breaks.
    ///
    /// The expression is evaluated against the component's inputs, so it can read the
    /// loop position (`${index}`) and the current loop variables.
    pub condition: Option<String>,
}

impl LoopBreakConfig {
    /// A configuration that breaks unconditionally.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the break condition expression.
    #[must_use]
    pub fn with_condition(mut self, condition: impl Into<String>) -> Self {
        self.condition = Some(condition.into());
        self
    }
}

/// The `loop_break` component: asks the enclosing loop to stop.
///
/// Standalone the component returns `{"__loop_break__": <bool>}`; inside a loop body the
/// enclosing container stops after the iteration and does not record the interrupted
/// round. Python raises `COMPONENT_LOOP_BREAK_EXECUTION_ERROR` when no controller was
/// injected; this port needs no controller, so the break is data rather than an error.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LoopBreakComponent {
    config: LoopBreakConfig,
}

impl LoopBreakComponent {
    /// A component that breaks unconditionally.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// A component configured by `config`.
    #[must_use]
    pub fn with_config(config: LoopBreakConfig) -> Self {
        Self { config }
    }

    /// Sets the break condition expression.
    #[must_use]
    pub fn with_condition(mut self, condition: impl Into<String>) -> Self {
        self.config = self.config.with_condition(condition);
        self
    }

    /// The component configuration.
    #[must_use]
    pub fn config(&self) -> &LoopBreakConfig {
        &self.config
    }

    /// Whether this invocation should break the loop for `inputs`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Workflow`] with code `101030` when the configured condition does
    /// not parse or cannot be evaluated against `inputs`.
    pub fn should_break(&self, inputs: &Value) -> Result<bool> {
        let Some(expression) = self.config.condition.as_deref() else {
            return Ok(true);
        };
        let condition = parse_condition(expression, "break condition")?;
        condition.evaluate(inputs).map_err(|error| {
            loop_param_invalid(&format!("break condition is invalid, {}", error.message()))
        })
    }
}

/// Parses `expression`, reporting a configuration error naming `what`.
fn parse_condition(expression: &str, what: &str) -> Result<Condition> {
    if expression.trim().is_empty() {
        return Err(loop_param_invalid(&format!("{what} is empty")));
    }
    Condition::new(expression)
        .map_err(|error| loop_param_invalid(&format!("{what} is invalid, {}", error.message())))
}

#[async_trait::async_trait]
impl Component for LoopBreakComponent {
    fn card(&self) -> ComponentCard {
        ComponentCard::new("loop_break", "loop_break").with_abilities(self.abilities())
    }

    fn abilities(&self) -> Vec<ComponentAbility> {
        vec![ComponentAbility::Invoke]
    }

    fn component_type(&self) -> &'static str {
        BREAK_COMPONENT_TYPE
    }

    /// Rejects a break condition that cannot be parsed, before any run.
    fn validate_config(&self) -> Result<()> {
        match self.config.condition.as_deref() {
            Some(expression) => parse_condition(expression, "break condition").map(|_| ()),
            None => Ok(()),
        }
    }

    async fn invoke(&self, inputs: Value, _ctx: &ComponentContext) -> Result<Value> {
        let broken = self.should_break(&inputs)?;
        Ok(json!({ LOOP_BREAK_MARKER: broken }))
    }
}

/// Static configuration of a [`LoopSetVariableComponent`].
///
/// Field names are `snake_case`, unknown fields are tolerated, and the type round-trips
/// through JSON so a workflow spec can be serialized and rebuilt.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct LoopSetVariableConfig {
    /// `name → value` assignments; a value may be a `${path}` template.
    ///
    /// Python writes the target as `${loop.<name>}` (the first segment names the node
    /// whose outputs are written); this port reads the same spelling and keeps the
    /// remainder of the reference as the loop-variable name.
    pub variable_mapping: BTreeMap<String, Value>,
}

impl LoopSetVariableConfig {
    /// A configuration with no assignments (rejected by [`Component::validate_config`]).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// A configuration that assigns `mapping`.
    #[must_use]
    pub fn with_variable_mapping(mut self, mapping: BTreeMap<String, Value>) -> Self {
        self.variable_mapping = mapping;
        self
    }

    /// Adds one assignment.
    #[must_use]
    pub fn with_variable(mut self, name: impl Into<String>, value: Value) -> Self {
        let _ = self.variable_mapping.insert(name.into(), value);
        self
    }
}

/// The `loop_set_variable` component: publishes loop-variable assignments.
///
/// Standalone the component returns `{"__loop_set_variables__": {<name>: <value>}}`;
/// inside a loop body the enclosing container folds the assignments into its loop
/// variables after the iteration, so the next iteration (and the closing condition) sees
/// them.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LoopSetVariableComponent {
    config: LoopSetVariableConfig,
}

impl LoopSetVariableComponent {
    /// A component configured by `config`.
    #[must_use]
    pub fn with_config(config: LoopSetVariableConfig) -> Self {
        Self { config }
    }

    /// A component assigning `mapping`.
    #[must_use]
    pub fn new(mapping: BTreeMap<String, Value>) -> Self {
        Self::with_config(LoopSetVariableConfig::new().with_variable_mapping(mapping))
    }

    /// The component configuration.
    #[must_use]
    pub fn config(&self) -> &LoopSetVariableConfig {
        &self.config
    }

    /// The configured assignments, in key order.
    #[must_use]
    pub fn mapping(&self) -> &BTreeMap<String, Value> {
        &self.config.variable_mapping
    }

    /// Resolves the assignments against `inputs`.
    ///
    /// A value that is a whole-string `${path}` reference is resolved against `inputs`
    /// (nested templates resolve too); every other value is copied verbatim.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Workflow`] with code `101030` when the mapping is empty or a
    /// variable name is not usable.
    pub fn assignments(&self, inputs: &Value) -> Result<Map<String, Value>> {
        if self.config.variable_mapping.is_empty() {
            return Err(loop_set_var_param_invalid(
                "variable_mapping is None or empty",
            ));
        }
        let template = Value::Object(
            self.config
                .variable_mapping
                .iter()
                .map(|(name, value)| (name.clone(), value.clone()))
                .collect(),
        );
        let Value::Object(resolved) = reference::resolve_schema(&template, inputs) else {
            return Ok(Map::new());
        };
        let mut assignments = Map::new();
        for (key, value) in resolved {
            let Some(name) = variable_name(&key) else {
                return Err(loop_set_var_param_invalid(&format!(
                    "variable name '{key}' is invalid"
                )));
            };
            let _ = assignments.insert(name, value);
        }
        Ok(assignments)
    }
}

/// Normalizes a configured variable key to its loop-variable name.
///
/// `${loop.count}` and `loop.count` both name `count`; `${count}` and a bare `count` are
/// accepted too. The remainder of a dotted reference is kept verbatim, so
/// `${loop.a.b}` names the flat variable `a.b` (Python nests it under a node's outputs).
fn variable_name(key: &str) -> Option<String> {
    let key = key.trim();
    if key.is_empty() {
        return None;
    }
    let key = if reference::is_ref_path(key) {
        reference::extract_origin_key(key)?
    } else {
        key
    };
    let key = key.trim();
    let name = key.strip_prefix(LOOP_SCOPE_PREFIX).unwrap_or(key).trim();
    if name.is_empty() || name.contains("${") {
        None
    } else {
        Some(name.to_owned())
    }
}

#[async_trait::async_trait]
impl Component for LoopSetVariableComponent {
    fn card(&self) -> ComponentCard {
        ComponentCard::new("loop_set_variable", "loop_set_variable")
            .with_abilities(self.abilities())
    }

    fn abilities(&self) -> Vec<ComponentAbility> {
        vec![ComponentAbility::Invoke]
    }

    fn component_type(&self) -> &'static str {
        "loop_set_variable"
    }

    /// Rejects an empty mapping or an unusable variable name, before any run.
    fn validate_config(&self) -> Result<()> {
        self.assignments(&Value::Null).map(|_| ())
    }

    async fn invoke(&self, inputs: Value, _ctx: &ComponentContext) -> Result<Value> {
        let assignments = self.assignments(&inputs)?;
        Ok(json!({ LOOP_SET_VARIABLES_MARKER: assignments }))
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

    fn context() -> ComponentContext {
        let session: Arc<dyn Session> = Arc::new(WorkflowSession::new("loop-control-tests"));
        ComponentContext::new(session, "loop_control", "wf")
    }

    fn mapping(pairs: &[(&str, Value)]) -> BTreeMap<String, Value> {
        pairs
            .iter()
            .map(|(name, value)| ((*name).to_owned(), value.clone()))
            .collect()
    }

    #[tokio::test]
    async fn a_bare_break_component_always_breaks() {
        let component = LoopBreakComponent::new();
        let output = component.invoke(json!({}), &context()).await.unwrap();
        assert_eq!(output, json!({LOOP_BREAK_MARKER: true}));
        assert_eq!(component.component_type(), "loop_break");
        assert_eq!(component.card().abilities, vec![ComponentAbility::Invoke]);
    }

    #[tokio::test]
    async fn a_break_condition_gates_the_marker() {
        let component = LoopBreakComponent::new().with_condition("${index} >= 2");
        let broken = component
            .invoke(json!({"index": 2}), &context())
            .await
            .unwrap();
        assert_eq!(broken, json!({LOOP_BREAK_MARKER: true}));
        let kept = component
            .invoke(json!({"index": 1}), &context())
            .await
            .unwrap();
        assert_eq!(kept, json!({LOOP_BREAK_MARKER: false}));
    }

    #[test]
    fn an_invalid_break_condition_is_a_loop_param_error() {
        let component = LoopBreakComponent::new().with_condition("${index} >");
        let error = component.validate_config().unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Workflow);
        assert!(error.message().contains("[101030]"), "{error}");
        assert!(error.message().contains("break condition is invalid"));

        let error = LoopBreakComponent::new().with_condition("  ");
        assert!(error.should_break(&json!({})).is_err());
    }

    #[test]
    fn a_valid_break_condition_compiles() {
        LoopBreakComponent::new()
            .with_condition("${done} == true")
            .validate_config()
            .unwrap();
        LoopBreakComponent::new().validate_config().unwrap();
    }

    #[tokio::test]
    async fn set_variable_resolves_templates_and_strips_the_loop_prefix() {
        let component = LoopSetVariableComponent::new(mapping(&[
            ("${loop.count}", json!("${index}")),
            ("label", json!("step")),
            ("${total}", json!({"nested": "${index}"})),
        ]));
        let output = component
            .invoke(json!({"index": 3}), &context())
            .await
            .unwrap();
        assert_eq!(
            output,
            json!({LOOP_SET_VARIABLES_MARKER: {
                "count": 3,
                "label": "step",
                "total": {"nested": 3},
            }})
        );
    }

    #[test]
    fn set_variable_rejects_an_empty_mapping() {
        let component = LoopSetVariableComponent::new(BTreeMap::new());
        let error = component.validate_config().unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Workflow);
        assert!(error.message().contains("[101031]"), "{error}");
        assert!(error
            .message()
            .contains("variable_mapping is None or empty"));
    }

    #[test]
    fn set_variable_names_are_normalized() {
        assert_eq!(variable_name("${loop.count}").as_deref(), Some("count"));
        assert_eq!(variable_name("loop.count").as_deref(), Some("count"));
        assert_eq!(variable_name("${count}").as_deref(), Some("count"));
        assert_eq!(variable_name("count").as_deref(), Some("count"));
        assert_eq!(variable_name("${loop.a.b}").as_deref(), Some("a.b"));
        assert_eq!(variable_name("${loop}").as_deref(), Some("loop"));
        assert_eq!(variable_name("   ").as_deref(), None);
        assert_eq!(variable_name("loop.").as_deref(), None);
    }

    #[test]
    fn an_invalid_variable_name_is_rejected() {
        let component = LoopSetVariableComponent::new(mapping(&[("${loop.}", json!(1))]));
        let error = component.validate_config().unwrap_err();
        assert!(error
            .message()
            .contains("variable name '${loop.}' is invalid"));
    }

    #[test]
    fn markers_are_taken_and_stripped() {
        let mut output = json!({
            LOOP_BREAK_MARKER: true,
            LOOP_SET_VARIABLES_MARKER: {"count": 1},
            "kept": 2,
        });
        let markers = take_markers(&mut output);
        assert!(markers.broken);
        let mut expected = Map::new();
        let _ = expected.insert("count".to_owned(), json!(1));
        assert_eq!(markers.variables, Some(expected));
        assert_eq!(output, json!({"kept": 2}));

        // A malformed marker is removed instead of leaking downstream.
        let mut output = json!({LOOP_BREAK_MARKER: "yes", LOOP_SET_VARIABLES_MARKER: 3});
        let markers = take_markers(&mut output);
        assert!(!markers.broken);
        assert!(markers.variables.is_none());
        assert_eq!(output, json!({}));

        // Non-objects carry no markers and are left alone.
        let mut output = json!([1, 2]);
        assert_eq!(take_markers(&mut output), BodyMarkers::default());
        assert_eq!(output, json!([1, 2]));
    }

    #[test]
    fn config_round_trips_and_tolerates_unknown_fields() {
        let config = LoopBreakConfig::new().with_condition("${index} > 1");
        let wire = serde_json::to_value(&config).unwrap();
        assert_eq!(wire, json!({"condition": "${index} > 1"}));
        let parsed: LoopBreakConfig =
            serde_json::from_value(json!({"condition": "${index} > 1", "future": 1})).unwrap();
        assert_eq!(parsed, config);

        let config = LoopSetVariableConfig::new().with_variable("${loop.count}", json!(1));
        let wire = serde_json::to_value(&config).unwrap();
        assert_eq!(wire, json!({"variable_mapping": {"${loop.count}": 1}}));
        let parsed: LoopSetVariableConfig =
            serde_json::from_value(json!({"variable_mapping": {"${loop.count}": 1}})).unwrap();
        assert_eq!(parsed, config);
        assert_eq!(
            serde_json::from_value::<LoopSetVariableConfig>(json!({})).unwrap(),
            LoopSetVariableConfig::default()
        );
        assert_eq!(
            serde_json::from_value::<LoopBreakConfig>(json!({})).unwrap(),
            LoopBreakConfig::default()
        );
    }
}
