//! Textual parameters and per-operator evolution state.
//!
//! Ports the parameter container of Python
//! `openjiuwen.agent_evolving.optimizer.base.TextualParameter` and lifts it to a
//! per-operator [`OperatorState`] so a checkpoint can round-trip the whole
//! search state:
//!
//! - [`TextualParameter`] — one named value (an instruction, a tool description,
//!   a memory entry list, …) plus the textual gradient accumulated since the last
//!   accepted update.
//! - [`OperatorState`] — the ordered parameter map of one operator, tagged with
//!   its [`OperatorKind`] so optimizers can route signals by layer.
//! - [`OperatorState::state`] / [`OperatorState::restore`] — the JSON bridge into
//!   `EvolveCheckpoint.operators_state`.
//!
//! Everything here is pure data: no clocks, no model calls, no environment reads.

use std::collections::BTreeMap;

use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

/// Longest accumulated gradient kept on a [`TextualParameter`], in characters.
///
/// Gradients are evidence handed to a rewrite prompt, so they are bounded. When a
/// new piece of feedback pushes the accumulator past the bound the *oldest* text
/// is dropped and the most recent feedback is kept, which matches the "learn from
/// the latest failure" intent of the Python optimizers.
pub const MAX_GRADIENT_CHARS: usize = 2_000;

/// The evolution layer an [`OperatorState`] belongs to.
///
/// The vocabulary is closed so the shared search engine can decide which signals
/// a plain (unqualified) parameter name applies to.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum OperatorKind {
    /// Prompt/instruction text (Python's `llm_call` optimizers).
    #[default]
    Instruction,
    /// A tool description (Python's `tool_call` optimizers).
    Tool,
    /// Stored long-term memory entries (Python's `memory_call` optimizers).
    Memory,
    /// A skill experience document (Python's `skill_call` optimizers).
    Skill,
    /// One context-window section set (Python's `context_evolve_call` / Metis).
    Context,
    /// A test-time skill experience tip bank (TTSE).
    Tip,
}

impl OperatorKind {
    /// The kind as its `snake_case` wire string.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Instruction => "instruction",
            Self::Tool => "tool",
            Self::Memory => "memory",
            Self::Skill => "skill",
            Self::Context => "context",
            Self::Tip => "tip",
        }
    }
}

/// One named textual value and its accumulated textual gradient.
///
/// `version` starts at `0` and is bumped by [`TextualParameter::set_value`] and
/// [`TextualParameter::apply_gradient`], so a checkpoint can tell whether a
/// proposal actually changed a value.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct TextualParameter {
    /// Stable parameter name, for example `instruction` or `entries`.
    pub name: String,
    /// The current textual value.
    #[serde(default)]
    pub value: String,
    /// Accumulated textual feedback (`\n`-joined), cleared by
    /// [`TextualParameter::set_value`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gradient: Option<String>,
    /// Human-readable description of what the parameter controls.
    #[serde(default)]
    pub description: String,
    /// Monotonic version counter.
    #[serde(default)]
    pub version: u64,
}

impl TextualParameter {
    /// Creates a version-`0` parameter with no gradient and no description.
    #[must_use]
    pub fn new(name: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            value: value.into(),
            gradient: None,
            description: String::new(),
            version: 0,
        }
    }

    /// Returns the parameter with `description` replaced.
    #[must_use]
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    /// The parameter name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The current value.
    #[must_use]
    pub fn value(&self) -> &str {
        &self.value
    }

    /// The accumulated gradient, if any.
    #[must_use]
    pub fn gradient(&self) -> Option<&str> {
        self.gradient.as_deref()
    }

    /// The parameter description.
    #[must_use]
    pub fn description(&self) -> &str {
        &self.description
    }

    /// The monotonic version counter.
    #[must_use]
    pub fn version(&self) -> u64 {
        self.version
    }

    /// Replaces the value, clears the accumulated gradient and bumps `version`.
    pub fn set_value(&mut self, value: impl Into<String>) {
        self.value = value.into();
        self.gradient = None;
        self.version = self.version.saturating_add(1);
    }

    /// Appends `gradient` to the accumulated feedback and bumps `version`.
    ///
    /// Lines are joined with `\n`; surrounding whitespace is trimmed and empty
    /// feedback is ignored. The accumulator keeps at most
    /// [`MAX_GRADIENT_CHARS`] characters, dropping the oldest text first, and the
    /// cut respects UTF-8 boundaries.
    pub fn apply_gradient(&mut self, gradient: impl Into<String>) {
        let gradient = gradient.into();
        let gradient = gradient.trim();
        if gradient.is_empty() {
            return;
        }
        let combined = match self.gradient.as_deref() {
            Some(existing) if !existing.is_empty() => format!("{existing}\n{gradient}"),
            _ => gradient.to_owned(),
        };
        self.gradient = Some(truncate_tail(&combined, MAX_GRADIENT_CHARS));
        self.version = self.version.saturating_add(1);
    }
}

/// The parameter map of one evolution operator.
///
/// `operator_id` is the key used inside `EvolveCheckpoint.operators_state`; the
/// tool optimizers use the documented `tool.{name}` convention.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct OperatorState {
    /// Stable operator identifier.
    pub operator_id: String,
    /// The evolution layer the operator belongs to.
    pub kind: OperatorKind,
    /// Ordered parameter map; [`BTreeMap`] keeps serialization deterministic.
    #[serde(default)]
    pub parameters: BTreeMap<String, TextualParameter>,
}

impl OperatorState {
    /// Creates an operator with no parameters.
    #[must_use]
    pub fn new(operator_id: impl Into<String>, kind: OperatorKind) -> Self {
        Self {
            operator_id: operator_id.into(),
            kind,
            parameters: BTreeMap::new(),
        }
    }

    /// Returns the operator with one extra parameter.
    #[must_use]
    pub fn with_parameter(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        let parameter = TextualParameter::new(name, value);
        self.parameters.insert(parameter.name.clone(), parameter);
        self
    }

    /// The operator identifier.
    #[must_use]
    pub fn operator_id(&self) -> &str {
        &self.operator_id
    }

    /// The evolution layer the operator belongs to.
    #[must_use]
    pub const fn kind(&self) -> OperatorKind {
        self.kind
    }

    /// The ordered parameter map.
    #[must_use]
    pub fn parameters(&self) -> &BTreeMap<String, TextualParameter> {
        &self.parameters
    }

    /// The number of parameters.
    #[must_use]
    pub fn len(&self) -> usize {
        self.parameters.len()
    }

    /// Whether the operator carries no parameters.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.parameters.is_empty()
    }

    /// One parameter by name.
    #[must_use]
    pub fn parameter(&self, name: &str) -> Option<&TextualParameter> {
        self.parameters.get(name)
    }

    /// One parameter by name, mutably.
    pub fn parameter_mut(&mut self, name: &str) -> Option<&mut TextualParameter> {
        self.parameters.get_mut(name)
    }

    /// Creates or updates one parameter, preserving its description.
    ///
    /// A new parameter starts with an empty description; an existing one keeps it
    /// but has its value replaced and its gradient cleared.
    pub fn set_parameter(&mut self, name: impl Into<String>, value: impl Into<String>) {
        let name = name.into();
        if let Some(parameter) = self.parameters.get_mut(&name) {
            parameter.set_value(value);
        } else {
            let parameter = TextualParameter::new(name.clone(), value);
            self.parameters.insert(name, parameter);
        }
    }

    /// Applies a parameter-name → value map and returns how many entries were
    /// written.
    ///
    /// Missing parameters are created so a caller restoring an older, smaller
    /// state never loses an update.
    pub fn apply(&mut self, values: &BTreeMap<String, String>) -> usize {
        let mut applied = 0;
        for (name, value) in values {
            self.set_parameter(name.clone(), value.clone());
            applied += 1;
        }
        applied
    }

    /// Projects the operator to the JSON object stored per operator in a
    /// checkpoint.
    ///
    /// The shape is `{parameter_name: {"value", "gradient", "description",
    /// "version"}}`; `gradient` is `null` when no feedback is accumulated.
    #[must_use]
    pub fn state(&self) -> Value {
        let parameters: Map<String, Value> = self
            .parameters
            .iter()
            .map(|(name, parameter)| {
                let gradient = parameter
                    .gradient
                    .clone()
                    .map_or(Value::Null, Value::String);
                let state = json!({
                    "value": parameter.value.clone(),
                    "gradient": gradient,
                    "description": parameter.description.clone(),
                    "version": parameter.version,
                });
                (name.clone(), state)
            })
            .collect();
        Value::Object(parameters)
    }

    /// Restores parameters from [`OperatorState::state`] output.
    ///
    /// The input is intentionally tolerant: unknown fields are ignored, missing
    /// fields keep their current value, a bare string is accepted as the value,
    /// and an object wrapped in `{"parameters": {...}}` is unwrapped. Only a
    /// non-object input is rejected.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] when `state` is not a JSON object, or when its
    /// `parameters` member is present but is not an object.
    pub fn restore(&mut self, state: &Value) -> Result<()> {
        let Some(object) = state.as_object() else {
            return Err(Error::validation(
                "OperatorState.restore expects a JSON object",
            ));
        };
        let parameters = match object.get("parameters") {
            Some(Value::Object(inner)) => inner,
            Some(_) => {
                return Err(Error::validation(
                    "OperatorState.restore: parameters must be a JSON object",
                ));
            }
            None => object,
        };
        for (name, value) in parameters {
            match value {
                Value::Object(fields) => {
                    let parameter = self
                        .parameters
                        .entry(name.clone())
                        .or_insert_with(|| TextualParameter::new(name.clone(), ""));
                    if let Some(value) = fields.get("value").and_then(Value::as_str) {
                        value.clone_into(&mut parameter.value);
                    }
                    match fields.get("gradient") {
                        Some(Value::String(gradient)) => {
                            parameter.gradient = Some(gradient.clone());
                        }
                        Some(Value::Null) | None => parameter.gradient = None,
                        Some(_) => {}
                    }
                    if let Some(description) = fields.get("description").and_then(Value::as_str) {
                        description.clone_into(&mut parameter.description);
                    }
                    if let Some(version) = fields.get("version").and_then(Value::as_u64) {
                        parameter.version = version;
                    }
                }
                Value::String(text) => {
                    if name == "operator_id" || name == "kind" {
                        continue;
                    }
                    let parameter = self
                        .parameters
                        .entry(name.clone())
                        .or_insert_with(|| TextualParameter::new(name.clone(), ""));
                    text.clone_into(&mut parameter.value);
                    parameter.gradient = None;
                    parameter.version = parameter.version.saturating_add(1);
                }
                Value::Null | Value::Bool(_) | Value::Number(_) | Value::Array(_) => {}
            }
        }
        Ok(())
    }
}

/// Keeps the last `max_chars` characters, respecting UTF-8 boundaries.
fn truncate_tail(text: &str, max_chars: usize) -> String {
    let total = text.chars().count();
    if total <= max_chars {
        return text.to_owned();
    }
    text.chars().skip(total - max_chars).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parameter_versions_and_gradient_accumulation() {
        let mut parameter = TextualParameter::new("instruction", "initial");
        assert_eq!(parameter.version(), 0);
        parameter.apply_gradient("first");
        parameter.apply_gradient("second");
        assert_eq!(parameter.gradient(), Some("first\nsecond"));
        assert_eq!(parameter.version(), 2);
        parameter.set_value("next");
        assert_eq!(parameter.value(), "next");
        assert_eq!(parameter.gradient(), None);
        assert_eq!(parameter.version(), 3);
        parameter.apply_gradient("   ");
        assert_eq!(parameter.gradient(), None);
        assert_eq!(parameter.version(), 3);
    }

    #[test]
    fn gradient_accumulator_is_bounded_and_utf8_safe() {
        let mut parameter = TextualParameter::new("instruction", "");
        parameter.apply_gradient("é".repeat(MAX_GRADIENT_CHARS));
        parameter.apply_gradient("tail");
        let gradient = parameter.gradient().expect("gradient");
        assert_eq!(gradient.chars().count(), MAX_GRADIENT_CHARS);
        assert!(gradient.ends_with("tail"));
    }

    #[test]
    fn operator_state_round_trips_through_json() {
        let mut operator = OperatorState::new("planner", OperatorKind::Instruction)
            .with_parameter("instruction", "You plan.")
            .with_parameter("style", "terse");
        operator
            .parameter_mut("instruction")
            .expect("parameter")
            .apply_gradient("add verification");
        let state = operator.state();
        assert_eq!(state["instruction"]["value"], json!("You plan."));
        assert_eq!(state["instruction"]["gradient"], json!("add verification"));
        assert_eq!(state["instruction"]["version"], json!(1));

        let mut restored = OperatorState::new("planner", OperatorKind::Instruction);
        restored.restore(&state).expect("restore");
        assert_eq!(restored, operator);
    }

    #[test]
    fn restore_tolerates_unknown_and_missing_fields() {
        let mut operator = OperatorState::new("tool.search", OperatorKind::Tool);
        operator
            .restore(&json!({
                "description": {"value": "Search.", "future_field": true},
                "legacy": "plain text",
                "operator_id": "ignored",
                "kind": "tool",
            }))
            .expect("restore");
        assert_eq!(
            operator
                .parameter("description")
                .map(TextualParameter::value),
            Some("Search.")
        );
        assert_eq!(
            operator.parameter("legacy").map(TextualParameter::value),
            Some("plain text")
        );
        assert!(operator.parameter("operator_id").is_none());
        assert!(operator.parameter("kind").is_none());
    }

    #[test]
    fn restore_rejects_non_objects() {
        let mut operator = OperatorState::new("planner", OperatorKind::Instruction);
        assert!(operator.restore(&json!(["nope"])).is_err());
        assert!(operator.restore(&json!({"parameters": "nope"})).is_err());
    }

    #[test]
    fn apply_creates_missing_parameters() {
        let mut operator = OperatorState::new("planner", OperatorKind::Instruction)
            .with_parameter("instruction", "old");
        let values = BTreeMap::from([
            ("instruction".to_owned(), "new".to_owned()),
            ("extra".to_owned(), "value".to_owned()),
        ]);
        assert_eq!(operator.apply(&values), 2);
        assert_eq!(
            operator
                .parameter("instruction")
                .map(TextualParameter::value),
            Some("new")
        );
        assert_eq!(
            operator
                .parameter("instruction")
                .map(TextualParameter::version),
            Some(1)
        );
        assert_eq!(
            operator.parameter("extra").map(TextualParameter::value),
            Some("value")
        );
    }
}
