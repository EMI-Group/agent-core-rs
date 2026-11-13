//! Operator base contract: tunables, structured updates, and the
//! [`Operator`] / [`PreviewableOperator`] traits.
//!
//! Ports `openjiuwen/core/operator/base.py`. An operator is a *parameter
//! handle*, not an executable unit: a consumer (the agent, a rail, or
//! `ReActAgentEvolve`) reads [`Operator::get_tunables`], mutates the handle
//! through [`Operator::set_parameter`] / [`Operator::apply_update`], and the
//! handle pushes the new value back through its
//! [`ParameterCallback`] so the live call path stays in sync.
//!
//! Two update entry points exist, mirroring Python:
//! - [`Operator::set_parameter`] — evolution updates; frozen targets are
//!   skipped.
//! - [`Operator::load_state`] — checkpoint recovery; freeze markers are
//!   bypassed and every restored field is pushed to the callback.
use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Callback fired after a parameter changed.
///
/// The first argument is the target name (the [`TunableSpec::name`]), the
/// second the new value. Implementations push it into the consumer that owns
/// the live call path.
pub type ParameterCallback = Arc<dyn Fn(&str, &Value) + Send + Sync>;

/// Type of a tunable parameter.
///
/// Wire names are `text`, `text_list`, `number`, `bool`, and `json`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TunableKind {
    /// A single string.
    #[default]
    Text,
    /// A list of strings.
    TextList,
    /// An integer or floating-point number.
    Number,
    /// A boolean.
    Bool,
    /// Any JSON value (maps, nested objects, arbitrary records).
    Json,
}

impl TunableKind {
    /// The snake-case wire name of the kind.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::TextList => "text_list",
            Self::Number => "number",
            Self::Bool => "bool",
            Self::Json => "json",
        }
    }
}

/// Describes one tunable parameter of an operator.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct TunableSpec {
    /// Target name accepted by [`Operator::set_parameter`].
    pub name: String,
    /// Human-readable description shown to the optimiser.
    pub description: String,
    /// Type the parameter accepts.
    pub kind: TunableKind,
    /// Whether the parameter is frozen against evolution updates.
    ///
    /// Frozen parameters still appear in [`Operator::get_tunables`] with this
    /// flag set; [`Operator::load_state`] restores them regardless.
    pub frozen: bool,
}

impl Default for TunableSpec {
    fn default() -> Self {
        Self {
            name: String::new(),
            description: String::new(),
            kind: TunableKind::Text,
            frozen: false,
        }
    }
}

impl TunableSpec {
    /// Builds an unfrozen spec for `name`.
    #[must_use]
    pub fn new(name: impl Into<String>, kind: TunableKind) -> Self {
        Self {
            name: name.into(),
            kind,
            ..Self::default()
        }
    }

    /// Sets the description.
    #[must_use]
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    /// Marks the spec frozen (or not).
    #[must_use]
    pub fn with_frozen(mut self, frozen: bool) -> Self {
        self.frozen = frozen;
        self
    }

    /// Type-checks `value` against this spec's [`TunableKind`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the value does not match the kind
    /// (for example a string for a [`TunableKind::Number`] spec).
    pub fn validate(&self, value: &Value) -> Result<()> {
        let ok = match self.kind {
            TunableKind::Text => value.is_string(),
            TunableKind::TextList => value
                .as_array()
                .is_some_and(|items| items.iter().all(Value::is_string)),
            TunableKind::Number => value.is_number(),
            TunableKind::Bool => value.is_boolean(),
            TunableKind::Json => true,
        };
        if ok {
            Ok(())
        } else {
            Err(Error::validation(format!(
                "tunable '{}' expects a {} value, got {}",
                self.name,
                self.kind.as_str(),
                json_type(value)
            )))
        }
    }
}

/// The JSON type name of `value`, used in validation messages.
fn json_type(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "bool",
        Value::Number(_) => "number",
        Value::String(_) => "text",
        Value::Array(_) => "list",
        Value::Object(_) => "object",
    }
}

/// A structured evolution update for one target.
///
/// Serializes to the Python `UpdateValue` shape: `{"mode": …, "payload": …}`.
/// The replace mode uses Python's wire name `replace` (and also accepts
/// `set` on deserialization).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "mode", content = "payload", rename_all = "snake_case")]
pub enum UpdateValue {
    /// Replace the target with the payload.
    #[serde(rename = "replace", alias = "set")]
    Set(Value),
    /// Merge the payload into the current value (object keys / array items).
    Merge(Value),
    /// Append the payload to the current value (arrays and text).
    Append(Value),
}

impl UpdateValue {
    /// Builds a replace update.
    #[must_use]
    pub fn set(payload: impl Into<Value>) -> Self {
        Self::Set(payload.into())
    }

    /// Builds a merge update.
    #[must_use]
    pub fn merge(payload: impl Into<Value>) -> Self {
        Self::Merge(payload.into())
    }

    /// Builds an append update.
    #[must_use]
    pub fn append(payload: impl Into<Value>) -> Self {
        Self::Append(payload.into())
    }

    /// The update payload.
    #[must_use]
    pub fn payload(&self) -> &Value {
        match self {
            Self::Set(value) | Self::Merge(value) | Self::Append(value) => value,
        }
    }

    /// The wire mode name: `replace`, `merge`, or `append`.
    #[must_use]
    pub const fn mode(&self) -> &'static str {
        match self {
            Self::Set(_) => "replace",
            Self::Merge(_) => "merge",
            Self::Append(_) => "append",
        }
    }
}

/// Result of applying one update to one target.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ApplyResult {
    /// Whether the update took effect.
    pub applied: bool,
    /// Target the update addressed.
    pub target: String,
    /// Why the update was skipped, or a short note when it was applied.
    pub reason: String,
}

impl ApplyResult {
    /// Records that the update was applied to `target`.
    #[must_use]
    pub fn applied(target: impl Into<String>) -> Self {
        Self {
            applied: true,
            target: target.into(),
            reason: String::new(),
        }
    }

    /// Records that the update to `target` was skipped for `reason`.
    #[must_use]
    pub fn skipped(target: impl Into<String>, reason: impl Into<String>) -> Self {
        Self {
            applied: false,
            target: target.into(),
            reason: reason.into(),
        }
    }

    /// Whether the update took effect.
    #[must_use]
    pub const fn ok(&self) -> bool {
        self.applied
    }
}

/// Computes the value a structured update would produce from `current`.
///
/// Objects merge shallowly (incoming keys win), arrays concatenate (an array
/// payload contributes its items, a scalar one is pushed), and every other
/// value is replaced.
#[must_use]
pub fn apply_merge(current: &Value, update: &UpdateValue) -> Value {
    match update {
        UpdateValue::Set(value) => value.clone(),
        UpdateValue::Merge(value) => match (current, value) {
            (Value::Object(current), Value::Object(patch)) => {
                let mut merged = current.clone();
                for (key, item) in patch {
                    merged.insert(key.clone(), item.clone());
                }
                Value::Object(merged)
            }
            _ => value.clone(),
        },
        UpdateValue::Append(value) => match current {
            Value::Array(items) => {
                let mut items = items.clone();
                match value {
                    Value::Array(patch) => items.extend(patch.iter().cloned()),
                    other => items.push(other.clone()),
                }
                Value::Array(items)
            }
            Value::String(text) => {
                let mut text = text.clone();
                text.push_str(&value_text(value));
                Value::String(text)
            }
            _ => value.clone(),
        },
    }
}

/// Renders a payload as text for string concatenation.
fn value_text(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        other => other.to_string(),
    }
}

/// A parameter handle of one call dimension.
///
/// Implementations own the authoritative parameter values; the consumer reads
/// them through [`get_state`](Self::get_state) and receives every change
/// through its [`ParameterCallback`].
#[async_trait]
pub trait Operator: Send + Sync {
    /// Stable operator identifier used for trajectory attribution and
    /// checkpointing (`"{agent_id}/{kind}_{name}"` by convention).
    fn operator_id(&self) -> &str;

    /// Describes the tunable parameters; frozen ones carry
    /// [`TunableSpec::frozen`].
    fn get_tunables(&self) -> Vec<TunableSpec>;

    /// The current parameter values as a JSON object.
    fn get_state(&self) -> Value;

    /// Sets one parameter (evolution entry point).
    ///
    /// Frozen targets return [`ApplyResult::skipped`]. Implementations fire
    /// their callback on success.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when `value` does not match the target.
    fn set_parameter(&mut self, target: &str, value: Value) -> Result<ApplyResult>;

    /// Restores every field present in `state` (checkpoint entry point).
    ///
    /// Freeze markers are not consulted and each restored field fires the
    /// callback.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when `state` is malformed.
    fn load_state(&mut self, state: Value) -> Result<()>;

    /// Applies a structured update to `target`.
    ///
    /// The default implementation merges/appends the payload into the current
    /// value ([`apply_merge`]) and delegates to
    /// [`set_parameter`](Self::set_parameter). Unknown targets are skipped.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the resulting value does not match
    /// the target spec.
    fn apply_update(&mut self, target: &str, update: UpdateValue) -> Result<ApplyResult> {
        let Some(spec) = self
            .get_tunables()
            .into_iter()
            .find(|spec| spec.name == target)
        else {
            return Ok(ApplyResult::skipped(target, "unknown target"));
        };
        let state = self.get_state();
        let current = state.get(target).cloned().unwrap_or(Value::Null);
        let next = apply_merge(&current, &update);
        spec.validate(&next)?;
        self.set_parameter(target, next)
    }
}

/// An operator that can preview an update without mutating itself.
#[async_trait]
pub trait PreviewableOperator: Operator {
    /// Whether `update` would apply to `target`, without mutating the handle.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the update payload does not match
    /// the target spec.
    fn preview_update(&self, target: &str, update: &UpdateValue) -> Result<ApplyResult>;
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::assert_is_empty
    )]

    use super::*;
    use serde_json::json;

    #[test]
    fn tunable_kinds_and_validation() {
        assert_eq!(TunableKind::TextList.as_str(), "text_list");
        assert!(TunableSpec::new("t", TunableKind::Text)
            .validate(&json!("hello"))
            .is_ok());
        assert!(TunableSpec::new("t", TunableKind::Text)
            .validate(&json!(1))
            .is_err());
        assert!(TunableSpec::new("l", TunableKind::TextList)
            .validate(&json!(["a", "b"]))
            .is_ok());
        assert!(TunableSpec::new("l", TunableKind::TextList)
            .validate(&json!(["a", 1]))
            .is_err());
        assert!(TunableSpec::new("n", TunableKind::Number)
            .validate(&json!(0.2))
            .is_ok());
        assert!(TunableSpec::new("b", TunableKind::Bool)
            .validate(&json!(false))
            .is_ok());
        assert!(TunableSpec::new("j", TunableKind::Json)
            .validate(&json!({"a": [1]}))
            .is_ok());
        let error = TunableSpec::new("n", TunableKind::Number)
            .validate(&json!("x"))
            .unwrap_err();
        assert!(error.to_string().contains("expects a number"));
        let spec = TunableSpec::new("s", TunableKind::Text)
            .with_description("a prompt")
            .with_frozen(true);
        assert!(spec.frozen);
        assert_eq!(spec.description, "a prompt");
        assert_eq!(spec.name, "s");
    }

    #[test]
    fn update_value_uses_python_wire_shape() {
        let set: UpdateValue =
            serde_json::from_value(json!({"mode": "replace", "payload": 1})).unwrap();
        assert_eq!(set, UpdateValue::Set(json!(1)));
        assert_eq!(set.mode(), "replace");
        let alias: UpdateValue =
            serde_json::from_value(json!({"mode": "set", "payload": 1})).unwrap();
        assert_eq!(alias, UpdateValue::Set(json!(1)));
        let append = UpdateValue::append(json!(["b"]));
        assert_eq!(
            serde_json::to_value(&append).unwrap(),
            json!({"mode": "append", "payload": ["b"]})
        );
        assert_eq!(append.payload(), &json!(["b"]));
        let merge = UpdateValue::merge(json!({"a": 1}));
        assert_eq!(merge.mode(), "merge");
        assert_eq!(UpdateValue::set(json!(1)).mode(), "replace");
    }

    #[test]
    fn apply_result_constructors_and_serde() {
        let applied = ApplyResult::applied("system_prompt");
        assert!(applied.ok());
        assert!(applied.reason.is_empty());
        let skipped = ApplyResult::skipped("user_prompt", "frozen");
        assert!(!skipped.ok());
        assert_eq!(skipped.reason, "frozen");
        let value = serde_json::to_value(&skipped).unwrap();
        assert_eq!(
            value,
            json!({"applied": false, "target": "user_prompt", "reason": "frozen"})
        );
        let round: ApplyResult = serde_json::from_value(value).unwrap();
        assert_eq!(round, skipped);
    }

    #[test]
    fn merge_is_shallow_and_append_concatenates() {
        let current = json!({"a": 1, "b": {"c": 2}});
        let merged = apply_merge(&current, &UpdateValue::merge(json!({"b": 3})));
        assert_eq!(merged, json!({"a": 1, "b": 3}));

        let items = apply_merge(&json!(["a"]), &UpdateValue::append(json!(["b", "c"])));
        assert_eq!(items, json!(["a", "b", "c"]));
        let one = apply_merge(&json!(["a"]), &UpdateValue::append(json!("b")));
        assert_eq!(one, json!(["a", "b"]));
        let text = apply_merge(&json!("ab"), &UpdateValue::append(json!("cd")));
        assert_eq!(text, json!("abcd"));

        assert_eq!(
            apply_merge(&json!({"a": 1}), &UpdateValue::set(json!(7))),
            json!(7)
        );
        assert_eq!(
            apply_merge(&json!(1), &UpdateValue::merge(json!({"a": 1}))),
            json!({"a": 1})
        );
    }
}
