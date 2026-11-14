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
use std::collections::BTreeMap;
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
/// Wire names are `text`, `text_list`, `number`, `bool`, `json`, and
/// `skill_experience` (a skill-experience record list).
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
    /// A list of skill-experience records, staged as a pending change.
    SkillExperience,
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
            Self::SkillExperience => "skill_experience",
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
    /// Path of the parameter inside its consumer (Python `TunableSpec.path`).
    #[serde(skip_serializing_if = "String::is_empty")]
    pub path: String,
    /// Optional constraint record handed to the optimiser (Python
    /// `TunableSpec.constraint`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub constraint: Option<Value>,
}

impl Default for TunableSpec {
    fn default() -> Self {
        Self {
            name: String::new(),
            description: String::new(),
            kind: TunableKind::Text,
            frozen: false,
            path: String::new(),
            constraint: None,
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

    /// Sets the consumer-side parameter path (Python `TunableSpec.path`).
    #[must_use]
    pub fn with_path(mut self, path: impl Into<String>) -> Self {
        self.path = path.into();
        self
    }

    /// Attaches the constraint record handed to the optimiser.
    #[must_use]
    pub fn with_constraint(mut self, constraint: Value) -> Self {
        self.constraint = Some(constraint);
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
            TunableKind::Json | TunableKind::SkillExperience => true,
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

/// How an update changes the target value.
///
/// Wire names are `replace`, `append`, and `merge`; deserialization also
/// accepts Python's legacy `set` spelling for [`UpdateMode::Replace`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UpdateMode {
    /// Replace the target with the payload.
    #[default]
    #[serde(rename = "replace", alias = "set")]
    Replace,
    /// Append the payload to the current value (arrays and text).
    Append,
    /// Merge the payload into the current value (object keys / array items).
    Merge,
}

impl UpdateMode {
    /// The wire name: `replace`, `append`, or `merge`.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Replace => "replace",
            Self::Append => "append",
            Self::Merge => "merge",
        }
    }

    /// Parses a wire name; `set` is accepted as the legacy spelling of
    /// `replace`.
    #[must_use]
    pub fn parse(text: &str) -> Option<Self> {
        match text.trim() {
            "replace" | "set" => Some(Self::Replace),
            "append" => Some(Self::Append),
            "merge" => Some(Self::Merge),
            _ => None,
        }
    }

    /// Whether this is [`UpdateMode::Replace`].
    #[must_use]
    pub const fn is_replace(&self) -> bool {
        matches!(self, Self::Replace)
    }
}

/// Whether an update mutates operator state immediately or stages a pending
/// change owned by the caller's lifecycle manager.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UpdateEffect {
    /// The update changes the operator's state right away.
    #[default]
    State,
    /// The update produces records that need caller-owned approval before they
    /// are persisted.
    PendingChange,
}

impl UpdateEffect {
    /// The wire name: `state` or `pending_change`.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::State => "state",
            Self::PendingChange => "pending_change",
        }
    }

    /// Parses a wire name.
    #[must_use]
    pub fn parse(text: &str) -> Option<Self> {
        match text.trim() {
            "state" => Some(Self::State),
            "pending_change" => Some(Self::PendingChange),
            _ => None,
        }
    }

    /// Whether this is [`UpdateEffect::State`].
    #[must_use]
    pub const fn is_state(&self) -> bool {
        matches!(self, Self::State)
    }
}

/// A structured evolution update for one target.
///
/// Serializes to the Python `UpdateValue` shape: `{"payload": …, "mode": …,
/// "effect": …}` with `change_type` and `metadata` present only when set.
/// Defaults are a `null` payload, [`UpdateMode::Replace`], and
/// [`UpdateEffect::State`].
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct UpdateValue {
    /// Value the update carries.
    pub payload: Value,
    /// How the payload changes the target.
    pub mode: UpdateMode,
    /// Whether the result is state or a caller-owned pending change.
    #[serde(skip_serializing_if = "UpdateEffect::is_state")]
    pub effect: UpdateEffect,
    /// Record-kind tag consumed by the staging lifecycle.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub change_type: Option<String>,
    /// Free-form attribution copied onto the [`ApplyResult`].
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, Value>,
}

impl UpdateValue {
    /// Builds an update with an explicit mode and the default state effect.
    #[must_use]
    pub fn new(payload: impl Into<Value>, mode: UpdateMode) -> Self {
        Self {
            payload: payload.into(),
            mode,
            ..Self::default()
        }
    }

    /// Builds a replace update.
    #[must_use]
    pub fn set(payload: impl Into<Value>) -> Self {
        Self::new(payload, UpdateMode::Replace)
    }

    /// Builds a merge update.
    #[must_use]
    pub fn merge(payload: impl Into<Value>) -> Self {
        Self::new(payload, UpdateMode::Merge)
    }

    /// Builds an append update.
    #[must_use]
    pub fn append(payload: impl Into<Value>) -> Self {
        Self::new(payload, UpdateMode::Append)
    }

    /// Sets the effect.
    #[must_use]
    pub fn with_effect(mut self, effect: UpdateEffect) -> Self {
        self.effect = effect;
        self
    }

    /// Sets the record-kind tag.
    #[must_use]
    pub fn with_change_type(mut self, change_type: impl Into<String>) -> Self {
        self.change_type = Some(change_type.into());
        self
    }

    /// Replaces the metadata object.
    #[must_use]
    pub fn with_metadata(mut self, metadata: BTreeMap<String, Value>) -> Self {
        self.metadata = metadata;
        self
    }

    /// Inserts one metadata entry.
    #[must_use]
    pub fn with_metadata_entry(mut self, key: impl Into<String>, value: Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// The update payload.
    #[must_use]
    pub fn payload(&self) -> &Value {
        &self.payload
    }

    /// The wire mode name: `replace`, `merge`, or `append`.
    #[must_use]
    pub const fn mode(&self) -> &'static str {
        self.mode.as_str()
    }

    /// Whether the update mutates operator state.
    #[must_use]
    pub const fn is_state(&self) -> bool {
        self.effect.is_state()
    }

    /// Whether the update stages a caller-owned pending change.
    #[must_use]
    pub const fn is_pending_change(&self) -> bool {
        matches!(self.effect, UpdateEffect::PendingChange)
    }
}

/// Result of applying one update to one target.
///
/// Serializes with the Python `ApplyResult` field names (`operator_id`,
/// `target`, `applied`, `mode`, `effect`, `value`, `records`, `change_type`,
/// `lifecycle_stage`, `pending_change_id`, `errors`, `metadata`); empty fields
/// are omitted so a plain state update stays compact. [`Self::reason`] is a
/// Rust-only note kept for the pre-existing constructors.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ApplyResult {
    /// Operator that processed the update (empty for an unattributed result).
    #[serde(skip_serializing_if = "String::is_empty")]
    pub operator_id: String,
    /// Target the update addressed.
    #[serde(skip_serializing_if = "String::is_empty")]
    pub target: String,
    /// Whether the update took effect.
    pub applied: bool,
    /// Mode of the applied update.
    #[serde(skip_serializing_if = "UpdateMode::is_replace")]
    pub mode: UpdateMode,
    /// Effect of the applied update.
    #[serde(skip_serializing_if = "UpdateEffect::is_state")]
    pub effect: UpdateEffect,
    /// Original update payload.
    #[serde(skip_serializing_if = "Value::is_null")]
    pub value: Value,
    /// Records produced by a preview or staged update.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub records: Vec<Value>,
    /// Record-kind tag of the produced records.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub change_type: Option<String>,
    /// Lifecycle stage reached by a preview (`local_apply_completed`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lifecycle_stage: Option<String>,
    /// Identifier of the staged pending change, when one exists.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pending_change_id: Option<String>,
    /// Why the update was rejected; an empty list means no failure.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub errors: Vec<String>,
    /// Free-form attribution copied from the update.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, Value>,
    /// Short human note from [`Self::applied`] / [`Self::skipped`] (Rust-only;
    /// Python's structured contract is [`Self::errors`]).
    #[serde(skip_serializing_if = "String::is_empty")]
    pub reason: String,
}

impl ApplyResult {
    /// Records that the update was applied to `target`.
    #[must_use]
    pub fn applied(target: impl Into<String>) -> Self {
        Self {
            applied: true,
            target: target.into(),
            ..Self::default()
        }
    }

    /// Records that the update to `target` was skipped for `reason`.
    #[must_use]
    pub fn skipped(target: impl Into<String>, reason: impl Into<String>) -> Self {
        Self {
            applied: false,
            target: target.into(),
            reason: reason.into(),
            ..Self::default()
        }
    }

    /// Builds an unattributed rejection mirroring `update`.
    ///
    /// Mode, effect, payload, change type, and metadata are copied from the
    /// update, matching Python's rejection results.
    #[must_use]
    pub fn rejection(
        operator_id: impl Into<String>,
        target: impl Into<String>,
        update: &UpdateValue,
        errors: Vec<String>,
    ) -> Self {
        Self {
            operator_id: operator_id.into(),
            target: target.into(),
            applied: false,
            mode: update.mode,
            effect: update.effect,
            value: update.payload.clone(),
            change_type: update.change_type.clone(),
            errors,
            metadata: update.metadata.clone(),
            ..Self::default()
        }
    }

    /// Sets the operator id.
    #[must_use]
    pub fn with_operator_id(mut self, operator_id: impl Into<String>) -> Self {
        self.operator_id = operator_id.into();
        self
    }

    /// Sets the update mode.
    #[must_use]
    pub fn with_mode(mut self, mode: UpdateMode) -> Self {
        self.mode = mode;
        self
    }

    /// Sets the update effect.
    #[must_use]
    pub fn with_effect(mut self, effect: UpdateEffect) -> Self {
        self.effect = effect;
        self
    }

    /// Sets the original payload.
    #[must_use]
    pub fn with_value(mut self, value: Value) -> Self {
        self.value = value;
        self
    }

    /// Sets the produced records.
    #[must_use]
    pub fn with_records(mut self, records: Vec<Value>) -> Self {
        self.records = records;
        self
    }

    /// Sets the record-kind tag.
    #[must_use]
    pub fn with_change_type(mut self, change_type: impl Into<String>) -> Self {
        self.change_type = Some(change_type.into());
        self
    }

    /// Sets the lifecycle stage.
    #[must_use]
    pub fn with_lifecycle_stage(mut self, stage: impl Into<String>) -> Self {
        self.lifecycle_stage = Some(stage.into());
        self
    }

    /// Sets the pending-change identifier.
    #[must_use]
    pub fn with_pending_change_id(mut self, change_id: impl Into<String>) -> Self {
        self.pending_change_id = Some(change_id.into());
        self
    }

    /// Appends one error.
    #[must_use]
    pub fn with_error(mut self, error: impl Into<String>) -> Self {
        self.errors.push(error.into());
        self
    }

    /// Replaces the metadata object.
    #[must_use]
    pub fn with_metadata(mut self, metadata: BTreeMap<String, Value>) -> Self {
        self.metadata = metadata;
        self
    }

    /// Inserts one metadata entry.
    #[must_use]
    pub fn with_metadata_entry(mut self, key: impl Into<String>, value: Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Whether the update took effect and reported no errors.
    #[must_use]
    pub fn ok(&self) -> bool {
        self.applied && self.errors.is_empty()
    }
}

/// Computes the value a structured update would produce from `current`.
///
/// Objects merge shallowly (incoming keys win), arrays concatenate (an array
/// payload contributes its items, a scalar one is pushed), and every other
/// value is replaced.
#[must_use]
pub fn apply_merge(current: &Value, update: &UpdateValue) -> Value {
    let value = &update.payload;
    match update.mode {
        UpdateMode::Replace => value.clone(),
        UpdateMode::Merge => match (current, value) {
            (Value::Object(current), Value::Object(patch)) => {
                let mut merged = current.clone();
                for (key, item) in patch {
                    merged.insert(key.clone(), item.clone());
                }
                Value::Object(merged)
            }
            _ => value.clone(),
        },
        UpdateMode::Append => match current {
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
    /// The default implementation mirrors Python's compatibility behaviour for
    /// state operators: a [`UpdateEffect::PendingChange`] update is rejected
    /// because only preview operators can produce pending changes. Otherwise
    /// the payload is merged/appended into the current value
    /// ([`apply_merge`]) and handed to [`set_parameter`](Self::set_parameter);
    /// unknown targets are skipped.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the resulting value does not match
    /// the target spec.
    fn apply_update(&mut self, target: &str, update: UpdateValue) -> Result<ApplyResult> {
        if !update.is_state() {
            return Ok(ApplyResult::rejection(
                self.operator_id(),
                target,
                &update,
                vec![format!(
                    "unsupported update mode/effect for compatibility operator: {}/{}",
                    update.mode(),
                    update.effect.as_str()
                )],
            ));
        }
        let Some(spec) = self
            .get_tunables()
            .into_iter()
            .find(|spec| spec.name == target)
        else {
            return Ok(ApplyResult::rejection(
                self.operator_id(),
                target,
                &update,
                vec![format!("unknown target '{target}'")],
            ));
        };
        let state = self.get_state();
        let current = state.get(target).cloned().unwrap_or(Value::Null);
        let next = apply_merge(&current, &update);
        spec.validate(&next)?;
        let mut result = self.set_parameter(target, next)?;
        result.operator_id = self.operator_id().to_owned();
        result.mode = update.mode;
        result.effect = update.effect;
        result.value = update.payload;
        result.change_type = update.change_type;
        if result.metadata.is_empty() {
            result.metadata = update.metadata;
        }
        Ok(result)
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
        assert_eq!(set, UpdateValue::set(json!(1)));
        assert_eq!(set.mode(), "replace");
        assert_eq!(set.mode, UpdateMode::Replace);
        assert_eq!(set.effect, UpdateEffect::State);
        assert!(set.is_state());
        let alias: UpdateValue =
            serde_json::from_value(json!({"mode": "set", "payload": 1})).unwrap();
        assert_eq!(alias, UpdateValue::set(json!(1)));
        let append = UpdateValue::append(json!(["b"]));
        assert_eq!(
            serde_json::to_value(&append).unwrap(),
            json!({"mode": "append", "payload": ["b"]})
        );
        assert_eq!(append.payload(), &json!(["b"]));
        let merge = UpdateValue::merge(json!({"a": 1}));
        assert_eq!(merge.mode(), "merge");
        assert_eq!(UpdateValue::set(json!(1)).mode(), "replace");
        let pending = UpdateValue::append(json!({"text": "x"}))
            .with_effect(UpdateEffect::PendingChange)
            .with_change_type("skill_experience_entry")
            .with_metadata_entry("source", json!("online"));
        assert!(pending.is_pending_change());
        let value = serde_json::to_value(&pending).unwrap();
        assert_eq!(value["effect"], json!("pending_change"));
        assert_eq!(value["change_type"], json!("skill_experience_entry"));
        assert_eq!(value["metadata"], json!({"source": "online"}));
        let round: UpdateValue = serde_json::from_value(value).unwrap();
        assert_eq!(round, pending);
        assert_eq!(UpdateValue::default(), UpdateValue::set(Value::Null));
        assert_eq!(UpdateValue::default().effect, UpdateEffect::State);
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

        let rich = ApplyResult::applied("experiences")
            .with_operator_id("skill_experience_alpha")
            .with_mode(UpdateMode::Append)
            .with_effect(UpdateEffect::PendingChange)
            .with_value(json!({"text": "x"}))
            .with_records(vec![json!({"text": "x"})])
            .with_change_type("skill_experience_entry")
            .with_lifecycle_stage("local_apply_completed")
            .with_pending_change_id("skill_evolve_00000001")
            .with_metadata_entry("skill_name", json!("alpha"));
        assert!(rich.ok());
        let value = serde_json::to_value(&rich).unwrap();
        assert_eq!(value["applied"], json!(true));
        assert_eq!(value["mode"], json!("append"));
        assert_eq!(value["effect"], json!("pending_change"));
        assert_eq!(value["lifecycle_stage"], json!("local_apply_completed"));
        assert_eq!(value["records"], json!([{"text": "x"}]));
        let round: ApplyResult = serde_json::from_value(value).unwrap();
        assert_eq!(round, rich);
        let failed = ApplyResult::rejection(
            "op",
            "experiences",
            &UpdateValue::set(json!(1)),
            vec!["nope".to_owned()],
        );
        assert!(!failed.ok());
        assert_eq!(failed.errors, ["nope"]);
        assert_eq!(ApplyResult::default().mode, UpdateMode::Replace);
        assert_eq!(ApplyResult::default().effect, UpdateEffect::State);
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
