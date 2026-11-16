//! Update contracts shared by the offline and online apply paths.
//!
//! Ports Python `openjiuwen.agent_evolving.types`: an update is keyed by
//! `(operator_id, target)` and valued by either a legacy raw payload
//! ([`Update::Raw`]) or the explicit [`UpdateValue`] contract (`mode`, `effect`,
//! `change_type`, `metadata`). [`normalize_update_value`] wraps legacy payloads
//! and gives `experiences` updates their append/pending-change shape.
//!
//! [`Updates`] is an insertion-ordered list of [`UpdateEntry`] records rather
//! than a `BTreeMap`: Python dicts preserve insertion order, and a JSON object
//! cannot key on a `(operator_id, target)` tuple, so the ordered list is both the
//! faithful and the serializable representation. Re-writing an existing key keeps
//! its original position, exactly like a Python dict.

use std::collections::BTreeMap;

use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;

/// The update target that stores skill-experience entries.
///
/// Python: `openjiuwen.agent_evolving.protocols.EXPERIENCES_TARGET`.
pub const EXPERIENCES_TARGET: &str = "experiences";

/// The change type recorded for [`EXPERIENCES_TARGET`] updates.
///
/// Python: `openjiuwen.agent_evolving.protocols.SKILL_EXPERIENCE_ENTRY`.
pub const SKILL_EXPERIENCE_ENTRY: &str = "skill_experience_entry";

/// How a target merges an update payload (Python `UpdateMode`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UpdateMode {
    /// Replace the parameter with the payload.
    #[default]
    Replace,
    /// Append the payload to the parameter.
    Append,
    /// Merge the payload into the parameter.
    Merge,
}

impl UpdateMode {
    /// The mode as its `snake_case` wire string.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Replace => "replace",
            Self::Append => "append",
            Self::Merge => "merge",
        }
    }
}

/// Where an update takes effect (Python `UpdateEffect`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UpdateEffect {
    /// The payload is applied to operator state immediately.
    #[default]
    State,
    /// The payload is queued as a pending change for a caller-owned approval step.
    PendingChange,
}

impl UpdateEffect {
    /// The effect as its `snake_case` wire string.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::State => "state",
            Self::PendingChange => "pending_change",
        }
    }
}

/// The operator domain a multi-dimensional update routes through.
///
/// Python splits the domains as `llm` (`LLMCall`), `tool` (`ToolCallOperator`)
/// and `memory` (`MemoryCallOperator`). The declaration order is the ordering
/// [`crate::updater::MultiDimUpdater`] applies domains in.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum UpdateDomain {
    /// LLM operators, the common case and therefore the default.
    #[default]
    Llm,
    /// Tool operators.
    Tool,
    /// Memory operators.
    Memory,
}

impl UpdateDomain {
    /// Every domain, in the order [`crate::updater::MultiDimUpdater`] applies them.
    pub const ALL: [Self; 3] = [Self::Llm, Self::Tool, Self::Memory];

    /// The domain as its `snake_case` wire string.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Llm => "llm",
            Self::Tool => "tool",
            Self::Memory => "memory",
        }
    }

    /// Parses the `snake_case` wire string back into a domain.
    #[must_use]
    pub fn parse(text: &str) -> Option<Self> {
        match text {
            "llm" => Some(Self::Llm),
            "tool" => Some(Self::Tool),
            "memory" => Some(Self::Memory),
            _ => None,
        }
    }
}

/// The lifecycle marker an apply result can carry.
///
/// Python types this as `Literal["local_apply_completed"] | None`; only the
/// richer operator families (`skill_call`, `context_evolve_call`) set it, so the
/// replace-only [`crate::updater::UpdateTarget`] seam leaves it `None`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LifecycleStage {
    /// The local apply finished; approval and persistence stay with the caller.
    LocalApplyCompleted,
}

impl LifecycleStage {
    /// The stage as its `snake_case` wire string.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        "local_apply_completed"
    }
}

/// The routing key of one update: `(operator_id, target)` (Python `UpdateKey`).
pub type UpdateKey = (String, String);

/// A structured update contract shared by the online and offline apply paths.
///
/// Python: `openjiuwen.agent_evolving.types.UpdateValue` (frozen dataclass).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct UpdateValue {
    /// The value the target receives.
    pub payload: Value,
    /// How the payload should be merged.
    pub mode: UpdateMode,
    /// Where the payload takes effect.
    pub effect: UpdateEffect,
    /// The change type, set for pending changes that need an approval category.
    pub change_type: Option<String>,
    /// Provenance carried to the target and to the [`ApplyResult`].
    pub metadata: BTreeMap<String, Value>,
}

impl UpdateValue {
    /// Creates a `replace` + `state` update carrying `payload`.
    #[must_use]
    pub fn new(payload: Value) -> Self {
        Self {
            payload,
            ..Self::default()
        }
    }
}

/// One value in an [`Updates`] map.
///
/// Python accepts both a raw payload (legacy) and an `UpdateValue` (structured)
/// in the same mapping; [`Update::Raw`] and [`Update::Structured`] keep that
/// distinction instead of guessing from the payload shape.
#[derive(Debug, Clone, PartialEq)]
pub enum Update {
    /// A structured update value.
    Structured(UpdateValue),
    /// A legacy raw payload, normalized on apply by [`normalize_update_value`].
    Raw(Value),
}

impl Update {
    /// Creates a raw update from any JSON value.
    #[must_use]
    pub fn raw(value: Value) -> Self {
        Self::Raw(value)
    }

    /// The payload the update carries, structured or not.
    #[must_use]
    pub fn payload(&self) -> &Value {
        match self {
            Self::Structured(value) => &value.payload,
            Self::Raw(value) => value,
        }
    }
}

impl From<Value> for Update {
    fn from(value: Value) -> Self {
        Self::Raw(value)
    }
}

impl From<UpdateValue> for Update {
    fn from(value: UpdateValue) -> Self {
        Self::Structured(value)
    }
}

impl Serialize for Update {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Structured(value) => value.serialize(serializer),
            Self::Raw(value) => value.serialize(serializer),
        }
    }
}

impl<'de> Deserialize<'de> for Update {
    /// Detects a structured value by its `payload` member.
    ///
    /// The `untagged` alternative would silently turn every JSON object into an
    /// empty [`UpdateValue`] because all of its fields are optional.
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        let structured = value
            .as_object()
            .is_some_and(|object| object.contains_key("payload"));
        if !structured {
            return Ok(Self::Raw(value));
        }
        UpdateValue::deserialize(value)
            .map(Self::Structured)
            .map_err(serde::de::Error::custom)
    }
}

/// One ordered entry of an [`Updates`] map.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UpdateEntry {
    /// The operator the update belongs to.
    pub operator_id: String,
    /// The operator target the update writes to.
    pub target: String,
    /// The update value; `None` reproduces Python's `None`-valued key.
    #[serde(default)]
    pub value: Option<Update>,
}

/// An insertion-ordered `(operator_id, target) -> value` map (Python `Updates`).
///
/// Iteration follows insertion order, re-writing an existing key keeps its
/// position, and a `None` value is kept rather than dropped so
/// [`crate::updater::execute_updates`] can report it.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Updates {
    entries: Vec<UpdateEntry>,
}

impl Updates {
    /// Creates an empty map.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of entries, including `None`-valued ones.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the map holds no entry.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Inserts `value` for `key`, returning the replaced value.
    ///
    /// Returns `None` when the key was absent and `Some(old)` when it existed;
    /// `old` is itself an `Option` because a key may map to a `None` value.
    pub fn insert(&mut self, key: UpdateKey, value: Option<Update>) -> Option<Option<Update>> {
        match self.entries.iter_mut().find(|entry| {
            entry.operator_id == key.0.as_str() && entry.target == key.1.as_str()
        }) {
            Some(entry) => Some(std::mem::replace(&mut entry.value, value)),
            None => {
                self.entries.push(UpdateEntry {
                    operator_id: key.0,
                    target: key.1,
                    value,
                });
                None
            }
        }
    }

    /// Inserts a structured or raw update value.
    pub fn insert_update(
        &mut self,
        operator_id: impl Into<String>,
        target: impl Into<String>,
        value: impl Into<Update>,
    ) -> Option<Option<Update>> {
        self.insert((operator_id.into(), target.into()), Some(value.into()))
    }

    /// Returns the value stored for `(operator_id, target)`.
    ///
    /// `None` means the key is absent, `Some(None)` means it maps to a `None`
    /// value, and `Some(Some(update))` carries the update itself.
    #[must_use]
    pub fn get(&self, operator_id: &str, target: &str) -> Option<Option<&Update>> {
        self.entries
            .iter()
            .find(|entry| entry.operator_id == operator_id && entry.target == target)
            .map(|entry| entry.value.as_ref())
    }

    /// Whether `(operator_id, target)` is present, whatever its value.
    #[must_use]
    pub fn contains_key(&self, operator_id: &str, target: &str) -> bool {
        self.get(operator_id, target).is_some()
    }

    /// Iterates `(key, value)` pairs in insertion order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &str, &Option<Update>)> {
        self.entries.iter().map(|entry| {
            (
                entry.operator_id.as_str(),
                entry.target.as_str(),
                &entry.value,
            )
        })
    }

    /// The ordered entries.
    #[must_use]
    pub fn entries(&self) -> &[UpdateEntry] {
        &self.entries
    }

    /// Consumes the map and returns the ordered entries.
    #[must_use]
    pub fn into_entries(self) -> Vec<UpdateEntry> {
        self.entries
    }

    /// Appends every entry of `other`, rejecting a key this map already holds.
    ///
    /// Merging is the multi-domain composition step, so a duplicate key is a real
    /// routing conflict rather than something to overwrite silently.
    pub fn merge(&mut self, other: Self) -> Result<()> {
        for entry in other.entries {
            if self.contains_key(&entry.operator_id, &entry.target) {
                return Err(Error::validation(format!(
                    "Updates.merge: duplicate update for operator {:?} target {:?}",
                    entry.operator_id, entry.target
                )));
            }
            self.insert((entry.operator_id, entry.target), entry.value);
        }
        Ok(())
    }
}

impl FromIterator<(UpdateKey, Option<Update>)> for Updates {
    fn from_iter<I: IntoIterator<Item = (UpdateKey, Option<Update>)>>(iter: I) -> Self {
        let mut updates = Self::new();
        for (key, value) in iter {
            updates.insert(key, value);
        }
        updates
    }
}

impl Serialize for Updates {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.entries.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Updates {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let entries = Vec::<UpdateEntry>::deserialize(deserializer)?;
        let mut updates = Self::new();
        for entry in entries {
            updates.insert((entry.operator_id, entry.target), entry.value);
        }
        Ok(updates)
    }
}

/// Wraps a value into the structured contract (Python `normalize_update_value`).
///
/// A [`Update::Structured`] value passes through unchanged. A raw payload
/// targeting [`EXPERIENCES_TARGET`] becomes `append` + `pending_change` with
/// `change_type = "skill_experience_entry"` (mirrored into `metadata`); every
/// other raw payload becomes `replace` + `state` without a change type.
#[must_use]
pub fn normalize_update_value(value: &Update, target: &str) -> UpdateValue {
    match value {
        Update::Structured(value) => value.clone(),
        Update::Raw(payload) if target == EXPERIENCES_TARGET => {
            let mut metadata = BTreeMap::new();
            metadata.insert(
                "change_type".to_owned(),
                Value::String(SKILL_EXPERIENCE_ENTRY.to_owned()),
            );
            UpdateValue {
                payload: payload.clone(),
                mode: UpdateMode::Append,
                effect: UpdateEffect::PendingChange,
                change_type: Some(SKILL_EXPERIENCE_ENTRY.to_owned()),
                metadata,
            }
        }
        Update::Raw(payload) => UpdateValue::new(payload.clone()),
    }
}

/// Normalizes a whole map, skipping `None`-valued keys (Python `normalize_updates`).
///
/// The returned vector keeps the map's insertion order.
#[must_use]
pub fn normalize_updates(updates: &Updates) -> Vec<(UpdateKey, UpdateValue)> {
    updates
        .iter()
        .filter_map(|(operator_id, target, value)| {
            value.as_ref().map(|value| {
                (
                    (operator_id.to_owned(), target.to_owned()),
                    normalize_update_value(value, target),
                )
            })
        })
        .collect()
}

/// The result of applying one normalized update to one evolution target.
///
/// Python: `openjiuwen.agent_evolving.types.ApplyResult`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ApplyResult {
    /// The operator the update was addressed to.
    pub operator_id: String,
    /// The operator target the update was addressed to.
    pub target: String,
    /// Whether the target accepted the payload.
    pub applied: bool,
    /// How the payload was meant to be merged.
    pub mode: UpdateMode,
    /// Where the payload was meant to take effect.
    pub effect: UpdateEffect,
    /// The payload that was applied (or rejected).
    pub value: Value,
    /// Records the target produced while applying.
    pub records: Vec<Value>,
    /// The change type carried by a pending change.
    pub change_type: Option<String>,
    /// The lifecycle marker, set only by targets that report one.
    pub lifecycle_stage: Option<LifecycleStage>,
    /// Identifier of the pending change the target queued, when it queued one.
    pub pending_change_id: Option<String>,
    /// Every failure that prevented the update from applying.
    pub errors: Vec<String>,
    /// Provenance copied from the [`UpdateValue`].
    pub metadata: BTreeMap<String, Value>,
}

impl ApplyResult {
    /// Whether the update applied without any error, i.e. Python's `ok` property.
    #[must_use]
    pub fn ok(&self) -> bool {
        self.applied && self.errors.is_empty()
    }

    /// Builds a rejected result carrying `error` for `(operator_id, target)`.
    #[must_use]
    pub fn rejected(
        operator_id: impl Into<String>,
        target: impl Into<String>,
        error: impl Into<String>,
    ) -> Self {
        Self {
            operator_id: operator_id.into(),
            target: target.into(),
            errors: vec![error.into()],
            ..Self::default()
        }
    }
}

/// Aggregate counts over a batch of [`ApplyResult`]s.
///
/// Python returns the same three numbers as a `dict` from
/// [`crate::updater::summarize_apply_results`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ApplySummary {
    /// Number of results.
    pub total: usize,
    /// Number of results with `applied == true`.
    pub applied: usize,
    /// Number of results with `applied == false`.
    pub failed: usize,
}

impl ApplySummary {
    /// Whether every result applied.
    #[must_use]
    pub fn is_ok(&self) -> bool {
        self.failed == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn legacy_payload_normalizes_to_replace_state() {
        let normalized = normalize_update_value(&Update::raw(json!({"description": "x"})), "body");
        assert_eq!(normalized.payload, json!({"description": "x"}));
        assert_eq!(normalized.mode, UpdateMode::Replace);
        assert_eq!(normalized.effect, UpdateEffect::State);
        assert_eq!(normalized.change_type, None);
        assert!(normalized.metadata.is_empty());
    }

    #[test]
    fn experiences_target_normalizes_to_append_pending_change() {
        let normalized = normalize_update_value(&Update::raw(json!(["entry"])), EXPERIENCES_TARGET);
        assert_eq!(normalized.payload, json!(["entry"]));
        assert_eq!(normalized.mode, UpdateMode::Append);
        assert_eq!(normalized.effect, UpdateEffect::PendingChange);
        assert_eq!(
            normalized.change_type.as_deref(),
            Some(SKILL_EXPERIENCE_ENTRY)
        );
        assert_eq!(
            normalized.metadata.get("change_type"),
            Some(&Value::String(SKILL_EXPERIENCE_ENTRY.to_owned()))
        );
    }

    #[test]
    fn structured_values_pass_through_unchanged() {
        let value = UpdateValue {
            payload: json!(1),
            mode: UpdateMode::Merge,
            effect: UpdateEffect::PendingChange,
            change_type: Some("custom".to_owned()),
            metadata: BTreeMap::new(),
        };
        let normalized = normalize_update_value(&Update::Structured(value.clone()), "experiences");
        assert_eq!(normalized, value);
    }

    #[test]
    fn updates_keep_insertion_order_and_replace_in_place() {
        let mut updates = Updates::new();
        updates.insert_update("op-b", "body", json!(1));
        updates.insert_update("op-a", "body", json!(2));
        updates.insert_update("op-b", "body", json!(3));
        let keys: Vec<(&str, &str)> = updates.iter().map(|(op, target, _)| (op, target)).collect();
        assert_eq!(keys, [("op-b", "body"), ("op-a", "body")]);
        assert_eq!(
            updates.get("op-b", "body").and_then(|u| u).map(Update::payload),
            Some(&json!(3))
        );
        assert_eq!(updates.len(), 2);
    }

    #[test]
    fn none_valued_keys_are_preserved() {
        let mut updates = Updates::new();
        updates.insert(("op".to_owned(), "body".to_owned()), None);
        assert_eq!(updates.len(), 1);
        assert!(updates.contains_key("op", "body"));
        assert!(matches!(updates.get("op", "body"), Some(None)));
        assert!(normalize_updates(&updates).is_empty());
    }

    #[test]
    fn merge_rejects_duplicate_keys() {
        let mut left = Updates::new();
        left.insert_update("op", "body", json!(1));
        let mut right = Updates::new();
        right.insert_update("op", "body", json!(2));
        let error = left.merge(right).expect_err("duplicate key");
        assert!(error.to_string().contains("duplicate update"));
        assert_eq!(
            left.get("op", "body").and_then(|u| u).map(Update::payload),
            Some(&json!(1))
        );
    }

    #[test]
    fn updates_round_trip_through_json() {
        let mut updates = Updates::new();
        updates.insert_update("op", "body", json!({"text": "hello"}));
        updates.insert(
            ("op".to_owned(), "experiences".to_owned()),
            Some(Update::Structured(UpdateValue {
                payload: json!([1, 2]),
                mode: UpdateMode::Append,
                effect: UpdateEffect::PendingChange,
                change_type: Some(SKILL_EXPERIENCE_ENTRY.to_owned()),
                metadata: BTreeMap::new(),
            })),
        );
        let encoded = serde_json::to_value(&updates).expect("serialize");
        assert_eq!(
            encoded,
            json!([
                {"operator_id": "op", "target": "body", "value": {"text": "hello"}},
                {
                    "operator_id": "op",
                    "target": "experiences",
                    "value": {
                        "payload": [1, 2],
                        "mode": "append",
                        "effect": "pending_change",
                        "change_type": SKILL_EXPERIENCE_ENTRY,
                        "metadata": {},
                    },
                },
            ])
        );
        let decoded: Updates = serde_json::from_value(encoded).expect("deserialize");
        assert_eq!(decoded, updates);
    }

    #[test]
    fn update_detection_distinguishes_structured_from_raw() {
        let raw: Update = serde_json::from_value(json!({"text": "x"})).expect("raw");
        assert!(matches!(raw, Update::Raw(_)));
        let structured: Update =
            serde_json::from_value(json!({"payload": "x"})).expect("structured");
        match structured {
            Update::Structured(value) => assert_eq!(value.payload, json!("x")),
            Update::Raw(_) => panic!("expected a structured update"),
        }
    }

    #[test]
    fn apply_result_ok_requires_applied_and_error_free() {
        let mut result = ApplyResult::default();
        assert!(!result.ok());
        result.applied = true;
        assert!(result.ok());
        result.errors.push("boom".to_owned());
        assert!(!result.ok());
        let rejected = ApplyResult::rejected("op", "body", "nope");
        assert!(!rejected.ok());
        assert_eq!(rejected.errors, ["nope"]);
    }

    #[test]
    fn domains_round_trip_through_their_wire_strings() {
        for domain in UpdateDomain::ALL {
            assert_eq!(UpdateDomain::parse(domain.as_str()), Some(domain));
        }
        assert_eq!(UpdateDomain::parse("unknown"), None);
        assert_eq!(UpdateDomain::ALL, [
            UpdateDomain::Llm,
            UpdateDomain::Tool,
            UpdateDomain::Memory
        ]);
        assert_eq!(UpdateMode::default().as_str(), "replace");
        assert_eq!(UpdateEffect::default().as_str(), "state");
        assert_eq!(LifecycleStage::LocalApplyCompleted.as_str(), "local_apply_completed");
    }
}
