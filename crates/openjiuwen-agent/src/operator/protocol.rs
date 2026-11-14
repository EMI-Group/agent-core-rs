//! Evolution protocol literals and update normalisation.
//!
//! Ports `openjiuwen/agent_evolving/protocols.py` (the literals shared by the
//! online and offline apply paths) and the normalisation/summary helpers of
//! `openjiuwen/agent_evolving/update_execution.py`.
//!
//! These values are the seam between an optimiser's legacy payload map and the
//! structured [`UpdateValue`] this module produces: `experiences` payloads
//! become append/pending-change updates tagged `skill_experience_entry`, while
//! every other target keeps the replace/state default.
use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::base::{ApplyResult, UpdateEffect, UpdateMode, UpdateValue};

/// Wire name of [`UpdateMode::Replace`].
pub const REPLACE_MODE: &str = "replace";
/// Wire name of [`UpdateMode::Append`].
pub const APPEND_MODE: &str = "append";
/// Wire name of [`UpdateMode::Merge`].
pub const MERGE_MODE: &str = "merge";
/// Wire name of [`UpdateEffect::State`].
pub const STATE_EFFECT: &str = "state";
/// Wire name of [`UpdateEffect::PendingChange`].
pub const PENDING_CHANGE_EFFECT: &str = "pending_change";
/// Lifecycle stage a local preview reaches before approval.
pub const LOCAL_APPLY_COMPLETED: &str = "local_apply_completed";
/// Target of skill-experience records.
pub const EXPERIENCES_TARGET: &str = "experiences";
/// Record kind of one skill-experience entry.
pub const SKILL_EXPERIENCE_ENTRY: &str = "skill_experience_entry";
/// Record kind of one generic experience entry.
pub const EXPERIENCE_ENTRY: &str = "experience_entry";
/// Target of task-memory records.
pub const TASK_MEMORY_TARGET: &str = "task_memory";
/// Record kind of one task-memory entry.
pub const TASK_MEMORY_ENTRY: &str = "task_memory_entry";

/// `(operator_id, target)` key of one update in a batch (`UpdateKey` in
/// Python).
pub type UpdateKey = (String, String);

/// Builds the key of `operator_id`'s `target` update.
#[must_use]
pub fn update_key(operator_id: impl Into<String>, target: impl Into<String>) -> UpdateKey {
    (operator_id.into(), target.into())
}

/// Wraps a legacy update payload into the structured [`UpdateValue`] contract.
///
/// - A JSON object that already carries a parseable `mode` and a `payload`
///   round-trips as a structured update (Python passes an `UpdateValue`
///   instance instead, which JSON cannot distinguish, so the shape is used).
/// - An `experiences` target becomes append + pending change, tagged
///   [`SKILL_EXPERIENCE_ENTRY`], with the same tag mirrored into metadata.
/// - Every other value becomes a replace/state update.
#[must_use]
pub fn normalize_update_value(value: impl Into<Value>, target: Option<&str>) -> UpdateValue {
    let value = value.into();
    if is_structured_update(&value) {
        if let Ok(update) = serde_json::from_value::<UpdateValue>(value.clone()) {
            return update;
        }
    }
    if target == Some(EXPERIENCES_TARGET) {
        return UpdateValue {
            payload: value,
            mode: UpdateMode::Append,
            effect: UpdateEffect::PendingChange,
            change_type: Some(SKILL_EXPERIENCE_ENTRY.to_owned()),
            metadata: BTreeMap::from([(
                "change_type".to_owned(),
                Value::String(SKILL_EXPERIENCE_ENTRY.to_owned()),
            )]),
        };
    }
    UpdateValue {
        payload: value,
        ..UpdateValue::default()
    }
}

/// Normalises a mixed batch of legacy and structured updates, preserving order.
#[must_use]
pub fn normalize_updates<I>(updates: I) -> Vec<(UpdateKey, UpdateValue)>
where
    I: IntoIterator<Item = (UpdateKey, Value)>,
{
    updates
        .into_iter()
        .map(|(key, value)| {
            let normalized = normalize_update_value(value, Some(&key.1));
            (key, normalized)
        })
        .collect()
}

/// Aggregate counts over a batch of apply results (Python returns a dict with
/// the same three keys).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ApplySummary {
    /// Number of results.
    pub total: usize,
    /// Results that applied.
    pub applied: usize,
    /// Results that did not apply.
    pub failed: usize,
}

impl ApplySummary {
    /// Whether at least one result was produced.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.total == 0
    }
}

/// Counts a batch of applied/failed results.
#[must_use]
pub fn summarize_apply_results(results: &[ApplyResult]) -> ApplySummary {
    let applied = results.iter().filter(|result| result.applied).count();
    ApplySummary {
        total: results.len(),
        applied,
        failed: results.len() - applied,
    }
}

/// Whether `value` looks like a serialized [`UpdateValue`].
fn is_structured_update(value: &Value) -> bool {
    let Some(object) = value.as_object() else {
        return false;
    };
    let structured_mode = object
        .get("mode")
        .and_then(Value::as_str)
        .is_some_and(|mode| UpdateMode::parse(mode).is_some());
    structured_mode && object.contains_key("payload")
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
    fn legacy_values_normalize_by_target() {
        let plain = normalize_update_value(json!("be terse"), Some("system_prompt"));
        assert_eq!(plain.mode, UpdateMode::Replace);
        assert_eq!(plain.effect, UpdateEffect::State);
        assert_eq!(plain.payload, json!("be terse"));

        let experience =
            normalize_update_value(json!({"text": "always retry"}), Some("experiences"));
        assert_eq!(experience.mode, UpdateMode::Append);
        assert_eq!(experience.effect, UpdateEffect::PendingChange);
        assert_eq!(
            experience.change_type.as_deref(),
            Some(SKILL_EXPERIENCE_ENTRY)
        );
        assert_eq!(
            experience.metadata.get("change_type"),
            Some(&json!(SKILL_EXPERIENCE_ENTRY))
        );
        assert_eq!(experience.payload, json!({"text": "always retry"}));

        let no_target = normalize_update_value(json!(1), None);
        assert_eq!(no_target, UpdateValue::set(json!(1)));
    }

    #[test]
    fn structured_values_pass_through() {
        let structured = json!({
            "payload": ["a"],
            "mode": "merge",
            "effect": "pending_change",
            "change_type": "experience_entry",
            "metadata": {"source": "offline"}
        });
        let update = normalize_update_value(structured, Some("other"));
        assert_eq!(update.mode, UpdateMode::Merge);
        assert_eq!(update.effect, UpdateEffect::PendingChange);
        assert_eq!(update.change_type.as_deref(), Some(EXPERIENCE_ENTRY));
        assert_eq!(update.metadata.get("source"), Some(&json!("offline")));
    }

    #[test]
    fn batches_keep_order_and_keys() {
        let updates = normalize_updates(vec![
            (
                update_key("react_llm", "system_prompt"),
                json!({"mode": "set", "payload": "terse"}),
            ),
            (
                update_key("skill_experience_a", "experiences"),
                json!([{"text": "x"}]),
            ),
        ]);
        assert_eq!(updates.len(), 2);
        assert_eq!(
            updates[0].0,
            ("react_llm".to_owned(), "system_prompt".to_owned())
        );
        assert_eq!(updates[0].1, UpdateValue::set(json!("terse")));
        assert_eq!(updates[1].1.effect, UpdateEffect::PendingChange);
    }

    #[test]
    fn summaries_count_applied_and_failed() {
        let results = vec![
            ApplyResult::applied("a"),
            ApplyResult::skipped("b", "nope"),
            ApplyResult::applied("c"),
        ];
        assert_eq!(
            summarize_apply_results(&results),
            ApplySummary {
                total: 3,
                applied: 2,
                failed: 1
            }
        );
        assert!(summarize_apply_results(&[]).is_empty());
        assert_eq!(
            serde_json::to_value(summarize_apply_results(&results)).unwrap(),
            json!({"total": 3, "applied": 2, "failed": 1})
        );
    }

    #[test]
    fn protocol_literals_match_python() {
        assert_eq!(REPLACE_MODE, UpdateMode::Replace.as_str());
        assert_eq!(APPEND_MODE, UpdateMode::Append.as_str());
        assert_eq!(MERGE_MODE, UpdateMode::Merge.as_str());
        assert_eq!(STATE_EFFECT, UpdateEffect::State.as_str());
        assert_eq!(PENDING_CHANGE_EFFECT, UpdateEffect::PendingChange.as_str());
        assert_eq!(LOCAL_APPLY_COMPLETED, "local_apply_completed");
        assert_eq!(EXPERIENCES_TARGET, "experiences");
        assert_eq!(SKILL_EXPERIENCE_ENTRY, "skill_experience_entry");
        assert_eq!(EXPERIENCE_ENTRY, "experience_entry");
        assert_eq!(TASK_MEMORY_TARGET, "task_memory");
        assert_eq!(TASK_MEMORY_ENTRY, "task_memory_entry");
    }
}
