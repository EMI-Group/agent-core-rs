//! Pure JSON diffing used by the session version history.
//!
//! [`diff_values`] walks two JSON documents and reports every added, removed,
//! or changed leaf; [`diff_snapshots`] applies it to two
//! [`SessionSnapshot`]s using the persisted snapshot shape, so a difference in
//! the session kind shows up as `kind`, in global state as `global.<path>`, in
//! component state as `components.<id>.<path>`, in the trace position as
//! `trace.<field>`, and in metadata as `metadata.<key>`.
//!
//! Paths use the crate's nested-path syntax (`a.b[0].c`); the root is `""`.
//! Object keys are visited in sorted order and arrays by ascending index, so the
//! output is deterministic. Arrays are diffed element-wise; a longer tail is
//! reported as one `Added` per index (and a shorter tail as one `Removed`).
use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::session::{SessionSnapshot, SessionType};
use crate::tracer::TracePosition;

/// What happened to a JSON path between two documents.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChangeKind {
    /// The path exists only on the right-hand side.
    Added,
    /// The path exists only on the left-hand side.
    Removed,
    /// The path exists on both sides with different values.
    #[default]
    Changed,
}

/// One difference between two JSON documents.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct JsonChange {
    /// The nested JSON path (`a.b[0].c`; `""` for the root).
    pub path: String,
    /// Whether the path was added, removed, or changed.
    pub kind: ChangeKind,
    /// The left-hand value, absent for [`ChangeKind::Added`].
    pub before: Option<Value>,
    /// The right-hand value, absent for [`ChangeKind::Removed`].
    pub after: Option<Value>,
}

/// Diffs two JSON documents and returns the leaf-level changes.
///
/// Objects recurse by key (sorted), arrays recurse by index, and any other pair
/// of unequal values yields one [`ChangeKind::Changed`] entry.
#[must_use]
pub fn diff_values(left: &Value, right: &Value) -> Vec<JsonChange> {
    let mut changes = Vec::new();
    diff_into(left, right, "", &mut changes);
    changes
}

/// Diffs two session snapshots using the persisted JSON shape.
///
/// See the module documentation for the path prefixes (`kind`, `global.*`,
/// `components.<id>.*`, `trace.*`, `metadata.*`).
#[must_use]
pub fn diff_snapshots(left: &SessionSnapshot, right: &SessionSnapshot) -> Vec<JsonChange> {
    diff_values(&snapshot_value(left), &snapshot_value(right))
}

fn diff_into(left: &Value, right: &Value, path: &str, changes: &mut Vec<JsonChange>) {
    if left == right {
        return;
    }
    match (left, right) {
        (Value::Object(left_map), Value::Object(right_map)) => {
            let keys: BTreeSet<&String> = left_map.keys().chain(right_map.keys()).collect();
            for key in keys {
                let child = child_path(path, key);
                match (left_map.get(key), right_map.get(key)) {
                    (Some(left), Some(right)) => diff_into(left, right, &child, changes),
                    (Some(left), None) => changes.push(JsonChange {
                        path: child,
                        kind: ChangeKind::Removed,
                        before: Some(left.clone()),
                        after: None,
                    }),
                    (None, Some(right)) => changes.push(JsonChange {
                        path: child,
                        kind: ChangeKind::Added,
                        before: None,
                        after: Some(right.clone()),
                    }),
                    (None, None) => {}
                }
            }
        }
        (Value::Array(left_items), Value::Array(right_items)) => {
            for index in 0..left_items.len().max(right_items.len()) {
                let child = index_path(path, index);
                match (left_items.get(index), right_items.get(index)) {
                    (Some(left), Some(right)) => diff_into(left, right, &child, changes),
                    (Some(left), None) => changes.push(JsonChange {
                        path: child,
                        kind: ChangeKind::Removed,
                        before: Some(left.clone()),
                        after: None,
                    }),
                    (None, Some(right)) => changes.push(JsonChange {
                        path: child,
                        kind: ChangeKind::Added,
                        before: None,
                        after: Some(right.clone()),
                    }),
                    (None, None) => {}
                }
            }
        }
        _ => changes.push(JsonChange {
            path: path.to_string(),
            kind: ChangeKind::Changed,
            before: Some(left.clone()),
            after: Some(right.clone()),
        }),
    }
}

fn child_path(base: &str, key: &str) -> String {
    if base.is_empty() {
        key.to_string()
    } else {
        format!("{base}.{key}")
    }
}

fn index_path(base: &str, index: usize) -> String {
    format!("{base}[{index}]")
}

/// Projects a snapshot onto the JSON shape persisted in `state.data`.
///
/// The conversion is manual so this module needs no fallible serialization
/// step; the keys mirror [`SessionSnapshot`]'s serde field names.
fn snapshot_value(snapshot: &SessionSnapshot) -> Value {
    let mut object = Map::new();
    object.insert(
        "session_id".to_string(),
        Value::String(snapshot.session_id.clone()),
    );
    object.insert(
        "kind".to_string(),
        Value::String(session_type_name(snapshot.kind).to_string()),
    );
    object.insert("global".to_string(), snapshot.global.clone());
    object.insert(
        "components".to_string(),
        Value::Object(
            snapshot
                .components
                .iter()
                .map(|(id, value)| (id.clone(), value.clone()))
                .collect(),
        ),
    );
    object.insert("trace".to_string(), trace_value(snapshot.trace.as_ref()));
    object.insert(
        "metadata".to_string(),
        Value::Object(
            snapshot
                .metadata
                .iter()
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect(),
        ),
    );
    Value::Object(object)
}

fn trace_value(position: Option<&TracePosition>) -> Value {
    match position {
        None => Value::Null,
        Some(position) => {
            let mut object = Map::new();
            object.insert(
                "trace_id".to_string(),
                Value::String(position.trace_id.clone()),
            );
            object.insert(
                "last_invoke_id".to_string(),
                position
                    .last_invoke_id
                    .clone()
                    .map_or(Value::Null, Value::String),
            );
            object.insert(
                "frame_count".to_string(),
                Value::Number(position.frame_count.into()),
            );
            Value::Object(object)
        }
    }
}

fn session_type_name(kind: SessionType) -> &'static str {
    match kind {
        SessionType::Workflow => "workflow",
        SessionType::Agent => "agent",
        SessionType::AgentTeam => "agent_team",
        SessionType::Node => "node",
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use serde_json::json;

    use super::{diff_values, ChangeKind};

    fn find<'a>(changes: &'a [super::JsonChange], path: &str) -> &'a super::JsonChange {
        changes
            .iter()
            .find(|change| change.path == path)
            .unwrap_or_else(|| panic!("no change reported at {path:?}"))
    }

    #[test]
    fn change_kinds_follow_types() {
        let changes = diff_values(&json!({"a": 1, "b": 2}), &json!({"a": 1, "c": 3}));
        assert_eq!(changes.len(), 2);
        assert_eq!(find(&changes, "b").kind, ChangeKind::Removed);
        assert_eq!(find(&changes, "b").after, None);
        assert_eq!(find(&changes, "c").kind, ChangeKind::Added);
        assert_eq!(find(&changes, "c").before, None);
    }

    #[test]
    fn nested_objects_and_arrays_use_path_syntax() {
        let left = json!({"user": {"age": 36}, "tags": ["a", "b"]});
        let right = json!({"user": {"age": 37}, "tags": ["a", "c", "d"]});
        let changes = diff_values(&left, &right);
        assert_eq!(find(&changes, "user.age").kind, ChangeKind::Changed);
        assert_eq!(find(&changes, "tags[1]").kind, ChangeKind::Changed);
        assert_eq!(find(&changes, "tags[2]").kind, ChangeKind::Added);
        assert_eq!(find(&changes, "tags[2]").after, Some(json!("d")));
    }

    #[test]
    fn array_shrink_reports_removed_tail_elements() {
        let changes = diff_values(&json!([1, 2, 3]), &json!([1, 2]));
        assert_eq!(changes.len(), 1);
        assert_eq!(find(&changes, "[2]").kind, ChangeKind::Removed);
    }

    #[test]
    fn equal_documents_report_nothing() {
        assert_eq!(
            diff_values(&json!({"a": [1, 2]}), &json!({"a": [1, 2]})).len(),
            0
        );
    }

    #[test]
    fn type_changes_are_changed_at_the_path() {
        let changes = diff_values(&json!({"a": {"b": 1}}), &json!({"a": 1}));
        assert_eq!(find(&changes, "a").kind, ChangeKind::Changed);
        assert_eq!(find(&changes, "a").before, Some(json!({"b": 1})));
        assert_eq!(find(&changes, "a").after, Some(json!(1)));
    }
}
