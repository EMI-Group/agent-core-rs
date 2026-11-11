//! Graph state: the JSON object passed between nodes and merged across
//! parallel branches.
//!
//! [`GraphState`] is a thin, transparent wrapper over [`serde_json::Map`].
//! Path access and merging delegate to [`openjiuwen_common::utils::json`], so a
//! graph state has exactly the session state's path (`a.b[0].c`) and merge
//! semantics: object-vs-object merges recurse, arrays and scalars replace.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::utils::json;

/// The JSON state a graph run threads through its nodes.
///
/// Nodes receive a mutable clone of the state and the engine merges their
/// result back in declaration order, so parallel branches never share mutable
/// state.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct GraphState(pub Map<String, Value>);

impl GraphState {
    /// Creates an empty state.
    #[must_use]
    pub fn new() -> Self {
        Self(Map::new())
    }

    /// Builds a state from a JSON value: an object is used as-is, `null` yields
    /// an empty state.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::ErrorKind::Validation`] when `value`
    /// is neither an object nor `null`.
    pub fn from_value(value: Value) -> Result<Self> {
        match value {
            Value::Object(map) => Ok(Self(map)),
            Value::Null => Ok(Self::new()),
            other => Err(Error::validation(format!(
                "graph state must be a JSON object, got {}",
                type_name(&other)
            ))),
        }
    }

    /// Wraps an existing map.
    #[must_use]
    pub fn from_map(map: Map<String, Value>) -> Self {
        Self(map)
    }

    /// Borrows the state as a JSON object value.
    #[must_use]
    pub fn as_value(&self) -> Value {
        Value::Object(self.0.clone())
    }

    /// Consumes the state and returns its JSON object value.
    #[must_use]
    pub fn into_value(self) -> Value {
        Value::Object(self.0)
    }

    /// Returns the value under `key`.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.0.get(key)
    }

    /// Reads a dotted/indexed path (`a.b[0].c`).
    ///
    /// An empty path addresses the state root, which is a map rather than a
    /// stored [`Value`], so it returns `None`.
    #[must_use]
    pub fn get_path(&self, path: &str) -> Option<&Value> {
        if path.is_empty() {
            return None;
        }
        let (key, rest) = split_first_segment(path);
        let value = self.0.get(key)?;
        if rest.is_empty() {
            Some(value)
        } else {
            json::get_path(value, rest)
        }
    }

    /// Sets `key` to `value`, replacing any previous value.
    pub fn set(&mut self, key: impl Into<String>, value: Value) {
        self.0.insert(key.into(), value);
    }

    /// Writes `value` at a dotted/indexed path, creating intermediate objects.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::ErrorKind::Validation`] when `path`
    /// is empty, malformed, or addresses something other than an object field.
    pub fn set_path(&mut self, path: &str, value: Value) -> Result<()> {
        if path.is_empty() || path.starts_with('[') {
            return Err(Error::validation(format!(
                "graph state path `{path}` must address an object field"
            )));
        }
        let mut root = Value::Object(std::mem::take(&mut self.0));
        let outcome = json::set_path(&mut root, path, value);
        match root {
            Value::Object(map) => {
                self.0 = map;
                outcome
            }
            other => {
                self.0 = Map::new();
                Err(Error::validation(format!(
                    "graph state path `{path}` must address an object field, got {}",
                    type_name(&other)
                )))
            }
        }
    }

    /// Removes `key` and returns its value.
    pub fn remove(&mut self, key: &str) -> Option<Value> {
        self.0.remove(key)
    }

    /// Reports whether `key` is present.
    #[must_use]
    pub fn contains_key(&self, key: &str) -> bool {
        self.0.contains_key(key)
    }

    /// Iterates the state's keys in insertion order.
    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.0.keys()
    }

    /// Returns the number of top-level keys.
    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Reports whether the state has no keys.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Deep-merges `other` into this state; `other` wins on conflicts.
    pub fn merge(&mut self, other: Self) {
        let mut root = Value::Object(std::mem::take(&mut self.0));
        json::merge(&mut root, Value::Object(other.0));
        if let Value::Object(map) = root {
            self.0 = map;
        }
    }

    /// Applies one node's state back onto the run state as a delta relative to
    /// `base`, the state the node was given when the superstep started.
    ///
    /// Every top-level key of `node_state` whose value differs from (or is
    /// absent in) `base` is deep-merged into `self`, so nested object writes
    /// still merge deeply and two parallel nodes touching different fields of
    /// one nested object both survive. Every top-level key present in `base` but
    /// absent from `node_state` is then removed from `self`: the node deleted
    /// it.
    ///
    /// Keys the node never saw, such as keys added by an earlier node in the
    /// same superstep, are left alone because the node could not have deleted
    /// them. Only top-level keys are diffed; a nested removal (removing `b`
    /// inside top-level `a`) does not propagate. Such a nested delete shows up
    /// as a changed top-level value and is re-merged from `base`.
    #[allow(clippy::needless_pass_by_value)] // by-value matches `merge` and lets callers hand over the node result
    pub fn merge_node_delta(&mut self, base: &Self, node_state: Self) {
        let mut delta = Map::new();
        for (key, value) in &node_state.0 {
            if base.0.get(key) != Some(value) {
                delta.insert(key.clone(), value.clone());
            }
        }
        if !delta.is_empty() {
            self.merge(Self(delta));
        }
        for key in base.0.keys() {
            if !node_state.0.contains_key(key) {
                self.0.remove(key);
            }
        }
    }
}

impl From<Map<String, Value>> for GraphState {
    fn from(map: Map<String, Value>) -> Self {
        Self(map)
    }
}

impl From<GraphState> for Value {
    fn from(state: GraphState) -> Self {
        state.into_value()
    }
}

/// Splits `path` into its first object key and the remainder, which either
/// starts with `.` (already stripped) or `[`.
fn split_first_segment(path: &str) -> (&str, &str) {
    match path.find(['.', '[']) {
        Some(index) => {
            let key = &path[..index];
            let rest = &path[index..];
            (key, rest.strip_prefix('.').unwrap_or(rest))
        }
        None => (path, ""),
    }
}

/// The JSON type name used in validation messages.
fn type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn from_value_accepts_objects_and_null() {
        let state = GraphState::from_value(json!({"a": 1})).unwrap();
        assert_eq!(state.get("a"), Some(&json!(1)));
        let empty = GraphState::from_value(Value::Null).unwrap();
        assert!(empty.is_empty());
    }

    #[test]
    fn from_value_rejects_scalars_and_arrays() {
        for value in [json!(3), json!("x"), json!([1, 2]), json!(true)] {
            let error = GraphState::from_value(value).unwrap_err();
            assert_eq!(
                error.kind(),
                openjiuwen_common::error::ErrorKind::Validation
            );
        }
    }

    #[test]
    fn set_get_remove_and_contains() {
        let mut state = GraphState::new();
        state.set("a", json!(1));
        assert!(state.contains_key("a"));
        assert_eq!(state.get("a"), Some(&json!(1)));
        assert_eq!(state.remove("a"), Some(json!(1)));
        assert!(!state.contains_key("a"));
        assert!(state.is_empty());
        assert_eq!(state.len(), 0);
    }

    #[test]
    fn get_path_reads_nested_objects_and_indices() {
        let state = GraphState::from_value(json!({"a": {"b": [{"c": 7}]}})).unwrap();
        assert_eq!(state.get_path("a.b[0].c"), Some(&json!(7)));
        assert_eq!(state.get_path("a"), Some(&json!({"b": [{"c": 7}]})));
        assert_eq!(state.get_path("missing"), None);
        assert_eq!(state.get_path(""), None);
        assert_eq!(state.get_path("a..b"), None);
        assert_eq!(state.get_path("[0]"), None);
    }

    #[test]
    fn set_path_creates_intermediates() {
        let mut state = GraphState::new();
        state.set_path("a.b[1].c", json!(7)).unwrap();
        assert_eq!(state.as_value(), json!({"a": {"b": [null, {"c": 7}]}}));
    }

    #[test]
    fn set_path_rejects_malformed_paths() {
        let mut state = GraphState::from_value(json!({"a": 1})).unwrap();
        assert!(state.set_path("", json!(2)).is_err());
        assert!(state.set_path("a..b", json!(2)).is_err());
        assert!(state.set_path("[0]", json!(2)).is_err());
        assert_eq!(state.as_value(), json!({"a": 1}));
    }

    #[test]
    fn merge_is_deep_and_overlay_wins() {
        let mut state = GraphState::from_value(json!({"a": {"b": 1, "c": 2}, "d": 3})).unwrap();
        state.merge(GraphState::from_value(json!({"a": {"c": 20}, "d": 4})).unwrap());
        assert_eq!(state.as_value(), json!({"a": {"b": 1, "c": 20}, "d": 4}));
    }

    #[test]
    fn merge_replaces_arrays() {
        let mut state = GraphState::from_value(json!({"a": [1, 2]})).unwrap();
        state.merge(GraphState::from_value(json!({"a": [3]})).unwrap());
        assert_eq!(state.as_value(), json!({"a": [3]}));
    }

    #[test]
    fn round_trips_through_json() {
        let state = GraphState::from_value(json!({"a": {"b": [1, 2]}, "c": null})).unwrap();
        let text = serde_json::to_string(&state).unwrap();
        assert_eq!(text, r#"{"a":{"b":[1,2]},"c":null}"#);
        let decoded: GraphState = serde_json::from_str(&text).unwrap();
        assert_eq!(decoded, state);
    }

    #[test]
    fn merge_node_delta_propagates_top_level_removals() {
        let base = GraphState::from_value(json!({"a": 1, "b": 2})).unwrap();
        let mut state = base.clone();
        let mut node = base.clone();
        node.remove("a");
        state.merge_node_delta(&base, node);
        assert_eq!(state.as_value(), json!({"b": 2}));
    }

    #[test]
    fn merge_node_delta_lets_a_later_node_win_over_an_earlier_deletion() {
        let base = GraphState::from_value(json!({"a": 1})).unwrap();
        let mut state = base.clone();

        let mut remover = base.clone();
        remover.remove("a");
        state.merge_node_delta(&base, remover);
        assert_eq!(state.as_value(), json!({}));

        let mut setter = base.clone();
        setter.set("a", json!(99));
        state.merge_node_delta(&base, setter);
        assert_eq!(state.as_value(), json!({"a": 99}));
    }

    #[test]
    fn merge_node_delta_does_not_resurrect_an_untouched_key() {
        let base = GraphState::from_value(json!({"a": 1, "b": 2})).unwrap();
        let mut state = base.clone();

        let mut remover = base.clone();
        remover.remove("a");
        state.merge_node_delta(&base, remover);

        // The later node still carried `a` from its clone of `base`, but it did
        // not touch it, so the earlier deletion stands.
        let later = base.clone();
        state.merge_node_delta(&base, later);
        assert_eq!(state.as_value(), json!({"b": 2}));
    }

    #[test]
    fn merge_node_delta_keeps_keys_added_by_earlier_nodes() {
        let base = GraphState::from_value(json!({"a": 1})).unwrap();
        let mut state = base.clone();

        let mut adder = base.clone();
        adder.set("new", json!(7));
        state.merge_node_delta(&base, adder);

        let later = base.clone();
        state.merge_node_delta(&base, later);
        assert_eq!(state.as_value(), json!({"a": 1, "new": 7}));
    }

    #[test]
    fn merge_node_delta_deep_merges_partial_nested_updates() {
        let base = GraphState::from_value(json!({"o": {"x": 0}})).unwrap();
        let mut state = base.clone();

        let mut first = base.clone();
        first.set_path("o.p", json!(1)).unwrap();
        state.merge_node_delta(&base, first);

        let mut second = base.clone();
        second.set_path("o.q", json!(2)).unwrap();
        state.merge_node_delta(&base, second);

        assert_eq!(state.as_value(), json!({"o": {"x": 0, "p": 1, "q": 2}}));
    }

    #[test]
    fn merge_node_delta_does_not_propagate_nested_removals() {
        let base = GraphState::from_value(json!({"a": {"b": 1, "c": 2}})).unwrap();
        let mut state = base.clone();

        let mut node = base.clone();
        if let Some(Value::Object(map)) = node.0.get_mut("a") {
            map.remove("b");
        }
        state.merge_node_delta(&base, node);

        // Documented behavior: only top-level keys are diffed, so the nested
        // delete is re-merged from `base`.
        assert_eq!(state.as_value(), json!({"a": {"b": 1, "c": 2}}));
    }

    #[test]
    fn keys_iterates_in_order() {
        let state = GraphState::from_value(json!({"b": 1, "a": 2})).unwrap();
        let keys: Vec<&String> = state.keys().collect();
        assert_eq!(keys, vec!["b", "a"]);
    }
}
