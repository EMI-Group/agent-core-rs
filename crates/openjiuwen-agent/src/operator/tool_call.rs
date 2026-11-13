//! `ToolCallOperator`: tool description parameters of a tool call.
//!
//! Ports `openjiuwen/core/operator/tool_call/base.py`. The operator caches the
//! description of every mounted tool, exposes them as one JSON map tunable
//! (`tool_description`) plus one text tunable per tool
//! (`tool_description.<name>`), and projects the cached descriptions onto
//! [`ToolCard`]s through [`ToolCallOperator::apply_to_cards`]. It never
//! executes a tool.
use std::collections::BTreeMap;

use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::ToolCard;
use serde_json::{json, Value};

use super::base::{ApplyResult, Operator, ParameterCallback, TunableKind, TunableSpec};

/// Target name of the whole description map.
pub const TOOL_DESCRIPTION_TARGET: &str = "tool_description";

/// Prefix of one per-tool description target (`tool_description.<name>`).
pub const TOOL_DESCRIPTION_PREFIX: &str = "tool_description.";

/// Tunable tool-description parameters of one tool-call dimension.
pub struct ToolCallOperator {
    operator_id: String,
    descriptions: BTreeMap<String, String>,
    on_parameter_updated: Option<ParameterCallback>,
}

impl ToolCallOperator {
    /// Creates a handle with no cached descriptions.
    #[must_use]
    pub fn new(operator_id: impl Into<String>) -> Self {
        Self {
            operator_id: operator_id.into(),
            descriptions: BTreeMap::new(),
            on_parameter_updated: None,
        }
    }

    /// Seeds the cached descriptions.
    #[must_use]
    pub fn with_descriptions(
        mut self,
        descriptions: impl IntoIterator<Item = (String, String)>,
    ) -> Self {
        self.descriptions = descriptions.into_iter().collect();
        self
    }

    /// Installs the parameter callback.
    #[must_use]
    pub fn with_callback(mut self, callback: ParameterCallback) -> Self {
        self.on_parameter_updated = Some(callback);
        self
    }

    /// The cached `tool name -> description` map.
    #[must_use]
    pub fn descriptions(&self) -> &BTreeMap<String, String> {
        &self.descriptions
    }

    /// The cached description of `name`, when known.
    #[must_use]
    pub fn description(&self, name: &str) -> Option<&str> {
        self.descriptions.get(name).map(String::as_str)
    }

    /// Overwrites the description of every card whose name has a cached
    /// description.
    pub fn apply_to_cards(&self, cards: &mut [ToolCard]) {
        for card in cards {
            if let Some(description) = self.descriptions.get(&card.name) {
                card.description.clone_from(description);
            }
        }
    }

    /// Replaces the whole map and returns its JSON form.
    fn store_map(&mut self, value: &Value) -> Result<Value> {
        let descriptions: BTreeMap<String, String> = serde_json::from_value(value.clone())
            .map_err(|_| {
                Error::validation(
                    "tunable 'tool_description' expects an object of tool name to text",
                )
            })?;
        self.descriptions = descriptions;
        Ok(json!(self.descriptions))
    }

    /// Stores one per-tool description (or removes it with `null`).
    fn store_one(&mut self, target: &str, name: &str, value: &Value) -> Result<Value> {
        if name.is_empty() {
            return Err(Error::validation(format!(
                "invalid tool description target '{target}'"
            )));
        }
        match value {
            Value::String(description) => {
                self.descriptions
                    .insert(name.to_owned(), description.clone());
            }
            Value::Null => {
                self.descriptions.remove(name);
            }
            _ => {
                return Err(Error::validation(format!(
                    "tunable '{target}' expects text or null"
                )))
            }
        }
        Ok(json!(self.descriptions))
    }

    /// Fires the callback with the whole map, mirroring the Python handle.
    fn fire(&self, value: &Value) {
        if let Some(callback) = &self.on_parameter_updated {
            callback(TOOL_DESCRIPTION_TARGET, value);
        }
    }
}

#[async_trait]
impl Operator for ToolCallOperator {
    fn operator_id(&self) -> &str {
        &self.operator_id
    }

    fn get_tunables(&self) -> Vec<TunableSpec> {
        if self.descriptions.is_empty() {
            return Vec::new();
        }
        let mut specs = vec![TunableSpec::new(TOOL_DESCRIPTION_TARGET, TunableKind::Json)
            .with_description("map of tool name to description")];
        specs.extend(self.descriptions.keys().map(|name| {
            TunableSpec::new(
                format!("{TOOL_DESCRIPTION_PREFIX}{name}"),
                TunableKind::Text,
            )
            .with_description(format!("description of tool '{name}'"))
        }));
        specs
    }

    fn get_state(&self) -> Value {
        json!({ TOOL_DESCRIPTION_TARGET: self.descriptions })
    }

    fn set_parameter(&mut self, target: &str, value: Value) -> Result<ApplyResult> {
        let stored = if target == TOOL_DESCRIPTION_TARGET {
            self.store_map(&value)
        } else if let Some(name) = target.strip_prefix(TOOL_DESCRIPTION_PREFIX) {
            self.store_one(target, name, &value)
        } else {
            return Ok(ApplyResult::skipped(target, "unknown target"));
        };
        let stored = match stored {
            Ok(stored) => stored,
            Err(error) => return Ok(ApplyResult::skipped(target, error.to_string())),
        };
        self.fire(&stored);
        Ok(ApplyResult::applied(target))
    }

    fn load_state(&mut self, state: Value) -> Result<()> {
        let Some(value) = state.get(TOOL_DESCRIPTION_TARGET) else {
            return Ok(());
        };
        let stored = self.store_map(value)?;
        self.fire(&stored);
        Ok(())
    }
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
    use std::sync::Arc;

    type Recorded = Arc<parking_lot::Mutex<Vec<(String, Value)>>>;

    fn recording_callback() -> (ParameterCallback, Recorded) {
        let log = Arc::new(parking_lot::Mutex::new(Vec::new()));
        let sink = Arc::clone(&log);
        let callback: ParameterCallback = Arc::new(move |target: &str, value: &Value| {
            sink.lock().push((target.to_owned(), value.clone()));
        });
        (callback, log)
    }

    fn operator() -> ToolCallOperator {
        ToolCallOperator::new("agent/tool_call").with_descriptions([
            ("search".to_owned(), "Search the web".to_owned()),
            ("read".to_owned(), "Read a file".to_owned()),
        ])
    }

    #[test]
    fn tunables_cover_the_map_and_each_tool() {
        assert!(ToolCallOperator::new("op").get_tunables().is_empty());
        let tunables = operator().get_tunables();
        let names: Vec<&str> = tunables.iter().map(|spec| spec.name.as_str()).collect();
        assert_eq!(
            names,
            [
                "tool_description",
                "tool_description.read",
                "tool_description.search"
            ]
        );
        assert_eq!(tunables[0].kind, TunableKind::Json);
        assert_eq!(tunables[1].kind, TunableKind::Text);
    }

    #[test]
    fn set_parameter_replaces_the_map_and_fires_it() {
        let (callback, log) = recording_callback();
        let mut operator = operator().with_callback(callback);
        let result = operator
            .set_parameter("tool_description", json!({"search": "Find things"}))
            .unwrap();
        assert!(result.applied);
        assert_eq!(operator.descriptions().len(), 1);
        assert_eq!(operator.description("search"), Some("Find things"));
        assert_eq!(
            log.lock().as_slice(),
            [(
                "tool_description".to_owned(),
                json!({"search": "Find things"})
            )]
        );
    }

    #[test]
    fn per_tool_targets_update_one_entry_and_fire_the_map() {
        let (callback, log) = recording_callback();
        let mut operator = operator().with_callback(callback);
        let result = operator
            .set_parameter("tool_description.read", json!("Read one file"))
            .unwrap();
        assert!(result.applied);
        assert_eq!(operator.description("read"), Some("Read one file"));
        assert_eq!(operator.description("search"), Some("Search the web"));
        assert_eq!(log.lock()[0].0, "tool_description");
        assert_eq!(operator.description("missing"), None);

        operator
            .set_parameter("tool_description.read", Value::Null)
            .unwrap();
        assert_eq!(operator.description("read"), None);
    }

    #[test]
    fn unknown_and_invalid_values_are_skipped() {
        let mut operator = operator();
        assert!(!operator.set_parameter("other", json!(1)).unwrap().applied);
        assert!(
            !operator
                .set_parameter("tool_description", json!(["not", "a", "map"]))
                .unwrap()
                .applied
        );
        assert_eq!(operator.descriptions().len(), 2);
        let result = operator
            .set_parameter("tool_description.", json!("x"))
            .unwrap();
        assert!(!result.applied);
    }

    #[test]
    fn apply_to_cards_rewrites_matching_descriptions() {
        let mut cards = vec![
            ToolCard::new("search", "old search", json!({})),
            ToolCard::new("other", "untouched", json!({})),
        ];
        operator().apply_to_cards(&mut cards);
        assert_eq!(cards[0].description, "Search the web");
        assert_eq!(cards[1].description, "untouched");
    }

    #[test]
    fn state_round_trips_and_load_fires_the_map() {
        let state = operator().get_state();
        assert_eq!(
            state,
            json!({"tool_description": {"read": "Read a file", "search": "Search the web"}})
        );
        let (callback, log) = recording_callback();
        let mut restored = ToolCallOperator::new("op").with_callback(callback);
        restored
            .load_state(serde_json::from_value(state).unwrap())
            .unwrap();
        assert_eq!(restored.descriptions().len(), 2);
        assert_eq!(log.lock()[0].0, "tool_description");
        assert_eq!(restored.get_state(), operator().get_state());
    }
}
