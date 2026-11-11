//! Request-local runtime context shared by single-agent and team execution.
//!
//! Port of Python `openjiuwen/core/common/run_context.py`. A caller inside the SDK
//! passes a typed [`RunContext`]; an external caller may pass a loosely-typed
//! mapping, which [`normalize_run_context`] folds into a [`RunContext`] while
//! preserving unknown top-level keys in [`RunContext::extra`].

use std::collections::BTreeMap;
use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::error::Error;

/// Why a heartbeat execution started (Python `HeartbeatReason`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[must_use]
pub enum HeartbeatReason {
    /// A scheduled heartbeat (`"interval"`), the default.
    #[default]
    Interval,
    /// A heartbeat triggered by an external call (`"manual"`).
    Manual,
}

impl HeartbeatReason {
    /// The Python enum value (`"interval"` or `"manual"`).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Interval => "interval",
            Self::Manual => "manual",
        }
    }
}

impl fmt::Display for HeartbeatReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for HeartbeatReason {
    type Err = Error;

    /// Parses a reason case-insensitively; anything else is a validation error.
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "interval" => Ok(Self::Interval),
            "manual" => Ok(Self::Manual),
            other => Err(Error::validation(format!(
                "unknown heartbeat reason: {other}"
            ))),
        }
    }
}

/// Request-local runtime data carried across one logical invocation.
///
/// Python `RunContext` dataclass; unknown mapping keys end up in [`Self::extra`].
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
#[must_use]
pub struct RunContext {
    /// Why the execution started, if the caller declared it.
    pub reason: Option<HeartbeatReason>,
    /// Session this invocation belongs to.
    pub session_id: Option<String>,
    /// Context mode requested by the caller (for example `"lightweight"`).
    pub context_mode: Option<String>,
    /// Extension fields supplied by the caller and preserved verbatim.
    pub extra: BTreeMap<String, Value>,
}

impl RunContext {
    /// Creates an empty context.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the heartbeat reason.
    pub fn with_reason(mut self, reason: HeartbeatReason) -> Self {
        self.reason = Some(reason);
        self
    }

    /// Sets the session id.
    pub fn with_session_id(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }

    /// Sets the context mode.
    pub fn with_context_mode(mut self, context_mode: impl Into<String>) -> Self {
        self.context_mode = Some(context_mode.into());
        self
    }

    /// Replaces the extension fields.
    pub fn with_extra(mut self, extra: BTreeMap<String, Value>) -> Self {
        self.extra = extra;
        self
    }
}

/// A public execution boundary accepts either a typed context or a mapping
/// (Python `RunContextInput` type alias).
#[derive(Debug, Clone, PartialEq)]
#[must_use]
pub enum RunContextInput {
    /// A typed context, passed through untouched by [`normalize_run_context`].
    Context(RunContext),
    /// A loosely-typed mapping that still has to be normalized.
    Mapping(Map<String, Value>),
}

impl From<RunContext> for RunContextInput {
    fn from(context: RunContext) -> Self {
        Self::Context(context)
    }
}

impl From<Map<String, Value>> for RunContextInput {
    fn from(map: Map<String, Value>) -> Self {
        Self::Mapping(map)
    }
}

impl From<Value> for RunContextInput {
    /// Objects become [`RunContextInput::Mapping`]; JSON null becomes a context with
    /// defaults; any other scalar becomes an empty mapping.
    fn from(value: Value) -> Self {
        match value {
            Value::Object(map) => Self::Mapping(map),
            Value::Null => Self::Context(RunContext::default()),
            _ => Self::Mapping(Map::new()),
        }
    }
}

/// Normalizes a public input into the shared runtime model (Python
/// `normalize_run_context`).
///
/// `None` stays `None`, a typed [`RunContext`] is returned unchanged, and a
/// mapping is rebuilt: `extra` must be a JSON object (otherwise it is dropped),
/// `reason` must be a recognized reason string, `session_id` and `context_mode`
/// must be strings, and every remaining key is folded into `extra`, overwriting a
/// same-named `extra` key (Python parity).
#[must_use]
pub fn normalize_run_context(input: Option<RunContextInput>) -> Option<RunContext> {
    match input? {
        RunContextInput::Context(context) => Some(context),
        RunContextInput::Mapping(map) => Some(normalize_mapping(map)),
    }
}

fn normalize_mapping(mut map: Map<String, Value>) -> RunContext {
    let extra = match map.remove("extra") {
        Some(Value::Object(entries)) => entries.into_iter().collect(),
        _ => BTreeMap::new(),
    };
    let reason = map.remove("reason").and_then(|value| match value {
        Value::String(text) => HeartbeatReason::from_str(&text).ok(),
        _ => None,
    });
    let session_id = string_field(map.remove("session_id"));
    let context_mode = string_field(map.remove("context_mode"));

    let mut context = RunContext {
        reason,
        session_id,
        context_mode,
        extra,
    };
    for (key, value) in map {
        context.extra.insert(key, value);
    }
    context
}

fn string_field(value: Option<Value>) -> Option<String> {
    match value {
        Some(Value::String(text)) => Some(text),
        _ => None,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    //! Port of `tests/unit_tests/core/common/test_run_context.py`.

    use serde_json::json;

    use super::*;

    #[test]
    fn heartbeat_reason_values_and_serde() {
        assert_eq!(HeartbeatReason::Interval.as_str(), "interval");
        assert_eq!(HeartbeatReason::Manual.to_string(), "manual");
        assert_eq!(HeartbeatReason::default(), HeartbeatReason::Interval);

        assert_eq!(
            serde_json::to_value(HeartbeatReason::Manual).unwrap(),
            json!("manual")
        );
        assert_eq!(
            serde_json::from_value::<HeartbeatReason>(json!("interval")).unwrap(),
            HeartbeatReason::Interval
        );

        assert_eq!(
            "manual".parse::<HeartbeatReason>().unwrap(),
            HeartbeatReason::Manual
        );
        assert_eq!(
            "INTERVAL".parse::<HeartbeatReason>().unwrap(),
            HeartbeatReason::Interval
        );
        assert!("sometimes".parse::<HeartbeatReason>().is_err());
    }

    #[test]
    fn normalize_keeps_typed_context_identity() {
        let context = RunContext::new()
            .with_extra(BTreeMap::from([("chain_id".to_owned(), json!("chain-1"))]));

        let normalized = normalize_run_context(Some(RunContextInput::Context(context.clone())))
            .expect("typed context is preserved");

        assert_eq!(normalized, context);
    }

    #[test]
    fn normalize_preserves_extension_fields_in_extra() {
        let input = json!({
            "reason": "interval",
            "session_id": "session-a",
            "context_mode": "lightweight",
            "extra": {"message_id": "message-1"},
            "chain_id": "chain-1",
        });

        let normalized = normalize_run_context(Some(input.into())).expect("mapping is normalized");

        assert_eq!(
            normalized,
            RunContext {
                reason: Some(HeartbeatReason::Interval),
                session_id: Some("session-a".to_owned()),
                context_mode: Some("lightweight".to_owned()),
                extra: BTreeMap::from([
                    ("message_id".to_owned(), json!("message-1")),
                    ("chain_id".to_owned(), json!("chain-1")),
                ]),
            }
        );
    }

    #[test]
    fn normalize_ignores_unknown_reason() {
        let normalized = normalize_run_context(Some(json!({"reason": "sometimes"}).into()))
            .expect("mapping is normalized");
        assert!(normalized.reason.is_none());

        let non_string = normalize_run_context(Some(json!({"reason": 42}).into()))
            .expect("mapping is normalized");
        assert!(non_string.reason.is_none());
    }

    #[test]
    fn normalize_ignores_non_object_extra() {
        for extra in [json!("nope"), json!([1, 2]), json!(null)] {
            let normalized =
                normalize_run_context(Some(json!({"extra": extra}).into())).expect("normalized");
            assert!(normalized.extra.is_empty());
        }
    }

    #[test]
    fn normalize_ignores_non_string_session_and_context_mode() {
        let normalized = normalize_run_context(Some(
            json!({"session_id": 7, "context_mode": ["lightweight"]}).into(),
        ))
        .expect("normalized");
        assert!(normalized.session_id.is_none());
        assert!(normalized.context_mode.is_none());
    }

    #[test]
    fn remaining_keys_overwrite_same_named_extra_keys() {
        let normalized = normalize_run_context(Some(
            json!({"extra": {"chain_id": "old"}, "chain_id": "new"}).into(),
        ))
        .expect("normalized");

        assert_eq!(normalized.extra.get("chain_id"), Some(&json!("new")));
    }

    #[test]
    fn normalize_maps_null_and_scalars() {
        assert_eq!(normalize_run_context(None), None);
        assert_eq!(
            normalize_run_context(Some(Value::Null.into())),
            Some(RunContext::default())
        );
        assert_eq!(
            normalize_run_context(Some(json!("scalar").into())),
            Some(RunContext::default())
        );
    }

    #[test]
    fn run_context_serde_round_trip() {
        let context = RunContext::new()
            .with_reason(HeartbeatReason::Manual)
            .with_session_id("session-a")
            .with_context_mode("lightweight");

        let json = serde_json::to_value(&context).unwrap();
        assert_eq!(json["reason"], "manual");
        assert_eq!(json["session_id"], "session-a");
        assert_eq!(json["context_mode"], "lightweight");
        assert_eq!(json["extra"], json!({}));

        let restored: RunContext = serde_json::from_value(json).unwrap();
        assert_eq!(restored, context);
    }

    #[test]
    fn run_context_tolerates_missing_and_unknown_fields() {
        let context: RunContext = serde_json::from_value(json!({"unknown": 1})).unwrap();
        assert_eq!(context, RunContext::default());
        assert!(context.reason.is_none());
    }
}
