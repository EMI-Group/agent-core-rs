//! Tool stubs and tool-call traces for tool-aware case evaluation.
//!
//! A tuning [`Case`](crate::tune::Case) may declare the tools an agent is allowed
//! to use, but scoring only sees the answer. This module supplies the other half
//! of that picture: [`ToolStub`] is a named, in-process stand-in a runner can
//! execute instead of a real tool, [`ToolStubs`] is the lookup table handed to a
//! runner, [`ToolCallRecord`] records one executed call and its outcome, and
//! [`ToolTrace`] is the ordered list of those calls. [`RunOutcome`] bundles an
//! answer map with the trace behind it, so an evaluator can score both what the
//! agent answered and which tools it actually used.
//!
//! Stubs are synchronous closures: they are deterministic fixtures, not a
//! transport, and every failure is an ordinary
//! [`Error`](openjiuwen_common::error::Error) value that the caller records
//! rather than a panic.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// The synchronous body of a [`ToolStub`]: arguments in, result out.
///
/// Stubs model offline fixtures, so the handler stays synchronous and shared
/// through an [`Arc`]; a runner that needs an async tool layer wraps it itself.
pub type ToolHandler = dyn Fn(&Value) -> Result<Value> + Send + Sync;

/// A named, in-process stand-in for one tool.
///
/// A stub answers a call with a fixed fixture (see [`ToolStub::constant`]) or
/// with whatever its handler computes, which keeps tool-aware evaluation
/// deterministic and free of any transport.
///
/// Cloning shares the handler through its [`Arc`], so a cloned table stays cheap.
#[derive(Clone)]
pub struct ToolStub {
    /// The tool name a case declares and a runner looks the stub up by.
    name: String,
    /// The callable body; shared so cloned tables stay cheap.
    handler: Arc<ToolHandler>,
}

impl ToolStub {
    /// Creates a stub from a name and a handler.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] when the trimmed name is blank, naming the stub's
    /// `name` field.
    pub fn new(
        name: impl Into<String>,
        handler: impl Fn(&Value) -> Result<Value> + Send + Sync + 'static,
    ) -> Result<Self> {
        let name = name.into();
        if name.trim().is_empty() {
            return Err(Error::validation("ToolStub.name must not be empty"));
        }
        Ok(Self {
            name,
            handler: Arc::new(handler),
        })
    }

    /// Creates a stub that always returns `value`, ignoring its arguments.
    ///
    /// This is the shape most evaluation fixtures need: the case only cares
    /// *that* the tool was called, not what it was called with.
    ///
    /// # Errors
    ///
    /// [`Error::validation`] when the trimmed name is blank.
    pub fn constant(name: impl Into<String>, value: Value) -> Result<Self> {
        Self::new(name, move |_arguments| Ok(value.clone()))
    }

    /// The stub's tool name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Invokes the handler with `arguments` and returns its result unchanged.
    ///
    /// # Errors
    ///
    /// Whatever the handler returns as an error.
    pub fn call(&self, arguments: &Value) -> Result<Value> {
        (self.handler)(arguments)
    }
}

#[allow(clippy::missing_fields_in_debug)] // `handler` is an opaque closure and has no `Debug` form
impl fmt::Debug for ToolStub {
    /// Renders the name only; handlers are opaque closures and are not `Debug`.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ToolStub")
            .field("name", &self.name)
            .finish()
    }
}

/// A name-keyed collection of [`ToolStub`]s.
///
/// The `BTreeMap` keeps [`ToolStubs::names`] sorted and every diagnostic stable,
/// so an error listing the registered tools never depends on insertion order.
#[derive(Debug, Clone, Default)]
pub struct ToolStubs {
    /// The registered stubs, keyed by tool name.
    stubs: BTreeMap<String, ToolStub>,
}

impl ToolStubs {
    /// An empty table.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers `stub`, returning the stub it replaced, if any.
    pub fn insert(&mut self, stub: ToolStub) -> Option<ToolStub> {
        self.stubs.insert(stub.name().to_owned(), stub)
    }

    /// Registers `stub` and returns the table, for builder chains.
    #[must_use]
    pub fn with_stub(mut self, stub: ToolStub) -> Self {
        self.insert(stub);
        self
    }

    /// The stub registered under `name`, when present.
    pub fn get(&self, name: &str) -> Option<&ToolStub> {
        self.stubs.get(name)
    }

    /// Whether a stub is registered under `name`.
    pub fn contains(&self, name: &str) -> bool {
        self.stubs.contains_key(name)
    }

    /// The registered names, sorted.
    pub fn names(&self) -> Vec<&str> {
        self.stubs.keys().map(String::as_str).collect()
    }

    /// How many stubs are registered.
    pub fn len(&self) -> usize {
        self.stubs.len()
    }

    /// Whether no stub is registered.
    pub fn is_empty(&self) -> bool {
        self.stubs.is_empty()
    }

    /// Calls the stub registered under `name`.
    ///
    /// # Errors
    ///
    /// [`Error::not_found`] naming the unknown tool and listing the registered
    /// names in sorted order, or whatever the stub's handler returned.
    pub fn call(&self, name: &str, arguments: &Value) -> Result<Value> {
        let Some(stub) = self.stubs.get(name) else {
            let registered = self.names();
            let listing = if registered.is_empty() {
                "(none)".to_owned()
            } else {
                registered.join(", ")
            };
            return Err(Error::not_found(format!(
                "unknown tool '{name}'; registered tools: {listing}"
            )));
        };
        stub.call(arguments)
    }
}

/// One executed tool call and its outcome.
///
/// Serializes with Python-ish `snake_case` field names `name`, `arguments`,
/// `result`, and `error`; deserialization tolerates unknown fields and a missing
/// `error`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ToolCallRecord {
    /// The tool that was called.
    pub name: String,
    /// The arguments the call was made with.
    pub arguments: Value,
    /// The result the tool produced.
    pub result: Value,
    /// The failure text when the stub failed (or the tool was unknown);
    /// `None` on success.
    pub error: Option<String>,
}

impl ToolCallRecord {
    /// Records a successful call.
    pub fn success(name: impl Into<String>, arguments: Value, result: Value) -> Self {
        Self {
            name: name.into(),
            arguments,
            result,
            error: None,
        }
    }

    /// Records a failed call; `result` mirrors the failure text so a report
    /// shows what the tool answered instead of `null`.
    pub fn failure(name: impl Into<String>, arguments: Value, error: impl Into<String>) -> Self {
        let error = error.into();
        Self {
            name: name.into(),
            arguments,
            result: Value::String(error.clone()),
            error: Some(error),
        }
    }

    /// Whether the call failed.
    pub fn is_error(&self) -> bool {
        self.error.is_some()
    }
}

/// The tool calls executed while running one case, in call order.
///
/// Serializes transparently as the bare JSON array of [`ToolCallRecord`]s, which
/// keeps the trace embeddable in a report without an extra wrapper object.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ToolTrace {
    /// The recorded calls, oldest first.
    records: Vec<ToolCallRecord>,
}

impl ToolTrace {
    /// Collects records into a trace, preserving their order.
    pub fn new(records: impl IntoIterator<Item = ToolCallRecord>) -> Self {
        Self {
            records: records.into_iter().collect(),
        }
    }

    /// Appends one record.
    pub fn push(&mut self, record: ToolCallRecord) {
        self.records.push(record);
    }

    /// Whether no call was recorded.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// How many calls were recorded.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// The recorded calls, oldest first.
    pub fn calls(&self) -> &[ToolCallRecord] {
        &self.records
    }

    /// Consumes the trace and returns its records.
    pub fn into_records(self) -> Vec<ToolCallRecord> {
        self.records
    }

    /// The called names in call order, duplicates kept.
    pub fn names(&self) -> Vec<&str> {
        self.records
            .iter()
            .map(|record| record.name.as_str())
            .collect()
    }

    /// Whether any call used `name`, successful or failed.
    pub fn used(&self, name: &str) -> bool {
        self.records.iter().any(|record| record.name == name)
    }

    /// Whether every name in `names` was used; an empty list is trivially used.
    pub fn used_all(&self, names: &[&str]) -> bool {
        names.iter().all(|name| self.used(name))
    }

    /// How many calls used `name`.
    pub fn count(&self, name: &str) -> usize {
        self.records
            .iter()
            .filter(|record| record.name == name)
            .count()
    }

    /// Fraction of `names` that appear in the trace.
    ///
    /// Duplicate names count once, an empty list is `1.0`, and the result is
    /// clamped into `0.0..=1.0`.
    pub fn coverage(&self, names: &[&str]) -> f64 {
        let expected = distinct(names);
        if expected.is_empty() {
            return 1.0;
        }
        let covered = expected
            .iter()
            .filter(|name| self.used(name.as_str()))
            .count();
        (covered as f64 / expected.len() as f64).clamp(0.0, 1.0)
    }

    /// The `names` the trace never used, in input order and deduplicated.
    pub fn missing(&self, names: &[&str]) -> Vec<String> {
        distinct(names)
            .into_iter()
            .filter(|name| !self.used(name.as_str()))
            .collect()
    }

    /// Call counts by tool name, for names that were actually called.
    pub fn usage(&self) -> BTreeMap<String, usize> {
        let mut usage = BTreeMap::new();
        for record in &self.records {
            *usage.entry(record.name.clone()).or_insert(0) += 1;
        }
        usage
    }
}

/// The distinct names of `names`, in first-occurrence order.
fn distinct(names: &[&str]) -> Vec<String> {
    let mut distinct: Vec<String> = Vec::with_capacity(names.len());
    for name in names {
        if !distinct.iter().any(|seen| seen == name) {
            distinct.push((*name).to_owned());
        }
    }
    distinct
}

/// What one run cost: wall-clock latency, token counts, and money.
///
/// Every field is optional, and `None` means *not reported* rather than zero: a
/// runner that measured nothing reports `RunTelemetry::default()`, a runner that
/// called no model leaves the token fields absent, and cost is never inferred
/// from tokens because pricing is the caller's decision.
///
/// Deserialization defaults every field, so an older payload without telemetry
/// still parses and a partial payload keeps only the counters it carries.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct RunTelemetry {
    /// Wall-clock time the run took, in milliseconds.
    pub latency_ms: Option<f64>,
    /// Tokens consumed by the prompts sent to the model.
    pub prompt_tokens: Option<u64>,
    /// Tokens produced by the model.
    pub completion_tokens: Option<u64>,
    /// Monetary cost the caller attributes to the run.
    pub cost: Option<f64>,
}

impl RunTelemetry {
    /// Whether nothing was reported at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.latency_ms.is_none()
            && self.prompt_tokens.is_none()
            && self.completion_tokens.is_none()
            && self.cost.is_none()
    }

    /// The token total, or `None` when neither side was reported.
    ///
    /// A reported side with an unreported counterpart counts the missing side as
    /// zero, so a prompt-only measurement still yields a total.
    #[must_use]
    pub fn total_tokens(&self) -> Option<u64> {
        match (self.prompt_tokens, self.completion_tokens) {
            (None, None) => None,
            (prompt, completion) => {
                Some(prompt.unwrap_or(0).saturating_add(completion.unwrap_or(0)))
            }
        }
    }

    /// Sets the measured latency in milliseconds.
    #[must_use]
    pub fn with_latency_ms(mut self, latency_ms: f64) -> Self {
        self.latency_ms = Some(latency_ms);
        self
    }

    /// Sets the measured prompt token count.
    #[must_use]
    pub fn with_prompt_tokens(mut self, prompt_tokens: u64) -> Self {
        self.prompt_tokens = Some(prompt_tokens);
        self
    }

    /// Sets the measured completion token count.
    #[must_use]
    pub fn with_completion_tokens(mut self, completion_tokens: u64) -> Self {
        self.completion_tokens = Some(completion_tokens);
        self
    }

    /// Sets the attributed cost.
    #[must_use]
    pub fn with_cost(mut self, cost: f64) -> Self {
        self.cost = Some(cost);
        self
    }

    /// Adds every value `other` reports, leaving absent values untouched.
    ///
    /// Adding an absent value changes nothing, so merging an empty telemetry is
    /// the identity; merging into an empty one copies `other`.
    pub fn merge(&mut self, other: RunTelemetry) {
        self.latency_ms = sum_f64(self.latency_ms, other.latency_ms);
        self.prompt_tokens = sum_u64(self.prompt_tokens, other.prompt_tokens);
        self.completion_tokens = sum_u64(self.completion_tokens, other.completion_tokens);
        self.cost = sum_f64(self.cost, other.cost);
    }
}

/// Adds two optional `f64`s, treating an absent value as "no contribution".
fn sum_f64(left: Option<f64>, right: Option<f64>) -> Option<f64> {
    match (left, right) {
        (None, None) => None,
        (Some(value), None) | (None, Some(value)) => Some(value),
        (Some(left), Some(right)) => Some(left + right),
    }
}

/// Adds two optional `u64`s, treating an absent value as "no contribution".
fn sum_u64(left: Option<u64>, right: Option<u64>) -> Option<u64> {
    match (left, right) {
        (None, None) => None,
        (Some(value), None) | (None, Some(value)) => Some(value),
        (Some(left), Some(right)) => Some(left.saturating_add(right)),
    }
}

/// What one run produced: the answer map plus the tool calls behind it.
///
/// Deserialization defaults every field, so an older payload that only carries
/// an answer still parses and reports an empty trace and absent telemetry.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct RunOutcome {
    /// The answer map a scorer compares against the case label.
    pub answer: BTreeMap<String, Value>,
    /// The tool calls executed while producing `answer`.
    pub trace: ToolTrace,
    /// What the run cost, when the runner measured it.
    pub telemetry: RunTelemetry,
}

impl RunOutcome {
    /// Wraps an answer with an empty trace and absent telemetry.
    pub fn new(answer: BTreeMap<String, Value>) -> Self {
        Self {
            answer,
            ..Self::default()
        }
    }

    /// Wraps an answer with the trace that produced it.
    pub fn with_trace(answer: BTreeMap<String, Value>, trace: ToolTrace) -> Self {
        Self {
            answer,
            trace,
            ..Self::default()
        }
    }

    /// Attaches what the run cost.
    #[must_use]
    pub fn with_telemetry(mut self, telemetry: RunTelemetry) -> Self {
        self.telemetry = telemetry;
        self
    }

    /// Consumes the outcome and returns its answer map.
    pub fn into_answer(self) -> BTreeMap<String, Value> {
        self.answer
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use openjiuwen_common::error::ErrorKind;
    use serde_json::json;

    /// A stub that echoes its arguments back as the result.
    fn echo(name: &str) -> ToolStub {
        ToolStub::new(name, |arguments| Ok(arguments.clone())).unwrap()
    }

    /// A sample successful call record.
    fn success(name: &str) -> ToolCallRecord {
        ToolCallRecord::success(name, json!({"q": "x"}), json!({"ok": true}))
    }

    #[test]
    fn stub_new_rejects_blank_names() {
        for name in ["", "   ", "\t\n"] {
            let error = ToolStub::new(name, |_| Ok(Value::Null)).unwrap_err();
            assert!(
                matches!(&error, Error::Validation(message) if message.contains("name")),
                "unexpected error for {name:?}: {error}"
            );
        }
    }

    #[test]
    fn stub_name_and_call_use_the_handler() {
        let stub = echo("lookup");
        assert_eq!(stub.name(), "lookup");
        assert_eq!(stub.call(&json!({"q": "hi"})).unwrap(), json!({"q": "hi"}));
    }

    #[test]
    fn constant_stub_ignores_its_arguments() {
        let stub = ToolStub::constant("weather", json!({"temp": 21})).unwrap();
        assert_eq!(stub.call(&Value::Null).unwrap(), json!({"temp": 21}));
        assert_eq!(
            stub.call(&json!({"city": "here"})).unwrap(),
            json!({"temp": 21})
        );
    }

    #[test]
    fn handler_errors_propagate_unchanged() {
        let stub = ToolStub::new("broken", |_| Err(Error::tool("handler failed"))).unwrap();
        let error = stub.call(&json!({})).unwrap_err();
        assert!(
            matches!(&error, Error::Tool(message) if message == "handler failed"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn stub_debug_shows_only_the_name() {
        let rendered = format!("{:?}", echo("lookup"));
        assert!(
            rendered.contains("ToolStub"),
            "unexpected debug: {rendered}"
        );
        assert!(rendered.contains("lookup"), "unexpected debug: {rendered}");
    }

    #[test]
    fn stubs_register_look_up_and_report_sorted_names() {
        let mut stubs = ToolStubs::new();
        assert!(stubs.is_empty());
        assert_eq!(stubs.len(), 0);
        assert_eq!(stubs.names(), Vec::<&str>::new());

        assert!(stubs.insert(echo("zeta")).is_none());
        stubs.insert(echo("alpha"));
        stubs.insert(echo("mike"));
        assert_eq!(stubs.len(), 3);
        assert!(!stubs.is_empty());
        assert_eq!(stubs.names(), vec!["alpha", "mike", "zeta"]);
        assert!(stubs.contains("mike"));
        assert!(!stubs.contains("nope"));
        assert_eq!(stubs.get("alpha").map(ToolStub::name), Some("alpha"));
        assert!(stubs.get("nope").is_none());
    }

    #[test]
    fn insert_replaces_a_same_named_stub() {
        let mut stubs = ToolStubs::new();
        stubs.insert(ToolStub::constant("weather", json!({"temp": 1})).unwrap());
        let replaced = stubs
            .insert(ToolStub::constant("weather", json!({"temp": 2})).unwrap())
            .expect("the first stub is returned");
        assert_eq!(replaced.call(&Value::Null).unwrap(), json!({"temp": 1}));
        assert_eq!(stubs.len(), 1);
        assert_eq!(
            stubs.call("weather", &Value::Null).unwrap(),
            json!({"temp": 2})
        );
    }

    #[test]
    fn with_stub_chains_and_dispatches() {
        let stubs = ToolStubs::new()
            .with_stub(echo("a"))
            .with_stub(ToolStub::constant("b", json!(7)).unwrap());
        assert_eq!(stubs.names(), vec!["a", "b"]);
        assert_eq!(stubs.call("a", &json!({"k": 1})).unwrap(), json!({"k": 1}));
        assert_eq!(stubs.call("b", &Value::Null).unwrap(), json!(7));
    }

    #[test]
    fn call_on_an_unknown_name_reports_not_found_with_the_registered_names() {
        let stubs = ToolStubs::new()
            .with_stub(echo("zeta"))
            .with_stub(echo("alpha"));
        let error = stubs.call("missing", &Value::Null).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::NotFound);
        let message = error.to_string();
        assert!(message.contains("missing"), "unexpected message: {message}");
        assert!(message.contains("alpha"), "unexpected message: {message}");
        assert!(message.contains("zeta"), "unexpected message: {message}");
        let alpha = message.find("alpha").unwrap();
        let zeta = message.find("zeta").unwrap();
        assert!(alpha < zeta, "names must be listed sorted: {message}");
    }

    #[test]
    fn call_on_an_unknown_name_with_no_stubs_says_none() {
        let error = ToolStubs::new().call("missing", &Value::Null).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::NotFound);
        assert!(error.to_string().contains("(none)"), "unexpected: {error}");
    }

    #[test]
    fn call_propagates_a_stub_failure() {
        let stubs = ToolStubs::new()
            .with_stub(ToolStub::new("broken", |_| Err(Error::tool("boom"))).unwrap());
        assert!(matches!(
            stubs.call("broken", &Value::Null),
            Err(Error::Tool(message)) if message == "boom"
        ));
    }

    #[test]
    fn record_constructors_and_is_error() {
        let record = success("lookup");
        assert_eq!(record.name, "lookup");
        assert_eq!(record.arguments, json!({"q": "x"}));
        assert_eq!(record.result, json!({"ok": true}));
        assert!(record.error.is_none());
        assert!(!record.is_error());

        let failed = ToolCallRecord::failure("lookup", json!({"q": "x"}), "no rows");
        assert_eq!(failed.error.as_deref(), Some("no rows"));
        assert_eq!(failed.result, Value::String("no rows".to_owned()));
        assert!(failed.is_error());
    }

    #[test]
    fn record_json_round_trip_uses_snake_case_fields() {
        let record = success("lookup");
        let value = serde_json::to_value(&record).unwrap();
        let object = value.as_object().unwrap();
        for key in ["name", "arguments", "result", "error"] {
            assert!(object.contains_key(key), "missing field {key}");
        }
        let restored: ToolCallRecord = serde_json::from_value(value).unwrap();
        assert_eq!(restored, record);
    }

    #[test]
    fn record_deserialization_tolerates_unknown_fields_and_a_missing_error() {
        let record: ToolCallRecord = serde_json::from_str(
            r#"{"name": "lookup", "arguments": {"q": "x"}, "result": null, "extra": [1]}"#,
        )
        .unwrap();
        assert_eq!(record.name, "lookup");
        assert_eq!(record.arguments, json!({"q": "x"}));
        assert_eq!(record.result, Value::Null);
        assert!(record.error.is_none());
        assert!(!record.is_error());
    }

    #[test]
    fn trace_tracks_push_order_and_names() {
        let mut trace = ToolTrace::new([success("first"), success("second")]);
        assert_eq!(trace.len(), 2);
        assert!(!trace.is_empty());
        trace.push(ToolCallRecord::failure("first", json!({}), "bad"));
        assert_eq!(trace.len(), 3);
        assert_eq!(trace.names(), vec!["first", "second", "first"]);
        assert_eq!(trace.calls()[2].name, "first");
        assert_eq!(trace.into_records().len(), 3);
        assert!(ToolTrace::new([]).is_empty());
    }

    #[test]
    fn trace_used_used_all_and_count_include_failures() {
        let trace = ToolTrace::new([
            success("a"),
            ToolCallRecord::failure("b", json!({}), "bad"),
            success("a"),
        ]);
        assert!(trace.used("a"));
        assert!(trace.used("b"), "a failed call still counts as used");
        assert!(!trace.used("c"));
        assert_eq!(trace.count("a"), 2);
        assert_eq!(trace.count("b"), 1);
        assert_eq!(trace.count("c"), 0);
        assert!(trace.used_all(&[]));
        assert!(trace.used_all(&["a", "b"]));
        assert!(!trace.used_all(&["a", "c"]));
    }

    #[test]
    fn trace_coverage_handles_empty_duplicates_and_unknowns() {
        let trace = ToolTrace::new([success("a"), success("c")]);
        assert!((trace.coverage(&[]) - 1.0).abs() < f64::EPSILON);
        assert!((trace.coverage(&["a", "b", "c"]) - 2.0 / 3.0).abs() < 1e-12);
        assert!(trace.coverage(&["x", "y"]).abs() < f64::EPSILON);
        assert!((trace.coverage(&["a"]) - 1.0).abs() < f64::EPSILON);
        assert!(
            (trace.coverage(&["a", "a", "c"]) - 1.0).abs() < 1e-12,
            "duplicates count once"
        );
        assert!((trace.coverage(&["a", "missing"]) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn trace_missing_preserves_order_and_deduplicates() {
        let trace = ToolTrace::new([success("a"), success("c")]);
        assert_eq!(trace.missing(&["b", "a", "b", "d", "c"]), vec!["b", "d"]);
        assert_eq!(trace.missing(&["a", "c"]), Vec::<String>::new());
        assert_eq!(trace.missing(&[]), Vec::<String>::new());
        assert_eq!(trace.missing(&["b", "a"]), vec!["b"]);
    }

    #[test]
    fn trace_usage_counts_are_sorted_and_skip_unused_names() {
        let trace = ToolTrace::new([
            success("b"),
            success("a"),
            ToolCallRecord::failure("b", json!({}), "bad"),
        ]);
        let usage = trace.usage();
        let entries: Vec<(&str, usize)> = usage
            .iter()
            .map(|(name, count)| (name.as_str(), *count))
            .collect();
        assert_eq!(entries, vec![("a", 1), ("b", 2)]);
        assert!(ToolTrace::new([]).usage().is_empty());
    }

    #[test]
    fn trace_serializes_as_a_bare_array() {
        let trace = ToolTrace::new([success("a"), success("b")]);
        let value = serde_json::to_value(&trace).unwrap();
        let array = value.as_array().expect("a bare JSON array");
        assert_eq!(array.len(), 2);
        assert_eq!(array[0]["name"], "a");
        let restored: ToolTrace = serde_json::from_value(value).unwrap();
        assert_eq!(restored, trace);
        let empty: ToolTrace = serde_json::from_str("[]").unwrap();
        assert!(empty.is_empty());
    }

    #[test]
    fn run_outcome_wraps_an_answer_and_a_trace() {
        let mut answer = BTreeMap::new();
        answer.insert("output".to_owned(), Value::String("4".to_owned()));

        let plain = RunOutcome::new(answer.clone());
        assert_eq!(plain.answer, answer);
        assert!(plain.trace.is_empty());

        let trace = ToolTrace::new([success("calc")]);
        let outcome = RunOutcome::with_trace(answer.clone(), trace.clone());
        assert_eq!(outcome.trace, trace);
        assert_eq!(outcome.into_answer(), answer);
    }

    #[test]
    fn run_outcome_json_round_trip_and_defaults() {
        let mut answer = BTreeMap::new();
        answer.insert("output".to_owned(), Value::String("4".to_owned()));
        let outcome = RunOutcome::with_trace(answer, ToolTrace::new([success("calc")]));
        let value = serde_json::to_value(&outcome).unwrap();
        let object = value.as_object().unwrap();
        for key in ["answer", "trace", "telemetry"] {
            assert!(object.contains_key(key), "missing field {key}");
        }
        assert!(object["trace"].is_array(), "the trace stays a bare array");
        let restored: RunOutcome = serde_json::from_value(value).unwrap();
        assert_eq!(restored, outcome);

        let partial: RunOutcome = serde_json::from_str(r#"{"answer": {"output": "4"}}"#).unwrap();
        assert!(partial.trace.is_empty());
        assert!(partial.telemetry.is_empty());
        assert_eq!(partial.answer.len(), 1);
    }

    #[test]
    fn telemetry_reports_absence_and_token_totals() {
        let empty = RunTelemetry::default();
        assert!(empty.is_empty());
        assert_eq!(empty.total_tokens(), None);
        let prompt_only = RunTelemetry::default().with_prompt_tokens(7);
        assert!(!prompt_only.is_empty());
        assert_eq!(prompt_only.total_tokens(), Some(7));
        let completion_only = RunTelemetry::default().with_completion_tokens(3);
        assert_eq!(completion_only.total_tokens(), Some(3));
        let both = RunTelemetry::default()
            .with_prompt_tokens(7)
            .with_completion_tokens(3);
        assert_eq!(both.total_tokens(), Some(10));
        let latency_only = RunTelemetry::default().with_latency_ms(12.5);
        assert!(!latency_only.is_empty());
        assert_eq!(latency_only.total_tokens(), None);
        assert_eq!(latency_only.latency_ms, Some(12.5));
        let priced = RunTelemetry::default().with_cost(0.25);
        assert_eq!(priced.cost, Some(0.25));
        assert_eq!(priced.total_tokens(), None);
    }

    #[test]
    fn telemetry_merge_sums_present_values() {
        let mut telemetry = RunTelemetry::default();
        telemetry.merge(RunTelemetry::default());
        assert!(telemetry.is_empty());
        telemetry.merge(RunTelemetry::default().with_latency_ms(10.0));
        assert_eq!(telemetry.latency_ms, Some(10.0));
        telemetry.merge(
            RunTelemetry::default()
                .with_latency_ms(2.5)
                .with_prompt_tokens(4)
                .with_completion_tokens(6)
                .with_cost(0.5),
        );
        assert_eq!(telemetry.latency_ms, Some(12.5));
        assert_eq!(telemetry.prompt_tokens, Some(4));
        assert_eq!(telemetry.completion_tokens, Some(6));
        assert_eq!(telemetry.total_tokens(), Some(10));
        assert_eq!(telemetry.cost, Some(0.5));
        let mut copy = RunTelemetry::default();
        copy.merge(telemetry);
        assert_eq!(copy, telemetry);
        assert_eq!(copy.total_tokens(), Some(10));
    }

    #[test]
    fn telemetry_json_round_trip_and_defaults() {
        let telemetry = RunTelemetry::default()
            .with_latency_ms(1.5)
            .with_prompt_tokens(2)
            .with_completion_tokens(3)
            .with_cost(0.004);
        let value = serde_json::to_value(telemetry).unwrap();
        assert_eq!(value["latency_ms"], json!(1.5));
        assert_eq!(value["prompt_tokens"], json!(2));
        assert_eq!(value["completion_tokens"], json!(3));
        assert_eq!(value["cost"], json!(0.004));
        let restored: RunTelemetry = serde_json::from_value(value).unwrap();
        assert_eq!(restored, telemetry);
        let partial: RunTelemetry = serde_json::from_str(r#"{"prompt_tokens": 5}"#).unwrap();
        assert_eq!(partial.prompt_tokens, Some(5));
        assert_eq!(partial.completion_tokens, None);
        assert_eq!(partial.total_tokens(), Some(5));
        let empty: RunTelemetry = serde_json::from_str("{}").unwrap();
        assert!(empty.is_empty());
        let unknown: RunTelemetry =
            serde_json::from_str(r#"{"latency_ms": 2.0, "wall": 9}"#).unwrap();
        assert_eq!(unknown, RunTelemetry::default().with_latency_ms(2.0));
    }

    #[test]
    fn run_outcome_carries_telemetry() {
        let mut answer = BTreeMap::new();
        answer.insert("output".to_owned(), Value::String("4".to_owned()));
        let plain = RunOutcome::new(answer.clone());
        assert!(plain.telemetry.is_empty());
        let measured = RunTelemetry::default().with_latency_ms(3.0);
        let outcome = RunOutcome::with_trace(answer, ToolTrace::new([success("calc")]))
            .with_telemetry(measured);
        assert_eq!(outcome.telemetry, measured);
        assert_eq!(outcome.telemetry.latency_ms, Some(3.0));
        assert_eq!(outcome.into_answer().len(), 1);
    }
}
