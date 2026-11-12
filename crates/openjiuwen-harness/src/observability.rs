//! Tracing spans/events and the telemetry sink seam.
//!
//! Python parity: `openjiuwen/harness/observability/` (run span, agent rail,
//! sub-agent hooks). The harness emits four kinds of observability signals:
//!
//! - span names, produced by the pure helpers [`run_span_name`],
//!   [`task_iteration_span_name`], [`invoke_span_name`],
//!   [`react_iteration_span_name`] and [`tool_span_name`], matching the Python
//!   `agent.<mode>.<session>` / `agent.<label>.task_iteration.<n>` /
//!   `agent.<name>.invoke` / `agent.<name>.react_iteration.<n>` /
//!   `execute_tool <tool>` hierarchy;
//! - `tracing` events on the [`TARGET`] target;
//! - redacted, non-blocking attribute records forwarded to an optional
//!   [`TelemetrySink`] implementation.
//!
//! The sink is only a seam: exporting to OpenTelemetry lives in
//! `openjiuwen-extensions`, which implements [`TelemetrySink`] over its tracer.
//! This module never blocks on I/O and never depends on an exporter.
//!
//! Secrets never reach this module: every attribute passed to a sink is already
//! reduced to non-sensitive values (names, ids, counters, booleans).

use std::fmt;
use std::sync::Arc;

use serde_json::{Map, Value};

/// Tracing target shared by every harness observability event (Python uses the
/// tracer name `openjiuwen.harness.observability`).
pub const TARGET: &str = "openjiuwen_harness::observability";

/// Sink event name for a task-loop round.
pub const EVENT_ROUND: &str = "deepagent.round";
/// Sink event name for one model call.
pub const EVENT_MODEL_CALL: &str = "deepagent.model_call";
/// Sink event name for one tool call.
pub const EVENT_TOOL_CALL: &str = "deepagent.tool_call";
/// Sink event name for one interrupt request.
pub const EVENT_INTERRUPT: &str = "deepagent.interrupt";

/// Destination for redacted observability attributes.
///
/// Implementations must be non-blocking: [`Observability`] calls `record` on the
/// task that produced the event, inside the hot loop. A sink that needs I/O
/// buffers or hands values to a channel and returns immediately.
///
/// Python parity: the exporter behind `init_observability` and the
/// `AgentObservabilityRail` span attributes.
pub trait TelemetrySink: Send + Sync {
    /// Records one event.
    ///
    /// `name` is one of the `EVENT_*` constants; `attributes` contains only
    /// already-redacted scalar values.
    fn record(&self, name: &str, attributes: &Map<String, Value>);
}

/// Sink that discards every record.
///
/// This is the default when no exporter is configured, so callers never have to
/// branch on observability being enabled.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoopTelemetry;

impl TelemetrySink for NoopTelemetry {
    fn record(&self, _name: &str, _attributes: &Map<String, Value>) {}
}

/// Emits the harness `tracing` events and forwards redacted attributes to an
/// optional [`TelemetrySink`].
///
/// The helper is cheap to clone (`Arc` sink) and safe to share across tasks.
/// Every `record_*` method both logs and forwards, so a run produces consistent
/// `tracing` output with or without an exporter attached.
#[derive(Clone, Default)]
pub struct Observability {
    sink: Option<Arc<dyn TelemetrySink>>,
}

impl Observability {
    /// Creates a helper with an optional sink; `None` behaves like
    /// [`NoopTelemetry`].
    #[must_use]
    pub const fn new(sink: Option<Arc<dyn TelemetrySink>>) -> Self {
        Self { sink }
    }

    /// Returns `true` when a sink is attached.
    #[must_use]
    pub fn has_sink(&self) -> bool {
        self.sink.is_some()
    }

    /// Records one task-loop round.
    ///
    /// `iteration` is the 1-based round number and `is_follow_up` marks rounds
    /// started from the follow-up queue rather than from the user turn.
    pub fn record_round(&self, round_id: &str, iteration: u32, is_follow_up: bool) {
        tracing::info!(
            target: TARGET,
            round_id,
            iteration,
            is_follow_up,
            "task-loop round"
        );
        let attributes = attributes([
            ("round_id", Value::from(round_id)),
            ("iteration", Value::from(iteration)),
            ("is_follow_up", Value::from(is_follow_up)),
        ]);
        self.forward(EVENT_ROUND, &attributes);
    }

    /// Records one model call and its token usage when the provider reports it.
    pub fn record_model_call(&self, agent: &str, iteration: u32, model: &str, tokens: Option<u64>) {
        tracing::debug!(
            target: TARGET,
            agent,
            iteration,
            model,
            tokens,
            "model call"
        );
        let attributes = attributes([
            ("agent", Value::from(agent)),
            ("iteration", Value::from(iteration)),
            ("model", Value::from(model)),
            ("tokens", tokens.map_or(Value::Null, Value::from)),
        ]);
        self.forward(EVENT_MODEL_CALL, &attributes);
    }

    /// Records one tool call with its outcome and wall-clock duration.
    pub fn record_tool_call(&self, tool: &str, call_id: &str, success: bool, duration_ms: u64) {
        tracing::info!(
            target: TARGET,
            tool,
            call_id,
            success,
            duration_ms,
            "tool call"
        );
        let attributes = attributes([
            ("tool", Value::from(tool)),
            ("call_id", Value::from(call_id)),
            ("success", Value::from(success)),
            ("duration_ms", Value::from(duration_ms)),
        ]);
        self.forward(EVENT_TOOL_CALL, &attributes);
    }

    /// Records one interrupt request (confirmation, ask-user, plan review, ...).
    pub fn record_interrupt(&self, kind: &str, request_id: &str) {
        tracing::info!(target: TARGET, kind, request_id, "interrupt");
        let attributes = attributes([
            ("kind", Value::from(kind)),
            ("request_id", Value::from(request_id)),
        ]);
        self.forward(EVENT_INTERRUPT, &attributes);
    }

    /// Forwards one redacted attribute map to the sink, when one is attached.
    fn forward(&self, name: &str, attributes: &Map<String, Value>) {
        if let Some(sink) = &self.sink {
            sink.record(name, attributes);
        }
    }
}

impl fmt::Debug for Observability {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Observability")
            .field("has_sink", &self.sink.is_some())
            .finish()
    }
}

/// Collects owned, redacted attributes.
fn attributes<const N: usize>(pairs: [(&'static str, Value); N]) -> Map<String, Value> {
    pairs
        .into_iter()
        .map(|(key, value)| (key.to_owned(), value))
        .collect()
}

/// Builds the root span name of a run: `agent.<mode>.<session_id>`.
///
/// Python parity: `observability.run_span.build_run_span_name` (via
/// `build_run_n`). Mode and session id are trimmed; an empty mode degrades to
/// `agent.run.<session_id>` (or `agent.run`), an empty session to
/// `agent.<mode>.run`.
#[must_use]
pub fn run_span_name(mode: &str, session_id: &str) -> String {
    let mode = mode.trim();
    let session = session_id.trim();
    match (mode.is_empty(), session.is_empty()) {
        (true, true) => "agent.run".to_owned(),
        (true, false) => format!("agent.run.{session}"),
        (false, true) => format!("agent.{mode}.run"),
        (false, false) => format!("agent.{mode}.{session}"),
    }
}

/// Builds the task-loop iteration span name: `agent.<label>.task_iteration.<n>`.
///
/// `label` is the agent display label and `iteration` the 1-based round number.
#[must_use]
pub fn task_iteration_span_name(label: &str, iteration: u32) -> String {
    format!("agent.{label}.task_iteration.{iteration}")
}

/// Builds the single-round invoke span name: `agent.<agent_name>.invoke`.
#[must_use]
pub fn invoke_span_name(agent_name: &str) -> String {
    format!("agent.{agent_name}.invoke")
}

/// Builds the `ReAct` step span name: `agent.<agent_name>.react_iteration.<n>`.
#[must_use]
pub fn react_iteration_span_name(agent_name: &str, iteration: u32) -> String {
    format!("agent.{agent_name}.react_iteration.{iteration}")
}

/// Builds the tool span name: `execute_tool <tool_name>`.
#[must_use]
pub fn tool_span_name(tool_name: &str) -> String {
    format!("execute_tool {tool_name}")
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::sync::{Arc, Mutex};

    use serde_json::{json, Map, Value};

    use super::{
        invoke_span_name, react_iteration_span_name, run_span_name, task_iteration_span_name,
        tool_span_name, NoopTelemetry, Observability, TelemetrySink, EVENT_INTERRUPT,
        EVENT_MODEL_CALL, EVENT_ROUND, EVENT_TOOL_CALL,
    };

    #[derive(Debug, Default)]
    struct RecordingSink {
        calls: Mutex<Vec<(String, Map<String, Value>)>>,
    }

    impl RecordingSink {
        fn calls(&self) -> Vec<(String, Map<String, Value>)> {
            self.calls.lock().expect("recording sink lock").clone()
        }
    }

    impl TelemetrySink for RecordingSink {
        fn record(&self, name: &str, attributes: &Map<String, Value>) {
            self.calls
                .lock()
                .expect("recording sink lock")
                .push((name.to_owned(), attributes.clone()));
        }
    }

    #[test]
    fn run_span_name_matches_python_hierarchy() {
        assert_eq!(
            run_span_name("agent.plan", "sess-1"),
            "agent.agent.plan.sess-1"
        );
        assert_eq!(
            run_span_name("code.normal", "sess-2"),
            "agent.code.normal.sess-2"
        );
        assert_eq!(run_span_name("", "sess-3"), "agent.run.sess-3");
        assert_eq!(run_span_name("agent.fast", ""), "agent.agent.fast.run");
        assert_eq!(run_span_name("", ""), "agent.run");
        assert_eq!(run_span_name("  ", "  "), "agent.run");
        assert_eq!(run_span_name(" agent.plan ", " s1 "), "agent.agent.plan.s1");
    }

    #[test]
    fn iteration_and_tool_span_names_match_python() {
        assert_eq!(
            task_iteration_span_name("deep_agent", 1),
            "agent.deep_agent.task_iteration.1"
        );
        assert_eq!(
            invoke_span_name("explore_agent"),
            "agent.explore_agent.invoke"
        );
        assert_eq!(
            react_iteration_span_name("explore_agent", 7),
            "agent.explore_agent.react_iteration.7"
        );
        assert_eq!(tool_span_name("bash"), "execute_tool bash");
        assert_eq!(tool_span_name(""), "execute_tool ");
    }

    #[test]
    fn sink_receives_redacted_attributes_for_every_event() {
        let sink = Arc::new(RecordingSink::default());
        let observability = Observability::new(Some(sink.clone()));
        assert!(observability.has_sink());

        observability.record_round("round-1", 2, true);
        observability.record_model_call("deep_agent", 2, "gpt-4o", Some(1234));
        observability.record_model_call("deep_agent", 3, "gpt-4o", None);
        observability.record_tool_call("bash", "call-9", false, 42);
        observability.record_interrupt("confirm", "req-7");

        let calls = sink.calls();
        assert_eq!(calls.len(), 5);

        let (name, attributes) = &calls[0];
        assert_eq!(name, EVENT_ROUND);
        assert_eq!(attributes.get("round_id"), Some(&json!("round-1")));
        assert_eq!(attributes.get("iteration"), Some(&json!(2)));
        assert_eq!(attributes.get("is_follow_up"), Some(&json!(true)));

        let (name, attributes) = &calls[1];
        assert_eq!(name, EVENT_MODEL_CALL);
        assert_eq!(attributes.get("agent"), Some(&json!("deep_agent")));
        assert_eq!(attributes.get("model"), Some(&json!("gpt-4o")));
        assert_eq!(attributes.get("tokens"), Some(&json!(1234)));
        assert!(!attributes.contains_key("api_key"));
        assert!(!attributes.contains_key("prompt"));

        let (_, attributes) = &calls[2];
        assert_eq!(attributes.get("tokens"), Some(&Value::Null));

        let (name, attributes) = &calls[3];
        assert_eq!(name, EVENT_TOOL_CALL);
        assert_eq!(attributes.get("tool"), Some(&json!("bash")));
        assert_eq!(attributes.get("call_id"), Some(&json!("call-9")));
        assert_eq!(attributes.get("success"), Some(&json!(false)));
        assert_eq!(attributes.get("duration_ms"), Some(&json!(42)));

        let (name, attributes) = &calls[4];
        assert_eq!(name, EVENT_INTERRUPT);
        assert_eq!(attributes.get("kind"), Some(&json!("confirm")));
        assert_eq!(attributes.get("request_id"), Some(&json!("req-7")));
    }

    #[test]
    fn noop_and_missing_sink_never_panic() {
        let noop: Arc<dyn TelemetrySink> = Arc::new(NoopTelemetry);
        noop.record(EVENT_ROUND, &Map::new());

        let observability = Observability::new(None);
        assert!(!observability.has_sink());
        observability.record_round("round-0", 0, false);
        observability.record_model_call("a", 0, "m", None);
        observability.record_tool_call("t", "c", true, 0);
        observability.record_interrupt("ask_user", "r");

        let default = Observability::default();
        default.record_round("round-0", 1, false);
        assert!(format!("{default:?}").contains("has_sink: false"));

        let with_noop = Observability::new(Some(Arc::new(NoopTelemetry)));
        with_noop.record_interrupt("plan", "req");
        assert!(with_noop.has_sink());
    }
}
