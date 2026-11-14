//! Controller configuration.
//!
//! Ports `openjiuwen/core/controller/config.py`. [`ControllerConfig`] carries
//! Python's scheduling, task-management, event-queue, and intent-recognition
//! knobs; the Rust port adds the routing fields the controller loop needs
//! (`max_intents`, `max_tasks`, `history_limit`, `default_task_type`).
use super::schema::IntentType;
use serde::{Deserialize, Serialize};
/// Fallback answer when no task ran or no executor produced output.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct DefaultResponse {
    /// Response text; `None` leaves the controller without a fallback.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
}
impl DefaultResponse {
    /// Builds an unset default response.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
    /// Builds a default response carrying `text`.
    #[must_use]
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            text: Some(text.into()),
        }
    }
    /// Whether a fallback text is configured.
    #[must_use]
    pub fn is_set(&self) -> bool {
        self.text.as_deref().is_some_and(|text| !text.is_empty())
    }
}
/// Default intent budget per invocation.
fn default_max_intents() -> usize {
    16
}
/// Default task budget per invocation.
fn default_max_tasks() -> usize {
    5
}
/// Default number of history entries kept.
fn default_history_limit() -> usize {
    50
}
/// Default task type routed to when the intent names none.
fn default_task_type() -> String {
    "agent".to_owned()
}
/// Default event queue capacity.
fn default_event_queue_size() -> usize {
    10_000
}
/// Default number of concurrently running tasks.
fn default_max_concurrent_tasks() -> usize {
    5
}
/// Default scheduler scan interval in milliseconds.
fn default_schedule_interval_ms() -> u64 {
    1000
}
/// Default task priority.
fn default_task_priority() -> i32 {
    1
}
/// Default event processing timeout in milliseconds.
fn default_event_timeout_ms() -> u64 {
    300_000
}
/// Default intent confidence threshold.
fn default_intent_confidence_threshold() -> f32 {
    0.7
}
/// Default recognized intent kinds (Python's list).
fn default_intent_type_list() -> Vec<IntentType> {
    vec![
        IntentType::CreateTask,
        IntentType::PauseTask,
        IntentType::ResumeTask,
        IntentType::CancelTask,
        IntentType::UnknownTask,
    ]
}
/// Default wait for the first stream frame in milliseconds.
fn default_stream_first_frame_timeout_ms() -> u64 {
    30_000
}
/// Controller behaviour knobs.
///
/// Field names and defaults follow Python's `ControllerConfig`; omitted JSON
/// fields fall back to the same values as [`Default`].
// Python parity fixes the boolean field set, which trips the lint's threshold.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ControllerConfig {
    /// Maximum number of intents recognized from one input (Rust routing).
    #[serde(default = "default_max_intents")]
    pub max_intents: usize,
    /// Maximum number of tasks scheduled from one input (Rust routing).
    #[serde(default = "default_max_tasks")]
    pub max_tasks: usize,
    /// Number of conversation entries kept for context (Rust routing).
    #[serde(default = "default_history_limit")]
    pub history_limit: usize,
    /// Answer used when nothing ran.
    #[serde(default)]
    pub default_response: DefaultResponse,
    /// Whether reasoning frames are exposed to the caller.
    #[serde(default)]
    pub expose_reasoning: bool,
    /// Task type used when an intent names none.
    #[serde(default = "default_task_type")]
    pub default_task_type: String,
    /// Event queue capacity (Python default `10000`).
    #[serde(default = "default_event_queue_size")]
    pub event_queue_size: usize,
    /// Maximum number of concurrently running tasks; `0` means unlimited.
    #[serde(default = "default_max_concurrent_tasks")]
    pub max_concurrent_tasks: usize,
    /// Scheduler scan interval in milliseconds (Python `1.0` seconds).
    #[serde(default = "default_schedule_interval_ms")]
    pub schedule_interval_ms: u64,
    /// Per-task timeout in milliseconds; `None` means no timeout.
    #[serde(default)]
    pub task_timeout_ms: Option<u64>,
    /// Priority assigned to tasks created without one.
    #[serde(default = "default_task_priority")]
    pub default_task_priority: i32,
    /// Whether task states are persisted for recovery.
    #[serde(default = "default_enable_task_persistence")]
    pub enable_task_persistence: bool,
    /// Event processing timeout in milliseconds (Python `300` seconds).
    #[serde(default = "default_event_timeout_ms")]
    pub event_timeout_ms: u64,
    /// Whether user intent is recognized at all.
    #[serde(default = "default_enable_intent_recognition")]
    pub enable_intent_recognition: bool,
    /// Confidence below which recognition degrades to `unknown_task`.
    #[serde(default = "default_intent_confidence_threshold")]
    pub intent_confidence_threshold: f32,
    /// Intent kinds this controller accepts.
    #[serde(default = "default_intent_type_list")]
    pub intent_type_list: Vec<IntentType>,
    /// Whether the `all_tasks_processed` signal is suppressed.
    #[serde(default)]
    pub suppress_completion_signal: bool,
    /// Maximum wait for the first stream frame in milliseconds.
    #[serde(default = "default_stream_first_frame_timeout_ms")]
    pub stream_first_frame_timeout_ms: u64,
}
/// Default of [`ControllerConfig::enable_intent_recognition`] (`true`).
fn default_enable_intent_recognition() -> bool {
    true
}
/// Default of [`ControllerConfig::enable_task_persistence`] (`true`).
fn default_enable_task_persistence() -> bool {
    true
}
impl Default for ControllerConfig {
    fn default() -> Self {
        Self {
            max_intents: default_max_intents(),
            max_tasks: default_max_tasks(),
            history_limit: default_history_limit(),
            default_response: DefaultResponse::default(),
            expose_reasoning: false,
            default_task_type: default_task_type(),
            event_queue_size: default_event_queue_size(),
            max_concurrent_tasks: default_max_concurrent_tasks(),
            schedule_interval_ms: default_schedule_interval_ms(),
            task_timeout_ms: None,
            default_task_priority: default_task_priority(),
            enable_task_persistence: default_enable_task_persistence(),
            event_timeout_ms: default_event_timeout_ms(),
            enable_intent_recognition: default_enable_intent_recognition(),
            intent_confidence_threshold: default_intent_confidence_threshold(),
            intent_type_list: default_intent_type_list(),
            suppress_completion_signal: false,
            stream_first_frame_timeout_ms: default_stream_first_frame_timeout_ms(),
        }
    }
}
/// Builder helpers for the controller's behaviour knobs.
///
/// Dependency gating has no builder here because it has no field: the
/// background scheduler reads the `depends_on` array off each task's metadata
/// (`metadata["depends_on"]`), so a host configures it per task, not per
/// controller.
impl ControllerConfig {
    /// Builds a configuration with the Python defaults; same as [`Default`].
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
    /// Sets the intent budget per invocation.
    #[must_use]
    pub fn with_max_intents(mut self, max_intents: usize) -> Self {
        self.max_intents = max_intents;
        self
    }
    /// Sets the task budget per invocation.
    #[must_use]
    pub fn with_max_tasks(mut self, max_tasks: usize) -> Self {
        self.max_tasks = max_tasks;
        self
    }
    /// Sets the task type used when an intent names none.
    #[must_use]
    pub fn with_default_task_type(mut self, task_type: impl Into<String>) -> Self {
        self.default_task_type = task_type.into();
        self
    }
    /// Sets the fallback answer.
    #[must_use]
    pub fn with_default_response(mut self, response: DefaultResponse) -> Self {
        self.default_response = response;
        self
    }
    /// Sets whether reasoning frames are exposed.
    #[must_use]
    pub fn with_expose_reasoning(mut self, expose_reasoning: bool) -> Self {
        self.expose_reasoning = expose_reasoning;
        self
    }
    /// Replaces the accepted intent kinds.
    #[must_use]
    pub fn with_intent_type_list(mut self, intent_type_list: Vec<IntentType>) -> Self {
        self.intent_type_list = intent_type_list;
        self
    }
    /// Sets whether task states are persisted.
    #[must_use]
    pub fn with_enable_task_persistence(mut self, enable: bool) -> Self {
        self.enable_task_persistence = enable;
        self
    }
    /// Sets whether the `all_tasks_processed` signal is suppressed.
    #[must_use]
    pub fn with_suppress_completion_signal(mut self, suppress: bool) -> Self {
        self.suppress_completion_signal = suppress;
        self
    }
    /// Sets how many tasks the background scheduler runs at once (`0` =
    /// unlimited).
    #[must_use]
    pub fn with_max_concurrent_tasks(mut self, max_concurrent_tasks: usize) -> Self {
        self.max_concurrent_tasks = max_concurrent_tasks;
        self
    }
    /// Sets the background scheduler's scan interval in milliseconds.
    ///
    /// A submit or a completion wakes the loop sooner; this is the longest it
    /// waits between scans.
    #[must_use]
    pub fn with_schedule_interval(mut self, schedule_interval_ms: u64) -> Self {
        self.schedule_interval_ms = schedule_interval_ms;
        self
    }
    /// Sets the per-task timeout in milliseconds; `None` disables it.
    #[must_use]
    pub fn with_task_timeout(mut self, task_timeout_ms: Option<u64>) -> Self {
        self.task_timeout_ms = task_timeout_ms;
        self
    }
    /// Whether `intent_type` is in [`intent_type_list`](Self::intent_type_list).
    #[must_use]
    pub fn accepts(&self, intent_type: IntentType) -> bool {
        self.intent_type_list.contains(&intent_type)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn default_equals_parsed_empty_object() {
        let parsed: ControllerConfig = serde_json::from_value(json!({})).unwrap();
        assert_eq!(parsed, ControllerConfig::default());
        assert_eq!(parsed.max_intents, 16);
        assert_eq!(parsed.max_tasks, 5);
        assert_eq!(parsed.default_task_type, "agent");
        assert!(parsed.enable_task_persistence);
        assert!(parsed.enable_intent_recognition);
        assert!((parsed.intent_confidence_threshold - 0.7).abs() < f32::EPSILON);
        assert_eq!(parsed.intent_type_list.len(), 5);
    }
    #[test]
    fn partial_json_keeps_the_other_defaults() {
        let parsed: ControllerConfig =
            serde_json::from_value(json!({"max_tasks": 2, "expose_reasoning": true})).unwrap();
        assert_eq!(parsed.max_tasks, 2);
        assert!(parsed.expose_reasoning);
        assert_eq!(parsed.max_intents, 16);
        assert_eq!(parsed.history_limit, 50);
        assert!(parsed.task_timeout_ms.is_none());
    }
    #[test]
    fn wire_names_are_snake_case() {
        let wire = serde_json::to_value(ControllerConfig::default()).unwrap();
        assert_eq!(wire["default_task_type"], json!("agent"));
        assert_eq!(wire["intent_type_list"][0], json!("create_task"));
        assert_eq!(wire["schedule_interval_ms"], json!(1000));
        let threshold = wire["intent_confidence_threshold"].as_f64().unwrap();
        assert!((threshold - 0.7).abs() < 1e-6);
    }
    #[test]
    fn accepts_checks_the_intent_list() {
        let config = ControllerConfig::new();
        assert!(config.accepts(IntentType::CreateTask));
        assert!(!config.accepts(IntentType::ModifyTask));
        let restricted = config.with_intent_type_list(vec![IntentType::CancelTask]);
        assert!(restricted.accepts(IntentType::CancelTask));
        assert!(!restricted.accepts(IntentType::CreateTask));
    }
    #[test]
    fn scheduler_builders_set_their_fields() {
        let config = ControllerConfig::new()
            .with_max_concurrent_tasks(1)
            .with_schedule_interval(25)
            .with_task_timeout(Some(500));
        assert_eq!(config.max_concurrent_tasks, 1);
        assert_eq!(config.schedule_interval_ms, 25);
        assert_eq!(config.task_timeout_ms, Some(500));
        assert_eq!(config.clone().with_task_timeout(None).task_timeout_ms, None);
        let wire = serde_json::to_value(&config).unwrap();
        assert_eq!(wire["max_concurrent_tasks"], json!(1));
        assert_eq!(wire["schedule_interval_ms"], json!(25));
        assert_eq!(wire["task_timeout_ms"], json!(500));
    }

    #[test]
    fn default_response_helpers() {
        let unset = DefaultResponse::new();
        assert!(!unset.is_set());
        assert_eq!(serde_json::to_value(&unset).unwrap(), json!({}));
        let set = DefaultResponse::text("nothing to do");
        assert!(set.is_set());
        assert_eq!(
            serde_json::to_value(&set).unwrap(),
            json!({"text": "nothing to do"})
        );
        let config = ControllerConfig::new().with_default_response(set.clone());
        assert_eq!(config.default_response, set);
    }
}
