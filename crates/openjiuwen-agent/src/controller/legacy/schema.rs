//! Legacy controller value objects: tasks, dependencies, events, and sources.
//!
//! Ports `openjiuwen/core/controller/legacy/task/task.py` and
//! `openjiuwen/core/controller/legacy/event/event.py`. These types predate the
//! event-queue pipeline modelled in [`crate::controller::schema`] and stay
//! deliberately separate from it: a [`LegacyTask`] is not a
//! [`crate::controller::schema::Task`] and a [`LegacyEvent`] is not a
//! [`crate::controller::schema::Event`], so the two families must never be
//! mixed in one pipeline.
//!
//! Field names follow the Python dataclasses/pydantic models and every type
//! tolerates unknown and missing JSON fields.
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use uuid::Uuid;
/// Legacy task lifecycle.
///
/// Python models this as an `auto()` integer enum; the wire names are the
/// lowercase variant names so a legacy task round-trips through JSON.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum LegacyTaskStatus {
    /// Planned but not started (Python `PENDING`).
    #[default]
    Pending,
    /// Currently executing.
    Running,
    /// Finished successfully.
    Success,
    /// Finished with an error.
    Failed,
    /// Cancelled before finishing.
    Cancelled,
    /// Paused, waiting for user input.
    Interrupted,
}
impl LegacyTaskStatus {
    /// Returns the Python wire string of the status.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Running => "running",
            Self::Success => "success",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
            Self::Interrupted => "interrupted",
        }
    }
    /// Whether the status ends the task's lifecycle.
    ///
    /// [`Self::Interrupted`] is not terminal: an interrupted task resumes.
    #[must_use]
    pub const fn is_terminal(self) -> bool {
        matches!(self, Self::Success | Self::Failed | Self::Cancelled)
    }
}
/// Kind of edge between two legacy tasks.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DependencyType {
    /// Runs after the dependency completes.
    #[default]
    Sequential,
    /// May run concurrently once the dependencies are met.
    Parallel,
    /// Runs when the dependency's condition holds.
    Conditional,
    /// Consumes the dependency's output data.
    Data,
}
/// Directed edge of the legacy task dependency graph.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct TaskDependency {
    /// Id of the task this task depends on.
    pub dependency_id: String,
    /// Kind of the dependency.
    pub dependency_type: DependencyType,
    /// Condition expression of a [`DependencyType::Conditional`] edge.
    pub condition: Option<String>,
    /// Output mapping `{source: target}` used by a data dependency.
    pub data_mapping: BTreeMap<String, String>,
    /// Whether the dependency must be satisfied before this task runs.
    pub required: bool,
}
impl Default for TaskDependency {
    fn default() -> Self {
        Self {
            dependency_id: String::new(),
            dependency_type: DependencyType::default(),
            condition: None,
            data_mapping: BTreeMap::new(),
            required: true,
        }
    }
}
impl TaskDependency {
    /// Builds a required sequential dependency on `dependency_id`.
    #[must_use]
    pub fn new(dependency_id: impl Into<String>) -> Self {
        Self {
            dependency_id: dependency_id.into(),
            ..Self::default()
        }
    }
}
/// Unified legacy task input (a tool, workflow, or MCP call target).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct TaskInput {
    /// Id of the invoked target.
    pub target_id: String,
    /// Display name of the invoked target.
    pub target_name: String,
    /// Arguments handed to the target.
    pub arguments: Value,
}
impl TaskInput {
    /// Builds an input naming `target_id`/`target_name` without arguments.
    #[must_use]
    pub fn new(target_id: impl Into<String>, target_name: impl Into<String>) -> Self {
        Self {
            target_id: target_id.into(),
            target_name: target_name.into(),
            arguments: Value::Null,
        }
    }
    /// Sets the target arguments.
    #[must_use]
    pub fn with_arguments(mut self, arguments: Value) -> Self {
        self.arguments = arguments;
        self
    }
}
/// Outcome recorded on a [`LegacyTask`].
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct LegacyTaskResult {
    /// Status the task finished with.
    pub status: LegacyTaskStatus,
    /// Output payload of a successful task.
    pub output: Value,
    /// Failure reason of a failed task.
    pub error: Option<String>,
    /// Free-form result metadata (timings, retries, …).
    pub metadata: BTreeMap<String, Value>,
}
impl LegacyTaskResult {
    /// Builds a successful result carrying `output`.
    #[must_use]
    pub fn success(output: impl Into<Value>) -> Self {
        Self {
            status: LegacyTaskStatus::Success,
            output: output.into(),
            ..Self::default()
        }
    }
    /// Builds a failed result carrying `error`.
    #[must_use]
    pub fn failed(error: impl Into<String>) -> Self {
        Self {
            status: LegacyTaskStatus::Failed,
            error: Some(error.into()),
            ..Self::default()
        }
    }
}
/// Legacy task record, with the dependency-graph attributes Python carries.
///
/// Distinct from [`crate::controller::schema::Task`]: the new pipeline's task
/// is scheduler-oriented, while this one is a node of a `dependencies`/
/// `dependents` graph plus an optional parent/child tree.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct LegacyTask {
    /// Owning agent, when the task belongs to one.
    pub agent_id: Option<String>,
    /// Task id; empty until a caller or the intent detector names it.
    pub task_id: String,
    /// Task kind, `"undefined"` when unset.
    pub task_type: String,
    /// Human-readable description of the work.
    pub description: Option<String>,
    /// Lifecycle status.
    pub status: LegacyTaskStatus,
    /// Free-form task metadata.
    pub metadata: BTreeMap<String, Value>,
    /// Invocation target and arguments.
    pub input: TaskInput,
    /// Result of a finished task.
    pub result: Option<LegacyTaskResult>,
    /// Tasks this task depends on.
    pub dependencies: Vec<TaskDependency>,
    /// Ids of the tasks depending on this one.
    pub dependents: BTreeSet<String>,
    /// Parent task id, for subtasks.
    pub parent_task_id: Option<String>,
    /// Ids of this task's subtasks.
    pub child_task_ids: BTreeSet<String>,
    /// Task group the task belongs to.
    pub group_id: Option<String>,
    /// Depth in the dependency graph; `0` is a root.
    pub level: u32,
}
impl Default for LegacyTask {
    fn default() -> Self {
        Self {
            agent_id: None,
            task_id: String::new(),
            task_type: Self::UNDEFINED_TASK_TYPE.to_owned(),
            description: None,
            status: LegacyTaskStatus::Pending,
            metadata: BTreeMap::new(),
            input: TaskInput::default(),
            result: None,
            dependencies: Vec::new(),
            dependents: BTreeSet::new(),
            parent_task_id: None,
            child_task_ids: BTreeSet::new(),
            group_id: None,
            level: 0,
        }
    }
}
impl LegacyTask {
    /// Task type Python's `TaskType.UNDEFINED` serializes to.
    pub const UNDEFINED_TASK_TYPE: &'static str = "undefined";
    /// Builds a pending, level-0 task of `task_type`.
    #[must_use]
    pub fn new(task_type: impl Into<String>) -> Self {
        Self {
            task_type: task_type.into(),
            ..Self::default()
        }
    }
    /// Sets the task description.
    #[must_use]
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }
    /// Sets the invocation target.
    #[must_use]
    pub fn with_input(mut self, input: TaskInput) -> Self {
        self.input = input;
        self
    }
    /// Inserts one metadata entry.
    #[must_use]
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<Value>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
    /// Sets the owning agent id.
    pub fn set_agent_id(&mut self, agent_id: impl Into<String>) {
        self.agent_id = Some(agent_id.into());
    }
    /// Whether the task's status ends its lifecycle.
    #[must_use]
    pub fn is_terminal(&self) -> bool {
        self.status.is_terminal()
    }
}
/// Kind of a legacy controller event.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LegacyEventType {
    /// User input event.
    #[default]
    UserInput,
    /// Agent answer event.
    AgentResponse,
    /// Hand-off of a conversation to another agent.
    AgentHandoff,
    /// A task finished.
    TaskCompleted,
    /// A task paused for user input.
    TaskInterrupted,
    /// Error notification.
    Error,
    /// Informational notification.
    Info,
}
impl LegacyEventType {
    /// Returns the Python wire string of the event kind.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::UserInput => "user_input",
            Self::AgentResponse => "agent_response",
            Self::AgentHandoff => "agent_handoff",
            Self::TaskCompleted => "task_completed",
            Self::TaskInterrupted => "task_interrupted",
            Self::Error => "error",
            Self::Info => "info",
        }
    }
}
/// Legacy event priority, matching Python's `1..=4` integers.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventPriority {
    /// `1`: processed last.
    Low,
    /// `2`: the default priority.
    #[default]
    Normal,
    /// `3`: processed before normal events.
    High,
    /// `4`: processed first.
    Urgent,
}
impl EventPriority {
    /// Returns the Python integer of the priority.
    #[must_use]
    pub const fn value(self) -> u8 {
        match self {
            Self::Low => 1,
            Self::Normal => 2,
            Self::High => 3,
            Self::Urgent => 4,
        }
    }
}
/// What produced an event.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceType {
    /// The user.
    #[default]
    User,
    /// A single agent; Python's wire value is `"single_agent"`.
    #[serde(rename = "single_agent")]
    Agent,
    /// A task signal.
    Task,
    /// A workflow signal.
    Workflow,
    /// The runtime itself.
    System,
}
/// Origin of a [`LegacyEvent`].
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct EventSource {
    /// Conversation the event belongs to.
    pub conversation_id: String,
    /// Kind of the producer.
    pub source_type: SourceType,
    /// User id, when the producer is a user.
    pub user_id: Option<String>,
}
impl EventSource {
    /// Builds a user source for `conversation_id`.
    #[must_use]
    pub fn user(conversation_id: impl Into<String>, user_id: Option<String>) -> Self {
        Self {
            conversation_id: conversation_id.into(),
            source_type: SourceType::User,
            user_id,
        }
    }
}
/// Payload of a [`LegacyEvent`]: text, interactive resume input, stream data,
/// task results, and extension fields.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct EventContent {
    /// Query text.
    pub query: Option<String>,
    /// Interactive resume payload (Python's `InteractiveInput`).
    pub interactive_input: Option<Value>,
    /// Streamed output frames (Python's `OutputSchema` list).
    pub stream_data: Vec<Value>,
    /// Task result payload, when the event reports one.
    pub task_result: Option<Value>,
    /// Extension fields Python keeps for truly uncertain data.
    pub extensions: BTreeMap<String, Value>,
}
impl EventContent {
    /// Builds text content carrying `query`.
    #[must_use]
    pub fn text(query: impl Into<String>) -> Self {
        Self {
            query: Some(query.into()),
            ..Self::default()
        }
    }
    /// Returns the query text of any supported content shape.
    ///
    /// Prefers [`query`](Self::query); otherwise extracts `raw_inputs`, then
    /// the first `user_inputs` value of an interactive payload; `""` when
    /// neither is present. Non-string values are rendered as JSON text, which
    /// mirrors Python's `str(...)` rendering.
    #[must_use]
    pub fn get_query(&self) -> String {
        if let Some(query) = &self.query {
            return query.clone();
        }
        let Some(interactive) = &self.interactive_input else {
            return String::new();
        };
        if let Some(raw) = interactive.get("raw_inputs").filter(|raw| !raw.is_null()) {
            return value_text(raw);
        }
        interactive
            .get("user_inputs")
            .and_then(Value::as_object)
            .and_then(|inputs| inputs.values().next())
            .map_or_else(String::new, value_text)
    }
    /// Returns the extension field named `key`.
    #[must_use]
    pub fn extension(&self, key: &str) -> Option<&Value> {
        self.extensions.get(key)
    }
}
/// Renders a JSON value as text: strings verbatim, everything else as JSON.
fn value_text(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        other => other.to_string(),
    }
}
/// Event-chain context: correlation, conversation, task, and workflow ids.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct EventContext {
    /// Correlation id used to track a chain of events.
    pub correlation_id: Option<String>,
    /// Conversation the event belongs to.
    pub conversation_id: Option<String>,
    /// Task the event relates to.
    pub task_id: Option<String>,
    /// Workflow the event relates to.
    pub workflow_id: Option<String>,
}
/// Legacy controller event.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct LegacyEvent {
    /// Unique event id.
    pub id: String,
    /// Event kind.
    pub event_type: LegacyEventType,
    /// Processing priority.
    pub priority: EventPriority,
    /// Producer of the event.
    pub source: EventSource,
    /// Event payload.
    pub content: EventContent,
    /// Event-chain context.
    pub context: EventContext,
    /// Free-form event metadata.
    pub metadata: BTreeMap<String, Value>,
    /// Target agent id for point-to-point routing.
    pub receiver_id: Option<String>,
    /// Custom event kind used by subscription routing.
    pub custom_event_type: Option<String>,
}
impl Default for LegacyEvent {
    fn default() -> Self {
        Self {
            id: new_event_id(),
            event_type: LegacyEventType::UserInput,
            priority: EventPriority::Normal,
            source: EventSource {
                conversation_id: "unknown".to_owned(),
                source_type: SourceType::System,
                user_id: None,
            },
            content: EventContent::default(),
            context: EventContext::default(),
            metadata: BTreeMap::new(),
            receiver_id: None,
            custom_event_type: None,
        }
    }
}
/// Mints an event id, matching Python's `str(uuid.uuid4())`.
fn new_event_id() -> String {
    Uuid::new_v4().to_string()
}
impl LegacyEvent {
    /// Builds a normal-priority event of `event_type` with a fresh id.
    #[must_use]
    pub fn new(event_type: LegacyEventType) -> Self {
        Self {
            event_type,
            ..Self::default()
        }
    }
    /// Builds a user input event on the `default_session` conversation.
    #[must_use]
    pub fn user_input(query: impl Into<String>) -> Self {
        let conversation = "default_session".to_owned();
        let mut event = Self::new(LegacyEventType::UserInput);
        event.source = EventSource::user(conversation.clone(), None);
        event.content = EventContent::text(query);
        event.context.conversation_id = Some(conversation);
        event.context.correlation_id = Some(new_event_id());
        event
    }
    /// Whether the event came from a user.
    #[must_use]
    pub fn is_from_user(&self) -> bool {
        self.source.source_type == SourceType::User
    }
    /// Whether the event came from an agent.
    #[must_use]
    pub fn is_from_agent(&self) -> bool {
        self.source.source_type == SourceType::Agent
    }
    /// Whether the event reports a task signal.
    #[must_use]
    pub fn is_task_related(&self) -> bool {
        matches!(
            self.event_type,
            LegacyEventType::TaskCompleted | LegacyEventType::TaskInterrupted
        )
    }
    /// Whether the event carries a workflow id.
    #[must_use]
    pub fn is_workflow_related(&self) -> bool {
        self.context.workflow_id.is_some()
    }
    /// Sets the correlation id.
    pub fn set_correlation(&mut self, correlation_id: impl Into<String>) {
        self.context.correlation_id = Some(correlation_id.into());
    }
    /// Sets the conversation id of the event context.
    pub fn set_conversation(&mut self, conversation_id: impl Into<String>) {
        self.context.conversation_id = Some(conversation_id.into());
    }
    /// Returns the query text of the event payload.
    #[must_use]
    pub fn get_query(&self) -> String {
        self.content.get_query()
    }
    /// Projects the event to its JSON wire form.
    ///
    /// A value that cannot be serialized (impossible for this field set) maps
    /// to [`Value::Null`].
    #[must_use]
    pub fn to_dict(&self) -> Value {
        serde_json::to_value(self).unwrap_or(Value::Null)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    fn dependency_graph_task() -> LegacyTask {
        let mut task = LegacyTask::new(LegacyTask::UNDEFINED_TASK_TYPE)
            .with_description("demo task")
            .with_input(TaskInput::new("tool-1", "echo").with_arguments(json!({"text": "hi"})))
            .with_metadata("task_source", "test");
        task.task_id = "t1".to_owned();
        task.dependencies = vec![TaskDependency::new("t0")];
        task.dependents.insert("t2".to_owned());
        task.child_task_ids.insert("t1.1".to_owned());
        task
    }
    #[test]
    fn legacy_task_round_trips_with_python_wire_names() {
        let task = dependency_graph_task();
        let value = serde_json::to_value(&task).unwrap();
        assert_eq!(value["status"], json!("pending"));
        assert_eq!(value["task_type"], json!("undefined"));
        assert_eq!(value["dependents"], json!(["t2"]));
        assert_eq!(value["child_task_ids"], json!(["t1.1"]));
        assert_eq!(
            value["dependencies"][0]["dependency_type"],
            json!("sequential")
        );
        assert_eq!(value["input"]["target_id"], json!("tool-1"));
        let restored: LegacyTask = serde_json::from_value(value).unwrap();
        assert_eq!(restored, task);
    }
    #[test]
    fn legacy_task_defaults_and_unknown_fields() {
        let restored: LegacyTask =
            serde_json::from_value(json!({"task_id": "t9", "unknown": true})).unwrap();
        assert_eq!(restored.task_id, "t9");
        assert_eq!(restored.task_type, LegacyTask::UNDEFINED_TASK_TYPE);
        assert_eq!(restored.status, LegacyTaskStatus::Pending);
        assert_eq!(restored.level, 0);
        assert_eq!(restored.dependencies, Vec::new());
        assert_eq!(LegacyTask::new("workflow").task_type, "workflow");
    }
    #[test]
    fn legacy_task_status_strings_and_terminality() {
        assert_eq!(LegacyTaskStatus::default(), LegacyTaskStatus::Pending);
        assert_eq!(LegacyTaskStatus::Success.as_str(), "success");
        assert_eq!(LegacyTaskStatus::Cancelled.as_str(), "cancelled");
        assert!(LegacyTaskStatus::Success.is_terminal());
        assert!(LegacyTaskStatus::Failed.is_terminal());
        assert!(LegacyTaskStatus::Cancelled.is_terminal());
        assert!(!LegacyTaskStatus::Pending.is_terminal());
        assert!(!LegacyTaskStatus::Running.is_terminal());
        assert!(!LegacyTaskStatus::Interrupted.is_terminal());
        let mut task = LegacyTask::new("workflow");
        assert!(!task.is_terminal());
        task.status = LegacyTaskStatus::Success;
        assert!(task.is_terminal());
    }
    #[test]
    fn task_dependency_defaults_to_required_sequential() {
        let dependency = TaskDependency::new("t0");
        assert_eq!(dependency.dependency_type, DependencyType::Sequential);
        assert!(dependency.required);
        assert!(dependency.data_mapping.is_empty());
        assert!(TaskDependency::default().required);
        let restored: TaskDependency =
            serde_json::from_value(json!({"dependency_id": "t7"})).unwrap();
        assert_eq!(restored, TaskDependency::new("t7"));
        let data = TaskDependency {
            dependency_type: DependencyType::Data,
            data_mapping: BTreeMap::from([("out".to_owned(), "in".to_owned())]),
            ..TaskDependency::new("t8")
        };
        let value = serde_json::to_value(&data).unwrap();
        assert_eq!(value["dependency_type"], json!("data"));
        assert_eq!(value["data_mapping"]["out"], json!("in"));
        assert_eq!(
            serde_json::from_value::<TaskDependency>(value).unwrap(),
            data
        );
    }
    #[test]
    fn task_results_carry_python_statuses() {
        let success = LegacyTaskResult::success(json!({"answer": 42}));
        assert_eq!(success.status, LegacyTaskStatus::Success);
        assert_eq!(success.output["answer"], json!(42));
        assert!(success.error.is_none());
        let failed = LegacyTaskResult::failed("boom");
        assert_eq!(failed.status, LegacyTaskStatus::Failed);
        assert_eq!(failed.error.as_deref(), Some("boom"));
        assert_eq!(
            serde_json::to_value(&failed).unwrap()["status"],
            json!("failed")
        );
    }
    #[test]
    fn task_input_round_trips() {
        let input = TaskInput::new("wf", "workflow").with_arguments(json!({"a": 1}));
        let restored: TaskInput =
            serde_json::from_value(serde_json::to_value(&input).unwrap()).unwrap();
        assert_eq!(restored, input);
        assert_eq!(TaskInput::default().target_id, "");
    }
    #[test]
    fn legacy_event_round_trips_with_python_wire_names() {
        let event = LegacyEvent::user_input("hello");
        assert_eq!(event.event_type, LegacyEventType::UserInput);
        assert_eq!(event.priority, EventPriority::Normal);
        let value = event.to_dict();
        assert_eq!(value["event_type"], json!("user_input"));
        assert_eq!(value["priority"], json!("normal"));
        assert_eq!(value["source"]["source_type"], json!("user"));
        assert_eq!(value["content"]["query"], json!("hello"));
        assert!(Uuid::parse_str(&event.id).is_ok(), "a fresh uuid id");
        let restored: LegacyEvent = serde_json::from_value(value).unwrap();
        assert_eq!(restored, event);
    }
    #[test]
    fn agent_events_use_the_single_agent_wire_value() {
        let mut event = LegacyEvent::new(LegacyEventType::AgentResponse);
        event.source = EventSource {
            conversation_id: "c1".to_owned(),
            source_type: SourceType::Agent,
            user_id: None,
        };
        event.priority = EventPriority::High;
        let value = event.to_dict();
        assert_eq!(value["source"]["source_type"], json!("single_agent"));
        assert_eq!(value["event_type"], json!("agent_response"));
        assert_eq!(value["priority"], json!("high"));
        let restored: LegacyEvent = serde_json::from_value(value).unwrap();
        assert_eq!(restored.source.source_type, SourceType::Agent);
        assert!(restored.is_from_agent());
        assert!(!restored.is_from_user());
    }
    #[test]
    fn legacy_event_defaults_and_unknown_fields() {
        let event: LegacyEvent = serde_json::from_value(json!({"extra": 1})).unwrap();
        assert_eq!(event.event_type, LegacyEventType::UserInput);
        assert_eq!(event.source.source_type, SourceType::System);
        assert_eq!(event.source.conversation_id, "unknown");
        assert!(Uuid::parse_str(&event.id).is_ok(), "a fresh uuid id");
        assert!(event.metadata.is_empty());
    }
    #[test]
    fn event_priority_values_match_python_ints() {
        assert_eq!(EventPriority::default().value(), 2);
        assert_eq!(EventPriority::Low.value(), 1);
        assert_eq!(EventPriority::High.value(), 3);
        assert_eq!(EventPriority::Urgent.value(), 4);
        assert!(EventPriority::Low < EventPriority::Urgent);
        assert_eq!(
            LegacyEventType::TaskInterrupted.as_str(),
            "task_interrupted"
        );
        assert_eq!(LegacyEventType::Info.as_str(), "info");
    }
    #[test]
    fn event_predicates_and_context_setters() {
        let mut event = LegacyEvent::new(LegacyEventType::TaskCompleted);
        assert!(event.is_task_related());
        assert!(!event.is_workflow_related());
        event.context.workflow_id = Some("wf-1".to_owned());
        assert!(event.is_workflow_related());
        event.set_correlation("corr");
        event.set_conversation("conv");
        assert_eq!(event.context.correlation_id.as_deref(), Some("corr"));
        assert_eq!(event.context.conversation_id.as_deref(), Some("conv"));
        assert!(!LegacyEvent::new(LegacyEventType::Info).is_task_related());
        assert!(LegacyEvent::user_input("hi").is_from_user());
    }
    #[test]
    fn event_content_get_query_prefers_query_then_interactive_input() {
        assert_eq!(EventContent::text("direct").get_query(), "direct");
        assert_eq!(EventContent::default().get_query(), "");
        let interactive = EventContent {
            interactive_input: Some(json!({"user_inputs": {"call-1": "yes"}})),
            ..EventContent::default()
        };
        assert_eq!(interactive.get_query(), "yes");
        let raw = EventContent {
            interactive_input: Some(json!({"raw_inputs": "raw text"})),
            ..EventContent::default()
        };
        assert_eq!(raw.get_query(), "raw text");
        let structured = EventContent {
            interactive_input: Some(json!({"user_inputs": {"call-1": {"approved": true}}})),
            ..EventContent::default()
        };
        assert_eq!(
            structured.get_query(),
            json!({"approved": true}).to_string()
        );
        let empty = EventContent {
            interactive_input: Some(json!({"user_inputs": {}})),
            ..EventContent::default()
        };
        assert_eq!(empty.get_query(), "");
    }
    #[test]
    fn event_content_extensions_are_addressable() {
        let content = EventContent {
            extensions: BTreeMap::from([("to_agent_id".to_owned(), json!("a2"))]),
            ..EventContent::default()
        };
        assert_eq!(content.extension("to_agent_id"), Some(&json!("a2")));
        assert_eq!(content.extension("missing"), None);
    }
    #[test]
    fn event_source_helper_and_round_trip() {
        let source = EventSource::user("c1", Some("u1".to_owned()));
        assert_eq!(source.source_type, SourceType::User);
        let value = serde_json::to_value(&source).unwrap();
        assert_eq!(value["conversation_id"], json!("c1"));
        assert_eq!(
            serde_json::from_value::<EventSource>(value).unwrap(),
            source
        );
        assert_eq!(
            SourceType::default(),
            SourceType::User,
            "user is the default source kind"
        );
    }
}
