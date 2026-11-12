//! The transport-agnostic client seam plus the tool catalogue each scope advertises.
//!
//! Python parity sources: the operation methods of `openjiuwen/agent_teams/external/client.py` and
//! the tool definitions of `openjiuwen/agent_teams/tools/tool_factory.py` (`create_team_tools`)
//! and `openjiuwen/agent_teams/mcp/server.py` (`_operator_tool_defs`).
//!
//! The module holds no state and opens no connection: [`TeamClient`] is the single seam the
//! `team-member` CLI dispatches every command through, so the same command tree can be driven from
//! an in-memory store (tests) or a live runtime bridge. [`member_tools_for`] and [`operator_tools`]
//! describe the two tool surfaces an external process is given; the definitions are wire-visible
//! (they are advertised over MCP), so names, order, schemas and `required` lists must match Python.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use openjiuwen_common::error::Result;

use crate::schema::{
    DispatchMode, MemberMode, MemberStatus, TaskDetail, TaskStatus, TaskSummary, TeamRole,
};
use crate::taskboard::TaskOpResult;

use super::descriptor::ClientScope;

/// Result of a `send_message` operation.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct SendOutcome {
    /// Whether the message was accepted by the transport and persisted.
    pub ok: bool,
    /// Identity of the persisted message row, when it was accepted.
    pub message_id: Option<String>,
    /// Why the send was refused, when [`ok`](Self::ok) is `false`.
    pub reason: Option<String>,
}

impl SendOutcome {
    /// Returns an accepted outcome carrying the persisted `message_id`.
    #[must_use]
    pub fn sent(message_id: impl Into<String>) -> Self {
        Self {
            ok: true,
            message_id: Some(message_id.into()),
            reason: None,
        }
    }

    /// Returns a refused outcome carrying `reason`.
    #[must_use]
    pub fn failed(reason: impl Into<String>) -> Self {
        Self {
            ok: false,
            message_id: None,
            reason: Some(reason.into()),
        }
    }
}

/// Result of a `create_task` operation.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct CreateOutcome {
    /// Whether the task row was created.
    pub ok: bool,
    /// Identity of the created task, when it was created.
    pub task_id: Option<String>,
    /// Why the creation was refused, when [`ok`](Self::ok) is `false`.
    pub reason: Option<String>,
}

impl CreateOutcome {
    /// Returns a created outcome carrying the new `task_id`.
    #[must_use]
    pub fn created(task_id: impl Into<String>) -> Self {
        Self {
            ok: true,
            task_id: Some(task_id.into()),
            reason: None,
        }
    }

    /// Returns a refused outcome carrying `reason`.
    #[must_use]
    pub fn failed(reason: impl Into<String>) -> Self {
        Self {
            ok: false,
            task_id: None,
            reason: Some(reason.into()),
        }
    }
}

/// Result of an operation that either applies or is refused with a reason.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct OpOutcome {
    /// Whether the operation was applied.
    pub ok: bool,
    /// Why the operation was refused, when [`ok`](Self::ok) is `false`.
    pub reason: Option<String>,
}

impl OpOutcome {
    /// Returns a successful outcome.
    #[must_use]
    pub const fn ok() -> Self {
        Self {
            ok: true,
            reason: None,
        }
    }

    /// Returns a refused outcome carrying `reason`.
    #[must_use]
    pub fn failed(reason: impl Into<String>) -> Self {
        Self {
            ok: false,
            reason: Some(reason.into()),
        }
    }
}

impl From<TaskOpResult> for OpOutcome {
    /// Lifts a board refusal (claim, assign, …) into the CLI's outcome type.
    fn from(result: TaskOpResult) -> Self {
        Self {
            ok: result.ok,
            reason: result.reason,
        }
    }
}

/// One roster row as the CLI renders it.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct MemberSummary {
    /// Member name (the addressing key).
    pub member_name: String,
    /// Role the member plays in the team.
    pub role: TeamRole,
    /// Current lifecycle status of the member.
    pub status: MemberStatus,
}

impl MemberSummary {
    /// Builds a roster row.
    #[must_use]
    pub fn new(member_name: impl Into<String>, role: TeamRole, status: MemberStatus) -> Self {
        Self {
            member_name: member_name.into(),
            role,
            status,
        }
    }
}

/// One tool definition advertised to an external agent.
///
/// `input_schema` is the raw JSON Schema the native team tool publishes as `card.input_params`, so
/// an external caller sees exactly the schema an in-process teammate sees.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct TeamToolDef {
    /// Tool name (the stable call key).
    pub name: String,
    /// One-line description shown to the model.
    pub description: String,
    /// JSON Schema of the tool arguments.
    pub input_schema: Value,
}

impl TeamToolDef {
    /// Builds a tool definition.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        input_schema: Value,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            input_schema,
        }
    }
}

/// The operations the `team-member` CLI performs against a team.
///
/// Every implementation is a thin adapter over an existing seam: the task board (list, get, claim,
/// complete, update), the store (messages and roster) and the messager (event publishing). The
/// trait is object-safe because the command tree holds one `Arc<dyn TeamClient>` and knows nothing
/// else about the transport.
///
/// Errors are reserved for infrastructure failures; a *refused* operation (task not claimable,
/// content over the limit) is a successful call returning an outcome with `ok = false`.
#[async_trait]
pub trait TeamClient: Send + Sync {
    /// Returns the scope this client implements; it selects the command tree in the CLI.
    fn scope(&self) -> ClientScope;

    /// Returns the member name the client acts as (the sender / claimer identity).
    fn member_name(&self) -> &str;

    /// Renders and consumes the member's unread messages, advancing its read state.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::Error`] when the inbox cannot be read or persisted.
    async fn read_inbox(&self) -> Result<String>;

    /// Sends `content` to `to`; `"*"` broadcasts to the whole team.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::Error`] when the message cannot be persisted or
    /// published. Over-limit content is a refusal ([`SendOutcome::failed`]), not an error.
    async fn send_message(&self, to: &str, content: &str) -> Result<SendOutcome>;

    /// Creates a task, optionally pinning `task_id` and declaring dependency task ids.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::Error`] when the board rejects the request as malformed
    /// (unknown dependency, duplicate id) or the write fails.
    async fn create_task(
        &self,
        title: &str,
        content: &str,
        task_id: Option<&str>,
        dependencies: &[String],
    ) -> Result<CreateOutcome>;

    /// Edits the title and/or content of a task that has not started yet.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::Error`] when the task is unknown or the write fails.
    async fn update_task(
        &self,
        task_id: &str,
        title: Option<&str>,
        content: Option<&str>,
    ) -> Result<OpOutcome>;

    /// Lists the board, optionally restricted to one status.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::Error`] when the board cannot be read.
    async fn list_tasks(&self, status: Option<TaskStatus>) -> Result<Vec<TaskSummary>>;

    /// Lists the pending, unclaimed tasks a member may claim.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::Error`] when the board cannot be read.
    async fn claimable_tasks(&self) -> Result<Vec<TaskSummary>>;

    /// Returns one task's full detail, or `None` when the id is unknown.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::Error`] when the board cannot be read.
    async fn get_task(&self, task_id: &str) -> Result<Option<TaskDetail>>;

    /// Claims a pending task for the client's member.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::Error`] when the task is unknown or the write fails.
    async fn claim_task(&self, task_id: &str) -> Result<OpOutcome>;

    /// Completes a task owned by the client's member.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::Error`] when the task is unknown or the write fails.
    async fn complete_task(&self, task_id: &str) -> Result<OpOutcome>;

    /// Lists the team roster.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::Error`] when the roster cannot be read.
    async fn list_members(&self) -> Result<Vec<MemberSummary>>;

    /// Invokes a tool by name with its raw JSON arguments.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::Error::NotFound`] for an unknown tool and whatever the
    /// tool implementation reports for an invalid argument set.
    async fn invoke_tool(&self, name: &str, arguments: Value) -> Result<Value>;

    /// Returns the tool definitions this client advertises for its scope.
    fn tools(&self) -> Vec<TeamToolDef>;
}

/// Returns the tool set of a spawned member under the given team settings.
///
/// Mirrors Python `create_team_tools(role="teammate")`: `view_task` and `send_message` are always
/// present, the task-closing tool follows the dispatch mode (`claim_task` when autonomous,
/// `member_complete_task` when scheduled), plan mode adds `submit_plan`, and a non-zero subtask
/// depth under scheduled dispatch adds `create_task`/`update_task` for the member's own subtasks.
#[must_use]
pub fn member_tools_for(
    mode: MemberMode,
    dispatch: DispatchMode,
    max_subtask_depth: usize,
) -> Vec<TeamToolDef> {
    let mut tools = vec![view_task_tool(), send_message_tool()];
    tools.push(match dispatch {
        DispatchMode::Autonomous => member_claim_task_tool(),
        DispatchMode::Scheduled => member_complete_task_tool(),
    });
    if mode == MemberMode::PlanMode {
        tools.push(submit_plan_tool());
    }
    if dispatch == DispatchMode::Scheduled && max_subtask_depth > 0 {
        tools.push(subtask_create_tool());
        tools.push(subtask_update_tool());
    }
    tools
}

/// Returns the tool set of an outside operator, in the Python MCP server order.
///
/// The operator is not a team member, so it has no claim loop: it controls the board, the mailbox
/// and the roster explicitly, and it is the only scope with the `read_inbox` pull tool.
#[must_use]
pub fn operator_tools() -> Vec<TeamToolDef> {
    vec![
        read_inbox_tool(),
        send_message_tool(),
        create_task_tool(),
        update_task_tool(),
        list_tasks_tool(),
        list_claimable_tasks_tool(),
        get_task_tool(),
        operator_claim_task_tool(),
        complete_task_tool(),
        list_members_tool(),
    ]
}

fn string_schema() -> Value {
    json!({ "type": "string" })
}

fn empty_object_schema() -> Value {
    json!({ "type": "object", "properties": {}, "required": [] })
}

fn object_schema(properties: &Value, required: &[&str]) -> Value {
    json!({ "type": "object", "properties": properties, "required": required })
}

fn string_array_schema() -> Value {
    json!({ "type": "array", "items": { "type": "string" } })
}

fn view_task_tool() -> TeamToolDef {
    TeamToolDef::new(
        "view_task",
        "View team task information: list tasks, inspect one task, list claimable tasks, read a completion note, or list tasks awaiting review.",
        object_schema(
            &json!({
                "action": {
                    "type": "string",
                    "enum": ["get", "note", "list", "claimable", "in_review"],
                },
                "task_id": string_schema(),
                "status": string_schema(),
            }),
            &[],
        ),
    )
}

fn send_message_tool() -> TeamToolDef {
    TeamToolDef::new(
        "send_message",
        "Send a direct message to a member (use \"*\" to broadcast to the team).",
        object_schema(
            &json!({ "to": string_schema(), "content": string_schema() }),
            &["to", "content"],
        ),
    )
}

fn member_claim_task_tool() -> TeamToolDef {
    TeamToolDef::new(
        "claim_task",
        "Claim or complete a task (teammate only); you may hold only one in-progress task at a time.",
        object_schema(
            &json!({
                "task_id": string_schema(),
                "status": { "type": "string", "enum": ["claimed", "completed"] },
                "summary": string_schema(),
                "note": string_schema(),
            }),
            &["task_id", "status"],
        ),
    )
}

fn member_complete_task_tool() -> TeamToolDef {
    TeamToolDef::new(
        "member_complete_task",
        "Mark a task assigned to you as completed.",
        object_schema(
            &json!({
                "task_id": string_schema(),
                "summary": string_schema(),
                "note": string_schema(),
            }),
            &["task_id"],
        ),
    )
}

fn submit_plan_tool() -> TeamToolDef {
    TeamToolDef::new(
        "submit_plan",
        "Submit a plan for the leader to approve before working on a task in plan mode.",
        object_schema(
            &json!({
                "task_id": string_schema(),
                "plan_id": string_schema(),
                "plan_path": string_schema(),
            }),
            &["task_id", "plan_path"],
        ),
    )
}

fn subtask_create_tool() -> TeamToolDef {
    TeamToolDef::new(
        "create_task",
        "Create your own subtasks, optionally with dependency edges and a declared owner.",
        object_schema(
            &json!({
                "tasks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": string_schema(),
                            "content": string_schema(),
                            "depends_on": string_array_schema(),
                            "member": string_schema(),
                        },
                        "required": ["title", "content"],
                    },
                },
            }),
            &["tasks"],
        ),
    )
}

fn subtask_update_tool() -> TeamToolDef {
    TeamToolDef::new(
        "update_task",
        "Edit, confirm or cancel the subtasks of your own task.",
        object_schema(
            &json!({
                "task_id": string_schema(),
                "title": string_schema(),
                "content": string_schema(),
                "change_note": string_schema(),
                "confirm": { "type": "boolean" },
                "status": { "type": "string", "enum": ["cancelled"] },
            }),
            &["task_id"],
        ),
    )
}

fn read_inbox_tool() -> TeamToolDef {
    TeamToolDef::new(
        "read_inbox",
        "Pull your unread messages (direct + broadcast) and the current task board as text; reading marks messages read, and an empty inbox is normal.",
        empty_object_schema(),
    )
}

fn create_task_tool() -> TeamToolDef {
    TeamToolDef::new(
        "create_task",
        "Create a team task members can claim. Optionally pin an id and dependency ids.",
        object_schema(
            &json!({
                "title": string_schema(),
                "content": string_schema(),
                "task_id": string_schema(),
                "dependencies": string_array_schema(),
            }),
            &["title", "content"],
        ),
    )
}

fn update_task_tool() -> TeamToolDef {
    TeamToolDef::new(
        "update_task",
        "Edit a task's title and/or content (pending / blocked tasks only).",
        object_schema(
            &json!({
                "task_id": string_schema(),
                "title": string_schema(),
                "content": string_schema(),
            }),
            &["task_id"],
        ),
    )
}

fn list_tasks_tool() -> TeamToolDef {
    TeamToolDef::new(
        "list_tasks",
        "List team tasks, optionally filtered by status.",
        object_schema(&json!({ "status": string_schema() }), &[]),
    )
}

fn list_claimable_tasks_tool() -> TeamToolDef {
    TeamToolDef::new(
        "claimable_tasks",
        "List pending tasks available to claim.",
        empty_object_schema(),
    )
}

fn get_task_tool() -> TeamToolDef {
    TeamToolDef::new(
        "get_task",
        "Get one task's full detail, or null if it does not exist.",
        object_schema(&json!({ "task_id": string_schema() }), &["task_id"]),
    )
}

fn operator_claim_task_tool() -> TeamToolDef {
    TeamToolDef::new(
        "claim_task",
        "Claim a pending task.",
        object_schema(&json!({ "task_id": string_schema() }), &["task_id"]),
    )
}

fn complete_task_tool() -> TeamToolDef {
    TeamToolDef::new(
        "complete_task",
        "Complete a claimed task.",
        object_schema(&json!({ "task_id": string_schema() }), &["task_id"]),
    )
}

fn list_members_tool() -> TeamToolDef {
    TeamToolDef::new(
        "list_members",
        "List the team's members.",
        empty_object_schema(),
    )
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn names(tools: &[TeamToolDef]) -> Vec<&str> {
        tools.iter().map(|tool| tool.name.as_str()).collect()
    }

    /// Asserts the tool's top-level `required` array and that each entry has a property.
    fn assert_required(tool: &TeamToolDef, required: &[&str]) {
        let schema = tool.input_schema.as_object().unwrap();
        assert_eq!(schema["type"], "object", "tool {}", tool.name);
        let actual: Vec<&str> = schema["required"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_str().unwrap())
            .collect();
        assert_eq!(actual, required, "tool {}", tool.name);
        let properties = schema["properties"].as_object().unwrap();
        for field in required {
            assert!(
                properties.contains_key(*field),
                "tool {} declares required property '{field}' but has no schema for it",
                tool.name
            );
        }
    }

    #[test]
    fn autonomous_build_mode_member_tools() {
        let tools = member_tools_for(MemberMode::BuildMode, DispatchMode::Autonomous, 0);
        assert_eq!(names(&tools), ["view_task", "send_message", "claim_task"]);
        assert_required(&tools[0], &[]);
        assert_required(&tools[1], &["to", "content"]);
        assert_required(&tools[2], &["task_id", "status"]);
    }

    #[test]
    fn scheduled_build_mode_member_tools() {
        let tools = member_tools_for(MemberMode::BuildMode, DispatchMode::Scheduled, 0);
        assert_eq!(
            names(&tools),
            ["view_task", "send_message", "member_complete_task"]
        );
        assert_required(&tools[2], &["task_id"]);
    }

    #[test]
    fn autonomous_plan_mode_member_tools() {
        let tools = member_tools_for(MemberMode::PlanMode, DispatchMode::Autonomous, 2);
        assert_eq!(
            names(&tools),
            ["view_task", "send_message", "claim_task", "submit_plan"]
        );
        assert_required(&tools[3], &["task_id", "plan_path"]);
    }

    #[test]
    fn scheduled_plan_mode_member_tools() {
        let tools = member_tools_for(MemberMode::PlanMode, DispatchMode::Scheduled, 2);
        assert_eq!(
            names(&tools),
            [
                "view_task",
                "send_message",
                "member_complete_task",
                "submit_plan",
                "create_task",
                "update_task",
            ]
        );
        assert_required(&tools[4], &["tasks"]);
        assert_required(&tools[5], &["task_id"]);
        let subtask = &tools[4].input_schema["properties"]["tasks"]["items"];
        assert_eq!(subtask["required"], json!(["title", "content"]));
    }

    #[test]
    fn scheduled_member_without_subtask_depth_gets_no_task_tools() {
        let tools = member_tools_for(MemberMode::BuildMode, DispatchMode::Scheduled, 0);
        assert!(!names(&tools).contains(&"create_task"));
        assert!(!names(&tools).contains(&"update_task"));
    }

    #[test]
    fn autonomous_member_never_gets_member_task_tools() {
        let tools = member_tools_for(MemberMode::BuildMode, DispatchMode::Autonomous, 3);
        assert_eq!(names(&tools), ["view_task", "send_message", "claim_task"]);
    }

    #[test]
    fn operator_tools_keep_the_python_order_and_schemas() {
        let tools = operator_tools();
        assert_eq!(
            names(&tools),
            [
                "read_inbox",
                "send_message",
                "create_task",
                "update_task",
                "list_tasks",
                "claimable_tasks",
                "get_task",
                "claim_task",
                "complete_task",
                "list_members",
            ]
        );
        let required: [&[&str]; 10] = [
            &[],
            &["to", "content"],
            &["title", "content"],
            &["task_id"],
            &[],
            &[],
            &["task_id"],
            &["task_id"],
            &["task_id"],
            &[],
        ];
        for (tool, expected) in tools.iter().zip(required) {
            assert_required(tool, expected);
        }
    }

    #[test]
    fn outcome_constructors_and_serde_shape() {
        let sent = SendOutcome::sent("m-1");
        assert!(sent.ok);
        assert_eq!(sent.message_id.as_deref(), Some("m-1"));
        assert_eq!(
            serde_json::to_value(&sent).unwrap(),
            json!({ "ok": true, "message_id": "m-1", "reason": null })
        );

        let refused = SendOutcome::failed("target unknown");
        assert!(!refused.ok);
        assert_eq!(refused.reason.as_deref(), Some("target unknown"));

        let created = CreateOutcome::created("t-1");
        assert_eq!(created.task_id.as_deref(), Some("t-1"));
        assert!(!CreateOutcome::failed("duplicate").ok);

        assert!(OpOutcome::ok().ok);
        assert_eq!(
            OpOutcome::failed("not pending").reason.as_deref(),
            Some("not pending")
        );
    }

    #[test]
    fn task_op_result_lifts_into_an_outcome() {
        assert_eq!(OpOutcome::from(TaskOpResult::ok()), OpOutcome::ok());
        let refused = OpOutcome::from(TaskOpResult::failed("task already claimed"));
        assert_eq!(refused, OpOutcome::failed("task already claimed"));
    }

    #[test]
    fn roster_rows_keep_their_wire_names() {
        let summary = MemberSummary::new("dev-1", TeamRole::Teammate, MemberStatus::Ready);
        assert_eq!(
            serde_json::to_value(&summary).unwrap(),
            json!({ "member_name": "dev-1", "role": "teammate", "status": "ready" })
        );
        assert_eq!(MemberSummary::default().role, TeamRole::Leader);
    }

    /// Minimal client proving the trait is object-safe and callable through `Arc<dyn TeamClient>`.
    struct FakeClient;

    #[async_trait]
    impl TeamClient for FakeClient {
        fn scope(&self) -> ClientScope {
            ClientScope::Member
        }

        fn member_name(&self) -> &'static str {
            "dev-1"
        }

        async fn read_inbox(&self) -> Result<String> {
            Ok("(inbox empty)".to_string())
        }

        async fn send_message(&self, to: &str, content: &str) -> Result<SendOutcome> {
            Ok(SendOutcome::sent(format!("{to}:{content}")))
        }

        async fn create_task(
            &self,
            _title: &str,
            _content: &str,
            _task_id: Option<&str>,
            _dependencies: &[String],
        ) -> Result<CreateOutcome> {
            Ok(CreateOutcome::created("t-1"))
        }

        async fn update_task(
            &self,
            _task_id: &str,
            _title: Option<&str>,
            _content: Option<&str>,
        ) -> Result<OpOutcome> {
            Ok(OpOutcome::ok())
        }

        async fn list_tasks(&self, _status: Option<TaskStatus>) -> Result<Vec<TaskSummary>> {
            Ok(Vec::new())
        }

        async fn claimable_tasks(&self) -> Result<Vec<TaskSummary>> {
            Ok(Vec::new())
        }

        async fn get_task(&self, _task_id: &str) -> Result<Option<TaskDetail>> {
            Ok(None)
        }

        async fn claim_task(&self, _task_id: &str) -> Result<OpOutcome> {
            Ok(OpOutcome::ok())
        }

        async fn complete_task(&self, _task_id: &str) -> Result<OpOutcome> {
            Ok(OpOutcome::failed("not assignee"))
        }

        async fn list_members(&self) -> Result<Vec<MemberSummary>> {
            Ok(vec![MemberSummary::new(
                "dev-1",
                TeamRole::Teammate,
                MemberStatus::Ready,
            )])
        }

        async fn invoke_tool(&self, name: &str, arguments: Value) -> Result<Value> {
            Ok(json!({ "tool": name, "arguments": arguments }))
        }

        fn tools(&self) -> Vec<TeamToolDef> {
            member_tools_for(MemberMode::BuildMode, DispatchMode::Autonomous, 0)
        }
    }

    #[tokio::test]
    async fn client_is_object_safe() {
        let client: std::sync::Arc<dyn TeamClient> = std::sync::Arc::new(FakeClient);
        assert_eq!(client.scope(), ClientScope::Member);
        assert_eq!(client.member_name(), "dev-1");
        assert_eq!(client.read_inbox().await.unwrap(), "(inbox empty)");
        assert_eq!(
            client.send_message("user", "done").await.unwrap(),
            SendOutcome::sent("user:done")
        );
        assert!(client.get_task("missing").await.unwrap().is_none());
        assert_eq!(client.tools().len(), 3);
        let echoed = client
            .invoke_tool("claim_task", json!({"task_id": "t-1"}))
            .await
            .unwrap();
        assert_eq!(echoed["tool"], "claim_task");
    }
}
