//! Offline suite over a scripted [`TeamClient`] fake.
//!
//! Everything here runs with and without the `mcp` feature (no `rmcp` types are involved): the
//! suite pins the tool tables, the scope-dependent instructions and the exact compact-JSON result
//! text consumers parse.
#![allow(clippy::unwrap_used, clippy::expect_used)]
use super::{TeamMcpServer, OPERATOR_INSTRUCTIONS, READ_INBOX};
use crate::schema::{MemberStatus, TaskDetail, TaskStatus, TaskSummary, TeamRole};
use crate::skill::{
    ClientScope, CreateOutcome, MemberSummary, OpOutcome, SendOutcome, TeamClient, TeamToolDef,
};
use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use parking_lot::Mutex;
use serde_json::{json, Value};
use std::sync::Arc;
// ---------------------------------------------------------------------------
// Fake client
// ---------------------------------------------------------------------------
// The fake deliberately builds every returned value field by field: the `skill` contract promises
// data shapes, not `Clone`/`Debug`/`Default` impls.
struct FakeClient {
    scope: ClientScope,
    tools: Vec<TeamToolDef>,
    inbox: String,
    send: SendOutcome,
    create: CreateOutcome,
    update: OpOutcome,
    claim: OpOutcome,
    complete: OpOutcome,
    tasks: Vec<TaskSummary>,
    claimable: Vec<TaskSummary>,
    detail: Option<TaskDetail>,
    members: Vec<MemberSummary>,
    invoke_value: Value,
    /// When set, every operation fails with this message.
    fail: Option<&'static str>,
    log: Arc<Mutex<Vec<(String, Value)>>>,
}
impl Default for FakeClient {
    fn default() -> Self {
        Self {
            scope: ClientScope::Operator,
            tools: Vec::new(),
            inbox: String::new(),
            send: SendOutcome {
                ok: false,
                message_id: None,
                reason: None,
            },
            create: CreateOutcome {
                ok: false,
                task_id: None,
                reason: None,
            },
            update: OpOutcome {
                ok: false,
                reason: None,
            },
            claim: OpOutcome {
                ok: false,
                reason: None,
            },
            complete: OpOutcome {
                ok: false,
                reason: None,
            },
            tasks: Vec::new(),
            claimable: Vec::new(),
            detail: None,
            members: Vec::new(),
            invoke_value: Value::Null,
            fail: None,
            log: Arc::new(Mutex::new(Vec::new())),
        }
    }
}
impl FakeClient {
    fn operator() -> Self {
        Self {
            scope: ClientScope::Operator,
            inbox: "(inbox empty)".to_string(),
            send: SendOutcome {
                ok: true,
                message_id: Some("m1".to_string()),
                reason: None,
            },
            create: CreateOutcome {
                ok: true,
                task_id: Some("t1".to_string()),
                reason: None,
            },
            update: OpOutcome {
                ok: false,
                reason: Some("nope".to_string()),
            },
            claim: OpOutcome {
                ok: true,
                reason: None,
            },
            complete: OpOutcome {
                ok: true,
                reason: None,
            },
            tasks: vec![summary()],
            claimable: vec![summary()],
            detail: Some(detail()),
            members: vec![faker_member()],
            invoke_value: json!({ "ok": true }),
            ..Self::default()
        }
    }
    fn member() -> Self {
        Self {
            scope: ClientScope::Member,
            tools: vec![
                tool_def("view_task", &json!(["action"])),
                tool_def("claim_task", &json!(["task_id"])),
                tool_def("send_message", &json!(["to", "content"])),
            ],
            invoke_value: json!({ "ok": true }),
            ..Self::default()
        }
    }
    fn failing(message: &'static str) -> Self {
        let mut client = Self::operator();
        client.fail = Some(message);
        client
    }
    fn calls(&self) -> Vec<(String, Value)> {
        self.log.lock().clone()
    }
    fn record(&self, name: &str, args: Value) -> Result<()> {
        if let Some(message) = self.fail {
            return Err(Error::validation(message));
        }
        self.log.lock().push((name.to_string(), args));
        Ok(())
    }
}
#[async_trait]
impl TeamClient for FakeClient {
    fn scope(&self) -> ClientScope {
        self.scope
    }
    fn member_name(&self) -> &'static str {
        "alice"
    }
    fn tools(&self) -> Vec<TeamToolDef> {
        self.tools
            .iter()
            .map(|tool| TeamToolDef {
                name: tool.name.clone(),
                description: tool.description.clone(),
                input_schema: tool.input_schema.clone(),
            })
            .collect()
    }
    async fn read_inbox(&self) -> Result<String> {
        self.record(READ_INBOX, Value::Null)?;
        Ok(self.inbox.clone())
    }
    async fn send_message(&self, to: &str, content: &str) -> Result<SendOutcome> {
        self.record("send_message", json!({ "to": to, "content": content }))?;
        Ok(SendOutcome {
            ok: self.send.ok,
            message_id: self.send.message_id.clone(),
            reason: self.send.reason.clone(),
        })
    }
    async fn create_task(
        &self,
        title: &str,
        content: &str,
        task_id: Option<&str>,
        dependencies: &[String],
    ) -> Result<CreateOutcome> {
        self.record(
            "create_task",
            json!({
                "title": title,
                "content": content,
                "task_id": task_id,
                "dependencies": dependencies,
            }),
        )?;
        Ok(CreateOutcome {
            ok: self.create.ok,
            task_id: self.create.task_id.clone(),
            reason: self.create.reason.clone(),
        })
    }
    async fn update_task(
        &self,
        task_id: &str,
        title: Option<&str>,
        content: Option<&str>,
    ) -> Result<OpOutcome> {
        self.record(
            "update_task",
            json!({ "task_id": task_id, "title": title, "content": content }),
        )?;
        Ok(OpOutcome {
            ok: self.update.ok,
            reason: self.update.reason.clone(),
        })
    }
    async fn list_tasks(&self, status: Option<TaskStatus>) -> Result<Vec<TaskSummary>> {
        self.record("list_tasks", json!({ "status": status }))?;
        Ok(self.tasks.clone())
    }
    async fn claimable_tasks(&self) -> Result<Vec<TaskSummary>> {
        self.record("claimable_tasks", Value::Null)?;
        Ok(self.claimable.clone())
    }
    async fn get_task(&self, task_id: &str) -> Result<Option<TaskDetail>> {
        self.record("get_task", json!({ "task_id": task_id }))?;
        Ok(self.detail.clone())
    }
    async fn claim_task(&self, task_id: &str) -> Result<OpOutcome> {
        self.record("claim_task", json!({ "task_id": task_id }))?;
        Ok(OpOutcome {
            ok: self.claim.ok,
            reason: self.claim.reason.clone(),
        })
    }
    async fn complete_task(&self, task_id: &str) -> Result<OpOutcome> {
        self.record("complete_task", json!({ "task_id": task_id }))?;
        Ok(OpOutcome {
            ok: self.complete.ok,
            reason: self.complete.reason.clone(),
        })
    }
    async fn list_members(&self) -> Result<Vec<MemberSummary>> {
        self.record("list_members", Value::Null)?;
        Ok(self
            .members
            .iter()
            .map(|member| MemberSummary {
                member_name: member.member_name.clone(),
                role: member.role,
                status: member.status,
            })
            .collect())
    }
    async fn invoke_tool(&self, name: &str, arguments: Value) -> Result<Value> {
        self.record(name, arguments)?;
        Ok(self.invoke_value.clone())
    }
}
// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------
fn summary() -> TaskSummary {
    TaskSummary {
        task_id: "t1".to_string(),
        title: "Do".to_string(),
        status: TaskStatus::Pending,
        ..TaskSummary::default()
    }
}
fn detail() -> TaskDetail {
    TaskDetail {
        task_id: "t1".to_string(),
        title: "Do".to_string(),
        content: "Body".to_string(),
        status: TaskStatus::InProgress,
        assignee: Some("alice".to_string()),
        ..TaskDetail::default()
    }
}
fn faker_member() -> MemberSummary {
    MemberSummary {
        member_name: "alice".to_string(),
        role: TeamRole::Teammate,
        status: MemberStatus::Ready,
    }
}
fn tool_def(name: &str, required: &Value) -> TeamToolDef {
    TeamToolDef {
        name: name.to_string(),
        description: format!("{name} description"),
        input_schema: json!({
            "type": "object",
            "properties": {},
            "required": required,
        }),
    }
}
fn required_of(tools: &[TeamToolDef], name: &str) -> Value {
    tools
        .iter()
        .find(|tool| tool.name == name)
        .map(|tool| {
            tool.input_schema
                .get("required")
                .cloned()
                .unwrap_or(Value::Null)
        })
        .expect("tool is listed")
}
fn names(tools: &[TeamToolDef]) -> Vec<&str> {
    tools.iter().map(|tool| tool.name.as_str()).collect()
}
// ---------------------------------------------------------------------------
// Tools and instructions
// ---------------------------------------------------------------------------
#[tokio::test]
async fn operator_list_tools_exposes_the_catalogue() {
    let server = TeamMcpServer::new(Arc::new(FakeClient::operator()));
    let tools = server.list_tools().await;
    let catalogue = crate::skill::operator_tools();
    assert_eq!(names(&tools), names(&catalogue));
    let expected = [
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
    ];
    for name in expected {
        assert!(
            names(&tools).contains(&name),
            "missing operator tool {name}"
        );
    }
    assert_eq!(required_of(&tools, "read_inbox"), json!([]));
    assert_eq!(
        required_of(&tools, "send_message"),
        json!(["to", "content"])
    );
    assert_eq!(
        required_of(&tools, "create_task"),
        json!(["title", "content"])
    );
    assert_eq!(required_of(&tools, "update_task"), json!(["task_id"]));
    assert_eq!(required_of(&tools, "list_tasks"), json!([]));
    assert_eq!(required_of(&tools, "claimable_tasks"), json!([]));
    assert_eq!(required_of(&tools, "get_task"), json!(["task_id"]));
    assert_eq!(required_of(&tools, "claim_task"), json!(["task_id"]));
    assert_eq!(required_of(&tools, "complete_task"), json!(["task_id"]));
    assert_eq!(required_of(&tools, "list_members"), json!([]));
}
#[tokio::test]
async fn member_list_tools_uses_the_client_tools() {
    let fake = FakeClient::member();
    let expected = names(&fake.tools)
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>();
    let server = TeamMcpServer::new(Arc::new(fake));
    let tools = server.list_tools().await;
    assert_eq!(
        names(&tools),
        vec!["view_task", "claim_task", "send_message"]
    );
    assert_eq!(
        names(&tools)
            .into_iter()
            .map(str::to_string)
            .collect::<Vec<_>>(),
        expected
    );
    assert!(!names(&tools).contains(&READ_INBOX));
    assert_eq!(required_of(&tools, "view_task"), json!(["action"]));
}
#[tokio::test]
async fn instructions_follow_the_scope() {
    let operator = TeamMcpServer::new(Arc::new(FakeClient::operator()));
    assert_eq!(operator.instructions(), Some(OPERATOR_INSTRUCTIONS));
    assert!(OPERATOR_INSTRUCTIONS.contains("read_inbox"));
    let member = TeamMcpServer::new(Arc::new(FakeClient::member()));
    assert_eq!(member.instructions(), None);
}
// ---------------------------------------------------------------------------
// Operator dispatch
// ---------------------------------------------------------------------------
#[tokio::test]
async fn read_inbox_renders_a_json_string() {
    let server = TeamMcpServer::new(Arc::new(FakeClient::operator()));
    let text = server.call_tool(READ_INBOX, json!({})).await;
    assert_eq!(text, "\"(inbox empty)\"");
}
#[tokio::test]
async fn send_message_renders_ok_and_message_id() {
    let fake = Arc::new(FakeClient::operator());
    let server = TeamMcpServer::new(fake.clone());
    let text = server
        .call_tool("send_message", json!({ "to": "*", "content": "hi" }))
        .await;
    assert_eq!(text, "{\"ok\":true,\"message_id\":\"m1\"}");
    assert_eq!(
        fake.calls(),
        vec![(
            "send_message".to_string(),
            json!({ "to": "*", "content": "hi" })
        )]
    );
}
#[tokio::test]
async fn send_message_failure_has_null_message_id() {
    let mut fake = FakeClient::operator();
    fake.send = SendOutcome {
        ok: false,
        message_id: None,
        reason: Some("refused".to_string()),
    };
    let server = TeamMcpServer::new(Arc::new(fake));
    let text = server
        .call_tool("send_message", json!({ "to": "bob", "content": "hi" }))
        .await;
    assert_eq!(text, "{\"ok\":false,\"message_id\":null}");
}
#[tokio::test]
async fn create_task_renders_empty_reason_and_task_id() {
    let server = TeamMcpServer::new(Arc::new(FakeClient::operator()));
    let text = server
        .call_tool("create_task", json!({ "title": "Do", "content": "Body" }))
        .await;
    assert_eq!(text, "{\"ok\":true,\"reason\":\"\",\"task_id\":\"t1\"}");
}
#[tokio::test]
async fn create_task_passes_optional_arguments() {
    let fake = Arc::new(FakeClient::operator());
    let server = TeamMcpServer::new(fake.clone());
    let _ = server
        .call_tool(
            "create_task",
            json!({
                "title": "Do",
                "content": "Body",
                "task_id": "t1",
                "dependencies": ["t0"],
            }),
        )
        .await;
    assert_eq!(
        fake.calls(),
        vec![(
            "create_task".to_string(),
            json!({
                "title": "Do",
                "content": "Body",
                "task_id": "t1",
                "dependencies": ["t0"],
            })
        )]
    );
}
#[tokio::test]
async fn update_task_renders_refusal() {
    let server = TeamMcpServer::new(Arc::new(FakeClient::operator()));
    let text = server
        .call_tool("update_task", json!({ "task_id": "t1", "title": "New" }))
        .await;
    assert_eq!(text, "{\"ok\":false,\"reason\":\"nope\"}");
}
#[tokio::test]
async fn list_tasks_renders_summaries() {
    let server = TeamMcpServer::new(Arc::new(FakeClient::operator()));
    let text = server
        .call_tool("list_tasks", json!({ "status": "pending" }))
        .await;
    assert_eq!(
        text,
        "[{\"task_id\":\"t1\",\"title\":\"Do\",\"status\":\"pending\",\"assignee\":null}]"
    );
}
#[tokio::test]
async fn claimable_tasks_renders_id_and_title() {
    let server = TeamMcpServer::new(Arc::new(FakeClient::operator()));
    let text = server.call_tool("claimable_tasks", json!({})).await;
    assert_eq!(text, "[{\"task_id\":\"t1\",\"title\":\"Do\"}]");
}
#[tokio::test]
async fn get_task_renders_detail() {
    let server = TeamMcpServer::new(Arc::new(FakeClient::operator()));
    let text = server
        .call_tool("get_task", json!({ "task_id": "t1" }))
        .await;
    assert_eq!(
        text,
        serde_json::to_string(&Some(detail())).expect("detail serializes")
    );
}
#[tokio::test]
async fn get_task_renders_null_when_missing() {
    let mut fake = FakeClient::operator();
    fake.detail = None;
    let server = TeamMcpServer::new(Arc::new(fake));
    let text = server
        .call_tool("get_task", json!({ "task_id": "gone" }))
        .await;
    assert_eq!(text, "null");
}
#[tokio::test]
async fn claim_and_complete_render_outcomes() {
    let server = TeamMcpServer::new(Arc::new(FakeClient::operator()));
    let claimed = server
        .call_tool("claim_task", json!({ "task_id": "t1" }))
        .await;
    assert_eq!(claimed, "{\"ok\":true,\"reason\":\"\"}");
    let completed = server
        .call_tool("complete_task", json!({ "task_id": "t1" }))
        .await;
    assert_eq!(completed, "{\"ok\":true,\"reason\":\"\"}");
}
#[tokio::test]
async fn list_members_renders_roster() {
    let server = TeamMcpServer::new(Arc::new(FakeClient::operator()));
    let text = server.call_tool("list_members", json!({})).await;
    assert_eq!(
        text,
        "[{\"member_name\":\"alice\",\"role\":\"teammate\",\"status\":\"ready\"}]"
    );
}
// ---------------------------------------------------------------------------
// Failure text
// ---------------------------------------------------------------------------
#[tokio::test]
async fn unknown_tool_answers_unknown_tool() {
    let server = TeamMcpServer::new(Arc::new(FakeClient::operator()));
    let text = server.call_tool("nope", json!({})).await;
    assert_eq!(text, "Unknown tool: nope");
}
#[tokio::test]
async fn client_error_answers_internal_error() {
    let server = TeamMcpServer::new(Arc::new(FakeClient::failing("boom")));
    let text = server.call_tool("list_members", json!({})).await;
    assert_eq!(text, "Internal error: validation: boom");
}
#[tokio::test]
async fn missing_argument_answers_tool_missing_argument() {
    let server = TeamMcpServer::new(Arc::new(FakeClient::operator()));
    let text = server.call_tool("get_task", json!({})).await;
    assert_eq!(text, "Tool get_task missing argument 'task_id'");
    let mistyped = server.call_tool("send_message", json!({ "to": 7 })).await;
    assert_eq!(mistyped, "Tool send_message missing argument 'to'");
}
// ---------------------------------------------------------------------------
// Member dispatch
// ---------------------------------------------------------------------------
#[tokio::test]
async fn member_scope_rejects_read_inbox() {
    let server = TeamMcpServer::new(Arc::new(FakeClient::member()));
    let text = server.call_tool(READ_INBOX, json!({})).await;
    assert_eq!(text, "Unknown tool: read_inbox");
}
#[tokio::test]
async fn member_scope_invokes_the_client_tool() {
    let fake = Arc::new(FakeClient::member());
    let server = TeamMcpServer::new(fake.clone());
    let text = server
        .call_tool("view_task", json!({ "action": "list" }))
        .await;
    assert_eq!(text, "{\"ok\":true}");
    assert_eq!(
        fake.calls(),
        vec![("view_task".to_string(), json!({ "action": "list" }))]
    );
}
#[tokio::test]
async fn member_scope_client_error_answers_internal_error() {
    let mut fake = FakeClient::member();
    fake.fail = Some("down");
    let server = TeamMcpServer::new(Arc::new(fake));
    let text = server.call_tool("view_task", json!({})).await;
    assert_eq!(text, "Internal error: validation: down");
}
// ---------------------------------------------------------------------------
// Live-team attach (stdio entry points, `mcp` feature only)
// ---------------------------------------------------------------------------
// These drive the descriptor-level seam `run_mcp_server_with_descriptor` rather than
// `run_mcp_server_with`: the latter loads `OPENJIUWEN_TEAM_JOIN`, and mutating the process
// environment inside a unit test races against every other test in this binary that reads it.
// The descriptor-level seam exercises exactly the same resolver + exit-code plumbing.
#[cfg(feature = "mcp")]
mod live_attach {
    use crate::mcp::stdio;
    use crate::skill::TeamJoinDescriptor;
    use openjiuwen_common::error::ErrorKind;
    fn unhosted_descriptor() -> TeamJoinDescriptor {
        TeamJoinDescriptor {
            team_name: "mcp-unhosted".to_string(),
            ..TeamJoinDescriptor::default()
        }
    }
    #[tokio::test]
    async fn resolve_from_registry_reports_unhosted_team() {
        let descriptor = unhosted_descriptor();
        // `Arc<dyn TeamClient>` is not `Debug`, so `Result::expect_err` is unavailable here.
        let error = stdio::resolve_from_registry(&descriptor)
            .err()
            .expect("an unhosted team does not resolve");
        assert_eq!(error.kind(), ErrorKind::Config);
        // `Error` renders as `<kind>: <message>`, so the pinned text carries the `config:` prefix.
        assert_eq!(
            error.to_string(),
            "config: no live team 'mcp-unhosted' hosted in this process"
        );
    }
    #[tokio::test]
    async fn run_mcp_server_with_descriptor_propagates_resolver_error() {
        let descriptor = unhosted_descriptor();
        let code =
            stdio::run_mcp_server_with_descriptor(descriptor, stdio::resolve_from_registry).await;
        assert_eq!(code, 1, "a resolver error exits 1 without opening stdio");
    }
    #[tokio::test]
    async fn run_mcp_server_with_descriptor_propagates_any_resolver_error() {
        // The exit code belongs to the seam, not to the default resolver: an injected resolver that
        // fails on any other ground exits `1` through the same path, before the transport starts.
        let descriptor = unhosted_descriptor();
        let code = stdio::run_mcp_server_with_descriptor(descriptor, |_| {
            Err(openjiuwen_common::error::Error::config("injected failure"))
        })
        .await;
        assert_eq!(code, 1);
    }
    #[test]
    fn stdio_transport_has_no_platform_gate() {
        // The transport is `rmcp::transport::stdio()`, which exists on every target. A `cfg` gate
        // would still compile on this host, so the source is where its absence can be pinned: the
        // previous build reopened `/dev/stdin` on Unix and refused every other target.
        const SOURCE: &str = include_str!("stdio.rs");
        for forbidden in [
            "cfg(unix)",
            "cfg(not(unix))",
            "/dev/stdin",
            "/dev/stdout",
            "process_streams",
            "Error::unsupported",
        ] {
            assert!(
                !SOURCE.contains(forbidden),
                "stdio.rs still carries the platform split: {forbidden}"
            );
        }
        // And the entry point itself is a plain, gate-free item on every target.
        let entry = stdio::serve_stdio;
        let _ = entry;
    }
}
