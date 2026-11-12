//! The `openjiuwen-team-mcp` tool surface, driven end to end over [`StoreTeamClient`].
//!
//! Python parity source: `openjiuwen/agent_teams/mcp/server.py`. This suite pins what an external
//! agent actually observes, through the crate's public API only:
//!
//! 1. `list_tools` returns the operator catalogue in order (and the member's own tool set for a
//!    member join descriptor), every tool describing itself and declaring its required arguments;
//! 2. `instructions` is the operator control workflow for the operator scope and `None` for a
//!    member (whose team prompt is injected at spawn time);
//! 3. `call_tool` renders the exact compact JSON text of all ten operator tools against a real
//!    in-memory store, task board and in-process messager — the strings an outside CLI parses;
//! 4. failures are ordinary result *text*, never protocol errors: `Unknown tool: <name>`,
//!    `Tool <name> missing argument '<arg>'`, `Internal error: <error>`, and the server stays
//!    usable for the next call;
//! 5. `build_mcp_server` keeps the connected client's scope, and the module constants carry the
//!    documented wire values.
//!
//! Everything is offline and deterministic: no clock assertions, no sleeps, no model. Task rows are
//! seeded through [`TeamStore::insert_task`] with explicit stamps so list ordering is pinned rather
//! than dependent on how fast the test creates them.
//!
//! No `mcp` cfg gate: `TeamMcpServer`, `build_mcp_server` and the constants live in
//! `src/mcp/mod.rs`, which is compiled with *and* without the `mcp` feature (only `server.rs` and
//! `stdio.rs` are feature-gated), so this suite runs against the no-default-features target too.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![allow(clippy::missing_panics_doc)]

use std::sync::Arc;

use openjiuwen_agent_teams::mcp::{
    build_mcp_server, TeamMcpServer, OPERATOR_INSTRUCTIONS, READ_INBOX, SERVER_NAME,
};
use openjiuwen_agent_teams::messager::{create_messager, Messager};
use openjiuwen_agent_teams::schema::{
    MemberStatus, TaskRecord, TaskStatus, TeamRole, TransportSpec,
};
use openjiuwen_agent_teams::skill::{
    ClientScope, StoreTeamClient, TeamClient, TeamJoinDescriptor, TeamToolDef,
};
use openjiuwen_agent_teams::storage::{InMemoryTeamStore, MemberRecord, MessageRecord, TeamStore};
use openjiuwen_agent_teams::taskboard::TaskBoard;
use serde_json::{json, Value};

/// Team every fixture drives.
const TEAM: &str = "alpha";

/// Member identity the operator-scope client acts as (the mailbox recipient in this suite).
const OPERATOR: &str = "operator";

/// Member identity a member-scope client acts as.
const MEMBER: &str = "dev-1";

/// The operator catalogue, in the Python MCP server order.
const OPERATOR_TOOL_NAMES: [&str; 10] = [
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

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

/// One team's store, board, bus and client, built the way the CLI builds them.
///
/// The client is kept as its concrete type so a test can assert both the trait-object seam
/// (`build_mcp_server`) and the client's own `tools()` catalogue.
struct Fixture {
    /// Persistence seam the client writes through.
    store: Arc<dyn TeamStore>,
    /// Task board of [`TEAM`], shared with the client.
    board: Arc<TaskBoard>,
    /// The connected client the server wraps.
    client: Arc<StoreTeamClient>,
}

impl Fixture {
    /// Builds a fixture whose client joins [`TEAM`] `member_name` under `scope`.
    fn new(scope: ClientScope, member_name: &str) -> Self {
        let store: Arc<dyn TeamStore> = Arc::new(InMemoryTeamStore::new());
        let board = Arc::new(TaskBoard::with_defaults(Arc::clone(&store), TEAM));
        let messager: Arc<dyn Messager> =
            create_messager(&TransportSpec::inprocess()).expect("the in-process transport exists");
        let descriptor = TeamJoinDescriptor {
            session_id: "s1".to_string(),
            team_name: TEAM.to_string(),
            member_name: member_name.to_string(),
            scope,
            ..TeamJoinDescriptor::default()
        };
        let client = Arc::new(StoreTeamClient::new(
            descriptor,
            Arc::clone(&store),
            Arc::clone(&board),
            messager,
        ));
        Self {
            store,
            board,
            client,
        }
    }

    /// Builds an operator-scope fixture acting as [`OPERATOR`].
    fn operator() -> Self {
        Self::new(ClientScope::Operator, OPERATOR)
    }

    /// Builds a member-scope fixture acting as [`MEMBER`].
    fn member() -> Self {
        Self::new(ClientScope::Member, MEMBER)
    }

    /// The server under test, built through the public factory.
    fn server(&self) -> TeamMcpServer {
        let client: Arc<dyn TeamClient> = self.client.clone();
        build_mcp_server(client)
    }

    /// Seeds one roster row.
    async fn seed_member(&self, name: &str, role: TeamRole, status: MemberStatus) {
        let mut record = MemberRecord::new(TEAM, name, role);
        record.status = status;
        self.store
            .insert_member(record)
            .await
            .expect("member seeded");
    }

    /// Seeds one task row with an explicit stamp, so board ordering is deterministic.
    async fn seed_task(
        &self,
        task_id: &str,
        title: &str,
        status: TaskStatus,
        assignee: Option<&str>,
        updated_at: u64,
    ) {
        let mut record = TaskRecord::new(TEAM, task_id, title, format!("body of {title}"));
        record.status = status;
        record.assignee = assignee.map(str::to_owned);
        record.updated_at = updated_at;
        self.store.insert_task(record).await.expect("task seeded");
    }

    /// Seeds one unread direct message addressed to `to`.
    async fn seed_message(&self, message_id: &str, from: &str, to: &str, content: &str, at: u64) {
        let record = MessageRecord::direct(TEAM, message_id, from, to, content, at);
        self.store
            .insert_message(record)
            .await
            .expect("message seeded");
    }

    /// Returns the stored task row, panicking when it is missing.
    async fn task(&self, task_id: &str) -> TaskRecord {
        self.store
            .task(TEAM, task_id)
            .await
            .expect("store readable")
            .unwrap_or_else(|| panic!("task '{task_id}' was not persisted"))
    }
}

/// Returns the names of `tools`, in order.
fn names(tools: &[TeamToolDef]) -> Vec<String> {
    tools.iter().map(|tool| tool.name.clone()).collect()
}

/// Returns the `required` array of `name`'s input schema.
fn required_of(tools: &[TeamToolDef], name: &str) -> Value {
    tools
        .iter()
        .find(|tool| tool.name == name)
        .unwrap_or_else(|| panic!("tool '{name}' is listed"))
        .input_schema
        .get("required")
        .cloned()
        .unwrap_or(Value::Null)
}

/// Returns the top-level key order of a JSON object rendered as text.
fn key_order(text: &str) -> Vec<String> {
    let value: Value = serde_json::from_str(text).expect("result is JSON");
    value
        .as_object()
        .expect("result is a JSON object")
        .keys()
        .cloned()
        .collect()
}

// ---------------------------------------------------------------------------
// Constants and wiring
// ---------------------------------------------------------------------------

#[test]
fn constants_carry_the_documented_wire_values() {
    assert_eq!(SERVER_NAME, "openjiuwen-team-member");
    assert_eq!(READ_INBOX, "read_inbox");
    // The operator workflow must name the tools it tells the outside agent to drive.
    for tool in [
        "read_inbox",
        "create_task",
        "update_task",
        "send_message",
        "list_members",
    ] {
        assert!(
            OPERATOR_INSTRUCTIONS.contains(tool),
            "OPERATOR_INSTRUCTIONS does not name {tool}"
        );
    }
}

#[tokio::test]
async fn build_mcp_server_keeps_the_connected_scope() {
    for (fixture, scope) in [
        (Fixture::operator(), ClientScope::Operator),
        (Fixture::member(), ClientScope::Member),
    ] {
        let client: Arc<dyn TeamClient> = Arc::clone(&fixture.client) as Arc<dyn TeamClient>;
        let server = build_mcp_server(Arc::clone(&client));
        assert_eq!(server.scope(), scope);
        assert!(Arc::ptr_eq(server.client(), &client));
        assert_eq!(TeamMcpServer::new(Arc::clone(&client)).scope(), scope);
    }
}

// ---------------------------------------------------------------------------
// Tool tables
// ---------------------------------------------------------------------------

#[tokio::test]
async fn operator_list_tools_is_the_catalogue_in_order() {
    let fixture = Fixture::operator();
    let tools = fixture.server().list_tools().await;

    assert_eq!(names(&tools), OPERATOR_TOOL_NAMES);
    for tool in &tools {
        assert!(
            !tool.description.trim().is_empty(),
            "tool '{}' has no description",
            tool.name
        );
        assert_eq!(
            tool.input_schema.get("type"),
            Some(&json!("object")),
            "tool '{}' does not advertise an object schema",
            tool.name
        );
    }

    // The documented required-argument set of every tool.
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

    // The declared properties are the ones the dispatcher actually reads.
    let create = tools
        .iter()
        .find(|tool| tool.name == "create_task")
        .expect("create_task is listed");
    let properties = create
        .input_schema
        .get("properties")
        .and_then(Value::as_object)
        .expect("create_task declares properties");
    assert_eq!(
        properties.keys().cloned().collect::<Vec<_>>(),
        ["title", "content", "task_id", "dependencies"]
    );
}

#[tokio::test]
async fn member_list_tools_is_the_client_tool_set() {
    let fixture = Fixture::member();
    let expected = names(&fixture.client.tools());
    assert_eq!(
        expected,
        ["view_task", "send_message", "claim_task"],
        "the autonomous build-mode teammate tool set"
    );

    let tools = fixture.server().list_tools().await;
    assert_eq!(names(&tools), expected);
    assert!(
        !names(&tools).contains(&READ_INBOX.to_string()),
        "a member is pushed its messages; read_inbox is operator-only"
    );
    assert_eq!(
        required_of(&tools, "send_message"),
        json!(["to", "content"])
    );
}

// ---------------------------------------------------------------------------
// Instructions
// ---------------------------------------------------------------------------

#[tokio::test]
async fn instructions_follow_the_scope() {
    assert_eq!(
        Fixture::operator().server().instructions(),
        Some(OPERATOR_INSTRUCTIONS)
    );
    assert_eq!(Fixture::member().server().instructions(), None);
}

// ---------------------------------------------------------------------------
// Operator dispatch: mailbox
// ---------------------------------------------------------------------------

#[tokio::test]
async fn read_inbox_renders_the_mailbox_then_empties() {
    let fixture = Fixture::operator();
    fixture
        .seed_message("msg-1", MEMBER, OPERATOR, "hi", 1_700_000_000_000)
        .await;
    let server = fixture.server();

    let first = server.call_tool(READ_INBOX, json!({})).await;
    assert_eq!(first, "\"[1700000000000] dev-1 -> operator : hi\"");

    // Reading consumed the row, so the next pull is the empty rendering.
    let second = server.call_tool(READ_INBOX, json!({})).await;
    assert_eq!(second, "\"(inbox empty)\"");
}

#[tokio::test]
async fn send_message_renders_ok_and_message_id() {
    let fixture = Fixture::operator();
    let server = fixture.server();

    let text = server
        .call_tool("send_message", json!({ "to": MEMBER, "content": "hi" }))
        .await;

    assert_eq!(key_order(&text), ["ok", "message_id"]);
    let value: Value = serde_json::from_str(&text).expect("result is JSON");
    assert_eq!(value["ok"], json!(true));
    let message_id = value["message_id"]
        .as_str()
        .expect("a sent message carries its id");
    assert_eq!(message_id.len(), 36, "the id is a UUID: {message_id}");
    assert_eq!(
        message_id.chars().filter(|c| *c == '-').count(),
        4,
        "the id is a UUID: {message_id}"
    );
    assert_eq!(
        text,
        format!("{{\"ok\":true,\"message_id\":\"{message_id}\"}}")
    );

    // The row landed in the store, addressed and owned by the operator.
    let rows = fixture
        .store
        .unread_messages(TEAM, MEMBER)
        .await
        .expect("mailbox readable");
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].message_id, message_id);
    assert_eq!(rows[0].from_member_name, OPERATOR);
    assert_eq!(rows[0].content, "hi");
    assert!(!rows[0].is_broadcast());
}

#[tokio::test]
async fn member_scope_rejects_read_inbox() {
    let fixture = Fixture::member();
    fixture
        .seed_message("msg-1", MEMBER, OPERATOR, "hi", 1_700_000_000_000)
        .await;

    let text = fixture.server().call_tool(READ_INBOX, json!({})).await;
    assert_eq!(text, "Unknown tool: read_inbox");
}

// ---------------------------------------------------------------------------
// Operator dispatch: task board
// ---------------------------------------------------------------------------

#[tokio::test]
async fn create_task_renders_ok_reason_and_task_id() {
    let fixture = Fixture::operator();
    let server = fixture.server();

    let pinned = server
        .call_tool(
            "create_task",
            json!({ "title": "Alpha", "content": "do alpha", "task_id": "t-1" }),
        )
        .await;
    assert_eq!(pinned, "{\"ok\":true,\"reason\":\"\",\"task_id\":\"t-1\"}");
    assert_eq!(key_order(&pinned), ["ok", "reason", "task_id"]);
    let row = fixture.task("t-1").await;
    assert_eq!(row.title, "Alpha");
    assert_eq!(row.content, "do alpha");
    assert_eq!(row.status, TaskStatus::Pending);

    // Without a pinned id the board mints `task_<32 lowercase hex>`.
    let minted = server
        .call_tool(
            "create_task",
            json!({ "title": "Beta", "content": "do beta" }),
        )
        .await;
    let value: Value = serde_json::from_str(&minted).expect("result is JSON");
    assert_eq!(value["ok"], json!(true));
    assert_eq!(value["reason"], json!(""));
    let task_id = value["task_id"].as_str().expect("a created task has an id");
    let suffix = task_id
        .strip_prefix("task_")
        .expect("the minted id is task_<hex>");
    assert_eq!(suffix.len(), 32, "minted id '{task_id}'");
    assert!(
        suffix
            .chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
        "minted id '{task_id}' is 32 lowercase hex digits"
    );
    assert_eq!(fixture.task(task_id).await.title, "Beta");

    // A duplicate id is a policy refusal, not a transport error.
    let duplicate = server
        .call_tool(
            "create_task",
            json!({ "title": "Again", "content": "again", "task_id": "t-1" }),
        )
        .await;
    assert_eq!(
        duplicate,
        "{\"ok\":false,\"reason\":\"validation: task 't-1' already exists in team 'alpha'\",\"task_id\":null}"
    );
}

#[tokio::test]
async fn create_task_passes_the_optional_dependencies() {
    let fixture = Fixture::operator();
    fixture
        .seed_task("t-dep", "Dep", TaskStatus::Pending, None, 1_000)
        .await;

    let text = fixture
        .server()
        .call_tool(
            "create_task",
            json!({
                "title": "Two",
                "content": "do two",
                "task_id": "t-two",
                "dependencies": ["t-dep"],
            }),
        )
        .await;
    assert_eq!(text, "{\"ok\":true,\"reason\":\"\",\"task_id\":\"t-two\"}");

    assert_eq!(
        fixture
            .store
            .dependencies(TEAM, "t-two")
            .await
            .expect("edges readable"),
        vec!["t-dep".to_string()]
    );
    assert_eq!(
        fixture
            .board
            .claimable()
            .await
            .expect("board readable")
            .len(),
        1
    );
    assert_eq!(fixture.task("t-two").await.status, TaskStatus::Blocked);
}

#[tokio::test]
async fn update_task_edits_a_pending_task_and_refuses_the_rest() {
    let fixture = Fixture::operator();
    fixture
        .seed_task("t-1", "Alpha", TaskStatus::Pending, None, 3_000)
        .await;
    fixture
        .seed_task(
            "t-run",
            "Running",
            TaskStatus::InProgress,
            Some(MEMBER),
            2_000,
        )
        .await;
    let server = fixture.server();

    let ok = server
        .call_tool(
            "update_task",
            json!({ "task_id": "t-1", "title": "Alpha 2", "content": "edited" }),
        )
        .await;
    assert_eq!(ok, "{\"ok\":true,\"reason\":\"\"}");
    let row = fixture.task("t-1").await;
    assert_eq!(row.title, "Alpha 2");
    assert_eq!(row.content, "edited");

    let unknown = server
        .call_tool("update_task", json!({ "task_id": "nope", "title": "x" }))
        .await;
    assert_eq!(
        unknown,
        "{\"ok\":false,\"reason\":\"task 'nope' not found\"}"
    );

    let started = server
        .call_tool("update_task", json!({ "task_id": "t-run", "title": "x" }))
        .await;
    assert_eq!(
        started,
        "{\"ok\":false,\"reason\":\"task 't-run' cannot be edited from status 'in_progress' (only pending or blocked tasks can be edited)\"}"
    );
}

#[tokio::test]
async fn claim_task_claims_a_pending_task_and_reports_refusals() {
    let fixture = Fixture::operator();
    fixture
        .seed_task("t-free", "Free", TaskStatus::Pending, None, 3_000)
        .await;
    fixture
        .seed_task("t-taken", "Taken", TaskStatus::Pending, Some(MEMBER), 2_000)
        .await;
    let server = fixture.server();

    let claimed = server
        .call_tool("claim_task", json!({ "task_id": "t-free" }))
        .await;
    assert_eq!(claimed, "{\"ok\":true,\"reason\":\"\"}");
    let row = fixture.task("t-free").await;
    assert_eq!(row.status, TaskStatus::InProgress);
    assert_eq!(row.assignee.as_deref(), Some(OPERATOR));

    let taken = server
        .call_tool("claim_task", json!({ "task_id": "t-taken" }))
        .await;
    assert_eq!(
        taken,
        "{\"ok\":false,\"reason\":\"task 't-taken' is already claimed by 'dev-1'\"}"
    );

    // An unknown id surfaces as the client error text, still ordinary result text.
    let unknown = server
        .call_tool("claim_task", json!({ "task_id": "nope" }))
        .await;
    assert_eq!(unknown, "Internal error: not_found: task 'nope' not found");
}

#[tokio::test]
async fn complete_task_completes_an_owned_task_and_refuses_the_rest() {
    let fixture = Fixture::operator();
    fixture
        .seed_task(
            "t-own",
            "Own",
            TaskStatus::InProgress,
            Some(OPERATOR),
            3_000,
        )
        .await;
    fixture
        .seed_task("t-free", "Free", TaskStatus::Pending, None, 2_000)
        .await;
    let server = fixture.server();

    let completed = server
        .call_tool("complete_task", json!({ "task_id": "t-own" }))
        .await;
    assert_eq!(completed, "{\"ok\":true,\"reason\":\"\"}");
    assert_eq!(fixture.task("t-own").await.status, TaskStatus::Completed);

    // A free task: the board refuses, and the client lifts the reason into `ok = false`.
    let refused = server
        .call_tool("complete_task", json!({ "task_id": "t-free" }))
        .await;
    assert_eq!(
        refused,
        "{\"ok\":false,\"reason\":\"validation: task is assigned to '<unassigned>'\"}"
    );

    let unknown = server
        .call_tool("complete_task", json!({ "task_id": "nope" }))
        .await;
    assert_eq!(
        unknown,
        "{\"ok\":false,\"reason\":\"not_found: task 'nope' not found\"}"
    );
}

#[tokio::test]
async fn list_tasks_renders_the_board_newest_first() {
    let fixture = Fixture::operator();
    fixture
        .seed_task("t-a", "Alpha", TaskStatus::Pending, None, 3_000)
        .await;
    fixture
        .seed_task("t-b", "Beta", TaskStatus::InProgress, Some(MEMBER), 2_000)
        .await;
    fixture
        .seed_task("t-c", "Gamma", TaskStatus::Pending, Some(MEMBER), 1_000)
        .await;
    let server = fixture.server();

    let all = server.call_tool("list_tasks", json!({})).await;
    assert_eq!(
        all,
        concat!(
            "[{\"task_id\":\"t-a\",\"title\":\"Alpha\",\"status\":\"pending\",\"assignee\":null},",
            "{\"task_id\":\"t-b\",\"title\":\"Beta\",\"status\":\"in_progress\",\"assignee\":\"dev-1\"},",
            "{\"task_id\":\"t-c\",\"title\":\"Gamma\",\"status\":\"pending\",\"assignee\":\"dev-1\"}]"
        )
    );

    let pending = server
        .call_tool("list_tasks", json!({ "status": "pending" }))
        .await;
    assert_eq!(
        pending,
        concat!(
            "[{\"task_id\":\"t-a\",\"title\":\"Alpha\",\"status\":\"pending\",\"assignee\":null},",
            "{\"task_id\":\"t-c\",\"title\":\"Gamma\",\"status\":\"pending\",\"assignee\":\"dev-1\"}]"
        )
    );
}

#[tokio::test]
async fn list_tasks_rejects_an_unknown_status_spelling() {
    let fixture = Fixture::operator();
    fixture
        .seed_task("t-a", "Alpha", TaskStatus::Pending, None, 3_000)
        .await;
    let server = fixture.server();

    let text = server
        .call_tool("list_tasks", json!({ "status": "not_a_status" }))
        .await;
    assert_eq!(text, "Tool list_tasks invalid argument 'status'");

    // The server is unchanged by the error.
    let ok = server
        .call_tool("list_tasks", json!({ "status": "pending" }))
        .await;
    assert_eq!(
        ok,
        "[{\"task_id\":\"t-a\",\"title\":\"Alpha\",\"status\":\"pending\",\"assignee\":null}]"
    );
}

#[tokio::test]
async fn claimable_tasks_renders_pending_unclaimed_rows() {
    let fixture = Fixture::operator();
    fixture
        .seed_task("t-a", "Alpha", TaskStatus::Pending, None, 3_000)
        .await;
    fixture
        .seed_task("t-b", "Beta", TaskStatus::Pending, Some(MEMBER), 2_000)
        .await;
    fixture
        .seed_task("t-c", "Gamma", TaskStatus::InProgress, None, 1_000)
        .await;

    let text = fixture
        .server()
        .call_tool("claimable_tasks", json!({}))
        .await;
    assert_eq!(text, "[{\"task_id\":\"t-a\",\"title\":\"Alpha\"}]");
}

#[tokio::test]
async fn get_task_renders_the_detail_and_null() {
    let fixture = Fixture::operator();
    fixture
        .seed_task("t-a", "Alpha", TaskStatus::Pending, None, 3_000)
        .await;
    let server = fixture.server();

    let text = server
        .call_tool("get_task", json!({ "task_id": "t-a" }))
        .await;
    assert_eq!(
        text,
        concat!(
            "{\"task_id\":\"t-a\",\"title\":\"Alpha\",\"content\":\"body of Alpha\",",
            "\"status\":\"pending\",\"assignee\":null,\"assignee_status\":null,",
            "\"reviewers\":[],\"review_round\":0,\"max_review_rounds\":3,",
            "\"summary\":null,\"note\":null,\"blocked_by\":[],\"blocks\":[],",
            "\"parent_task_id\":null,\"subtasks\":[],\"revision\":0,\"updated_at\":3000}"
        )
    );

    let missing = server
        .call_tool("get_task", json!({ "task_id": "gone" }))
        .await;
    assert_eq!(missing, "null");
}

#[tokio::test]
async fn list_members_renders_the_sorted_roster() {
    let fixture = Fixture::operator();
    fixture
        .seed_member("dev-1", TeamRole::Teammate, MemberStatus::Ready)
        .await;
    fixture
        .seed_member("team_leader", TeamRole::Leader, MemberStatus::Busy)
        .await;

    let text = fixture.server().call_tool("list_members", json!({})).await;
    assert_eq!(
        text,
        concat!(
            "[{\"member_name\":\"dev-1\",\"role\":\"teammate\",\"status\":\"ready\"},",
            "{\"member_name\":\"team_leader\",\"role\":\"leader\",\"status\":\"busy\"}]"
        )
    );
}

// ---------------------------------------------------------------------------
// Member dispatch
// ---------------------------------------------------------------------------

#[tokio::test]
async fn member_scope_delegates_to_the_client_tool() {
    let fixture = Fixture::member();
    fixture
        .seed_task("t-a", "Alpha", TaskStatus::Pending, None, 3_000)
        .await;
    let server = fixture.server();

    let listed = server
        .call_tool("view_task", json!({ "action": "list" }))
        .await;
    assert_eq!(
        listed,
        "[{\"task_id\":\"t-a\",\"title\":\"Alpha\",\"status\":\"pending\",\"assignee\":null}]"
    );

    // A tool the client does not implement is a client error, rendered as text.
    let unknown = server.call_tool("nope", json!({})).await;
    assert_eq!(unknown, "Internal error: not_found: unknown tool 'nope'");

    // A missing required argument is the client's validation error, again as text.
    let missing = server
        .call_tool("view_task", json!({ "action": "get" }))
        .await;
    assert_eq!(missing, "Internal error: validation: 'task_id' is required");
}

// ---------------------------------------------------------------------------
// Failure text and server durability
// ---------------------------------------------------------------------------

#[tokio::test]
async fn unknown_operator_tool_answers_unknown_tool() {
    let text = Fixture::operator()
        .server()
        .call_tool("nope", json!({}))
        .await;
    assert_eq!(text, "Unknown tool: nope");
}

#[tokio::test]
async fn malformed_arguments_are_text_and_the_server_stays_usable() {
    let fixture = Fixture::operator();
    fixture
        .seed_task("t-a", "Alpha", TaskStatus::Pending, None, 3_000)
        .await;
    let server = fixture.server();

    let missing = server.call_tool("get_task", json!({})).await;
    assert_eq!(missing, "Tool get_task missing argument 'task_id'");

    // A wrong JSON type takes the same path (there is no coercion).
    let mistyped = server.call_tool("get_task", json!({ "task_id": 7 })).await;
    assert_eq!(mistyped, "Tool get_task missing argument 'task_id'");

    let missing_to = server
        .call_tool("send_message", json!({ "content": "hi" }))
        .await;
    assert_eq!(missing_to, "Tool send_message missing argument 'to'");

    let missing_create = server
        .call_tool("create_task", json!({ "title": "x" }))
        .await;
    assert_eq!(
        missing_create,
        "Tool create_task missing argument 'content'"
    );

    // Every failure above left the server usable.
    let ok = server
        .call_tool(
            "create_task",
            json!({ "title": "Later", "content": "still works", "task_id": "t-later" }),
        )
        .await;
    assert_eq!(ok, "{\"ok\":true,\"reason\":\"\",\"task_id\":\"t-later\"}");
}
