# src/mcp — `openjiuwen-team-mcp` stdio MCP server

## Intent

Project the team collaboration surface over MCP (stdio) to an agent that lives outside the team process.

Parity source: `openjiuwen/agent_teams/mcp/server.py` (309 lines), built on the low-level MCP server rather than FastMCP so member tools advertise their own card schema and every result is a single plain-text block.

The scenario is chosen by the join descriptor's `scope`: `member` (an external CLI acting as a first-class teammate, driven by the real teammate tool set) or `operator` (default; an outside team-control surface over the task board, messaging and roster).

This node owns no team logic: it re-exposes `crate::skill::TeamClient` operations and the `crate::skill` tool catalogue.

The descriptor and its `OPENJIUWEN_TEAM_JOIN` environment variable are owned by `crate::skill`; the process wiring (`openjiuwen-team-mcp` binary) is `openjiuwen-cli`'s job.

## API Surface

```rust
// mod.rs — available with and without the `mcp` feature
pub const SERVER_NAME: &str = "openjiuwen-team-member";
pub const READ_INBOX: &str = "read_inbox";
pub const OPERATOR_INSTRUCTIONS: &str;                 // fixed control-workflow text

pub struct TeamMcpServer;
impl TeamMcpServer {
    pub fn new(client: Arc<dyn TeamClient>) -> Self;
    pub fn scope(&self) -> ClientScope;                // client.scope()
    pub async fn list_tools(&self) -> Vec<TeamToolDef>; // member → client.tools(), operator → skill::operator_tools()
    pub async fn call_tool(&self, name: &str, arguments: Value) -> String;
    pub fn instructions(&self) -> Option<&'static str>; // None (member) | Some(OPERATOR_INSTRUCTIONS)
}
pub fn build_mcp_server(client: Arc<dyn TeamClient>) -> TeamMcpServer;

// dispatch.rs — available with and without the `mcp` feature
pub async fn dispatch_operator(client: &dyn TeamClient, name: &str, args: &Value) -> String;
pub async fn dispatch_member(client: &dyn TeamClient, name: &str, args: &Value) -> String;

// server.rs — #[cfg(feature = "mcp")]
pub struct TeamServerHandler;           // impl rmcp::ServerHandler
impl TeamServerHandler { pub fn new(server: TeamMcpServer) -> Self; }

// stdio.rs — #[cfg(feature = "mcp")]
pub async fn serve_stdio(server: TeamMcpServer) -> Result<()>;
pub async fn run_mcp_server_from_env() -> i32;
pub async fn run_mcp_server_with<F>(resolver: F) -> i32 where F: FnOnce(&TeamJoinDescriptor) -> Result<Arc<dyn TeamClient>>;
```

`TeamClient`, `TeamToolDef`, `ClientScope`, `SendOutcome`, `CreateOutcome`, `OpOutcome`, `MemberSummary`, `operator_tools()` and `TeamJoinDescriptor` come from `crate::skill`.
`TeamToolDef { name, description, input_schema }` is the only tool representation crossing this node; rmcp `Tool` values are built from it inside `server.rs`.

## Tool tables per scope

### `member` scope

`list_tools` returns one `TeamToolDef` per `client.tools()` entry, in the client's order; never a hardcoded list.
`call_tool` delegates every name to `client.invoke_tool(name, arguments)` and returns the rendered text.
`read_inbox` is deliberately absent (inbound messages are pushed by the parent process) and answers `Unknown tool: read_inbox`.
`instructions()` is `None`: the team system prompt is injected into the CLI at spawn time.

### `operator` scope

`list_tools` returns `skill::operator_tools()` (this node does not define the catalogue).

| tool | arguments | result text |
| --- | --- | --- |
| `read_inbox` | — | the rendered inbox text, serialized as a JSON string |
| `send_message` | `to`, `content` | `{"ok","message_id"}` |
| `create_task` | `title`, `content`, optional `task_id`, `dependencies` | `{"ok","reason","task_id"}` |
| `update_task` | `task_id`, optional `title`, `content` | `{"ok","reason"}` |
| `list_tasks` | optional `status` | `[{"task_id","title","status","assignee"}]` |
| `claimable_tasks` | — | `[{"task_id","title"}]` |
| `get_task` | `task_id` | the `TaskDetail` JSON, or `null` |
| `claim_task` | `task_id` | `{"ok","reason"}` |
| `complete_task` | `task_id` | `{"ok","reason"}` |
| `list_members` | — | `[{"member_name","role","status"}]` |

`instructions()` is `Some(OPERATOR_INSTRUCTIONS)`, naming `read_inbox`, `create_task`, `update_task`, `send_message` and `list_members`.

## `call_tool` return contract

Every result is a single plain-text MCP content block (`is_error` is always false); failures are ordinary text, never MCP protocol errors.

- Operator results are `serde_json::to_string` of the shapes above (compact, no spaces).
- A missing `reason` serializes as `""` and a missing `task_id` / `message_id` as `null`, matching the Python dataclass defaults.
- Member results are the serialized value returned by `client.invoke_tool`.
- `list_tasks` parses the `status` wire spelling (`pending`, `in_progress`, …); an unknown spelling is rejected, never silently ignored.
- Unknown tool (either scope, and member-scope `read_inbox`): `Unknown tool: <name>`.
- Any `Err` from the client: `Internal error: <error>`, logged with `tracing::error!`.
- Missing required argument: `Tool <name> missing argument '<arg>'`.
- Unparseable optional argument: `Tool <name> invalid argument '<arg>'`.
- Malformed arguments (wrong JSON type) take the same text path; `call_tool` never panics.

## Constraints

- Only `src/mcp/**` is edited; `src/lib.rs`, `Cargo.toml`, `src/schema/**` and `src/skill/**` are out of scope.
- Dependency direction: `crate::skill` (client + tool catalogue), `crate::schema` (`TaskSummary`, `TaskDetail`, `TaskStatus`, `MemberStatus`, `TeamRole`); nothing may depend on this node.
- `openjiuwen_common::error::{Error, Result}` only; `tracing` only (stderr is the log channel — stdout is the MCP wire).
- No `unsafe`, no `unwrap`/`expect` outside `#[cfg(test)]`, no panics on user input.
- `--no-default-features` must build: `mod.rs`, `dispatch.rs` and the tests are rmcp-free; every `rmcp` type stays inside `#[cfg(feature = "mcp")]`.
- The workspace does not enable `rmcp/transport-io`, so `rmcp::transport::stdio()` and `tokio::io::stdin/stdout` (`io-std`) are unavailable; `stdio.rs` adapts the process streams through `tokio::fs::File` and returns `Error::Unsupported` on non-Unix targets.
- A handler never closes the client: process exit releases handles (Python parity).
- Files ≤1000 lines; split by responsibility.

## Routing Table

- `./mod.rs` → constants, `TeamMcpServer`, scope routing, `build_mcp_server`, flat re-exports
- `./dispatch.rs` → operator/member dispatch over `TeamClient`, argument extraction, error text
- `./server.rs` → `TeamServerHandler`: `rmcp::ServerHandler` (`get_info`, `list_tools`, `call_tool`)
- `./stdio.rs` → `serve_stdio` and the environment-driven entry point
- `./tests.rs` → offline `FakeClient` suite (runs with and without the `mcp` feature)

## Test Strategy

- `#[cfg(test)]` suite in `tests.rs`, offline and deterministic, over a `FakeClient` implementing `TeamClient` (records calls, returns scripted outcomes).
- `list_tools` in both scopes: exact names and each tool's `input_schema.required` list; member scope reflects `client.tools()` order and never injects `read_inbox`.
- `instructions` per scope: `None` for member, `Some(OPERATOR_INSTRUCTIONS)` for operator.
- `call_tool` for all ten operator tools, asserting the exact compact JSON text; `get_task` missing → `null`.
- Failure text: unknown tool, member-scope `read_inbox`, missing required argument, and a client `Err` → `Internal error: ...`.
- `run_mcp_server_from_env` is not covered (process env + stdio): its descriptor failure path is exercised through the resolver seam only if cheap; the transport itself is out of scope.
- Definition of done: `cargo fmt -p openjiuwen-agent-teams`, `cargo test -p openjiuwen-agent-teams`, `cargo test -p openjiuwen-agent-teams --no-default-features`, `cargo clippy -p openjiuwen-agent-teams --all-targets -- -D warnings` green.

## Deferred

- HTTP/SSE transport: remote exposure is deployment's concern; stdio is the only transport, as in Python.
- Elicitation, prompts, resources and sampling: only the `tools` capability is advertised.
- Live client resolution: `run_mcp_server_from_env` validates the descriptor but resolves the `TeamClient` through an injected resolver; the default resolver reports `Error::Unsupported` until `crate::runtime`'s process-global team registry lands.
- Structured content / annotations: results are a single text block (Python parity).
- Resumability, progress notifications and graceful shutdown hooks.

## Notes for Agents

- `SERVER_NAME` is the MCP `serverInfo` name; the logical registration name CLIs use is `openjiuwen-team` and lives in the spawn layer, not here.
- The tool surface is resolved per request from the connected client's scope (`scope()`), never frozen at construction; `TeamMcpServer` is cheap and `Send + Sync`.
- Python's `_ClientHolder` lazily connects and re-binds session context per call; in Rust the `TeamClient` is already constructed and bound, so `TeamMcpServer` only holds the `Arc`.
- `READ_INBOX` is the one operator-only pull tool; keep the name stable (CLI prompts reference it).
- Keep the operator result shapes byte-identical to the table above: consumers (external CLIs) parse them as JSON.
