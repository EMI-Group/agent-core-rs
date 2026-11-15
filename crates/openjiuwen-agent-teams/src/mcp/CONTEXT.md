# src/mcp — `openjiuwen-team-mcp` stdio MCP server

## Intent

Project the team collaboration surface over MCP (stdio) to an agent that lives outside the team process.

Parity source: `openjiuwen/agent_teams/mcp/server.py` (309 lines), built on the low-level MCP server rather than FastMCP so member tools advertise their own card schema and every result is a single plain-text block.

The scenario is chosen by the join descriptor's `scope`: `member` (an external CLI acting as a first-class teammate, driven by the real teammate tool set) or `operator` (default; an outside team-control surface over the task board, messaging and roster).

This node owns no team logic: it re-exposes `crate::skill::TeamClient` operations and the `crate::skill` tool catalogue.

The descriptor and its `OPENJIUWEN_TEAM_JOIN` environment variable are owned by `crate::skill`; the process wiring is this crate's `src/bin/openjiuwen-team-mcp.rs` (built with the `mcp` feature), which installs the stderr log subscriber and forwards to `mcp::stdio::run_mcp_server_from_env`.
`openjiuwen-cli` is a sibling binary crate (read-only from this node) and does not host this server.

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
pub async fn run_mcp_server_from_env() -> i32;                                     // OPENJIUWEN_TEAM_JOIN → run_mcp_server_with_descriptor
pub(crate) fn resolve_from_registry(descriptor: &TeamJoinDescriptor) -> Result<Arc<dyn TeamClient>>;  // in-process team, else Error::Config
pub(crate) async fn run_mcp_server_with_descriptor<F>(descriptor: TeamJoinDescriptor, resolver: F) -> i32;  // env-free seam
pub async fn run_mcp_server_with<F>(resolver: F) -> i32 where F: FnOnce(&TeamJoinDescriptor) -> Result<Arc<dyn TeamClient>>;
```

`TeamClient`, `TeamToolDef`, `ClientScope`, `SendOutcome`, `CreateOutcome`, `OpOutcome`, `MemberSummary`, `operator_tools()` and `TeamJoinDescriptor` come from `crate::skill`.
`TeamToolDef { name, description, input_schema }` is the only tool representation crossing this node; rmcp `Tool` values are built from it inside `server.rs`.

## Live-team attach

The default resolver `resolve_from_registry` calls `crate::skill::connect_live_client`, so the server attaches to a team hosted in *this* process through the process-global runtime registry.
A team this process does not host is `Error::Config` with the text `no live team '<team_name>' hosted in this process`, logged with `tracing::error!`, and the entry point exits `1`.
`run_mcp_server_from_env` is `run_mcp_server_with(`[`resolve_from_registry`]`); both load `OPENJIUWEN_TEAM_JOIN` and then delegate to `run_mcp_server_with_descriptor`, the env-free seam that carries the descriptor and therefore needs no process environment.

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
- `serve_stdio` serves over `rmcp::transport::stdio()` (the crate enables `rmcp/transport-io` and `tokio/io-std`), one platform-independent implementation with no `cfg(unix)`/`cfg(not(unix))` gate and no target-specific refusal.
- `(tokio::io::Stdin, tokio::io::Stdout)` reaches the handler through rmcp's already-enabled `transport-async-rw` feature; a failed session initialization or a failed waiting task is `Error::config`.
- A handler never closes the client: process exit releases handles (Python parity).
- Files ≤1000 lines; split by responsibility.

## Routing Table

- `./mod.rs` → constants, `TeamMcpServer`, scope routing, `build_mcp_server`, flat re-exports
- `./dispatch.rs` → operator/member dispatch over `TeamClient`, argument extraction, error text
- `./server.rs` → `TeamServerHandler`: `rmcp::ServerHandler` (`get_info`, `list_tools`, `call_tool`)
- `./stdio.rs` → `serve_stdio`, the environment entry point, the registry resolver and the env-free descriptor seam
- `./tests.rs` → offline `FakeClient` suite (runs with and without the `mcp` feature)

## Test Strategy

- `#[cfg(test)]` suite in `tests.rs`, offline and deterministic, over a `FakeClient` implementing `TeamClient` (records calls, returns scripted outcomes).
- `list_tools` in both scopes: exact names and each tool's `input_schema.required` list; member scope reflects `client.tools()` order and never injects `read_inbox`.
- `instructions` per scope: `None` for member, `Some(OPERATOR_INSTRUCTIONS)` for operator.
- `call_tool` for all ten operator tools, asserting the exact compact JSON text; `get_task` missing → `null`.
- Failure text: unknown tool, member-scope `read_inbox`, missing required argument, and a client `Err` → `Internal error: ...`.
- `run_mcp_server_from_env` is not covered (process env + stdio): the transport is out of scope, and the process environment is never mutated in a test.
- `resolve_from_registry` for an unhosted team returns `ErrorKind::Config` with the exact message, and `run_mcp_server_with_descriptor` returns exit code `1` for that error and for an injected resolver `Err` (all `#[cfg(feature = "mcp")]`; none installs or clears the global registry, and none mutates the process environment).
- `stdio_transport_has_no_platform_gate` pins the platform-independent transport from the source (`include_str!("stdio.rs")` carries no `cfg(unix)`/`cfg(not(unix))`, no `/dev/stdin`, no `process_streams` and no `Error::unsupported`) and takes `serve_stdio` as a plain item; the process-level stdio round trip itself is not unit-tested.
- Definition of done: `cargo fmt -p openjiuwen-agent-teams`, `cargo test -p openjiuwen-agent-teams`, `cargo test -p openjiuwen-agent-teams --no-default-features`, `cargo clippy -p openjiuwen-agent-teams --all-targets -- -D warnings` green.

## Deferred

- HTTP/SSE transport: remote exposure is deployment's concern; stdio is the only transport, as in Python.
- Elicitation, prompts, resources and sampling: only the `tools` capability is advertised.
- Live client resolution: the MCP process serves a team through `crate::skill::connect_live_client` — a team this process hosts, else the attach socket the descriptor's `socket_path` or `OPENJIUWEN_TEAM_SOCKET` names; a descriptor that resolves neither is `Error::Config`.
  Cross-process attach through a shared sqlite store or a distributed transport is deferred.
- Structured content / annotations: results are a single text block (Python parity).
- Resumability, progress notifications and graceful shutdown hooks.

## Notes for Agents

- `SERVER_NAME` is the MCP `serverInfo` name; the logical registration name CLIs use is `openjiuwen-team` and lives in the spawn layer, not here.
- The tool surface is resolved per request from the connected client's scope (`scope()`), never frozen at construction; `TeamMcpServer` is cheap and `Send + Sync`.
- Python's `_ClientHolder` lazily connects and re-binds session context per call; in Rust the `TeamClient` is already constructed and bound, so `TeamMcpServer` only holds the `Arc`.
- `READ_INBOX` is the one operator-only pull tool; keep the name stable (CLI prompts reference it).
- Keep the operator result shapes byte-identical to the table above: consumers (external CLIs) parse them as JSON.
- `run_mcp_server_from_env` loads `OPENJIUWEN_TEAM_JOIN` and delegates to `run_mcp_server_with_descriptor`, which is the env-free seam tests drive.
