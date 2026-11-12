# src/skill — `team-member` skill CLI

## Intent

The non-interactive `team-member` command line an external process uses to join and drive a team.
Python parity source: `openjiuwen/agent_teams/skill/cli.py` (293 lines), `external/descriptor.py` (172 lines) and the operation methods of `external/client.py`.
Two scenarios are selected by the join descriptor's `scope`: `member` (a spawned CLI acting as a first-class member, driven through the real teammate tool set) and `operator` (default; an outside team-control surface over the task board, messaging and roster).
One operation per invocation, plain-text output, and the exit code carries the outcome.
The module is deliberately transport-agnostic: it talks to the `TeamStore`, `TaskBoard` and `Messager` seams, so the same code runs against the in-memory backend (tests) and a live team (the runtime bridge).
The live bridge is `live::connect_live_client`: it resolves the joined team from the process-global runtime registry and builds the store-backed client over that team's own store, task board and messager.

## API Surface

```rust
// descriptor.rs
pub const TEAM_JOIN_ENV: &str;            // re-exported from crate::constants
pub enum ClientScope { Operator, Member } // serde "operator" | "member", default operator
pub struct TeamJoinDescriptor { /* see below */ }
impl TeamJoinDescriptor {
    pub fn from_env() -> Result<Self>;
    pub fn from_env_map(get: impl Fn(&str) -> Option<String>) -> Result<Self>; // injectable env
    pub fn from_json(raw: &str) -> Result<Self>;
    pub fn to_json(&self) -> Result<String>;      // compact JSON
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self>;
    pub fn to_env(&self) -> Result<BTreeMap<String, String>>;
}

// client.rs
pub trait TeamClient { /* async ops listed below */ }
pub struct SendOutcome { pub ok: bool, pub message_id: Option<String>, pub reason: Option<String> }
pub struct CreateOutcome { pub ok: bool, pub task_id: Option<String>, pub reason: Option<String> }
pub struct OpOutcome { pub ok: bool, pub reason: Option<String> }
pub struct MemberSummary { pub member_name: String, pub role: TeamRole, pub status: MemberStatus }
pub struct TeamToolDef { pub name: String, pub description: String, pub input_schema: Value }
pub fn member_tools_for(mode: MemberMode, dispatch: DispatchMode, max_subtask_depth: usize) -> Vec<TeamToolDef>;
pub fn operator_tools() -> Vec<TeamToolDef>;

// store_client.rs
pub struct StoreTeamClient;   // impl TeamClient over Arc<dyn TeamStore> + Arc<TaskBoard> + Arc<dyn Messager>
impl StoreTeamClient { pub fn new(descriptor: TeamJoinDescriptor, store: Arc<dyn TeamStore>, taskboard: Arc<TaskBoard>, messager: Arc<dyn Messager>) -> Self }

// live.rs
pub fn connect_live_client(descriptor: &TeamJoinDescriptor) -> Option<Arc<dyn TeamClient>>;   // crate::runtime::try_global_registry() → manager.live(team) → TeamAgent::backend()
pub(crate) fn resolve_live_client(manager: &TeamRuntimeManager, descriptor: &TeamJoinDescriptor) -> Option<Arc<dyn TeamClient>>;  // registry-free half

// cli/
pub struct SkillCli;
impl SkillCli {
    pub fn parse_descriptor_args(argv: &[String]) -> Result<DescriptorArgs>;
    pub fn try_parse_from(argv: Vec<String>, scope: ClientScope) -> Result<Self>;
    pub async fn run(&self, client: Arc<dyn TeamClient>, out: &mut dyn Write) -> i32;
    pub async fn run_with(&self, client: Arc<dyn TeamClient>, out: &mut dyn Write, err: &mut dyn Write) -> i32;
}
pub struct DescriptorArgs { pub descriptor_file: Option<String>, pub descriptor_json: Option<String> }
pub fn run_skill_cli(argv: Vec<String>) -> i32;
```

`TeamJoinDescriptor` (serde snake_case, `#[serde(default)]`, unknown fields ignored):

| field | type | default |
| --- | --- | --- |
| `session_id` | `String` | `""` |
| `team_name` | `String` | `""` |
| `member_name` | `String` | `""` |
| `role` | `TeamRole` | `teammate` |
| `scope` | `ClientScope` | `operator` |
| `language` | `String` | `"cn"` |
| `dispatch_mode` | `DispatchMode` | `autonomous` |
| `enable_member_messaging` | `bool` | `false` |
| `max_subtask_depth` | `usize` | `0` |
| `teammate_mode` | `MemberMode` | `build_mode` |
| `db_config` | `Value` | `{}` |
| `transport_config` | `Value` | `{}` |
| `workspace_config` | `Value` | `{}` |
| `workspace_path` | `Option<String>` | `None` |

`TeamClient` operations: `scope()`, `member_name()`, `read_inbox()`, `send_message(to, content)`, `create_task(title, content, task_id, dependencies)`, `update_task(task_id, title, content)`, `list_tasks(status)`, `claimable_tasks()`, `get_task(task_id)`, `claim_task(task_id)`, `complete_task(task_id)`, `list_members()`, `invoke_tool(name, arguments)`, `tools()`.
All async methods are `#[async_trait]`.

## Command tree

Root: `team-member`, connection options `--descriptor-file <PATH>` and `--descriptor-json <JSON>` are `global = true` (accepted before or after the subcommand).
Descriptor precedence: `--descriptor-json` > `--descriptor-file` > `TEAM_JOIN_ENV` (`OPENJIUWEN_TEAM_JOIN`).
The scope-specific parser is chosen after the descriptor is loaded (two-stage parse).

### `scope = "member"`

| command | arguments | notes |
| --- | --- | --- |
| `inbox` | `[--watch]` | renders unread messages; `--watch` is accepted for parity but is one-shot (streaming deferred) |
| `view_task` | `[--action list\|get\|claimable] [--task-id <ID>] [--status <STATUS>]` | default action `list`; `get` requires `--task-id` |
| `claim_task` | `<TASK_ID> <STATUS: claimed\|completed>` | dispatched through `invoke_tool("claim_task", …)` |
| `send_message` | `<TO> <CONTENT>` | `TO="*"` broadcasts |

### `scope = "operator"` (default)

| command | arguments | stdout on success |
| --- | --- | --- |
| `inbox` | `[--watch]` | inbox text |
| `send` | `<TO> <CONTENT>` | `sent <message_id> -> <to>` |
| `broadcast` | `<CONTENT>` | `broadcast <message_id>` |
| `create_task` | `<TITLE> <CONTENT> [--task-id <ID>]` | `created <task_id>` |
| `task list` | `[--status <STATUS>]` | `[<task_id>] <status> <title> (<assignee or ->)` per line |
| `task claimable` | — | `[<task_id>] <title>` per line |
| `task get` | `<TASK_ID>` | 7-line block: `id:`, `title:`, `status:`, `assignee:`, `blocked_by:`, `blocks:`, `content:` |
| `claim` | `<TASK_ID>` | `claimed <task_id>` |
| `complete` | `<TASK_ID>` | `completed <task_id>` |
| `update` | `<TASK_ID> [--title <T>] [--content <C>]` | `updated <task_id>` |
| `members` | — | `<member_name> (<role>) [<status>]` per line, sorted by name |

Failure output: the operation's `reason` on stderr (or on `out` when `run` is used without a separate stderr).

## Exit codes

- `0` — the operation succeeded.
- `1` — the operation was refused or the team could not be resolved (message on stderr).
- `2` — CLI-side error: unknown command/flag, missing required argument, malformed descriptor.
- `130` — a cancelled operation (`Error::Cancelled`-style timeout/cancel is mapped here; no signal handler is installed).

## Rendering contracts (wire-visible)

- Inbox line: `[<timestamp>] <sender> -> <you|*> : <content>` (`*` for a broadcast row), ordered by timestamp then id. Reading advances the per-member broadcast watermark and marks direct rows read. Empty inbox renders `(inbox empty)`.
- `send_message` rejects content longer than `MAX_MESSAGE_CONTENT_CHARS` with `content too long: <n> characters exceeds the limit of <max>`.
- `task get` lines use the 7 labels above verbatim; empty assignee/link lists render `-`.
- Member messages persist through `TeamStore::insert_message` and publish `EventPayload::Message` / `EventPayload::Broadcast` on the `Message` topic with `sender_id = member_name`.

## Constraints

- Only `src/skill/**` is edited; `Cargo.toml`, `src/lib.rs`, sibling modules and `src/skill/CONTEXT.md`'s parent are out of scope.
- `openjiuwen_common::error::{Error, Result}` only; `Error::config` for a missing/unreadable connection source, `Error::validation` for bad JSON or caller input, `Error::not_found` for unknown tools/tasks.
- CLI writes to the supplied `&mut dyn Write`; no `println!`/`eprintln!` in this module. `tracing` only.
- No `unsafe`, no `unwrap`/`expect` outside `#[cfg(test)]`.
- Files ≤1000 lines; split member/operator dispatch and output helpers.

## Routing Table

- `./mod.rs` → module wiring and the flat re-export of the node's public surface
- `./descriptor.rs` → `TeamJoinDescriptor`, `ClientScope`, env/file/JSON loading
- `./client.rs` → `TeamClient` trait, outcome types, tool catalogue (`member_tools_for`, `operator_tools`)
- `./store_client.rs` → `StoreTeamClient`: the store/task-board/messager implementation of the client
- `./live.rs` → `connect_live_client` / `resolve_live_client`: attach to a team hosted in this process through `crate::runtime::try_global_registry()`
- `./cli/mod.rs` → two-stage descriptor scan, `SkillCli::run`/`run_with`, exit codes, `run_skill_cli`, delegation to `live::connect_live_client`
- `./cli/command.rs` → the two clap trees (member/operator), subcommand and value enums
- `./cli/member.rs` → member-scope dispatch (real teammate tool set)
- `./cli/operator.rs` → operator-scope dispatch (board, messaging, roster)
- `./cli/output.rs` → line renderers shared by both scopes

## Test Strategy

- Unit tests beside each module (`#[cfg(test)]`), all offline and deterministic against `InMemoryTeamStore` + `TaskBoard` + `InProcessMessager`.
- `descriptor.rs`: defaults, unknown fields, file/JSON/env precedence, missing env message, malformed JSON message.
- `client.rs`: tool names and required schema properties for every dispatch/teammate-mode combination; operator tool order and schemas.
- `store_client.rs`: persisted message/task rows, published event topics and sender, inbox rendering + read cursor, over-limit rejection, member-scope claim/complete, `invoke_tool` shapes.
- `live.rs`: unknown team → `None` without touching the registry; registry install/clear through a local lock plus a restore guard; end-to-end attach to a team activated over the scripted fakes (roster, empty inbox, a created task claimable on the live team's own board).
- `cli/`: parsing of every subcommand, `--descriptor-json` before and after the subcommand, unknown command → 2, refusal → 1 with the reason, happy-path output lines for every operator command, member `inbox`/`view_task`/`claim_task`/`send_message`, roster ordering.
- Definition of done: `cargo fmt -p openjiuwen-agent-teams`, `cargo test -p openjiuwen-agent-teams`, `cargo test -p openjiuwen-agent-teams --no-default-features`, `cargo clippy -p openjiuwen-agent-teams --all-targets -- -D warnings` green for these files.

## Deferred

- Cross-process attach: only a team hosted in *this* process resolves (through `crate::runtime::try_global_registry()`); reaching a team hosted in another process through a shared sqlite DB or a transport is deferred, and an unhosted team reports `no live team '<name>' in this process` and exits `1`.
- `--watch` streaming: a single inbox read is performed; the event-driven re-render loop waits on the runtime's watch hook.
- External CLI members (`claude`/`codex`), hybrid WebSocket publishing, file-backed SQLite descriptors and workspace metadata tools are not ported (see the crate "Deferred" list).

## Notes for Agents

- Python loads the descriptor with `argparse.parse_known_args`; here `parse_descriptor_args` scans the raw argv for the two connection flags so an unknown subcommand still reaches the scope parser (which owns exit code 2).
- `SkillCli::run` writes both stdout and failures to one writer; `run_skill_cli` gives the process stdout/stderr separately.
- `StoreTeamClient` deliberately bypasses the task board for `update_task` (there is no board edit operation): it applies a `TaskPatch` through `TaskBoard::store()` after refusing non-`pending`/`blocked` rows.
- Member-scope `claim_task` claims for the descriptor's `member_name`; the one-active-task check of the native tool is a runtime policy and is not duplicated here.
- `store_client.rs` is legitimately long (~1200 lines, half of it its inline test suite): the `TeamClient` impl is one contract whose operations are each a store/board call plus the event it publishes, so splitting it would scatter that contract. The top-of-file note records the reason.
- The live-team resolver is `live::connect_live_client`, the single wiring point both the `team-member` CLI (`cli::live_client`) and the `openjiuwen-team-mcp` server go through; `live::resolve_live_client` is its registry-free half, which is what the unit tests drive against a local manager.
- `cli::live_client` only delegates: the team is resolved through `crate::runtime::try_global_registry()` → `TeamRuntimeManager::live(team_name)` → `TeamAgent::backend()`, so a descriptor naming a team this process does not host reports exit `1`.
