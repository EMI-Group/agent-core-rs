# src/skill — `team-member` skill CLI

## Intent

The non-interactive `team-member` command line an external process uses to join and drive a team.
Python parity source: `openjiuwen/agent_teams/skill/cli.py` (293 lines), `external/descriptor.py` (172 lines) and the operation methods of `external/client.py`.
Two scenarios are selected by the join descriptor's `scope`: `member` (a spawned CLI acting as a first-class member, driven through the real teammate tool set) and `operator` (default; an outside team-control surface over the task board, messaging and roster).
One operation per invocation, plain-text output, and the exit code carries the outcome.
The module is deliberately backend-agnostic: it talks to the `TeamStore`, `TaskBoard` and `Messager` seams, so the same code runs against the in-memory backend (tests) and a live team (the runtime bridge).
The live bridge is `live::connect_live_client`: it resolves a team this process hosts from the process-global runtime registry and builds the store-backed client over that team's own store, task board and messager.
When this process hosts no such team it falls back to the attach socket a hosting process published, so a CLI/MCP process that does not host the team can still drive it.

## API Surface

```rust
// descriptor.rs
pub const TEAM_JOIN_ENV: &str;            // re-exported from crate::constants
pub const TEAM_SOCKET_ENV: &str;         // "OPENJIUWEN_TEAM_SOCKET"; no crate-root or `skill` re-export, reach it as `openjiuwen_agent_teams::skill::descriptor::TEAM_SOCKET_ENV`
pub enum ClientScope { Operator, Member } // serde "operator" | "member", default operator
pub struct TeamJoinDescriptor { /* see below */ }
impl TeamJoinDescriptor {
    pub fn from_env() -> Result<Self>;
    pub fn from_env_map(get: impl Fn(&str) -> Option<String>) -> Result<Self>; // injectable env
    pub fn from_json(raw: &str) -> Result<Self>;
    pub fn to_json(&self) -> Result<String>;      // compact JSON
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self>;
    pub fn to_env(&self) -> Result<BTreeMap<String, String>>;
    pub fn configured_socket_path(&self) -> Option<&str>;                        // trimmed, blank = absent
    pub fn resolve_socket_path(&self) -> Option<PathBuf>;                        // descriptor, then TEAM_SOCKET_ENV
    pub fn resolve_socket_path_with(&self, get: impl Fn(&str) -> Option<String>) -> Option<PathBuf>; // injectable env
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

// socket_client.rs
pub struct SocketTeamClient { /* descriptor, path, timeout, tokio::sync::Mutex<Option<AttachClient>> */ }
impl SocketTeamClient {
    pub fn new(descriptor: TeamJoinDescriptor, path: impl Into<PathBuf>) -> Self;            // lazy: no I/O
    pub fn with_timeout(descriptor: TeamJoinDescriptor, path: impl Into<PathBuf>, timeout: Duration) -> Self;
    pub async fn connect(descriptor: TeamJoinDescriptor, path: impl AsRef<Path>) -> Result<Self>; // eager
    pub fn path(&self) -> &Path;
    pub fn descriptor(&self) -> &TeamJoinDescriptor;
    pub fn timeout(&self) -> Duration;
    pub async fn is_connected(&self) -> bool;  // never probes the socket
}   // impl TeamClient over one attach connection, one in-flight request at a time

// live.rs
pub fn connect_live_client(descriptor: &TeamJoinDescriptor) -> Option<Arc<dyn TeamClient>>;   // registry → live team, else published socket
pub async fn connect_socket_client(path: impl AsRef<Path>, descriptor: TeamJoinDescriptor) -> Result<Arc<dyn TeamClient>>; // eager socket attach
pub(crate) fn connect_with_registry(manager: Option<&TeamRuntimeManager>, descriptor: &TeamJoinDescriptor, get: impl Fn(&str) -> Option<String>) -> Option<Arc<dyn TeamClient>>; // both halves, no globals
pub(crate) fn resolve_live_client(manager: &TeamRuntimeManager, descriptor: &TeamJoinDescriptor) -> Option<Arc<dyn TeamClient>>;  // registry-free half
pub(crate) fn resolve_socket_client(descriptor: &TeamJoinDescriptor, get: impl Fn(&str) -> Option<String>) -> Option<Arc<dyn TeamClient>>; // socket half, needs a live socket file

// cli/
pub struct SkillCli;
impl SkillCli {
    pub fn parse_descriptor_args(argv: &[String]) -> Result<DescriptorArgs>;
    pub fn try_parse_from(argv: Vec<String>, scope: ClientScope) -> Result<Self>;
    pub fn descriptor_args(&self) -> &DescriptorArgs;   // connection flags the invocation carried
    pub fn command(&self) -> &CliCommand;              // the parsed subcommand, scope tree included
    pub async fn run(&self, client: Arc<dyn TeamClient>, out: &mut dyn Write) -> i32;
    pub async fn run_with(&self, client: Arc<dyn TeamClient>, out: &mut dyn Write, err: &mut dyn Write) -> i32;
}
pub struct DescriptorArgs { pub descriptor_file: Option<String>, pub descriptor_json: Option<String> }
pub fn run_skill_cli(argv: Vec<String>) -> i32;

// serve.rs — the child end of the member protocol (`team-member serve`)
pub const SERVE_DEFAULT_SESSION_ID: &str;   // "serve"
pub const SERVE_DEFAULT_MEMBER_NAME: &str;   // "member"
pub struct ServeContext { pub team_name: String, pub member_name: String, pub session_id: String,
                          pub workspace_root: PathBuf, pub checkpoint_enabled: bool,
                          pub checkpoint_dir: Option<PathBuf>, pub language: String }
impl ServeContext {
    pub fn from_env() -> Self;                                        // TEAM_NAME/MEMBER_NAME/SESSION_ID/workspace + checkpoint env
    pub fn from_env_map(get: impl Fn(&str) -> Option<String>) -> Self; // injectable, blank = absent
    pub fn with_session_id(self, session_id: impl Into<String>) -> Self; // blank keeps the env value
    pub fn with_checkpoint_dir(self, dir: impl Into<PathBuf>) -> Self;   // file-backed checkpoint root
    pub fn checkpoint_config(&self) -> CheckpointConfig;              // disabled | file(dir) | memory
    pub fn harness_context(&self) -> MemberHarnessContext;            // build mode
}
pub async fn serve_stdio() -> i32;                                    // NativeMemberFactory::from_env, /dev streams
pub async fn serve_with<R, W>(input: R, output: W, factory: Arc<dyn MemberHarnessFactory>) -> i32
    where R: AsyncRead + Unpin + Send, W: AsyncWrite + Unpin + Send + 'static;
pub async fn serve_with_context<R, W>(input: R, output: W, factory: Arc<dyn MemberHarnessFactory>, context: &ServeContext) -> i32; // testable form

// the parsed command tree, publicly matchable (exhaustive, clap internals stay private)
pub enum CliCommand { Member(MemberCommand), Operator(OperatorCommand), Serve(ServeCommand) }
pub enum ServeCommand { Serve }                     // scope-independent; parsed before any descriptor
pub enum MemberCommand { Inbox { watch: bool }, ViewTask { action: ViewAction, task_id: Option<String>, status: Option<String> }, ClaimTask { task_id: String, status: ClaimStatus }, SendMessage { to: String, content: String } }
pub enum OperatorCommand { Inbox { watch: bool }, Send { to: String, content: String }, Broadcast { content: String }, CreateTask { title: String, content: String, task_id: Option<String> }, Task { action: TaskCommand }, Claim { task_id: String }, Complete { task_id: String }, Update { task_id: String, title: Option<String>, content: Option<String> }, Members }
pub enum TaskCommand { List { status: Option<String> }, Claimable, Get { task_id: String } }
pub enum ViewAction { List, Get, Claimable }        // const fn as_str(self) -> &'static str
pub enum ClaimStatus { Claimed, Completed }         // const fn as_str(self) -> &'static str
```

The command enums are re-exported from `src/skill/mod.rs`; the clap root structs (`MemberCli`/`OperatorCli`/`ServeCli`), `PROG`, `parse_member`, `parse_operator`, `parse_serve` and `names_serve` stay crate-private.

`serve` is scope-independent: it is the child end of the JSON-lines member protocol (see `../member/protocol.rs`), takes its identity from the environment (the same variables `member::spawn` exports), needs no join descriptor, and is dispatched before the descriptor is loaded.
It reads one `MemberRequest` per line from stdin, writes `MemberResponse` frames (`ready`, `state`, `round`, `chunk`, `error`) to stdout, and keeps every diagnostic on stderr.
It also reads the two checkpoint variables `member::spawn` exports — `OPENJIUWEN_MEMBER_CHECKPOINTS` (`off`/`false`/`0`, case-insensitively, disables; absent or any other value enables) and `OPENJIUWEN_MEMBER_CHECKPOINT_DIR` (a file-backed store root; absent or blank keeps the process-local in-memory store) — and defaults both the way the served member documents them.
`serve`'s member runtime is wired like an in-process one: `build_member` goes through `runtime::checkpoints::wire_runtime` with the card id (`runtime::checkpoints::member_card_id` over the default spec, so the member name), the policy and store of `ServeContext::checkpoint_config` and the workspace root as the working directory, with no shared store (`shared = None`) because the child process owns its store.
A file-backed served member therefore writes `<dir>/<session>/<member>/checkpoint.json`, while an enabled member without a directory captures into a process-local in-memory store that dies with the process.

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
| `socket_path` | `Option<String>` | `None` |

`TeamClient` operations: `scope()`, `member_name()`, `read_inbox()`, `send_message(to, content)`, `create_task(title, content, task_id, dependencies)`, `update_task(task_id, title, content)`, `list_tasks(status)`, `claimable_tasks()`, `get_task(task_id)`, `claim_task(task_id)`, `complete_task(task_id)`, `list_members()`, `invoke_tool(name, arguments)`, `tools()`.
All async methods are `#[async_trait]`.

### Resolution order

`live::connect_live_client` resolves a client in this order:

1. the process-global runtime registry → `TeamRuntimeManager::live(team_name)` → the live team's backend → a `StoreTeamClient`;
2. otherwise the descriptor's `socket_path` field (then `OPENJIUWEN_TEAM_SOCKET`) when that path is an existing Unix socket → a lazily connecting `SocketTeamClient`;
3. otherwise `None`, which keeps the `no live team '<name>' in this process` message on stderr and exit code `1`.

The socket connect is lazy because the resolver is synchronous and may run inside a runtime (the MCP server's task), so it only `stat`s the path.
An unreachable or hung socket therefore surfaces as a typed `Transport`/`Timeout` error on the first operation, which the CLI still maps to exit `1`.
`live::connect_socket_client` is the eager counterpart for a caller that wants the connect failure up front.

## Command tree

Root: `team-member`, connection options `--descriptor-file <PATH>` and `--descriptor-json <JSON>` are `global = true` (accepted before or after the subcommand).
Descriptor precedence: `--descriptor-json` > `--descriptor-file` > `TEAM_JOIN_ENV` (`OPENJIUWEN_TEAM_JOIN`).
The scope-specific parser is chosen after the descriptor is loaded (two-stage parse).
`--help` is the exception: when the descriptor does not resolve, help is rendered from the default operator tree and still exits `0`.

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
- `0` — `--help`/`-h` was rendered on stdout (before or after the subcommand); this needs no join descriptor, so help wins over a missing descriptor.
- `0` — `serve` ended on a `stop` frame or end of input (the member was stopped and the settled state published).
- `1` — the operation was refused or the team could not be resolved (message on stderr).
- `1` — `serve` could not build or start its member harness (an `error` frame is written first).
- `2` — CLI-side error: unknown command/flag, missing required argument, malformed or absent descriptor.
- `130` — a cancelled operation (`Error::Cancelled`-style timeout/cancel is mapped here; no signal handler is installed).
- `--version` is not a flag in this port (the clap roots carry no `version` attribute), so it takes the unknown-argument path and exits `2`.

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
- `./socket_client.rs` → `SocketTeamClient`: the same seam over one attach connection to a team hosted by another process
- `./live.rs` → `connect_live_client` / `connect_with_registry`, `resolve_live_client` (this process's registry) and `resolve_socket_client` / `connect_socket_client` (a published attach socket)
- `./socket_client_tests.rs` → unix-only suite over a real temp socket: every `TeamClient` method, refusals, local checks, transport faults, reconnect
- `./live_socket_tests.rs` → unix-only suite for the resolution order (registry, descriptor path, environment, nothing configured)
- `./cli/mod.rs` → two-stage descriptor scan, `SkillCli::run`/`run_with`, exit codes, `run_skill_cli`, the scope-independent serve stage, delegation to `live::connect_live_client`
- `./cli/command.rs` → the clap trees (member/operator plus the scope-independent `serve` root), subcommand and value enums, help rendering (`display_text`, `serve_display_text`, `names_serve`)
- `./cli/member.rs` → member-scope dispatch (real teammate tool set)
- `./cli/operator.rs` → operator-scope dispatch (board, messaging, roster)
- `./cli/output.rs` → line renderers shared by both scopes
- `./serve.rs` → the `team-member serve` loop: env identity and checkpoint knobs, factory-built runtime wired with card id/policy/store, frame writer task, request dispatch, teardown
- `./serve_tests.rs` → the serve framing tests over an in-memory duplex pipe and a scripted fake factory/harness

## Test Strategy

- Unit tests beside each module (`#[cfg(test)]`), all offline and deterministic against `InMemoryTeamStore` + `TaskBoard` + `InProcessMessager`.
- `descriptor.rs`: defaults, unknown fields, file/JSON/env precedence, missing env message, malformed JSON message.
- `client.rs`: tool names and required schema properties for every dispatch/teammate-mode combination; operator tool order and schemas.
- `store_client.rs`: persisted message/task rows, published event topics and sender, inbox rendering + read cursor, over-limit rejection, member-scope claim/complete, `invoke_tool` shapes.
- `live.rs`: unknown team → `None` without touching the registry; registry install/clear through a local lock plus a restore guard; end-to-end attach to a team activated over the scripted fakes (roster, empty inbox, a created task claimable on the live team's own board); the socket half is covered by the two sibling suites below, which need real Unix sockets.
- `socket_client_tests.rs` (unix, 10 tests): a real temp socket (`<tmp>/s`, `AttachServer` + a recording handler answering every `AttachOp` with fixture data) drives one case per contract — `every_team_client_method_decodes` (all `TeamClient` methods, the ops recorded on the wire, the roster projection, `tools()` for the operator scope), `task_detail_miss_decodes_to_none`, `member_scope_tools_follow_the_descriptor` (plan mode + scheduled dispatch + depth 2), `refusals_keep_the_outcome_shapes` (`ok = false` never becomes `Err`), `remote_errors_keep_their_kind_and_the_connection`, `local_checks_run_before_the_wire` (nothing recorded on refusal), `missing_socket_is_a_typed_error` (`Transport` for lazy, eager and `connect_socket_client`), `concurrent_calls_are_safe_on_one_client` (8 joined calls, 8 ops), `a_transport_failure_drops_the_connection_for_a_reconnect` and `clean_shutdown_unlinks_the_socket_and_fails_typed`.
- `live_socket_tests.rs` (unix, 6 tests): the resolution order, driven with an explicit manager and an injected `get` closure — no process-global registry and no environment mutation. `socket_fallback_resolves_when_no_team_is_hosted`, `descriptor_path_wins_over_the_environment`, `environment_path_is_the_fallback` (live path, blank value, `None`), `nothing_configured_or_not_a_socket_resolves_to_none` (no path, regular file, nonexistent path), `the_in_process_registry_is_preferred_over_the_socket` (a team activated over the scripted fakes wins over a socket that answers a distinctive inbox text) and `unhosted_team_without_a_socket_stays_none`.
- `cli/`: parsing of every subcommand, `--descriptor-json` before and after the subcommand, unknown command → 2, refusal → 1 with the reason, happy-path output lines for every operator command, member `inbox`/`view_task`/`claim_task`/`send_message`, roster ordering; `parsed_shapes_are_publicly_inspectable` pins one operator and one member invocation through `SkillCli::command()`/`descriptor_args()`.
- `cli/` help parity: `--help` exits `0` with help on stdout and empty stderr (with a descriptor, before/after the subcommand, and with a malformed descriptor — help is answered before the descriptor is required); an unknown flag still exits `2` with stdout empty; `tests/bins.rs` pins the same contract on the real `team-member` binary with no descriptor at all.
- `serve_tests.rs` (15 tests over an in-memory duplex pipe and a scripted fake factory/harness): `start_frame_answers_ready_then_state`, `the_start_frame_session_id_is_authoritative`, `send_frame_streams_round_and_chunk_frames`, `unparsable_line_is_answered_and_the_loop_keeps_serving`, `send_before_start_is_answered_with_an_error`, `stop_frame_stops_the_member_and_exits_zero`, `end_of_input_stops_the_member_and_exits_zero`, `factory_failure_answers_with_an_error_and_exits_one`, `abort_pause_and_resume_are_forwarded_to_the_runtime`, plus `context_defaults_match_the_documented_contract` and `a_blank_start_session_id_keeps_the_context_session` for the env/default contract.
- `serve_tests.rs` checkpoint cases: `checkpoint_env_defaults_to_enabled_on_the_memory_store`, `checkpoint_env_reads_the_directory_and_the_switch` (`off`/`false`/`0` case-insensitively, blank as absent), `checkpoint_config_uses_the_file_backend_when_a_directory_is_known`, and `a_served_member_writes_its_checkpoint_under_the_configured_directory`, which drives `start` → `ready` → `stop` against a temp checkpoint directory and reads back the persisted `<dir>/s1/dev-1/checkpoint.json` (session, card id and cwd).
- `cli/` serve: `serve` and `serve --help` parse through the scope-independent root, `serve --help` reaches stdout and exits `0` with no descriptor, an unknown `serve` flag exits `2`, and `names_serve` ignores flag values.
- Definition of done: `cargo fmt -p openjiuwen-agent-teams`, `cargo test -p openjiuwen-agent-teams`, `cargo test -p openjiuwen-agent-teams --no-default-features`, `cargo clippy -p openjiuwen-agent-teams --all-targets -- -D warnings` green for these files.

## Deferred

- Hosting side: this node only *attaches*; there is no `AttachHandler` adapter or `AttachServer` lifecycle here, so publishing `OPENJIUWEN_TEAM_SOCKET` and serving the socket belong to the runtime/host layer.
- Security: the attach socket has no authentication or TLS beyond the filesystem (the server binds a `0600` socket inside an owner-only directory), and there is no Windows/named-pipe or broker transport.
- No streaming or subscriptions on the socket: every call is one request/response, so `--watch` stays a one-shot inbox read, and `tools()` stays local metadata rather than a remote catalogue read.
- No cross-process attach through a shared sqlite DB; a descriptor whose team is hosted elsewhere and whose socket path does not resolve reports `no live team '<name>' in this process` and exits `1`.
- `serve` reopens the process streams as `/dev/stdin` and `/dev/stdout` (the workspace does not enable tokio's `io-std` feature), so on non-Unix targets it reports `Unsupported` and exits `1`.
- `serve` builds the member from `DeepAgentSpec::default()`: only identity (team/member/session/workspace), the checkpoint settings and the model environment cross the process boundary, so per-member tools, rails, MCP servers and prompts configured on the leader side are not carried into the child yet.
- Vendor CLI member adapters (`claude`/`codex` protocol translation), hybrid WebSocket publishing, file-backed SQLite descriptors and workspace metadata tools are not ported (see the crate "Deferred" list).

## Notes for Agents

- Python loads the descriptor with `argparse.parse_known_args`; here `parse_descriptor_args` scans the raw argv for the two connection flags so an unknown subcommand still reaches the scope parser (which owns exit code 2).
- `--help`/`-h` is answered by the scope parser (`cli::command::display_text`) *before* the descriptor is required, so help reaches stdout with exit `0`; only an explicit help flag counts, a bare invocation still reports the missing descriptor and exits `2`.
- `openjiuwen-team-mcp` installs a stderr log subscriber (`LoggingConfig::from_env().with_writer(LogWriter::Stderr)`) because stdout is the MCP wire, so a descriptor/client/session failure is diagnosable while stdout stays clean.
- `SkillCli::run` writes both stdout and failures to one writer; `run_skill_cli` gives the process stdout/stderr separately.
- `StoreTeamClient` deliberately bypasses the task board for `update_task` (there is no board edit operation): it applies a `TaskPatch` through `TaskBoard::store()` after refusing non-`pending`/`blocked` rows.
- Member-scope `claim_task` claims for the descriptor's `member_name`; the one-active-task check of the native tool is a runtime policy and is not duplicated here.
- `store_client.rs` is legitimately long (~1200 lines, half of it its inline test suite): the `TeamClient` impl is one contract whose operations are each a store/board call plus the event it publishes, so splitting it would scatter that contract. The top-of-file note records the reason.
- The live-team resolver is `live::connect_live_client`, the single wiring point both the `team-member` CLI (`cli::live_client`) and the `openjiuwen-team-mcp` server go through; `live::connect_with_registry` is the same order with the registry slot and the environment lookup injected, which is what keeps the tests free of process globals.
- `cli::live_client` only delegates: `live::connect_live_client` resolves through `crate::runtime::try_global_registry()` → `TeamRuntimeManager::live(team_name)` → `TeamAgent::backend()`, and otherwise through the descriptor's socket path, so a descriptor naming a team this process does not host *and* publishing no socket reports exit `1`.
- `SocketTeamClient` connects lazily on purpose: the resolver is synchronous and may run inside a runtime, so `connect_live_client` only `stat`s the path and the bounded connect happens on the first operation.
  A `Transport`/`Timeout` failure drops the connection so the next call reconnects, while an error the remote team reported in-band keeps both its kind and the healthy connection.
- The socket client's outcome types and local checks (`send_message` target/content, `MemberSummary` projection, `tools()` rule) are the same as `StoreTeamClient`'s, so the CLI and MCP text and exit codes do not change with the transport.
- `socket_client.rs` is ~450 lines including its module documentation (no test module inside); no split is needed.
