# openjiuwen-agent-teams

## Intent

Leader/teammate multi-agent runtime: the Rust port of the Python package `openjiuwen.agent_teams` (v0.1.20, ~108k lines / 336 files).
The crate owns team specs and the expert catalog, the leader/teammate runtime (dispatch, task board, mailbox, member lifecycle, workspace isolation), HITL forwarding, the harness-protocol adapter, the organisation layer, model-pool routing and KV-cache affinity, observability and the monitor status stream, cross-process attach, sqlite persistence, the `team-member` CLI (including its `serve` child-process server) and the `openjiuwen-team-mcp` MCP server.
A member runs in-process, as an OS-process child speaking the JSON-lines member protocol, as an external CLI adapter, or over SSH; its workspace is a local directory or a git worktree.
Transport stays in-process (`InProcessMessager`); persistence is in-memory or SQLite.
Parity rule: serialized JSON names, defaults, status values, tool/command names and observable lifecycle behaviour follow Python; internals are idiomatic Rust.

## API Surface

`src/lib.rs` re-exports the public surface flat at the crate root, grouped as follows (the full item lists are in that file and in each module's `CONTEXT.md`).

```rust
// identity, defaults, addressing
pub use constants::{is_reserved_name, parse_mention, BROADCAST_TARGET, DEFAULT_LANGUAGE, …};
// `DEFAULT_*` names, HUMAN_AGENT_MEMBER_NAME, RESERVED_MEMBER_NAMES, MAX_MESSAGE_CONTENT_CHARS,
// OPENJIUWEN_HOME_ENV, TEAM_JOIN_ENV, USER_PSEUDO_MEMBER_NAME
pub use schema::*;  pub use storage::*;  pub use workspace::*;  pub use member::*;
pub use taskboard::{…};    // TaskBoard, TaskBoardLimits, TaskMutation, TaskOpResult, ReviewTally, detect_cycle
pub use messager::{…};     // Messager, InProcessMessager, EventSink, create_messager, peer/transport configs
pub use experts::{…};      // ExpertCatalog/ExpertRef, package loading, template composition, process catalog slot
pub use interaction::{…};  // InteractPayload, DeliverResult, InteractGate, inbox, approval bridge, routing grammar
pub use runtime::{…};      // TeamAgent, TeamBackend, TeamBuildDeps, SpecBuild, runtime manager/pool,
                           // member spawner, attach host, model allocator/router, org binding, metadata parsers,
                           // checkpoint config (CheckpointConfig, CheckpointStoreConfig, checkpoint_config),
                           // health projection (TeamHealth, MemberHealth)
pub use harness::{…};      // MemberHarnessFactory/Context, HarnessMemberRuntime, NativeMemberFactory,
                           // ModelResolver/EnvModelResolver, NativeCheckpoint, mapping helpers,
                           // checkpoint store (MemberCheckpointPolicy, CheckpointResume, MemberCheckpointRecord,
                           // MemberCheckpointStore, InMemoryMemberCheckpointStore, FileMemberCheckpointStore,
                           // MemberCheckpointSink)
pub use mcp::{…};          // TeamMcpServer, build_mcp_server, OPERATOR_INSTRUCTIONS, SERVER_NAME
pub use skill::{…};        // TeamClient + store/socket clients, skill CLI, join descriptor, tool catalogues,
                           // connect_live_client/connect_socket_client, outcome types
// organisation, model routing, KV-cache affinity, observability
pub use org::*;  pub use models::*;  pub use kv_cache::*;  pub use observability::*;
// team memory: the wiring handle, its metadata config, and the member tool surface
pub use memory::{…};       // TeamMemoryWiring, MemoryConfig, MemoryStoreConfig, MemoryExtractorKind,
                           // memory_config, MEMORY_METADATA_KEY, TEAM_MEMORY_FILE, TeamMemoryTool,
                           // team_memory_tools, OtelSpanSink
// live status stream and attach: item-by-item, because a glob would re-export `monitor::stream`
// (collides with `schema::stream`) and `attach::protocol` (collides with `member::protocol`)
pub use monitor::{…};      // TeamMonitor, StatusStream, MonitorEmitter/Receiver, BackpressurePolicy, snapshot_events
pub use attach::{…};       // AttachClient, AttachServer, AttachHandler, AttachOp, frames, errors, limits, version
pub use openjiuwen_harness_protocol::HarnessState;   // re-export, never redefine
```

Module roles beyond the kernel seams, one line each:

- `org` → organisation chart, reporting lines, delegation router, org task pool, org mailbox, swarmflow, fork/compact.
- `models` → `ModelPool` validation and lookup, `ModelRouter` strategies (`by_name`, `round_robin`, `least_busy`, `rule_based`) and per-member allocation.
- `kv_cache` → member cache identity, fork lineage, the advisory env bag and release/cleanup hooks.
- `observability` → the `ObservabilityHooks` seam and an in-memory span sink; the `otel` feature adds structured hooks over the same seam.
- `memory` → `TeamMemoryWiring` (the `openjiuwen_memory::team` facade + lifecycle hooks + per-member toolkits + `TEAM_MEMORY.md` export/import), the tolerant `metadata["memory"]` config, the four member-memory tools as `Tool` handles, and the tracing/`otel` observer bridge.
- `monitor` → plain-text status stream for a live team (`TeamMonitor`, `StatusStream`, backpressure policy).
- `attach` → the Unix-socket wire contract (`protocol`) plus `AttachClient` and `AttachServer`.
- `harness::checkpoint_store` → `MemberCheckpointPolicy`/`CheckpointResume` capture knobs, `MemberCheckpointRecord`, the `MemberCheckpointStore` seam (`InMemoryMemberCheckpointStore`, `FileMemberCheckpointStore`) and the `MemberCheckpointSink` adapter.
- `runtime::health` → `TeamHealth`/`MemberHealth`, the member-liveness projection behind `TeamBackend::health()` and `TeamAgent::health()`.

Checkpoint and health seams: `TeamBuildDeps::with_checkpoint_store` injects a store and `has_injected_checkpoints` reports it.
`HarnessMemberRuntime` exposes `with_card_id`, `with_checkpoint_store`, `with_checkpoint_policy`, `with_cwd`, `latest_checkpoint` and `capture_checkpoint`.
`MemberHarnessContext` carries `card_id` alongside `member_session_id(host, member)`.

Team memory seams: `TeamBuildDeps::with_memory` injects a `TeamMemoryWiring` and `has_injected_memory` reports it; the build path resolves `metadata["memory"]` into one wiring per team and exposes it as `TeamAgent::memory()`.
`MemberHarnessContext::with_memory_toolkit` carries a member's `MemberMemoryToolkit` and `memory_tools()` projects it onto the four `Tool` handles the native factory mounts; `skill::ServeContext::checkpoint_config` and `member::spawn::{CHECKPOINT_ENV, CHECKPOINT_DIR_ENV}` carry the child-process checkpoint contract.

Binaries: `team-member` (`src/bin/team-member.rs` → `skill::run_skill_cli`, exit `0/1/2/130`) and `openjiuwen-team-mcp` (`src/bin/openjiuwen-team-mcp.rs`, `mcp` feature, stdio MCP server).

## Cross-module seams (change only together with every dependent node)

Public data types are `#[serde(rename_all = "snake_case", default)]`, tolerate unknown fields and default collections to empty.

- `storage` → everyone: `TeamStore` is `fn name()` plus 23 async methods over the team row, roster (with CAS `transition_member`), tasks (with `transition_task`), dependency edges, review votes, mailbox and the per-member read watermark; `InMemoryTeamStore` is the deterministic default, `SqliteTeamStore` (feature `sqlite`) the persistent one, and `open_store(&StorageSpec)` maps `memory`/`sqlite` while reporting `postgresql`/`mysql`/unknown kinds as `Error::Unsupported`.
- `messager` → `runtime`, `member`: `Messager` (`name`, `subscribe`, `unsubscribe`, `publish`, `send`, `subscribe_transport`) over `EventSink = Arc<dyn Fn(EventMessage) + Send + Sync>`; `create_messager(&TransportSpec)` builds `InProcessMessager` for `inprocess` and rejects `pyzmq`/`hybrid`; a sink failure is logged and a blank `sender_id` is stamped with the sending member.
- `member` → `runtime`: `MemberRuntime` (`session_id`, `state`, `start`, `stop`, `send`, `abort`, `pause`, `resume`, `subscribe`), `MemberObserver` (`on_state`/`on_round`/`on_chunk`), `TeamMember` (roster CAS plus a cached status projection), `MemberSpawner`/`MemberSpawnOptions` with the in-process, process, external-CLI and SSH spawners, and `InProcessSpawnHandle`.
- `runtime` entry points used by CLI/MCP/attach/host: `TeamAgent::{card, invoke, stream, start, stop, state, interact, deliver, snapshot, backend, spawn_member, shutdown_member, auto_start_all, create_task, send_message, members, monitor_stream, signals, kv_cache, kv_env, org, attach_socket, create_org_task, delegate_org_task}`, `TeamRuntimeManager::{activate, live, live_state, interact, finalize_member, stop_team, remove, shutdown, snapshot, pool}`, and `TeamBuildDeps { harness_factory, store, store_injected, workspace, language: Option<String>, observability_hooks, kv_cache, routing, checkpoints, memory, memory_observer }`.

## Constraints

- Only `crates/openjiuwen-agent-teams/**` is edited; root manifest, sibling crates and `docs/` are out of scope.
- Dependency direction: this crate depends on `openjiuwen-harness`, `openjiuwen-harness-protocol`, `openjiuwen-core`/`openjiuwen-multiagent`/`openjiuwen-agent`/`openjiuwen-session`/`openjiuwen-memory`, `openjiuwen-common` and `openjiuwen-foundation`, plus the optional `openjiuwen-extensions` behind `otlp`; nothing may depend on this crate, and there are no `openjiuwen-cli`/`openjiuwen` imports.
- Async-first tokio; `#[async_trait]` for object-safe async seams; locks never held across `.await` (`parking_lot` for short sections, `tokio::sync` for awaitable coordination); cancellation via `tokio_util::sync::CancellationToken`.
- Errors: `openjiuwen_common::error::{Error, Result}` only; user input never panics; `Error::validation` for user-supplied names and knobs, `Error::config` for missing build inputs, `Error::not_found` for unknown members/tasks, and `Error::Unsupported` naming the Python capability for anything deferred.
- Serde on every public data type (snake_case, unknown fields tolerated); Cards crossing the wire are `openjiuwen_agent::AgentCard`.
- `tracing` only: no `println!`/`eprintln!` in library code, and the only stdout writes are the CLI's rendered lines and the `team-member serve` frame stream (its diagnostics go to stderr).
- No `unsafe`. No undocumented global mutable state: the process-global expert catalog and the process-global runtime registry (team name → runtime manager) are the two documented singletons; tests construct local instances.
- Files around ≤1000 lines; split by responsibility; a legitimately long file carries a top-of-file note.
- Every feature combination builds offline: default `mcp`, plus the opt-in `sqlite`, `otel`, `otlp`, `distributed` and `live-tests`; `--no-default-features` builds without `rmcp`/`sqlx`.
- `otlp` is the one optional dependency on a same-or-higher framework crate: `openjiuwen-extensions` (layer 5) with its own `otel` feature, so it pulls the `opentelemetry` stack; `crates/openjiuwen-extensions` has no dependency on this crate, so the edge is acyclic and the SDK stays out of every default build.
- Tests are offline and deterministic: scripted harnesses or a loopback mock model, temp directories, in-process transport, real sockets/binaries only where a suite says so, bounded waits, no `#[ignore]`.

## Behaviour parity notes (wire-visible)

- `MemberStatus`: `unstarted, starting, ready, busy, paused, stopped, restarting, shutdown_requested, shut_down, dormant, error`.
- `TaskStatus`: `pending, blocked, planning, in_progress, in_review, completed, cancelled`; terminal = `completed|cancelled`; claimable = `pending`.
- `ExecutionStatus`: `idle, starting, running, cancel_requested, cancelling, cancelled, completing, completed, failed, timed_out`.
- Topics: `session:<sid>:team:<team>:{team|task|message}` with `%`→`%25` and `:`→`%3A` escaping.
- Addressing: `"*"` = broadcast, `targets=[…]` = multicast, `"leader"`/`"user"` are role placeholders resolved at delivery; `parse_mention` grammar `@name body`.
- Member names match `^[a-z][a-z0-9_-]*$` and may not collide with the reserved names (`user`, `team_leader`, `human_agent`).
- Review gate: an author completing a task enters `in_review` when reviewers exist (autonomous: first verdict; scheduled: per-round vote tally); otherwise it is `completed` and dependent edges unblock (`blocked` → `pending`).
- Broadcast read state is a single per-member watermark, never a row flag.
- `team-member` exit codes: `0` success and `--help`, `1` refused/unhosted team or a failed served member start, `2` CLI-side error (unknown command/flag, missing argument, malformed or absent descriptor), `130` a cancelled operation.
- `team-member serve` contract: one JSON object per line on stdin/stdout (`start`/`send`/`abort`/`pause`/`resume`/`stop` in, `ready`/`state`/`round`/`chunk`/`error` out); stdout carries protocol frames only and diagnostics go to stderr; a `start` answers `ready` then one `state`; an unparsable line is answered with `error` and the loop keeps serving; `stop` or EOF exits `0`; a runtime that cannot be built or started exits `1` after one `error` frame.
- Attach protocol: one `{"v":1,"id":<n>,"op":"…","args":{…}}` request per connection answered by one `{"v":1,"id":<n>,"ok":…}` response; an op-level error travels in band and keeps the connection, a frame-level error is answered once and ends that connection; `MAX_FRAME_BYTES` (1 MiB) bounds one line; `stop_team` answers `{"stopping": true}`; `OPENJIUWEN_TEAM_SOCKET` is the read-side fallback when the metadata names no socket (the host never sets it).
- Run completion: a settled run publishes one `EventPayload::TeamCompleted(TeamCompletedEvent { team_name, reason: Some("settled") })` on `TeamTopic::Team` in addition to the unchanged `team.completed` stream marker, which remains the last frame, and a cancelled run publishes neither.
- `TeamAgentSpec::metadata` keys, all optional and tolerant (a malformed value logs and keeps the default, except `organization`, where a configured-but-broken chart fails the build):
  - `spawn` → bare `MemberSpawnOptions` (`{"kind": "in_process"|"process"|"external_cli"|"ssh", …}`) or `{"options": …, "fork": …}`; selects the spawner.
  - `fork` → `{"parent": "dev-1", "task": {…}|null, "tools": {…}}`, also read from `spawn.fork`; forks a member from `parent` and registers its KV lineage.
  - `workspace` → `"worktree"` or `{"isolation": "worktree"}`; anything else stays local.
  - `organization` → `{"spec": OrganizationSpec, "rules": DelegationRules}`; installs the `OrgBinding`.
  - `kv_cache` → `{"enabled": bool, "max_entries": n}` (absent means enabled).
  - `attach` → `"/path/team.sock"` or `{"socket"|"path": "…", "required": bool}`; `OPENJIUWEN_TEAM_SOCKET` is the fallback and an explicit metadata socket wins.
  - `checkpoint` → `{"enabled", "on_turn", "on_state", "on_stop", "resume"}` policy, optionally wrapped as `{"policy": {…}, "store": "memory"|"file", "dir": "…"}`; absent means an enabled policy with a team-shared in-memory store, and a malformed value warns and keeps the default.
  - `memory` → `{"enabled", "store", "extractor", "export_on_stop", "import_on_build", "trace", "hooks"}`; absent or `{"enabled": false}` allocates no wiring at all, and a malformed value warns and keeps the default (on: in-memory store, rule extractor, export and import on, tracing on).

## Deferred (documented, `Error::Unsupported` through seams)

- Transports: ZeroMQ/pyzmq, hybrid WebSocket publishing and Pulsar — `Messager` seam plus `create_messager` rejection.
- Storage: PostgreSQL and MySQL `TeamStore` backends; distributed workspace transport (only the `RemoteDispatch` seam and its protocol types ship).
- OTLP span export for everything except the team-memory bridge: the `otel` feature is the in-memory/tracing seam, `otlp` forwards team-memory events into `openjiuwen-extensions`' `OtelTraceSink`, and the member/round/task spans stay on the in-memory seam.
- Monitor TUI and DB-backed monitor readers: the text stream is the contract and presentation belongs to the CLI front-end.
- Attach security: filesystem permissions only (a `0600` socket in an owner-only directory), no token, `SO_PEERCRED` check, TLS or named-pipe transport.
- Vendor CLI harness translation: `claudecode`/`codex`/`dsh` stay `Unsupported` in `openjiuwen-harness-providers`; this crate ships the generic stdio child and a scriptable external-CLI adapter.
- Cross-process organisation/team federation: shared-store resume, leader-turn queues, summary teams and the org transport handshake.
- LLM-driven context compaction (fork/compact emits the deterministic template summary) and `IntelliRouter`/builtin-model selection (four router strategies and two pool strategies are the full surface).
- Platform-gated typed refusals: on non-Unix targets `mcp::stdio::serve_impl`, `skill::serve::process_streams`, `attach::AttachServer::serve` and `attach::AttachClient::call` return `Error::Unsupported`, and `runtime::attach_host` refuses the `invoke_tool` attach op with an `AttachError` of kind `Unsupported`.
- `ModelPoolStrategy` has no `intelli_router` variant and `runtime::allocator::ModelAllocator::from_state_dict` restores an unknown strategy as `RoundRobin`, so an IntelliRouter spec fails to deserialize and an IntelliRouter state dict mis-restores.
- Round/pause-boundary memory extraction is unwired: `TeamMemoryWiring::on_turn` has no runtime caller, so memory is only written by lifecycle hooks and flushed/exported on `stop`/`snapshot`.

## Routing Table

- `./src/schema/` → spec/blueprint types, team runtime types, statuses/transitions, task records, event payloads, stream frames (`src/schema/CONTEXT.md`)
- `./src/experts/` → `ExpertCatalog`, `ExpertRef`, agent-group packages, template→member composition (`src/experts/CONTEXT.md`)
- `./src/storage/` → `TeamStore` trait, records/patches, in-memory store, `open_store` (`src/storage/CONTEXT.md`)
- `./src/storage/sqlite/` → `SqliteTeamStore`: schema, row encoding, statements and its contract suites (`src/storage/sqlite/CONTEXT.md`)
- `./src/taskboard/` → dependency graph, claims, completion, reviews/votes, revisions, subtasks (`src/taskboard/CONTEXT.md`)
- `./src/messager/` → `Messager` seam, in-process bus, transport factories, distributed stubs (`src/messager/CONTEXT.md`)
- `./src/workspace/` → allocation seam and layout, `local.rs` provider, `worktree.rs`/`worktree_git.rs` git backend, `worktree_remote.rs` remote protocol, `git.rs`, locks (`src/workspace/CONTEXT.md`)
- `./src/member/` → `MemberRuntime`/`MemberObserver`, `TeamMember` status machine, spawn requests, `protocol.rs` frames, in-process/process/external-CLI/SSH spawners (`src/member/CONTEXT.md`)
- `./src/interaction/` → HITL payloads, routing grammar, `InteractGate`, human inbox, tool-approval bridge (`src/interaction/CONTEXT.md`)
- `./src/runtime/` → `TeamAgent` and the leader/teammate loop, `TeamBackend`, dispatch, build path, `spec_meta`, `attach_host`, manager/pool, model allocator/router (`src/runtime/CONTEXT.md`)
- `./src/harness/` → harness-protocol adapter, native factory, model resolvers, checkpoints (`src/harness/CONTEXT.md`)
- `./src/org/` → roles, delegation router, org task pool, org mailbox, swarmflow, fork/compact (`src/org/CONTEXT.md`)
- `./src/models/` → `ModelPool`, `ModelRouter` strategies, allocation helpers (`src/models/CONTEXT.md`)
- `./src/kv_cache/` → `TeamKvCacheAffinity`, cache identity, cleanup/eviction hooks (`src/kv_cache/CONTEXT.md`)
- `./src/observability/` → span/hook types, `TeamObservability`, the `otel` feature, the memory bridge and the `otlp` OTLP bridge (`src/observability/CONTEXT.md`)
- `./src/memory/` → `TeamMemoryWiring`, the `metadata["memory"]` config, the member tool surface and `TEAM_MEMORY.md` export/import (`src/memory/CONTEXT.md`)
- `./src/monitor/` → `TeamMonitor`, `StatusStream`, event stamping and rendering (`src/monitor/CONTEXT.md`)
- `./src/attach/` → `protocol.rs` frames/ops/errors, `client.rs`, `server.rs` (`src/attach/CONTEXT.md`)
- `./src/skill/` → `team-member` CLI and join descriptor, `TeamClient` with `StoreTeamClient`/`SocketTeamClient`, `live.rs` resolution, `socket_client.rs`, `serve.rs` child-process server (`src/skill/CONTEXT.md`)
- `./src/mcp/` → `openjiuwen-team-mcp`: tool catalogue, dispatch, stdio serve (`src/mcp/CONTEXT.md`)
- `./src/bin/` → the two binaries: `team-member` (exit `0/1/2/130`) and `openjiuwen-team-mcp` (`mcp` feature)
- `./src/constants.rs` → reserved names, defaults, mention parsing
- `./tests/` → crate integration suites (`tests/CONTEXT.md`)

### Sibling consumers (read-only from this node)

- `../openjiuwen/` → umbrella crate: `pub use openjiuwen_agent_teams as agent_teams`, forwards the `live-tests` feature; never edited here.
- `../openjiuwen-cli/` → the `openjiuwen` binary front-end (sibling crate, no dependency on this crate).

## Test Strategy

- Unit tests live beside their modules (`#[cfg(test)]`): `cargo test -p openjiuwen-agent-teams --lib` reports 1528 in the default build, 1526 with `--no-default-features`, 1547 with `--features sqlite`, 1545 with `--features otel`, 1544 with `--features otlp` and 1581 with `--all-features`, concentrated in `runtime`, `member`, `skill`, `org`, `schema`, `attach`, `workspace`, `taskboard`, `monitor`, `storage`, `models`, `interaction`, `kv_cache`, `observability`, `memory`, `harness`, `messager`, `mcp` and `experts`.
- Checkpoint and health suites: `src/runtime/wiring_tests/checkpoints.rs` drives the metadata-driven checkpoint wiring, `src/harness/checkpoint_store/` carries the store/policy unit tests, and `src/runtime/health.rs` covers the health projection.
- Doctests: 38 run plus one `ignore` block (`cargo test --doc`), and two `no_run` examples that compile but are not run; 13 `text` blocks are not doctests.
- Integration suites in `./tests/` drive the public API only: 219 tests over 15 binaries (215 in the default build, 219 with `--features sqlite`) — `spec_build` (12), `team_run` (9), `member_and_mail` (14), `task_flow` (32), `hitl` (33), `protocol_adapter` (13), `cli` (39), `mcp` (21), `expert_catalog` (12), `bins` (6), `full_team_sqlite` (4), `member_process_kind` (5), `attach_e2e` (6), `org_flow` (9) and `workspace_isolation` (4).
- `full_team_sqlite` compiles only with `--features sqlite`; `attach_e2e` is `#![cfg(unix)]`; `bins`/`member_process_kind` launch the real binaries through `CARGO_BIN_EXE_*`; `workspace_isolation` needs `git` on `PATH`.
- Fakes live in `./tests/support/` plus `openjiuwen-test-support` as a dev-dependency; nothing in the suite needs the network.
- Definition of done (all green): `cargo fmt -p openjiuwen-agent-teams`, `cargo test -p openjiuwen-agent-teams` in the default, `--no-default-features`, `--features sqlite`, `--features otel`, `--features otlp` and `--all-features` configurations, `cargo clippy -p openjiuwen-agent-teams --all-targets -- -D warnings`, and `cargo doc -p openjiuwen-agent-teams --no-deps` warning-free.

## Notes for Agents

- Python `TeamAgent` is one class for both roles. Keep that: `TeamAgent` carries a `TeamRole`; the leader owns dispatch, teammates poll the board.
- Python `spec.build()` mutates the spec (fills `language`); the Rust `build_with` takes `&self`, does not mutate, and `TeamBuildDeps::language` is `Option<String>` resolved ahead of the spec.
- `HarnessState` comes from `openjiuwen_harness_protocol`; re-export it, never define a second enum.
- The runtime reads eight optional `TeamAgentSpec::metadata` keys (`spawn`, `fork`, `workspace`, `organization`, `kv_cache`, `attach`, `checkpoint`, `memory`); their shapes, degradation rules and wiring points are tabulated in `src/runtime/CONTEXT.md` §"Metadata-driven wiring" and §"Memory wiring".
- `TeamBuildDeps` carries only injection hooks (`with_store`/`has_injected_store`, `with_workspace`, `with_language`, `with_checkpoint_store`/`has_injected_checkpoints`, `with_memory`/`has_injected_memory`, `with_memory_observer`/`has_memory_observer`, `with_observability_hook`, `with_kv_cache`, `with_routing`); `monitor_stream` is the monitor knob, and organisation/attach come from metadata.
- Workspace lifecycle: `build_with` allocates the leader's workspace and `spawn_member` one per teammate, both retained on the handle; `shutdown_member` releases that member's workspace and `stop()` releases everything still retained, each workspace exactly once.
- `team-member serve` lives in `src/skill/serve.rs` over the frames in `src/member/protocol.rs`; `StdioMemberRuntime` launches it with the identity in `OPENJIUWEN_TEAM_NAME`/`OPENJIUWEN_MEMBER_NAME`/`OPENJIUWEN_SESSION_ID`/`OPENJIUWEN_MEMBER_WORKSPACE`, and `serve_with_context` drives the same loop over any pipe pair.
- Attach wiring: `metadata["attach"]` makes `start()` bind `AttachHost`, which serves `BackendAttachHandler` (backend roster/mail/board/snapshot plus `stop_team`) over `AttachServer`; `SocketTeamClient` is the dialing side, and `src/skill/live.rs` resolves the in-process registry first, then the descriptor path, then `OPENJIUWEN_TEAM_SOCKET`.
- Models reach a member only through `MemberHarnessFactory::create(&DeepAgentSpec, &MemberHarnessContext)`: the name comes from `DeepAgentSpec::model` (else the resolver default), credentials/provider from the factory's `ModelResolver` (`NativeMemberFactory::from_env` → `EnvModelResolver` reads `MODEL_PROVIDER`/`MODEL_NAME`/`API_KEY`/`API_BASE`), and `NativeMemberFactory` also writes that identity onto the `DeepAgentConfig` so provider requests carry a non-empty `model_name`.
- `TeamAgentSpec::model_pool` allocation is recorded on `MemberRecord::model_ref` and passed to the harness factory through the routed `DeepAgentSpec`, so a pool entry selects the member's endpoint.
- Cross-node deviations from Python, each owned by its child node: `SpecBuild` is `#[async_trait]` and `build_with` takes `&self` without mutating the spec; the worktree name suffix is a FNV-1a digest instead of `sha256(..)[:10]` and member state lives at `<base>/<team>/sessions/<sid>/…`; `Team::add_agent` writes its row on the current runtime and a team runs one at a time; `HarnessMemberRuntime::send` drops the transport-level `SendReceipt`; `TaskStatus` transitions are narrower than Python's (no `in_progress→pending`, `planning→pending`).
- Module-level `//!` docs resolve intra-doc links in the *parent* module's scope, so a module doc needs a fully-qualified target (`[`X`](crate::schema::X)`) for anything the parent does not import; item-level `///` docs resolve in the item's module. Keep `cargo doc -p openjiuwen-agent-teams --no-deps` warning-free (no `#[allow(rustdoc::…)]`, link private items as plain code spans).
- Team memory is opt-in per spec: `metadata["memory"]` (or `TeamBuildDeps::with_memory`) allocates one `TeamMemoryWiring` per team, exposed as `TeamAgent::memory()`; no key means no wiring, no hook and no write.
- `TeamBuildDeps::with_memory_observer` only affects the metadata path: with no `metadata["memory"]` block and no injected wiring, no wiring is allocated and the injected observer sees nothing (`src/runtime/memory_bind.rs:74-106`).
- `TeamMemoryObserver`, `MemberMemoryToolkit`, `MemberMemoryContext`, `TeamMemoryHookConfig`, `TeamTurn`, `TeamMemoryRecord` and `TeamMemoryFlush` are engine types from `openjiuwen_memory::team`; this crate re-exports none of them, and `MemorySpanObserver::from_root` does not exist (a wiring's `memory.*` spans stay a sibling trace of the runtime's `team.{name}` trace).
- The runtime never flushes memory per member turn: messages buffer through `TeamBackend::record_turn` → `on_turn_message`, and the only flush/export points are `stop` (`finish`) and `snapshot` (`export_document`); `TeamMemoryWiring::on_turn` has no runtime caller.
- Memory hook failures are advisory: `start`, `spawn_member`, `shutdown_member`, `stop` and `send_message` log a `warn!` and continue, while `TeamMemoryWiring::open` failing during the build does fail the build.
- Only in-process members mount the team-memory tools: the toolkit rides `MemberHarnessContext::with_memory_toolkit`, which a `process`/`external_cli`/`ssh` child cannot receive, so those members have no `team_memory_*` tool surface.
- Team memory is written by the lifecycle hooks and buffered turns, and read back through the `TEAM_MEMORY.md` document: `stop` flushes then exports, `snapshot` exports best-effort, and both gate on `export_on_stop`.
- The team-memory observer precedence is `enabled` → `trace` → caller-supplied `observer` → the built-in `MemorySpanObserver`; `trace: false` attaches nothing even when an observer was injected.
- Child-process members checkpoint through `OPENJIUWEN_MEMBER_CHECKPOINTS`/`OPENJIUWEN_MEMBER_CHECKPOINT_DIR`, exported by `member::spawn` from the resolved `CheckpointConfig` and read by `skill::ServeContext`.
- Offline runs need either an injected factory (the test pattern) or `NativeMemberFactory::new(resolver)` with a resolver returning a scripted `openjiuwen_foundation::llm::Model`; `TeamAgentSpec::build()` over `TeamBuildDeps::default()` fails with `ErrorKind::Config` when no provider/model is configured.
- The `live-tests` feature is declared in `Cargo.toml` and read by no test or module in this crate: nothing here needs the network.
- `NativeCheckpoint` is produced at every member turn/state boundary the policy enables and on `stop`, persisted through the resolved `MemberCheckpointStore`, and restored on the next `start` for the same identity.
- Checkpoint session convention: every member runs under the derived session id `"<team session>:<member name>"` while `TeamAgent::session_id()` stays the team session.
- A member's harness `host_session_id` is that derived session, because runtimes start with `start(None)`.
- Checkpoint identity is `(derived session, card id)`, where the card id is the effective spec's card id else the member name.
- Checkpoint store resolution order: `TeamBuildDeps::with_checkpoint_store` injection, then metadata `checkpoint.store` (`FileMemberCheckpointStore`, its directory defaulting to `<workspace root>/.openjiuwen/checkpoints`), then one `InMemoryMemberCheckpointStore` shared per team.
- A disabled checkpoint policy attaches no store at all.
- `crates/openjiuwen-e2e/tests/team_backend.rs` still asserts completion is stream-only, so its assertion needs updating by the e2e owner; this crate cannot edit that suite.
- `native_v2` in `openjiuwen-harness-providers` builds a team-backed `HarnessProtocol` harness over `TeamRuntimeManager`; this crate is a protocol consumer only.
- The `openjiuwen` umbrella's `sqlite` feature does not forward to this crate (`crates/openjiuwen/Cargo.toml`), so a consumer needs `openjiuwen-agent-teams/sqlite` enabled explicitly to reach `SqliteTeamStore`; `open_store` reports the missing feature otherwise.
