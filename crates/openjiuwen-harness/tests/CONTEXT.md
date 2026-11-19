# harness integration tests

## Intent

Offline, deterministic integration suites for `openjiuwen-harness`: they drive the public API end to end with `openjiuwen_test_support::ScriptedModel`, temp workspaces, and fake permission/telemetry hosts.
Unit tests live beside their modules; these cover cross-module behavior.

## API Surface

Test files (one concern each, sixteen binaries, 148 tests with `--all-features`):
- `deep_agent_loop.rs` (6) — no-tool answer, real `FunctionTool` round, built-in `bash` permission denial, max-iteration truncation, follow-up queue drain, idempotent abort.
- `workspace_safety.rs` (9) — cn/en schema seeding and markers, traversal/absolute/drive/UNC rejection, top-level-only `get_node_path`, idempotent rebuild, `is_safe_relative_path` truth table, unsafe node paths.
- `permissions.rs` (14) — allow/ask/deny precedence and `strictest` fold, file-guard fold, session-scoped approvals via the rail, rail `skip_tool` denial, `PermissionConfirmResponse::parse` shapes.
- `state_persistence.rs` (7) — `DeepAgentState` round-trip, exact `deepagent` key set, plan-mode/follow-up survival, default state, repeated-save null-key preservation, `todo.json` location and item keys.
- `subagents_skills.rs` (19) — built-in presets, fake factory spawn/wait/close/capacity, quota refusal/free and per-parent/depth limits, activity lifecycle and LRU eviction with transcript digests, subagent tool payloads, skill discovery/budget/disabled state, `SkillRootMount`.
- `memory_tools.rs` (9) — the eight memory tools over the file-backed `WorkspaceMemoryBackend`, driven only through `Tool::invoke` on a temp workspace: write/read round trip with line windows, Python scope/path mapping, edit success and failure payloads, keyword-search result objects and typed traversal failures, plus `DeepAgentBuilder` mounting the backend by default.
- `subagent_spawner.rs` (5) — the production `RuntimeSubAgentSpawner`: a built `DeepAgent` exposes a spawner over its own runtime (a real `DeepAgentSubAgentFactory` child answering from a `ScriptedModel` runs to `Completed` through `spawn`/`wait`/`list`/`close`), a caller `subagent` rail wins (accessor `None`), no mounted sub-agents means no spawner, a saturated single-permit spawner rejects the extra spawn with the typed limit error and reuses the slot once final, and unknown ids are `NotFound`.
- `protocol.rs` (8) — `NativeHarness` lifecycle, capability gating, single-consumer cursors, checkpoint export, provider `create` from JSON, capability card.
- `cli.rs` (22) — clap grammar/flags/formats, headless `run_with_model` in all three output formats, scripted REPL transcript, permission answer mapping.
- `config_flags.rs` (16) — the `DeepAgentConfig` knobs end to end over a fake failing `Tool` and `ScriptedModel`: `configure` rejects an invalid config and keeps the previous one, the retry budget retries a retryable failure and stops at the bounded attempt count, the switch-off and the non-idempotent cases get exactly one attempt, task-planning tools appear only with `enable_task_planning`, the invocation journal records a turn (and writes nothing when unset, warning only when the directory is unusable), the project-root boundary denies an outside absolute path while an unset root keeps the workspace boundary, the builder threads the root into the mounted tools and the permission engine denies outside it, and the default/async/runtime sub-agent modes mount `task_tool` alone, the four-tool lifecycle subset and the seven runtime tools respectively.
- `public_surface.rs` (14) — every newly landed item resolves flat from the crate root and does real work: `MoaConfig`/`MoaRail` (asserting the published consultation protocol text in both languages), `ExecutionLog`/`WindowAssembler`, `HeartbeatRail`, `PersonalContextStore`, `WorktreeLock`/`SparseConfig`, `ActivityThrottle`, `drain_rail_events`, `DiagnosticRegistry`/`Diagnostic`, `WorkspaceWatcher` (a real delivered write plus a silent shutdown with `notify`, its typed `Unsupported` refusal without), `GoalRail`, the resume surface (`ExecutionCatalog`/`ExecutionSummary`, `PersistentAgentLayout` with the `LOGS_DIR`/`SESSIONS_DIR`/`LINKS_DIR`/`PERSISTENT_AGENT_NODE` constants, `FileSessionStore::list` → `SessionSummary`, `percent_decode_identifier`, `seed_state`, plus signature pins of `run_once_resumed`, `run_with_model_resumed`, `Repl::seed_history` and `DeepAgent::seed_history`), and the completion/MCP-notification surface (`CompletionFn`, `CompletionSummarizer` over a seam, `NotificationOutcome` with its accessors, the four method constants and both live-client refusals).
- `moa_rail.rs` (4) — two scripted proposers through `consult`/`rank_with_config`, the rail's prompt section and `MoaSnapshot`/`MoaOutcome` extra keys, the all-proposers-fail `Error::Llm`, and the refusal of a reference-free configuration.
- `persistent_agent_resume.rs` (4) — a rotated journal reopened from disk (`replay`, digest, monotone sequences), `resume` → `ResumePlan` with a rebuilt window and continuation prompt, `ShareFile` export/import round trip plus tamper detection, the typed `Unsupported` of the remote/distributed handoffs, and the end-to-end host restart: one CLI turn (`run_with_model_resumed` on a `ScriptedModel`) journalled through `ExecutionLog` with a `PersistentSessionRecord` and an `ExecutionLink`, every handle dropped, the execution and session rediscovered through `ExecutionCatalog::list`/`FileSessionStore::list`, the plan round-tripped through JSON, and the continued turn sending the restored transcript ahead of the new query while the CLI record keeps both turns.
- `persistent_deep_agent.rs` (2) — `persistent_agent` recording end to end: two real `DeepAgent` turns journalled as `execution_started`/`model_turn`/`execution_final` entries plus a `PersistentSessionRecord`, everything re-derived from disk alone, `resume` rebuilding the window for a second agent that continues the seeded transcript, and recording off by default.
- `goal_rail_events.rs` (4) — a settled goal auto-registers `GoalRail` through a mounted `get_current_goal` tool and reaches both the `stream` interaction frame and `drain_rail_events` exactly once; an active goal publishes the reminder and no event; no goal tool means no rail; the event's output frame round-trips.
- `watcher_targets.rs` (6) — `#![cfg(feature = "notify")]`: independent per-target steps (a new skill file, an `IDENTITY.md` rewrite, an MCP-config rewrite) each await their own event, one watcher keeps sequences monotonic across targets, a write burst coalesces into one event, and a not-yet-created file target still fires while shutdown is idempotent and publishes nothing afterwards.

## Constraints

- No network, no sleeps longer than a few milliseconds, no reliance on wall-clock ordering; use `ScriptedModel` and `tokio::test`.
- Every test creates its own `tempfile::TempDir` workspace; never writes outside it.
- Assert persisted JSON keys exactly where parity matters: the `deepagent` session payload (`iteration`, `pending_follow_ups`, `plan_mode`, `stop_condition_state`, `task_plan`) and `todo.json` (`activeForm`, item keys); todos never live inside the `deepagent` payload.
- No `unwrap` in async tasks that could mask panics; prefer `expect("message")` in tests.
- Every test file opens with `#![allow(clippy::unwrap_used, clippy::expect_used)]` because the workspace runs `clippy::all` + `pedantic` with `-D warnings`.
- Import from `openjiuwen_harness::{agent,schema,…}`: module paths are canonical even though the crate root also re-exports the primary items flat.
- Feature-gated areas (browser/cua/mobile) assert `Error::Unsupported` only.
- `watcher_targets.rs` is gated by `#![cfg(feature = "notify")]`, so a `--no-default-features` run compiles it to an empty suite; every await carries its own 10 s per-target budget (never one deadline shared by several targets) and the only asserted count is the single event one debounce window owns.
- `watcher_targets.rs` creates every watched parent directory before `start()` (including `skills/demo/`), because `notify` adds a watch for a newly created subdirectory only after its own event-loop thread processes the `Create(IsDir)` notification, so a file written immediately into a directory created after `start()` can be missed outright and no timeout can recover it; the module docs record that mechanism.
- `public_surface.rs` keeps the `notify`-off branch (`Error::Unsupported` from `WorkspaceWatcher::start`) and the `notify`-on branch as two `#[cfg]` tests, so it passes in both configurations.

## Routing Table

- `./deep_agent_loop.rs` → loop, tools, interrupts
- `./workspace_safety.rs` → workspace and path safety
- `./permissions.rs` → permission engine and rail
- `./state_persistence.rs` → session state round-trip
- `./subagents_skills.rs` → subagents and skills
- `./memory_tools.rs` → the eight memory tools over the workspace backend
- `./subagent_spawner.rs` → the production subagent spawner over a built agent's runtime
- `./protocol.rs` → native harness protocol
- `./cli.rs` → CLI parsing and run
- `./config_flags.rs` → configuration knobs: `configure` validation, tool retries, task planning, the invocation journal, the project-root boundary and the sub-agent modes
- `./public_surface.rs` → flat crate-root re-exports of the new modules
- `./moa_rail.rs` → MOA consultation, ranking and rail events
- `./persistent_agent_resume.rs` → journal replay, resume plan and share bundles
- `./persistent_deep_agent.rs` → per-turn persistent-agent recording and resume from disk
- `./goal_rail_events.rs` → goal rail auto-registration, stream and drain
- `./watcher_targets.rs` → `notify` watcher per-target delivery, cross-target sequences, coalescing and shutdown (feature-gated)

## Known Issues

- `DeepAgent::load_state`/`save_state` need a session bound by an invocation; there is no setter, so cross-instance assertions must use the public `openjiuwen_harness::agent::{load_state, save_state}` free functions.
- Permission denial keeps `result_type == Answer`; the observable surface is `[PERMISSION_DENIED]` in `AgentOutput::observations` and in the next model call's tool message.

## Test Strategy

- Run with `cargo test -p openjiuwen-harness` (default features) and `--no-default-features`; both must pass.
- Tests must not depend on `notify`, `browser`, `cua`, or `mobile` features.

## Test Harness Recipes

The `// :123` line numbers in the snippets below are indicative only; the files keep growing, so verify a signature against the source rather than trusting a number.
`openjiuwen_harness` re-exports the primary items flat from the crate root (`src/lib.rs`, the `pub use` block after `pub mod`), but these recipes import module paths (`openjiuwen_harness::{agent,schema,permission,workspace,subagents,skills,protocol,cli}::…`) because those are canonical.
`openjiuwen_harness::clap` re-exports the exact clap version `Cli` derives from, so `tests/cli.rs` names clap only through it (`use openjiuwen_harness::clap::{CommandFactory, Parser};`) and the crate needs no clap dev-dependency.
Crate `dev-dependencies`: `openjiuwen-test-support`, `anyhow`, `tempfile`; normal deps usable from tests include `openjiuwen-agent`, `openjiuwen-session`, `openjiuwen-foundation`, `openjiuwen-sysop`, `openjiuwen-common`, `openjiuwen-harness-protocol`, `tokio`, `futures`, `async-trait`, `parking_lot`, `serde_json`.

### 1. ScriptedModel (`crates/openjiuwen-test-support/src/scripted_model.rs`)

```rust
pub struct ScriptedModel;                                        // :230 (Clone shares script/recording)
pub fn builder() -> ScriptedModelBuilder;                        // :247
pub fn new(turns: impl IntoIterator<Item = ScriptedTurn>) -> Self; // :253
pub fn text(content: impl Into<String>) -> Self;                 // :261 (one text turn)
pub fn push(&self, turn: ScriptedTurn);                          // :266
pub fn push_many(&self, turns: impl IntoIterator<Item = ScriptedTurn>); // :271
pub fn set_fallback(&self, turn: ScriptedTurn);                  // :276
pub fn remaining(&self) -> usize;                                // :282
pub fn call_count(&self) -> usize;                               // :288
pub fn calls(&self) -> Vec<ModelCall>;                           // :294 (oldest first)
pub fn last_call(&self) -> Option<ModelCall>;                    // :300
pub async fn wait_for_calls(&self, count: usize) -> Result<()>;  // :317
pub async fn wait_for_calls_timeout(&self, count: usize, timeout: Duration) -> Result<()>; // :328
```

```rust
pub enum ScriptedTurn { Response(ChatResponse), Chunks(Vec<StreamChunk>), Fail(ScriptedError) } // :134
ScriptedTurn::text(impl Into<String>)  // :151
ScriptedTurn::chunks(Vec<StreamChunk>) // :157
ScriptedTurn::fail(impl Into<String>)  // :163 (ErrorKind::Llm)
ScriptedTurn::fail_with(ScriptedError) // :169

ScriptedModelBuilder: provider/ model_name/ respond(ChatResponse) / respond_text/ completion(MockCompletion) /
  chunks(Vec<StreamChunk>) / stream_text/ fail(ScriptedError) / fallback(ScriptedTurn) / build  // :473-545, defaults "scripted"/"scripted-model"

pub enum CallMode { Invoke, Stream }                             // :176
pub struct ModelCall { pub mode: CallMode, pub messages: Vec<BaseMessage>, pub request: ModelRequestConfig } // :185
ModelCall::last_user_text(&self) -> Option<String>               // :197
```

Tool calls and usage come from `MockCompletion` (`crates/openjiuwen-test-support/src/responses.rs`), passed via `ScriptedModelBuilder::completion`:

```rust
MockCompletion::text(impl Into<String>)                                  // :130
MockCompletion::tool_call(name: impl Into<String>, arguments: Value)     // :163 (id "call_<name>", finish ToolCalls)
MockCompletion::tool_calls(Vec<MockToolCall>)                            // :169
MockCompletion::with_usage(self, usage: Usage)                           // :193
MockToolCall::new(name: impl Into<String>, arguments: Value)             // :60 (compact JSON args)
text_chunks(content) / tool_call_chunks(calls)                           // :420 / :427
```

`Usage` is `openjiuwen_foundation::Usage` (`crates/openjiuwen-foundation/src/prompt/usage.rs:29`): `Usage::new(input_tokens, output_tokens)` sets `total_tokens`.
`Model` is `openjiuwen_foundation::llm::Model`; `invoke` pops one turn, `stream` pops one turn, and calls are recorded at call time.
Empty response id/model are filled from the builder (`scripted-<n>` / configured model name); the fallback turn is reused without consuming the queue.

```rust
use std::sync::Arc;
use openjiuwen_foundation::llm::Model;
use openjiuwen_foundation::Usage;
use openjiuwen_test_support::{MockCompletion, ScriptedModel};
use serde_json::json;

let model = ScriptedModel::builder()
    .model_name("fake-model")
    .completion(MockCompletion::tool_call("bash", json!({"command": "ls"}))
        .with_usage(Usage::new(11, 7)))
    .respond_text("done")
    .build();
let scripted = model.clone();                       // keep a handle to inspect calls
let dyn_model: Arc<dyn Model> = Arc::new(model);
// … drive the agent, then:
assert_eq!(scripted.call_count(), 1);
assert_eq!(scripted.calls()[0].mode, openjiuwen_test_support::CallMode::Invoke);
assert_eq!(scripted.calls()[0].last_user_text().as_deref(), Some("hi"));
```

### 2. DeepAgent construction (`src/agent/builder.rs`, `src/agent/factory.rs`)

```rust
DeepAgent::new(card: openjiuwen_agent::AgentCard) -> Self              // agent/mod.rs:167 (unwired; call configure + ensure_initialized)
DeepAgentBuilder::new(card: AgentCard) -> Self                        // builder.rs:82
  .config(DeepAgentConfig)                                            // :100
  .model(Arc<dyn Model>)                                              // :107
  .sys_operation(Arc<openjiuwen_sysop::SysOperation>)                 // :114
  .workspace(openjiuwen_harness::workspace::Workspace)                // :121
  .tool(Arc<dyn Tool>) / .tools(impl IntoIterator<Item = Arc<dyn Tool>>) // :128 / :135
  .rail(Arc<dyn HarnessRail>)                                         // :142
  .permission_host(Arc<dyn PermissionHost>)                           // :149
  .user_asker(Arc<dyn UserAsker>)                                     // :156
  .memory(Arc<dyn MemoryRecall>, Arc<dyn MemoryWriteBack>)            // :163
  .subagent_factory(Arc<dyn SubAgentFactory>)                         // :174
  .build() -> Result<Arc<DeepAgent>>                                  // :186
```

There is **no** builder setter for a session, prompt text, tempdir path, or a `PermissionEngine`; the permission engine is derived from `DeepAgentConfig::permissions`, and the filesystem root is set through `.workspace(...)` / `DeepAgentConfig::with_workspace`.
`create_deep_agent(model, card: Option<AgentCard>, config: Option<DeepAgentConfig>, tools: Vec<Arc<dyn Tool>>, rails: Vec<Arc<dyn HarnessRail>>, workspace: Option<Workspace>, skills: Vec<String>, subagents: Vec<SubAgentConfig>, enable_task_loop: bool, max_iterations: Option<u32>, language: Option<String>, prompt_mode: Option<PromptMode>, restrict_to_work_dir: bool, permissions: Option<PermissionsSection>, permission_host: Option<Arc<dyn PermissionHost>>) -> Result<Arc<DeepAgent>>` (`factory.rs:249`).

Minimal recipe (mirrors the integration test `tests/deep_agent_loop.rs:112-131`; `src/agent/tests.rs:224-233` is the same shape with `session = None`):

```rust
use std::sync::Arc;
use openjiuwen_agent::{Agent, AgentCard};
use openjiuwen_foundation::llm::Model;
use openjiuwen_harness::agent::DeepAgentBuilder;
use openjiuwen_harness::schema::DeepAgentConfig;
use openjiuwen_harness::workspace::Workspace;
use openjiuwen_session::AgentSession;
use openjiuwen_test_support::{ScriptedModel, TempWorkspace};
use serde_json::json;

let temp = TempWorkspace::new().expect("temp workspace");              // <root>/{home,workspace,logs}
let workspace = Workspace::for_language(temp.workspace(), "en");       // or Workspace::new(temp.workspace())
let model = ScriptedModel::text("hello");

let agent = DeepAgentBuilder::new(AgentCard::new("test-agent").with_description("test"))
    .config(DeepAgentConfig::default().with_language("en".to_string()).with_workspace(workspace))
    .model(Arc::new(model.clone()) as Arc<dyn Model>)
    .build()
    .expect("agent builds");

let session: std::sync::Arc<dyn openjiuwen_session::Session> =
    Arc::new(AgentSession::new("s1"));
let out = agent.invoke(json!({"query": "hi"}), Some(session)).await.expect("invoke");
assert_eq!(out.output, "hello");
assert_eq!(model.call_count(), 1);
```

### 3. `Agent` trait, request, response, stream (`crates/openjiuwen-agent/src/agent.rs`, `output.rs`)

```rust
#[async_trait::async_trait]
pub trait Agent: Send + Sync {                                    // agent.rs:15
    fn card(&self) -> &AgentCard;                                 // :17
    async fn invoke(&self, inputs: serde_json::Value, session: Option<Arc<dyn Session>>)
        -> Result<openjiuwen_agent::AgentOutput>;                 // :29
    fn stream(&self, inputs: serde_json::Value, session: Option<Arc<dyn Session>>)
        -> futures::stream::BoxStream<'static, Result<openjiuwen_agent::AgentChunk>>; // :32
}
```

There is **no `AgentRequest` type**: `invoke`/`stream` take a raw `Value` plus an optional session handle; `use openjiuwen_agent::Agent;` is required to call them.
The parsed input view is `AgentInput` (`input.rs:30`): `query`, `conversation_id`, `parent_session_id`, `user_id`, `run_kind`, `max_iterations`, `extra`; it has no `messages` field.
`AgentOutput` (`output.rs:84`): `output: String` (final answer text), `result_type: ResultType`, `finish_reason: Option<String>`, `reasoning_present: bool`, `termination_reason: Option<String>`, `truncated_by_max_iterations: bool`, `iterations_used: Option<usize>`, `observations: Vec<Observation>`, `finalization_error: Option<String>`, `state: Vec<OutputFrame>`, `interrupt_ids: Vec<String>`, `component_ids: Vec<String>`, `workflow_output`, `data: Option<Value>`, `extra: BTreeMap<String, Value>`.
`ResultType` (`output.rs:19`): `Answer` (default) | `Error` | `Interrupt`; wire strings `answer`/`error`/`interrupt`.
`termination_reason` is **not an enum**; today the emitted strings are `"max_iterations"`, `"token_budget"`, `"workflow_error"` (and documented `"cancelled"`).
`AgentChunk` (`output.rs:233`): `kind: String` (serde `"type"`), `index: i64`, `payload: Value`; constants `LLM_OUTPUT`, `LLM_REASONING`, `LLM_USAGE`, `ANSWER`, `INTERACTION`, `WORKFLOW_FINAL`, `CONTEXT_USAGE` live in `openjiuwen_agent::output`, not the crate root.
`DeepAgent::stream` emits exactly one terminal chunk `kind == "answer"` whose payload is the full `AgentOutput` JSON; an interrupt run emits `__interaction__` frames and **no** terminal answer frame, and a concurrent/ pre-flight failure is exactly one `Err` item.
Only one invocation may run per `DeepAgent`; a second `invoke` fails with `Error::Config("DeepAgent invocation already running…")`.

### 4. Permission (`src/permission/`)

```rust
PermissionEngineConfig::new(PermissionsSection) -> Self                    // engine.rs:72
  .with_workspace_root(impl Into<PathBuf>)                                 // :82
  .with_trusted_dirs(Vec<PathBuf>)                                         // :89
  .with_enabled(bool)                                                      // :96
PermissionEngine::new(PermissionEngineConfig) -> Self                      // :244 (no Clone/Default)
  .update_permissions(PermissionsSection)                                  // :296 (replaces all rules)
  .check_permission(&self, tool_name: &str, args: &serde_json::Value) -> PermissionResult // :322
```

`PermissionLevel` (`model.rs:36`): `Ask` (default) | `Allow` | `Deny`; wire `ask`/`allow`/`deny`; no `Ord` — precedence only via `PermissionEngine`'s `strictest` (`deny < ask < allow`).
`PermissionResult` is a struct (`model.rs:82`): `permission: PermissionLevel`, `matched_rule: Option<String>`, `reason: Option<String>`, `external_paths: Option<Vec<String>>`; helpers `is_allowed/is_denied/needs_approval`.
`PermissionConfirmResponse` (`model.rs:156`): `approved: bool`, `feedback: String`, `auto_confirm: bool`, `persist_allow: Option<bool>`; `parse(&Value) -> Option<Self>`.
No-opinion is `None`: `Option<PermissionLevel>` from `level_for_tool`/`evaluate_global_policy_directly`, or `Option<PermissionResult>` from the file/net guards.

`PermissionRule` (`model.rs:415`): `id: String`, `tools: Vec<String>` (exact names; `"shell"` is a category token that includes `bash`; `"*"` is literal), `pattern: String`, `match_type: RuleMatchType::{Command, Path}`, `action: Option<PermissionLevel>`, `severity: Option<String>`, `description`, `layer`.
`Command` patterns: exact text after trim, unless a `*`/`?`/`[` makes it a shell wildcard, or a `re:` prefix selects an unanchored regex; the command text is `args["command"]` then `args["cmd"]`.
`PermissionsSection` (`model.rs:612`) fields: `enabled`, `package_builtin_rules`, `permission_mode`, `defaults: BTreeMap<String, PermissionLevel>`, `tools: BTreeMap<String, PermissionLevel>`, `categories`, `rules`, `approval_overrides`, `file_guard`, `net_guard`, `shell_guard`.
Default section sets `package_builtin_rules = true` and `defaults["*"] = Ask`, so set `package_builtin_rules: false` in tests that want only their own rule.

```rust
use std::collections::BTreeMap;
use openjiuwen_harness::permission::{PermissionEngine, PermissionEngineConfig};
use openjiuwen_harness::permission::{
    PermissionLevel, PermissionRule, PermissionsSection, RuleMatchType,
};

let section = PermissionsSection {
    package_builtin_rules: false,
    defaults: BTreeMap::from([("*".to_owned(), PermissionLevel::Ask)]),
    rules: vec![PermissionRule {
        id: "no_rm".into(),
        tools: vec!["bash".into()],
        pattern: "rm -rf *".into(),                 // or "re:^rm\\s"
        match_type: RuleMatchType::Command,
        action: Some(PermissionLevel::Deny),
        ..PermissionRule::default()
    }],
    ..PermissionsSection::default()
};
let engine = PermissionEngine::new(PermissionEngineConfig::new(section));
let result = engine.check_permission("bash", &serde_json::json!({"command": "rm -rf /tmp/x"}));
assert!(result.is_denied());                        // matched_rule = Some("tiered_policy:rules:deny:rules[no_rm]")
```

`PermissionInterruptRail::new(Arc<PermissionEngine>)` (`interrupt.rs:98`) with `.with_host(Arc<dyn PermissionHost>)` (`:108`) and `.with_store(Arc<SessionPermissionStore>)` (`:115`); it implements `openjiuwen_agent::rails::AgentRail` and raises interrupts in `before_tool_call`.
Session-scoped approvals: `SessionPermissionStore::new()`, `is_auto_confirmed(&str)`, `store_auto_confirm(&str)`, `clear()` (`session.rs:191-243`).
The `PermissionHost` trait (`session.rs:124`) has defaulted methods, so `NoopPermissionHost` (or an empty impl) is enough for engine tests.

### 5. Workspace (`src/workspace/`)

```rust
Workspace::new(impl Into<PathBuf>) -> Self                                   // workspace.rs:47 (language "cn")
Workspace::for_language(impl Into<PathBuf>, impl Into<String>) -> Self       // :56 (only exact "en" selects en)
  .with_language(impl Into<String>) -> Self                                  // :70
  .validate() -> Result<()>                                                  // :105
  .get_directory(&self, name: &str) -> Option<String>                        // :152 (relative path; falls back to default schema)
  .get_node_path(&self, name: &str) -> Option<PathBuf>                       // :170 (top-level only, not canonicalized)
  .get_node_path_for(WorkspaceNode) -> Option<PathBuf>                       // :179
  async .resolve(&self, path: &str) -> Result<PathBuf>                       // resolve.rs:29 (canonical absolute; need not exist)
  async .resolve_existing(&self, path: &str) -> Result<PathBuf>              // :50 (adds NotFound)
```

There is no `Workspace::root()` and no `get_node()`; read `workspace.root_path`.
`DirectoryBuilder::new(sys_operation: openjiuwen_sysop::SysOperation, root_path: impl Into<PathBuf>)` (`directory_builder.rs:36`) and `async build(&self, &[DirectoryNode]) -> Result<()>` (`:55`); `is_safe_relative_path(&str) -> bool` (`:129`).
Seeding recipe:

```rust
use openjiuwen_harness::workspace::{DirectoryBuilder, Workspace};
use openjiuwen_sysop::{LocalWorkConfig, SysOperation};

let workspace = Workspace::for_language(temp.workspace(), "en");
let op = SysOperation::local(LocalWorkConfig::unrestricted(
    temp.workspace().to_string_lossy().to_string())).expect("local sysop");
DirectoryBuilder::new(op, temp.workspace()).build(&workspace.directories).await.expect("seed");
assert!(temp.workspace().join("AGENT.md").is_file());
```

Default `cn` schema (`schema.rs:56-86`): files `AGENT.md`, `SOUL.md`, `HEARTBEAT.md`, `IDENTITY.md`, `USER.md`, dirs `memory/` (with `MEMORY.md`, `daily_memory/`), `coding_memory/` (with `MEMORY.md`), `todo/`, `messages/`, `skills/`, `agents/`, `context/` (with `session_memory.md`).
The `en` schema (`schema.rs:89-135`) is identical except it has **no `coding_memory/`**.
Each directory is marked by an empty `.workspace` file (`WORKSPACE_MARKER`); existing files are never overwritten and a rebuild is idempotent.
Path rejection is `openjiuwen_common::error::Error` with `kind() == ErrorKind::Validation` and message `"unsafe workspace path: {path}"` (`resolve.rs:30`); escape after canonicalization is `ErrorKind::Security` `"path escapes workspace root: {path}"` (`:36`).
`DirectoryBuilder::build` rejects unsafe node paths with `Validation("Unsafe path detected: {relative}")` and a missing filesystem backend with `Unsupported("workspace directory builder requires a filesystem backend")`.

### 6. State (`src/schema/state.rs`, `src/agent/state_store.rs`)

`DeepAgentState` (`state.rs:126`, `#[serde(default)]`, `Default`, `PartialEq`): `iteration: usize`, `task_plan: Option<TaskPlan>`, `stop_condition_state: Option<Value>`, `pending_follow_ups: Vec<String>`, `plan_mode: PlanModeState`.
The persisted `"deepagent"` payload is **exactly these five keys**; there is no `todos` or `skills_state` field in it (todos live in `<workspace>/<session>/todo.json`; skills state in `<root>/skills_state.json`).
`PlanModeState` (`state.rs:39`): `mode: String` (default `"normal"`), `pre_plan_mode: Option<String>` (default `Some("normal")`), `plan_slug`, `prompt_context`; `to_session_dict`/`from_session_dict`.
`AgentMode` (`mode.rs:22`): `Normal` | `Plan`; serde `"normal"`/`"plan"`; `as_str()`, `parse(&str)`, no `Display`.

```rust
DeepAgent::save_state(&self) -> Result<()>                          // agent/mod.rs:413 (last invoked session)
DeepAgent::load_state(&self) -> Result<DeepAgentState>              // :428
DeepAgent::clear_state(&self) -> Result<()>                         // :443
DeepAgent::switch_mode(&self, mode: AgentMode) -> Result<()>        // :381
agent::state_store::{load_state(session: &dyn Session) -> Result<DeepAgentState>} // state_store.rs:35
agent::state_store::{save_state(session: &dyn Session, state: &DeepAgentState) -> Result<()>} // :68
const SESSION_STATE_KEY: &str = "deepagent"                         // schema/state.rs:21
```

The agent-method forms fail with `Error::Validation("no session bound to this DeepAgent")` before any invocation; use the free functions with `AgentSession::new("s1")` for pure round-trip tests.

### 7. Subagents (`src/subagents/`)

```rust
#[async_trait::async_trait]
pub trait SubAgentFactory: Send + Sync {                                           // runtime.rs:174
    async fn create(&self, config: &SubAgentConfig, session_id: &str)
        -> Result<Arc<dyn openjiuwen_agent::Agent>>;
}
SubAgentRegistry::new() -> Self                                                    // mod.rs:90
SubAgentRegistry::with_builtin_presets(language: &str) -> Self                    // :100 (code/explore/plan/research/verification)
  .register(SubAgentSpec) / .get(&str) -> Option<SubAgentConfig> / .get_spec / .list / .names / .general_purpose() // :117-180
SubAgentRuntime::new(registry: Arc<SubAgentRegistry>, factory: Arc<dyn SubAgentFactory>,
                     parent_session_id: impl Into<String>) -> Self                 // runtime.rs:291
  .with_limits(max_subagents: usize, max_concurrent_running: usize) -> Self       // :315 (defaults 10 / 5)
  .with_turn_timeout(Duration) -> Self                                            // :323
  async .spawn(SubAgentSpawnRequest) -> Result<SubAgentId>                        // :339
  async .wait(&[SubAgentId], Duration) -> Vec<SubAgentOutcome>                    // :400
  .get_status(&str) -> SubagentStatus / .get_outcome(&str) / .list() / .capacity() // :419-449
  async .send_input(&str, &str, interrupt: bool) -> Result<()>                    // :470
  async .resume(&str) -> Result<ResumeReport> / async .close(&str) -> Result<SubagentStatus> // :527 / :570
SubAgentSpawnRequest::new(subagent_type, task_description)                        // :101
```

`SubAgentConfig` (`schema/config/subagent.rs:59`): `agent_card`, `system_prompt`, `tools: Vec<String>`, `skills`, `max_iterations`, `enable_task_loop`, `language`, `prompt_mode`, `factory_name`, `factory_kwargs: BTreeMap<String, Value>`, `enable_plan_mode`, `parallel_tool_calls`, `restrict_to_work_dir`, `display_name`, `role`, `subagent_type`.
`src/subagents/test_support.rs` (`FakeFactory`/`FakeAgent`/`behavior`) is `#[cfg(test)] pub(crate)` and **not reachable from `tests/`**; define the same fake locally implementing both `SubAgentFactory` and `openjiuwen_agent::Agent`.
`TaskTool::new(Arc<SubAgentRuntime>)` and `subagent_tools(runtime) -> Vec<Arc<dyn Tool>>` are public; the tool payloads use `success/data/error` and never return `Err` for expected failures.
`subagent_wait` clamps `timeout_ms` to `>= 10_000`, so use `SubAgentRuntime::wait(..., Duration::from_millis(..))` directly for short-timeout assertions.

### 8. Skills (`src/skills/`)

```rust
SkillLibrary::new() -> Self                                                  // mod.rs:135
SkillLibrary::with_roots<R: Into<PathBuf>, I: IntoIterator<Item = R>>(roots: I) -> Self // :141
  .add_root(impl Into<PathBuf>) -> bool / .remove_root(&Path) -> bool        // :153 / :160
  .is_disabled(&str) / .set_disabled(name, bool) / .disabled_names()         // :175-191
  async .discover() -> Result<Vec<Skill>>                                    // :203 (never errors today)
  async .discover_with_language(Option<&str>) -> Result<Vec<Skill>>          // :216
  .select(&[Skill], SkillBudget) -> Vec<Skill>                               // :228
  async .load_state(root) / .load_all_state() / .save_state(root)            // :277-294
SkillBudget { max_skills: Option<usize>, max_total_chars: Option<usize> }    // :75; ::unlimited(), .with_max_skills(n), .with_max_total_chars(n)
Skill { pub name: String, pub description: String, pub directory: PathBuf }  // :38; Skill::new(name, description, directory), .path() -> SKILL.md
SkillRootMount::mount(&mut Vec<PathBuf>, impl Into<PathBuf>) -> Self         // mount.rs:124 (in-memory only)
  .root(&self) / .owned(&self) / .is_mounted(&[PathBuf]) / .ensure_mounted(&mut Vec<PathBuf>) / .unmount(&mut Vec<PathBuf>) // :135-170
SkillResourceValidator::new(name, directory) / async .validate(&dyn FsOperation) / async .validate_local() // mount.rs:238,275,313
```

There is no `SkillLibraryState` struct: `library_state.rs` exposes `SKILLS_STATE_FILENAME = "skills_state.json"`, `read_disabled_skills(&Path)`, `collect_disabled_skills(&[PathBuf])`, `save_disabled_skills(&Path, &BTreeSet<String>)`; the file shape is `{"skill_configs":{"<name>":{"enabled":false}}}`.
`Skill` has no `enabled`/`resources`/`metadata` field; disabling is library state keyed by name.
Required `SKILL.md` layout: `<root>/[<group>/]<skill-dir>/SKILL.md`, front matter starting at byte 0 with a literal `---` fence, YAML keys `name` (must equal the directory name for validation), `description` (non-empty for validation), optional `description_<lang>`; discovery falls back to `Skill located in <dir>` and ignores front-matter `name`.
Discovery skips dot-directories and `output`, `temp`, `assets`, `node_modules`.

### 9. Protocol (`src/protocol/` + `crates/openjiuwen-harness-protocol`)

```rust
NativeHarness::new(agent: Arc<DeepAgent>, options: NativeHarnessOptions) -> Self // mod.rs:189 (starts Terminated)
NativeHarnessOptions::new() -> Self                                              // :134 (capacity 1024, CoalesceOrBlock)
  .with_event_buffer_capacity(u64) / .with_overflow(EventOverflowPolicy)         // :143 / :150
HarnessProtocol::start(&self, HarnessContext) -> HarnessResult<()>               // protocol.rs:88 (impl mod.rs:430)
HarnessProtocol::send(&self, HarnessInput, DeliveryMode) -> HarnessResult<SendReceipt> // :143 (mod.rs:618)
HarnessProtocol::send_auto(&self, HarnessInput) -> HarnessResult<SendReceipt>    // :153 (default)
HarnessProtocol::stop(&self) -> HarnessResult<()>                                // :101 (mod.rs:542)
HarnessProtocol::abort(&self, AbortMode) -> HarnessResult<()>                    // :172 (mod.rs:694)
HarnessProtocol::pause(&self) / resume(Option<HarnessInput>)                     // :195 / :208
HarnessProtocol::events() -> Box<dyn HarnessEventCursor>                         // :112 (mod.rs:602)
HarnessProtocol::turn_events(Option<&str>) -> Box<dyn HarnessEventCursor>        // :127 (mod.rs:606)
HarnessProtocol::export_checkpoint(&self) -> HarnessResult<Option<HarnessCheckpoint>> // :224 (mod.rs:817)
HarnessModelControl::list_models(&self) / set_model(&self, ModelSelection)       // :251 / :268 (separate trait, impl mod.rs:836/868)
HarnessEventCursor::next_event(&mut self) -> HarnessResult<Option<HarnessEvent>> // stream.rs:66
```

`NativeHarnessProvider::new()` / `.with_options(NativeHarnessOptions)` (`provider.rs:92,101`); `HarnessProvider::create(&self, config: JsonObject) -> HarnessResult<Box<dyn HarnessProtocol>>` (`provider.rs:119`) is synchronous and takes a plain `BTreeMap<String, serde_json::Value>` (build with `serde_json::from_value(json!({...}))`), not a typed struct.
Known config keys: `model`, `provider`, `api_key`, `api_base`, `temperature`, `workspace`, `enable_task_loop`, `max_iterations`, `system_prompt`, `permissions`, `language`; unknown keys are `HarnessError::Validation`.
`HarnessCapability` (`card.rs:41`) variants include `Steer`, `GracefulAbort`, `ForceAbort`, `PauseResume`, `PersistentSession`, `Checkpoint`, `NativeTools`, `McpTools`, `ModelSelection`, `ModelDiscovery`, plus `Hooks`, `EndpointModels`, `AuthFallback`; `native_card` (`mod.rs:85`) declares the first ten.
`HarnessError::UnsupportedCapability(String)` (`errors.rs:27`) carries only a message; it is `#[non_exhaustive]` and is not `Clone`/`PartialEq`, so assert `err.kind() == HarnessErrorKind::UnsupportedCapability` and `err.message()`.
The only reachable end-to-end gating failures through the public API are a non-empty `HarnessContext::mcp_servers` in `start` (`"declare MCP servers in the agent manifest; the native harness does not mount context MCP servers"`) and endpoint/auth-fallback model config; the steer/abort/model-control gating branches are unreachable because `NativeHarness::new` hard-wires `native_card`.
Cursor semantics: `events()`/`turn_events()` return a destructive single-consumer cursor; a second concurrent cursor is rejected on the first `next_event` with `HarnessError::State("the harness observation stream already has an active consumer")`, and no active cycle yields `State("native harness has no active event cycle")`.
Checkpoint export type: `HarnessCheckpoint { provider, schema_version, agent_id, host_session_id, checkpoint_id, sequence, data: JsonObject, provider_session_id, revision }`; the native payload is `NativeCheckpointState` (`bridge.rs:47`), importable only through `HarnessContext::with_checkpoint` + `ResumePolicy`.

### 10. CLI (`src/cli/`)

```rust
#[derive(clap::Parser)] pub struct Cli {                 // mod.rs:64
    model: Option<String>, provider: Option<String>, api_key: Option<String>, api_base: Option<String>,
    workspace: Option<PathBuf>, verbose: bool, command: Option<Command>,  // all global
}
pub enum Command { Run(RunArgs), Chat(ChatArgs) }        // mod.rs:98
pub struct RunArgs { pub prompt: Option<String>, pub output_format: OutputFormat } // :113
pub enum OutputFormat { Text, Json, StreamJson }         // :125 (wire text/json/stream-json; clap ValueEnum)
pub async fn run_cli(cli: Cli) -> Result<()>             // :205
pub async fn run(cli: Cli) -> Result<()>                 // :233
```

`Repl::new(agent: Arc<DeepAgent>, options: CliOptions)` (`repl.rs:161`), `Repl::with_io(agent, options, io: CliIo)` (`:177`, test path), `Repl::with_io_and_signals(...)` (`:171`), `async run(&mut self) -> Result<()>` (`:211`).
`CliIo::new(reader: Box<dyn LineReader>, writer: Box<dyn Writer>)` (`io.rs:70`); `LineReader`/`Writer` are `#[async_trait]` with `read_line`/`write`/`flush` (`:26/:37`), so tests inject `ScriptedReader`/`SharedWriter` doubles (pattern in `src/cli/tests/mod.rs:39-95`).
`async run_with_model(&CliOptions, &RunArgs, &CliIo, Arc<dyn Model>) -> Result<RunOutcome>` (`run.rs:210`) and `build_agent(model: Arc<dyn Model>, &CliOptions, &CliIo) -> Result<Arc<DeepAgent>>` (`:133`) are the headless seams; `run_cli` reads real env/home and builds a network model, so do not use it in tests.
`CliOptions` has no `Default`; construct `CliOptions::resolve_with(&Cli, &dyn EnvSource, Option<&Settings>, cwd)` (`config.rs:293`) or fill its public fields.
Permission prompt (`permissions.rs:56-66`): answers are trimmed/lowercased — approve `""|y|yes|ok|approve|allow|allow_once|true|1`, always `a|always|allow_always`, reject `n|no|deny|reject`, anything else is `Feedback`.
`PermissionAnswer::to_response()` (`:75`) maps always to `{approved:true, auto_confirm:true, persist_allow:None}`; `CliPermissionHost::request_confirmation` (`:221`) returns `ConfirmationOutcome::Failed` without writing when `!io.is_interactive()`.

### 11. Host resume (CLI turn + persistent agent)

```rust
ResumedTurn::new(session_id).with_history(Vec<BaseMessage>).with_state(Value)       // src/cli/resume.rs:65/75/85
async run_with_model_resumed(&CliOptions, &RunArgs, &CliIo, Arc<dyn Model>, ResumedTurn) -> Result<RunOutcome> // :118
async run_once_resumed(&CliOptions, &RunArgs, &CliIo, ResumedTurn) -> Result<RunOutcome> // :172 (builds the model: credentials)
DeepAgent::seed_history(&Arc<dyn Session>, Vec<BaseMessage>) -> Result<()>            // src/agent/seed.rs:71
async seed_state(&dyn Session, Value) -> Result<()>                                  // replaces the `deepagent` lane
Repl::seed_history(&mut self, Vec<BaseMessage>) -> Result<()>                        // src/cli/repl.rs:523
ExecutionCatalog::new(root, LogCapacity).with_clock(Arc<dyn Clock>).list()/open(id)   // src/persistent_agent/catalog.rs
FileSessionStore::new(layout.sessions_root()).list() -> Vec<SessionSummary>           // src/persistent_agent/session.rs:636
PersistentAgentLayout::{new, for_workspace}.{base, logs_root, sessions_root, links_root} // src/persistent_agent/layout.rs
```

`ResumePlan` is `Serialize`/`Deserialize`, so a host persists it and continues from the parsed copy; `seed_messages()`/`seed_history()` return the window it rebuilt from the journal.
A resumed turn seeds *after* the agent build (`configure` drops pooled contexts), continues an existing CLI record instead of truncating it, and appends both new messages.
The `run_with_model_resumed` seam is the test path (no credentials); `run_once_resumed` builds a provider model and is only signature-pinned.
Test recipe (`persistent_agent_resume.rs`): fill `CliOptions` by hand (`workspace = <root>/workspace`, `home = OpenJiuwenHome::from_root(<root>)`, no key/base) and call `ensure_workspace()`, pass `CliIo::new` a never-consulted `LineReader` plus a discarding `Writer`, keep a `ScriptedModel` clone as the recording handle, and journal the turn with `ExecutionLog::open_with_clock` under `layout.logs_root().join(identity.directory_name())` (the `ExecutionCatalog`/`RootLogSource` path rule).
A `ManualClock` frozen at `NOW_MS` makes `ExecutionSummary::created_at` and the session timestamps deterministic.

### 12. Commands and known state

- `cargo fmt -p openjiuwen-harness`; `cargo test -p openjiuwen-harness`; `cargo test -p openjiuwen-harness --no-default-features`; `cargo clippy -p openjiuwen-harness --all-targets -- -D warnings`.
- The thirteen `*.rs` files above are the complete `tests/` tree; all thirteen compile and pass (`cargo test -p openjiuwen-harness`: 2945 unit + 119 integration + 20 doc tests with `--all-features`; the `notify` binary is empty without the feature).
- Non-public APIs tests must not reach for: `SubAgentRuntime`'s `test_support` module, `discover_skills`, `Workspace::root`/`get_node` (absent), CLI child modules and `Repl::build`, `DeepLoopEvent` fields (serde-only), `DeepAgent` private fields, protocol `inner`/`cursor`/`events`/`bridge`/`turn`.
- No panics were found on the public surfaces above; `PermissionLevel` has no `Ord`, `HarnessError` is `#[non_exhaustive]` and not comparable, and `ScriptedModel` is `Clone` but not `Debug`/`Serialize`.
