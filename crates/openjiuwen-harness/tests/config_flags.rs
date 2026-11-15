//! Integration suite of the `DeepAgentConfig` flags that used to be inert.
//!
//! Each test drives the *public* surface of a real [`DeepAgent`] (or the registry the builder
//! runs) so the flag's effect is observable behaviour, not a field:
//!
//! - [`DeepAgent::configure`] validates before mutating: an invalid configuration is rejected
//!   with the Python `__post_init__` message and the previously installed one survives;
//! - `enable_tool_resilience_rail` installs the kernel retry budget (three attempts with
//!   backoff for idempotent tools, one for non-idempotent ones) and switching it off restores
//!   the single-attempt behaviour;
//! - `enable_task_planning` gates the four `todo_*` tools and the TODO prompt section;
//! - `invocation_log_dir` journals one JSONL line per turn, is absent by default and tolerates
//!   an unusable directory;
//! - `project_root` narrows path resolution to the project root in both the tools and the
//!   permission engine;
//! - the sub-agent mode (`enable_async_subagent` / `enable_subagent_runtime`) selects the
//!   mounted sub-agent tool surface.
//!
//! Everything is offline and deterministic: temp directories, `ScriptedModel` scripts, fake
//! [`Tool`] implementations and fixed paths — no network, no wall-clock ordering.
//!
//! The sub-agent mode's per-mode matrix is covered at unit level in
//! `src/agent/subagent_factory.rs`; the test here proves the same matrix through
//! [`DeepAgentBuilder`] and [`DeepAgent::tool_cards`].
#![allow(clippy::unwrap_used, clippy::expect_used)]
use async_trait::async_trait;
use openjiuwen_agent::{Agent, AgentCard};
use openjiuwen_common::error::{Error, ErrorKind, Result};
use openjiuwen_foundation::tool::{Tool, ToolCard, ToolContext};
use openjiuwen_foundation::Role;
use openjiuwen_harness::agent::{DeepAgent, DeepAgentBuilder};
use openjiuwen_harness::permission::{
    PermissionEngine, PermissionEngineConfig, PermissionLevel, PermissionsSection,
};
use openjiuwen_harness::schema::{DeepAgentConfig, SubAgentConfig};
use openjiuwen_harness::tools::ToolRegistry;
use openjiuwen_harness::workspace::Workspace;
use openjiuwen_session::{AgentSession, Session};
use openjiuwen_test_support::{MockCompletion, ScriptedModel};
use serde_json::{json, Value};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tempfile::TempDir;
/// The Python `__post_init__` message for a ceiling combined with the task loop.
const CEILING_NEEDS_SINGLE_ROUND: &str =
    "max_iterations_hard_limit requires single-round mode (enable_task_loop=False)";
/// The Python `__post_init__` message for a non-positive iteration cap.
const ITERATIONS_NOT_POSITIVE: &str = "max_iterations must be a positive integer";
/// The shared prefix of the access-boundary rejection of the file tools.
const OUTSIDE_PROJECT_ROOT: &str = "outside the project root";
/// A throwaway workspace root; the caller keeps the handle alive for the test.
fn temp_workspace() -> (TempDir, Workspace) {
    let temp = TempDir::new().expect("temp workspace directory");
    let workspace = Workspace::for_language(temp.path(), "en");
    (temp, workspace)
}
/// The stock configuration: English prompts over `workspace`, task loop off.
fn base_config(workspace: &Workspace) -> DeepAgentConfig {
    DeepAgentConfig::default()
        .with_language("en".to_string())
        .with_workspace(workspace.clone())
}
/// Builds a `DeepAgent` over `config` with one scripted model and no extra tools.
fn build_agent(config: DeepAgentConfig, model: ScriptedModel) -> Arc<DeepAgent> {
    DeepAgentBuilder::new(AgentCard::new("flags-agent").with_description("Config flags"))
        .config(config)
        .model(Arc::new(model))
        .build()
        .expect("agent builds")
}
/// Runs one invocation on a fresh session and returns the answer.
async fn invoke_once(agent: &Arc<DeepAgent>, query: &str) -> String {
    let session: Arc<dyn Session> = Arc::new(AgentSession::new(format!("{query}-session")));
    agent
        .invoke(json!({"query": query}), Some(session))
        .await
        .expect("invocation succeeds")
        .output
}
/// The system message of the most recent model call, if it had one.
fn system_text(model: &ScriptedModel) -> String {
    model
        .last_call()
        .expect("the model was called")
        .messages
        .iter()
        .find(|message| message.role == Role::System)
        .map(|message| message.text().into_owned())
        .expect("every model call carries a system prompt")
}
/// A [`Tool`] that fails with a retryable transport error for its first `fail_until` calls.
///
/// Idempotency is the caller's choice: the kernel resilience policy retries an idempotent
/// tool and never retries a non-idempotent one, which is exactly what the suite asserts.
struct FlakyTool {
    /// The model-facing card, carrying the idempotency flag under test.
    card: ToolCard,
    /// Calls accepted so far, including the failed ones.
    calls: Arc<AtomicUsize>,
    /// How many leading calls fail before the tool answers successfully.
    fail_until: usize,
}
impl FlakyTool {
    /// Builds a tool called `name` over `calls` that fails `fail_until` times.
    fn new(name: &str, idempotent: bool, fail_until: usize) -> (Arc<Self>, Arc<AtomicUsize>) {
        let calls = Arc::new(AtomicUsize::new(0));
        let card = ToolCard::new(
            name,
            "Always-failing test tool",
            json!({"type": "object", "properties": {}}),
        )
        .with_idempotent(idempotent);
        (
            Arc::new(Self {
                card,
                calls: Arc::clone(&calls),
                fail_until,
            }),
            calls,
        )
    }
}
#[async_trait]
impl Tool for FlakyTool {
    fn card(&self) -> &ToolCard {
        &self.card
    }
    async fn invoke(&self, _inputs: Value, _ctx: &ToolContext) -> Result<Value> {
        let attempt = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
        if attempt <= self.fail_until {
            // `Transport` is retryable by the kernel rule (only `Cancelled` and `Validation`
            // are not), which is what a real connection reset reports.
            return Err(Error::transport("connection reset by peer"));
        }
        Ok(json!({"success": true, "data": {"content": "recovered"}}))
    }
}
/// Builds an agent whose model asks for `name` once and then answers.
fn tool_round_agent(
    config: DeepAgentConfig,
    tool: Arc<dyn Tool>,
    name: &str,
) -> (Arc<DeepAgent>, ScriptedModel) {
    let model = ScriptedModel::builder()
        .completion(MockCompletion::tool_call(name, json!({})))
        .respond_text("tool round done")
        .build();
    let agent = DeepAgentBuilder::new(AgentCard::new("resilience-agent"))
        .config(config)
        .model(Arc::new(model.clone()))
        .tool(tool)
        .build()
        .expect("agent builds");
    (agent, model)
}
/// The tool result text the second model call received.
fn tool_result_text(model: &ScriptedModel) -> String {
    let calls = model.calls();
    assert!(
        calls.len() >= 2,
        "the model must be called again after the tool round"
    );
    calls[1]
        .messages
        .iter()
        .find(|message| message.role == Role::Tool)
        .map(|message| message.text().into_owned())
        .expect("the second model call carries the tool result")
}
// ---------------------------------------------------------------------------
// (a) `configure` rejects an invalid configuration without mutating the agent.
// ---------------------------------------------------------------------------
#[tokio::test]
async fn configure_rejects_an_invalid_config_and_keeps_the_previous_one() {
    let (_temp, workspace) = temp_workspace();
    let agent = build_agent(base_config(&workspace), ScriptedModel::text("hi"));
    let installed = agent.config();
    assert!(!installed.enable_task_loop);
    assert_eq!(installed.max_iterations, None);
    let ceiling = installed
        .clone()
        .with_enable_task_loop(true)
        .with_max_iterations_hard_limit(Some(1));
    let error = agent.configure(ceiling).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Config);
    assert_eq!(error.message(), CEILING_NEEDS_SINGLE_ROUND);
    let zero = installed.clone().with_max_iterations(Some(0));
    let error = agent.configure(zero).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Config);
    assert_eq!(error.message(), ITERATIONS_NOT_POSITIVE);
    // Nothing was installed by either rejected call.
    let after = agent.config();
    assert!(!after.enable_task_loop);
    assert_eq!(after.max_iterations, None);
    assert_eq!(after.max_iterations_hard_limit, None);
    // A valid configuration is still accepted afterwards.
    agent
        .configure(installed.clone().with_enable_task_planning(true))
        .expect("a valid configuration is accepted");
    assert!(agent.config().enable_task_planning);
    assert_eq!(invoke_once(&agent, "after configure").await, "hi");
}
// ---------------------------------------------------------------------------
// (b) `enable_tool_resilience_rail` installs the kernel retry budget.
// ---------------------------------------------------------------------------
#[tokio::test]
async fn a_retryable_tool_failure_is_retried_under_the_default_config() {
    let (_temp, workspace) = temp_workspace();
    let (tool, calls) = FlakyTool::new("flaky", true, 1);
    let (agent, model) = tool_round_agent(base_config(&workspace), tool, "flaky");
    let answer = invoke_once(&agent, "retry it").await;
    assert_eq!(answer, "tool round done");
    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "attempt 1 fails, attempt 2 succeeds"
    );
    assert!(tool_result_text(&model).contains("recovered"));
}
#[tokio::test]
async fn the_resilience_switch_off_means_exactly_one_attempt() {
    let (_temp, workspace) = temp_workspace();
    let (tool, calls) = FlakyTool::new("flaky", true, 1);
    let config = base_config(&workspace).with_enable_tool_resilience_rail(false);
    let (agent, model) = tool_round_agent(config, tool, "flaky");
    invoke_once(&agent, "no retry").await;
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert!(tool_result_text(&model).contains("connection reset by peer"));
}
#[tokio::test]
async fn a_non_idempotent_tool_is_never_retried() {
    let (_temp, workspace) = temp_workspace();
    // `write`-like tools are non-idempotent: re-running one would repeat the side effect.
    let (tool, calls) = FlakyTool::new("flaky_write", false, 1);
    let (agent, _model) = tool_round_agent(base_config(&workspace), tool, "flaky_write");
    invoke_once(&agent, "write once").await;
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}
#[tokio::test]
async fn an_always_failing_idempotent_tool_stops_at_the_bounded_attempt_count() {
    let (_temp, workspace) = temp_workspace();
    let (tool, calls) = FlakyTool::new("hopeless", true, usize::MAX);
    let (agent, model) = tool_round_agent(base_config(&workspace), tool, "hopeless");
    let answer = invoke_once(&agent, "fail it").await;
    assert_eq!(answer, "tool round done");
    assert_eq!(
        calls.load(Ordering::SeqCst),
        3,
        "the budget is three attempts, counting the first try"
    );
    assert!(tool_result_text(&model).contains("connection reset by peer"));
}
// ---------------------------------------------------------------------------
// (c) `enable_task_planning` gates the todo tools and the prompt section.
// ---------------------------------------------------------------------------
#[tokio::test]
async fn task_planning_is_off_until_the_flag_is_set() {
    let (_temp, workspace) = temp_workspace();
    let model = ScriptedModel::text("planned");
    let agent = build_agent(base_config(&workspace), model.clone());
    let names: Vec<String> = agent
        .tool_cards()
        .into_iter()
        .map(|card| card.name)
        .collect();
    assert!(
        names.iter().all(|name| !name.starts_with("todo_")),
        "a default agent has no plan tool: {names:?}"
    );
    invoke_once(&agent, "plan nothing").await;
    assert!(!system_text(&model).contains("## Task Plan"));
    let model = ScriptedModel::text("planned");
    let agent = build_agent(
        base_config(&workspace).with_enable_task_planning(true),
        model.clone(),
    );
    let names: Vec<String> = agent
        .tool_cards()
        .into_iter()
        .map(|card| card.name)
        .collect();
    for expected in ["todo_create", "todo_list", "todo_get", "todo_modify"] {
        assert!(
            names.iter().any(|name| name == expected),
            "missing plan tool {expected}: {names:?}"
        );
    }
    invoke_once(&agent, "plan it").await;
    let system = system_text(&model);
    assert!(
        system.contains("## Task Plan"),
        "the todo section is missing"
    );
    assert!(system.contains("todo_create"));
}
// ---------------------------------------------------------------------------
// (d) `invocation_log_dir` journals one line per turn.
// ---------------------------------------------------------------------------
#[tokio::test]
async fn a_configured_invocation_log_records_the_turn() {
    let (_temp, workspace) = temp_workspace();
    let log_dir = workspace.root_path.join("logs");
    assert!(
        !log_dir.exists(),
        "the fixture starts without the directory"
    );
    let agent = build_agent(
        base_config(&workspace).with_invocation_log_dir(Some(log_dir.clone())),
        ScriptedModel::text("journaled answer"),
    );
    let session_id = "journal-session";
    let session: Arc<dyn Session> = Arc::new(AgentSession::new(session_id));
    let output = agent
        .invoke(json!({"query": "log this turn"}), Some(session))
        .await
        .expect("invocation succeeds");
    assert_eq!(output.output, "journaled answer");
    let file = log_dir.join(format!("{}-{session_id}.jsonl", agent.card().id));
    let content = std::fs::read_to_string(&file).expect("the journal file exists");
    let lines: Vec<&str> = content.lines().collect();
    assert_eq!(lines.len(), 1, "one JSONL line per turn: {content}");
    let entry: Value = serde_json::from_str(lines[0]).expect("the line is JSON");
    assert_eq!(entry["query"], "log this turn");
    assert_eq!(entry["answer"], "journaled answer");
    assert_eq!(entry["status"], "answer");
    assert_eq!(entry["session_id"], session_id);
    assert_eq!(entry["agent_id"], agent.card().id);
}
#[tokio::test]
async fn an_unset_invocation_log_dir_writes_nothing() {
    let (_temp, workspace) = temp_workspace();
    let agent = build_agent(base_config(&workspace), ScriptedModel::text("quiet"));
    assert_eq!(agent.config().invocation_log_dir, None);
    invoke_once(&agent, "stay quiet").await;
    assert!(
        !workspace.root_path.join("logs").exists(),
        "no journal directory is created without `invocation_log_dir`"
    );
    let entries: Vec<_> = std::fs::read_dir(&workspace.root_path)
        .expect("the workspace is readable")
        .filter_map(std::result::Result::ok)
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .filter(|name| {
            Path::new(name)
                .extension()
                .is_some_and(|ext| ext == "jsonl")
        })
        .collect();
    assert!(
        entries.is_empty(),
        "no journal file was written: {entries:?}"
    );
}
#[tokio::test]
async fn an_unusable_invocation_log_dir_is_warn_only() {
    let (temp, workspace) = temp_workspace();
    let blocker = temp.path().join("blocked-logs");
    std::fs::write(&blocker, b"not a directory").expect("the blocker file exists");
    let agent = build_agent(
        base_config(&workspace).with_invocation_log_dir(Some(blocker.clone())),
        ScriptedModel::text("still answers"),
    );
    let answer = invoke_once(&agent, "fail to journal").await;
    assert_eq!(answer, "still answers", "journaling never costs the turn");
    assert_eq!(
        std::fs::read(&blocker).expect("the blocker is untouched"),
        b"not a directory"
    );
}
// ---------------------------------------------------------------------------
// (e) `project_root` is an access boundary for the tools and the permission engine.
// ---------------------------------------------------------------------------
/// Creates `<root>/project/notes.md` and `<root>/sibling/notes.md` and returns their paths.
fn project_layout(root: &Path) -> (std::path::PathBuf, std::path::PathBuf) {
    let project = root.join("project");
    let sibling = root.join("sibling");
    std::fs::create_dir_all(&project).expect("project directory");
    std::fs::create_dir_all(&sibling).expect("sibling directory");
    std::fs::write(project.join("notes.md"), "inside the project").expect("project file");
    std::fs::write(sibling.join("notes.md"), "outside the project").expect("sibling file");
    (project, sibling)
}
/// The mounted tool called `name`.
fn tool<'a>(tools: &'a [Arc<dyn Tool>], name: &str) -> &'a Arc<dyn Tool> {
    tools
        .iter()
        .find(|tool| tool.card().name == name)
        .unwrap_or_else(|| panic!("{name} must be mounted"))
}
/// Invokes a tool and returns its payload.
async fn call(tools: &[Arc<dyn Tool>], name: &str, args: Value) -> Value {
    tool(tools, name)
        .invoke(args, &ToolContext::new())
        .await
        .expect("the file tools report failures as payloads")
}
#[tokio::test]
async fn project_root_denies_an_absolute_sibling_path_and_allows_the_project() {
    let (temp, workspace) = temp_workspace();
    let (project, sibling) = project_layout(temp.path());
    let tools = ToolRegistry::new(workspace.clone())
        .with_project_root(Some(project.clone()))
        .build();
    let outside = sibling.join("notes.md");
    let denied = call(
        &tools,
        "read_file",
        json!({"file_path": outside.to_string_lossy()}),
    )
    .await;
    assert_eq!(denied["success"], json!(false), "payload: {denied}");
    assert!(
        denied["error"]
            .as_str()
            .is_some_and(|error| error.contains(OUTSIDE_PROJECT_ROOT)),
        "the read must be attributed to the project root: {denied}"
    );
    let denied = call(
        &tools,
        "write_file",
        json!({
            "file_path": sibling.join("written.md").to_string_lossy(),
            "content": "nope",
        }),
    )
    .await;
    assert_eq!(denied["success"], json!(false), "payload: {denied}");
    assert!(
        denied["error"]
            .as_str()
            .is_some_and(|error| error.contains(OUTSIDE_PROJECT_ROOT)),
        "the write must be attributed to the project root: {denied}"
    );
    assert!(!sibling.join("written.md").exists());
    let inside = project.join("notes.md");
    let allowed = call(
        &tools,
        "read_file",
        json!({"file_path": inside.to_string_lossy()}),
    )
    .await;
    assert_eq!(allowed["success"], json!(true), "payload: {allowed}");
    assert!(allowed.to_string().contains("inside the project"));
}
#[tokio::test]
async fn without_a_project_root_the_workspace_stays_the_only_boundary() {
    let (temp, workspace) = temp_workspace();
    let (_project, sibling) = project_layout(temp.path());
    let tools = ToolRegistry::new(workspace.clone())
        .with_project_root(None)
        .build();
    let read = call(
        &tools,
        "read_file",
        json!({"file_path": sibling.join("notes.md").to_string_lossy()}),
    )
    .await;
    assert_eq!(read["success"], json!(true), "payload: {read}");
    assert!(read.to_string().contains("outside the project"));
}
#[tokio::test]
async fn the_builder_threads_project_root_into_the_mounted_tools() {
    let (temp, workspace) = temp_workspace();
    let (project, sibling) = project_layout(temp.path());
    let model = ScriptedModel::builder()
        .completion(MockCompletion::tool_call(
            "read_file",
            json!({"file_path": sibling.join("notes.md").to_string_lossy()}),
        ))
        .respond_text("done")
        .build();
    let agent = build_agent(
        base_config(&workspace).with_project_root(Some(project)),
        model.clone(),
    );
    invoke_once(&agent, "read the sibling").await;
    assert!(
        tool_result_text(&model).contains(OUTSIDE_PROJECT_ROOT),
        "the configured project root must reach the mounted tool: {}",
        tool_result_text(&model)
    );
}
#[test]
fn the_permission_engine_denies_a_path_outside_the_project_root() {
    let (temp, workspace) = temp_workspace();
    let (project, sibling) = project_layout(temp.path());
    let section = PermissionsSection {
        package_builtin_rules: false,
        ..PermissionsSection::default()
    };
    let engine = PermissionEngine::new(
        PermissionEngineConfig::new(section)
            .with_workspace_root(workspace.root_path.clone())
            .with_project_root(project.clone()),
    );
    let outside = json!({"file_path": sibling.join("notes.md").to_string_lossy()});
    let result = engine.check_permission("read_file", &outside);
    assert_eq!(result.permission, PermissionLevel::Deny);
    assert!(
        result
            .matched_rule
            .as_deref()
            .is_some_and(|rule| rule.contains("project_root")),
        "the rule must name the boundary: {result:?}"
    );
    assert!(result.external_paths.is_some_and(|paths| !paths.is_empty()));
    let inside = json!({"file_path": project.join("notes.md").to_string_lossy()});
    let result = engine.check_permission("read_file", &inside);
    assert_ne!(
        result.permission,
        PermissionLevel::Deny,
        "a path inside the project root is not denied: {result:?}"
    );
    assert!(result
        .matched_rule
        .as_deref()
        .is_none_or(|rule| !rule.contains("project_root")));
}
// ---------------------------------------------------------------------------
// (f) the sub-agent mode selects the mounted sub-agent tool surface.
// ---------------------------------------------------------------------------
/// The mounted tool names of an agent with one sub-agent under `config`.
fn subagent_tool_names(config: &DeepAgentConfig, workspace: &Workspace) -> Vec<String> {
    let agent = DeepAgentBuilder::new(AgentCard::new("subagent-mode-agent"))
        .config(
            config
                .clone()
                .with_workspace(workspace.clone())
                .with_language("en".to_string())
                .with_subagents(vec![SubAgentConfig::new(AgentCard::new("helper"))]),
        )
        .model(Arc::new(ScriptedModel::text("hi")))
        .build()
        .expect("agent builds");
    agent
        .tool_cards()
        .into_iter()
        .map(|card| card.name)
        .collect()
}
#[test]
fn the_default_subagent_mode_mounts_only_the_task_tool() {
    let (_temp, workspace) = temp_workspace();
    let names = subagent_tool_names(&DeepAgentConfig::default(), &workspace);
    assert!(names.iter().any(|name| name == "task_tool"), "{names:?}");
    assert!(
        names.iter().all(|name| !name.starts_with("subagent_")),
        "the default mounts no persistent runtime tool: {names:?}"
    );
}
#[test]
fn the_async_subagent_mode_mounts_the_lifecycle_subset() {
    let (_temp, workspace) = temp_workspace();
    let names = subagent_tool_names(
        &DeepAgentConfig::default().with_enable_async_subagent(true),
        &workspace,
    );
    let subagent: Vec<&str> = names
        .iter()
        .filter(|name| name.starts_with("subagent_"))
        .map(String::as_str)
        .collect();
    assert_eq!(
        subagent,
        vec![
            "subagent_spawn",
            "subagent_wait",
            "subagent_list",
            "subagent_close"
        ],
        "the async mode mounts the four lifecycle tools: {names:?}"
    );
    assert!(names.iter().any(|name| name == "task_tool"), "{names:?}");
}
#[test]
fn the_subagent_runtime_mode_mounts_the_runtime_surface() {
    let (_temp, workspace) = temp_workspace();
    let names = subagent_tool_names(
        &DeepAgentConfig::default().with_enable_subagent_runtime(true),
        &workspace,
    );
    assert_eq!(
        names
            .iter()
            .filter(|name| name.starts_with("subagent_"))
            .count(),
        7,
        "the runtime mode mounts the seven-tool surface: {names:?}"
    );
    assert!(names.iter().any(|name| name == "task_tool"), "{names:?}");
}
