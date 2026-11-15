//! One-shot execution behind `openjiuwen run`.
//!
//! [`run_once`] is what the binary calls: it resolves the prompt (argument or
//! piped stdin), builds the model from [`CliOptions`], and hands both to
//! [`run_with_model`]. The latter is the test seam — it accepts a prebuilt
//! [`Model`] so tests drive a scripted model and never touch the network.
//!
//! A run streams exactly one agent invocation and renders it per
//! [`OutputFormat`]:
//!
//! - `text` — chunks are rendered to the terminal while they arrive.
//! - `json` — nothing is printed while streaming; one pretty-printed
//!   [`RunOutcome::to_json`] object is printed at the end.
//! - `stream-json` — one `{"type", "index", "payload"}` line per chunk plus a
//!   terminal `{"type": "result", …}` line so consumers learn the session id.
//!
//! Session records are append-only and best-effort: a failed write is logged and
//! never loses the answer.
//!
//! The agent built here composes four CLI-owned inputs:
//!
//! - **instruction files** — `IDENTITY.md`/`OPENJIUWEN.md` under the workspace
//!   and the home directory become the system prompt ([`InstructionFiles`]);
//! - **built-in tools** — the builder mounts the twenty default coding tools and
//!   this module adds the thirteen extended tools it can drive (`fetch_webpage`,
//!   `web_search`, `image_metadata`, `audio_metadata`, `cron`, the two MCP
//!   resource tools, `tool_search`, the two goal tools and the three worktree
//!   tools); `lsp` stays out because it needs a per-workspace server
//!   configuration the CLI does not resolve yet, and `mcp_resource_subscription`
//!   stays out because it needs a subscription registry the CLI does not wire;
//! - **MCP tools** — the servers declared in `mcp.json` are connected through
//!   [`McpToolProvider`] and mounted next to the built-in coding tools;
//! - **permission rules** — the permanent allow rules under the CLI home are
//!   merged onto the built-in section and handed to the agent and to
//!   [`CliPermissionHost`], which writes new "always allow" answers back.
// Size note: ~520 lines of implementation plus an inline test module covering every
// output format, prompt source and failure path; the resolution helpers live in `paths`,
// `usage` and `options`.

use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use openjiuwen_agent::{Agent, AgentChunk};
use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::llm::Model;
use openjiuwen_foundation::tool::mcp::{
    connect_mcp_server, load_mcp_tools, McpServerConfig, McpTool, McpToolOptions,
};
use openjiuwen_foundation::tool::Tool;
use openjiuwen_session::{AgentSession, Session};
use serde_json::{json, Value};

use super::allow_rules::AllowRules;
use super::config::{CliOptions, ProcessEnv, DEFAULT_LANGUAGE, SESSION_PREFIX};
use super::instruction::InstructionFiles;
use super::io::CliIo;
use super::mcp_config;
use super::permissions::CliPermissionHost;
use super::render::RenderState;
use super::style::Style;
use super::RunArgs;
use crate::agent::{create_deep_agent, DeepAgent};
use crate::permission::{
    PermissionEngine, PermissionEngineConfig, PermissionHost, PermissionsSection,
};
use crate::schema::DeepAgentConfig;
use crate::tools::{BuiltinTool, ToolRegistry};
use crate::workspace::Workspace;

/// Message shown when a run has no prompt at all.
const EMPTY_PROMPT: &str = "A prompt argument is required, or pipe via stdin.";

/// Session role of a user message in a persisted record.
const ROLE_USER: &str = "user";

/// Session role of an assistant message in a persisted record.
const ROLE_ASSISTANT: &str = "assistant";

/// Result of one completed run.
#[derive(Debug, Clone, PartialEq)]
pub struct RunOutcome {
    /// Final answer of the invocation.
    pub answer: String,
    /// Result type reported by the terminal `answer` frame (`answer`, `error`, …).
    pub result_type: String,
    /// Token usage captured from the last `llm_usage` frame, when reported.
    pub usage: Option<Value>,
    /// Session id the run used.
    pub session_id: String,
    /// Number of streamed chunks consumed.
    pub chunks: usize,
}

impl RunOutcome {
    /// Renders the outcome as the `json` output format's object.
    ///
    /// The keys are `answer`, `result_type`, `usage` and `session_id`; usage is
    /// `null` when the model reported none.
    #[must_use]
    pub fn to_json(&self) -> Value {
        json!({
            "answer": self.answer,
            "result_type": self.result_type,
            "usage": self.usage,
            "session_id": self.session_id,
        })
    }
}

/// Renders one streamed chunk as the `stream-json` output format's line.
#[must_use]
pub fn stream_json_line(chunk: &AgentChunk) -> Value {
    json!({
        "type": chunk.kind,
        "index": chunk.index,
        "payload": chunk.payload,
    })
}

/// Renders the terminal line of the `stream-json` output format.
///
/// It is [`RunOutcome::to_json`] with `"type": "result"` added, so a consumer
/// learns the session id without a separate channel.
#[must_use]
pub fn result_json_line(outcome: &RunOutcome) -> Value {
    let mut value = outcome.to_json();
    if let Value::Object(object) = &mut value {
        object.insert("type".to_owned(), Value::String("result".to_owned()));
    }
    value
}

/// Creates a new CLI session id: [`SESSION_PREFIX`] plus eight hex characters.
#[must_use]
pub fn session_id() -> String {
    format!(
        "{}{}",
        SESSION_PREFIX,
        &uuid::Uuid::new_v4().simple().to_string()[..8]
    )
}

/// Builds a kernel session for `session_id`.
#[must_use]
pub fn new_session(session_id: &str) -> Arc<dyn Session> {
    Arc::new(AgentSession::new(session_id))
}

/// Loads MCP servers and turns them into agent tools.
///
/// The CLI build depends on this seam rather than on concrete MCP transports, so
/// a test can hand [`build_agent_with`] a recording fake and assert which servers
/// were requested without spawning a server process.
#[async_trait]
pub trait McpToolProvider: Send + Sync {
    /// Returns one tool per remote tool advertised by `servers`.
    ///
    /// # Errors
    ///
    /// Returns the error the provider maps the failure to. The foundation-backed
    /// implementation never fails: it logs and skips a broken server.
    async fn load_tools(&self, servers: &[McpServerConfig]) -> Result<Vec<Arc<dyn Tool>>>;
}

/// Connects real MCP servers through `openjiuwen-foundation`.
///
/// Each server is connected with
/// [`connect_mcp_server`](openjiuwen_foundation::tool::mcp::connect_mcp_server)
/// and its tools are loaded with
/// [`load_mcp_tools`](openjiuwen_foundation::tool::mcp::load_mcp_tools) using
/// [`McpToolOptions::default`].
///
/// A server that cannot be connected or listed is logged with `tracing::warn!`
/// and skipped: one broken `mcp.json` entry never blocks the CLI, and the
/// remaining servers keep working.
#[derive(Debug, Default, Clone, Copy)]
pub struct FoundationMcpToolProvider;

impl FoundationMcpToolProvider {
    /// Returns a provider backed by the foundation MCP clients.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl McpToolProvider for FoundationMcpToolProvider {
    async fn load_tools(&self, servers: &[McpServerConfig]) -> Result<Vec<Arc<dyn Tool>>> {
        let mut tools: Vec<Arc<dyn Tool>> = Vec::new();
        for config in servers {
            let client = match connect_mcp_server(config) {
                Ok(client) => client,
                Err(error) => {
                    tracing::warn!(
                        server = %config.server_name,
                        %error,
                        "skipping MCP server that could not be connected"
                    );
                    continue;
                }
            };
            match load_mcp_tools(client, McpToolOptions::default()).await {
                Ok(loaded) => tools.extend(loaded.into_iter().map(McpTool::boxed)),
                Err(error) => {
                    tracing::warn!(
                        server = %config.server_name,
                        %error,
                        "skipping MCP server whose tools could not be listed"
                    );
                }
            }
        }
        Ok(tools)
    }
}

/// Builds the `DeepAgent` a CLI run or REPL turn executes on.
///
/// Delegates to [`build_agent_with`] with [`FoundationMcpToolProvider`], so the
/// default build connects the MCP servers configured in `mcp.json` (none by
/// default, which keeps the crate fully offline).
///
/// # Errors
///
/// Returns whatever agent construction reports: [`Error::Config`] for an invalid
/// configuration, [`Error::Validation`] for an unusable workspace.
pub async fn build_agent(
    model: Arc<dyn Model>,
    options: &CliOptions,
    io: &CliIo,
) -> Result<Arc<DeepAgent>> {
    build_agent_with(model, options, io, &FoundationMcpToolProvider).await
}

/// Builds the `DeepAgent` of a CLI invocation with an explicit MCP provider.
///
/// The configuration comes from [`CliOptions`]: the iteration budget, the model
/// identity (name, provider, key, base URL) and the CLI language. The working
/// directory is the process directory so the agent reads and writes the project
/// the user invoked it from, falling back to the workspace when the process
/// directory cannot be determined.
///
/// Four CLI-owned inputs are folded in here:
///
/// - [`InstructionFiles::load`] reads `IDENTITY.md`/`OPENJIUWEN.md` from the
///   workspace and the home directory and hands the composed text to
///   [`DeepAgentConfig::with_system_prompt`]. That replaces the built-in identity
///   preamble — the same way the Python CLI composes its identity and memory
///   layers into one system section — and leaves the built-in prompt untouched
///   when no instruction file exists.
/// - The thirteen extended built-in tools the CLI can drive (`fetch_webpage`,
///   `web_search`, `image_metadata`, `audio_metadata`, `cron`, the two MCP resource
///   tools, `tool_search`, the two goal tools and `worktree`/`enter_worktree`/
///   `exit_worktree`) are mounted on top of [`ToolRegistry`]'s default set, so a
///   CLI turn can fetch pages, search, report media metadata, schedule cron jobs,
///   search the tool index, read MCP resources, track a goal and manage worktrees.
///   `lsp` and `mcp_resource_subscription` are left out — see `CLI_EXTENDED_TOOLS`.
/// - `mcp.json` (see [`mcp_config::load_for`]) is loaded and every server entry
///   is passed to `provider`, whose tools are mounted beside the built-in coding
///   tools. With no configuration file the list is empty.
/// - The permanent allow rules under the CLI home ([`AllowRules::for_home`]) are
///   merged onto [`CliOptions::permissions`] and handed to both the agent and
///   [`CliPermissionHost`], which writes new "always allow" answers back to
///   `<home>/permissions.yaml`.
///
/// `io` is handed to [`CliPermissionHost`], which is what makes permission asks
/// answerable in the REPL and fail closed in a one-shot run. The host prompts in
/// colour exactly when `io` has colour enabled.
///
/// # Errors
///
/// Returns whatever agent construction reports: [`Error::Config`] for an invalid
/// configuration or an unreadable `mcp.json`/instruction file,
/// [`Error::Validation`] for an unusable workspace.
pub async fn build_agent_with(
    model: Arc<dyn Model>,
    options: &CliOptions,
    io: &CliIo,
    provider: &dyn McpToolProvider,
) -> Result<Arc<DeepAgent>> {
    let cwd = std::env::current_dir().unwrap_or_else(|_| options.workspace.clone());
    let instructions = InstructionFiles::load(&options.workspace, &options.home)?;
    let config = DeepAgentConfig {
        enable_task_planning: true,
        max_iterations: Some(options.max_iterations),
        language: DEFAULT_LANGUAGE.to_owned(),
        cwd: Some(cwd.clone()),
        project_root: Some(cwd),
        model_name: Some(options.model.clone()),
        model_provider: Some(options.provider.clone()),
        api_key: options.api_key.clone(),
        api_base: options.api_base.clone(),
        ..DeepAgentConfig::default()
    }
    .with_system_prompt(instructions.system_prompt());
    let servers = mcp_config::load_for(&options.home, &options.workspace, &ProcessEnv)?;
    let mcp_tools = provider.load_tools(&servers).await?;
    let (rules, permissions) = resolve_permissions(options)?;
    let workspace = Workspace::for_language(&options.workspace, DEFAULT_LANGUAGE);
    let mut tools = cli_extended_tools(&workspace, &permissions);
    tools.extend(mcp_tools);
    let host: Arc<dyn PermissionHost> = Arc::new(permission_host(io, options, rules, &permissions));
    create_deep_agent(
        model,
        None,
        Some(config),
        tools,
        Vec::new(),
        Some(workspace),
        Vec::new(),
        Vec::new(),
        true,
        Some(options.max_iterations),
        Some(DEFAULT_LANGUAGE.to_owned()),
        None,
        false,
        Some(permissions),
        Some(host),
    )
    .await
}

/// The extended built-in tools a CLI run or REPL turn mounts.
///
/// These are [`BuiltinTool::EXTENDED`] minus [`BuiltinTool::DEFAULT`] (which the builder already
/// mounts) minus two documented exclusions:
///
/// - `lsp` needs a per-workspace server configuration the CLI does not resolve yet: mounting it
///   would only offer the model a tool that always answers "no LSP server configured".
/// - `mcp_resource_subscription` needs a subscription registry the CLI does not wire, so mounting
///   it would only offer a tool that never has a subscription to report.
///
/// Everything else is mountable without host wiring: the online media/resource tools answer with
/// their typed `success=false` payload when no backend is configured, and the worktree tools share
/// the registry's in-process [`crate::worktree::MemoryWorktreeSessionStore`] and degrade to the
/// manager's typed git error outside a repository. Mounting a goal tool also makes the factory
/// register the goal rail.
const CLI_EXTENDED_TOOLS: [BuiltinTool; 13] = [
    BuiltinTool::FetchWebpage,
    BuiltinTool::WebSearch,
    BuiltinTool::ImageMetadata,
    BuiltinTool::AudioMetadata,
    BuiltinTool::Cron,
    BuiltinTool::ListMcpResources,
    BuiltinTool::ReadMcpResource,
    BuiltinTool::ToolSearch,
    BuiltinTool::GetCurrentGoal,
    BuiltinTool::SubmitGoalReport,
    BuiltinTool::Worktree,
    BuiltinTool::EnterWorktree,
    BuiltinTool::ExitWorktree,
];

/// Builds [`CLI_EXTENDED_TOOLS`] over `workspace` for a CLI run.
///
/// The tools come from a [`ToolRegistry`] restricted to that set, which reuses the builder's
/// seams: the permission engine of `permissions` folds the network guard of `fetch_webpage`, the
/// worktree tools' guard and the metadata tools' path checks. `web_search`, the two MCP resource
/// tools and the worktree tools need a host-supplied backend or a git repository; without one the
/// mounted tools answer with their documented `success=false` payload instead of failing the run,
/// and no network default is added here. Mounting a goal tool also makes the factory register the
/// goal rail.
fn cli_extended_tools(
    workspace: &Workspace,
    permissions: &PermissionsSection,
) -> Vec<Arc<dyn Tool>> {
    let mut registry = ToolRegistry::new(workspace.clone())
        .with_tools(CLI_EXTENDED_TOOLS)
        .with_language(DEFAULT_LANGUAGE);
    if permissions.enabled {
        registry = registry.with_permissions(Arc::new(PermissionEngine::new(
            PermissionEngineConfig::new(permissions.clone())
                .with_workspace_root(workspace.root_path.clone()),
        )));
    }
    registry.build()
}

/// Resolves the effective permission section of `options`.
///
/// The permanent rules under the CLI home are merged on top of the built-in
/// section, so a stored `write_file: allow` reaches both the agent and the
/// permission host. Returns the rule handle too, so the host can write back.
fn resolve_permissions(options: &CliOptions) -> Result<(AllowRules, PermissionsSection)> {
    let rules = AllowRules::for_home(&options.home);
    let permissions = rules.load_merged(&options.permissions())?;
    Ok((rules, permissions))
}

/// Builds the permission host the agent answers asks through.
///
/// It is registered over the freshly generated CLI session id, persists
/// "always allow" answers through `rules` and styles its prompts per `io`.
fn permission_host(
    io: &CliIo,
    options: &CliOptions,
    rules: AllowRules,
    permissions: &PermissionsSection,
) -> CliPermissionHost {
    CliPermissionHost::new(
        io.clone(),
        session_id(),
        options.workspace.clone(),
        permissions.clone(),
    )
    .with_allow_rules(rules)
    .with_style(Style::new(io.is_color()))
}

/// Runs one prompt and returns what happened.
///
/// The prompt is resolved (and validated) before the model is built, so an empty
/// prompt fails fast without touching credentials or the network.
///
/// # Errors
///
/// Returns [`Error::Validation`] for an empty prompt, and whatever model or
/// agent construction reports.
pub async fn run_once(options: &CliOptions, args: &RunArgs, io: &CliIo) -> Result<RunOutcome> {
    let prompt = resolve_prompt(args, io).await?;
    let model = options.build_model().await?;
    // Hand the resolved prompt down so stdin is read once, not twice.
    let resolved = RunArgs {
        prompt: Some(prompt),
        output_format: args.output_format,
    };
    run_with_model(options, &resolved, io, model).await
}

/// Runs one prompt on a prebuilt model (the test seam).
///
/// See the module docs for the per-format output shapes.
///
/// This resolves the prompt, builds the agent and creates the run's CLI session
/// record, then hands the turn to `stream_run`, which owns streaming, rendering
/// and the transcript append.
///
/// # Errors
///
/// Returns [`Error::Validation`] for an empty prompt, and whatever agent
/// construction or the stream reports. Session-write failures are logged, never
/// returned.
pub async fn run_with_model(
    options: &CliOptions,
    args: &RunArgs,
    io: &CliIo,
    model: Arc<dyn Model>,
) -> Result<RunOutcome> {
    let prompt = resolve_prompt(args, io).await?;
    let agent = build_agent(model, options, io).await?;
    let run_session_id = session_id();
    let session = new_session(&run_session_id);
    let store = options.sessions();
    if let Err(error) = store.new_session(&run_session_id, &options.model) {
        tracing::warn!(%error, "failed to create CLI session record");
    }

    stream_run(
        options,
        args,
        io,
        &agent,
        &session,
        &run_session_id,
        &prompt,
    )
    .await
}

/// Streams one agent turn, renders it per `args.output_format` and records it in
/// the CLI session.
///
/// It owns the per-turn body: streaming `prompt` through `agent` on `session`,
/// rendering every chunk (to `io` for `text`, buffered for `json`, one line per
/// chunk plus a terminal result line for `stream-json`), building the
/// [`RunOutcome`], writing the format's terminal output, and appending the user and
/// assistant messages to the CLI session record of `session_id` (warn-only, so a
/// failed write never loses the answer).
///
/// Session-record *creation* stays with the caller: `SessionStore::new_session`
/// truncates an existing record, so only a caller starting a fresh run may call it.
///
/// # Errors
///
/// Returns the stream error, and any render or terminal-write failure.
pub(crate) async fn stream_run(
    options: &CliOptions,
    args: &RunArgs,
    io: &CliIo,
    agent: &Arc<DeepAgent>,
    session: &Arc<dyn Session>,
    session_id: &str,
    prompt: &str,
) -> Result<RunOutcome> {
    // Plain output stays byte-identical: colour is only on when the handle says so.
    let mut render = RenderState::styled(io.is_color());
    let mut chunks = 0_usize;
    let mut stream = agent.stream(json!({"query": prompt}), Some(Arc::clone(session)));
    while let Some(item) = stream.next().await {
        let chunk = item?;
        chunks += 1;
        match args.output_format {
            super::OutputFormat::Text => render.render(&chunk, Some(io), options.verbose).await?,
            super::OutputFormat::Json => render.render(&chunk, None, false).await?,
            super::OutputFormat::StreamJson => {
                let line =
                    serde_json::to_string(&stream_json_line(&chunk)).map_err(Error::external)?;
                io.write_line(&line).await?;
                render.render(&chunk, None, false).await?;
            }
        }
    }

    let outcome = RunOutcome {
        answer: render.answer().to_owned(),
        result_type: render.result_type().to_owned(),
        usage: render.usage.clone(),
        session_id: session_id.to_owned(),
        chunks,
    };

    match args.output_format {
        super::OutputFormat::Text => render.finish(Some(io)).await?,
        super::OutputFormat::Json => {
            let text = serde_json::to_string_pretty(&outcome.to_json()).map_err(Error::external)?;
            io.write_line(&text).await?;
        }
        super::OutputFormat::StreamJson => {
            let line =
                serde_json::to_string(&result_json_line(&outcome)).map_err(Error::external)?;
            io.write_line(&line).await?;
        }
    }

    let store = options.sessions();
    if let Err(error) = store.append_message(session_id, ROLE_USER, prompt) {
        tracing::warn!(%error, "failed to persist CLI user message");
    }
    if let Err(error) = store.append_message(session_id, ROLE_ASSISTANT, &outcome.answer) {
        tracing::warn!(%error, "failed to persist CLI assistant message");
    }

    Ok(outcome)
}

/// Resolves the prompt of a run: the argument when given, stdin otherwise.
///
/// A `None` prompt or the literal `-` reads stdin to end of input and joins the
/// lines with `\n`; the result is trimmed in both cases.
///
/// Shared with the resumed runner of [`super::resume`], so a resumed prompt is
/// resolved by exactly the same rules as a fresh one.
pub(crate) async fn resolve_prompt(args: &RunArgs, io: &CliIo) -> Result<String> {
    let prompt = match &args.prompt {
        Some(prompt) if prompt != "-" => prompt.trim().to_owned(),
        _ => {
            let mut lines = Vec::new();
            while let Some(line) = io.prompt_line("").await? {
                lines.push(line);
            }
            lines.join("\n").trim().to_owned()
        }
    };
    if prompt.is_empty() {
        return Err(Error::validation(EMPTY_PROMPT));
    }
    Ok(prompt)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::collections::VecDeque;
    use std::path::Path;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    use openjiuwen_agent::Agent;
    use openjiuwen_common::config::OpenJiuwenHome;
    use openjiuwen_foundation::llm::Model;
    use openjiuwen_foundation::tool::mcp::McpClientType;
    use openjiuwen_foundation::tool::{ToolCard, ToolContext};
    use openjiuwen_foundation::Role;
    use openjiuwen_test_support::{MockCompletion, ScriptedModel};
    use tempfile::TempDir;

    use super::*;
    use crate::cli::config::{
        DEFAULT_MAX_ITERATIONS, DEFAULT_MAX_TOKENS, DEFAULT_MODEL, DEFAULT_PROVIDER,
    };
    use crate::cli::{LineReader, OutputFormat, Writer};
    use crate::permission::PermissionLevel;

    /// [`LineReader`] answering from a fixed script, then EOF.
    struct ScriptedReader {
        /// Lines still to hand out.
        lines: VecDeque<String>,
    }

    #[async_trait]
    impl LineReader for ScriptedReader {
        async fn read_line(&mut self) -> Result<Option<String>> {
            Ok(self.lines.pop_front())
        }
    }

    /// [`Writer`] discarding everything it is given.
    struct SilentWriter;

    #[async_trait]
    impl Writer for SilentWriter {
        async fn write(&mut self, _text: &str) -> Result<()> {
            Ok(())
        }

        async fn flush(&mut self) -> Result<()> {
            Ok(())
        }
    }

    /// Builds a scripted non-interactive handle (the one-shot `run` path).
    fn scripted_io(lines: &[&str]) -> CliIo {
        let reader = Box::new(ScriptedReader {
            lines: lines.iter().map(|line| (*line).to_owned()).collect(),
        });
        CliIo::new(reader, Box::new(SilentWriter))
    }

    /// Builds the resolved options of a CLI run rooted at `root`.
    fn test_options(root: &Path) -> CliOptions {
        CliOptions {
            model: DEFAULT_MODEL.to_owned(),
            provider: DEFAULT_PROVIDER.to_owned(),
            api_key: None,
            api_base: None,
            workspace: root.join("workspace"),
            max_iterations: DEFAULT_MAX_ITERATIONS,
            max_tokens: DEFAULT_MAX_TOKENS,
            verbose: false,
            home: OpenJiuwenHome::from_root(root),
        }
    }

    /// Minimal [`Tool`] standing in for a remote MCP tool.
    struct FakeTool {
        /// Card the agent sees.
        card: ToolCard,
    }

    #[async_trait]
    impl Tool for FakeTool {
        fn card(&self) -> &ToolCard {
            &self.card
        }

        async fn invoke(&self, inputs: Value, _ctx: &ToolContext) -> Result<Value> {
            Ok(inputs)
        }
    }

    /// [`McpToolProvider`] recording its calls and returning fixed tools.
    struct RecordingProvider {
        /// Number of `load_tools` calls.
        calls: AtomicUsize,
        /// Servers of the most recent call.
        servers: Mutex<Vec<McpServerConfig>>,
        /// Tools handed back to the build.
        tools: Vec<Arc<dyn Tool>>,
    }

    impl RecordingProvider {
        /// Builds a provider returning `tools` and recording every call.
        fn new(tools: Vec<Arc<dyn Tool>>) -> Self {
            Self {
                calls: AtomicUsize::new(0),
                servers: Mutex::new(Vec::new()),
                tools,
            }
        }

        /// Returns the server names recorded by the last call.
        fn recorded_names(&self) -> Vec<String> {
            self.servers
                .lock()
                .unwrap()
                .iter()
                .map(|config| config.server_name.clone())
                .collect()
        }
    }

    #[async_trait]
    impl McpToolProvider for RecordingProvider {
        async fn load_tools(&self, servers: &[McpServerConfig]) -> Result<Vec<Arc<dyn Tool>>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            *self.servers.lock().unwrap() = servers.to_vec();
            Ok(self.tools.clone())
        }
    }

    #[tokio::test]
    async fn build_agent_without_mcp_config_runs_one_turn() {
        let temp = TempDir::new().unwrap();
        let options = test_options(temp.path());
        std::fs::create_dir_all(&options.workspace).unwrap();
        let io = scripted_io(&[]);
        let model: Arc<dyn Model> = Arc::new(ScriptedModel::text("scripted answer"));

        let agent = build_agent(model, &options, &io).await.unwrap();
        let output = agent
            .invoke(json!({"query": "hi"}), Some(new_session("s-1")))
            .await
            .unwrap();
        assert_eq!(output.output, "scripted answer");
    }

    #[tokio::test]
    async fn mcp_config_servers_reach_the_provider_and_its_tools_are_mounted() {
        let temp = TempDir::new().unwrap();
        let options = test_options(temp.path());
        std::fs::create_dir_all(&options.workspace).unwrap();
        std::fs::write(
            temp.path().join("mcp.json"),
            r#"{"mcpServers": {"filesystem": {"command": "npx"}}}"#,
        )
        .unwrap();
        let tool: Arc<dyn Tool> = Arc::new(FakeTool {
            card: ToolCard::new(
                "remote_lookup",
                "looks things up",
                json!({"type": "object"}),
            ),
        });
        let provider = RecordingProvider::new(vec![tool]);
        let io = scripted_io(&[]);
        let model: Arc<dyn Model> = Arc::new(ScriptedModel::text("done"));

        let agent = build_agent_with(model, &options, &io, &provider)
            .await
            .unwrap();
        assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
        assert_eq!(provider.recorded_names(), vec!["filesystem".to_owned()]);
        let recorded = provider.servers.lock().unwrap().clone();
        assert_eq!(recorded[0].client_type, McpClientType::Stdio);
        // The fake tool was handed to the agent build; one turn still completes.
        let output = agent
            .invoke(json!({"query": "hi"}), Some(new_session("s-1")))
            .await
            .unwrap();
        assert_eq!(output.output, "done");
    }

    #[tokio::test]
    async fn instruction_files_reach_the_system_prompt() {
        let temp = TempDir::new().unwrap();
        let options = test_options(temp.path());
        std::fs::create_dir_all(&options.workspace).unwrap();
        std::fs::write(options.workspace.join("OPENJIUWEN.md"), "PROJECT RULE 42\n").unwrap();
        let io = scripted_io(&[]);
        let model: Arc<dyn Model> = Arc::new(ScriptedModel::text("done"));

        let agent = build_agent_with(model, &options, &io, &FoundationMcpToolProvider)
            .await
            .unwrap();
        let prompt = agent.config().system_prompt.unwrap();
        assert!(
            prompt.contains("PROJECT RULE 42"),
            "the workspace memory must reach the system prompt:\n{prompt}"
        );
    }

    #[tokio::test]
    async fn persisted_allow_rules_reach_the_agent_and_the_permission_host() {
        let temp = TempDir::new().unwrap();
        let options = test_options(temp.path());
        std::fs::create_dir_all(&options.workspace).unwrap();
        let mut stored = PermissionsSection::default();
        stored
            .tools
            .insert("write_file".to_owned(), PermissionLevel::Allow);
        AllowRules::for_home(&options.home)
            .persist(&stored)
            .unwrap();

        let io = scripted_io(&[]);
        let model: Arc<dyn Model> = Arc::new(ScriptedModel::text("done"));
        let agent = build_agent_with(model, &options, &io, &FoundationMcpToolProvider)
            .await
            .unwrap();
        let config = agent.config();
        let permissions = config.permissions.unwrap();
        assert_eq!(
            permissions.tools.get("write_file"),
            Some(&PermissionLevel::Allow),
            "the stored allow must override the built-in ask"
        );

        let (rules, merged) = resolve_permissions(&options).unwrap();
        let host = permission_host(&io, &options, rules, &merged);
        let snapshot = host.permissions_snapshot().unwrap();
        assert_eq!(
            snapshot.tools.get("write_file"),
            Some(&PermissionLevel::Allow)
        );
        assert_eq!(
            host.permission_yaml_path(),
            Some(temp.path().join("permissions.yaml"))
        );
    }

    /// Extended tools [`CLI_EXTENDED_TOOLS`] deliberately leaves unmounted, with the reason the
    /// CLI cannot serve them; the mounted set is [`BuiltinTool::EXTENDED`] minus
    /// [`BuiltinTool::DEFAULT`] minus this list.
    const CLI_EXCLUDED_EXTENDED_TOOLS: [BuiltinTool; 2] = [
        // Needs a per-workspace language-server configuration the CLI does not resolve.
        BuiltinTool::Lsp,
        // Needs a subscription registry the CLI does not wire.
        BuiltinTool::McpResourceSubscription,
    ];

    /// The model-facing names of [`CLI_EXTENDED_TOOLS`], in registry order.
    fn cli_extended_names() -> Vec<&'static str> {
        CLI_EXTENDED_TOOLS.iter().map(|tool| tool.name()).collect()
    }

    #[test]
    fn cli_extended_tools_are_the_extended_set_minus_the_documented_exclusions() {
        let expected: Vec<BuiltinTool> = BuiltinTool::EXTENDED
            .into_iter()
            .filter(|tool| !BuiltinTool::DEFAULT.contains(tool))
            .filter(|tool| !CLI_EXCLUDED_EXTENDED_TOOLS.contains(tool))
            .collect();
        assert_eq!(
            CLI_EXTENDED_TOOLS.to_vec(),
            expected,
            "the CLI mounts every extended tool it can serve"
        );
        assert_eq!(
            cli_extended_names(),
            expected.iter().map(|tool| tool.name()).collect::<Vec<_>>(),
            "the mounted names must follow the registry order"
        );
        // Every extended tool is defaulted, mounted or a documented exclusion — nothing is
        // silently dropped when the registry grows.
        for tool in BuiltinTool::EXTENDED {
            assert!(
                BuiltinTool::DEFAULT.contains(&tool)
                    || CLI_EXTENDED_TOOLS.contains(&tool)
                    || CLI_EXCLUDED_EXTENDED_TOOLS.contains(&tool),
                "{tool:?} must be mounted, defaulted or documented as excluded"
            );
        }
        for excluded in CLI_EXCLUDED_EXTENDED_TOOLS {
            assert!(
                !CLI_EXTENDED_TOOLS.contains(&excluded),
                "{excluded:?} needs host wiring the CLI does not provide"
            );
        }
        assert!(
            CLI_EXTENDED_TOOLS
                .iter()
                .all(|tool| !BuiltinTool::DEFAULT.contains(tool)),
            "the builder already mounts the default twenty"
        );
    }

    #[tokio::test]
    async fn build_agent_mounts_the_extended_tools_and_skips_the_excluded_ones() {
        let temp = TempDir::new().unwrap();
        let options = test_options(temp.path());
        std::fs::create_dir_all(&options.workspace).unwrap();
        let io = scripted_io(&[]);
        let model = ScriptedModel::text("done");
        let scripted = model.clone();

        let agent = build_agent(Arc::new(model), &options, &io).await.unwrap();
        let output = agent
            .invoke(json!({"query": "hi"}), Some(new_session("s-tools")))
            .await
            .unwrap();
        assert_eq!(output.output, "done");

        // The tools section of the first model call lists every mounted card.
        let prompt = scripted.calls()[0]
            .messages
            .iter()
            .find(|message| message.role.is_system())
            .map(|message| message.text().into_owned())
            .expect("the first model call must carry the system prompt");
        for name in cli_extended_names() {
            assert!(
                prompt.contains(&format!("- **{name}**: ")),
                "the CLI agent must mount {name}:\n{prompt}"
            );
        }
        for name in crate::tools::tool_names() {
            assert!(
                prompt.contains(&format!("- **{name}**: ")),
                "the builder baseline must stay mounted ({name}):\n{prompt}"
            );
        }
        for excluded in CLI_EXCLUDED_EXTENDED_TOOLS {
            let name = excluded.name();
            assert!(
                !prompt.contains(&format!("- **{name}**: ")),
                "the CLI must not mount {name}:\n{prompt}"
            );
        }
    }

    #[tokio::test]
    async fn goal_tools_register_the_goal_rail() {
        let temp = TempDir::new().unwrap();
        let options = test_options(temp.path());
        std::fs::create_dir_all(&options.workspace).unwrap();
        let io = scripted_io(&[]);
        let model: Arc<dyn Model> = Arc::new(ScriptedModel::text("done"));

        let agent = build_agent(model, &options, &io).await.unwrap();
        assert!(
            agent.rails().names().iter().any(|name| name == "goal"),
            "mounting a goal tool must register the goal rail: {:?}",
            agent.rails().names()
        );
    }

    #[tokio::test]
    async fn web_search_run_without_a_backend_reports_the_typed_payload() {
        let temp = TempDir::new().unwrap();
        let options = test_options(temp.path());
        std::fs::create_dir_all(&options.workspace).unwrap();
        // The default section answers every tool with `ask`, and a one-shot run fails
        // closed; an interactive handle approves the call so the tool itself runs.
        let io = CliIo::with_interactive(
            Box::new(ScriptedReader {
                lines: VecDeque::from(["a".to_owned()]),
            }),
            Box::new(SilentWriter),
        );
        let model = ScriptedModel::builder()
            .completion(MockCompletion::tool_call(
                "web_search",
                json!({"query": "rust"}),
            ))
            .respond_text("searched")
            .build();
        let scripted = model.clone();
        let args = RunArgs {
            prompt: Some("find the rust docs".to_owned()),
            output_format: OutputFormat::Json,
        };

        let outcome = run_with_model(&options, &args, &io, Arc::new(model))
            .await
            .unwrap();
        assert_eq!(outcome.answer, "searched");
        let calls = scripted.calls();
        assert_eq!(calls.len(), 2, "the tool round must reach the model again");
        let result = calls[1]
            .messages
            .iter()
            .find(|message| message.role == Role::Tool)
            .map(|message| message.text().into_owned())
            .expect("the tool result must reach the second model call");
        assert!(
            result.contains("web search backend is not configured"),
            "the run must surface the documented payload, not panic: {result}"
        );
    }

    #[tokio::test]
    async fn mounted_online_tools_report_their_typed_not_configured_payloads() {
        let temp = TempDir::new().unwrap();
        let workspace = Workspace::for_language(temp.path(), DEFAULT_LANGUAGE);
        let tools = cli_extended_tools(&workspace, &PermissionsSection::default());
        let names: Vec<&str> = tools.iter().map(|tool| tool.card().name.as_str()).collect();
        assert_eq!(names, cli_extended_names());
        for excluded in CLI_EXCLUDED_EXTENDED_TOOLS {
            assert!(
                !names.contains(&excluded.name()),
                "the registry must not mount {}",
                excluded.name()
            );
        }

        let ctx = ToolContext::new();
        // `web_search` has no backend in the CLI: the documented payload, never a panic.
        let search = tools
            .iter()
            .find(|tool| tool.card().name == "web_search")
            .unwrap()
            .invoke(json!({"query": "rust"}), &ctx)
            .await
            .unwrap();
        assert_eq!(search["success"], json!(false));
        assert!(
            search["error"]
                .as_str()
                .unwrap_or_default()
                .contains("web search backend is not configured"),
            "{search}"
        );

        // `list_mcp_resources` has no server either.
        let resources = tools
            .iter()
            .find(|tool| tool.card().name == "list_mcp_resources")
            .unwrap()
            .invoke(json!({}), &ctx)
            .await
            .unwrap();
        assert_eq!(resources["success"], json!(false));
        assert!(
            resources["error"]
                .as_str()
                .unwrap_or_default()
                .contains("not configured"),
            "{resources}"
        );

        // `fetch_webpage` rejects a non-http(s) URL before any socket is opened.
        let fetch = tools
            .iter()
            .find(|tool| tool.card().name == "fetch_webpage")
            .unwrap()
            .invoke(json!({"url": "file:///etc/passwd"}), &ctx)
            .await
            .unwrap();
        assert_eq!(fetch["success"], json!(false));
        assert!(
            fetch["error"]
                .as_str()
                .unwrap_or_default()
                .contains("unsupported URL scheme"),
            "{fetch}"
        );
    }

    #[tokio::test]
    async fn worktree_session_tools_are_mounted_and_degrade_outside_a_repository() {
        let temp = TempDir::new().unwrap();
        let workspace = Workspace::for_language(temp.path(), DEFAULT_LANGUAGE);
        let tools = cli_extended_tools(&workspace, &PermissionsSection::default());
        let by_name = |name: &str| {
            tools
                .iter()
                .find(|tool| tool.card().name == name)
                .unwrap_or_else(|| panic!("the CLI must mount {name}"))
        };
        let ctx = ToolContext::new();

        // `exit_worktree` has no session recorded: the documented payload, no git call.
        let exit = by_name("exit_worktree")
            .invoke(json!({"action": "keep"}), &ctx)
            .await
            .unwrap();
        assert_eq!(exit["success"], json!(false));
        assert_eq!(exit["error"], json!("No active worktree session to exit."));

        // `enter_worktree` shares the registry's manager, rooted at the workspace: outside a git
        // repository the manager's typed git error becomes the payload.
        let enter = by_name("enter_worktree")
            .invoke(json!({"name": "cli-demo"}), &ctx)
            .await
            .unwrap();
        assert_eq!(enter["success"], json!(false));
        assert!(
            enter["error"]
                .as_str()
                .unwrap_or_default()
                .contains("Failed to create worktree"),
            "{enter}"
        );
    }
}
