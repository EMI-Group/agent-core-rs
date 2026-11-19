//! Interactive chat REPL.
//!
//! [`Repl`] loops over `You> ` prompts, runs one agent invocation per line, and
//! renders the stream through [`RenderState`] — the same renderer the one-shot
//! runner uses. Lines are parsed by [`Repl::handle_input`], which is a pure
//! function of the typed line plus the discovered skills, so command handling is
//! testable without a terminal.
//!
//! The command set mirrors the Python UI: `/help`, `/exit` (`/quit`), `/new`,
//! `/clear`, `/status`, `/cost`, `/compact`, `/sessions`, `/mode`, the `!cmd`
//! shell passthrough and `/<skill-name>` for every discovered skill. Each command
//! prints and returns; a failing command never ends the loop.
//!
//! Three pieces of state are wired from the injected [`CliOptions`]/[`CliIo`] in
//! [`Repl::build`], so `run_cli` needs no extra steps:
//!
//! - [`History`], the persistent input history at
//!   [`paths::history_path`]; every submitted line is recorded best-effort, and
//!   with [`paths::history_replay_enabled`] the stored entries are seeded into
//!   the reader at startup (the offline analogue of pre-filling a line editor's
//!   history buffer);
//! - [`UsageTotals`] and [`ConversationHistory`], the per-session counters behind
//!   `/cost`, `/status` and `/compact`;
//! - the discovered skills plus the shell runner behind `!cmd`.
//!
//! Interrupts are cooperative: a Ctrl-C watcher task (installed only for the
//! stdio REPL) calls [`DeepAgent::abort`](crate::agent::DeepAgent::abort) on the
//! first press and asks for exit on the second, so the loop never needs a
//! `select!` over the chunk stream. Cancellation surfaces as the stream ending,
//! exactly as it does for a non-interactive run.
//!
//! The banner prints the harness crate's version (`env!("CARGO_PKG_VERSION")`),
//! which is the version of the CLI logic the binary embeds.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use futures::StreamExt;
use openjiuwen_agent::Agent;
use openjiuwen_common::error::Result;
use openjiuwen_foundation::BaseMessage;
use openjiuwen_session::Session;
use parking_lot::Mutex;
use serde_json::json;
use tokio_util::sync::CancellationToken;

use super::compact::ConversationHistory;
use super::config::{CliOptions, ProcessEnv};
use super::history::History;
use super::io::CliIo;
use super::paths;
use super::render::RenderState;
use super::run::{new_session, session_id};
use super::session_store::MessageRecord;
use super::shell::{ProcessShellRunner, ShellRunner};
use super::skill_command;
use super::style::Style;
use super::usage::UsageTotals;
use crate::agent::DeepAgent;
use crate::schema::AgentMode;
use crate::skills::Skill;

/// Prompt shown before every user turn.
const PROMPT: &str = "You> ";

/// Message written when the REPL stops.
const GOODBYE: &str = "Goodbye!";

/// Message written when a turn was aborted by Ctrl-C.
const INTERRUPTED: &str = "⏹ Interrupted";

/// Usage line printed for a malformed `/mode` invocation.
const MODE_USAGE: &str = "Usage: /mode <normal|plan>";

/// Message printed by `/compact` when the transcript already fits.
const COMPACT_NOOP: &str = "Nothing to compact.";

/// Message printed by `/sessions` when no record exists yet.
const NO_SESSIONS: &str = "No saved sessions.";

/// Header of the `/sessions` table.
const SESSIONS_HEADER: &str = "ID  Model  Created  Turns";

/// Heading of the skill section appended to `/help`.
const SKILLS_HEADING: &str = "Skills (invoke with /<skill-name>):";

/// Fixed command table printed by `/help`.
const HELP_TABLE: &str = "\
Available commands:
  /help               Show this help
  /exit, /quit        Exit the REPL
  /new                Start a new session
  /clear              Reset the session, conversation history and counters
  /status             Show the model, provider, session id and token usage
  /cost               Show accumulated token and tool-call usage
  /compact            Compact the conversation history
  /sessions           List saved session records
  /mode <normal|plan> Switch the agent mode
  ! <cmd>             Run a shell command";

/// Session role of a user message in a persisted record.
const ROLE_USER: &str = "user";

/// Session role of an assistant message in a persisted record.
const ROLE_ASSISTANT: &str = "assistant";

/// What one typed line asks the REPL to do.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReplCommand {
    /// Run this text as a turn.
    Prompt(String),
    /// Leave the REPL.
    Exit,
    /// Start a new session.
    New,
    /// Reset the session, the conversation mirror and the counters.
    Clear,
    /// Switch to this agent mode.
    Mode(AgentMode),
    /// Print the command table.
    Help,
    /// Print the current model, provider, session and usage.
    Status,
    /// Print the accumulated token and call usage.
    Cost,
    /// Compact the conversation mirror and the persisted transcript.
    Compact,
    /// Print the saved session records.
    Sessions,
    /// Run this command line through the shell.
    Shell(String),
    /// Run the named skill with these arguments as a turn.
    Skill {
        /// Name of the skill, without the leading `/`.
        name: String,
        /// Argument text typed after the skill name.
        query: String,
    },
    /// Print this fixed usage line (malformed `/mode`).
    Usage(&'static str),
    /// Unrecognised slash command, carrying its command token.
    Unknown(String),
    /// A blank line; nothing happens.
    Empty,
}

/// Result of one streamed REPL turn.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TurnOutcome {
    /// Final answer of the turn.
    pub answer: String,
    /// Number of streamed chunks consumed.
    pub chunks: usize,
}

/// What an interrupt press asks the REPL to do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterruptAction {
    /// Abort the running turn and keep the REPL alive.
    Abort,
    /// Leave the REPL.
    Exit,
}

/// Counts Ctrl-C presses within one turn.
///
/// The first press aborts, the second and every later press asks for exit, which
/// matches the Python UI. The counter is reset at the start of each turn.
#[derive(Debug, Default)]
pub struct InterruptCounter {
    count: AtomicUsize,
}

impl InterruptCounter {
    /// Creates a counter with no recorded presses.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Forgets all recorded presses.
    pub fn reset(&self) {
        self.count.store(0, Ordering::SeqCst);
    }

    /// Records one press and returns what it asks for.
    pub fn press(&self) -> InterruptAction {
        if self.count.fetch_add(1, Ordering::SeqCst) == 0 {
            InterruptAction::Abort
        } else {
            InterruptAction::Exit
        }
    }
}

/// The interactive chat loop.
pub struct Repl {
    /// Agent every turn runs on.
    agent: Arc<DeepAgent>,
    /// Resolved CLI options (model, verbosity, session store).
    options: CliOptions,
    /// Terminal the REPL reads and writes.
    io: CliIo,
    /// Session reused across turns.
    session: Arc<dyn Session>,
    /// Id of [`Repl::session`].
    session_id: String,
    /// Ctrl-C presses of the current turn.
    interrupts: Arc<InterruptCounter>,
    /// Cancellation of the current turn (Ctrl-C), shared with the signal watcher.
    ///
    /// A `!cmd` command races this token, so an interrupt returns from the shell
    /// instead of waiting for the command. It is replaced at the start of every
    /// turn, exactly like [`Repl::interrupts`].
    cancel: Arc<Mutex<CancellationToken>>,
    /// Whether to install the Ctrl-C watcher (stdio REPL only).
    watch_signals: bool,
    /// Set by the watcher when a second press asks for exit.
    exited: Arc<AtomicBool>,
    /// Persistent input history of submitted lines.
    history: History,
    /// Whether to seed the reader with the stored history at startup.
    replay_history: bool,
    /// Token and call counters behind `/cost` and `/status`.
    usage: UsageTotals,
    /// CLI-side transcript mirror behind `/compact`.
    conversation: ConversationHistory,
    /// Skills discovered at startup; invoked as `/<name>`.
    skills: Vec<Skill>,
    /// Runner behind the `!cmd` passthrough.
    shell: Arc<dyn ShellRunner>,
    /// ANSI styling helper (plain unless the handle has colour on).
    style: Style,
}

impl Repl {
    /// Builds the terminal-backed REPL (the binary path).
    ///
    /// The session is new and the Ctrl-C watcher is enabled; use
    /// [`Repl::with_io_and_signals`] to share one stdio handle between the
    /// renderer and the permission host, or [`Repl::with_io`] to drive a
    /// scripted terminal instead.
    #[must_use]
    pub fn new(agent: Arc<DeepAgent>, options: CliOptions) -> Self {
        Self::with_io_and_signals(agent, options, CliIo::stdio())
    }

    /// Builds the terminal-backed REPL over an existing stdio handle, with the
    /// Ctrl-C watcher enabled (the binary path).
    ///
    /// The caller passes the same [`CliIo`] handle it gave the agent, so the
    /// renderer and the agent's permission host share one terminal.
    #[must_use]
    pub fn with_io_and_signals(agent: Arc<DeepAgent>, options: CliOptions, io: CliIo) -> Self {
        Self::build(agent, options, io, true)
    }

    /// Builds a REPL over an injected terminal, without a signal watcher.
    #[must_use]
    pub fn with_io(agent: Arc<DeepAgent>, options: CliOptions, io: CliIo) -> Self {
        Self::build(agent, options, io, false)
    }

    /// Replaces the input history (test seam).
    ///
    /// The history file location and capacity come from the injected value, so a
    /// test never touches the process home directory.
    #[must_use]
    pub fn with_history(mut self, history: History) -> Self {
        self.history = history;
        self
    }

    /// Overrides whether the stored history is replayed at startup (test seam).
    #[must_use]
    pub fn with_history_replay(mut self, replay: bool) -> Self {
        self.replay_history = replay;
        self
    }

    /// Replaces the shell runner behind `!cmd` (test seam).
    #[must_use]
    pub fn with_shell(mut self, runner: Arc<dyn ShellRunner>) -> Self {
        self.shell = runner;
        self
    }

    /// Replaces the startup skill discovery result (test seam).
    ///
    /// A non-empty injected list is authoritative: [`Repl::run`] then skips
    /// discovery, so a test can pin the skill set without touching the workspace.
    #[must_use]
    pub fn with_skills(mut self, skills: Vec<Skill>) -> Self {
        self.skills = skills;
        self
    }

    /// Shared constructor of both entry points.
    fn build(agent: Arc<DeepAgent>, options: CliOptions, io: CliIo, watch_signals: bool) -> Self {
        let id = session_id();
        let env = ProcessEnv;
        let history = History::new(paths::history_path(&options.home, &env));
        let replay_history = paths::history_replay_enabled(&env);
        let style = Style::new(io.is_color());
        Self {
            agent,
            options,
            io,
            session: new_session(&id),
            session_id: id,
            interrupts: Arc::new(InterruptCounter::new()),
            cancel: Arc::new(Mutex::new(CancellationToken::new())),
            watch_signals,
            exited: Arc::new(AtomicBool::new(false)),
            history,
            replay_history,
            usage: UsageTotals::new(),
            conversation: ConversationHistory::new(),
            skills: Vec::new(),
            shell: Arc::new(ProcessShellRunner::new()),
            style,
        }
    }

    /// Returns a handle that is cancelled when the current turn is interrupted.
    ///
    /// It is the token a `!cmd` runner races, exposed so a host (or a test) can
    /// observe the interrupt without driving a real terminal. The REPL replaces
    /// the token at the start of every turn, so a handle taken now goes stale —
    /// which is exactly what makes one Ctrl-C stop one turn.
    #[must_use]
    pub fn turn_cancellation(&self) -> CancellationToken {
        self.cancel.lock().clone()
    }

    /// Returns the id of the current session.
    #[must_use]
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Adopts a persisted session id instead of minting a new one.
    ///
    /// The kernel session is rebuilt around `id`, so every following
    /// [`Repl::run_turn`] continues that kernel session and the REPL record
    /// [`Repl::run_turn`] maintains is the existing `<home>/sessions/<id>.json`
    /// — appended to, never truncated (unlike `Repl::start_new_session`). The
    /// caller owns the transcript: seed it from a
    /// [`ResumePlan`](crate::persistent_agent::ResumePlan) with
    /// [`Repl::seed_history`] *before* the first turn, after any
    /// [`DeepAgent::configure`](crate::agent::DeepAgent::configure).
    ///
    /// Nothing is created on disk here: adopting an id whose record does not
    /// exist simply starts a new transcript under that id.
    #[must_use]
    pub fn with_session_id(mut self, id: impl Into<String>) -> Self {
        self.session_id = id.into();
        self.session = new_session(&self.session_id);
        self
    }

    /// Runs the REPL until the user exits or the input ends.
    ///
    /// A failed turn is reported on the terminal and never ends the loop; only
    /// end of input, `/exit`, or a second Ctrl-C stop it.
    ///
    /// # Errors
    ///
    /// Returns the terminal error when a banner, prompt, render or command write
    /// fails, and whatever [`DeepAgent::switch_mode`] reports.
    pub async fn run(&mut self) -> Result<()> {
        self.load_history();
        self.discover_skills().await;
        self.io
            .write_line(
                &self
                    .style
                    .bold_cyan(&format!("OpenJiuwen CLI v{}", env!("CARGO_PKG_VERSION"))),
            )
            .await?;
        if self.watch_signals {
            self.spawn_signal_watcher();
        }
        self.replay_input_history().await;
        loop {
            if self.exited.load(Ordering::SeqCst) {
                break;
            }
            let Some(line) = self.io.prompt_line(PROMPT).await? else {
                break;
            };
            self.interrupts.reset();
            *self.cancel.lock() = CancellationToken::new();
            self.remember(&line);
            match self.handle_input(&line) {
                ReplCommand::Empty => {}
                ReplCommand::Exit => {
                    self.io.write_line(GOODBYE).await?;
                    break;
                }
                ReplCommand::New => self.start_new_session().await?,
                ReplCommand::Clear => self.clear_session().await?,
                ReplCommand::Help => self.write_help().await?,
                ReplCommand::Status => self.write_status().await?,
                ReplCommand::Cost => self.io.write_line(&self.usage.render()).await?,
                ReplCommand::Compact => self.compact_conversation().await?,
                ReplCommand::Sessions => self.write_sessions().await?,
                ReplCommand::Mode(mode) => {
                    self.agent.switch_mode(mode)?;
                    self.io
                        .write_line(&format!("Mode: {}", mode.as_str()))
                        .await?;
                }
                ReplCommand::Usage(text) => self.io.write_line(text).await?,
                ReplCommand::Shell(command) => self.run_shell(&command).await?,
                ReplCommand::Skill { name, query } => self.run_skill(&name, &query).await?,
                ReplCommand::Unknown(command) => {
                    self.io
                        .write_line(&format!("Unknown command: {command}"))
                        .await?;
                    self.io
                        .write_line("Type /help to see available commands.")
                        .await?;
                }
                ReplCommand::Prompt(text) => {
                    if let Err(error) = self.run_turn(&text).await {
                        self.io.write_line(&format!("✗ Error: {error}")).await?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Parses one typed line into the command it asks for.
    ///
    /// Parsing is pure and ordered:
    ///
    /// 1. a blank line is [`ReplCommand::Empty`];
    /// 2. a line starting with `!` is a [`ReplCommand::Shell`] of the rest, or
    ///    [`ReplCommand::Empty`] when the rest is blank;
    /// 3. a line starting with `/` is matched, case-insensitively, against the
    ///    built-in commands; an unmatched name is then offered to
    ///    [`skill_command::parse_skill_command`], and only when that fails is it
    ///    [`ReplCommand::Unknown`] carrying the token as typed;
    /// 4. anything else is a [`ReplCommand::Prompt`].
    #[must_use]
    pub fn handle_input(&self, line: &str) -> ReplCommand {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            return ReplCommand::Empty;
        }
        if let Some(rest) = trimmed.strip_prefix('!') {
            let command = rest.trim();
            return if command.is_empty() {
                ReplCommand::Empty
            } else {
                ReplCommand::Shell(command.to_owned())
            };
        }
        if !trimmed.starts_with('/') {
            return ReplCommand::Prompt(trimmed.to_owned());
        }
        let mut parts = trimmed.split_whitespace();
        let Some(token) = parts.next() else {
            return ReplCommand::Empty;
        };
        match token.to_ascii_lowercase().as_str() {
            "/exit" | "/quit" => ReplCommand::Exit,
            "/new" => ReplCommand::New,
            "/clear" => ReplCommand::Clear,
            "/help" => ReplCommand::Help,
            "/status" => ReplCommand::Status,
            "/cost" => ReplCommand::Cost,
            "/compact" => ReplCommand::Compact,
            "/sessions" => ReplCommand::Sessions,
            "/mode" => match (parts.next(), parts.next()) {
                (Some(name), None) => {
                    AgentMode::parse(name).map_or(ReplCommand::Usage(MODE_USAGE), ReplCommand::Mode)
                }
                _ => ReplCommand::Usage(MODE_USAGE),
            },
            _ => match skill_command::parse_skill_command(trimmed, &self.skills) {
                Some((name, query)) => ReplCommand::Skill { name, query },
                None => ReplCommand::Unknown(token.to_owned()),
            },
        }
    }

    /// Records one interrupt press, aborting the running turn on the first.
    ///
    /// Called by the signal watcher; tests drive it directly. Calling it twice
    /// aborts once and then asks for exit, which is the Python behaviour.
    #[must_use]
    pub fn on_interrupt(&self) -> InterruptAction {
        press_interrupt(&self.interrupts, &self.agent, &self.cancel)
    }

    /// Runs one prompt as a streamed turn.
    ///
    /// Every streamed chunk is folded into [`UsageTotals`], the turn counter is
    /// bumped once the stream ends, and the user plus assistant messages join the
    /// [`ConversationHistory`] mirror used by `/compact`.
    ///
    /// # Errors
    ///
    /// Returns the stream error of the invocation and the terminal error of a
    /// failed render. Session-write failures are logged, never returned.
    pub async fn run_turn(&mut self, input: &str) -> Result<TurnOutcome> {
        let mut render = RenderState::styled(self.io.is_color());
        let mut chunks = 0_usize;
        let mut stream = self
            .agent
            .stream(json!({"query": input}), Some(Arc::clone(&self.session)));
        while let Some(item) = stream.next().await {
            let chunk = item?;
            chunks += 1;
            self.usage.record(&chunk);
            render
                .render(&chunk, Some(&self.io), self.options.verbose)
                .await?;
        }
        render.finish(Some(&self.io)).await?;
        self.usage.record_turn();

        let outcome = TurnOutcome {
            answer: render.answer().to_owned(),
            chunks,
        };
        self.conversation.push(ROLE_USER, input);
        self.conversation.push(ROLE_ASSISTANT, &outcome.answer);

        let store = self.options.sessions();
        let turn = vec![
            MessageRecord::new(ROLE_USER, input),
            MessageRecord::new(ROLE_ASSISTANT, &outcome.answer),
        ];
        if let Err(error) = store.append_turn(&self.session_id, turn).await {
            tracing::warn!(%error, "failed to persist the REPL turn");
        }
        Ok(outcome)
    }

    /// Seeds the conversation of the current session from a persisted history.
    ///
    /// This is the REPL side of resuming: the messages are pushed into the
    /// kernel `"main"` context of the REPL's session, so every following
    /// [`Repl::run_turn`] continues that same kernel session and the model sees
    /// the restored transcript ahead of the new query. The user and assistant
    /// texts are also mirrored into the [`ConversationHistory`] behind
    /// `/compact` and `/cost`, so those commands reason over the same transcript
    /// the model sees; system and tool messages reach the kernel only.
    ///
    /// Call it **after** the REPL is built ([`Repl::new`], [`Repl::with_io`] or
    /// [`Repl::with_io_and_signals`], and after any
    /// [`DeepAgent::configure`](crate::agent::DeepAgent::configure) — that
    /// re-applies the kernel configuration and drops pooled contexts) and
    /// **before** the first [`Repl::run_turn`]/[`Repl::run`]. Seeding the same
    /// history twice yields one copy, so a repeated identical seed is harmless.
    ///
    /// # Errors
    ///
    /// Returns whatever the kernel context seed reports.
    pub async fn seed_history(&mut self, history: Vec<BaseMessage>) -> Result<()> {
        self.agent
            .seed_history(&self.session, history.clone())
            .await?;
        for message in &history {
            let role = if message.role.is_user() {
                ROLE_USER
            } else if message.role.is_assistant() {
                ROLE_ASSISTANT
            } else {
                continue;
            };
            let text = message.text();
            if text.trim().is_empty() {
                continue;
            }
            self.conversation.push(role, text.as_ref());
        }
        Ok(())
    }

    /// Loads the persistent input history, logging and continuing on failure.
    fn load_history(&mut self) {
        if let Err(error) = self.history.load() {
            tracing::warn!(%error, "failed to load the CLI history file");
        }
    }

    /// Seeds the reader with the loaded history when replay is enabled.
    ///
    /// This is the offline analogue of pre-filling a line editor's history
    /// buffer: the stored entries are queued ahead of the real input, so the
    /// session begins by re-running what the user typed in earlier sessions.
    async fn replay_input_history(&self) {
        if !self.replay_history || self.history.is_empty() {
            return;
        }
        self.io
            .seed_lines(self.history.entries().iter().cloned())
            .await;
    }

    /// Discovers the skills exposed as `/<name>` commands.
    ///
    /// A discovery failure is logged and leaves the REPL with no skills, because
    /// a broken skill directory must never stop the CLI. A non-empty skill list
    /// injected through [`Repl::with_skills`] is authoritative and skips
    /// discovery.
    async fn discover_skills(&mut self) {
        if !self.skills.is_empty() {
            return;
        }
        let roots = skill_command::default_skill_roots(&self.options.workspace, &self.options.home);
        match skill_command::load_skills(&roots).await {
            Ok(skills) => self.skills = skills,
            Err(error) => tracing::warn!(%error, "failed to discover CLI skills"),
        }
    }

    /// Records one submitted line in the history, logging and continuing on
    /// failure (history is a convenience, never a precondition).
    fn remember(&mut self, line: &str) {
        if line.trim().is_empty() {
            return;
        }
        if let Err(error) = self.history.record(line) {
            tracing::warn!(%error, "failed to record REPL history entry");
        }
    }

    /// Replaces the session, clearing the agent state of the previous one.
    async fn start_new_session(&mut self) -> Result<()> {
        let id = session_id();
        self.session = new_session(&id);
        self.session_id = id;
        if let Err(error) = self.agent.clear_state().await {
            tracing::warn!(%error, "failed to clear agent state");
        }
        self.io
            .write_line(&format!("Started a new session: {}", self.session_id))
            .await
    }

    /// Resets the session, the conversation mirror and the usage counters.
    ///
    /// `/clear` is deliberately stronger than `/new`: it also drops the turns
    /// `/compact` and `/cost` reason about, so the next turn starts from an empty
    /// transcript.
    async fn clear_session(&mut self) -> Result<()> {
        let id = session_id();
        self.session = new_session(&id);
        self.session_id = id;
        if let Err(error) = self.agent.clear_state().await {
            tracing::warn!(%error, "failed to clear agent state");
        }
        self.conversation.clear();
        self.usage.reset();
        self.io
            .write_line(&format!(
                "Cleared conversation and started a new session: {}",
                self.session_id
            ))
            .await
    }

    /// Compacts the conversation mirror and rewrites the persisted transcript.
    async fn compact_conversation(&mut self) -> Result<()> {
        let report = self.conversation.compact();
        if !report.dropped_any() {
            return self.io.write_line(COMPACT_NOOP).await;
        }
        let timestamp = chrono::Utc::now().to_rfc3339();
        let messages = self
            .conversation
            .messages()
            .iter()
            .map(|message| MessageRecord {
                role: message.role.clone(),
                content: message.content.clone(),
                timestamp: timestamp.clone(),
            })
            .collect::<Vec<_>>();
        if let Err(error) = self
            .options
            .sessions()
            .replace_messages_async(&self.session_id, messages)
            .await
        {
            tracing::warn!(%error, "failed to persist the compacted transcript");
        }
        self.io
            .write_line(&format!(
                "Compacted conversation: {} message(s) dropped, {} retained.",
                report.dropped, report.after
            ))
            .await
    }

    /// Runs one `!cmd` line through the shell, bypassing the model.
    ///
    /// The command races the turn's cancellation token, so a Ctrl-C kills the
    /// command's process group and returns to the prompt instead of wedging the
    /// REPL on a command that never ends.
    async fn run_shell(&self, command: &str) -> Result<()> {
        let cancel = self.cancel.lock().clone();
        match self
            .shell
            .run_cancellable(command, &self.options.workspace, &cancel)
            .await
        {
            Ok(output) => {
                if !output.stdout.is_empty() {
                    self.io.write(&output.stdout).await?;
                }
                if !output.stderr.is_empty() {
                    self.io.write(&self.style.red(&output.stderr)).await?;
                }
                if let Some(status) = output.status_line() {
                    self.io.write_line(&status).await?;
                }
                Ok(())
            }
            Err(error) => self.io.write_line(&format!("✗ Error: {error}")).await,
        }
    }

    /// Runs one `/<skill-name>` line as a normal turn.
    ///
    /// The wrapped prompt (the whole `SKILL.md` plus the typed request) is what
    /// gets streamed and persisted, matching Python.
    async fn run_skill(&mut self, name: &str, query: &str) -> Result<()> {
        let rendered = skill_command::render_skill_prompt(&self.skills, name, query).await;
        match rendered {
            Ok(Some(prompt)) => {
                if let Err(error) = self.run_turn(&prompt).await {
                    self.io.write_line(&format!("✗ Error: {error}")).await?;
                }
                Ok(())
            }
            Ok(None) => self.io.write_line(&format!("Unknown skill: {name}")).await,
            Err(error) => self.io.write_line(&format!("✗ Error: {error}")).await,
        }
    }

    /// Writes the `/help` table plus the skill section when skills exist.
    async fn write_help(&self) -> Result<()> {
        let mut text = HELP_TABLE.to_owned();
        if !self.skills.is_empty() {
            text.push('\n');
            text.push_str(SKILLS_HEADING);
            for row in skill_command::help_rows(&self.skills) {
                text.push('\n');
                text.push_str(&row);
            }
        }
        self.io.write_line(&text).await
    }

    /// Writes the `/sessions` table.
    async fn write_sessions(&self) -> Result<()> {
        let sessions = match self.options.sessions().list() {
            Ok(sessions) => sessions,
            Err(error) => return self.io.write_line(&format!("✗ Error: {error}")).await,
        };
        if sessions.is_empty() {
            return self.io.write_line(NO_SESSIONS).await;
        }
        self.io.write_line(SESSIONS_HEADER).await?;
        for session in sessions {
            self.io
                .write_line(&format!(
                    "{}  {}  {}  {}",
                    session.session_id, session.model, session.created_at, session.messages
                ))
                .await?;
        }
        Ok(())
    }

    /// Writes the `/status` block.
    async fn write_status(&self) -> Result<()> {
        self.io
            .write_line(&format!("Model: {}", self.options.model))
            .await?;
        self.io
            .write_line(&format!("Provider: {}", self.options.provider))
            .await?;
        self.io
            .write_line(&format!("Session: {}", self.session_id))
            .await?;
        self.io.write_line(&self.usage.render_line()).await
    }

    /// Spawns the Ctrl-C watcher that aborts the running turn.
    ///
    /// The task owns clones of what it needs, so the REPL keeps its exclusive
    /// borrow and no lock is shared across the stream.
    fn spawn_signal_watcher(&self) {
        let interrupts = Arc::clone(&self.interrupts);
        let cancel = Arc::clone(&self.cancel);
        let agent = Arc::clone(&self.agent);
        let io = self.io.clone();
        let exited = Arc::clone(&self.exited);
        tokio::spawn(async move {
            loop {
                if tokio::signal::ctrl_c().await.is_err() {
                    return;
                }
                match press_interrupt(&interrupts, &agent, &cancel) {
                    InterruptAction::Abort => {
                        if io.write_line(&format!("\n{INTERRUPTED}")).await.is_err() {
                            return;
                        }
                    }
                    InterruptAction::Exit => {
                        exited.store(true, Ordering::SeqCst);
                        return;
                    }
                }
            }
        });
    }
}

/// Records one press, aborts `agent` and cancels the turn's shell command.
///
/// Shared by [`Repl::on_interrupt`] and the signal watcher so both paths behave
/// identically without the watcher needing a borrow of the REPL: aborting the
/// agent cannot reach a `!cmd` await, so the same press also cancels the token the
/// shell runner races.
fn press_interrupt(
    interrupts: &InterruptCounter,
    agent: &DeepAgent,
    cancel: &Mutex<CancellationToken>,
) -> InterruptAction {
    let action = interrupts.press();
    if action == InterruptAction::Abort {
        agent.abort();
        cancel.lock().cancel();
    }
    action
}
