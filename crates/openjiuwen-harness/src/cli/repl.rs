//! Interactive chat REPL.
//!
//! [`Repl`] loops over `You> ` prompts, runs one agent invocation per line, and
//! renders the stream through [`RenderState`] — the same renderer the one-shot
//! runner uses. Slash commands are parsed by [`Repl::handle_input`], which is a
//! pure function of the typed line, so command handling is testable without a
//! terminal.
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
use openjiuwen_session::Session;
use serde_json::json;

use super::config::CliOptions;
use super::io::CliIo;
use super::render::RenderState;
use super::run::{new_session, session_id};
use crate::agent::DeepAgent;
use crate::schema::AgentMode;

/// Prompt shown before every user turn.
const PROMPT: &str = "You> ";

/// Message written when the REPL stops.
const GOODBYE: &str = "Goodbye!";

/// Message written when a turn was aborted by Ctrl-C.
const INTERRUPTED: &str = "⏹ Interrupted";

/// Usage line printed for a malformed `/mode` invocation.
const MODE_USAGE: &str = "Usage: /mode <normal|plan>";

/// Fixed command table printed by `/help`.
const HELP_TABLE: &str = "\
Available commands:
  /help               Show this help
  /exit, /quit        Exit the REPL
  /new                Start a new session
  /mode <normal|plan> Switch the agent mode
  /status             Show the model, provider and session id";

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
    /// Switch to this agent mode.
    Mode(AgentMode),
    /// Print the command table.
    Help,
    /// Print the current model, provider and session.
    Status,
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
    /// Whether to install the Ctrl-C watcher (stdio REPL only).
    watch_signals: bool,
    /// Set by the watcher when a second press asks for exit.
    exited: Arc<AtomicBool>,
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

    /// Shared constructor of both entry points.
    fn build(agent: Arc<DeepAgent>, options: CliOptions, io: CliIo, watch_signals: bool) -> Self {
        let id = session_id();
        Self {
            agent,
            options,
            io,
            session: new_session(&id),
            session_id: id,
            interrupts: Arc::new(InterruptCounter::new()),
            watch_signals,
            exited: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Returns the id of the current session.
    #[must_use]
    pub fn session_id(&self) -> &str {
        &self.session_id
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
        self.io
            .write_line(&format!("OpenJiuwen CLI v{}", env!("CARGO_PKG_VERSION")))
            .await?;
        if self.watch_signals {
            self.spawn_signal_watcher();
        }
        loop {
            if self.exited.load(Ordering::SeqCst) {
                break;
            }
            let Some(line) = self.io.prompt_line(PROMPT).await? else {
                break;
            };
            self.interrupts.reset();
            match self.handle_input(&line) {
                ReplCommand::Empty => {}
                ReplCommand::Exit => {
                    self.io.write_line(GOODBYE).await?;
                    break;
                }
                ReplCommand::New => self.start_new_session().await?,
                ReplCommand::Help => self.io.write_line(HELP_TABLE).await?,
                ReplCommand::Status => self.write_status().await?,
                ReplCommand::Mode(mode) => {
                    self.agent.switch_mode(mode)?;
                    self.io
                        .write_line(&format!("Mode: {}", mode.as_str()))
                        .await?;
                }
                ReplCommand::Usage(text) => self.io.write_line(text).await?,
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
    /// The parser is pure: unknown slash commands become
    /// [`ReplCommand::Unknown`] carrying the command token only, and anything not
    /// starting with `/` is a prompt.
    #[allow(clippy::unused_self)] // kept a method: command handling belongs to the REPL
    #[must_use]
    pub fn handle_input(&self, line: &str) -> ReplCommand {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            return ReplCommand::Empty;
        }
        if !trimmed.starts_with('/') {
            return ReplCommand::Prompt(trimmed.to_owned());
        }
        let mut parts = trimmed.split_whitespace();
        let Some(command) = parts.next() else {
            return ReplCommand::Empty;
        };
        match command {
            "/exit" | "/quit" => ReplCommand::Exit,
            "/new" => ReplCommand::New,
            "/help" => ReplCommand::Help,
            "/status" => ReplCommand::Status,
            "/mode" => match (parts.next(), parts.next()) {
                (Some(name), None) => {
                    AgentMode::parse(name).map_or(ReplCommand::Usage(MODE_USAGE), ReplCommand::Mode)
                }
                _ => ReplCommand::Usage(MODE_USAGE),
            },
            other => ReplCommand::Unknown(other.to_owned()),
        }
    }

    /// Records one interrupt press, aborting the running turn on the first.
    ///
    /// Called by the signal watcher; tests drive it directly. Calling it twice
    /// aborts once and then asks for exit, which is the Python behaviour.
    #[must_use]
    pub fn on_interrupt(&self) -> InterruptAction {
        press_interrupt(&self.interrupts, &self.agent)
    }

    /// Runs one prompt as a streamed turn.
    ///
    /// # Errors
    ///
    /// Returns the stream error of the invocation and the terminal error of a
    /// failed render. Session-write failures are logged, never returned.
    pub async fn run_turn(&mut self, input: &str) -> Result<TurnOutcome> {
        let mut render = RenderState::new();
        let mut chunks = 0_usize;
        let mut stream = self
            .agent
            .stream(json!({"query": input}), Some(Arc::clone(&self.session)));
        while let Some(item) = stream.next().await {
            let chunk = item?;
            chunks += 1;
            render
                .render(&chunk, Some(&self.io), self.options.verbose)
                .await?;
        }
        render.finish(Some(&self.io)).await?;

        let outcome = TurnOutcome {
            answer: render.answer().to_owned(),
            chunks,
        };
        let store = self.options.sessions();
        if let Err(error) = store.append_message(&self.session_id, ROLE_USER, input) {
            tracing::warn!(%error, "failed to persist REPL user message");
        }
        if let Err(error) = store.append_message(&self.session_id, ROLE_ASSISTANT, &outcome.answer)
        {
            tracing::warn!(%error, "failed to persist REPL assistant message");
        }
        Ok(outcome)
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
            .await
    }

    /// Spawns the Ctrl-C watcher that aborts the running turn.
    ///
    /// The task owns clones of what it needs, so the REPL keeps its exclusive
    /// borrow and no lock is shared across the stream.
    fn spawn_signal_watcher(&self) {
        let interrupts = Arc::clone(&self.interrupts);
        let agent = Arc::clone(&self.agent);
        let io = self.io.clone();
        let exited = Arc::clone(&self.exited);
        tokio::spawn(async move {
            loop {
                if tokio::signal::ctrl_c().await.is_err() {
                    return;
                }
                match press_interrupt(&interrupts, &agent) {
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

/// Records one press and aborts `agent` when it asks for an abort.
///
/// Shared by [`Repl::on_interrupt`] and the signal watcher so both paths behave
/// identically without the watcher needing a borrow of the REPL.
fn press_interrupt(interrupts: &InterruptCounter, agent: &DeepAgent) -> InterruptAction {
    let action = interrupts.press();
    if action == InterruptAction::Abort {
        agent.abort();
    }
    action
}
