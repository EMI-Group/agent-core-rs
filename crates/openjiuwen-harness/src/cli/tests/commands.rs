//! Phase-3 REPL tests: slash commands, history, shell passthrough and skills.
//!
//! Everything here runs offline against the shared harness of [`super`]: a
//! scripted [`CliIo`], a scripted model, and a temporary home/workspace. No test
//! reads the process environment (history and shell are injected through
//! [`Repl::with_history`], [`Repl::with_history_replay`], [`Repl::with_shell`] and
//! [`Repl::with_skills`]) and no test spawns a real process or sleeps.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::Path;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::llm::Model;
use openjiuwen_foundation::Usage;
use openjiuwen_test_support::{MockCompletion, ScriptedModel};
use tempfile::TempDir;
use tokio_util::sync::CancellationToken;

use crate::cli::history::History;
use crate::cli::shell::{ShellOutput, ShellRunner};
use crate::schema::AgentMode;
use crate::skills::{Skill, SKILL_FILENAME};

use super::*;

/// Every line the `/help` table must print, in order.
const HELP_LINES: &[&str] = &[
    "Available commands:",
    "  /help               Show this help",
    "  /exit, /quit        Exit the REPL",
    "  /new                Start a new session",
    "  /clear              Reset the session, conversation history and counters",
    "  /status             Show the model, provider, session id and token usage",
    "  /cost               Show accumulated token and tool-call usage",
    "  /compact            Compact the conversation history",
    "  /sessions           List saved session records",
    "  /mode <normal|plan> Switch the agent mode",
    "  ! <cmd>             Run a shell command",
];

/// Usage line printed for a malformed `/mode` invocation (parity with the REPL).
const MODE_USAGE_LINE: &str = "Usage: /mode <normal|plan>";

/// What a [`FakeShell`] answers a command with.
enum FakeReply {
    /// A captured output.
    Output(ShellOutput),
    /// A runner failure with this message.
    Error(String),
}

/// [`ShellRunner`] that records commands and answers from a fixed reply.
struct FakeShell {
    /// `(command, cwd)` pairs in call order.
    calls: Mutex<Vec<(String, String)>>,
    /// What every call returns.
    reply: FakeReply,
    /// Whether the cancellation token was already cancelled on each call.
    cancellations: Mutex<Vec<bool>>,
}

impl FakeShell {
    /// Builds a runner answering `reply`.
    fn new(reply: FakeReply) -> Arc<Self> {
        Arc::new(Self {
            calls: Mutex::new(Vec::new()),
            reply,
            cancellations: Mutex::new(Vec::new()),
        })
    }

    /// Returns the recorded `(command, cwd)` pairs.
    fn calls(&self) -> Vec<(String, String)> {
        self.calls.lock().unwrap().clone()
    }

    /// Returns whether the cancellation token was already cancelled on each call.
    fn cancellations(&self) -> Vec<bool> {
        self.cancellations.lock().unwrap().clone()
    }
}

#[async_trait]
impl ShellRunner for FakeShell {
    async fn run_cancellable(
        &self,
        command: &str,
        cwd: &Path,
        cancel: &CancellationToken,
    ) -> Result<ShellOutput> {
        self.cancellations
            .lock()
            .unwrap()
            .push(cancel.is_cancelled());
        self.calls
            .lock()
            .unwrap()
            .push((command.to_owned(), cwd.display().to_string()));
        match &self.reply {
            FakeReply::Output(output) => Ok(output.clone()),
            FakeReply::Error(message) => {
                Err(Error::external(std::io::Error::other(message.clone())))
            }
        }
    }
}

/// Builds a REPL over an explicit terminal and model.
async fn repl_over(options: &CliOptions, io: CliIo, model: Arc<dyn Model>) -> Repl {
    let agent = build_agent(model, options, &io).await.unwrap();
    Repl::with_io(agent, options.clone(), io)
}

/// Builds a REPL over a scripted reader plus the returned output buffer.
async fn scripted_repl(
    options: &CliOptions,
    lines: &[&str],
    model: Arc<dyn Model>,
) -> (Repl, Arc<Mutex<String>>) {
    let (io, buffer) = scripted_io(lines);
    (repl_over(options, io, model).await, buffer)
}

/// Writes `<workspace>/skills/<name>/SKILL.md` so discovery finds the skill.
fn write_skill(workspace: &Path, name: &str, description: &str, body: &str) {
    let directory = workspace.join("skills").join(name);
    std::fs::create_dir_all(&directory).unwrap();
    std::fs::write(
        directory.join(SKILL_FILENAME),
        format!("---\nname: {name}\ndescription: {description}\n---\n{body}"),
    )
    .unwrap();
}

/// Returns the workspace-rooted options of a temporary CLI home.
fn prepared_options(temp: &TempDir) -> CliOptions {
    let options = test_options(temp.path());
    options.ensure_workspace().unwrap();
    options
}

// --- handle_input ---------------------------------------------------------

#[tokio::test]
async fn repl_handle_input_covers_every_builtin() {
    let temp = TempDir::new().unwrap();
    let options = prepared_options(&temp);
    let (repl, _buffer) =
        scripted_repl(&options, &[], Arc::new(ScriptedModel::text("unused"))).await;

    for (line, expected) in [
        ("/help", ReplCommand::Help),
        ("/exit", ReplCommand::Exit),
        ("/quit", ReplCommand::Exit),
        ("/EXIT", ReplCommand::Exit),
        ("/new", ReplCommand::New),
        ("/clear", ReplCommand::Clear),
        ("/status", ReplCommand::Status),
        ("/cost", ReplCommand::Cost),
        ("/compact", ReplCommand::Compact),
        ("/sessions", ReplCommand::Sessions),
        ("/mode plan", ReplCommand::Mode(AgentMode::Plan)),
        ("/mode normal", ReplCommand::Mode(AgentMode::Normal)),
        ("/mode", ReplCommand::Usage(MODE_USAGE_LINE)),
        ("/mode sideways", ReplCommand::Usage(MODE_USAGE_LINE)),
        ("/mode plan extra", ReplCommand::Usage(MODE_USAGE_LINE)),
        ("", ReplCommand::Empty),
        ("   ", ReplCommand::Empty),
        ("hello there", ReplCommand::Prompt("hello there".to_owned())),
        (
            "  padded prompt  ",
            ReplCommand::Prompt("padded prompt".to_owned()),
        ),
        ("!echo hi", ReplCommand::Shell("echo hi".to_owned())),
        ("! ls -l ", ReplCommand::Shell("ls -l".to_owned())),
        ("!", ReplCommand::Empty),
        ("!   ", ReplCommand::Empty),
        ("/bogus", ReplCommand::Unknown("/bogus".to_owned())),
        ("/Bogus", ReplCommand::Unknown("/Bogus".to_owned())),
    ] {
        assert_eq!(repl.handle_input(line), expected, "line {line:?}");
    }
}

#[tokio::test]
async fn repl_handle_input_maps_known_skills_and_rejects_unknown_ones() {
    let temp = TempDir::new().unwrap();
    let options = prepared_options(&temp);
    let directory = options.workspace.join("skills").join("alpha");
    let skills = vec![Skill::new("alpha", "First skill", &directory)];
    let (repl, _buffer) =
        scripted_repl(&options, &[], Arc::new(ScriptedModel::text("unused"))).await;
    let repl = repl.with_skills(skills);

    assert_eq!(
        repl.handle_input("/alpha"),
        ReplCommand::Skill {
            name: "alpha".to_owned(),
            query: String::new(),
        }
    );
    assert_eq!(
        repl.handle_input("/alpha do it now"),
        ReplCommand::Skill {
            name: "alpha".to_owned(),
            query: "do it now".to_owned(),
        }
    );
    // Built-in names are resolved before the skill table, and a name no skill
    // declares stays unknown.
    assert_eq!(repl.handle_input("/new"), ReplCommand::New);
    assert_eq!(
        repl.handle_input("/beta"),
        ReplCommand::Unknown("/beta".to_owned())
    );
}

// --- /help ---------------------------------------------------------------

#[tokio::test]
async fn repl_help_lists_every_command() {
    let temp = TempDir::new().unwrap();
    let options = prepared_options(&temp);
    let (mut repl, buffer) = scripted_repl(
        &options,
        &["/help", "/exit"],
        Arc::new(ScriptedModel::text("unused")),
    )
    .await;
    repl.run().await.unwrap();

    let text = output(&buffer);
    for line in HELP_LINES {
        assert!(text.contains(line), "missing {line:?} in {text}");
    }
    assert!(!text.contains("Skills (invoke with /<skill-name>):"));
}

#[tokio::test]
async fn repl_help_lists_discovered_skills() {
    let temp = TempDir::new().unwrap();
    let options = prepared_options(&temp);
    write_skill(&options.workspace, "alpha", "First skill", "# alpha body\n");
    let (mut repl, buffer) = scripted_repl(
        &options,
        &["/help", "/exit"],
        Arc::new(ScriptedModel::text("unused")),
    )
    .await;
    repl.run().await.unwrap();

    let text = output(&buffer);
    assert!(text.contains("Skills (invoke with /<skill-name>):"));
    assert!(text.contains("  /alpha  First skill"), "skill row: {text}");
    for line in HELP_LINES {
        assert!(text.contains(line), "missing {line:?} in {text}");
    }
}

// --- /cost and /status ---------------------------------------------------

#[tokio::test]
async fn repl_cost_reports_the_accumulated_usage() {
    let temp = TempDir::new().unwrap();
    let options = prepared_options(&temp);
    let scripted = ScriptedModel::builder()
        .completion(MockCompletion::text("hi").with_usage(Usage::new(3, 4)))
        .build();
    let probe = scripted.clone();
    let (mut repl, buffer) =
        scripted_repl(&options, &["hello", "/cost", "/exit"], Arc::new(scripted)).await;
    repl.run().await.unwrap();

    let text = output(&buffer);
    assert!(
        text.contains(
            "Token usage:\n  Input:  3 tokens\n  Output: 4 tokens\n  Total:  7 tokens\n  Calls:  1\n  Tool calls: 0"
        ),
        "cost block: {text}"
    );
    assert_eq!(probe.call_count(), 1);
}

#[tokio::test]
async fn repl_status_appends_the_usage_line() {
    let temp = TempDir::new().unwrap();
    let options = prepared_options(&temp);
    let scripted = ScriptedModel::builder()
        .completion(MockCompletion::text("hi").with_usage(Usage::new(3, 4)))
        .build();
    let (mut repl, buffer) =
        scripted_repl(&options, &["hello", "/status", "/exit"], Arc::new(scripted)).await;
    repl.run().await.unwrap();

    let text = output(&buffer);
    assert!(text.contains(&format!("Model: {}", options.model)));
    assert!(text.contains(&format!("Provider: {}", options.provider)));
    assert!(text.contains("Session: cli-"));
    assert!(
        text.contains("Input 3 | Output 4 | Total 7 tokens | Calls 1 | Tool calls 0"),
        "usage line: {text}"
    );
}

// --- /clear --------------------------------------------------------------

#[tokio::test]
async fn repl_clear_starts_a_new_session_and_resets_the_counters() {
    let temp = TempDir::new().unwrap();
    let options = prepared_options(&temp);
    let scripted = ScriptedModel::builder()
        .completion(MockCompletion::text("hi").with_usage(Usage::new(3, 4)))
        .build();
    let (mut repl, buffer) = scripted_repl(
        &options,
        &["hello", "/clear", "again", "/cost", "/exit"],
        Arc::new(scripted),
    )
    .await;
    let before = repl.session_id().to_owned();
    repl.run().await.unwrap();
    let after = repl.session_id().to_owned();

    assert_ne!(before, after, "/clear must start a new session");
    let text = output(&buffer);
    assert!(
        text.contains(&format!(
            "Cleared conversation and started a new session: {after}"
        )),
        "clear banner: {text}"
    );
    // The counters were reset, so only the turn after `/clear` is accounted.
    assert!(
        text.contains("Input:  0 tokens"),
        "reset cost block: {text}"
    );

    let store = options.sessions();
    let first = store.load(&before).unwrap().unwrap();
    assert_eq!(first.messages.len(), 2, "the first session keeps its turn");
    let second = store.load(&after).unwrap().unwrap();
    assert_eq!(second.messages.len(), 2, "the new session records its turn");
}

// --- /compact ------------------------------------------------------------

#[tokio::test]
async fn repl_compact_of_a_short_conversation_is_a_no_op() {
    let temp = TempDir::new().unwrap();
    let options = prepared_options(&temp);
    let (mut repl, buffer) = scripted_repl(
        &options,
        &["hello", "/compact", "/exit"],
        Arc::new(ScriptedModel::text("scripted answer")),
    )
    .await;
    let id = repl.session_id().to_owned();
    repl.run().await.unwrap();

    assert!(output(&buffer).contains("Nothing to compact."));
    let record = options.sessions().load(&id).unwrap().unwrap();
    assert_eq!(record.messages.len(), 2);
}

#[tokio::test]
async fn repl_compact_truncates_the_persisted_transcript() {
    let temp = TempDir::new().unwrap();
    let options = prepared_options(&temp);
    // Three turns with long, distinct answers put the mirror over its
    // 12 000-character budget while staying inside the eight-message window.
    let mut builder = ScriptedModel::builder();
    for turn in 0..3 {
        builder = builder.respond_text(format!("{turn}{}", "x".repeat(4_999)));
    }
    let model = builder.build();
    let (mut repl, buffer) = scripted_repl(&options, &["/compact", "/exit"], Arc::new(model)).await;
    for turn in 0..3 {
        repl.run_turn(&format!("message {turn}")).await.unwrap();
    }
    let id = repl.session_id().to_owned();
    repl.run().await.unwrap();

    assert!(
        output(&buffer).contains("Compacted conversation: 2 message(s) dropped, 4 retained."),
        "compaction report: {}",
        output(&buffer)
    );
    let record = options.sessions().load(&id).unwrap().unwrap();
    assert_eq!(record.messages.len(), 4, "persisted transcript shrank");
    assert_eq!(record.messages[0].role, "user");
    assert_eq!(record.messages[0].content, "message 1");
    assert_eq!(record.messages[3].role, "assistant");
    assert_eq!(record.messages[3].content.chars().count(), 5_000);
}

// --- /sessions -----------------------------------------------------------

#[tokio::test]
async fn repl_sessions_reports_an_empty_home() {
    let temp = TempDir::new().unwrap();
    let options = prepared_options(&temp);
    let (mut repl, buffer) = scripted_repl(
        &options,
        &["/sessions", "/exit"],
        Arc::new(ScriptedModel::text("unused")),
    )
    .await;
    repl.run().await.unwrap();

    assert!(output(&buffer).contains("No saved sessions."));
}

#[tokio::test]
async fn repl_sessions_lists_the_persisted_record() {
    let temp = TempDir::new().unwrap();
    let options = prepared_options(&temp);
    let (mut repl, buffer) = scripted_repl(
        &options,
        &["hello", "/sessions", "/exit"],
        Arc::new(ScriptedModel::text("scripted answer")),
    )
    .await;
    let id = repl.session_id().to_owned();
    repl.run().await.unwrap();

    let text = output(&buffer);
    assert!(text.contains("ID  Model  Created  Turns"), "header: {text}");
    let row = text
        .lines()
        .find(|line| line.contains(&id))
        .expect("the saved session has a row");
    // A record written by the turn has no model or timestamp yet, so only the
    // transcript length is guaranteed on the row.
    assert!(row.starts_with(&id), "row layout: {row}");
    assert!(row.ends_with("  2"), "two messages: {row}");
    assert!(!text.contains("No saved sessions."));
}

// --- shell passthrough ---------------------------------------------------

#[tokio::test]
async fn repl_interrupt_cancels_the_turn_and_a_new_turn_gets_a_fresh_token() {
    let temp = TempDir::new().unwrap();
    let options = prepared_options(&temp);
    let shell = FakeShell::new(FakeReply::Output(ShellOutput {
        command: "echo hi".to_owned(),
        stdout: "hi\n".to_owned(),
        stderr: String::new(),
        code: Some(0),
        ..ShellOutput::default()
    }));
    let (io, _buffer) = scripted_io(&["!echo hi", "/exit"]);
    let mut repl = repl_over(&options, io, Arc::new(ScriptedModel::text("unused")))
        .await
        .with_shell(shell.clone());

    // One press aborts the turn: the `!cmd` runner races this very token.
    let turn = repl.turn_cancellation();
    assert!(!turn.is_cancelled());
    assert_eq!(repl.on_interrupt(), InterruptAction::Abort);
    assert!(
        turn.is_cancelled(),
        "Ctrl-C must reach the shell command, not only the agent"
    );

    // The next turn starts a fresh token, so the interrupt does not leak forward.
    repl.run().await.unwrap();
    assert_eq!(shell.calls().len(), 1);
    assert_eq!(
        shell.cancellations(),
        vec![false],
        "a new turn must hand the shell an uncancelled token"
    );
}

#[tokio::test]
async fn repl_reports_an_interrupted_shell_command() {
    let temp = TempDir::new().unwrap();
    let options = prepared_options(&temp);
    let shell = FakeShell::new(FakeReply::Output(ShellOutput {
        command: "sleep 9999".to_owned(),
        stdout: String::new(),
        stderr: String::new(),
        code: None,
        cancelled: true,
        ..ShellOutput::default()
    }));
    let (io, buffer) = scripted_io(&["!sleep 9999", "/exit"]);
    let mut repl = repl_over(&options, io, Arc::new(ScriptedModel::text("unused")))
        .await
        .with_shell(shell);
    repl.run().await.unwrap();
    let text = output(&buffer);
    assert!(text.contains("Interrupted"), "{text}");
    assert!(!text.contains("Exit code"), "{text}");
}

#[tokio::test]
async fn repl_shell_command_prints_output_without_calling_the_model() {
    let temp = TempDir::new().unwrap();
    let options = prepared_options(&temp);
    let shell = FakeShell::new(FakeReply::Output(ShellOutput {
        command: "echo hi".to_owned(),
        stdout: "hi\n".to_owned(),
        stderr: "oops\n".to_owned(),
        code: Some(3),
        ..ShellOutput::default()
    }));
    let scripted = ScriptedModel::text("unused");
    let probe = scripted.clone();
    let (io, buffer) = scripted_io(&["!echo hi", "/exit"]);
    let mut repl = repl_over(&options, io.with_color(true), Arc::new(scripted))
        .await
        .with_shell(shell.clone());
    repl.run().await.unwrap();

    let text = output(&buffer);
    assert!(
        text.contains("hi\n\x1b[31moops\n\x1b[0mExit code: 3"),
        "shell output: {text:?}"
    );
    assert!(text.contains("Goodbye!"));
    assert_eq!(probe.call_count(), 0, "the shell never reaches the model");
    assert_eq!(
        shell.calls(),
        vec![(
            "echo hi".to_owned(),
            options.workspace.display().to_string()
        )]
    );
}

#[tokio::test]
async fn repl_shell_failure_is_reported_and_the_loop_continues() {
    let temp = TempDir::new().unwrap();
    let options = prepared_options(&temp);
    let shell = FakeShell::new(FakeReply::Error("shell exploded".to_owned()));
    let (repl, buffer) = scripted_repl(
        &options,
        &["!boom", "/exit"],
        Arc::new(ScriptedModel::text("unused")),
    )
    .await;
    let mut repl = repl.with_shell(shell);
    repl.run().await.unwrap();

    let text = output(&buffer);
    assert!(text.contains("✗ Error: "), "error prefix: {text}");
    assert!(text.contains("shell exploded"), "error detail: {text}");
    assert!(text.contains("Goodbye!"), "the loop kept going: {text}");
}

// --- skill turns ---------------------------------------------------------

#[tokio::test]
async fn repl_skill_turn_sends_and_persists_the_wrapped_prompt() {
    let temp = TempDir::new().unwrap();
    let options = prepared_options(&temp);
    write_skill(&options.workspace, "alpha", "First skill", "# alpha body\n");
    let scripted = ScriptedModel::text("done");
    let probe = scripted.clone();
    let (mut repl, buffer) =
        scripted_repl(&options, &["/alpha do it", "/exit"], Arc::new(scripted)).await;
    let id = repl.session_id().to_owned();
    repl.run().await.unwrap();

    assert!(output(&buffer).contains("done"));
    let prompt = probe
        .last_call()
        .expect("one model call")
        .last_user_text()
        .expect("a user message");
    assert!(
        prompt.starts_with("<skill-instructions>\n"),
        "wrapper: {prompt}"
    );
    assert!(prompt.contains("# alpha body"), "skill body: {prompt}");
    assert!(
        prompt.ends_with("</skill-instructions>\n\nUser request: do it"),
        "request: {prompt}"
    );

    let record = options.sessions().load(&id).unwrap().unwrap();
    assert_eq!(record.messages[0].content, prompt);
}

// --- input history -------------------------------------------------------

#[tokio::test]
async fn repl_history_records_submitted_lines() {
    let temp = TempDir::new().unwrap();
    let options = prepared_options(&temp);
    let path = temp.path().join("history.txt");
    let history = History::with_capacity(&path, 100);
    let (io, buffer) = scripted_io(&["hello", "", "/status", "/exit"]);
    let mut repl = repl_over(
        &options,
        io,
        Arc::new(ScriptedModel::text("scripted answer")),
    )
    .await
    .with_history(history);
    repl.run().await.unwrap();

    assert!(output(&buffer).contains("scripted answer"));
    let text = std::fs::read_to_string(&path).unwrap();
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines, ["hello", "/status", "/exit"]);
}

#[tokio::test]
async fn repl_history_replay_feeds_stored_lines_before_the_reader() {
    let temp = TempDir::new().unwrap();
    let options = prepared_options(&temp);
    let path = temp.path().join("history.txt");
    std::fs::write(&path, "first\nsecond\n").unwrap();
    let scripted = ScriptedModel::text("ok");
    let probe = scripted.clone();
    let (io, _buffer) = scripted_io(&["third", "/exit"]);
    let mut repl = repl_over(&options, io, Arc::new(scripted))
        .await
        .with_history(History::with_capacity(&path, 100))
        .with_history_replay(true);
    repl.run().await.unwrap();

    let prompts: Vec<String> = probe
        .calls()
        .iter()
        .filter_map(openjiuwen_test_support::ModelCall::last_user_text)
        .collect();
    assert_eq!(prompts, ["first", "second", "third"]);
}
