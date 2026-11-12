//! Integration suite for the `openjiuwen` command-line front-end.
//!
//! Every test drives the public `openjiuwen_harness::cli` surface with an
//! injected terminal ([`CliIo`] over a scripted reader and a buffered writer) and
//! a [`ScriptedModel`], so nothing touches the network, the real process
//! environment, the home directory or a real terminal.
//!
//! Covered here: the clap grammar (global flags, subcommands, defaults,
//! rejections, help/version), the one-shot runner's three output formats through
//! the headless `run_with_model` seam, a bounded scripted REPL transcript, the
//! permission answer mapping, and the `stream-json` line shape.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::VecDeque;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use clap::error::ErrorKind;
use clap::Parser;
use openjiuwen_common::config::OpenJiuwenHome;
use openjiuwen_common::error::Result as OJResult;
use openjiuwen_foundation::llm::Model;
use openjiuwen_harness::cli::{
    build_agent, run_with_model, Cli, CliIo, CliOptions, Command, LineReader, OutputFormat,
    PermissionAnswer, Repl, RunArgs, RunOutcome, Writer, DEFAULT_MAX_ITERATIONS,
    DEFAULT_MAX_TOKENS, DEFAULT_MODEL, DEFAULT_PROVIDER,
};
use openjiuwen_test_support::ScriptedModel;
use serde_json::Value;
use tempfile::TempDir;

/// Longest a scripted REPL transcript may take before the test fails.
const REPL_TIMEOUT: Duration = Duration::from_secs(10);

// --- scripted terminal ----------------------------------------------------

/// [`LineReader`] answering from a fixed script, then reporting end of input.
struct ScriptedReader {
    /// Lines still to hand out, front to back.
    lines: VecDeque<String>,
}

#[async_trait]
impl LineReader for ScriptedReader {
    async fn read_line(&mut self) -> OJResult<Option<String>> {
        Ok(self.lines.pop_front())
    }
}

/// [`Writer`] appending everything it receives to a buffer the test inspects.
struct SharedWriter {
    /// Text written so far, shared with the test.
    buffer: Arc<Mutex<String>>,
}

#[async_trait]
impl Writer for SharedWriter {
    async fn write(&mut self, text: &str) -> OJResult<()> {
        self.buffer
            .lock()
            .expect("writer buffer lock")
            .push_str(text);
        Ok(())
    }

    async fn flush(&mut self) -> OJResult<()> {
        Ok(())
    }
}

/// Builds a reader that yields `lines`, then end of input.
fn reader(lines: &[&str]) -> Box<dyn LineReader> {
    Box::new(ScriptedReader {
        lines: lines.iter().map(|line| (*line).to_owned()).collect(),
    })
}

/// Builds the pair of terminal handle and inspected buffer.
fn terminal(lines: &[&str], interactive: bool) -> (CliIo, Arc<Mutex<String>>) {
    let buffer = Arc::new(Mutex::new(String::new()));
    let writer = Box::new(SharedWriter {
        buffer: Arc::clone(&buffer),
    });
    let io = if interactive {
        CliIo::with_interactive(reader(lines), writer)
    } else {
        CliIo::new(reader(lines), writer)
    };
    (io, buffer)
}

/// Returns everything written to `buffer` so far.
fn rendered(buffer: &Arc<Mutex<String>>) -> String {
    buffer.lock().expect("writer buffer lock").clone()
}

// --- fixtures -------------------------------------------------------------

/// Builds the resolved options of a CLI run whose runtime state lives under `root`.
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

/// Parses `args` the way the `openjiuwen` binary would.
fn cli(args: &[&str]) -> Cli {
    Cli::try_parse_from(std::iter::once("openjiuwen").chain(args.iter().copied()))
        .expect("arguments must parse")
}

/// Parses `args` expecting a clap rejection.
fn cli_error(args: &[&str]) -> clap::Error {
    Cli::try_parse_from(std::iter::once("openjiuwen").chain(args.iter().copied()))
        .expect_err("arguments must be rejected")
}

/// Builds a scripted model answering `text` and a probe sharing its call history.
fn scripted_model(text: &str) -> (Arc<dyn Model>, ScriptedModel) {
    let model = ScriptedModel::text(text);
    let probe = model.clone();
    (Arc::new(model) as Arc<dyn Model>, probe)
}

/// Returns the `run` arguments of a parsed command line.
fn run_args(parsed: &Cli) -> &RunArgs {
    match &parsed.command {
        Some(Command::Run(args)) => args,
        other => panic!("expected a run command, got {other:?}"),
    }
}

// --- clap grammar ---------------------------------------------------------

#[test]
fn clap_parses_global_flags() {
    let parsed = cli(&[
        "--model",
        "m1",
        "--provider",
        "anthropic",
        "--api-key",
        "sk-test",
        "--api-base",
        "https://api.test",
        "--workspace",
        "/tmp/ws",
        "--verbose",
        "chat",
    ]);

    assert_eq!(parsed.model.as_deref(), Some("m1"));
    assert_eq!(parsed.provider.as_deref(), Some("anthropic"));
    assert_eq!(parsed.api_key.as_deref(), Some("sk-test"));
    assert_eq!(parsed.api_base.as_deref(), Some("https://api.test"));
    assert_eq!(parsed.workspace.as_deref(), Some(Path::new("/tmp/ws")));
    assert!(parsed.verbose);
    assert!(matches!(parsed.command, Some(Command::Chat(_))));
}

#[test]
fn clap_short_flags_match_long_forms() {
    let parsed = cli(&["-m", "m2", "-w", "ws2", "-v", "run", "hi"]);

    assert_eq!(parsed.model.as_deref(), Some("m2"));
    assert_eq!(parsed.workspace.as_deref(), Some(Path::new("ws2")));
    assert!(parsed.verbose);
    assert_eq!(run_args(&parsed).prompt.as_deref(), Some("hi"));
}

#[test]
fn clap_parses_run_prompt_and_output_formats() {
    let parsed = cli(&["run", "hi"]);
    assert_eq!(run_args(&parsed).prompt.as_deref(), Some("hi"));
    assert_eq!(run_args(&parsed).output_format, OutputFormat::Text);

    let parsed = cli(&["run", "hi", "--output-format", "json"]);
    assert_eq!(run_args(&parsed).output_format, OutputFormat::Json);
    assert_eq!(run_args(&parsed).output_format.as_str(), "json");

    let parsed = cli(&["run", "hi", "-f", "stream-json"]);
    assert_eq!(run_args(&parsed).output_format, OutputFormat::StreamJson);
    assert_eq!(run_args(&parsed).output_format.as_str(), "stream-json");

    let parsed = cli(&["run", "hi", "-f", "text"]);
    assert_eq!(run_args(&parsed).output_format, OutputFormat::Text);
    assert_eq!(run_args(&parsed).output_format.as_str(), "text");
}

#[test]
fn clap_parses_chat_subcommand() {
    assert!(matches!(cli(&["chat"]).command, Some(Command::Chat(_))));
    // Global flags are accepted after the subcommand too.
    let parsed = cli(&["chat", "--model", "m3", "-v"]);
    assert_eq!(parsed.model.as_deref(), Some("m3"));
    assert!(parsed.verbose);
    assert!(matches!(parsed.command, Some(Command::Chat(_))));
}

#[test]
fn clap_defaults_without_subcommand() {
    let parsed = cli(&[]);
    assert!(parsed.command.is_none());
    assert!(parsed.model.is_none());
    assert!(parsed.provider.is_none());
    assert!(parsed.api_key.is_none());
    assert!(parsed.api_base.is_none());
    assert!(parsed.workspace.is_none());
    assert!(!parsed.verbose);

    let defaults = RunArgs::default();
    assert_eq!(defaults.prompt, None);
    assert_eq!(defaults.output_format, OutputFormat::Text);
    assert_eq!(OutputFormat::default(), OutputFormat::Text);
}

#[test]
fn clap_rejects_unknown_flag() {
    let error = cli_error(&["--bogus"]);
    assert_eq!(error.kind(), ErrorKind::UnknownArgument);

    let error = cli_error(&["run", "hi", "--bogus"]);
    assert_eq!(error.kind(), ErrorKind::UnknownArgument);
}

#[test]
fn clap_rejects_unknown_subcommand() {
    let error = cli_error(&["bogus"]);
    assert_eq!(error.kind(), ErrorKind::InvalidSubcommand);
}

#[test]
fn clap_rejects_invalid_output_format() {
    let error = cli_error(&["run", "hi", "--output-format", "bogus"]);
    assert_eq!(error.kind(), ErrorKind::InvalidValue);
    assert!(error.to_string().contains("bogus"));
}

#[test]
fn clap_help_and_version() {
    let help = cli_error(&["--help"]);
    assert_eq!(help.kind(), ErrorKind::DisplayHelp);
    let text = help.to_string();
    assert!(text.contains("Usage"), "help must show a usage line");
    assert!(text.contains("run"), "help must list the run subcommand");
    assert!(text.contains("chat"), "help must list the chat subcommand");

    let version = cli_error(&["-V"]);
    assert_eq!(version.kind(), ErrorKind::DisplayVersion);
    assert!(
        version.to_string().contains(env!("CARGO_PKG_VERSION")),
        "version output must carry the crate version"
    );
}

// --- one-shot runner ------------------------------------------------------

#[tokio::test]
async fn run_with_model_text_renders_answer() {
    let temp = TempDir::new().expect("temp dir");
    let options = test_options(temp.path());
    options.ensure_workspace().expect("workspace");
    let (io, buffer) = terminal(&[], false);
    let (model, probe) = scripted_model("hi there");
    let args = RunArgs {
        prompt: Some("hello".to_owned()),
        output_format: OutputFormat::Text,
    };

    let outcome = run_with_model(&options, &args, &io, model)
        .await
        .expect("text run succeeds");

    assert_eq!(outcome.answer, "hi there");
    assert_eq!(outcome.result_type, "answer");
    assert!(outcome.chunks > 0, "a run consumes at least one chunk");
    assert!(outcome.session_id.starts_with("cli-"));
    assert_eq!(probe.call_count(), 1, "the model is called once per run");
    let text = rendered(&buffer);
    assert!(text.contains("hi there"), "answer must be rendered: {text}");
}

#[tokio::test]
async fn run_with_model_json_prints_one_object() {
    let temp = TempDir::new().expect("temp dir");
    let options = test_options(temp.path());
    options.ensure_workspace().expect("workspace");
    let (io, buffer) = terminal(&[], false);
    let (model, probe) = scripted_model("hi there");
    let args = RunArgs {
        prompt: Some("hello".to_owned()),
        output_format: OutputFormat::Json,
    };

    let outcome = run_with_model(&options, &args, &io, model)
        .await
        .expect("json run succeeds");

    let value: Value =
        serde_json::from_str(rendered(&buffer).trim()).expect("json output parses as one object");
    let object = value.as_object().expect("json output is an object");
    for key in ["answer", "result_type", "usage", "session_id"] {
        assert!(object.contains_key(key), "missing key {key}");
    }
    assert_eq!(value["answer"], "hi there");
    assert_eq!(value["result_type"], "answer");
    assert_eq!(value["session_id"], outcome.session_id);
    assert_eq!(probe.call_count(), 1);
}

#[tokio::test]
async fn run_with_model_stream_json_prints_typed_lines() {
    let temp = TempDir::new().expect("temp dir");
    let options = test_options(temp.path());
    options.ensure_workspace().expect("workspace");
    let (io, buffer) = terminal(&[], false);
    let (model, probe) = scripted_model("hi there");
    let args = RunArgs {
        prompt: Some("hello".to_owned()),
        output_format: OutputFormat::StreamJson,
    };

    let outcome = run_with_model(&options, &args, &io, model)
        .await
        .expect("stream-json run succeeds");

    let text = rendered(&buffer);
    let lines: Vec<&str> = text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .collect();
    assert!(
        lines.len() >= 2,
        "expected chunk lines plus a terminal result line, got {lines:?}"
    );
    let values: Vec<Value> = lines
        .iter()
        .map(|line| serde_json::from_str(line).expect("every line is JSON"))
        .collect();
    for value in &values[..values.len() - 1] {
        let object = value.as_object().expect("a chunk line is an object");
        assert!(object.contains_key("type"), "chunk line needs a type field");
        assert!(value["type"].is_string(), "chunk type must be a string");
        for key in ["index", "payload"] {
            assert!(object.contains_key(key), "missing chunk key {key}");
        }
    }
    let kinds: Vec<&str> = values[..values.len() - 1]
        .iter()
        .filter_map(|value| value["type"].as_str())
        .collect();
    assert!(
        kinds.contains(&"llm_output"),
        "streamed text must appear as an llm_output line, got {kinds:?}"
    );

    let last = values.last().expect("a terminal result line");
    assert_eq!(last["type"], "result");
    assert_eq!(last["answer"], "hi there");
    assert_eq!(last["session_id"], outcome.session_id);
    assert_eq!(probe.call_count(), 1);
}

#[tokio::test]
async fn run_with_model_reads_prompt_from_stdin() {
    let temp = TempDir::new().expect("temp dir");
    let options = test_options(temp.path());
    options.ensure_workspace().expect("workspace");
    let (io, buffer) = terminal(&["piped prompt"], false);
    let (model, probe) = scripted_model("hi there");
    let args = RunArgs {
        prompt: None,
        output_format: OutputFormat::Text,
    };

    let outcome = run_with_model(&options, &args, &io, model)
        .await
        .expect("stdin run succeeds");

    assert_eq!(outcome.answer, "hi there");
    assert_eq!(probe.call_count(), 1);
    let last_prompt = probe
        .last_call()
        .expect("one recorded call")
        .last_user_text();
    assert!(
        last_prompt.expect("a user turn").contains("piped prompt"),
        "the piped line must reach the model"
    );
    assert!(rendered(&buffer).contains("hi there"));
}

#[tokio::test]
async fn run_outcome_json_matches_the_rendered_object() {
    let outcome = RunOutcome {
        answer: "done".to_owned(),
        result_type: "answer".to_owned(),
        usage: None,
        session_id: "cli-abc12345".to_owned(),
        chunks: 2,
    };

    let value = outcome.to_json();
    assert_eq!(value["answer"], "done");
    assert_eq!(value["result_type"], "answer");
    assert_eq!(value["session_id"], "cli-abc12345");
    assert!(value["usage"].is_null());
}

// --- REPL -----------------------------------------------------------------

/// Builds a REPL over a scripted terminal and a scripted-model agent.
async fn build_repl(
    options: &CliOptions,
    inputs: &[&str],
) -> (Repl, Arc<Mutex<String>>, ScriptedModel) {
    let (io, buffer) = terminal(inputs, true);
    let (model, probe) = scripted_model("scripted answer");
    let agent = build_agent(model, options, &io)
        .await
        .expect("agent builds");
    (Repl::with_io(agent, options.clone(), io), buffer, probe)
}

#[tokio::test]
async fn repl_scripted_transcript() {
    let temp = TempDir::new().expect("temp dir");
    let options = test_options(temp.path());
    options.ensure_workspace().expect("workspace");
    let (mut repl, buffer, probe) =
        build_repl(&options, &["/help", "/mode plan", "hello there", "/exit"]).await;

    let run = tokio::time::timeout(REPL_TIMEOUT, repl.run())
        .await
        .expect("the scripted transcript terminates");
    run.expect("the REPL exits cleanly");

    let text = rendered(&buffer);
    assert!(text.contains("OpenJiuwen CLI v"), "banner: {text}");
    assert!(text.contains("Available commands:"), "help table: {text}");
    assert!(text.contains("/mode <normal|plan>"), "help listing: {text}");
    assert!(
        text.contains("Mode: plan"),
        "plan mode confirmation: {text}"
    );
    assert!(
        text.contains("You> "),
        "the REPL prompts for every line: {text}"
    );
    assert!(
        text.contains("scripted answer"),
        "the conversational answer is rendered: {text}"
    );
    assert!(
        text.contains("Goodbye!"),
        "the exit path says goodbye: {text}"
    );
    assert_eq!(
        probe.call_count(),
        1,
        "exactly one conversational turn reaches the model"
    );
    assert_eq!(
        probe
            .last_call()
            .expect("one recorded call")
            .last_user_text()
            .as_deref(),
        Some("hello there")
    );
}

#[tokio::test]
async fn repl_mode_switch_and_unknown_commands() {
    let temp = TempDir::new().expect("temp dir");
    let options = test_options(temp.path());
    options.ensure_workspace().expect("workspace");
    let (mut repl, buffer, probe) =
        build_repl(&options, &["/bogus", "/mode sideways", "/status", "/quit"]).await;

    tokio::time::timeout(REPL_TIMEOUT, repl.run())
        .await
        .expect("the scripted transcript terminates")
        .expect("the REPL exits cleanly");

    let text = rendered(&buffer);
    assert!(text.contains("Unknown command: /bogus"));
    assert!(text.contains("Type /help to see available commands."));
    assert!(text.contains("Usage: /mode <normal|plan>"));
    assert!(text.contains(&format!("Model: {}", options.model)));
    assert!(text.contains(&format!("Provider: {}", options.provider)));
    assert!(text.contains("Goodbye!"));
    assert_eq!(probe.call_count(), 0, "no conversational turn was typed");
}

// --- permission answers ---------------------------------------------------

#[test]
fn permission_answers_approve_once() {
    for answer in [
        "",
        "y",
        "Y",
        "yes",
        "YES",
        "ok",
        "approve",
        "allow",
        "allow_once",
        "true",
        "1",
        " y ",
    ] {
        assert_eq!(
            PermissionAnswer::parse(answer),
            PermissionAnswer::Approve,
            "answer {answer:?} must approve once"
        );
    }

    let response = PermissionAnswer::Approve.to_response();
    assert!(response.approved);
    assert_eq!(response.feedback, "");
    assert!(!response.auto_confirm);
    assert_eq!(response.persist_allow, None);
}

#[test]
fn permission_answers_always_set_auto_confirm() {
    for answer in ["a", "A", "always", "ALWAYS", "allow_always", " always "] {
        assert_eq!(
            PermissionAnswer::parse(answer),
            PermissionAnswer::Always,
            "answer {answer:?} must approve always"
        );
    }

    let response = PermissionAnswer::Always.to_response();
    assert!(response.approved);
    assert_eq!(response.feedback, "");
    assert!(response.auto_confirm);
    assert_eq!(response.persist_allow, None);
}

#[test]
fn permission_answers_reject() {
    for answer in ["n", "N", "no", "deny", "reject", " REJECT "] {
        assert_eq!(
            PermissionAnswer::parse(answer),
            PermissionAnswer::Reject,
            "answer {answer:?} must reject"
        );
    }

    let response = PermissionAnswer::Reject.to_response();
    assert!(!response.approved);
    assert_eq!(response.feedback, "");
    assert!(!response.auto_confirm);
    assert_eq!(response.persist_allow, None);
}

#[test]
fn permission_answers_anything_else_is_feedback() {
    assert_eq!(
        PermissionAnswer::parse("  use another path  "),
        PermissionAnswer::Feedback("use another path".to_owned()),
        "feedback keeps the trimmed original text"
    );
    assert_eq!(
        PermissionAnswer::parse("Yess"),
        PermissionAnswer::Feedback("Yess".to_owned())
    );

    let response = PermissionAnswer::parse("  use another path  ").to_response();
    assert!(!response.approved);
    assert_eq!(response.feedback, "use another path");
    assert!(!response.auto_confirm);
    assert_eq!(response.persist_allow, None);
}
