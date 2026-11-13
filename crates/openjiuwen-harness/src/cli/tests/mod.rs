//! Offline, deterministic test suite of the `cli` node.
//!
//! Every test drives the real CLI entry points with scripted IO handles,
//! [`openjiuwen_test_support::ScriptedModel`] and a temporary home/workspace, so
//! nothing touches the network, the process environment or a real terminal.
//!
//! The suite is split in two files:
//!
//! - this one holds the shared harness (scripted reader/writer, option and agent
//!   builders) plus the parity tests: the clap grammar, dispatch, option
//!   precedence (flags, `OPENJIUWEN_*`, generic SDK variables, settings,
//!   defaults), settings parsing, permission prompts and their answers, session
//!   records, the chunk renderer, the one-shot runner's three output formats, and
//!   a scripted REPL transcript;
//! - `commands` covers the phase-3 REPL surface: the full slash-command table,
//!   `/cost`, `/compact`, `/sessions`, `/clear`, the `!cmd` shell passthrough,
//!   `/<skill>` turns and input history.
#![allow(clippy::unwrap_used, clippy::expect_used)]

mod commands;

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use clap::Parser;
use openjiuwen_agent::AgentChunk;
use openjiuwen_common::config::env::{
    API_BASE, API_KEY, MODEL_NAME, MODEL_PROVIDER, OPENAI_API_KEY, OPENAI_BASE_URL,
};
use openjiuwen_common::config::OpenJiuwenHome;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::security::SecretString;
use openjiuwen_foundation::llm::Model;
use openjiuwen_test_support::ScriptedModel;
use serde_json::{json, Value};
use tempfile::TempDir;

use crate::permission::{
    ConfirmationOutcome, PermissionConfirmResponse, PermissionConfirmationRequest, PermissionHost,
    PermissionLevel, PermissionResult, PermissionsSection,
};

use super::*;

/// [`super::LineReader`] that answers from a fixed script.
struct ScriptedReader {
    lines: VecDeque<String>,
}

#[async_trait]
impl LineReader for ScriptedReader {
    async fn read_line(&mut self) -> Result<Option<String>> {
        Ok(self.lines.pop_front())
    }
}

/// [`super::Writer`] that appends everything to a shared buffer.
struct SharedWriter {
    buffer: Arc<Mutex<String>>,
}

#[async_trait]
impl Writer for SharedWriter {
    async fn write(&mut self, text: &str) -> Result<()> {
        self.buffer.lock().unwrap().push_str(text);
        Ok(())
    }

    async fn flush(&mut self) -> Result<()> {
        Ok(())
    }
}

/// Builds a `Box<dyn LineReader>` that yields `lines`, then EOF.
fn reader(lines: &[&str]) -> Box<dyn LineReader> {
    Box::new(ScriptedReader {
        lines: lines.iter().map(|line| (*line).to_owned()).collect(),
    })
}

/// Builds a non-interactive handle (one-shot `run` semantics).
fn scripted_io(lines: &[&str]) -> (CliIo, Arc<Mutex<String>>) {
    let buffer = Arc::new(Mutex::new(String::new()));
    let writer = Box::new(SharedWriter {
        buffer: Arc::clone(&buffer),
    });
    (CliIo::new(reader(lines), writer), buffer)
}

/// Builds an interactive handle (permission prompts are answerable).
fn interactive_io(lines: &[&str]) -> (CliIo, Arc<Mutex<String>>) {
    let buffer = Arc::new(Mutex::new(String::new()));
    let writer = Box::new(SharedWriter {
        buffer: Arc::clone(&buffer),
    });
    (CliIo::with_interactive(reader(lines), writer), buffer)
}

/// Returns everything written so far.
fn output(buffer: &Arc<Mutex<String>>) -> String {
    buffer.lock().unwrap().clone()
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

/// Parses `args` as the `openjiuwen` binary would.
fn cli(args: &[&str]) -> Cli {
    Cli::try_parse_from(std::iter::once("openjiuwen").chain(args.iter().copied())).unwrap()
}

/// Returns the API key of `options` as plain text, when set.
fn api_key_of(options: &CliOptions) -> Option<String> {
    options
        .api_key
        .as_ref()
        .map(|secret| secret.expose_secret().to_owned())
}

/// Asserts that `value` is a JSON object carrying every key in `keys`.
fn assert_object_keys(value: &Value, keys: &[&str]) {
    let object = value.as_object().expect("expected a JSON object");
    for key in keys {
        assert!(object.contains_key(*key), "missing key {key}");
    }
}

/// Builds a permission request as the interrupt rail would.
fn confirmation_request() -> PermissionConfirmationRequest {
    PermissionConfirmationRequest {
        tool_name: "write_file".to_owned(),
        tool_call_id: "call-1".to_owned(),
        arguments: json!({"path": "a.txt"}),
        result: PermissionResult::new(PermissionLevel::Ask),
        auto_confirm_key: String::new(),
    }
}

/// Builds a permission host over `io`.
fn permission_host(io: CliIo) -> CliPermissionHost {
    CliPermissionHost::new(
        io,
        "cli-test".to_owned(),
        PathBuf::from("/workspace"),
        PermissionsSection::default(),
    )
}

/// Unwraps the response of a host answer, failing on any other outcome.
fn response_of(outcome: ConfirmationOutcome) -> PermissionConfirmResponse {
    match outcome {
        ConfirmationOutcome::Response(response) => response,
        other => panic!("expected a response, got {other:?}"),
    }
}

/// Builds a REPL over a scripted terminal and a scripted-model agent.
async fn build_repl(options: &CliOptions, inputs: &[&str]) -> (Repl, Arc<Mutex<String>>) {
    let (io, buffer) = scripted_io(inputs);
    let model: Arc<dyn Model> = Arc::new(ScriptedModel::text("scripted answer"));
    let agent = super::run::build_agent(model, options, &io).await.unwrap();
    (Repl::with_io(agent, options.clone(), io), buffer)
}

// --- IO handles -----------------------------------------------------------

#[test]
fn stdio_handle_interactivity_flags() {
    // Constructing a stdio handle performs no IO; only the flags are observed.
    assert!(!CliIo::stdio_non_interactive().is_interactive());
    assert!(CliIo::stdio().is_interactive());
}

// --- clap parsing ---------------------------------------------------------

#[test]
fn clap_no_subcommand_has_no_command() {
    let parsed = cli(&[]);
    assert!(parsed.command.is_none());
    assert!(parsed.model.is_none());
    assert!(parsed.provider.is_none());
    assert!(parsed.api_key.is_none());
    assert!(parsed.api_base.is_none());
    assert!(parsed.workspace.is_none());
    assert!(!parsed.verbose);
}

#[test]
fn clap_parses_run_prompt_and_output_formats() {
    let parsed = cli(&["run", "hi"]);
    match parsed.command {
        Some(Command::Run(args)) => {
            assert_eq!(args.prompt.as_deref(), Some("hi"));
            assert_eq!(args.output_format, OutputFormat::Text);
        }
        other => panic!("expected a run command, got {other:?}"),
    }

    let parsed = cli(&["run", "hi", "-f", "json"]);
    match parsed.command {
        Some(Command::Run(args)) => assert_eq!(args.output_format, OutputFormat::Json),
        other => panic!("expected a run command, got {other:?}"),
    }

    let parsed = cli(&["run", "hi", "--output-format", "stream-json"]);
    match parsed.command {
        Some(Command::Run(args)) => assert_eq!(args.output_format, OutputFormat::StreamJson),
        other => panic!("expected a run command, got {other:?}"),
    }
    assert_eq!(OutputFormat::StreamJson.as_str(), "stream-json");
}

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
    assert_eq!(parsed.workspace, Some(PathBuf::from("/tmp/ws")));
    assert!(parsed.verbose);
    assert!(matches!(parsed.command, Some(Command::Chat(_))));
}

#[test]
fn clap_rejects_invalid_arguments() {
    assert!(
        Cli::try_parse_from(["openjiuwen", "run", "--output-format", "bogus"]).is_err(),
        "an unknown output format must be rejected"
    );
    assert!(
        Cli::try_parse_from(["openjiuwen", "bogus"]).is_err(),
        "an unknown subcommand must be rejected"
    );
}

// --- dispatch -------------------------------------------------------------

#[test]
fn dispatch_without_subcommand_follows_terminal() {
    let parsed = cli(&[]);
    assert_eq!(parsed.dispatch(true), Dispatch::Chat);
    assert_eq!(parsed.dispatch(false), Dispatch::Run(RunArgs::default()));
}

#[test]
fn dispatch_explicit_subcommands_win() {
    let parsed = cli(&["run", "hi", "-f", "json"]);
    let expected = Dispatch::Run(RunArgs {
        prompt: Some("hi".to_owned()),
        output_format: OutputFormat::Json,
    });
    assert_eq!(parsed.dispatch(true), expected);
    assert_eq!(parsed.dispatch(false), expected);

    let parsed = cli(&["chat"]);
    assert_eq!(parsed.dispatch(true), Dispatch::Chat);
    assert_eq!(parsed.dispatch(false), Dispatch::Chat);
}

// --- option resolution ----------------------------------------------------

#[test]
fn resolve_flags_override_env_and_settings() {
    let temp = TempDir::new().unwrap();
    let cwd = temp.path().join("cwd");
    let env = MapEnv::from_pairs([
        (OPENJIUWEN_MODEL, "env-model"),
        (OPENJIUWEN_PROVIDER, "env-provider"),
        (OPENJIUWEN_API_KEY, "env-key"),
        (OPENJIUWEN_API_BASE, "https://env.test"),
        (OPENJIUWEN_WORKSPACE, "/env/workspace"),
    ]);
    let settings = Settings {
        model: Some("settings-model".to_owned()),
        provider: Some("settings-provider".to_owned()),
        api_key: Some(SecretString::new("settings-key")),
        api_base: Some("https://settings.test".to_owned()),
        workspace: Some(PathBuf::from("/settings/workspace")),
        ..Settings::default()
    };
    let parsed = cli(&[
        "--model",
        "flag-model",
        "--provider",
        "flag-provider",
        "--api-key",
        "flag-key",
        "--api-base",
        "https://flag.test",
        "--workspace",
        "flag-ws",
    ]);
    let options = CliOptions::resolve_with(&parsed, &env, Some(&settings), &cwd).unwrap();
    assert_eq!(options.model, "flag-model");
    assert_eq!(options.provider, "flag-provider");
    assert_eq!(api_key_of(&options).as_deref(), Some("flag-key"));
    assert_eq!(options.api_base.as_deref(), Some("https://flag.test"));
    assert_eq!(options.workspace, cwd.join("flag-ws"));
}

#[test]
fn resolve_model_and_provider_precedence() {
    let temp = TempDir::new().unwrap();
    let cwd = temp.path();
    let settings = Settings {
        model: Some("settings-model".to_owned()),
        provider: Some("settings-provider".to_owned()),
        ..Settings::default()
    };

    let env = MapEnv::from_pairs([
        (OPENJIUWEN_MODEL, "openjiuwen-model"),
        (MODEL_NAME, "generic-model"),
        (OPENJIUWEN_PROVIDER, "openjiuwen-provider"),
        (MODEL_PROVIDER, "generic-provider"),
    ]);
    let options = CliOptions::resolve_with(&cli(&[]), &env, Some(&settings), cwd).unwrap();
    assert_eq!(options.model, "openjiuwen-model");
    assert_eq!(options.provider, "openjiuwen-provider");

    let env = MapEnv::from_pairs([
        (MODEL_NAME, "generic-model"),
        (MODEL_PROVIDER, "generic-provider"),
    ]);
    let options = CliOptions::resolve_with(&cli(&[]), &env, Some(&settings), cwd).unwrap();
    assert_eq!(options.model, "generic-model");
    assert_eq!(options.provider, "generic-provider");

    let options =
        CliOptions::resolve_with(&cli(&[]), &MapEnv::new(), Some(&settings), cwd).unwrap();
    assert_eq!(options.model, "settings-model");
    assert_eq!(options.provider, "settings-provider");

    let options = CliOptions::resolve_with(&cli(&[]), &MapEnv::new(), None, cwd).unwrap();
    assert_eq!(options.model, DEFAULT_MODEL);
    assert_eq!(options.provider, DEFAULT_PROVIDER);
}

#[test]
fn resolve_api_key_precedence() {
    let temp = TempDir::new().unwrap();
    let cwd = temp.path();
    let settings = Settings {
        api_key: Some(SecretString::new("settings-key")),
        ..Settings::default()
    };

    let env = MapEnv::from_pairs([
        (OPENJIUWEN_API_KEY, "openjiuwen-key"),
        (API_KEY, "api-key"),
        (OPENAI_API_KEY, "openai-key"),
    ]);
    let options = CliOptions::resolve_with(&cli(&[]), &env, Some(&settings), cwd).unwrap();
    assert_eq!(api_key_of(&options).as_deref(), Some("openjiuwen-key"));

    let env = MapEnv::from_pairs([(API_KEY, "api-key"), (OPENAI_API_KEY, "openai-key")]);
    let options = CliOptions::resolve_with(&cli(&[]), &env, Some(&settings), cwd).unwrap();
    assert_eq!(api_key_of(&options).as_deref(), Some("api-key"));

    let env = MapEnv::from_pairs([(OPENAI_API_KEY, "openai-key")]);
    let options = CliOptions::resolve_with(&cli(&[]), &env, Some(&settings), cwd).unwrap();
    assert_eq!(api_key_of(&options).as_deref(), Some("openai-key"));

    let options =
        CliOptions::resolve_with(&cli(&[]), &MapEnv::new(), Some(&settings), cwd).unwrap();
    assert_eq!(api_key_of(&options).as_deref(), Some("settings-key"));

    let options = CliOptions::resolve_with(&cli(&[]), &MapEnv::new(), None, cwd).unwrap();
    assert!(options.api_key.is_none());
}

#[test]
fn resolve_api_base_precedence() {
    let temp = TempDir::new().unwrap();
    let cwd = temp.path();
    let settings = Settings {
        api_base: Some("https://settings.test".to_owned()),
        ..Settings::default()
    };

    let env = MapEnv::from_pairs([
        (OPENJIUWEN_API_BASE, "https://openjiuwen.test"),
        (API_BASE, "https://api.test"),
        (OPENAI_BASE_URL, "https://openai.test"),
    ]);
    let options = CliOptions::resolve_with(&cli(&[]), &env, Some(&settings), cwd).unwrap();
    assert_eq!(options.api_base.as_deref(), Some("https://openjiuwen.test"));

    let env = MapEnv::from_pairs([
        (API_BASE, "https://api.test"),
        (OPENAI_BASE_URL, "https://openai.test"),
    ]);
    let options = CliOptions::resolve_with(&cli(&[]), &env, Some(&settings), cwd).unwrap();
    assert_eq!(options.api_base.as_deref(), Some("https://api.test"));

    let env = MapEnv::from_pairs([(OPENAI_BASE_URL, "https://openai.test")]);
    let options = CliOptions::resolve_with(&cli(&[]), &env, Some(&settings), cwd).unwrap();
    assert_eq!(options.api_base.as_deref(), Some("https://openai.test"));

    let options =
        CliOptions::resolve_with(&cli(&[]), &MapEnv::new(), Some(&settings), cwd).unwrap();
    assert_eq!(options.api_base.as_deref(), Some("https://settings.test"));

    let options = CliOptions::resolve_with(&cli(&[]), &MapEnv::new(), None, cwd).unwrap();
    assert!(options.api_base.is_none());
}

#[test]
fn resolve_workspace_precedence() {
    let temp = TempDir::new().unwrap();
    let root = temp.path();
    let cwd = root.join("cwd");
    let env_workspace = root.join("env-workspace");
    let settings_workspace = root.join("settings-workspace");

    let env = MapEnv::from_pairs([
        (OPENJIUWEN_WORKSPACE, env_workspace.to_string_lossy()),
        (OPENJIUWEN_HOME, root.to_string_lossy()),
    ]);
    let settings = Settings {
        workspace: Some(settings_workspace.clone()),
        ..Settings::default()
    };
    let options = CliOptions::resolve_with(&cli(&[]), &env, Some(&settings), &cwd).unwrap();
    assert_eq!(options.workspace, env_workspace);

    let env = MapEnv::from_pairs([(OPENJIUWEN_HOME, root.to_string_lossy())]);
    let options = CliOptions::resolve_with(&cli(&[]), &env, Some(&settings), &cwd).unwrap();
    assert_eq!(options.workspace, settings_workspace);

    let options = CliOptions::resolve_with(&cli(&[]), &env, None, &cwd).unwrap();
    assert_eq!(options.workspace, options.home.join("workspace"));
    assert_eq!(options.workspace, root.join("workspace"));
}

#[test]
fn resolve_defaults() {
    let temp = TempDir::new().unwrap();
    let root = temp.path();
    let env = MapEnv::from_pairs([(OPENJIUWEN_HOME, root.to_string_lossy())]);
    let options = CliOptions::resolve_with(&cli(&[]), &env, None, root).unwrap();
    assert_eq!(options.model, DEFAULT_MODEL);
    assert_eq!(options.provider, DEFAULT_PROVIDER);
    assert_eq!(options.max_iterations, DEFAULT_MAX_ITERATIONS);
    assert_eq!(options.max_tokens, DEFAULT_MAX_TOKENS);
    assert!(options.api_key.is_none());
    assert!(options.api_base.is_none());
    assert!(!options.verbose);
    assert_eq!(options.workspace, root.join("workspace"));
    assert_eq!(options.home.root(), root);
}

#[test]
fn resolve_debug_redacts_api_key() {
    let temp = TempDir::new().unwrap();
    let mut options = test_options(temp.path());
    options.api_key = Some(SecretString::new("super-secret-key"));
    let debug = format!("{options:?}");
    assert!(!debug.contains("super-secret-key"));
    assert!(debug.contains("redacted"));
}

#[test]
fn resolve_rejects_small_token_budget() {
    let temp = TempDir::new().unwrap();
    let env = MapEnv::from_pairs([(OPENJIUWEN_MAX_TOKENS, "100")]);
    let error = CliOptions::resolve_with(&cli(&[]), &env, None, temp.path()).unwrap_err();
    match error {
        Error::Validation(message) => assert!(message.contains("dangerously small")),
        other => panic!("expected a validation error, got {other:?}"),
    }
}

#[test]
fn resolve_rejects_zero_iterations() {
    let temp = TempDir::new().unwrap();
    let env = MapEnv::from_pairs([(OPENJIUWEN_MAX_ITERATIONS, "0")]);
    let error = CliOptions::resolve_with(&cli(&[]), &env, None, temp.path()).unwrap_err();
    match error {
        Error::Validation(message) => assert!(message.contains("must be >= 1")),
        other => panic!("expected a validation error, got {other:?}"),
    }
}

#[test]
fn resolve_rejects_non_integer_token_budget() {
    let temp = TempDir::new().unwrap();
    let env = MapEnv::from_pairs([(OPENJIUWEN_MAX_TOKENS, "abc")]);
    let error = CliOptions::resolve_with(&cli(&[]), &env, None, temp.path()).unwrap_err();
    match error {
        Error::Validation(message) => {
            assert!(message.contains("must be an integer, got abc"));
        }
        other => panic!("expected a validation error, got {other:?}"),
    }
}

#[test]
fn resolve_permissions_ask_for_file_writes() {
    let temp = TempDir::new().unwrap();
    let section = test_options(temp.path()).permissions();
    assert_eq!(section.tools.get("write_file"), Some(&PermissionLevel::Ask));
    assert_eq!(section.tools.get("edit_file"), Some(&PermissionLevel::Ask));
}

// --- settings -------------------------------------------------------------

#[test]
fn settings_accepts_aliases_and_ignores_unknown() {
    let settings = Settings::from_json(
        r#"{
            "apiKey": "k",
            "apiBase": "https://base.test",
            "maxIterations": 3,
            "maxTokens": 512,
            "model": "m",
            "unknown": 42
        }"#,
    )
    .unwrap();
    assert_eq!(
        settings.api_key.as_ref().map(SecretString::expose_secret),
        Some("k")
    );
    assert_eq!(settings.api_base.as_deref(), Some("https://base.test"));
    assert_eq!(settings.max_iterations, Some(3));
    assert_eq!(settings.max_tokens, Some(512));
    assert_eq!(settings.model.as_deref(), Some("m"));
}

#[test]
fn settings_load_missing_file_is_none() {
    let temp = TempDir::new().unwrap();
    let settings = Settings::load(&temp.path().join("missing.json")).unwrap();
    assert!(settings.is_none());
}

#[test]
fn settings_load_malformed_file_is_none() {
    let temp = TempDir::new().unwrap();
    let path = temp.path().join("settings.json");
    std::fs::write(&path, "{not json").unwrap();
    let settings = Settings::load(&path).unwrap();
    assert!(settings.is_none());
}

// --- permissions ----------------------------------------------------------

#[test]
fn permission_answer_parse_table() {
    for answer in [
        "",
        "y",
        "Y",
        "yes",
        "ok",
        "approve",
        "allow",
        "allow_once",
        "true",
        "1",
    ] {
        assert_eq!(PermissionAnswer::parse(answer), PermissionAnswer::Approve);
    }
    for answer in ["a", "always", "allow_always"] {
        assert_eq!(PermissionAnswer::parse(answer), PermissionAnswer::Always);
    }
    for answer in ["n", "no", "deny", "reject"] {
        assert_eq!(PermissionAnswer::parse(answer), PermissionAnswer::Reject);
    }
    assert_eq!(
        PermissionAnswer::parse("  use another path  "),
        PermissionAnswer::Feedback("use another path".to_owned())
    );
}

#[test]
fn permission_answer_maps_to_response() {
    let approve = PermissionAnswer::Approve.to_response();
    assert!(approve.approved);
    assert!(!approve.auto_confirm);
    assert_eq!(approve.feedback, "");

    let always = PermissionAnswer::Always.to_response();
    assert!(always.approved);
    assert!(always.auto_confirm);

    let reject = PermissionAnswer::Reject.to_response();
    assert!(!reject.approved);
    assert!(!reject.auto_confirm);
    assert_eq!(reject.feedback, "");

    let feedback = PermissionAnswer::Feedback("use another path".to_owned()).to_response();
    assert!(!feedback.approved);
    assert!(!feedback.auto_confirm);
    assert_eq!(feedback.feedback, "use another path");
}

#[test]
fn render_prompt_layout_and_truncation() {
    let prompt = render_prompt("write_file", &json!({"path": "a.txt", "content": "x"}));
    assert!(prompt.starts_with("⚠ Approve write_file?"));
    assert!(prompt.contains("  path: a.txt"));
    assert!(prompt.contains("  content: x"));
    assert!(prompt.contains("(y/yes, a/always, n/no, or feedback)"));
    assert!(prompt.ends_with("Approve? (y/a/n)"));

    let long = "x".repeat(300);
    let prompt = render_prompt("write_file", &json!({"path": long}));
    let truncated = format!("  path: {}...", "x".repeat(200));
    assert!(prompt.contains(&truncated));
}

#[tokio::test]
async fn permission_host_approves_once() {
    let (io, buffer) = interactive_io(&["y"]);
    let outcome = permission_host(io)
        .request_confirmation(&confirmation_request())
        .await;
    let response = response_of(outcome);
    assert!(response.approved);
    assert!(!response.auto_confirm);
    assert!(output(&buffer).contains("⚠ Approve write_file?"));
}

#[tokio::test]
async fn permission_host_always_sets_auto_confirm() {
    let (io, _buffer) = interactive_io(&["a"]);
    let outcome = permission_host(io)
        .request_confirmation(&confirmation_request())
        .await;
    let response = response_of(outcome);
    assert!(response.approved);
    assert!(response.auto_confirm);
}

#[tokio::test]
async fn permission_host_rejects() {
    let (io, _buffer) = interactive_io(&["n"]);
    let outcome = permission_host(io)
        .request_confirmation(&confirmation_request())
        .await;
    let response = response_of(outcome);
    assert!(!response.approved);
    assert!(!response.auto_confirm);
    assert_eq!(response.feedback, "");
}

#[tokio::test]
async fn permission_host_returns_feedback() {
    let (io, _buffer) = interactive_io(&["use another path"]);
    let outcome = permission_host(io)
        .request_confirmation(&confirmation_request())
        .await;
    let response = response_of(outcome);
    assert!(!response.approved);
    assert_eq!(response.feedback, "use another path");
}

#[tokio::test]
async fn permission_host_eof_fails_closed() {
    let (io, _buffer) = interactive_io(&[]);
    let outcome = permission_host(io)
        .request_confirmation(&confirmation_request())
        .await;
    assert_eq!(outcome, ConfirmationOutcome::Failed);
}

#[tokio::test]
async fn permission_host_non_interactive_fails_without_prompt() {
    let (io, buffer) = scripted_io(&["y"]);
    let outcome = permission_host(io)
        .request_confirmation(&confirmation_request())
        .await;
    assert_eq!(outcome, ConfirmationOutcome::Failed);
    assert_eq!(output(&buffer), "");
}

// --- session store --------------------------------------------------------

#[test]
fn session_store_new_session_writes_python_shape() {
    let temp = TempDir::new().unwrap();
    let home = OpenJiuwenHome::from_root(temp.path());
    let store = SessionStore::new(&home);
    let record = store.new_session("cli-test", "gpt-4o").unwrap();
    assert_eq!(record.session_id, "cli-test");
    assert_eq!(record.model, "gpt-4o");

    let path = store.path_for("cli-test");
    assert!(path.ends_with("sessions/cli-test.json"));
    let json: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    assert_object_keys(&json, &["session_id", "model", "created_at", "messages"]);
    assert_eq!(json["session_id"], "cli-test");
    assert_eq!(json["model"], "gpt-4o");
    assert_eq!(json["messages"], json!([]));
}

#[test]
fn session_store_appends_messages_and_preserves_created_at() {
    let temp = TempDir::new().unwrap();
    let home = OpenJiuwenHome::from_root(temp.path());
    let store = SessionStore::new(&home);
    let record = store.new_session("cli-test", "gpt-4o").unwrap();

    store.append_message("cli-test", "user", "hi").unwrap();
    store
        .append_message("cli-test", "assistant", "hello")
        .unwrap();

    let loaded = store.load("cli-test").unwrap().unwrap();
    assert_eq!(loaded.created_at, record.created_at);
    assert_eq!(loaded.messages.len(), 2);
    assert_eq!(loaded.messages[0].role, "user");
    assert_eq!(loaded.messages[0].content, "hi");
    assert_eq!(loaded.messages[1].role, "assistant");
    assert_eq!(loaded.messages[1].content, "hello");
}

#[test]
fn session_store_load_unknown_is_none() {
    let temp = TempDir::new().unwrap();
    let home = OpenJiuwenHome::from_root(temp.path());
    let store = SessionStore::new(&home);
    assert!(store.load("cli-unknown").unwrap().is_none());
}

// --- renderer -------------------------------------------------------------

#[tokio::test]
async fn render_accumulates_llm_output() {
    let (io, buffer) = scripted_io(&[]);
    let mut state = RenderState::new();
    state
        .render(
            &AgentChunk::new("llm_output", 0, json!({"content": "hello "})),
            Some(&io),
            false,
        )
        .await
        .unwrap();
    state
        .render(
            &AgentChunk::new("llm_output", 1, json!({"content": "world"})),
            Some(&io),
            false,
        )
        .await
        .unwrap();
    assert_eq!(state.text, "hello world");
    assert_eq!(state.answer(), "hello world");
    assert_eq!(output(&buffer), "hello world");
}

#[tokio::test]
async fn render_answer_fallback_when_nothing_streamed() {
    let (io, buffer) = scripted_io(&[]);
    let mut state = RenderState::new();
    let chunk = AgentChunk::new(
        "answer",
        0,
        json!({"output": "fallback", "result_type": "interrupt"}),
    );
    state.render(&chunk, Some(&io), false).await.unwrap();
    assert_eq!(state.answer(), "fallback");
    assert_eq!(state.result_type(), "interrupt");
    assert!(output(&buffer).contains("fallback"));
}

#[tokio::test]
async fn render_tool_call_and_result() {
    let (io, buffer) = scripted_io(&[]);
    let mut state = RenderState::new();
    let call = AgentChunk::new(
        "tool_call",
        0,
        json!({"name": "bash", "arguments": {"command": "ls"}}),
    );
    state.render(&call, Some(&io), false).await.unwrap();
    let result = AgentChunk::new("tool_result", 1, json!({"result": "ok"}));
    state.render(&result, Some(&io), false).await.unwrap();

    let text = output(&buffer);
    assert!(text.contains("● bash("));
    assert!(text.contains("ls"));
    assert!(text.contains("  ⎿  ok\n\n"));
}

#[tokio::test]
async fn render_reasoning_is_verbose_only() {
    let (io, buffer) = scripted_io(&[]);
    let chunk = AgentChunk::new("llm_reasoning", 0, json!({"content": "thinking"}));
    let mut state = RenderState::new();
    state.render(&chunk, Some(&io), false).await.unwrap();
    assert_eq!(output(&buffer), "");

    let (io, buffer) = scripted_io(&[]);
    let mut state = RenderState::new();
    state.render(&chunk, Some(&io), true).await.unwrap();
    assert!(output(&buffer).contains("thinking"));
}

#[tokio::test]
async fn render_interaction_recorded_not_printed() {
    let (io, buffer) = scripted_io(&[]);
    let mut state = RenderState::new();
    let chunk = AgentChunk::new(
        "__interaction__",
        0,
        json!({"id": "ask-1", "value": {"question": "q"}}),
    );
    state.render(&chunk, Some(&io), false).await.unwrap();
    assert_eq!(state.interactions.len(), 1);
    assert_eq!(state.interactions[0]["id"], "ask-1");
    assert_eq!(output(&buffer), "");
}

#[tokio::test]
async fn render_captures_usage() {
    let (io, _buffer) = scripted_io(&[]);
    let mut state = RenderState::new();
    let chunk = AgentChunk::new(
        "llm_usage",
        0,
        json!({"usage_metadata": {"total_tokens": 5}}),
    );
    state.render(&chunk, Some(&io), false).await.unwrap();
    assert_eq!(state.usage, Some(json!({"total_tokens": 5})));
}

#[tokio::test]
async fn render_records_unknown_kind() {
    let (io, buffer) = scripted_io(&[]);
    let mut state = RenderState::new();
    let chunk = AgentChunk::new("mystery", 0, json!({}));
    state.render(&chunk, Some(&io), false).await.unwrap();
    assert_eq!(state.chunk_types, vec!["mystery".to_owned()]);
    assert_eq!(output(&buffer), "");
}

#[tokio::test]
async fn render_without_output_writes_nothing() {
    let (_io, buffer) = scripted_io(&[]);
    let mut state = RenderState::new();
    let chunk = AgentChunk::new("llm_output", 0, json!({"content": "hello"}));
    state.render(&chunk, None, false).await.unwrap();
    assert_eq!(state.text, "hello");
    assert_eq!(output(&buffer), "");
}

// --- one-shot runner ------------------------------------------------------

#[tokio::test]
async fn run_with_model_text_streams_answer() {
    let temp = TempDir::new().unwrap();
    let options = test_options(temp.path());
    options.ensure_workspace().unwrap();
    let (io, buffer) = scripted_io(&[]);
    let args = RunArgs {
        prompt: Some("hello".to_owned()),
        output_format: OutputFormat::Text,
    };
    let outcome = run_with_model(
        &options,
        &args,
        &io,
        Arc::new(ScriptedModel::text("scripted answer")),
    )
    .await
    .unwrap();

    assert!(output(&buffer).contains("scripted answer"));
    assert_ne!(outcome.answer, "");
    assert!(outcome.session_id.starts_with("cli-"));
    assert!(outcome.chunks > 0);
}

#[tokio::test]
async fn run_with_model_json_emits_one_object() {
    let temp = TempDir::new().unwrap();
    let options = test_options(temp.path());
    options.ensure_workspace().unwrap();
    let (io, buffer) = scripted_io(&[]);
    let args = RunArgs {
        prompt: Some("hello".to_owned()),
        output_format: OutputFormat::Json,
    };
    let outcome = run_with_model(
        &options,
        &args,
        &io,
        Arc::new(ScriptedModel::text("scripted answer")),
    )
    .await
    .unwrap();

    let text = output(&buffer);
    let value: Value = serde_json::from_str(text.trim()).unwrap();
    assert_object_keys(&value, &["answer", "result_type", "usage", "session_id"]);
    assert_eq!(value["answer"], outcome.answer);
    assert_eq!(value["session_id"], outcome.session_id);
}

#[tokio::test]
async fn run_with_model_stream_json_emits_lines() {
    let temp = TempDir::new().unwrap();
    let options = test_options(temp.path());
    options.ensure_workspace().unwrap();
    let (io, buffer) = scripted_io(&[]);
    let args = RunArgs {
        prompt: Some("hello".to_owned()),
        output_format: OutputFormat::StreamJson,
    };
    let outcome = run_with_model(
        &options,
        &args,
        &io,
        Arc::new(ScriptedModel::text("scripted answer")),
    )
    .await
    .unwrap();

    let text = output(&buffer);
    let lines: Vec<&str> = text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .collect();
    assert!(lines.len() >= 2, "expected chunk lines plus a result line");
    let values: Vec<Value> = lines
        .iter()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    for value in &values[..values.len() - 1] {
        assert_object_keys(value, &["type", "index", "payload"]);
    }
    let last = values.last().unwrap();
    assert_eq!(last["type"], "result");
    assert_object_keys(last, &["answer", "result_type", "usage", "session_id"]);
    assert_eq!(last["session_id"], outcome.session_id);
}

#[tokio::test]
async fn run_with_model_rejects_empty_prompt() {
    let temp = TempDir::new().unwrap();
    let options = test_options(temp.path());
    options.ensure_workspace().unwrap();
    let (io, _buffer) = scripted_io(&[]);
    let args = RunArgs {
        prompt: Some("   ".to_owned()),
        output_format: OutputFormat::Text,
    };
    let error = run_with_model(
        &options,
        &args,
        &io,
        Arc::new(ScriptedModel::text("unused")),
    )
    .await
    .unwrap_err();
    assert!(error.to_string().contains("A prompt argument is required"));
}

// --- REPL -----------------------------------------------------------------

#[tokio::test]
async fn repl_help_and_exit() {
    let temp = TempDir::new().unwrap();
    let options = test_options(temp.path());
    options.ensure_workspace().unwrap();
    let (mut repl, buffer) = build_repl(&options, &["/help", "/exit"]).await;
    repl.run().await.unwrap();

    let text = output(&buffer);
    for needle in ["/help", "/exit", "/mode", "/status", "Goodbye!"] {
        assert!(text.contains(needle), "missing {needle} in REPL output");
    }
}

#[tokio::test]
async fn repl_streams_one_turn() {
    let temp = TempDir::new().unwrap();
    let options = test_options(temp.path());
    options.ensure_workspace().unwrap();
    let (mut repl, buffer) = build_repl(&options, &["hello", "/exit"]).await;
    repl.run().await.unwrap();

    let text = output(&buffer);
    assert!(text.contains("You> "));
    assert!(text.contains("scripted answer"));
    assert!(text.contains("Goodbye!"));
}

#[tokio::test]
async fn repl_new_session() {
    let temp = TempDir::new().unwrap();
    let options = test_options(temp.path());
    options.ensure_workspace().unwrap();
    let (mut repl, buffer) = build_repl(&options, &["/new", "/exit"]).await;
    repl.run().await.unwrap();

    let text = output(&buffer);
    assert!(text.contains("Started a new session: cli-"));
}

#[tokio::test]
async fn repl_mode_plan() {
    let temp = TempDir::new().unwrap();
    let options = test_options(temp.path());
    options.ensure_workspace().unwrap();
    let (mut repl, buffer) = build_repl(&options, &["/mode plan", "/exit"]).await;
    repl.run().await.unwrap();

    let text = output(&buffer);
    assert!(text.contains("Mode: plan"));
}

#[tokio::test]
async fn repl_mode_usage_for_bogus() {
    let temp = TempDir::new().unwrap();
    let options = test_options(temp.path());
    options.ensure_workspace().unwrap();
    let (mut repl, buffer) = build_repl(&options, &["/mode bogus", "/exit"]).await;
    repl.run().await.unwrap();

    let text = output(&buffer);
    assert!(text.contains("Usage: /mode <normal|plan>"));
}

#[tokio::test]
async fn repl_unknown_command_hint() {
    let temp = TempDir::new().unwrap();
    let options = test_options(temp.path());
    options.ensure_workspace().unwrap();
    let (mut repl, buffer) = build_repl(&options, &["/bogus", "/exit"]).await;
    repl.run().await.unwrap();

    let text = output(&buffer);
    assert!(text.contains("Unknown command: /bogus"));
    assert!(text.contains("Type /help to see available commands."));
}

#[tokio::test]
async fn repl_interrupt_aborts_once_then_exits() {
    let temp = TempDir::new().unwrap();
    let options = test_options(temp.path());
    options.ensure_workspace().unwrap();
    let (repl, _buffer) = build_repl(&options, &[]).await;

    assert_eq!(repl.on_interrupt(), InterruptAction::Abort);
    assert_eq!(repl.on_interrupt(), InterruptAction::Exit);
    assert_eq!(repl.on_interrupt(), InterruptAction::Exit);
    // Further presses are harmless: abort is idempotent.
    assert_eq!(repl.on_interrupt(), InterruptAction::Exit);
    assert_eq!(repl.on_interrupt(), InterruptAction::Exit);
}

#[tokio::test]
async fn repl_session_id_shape() {
    let temp = TempDir::new().unwrap();
    let options = test_options(temp.path());
    options.ensure_workspace().unwrap();
    let (repl, _buffer) = build_repl(&options, &[]).await;

    let id = repl.session_id();
    assert!(id.starts_with("cli-"));
    let suffix = &id["cli-".len()..];
    assert_eq!(suffix.len(), 8);
    assert!(suffix.chars().all(|ch| ch.is_ascii_hexdigit()));
}
