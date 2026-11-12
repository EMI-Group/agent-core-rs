//! CLI scenario: the harness CLI layer driven offline through its injected-model
//! seam, plus the process boundary where the crate graph allows it.
//!
//! The real entry points (`run_cli`/`run`) are never called here: they resolve
//! their options from the process environment and the home directory, and they
//! build a live model client. Instead the one-shot runner is driven through
//! [`run_with_model`], which takes the resolved [`CliOptions`], the run
//! arguments, an injected terminal ([`CliIo`] over a scripted reader and a
//! buffered writer) and a [`ScriptedModel`], so every case is offline,
//! deterministic and leaves nothing outside its temp directory.
//!
//! The clap `--help`/`--version` verdicts are answered by clap inside the
//! `openjiuwen-cli` binary; this crate cannot reach them because `clap` is not a
//! dependency here and is not re-exported by `openjiuwen-harness` or the
//! `openjiuwen` umbrella (see the note on the last two tests).
#![allow(clippy::unwrap_used, clippy::expect_used)]

mod support;

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use openjiuwen::core::common::config::OpenJiuwenHome;
use openjiuwen::core::common::error::{ErrorKind, Result as OjResult};
use openjiuwen::core::foundation::llm::Model;
use openjiuwen::harness::cli::{
    run_with_model, Cli, CliIo, CliOptions, Command, Dispatch, LineReader, MapEnv, OutputFormat,
    RunArgs, Writer, DEFAULT_MAX_ITERATIONS, DEFAULT_MAX_TOKENS, DEFAULT_MODEL, DEFAULT_PROVIDER,
    MIN_MAX_TOKENS, OPENJIUWEN_HOME, OPENJIUWEN_MAX_TOKENS,
};
use openjiuwen_test_support::ScriptedModel;
use serde_json::Value;
use tempfile::TempDir;

// --- scripted terminal ----------------------------------------------------
/// [`LineReader`] answering from a fixed script, then reporting end of input.
struct ScriptedReader {
    /// Lines still to hand out, front to back.
    lines: VecDeque<String>,
}

#[async_trait]
impl LineReader for ScriptedReader {
    async fn read_line(&mut self) -> OjResult<Option<String>> {
        Ok(self.lines.pop_front())
    }
}

/// [`Writer`] appending everything it receives to a shared buffer.
struct SharedWriter {
    /// Text written so far, shared with the test body.
    buffer: Arc<Mutex<String>>,
}

#[async_trait]
impl Writer for SharedWriter {
    async fn write(&mut self, text: &str) -> OjResult<()> {
        self.buffer
            .lock()
            .expect("writer buffer lock")
            .push_str(text);
        Ok(())
    }

    async fn flush(&mut self) -> OjResult<()> {
        Ok(())
    }
}

/// Builds the pair of terminal handle and inspected buffer.
fn terminal(lines: &[&str], interactive: bool) -> (CliIo, Arc<Mutex<String>>) {
    let buffer = Arc::new(Mutex::new(String::new()));
    let reader = Box::new(ScriptedReader {
        lines: lines.iter().map(|line| (*line).to_owned()).collect(),
    });
    let writer = Box::new(SharedWriter {
        buffer: Arc::clone(&buffer),
    });
    let io = if interactive {
        CliIo::with_interactive(reader, writer)
    } else {
        CliIo::new(reader, writer)
    };
    (io, buffer)
}

/// Returns everything written to `buffer` so far.
fn rendered(buffer: &Arc<Mutex<String>>) -> String {
    buffer.lock().expect("writer buffer lock").clone()
}

// --- fixtures -------------------------------------------------------------
/// Builds resolved options whose runtime state lives under `root`.
///
/// Every field is a struct literal on purpose: no `Default` impl, no process
/// environment read and no home-directory probe are involved.
fn options(root: &Path) -> CliOptions {
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

/// Builds a model answering `text` and a probe sharing its call history.
fn scripted(text: &str) -> (Arc<dyn Model>, ScriptedModel) {
    let model = ScriptedModel::text(text);
    let probe = model.clone();
    (Arc::new(model) as Arc<dyn Model>, probe)
}

/// Builds a command line with every option unset, as clap would with no args.
fn blank_cli() -> Cli {
    Cli {
        model: None,
        provider: None,
        api_key: None,
        api_base: None,
        workspace: None,
        verbose: false,
        command: None,
    }
}

// --- one-shot runner output formats ---------------------------------------
#[tokio::test]
async fn one_shot_json_output_reports_answer_and_session() {
    let temp = TempDir::new().expect("temp dir");
    let options = options(temp.path());
    options.ensure_workspace().expect("workspace");
    let (io, buffer) = terminal(&[], false);
    let (model, probe) = scripted("hi there");
    let args = RunArgs {
        prompt: Some("hello".to_owned()),
        output_format: OutputFormat::Json,
    };
    let outcome = support::within("json one-shot", run_with_model(&options, &args, &io, model))
        .await
        .expect("json run succeeds");

    // Parsing the whole trimmed buffer proves the run emitted exactly one JSON
    // document: a second line of log noise would make this fail.
    let value: Value = serde_json::from_str(rendered(&buffer).trim())
        .expect("json output is a single parseable document");
    let object = value.as_object().expect("json output is an object");
    for key in ["answer", "result_type", "usage", "session_id"] {
        assert!(object.contains_key(key), "missing key {key}: {value}");
    }
    assert_eq!(value["answer"], "hi there");
    assert_eq!(value["result_type"], "answer");
    assert!(
        value["usage"].is_object(),
        "usage must be a JSON object: {value}"
    );
    let session_id = value["session_id"]
        .as_str()
        .expect("session_id must be a string");
    assert!(!session_id.is_empty(), "session id must not be empty");
    assert!(
        session_id.starts_with("cli-"),
        "session id must carry the cli- prefix: {session_id}"
    );
    assert_eq!(session_id, outcome.session_id);
    assert_eq!(probe.call_count(), 1, "the model is called once per run");
}

#[tokio::test]
async fn one_shot_stream_json_emits_chunk_lines_and_a_result_frame() {
    let temp = TempDir::new().expect("temp dir");
    let options = options(temp.path());
    options.ensure_workspace().expect("workspace");
    let (io, buffer) = terminal(&[], false);
    let (model, probe) = scripted("hi there");
    let args = RunArgs {
        prompt: Some("hello".to_owned()),
        output_format: OutputFormat::StreamJson,
    };
    let outcome = support::within(
        "stream-json one-shot",
        run_with_model(&options, &args, &io, model),
    )
    .await
    .expect("stream-json run succeeds");

    let text = rendered(&buffer);
    let lines: Vec<&str> = text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .collect();
    assert!(
        lines.len() >= 2,
        "expected chunk lines plus a result line, got {lines:?}"
    );
    let values: Vec<Value> = lines
        .iter()
        .map(|line| serde_json::from_str(line).expect("every line is JSON"))
        .collect();

    let (chunks, last) = values.split_at(values.len() - 1);
    for chunk in chunks {
        let object = chunk
            .as_object()
            .expect("a streamed chunk line must be an object");
        assert!(
            chunk["type"].is_string(),
            "chunk line needs a string type: {chunk}"
        );
        for key in ["index", "payload"] {
            assert!(object.contains_key(key), "chunk line needs {key}: {chunk}");
        }
    }
    let kinds: Vec<&str> = chunks
        .iter()
        .filter_map(|chunk| chunk["type"].as_str())
        .collect();
    assert!(
        kinds.contains(&"llm_output"),
        "streamed text must appear as an llm_output line, got {kinds:?}"
    );

    let result = &last[0];
    assert_eq!(
        result["type"], "result",
        "the last line is the result frame"
    );
    assert_eq!(result["answer"], "hi there");
    assert_eq!(
        result["session_id"].as_str(),
        Some(outcome.session_id.as_str())
    );
    let count = values
        .iter()
        .filter(|value| value["type"] == "result")
        .count();
    assert_eq!(count, 1, "exactly one result frame is emitted");
    assert_eq!(probe.call_count(), 1, "the model is called once per run");
}

#[tokio::test]
async fn one_shot_text_output_streams_the_answer() {
    let temp = TempDir::new().expect("temp dir");
    let options = options(temp.path());
    options.ensure_workspace().expect("workspace");
    let (io, buffer) = terminal(&[], false);
    let (model, probe) = scripted("hi there");
    let args = RunArgs {
        prompt: Some("hello".to_owned()),
        output_format: OutputFormat::Text,
    };
    let outcome = support::within("text one-shot", run_with_model(&options, &args, &io, model))
        .await
        .expect("text run succeeds");

    let text = rendered(&buffer);
    assert!(text.contains("hi there"), "answer must be streamed: {text}");
    assert_eq!(outcome.answer, "hi there");
    assert_eq!(probe.call_count(), 1, "the model is called once per run");
    assert!(
        serde_json::from_str::<Value>(text.trim()).is_err(),
        "text output is not a JSON document: {text}"
    );
}

#[tokio::test]
async fn usage_errors_are_rejected_before_any_run() {
    // Grammar/option layer: an out-of-range token budget is a usage error.
    //
    // The clap rejection path (`UnknownArgument`/`InvalidSubcommand`, exit code
    // 2) is unreachable from this crate: `clap` is not a dependency and is not
    // re-exported, so `Cli::try_parse_from` cannot be called here. Those
    // verdicts are asserted in `crates/openjiuwen-harness/tests/cli.rs` and
    // `crates/openjiuwen-cli/tests/cli_smoke.rs`. The case below is the same
    // verdict class: a usage error rejected before any run.
    let temp = TempDir::new().expect("temp dir");
    let root = temp.path();
    let root_str = root.to_str().expect("utf-8 temp path");
    let env = MapEnv::from_pairs([(OPENJIUWEN_HOME, root_str), (OPENJIUWEN_MAX_TOKENS, "128")]);
    let error = CliOptions::resolve_with(&blank_cli(), &env, None, root)
        .expect_err("a token budget below the minimum must be rejected");
    assert_eq!(error.kind(), ErrorKind::Validation);
    let message = error.to_string();
    assert!(
        message.contains("max_tokens"),
        "the message must name max_tokens: {message}"
    );
    assert!(
        message.contains(&MIN_MAX_TOKENS.to_string()),
        "the message must name the minimum {MIN_MAX_TOKENS}: {message}"
    );

    // Runner layer: a blank prompt fails validation before the agent is built,
    // so the model is never called and nothing is rendered.
    let options = options(root);
    let (io, buffer) = terminal(&[], false);
    let (model, probe) = scripted("hi there");
    let args = RunArgs {
        prompt: Some("   ".to_owned()),
        output_format: OutputFormat::Text,
    };
    let error = support::within("blank prompt", run_with_model(&options, &args, &io, model))
        .await
        .expect_err("a blank prompt must be rejected");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert_eq!(
        probe.call_count(),
        0,
        "validation happens before any model call"
    );
    assert!(
        rendered(&buffer).is_empty(),
        "a rejected run must render nothing: {:?}",
        rendered(&buffer)
    );
}

#[test]
fn version_and_help_are_answered_without_a_model() {
    // clap's `--help`/`--version` verdicts (`DisplayHelp`/`DisplayVersion`) are
    // answered inside the `openjiuwen-cli` binary and are NOT reachable here:
    // this crate has no `clap` dependency and the grammar type is not re-exported
    // with a `Parser` impl usable without it. Those verdicts live in
    // `crates/openjiuwen-harness/tests/cli.rs::clap_help_and_version` and
    // `crates/openjiuwen-cli/tests/cli_smoke.rs`. What IS reachable at this
    // boundary is that the grammar's data layer and dispatch need no model, no
    // credential and no network.
    let temp = TempDir::new().expect("temp dir");
    let root = temp.path();
    let root_str = root.to_str().expect("utf-8 temp path");
    let env = MapEnv::from_pairs([(OPENJIUWEN_HOME, root_str)]);
    let resolved = CliOptions::resolve_with(&blank_cli(), &env, None, root)
        .expect("blank options resolve to the documented defaults");
    assert_eq!(resolved.model, DEFAULT_MODEL);
    assert_eq!(resolved.provider, DEFAULT_PROVIDER);
    assert_eq!(resolved.max_iterations, DEFAULT_MAX_ITERATIONS);
    assert_eq!(resolved.max_tokens, DEFAULT_MAX_TOKENS);
    assert_eq!(
        resolved.workspace,
        root.join("workspace"),
        "a relative default workspace resolves against the injected cwd"
    );
    assert_eq!(
        resolved.home.root(),
        root,
        "the home directory is rooted at OPENJIUWEN_HOME"
    );

    // Dispatch is a pure function of the parsed command line plus one boolean.
    assert_eq!(
        blank_cli().dispatch(false),
        Dispatch::Run(RunArgs::default())
    );
    assert_eq!(blank_cli().dispatch(true), Dispatch::Chat);

    // The documented output-format spellings.
    assert_eq!(OutputFormat::default(), OutputFormat::Text);
    assert_eq!(OutputFormat::Text.as_str(), "text");
    assert_eq!(OutputFormat::Json.as_str(), "json");
    assert_eq!(OutputFormat::StreamJson.as_str(), "stream-json");

    // The grammar model is usable without clap parsing: a struct-literal `Cli`
    // dispatches to exactly the run arguments it was built with.
    let run_args = RunArgs {
        prompt: Some("hi".to_owned()),
        output_format: OutputFormat::Json,
    };
    let cli = Cli {
        command: Some(Command::Run(run_args.clone())),
        ..blank_cli()
    };
    assert_eq!(cli.dispatch(false), Dispatch::Run(run_args));
}

#[ignore = "the openjiuwen binary belongs to the openjiuwen-cli package; process-level smoke lives in crates/openjiuwen-cli/tests/cli_smoke.rs and the binary may be absent from a -p openjiuwen-e2e build graph"]
#[test]
fn binary_version_over_the_process_boundary() {
    // This is the only process-level assertion available here:
    // `env!("CARGO_BIN_EXE_openjiuwen")` is defined only for tests of the
    // package that owns the binary, so `openjiuwen-cli` asserts it and this
    // crate resolves the binary at runtime instead.
    let exe = std::env::current_exe().expect("the test binary has a path");
    let suffix = std::env::consts::EXE_SUFFIX;
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Some(profile) = exe.parent().and_then(Path::parent) {
        candidates.push(profile.join(format!("openjiuwen{suffix}")));
    }
    if let Some(deps) = exe.parent() {
        candidates.push(deps.join(format!("openjiuwen{suffix}")));
    }
    let found = candidates.iter().find(|path| path.is_file());
    let Some(binary) = found else {
        eprintln!(
            "skipping binary_version_over_the_process_boundary: no openjiuwen binary at {candidates:?}"
        );
        return;
    };

    let version = std::process::Command::new(binary)
        .arg("--version")
        .output()
        .expect("the openjiuwen binary should run");
    assert!(
        version.status.success(),
        "--version failed: {}",
        String::from_utf8_lossy(&version.stderr)
    );
    assert_eq!(version.status.code(), Some(0), "--version must exit 0");
    let version_stdout = String::from_utf8_lossy(&version.stdout);
    assert!(
        version_stdout.contains("openjiuwen"),
        "--version must name the binary: {version_stdout}"
    );
    assert!(
        version_stdout.contains(env!("CARGO_PKG_VERSION")),
        "--version must report the workspace version: {version_stdout}"
    );

    let help = std::process::Command::new(binary)
        .arg("--help")
        .output()
        .expect("the openjiuwen binary should run");
    assert!(
        help.status.success(),
        "--help failed: {}",
        String::from_utf8_lossy(&help.stderr)
    );
    let help_stdout = String::from_utf8_lossy(&help.stdout);
    // Assert documented subcommands, never exact banner wording.
    assert!(
        help_stdout.contains("run"),
        "--help must list the run subcommand: {help_stdout}"
    );
    assert!(
        help_stdout.contains("chat"),
        "--help must list the chat subcommand: {help_stdout}"
    );
}
