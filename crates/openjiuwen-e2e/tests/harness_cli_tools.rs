//! Extended harness tools driven the way the `openjiuwen` binary drives them.
//!
//! Three seams are composed here that no single crate's unit suite can combine:
//!
//! - the `fetch_webpage` built-in, mounted through [`ToolRegistry`] and invoked
//!   through the umbrella's `Tool` trait against a wiremock loopback server, so
//!   the real `reqwest` transport, the HTML/JSON rendering and the failure
//!   envelope are all exercised;
//! - the persisted cron scheduler through the model-facing `cron` tool plus the
//!   scheduler's own `list`/`due`/`cancel` surface, with the job's `next_run_at`
//!   fed straight back into `due` instead of reading the wall clock (this crate
//!   has no `chrono` dependency and never names a `chrono` type);
//! - the CLI's colour gating (`--no-color`, `NO_COLOR`, tty detection), its
//!   `Style` byte output, and the REPL slash-command grammar over a scripted
//!   terminal double.
//!
//! Everything is offline and deterministic: loopback sockets only, an injected
//! cron clock, scripted models, and temp directories that own every file.

#![allow(clippy::unwrap_used, clippy::expect_used)]

mod support;

use std::collections::VecDeque;
use std::path::Path;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use openjiuwen::core::common::config::OpenJiuwenHome;
use openjiuwen::core::common::error::Result as OjResult;
use openjiuwen::core::foundation::llm::Model;
use openjiuwen::core::foundation::tool::{Tool, ToolContext};
use openjiuwen_harness::clap::Parser;
use openjiuwen_harness::cli::paths::{color_enabled, NO_COLOR};
use openjiuwen_harness::cli::style::Style;
use openjiuwen_harness::cli::{
    build_agent, Cli, CliIo, CliOptions, LineReader, MapEnv, Repl, ReplCommand, Writer,
    DEFAULT_MAX_ITERATIONS, DEFAULT_MAX_TOKENS, DEFAULT_MODEL, DEFAULT_PROVIDER,
};
use openjiuwen_harness::tools::cron::{CronJob, CronScheduler, CronTool};
use openjiuwen_harness::tools::{BuiltinTool, ToolRegistry};
use openjiuwen_harness::workspace::Workspace;
use openjiuwen_harness::AgentMode;
use openjiuwen_test_support::wiremock::matchers::{method, path};
use openjiuwen_test_support::wiremock::{Mock, MockServer, ResponseTemplate};
use openjiuwen_test_support::ScriptedModel;
use serde_json::{json, Value};
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

/// Builds an interactive terminal double plus the buffer it writes to.
fn terminal(lines: &[&str]) -> (CliIo, Arc<Mutex<String>>) {
    let buffer = Arc::new(Mutex::new(String::new()));
    let reader = Box::new(ScriptedReader {
        lines: lines.iter().map(|line| (*line).to_owned()).collect(),
    });
    let writer = Box::new(SharedWriter {
        buffer: Arc::clone(&buffer),
    });
    (CliIo::with_interactive(reader, writer), buffer)
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

/// Builds the `fetch_webpage` tool over the throwaway workspace at `dir`.
fn fetch_webpage(dir: &Path) -> Arc<dyn Tool> {
    ToolRegistry::new(Workspace::new(dir))
        .with_tools([BuiltinTool::FetchWebpage])
        .build()
        .into_iter()
        .find(|tool| tool.card().name == "fetch_webpage")
        .expect("the fetch tool is mounted")
}

/// Invokes `tool` with `inputs` and unwraps the payload envelope.
async fn invoke(tool: &dyn Tool, inputs: Value) -> Value {
    support::within("tool invoke", tool.invoke(inputs, &ToolContext::new()))
        .await
        .expect("a tool reports failures in the payload, never as Err")
}

/// Runs a scripted REPL transcript and returns the rendered text plus the model
/// call count.
async fn repl_transcript(inputs: &[&str], colour: bool) -> (String, usize) {
    let temp = TempDir::new().expect("temp dir");
    let options = options(temp.path());
    options.ensure_workspace().expect("workspace");
    let (io, buffer) = terminal(inputs);
    let io = io.with_color(colour);
    let (model, probe) = scripted("scripted answer");
    let agent = support::within("agent builds", build_agent(model, &options, &io))
        .await
        .expect("the DeepAgent builds from the options");
    let mut repl = Repl::with_io(agent, options, io);
    support::within("repl transcript", repl.run())
        .await
        .expect("the scripted transcript terminates cleanly");
    (rendered(&buffer), probe.call_count())
}

// --- fetch_webpage: loopback HTTP -----------------------------------------
/// The served document: a title, a heading, an entity and script/style noise.
const DOC_HTML: &str = "<html><head><title>Doc</title><style>b{color:red}</style></head>\
     <body><h1>Hi</h1><p>World &amp; friends</p><script>x()</script></body></html>";

#[tokio::test]
async fn fetch_html_page_is_extracted_from_a_loopback_server() {
    let dir = TempDir::new().expect("workspace");
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/doc"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(DOC_HTML, "text/html; charset=utf-8"))
        .mount(&server)
        .await;

    let url = format!("{}/doc", server.uri());
    let payload = invoke(fetch_webpage(dir.path()).as_ref(), json!({ "url": url })).await;

    assert_eq!(payload["success"], json!(true), "fetch failed: {payload}");
    // The envelope always carries all six keys; this one has no extra content.
    assert!(
        payload.get("extracted_content").is_some(),
        "the payload envelope keeps its fixed keys: {payload}"
    );
    assert_eq!(payload["error"], Value::Null, "a success carries no error");

    let data = &payload["data"];
    assert_eq!(data["status"], json!(200));
    assert_eq!(data["url"], json!(url));
    assert_eq!(data["final_url"], json!(url));
    // `<title>` is lifted out of the HTML and the body is flattened to text:
    // no tags, no script/style, the entity decoded.
    assert_eq!(data["title"], json!("Doc"));
    assert_eq!(data["content"], json!("Hi\nWorld & friends"));
    assert_eq!(data["truncated"], json!(false));
    let content_type = data["content_type"].as_str().expect("content type");
    assert!(
        content_type.starts_with("text/html"),
        "content type: {content_type}"
    );
}

#[tokio::test]
async fn fetch_json_endpoint_is_pretty_printed_without_a_title() {
    let dir = TempDir::new().expect("workspace");
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_raw(r#"{"name":"value","count":2}"#, "application/json"),
        )
        .mount(&server)
        .await;

    let url = format!("{}/api", server.uri());
    let payload = invoke(fetch_webpage(dir.path()).as_ref(), json!({ "url": url })).await;

    assert_eq!(payload["success"], json!(true), "fetch failed: {payload}");
    let data = &payload["data"];
    assert_eq!(data["status"], json!(200));
    let content = data["content"].as_str().expect("rendered json content");
    // Pretty-printed, so the separators are the JSON ones, not the wire ones.
    assert!(
        content.contains("\"name\": \"value\""),
        "content: {content}"
    );
    assert!(content.contains("\"count\": 2"), "content: {content}");
    // A JSON document has no title, so the key is absent rather than null.
    assert!(data.get("title").is_none(), "json has no title: {data}");
}

#[tokio::test]
async fn fetch_error_status_becomes_a_failure_payload() {
    let dir = TempDir::new().expect("workspace");
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/boom"))
        .respond_with(ResponseTemplate::new(500).set_body_raw("upstream exploded", "text/plain"))
        .mount(&server)
        .await;

    let url = format!("{}/boom", server.uri());
    let payload = invoke(fetch_webpage(dir.path()).as_ref(), json!({ "url": url })).await;

    // A 5xx is a payload failure, not an `Err` and not a hang.
    assert_eq!(payload["success"], json!(false), "payload: {payload}");
    let error = payload["error"]
        .as_str()
        .expect("a failure carries a message");
    assert!(
        error.contains("500"),
        "the status is in the message: {error}"
    );
    assert!(error.contains(&url), "the message names the url: {error}");
    assert_eq!(payload["data"]["status"], json!(500));
    assert_eq!(payload["data"]["url"], json!(url));
}

// --- cron: persisted scheduling with an injected clock --------------------
/// The usage-independent prompt the cron job will carry.
const CRON_PROMPT: &str = "check the build";

#[tokio::test]
async fn cron_job_round_trips_through_create_list_due_cancel() {
    let dir = TempDir::new().expect("cron directory");
    let scheduler = CronScheduler::new(dir.path());
    let tool = CronTool::new(Arc::clone(&scheduler), "en");

    // Create through the model-facing `Tool` seam (the wall clock is only ever
    // read inside the tool; nothing here asserts on the absolute instant).
    let created = invoke(
        &tool,
        json!({ "action": "create", "cron": "*/15 * * * *", "prompt": CRON_PROMPT }),
    )
    .await;
    assert_eq!(created["success"], json!(true), "create: {created}");
    let job: CronJob = serde_json::from_value(created["data"]["job"].clone())
        .expect("the created job deserializes into CronJob");

    // `list` agrees with the create response, and so does a fresh scheduler over
    // the same directory: the job is persisted, not held in memory.
    let listed = support::within("cron list", scheduler.list())
        .await
        .expect("list reads the job file");
    assert_eq!(listed.len(), 1, "one job is scheduled: {listed:?}");
    assert_eq!(listed[0].id, job.id);
    assert_eq!(listed[0].cron, "*/15 * * * *");
    assert_eq!(listed[0].prompt, CRON_PROMPT);
    assert_eq!(listed[0], job, "the create payload is the stored job");

    let reopened = CronScheduler::new(dir.path());
    let seen = support::within("cron list after reopen", reopened.list())
        .await
        .expect("a second scheduler reads the same file");
    assert_eq!(seen, listed, "the job survives a second scheduler instance");

    // The first occurrence fires: `due` at exactly the job's `next_run_at`
    // yields the job, and the same instant is not due again afterwards.
    let early = support::within("cron due before", reopened.due(job.next_run_at))
        .await
        .expect("due reads the job file");
    assert_eq!(early.len(), 1, "the job is due at exactly its next run");
    assert_eq!(early[0].id, job.id);
    assert_eq!(early[0].prompt, CRON_PROMPT);

    // Firing advances the stored occurrence by one cron period.
    let after = support::within("cron list after due", scheduler.list())
        .await
        .expect("list reads the advanced job");
    assert_eq!(after.len(), 1);
    let advanced = after[0].next_run_at;
    assert!(
        advanced > job.next_run_at,
        "due must advance next_run_at: {advanced:?} vs {:?}",
        job.next_run_at
    );

    let repeat = support::within("cron due repeat", scheduler.due(job.next_run_at))
        .await
        .expect("due reads the job file");
    assert!(
        repeat.is_empty(),
        "the occurrence fired only once: {repeat:?}"
    );

    // Cancelling removes it everywhere.
    let cancelled = support::within("cron cancel", scheduler.cancel(&job.id))
        .await
        .expect("cancel reads the job file");
    assert_eq!(
        cancelled.map(|job| job.id),
        Some(job.id.clone()),
        "cancel reports the removed job"
    );
    assert!(
        support::within("cron list after cancel", scheduler.list())
            .await
            .expect("list reads the empty job file")
            .is_empty(),
        "cancelling leaves no job behind"
    );
    let reopened_after = support::within("cron list reopened after cancel", reopened.list())
        .await
        .expect("the second scheduler sees the cancellation");
    assert!(
        reopened_after.is_empty(),
        "cancel is persisted: {reopened_after:?}"
    );

    // And a cancelled job never fires.
    let never = support::within("cron due after cancel", reopened.due(advanced))
        .await
        .expect("due reads the empty job file");
    assert!(never.is_empty(), "cancelled jobs are not due: {never:?}");
}

// --- --no-color ------------------------------------------------------------
/// The documented usage line a malformed `/mode` prints.
const MODE_USAGE: &str = "Usage: /mode <normal|plan>";

#[test]
fn no_color_flag_is_global_and_gates_the_style_output() {
    // The flag is global: clap accepts it before and after the subcommand, and
    // it is a field of `Cli` rather than of `CliOptions`.
    for argv in [
        ["openjiuwen", "--no-color", "run", "hi"].as_slice(),
        ["openjiuwen", "run", "hi", "--no-color"].as_slice(),
        ["openjiuwen", "--no-color", "chat"].as_slice(),
        ["openjiuwen", "chat", "--no-color"].as_slice(),
    ] {
        let cli = Cli::try_parse_from(argv).expect("the global flag parses");
        assert!(cli.no_color, "--no-color must be set for {argv:?}");
    }
    let cli = Cli::try_parse_from(["openjiuwen", "run", "hi"]).expect("a plain command line");
    assert!(!cli.no_color, "colour is not disabled by default");

    // Colour gating: tty AND no flag AND `NO_COLOR` unset.
    let clean = MapEnv::new();
    let no_color = MapEnv::from_pairs([(NO_COLOR, "1")]);
    let table = [
        (false, false, true, true),
        (true, false, true, false),
        (false, true, true, false),
        (false, false, false, false),
        (true, true, true, false),
    ];
    for (flag, env_set, is_terminal, expected) in table {
        let env = if env_set { &no_color } else { &clean };
        assert_eq!(
            color_enabled(flag, env, is_terminal),
            expected,
            "flag={flag} NO_COLOR={env_set} tty={is_terminal}"
        );
    }

    // The handle mirrors the gate: every constructor starts colourless.
    let (io, _buffer) = terminal(&[]);
    assert!(!io.is_color(), "a scripted terminal starts plain");
    assert!(io.clone().with_color(true).is_color(), "colour is opt-in");
    assert!(!io.clone().with_color(false).is_color());

    // `Style` is where the decision becomes bytes: plain output is unchanged and
    // carries no escape sequence, coloured output is wrapped in the code.
    let plain = Style::plain();
    let disabled = Style::new(false);
    let enabled = Style::new(true);
    assert!(!plain.enabled() && !disabled.enabled() && enabled.enabled());
    assert_eq!(plain.cyan("x"), "x");
    assert_eq!(disabled.cyan("x"), "x");
    assert_eq!(enabled.cyan("x"), "\x1b[36mx\x1b[0m");
    assert_eq!(
        enabled.cyan("x").matches('\x1b').count(),
        2,
        "one opening code and one reset"
    );
    assert_eq!(plain.bold("hi"), "hi");
    assert_eq!(enabled.bold("hi"), "\x1b[1mhi\x1b[0m");
    assert_eq!(enabled.paint("\x1b[31m", ""), "", "blank lines stay blank");
    assert!(
        !plain.bold_cyan("OpenJiuwen").contains('\x1b'),
        "plain styles never emit escapes"
    );
    assert!(
        enabled.bold_cyan("OpenJiuwen").contains('\x1b'),
        "the enabled style emits escapes"
    );
}

// --- slash commands and the scripted transcript ---------------------------
#[tokio::test]
async fn slash_commands_map_onto_repl_actions() {
    let temp = TempDir::new().expect("temp dir");
    let options = options(temp.path());
    options.ensure_workspace().expect("workspace");
    let (io, _buffer) = terminal(&[]);
    let (model, _probe) = scripted("unused");
    let agent = support::within("agent builds", build_agent(model, &options, &io))
        .await
        .expect("the DeepAgent builds from the options");
    let repl = Repl::with_io(agent, options, io);

    let cases = [
        ("/help", ReplCommand::Help),
        ("/clear", ReplCommand::Clear),
        ("/compact", ReplCommand::Compact),
        ("/mode plan", ReplCommand::Mode(AgentMode::Plan)),
        ("/mode", ReplCommand::Usage(MODE_USAGE)),
        ("/mode turbo", ReplCommand::Usage(MODE_USAGE)),
        ("!echo hi", ReplCommand::Shell("echo hi".to_owned())),
        ("!", ReplCommand::Empty),
        (
            "/unknowncmd",
            ReplCommand::Unknown("/unknowncmd".to_owned()),
        ),
        ("hello world", ReplCommand::Prompt("hello world".to_owned())),
        ("   ", ReplCommand::Empty),
    ];
    for (line, expected) in cases {
        assert_eq!(repl.handle_input(line), expected, "line {line:?}");
    }
}

#[tokio::test]
async fn scripted_repl_turn_is_escape_free_without_colour() {
    let inputs = ["/help", "/status", "hello there", "/exit"];

    let (plain, calls) = repl_transcript(&inputs, false).await;
    assert!(
        !plain.contains('\x1b'),
        "colour is off, so the transcript carries no ANSI escapes: {plain:?}"
    );
    assert!(plain.contains("OpenJiuwen CLI v"), "banner: {plain}");
    assert!(plain.contains("Available commands:"), "help table: {plain}");
    assert!(
        plain.contains("/mode <normal|plan>"),
        "the help table lists the /mode usage: {plain}"
    );
    assert!(plain.contains("You> "), "the REPL prompts: {plain}");
    assert!(
        plain.contains("scripted answer"),
        "the conversational turn is rendered: {plain}"
    );
    assert!(
        plain.contains("Goodbye!"),
        "the exit path says goodbye: {plain}"
    );
    assert_eq!(calls, 1, "exactly one typed line reaches the model");

    // The same transcript with colour enabled emits escapes, proving the
    // escape-free assertion above is about the gate rather than about the text.
    let (coloured, coloured_calls) = repl_transcript(&inputs, true).await;
    assert!(
        coloured.contains('\x1b'),
        "colour on must emit escapes: {coloured:?}"
    );
    assert!(coloured.contains("scripted answer"), "answer: {coloured}");
    assert_eq!(coloured_calls, 1, "one turn per typed prompt");
}
