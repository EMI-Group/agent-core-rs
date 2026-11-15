//! Offline, process-level `run --resume` / `chat --resume` scenarios.
//!
//! Every case drives the real binary through a real loopback model server
//! ([`MockLlmServer`]), so the resumed turn takes the whole path a user's turn
//! takes: option resolution, the catalogue lookup, the kernel seed, the HTTP
//! request and the rendering of the chosen `-f` format.
//!
//! The three resumable sources are covered in the order the catalogue documents:
//!
//! 1. a **CLI session record** written by an earlier `run` — plain resume,
//!    `--continue`, and the ordering of the restored transcript inside the model
//!    request;
//! 2. a **persistent-agent session record** whose ending names an execution, so
//!    the resume traces the journal into a [`ResumePlan`] and seeds its history;
//! 3. an **execution journal** resolved by its own id, which also creates the CLI
//!    record of the session it belongs to.
//!
//! The persistent-agent fixtures are written with the harness API in the test
//! process ([`seed_persistent_state`]), because nothing in the CLI writes
//! journals: they are what *another* host persists.
//!
//! Typed failures close the suite: an unknown id, and a `--continue` on a home
//! with no session at all. `chat --resume` is covered through the REPL's stdin
//! EOF path, with the note that names both session ids.
//!
//! Nothing needs the network, credentials or a terminal: the mock server binds
//! loopback, every child runs in a temporary cwd/home/workspace and stdin is
//! `/dev/null`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod support;

use openjiuwen_core::foundation::prompt::BaseMessage;
use openjiuwen_harness::{
    EntryPayload, ExecutionIdentity, ExecutionLink, ExecutionLinkSource, ExecutionLog,
    FileLinkStore, FileSessionStore, LinkRelation, LogCapacity, ModelUsage, PersistentAgentLayout,
    PersistentSessionRecord, RecordEnding, SessionBinding, SessionStateStore, Workspace,
};
use openjiuwen_test_support::{MockLlmServer, ScriptedResponse, TEST_API_KEY};
use serde_json::Value;
use support::{assert_exit, assert_no_ansi, assert_success, parse_json, stderr, Sandbox};

/// The answer the scripted model returns for every request.
const ANSWER: &str = "hello from the scripted model";

/// The first prompt of the CLI-record scenarios.
const FIRST_QUESTION: &str = "first question";
/// The prompt of the resumed turn.
const SECOND_QUESTION: &str = "second question";
/// The prompt of the second `--continue` turn.
const THIRD_QUESTION: &str = "third question";

/// Agent the persistent-agent fixtures belong to.
const AGENT_ID: &str = "agent-1";
/// Session id of the persistent-agent fixture conversation.
const STATE_SESSION: &str = "sess-exec";
/// Execution id of the journalled first turn.
const EXECUTION_ID: &str = "exec-1";
/// Request id the journalled turn ran under.
const REQUEST_ID: &str = "req-1";
/// Question the journal's first turn asked.
const EARLIER_QUESTION: &str = "earlier question";
/// Answer the journal's first turn received.
const EARLIER_ANSWER: &str = "earlier answer";
/// Prompt of the turn that resumes the persistent-agent fixture.
const NEXT_QUESTION: &str = "next question";

/// Starts a mock model server whose reusable fallback answers [`ANSWER`].
async fn scripted_server() -> MockLlmServer {
    let server = MockLlmServer::start().await;
    server.set_fallback(ScriptedResponse::text(ANSWER));
    server
}

/// Builds the model-connection and workspace flags of one invocation.
fn connection_args(sandbox: &Sandbox, server: &MockLlmServer) -> Vec<String> {
    vec![
        "--provider".to_owned(),
        "openai".to_owned(),
        "--api-base".to_owned(),
        server.base_url().to_owned(),
        "--api-key".to_owned(),
        TEST_API_KEY.to_owned(),
        "-m".to_owned(),
        "mock-model".to_owned(),
        "-w".to_owned(),
        sandbox.workspace().display().to_string(),
    ]
}

/// Builds the flags of one `run`/`chat` invocation against `server`.
fn cli_args(sandbox: &Sandbox, server: &MockLlmServer, tail: &[&str]) -> Vec<String> {
    let mut args = connection_args(sandbox, server);
    args.extend(tail.iter().map(|arg| (*arg).to_owned()));
    args
}

/// Builds the flags of one invocation that needs no model connection.
fn bare_args(sandbox: &Sandbox, tail: &[&str]) -> Vec<String> {
    let mut args = vec!["-w".to_owned(), sandbox.workspace().display().to_string()];
    args.extend(tail.iter().map(|arg| (*arg).to_owned()));
    args
}

/// Builds the flags of one `sessions` invocation of the same sandbox.
fn sessions_args(sandbox: &Sandbox, tail: &[&str]) -> Vec<String> {
    let mut args = vec![
        "-w".to_owned(),
        sandbox.workspace().display().to_string(),
        "sessions".to_owned(),
    ];
    args.extend(tail.iter().map(|arg| (*arg).to_owned()));
    args
}

/// Returns the `session_id` of a `run -f json` document.
fn session_id_of(value: &Value) -> String {
    value["session_id"]
        .as_str()
        .unwrap_or_else(|| panic!("the run document carries a session id: {value}"))
        .to_owned()
}

/// Returns the `(role, content)` pairs of a `sessions show --json` document.
fn shown_messages(value: &Value) -> Vec<(String, String)> {
    value["messages"]
        .as_array()
        .unwrap_or_else(|| panic!("the show document carries messages: {value}"))
        .iter()
        .map(|message| {
            (
                message["role"].as_str().unwrap_or_default().to_owned(),
                message["content"].as_str().unwrap_or_default().to_owned(),
            )
        })
        .collect()
}

/// Asserts `needle` is reached before `later` in the recorded user texts.
///
/// The window rebuilt from a journal renders an `execution_started` entry as
/// `【本次执行】<text>`, so the needles are matched as substrings: the assertion is
/// about the restored turn preceding the new query, not about the exact frame
/// the window assembler adds.
fn assert_precedes(request: &openjiuwen_test_support::RecordedRequest, needle: &str, later: &str) {
    let texts = request.user_texts();
    let at = texts.iter().position(|text| text.contains(needle));
    let later_at = texts.iter().position(|text| text.contains(later));
    assert!(
        matches!((at, later_at), (Some(at), Some(later_at)) if at < later_at),
        "the model request must carry {needle:?} before {later:?}: {texts:?}"
    );
}

/// Seeds the persistent-agent state another host would have left behind.
///
/// Three files, in the layout the harness writes:
///
/// - `<workspace>/persistent_agent/logs/exec-1/` — the journalled first turn
///   ([`EARLIER_QUESTION`], [`EARLIER_ANSWER`]);
/// - `<workspace>/persistent_agent/sessions/` — the session record whose
///   `ending` names `exec-1`, so a resume traces that journal;
/// - `<workspace>/persistent_agent/links/` — the link that proves `exec-1` is
///   the first execution of the conversation.
async fn seed_persistent_state(sandbox: &Sandbox) {
    let layout = PersistentAgentLayout::for_workspace(&Workspace::new(sandbox.workspace()));
    let identity = ExecutionIdentity::new(
        "host",
        AGENT_ID,
        STATE_SESSION,
        REQUEST_ID,
        EXECUTION_ID,
        "cli",
    )
    .expect("a valid execution identity");
    let log = ExecutionLog::open(
        layout.logs_root().join(identity.directory_name()),
        identity.clone(),
        LogCapacity::default(),
    )
    .await
    .expect("a journalled first turn");
    log.append(EntryPayload::execution_started(EARLIER_QUESTION))
        .await
        .expect("the journal starts with the question");
    log.append(EntryPayload::model_turn(
        "mc_1",
        EARLIER_ANSWER,
        ModelUsage::known(10, 5),
    ))
    .await
    .expect("the journal records the answer");

    let binding = SessionBinding::new(AGENT_ID, STATE_SESSION).expect("a valid binding");
    let store = FileSessionStore::new(layout.sessions_root());
    let mut record = PersistentSessionRecord::new(&binding).expect("a draft record");
    record.messages = vec![BaseMessage::user(EARLIER_QUESTION)];
    let record = record
        .with_ending(RecordEnding {
            outcome: "completed".to_owned(),
            request_id: REQUEST_ID.to_owned(),
            execution_id: EXECUTION_ID.to_owned(),
            ..RecordEnding::default()
        })
        .expect("an ended record");
    store.save(&record).await.expect("a saved record");

    let links = FileLinkStore::new(layout.links_root());
    let link = ExecutionLink::from_identity(&identity, None, LinkRelation::First)
        .expect("a valid execution link");
    links.write_link(&link).await.expect("a written link");
}

/// Runs the first turn, then resumes it by id and checks what both wrote.
#[tokio::test]
async fn a_fresh_run_then_resume_by_id_continues_the_record() {
    let sandbox = Sandbox::new();
    let server = scripted_server().await;

    let first = assert_success(
        &sandbox
            .run(&cli_args(
                &sandbox,
                &server,
                &["run", FIRST_QUESTION, "-f", "json"],
            ))
            .await,
    );
    let session_id = session_id_of(&parse_json(&first));
    let id = session_id.as_str();
    assert!(
        session_id.starts_with("cli-"),
        "a fresh run mints a `cli-` session: {session_id}"
    );

    let shown = parse_json(&assert_success(
        &sandbox
            .run(&sessions_args(&sandbox, &["show", id, "--json"]))
            .await,
    ));
    assert_eq!(
        shown_messages(&shown).len(),
        2,
        "the first turn persisted exactly its question and answer: {shown}"
    );

    let output = sandbox
        .run(&cli_args(
            &sandbox,
            &server,
            &["run", "--resume", id, "-f", "json", SECOND_QUESTION],
        ))
        .await;
    let text = assert_success(&output);
    assert_no_ansi(&text);
    let resumed = parse_json(&text);
    assert_eq!(resumed["answer"], ANSWER, "the resumed turn was answered");
    assert_eq!(
        resumed["session_id"],
        session_id.as_str(),
        "the resumed turn keeps the session id"
    );
    let note = stderr(&output);
    assert!(
        note.contains(&format!(
            "resuming session {session_id} from the CLI session record (2 messages)"
        )),
        "got {note:?}"
    );

    let shown = parse_json(&assert_success(
        &sandbox
            .run(&sessions_args(&sandbox, &["show", id, "--json"]))
            .await,
    ));
    assert_eq!(
        shown_messages(&shown),
        vec![
            ("user".to_owned(), FIRST_QUESTION.to_owned()),
            ("assistant".to_owned(), ANSWER.to_owned()),
            ("user".to_owned(), SECOND_QUESTION.to_owned()),
            ("assistant".to_owned(), ANSWER.to_owned()),
        ],
        "the record continues both turns in order: {shown}"
    );

    let requests = server.requests();
    let last = requests
        .last()
        .unwrap_or_else(|| panic!("the resumed turn never reached the model server"));
    assert_precedes(last, FIRST_QUESTION, SECOND_QUESTION);
    server.shutdown().await;
}

/// `--continue` picks the newest CLI record and keeps extending it.
#[tokio::test]
async fn continue_picks_the_most_recent_session() {
    let sandbox = Sandbox::new();
    let server = scripted_server().await;

    let first = assert_success(
        &sandbox
            .run(&cli_args(
                &sandbox,
                &server,
                &["run", FIRST_QUESTION, "-f", "json"],
            ))
            .await,
    );
    let session_id = session_id_of(&parse_json(&first));
    let id = session_id.as_str();

    let output = sandbox
        .run(&cli_args(
            &sandbox,
            &server,
            &["run", "--continue", "-f", "json", SECOND_QUESTION],
        ))
        .await;
    let text = assert_success(&output);
    assert_eq!(session_id_of(&parse_json(&text)), session_id);
    let note = stderr(&output);
    assert!(
        note.contains(&format!(
            "resuming the most recent session {session_id} from the CLI session record (2 messages)"
        )),
        "got {note:?}"
    );

    let output = sandbox
        .run(&cli_args(
            &sandbox,
            &server,
            &["run", "--continue", "-f", "json", THIRD_QUESTION],
        ))
        .await;
    let text = assert_success(&output);
    assert_eq!(session_id_of(&parse_json(&text)), session_id);
    let note = stderr(&output);
    assert!(
        note.contains("resuming the most recent session cli-"),
        "the `--continue` note names the picked session: {note:?}"
    );

    let shown = parse_json(&assert_success(
        &sandbox
            .run(&sessions_args(&sandbox, &["show", id, "--json"]))
            .await,
    ));
    let messages = shown_messages(&shown);
    assert_eq!(
        messages.len(),
        6,
        "three turns of question and answer: {messages:?}"
    );
    assert_eq!(messages[4], ("user".to_owned(), THIRD_QUESTION.to_owned()));
    assert_eq!(messages[5], ("assistant".to_owned(), ANSWER.to_owned()));
    server.shutdown().await;
}

/// An unknown id is refused before any model work happens.
#[tokio::test]
async fn an_unknown_id_is_a_typed_not_found() {
    let sandbox = Sandbox::new();
    let output = sandbox
        .run(&bare_args(
            &sandbox,
            &["run", "--resume", "does-not-exist", "hi"],
        ))
        .await;
    let text = assert_exit(&output, 1);
    assert!(text.is_empty(), "nothing reaches stdout: {text:?}");
    let note = stderr(&output);
    assert!(
        note.contains("error: not_found: unknown session: does-not-exist"),
        "got {note:?}"
    );
}

/// A home with no persisted session has nothing to continue.
#[tokio::test]
async fn continue_without_any_session_is_a_typed_not_found() {
    let sandbox = Sandbox::new();
    let output = sandbox
        .run(&bare_args(&sandbox, &["run", "--continue", "hi"]))
        .await;
    let text = assert_exit(&output, 1);
    assert!(text.is_empty(), "nothing reaches stdout: {text:?}");
    let note = stderr(&output);
    assert!(note.contains("no resumable session found"), "got {note:?}");
}

/// A persistent session record resumes through the execution its ending names.
#[tokio::test]
async fn a_persistent_state_record_resumes_through_its_execution_plan() {
    let sandbox = Sandbox::new();
    seed_persistent_state(&sandbox).await;
    let server = scripted_server().await;

    let output = sandbox
        .run(&cli_args(
            &sandbox,
            &server,
            &[
                "run",
                "--resume",
                STATE_SESSION,
                "-f",
                "json",
                NEXT_QUESTION,
            ],
        ))
        .await;
    let text = assert_success(&output);
    assert_no_ansi(&text);
    let resumed = parse_json(&text);
    assert_eq!(resumed["answer"], ANSWER);
    assert_eq!(resumed["session_id"], STATE_SESSION);
    let note = stderr(&output);
    assert!(
        note.contains(&format!(
            "from the persistent session record via execution {EXECUTION_ID}"
        )),
        "the ending names the execution the resume traced: {note:?}"
    );

    let shown = parse_json(&assert_success(
        &sandbox
            .run(&sessions_args(&sandbox, &["show", STATE_SESSION, "--json"]))
            .await,
    ));
    assert!(
        shown["plan"].is_object(),
        "the reuse traces the journal into a plan: {shown}"
    );
    let messages = shown_messages(&shown);
    assert!(
        messages
            .iter()
            .any(|(_, content)| content.contains(EARLIER_QUESTION)),
        "the restored window carries the journalled question: {messages:?}"
    );

    let requests = server.requests();
    let last = requests
        .last()
        .unwrap_or_else(|| panic!("the resumed turn never reached the model server"));
    assert_precedes(last, EARLIER_QUESTION, NEXT_QUESTION);
    server.shutdown().await;
}

/// An execution id resumes its journal and creates the session's CLI record.
#[tokio::test]
async fn an_execution_id_resumes_its_journal_and_creates_the_cli_record() {
    let sandbox = Sandbox::new();
    seed_persistent_state(&sandbox).await;
    let server = scripted_server().await;

    let output = sandbox
        .run(&cli_args(
            &sandbox,
            &server,
            &["run", "--resume", EXECUTION_ID, "-f", "json", NEXT_QUESTION],
        ))
        .await;
    let text = assert_success(&output);
    assert_no_ansi(&text);
    let resumed = parse_json(&text);
    assert_eq!(resumed["answer"], ANSWER);
    assert_eq!(
        resumed["session_id"], STATE_SESSION,
        "the execution's own session id is continued"
    );
    let note = stderr(&output);
    assert!(
        note.contains(&format!(
            "resuming session {STATE_SESSION} from execution {EXECUTION_ID} (2 messages restored)"
        )),
        "got {note:?}"
    );

    let rows = parse_json(&assert_success(
        &sandbox
            .run(&sessions_args(&sandbox, &["list", "--json"]))
            .await,
    ));
    let rows = rows.as_array().expect("a JSON array of rows");
    assert!(
        rows.iter().any(|row| row["kind"] == "cli"),
        "the resumed execution leaves a CLI record behind: {rows:?}"
    );
    assert!(
        rows.iter()
            .any(|row| row["kind"] == "cli" && row["id"] == STATE_SESSION),
        "the CLI record belongs to the execution's session: {rows:?}"
    );
    server.shutdown().await;
}

/// `chat --resume` seeds the restored transcript into a fresh REPL session.
#[tokio::test]
async fn chat_resume_seeds_the_transcript_into_a_fresh_repl() {
    let sandbox = Sandbox::new();
    let server = scripted_server().await;

    let first = assert_success(
        &sandbox
            .run(&cli_args(
                &sandbox,
                &server,
                &["run", FIRST_QUESTION, "-f", "json"],
            ))
            .await,
    );
    let session_id = session_id_of(&parse_json(&first));
    let id = session_id.as_str();

    // stdin is `/dev/null`, so the REPL sees EOF right after the note and exits.
    let output = sandbox
        .run(&cli_args(&sandbox, &server, &["chat", "--resume", id]))
        .await;
    let text = assert_success(&output);
    assert_no_ansi(&text);
    let note = stderr(&output);
    assert!(
        note.contains(&format!("resuming session {session_id}")),
        "the note names the resumed session: {note:?}"
    );
    assert!(
        note.contains("the REPL continues as cli-"),
        "the note names the fresh REPL session: {note:?}"
    );
    server.shutdown().await;
}
