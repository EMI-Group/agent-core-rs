//! Resume-surface tests: one-shot continuation, `Repl::seed_history` and output
//! parity with a fresh run.
//!
//! Everything runs offline against the shared harness of [`super`]: scripted IO
//! handles, [`openjiuwen_test_support::ScriptedModel`] and a temporary
//! home/workspace. The one-shot tests pre-create the CLI session record through
//! `CliOptions::sessions()` to prove a resumed run continues it instead of
//! truncating it; the REPL test seeds a built REPL and then takes a normal turn.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use tempfile::TempDir;

use super::*;
use openjiuwen_foundation::BaseMessage;

/// Role string of a user message in a CLI session record.
const USER_ROLE: &str = "user";

/// Role string of an assistant message in a CLI session record.
const ASSISTANT_ROLE: &str = "assistant";

/// Returns the workspace-rooted options of a temporary CLI home.
fn prepared_options(temp: &TempDir) -> CliOptions {
    let options = test_options(temp.path());
    options.ensure_workspace().unwrap();
    options
}

/// Pre-creates the CLI session record `session_id` holding one user message.
fn seed_cli_record(options: &CliOptions, session_id: &str, question: &str) {
    let store = options.sessions();
    store.new_session(session_id, &options.model).unwrap();
    store
        .append_message(session_id, USER_ROLE, question)
        .unwrap();
}

/// Returns the prior window the resumed runs continue.
fn earlier_turns() -> Vec<BaseMessage> {
    vec![
        BaseMessage::user("earlier question"),
        BaseMessage::assistant("earlier answer"),
    ]
}

/// Returns the index of the first message whose text contains `needle`.
fn index_of(messages: &[BaseMessage], needle: &str) -> Option<usize> {
    messages
        .iter()
        .position(|message| message.text().contains(needle))
}

/// Asserts that both earlier texts reach the model ahead of the new query.
fn assert_seeded_window_precedes_query(messages: &[BaseMessage]) {
    let question = index_of(messages, "earlier question").expect("seeded user message");
    let answer = index_of(messages, "earlier answer").expect("seeded assistant message");
    let query = index_of(messages, "new question").expect("new user query");
    assert!(question < answer, "history keeps its oldest-first order");
    assert!(answer < query, "the seeded window precedes the new query");
}

// --- one-shot resume ------------------------------------------------------
#[tokio::test]
async fn resume_continues_the_cli_record_instead_of_truncating_it() {
    let temp = TempDir::new().unwrap();
    let options = prepared_options(&temp);
    seed_cli_record(&options, "resumed-id", "earlier question");

    let (io, _buffer) = scripted_io(&[]);
    let args = RunArgs {
        prompt: Some("new question".to_owned()),
        output_format: OutputFormat::Text,
    };
    let outcome = run_with_model_resumed(
        &options,
        &args,
        &io,
        Arc::new(ScriptedModel::text("scripted answer")),
        ResumedTurn::new("resumed-id").with_history(earlier_turns()),
    )
    .await
    .unwrap();

    assert_eq!(outcome.session_id, "resumed-id");
    assert_eq!(outcome.answer, "scripted answer");

    let record = options
        .sessions()
        .load("resumed-id")
        .unwrap()
        .expect("the continued record exists");
    let contents: Vec<&str> = record
        .messages
        .iter()
        .map(|message| message.content.as_str())
        .collect();
    assert_eq!(
        contents,
        vec!["earlier question", "new question", "scripted answer"],
        "the pre-existing record is continued, not truncated"
    );
    let roles: Vec<&str> = record
        .messages
        .iter()
        .map(|message| message.role.as_str())
        .collect();
    assert_eq!(roles, vec![USER_ROLE, USER_ROLE, ASSISTANT_ROLE]);
}

#[tokio::test]
async fn resume_seeds_the_window_ahead_of_the_new_query() {
    let temp = TempDir::new().unwrap();
    let options = prepared_options(&temp);
    seed_cli_record(&options, "resumed-id", "earlier question");

    let (io, buffer) = scripted_io(&[]);
    let args = RunArgs {
        prompt: Some("new question".to_owned()),
        output_format: OutputFormat::Text,
    };
    let model = Arc::new(ScriptedModel::text("scripted answer"));
    let handle = Arc::clone(&model);
    run_with_model_resumed(
        &options,
        &args,
        &io,
        model,
        ResumedTurn::new("resumed-id").with_history(earlier_turns()),
    )
    .await
    .unwrap();

    let call = handle.last_call().expect("the model was called");
    assert_seeded_window_precedes_query(&call.messages);
    assert!(output(&buffer).contains("scripted answer"));
}

// --- REPL resume ----------------------------------------------------------
#[tokio::test]
async fn repl_seed_history_survives_a_turn() {
    let temp = TempDir::new().unwrap();
    let options = prepared_options(&temp);
    let (io, buffer) = scripted_io(&[]);
    let model = ScriptedModel::text("scripted answer");
    let agent = super::run::build_agent(Arc::new(model.clone()), &options, &io)
        .await
        .unwrap();
    let mut repl = Repl::with_io(agent, options.clone(), io);

    repl.seed_history(earlier_turns()).await.unwrap();
    let turn = repl.run_turn("new question").await.unwrap();

    assert_eq!(turn.answer, "scripted answer");
    let call = model.last_call().expect("the model was called");
    assert_seeded_window_precedes_query(&call.messages);
    assert!(output(&buffer).contains("scripted answer"));
}

// --- builders and output parity ------------------------------------------
#[test]
fn resumed_turn_builders_expose_their_fields() {
    let state = json!({"iteration": 3});
    let turn = ResumedTurn::new("id")
        .with_history(vec![BaseMessage::user("earlier question")])
        .with_state(state.clone());

    assert_eq!(turn.session_id, "id");
    assert_eq!(turn.history.len(), 1);
    assert_eq!(turn.history[0].text().as_ref(), "earlier question");
    assert_eq!(turn.state, Some(state));
}

#[tokio::test]
async fn resumed_text_run_matches_fresh_output_bytes() {
    let temp = TempDir::new().unwrap();
    let options = prepared_options(&temp);
    let args = RunArgs {
        prompt: Some("hello".to_owned()),
        output_format: OutputFormat::Text,
    };

    let (fresh_io, fresh_buffer) = scripted_io(&[]);
    let fresh = run_with_model(
        &options,
        &args,
        &fresh_io,
        Arc::new(ScriptedModel::text("scripted answer")),
    )
    .await
    .unwrap();

    let (resumed_io, resumed_buffer) = scripted_io(&[]);
    let resumed = run_with_model_resumed(
        &options,
        &args,
        &resumed_io,
        Arc::new(ScriptedModel::text("scripted answer")),
        ResumedTurn::new("resumed-parity"),
    )
    .await
    .unwrap();

    assert_eq!(
        output(&fresh_buffer),
        output(&resumed_buffer),
        "the resumed path renders through the same tail as a fresh run"
    );
    assert_eq!(fresh.answer, resumed.answer);
    assert_eq!(resumed.session_id, "resumed-parity");
    assert!(output(&resumed_buffer).contains("scripted answer"));
}
