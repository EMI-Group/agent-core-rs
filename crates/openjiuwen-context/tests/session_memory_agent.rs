#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//! Public-API tests for the LLM-driven session-memory agent and the
//! compaction branch it makes reachable.
//!
//! The agent is pinned by its honesty contract: a model reply that parses writes
//! the notes it asks for and reports `NoteSource::Model`; a transport failure
//! falls back to the deterministic offline extractor and still reports the model
//! failure; a reply that carries no usable directive writes nothing at all; and
//! a history too short to hold a session is skipped without calling the model.
//!
//! The compaction branch is pinned end to end: a `FullCompactProcessor` with a
//! session-memory store and no notes extracts them, commits them, and reinjects
//! the resulting summary, while a processor that cannot run the branch reports
//! why — through `tracing` and its own checkpoint — instead of doing nothing.
//!
//! Everything is offline and deterministic: token counts come from the
//! one-character-per-token counter, and the only model is
//! [`openjiuwen_test_support::ScriptedModel`].
use std::sync::Arc;

use openjiuwen_context::processor::{
    ContextProcessor, FullCompactProcessor, FullCompactProcessorConfig, ProcessorConfig,
    ProcessorContext,
};
use openjiuwen_context::session_memory_agent::{
    apply_note_directives, extract_offline_notes, NoteDirective, NoteSource, SessionMemoryAgent,
    SessionMemoryAgentConfig, SessionMemoryOutcome, SessionMemoryUpdateMode,
    SESSION_MEMORY_SYSTEM_PROMPT,
};
use openjiuwen_context::{
    count_messages, ContextConfig, ContextEngine, ContextOptions, InMemoryOffloadStore,
    InMemorySessionMemoryStore, ModelContext, OffloadScope, OffloadStore, SessionMemoryNotes,
    SessionMemoryStore, TokenCounterConfig, TokenCounterKind, CONTEXT_MESSAGE_ID_KEY,
};
use openjiuwen_foundation::llm::Model;
use openjiuwen_foundation::prompt::{ApproxTokenCounter, BaseMessage, Role};
use openjiuwen_session::{AgentSession, Session};
use openjiuwen_test_support::{ScriptedError, ScriptedModel};
use serde_json::json;
use tokio_util::sync::CancellationToken;

/// The one-character-per-token counter every test uses.
fn counter() -> ApproxTokenCounter {
    ApproxTokenCounter::with_chars_per_token(1)
}

/// The shared processor inputs of the compaction tests.
struct Fixture {
    config: ContextConfig,
    counter: ApproxTokenCounter,
    offload: Arc<dyn OffloadStore>,
    scope: OffloadScope,
    cancellation: CancellationToken,
}

impl Fixture {
    fn new() -> Self {
        Self {
            config: ContextConfig::default(),
            counter: counter(),
            offload: Arc::new(InMemoryOffloadStore::new()),
            scope: OffloadScope::new("session-1", "main"),
            cancellation: CancellationToken::new(),
        }
    }

    /// A processor context with no session-memory seam at all.
    fn context(&self) -> ProcessorContext<'_> {
        ProcessorContext {
            config: &self.config,
            counter: &self.counter,
            model: None,
            offload: &*self.offload,
            scope: &self.scope,
            budget: 1_000,
            trigger_tokens: 1_000,
            manual: false,
            cancellation: &self.cancellation,
            recall: None,
            session_memory: None,
            session_memory_agent: None,
        }
    }

    /// A processor context that carries the store, the model and (optionally)
    /// a per-request agent.
    fn session_memory_context<'a>(
        &'a self,
        store: &'a Arc<dyn SessionMemoryStore>,
        model: Option<&'a Arc<dyn Model>>,
        agent: Option<Arc<SessionMemoryAgent>>,
    ) -> ProcessorContext<'a> {
        ProcessorContext {
            model,
            session_memory: Some(&**store),
            session_memory_agent: agent,
            ..self.context()
        }
    }
}

/// Notes with no anchor yet.
fn notes(text: &str) -> SessionMemoryNotes {
    SessionMemoryNotes {
        notes: text.to_owned(),
        anchor_message_id: None,
    }
}

/// A conversation worth noting: a task, a decision and a question.
fn conversation() -> Vec<BaseMessage> {
    vec![
        BaseMessage::system("persona"),
        BaseMessage::user("TODO: port the session-memory agent into compact.rs"),
        BaseMessage::assistant("We decided to keep the store seam and src/compact.rs untouched."),
        BaseMessage::user("should we also cover the offline extractor?"),
    ]
}

/// A conversation long enough to force a full compaction.
fn long_conversation() -> Vec<BaseMessage> {
    let body = |text: &str| format!("{text}: {}", "detail ".repeat(700));
    vec![
        BaseMessage::system("persona"),
        BaseMessage::user(body("u1")),
        BaseMessage::assistant(body("a1")),
        BaseMessage::user(body("u2")),
        BaseMessage::assistant(body("a2")),
    ]
}

/// A compaction configuration that triggers on every long history here.
fn compact_config() -> FullCompactProcessorConfig {
    FullCompactProcessorConfig {
        trigger_total_tokens: 400,
        compression_call_max_tokens: 100_000,
        messages_to_keep: 0,
        ..FullCompactProcessorConfig::default()
    }
}

/// An in-memory session-memory store behind the trait object.
fn store() -> Arc<dyn SessionMemoryStore> {
    Arc::new(InMemorySessionMemoryStore::new())
}

// ---------------------------------------------------------------------------
// The agent: model replies
// ---------------------------------------------------------------------------

#[tokio::test]
async fn a_model_reply_writes_the_notes_it_asks_for() {
    let scripted = ScriptedModel::text(
        "ADD: - ported the session-memory agent\n\
         REPLACE: - old task => - updated task\n\
         REMOVE: - stale note",
    );
    let model: Arc<dyn Model> = Arc::new(scripted.clone());
    let agent = SessionMemoryAgent::default();
    let mut messages = conversation();
    let current = notes("- old task\n- stale note\n- keep me");

    let outcome = agent.update(&mut messages, &current, Some(&model)).await;
    let SessionMemoryOutcome::Written {
        source,
        notes,
        directives_applied,
        model_failure,
    } = outcome
    else {
        panic!("the reply must write the notes it asks for: {outcome:?}");
    };
    assert_eq!(source, NoteSource::Model);
    assert_eq!(model_failure, None);
    assert_eq!(directives_applied, 3);
    assert_eq!(
        notes.notes,
        "- updated task\n- keep me\n- ported the session-memory agent"
    );

    // The notes anchor on the last message they cover, so a reinjection can
    // replace that prefix even though the host never stamped an id.
    let anchor = notes
        .anchor_message_id
        .clone()
        .expect("the notes carry an anchor");
    assert_eq!(
        messages.last().unwrap().metadata[CONTEXT_MESSAGE_ID_KEY],
        json!(anchor)
    );

    // The prompt carried the protocol, the current notes and the transcript.
    let call = scripted.last_call().expect("the model was called once");
    assert_eq!(call.mode, openjiuwen_test_support::CallMode::Invoke);
    assert_eq!(call.messages[0].role, Role::System);
    assert_eq!(call.messages[0].text(), SESSION_MEMORY_SYSTEM_PROMPT);
    let prompt = call.messages[1].text().into_owned();
    assert!(prompt.contains("- old task\n- stale note\n- keep me"));
    assert!(prompt.contains("[2] user: TODO: port the session-memory agent into compact.rs"));
    assert_eq!(scripted.call_count(), 1);

    // Storing the result is the caller's job; the store round-trips it verbatim.
    let store = InMemorySessionMemoryStore::new();
    let scope = OffloadScope::new("session-1", "main");
    store.put(&scope, notes.clone()).await.unwrap();
    assert_eq!(store.get(&scope).await.unwrap(), Some(notes));
    assert_eq!(store.len(), 1);
}

#[tokio::test]
async fn a_direct_replace_reply_rewrites_the_whole_notes_file() {
    let scripted = ScriptedModel::text("DIRECT_REPLACE: # Notes\n\n## Tasks\n- everything anew");
    let model: Arc<dyn Model> = Arc::new(scripted);
    let agent = SessionMemoryAgent::default();
    let mut messages = conversation();

    let outcome = agent
        .update(&mut messages, &notes("- stale line"), Some(&model))
        .await;
    let SessionMemoryOutcome::Written {
        source,
        notes,
        directives_applied,
        ..
    } = outcome
    else {
        panic!("the direct replacement must write notes: {outcome:?}");
    };
    assert_eq!(source, NoteSource::Model);
    assert_eq!(directives_applied, 1);
    assert_eq!(notes.notes, "# Notes\n\n## Tasks\n- everything anew");
}

#[tokio::test]
async fn a_failed_model_call_falls_back_to_the_offline_extractor_and_says_so() {
    let scripted = ScriptedModel::builder()
        .fail(ScriptedError::transport("connection reset"))
        .build();
    let model: Arc<dyn Model> = Arc::new(scripted.clone());
    let agent = SessionMemoryAgent::default();
    let mut messages = conversation();

    let outcome = agent.update(&mut messages, &notes(""), Some(&model)).await;
    let SessionMemoryOutcome::Written {
        source,
        notes,
        directives_applied,
        model_failure,
    } = outcome
    else {
        panic!("the offline extractor must still produce notes: {outcome:?}");
    };
    // The notes are reported as offline output, with the failure attached.
    assert_eq!(source, NoteSource::Offline);
    assert_eq!(directives_applied, 0);
    let failure = model_failure.expect("the model failure is reported");
    assert!(failure.contains("connection reset"), "{failure}");
    assert!(notes.notes.contains("## Tasks"));
    assert!(notes
        .notes
        .contains("TODO: port the session-memory agent into compact.rs"));
    assert!(notes.anchor_message_id.is_some());
    assert_eq!(scripted.call_count(), 1);
}

#[tokio::test]
async fn a_missing_model_uses_the_offline_extractor() {
    let agent = SessionMemoryAgent::default();
    let mut messages = conversation();
    let outcome = agent.update(&mut messages, &notes(""), None).await;
    let SessionMemoryOutcome::Written {
        source,
        model_failure,
        notes: written,
        ..
    } = outcome
    else {
        panic!("a missing model must not block the notes: {outcome:?}");
    };
    assert_eq!(source, NoteSource::Offline);
    assert!(model_failure.unwrap().contains("no model"));
    assert!(written.notes.contains("## Decisions"));
    assert!(written.notes.contains("We decided to keep the store seam"));

    // With the fallback disabled the same case is a reported model failure.
    let strict = SessionMemoryAgent::new(SessionMemoryAgentConfig {
        offline_fallback: false,
        ..SessionMemoryAgentConfig::default()
    });
    let mut messages = conversation();
    let outcome = strict.update(&mut messages, &notes(""), None).await;
    assert!(matches!(outcome, SessionMemoryOutcome::ModelFailed { .. }));
    assert!(outcome.notes().is_none());
}

#[tokio::test]
async fn an_unparsable_reply_never_produces_notes() {
    let scripted = ScriptedModel::text("Sure! Here is a short summary of the session so far.");
    let model: Arc<dyn Model> = Arc::new(scripted);
    let agent = SessionMemoryAgent::default();
    let mut messages = conversation();

    let outcome = agent
        .update(&mut messages, &notes("- existing note"), Some(&model))
        .await;
    let SessionMemoryOutcome::Unchanged { reason } = &outcome else {
        panic!("an unparsable reply must not write notes: {outcome:?}");
    };
    assert!(reason.contains("no applicable directive"), "{reason}");
    assert!(outcome.notes().is_none());
    assert_eq!(outcome.source(), None);
    // The offline fallback is reserved for model failures, never for a reply
    // that simply did not parse.
    assert!(!reason.contains("offline"));
}

#[tokio::test]
async fn a_reply_that_repeats_the_notes_is_unchanged() {
    let scripted = ScriptedModel::text("ADD: - keep me\nREMOVE: - nothing alike");
    let model: Arc<dyn Model> = Arc::new(scripted);
    let agent = SessionMemoryAgent::default();
    let mut messages = conversation();

    let outcome = agent
        .update(&mut messages, &notes("- keep me"), Some(&model))
        .await;
    let SessionMemoryOutcome::Unchanged { reason } = outcome else {
        panic!("a duplicate-only reply must not rewrite the notes: {outcome:?}");
    };
    assert!(reason.contains("no applicable directive"), "{reason}");
}

#[tokio::test]
async fn a_short_history_or_a_disabled_agent_is_skipped() {
    let scripted = ScriptedModel::text("ADD: - never asked");
    let model: Arc<dyn Model> = Arc::new(scripted.clone());
    let agent = SessionMemoryAgent::default();
    let mut short = vec![BaseMessage::user("hi")];

    let outcome = agent.update(&mut short, &notes(""), Some(&model)).await;
    let SessionMemoryOutcome::Skipped { reason } = outcome else {
        panic!("a one-message history cannot hold a session: {outcome:?}");
    };
    assert!(reason.contains("fewer than the 2"), "{reason}");
    assert_eq!(scripted.call_count(), 0);

    let disabled = SessionMemoryAgent::new(SessionMemoryAgentConfig {
        enabled: false,
        ..SessionMemoryAgentConfig::default()
    });
    let mut messages = conversation();
    let outcome = disabled
        .update(&mut messages, &notes(""), Some(&model))
        .await;
    assert!(matches!(outcome, SessionMemoryOutcome::Skipped { .. }));
    assert_eq!(scripted.call_count(), 0);
}

#[tokio::test]
async fn the_direct_protocol_asks_for_the_whole_file() {
    let scripted = ScriptedModel::text("# Notes\n\n## Tasks\n- one thing");
    let model: Arc<dyn Model> = Arc::new(scripted.clone());
    let agent = SessionMemoryAgent::new(SessionMemoryAgentConfig {
        update_mode: SessionMemoryUpdateMode::DirectReplace,
        max_notes_chars: 8,
        ..SessionMemoryAgentConfig::default()
    });
    let mut messages = conversation();

    let outcome = agent
        .update(&mut messages, &notes("- old"), Some(&model))
        .await;
    let SessionMemoryOutcome::Written { notes, source, .. } = outcome else {
        panic!("the whole-file protocol must write notes: {outcome:?}");
    };
    assert_eq!(source, NoteSource::Model);
    // The character budget still applies to a whole-file reply.
    assert_eq!(notes.notes, "# Notes");
    let call = scripted.last_call().unwrap();
    assert!(call.messages[0]
        .text()
        .contains("Reply with the COMPLETE updated notes file"));
    assert!(!call.messages[0].text().contains("ADD:"));
}

// ---------------------------------------------------------------------------
// Parsing and the offline extractor
// ---------------------------------------------------------------------------

#[test]
fn the_directive_vocabulary_is_enforced() {
    let strict = SessionMemoryAgentConfig {
        directives: vec![NoteDirective::Add],
        ..SessionMemoryAgentConfig::default()
    };
    let edit = apply_note_directives("- a", "REMOVE: - a", &strict);
    assert_eq!(edit.notes, None);
    assert_eq!(edit.unrecognised, 1);
    // A directive that parses but matches nothing is reported, not invented.
    let edit = apply_note_directives(
        "- a",
        "REPLACE: missing => new",
        &SessionMemoryAgentConfig::default(),
    );
    assert_eq!(edit.notes, None);
    assert_eq!(edit.unapplied, 1);
    assert_eq!(edit.unrecognised, 0);
}

#[test]
fn the_offline_extractor_is_deterministic_and_bounded() {
    let messages = conversation();
    let first = extract_offline_notes(&messages, 10_000);
    let second = extract_offline_notes(&messages, 10_000);
    assert_eq!(first, second);
    assert!(first.contains("## Open Questions\n- should we also cover the offline extractor?"));
    // A budget truncates the notes instead of growing them.
    let small = extract_offline_notes(&messages, 20);
    assert_eq!(small.chars().count(), 20);
    assert!(first.starts_with(&small));
}

// ---------------------------------------------------------------------------
// The compaction branch
// ---------------------------------------------------------------------------

#[tokio::test]
async fn full_compaction_extracts_notes_it_lacks_and_reinjects_them() {
    let fixture = Fixture::new();
    let store = store();
    let scripted = ScriptedModel::text(
        "ADD: - ported the session-memory agent\nADD: - the store holds the notes",
    );
    let model: Arc<dyn Model> = Arc::new(scripted.clone());
    let compressor = FullCompactProcessor::new(
        compact_config(),
        Some(Arc::clone(&model)),
        Some(Arc::clone(&store)),
    )
    .with_session_memory_agent(Arc::new(SessionMemoryAgent::default()));
    let mut messages = long_conversation();
    let before = count_messages(&fixture.counter, &messages);

    let event = compressor
        .process(
            &mut messages,
            &fixture.session_memory_context(&store, Some(&model), None),
        )
        .await
        .unwrap()
        .expect("the extracted notes must compact the history");

    // The summary is the notes the agent extracted, not a model transcript summary.
    assert!(messages[0]
        .text()
        .starts_with(openjiuwen_context::SESSION_MEMORY_BOUNDARY));
    assert!(messages[1]
        .text()
        .contains("ported the session-memory agent"));
    assert!(event.summary.unwrap().contains("the store holds the notes"));
    assert!(event.tokens_after < before);
    assert_eq!(
        compressor.state.text("session_memory_source").as_deref(),
        Some("model")
    );
    assert_eq!(compressor.state.counter("session_memory_extractions"), 1);
    assert_eq!(compressor.state.counter("session_memory_compactions"), 1);
    assert_eq!(compressor.state.text("session_memory_refusal"), None);
    assert_eq!(scripted.call_count(), 1);

    // The notes were committed to the store, anchored on the message they cover.
    let stored = store
        .get(&fixture.scope)
        .await
        .unwrap()
        .expect("the extracted notes are persisted");
    assert!(stored.notes.contains("ported the session-memory agent"));
    assert!(stored.anchor_message_id.is_some());
}

#[tokio::test]
async fn a_per_request_agent_wins_over_the_processors_own() {
    let fixture = Fixture::new();
    let store = store();
    let scripted = ScriptedModel::text("ADD: - written by the request agent");
    let model: Arc<dyn Model> = Arc::new(scripted.clone());
    // The processor holds no agent; the request supplies one.
    let compressor = FullCompactProcessor::new(
        compact_config(),
        Some(Arc::clone(&model)),
        Some(Arc::clone(&store)),
    );
    let mut messages = long_conversation();

    let event = compressor
        .process(
            &mut messages,
            &fixture.session_memory_context(
                &store,
                Some(&model),
                Some(Arc::new(SessionMemoryAgent::default())),
            ),
        )
        .await
        .unwrap()
        .expect("the request agent must extract the notes");

    assert!(event
        .summary
        .unwrap()
        .contains("written by the request agent"));
    assert_eq!(scripted.call_count(), 1);
}

#[tokio::test]
async fn a_processor_with_no_store_reports_why_it_cannot_use_session_memory() {
    let fixture = Fixture::new();
    let compressor = FullCompactProcessor::new(compact_config(), None, None);
    let mut messages = long_conversation();

    let event = compressor
        .process(&mut messages, &fixture.context())
        .await
        .unwrap()
        .expect("the summary path still compacts");

    assert!(event.summary.unwrap().starts_with("Summary:\n"));
    let reason = compressor
        .state
        .text("session_memory_refusal")
        .expect("the refusal is reported");
    assert!(
        reason.contains("no session-memory store is mounted"),
        "{reason}"
    );
    assert_eq!(compressor.state.counter("session_memory_refusals"), 1);
}

#[tokio::test]
async fn an_empty_store_without_an_agent_reports_the_missing_seam() {
    let fixture = Fixture::new();
    let store = store();
    let compressor = FullCompactProcessor::new(compact_config(), None, Some(Arc::clone(&store)));
    let mut messages = long_conversation();

    let event = compressor
        .process(
            &mut messages,
            &fixture.session_memory_context(&store, None, None),
        )
        .await
        .unwrap()
        .expect("the summary path still compacts");

    assert!(event.summary.is_some());
    let reason = compressor
        .state
        .text("session_memory_refusal")
        .expect("the refusal is reported");
    assert!(
        reason.contains("no session-memory agent is configured"),
        "{reason}"
    );
    // Nothing was invented and nothing was stored.
    assert!(store.get(&fixture.scope).await.unwrap().is_none());
}

#[tokio::test]
async fn a_rejected_offline_extraction_reports_the_reason() {
    let fixture = Fixture::new();
    let store = store();
    // The agent needs two messages; one is too short to hold a session, so the
    // branch cannot run and must say so instead of compacting from nothing.
    let agent = SessionMemoryAgent::new(SessionMemoryAgentConfig {
        min_messages: 64,
        ..SessionMemoryAgentConfig::default()
    });
    let compressor = FullCompactProcessor::new(compact_config(), None, Some(Arc::clone(&store)))
        .with_session_memory_agent(Arc::new(agent));
    let mut messages = long_conversation();

    compressor
        .process(
            &mut messages,
            &fixture.session_memory_context(&store, None, None),
        )
        .await
        .unwrap()
        .expect("the summary path still compacts");

    let reason = compressor.state.text("session_memory_refusal").unwrap();
    assert!(
        reason.starts_with("session memory could not be extracted"),
        "{reason}"
    );
    assert!(reason.contains("fewer than the 64"), "{reason}");
    assert!(store.get(&fixture.scope).await.unwrap().is_none());
}

// ---------------------------------------------------------------------------
// Engine wiring: the default agent reaches the branch through the pipeline
// ---------------------------------------------------------------------------

#[tokio::test]
async fn an_engine_context_extracts_notes_with_the_default_agent() {
    let store: Arc<dyn SessionMemoryStore> = Arc::new(InMemorySessionMemoryStore::new());
    let scripted = ScriptedModel::text("ADD: - extracted by the engine context");
    let model: Arc<dyn Model> = Arc::new(scripted.clone());
    let session: Arc<dyn Session> = Arc::new(AgentSession::new("s"));
    let engine = ContextEngine::with_stores(
        ContextConfig {
            token_counter: TokenCounterConfig {
                kind: TokenCounterKind::Characters,
                chars_per_token: 1,
            },
            processors: vec![ProcessorConfig::FullCompact(compact_config())],
            ..ContextConfig::default()
        },
        Arc::new(InMemoryOffloadStore::new()),
        None,
        Some(Arc::clone(&store)),
    );
    let context = engine
        .create_context_with(
            "main",
            ContextOptions {
                session: Some(session),
                model: Some(Arc::clone(&model)),
                history: long_conversation(),
                ..ContextOptions::default()
            },
        )
        .await
        .unwrap();
    let before = context.messages().await.len();

    let report = engine.compress(&context).await.unwrap();
    assert!(report.is_compressed(), "{report:?}");
    // The history is replaced by the boundary plus the extracted notes.
    let messages = context.messages().await;
    assert!(messages.len() < before);
    assert!(messages
        .iter()
        .any(|message| message.text().contains("extracted by the engine context")));
    assert_eq!(scripted.call_count(), 1);
    let stored = store
        .get(&context.scope())
        .await
        .unwrap()
        .expect("the engine context persists the notes");
    assert!(stored.notes.contains("extracted by the engine context"));
}
