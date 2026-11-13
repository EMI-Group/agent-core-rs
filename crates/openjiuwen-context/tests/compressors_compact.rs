#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//! Public-API tests for [`FullCompactProcessor`] and [`MicroCompactProcessor`].
//!
//! Full compaction is pinned by structure: the boundary splits the history into a
//! preserved prefix and a compacted active part, the summary message carries the
//! intro and the recent-messages notice, the kept tail pulls its owning tool
//! calls along, and only a real saving is accepted. Both the deterministic
//! offline fallback and the model path are covered, as is the session-memory
//! path with a real in-memory store.
//!
//! Micro compaction is pinned by its narrow contract: only the text of stale
//! tool results changes, grouping is per tool name, the newest results stay
//! verbatim, identity fields survive, and a second run is a no-op.
//!
//! Everything is offline and deterministic: token counts come from the
//! one-character-per-token counter, and the only model is
//! [`openjiuwen_test_support::ScriptedModel`].
//!
//! [`FullCompactProcessor`]: openjiuwen_context::processor::FullCompactProcessor
//! [`MicroCompactProcessor`]: openjiuwen_context::processor::MicroCompactProcessor
use std::sync::Arc;

use openjiuwen_context::processor::{
    ContextPipeline, ContextProcessor, FullCompactProcessor, FullCompactProcessorConfig,
    MicroCompactProcessor, MicroCompactProcessorConfig, ProcessorConfig, ProcessorContext,
    RoundLevelCompressorConfig, FULL_COMPACT_BOUNDARY, FULL_COMPACT_BOUNDARY_TEXT,
    FULL_COMPACT_SUMMARY_INTRO, MICRO_COMPACT_CLEARED_MARKER,
};
use openjiuwen_context::{
    count_messages, stamp_message_id, validate_tool_pairs, ContextConfig, ContextEventKind,
    InMemoryOffloadStore, InMemorySessionMemoryStore, OffloadScope, OffloadStore,
    SessionMemoryNotes, SessionMemoryStore, SESSION_MEMORY_BOUNDARY,
};
use openjiuwen_foundation::llm::Model;
use openjiuwen_foundation::prompt::{ApproxTokenCounter, BaseMessage, Role, ToolCall};
use openjiuwen_test_support::ScriptedModel;
use serde_json::json;
use tokio_util::sync::CancellationToken;

/// The one-character-per-token counter every test uses.
fn counter() -> ApproxTokenCounter {
    ApproxTokenCounter::with_chars_per_token(1)
}

/// A body long enough that replacing it with a summary is a real saving.
fn long(text: &str) -> String {
    format!("{text}: {}", "detail ".repeat(1_200))
}

/// Shared processor inputs.
struct Fixture {
    config: ContextConfig,
    counter: ApproxTokenCounter,
    store: Arc<dyn OffloadStore>,
    scope: OffloadScope,
    cancellation: CancellationToken,
}

impl Fixture {
    fn new() -> Self {
        Self {
            config: ContextConfig::default(),
            counter: counter(),
            store: Arc::new(InMemoryOffloadStore::new()),
            scope: OffloadScope::new("session-1", "main"),
            cancellation: CancellationToken::new(),
        }
    }

    fn context(&self) -> ProcessorContext<'_> {
        ProcessorContext {
            config: &self.config,
            counter: &self.counter,
            model: None,
            offload: &*self.store,
            scope: &self.scope,
            budget: 1_000,
            trigger_tokens: 1_000,
            manual: false,
            cancellation: &self.cancellation,
        }
    }

    fn manual_context(&self) -> ProcessorContext<'_> {
        ProcessorContext {
            manual: true,
            ..self.context()
        }
    }

    fn model_context<'a>(&'a self, model: &'a Arc<dyn Model>) -> ProcessorContext<'a> {
        ProcessorContext {
            model: Some(model),
            ..self.context()
        }
    }
}

/// A compaction configuration that triggers on every long history here.
fn compact_config() -> FullCompactProcessorConfig {
    FullCompactProcessorConfig {
        trigger_total_tokens: 2_000,
        compression_call_max_tokens: 100_000,
        // Keep no tail unless a test asks for one.
        messages_to_keep: 0,
        ..FullCompactProcessorConfig::default()
    }
}

/// A four-message conversation with long bodies.
fn conversation() -> Vec<BaseMessage> {
    vec![
        BaseMessage::system("persona"),
        BaseMessage::user(long("u1")),
        BaseMessage::assistant(long("a1")),
        BaseMessage::user(long("u2")),
        BaseMessage::assistant(long("a2")),
    ]
}

/// A conversation that already carries a compaction boundary after a short
/// preserved prefix.
fn conversation_with_boundary(marker: &str) -> Vec<BaseMessage> {
    vec![
        BaseMessage::system("persona"),
        BaseMessage::user("u1"),
        BaseMessage::system(format!("{marker}\nConversation compacted once")),
        BaseMessage::user(long("u2")),
        BaseMessage::assistant(long("a2")),
    ]
}

#[tokio::test]
async fn full_compaction_keeps_the_prefix_and_replaces_the_active_part() {
    let fixture = Fixture::new();
    let compressor = FullCompactProcessor::new(compact_config(), None, None);
    let mut messages = conversation_with_boundary(FULL_COMPACT_BOUNDARY);
    let before = count_messages(&fixture.counter, &messages);

    let event = compressor
        .process(&mut messages, &fixture.context())
        .await
        .unwrap()
        .expect("an oversized history must compact");

    // The prefix before the boundary survives verbatim.
    assert_eq!(messages.len(), 4);
    assert_eq!(messages[0].text(), "persona");
    assert_eq!(messages[1].text(), "u1");
    // One new boundary, then the summary pair.
    assert_eq!(messages[2].role, Role::System);
    assert_eq!(messages[2].text(), FULL_COMPACT_BOUNDARY_TEXT);
    assert_eq!(messages[3].role, Role::User);
    let summary_message = messages[3].text().into_owned();
    assert!(summary_message.starts_with(FULL_COMPACT_SUMMARY_INTRO));
    assert!(summary_message.contains("\n\nSummary:\n[1] user: u2: detail"));

    assert_eq!(event.processor, "full_compact");
    assert_eq!(event.kind, ContextEventKind::Summarized);
    assert_eq!(event.messages_changed, (0..5).collect::<Vec<_>>());
    assert_eq!(event.tokens_before, before);
    assert!(event.tokens_after < before);
    let summary = event.summary.expect("the event carries the summary");
    assert!(summary.starts_with("Summary:\n[1] user: u2: detail"));
}

#[tokio::test]
async fn a_session_memory_boundary_also_splits_the_history() {
    let fixture = Fixture::new();
    let compressor = FullCompactProcessor::new(compact_config(), None, None);
    let mut messages = conversation_with_boundary(SESSION_MEMORY_BOUNDARY);

    let event = compressor
        .process(&mut messages, &fixture.context())
        .await
        .unwrap()
        .expect("the boundary must be honoured");

    assert_eq!(messages[0].text(), "persona");
    assert_eq!(messages[1].text(), "u1");
    assert_eq!(messages[2].text(), FULL_COMPACT_BOUNDARY_TEXT);
    assert!(event.tokens_after < event.tokens_before);
}

#[tokio::test]
async fn without_a_boundary_the_whole_history_is_active() {
    let fixture = Fixture::new();
    let compressor = FullCompactProcessor::new(compact_config(), None, None);
    let mut messages = conversation();

    compressor
        .process(&mut messages, &fixture.context())
        .await
        .unwrap()
        .expect("the whole history is compactable");

    // Boundary plus summary only: there is no prefix to preserve.
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0].text(), FULL_COMPACT_BOUNDARY_TEXT);
    assert!(messages[1].text().contains(FULL_COMPACT_SUMMARY_INTRO));
}

#[tokio::test]
async fn an_unfinished_api_round_blocks_the_pass() {
    let fixture = Fixture::new();
    let compressor = FullCompactProcessor::new(compact_config(), None, None);
    let mut messages = conversation();
    messages.push(BaseMessage::user(long("u3")));
    let untouched = messages.clone();

    assert!(compressor
        .process(&mut messages, &fixture.context())
        .await
        .unwrap()
        .is_none());
    assert_eq!(messages, untouched);

    // Manual compaction does not wait for the round to close.
    assert!(compressor
        .process(&mut messages, &fixture.manual_context())
        .await
        .unwrap()
        .is_some());
}

#[tokio::test]
async fn nothing_happens_below_the_threshold() {
    let fixture = Fixture::new();
    let compressor = FullCompactProcessor::new(FullCompactProcessorConfig::default(), None, None);
    let mut messages = vec![BaseMessage::user("hi"), BaseMessage::assistant("hello")];
    let untouched = messages.clone();
    assert!(compressor
        .process(&mut messages, &fixture.context())
        .await
        .unwrap()
        .is_none());
    assert_eq!(messages, untouched);
}

#[tokio::test]
async fn the_kept_tail_pulls_its_owning_tool_call_along() {
    let fixture = Fixture::new();
    let config = FullCompactProcessorConfig {
        messages_to_keep: 2,
        ..compact_config()
    };
    let compressor = FullCompactProcessor::new(config.clone(), None, None);
    let mut messages = vec![
        BaseMessage::system("persona"),
        BaseMessage::user(long("u1")),
        BaseMessage::assistant(long("a1")),
        BaseMessage::user("u2"),
        BaseMessage::assistant("")
            .with_tool_calls(vec![ToolCall::new("read_file").with_id("call_1")]),
        BaseMessage::tool("call_1", long("tool output")),
        BaseMessage::assistant("a3"),
    ];
    let original = messages.clone();

    compressor
        .process(&mut messages, &fixture.context())
        .await
        .unwrap()
        .expect("the history must compact");

    // The kept tail starts at the tool result, so its owning assistant is kept
    // too and no tool result is orphaned.
    assert_eq!(messages.len(), 5);
    assert_eq!(messages[2].role, Role::Assistant);
    assert_ne!(
        messages[2].tool_calls,
        Vec::<openjiuwen_foundation::ToolCall>::new()
    );
    assert_eq!(messages[3].role, Role::Tool);
    assert_eq!(messages[4].text(), "a3");
    assert_eq!(validate_tool_pairs(&messages), Vec::<usize>::new());
    assert!(messages[1]
        .text()
        .contains("Recent messages are preserved verbatim."));

    // Without pair extension the assistant would be dropped.
    let mut without = original;
    let compressor = FullCompactProcessor::new(
        FullCompactProcessorConfig {
            keep_tool_message_pairs: false,
            ..config
        },
        None,
        None,
    );
    compressor
        .process(&mut without, &fixture.context())
        .await
        .unwrap()
        .expect("the history must still compact");
    assert_eq!(without.len(), 4);
    assert_eq!(without[2].role, Role::Tool);
    assert_eq!(without[3].text(), "a3");
}

#[tokio::test]
async fn the_model_path_uses_the_model_summary() {
    let fixture = Fixture::new();
    let model: Arc<dyn Model> = Arc::new(ScriptedModel::text("MODEL SUMMARY"));
    let config = FullCompactProcessorConfig {
        messages_to_keep: 1,
        ..compact_config()
    };
    let compressor = FullCompactProcessor::new(config, None, None);
    let mut messages = conversation();

    let event = compressor
        .process(&mut messages, &fixture.model_context(&model))
        .await
        .unwrap()
        .expect("the history must compact");

    let summary_message = messages[1].text().into_owned();
    assert!(summary_message.starts_with(FULL_COMPACT_SUMMARY_INTRO));
    assert!(summary_message.contains("MODEL SUMMARY"));
    assert!(summary_message.contains("Recent messages are preserved verbatim."));
    assert_eq!(messages.len(), 3);
    assert_eq!(messages[2].text(), long("a2"));
    assert_eq!(event.summary.as_deref(), Some("MODEL SUMMARY"));
}

#[tokio::test]
async fn the_session_memory_path_prefers_the_stored_notes() {
    let fixture = Fixture::new();
    let store = Arc::new(InMemorySessionMemoryStore::new());

    let mut messages = vec![BaseMessage::system("persona")];
    for index in 1..=4 {
        let mut user = BaseMessage::user(long(&format!("u{index}")));
        stamp_message_id(&mut user);
        messages.push(user);
        let mut assistant = BaseMessage::assistant(long(&format!("a{index}")));
        stamp_message_id(&mut assistant);
        messages.push(assistant);
    }
    let anchor = messages[4].metadata["context_message_id"]
        .as_str()
        .unwrap()
        .to_owned();
    store
        .put(
            &fixture.scope,
            SessionMemoryNotes {
                notes: "the notes".to_owned(),
                anchor_message_id: Some(anchor),
            },
        )
        .await
        .unwrap();

    let config = FullCompactProcessorConfig {
        // The stored notes cover five of the nine messages, so the candidate is
        // far under the trigger while the raw history is far above it.
        trigger_total_tokens: 40_000,
        messages_to_keep: 10,
        ..FullCompactProcessorConfig::default()
    };
    let compressor = FullCompactProcessor::new(config, None, Some(store));
    let before = count_messages(&fixture.counter, &messages);

    let event = compressor
        .process(&mut messages, &fixture.context())
        .await
        .unwrap()
        .expect("the session-memory candidate must be accepted");

    // Boundary + notes summary, then the messages after the anchor verbatim.
    assert_eq!(messages.len(), 6);
    assert_eq!(messages[0].role, Role::System);
    assert!(messages[0].text().starts_with(SESSION_MEMORY_BOUNDARY));
    assert!(messages[1].text().contains("the notes"));
    assert!(messages[1]
        .text()
        .contains("Recent messages are preserved verbatim."));
    assert_eq!(messages[2].text(), long("u3"));
    assert_eq!(messages[5].text(), long("a4"));
    assert_eq!(event.summary.as_deref(), Some("the notes"));
    assert!(event.tokens_after < before);
}

#[tokio::test]
async fn the_session_memory_path_falls_back_when_notes_are_missing() {
    let fixture = Fixture::new();
    let store = Arc::new(InMemorySessionMemoryStore::new());
    let compressor = FullCompactProcessor::new(
        compact_config(),
        None,
        Some(store as Arc<dyn SessionMemoryStore>),
    );
    let mut messages = conversation();

    let event = compressor
        .process(&mut messages, &fixture.context())
        .await
        .unwrap()
        .expect("the summary path takes over");

    assert!(event
        .summary
        .is_some_and(|text| text.starts_with("Summary:\n")));
    assert_eq!(messages[0].text(), FULL_COMPACT_BOUNDARY_TEXT);
}

#[test]
fn full_compact_config_round_trips() {
    let config = FullCompactProcessorConfig::default();
    assert!(config.session_memory_enabled);
    assert!(config.keep_tool_message_pairs);
    let json = serde_json::to_string(&config).unwrap();
    assert_eq!(
        serde_json::from_str::<FullCompactProcessorConfig>(&json).unwrap(),
        config
    );
    assert_eq!(
        serde_json::from_str::<FullCompactProcessorConfig>("{}").unwrap(),
        config
    );
}

/// A conversation with `per_tool` results of each tool, closing on an assistant.
fn tools_history(tools: &[&str], per_tool: usize) -> Vec<BaseMessage> {
    let mut messages = vec![BaseMessage::user("go")];
    for tool in tools {
        for index in 0..per_tool {
            messages.push(BaseMessage::assistant("").with_tool_calls(vec![
                ToolCall::new(*tool).with_id(format!("{tool}_{index}")),
            ]));
            messages.push(BaseMessage::tool(
                format!("{tool}_{index}"),
                format!("{tool} output {index}: body"),
            ));
        }
    }
    messages.push(BaseMessage::assistant("done"));
    messages
}

/// A micro-compact configuration with a small keep window.
fn micro_config() -> MicroCompactProcessorConfig {
    MicroCompactProcessorConfig {
        trigger_threshold: 2,
        keep_recent_per_tool: 3,
        ..MicroCompactProcessorConfig::default()
    }
}

/// The tool results that still carry their original text.
fn uncleared(messages: &[BaseMessage]) -> Vec<usize> {
    messages
        .iter()
        .enumerate()
        .filter(|(_, message)| {
            message.role == Role::Tool && message.text() != MICRO_COMPACT_CLEARED_MARKER
        })
        .map(|(index, _)| index)
        .collect()
}

#[tokio::test]
async fn micro_compact_keeps_the_newest_results_per_tool() {
    let fixture = Fixture::new();
    let compressor = MicroCompactProcessor::new(micro_config());
    let mut messages = tools_history(&["read_file"], 10);
    let before = count_messages(&fixture.counter, &messages);

    let event = compressor
        .process(&mut messages, &fixture.context())
        .await
        .unwrap()
        .expect("ten results clear the gate");

    assert_eq!(event.processor, "micro_compact");
    assert_eq!(event.kind, ContextEventKind::ToolResultTrimmed);
    // Ten results, three kept: seven cleared.
    assert_eq!(event.messages_changed.len(), 7);
    assert_eq!(uncleared(&messages).len(), 3);
    assert_eq!(messages.len(), tools_history(&["read_file"], 10).len());
    assert_eq!(validate_tool_pairs(&messages), Vec::<usize>::new());
    assert_eq!(event.tokens_before, before);
    assert_eq!(
        event.tokens_after,
        count_messages(&fixture.counter, &messages)
    );
    for index in &event.messages_changed {
        assert_eq!(messages[*index].text(), MICRO_COMPACT_CLEARED_MARKER);
    }
}

#[tokio::test]
async fn micro_compact_groups_results_by_tool_name() {
    let fixture = Fixture::new();
    let compressor = MicroCompactProcessor::new(micro_config());
    let mut messages = tools_history(&["grep", "read_file"], 6);

    compressor
        .process(&mut messages, &fixture.context())
        .await
        .unwrap()
        .expect("both groups clear the gate");

    // Six results per tool, three kept each.
    assert_eq!(uncleared(&messages).len(), 6);
    let cleared_grep = messages
        .iter()
        .filter(|message| {
            message.role == Role::Tool
                && message
                    .tool_call_id
                    .as_deref()
                    .is_some_and(|id| id.starts_with("grep"))
                && message.text() == MICRO_COMPACT_CLEARED_MARKER
        })
        .count();
    assert_eq!(cleared_grep, 3);
}

#[tokio::test]
async fn micro_compact_ignores_tools_outside_the_list() {
    let fixture = Fixture::new();
    let compressor = MicroCompactProcessor::new(micro_config());
    let mut messages = tools_history(&["shell"], 10);
    let untouched = messages.clone();

    assert!(compressor
        .process(&mut messages, &fixture.context())
        .await
        .unwrap()
        .is_none());
    assert_eq!(messages, untouched);
}

#[tokio::test]
async fn micro_compact_honours_the_threshold_and_manual_mode() {
    let fixture = Fixture::new();
    let compressor = MicroCompactProcessor::new(micro_config());

    // Exactly keep + threshold results: the gate is not exceeded.
    let mut at_gate = tools_history(&["glob"], 5);
    let untouched = at_gate.clone();
    assert!(compressor
        .process(&mut at_gate, &fixture.context())
        .await
        .unwrap()
        .is_none());
    assert_eq!(at_gate, untouched);

    // Manual mode only keeps the tail.
    let mut manual = tools_history(&["glob"], 4);
    let event = compressor
        .process(&mut manual, &fixture.manual_context())
        .await
        .unwrap()
        .expect("manual mode clears above the keep window");
    assert_eq!(event.messages_changed.len(), 1);
    assert_eq!(uncleared(&manual).len(), 3);
}

#[tokio::test]
async fn micro_compact_preserves_identity_and_is_idempotent() {
    let fixture = Fixture::new();
    let compressor = MicroCompactProcessor::new(micro_config());
    let mut messages = vec![
        BaseMessage::user("go"),
        BaseMessage::assistant("")
            .with_tool_calls(vec![ToolCall::new("read_file").with_id("call_0")]),
        BaseMessage::tool("call_0", "first body")
            .with_name("read_file")
            .with_metadata("source", json!("index")),
    ];
    for index in 1..=5 {
        messages.push(BaseMessage::assistant("").with_tool_calls(vec![
            ToolCall::new("read_file").with_id(format!("call_{index}")),
        ]));
        messages.push(BaseMessage::tool(
            format!("call_{index}"),
            format!("body {index}"),
        ));
    }
    messages.push(BaseMessage::assistant("done"));

    compressor
        .process(&mut messages, &fixture.context())
        .await
        .unwrap()
        .expect("six results clear the gate");

    // Identity fields of the cleared message survive the rewrite.
    let cleared = &messages[2];
    assert_eq!(cleared.text(), MICRO_COMPACT_CLEARED_MARKER);
    assert_eq!(cleared.role, Role::Tool);
    assert_eq!(cleared.tool_call_id.as_deref(), Some("call_0"));
    assert_eq!(cleared.name.as_deref(), Some("read_file"));
    assert_eq!(cleared.metadata.get("source"), Some(&json!("index")));

    // A second run finds nothing left to clear.
    let after_first = messages.clone();
    assert!(compressor
        .process(&mut messages, &fixture.context())
        .await
        .unwrap()
        .is_none());
    assert_eq!(messages, after_first);
}

#[tokio::test]
async fn micro_compact_waits_for_a_completed_api_round() {
    let fixture = Fixture::new();
    let compressor = MicroCompactProcessor::new(micro_config());
    let mut messages = tools_history(&["grep"], 10);
    messages.push(BaseMessage::user("next turn"));
    let untouched = messages.clone();

    assert!(compressor
        .process(&mut messages, &fixture.context())
        .await
        .unwrap()
        .is_none());
    assert_eq!(messages, untouched);
}

#[test]
fn processor_config_tags_and_pipeline_order() {
    let json = r#"[
        {"type": "micro_compact"},
        {"type": "round_level"},
        {"type": "full_compact"}
    ]"#;
    let configs: Vec<ProcessorConfig> = serde_json::from_str(json).unwrap();
    assert!(matches!(configs[0], ProcessorConfig::MicroCompact(_)));
    assert!(matches!(configs[1], ProcessorConfig::RoundLevel(_)));
    assert!(matches!(configs[2], ProcessorConfig::FullCompact(_)));
    assert_eq!(
        serde_json::from_str::<Vec<ProcessorConfig>>(&serde_json::to_string(&configs).unwrap())
            .unwrap(),
        configs
    );

    let store: Arc<dyn OffloadStore> = Arc::new(InMemoryOffloadStore::new());
    let pipeline = ContextPipeline::from_configs(&configs, None, store).unwrap();
    let names: Vec<&str> = pipeline
        .processors()
        .iter()
        .map(|processor| processor.name())
        .collect();
    assert_eq!(names, vec!["micro_compact", "round_level", "full_compact"]);

    // Invalid configurations are rejected while building the pipeline.
    let store: Arc<dyn OffloadStore> = Arc::new(InMemoryOffloadStore::new());
    let error = ContextPipeline::from_configs(
        &[ProcessorConfig::RoundLevel(RoundLevelCompressorConfig {
            target_total_tokens: 0,
            ..RoundLevelCompressorConfig::default()
        })],
        None,
        store,
    )
    .unwrap_err();
    assert!(matches!(error, openjiuwen_common::error::Error::Config(_)));

    let store: Arc<dyn OffloadStore> = Arc::new(InMemoryOffloadStore::new());
    let error = ContextPipeline::from_configs(
        &[ProcessorConfig::MicroCompact(MicroCompactProcessorConfig {
            trigger_threshold: 0,
            ..MicroCompactProcessorConfig::default()
        })],
        None,
        store,
    )
    .unwrap_err();
    assert!(matches!(error, openjiuwen_common::error::Error::Config(_)));
}
