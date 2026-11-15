#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//! Integration tests for the engine/context plumbing.
//!
//! These pin how the subsystems added around the core engine are wired into it:
//! the session-memory and recall stores the engine resolves and hands to a
//! context, how [`ContextPipeline::from_configs_with`] injects them into the
//! processors it builds, the reload hint [`ConversationContext::prepare`]
//! injects when offload markers are present, and the [`ProcessorContext`] fields
//! a custom processor sees.
//!
//! Everything is offline and deterministic: the counter is
//! [`ApproxTokenCounter::with_chars_per_token`] at one character per token,
//! offload and memory stores are the in-memory implementations, and the only
//! model is [`openjiuwen_test_support::ScriptedModel`].
//!
//! [`ContextPipeline::from_configs_with`]: openjiuwen_context::processor::ContextPipeline::from_configs_with
//! [`ConversationContext::prepare`]: openjiuwen_context::ConversationContext::prepare
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use openjiuwen_common::error::Result;
use openjiuwen_context::processor::MessageSummaryOffloaderConfig;
use openjiuwen_context::{
    has_offload_markers, CompressionRecallConfig, CompressionStatus, ContextConfig, ContextEngine,
    ContextEvent, ContextOptions, ContextPipeline, ContextProcessor, ConversationContext,
    FullCompactProcessorConfig, InMemoryOffloadStore, InMemoryRecallArchiveStore,
    InMemorySessionMemoryStore, ModelContext, OffloadMarker, OffloadScope, OffloadStore,
    PrepareOptions, ProcessorConfig, ProcessorContext, RecallArchiveStore, SessionMemoryNotes,
    SessionMemoryStore, SummarizationConfig, TokenCounterConfig, TokenCounterKind,
    TruncationConfig, SESSION_MEMORY_BOUNDARY,
};
use openjiuwen_foundation::llm::Model;
use openjiuwen_foundation::prompt::{ApproxTokenCounter, BaseMessage, Role, ToolCall};
use openjiuwen_test_support::ScriptedModel;
use tokio_util::sync::CancellationToken;

/// The one-character-per-token counter every test uses.
fn counter() -> Arc<ApproxTokenCounter> {
    Arc::new(ApproxTokenCounter::with_chars_per_token(1))
}

/// Context options carrying the deterministic counter.
fn options() -> ContextOptions {
    ContextOptions {
        token_counter: Some(counter()),
        ..ContextOptions::default()
    }
}

/// A body long enough that replacing it with a summary is a real saving.
fn long(text: &str) -> String {
    format!("{text}: {}", "detail ".repeat(300))
}

/// A four-round conversation whose messages carry stable ids.
fn stamped_conversation() -> Vec<BaseMessage> {
    let mut messages = vec![BaseMessage::system("persona")];
    for index in 1..=4 {
        let mut user = BaseMessage::user(long(&format!("u{index}")));
        openjiuwen_context::stamp_message_id(&mut user);
        messages.push(user);
        let mut assistant = BaseMessage::assistant(long(&format!("a{index}")));
        openjiuwen_context::stamp_message_id(&mut assistant);
        messages.push(assistant);
    }
    messages
}

/// The `context_message_id` stamped into `message`.
fn message_id(message: &BaseMessage) -> String {
    message.metadata["context_message_id"]
        .as_str()
        .unwrap()
        .to_owned()
}

// ---------------------------------------------------------------------------
// Session-memory wiring: FullCompact built by from_configs_with
// ---------------------------------------------------------------------------

#[tokio::test]
async fn full_compact_uses_the_engine_session_memory_store() {
    let history = stamped_conversation();
    // The notes cover everything through the second answer.
    let anchor = message_id(&history[4]);
    let before = history.len();

    let session_store = Arc::new(InMemorySessionMemoryStore::new());
    let engine = ContextEngine::with_stores(
        ContextConfig {
            processors: vec![ProcessorConfig::FullCompact(FullCompactProcessorConfig {
                // The candidate (boundary + notes + the four messages after the
                // anchor) fits; the raw history does not.
                trigger_total_tokens: 12_000,
                messages_to_keep: 10,
                ..FullCompactProcessorConfig::default()
            })],
            ..ContextConfig::default()
        },
        Arc::new(InMemoryOffloadStore::new()),
        None,
        Some(session_store.clone()),
    );
    let context = engine.create_context_with("main", options()).await.unwrap();
    assert!(context.session_memory_store().is_some());
    assert_eq!(context.processors()[0].name(), "full_compact");

    session_store
        .put(
            &context.scope(),
            SessionMemoryNotes {
                notes: "the notes body".to_owned(),
                anchor_message_id: Some(anchor),
            },
        )
        .await
        .unwrap();
    context.set_messages(history.clone()).await.unwrap();

    let report = engine.compress(&context).await.unwrap();
    assert_eq!(report.status, CompressionStatus::Compressed);
    assert!(report.tokens_after < report.tokens_before);

    let messages = context.messages().await;
    assert!(messages.len() < before);
    assert_eq!(messages[0].role, Role::System);
    assert!(messages[0].text().starts_with(SESSION_MEMORY_BOUNDARY));
    assert!(messages
        .iter()
        .any(|message| message.text().contains("the notes body")));
    // The messages after the anchor are kept verbatim and stay paired.
    assert_eq!(messages.last().unwrap().text(), long("a4"));
    assert_eq!(
        openjiuwen_context::validate_tool_pairs(&messages),
        Vec::<usize>::new()
    );
}

// ---------------------------------------------------------------------------
// Model wiring: MessageSummaryOffloader built by from_configs_with
// ---------------------------------------------------------------------------

#[tokio::test]
async fn summary_offloader_uses_the_engine_model_and_offload_store() {
    let scripted = ScriptedModel::builder()
        .respond_text("a compact summary")
        .build();
    let model: Arc<dyn Model> = Arc::new(scripted.clone());
    let store: Arc<dyn OffloadStore> = Arc::new(InMemoryOffloadStore::new());
    let engine = ContextEngine::with_store(
        ContextConfig {
            processors: vec![ProcessorConfig::MessageSummaryOffloader(
                MessageSummaryOffloaderConfig {
                    large_message_threshold: 100,
                    ..MessageSummaryOffloaderConfig::default()
                },
            )],
            ..ContextConfig::default()
        },
        Arc::clone(&store),
    );
    let context = engine
        .create_context_with(
            "main",
            ContextOptions {
                model: Some(model),
                history: vec![
                    BaseMessage::user("run"),
                    BaseMessage::assistant("")
                        .with_tool_calls(vec![ToolCall::new("grep").with_id("call_1")]),
                    BaseMessage::tool("call_1", "z".repeat(3_000)),
                ],
                ..options()
            },
        )
        .await
        .unwrap();

    let report = engine.compress(&context).await.unwrap();
    assert_eq!(report.status, CompressionStatus::Compressed);
    assert_eq!(report.events.len(), 1);
    assert_eq!(scripted.call_count(), 1);

    let messages = context.messages().await;
    let text = messages[2].text().to_string();
    assert!(text.starts_with("a compact summary\n\n"));
    assert!(text.contains("original_chars: 3000"));
    let marker = OffloadMarker::parse(&text).unwrap();
    let stored = store
        .get(&context.scope(), &marker.handle)
        .await
        .unwrap()
        .expect("the original tool result is offloaded");
    assert_eq!(stored.messages[0].text(), "z".repeat(3_000));
    assert_eq!(messages[2].tool_call_id.as_deref(), Some("call_1"));
}

// ---------------------------------------------------------------------------
// Recall store resolution
// ---------------------------------------------------------------------------

#[tokio::test]
async fn compression_recall_enables_a_default_recall_store() {
    let engine = ContextEngine::new(ContextConfig {
        compression_recall: CompressionRecallConfig {
            enabled: true,
            ..CompressionRecallConfig::default()
        },
        ..ContextConfig::default()
    });
    let context = engine.create_context("main", None).await.unwrap();
    assert!(context.recall_store().is_some());

    let disabled = ContextEngine::new(ContextConfig::default());
    let context = disabled.create_context("main", None).await.unwrap();
    assert!(context.recall_store().is_none());
}

#[tokio::test]
async fn an_explicit_recall_store_is_used_unchanged() {
    let store: Arc<dyn RecallArchiveStore> = Arc::new(InMemoryRecallArchiveStore::new());
    let engine = ContextEngine::new(ContextConfig::default());
    let context = engine
        .create_context_with(
            "main",
            ContextOptions {
                recall: Some(Arc::clone(&store)),
                ..options()
            },
        )
        .await
        .unwrap();
    let wired = context.recall_store().expect("the explicit store is wired");
    assert!(Arc::ptr_eq(wired, &store));
}

// ---------------------------------------------------------------------------
// Session-memory store resolution
// ---------------------------------------------------------------------------

#[tokio::test]
async fn full_compaction_enables_a_default_session_memory_store() {
    let engine = ContextEngine::new(ContextConfig {
        processors: vec![ProcessorConfig::FullCompact(
            FullCompactProcessorConfig::default(),
        )],
        ..ContextConfig::default()
    });
    let context = engine.create_context("main", None).await.unwrap();
    assert!(context.session_memory_store().is_some());

    let engine = ContextEngine::new(ContextConfig {
        processors: vec![ProcessorConfig::MessageSummaryOffloader(
            MessageSummaryOffloaderConfig::default(),
        )],
        ..ContextConfig::default()
    });
    let context = engine.create_context("main", None).await.unwrap();
    assert!(context.session_memory_store().is_some());

    // The default pipeline neither compacts nor summary-offloads.
    let engine = ContextEngine::new(ContextConfig::default());
    let context = engine.create_context("main", None).await.unwrap();
    assert!(context.session_memory_store().is_none());
}

// ---------------------------------------------------------------------------
// Reload hint injection during prepare
// ---------------------------------------------------------------------------

/// A history whose tool result is hidden behind an offload marker.
fn marked_history() -> Vec<BaseMessage> {
    vec![
        BaseMessage::system("persona"),
        BaseMessage::user("run"),
        BaseMessage::assistant("").with_tool_calls(vec![ToolCall::new("grep").with_id("call_1")]),
        BaseMessage::tool("call_1", "[[OFFLOAD: handle=x, type=in_memory]]"),
    ]
}

/// The prepared messages with no tools and no window limit.
async fn prepared_messages(engine: &ContextEngine) -> Vec<BaseMessage> {
    let context = engine
        .create_context_with(
            "main",
            ContextOptions {
                history: marked_history(),
                processors: Some(Vec::new()),
                ..options()
            },
        )
        .await
        .unwrap();
    let prepared = engine
        .prepare(&context, &PrepareOptions::default())
        .await
        .unwrap();
    prepared.messages
}

#[tokio::test]
async fn prepare_injects_the_reload_hint_after_the_system_messages() {
    let engine = ContextEngine::new(ContextConfig {
        enable_reload: true,
        ..ContextConfig::default()
    });
    let messages = prepared_messages(&engine).await;
    assert!(has_offload_markers(&messages));
    assert_eq!(messages.len(), 5);
    // The existing system message stays first, the hint follows it, and the
    // conversation keeps its order.
    assert_eq!(messages[0].role, Role::System);
    assert_eq!(messages[0].text(), "persona");
    assert_eq!(messages[1].role, Role::System);
    assert!(messages[1]
        .text()
        .starts_with("# Context Compression and Recovery"));
    assert_eq!(messages[2].text(), "run");

    // Preparing twice never duplicates the section.
    let context = engine.create_context_with("main", options()).await.unwrap();
    context.set_messages(marked_history()).await.unwrap();
    let first = engine
        .prepare(&context, &PrepareOptions::default())
        .await
        .unwrap();
    let second = engine
        .prepare(&context, &PrepareOptions::default())
        .await
        .unwrap();
    assert_eq!(first.messages, second.messages);
    assert_eq!(
        first
            .messages
            .iter()
            .filter(|message| message
                .text()
                .contains("# Context Compression and Recovery"))
            .count(),
        1
    );
}

#[tokio::test]
async fn the_reload_hint_needs_markers_and_the_enabled_flag() {
    // No marker: nothing is injected even with reload enabled.
    let engine = ContextEngine::new(ContextConfig {
        enable_reload: true,
        ..ContextConfig::default()
    });
    let context = engine
        .create_context_with(
            "main",
            ContextOptions {
                history: vec![
                    BaseMessage::system("persona"),
                    BaseMessage::user("hello"),
                    BaseMessage::assistant("hi"),
                ],
                processors: Some(Vec::new()),
                ..options()
            },
        )
        .await
        .unwrap();
    let prepared = engine
        .prepare(&context, &PrepareOptions::default())
        .await
        .unwrap();
    assert!(!has_offload_markers(&prepared.messages));
    assert!(!prepared.messages.iter().any(|message| message
        .text()
        .starts_with("# Context Compression and Recovery")));

    // Markers present but reload disabled: nothing is injected.
    let engine = ContextEngine::new(ContextConfig {
        enable_reload: false,
        ..ContextConfig::default()
    });
    let messages = prepared_messages(&engine).await;
    assert!(has_offload_markers(&messages));
    assert!(!messages.iter().any(|message| message
        .text()
        .starts_with("# Context Compression and Recovery")));
}

// ---------------------------------------------------------------------------
// ProcessorContext carries the engine-wired stores
// ---------------------------------------------------------------------------

/// A processor that records whether the optional stores were reachable.
struct StoreProbe {
    seen: Arc<Mutex<Option<(bool, bool)>>>,
}

#[async_trait]
impl ContextProcessor for StoreProbe {
    fn name(&self) -> &'static str {
        "store_probe"
    }

    async fn process(
        &self,
        _messages: &mut Vec<BaseMessage>,
        cx: &ProcessorContext<'_>,
    ) -> Result<Option<ContextEvent>> {
        *self.seen.lock().unwrap() = Some((cx.recall.is_some(), cx.session_memory.is_some()));
        Ok(None)
    }
}

/// Runs [`StoreProbe`] with the stores the engine wired into `context`.
async fn probe_stores(context: &Arc<ConversationContext>) -> (bool, bool) {
    let seen = Arc::new(Mutex::new(None));
    let pipeline = ContextPipeline::new(vec![Arc::new(StoreProbe {
        seen: Arc::clone(&seen),
    })]);
    // The probe does not count, so a local counter keeps the borrows simple;
    // `ModelContext::token_counter` erases the `Send + Sync` bounds the field
    // requires.
    let counter: Arc<dyn openjiuwen_foundation::prompt::TokenCounter + Send + Sync> =
        Arc::new(ApproxTokenCounter::with_chars_per_token(1));
    let model = context.model();
    let scope = context.scope();
    let cancellation = CancellationToken::new();
    let cx = ProcessorContext {
        config: context.config(),
        counter: counter.as_ref(),
        model: model.as_ref(),
        offload: context.offload_store().as_ref(),
        scope: &scope,
        budget: 1_000,
        trigger_tokens: 800,
        manual: false,
        cancellation: &cancellation,
        recall: context.recall_store().map(|store| &**store),
        session_memory: context.session_memory_store().map(|store| &**store),
        session_memory_agent: None,
    };
    let mut messages = vec![BaseMessage::user("hi")];
    pipeline.run(&mut messages, &cx).await.unwrap();
    let seen = *seen.lock().unwrap();
    seen.expect("the probe processor ran")
}

#[tokio::test]
async fn a_processor_sees_the_stores_the_engine_wired() {
    let recall: Arc<dyn RecallArchiveStore> = Arc::new(InMemoryRecallArchiveStore::new());
    let session_memory: Arc<dyn SessionMemoryStore> = Arc::new(InMemorySessionMemoryStore::new());
    let engine = ContextEngine::with_stores(
        ContextConfig::default(),
        Arc::new(InMemoryOffloadStore::new()),
        Some(recall),
        Some(session_memory),
    );
    let context = engine.create_context_with("main", options()).await.unwrap();
    assert_eq!(probe_stores(&context).await, (true, true));

    // Without either store the probe sees none.
    let engine = ContextEngine::new(ContextConfig {
        token_counter: TokenCounterConfig {
            kind: TokenCounterKind::Characters,
            chars_per_token: 1,
        },
        ..ContextConfig::default()
    });
    let context = engine.create_context("main", None).await.unwrap();
    assert_eq!(probe_stores(&context).await, (false, false));
}

// ---------------------------------------------------------------------------
// Pipeline wiring
// ---------------------------------------------------------------------------

#[tokio::test]
async fn from_configs_with_wires_every_store_it_is_given() {
    let offload: Arc<dyn OffloadStore> = Arc::new(InMemoryOffloadStore::new());
    let session_memory: Arc<dyn SessionMemoryStore> = Arc::new(InMemorySessionMemoryStore::new());
    let recall: Arc<dyn RecallArchiveStore> = Arc::new(InMemoryRecallArchiveStore::new());
    let deps = openjiuwen_context::PipelineDeps::new(Arc::clone(&offload))
        .with_recall(Some(Arc::clone(&recall)))
        .with_session_memory(Some(Arc::clone(&session_memory)));
    assert!(format!("{deps:?}").contains("has_session_memory: true"));

    let pipeline = ContextPipeline::from_configs_with(
        &[
            ProcessorConfig::FullCompact(FullCompactProcessorConfig::default()),
            ProcessorConfig::MessageSummaryOffloader(MessageSummaryOffloaderConfig::default()),
            ProcessorConfig::Truncation(TruncationConfig::default()),
        ],
        &deps,
    )
    .unwrap();
    let names: Vec<&str> = pipeline
        .processors()
        .iter()
        .map(|processor| processor.name())
        .collect();
    assert_eq!(
        names,
        vec!["full_compact", "message_summary_offloader", "truncation"]
    );

    // Summarisation still demands a model; the other builders do not.
    let error = ContextPipeline::from_configs_with(
        &[ProcessorConfig::Summarization(
            SummarizationConfig::default(),
        )],
        &deps,
    )
    .unwrap_err();
    assert!(matches!(error, openjiuwen_common::error::Error::Config(_)));

    // The scope helper stays usable with the wiring above.
    let scope = OffloadScope::new("session", "main");
    assert_eq!(scope.session_id, "session");
}
