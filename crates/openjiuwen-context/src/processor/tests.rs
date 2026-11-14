//! Unit tests of the pipeline: ordering, event collection, error propagation,
//! cancellation, orphan-tool-result repair, and the serialized processor
//! configuration.

use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::prompt::{
    ApproxTokenCounter, BaseMessage, Role, TokenCounter, ToolCall,
};
use tokio_util::sync::CancellationToken;

use super::*;
use crate::config::ContextConfig;
use crate::events::{ContextEvent, ContextEventKind};
use crate::offload::{
    InMemoryOffloadStore, OffloadKind, OffloadRecord, OffloadScope, OffloadStore,
};

fn counter() -> ApproxTokenCounter {
    ApproxTokenCounter::with_chars_per_token(1)
}

/// A processor that records its name and appends nothing.
struct Recorder {
    name: &'static str,
    seen: Arc<std::sync::Mutex<Vec<&'static str>>>,
}

#[async_trait]
impl ContextProcessor for Recorder {
    fn name(&self) -> &'static str {
        self.name
    }

    async fn process(
        &self,
        _messages: &mut Vec<BaseMessage>,
        _cx: &ProcessorContext<'_>,
    ) -> Result<Option<ContextEvent>> {
        self.seen
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(self.name);
        Ok(Some(ContextEvent {
            processor: self.name.to_owned(),
            kind: ContextEventKind::Truncated,
            ..ContextEvent::default()
        }))
    }
}

struct Failing;

#[async_trait]
impl ContextProcessor for Failing {
    fn name(&self) -> &'static str {
        "failing"
    }

    async fn process(
        &self,
        _messages: &mut Vec<BaseMessage>,
        _cx: &ProcessorContext<'_>,
    ) -> Result<Option<ContextEvent>> {
        Err(Error::Llm("boom".to_owned()))
    }
}

#[test]
fn configs_deserialize_by_tag_and_round_trip() {
    let json = r#"[
            {"type": "tool_result", "max_result_tokens": 50},
            {"type": "truncation", "keep_recent_messages": 2},
            {"type": "summarization"}
        ]"#;
    let configs: Vec<ProcessorConfig> = serde_json::from_str(json).unwrap();
    assert!(matches!(configs[0], ProcessorConfig::ToolResult(_)));
    assert!(matches!(configs[1], ProcessorConfig::Truncation(_)));
    assert!(matches!(configs[2], ProcessorConfig::Summarization(_)));
    assert_eq!(
        serde_json::from_str::<Vec<ProcessorConfig>>(&serde_json::to_string(&configs).unwrap())
            .unwrap(),
        configs
    );
}

#[test]
fn from_configs_requires_a_model_for_summarization() {
    let store: Arc<dyn OffloadStore> = Arc::new(InMemoryOffloadStore::new());
    let error = ContextPipeline::from_configs(
        &[ProcessorConfig::Summarization(
            SummarizationConfig::default(),
        )],
        None,
        store,
    )
    .unwrap_err();
    assert!(matches!(error, Error::Config(_)));

    let store: Arc<dyn OffloadStore> = Arc::new(InMemoryOffloadStore::new());
    let pipeline = ContextPipeline::from_configs(
        &[
            ProcessorConfig::ToolResult(ToolResultConfig::default()),
            ProcessorConfig::Truncation(TruncationConfig::default()),
        ],
        None,
        store,
    )
    .unwrap();
    assert_eq!(pipeline.processors().len(), 2);
    assert_eq!(pipeline.processors()[0].name(), "tool_result");
    assert_eq!(pipeline.processors()[1].name(), "truncation");
}

fn context<'a>(
    config: &'a ContextConfig,
    counter: &'a (dyn TokenCounter + Send + Sync),
    store: &'a dyn OffloadStore,
    scope: &'a OffloadScope,
    cancellation: &'a CancellationToken,
) -> ProcessorContext<'a> {
    ProcessorContext {
        config,
        counter,
        model: None,
        offload: store,
        scope,
        budget: 1000,
        trigger_tokens: 800,
        manual: false,
        cancellation,
        recall: None,
        session_memory: None,
    }
}

#[tokio::test]
async fn run_is_ordered_and_collects_events() {
    let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
    let pipeline = ContextPipeline::new(vec![
        Arc::new(Recorder {
            name: "first",
            seen: Arc::clone(&seen),
        }),
        Arc::new(Recorder {
            name: "second",
            seen: Arc::clone(&seen),
        }),
    ]);
    let config = ContextConfig::default();
    let counter = counter();
    let store = InMemoryOffloadStore::new();
    let scope = OffloadScope::new("s", "main");
    let cancellation = CancellationToken::new();
    let cx = context(&config, &counter, &store, &scope, &cancellation);
    let mut messages = vec![BaseMessage::user("hi")];

    let events = pipeline.run(&mut messages, &cx).await.unwrap();
    assert_eq!(events.len(), 2);
    assert_eq!(events[0].processor, "first");
    assert_eq!(events[1].processor, "second");
    assert_eq!(
        *seen
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner),
        vec!["first", "second"]
    );
}

#[tokio::test]
async fn run_propagates_errors_and_checks_cancellation() {
    let config = ContextConfig::default();
    let counter = counter();
    let store = InMemoryOffloadStore::new();
    let scope = OffloadScope::new("s", "main");
    let cancellation = CancellationToken::new();
    let cx = context(&config, &counter, &store, &scope, &cancellation);
    let mut messages = vec![BaseMessage::user("hi")];

    let pipeline = ContextPipeline::new(vec![Arc::new(Failing)]);
    let error = pipeline.run(&mut messages, &cx).await.unwrap_err();
    assert!(matches!(error, Error::Llm(_)));

    let cancelled = CancellationToken::new();
    cancelled.cancel();
    let pipeline = ContextPipeline::new(vec![Arc::new(Recorder {
        name: "never",
        seen: Arc::new(std::sync::Mutex::new(Vec::new())),
    })]);
    let cx = ProcessorContext {
        config: &config,
        counter: &counter,
        model: None,
        offload: &store,
        scope: &scope,
        budget: 1000,
        trigger_tokens: 800,
        manual: false,
        cancellation: &cancelled,
        recall: None,
        session_memory: None,
    };
    let error = pipeline.run(&mut messages, &cx).await.unwrap_err();
    assert!(matches!(error, Error::Cancelled));
}

#[tokio::test]
async fn run_repairs_orphan_tool_results() {
    struct DropAssistant;
    #[async_trait]
    impl ContextProcessor for DropAssistant {
        fn name(&self) -> &'static str {
            "drop_assistant"
        }
        async fn process(
            &self,
            messages: &mut Vec<BaseMessage>,
            _cx: &ProcessorContext<'_>,
        ) -> Result<Option<ContextEvent>> {
            messages.retain(|message| message.role != Role::Assistant);
            Ok(None)
        }
    }

    let config = ContextConfig::default();
    let counter = counter();
    let store = InMemoryOffloadStore::new();
    let scope = OffloadScope::new("s", "main");
    let cancellation = CancellationToken::new();
    let cx = context(&config, &counter, &store, &scope, &cancellation);
    let mut messages = vec![
        BaseMessage::user("q"),
        BaseMessage::assistant("").with_tool_calls(vec![ToolCall::new("t").with_id("call_1")]),
        BaseMessage::tool("call_1", "result"),
    ];
    let pipeline = ContextPipeline::new(vec![Arc::new(DropAssistant)]);
    pipeline.run(&mut messages, &cx).await.unwrap();
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0].role, Role::User);
}

#[test]
fn processors_are_dyn_compatible() {
    let processors: Vec<Arc<dyn ContextProcessor>> = vec![
        Arc::new(TruncationProcessor {
            config: TruncationConfig::default(),
        }),
        Arc::new(ToolResultProcessor {
            config: ToolResultConfig::default(),
            store: Arc::new(InMemoryOffloadStore::new()),
        }),
    ];
    assert_eq!(processors[0].name(), "truncation");
    assert_eq!(processors[1].name(), "tool_result");
    assert_eq!(processors[1].save_state(), serde_json::Value::Null);
    let _ = OffloadKind::InMemory;
    let _ = OffloadRecord::default();
}
