#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//! Public-API tests for the four offloader processors.
//!
//! The suites drive [`MessageOffloader`], [`ToolResultBudgetProcessor`],
//! [`ToolResultWindowProcessor`], and [`MessageSummaryOffloader`] the way the
//! pipeline does: they assert what is rewritten, what survives, the offload
//! store contents, the `<persisted-output>` / `<windowed-output>` placeholders,
//! the managed metadata flag, the reported [`ContextEvent`], and the
//! `ProcessorConfig` wire tags.
//!
//! Everything is offline and deterministic: the counter is the `characters`
//! counter with a one-character-per-token ratio, offload storage is
//! [`InMemoryOffloadStore`], and the only model is
//! [`openjiuwen_test_support::ScriptedModel`].
use std::sync::Arc;

use openjiuwen_common::error::Error;
use openjiuwen_context::processor::{
    wildcard_match, ContextProcessor, MessageOffloadRole, MessageOffloader, MessageOffloaderConfig,
    MessageSummaryOffloader, MessageSummaryOffloaderConfig, ProcessorConfig, ProcessorContext,
    ToolResultBudgetProcessor, ToolResultBudgetProcessorConfig, ToolResultWindowProcessor,
    ToolResultWindowProcessorConfig, OFFLOADED_INFO_TAG, OMIT_STRING,
    SUMMARY_COMPRESSION_SYSTEM_PROMPT, WINDOWED_OUTPUT_TAG, WINDOW_MANAGED_METADATA_KEY,
};
use openjiuwen_context::{
    ContextConfig, ContextEventKind, ContextPipeline, InMemoryOffloadStore, OffloadKind,
    OffloadMarker, OffloadScope, OffloadStore, TokenCounterConfig, TokenCounterKind,
    PERSISTED_OUTPUT_TAG,
};
use openjiuwen_foundation::llm::Model;
use openjiuwen_foundation::prompt::{ApproxTokenCounter, BaseMessage, ToolCall};
use openjiuwen_test_support::ScriptedModel;
use tokio_util::sync::CancellationToken;

/// The one-character-per-token counter every test uses.
fn counter() -> ApproxTokenCounter {
    ApproxTokenCounter::with_chars_per_token(1)
}

/// A deterministic configuration: one character per token.
fn char_config() -> ContextConfig {
    ContextConfig {
        token_counter: TokenCounterConfig {
            kind: TokenCounterKind::Characters,
            chars_per_token: 1,
        },
        ..ContextConfig::default()
    }
}

/// Builds one tool exchange: a user turn, a tool-calling assistant, and its result.
fn tool_exchange(id: &str, tool: &str, output: &str) -> Vec<BaseMessage> {
    vec![
        BaseMessage::user("run"),
        BaseMessage::assistant("").with_tool_calls(vec![ToolCall::new(tool).with_id(id)]),
        BaseMessage::tool(id, output),
    ]
}

/// Builds one exchange whose tool call carries JSON arguments.
fn tool_exchange_with_args(
    id: &str,
    tool: &str,
    arguments: serde_json::Value,
    output: &str,
) -> Vec<BaseMessage> {
    vec![
        BaseMessage::user("run"),
        BaseMessage::assistant("").with_tool_calls(vec![ToolCall::new(tool)
            .with_id(id)
            .with_arguments(arguments)]),
        BaseMessage::tool(id, output),
    ]
}

/// A round with two tool results of the given sizes.
fn two_result_round(first: usize, second: usize) -> Vec<BaseMessage> {
    vec![
        BaseMessage::user("go"),
        BaseMessage::assistant("").with_tool_calls(vec![
            ToolCall::new("shell").with_id("c1"),
            ToolCall::new("shell").with_id("c2"),
        ]),
        BaseMessage::tool("c1", "a".repeat(first)),
        BaseMessage::tool("c2", "b".repeat(second)),
    ]
}

/// Shared processor inputs, so each test only spells out what it is testing.
struct Fixture {
    config: ContextConfig,
    counter: ApproxTokenCounter,
    store: Arc<InMemoryOffloadStore>,
    scope: OffloadScope,
    cancellation: CancellationToken,
    model: Option<Arc<dyn Model>>,
}

impl Fixture {
    /// A fixture with an empty store and no model.
    fn new() -> Self {
        Self {
            config: char_config(),
            counter: counter(),
            store: Arc::new(InMemoryOffloadStore::new()),
            scope: OffloadScope::new("session-1", "main"),
            cancellation: CancellationToken::new(),
            model: None,
        }
    }

    /// A fixture whose `ProcessorContext` carries `model`.
    fn with_context_model(model: Arc<dyn Model>) -> Self {
        Self {
            model: Some(model),
            ..Self::new()
        }
    }

    /// The `Arc<dyn OffloadStore>` handed to a processor.
    fn store(&self) -> Arc<dyn OffloadStore> {
        Arc::clone(&self.store) as Arc<dyn OffloadStore>
    }

    /// A context over the fixture.
    fn context(&self) -> ProcessorContext<'_> {
        ProcessorContext {
            config: &self.config,
            counter: &self.counter,
            model: self.model.as_ref(),
            offload: &*self.store,
            scope: &self.scope,
            budget: 100_000,
            trigger_tokens: 80_000,
            manual: false,
            cancellation: &self.cancellation,
        }
    }

    /// The stored original behind `marker_handle`.
    async fn stored(&self, handle: &str) -> Vec<BaseMessage> {
        self.store
            .get(&self.scope, handle)
            .await
            .unwrap()
            .unwrap()
            .messages
    }
}

// ---------------------------------------------------------------------------
// MessageOffloader
// ---------------------------------------------------------------------------

fn offloader_config() -> MessageOffloaderConfig {
    MessageOffloaderConfig {
        messages_threshold: None,
        tokens_threshold: 10,
        large_message_threshold: 100,
        offload_message_types: vec![MessageOffloadRole::Tool],
        protected_tool_names: vec!["read_file".to_owned()],
        trim_size: 20,
        messages_to_keep: None,
        keep_last_round: true,
    }
}

#[tokio::test]
async fn message_offloader_rewrites_old_messages_with_a_trim_and_a_marker() {
    let fixture = Fixture::new();
    let original = "x".repeat(1500);
    let mut messages = tool_exchange("call_1", "grep", &original);
    messages.push(BaseMessage::user("q2"));
    messages.push(BaseMessage::assistant("done"));

    let processor = MessageOffloader::new(offloader_config(), fixture.store());
    let event = processor
        .process(&mut messages, &fixture.context())
        .await
        .unwrap()
        .unwrap();

    assert_eq!(event.processor, "message_offloader");
    assert_eq!(event.kind, ContextEventKind::Offloaded);
    assert_eq!(event.messages_changed, vec![2]);
    assert!(event.tokens_after < event.tokens_before);

    // Trim + omission marker + offload marker, in that order.
    let text = messages[2].text().to_string();
    assert!(text.starts_with(&"x".repeat(20)));
    assert!(text.contains(&format!("{OMIT_STRING}[[OFFLOAD:")));
    let marker = OffloadMarker::parse(&text).unwrap();
    assert_eq!(marker.kind, OffloadKind::InMemory);
    assert_eq!(Some(marker.handle.clone()), event.offload_handle);

    // The pairing and the message identity survive the rewrite.
    assert_eq!(messages[2].role, openjiuwen_foundation::prompt::Role::Tool);
    assert_eq!(messages[2].tool_call_id.as_deref(), Some("call_1"));
    // Everything from the newest plain assistant on is untouched.
    assert_eq!(messages[3].text(), "q2");
    assert_eq!(messages[4].text(), "done");

    // The original message is in the store under the reported handle.
    let stored = fixture.stored(&marker.handle).await;
    assert_eq!(stored.len(), 1);
    assert_eq!(stored[0].text(), original);
    assert_eq!(stored[0].tool_call_id.as_deref(), Some("call_1"));

    // A second pass finds an offload marker and does nothing.
    assert!(processor
        .process(&mut messages, &fixture.context())
        .await
        .unwrap()
        .is_none());
}

#[tokio::test]
async fn message_offloader_keeps_the_last_round_only_with_keep_last_round() {
    let fixture = Fixture::new();
    let messages_builder = || {
        let mut messages = tool_exchange("call_1", "grep", "small");
        messages.push(BaseMessage::assistant("done"));
        messages.push(BaseMessage::user("u".repeat(1500)));
        messages
    };
    let config = MessageOffloaderConfig {
        offload_message_types: vec![MessageOffloadRole::User, MessageOffloadRole::Tool],
        ..offloader_config()
    };

    // The trailing user turn follows the newest plain assistant, so it is kept.
    let mut messages = messages_builder();
    let processor = MessageOffloader::new(config.clone(), fixture.store());
    assert!(processor
        .process(&mut messages, &fixture.context())
        .await
        .unwrap()
        .is_none());

    // Without the round rule the same history is offloaded.
    let mut messages = messages_builder();
    let processor = MessageOffloader::new(
        MessageOffloaderConfig {
            keep_last_round: false,
            ..config
        },
        fixture.store(),
    );
    let event = processor
        .process(&mut messages, &fixture.context())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(event.messages_changed, vec![4]);
    assert!(messages[4].text().contains("[[OFFLOAD:"));
}

#[tokio::test]
async fn message_offloader_respects_the_eligible_roles() {
    let fixture = Fixture::new();
    let history = || {
        vec![
            BaseMessage::user("u".repeat(1500)),
            BaseMessage::assistant("done"),
        ]
    };

    let mut messages = history();
    let processor = MessageOffloader::new(offloader_config(), fixture.store());
    assert!(processor
        .process(&mut messages, &fixture.context())
        .await
        .unwrap()
        .is_none());

    let mut messages = history();
    let processor = MessageOffloader::new(
        MessageOffloaderConfig {
            offload_message_types: vec![MessageOffloadRole::User],
            ..offloader_config()
        },
        fixture.store(),
    );
    let event = processor
        .process(&mut messages, &fixture.context())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(event.messages_changed, vec![0]);
}

#[tokio::test]
async fn message_offloader_skips_protected_tools_by_resolved_name_and_pattern() {
    let fixture = Fixture::new();
    let build = |tool: &str, arguments: Option<serde_json::Value>| {
        let mut messages = match arguments {
            Some(arguments) => {
                tool_exchange_with_args("call_1", tool, arguments, &"r".repeat(1500))
            }
            None => tool_exchange("call_1", tool, &"r".repeat(1500)),
        };
        messages.push(BaseMessage::assistant("done"));
        messages
    };
    let plain = MessageOffloader::new(offloader_config(), fixture.store());

    // `read_file` is protected by name.
    let mut messages = build("read_file", None);
    assert!(plain
        .process(&mut messages, &fixture.context())
        .await
        .unwrap()
        .is_none());

    // `grep:*.md` protects only calls whose string arguments match the pattern.
    let mut messages = build("grep", Some(serde_json::json!({"path": "docs/guide.md"})));
    let processor = MessageOffloader::new(
        MessageOffloaderConfig {
            protected_tool_names: vec!["grep:*.md".to_owned()],
            ..offloader_config()
        },
        fixture.store(),
    );
    assert!(processor
        .process(&mut messages, &fixture.context())
        .await
        .unwrap()
        .is_none());

    let processor = MessageOffloader::new(
        MessageOffloaderConfig {
            protected_tool_names: vec!["grep:*.rs".to_owned()],
            ..offloader_config()
        },
        fixture.store(),
    );
    let event = processor
        .process(&mut messages, &fixture.context())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(event.messages_changed, vec![2]);
}

#[tokio::test]
async fn message_offloader_skips_text_that_already_carries_a_marker() {
    let fixture = Fixture::new();
    let body = format!(
        "{} tail [[OFFLOAD: handle=abc, type=in_memory]]",
        "z".repeat(1500)
    );
    let mut messages = tool_exchange("call_1", "grep", &body);
    messages.push(BaseMessage::assistant("done"));

    let processor = MessageOffloader::new(offloader_config(), fixture.store());
    assert!(processor
        .process(&mut messages, &fixture.context())
        .await
        .unwrap()
        .is_none());
    assert_eq!(messages[2].text(), body);
    assert!(fixture.store.is_empty());
}

#[tokio::test]
async fn message_offloader_triggers_on_messages_or_tokens_and_needs_a_candidate() {
    let fixture = Fixture::new();
    let history = || {
        let mut messages = tool_exchange("call_1", "grep", &"x".repeat(1500));
        messages.push(BaseMessage::assistant("done"));
        messages
    };
    let short_history = || {
        vec![
            BaseMessage::user("a"),
            BaseMessage::assistant("b"),
            BaseMessage::user("c"),
            BaseMessage::assistant("d"),
        ]
    };

    // Message-count trigger.
    let mut messages = history();
    let processor = MessageOffloader::new(
        MessageOffloaderConfig {
            messages_threshold: Some(3),
            tokens_threshold: 1_000_000,
            ..offloader_config()
        },
        fixture.store(),
    );
    assert!(processor
        .process(&mut messages, &fixture.context())
        .await
        .unwrap()
        .is_some());

    // Thresholds too high: nothing happens.
    let mut messages = history();
    let processor = MessageOffloader::new(
        MessageOffloaderConfig {
            messages_threshold: Some(100),
            tokens_threshold: 1_000_000,
            ..offloader_config()
        },
        fixture.store(),
    );
    assert!(processor
        .process(&mut messages, &fixture.context())
        .await
        .unwrap()
        .is_none());

    // Token trigger.
    let mut messages = history();
    let processor = MessageOffloader::new(
        MessageOffloaderConfig {
            messages_threshold: None,
            tokens_threshold: 10,
            ..offloader_config()
        },
        fixture.store(),
    );
    assert!(processor
        .process(&mut messages, &fixture.context())
        .await
        .unwrap()
        .is_some());

    // Thresholds exceeded but every message is small: no candidate, no event.
    let mut messages = short_history();
    let processor = MessageOffloader::new(
        MessageOffloaderConfig {
            messages_threshold: Some(1),
            tokens_threshold: 1,
            ..offloader_config()
        },
        fixture.store(),
    );
    assert!(processor
        .process(&mut messages, &fixture.context())
        .await
        .unwrap()
        .is_none());
}

#[tokio::test]
async fn message_offloader_reports_every_index_and_the_last_handle() {
    let fixture = Fixture::new();
    let mut messages = Vec::new();
    messages.extend(tool_exchange("call_1", "grep", &"a".repeat(1500)));
    messages.extend(tool_exchange("call_2", "grep", &"b".repeat(1500)));
    messages.push(BaseMessage::assistant("done"));

    let processor = MessageOffloader::new(offloader_config(), fixture.store());
    let event = processor
        .process(&mut messages, &fixture.context())
        .await
        .unwrap()
        .unwrap();

    assert_eq!(event.messages_changed, vec![2, 5]);
    // Newest first, so the last handle belongs to the oldest rewritten message.
    let oldest = OffloadMarker::parse(&messages[2].text()).unwrap();
    assert_eq!(Some(oldest.handle), event.offload_handle);
    assert_eq!(fixture.store.len(), 2);
}

#[tokio::test]
async fn message_offloader_cancellation_is_honoured() {
    let fixture = Fixture::new();
    fixture.cancellation.cancel();
    let mut messages = tool_exchange("call_1", "grep", &"x".repeat(1500));
    messages.push(BaseMessage::assistant("done"));

    let processor = MessageOffloader::new(offloader_config(), fixture.store());
    let error = processor
        .process(&mut messages, &fixture.context())
        .await
        .unwrap_err();
    assert!(matches!(error, Error::Cancelled));
    assert!(fixture.store.is_empty());
}

#[test]
fn message_offloader_config_validates_and_round_trips() {
    let config = MessageOffloaderConfig::default();
    assert_eq!(config.trim_size, 100);
    assert_eq!(config.large_message_threshold, 1_000);
    assert_eq!(config.offload_message_types, vec![MessageOffloadRole::Tool]);
    assert_eq!(config.protected_tool_names, vec!["read_file".to_owned()]);
    assert!(config.validate().is_ok());

    let error = MessageOffloaderConfig {
        trim_size: 1_000,
        ..MessageOffloaderConfig::default()
    }
    .validate()
    .unwrap_err();
    assert!(matches!(error, Error::Config(_)));

    let error = MessageOffloaderConfig {
        messages_to_keep: Some(5),
        messages_threshold: Some(5),
        ..MessageOffloaderConfig::default()
    }
    .validate()
    .unwrap_err();
    assert!(matches!(error, Error::Config(_)));

    let json = serde_json::to_string(&MessageOffloaderConfig {
        offload_message_types: vec![MessageOffloadRole::Assistant, MessageOffloadRole::User],
        ..MessageOffloaderConfig::default()
    })
    .unwrap();
    assert!(json.contains("\"assistant\""));
    let parsed: MessageOffloaderConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.offload_message_types.len(), 2);
    let defaulted: MessageOffloaderConfig = serde_json::from_str("{}").unwrap();
    assert_eq!(defaulted, MessageOffloaderConfig::default());
}

#[test]
fn wildcard_matcher_handles_python_patterns() {
    assert!(wildcard_match("*.md", "docs/guide.md"));
    assert!(wildcard_match("a?c", "abc"));
    assert!(!wildcard_match("a?c", "ac"));
    assert!(wildcard_match("*", ""));
}

// ---------------------------------------------------------------------------
// ToolResultBudgetProcessor
// ---------------------------------------------------------------------------

fn budget_config() -> ToolResultBudgetProcessorConfig {
    ToolResultBudgetProcessorConfig {
        tokens_threshold: 5_000,
        large_message_threshold: 500,
        trim_size: 50,
        tool_name_allowlist: None,
    }
}

#[tokio::test]
async fn tool_result_budget_offloads_the_largest_result_until_the_round_fits() {
    let fixture = Fixture::new();
    let mut messages = two_result_round(2_000, 6_000);
    let processor = ToolResultBudgetProcessor::new(budget_config(), fixture.store());
    let event = processor
        .process(&mut messages, &fixture.context())
        .await
        .unwrap()
        .unwrap();

    assert_eq!(event.kind, ContextEventKind::Offloaded);
    assert_eq!(event.messages_changed, vec![3]);
    assert!(event.tokens_after < event.tokens_before);

    // The placeholder keeps the tool pairing and reports the original size.
    let text = messages[3].text().to_string();
    let marker = OffloadMarker::parse(&text).unwrap();
    assert_eq!(
        text,
        format!(
            "<{PERSISTED_OUTPUT_TAG}>\nOutput too large (6000 chars).\n{}\n\
             Preview (first 50 chars):\n{}\n</{PERSISTED_OUTPUT_TAG}>",
            marker.to_marker_string(),
            "b".repeat(50)
        )
    );
    assert_eq!(messages[3].tool_call_id.as_deref(), Some("c2"));
    assert_eq!(Some(marker.handle.clone()), event.offload_handle);

    let stored = fixture.stored(&marker.handle).await;
    assert_eq!(stored[0].text(), "b".repeat(6_000));

    // The smaller result still fits and is untouched.
    assert_eq!(messages[2].text(), "a".repeat(2_000));
}

#[tokio::test]
async fn tool_result_budget_keeps_offloading_the_largest_until_no_candidate_remains() {
    let fixture = Fixture::new();
    let mut messages = vec![
        BaseMessage::user("go"),
        BaseMessage::assistant("").with_tool_calls(vec![
            ToolCall::new("shell").with_id("c1"),
            ToolCall::new("shell").with_id("c2"),
            ToolCall::new("shell").with_id("c3"),
        ]),
        BaseMessage::tool("c1", "a".repeat(2_000)),
        BaseMessage::tool("c2", "b".repeat(4_000)),
        BaseMessage::tool("c3", "c".repeat(6_000)),
    ];
    let processor = ToolResultBudgetProcessor::new(budget_config(), fixture.store());
    let event = processor
        .process(&mut messages, &fixture.context())
        .await
        .unwrap()
        .unwrap();

    // 6_000 then 4_000 leaves 2_000 inside the 5_000 budget.
    assert_eq!(event.messages_changed, vec![3, 4]);
    assert_eq!(messages[2].text(), "a".repeat(2_000));
    assert!(messages[3]
        .text()
        .starts_with(&format!("<{PERSISTED_OUTPUT_TAG}>")));
    assert!(messages[4]
        .text()
        .starts_with(&format!("<{PERSISTED_OUTPUT_TAG}>")));
}

#[tokio::test]
async fn tool_result_budget_respects_the_allowlist_and_leaves_other_rounds_alone() {
    let fixture = Fixture::new();
    let mut messages = vec![
        // Round one: one allowlisted tool result next to an offloadable one.
        BaseMessage::user("first"),
        BaseMessage::assistant("").with_tool_calls(vec![
            ToolCall::new("read_file").with_id("c1"),
            ToolCall::new("shell").with_id("c2"),
        ]),
        BaseMessage::tool("c1", "r".repeat(6_000)),
        BaseMessage::tool("c2", "s".repeat(6_000)),
        // Round two: a huge user turn with no tool results at all.
        BaseMessage::user("u".repeat(9_000)),
        BaseMessage::assistant("done"),
    ];
    let processor = ToolResultBudgetProcessor::new(
        ToolResultBudgetProcessorConfig {
            tool_name_allowlist: Some(vec!["read_file".to_owned()]),
            ..budget_config()
        },
        fixture.store(),
    );
    let event = processor
        .process(&mut messages, &fixture.context())
        .await
        .unwrap()
        .unwrap();

    // Only the unprotected tool result of the over-budget round changes.
    assert_eq!(event.messages_changed, vec![3]);
    assert_eq!(messages[2].text(), "r".repeat(6_000));
    assert_eq!(messages[4].text(), "u".repeat(9_000));
    assert!(messages[3].text().contains("[[OFFLOAD:"));
}

#[tokio::test]
async fn tool_result_budget_is_a_noop_when_every_result_is_small() {
    let fixture = Fixture::new();
    let mut messages = two_result_round(100, 200);
    let processor = ToolResultBudgetProcessor::new(
        ToolResultBudgetProcessorConfig {
            tokens_threshold: 1,
            ..budget_config()
        },
        fixture.store(),
    );
    assert!(processor
        .process(&mut messages, &fixture.context())
        .await
        .unwrap()
        .is_none());
    assert!(fixture.store.is_empty());
}

#[test]
fn tool_result_budget_config_validates() {
    assert!(budget_config().validate().is_ok());
    let error = ToolResultBudgetProcessorConfig {
        trim_size: 500,
        large_message_threshold: 500,
        ..budget_config()
    }
    .validate()
    .unwrap_err();
    assert!(matches!(error, Error::Config(_)));
}

// ---------------------------------------------------------------------------
// ToolResultWindowProcessor
// ---------------------------------------------------------------------------

fn window_config(tool_names: &[&str], keep_last_k: usize) -> ToolResultWindowProcessorConfig {
    ToolResultWindowProcessorConfig {
        tool_names: tool_names.iter().map(|name| (*name).to_owned()).collect(),
        keep_last_k,
        trim_size: 50,
        min_offload_chars: 0,
        small_result_trim_size: 100,
    }
}

#[tokio::test]
async fn tool_result_window_matches_suffixes_and_keeps_the_global_newest_k() {
    let fixture = Fixture::new();
    let mut messages = Vec::new();
    messages.extend(tool_exchange("c1", "fs_read_file", &"a".repeat(2_000)));
    messages.extend(tool_exchange("c2", "grep", &"g".repeat(2_000)));
    messages.extend(tool_exchange("c3", "fs_read_file", &"b".repeat(2_000)));
    messages.extend(tool_exchange("c4", "fs_read_file", &"c".repeat(2_000)));
    messages.extend(tool_exchange("c5", "fs_read_file", &"d".repeat(2_000)));
    // Tool results sit at indices 2, 5, 8, 11, 14.
    let processor =
        ToolResultWindowProcessor::new(window_config(&["read_file"], 2), fixture.store());
    let event = processor
        .process(&mut messages, &fixture.context())
        .await
        .unwrap()
        .unwrap();

    assert_eq!(event.kind, ContextEventKind::Offloaded);
    assert_eq!(event.messages_changed, vec![2, 8]);
    // The newest two matching results and the non-matching tool stay whole.
    assert_eq!(messages[11].text(), "c".repeat(2_000));
    assert_eq!(messages[14].text(), "d".repeat(2_000));
    assert_eq!(messages[5].text(), "g".repeat(2_000));

    for index in [2, 8] {
        let text = messages[index].text().to_string();
        assert!(text.starts_with(&format!("<{PERSISTED_OUTPUT_TAG}>")));
        assert_eq!(
            messages[index].metadata.get(WINDOW_MANAGED_METADATA_KEY),
            Some(&serde_json::Value::Bool(true))
        );
        assert_eq!(
            messages[index].tool_call_id.as_deref(),
            Some(if index == 2 { "c1" } else { "c3" })
        );
    }
    assert_eq!(fixture.store.len(), 2);

    // Managed messages are skipped on the next run.
    assert!(processor
        .process(&mut messages, &fixture.context())
        .await
        .unwrap()
        .is_none());
}

#[tokio::test]
async fn tool_result_window_compacts_small_results_without_persisting_them() {
    let fixture = Fixture::new();
    let mut messages = Vec::new();
    for index in 0..3 {
        messages.extend(tool_exchange(
            &format!("c{index}"),
            "read_file",
            &"r".repeat(500),
        ));
    }
    let processor = ToolResultWindowProcessor::new(
        ToolResultWindowProcessorConfig {
            min_offload_chars: 1_000,
            ..window_config(&["read_file"], 1)
        },
        fixture.store(),
    );
    let event = processor
        .process(&mut messages, &fixture.context())
        .await
        .unwrap()
        .unwrap();

    assert_eq!(event.messages_changed, vec![2, 5]);
    assert_eq!(event.offload_handle, None);
    assert!(fixture.store.is_empty());

    let text = messages[2].text().to_string();
    assert!(text.starts_with(&format!("<{WINDOWED_OUTPUT_TAG}>")));
    assert!(text.contains("Older tool result compacted in context (500 chars)"));
    assert!(text.ends_with(&format!("</{WINDOWED_OUTPUT_TAG}>")));
    assert_eq!(
        messages[2].metadata.get(WINDOW_MANAGED_METADATA_KEY),
        Some(&serde_json::Value::Bool(true))
    );
    // The newest matching result is untouched.
    assert_eq!(messages[8].text(), "r".repeat(500));

    assert!(processor
        .process(&mut messages, &fixture.context())
        .await
        .unwrap()
        .is_none());
}

#[tokio::test]
async fn tool_result_window_without_tool_names_is_a_noop() {
    let fixture = Fixture::new();
    let mut messages = Vec::new();
    for index in 0..4 {
        messages.extend(tool_exchange(
            &format!("c{index}"),
            "read_file",
            &"r".repeat(2_000),
        ));
    }
    let processor = ToolResultWindowProcessor::new(window_config(&[], 1), fixture.store());
    assert!(processor
        .process(&mut messages, &fixture.context())
        .await
        .unwrap()
        .is_none());
    assert!(fixture.store.is_empty());
}

#[test]
fn tool_result_window_config_validates() {
    assert!(window_config(&["read_file"], 3).validate().is_ok());
    let error = window_config(&["read_file"], 0).validate().unwrap_err();
    assert!(matches!(error, Error::Config(_)));
}

// ---------------------------------------------------------------------------
// MessageSummaryOffloader
// ---------------------------------------------------------------------------

fn summary_config() -> MessageSummaryOffloaderConfig {
    MessageSummaryOffloaderConfig {
        large_message_threshold: 100,
        offload_message_types: vec![MessageOffloadRole::Tool],
        protected_tool_names: vec!["read_file".to_owned()],
        summary_max_tokens: 900,
        content_max_chars_for_compression: 300,
    }
}

fn summary_history() -> Vec<BaseMessage> {
    let mut messages = tool_exchange("call_1", "grep", &"z".repeat(3_000));
    messages.push(BaseMessage::assistant("done"));
    messages
}

#[tokio::test]
async fn summary_offloader_falls_back_to_a_head_and_tail_summary_without_a_model() {
    let fixture = Fixture::new();
    let mut messages = summary_history();
    let processor = MessageSummaryOffloader::new(summary_config(), None, fixture.store());
    let event = processor
        .process(&mut messages, &fixture.context())
        .await
        .unwrap()
        .unwrap();

    assert_eq!(event.kind, ContextEventKind::Offloaded);
    assert_eq!(event.messages_changed, vec![2]);
    assert!(event.tokens_after < event.tokens_before);

    let text = messages[2].text().to_string();
    // Head (~2/3 of the fallback budget), then the truncation marker.
    assert!(text.starts_with(&"z".repeat(200)));
    assert!(text.contains("...[TRUNCATED]..."));
    assert!(text.contains(OFFLOADED_INFO_TAG));
    assert!(text.contains("original_chars: 3000"));
    let marker = OffloadMarker::parse(&text).unwrap();
    assert_eq!(Some(marker.handle.clone()), event.offload_handle);
    // `[offloaded_info]` sits on its own line.
    assert!(text.contains(&format!("\n\n{OFFLOADED_INFO_TAG}\n")));

    let stored = fixture.stored(&marker.handle).await;
    assert_eq!(stored[0].text(), "z".repeat(3_000));
    assert_eq!(messages[2].tool_call_id.as_deref(), Some("call_1"));

    // The replacement carries the marker, so a second pass skips it.
    assert!(processor
        .process(&mut messages, &fixture.context())
        .await
        .unwrap()
        .is_none());
}

#[tokio::test]
async fn summary_offloader_uses_the_configured_model() {
    let fixture = Fixture::new();
    let scripted = ScriptedModel::builder()
        .respond_text("  a compact summary  ")
        .build();
    let model: Arc<dyn Model> = Arc::new(scripted.clone());
    let mut messages = summary_history();
    let processor =
        MessageSummaryOffloader::new(summary_config(), Some(Arc::clone(&model)), fixture.store());
    let event = processor
        .process(&mut messages, &fixture.context())
        .await
        .unwrap()
        .unwrap();

    assert_eq!(scripted.call_count(), 1);
    let call = scripted.last_call().unwrap();
    assert_eq!(call.messages[0].text(), SUMMARY_COMPRESSION_SYSTEM_PROMPT);
    assert!(call.messages[1].text().starts_with("zzz"));

    let text = messages[2].text().to_string();
    assert!(text.starts_with("a compact summary\n\n"));
    assert!(text.contains(OFFLOADED_INFO_TAG));
    OffloadMarker::parse(&text).unwrap();
    assert_eq!(event.messages_changed, vec![2]);
}

#[tokio::test]
async fn summary_offloader_uses_the_context_model_when_none_is_configured() {
    let scripted = ScriptedModel::builder()
        .respond_text("context summary")
        .build();
    let model: Arc<dyn Model> = Arc::new(scripted.clone());
    let fixture = Fixture::with_context_model(model);
    let mut messages = summary_history();
    let processor = MessageSummaryOffloader::new(summary_config(), None, fixture.store());
    processor
        .process(&mut messages, &fixture.context())
        .await
        .unwrap()
        .unwrap();

    assert_eq!(scripted.call_count(), 1);
    assert!(messages[2].text().starts_with("context summary\n\n"));
}

#[tokio::test]
async fn summary_offloader_skips_protected_results_and_tool_calling_assistants() {
    let fixture = Fixture::new();
    let mut messages = vec![
        BaseMessage::user("q"),
        BaseMessage::assistant("a".repeat(3_000))
            .with_tool_calls(vec![ToolCall::new("read_file").with_id("c1")]),
        BaseMessage::tool("c1", "t".repeat(3_000)),
        BaseMessage::assistant("done"),
    ];
    let processor = MessageSummaryOffloader::new(
        MessageSummaryOffloaderConfig {
            offload_message_types: vec![MessageOffloadRole::Assistant, MessageOffloadRole::Tool],
            ..summary_config()
        },
        None,
        fixture.store(),
    );
    assert!(processor
        .process(&mut messages, &fixture.context())
        .await
        .unwrap()
        .is_none());
    assert_eq!(messages[1].text(), "a".repeat(3_000));
    assert_eq!(messages[2].text(), "t".repeat(3_000));
    assert!(fixture.store.is_empty());
}

#[tokio::test]
async fn summary_offloader_cancellation_is_honoured() {
    let fixture = Fixture::new();
    fixture.cancellation.cancel();
    let mut messages = summary_history();
    let processor = MessageSummaryOffloader::new(summary_config(), None, fixture.store());
    let error = processor
        .process(&mut messages, &fixture.context())
        .await
        .unwrap_err();
    assert!(matches!(error, Error::Cancelled));
}

#[test]
fn summary_offloader_config_round_trips() {
    let config = MessageSummaryOffloaderConfig::default();
    assert_eq!(config.large_message_threshold, 1_000);
    assert_eq!(config.summary_max_tokens, 900);
    assert_eq!(config.content_max_chars_for_compression, 200_000);

    let json = serde_json::to_string(&MessageSummaryOffloaderConfig {
        large_message_threshold: 42,
        offload_message_types: vec![MessageOffloadRole::User, MessageOffloadRole::Tool],
        protected_tool_names: vec!["view".to_owned()],
        summary_max_tokens: 128,
        content_max_chars_for_compression: 5_000,
    })
    .unwrap();
    let parsed: MessageSummaryOffloaderConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.offload_message_types.len(), 2);
    assert_eq!(parsed.summary_max_tokens, 128);

    let partial: MessageSummaryOffloaderConfig =
        serde_json::from_str(r#"{"summary_max_tokens": 7}"#).unwrap();
    assert_eq!(partial.summary_max_tokens, 7);
    assert_eq!(partial.large_message_threshold, 1_000);
}

// ---------------------------------------------------------------------------
// Pipeline wiring
// ---------------------------------------------------------------------------

#[test]
fn processor_config_tags_round_trip() {
    let json = r#"[
        {"type": "message_offloader", "trim_size": 10},
        {"type": "tool_result_budget", "tokens_threshold": 100},
        {"type": "tool_result_window", "tool_names": ["read_file"], "keep_last_k": 2},
        {"type": "message_summary_offloader", "large_message_threshold": 5}
    ]"#;
    let configs: Vec<ProcessorConfig> = serde_json::from_str(json).unwrap();
    assert!(matches!(configs[0], ProcessorConfig::MessageOffloader(_)));
    assert!(matches!(configs[1], ProcessorConfig::ToolResultBudget(_)));
    assert!(matches!(configs[2], ProcessorConfig::ToolResultWindow(_)));
    assert!(matches!(
        configs[3],
        ProcessorConfig::MessageSummaryOffloader(_)
    ));
    assert_eq!(
        serde_json::from_str::<Vec<ProcessorConfig>>(&serde_json::to_string(&configs).unwrap())
            .unwrap(),
        configs
    );
}

#[test]
fn from_configs_builds_the_offloaders_in_order_and_validates_them() {
    let store: Arc<dyn OffloadStore> = Arc::new(InMemoryOffloadStore::new());
    let pipeline = ContextPipeline::from_configs(
        &[
            ProcessorConfig::MessageOffloader(MessageOffloaderConfig::default()),
            ProcessorConfig::ToolResultBudget(ToolResultBudgetProcessorConfig::default()),
            ProcessorConfig::ToolResultWindow(ToolResultWindowProcessorConfig::default()),
            ProcessorConfig::MessageSummaryOffloader(MessageSummaryOffloaderConfig::default()),
        ],
        None,
        Arc::clone(&store),
    )
    .unwrap();
    let names: Vec<&'static str> = pipeline
        .processors()
        .iter()
        .map(|processor| processor.name())
        .collect();
    assert_eq!(
        names,
        vec![
            "message_offloader",
            "tool_result_budget",
            "tool_result_window",
            "message_summary_offloader"
        ]
    );

    let error = ContextPipeline::from_configs(
        &[ProcessorConfig::MessageOffloader(MessageOffloaderConfig {
            trim_size: 1_000,
            ..MessageOffloaderConfig::default()
        })],
        None,
        Arc::clone(&store),
    )
    .unwrap_err();
    assert!(matches!(error, Error::Config(_)));

    let error = ContextPipeline::from_configs(
        &[ProcessorConfig::ToolResultWindow(
            ToolResultWindowProcessorConfig {
                keep_last_k: 0,
                ..ToolResultWindowProcessorConfig::default()
            },
        )],
        None,
        store,
    )
    .unwrap_err();
    assert!(matches!(error, Error::Config(_)));
}

#[tokio::test]
async fn pipeline_runs_the_offloaders_end_to_end() {
    let fixture = Fixture::new();
    let pipeline = ContextPipeline::from_configs(
        &[
            ProcessorConfig::ToolResultWindow(window_config(&["read"], 1)),
            ProcessorConfig::MessageSummaryOffloader(summary_config()),
        ],
        None,
        fixture.store(),
    )
    .unwrap();
    let mut messages = Vec::new();
    for index in 0..3 {
        messages.extend(tool_exchange(
            &format!("c{index}"),
            "fs_read",
            &"r".repeat(3_000),
        ));
    }
    let cx = fixture.context();
    let events = pipeline.run(&mut messages, &cx).await.unwrap();

    // The window offloads its older results; the summary offloader then skips
    // the placeholders (they carry an offload marker) and summarises the one
    // remaining full-size result.
    assert_eq!(events.len(), 2);
    assert_eq!(events[0].processor, "tool_result_window");
    assert_eq!(events[0].kind, ContextEventKind::Offloaded);
    assert_eq!(events[0].messages_changed, vec![2, 5]);
    assert_eq!(events[1].processor, "message_summary_offloader");
    assert_eq!(events[1].messages_changed, vec![8]);
    assert!(messages[8].text().contains(OFFLOADED_INFO_TAG));
    assert!(messages[2]
        .text()
        .starts_with(&format!("<{PERSISTED_OUTPUT_TAG}>")));
}
