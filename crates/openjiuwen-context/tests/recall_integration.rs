#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//! Compression recall wired into the compressor family.
//!
//! Every compressor that replaces history now archives what it dropped in the
//! recall store and marks its memory block with
//! `[[COMPRESSION_RECALL: id=...]]`, so [`recall_compressed_context`] can later
//! return the exact text the block summarised. This suite pins that contract per
//! compressor: the archive contents, the marker's position at the end of the
//! block, the merge stages keeping every marker exactly once, the hard-truncation
//! archive of the removed prefix, both full-compaction paths, the rollback of an
//! archive whose replacement a benefit gate rejects, and the two ways nothing
//! happens (recall disabled, or no store on the processor context).
//!
//! Everything is offline and deterministic: token counts come from the
//! one-character-per-token counter, no model is configured, and archives are read
//! back through [`recall_compressed_context`] with default options.
//!
//! [`recall_compressed_context`]: openjiuwen_context::recall_compressed_context

use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_context::processor::{
    is_memory_block, ContextProcessor, CurrentRoundCompressor, CurrentRoundCompressorConfig,
    DialogueCompressor, DialogueCompressorConfig, FullCompactProcessor, FullCompactProcessorConfig,
    MemoryBlockKind, ProcessorContext, RoundLevelCompressor, RoundLevelCompressorConfig,
    FULL_COMPACT_SUMMARY_INTRO, TRUNCATED_FULL_CONTEXT_SCOPE,
};
use openjiuwen_context::{
    count_messages, parse_recall_marker, recall_compressed_context, recall_marker,
    stamp_message_id, CompressionRecallConfig, ContextConfig, InMemoryOffloadStore,
    InMemoryRecallArchiveStore, InMemorySessionMemoryStore, OffloadScope, RecallArchive,
    RecallArchiveStore, RecallOptions, SessionMemoryNotes, SessionMemoryStore, TokenCounterConfig,
    TokenCounterKind, CONTEXT_MESSAGE_ID_KEY, CURRENT_ROUND_MEMORY_BLOCK, DIALOGUE_MEMORY_BLOCK,
    ROUND_LEVEL_MEMORY_BLOCK,
};
use openjiuwen_foundation::prompt::{ApproxTokenCounter, BaseMessage, ToolCall};
use tokio_util::sync::CancellationToken;

/// The one-character-per-token counter every test uses.
fn counter() -> ApproxTokenCounter {
    ApproxTokenCounter::with_chars_per_token(1)
}

/// A body long enough that replacing it with a memory block is a real saving.
fn long(text: &str) -> String {
    format!("{text}: {}", "detail ".repeat(1_200))
}

/// One round that exercises a tool exchange, so it is longer than two messages.
///
/// The closing assistant repeats the tool-output phrase: recall ranks turns by
/// their query and answer, so the phrase has to be reachable from the turn text
/// as well as from the archived chunk.
fn tool_round(index: usize) -> Vec<BaseMessage> {
    vec![
        BaseMessage::user(format!("u{index}")),
        BaseMessage::assistant("").with_tool_calls(vec![
            ToolCall::new("read_file").with_id(format!("call_{index}"))
        ]),
        BaseMessage::tool(
            format!("call_{index}"),
            long(&format!("tool output {index}")),
        ),
        BaseMessage::assistant(format!("tool output {index} confirmed: {}", long("done"))),
    ]
}

/// A persona plus `rounds` tool rounds.
fn history(rounds: usize) -> Vec<BaseMessage> {
    let mut messages = vec![BaseMessage::system("persona")];
    for index in 1..=rounds {
        messages.extend(tool_round(index));
    }
    messages
}

/// A conversation full-compaction can replace wholesale.
fn conversation() -> Vec<BaseMessage> {
    vec![
        BaseMessage::system("persona"),
        BaseMessage::user(long("u1 borrow")),
        BaseMessage::assistant(long("a1")),
        BaseMessage::user(long("u2")),
        BaseMessage::assistant(long("a2")),
    ]
}

/// Shared processor inputs, with a live in-memory recall store.
struct Fixture {
    config: ContextConfig,
    counter: ApproxTokenCounter,
    offload: InMemoryOffloadStore,
    recall: InMemoryRecallArchiveStore,
    scope: OffloadScope,
    cancellation: CancellationToken,
}

impl Fixture {
    /// A fixture with compression recall enabled.
    fn new() -> Self {
        Self::with_recall(true)
    }

    /// A fixture whose configuration flag is `enabled`.
    fn with_recall(enabled: bool) -> Self {
        Self {
            config: ContextConfig {
                token_counter: TokenCounterConfig {
                    kind: TokenCounterKind::Characters,
                    chars_per_token: 1,
                },
                compression_recall: CompressionRecallConfig {
                    enabled,
                    ..CompressionRecallConfig::default()
                },
                ..ContextConfig::default()
            },
            counter: counter(),
            offload: InMemoryOffloadStore::new(),
            recall: InMemoryRecallArchiveStore::new(),
            scope: OffloadScope::new("session-1", "main"),
            cancellation: CancellationToken::new(),
        }
    }

    /// A processor context offering `recall` to the processors.
    fn context_with<'a>(
        &'a self,
        recall: Option<&'a dyn RecallArchiveStore>,
        manual: bool,
    ) -> ProcessorContext<'a> {
        ProcessorContext {
            config: &self.config,
            counter: &self.counter,
            model: None,
            offload: &self.offload,
            scope: &self.scope,
            budget: 1_000,
            trigger_tokens: 1_000,
            manual,
            cancellation: &self.cancellation,
            recall,
            session_memory: None,
        }
    }

    /// A processor context wired to the fixture's in-memory recall store.
    fn context(&self, manual: bool) -> ProcessorContext<'_> {
        self.context_with(Some(&self.recall), manual)
    }

    /// A processor context with recall enabled in the config but no store.
    fn context_without_store(&self, manual: bool) -> ProcessorContext<'_> {
        self.context_with(None, manual)
    }

    /// The store as the recall API takes it.
    fn store(&self) -> &dyn RecallArchiveStore {
        &self.recall
    }

    /// The archive ids stored for the fixture scope, sorted.
    async fn archives(&self) -> Vec<String> {
        let mut ids = self.recall.list(&self.scope).await.unwrap();
        ids.sort();
        ids
    }

    /// A recall query against one archive.
    async fn recall_chunks(&self, memory_id: &str, query: &str) -> Vec<String> {
        recall_compressed_context(
            self.store(),
            &self.scope,
            Some(memory_id),
            query,
            &RecallOptions::default(),
        )
        .await
        .unwrap()
        .chunks
        .into_iter()
        .map(|chunk| chunk.text)
        .collect()
    }
}

/// A recall store whose writes always fail.
struct FailingRecallStore;

#[async_trait]
impl RecallArchiveStore for FailingRecallStore {
    async fn put(&self, _scope: &OffloadScope, _archive: RecallArchive) -> Result<()> {
        Err(Error::Validation("the recall store is down".to_owned()))
    }

    async fn get(&self, _scope: &OffloadScope, _memory_id: &str) -> Result<Option<RecallArchive>> {
        Ok(None)
    }

    async fn list(&self, _scope: &OffloadScope) -> Result<Vec<String>> {
        Ok(Vec::new())
    }

    async fn remove(&self, _scope: &OffloadScope, _memory_id: &str) -> Result<()> {
        Ok(())
    }
}

/// The archive id a block ends with, when its last line is a recall marker.
fn trailing_marker(message: &BaseMessage) -> Option<String> {
    let text = message.text();
    let last = text.lines().last()?;
    parse_recall_marker(last)
}

/// The `scope:` line of a rendered block.
fn scope_of(message: &BaseMessage) -> String {
    message
        .text()
        .lines()
        .find(|line| line.starts_with("scope: "))
        .unwrap_or_default()
        .trim_start_matches("scope: ")
        .to_owned()
}

/// The archive ids every block of `messages` ends with, in order.
fn markers(messages: &[BaseMessage], block_marker: &str) -> Vec<String> {
    messages
        .iter()
        .filter(|message| is_memory_block(message, block_marker))
        .map(|message| trailing_marker(message).expect("every block carries a recall marker"))
        .collect()
}

/// A configuration that compresses aggressively, leaving room for the marker.
fn dialogue_config() -> DialogueCompressorConfig {
    DialogueCompressorConfig {
        tokens_threshold: 100,
        compression_target_tokens: 200,
        ..DialogueCompressorConfig::default()
    }
}

/// A current-round configuration whose stage-1 span is worth compressing.
fn current_round_config() -> CurrentRoundCompressorConfig {
    CurrentRoundCompressorConfig {
        messages_to_keep: 1,
        min_selected_tokens_for_compression: 1,
        compression_target_tokens: 200,
        ..CurrentRoundCompressorConfig::default()
    }
}

/// A round-level configuration with small pass targets.
fn round_level_config(target: usize, keep_recent: usize) -> RoundLevelCompressorConfig {
    RoundLevelCompressorConfig {
        target_total_tokens: target,
        keep_recent_messages: keep_recent,
        first_pass_target_tokens: 200,
        second_pass_target_tokens: 150,
        third_pass_target_tokens: 150,
        ..RoundLevelCompressorConfig::default()
    }
}

/// A compaction configuration that triggers on every long history here.
fn compact_config() -> FullCompactProcessorConfig {
    FullCompactProcessorConfig {
        trigger_total_tokens: 2_000,
        compression_call_max_tokens: 100_000,
        messages_to_keep: 0,
        ..FullCompactProcessorConfig::default()
    }
}

/// A pre-rendered current-round block, optionally carrying a recall marker.
fn current_round_block(memory_id: Option<&str>, body: &str) -> BaseMessage {
    let mut text = MemoryBlockKind::CurrentRound.render(body);
    if let Some(memory_id) = memory_id {
        text.push('\n');
        text.push_str(&recall_marker(memory_id));
    }
    BaseMessage::user(text)
}

#[tokio::test]
async fn dialogue_compression_archives_the_replaced_exchange() {
    let fixture = Fixture::new();
    let compressor = DialogueCompressor::new(dialogue_config(), None);
    let mut messages = history(3);
    let before = count_messages(&fixture.counter, &messages);

    compressor
        .process(&mut messages, &fixture.context(true))
        .await
        .unwrap()
        .expect("the pass must compress");

    let ids = markers(&messages, DIALOGUE_MEMORY_BLOCK);
    assert_eq!(ids.len(), 2);
    assert_ne!(ids[0], ids[1]);
    assert_eq!(fixture.archives().await, {
        let mut expected = ids.clone();
        expected.sort();
        expected
    });

    // The marker is the block's last line and parses back to its archive id.
    for (id, block) in ids.iter().zip(
        messages
            .iter()
            .filter(|message| is_memory_block(message, DIALOGUE_MEMORY_BLOCK)),
    ) {
        assert!(block.text().ends_with(&recall_marker(id)));
        assert_eq!(parse_recall_marker(&block.text()), Some(id.clone()));
    }

    // The archive of the first round keeps the replaced exchange, not the anchor
    // user message and not the system message.
    let archive = fixture
        .recall
        .get(&fixture.scope, &ids[0])
        .await
        .unwrap()
        .expect("the archive exists");
    assert_eq!(archive.memory_id, ids[0]);
    assert_eq!(archive.session_id, "session-1");
    assert_eq!(archive.context_id, "main");
    assert!(archive.created_at.starts_with("unix:"));
    let archived: String = archive
        .chunks
        .iter()
        .map(|chunk| chunk.text.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(archived.contains("tool output 1"));
    assert!(!archived.contains("persona"));
    assert!(!archived.contains("## User\nu1"));

    // The dropped tool output is retrievable through the marker's archive.
    let chunks = fixture.recall_chunks(&ids[0], "tool output 1").await;
    assert!(
        chunks.iter().any(|chunk| chunk.contains("tool output 1")),
        "the recalled chunk must carry the removed content"
    );
    // The compression itself still paid off.
    assert!(count_messages(&fixture.counter, &messages) < before);
}

#[tokio::test]
async fn dialogue_compression_without_recall_writes_no_marker() {
    let fixture = Fixture::with_recall(false);
    let compressor = DialogueCompressor::new(dialogue_config(), None);
    let mut messages = history(3);

    compressor
        .process(&mut messages, &fixture.context(true))
        .await
        .unwrap()
        .expect("the pass must compress");

    let blocks: Vec<&BaseMessage> = messages
        .iter()
        .filter(|message| is_memory_block(message, DIALOGUE_MEMORY_BLOCK))
        .collect();
    assert_eq!(blocks.len(), 2);
    assert!(blocks
        .iter()
        .all(|block| parse_recall_marker(&block.text()).is_none()));
    assert_eq!(fixture.archives().await, Vec::<String>::new());
}

#[tokio::test]
async fn no_store_means_no_marker_even_when_recall_is_enabled() {
    let fixture = Fixture::new();
    let compressor = DialogueCompressor::new(dialogue_config(), None);
    let mut messages = history(3);

    compressor
        .process(&mut messages, &fixture.context_without_store(true))
        .await
        .unwrap()
        .expect("the pass must compress");

    let blocks: Vec<&BaseMessage> = messages
        .iter()
        .filter(|message| is_memory_block(message, DIALOGUE_MEMORY_BLOCK))
        .collect();
    assert_eq!(blocks.len(), 2);
    assert!(blocks
        .iter()
        .all(|block| parse_recall_marker(&block.text()).is_none()));
    assert_eq!(fixture.archives().await, Vec::<String>::new());
}

#[tokio::test]
async fn a_failing_recall_store_never_blocks_compression() {
    let fixture = Fixture::new();
    let failing = FailingRecallStore;
    let compressor = DialogueCompressor::new(dialogue_config(), None);
    let mut messages = history(3);
    let before = count_messages(&fixture.counter, &messages);

    compressor
        .process(&mut messages, &fixture.context_with(Some(&failing), true))
        .await
        .unwrap()
        .expect("compression must proceed without the archive");

    let blocks: Vec<&BaseMessage> = messages
        .iter()
        .filter(|message| is_memory_block(message, DIALOGUE_MEMORY_BLOCK))
        .collect();
    assert_eq!(blocks.len(), 2);
    assert!(blocks
        .iter()
        .all(|block| parse_recall_marker(&block.text()).is_none()));
    assert!(count_messages(&fixture.counter, &messages) < before);
    assert_eq!(fixture.archives().await, Vec::<String>::new());
}

#[tokio::test]
async fn a_rejected_replacement_rolls_back_its_archive() {
    let fixture = Fixture::new();
    let compressor = DialogueCompressor::new(dialogue_config(), None);
    // One short tool round that is a valid target, but the block envelope — with
    // the marker — costs more than the messages it would replace.
    let mut messages = vec![
        BaseMessage::user("u1"),
        BaseMessage::assistant("").with_tool_calls(vec![ToolCall::new("read_file").with_id("c1")]),
        BaseMessage::tool("c1", "short"),
        BaseMessage::assistant("final1"),
        BaseMessage::user("u2"),
        BaseMessage::assistant("newest"),
    ];
    let before = messages.clone();

    let event = compressor
        .process(&mut messages, &fixture.context(true))
        .await
        .unwrap();

    assert!(event.is_none(), "the envelope is larger than the body");
    assert_eq!(messages, before);
    assert_eq!(
        fixture.archives().await,
        Vec::<String>::new(),
        "the rejected replacement must remove the archive it wrote"
    );
}

#[tokio::test]
async fn the_current_round_span_is_archived() {
    let fixture = Fixture::new();
    let compressor = CurrentRoundCompressor::new(current_round_config(), None);
    let mut messages = vec![
        BaseMessage::user("q"),
        BaseMessage::assistant("").with_tool_calls(vec![ToolCall::new("read_file").with_id("c1")]),
        BaseMessage::tool("c1", long("tool output 1")),
        BaseMessage::assistant(format!("tool output 1 confirmed: {}", long("halfway"))),
        BaseMessage::assistant("done"),
    ];

    compressor
        .process(&mut messages, &fixture.context(true))
        .await
        .unwrap()
        .expect("stage one must compress the span");

    let blocks: Vec<&BaseMessage> = messages
        .iter()
        .filter(|message| is_memory_block(message, CURRENT_ROUND_MEMORY_BLOCK))
        .collect();
    assert_eq!(blocks.len(), 1);
    let id = trailing_marker(blocks[0]).expect("the stage-one block is marked");
    assert_eq!(fixture.archives().await, vec![id.clone()]);
    let chunks = fixture.recall_chunks(&id, "tool output 1").await;
    assert!(chunks.iter().any(|chunk| chunk.contains("tool output 1")));
    // The user request and the newest assistant turn stay raw.
    assert_eq!(messages[0].text(), "q");
    assert_eq!(messages.last().unwrap().text(), "done");
}

#[tokio::test]
async fn current_round_merges_keep_every_marker_once() {
    let fixture = Fixture::new();
    let mut config = current_round_config();
    config.summary_merge_min_blocks = 3;
    let compressor = CurrentRoundCompressor::new(config, None);

    let mut messages = vec![
        BaseMessage::user("q"),
        current_round_block(Some("id1"), &long("body a")),
        current_round_block(Some("id2"), &long("body b")),
        current_round_block(Some("id3"), &long("body c")),
        BaseMessage::assistant("done"),
    ];
    compressor
        .process(&mut messages, &fixture.context(true))
        .await
        .unwrap()
        .expect("the merge must collapse the block run");

    assert_eq!(messages.len(), 3);
    let merged = &messages[1];
    assert!(is_memory_block(merged, CURRENT_ROUND_MEMORY_BLOCK));
    assert!(merged.text().contains("\n---\n"));
    for id in ["id1", "id2", "id3"] {
        assert_eq!(
            merged.text().matches(&recall_marker(id)).count(),
            1,
            "marker {id} must survive exactly once"
        );
    }
    // Merging never writes a new archive: the individual ones stay valid.
    assert_eq!(fixture.archives().await, Vec::<String>::new());

    // A second merge pass over the already merged block keeps the ids once, and
    // adds the new ones without duplicating anything.
    let mut second = vec![
        BaseMessage::user("q"),
        merged.clone(),
        current_round_block(Some("id4"), &long("body d")),
        current_round_block(Some("id5"), &long("body e")),
        BaseMessage::assistant("done"),
    ];
    compressor
        .process(&mut second, &fixture.context(true))
        .await
        .unwrap()
        .expect("the second merge must collapse the run");

    assert_eq!(second.len(), 3);
    let merged = &second[1];
    for id in ["id1", "id2", "id3", "id4", "id5"] {
        assert_eq!(
            merged.text().matches(&recall_marker(id)).count(),
            1,
            "marker {id} must survive exactly once across both passes"
        );
    }
}

#[tokio::test]
async fn round_level_phase_one_blocks_are_marked() {
    let fixture = Fixture::new();
    let compressor = RoundLevelCompressor::new(round_level_config(5_000, 0), None);
    let mut messages = history(3);

    compressor
        .process(&mut messages, &fixture.context(false))
        .await
        .unwrap()
        .expect("the ladder must run above the target");

    let ids = markers(&messages, ROUND_LEVEL_MEMORY_BLOCK);
    assert_eq!(ids.len(), 3);
    let mut stored = ids.clone();
    stored.sort();
    assert_eq!(fixture.archives().await, stored);
    let chunks = fixture.recall_chunks(&ids[0], "tool output 1").await;
    assert!(
        chunks.iter().any(|chunk| chunk.contains("tool output 1")),
        "each phase-one block archives its own round"
    );
}

#[tokio::test]
async fn hard_truncation_archives_the_removed_prefix() {
    let fixture = Fixture::new();
    // First- and second-pass blocks are larger than their rounds, so only the
    // hard truncation can bring the history down; the rejected candidates roll
    // their archives back.
    let mut config = round_level_config(2_000, 0);
    config.first_pass_target_tokens = 250_000;
    config.second_pass_target_tokens = 250_000;
    let compressor = RoundLevelCompressor::new(config, None);
    let mut messages = history(4);

    compressor
        .process(&mut messages, &fixture.context(false))
        .await
        .unwrap()
        .expect("hard truncation must run");

    assert_eq!(messages.len(), 2);
    let block = &messages[1];
    assert!(is_memory_block(block, ROUND_LEVEL_MEMORY_BLOCK));
    assert_eq!(scope_of(block), TRUNCATED_FULL_CONTEXT_SCOPE);
    assert!(block.text().contains("[TRUNCATED]"));
    let id = trailing_marker(block).expect("the truncated block is marked");
    assert_eq!(block.text().matches(&recall_marker(&id)).count(), 1);
    // Every rejected phase-one candidate was rolled back here.
    assert_eq!(fixture.archives().await, vec![id.clone()]);
    let chunks = fixture.recall_chunks(&id, "tool output 1").await;
    assert!(
        chunks.iter().any(|chunk| chunk.contains("tool output 1")),
        "the removed prefix keeps its tool output"
    );
}

#[tokio::test]
async fn round_level_merges_keep_every_marker_once() {
    let fixture = Fixture::new();
    let mut config = round_level_config(6_000, 0);
    // A generous merge cap keeps the joined bodies, so every marker survives.
    config.third_pass_target_tokens = 10_000;
    let compressor = RoundLevelCompressor::new(config, None);
    let mut messages = history(6);

    compressor
        .process(&mut messages, &fixture.context(false))
        .await
        .unwrap()
        .expect("the ladder must run");

    let blocks: Vec<&BaseMessage> = messages
        .iter()
        .filter(|message| is_memory_block(message, ROUND_LEVEL_MEMORY_BLOCK))
        .collect();
    assert_eq!(blocks.len(), 1);
    let text = blocks[0].text().into_owned();
    // Every phase-one archive survives the merge, and the merged block carries
    // each of its markers exactly once.
    let stored = fixture.archives().await;
    assert_eq!(stored.len(), 6);
    for id in &stored {
        assert_eq!(text.matches(&recall_marker(id)).count(), 1);
    }
    assert_eq!(blocks[0].text().matches("\n---\n").count(), 5);
}

#[tokio::test]
async fn full_compaction_archives_the_active_history() {
    let fixture = Fixture::new();
    let compressor = FullCompactProcessor::new(compact_config(), None, None);
    let mut messages = conversation();
    let before = count_messages(&fixture.counter, &messages);

    let event = compressor
        .process(&mut messages, &fixture.context(false))
        .await
        .unwrap()
        .expect("an oversized history must compact");

    assert_eq!(messages.len(), 2);
    assert!(messages[1].text().starts_with(FULL_COMPACT_SUMMARY_INTRO));
    let id = trailing_marker(&messages[1]).expect("the summary message is marked");
    assert_eq!(fixture.archives().await, vec![id.clone()]);
    let chunks = fixture.recall_chunks(&id, "borrow").await;
    assert!(
        chunks.iter().any(|chunk| chunk.contains("borrow")),
        "the compacted history is recoverable"
    );
    assert!(event.tokens_after < before);
}

#[tokio::test]
async fn full_compaction_session_memory_path_also_archives() {
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
    let anchor = messages[4].metadata[CONTEXT_MESSAGE_ID_KEY]
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
        trigger_total_tokens: 40_000,
        messages_to_keep: 10,
        ..FullCompactProcessorConfig::default()
    };
    let compressor = FullCompactProcessor::new(config, None, Some(store));

    compressor
        .process(&mut messages, &fixture.context(false))
        .await
        .unwrap()
        .expect("the session-memory candidate must be accepted");

    // Boundary, then the notes summary carrying the recall marker.
    assert!(messages[0].text().starts_with("[SESSION_MEMORY_BOUNDARY]"));
    assert!(messages[1].text().contains("the notes"));
    let id = trailing_marker(&messages[1]).expect("the notes summary is marked");
    assert_eq!(fixture.archives().await, vec![id.clone()]);
    let chunks = fixture.recall_chunks(&id, "u1").await;
    assert!(chunks.iter().any(|chunk| chunk.contains("u1: detail")));
}
