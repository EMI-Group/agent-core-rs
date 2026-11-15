#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//! Public-API integration tests for the compression-recall subsystem.
//!
//! Everything is offline and deterministic: archives are built from explicit
//! message lists or struct literals with caller-supplied ids and timestamps, and
//! no model, network, or wall clock is involved.
use std::sync::Arc;

use openjiuwen_common::error::Error;
use openjiuwen_context::recall::{estimate_tokens, split_text, tokenize};
use openjiuwen_context::{
    archive_from_messages, group_turns, parse_recall_marker, recall_compressed_context,
    recall_marker, try_archive_from_messages, Bm25Index, CompressionRecallConfig,
    InMemoryRecallArchiveStore, KvRecallArchiveStore, OffloadScope, RecallArchive,
    RecallArchiveStore, RecallChunk, RecallOptions, RecallTurn, COMPRESSION_RECALL_MARKER_PREFIX,
};
use openjiuwen_foundation::prompt::{BaseMessage, ToolCall};
use openjiuwen_foundation::store::{InMemoryKvStore, KvStore};

/// A scope over `session`/`context`.
fn scope(session: &str, context: &str) -> OffloadScope {
    OffloadScope::new(session, context)
}

/// The default recall options.
fn options() -> RecallOptions {
    RecallOptions::default()
}

/// An archive with one turn and one chunk.
fn single_turn_archive(memory_id: &str, query: &str, text: &str) -> RecallArchive {
    RecallArchive {
        memory_id: memory_id.to_owned(),
        session_id: "session".to_owned(),
        context_id: "main".to_owned(),
        created_at: "2026-01-01T00:00:00Z".to_owned(),
        turns: vec![RecallTurn {
            turn_id: "turn_0001".to_owned(),
            query: query.to_owned(),
            answer: String::new(),
            chunk_ids: vec!["turn_0001_chunk_000".to_owned()],
        }],
        chunks: vec![RecallChunk {
            chunk_id: "turn_0001_chunk_000".to_owned(),
            turn_id: "turn_0001".to_owned(),
            chunk_index: 0,
            text: text.to_owned(),
            tokens: estimate_tokens(text),
            estimated: true,
        }],
    }
}

#[test]
fn internal_user_prefixes_do_not_open_turns() {
    let messages = vec![
        BaseMessage::user("real question"),
        BaseMessage::assistant("real answer"),
        BaseMessage::user("<system-reminder>internal"),
        BaseMessage::user("  <recovered_context>more internal"),
        BaseMessage::user("[STATE_REINJECTION]state"),
        BaseMessage::assistant("trailing answer"),
    ];
    let turns = group_turns(&messages);
    assert_eq!(turns.len(), 1);
    assert_eq!(turns[0].turn_id, "turn_0001");
    assert_eq!(turns[0].query, "real question");
    assert_eq!(turns[0].answer, "trailing answer");
    assert_eq!(turns[0].message_indices, vec![0, 1, 2, 3, 4, 5]);
}

#[test]
fn tool_rounds_attach_to_the_preceding_turn() {
    let messages = vec![
        BaseMessage::user("inspect the repo"),
        BaseMessage::assistant("calling").with_tool_calls(vec![ToolCall::new("list_dir")
            .with_id("call_1")
            .with_arguments(serde_json::json!({"path": "."}))]),
        BaseMessage::tool("call_1", "src\nCargo.toml"),
        BaseMessage::assistant("the crate has a src directory"),
    ];
    let turns = group_turns(&messages);
    assert_eq!(turns.len(), 1);
    assert_eq!(turns[0].query, "inspect the repo");
    assert_eq!(turns[0].answer, "the crate has a src directory");
    assert_eq!(turns[0].message_indices, vec![0, 1, 2, 3]);
}

#[test]
fn leading_assistant_and_tool_messages_form_an_implicit_turn() {
    let messages = vec![
        BaseMessage::assistant("prologue"),
        BaseMessage::tool("call_0", "output"),
        BaseMessage::user("question"),
    ];
    let turns = group_turns(&messages);
    assert_eq!(turns.len(), 2);
    assert_eq!(turns[0].query, "");
    assert_eq!(turns[0].answer, "prologue");
    assert_eq!(turns[0].message_indices, vec![0, 1]);
    assert_eq!(turns[1].query, "question");
}

#[test]
fn chunking_respects_multibyte_character_boundaries() {
    let chunks = split_text("你好世界你好世界", 2, 0).unwrap();
    assert_eq!(chunks.concat(), "你好世界你好世界");
    assert!(chunks.iter().all(|chunk| chunk.chars().count() == 2));
    assert!(chunks
        .iter()
        .all(|chunk| chunk.chars().all(char::is_alphabetic)));
}

#[test]
fn chunking_honours_the_configured_overlap() {
    let chunks = split_text("abcdefghij", 2, 1).unwrap();
    assert_eq!(chunks, vec!["abcdefgh", "efghij"]);
    assert_eq!(chunks[0].chars().rev().take(4).collect::<String>(), "hgfe");
}

#[test]
fn chunking_emits_one_chunk_per_wide_character() {
    let chunks = split_text("😀😀", 1, 0).unwrap();
    assert_eq!(chunks, vec!["😀", "😀"]);
}

#[test]
fn chunk_tokens_use_the_documented_estimate() {
    assert_eq!(estimate_tokens("abc"), 1);
    assert_eq!(estimate_tokens("abcd"), 1);
    assert_eq!(estimate_tokens("abcde"), 1);
    assert_eq!(estimate_tokens("abcdef"), 2);
    // Characters, not UTF-8 bytes: `你好` is two characters and six bytes.
    assert_eq!(estimate_tokens("你好"), 1);
    assert_eq!(estimate_tokens("你好世界你好"), 2);
}

#[test]
fn archive_from_messages_groups_turns_and_chunks() {
    let messages = vec![
        BaseMessage::user("question one"),
        BaseMessage::assistant("answer one"),
        BaseMessage::user("<system-reminder>note"),
        BaseMessage::user("question two"),
        BaseMessage::assistant("answer two"),
    ];
    let config = CompressionRecallConfig {
        enabled: true,
        chunk_size_tokens: 3_000,
        chunk_overlap_tokens: 300,
    };
    let archive = archive_from_messages(
        "mem_1",
        "session",
        "main",
        "2026-01-01T00:00:00Z",
        &messages,
        &config,
    );
    assert_eq!(archive.memory_id, "mem_1");
    assert_eq!(archive.turns.len(), 2);
    assert_eq!(archive.turns[0].query, "question one");
    assert_eq!(archive.turns[0].answer, "answer one");
    assert_eq!(archive.turns[1].query, "question two");
    assert_eq!(archive.turns[0].chunk_ids, vec!["turn_0001_chunk_000"]);
    assert_eq!(archive.chunks.len(), 2);
    assert_eq!(archive.chunks[0].chunk_index, 0);
    assert_eq!(
        archive.chunks[0].tokens,
        estimate_tokens(&archive.chunks[0].text)
    );
    assert!(archive.chunks[0].text.contains("## User\nquestion one"));
    assert!(archive.chunks[0].text.contains("## Assistant\nanswer one"));
    assert!(archive.chunk("turn_0001_chunk_000").is_some());
    assert!(archive.turn("turn_0002").is_some());
}

#[test]
fn empty_messages_produce_an_empty_archive() {
    let archive = archive_from_messages(
        "mem_1",
        "session",
        "main",
        "2026-01-01T00:00:00Z",
        &[],
        &CompressionRecallConfig::default(),
    );
    assert!(archive.is_empty());
    assert_eq!(archive.memory_id, "mem_1");
    assert_eq!(archive.created_at, "2026-01-01T00:00:00Z");
}

#[test]
fn invalid_recall_configuration_is_a_config_error() {
    let config = CompressionRecallConfig {
        enabled: true,
        chunk_size_tokens: 100,
        chunk_overlap_tokens: 100,
    };
    let messages = vec![BaseMessage::user("hello")];
    let error = try_archive_from_messages("m", "s", "main", "t", &messages, &config).unwrap_err();
    assert!(matches!(error, Error::Config(_)));

    let fallback = archive_from_messages("m", "s", "main", "t", &messages, &config);
    assert!(fallback.is_empty());
}

#[test]
fn bm25_ranks_documents_and_ties_deterministically() {
    let index = Bm25Index::build(&["the quick brown fox", "quick brown", "lazy dog"]);
    let scores = index.scores("fox");
    assert!(scores[0] > 0.0);
    assert_eq!(scores[1], 0.0);
    assert_eq!(scores[2], 0.0);

    let tied = Bm25Index::build(&["same text", "same text"]);
    let scores = tied.scores("same");
    assert!((scores[0] - scores[1]).abs() < 1e-12);
}

#[test]
fn bm25_idf_favours_rare_terms() {
    let index = Bm25Index::build(&[
        "rare token here",
        "common word",
        "common word too",
        "common again",
    ]);
    let scores = index.scores("rare common");
    assert!(scores[0] > scores[1]);
}

#[test]
fn bm25_matches_cjk_queries() {
    let index = Bm25Index::build(&["你好世界", "hello world"]);
    let scores = index.scores("你好");
    assert!(scores[0] > 0.0);
    assert_eq!(scores[1], 0.0);
    assert_eq!(tokenize("你好世界")[0], "你好世界");
}

#[tokio::test]
async fn archive_store_round_trips_and_isolates_scopes() {
    let store = InMemoryRecallArchiveStore::new();
    assert!(store.is_empty());
    let first = scope("s1", "main");
    let second = scope("s2", "main");
    store
        .put(&first, single_turn_archive("m1", "rust", "rust ownership"))
        .await
        .unwrap();
    store
        .put(
            &second,
            single_turn_archive("m1", "python", "python packaging"),
        )
        .await
        .unwrap();
    assert_eq!(store.len(), 2);

    assert_eq!(
        store.get(&first, "m1").await.unwrap().unwrap().turns[0].query,
        "rust"
    );
    assert_eq!(
        store.get(&second, "m1").await.unwrap().unwrap().turns[0].query,
        "python"
    );
    assert_eq!(store.list(&first).await.unwrap(), vec!["m1"]);
    assert!(store
        .get(&scope("s3", "main"), "m1")
        .await
        .unwrap()
        .is_none());

    store.remove(&first, "m1").await.unwrap();
    assert!(store.get(&first, "m1").await.unwrap().is_none());
    assert!(store.get(&second, "m1").await.unwrap().is_some());
    store.remove(&first, "missing").await.unwrap();
}

#[tokio::test]
async fn kv_archive_store_round_trips_and_isolates_scopes() {
    let kv: Arc<dyn KvStore> = Arc::new(InMemoryKvStore::new());
    let store = KvRecallArchiveStore::new(Arc::clone(&kv));
    let first = scope("s1", "main");
    let second = scope("s2", "main");

    store
        .put(&first, single_turn_archive("m1", "rust", "rust ownership"))
        .await
        .unwrap();
    store
        .put(
            &second,
            single_turn_archive("m1", "python", "python packaging"),
        )
        .await
        .unwrap();

    assert_eq!(
        store.get(&first, "m1").await.unwrap().unwrap().turns[0].query,
        "rust"
    );
    assert_eq!(store.list(&first).await.unwrap(), vec!["m1"]);
    assert!(store
        .get(&scope("s3", "main"), "m1")
        .await
        .unwrap()
        .is_none());

    store.remove(&first, "m1").await.unwrap();
    assert!(store.get(&first, "m1").await.unwrap().is_none());
    assert!(store.get(&second, "m1").await.unwrap().is_some());
    assert_eq!(store.key(&first, "m1"), "context:recall:s1_main_m1");
}

#[test]
fn archive_round_trips_through_json() {
    let archive = single_turn_archive("m1", "question", "answer");
    let json = serde_json::to_string(&archive).unwrap();
    let restored: RecallArchive = serde_json::from_str(&json).unwrap();
    assert_eq!(archive, restored);

    let partial: RecallArchive = serde_json::from_str(r#"{"memory_id": "m9"}"#).unwrap();
    assert_eq!(partial.memory_id, "m9");
    assert_eq!(partial.turns.len(), 0);
}

#[tokio::test]
async fn recall_returns_the_top_turn_and_chunks() {
    let store = InMemoryRecallArchiveStore::new();
    let scoped = scope("session", "main");
    let archive = RecallArchive {
        memory_id: "m1".to_owned(),
        session_id: "session".to_owned(),
        context_id: "main".to_owned(),
        created_at: "2026-01-01T00:00:00Z".to_owned(),
        turns: vec![
            RecallTurn {
                turn_id: "turn_0001".to_owned(),
                query: "rust ownership".to_owned(),
                answer: "borrow checker".to_owned(),
                chunk_ids: vec!["c1".to_owned(), "c2".to_owned()],
            },
            RecallTurn {
                turn_id: "turn_0002".to_owned(),
                query: "python packaging".to_owned(),
                answer: "wheels".to_owned(),
                chunk_ids: vec!["c3".to_owned()],
            },
        ],
        chunks: vec![
            RecallChunk {
                chunk_id: "c1".to_owned(),
                turn_id: "turn_0001".to_owned(),
                chunk_index: 0,
                text: "rust ownership and the borrow checker".to_owned(),
                tokens: 10,
                estimated: true,
            },
            RecallChunk {
                chunk_id: "c2".to_owned(),
                turn_id: "turn_0001".to_owned(),
                chunk_index: 1,
                text: "unrelated appendix".to_owned(),
                tokens: 10,
                estimated: true,
            },
            RecallChunk {
                chunk_id: "c3".to_owned(),
                turn_id: "turn_0002".to_owned(),
                chunk_index: 0,
                text: "python packaging wheels".to_owned(),
                tokens: 10,
                estimated: true,
            },
        ],
    };
    store.put(&scoped, archive).await.unwrap();

    let result =
        recall_compressed_context(&store, &scoped, Some("m1"), "borrow checker", &options())
            .await
            .unwrap();
    assert_eq!(result.memory_id.as_deref(), Some("m1"));
    assert_eq!(result.archives, vec!["m1"]);
    assert_eq!(result.matched_turns.len(), 1);
    assert_eq!(result.matched_turns[0].turn_id, "turn_0001");
    assert!(result.matched_turns[0].score > 0.0);
    assert_eq!(result.chunks.len(), 1);
    assert_eq!(result.chunks[0].chunk_id, "c1");
    assert_eq!(result.chunks[0].memory_id, "m1");
    assert_eq!(result.returned_tokens, 10);
    assert!(!result.truncated);
}

#[tokio::test]
async fn turn_ratio_cutoff_drops_weak_turns() {
    let store = InMemoryRecallArchiveStore::new();
    let scoped = scope("session", "main");
    let archive = RecallArchive {
        memory_id: "m1".to_owned(),
        session_id: "session".to_owned(),
        context_id: "main".to_owned(),
        created_at: "2026-01-01T00:00:00Z".to_owned(),
        turns: vec![
            RecallTurn {
                turn_id: "turn_0001".to_owned(),
                query: "alpha alpha alpha alpha".to_owned(),
                answer: String::new(),
                chunk_ids: vec!["c1".to_owned()],
            },
            RecallTurn {
                turn_id: "turn_0002".to_owned(),
                query: "alpha".to_owned(),
                answer:
                    "beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron \
                         pi rho sigma tau"
                        .to_owned(),
                chunk_ids: vec!["c2".to_owned()],
            },
        ],
        chunks: vec![
            RecallChunk {
                chunk_id: "c1".to_owned(),
                turn_id: "turn_0001".to_owned(),
                chunk_index: 0,
                text: "alpha alpha".to_owned(),
                tokens: 5,
                estimated: true,
            },
            RecallChunk {
                chunk_id: "c2".to_owned(),
                turn_id: "turn_0002".to_owned(),
                chunk_index: 0,
                text: "alpha".to_owned(),
                tokens: 5,
                estimated: true,
            },
        ],
    };
    store.put(&scoped, archive).await.unwrap();

    let default_result =
        recall_compressed_context(&store, &scoped, Some("m1"), "alpha", &options())
            .await
            .unwrap();
    assert_eq!(default_result.matched_turns.len(), 2);

    let strict = RecallOptions {
        turn_score_ratio: 0.5,
        ..RecallOptions::default()
    };
    let strict_result = recall_compressed_context(&store, &scoped, Some("m1"), "alpha", &strict)
        .await
        .unwrap();
    assert_eq!(strict_result.matched_turns.len(), 1);
    assert_eq!(strict_result.matched_turns[0].turn_id, "turn_0001");
}

#[tokio::test]
async fn budget_skips_over_limit_chunks_and_sets_truncated() {
    let store = InMemoryRecallArchiveStore::new();
    let scoped = scope("session", "main");
    let archive = RecallArchive {
        memory_id: "m1".to_owned(),
        session_id: "session".to_owned(),
        context_id: "main".to_owned(),
        created_at: "2026-01-01T00:00:00Z".to_owned(),
        turns: vec![RecallTurn {
            turn_id: "turn_0001".to_owned(),
            query: "alpha".to_owned(),
            answer: String::new(),
            chunk_ids: vec!["c1".to_owned(), "c2".to_owned()],
        }],
        chunks: vec![
            RecallChunk {
                chunk_id: "c1".to_owned(),
                turn_id: "turn_0001".to_owned(),
                chunk_index: 0,
                text: "alpha first".to_owned(),
                tokens: 100,
                estimated: true,
            },
            RecallChunk {
                chunk_id: "c2".to_owned(),
                turn_id: "turn_0001".to_owned(),
                chunk_index: 1,
                text: "alpha second".to_owned(),
                tokens: 100,
                estimated: true,
            },
        ],
    };
    store.put(&scoped, archive).await.unwrap();

    let tight = RecallOptions {
        max_content_tokens: 100,
        ..RecallOptions::default()
    };
    let result = recall_compressed_context(&store, &scoped, Some("m1"), "alpha", &tight)
        .await
        .unwrap();
    assert_eq!(result.chunks.len(), 1);
    assert_eq!(result.chunks[0].chunk_id, "c1");
    assert_eq!(result.returned_tokens, 100);
    assert!(result.truncated);
}

#[tokio::test]
async fn blank_query_and_missing_archive_are_empty_results() {
    let store = InMemoryRecallArchiveStore::new();
    let scoped = scope("session", "main");
    store
        .put(&scoped, single_turn_archive("m1", "rust", "rust ownership"))
        .await
        .unwrap();

    let blank = recall_compressed_context(&store, &scoped, None, "   ", &options())
        .await
        .unwrap();
    assert_eq!(blank.matched_turns.len(), 0);
    assert_eq!(blank.chunks.len(), 0);
    assert_eq!(blank.archives.len(), 0);

    let missing = recall_compressed_context(&store, &scoped, Some("absent"), "rust", &options())
        .await
        .unwrap();
    assert_eq!(missing.memory_id.as_deref(), Some("absent"));
    assert_eq!(missing.matched_turns.len(), 0);
    assert_eq!(missing.chunks.len(), 0);
}

#[tokio::test]
async fn cross_archive_merge_normalizes_scores_and_breaks_ties() {
    let store = InMemoryRecallArchiveStore::new();
    let scoped = scope("session", "main");
    store
        .put(
            &scoped,
            single_turn_archive("mem-a", "alpha", "alpha from archive a"),
        )
        .await
        .unwrap();
    store
        .put(
            &scoped,
            single_turn_archive("mem-b", "alpha", "alpha from archive b beta beta beta beta"),
        )
        .await
        .unwrap();

    let result = recall_compressed_context(&store, &scoped, None, "alpha", &options())
        .await
        .unwrap();
    assert_eq!(result.archives, vec!["mem-a", "mem-b"]);
    assert_eq!(result.matched_turns.len(), 2);
    assert!(result.matched_turns.iter().all(|turn| turn.score <= 1.0));
    assert!((result.matched_turns[0].score - 1.0).abs() < f32::EPSILON);
    assert_eq!(result.matched_turns[0].memory_id, "mem-a");
    assert_eq!(result.chunks.len(), 2);
    assert_eq!(result.chunks[0].memory_id, "mem-a");
    assert!((result.chunks[0].score - 1.0).abs() < f32::EPSILON);
}

#[tokio::test]
async fn cross_archive_ties_are_deterministic() {
    let store = InMemoryRecallArchiveStore::new();
    let scoped = scope("session", "main");
    for memory_id in ["mem-b", "mem-a"] {
        store
            .put(
                &scoped,
                single_turn_archive(memory_id, "alpha", "alpha beta"),
            )
            .await
            .unwrap();
    }
    let result = recall_compressed_context(&store, &scoped, None, "alpha", &options())
        .await
        .unwrap();
    assert_eq!(result.chunks.len(), 2);
    assert_eq!(result.chunks[0].memory_id, "mem-a");
    assert_eq!(result.chunks[1].memory_id, "mem-b");
}

#[test]
fn recall_marker_round_trips() {
    assert_eq!(recall_marker("mem_1"), "[[COMPRESSION_RECALL: id=mem_1]]");
    assert!(recall_marker("mem_1").starts_with(COMPRESSION_RECALL_MARKER_PREFIX));
    let text = format!(
        "summary text\n\n{}\nmore text",
        recall_marker("20260101T000000000000Z")
    );
    assert_eq!(
        parse_recall_marker(&text).as_deref(),
        Some("20260101T000000000000Z")
    );
    assert_eq!(parse_recall_marker("no marker here"), None);
    assert_eq!(parse_recall_marker("[[COMPRESSION_RECALL: id=]]"), None);
    assert_eq!(parse_recall_marker("[[COMPRESSION_RECALL: other=x]]"), None);
}

#[test]
fn recall_options_round_trip_through_json() {
    let options = RecallOptions {
        max_content_tokens: 1_234,
        turn_top_k: 3,
        chunk_top_k: 4,
        turn_score_ratio: 0.25,
    };
    let json = serde_json::to_string(&options).unwrap();
    let restored: RecallOptions = serde_json::from_str(&json).unwrap();
    assert_eq!(options, restored);

    let defaults: RecallOptions = serde_json::from_str(r#"{"unknown": true}"#).unwrap();
    assert_eq!(defaults, RecallOptions::default());
    assert_eq!(defaults.max_content_tokens, 8_000);
    assert_eq!(defaults.turn_top_k, 2);
    assert_eq!(defaults.chunk_top_k, 2);
    assert!((defaults.turn_score_ratio - 0.3).abs() < f32::EPSILON);
}
