//! Behaviour tests for trace ingestion, clustering, and the experience collector.
//!
//! Everything is offline and deterministic: `tempfile` workspaces for the session fixtures
//! and the catalogs, explicit numeric timestamps instead of the wall clock,
//! `HashEmbedding` for the persisted index path, and no assertion on hash-map ordering.
#![allow(clippy::unwrap_used, clippy::expect_used)]
#![allow(clippy::float_cmp)]
#![allow(clippy::assert_is_empty)]
use std::path::Path;
use std::sync::Arc;

use serde_json::{json, Value};

use openjiuwen_retrieval::HashEmbedding;
use openjiuwen_symphony::experience::{
    cluster_traces, ExperienceBank, ExperienceCollector, ExperienceIndex, ExperienceRanking,
    ExperienceSink, TraceRecord, TraceStore, TraceToolCall, CLUSTER_SIMILARITY_THRESHOLD,
    PATTERN_MERGE_THRESHOLD, PENDING_FLUSH_THRESHOLD, QUERY_EXAMPLES_COUNT,
};

/// Writes one session directory (`metadata.json` and, when given, `history.json`).
async fn write_session(root: &Path, session_id: &str, metadata: &Value, history: Option<&Value>) {
    let session_dir = root.join("sessions").join(session_id);
    tokio::fs::create_dir_all(&session_dir)
        .await
        .expect("create session dir");
    tokio::fs::write(
        session_dir.join("metadata.json"),
        serde_json::to_vec(metadata).expect("encode metadata"),
    )
    .await
    .expect("write metadata");
    if let Some(history) = history {
        tokio::fs::write(
            session_dir.join("history.json"),
            serde_json::to_vec(history).expect("encode history"),
        )
        .await
        .expect("write history");
    }
}

/// A session that succeeds with two tool calls selecting three skills.
async fn write_success_session(root: &Path) {
    let metadata = json!({"session_id": "sess-success", "status": "succeeded"});
    let history = json!([
        {
            "request_id": "req-1",
            "role": "user",
            "content": "Summarize the quarterly report",
            "timestamp": 1_700_000_000_u64
        },
        {
            "request_id": "req-1",
            "role": "assistant",
            "event_type": "chat.tool_call",
            "timestamp": 1_700_000_001_u64,
            "tool_call": {
                "name": "skill_tool",
                "arguments": "{\"skill_name\": \"summarizer\"}"
            }
        },
        {
            "request_id": "req-1",
            "role": "assistant",
            "event_type": "chat.tool_result",
            "tool_name": "skill_tool",
            "result": "ok",
            "duration_ms": 120
        },
        {
            "request_id": "req-1",
            "role": "assistant",
            "event_type": "chat.tool_call",
            "tool_call": {
                "name": "symphony_compose_graph",
                "arguments": {"candidate_skill_ids": ["formatter", "summarizer"]}
            }
        },
        {
            "request_id": "req-1",
            "role": "assistant",
            "event_type": "chat.tool_result",
            "tool_name": "symphony_compose_graph",
            "result": "done",
            "raw_output": {"plan": {"steps": [{"skill_id": "formatter"}, {"name": "charts"}]}}
        },
        {
            "request_id": "req-1",
            "role": "assistant",
            "event_type": "chat.delta",
            "content": "Here is "
        },
        {
            "request_id": "req-1",
            "role": "assistant",
            "event_type": "chat.final",
            "content": "Here is the summary.",
            "duration_ms": 250
        }
    ]);
    write_session(root, "sess-success", &metadata, Some(&history)).await;
}

/// A session that fails with an error event and a failed metadata status.
async fn write_failure_session(root: &Path) {
    let metadata = json!({"status": "failed", "error": "RuntimeError"});
    let history = json!([
        {
            "request_id": "req-1",
            "role": "user",
            "content": "Generate the invoice",
            "timestamp": 1_700_000_010_u64
        },
        {
            "request_id": "req-1",
            "role": "assistant",
            "event_type": "chat.error",
            "content": "boom"
        }
    ]);
    write_session(root, "sess-failure", &metadata, Some(&history)).await;
}

/// The expected record of [`write_success_session`].
fn expected_success_record() -> TraceRecord {
    TraceRecord {
        trace_id: "sess-success/req-1".to_string(),
        session_id: "sess-success".to_string(),
        request_id: "req-1".to_string(),
        query: "Summarize the quarterly report".to_string(),
        final_output: "Here is the summary.".to_string(),
        skills: vec![
            "charts".to_string(),
            "formatter".to_string(),
            "summarizer".to_string(),
        ],
        success: true,
        duration_ms: 250,
        tool_calls: vec![
            TraceToolCall {
                name: "skill_tool".to_string(),
                arguments: json!({"skill_name": "summarizer"}),
                result: Some("ok".to_string()),
            },
            TraceToolCall {
                name: "symphony_compose_graph".to_string(),
                arguments: json!({"candidate_skill_ids": ["formatter", "summarizer"]}),
                result: Some("done".to_string()),
            },
        ],
        errors: Vec::new(),
        timestamp: 1_700_000_000.0,
    }
}

/// The expected record of [`write_failure_session`].
fn expected_failure_record() -> TraceRecord {
    TraceRecord {
        trace_id: "sess-failure/req-1".to_string(),
        session_id: "sess-failure".to_string(),
        request_id: "req-1".to_string(),
        query: "Generate the invoice".to_string(),
        final_output: String::new(),
        skills: Vec::new(),
        success: false,
        duration_ms: 0,
        tool_calls: Vec::new(),
        errors: vec!["boom".to_string()],
        timestamp: 1_700_000_010.0,
    }
}

/// Builds a minimal record for the clustering and collector tests.
fn record(trace_id: &str, query: &str, skills: &[&str], success: bool) -> TraceRecord {
    TraceRecord {
        trace_id: trace_id.to_string(),
        session_id: "s".to_string(),
        request_id: trace_id.to_string(),
        query: query.to_string(),
        skills: skills.iter().map(|skill| (*skill).to_string()).collect(),
        success,
        ..TraceRecord::default()
    }
}

#[tokio::test]
async fn parses_two_sessions_into_exact_records() {
    let dir = tempfile::tempdir().expect("tempdir");
    write_success_session(dir.path()).await;
    write_failure_session(dir.path()).await;

    let store = TraceStore::open(dir.path()).await.expect("open store");
    let records = store.parse_all_sessions().await.expect("parse");
    assert_eq!(records.len(), 2);
    assert_eq!(records[0], expected_success_record());
    assert_eq!(records[1], expected_failure_record());

    let index = tokio::fs::read_to_string(store.processed_index_path())
        .await
        .expect("read index");
    let index: Value = serde_json::from_str(&index).expect("decode index");
    assert_eq!(
        index["processed_ids"],
        json!(["sess-failure", "sess-success"])
    );
}

#[tokio::test]
async fn reparsing_is_idempotent() {
    let dir = tempfile::tempdir().expect("tempdir");
    write_success_session(dir.path()).await;
    write_failure_session(dir.path()).await;

    let store = TraceStore::open(dir.path()).await.expect("open store");
    assert_eq!(
        store.parse_all_sessions().await.expect("first parse").len(),
        2
    );
    assert!(store
        .parse_all_sessions()
        .await
        .expect("second parse")
        .is_empty());
    assert_eq!(store.load_all_records().await.expect("load").len(), 2);

    // A new session is parsed on its own, and the earlier records keep their order.
    write_session(
        dir.path(),
        "sess-new",
        &json!({"status": "ok"}),
        Some(&json!([
            {"request_id": "req-9", "role": "user", "content": "Translate this page", "timestamp": 1_700_000_020_u64},
            {"request_id": "req-9", "role": "assistant", "event_type": "chat.tool_call", "tool_call": {"name": "skill_tool", "arguments": {"skill_name": "translator"}}}
        ])),
    )
    .await;
    let fresh = store.parse_all_sessions().await.expect("third parse");
    assert_eq!(fresh.len(), 1);
    let all = store.load_all_records().await.expect("load all");
    assert_eq!(all.len(), 3);
    assert_eq!(all[0], expected_success_record());
    assert_eq!(all[1], expected_failure_record());
    assert_eq!(all[2].skills, vec!["translator"]);
}

#[tokio::test]
async fn malformed_and_missing_session_files_are_skipped() {
    let dir = tempfile::tempdir().expect("tempdir");
    write_success_session(dir.path()).await;

    // Corrupt metadata JSON: the session is skipped with a warning.
    let broken = dir.path().join("sessions").join("broken");
    tokio::fs::create_dir_all(&broken).await.expect("mkdir");
    tokio::fs::write(broken.join("metadata.json"), b"{not json")
        .await
        .expect("write");
    tokio::fs::write(broken.join("history.json"), b"[]")
        .await
        .expect("write");

    // No metadata at all: nothing to parse.
    let no_metadata = dir.path().join("sessions").join("no-metadata");
    tokio::fs::create_dir_all(&no_metadata)
        .await
        .expect("mkdir");
    tokio::fs::write(no_metadata.join("history.json"), b"[]")
        .await
        .expect("write");

    // Valid metadata, corrupt history: an empty history, not a failed scan.
    write_session(dir.path(), "bad-history", &json!({"status": "ok"}), None).await;
    tokio::fs::write(
        dir.path()
            .join("sessions")
            .join("bad-history")
            .join("history.json"),
        b"{not history",
    )
    .await
    .expect("write");

    // Heartbeat directories are never sessions.
    let heartbeat = dir.path().join("sessions").join("heartbeat_1");
    tokio::fs::create_dir_all(&heartbeat).await.expect("mkdir");
    tokio::fs::write(heartbeat.join("metadata.json"), b"{}")
        .await
        .expect("write");

    let store = TraceStore::open(dir.path()).await.expect("open store");
    let records = store.parse_all_sessions().await.expect("parse");
    assert_eq!(records.len(), 1);
    assert_eq!(records[0], expected_success_record());
}

#[tokio::test]
async fn single_quoted_tool_payloads_are_tolerated() {
    let dir = tempfile::tempdir().expect("tempdir");
    let history = json!([
        {"request_id": "r", "role": "user", "content": "Translate the contract"},
        {
            "request_id": "r",
            "role": "assistant",
            "event_type": "chat.tool_call",
            "tool_call": {"name": "skill_tool", "arguments": "{'skill_name': 'translator'}"}
        },
        {
            "request_id": "r",
            "role": "assistant",
            "event_type": "chat.tool_call",
            "tool_call": {"name": "symphony_compose_graph", "arguments": "{'candidate_skill_ids': ['formatter'], 'nested': {'ok': True}}"}
        },
        {
            "request_id": "r",
            "role": "assistant",
            "event_type": "chat.tool_call",
            "tool_call": {"name": "unknown_tool", "arguments": "{'a': b'bytes'}"}
        }
    ]);
    write_session(dir.path(), "sess", &json!({"status": "ok"}), Some(&history)).await;

    let store = TraceStore::open(dir.path()).await.expect("open store");
    let records = store.parse_all_sessions().await.expect("parse");
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].skills, vec!["formatter", "translator"]);
    assert_eq!(records[0].tool_calls.len(), 3);
    assert_eq!(
        records[0].tool_calls[0].arguments["skill_name"],
        "translator"
    );
    assert_eq!(
        records[0].tool_calls[1].arguments["candidate_skill_ids"][0],
        "formatter"
    );
    // The undecodable payload is dropped, not fatal.
    assert!(records[0].tool_calls[2].arguments.is_null());
    assert!(records[0].success);
}

#[tokio::test]
async fn clear_removes_records_and_the_processed_index() {
    let dir = tempfile::tempdir().expect("tempdir");
    write_success_session(dir.path()).await;
    let store = TraceStore::open(dir.path()).await.expect("open store");
    assert_eq!(store.parse_all_sessions().await.expect("parse").len(), 1);
    store.clear().await.expect("clear");
    assert!(store.load_all_records().await.expect("load").is_empty());
    assert!(!store.processed_index_path().exists());
    // Clearing twice is fine, and the cache directory stays.
    store.clear().await.expect("clear again");
    assert!(store.trace_store_dir().is_dir());
}

#[tokio::test]
async fn clustering_groups_by_skill_set_then_similarity() {
    let records = vec![
        record("t1", "deploy the kubernetes cluster", &["ops"], true),
        record("t2", "deploy the kubernetes cluster now", &["ops"], true),
        record("t3", "bake a chocolate cake", &["ops"], true),
        record(
            "t4",
            "deploy the kubernetes cluster",
            &["ops", "logging"],
            true,
        ),
    ];
    let clusters = cluster_traces(&records);
    assert_eq!(clusters.len(), 3);
    // Sorted by (skill_ids, pattern): ["logging", "ops"] sorts before ["ops"], and inside
    // the ["ops"] group the "bake ..." centroid sorts before "deploy ...".
    assert_eq!(clusters[0].skill_ids, vec!["logging", "ops"]);
    assert_eq!(clusters[0].members, vec!["t4"]);
    assert_eq!(clusters[1].skill_ids, vec!["ops"]);
    assert_eq!(clusters[1].members, vec!["t3"]);
    assert_eq!(clusters[2].skill_ids, vec!["ops"]);
    assert_eq!(clusters[2].members, vec!["t1", "t2"]);
    assert_eq!(clusters[2].pattern, "deploy the kubernetes cluster now");
    assert_eq!(
        clusters[2].query_examples,
        vec![
            "deploy the kubernetes cluster",
            "deploy the kubernetes cluster now"
        ]
    );

    // Deterministic: the same input always yields the same output.
    assert_eq!(clusters, cluster_traces(&records));
    assert!(CLUSTER_SIMILARITY_THRESHOLD > 0.0 && CLUSTER_SIMILARITY_THRESHOLD < 1.0);
}

#[tokio::test]
async fn collector_drops_unusable_traces_and_reports_stats() {
    let bank = Arc::new(ExperienceBank::new());
    let collector = ExperienceCollector::for_bank(Arc::clone(&bank));
    let mut failure = record("t4", "a failed run", &["summarizer"], false);
    failure.success = false;
    let records = vec![
        record("t1", "summarize the report", &["summarizer"], true),
        record("t2", "no skills here", &[], true),
        record("t3", "   ", &["summarizer"], true),
        failure,
    ];
    let produced = collector.add(&records).await.expect("add");
    assert_eq!(produced, 1);
    assert_eq!(collector.pending(), 1);
    assert_eq!(
        collector.stats(),
        openjiuwen_symphony::experience::CollectorStats {
            accepted: 1,
            dropped: 3,
            merged: 0,
            flushed: 0,
        }
    );

    assert_eq!(collector.flush().await.expect("flush"), 1);
    assert_eq!(collector.pending(), 0);
    assert_eq!(collector.flush().await.expect("flush again"), 0);
    assert_eq!(collector.stats().flushed, 1);
    assert_eq!(bank.len(), 1);
    assert_eq!(bank.list()[0].skill_ids, vec!["summarizer"]);
}

#[tokio::test]
async fn collector_auto_flushes_at_the_threshold() {
    let bank = Arc::new(ExperienceBank::new());
    let collector = ExperienceCollector::for_bank(Arc::clone(&bank));
    let records: Vec<TraceRecord> = (0..PENDING_FLUSH_THRESHOLD)
        .map(|index| {
            record(
                &format!("t{index}"),
                &format!("request number {index}"),
                &[&format!("skill-{index}")],
                true,
            )
        })
        .collect();
    let produced = collector.add(&records).await.expect("add");
    assert_eq!(produced, PENDING_FLUSH_THRESHOLD);
    assert_eq!(collector.pending(), 0);
    assert_eq!(collector.stats().flushed, PENDING_FLUSH_THRESHOLD);
    assert_eq!(bank.len(), PENDING_FLUSH_THRESHOLD);
}

#[tokio::test]
async fn collector_merges_similar_patterns_with_the_same_skills() {
    let bank = Arc::new(ExperienceBank::new());
    let collector = ExperienceCollector::for_bank(Arc::clone(&bank));
    let base = "alpha bravo charlie delta echo foxtrot golf hotel india";
    let extended = format!("{base} juliet");

    assert_eq!(
        collector
            .add(&[record("t1", base, &["a"], true)])
            .await
            .expect("first add"),
        1
    );
    assert_eq!(
        collector
            .add(&[record("t2", &extended, &["a"], true)])
            .await
            .expect("second add"),
        1
    );
    assert_eq!(collector.pending(), 1);
    assert_eq!(collector.stats().merged, 1);
    assert!(PATTERN_MERGE_THRESHOLD <= 0.9);

    assert_eq!(collector.flush().await.expect("flush"), 1);
    let items = bank.list();
    assert_eq!(items.len(), 1);
    assert_eq!(items[0].query_pattern, base);
    assert_eq!(
        items[0].query_examples,
        vec![base.to_string(), extended.clone()]
    );
    assert_eq!(items[0].success_count, 2);
}

#[tokio::test]
async fn end_to_end_trace_store_to_bank_search() {
    let dir = tempfile::tempdir().expect("tempdir");
    write_success_session(dir.path()).await;
    write_failure_session(dir.path()).await;
    let store = TraceStore::open(dir.path()).await.expect("open store");
    let records = store.parse_all_sessions().await.expect("parse");

    let clusters = cluster_traces(&records);
    assert_eq!(clusters.len(), 2);
    assert_eq!(
        clusters
            .iter()
            .find(|cluster| cluster.skill_ids.is_empty())
            .expect("failure cluster")
            .members,
        vec!["sess-failure/req-1"]
    );

    let bank = Arc::new(ExperienceBank::new());
    let collector = ExperienceCollector::for_bank(Arc::clone(&bank));
    assert_eq!(collector.add(&records).await.expect("add"), 1);
    assert_eq!(collector.flush().await.expect("flush"), 1);

    let hits = bank
        .search("summarize quarterly report", 5)
        .await
        .expect("search");
    assert!(!hits.is_empty());
    assert!(hits[0].item.has_skill("summarizer"));
    assert_eq!(hits[0].item.query_pattern, "Summarize the quarterly report");
    assert_eq!(hits[0].item.query_examples.len(), 1);
    assert!(hits[0].item.query_examples.len() <= QUERY_EXAMPLES_COUNT);
}

#[tokio::test]
async fn collector_writes_through_the_persisted_index_sink() {
    let dir = tempfile::tempdir().expect("tempdir");
    let index = Arc::new(
        ExperienceIndex::open(dir.path().join("index"), Arc::new(HashEmbedding::new(32)))
            .await
            .expect("open index"),
    );
    let collector = ExperienceCollector::new(index.clone() as Arc<dyn ExperienceSink>);
    assert_eq!(
        collector
            .add(&[record(
                "t1",
                "summarize the quarterly report",
                &["summarizer"],
                true
            )])
            .await
            .expect("add"),
        1
    );
    assert_eq!(collector.flush().await.expect("flush"), 1);
    assert_eq!(index.count(), 1);

    let hits = index
        .search("summarize the quarterly report", &ExperienceRanking::new(3))
        .await
        .expect("search");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].item.skill_ids, vec!["summarizer"]);
}
