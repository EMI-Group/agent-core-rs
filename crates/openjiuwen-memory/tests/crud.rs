#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//! Record CRUD integration tests for the [`openjiuwen_memory::facade::LongTermMemory`] surface.
//!
//! Covers `add`/`add_many`/`get`/`update`/`delete`/`clear`/`list`/`stats`, input and
//! scope validation, the Python-compatible [`openjiuwen_memory::record::MemoryRecord`]
//! JSON shape, and cancellation. Everything runs offline against the in-memory stores
//! with explicit timestamps, so no assertion depends on the wall clock.

use std::collections::HashSet;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use openjiuwen_common::error::Error;
use openjiuwen_memory::config::MemoryEngineConfig;
use openjiuwen_memory::facade::LongTermMemory;
use openjiuwen_memory::record::{
    AddOptions, CallOptions, MemoryFilter, MemoryRecord, MemoryScope, MemoryType, MemoryUpdate,
    NewMemory, SearchOptions,
};
use serde_json::json;
use tokio_util::sync::CancellationToken;

/// Builds an empty in-memory facade.
async fn facade() -> Arc<LongTermMemory> {
    LongTermMemory::in_memory(MemoryEngineConfig::default())
        .await
        .expect("in-memory facade")
}

/// The scope every test owns its records under unless it asserts isolation.
fn scope() -> MemoryScope {
    MemoryScope::new("alice", "ws")
}

/// A deterministic timestamp `offset` seconds after a fixed epoch.
fn ts(offset: i64) -> DateTime<Utc> {
    DateTime::from_timestamp(1_700_000_000 + offset, 0).expect("fixed timestamp")
}

#[tokio::test]
async fn add_returns_the_created_record_and_get_reads_it() {
    let ltm = facade().await;
    let scope = scope();

    let outcome = ltm
        .add(
            &scope,
            NewMemory::user_profile("alice lives in berlin"),
            &AddOptions::default(),
        )
        .await
        .expect("add");
    assert_eq!(outcome.added.len(), 1);
    assert_eq!(outcome.merged.len(), 0);
    assert_eq!(outcome.deleted.len(), 0);
    assert_eq!(outcome.skipped.len(), 0);

    let created = &outcome.added[0];
    assert_eq!(created.content, "alice lives in berlin");
    assert_eq!(created.kind, MemoryType::UserProfile);
    assert_eq!(created.scope, scope);
    assert_eq!(created.id.len(), 24);
    assert_eq!(created.access_count, 0);

    let stored = ltm
        .get(&scope, &created.id)
        .await
        .expect("get")
        .expect("stored record");
    assert_eq!(&stored, created);

    assert_eq!(
        ltm.get(&scope, "0123456789abcdef01234567")
            .await
            .expect("get unknown id"),
        None
    );
}

#[tokio::test]
async fn add_many_stores_every_memory_in_one_call() {
    let ltm = facade().await;
    let scope = scope();

    let memories = vec![
        NewMemory::semantic("alice prefers dark mode"),
        NewMemory::episodic("alice attended the rust meetup"),
        NewMemory::summary("alice asked about editors"),
    ];
    let outcome = ltm
        .add_many(&scope, memories, &AddOptions::default())
        .await
        .expect("add many");
    assert_eq!(outcome.added.len(), 3);
    assert_eq!(outcome.merged.len(), 0);

    let kinds: HashSet<MemoryType> = outcome.added.iter().map(|record| record.kind).collect();
    assert_eq!(kinds.len(), 3);
    assert!(kinds.contains(&MemoryType::SemanticMemory));
    assert!(kinds.contains(&MemoryType::EpisodicMemory));
    assert!(kinds.contains(&MemoryType::Summary));

    let ids: HashSet<&str> = outcome
        .added
        .iter()
        .map(|record| record.id.as_str())
        .collect();
    assert_eq!(ids.len(), 3, "every stored memory must have its own id");

    let listed = ltm.list(&scope, &[]).await.expect("list");
    assert_eq!(listed.len(), 3);
    let listed_ids: HashSet<&str> = listed.iter().map(|record| record.id.as_str()).collect();
    assert_eq!(listed_ids, ids);
}

#[tokio::test]
async fn update_changes_fields_and_merges_metadata() {
    let ltm = facade().await;
    let scope = scope();

    let outcome = ltm
        .add(
            &scope,
            NewMemory::semantic("alice prefers dark mode")
                .with_metadata("keep", json!("yes"))
                .with_metadata("patched", json!("old")),
            &AddOptions::default(),
        )
        .await
        .expect("add");
    let id = outcome.added[0].id.clone();

    let updated = ltm
        .update(
            &scope,
            &id,
            MemoryUpdate::content("alice prefers light mode")
                .with_importance(0.9)
                .with_metadata("patched", json!("new"))
                .with_metadata("inserted", json!(true)),
            &CallOptions::default(),
        )
        .await
        .expect("update");
    assert_eq!(updated.id, id);
    assert_eq!(updated.content, "alice prefers light mode");
    assert!((updated.importance - 0.9).abs() < 1e-6);
    assert_eq!(updated.metadata.get("keep"), Some(&json!("yes")));
    assert_eq!(updated.metadata.get("patched"), Some(&json!("new")));
    assert_eq!(updated.metadata.get("inserted"), Some(&json!(true)));

    let stored = ltm
        .get(&scope, &id)
        .await
        .expect("get")
        .expect("stored record");
    assert_eq!(stored, updated);

    let error = ltm
        .update(
            &scope,
            "0123456789abcdef01234567",
            MemoryUpdate::content("does not matter"),
            &CallOptions::default(),
        )
        .await
        .expect_err("unknown id");
    assert!(matches!(error, Error::NotFound(_)), "{error:?}");
}

#[tokio::test]
async fn delete_reports_removal_and_drops_the_record() {
    let ltm = facade().await;
    let scope = scope();

    let outcome = ltm
        .add(
            &scope,
            NewMemory::semantic("alice prefers dark mode"),
            &AddOptions::default(),
        )
        .await
        .expect("add");
    let id = outcome.added[0].id.clone();
    assert_eq!(ltm.list(&scope, &[]).await.expect("list").len(), 1);

    assert!(ltm.delete(&scope, &id).await.expect("delete"));
    assert!(!ltm.delete(&scope, &id).await.expect("delete again"));
    assert_eq!(ltm.get(&scope, &id).await.expect("get"), None);
    assert_eq!(ltm.list(&scope, &[]).await.expect("list").len(), 0);
}

#[tokio::test]
async fn clear_removes_only_the_matching_records() {
    let ltm = facade().await;
    let scope = scope();
    let other = MemoryScope::new("bob", "ws");

    for content in [
        "alice prefers dark mode",
        "alice lives in berlin",
        "alice works on compilers",
    ] {
        ltm.add(&scope, NewMemory::semantic(content), &AddOptions::default())
            .await
            .expect("add alice record");
    }
    ltm.add(
        &other,
        NewMemory::semantic("bob prefers light mode"),
        &AddOptions::default(),
    )
    .await
    .expect("add bob record");

    let cleared = ltm
        .clear(&MemoryFilter::for_scope(&scope), &CallOptions::default())
        .await
        .expect("clear scope");
    assert_eq!(cleared, 3);
    assert_eq!(
        ltm.list(&scope, &[]).await.expect("alice list").len(),
        0,
        "every record of the scope must be cleared"
    );
    assert_eq!(
        ltm.list(&other, &[]).await.expect("bob list").len(),
        1,
        "another user's records must survive a scoped clear"
    );

    // An `ids`-restricted filter clears only the named records of the scope.
    let mut ids = Vec::new();
    for content in [
        "alice rides a bicycle",
        "alice reads science fiction",
        "alice plays the cello",
    ] {
        let outcome = ltm
            .add(&scope, NewMemory::semantic(content), &AddOptions::default())
            .await
            .expect("add alice record");
        ids.push(outcome.added[0].id.clone());
    }
    let mut filter = MemoryFilter::for_scope(&scope);
    filter.ids = ids[..2].to_vec();
    assert_eq!(
        ltm.clear(&filter, &CallOptions::default())
            .await
            .expect("clear ids"),
        2
    );
    let remaining = ltm.list(&scope, &[]).await.expect("list");
    assert_eq!(remaining.len(), 1);
    assert_eq!(remaining[0].id, ids[2]);
}

#[tokio::test]
async fn list_filters_by_kind_and_orders_newest_first() {
    let ltm = facade().await;
    let scope = scope();

    for memory in [
        NewMemory::semantic("alice prefers dark mode").with_timestamp(ts(0)),
        NewMemory::summary("alice asked about editors").with_timestamp(ts(20)),
        NewMemory::user_profile("alice lives in berlin").with_timestamp(ts(10)),
    ] {
        ltm.add(&scope, memory, &AddOptions::default())
            .await
            .expect("add");
    }

    let all = ltm.list(&scope, &[]).await.expect("all kinds");
    assert_eq!(all.len(), 3);
    let contents: Vec<&str> = all.iter().map(|record| record.content.as_str()).collect();
    assert_eq!(
        contents,
        vec![
            "alice asked about editors",
            "alice lives in berlin",
            "alice prefers dark mode",
        ],
        "list must be newest first"
    );

    let summaries = ltm
        .list(&scope, &[MemoryType::Summary])
        .await
        .expect("summary list");
    assert_eq!(summaries.len(), 1);
    assert_eq!(summaries[0].kind, MemoryType::Summary);

    let mixed = ltm
        .list(&scope, &[MemoryType::Summary, MemoryType::SemanticMemory])
        .await
        .expect("mixed list");
    assert_eq!(mixed.len(), 2);
    assert!(mixed.iter().all(|record| matches!(
        record.kind,
        MemoryType::Summary | MemoryType::SemanticMemory
    )));
}

#[tokio::test]
async fn stats_report_totals_kinds_and_time_bounds() {
    let ltm = facade().await;
    let scope = scope();

    for memory in [
        NewMemory::semantic("alice prefers dark mode").with_timestamp(ts(0)),
        NewMemory::semantic("alice works on compilers").with_timestamp(ts(20)),
        NewMemory::user_profile("alice lives in berlin").with_timestamp(ts(10)),
    ] {
        ltm.add(&scope, memory, &AddOptions::default())
            .await
            .expect("add");
    }

    let stats = ltm.stats(&scope, &[]).await.expect("stats");
    assert_eq!(stats.total, 3);
    assert_eq!(stats.by_kind.get("semantic_memory"), Some(&2));
    assert_eq!(stats.by_kind.get("user_profile"), Some(&1));
    assert_eq!(stats.oldest, Some(ts(0)));
    assert_eq!(stats.newest, Some(ts(20)));
    assert!(stats.total_tokens > 0);

    let restricted = ltm
        .stats(&scope, &[MemoryType::UserProfile])
        .await
        .expect("profile stats");
    assert_eq!(restricted.total, 1);
    assert_eq!(restricted.by_kind.len(), 1);
    assert_eq!(restricted.by_kind.get("user_profile"), Some(&1));
    assert_eq!(restricted.oldest, Some(ts(10)));
    assert_eq!(restricted.newest, Some(ts(10)));
}

#[tokio::test]
async fn validation_rejects_blank_inputs_and_invalid_scopes() {
    let ltm = facade().await;
    let scope = scope();

    let error = ltm
        .add(&scope, NewMemory::semantic("   "), &AddOptions::default())
        .await
        .expect_err("blank content");
    assert!(matches!(error, Error::Validation(_)), "{error:?}");
    assert_eq!(ltm.stats(&scope, &[]).await.expect("stats").total, 0);

    let error = ltm
        .search(&scope, "   ", &SearchOptions::default())
        .await
        .expect_err("blank query");
    assert!(matches!(error, Error::Validation(_)), "{error:?}");

    let error = ltm
        .add(
            &MemoryScope::new("", "ws"),
            NewMemory::semantic("alice prefers dark mode"),
            &AddOptions::default(),
        )
        .await
        .expect_err("empty user id");
    assert!(matches!(error, Error::Validation(_)), "{error:?}");

    let error = ltm
        .add(
            &MemoryScope::new("alice", "bad/scope"),
            NewMemory::semantic("alice prefers dark mode"),
            &AddOptions::default(),
        )
        .await
        .expect_err("scope id with a slash");
    assert!(matches!(error, Error::Validation(_)), "{error:?}");

    let error = ltm
        .get(
            &MemoryScope::new("alice", "bad/scope"),
            "0123456789abcdef01234567",
        )
        .await
        .expect_err("invalid scope on get");
    assert!(matches!(error, Error::Validation(_)), "{error:?}");
    assert_eq!(ltm.stats(&scope, &[]).await.expect("stats").total, 0);
}

#[test]
fn memory_record_json_round_trip_uses_python_field_names() {
    let record = MemoryRecord::new(
        scope(),
        "alice prefers dark mode",
        MemoryType::SemanticMemory,
    )
    .with_id("0123456789abcdef01234567")
    .with_importance(0.8)
    .with_metadata("lang", json!("en"));

    let value = serde_json::to_value(&record).expect("serialize record");
    assert_eq!(value["text"], json!("alice prefers dark mode"));
    assert_eq!(value["type"], json!("semantic_memory"));
    assert_eq!(value["fields"]["lang"], json!("en"));
    assert_eq!(value["user_id"], json!("alice"));
    assert_eq!(value["scope_id"], json!("ws"));
    assert_eq!(value["id"], json!("0123456789abcdef01234567"));
    assert!(value.get("content").is_none());
    assert!(value.get("kind").is_none());
    assert!(value.get("metadata").is_none());

    let restored: MemoryRecord = serde_json::from_value(value.clone()).expect("deserialize");
    assert_eq!(restored, record);

    // Python alternatives: `mem` for text, `mem_type` for type, `metadata` for fields.
    let aliased: MemoryRecord = serde_json::from_value(json!({
        "id": "fedcba9876543210fedcba98",
        "mem": "bob writes rust",
        "mem_type": "user_profile",
        "metadata": {"team": "core"},
        "user_id": "bob",
        "scope_id": "ws",
    }))
    .expect("deserialize aliases");
    assert_eq!(aliased.content, "bob writes rust");
    assert_eq!(aliased.kind, MemoryType::UserProfile);
    assert_eq!(aliased.metadata.get("team"), Some(&json!("core")));
    assert_eq!(aliased.scope, MemoryScope::new("bob", "ws"));
    assert!((aliased.importance - 0.5).abs() < 1e-6);
    assert_eq!(aliased.access_count, 0);
    assert_eq!(aliased.embedding, None);

    // Unknown extra fields and absent optional fields are tolerated.
    let extended: MemoryRecord = serde_json::from_value(json!({
        "id": "00112233445566778899aabb",
        "text": "alice plays the cello",
        "type": "semantic_memory",
        "user_id": "alice",
        "scope_id": "ws",
        "extra": {"nested": [1, 2, 3]},
        "another_unknown": "ignored",
    }))
    .expect("deserialize unknown fields");
    assert_eq!(extended.content, "alice plays the cello");
    assert_eq!(extended.kind, MemoryType::SemanticMemory);
    assert!(extended.metadata.is_empty());
}

#[tokio::test]
async fn cancelled_calls_fail_without_side_effects() {
    let ltm = facade().await;
    let scope = scope();

    let token = CancellationToken::new();
    token.cancel();
    let add_options = AddOptions {
        call: CallOptions::default().with_cancellation(token),
        ..Default::default()
    };
    let error = ltm
        .add(
            &scope,
            NewMemory::semantic("alice prefers dark mode"),
            &add_options,
        )
        .await
        .expect_err("cancelled add");
    assert!(matches!(error, Error::Cancelled), "{error:?}");
    assert_eq!(ltm.list(&scope, &[]).await.expect("list").len(), 0);
    assert_eq!(ltm.stats(&scope, &[]).await.expect("stats").total, 0);

    ltm.add(
        &scope,
        NewMemory::semantic("alice lives in berlin"),
        &AddOptions::default(),
    )
    .await
    .expect("seed record");

    let token = CancellationToken::new();
    token.cancel();
    let error = ltm
        .clear(
            &MemoryFilter::for_scope(&scope),
            &CallOptions::default().with_cancellation(token),
        )
        .await
        .expect_err("cancelled clear");
    assert!(matches!(error, Error::Cancelled), "{error:?}");
    assert_eq!(
        ltm.stats(&scope, &[]).await.expect("stats").total,
        1,
        "a cancelled clear must not delete anything"
    );
}
