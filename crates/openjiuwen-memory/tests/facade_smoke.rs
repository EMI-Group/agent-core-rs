//! End-to-end smoke tests of the [`openjiuwen_memory::facade::LongTermMemory`] surface.
//!
//! These drive the public facade only: constructors, record CRUD, recall (keyword and
//! semantic), variables, scope configuration, and the `add_messages` history +
//! extraction pipeline with the deterministic `ScriptedModel`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::sync::Arc;

use openjiuwen_foundation::llm::Model;
use openjiuwen_foundation::prompt::BaseMessage;
use openjiuwen_memory::config::{AgentMemoryConfig, MemoryEngineConfig, MemoryScopeConfig};
use openjiuwen_memory::facade::LongTermMemory;
use openjiuwen_memory::record::{
    AddOptions, CallOptions, MemoryFilter, MemoryScope, MemoryType, MemoryUpdate, NewMemory,
    SearchOptions,
};
use openjiuwen_retrieval::embedding::HashEmbedding;
use openjiuwen_test_support::ScriptedModel;

fn scope() -> MemoryScope {
    MemoryScope::new("alice", "ws")
}

#[tokio::test]
async fn in_memory_crud_search_and_stats() {
    let ltm = LongTermMemory::in_memory(MemoryEngineConfig::default())
        .await
        .expect("facade");
    let scope = scope();

    let added = ltm
        .add(
            &scope,
            NewMemory::semantic("alice prefers dark mode for every editor"),
            &AddOptions::default(),
        )
        .await
        .expect("add");
    let id = added.added[0].id.clone();
    let stored = ltm.get(&scope, &id).await.expect("get").expect("present");
    assert_eq!(stored.content, "alice prefers dark mode for every editor");
    assert_eq!(stored.kind, MemoryType::SemanticMemory);

    let hits = ltm
        .search(&scope, "dark mode editor", &SearchOptions::default())
        .await
        .expect("search");
    assert!(!hits.is_empty(), "keyword recall must find the record");
    assert_eq!(hits[0].record.id, id);

    ltm.add(
        &scope,
        NewMemory::user_profile("alice lives in Berlin and works on compilers"),
        &AddOptions::default(),
    )
    .await
    .expect("add profile");
    let stats = ltm.stats(&scope, &[]).await.expect("stats");
    assert_eq!(stats.total, 2);
    assert_eq!(stats.by_kind.get("semantic_memory"), Some(&1));
    assert_eq!(stats.by_kind.get("user_profile"), Some(&1));
    assert!(stats.total_tokens > 0);
    assert!(stats.oldest.is_some() && stats.newest.is_some());

    let updated = ltm
        .update(
            &scope,
            &id,
            MemoryUpdate::content("alice prefers light mode for every editor"),
            &CallOptions::default(),
        )
        .await
        .expect("update");
    assert_eq!(updated.content, "alice prefers light mode for every editor");
    assert_eq!(ltm.list(&scope, &[]).await.expect("list").len(), 2);
    assert_eq!(
        ltm.list(&scope, &[MemoryType::UserProfile])
            .await
            .expect("list profiles")
            .len(),
        1
    );

    assert!(ltm.delete(&scope, &id).await.expect("delete"));
    assert!(!ltm.delete(&scope, &id).await.expect("delete again"));
    assert_eq!(ltm.stats(&scope, &[]).await.expect("stats").total, 1);
    assert_eq!(
        ltm.clear(&MemoryFilter::for_scope(&scope), &CallOptions::default())
            .await
            .expect("clear"),
        1
    );
    assert_eq!(ltm.stats(&scope, &[]).await.expect("stats").total, 0);
}

#[tokio::test]
async fn semantic_recall_with_an_embedding_model() {
    let ltm = LongTermMemory::builder(MemoryEngineConfig::default())
        .embedding(Arc::new(HashEmbedding::new(16)))
        .build()
        .expect("facade");
    let scope = scope();
    let added = ltm
        .add(
            &scope,
            NewMemory::semantic("Alice prefers dark mode"),
            &AddOptions::default(),
        )
        .await
        .expect("add");
    let id = added.added[0].id.clone();
    let hits = ltm
        .search(&scope, "dark mode", &SearchOptions::default())
        .await
        .expect("search");
    assert!(
        hits.iter().any(|hit| hit.record.id == id),
        "semantic recall must return the stored record"
    );
    assert_eq!(hits[0].record.id, id);
    assert!(hits[0].similarity > 0.0);
}

#[tokio::test]
async fn variables_and_scope_config_round_trip() {
    let ltm = LongTermMemory::in_memory(MemoryEngineConfig::default())
        .await
        .expect("facade");
    let scope = scope();
    ltm.set_variable(&scope, "theme", "dark")
        .await
        .expect("set variable");
    ltm.set_variable(&scope, "city", "Berlin")
        .await
        .expect("set variable");
    let variables = ltm.get_variables(&scope, None).await.expect("variables");
    assert_eq!(variables.len(), 2);
    assert_eq!(variables.get("theme").map(String::as_str), Some("dark"));
    assert_eq!(
        ltm.delete_variables(&scope, &["city".to_owned()])
            .await
            .expect("delete variable"),
        1
    );
    assert_eq!(
        ltm.get_variables(&scope, None)
            .await
            .expect("variables")
            .len(),
        1
    );

    let config = MemoryScopeConfig {
        user_profile_definition: Some("identity facts".to_owned()),
        semantic_memory_definition: Some("durable facts".to_owned()),
        episodic_memory_definition: None,
    };
    assert!(!ltm
        .set_scope_config("ws", config.clone())
        .await
        .expect("create scope config"));
    assert_eq!(
        ltm.get_scope_config("ws").await.expect("scope config"),
        Some(config)
    );
    assert!(ltm.delete_scope_config("ws").await.expect("delete config"));
    assert_eq!(
        ltm.get_scope_config("ws").await.expect("scope config"),
        None
    );
}

#[tokio::test]
async fn add_messages_heuristic_path_and_history() {
    let ltm = LongTermMemory::in_memory(MemoryEngineConfig::default())
        .await
        .expect("facade");
    let scope = scope();
    let agent = AgentMemoryConfig::default();

    let user = BaseMessage::user("I always prefer dark mode in every editor");
    let result = ltm
        .add_messages(
            &scope,
            std::slice::from_ref(&user),
            &agent,
            &AddOptions::default(),
        )
        .await
        .expect("add user message");
    assert_eq!(result.episodic_memory.len(), 1);
    assert_eq!(result.episodic_memory[0].content, user.text());

    // A second turn; assertions below are order-independent because messages of one
    // batch may share a timestamp, so history ordering is not a contract.
    let follow_up = BaseMessage::user("Remember that my dark mode preference is permanent");
    let assistant = BaseMessage::assistant("Noted, dark mode it is for your editors");
    let result = ltm
        .add_messages(
            &scope,
            &[follow_up.clone(), assistant.clone()],
            &agent,
            &AddOptions::default(),
        )
        .await
        .expect("add follow-up turn");
    assert_eq!(result.episodic_memory.len(), 2);

    let history = ltm.get_recent_messages(&scope, 10).await.expect("history");
    assert_eq!(history.len(), 3);
    assert!(
        history.iter().any(|message| message.text() == user.text()),
        "history must contain the original user message: {history:?}"
    );
    assert!(
        history.iter().any(|message| message.role.is_assistant()),
        "history must contain an assistant message: {history:?}"
    );
    assert!(
        history
            .iter()
            .any(|message| message.role.is_assistant() && message.text() == assistant.text()),
        "history must contain the assistant reply: {history:?}"
    );

    // The heuristic path also writes one summary per batch of the recent history.
    let summaries = ltm
        .list(&scope, &[MemoryType::Summary])
        .await
        .expect("summaries");
    assert_eq!(summaries.len(), 2);
}

#[tokio::test]
async fn add_messages_with_a_scripted_model_extracts_facts_and_variables() {
    let payload = r#"{"facts":[
        {"content":"Alice prefers dark mode","type":"user_profile","importance":0.9},
        {"content":"Alice writes Rust","type":"semantic_memory","importance":0.6}
    ],"variables":{"theme":"dark"}}"#;
    let scripted = Arc::new(ScriptedModel::text(payload));
    let ltm = LongTermMemory::builder(MemoryEngineConfig::default())
        .model(scripted.clone())
        .build()
        .expect("facade");
    let scope = scope();
    let agent = AgentMemoryConfig {
        mem_variables: vec!["theme".to_owned()],
        ..Default::default()
    };
    let messages = vec![
        BaseMessage::user("I always prefer dark mode in every editor"),
        BaseMessage::assistant("Understood: dark mode everywhere"),
    ];
    let result = ltm
        .add_messages(&scope, &messages, &agent, &AddOptions::default())
        .await
        .expect("add messages");
    assert_eq!(scripted.call_count(), 1);
    assert_eq!(result.user_profile.len(), 1);
    assert_eq!(result.user_profile[0].content, "Alice prefers dark mode");
    assert!((result.user_profile[0].importance - 0.9).abs() < 1e-6);
    assert_eq!(result.semantic_memory.len(), 1);
    assert_eq!(result.semantic_memory[0].content, "Alice writes Rust");
    assert_eq!(result.summary.len(), 0);
    assert_eq!(
        result.variables.get("theme").map(String::as_str),
        Some("dark")
    );

    let variables = ltm
        .get_variables(&scope, Some(&["theme".to_owned()]))
        .await
        .expect("variables");
    assert_eq!(variables.get("theme").map(String::as_str), Some("dark"));
    assert_eq!(
        ltm.list(&scope, &[MemoryType::UserProfile])
            .await
            .expect("profiles")
            .len(),
        1
    );
    assert_eq!(
        ltm.get_recent_messages(&scope, 10)
            .await
            .expect("history")
            .len(),
        2
    );
    assert_eq!(Model::model_name(scripted.as_ref()), "scripted-model");
}
