//! Integration coverage for the team-memory *runtime adapters*: the caller-driven
//! lifecycle hooks, the model-facing member toolkit, the shared `TEAM_MEMORY.md`
//! document, the rule/LLM extractors, and the observer seam.
//!
//! The facade itself (CRUD, filters, ranking, SQLite persistence) is covered by
//! `team_memory.rs`; this suite drives what a team runtime wires around it:
//! `openjiuwen_memory::team::{hooks, toolkit, document, extractor, observe}`.
//!
//! Everything runs offline and deterministically: the in-memory backend, fixtures pinned
//! to the shared test epoch (`1_700_000_000`), and `ScriptedModel` wherever a model would
//! be called, so no assertion depends on the wall clock or the network.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::llm::Model;
use openjiuwen_memory::record::CallOptions;
use openjiuwen_memory::team::{
    document_team_id, truncate_for_prompt, LlmTeamExtractor, MemberMemoryContext,
    MemberMemoryToolkit, NewTeamMemory, NoopObserver, RuleTeamExtractor, TeamExtractConfig,
    TeamExtraction, TeamMemory, TeamMemoryDocument, TeamMemoryEvent, TeamMemoryExtractor,
    TeamMemoryFilter, TeamMemoryHookConfig, TeamMemoryHooks, TeamMemoryKind, TeamMemoryObserver,
    TeamMemoryOperation, TeamMemoryOutcome, TeamMemoryRecord, TeamMemoryScope, TeamSearchOptions,
    TeamTranscript, TeamTurn, TeamVisibility, TracingObserver, LLM_SOURCE, RULE_SOURCE,
    TEAM_MEMORY_FILE, TEAM_MEMORY_FORMAT, TEAM_MEMORY_MAX_READ_LINES, TOOL_FORGET, TOOL_REMEMBER,
    TOOL_SEARCH, TOOL_STATS,
};
use openjiuwen_test_support::{ScriptedError, ScriptedModel};
use serde_json::{json, Value};
use tokio_util::sync::CancellationToken;

/// The team every fixture writes to.
const TEAM: &str = "review-team";

/// A fixed instant `offset` seconds after the shared test epoch (2023-11-14T22:13:20Z).
fn at(offset: i64) -> DateTime<Utc> {
    DateTime::from_timestamp(1_700_000_000 + offset, 0).expect("a valid fixed timestamp")
}

/// Pins both timestamps of `record`, so ordering and rendering never depend on "now".
fn pin(mut record: TeamMemoryRecord, created: i64, updated: i64) -> TeamMemoryRecord {
    record.created_at = at(created);
    record.updated_at = at(updated);
    record
}

/// One buffered team turn, timestamped at the shared epoch.
fn turn(from_member: &str, content: &str) -> TeamTurn {
    TeamTurn::new(from_member, content, at(0))
}

/// A turn the rule extractor classifies as a decision.
fn decision_turn() -> TeamTurn {
    turn("alice", "We decided to use SQLite for the shared store.")
}

/// A turn the rule extractor classifies as a lesson.
fn lesson_turn() -> TeamTurn {
    turn("bob", "Next time we should run the migration first")
}

/// Rule-extractor hooks over a fresh in-memory facade.
fn rule_hooks(config: TeamMemoryHookConfig) -> (Arc<TeamMemory>, TeamMemoryHooks) {
    let memory = Arc::new(TeamMemory::in_memory());
    let hooks = TeamMemoryHooks::with_rule_extractor(TEAM, Arc::clone(&memory), config);
    (memory, hooks)
}

/// Every record of the test team, as the store holds it.
async fn stored(memory: &TeamMemory) -> Vec<TeamMemoryRecord> {
    memory
        .list(&TeamMemoryFilter::for_team(TEAM))
        .await
        .expect("the team lists")
}

/// Indexes records by id so two record sets can be compared order-insensitively.
fn by_id(records: Vec<TeamMemoryRecord>) -> BTreeMap<String, TeamMemoryRecord> {
    records
        .into_iter()
        .map(|record| (record.id.clone(), record))
        .collect()
}

/// A pre-triggered cancellation token wrapped in call options.
fn cancelled_call() -> CallOptions {
    let token = CancellationToken::new();
    token.cancel();
    CallOptions::new().with_cancellation(token)
}

/// An extractor that always fails, standing in for an unavailable extraction pipeline.
struct FailingExtractor;

#[async_trait]
impl TeamMemoryExtractor for FailingExtractor {
    #[allow(clippy::unnecessary_literal_bound)] // the trait signature ties `&str` to `self`
    fn name(&self) -> &str {
        "failing"
    }

    async fn extract(
        &self,
        _transcript: &TeamTranscript,
        options: &CallOptions,
    ) -> Result<TeamExtraction> {
        options.check()?;
        Err(Error::memory("the extraction model is unavailable"))
    }
}

/// Collects every event an attached facade emits; clones share one buffer.
#[derive(Clone, Default)]
struct RecordingObserver {
    events: Arc<Mutex<Vec<TeamMemoryEvent>>>,
}

impl RecordingObserver {
    /// Drains the recorded events in emission order.
    fn take(&self) -> Vec<TeamMemoryEvent> {
        std::mem::take(&mut *self.events.lock().expect("the observer lock"))
    }
}

impl TeamMemoryObserver for RecordingObserver {
    fn observe(&self, event: &TeamMemoryEvent) {
        self.events
            .lock()
            .expect("the observer lock")
            .push(event.clone());
    }
}

// ---------------------------------------------------------------------------
// Scenario 1: lifecycle hooks write, roll back on failure, and stay idempotent.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn hooks_lifecycle_writes_profiles_and_extracted_records() {
    let (memory, hooks) = rule_hooks(TeamMemoryHookConfig::default());
    let call = CallOptions::new();
    hooks.on_session_start("s-1", &call).await.unwrap();
    let profile = hooks
        .on_member_join("alice", &call)
        .await
        .unwrap()
        .expect("a profile record");
    assert_eq!(profile.kind, TeamMemoryKind::MemberProfile);
    hooks.on_turn(decision_turn(), &call).await.unwrap();
    hooks.on_turn(lesson_turn(), &call).await.unwrap();
    assert_eq!(hooks.pending_turns(), 2, "turns are buffered, not written");
    assert_eq!(stored(&memory).await.len(), 1);

    let flush = hooks.on_session_end(&call).await.unwrap();
    assert_eq!(flush.turns, 2);
    assert_eq!(flush.len(), 2);
    assert!(flush.degraded.is_none());
    assert_eq!(hooks.pending_turns(), 0);
    let kinds: Vec<TeamMemoryKind> = flush
        .records
        .iter()
        .map(|record| record.kind.clone())
        .collect();
    assert_eq!(
        kinds,
        vec![TeamMemoryKind::Decision, TeamMemoryKind::HandoffNote]
    );
    for record in &flush.records {
        let saved = memory
            .get(TEAM, &record.id)
            .await
            .unwrap()
            .expect("the flushed record is in the store");
        assert_eq!(&saved, record);
        assert_eq!(record.session_id.as_deref(), Some("s-1"));
    }
    assert_eq!(stored(&memory).await.len(), 3);
    assert_eq!(hooks.members(), vec!["alice".to_owned()]);
}

#[tokio::test]
async fn a_failing_extractor_leaves_the_store_untouched() {
    let memory = Arc::new(TeamMemory::in_memory());
    let call = CallOptions::new();
    memory
        .remember(
            NewTeamMemory::decision(TEAM, "Use SQLite for the shared store."),
            &call,
        )
        .await
        .unwrap();
    let hooks = TeamMemoryHooks::new(
        TEAM,
        Arc::clone(&memory),
        Arc::new(FailingExtractor),
        TeamMemoryHookConfig::default(),
    );
    assert_eq!(hooks.extractor_name(), "failing");
    hooks.on_session_start("s-2", &call).await.unwrap();
    hooks.sync_turn(decision_turn(), &call).await.unwrap();
    let error = hooks.on_session_end(&call).await.unwrap_err();
    assert!(matches!(error, Error::Memory(_)), "{error}");
    assert_eq!(stored(&memory).await.len(), 1, "no partial write happened");
    assert_eq!(hooks.pending_turns(), 1, "the buffer survives the failure");

    let retry = hooks.on_session_end(&call).await.unwrap_err();
    assert!(matches!(retry, Error::Memory(_)), "{retry}");
    assert_eq!(stored(&memory).await.len(), 1);
}

#[tokio::test]
async fn a_repeated_session_end_flush_is_idempotent() {
    let (memory, hooks) = rule_hooks(TeamMemoryHookConfig::default());
    let call = CallOptions::new();
    hooks.on_session_start("s-1", &call).await.unwrap();
    hooks.on_turn(decision_turn(), &call).await.unwrap();
    let first = hooks.on_session_end(&call).await.unwrap();
    assert_eq!(first.len(), 1);
    let second = hooks.on_session_end(&call).await.unwrap();
    assert!(second.is_empty(), "a drained buffer flushes nothing");
    assert_eq!(second.turns, 0);

    hooks.on_turn(decision_turn(), &call).await.unwrap();
    let third = hooks.on_session_end(&call).await.unwrap();
    assert_eq!(third.len(), 1);
    assert_eq!(
        third.records[0].id, first.records[0].id,
        "the id is derived from team/session/kind/content"
    );
    assert_eq!(
        stored(&memory).await.len(),
        1,
        "the retry upserted, not added"
    );
}

// ---------------------------------------------------------------------------
// Scenario 2: bounds, switches, invalid configuration, and cancellation.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn the_turn_buffer_drops_the_oldest_turns_past_the_cap() {
    let config = TeamMemoryHookConfig {
        max_pending_turns: 2,
        ..Default::default()
    };
    let (memory, hooks) = rule_hooks(config);
    let call = CallOptions::new();
    hooks.on_session_start("s-1", &call).await.unwrap();
    hooks.on_turn(decision_turn(), &call).await.unwrap();
    hooks
        .on_turn(
            turn("bob", "We decided to drop the cache layer first."),
            &call,
        )
        .await
        .unwrap();
    hooks
        .on_turn(
            turn("carol", "We chose to publish the migration script today."),
            &call,
        )
        .await
        .unwrap();
    assert_eq!(hooks.pending_turns(), 2, "the oldest turn was dropped");
    assert_eq!(stored(&memory).await.len(), 0);

    let flush = hooks.on_session_end(&call).await.unwrap();
    assert_eq!(flush.turns, 2);
    assert_eq!(flush.len(), 2);
    assert!(
        flush
            .records
            .iter()
            .all(|record| !record.content.contains("SQLite")),
        "the dropped turn is not extracted"
    );
    assert_eq!(stored(&memory).await.len(), 2);
}

#[tokio::test]
async fn a_disabled_hook_config_writes_nothing_and_keeps_the_buffer() {
    let config = TeamMemoryHookConfig {
        enabled: false,
        ..Default::default()
    };
    let (memory, hooks) = rule_hooks(config);
    let call = CallOptions::new();
    hooks.on_session_start("s-1", &call).await.unwrap();
    assert_eq!(hooks.on_member_join("alice", &call).await.unwrap(), None);
    assert_eq!(hooks.members(), vec!["alice".to_owned()]);
    hooks.on_turn(decision_turn(), &call).await.unwrap();
    assert_eq!(hooks.pending_turns(), 1);
    let flush = hooks.on_session_end(&call).await.unwrap();
    assert!(flush.is_empty());
    assert_eq!(
        hooks.pending_turns(),
        1,
        "a disabled flush keeps the buffer for a later manual call"
    );
    assert!(hooks
        .on_member_leave("alice", &call)
        .await
        .unwrap()
        .is_none());
    assert_eq!(hooks.members().len(), 0);
    assert_eq!(stored(&memory).await.len(), 0);
}

#[tokio::test]
async fn an_invalid_hook_config_is_rejected_before_writing() {
    assert!(TeamMemoryHookConfig::default().validate().is_ok());

    let (memory, hooks) = rule_hooks(TeamMemoryHookConfig {
        handoff_importance: 1.5,
        ..Default::default()
    });
    let call = CallOptions::new();
    let error = hooks.on_member_join("alice", &call).await.unwrap_err();
    assert!(matches!(error, Error::Validation(_)), "{error}");
    assert!(error.to_string().contains("handoff_importance"), "{error}");
    assert_eq!(stored(&memory).await.len(), 0);

    let (memory, hooks) = rule_hooks(TeamMemoryHookConfig {
        max_pending_turns: 0,
        ..Default::default()
    });
    hooks.on_turn(decision_turn(), &call).await.unwrap();
    let error = hooks.on_session_end(&call).await.unwrap_err();
    assert!(matches!(error, Error::Validation(_)), "{error}");
    assert!(error.to_string().contains("max_pending_turns"), "{error}");
    assert_eq!(stored(&memory).await.len(), 0);
    assert_eq!(hooks.pending_turns(), 1);
}

#[tokio::test]
async fn a_cancelled_hook_call_writes_nothing() {
    let (memory, hooks) = rule_hooks(TeamMemoryHookConfig::default());
    let call = cancelled_call();
    let error = hooks.on_session_start("s-1", &call).await.unwrap_err();
    assert!(matches!(error, Error::Cancelled), "{error}");
    assert!(hooks.session_id().is_none());
    let error = hooks.on_member_join("alice", &call).await.unwrap_err();
    assert!(matches!(error, Error::Cancelled), "{error}");
    let error = hooks.on_turn(decision_turn(), &call).await.unwrap_err();
    assert!(matches!(error, Error::Cancelled), "{error}");
    assert_eq!(hooks.pending_turns(), 0, "a cancelled turn is not buffered");
    let error = hooks.on_session_end(&call).await.unwrap_err();
    assert!(matches!(error, Error::Cancelled), "{error}");
    let error = hooks.on_member_leave("alice", &call).await.unwrap_err();
    assert!(matches!(error, Error::Cancelled), "{error}");
    assert_eq!(hooks.members().len(), 0);
    assert_eq!(stored(&memory).await.len(), 0);
}

// ---------------------------------------------------------------------------
// Scenario 3: the member toolkit's schemas, dispatcher, and privacy rules.
// ---------------------------------------------------------------------------

#[test]
fn toolkit_schemas_expose_the_four_documented_tools() {
    let memory = Arc::new(TeamMemory::in_memory());
    let toolkit = MemberMemoryToolkit::new(
        memory.clone(),
        MemberMemoryContext::new(TEAM).with_member("alice"),
    );
    let tools = toolkit.tool_names();
    assert_eq!(
        tools,
        vec![TOOL_REMEMBER, TOOL_SEARCH, TOOL_FORGET, TOOL_STATS]
    );
    let schemas = toolkit.get_tool_schemas();
    assert_eq!(schemas.len(), 4);
    for (schema, name) in schemas.iter().zip(tools.iter().copied()) {
        assert_eq!(schema["name"], json!(name));
        assert_eq!(
            schema["id"],
            json!(format!("team_memory.{TEAM}.alice.{name}"))
        );
        assert_eq!(schema["input_schema"]["type"], json!("object"));
        assert!(
            schema["description"]
                .as_str()
                .is_some_and(|text| !text.is_empty()),
            "`{name}` needs a description"
        );
        for (field, property) in schema["input_schema"]["properties"]
            .as_object()
            .expect("a properties object")
        {
            assert!(
                property["description"]
                    .as_str()
                    .is_some_and(|text| !text.is_empty()),
                "property `{field}` of `{name}` needs a description"
            );
        }
    }
    assert_eq!(schemas[0]["input_schema"]["required"], json!(["content"]));
    assert_eq!(schemas[1]["input_schema"]["required"], json!(["query"]));
    assert_eq!(schemas[2]["input_schema"]["required"], json!(["id"]));
    assert_eq!(schemas[3]["input_schema"]["required"], json!([]));
    let team_wide = MemberMemoryToolkit::new(memory, MemberMemoryContext::new(TEAM));
    assert_eq!(
        team_wide.get_tool_schemas()[0]["id"],
        json!(format!("team_memory.{TEAM}.team.{TOOL_REMEMBER}"))
    );
}

#[tokio::test]
async fn toolkit_remember_writes_a_record_visible_through_the_facade() {
    let memory = Arc::new(TeamMemory::in_memory());
    let context = MemberMemoryContext::new(TEAM)
        .with_member("alice")
        .with_session("s-1");
    let toolkit = MemberMemoryToolkit::new(Arc::clone(&memory), context);
    let envelope = toolkit
        .handle_tool_call(
            TOOL_REMEMBER,
            json!({
                "content": "Use SQLite for the shared deployment store.",
                "title": "Storage choice",
                "kind": "decision",
                "tags": ["storage"],
                "importance": 0.8,
            }),
        )
        .await
        .unwrap();
    assert_eq!(envelope["ok"], json!(true));
    assert_eq!(envelope["tool"], json!(TOOL_REMEMBER));
    let record: TeamMemoryRecord =
        serde_json::from_value(envelope["result"].clone()).expect("the stored record");
    assert_eq!(record.team_id, TEAM);
    assert_eq!(record.member_id.as_deref(), Some("alice"));
    assert_eq!(record.session_id.as_deref(), Some("s-1"));
    assert_eq!(record.kind, TeamMemoryKind::Decision);
    assert_eq!(record.title.as_deref(), Some("Storage choice"));
    assert_eq!(record.tags, vec!["storage".to_owned()]);
    assert!((record.importance - 0.8).abs() < f32::EPSILON);
    assert_eq!(record.visibility, TeamVisibility::Team);
    let saved = memory
        .get(TEAM, &record.id)
        .await
        .unwrap()
        .expect("the toolkit wrote through the facade");
    assert_eq!(saved, record);
}

#[tokio::test]
async fn toolkit_search_stats_and_forget_see_the_same_record() {
    let memory = Arc::new(TeamMemory::in_memory());
    let toolkit = MemberMemoryToolkit::new(
        Arc::clone(&memory),
        MemberMemoryContext::new(TEAM).with_member("alice"),
    );
    let envelope = toolkit
        .handle_tool_call(
            TOOL_REMEMBER,
            json!({"content": "Use SQLite for the shared deployment store.", "kind": "decision"}),
        )
        .await
        .unwrap();
    let record: TeamMemoryRecord =
        serde_json::from_value(envelope["result"].clone()).expect("the stored record");

    let found = toolkit
        .handle_tool_call(TOOL_SEARCH, json!({"query": "sqlite deployment store"}))
        .await
        .unwrap();
    assert_eq!(found["ok"], json!(true));
    assert_eq!(found["result"]["count"], json!(1));
    assert_eq!(found["result"]["results"][0]["id"], json!(record.id));

    let stats = toolkit
        .handle_tool_call(TOOL_STATS, json!({}))
        .await
        .unwrap();
    assert_eq!(stats["result"]["total"], json!(1));
    assert_eq!(stats["result"]["by_kind"]["decision"], json!(1));
    assert_eq!(stats["result"]["by_member"]["alice"], json!(1));

    let forgotten = toolkit
        .handle_tool_call(TOOL_FORGET, json!({"id": record.id}))
        .await
        .unwrap();
    assert_eq!(forgotten["result"]["deleted"], json!(true));
    assert_eq!(forgotten["result"]["id"], json!(record.id));
    assert!(memory.get(TEAM, &record.id).await.unwrap().is_none());
    let empty = toolkit
        .handle_tool_call(TOOL_STATS, json!({}))
        .await
        .unwrap();
    assert_eq!(empty["result"]["total"], json!(0));
    let absent = toolkit
        .handle_tool_call(TOOL_FORGET, json!({"id": "no-such-record"}))
        .await
        .unwrap();
    assert_eq!(absent["result"]["deleted"], json!(false));
}

#[tokio::test]
async fn toolkit_argument_errors_render_a_failure_envelope_and_write_nothing() {
    let memory = Arc::new(TeamMemory::in_memory());
    let toolkit = MemberMemoryToolkit::new(
        Arc::clone(&memory),
        MemberMemoryContext::new(TEAM).with_member("alice"),
    );
    let cases: [(&str, Value, &str); 10] = [
        (
            TOOL_REMEMBER,
            json!({"content": "   "}),
            "must not be blank",
        ),
        (TOOL_REMEMBER, json!({}), "is required"),
        (
            TOOL_REMEMBER,
            json!({"content": "Release freeze on Friday.", "importance": 1.5}),
            "0.0..=1.0",
        ),
        (
            TOOL_REMEMBER,
            json!({"content": "Release freeze on Friday.", "visibility": "everyone"}),
            "visibility",
        ),
        (
            TOOL_REMEMBER,
            json!({"content": "Release freeze on Friday.", "bogus": 1}),
            "does not accept",
        ),
        (TOOL_SEARCH, json!({"query": "  "}), "must not be blank"),
        (
            TOOL_SEARCH,
            json!({"query": "release", "max_results": 0}),
            "1..=100",
        ),
        (TOOL_FORGET, json!({}), "is required"),
        (TOOL_STATS, json!({"nope": 1}), "does not accept"),
        ("team_memory_bogus", json!({}), "unknown team memory tool"),
    ];
    for (tool, args, needle) in cases {
        let error = toolkit
            .handle_tool_call(tool, args)
            .await
            .expect_err("the arguments are rejected");
        assert!(matches!(error, Error::Validation(_)), "{tool}: {error}");
        assert!(error.to_string().contains(needle), "{tool}: {error}");
        let envelope = MemberMemoryToolkit::error_envelope(tool, &error);
        assert_eq!(envelope["ok"], json!(false));
        assert_eq!(envelope["tool"], json!(tool));
        assert_eq!(envelope["error"]["kind"], json!("validation"));
        assert!(envelope["error"]["message"]
            .as_str()
            .is_some_and(|message| !message.is_empty()));
    }
    assert!(stored(&memory).await.is_empty(), "nothing was written");
}

#[tokio::test]
async fn an_unknown_kind_is_preserved_as_a_custom_kind() {
    let memory = Arc::new(TeamMemory::in_memory());
    let toolkit = MemberMemoryToolkit::new(
        Arc::clone(&memory),
        MemberMemoryContext::new(TEAM).with_member("alice"),
    );
    let envelope = toolkit
        .handle_tool_call(
            TOOL_REMEMBER,
            json!({"content": "Release freeze on Friday.", "kind": "release_rule"}),
        )
        .await
        .unwrap();
    let record: TeamMemoryRecord =
        serde_json::from_value(envelope["result"].clone()).expect("the stored record");
    assert_eq!(
        record.kind,
        TeamMemoryKind::Custom("release_rule".to_owned()),
        "kind parsing is total: an unknown kind is a custom kind, not an argument error"
    );
    assert_eq!(stored(&memory).await.len(), 1);
}

#[tokio::test]
async fn a_read_only_toolkit_context_refuses_writes_but_still_searches() {
    let memory = Arc::new(TeamMemory::in_memory());
    let writer = MemberMemoryToolkit::new(
        Arc::clone(&memory),
        MemberMemoryContext::new(TEAM).with_member("alice"),
    );
    let envelope = writer
        .handle_tool_call(
            TOOL_REMEMBER,
            json!({"content": "Use SQLite for the shared deployment store."}),
        )
        .await
        .unwrap();
    let record: TeamMemoryRecord =
        serde_json::from_value(envelope["result"].clone()).expect("the stored record");

    let reader = MemberMemoryToolkit::new(
        Arc::clone(&memory),
        MemberMemoryContext::new(TEAM)
            .with_member("alice")
            .read_only(),
    );
    for (tool, args) in [
        (
            TOOL_REMEMBER,
            json!({"content": "A read-only member may not write this."}),
        ),
        (TOOL_FORGET, json!({"id": "no-such-record"})),
    ] {
        let error = reader
            .handle_tool_call(tool, args)
            .await
            .expect_err("a read-only context refuses writes");
        assert!(matches!(error, Error::Unsupported(_)), "{tool}: {error}");
        let envelope = MemberMemoryToolkit::error_envelope(tool, &error);
        assert_eq!(envelope["ok"], json!(false));
        assert_eq!(envelope["error"]["kind"], json!("unsupported"));
    }
    let found = reader
        .handle_tool_call(TOOL_SEARCH, json!({"query": "sqlite deployment"}))
        .await
        .unwrap();
    assert_eq!(found["result"]["count"], json!(1));
    assert_eq!(found["result"]["results"][0]["id"], json!(record.id));
    assert_eq!(stored(&memory).await.len(), 1, "the refusals wrote nothing");
}

#[tokio::test]
async fn toolkit_search_hides_another_members_private_record() {
    let memory = Arc::new(TeamMemory::in_memory());
    let bob = MemberMemoryToolkit::new(
        Arc::clone(&memory),
        MemberMemoryContext::new(TEAM).with_member("bob"),
    );
    let envelope = bob
        .handle_tool_call(
            TOOL_REMEMBER,
            json!({
                "content": "Bob keeps the deployment credentials in the vault.",
                "visibility": "private",
            }),
        )
        .await
        .unwrap();
    let private: TeamMemoryRecord =
        serde_json::from_value(envelope["result"].clone()).expect("the stored record");
    assert_eq!(private.visibility, TeamVisibility::Private);

    let bob_hits = bob
        .handle_tool_call(
            TOOL_SEARCH,
            json!({"query": "deployment credentials vault"}),
        )
        .await
        .unwrap();
    assert_eq!(bob_hits["result"]["count"], json!(1), "the author reads it");

    let alice = MemberMemoryToolkit::new(
        Arc::clone(&memory),
        MemberMemoryContext::new(TEAM).with_member("alice"),
    );
    let alice_hits = alice
        .handle_tool_call(
            TOOL_SEARCH,
            json!({"query": "deployment credentials vault"}),
        )
        .await
        .unwrap();
    assert_eq!(alice_hits["result"]["count"], json!(0));
    assert_eq!(
        alice.handle_tool_call(TOOL_STATS, json!({})).await.unwrap()["result"]["total"],
        json!(0)
    );
    let error = alice
        .handle_tool_call(TOOL_FORGET, json!({"id": private.id}))
        .await
        .expect_err("another member's private record is not deletable");
    assert!(matches!(error, Error::NotFound(_)), "{error}");
    assert_eq!(stored(&memory).await.len(), 1, "the record still exists");
}

// ---------------------------------------------------------------------------
// Scenario 4: the `TEAM_MEMORY.md` document.
// ---------------------------------------------------------------------------

/// The records the document round trip seeds: every builtin kind plus a caller-defined
/// kind, all three visibilities, and the content shapes the format must preserve.
fn markdown_records() -> Vec<TeamMemoryRecord> {
    let mut decision = pin(
        TeamMemoryRecord::new(
            &TeamMemoryScope::member(TEAM, "alice"),
            "使用 SQLite 作为共享存储。\n## 权衡\n# 不是 PostgreSQL\nfast enough",
            TeamMemoryKind::Decision,
        )
        .with_id("018f1d2e3f4a5b6c7d8e9001")
        .with_title("Storage choice")
        .with_importance(0.8)
        .with_tags(["storage", "sqlite"])
        .with_metadata("issue", json!(42))
        .with_source("leader-extraction")
        .with_visibility(TeamVisibility::members(["alice", "bob"]))
        .with_expires_at(at(100_000_000)),
        0,
        300,
    );
    decision.session_id = Some("s-1".to_owned());
    let profile = pin(
        TeamMemoryRecord::new(
            &TeamMemoryScope::member(TEAM, "bob"),
            "Bob owns the storage layer and its migrations.",
            TeamMemoryKind::MemberProfile,
        )
        .with_id("018f1d2e3f4a5b6c7d8e9002")
        .with_title("Backend owner")
        .with_visibility(TeamVisibility::Private),
        10,
        200,
    );
    let lesson = pin(
        TeamMemoryRecord::new(
            &TeamMemoryScope::new(TEAM),
            "Align interfaces before parallel work.",
            TeamMemoryKind::HandoffNote,
        )
        .with_id("018f1d2e3f4a5b6c7d8e9003")
        .with_importance(0.65),
        20,
        100,
    );
    let fact = pin(
        TeamMemoryRecord::new(
            &TeamMemoryScope::new(TEAM),
            "  indented first line\n\\already escaped\nsecond line",
            TeamMemoryKind::SharedFact,
        )
        .with_id("018f1d2e3f4a5b6c7d8e9004"),
        30,
        50,
    );
    let task = pin(
        TeamMemoryRecord::new(
            &TeamMemoryScope::new(TEAM),
            "Run the SQLite migration before the release.",
            TeamMemoryKind::Task,
        )
        .with_id("018f1d2e3f4a5b6c7d8e9005")
        .with_title("Wire the migration")
        .with_importance(0.9)
        .with_tags(["release"]),
        40,
        400,
    );
    let artifact = pin(
        TeamMemoryRecord::new(
            &TeamMemoryScope::new(TEAM),
            "Design doc for the shared store.",
            TeamMemoryKind::Artifact,
        )
        .with_id("018f1d2e3f4a5b6c7d8e9006")
        .with_title("docs/design.md"),
        50,
        250,
    );
    let custom = pin(
        TeamMemoryRecord::new(
            &TeamMemoryScope::new(TEAM),
            "Two approvals for schema changes.",
            TeamMemoryKind::Custom("review_rule".to_owned()),
        )
        .with_id("018f1d2e3f4a5b6c7d8e9007")
        .with_title("Review rule")
        .with_metadata("threshold", json!(2)),
        60,
        50,
    );
    vec![decision, profile, lesson, fact, task, artifact, custom]
}

/// Seeds `markdown_records()` through the store seam and renders the team document.
async fn exported_markdown(memory: &TeamMemory) -> String {
    for record in markdown_records() {
        memory.store().upsert(&record).await.expect("a seed record");
    }
    memory
        .export_markdown(&TeamMemoryFilter::for_team(TEAM))
        .await
        .expect("the document renders")
}

#[tokio::test]
async fn team_memory_markdown_round_trips_every_kind_and_visibility() {
    let memory = TeamMemory::in_memory();
    let seeded = markdown_records();
    let markdown = exported_markdown(&memory).await;
    assert!(markdown.starts_with(&format!(
        "---\nteam_memory: {TEAM_MEMORY_FORMAT}\nteam_id: {TEAM}\n"
    )));
    assert!(markdown.contains("entries: 7"), "{markdown}");
    assert!(markdown.contains(&format!("# Team memory — {TEAM}")));
    for section in ["decision", "member", "lesson", "context", "custom"] {
        assert!(
            markdown.contains(&format!("\n## {section}\n")),
            "the `{section}` section is missing"
        );
    }
    assert!(markdown.contains("### [decision] Storage choice"));
    assert!(markdown.contains("<!-- team-memory: {"));
    assert!(markdown.contains("*id: 018f1d2e3f4a5b6c7d8e9001"));

    let parsed = TeamMemoryDocument::parse(&markdown, None).expect("the document parses");
    assert_eq!(parsed.team_id, TEAM);
    assert_eq!(parsed.len(), 7);
    assert_eq!(by_id(parsed.entries), by_id(seeded));
}

#[tokio::test]
async fn importing_the_same_markdown_twice_writes_each_record_once() {
    let source = TeamMemory::in_memory();
    let markdown = exported_markdown(&source).await;
    let memory = TeamMemory::in_memory();
    let call = CallOptions::new();
    let first = memory.import_markdown(&markdown, &call).await.unwrap();
    assert_eq!(first.len(), 7);
    let second = memory.import_markdown(&markdown, &call).await.unwrap();
    assert_eq!(stored(&memory).await.len(), 7, "import is idempotent");
    let first_ids: Vec<&str> = first.iter().map(|record| record.id.as_str()).collect();
    let second_ids: Vec<&str> = second.iter().map(|record| record.id.as_str()).collect();
    assert_eq!(first_ids, second_ids, "the ids come from the payload");
    let seeded = by_id(markdown_records());
    for record in &second {
        assert_eq!(Some(record), seeded.get(&record.id), "{}", record.id);
    }
    let cancelled = memory.import_markdown(&markdown, &cancelled_call()).await;
    assert!(
        matches!(cancelled, Err(Error::Cancelled)),
        "nothing written"
    );
    assert_eq!(stored(&memory).await.len(), 7);
}

#[tokio::test]
async fn document_team_id_and_truncate_for_prompt_follow_the_documented_rules() {
    let memory = TeamMemory::in_memory();
    let markdown = exported_markdown(&memory).await;
    assert_eq!(document_team_id(&markdown).as_deref(), Some(TEAM));
    assert_eq!(
        document_team_id("### [decision] A Python-authored entry"),
        None
    );
    assert_eq!(truncate_for_prompt(&markdown), markdown.trim());

    let long = (1..=2 * TEAM_MEMORY_MAX_READ_LINES)
        .map(|line| format!("line {line}"))
        .collect::<Vec<String>>()
        .join("\n");
    let truncated = truncate_for_prompt(&long);
    assert_eq!(truncated.lines().count(), TEAM_MEMORY_MAX_READ_LINES);
    assert_eq!(truncated.lines().next(), Some("line 1"));
    let last = format!("line {TEAM_MEMORY_MAX_READ_LINES}");
    assert_eq!(truncated.lines().last(), Some(last.as_str()));
}

#[tokio::test]
async fn a_python_authored_document_parses_into_derived_records() {
    let python_authored = "### [decision] 选择方案 A\n原因是更简单。\n\
### [lesson] 并行任务需要先对齐接口\n上次因为没对齐导致返工 2 天。\n\
### [context] Constraint: Python 3.11\nThe team may not upgrade the runtime.\n";
    let document = TeamMemoryDocument::parse(python_authored, Some(TEAM)).expect("the file parses");
    assert_eq!(document.team_id, TEAM);
    assert_eq!(document.len(), 3);
    let kinds: Vec<TeamMemoryKind> = document
        .entries
        .iter()
        .map(|entry| entry.kind.clone())
        .collect();
    assert_eq!(
        kinds,
        vec![
            TeamMemoryKind::Decision,
            TeamMemoryKind::HandoffNote,
            TeamMemoryKind::SharedFact
        ]
    );
    assert_eq!(document.entries[0].title.as_deref(), Some("选择方案 A"));
    assert_eq!(document.entries[0].content, "原因是更简单。");
    assert_eq!(
        document.entries[2].content,
        "The team may not upgrade the runtime."
    );
    for entry in &document.entries {
        assert!(entry.id.starts_with("md-"), "{}", entry.id);
        assert_eq!(entry.id.len(), 19);
        assert_eq!(entry.source.as_deref(), Some(TEAM_MEMORY_FILE));
        assert_eq!(entry.visibility, TeamVisibility::Team);
        assert_eq!(entry.created_at, DateTime::UNIX_EPOCH);
        assert_eq!(entry.updated_at, DateTime::UNIX_EPOCH);
    }
    let memory = TeamMemory::in_memory();
    for _ in 0..2 {
        for entry in &document.entries {
            memory
                .store()
                .upsert(entry)
                .await
                .expect("the entry is stored");
        }
        assert_eq!(stored(&memory).await.len(), 3, "re-import adds nothing");
    }
    let again = TeamMemoryDocument::parse(python_authored, Some(TEAM)).expect("the same ids");
    let ids: Vec<&str> = document.entries.iter().map(|e| e.id.as_str()).collect();
    let again_ids: Vec<&str> = again.entries.iter().map(|e| e.id.as_str()).collect();
    assert_eq!(ids, again_ids);
}

// ---------------------------------------------------------------------------
// Scenario 5: the LLM extractor and its rule fallback.
// ---------------------------------------------------------------------------

/// A transcript the rule extractor turns into one decision record.
fn decision_transcript() -> TeamTranscript {
    TeamTranscript::new(TEAM)
        .with_session("s-1")
        .with_turn(decision_turn())
}

/// Builds the LLM extractor over `model`, keeping a clone of the script for assertions.
fn llm_extractor(model: &ScriptedModel) -> LlmTeamExtractor {
    LlmTeamExtractor::new(
        Arc::new(model.clone()) as Arc<dyn Model>,
        TeamExtractConfig::default(),
    )
    .with_fallback(RuleTeamExtractor::new(TeamExtractConfig::default()))
}

#[tokio::test]
async fn llm_extraction_parses_a_valid_answer_without_degrading() {
    let answer = r#"{"memories":[
        {"kind":"decision","content":"Use SQLite for the shared store.","title":"Storage",
         "tags":["storage"],"importance":0.8,"member_id":"alice"}
    ]}"#;
    let model = ScriptedModel::text(answer);
    let extractor = llm_extractor(&model);
    let extraction = extractor
        .extract(&decision_transcript(), &CallOptions::new())
        .await
        .unwrap();
    assert_eq!(extraction.degraded, None);
    assert_eq!(extraction.len(), 1);
    let record = &extraction.records[0];
    assert_eq!(record.team_id, TEAM);
    assert_eq!(record.kind, TeamMemoryKind::Decision);
    assert_eq!(record.content, "Use SQLite for the shared store.");
    assert_eq!(record.title.as_deref(), Some("Storage"));
    assert_eq!(record.member_id.as_deref(), Some("alice"));
    assert_eq!(record.source.as_deref(), Some(LLM_SOURCE));
    assert!((record.importance.expect("an importance") - 0.8).abs() < f32::EPSILON);
    assert_eq!(model.call_count(), 1);
}

#[tokio::test]
async fn llm_extraction_falls_back_to_rules_on_prose() {
    let model = ScriptedModel::builder()
        .respond_text("Sure! The team agreed on a few things.")
        .respond_text("Still not JSON I am afraid.")
        .build();
    let extractor = llm_extractor(&model);
    let extraction = extractor
        .extract(&decision_transcript(), &CallOptions::new())
        .await
        .unwrap();
    let reason = extraction.degraded.as_ref().expect("a degradation reason");
    assert!(
        reason.starts_with("llm team memory extraction fell back to rules:"),
        "{reason}"
    );
    assert!(
        reason.contains("the model answer had no usable team memories"),
        "{reason}"
    );
    assert_eq!(extraction.len(), 1);
    assert_eq!(extraction.records[0].kind, TeamMemoryKind::Decision);
    assert_eq!(extraction.records[0].source.as_deref(), Some(RULE_SOURCE));
    assert_eq!(model.call_count(), 2, "one retry after the first failure");
}

#[tokio::test]
async fn llm_extraction_falls_back_to_rules_on_a_provider_error() {
    let model = ScriptedModel::builder()
        .fail(ScriptedError::transport("the provider is unreachable"))
        .fail(ScriptedError::transport("the provider is unreachable"))
        .build();
    let extractor = llm_extractor(&model);
    let extraction = extractor
        .extract(&decision_transcript(), &CallOptions::new())
        .await
        .unwrap();
    let reason = extraction.degraded.as_ref().expect("a degradation reason");
    assert!(reason.contains("the model call failed"), "{reason}");
    assert_eq!(extraction.records[0].source.as_deref(), Some(RULE_SOURCE));
    assert_eq!(model.call_count(), 2);
}

#[tokio::test]
async fn a_cancelled_llm_extraction_makes_no_model_call() {
    let model = ScriptedModel::text(r#"{"memories":[{"kind":"decision","content":"x"}]}"#);
    let extractor = llm_extractor(&model);
    let error = extractor
        .extract(&decision_transcript(), &cancelled_call())
        .await
        .unwrap_err();
    assert!(matches!(error, Error::Cancelled), "{error}");
    assert_eq!(model.call_count(), 0);
}

// ---------------------------------------------------------------------------
// Scenario 6: observer events.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn the_observer_reports_successful_operations() {
    let observer = RecordingObserver::default();
    let memory = TeamMemory::in_memory()
        .with_observer(Arc::new(observer.clone()) as Arc<dyn TeamMemoryObserver>);
    assert!(memory.observer().is_some());
    let call = CallOptions::new();
    let record = memory
        .remember(
            NewTeamMemory::decision(TEAM, "Use SQLite for the shared store.").with_member("alice"),
            &call,
        )
        .await
        .unwrap();
    let writes = observer.take();
    assert_eq!(writes.len(), 1);
    assert_eq!(writes[0].operation, TeamMemoryOperation::Write);
    assert_eq!(writes[0].team_id, TEAM);
    assert_eq!(writes[0].member_id.as_deref(), Some("alice"));
    assert_eq!(writes[0].kind, Some(TeamMemoryKind::Decision));
    assert_eq!(writes[0].count, 1);
    assert_eq!(writes[0].outcome, TeamMemoryOutcome::Ok);
    assert!(!writes[0].is_failure());
    assert!(writes[0].error.is_none());

    let hits = memory
        .search(
            &TeamMemoryScope::member(TEAM, "alice"),
            "sqlite shared store",
            &TeamSearchOptions::default(),
        )
        .await
        .unwrap();
    assert_eq!(hits.len(), 1);
    let searches = observer.take();
    assert_eq!(searches.len(), 1);
    assert_eq!(searches[0].operation, TeamMemoryOperation::Search);
    assert_eq!(searches[0].member_id.as_deref(), Some("alice"));
    assert_eq!(searches[0].count, 1);
    assert!(!searches[0].is_failure());

    assert!(memory.forget(TEAM, &record.id).await.unwrap());
    let forgets = observer.take();
    assert_eq!(forgets.len(), 1);
    assert_eq!(forgets[0].operation, TeamMemoryOperation::Forget);
    assert_eq!(forgets[0].count, 1);
    assert!(!forgets[0].is_failure());
}

#[tokio::test]
async fn the_observer_reports_a_failed_operation() {
    let observer = RecordingObserver::default();
    let memory = TeamMemory::in_memory()
        .with_observer(Arc::new(observer.clone()) as Arc<dyn TeamMemoryObserver>);
    let call = CallOptions::new();
    let error = memory
        .remember(NewTeamMemory::decision(TEAM, "   "), &call)
        .await
        .unwrap_err();
    assert!(matches!(error, Error::Validation(_)), "{error}");
    let events = observer.take();
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].operation, TeamMemoryOperation::Write);
    assert_eq!(events[0].outcome, TeamMemoryOutcome::Failed);
    assert!(events[0].is_failure());
    assert_eq!(events[0].count, 0);
    assert!(
        events[0]
            .error
            .as_deref()
            .is_some_and(|text| text.contains("content")),
        "{:?}",
        events[0].error
    );
    assert!(stored(&memory).await.is_empty(), "nothing was written");
}

#[tokio::test]
async fn event_builders_and_the_noop_and_tracing_observers() {
    let event = TeamMemoryEvent::new(TeamMemoryOperation::Extract, TEAM)
        .with_member("alice")
        .with_kind(TeamMemoryKind::Decision)
        .with_count(3);
    assert_eq!(event.operation, TeamMemoryOperation::Extract);
    assert_eq!(event.team_id, TEAM);
    assert_eq!(event.member_id.as_deref(), Some("alice"));
    assert_eq!(event.kind, Some(TeamMemoryKind::Decision));
    assert_eq!(event.count, 3);
    assert_eq!(event.outcome, TeamMemoryOutcome::Ok);
    assert!(!event.is_failure());
    assert!(event.error.is_none());

    let failed = event.clone().failed("the extractor is unavailable");
    assert!(failed.is_failure());
    assert_eq!(failed.outcome, TeamMemoryOutcome::Failed);
    assert_eq!(
        failed.error.as_deref(),
        Some("the extractor is unavailable")
    );
    assert_eq!(failed.count, 3);

    NoopObserver.observe(&event);
    NoopObserver.observe(&failed);
    TracingObserver.observe(&event);
    TracingObserver.observe(&failed);

    let call = CallOptions::new();
    let traced = TeamMemory::in_memory().with_observer(Arc::new(TracingObserver));
    assert!(traced
        .remember(NewTeamMemory::decision(TEAM, "Use SQLite."), &call)
        .await
        .is_ok());
    let silent = TeamMemory::in_memory().with_observer(Arc::new(NoopObserver));
    assert!(silent
        .remember(NewTeamMemory::decision(TEAM, "Use SQLite."), &call)
        .await
        .is_ok());
}
