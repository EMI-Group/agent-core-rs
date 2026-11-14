//! Team memory cross-crate scenario: lifecycle hooks, the member toolkit, and `TEAM_MEMORY.md`.
//!
//! `openjiuwen::core::memory::team` exposes team-scoped memory as caller-driven adapters over one
//! facade, and `openjiuwen::agent_teams` owns the team runtime. This suite drives the two the way
//! a team host would, from outside both crates:
//!
//! 1. **Hooks** — one team session runs two member turns through [`TeamMemoryHooks`] with the
//!    deterministic rule extractor. Member A's fact-bearing turn is buffered, flushed into a
//!    non-empty [`TeamMemoryFlush`], and then read back by member B through the shared
//!    [`TeamMemory`] (the toolkit search member B's turn performs).
//! 2. **Toolkit** — two [`MemberMemoryToolkit`] instances over the same facade: the schemas name
//!    the calling member, a write by A is visible to B's team scope but A's private record is not,
//!    and a `read_only()` context refuses writes with a typed error plus the documented envelope.
//! 3. **Document** — the team memory document is rendered to `TEAM_MEMORY.md` inside a test-owned
//!    [`TempDir`], its section/payload invariants are pinned, and the file is imported into a fresh
//!    facade twice so the records survive with their ids/salience/tags and are not duplicated.
//! 4. **Runtime composition** — a board task is created, claimed and completed through
//!    `openjiuwen::agent_teams::taskboard::TaskBoard`, and the completed task's own status event
//!    becomes the member turn the hooks flush. The agent-teams runtime exposes no team-memory
//!    wiring (no `TeamMemory`/`TeamMemoryHooks` integration point, no `openjiuwen-memory`
//!    dependency), so the caller-side composition is what is exercised — see the Known Gap note
//!    on [`a_completed_board_task_becomes_a_flushed_member_memory`].
//!
//! Everything is offline and deterministic: the in-memory team memory store, one scripted rule
//! extractor, one temp directory, and every await bounded by [`support::BOUND`].
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::too_many_lines)]

mod support;

use std::collections::BTreeMap;
use std::fs;
use std::sync::Arc;

use openjiuwen::agent_teams::schema::{TaskGraphSpec, TaskStatus};
use openjiuwen::agent_teams::storage::InMemoryTeamStore;
use openjiuwen::agent_teams::taskboard::TaskBoard;
use openjiuwen::core::common::error::Error;
use openjiuwen::core::memory::team::{
    document_team_id, truncate_for_prompt, MemberMemoryContext, MemberMemoryToolkit, TeamMemory,
    TeamMemoryDocument, TeamMemoryFilter, TeamMemoryHookConfig, TeamMemoryHooks, TeamMemoryKind,
    TeamMemoryRecord, TeamMemoryScope, TeamSearchOptions, TeamTurn, TeamVisibility, HOOK_SOURCE,
    RULE_SOURCE, TEAM_MEMORY_FILE, TEAM_MEMORY_FORMAT, TOOL_FORGET, TOOL_REMEMBER, TOOL_SEARCH,
    TOOL_STATS,
};
use openjiuwen::core::memory::CallOptions;
use serde_json::json;
use tempfile::TempDir;

use support::within;

/// Team every fixture reads and writes.
const TEAM: &str = "review_team";

/// The coordination session the hooks bind and every record is stamped with.
const SESSION: &str = "session-1";

/// The member whose fact-bearing turn opens the session.
const ALICE: &str = "alice";

/// The member who reads Alice's fact and then contributes its own turn.
const BOB: &str = "bob";

/// The board member the completion turn is attributed to.
const DEV: &str = "dev";

/// A turn the rule extractor classifies as a `Decision`.
const DECISION: &str = "We decided to use SQLite for the shared store.";

/// A turn the rule extractor classifies as a `HandoffNote` (the `lesson` section).
const LESSON: &str = "Next time we should run the migration first";

/// A turn the rule extractor classifies as a `Task` (`action item` marker).
const ACTION_ITEM: &str = "Action item: ship the release";

/// One buffered team turn.
///
/// The timestamp is left to [`TeamTurn`]'s serde default (the Unix epoch), so no scenario depends
/// on the wall clock and no time type is named in this crate.
fn turn(from_member: &str, content: &str) -> TeamTurn {
    serde_json::from_value(json!({"from_member": from_member, "content": content}))
        .expect("a team turn")
}

/// Rule-extractor hooks over `memory`, wired the way a team host wires them.
fn hooks(memory: &Arc<TeamMemory>) -> TeamMemoryHooks {
    TeamMemoryHooks::with_rule_extractor(TEAM, Arc::clone(memory), TeamMemoryHookConfig::default())
}

/// Every record of [`TEAM`], as the store holds it.
async fn stored(memory: &TeamMemory) -> Vec<TeamMemoryRecord> {
    within("team list", memory.list(&TeamMemoryFilter::for_team(TEAM)))
        .await
        .expect("the team lists")
}

/// Indexes records by id so two record sets compare order-insensitively.
fn by_id(records: Vec<TeamMemoryRecord>) -> BTreeMap<String, TeamMemoryRecord> {
    records
        .into_iter()
        .map(|record| (record.id.clone(), record))
        .collect()
}

/// Parses a toolkit success envelope's `result` into the record it carries.
fn record_of(envelope: &serde_json::Value) -> TeamMemoryRecord {
    serde_json::from_value(envelope["result"].clone()).expect("the stored record")
}

// ---------------------------------------------------------------------------
// Scenario 1: two member turns through the lifecycle hooks.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn member_a_flushes_a_fact_and_member_b_reads_it_from_the_shared_memory() {
    let memory = Arc::new(TeamMemory::in_memory());
    let hooks = hooks(&memory);
    let call = CallOptions::new();

    within("session start", hooks.on_session_start(SESSION, &call))
        .await
        .expect("the session starts");
    assert_eq!(hooks.session_id().as_deref(), Some(SESSION));
    assert_eq!(hooks.extractor_name(), "rule");
    let profile = within("member join", hooks.on_member_join(ALICE, &call))
        .await
        .expect("the join is recorded")
        .expect("a profile record");
    assert_eq!(profile.kind, TeamMemoryKind::MemberProfile);
    assert_eq!(profile.member_id.as_deref(), Some(ALICE));
    assert_eq!(profile.session_id.as_deref(), Some(SESSION));
    assert_eq!(profile.source.as_deref(), Some(HOOK_SOURCE));
    assert_eq!(profile.visibility, TeamVisibility::Team);
    assert_eq!(
        profile.tags,
        vec!["member".to_owned(), "presence".to_owned()]
    );
    assert_eq!(stored(&memory).await.len(), 1, "the join wrote one profile");

    within("turn a", hooks.on_turn(turn(ALICE, DECISION), &call))
        .await
        .expect("the turn is buffered");
    assert_eq!(hooks.pending_turns(), 1, "turns buffer until the flush");
    assert_eq!(stored(&memory).await.len(), 1, "buffering writes nothing");

    let flush = within("flush a", hooks.on_session_end(&call))
        .await
        .expect("the flush extracts the buffered turn");
    assert!(!flush.is_empty(), "{flush:?}");
    assert_eq!(flush.turns, 1);
    assert_eq!(flush.len(), 1);
    assert!(
        flush.degraded.is_none(),
        "the rule extractor never degrades"
    );
    assert_eq!(hooks.pending_turns(), 0, "the flush drains the buffer");

    let decision = &flush.records[0];
    assert_eq!(decision.team_id, TEAM);
    assert_eq!(decision.member_id.as_deref(), Some(ALICE));
    assert_eq!(decision.session_id.as_deref(), Some(SESSION));
    assert_eq!(decision.kind, TeamMemoryKind::Decision);
    assert_eq!(decision.content, DECISION);
    assert_eq!(decision.source.as_deref(), Some(RULE_SOURCE));
    assert_eq!(decision.tags, vec!["decision".to_owned()]);
    assert_eq!(decision.visibility, TeamVisibility::Team);
    assert!(
        decision.id.starts_with("hook-"),
        "the flush writes a deterministic id: {}",
        decision.id
    );
    assert_eq!(stored(&memory).await.len(), 2, "profile plus the decision");

    // Member B's turn reads the shared memory before it contributes anything.
    let bob = MemberMemoryToolkit::new(
        Arc::clone(&memory),
        MemberMemoryContext::new(TEAM)
            .with_member(BOB)
            .with_session(SESSION),
    );
    let hits = within(
        "bob search",
        bob.handle_tool_call(TOOL_SEARCH, json!({"query": "sqlite shared store"})),
    )
    .await
    .expect("the toolkit serves the search");
    assert_eq!(hits["ok"], json!(true));
    assert_eq!(
        hits["result"]["count"],
        json!(1),
        "member B sees exactly A's fact: {hits}"
    );
    assert_eq!(hits["result"]["results"][0]["id"], json!(decision.id));
    assert_eq!(hits["result"]["results"][0]["content"], json!(DECISION));
    assert_eq!(hits["result"]["results"][0]["member_id"], json!(ALICE));

    within("turn b", hooks.sync_turn(turn(BOB, LESSON), &call))
        .await
        .expect("the documented alias buffers the turn");
    assert_eq!(hooks.pending_turns(), 1);
    let flush_b = within("flush b", hooks.on_session_end(&call))
        .await
        .expect("member B's turn is flushed");
    assert_eq!(flush_b.turns, 1);
    assert_eq!(flush_b.len(), 1);
    assert!(flush_b.degraded.is_none());
    assert_eq!(
        hooks.pending_turns(),
        0,
        "the second flush drains the buffer"
    );
    let lesson = &flush_b.records[0];
    assert_eq!(lesson.member_id.as_deref(), Some(BOB));
    assert_eq!(lesson.session_id.as_deref(), Some(SESSION));
    assert_eq!(lesson.kind, TeamMemoryKind::HandoffNote);
    assert_eq!(lesson.content, LESSON);
    assert_eq!(lesson.tags, vec!["lesson".to_owned()]);

    let all = stored(&memory).await;
    assert_eq!(all.len(), 3, "profile + A's decision + B's lesson");
    let scope = TeamMemoryScope::member(TEAM, BOB).with_session(SESSION);
    let readable: Vec<&str> = all
        .iter()
        .filter(|record| scope.allows(record))
        .map(|record| record.content.as_str())
        .collect();
    assert!(readable.contains(&DECISION), "{readable:?}");
    assert!(readable.contains(&LESSON), "{readable:?}");
    assert_eq!(hooks.members(), vec![ALICE.to_owned()], "only A joined");
}

// ---------------------------------------------------------------------------
// Scenario 2: two member toolkits over one shared store.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn member_toolkits_share_team_records_but_refuse_read_only_writes() {
    let memory = Arc::new(TeamMemory::in_memory());
    let alice = MemberMemoryToolkit::new(
        Arc::clone(&memory),
        MemberMemoryContext::new(TEAM)
            .with_member(ALICE)
            .with_session(SESSION),
    );
    let bob = MemberMemoryToolkit::new(
        Arc::clone(&memory),
        MemberMemoryContext::new(TEAM)
            .with_member(BOB)
            .with_session(SESSION),
    );

    let alice_schemas = alice.get_tool_schemas();
    let names = alice.tool_names();
    assert_eq!(
        names,
        vec![TOOL_REMEMBER, TOOL_SEARCH, TOOL_FORGET, TOOL_STATS]
    );
    assert_eq!(alice_schemas.len(), 4);
    for (schema, name) in alice_schemas.iter().zip(names.iter().copied()) {
        assert_eq!(schema["name"], json!(name));
        assert_eq!(
            schema["id"],
            json!(format!("team_memory.{TEAM}.{ALICE}.{name}"))
        );
        assert_eq!(schema["input_schema"]["type"], json!("object"));
        assert!(
            schema["description"]
                .as_str()
                .is_some_and(|text| !text.is_empty()),
            "`{name}` needs a description"
        );
        let expected_required: &[&str] = match name {
            TOOL_REMEMBER => &["content"],
            TOOL_SEARCH => &["query"],
            TOOL_FORGET => &["id"],
            _ => &[],
        };
        assert_eq!(
            schema["input_schema"]["required"],
            json!(expected_required),
            "the `{name}` required arguments"
        );
    }
    let bob_schemas = bob.get_tool_schemas();
    assert_eq!(
        bob_schemas[0]["id"],
        json!(format!("team_memory.{TEAM}.{BOB}.{TOOL_REMEMBER}")),
        "the schema id names the calling member"
    );
    assert_ne!(bob_schemas[0]["id"], alice_schemas[0]["id"]);

    let shared = within(
        "alice remember",
        alice.handle_tool_call(
            TOOL_REMEMBER,
            json!({
                "content": "Use SQLite for the shared deployment store.",
                "title": "Storage choice",
                "kind": "decision",
                "tags": ["storage"],
                "importance": 0.8,
            }),
        ),
    )
    .await
    .expect("the toolkit writes");
    assert_eq!(shared["ok"], json!(true));
    assert_eq!(shared["tool"], json!(TOOL_REMEMBER));
    let shared_record = record_of(&shared);
    assert_eq!(shared_record.member_id.as_deref(), Some(ALICE));
    assert_eq!(shared_record.session_id.as_deref(), Some(SESSION));
    assert_eq!(shared_record.kind, TeamMemoryKind::Decision);
    assert_eq!(shared_record.visibility, TeamVisibility::Team);
    assert_eq!(shared_record.tags, vec!["storage".to_owned()]);
    assert!((shared_record.importance - 0.8).abs() < f32::EPSILON);

    let private = within(
        "alice private write",
        alice.handle_tool_call(
            TOOL_REMEMBER,
            json!({
                "content": "Alice keeps the rollout keys in the vault.",
                "visibility": "private",
            }),
        ),
    )
    .await
    .expect("the private write");
    let private_record = record_of(&private);
    assert_eq!(private_record.visibility, TeamVisibility::Private);

    let bob_hits = within(
        "bob search",
        bob.handle_tool_call(
            TOOL_SEARCH,
            json!({"query": "sqlite shared deployment store"}),
        ),
    )
    .await
    .expect("the toolkit serves member B's search");
    assert_eq!(
        bob_hits["result"]["count"],
        json!(1),
        "another member reads the team-visible record: {bob_hits}"
    );
    assert_eq!(
        bob_hits["result"]["results"][0]["id"],
        json!(shared_record.id)
    );
    let hidden = within(
        "bob private search",
        bob.handle_tool_call(TOOL_SEARCH, json!({"query": "rollout keys vault"})),
    )
    .await
    .expect("the toolkit serves the second search");
    assert_eq!(
        hidden["result"]["count"],
        json!(0),
        "the private record stays with its author"
    );
    let stats = within("bob stats", bob.handle_tool_call(TOOL_STATS, json!({})))
        .await
        .expect("the toolkit aggregates");
    assert_eq!(stats["result"]["total"], json!(1));
    assert_eq!(stats["result"]["by_member"][ALICE], json!(1));

    let reader = MemberMemoryToolkit::new(
        Arc::clone(&memory),
        MemberMemoryContext::new(TEAM)
            .with_member(BOB)
            .with_session(SESSION)
            .read_only(),
    );
    assert!(reader.context().read_only);
    for (tool, args) in [
        (
            TOOL_REMEMBER,
            json!({"content": "A read-only member may not write this."}),
        ),
        (TOOL_FORGET, json!({"id": shared_record.id})),
    ] {
        let error = within("read-only refusal", reader.handle_tool_call(tool, args))
            .await
            .expect_err("a read-only context refuses writes");
        assert!(matches!(error, Error::Unsupported(_)), "{tool}: {error}");
        let envelope = MemberMemoryToolkit::error_envelope(tool, &error);
        assert_eq!(envelope["ok"], json!(false));
        assert_eq!(envelope["tool"], json!(tool));
        assert_eq!(envelope["error"]["kind"], json!("unsupported"));
        assert!(
            envelope["error"]["message"]
                .as_str()
                .is_some_and(|message| message.contains("read-only")),
            "{envelope}"
        );
    }
    let still_reads = within(
        "read-only search",
        reader.handle_tool_call(
            TOOL_SEARCH,
            json!({"query": "sqlite shared deployment store"}),
        ),
    )
    .await
    .expect("a read-only context still searches");
    assert_eq!(still_reads["result"]["count"], json!(1));
    assert_eq!(
        stored(&memory).await.len(),
        2,
        "the refusals wrote nothing beyond the two records"
    );
}

// ---------------------------------------------------------------------------
// Scenario 3: the `TEAM_MEMORY.md` file round trip.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn team_memory_markdown_exports_to_a_file_and_imports_without_duplicates() {
    let dir = TempDir::new().expect("a temp directory");
    let memory = Arc::new(TeamMemory::in_memory());
    let hooks = hooks(&memory);
    let call = CallOptions::new();

    within("session start", hooks.on_session_start(SESSION, &call))
        .await
        .expect("the session starts");
    let profile = within("alice joins", hooks.on_member_join(ALICE, &call))
        .await
        .expect("the join is recorded")
        .expect("a profile record");
    within("turn a", hooks.on_turn(turn(ALICE, DECISION), &call))
        .await
        .expect("the decision is buffered");
    let first = within("flush a", hooks.on_session_end(&call))
        .await
        .expect("the decision is flushed");
    within("turn b", hooks.on_turn(turn(BOB, LESSON), &call))
        .await
        .expect("the lesson is buffered");
    let second = within("flush b", hooks.on_session_end(&call))
        .await
        .expect("the lesson is flushed");
    let mut expected = vec![profile];
    expected.extend(first.records.iter().cloned());
    expected.extend(second.records.iter().cloned());
    assert_eq!(expected.len(), 3, "profile + decision + lesson");

    let markdown = within(
        "export",
        memory.export_markdown(&TeamMemoryFilter::for_team(TEAM)),
    )
    .await
    .expect("the document renders");
    let path = dir.path().join(TEAM_MEMORY_FILE);
    fs::write(&path, &markdown).expect("the document is written to the temp directory");
    assert!(path.is_file());
    let on_disk = fs::read_to_string(&path).expect("the document reads back");
    assert_eq!(on_disk, markdown, "the file carries the export verbatim");

    assert!(
        on_disk.starts_with(&format!(
            "---\nteam_memory: {TEAM_MEMORY_FORMAT}\nteam_id: {TEAM}\n"
        )),
        "{on_disk}"
    );
    assert!(on_disk.contains("entries: 3"), "{on_disk}");
    assert!(on_disk.contains(&format!("# Team memory — {TEAM}")));
    for section in ["decision", "member", "lesson"] {
        assert!(
            on_disk.contains(&format!("\n## {section}\n")),
            "the `{section}` section is missing: {on_disk}"
        );
    }
    assert!(on_disk.contains(&format!("### [decision] {DECISION}")));
    assert!(on_disk.contains(&format!("### [member] Member {ALICE} joined")));
    assert!(on_disk.contains(&format!("### [lesson] {LESSON}")));
    assert!(on_disk.contains("<!-- team-memory: {"));
    for record in &expected {
        assert!(
            on_disk.contains(&format!("*id: {}", record.id)),
            "{}",
            record.id
        );
    }
    let decision = expected
        .iter()
        .find(|record| record.kind == TeamMemoryKind::Decision)
        .expect("the flushed decision");
    let metadata_line = format!(
        "*id: {} · member: {ALICE} · importance: {:.2}",
        decision.id, decision.importance
    );
    assert!(on_disk.contains(&metadata_line), "{metadata_line}");
    assert!(on_disk.contains("tags: decision"), "{on_disk}");
    assert_eq!(document_team_id(&on_disk).as_deref(), Some(TEAM));
    assert_eq!(
        truncate_for_prompt(&on_disk),
        on_disk.trim(),
        "a short document is the whole prompt view"
    );

    let parsed = TeamMemoryDocument::parse(&on_disk, None).expect("the document parses");
    assert_eq!(parsed.team_id, TEAM);
    assert_eq!(parsed.len(), 3);
    assert_eq!(
        by_id(parsed.entries),
        by_id(expected.clone()),
        "the render/parse round trip is lossless"
    );

    let imported_memory = TeamMemory::in_memory();
    let imported = within("import", imported_memory.import_markdown(&on_disk, &call))
        .await
        .expect("the document imports");
    assert_eq!(imported.len(), 3);
    assert_eq!(
        by_id(imported.clone()),
        by_id(expected.clone()),
        "ids, salience, tags and timestamps survive the file"
    );
    assert_eq!(
        by_id(stored(&imported_memory).await),
        by_id(expected.clone())
    );

    let again = within(
        "re-import",
        imported_memory.import_markdown(&on_disk, &call),
    )
    .await
    .expect("the same document imports again");
    assert_eq!(again.len(), 3);
    let after_second = stored(&imported_memory).await;
    assert_eq!(
        after_second.len(),
        3,
        "a second import upserts instead of duplicating: {after_second:?}"
    );
    assert_eq!(
        by_id(after_second),
        by_id(expected),
        "the re-import preserved every field"
    );
}

// ---------------------------------------------------------------------------
// Scenario 4: the board's completion is the member turn the hooks flush.
// ---------------------------------------------------------------------------

/// Drives a `TaskBoard` task to `completed`, then flushes the member turn derived from it.
///
/// Known Gap: `openjiuwen-agent-teams` exposes no team-memory wiring — it has no `TeamMemory` or
/// `TeamMemoryHooks` integration point and no `openjiuwen-memory` dependency — so the host composes
/// the two surfaces itself: the board event and the task's own summary are turned into a
/// [`TeamTurn`] that the hooks extract. Nothing here may be "fixed" inside the sibling crate; the
/// scenario pins the caller-side composition the runtime would need.
#[tokio::test]
async fn a_completed_board_task_becomes_a_flushed_member_memory() {
    let store = Arc::new(InMemoryTeamStore::new());
    let board = TaskBoard::with_defaults(store, TEAM);

    let created = within(
        "task create",
        board.create_task(TaskGraphSpec {
            title: "ship the release".to_owned(),
            content: "cut the tag and publish the shared store".to_owned(),
            ..TaskGraphSpec::default()
        }),
    )
    .await
    .expect("the board creates the task");
    let task_id = created.task.task_id.clone();
    assert_eq!(
        created
            .events
            .iter()
            .map(openjiuwen::agent_teams::schema::EventPayload::event_type)
            .collect::<Vec<_>>(),
        ["task_created", "task_status_changed"]
    );

    let claimed = within("task claim", board.claim(&task_id, DEV))
        .await
        .expect("the claim is applied");
    assert!(claimed.ok, "{claimed:?}");
    let completed = within(
        "task complete",
        board.complete(&task_id, DEV, "shipped the shared store"),
    )
    .await
    .expect("the completion is applied");
    assert_eq!(completed.task.status, TaskStatus::Completed);
    assert_eq!(completed.task.assignee.as_deref(), Some(DEV));
    assert!(within("board settled", board.is_settled()).await.unwrap());

    let status_event = completed
        .events
        .iter()
        .find(|event| event.event_type() == "task_status_changed")
        .expect("the completion publishes the status transition");
    let status_payload = status_event
        .payload_value()
        .expect("the status payload serializes");
    assert_eq!(status_payload["status"], json!("completed"));
    assert_eq!(status_payload["assignee"], json!(DEV));
    assert_eq!(status_payload["task_id"], json!(task_id));

    let memory = Arc::new(TeamMemory::in_memory());
    let hooks = hooks(&memory);
    let call = CallOptions::new();
    within("session start", hooks.on_session_start(SESSION, &call))
        .await
        .expect("the session starts");
    within("dev joins", hooks.on_member_join(DEV, &call))
        .await
        .expect("the join is recorded")
        .expect("a profile record");
    let summary = completed
        .task
        .summary
        .clone()
        .expect("the completion summary");
    let turn_content = format!("{ACTION_ITEM} — {summary}");
    within("board turn", hooks.on_turn(turn(DEV, &turn_content), &call))
        .await
        .expect("the board's completion becomes a member turn");
    let flush = within("flush", hooks.on_session_end(&call))
        .await
        .expect("the board turn is extracted");
    assert_eq!(flush.turns, 1);
    assert_eq!(flush.len(), 1);
    assert!(flush.degraded.is_none());
    assert_eq!(hooks.pending_turns(), 0);

    let record = &flush.records[0];
    assert_eq!(record.team_id, TEAM);
    assert_eq!(record.member_id.as_deref(), Some(DEV));
    assert_eq!(record.session_id.as_deref(), Some(SESSION));
    assert_eq!(record.kind, TeamMemoryKind::Task);
    assert_eq!(record.content, turn_content);
    assert!(record.content.contains("ship the release"));
    assert!(record.content.contains("shipped the shared store"));
    assert_eq!(record.source.as_deref(), Some(RULE_SOURCE));
    assert_eq!(
        record.tags,
        vec!["task".to_owned(), "context".to_owned()],
        "a rule match earns its marker category plus the kind's `TEAM_MEMORY.md` tag"
    );
    assert_eq!(stored(&memory).await.len(), 2, "profile plus the task");

    let hits = within(
        "task recall",
        memory.search(
            &TeamMemoryScope::new(TEAM),
            "shipped shared store",
            &TeamSearchOptions::default(),
        ),
    )
    .await
    .expect("the task memory is recallable");
    assert_eq!(hits.len(), 1, "{hits:?}");
    assert_eq!(hits[0].record.id, record.id);
    assert_eq!(hits[0].record.content, turn_content);
}
