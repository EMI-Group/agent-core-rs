#![allow(clippy::unwrap_used, clippy::expect_used)]

use openjiuwen_harness::cli::MapEnv;
use openjiuwen_harness::{PersistentSessionRecord, RecordEnding, SessionBinding};
use openjiuwen_test_support::TempWorkspace;

use super::*;
use clap::Parser;

/// Resolves the harness options of an isolated temp home and workspace.
fn options(fixtures: &TempWorkspace) -> CliOptions {
    let env = MapEnv::from_pairs([("OPENJIUWEN_HOME", fixtures.home().display().to_string())]);
    let cli = openjiuwen_harness::cli::Cli::parse_from(["openjiuwen"]);
    CliOptions::resolve_with(&cli, &env, None, &fixtures.workspace()).expect("test options")
}

/// Builds one row from `detail`, as the store loops build them.
fn row_of(id: &str, created_at: Option<&str>, data: Detail) -> Row {
    Row {
        id: id.to_owned(),
        kind: data.kind(),
        created_at: created_at.map(str::to_owned),
        detail: String::new(),
        data,
        sources: Vec::new(),
    }
}

/// Builds a CLI record row.
fn row(id: &str, created_at: Option<&str>) -> Row {
    row_of(
        id,
        created_at,
        Detail::Cli(SessionSummary {
            session_id: id.to_owned(),
            ..SessionSummary::default()
        }),
    )
}

/// Builds an execution journal row.
fn execution_row(
    id: &str,
    agent: &str,
    session: &str,
    created_at: Option<&str>,
    usable: bool,
) -> Row {
    row_of(
        id,
        created_at,
        Detail::Execution(ExecutionSummary {
            execution_id: id.to_owned(),
            agent_id: agent.to_owned(),
            session_id: session.to_owned(),
            created_at: created_at.unwrap_or_default().to_owned(),
            usable,
            ..ExecutionSummary::default()
        }),
    )
}

/// Builds a persistent session record row.
fn state_row(agent: &str, session: &str, created_at: Option<&str>) -> Row {
    row_of(
        session,
        created_at,
        Detail::State(StateSummary {
            binding: SessionBinding::new(agent, session).expect("a valid binding"),
            waiting: false,
            current_window_id: None,
            window_seq: None,
            has_checkpoint: false,
            has_ending: true,
            state_path: Path::new(agent).join(session).join("state.json"),
        }),
    )
}

#[test]
fn rows_sort_newest_first_with_timestampless_rows_last() {
    let mut rows = vec![
        row("b", None),
        row("a", Some("2024-01-01T00:00:00+00:00")),
        row("d", Some("2024-03-01T00:00:00Z")),
        row("c", Some("2024-01-01T00:00:00Z")),
        row("e", None),
    ];
    sort_rows(&mut rows);
    let ids: Vec<&str> = rows.iter().map(|row| row.id.as_str()).collect();
    assert_eq!(ids, vec!["d", "a", "c", "b", "e"]);
}

#[test]
fn an_unparsable_timestamp_sorts_without_one() {
    let mut rows = vec![
        row("a", Some("not a timestamp")),
        row("b", Some("2024-01-01T00:00:00Z")),
    ];
    sort_rows(&mut rows);
    assert_eq!(rows[0].id, "b");
    assert_eq!(rows[1].id, "a");
}

#[test]
fn cli_messages_map_onto_kernel_roles() {
    let message = |role: &str| MessageRecord {
        role: role.to_owned(),
        content: "hi".to_owned(),
        timestamp: String::new(),
    };
    assert_eq!(
        message_from_record(&message("user")).map(|message| message.role.as_str().to_owned()),
        Some("user".to_owned())
    );
    assert_eq!(
        message_from_record(&message("assistant")).map(|message| message.role.as_str().to_owned()),
        Some("assistant".to_owned())
    );
    assert_eq!(
        message_from_record(&message("system")).map(|message| message.role.as_str().to_owned()),
        Some("system".to_owned())
    );
    let tool = MessageRecord {
        role: "tool".to_owned(),
        content: "result".to_owned(),
        timestamp: String::new(),
    };
    assert!(message_from_record(&tool).is_none());
}

#[test]
fn resume_notes_use_the_documented_prefix() {
    assert_eq!(history_prefix("cli-1", false), "resuming session cli-1");
    assert_eq!(
        history_prefix("exec-1", true),
        "resuming the most recent session exec-1"
    );
}

#[test]
fn blank_timestamps_are_reported_as_missing() {
    assert_eq!(non_empty("  "), None);
    assert_eq!(
        non_empty("2024-01-01T00:00:00Z").as_deref(),
        Some("2024-01-01T00:00:00Z")
    );
}

#[tokio::test]
async fn a_persistent_state_record_lists_resolves_and_prunes() {
    let fixtures = TempWorkspace::new().expect("temp workspace");
    let options = options(&fixtures);
    let layout = PersistentAgentLayout::for_workspace(&Workspace::new(options.workspace.clone()));
    let store = FileSessionStore::new(layout.sessions_root());
    let binding = SessionBinding {
        agent_id: "agent-one".to_owned(),
        session_id: "sess-1".to_owned(),
    };
    let mut record = PersistentSessionRecord::new(&binding).expect("draft record");
    record.messages = vec![BaseMessage::user("hi"), BaseMessage::assistant("hello")];
    let record = record
        .with_ending(RecordEnding {
            outcome: "completed".to_owned(),
            request_id: "req-1".to_owned(),
            ..RecordEnding::default()
        })
        .expect("ended record");
    store.save(&record).await.expect("save record");

    let catalog = Catalog::for_options(&options, None);
    let rows = catalog.list().await.expect("list");
    let row = rows
        .iter()
        .find(|row| row.id == "sess-1")
        .expect("state row");
    assert_eq!(row.kind, KIND_STATE);
    assert!(row.created_at.is_some());
    assert_eq!(row.detail, "agent agent-one, waiting false, ending true");

    let resolved = catalog
        .resolve(ResumeRequest::Id("sess-1"))
        .await
        .expect("resolve id");
    assert_eq!(resolved.kind, KIND_STATE);
    assert_eq!(resolved.history.len(), 2);
    assert!(resolved.plan.is_none());
    assert_eq!(
        resolved.summary,
        "resuming session sess-1 from the persistent session record (2 messages restored)"
    );

    let continued = catalog
        .resolve(ResumeRequest::Continue)
        .await
        .expect("resolve latest");
    assert_eq!(continued.id, "sess-1");
    assert!(continued
        .summary
        .starts_with("resuming the most recent session sess-1"));

    let pruned = catalog.prune(None, true, false).await.expect("prune");
    assert_eq!(pruned.len(), 1);
    assert!(catalog.list().await.expect("list after prune").is_empty());
}

#[tokio::test]
async fn a_cli_record_resolves_with_its_transcript() {
    let fixtures = TempWorkspace::new().expect("temp workspace");
    let options = options(&fixtures);
    let store = SessionStore::new(&options.home);
    store
        .new_session("cli-1", "mock-model")
        .expect("new session");
    store
        .append_message("cli-1", "user", "hi")
        .expect("append user");
    store
        .append_message("cli-1", "assistant", "hello")
        .expect("append assistant");

    let catalog = Catalog::for_options(&options, None);
    let resolved = catalog
        .resolve(ResumeRequest::Id("cli-1"))
        .await
        .expect("resolve id");
    assert_eq!(resolved.kind, KIND_CLI);
    assert_eq!(resolved.history.len(), 2);
    assert_eq!(resolved.history[0].role.as_str(), "user");
    assert_eq!(
        resolved.summary,
        "resuming session cli-1 from the CLI session record (2 messages)"
    );
    assert_eq!(
        resolved.record.expect("cli record").model,
        "mock-model".to_owned()
    );

    let (row, shown) = catalog.show("cli-1").await.expect("show");
    assert_eq!(row.kind, KIND_CLI);
    assert_eq!(shown.id, "cli-1");
}

#[tokio::test]
async fn an_unknown_id_and_an_empty_store_are_typed_not_found() {
    let fixtures = TempWorkspace::new().expect("temp workspace");
    let options = options(&fixtures);
    let catalog = Catalog::for_options(&options, None);
    let missing = catalog
        .resolve(ResumeRequest::Id("nope"))
        .await
        .expect_err("unknown id");
    assert!(missing.to_string().contains("unknown session: nope"));
    let empty = catalog
        .resolve(ResumeRequest::Continue)
        .await
        .expect_err("nothing to continue");
    assert!(empty.to_string().contains("no resumable session found"));
    let invalid = catalog
        .prune(None, false, false)
        .await
        .expect_err("no selector");
    assert!(invalid.to_string().contains("--older-than"));
}

#[test]
fn one_session_with_state_everywhere_folds_into_one_row() {
    let rows = merge_rows(vec![
        execution_row(
            "exec-1",
            CLI_AGENT_ID,
            "cli-1",
            Some("2024-01-01T00:00:00Z"),
            true,
        ),
        row("cli-1", Some("2024-01-02T00:00:00Z")),
        state_row(CLI_AGENT_ID, "cli-1", Some("2024-01-03T00:00:00Z")),
    ]);
    assert_eq!(rows.len(), 1, "one row per session: {rows:?}");
    let kept = &rows[0];
    assert_eq!(kept.id, "cli-1");
    assert_eq!(kept.kind, KIND_STATE, "the richest source is kept");
    assert_eq!(
        kept.created_at.as_deref(),
        Some("2024-01-03T00:00:00Z"),
        "the kept source keeps its own timestamp"
    );
    let kinds: Vec<&str> = kept.sources.iter().map(Detail::kind).collect();
    assert_eq!(kinds, vec![KIND_CLI, KIND_EXECUTION], "richest first");
}

#[test]
fn rows_of_different_sessions_stay_separate() {
    let rows = merge_rows(vec![
        row("cli-1", Some("2024-01-01T00:00:00Z")),
        row("cli-2", Some("2024-01-02T00:00:00Z")),
        state_row("another-agent", "cli-1", Some("2024-01-03T00:00:00Z")),
    ]);
    assert_eq!(
        rows.len(),
        3,
        "different agents and different sessions never group: {rows:?}"
    );
}

#[test]
fn a_state_row_inherits_a_timestamp_it_does_not_have() {
    let rows = merge_rows(vec![
        state_row(CLI_AGENT_ID, "cli-1", None),
        row("cli-1", Some("2024-01-02T00:00:00Z")),
    ]);
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].kind, KIND_STATE);
    assert_eq!(
        rows[0].created_at.as_deref(),
        Some("2024-01-02T00:00:00Z"),
        "the row inherits the newest timestamp of the group: {rows:?}"
    );
}

#[test]
fn a_sessionless_execution_is_keyed_by_its_own_id() {
    let rows = merge_rows(vec![
        execution_row(
            "exec-1",
            CLI_AGENT_ID,
            "",
            Some("2024-01-01T00:00:00Z"),
            true,
        ),
        execution_row(
            "exec-2",
            CLI_AGENT_ID,
            "",
            Some("2024-01-02T00:00:00Z"),
            true,
        ),
    ]);
    assert_eq!(
        rows.len(),
        2,
        "executions without a session keep their own identity: {rows:?}"
    );
}

#[test]
fn an_execution_of_a_session_folds_into_its_cli_row() {
    let rows = merge_rows(vec![
        execution_row(
            "exec-1",
            CLI_AGENT_ID,
            "cli-1",
            Some("2024-01-01T00:00:00Z"),
            true,
        ),
        row("cli-1", Some("2024-01-02T00:00:00Z")),
    ]);
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].id, "cli-1");
    assert_eq!(rows[0].kind, KIND_CLI, "cli outranks execution");
}

#[test]
fn every_member_of_a_group_survives_as_a_source() {
    let rows = merge_rows(vec![
        execution_row(
            "exec-1",
            CLI_AGENT_ID,
            "cli-1",
            Some("2024-01-01T00:00:00Z"),
            true,
        ),
        execution_row(
            "exec-2",
            CLI_AGENT_ID,
            "cli-1",
            Some("2024-01-02T00:00:00Z"),
            true,
        ),
        row("cli-1", Some("2024-01-03T00:00:00Z")),
        state_row(CLI_AGENT_ID, "cli-1", None),
    ]);
    assert_eq!(rows.len(), 1);
    assert_eq!(
        rows[0].sources.len(),
        3,
        "every non-kept member is a source: {rows:?}"
    );
    let kinds: Vec<&str> = rows[0].sources.iter().map(Detail::kind).collect();
    assert_eq!(
        kinds,
        vec![KIND_CLI, KIND_EXECUTION, KIND_EXECUTION],
        "sources are richest first, then by id"
    );
}

#[test]
fn has_kind_sees_every_source_of_a_session() {
    let rows = merge_rows(vec![
        execution_row("exec-1", CLI_AGENT_ID, "cli-1", None, true),
        row("cli-1", None),
        state_row(CLI_AGENT_ID, "cli-1", None),
    ]);
    let kept = &rows[0];
    for kind in [KIND_CLI, KIND_EXECUTION, KIND_STATE] {
        assert!(kept.has_kind(kind), "`{kind}` is one of the row's sources");
    }
    assert!(!kept.has_kind("unknown"));
}

#[test]
fn only_a_group_of_unusable_journals_is_unresumable() {
    let unusable = merge_rows(vec![
        execution_row("exec-1", CLI_AGENT_ID, "cli-1", None, false),
        execution_row("exec-2", CLI_AGENT_ID, "cli-1", None, false),
    ]);
    assert_eq!(unusable.len(), 1);
    assert!(
        !unusable[0].is_resumable(),
        "a session made only of damaged journals cannot be continued"
    );

    let mixed = merge_rows(vec![
        execution_row("exec-1", CLI_AGENT_ID, "cli-1", None, false),
        row("cli-1", None),
    ]);
    assert!(
        mixed[0].is_resumable(),
        "the CLI record behind the row is resumable"
    );

    assert!(
        row("cli-1", None).is_resumable(),
        "a CLI record is resumable"
    );
}
