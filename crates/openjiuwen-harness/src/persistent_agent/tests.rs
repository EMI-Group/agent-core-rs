//! Cross-module offline tests for the persistent-agent module.
//!
//! The suites drive the real file-backed stores against a temporary root and an injected
//! [`ManualClock`], so the flows here never touch a network, a wall clock or a model: a simulated
//! crash and resume, an in-process handoff, a share-bundle round trip, the independence of the
//! event emitter and the journal, and the typed refusals of the deferred capabilities.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;
use std::sync::Arc;

use openjiuwen_common::error::Error;
use openjiuwen_foundation::prompt::message::BaseMessage;

use super::*;

/// Frozen instant every fixture clock reports: `2023-11-14T22:13:20.000Z`.
const NOW_MS: i64 = 1_700_000_000_000;

/// A manual clock frozen at [`NOW_MS`].
fn frozen_clock() -> Arc<ManualClock> {
    Arc::new(ManualClock::new(NOW_MS))
}

/// The `(agent, session)` pair every fixture targets.
fn binding() -> SessionBinding {
    SessionBinding::new("agent", "session").unwrap()
}

/// The source execution of the handoff fixtures.
fn identity() -> ExecutionIdentity {
    ExecutionIdentity::new("ns", "agent", "session", "request", "execution", "source").unwrap()
}

/// A completed record of the fixture session holding one message.
fn completed_record() -> PersistentSessionRecord {
    let mut record = PersistentSessionRecord::new(&binding()).unwrap();
    record.messages = vec![BaseMessage::user("state")];
    record
        .with_ending(RecordEnding {
            outcome: "completed".to_string(),
            request_id: "request".to_string(),
            execution_id: "execution".to_string(),
            ..RecordEnding::default()
        })
        .unwrap()
}

/// A consistent transfer of [`completed_record`] to the fixture binding.
fn transfer() -> HandoffTransfer {
    HandoffTransfer {
        from: identity(),
        to: binding(),
        record: completed_record(),
        cursor: ContinuationCursor {
            recovery_ref: Some("execution".to_string()),
            predecessor_id: None,
            relation: Some(LinkRelation::First),
            resume_seq: 2,
            last_group_message_id: None,
        },
        window: WindowAssembler::new(Capacity::default()).assemble(&[]),
        next_goal: "summarise the run".to_string(),
        shared_files: vec![SharedFileEntry {
            path: "report.md".to_string(),
            share_mode: ShareMode::Upload,
            caption: None,
            sha256: None,
            size: None,
        }],
    }
}

#[tokio::test]
async fn crash_reopen_resume_rebuilds_cursor_window_and_digest() {
    let root = tempfile::TempDir::new().unwrap();
    let binding = binding();
    let identity =
        ExecutionIdentity::new("ns", "agent", "session", "req1", "exec-1", "source").unwrap();
    let log_dir = root.path().join("logs").join(identity.directory_name());
    {
        let log = ExecutionLog::open_with_clock(
            &log_dir,
            identity.clone(),
            LogCapacity::default(),
            frozen_clock(),
        )
        .await
        .unwrap();
        let arguments = r#"{"path":"notes.txt","text":"hello"}"#;
        log.append(EntryPayload::execution_started("do work"))
            .await
            .unwrap();
        log.append(EntryPayload::tool_intent(
            "call_1",
            "write_file",
            args_summary(arguments),
            params_digest(arguments).unwrap(),
            operation_id(&identity, "call_1"),
        ))
        .await
        .unwrap();
        log.append(EntryPayload::tool_completed(
            "call_1",
            "write_file",
            ToolStatus::Succeeded,
            args_summary(arguments),
            result_summary(&serde_json::json!({"ok": true})),
        ))
        .await
        .unwrap();
        log.append(EntryPayload::model_turn(
            "mc_1",
            "done",
            ModelUsage::known(10, 5),
        ))
        .await
        .unwrap();
        log.append(EntryPayload::execution_final(FinalStatus::Completed, None))
            .await
            .unwrap();
    }
    // The block ended, so the handle is gone: the reopen below is a fresh process reading disk.

    let store = FileSessionStore::new(root.path().join("sessions"));
    let mut record = PersistentSessionRecord::new(&binding).unwrap();
    record.messages = vec![BaseMessage::user("do work")];
    let record = record
        .with_ending(RecordEnding {
            outcome: "completed".to_string(),
            request_id: "req1".to_string(),
            execution_id: "exec-1".to_string(),
            ..RecordEnding::default()
        })
        .unwrap();
    store.save(&record).await.unwrap();
    let links = FileLinkStore::new(root.path().join("links"));
    links
        .write_link(&ExecutionLink::from_identity(&identity, None, LinkRelation::First).unwrap())
        .await
        .unwrap();

    let continuation = Continuation::new("exec-1", Some("keep going"), 4).unwrap();
    let plan = resume(
        &store,
        &RootLogSource::new(root.path().join("logs"), LogCapacity::default())
            .with_clock(frozen_clock()),
        &links,
        &binding,
        "req2",
        &continuation,
        &WindowAssembler::new(Capacity::default()),
    )
    .await
    .unwrap();

    assert_eq!(plan.cursor.resume_seq, 6, "five entries, so continue at 6");
    assert_eq!(plan.cursor.recovery_ref.as_deref(), Some("exec-1"));
    assert_eq!(plan.cursor.relation, Some(LinkRelation::First));
    assert_eq!(plan.record.continuation, Some(plan.cursor.clone()));
    assert!(!plan.window.messages.is_empty(), "the window is rebuilt");
    assert!(
        plan.window
            .messages
            .iter()
            .any(|message| message.text().contains("do work")),
        "the recovery journal is the window source"
    );
    assert_eq!(plan.digest.entries, 5);
    assert_eq!(plan.digest.last_seq, Some(5));
    assert_eq!(plan.digest.input_tokens, 10);
    assert_eq!(plan.chain.len(), 1);
    assert!(plan.background.contains("exec-1"), "{}", plan.background);
    assert!(plan.background.contains("续做规则"), "{}", plan.background);
    assert!(
        plan.background.contains("execution_final"),
        "the preview lists the journalled lifecycle entry"
    );
}

#[tokio::test]
async fn handoff_moves_record_window_and_files_to_a_successor() {
    let target = LocalHandoff::new(frozen_clock());
    let transfer = transfer();
    let receipt = hand_off(&target, transfer.clone()).await.unwrap();

    assert_eq!(receipt.successor_session, binding());
    assert_eq!(receipt.accepted_entries, 1);
    assert_eq!(receipt.transferred_at, "2023-11-14T22:13:20.000Z");
    assert_eq!(receipt.successor_execution_id, None);
    assert_eq!(target.transport(), HandoffTransport::InProcess);

    assert_eq!(
        target.take(),
        Some(transfer),
        "the state is handed over once"
    );
    assert!(target.take().is_none(), "a second take finds nothing");
}

#[tokio::test]
async fn share_bundle_round_trip_supports_handoff_files() {
    let dir = tempfile::TempDir::new().unwrap();
    let report = dir.path().join("report.md");
    let chart = dir.path().join("chart.bin");
    tokio::fs::write(&report, b"# report\n").await.unwrap();
    tokio::fs::write(&chart, [0_u8, 1, 2]).await.unwrap();
    let service = ShareFile::new(dir.path().join("bundles")).with_clock(frozen_clock());
    let files = vec![
        SharedFileEntry {
            path: "report.md".to_string(),
            share_mode: ShareMode::Upload,
            caption: Some("the report".to_string()),
            sha256: None,
            size: None,
        },
        SharedFileEntry {
            path: "out/chart.bin".to_string(),
            share_mode: ShareMode::Attachment,
            caption: None,
            sha256: None,
            size: None,
        },
    ];
    let sources = BTreeMap::from([
        ("report.md".to_string(), report),
        ("out/chart.bin".to_string(), chart),
    ]);
    let bundle = service
        .export("bundle-1", &binding(), &files, &sources)
        .await
        .unwrap();
    assert_eq!(bundle.exported_at, "2023-11-14T22:13:20.000Z");
    assert_eq!(bundle.files.len(), 2);

    let imported = service.import_bundle("bundle-1").await.unwrap();
    assert_eq!(imported.bundle, bundle);
    assert_eq!(imported.files.len(), 2);
    for (entry, path) in &imported.files {
        assert!(path.exists(), "{} has a payload", entry.path);
        assert_eq!(entry.sha256.as_deref().map(str::len), Some(64));
        assert!(entry.size.unwrap() > 0);
    }
    assert_eq!(
        tokio::fs::read(&imported.files[0].1).await.unwrap(),
        b"# report\n"
    );

    tokio::fs::write(&imported.files[0].1, b"tampered")
        .await
        .unwrap();
    let error = service.import_bundle("bundle-1").await.unwrap_err();
    assert!(matches!(error, Error::Validation(_)), "got {error:?}");
    assert!(
        error
            .to_string()
            .contains("does not match its recorded digest"),
        "got {error}"
    );
}

#[tokio::test]
async fn emitter_and_journal_are_independent_sequences() {
    let identity = identity();
    let sink = Arc::new(InMemorySink::new());
    let emitter = Emitter::new(identity.clone(), sink.clone(), frozen_clock());
    let started = emitter
        .emit(PersistentEvent::ExecutionStarted {
            goal: "go".to_string(),
            decision_event_id: None,
        })
        .await
        .unwrap();
    let ended = emitter
        .emit(PersistentEvent::ExecutionFinal {
            status: FinalStatus::Completed,
            reason: None,
            output_refs: Vec::new(),
        })
        .await
        .unwrap();

    assert_eq!(started.local_seq, 1);
    assert_eq!(ended.local_seq, 2);
    assert_eq!(started.event_id, identity.event_id(1));
    assert_eq!(ended.event_id, identity.event_id(2));
    assert_ne!(
        started.event_id, ended.event_id,
        "ids are sequence-specific"
    );
    assert_eq!(started.occurred_at, "2023-11-14T22:13:20.000Z");
    assert_eq!(sink.events(), vec![started, ended]);
    assert_eq!(emitter.last_seq(), 2);

    let dir = tempfile::TempDir::new().unwrap();
    let log =
        ExecutionLog::open_with_clock(dir.path(), identity, LogCapacity::default(), frozen_clock())
            .await
            .unwrap();
    let entry = log.append(EntryPayload::note("independent")).await.unwrap();
    assert_eq!(entry.seq, 1, "the journal numbers from one on its own");
    assert_eq!(log.next_seq().await, 2);
    assert_eq!(log.replay().await.unwrap().cursor.generation, 1);
}

#[tokio::test]
async fn unsupported_paths_are_typed() {
    let error = hand_off(&RemoteHandoff, transfer()).await.unwrap_err();
    match error {
        Error::Unsupported(message) => assert_eq!(
            message,
            "persistent agent handoff over a remote transport is not implemented; use an \
             in-process HandoffTarget"
        ),
        other => panic!("expected a typed refusal, got {other:?}"),
    }

    let error = hand_off(&DistributedHandoff, transfer()).await.unwrap_err();
    match error {
        Error::Unsupported(message) => assert_eq!(
            message,
            "persistent agent handoff over a distributed transport is not implemented; use an \
             in-process HandoffTarget"
        ),
        other => panic!("expected a typed refusal, got {other:?}"),
    }

    let error = UnsupportedModelSummarizer.summarize(&[]).unwrap_err();
    match error {
        Error::Unsupported(message) => assert_eq!(
            message,
            "persistent window summarisation requires an injected WindowSummarizer; the \
             model-backed summariser is not implemented"
        ),
        other => panic!("expected a typed refusal, got {other:?}"),
    }
    assert_eq!(HandoffTransport::InProcess.reason(), "");
}

#[tokio::test]
async fn layout_catalogue_and_resume_round_trip_from_disk() {
    let dir = tempfile::TempDir::new().unwrap();
    let layout = PersistentAgentLayout::new(dir.path());
    let identity =
        ExecutionIdentity::new("ns", "agent", "session", "request", "exec-1", "source").unwrap();

    {
        let log = ExecutionLog::open_with_clock(
            layout.logs_root().join(identity.directory_name()),
            identity.clone(),
            LogCapacity::default(),
            frozen_clock(),
        )
        .await
        .unwrap();
        log.append(EntryPayload::execution_started("do work"))
            .await
            .unwrap();
        log.append(EntryPayload::model_turn(
            "mc_1",
            "done",
            ModelUsage::known(10, 5),
        ))
        .await
        .unwrap();
    }

    let store = FileSessionStore::new(layout.sessions_root());
    let mut record = PersistentSessionRecord::new(&binding()).unwrap();
    record.messages = vec![BaseMessage::user("do work")];
    let record = record
        .with_ending(RecordEnding {
            outcome: "completed".to_string(),
            request_id: "request".to_string(),
            execution_id: "exec-1".to_string(),
            ..RecordEnding::default()
        })
        .unwrap();
    store.save(&record).await.unwrap();

    let links = FileLinkStore::new(layout.links_root());
    links
        .write_link(&ExecutionLink::from_identity(&identity, None, LinkRelation::First).unwrap())
        .await
        .unwrap();

    let catalog = ExecutionCatalog::new(layout.logs_root(), LogCapacity::default());
    let summaries = catalog.list().await.unwrap();
    assert_eq!(summaries.len(), 1);
    assert_eq!(summaries[0].execution_id, "exec-1");
    assert_eq!(summaries[0].entries, 2);
    assert_eq!(summaries[0].last_seq, Some(2));
    assert!(summaries[0].usable);
    assert_eq!(summaries[0].created_at, "2023-11-14T22:13:20.000Z");
    assert_eq!(
        catalog.open("exec-1").await.unwrap().identity(),
        &identity,
        "the catalogue opens the very journal it listed"
    );

    let sessions = store.list().await.unwrap();
    assert_eq!(sessions.len(), 1);
    assert_eq!(sessions[0].binding, binding());
    assert!(!sessions[0].waiting);
    assert!(sessions[0].has_ending);
    assert!(!sessions[0].has_checkpoint);
    assert_eq!(sessions[0].state_path, store.path_for(&binding()));

    let plan = resume(
        &store,
        &RootLogSource::new(layout.logs_root(), LogCapacity::default()).with_clock(frozen_clock()),
        &links,
        &binding(),
        "request-2",
        &Continuation::new("exec-1", None, 2).unwrap(),
        &WindowAssembler::new(Capacity::default()),
    )
    .await
    .unwrap();
    let encoded = serde_json::to_string(&plan).unwrap();
    let decoded: ResumePlan = serde_json::from_str(&encoded).unwrap();
    assert!(
        decoded
            .seed_messages()
            .iter()
            .any(|message| message.text().contains("do work")),
        "the reconstructed messages still carry the run's text"
    );
    assert_eq!(
        decoded.cursor.resume_seq, 3,
        "two entries, so continue at 3"
    );
    assert_eq!(decoded.digest.entries, 2);
    assert_eq!(decoded, plan, "the JSON view is lossless");
}
