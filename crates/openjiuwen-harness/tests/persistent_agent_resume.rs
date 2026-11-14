//! Offline integration coverage for the persistent-agent surface, driven through the crate root.
//!
//! The suites replay the two flows a crashed coding session depends on — resuming a journalled
//! execution from disk, and moving its files through a share bundle — plus the typed refusals of
//! the deferred transports. Everything runs against a `tempfile` root with an injected
//! [`ManualClock`], so no network, wall clock or model is involved.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;
use std::sync::Arc;

use anyhow::{ensure, Result};
use openjiuwen_common::error::{Error, ErrorKind};
use openjiuwen_foundation::prompt::message::BaseMessage;
use openjiuwen_harness::{
    hand_off, resume, Capacity, Continuation, DistributedHandoff, EntryPayload, ExecutionIdentity,
    ExecutionLink, ExecutionLinkSource, ExecutionLog, FileLinkStore, FileSessionStore, FinalStatus,
    LinkRelation, LogCapacity, ManualClock, ModelUsage, PersistentSessionRecord, RecordEnding,
    RemoteHandoff, RootLogSource, SessionBinding, SessionStateStore, ShareFile, ShareMode,
    SharedFileEntry, WindowAssembler,
};

/// Frozen instant every fixture clock reports: `2023-11-14T22:13:20.000Z`.
const NOW_MS: i64 = 1_700_000_000_000;

/// A manual clock frozen at [`NOW_MS`], so every persisted timestamp is deterministic.
fn clock() -> Arc<ManualClock> {
    Arc::new(ManualClock::new(NOW_MS))
}

/// The `(agent, session)` pair every fixture targets.
fn binding() -> Result<SessionBinding> {
    Ok(SessionBinding::new("agent", "session")?)
}

/// The source execution the resume fixtures recover.
fn identity() -> Result<ExecutionIdentity> {
    Ok(ExecutionIdentity::new(
        "ns", "agent", "session", "req1", "exec-1", "source",
    )?)
}

/// Writes a completed journal of five entries, rotating in the middle so the journal spans two
/// segments, and returns the segment directory.
async fn journal_completed_execution(root: &std::path::Path) -> Result<std::path::PathBuf> {
    let identity = identity()?;
    let log_dir = root.join("logs").join(identity.directory_name());
    let log =
        ExecutionLog::open_with_clock(&log_dir, identity, LogCapacity::default(), clock()).await?;
    log.append(EntryPayload::execution_started("do work"))
        .await?;
    log.append(EntryPayload::note("first step")).await?;
    log.append(EntryPayload::model_turn(
        "mc_1",
        "done",
        ModelUsage::known(10, 5),
    ))
    .await?;
    // Rotating mid-run must not lose entries nor renumber them.
    log.rotate_now().await?;
    log.append(EntryPayload::note("second step")).await?;
    log.append(EntryPayload::execution_final(FinalStatus::Completed, None))
        .await?;
    ensure!(log.next_seq().await == 6, "five entries, so append at 6");
    Ok(log_dir)
}

/// Persists the session record and the link the resume resolves through.
async fn session_state(root: &std::path::Path) -> Result<(FileSessionStore, FileLinkStore)> {
    let binding = binding()?;
    let store = FileSessionStore::new(root.join("sessions"));
    let mut record = PersistentSessionRecord::new(&binding)?;
    record.messages = vec![BaseMessage::user("do work")];
    let record = record.with_ending(RecordEnding {
        outcome: "completed".to_string(),
        request_id: "req1".to_string(),
        execution_id: "exec-1".to_string(),
        ..RecordEnding::default()
    })?;
    store.save(&record).await?;
    let links = FileLinkStore::new(root.join("links"));
    links
        .write_link(&ExecutionLink::from_identity(
            &identity()?,
            None,
            LinkRelation::First,
        )?)
        .await?;
    Ok((store, links))
}

/// A crashed session reopens from disk, rebuilds its cursor, window and digest, and resumes.
#[tokio::test]
async fn a_crashed_session_resumes_from_its_journal() -> Result<()> {
    let root = tempfile::TempDir::new()?;
    let log_dir = journal_completed_execution(root.path()).await?;

    // The handle above was dropped, so everything below reads what actually reached disk.
    let segments = std::fs::read_dir(&log_dir)?
        .filter_map(std::result::Result::ok)
        .count();
    ensure!(
        segments >= 3,
        "two journal segments plus the metadata file, found {segments} entries in {log_dir:?}"
    );

    let reopened =
        ExecutionLog::open_with_clock(&log_dir, identity()?, LogCapacity::default(), clock())
            .await?;
    let replay = reopened.replay().await?;
    ensure!(replay.entries.len() == 5, "a rotation keeps every entry");
    ensure!(!replay.truncated, "the default capacity retains the run");
    ensure!(
        replay.cursor.next_seq == 6,
        "continue at 6, got {}",
        replay.cursor.next_seq
    );
    ensure!(
        replay.digest.first_seq == Some(1) && replay.digest.last_seq == Some(5),
        "sequence numbers survive rotation: {:?}",
        replay.digest
    );
    ensure!(
        replay.digest.input_tokens == 10,
        "the digest totals the model usage"
    );

    let (store, links) = session_state(root.path()).await?;
    let continuation = Continuation::new("exec-1", Some("keep going"), 4)?;
    let plan = resume(
        &store,
        &RootLogSource::new(root.path().join("logs"), LogCapacity::default()).with_clock(clock()),
        &links,
        &binding()?,
        "req2",
        &continuation,
        &WindowAssembler::new(Capacity::default()),
    )
    .await?;

    ensure!(
        plan.cursor.resume_seq == 6,
        "the plan continues the journal"
    );
    ensure!(
        plan.cursor.recovery_ref.as_deref() == Some("exec-1"),
        "the plan points at the journalled execution"
    );
    ensure!(plan.digest.entries == 5, "the plan carries the digest");
    ensure!(
        !plan.window.messages.is_empty(),
        "the recovery journal rebuilds a non-empty window"
    );
    ensure!(
        plan.window
            .messages
            .iter()
            .any(|message| message.text().contains("do work")),
        "the window is assembled from the journal, not from an empty default"
    );
    ensure!(
        plan.background.contains("exec-1"),
        "the continuation prompt names the recovered execution: {}",
        plan.background
    );
    ensure!(
        plan.record.continuation.as_ref() == Some(&plan.cursor),
        "the resumed record keeps the new cursor"
    );
    Ok(())
}

/// A share bundle round-trips its files and payload digests through the public service.
#[tokio::test]
async fn share_bundles_export_and_import_files() -> Result<()> {
    let root = tempfile::TempDir::new()?;
    let report = root.path().join("report.md");
    let chart = root.path().join("chart.bin");
    tokio::fs::write(&report, b"# report\n").await?;
    tokio::fs::write(&chart, [0_u8, 1, 2]).await?;

    let service = ShareFile::new(root.path().join("bundles")).with_clock(clock());
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
        .export("bundle-1", &binding()?, &files, &sources)
        .await?;
    ensure!(
        bundle.exported_at == "2023-11-14T22:13:20.000Z",
        "the export is stamped by the injected clock, got {}",
        bundle.exported_at
    );
    ensure!(bundle.files.len() == 2, "both files are listed");

    let imported = service.import_bundle("bundle-1").await?;
    ensure!(
        imported.bundle == bundle,
        "the imported record round-trips through JSON"
    );
    ensure!(imported.files.len() == 2, "both payloads are restored");
    let report_bytes = tokio::fs::read(&imported.files[0].1).await?;
    ensure!(report_bytes == b"# report\n", "the payload bytes match");
    ensure!(
        imported.files[0].0.sha256.as_deref().map(str::len) == Some(64),
        "the digest is recorded as sha256 hex"
    );
    ensure!(
        imported.files[0].0.size == Some(9),
        "the size is recorded from the copied bytes"
    );

    // Importing by path is the same verification path.
    let by_path = service.import(&service.bundle_path("bundle-1")).await?;
    ensure!(by_path.files.len() == 2, "the bundle file imports directly");

    // A tampered payload fails the recorded-digest check instead of importing silently.
    tokio::fs::write(&imported.files[0].1, b"tampered").await?;
    let error = service
        .import_bundle("bundle-1")
        .await
        .expect_err("a tampered payload must be refused");
    ensure!(error.kind() == ErrorKind::Validation, "got {error:?}");
    ensure!(
        error
            .to_string()
            .contains("does not match its recorded digest"),
        "got {error}"
    );
    Ok(())
}

/// The deferred handoff transports answer with their documented typed refusal.
#[tokio::test]
async fn deferred_handoff_transports_are_typed_unsupported() -> Result<()> {
    let root = tempfile::TempDir::new()?;
    let identity = identity()?;
    let log_dir = root.path().join("logs").join(identity.directory_name());
    let log =
        ExecutionLog::open_with_clock(&log_dir, identity.clone(), LogCapacity::default(), clock())
            .await?;
    let entry = log.append(EntryPayload::note("handoff payload")).await?;
    ensure!(entry.seq == 1, "the journal numbers from one");
    ensure!(log.cursor().await.next_seq == 2, "and advances");

    let transfer = openjiuwen_harness::HandoffTransfer {
        from: identity,
        to: binding()?,
        record: PersistentSessionRecord::new(&binding()?)?,
        cursor: openjiuwen_harness::ContinuationCursor {
            recovery_ref: Some("exec-1".to_string()),
            predecessor_id: None,
            relation: Some(LinkRelation::First),
            resume_seq: 1,
            last_group_message_id: None,
        },
        window: WindowAssembler::new(Capacity::default()).assemble(&[]),
        next_goal: "summarise the run".to_string(),
        shared_files: Vec::new(),
    };

    let remote = hand_off(&RemoteHandoff, transfer.clone())
        .await
        .expect_err("the remote transport is deferred");
    match remote {
        Error::Unsupported(message) => ensure!(
            message
                == "persistent agent handoff over a remote transport is not implemented; use an \
                    in-process HandoffTarget",
            "unexpected refusal message: {message}"
        ),
        other => anyhow::bail!("expected a typed refusal, got {other:?}"),
    }

    let distributed = hand_off(&DistributedHandoff, transfer)
        .await
        .expect_err("the distributed transport is deferred");
    ensure!(
        distributed.kind() == ErrorKind::Unsupported,
        "the refusal carries the Unsupported kind"
    );
    match distributed {
        Error::Unsupported(message) => ensure!(
            message
                == "persistent agent handoff over a distributed transport is not implemented; use \
                    an in-process HandoffTarget",
            "unexpected refusal message: {message}"
        ),
        other => anyhow::bail!("expected a typed refusal, got {other:?}"),
    }
    Ok(())
}
