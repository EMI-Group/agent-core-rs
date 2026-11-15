//! Persistent-agent journal durability and resume, driven through the CLI-facing surface.
//!
//! A long-lived coding session survives a crash because its execution journal is the durability
//! boundary: a resume replays what reached disk instead of trusting an in-memory transcript. This
//! binary pins that contract from the outside, through `openjiuwen::harness::persistent_agent`
//! (the path the umbrella exposes and the CLI uses):
//!
//! 1. **A completed journal survives a crash and a reopen** — five appended entries, a rotation in
//!    the middle, then a fresh handle on the same directory: the sequence numbers, the rotated
//!    segment, the cursor, the per-kind counts and the accumulated token counters all read back.
//! 2. **`resume` rebuilds cursor, window and chain** — the saved session record, the journal and
//!    the link store are the only inputs; the returned plan continues the journalled sequence,
//!    rebuilds a non-empty model window from the journal, traces the execution chain and points the
//!    record at the new cursor without rewriting the stored one.
//! 3. **A damaged journal degrades instead of failing** — a retried append, a conflicting rewrite
//!    and a torn trailing line are reported by a read that still yields the surviving entries, and
//!    the journal keeps accepting appends past the damage without reusing a sequence number.
//! 4. **Deferred handoff transports stay typed** — the remote and distributed transports answer
//!    `Error::Unsupported` with their documented reason rather than silently degrading.
//!
//! Both halves of the damage scenario are expressible through the public surface: an
//! [`ExecutionEntry`] serializes to exactly the line the journal writes, so the duplicate and the
//! conflicting rewrite are hand-written in that same encoding instead of being injected through a
//! private test hook.
//!
//! Everything here is offline and deterministic: one `tempfile` root per scenario, the harness's
//! [`ManualClock`] frozen at a single instant (so every persisted timestamp is the injected one and
//! no `chrono` type is ever named), local file IO only, no model and no network.
#![allow(clippy::unwrap_used, clippy::expect_used)]
mod support;
use anyhow::{bail, ensure, Result};
use openjiuwen::core::common::error::{Error, ErrorKind};
use openjiuwen::core::foundation::prompt::message::BaseMessage;
use openjiuwen::harness::persistent_agent::{
    hand_off, resume, Capacity, Continuation, ContinuationCursor, DistributedHandoff, EntryKind,
    EntryPayload, ExecutionEntry, ExecutionIdentity, ExecutionLink, ExecutionLinkSource,
    ExecutionLog, FileLinkStore, FileSessionStore, FinalStatus, HandoffTransfer, HandoffTransport,
    LinkRelation, LogCapacity, ManualClock, ModelUsage, PersistentSessionRecord, RecordEnding,
    RemoteHandoff, RootLogSource, SessionBinding, SessionStateStore, WindowAssembler,
    CONTINUATION_RULES, LOG_ENCODING, LOG_ENTRY_FILE, LOG_META_FILE,
};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use support::within;
/// Frozen instant every fixture clock reports: `2023-11-14T22:13:20.000Z`.
const NOW_MS: i64 = 1_700_000_000_000;
/// The stamp the injected clock writes into every entry and journal: [`NOW_MS`] as RFC 3339.
const NOW_RFC3339: &str = "2023-11-14T22:13:20.000Z";
/// Agent id of every fixture binding.
const AGENT: &str = "agent";
/// Session id of every fixture binding.
const SESSION: &str = "session";
/// The task input every fixture execution starts with.
const PROMPT: &str = "do the work";
/// A manual clock frozen at [`NOW_MS`], so every persisted timestamp is deterministic.
fn clock() -> Arc<ManualClock> {
    Arc::new(ManualClock::new(NOW_MS))
}
/// The `(agent, session)` pair every fixture targets.
fn binding() -> Result<SessionBinding> {
    Ok(SessionBinding::new(AGENT, SESSION)?)
}
/// The identity of execution `execution`, all of whose runs belong to one request.
fn identity(execution: &str) -> Result<ExecutionIdentity> {
    Ok(ExecutionIdentity::new(
        "ns", AGENT, SESSION, "req1", execution, "source",
    )?)
}
/// The journal directory of `execution` under `root/logs`, spelled the way [`RootLogSource`] reads
/// it back.
fn log_dir(root: &Path, execution: &str) -> Result<PathBuf> {
    Ok(root
        .join("logs")
        .join(identity(execution)?.directory_name()))
}
/// Opens (and creates) the journal of `execution`, bounded so a stalled open fails the scenario.
async fn open_log(dir: &Path, execution: &str) -> Result<ExecutionLog> {
    Ok(within(
        "open the execution journal",
        ExecutionLog::open_with_clock(dir, identity(execution)?, LogCapacity::default(), clock()),
    )
    .await?)
}
/// Appends one payload, bounded for the same reason as [`open_log`].
async fn append(log: &ExecutionLog, payload: EntryPayload) -> Result<ExecutionEntry> {
    Ok(within("append to the execution journal", log.append(payload)).await?)
}
/// Writes a fresh journal of `execution` holding `payloads` in order.
async fn write_journal(
    root: &Path,
    execution: &str,
    payloads: Vec<EntryPayload>,
) -> Result<PathBuf> {
    let dir = log_dir(root, execution)?;
    let log = open_log(&dir, execution).await?;
    for payload in payloads {
        append(&log, payload).await?;
    }
    Ok(dir)
}
/// Renders `entry` exactly as the journal writes it: one compact JSON line without the newline.
fn line(entry: &ExecutionEntry) -> Result<String> {
    Ok(serde_json::to_string(entry)?)
}
/// The sequence numbers of `entries`, in read order.
fn seqs(entries: &[ExecutionEntry]) -> Vec<u64> {
    entries.iter().map(|entry| entry.seq).collect()
}
/// The completed execution of the durability scenario: three entries, a rotation, then two more.
async fn journalled_execution(root: &Path) -> Result<(PathBuf, Vec<ExecutionEntry>)> {
    let dir = log_dir(root, "exec-1")?;
    let execution = identity("exec-1")?;
    let log = open_log(&dir, "exec-1").await?;
    ensure!(
        log.dir() == dir.as_path(),
        "the journal owns the directory it was opened at"
    );
    ensure!(
        log.identity() == &execution,
        "the journal is bound to its execution"
    );
    ensure!(
        log.capacity() == LogCapacity::default(),
        "the default retention applies"
    );

    let mut entries = vec![
        append(&log, EntryPayload::execution_started(PROMPT)).await?,
        append(&log, EntryPayload::note("first step")).await?,
        append(
            &log,
            EntryPayload::model_turn("mc_1", "done", ModelUsage::known(10, 5)),
        )
        .await?,
    ];
    // Rotating mid-run must neither lose entries nor renumber them.
    within("rotate the journal", log.rotate_now()).await?;
    entries.push(append(&log, EntryPayload::note("second step")).await?);
    entries.push(
        append(
            &log,
            EntryPayload::execution_final(FinalStatus::Completed, None),
        )
        .await?,
    );
    ensure!(
        within("read the next sequence", log.next_seq()).await == 6,
        "five entries, so the next append is 6"
    );
    Ok((dir, entries))
}
/// Scenario 1: what one finished execution wrote survives the handle and the process.
#[tokio::test]
#[allow(clippy::too_many_lines)]
async fn a_completed_journal_survives_a_crash_and_a_reopen() -> Result<()> {
    let root = tempfile::TempDir::new()?;
    let (dir, entries) = journalled_execution(root.path()).await?;

    // The handle is gone, so everything below is what actually reached disk.
    ensure!(
        seqs(&entries) == vec![1, 2, 3, 4, 5],
        "the journal numbers from one without gaps: {:?}",
        seqs(&entries)
    );
    ensure!(
        entries.iter().all(|entry| entry.occurred_at == NOW_RFC3339),
        "every entry is stamped by the injected clock"
    );
    ensure!(
        entries[2].kind() == EntryKind::ModelTurn,
        "the model turn keeps its kind"
    );

    // The run spans two segments plus the metadata file: 3 archived lines and 2 active ones.
    let archived = std::fs::read_to_string(dir.join("entries.1.jsonl"))?;
    ensure!(
        archived.lines().count() == 3,
        "the rotated segment holds the three pre-rotation entries, got {}",
        archived.lines().count()
    );
    let active = std::fs::read_to_string(dir.join(LOG_ENTRY_FILE))?;
    ensure!(
        active.lines().count() == 2,
        "the active segment holds the two post-rotation entries, got {}",
        active.lines().count()
    );
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join(LOG_META_FILE))?)?;
    ensure!(
        meta["encoding"] == LOG_ENCODING,
        "the metadata states the journal encoding, got {}",
        meta["encoding"]
    );
    ensure!(
        meta["generation"] == 2,
        "the rotation advanced the segment counter, got {}",
        meta["generation"]
    );

    let reopened = open_log(&dir, "exec-1").await?;
    let replay = within("replay the journal", reopened.replay()).await?;
    ensure!(
        replay.entries.len() == 5,
        "a rotation keeps every entry, got {}",
        replay.entries.len()
    );
    ensure!(!replay.truncated, "the default capacity retains the run");
    ensure!(
        seqs(&replay.entries) == vec![1, 2, 3, 4, 5],
        "the segments replay in sequence order: {:?}",
        seqs(&replay.entries)
    );
    ensure!(
        replay.cursor.next_seq == 6,
        "the next append continues at 6, got {}",
        replay.cursor.next_seq
    );
    ensure!(
        replay.cursor.last_seq == 5,
        "the newest entry is 5, got {}",
        replay.cursor.last_seq
    );
    ensure!(
        replay.cursor.generation == 2,
        "the reopen resumes the rotated generation, got {}",
        replay.cursor.generation
    );
    ensure!(
        replay.digest.first_seq == Some(1) && replay.digest.last_seq == Some(5),
        "sequence numbers survive rotation: {:?}",
        replay.digest
    );
    ensure!(
        replay.digest.entries == 5,
        "the digest counts the whole run, got {}",
        replay.digest.entries
    );
    ensure!(
        replay.digest.input_tokens == 10
            && replay.digest.output_tokens == 5
            && replay.digest.unknown_usage == 0,
        "the digest totals the model usage: {:?}",
        replay.digest
    );
    ensure!(
        replay.digest.counts.get(&EntryKind::Note) == Some(&2)
            && replay.digest.counts.get(&EntryKind::ExecutionStarted) == Some(&1)
            && replay.digest.counts.get(&EntryKind::ModelTurn) == Some(&1)
            && replay.digest.counts.get(&EntryKind::ExecutionFinal) == Some(&1),
        "the digest counts every kind: {:?}",
        replay.digest.counts
    );

    // A plain read agrees with the replay, and the durability boundary reports no damage.
    let snapshot = within("read the journal", reopened.read()).await?;
    ensure!(
        snapshot.entries == replay.entries,
        "the read and the replay see the same entries"
    );
    ensure!(
        snapshot.is_usable() && !snapshot.is_truncated(),
        "a clean journal is usable and untruncated: {snapshot:?}"
    );
    ensure!(
        snapshot.unusable.is_none() && snapshot.skipped_invalid == 0,
        "nothing was skipped: {snapshot:?}"
    );
    ensure!(
        snapshot.digest() == replay.digest,
        "the snapshot and the replay report the same digest"
    );
    ensure!(
        within("digest the journal", reopened.digest()).await? == replay.digest,
        "the digest accessor agrees with the replay"
    );
    Ok(())
}
/// Scenario 2: a crashed session is continued from the record, the journal and the links alone.
#[tokio::test]
#[allow(clippy::too_many_lines)]
async fn resume_rebuilds_the_cursor_window_and_execution_chain() -> Result<()> {
    let root = tempfile::TempDir::new()?;
    // Two executions of one request: `exec-0` ran first, `exec-1` continued it and crashed after
    // journalling its own final entry (the process died before the next request was persisted).
    write_journal(
        root.path(),
        "exec-0",
        vec![
            EntryPayload::execution_started("early attempt"),
            EntryPayload::execution_final(FinalStatus::Failed, Some("crashed".to_string())),
        ],
    )
    .await?;
    let resumed_dir = write_journal(
        root.path(),
        "exec-1",
        vec![
            EntryPayload::execution_started(PROMPT),
            EntryPayload::model_turn("mc_1", "interim answer", ModelUsage::known(7, 3)),
            EntryPayload::execution_final(FinalStatus::Completed, None),
        ],
    )
    .await?;
    ensure!(
        resumed_dir == log_dir(root.path(), "exec-1")?,
        "the recovery journal lives where the log source reads it"
    );

    let binding = binding()?;
    let store = FileSessionStore::new(root.path().join("sessions"));
    let mut record = PersistentSessionRecord::new(&binding)?;
    record.messages = vec![BaseMessage::user(PROMPT)];
    let record = record.with_ending(RecordEnding {
        outcome: "completed".to_string(),
        request_id: "req1".to_string(),
        execution_id: "exec-1".to_string(),
        ..RecordEnding::default()
    })?;
    store.save(&record).await?;

    let links = FileLinkStore::new(root.path().join("links"));
    for (execution, predecessor, relation) in [
        ("exec-0", None, LinkRelation::First),
        ("exec-1", Some("exec-0"), LinkRelation::Continuation),
    ] {
        links
            .write_link(&ExecutionLink::from_identity(
                &identity(execution)?,
                predecessor,
                relation,
            )?)
            .await?;
    }
    ensure!(
        links.link_path("exec-1").exists(),
        "the link is written where the resume resolves it"
    );

    let logs =
        RootLogSource::new(root.path().join("logs"), LogCapacity::default()).with_clock(clock());
    let continuation = Continuation::new("exec-1", Some("keep going"), 4)?;
    let plan = within(
        "resume the crashed session",
        resume(
            &store,
            &logs,
            &links,
            &binding,
            "req2",
            &continuation,
            &WindowAssembler::new(Capacity::default()),
        ),
    )
    .await?;

    ensure!(
        plan.cursor.resume_seq == 4,
        "the plan continues after the three journalled entries, got {}",
        plan.cursor.resume_seq
    );
    ensure!(
        plan.cursor.recovery_ref.as_deref() == Some("exec-1"),
        "the plan points at the journalled execution"
    );
    ensure!(
        plan.cursor.predecessor_id.as_deref() == Some("exec-0"),
        "the plan records the predecessor it traced: {:?}",
        plan.cursor
    );
    ensure!(
        plan.cursor.relation == Some(LinkRelation::Continuation),
        "a continuation follows its predecessor: {:?}",
        plan.cursor
    );
    ensure!(
        plan.digest.entries == 3 && plan.digest.last_seq == Some(3),
        "the digest covers the recovery execution only: {:?}",
        plan.digest
    );
    ensure!(
        plan.digest.input_tokens == 7 && plan.digest.output_tokens == 3,
        "the plan carries the recovery usage: {:?}",
        plan.digest
    );
    ensure!(
        !plan.window.messages.is_empty(),
        "the recovery journal rebuilds a non-empty window"
    );
    ensure!(
        plan.window
            .messages
            .iter()
            .any(|message| message.text().contains(PROMPT)),
        "the window is assembled from the journal, not from an empty default"
    );
    ensure!(
        plan.background.contains("exec-1") && plan.background.contains("exec-0"),
        "the continuation prompt names the traced chain: {}",
        plan.background
    );
    ensure!(
        plan.background.contains(CONTINUATION_RULES),
        "the continuation prompt ends with the continuation rules"
    );

    ensure!(
        plan.chain.len() == 2,
        "the chain traces exec-1 back to exec-0, got {}",
        plan.chain.len()
    );
    ensure!(
        plan.chain[0].execution_id == "exec-1"
            && plan.chain[0].relation == LinkRelation::Continuation
            && plan.chain[0].predecessor_id.as_deref() == Some("exec-0"),
        "the chain starts at the recovery execution: {:?}",
        plan.chain[0]
    );
    ensure!(
        plan.chain[1].execution_id == "exec-0"
            && plan.chain[1].relation == LinkRelation::First
            && plan.chain[1].predecessor_id.is_none(),
        "the chain ends at the first execution: {:?}",
        plan.chain[1]
    );
    ensure!(
        plan.chain.iter().all(|link| link.matches(&binding)),
        "every traced link belongs to the resumed session"
    );

    ensure!(
        plan.record.continuation.as_ref() == Some(&plan.cursor),
        "the resumed record keeps the new cursor"
    );
    ensure!(
        plan.record
            .messages
            .iter()
            .any(|message| message.text().contains(PROMPT)),
        "the persisted window still carries the original prompt"
    );
    ensure!(
        plan.record
            .ending
            .as_ref()
            .map(|ending| ending.outcome.as_str())
            == Some("completed"),
        "the earlier ending is preserved: {:?}",
        plan.record.ending
    );

    // The plan is a proposal: resuming never rewrites the stored record.
    let stored = store
        .load(&binding)
        .await?
        .expect("the saved record stays on disk");
    ensure!(
        stored.continuation.is_none(),
        "the caller persists the plan; the resume itself does not"
    );
    Ok(())
}
/// Scenario 3: damage reaches the read as a report, never as a broken journal.
#[tokio::test]
#[allow(clippy::too_many_lines)]
async fn a_damaged_journal_degrades_instead_of_failing() -> Result<()> {
    let root = tempfile::TempDir::new()?;
    let identity = identity("exec-damaged")?;
    let dir = log_dir(root.path(), "exec-damaged")?;
    {
        let log = open_log(&dir, "exec-damaged").await?;
        append(&log, EntryPayload::execution_started(PROMPT)).await?;
        append(&log, EntryPayload::note("first step")).await?;
        append(&log, EntryPayload::note("second step")).await?;
    }

    // Damage the journal the way a crash mid-append and a retried write do: an exact copy of
    // seq 2 (the retry that reached disk twice), a rewrite of seq 3 that disagrees with the
    // committed one, and a truncated trailing line (the append that never finished).
    let path = dir.join(LOG_ENTRY_FILE);
    let mut damaged = std::fs::read_to_string(&path)?;
    for entry in [
        ExecutionEntry::new(2, NOW_RFC3339, EntryPayload::note("first step")),
        ExecutionEntry::new(3, NOW_RFC3339, EntryPayload::note("a later rewrite")),
    ] {
        damaged.push_str(&line(&entry)?);
        damaged.push('\n');
    }
    damaged.push_str(
        r#"{"seq":4,"occurred_at":"2023-11-14T22:13:20.000Z","kind":"note","text":"never fini"#,
    );
    damaged.push('\n');
    std::fs::write(&path, damaged)?;

    let log = open_log(&dir, "exec-damaged").await?;
    let snapshot = within("read the damaged journal", log.read()).await?;
    ensure!(
        snapshot.identity == identity,
        "the reopened journal still belongs to its execution"
    );
    ensure!(
        snapshot.is_usable(),
        "one surviving entry keeps the journal usable: {snapshot:?}"
    );
    ensure!(
        snapshot.unusable.is_none(),
        "a damaged but readable journal is not unusable: {snapshot:?}"
    );
    ensure!(
        snapshot.skipped_invalid == 1,
        "the torn line is skipped and counted, got {}",
        snapshot.skipped_invalid
    );
    ensure!(
        snapshot.duplicates_removed == 1,
        "the retried append is collapsed, got {}",
        snapshot.duplicates_removed
    );
    ensure!(
        snapshot.conflicting_seqs == vec![3],
        "the disagreeing rewrite drops its whole group, got {:?}",
        snapshot.conflicting_seqs
    );
    ensure!(
        snapshot.retained_out == 0,
        "nothing was dropped by retention, got {}",
        snapshot.retained_out
    );
    ensure!(
        seqs(&snapshot.entries) == vec![1, 2],
        "the valid entries are preserved in order: {:?}",
        seqs(&snapshot.entries)
    );
    ensure!(
        snapshot.missing_ranges.is_empty(),
        "the dropped group is the tail of the readable span, so no interior hole is reported yet, \
         got {:?}",
        snapshot.missing_ranges
    );
    ensure!(
        snapshot.is_truncated(),
        "a degraded read says so instead of hiding the damage"
    );

    let replay = within("replay the damaged journal", log.replay()).await?;
    ensure!(replay.truncated, "the replay reports the damage too");
    ensure!(
        replay.cursor.next_seq == 4,
        "the writer never reuses a scanned sequence, got {}",
        replay.cursor.next_seq
    );
    ensure!(
        replay.digest.entries == 2 && replay.digest.last_seq == Some(2),
        "the digest counts only the surviving entries: {:?}",
        replay.digest
    );
    ensure!(
        replay.digest.skipped_invalid == 1 && replay.digest.duplicates_removed == 1,
        "the digest repeats both damage counters: {:?}",
        replay.digest
    );
    ensure!(
        replay.digest.summary().contains("skipped_invalid=1")
            && replay.digest.summary().contains("duplicates_removed=1"),
        "the digest summary names the damage: {}",
        replay.digest.summary()
    );

    // The journal keeps working: the next append continues past the damage instead of renumbering,
    // and every surviving entry stays readable beside it.
    let appended = append(&log, EntryPayload::note("after the damage")).await?;
    ensure!(
        appended.seq == 4,
        "the append continues after the damaged sequence, got {}",
        appended.seq
    );
    ensure!(
        appended.occurred_at == NOW_RFC3339,
        "the repaired journal keeps stamping through the injected clock"
    );
    let after = within("re-read the repaired journal", log.read()).await?;
    ensure!(
        seqs(&after.entries) == vec![1, 2, 4],
        "the new entry lands beside the survivors: {:?}",
        seqs(&after.entries)
    );
    ensure!(
        after.conflicting_seqs == vec![3],
        "the dropped group stays dropped, got {:?}",
        after.conflicting_seqs
    );
    ensure!(
        after.missing_ranges == vec![(3, 3)],
        "continuing past the dropped sequence reports the interior hole, got {:?}",
        after.missing_ranges
    );
    ensure!(
        after.skipped_invalid == 1,
        "the torn line stays counted, got {}",
        after.skipped_invalid
    );
    ensure!(
        after.digest().entries == 3 && after.digest().last_seq == Some(4),
        "the digest now covers the appended entry: {:?}",
        after.digest()
    );
    Ok(())
}
/// Scenario 4: the deferred handoff transports answer with their documented typed refusal.
#[tokio::test]
async fn deferred_handoff_transports_stay_typed_unsupported() -> Result<()> {
    let root = tempfile::TempDir::new()?;
    let dir = log_dir(root.path(), "exec-1")?;
    let log = open_log(&dir, "exec-1").await?;
    let entry = append(&log, EntryPayload::note("handoff payload")).await?;
    ensure!(entry.seq == 1, "the journal numbers from one");
    ensure!(
        within("read the journal cursor", log.cursor())
            .await
            .next_seq
            == 2,
        "and the cursor advances"
    );

    let binding = binding()?;
    let transfer = HandoffTransfer {
        from: identity("exec-1")?,
        to: binding.clone(),
        record: PersistentSessionRecord::new(&binding)?,
        cursor: ContinuationCursor {
            recovery_ref: Some("exec-1".to_string()),
            predecessor_id: None,
            relation: Some(LinkRelation::First),
            resume_seq: 2,
            last_group_message_id: None,
        },
        window: WindowAssembler::new(Capacity::default()).assemble(&[]),
        next_goal: "summarise the run".to_string(),
        shared_files: Vec::new(),
    };

    let remote = hand_off(&RemoteHandoff, transfer.clone())
        .await
        .expect_err("the remote transport is deferred");
    ensure!(
        remote.kind() == ErrorKind::Unsupported,
        "the remote refusal carries the Unsupported kind, got {remote:?}"
    );
    match remote {
        Error::Unsupported(message) => {
            ensure!(!message.is_empty(), "the refusal carries a reason");
            ensure!(
                message == HandoffTransport::Remote.reason(),
                "the reason is the transport's documented one, got {message}"
            );
            ensure!(
                message.contains("not implemented") && message.contains("in-process HandoffTarget"),
                "the reason names what is missing and what to use instead: {message}"
            );
        }
        other => bail!("expected a typed refusal, got {other:?}"),
    }

    let distributed = hand_off(&DistributedHandoff, transfer)
        .await
        .expect_err("the distributed transport is deferred");
    ensure!(
        distributed.kind() == ErrorKind::Unsupported,
        "the distributed refusal carries the Unsupported kind, got {distributed:?}"
    );
    match distributed {
        Error::Unsupported(message) => {
            ensure!(!message.is_empty(), "the refusal carries a reason");
            ensure!(
                message == HandoffTransport::Distributed.reason(),
                "the reason is the transport's documented one, got {message}"
            );
            ensure!(
                message.contains("not implemented") && message.contains("in-process HandoffTarget"),
                "the reason names what is missing and what to use instead: {message}"
            );
        }
        other => bail!("expected a typed refusal, got {other:?}"),
    }

    // The in-process transport is the supported one, so it has nothing to refuse.
    ensure!(
        HandoffTransport::InProcess.reason().is_empty(),
        "the in-process transport is implemented"
    );
    Ok(())
}
