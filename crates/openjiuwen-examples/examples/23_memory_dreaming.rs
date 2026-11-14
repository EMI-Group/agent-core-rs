//! `23_memory_dreaming` — the durable dream queue, drained one injected tick at a time.
//!
//! Python counterpart: `openjiuwen/core/memory/dreaming/orchestrator.py`, the ~100-line
//! `DreamingOrchestrator` that sweeps on a timer, floors its interval at 60 seconds, and
//! backs off while the application is busy. The Rust port splits that into pieces this
//! example drives explicitly: the queue (`DreamRepository`, here `SqliteDreamRepository`
//! over one `dream_jobs` table), the application's pass (`DreamConsolidator`), the worker
//! (`DreamPlanner::run_due`: claim a batch, consolidate, record the transition and its
//! attempt accounting), and the loop (`DreamOrchestrator::tick`, which takes `now` as a
//! parameter so the backoff ladder prints from fixed timestamps, not a real timer).
//!
//! The story: five jobs of different kinds and attempt budgets are claimed in priority
//! order. Merge and summarize finish `succeeded`; the dedupe job fails once and succeeds
//! on attempt 2 of 2; the corrupt plan finishes `failed` by the consolidator's own
//! verdict; the forget pass exhausts its single attempt and is `failed` with a
//! `last_error`. A sixth job then shows cancellation. Every row is read back from
//! `SQLite`, and four empty sweeps print the backoff: 60 s, 120 s, 240 s, 480 s, then the
//! 600 s cap. The orchestration writes only the queue: merging records back into memory
//! is the consolidator's job, and this one only reports counters.
//!
//! Offline and deterministic: a `tempfile` directory, no model, no network, and every
//! timestamp comes from `stamp`.
#![allow(
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::must_use_candidate,
    clippy::items_after_statements
)]

mod common;

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, ensure, Context, Result};
use async_trait::async_trait;
use chrono::{DateTime, Duration, Utc};
use openjiuwen::core::common::error::{Error, Result as SdkResult};
use openjiuwen_memory::dreaming::{
    DreamConfig, DreamConsolidator, DreamFilter, DreamJob, DreamKind, DreamMetrics,
    DreamOrchestrator, DreamOrchestratorConfig, DreamOutcome, DreamPlanner, DreamRepository,
    DreamRunSummary, DreamState, DreamTrigger, SqliteDreamRepository,
};
use openjiuwen_memory::record::{CallOptions, MemoryScope};
use tempfile::TempDir;
use tokio_util::sync::CancellationToken;

use common::banner;

// --- the cast --------------------------------------------------------------

/// The user owning every dream job.
const USER: &str = "alice";
/// The workspace owning every dream job.
const WORKSPACE: &str = "openjiuwen-rs";
/// Database file inside the `TempDir`; `connect` creates the file, the `TempDir` is the
/// parent directory it does not create.
const DB_FILE: &str = "dreams.db";
/// The fixed epoch every timestamp is relative to (2023-11-14T22:13:20Z).
const EPOCH_SECS: i64 = 1_700_000_000;
/// Jobs claimed by one sweep.
const BATCH_SIZE: usize = 5;
/// Base interval between sweeps, in milliseconds.
const POLL_MS: u64 = 60_000;
/// Ceiling of the empty-sweep backoff, in milliseconds.
const CAP_MS: u64 = 600_000;

// The five jobs of the first sweep, in claim order: two succeed, one retries, one fails
// by the consolidator's verdict, one exhausts its only attempt.
/// Merge pass: succeeds on its first attempt.
const MERGE: &str = "dream-merge";
/// Summarize pass: succeeds on its first attempt with different counters.
const SUMMARIZE: &str = "dream-summarize";
/// Dedupe pass: fails once, succeeds on attempt 2 of 2.
const RETRY: &str = "dream-retry";
/// Merge pass with a corrupt plan: the consolidator itself declares it failed.
const CORRUPT: &str = "dream-corrupt";
/// Forget pass: fails its only attempt and exhausts its budget.
const EXHAUST: &str = "dream-exhaust";
/// Graph pass used for the cancellation section; enqueued later.
const CANCEL: &str = "dream-cancel";
/// Every job of the first sweep, in claim order.
const FIRST_BATCH: [&str; 5] = [MERGE, SUMMARIZE, RETRY, CORRUPT, EXHAUST];

// --- the run ---------------------------------------------------------------

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mode = common::setup()?;
    banner::title("23 memory dreaming");
    banner::note(&format!(
        "run mode: {}; no model is built here",
        mode.describe()
    ));

    let workspace = common::tempdir()?;
    let (repository, consolidator) = open_queue(&workspace).await?;
    let orchestrator = build_orchestrator(&repository, &consolidator)?;
    let epoch = stamp(0)?;

    seed(&repository, epoch).await?;
    banner::title("tick 1: the batch drains");
    sweep(&orchestrator, &repository, 0, (BATCH_SIZE, 2), &FIRST_BATCH).await?;
    banner::note(
        "`failed` counts failed attempts, so the requeued retry is included; the stored row says whether it was requeued or exhausted",
    );

    banner::title("tick 2: the retry");
    sweep(&orchestrator, &repository, 60, (1, 1), &[RETRY]).await?;
    print_status(
        &orchestrator,
        "after tick 2 (the retry reset the empty count)",
    );

    idle_ladder(&orchestrator).await?;
    queue_table(&repository).await?;
    cancellation(&orchestrator, &repository, &consolidator, epoch).await?;

    banner::title("done");
    banner::answer(
        "every job reached a terminal state in SQLite, and the backoff came from an injected clock",
    );
    Ok(())
}

// --- setup ----------------------------------------------------------------

/// Opens the `SQLite` queue inside `workspace` and the consolidator that drains it.
async fn open_queue(
    workspace: &TempDir,
) -> Result<(Arc<dyn DreamRepository>, Arc<RecordingConsolidator>)> {
    let url = format!("sqlite:{}", workspace.path().join(DB_FILE).display());
    let repository: Arc<dyn DreamRepository> = Arc::new(
        SqliteDreamRepository::connect(&url)
            .await
            .context("failed to open the dream queue")?,
    );
    banner::step(&format!(
        "opened the `dream_jobs` queue in `{DB_FILE}` (a temp directory; `connect` bootstraps the schema)"
    ));
    let consolidator = Arc::new(RecordingConsolidator::new(DreamMetrics {
        records_scanned: 6,
        records_merged: 5,
        records_created: 1,
        tokens_saved: 420,
        duration_ms: 12,
        ..DreamMetrics::default()
    }));
    Ok((repository, consolidator))
}

/// Builds the planner and the orchestrator over the shared queue.
fn build_orchestrator(
    repository: &Arc<dyn DreamRepository>,
    consolidator: &Arc<RecordingConsolidator>,
) -> Result<DreamOrchestrator> {
    let pass: Arc<dyn DreamConsolidator> = consolidator.clone();
    let planner = DreamPlanner::new(
        Arc::clone(repository),
        pass,
        DreamConfig::default().with_batch_size(BATCH_SIZE),
    )
    .context("failed to build the dream planner")?;
    let config = DreamOrchestratorConfig::default()
        .with_poll_interval_ms(POLL_MS)
        .with_initial_delay_ms(0)
        .with_max_interval_ms(CAP_MS)
        .with_backoff_multiplier(2.0)
        .with_max_backoff_steps(4)
        .with_run_retention_on_tick(false);
    let orchestrator = DreamOrchestrator::new(Arc::new(planner), Arc::clone(repository), config)
        .context("failed to build the dream orchestrator")?;
    banner::step(&format!(
        "planner batch size {BATCH_SIZE}; orchestrator base {POLL_MS} ms, cap {CAP_MS} ms, 2.0x backoff over 4 steps, retention off"
    ));
    Ok(orchestrator)
}

/// Enqueues the five jobs of the first sweep, all stamped with `epoch`.
async fn seed(repository: &Arc<dyn DreamRepository>, epoch: DateTime<Utc>) -> Result<()> {
    banner::title("seed the queue");
    let scope = MemoryScope::new(USER, WORKSPACE);
    let jobs = [
        job(MERGE, DreamKind::Consolidation, 50, 2, &scope, epoch),
        job(SUMMARIZE, DreamKind::Summarize, 40, 2, &scope, epoch),
        job(RETRY, DreamKind::Dedupe, 30, 2, &scope, epoch),
        job(CORRUPT, DreamKind::Consolidation, 20, 3, &scope, epoch),
        job(EXHAUST, DreamKind::Forget, 10, 1, &scope, epoch),
    ];
    for entry in jobs {
        repository
            .enqueue(entry)
            .await
            .context("failed to enqueue a dream job")?;
    }
    banner::step(
        "enqueued 5 pending jobs for scope `alice/openjiuwen-rs`; `priority DESC` is the claim order",
    );
    banner::note(
        "a job is claimable when state is `pending`, `attempts < max_attempts`, and `started_at` is before the tick's clock",
    );
    Ok(())
}

// --- the ticks ------------------------------------------------------------

/// Runs one tick at `offset` seconds and prints what it claimed and what it recorded.
async fn sweep(
    orchestrator: &DreamOrchestrator,
    repository: &Arc<dyn DreamRepository>,
    offset: i64,
    expected: (usize, usize),
    rows: &[&str],
) -> Result<()> {
    let now = stamp(offset)?;
    let summary = orchestrator
        .tick(now)
        .await
        .with_context(|| format!("the tick at +{offset}s failed"))?;
    ensure!(
        (summary.scanned, summary.succeeded) == expected,
        "the tick at +{offset}s must scan {} job(s) and succeed {}",
        expected.0,
        expected.1
    );
    banner::step(&format!("tick at +{offset}s ({}):", stamp_text(now)));
    print_summary(&summary);
    print_outcomes(&summary);
    read_back(repository, rows).await
}

/// Ticks 3-6: an empty queue and the backoff ladder it produces.
async fn idle_ladder(orchestrator: &DreamOrchestrator) -> Result<()> {
    banner::title("ticks 3-6: idle sweeps and the backoff ladder");
    for offset in [120, 180, 240, 300] {
        let summary = orchestrator
            .tick(stamp(offset)?)
            .await
            .with_context(|| format!("the idle tick at +{offset}s failed"))?;
        ensure!(
            summary.is_empty(),
            "an idle tick must claim and record nothing"
        );
        print_status(orchestrator, &format!("tick at +{offset}s (scanned 0)"));
    }
    banner::note(
        "interval = min(poll_interval_ms * multiplier^min(empty_sweeps, max_steps), max_interval_ms): 60s, 120s, 240s, 480s, then the 600s cap",
    );
    let status = orchestrator.status();
    let retention = if status.last_retention.is_some() {
        "present"
    } else {
        "none"
    };
    banner::step(&format!(
        "after six ticks: last_error {}, last_retention {retention}",
        status.last_error.as_deref().unwrap_or("none")
    ));
    Ok(())
}

// --- inspection -----------------------------------------------------------

/// Prints every queue row, the (absent) lease, and the per-state counts.
async fn queue_table(repository: &Arc<dyn DreamRepository>) -> Result<()> {
    banner::title("the queue as the repository sees it");
    let jobs = repository
        .list(&DreamFilter::default())
        .await
        .context("failed to list the dream queue")?;
    ensure!(
        jobs.len() == 5,
        "the queue must hold the five seeded jobs before the cancellation section"
    );
    banner::step("`list(&DreamFilter::default())` in claim order:");
    for entry in &jobs {
        banner::step(&format!(
            "  {:<16} {:<14} priority {:>2} attempts {}/{} -> `{}`",
            entry.id, entry.kind, entry.priority, entry.attempts, entry.max_attempts, entry.state
        ));
    }
    banner::note(
        "`list` orders `priority DESC, created_at ASC, id ASC`, exactly like `claim_next`",
    );

    let lease = repository
        .lease(RETRY)
        .await
        .context("failed to read the lease")?;
    ensure!(
        lease.is_none(),
        "no lease may be taken without a lease config"
    );
    banner::step(
        "`lease()` reports no row: the sweep claimed through the plain planner path, not `DreamOrchestratorConfig::with_lease`",
    );

    let counts = repository
        .count_by_state()
        .await
        .context("failed to count the queue")?;
    print_counts(&counts);
    Ok(())
}

/// Prints every state with its count, including the zeroes.
fn print_counts(counts: &BTreeMap<DreamState, usize>) {
    banner::step("`count_by_state()` over all six states:");
    for state in DreamState::ALL {
        let count = counts.get(&state).copied().unwrap_or(0);
        banner::step(&format!("  {:<10} {count}", state.as_str()));
    }
}

/// Prints the observable orchestrator snapshot after `label`.
fn print_status(orchestrator: &DreamOrchestrator, label: &str) {
    let status = orchestrator.status();
    banner::step(&format!(
        "{label}: ticks {}, consecutive_empty_sweeps {}, next_interval_ms {}, lock_skips {}, last_tick_at {}",
        status.ticks,
        status.consecutive_empty_sweeps,
        status.next_interval_ms,
        status.lock_skips,
        status
            .last_tick_at
            .map_or_else(|| "none".to_owned(), stamp_text)
    ));
}

/// Enqueues the cancellation job and walks the two cancellation paths.
async fn cancellation(
    orchestrator: &DreamOrchestrator,
    repository: &Arc<dyn DreamRepository>,
    consolidator: &Arc<RecordingConsolidator>,
    epoch: DateTime<Utc>,
) -> Result<()> {
    banner::title("cancellation: a shutdown is never recorded as a success");
    let scope = MemoryScope::new(USER, WORKSPACE);
    let created = epoch + Duration::seconds(360);
    repository
        .enqueue(job(
            CANCEL,
            DreamKind::GraphExtraction,
            0,
            1,
            &scope,
            created,
        ))
        .await
        .context("failed to enqueue the cancellation job")?;

    // Half one: the token is already cancelled, so the planner never claims.
    let stop = CancellationToken::new();
    stop.cancel();
    let skipped = orchestrator
        .tick_with(created, &CallOptions::new().with_cancellation(stop))
        .await
        .context("the pre-cancelled tick failed")?;
    ensure!(
        skipped.scanned == 0,
        "a pre-cancelled tick must claim nothing"
    );
    let pending = load(repository, CANCEL).await?;
    ensure!(
        pending.state == DreamState::Pending,
        "the unclaimed job must stay pending"
    );
    banner::step(&format!(
        "a pre-cancelled token -> scanned 0; `{CANCEL}` stays `{}` because the planner checks before every claim",
        pending.state
    ));

    // Half two: the shutdown lands while the pass runs, so the consolidator reports it.
    let stopped = orchestrator
        .tick(epoch + Duration::seconds(400))
        .await
        .context("the cancelled tick failed")?;
    ensure!(
        stopped.cancelled == 1,
        "the cancelled attempt must be counted as cancelled"
    );
    let outcome = stopped
        .outcomes
        .first()
        .context("the cancelled attempt recorded no outcome")?;
    banner::step(&format!("  outcome {}", describe_outcome(outcome)));
    let cancelled = load(repository, CANCEL).await?;
    ensure!(
        cancelled.state == DreamState::Cancelled,
        "an `Error::Cancelled` must complete the job as `cancelled`"
    );
    banner::step(&format!(
        "`{CANCEL}` is `{}` (never `succeeded`) and the sweep stopped there",
        cancelled.state
    ));

    let claims = consolidator.claims()?;
    banner::step(&format!(
        "the consolidator was handed {} job(s), in call order:",
        claims.len()
    ));
    banner::step(&format!("  {}", claims.join(" -> ")));

    let counts = repository
        .count_by_state()
        .await
        .context("failed to count the queue")?;
    print_counts(&counts);
    print_status(orchestrator, "final status");
    Ok(())
}

// --- helpers --------------------------------------------------------------

/// Builds one pending job; the scope is cloned because the caller reuses it.
fn job(
    id: &str,
    kind: DreamKind,
    priority: i32,
    max_attempts: u32,
    scope: &MemoryScope,
    now: DateTime<Utc>,
) -> DreamJob {
    DreamJob::new(scope.clone(), kind, DreamTrigger::Manual, now)
        .with_id(id)
        .with_priority(priority)
        .with_max_attempts(max_attempts)
}

/// Loads `id` from the queue, failing when the row is missing.
async fn load(repository: &Arc<dyn DreamRepository>, id: &str) -> Result<DreamJob> {
    repository
        .get(id)
        .await
        .with_context(|| format!("failed to read `{id}`"))?
        .with_context(|| format!("`{id}` is not in the queue"))
}

/// Reads `rows` back from `SQLite` and prints one line per row.
async fn read_back(repository: &Arc<dyn DreamRepository>, rows: &[&str]) -> Result<()> {
    banner::step("read back from `SQLite`:");
    for id in rows {
        let entry = load(repository, id).await?;
        banner::step(&format!("  {}", describe_job(&entry)));
    }
    Ok(())
}

/// One line of counters: `scanned 6, merged 5, ...`.
fn describe_metrics(metrics: DreamMetrics) -> String {
    format!(
        "scanned {}, merged {}, deleted {}, created {}, tokens_saved {}, {} ms",
        metrics.records_scanned,
        metrics.records_merged,
        metrics.records_deleted,
        metrics.records_created,
        metrics.tokens_saved,
        metrics.duration_ms
    )
}

/// One line per recorded outcome.
fn describe_outcome(outcome: &DreamOutcome) -> String {
    format!(
        "{} -> `{}` [{}]",
        outcome.job_id,
        outcome.state,
        describe_metrics(outcome.metrics)
    )
}

/// One line per stored row, including whether the repository stamped a finish time.
fn describe_job(entry: &DreamJob) -> String {
    let finished = if entry.finished_at.is_some() {
        "a finish stamp"
    } else {
        "no finish stamp"
    };
    let error = entry
        .last_error
        .as_deref()
        .map_or_else(String::new, |error| format!("; last_error `{error}`"));
    format!(
        "{}: state `{}`, attempts {}/{}, {}, metrics [{}]{}",
        entry.id,
        entry.state,
        entry.attempts,
        entry.max_attempts,
        finished,
        describe_metrics(entry.metrics),
        error
    )
}

/// Prints the counters of one sweep.
fn print_summary(summary: &DreamRunSummary) {
    banner::step(&format!(
        "summary: scanned {}, succeeded {}, failed {}, skipped {}, cancelled {}",
        summary.scanned, summary.succeeded, summary.failed, summary.skipped, summary.cancelled
    ));
}

/// Prints every recorded outcome of one sweep, with the notes a failed attempt carries.
fn print_outcomes(summary: &DreamRunSummary) {
    for outcome in &summary.outcomes {
        banner::step(&format!("  outcome {}", describe_outcome(outcome)));
        for note in &outcome.notes {
            banner::step(&format!("  note ({}): {note}", outcome.job_id));
        }
    }
}

/// Renders an instant as the fixed epoch-relative clock this example injects.
fn stamp_text(instant: DateTime<Utc>) -> String {
    instant.to_rfc3339()
}

/// The fixed instant `offset_secs` seconds after `EPOCH_SECS`.
fn stamp(offset_secs: i64) -> Result<DateTime<Utc>> {
    let seconds = EPOCH_SECS + offset_secs;
    DateTime::from_timestamp(seconds, 0)
        .ok_or_else(|| anyhow!("epoch + {seconds}s is not a representable instant"))
}

// --- the consolidator -----------------------------------------------------

/// The application's consolidation pass: it records every job it is handed and reports
/// deterministic outcomes keyed on the job id, so the whole run is reproducible.
struct RecordingConsolidator {
    /// Counters the merge pass reports.
    metrics: DreamMetrics,
    /// Job ids in call order, so the narrative can prove the claim sequence.
    claims: Mutex<Vec<String>>,
}

impl RecordingConsolidator {
    /// Builds the pass with `metrics` as the merge counters.
    fn new(metrics: DreamMetrics) -> Self {
        Self {
            metrics,
            claims: Mutex::new(Vec::new()),
        }
    }

    /// The job ids this pass was handed, in call order.
    fn claims(&self) -> SdkResult<Vec<String>> {
        self.claims
            .lock()
            .map(|claims| claims.clone())
            .map_err(|poisoned| Error::memory(format!("the claim log is poisoned: {poisoned}")))
    }

    /// Counters for a successful pass; each kind reports its own shape of work.
    fn metrics_for(&self, kind: &DreamKind) -> DreamMetrics {
        match kind {
            DreamKind::Consolidation => self.metrics,
            DreamKind::Summarize => DreamMetrics {
                records_scanned: 9,
                records_created: 2,
                tokens_saved: 780,
                duration_ms: 7,
                ..DreamMetrics::default()
            },
            _ => DreamMetrics {
                records_scanned: 3,
                records_merged: 2,
                tokens_saved: 120,
                duration_ms: 4,
                ..DreamMetrics::default()
            },
        }
    }
}

#[async_trait]
impl DreamConsolidator for RecordingConsolidator {
    /// Consolidates one claimed job.
    ///
    /// The rules, all keyed on the job id, are the point of the example: a job may fail an
    /// *attempt* (an `Err`, which the planner records through `DreamRepository::fail` and
    /// requeues while attempts remain), may fail *terminally* by the consolidator's own
    /// verdict (a `DreamOutcome::failed`, stored as-is), or may report cancellation.
    async fn consolidate(&self, job: &DreamJob, ctx: &CallOptions) -> SdkResult<DreamOutcome> {
        ctx.check()?;
        {
            let mut claims = self.claims.lock().map_err(|poisoned| {
                Error::memory(format!("the claim log is poisoned: {poisoned}"))
            })?;
            claims.push(job.id.clone());
        }
        match job.id.as_str() {
            RETRY if job.attempts < 2 => Err(Error::memory(format!(
                "the dedupe pass lost its transaction on attempt {}",
                job.attempts
            ))),
            CORRUPT => Ok(DreamOutcome::failed(
                job.id.clone(),
                "the merge plan references a record that no longer exists",
            )),
            EXHAUST => Err(Error::memory(
                "the forget pass found a corrupt index entry and cannot continue",
            )),
            CANCEL => Err(Error::cancelled()),
            _ => Ok(DreamOutcome::succeeded(
                job.id.clone(),
                self.metrics_for(&job.kind),
            )),
        }
    }
}
