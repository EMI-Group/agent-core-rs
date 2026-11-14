//! Memory dreaming scenario: long-term write/recall followed by a durable dream tick.
//!
//! The seam under test is the one an embedding application owns: `LongTermMemory` writes and
//! recalls scoped records, and the dream stack (`DreamPlanner` over a `DreamRepository`,
//! driven by `DreamOrchestrator::tick`, scored by `rank_by_salience`, trimmed by
//! `DreamRetention`) consolidates them on a caller-injected clock. The consolidator is the
//! application's own code — the kernel only offers the seam — so this scenario implements the
//! smallest deterministic one that is still useful: score the scope's records at the pass
//! instant, forget everything under a fixed threshold, and rewrite the surviving importance
//! to its salience score. "Decayed" is that in-place rewrite (`records_merged`), "archived
//! out" is the forget decision (`records_deleted`); nothing creates new records here.
//!
//! Every scenario is offline and deterministic: in-memory stores (plus temp-file SQLite
//! behind the `sqlite` feature), a pinned clock, no sleeps, no wall-clock assertions, and no
//! assertion on an id the API did not hand back.
//!
//! The clock is pinned without `chrono`, which is deliberately not a dependency of this
//! crate and which the kernel exposes no alias for: the scenario feeds the memory crate's
//! public timestamp parser a unix-seconds value (see the `at!` macro) and passes the result
//! straight to the dream API, which is exactly the "inject an explicit `now`" contract the
//! orchestrator asks for.
#![allow(clippy::unwrap_used, clippy::expect_used)]

mod support;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen::core::common::error::Result;
use openjiuwen::core::memory::external::rest_json::parse_timestamp;
use openjiuwen::core::memory::{
    rank_by_salience, salience_score, AddOptions, CallOptions, DreamConfig, DreamConsolidator,
    DreamFilter, DreamJob, DreamKind, DreamMetrics, DreamOrchestrator, DreamOrchestratorConfig,
    DreamOutcome, DreamPlanner, DreamRepository, DreamRetention, DreamState, DreamTrigger,
    LongTermMemory, MemoryEngineConfig, MemoryRecord, MemoryScope, MemoryUpdate, NewMemory,
    SalienceConfig, SearchMode, SearchOptions,
};
use serde_json::{json, Value};
use support::within;

/// The suite epoch (2023-11-14T22:13:20Z): every instant here is an offset in seconds.
const EPOCH_SECS: i64 = 1_700_000_000;

/// One day in seconds, so the fixed offsets below stay readable.
const DAY_SECS: i64 = 86_400;

/// The pass clock of the retention scenarios: 90 days after the epoch, i.e. exactly three
/// half-lives of [`HALF_LIFE_SECS`], so an epoch-dated record keeps an eighth of its recency
/// term while a record stamped at the pass instant keeps all of it.
const NINETY_DAYS: i64 = 90 * DAY_SECS;

/// The salience half-life every scenario scores with: the kernel's 30-day default, written
/// out because the arithmetic below depends on it.
const HALF_LIFE_SECS: u64 = 2_592_000;

/// The forget threshold the retention scenarios use: a record scoring below it is dropped.
const FORGET_BELOW: f32 = 0.35;

/// The JSON envelope the memory crate's public timestamp parser reads for an instant
/// `offset_secs` after [`EPOCH_SECS`].
fn instant_json(offset_secs: i64) -> Value {
    json!({ "at": EPOCH_SECS + offset_secs })
}

/// The injectable `now`, `offset_secs` seconds after [`EPOCH_SECS`].
///
/// A macro rather than a helper because a helper would have to name the clock type, and
/// `chrono` is not a dependency of this crate. The expansion is a plain expression of the
/// type `parse_timestamp` returns, which is the type every dream entry point accepts.
macro_rules! at {
    ($offset_secs:expr) => {
        parse_timestamp(&instant_json($offset_secs), &["at"])
            .expect("a fixed epoch offset parses as a UTC instant")
    };
}

/// The scope every record and every job in this scenario belongs to.
fn scope() -> MemoryScope {
    MemoryScope::new("alice", "dreaming")
}

/// The deterministic scorer: the heuristic backend on a 30-day half-life.
fn salience() -> SalienceConfig {
    SalienceConfig::default().with_half_life_secs(HALF_LIFE_SECS)
}

/// Search options that keep every candidate the keyword index can produce.
fn keyword(top_k: usize) -> SearchOptions {
    SearchOptions::default()
        .with_threshold(0.0)
        .with_top_k(top_k)
        .with_mode(SearchMode::Keyword)
}

/// Stores one memory and returns the id the engine assigned it.
async fn remember(ltm: &LongTermMemory, memory: NewMemory) -> String {
    let outcome = within(
        "store a memory fact",
        ltm.add(&scope(), memory, &AddOptions::default()),
    )
    .await
    .expect("the fact is stored");
    assert_eq!(
        outcome.added.len(),
        1,
        "a distinct fact is added: {outcome:?}"
    );
    outcome
        .added
        .first()
        .expect("the write reported its new record")
        .id
        .clone()
}

/// The ids the keyword index returns for `query`, in rank order.
async fn recall(ltm: &LongTermMemory, query: &str) -> Vec<String> {
    within(
        "recall the scope",
        ltm.search(&scope(), query, &keyword(10)),
    )
    .await
    .expect("keyword recall")
    .into_iter()
    .map(|hit| hit.record.id)
    .collect()
}

/// Every record currently stored in the scope.
async fn records(ltm: &LongTermMemory) -> Vec<MemoryRecord> {
    within("list the scope's records", ltm.list(&scope(), &[]))
        .await
        .expect("list")
}

/// The salience of record `id` at `offset_secs` after the epoch, from `listed`.
///
/// Reading the score from the same public scorer the consolidator uses is what makes the
/// assertion on the rewritten importance deterministic: both sides compute it from the same
/// record and the same instant.
fn salience_of(listed: &[MemoryRecord], id: &str, offset_secs: i64) -> f32 {
    let record = listed
        .iter()
        .find(|record| record.id == id)
        .expect("the record is listed");
    salience_score(record, at!(offset_secs), &salience())
        .expect("the heuristic scorer needs no runtime")
        .total
}

/// The deterministic dream consolidator this scenario injects.
///
/// It is the application half of the dreaming seam: it owns the memory mutation, never
/// touches the queue, and derives its clock from the job (`created_at` is the instant the
/// caller stamped the job with, i.e. the same `now` the orchestrator ticked at).
struct SalienceConsolidator {
    /// The facade whose records this pass rewrites.
    ltm: Arc<LongTermMemory>,
    /// The scope the pass consolidates.
    scope: MemoryScope,
    /// The scorer applied to every record.
    config: SalienceConfig,
    /// Records scoring strictly below this total are forgotten; `0.0` keeps everything.
    forget_below: f32,
    /// How many passes actually ran, so a tick that claims nothing is provably a no-op.
    calls: AtomicUsize,
}

impl SalienceConsolidator {
    /// The number of consolidation passes that ran.
    fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl DreamConsolidator for SalienceConsolidator {
    async fn consolidate(&self, job: &DreamJob, ctx: &CallOptions) -> Result<DreamOutcome> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        ctx.check()?;
        assert_eq!(
            job.scope, self.scope,
            "the job and the pass share one scope"
        );
        let listed = self.ltm.list(&self.scope, &[]).await?;
        let ranked = rank_by_salience(&listed, job.created_at, &self.config)?;
        let mut metrics = DreamMetrics {
            records_scanned: u64::try_from(listed.len()).expect("the record count fits in u64"),
            ..DreamMetrics::default()
        };
        for (id, score) in ranked {
            ctx.check()?;
            if score.total < self.forget_below {
                if self.ltm.delete(&self.scope, &id).await? {
                    metrics.records_deleted = metrics.records_deleted.saturating_add(1);
                }
            } else {
                // In-place rewrite: one record in, one record out, no new row.
                let importance =
                    MemoryUpdate::default().with_importance(score.total.clamp(0.0, 1.0));
                self.ltm.update(&self.scope, &id, importance, ctx).await?;
                metrics.records_merged = metrics.records_merged.saturating_add(1);
            }
        }
        Ok(DreamOutcome::succeeded(job.id.clone(), metrics))
    }
}

/// The consolidator over `ltm`'s scope, forgetting anything below `forget_below`.
fn salience_consolidator(
    ltm: &Arc<LongTermMemory>,
    forget_below: f32,
) -> Arc<SalienceConsolidator> {
    Arc::new(SalienceConsolidator {
        ltm: Arc::clone(ltm),
        scope: scope(),
        config: salience(),
        forget_below,
        calls: AtomicUsize::new(0),
    })
}

/// A planner over `repository` (the facade's queue) running `consolidator`.
fn planner_over(
    repository: Arc<dyn DreamRepository>,
    consolidator: Arc<SalienceConsolidator>,
) -> Arc<DreamPlanner> {
    let consolidator: Arc<dyn DreamConsolidator> = consolidator;
    Arc::new(
        DreamPlanner::new(repository, consolidator, DreamConfig::default())
            .expect("the default dream config validates"),
    )
}

/// An orchestrator over `planner` that reuses the planner's queue for retention.
fn orchestrator_over(
    planner: Arc<DreamPlanner>,
    config: DreamOrchestratorConfig,
) -> DreamOrchestrator {
    let repository: Arc<dyn DreamRepository> = Arc::clone(planner.repository());
    DreamOrchestrator::new(planner, repository, config)
        .expect("the orchestrator schedule validates")
}

/// Enqueues `job` on `planner`'s queue and returns its id.
async fn enqueue(planner: &DreamPlanner, job: DreamJob) -> String {
    let id = job.id.clone();
    within("enqueue a dream job", planner.repository().enqueue(job))
        .await
        .expect("the dream job is queued");
    id
}

// A scenario body reads as one narrative; its length is the scenario, not a smell.
#[allow(clippy::too_many_lines)]
#[tokio::test]
async fn a_tick_decays_both_memories_and_keeps_them_recallable() {
    let ltm = within(
        "build the in-memory facade",
        LongTermMemory::in_memory(MemoryEngineConfig::default()),
    )
    .await
    .expect("the in-memory facade builds");
    let deploy = remember(
        &ltm,
        NewMemory::semantic("Deploys land through the blue green pipeline")
            .with_importance(1.0)
            .with_timestamp(at!(0)),
    )
    .await;
    let style = remember(
        &ltm,
        NewMemory::semantic("Alice prefers dark mode for late review sessions")
            .with_importance(0.95)
            .with_timestamp(at!(0)),
    )
    .await;

    // Both facts are recallable before the pass. The blended recall score keeps every
    // candidate at a zero threshold, so the assertion is on the fact a query returns, not
    // on the hit count.
    assert!(
        recall(&ltm, "blue green pipeline").await.contains(&deploy),
        "the deploy fact is recallable"
    );
    assert!(
        recall(&ltm, "dark mode").await.contains(&style),
        "the style fact is recallable"
    );

    // ... and the pass rewrites their importance to the salience score it computed. The
    // clock is the job's creation instant, so the scenario pins it to the tick instant.
    let queue = ltm
        .dreaming_in_memory()
        .expect("the facade's in-memory dream queue builds");
    let consolidator = salience_consolidator(&ltm, 0.0);
    let planner = planner_over(Arc::clone(queue.repository()), Arc::clone(&consolidator));
    let config = DreamOrchestratorConfig::default().with_poll_interval_ms(30_000);
    let orchestrator = orchestrator_over(Arc::clone(&planner), config.clone());
    let job = DreamJob::new(
        scope(),
        DreamKind::Consolidation,
        DreamTrigger::Manual,
        at!(3_600),
    )
    .with_id("in-memory-pass");
    assert_eq!(enqueue(&planner, job).await, "in-memory-pass");

    // What the consolidator will see, and therefore what it will write.
    let before = records(&ltm).await;
    assert_eq!(before.len(), 2, "the scope holds both facts: {before:?}");
    let deploy_before = salience_of(&before, &deploy, 3_600);
    let style_before = salience_of(&before, &style, 3_600);
    assert!(
        deploy_before > style_before,
        "{deploy_before} vs {style_before}"
    );

    let summary = within("tick the dream pass", orchestrator.tick(at!(3_600)))
        .await
        .expect("the tick sweeps");
    assert_eq!(
        (summary.scanned, summary.succeeded, summary.failed),
        (1, 1, 0),
        "one queued job was claimed and consolidated: {summary:?}"
    );
    let metrics = summary
        .outcomes
        .first()
        .expect("the sweep recorded its outcome")
        .metrics;
    assert_eq!(metrics.records_scanned, 2, "both records were scored");
    assert_eq!(metrics.records_merged, 2, "both importances were rewritten");
    assert_eq!(metrics.records_deleted, 0, "nothing was forgotten");
    assert_eq!(metrics.records_created, 0, "decay creates no new records");

    // The retained records carry their salience score as importance, so both were decayed.
    let after = records(&ltm).await;
    assert_eq!(after.len(), 2, "nothing was dropped: {after:?}");
    for (id, was, decayed_to) in [
        (&deploy, 1.0_f32, deploy_before),
        (&style, 0.95_f32, style_before),
    ] {
        let record = after
            .iter()
            .find(|record| record.id == *id)
            .expect("the record is still stored");
        assert!(
            (record.importance - decayed_to).abs() < 1e-6,
            "{} keeps its salience score as importance: {} vs {decayed_to}",
            record.id,
            record.importance
        );
        assert!(
            record.importance < was,
            "the pass decayed {} below its written importance: {} vs {was}",
            record.id,
            record.importance
        );
    }

    // A decayed record is still a memory: recall returns it unchanged.
    assert!(
        recall(&ltm, "blue green pipeline").await.contains(&deploy),
        "the decayed deploy fact is still recallable"
    );
    assert!(
        recall(&ltm, "dark mode").await.contains(&style),
        "the decayed style fact is still recallable"
    );
    assert_eq!(consolidator.calls(), 1, "exactly one pass ran");

    // The queue and the schedule are observable, and a claimed job resets the backoff.
    let status = orchestrator.status();
    assert_eq!(status.ticks, 1);
    assert!(status.last_tick_at.is_some());
    assert_eq!(status.consecutive_empty_sweeps, 0);
    assert_eq!(status.next_interval_ms, config.poll_interval_ms);
    assert_eq!(
        status
            .last_summary
            .as_ref()
            .map(|summary| summary.succeeded),
        Some(1)
    );
    let stored = within(
        "read the recorded job",
        planner.repository().get("in-memory-pass"),
    )
    .await
    .expect("get")
    .expect("the job row survives");
    assert_eq!(stored.state, DreamState::Succeeded);
    assert_eq!(stored.metrics.records_deleted, 0);
    assert_eq!(status.last_error, None);
}

// A scenario body reads as one narrative; its length is the scenario, not a smell.
#[allow(clippy::too_many_lines)]
#[tokio::test]
async fn an_aged_low_salience_memory_is_forgotten_while_a_fresh_one_survives() {
    let ltm = within(
        "build the in-memory facade",
        LongTermMemory::in_memory(MemoryEngineConfig::default()),
    )
    .await
    .expect("the in-memory facade builds");
    // The aged record is three half-lives old and nearly worthless; the fresh one is
    // stamped at the pass instant and written as important.
    let aged = remember(
        &ltm,
        NewMemory::semantic("The old wiki listed a deprecated endpoint")
            .with_importance(0.02)
            .with_timestamp(at!(0)),
    )
    .await;
    let fresh = remember(
        &ltm,
        NewMemory::semantic("Releases now run through the release train")
            .with_importance(0.95)
            .with_timestamp(at!(NINETY_DAYS)),
    )
    .await;

    // Pin the arithmetic: (importance + recency + access + kind) / 4 with the default
    // weights, recency = 0.5^3 = 0.125 for the aged record and 1.0 for the fresh one, no
    // recall yet (access 0.0), and the semantic-memory kind weight of 0.9.
    let before = records(&ltm).await;
    assert_eq!(before.len(), 2, "the scope holds both facts: {before:?}");
    let aged_before = salience_of(&before, &aged, NINETY_DAYS);
    let fresh_before = salience_of(&before, &fresh, NINETY_DAYS);
    assert!(
        (aged_before - 0.261_25).abs() < 1e-5,
        "the aged record's salience is (0.02 + 0.125 + 0 + 0.9) / 4: {aged_before}"
    );
    assert!(
        (fresh_before - 0.712_5).abs() < 1e-5,
        "the fresh record's salience is (0.95 + 1.0 + 0 + 0.9) / 4: {fresh_before}"
    );
    let ranked = rank_by_salience(&before, at!(NINETY_DAYS), &salience()).expect("ranking");
    assert_eq!(
        ranked.first().map(|(id, _)| id.clone()),
        Some(fresh.clone()),
        "the fresh record ranks first: {ranked:?}"
    );

    let queue = ltm
        .dreaming_in_memory()
        .expect("the facade's in-memory dream queue builds");
    let consolidator = salience_consolidator(&ltm, FORGET_BELOW);
    let planner = planner_over(Arc::clone(queue.repository()), Arc::clone(&consolidator));
    let orchestrator = orchestrator_over(Arc::clone(&planner), DreamOrchestratorConfig::default());
    let job = DreamJob::new(
        scope(),
        DreamKind::Consolidation,
        DreamTrigger::Manual,
        at!(NINETY_DAYS),
    )
    .with_id("retention-pass");
    enqueue(&planner, job).await;

    let summary = within(
        "tick the retention pass",
        orchestrator.tick(at!(NINETY_DAYS)),
    )
    .await
    .expect("the tick sweeps");
    assert_eq!((summary.scanned, summary.succeeded), (1, 1), "{summary:?}");
    let outcome = summary
        .outcomes
        .first()
        .expect("the sweep recorded its outcome");
    assert_eq!(outcome.state, DreamState::Succeeded);
    assert_eq!(
        (
            outcome.metrics.records_scanned,
            outcome.metrics.records_deleted,
            outcome.metrics.records_merged,
        ),
        (2, 1, 1),
        "one record was archived out and one decayed: {outcome:?}"
    );

    // Per-record outcomes: the aged record is gone, the fresh one is kept and re-scored.
    assert!(
        within("read the forgotten record", ltm.get(&scope(), &aged))
            .await
            .expect("get")
            .is_none(),
        "the aged record was archived out"
    );
    let kept = within("read the retained record", ltm.get(&scope(), &fresh))
        .await
        .expect("get")
        .expect("the fresh record is retained");
    assert!(
        (kept.importance - fresh_before).abs() < 1e-6,
        "the retained record's importance is its salience score: {} vs {fresh_before}",
        kept.importance
    );
    assert!(
        kept.importance > FORGET_BELOW,
        "the retained record stays above the threshold: {}",
        kept.importance
    );

    // Recall agrees with the pass: the survivor is findable, the forgotten fact is not.
    assert!(
        recall(&ltm, "release train").await.contains(&fresh),
        "the retained record is still recallable"
    );
    let after_forget = recall(&ltm, "deprecated endpoint").await;
    assert!(
        !after_forget.contains(&aged),
        "the forgotten record is no longer recallable: {after_forget:?}"
    );
}

#[tokio::test]
async fn a_tick_with_nothing_due_does_no_work_and_the_schedule_backs_off() {
    let ltm = within(
        "build the in-memory facade",
        LongTermMemory::in_memory(MemoryEngineConfig::default()),
    )
    .await
    .expect("the in-memory facade builds");
    let fact = remember(
        &ltm,
        NewMemory::semantic("Deploys land through the blue green pipeline")
            .with_importance(1.0)
            .with_timestamp(at!(0)),
    )
    .await;

    let queue = ltm
        .dreaming_in_memory()
        .expect("the facade's in-memory dream queue builds");
    let consolidator = salience_consolidator(&ltm, 0.0);
    let planner = planner_over(Arc::clone(queue.repository()), Arc::clone(&consolidator));
    let config = DreamOrchestratorConfig::default()
        .with_poll_interval_ms(30_000)
        .with_max_interval_ms(240_000)
        .with_max_backoff_steps(3);
    let orchestrator = orchestrator_over(Arc::clone(&planner), config.clone());

    // Nothing has been scheduled before the first tick.
    assert_eq!(orchestrator.status().next_interval_ms, 0);
    assert_eq!(orchestrator.status().ticks, 0);

    // The queue is empty, so nothing is due: the tick claims nothing and touches no record.
    let first = within("tick with an empty queue", orchestrator.tick(at!(0)))
        .await
        .expect("the tick succeeds");
    assert!(first.is_empty(), "an empty queue does no work: {first:?}");
    let empty_sweep = orchestrator.status();
    assert_eq!(empty_sweep.ticks, 1);
    assert_eq!(empty_sweep.consecutive_empty_sweeps, 1);
    assert_eq!(
        empty_sweep.next_interval_ms,
        u64::try_from(orchestrator.interval_for_empty_sweeps(1).as_millis())
            .expect("the interval fits in u64"),
        "the documented schedule is what the status reports"
    );
    assert_eq!(empty_sweep.next_interval_ms, 60_000, "30 s x 2");
    assert_eq!(consolidator.calls(), 0, "no pass ran");

    // A second tick at the very same instant is the same no-op; only the backoff moves.
    let second = within("tick again at the same instant", orchestrator.tick(at!(0)))
        .await
        .expect("the tick succeeds");
    assert!(second.is_empty(), "{second:?}");
    let second_sweep = orchestrator.status();
    assert_eq!(second_sweep.ticks, 2);
    assert_eq!(second_sweep.consecutive_empty_sweeps, 2);
    assert_eq!(second_sweep.next_interval_ms, 120_000, "30 s x 4");
    assert_eq!(consolidator.calls(), 0);
    let untouched = records(&ltm).await;
    assert_eq!(untouched.len(), 1, "{untouched:?}");
    assert!(
        (untouched[0].importance - 1.0).abs() < f32::EPSILON,
        "an empty tick rewrites no importance: {}",
        untouched[0].importance
    );

    // Queue one job at the same clock: the next tick at that instant does the work.
    let job = DreamJob::new(
        scope(),
        DreamKind::Consolidation,
        DreamTrigger::Manual,
        at!(0),
    )
    .with_id("due-pass");
    enqueue(&planner, job).await;
    let third = within("tick with the queued job", orchestrator.tick(at!(0)))
        .await
        .expect("the tick sweeps");
    assert_eq!((third.scanned, third.succeeded), (1, 1), "{third:?}");
    assert_eq!(consolidator.calls(), 1, "the due job ran once");
    let worked = orchestrator.status();
    assert_eq!(worked.ticks, 3);
    assert_eq!(
        worked.consecutive_empty_sweeps, 0,
        "a sweep that claimed a job resets the backoff"
    );
    assert_eq!(worked.next_interval_ms, config.poll_interval_ms);

    // The job is terminal now, so a fourth tick at the same instant finds nothing again.
    let fourth = within("tick after the job finished", orchestrator.tick(at!(0)))
        .await
        .expect("the tick succeeds");
    assert!(fourth.is_empty(), "{fourth:?}");
    assert_eq!(consolidator.calls(), 1, "a finished job is never re-run");
    let idle = orchestrator.status();
    assert_eq!(idle.ticks, 4);
    assert_eq!(idle.consecutive_empty_sweeps, 1);
    assert_eq!(idle.next_interval_ms, 60_000);

    // The whole documented sequence, clamped at the cap.
    assert_eq!(
        [0_u32, 1, 2, 3, 9]
            .into_iter()
            .map(
                |empty| u64::try_from(orchestrator.interval_for_empty_sweeps(empty).as_millis())
                    .expect("the interval fits in u64")
            )
            .collect::<Vec<u64>>(),
        vec![30_000, 60_000, 120_000, 240_000, 240_000]
    );
    assert!(recall(&ltm, "blue green pipeline").await.contains(&fact));
}

// A scenario body reads as one narrative; its length is the scenario, not a smell.
#[allow(clippy::too_many_lines)]
#[tokio::test]
async fn the_configured_retention_policy_purges_finished_jobs_only() {
    let ltm = within(
        "build the in-memory facade",
        LongTermMemory::in_memory(MemoryEngineConfig::default()),
    )
    .await
    .expect("the in-memory facade builds");
    let fact = remember(
        &ltm,
        NewMemory::semantic("Deploys land through the blue green pipeline")
            .with_importance(1.0)
            .with_timestamp(at!(0)),
    )
    .await;

    let queue = ltm
        .dreaming_in_memory()
        .expect("the facade's in-memory dream queue builds");
    let consolidator = salience_consolidator(&ltm, 0.0);
    let planner = planner_over(Arc::clone(queue.repository()), Arc::clone(&consolidator));
    // Three queued jobs, consolidated by one sweep (`DreamConfig::batch_size` is 10).
    let job_ids = ["retained-a", "retained-b", "retained-c"];
    for id in job_ids {
        let job = DreamJob::new(
            scope(),
            DreamKind::Consolidation,
            DreamTrigger::Manual,
            at!(0),
        )
        .with_id(id);
        enqueue(&planner, job).await;
    }
    let ticking = orchestrator_over(Arc::clone(&planner), DreamOrchestratorConfig::default());
    let first_sweep = within("tick the pass", ticking.tick(at!(0)))
        .await
        .expect("the tick sweeps");
    assert_eq!(
        (first_sweep.scanned, first_sweep.succeeded),
        (3, 3),
        "{first_sweep:?}"
    );
    let counts = within(
        "count the finished jobs",
        planner.repository().count_by_state(),
    )
    .await
    .expect("counts");
    assert_eq!(
        counts.get(&DreamState::Succeeded).copied(),
        Some(3),
        "{counts:?}"
    );

    // The second orchestrator keeps only the newest finished row. The count cap is used
    // rather than the age window because `DreamRepository::complete` stamps `finished_at`
    // from the process clock, not from the tick's `now`; the cap is clock-free, so the
    // outcome depends on nothing but the rows themselves.
    let retained = orchestrator_over(
        Arc::clone(&planner),
        DreamOrchestratorConfig::default()
            .with_retention(DreamRetention::new(0).with_max_per_state(1)),
    );
    let sweep = within("tick the retention pass", retained.tick(at!(3_600)))
        .await
        .expect("the tick succeeds");
    assert!(
        sweep.is_empty(),
        "finished jobs are not claimable: {sweep:?}"
    );
    let status = retained.status();
    let report = status
        .last_retention
        .expect("the tick reports what retention removed");
    assert_eq!(
        (report.deleted_expired, report.deleted_over_cap),
        (0, 2),
        "the cap keeps one of the three finished rows: {report:?}"
    );
    assert_eq!(report.total(), 2);
    assert!(!report.is_empty());
    assert_eq!(status.last_error, None);
    let remaining = within(
        "list the queue after retention",
        planner.repository().list(&DreamFilter::new()),
    )
    .await
    .expect("list");
    assert_eq!(
        remaining.len(),
        1,
        "one row survives the cap: {remaining:?}"
    );
    let survivor = &remaining[0];
    assert_eq!(survivor.state, DreamState::Succeeded);
    assert!(
        job_ids.contains(&survivor.id.as_str()),
        "the survivor is one of the consolidated jobs: {survivor:?}"
    );
    // The purged rows are really gone, not merely hidden from the listing.
    let purged = job_ids
        .iter()
        .find(|id| **id != survivor.id)
        .expect("two of the three ids were purged");
    assert!(
        within("read a purged job", planner.repository().get(purged))
            .await
            .expect("get")
            .is_none(),
        "the purged row was deleted"
    );

    // Retention is housekeeping, not memory: the record and its decay are untouched, and a
    // second pass at the same instant removes nothing else.
    let survivors = records(&ltm).await;
    assert_eq!(survivors.len(), 1, "{survivors:?}");
    assert_eq!(survivors[0].id, fact);
    let decayed = survivors[0].importance;
    assert!(
        decayed < 1.0,
        "the earlier tick decayed the record: {decayed}"
    );
    let again = within("tick retention again", retained.tick(at!(3_600)))
        .await
        .expect("the tick succeeds");
    assert!(again.is_empty(), "{again:?}");
    let second_report = retained
        .status()
        .last_retention
        .expect("the pass reports an empty report rather than nothing");
    assert!(
        second_report.is_empty(),
        "the cap is idempotent at the same instant: {second_report:?}"
    );
    assert_eq!(retained.status().ticks, 2);
    let kept = within("read the surviving record", ltm.get(&scope(), &fact))
        .await
        .expect("get")
        .expect("the record outlives the jobs that consolidated it");
    assert!((kept.importance - decayed).abs() < f32::EPSILON, "{kept:?}");
    assert!(recall(&ltm, "blue green pipeline").await.contains(&fact));
}

#[cfg(feature = "sqlite")]
// A scenario body reads as one narrative; its length is the scenario, not a smell.
#[allow(clippy::too_many_lines)]
#[tokio::test]
async fn sqlite_consolidation_is_durable_across_reopened_handles() {
    let directory = tempfile::tempdir().expect("a temp directory for the databases");
    let memory_path = directory.path().join("memory.sqlite");
    let dream_path = directory.path().join("dreams.sqlite");

    let ltm = within(
        "open the sqlite memory facade",
        LongTermMemory::sqlite(MemoryEngineConfig::default(), &memory_path),
    )
    .await
    .expect("the sqlite facade builds");
    let fresh = remember(
        &ltm,
        NewMemory::semantic("Releases now run through the release train")
            .with_importance(0.95)
            .with_timestamp(at!(NINETY_DAYS)),
    )
    .await;
    let aged = remember(
        &ltm,
        NewMemory::semantic("The old wiki listed a deprecated endpoint")
            .with_importance(0.02)
            .with_timestamp(at!(0)),
    )
    .await;

    let before = records(&ltm).await;
    assert_eq!(before.len(), 2, "{before:?}");
    let fresh_before = salience_of(&before, &fresh, NINETY_DAYS);
    assert!(fresh_before > FORGET_BELOW, "{fresh_before}");
    assert!(salience_of(&before, &aged, NINETY_DAYS) < FORGET_BELOW);

    let queue = within(
        "open the sqlite dream queue",
        ltm.dreaming_sqlite(&dream_path),
    )
    .await
    .expect("the sqlite dream queue builds");
    let consolidator = salience_consolidator(&ltm, FORGET_BELOW);
    let planner = planner_over(Arc::clone(queue.repository()), Arc::clone(&consolidator));
    let orchestrator = orchestrator_over(Arc::clone(&planner), DreamOrchestratorConfig::default());
    let job = DreamJob::new(
        scope(),
        DreamKind::Consolidation,
        DreamTrigger::Manual,
        at!(NINETY_DAYS),
    )
    .with_id("sqlite-pass");
    assert_eq!(enqueue(&planner, job).await, "sqlite-pass");

    let summary = within("tick the sqlite pass", orchestrator.tick(at!(NINETY_DAYS)))
        .await
        .expect("the sqlite tick sweeps");
    assert_eq!((summary.scanned, summary.succeeded), (1, 1), "{summary:?}");
    let metrics = summary
        .outcomes
        .first()
        .expect("the sweep recorded its outcome")
        .metrics;
    assert_eq!(
        (
            metrics.records_scanned,
            metrics.records_deleted,
            metrics.records_merged,
        ),
        (2, 1, 1),
        "{metrics:?}"
    );

    // Close every handle: what survives is on disk, not in memory.
    drop(orchestrator);
    drop(planner);
    drop(queue);
    drop(ltm);
    assert!(memory_path.exists(), "the memory database was written");
    assert!(dream_path.exists(), "the dream queue was written");

    // A second handle on the same files reads the consolidated state back.
    let reopened = within(
        "reopen the sqlite memory facade",
        LongTermMemory::sqlite(MemoryEngineConfig::default(), &memory_path),
    )
    .await
    .expect("the sqlite facade reopens");
    assert!(
        within("read the forgotten record", reopened.get(&scope(), &aged))
            .await
            .expect("get")
            .is_none(),
        "the forgotten record did not come back"
    );
    let survivor = within("read the retained record", reopened.get(&scope(), &fresh))
        .await
        .expect("get")
        .expect("the retained record is on disk");
    assert!(
        (survivor.importance - fresh_before).abs() < 1e-6,
        "the decayed importance survived: {} vs {fresh_before}",
        survivor.importance
    );
    assert!(
        recall(&reopened, "release train").await.contains(&fresh),
        "the retained record is still recallable through the reopened facade"
    );

    let reopened_queue = within(
        "reopen the sqlite dream queue",
        reopened.dreaming_sqlite(&dream_path),
    )
    .await
    .expect("the sqlite dream queue reopens");
    let stored = within(
        "read the recorded job",
        reopened_queue.repository().get("sqlite-pass"),
    )
    .await
    .expect("get")
    .expect("the job row is durable");
    assert_eq!(stored.state, DreamState::Succeeded);
    assert_eq!(stored.metrics.records_scanned, 2);
    assert_eq!(stored.metrics.records_deleted, 1);
    assert_eq!(stored.metrics.records_merged, 1);
    let counts = within(
        "count the queue's states",
        reopened_queue.repository().count_by_state(),
    )
    .await
    .expect("counts");
    assert_eq!(
        counts.get(&DreamState::Succeeded).copied(),
        Some(1),
        "{counts:?}"
    );
    assert_eq!(
        counts.get(&DreamState::Pending).copied(),
        Some(0),
        "{counts:?}"
    );
}
