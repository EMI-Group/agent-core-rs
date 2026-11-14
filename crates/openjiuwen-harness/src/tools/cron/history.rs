//! The bounded run history of the cron scheduler.
//!
//! Every execution the scheduler records — a job firing through [`super::CronScheduler::due`]
//! or a manual [`super::CronScheduler::run_now`] — is appended to `<dir>/runs.json` as one flat
//! JSON array in chronological order (oldest first). The file is capped: at most
//! [`MAX_RUNS_PER_JOB`] records survive per job and [`MAX_RUNS_TOTAL`] in total, so a
//! long-running schedule cannot grow the file without bound. Rotation drops the oldest records
//! first.
//!
//! The history has no lock of its own: it is only touched while the owning scheduler holds its
//! async mutex, which keeps the job list and the history consistent.

use std::collections::HashMap;
use std::path::PathBuf;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use openjiuwen_common::error::Result;

use super::store::write_atomic;

/// File holding the run array, relative to the scheduler directory.
const RUNS_FILE: &str = "runs.json";

/// How many run records are kept per job; the oldest are dropped first.
pub(crate) const MAX_RUNS_PER_JOB: usize = 20;

/// How many run records are kept in total, across every job.
pub(crate) const MAX_RUNS_TOTAL: usize = 200;

/// What triggered a recorded run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CronRunTrigger {
    /// The host reported the job through `run`, i.e. an explicit manual execution.
    Manual,
    /// The job's `next_run_at` came due and `due` fired it.
    Scheduled,
}

impl CronRunTrigger {
    /// The trigger name as it is persisted and reported.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Manual => "manual",
            Self::Scheduled => "scheduled",
        }
    }
}

/// One recorded execution of one job.
///
/// The run is a receipt, not a job: the prompt is still executed by the host that reads the
/// record, so `occurrence` says which scheduled instant fired (or when a manual run was asked
/// for) and `next_run_at` says where the job went afterwards.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CronRun {
    /// Unique run identifier, a fresh `uuid4` string.
    pub id: String,
    /// The job that ran.
    pub job_id: String,
    /// Whether the run was scheduled or manual.
    pub trigger: CronRunTrigger,
    /// When the run was recorded, in UTC; the injected clock value of the call.
    pub recorded_at: DateTime<Utc>,
    /// The scheduled occurrence that fired, or the manual run's instant.
    pub occurrence: DateTime<Utc>,
    /// The job's next occurrence after this run; equal to `occurrence` for a job that can never
    /// run again.
    pub next_run_at: DateTime<Utc>,
}

impl CronRun {
    /// Builds a run record with a fresh identifier.
    #[must_use]
    pub fn new(
        job_id: impl Into<String>,
        trigger: CronRunTrigger,
        recorded_at: DateTime<Utc>,
        occurrence: DateTime<Utc>,
        next_run_at: DateTime<Utc>,
    ) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            job_id: job_id.into(),
            trigger,
            recorded_at,
            occurrence,
            next_run_at,
        }
    }
}

/// The persisted, bounded run history of one scheduler directory.
pub(crate) struct CronRunHistory {
    /// Directory holding `runs.json`.
    dir: PathBuf,
}

impl CronRunHistory {
    /// Builds a history persisting at `<dir>/runs.json`.
    pub(crate) fn new(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into() }
    }

    /// The path of `runs.json`.
    pub(crate) fn file(&self) -> PathBuf {
        self.dir.join(RUNS_FILE)
    }

    /// Loads every record, oldest first; a missing file means "no runs yet".
    ///
    /// # Errors
    ///
    /// Returns an error when the file exists but cannot be read or does not hold a JSON array
    /// of [`CronRun`] values.
    pub(crate) async fn load(&self) -> Result<Vec<CronRun>> {
        let bytes = match tokio::fs::read(self.file()).await {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => return Err(error.into()),
        };
        Ok(serde_json::from_slice(&bytes)?)
    }

    /// Appends `run`, rotates the file to its caps and persists it.
    ///
    /// # Errors
    ///
    /// Returns a storage error when the file cannot be read or written.
    pub(crate) async fn record(&self, run: CronRun) -> Result<CronRun> {
        let mut runs = self.load().await?;
        runs.push(run.clone());
        let rotated = rotate(runs);
        let rendered = serde_json::to_string_pretty(&rotated)?;
        write_atomic(&self.dir, RUNS_FILE, &rendered).await?;
        Ok(run)
    }

    /// Drops every record of `job_id`, leaving the other jobs untouched.
    ///
    /// # Errors
    ///
    /// Returns a storage error when the file cannot be read or written.
    pub(crate) async fn remove_job(&self, job_id: &str) -> Result<()> {
        let runs = self.load().await?;
        if runs.iter().all(|run| run.job_id != job_id) {
            return Ok(());
        }
        let kept: Vec<CronRun> = runs
            .into_iter()
            .filter(|run| run.job_id != job_id)
            .collect();
        let rendered = serde_json::to_string_pretty(&kept)?;
        write_atomic(&self.dir, RUNS_FILE, &rendered).await
    }
}

/// Drops the oldest records until both caps hold, keeping chronological order.
fn rotate(runs: Vec<CronRun>) -> Vec<CronRun> {
    let mut per_job: HashMap<String, usize> = HashMap::new();
    let mut kept: Vec<CronRun> = Vec::with_capacity(runs.len());
    for run in runs.into_iter().rev() {
        let seen = per_job.entry(run.job_id.clone()).or_insert(0);
        if *seen >= MAX_RUNS_PER_JOB {
            continue;
        }
        *seen += 1;
        kept.push(run);
    }
    kept.reverse();
    if kept.len() > MAX_RUNS_TOTAL {
        let excess = kept.len() - MAX_RUNS_TOTAL;
        kept.drain(..excess);
    }
    kept
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use chrono::TimeZone;

    use super::*;

    /// Builds a UTC instant at the given wall-clock minute.
    fn at(year: i32, month: u32, day: u32, hour: u32, minute: u32) -> DateTime<Utc> {
        Utc.with_ymd_and_hms(year, month, day, hour, minute, 0)
            .unwrap()
    }

    /// Builds a manual run of `job_id` recorded at `minute` (minutes after the epoch base).
    fn run(job_id: &str, minute: u32) -> CronRun {
        let recorded_at = at(2024, 1, 1, 0, 0) + chrono::Duration::minutes(i64::from(minute));
        CronRun::new(
            job_id,
            CronRunTrigger::Manual,
            recorded_at,
            recorded_at,
            recorded_at,
        )
    }

    #[test]
    fn rotation_keeps_the_newest_records_per_job() {
        let runs: Vec<CronRun> = (0..30).map(|minute| run("a", minute)).collect();
        let kept = rotate(runs);
        assert_eq!(kept.len(), MAX_RUNS_PER_JOB);
        assert_eq!(kept[0].occurrence, at(2024, 1, 1, 0, 10));
        assert_eq!(kept[MAX_RUNS_PER_JOB - 1].occurrence, at(2024, 1, 1, 0, 29));
    }

    #[test]
    fn rotation_caps_the_whole_file() {
        let mut runs = Vec::new();
        for job in 0..30 {
            for minute in 0..MAX_RUNS_PER_JOB {
                runs.push(run(&format!("job-{job}"), u32::try_from(minute).unwrap()));
            }
        }
        let kept = rotate(runs);
        assert_eq!(kept.len(), MAX_RUNS_TOTAL);
        assert_eq!(kept[0].job_id, "job-20", "the oldest jobs drop out first");
    }

    #[test]
    fn rotation_preserves_order_and_other_jobs() {
        let runs = vec![run("a", 0), run("b", 1), run("a", 2)];
        let kept = rotate(runs);
        let ids: Vec<&str> = kept.iter().map(|run| run.job_id.as_str()).collect();
        assert_eq!(ids, vec!["a", "b", "a"]);
    }

    #[tokio::test]
    async fn a_missing_file_is_an_empty_history() {
        let dir = tempfile::tempdir().unwrap();
        let history = CronRunHistory::new(dir.path());
        assert_eq!(history.load().await.unwrap(), Vec::new());
        assert_eq!(history.file(), dir.path().join("runs.json"));
    }

    #[tokio::test]
    async fn record_appends_and_persists() {
        let dir = tempfile::tempdir().unwrap();
        let history = CronRunHistory::new(dir.path());
        let first = history.record(run("a", 0)).await.unwrap();
        let second = history.record(run("a", 1)).await.unwrap();
        assert_eq!(
            history.load().await.unwrap(),
            vec![first.clone(), second.clone()]
        );
        assert_eq!(first.trigger.as_str(), "manual");

        let stored: serde_json::Value =
            serde_json::from_slice(&tokio::fs::read(history.file()).await.unwrap()).unwrap();
        for key in [
            "id",
            "job_id",
            "trigger",
            "recorded_at",
            "occurrence",
            "next_run_at",
        ] {
            assert!(stored[0].get(key).is_some(), "missing key: {key}");
        }
        assert_eq!(stored[0]["trigger"], "manual");
    }

    #[tokio::test]
    async fn record_enforces_the_per_job_cap() {
        let dir = tempfile::tempdir().unwrap();
        let history = CronRunHistory::new(dir.path());
        for minute in 0..(MAX_RUNS_PER_JOB + 5) {
            history
                .record(run("a", u32::try_from(minute).unwrap()))
                .await
                .unwrap();
        }
        let kept = history.load().await.unwrap();
        assert_eq!(kept.len(), MAX_RUNS_PER_JOB);
        assert_eq!(kept[0].occurrence, at(2024, 1, 1, 0, 5));
    }

    #[tokio::test]
    async fn remove_job_drops_only_that_job() {
        let dir = tempfile::tempdir().unwrap();
        let history = CronRunHistory::new(dir.path());
        history.record(run("a", 0)).await.unwrap();
        history.record(run("b", 1)).await.unwrap();
        history.remove_job("a").await.unwrap();
        let kept = history.load().await.unwrap();
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].job_id, "b");

        // Removing a job with no records is a no-op that never creates the file.
        let empty = tempfile::tempdir().unwrap();
        let fresh = CronRunHistory::new(empty.path());
        fresh.remove_job("nobody").await.unwrap();
        assert!(!fresh.file().exists());
    }

    #[tokio::test]
    async fn load_reports_corrupt_files() {
        let dir = tempfile::tempdir().unwrap();
        let history = CronRunHistory::new(dir.path());
        tokio::fs::write(history.file(), b"{not json}")
            .await
            .unwrap();
        assert!(history.load().await.is_err());
    }
}
