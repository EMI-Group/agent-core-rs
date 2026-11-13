//! The persisted cron job list and the scheduler that owns it.
//!
//! [`CronScheduler`] keeps every job in one JSON array at `<dir>/jobs.json`. It has no
//! background task and no daemon thread: the host polls [`CronScheduler::due`] with its own
//! clock, and each call returns the jobs whose turn has come while atomically advancing their
//! `next_run_at`. All timestamps are passed in by the caller, so the scheduler never reads the
//! wall clock itself and tests stay deterministic.
//!
//! Concurrency: one async lock per scheduler instance serializes every read-modify-write, and
//! no process-global state exists. Two schedulers over the same directory share only the file;
//! a caller that needs mutual exclusion across instances has to avoid two writers at once.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;

use openjiuwen_common::error::{Error, Result};

use super::schedule::CronSchedule;
use crate::workspace::Workspace;

/// File holding the job array, relative to the scheduler directory.
const JOBS_FILE: &str = "jobs.json";

/// Top-level workspace directory used when the schema has no `cron` node.
const CRON_DIR: &str = "cron";

/// One scheduled prompt.
///
/// Python parity: the job payload of the cron backend (`id`, `schedule.cron`, `nextRunAt`,
/// `createdAt`, `enabled`, `payload.text`). The Rust shape is flat, keeps the ISO-8601
/// timestamps in RFC 3339 form, and stores the prompt text and its human description
/// separately instead of nesting a `payload`/`schedule` object.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CronJob {
    /// Unique job identifier, a fresh `uuid4` string generated on creation.
    pub id: String,
    /// The cron expression as it was supplied; re-parsed on every scheduling decision.
    pub cron: String,
    /// When the job should fire next, in UTC.
    pub next_run_at: DateTime<Utc>,
    /// When the job was created, in UTC; the injected clock value of `create`.
    pub created_at: DateTime<Utc>,
    /// Whether the job is still scheduled; a disabled job is kept but never becomes due.
    #[serde(default = "enabled_by_default")]
    pub enabled: bool,
    /// The prompt handed to the agent when the job fires.
    #[serde(default)]
    pub prompt: String,
    /// Optional human description of the job, shown by `list` but never executed.
    #[serde(default)]
    pub description: String,
}

/// Serde default for [`CronJob::enabled`]: a job without the field is scheduled.
fn enabled_by_default() -> bool {
    true
}

/// The persisted cron job list of one scheduler directory.
///
/// The scheduler is cheap to clone through [`Arc`] and holds a single async lock, so every
/// mutation is a consistent load-modify-save sequence.
pub struct CronScheduler {
    /// Directory holding `jobs.json`.
    dir: PathBuf,
    /// Serializes the load-modify-save sequences of this instance.
    lock: Mutex<()>,
}

impl CronScheduler {
    /// Builds a scheduler persisting at `<dir>/jobs.json`.
    #[must_use]
    pub fn new(dir: impl Into<PathBuf>) -> Arc<Self> {
        Arc::new(Self {
            dir: dir.into(),
            lock: Mutex::new(()),
        })
    }

    /// Builds a scheduler over the workspace `cron` node, falling back to `<root_path>/cron`.
    ///
    /// The default workspace schema has no `cron` node, so the fallback is the common case;
    /// a schema that declares one keeps the jobs under that node instead.
    #[must_use]
    pub fn for_workspace(workspace: &Workspace) -> Arc<Self> {
        let dir = workspace
            .get_node_path(CRON_DIR)
            .unwrap_or_else(|| workspace.root_path.join(CRON_DIR));
        Self::new(dir)
    }

    /// The directory holding the job file.
    #[must_use]
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// The absolute or workspace-relative path of `jobs.json`.
    #[must_use]
    pub fn path(&self) -> PathBuf {
        self.dir.join(JOBS_FILE)
    }

    /// Loads every job, enabled or not; a missing file means "no jobs yet".
    ///
    /// # Errors
    ///
    /// Returns an error when the file exists but cannot be read or does not hold a JSON array
    /// of [`CronJob`] values.
    pub async fn list(&self) -> Result<Vec<CronJob>> {
        let path = self.path();
        let bytes = match tokio::fs::read(&path).await {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => return Err(error.into()),
        };
        Ok(serde_json::from_slice(&bytes)?)
    }

    /// Creates a job from `schedule` and persists it.
    ///
    /// `prompt` is the text handed to the agent when the job fires; use
    /// [`Self::create_with_description`] to attach a human description as well. The first run
    /// is the first occurrence strictly after `now`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`](openjiuwen_common::error::Error::Validation) when the
    /// schedule never matches in the search horizon, and a storage error when the job file
    /// cannot be read or written.
    pub async fn create(
        &self,
        schedule: &CronSchedule,
        prompt: impl Into<String>,
        now: DateTime<Utc>,
    ) -> Result<CronJob> {
        self.create_with_description(schedule, prompt, String::new(), now)
            .await
    }

    /// Creates a job with both a prompt and a description, and persists it.
    ///
    /// # Errors
    ///
    /// Same conditions as [`Self::create`].
    pub async fn create_with_description(
        &self,
        schedule: &CronSchedule,
        prompt: impl Into<String>,
        description: impl Into<String>,
        now: DateTime<Utc>,
    ) -> Result<CronJob> {
        let next_run_at = schedule.next_after(now).ok_or_else(|| {
            Error::validation(format!(
                "cron expression never matches a future time: {:?}",
                schedule.expression()
            ))
        })?;
        let job = CronJob {
            id: uuid::Uuid::new_v4().to_string(),
            cron: schedule.expression().to_owned(),
            next_run_at,
            created_at: now,
            enabled: true,
            prompt: prompt.into(),
            description: description.into(),
        };

        let _guard = self.lock.lock().await;
        let mut jobs = self.list().await?;
        jobs.push(job.clone());
        self.persist(&jobs).await?;
        Ok(job)
    }

    /// Removes the job with `id`, returning it, or `None` when no such job exists.
    ///
    /// # Errors
    ///
    /// Returns a storage error when the job file cannot be read or written.
    pub async fn cancel(&self, id: &str) -> Result<Option<CronJob>> {
        let _guard = self.lock.lock().await;
        let mut jobs = self.list().await?;
        let Some(index) = jobs.iter().position(|job| job.id == id) else {
            return Ok(None);
        };
        let removed = jobs.remove(index);
        self.persist(&jobs).await?;
        Ok(Some(removed))
    }

    /// Returns the jobs that are due at `now` and advances their next occurrence.
    ///
    /// A job is due when it is enabled and `next_run_at <= now`. Each returned value is the
    /// snapshot taken *before* the advance, so its `next_run_at` is the occurrence that just
    /// fired; the persisted job keeps the first occurrence strictly after `now`. A job whose
    /// stored expression no longer parses, or can never match again, is returned once and then
    /// disabled, with a `tracing` warning naming the job, so a corrupt entry cannot fire
    /// forever.
    ///
    /// # Errors
    ///
    /// Returns a storage error when the job file cannot be read or written.
    pub async fn due(&self, now: DateTime<Utc>) -> Result<Vec<CronJob>> {
        let _guard = self.lock.lock().await;
        let mut jobs = self.list().await?;
        let mut due = Vec::new();
        for job in &mut jobs {
            if !job.enabled || job.next_run_at > now {
                continue;
            }
            let fired = job.clone();
            let advanced = CronSchedule::parse(&job.cron).map(|schedule| schedule.next_after(now));
            if let Ok(Some(next)) = advanced {
                job.next_run_at = next;
            } else {
                tracing::warn!(
                    job_id = %job.id,
                    cron = %job.cron,
                    "cron job can never run again; disabling it"
                );
                job.enabled = false;
            }
            due.push(fired);
        }
        if due.is_empty() {
            return Ok(due);
        }
        self.persist(&jobs).await?;
        Ok(due)
    }

    /// Replaces `jobs.json` through an atomic rename.
    async fn persist(&self, jobs: &[CronJob]) -> Result<()> {
        let path = self.path();
        tokio::fs::create_dir_all(&self.dir).await?;
        let rendered = serde_json::to_string_pretty(jobs)?;
        let temporary = self
            .dir
            .join(format!("{JOBS_FILE}.tmp-{}", uuid::Uuid::new_v4()));
        tokio::fs::write(&temporary, rendered.as_bytes()).await?;
        if let Err(error) = tokio::fs::rename(&temporary, &path).await {
            // Windows cannot replace the target in place; remove it and retry once.
            tracing::debug!(%error, path = %path.display(), "retrying cron rename");
            let _ = tokio::fs::remove_file(&path).await;
            if let Err(error) = tokio::fs::rename(&temporary, &path).await {
                let _ = tokio::fs::remove_file(&temporary).await;
                return Err(error.into());
            }
        }
        Ok(())
    }
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

    /// A parsed `*/15 * * * *` schedule (quarter-hourly).
    fn quarter_hourly() -> CronSchedule {
        CronSchedule::parse("*/15 * * * *").unwrap()
    }

    #[tokio::test]
    async fn a_missing_file_is_an_empty_list() {
        let dir = tempfile::tempdir().unwrap();
        let scheduler = CronScheduler::new(dir.path());
        assert_eq!(scheduler.list().await.unwrap(), Vec::new());
        assert_eq!(scheduler.path(), dir.path().join("jobs.json"));
        assert_eq!(scheduler.dir(), dir.path());
    }

    #[tokio::test]
    async fn create_persists_a_job() {
        let dir = tempfile::tempdir().unwrap();
        let scheduler = CronScheduler::new(dir.path());
        let now = at(2024, 1, 1, 0, 0);

        let job = scheduler
            .create_with_description(&quarter_hourly(), "say hi", "greeting", now)
            .await
            .unwrap();

        assert_eq!(job.cron, "*/15 * * * *");
        assert_eq!(job.prompt, "say hi");
        assert_eq!(job.description, "greeting");
        assert_eq!(job.created_at, now);
        assert_eq!(job.next_run_at, at(2024, 1, 1, 0, 15));
        assert!(job.enabled);
        assert_eq!(job.id.len(), 36, "a hyphenated uuid4 string");
        assert_eq!(scheduler.list().await.unwrap(), vec![job]);
    }

    #[tokio::test]
    async fn create_without_a_description_leaves_it_empty() {
        let dir = tempfile::tempdir().unwrap();
        let scheduler = CronScheduler::new(dir.path());
        let job = scheduler
            .create(&quarter_hourly(), "say hi", at(2024, 1, 1, 0, 0))
            .await
            .unwrap();
        assert_eq!(job.description, "");
    }

    #[tokio::test]
    async fn jobs_survive_a_second_scheduler_over_the_same_directory() {
        let dir = tempfile::tempdir().unwrap();
        let now = at(2024, 1, 1, 0, 0);
        let job = CronScheduler::new(dir.path())
            .create(&quarter_hourly(), "say hi", now)
            .await
            .unwrap();

        let reopened = CronScheduler::new(dir.path());
        assert_eq!(reopened.list().await.unwrap(), vec![job.clone()]);
        assert!(reopened.cancel(&job.id).await.unwrap().is_some());
        assert_eq!(
            CronScheduler::new(dir.path()).list().await.unwrap(),
            Vec::new()
        );
    }

    #[tokio::test]
    async fn create_rejects_a_schedule_that_never_matches() {
        let dir = tempfile::tempdir().unwrap();
        let scheduler = CronScheduler::new(dir.path());
        let schedule = CronSchedule::parse("0 0 31 2 *").unwrap();
        let error = scheduler
            .create(&schedule, "never", at(2024, 1, 1, 0, 0))
            .await
            .unwrap_err();
        assert_eq!(
            error.message(),
            "cron expression never matches a future time: \"0 0 31 2 *\""
        );
        assert_eq!(scheduler.list().await.unwrap(), Vec::new());
    }

    #[tokio::test]
    async fn due_only_returns_jobs_whose_time_has_come() {
        let dir = tempfile::tempdir().unwrap();
        let scheduler = CronScheduler::new(dir.path());
        let job = scheduler
            .create(&quarter_hourly(), "say hi", at(2024, 1, 1, 0, 0))
            .await
            .unwrap();

        assert_eq!(
            scheduler.due(at(2024, 1, 1, 0, 14)).await.unwrap(),
            Vec::new()
        );

        let due = scheduler.due(at(2024, 1, 1, 0, 15)).await.unwrap();
        assert_eq!(due.len(), 1);
        assert_eq!(due[0].id, job.id);
        assert_eq!(
            due[0].next_run_at,
            at(2024, 1, 1, 0, 15),
            "the fired occurrence"
        );

        let persisted = scheduler.list().await.unwrap();
        assert_eq!(
            persisted[0].next_run_at,
            at(2024, 1, 1, 0, 30),
            "the stored job advances to the next occurrence"
        );
        assert_eq!(
            scheduler.due(at(2024, 1, 1, 0, 15)).await.unwrap(),
            Vec::new()
        );
    }

    #[tokio::test]
    async fn due_catches_up_with_the_latest_occurrence() {
        let dir = tempfile::tempdir().unwrap();
        let scheduler = CronScheduler::new(dir.path());
        scheduler
            .create(&quarter_hourly(), "say hi", at(2024, 1, 1, 0, 0))
            .await
            .unwrap();

        let due = scheduler.due(at(2024, 1, 1, 3, 47)).await.unwrap();
        assert_eq!(due.len(), 1);
        assert_eq!(due[0].next_run_at, at(2024, 1, 1, 0, 15));
        assert_eq!(
            scheduler.list().await.unwrap()[0].next_run_at,
            at(2024, 1, 1, 4, 0)
        );
    }

    #[tokio::test]
    async fn due_skips_disabled_jobs() {
        let dir = tempfile::tempdir().unwrap();
        let scheduler = CronScheduler::new(dir.path());
        scheduler
            .create(&quarter_hourly(), "say hi", at(2024, 1, 1, 0, 0))
            .await
            .unwrap();

        let path = scheduler.path();
        let mut jobs: Vec<CronJob> =
            serde_json::from_slice(&tokio::fs::read(&path).await.unwrap()).unwrap();
        jobs[0].enabled = false;
        tokio::fs::write(&path, serde_json::to_vec(&jobs).unwrap())
            .await
            .unwrap();

        assert_eq!(
            scheduler.due(at(2024, 1, 1, 1, 0)).await.unwrap(),
            Vec::new()
        );
    }

    #[tokio::test]
    async fn due_disables_a_job_whose_expression_broke() {
        let dir = tempfile::tempdir().unwrap();
        let scheduler = CronScheduler::new(dir.path());
        let job = scheduler
            .create(&quarter_hourly(), "say hi", at(2024, 1, 1, 0, 0))
            .await
            .unwrap();

        let path = scheduler.path();
        let mut jobs: Vec<CronJob> =
            serde_json::from_slice(&tokio::fs::read(&path).await.unwrap()).unwrap();
        jobs[0].cron = "not a cron expression".to_owned();
        tokio::fs::write(&path, serde_json::to_vec(&jobs).unwrap())
            .await
            .unwrap();

        let due = scheduler.due(at(2024, 1, 1, 0, 15)).await.unwrap();
        assert_eq!(due.len(), 1);
        assert_eq!(due[0].id, job.id);

        let persisted = scheduler.list().await.unwrap();
        assert!(!persisted[0].enabled, "a broken job is disabled");
        assert_eq!(
            scheduler.due(at(2024, 1, 1, 1, 0)).await.unwrap(),
            Vec::new()
        );
    }

    #[tokio::test]
    async fn due_disables_a_job_that_never_matches_again() {
        let dir = tempfile::tempdir().unwrap();
        let scheduler = CronScheduler::new(dir.path());
        scheduler
            .create(&quarter_hourly(), "say hi", at(2024, 1, 1, 0, 0))
            .await
            .unwrap();

        let path = scheduler.path();
        let mut jobs: Vec<CronJob> =
            serde_json::from_slice(&tokio::fs::read(&path).await.unwrap()).unwrap();
        jobs[0].cron = "0 0 31 2 *".to_owned();
        tokio::fs::write(&path, serde_json::to_vec(&jobs).unwrap())
            .await
            .unwrap();

        let due = scheduler.due(at(2024, 1, 1, 0, 15)).await.unwrap();
        assert_eq!(due.len(), 1);
        assert!(!scheduler.list().await.unwrap()[0].enabled);
    }

    #[tokio::test]
    async fn cancel_removes_only_the_named_job() {
        let dir = tempfile::tempdir().unwrap();
        let scheduler = CronScheduler::new(dir.path());
        let first = scheduler
            .create(&quarter_hourly(), "first", at(2024, 1, 1, 0, 0))
            .await
            .unwrap();
        let second = scheduler
            .create(&quarter_hourly(), "second", at(2024, 1, 1, 0, 0))
            .await
            .unwrap();

        let removed = scheduler.cancel(&first.id).await.unwrap().unwrap();
        assert_eq!(removed, first);
        assert!(scheduler.cancel(&first.id).await.unwrap().is_none());
        assert!(scheduler.cancel("missing").await.unwrap().is_none());
        assert_eq!(scheduler.list().await.unwrap(), vec![second]);
    }

    #[tokio::test]
    async fn list_reports_corrupt_files() {
        let dir = tempfile::tempdir().unwrap();
        let scheduler = CronScheduler::new(dir.path());
        tokio::fs::write(scheduler.path(), b"{not json}")
            .await
            .unwrap();
        assert!(scheduler.list().await.is_err());
    }

    #[test]
    fn for_workspace_uses_the_cron_directory() {
        let dir = tempfile::tempdir().unwrap();
        let workspace = Workspace::new(dir.path());
        let scheduler = CronScheduler::for_workspace(&workspace);
        assert_eq!(scheduler.dir(), dir.path().join("cron"));
        assert_eq!(scheduler.path(), dir.path().join("cron").join("jobs.json"));
    }

    #[tokio::test]
    async fn persisted_json_keeps_the_python_keys() {
        let dir = tempfile::tempdir().unwrap();
        let scheduler = CronScheduler::new(dir.path());
        scheduler
            .create_with_description(
                &quarter_hourly(),
                "say hi",
                "greeting",
                at(2024, 1, 1, 0, 0),
            )
            .await
            .unwrap();

        let stored: serde_json::Value =
            serde_json::from_slice(&tokio::fs::read(scheduler.path()).await.unwrap()).unwrap();
        let job = &stored[0];
        for key in [
            "id",
            "cron",
            "next_run_at",
            "created_at",
            "enabled",
            "prompt",
            "description",
        ] {
            assert!(job.get(key).is_some(), "missing key: {key}");
        }
        assert_eq!(job["next_run_at"], "2024-01-01T00:15:00Z");
    }
}
