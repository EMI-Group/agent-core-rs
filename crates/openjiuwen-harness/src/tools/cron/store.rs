//! The persisted cron job list, its run history and the scheduler that owns both.
//!
//! [`CronScheduler`] keeps every job in one JSON array at `<dir>/jobs.json` and the bounded run
//! history in `<dir>/runs.json`. It has no background task and no daemon thread: the host polls
//! [`CronScheduler::due`] with its own clock, and each call returns the jobs whose turn has come
//! while atomically advancing their `next_run_at`. All timestamps are passed in by the caller,
//! so the scheduler never reads the wall clock itself and tests stay deterministic.
//!
//! Concurrency: one async lock per scheduler instance serializes every read-modify-write, and
//! no process-global state exists. Two schedulers over the same directory share only the files;
//! a caller that needs mutual exclusion across instances has to avoid two writers at once.
//! Both files are written through a temporary file plus rename.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;

use openjiuwen_common::error::{Error, Result};

use super::history::{CronRun, CronRunHistory, CronRunTrigger};
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

/// A partial change to a stored job, applied by [`CronScheduler::update`].
///
/// Every field is optional: `None` leaves the stored value untouched, so a patch carries only
/// what the caller wants to change. [`Self::enabled`] recomputes `next_run_at` when it turns a
/// job *on* and holds the stored value when it turns one *off*.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CronUpdate {
    /// Replacement cron expression, re-parsed and validated before it is stored.
    pub cron: Option<String>,
    /// Replacement prompt text.
    pub prompt: Option<String>,
    /// Replacement human description.
    pub description: Option<String>,
    /// Replacement enabled flag.
    pub enabled: Option<bool>,
}

impl CronUpdate {
    /// A patch that only changes the enabled flag.
    #[must_use]
    pub fn enabled(enabled: bool) -> Self {
        Self {
            enabled: Some(enabled),
            ..Self::default()
        }
    }

    /// Whether the patch would change anything.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.cron.is_none()
            && self.prompt.is_none()
            && self.description.is_none()
            && self.enabled.is_none()
    }
}

/// Summary of the job list returned by [`CronScheduler::status`].
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CronStatus {
    /// Number of stored jobs, enabled or not.
    pub total: usize,
    /// Number of enabled jobs.
    pub enabled: usize,
    /// Number of disabled jobs.
    pub disabled: usize,
    /// Number of enabled jobs whose `next_run_at` has already passed at the queried instant.
    pub due: usize,
    /// Earliest pending occurrence among the enabled jobs.
    pub next_run_at: Option<DateTime<Utc>>,
    /// Number of run-history records currently stored, across every job.
    pub runs: usize,
}

/// The persisted cron job list of one scheduler directory.
///
/// The scheduler is cheap to clone through [`Arc`] and holds a single async lock, so every
/// mutation is a consistent load-modify-save sequence.
pub struct CronScheduler {
    /// Directory holding `jobs.json` and `runs.json`.
    dir: PathBuf,
    /// Serializes the load-modify-save sequences of this instance.
    lock: Mutex<()>,
    /// The bounded run history recorded by `due` and `run_now`.
    history: CronRunHistory,
}

impl CronScheduler {
    /// Builds a scheduler persisting at `<dir>/jobs.json` and `<dir>/runs.json`.
    #[must_use]
    pub fn new(dir: impl Into<PathBuf>) -> Arc<Self> {
        let dir = dir.into();
        Arc::new(Self {
            history: CronRunHistory::new(dir.clone()),
            dir,
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

    /// The directory holding the job and history files.
    #[must_use]
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// The absolute or workspace-relative path of `jobs.json`.
    #[must_use]
    pub fn path(&self) -> PathBuf {
        self.dir.join(JOBS_FILE)
    }

    /// The absolute or workspace-relative path of `runs.json`.
    #[must_use]
    pub fn history_path(&self) -> PathBuf {
        self.history.file()
    }

    /// Loads every job, enabled or not; a missing file means "no jobs yet".
    ///
    /// # Errors
    ///
    /// Returns an error when the file exists but cannot be read or does not hold a JSON array
    /// of [`CronJob`] values.
    pub async fn list(&self) -> Result<Vec<CronJob>> {
        let bytes = match tokio::fs::read(self.path()).await {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => return Err(error.into()),
        };
        Ok(serde_json::from_slice(&bytes)?)
    }

    /// Returns the job with `id`, or `None` when no such job exists.
    ///
    /// # Errors
    ///
    /// Returns a storage error when the job file cannot be read.
    pub async fn get(&self, id: &str) -> Result<Option<CronJob>> {
        let _guard = self.lock.lock().await;
        Ok(self.list().await?.into_iter().find(|job| job.id == id))
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
        let next_run_at = schedule
            .next_after(now)
            .ok_or_else(|| never_matches(schedule))?;
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

    /// Applies `patch` to the job with `id`, returning the updated job.
    ///
    /// Changing the expression recomputes `next_run_at` from `now`, and enabling a job does the
    /// same; disabling a job keeps its stored occurrence so the value is not lost. A missing job
    /// yields `Ok(None)`, an empty patch yields the untouched job.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`](openjiuwen_common::error::Error::Validation) when a new
    /// expression does not parse or can never match, and a storage error when the job file
    /// cannot be read or written. A rejected patch leaves the stored job untouched.
    pub async fn update(
        &self,
        id: &str,
        patch: &CronUpdate,
        now: DateTime<Utc>,
    ) -> Result<Option<CronJob>> {
        let patch = patch.clone();
        self.patch(id, move |_| patch, now).await
    }

    /// Sets the enabled flag of the job with `id`, returning the updated job.
    ///
    /// Enabling recomputes `next_run_at` from `now`; disabling keeps the stored value, so a job
    /// that is re-enabled later always lands on a fresh future occurrence.
    ///
    /// # Errors
    ///
    /// Same conditions as [`Self::update`].
    pub async fn set_enabled(
        &self,
        id: &str,
        enabled: bool,
        now: DateTime<Utc>,
    ) -> Result<Option<CronJob>> {
        self.patch(id, move |_| CronUpdate::enabled(enabled), now)
            .await
    }

    /// Flips the enabled flag of the job with `id`, returning the updated job.
    ///
    /// # Errors
    ///
    /// Same conditions as [`Self::update`].
    pub async fn toggle(&self, id: &str, now: DateTime<Utc>) -> Result<Option<CronJob>> {
        self.patch(id, |job| CronUpdate::enabled(!job.enabled), now)
            .await
    }

    /// Returns up to `count` occurrences of the job with `id` strictly after `now`.
    ///
    /// Preview never mutates state: the stored `next_run_at` and the enabled flag are left
    /// exactly as they are, and a disabled job still previews its expression. A missing job
    /// yields `Ok(None)`.
    ///
    /// # Errors
    ///
    /// Returns a validation error when the stored expression no longer parses, and a storage
    /// error when the job file cannot be read.
    pub async fn preview(
        &self,
        id: &str,
        count: usize,
        now: DateTime<Utc>,
    ) -> Result<Option<Vec<DateTime<Utc>>>> {
        let _guard = self.lock.lock().await;
        let jobs = self.list().await?;
        let Some(job) = jobs.iter().find(|job| job.id == id) else {
            return Ok(None);
        };
        let schedule = CronSchedule::parse(&job.cron)?;
        Ok(Some(schedule.next_occurrences(now, count)))
    }

    /// Records a manual execution of the job with `id` and advances its schedule.
    ///
    /// The prompt is **not** executed here: the host runs it when it sees the run record. The
    /// stored job keeps its enabled flag (an explicit manual run is allowed on a disabled job)
    /// and its `next_run_at` moves to the first occurrence strictly after `now`.
    ///
    /// # Errors
    ///
    /// Returns a validation error when the stored expression no longer parses or can never
    /// match, and a storage error when either file cannot be read or written. The schedule is
    /// persisted before the run record, so a history failure leaves an advanced job with no
    /// record.
    pub async fn run_now(&self, id: &str, now: DateTime<Utc>) -> Result<Option<CronRun>> {
        let _guard = self.lock.lock().await;
        let mut jobs = self.list().await?;
        let Some(index) = jobs.iter().position(|job| job.id == id) else {
            return Ok(None);
        };
        let schedule = CronSchedule::parse(&jobs[index].cron)?;
        let next_run_at = schedule
            .next_after(now)
            .ok_or_else(|| never_matches(&schedule))?;
        jobs[index].next_run_at = next_run_at;
        let job_id = jobs[index].id.clone();
        self.persist(&jobs).await?;
        let run = CronRun::new(job_id, CronRunTrigger::Manual, now, now, next_run_at);
        self.history.record(run.clone()).await?;
        Ok(Some(run))
    }

    /// Returns the run history of one job, or of every job, newest first.
    ///
    /// `job_id == None` reads the whole file. `limit` caps the returned entries after sorting;
    /// `0` means "no limit". The file itself is bounded (see the `history` module).
    ///
    /// # Errors
    ///
    /// Returns a storage error when the history file cannot be read.
    pub async fn runs(&self, job_id: Option<&str>, limit: usize) -> Result<Vec<CronRun>> {
        let _guard = self.lock.lock().await;
        let mut runs = self.history.load().await?;
        if let Some(job_id) = job_id {
            runs.retain(|run| run.job_id == job_id);
        }
        runs.reverse();
        if limit > 0 {
            runs.truncate(limit);
        }
        Ok(runs)
    }

    /// Summarizes the job list and the run history at `now`.
    ///
    /// # Errors
    ///
    /// Returns a storage error when either file cannot be read.
    pub async fn status(&self, now: DateTime<Utc>) -> Result<CronStatus> {
        let _guard = self.lock.lock().await;
        let jobs = self.list().await?;
        let runs = self.history.load().await?;
        let enabled = jobs.iter().filter(|job| job.enabled).count();
        let due = jobs
            .iter()
            .filter(|job| job.enabled && job.next_run_at <= now)
            .count();
        let next_run_at = jobs
            .iter()
            .filter(|job| job.enabled)
            .map(|job| job.next_run_at)
            .min();
        Ok(CronStatus {
            total: jobs.len(),
            enabled,
            disabled: jobs.len() - enabled,
            due,
            next_run_at,
            runs: runs.len(),
        })
    }

    /// Removes the job with `id` and its run history, returning the job.
    ///
    /// Returns `None` when no such job exists. The history file is trimmed of the job's records
    /// so `runs` never reports entries of a job that no longer exists.
    ///
    /// # Errors
    ///
    /// Returns a storage error when either file cannot be read or written.
    pub async fn cancel(&self, id: &str) -> Result<Option<CronJob>> {
        let _guard = self.lock.lock().await;
        let mut jobs = self.list().await?;
        let Some(index) = jobs.iter().position(|job| job.id == id) else {
            return Ok(None);
        };
        let removed = jobs.remove(index);
        self.persist(&jobs).await?;
        self.history.remove_job(&removed.id).await?;
        Ok(Some(removed))
    }

    /// Returns the jobs that are due at `now`, advances them and records a run for each.
    ///
    /// A job is due when it is enabled and `next_run_at <= now`. Each returned value is the
    /// snapshot taken *before* the advance, so its `next_run_at` is the occurrence that just
    /// fired; the persisted job keeps the first occurrence strictly after `now`, and a
    /// [`CronRun`] with [`CronRunTrigger::Scheduled`] is appended to the bounded history. A job
    /// whose stored expression no longer parses, or can never match again, is returned once and
    /// then disabled, with a `tracing` warning naming the job, so a corrupt entry cannot fire
    /// forever.
    ///
    /// # Errors
    ///
    /// Returns a storage error when a file cannot be read or written. The job list is persisted
    /// before the run records, so a history failure leaves advanced jobs with fewer records.
    pub async fn due(&self, now: DateTime<Utc>) -> Result<Vec<CronJob>> {
        let _guard = self.lock.lock().await;
        let mut jobs = self.list().await?;
        let mut due = Vec::new();
        let mut fired = Vec::new();
        for job in &mut jobs {
            if !job.enabled || job.next_run_at > now {
                continue;
            }
            let occurrence = job.next_run_at;
            let snapshot = job.clone();
            let advanced = CronSchedule::parse(&job.cron).map(|schedule| schedule.next_after(now));
            if let Ok(Some(next)) = advanced {
                job.next_run_at = next;
                fired.push(CronRun::new(
                    job.id.clone(),
                    CronRunTrigger::Scheduled,
                    now,
                    occurrence,
                    next,
                ));
            } else {
                tracing::warn!(
                    job_id = %job.id,
                    cron = %job.cron,
                    "cron job can never run again; disabling it"
                );
                job.enabled = false;
                fired.push(CronRun::new(
                    job.id.clone(),
                    CronRunTrigger::Scheduled,
                    now,
                    occurrence,
                    occurrence,
                ));
            }
            due.push(snapshot);
        }
        if due.is_empty() {
            return Ok(due);
        }
        self.persist(&jobs).await?;
        for run in fired {
            self.history.record(run).await?;
        }
        Ok(due)
    }

    /// Applies `patch` — computed from the current job — under the scheduler lock.
    async fn patch(
        &self,
        id: &str,
        make: impl FnOnce(&CronJob) -> CronUpdate,
        now: DateTime<Utc>,
    ) -> Result<Option<CronJob>> {
        let _guard = self.lock.lock().await;
        let mut jobs = self.list().await?;
        let Some(index) = jobs.iter().position(|job| job.id == id) else {
            return Ok(None);
        };
        let patch = make(&jobs[index]);
        apply_patch(&mut jobs[index], &patch, now)?;
        let updated = jobs[index].clone();
        self.persist(&jobs).await?;
        Ok(Some(updated))
    }

    /// Replaces `jobs.json` through an atomic rename.
    async fn persist(&self, jobs: &[CronJob]) -> Result<()> {
        let rendered = serde_json::to_string_pretty(jobs)?;
        write_atomic(&self.dir, JOBS_FILE, &rendered).await
    }
}

/// Applies one patch to one job, validating a replacement expression.
fn apply_patch(job: &mut CronJob, patch: &CronUpdate, now: DateTime<Utc>) -> Result<()> {
    if let Some(cron) = &patch.cron {
        let schedule = CronSchedule::parse(cron)?;
        let next_run_at = schedule
            .next_after(now)
            .ok_or_else(|| never_matches(&schedule))?;
        schedule.expression().clone_into(&mut job.cron);
        job.next_run_at = next_run_at;
    }
    if let Some(prompt) = &patch.prompt {
        job.prompt.clone_from(prompt);
    }
    if let Some(description) = &patch.description {
        job.description.clone_from(description);
    }
    if let Some(enabled) = patch.enabled {
        job.enabled = enabled;
        if enabled {
            let schedule = CronSchedule::parse(&job.cron)?;
            let next_run_at = schedule
                .next_after(now)
                .ok_or_else(|| never_matches(&schedule))?;
            job.next_run_at = next_run_at;
        }
    }
    Ok(())
}

/// The shared "this expression can never run" validation error.
fn never_matches(schedule: &CronSchedule) -> Error {
    Error::validation(format!(
        "cron expression never matches a future time: {:?}",
        schedule.expression()
    ))
}

/// Writes `contents` to `<dir>/<file>` through a temporary file and an atomic rename.
///
/// Creates the directory when needed. A platform that cannot replace the target in place
/// (Windows) gets one retry after removing the target, so the write stays all-or-nothing.
pub(super) async fn write_atomic(dir: &Path, file: &str, contents: &str) -> Result<()> {
    tokio::fs::create_dir_all(dir).await?;
    let path = dir.join(file);
    let temporary = dir.join(format!("{file}.tmp-{}", uuid::Uuid::new_v4()));
    tokio::fs::write(&temporary, contents.as_bytes()).await?;
    if let Err(error) = tokio::fs::rename(&temporary, &path).await {
        tracing::debug!(%error, path = %path.display(), "retrying cron rename");
        let _ = tokio::fs::remove_file(&path).await;
        if let Err(error) = tokio::fs::rename(&temporary, &path).await {
            let _ = tokio::fs::remove_file(&temporary).await;
            return Err(error.into());
        }
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
#[path = "store_tests.rs"]
mod tests;
