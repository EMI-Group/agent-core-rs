//! Unit tests of the persisted job list, its mutations and its run history.
//!
//! Every scenario runs offline over a `tempfile` directory with an injected clock.

use chrono::TimeZone;

use super::*;

/// Builds a UTC instant at the given wall-clock minute.
fn at(year: i32, month: u32, day: u32, hour: u32, minute: u32) -> DateTime<Utc> {
    Utc.with_ymd_and_hms(year, month, day, hour, minute, 0)
        .unwrap()
}

/// Builds a UTC instant at the given wall-clock second.
fn at_second(
    year: i32,
    month: u32,
    day: u32,
    hour: u32,
    minute: u32,
    second: u32,
) -> DateTime<Utc> {
    Utc.with_ymd_and_hms(year, month, day, hour, minute, second)
        .unwrap()
}

/// A parsed `*/15 * * * *` schedule (quarter-hourly).
fn quarter_hourly() -> CronSchedule {
    CronSchedule::parse("*/15 * * * *").unwrap()
}

/// A scheduler plus its temporary directory, kept alive by the caller.
fn scheduler(dir: &tempfile::TempDir) -> Arc<CronScheduler> {
    CronScheduler::new(dir.path())
}

#[tokio::test]
async fn a_missing_file_is_an_empty_list() {
    let dir = tempfile::tempdir().unwrap();
    let scheduler = CronScheduler::new(dir.path());
    assert_eq!(scheduler.list().await.unwrap(), Vec::new());
    assert_eq!(scheduler.path(), dir.path().join("jobs.json"));
    assert_eq!(scheduler.history_path(), dir.path().join("runs.json"));
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
    assert_eq!(
        scheduler.history_path(),
        dir.path().join("cron").join("runs.json")
    );
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

#[tokio::test]
async fn get_returns_the_job_or_none() {
    let dir = tempfile::tempdir().unwrap();
    let scheduler = scheduler(&dir);
    let job = scheduler
        .create(&quarter_hourly(), "say hi", at(2024, 1, 1, 0, 0))
        .await
        .unwrap();
    assert_eq!(scheduler.get(&job.id).await.unwrap(), Some(job));
    assert_eq!(scheduler.get("missing").await.unwrap(), None);
}

#[tokio::test]
async fn update_patches_fields_and_recomputes_the_next_run() {
    let dir = tempfile::tempdir().unwrap();
    let scheduler = scheduler(&dir);
    let job = scheduler
        .create(&quarter_hourly(), "say hi", at(2024, 1, 1, 0, 0))
        .await
        .unwrap();

    let patch = CronUpdate {
        cron: Some("0 9 * * MON-FRI".to_owned()),
        prompt: Some("standup".to_owned()),
        description: Some("weekday standup".to_owned()),
        enabled: None,
    };
    let updated = scheduler
        .update(&job.id, &patch, at(2024, 1, 1, 0, 0))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(updated.cron, "0 9 * * MON-FRI");
    assert_eq!(updated.prompt, "standup");
    assert_eq!(updated.description, "weekday standup");
    assert_eq!(updated.next_run_at, at(2024, 1, 1, 9, 0));
    assert_eq!(scheduler.list().await.unwrap(), vec![updated.clone()]);

    // An empty patch changes nothing, and an unknown id is not an error.
    let untouched = scheduler
        .update(&job.id, &CronUpdate::default(), at(2024, 6, 1, 0, 0))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(untouched, updated);
    assert_eq!(
        scheduler
            .update("missing", &patch, at(2024, 1, 1, 0, 0))
            .await
            .unwrap(),
        None
    );
}

#[tokio::test]
async fn update_rejects_invalid_and_never_matching_expressions() {
    let dir = tempfile::tempdir().unwrap();
    let scheduler = scheduler(&dir);
    let job = scheduler
        .create(&quarter_hourly(), "say hi", at(2024, 1, 1, 0, 0))
        .await
        .unwrap();

    for (cron, expected) in [
        ("75 * * * *", "invalid minute field: \"75\""),
        (
            "0 0 31 2 *",
            "cron expression never matches a future time: \"0 0 31 2 *\"",
        ),
    ] {
        let error = scheduler
            .update(
                &job.id,
                &CronUpdate {
                    cron: Some(cron.to_owned()),
                    ..CronUpdate::default()
                },
                at(2024, 1, 1, 0, 0),
            )
            .await
            .unwrap_err();
        assert_eq!(error.message(), expected);
    }
    assert_eq!(
        scheduler.get(&job.id).await.unwrap(),
        Some(job),
        "a rejected patch leaves the job untouched"
    );
}

#[tokio::test]
async fn disabling_holds_the_next_run_and_enabling_recomputes_it() {
    let dir = tempfile::tempdir().unwrap();
    let scheduler = scheduler(&dir);
    let job = scheduler
        .create(&quarter_hourly(), "say hi", at(2024, 1, 1, 0, 0))
        .await
        .unwrap();
    assert_eq!(job.next_run_at, at(2024, 1, 1, 0, 15));

    let disabled = scheduler
        .set_enabled(&job.id, false, at(2024, 1, 1, 0, 5))
        .await
        .unwrap()
        .unwrap();
    assert!(!disabled.enabled);
    assert_eq!(
        disabled.next_run_at,
        at(2024, 1, 1, 0, 15),
        "disabling keeps the stored occurrence"
    );
    assert_eq!(
        scheduler.due(at(2024, 1, 1, 0, 20)).await.unwrap(),
        Vec::new()
    );

    let enabled = scheduler
        .set_enabled(&job.id, true, at(2024, 1, 1, 5, 7))
        .await
        .unwrap()
        .unwrap();
    assert!(enabled.enabled);
    assert_eq!(
        enabled.next_run_at,
        at(2024, 1, 1, 5, 15),
        "enabling lands on the next occurrence after the clock"
    );
}

#[tokio::test]
async fn toggle_flips_the_enabled_flag() {
    let dir = tempfile::tempdir().unwrap();
    let scheduler = scheduler(&dir);
    let job = scheduler
        .create(&quarter_hourly(), "say hi", at(2024, 1, 1, 0, 0))
        .await
        .unwrap();
    let now = at(2024, 1, 1, 0, 0);

    let off = scheduler.toggle(&job.id, now).await.unwrap().unwrap();
    assert!(!off.enabled);
    let on = scheduler.toggle(&job.id, now).await.unwrap().unwrap();
    assert!(on.enabled);
    assert_eq!(on.next_run_at, at(2024, 1, 1, 0, 15));
    assert_eq!(scheduler.toggle("missing", now).await.unwrap(), None);
}

#[tokio::test]
async fn preview_lists_occurrences_without_mutating_state() {
    let dir = tempfile::tempdir().unwrap();
    let scheduler = scheduler(&dir);
    let job = scheduler
        .create(&quarter_hourly(), "say hi", at(2024, 1, 1, 0, 0))
        .await
        .unwrap();

    let occurrences = scheduler
        .preview(&job.id, 3, at(2024, 1, 1, 0, 0))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        occurrences,
        vec![
            at(2024, 1, 1, 0, 15),
            at(2024, 1, 1, 0, 30),
            at(2024, 1, 1, 0, 45)
        ]
    );
    assert_eq!(
        scheduler.list().await.unwrap()[0],
        job,
        "preview never advances or rewrites the job"
    );
    assert_eq!(
        scheduler
            .preview("missing", 3, at(2024, 1, 1, 0, 0))
            .await
            .unwrap(),
        None
    );

    // A disabled job still previews its expression, and an impossible one is empty.
    scheduler
        .set_enabled(&job.id, false, at(2024, 1, 1, 0, 0))
        .await
        .unwrap();
    assert_eq!(
        scheduler
            .preview(&job.id, 1, at(2024, 1, 1, 0, 0))
            .await
            .unwrap()
            .unwrap()
            .len(),
        1
    );
    let path = scheduler.path();
    let mut jobs: Vec<CronJob> =
        serde_json::from_slice(&tokio::fs::read(&path).await.unwrap()).unwrap();
    jobs[0].cron = "0 0 31 2 *".to_owned();
    tokio::fs::write(&path, serde_json::to_vec(&jobs).unwrap())
        .await
        .unwrap();
    assert!(scheduler
        .preview(&job.id, 5, at(2024, 1, 1, 0, 0))
        .await
        .unwrap()
        .unwrap()
        .is_empty());
}

#[tokio::test]
async fn run_now_advances_the_schedule_and_records_a_run() {
    let dir = tempfile::tempdir().unwrap();
    let scheduler = scheduler(&dir);
    let job = scheduler
        .create(&quarter_hourly(), "say hi", at(2024, 1, 1, 0, 0))
        .await
        .unwrap();

    let now = at(2024, 1, 1, 0, 32);
    let run = scheduler.run_now(&job.id, now).await.unwrap().unwrap();
    assert_eq!(run.job_id, job.id);
    assert_eq!(run.trigger, CronRunTrigger::Manual);
    assert_eq!(run.recorded_at, now);
    assert_eq!(run.occurrence, now);
    assert_eq!(run.next_run_at, at(2024, 1, 1, 0, 45));

    let stored = scheduler.get(&job.id).await.unwrap().unwrap();
    assert_eq!(stored.next_run_at, at(2024, 1, 1, 0, 45));
    assert_eq!(scheduler.runs(Some(&job.id), 10).await.unwrap(), vec![run]);
    assert_eq!(scheduler.run_now("missing", now).await.unwrap(), None);
}

#[tokio::test]
async fn run_now_works_on_a_disabled_job() {
    let dir = tempfile::tempdir().unwrap();
    let scheduler = scheduler(&dir);
    let job = scheduler
        .create(&quarter_hourly(), "say hi", at(2024, 1, 1, 0, 0))
        .await
        .unwrap();
    scheduler
        .set_enabled(&job.id, false, at(2024, 1, 1, 0, 0))
        .await
        .unwrap();

    let run = scheduler
        .run_now(&job.id, at(2024, 1, 1, 0, 32))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(run.trigger, CronRunTrigger::Manual);
    let stored = scheduler.get(&job.id).await.unwrap().unwrap();
    assert!(!stored.enabled, "a manual run does not re-enable the job");
}

#[tokio::test]
async fn runs_are_newest_first_and_filtered() {
    let dir = tempfile::tempdir().unwrap();
    let scheduler = scheduler(&dir);
    let first = scheduler
        .create(&quarter_hourly(), "first", at(2024, 1, 1, 0, 0))
        .await
        .unwrap();
    let second = scheduler
        .create(&quarter_hourly(), "second", at(2024, 1, 1, 0, 0))
        .await
        .unwrap();

    scheduler
        .run_now(&first.id, at(2024, 1, 1, 0, 1))
        .await
        .unwrap();
    scheduler
        .run_now(&second.id, at(2024, 1, 1, 0, 2))
        .await
        .unwrap();
    scheduler
        .run_now(&first.id, at(2024, 1, 1, 0, 3))
        .await
        .unwrap();

    let all = scheduler.runs(None, 0).await.unwrap();
    assert_eq!(all.len(), 3);
    assert_eq!(all[0].recorded_at, at(2024, 1, 1, 0, 3), "newest first");
    assert_eq!(all[2].recorded_at, at(2024, 1, 1, 0, 1));

    let only_first = scheduler.runs(Some(&first.id), 10).await.unwrap();
    assert_eq!(only_first.len(), 2);
    assert!(only_first.iter().all(|run| run.job_id == first.id));

    assert_eq!(scheduler.runs(None, 2).await.unwrap().len(), 2);
    assert_eq!(
        scheduler.runs(Some("missing"), 10).await.unwrap(),
        Vec::new()
    );
}

#[tokio::test]
async fn due_records_a_scheduled_run() {
    let dir = tempfile::tempdir().unwrap();
    let scheduler = scheduler(&dir);
    let job = scheduler
        .create(&quarter_hourly(), "say hi", at(2024, 1, 1, 0, 0))
        .await
        .unwrap();

    assert_eq!(scheduler.due(at(2024, 1, 1, 0, 15)).await.unwrap().len(), 1);
    let runs = scheduler.runs(Some(&job.id), 10).await.unwrap();
    assert_eq!(runs.len(), 1);
    assert_eq!(runs[0].trigger, CronRunTrigger::Scheduled);
    assert_eq!(runs[0].occurrence, at(2024, 1, 1, 0, 15));
    assert_eq!(runs[0].next_run_at, at(2024, 1, 1, 0, 30));
    assert_eq!(runs[0].recorded_at, at(2024, 1, 1, 0, 15));

    // A job that can never run again still records the firing that disabled it.
    let path = scheduler.path();
    let mut jobs: Vec<CronJob> =
        serde_json::from_slice(&tokio::fs::read(&path).await.unwrap()).unwrap();
    jobs[0].cron = "0 0 31 2 *".to_owned();
    tokio::fs::write(&path, serde_json::to_vec(&jobs).unwrap())
        .await
        .unwrap();
    scheduler.due(at(2024, 1, 1, 1, 0)).await.unwrap();
    let runs = scheduler.runs(Some(&job.id), 10).await.unwrap();
    assert_eq!(runs.len(), 2);
    assert_eq!(runs[0].trigger, CronRunTrigger::Scheduled);
}

#[tokio::test]
async fn history_survives_a_reopen_and_drops_with_the_job() {
    let dir = tempfile::tempdir().unwrap();
    let job = {
        let scheduler = CronScheduler::new(dir.path());
        let job = scheduler
            .create(&quarter_hourly(), "say hi", at(2024, 1, 1, 0, 0))
            .await
            .unwrap();
        scheduler
            .run_now(&job.id, at(2024, 1, 1, 0, 20))
            .await
            .unwrap();
        job
    };

    let reopened = CronScheduler::new(dir.path());
    assert_eq!(reopened.runs(Some(&job.id), 10).await.unwrap().len(), 1);
    assert_eq!(reopened.runs(None, 10).await.unwrap().len(), 1);

    reopened.cancel(&job.id).await.unwrap().unwrap();
    assert_eq!(reopened.runs(None, 10).await.unwrap(), Vec::new());
    let stored: serde_json::Value =
        serde_json::from_slice(&tokio::fs::read(reopened.history_path()).await.unwrap()).unwrap();
    assert_eq!(stored, serde_json::json!([]));
}

#[tokio::test]
async fn status_summarizes_jobs_and_runs() {
    let dir = tempfile::tempdir().unwrap();
    let scheduler = scheduler(&dir);
    let now = at(2024, 1, 1, 0, 0);

    let empty = scheduler.status(now).await.unwrap();
    assert_eq!(empty, CronStatus::default());
    assert_eq!(empty.next_run_at, None);

    let first = scheduler
        .create(&quarter_hourly(), "first", now)
        .await
        .unwrap();
    let second = scheduler
        .create(&quarter_hourly(), "second", now)
        .await
        .unwrap();
    scheduler.set_enabled(&second.id, false, now).await.unwrap();
    scheduler
        .run_now(&first.id, at(2024, 1, 1, 0, 1))
        .await
        .unwrap();

    let status = scheduler.status(now).await.unwrap();
    assert_eq!(status.total, 2);
    assert_eq!(status.enabled, 1);
    assert_eq!(status.disabled, 1);
    assert_eq!(status.due, 0);
    assert_eq!(status.runs, 1);
    // `run_now` at 00:01 advanced the first job to the next quarter hour.
    assert_eq!(status.next_run_at, Some(at(2024, 1, 1, 0, 15)));

    scheduler.due(at(2024, 1, 1, 0, 15)).await.unwrap();
    let late = scheduler.status(at(2024, 1, 1, 0, 15)).await.unwrap();
    assert_eq!(late.next_run_at, Some(at(2024, 1, 1, 0, 30)));
}

#[tokio::test]
async fn a_quartz_job_round_trips_through_the_store() {
    let dir = tempfile::tempdir().unwrap();
    let scheduler = scheduler(&dir);
    let schedule = CronSchedule::parse("*/10 0 9 ? * MON-FRI").unwrap();
    let job = scheduler
        .create(&schedule, "tick", at(2024, 1, 1, 0, 0))
        .await
        .unwrap();
    assert_eq!(job.cron, "*/10 0 9 ? * MON-FRI");
    assert_eq!(job.next_run_at, at(2024, 1, 1, 9, 0));

    let occurrences = scheduler
        .preview(&job.id, 2, at(2024, 1, 1, 0, 0))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        occurrences,
        vec![at(2024, 1, 1, 9, 0), at_second(2024, 1, 1, 9, 0, 10)]
    );

    let due = scheduler.due(at(2024, 1, 1, 9, 0)).await.unwrap();
    assert_eq!(due.len(), 1);
    assert_eq!(due[0].cron, "*/10 0 9 ? * MON-FRI");
    assert_eq!(
        scheduler.list().await.unwrap()[0].next_run_at,
        at_second(2024, 1, 1, 9, 0, 10),
        "the stored job advances by one second step"
    );
}
