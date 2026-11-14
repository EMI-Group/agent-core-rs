//! Cross-module cron suite: the tool, the scheduler, the history and the expression engine end
//! to end.
//!
//! Every scenario runs offline over a `tempfile` directory with an injected clock, so the
//! suite is deterministic and needs no daemon.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;

use chrono::{DateTime, TimeZone, Utc};
use serde_json::json;

use openjiuwen_common::error::Result;
use openjiuwen_foundation::tool::Tool;
use openjiuwen_foundation::tool::ToolContext;

use super::{CronJob, CronRun, CronRunTrigger, CronSchedule, CronScheduler, CronStatus, CronTool};
use crate::tools::support::ToolPayload;
use crate::workspace::Workspace;

/// Builds a UTC instant at the given wall-clock minute.
fn at(year: i32, month: u32, day: u32, hour: u32, minute: u32) -> DateTime<Utc> {
    Utc.with_ymd_and_hms(year, month, day, hour, minute, 0)
        .unwrap()
}

/// Creates a job through the tool and returns the persisted job.
async fn create_job(tool: &CronTool, cron: &str, prompt: &str, now: DateTime<Utc>) -> CronJob {
    let payload = tool
        .invoke_at(
            &json!({"action": "create", "cron": cron, "prompt": prompt}),
            now,
        )
        .await;
    assert!(payload.success, "create failed: {:?}", payload.error);
    let data = payload.data.unwrap();
    serde_json::from_value(data["job"].clone()).unwrap()
}

#[tokio::test]
async fn end_to_end_create_list_due_cancel() -> Result<()> {
    let dir = tempfile::tempdir().unwrap();
    let scheduler = CronScheduler::new(dir.path());
    let tool = CronTool::new(Arc::clone(&scheduler), "en");
    let created = at(2024, 1, 1, 0, 0);

    let job = create_job(&tool, "*/15 * * * *", "say hi", created).await;
    assert_eq!(job.next_run_at, at(2024, 1, 1, 0, 15));

    let listed = tool.invoke_at(&json!({"action": "list"}), created).await;
    assert!(listed.success);
    let jobs = listed.data.unwrap()["jobs"].as_array().unwrap().clone();
    assert_eq!(jobs.len(), 1);
    assert_eq!(jobs[0]["id"], job.id);

    // Nothing is due before the first occurrence.
    assert_eq!(scheduler.due(at(2024, 1, 1, 0, 14)).await?, Vec::new());

    // The first occurrence fires once, then the stored job advances.
    let due = scheduler.due(at(2024, 1, 1, 0, 15)).await?;
    assert_eq!(due.len(), 1);
    assert_eq!(due[0].id, job.id);
    assert_eq!(due[0].prompt, "say hi");
    assert_eq!(
        scheduler.list().await?[0].next_run_at,
        at(2024, 1, 1, 0, 30)
    );
    assert_eq!(scheduler.due(at(2024, 1, 1, 0, 15)).await?, Vec::new());

    // The firing is recorded in the run history, visible through the tool.
    let runs = tool.invoke_at(&json!({"action": "runs"}), created).await;
    assert!(runs.success);
    let data = runs.data.unwrap();
    assert_eq!(data["count"], 1);
    assert_eq!(data["runs"][0]["trigger"], "scheduled");
    let run: CronRun = serde_json::from_value(data["runs"][0].clone()).unwrap();
    assert_eq!(run.trigger, CronRunTrigger::Scheduled);
    assert_eq!(run.occurrence, at(2024, 1, 1, 0, 15));

    // The next occurrence fires too, at its own time.
    let due = scheduler.due(at(2024, 1, 1, 0, 44)).await?;
    assert_eq!(due.len(), 1);
    assert_eq!(due[0].next_run_at, at(2024, 1, 1, 0, 30));
    assert_eq!(
        scheduler.list().await?[0].next_run_at,
        at(2024, 1, 1, 0, 45)
    );

    let cancelled = tool
        .invoke_at(&json!({"action": "cancel", "job_id": job.id}), created)
        .await;
    assert!(cancelled.success, "cancel failed: {:?}", cancelled.error);
    assert_eq!(cancelled.data.unwrap()["job_id"], job.id);

    let listed = tool.invoke_at(&json!({"action": "list"}), created).await;
    assert_eq!(listed.data.unwrap()["jobs"], json!([]));
    assert_eq!(scheduler.due(at(2024, 1, 1, 5, 0)).await?, Vec::new());
    // Cancelling drops the job's history with it.
    assert_eq!(scheduler.runs(None, 0).await?, Vec::new());
    Ok(())
}

#[tokio::test]
async fn end_to_end_get_update_toggle_preview_run_status() -> Result<()> {
    let dir = tempfile::tempdir().unwrap();
    let scheduler = CronScheduler::new(dir.path());
    let tool = CronTool::new(Arc::clone(&scheduler), "en");
    let now = at(2024, 1, 1, 0, 0);

    let job = create_job(&tool, "*/15 * * * *", "say hi", now).await;

    let fetched = tool
        .invoke_at(&json!({"action": "get_job", "job_id": job.id}), now)
        .await;
    assert!(fetched.success, "get_job failed: {:?}", fetched.error);
    let got: CronJob = serde_json::from_value(fetched.data.unwrap()["job"].clone()).unwrap();
    assert_eq!(got, job);

    let updated = tool
        .invoke_at(
            &json!({
                "action": "update",
                "job_id": job.id,
                "cron": "0 9 * * MON-FRI",
                "prompt": "standup",
            }),
            now,
        )
        .await;
    assert!(updated.success, "update failed: {:?}", updated.error);
    let updated: CronJob = serde_json::from_value(updated.data.unwrap()["job"].clone()).unwrap();
    assert_eq!(updated.cron, "0 9 * * MON-FRI");
    assert_eq!(updated.next_run_at, at(2024, 1, 1, 9, 0));

    let toggled = tool
        .invoke_at(&json!({"action": "toggle", "job_id": job.id}), now)
        .await;
    assert!(toggled.success);
    assert_eq!(toggled.data.unwrap()["job"]["enabled"], false);
    assert_eq!(scheduler.due(at(2024, 1, 1, 9, 0)).await?, Vec::new());

    let preview = tool
        .invoke_at(
            &json!({"action": "preview", "job_id": job.id, "count": 2}),
            at(2024, 1, 5, 10, 0),
        )
        .await;
    assert!(preview.success);
    assert_eq!(
        preview.data.unwrap()["occurrences"],
        json!(["2024-01-08T09:00:00Z", "2024-01-09T09:00:00Z"])
    );

    let run = tool
        .invoke_at(
            &json!({"action": "run", "job_id": job.id}),
            at(2024, 1, 5, 10, 0),
        )
        .await;
    assert!(run.success, "run failed: {:?}", run.error);
    let recorded: CronRun = serde_json::from_value(run.data.unwrap()["run"].clone()).unwrap();
    assert_eq!(recorded.trigger, CronRunTrigger::Manual);
    assert!(!scheduler.get(&job.id).await?.unwrap().enabled);

    let status = tool.invoke_at(&json!({"action": "status"}), now).await;
    assert!(status.success);
    let status: CronStatus =
        serde_json::from_value(status.data.unwrap()["status"].clone()).unwrap();
    assert_eq!(status.total, 1);
    assert_eq!(status.disabled, 1);
    assert_eq!(status.runs, 1);
    Ok(())
}

#[tokio::test]
async fn a_quartz_job_fires_at_second_granularity() -> Result<()> {
    let dir = tempfile::tempdir().unwrap();
    let scheduler = CronScheduler::new(dir.path());
    let tool = CronTool::new(Arc::clone(&scheduler), "en");

    // Every 10 seconds within the 09:00 minute of every weekday.
    let job = create_job(&tool, "*/10 0 9 ? * MON-FRI", "tick", at(2024, 1, 1, 0, 0)).await;
    assert_eq!(job.next_run_at, at(2024, 1, 1, 9, 0));

    let due = scheduler.due(at(2024, 1, 1, 9, 0)).await?;
    assert_eq!(due.len(), 1);
    assert_eq!(
        scheduler.list().await?[0].next_run_at,
        at(2024, 1, 1, 9, 0) + chrono::Duration::seconds(10)
    );

    // The last Friday of January 2024 is the 26th.
    let last_friday = create_job(&tool, "0 0 0 ? * 5L", "retro", at(2024, 1, 1, 0, 0)).await;
    assert_eq!(last_friday.next_run_at, at(2024, 1, 26, 0, 0));

    // A year-restricted job keeps its occurrence beyond the five-year horizon.
    let yearly = create_job(&tool, "0 0 12 1 1 ? 2030", "new year", at(2024, 1, 1, 0, 0)).await;
    assert_eq!(yearly.next_run_at, at(2030, 1, 1, 12, 0));
    Ok(())
}

#[tokio::test]
async fn jobs_and_history_survive_a_second_scheduler_instance() -> Result<()> {
    let dir = tempfile::tempdir().unwrap();
    let now = at(2024, 1, 1, 0, 0);
    let schedule = CronSchedule::parse("0 9 * * MON-FRI")?;

    let job = CronScheduler::new(dir.path())
        .create_with_description(&schedule, "standup", "morning standup", now)
        .await?;

    let reopened = CronScheduler::new(dir.path());
    let jobs = reopened.list().await?;
    assert_eq!(jobs, vec![job.clone()]);
    assert_eq!(jobs[0].description, "morning standup");

    // The reopened scheduler schedules the same job and records its firing.
    let due = reopened.due(at(2024, 1, 1, 9, 0)).await?;
    assert_eq!(due.len(), 1);
    assert_eq!(due[0].id, job.id);
    assert_eq!(CronScheduler::new(dir.path()).runs(None, 0).await?.len(), 1);

    assert!(reopened.cancel(&job.id).await?.is_some());
    assert_eq!(CronScheduler::new(dir.path()).list().await?, Vec::new());
    Ok(())
}

#[tokio::test]
async fn invalid_cron_reports_a_failure_payload() {
    let dir = tempfile::tempdir().unwrap();
    let tool = CronTool::new(CronScheduler::new(dir.path()), "en");
    let now = at(2024, 1, 1, 0, 0);

    for (cron, expected) in [
        ("75 * * * *", "invalid minute field: \"75\""),
        ("0 0 31 2 *", "cron expression never matches a future time: \"0 0 31 2 *\""),
        ("not a cron expression", "invalid cron expression: expected 5 fields (minute hour day-of-month month day-of-week) or 6/7 Quartz fields (second minute hour day-of-month month day-of-week [year]), got 4: \"not a cron expression\""),
        ("@sometimes", "unknown cron shorthand: \"@sometimes\""),
        (
            "0 0 12 1 * MON",
            "invalid cron expression: both day-of-month and day-of-week are specified (one must be \"?\" in a Quartz expression): \"0 0 12 1 * MON\"",
        ),
        (
            "0 0 0 15W * ?",
            "unsupported cron token: \"15W\": \"W\" (nearest weekday) is not supported",
        ),
        (
            "0 0 0 ? * 1#6",
            "invalid day-of-week field: \"1#6\"",
        ),
    ] {
        let payload = tool
            .invoke_at(&json!({"action": "create", "cron": cron, "prompt": "x"}), now)
            .await;
        assert!(!payload.success, "accepted: {cron}");
        assert_eq!(payload.error.as_deref(), Some(expected), "for: {cron}");
        assert!(payload.data.is_none());
    }
    assert_eq!(tool.scheduler().unwrap().list().await.unwrap(), Vec::new());
}

#[tokio::test]
async fn workspace_backed_scheduler_persists_under_the_cron_node() -> Result<()> {
    let dir = tempfile::tempdir().unwrap();
    let workspace = Workspace::new(dir.path());
    let scheduler = CronScheduler::for_workspace(&workspace);
    assert_eq!(scheduler.path(), dir.path().join("cron").join("jobs.json"));

    let tool = CronTool::new(Arc::clone(&scheduler), "en");
    let job = create_job(&tool, "@hourly", "tick", at(2024, 1, 1, 0, 30)).await;
    assert!(scheduler.path().exists());
    assert_eq!(
        CronScheduler::for_workspace(&workspace).list().await?,
        vec![job]
    );
    Ok(())
}

#[tokio::test]
async fn tool_invoke_through_the_trait_keeps_the_payload_shape() -> Result<()> {
    let dir = tempfile::tempdir().unwrap();
    let scheduler = CronScheduler::new(dir.path());
    let tool = CronTool::new(Arc::clone(&scheduler), "en");
    let output = tool
        .invoke(
            json!({"action": "create", "cron": "@daily", "prompt": "nightly"}),
            &ToolContext::new(),
        )
        .await?;

    let payload: ToolPayload = serde_json::from_value(output).unwrap();
    assert!(payload.success);
    let data = payload.data.unwrap();
    assert!(data.get("job").is_some());
    assert!(data.get("message").is_some());
    assert_eq!(scheduler.list().await?.len(), 1);
    Ok(())
}

#[tokio::test]
async fn disabled_jobs_are_listed_but_never_due() -> Result<()> {
    let dir = tempfile::tempdir().unwrap();
    let scheduler = CronScheduler::new(dir.path());
    let tool = CronTool::new(Arc::clone(&scheduler), "en");
    let job = create_job(&tool, "@daily", "nightly", at(2024, 1, 1, 0, 0)).await;

    let toggled = tool
        .invoke_at(
            &json!({"action": "toggle", "job_id": job.id}),
            at(2024, 1, 1, 0, 0),
        )
        .await;
    assert!(toggled.success);
    assert_eq!(toggled.data.unwrap()["job"]["enabled"], false);

    let listed = tool
        .invoke_at(&json!({"action": "list"}), at(2024, 1, 2, 0, 0))
        .await;
    assert_eq!(listed.data.unwrap()["jobs"].as_array().unwrap().len(), 1);
    assert_eq!(scheduler.due(at(2024, 1, 2, 0, 0)).await?, Vec::new());
    assert_eq!(scheduler.list().await?[0].id, job.id);
    Ok(())
}
