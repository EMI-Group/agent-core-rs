//! Unit tests of the model-facing `cron` tool: the card, the dispatch and every action.
//!
//! All scenarios run offline over a temporary directory with an injected clock.

use chrono::TimeZone;
use serde_json::json;

use super::super::CronJob;
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

/// Builds a tool over a fresh scheduler in a temporary directory.
fn tool(dir: &tempfile::TempDir) -> CronTool {
    CronTool::new(CronScheduler::new(dir.path()), "en")
}

/// Runs one call and unwraps the success data.
async fn ok_data(tool: &CronTool, inputs: Value, now: DateTime<Utc>) -> Value {
    let payload = tool.invoke_at(&inputs, now).await;
    assert!(payload.success, "unexpected failure: {:?}", payload.error);
    payload.data.unwrap()
}

/// Runs one call and unwraps the failure message.
async fn fail_message(tool: &CronTool, inputs: Value, now: DateTime<Utc>) -> String {
    let payload = tool.invoke_at(&inputs, now).await;
    assert!(!payload.success, "unexpected success: {:?}", payload.data);
    payload.error.unwrap()
}

/// Creates a job through the tool and returns it.
async fn create(tool: &CronTool, cron: &str, prompt: &str, now: DateTime<Utc>) -> CronJob {
    let data = ok_data(
        tool,
        json!({"action": "create", "cron": cron, "prompt": prompt}),
        now,
    )
    .await;
    serde_json::from_value(data["job"].clone()).unwrap()
}

#[tokio::test]
async fn create_list_cancel_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let tool = tool(&dir);
    let now = at(2024, 1, 1, 0, 0);

    let created = ok_data(
        &tool,
        json!({"action": "create", "cron": "*/15 * * * *", "prompt": "say hi"}),
        now,
    )
    .await;
    let job: CronJob = serde_json::from_value(created["job"].clone()).unwrap();
    assert_eq!(job.cron, "*/15 * * * *");
    assert_eq!(job.prompt, "say hi");
    assert_eq!(job.next_run_at, at(2024, 1, 1, 0, 15));
    let message = created["message"].as_str().unwrap();
    assert!(message.starts_with("Created cron job "), "{message}");
    assert!(message.contains(&job.id), "{message}");

    let listed = ok_data(&tool, json!({"action": "list"}), now).await;
    assert_eq!(listed["jobs"].as_array().unwrap().len(), 1);

    let cancelled = ok_data(&tool, json!({"action": "cancel", "job_id": job.id}), now).await;
    assert_eq!(cancelled["job_id"], job.id);
    assert_eq!(
        cancelled["message"],
        format!("Cancelled cron job {}", job.id)
    );

    let empty = ok_data(&tool, json!({"action": "list"}), now).await;
    assert_eq!(empty["jobs"], json!([]));
}

#[tokio::test]
async fn create_falls_back_to_the_description_as_prompt() {
    let dir = tempfile::tempdir().unwrap();
    let tool = tool(&dir);
    let created = ok_data(
        &tool,
        json!({"action": "create", "cron": "@daily", "description": "daily standup"}),
        at(2024, 1, 1, 0, 0),
    )
    .await;
    let job: CronJob = serde_json::from_value(created["job"].clone()).unwrap();
    assert_eq!(job.prompt, "daily standup");
    assert_eq!(job.description, "daily standup");
}

#[tokio::test]
async fn create_reports_an_invalid_expression() {
    let dir = tempfile::tempdir().unwrap();
    let tool = tool(&dir);
    let message = fail_message(
        &tool,
        json!({"action": "create", "cron": "75 * * * *", "prompt": "x"}),
        at(2024, 1, 1, 0, 0),
    )
    .await;
    assert_eq!(message, "invalid minute field: \"75\"");
    assert_eq!(tool.scheduler().unwrap().list().await.unwrap(), Vec::new());
}

#[tokio::test]
async fn create_accepts_a_quartz_expression() {
    let dir = tempfile::tempdir().unwrap();
    let tool = tool(&dir);
    let job = create(
        &tool,
        "0 0 9 ? * MON-FRI 2026",
        "standup",
        at(2024, 1, 1, 0, 0),
    )
    .await;
    assert_eq!(job.cron, "0 0 9 ? * MON-FRI 2026");
    assert_eq!(job.next_run_at, at(2026, 1, 1, 9, 0));
}

#[tokio::test]
async fn create_requires_cron_and_prompt() {
    let dir = tempfile::tempdir().unwrap();
    let tool = tool(&dir);
    let now = at(2024, 1, 1, 0, 0);
    assert_eq!(
        fail_message(&tool, json!({"action": "create", "prompt": "x"}), now).await,
        "cron is required for action=create"
    );
    assert_eq!(
        fail_message(&tool, json!({"action": "create", "cron": "@daily"}), now).await,
        "prompt is required for action=create"
    );
}

#[tokio::test]
async fn create_reports_a_schedule_that_never_matches() {
    let dir = tempfile::tempdir().unwrap();
    let tool = tool(&dir);
    let message = fail_message(
        &tool,
        json!({"action": "create", "cron": "0 0 31 2 *", "prompt": "x"}),
        at(2024, 1, 1, 0, 0),
    )
    .await;
    assert_eq!(
        message,
        "cron expression never matches a future time: \"0 0 31 2 *\""
    );
}

#[tokio::test]
async fn cancel_reports_missing_jobs_and_ids() {
    let dir = tempfile::tempdir().unwrap();
    let tool = tool(&dir);
    let now = at(2024, 1, 1, 0, 0);
    assert_eq!(
        fail_message(&tool, json!({"action": "cancel"}), now).await,
        "job_id is required for action=cancel"
    );
    assert_eq!(
        fail_message(&tool, json!({"action": "cancel", "job_id": "nope"}), now).await,
        "cron job not found: nope"
    );
}

#[tokio::test]
async fn reports_missing_and_unknown_actions() {
    let dir = tempfile::tempdir().unwrap();
    let tool = tool(&dir);
    let now = at(2024, 1, 1, 0, 0);
    assert_eq!(
        fail_message(&tool, json!({}), now).await,
        "action is required (expected create, list, cancel, get_job, update, toggle, preview, run, runs or status)"
    );
    assert_eq!(
        fail_message(&tool, json!({"action": "wake"}), now).await,
        "invalid action: \"wake\" (expected create, list, cancel, get_job, update, toggle, preview, run, runs or status)"
    );
}

#[tokio::test]
async fn get_job_reports_the_job_or_not_found() {
    let dir = tempfile::tempdir().unwrap();
    let tool = tool(&dir);
    let now = at(2024, 1, 1, 0, 0);
    let job = create(&tool, "@daily", "nightly", now).await;

    let data = ok_data(&tool, json!({"action": "get_job", "job_id": job.id}), now).await;
    let fetched: CronJob = serde_json::from_value(data["job"].clone()).unwrap();
    assert_eq!(fetched, job);
    assert_eq!(
        data["message"],
        format!(
            "Cron job {} (@daily) is enabled, next run at 2024-01-02T00:00:00Z",
            job.id
        )
    );

    assert_eq!(
        fail_message(&tool, json!({"action": "get_job"}), now).await,
        "job_id is required for action=get_job"
    );
    assert_eq!(
        fail_message(&tool, json!({"action": "get_job", "job_id": "nope"}), now).await,
        "cron job not found: nope"
    );
}

#[tokio::test]
async fn update_patches_the_stored_job() {
    let dir = tempfile::tempdir().unwrap();
    let tool = tool(&dir);
    let now = at(2024, 1, 1, 0, 0);
    let job = create(&tool, "*/15 * * * *", "say hi", now).await;

    let data = ok_data(
        &tool,
        json!({
            "action": "update",
            "job_id": job.id,
            "cron": "0 9 * * MON-FRI",
            "prompt": "standup",
            "description": "weekday standup",
        }),
        now,
    )
    .await;
    let updated: CronJob = serde_json::from_value(data["job"].clone()).unwrap();
    assert_eq!(updated.cron, "0 9 * * MON-FRI");
    assert_eq!(updated.prompt, "standup");
    assert_eq!(updated.description, "weekday standup");
    assert_eq!(updated.next_run_at, at(2024, 1, 1, 9, 0));
    assert!(data["message"]
        .as_str()
        .unwrap()
        .starts_with("Updated cron job "));

    // `enabled` is part of the same patch surface.
    let disabled = ok_data(
        &tool,
        json!({"action": "update", "job_id": job.id, "enabled": false}),
        now,
    )
    .await;
    let disabled: CronJob = serde_json::from_value(disabled["job"].clone()).unwrap();
    assert!(!disabled.enabled);

    assert_eq!(
        fail_message(&tool, json!({"action": "update", "job_id": job.id}), now).await,
        "nothing to update for action=update (provide cron, prompt, description or enabled)"
    );
    assert_eq!(
        fail_message(
            &tool,
            json!({"action": "update", "job_id": job.id, "cron": "75 * * * *"}),
            now
        )
        .await,
        "invalid minute field: \"75\""
    );
    assert_eq!(
        fail_message(
            &tool,
            json!({"action": "update", "job_id": job.id, "cron": "0 0 31 2 *"}),
            now
        )
        .await,
        "cron expression never matches a future time: \"0 0 31 2 *\""
    );
    assert_eq!(
        fail_message(
            &tool,
            json!({"action": "update", "job_id": "nope", "prompt": "x"}),
            now
        )
        .await,
        "cron job not found: nope"
    );
    assert_eq!(
        fail_message(&tool, json!({"action": "update", "prompt": "x"}), now).await,
        "job_id is required for action=update"
    );
}

#[tokio::test]
async fn toggle_flips_and_forces_the_enabled_flag() {
    let dir = tempfile::tempdir().unwrap();
    let tool = tool(&dir);
    let now = at(2024, 1, 1, 0, 0);
    let job = create(&tool, "@daily", "nightly", now).await;

    let off = ok_data(&tool, json!({"action": "toggle", "job_id": job.id}), now).await;
    let off_job: CronJob = serde_json::from_value(off["job"].clone()).unwrap();
    assert!(!off_job.enabled);
    assert!(off["message"]
        .as_str()
        .unwrap()
        .starts_with("Disabled cron job "));

    let forced_on = ok_data(
        &tool,
        json!({"action": "toggle", "job_id": job.id, "enabled": true}),
        at(2024, 1, 1, 3, 0),
    )
    .await;
    let forced_on: CronJob = serde_json::from_value(forced_on["job"].clone()).unwrap();
    assert!(forced_on.enabled);
    assert_eq!(
        forced_on.next_run_at,
        at(2024, 1, 2, 0, 0),
        "enabling recomputes the next occurrence from the clock"
    );

    let still_on = ok_data(&tool, json!({"action": "toggle", "job_id": job.id}), now).await;
    let still_on: CronJob = serde_json::from_value(still_on["job"].clone()).unwrap();
    assert!(!still_on.enabled, "a bare toggle flips the flag");

    assert_eq!(
        fail_message(&tool, json!({"action": "toggle", "job_id": "nope"}), now).await,
        "cron job not found: nope"
    );
}

#[tokio::test]
async fn preview_reports_occurrences_without_mutating_the_job() {
    let dir = tempfile::tempdir().unwrap();
    let tool = tool(&dir);
    let created = at(2024, 1, 1, 0, 0);
    let job = create(&tool, "*/15 * * * *", "say hi", created).await;

    let data = ok_data(
        &tool,
        json!({"action": "preview", "job_id": job.id, "count": 3}),
        at(2024, 1, 1, 0, 20),
    )
    .await;
    assert_eq!(
        data["occurrences"],
        json!([
            "2024-01-01T00:30:00Z",
            "2024-01-01T00:45:00Z",
            "2024-01-01T01:00:00Z"
        ])
    );
    assert_eq!(data["count"], 3);
    assert_eq!(data["after"], "2024-01-01T00:20:00Z");
    assert_eq!(
        tool.scheduler().unwrap().list().await.unwrap(),
        vec![job.clone()],
        "preview never rewrites the stored job"
    );

    // The default count is 5 and the argument is clamped to the engine limit.
    let default = ok_data(
        &tool,
        json!({"action": "preview", "job_id": job.id}),
        created,
    )
    .await;
    assert_eq!(default["occurrences"].as_array().unwrap().len(), 5);
    let clamped = ok_data(
        &tool,
        json!({"action": "preview", "job_id": job.id, "count": 500}),
        created,
    )
    .await;
    assert_eq!(
        clamped["occurrences"].as_array().unwrap().len(),
        MAX_PREVIEW_COUNT as usize
    );

    assert_eq!(
        fail_message(
            &tool,
            json!({"action": "preview", "job_id": "nope"}),
            created
        )
        .await,
        "cron job not found: nope"
    );
}

#[tokio::test]
async fn preview_uses_second_granularity_for_quartz() {
    let dir = tempfile::tempdir().unwrap();
    let tool = tool(&dir);
    let job = create(&tool, "0/20 * * * * ?", "tick", at(2024, 1, 1, 0, 0)).await;
    let data = ok_data(
        &tool,
        json!({"action": "preview", "job_id": job.id, "count": 2}),
        at_second(2024, 1, 1, 0, 0, 5),
    )
    .await;
    assert_eq!(
        data["occurrences"],
        json!(["2024-01-01T00:00:20Z", "2024-01-01T00:00:40Z"])
    );
}

#[tokio::test]
async fn run_records_a_manual_execution() {
    let dir = tempfile::tempdir().unwrap();
    let tool = tool(&dir);
    let created = at(2024, 1, 1, 0, 0);
    let job = create(&tool, "*/15 * * * *", "say hi", created).await;

    let data = ok_data(
        &tool,
        json!({"action": "run", "job_id": job.id}),
        at(2024, 1, 1, 0, 32),
    )
    .await;
    assert_eq!(data["run"]["job_id"], job.id);
    assert_eq!(data["run"]["trigger"], "manual");
    assert_eq!(data["run"]["occurrence"], "2024-01-01T00:32:00Z");
    assert_eq!(data["run"]["next_run_at"], "2024-01-01T00:45:00Z");
    assert_eq!(
        data["message"],
        format!(
            "Ran cron job {} manually; next run at 2024-01-01T00:45:00Z",
            job.id
        )
    );

    let stored = tool
        .scheduler()
        .unwrap()
        .get(&job.id)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(stored.next_run_at, at(2024, 1, 1, 0, 45));

    assert_eq!(
        fail_message(&tool, json!({"action": "run", "job_id": "nope"}), created).await,
        "cron job not found: nope"
    );
}

#[tokio::test]
async fn runs_reports_the_bounded_history() {
    let dir = tempfile::tempdir().unwrap();
    let tool = tool(&dir);
    let created = at(2024, 1, 1, 0, 0);
    let first = create(&tool, "*/15 * * * *", "first", created).await;
    let second = create(&tool, "*/15 * * * *", "second", created).await;

    ok_data(
        &tool,
        json!({"action": "run", "job_id": first.id}),
        at(2024, 1, 1, 0, 1),
    )
    .await;
    ok_data(
        &tool,
        json!({"action": "run", "job_id": second.id}),
        at(2024, 1, 1, 0, 2),
    )
    .await;
    ok_data(
        &tool,
        json!({"action": "run", "job_id": first.id}),
        at(2024, 1, 1, 0, 3),
    )
    .await;

    let all = ok_data(&tool, json!({"action": "runs"}), created).await;
    assert_eq!(all["count"], 3);
    assert_eq!(
        all["runs"][0]["recorded_at"], "2024-01-01T00:03:00Z",
        "newest first"
    );

    let filtered = ok_data(
        &tool,
        json!({"action": "runs", "job_id": first.id}),
        created,
    )
    .await;
    assert_eq!(filtered["count"], 2);
    assert!(filtered["runs"]
        .as_array()
        .unwrap()
        .iter()
        .all(|run| run["job_id"] == first.id.as_str()));

    let limited = ok_data(&tool, json!({"action": "runs", "limit": 1}), created).await;
    assert_eq!(limited["count"], 1);
    assert_eq!(limited["runs"][0]["job_id"], first.id);

    let default = ok_data(&tool, json!({"action": "runs"}), created).await;
    assert_eq!(default["count"], 3, "the default limit is 10");
}

#[tokio::test]
async fn status_summarizes_the_job_list() {
    let dir = tempfile::tempdir().unwrap();
    let tool = tool(&dir);
    let now = at(2024, 1, 1, 0, 0);
    let first = create(&tool, "*/15 * * * *", "first", now).await;
    create(&tool, "@daily", "second", now).await;
    ok_data(&tool, json!({"action": "toggle", "job_id": first.id}), now).await;
    ok_data(&tool, json!({"action": "run", "job_id": first.id}), now).await;

    let data = ok_data(&tool, json!({"action": "status"}), now).await;
    assert_eq!(data["status"]["total"], 2);
    assert_eq!(data["status"]["enabled"], 1);
    assert_eq!(data["status"]["disabled"], 1);
    assert_eq!(data["status"]["runs"], 1);
    assert_eq!(data["status"]["next_run_at"], "2024-01-02T00:00:00Z");
    assert_eq!(
        data["message"],
        "2 cron job(s): 1 enabled, 1 disabled, 0 due; 1 run record(s)"
    );
}

#[tokio::test]
async fn an_unconfigured_tool_reports_a_failure_payload() {
    let tool = CronTool::unconfigured("en");
    assert!(tool.scheduler().is_none());
    assert_eq!(tool.card().name, "cron");
    for action in ["list", "status", "runs"] {
        let payload = tool
            .invoke_at(&json!({"action": action}), at(2024, 1, 1, 0, 0))
            .await;
        assert!(!payload.success);
        assert_eq!(payload.error.as_deref(), Some(NO_SCHEDULER_MESSAGE));
    }
}

#[test]
fn card_carries_the_action_surface() {
    let card = CronTool::card_for("en");
    assert_eq!(card.name, "cron");
    // The card mutates the persisted job list, so it never runs in parallel.
    assert!(!card.parallel_safe);
    let properties = card.input_params["properties"].as_object().unwrap();
    assert_eq!(
        properties["action"]["enum"],
        json!([
            "create", "list", "cancel", "get_job", "update", "toggle", "preview", "run", "runs",
            "status"
        ])
    );
    assert_eq!(card.input_params["required"], json!(["action"]));
    assert!(!card.input_params.to_string().contains("$ref"));
    assert!(card.input_params.get("$defs").is_none());
    for field in ["cron", "prompt", "description", "job_id"] {
        assert_eq!(properties[field]["type"], "string", "{field}");
    }
    assert_eq!(properties["count"]["type"], "integer");
    assert_eq!(properties["count"]["maximum"], MAX_PREVIEW_COUNT);
    assert_eq!(properties["count"]["default"], DEFAULT_PREVIEW_COUNT);
    assert_eq!(properties["limit"]["type"], "integer");
    assert_eq!(properties["limit"]["maximum"], MAX_RUN_LIMIT);
    assert_eq!(properties["limit"]["default"], DEFAULT_RUN_LIMIT);
    assert_eq!(properties["enabled"]["type"], "boolean");
    assert_eq!(properties.len(), 8);
}

#[test]
fn card_hints_follow_the_language() {
    let english = CronTool::card_for("en");
    let chinese = CronTool::card_for("cn");
    assert_eq!(
        english.input_params["properties"]["action"]["description"],
        "Action to perform: create, list, cancel, get_job, update, toggle, preview, run, runs or status."
    );
    assert_eq!(
        chinese.input_params["properties"]["action"]["description"],
        "要执行的操作：create、list、cancel、get_job、update、toggle、preview、run、runs 或 status。"
    );
    assert!(english.input_params["properties"]["cron"]["description"]
        .as_str()
        .unwrap()
        .contains("Quartz"));
    assert!(chinese.input_params["properties"]["cron"]["description"]
        .as_str()
        .unwrap()
        .contains("Quartz"));
    assert_eq!(
        english.input_params["properties"]["count"]["description"],
        "How many upcoming occurrences `preview` reports (1-20, default 5)."
    );
    assert_eq!(
        english.input_params["properties"]["limit"]["description"],
        "How many run-history entries `runs` returns, newest first (1-50, default 10)."
    );
    // An unknown language falls back to the Chinese hints, like the description table.
    let unknown = CronTool::card_for("fr");
    for field in [
        "action",
        "cron",
        "prompt",
        "description",
        "job_id",
        "count",
        "limit",
        "enabled",
    ] {
        assert_eq!(
            unknown.input_params["properties"][field]["description"],
            chinese.input_params["properties"][field]["description"],
            "{field}"
        );
    }
}

#[test]
fn action_table_and_names_agree() {
    for (index, action) in CronAction::ALL.into_iter().enumerate() {
        assert_eq!(action.as_str(), ACTION_NAMES[index]);
        assert_eq!(CronAction::parse(action.as_str()), Some(action));
    }
    assert_eq!(CronAction::parse("nope"), None);
    assert_eq!(CronAction::parse("Create"), None, "actions are lower case");
    assert!(ACTION_LIST.ends_with("runs or status"));
    assert_eq!(
        MAX_PREVIEW_COUNT as usize,
        super::super::schedule::PREVIEW_LIMIT,
        "the model-facing clamp matches the engine limit"
    );
}

#[test]
fn the_card_description_stays_the_shared_table_text() {
    // The shared `descriptions` table owns the text; this module only adds property hints.
    for language in ["en", "cn", "fr"] {
        assert_eq!(
            CronTool::card_for(language).description,
            descriptions::description(TOOL_NAME, language),
            "{language}"
        );
    }
}

#[tokio::test]
async fn invoke_uses_the_wall_clock() {
    let dir = tempfile::tempdir().unwrap();
    let tool = tool(&dir);
    let output = tool
        .invoke(
            json!({"action": "create", "cron": "@daily", "prompt": "nightly"}),
            &ToolContext::new(),
        )
        .await
        .unwrap();
    assert_eq!(output["success"], true);
    let job: CronJob = serde_json::from_value(output["data"]["job"].clone()).unwrap();
    assert!(job.created_at <= Utc::now());
    assert!(job.next_run_at > Utc::now());
}

#[tokio::test]
async fn render_for_llm_prefers_the_message() {
    let dir = tempfile::tempdir().unwrap();
    let tool = tool(&dir);
    let now = at(2024, 1, 1, 0, 0);

    let created = tool
        .invoke_at(
            &json!({"action": "create", "cron": "@daily", "prompt": "nightly"}),
            now,
        )
        .await;
    let job: CronJob =
        serde_json::from_value(created.data.clone().unwrap()["job"].clone()).unwrap();
    let rendered = tool.render_for_llm(&created.to_json());
    assert!(rendered.starts_with("Created cron job "), "{rendered}");

    let fetched = tool
        .invoke_at(&json!({"action": "get_job", "job_id": job.id}), now)
        .await;
    let rendered = tool.render_for_llm(&fetched.to_json());
    assert!(rendered.starts_with("Cron job "), "{rendered}");

    let listed = tool.invoke_at(&json!({"action": "list"}), now).await;
    let rendered = tool.render_for_llm(&listed.to_json());
    assert!(rendered.contains("\"cron\":\"@daily\""), "{rendered}");

    let runs = tool.invoke_at(&json!({"action": "runs"}), now).await;
    let rendered = tool.render_for_llm(&runs.to_json());
    assert!(rendered.contains("\"runs\":[]"), "{rendered}");

    let failed = tool
        .invoke_at(&json!({"action": "cancel", "job_id": "nope"}), now)
        .await;
    assert_eq!(
        tool.render_for_llm(&failed.to_json()),
        "cron job not found: nope"
    );
}
