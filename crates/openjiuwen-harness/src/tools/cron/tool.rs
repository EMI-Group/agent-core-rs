//! The model-facing `cron` tool.
//!
//! One tool with three actions, mirroring the Python `cron` tool's single entry point (Python
//! spells the actions `add`/`list`/`remove` among others; this port uses `create`/`list`/
//! `cancel`):
//!
//! - `create` — `cron` expression plus `prompt` (and optionally `description`) → the stored
//!   job,
//! - `list` — every stored job, enabled or not,
//! - `cancel` — `job_id` → the removed job id.
//!
//! The tool never returns [`Err`] for an expected failure: a missing scheduler, a rejected
//! expression, a validation error, a storage error or an unknown job id all become a
//! `success: false` payload. Only cancellation (through `ToolContext::check_cancelled`)
//! produces an [`Err`](openjiuwen_common::error::Error).
//!
//! # Clock
//!
//! [`Tool::invoke`] reads the wall clock exactly once, at the call site, and hands the instant
//! to [`CronTool::invoke_at`]. Tests and hosts that own a clock call `invoke_at` directly; the
//! scheduler itself never reads the clock.
//!
//! # Deferred Python actions
//!
//! Python's tool additionally exposes `status`, `update`, `run`, `runs`, `wake`,
//! `get_job`, `toggle_job` and `preview_job`, and its job payload accepts `schedule.at` /
//! `schedule.every` one-shot and interval forms plus `payload` / `delivery` routing. None of
//! that is ported here: only the three actions above and the 5-field cron expression exist,
//! and a job carries a flat prompt/description instead of a structured payload.

use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use openjiuwen_common::error::Result;
use openjiuwen_foundation::tool::{Tool, ToolCard, ToolContext};

use crate::tools::descriptions;
use crate::tools::support::{
    compact_json, is_english, parse_args, schema_for, tool_card, ToolPayload,
};

use super::schedule::CronSchedule;
use super::store::CronScheduler;

/// The model-facing tool name.
const TOOL_NAME: &str = "cron";

/// The failure reported when the tool was built without a scheduler.
const NO_SCHEDULER_MESSAGE: &str = "cron is not available: no cron scheduler is configured";

/// The action of one `cron` call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CronAction {
    /// Create and persist a new job.
    Create,
    /// List the stored jobs.
    List,
    /// Remove a stored job.
    Cancel,
}

impl CronAction {
    /// The action name as the model writes it.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Create => "create",
            Self::List => "list",
            Self::Cancel => "cancel",
        }
    }

    /// Parses an action name, accepting surrounding whitespace only.
    #[must_use]
    pub fn parse(action: &str) -> Option<Self> {
        match action {
            "create" => Some(Self::Create),
            "list" => Some(Self::List),
            "cancel" => Some(Self::Cancel),
            _ => None,
        }
    }
}

/// Arguments of the `cron` tool.
///
/// One flat object for all three actions: `cron`/`prompt`/`description` belong to `create` and
/// `job_id` belongs to `cancel`. Every field is optional at the serde level so the tool can
/// report its own validation text instead of a deserialization error; `action` is the only
/// required field in the card schema.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct CronArgs {
    /// The action to perform: `create`, `list` or `cancel`.
    #[serde(default)]
    pub action: String,
    /// A 5-field cron expression (`minute hour day-of-month month day-of-week`); `create` only.
    #[serde(default)]
    pub cron: String,
    /// The prompt handed to the agent when the job fires; `create` only.
    #[serde(default)]
    pub prompt: String,
    /// A human description of the job; `create` only, and used as the prompt when `prompt` is
    /// absent.
    #[serde(default)]
    pub description: String,
    /// The identifier of the job to remove; `cancel` only.
    #[serde(default)]
    pub job_id: String,
}

/// Creates, lists and cancels persisted cron jobs (the `cron` tool).
pub struct CronTool {
    /// The scheduler holding the jobs; without one every call reports a failure payload.
    scheduler: Option<Arc<CronScheduler>>,
    /// The model-facing card built from the description language.
    card: ToolCard,
}

impl CronTool {
    /// Builds the tool over `scheduler` with the card description language `language`.
    #[must_use]
    pub fn new(scheduler: Arc<CronScheduler>, language: impl Into<String>) -> Self {
        let language = language.into();
        Self {
            card: Self::card_for(&language),
            scheduler: Some(scheduler),
        }
    }

    /// Builds a tool with no scheduler wired, so every action reports a failure payload.
    ///
    /// This is the seam for hosts that disable scheduling (a read-only deployment, a test that
    /// only checks the card): the card stays identical, only the execution differs.
    #[must_use]
    pub fn unconfigured(language: impl Into<String>) -> Self {
        let language = language.into();
        Self {
            card: Self::card_for(&language),
            scheduler: None,
        }
    }

    /// Builds the model-facing card in `language`.
    ///
    /// The description text comes from the shared
    /// [`descriptions`] table; unknown languages fall back to the Python default (`cn`). The
    /// per-property hints are inlined here because they are action-shaped rather than a
    /// copy of a Python prompt constant.
    #[must_use]
    pub fn card_for(language: &str) -> ToolCard {
        let mut input_params = schema_for::<CronArgs>();
        install_input_params(&mut input_params, language);
        tool_card(
            TOOL_NAME,
            descriptions::description(TOOL_NAME, language),
            input_params,
            false,
        )
    }

    /// The scheduler backing the tool, when one is wired.
    #[must_use]
    pub fn scheduler(&self) -> Option<&Arc<CronScheduler>> {
        self.scheduler.as_ref()
    }

    /// Runs one action against the injected clock value `now`.
    ///
    /// This is the deterministic entry point: [`Tool::invoke`] passes the wall clock in, tests
    /// pass a fixed instant, and nothing below this call reads the clock.
    pub async fn invoke_at(&self, inputs: &Value, now: DateTime<Utc>) -> ToolPayload {
        let Some(scheduler) = self.scheduler.as_deref() else {
            return ToolPayload::fail(NO_SCHEDULER_MESSAGE);
        };
        let args: CronArgs = match parse_args(inputs) {
            Ok(args) => args,
            Err(payload) => return payload,
        };
        if args.action.trim().is_empty() {
            return ToolPayload::fail("action is required (expected create, list or cancel)");
        }
        match CronAction::parse(args.action.trim()) {
            Some(CronAction::Create) => create(scheduler, &args, now).await,
            Some(CronAction::List) => list(scheduler).await,
            Some(CronAction::Cancel) => cancel(scheduler, &args).await,
            None => ToolPayload::fail(format!(
                "invalid action: {:?} (expected create, list or cancel)",
                args.action
            )),
        }
    }
}

#[async_trait]
impl Tool for CronTool {
    fn card(&self) -> &ToolCard {
        &self.card
    }

    async fn invoke(&self, inputs: Value, ctx: &ToolContext) -> Result<Value> {
        ctx.check_cancelled()?;
        Ok(self.invoke_at(&inputs, Utc::now()).await.to_json())
    }

    /// Renders the payload as text: the success `message` when there is one, the compact JSON
    /// data otherwise, and the error text on failure.
    fn render_for_llm(&self, output: &Value) -> String {
        let Ok(payload) = serde_json::from_value::<ToolPayload>(output.clone()) else {
            return compact_json(output);
        };
        if !payload.success {
            return payload.error.unwrap_or_default();
        }
        match payload.data {
            Some(data) => data
                .get("message")
                .and_then(Value::as_str)
                .map_or_else(|| compact_json(&data), str::to_owned),
            None => String::new(),
        }
    }
}

/// Handles `action=create`.
async fn create(scheduler: &CronScheduler, args: &CronArgs, now: DateTime<Utc>) -> ToolPayload {
    if args.cron.trim().is_empty() {
        return ToolPayload::fail("cron is required for action=create");
    }
    let prompt = if args.prompt.trim().is_empty() {
        args.description.trim()
    } else {
        args.prompt.trim()
    };
    if prompt.is_empty() {
        return ToolPayload::fail("prompt is required for action=create");
    }
    let schedule = match CronSchedule::parse(&args.cron) {
        Ok(schedule) => schedule,
        Err(error) => return ToolPayload::fail(error.message().into_owned()),
    };
    match scheduler
        .create_with_description(&schedule, prompt, args.description.trim(), now)
        .await
    {
        Ok(job) => {
            let message = format!(
                "Created cron job {} ({}) next run at {}",
                job.id,
                job.cron,
                job.next_run_at.to_rfc3339()
            );
            ToolPayload::ok(json!({"job": job, "message": message}))
        }
        Err(error) => ToolPayload::fail(error.message().into_owned()),
    }
}

/// Handles `action=list`.
async fn list(scheduler: &CronScheduler) -> ToolPayload {
    match scheduler.list().await {
        Ok(jobs) => ToolPayload::ok(json!({"jobs": jobs})),
        Err(error) => ToolPayload::fail(error.message().into_owned()),
    }
}

/// Handles `action=cancel`.
async fn cancel(scheduler: &CronScheduler, args: &CronArgs) -> ToolPayload {
    let job_id = args.job_id.trim();
    if job_id.is_empty() {
        return ToolPayload::fail("job_id is required for action=cancel");
    }
    match scheduler.cancel(job_id).await {
        Ok(Some(job)) => ToolPayload::ok(json!({
            "job_id": job.id,
            "message": format!("Cancelled cron job {}", job.id),
        })),
        Ok(None) => ToolPayload::fail(format!("cron job not found: {job_id}")),
        Err(error) => ToolPayload::fail(error.message().into_owned()),
    }
}

/// Replaces the generated arguments schema with the card shape the model sees.
///
/// [`CronArgs`] deliberately types `action` as a plain string (so the tool can report its own
/// validation text), so the card installs the three-member enum, the per-field hints and the
/// `required` list itself. Every property is a primitive, so the schema has no `$ref`.
fn install_input_params(schema: &mut Value, language: &str) {
    let Some(properties) = schema.get_mut("properties").and_then(Value::as_object_mut) else {
        return;
    };
    properties.insert(
        "action".to_owned(),
        json!({
            "type": "string",
            "enum": ["create", "list", "cancel"],
            "description": hint("action", language),
        }),
    );
    for field in ["cron", "prompt", "description", "job_id"] {
        if let Some(property) = properties.get_mut(field) {
            property["description"] = Value::String(hint(field, language).to_owned());
        }
    }
    if let Some(object) = schema.as_object_mut() {
        object.insert("required".to_owned(), json!(["action"]));
    }
}

/// The per-property hint of the `cron` card, in English or Chinese.
fn hint(field: &str, language: &str) -> &'static str {
    let english = is_english(language);
    match (field, english) {
        ("action", true) => "Action to perform: create, list or cancel.",
        ("action", false) => "要执行的操作：create、list 或 cancel。",
        ("cron", true) => {
            "5-field cron expression: minute hour day-of-month month day-of-week, for example `0 9 * * MON-FRI` or `@daily`."
        }
        ("cron", false) => {
            "5 段式 cron 表达式：分 时 日 月 周，例如 `0 9 * * MON-FRI` 或 `@daily`。"
        }
        ("prompt", true) => "The prompt handed to the agent when the job fires.",
        ("prompt", false) => "任务触发时交给 agent 的提示词。",
        ("description", true) => {
            "Optional human description of the job; used as the prompt when `prompt` is absent."
        }
        ("description", false) => "可选的任务说明；未提供 `prompt` 时作为提示词使用。",
        ("job_id", true) => "Identifier of the job to cancel.",
        ("job_id", false) => "要取消的任务 ID。",
        _ => "",
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use chrono::TimeZone;
    use serde_json::json;

    use super::super::CronJob;
    use super::*;

    /// Builds a UTC instant at the given wall-clock minute.
    fn at(year: i32, month: u32, day: u32, hour: u32, minute: u32) -> DateTime<Utc> {
        Utc.with_ymd_and_hms(year, month, day, hour, minute, 0)
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
            "action is required (expected create, list or cancel)"
        );
        assert_eq!(
            fail_message(&tool, json!({"action": "run"}), now).await,
            "invalid action: \"run\" (expected create, list or cancel)"
        );
    }

    #[tokio::test]
    async fn non_object_arguments_are_a_protocol_failure() {
        let dir = tempfile::tempdir().unwrap();
        let tool = tool(&dir);
        let payload = tool.invoke_at(&json!([]), at(2024, 1, 1, 0, 0)).await;
        assert!(!payload.success);
        assert_eq!(
            payload.error.as_deref(),
            Some("tool arguments must be an object")
        );
    }

    #[tokio::test]
    async fn an_unconfigured_tool_reports_a_failure_payload() {
        let tool = CronTool::unconfigured("en");
        assert!(tool.scheduler().is_none());
        assert_eq!(tool.card().name, "cron");
        let payload = tool
            .invoke_at(&json!({"action": "list"}), at(2024, 1, 1, 0, 0))
            .await;
        assert!(!payload.success);
        assert_eq!(payload.error.as_deref(), Some(NO_SCHEDULER_MESSAGE));
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
            json!(["create", "list", "cancel"])
        );
        assert_eq!(card.input_params["required"], json!(["action"]));
        assert!(!card.input_params.to_string().contains("$ref"));
        assert!(card.input_params.get("$defs").is_none());
        for field in ["cron", "prompt", "description", "job_id"] {
            assert_eq!(properties[field]["type"], "string", "{field}");
        }
    }

    #[test]
    fn card_hints_follow_the_language() {
        let english = CronTool::card_for("en");
        let chinese = CronTool::card_for("cn");
        assert_eq!(
            english.input_params["properties"]["action"]["description"],
            "Action to perform: create, list or cancel."
        );
        assert_eq!(
            chinese.input_params["properties"]["action"]["description"],
            "要执行的操作：create、list 或 cancel。"
        );
        // An unknown language falls back to the Chinese hints, like the description table.
        let unknown = CronTool::card_for("fr");
        assert_eq!(
            unknown.input_params["properties"]["job_id"]["description"],
            chinese.input_params["properties"]["job_id"]["description"]
        );
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
        let rendered = tool.render_for_llm(&created.to_json());
        assert!(rendered.starts_with("Created cron job "), "{rendered}");

        let listed = tool.invoke_at(&json!({"action": "list"}), now).await;
        let rendered = tool.render_for_llm(&listed.to_json());
        assert!(rendered.contains("\"cron\":\"@daily\""), "{rendered}");

        let failed = tool
            .invoke_at(&json!({"action": "cancel", "job_id": "nope"}), now)
            .await;
        assert_eq!(
            tool.render_for_llm(&failed.to_json()),
            "cron job not found: nope"
        );
    }
}
