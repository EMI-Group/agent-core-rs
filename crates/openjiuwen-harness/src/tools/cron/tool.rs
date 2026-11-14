//! The model-facing `cron` tool.
//!
//! One tool with ten actions, mirroring the Python `cron` tool's single entry point (Python
//! spells the actions `add`/`list`/`remove`/`update`/`run`/`runs`/`status`/`wake` among others;
//! this port uses `create`/`list`/`cancel`, adds `get_job`/`update`/`toggle`/`preview`/`run`/
//! `runs`/`status`, and leaves `wake` deferred). The handlers live in [`super::actions`]; this
//! module owns the tool handle, the card and the dispatch.
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
//! # Still deferred
//!
//! Python's `wake` action, its `schedule.at` / `schedule.every` one-shot and interval forms,
//! its `payload` / `delivery` routing, `wake_offset_seconds`, the `contextMessages` history
//! window and the legacy per-action tool set (`cron_list_jobs`, `cron_get_job`, …). A job here
//! carries a flat prompt/description instead of a structured payload.

use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde_json::{json, Value};

use openjiuwen_common::error::Result;
use openjiuwen_foundation::tool::{Tool, ToolCard, ToolContext};

use crate::tools::descriptions;
use crate::tools::support::{
    compact_json, is_english, parse_args, schema_for, tool_card, ToolPayload,
};

use super::actions::{
    self, CronAction, CronArgs, ACTION_LIST, ACTION_NAMES, DEFAULT_PREVIEW_COUNT,
    DEFAULT_RUN_LIMIT, MAX_PREVIEW_COUNT, MAX_RUN_LIMIT,
};
use super::store::CronScheduler;

/// The model-facing tool name.
const TOOL_NAME: &str = "cron";

/// The failure reported when the tool was built without a scheduler.
const NO_SCHEDULER_MESSAGE: &str = "cron is not available: no cron scheduler is configured";

/// Creates, lists, inspects, mutates and runs persisted cron jobs (the `cron` tool).
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
    /// The description text comes from the shared [`descriptions`] table; unknown languages
    /// fall back to the Python default (`cn`). The per-property hints and the `action` enum are
    /// inlined here because they are action-shaped rather than a copy of a Python prompt
    /// constant, which keeps the card self-contained for the ten actions.
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
        let action_text = args.action.trim();
        if action_text.is_empty() {
            return ToolPayload::fail(format!("action is required (expected {ACTION_LIST})"));
        }
        let Some(action) = CronAction::parse(action_text) else {
            return ToolPayload::fail(format!(
                "invalid action: {:?} (expected {ACTION_LIST})",
                args.action
            ));
        };
        actions::dispatch(scheduler, action, &args, now).await
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

/// Replaces the generated arguments schema with the card shape the model sees.
///
/// [`CronArgs`] deliberately types `action` as a plain string (so the tool can report its own
/// validation text), so the card installs the action enum, the per-field hints, the integer
/// bounds and the `required` list itself. Every property is a primitive, so the schema has no
/// `$ref`.
fn install_input_params(schema: &mut Value, language: &str) {
    let Some(properties) = schema.get_mut("properties").and_then(Value::as_object_mut) else {
        return;
    };
    properties.insert(
        "action".to_owned(),
        json!({
            "type": "string",
            "enum": ACTION_NAMES,
            "description": hint("action", language),
        }),
    );
    for field in ["cron", "prompt", "description", "job_id"] {
        if let Some(property) = properties.get_mut(field) {
            property["description"] = Value::String(hint(field, language).to_owned());
        }
    }
    properties.insert(
        "count".to_owned(),
        json!({
            "type": "integer",
            "minimum": 1,
            "maximum": MAX_PREVIEW_COUNT,
            "default": DEFAULT_PREVIEW_COUNT,
            "description": hint("count", language),
        }),
    );
    properties.insert(
        "limit".to_owned(),
        json!({
            "type": "integer",
            "minimum": 1,
            "maximum": MAX_RUN_LIMIT,
            "default": DEFAULT_RUN_LIMIT,
            "description": hint("limit", language),
        }),
    );
    properties.insert(
        "enabled".to_owned(),
        json!({
            "type": "boolean",
            "description": hint("enabled", language),
        }),
    );
    if let Some(object) = schema.as_object_mut() {
        object.insert("required".to_owned(), json!(["action"]));
    }
}

/// The per-property hint of the `cron` card, in English or Chinese.
fn hint(field: &str, language: &str) -> &'static str {
    let english = is_english(language);
    match (field, english) {
        ("action", true) => {
            "Action to perform: create, list, cancel, get_job, update, toggle, preview, run, runs or status."
        }
        ("action", false) => {
            "要执行的操作：create、list、cancel、get_job、update、toggle、preview、run、runs 或 status。"
        }
        ("cron", true) => {
            "Cron expression: 5-field classic (minute hour day-of-month month day-of-week, for example `0 9 * * MON-FRI` or `@daily`) or 6/7-field Quartz (second minute hour day-of-month month day-of-week [year]); Quartz adds `?` in exactly one day field, `L` / `L-n` (last day), `nL` (last weekday), `d#n` (nth weekday), month and day names, and wrap-around ranges such as `FRI-MON` or `22-2`."
        }
        ("cron", false) => {
            "cron 表达式：5 段式经典格式（分 时 日 月 周，例如 `0 9 * * MON-FRI` 或 `@daily`），或 6/7 段式 Quartz 格式（秒 分 时 日 月 周 [年]）；Quartz 支持两个日期字段之一使用 `?`、`L` / `L-n`（月末）、`nL`（最后一个星期几）、`d#n`（第 n 个星期几）、月份与星期英文名，以及 `FRI-MON`、`22-2` 这类环绕区间。"
        }
        ("prompt", true) => "The prompt handed to the agent when the job fires.",
        ("prompt", false) => "任务触发时交给 agent 的提示词。",
        ("description", true) => {
            "Optional human description of the job; used as the prompt when `prompt` is absent."
        }
        ("description", false) => "可选的任务说明；未提供 `prompt` 时作为提示词使用。",
        ("job_id", true) => {
            "Identifier of the target job (required by every action except create, list, runs and status)."
        }
        ("job_id", false) => "目标任务 ID（除 create、list、runs、status 外均必填）。",
        ("count", true) => "How many upcoming occurrences `preview` reports (1-20, default 5).",
        ("count", false) => "`preview` 返回的后续触发时间点个数（1-20，默认 5）。",
        ("limit", true) => {
            "How many run-history entries `runs` returns, newest first (1-50, default 10)."
        }
        ("limit", false) => "`runs` 返回的运行记录条数，最新的在前（1-50，默认 10）。",
        ("enabled", true) => {
            "`toggle`: enable (true) or disable (false) the job instead of flipping it; `update`: the new enabled flag."
        }
        ("enabled", false) => {
            "`toggle`：直接启用（true）或停用（false）而不翻转；`update`：新的启用状态。"
        }
        _ => "",
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
#[path = "tool_tests.rs"]
mod tests;
