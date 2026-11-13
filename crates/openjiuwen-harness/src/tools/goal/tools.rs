//! The two model-facing goal tools: `get_current_goal` and `submit_goal_report`.
//!
//! Ports the model-facing half of `openjiuwen/harness/tools/goal.py`. Python's
//! `SubmitGoalReportTool` writes to a per-round `GoalReportSink` that the `TaskCompletionRail`
//! consumes after the round; here the tool writes the assessment straight into the
//! [`GoalStore`] and settles the record, so the model's report is durable even when no rail is
//! mounted. `GetCurrentGoalTool` stays a pure read.
//!
//! Both tools return the module's six-key [`ToolPayload`] envelope. Expected failures — no
//! store wired, an unreadable record — are `success=false` payloads, never `Err`.
use std::sync::Arc;

use async_trait::async_trait;
use chrono::Utc;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use openjiuwen_common::error::Result;
use openjiuwen_foundation::tool::{Tool, ToolCard, ToolContext};

use super::schema::{
    deserialize_lenient_opt_string, deserialize_lenient_string, GoalAssessment,
    GoalAssessmentStatus,
};
use super::store::GoalStore;
use crate::tools::descriptions::description;
use crate::tools::is_english;
use crate::tools::support::{parse_args, render_fields, schema_for, tool_card, ToolPayload};

/// Failure reported when a goal tool is built without a store.
const NO_STORE_MESSAGE: &str = "goal store is not configured";

/// Message of `get_current_goal` when no record exists (Python `en` text).
const NO_GOAL_MESSAGE_EN: &str = "No persistent goal is set for this session.";

/// Message of `get_current_goal` when no record exists (Python `cn` text).
const NO_GOAL_MESSAGE_CN: &str = "当前会话没有设置持续目标。";

/// The Python no-goal message for `language`; anything but `"en"` selects Chinese.
fn no_goal_message(language: &str) -> &'static str {
    if is_english(language) {
        NO_GOAL_MESSAGE_EN
    } else {
        NO_GOAL_MESSAGE_CN
    }
}

/// Arguments of `get_current_goal`: an empty object.
///
/// Python's card declares `{"type": "object", "properties": {}, "required": []}`, and the tool
/// ignores whatever the model sends, so nothing is parsed here.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct GetCurrentGoalArgs {}

/// Arguments of `submit_goal_report`.
///
/// Python parity: `status`, `evidence`, `remaining_work` and `next_instruction`. Every field is
/// optional on the wire (Python defaults `status` to `continue` and `evidence` to `""`), and
/// every field tolerates `null` or a non-string value instead of failing the whole call.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct SubmitGoalReportArgs {
    /// Outcome of the current goal attempt: `continue`, `complete` or `blocked`.
    ///
    /// Use `continue` while the objective can still be advanced, `complete` only with auditable
    /// completion evidence, and `blocked` only when nothing can make progress.
    #[serde(default, deserialize_with = "deserialize_lenient_string")]
    pub status: String,
    /// Evidence for the reported status, as concrete as possible.
    #[serde(default, deserialize_with = "deserialize_lenient_string")]
    pub evidence: String,
    /// What is still missing to reach the objective.
    #[serde(default, deserialize_with = "deserialize_lenient_opt_string")]
    pub remaining_work: Option<String>,
    /// What the next attempt should do.
    #[serde(default, deserialize_with = "deserialize_lenient_opt_string")]
    pub next_instruction: Option<String>,
}

/// Reads the current session goal so the model can re-orient (`get_current_goal`).
///
/// Python parity: `GetCurrentGoalTool`. The main model loses the goal context on non-goal turns,
/// so this read-only tool recovers the objective, status, attempt count and last assessment on
/// demand.
pub struct GetCurrentGoalTool {
    /// The store read by every call; `None` reports the "not configured" failure.
    store: Option<Arc<GoalStore>>,
    /// Description language, used for the no-goal message.
    language: String,
    /// The model-facing card, built once from the language.
    card: ToolCard,
}

impl GetCurrentGoalTool {
    /// Builds the tool over `store` with descriptions in `language`.
    #[must_use]
    pub fn new(store: Arc<GoalStore>, language: impl Into<String>) -> Self {
        let language = language.into();
        Self {
            card: Self::card_for(&language),
            store: Some(store),
            language,
        }
    }

    /// Builds the tool without a store; every call reports `goal store is not configured`.
    #[must_use]
    pub fn unavailable(language: impl Into<String>) -> Self {
        let language = language.into();
        Self {
            card: Self::card_for(&language),
            store: None,
            language,
        }
    }

    /// Builds the model-facing card in `language`; unknown languages fall back to `cn` inside
    /// [`description`].
    #[must_use]
    pub fn card_for(language: &str) -> ToolCard {
        tool_card(
            "get_current_goal",
            description("get_current_goal", language),
            schema_for::<GetCurrentGoalArgs>(),
            true,
        )
    }

    /// The store this tool reads, when one is wired.
    #[must_use]
    pub fn store(&self) -> Option<&Arc<GoalStore>> {
        self.store.as_ref()
    }

    /// The description language of this tool.
    #[must_use]
    pub fn language(&self) -> &str {
        &self.language
    }

    /// Builds the payload of one call.
    async fn snapshot(&self) -> ToolPayload {
        let Some(store) = self.store.as_ref() else {
            return ToolPayload::fail(NO_STORE_MESSAGE);
        };
        match store.load().await {
            Ok(Some(record)) => ToolPayload::ok(json!({ "goal": record.to_dict() })),
            Ok(None) => ToolPayload::ok(json!({
                "goal": Value::Null,
                "message": no_goal_message(&self.language),
            })),
            // Python logs the failure and reports "no goal"; returning the problem keeps a
            // corrupt `goal.json` visible instead of pretending the session has no goal.
            Err(error) => ToolPayload::fail(error.message().into_owned()),
        }
    }
}

#[async_trait]
impl Tool for GetCurrentGoalTool {
    fn card(&self) -> &ToolCard {
        &self.card
    }

    async fn invoke(&self, _inputs: Value, ctx: &ToolContext) -> Result<Value> {
        ctx.check_cancelled()?;
        Ok(self.snapshot().await.to_json())
    }

    fn render_for_llm(&self, output: &Value) -> String {
        let data = output.get("data");
        if let Some(goal) = data
            .and_then(|data| data.get("goal"))
            .filter(|goal| goal.is_object())
        {
            return render_fields(&[
                (
                    "goal_id",
                    goal.get("goal_id").cloned().unwrap_or(Value::Null),
                ),
                (
                    "objective",
                    goal.get("objective").cloned().unwrap_or(Value::Null),
                ),
                ("status", goal.get("status").cloned().unwrap_or(Value::Null)),
                (
                    "attempt_count",
                    goal.get("attempt_count").cloned().unwrap_or(Value::Null),
                ),
                (
                    "last_assessment",
                    goal.get("last_assessment").cloned().unwrap_or(Value::Null),
                ),
            ]);
        }
        if let Some(message) = data
            .and_then(|data| data.get("message"))
            .and_then(Value::as_str)
        {
            if !message.is_empty() {
                return message.to_owned();
            }
        }
        output
            .get("error")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned()
    }
}

/// Submits the structured result of the current goal attempt (`submit_goal_report`).
///
/// Python parity: `SubmitGoalReportTool` argument names, payload shape and status normalization.
/// The assessment is committed to the [`GoalStore`] record, and a settling status
/// (`complete`/`blocked`) moves the record to `completed`/`blocked`.
pub struct SubmitGoalReportTool {
    /// The store the assessment is committed to; `None` reports the "not configured" failure.
    store: Option<Arc<GoalStore>>,
    /// Description language, used for the no-goal message.
    language: String,
    /// The model-facing card, built once from the language.
    card: ToolCard,
}

impl SubmitGoalReportTool {
    /// Builds the tool over `store` with descriptions in `language`.
    #[must_use]
    pub fn new(store: Arc<GoalStore>, language: impl Into<String>) -> Self {
        let language = language.into();
        Self {
            card: Self::card_for(&language),
            store: Some(store),
            language,
        }
    }

    /// Builds the tool without a store; every call reports `goal store is not configured`.
    #[must_use]
    pub fn unavailable(language: impl Into<String>) -> Self {
        let language = language.into();
        Self {
            card: Self::card_for(&language),
            store: None,
            language,
        }
    }

    /// Builds the model-facing card in `language`; unknown languages fall back to `cn` inside
    /// [`description`].
    #[must_use]
    pub fn card_for(language: &str) -> ToolCard {
        let mut input_params = schema_for::<SubmitGoalReportArgs>();
        install_submit_goal_report_shape(&mut input_params);
        tool_card(
            "submit_goal_report",
            description("submit_goal_report", language),
            input_params,
            false,
        )
    }

    /// The store this tool writes, when one is wired.
    #[must_use]
    pub fn store(&self) -> Option<&Arc<GoalStore>> {
        self.store.as_ref()
    }

    /// The description language of this tool.
    #[must_use]
    pub fn language(&self) -> &str {
        &self.language
    }

    /// Validates `inputs` and commits the assessment.
    async fn report(&self, inputs: &Value) -> ToolPayload {
        let args: SubmitGoalReportArgs = match parse_args(inputs) {
            Ok(args) => args,
            Err(payload) => return payload,
        };
        // Python normalizes an unknown or missing status to `continue`.
        let status = GoalAssessmentStatus::parse_lenient(&args.status);
        let assessment = GoalAssessment::new(
            status,
            args.evidence,
            args.remaining_work,
            args.next_instruction,
        );

        let Some(store) = self.store.as_ref() else {
            return ToolPayload::fail(NO_STORE_MESSAGE);
        };
        let now = Utc::now();
        let updated = store
            .update(|record| record.record_progress(&assessment, now), now)
            .await;
        match updated {
            Ok(Some(record)) => ToolPayload::ok(json!({
                "result": "report_accepted",
                "status": status.as_str(),
                "goal_id": record.goal_id,
                "goal_status": record.status.as_str(),
            })),
            // No goal is recorded: the report is still accepted (Python's sink does not check),
            // and the envelope says so instead of creating a goal behind the host's back.
            Ok(None) => ToolPayload::ok(json!({
                "result": "report_accepted",
                "status": status.as_str(),
                "message": no_goal_message(&self.language),
            })),
            Err(error) => ToolPayload::fail(error.message().into_owned()),
        }
    }
}

#[async_trait]
impl Tool for SubmitGoalReportTool {
    fn card(&self) -> &ToolCard {
        &self.card
    }

    async fn invoke(&self, inputs: Value, ctx: &ToolContext) -> Result<Value> {
        ctx.check_cancelled()?;
        Ok(self.report(&inputs).await.to_json())
    }

    fn render_for_llm(&self, output: &Value) -> String {
        let status = output
            .get("data")
            .and_then(|data| data.get("status"))
            .and_then(Value::as_str)
            .unwrap_or_else(|| GoalAssessmentStatus::Continue.as_str());
        format!("Goal report accepted (status: {status}).")
    }
}

/// Rewrites the generated [`SubmitGoalReportArgs`] schema into the Python card shape.
///
/// Two things differ: the `status` property carries Python's `continue|complete|blocked` enum
/// instead of the bare string, and the two optional properties are plain strings (schemars would
/// describe `Option<String>` as nullable). `required` is set to Python's `["status", "evidence"]`
/// even though serde tolerates both missing, because the model should always send them.
fn install_submit_goal_report_shape(schema: &mut Value) {
    let Some(object) = schema.as_object_mut() else {
        return;
    };
    if let Some(properties) = object.get_mut("properties").and_then(Value::as_object_mut) {
        if let Some(status) = properties.get_mut("status") {
            make_string_property(status);
            if let Some(status) = status.as_object_mut() {
                status.insert(
                    "enum".to_owned(),
                    json!(GoalAssessmentStatus::ALL.map(GoalAssessmentStatus::as_str)),
                );
            }
        }
        for field in ["evidence", "remaining_work", "next_instruction"] {
            if let Some(property) = properties.get_mut(field) {
                make_string_property(property);
            }
        }
    }
    object.insert("required".to_owned(), json!(["status", "evidence"]));
}

/// Reduces a property schema to a plain `{"type": "string"}` object, keeping its description.
fn make_string_property(property: &mut Value) {
    let Some(object) = property.as_object_mut() else {
        return;
    };
    let description = object.get("description").cloned();
    object.clear();
    object.insert("type".to_owned(), json!("string"));
    if let Some(description) = description {
        object.insert("description".to_owned(), description);
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use chrono::{TimeZone, Utc};
    use serde_json::json;

    use super::*;
    use crate::tools::goal::schema::{GoalStatus, DEFAULT_SESSION_ID};

    /// A fixed clock.
    fn now() -> chrono::DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 1, 2, 3, 4, 5).unwrap()
    }

    /// A store inside a fresh temporary directory.
    fn store(dir: &tempfile::TempDir) -> Arc<GoalStore> {
        GoalStore::for_session(dir.path().join("goal").join("goal.json"), "s-1")
    }

    /// A tool context for the tests.
    fn context() -> ToolContext {
        ToolContext::new().with_session("s-1")
    }

    #[test]
    fn cards_carry_the_python_names_and_schemas() {
        let card = GetCurrentGoalTool::card_for("en");
        assert_eq!(card.name, "get_current_goal");
        assert!(card.parallel_safe);
        assert_eq!(card.input_params["type"], json!("object"));
        assert_eq!(card.input_params["properties"], json!({}));
        assert_eq!(card.input_params.get("required"), None);

        let card = SubmitGoalReportTool::card_for("en");
        assert_eq!(card.name, "submit_goal_report");
        // The card mutates the persisted goal record, so it never runs in parallel.
        assert!(!card.parallel_safe);
        assert_eq!(
            card.input_params["properties"]["status"]["enum"],
            json!(["continue", "complete", "blocked"])
        );
        assert_eq!(
            card.input_params["properties"]["status"]["type"],
            json!("string")
        );
        for field in ["evidence", "remaining_work", "next_instruction"] {
            assert_eq!(
                card.input_params["properties"][field]["type"],
                json!("string"),
                "unexpected {field} schema: {}",
                card.input_params["properties"][field]
            );
        }
        assert_eq!(card.input_params["required"], json!(["status", "evidence"]));
        assert!(card.input_params["properties"]["status"]["description"].is_string());
        // Both languages build; the description text is owned by the shared table.
        assert_eq!(GetCurrentGoalTool::card_for("cn").name, "get_current_goal");
        assert_eq!(
            SubmitGoalReportTool::card_for("cn").input_params["required"],
            json!(["status", "evidence"])
        );
    }

    #[tokio::test]
    async fn get_current_goal_without_a_store_fails_softly() {
        let tool = GetCurrentGoalTool::unavailable("en");
        assert!(tool.store().is_none());
        assert_eq!(tool.language(), "en");
        let output = tool.invoke(json!({}), &context()).await.unwrap();
        assert_eq!(output["success"], json!(false));
        assert_eq!(output["error"], json!(NO_STORE_MESSAGE));
        assert_eq!(tool.render_for_llm(&output), NO_STORE_MESSAGE);
    }

    #[tokio::test]
    async fn get_current_goal_without_a_record_reports_null() {
        let dir = tempfile::tempdir().unwrap();
        let tool = GetCurrentGoalTool::new(store(&dir), "en");
        assert!(tool.store().is_some());
        let output = tool.invoke(json!({}), &context()).await.unwrap();
        assert_eq!(output["success"], json!(true));
        assert_eq!(output["data"]["goal"], Value::Null);
        assert_eq!(output["data"]["message"], json!(NO_GOAL_MESSAGE_EN));
        assert_eq!(tool.render_for_llm(&output), NO_GOAL_MESSAGE_EN);

        let chinese = GetCurrentGoalTool::new(store(&dir), "cn");
        let output = chinese.invoke(json!({}), &context()).await.unwrap();
        assert_eq!(output["data"]["message"], json!(NO_GOAL_MESSAGE_CN));
        let unknown = GetCurrentGoalTool::new(store(&dir), "fr");
        let output = unknown.invoke(json!({}), &context()).await.unwrap();
        assert_eq!(output["data"]["message"], json!(NO_GOAL_MESSAGE_CN));
    }

    #[tokio::test]
    async fn get_current_goal_returns_the_record() {
        let dir = tempfile::tempdir().unwrap();
        let store = store(&dir);
        let record = store.start("ship it", now()).await.unwrap();
        let tool = GetCurrentGoalTool::new(Arc::clone(&store), "en");

        // The tool tolerates a non-object payload, like the Python implementation.
        let output = tool.invoke(json!("ignored"), &context()).await.unwrap();
        assert_eq!(output["success"], json!(true));
        let goal = &output["data"]["goal"];
        assert_eq!(goal["goal_id"], json!(record.goal_id));
        assert_eq!(goal["objective"], json!("ship it"));
        assert_eq!(goal["status"], json!("active"));
        assert_eq!(goal["attempt_count"], json!(0));
        assert_eq!(goal["last_assessment"], Value::Null);
        assert_eq!(output["data"].get("message"), None);
        assert_eq!(
            tool.render_for_llm(&output),
            format!(
                "goal_id: {}\nobjective: ship it\nstatus: active\nattempt_count: 0",
                record.goal_id
            )
        );

        store
            .update(
                |record| {
                    record.record_progress(
                        &GoalAssessment::new(
                            GoalAssessmentStatus::Continue,
                            "halfway",
                            Some("write docs".to_owned()),
                            None,
                        ),
                        now(),
                    );
                },
                now(),
            )
            .await
            .unwrap();
        let output = tool.invoke(json!({}), &context()).await.unwrap();
        let rendered = tool.render_for_llm(&output);
        assert!(rendered.contains("\"status\":\"continue\""), "{rendered}");
        assert!(rendered.contains("\"evidence\":\"halfway\""), "{rendered}");
    }

    #[tokio::test]
    async fn get_current_goal_surfaces_a_corrupt_record() {
        let dir = tempfile::tempdir().unwrap();
        let store = store(&dir);
        tokio::fs::create_dir_all(store.path().parent().unwrap())
            .await
            .unwrap();
        tokio::fs::write(store.path(), b"{oops").await.unwrap();
        let tool = GetCurrentGoalTool::new(store, "en");
        let output = tool.invoke(json!({}), &context()).await.unwrap();
        assert_eq!(output["success"], json!(false));
        assert!(output["error"]
            .as_str()
            .unwrap()
            .starts_with("invalid goal record at"));
    }

    #[tokio::test]
    async fn submit_without_a_store_fails_softly() {
        let tool = SubmitGoalReportTool::unavailable("en");
        assert!(tool.store().is_none());
        let output = tool
            .invoke(
                json!({"status": "complete", "evidence": "done"}),
                &context(),
            )
            .await
            .unwrap();
        assert_eq!(output["success"], json!(false));
        assert_eq!(output["error"], json!(NO_STORE_MESSAGE));
        assert_eq!(
            tool.render_for_llm(&output),
            "Goal report accepted (status: continue)."
        );
    }

    #[tokio::test]
    async fn submit_complete_settles_the_record() {
        let dir = tempfile::tempdir().unwrap();
        let store = store(&dir);
        store.start("ship it", now()).await.unwrap();
        let tool = SubmitGoalReportTool::new(Arc::clone(&store), "en");

        let output = tool
            .invoke(
                json!({
                    "status": "complete",
                    "evidence": "all tests pass",
                    "remaining_work": "",
                    "next_instruction": null,
                }),
                &context(),
            )
            .await
            .unwrap();
        assert_eq!(output["success"], json!(true));
        assert_eq!(output["data"]["result"], json!("report_accepted"));
        assert_eq!(output["data"]["status"], json!("complete"));
        assert_eq!(output["data"]["goal_status"], json!("completed"));
        assert_eq!(
            tool.render_for_llm(&output),
            "Goal report accepted (status: complete)."
        );

        let record = store.load().await.unwrap().unwrap();
        assert_eq!(record.status, GoalStatus::Completed);
        assert_eq!(record.last_stop_reason.as_deref(), Some("completed"));
        let assessment = record.last_assessment.unwrap();
        assert_eq!(assessment.status, GoalAssessmentStatus::Complete);
        assert_eq!(assessment.evidence, "all tests pass");
        assert_eq!(assessment.remaining_work, None);
        assert_eq!(assessment.next_instruction, None);
        assert_eq!(output["data"]["goal_id"], json!(record.goal_id));
    }

    #[tokio::test]
    async fn submit_continue_keeps_the_goal_active() {
        let dir = tempfile::tempdir().unwrap();
        let store = store(&dir);
        store.start("ship it", now()).await.unwrap();
        let tool = SubmitGoalReportTool::new(Arc::clone(&store), "en");

        let output = tool
            .invoke(
                json!({
                    "status": "continue",
                    "evidence": "wrote the module",
                    "remaining_work": "wire the rail",
                    "next_instruction": "then run the suite",
                }),
                &context(),
            )
            .await
            .unwrap();
        assert_eq!(output["data"]["status"], json!("continue"));
        assert_eq!(output["data"]["goal_status"], json!("active"));

        let record = store.load().await.unwrap().unwrap();
        assert_eq!(record.status, GoalStatus::Active);
        assert_eq!(record.last_stop_reason, None);
        let assessment = record.last_assessment.unwrap();
        assert_eq!(assessment.remaining_work.as_deref(), Some("wire the rail"));
        assert_eq!(
            assessment.next_instruction.as_deref(),
            Some("then run the suite")
        );
    }

    #[tokio::test]
    async fn submit_normalizes_an_unknown_status() {
        let dir = tempfile::tempdir().unwrap();
        let store = store(&dir);
        store.start("ship it", now()).await.unwrap();
        let tool = SubmitGoalReportTool::new(Arc::clone(&store), "en");

        let output = tool
            .invoke(
                json!({"status": "achieved", "evidence": "done"}),
                &context(),
            )
            .await
            .unwrap();
        assert_eq!(output["success"], json!(true));
        assert_eq!(output["data"]["status"], json!("continue"));
        assert_eq!(
            store.load().await.unwrap().unwrap().status,
            GoalStatus::Active
        );

        // A missing status defaults to `continue` too.
        let output = tool.invoke(json!({}), &context()).await.unwrap();
        assert_eq!(output["data"]["status"], json!("continue"));

        // A non-object payload is a protocol failure, not a crash.
        let output = tool.invoke(json!([]), &context()).await.unwrap();
        assert_eq!(output["success"], json!(false));
        assert_eq!(output["error"], json!("tool arguments must be an object"));
    }

    #[tokio::test]
    async fn submit_blocked_settles_without_a_previous_record() {
        let dir = tempfile::tempdir().unwrap();
        let store = store(&dir);
        let tool = SubmitGoalReportTool::new(Arc::clone(&store), "en");

        let output = tool
            .invoke(
                json!({"status": "blocked", "evidence": "no key"}),
                &context(),
            )
            .await
            .unwrap();
        assert_eq!(output["success"], json!(true));
        assert_eq!(output["data"]["result"], json!("report_accepted"));
        assert_eq!(output["data"]["status"], json!("blocked"));
        assert_eq!(output["data"]["message"], json!(NO_GOAL_MESSAGE_EN));
        assert_eq!(output["data"].get("goal_status"), None);
        // Nothing was created behind the host's back.
        assert!(!store.path().exists());
    }

    #[tokio::test]
    async fn submit_blocked_settles_an_existing_record() {
        let dir = tempfile::tempdir().unwrap();
        let store = store(&dir);
        store.start("ship it", now()).await.unwrap();
        let tool = SubmitGoalReportTool::new(Arc::clone(&store), "cn");

        let output = tool
            .invoke(
                json!({"status": "blocked", "evidence": "missing key"}),
                &context(),
            )
            .await
            .unwrap();
        assert_eq!(output["data"]["goal_status"], json!("blocked"));
        let record = store.load().await.unwrap().unwrap();
        assert_eq!(record.status, GoalStatus::Blocked);
        assert_eq!(record.last_stop_reason.as_deref(), Some("blocked"));
    }

    #[tokio::test]
    async fn store_session_is_stamped_on_created_records() {
        let dir = tempfile::tempdir().unwrap();
        let store = store(&dir);
        assert_eq!(store.session_id(), "s-1");
        let record = store.start("ship it", now()).await.unwrap();
        assert_eq!(record.session_id, "s-1");
        assert_ne!(record.session_id, DEFAULT_SESSION_ID);
    }
}
