//! The guardrail binding: stage mapping and the event handler that runs a
//! [`GuardrailPipeline`].
//!
//! Python registers a guardrail object against callback event strings and lets
//! the callback framework swallow its `GuardrailError`; here the same two-tier
//! policy is expressed with typed events: a blocking outcome becomes
//! [`CallbackVerdict::Abort`], which `trigger` turns into the call's error, and
//! a non-blocking finding is logged and dropped.
#![allow(clippy::module_name_repetitions)]
// A guardrail handler's name is the constant "guardrail"; the trait stays `&str`
// so runtime-named handlers implement it too.
#![allow(clippy::unnecessary_literal_bound)]
#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::unnecessary_wraps
    )
)]

use async_trait::async_trait;
use openjiuwen_security::{GuardrailContext, GuardrailPipeline, ScanStage};
use serde_json::Value;
use std::collections::BTreeMap;
use std::sync::Arc;

use super::context::{CallbackContext, CallbackVerdict};
use super::event::CallbackEvent;
use super::framework::{CallbackFramework, CallbackId, CallbackRegistration};
use super::handler::{CallbackHandler, CallbackResult};
use crate::callback::LOG_TARGET;

/// Priority the binding registers guardrails at (Python `GUARDRAIL_PRIORITY`).
///
/// A guardrail runs before ordinary observers, so a blocked payload is never
/// delivered to a lower-priority handler.
pub const GUARDRAIL_PRIORITY: i32 = 100;

/// Namespace the binding registers guardrails in (Python `"guardrail"`).
pub const GUARDRAIL_NAMESPACE: &str = "guardrail";

/// Tag the binding attaches to every guardrail handler.
pub const GUARDRAIL_TAG: &str = "guardrail";

/// The stages [`bind_guardrails`] subscribes to.
///
/// Python's four framework events. [`ScanStage::UserInput`] is intentionally
/// absent: it has no callback event (`"user_input"` is not `_framework:`-scoped)
/// and is driven through an explicit scan instead.
pub const GUARDRAIL_STAGES: [ScanStage; 4] = [
    ScanStage::LlmInvokeInput,
    ScanStage::LlmInvokeOutput,
    ScanStage::ToolInvokeInput,
    ScanStage::ToolInvokeOutput,
];

/// The scan stage a callback event guards, when it is a guarded stage.
///
/// The four model/tool invoke events map directly; a
/// [`CallbackEvent::Custom`] carrying a known stage name (including
/// `"user_input"`) maps too, so [`event_for_stage`] round-trips.
#[allow(clippy::needless_pass_by_value)]
#[must_use]
pub fn scan_stage_for(event: CallbackEvent) -> Option<ScanStage> {
    match event {
        CallbackEvent::LlmInvokeInput => Some(ScanStage::LlmInvokeInput),
        CallbackEvent::LlmInvokeOutput => Some(ScanStage::LlmInvokeOutput),
        CallbackEvent::ToolInvokeInput => Some(ScanStage::ToolInvokeInput),
        CallbackEvent::ToolInvokeOutput => Some(ScanStage::ToolInvokeOutput),
        CallbackEvent::Custom(name) => match ScanStage::from_event(&name) {
            stage @ (ScanStage::LlmInvokeInput
            | ScanStage::LlmInvokeOutput
            | ScanStage::ToolInvokeInput
            | ScanStage::ToolInvokeOutput
            | ScanStage::UserInput) => Some(stage),
            ScanStage::Other(_) => None,
        },
        _ => None,
    }
}

/// The callback event a scan stage is dispatched as.
///
/// [`ScanStage::UserInput`] has no preset event and becomes
/// `CallbackEvent::Custom("user_input")`; an unknown stage becomes
/// `CallbackEvent::Custom(<event name>)`.
#[must_use]
pub fn event_for_stage(stage: &ScanStage) -> CallbackEvent {
    match stage {
        ScanStage::LlmInvokeInput => CallbackEvent::LlmInvokeInput,
        ScanStage::LlmInvokeOutput => CallbackEvent::LlmInvokeOutput,
        ScanStage::ToolInvokeInput => CallbackEvent::ToolInvokeInput,
        ScanStage::ToolInvokeOutput => CallbackEvent::ToolInvokeOutput,
        ScanStage::UserInput => CallbackEvent::Custom("user_input".to_owned()),
        ScanStage::Other(name) => {
            CallbackEvent::from_name(name).unwrap_or_else(|| CallbackEvent::Custom(name.clone()))
        }
    }
}

/// A [`CallbackHandler`] that runs one stage of a [`GuardrailPipeline`].
///
/// The content to scan is taken from [`CallbackContext::guardrail`] when the
/// caller built it; otherwise it is derived from the context's input (for input
/// stages) or output (for output stages).
pub struct GuardrailHandler {
    pipeline: Arc<GuardrailPipeline>,
    stage: ScanStage,
}

impl GuardrailHandler {
    /// A handler for `stage` over `pipeline`.
    #[must_use]
    pub fn new(pipeline: Arc<GuardrailPipeline>, stage: ScanStage) -> Self {
        Self { pipeline, stage }
    }

    /// The stage this handler scans.
    #[must_use]
    pub fn stage(&self) -> &ScanStage {
        &self.stage
    }

    /// The pipeline this handler runs.
    #[must_use]
    pub fn pipeline(&self) -> &Arc<GuardrailPipeline> {
        &self.pipeline
    }
}

impl std::fmt::Debug for GuardrailHandler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GuardrailHandler")
            .field("stage", &self.stage)
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl CallbackHandler for GuardrailHandler {
    fn name(&self) -> &str {
        "guardrail"
    }

    async fn handle(&self, ctx: &CallbackContext) -> CallbackResult {
        let scan_ctx = ctx
            .guardrail
            .clone()
            .unwrap_or_else(|| derive_context(ctx, &self.stage));
        let outcome = self.pipeline.run(&scan_ctx).await;

        match outcome.enforce() {
            Ok(()) => {
                if !outcome.findings.is_empty() {
                    // Python swallows the non-blocking GuardrailError; the
                    // finding stays visible in logs instead of aborting.
                    tracing::debug!(
                        target: LOG_TARGET,
                        stage = %self.stage,
                        risk_level = %outcome.risk_level,
                        findings = outcome.findings.len(),
                        "guardrail findings do not block the call"
                    );
                }
                Ok(CallbackVerdict::Continue)
            }
            Err(error) => {
                tracing::warn!(
                    target: LOG_TARGET,
                    stage = %self.stage,
                    risk_level = %outcome.risk_level,
                    "guardrail blocked the call"
                );
                Ok(CallbackVerdict::Abort(error))
            }
        }
    }
}

/// Builds a [`GuardrailContext`] from a callback context when the caller did not
/// supply one.
///
/// Input stages scan [`CallbackContext::input`], output stages scan
/// [`CallbackContext::output`]; the component id and a `tool_name` metadata
/// entry are forwarded so a detector can attribute the finding.
fn derive_context(ctx: &CallbackContext, stage: &ScanStage) -> GuardrailContext {
    let content = match stage {
        ScanStage::LlmInvokeOutput | ScanStage::ToolInvokeOutput => ctx
            .output
            .clone()
            .or_else(|| ctx.input.clone())
            .unwrap_or(Value::Null),
        _ => ctx
            .input
            .clone()
            .or_else(|| ctx.output.clone())
            .unwrap_or(Value::Null),
    };
    let mut metadata = BTreeMap::new();
    if !ctx.component.is_empty() {
        metadata.insert("component".to_owned(), Value::String(ctx.component.clone()));
    }
    if let Some(tool_name) = ctx.metadata.get("tool_name") {
        metadata.insert("tool_name".to_owned(), tool_name.clone());
    }
    GuardrailContext {
        content_type: stage.content_type(),
        content,
        event: stage.event().to_owned(),
        metadata,
    }
}

/// Registers one [`GuardrailHandler`] per [`GUARDRAIL_STAGES`] entry.
///
/// Every handler is registered at [`GUARDRAIL_PRIORITY`] in
/// [`GUARDRAIL_NAMESPACE`] and tagged [`GUARDRAIL_TAG`]. Returns the removal
/// handles in stage order.
// The by-value pipeline is the documented API; every handler shares a clone.
#[allow(clippy::needless_pass_by_value)]
pub fn bind_guardrails(
    framework: &CallbackFramework,
    pipeline: Arc<GuardrailPipeline>,
) -> Vec<CallbackId> {
    GUARDRAIL_STAGES
        .iter()
        .map(|stage| {
            let registration = CallbackRegistration::new(
                event_for_stage(stage),
                Arc::new(GuardrailHandler::new(Arc::clone(&pipeline), stage.clone())),
            )
            .with_priority(GUARDRAIL_PRIORITY)
            .with_namespace(GUARDRAIL_NAMESPACE)
            .with_tag(GUARDRAIL_TAG);
            framework.register(registration)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use openjiuwen_common::error::Result;
    use openjiuwen_security::{Guardrail, GuardrailResult, RiskLevel};
    use serde_json::json;

    /// A guardrail that reports a fixed level for one text.
    struct Scripted {
        needle: &'static str,
        level: RiskLevel,
        risk_type: &'static str,
    }

    #[async_trait]
    impl Guardrail for Scripted {
        fn name(&self) -> &str {
            "scripted"
        }

        async fn check(&self, ctx: &GuardrailContext) -> Result<GuardrailResult> {
            if ctx.text_view().contains(self.needle) {
                Ok(GuardrailResult::block(
                    self.level,
                    self.risk_type,
                    None,
                    None,
                ))
            } else {
                Ok(GuardrailResult::pass())
            }
        }
    }

    fn pipeline(needle: &'static str, level: RiskLevel) -> GuardrailPipeline {
        let mut pipeline = GuardrailPipeline::new();
        pipeline.add(Scripted {
            needle,
            level,
            risk_type: "test_risk",
        });
        pipeline
    }

    #[test]
    fn stage_mapping_round_trips_every_guarded_stage() {
        for stage in GUARDRAIL_STAGES {
            let event = event_for_stage(&stage);
            assert_eq!(
                scan_stage_for(event.clone()),
                Some(stage.clone()),
                "{stage}"
            );
            assert_eq!(event.as_str(), stage.event());
        }
        assert_eq!(
            scan_stage_for(CallbackEvent::LlmInvokeInput),
            Some(ScanStage::LlmInvokeInput)
        );
        assert_eq!(
            scan_stage_for(CallbackEvent::ToolInvokeOutput),
            Some(ScanStage::ToolInvokeOutput)
        );
    }

    #[test]
    fn user_input_has_no_preset_event_but_round_trips_as_custom() {
        let event = event_for_stage(&ScanStage::UserInput);
        assert_eq!(event, CallbackEvent::Custom("user_input".to_owned()));
        assert_eq!(event.as_str(), "user_input");
        assert_eq!(scan_stage_for(event), Some(ScanStage::UserInput));
    }

    #[test]
    fn unknown_stages_and_events_do_not_map() {
        assert_eq!(scan_stage_for(CallbackEvent::AgentInvokeInput), None);
        assert_eq!(
            scan_stage_for(CallbackEvent::Custom("_framework:tool_auth".to_owned())),
            None
        );
        assert_eq!(
            event_for_stage(&ScanStage::Other("_framework:tool_auth".to_owned())),
            CallbackEvent::ToolAuth
        );
        assert_eq!(
            event_for_stage(&ScanStage::Other("custom:thing".to_owned())),
            CallbackEvent::Custom("custom:thing".to_owned())
        );
    }

    #[tokio::test]
    async fn an_allowed_outcome_continues() {
        let handler = GuardrailHandler::new(
            Arc::new(pipeline("forbidden", RiskLevel::Critical)),
            ScanStage::LlmInvokeInput,
        );
        let ctx =
            CallbackContext::new(CallbackEvent::LlmInvokeInput).with_input(json!("perfectly fine"));
        assert!(matches!(
            handler.handle(&ctx).await.unwrap(),
            CallbackVerdict::Continue
        ));
        assert_eq!(handler.stage(), &ScanStage::LlmInvokeInput);
    }

    #[tokio::test]
    async fn a_non_blocking_finding_is_dropped_and_dispatch_continues() {
        let handler = GuardrailHandler::new(
            Arc::new(pipeline("suspicious", RiskLevel::High)),
            ScanStage::LlmInvokeInput,
        );
        let ctx = CallbackContext::new(CallbackEvent::LlmInvokeInput)
            .with_input(json!("a suspicious sentence"));
        assert!(matches!(
            handler.handle(&ctx).await.unwrap(),
            CallbackVerdict::Continue
        ));
    }

    #[tokio::test]
    async fn a_blocking_finding_aborts_with_a_security_error() {
        let handler = GuardrailHandler::new(
            Arc::new(pipeline("forbidden", RiskLevel::Critical)),
            ScanStage::LlmInvokeInput,
        );
        let ctx = CallbackContext::new(CallbackEvent::LlmInvokeInput)
            .with_input(json!("forbidden words"));
        let verdict = handler.handle(&ctx).await.unwrap();
        match verdict {
            CallbackVerdict::Abort(error) => {
                assert_eq!(error.kind(), openjiuwen_common::error::ErrorKind::Security);
            }
            other => panic!("expected an abort, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn the_derived_context_reads_output_for_output_stages() {
        let handler = GuardrailHandler::new(
            Arc::new(pipeline("forbidden", RiskLevel::Critical)),
            ScanStage::ToolInvokeOutput,
        );
        let ctx = CallbackContext::new(CallbackEvent::ToolInvokeOutput)
            .with_component("search")
            .with_output(json!({"text": "forbidden"}));
        assert!(matches!(
            handler.handle(&ctx).await.unwrap(),
            CallbackVerdict::Abort(_)
        ));
    }

    #[tokio::test]
    async fn a_supplied_guardrail_context_wins_over_the_payload() {
        let handler = GuardrailHandler::new(
            Arc::new(pipeline("forbidden", RiskLevel::Critical)),
            ScanStage::LlmInvokeInput,
        );
        let supplied = GuardrailContext::text("forbidden", ScanStage::LlmInvokeInput);
        let ctx = CallbackContext::new(CallbackEvent::LlmInvokeInput)
            .with_input(json!("harmless"))
            .with_guardrail(supplied);
        assert!(matches!(
            handler.handle(&ctx).await.unwrap(),
            CallbackVerdict::Abort(_)
        ));
    }

    #[tokio::test]
    async fn binding_registers_one_handler_per_stage() {
        let framework = CallbackFramework::new();
        let ids = bind_guardrails(&framework, Arc::new(GuardrailPipeline::new()));
        assert_eq!(ids.len(), 4);
        assert_eq!(framework.len(), 4);
        for stage in GUARDRAIL_STAGES {
            let event = event_for_stage(&stage);
            let handlers = framework.handlers(event);
            assert_eq!(handlers.len(), 1, "{stage}");
            assert_eq!(handlers[0].priority, GUARDRAIL_PRIORITY);
            assert_eq!(handlers[0].namespace, GUARDRAIL_NAMESPACE);
            assert!(handlers[0].tags.contains(GUARDRAIL_TAG));
            assert_eq!(handlers[0].name, "guardrail");
        }
        assert_eq!(framework.unregister_tag(GUARDRAIL_TAG), 4);
        assert!(framework.is_empty());
    }

    #[tokio::test]
    async fn a_bound_guardrail_aborts_and_stops_lower_priority_handlers() {
        let framework = CallbackFramework::new();
        bind_guardrails(
            &framework,
            Arc::new(pipeline("forbidden", RiskLevel::Critical)),
        );
        let ran = Arc::new(std::sync::Mutex::new(false));
        let ran_clone = Arc::clone(&ran);
        framework.register_sync(
            CallbackEvent::LlmInvokeInput,
            0,
            move |_: &CallbackContext| {
                *ran_clone.lock().unwrap() = true;
                Ok(CallbackVerdict::Continue)
            },
        );

        let error = framework
            .trigger(
                CallbackEvent::LlmInvokeInput,
                &CallbackContext::new(CallbackEvent::LlmInvokeInput).with_input(json!("forbidden")),
            )
            .await
            .unwrap_err();
        assert_eq!(error.kind(), openjiuwen_common::error::ErrorKind::Security);
        assert!(!*ran.lock().unwrap());
    }
}
