//! The ordered guardrail pipeline.
//!
//! Python binds guardrails to callback events and never produces a combined
//! verdict; this pipeline is the addition that runs the registered guardrails
//! for one scan, aggregates their verdicts, and applies a
//! [`SecurityPolicy`].
//!
//! Runs are **infallible by design**: a guardrail failure is data (recorded in
//! [`GuardrailOutcome::errors`](crate::GuardrailOutcome::errors) and logged),
//! never an early return, so the caller decides whether to continue by calling
//! [`GuardrailOutcome::enforce`](crate::GuardrailOutcome::enforce).
//!
//! Failures are **fail-closed by default**
//! ([`FailurePolicy::FailClosed`](crate::FailurePolicy::FailClosed)): a guardrail
//! that errors, and a payload [`GuardrailContext::text_view_checked`] cannot
//! project, both become a blocking [`RiskLevel::Critical`] finding whose message
//! carries the reason, so an unusable detector can never read as "safe".
//! [`FailurePolicy::FailOpen`](crate::FailurePolicy::FailOpen) is the explicit
//! opt-in that records the failure and lets the call continue.

use std::sync::Arc;

use openjiuwen_foundation::prompt::BaseMessage;
use serde_json::Value;

use crate::finding::finding_message;
use crate::{
    FailurePolicy, Finding, Guardrail, GuardrailContext, GuardrailEvaluation, GuardrailOutcome,
    RiskLevel, ScanStage, SecurityDecision, SecurityPolicy,
};

/// `risk_type` of a finding derived from a guardrail failure.
pub const GUARDRAIL_ERROR_RISK_TYPE: &str = "guardrail_error";

/// `risk_type` of a finding derived from a payload no detector can read.
pub const UNSCANNABLE_RISK_TYPE: &str = "unscannable_content";

/// Name reported for the payload pre-check in `evaluations` and `findings`.
pub const CONTEXT_GUARDRAIL: &str = "context";

/// Guardrails plus the policy that decides which findings block.
///
/// Ordering is [`Guardrail::priority`] descending, registration order as the
/// tiebreak, so a pipeline is deterministic for a fixed registration sequence.
pub struct GuardrailPipeline {
    /// Registered guardrails, in registration order.
    guardrails: Vec<Arc<dyn Guardrail>>,
    /// Thresholds and failure handling.
    policy: SecurityPolicy,
}

impl GuardrailPipeline {
    /// An empty pipeline with the default policy.
    #[must_use]
    pub fn new() -> Self {
        Self {
            guardrails: Vec::new(),
            policy: SecurityPolicy::default(),
        }
    }

    /// An empty pipeline with `policy`.
    #[must_use]
    pub fn with_policy(mut self, policy: SecurityPolicy) -> Self {
        self.policy = policy;
        self
    }

    /// The policy the pipeline applies.
    #[must_use]
    pub fn policy(&self) -> &SecurityPolicy {
        &self.policy
    }

    /// Registers one guardrail; it is wrapped in an [`Arc`].
    pub fn add(&mut self, guardrail: impl Guardrail + 'static) {
        self.guardrails.push(Arc::new(guardrail));
    }

    /// Registers an already shared guardrail.
    pub fn add_arc(&mut self, guardrail: Arc<dyn Guardrail>) {
        self.guardrails.push(guardrail);
    }

    /// Registers several guardrails, keeping the iterator order.
    pub fn extend(&mut self, guardrails: impl IntoIterator<Item = Arc<dyn Guardrail>>) {
        self.guardrails.extend(guardrails);
    }

    /// The registered guardrails, in registration order.
    #[must_use]
    pub fn guardrails(&self) -> &[Arc<dyn Guardrail>] {
        &self.guardrails
    }

    /// Runs every guardrail that handles the context's stage and aggregates the
    /// verdicts.
    ///
    /// Never fails: guardrail errors and an unscannable context are recorded and
    /// resolved by the policy — blocking under the default
    /// [`FailurePolicy::FailClosed`](crate::FailurePolicy::FailClosed), recorded
    /// only under the opt-in [`FailurePolicy::FailOpen`](crate::FailurePolicy::FailOpen).
    pub async fn run(&self, ctx: &GuardrailContext) -> GuardrailOutcome {
        let stage = ctx.stage();
        let mut selected: Vec<&Arc<dyn Guardrail>> = self
            .guardrails
            .iter()
            .filter(|guardrail| guardrail.handles(stage.clone()))
            .collect();
        // Stable sort: equal priorities keep registration order.
        selected.sort_by_key(|guardrail| std::cmp::Reverse(guardrail.priority()));

        let mut outcome = GuardrailOutcome::allowed(stage);
        if selected.is_empty() {
            return outcome;
        }

        // A payload no detector can read is a scan failure, never a pass: the
        // checked projection reports the reason and the policy decides.
        if let Err(err) = ctx.text_view_checked() {
            tracing::warn!(
                target: "openjiuwen_security::pipeline",
                stage = %ctx.event,
                content_type = ?ctx.content_type,
                error = %err,
                "guardrail context cannot be scanned"
            );
            let blocked =
                self.record_failure(&mut outcome, CONTEXT_GUARDRAIL, UNSCANNABLE_RISK_TYPE, &err);
            if blocked && self.policy.stop_on_block {
                return outcome;
            }
        }

        for guardrail in selected {
            let name = guardrail.name().to_owned();
            match guardrail.check(ctx).await {
                Ok(result) => {
                    outcome.risk_level = outcome.risk_level.max(result.risk_level);
                    outcome.evaluations.push(GuardrailEvaluation {
                        guardrail: name.clone(),
                        result: Some(result.clone()),
                        error: None,
                    });
                    if !result.is_safe() {
                        let blocking = result.risk_level >= self.policy.block_level;
                        let risk_type = result.risk_type_or_unknown().to_owned();
                        outcome.findings.push(Finding {
                            guardrail: name.clone(),
                            message: finding_message(&risk_type, result.risk_level, blocking),
                            risk_level: result.risk_level,
                            risk_type,
                            blocking,
                            details: result.details.clone(),
                            modified_data: result.modified_data.clone(),
                        });
                        if blocking {
                            outcome.decision = SecurityDecision::Block;
                            if outcome.blocked_by.is_none() {
                                outcome.blocked_by = Some(name);
                            }
                            if self.policy.stop_on_block {
                                break;
                            }
                        }
                    }
                }
                Err(err) => {
                    let blocked =
                        self.record_failure(&mut outcome, &name, GUARDRAIL_ERROR_RISK_TYPE, &err);
                    if blocked && self.policy.stop_on_block {
                        break;
                    }
                }
            }
        }
        outcome
    }

    /// Records one scan failure and applies the [`FailurePolicy`].
    ///
    /// The entry is `"<guardrail>: <error>"` (the message never carries scanned
    /// content: only the guardrail name and the error text), and the evaluation
    /// keeps `result: None` so the failure can never be mistaken for a verdict.
    /// Under [`FailurePolicy::FailClosed`] the failure also becomes a blocking
    /// [`RiskLevel::Critical`] finding whose message and `details.error` carry
    /// the reason; under [`FailurePolicy::FailOpen`] it is only recorded.
    ///
    /// Returns whether the failure blocked the call.
    fn record_failure(
        &self,
        outcome: &mut GuardrailOutcome,
        guardrail: &str,
        risk_type: &str,
        err: &openjiuwen_common::error::Error,
    ) -> bool {
        tracing::warn!(
            target: "openjiuwen_security::pipeline",
            stage = %outcome.stage,
            guardrail = %guardrail,
            failure_policy = ?self.policy.failure_policy,
            error = %err,
            "guardrail evaluation failed"
        );
        let entry = format!("{guardrail}: {err}");
        outcome.errors.push(entry.clone());
        outcome.evaluations.push(GuardrailEvaluation {
            guardrail: guardrail.to_owned(),
            result: None,
            error: Some(entry),
        });
        if self.policy.failure_policy != FailurePolicy::FailClosed {
            return false;
        }

        let reason = err.to_string();
        outcome.risk_level = RiskLevel::Critical;
        outcome.findings.push(Finding {
            guardrail: guardrail.to_owned(),
            risk_level: RiskLevel::Critical,
            risk_type: risk_type.to_owned(),
            message: format!("Guardrail blocked: {risk_type} risk detected ({reason})"),
            blocking: true,
            details: Some(serde_json::json!({ "error": reason })),
            modified_data: None,
        });
        outcome.decision = SecurityDecision::Block;
        if outcome.blocked_by.is_none() {
            outcome.blocked_by = Some(guardrail.to_owned());
        }
        true
    }

    /// Scans `text` at `stage`.
    pub async fn scan_text(&self, stage: ScanStage, text: impl Into<String>) -> GuardrailOutcome {
        self.run(&GuardrailContext::text(text, stage)).await
    }

    /// Scans raw user input ([`ScanStage::UserInput`]).
    pub async fn scan_user_input(&self, text: impl Into<String>) -> GuardrailOutcome {
        self.scan_text(ScanStage::UserInput, text).await
    }

    /// Scans the messages about to be sent to the model
    /// ([`ScanStage::LlmInvokeInput`]).
    pub async fn scan_prompt(&self, messages: &[BaseMessage]) -> GuardrailOutcome {
        self.run(&GuardrailContext::messages(
            messages,
            ScanStage::LlmInvokeInput,
        ))
        .await
    }

    /// Scans the model's response ([`ScanStage::LlmInvokeOutput`]).
    pub async fn scan_model_output(&self, text: impl Into<String>) -> GuardrailOutcome {
        self.scan_text(ScanStage::LlmInvokeOutput, text).await
    }

    /// Scans tool-call arguments ([`ScanStage::ToolInvokeInput`]).
    pub async fn scan_tool_input(
        &self,
        tool_name: impl Into<String>,
        arguments: &Value,
    ) -> GuardrailOutcome {
        self.run(&GuardrailContext::tool_call(
            tool_name,
            arguments,
            ScanStage::ToolInvokeInput,
        ))
        .await
    }

    /// Scans a tool result ([`ScanStage::ToolInvokeOutput`]).
    pub async fn scan_tool_output(
        &self,
        tool_name: impl Into<String>,
        output: &Value,
    ) -> GuardrailOutcome {
        self.run(&GuardrailContext::tool_output(
            tool_name,
            output,
            ScanStage::ToolInvokeOutput,
        ))
        .await
    }
}

impl Default for GuardrailPipeline {
    /// An empty pipeline with the default policy.
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use async_trait::async_trait;
    use openjiuwen_common::error::{Error, ErrorKind, Result};
    use serde_json::json;

    use super::*;
    use crate::{GuardrailResult, GuardrailViolation};

    /// What a [`Recording`] guardrail answers when checked.
    #[derive(Clone)]
    enum Script {
        Pass,
        Unsafe {
            level: RiskLevel,
            risk_type: &'static str,
            details: Option<Value>,
            modified_data: Option<Value>,
        },
        Fail(&'static str),
    }

    /// A guardrail that records every call and its context.
    struct Recording {
        name: &'static str,
        priority: i32,
        stages: Vec<ScanStage>,
        script: Script,
        calls: Arc<Mutex<Vec<String>>>,
        contexts: Arc<Mutex<Vec<GuardrailContext>>>,
    }

    impl Recording {
        fn new(name: &'static str, script: Script) -> Self {
            Self {
                name,
                priority: crate::DEFAULT_PRIORITY,
                stages: Vec::new(),
                script,
                calls: Arc::new(Mutex::new(Vec::new())),
                contexts: Arc::new(Mutex::new(Vec::new())),
            }
        }

        fn with_priority(mut self, priority: i32) -> Self {
            self.priority = priority;
            self
        }

        fn with_stages(mut self, stages: Vec<ScanStage>) -> Self {
            self.stages = stages;
            self
        }
    }

    #[async_trait]
    impl Guardrail for Recording {
        fn name(&self) -> &str {
            self.name
        }

        fn priority(&self) -> i32 {
            self.priority
        }

        fn stages(&self) -> &[ScanStage] {
            &self.stages
        }

        async fn check(&self, ctx: &GuardrailContext) -> Result<GuardrailResult> {
            self.calls.lock().unwrap().push(self.name.to_owned());
            self.contexts.lock().unwrap().push(ctx.clone());
            match &self.script {
                Script::Pass => Ok(GuardrailResult::pass()),
                Script::Unsafe {
                    level,
                    risk_type,
                    details,
                    modified_data,
                } => Ok(GuardrailResult::block(
                    *level,
                    *risk_type,
                    details.clone(),
                    modified_data.clone(),
                )),
                Script::Fail(message) => Err(Error::security(*message)),
            }
        }
    }

    fn blocking(level: RiskLevel, risk_type: &'static str) -> Script {
        Script::Unsafe {
            level,
            risk_type,
            details: Some(json!({"pattern": risk_type})),
            modified_data: None,
        }
    }

    // ports tests/unit_tests/core/security/guardrail/test_guardrail_base.py
    #[tokio::test]
    async fn runs_by_descending_priority_with_registration_tiebreak() {
        let calls: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let mut first = Recording::new("first", Script::Pass).with_priority(100);
        first.calls = calls.clone();
        let mut urgent = Recording::new("urgent", Script::Pass).with_priority(500);
        urgent.calls = calls.clone();
        let mut last = Recording::new("last", Script::Pass).with_priority(100);
        last.calls = calls.clone();
        let mut pipeline = GuardrailPipeline::new();
        pipeline.add(first);
        pipeline.add(urgent);
        pipeline.add(last);
        assert_eq!(pipeline.guardrails().len(), 3);

        let outcome = pipeline.scan_user_input("hello").await;
        assert_eq!(
            calls.lock().unwrap().clone(),
            vec!["urgent".to_owned(), "first".to_owned(), "last".to_owned()]
        );
        assert_eq!(outcome.evaluations.len(), 3);
        assert!(outcome.is_safe());
    }

    #[tokio::test]
    async fn stage_filtering_skips_irrelevant_guardrails() {
        let tool_only =
            Recording::new("tool", Script::Pass).with_stages(vec![ScanStage::ToolInvokeInput]);
        let calls = tool_only.calls.clone();
        let mut pipeline = GuardrailPipeline::new();
        pipeline.add(tool_only);

        let text = pipeline.scan_user_input("hello").await;
        assert_eq!(text.evaluations.len(), 0);
        assert_eq!(calls.lock().unwrap().len(), 0);

        let tool = pipeline
            .scan_tool_input("shell", &json!({"cmd": "ls"}))
            .await;
        assert_eq!(tool.evaluations.len(), 1);
        assert_eq!(tool.evaluations[0].guardrail, "tool");
        assert_eq!(calls.lock().unwrap().len(), 1);
    }

    // ports tests/unit_tests/core/security/guardrail/test_guardrail_base.py
    #[tokio::test]
    async fn severity_matrix_allows_below_critical_and_blocks_critical() {
        for level in [RiskLevel::Low, RiskLevel::Medium, RiskLevel::High] {
            let mut pipeline = GuardrailPipeline::new();
            pipeline.add(Recording::new("g", blocking(level, "pii")));
            let outcome = pipeline.scan_user_input("x").await;
            assert!(outcome.is_safe(), "{level} must not block by default");
            assert_eq!(outcome.risk_level, level);
            assert_eq!(outcome.findings.len(), 1);
            assert!(!outcome.findings[0].blocking);
            assert_eq!(
                outcome.findings[0].message,
                format!("Security finding: pii (level={})", level.name())
            );
            assert_eq!(outcome.blocked_by, None);
            assert!(outcome.enforce().is_ok());
        }

        let mut pipeline = GuardrailPipeline::new();
        pipeline.add(Recording::new(
            "critical_guard",
            blocking(RiskLevel::Critical, "jailbreak"),
        ));
        let outcome = pipeline.scan_user_input("x").await;
        assert!(outcome.is_blocked());
        assert_eq!(outcome.blocked_by.as_deref(), Some("critical_guard"));
        assert_eq!(outcome.findings.len(), 1);
        assert!(outcome.findings[0].blocking);
        assert_eq!(
            outcome.findings[0].message,
            "Critical security risk detected: jailbreak"
        );
        assert_eq!(outcome.enforce().unwrap_err().kind(), ErrorKind::Security);
    }

    #[tokio::test]
    async fn lowered_block_level_blocks_with_guardrail_error_wording() {
        let mut pipeline = GuardrailPipeline::new().with_policy(SecurityPolicy {
            block_level: RiskLevel::High,
            ..SecurityPolicy::default()
        });
        pipeline.add(Recording::new("g", blocking(RiskLevel::High, "pii")));

        let outcome = pipeline.scan_user_input("x").await;
        assert!(outcome.is_blocked());
        assert_eq!(outcome.risk_level, RiskLevel::High);
        assert_eq!(
            outcome.findings[0].message,
            "Guardrail blocked: pii risk detected"
        );
        assert_eq!(pipeline.policy().block_level, RiskLevel::High);
    }

    #[tokio::test]
    async fn stop_on_block_false_runs_the_remaining_guardrails() {
        let blocker = Recording::new("blocker", blocking(RiskLevel::Critical, "jailbreak"))
            .with_priority(200);
        let after = Recording::new("after", Script::Pass).with_priority(100);
        let calls = after.calls.clone();

        let mut stopping = GuardrailPipeline::new();
        stopping.add(
            Recording::new("blocker", blocking(RiskLevel::Critical, "jailbreak"))
                .with_priority(200),
        );
        stopping.add(after);
        let outcome = stopping.scan_user_input("x").await;
        assert!(outcome.is_blocked());
        assert!(calls.lock().unwrap().is_empty(), "stop_on_block must break");
        assert_eq!(outcome.evaluations.len(), 1);

        let calls = blocker.calls.clone();
        let tail_calls: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let mut lenient = GuardrailPipeline::new().with_policy(SecurityPolicy {
            stop_on_block: false,
            ..SecurityPolicy::default()
        });
        lenient.add(blocker);
        lenient.add({
            let mut tail = Recording::new("tail", Script::Pass).with_priority(100);
            tail.calls = tail_calls.clone();
            tail
        });
        let outcome = lenient.scan_user_input("x").await;
        assert!(outcome.is_blocked());
        assert_eq!(outcome.blocked_by.as_deref(), Some("blocker"));
        assert_eq!(
            calls.lock().unwrap().clone(),
            vec!["blocker".to_owned()],
            "the blocker ran"
        );
        assert_eq!(
            tail_calls.lock().unwrap().clone(),
            vec!["tail".to_owned()],
            "the tail guardrail still ran"
        );
        assert_eq!(outcome.evaluations.len(), 2);
    }

    #[tokio::test]
    async fn fail_open_opt_in_records_the_error_and_allows() {
        let mut pipeline = GuardrailPipeline::new().with_policy(SecurityPolicy {
            failure_policy: FailurePolicy::FailOpen,
            ..SecurityPolicy::default()
        });
        pipeline.add(Recording::new("broken", Script::Fail("backend exploded")));
        pipeline.add(Recording::new("fine", Script::Pass));

        let outcome = pipeline.scan_user_input("x").await;
        assert!(outcome.is_safe());
        assert_eq!(outcome.errors.len(), 1);
        assert_eq!(outcome.errors[0], "broken: security: backend exploded");
        assert_eq!(outcome.evaluations.len(), 2);
        assert_eq!(outcome.evaluations[0].result, None);
        assert_eq!(
            outcome.evaluations[0].error.as_deref(),
            Some("broken: security: backend exploded")
        );
        assert_eq!(outcome.findings.len(), 0);
    }

    #[tokio::test]
    async fn a_guardrail_error_blocks_by_default() {
        // The default policy is fail-closed: an unusable guardrail never reads
        // as "safe".
        let mut pipeline = GuardrailPipeline::new();
        assert_eq!(
            pipeline.policy().failure_policy,
            FailurePolicy::FailClosed,
            "the default policy must fail closed"
        );
        pipeline.add(Recording::new("broken", Script::Fail("backend exploded")));

        let outcome = pipeline.scan_user_input("x").await;
        assert!(outcome.is_blocked());
        assert_eq!(outcome.risk_level, RiskLevel::Critical);
        assert_eq!(outcome.blocked_by.as_deref(), Some("broken"));
        assert_eq!(outcome.findings.len(), 1);
        assert_eq!(outcome.findings[0].guardrail, "broken");
        assert_eq!(outcome.findings[0].risk_type, GUARDRAIL_ERROR_RISK_TYPE);
        assert_eq!(outcome.findings[0].risk_level, RiskLevel::Critical);
        assert!(outcome.findings[0].blocking);
        assert_eq!(
            outcome.findings[0].message,
            "Guardrail blocked: guardrail_error risk detected (security: backend exploded)"
        );
        assert_eq!(
            outcome.findings[0].details,
            Some(json!({"error": "security: backend exploded"}))
        );
        assert_eq!(outcome.errors.len(), 1);
        assert_eq!(outcome.errors[0], "broken: security: backend exploded");
        assert_eq!(
            outcome.violation().unwrap().risk_type,
            GUARDRAIL_ERROR_RISK_TYPE
        );
    }

    #[tokio::test]
    async fn an_unscannable_context_blocks_by_default() {
        let mut pipeline = GuardrailPipeline::new();
        pipeline.add(Recording::new("recorder", Script::Pass));

        let outcome = pipeline
            .run(&GuardrailContext::raw("user_input", Value::Null))
            .await;
        assert!(outcome.is_blocked());
        assert_eq!(outcome.blocked_by.as_deref(), Some(CONTEXT_GUARDRAIL));
        assert_eq!(outcome.findings.len(), 1);
        assert_eq!(outcome.findings[0].risk_type, UNSCANNABLE_RISK_TYPE);
        assert!(outcome.findings[0].message.contains("unscannable content"));
        assert_eq!(outcome.evaluations.len(), 1);
        assert_eq!(outcome.evaluations[0].guardrail, CONTEXT_GUARDRAIL);
        assert!(outcome.evaluations[0].error.is_some());
        assert!(outcome.enforce().is_err());
    }

    #[tokio::test]
    async fn an_unscannable_context_is_only_recorded_under_fail_open() {
        let mut pipeline = GuardrailPipeline::new().with_policy(SecurityPolicy {
            failure_policy: FailurePolicy::FailOpen,
            ..SecurityPolicy::default()
        });
        pipeline.add(Recording::new("recorder", Script::Pass));

        let outcome = pipeline
            .run(&GuardrailContext::raw("user_input", Value::Null))
            .await;
        assert!(outcome.is_safe());
        assert_eq!(outcome.errors.len(), 1);
        assert_eq!(outcome.evaluations.len(), 2, "the guardrail still ran");
        assert_eq!(outcome.findings.len(), 0);
    }

    #[tokio::test]
    async fn an_empty_pipeline_does_not_scan_and_does_not_fail() {
        // Without a registered guardrail there is no detector whose failure
        // could be hidden, so the pre-check stays silent.
        let pipeline = GuardrailPipeline::default();
        let outcome = pipeline
            .run(&GuardrailContext::raw("user_input", Value::Null))
            .await;
        assert!(outcome.is_safe());
        assert_eq!(outcome.errors, Vec::<String>::new());
        assert_eq!(outcome.evaluations, Vec::<GuardrailEvaluation>::new());
    }

    #[tokio::test]
    async fn modified_data_is_merged_across_guardrails() {
        let mut pipeline = GuardrailPipeline::new();
        pipeline.add(
            Recording::new(
                "first",
                Script::Unsafe {
                    level: RiskLevel::High,
                    risk_type: "pii",
                    details: None,
                    modified_data: Some(json!({"text": "one", "keep": 1})),
                },
            )
            .with_priority(200),
        );
        pipeline.add(Recording::new(
            "second",
            Script::Unsafe {
                level: RiskLevel::Medium,
                risk_type: "secrets",
                details: None,
                modified_data: Some(json!({"text": "two"})),
            },
        ));

        let outcome = pipeline.scan_user_input("x").await;
        assert_eq!(
            outcome.modified_data(),
            Some(json!({"text": "two", "keep": 1}))
        );
    }

    #[tokio::test]
    async fn scan_helpers_build_the_documented_contexts() {
        let recorder = Recording::new("recorder", Script::Pass);
        let contexts = recorder.contexts.clone();
        let mut pipeline = GuardrailPipeline::new();
        pipeline.add(recorder);

        let last = || -> GuardrailContext {
            contexts
                .lock()
                .unwrap()
                .last()
                .cloned()
                .expect("a recorded context")
        };

        pipeline.scan_user_input("hello").await;
        let ctx = last();
        assert_eq!(ctx.stage(), ScanStage::UserInput);
        assert_eq!(ctx.event, crate::EVENT_USER_INPUT);
        assert_eq!(ctx.get_text(), Some("hello"));

        pipeline
            .scan_prompt(&[BaseMessage::user("first"), BaseMessage::assistant("second")])
            .await;
        let ctx = last();
        assert_eq!(ctx.stage(), ScanStage::LlmInvokeInput);
        assert_eq!(ctx.event, crate::EVENT_LLM_INVOKE_INPUT);
        assert_eq!(ctx.get_messages().map(<[Value]>::len), Some(2));

        pipeline
            .scan_tool_input("search", &json!({"q": "rust"}))
            .await;
        let ctx = last();
        assert_eq!(ctx.stage(), ScanStage::ToolInvokeInput);
        assert_eq!(ctx.get_tool_name(), Some("search"));
        assert_eq!(ctx.content, json!({"q": "rust"}));

        pipeline
            .scan_tool_output("search", &json!({"hits": 3}))
            .await;
        let ctx = last();
        assert_eq!(ctx.stage(), ScanStage::ToolInvokeOutput);
        assert_eq!(ctx.event, crate::EVENT_TOOL_INVOKE_OUTPUT);
        assert_eq!(ctx.get_tool_name(), Some("search"));

        pipeline.scan_model_output("answer").await;
        let ctx = last();
        assert_eq!(ctx.stage(), ScanStage::LlmInvokeOutput);
        assert_eq!(ctx.event, crate::EVENT_LLM_INVOKE_OUTPUT);

        assert_eq!(contexts.lock().unwrap().len(), 5);
    }

    #[tokio::test]
    async fn enforce_on_a_blocked_outcome_yields_a_security_error() {
        let mut pipeline = GuardrailPipeline::new();
        pipeline.add(Recording::new("g", blocking(RiskLevel::Critical, "pii")));
        let outcome = pipeline.scan_user_input("x").await;

        let error = outcome.enforce().unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Security);
        assert!(error
            .to_string()
            .contains("Critical security risk detected"));
        let violation: GuardrailViolation = outcome.violation().unwrap();
        assert_eq!(violation.risk_type, "pii");
        assert_eq!(violation.event, crate::EVENT_USER_INPUT);
    }

    #[tokio::test]
    async fn empty_pipeline_allows_the_scan() {
        let pipeline = GuardrailPipeline::default();
        let outcome = pipeline.scan_text(ScanStage::ToolInvokeOutput, "x").await;
        assert!(outcome.is_safe());
        assert_eq!(outcome.evaluations.len(), 0);
        assert_eq!(outcome.stage, ScanStage::ToolInvokeOutput);
        assert_eq!(pipeline.guardrails().len(), 0);
        assert_eq!(pipeline.policy(), &SecurityPolicy::default());
    }
}
