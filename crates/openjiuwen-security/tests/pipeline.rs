#![allow(clippy::unwrap_used, clippy::expect_used)]

//! Behavior suite for the ordered guardrail pipeline.
//!
//! Ports the two-tier policy of
//! `tests/unit_tests/core/security/guardrail/test_guardrail_base.py` (only
//! `CRITICAL` aborts, lower severities are recorded) and pins the aggregation,
//! ordering, stage filtering, failure policy, scan helpers, and redacted
//! reporting this crate adds on top.
//!
//! Everything runs offline against an in-file `FakeGuardrail`; no backend,
//! network, or wall clock is involved.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use openjiuwen_common::error::{Error, ErrorKind, Result};
use openjiuwen_foundation::prompt::BaseMessage;
use openjiuwen_security::finding::finding_message;
use openjiuwen_security::{
    FailurePolicy, Guardrail, GuardrailContext, GuardrailPipeline, GuardrailResult,
    RedactedOutcome, RiskLevel, ScanStage, SecurityDecision, SecurityPolicy,
};
use serde_json::{json, Value};

/// What a [`FakeGuardrail`] answers when it is checked.
#[derive(Clone)]
enum Outcome {
    /// A safe pass without details.
    Pass,
    /// A precise verdict, safe or unsafe.
    Result(GuardrailResult),
    /// A guardrail failure carrying this message.
    Fail(String),
}

/// A guardrail that records every call (name and context) and answers a fixed
/// script, so ordering, filtering, and failure handling are observable.
struct FakeGuardrail {
    /// Stable name reported to the pipeline.
    name: String,
    /// Ordering weight.
    priority: i32,
    /// Stages this guardrail handles; empty means every stage.
    stages: Vec<ScanStage>,
    /// The verdict(s) it returns.
    script: Outcome,
    /// Shared recorder of guardrail names, in call order.
    calls: Arc<Mutex<Vec<String>>>,
    /// Contexts seen by this guardrail, oldest first.
    seen: Arc<Mutex<Vec<GuardrailContext>>>,
}

impl FakeGuardrail {
    /// A guardrail handling every stage with its own call recorder.
    fn new(name: &str, priority: i32, script: Outcome) -> Self {
        Self {
            name: name.to_owned(),
            priority,
            stages: Vec::new(),
            script,
            calls: Arc::new(Mutex::new(Vec::new())),
            seen: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Restricts the guardrail to `stages`; empty scans every stage.
    fn with_stages(mut self, stages: Vec<ScanStage>) -> Self {
        self.stages = stages;
        self
    }

    /// Shares `calls` with the test so several guardrails record one order.
    fn recording_calls(mut self, calls: Arc<Mutex<Vec<String>>>) -> Self {
        self.calls = calls;
        self
    }
}

#[async_trait]
impl Guardrail for FakeGuardrail {
    fn name(&self) -> &str {
        &self.name
    }

    fn priority(&self) -> i32 {
        self.priority
    }

    fn stages(&self) -> &[ScanStage] {
        &self.stages
    }

    async fn check(&self, ctx: &GuardrailContext) -> Result<GuardrailResult> {
        self.calls.lock().unwrap().push(self.name.clone());
        self.seen.lock().unwrap().push(ctx.clone());
        match &self.script {
            Outcome::Pass => Ok(GuardrailResult::pass()),
            Outcome::Result(result) => Ok(result.clone()),
            Outcome::Fail(message) => Err(Error::security(message.clone())),
        }
    }
}

/// A shared call-order recorder.
fn call_log() -> Arc<Mutex<Vec<String>>> {
    Arc::new(Mutex::new(Vec::new()))
}

/// The guardrail names recorded so far, in call order.
fn call_order(calls: &Arc<Mutex<Vec<String>>>) -> Vec<String> {
    calls.lock().unwrap().clone()
}

/// The context recorded by the most recent check.
fn last_context(seen: &Arc<Mutex<Vec<GuardrailContext>>>) -> GuardrailContext {
    seen.lock()
        .unwrap()
        .last()
        .cloned()
        .expect("a recorded guardrail context")
}

/// An unsafe verdict at `level` with non-sensitive evidence.
fn unsafe_result(level: RiskLevel, risk_type: &str) -> Outcome {
    Outcome::Result(GuardrailResult::block(
        level,
        risk_type,
        Some(json!({"pattern": risk_type})),
        None,
    ))
}

// ports tests/unit_tests/core/security/guardrail/test_guardrail_base.py
#[tokio::test]
async fn ordering_is_priority_descending_with_registration_tiebreak() {
    let calls = call_log();
    let mut pipeline = GuardrailPipeline::new();
    for (name, priority) in [("low", 50), ("high", 200), ("mid1", 100), ("mid2", 100)] {
        pipeline
            .add(FakeGuardrail::new(name, priority, Outcome::Pass).recording_calls(calls.clone()));
    }

    let outcome = pipeline.scan_user_input("hi").await;
    assert_eq!(call_order(&calls), ["high", "mid1", "mid2", "low"]);
    assert!(outcome.is_safe());
    assert_eq!(outcome.evaluations.len(), 4);
    assert_eq!(outcome.stage, ScanStage::UserInput);
}

#[tokio::test]
async fn stage_filtering_runs_only_the_matching_guardrails() {
    let tool_only = FakeGuardrail::new("tool_only", 100, Outcome::Pass)
        .with_stages(vec![ScanStage::ToolInvokeInput]);
    let tool_calls = tool_only.calls.clone();

    let every_stage = FakeGuardrail::new("every_stage", 100, Outcome::Pass);
    let every_calls = every_stage.calls.clone();

    let mut pipeline = GuardrailPipeline::new();
    pipeline.add(tool_only);
    pipeline.add(every_stage);

    let user = pipeline.scan_user_input("hi").await;
    assert_eq!(call_order(&tool_calls).len(), 0);
    assert_eq!(call_order(&every_calls), ["every_stage"]);
    assert_eq!(user.evaluations.len(), 1);
    assert_eq!(user.evaluations[0].guardrail, "every_stage");

    let tool = pipeline.scan_tool_input("search", &json!({"q": "x"})).await;
    assert_eq!(call_order(&tool_calls), ["tool_only"]);
    assert_eq!(call_order(&every_calls), ["every_stage", "every_stage"]);
    assert_eq!(tool.evaluations.len(), 2);
}

#[tokio::test]
async fn empty_stages_handle_every_stage() {
    let every_stage = FakeGuardrail::new("every_stage", 100, Outcome::Pass);
    let calls = every_stage.calls.clone();
    let mut pipeline = GuardrailPipeline::new();
    pipeline.add(every_stage);

    pipeline.scan_user_input("hi").await;
    pipeline.scan_model_output("out").await;
    pipeline.scan_tool_input("search", &json!({"q": "x"})).await;
    pipeline.scan_tool_output("search", &json!({"r": 1})).await;
    pipeline.scan_prompt(&[BaseMessage::user("u")]).await;

    assert_eq!(
        call_order(&calls),
        [
            "every_stage",
            "every_stage",
            "every_stage",
            "every_stage",
            "every_stage",
        ]
    );
}

// ports tests/unit_tests/core/security/guardrail/test_guardrail_base.py
#[tokio::test]
async fn findings_below_the_block_level_are_recorded_but_allowed() {
    for level in [RiskLevel::Low, RiskLevel::Medium, RiskLevel::High] {
        let mut pipeline = GuardrailPipeline::new();
        assert_eq!(pipeline.policy(), &SecurityPolicy::default());
        pipeline.add(FakeGuardrail::new(
            "detector",
            100,
            unsafe_result(level, "pii"),
        ));

        let outcome = pipeline.scan_user_input("x").await;
        assert_eq!(outcome.decision, SecurityDecision::Allow);
        assert!(outcome.is_safe());
        assert!(!outcome.is_blocked());
        assert_eq!(outcome.blocked_by, None);
        assert_eq!(outcome.risk_level, level);
        assert_eq!(outcome.errors.len(), 0);
        assert_eq!(outcome.findings.len(), 1);

        let finding = &outcome.findings[0];
        assert_eq!(finding.guardrail, "detector");
        assert_eq!(finding.risk_type, "pii");
        assert_eq!(finding.risk_level, level);
        assert!(!finding.blocking);
        assert_eq!(
            finding.message,
            format!("Security finding: pii (level={})", level.name())
        );
        assert_eq!(finding.message, finding_message("pii", level, false));
        assert_eq!(outcome.evaluations.len(), 1);
        assert!(outcome.enforce().is_ok());
    }
}

#[tokio::test]
async fn critical_findings_block_the_call() {
    let mut pipeline = GuardrailPipeline::new();
    pipeline.add(FakeGuardrail::new(
        "blocker",
        100,
        unsafe_result(RiskLevel::Critical, "jailbreak"),
    ));

    let outcome = pipeline.scan_user_input("x").await;
    assert_eq!(outcome.decision, SecurityDecision::Block);
    assert!(outcome.is_blocked());
    assert!(!outcome.is_safe());
    assert_eq!(outcome.blocked_by.as_deref(), Some("blocker"));
    assert_eq!(outcome.risk_level, RiskLevel::Critical);
    assert_eq!(outcome.findings.len(), 1);

    let finding = &outcome.findings[0];
    assert!(finding.blocking);
    assert_eq!(finding.risk_level, RiskLevel::Critical);
    assert_eq!(
        finding.message,
        "Critical security risk detected: jailbreak"
    );
    assert_eq!(
        finding.message,
        finding_message("jailbreak", RiskLevel::Critical, true)
    );
}

#[tokio::test]
async fn the_highest_finding_level_wins_across_guardrails() {
    let mut pipeline = GuardrailPipeline::new();
    pipeline.add(FakeGuardrail::new(
        "low",
        100,
        unsafe_result(RiskLevel::Low, "style"),
    ));
    pipeline.add(FakeGuardrail::new(
        "high",
        100,
        unsafe_result(RiskLevel::High, "pii"),
    ));

    let outcome = pipeline.scan_user_input("x").await;
    assert_eq!(outcome.findings.len(), 2);
    assert_eq!(outcome.risk_level, RiskLevel::High);
    assert!(outcome.is_safe());
}

#[tokio::test]
async fn stop_on_block_skips_the_guardrails_that_follow() {
    let calls = call_log();
    let mut pipeline = GuardrailPipeline::new();
    pipeline.add(
        FakeGuardrail::new(
            "blocker",
            100,
            unsafe_result(RiskLevel::Critical, "jailbreak"),
        )
        .recording_calls(calls.clone()),
    );
    pipeline.add(FakeGuardrail::new("after", 100, Outcome::Pass).recording_calls(calls.clone()));

    let outcome = pipeline.scan_user_input("x").await;
    assert_eq!(call_order(&calls), ["blocker"]);
    assert!(outcome.is_blocked());
    assert_eq!(outcome.blocked_by.as_deref(), Some("blocker"));
    assert_eq!(outcome.evaluations.len(), 1);
}

#[tokio::test]
async fn stop_on_block_false_runs_the_rest_and_keeps_the_first_blocker() {
    let calls = call_log();
    let mut pipeline = GuardrailPipeline::new().with_policy(SecurityPolicy {
        stop_on_block: false,
        ..Default::default()
    });
    pipeline.add(
        FakeGuardrail::new(
            "first_blocker",
            200,
            unsafe_result(RiskLevel::Critical, "first"),
        )
        .recording_calls(calls.clone()),
    );
    pipeline.add(
        FakeGuardrail::new(
            "second_blocker",
            100,
            unsafe_result(RiskLevel::Critical, "second"),
        )
        .recording_calls(calls.clone()),
    );
    pipeline.add(FakeGuardrail::new("passer", 50, Outcome::Pass).recording_calls(calls.clone()));

    let outcome = pipeline.scan_user_input("x").await;
    assert_eq!(
        call_order(&calls),
        ["first_blocker", "second_blocker", "passer"]
    );
    assert_eq!(outcome.decision, SecurityDecision::Block);
    assert_eq!(outcome.blocked_by.as_deref(), Some("first_blocker"));
    assert_eq!(outcome.findings.len(), 2);
    assert!(outcome.findings.iter().all(|finding| finding.blocking));
}

#[tokio::test]
async fn a_lowered_block_level_blocks_high_but_still_records_medium() {
    let policy = SecurityPolicy {
        block_level: RiskLevel::High,
        ..Default::default()
    };

    let mut high = GuardrailPipeline::new().with_policy(policy);
    high.add(FakeGuardrail::new(
        "g",
        100,
        unsafe_result(RiskLevel::High, "pii"),
    ));
    let blocked = high.scan_user_input("x").await;
    assert!(blocked.is_blocked());
    assert_eq!(blocked.blocked_by.as_deref(), Some("g"));
    assert_eq!(blocked.risk_level, RiskLevel::High);
    assert_eq!(blocked.findings.len(), 1);
    assert!(blocked.findings[0].blocking);
    assert_eq!(
        blocked.findings[0].message,
        "Guardrail blocked: pii risk detected"
    );
    assert_eq!(
        blocked.findings[0].message,
        finding_message("pii", RiskLevel::High, true)
    );

    let mut medium = GuardrailPipeline::new().with_policy(policy);
    medium.add(FakeGuardrail::new(
        "g",
        100,
        unsafe_result(RiskLevel::Medium, "spam"),
    ));
    let allowed = medium.scan_user_input("x").await;
    assert!(allowed.is_safe());
    assert_eq!(allowed.blocked_by, None);
    assert_eq!(allowed.findings.len(), 1);
    assert!(!allowed.findings[0].blocking);
    assert_eq!(
        allowed.findings[0].message,
        "Security finding: spam (level=MEDIUM)"
    );
}

#[tokio::test]
async fn fail_open_records_the_error_and_continues() {
    let calls = call_log();
    let mut pipeline = GuardrailPipeline::new();
    pipeline.add(
        FakeGuardrail::new("broken", 200, Outcome::Fail("boom".to_owned()))
            .recording_calls(calls.clone()),
    );
    pipeline.add(FakeGuardrail::new("healthy", 100, Outcome::Pass).recording_calls(calls.clone()));

    let outcome = pipeline.scan_user_input("x").await;
    // The contract phrases the entry as "<name>: boom"; `Error`'s Display keeps
    // its kind prefix, so the recorded entry is "<name>: security: boom".
    let expected = format!("broken: {}", Error::security("boom"));
    assert_eq!(expected, "broken: security: boom");

    assert_eq!(outcome.decision, SecurityDecision::Allow);
    assert!(outcome.is_safe());
    assert_eq!(outcome.risk_level, RiskLevel::Safe);
    assert_eq!(outcome.findings.len(), 0);
    assert_eq!(outcome.errors, vec![expected.clone()]);
    assert_eq!(outcome.evaluations.len(), 2);
    assert_eq!(outcome.evaluations[0].guardrail, "broken");
    assert_eq!(outcome.evaluations[0].result, None);
    assert_eq!(
        outcome.evaluations[0].error.as_deref(),
        Some(expected.as_str())
    );
    assert_eq!(call_order(&calls), ["broken", "healthy"]);
    assert!(outcome.enforce().is_ok());
}

#[tokio::test]
async fn fail_closed_turns_the_error_into_a_blocking_finding() {
    let mut pipeline = GuardrailPipeline::new().with_policy(SecurityPolicy {
        failure_policy: FailurePolicy::FailClosed,
        ..Default::default()
    });
    pipeline.add(FakeGuardrail::new(
        "broken",
        100,
        Outcome::Fail("boom".to_owned()),
    ));

    let outcome = pipeline.scan_user_input("x").await;
    assert_eq!(outcome.decision, SecurityDecision::Block);
    assert!(outcome.is_blocked());
    assert_eq!(outcome.blocked_by.as_deref(), Some("broken"));
    assert_eq!(outcome.risk_level, RiskLevel::Critical);
    assert_eq!(outcome.errors.len(), 1);
    assert_eq!(outcome.evaluations.len(), 1);
    assert_eq!(outcome.evaluations[0].result, None);
    assert!(outcome.evaluations[0].error.is_some());
    assert_eq!(outcome.findings.len(), 1);

    let finding = &outcome.findings[0];
    assert_eq!(finding.guardrail, "broken");
    assert_eq!(finding.risk_type, "guardrail_error");
    assert_eq!(finding.risk_level, RiskLevel::Critical);
    assert!(finding.blocking);
    assert_eq!(
        finding.message,
        "Critical security risk detected: guardrail_error"
    );

    let error = outcome.enforce().unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Security);
    assert_eq!(
        error.to_string(),
        "security: Critical security risk detected: guardrail_error"
    );
}

// ports tests/unit_tests/core/security/guardrail/test_guardrail_base.py
#[tokio::test]
async fn enforce_and_violation_describe_the_blocking_finding() {
    let mut pipeline = GuardrailPipeline::new();
    pipeline.add(FakeGuardrail::new(
        "secrets",
        100,
        unsafe_result(RiskLevel::Critical, "secret"),
    ));

    let outcome = pipeline.scan_user_input("x").await;
    let violation = outcome.violation().expect("a blocking violation");
    assert_eq!(violation.event, "user_input");
    assert_eq!(violation.risk_type, "secret");
    assert_eq!(violation.risk_level, RiskLevel::Critical);
    assert_eq!(violation.message, "Critical security risk detected: secret");
    assert_eq!(violation.details, Some(json!({"pattern": "secret"})));

    let error = outcome.enforce().unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Security);
    assert_eq!(
        error.to_string(),
        "security: Critical security risk detected: secret"
    );
}

#[tokio::test]
async fn modified_data_merges_in_evaluation_order() {
    let first = GuardrailResult {
        modified_data: Some(json!({"a": 1, "shared": "first"})),
        ..GuardrailResult::pass()
    };
    let second = GuardrailResult {
        modified_data: Some(json!({"b": 2, "shared": "second"})),
        ..GuardrailResult::pass()
    };

    let mut pipeline = GuardrailPipeline::new();
    pipeline.add(FakeGuardrail::new("first", 200, Outcome::Result(first)));
    pipeline.add(FakeGuardrail::new("second", 100, Outcome::Result(second)));

    let outcome = pipeline.scan_user_input("x").await;
    assert!(outcome.is_safe());
    assert_eq!(
        outcome.modified_data(),
        Some(json!({"a": 1, "b": 2, "shared": "second"}))
    );

    let mut untouched = GuardrailPipeline::new();
    untouched.add(FakeGuardrail::new("plain", 100, Outcome::Pass));
    assert_eq!(untouched.scan_user_input("x").await.modified_data(), None);
}

#[tokio::test]
async fn scan_helpers_build_the_documented_contexts() {
    let recorder = FakeGuardrail::new("recorder", 100, Outcome::Pass);
    let seen = recorder.seen.clone();
    let mut pipeline = GuardrailPipeline::new();
    pipeline.add(recorder);

    pipeline.scan_user_input("hi").await;
    let ctx = last_context(&seen);
    assert_eq!(ctx.stage(), ScanStage::UserInput);
    assert_eq!(ctx.event, "user_input");
    assert_eq!(ctx.get_text(), Some("hi"));

    pipeline
        .scan_prompt(&[BaseMessage::system("s"), BaseMessage::user("u")])
        .await;
    let ctx = last_context(&seen);
    assert_eq!(ctx.stage(), ScanStage::LlmInvokeInput);
    assert_eq!(ctx.event, "_framework:llm_invoke_input");
    assert_eq!(ctx.get_messages().map(<[Value]>::len), Some(2));
    assert_eq!(ctx.metadata.get("message_count"), Some(&json!(2)));

    pipeline.scan_model_output("out").await;
    let ctx = last_context(&seen);
    assert_eq!(ctx.stage(), ScanStage::LlmInvokeOutput);
    assert_eq!(ctx.event, "_framework:llm_invoke_output");

    pipeline.scan_tool_input("search", &json!({"q": "x"})).await;
    let ctx = last_context(&seen);
    assert_eq!(ctx.stage(), ScanStage::ToolInvokeInput);
    assert_eq!(ctx.event, "_framework:tool_invoke_input");
    assert_eq!(ctx.get_tool_name(), Some("search"));
    assert_eq!(ctx.content, json!({"q": "x"}));

    pipeline.scan_tool_output("search", &json!({"r": 1})).await;
    let ctx = last_context(&seen);
    assert_eq!(ctx.stage(), ScanStage::ToolInvokeOutput);
    assert_eq!(ctx.event, "_framework:tool_invoke_output");
    assert_eq!(ctx.get_tool_name(), Some("search"));

    assert_eq!(seen.lock().unwrap().len(), 5);
}

#[tokio::test]
async fn an_empty_pipeline_allows_the_scan() {
    let pipeline = GuardrailPipeline::new();
    let outcome = pipeline.scan_user_input("x").await;
    assert!(outcome.is_safe());
    assert_eq!(outcome.decision, SecurityDecision::Allow);
    assert_eq!(outcome.stage, ScanStage::UserInput);
    assert_eq!(outcome.risk_level, RiskLevel::Safe);
    assert_eq!(outcome.findings.len(), 0);
    assert_eq!(outcome.errors.len(), 0);
    assert_eq!(outcome.evaluations.len(), 0);
    assert_eq!(outcome.blocked_by, None);
    assert_eq!(pipeline.guardrails().len(), 0);

    let policy = pipeline.policy();
    assert_eq!(policy, &SecurityPolicy::default());
    assert_eq!(policy.block_level, RiskLevel::Critical);
    assert_eq!(policy.failure_policy, FailurePolicy::FailOpen);
    assert!(policy.stop_on_block);

    let mut registered = GuardrailPipeline::new();
    registered.add(FakeGuardrail::new("added", 100, Outcome::Pass));
    assert_eq!(registered.guardrails().len(), 1);
    assert_eq!(registered.guardrails()[0].name(), "added");

    let shared: Arc<dyn Guardrail> = Arc::new(FakeGuardrail::new("shared", 100, Outcome::Pass));
    registered.add_arc(shared);
    assert_eq!(registered.guardrails().len(), 2);
    assert_eq!(registered.guardrails()[1].name(), "shared");

    let extras: Vec<Arc<dyn Guardrail>> = vec![
        Arc::new(FakeGuardrail::new("extra1", 100, Outcome::Pass)),
        Arc::new(FakeGuardrail::new("extra2", 100, Outcome::Pass)),
    ];
    registered.extend(extras);
    assert_eq!(registered.guardrails().len(), 4);
    assert_eq!(registered.guardrails()[2].name(), "extra1");
    assert_eq!(registered.guardrails()[3].name(), "extra2");
}

#[tokio::test]
async fn redacted_outcome_summarises_a_blocked_run() {
    let mut pipeline = GuardrailPipeline::new();
    pipeline.add(FakeGuardrail::new("passer", 200, Outcome::Pass));
    pipeline.add(FakeGuardrail::new(
        "blocker",
        100,
        unsafe_result(RiskLevel::Critical, "jailbreak"),
    ));

    let outcome = pipeline.scan_user_input("x").await;
    let redacted: RedactedOutcome = outcome.redacted();
    assert_eq!(redacted.decision, SecurityDecision::Block);
    assert_eq!(redacted.risk_level, RiskLevel::Critical);
    assert_eq!(redacted.stage, ScanStage::UserInput);
    assert_eq!(redacted.guardrails, ["passer", "blocker"]);
    assert_eq!(redacted.findings.len(), 1);
    assert_eq!(redacted.errors.len(), 0);

    let summary = redacted.to_string();
    assert!(summary.contains("decision=block"), "{summary}");
    assert!(summary.contains("risk_level=CRITICAL"), "{summary}");
}
