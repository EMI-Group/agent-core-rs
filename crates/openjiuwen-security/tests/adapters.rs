#![allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]

//! Behavior suite for the vendor moderation adapters and the generic `HTTP`
//! moderation transport, driven end to end through the public pipeline.
//!
//! Every adapter is pointed at a loopback `wiremock` server, so the suite
//! pins the contract of
//! `openjiuwen_security::backends::vendors::{openai, azure, aliyun}` and of
//! `ApiModelBackend` + `HttpModerationTransport` without network access:
//! benign, blocking, sub-threshold, transport-failure, stage-binding, and
//! content-leak behavior, plus the recorded wire request of the generic
//! transport.
//!
//! Everything runs offline and deterministically: fixed replies, retries
//! disabled, and no wall-clock assertions.

use std::sync::Arc;

use openjiuwen_common::error::ErrorKind;
use openjiuwen_common::security::SecretString;
use openjiuwen_security::backends::http::{
    HttpAuth, HttpModerationConfig, HttpModerationTransport,
};
use openjiuwen_security::backends::vendors::aliyun::{
    AliyunGreenBackend, AliyunGreenConfig, NAME as ALIYUN,
};
use openjiuwen_security::backends::vendors::azure::{
    AzureContentSafetyBackend, AzureContentSafetyConfig, NAME as AZURE,
};
use openjiuwen_security::backends::vendors::openai::{
    OpenAiModerationBackend, OpenAiModerationConfig, NAME as OPENAI,
};
use openjiuwen_security::{
    ApiModelBackend, ApiModelBackendConfig, BackendGuardrail, FailurePolicy, GuardrailBackend,
    GuardrailContext, GuardrailOutcome, GuardrailPipeline, RiskLevel, ScanStage, SecurityDecision,
    SecurityPolicy,
};
use serde_json::{json, Value};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// Text sent to every adapter; no serialized outcome may ever repeat it.
const SCANNED: &str = "SENSITIVE-PROMPT-42 please grade this text";

/// Vendor matched-word fragment planted in every risky fixture: adapters drop
/// such fields, so it must never reach a finding or a serialized outcome.
const VENDOR_WORD: &str = "leakedvendorword";

/// Guardrail name of the generic `ApiModelBackend` adapter.
const API_MODEL: &str = "api_model";

// --- pipelines -------------------------------------------------------------

/// A pipeline holding one backend guardrail, bound to every stage.
fn pipeline_with(backend: Arc<dyn GuardrailBackend>) -> GuardrailPipeline {
    let name = backend.name().to_owned();
    let mut pipeline = GuardrailPipeline::new();
    pipeline.add(BackendGuardrail::new(name, backend));
    pipeline
}

/// A pipeline whose one guardrail runs only at `stage`.
fn only_stage(backend: Arc<dyn GuardrailBackend>, stage: ScanStage) -> GuardrailPipeline {
    let name = backend.name().to_owned();
    let mut pipeline = GuardrailPipeline::new();
    pipeline.add(BackendGuardrail::new(name, backend).with_stages(vec![stage]));
    pipeline
}

/// The same pipeline with the explicit `FailOpen` opt-in.
///
/// The default policy already fails closed, so a failure path is paired: one
/// pipeline built with `FailOpen` records the failure and allows the call, the
/// default pipeline blocks it.
fn fail_open(backend: Arc<dyn GuardrailBackend>) -> GuardrailPipeline {
    pipeline_with(backend).with_policy(SecurityPolicy {
        failure_policy: FailurePolicy::FailOpen,
        ..SecurityPolicy::default()
    })
}

/// The same pipeline with a lowered blocking threshold.
fn blocking_at(backend: Arc<dyn GuardrailBackend>, level: RiskLevel) -> GuardrailPipeline {
    pipeline_with(backend).with_policy(SecurityPolicy {
        block_level: level,
        ..SecurityPolicy::default()
    })
}

/// A plain-text context at `stage`.
fn ctx(text: &str, stage: ScanStage) -> GuardrailContext {
    GuardrailContext::text(text, stage)
}

// --- loopback servers ------------------------------------------------------

/// A server answering `calls` `POST`s to `path_match` with the JSON `body`.
async fn server_with(path_match: &str, body: Value, calls: u64) -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(path_match))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .expect(calls)
        .mount(&server)
        .await;
    server
}

/// A server failing `calls` `POST`s to `path_match` with `HTTP 500`.
async fn failing_server(path_match: &str, calls: u64) -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(path_match))
        .respond_with(ResponseTemplate::new(500).set_body_string("upstream unavailable"))
        .expect(calls)
        .mount(&server)
        .await;
    server
}

/// A server answering `calls` `POST`s to `path_match` with a raw `HTTP 200`
/// body, for replies that are not JSON at all.
async fn server_with_raw(path_match: &str, body: &str, calls: u64) -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(path_match))
        .respond_with(ResponseTemplate::new(200).set_body_string(body))
        .expect(calls)
        .mount(&server)
        .await;
    server
}

// --- adapters under test ---------------------------------------------------

/// The `OpenAI` adapter, retries disabled, pointed at `server`.
fn openai_backend(server: &MockServer) -> Arc<dyn GuardrailBackend> {
    Arc::new(
        OpenAiModerationBackend::new(OpenAiModerationConfig {
            api_url: format!("{}/v1/moderations", server.uri()),
            api_key: Some(SecretString::from("sk-openai-test")),
            max_retries: 0,
            ..OpenAiModerationConfig::default()
        })
        .unwrap(),
    )
}

/// The Azure adapter, retries disabled, pointed at `server`.
fn azure_backend(server: &MockServer) -> Arc<dyn GuardrailBackend> {
    Arc::new(
        AzureContentSafetyBackend::new(AzureContentSafetyConfig {
            endpoint: server.uri(),
            api_key: Some(SecretString::from("azure-test-key")),
            max_retries: 0,
            ..AzureContentSafetyConfig::default()
        })
        .unwrap(),
    )
}

/// The Aliyun adapter, retries disabled, pointed at `server`'s root.
fn aliyun_backend(server: &MockServer) -> Arc<dyn GuardrailBackend> {
    Arc::new(
        AliyunGreenBackend::new(AliyunGreenConfig {
            endpoint: server.uri(),
            max_retries: 0,
            ..AliyunGreenConfig::default()
        })
        .unwrap(),
    )
}

/// The generic `ApiModelBackend` speaking `{"text": ...}` through
/// `HttpModerationTransport`, pointed at `server`'s `/detect` endpoint.
fn api_model_backend(server: &MockServer) -> ApiModelBackend {
    let url = format!("{}/detect", server.uri());
    let transport = HttpModerationTransport::new(
        HttpModerationConfig::default()
            .with_url(url.clone())
            .with_header("x-tenant", "acme")
            .with_auth(HttpAuth::Bearer(SecretString::from("moderation-token")))
            .with_max_retries(0),
    )
    .unwrap();
    ApiModelBackend::new(ApiModelBackendConfig {
        api_url: url,
        ..ApiModelBackendConfig::default()
    })
    .with_transport(Arc::new(transport))
}

// --- fixtures --------------------------------------------------------------

/// A flagged `omni-moderation` reply whose top `category_scores` value is
/// `score`; `score >= 0.90` reports `CRITICAL`.
fn openai_reply(score: f64) -> Value {
    json!({
        "id": "modr-test-1",
        "model": "omni-moderation-latest",
        "results": [{
            "flagged": true,
            "categories": {"hate": true, "violence": false},
            "category_scores": {"hate": score, "violence": 0.01},
            "matched_words": [VENDOR_WORD],
        }],
    })
}

/// A benign `omni-moderation` reply.
fn openai_benign() -> Value {
    json!({
        "id": "modr-benign-1",
        "model": "omni-moderation-latest",
        "results": [{
            "flagged": false,
            "categories": {"hate": false},
            "category_scores": {"hate": 0.0001},
        }],
    })
}

/// An Azure `text:analyze` reply reporting `Hate` at `severity`
/// (`>= 7` is `CRITICAL`).
fn azure_reply(severity: i64) -> Value {
    json!({
        "blocklistsMatch": [{
            "blocklistName": "default",
            "blocklistItemId": "1",
            "blocklistItemText": VENDOR_WORD,
        }],
        "categoriesAnalysis": [
            {"category": "Hate", "severity": severity},
            {"category": "SelfHarm", "severity": 0},
        ],
    })
}

/// An Aliyun `TextModeration` reply labelling the text `political_content` at
/// `confidence` (`>= 90` is `CRITICAL`).
fn aliyun_reply(confidence: f64) -> Value {
    json!({
        "Code": 200,
        "Message": "OK",
        "RequestId": "req-aliyun-test",
        "Data": {
            "Labels": "political_content",
            "Reason": VENDOR_WORD,
            "Result": [{
                "Label": "political_content",
                "Confidence": confidence,
                "RiskWords": VENDOR_WORD,
            }],
        },
    })
}

/// A benign Aliyun reply: only the vendor's `nonLabel` marker.
fn aliyun_benign() -> Value {
    json!({
        "Code": 200,
        "Message": "OK",
        "RequestId": "req-aliyun-benign",
        "Data": {"Labels": "nonLabel", "Result": []},
    })
}

/// A `BertBinaryParser` attack reply at `confidence`
/// (`>= 0.95` is `HIGH`, `0.85..0.95` is `MEDIUM`).
fn bert_reply(confidence: f64) -> Value {
    json!({
        "predicted_class": 1,
        "confidence": confidence,
        "matched_words": [VENDOR_WORD],
    })
}

// --- assertions ------------------------------------------------------------

/// Asserts an allowed outcome: one safe evaluation, no findings, no failures.
fn assert_allowed(outcome: &GuardrailOutcome, guardrail: &str) {
    assert_eq!(outcome.decision, SecurityDecision::Allow);
    assert!(!outcome.is_blocked());
    assert!(outcome.findings.is_empty());
    assert!(outcome.errors.is_empty());
    assert_eq!(outcome.risk_level, RiskLevel::Safe);
    assert_eq!(outcome.evaluations.len(), 1);
    assert_eq!(outcome.evaluations[0].guardrail, guardrail);
    assert!(outcome.enforce().is_ok());
}

/// Asserts a recorded but non-blocking finding at `level` under the default
/// `CRITICAL` block level.
fn assert_finding(outcome: &GuardrailOutcome, guardrail: &str, risk_type: &str, level: RiskLevel) {
    assert_eq!(outcome.decision, SecurityDecision::Allow);
    assert_eq!(outcome.risk_level, level);
    assert_eq!(outcome.findings.len(), 1);
    let finding = &outcome.findings[0];
    assert_eq!(finding.guardrail, guardrail);
    assert_eq!(finding.risk_type, risk_type);
    assert_eq!(finding.risk_level, level);
    assert!(!finding.blocking);
    assert_eq!(
        finding.message,
        format!("Security finding: {risk_type} (level={})", level.name())
    );
    assert!(
        outcome.enforce().is_ok(),
        "a sub-threshold finding does not block"
    );
}

/// Asserts a blocking outcome with one finding and the Python-parity message
/// for its level.
fn assert_blocked(outcome: &GuardrailOutcome, guardrail: &str, risk_type: &str, level: RiskLevel) {
    assert_eq!(outcome.decision, SecurityDecision::Block);
    assert!(outcome.is_blocked());
    assert_eq!(outcome.risk_level, level);
    assert_eq!(outcome.findings.len(), 1);
    let finding = &outcome.findings[0];
    assert_eq!(finding.guardrail, guardrail);
    assert_eq!(finding.risk_type, risk_type);
    assert_eq!(finding.risk_level, level);
    assert!(finding.blocking);
    assert_eq!(outcome.blocked_by.as_deref(), Some(guardrail));
    let expected = if level == RiskLevel::Critical {
        format!("Critical security risk detected: {risk_type}")
    } else {
        format!("Guardrail blocked: {risk_type} risk detected")
    };
    assert_eq!(finding.message, expected);
    let error = outcome
        .enforce()
        .expect_err("a blocked outcome must enforce");
    assert_eq!(error.kind(), ErrorKind::Security);
    assert_eq!(error.message(), expected.as_str());
}

/// Asserts the fail-open handling of a backend failure: recorded, allowed.
fn assert_fail_open(outcome: &GuardrailOutcome, guardrail: &str) {
    assert_eq!(outcome.decision, SecurityDecision::Allow);
    assert!(outcome.findings.is_empty());
    assert_eq!(outcome.evaluations.len(), 1);
    assert!(
        outcome.evaluations[0].result.is_none(),
        "a failure is not a verdict"
    );
    assert_eq!(outcome.errors.len(), 1);
    assert!(
        outcome.errors[0].starts_with(&format!("{guardrail}: ")),
        "{:?}",
        outcome.errors
    );
    assert!(outcome.errors[0].contains("500"), "{:?}", outcome.errors);
    assert!(
        outcome.enforce().is_ok(),
        "fail-open keeps the guarded call allowed"
    );
}

/// Asserts the fail-closed handling of a backend failure: a blocking
/// `guardrail_error` finding at `CRITICAL` whose message and details carry
/// `reason`.
///
/// The failure finding is *not* a detection: the pipeline builds it with the
/// `Guardrail blocked: guardrail_error risk detected (<reason>)` wording, so a
/// failure can never be mistaken for a graded verdict.
fn assert_fail_closed(outcome: &GuardrailOutcome, guardrail: &str, reason: &str) {
    assert!(!outcome.errors.is_empty(), "the failure is still recorded");
    assert_eq!(outcome.decision, SecurityDecision::Block);
    assert!(outcome.is_blocked());
    assert_eq!(outcome.risk_level, RiskLevel::Critical);
    assert_eq!(outcome.blocked_by.as_deref(), Some(guardrail));
    assert_eq!(outcome.findings.len(), 1);
    let finding = &outcome.findings[0];
    assert_eq!(finding.guardrail, guardrail);
    assert_eq!(finding.risk_type, "guardrail_error");
    assert_eq!(finding.risk_level, RiskLevel::Critical);
    assert!(finding.blocking);
    let expected = "Guardrail blocked: guardrail_error risk detected";
    assert!(finding.message.starts_with(expected), "{}", finding.message);
    assert!(
        finding.message.contains(reason),
        "the message carries the reason: {}",
        finding.message
    );
    let details = finding.details.as_ref().expect("a failure carries details");
    assert!(
        details["error"].as_str().unwrap().contains(reason),
        "{details}"
    );
    assert_eq!(outcome.evaluations.len(), 1);
    assert!(
        outcome.evaluations[0].result.is_none(),
        "a failure is not a verdict"
    );
    assert!(
        outcome.evaluations[0]
            .error
            .as_deref()
            .is_some_and(|error| error.contains(reason)),
        "{:?}",
        outcome.evaluations
    );
    assert!(
        outcome.errors.iter().any(|error| error.contains(reason)),
        "{:?}",
        outcome.errors
    );
    let error = outcome
        .enforce()
        .expect_err("a failure under fail-closed must enforce");
    assert_eq!(error.kind(), ErrorKind::Security);
    assert_eq!(error.message(), finding.message.as_str());
}

/// Asserts the serialized outcome carries neither the scanned text nor a
/// vendor matched-word fragment.
fn assert_no_leak(outcome: &GuardrailOutcome) {
    let serialized = serde_json::to_string(outcome).unwrap();
    assert!(!serialized.contains(SCANNED), "{serialized}");
    assert!(!serialized.contains(VENDOR_WORD), "{serialized}");
}

/// One recorded request header, when the server saw it as visible ASCII.
fn header<'a>(request: &'a wiremock::Request, name: &str) -> Option<&'a str> {
    request.headers.get(name)?.to_str().ok()
}

/// The JSON body of one recorded request.
fn body_json(request: &wiremock::Request) -> Value {
    serde_json::from_slice(&request.body).unwrap()
}

// --- OpenAI moderation -----------------------------------------------------

#[tokio::test]
async fn openai_benign_reply_allows() {
    let server = server_with("/v1/moderations", openai_benign(), 1).await;
    let outcome = pipeline_with(openai_backend(&server))
        .scan_user_input(SCANNED)
        .await;
    assert_allowed(&outcome, OPENAI);
}

#[tokio::test]
async fn openai_critical_reply_blocks() {
    let server = server_with("/v1/moderations", openai_reply(0.95), 1).await;
    let outcome = pipeline_with(openai_backend(&server))
        .scan_user_input(SCANNED)
        .await;
    assert_blocked(&outcome, OPENAI, "hate", RiskLevel::Critical);
}

#[tokio::test]
async fn openai_medium_reply_blocks_only_when_the_policy_lowers_the_bar() {
    let server = server_with("/v1/moderations", openai_reply(0.55), 2).await;
    let backend = openai_backend(&server);
    let recorded = pipeline_with(Arc::clone(&backend))
        .scan_user_input(SCANNED)
        .await;
    assert_finding(&recorded, OPENAI, "hate", RiskLevel::Medium);
    let blocked = blocking_at(backend, RiskLevel::Medium)
        .scan_user_input(SCANNED)
        .await;
    assert_blocked(&blocked, OPENAI, "hate", RiskLevel::Medium);
}

#[tokio::test]
async fn openai_transport_failure_follows_the_failure_policy() {
    let server = failing_server("/v1/moderations", 2).await;
    let backend = openai_backend(&server);
    // `FailOpen` is the explicit opt-in: the failure is recorded only.
    let open = fail_open(Arc::clone(&backend))
        .scan_user_input(SCANNED)
        .await;
    assert_fail_open(&open, OPENAI);
    // The default policy fails closed: the same failure blocks the call.
    let closed = pipeline_with(backend).scan_user_input(SCANNED).await;
    assert_fail_closed(&closed, OPENAI, "500");
}

#[tokio::test]
async fn openai_stage_binding_skips_other_stages() {
    let server = server_with("/v1/moderations", openai_reply(0.95), 1).await;
    let pipeline = only_stage(openai_backend(&server), ScanStage::ToolInvokeInput);
    let skipped = pipeline.run(&ctx(SCANNED, ScanStage::LlmInvokeInput)).await;
    assert_eq!(skipped.decision, SecurityDecision::Allow);
    assert!(
        skipped.evaluations.is_empty(),
        "the stage filter excludes the guardrail entirely"
    );
    assert!(skipped.findings.is_empty());
    let arguments = json!({"query": SCANNED});
    let scanned = pipeline.scan_tool_input("lookup", &arguments).await;
    assert_blocked(&scanned, OPENAI, "hate", RiskLevel::Critical);
    // Exactly one request: the excluded stage never reached the adapter.
    assert_eq!(server.received_requests().await.unwrap().len(), 1);
}

#[tokio::test]
async fn openai_outcome_never_carries_the_scanned_text() {
    let server = server_with("/v1/moderations", openai_reply(0.95), 1).await;
    let outcome = pipeline_with(openai_backend(&server))
        .scan_user_input(SCANNED)
        .await;
    assert_blocked(&outcome, OPENAI, "hate", RiskLevel::Critical);
    assert_no_leak(&outcome);
}

// --- Azure content safety --------------------------------------------------

#[tokio::test]
async fn azure_benign_reply_allows() {
    let server = server_with("/contentsafety/text:analyze", azure_reply(0), 1).await;
    let outcome = pipeline_with(azure_backend(&server))
        .scan_user_input(SCANNED)
        .await;
    assert_allowed(&outcome, AZURE);
}

#[tokio::test]
async fn azure_critical_reply_blocks() {
    let server = server_with("/contentsafety/text:analyze", azure_reply(7), 1).await;
    let outcome = pipeline_with(azure_backend(&server))
        .scan_user_input(SCANNED)
        .await;
    assert_blocked(&outcome, AZURE, "Hate", RiskLevel::Critical);
}

#[tokio::test]
async fn azure_medium_reply_blocks_only_when_the_policy_lowers_the_bar() {
    let server = server_with("/contentsafety/text:analyze", azure_reply(4), 2).await;
    let backend = azure_backend(&server);
    let recorded = pipeline_with(Arc::clone(&backend))
        .scan_user_input(SCANNED)
        .await;
    assert_finding(&recorded, AZURE, "Hate", RiskLevel::Medium);
    let blocked = blocking_at(backend, RiskLevel::Medium)
        .scan_user_input(SCANNED)
        .await;
    assert_blocked(&blocked, AZURE, "Hate", RiskLevel::Medium);
}

#[tokio::test]
async fn azure_transport_failure_follows_the_failure_policy() {
    let server = failing_server("/contentsafety/text:analyze", 2).await;
    let backend = azure_backend(&server);
    // `FailOpen` is the explicit opt-in: the failure is recorded only.
    let open = fail_open(Arc::clone(&backend))
        .scan_user_input(SCANNED)
        .await;
    assert_fail_open(&open, AZURE);
    // The default policy fails closed: the same failure blocks the call.
    let closed = pipeline_with(backend).scan_user_input(SCANNED).await;
    assert_fail_closed(&closed, AZURE, "500");
}

#[tokio::test]
async fn azure_stage_binding_skips_other_stages() {
    let server = server_with("/contentsafety/text:analyze", azure_reply(7), 1).await;
    let pipeline = only_stage(azure_backend(&server), ScanStage::ToolInvokeInput);
    let skipped = pipeline.run(&ctx(SCANNED, ScanStage::LlmInvokeInput)).await;
    assert_eq!(skipped.decision, SecurityDecision::Allow);
    assert!(skipped.evaluations.is_empty());
    assert!(skipped.findings.is_empty());
    let arguments = json!({"query": SCANNED});
    let scanned = pipeline.scan_tool_input("lookup", &arguments).await;
    assert_blocked(&scanned, AZURE, "Hate", RiskLevel::Critical);
    assert_eq!(server.received_requests().await.unwrap().len(), 1);
}

#[tokio::test]
async fn azure_outcome_never_carries_the_scanned_text() {
    let server = server_with("/contentsafety/text:analyze", azure_reply(7), 1).await;
    let outcome = pipeline_with(azure_backend(&server))
        .scan_user_input(SCANNED)
        .await;
    assert_blocked(&outcome, AZURE, "Hate", RiskLevel::Critical);
    assert_no_leak(&outcome);
}

// --- Aliyun Green ----------------------------------------------------------

#[tokio::test]
async fn aliyun_benign_reply_allows() {
    let server = server_with("/", aliyun_benign(), 1).await;
    let outcome = pipeline_with(aliyun_backend(&server))
        .scan_user_input(SCANNED)
        .await;
    assert_allowed(&outcome, ALIYUN);
}

#[tokio::test]
async fn aliyun_critical_reply_blocks() {
    let server = server_with("/", aliyun_reply(97.0), 1).await;
    let outcome = pipeline_with(aliyun_backend(&server))
        .scan_user_input(SCANNED)
        .await;
    assert_blocked(&outcome, ALIYUN, "political_content", RiskLevel::Critical);
}

#[tokio::test]
async fn aliyun_medium_reply_blocks_only_when_the_policy_lowers_the_bar() {
    let server = server_with("/", aliyun_reply(55.0), 2).await;
    let backend = aliyun_backend(&server);
    let recorded = pipeline_with(Arc::clone(&backend))
        .scan_user_input(SCANNED)
        .await;
    assert_finding(&recorded, ALIYUN, "political_content", RiskLevel::Medium);
    let blocked = blocking_at(backend, RiskLevel::Medium)
        .scan_user_input(SCANNED)
        .await;
    assert_blocked(&blocked, ALIYUN, "political_content", RiskLevel::Medium);
}

#[tokio::test]
async fn aliyun_transport_failure_follows_the_failure_policy() {
    let server = failing_server("/", 2).await;
    let backend = aliyun_backend(&server);
    // `FailOpen` is the explicit opt-in: the failure is recorded only.
    let open = fail_open(Arc::clone(&backend))
        .scan_user_input(SCANNED)
        .await;
    assert_fail_open(&open, ALIYUN);
    // The default policy fails closed: the same failure blocks the call.
    let closed = pipeline_with(backend).scan_user_input(SCANNED).await;
    assert_fail_closed(&closed, ALIYUN, "500");
}

#[tokio::test]
async fn aliyun_stage_binding_skips_other_stages() {
    let server = server_with("/", aliyun_reply(97.0), 1).await;
    let pipeline = only_stage(aliyun_backend(&server), ScanStage::ToolInvokeInput);
    let skipped = pipeline.run(&ctx(SCANNED, ScanStage::LlmInvokeInput)).await;
    assert_eq!(skipped.decision, SecurityDecision::Allow);
    assert!(skipped.evaluations.is_empty());
    assert!(skipped.findings.is_empty());
    let arguments = json!({"query": SCANNED});
    let scanned = pipeline.scan_tool_input("lookup", &arguments).await;
    assert_blocked(&scanned, ALIYUN, "political_content", RiskLevel::Critical);
    assert_eq!(server.received_requests().await.unwrap().len(), 1);
}

#[tokio::test]
async fn aliyun_outcome_never_carries_the_scanned_text() {
    let server = server_with("/", aliyun_reply(97.0), 1).await;
    let outcome = pipeline_with(aliyun_backend(&server))
        .scan_user_input(SCANNED)
        .await;
    assert_blocked(&outcome, ALIYUN, "political_content", RiskLevel::Critical);
    assert_no_leak(&outcome);
}

// --- generic HTTP moderation transport -------------------------------------

#[tokio::test]
async fn generic_transport_reaches_the_wire_and_produces_a_finding() {
    let server = server_with("/detect", bert_reply(0.96), 1).await;
    let outcome = pipeline_with(Arc::new(api_model_backend(&server)))
        .scan_user_input(SCANNED)
        .await;
    assert_finding(&outcome, API_MODEL, "attack_detected", RiskLevel::High);

    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests.len(), 1);
    let sent = &requests[0];
    assert_eq!(sent.method.as_str(), "POST");
    assert_eq!(sent.url.path(), "/detect");
    assert_eq!(
        header(sent, "authorization"),
        Some("Bearer moderation-token")
    );
    assert_eq!(header(sent, "x-tenant"), Some("acme"));
    assert_eq!(body_json(sent), json!({"text": SCANNED}));
    assert_no_leak(&outcome);
}

#[tokio::test]
async fn generic_medium_reply_blocks_only_when_the_policy_lowers_the_bar() {
    let server = server_with("/detect", bert_reply(0.90), 2).await;
    let recorded = pipeline_with(Arc::new(api_model_backend(&server)))
        .scan_user_input(SCANNED)
        .await;
    assert_finding(&recorded, API_MODEL, "attack_detected", RiskLevel::Medium);
    let blocked = blocking_at(Arc::new(api_model_backend(&server)), RiskLevel::Medium)
        .scan_user_input(SCANNED)
        .await;
    assert_blocked(&blocked, API_MODEL, "attack_detected", RiskLevel::Medium);
}

#[tokio::test]
async fn generic_transport_failure_follows_the_failure_policy() {
    let server = failing_server("/detect", 2).await;
    let backend: Arc<dyn GuardrailBackend> = Arc::new(api_model_backend(&server));
    let open = fail_open(Arc::clone(&backend))
        .scan_user_input(SCANNED)
        .await;
    assert_fail_open(&open, API_MODEL);
    let closed = pipeline_with(backend).scan_user_input(SCANNED).await;
    assert_fail_closed(&closed, API_MODEL, "500");
}

// --- unrecognized 2xx replies ----------------------------------------------

/// Asserts an unrecognized `2xx` reply blocks as a `guardrail_error` failure
/// whose recorded reason contains `reason`, without echoing the payload.
async fn assert_unrecognized_reply_blocks(backend: Arc<dyn GuardrailBackend>, reason: &str) {
    let name = backend.name().to_owned();
    let outcome = pipeline_with(backend).scan_user_input(SCANNED).await;
    assert_eq!(outcome.errors.len(), 1, "one failure per scan");
    assert_fail_closed(&outcome, &name, reason);
    assert_no_leak(&outcome);
}

// No Python equivalent: Python's vendor adapters graded an unparsable reply as
// safe. A `2xx` reply outside the documented schema is a validation failure, so
// the default fail-closed policy blocks the scan.
#[tokio::test]
async fn openai_unrecognized_reply_blocks_instead_of_reading_as_safe() {
    let server = server_with("/v1/moderations", json!({}), 1).await;
    assert_unrecognized_reply_blocks(openai_backend(&server), "openai moderation reply").await;
}

#[tokio::test]
async fn azure_unrecognized_reply_blocks_instead_of_reading_as_safe() {
    let server = server_with("/contentsafety/text:analyze", json!({}), 1).await;
    assert_unrecognized_reply_blocks(azure_backend(&server), "azure content safety reply").await;
}

#[tokio::test]
async fn aliyun_unrecognized_reply_blocks_instead_of_reading_as_safe() {
    let server = server_with("/", json!({}), 1).await;
    assert_unrecognized_reply_blocks(aliyun_backend(&server), "aliyun green reply").await;
}

#[tokio::test]
async fn generic_unrecognized_json_reply_blocks_instead_of_reading_as_safe() {
    let server = server_with("/detect", json!({}), 1).await;
    let backend: Arc<dyn GuardrailBackend> = Arc::new(api_model_backend(&server));
    assert_unrecognized_reply_blocks(backend, "bert_binary parser: unrecognized model output")
        .await;
    assert_eq!(
        server.received_requests().await.unwrap().len(),
        1,
        "a reply that fails to parse is not retried"
    );
}

#[tokio::test]
async fn generic_non_json_reply_blocks_instead_of_reading_as_safe() {
    let server = server_with_raw("/detect", "garbage", 1).await;
    let backend: Arc<dyn GuardrailBackend> = Arc::new(api_model_backend(&server));
    let name = backend.name().to_owned();
    let outcome = pipeline_with(backend).scan_user_input(SCANNED).await;
    assert_fail_closed(&outcome, &name, "non-JSON body");
    assert_no_leak(&outcome);
    assert!(
        !serde_json::to_string(&outcome).unwrap().contains("garbage"),
        "the payload of a rejected reply is never recorded"
    );
    assert_eq!(
        server.received_requests().await.unwrap().len(),
        1,
        "a bad 2xx body is never retried"
    );
}

#[tokio::test]
async fn generic_json_non_object_reply_blocks_instead_of_reading_as_safe() {
    let server = server_with("/detect", json!("garbage"), 1).await;
    let backend: Arc<dyn GuardrailBackend> = Arc::new(api_model_backend(&server));
    let name = backend.name().to_owned();
    let outcome = pipeline_with(backend).scan_user_input(SCANNED).await;
    assert_fail_closed(&outcome, &name, "JSON string body");
    assert_no_leak(&outcome);
    assert!(
        !serde_json::to_string(&outcome).unwrap().contains("garbage"),
        "the payload of a rejected reply is never recorded"
    );
    assert_eq!(
        server.received_requests().await.unwrap().len(),
        1,
        "a bad 2xx body is never retried"
    );
}
