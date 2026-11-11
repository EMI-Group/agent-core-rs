#![allow(clippy::unwrap_used, clippy::expect_used)]

//! Behavior suite for the tool-argument guardrail.
//!
//! **The detector itself is an addition with no Python equivalent.** The Python
//! package (`openjiuwen/core/security/guardrail/`, v0.1.20) ships prompt
//! injection only: there is no tool-argument validator and no
//! `tests/unit_tests/core/security/guardrail/test_tool_arguments.py` to port.
//! What *is* a port is the two-tier blocking policy of
//! `tests/unit_tests/core/security/guardrail/test_guardrail_base.py`: only
//! `CRITICAL` aborts the caller, and every lower severity becomes a recorded
//! finding while execution continues. Each denial path below pins that contract,
//! and the detector's own semantics are pinned as an addition:
//!
//! - checks run in order and the first failure wins — denied tool, tool outside
//!   a non-empty allowlist, oversized arguments, a deny rule, sensitive data;
//! - `details` never carry the matched argument text, only the rule name, its
//!   regex source, the tool name, and counts;
//! - sanitized arguments are exposed only through `modified_data`.
//!
//! Everything runs offline against the built-in deny table and PII table: no
//! backend, network, model runtime, or wall clock is involved.

use openjiuwen_common::error::ErrorKind;
use openjiuwen_security::{
    ContentRule, Finding, Guardrail, GuardrailContext, GuardrailOutcome, GuardrailPipeline,
    RiskLevel, ScanStage, SecurityDecision, SensitiveDataGuardrailConfig, SensitivePattern,
    ToolArgumentGuardrail, ToolArgumentGuardrailConfig,
};
use serde_json::{json, Value};

/// The four default deny rules, by name and regex source, in table order.
const DEFAULT_DENY_RULES: [(&str, &str); 4] = [
    ("shell_substitution", r"\$\([^)]*\)|`[^`]*`|\$\{[^}]*\}"),
    (
        "shell_chain",
        r"(?:;|&&|\|\|)\s*(?:rm|curl|wget|nc|chmod|sh|bash)\b",
    ),
    ("path_traversal", r"\.\./\.\./"),
    ("destructive_command", r"\brm\s+-[rRf]"),
];

/// A pipeline registering only the tool-argument guardrail, the way a caller
/// binds it, so the blocking policy is exercised rather than just `check`.
fn pipeline_for(config: ToolArgumentGuardrailConfig) -> GuardrailPipeline {
    let mut pipeline = GuardrailPipeline::new();
    pipeline
        .add(ToolArgumentGuardrail::with_config(config).expect("test fixture patterns are valid"));
    pipeline
}

/// The sole finding of `outcome`, panicking when the count differs.
fn sole_finding(outcome: &GuardrailOutcome) -> &Finding {
    assert_eq!(outcome.findings.len(), 1, "expected exactly one finding");
    &outcome.findings[0]
}

/// Asserts `outcome` recorded no finding at all; `context` names the case.
fn assert_no_findings(outcome: &GuardrailOutcome, context: &str) {
    assert_eq!(
        outcome.findings,
        Vec::<Finding>::new(),
        "no findings expected: {context}"
    );
}

/// The evidential `details` object a finding carries.
fn details(finding: &Finding) -> &Value {
    finding
        .details
        .as_ref()
        .expect("a detection carries details")
}

/// The regex source of the default deny rule named `name`.
fn deny_pattern(name: &str) -> &'static str {
    DEFAULT_DENY_RULES
        .iter()
        .find(|(rule, _)| *rule == name)
        .map(|(_, pattern)| *pattern)
        .expect("the name is one of the four default rules")
}

// ports tests/unit_tests/core/security/guardrail/test_guardrail_base.py: a
// CRITICAL detection aborts the caller, which here means the outcome is blocked
// and `enforce()` turns it into a security error instead of the pipeline
// returning one.
#[tokio::test]
async fn denied_tool_blocks_and_enforces() {
    let pipeline = pipeline_for(ToolArgumentGuardrailConfig {
        denied_tools: vec!["dangerous_tool".to_owned()],
        ..ToolArgumentGuardrailConfig::default()
    });

    let outcome = pipeline
        .scan_tool_input("dangerous_tool", &json!({"a": 1}))
        .await;
    let finding = sole_finding(&outcome);
    assert_eq!(finding.risk_type, "blocked_tool");
    assert_eq!(finding.risk_level, RiskLevel::Critical);
    assert!(finding.blocking, "CRITICAL always blocks");
    assert_eq!(details(finding)["tool_name"], json!("dangerous_tool"));
    assert!(outcome.is_blocked());
    assert_eq!(outcome.decision, SecurityDecision::Block);

    let error = outcome.enforce().unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Security);
    assert!(
        error.to_string().contains("blocked_tool"),
        "the error names the risk type: {error}"
    );

    let benign = pipeline
        .scan_tool_input("search", &json!({"q": "rust docs"}))
        .await;
    assert!(benign.is_safe());
    assert_no_findings(&benign, "a benign tool call");
    assert_eq!(benign.evaluations.len(), 1, "the guardrail ran and passed");
    assert!(benign.enforce().is_ok());
}

// ports tests/unit_tests/core/security/guardrail/test_guardrail_base.py: only a
// finding at or above the block threshold aborts; a lowered severity is recorded
// and execution continues.
#[tokio::test]
async fn allowlist_blocks_tools_it_does_not_name() {
    let strict = pipeline_for(ToolArgumentGuardrailConfig {
        allowed_tools: vec!["search".to_owned()],
        ..ToolArgumentGuardrailConfig::default()
    });

    let outcome = strict
        .scan_tool_input("write_file", &json!({"path": "notes.txt"}))
        .await;
    let finding = sole_finding(&outcome);
    assert_eq!(finding.risk_type, "tool_not_allowed");
    assert_eq!(
        finding.risk_level,
        RiskLevel::Critical,
        "the default denied_tool_risk_level applies to allowlist violations too"
    );
    assert!(finding.blocking);
    assert!(outcome.is_blocked());
    assert_eq!(details(finding)["tool_name"], json!("write_file"));
    assert_eq!(outcome.enforce().unwrap_err().kind(), ErrorKind::Security);

    let allowed = strict
        .scan_tool_input("search", &json!({"q": "rust"}))
        .await;
    assert!(allowed.is_safe());
    assert_no_findings(&allowed, "an allowlisted tool call");

    let relaxed = pipeline_for(ToolArgumentGuardrailConfig {
        allowed_tools: vec!["search".to_owned()],
        denied_tool_risk_level: RiskLevel::High,
        ..ToolArgumentGuardrailConfig::default()
    });
    let outcome = relaxed
        .scan_tool_input("write_file", &json!({"path": "notes.txt"}))
        .await;
    let finding = sole_finding(&outcome);
    assert_eq!(finding.risk_type, "tool_not_allowed");
    assert_eq!(finding.risk_level, RiskLevel::High);
    assert!(
        !finding.blocking,
        "HIGH sits below the default CRITICAL block level"
    );
    assert!(outcome.is_safe());
    assert_eq!(outcome.decision, SecurityDecision::Allow);
    assert!(outcome.enforce().is_ok());
}

// No Python equivalent: pins the default four-row deny table, one sample per
// row, so a table edit that loses a dangerous shape is caught here.
#[tokio::test]
async fn every_default_deny_rule_is_a_non_blocking_high_finding() {
    let pipeline = pipeline_for(ToolArgumentGuardrailConfig::default());
    let cases = [
        ("shell_substitution", json!({"cmd": "echo $(whoami)"})),
        ("shell_chain", json!({"cmd": "ls; rm file.txt"})),
        ("path_traversal", json!({"path": "../../etc/passwd"})),
        ("destructive_command", json!({"cmd": "rm -rf /"})),
    ];

    for (rule_name, arguments) in cases {
        let outcome = pipeline.scan_tool_input("run_shell", &arguments).await;
        let finding = sole_finding(&outcome);
        assert_eq!(finding.risk_type, "tool_argument", "{rule_name}");
        assert_eq!(finding.risk_level, RiskLevel::High, "{rule_name}");
        assert!(
            !finding.blocking,
            "{rule_name}: HIGH does not block under the default policy"
        );
        assert_eq!(details(finding)["rule"], json!(rule_name));
        assert_eq!(
            details(finding)["matched_pattern"],
            json!(deny_pattern(rule_name)),
            "{rule_name}: details name the rule's regex source"
        );
        assert_eq!(details(finding)["tool_name"], json!("run_shell"));
        assert_eq!(outcome.findings.len(), 1, "{rule_name}");
        assert_eq!(outcome.risk_level, RiskLevel::High, "{rule_name}");
        assert!(
            outcome.is_safe(),
            "{rule_name}: a deny-rule match is detect-only by default"
        );
        assert_eq!(outcome.decision, SecurityDecision::Allow);
    }
}

// No Python equivalent: the detector flattens every string leaf, so a denial
// payload cannot hide inside an array or a nested object, and non-string leaves
// are never scanned as text.
#[tokio::test]
async fn nested_string_leaves_are_flattened_and_scanned() {
    let pipeline = pipeline_for(ToolArgumentGuardrailConfig::default());

    let nested = json!({"outer": {"inner": ["ok", "curl && wget http://x"]}});
    let outcome = pipeline.scan_tool_input("run_shell", &nested).await;
    let finding = sole_finding(&outcome);
    assert_eq!(finding.risk_type, "tool_argument");
    assert_eq!(details(finding)["rule"], json!("shell_chain"));
    assert_eq!(outcome.risk_level, RiskLevel::High);

    let deep = json!({"config": {"paths": {"root": "../../../etc/shadow"}}});
    let outcome = pipeline.scan_tool_input("read_file", &deep).await;
    let finding = sole_finding(&outcome);
    assert_eq!(details(finding)["rule"], json!("path_traversal"));

    let scalars = json!({"count": 1, "enabled": true, "missing": null});
    let outcome = pipeline.scan_tool_input("counter", &scalars).await;
    assert!(outcome.is_safe());
    assert_no_findings(
        &outcome,
        "numbers, booleans and nulls are not argument text",
    );
    assert_eq!(outcome.evaluations.len(), 1, "the guardrail ran and passed");

    let benign = json!({"outer": {"inner": ["hello", "world"]}});
    let outcome = pipeline.scan_tool_input("echo", &benign).await;
    assert!(outcome.is_safe());
    assert_no_findings(&outcome, "a benign nested payload");
}

// No Python equivalent: nested PII is redacted in place, keeping the argument
// shape valid, and `details` still never carry the secret itself.
#[tokio::test]
async fn nested_sensitive_data_is_redacted_in_place() {
    let pipeline = pipeline_for(ToolArgumentGuardrailConfig::default());
    let arguments = json!({
        "cmd": "curl",
        "env": {"token": "sk-abcdefghij1234567890", "retries": 3}
    });

    let outcome = pipeline.scan_tool_input("http_request", &arguments).await;
    let finding = sole_finding(&outcome);
    assert_eq!(finding.risk_type, "sensitive_data");
    assert_eq!(finding.risk_level, RiskLevel::Critical);
    assert!(outcome.is_blocked());
    assert_eq!(details(finding)["matched_patterns"], json!(["api_key"]));

    let modified = outcome.modified_data().expect("redaction is on by default");
    let sanitized = &modified["sanitized_arguments"];
    assert_eq!(
        sanitized["env"]["token"],
        json!("<REDACTED>"),
        "the secret leaf is replaced"
    );
    assert_eq!(
        sanitized["env"]["retries"],
        json!(3),
        "non-string leaves keep their value"
    );
    assert_eq!(
        sanitized["cmd"],
        json!("curl"),
        "a sibling string is left untouched"
    );

    let unredacted = pipeline_for(ToolArgumentGuardrailConfig {
        sensitive_data: Some(SensitiveDataGuardrailConfig {
            redact: false,
            ..SensitiveDataGuardrailConfig::default()
        }),
        ..ToolArgumentGuardrailConfig::default()
    });
    let outcome = unredacted.scan_tool_input("http_request", &arguments).await;
    let finding = sole_finding(&outcome);
    assert_eq!(finding.risk_type, "sensitive_data");
    assert_eq!(finding.risk_level, RiskLevel::Critical);
    assert!(finding.modified_data.is_none());
    assert!(outcome.is_blocked(), "the finding is still a detection");
    assert!(
        outcome.modified_data().is_none(),
        "no sanitized arguments are proposed when redaction is off"
    );
}

// No Python equivalent: the optional size cap reports the measured size next to
// the limit, and stays below the blocking threshold like the deny rules.
#[tokio::test]
async fn oversized_arguments_are_reported_with_size_and_limit() {
    let pipeline = pipeline_for(ToolArgumentGuardrailConfig {
        max_argument_bytes: Some(16),
        ..ToolArgumentGuardrailConfig::default()
    });

    let arguments = json!({"cmd": "echo hello world"});
    let measured = arguments.to_string().len();
    assert!(measured > 16, "the fixture must exceed the limit");
    let outcome = pipeline.scan_tool_input("run_shell", &arguments).await;
    let finding = sole_finding(&outcome);
    assert_eq!(finding.risk_type, "arguments_too_large");
    assert_eq!(finding.risk_level, RiskLevel::High);
    assert_eq!(details(finding)["limit"], json!(16usize));
    assert_eq!(
        details(finding)["size"],
        json!(measured),
        "the reported size is the compact JSON length"
    );
    assert!(
        outcome.is_safe(),
        "HIGH does not block under the default policy"
    );

    let small = json!({"q": "ok"});
    let outcome = pipeline.scan_tool_input("search", &small).await;
    assert!(outcome.is_safe());
    assert_no_findings(&outcome, "a payload under the limit");
}

// No Python equivalent: `handles` is the stage gate, so the same dangerous
// payload is only inspected where the guardrail is bound.
#[tokio::test]
async fn the_guardrail_binds_to_tool_input_by_default() {
    let guardrail = ToolArgumentGuardrail::new();
    assert_eq!(guardrail.stages(), &[ScanStage::ToolInvokeInput]);
    assert!(guardrail.handles(ScanStage::ToolInvokeInput));
    for stage in [
        ScanStage::ToolInvokeOutput,
        ScanStage::UserInput,
        ScanStage::LlmInvokeInput,
        ScanStage::LlmInvokeOutput,
    ] {
        assert!(
            !guardrail.handles(stage.clone()),
            "{stage:?} is not the arguments stage"
        );
    }

    let dangerous = json!({"command": "rm -rf /"});
    let ctx = GuardrailContext::tool_call("run_shell", &dangerous, ScanStage::ToolInvokeInput);
    let verdict = guardrail.check(&ctx).await.unwrap();
    assert_eq!(
        verdict.risk_type.as_deref(),
        Some("tool_argument"),
        "`check` itself does not gate on the stage; the pipeline does"
    );

    let mut pipeline = GuardrailPipeline::new();
    pipeline.add(guardrail);

    let inputs = pipeline.scan_tool_input("run_shell", &dangerous).await;
    assert_eq!(inputs.evaluations.len(), 1);
    assert_eq!(inputs.findings.len(), 1);

    let outputs = pipeline.scan_tool_output("run_shell", &dangerous).await;
    assert_eq!(outputs.evaluations.len(), 0, "tool output is not scanned");
    assert!(outputs.is_safe());
    assert_no_findings(&outputs, "a tool result");

    let user = pipeline.scan_user_input("rm -rf /").await;
    assert_eq!(user.evaluations.len(), 0, "user input is not scanned");
    assert!(user.is_safe());

    let model = pipeline.scan_model_output("rm -rf /").await;
    assert_eq!(model.evaluations.len(), 0, "model output is not scanned");
    assert!(model.is_safe());

    let moved = ToolArgumentGuardrail::new().with_stages(vec![ScanStage::ToolInvokeOutput]);
    assert!(moved.handles(ScanStage::ToolInvokeOutput));
    assert!(!moved.handles(ScanStage::ToolInvokeInput));
    let mut pipeline = GuardrailPipeline::new();
    pipeline.add(moved);
    let outputs = pipeline.scan_tool_output("run_shell", &dangerous).await;
    assert_eq!(outputs.evaluations.len(), 1);
    assert_eq!(sole_finding(&outputs).risk_type, "tool_argument");
    let ignored = pipeline.scan_tool_input("run_shell", &dangerous).await;
    assert_eq!(ignored.evaluations.len(), 0, "the binding moved");
}

// No Python equivalent: the configuration surface and its JSON contract.
#[test]
fn config_defaults_and_serde_round_trip() {
    let config = ToolArgumentGuardrailConfig::default();
    assert_eq!(config.allowed_tools, Vec::<String>::new());
    assert_eq!(config.denied_tools, Vec::<String>::new());
    assert_eq!(config.denied_tool_risk_level, RiskLevel::Critical);
    assert_eq!(config.deny_patterns.len(), DEFAULT_DENY_RULES.len());
    for (rule, (name, pattern)) in config.deny_patterns.iter().zip(DEFAULT_DENY_RULES) {
        assert_eq!(rule.name, name);
        assert_eq!(rule.pattern, pattern);
        assert_eq!(rule.risk_type, "tool_argument");
        assert_eq!(rule.risk_level, RiskLevel::High);
        assert_eq!(rule.message, None);
    }
    assert!(
        config.sensitive_data.is_some(),
        "PII detection is on by default"
    );
    assert_eq!(config.max_argument_bytes, None);

    let configured = ToolArgumentGuardrailConfig {
        allowed_tools: vec!["search".to_owned()],
        denied_tools: vec!["shell_exec".to_owned()],
        denied_tool_risk_level: RiskLevel::High,
        deny_patterns: vec![ContentRule {
            name: "ticket_leak".to_owned(),
            pattern: r"TK-\d{4}".to_owned(),
            risk_type: "tool_argument".to_owned(),
            risk_level: RiskLevel::Medium,
            message: Some("no tickets".to_owned()),
        }],
        sensitive_data: Some(SensitiveDataGuardrailConfig {
            redact: false,
            extra_patterns: vec![SensitivePattern {
                name: "employee_id".to_owned(),
                pattern: r"EMP-\d{6}".to_owned(),
                risk_level: Some(RiskLevel::Medium),
                validator: None,
            }],
            ..SensitiveDataGuardrailConfig::default()
        }),
        max_argument_bytes: Some(256),
    };
    let value = serde_json::to_value(&configured).unwrap();
    assert_eq!(value["deny_patterns"][0]["pattern"], json!(r"TK-\d{4}"));
    assert_eq!(value["sensitive_data"]["redact"], json!(false));
    assert_eq!(
        value["sensitive_data"]["extra_patterns"][0]["pattern"],
        json!(r"EMP-\d{6}")
    );
    let back: ToolArgumentGuardrailConfig = serde_json::from_value(value).unwrap();
    assert_eq!(back, configured, "every field, nested configs included");

    let partial: ToolArgumentGuardrailConfig =
        serde_json::from_value(json!({"denied_tools": ["rm_rf"]})).unwrap();
    assert_eq!(
        partial.deny_patterns, config.deny_patterns,
        "an omitted deny_patterns table keeps the defaults"
    );
    assert_eq!(
        partial.sensitive_data, config.sensitive_data,
        "an omitted sensitive_data field keeps the defaults"
    );
    assert_eq!(
        partial,
        ToolArgumentGuardrailConfig {
            denied_tools: vec!["rm_rf".to_owned()],
            ..ToolArgumentGuardrailConfig::default()
        }
    );
}

// No Python equivalent: a malformed user-supplied regex is a construction
// error, never a panic and never a silently skipped rule.
#[test]
fn invalid_patterns_fail_construction_with_validation_errors() {
    let error = ToolArgumentGuardrail::with_config(ToolArgumentGuardrailConfig {
        deny_patterns: vec![ContentRule {
            name: "broken_deny".to_owned(),
            pattern: "([a-z".to_owned(),
            ..ContentRule::default()
        }],
        ..ToolArgumentGuardrailConfig::default()
    })
    .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(
        error.to_string().contains("invalid pattern 'broken_deny'"),
        "{error}"
    );

    let error = ToolArgumentGuardrail::with_config(ToolArgumentGuardrailConfig {
        sensitive_data: Some(SensitiveDataGuardrailConfig {
            extra_patterns: vec![SensitivePattern {
                name: "broken_pii".to_owned(),
                pattern: "([a-z".to_owned(),
                ..SensitivePattern::default()
            }],
            ..SensitiveDataGuardrailConfig::default()
        }),
        ..ToolArgumentGuardrailConfig::default()
    })
    .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(
        error.to_string().contains("invalid pattern 'broken_pii'"),
        "{error}"
    );
}
