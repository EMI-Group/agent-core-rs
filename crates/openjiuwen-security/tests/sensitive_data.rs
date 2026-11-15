#![allow(clippy::unwrap_used, clippy::expect_used)]

//! Behavior suite for the sensitive-data/PII guardrail.
//!
//! **This suite is an addition with no Python equivalent.** The Python package
//! advertises PII guardrails it never shipped: `openjiuwen/core/security/
//! guardrail/builtin.py` (v0.1.20) defines prompt injection only, and no
//! `tests/unit_tests/core/security/guardrail/test_*` file covers redaction. The
//! observable contract pinned here is therefore Python's rule-backend contract —
//! a match is a finding at the pattern's severity, the content is redacted when
//! the config asks for it, and `modified_data` carries the sanitized text — plus
//! the two-tier blocking policy of
//! `tests/unit_tests/core/security/guardrail/test_guardrail_base.py`: only
//! `CRITICAL` aborts the caller, every lower severity is recorded and execution
//! continues.
//!
//! Everything runs offline against the built-in regex table: no backend, no
//! network, no model runtime, no wall clock.

use openjiuwen_common::error::ErrorKind;
use openjiuwen_security::guardrails::sensitive_data::{
    SensitiveDataGuardrail, SensitiveDataGuardrailConfig, SensitivePattern, SensitiveValidator,
};
use openjiuwen_security::{
    Guardrail, GuardrailContext, GuardrailOutcome, GuardrailPipeline, GuardrailResult, RiskLevel,
    ScanStage, SecurityDecision,
};
use serde_json::{json, Value};

/// Builds a guardrail from `config`, panicking on an invalid pattern (every
/// fixture in this suite is statically valid).
fn guardrail(config: SensitiveDataGuardrailConfig) -> SensitiveDataGuardrail {
    SensitiveDataGuardrail::with_config(config).expect("test fixture patterns are valid")
}

/// A user-input context over `text`.
fn text_ctx(text: &str) -> GuardrailContext {
    GuardrailContext::text(text, ScanStage::UserInput)
}

/// A pattern row without a validator, so tests only spell out what they vary.
fn pattern(name: &str, source: &str, risk_level: Option<RiskLevel>) -> SensitivePattern {
    SensitivePattern {
        name: name.to_owned(),
        pattern: source.to_owned(),
        risk_level,
        validator: None,
    }
}

/// Runs `guardrail` through a pipeline over user input, the way a caller binds
/// it, so the blocking policy is exercised rather than just `check`.
async fn scan(guardrail: SensitiveDataGuardrail, text: &str) -> GuardrailOutcome {
    let mut pipeline = GuardrailPipeline::new();
    pipeline.add(guardrail);
    pipeline.scan_user_input(text).await
}

/// The `sanitized_text` a check exposed, as an owned string.
fn sanitized_text(result: &GuardrailResult) -> String {
    result
        .modified_data
        .as_ref()
        .expect("a redacted verdict carries modified_data")["sanitized_text"]
        .as_str()
        .expect("sanitized_text is a string")
        .to_owned()
}

/// The evidential `details` object a check exposed.
fn details(result: &GuardrailResult) -> &Value {
    result
        .details
        .as_ref()
        .expect("a detection carries details")
}

// No Python equivalent: pins the default seven-row table this crate ships, one
// sample per row, so a table edit that breaks detection is caught here.
#[tokio::test]
async fn every_default_pattern_detects_and_redacts_its_own_sample() {
    let guardrail = SensitiveDataGuardrail::new();
    let cases = [
        ("email", "user@example.com", RiskLevel::High),
        ("cn_mobile", "13812345678", RiskLevel::Medium),
        ("cn_id_card", "110101199003071234", RiskLevel::High),
        ("credit_card", "4111 1111 1111 1111", RiskLevel::High),
        ("us_ssn", "123-45-6789", RiskLevel::High),
        ("api_key", "sk-abcdefghij1234567890", RiskLevel::Critical),
        (
            "private_key",
            "-----BEGIN RSA PRIVATE KEY-----",
            RiskLevel::Critical,
        ),
    ];

    for (name, sample, level) in cases {
        let result = guardrail.check(&text_ctx(sample)).await.unwrap();
        assert!(!result.is_safe(), "{name}: {sample:?} must be detected");
        assert_eq!(result.risk_level, level, "{name}: severity");
        assert_eq!(
            result.risk_type.as_deref(),
            Some("sensitive_data"),
            "{name}: risk type"
        );
        assert_eq!(
            details(&result)["matched_patterns"],
            json!([name]),
            "{name}: only the row that owns the sample reports it"
        );
        assert!(
            details(&result)["match_count"].as_u64().unwrap() >= 1,
            "{name}: at least one match"
        );
        assert_eq!(
            sanitized_text(&result),
            "<REDACTED>",
            "{name}: the value is replaced in place"
        );
    }
}

#[tokio::test]
async fn failed_luhn_candidate_is_not_a_match() {
    let guardrail = SensitiveDataGuardrail::new();
    let text = "card 4111 1111 1111 1112 expires soon";

    let result = guardrail.check(&text_ctx(text)).await.unwrap();
    assert!(
        result.is_safe(),
        "a failing candidate is not a match at all"
    );
    assert_eq!(result.details, None);
    assert_eq!(result.modified_data, None);

    let outcome = guardrail.redact_text(text);
    assert_eq!(outcome.matches.len(), 0);
    assert_eq!(outcome.redacted, text, "nothing is redacted either");
}

#[tokio::test]
async fn valid_card_is_redacted_in_place() {
    let guardrail = SensitiveDataGuardrail::new();
    let text = "call 4111 1111 1111 1111 now";

    let result = guardrail.check(&text_ctx(text)).await.unwrap();
    assert_eq!(
        result.modified_data,
        Some(json!({"sanitized_text": "call <REDACTED> now"}))
    );
    assert_eq!(
        result.details,
        Some(json!({"matched_patterns": ["credit_card"], "match_count": 1}))
    );
}

#[tokio::test]
async fn matched_patterns_are_deduplicated_in_table_order() {
    let guardrail = SensitiveDataGuardrail::new();
    let text = "mail user@example.com, phone 13812345678, key sk-abcdefghij1234567890";

    let result = guardrail.check(&text_ctx(text)).await.unwrap();
    assert_eq!(
        result.details,
        Some(json!({
            "matched_patterns": ["email", "cn_mobile", "api_key"],
            "match_count": 3,
        })),
        "table order, one entry per pattern"
    );
    assert_eq!(
        result.risk_level,
        RiskLevel::Critical,
        "the result severity is the worst finding (the api key)"
    );

    let sanitized = sanitized_text(&result);
    assert_eq!(sanitized.matches("<REDACTED>").count(), 3);
    assert!(!sanitized.contains("user@"), "{sanitized}");
    assert!(!sanitized.contains("13812345678"), "{sanitized}");
    assert!(!sanitized.contains("sk-"), "{sanitized}");
}

#[tokio::test]
async fn benign_prose_and_innocuous_hex_runs_do_not_match() {
    let guardrail = SensitiveDataGuardrail::new();
    let samples = [
        "The quick brown fox jumps over the lazy dog while the build finishes.",
        // A UUID's digit runs are glued to hex letters, so the word boundary
        // every numeric pattern needs never lands inside one, and its longest
        // standalone digit run (12) is below the card pattern's 13-digit floor.
        "build 550e8400-e29b-41d4-a716-446655440000",
    ];

    for text in samples {
        let result = guardrail.check(&text_ctx(text)).await.unwrap();
        assert!(result.is_safe(), "{text:?} must pass");
        assert_eq!(result.details, None, "{text:?}");
        assert_eq!(result.modified_data, None, "{text:?}");
        assert!(
            guardrail.redact_text(text).matches.is_empty(),
            "{text:?}: no match to redact"
        );
    }
}

#[tokio::test]
async fn extra_patterns_detect_and_the_default_table_wins_overlaps() {
    let mut config = SensitiveDataGuardrailConfig::default();
    config.extra_patterns.push(pattern(
        "employee_id",
        r"\bEMP-\d{5}\b",
        Some(RiskLevel::Medium),
    ));
    let badge = guardrail(config);

    let text = "badge EMP-12345 assigned";
    let result = badge.check(&text_ctx(text)).await.unwrap();
    assert!(!result.is_safe());
    assert_eq!(result.risk_level, RiskLevel::Medium);
    assert_eq!(
        result.details,
        Some(json!({"matched_patterns": ["employee_id"], "match_count": 1}))
    );
    assert_eq!(
        result.modified_data,
        Some(json!({"sanitized_text": "badge <REDACTED> assigned"}))
    );

    // Extras are compiled after `patterns`, so a default row claims an
    // overlapping range even when an extra pattern matches it too.
    let mut shadowing = SensitiveDataGuardrailConfig::default();
    shadowing.extra_patterns.push(pattern(
        "email_copy",
        r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}",
        Some(RiskLevel::Critical),
    ));
    let shadowing = guardrail(shadowing);
    let outcome = shadowing.redact_text("write user@example.com");
    assert_eq!(outcome.matches.len(), 1);
    assert_eq!(outcome.matches[0].pattern, "email", "the default row wins");
    assert_eq!(outcome.matches[0].risk_level, RiskLevel::High);
}

#[tokio::test]
async fn allowlist_skips_only_the_exact_match() {
    let guardrail = guardrail(SensitiveDataGuardrailConfig {
        allowlist: vec!["support@example.com".to_owned()],
        ..SensitiveDataGuardrailConfig::default()
    });
    let text = "support@example.com and user@example.com";

    let outcome = guardrail.redact_text(text);
    assert_eq!(
        outcome.matches.len(),
        1,
        "the allowlisted address is skipped"
    );
    assert_eq!(outcome.matches[0].pattern, "email");
    assert_eq!(outcome.redacted, "support@example.com and <REDACTED>");

    let result = guardrail.check(&text_ctx(text)).await.unwrap();
    assert_eq!(
        result.details,
        Some(json!({"matched_patterns": ["email"], "match_count": 1}))
    );
    assert_eq!(
        sanitized_text(&result),
        "support@example.com and <REDACTED>"
    );
}

#[tokio::test]
async fn marker_override_is_used_for_redaction() {
    let guardrail = guardrail(SensitiveDataGuardrailConfig {
        marker: "[MASKED]".to_owned(),
        ..SensitiveDataGuardrailConfig::default()
    });

    let result = guardrail
        .check(&text_ctx("mail user@example.com"))
        .await
        .unwrap();
    let sanitized = sanitized_text(&result);
    assert_eq!(sanitized, "mail [MASKED]");
    assert!(!sanitized.contains("<REDACTED>"), "{sanitized}");
}

#[tokio::test]
async fn redact_false_reports_the_finding_without_sanitized_text() {
    let guardrail = guardrail(SensitiveDataGuardrailConfig {
        redact: false,
        ..SensitiveDataGuardrailConfig::default()
    });
    let text = "key sk-abcdefghij1234567890";

    let result = guardrail.check(&text_ctx(text)).await.unwrap();
    assert!(!result.is_safe(), "the finding is still recorded");
    assert_eq!(result.risk_level, RiskLevel::Critical);
    assert_eq!(result.modified_data, None);
    assert!(result.details.is_some(), "the evidence is still recorded");

    let outcome = scan(guardrail, text).await;
    assert_eq!(outcome.findings.len(), 1);
    assert_eq!(
        outcome.modified_data(),
        None,
        "no guardrail contributes sanitized content"
    );
}

// ports tests/unit_tests/core/security/guardrail/test_guardrail_base.py:
// only a CRITICAL verdict aborts the call; a lower severity is recorded and
// execution continues.
#[tokio::test]
async fn default_policy_blocks_only_critical_findings() {
    for (name, text) in [
        ("api_key", "key sk-abcdefghij1234567890"),
        ("private_key", "-----BEGIN OPENSSH PRIVATE KEY-----"),
    ] {
        let outcome = scan(SensitiveDataGuardrail::new(), text).await;
        assert_eq!(outcome.risk_level, RiskLevel::Critical, "{name}");
        assert_eq!(outcome.decision, SecurityDecision::Block, "{name}");
        assert!(outcome.is_blocked(), "{name}");
        assert_eq!(
            outcome.blocked_by.as_deref(),
            Some("sensitive_data"),
            "{name}"
        );
        let error = outcome.enforce().unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Security, "{name}");
        assert_eq!(
            error.message(),
            "Critical security risk detected: sensitive_data",
            "{name}: Python AbortError wording"
        );
    }

    let outcome = scan(SensitiveDataGuardrail::new(), "mail user@example.com").await;
    assert_eq!(outcome.risk_level, RiskLevel::High);
    assert_eq!(outcome.decision, SecurityDecision::Allow);
    assert!(
        outcome.is_safe(),
        "a HIGH finding does not block by default"
    );
    assert_eq!(outcome.findings.len(), 1);
    assert_eq!(outcome.findings[0].guardrail, "sensitive_data");
    assert_eq!(outcome.findings[0].risk_type, "sensitive_data");
    assert!(!outcome.findings[0].blocking);
    assert!(outcome.enforce().is_ok());
}

#[test]
fn redact_text_exposes_byte_ranges_of_the_original_text() {
    let guardrail = SensitiveDataGuardrail::new();

    let ascii = "call 4111 1111 1111 1111 now";
    let outcome = guardrail.redact_text(ascii);
    assert_eq!(outcome.matches.len(), 1);
    let found = &outcome.matches[0];
    assert_eq!(found.pattern, "credit_card");
    assert_eq!(found.start, 5);
    assert_eq!(found.end, 24);
    assert_eq!(&ascii[found.start..found.end], "4111 1111 1111 1111");

    // Byte offsets, not character indices: the CJK surroundings stay intact.
    let text = "联系人邮箱：user@example.com 谢谢";
    let outcome = guardrail.redact_text(text);
    assert_eq!(outcome.matches.len(), 1);
    let found = &outcome.matches[0];
    assert_eq!(found.pattern, "email");
    assert_eq!(found.risk_level, RiskLevel::High);
    assert_eq!(&text[found.start..found.end], "user@example.com");
    assert_eq!(outcome.redacted, "联系人邮箱：<REDACTED> 谢谢");
}

#[test]
fn config_round_trips_through_json() {
    let config = SensitiveDataGuardrailConfig::default();
    let value = serde_json::to_value(&config).unwrap();

    let rows = value["patterns"]
        .as_array()
        .expect("patterns serialize as an array");
    assert_eq!(rows.len(), 7);
    let names: Vec<&str> = rows
        .iter()
        .map(|row| row["name"].as_str().expect("a plain string name"))
        .collect();
    assert_eq!(
        names,
        [
            "email",
            "cn_mobile",
            "cn_id_card",
            "credit_card",
            "us_ssn",
            "api_key",
            "private_key",
        ]
    );
    for row in rows {
        assert!(row["pattern"].is_string(), "pattern source is plain: {row}");
        assert!(row["name"].is_string(), "name is plain: {row}");
    }
    assert_eq!(
        rows[3]["validator"],
        json!("luhn"),
        "only the card row carries a validator"
    );
    assert_eq!(rows[0]["validator"], Value::Null);
    assert_eq!(
        serde_json::to_value(SensitiveValidator::Luhn).unwrap(),
        json!("luhn")
    );

    let restored: SensitiveDataGuardrailConfig = serde_json::from_value(value).unwrap();
    assert_eq!(restored, config, "defaults round-trip unchanged");
    assert_eq!(
        restored.patterns,
        SensitiveDataGuardrailConfig::default().patterns
    );

    let validator: SensitiveValidator = serde_json::from_value(json!("luhn")).unwrap();
    assert_eq!(validator, SensitiveValidator::Luhn);
}

// No Python equivalent: a payload the detector cannot project is a failure,
// never a pass. The direct call reports the typed error; the pipeline resolves
// it through the default fail-closed policy.
#[tokio::test]
async fn unscannable_payload_blocks_instead_of_passing() {
    let guardrail = SensitiveDataGuardrail::new();
    let ctx = GuardrailContext::raw("user_input", Value::Null);

    let error = guardrail.check(&ctx).await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("unscannable content"), "{error}");

    let mut pipeline = GuardrailPipeline::new();
    pipeline.add(SensitiveDataGuardrail::new());
    let outcome = pipeline.run(&ctx).await;
    assert!(
        outcome.is_blocked(),
        "an unscannable payload must never read as safe"
    );
    assert_eq!(outcome.decision, SecurityDecision::Block);
    assert_eq!(outcome.risk_level, RiskLevel::Critical);
    assert_eq!(outcome.blocked_by.as_deref(), Some("context"));
    let finding = &outcome.findings[0];
    assert_eq!(finding.guardrail, "context");
    assert_eq!(finding.risk_type, "unscannable_content");
    assert_eq!(finding.risk_level, RiskLevel::Critical);
    assert!(finding.blocking);
    assert!(
        finding.message.contains("unscannable_content"),
        "{}",
        finding.message
    );
    assert_eq!(outcome.errors.len(), 1);
    assert!(
        outcome.errors[0].contains("unscannable content"),
        "{:?}",
        outcome.errors
    );

    let error = outcome.enforce().unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Security);
    assert!(error.to_string().contains("unscannable content"), "{error}");
}
