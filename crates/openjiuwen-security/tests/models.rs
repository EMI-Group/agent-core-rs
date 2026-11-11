#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Severity/model serde, `GuardrailResult` shape, and redacting rendering.
//!
//! Ports the assertion set of
//! `tests/unit_tests/core/security/guardrail/test_guardrail_models.py` and the
//! rendering half of `test_guardrail_base.py` against the public
//! `openjiuwen_security` API only.

use openjiuwen_common::error::{Error, ErrorKind};
use openjiuwen_security::context::{
    EVENT_LLM_INVOKE_INPUT, EVENT_LLM_INVOKE_OUTPUT, EVENT_TOOL_INVOKE_INPUT,
    EVENT_TOOL_INVOKE_OUTPUT, EVENT_USER_INPUT,
};
use openjiuwen_security::finding::finding_message;
use openjiuwen_security::{
    Finding, GuardrailContentType, GuardrailContext, GuardrailEvaluation, GuardrailOutcome,
    GuardrailResult, GuardrailViolation, RedactedOutcome, RiskAssessment, RiskLevel, ScanStage,
    SecurityDecision,
};
use serde_json::json;

/// A secret that must never survive a `Debug`/`Display` rendering.
const SECRET: &str = "sk-live-123456";

/// The five levels in severity order.
fn ordered_levels() -> [RiskLevel; 5] {
    [
        RiskLevel::Safe,
        RiskLevel::Low,
        RiskLevel::Medium,
        RiskLevel::High,
        RiskLevel::Critical,
    ]
}

/// A blocking finding whose `details` leak a secret under a masked key.
fn secret_finding() -> Finding {
    Finding {
        guardrail: "secret".to_owned(),
        risk_level: RiskLevel::Critical,
        risk_type: "secret".to_owned(),
        message: finding_message("secret", RiskLevel::Critical, true),
        blocking: true,
        details: Some(json!({"query": SECRET, "pattern": "aws_key"})),
        modified_data: None,
    }
}

mod risk_level {
    use super::*;

    // ports tests/unit_tests/core/security/guardrail/test_guardrail_models.py
    #[test]
    fn as_str_keeps_python_lowercase_values() {
        assert_eq!(RiskLevel::Safe.as_str(), "safe");
        assert_eq!(RiskLevel::Low.as_str(), "low");
        assert_eq!(RiskLevel::Medium.as_str(), "medium");
        assert_eq!(RiskLevel::High.as_str(), "high");
        assert_eq!(RiskLevel::Critical.as_str(), "critical");
    }

    #[test]
    fn name_and_display_are_uppercase() {
        for level in ordered_levels() {
            assert_eq!(level.name(), level.as_str().to_ascii_uppercase());
            assert_eq!(level.to_string(), level.name());
        }
    }

    #[test]
    fn default_is_safe() {
        assert_eq!(RiskLevel::default(), RiskLevel::Safe);
    }

    #[test]
    fn is_risky_is_false_only_for_safe() {
        assert!(!RiskLevel::Safe.is_risky());
        for level in [
            RiskLevel::Low,
            RiskLevel::Medium,
            RiskLevel::High,
            RiskLevel::Critical,
        ] {
            assert!(level.is_risky(), "{level} must be risky");
        }
    }

    // ports tests/unit_tests/core/security/guardrail/test_guardrail_models.py
    #[test]
    fn ordering_is_safe_to_critical() {
        let levels = ordered_levels();
        for (index, window) in levels.windows(2).enumerate() {
            assert!(
                window[0] < window[1],
                "level {index}: {} must sort before {}",
                window[0],
                window[1]
            );
        }
        assert!(RiskLevel::Safe < RiskLevel::Critical);
    }

    #[test]
    fn at_least_follows_the_severity_order() {
        for (index, level) in ordered_levels().into_iter().enumerate() {
            for (threshold_index, threshold) in ordered_levels().into_iter().enumerate() {
                assert_eq!(
                    level.at_least(threshold),
                    index >= threshold_index,
                    "{level}.at_least({threshold})"
                );
            }
        }
    }

    #[test]
    fn from_value_parses_every_value() {
        for level in ordered_levels() {
            assert_eq!(RiskLevel::from_value(level.as_str()), Some(level));
        }
    }

    #[test]
    fn from_value_ignores_case_and_whitespace() {
        assert_eq!(RiskLevel::from_value("CRITICAL"), Some(RiskLevel::Critical));
        assert_eq!(RiskLevel::from_value("  High "), Some(RiskLevel::High));
        assert_eq!(RiskLevel::from_value("\tmedium\n"), Some(RiskLevel::Medium));
    }

    #[test]
    fn from_value_unknown_is_none() {
        assert_eq!(RiskLevel::from_value("severe"), None);
        assert_eq!(RiskLevel::from_value(""), None);
        assert_eq!(RiskLevel::from_value("critical!"), None);
    }

    #[test]
    fn serde_uses_the_python_value() {
        assert_eq!(
            serde_json::to_value(RiskLevel::Critical).unwrap(),
            json!("critical")
        );
        assert_eq!(
            serde_json::to_value(RiskLevel::Safe).unwrap(),
            json!("safe")
        );
    }

    #[test]
    fn serde_deserializes_a_known_level() {
        let level: RiskLevel = serde_json::from_value(json!("high")).unwrap();
        assert_eq!(level, RiskLevel::High);
    }

    #[test]
    fn serde_rejects_an_unknown_level() {
        assert!(serde_json::from_value::<RiskLevel>(json!("severe")).is_err());
        assert!(serde_json::from_value::<RiskLevel>(json!("HIGH")).is_err());
    }
}

mod content_type {
    use super::*;

    #[test]
    fn serde_names_are_snake_case() {
        assert_eq!(
            serde_json::to_value(GuardrailContentType::Text).unwrap(),
            json!("text")
        );
        assert_eq!(
            serde_json::to_value(GuardrailContentType::Messages).unwrap(),
            json!("messages")
        );
        assert_eq!(
            serde_json::to_value(GuardrailContentType::ToolCall).unwrap(),
            json!("tool_call")
        );
        assert_eq!(
            serde_json::to_value(GuardrailContentType::Raw).unwrap(),
            json!("raw")
        );
    }

    #[test]
    fn default_is_text() {
        assert_eq!(GuardrailContentType::default(), GuardrailContentType::Text);
    }
}

mod scan_stage {
    use super::*;

    #[test]
    fn event_constants_match_python() {
        assert_eq!(EVENT_LLM_INVOKE_INPUT, "_framework:llm_invoke_input");
        assert_eq!(EVENT_LLM_INVOKE_OUTPUT, "_framework:llm_invoke_output");
        assert_eq!(EVENT_TOOL_INVOKE_INPUT, "_framework:tool_invoke_input");
        assert_eq!(EVENT_TOOL_INVOKE_OUTPUT, "_framework:tool_invoke_output");
        assert_eq!(EVENT_USER_INPUT, "user_input");
    }

    #[test]
    fn known_variants_serialize_to_their_event() {
        let pairs = [
            (ScanStage::LlmInvokeInput, json!(EVENT_LLM_INVOKE_INPUT)),
            (ScanStage::LlmInvokeOutput, json!(EVENT_LLM_INVOKE_OUTPUT)),
            (ScanStage::ToolInvokeInput, json!(EVENT_TOOL_INVOKE_INPUT)),
            (ScanStage::ToolInvokeOutput, json!(EVENT_TOOL_INVOKE_OUTPUT)),
            (ScanStage::UserInput, json!(EVENT_USER_INPUT)),
        ];
        for (stage, expected) in pairs {
            assert_eq!(serde_json::to_value(&stage).unwrap(), expected);
        }
    }

    #[test]
    fn other_serializes_as_its_event_and_round_trips() {
        let stage = ScanStage::Other("x.y".to_owned());
        let value = serde_json::to_value(&stage).unwrap();
        assert_eq!(value, json!("x.y"));
        let back: ScanStage = serde_json::from_value(value).unwrap();
        assert_eq!(back, stage);
    }

    #[test]
    fn from_event_inverts_event() {
        for (stage, event) in [
            (ScanStage::LlmInvokeInput, EVENT_LLM_INVOKE_INPUT),
            (ScanStage::LlmInvokeOutput, EVENT_LLM_INVOKE_OUTPUT),
            (ScanStage::ToolInvokeInput, EVENT_TOOL_INVOKE_INPUT),
            (ScanStage::ToolInvokeOutput, EVENT_TOOL_INVOKE_OUTPUT),
            (ScanStage::UserInput, EVENT_USER_INPUT),
        ] {
            assert_eq!(stage.event(), event);
            assert_eq!(ScanStage::from_event(event), stage);
        }
    }

    #[test]
    fn from_event_unknown_becomes_other() {
        assert_eq!(
            ScanStage::from_event("x.y"),
            ScanStage::Other("x.y".to_owned())
        );
    }

    #[test]
    fn display_is_the_event_string() {
        for stage in [
            ScanStage::LlmInvokeInput,
            ScanStage::UserInput,
            ScanStage::Other("x.y".to_owned()),
        ] {
            assert_eq!(stage.to_string(), stage.event());
        }
    }

    #[test]
    fn content_type_mapping() {
        for stage in [
            ScanStage::LlmInvokeInput,
            ScanStage::LlmInvokeOutput,
            ScanStage::UserInput,
        ] {
            assert_eq!(stage.content_type(), GuardrailContentType::Text);
        }
        for stage in [ScanStage::ToolInvokeInput, ScanStage::ToolInvokeOutput] {
            assert_eq!(stage.content_type(), GuardrailContentType::ToolCall);
        }
        assert_eq!(
            ScanStage::Other("x.y".to_owned()).content_type(),
            GuardrailContentType::Raw
        );
    }
}

mod guardrail_result {
    use super::*;

    // ports tests/unit_tests/core/security/guardrail/test_guardrail_models.py
    #[test]
    fn pass_is_safe_and_empty() {
        let result = GuardrailResult::pass();
        assert!(result.is_safe());
        assert_eq!(result.risk_level, RiskLevel::Safe);
        assert_eq!(result.risk_type, None);
        assert_eq!(result.details, None);
        assert_eq!(result.modified_data, None);
    }

    #[test]
    fn pass_json_omits_the_none_fields() {
        let value = serde_json::to_value(GuardrailResult::pass()).unwrap();
        assert_eq!(value, json!({"is_safe": true, "risk_level": "safe"}));
    }

    #[test]
    fn pass_with_keeps_details() {
        let result = GuardrailResult::pass_with(json!({"source": "x"}));
        assert!(result.is_safe());
        assert_eq!(result.risk_level, RiskLevel::Safe);
        assert_eq!(result.details, Some(json!({"source": "x"})));
        assert_eq!(result.risk_type, None);
    }

    // ports test_guardrail_base.py (GuardrailError wording below CRITICAL)
    #[test]
    fn block_at_low_uses_the_guardrail_error_message() {
        let result = GuardrailResult::block(
            RiskLevel::Low,
            "prompt_injection",
            Some(json!({"pattern": "p"})),
            None,
        );
        assert!(!result.is_safe());
        assert_eq!(result.risk_level, RiskLevel::Low);
        assert_eq!(result.risk_type_or_unknown(), "prompt_injection");
        assert_eq!(result.details, Some(json!({"pattern": "p"})));
        assert_eq!(
            result.message(),
            "Guardrail blocked: prompt_injection risk detected"
        );
    }

    // ports test_guardrail_base.py (AbortError reason at CRITICAL)
    #[test]
    fn block_at_critical_uses_the_abort_message() {
        let result = GuardrailResult::block(RiskLevel::Critical, "attack", None, None);
        assert!(!result.is_safe());
        assert_eq!(result.message(), "Critical security risk detected: attack");
    }

    #[test]
    fn block_without_risk_type_degrades_to_unknown() {
        let missing = GuardrailResult {
            is_safe: false,
            risk_level: RiskLevel::High,
            risk_type: None,
            details: None,
            modified_data: None,
        };
        assert_eq!(missing.risk_type_or_unknown(), "unknown");
        assert_eq!(
            missing.message(),
            "Guardrail blocked: unknown risk detected"
        );

        let critical = GuardrailResult {
            risk_level: RiskLevel::Critical,
            ..missing
        };
        assert_eq!(
            critical.message(),
            "Critical security risk detected: unknown"
        );
    }

    #[test]
    fn json_field_names_are_the_python_dataclass_fields() {
        let result = GuardrailResult {
            is_safe: false,
            risk_level: RiskLevel::High,
            risk_type: Some("pii".to_owned()),
            details: Some(json!({"pattern": "email"})),
            modified_data: Some(json!({"sanitized_text": "<REDACTED>"})),
        };
        let value = serde_json::to_value(&result).unwrap();
        let keys: Vec<&str> = value
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(
            keys,
            [
                "is_safe",
                "risk_level",
                "risk_type",
                "details",
                "modified_data"
            ]
        );
        assert_eq!(value["risk_level"], json!("high"));
    }

    #[test]
    fn fully_populated_result_round_trips() {
        let result = GuardrailResult {
            is_safe: false,
            risk_level: RiskLevel::Medium,
            risk_type: Some("toxicity".to_owned()),
            details: Some(json!({"score": 0.5, "patterns": ["a", "b"]})),
            modified_data: Some(json!({"sanitized_text": "clean"})),
        };
        let value = serde_json::to_value(&result).unwrap();
        let back: GuardrailResult = serde_json::from_value(value).unwrap();
        assert_eq!(back, result);
    }
}

mod risk_assessment {
    use super::*;

    #[test]
    fn defaults_are_safe_and_empty() {
        let assessment = RiskAssessment::default();
        assert!(!assessment.has_risk);
        assert_eq!(assessment.risk_level, RiskLevel::Safe);
        assert_eq!(assessment.risk_type, None);
        assert!(assessment.confidence.abs() < f64::EPSILON);
        assert_eq!(assessment.details, None);
    }

    #[test]
    fn empty_object_fills_the_defaults() {
        let assessment: RiskAssessment = serde_json::from_value(json!({})).unwrap();
        assert_eq!(assessment, RiskAssessment::default());
    }

    #[test]
    fn into_result_maps_a_safe_assessment_verbatim() {
        let assessment = RiskAssessment {
            has_risk: false,
            risk_level: RiskLevel::Low,
            risk_type: Some("spam".to_owned()),
            confidence: 0.1,
            details: Some(json!({"score": 0.1})),
        };
        let result = assessment.into_result();
        assert!(result.is_safe());
        assert_eq!(result.risk_level, RiskLevel::Low);
        assert_eq!(result.risk_type.as_deref(), Some("spam"));
        assert_eq!(result.details, Some(json!({"score": 0.1})));
        assert_eq!(result.modified_data, None);
    }

    #[test]
    fn into_result_maps_a_risky_assessment_and_drops_confidence() {
        let assessment = RiskAssessment {
            has_risk: true,
            risk_level: RiskLevel::High,
            risk_type: Some("attack_detected".to_owned()),
            confidence: 0.92,
            details: Some(json!({"score": 0.92, "label": "attack"})),
        };
        let result = assessment.into_result();
        assert!(!result.is_safe());
        assert_eq!(result.risk_level, RiskLevel::High);
        assert_eq!(result.risk_type.as_deref(), Some("attack_detected"));
        assert_eq!(
            result.details,
            Some(json!({"score": 0.92, "label": "attack"}))
        );
        assert_eq!(result.modified_data, None);
        // Confidence is not part of GuardrailResult: only the projected fields survive.
        let value = serde_json::to_value(&result).unwrap();
        assert_eq!(value.get("confidence"), None);
    }
}

mod outcome_serde {
    use super::*;

    #[test]
    fn allowed_outcome_json_round_trips() {
        let outcome = GuardrailOutcome::allowed(ScanStage::ToolInvokeOutput);
        let value = serde_json::to_value(&outcome).unwrap();
        assert_eq!(value["stage"], json!("_framework:tool_invoke_output"));
        assert_eq!(value["decision"], json!("allow"));
        assert_eq!(value["risk_level"], json!("safe"));
        let back: GuardrailOutcome = serde_json::from_value(value).unwrap();
        assert_eq!(back, outcome);
    }

    #[test]
    fn blocked_outcome_round_trips_with_findings_and_evaluations() {
        let mut outcome = GuardrailOutcome::allowed(ScanStage::UserInput);
        outcome.evaluations.push(GuardrailEvaluation {
            guardrail: "secret".to_owned(),
            result: Some(GuardrailResult::block(
                RiskLevel::Critical,
                "secret",
                Some(json!({"pattern": "aws_key"})),
                None,
            )),
            error: None,
        });
        outcome.findings.push(secret_finding());
        outcome.risk_level = RiskLevel::Critical;
        outcome.decision = SecurityDecision::Block;
        outcome.blocked_by = Some("secret".to_owned());

        let value = serde_json::to_value(&outcome).unwrap();
        assert_eq!(value["decision"], json!("block"));
        assert_eq!(value["risk_level"], json!("critical"));
        assert_eq!(value["blocked_by"], json!("secret"));
        assert_eq!(value["findings"][0]["blocking"], json!(true));

        let back: GuardrailOutcome = serde_json::from_value(value).unwrap();
        assert_eq!(back, outcome);
        assert!(back.findings[0].blocking);
        assert_eq!(back.blocked_by.as_deref(), Some("secret"));
        assert_eq!(back.evaluations.len(), 1);
    }

    #[test]
    fn security_decision_block_serializes_as_block() {
        assert_eq!(
            serde_json::to_value(SecurityDecision::Block).unwrap(),
            json!("block")
        );
        assert_eq!(
            serde_json::to_value(SecurityDecision::Allow).unwrap(),
            json!("allow")
        );
    }
}

mod redaction {
    use super::*;

    #[test]
    fn context_debug_redacts_content() {
        let ctx = GuardrailContext::text("secret sk-live-123456", ScanStage::UserInput);
        let debug = format!("{ctx:?}");
        assert!(debug.contains("<REDACTED>"), "{debug}");
        assert!(debug.contains("content_type"), "{debug}");
        assert!(debug.contains("user_input"), "{debug}");
        assert!(!debug.contains(SECRET), "{debug}");
    }

    #[test]
    fn outcome_debug_is_the_redacted_display_form() {
        let outcome = GuardrailOutcome {
            findings: vec![secret_finding()],
            risk_level: RiskLevel::Critical,
            ..GuardrailOutcome::allowed(ScanStage::UserInput)
        };
        let debug = format!("{outcome:?}");
        assert_eq!(
            debug,
            format!("GuardrailOutcome {{ {} }}", outcome.redacted()),
            "{debug}"
        );
        assert!(debug.starts_with("GuardrailOutcome {"), "{debug}");
        assert!(debug.contains("decision=allow"), "{debug}");
        assert!(!debug.contains(SECRET), "{debug}");
    }

    #[test]
    fn redacted_outcome_masks_masked_key_values() {
        let outcome = GuardrailOutcome {
            findings: vec![secret_finding()],
            risk_level: RiskLevel::Critical,
            ..GuardrailOutcome::allowed(ScanStage::UserInput)
        };
        let redacted: RedactedOutcome = outcome.redacted();
        let masked = redacted.findings[0].details.clone().unwrap();
        assert_ne!(
            masked,
            outcome.findings[0].details.clone().unwrap(),
            "redaction must change the details value"
        );
        assert_eq!(masked["query"], json!("<REDACTED>"));
        assert_eq!(masked["pattern"], json!("aws_key"));
        assert!(!masked.to_string().contains(SECRET), "{masked}");
    }

    #[test]
    fn redacted_outcome_display_shape_is_exact() {
        let outcome = GuardrailOutcome {
            evaluations: vec![GuardrailEvaluation {
                guardrail: "secret".to_owned(),
                result: Some(GuardrailResult::block(
                    RiskLevel::Critical,
                    "secret",
                    Some(json!({"query": SECRET})),
                    None,
                )),
                error: None,
            }],
            findings: vec![secret_finding()],
            risk_level: RiskLevel::Critical,
            decision: SecurityDecision::Block,
            blocked_by: Some("secret".to_owned()),
            ..GuardrailOutcome::allowed(ScanStage::UserInput)
        };
        assert_eq!(
            outcome.redacted().to_string(),
            "decision=block risk_level=CRITICAL stage=user_input findings=1 guardrails=secret"
        );
    }
}

mod violation_conversion {
    use super::*;

    #[test]
    fn violation_converts_to_a_security_error_with_the_message() {
        let message = "Critical security risk detected: jailbreak";
        let violation = GuardrailViolation {
            risk_level: RiskLevel::Critical,
            risk_type: "jailbreak".to_owned(),
            message: message.to_owned(),
            event: EVENT_LLM_INVOKE_OUTPUT.to_owned(),
            details: Some(json!({"pattern": "dan"})),
        };
        let error: Error = violation.into();
        assert_eq!(error.kind(), ErrorKind::Security);
        assert_eq!(error.message().as_ref(), message);
    }
}
