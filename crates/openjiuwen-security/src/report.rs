//! Redacted rendering of a [`GuardrailOutcome`].
//!
//! Findings and errors never carry raw matched text by construction, but
//! `details` may still hold caller-supplied text. [`RedactedOutcome`] is the
//! projection that is safe to log, serialize into a trace, or show in a CLI.

use std::fmt;

use openjiuwen_common::security::redact_json;
use serde::{Deserialize, Serialize};

use crate::{Finding, GuardrailOutcome, RiskLevel, ScanStage, SecurityDecision};

/// Longest error string kept in a [`RedactedOutcome`].
const MAX_ERROR_CHARS: usize = 200;

/// Content-free summary of one pipeline run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct RedactedOutcome {
    /// The aggregated verdict.
    pub decision: SecurityDecision,
    /// The highest level across every finding.
    pub risk_level: RiskLevel,
    /// Findings with every `details` and `modified_data` value redacted.
    pub findings: Vec<Finding>,
    /// Failures, shortened to [`MAX_ERROR_CHARS`] characters.
    pub errors: Vec<String>,
    /// Names of the guardrails that ran, in run order.
    pub guardrails: Vec<String>,
    /// The stage that was scanned.
    pub stage: ScanStage,
}

impl fmt::Display for RedactedOutcome {
    /// `decision=<allow|block> risk_level=<LEVEL> stage=<event> findings=<n>
    /// guardrails=<comma list>`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let decision = match self.decision {
            SecurityDecision::Allow => "allow",
            SecurityDecision::Block => "block",
        };
        let guardrails = self.guardrails.join(", ");
        write!(
            f,
            "decision={decision} risk_level={} stage={} findings={} guardrails={guardrails}",
            self.risk_level.name(),
            self.stage.event(),
            self.findings.len()
        )
    }
}

impl GuardrailOutcome {
    /// Projects the outcome into a content-free [`RedactedOutcome`].
    ///
    /// Every `details` and `modified_data` value passes through
    /// [`redact_json`] with the default sensitive-field list, so common leaks
    /// (`query`, `arguments`, `messages`, ...) are masked even if a guardrail
    /// put them there. Errors are truncated char-safely, not grepped.
    #[must_use]
    pub fn redacted(&self) -> RedactedOutcome {
        RedactedOutcome {
            decision: self.decision,
            risk_level: self.risk_level,
            findings: self
                .findings
                .iter()
                .map(|finding| Finding {
                    details: finding
                        .details
                        .as_ref()
                        .map(|value| redact_json(value, None)),
                    modified_data: finding
                        .modified_data
                        .as_ref()
                        .map(|value| redact_json(value, None)),
                    ..finding.clone()
                })
                .collect(),
            errors: self
                .errors
                .iter()
                .map(|error| error.chars().take(MAX_ERROR_CHARS).collect())
                .collect(),
            guardrails: self
                .evaluations
                .iter()
                .map(|evaluation| evaluation.guardrail.clone())
                .collect(),
            stage: self.stage.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::finding::finding_message;
    use crate::{GuardrailEvaluation, GuardrailResult};

    fn outcome(decision: SecurityDecision) -> GuardrailOutcome {
        GuardrailOutcome {
            evaluations: vec![GuardrailEvaluation {
                guardrail: "pii".to_owned(),
                result: Some(GuardrailResult::pass()),
                error: None,
            }],
            decision,
            ..GuardrailOutcome::allowed(ScanStage::UserInput)
        }
    }

    #[test]
    fn display_matches_the_documented_shape() {
        assert_eq!(
            outcome(SecurityDecision::Allow).redacted().to_string(),
            "decision=allow risk_level=SAFE stage=user_input findings=0 guardrails=pii"
        );
        assert_eq!(
            outcome(SecurityDecision::Block).redacted().to_string(),
            "decision=block risk_level=SAFE stage=user_input findings=0 guardrails=pii"
        );
    }

    #[test]
    fn display_keeps_an_empty_guardrail_list() {
        let empty = GuardrailOutcome::allowed(ScanStage::ToolInvokeOutput);
        assert_eq!(
            empty.redacted().to_string(),
            "decision=allow risk_level=SAFE stage=_framework:tool_invoke_output findings=0 guardrails="
        );
    }

    #[test]
    fn details_are_redacted() {
        let secret = "sk-live-abcdef123456";
        let blocked = GuardrailOutcome {
            findings: vec![Finding {
                guardrail: "secret".to_owned(),
                risk_level: RiskLevel::Critical,
                risk_type: "secret".to_owned(),
                message: finding_message("secret", RiskLevel::Critical, true),
                blocking: true,
                details: Some(json!({"query": secret, "pattern": "aws_key"})),
                modified_data: Some(json!({"messages": secret})),
            }],
            risk_level: RiskLevel::Critical,
            decision: SecurityDecision::Block,
            blocked_by: Some("secret".to_owned()),
            ..GuardrailOutcome::allowed(ScanStage::UserInput)
        };

        let redacted = blocked.redacted();
        let details = redacted.findings[0].details.clone().unwrap();
        assert_ne!(details, blocked.findings[0].details.clone().unwrap());
        assert_eq!(details["query"], json!("<REDACTED>"));
        assert_eq!(details["pattern"], json!("aws_key"));
        assert!(!details.to_string().contains(secret));

        let modified = redacted.findings[0].modified_data.clone().unwrap();
        assert_eq!(modified["messages"], json!("<REDACTED>"));
        assert!(!modified.to_string().contains(secret));

        let debug = format!("{blocked:?}");
        assert!(!debug.contains(secret), "{debug}");
        assert!(debug.contains("findings=1"), "{debug}");
    }

    #[test]
    fn errors_are_truncated_char_safely() {
        let long = "é".repeat(MAX_ERROR_CHARS + 50);
        let blocked = GuardrailOutcome {
            errors: vec![format!("guardrail: {long}")],
            ..GuardrailOutcome::allowed(ScanStage::UserInput)
        };
        let redacted = blocked.redacted();
        assert_eq!(redacted.errors.len(), 1);
        assert_eq!(redacted.errors[0].chars().count(), MAX_ERROR_CHARS);
        let expected: String = format!("guardrail: {long}")
            .chars()
            .take(MAX_ERROR_CHARS)
            .collect();
        assert_eq!(redacted.errors[0], expected);
    }

    #[test]
    fn redacted_outcome_round_trips_through_json() {
        let redacted = outcome(SecurityDecision::Allow).redacted();
        let value = serde_json::to_value(&redacted).unwrap();
        assert_eq!(value["decision"], json!("allow"));
        assert_eq!(value["stage"], json!("user_input"));
        assert_eq!(value["guardrails"], json!(["pii"]));
        let back: RedactedOutcome = serde_json::from_value(value).unwrap();
        assert_eq!(back, redacted);
    }
}
