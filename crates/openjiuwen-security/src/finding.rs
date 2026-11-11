//! Findings, evaluations, and the aggregated outcome of one scan.
//!
//! Python never aggregates: each guardrail raises (`AbortError` at
//! `CRITICAL`, a swallowed `GuardrailError` below it). The outcome here is the
//! one ordered result of a whole [`crate::GuardrailPipeline`] run: what each
//! guardrail said, which findings block the caller, and whether the call is
//! allowed.

use std::fmt;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use openjiuwen_common::error::{Error, Result};

use crate::{GuardrailResult, RiskLevel, ScanStage};

/// One recorded detection.
///
/// The message wording is the Python contract, built here by
/// [`finding_message`] so the pipeline and every later caller agree:
///
/// - blocking at [`RiskLevel::Critical`] → `"Critical security risk detected:
///   <risk_type>"` (Python `AbortError` reason);
/// - blocking at a lowered [`crate::SecurityPolicy::block_level`] →
///   `"Guardrail blocked: <risk_type> risk detected"` (Python
///   `GuardrailError` message);
/// - non-blocking → `"Security finding: <risk_type> (level=<LEVEL>)"`.
///
/// `details` and `modified_data` carry pattern names, rule names, counts, and
/// scores — never raw matched text.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "snake_case")]
pub struct Finding {
    /// Name of the guardrail that produced the finding.
    pub guardrail: String,
    /// Severity the guardrail reported.
    pub risk_level: RiskLevel,
    /// Machine-readable risk name, `"unknown"` when the guardrail gave none.
    pub risk_type: String,
    /// Human-readable summary, built by [`finding_message`].
    pub message: String,
    /// Whether this finding aborts the guarded call under the run's policy.
    pub blocking: bool,
    /// Non-sensitive evidence from the guardrail.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<Value>,
    /// Sanitized content the guardrail proposed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub modified_data: Option<Value>,
}

/// Builds the canonical message for a finding.
///
/// Kept in one place so the pipeline, later integrations, and serialized
/// findings never disagree on wording.
#[must_use]
pub fn finding_message(risk_type: &str, risk_level: RiskLevel, blocking: bool) -> String {
    if !blocking {
        return format!(
            "Security finding: {risk_type} (level={})",
            risk_level.name()
        );
    }
    if risk_level == RiskLevel::Critical {
        format!("Critical security risk detected: {risk_type}")
    } else {
        format!("Guardrail blocked: {risk_type} risk detected")
    }
}

/// The pipeline's verdict for a scan.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SecurityDecision {
    /// The caller may continue; findings may still be recorded.
    #[default]
    Allow,
    /// The caller must abort with a [`GuardrailViolation`].
    Block,
}

/// What one guardrail said, successfully or not.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "snake_case")]
pub struct GuardrailEvaluation {
    /// Name of the guardrail.
    pub guardrail: String,
    /// Its verdict, when it returned one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<GuardrailResult>,
    /// Its failure, rendered as `"<name>: <error>"`, when it failed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// The aggregated result of one pipeline run.
///
/// `stage` is always serialized (it is the event string of the scan); every
/// other field defaults when absent, so an older or partial payload still
/// deserializes.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct GuardrailOutcome {
    /// The stage that was scanned.
    pub stage: ScanStage,
    /// One entry per guardrail that ran, in run order.
    #[serde(default)]
    pub evaluations: Vec<GuardrailEvaluation>,
    /// One entry per unsafe verdict, in run order.
    #[serde(default)]
    pub findings: Vec<Finding>,
    /// Failures, rendered as `"<name>: <error>"`, in run order.
    #[serde(default)]
    pub errors: Vec<String>,
    /// The highest level across every finding, [`RiskLevel::Safe`] when none.
    #[serde(default)]
    pub risk_level: RiskLevel,
    /// The aggregated verdict.
    #[serde(default)]
    pub decision: SecurityDecision,
    /// Name of the guardrail whose finding blocked the call, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blocked_by: Option<String>,
}

impl GuardrailOutcome {
    /// An outcome that allows the call and carries nothing else.
    #[must_use]
    pub fn allowed(stage: ScanStage) -> Self {
        Self {
            stage,
            evaluations: Vec::new(),
            findings: Vec::new(),
            errors: Vec::new(),
            risk_level: RiskLevel::Safe,
            decision: SecurityDecision::Allow,
            blocked_by: None,
        }
    }

    /// Whether the call may continue.
    #[must_use]
    pub fn is_safe(&self) -> bool {
        self.decision == SecurityDecision::Allow
    }

    /// Whether the call must abort.
    #[must_use]
    pub fn is_blocked(&self) -> bool {
        self.decision == SecurityDecision::Block
    }

    /// Converts a blocking outcome into an [`Error::Security`].
    ///
    /// # Errors
    ///
    /// Returns the first blocking violation as [`Error::Security`] when the
    /// outcome is blocked, and `Ok(())` when it is allowed. A blocked outcome
    /// without a violation (which the pipeline never produces) still fails with
    /// `"guardrail blocked execution"`.
    pub fn enforce(&self) -> Result<()> {
        if self.is_safe() {
            return Ok(());
        }
        match self.violation() {
            Some(violation) => Err(violation.into()),
            None => Err(Error::security("guardrail blocked execution")),
        }
    }

    /// The first blocking finding as a [`GuardrailViolation`].
    ///
    /// A blocked outcome without a blocking finding falls back to
    /// [`Self::blocked_by`], reported as `risk_type = "unknown"`.
    #[must_use]
    pub fn violation(&self) -> Option<GuardrailViolation> {
        if let Some(finding) = self.findings.iter().find(|finding| finding.blocking) {
            return Some(GuardrailViolation {
                risk_level: finding.risk_level,
                risk_type: finding.risk_type.clone(),
                message: finding.message.clone(),
                event: self.stage.event().to_owned(),
                details: finding.details.clone(),
            });
        }
        self.blocked_by.as_ref().map(|_| GuardrailViolation {
            risk_level: self.risk_level,
            risk_type: "unknown".to_owned(),
            message: "Guardrail blocked execution".to_owned(),
            event: self.stage.event().to_owned(),
            details: None,
        })
    }

    /// The merged sanitized content every guardrail proposed.
    ///
    /// Contributions are merged in evaluation order, so a later guardrail wins
    /// on a shared key. Non-object `modified_data` values are ignored, and the
    /// result is `None` when nothing contributed.
    #[must_use]
    pub fn modified_data(&self) -> Option<Value> {
        let mut merged = Map::new();
        let mut contributed = false;
        for evaluation in &self.evaluations {
            if let Some(Value::Object(map)) = evaluation
                .result
                .as_ref()
                .and_then(|result| result.modified_data.as_ref())
            {
                contributed = true;
                for (key, value) in map {
                    merged.insert(key.clone(), value.clone());
                }
            }
        }
        contributed.then_some(Value::Object(merged))
    }
}

impl fmt::Debug for GuardrailOutcome {
    /// Prints the redacted summary so raw details can never leak through a log
    /// line that formats an outcome.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GuardrailOutcome {{ {} }}", self.redacted())
    }
}

/// The blocking finding of an outcome, with the scan event it was detected at.
///
/// This is the value a caller raises (or matches on) when it chooses not to use
/// [`GuardrailOutcome::enforce`]; converted with `?` it becomes
/// [`openjiuwen_common::error::Error::Security`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct GuardrailViolation {
    /// Severity of the blocking finding.
    pub risk_level: RiskLevel,
    /// Machine-readable risk name.
    pub risk_type: String,
    /// Message from [`finding_message`].
    pub message: String,
    /// The callback event of the scanned stage.
    pub event: String,
    /// Non-sensitive evidence.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<Value>,
}

impl From<GuardrailViolation> for Error {
    fn from(violation: GuardrailViolation) -> Self {
        Self::security(violation.message)
    }
}

#[cfg(test)]
mod tests {
    use openjiuwen_common::error::ErrorKind;
    use serde_json::json;

    use super::*;

    fn evaluation(name: &str, result: GuardrailResult) -> GuardrailEvaluation {
        GuardrailEvaluation {
            guardrail: name.to_owned(),
            result: Some(result),
            error: None,
        }
    }

    #[test]
    fn allowed_outcome_shape() {
        let outcome = GuardrailOutcome::allowed(ScanStage::UserInput);
        assert!(outcome.is_safe());
        assert!(!outcome.is_blocked());
        assert_eq!(outcome.risk_level, RiskLevel::Safe);
        assert_eq!(outcome.blocked_by, None);
        assert_eq!(outcome.evaluations.len(), 0);
        assert_eq!(outcome.findings.len(), 0);
        assert_eq!(outcome.errors.len(), 0);
        assert_eq!(outcome.modified_data(), None);
        assert_eq!(outcome.violation(), None);
        assert!(outcome.enforce().is_ok());
        assert_eq!(SecurityDecision::default(), SecurityDecision::Allow);
    }

    #[test]
    fn blocked_outcome_enforces_the_violation_message() {
        let outcome = GuardrailOutcome {
            stage: ScanStage::LlmInvokeOutput,
            findings: vec![Finding {
                guardrail: "jailbreak".to_owned(),
                risk_level: RiskLevel::Critical,
                risk_type: "jailbreak".to_owned(),
                message: finding_message("jailbreak", RiskLevel::Critical, true),
                blocking: true,
                details: Some(json!({"pattern": "dan"})),
                modified_data: None,
            }],
            risk_level: RiskLevel::Critical,
            decision: SecurityDecision::Block,
            blocked_by: Some("jailbreak".to_owned()),
            ..GuardrailOutcome::allowed(ScanStage::LlmInvokeOutput)
        };
        assert!(outcome.is_blocked());
        assert!(!outcome.is_safe());

        let error = outcome.enforce().unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Security);
        assert_eq!(
            error.to_string(),
            "security: Critical security risk detected: jailbreak"
        );

        let violation = outcome.violation().unwrap();
        assert_eq!(violation.event, crate::EVENT_LLM_INVOKE_OUTPUT);
        assert_eq!(violation.details, Some(json!({"pattern": "dan"})));
        let from_violation: Error = violation.into();
        assert_eq!(from_violation.kind(), ErrorKind::Security);
    }

    #[test]
    fn violation_picks_the_first_blocking_finding() {
        let non_blocking = Finding {
            guardrail: "pii".to_owned(),
            risk_level: RiskLevel::High,
            risk_type: "pii".to_owned(),
            message: finding_message("pii", RiskLevel::High, true),
            blocking: false,
            details: None,
            modified_data: None,
        };
        let blocking = Finding {
            guardrail: "shell".to_owned(),
            risk_level: RiskLevel::Critical,
            risk_type: "shell_injection".to_owned(),
            message: finding_message("shell_injection", RiskLevel::Critical, true),
            blocking: true,
            ..non_blocking.clone()
        };
        let outcome = GuardrailOutcome {
            findings: vec![non_blocking, blocking.clone()],
            risk_level: RiskLevel::Critical,
            decision: SecurityDecision::Block,
            blocked_by: Some("shell".to_owned()),
            ..GuardrailOutcome::allowed(ScanStage::ToolInvokeInput)
        };
        let violation = outcome.violation().unwrap();
        assert_eq!(violation.risk_type, "shell_injection");
        assert_eq!(violation.message, blocking.message);
        assert_eq!(violation.event, crate::EVENT_TOOL_INVOKE_INPUT);
        assert_eq!(violation.details, None);
    }

    #[test]
    fn violation_falls_back_to_blocked_by() {
        let outcome = GuardrailOutcome {
            decision: SecurityDecision::Block,
            risk_level: RiskLevel::High,
            blocked_by: Some("custom".to_owned()),
            ..GuardrailOutcome::allowed(ScanStage::UserInput)
        };
        let violation = outcome.violation().unwrap();
        assert_eq!(violation.risk_type, "unknown");
        assert_eq!(violation.message, "Guardrail blocked execution");
        assert_eq!(violation.risk_level, RiskLevel::High);
        assert_eq!(
            outcome.enforce().unwrap_err().to_string(),
            "security: Guardrail blocked execution"
        );
    }

    #[test]
    fn modified_data_merges_in_evaluation_order() {
        let outcome = GuardrailOutcome {
            evaluations: vec![
                evaluation(
                    "a",
                    GuardrailResult::block(
                        RiskLevel::High,
                        "first",
                        None,
                        Some(json!({"text": "one", "keep": 1})),
                    ),
                ),
                evaluation(
                    "b",
                    GuardrailResult::block(
                        RiskLevel::High,
                        "second",
                        None,
                        Some(json!({"text": "two"})),
                    ),
                ),
            ],
            ..GuardrailOutcome::allowed(ScanStage::UserInput)
        };
        assert_eq!(
            outcome.modified_data(),
            Some(json!({"text": "two", "keep": 1})),
            "later guardrails win on shared keys"
        );
    }

    #[test]
    fn modified_data_ignores_non_objects_and_empty_input() {
        let outcome = GuardrailOutcome {
            evaluations: vec![
                evaluation(
                    "a",
                    GuardrailResult::block(RiskLevel::Low, "x", None, Some(json!("plain"))),
                ),
                evaluation("b", GuardrailResult::pass()),
            ],
            ..GuardrailOutcome::allowed(ScanStage::UserInput)
        };
        assert_eq!(outcome.modified_data(), None);

        let objects = GuardrailOutcome {
            evaluations: vec![
                evaluation(
                    "a",
                    GuardrailResult::block(RiskLevel::Low, "x", None, Some(json!({"k": 1}))),
                ),
                evaluation(
                    "b",
                    GuardrailResult::block(RiskLevel::Low, "x", None, Some(json!({}))),
                ),
            ],
            ..GuardrailOutcome::allowed(ScanStage::UserInput)
        };
        assert_eq!(objects.modified_data(), Some(json!({"k": 1})));
    }

    #[test]
    fn debug_uses_the_redacted_summary() {
        let outcome = GuardrailOutcome {
            findings: vec![Finding {
                guardrail: "secret".to_owned(),
                risk_level: RiskLevel::Critical,
                risk_type: "secret".to_owned(),
                message: finding_message("secret", RiskLevel::Critical, true),
                blocking: true,
                details: Some(json!({"query": "sk-live-1234567890"})),
                modified_data: None,
            }],
            risk_level: RiskLevel::Critical,
            decision: SecurityDecision::Block,
            blocked_by: Some("secret".to_owned()),
            ..GuardrailOutcome::allowed(ScanStage::UserInput)
        };
        let debug = format!("{outcome:?}");
        assert!(debug.starts_with("GuardrailOutcome {"), "{debug}");
        assert!(debug.contains("decision=block"), "{debug}");
        assert!(!debug.contains("sk-live-1234567890"), "{debug}");
    }

    #[test]
    fn finding_message_matches_python_wording() {
        assert_eq!(
            finding_message("jailbreak", RiskLevel::Critical, true),
            "Critical security risk detected: jailbreak"
        );
        assert_eq!(
            finding_message("pii", RiskLevel::High, true),
            "Guardrail blocked: pii risk detected"
        );
        assert_eq!(
            finding_message("pii", RiskLevel::High, false),
            "Security finding: pii (level=HIGH)"
        );
    }

    #[test]
    fn serde_keeps_optional_fields_absent_and_stage_required() {
        let outcome = GuardrailOutcome::allowed(ScanStage::UserInput);
        let value = serde_json::to_value(&outcome).unwrap();
        assert_eq!(value["stage"], json!("user_input"));
        assert_eq!(value["decision"], json!("allow"));
        assert_eq!(value["risk_level"], json!("safe"));
        assert_eq!(value.get("blocked_by"), None);
        assert_eq!(value["findings"], json!([]));

        let back: GuardrailOutcome = serde_json::from_value(value).unwrap();
        assert_eq!(back, outcome);

        let partial: GuardrailOutcome =
            serde_json::from_value(json!({"stage": "user_input"})).unwrap();
        assert_eq!(partial, outcome);

        let missing = serde_json::from_value::<GuardrailOutcome>(json!({})).unwrap_err();
        assert!(missing.to_string().contains("stage"), "{missing}");

        let finding = Finding {
            guardrail: "g".to_owned(),
            risk_level: RiskLevel::High,
            risk_type: "pii".to_owned(),
            message: "m".to_owned(),
            blocking: false,
            details: None,
            modified_data: None,
        };
        let value = serde_json::to_value(&finding).unwrap();
        assert_eq!(value.get("details"), None);
        assert_eq!(value.get("modified_data"), None);
        let empty: Finding = serde_json::from_value(json!({})).unwrap();
        assert_eq!(empty, Finding::default());
    }
}
