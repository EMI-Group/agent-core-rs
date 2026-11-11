//! Security policy and scan-target configuration.
//!
//! The policy decides which findings block a call and how guardrail failures are
//! handled; [`ScanTargets`] decides which call stages are scanned at all.
//! [`SecurityConfig`](crate::config) itself arrives with the pipeline in a later
//! phase.

use serde::{Deserialize, Serialize};

use crate::{RiskLevel, ScanStage};

/// How guardrail failures (errors, not findings) are treated.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FailurePolicy {
    /// Record the failure and continue; the default, and Python's behavior.
    #[default]
    FailOpen,
    /// Treat the failure as a blocking [`crate::RiskLevel::Critical`] finding.
    FailClosed,
}

/// Thresholds and switches of the content-safety pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SecurityPolicy {
    /// Findings at or above this level block the call.
    #[serde(default = "default_block_level")]
    pub block_level: RiskLevel,
    /// How guardrail failures are treated.
    pub failure_policy: FailurePolicy,
    /// Whether the remaining guardrails are skipped after a blocking finding.
    pub stop_on_block: bool,
}

impl Default for SecurityPolicy {
    /// Python's effective policy: block at `CRITICAL`, fail open, stop on block.
    fn default() -> Self {
        Self {
            block_level: default_block_level(),
            failure_policy: FailurePolicy::FailOpen,
            stop_on_block: true,
        }
    }
}

/// Python's blocking threshold, kept for the serde field default.
fn default_block_level() -> RiskLevel {
    RiskLevel::Critical
}

/// The call stages that are scanned.
///
/// Five independent on/off switches mirror the Python configuration; grouping
/// them is a deliberate parity choice, not an oversight.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ScanTargets {
    /// Scan the prompt sent to the model.
    pub prompts: bool,
    /// Scan raw user input.
    pub user_input: bool,
    /// Scan the model's response.
    pub model_outputs: bool,
    /// Scan tool-call arguments.
    pub tool_inputs: bool,
    /// Scan tool results.
    pub tool_outputs: bool,
}

impl Default for ScanTargets {
    /// Every target is enabled.
    fn default() -> Self {
        Self {
            prompts: true,
            user_input: true,
            model_outputs: true,
            tool_inputs: true,
            tool_outputs: true,
        }
    }
}

impl ScanTargets {
    /// The enabled stages, in pipeline order.
    #[must_use]
    pub fn stages(&self) -> Vec<ScanStage> {
        let mut stages = Vec::new();
        if self.prompts {
            stages.push(ScanStage::LlmInvokeInput);
        }
        if self.user_input {
            stages.push(ScanStage::UserInput);
        }
        if self.model_outputs {
            stages.push(ScanStage::LlmInvokeOutput);
        }
        if self.tool_inputs {
            stages.push(ScanStage::ToolInvokeInput);
        }
        if self.tool_outputs {
            stages.push(ScanStage::ToolInvokeOutput);
        }
        stages
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn policy_defaults_match_python() {
        let policy = SecurityPolicy::default();
        assert_eq!(policy.block_level, RiskLevel::Critical);
        assert_eq!(policy.failure_policy, FailurePolicy::FailOpen);
        assert!(policy.stop_on_block);
        assert_eq!(FailurePolicy::default(), FailurePolicy::FailOpen);
    }

    #[test]
    fn policy_deserializes_missing_fields_to_defaults() {
        let policy: SecurityPolicy = serde_json::from_value(json!({})).unwrap();
        assert_eq!(policy, SecurityPolicy::default());
        assert_eq!(policy.block_level, RiskLevel::Critical);

        let lowered: SecurityPolicy = serde_json::from_value(json!({
            "block_level": "high",
            "failure_policy": "fail_closed",
            "stop_on_block": false
        }))
        .unwrap();
        assert_eq!(lowered.block_level, RiskLevel::High);
        assert_eq!(lowered.failure_policy, FailurePolicy::FailClosed);
        assert!(!lowered.stop_on_block);

        let partial: SecurityPolicy =
            serde_json::from_value(json!({"stop_on_block": false})).unwrap();
        assert_eq!(partial.block_level, RiskLevel::Critical);
        assert!(!partial.stop_on_block);
    }

    #[test]
    fn policy_serde_round_trip() {
        let policy = SecurityPolicy {
            block_level: RiskLevel::Medium,
            failure_policy: FailurePolicy::FailClosed,
            stop_on_block: false,
        };
        let value = serde_json::to_value(policy).unwrap();
        assert_eq!(value["block_level"], json!("medium"));
        assert_eq!(value["failure_policy"], json!("fail_closed"));
        let back: SecurityPolicy = serde_json::from_value(value).unwrap();
        assert_eq!(back, policy);
    }

    #[test]
    fn scan_targets_default_to_all_enabled() {
        let targets = ScanTargets::default();
        assert!(targets.prompts);
        assert!(targets.user_input);
        assert!(targets.model_outputs);
        assert!(targets.tool_inputs);
        assert!(targets.tool_outputs);
        assert_eq!(
            targets.stages(),
            vec![
                ScanStage::LlmInvokeInput,
                ScanStage::UserInput,
                ScanStage::LlmInvokeOutput,
                ScanStage::ToolInvokeInput,
                ScanStage::ToolInvokeOutput,
            ]
        );
    }

    #[test]
    fn scan_targets_filter_stages() {
        let targets = ScanTargets {
            prompts: false,
            user_input: true,
            model_outputs: false,
            tool_inputs: true,
            tool_outputs: false,
        };
        assert_eq!(
            targets.stages(),
            vec![ScanStage::UserInput, ScanStage::ToolInvokeInput]
        );

        let none = ScanTargets {
            prompts: false,
            user_input: false,
            model_outputs: false,
            tool_inputs: false,
            tool_outputs: false,
        };
        assert_eq!(none.stages(), Vec::<ScanStage>::new());

        let partial: ScanTargets = serde_json::from_value(json!({"tool_outputs": false})).unwrap();
        assert!(partial.prompts);
        assert!(partial.tool_inputs);
        assert!(!partial.tool_outputs);
    }
}
