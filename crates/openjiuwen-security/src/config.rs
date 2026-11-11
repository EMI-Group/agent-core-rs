//! Security policy, scan-target, and pipeline configuration.
//!
//! The policy decides which findings block a call and how guardrail failures are
//! handled; [`ScanTargets`] decides which call stages are scanned at all.
//! [`SecurityConfig`] combines those with every built-in detector's
//! configuration and builds the ordered [`GuardrailPipeline`].
#![allow(clippy::module_name_repetitions)]

use openjiuwen_common::error::Result;
use serde::{Deserialize, Serialize};

use crate::guardrails::content_rules::{ContentRuleGuardrail, ContentRuleGuardrailConfig};
use crate::guardrails::prompt_injection::{
    PromptInjectionGuardrail, PromptInjectionGuardrailConfig,
};
use crate::guardrails::sensitive_data::{SensitiveDataGuardrail, SensitiveDataGuardrailConfig};
use crate::guardrails::tool_arguments::{ToolArgumentGuardrail, ToolArgumentGuardrailConfig};
use crate::{GuardrailPipeline, RiskLevel, ScanStage};

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

/// The umbrella configuration of the content-safety pipeline.
///
/// [`SecurityConfig::default`] disables security entirely (`enabled` is `false`),
/// so a caller opts in with [`SecurityConfig::protected`] or by enabling it and
/// supplying its own sub-configs. [`SecurityConfig::pipeline`] turns the record
/// into an ordered [`GuardrailPipeline`].
///
/// The detector configs `prompt_injection`, `sensitive_data`, `content_rules`,
/// and `tool_arguments` are only consulted when [`Self::enabled`] is `true`;
/// with security disabled they still round-trip through JSON unchanged.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "snake_case")]
pub struct SecurityConfig {
    /// Whether security is enabled at all; `false` means "security disabled".
    ///
    /// A disabled config still carries its [`SecurityConfig::policy`] into
    /// [`SecurityConfig::pipeline`], but no guardrail is registered, so the
    /// resulting pipeline never produces a finding.
    pub enabled: bool,
    /// Thresholds and failure handling applied to every scan.
    pub policy: SecurityPolicy,
    /// The call stages that are scanned.
    pub targets: ScanTargets,
    /// Prompt-injection detector configuration, consulted only when
    /// [`Self::enabled`] is `true`.
    pub prompt_injection: PromptInjectionGuardrailConfig,
    /// Sensitive-data/PII detector configuration, consulted only when
    /// [`Self::enabled`] is `true` and its own `enabled` flag is set.
    pub sensitive_data: SensitiveDataGuardrailConfig,
    /// User-supplied content-rule configuration, consulted only when
    /// [`Self::enabled`] is `true` and its rule table is non-empty.
    pub content_rules: ContentRuleGuardrailConfig,
    /// Tool-argument validator configuration, consulted only when
    /// [`Self::enabled`] is `true`, [`ScanTargets::tool_inputs`] is set, and the
    /// config carries at least one rule.
    pub tool_arguments: ToolArgumentGuardrailConfig,
}

impl Default for SecurityConfig {
    /// Security disabled, with every policy, target, and detector at its
    /// documented default.
    ///
    /// A missing or partial JSON payload therefore reconstitutes exactly these
    /// values instead of enabling detection by accident.
    fn default() -> Self {
        Self {
            enabled: false,
            policy: SecurityPolicy::default(),
            targets: ScanTargets::default(),
            prompt_injection: PromptInjectionGuardrailConfig::default(),
            sensitive_data: SensitiveDataGuardrailConfig::default(),
            content_rules: ContentRuleGuardrailConfig::default(),
            tool_arguments: ToolArgumentGuardrailConfig::default(),
        }
    }
}

impl SecurityConfig {
    /// The default config with security enabled, stock detectors included.
    ///
    /// This is the conventional opt-in (`DEFAULT_ENABLED` in Python terms): the
    /// policy, the scan targets, and every detector config keep their defaults,
    /// so [`Self::pipeline`] registers prompt-injection, sensitive-data, and
    /// tool-argument detection (content rules stay out until a rule is added).
    #[must_use]
    pub fn protected() -> Self {
        Self {
            enabled: true,
            ..Self::default()
        }
    }

    /// Builds the ordered pipeline this config describes.
    ///
    /// Guardrails are registered in a fixed order — prompt injection, sensitive
    /// data, content rules, tool arguments — and every built-in uses the same
    /// priority, so registration order is the run order for equal priorities.
    ///
    /// Detector selection:
    ///
    /// - disabled config: no guardrails, policy preserved;
    /// - no enabled [`ScanTargets`]: no guardrails (an empty stage list would
    ///   otherwise mean "every stage");
    /// - prompt injection only at the configured intersection of
    ///   [`PromptInjectionGuardrail::DEFAULT_STAGES`] and the enabled stages;
    /// - sensitive data only when its own `enabled` flag is set, at every
    ///   enabled stage;
    /// - content rules only when the rule table is non-empty, at every enabled
    ///   stage;
    /// - tool arguments only when tool inputs are scanned and the config has a
    ///   rule to apply, at [`ScanStage::ToolInvokeInput`].
    ///
    /// # Errors
    ///
    /// Propagates a [`openjiuwen_common::error::ErrorKind::Validation`] error
    /// unchanged when a detector rejects its configuration: an unknown
    /// prompt-injection mode, a missing endpoint for `api` mode, or an invalid
    /// regular expression.
    pub fn pipeline(&self) -> Result<GuardrailPipeline> {
        let mut pipeline = GuardrailPipeline::new().with_policy(self.policy);
        if !self.enabled {
            return Ok(pipeline);
        }
        let stages = self.targets.stages();
        if stages.is_empty() {
            return Ok(pipeline);
        }

        let prompt_stages = intersect_stages(PromptInjectionGuardrail::DEFAULT_STAGES, &stages);
        if !prompt_stages.is_empty() {
            pipeline.add(
                PromptInjectionGuardrail::with_config(self.prompt_injection.clone())?
                    .with_stages(prompt_stages),
            );
        }

        if self.sensitive_data.enabled {
            pipeline.add(
                SensitiveDataGuardrail::with_config(self.sensitive_data.clone())?
                    .with_stages(stages.clone()),
            );
        }

        if !self.content_rules.rules.is_empty() {
            pipeline.add(
                ContentRuleGuardrail::new(self.content_rules.clone())?.with_stages(stages.clone()),
            );
        }

        if self.targets.tool_inputs && has_tool_argument_rules(&self.tool_arguments) {
            pipeline.add(
                ToolArgumentGuardrail::with_config(self.tool_arguments.clone())?
                    .with_stages([ScanStage::ToolInvokeInput]),
            );
        }

        Ok(pipeline)
    }
}

/// `reference` stages that also appear in `enabled`, in `reference` order.
fn intersect_stages(reference: &[ScanStage], enabled: &[ScanStage]) -> Vec<ScanStage> {
    reference
        .iter()
        .filter(|stage| enabled.contains(stage))
        .cloned()
        .collect()
}

/// Whether a tool-argument config has anything to enforce.
///
/// The guardrail itself always applies its (possibly empty) rule set; the
/// builder only installs it when at least one rule, list, or limit is set, so a
/// stock config does not add a no-op guardrail to the pipeline.
fn has_tool_argument_rules(config: &ToolArgumentGuardrailConfig) -> bool {
    !config.allowed_tools.is_empty()
        || !config.denied_tools.is_empty()
        || !config.deny_patterns.is_empty()
        || config.sensitive_data.is_some()
        || config.max_argument_bytes.is_some()
}

#[cfg(test)]
mod tests {
    use openjiuwen_common::error::ErrorKind;
    use serde_json::json;

    use super::*;
    use crate::guardrails::content_rules::ContentRule;

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

    #[test]
    fn default_security_config_is_disabled() {
        let config = SecurityConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.policy, SecurityPolicy::default());
        assert_eq!(config.targets, ScanTargets::default());
        assert_eq!(
            config.prompt_injection,
            PromptInjectionGuardrailConfig::default()
        );
        assert_eq!(
            config.sensitive_data,
            SensitiveDataGuardrailConfig::default()
        );
        assert_eq!(config.content_rules, ContentRuleGuardrailConfig::default());
        assert_eq!(
            config.tool_arguments,
            ToolArgumentGuardrailConfig::default()
        );

        let pipeline = config.pipeline().unwrap();
        assert!(pipeline.guardrails().is_empty());
        assert_eq!(pipeline.policy(), &SecurityPolicy::default());

        let policy = SecurityPolicy {
            block_level: RiskLevel::Medium,
            failure_policy: FailurePolicy::FailClosed,
            stop_on_block: false,
        };
        let pipeline = SecurityConfig {
            policy,
            ..SecurityConfig::default()
        }
        .pipeline()
        .unwrap();
        assert!(
            pipeline.guardrails().is_empty(),
            "a disabled config registers no guardrails"
        );
        assert_eq!(pipeline.policy(), &policy, "the policy is still preserved");
    }

    #[test]
    fn disabled_targets_register_no_guardrails() {
        let config = SecurityConfig {
            targets: ScanTargets {
                prompts: false,
                user_input: false,
                model_outputs: false,
                tool_inputs: false,
                tool_outputs: false,
            },
            ..SecurityConfig::protected()
        };
        assert_eq!(config.targets.stages(), Vec::<ScanStage>::new());
        let pipeline = config.pipeline().unwrap();
        assert!(pipeline.guardrails().is_empty());
        assert_eq!(pipeline.policy(), &SecurityPolicy::default());
    }

    #[test]
    fn protected_registers_the_built_in_guardrails_in_order() {
        let stock = SecurityConfig::protected().pipeline().unwrap();
        let names: Vec<&str> = stock.guardrails().iter().map(|g| g.name()).collect();
        assert_eq!(
            names,
            vec!["prompt_injection", "sensitive_data", "tool_arguments"],
            "an empty content-rule table registers no content guardrail"
        );

        let config = SecurityConfig {
            content_rules: ContentRuleGuardrailConfig {
                rules: vec![ContentRule {
                    name: "banned".to_owned(),
                    pattern: "forbidden".to_owned(),
                    risk_type: "banned_word".to_owned(),
                    ..ContentRule::default()
                }],
                ..ContentRuleGuardrailConfig::default()
            },
            ..SecurityConfig::protected()
        };
        let pipeline = config.pipeline().unwrap();
        let guardrails = pipeline.guardrails();
        let names: Vec<&str> = guardrails.iter().map(|g| g.name()).collect();
        assert_eq!(
            names,
            vec![
                "prompt_injection",
                "sensitive_data",
                "content_rules",
                "tool_arguments"
            ],
            "registration order is the documented run order"
        );

        let all_stages = config.targets.stages();
        assert_eq!(
            guardrails[0].stages(),
            &[ScanStage::LlmInvokeInput, ScanStage::ToolInvokeOutput],
            "prompt injection keeps DEFAULT_STAGES order"
        );
        assert_eq!(guardrails[1].stages(), all_stages.as_slice());
        assert_eq!(guardrails[2].stages(), all_stages.as_slice());
        assert_eq!(guardrails[3].stages(), &[ScanStage::ToolInvokeInput]);
    }

    #[test]
    fn invalid_prompt_injection_mode_is_a_validation_error() {
        let config = SecurityConfig {
            prompt_injection: PromptInjectionGuardrailConfig {
                mode: "bogus".to_owned(),
                ..PromptInjectionGuardrailConfig::default()
            },
            ..SecurityConfig::protected()
        };
        let Err(error) = config.pipeline() else {
            panic!("an invalid prompt-injection mode must fail construction");
        };
        assert_eq!(error.kind(), ErrorKind::Validation);
    }
}
