//! Synchronous control-plane hooks for third-party agent harnesses.
//!
//! Mirrors `openjiuwen.harness_protocol.hooks` (v0.1.20): [`ToolDecisionKind`],
//! [`BeforePromptContext`]/[`BeforePromptResult`], [`BeforeToolContext`]/[`ToolDecision`],
//! [`AfterToolContext`]/[`AfterToolResult`], [`StopHookContext`], and the
//! [`HarnessHookDispatcher`] SPI.
//!
//! Hook contexts and results are pure JSON, so they are `Serialize + Deserialize + JsonSchema`
//! without further rules. Python freezes `arguments`/`result`/`metadata` in `__post_init__`;
//! [`JsonValue`]/[`JsonObject`] are already owned and JSON-compatible, so value semantics replace
//! deep immutability. Constructors do not validate implicitly — decode paths and callers that need
//! the Python invariants call [`Validate::validate`].

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::errors::{HarnessError, HarnessResult};
use crate::json::{JsonObject, JsonValue};
use crate::validation::Validate;

/// Decision returned before a tool invocation.
///
/// Wire values are `allow`, `deny`, `rewrite`, `ask`, and `provider_policy`.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    PartialOrd,
    Ord,
    Serialize,
    Deserialize,
    schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum ToolDecisionKind {
    /// Execute the tool with the arguments as requested.
    Allow,
    /// Reject the tool invocation.
    Deny,
    /// Execute the tool with `updated_arguments` instead of the requested arguments.
    Rewrite,
    /// Ask the host for a permission decision before executing the tool.
    Ask,
    /// Defer to the provider's own permission policy.
    ProviderPolicy,
}

/// Serde default for [`BeforePromptResult::continue_execution`] (Python `True`).
fn default_true() -> bool {
    true
}

/// Context supplied before an input reaches the external agent.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub struct BeforePromptContext {
    /// Name of the agent that would receive the prompt.
    pub agent_name: String,
    /// Provider-side session identifier, when the harness keeps a persistent session.
    pub provider_session_id: Option<String>,
    /// Identifier of the turn the prompt belongs to, when a turn is already active.
    pub turn_id: Option<String>,
    /// Prompt the host is about to deliver.
    pub prompt: String,
}

/// Potential prompt rewrite or stop decision returned by a hook.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub struct BeforePromptResult {
    /// Prompt to deliver; equal to the incoming prompt when the hook does not rewrite it.
    pub prompt: String,
    /// Whether execution should proceed; defaults to `true` (Python `True`).
    #[serde(default = "default_true")]
    pub continue_execution: bool,
    /// Human-readable reason, typically set when `continue_execution` is `false`.
    pub reason: Option<String>,
    /// Extra context the provider should append to its own prompt.
    pub additional_context: Option<String>,
}

/// Context supplied before an external agent invokes a tool.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub struct BeforeToolContext {
    /// Name of the agent that wants to invoke the tool.
    pub agent_name: String,
    /// Provider-side session identifier, when the harness keeps a persistent session.
    pub provider_session_id: Option<String>,
    /// Identifier of the turn the invocation belongs to, when known.
    pub turn_id: Option<String>,
    /// Provider-assigned identifier of this tool call.
    pub call_id: String,
    /// Name of the tool being invoked.
    pub tool_name: String,
    /// Arguments the provider intends to pass to the tool.
    pub arguments: JsonObject,
}

/// Permission and optional input rewrite returned before tool execution.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub struct ToolDecision {
    /// Permission decision for the tool invocation.
    pub decision: ToolDecisionKind,
    /// Human-readable reason for the decision.
    pub reason: Option<String>,
    /// Replacement arguments; required exactly when `decision` is
    /// [`ToolDecisionKind::Rewrite`].
    pub updated_arguments: Option<JsonObject>,
    /// Extra context the provider should append to its own conversation.
    pub additional_context: Option<String>,
}

impl Validate for ToolDecision {
    /// Checks the rewrite/`updated_arguments` coupling in Python's order.
    fn validate(&self) -> HarnessResult<()> {
        if self.decision == ToolDecisionKind::Rewrite && self.updated_arguments.is_none() {
            return Err(HarnessError::validation(
                "rewrite tool decision requires updated_arguments",
            ));
        }
        if self.decision != ToolDecisionKind::Rewrite && self.updated_arguments.is_some() {
            return Err(HarnessError::validation(
                "updated_arguments is only valid for a rewrite tool decision",
            ));
        }
        Ok(())
    }
}

/// Context supplied after a successful or failed tool invocation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub struct AfterToolContext {
    /// Name of the agent that invoked the tool.
    pub agent_name: String,
    /// Provider-side session identifier, when the harness keeps a persistent session.
    pub provider_session_id: Option<String>,
    /// Identifier of the turn the invocation belonged to, when known.
    pub turn_id: Option<String>,
    /// Provider-assigned identifier of this tool call.
    pub call_id: String,
    /// Name of the tool that was invoked.
    pub tool_name: String,
    /// Arguments the provider passed to the tool.
    pub arguments: JsonObject,
    /// Tool result, or JSON `null` when the invocation failed; defaults to `null`.
    #[serde(default)]
    pub result: JsonValue,
    /// Error message when the invocation failed, `None` on success.
    pub error: Option<String>,
}

/// Optional tool-result rewrite and context returned by a hook.
///
/// Every field carries a Python default, so the whole struct is defaulted on decode.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct AfterToolResult {
    /// Whether the provider should replace its tool result with `updated_result`.
    pub replace_result: bool,
    /// Replacement result; JSON `null` (the default) when the hook does not rewrite it.
    pub updated_result: JsonValue,
    /// Extra context the provider should append to its own conversation.
    pub additional_context: Option<String>,
}

/// Context supplied when a turn is about to stop.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub struct StopHookContext {
    /// Name of the agent whose turn is stopping.
    pub agent_name: String,
    /// Provider-side session identifier, when the harness keeps a persistent session.
    pub provider_session_id: Option<String>,
    /// Identifier of the turn that is stopping, when known.
    pub turn_id: Option<String>,
    /// Reason the turn is stopping, when the provider reports one.
    pub reason: Option<String>,
    /// Free-form metadata attached to the stop; defaults to an empty object.
    #[serde(default)]
    pub metadata: JsonObject,
}

/// Control-plane callbacks awaited by a harness at execution boundaries.
///
/// Hook return values may change or stop execution. Hook lifecycle events in [`crate::events`]
/// are only observational mirrors and must never be used to make permission decisions.
///
/// The trait is object-safe (`async-trait` + `Send + Sync`), so hosts can hold
/// `Arc<dyn HarnessHookDispatcher>`.
#[async_trait]
pub trait HarnessHookDispatcher: Send + Sync {
    /// Inspects or rewrites an input before model execution.
    ///
    /// Returning `continue_execution = false` stops the turn with the supplied `reason`;
    /// otherwise the returned `prompt` (which may be unchanged) is what the provider sees.
    ///
    /// # Errors
    ///
    /// Returns [`HarnessError`] when the hook itself fails; the harness treats the failure as a
    /// turn error rather than silently continuing with unvetted input.
    async fn before_prompt(
        &self,
        context: &BeforePromptContext,
    ) -> HarnessResult<BeforePromptResult>;

    /// Authorizes, rejects, rewrites, asks the host, or defers to provider policy.
    ///
    /// # Errors
    ///
    /// Returns [`HarnessError`] when the hook itself fails. Failing to obtain a decision must not
    /// be interpreted as an implicit [`ToolDecisionKind::Allow`].
    async fn before_tool(&self, context: &BeforeToolContext) -> HarnessResult<ToolDecision>;

    /// Inspects or rewrites a tool result after execution.
    ///
    /// # Errors
    ///
    /// Returns [`HarnessError`] when the hook itself fails; the harness reports the turn error
    /// instead of continuing with a result the hook never approved.
    async fn after_tool(&self, context: &AfterToolContext) -> HarnessResult<AfterToolResult>;

    /// Observes a pending turn stop before it is finalized.
    ///
    /// # Errors
    ///
    /// Returns [`HarnessError`] when the hook itself fails; the stop still happens, and the error
    /// is surfaced to the host.
    async fn on_stop(&self, context: &StopHookContext) -> HarnessResult<()>;
}

#[cfg(test)]
mod tests {
    // Tests assert by panicking on failure, which is the intent of a test.
    #![allow(clippy::expect_used, clippy::unwrap_used)]

    use serde_json::json;

    use super::{
        AfterToolContext, AfterToolResult, BeforePromptResult, StopHookContext, ToolDecision,
        ToolDecisionKind,
    };
    use crate::json::empty_json_object;
    use crate::Validate;

    #[test]
    fn tool_decision_kind_wire_values_match_python() {
        for (kind, wire) in [
            (ToolDecisionKind::Allow, "allow"),
            (ToolDecisionKind::Deny, "deny"),
            (ToolDecisionKind::Rewrite, "rewrite"),
            (ToolDecisionKind::Ask, "ask"),
            (ToolDecisionKind::ProviderPolicy, "provider_policy"),
        ] {
            assert_eq!(serde_json::to_value(kind).expect("serializes"), json!(wire));
            let decoded: ToolDecisionKind =
                serde_json::from_value(json!(wire)).expect("deserializes");
            assert_eq!(decoded, kind);
        }
    }

    #[test]
    fn rewrite_decision_requires_updated_arguments() {
        let rewrite_without_arguments = ToolDecision {
            decision: ToolDecisionKind::Rewrite,
            reason: None,
            updated_arguments: None,
            additional_context: None,
        };
        let error = rewrite_without_arguments
            .validate()
            .expect_err("rewrite without arguments");
        assert_eq!(
            error.message(),
            "rewrite tool decision requires updated_arguments"
        );

        let rewrite = ToolDecision {
            updated_arguments: Some(
                [(String::from("path"), json!("/tmp/x"))]
                    .into_iter()
                    .collect(),
            ),
            ..rewrite_without_arguments
        };
        assert!(rewrite.validate().is_ok());
    }

    #[test]
    fn updated_arguments_are_rejected_for_non_rewrite_decisions() {
        for decision in [
            ToolDecisionKind::Allow,
            ToolDecisionKind::Deny,
            ToolDecisionKind::Ask,
            ToolDecisionKind::ProviderPolicy,
        ] {
            let value = ToolDecision {
                decision,
                reason: None,
                updated_arguments: Some(empty_json_object()),
                additional_context: None,
            };
            let error = value.validate().expect_err("misplaced arguments");
            assert_eq!(
                error.message(),
                "updated_arguments is only valid for a rewrite tool decision"
            );
        }

        let plain = ToolDecision {
            decision: ToolDecisionKind::Allow,
            reason: Some("looks safe".to_owned()),
            updated_arguments: None,
            additional_context: None,
        };
        assert!(plain.validate().is_ok());
    }

    #[test]
    fn before_prompt_result_decodes_with_defaults() {
        let decoded: BeforePromptResult =
            serde_json::from_str(r#"{"prompt": "hi"}"#).expect("valid result");
        assert!(decoded.continue_execution, "Python default is True");
        assert_eq!(decoded.reason, None);
        assert_eq!(decoded.additional_context, None);

        let stopped: BeforePromptResult =
            serde_json::from_str(r#"{"prompt": "hi", "continue_execution": false}"#)
                .expect("valid result");
        assert!(!stopped.continue_execution);

        assert_eq!(
            serde_json::to_value(&decoded).expect("serializes"),
            json!({
                "prompt": "hi",
                "continue_execution": true,
                "reason": null,
                "additional_context": null,
            })
        );
    }

    #[test]
    fn after_tool_result_is_all_defaults() {
        let decoded: AfterToolResult = serde_json::from_str("{}").expect("all defaults");
        assert_eq!(decoded, AfterToolResult::default());
        assert!(!decoded.replace_result);
        assert_eq!(decoded.updated_result, serde_json::Value::Null);
        assert_eq!(decoded.additional_context, None);
        assert_eq!(
            serde_json::to_value(&decoded).expect("serializes"),
            json!({
                "replace_result": false,
                "updated_result": null,
                "additional_context": null,
            })
        );
    }

    #[test]
    fn after_tool_context_defaults_result_to_null_and_stop_metadata_to_empty() {
        let decoded: AfterToolContext = serde_json::from_str(
            r#"{"agent_name": "a", "provider_session_id": null, "turn_id": null,
                "call_id": "c1", "tool_name": "shell", "arguments": {}, "error": "boom"}"#,
        )
        .expect("valid context");
        assert_eq!(decoded.result, serde_json::Value::Null);
        assert_eq!(decoded.error.as_deref(), Some("boom"));

        let stop: StopHookContext = serde_json::from_str(
            r#"{"agent_name": "a", "provider_session_id": null, "turn_id": null}"#,
        )
        .expect("valid stop context");
        assert!(stop.metadata.is_empty());
        assert_eq!(stop.reason, None);
    }
}
