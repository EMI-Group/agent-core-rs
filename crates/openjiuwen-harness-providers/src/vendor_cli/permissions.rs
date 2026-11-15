//! Host permission translation shared by the vendor CLI harnesses.
//!
//! A vendor CLI has its own permission model: Claude Code asks for tool approval and user input,
//! Codex asks for command approval and user input. The adapters that receive such prompts answer
//! them from the *host's* declared capabilities ([`HostCapability::ToolApproval`],
//! [`HostCapability::UserInput`], [`HostCapability::DynamicToolCall`],
//! [`HostCapability::NativeToolGateway`]) and the host's installed
//! [`HarnessInteractionHandler`].
//!
//! [`PermissionPolicy`] captures that decision once per start; [`ask_tool_approval`] and
//! [`ask_user_input`] are the two request shapes those adapters reduce to. When the host did not
//! declare the capability, or installed no handler, both helpers return the host's auto-approval
//! policy ([`ApprovalDecision::Allow`] / [`None`]) instead of blocking a turn on a prompt nobody can
//! answer.
//!
//! The DSH driver uses neither helper: it answers every server-to-client request with a JSON-RPC
//! "method not found" error, because the adapter exposes no client-side capability the runtime may
//! call.

use std::sync::Arc;

use openjiuwen_harness_protocol::{
    json_object_to_value, HarnessContext, HarnessInteractionHandler, HarnessInteractionRequest,
    HarnessInteractionResponse, HostCapability, InteractionResponseStatus, JsonObject, JsonValue,
    ToolApprovalDecision, ToolApprovalRequest, ToolApprovalResponse, UserInputRequest,
    UserInputResponse, Validate,
};

/// What the host allows a vendor CLI to ask during a turn.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[allow(clippy::struct_excessive_bools)] // mirrors the four independent host capability bits
pub struct PermissionPolicy {
    /// Whether the host answers user-input prompts.
    pub user_input: bool,
    /// Whether the host approves or denies tool calls.
    pub tool_approval: bool,
    /// Whether the host executes tools discovered dynamically at run time.
    pub dynamic_tools: bool,
    /// Whether the host executes native tool calls through its tool gateway.
    pub native_tools: bool,
}

impl PermissionPolicy {
    /// Derives the policy from the host capabilities declared in `context`.
    #[must_use]
    pub fn from_context(context: &HarnessContext) -> Self {
        let declared = |capability: HostCapability| context.host_capabilities.contains(&capability);
        Self {
            user_input: declared(HostCapability::UserInput),
            tool_approval: declared(HostCapability::ToolApproval),
            dynamic_tools: declared(HostCapability::DynamicToolCall),
            native_tools: declared(HostCapability::NativeToolGateway),
        }
    }

    /// Returns whether any interactive host service is declared.
    #[must_use]
    pub fn interactive(&self) -> bool {
        self.user_input || self.tool_approval
    }
}

/// How the harness should proceed after asking the host about one tool call.
#[derive(Debug, Clone, PartialEq)]
pub enum ApprovalDecision {
    /// Run the call, optionally with host-adjusted arguments.
    Allow {
        /// Rewritten arguments to execute instead of the provider's own, when the host adjusted
        /// them.
        updated_arguments: Option<JsonValue>,
    },
    /// Do not run the call.
    Deny {
        /// Explanation handed back to the CLI and reported to the provider.
        message: String,
        /// Whether the denial must also interrupt the active turn.
        interrupt: bool,
    },
}

/// Message used when the host denies a call without explaining why.
const DEFAULT_DENIAL: &str = "denied by host";

/// Asks the host whether `tool_name` may run with `arguments`.
///
/// The host's auto-approval policy applies when [`PermissionPolicy::tool_approval`] is not declared
/// or no handler is installed: the call is allowed unchanged rather than blocking a turn.
///
/// # Errors and failures
///
/// This helper never returns an error. A handler failure, an invalid request and a response of the
/// wrong interaction type all become [`ApprovalDecision::Deny`], because a vendor CLI must always
/// receive an answer.
pub async fn ask_tool_approval(
    handler: Option<&Arc<dyn HarnessInteractionHandler>>,
    policy: &PermissionPolicy,
    request_id: String,
    call_id: Option<String>,
    tool_name: String,
    arguments: JsonValue,
    provider_data: JsonObject,
) -> ApprovalDecision {
    let Some(handler) = handler.filter(|_| policy.tool_approval) else {
        return ApprovalDecision::Allow {
            updated_arguments: None,
        };
    };
    // The provider's arguments are a JSON value; an approval prompt wants an object.
    let arguments = match arguments {
        JsonValue::Object(object) => object.into_iter().collect::<JsonObject>(),
        other => {
            let mut object = JsonObject::new();
            object.insert("value".to_owned(), other);
            object
        }
    };
    let call_id = call_id.unwrap_or_else(|| request_id.clone());
    let request = ToolApprovalRequest {
        request_id,
        call_id,
        tool_name,
        arguments,
        provider_session_id: None,
        turn_id: None,
        title: None,
        description: None,
        reason: None,
        deadline_at: None,
        provider_data,
    };
    let interaction = HarnessInteractionRequest::ToolApproval(request);
    if let Err(error) = interaction.validate() {
        tracing::warn!(error = %error, "invalid tool approval request");
        return ApprovalDecision::Deny {
            message: format!("invalid tool approval request: {}", error.message()),
            interrupt: false,
        };
    }
    match handler.handle(&interaction).await {
        Ok(HarnessInteractionResponse::ToolApproval(response)) => translate_approval(response),
        Ok(other) => {
            let message = interaction.validate_response(&other).err().map_or_else(
                || "unexpected response type".to_owned(),
                |error| error.message().to_owned(),
            );
            tracing::warn!(message = %message, "host answered a tool approval with the wrong type");
            ApprovalDecision::Deny {
                message: format!("harness interaction failed: {message}"),
                interrupt: false,
            }
        }
        Err(error) => {
            tracing::warn!(error = %error, "host failed to answer a tool approval");
            ApprovalDecision::Deny {
                message: format!("harness interaction failed: {error}"),
                interrupt: false,
            }
        }
    }
}

/// Asks the user for input, returning the answer as a JSON value.
///
/// Returns [`None`] when [`PermissionPolicy::user_input`] is not declared, when no handler is
/// installed (the host's auto-approval policy), when the host declined or failed to answer, or when
/// the host answered with the wrong interaction type.
pub async fn ask_user_input(
    handler: Option<&Arc<dyn HarnessInteractionHandler>>,
    policy: &PermissionPolicy,
    request_id: String,
    prompt: String,
    choices: Vec<String>,
    provider_data: JsonObject,
) -> Option<JsonValue> {
    let handler = handler.filter(|_| policy.user_input)?;
    let request = UserInputRequest {
        request_id,
        prompt,
        provider_session_id: None,
        turn_id: None,
        choices,
        deadline_at: None,
        provider_data,
    };
    let interaction = HarnessInteractionRequest::UserInput(request);
    if let Err(error) = interaction.validate() {
        tracing::warn!(error = %error, "invalid user input request");
        return None;
    }
    match handler.handle(&interaction).await {
        Ok(HarnessInteractionResponse::UserInput(response)) => answer_value(response),
        Ok(other) => {
            tracing::warn!(
                response = other.request_id(),
                "host answered a user input request with the wrong type"
            );
            None
        }
        Err(error) => {
            tracing::warn!(error = %error, "host failed to answer a user input request");
            None
        }
    }
}

/// Translates a host approval response into the harness decision.
fn translate_approval(response: ToolApprovalResponse) -> ApprovalDecision {
    match response.decision {
        ToolApprovalDecision::Allow | ToolApprovalDecision::AllowForSession => {
            ApprovalDecision::Allow {
                updated_arguments: response
                    .updated_arguments
                    .as_ref()
                    .map(json_object_to_value),
            }
        }
        ToolApprovalDecision::Deny | ToolApprovalDecision::Abort => ApprovalDecision::Deny {
            message: response.reason.unwrap_or_else(|| DEFAULT_DENIAL.to_owned()),
            interrupt: response.interrupt,
        },
    }
}

/// Flattens a completed user-input response into the answer value.
///
/// A single answered question yields that question's raw value; several answers (or a non-object
/// payload) are returned unchanged so the caller can decide how to render them.
fn answer_value(response: UserInputResponse) -> Option<JsonValue> {
    if response.status != InteractionResponseStatus::Completed {
        tracing::debug!(?response.status, "user input was not completed");
        return None;
    }
    match response.content {
        JsonValue::Object(answers) if answers.len() == 1 => answers.values().next().cloned(),
        content => Some(content),
    }
}

#[cfg(test)]
mod tests {
    // Tests assert by panicking on failure, which is the intent of a test.
    #![allow(clippy::expect_used, clippy::unwrap_used)]

    use std::sync::{Arc, Mutex};

    use async_trait::async_trait;
    use openjiuwen_harness_protocol::{
        HarnessContext, HarnessError, HarnessInteractionHandler, HarnessInteractionRequest,
        HarnessInteractionResponse, HarnessResult, HostCapability, InteractionCancelReason,
        InteractionResponseStatus, JsonObject, JsonValue, ToolApprovalDecision,
        ToolApprovalResponse, UserInputResponse,
    };
    use serde_json::json;

    use super::{ask_tool_approval, ask_user_input, ApprovalDecision, PermissionPolicy};

    /// Host handler that records the request and returns a canned outcome.
    struct FakeHandler {
        /// Requests the harness raised, in order.
        requests: Mutex<Vec<HarnessInteractionRequest>>,
        /// Response returned by [`handle`](HarnessInteractionHandler::handle).
        response: Option<HarnessInteractionResponse>,
        /// Failure message returned by [`handle`](HarnessInteractionHandler::handle).
        failure: Option<String>,
    }

    impl FakeHandler {
        /// Handler answering with `response`.
        fn answering(response: HarnessInteractionResponse) -> Arc<Self> {
            Arc::new(Self {
                requests: Mutex::new(Vec::new()),
                response: Some(response),
                failure: None,
            })
        }

        /// Handler whose `handle` fails with `message`.
        fn failing(message: &str) -> Arc<Self> {
            Arc::new(Self {
                requests: Mutex::new(Vec::new()),
                response: None,
                failure: Some(message.to_owned()),
            })
        }

        /// Returns the recorded requests.
        fn requests(&self) -> Vec<HarnessInteractionRequest> {
            self.requests.lock().expect("the request log").clone()
        }
    }

    #[async_trait]
    impl HarnessInteractionHandler for FakeHandler {
        async fn handle(
            &self,
            request: &HarnessInteractionRequest,
        ) -> HarnessResult<HarnessInteractionResponse> {
            self.requests
                .lock()
                .expect("the request log")
                .push(request.clone());
            match (&self.response, &self.failure) {
                (Some(response), _) => Ok(response.clone()),
                (None, Some(message)) => Err(HarnessError::protocol(message.clone())),
                (None, None) => Err(HarnessError::protocol("no canned outcome")),
            }
        }

        async fn cancel(
            &self,
            _request_id: &str,
            _reason: InteractionCancelReason,
        ) -> HarnessResult<()> {
            Ok(())
        }
    }

    /// Policy with both interactive capabilities declared.
    fn interactive_policy() -> PermissionPolicy {
        PermissionPolicy {
            user_input: true,
            tool_approval: true,
            dynamic_tools: true,
            native_tools: true,
        }
    }

    /// Approval response with `decision`.
    fn approval(decision: ToolApprovalDecision) -> HarnessInteractionResponse {
        HarnessInteractionResponse::ToolApproval(ToolApprovalResponse {
            request_id: "req-1".to_owned(),
            decision,
            updated_arguments: None,
            reason: None,
            interrupt: false,
            provider_data: JsonObject::new(),
        })
    }

    /// User-input response with `content` and `status`.
    fn user_input(
        status: InteractionResponseStatus,
        content: JsonValue,
    ) -> HarnessInteractionResponse {
        HarnessInteractionResponse::UserInput(UserInputResponse {
            request_id: "req-1".to_owned(),
            status,
            content,
            provider_data: JsonObject::new(),
        })
    }

    #[tokio::test]
    async fn an_undeclared_capability_auto_allows() {
        let handler = FakeHandler::answering(approval(ToolApprovalDecision::Deny));
        let policy = PermissionPolicy::default();
        let decision = ask_tool_approval(
            Some(&(handler.clone() as Arc<dyn HarnessInteractionHandler>)),
            &policy,
            "req-1".to_owned(),
            None,
            "Bash".to_owned(),
            json!({"command": "ls"}),
            JsonObject::new(),
        )
        .await;
        assert_eq!(
            decision,
            ApprovalDecision::Allow {
                updated_arguments: None
            }
        );
        assert_eq!(handler.requests().len(), 0);
    }

    #[tokio::test]
    async fn a_missing_handler_auto_allows() {
        let decision = ask_tool_approval(
            None,
            &interactive_policy(),
            "req-1".to_owned(),
            None,
            "Bash".to_owned(),
            json!({}),
            JsonObject::new(),
        )
        .await;
        assert_eq!(
            decision,
            ApprovalDecision::Allow {
                updated_arguments: None
            }
        );
    }

    #[tokio::test]
    async fn the_request_carries_the_call_identity_and_arguments() {
        let handler = FakeHandler::answering(approval(ToolApprovalDecision::Allow));
        let dynamic = handler.clone() as Arc<dyn HarnessInteractionHandler>;
        let mut provider_data = JsonObject::new();
        provider_data.insert("source".to_owned(), json!("claude"));
        ask_tool_approval(
            Some(&dynamic),
            &interactive_policy(),
            "req-1".to_owned(),
            Some("call-9".to_owned()),
            "Write".to_owned(),
            json!({"path": "a.txt"}),
            provider_data.clone(),
        )
        .await;
        let requests = handler.requests();
        let HarnessInteractionRequest::ToolApproval(request) = &requests[0] else {
            panic!("expected a tool approval request");
        };
        assert_eq!(request.request_id, "req-1");
        assert_eq!(request.call_id, "call-9");
        assert_eq!(request.tool_name, "Write");
        assert_eq!(request.arguments.get("path"), Some(&json!("a.txt")));
        assert_eq!(request.provider_data, provider_data);
    }

    #[tokio::test]
    async fn a_missing_call_id_falls_back_to_the_request_id() {
        let handler = FakeHandler::answering(approval(ToolApprovalDecision::Allow));
        let dynamic = handler.clone() as Arc<dyn HarnessInteractionHandler>;
        ask_tool_approval(
            Some(&dynamic),
            &interactive_policy(),
            "req-7".to_owned(),
            None,
            "Read".to_owned(),
            json!({}),
            JsonObject::new(),
        )
        .await;
        let requests = handler.requests();
        let HarnessInteractionRequest::ToolApproval(request) = &requests[0] else {
            panic!("expected a tool approval request");
        };
        assert_eq!(request.call_id, "req-7");
    }

    #[tokio::test]
    async fn non_object_arguments_are_wrapped() {
        let handler = FakeHandler::answering(approval(ToolApprovalDecision::Allow));
        let dynamic = handler.clone() as Arc<dyn HarnessInteractionHandler>;
        ask_tool_approval(
            Some(&dynamic),
            &interactive_policy(),
            "req-1".to_owned(),
            Some("call-1".to_owned()),
            "Read".to_owned(),
            json!("raw"),
            JsonObject::new(),
        )
        .await;
        let requests = handler.requests();
        let HarnessInteractionRequest::ToolApproval(request) = &requests[0] else {
            panic!("expected a tool approval request");
        };
        assert_eq!(request.arguments.get("value"), Some(&json!("raw")));
    }

    #[tokio::test]
    async fn allow_translates_updated_arguments() {
        let response = HarnessInteractionResponse::ToolApproval(ToolApprovalResponse {
            request_id: "req-1".to_owned(),
            decision: ToolApprovalDecision::Allow,
            updated_arguments: Some(
                serde_json::from_value(json!({"path": "safe.txt"})).expect("a JSON object"),
            ),
            reason: None,
            interrupt: false,
            provider_data: JsonObject::new(),
        });
        let handler = FakeHandler::answering(response);
        let dynamic = handler.clone() as Arc<dyn HarnessInteractionHandler>;
        let decision = ask_tool_approval(
            Some(&dynamic),
            &interactive_policy(),
            "req-1".to_owned(),
            None,
            "Write".to_owned(),
            json!({}),
            JsonObject::new(),
        )
        .await;
        assert_eq!(
            decision,
            ApprovalDecision::Allow {
                updated_arguments: Some(json!({"path": "safe.txt"}))
            }
        );
    }

    #[tokio::test]
    async fn allow_for_session_is_an_allow() {
        let handler = FakeHandler::answering(approval(ToolApprovalDecision::AllowForSession));
        let dynamic = handler.clone() as Arc<dyn HarnessInteractionHandler>;
        let decision = ask_tool_approval(
            Some(&dynamic),
            &interactive_policy(),
            "req-1".to_owned(),
            None,
            "Bash".to_owned(),
            json!({}),
            JsonObject::new(),
        )
        .await;
        assert_eq!(
            decision,
            ApprovalDecision::Allow {
                updated_arguments: None
            }
        );
    }

    #[tokio::test]
    async fn deny_translates_the_reason_and_interrupt() {
        let response = HarnessInteractionResponse::ToolApproval(ToolApprovalResponse {
            request_id: "req-1".to_owned(),
            decision: ToolApprovalDecision::Deny,
            updated_arguments: None,
            reason: Some("not in this workspace".to_owned()),
            interrupt: true,
            provider_data: JsonObject::new(),
        });
        let handler = FakeHandler::answering(response);
        let dynamic = handler.clone() as Arc<dyn HarnessInteractionHandler>;
        let decision = ask_tool_approval(
            Some(&dynamic),
            &interactive_policy(),
            "req-1".to_owned(),
            None,
            "Bash".to_owned(),
            json!({}),
            JsonObject::new(),
        )
        .await;
        assert_eq!(
            decision,
            ApprovalDecision::Deny {
                message: "not in this workspace".to_owned(),
                interrupt: true
            }
        );
    }

    #[tokio::test]
    async fn deny_without_a_reason_uses_the_default_message() {
        let handler = FakeHandler::answering(approval(ToolApprovalDecision::Deny));
        let dynamic = handler.clone() as Arc<dyn HarnessInteractionHandler>;
        let decision = ask_tool_approval(
            Some(&dynamic),
            &interactive_policy(),
            "req-1".to_owned(),
            None,
            "Bash".to_owned(),
            json!({}),
            JsonObject::new(),
        )
        .await;
        assert_eq!(
            decision,
            ApprovalDecision::Deny {
                message: "denied by host".to_owned(),
                interrupt: false
            }
        );
    }

    #[tokio::test]
    async fn abort_is_a_denial() {
        let handler = FakeHandler::answering(approval(ToolApprovalDecision::Abort));
        let dynamic = handler.clone() as Arc<dyn HarnessInteractionHandler>;
        let decision = ask_tool_approval(
            Some(&dynamic),
            &interactive_policy(),
            "req-1".to_owned(),
            None,
            "Bash".to_owned(),
            json!({}),
            JsonObject::new(),
        )
        .await;
        assert!(matches!(decision, ApprovalDecision::Deny { .. }));
    }

    #[tokio::test]
    async fn a_handler_failure_denies_the_call() {
        let handler = FakeHandler::failing("host unavailable");
        let dynamic = handler.clone() as Arc<dyn HarnessInteractionHandler>;
        let decision = ask_tool_approval(
            Some(&dynamic),
            &interactive_policy(),
            "req-1".to_owned(),
            None,
            "Bash".to_owned(),
            json!({}),
            JsonObject::new(),
        )
        .await;
        assert_eq!(
            decision,
            ApprovalDecision::Deny {
                message: "harness interaction failed: harness protocol error: host unavailable"
                    .to_owned(),
                interrupt: false
            }
        );
    }

    #[tokio::test]
    async fn a_wrong_response_type_denies_the_call() {
        let handler = FakeHandler::answering(user_input(
            InteractionResponseStatus::Completed,
            json!("yes"),
        ));
        let dynamic = handler.clone() as Arc<dyn HarnessInteractionHandler>;
        let decision = ask_tool_approval(
            Some(&dynamic),
            &interactive_policy(),
            "req-1".to_owned(),
            None,
            "Bash".to_owned(),
            json!({}),
            JsonObject::new(),
        )
        .await;
        match decision {
            ApprovalDecision::Deny { message, interrupt } => {
                assert!(message.starts_with("harness interaction failed:"));
                assert!(message.contains("ToolApprovalRequest requires ToolApprovalResponse"));
                assert!(!interrupt);
            }
            ApprovalDecision::Allow { .. } => panic!("a wrong response type must deny"),
        }
    }

    #[tokio::test]
    async fn user_input_without_the_capability_is_none() {
        let handler = FakeHandler::answering(user_input(
            InteractionResponseStatus::Completed,
            json!({"question": "continue?"}),
        ));
        let dynamic = handler.clone() as Arc<dyn HarnessInteractionHandler>;
        let answer = ask_user_input(
            Some(&dynamic),
            &PermissionPolicy::default(),
            "req-1".to_owned(),
            "continue?".to_owned(),
            Vec::new(),
            JsonObject::new(),
        )
        .await;
        assert!(answer.is_none());
        assert_eq!(handler.requests().len(), 0);
    }

    #[tokio::test]
    async fn a_single_answer_yields_its_raw_value() {
        let handler = FakeHandler::answering(user_input(
            InteractionResponseStatus::Completed,
            json!({"question": "yes"}),
        ));
        let dynamic = handler.clone() as Arc<dyn HarnessInteractionHandler>;
        let answer = ask_user_input(
            Some(&dynamic),
            &interactive_policy(),
            "req-1".to_owned(),
            "continue?".to_owned(),
            vec!["yes".to_owned(), "no".to_owned()],
            JsonObject::new(),
        )
        .await;
        assert_eq!(answer, Some(json!("yes")));
        let requests = handler.requests();
        let HarnessInteractionRequest::UserInput(request) = &requests[0] else {
            panic!("expected a user input request");
        };
        assert_eq!(request.prompt, "continue?");
        assert_eq!(request.choices, vec!["yes".to_owned(), "no".to_owned()]);
    }

    #[tokio::test]
    async fn several_answers_are_returned_as_an_object() {
        let handler = FakeHandler::answering(user_input(
            InteractionResponseStatus::Completed,
            json!({"a": "one", "b": "two"}),
        ));
        let dynamic = handler.clone() as Arc<dyn HarnessInteractionHandler>;
        let answer = ask_user_input(
            Some(&dynamic),
            &interactive_policy(),
            "req-1".to_owned(),
            "fill both".to_owned(),
            Vec::new(),
            JsonObject::new(),
        )
        .await;
        assert_eq!(answer, Some(json!({"a": "one", "b": "two"})));
    }

    #[tokio::test]
    async fn a_declined_user_input_is_none() {
        let handler = FakeHandler::answering(user_input(
            InteractionResponseStatus::Declined,
            JsonValue::Null,
        ));
        let dynamic = handler.clone() as Arc<dyn HarnessInteractionHandler>;
        let answer = ask_user_input(
            Some(&dynamic),
            &interactive_policy(),
            "req-1".to_owned(),
            "continue?".to_owned(),
            Vec::new(),
            JsonObject::new(),
        )
        .await;
        assert!(answer.is_none());
    }

    #[tokio::test]
    async fn a_failing_user_input_handler_is_none() {
        let handler = FakeHandler::failing("no ui");
        let dynamic = handler.clone() as Arc<dyn HarnessInteractionHandler>;
        let answer = ask_user_input(
            Some(&dynamic),
            &interactive_policy(),
            "req-1".to_owned(),
            "continue?".to_owned(),
            Vec::new(),
            JsonObject::new(),
        )
        .await;
        assert!(answer.is_none());
    }

    #[test]
    fn the_policy_reads_the_declared_capabilities() {
        let mut context = HarnessContext::new("agent", "agent-a", "host-session", "prompt");
        assert_eq!(
            PermissionPolicy::from_context(&context),
            PermissionPolicy::default()
        );
        assert!(!PermissionPolicy::from_context(&context).interactive());
        context.host_capabilities.insert(HostCapability::UserInput);
        context
            .host_capabilities
            .insert(HostCapability::DynamicToolCall);
        let policy = PermissionPolicy::from_context(&context);
        assert!(policy.user_input);
        assert!(policy.interactive());
        assert!(policy.dynamic_tools);
        assert!(!policy.tool_approval);
        assert!(!policy.native_tools);
    }
}
