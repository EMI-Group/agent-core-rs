//! Routing of harness interaction requests to automatic or host-provided answers.
//!
//! The adapter answers what it can on its own and escalates the rest to the host through the
//! kernel's interactive-input contract: the request is published as one
//! [`INTERACTION_KIND`](crate::input::INTERACTION_KIND) chunk, the answering call is parked on a
//! one-shot channel, and [`send`](super::HarnessIOAdapter::send) later turns the host answer into
//! the typed response.
//!
//! Routing order (Python parity):
//!
//! 1. tool approvals — granted automatically unless the host opted into asking, and never for a
//!    provider interrupt (`provider_data.interrupt_id`);
//! 2. user input — always asked;
//! 3. MCP elicitation — always declined, never prompted;
//! 4. dynamic tool call — executed through the host tool gateway, or failed when the host
//!    declared none;
//! 5. provider interaction — answered by the configured provider handler, or declined.

use openjiuwen_agent::AgentChunk;
use openjiuwen_harness_protocol::{
    DynamicToolCallRequest, DynamicToolCallResponse, HarnessError, HarnessInteractionRequest,
    HarnessInteractionResponse, HarnessResult, InteractionCancelReason, InteractionResponseStatus,
    JsonObject, JsonValue, McpElicitationRequest, McpElicitationResponse,
    ProviderInteractionResponse, ToolApprovalDecision, ToolApprovalRequest, ToolApprovalResponse,
    ToolInvocation,
};
use openjiuwen_session::interaction::InteractiveInput;
use serde_json::json;
use tokio::sync::oneshot;

use super::answers;
use super::{lock, HarnessIOAdapter, PendingInteraction, PendingKind};
use crate::input::INTERACTION_KIND;

/// Failure message returned for a dynamic tool call when the host has no tool gateway.
const NO_TOOL_GATEWAY_MESSAGE: &str =
    "This host cannot run client tools (it did not declare DYNAMIC_TOOL_CALL).";

/// Returns the response the harness awaits for `request`.
///
/// # Errors
///
/// Returns [`HarnessError::State`] when the same request id is already pending or the pending
/// request was cancelled while waiting for the host, and the configured provider handler's own
/// error for a provider interaction.
pub(super) async fn route(
    adapter: &HarnessIOAdapter,
    request: &HarnessInteractionRequest,
) -> HarnessResult<HarnessInteractionResponse> {
    match request {
        HarnessInteractionRequest::ToolApproval(approval) => {
            if adapter.config.auto_approve_tools
                && !approval.provider_data.contains_key("interrupt_id")
            {
                return Ok(HarnessInteractionResponse::ToolApproval(auto_approval(
                    approval,
                )));
            }
            ask_host(adapter, request, PendingKind::ToolApproval).await
        }
        HarnessInteractionRequest::UserInput(_) => {
            ask_host(adapter, request, PendingKind::UserInput).await
        }
        HarnessInteractionRequest::McpElicitation(elicitation) => Ok(
            HarnessInteractionResponse::McpElicitation(decline_elicitation(elicitation)),
        ),
        HarnessInteractionRequest::DynamicToolCall(call) => dynamic_tool_call(adapter, call).await,
        HarnessInteractionRequest::Provider(provider) => {
            if let Some(handler) = adapter.config.provider_interaction_handler.as_ref() {
                return Ok(HarnessInteractionResponse::Provider(
                    handler.handle(provider).await?,
                ));
            }
            Ok(HarnessInteractionResponse::Provider(
                ProviderInteractionResponse {
                    request_id: provider.request_id.clone(),
                    status: InteractionResponseStatus::Declined,
                    payload: JsonObject::new(),
                },
            ))
        }
    }
}

/// Cancels a pending interaction, releasing its waiter.
///
/// Idempotent: cancelling an unknown or already settled request logs and returns.
pub(super) fn cancel(
    adapter: &HarnessIOAdapter,
    request_id: &str,
    reason: InteractionCancelReason,
) {
    let removed = lock(&adapter.pending).remove(request_id).is_some();
    tracing::debug!(
        %request_id,
        ?reason,
        was_pending = removed,
        "harness IO adapter interaction cancelled"
    );
}

/// Resolves every pending interaction `input` answers and returns how many were resolved.
///
/// `interrupt_answers` is matched before `user_inputs`, both keyed by request id; answers that
/// match no pending request are ignored.
pub(super) fn resolve_pending(adapter: &HarnessIOAdapter, input: &InteractiveInput) -> usize {
    let by_interrupt = input
        .interrupt_answers
        .iter()
        .map(|(request_id, answer)| resolve_one(adapter, request_id, answer));
    let by_node = input
        .user_inputs
        .iter()
        .map(|(request_id, answer)| resolve_one(adapter, request_id, answer));
    by_interrupt.chain(by_node).sum()
}

/// Resolves one pending interaction, if `request_id` is pending.
fn resolve_one(adapter: &HarnessIOAdapter, request_id: &str, answer: &JsonValue) -> usize {
    let Some(pending) = lock(&adapter.pending).remove(request_id) else {
        return 0;
    };
    let response = match pending.kind {
        PendingKind::ToolApproval => HarnessInteractionResponse::ToolApproval(
            answers::tool_approval_response(request_id, answer),
        ),
        PendingKind::UserInput => {
            HarnessInteractionResponse::UserInput(answers::user_input_response(request_id, answer))
        }
    };
    if pending.sender.send(response).is_err() {
        tracing::debug!(
            %request_id,
            "harness IO adapter interaction waiter was dropped"
        );
    }
    1
}

/// Builds the automatic approval granted when the host does not want to be asked.
fn auto_approval(request: &ToolApprovalRequest) -> ToolApprovalResponse {
    ToolApprovalResponse {
        request_id: request.request_id.clone(),
        decision: ToolApprovalDecision::Allow,
        updated_arguments: None,
        reason: None,
        interrupt: false,
        provider_data: JsonObject::new(),
    }
}

/// Builds the unconditional decline returned for an MCP elicitation.
fn decline_elicitation(request: &McpElicitationRequest) -> McpElicitationResponse {
    McpElicitationResponse {
        request_id: request.request_id.clone(),
        status: InteractionResponseStatus::Declined,
        values: JsonObject::new(),
        provider_data: JsonObject::new(),
    }
}

/// Executes a dynamic tool call through the host tool gateway.
async fn dynamic_tool_call(
    adapter: &HarnessIOAdapter,
    request: &DynamicToolCallRequest,
) -> HarnessResult<HarnessInteractionResponse> {
    let gateway = lock(&adapter.tool_gateway).clone();
    let Some(gateway) = gateway else {
        return Ok(HarnessInteractionResponse::DynamicToolCall(failed_call(
            request,
            NO_TOOL_GATEWAY_MESSAGE.to_owned(),
        )));
    };
    let invocation = ToolInvocation {
        call_id: request.call_id.clone(),
        name: request.tool_name.clone(),
        arguments: request.arguments.clone(),
    };
    let response = match gateway.invoke(&invocation).await {
        Ok(result) => DynamicToolCallResponse {
            request_id: request.request_id.clone(),
            status: InteractionResponseStatus::Completed,
            result: result.content,
            is_error: result.is_error,
            error_message: None,
            provider_data: JsonObject::new(),
        },
        Err(error) => {
            tracing::warn!(
                %error,
                request_id = %request.request_id,
                tool = %request.tool_name,
                "harness IO adapter dynamic tool call failed"
            );
            failed_call(request, error.message().to_owned())
        }
    };
    Ok(HarnessInteractionResponse::DynamicToolCall(response))
}

/// Builds the failed response for a dynamic tool call the host could not run.
fn failed_call(request: &DynamicToolCallRequest, message: String) -> DynamicToolCallResponse {
    DynamicToolCallResponse {
        request_id: request.request_id.clone(),
        status: InteractionResponseStatus::Failed,
        result: JsonValue::Null,
        is_error: true,
        error_message: Some(message),
        provider_data: JsonObject::new(),
    }
}

/// Publishes `request` as an interaction chunk and waits for the host answer.
async fn ask_host(
    adapter: &HarnessIOAdapter,
    request: &HarnessInteractionRequest,
    kind: PendingKind,
) -> HarnessResult<HarnessInteractionResponse> {
    let request_id = request.request_id().to_owned();
    let value = serde_json::to_value(request).map_err(|error| {
        HarnessError::protocol(format!(
            "interaction '{request_id}' could not be serialized: {error}"
        ))
    })?;

    let (sender, receiver) = oneshot::channel();
    {
        let mut pending = lock(&adapter.pending);
        if pending.contains_key(&request_id) {
            return Err(HarnessError::state(format!(
                "interaction '{request_id}' is already pending"
            )));
        }
        pending.insert(request_id.clone(), PendingInteraction { kind, sender });
    }

    let chunk = AgentChunk::new(
        INTERACTION_KIND,
        adapter.next_chunk_index(),
        json!({"id": request_id.clone(), "value": value}),
    );
    let _ = adapter.event_tx.send(chunk);

    let response = receiver.await;
    lock(&adapter.pending).remove(&request_id);
    response.map_err(|_| HarnessError::state(format!("interaction '{request_id}' was cancelled")))
}
