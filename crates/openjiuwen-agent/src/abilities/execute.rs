//! Tool-call execution engine.
//!
//! [`execute_calls`] is the single place where a list of model-requested
//! [`ToolCall`]s is turned into [`ToolCallResult`]s. It is rail-free: the
//! `ReAct` loop fires the tool rails around each batch, so the engine only
//! executes.
use super::{parse_tool_arguments, ToolCallResult, ToolExecutionOptions};
use indexmap::IndexMap;
use openjiuwen_foundation::{Tool, ToolCall, ToolContext};
use std::sync::Arc;

/// Executes `calls` against `tools` and returns one result per call.
///
/// Results keep the model's call order. Failures are results, never `Err`:
/// unknown tools, malformed arguments, and tool errors all become
/// [`ToolCallResult::failure`], so the model always receives a tool message it
/// can react to.
///
/// The baseline implementation is sequential and checks [`ToolExecutionOptions::cancellation`]
/// before each call; once cancelled, every remaining call yields a `cancelled`
/// failure.
///
// TODO: honour `options.parallel` by running contiguous `parallel_safe` batches
// concurrently, with non-parallel-safe tools acting as barriers, and wrap calls
// in `tokio::time::timeout` using `options.timeout`.
pub(crate) async fn execute_calls(
    tools: &IndexMap<String, Arc<dyn Tool>>,
    calls: &[ToolCall],
    options: &ToolExecutionOptions,
) -> Vec<ToolCallResult> {
    let mut results = Vec::with_capacity(calls.len());
    for call in calls {
        if options.cancellation.is_cancelled() {
            results.push(ToolCallResult::failure(call.clone(), "cancelled"));
            continue;
        }
        let Some(tool) = tools.get(&call.name) else {
            results.push(ToolCallResult::failure(
                call.clone(),
                format!("unknown tool '{}'", call.name),
            ));
            continue;
        };
        let arguments = match parse_tool_arguments(&call.arguments) {
            Ok(arguments) => arguments,
            Err(error) => {
                results.push(ToolCallResult::failure(
                    call.clone(),
                    error.message().to_string(),
                ));
                continue;
            }
        };
        let mut ctx = ToolContext::new();
        // Tools observe the caller's cancellation signal, not a fresh token.
        ctx.cancellation = options.cancellation.clone();
        if let Some(session_id) = &options.session_id {
            ctx = ctx.with_session(session_id.clone());
        }
        if let Some(timeout) = options.timeout {
            ctx = ctx.with_timeout(timeout);
        }
        ctx = ctx.with_metadata(options.metadata.clone());
        match tool.invoke(arguments, &ctx).await {
            Ok(output) => {
                let text = tool.render_for_llm(&output);
                results.push(ToolCallResult::success(call.clone(), output, text));
            }
            Err(error) => {
                results.push(ToolCallResult::failure(
                    call.clone(),
                    error.message().to_string(),
                ));
            }
        }
    }
    results
}
