//! Tool-call execution engine.
//!
//! [`execute_calls`] is the single place where a list of model-requested
//! [`ToolCall`]s is turned into [`ToolCallResult`]s. It is rail-free: the
//! `ReAct` loop fires the tool rails around each batch, so the engine only
//! executes.
//!
//! Execution policy:
//! - results always keep the model's call order, one result per call;
//! - failures (unknown tool, malformed arguments, tool error, timeout,
//!   cancellation) are results, never `Err`;
//! - when [`ToolExecutionOptions::parallel`] is set, runs of consecutive
//!   `parallel_safe` tools (at most [`MAX_PARALLEL_BATCH`]) are executed
//!   concurrently with [`futures::future::join_all`]; a tool that is not
//!   parallel-safe (or is unknown) is a barrier and runs on its own;
//! - every invocation is wrapped in [`tokio::time::timeout`] when a deadline
//!   applies, taken from the per-tool `crate::policy::ToolResilience` budget or
//!   else [`ToolExecutionOptions::timeout`];
//! - a call is retried up to its budget's attempt count when the tool errors or
//!   times out, sleeping the budget's exponential backoff between attempts;
//!   cancellation and validation failures are never retried, and cancellation
//!   stops retrying immediately;
//! - a successful result keeps the tool's text rendering plus the multimodal
//!   content parts [`extract_parts`](super::parts::extract_parts) finds in its
//!   raw output; failures carry no parts.
use super::{parse_tool_arguments, ToolCallResult, ToolExecutionOptions};
use crate::policy::ToolResiliencePolicy;
use futures::future::join_all;
use indexmap::IndexMap;
use openjiuwen_common::error::ErrorKind;
use openjiuwen_foundation::{Tool, ToolCall, ToolContext};
use serde_json::Value;
use std::sync::Arc;
use std::time::Duration;

/// Upper bound on one concurrent batch, so a large tool-call list cannot
/// exhaust the runtime with unbounded concurrency.
const MAX_PARALLEL_BATCH: usize = 8;

/// Executes `calls` against `tools` and returns one result per call.
///
/// This is [`execute_calls_with`] under the default [`ToolResiliencePolicy`]
/// (no per-tool timeout, a single attempt).
pub(crate) async fn execute_calls(
    tools: &IndexMap<String, Arc<dyn Tool>>,
    calls: &[ToolCall],
    options: &ToolExecutionOptions,
) -> Vec<ToolCallResult> {
    execute_calls_with(tools, calls, options, &ToolResiliencePolicy::default()).await
}

/// Executes `calls` like [`execute_calls`], applying each tool's resilience
/// budget.
///
/// Results keep the model's call order. Failures are results, never `Err`:
/// unknown tools, malformed arguments, timeouts, cancellation, and tool errors
/// all become [`ToolCallResult::failure`], so the model always receives a tool
/// message it can react to.
pub(crate) async fn execute_calls_with(
    tools: &IndexMap<String, Arc<dyn Tool>>,
    calls: &[ToolCall],
    options: &ToolExecutionOptions,
    resilience: &ToolResiliencePolicy,
) -> Vec<ToolCallResult> {
    let mut results = Vec::with_capacity(calls.len());
    let mut index = 0;
    while index < calls.len() {
        let batch_len = if options.parallel {
            parallel_batch_len(tools, &calls[index..])
        } else {
            1
        };
        if batch_len <= 1 {
            results.push(execute_one(tools, &calls[index], options, resilience).await);
            index += 1;
            continue;
        }
        let batch = &calls[index..index + batch_len];
        let futures = batch
            .iter()
            .map(|call| execute_one(tools, call, options, resilience));
        // `join_all` polls the futures in order and preserves that order in the
        // returned vector, so the result order stays deterministic.
        results.extend(join_all(futures).await);
        index += batch_len;
    }
    results
}

/// Returns the length of the run of consecutive, known, parallel-safe calls at
/// the head of `calls`, capped at [`MAX_PARALLEL_BATCH`].
///
/// A batch needs at least two calls to be worth running concurrently; any
/// other call (including an unknown tool, whose safety cannot be determined)
/// ends the run.
fn parallel_batch_len(tools: &IndexMap<String, Arc<dyn Tool>>, calls: &[ToolCall]) -> usize {
    let mut len = 0;
    for call in calls {
        if len == MAX_PARALLEL_BATCH {
            break;
        }
        match tools.get(&call.name) {
            Some(tool) if tool.is_parallel_safe() => len += 1,
            _ => break,
        }
    }
    if len < 2 {
        1
    } else {
        len
    }
}

/// Executes one call under its resilience budget, converting every failure
/// into a failed result.
async fn execute_one(
    tools: &IndexMap<String, Arc<dyn Tool>>,
    call: &ToolCall,
    options: &ToolExecutionOptions,
    resilience: &ToolResiliencePolicy,
) -> ToolCallResult {
    if options.cancellation.is_cancelled() {
        return ToolCallResult::failure(
            call.clone(),
            format!("Tool '{}' execution was cancelled by user.", call.name),
        );
    }
    let Some(tool) = tools.get(&call.name) else {
        return ToolCallResult::failure(call.clone(), format!("Tool '{}' not found", call.name));
    };
    let arguments = match parse_tool_arguments(&call.arguments) {
        Ok(arguments) => arguments,
        Err(error) => {
            return ToolCallResult::failure(call.clone(), error.message().to_string());
        }
    };
    let budget = resilience.for_tool(&call.name);
    // A per-tool timeout overrides the invocation-wide default.
    let timeout = budget.timeout.or(options.timeout);
    let attempts = budget.attempts();
    let mut ctx = ToolContext::new().with_metadata(options.metadata.clone());
    if let Some(session_id) = &options.session_id {
        // Tools observe the caller's cancellation signal, not a fresh token.
        ctx = ctx.with_session(session_id.clone());
    }
    if let Some(timeout) = timeout {
        ctx = ctx.with_timeout(timeout);
    }
    ctx.cancellation = options.cancellation.clone();
    let mut attempt = 1;
    loop {
        match invoke_once(tool.as_ref(), arguments.clone(), &ctx, timeout).await {
            Ok(output) => {
                let text = Arc::clone(tool).render_for_llm(&output);
                let parts = super::parts::extract_parts(&output);
                return ToolCallResult::success(call.clone(), output, text).with_parts(parts);
            }
            Err(failure) => {
                let retry = attempt < attempts
                    && retryable(&failure)
                    && !options.cancellation.is_cancelled();
                if !retry {
                    return failure_result(call, failure);
                }
                if let Some(delay) = budget.backoff_for(attempt) {
                    tokio::time::sleep(delay).await;
                }
                attempt += 1;
            }
        }
    }
}

/// Invokes `tool` once, mapping a tool error or the elapsed deadline to a
/// [`Failure`].
async fn invoke_once(
    tool: &dyn Tool,
    arguments: Value,
    ctx: &ToolContext,
    timeout: Option<Duration>,
) -> std::result::Result<Value, Failure> {
    match timeout {
        Some(limit) => match tokio::time::timeout(limit, tool.invoke(arguments, ctx)).await {
            Ok(result) => result.map_err(Failure::Error),
            Err(_) => Err(Failure::Timeout(limit)),
        },
        None => tool.invoke(arguments, ctx).await.map_err(Failure::Error),
    }
}

/// `true` when another attempt may succeed: tool errors and timeouts are
/// retryable, cancellation and validation failures are not.
fn retryable(failure: &Failure) -> bool {
    match failure {
        Failure::Timeout(_) => true,
        Failure::Error(error) => {
            !matches!(error.kind(), ErrorKind::Cancelled | ErrorKind::Validation)
        }
    }
}

/// The final failed result of a call, with the engine's message formats.
fn failure_result(call: &ToolCall, failure: Failure) -> ToolCallResult {
    match failure {
        Failure::Timeout(timeout) => ToolCallResult::failure(
            call.clone(),
            format!(
                "Tool '{}' timed out after {}s",
                call.name,
                timeout.as_secs_f64()
            ),
        ),
        Failure::Error(error) => ToolCallResult::failure(call.clone(), error.to_string()),
    }
}

/// Internal distinction between a tool error and the engine's own deadline.
enum Failure {
    /// The tool reported an error.
    Error(openjiuwen_common::error::Error),
    /// The per-call deadline elapsed.
    Timeout(Duration),
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use crate::policy::ToolResilience;
    use async_trait::async_trait;
    use openjiuwen_common::error::{Error, Result as ToolResult};
    use openjiuwen_foundation::{ContentPart, ImageUrl, Role, ToolCard};
    use parking_lot::Mutex;
    use serde_json::{json, Value};
    use std::collections::BTreeMap;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    /// Records `start:<name>` / `end:<name>` around a configurable delay.
    struct RecordingTool {
        card: ToolCard,
        log: Arc<Mutex<Vec<String>>>,
        delay: Duration,
        fail: bool,
    }

    #[async_trait]
    impl Tool for RecordingTool {
        fn card(&self) -> &ToolCard {
            &self.card
        }

        async fn invoke(&self, inputs: Value, _ctx: &ToolContext) -> ToolResult<Value> {
            self.log.lock().push(format!("start:{}", self.card.name));
            if !self.delay.is_zero() {
                tokio::time::sleep(self.delay).await;
            }
            self.log.lock().push(format!("end:{}", self.card.name));
            if self.fail {
                return Err(Error::tool("boom"));
            }
            Ok(inputs)
        }
    }

    fn recording_tool(
        name: &str,
        log: &Arc<Mutex<Vec<String>>>,
        delay: Duration,
        parallel_safe: bool,
    ) -> Arc<dyn Tool> {
        Arc::new(RecordingTool {
            card: ToolCard::new(name, "recording tool", json!({"type": "object"}))
                .with_parallel_safe(parallel_safe),
            log: Arc::clone(log),
            delay,
            fail: false,
        })
    }

    fn registry(tools: &[Arc<dyn Tool>]) -> IndexMap<String, Arc<dyn Tool>> {
        let mut map = IndexMap::new();
        for tool in tools {
            map.insert(tool.card().name.clone(), Arc::clone(tool));
        }
        map
    }

    /// Counts invocations and fails the first `failures` of them with `error`.
    struct FlakyTool {
        card: ToolCard,
        invocations: Arc<AtomicUsize>,
        failures: usize,
        error: fn() -> Error,
        delay: Duration,
    }

    #[async_trait]
    impl Tool for FlakyTool {
        fn card(&self) -> &ToolCard {
            &self.card
        }

        async fn invoke(&self, inputs: Value, _ctx: &ToolContext) -> ToolResult<Value> {
            let attempt = self.invocations.fetch_add(1, Ordering::SeqCst) + 1;
            if !self.delay.is_zero() {
                tokio::time::sleep(self.delay).await;
            }
            if attempt <= self.failures {
                return Err((self.error)());
            }
            Ok(inputs)
        }
    }

    fn tool_error() -> Error {
        Error::tool("boom")
    }

    fn validation_error() -> Error {
        Error::validation("bad")
    }

    fn flaky_tool(
        name: &str,
        invocations: &Arc<AtomicUsize>,
        failures: usize,
        error: fn() -> Error,
        delay: Duration,
    ) -> Arc<dyn Tool> {
        Arc::new(FlakyTool {
            card: ToolCard::new(name, "flaky tool", json!({"type": "object"})),
            invocations: Arc::clone(invocations),
            failures,
            error,
            delay,
        })
    }

    fn position(log: &Arc<Mutex<Vec<String>>>, entry: &str) -> usize {
        log.lock()
            .iter()
            .position(|item| item == entry)
            .unwrap_or_else(|| panic!("missing log entry {entry}"))
    }

    #[tokio::test]
    async fn parallel_safe_batch_starts_before_any_call_finishes() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let slow = recording_tool("slow", &log, Duration::from_millis(30), true);
        let fast = recording_tool("fast", &log, Duration::from_millis(0), true);
        let tools = registry(&[Arc::clone(&slow), Arc::clone(&fast)]);
        let calls = vec![ToolCall::new("slow"), ToolCall::new("fast")];
        let results = execute_calls(&tools, &calls, &ToolExecutionOptions::default()).await;
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|result| result.success));
        // Both futures are polled before `slow` yields: `fast` starts before
        // `slow` finishes.
        assert!(position(&log, "start:fast") < position(&log, "end:slow"));
        assert_eq!(results[0].call.name, "slow");
        assert_eq!(results[1].call.name, "fast");
    }

    #[tokio::test]
    async fn non_parallel_safe_call_is_a_barrier() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let first = recording_tool("first", &log, Duration::from_millis(20), true);
        let barrier = recording_tool("barrier", &log, Duration::from_millis(0), false);
        let third = recording_tool("third", &log, Duration::from_millis(0), true);
        let tools = registry(&[Arc::clone(&first), Arc::clone(&barrier), Arc::clone(&third)]);
        let calls = vec![
            ToolCall::new("first"),
            ToolCall::new("barrier"),
            ToolCall::new("third"),
        ];
        let results = execute_calls(&tools, &calls, &ToolExecutionOptions::default()).await;
        assert_eq!(results.len(), 3);
        assert!(results.iter().all(|result| result.success));
        assert!(position(&log, "end:first") < position(&log, "start:barrier"));
        assert!(position(&log, "end:barrier") < position(&log, "start:third"));
    }

    #[tokio::test]
    async fn parallel_disabled_runs_every_call_sequentially() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let a = recording_tool("a", &log, Duration::from_millis(5), true);
        let b = recording_tool("b", &log, Duration::from_millis(0), true);
        let tools = registry(&[Arc::clone(&a), Arc::clone(&b)]);
        let calls = vec![ToolCall::new("a"), ToolCall::new("b")];
        let options = ToolExecutionOptions {
            parallel: false,
            ..ToolExecutionOptions::default()
        };
        let results = execute_calls(&tools, &calls, &options).await;
        assert_eq!(results.len(), 2);
        assert!(position(&log, "end:a") < position(&log, "start:b"));
    }

    #[tokio::test]
    async fn multimodal_tool_output_keeps_its_text_and_parts() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let tools = registry(&[recording_tool("shot", &log, Duration::ZERO, true)]);
        let output = json!({"content": [
            {"type": "text", "text": "ok"},
            {"type": "image", "image_url": {"url": "data:image/png;base64,AAA"}},
        ]});
        let calls = vec![ToolCall::new("shot")
            .with_id("call-1")
            .with_arguments(output.clone())];
        let results = execute_calls(&tools, &calls, &ToolExecutionOptions::default()).await;
        assert_eq!(results.len(), 1);
        assert!(results[0].success);
        assert_eq!(
            results[0].text,
            r#"[{"type":"text","text":"ok"},{"type":"image","image_url":{"url":"data:image/png;base64,AAA"}}]"#
        );
        assert_eq!(results[0].parts.len(), 2);
        assert_eq!(
            results[0].parts[0],
            ContentPart::Text {
                text: "ok".to_owned()
            }
        );
        assert_eq!(
            results[0].parts[1],
            ContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: "data:image/png;base64,AAA".to_owned(),
                    detail: None,
                },
            }
        );
    }

    #[tokio::test]
    async fn text_only_tool_output_has_no_parts() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let tools = registry(&[recording_tool("echo", &log, Duration::ZERO, true)]);
        let calls = vec![ToolCall::new("echo").with_arguments(json!({"x": 1}))];
        let results = execute_calls(&tools, &calls, &ToolExecutionOptions::default()).await;
        assert!(results[0].success);
        assert_eq!(results[0].parts, Vec::new());
    }

    #[tokio::test]
    async fn failures_carry_no_parts() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let tools = registry(&[recording_tool("echo", &log, Duration::ZERO, true)]);
        let calls = vec![
            ToolCall::new("missing"),
            ToolCall {
                arguments: "not json".to_owned(),
                ..ToolCall::new("echo")
            },
        ];
        let results = execute_calls(&tools, &calls, &ToolExecutionOptions::default()).await;
        assert!(results.iter().all(|result| result.parts.is_empty()));
    }

    #[tokio::test]
    async fn unknown_tool_and_bad_arguments_are_failures() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let echo = recording_tool("echo", &log, Duration::ZERO, true);
        let tools = registry(&[echo]);
        let calls = vec![
            ToolCall::new("missing"),
            ToolCall {
                arguments: "not json".to_owned(),
                ..ToolCall::new("echo")
            },
        ];
        let results = execute_calls(&tools, &calls, &ToolExecutionOptions::default()).await;
        assert_eq!(results.len(), 2);
        assert_eq!(
            results[0].error.as_deref(),
            Some("Tool 'missing' not found")
        );
        assert!(results[1]
            .error
            .as_deref()
            .is_some_and(|error| error.contains("Invalid tool arguments JSON")));
    }

    #[tokio::test]
    async fn tool_errors_become_failure_results_with_a_tool_message() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let failing = Arc::new(RecordingTool {
            card: ToolCard::new("failing", "fails", json!({"type": "object"})),
            log: Arc::clone(&log),
            delay: Duration::ZERO,
            fail: true,
        });
        let tools = registry(&[failing]);
        let calls = vec![ToolCall::new("failing").with_id("call-1")];
        let results = execute_calls(&tools, &calls, &ToolExecutionOptions::default()).await;
        assert_eq!(results.len(), 1);
        assert!(!results[0].success);
        assert!(results[0].text.contains("boom"));
        let message = results[0].message();
        assert_eq!(message.role, Role::Tool);
        assert_eq!(message.name.as_deref(), Some("failing"));
        assert_eq!(message.tool_call_id.as_deref(), Some("call-1"));
    }

    #[tokio::test]
    async fn timeouts_become_failure_results() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let slow = recording_tool("slow", &log, Duration::from_millis(100), true);
        let tools = registry(&[slow]);
        let calls = vec![ToolCall::new("slow")];
        let options = ToolExecutionOptions {
            timeout: Some(Duration::from_millis(10)),
            ..ToolExecutionOptions::default()
        };
        let results = execute_calls(&tools, &calls, &options).await;
        assert_eq!(results.len(), 1);
        assert!(!results[0].success);
        assert!(results[0]
            .error
            .as_deref()
            .is_some_and(|error| error.starts_with("Tool 'slow' timed out after")));
    }

    #[tokio::test]
    async fn cancellation_before_a_call_is_a_failure_result() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let echo = recording_tool("echo", &log, Duration::ZERO, true);
        let tools = registry(&[echo]);
        let options = ToolExecutionOptions::default();
        options.cancellation.cancel();
        let calls = vec![ToolCall::new("echo"), ToolCall::new("echo")];
        let results = execute_calls(&tools, &calls, &options).await;
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|result| !result.success));
        assert!(results.iter().all(|result| result
            .error
            .as_deref()
            .is_some_and(|error| error == "Tool 'echo' execution was cancelled by user.")));
        assert!(log.lock().is_empty());
    }

    #[tokio::test]
    async fn every_result_keeps_the_model_call_order() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let a = recording_tool("a", &log, Duration::from_millis(20), true);
        let b = recording_tool("b", &log, Duration::from_millis(0), true);
        let c = recording_tool("c", &log, Duration::from_millis(0), true);
        let tools = registry(&[Arc::clone(&a), Arc::clone(&b), Arc::clone(&c)]);
        let calls = vec![
            ToolCall::new("a").with_id("1"),
            ToolCall::new("b").with_id("2"),
            ToolCall::new("c").with_id("3"),
        ];
        let results = execute_calls(&tools, &calls, &ToolExecutionOptions::default()).await;
        let ids: Vec<_> = results
            .iter()
            .map(|result| result.call.id.clone().unwrap_or_default())
            .collect();
        assert_eq!(ids, vec!["1", "2", "3"]);
    }

    #[tokio::test]
    async fn batches_are_capped_but_all_calls_run() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let tools = registry(&[recording_tool("echo", &log, Duration::ZERO, true)]);
        let calls: Vec<_> = (0..MAX_PARALLEL_BATCH + 3)
            .map(|index| ToolCall::new("echo").with_id(index.to_string()))
            .collect();
        let results = execute_calls(&tools, &calls, &ToolExecutionOptions::default()).await;
        assert_eq!(results.len(), calls.len());
        assert!(results.iter().all(|result| result.success));
        assert_eq!(log.lock().len(), calls.len() * 2);
    }

    #[test]
    fn batch_length_counts_only_consecutive_parallel_safe_calls() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let tools = registry(&[
            recording_tool("a", &log, Duration::ZERO, true),
            recording_tool("b", &log, Duration::ZERO, false),
            recording_tool("c", &log, Duration::ZERO, true),
            recording_tool("d", &log, Duration::ZERO, true),
        ]);
        let calls = vec![
            ToolCall::new("a"),
            ToolCall::new("a"),
            ToolCall::new("b"),
            ToolCall::new("c"),
            ToolCall::new("d"),
            ToolCall::new("missing"),
            ToolCall::new("a"),
        ];
        assert_eq!(parallel_batch_len(&tools, &calls), 2);
        assert_eq!(parallel_batch_len(&tools, &calls[2..]), 1);
        assert_eq!(parallel_batch_len(&tools, &calls[3..]), 2);
        assert_eq!(parallel_batch_len(&tools, &calls[5..]), 1);
    }

    #[tokio::test]
    async fn context_carries_session_metadata_and_deadline() {
        type SeenContext = (Option<String>, BTreeMap<String, Value>, Option<Duration>);
        struct ContextTool {
            card: ToolCard,
            seen: Arc<Mutex<Option<SeenContext>>>,
        }

        #[async_trait]
        impl Tool for ContextTool {
            fn card(&self) -> &ToolCard {
                &self.card
            }

            async fn invoke(&self, _inputs: Value, ctx: &ToolContext) -> ToolResult<Value> {
                *self.seen.lock() =
                    Some((ctx.session_id.clone(), ctx.metadata.clone(), ctx.timeout));
                assert_ne!(ctx.invocation_id, "");
                Ok(json!("ok"))
            }
        }

        let seen = Arc::new(Mutex::new(None));
        let tool: Arc<dyn Tool> = Arc::new(ContextTool {
            card: ToolCard::new("ctx", "context", json!({"type": "object"})),
            seen: Arc::clone(&seen),
        });
        let tools = registry(&[tool]);
        let options = ToolExecutionOptions {
            session_id: Some("session-1".to_owned()),
            metadata: BTreeMap::from([("k".to_owned(), json!("v"))]),
            timeout: Some(Duration::from_secs(5)),
            ..ToolExecutionOptions::default()
        };
        let results = execute_calls(&tools, &[ToolCall::new("ctx")], &options).await;
        assert!(results[0].success);
        let (session_id, metadata, timeout) = seen.lock().clone().unwrap();
        assert_eq!(session_id.as_deref(), Some("session-1"));
        assert_eq!(metadata["k"], json!("v"));
        assert_eq!(timeout, Some(Duration::from_secs(5)));
    }

    fn retry_policy(attempts: u32) -> ToolResiliencePolicy {
        let mut policy = ToolResiliencePolicy::new();
        policy.set_default(ToolResilience::new().with_attempts(attempts));
        policy
    }

    #[tokio::test]
    async fn retries_a_flaky_tool_until_it_succeeds() {
        let invocations = Arc::new(AtomicUsize::new(0));
        let tools = registry(&[flaky_tool(
            "flaky",
            &invocations,
            2,
            tool_error,
            Duration::ZERO,
        )]);
        let results = execute_calls_with(
            &tools,
            &[ToolCall::new("flaky")],
            &ToolExecutionOptions::default(),
            &retry_policy(3),
        )
        .await;
        assert_eq!(results.len(), 1);
        assert!(results[0].success);
        assert_eq!(invocations.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn stops_after_the_attempt_budget_is_spent() {
        let invocations = Arc::new(AtomicUsize::new(0));
        let tools = registry(&[flaky_tool(
            "failing",
            &invocations,
            usize::MAX,
            tool_error,
            Duration::ZERO,
        )]);
        let results = execute_calls_with(
            &tools,
            &[ToolCall::new("failing")],
            &ToolExecutionOptions::default(),
            &retry_policy(2),
        )
        .await;
        assert_eq!(results.len(), 1);
        assert!(!results[0].success);
        assert!(results[0].text.contains("boom"));
        assert_eq!(invocations.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn default_policy_attempts_once() {
        let invocations = Arc::new(AtomicUsize::new(0));
        let tools = registry(&[flaky_tool(
            "failing",
            &invocations,
            usize::MAX,
            tool_error,
            Duration::ZERO,
        )]);
        let results = execute_calls_with(
            &tools,
            &[ToolCall::new("failing")],
            &ToolExecutionOptions::default(),
            &ToolResiliencePolicy::default(),
        )
        .await;
        assert_eq!(results.len(), 1);
        assert!(!results[0].success);
        assert_eq!(invocations.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn validation_failures_are_not_retried() {
        let invocations = Arc::new(AtomicUsize::new(0));
        let tools = registry(&[flaky_tool(
            "invalid",
            &invocations,
            usize::MAX,
            validation_error,
            Duration::ZERO,
        )]);
        let results = execute_calls_with(
            &tools,
            &[ToolCall::new("invalid")],
            &ToolExecutionOptions::default(),
            &retry_policy(3),
        )
        .await;
        assert_eq!(results.len(), 1);
        assert!(!results[0].success);
        assert!(results[0].text.contains("bad"));
        assert_eq!(invocations.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn per_tool_timeout_overrides_the_invocation_timeout() {
        let invocations = Arc::new(AtomicUsize::new(0));
        let tools = registry(&[flaky_tool(
            "slow",
            &invocations,
            0,
            tool_error,
            Duration::from_millis(200),
        )]);
        let mut policy = ToolResiliencePolicy::new();
        policy.set(
            "slow",
            ToolResilience::new().with_timeout(Duration::from_millis(20)),
        );
        let options = ToolExecutionOptions {
            timeout: Some(Duration::from_secs(5)),
            ..ToolExecutionOptions::default()
        };
        let results = execute_calls_with(&tools, &[ToolCall::new("slow")], &options, &policy).await;
        assert_eq!(results.len(), 1);
        assert!(!results[0].success);
        assert!(results[0]
            .error
            .as_deref()
            .is_some_and(|error| error.contains("timed out")));
        assert_eq!(invocations.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn cancellation_beats_the_attempt_budget() {
        let invocations = Arc::new(AtomicUsize::new(0));
        let tools = registry(&[flaky_tool(
            "echo",
            &invocations,
            usize::MAX,
            tool_error,
            Duration::ZERO,
        )]);
        let options = ToolExecutionOptions::default();
        options.cancellation.cancel();
        let results =
            execute_calls_with(&tools, &[ToolCall::new("echo")], &options, &retry_policy(3)).await;
        assert_eq!(results.len(), 1);
        assert!(!results[0].success);
        assert_eq!(
            results[0].error.as_deref(),
            Some("Tool 'echo' execution was cancelled by user.")
        );
        assert_eq!(invocations.load(Ordering::SeqCst), 0);
    }
}
