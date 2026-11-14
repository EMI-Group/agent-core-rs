//! Harness rails that record model turns and tool intents/outcomes into the execution journal.
//!
//! [`ExecutionRecordRail`] is the durability writer of a persistent execution: it turns the
//! harness lifecycle hooks of [`HarnessRail`] into [`EntryPayload`] lines of an [`ExecutionLog`],
//! so a resumed session replays what happened instead of trusting an in-memory transcript. It
//! appends a `tool_intent` before a non-read-only tool call, the matching `tool_completed` outcome
//! afterwards (also on the exception path) and one `model_turn` per model call.
//!
//! [`ModelCallRail`] answers the admission question the journal cannot: a host implements
//! [`ModelCallGate`] to admit or deny a model call against its own budget service and to receive
//! the resulting usage. The rail journals the admitted turn itself, so the journal stays complete
//! even when the gate only meters calls.
//!
//! Both rails are bookkeeping only: they never call the model, never execute a tool and never hold
//! a lock across an `.await`. The one error that stops a run is a denial from the gate, which
//! surfaces as [`Error::Config`] for the host to turn into a pause; gate *report* failures are
//! logged and swallowed, because losing a metering receipt must not lose the execution.
// Size note: the implementation is ~537 lines; the file exceeds the crate's 1000-line guideline
// only because the inline test module below covers the rails' hook matrix against real
// temporary journals.
use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_agent::react::model_step::ModelReply;
use openjiuwen_agent::ToolCallResult;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::{ToolCall, Usage};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use super::entry::{
    args_summary, operation_id, params_digest, result_summary, EntryPayload, ModelUsage,
    ToolStatus, SUMMARY_LIMIT,
};
use super::identity::ExecutionIdentity;
use super::log::ExecutionLog;
use super::util::{digest_hex, truncate_chars};
use crate::rails::{HarnessRail, HarnessRailContext};

/// Name [`ExecutionRecordRail`] registers under.
const RECORD_RAIL_NAME: &str = "persistent_execution_record";

/// Name [`ModelCallRail`] registers under.
const MODEL_RAIL_NAME: &str = "persistent_model_call";

/// Key in [`HarnessRailContext::extra`] holding the admitted model-call id.
pub const MODEL_CALL_ID_EXTRA_KEY: &str = "persistent.model_call_id";

/// Tool names that cannot change persistent state, so an intent for them need not be journalled.
///
/// The list mirrors the Python read-only set: file and group-history reads, the share bundle and
/// the context-management tools.
pub const READ_ONLY_TOOLS: &[&str] = &[
    "read_file",
    "list_files",
    "share_file",
    "read_execution_record",
    "search_group_messages",
    "search_session_history",
    "new_context",
];

/// Outcome text recorded when a tool raised instead of returning a result.
const EXCEPTION_REASON: &str = "Tool raised an exception; result content omitted";

/// Reason recorded when a model call reports no counters.
const NO_USAGE_REASON: &str = "model returned no usage";

/// Note recorded when a model call fails.
const MODEL_FAILED_REASON: &str = "model call failed";

/// Returns whether `name` is one of the [`READ_ONLY_TOOLS`].
#[must_use]
pub fn is_read_only_tool(name: &str) -> bool {
    READ_ONLY_TOOLS.contains(&name)
}

/// How tool calls are journalled.
///
/// The default records only calls that can change state (`record_read_only` off) and bounds every
/// result summary at [`SUMMARY_LIMIT`] characters.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ToolRecordPolicy {
    /// Whether read-only tool calls also get a `tool_intent` entry.
    ///
    /// They always get their `tool_completed` outcome; the switch only controls the intent, which
    /// is what a recovery decision needs in order to replay a call that may have had an effect.
    pub record_read_only: bool,
    /// Maximum number of characters kept in a recorded result or error summary.
    pub summarize_limit: usize,
}

impl Default for ToolRecordPolicy {
    fn default() -> Self {
        Self {
            record_read_only: false,
            summarize_limit: SUMMARY_LIMIT,
        }
    }
}

impl ToolRecordPolicy {
    /// Builds the default policy: no read-only intents, [`SUMMARY_LIMIT`]-character summaries.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

/// What the rail remembers about one dispatched call between its two hooks.
#[derive(Debug, Clone)]
struct ToolCallState {
    /// Operation id of the journalled intent; `None` when the intent was not recorded.
    operation_id: Option<String>,
    /// Digest of the canonical arguments; `None` when the intent was not recorded.
    params_digest: Option<String>,
    /// Redacted, bounded argument summary reused by the outcome entry.
    args_summary: String,
    /// Whether the call was dispatched; a state is only ever stored for a dispatched call.
    dispatched: bool,
    /// Whether the outcome has already been journalled.
    completed: bool,
}

/// Journals model turns and tool intents/outcomes of one execution.
///
/// Dispatch priority is `5`, below every kernel rail, so the journal observes the calls the other
/// rails allowed rather than the ones they would have skipped. One instance belongs to one
/// [`ExecutionIdentity`] and one [`ExecutionLog`].
///
/// Pending calls are keyed by tool-call id behind a [`Mutex`], so concurrent tool batches cannot
/// lose each other's intent; the guard is never held across an `.await`.
pub struct ExecutionRecordRail {
    /// Journal the entries are appended to.
    log: Arc<ExecutionLog>,
    /// Execution the journal and the operation ids belong to.
    identity: ExecutionIdentity,
    /// What is journalled and how long a summary may be.
    policy: ToolRecordPolicy,
    /// Calls dispatched but not yet completed, keyed by tool-call id.
    state: Mutex<BTreeMap<String, ToolCallState>>,
}

impl ExecutionRecordRail {
    /// Builds a rail writing `log` for `identity` under the default [`ToolRecordPolicy`].
    #[must_use]
    pub fn new(log: Arc<ExecutionLog>, identity: ExecutionIdentity) -> Self {
        Self {
            log,
            identity,
            policy: ToolRecordPolicy::default(),
            state: Mutex::new(BTreeMap::new()),
        }
    }

    /// Returns this rail with `policy` as its recording policy.
    #[must_use]
    pub fn with_policy(mut self, policy: ToolRecordPolicy) -> Self {
        self.policy = policy;
        self
    }

    /// Returns the pending state of `call_id`, when the call was dispatched through this rail.
    fn pending(&self, call_id: &str) -> Option<ToolCallState> {
        self.state.lock().get(call_id).cloned()
    }

    /// Marks the outcome of `call_id` as journalled, so a repeat hook is a no-op.
    fn mark_completed(&self, call_id: &str) {
        if let Some(state) = self.state.lock().get_mut(call_id) {
            state.completed = true;
        }
    }

    /// Returns the outcome summary of `result`, bounded by the policy limit.
    ///
    /// A failure records its error text — the only place the failure reason is available — while a
    /// success records the redacted output, or the redacted text when the tool reported none.
    fn outcome_summary(&self, result: &ToolCallResult) -> String {
        let limit = self.policy.summarize_limit;
        if !result.success {
            if let Some(error) = result.error.as_deref() {
                return truncate_chars(error, limit);
            }
        }
        let summary = result.output.as_ref().map_or_else(
            || result_summary(&Value::String(result.text.clone())),
            result_summary,
        );
        truncate_chars(&summary, limit)
    }
}

/// Returns the journal key of `call`.
///
/// Providers normally assign a call id. A provider that omits it (a stream fragment, a local
/// model) gets a synthetic but stable key instead, so the intent and its outcome still match.
fn call_key(call: &ToolCall) -> String {
    match &call.id {
        Some(id) if !id.is_empty() => id.clone(),
        _ => format!("{}:{}", call.name, call.index.unwrap_or(0)),
    }
}

/// Maps provider `usage` onto a journal [`ModelUsage`].
///
/// A provider that reports nothing leaves every counter at zero; that is recorded as *unknown*, so
/// a reader never mistakes it for a real zero-token call.
fn model_usage(usage: &Usage) -> ModelUsage {
    if usage.total_tokens > 0 || usage.input_tokens > 0 || usage.output_tokens > 0 {
        ModelUsage::known(usage.input_tokens, usage.output_tokens)
    } else {
        ModelUsage::unknown(NO_USAGE_REASON)
    }
}

/// Returns the response text of `reply`.
fn reply_text(reply: &ModelReply) -> String {
    reply.message.text().into_owned()
}

/// Returns the deterministic id of the model call of `iteration`.
///
/// Python mints a `uuid4` here; the journal has no use for a random id, and a deterministic one
/// survives a retry of the same iteration. The id stays opaque to every consumer.
#[must_use]
pub fn deterministic_model_call_id(identity: &ExecutionIdentity, iteration: usize) -> String {
    let value = json!({
        "namespace_id": identity.namespace_id,
        "agent_id": identity.agent_id,
        "session_id": identity.session_id,
        "request_id": identity.request_id,
        "iteration": iteration,
    });
    let digest = digest_hex(&value);
    format!("mc_{}", digest.chars().take(32).collect::<String>())
}

#[async_trait]
impl HarnessRail for ExecutionRecordRail {
    fn name(&self) -> &str {
        RECORD_RAIL_NAME
    }

    fn priority(&self) -> i32 {
        5
    }

    async fn before_tool_call(&self, ctx: &mut HarnessRailContext) -> Result<()> {
        let Some(call) = ctx.base.tool_call.clone() else {
            return Ok(());
        };
        let call_id = call_key(&call);
        let summary = args_summary(&call.arguments);
        if is_read_only_tool(&call.name) && !self.policy.record_read_only {
            self.state.lock().insert(
                call_id,
                ToolCallState {
                    operation_id: None,
                    params_digest: None,
                    args_summary: summary,
                    dispatched: true,
                    completed: false,
                },
            );
            return Ok(());
        }
        // A malformed argument payload is a caller error, not a journaling error: the intent is
        // refused, so the journal never claims an argument digest it does not have.
        let digest = params_digest(&call.arguments)?;
        // A retry may re-dispatch a call whose intent is already journalled and whose outcome is
        // still pending; the digest tells the two dispatches apart, and a matching digest means
        // there is nothing new to record.
        if let Some(state) = self.pending(&call_id) {
            if !state.completed && state.params_digest.as_deref() == Some(digest.as_str()) {
                return Ok(());
            }
        }
        let operation_id = operation_id(&self.identity, &call_id);
        self.log
            .append(EntryPayload::tool_intent(
                call_id.clone(),
                call.name.clone(),
                summary.clone(),
                digest.clone(),
                operation_id.clone(),
            ))
            .await?;
        self.state.lock().insert(
            call_id,
            ToolCallState {
                operation_id: Some(operation_id),
                params_digest: Some(digest),
                args_summary: summary,
                dispatched: true,
                completed: false,
            },
        );
        Ok(())
    }

    async fn after_tool_call(&self, ctx: &mut HarnessRailContext) -> Result<()> {
        let (Some(call), Some(result)) = (ctx.base.tool_call.clone(), ctx.base.tool_result.clone())
        else {
            return Ok(());
        };
        let call_id = call_key(&call);
        let Some(state) = self.pending(&call_id) else {
            return Ok(());
        };
        if !state.dispatched || state.completed {
            return Ok(());
        }
        // A failed call that had an intent may already have had its effect, so its outcome is
        // unknown rather than failed; without an intent nothing could have happened.
        let status = if result.success {
            ToolStatus::Succeeded
        } else if state.operation_id.is_some() {
            ToolStatus::Unknown
        } else {
            ToolStatus::Failed
        };
        let summary = self.outcome_summary(&result);
        let mut payload = EntryPayload::tool_completed(
            call_id.clone(),
            call.name.clone(),
            status,
            state.args_summary.clone(),
            summary,
        );
        if let Some(operation_id) = state.operation_id.clone() {
            payload = payload.with_operation_id(operation_id);
        }
        self.log.append(payload).await?;
        self.mark_completed(&call_id);
        Ok(())
    }

    async fn on_tool_exception(&self, ctx: &mut HarnessRailContext) -> Result<()> {
        let Some(call) = ctx.base.tool_call.clone() else {
            return Ok(());
        };
        let call_id = call_key(&call);
        let Some(state) = self.pending(&call_id) else {
            return Ok(());
        };
        if !state.dispatched || state.completed {
            return Ok(());
        }
        // No result value exists on this path, so the journal records the shape of the outcome and
        // never fabricates a summary of content it does not have.
        self.log
            .append(EntryPayload::ToolCompleted {
                tool_call_id: call_id.clone(),
                tool_name: call.name.clone(),
                args_summary: state.args_summary.clone(),
                status: ToolStatus::Unknown,
                result_summary: EXCEPTION_REASON.to_string(),
                operation_id: state.operation_id.clone(),
                reason: Some(EXCEPTION_REASON.to_string()),
            })
            .await?;
        self.mark_completed(&call_id);
        Ok(())
    }

    async fn after_model_call(&self, ctx: &mut HarnessRailContext) -> Result<()> {
        let Some(reply) = ctx.base.response.as_ref() else {
            return Ok(());
        };
        let text = reply_text(reply);
        let usage = model_usage(&reply.usage);
        let call_id = deterministic_model_call_id(&self.identity, ctx.base.iteration);
        self.log
            .append(EntryPayload::model_turn(call_id, text, usage))
            .await?;
        Ok(())
    }
}

/// How a host admits and meters the model calls of a persistent execution.
///
/// A budget service, a reservation store or an in-process counter implements this trait; the rail
/// calls [`ModelCallGate::admit`] before every model call and [`ModelCallGate::report`] after it.
#[async_trait]
pub trait ModelCallGate: Send + Sync {
    /// Decides whether the model call `call_id` of `identity` may run.
    ///
    /// # Errors
    /// Any error the gate reports aborts the call: the rail propagates it unchanged, so a gate
    /// that cannot decide fails closed.
    async fn admit(&self, identity: &ExecutionIdentity, call_id: &str) -> Result<ModelAdmission>;

    /// Receives the token usage of the completed model call `call_id`.
    ///
    /// The default accepts every receipt, which is the right behaviour for a read-only meter.
    ///
    /// # Errors
    /// The rail logs a failure of this hook and does not abort the run.
    async fn report(
        &self,
        _identity: &ExecutionIdentity,
        _call_id: &str,
        _usage: &ModelUsage,
    ) -> Result<()> {
        Ok(())
    }
}

/// Answer of a [`ModelCallGate`] to one admission request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum ModelAdmission {
    /// The call may run.
    Admitted {
        /// Opaque reference of the reservation the gate took, when it took one.
        #[serde(default)]
        reservation_ref: Option<String>,
    },
    /// The call may not run.
    Denied {
        /// Human-readable reason, rendered into the error the host turns into a pause.
        reason: String,
    },
}

/// Admits model calls through a host [`ModelCallGate`] and journals the admitted turns.
///
/// Dispatch priority is `10`, so the gate runs after the kernel rails and observes their final
/// decision to call the model. A denial returns [`Error::Config`] instead of pausing directly: the
/// task loop owns pausing, and this rail owns only the decision's record.
pub struct ModelCallRail {
    /// Admission and metering seam owned by the host.
    gate: Arc<dyn ModelCallGate>,
    /// Journal the admitted turns are appended to.
    log: Arc<ExecutionLog>,
    /// Execution the gate is asked about.
    identity: ExecutionIdentity,
}

impl ModelCallRail {
    /// Builds a rail admitting through `gate` and journalling into `log`.
    #[must_use]
    pub fn new(
        gate: Arc<dyn ModelCallGate>,
        log: Arc<ExecutionLog>,
        identity: ExecutionIdentity,
    ) -> Self {
        Self {
            gate,
            log,
            identity,
        }
    }

    /// Returns the call id the admission hook stored, or rebuilds it deterministically.
    ///
    /// Rebuilding keeps the rail correct when the two hooks observe different contexts, which
    /// happens when a host fires them from different layers of the loop.
    fn call_id(&self, ctx: &HarnessRailContext) -> String {
        ctx.base
            .extra
            .get(MODEL_CALL_ID_EXTRA_KEY)
            .and_then(Value::as_str)
            .map_or_else(
                || deterministic_model_call_id(&self.identity, ctx.base.iteration),
                str::to_owned,
            )
    }
}

#[async_trait]
impl HarnessRail for ModelCallRail {
    fn name(&self) -> &str {
        MODEL_RAIL_NAME
    }

    fn priority(&self) -> i32 {
        10
    }

    async fn before_model_call(&self, ctx: &mut HarnessRailContext) -> Result<()> {
        let call_id = deterministic_model_call_id(&self.identity, ctx.base.iteration);
        match self.gate.admit(&self.identity, &call_id).await? {
            ModelAdmission::Admitted { .. } => {
                ctx.base
                    .extra
                    .insert(MODEL_CALL_ID_EXTRA_KEY.to_string(), Value::String(call_id));
                Ok(())
            }
            ModelAdmission::Denied { reason } => Err(Error::config(format!(
                "persistent model call denied: {reason}"
            ))),
        }
    }

    async fn after_model_call(&self, ctx: &mut HarnessRailContext) -> Result<()> {
        let Some(reply) = ctx.base.response.as_ref() else {
            return Ok(());
        };
        let text = reply_text(reply);
        let usage = model_usage(&reply.usage);
        let call_id = self.call_id(ctx);
        self.log
            .append(EntryPayload::model_turn(
                call_id.clone(),
                text,
                usage.clone(),
            ))
            .await?;
        if let Err(error) = self.gate.report(&self.identity, &call_id, &usage).await {
            tracing::warn!(
                call_id,
                error = %error,
                "persistent model call receipt was not reported; continuing"
            );
        }
        Ok(())
    }

    async fn on_model_exception(&self, ctx: &mut HarnessRailContext) -> Result<()> {
        let call_id = self.call_id(ctx);
        let usage = ModelUsage::unknown(MODEL_FAILED_REASON);
        if let Err(error) = self.gate.report(&self.identity, &call_id, &usage).await {
            tracing::warn!(
                call_id,
                error = %error,
                "persistent model call failure was not reported; continuing"
            );
        }
        self.log
            .append(EntryPayload::note(MODEL_FAILED_REASON))
            .await?;
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::sync::Arc;

    use async_trait::async_trait;
    use openjiuwen_agent::rails::RailEvent;
    use openjiuwen_agent::ToolCallResult;
    use openjiuwen_common::error::{Error, Result};
    use openjiuwen_foundation::{BaseMessage, ToolCall, Usage};
    use serde_json::{json, Value};

    use super::{
        args_summary, deterministic_model_call_id, is_read_only_tool, model_usage, operation_id,
        params_digest, result_summary, ExecutionRecordRail, ModelAdmission, ModelCallGate,
        ModelCallRail, ModelReply, ToolRecordPolicy, EXCEPTION_REASON, MODEL_CALL_ID_EXTRA_KEY,
        READ_ONLY_TOOLS,
    };
    use crate::persistent_agent::entry::{EntryKind, EntryPayload, ModelUsage, ToolStatus};
    use crate::persistent_agent::identity::ExecutionIdentity;
    use crate::persistent_agent::log::{ExecutionLog, LogCapacity};
    use crate::persistent_agent::ManualClock;
    use crate::rails::{HarnessRail, HarnessRailContext};

    /// Fixed clock reading of every test.
    const STAMP: i64 = 1_700_000_000_000;

    fn identity() -> ExecutionIdentity {
        ExecutionIdentity::new("ns", "agent", "session", "request", "execution", "source").unwrap()
    }

    fn other_identity() -> ExecutionIdentity {
        // The call id hashes the request, not the execution, so the request is what differs here.
        ExecutionIdentity::new("ns", "agent", "session", "request2", "exec2", "source").unwrap()
    }

    fn call(id: &str, name: &str, arguments: &Value) -> ToolCall {
        ToolCall {
            id: Some(id.to_owned()),
            name: name.to_owned(),
            arguments: arguments.to_string(),
            ..ToolCall::default()
        }
    }

    fn reply(text: &str, input: u64, output: u64) -> ModelReply {
        ModelReply {
            message: BaseMessage::assistant(text),
            usage: Usage::new(input, output),
            ..ModelReply::default()
        }
    }

    fn intent_ctx(tool_call: ToolCall) -> HarnessRailContext {
        let mut ctx = HarnessRailContext::for_event(RailEvent::BeforeToolCall);
        ctx.base.tool_call = Some(tool_call);
        ctx
    }

    fn completed_ctx(tool_call: ToolCall, result: ToolCallResult) -> HarnessRailContext {
        let mut ctx = HarnessRailContext::for_event(RailEvent::AfterToolCall);
        ctx.base.tool_call = Some(tool_call);
        ctx.base.tool_result = Some(result);
        ctx
    }

    async fn open_log(dir: &tempfile::TempDir) -> Arc<ExecutionLog> {
        Arc::new(
            ExecutionLog::open_with_clock(
                dir.path(),
                identity(),
                LogCapacity::default(),
                Arc::new(ManualClock::new(STAMP)),
            )
            .await
            .unwrap(),
        )
    }

    async fn payloads(log: &ExecutionLog) -> Vec<EntryPayload> {
        log.read()
            .await
            .unwrap()
            .entries
            .into_iter()
            .map(|entry| entry.payload)
            .collect()
    }

    #[tokio::test]
    async fn tool_intent_and_completed_share_the_stable_operation_id() {
        let dir = tempfile::TempDir::new().unwrap();
        let log = open_log(&dir).await;
        let rail = ExecutionRecordRail::new(Arc::clone(&log), identity());
        let tool_call = call(
            "call_1",
            "write_file",
            &json!({"path": "a.txt", "content": "hi"}),
        );

        rail.before_tool_call(&mut intent_ctx(tool_call.clone()))
            .await
            .unwrap();
        let result = ToolCallResult::success(tool_call.clone(), json!({"ok": true}), "written");
        rail.after_tool_call(&mut completed_ctx(tool_call.clone(), result))
            .await
            .unwrap();

        let expected_operation = operation_id(&identity(), "call_1");
        let expected_args = args_summary(&tool_call.arguments);
        let expected_digest = params_digest(&tool_call.arguments).unwrap();
        let entries = payloads(&log).await;
        assert_eq!(entries.len(), 2);
        match &entries[0] {
            EntryPayload::ToolIntent {
                tool_call_id,
                tool_name,
                args_summary,
                params_digest,
                operation_id,
                target_ref,
            } => {
                assert_eq!(tool_call_id, "call_1");
                assert_eq!(tool_name, "write_file");
                assert_eq!(args_summary, &expected_args);
                assert_eq!(params_digest, &expected_digest);
                assert_eq!(operation_id, &expected_operation);
                assert!(target_ref.is_none());
            }
            other => panic!("expected a tool intent, got {other:?}"),
        }
        match &entries[1] {
            EntryPayload::ToolCompleted {
                tool_call_id,
                tool_name,
                args_summary,
                status,
                result_summary: recorded,
                operation_id,
                reason,
            } => {
                assert_eq!(tool_call_id, "call_1");
                assert_eq!(tool_name, "write_file");
                assert_eq!(args_summary, &expected_args);
                assert_eq!(*status, ToolStatus::Succeeded);
                assert_eq!(*recorded, result_summary(&json!({"ok": true})));
                assert_eq!(operation_id.as_deref(), Some(expected_operation.as_str()));
                assert!(reason.is_none());
            }
            other => panic!("expected a tool outcome, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn read_only_tool_records_a_failed_outcome_without_an_intent() {
        let dir = tempfile::TempDir::new().unwrap();
        let log = open_log(&dir).await;
        let rail = ExecutionRecordRail::new(Arc::clone(&log), identity());
        let tool_call = call("call_2", "read_file", &json!({"path": "a.txt"}));

        rail.before_tool_call(&mut intent_ctx(tool_call.clone()))
            .await
            .unwrap();
        assert!(payloads(&log).await.is_empty(), "no entries expected");

        let result = ToolCallResult::failure(tool_call.clone(), "missing file");
        rail.after_tool_call(&mut completed_ctx(tool_call, result))
            .await
            .unwrap();

        let entries = payloads(&log).await;
        assert_eq!(entries.len(), 1);
        match &entries[0] {
            EntryPayload::ToolCompleted {
                status,
                operation_id,
                result_summary,
                reason,
                ..
            } => {
                assert_eq!(*status, ToolStatus::Failed);
                assert!(operation_id.is_none());
                assert_eq!(result_summary, "missing file");
                assert!(reason.is_none());
            }
            other => panic!("expected a tool outcome, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn recording_read_only_calls_marks_a_failure_unknown() {
        let dir = tempfile::TempDir::new().unwrap();
        let log = open_log(&dir).await;
        let policy = ToolRecordPolicy {
            record_read_only: true,
            ..ToolRecordPolicy::new()
        };
        let rail = ExecutionRecordRail::new(Arc::clone(&log), identity()).with_policy(policy);
        let tool_call = call("call_3", "list_files", &json!({"path": "."}));

        rail.before_tool_call(&mut intent_ctx(tool_call.clone()))
            .await
            .unwrap();
        let result = ToolCallResult::failure(tool_call.clone(), "denied");
        rail.after_tool_call(&mut completed_ctx(tool_call, result))
            .await
            .unwrap();

        let entries = payloads(&log).await;
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].kind(), EntryKind::ToolIntent);
        match &entries[1] {
            EntryPayload::ToolCompleted {
                status,
                operation_id,
                ..
            } => {
                assert_eq!(*status, ToolStatus::Unknown);
                assert!(operation_id.is_some());
            }
            other => panic!("expected a tool outcome, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn tool_exception_records_unknown_with_the_reason() {
        let dir = tempfile::TempDir::new().unwrap();
        let log = open_log(&dir).await;
        let rail = ExecutionRecordRail::new(Arc::clone(&log), identity());
        let tool_call = call("call_4", "bash", &json!({"command": "ls"}));

        // An exception for a call the rail never saw is ignored, not invented.
        let mut unseen = HarnessRailContext::for_event(RailEvent::OnToolException);
        unseen.base.tool_call = Some(call("call_9", "bash", &json!({"command": "ls"})));
        rail.on_tool_exception(&mut unseen).await.unwrap();
        assert!(payloads(&log).await.is_empty(), "no entries expected");

        rail.before_tool_call(&mut intent_ctx(tool_call.clone()))
            .await
            .unwrap();
        let mut ctx = HarnessRailContext::for_event(RailEvent::OnToolException);
        ctx.base.tool_call = Some(tool_call);
        rail.on_tool_exception(&mut ctx).await.unwrap();

        let entries = payloads(&log).await;
        assert_eq!(entries.len(), 2);
        match &entries[1] {
            EntryPayload::ToolCompleted {
                status,
                operation_id,
                result_summary,
                reason,
                ..
            } => {
                assert_eq!(*status, ToolStatus::Unknown);
                assert!(operation_id.is_some());
                assert_eq!(result_summary, EXCEPTION_REASON);
                assert_eq!(reason.as_deref(), Some(EXCEPTION_REASON));
            }
            other => panic!("expected a tool outcome, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn repeated_outcomes_are_recorded_once() {
        let dir = tempfile::TempDir::new().unwrap();
        let log = open_log(&dir).await;
        let rail = ExecutionRecordRail::new(Arc::clone(&log), identity());
        let tool_call = call("call_5", "edit_file", &json!({"path": "a.txt"}));

        rail.before_tool_call(&mut intent_ctx(tool_call.clone()))
            .await
            .unwrap();
        for _ in 0..2 {
            let result = ToolCallResult::success(tool_call.clone(), json!({"ok": true}), "done");
            rail.after_tool_call(&mut completed_ctx(tool_call.clone(), result))
                .await
                .unwrap();
        }
        let mut exception = HarnessRailContext::for_event(RailEvent::OnToolException);
        exception.base.tool_call = Some(tool_call);
        rail.on_tool_exception(&mut exception).await.unwrap();

        assert_eq!(payloads(&log).await.len(), 2);
    }

    #[tokio::test]
    async fn repeated_identical_dispatches_record_one_intent() {
        let dir = tempfile::TempDir::new().unwrap();
        let log = open_log(&dir).await;
        let rail = ExecutionRecordRail::new(Arc::clone(&log), identity());
        let tool_call = call("call_7", "write_file", &json!({"path": "c.txt"}));

        rail.before_tool_call(&mut intent_ctx(tool_call.clone()))
            .await
            .unwrap();
        rail.before_tool_call(&mut intent_ctx(tool_call.clone()))
            .await
            .unwrap();
        assert_eq!(payloads(&log).await.len(), 1);

        // New arguments mean a new operation, so a second intent is journalled.
        let changed = call("call_7", "write_file", &json!({"path": "d.txt"}));
        rail.before_tool_call(&mut intent_ctx(changed.clone()))
            .await
            .unwrap();
        let entries = payloads(&log).await;
        assert_eq!(entries.len(), 2);
        let result = ToolCallResult::success(changed, json!({}), "written");
        rail.after_tool_call(&mut completed_ctx(tool_call, result))
            .await
            .unwrap();
        assert_eq!(payloads(&log).await.len(), 3);
    }

    #[tokio::test]
    async fn model_turn_records_the_response_and_known_usage() {
        let dir = tempfile::TempDir::new().unwrap();
        let log = open_log(&dir).await;
        let rail = ExecutionRecordRail::new(Arc::clone(&log), identity());
        let mut ctx = HarnessRailContext::for_event(RailEvent::AfterModelCall);
        ctx.base.iteration = 3;
        ctx.base.response = Some(reply("hello", 7, 5));

        rail.after_model_call(&mut ctx).await.unwrap();

        match &payloads(&log).await[0] {
            EntryPayload::ModelTurn {
                model_call_id,
                response,
                usage,
            } => {
                assert_eq!(model_call_id, &deterministic_model_call_id(&identity(), 3));
                assert_eq!(response, "hello");
                assert_eq!(*usage, ModelUsage::known(7, 5));
            }
            other => panic!("expected a model turn, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn zero_usage_is_recorded_as_unknown() {
        let dir = tempfile::TempDir::new().unwrap();
        let log = open_log(&dir).await;
        let rail = ExecutionRecordRail::new(Arc::clone(&log), identity());
        let mut ctx = HarnessRailContext::for_event(RailEvent::AfterModelCall);
        ctx.base.iteration = 1;
        ctx.base.response = Some(ModelReply {
            message: BaseMessage::assistant("hi"),
            ..ModelReply::default()
        });

        rail.after_model_call(&mut ctx).await.unwrap();

        match &payloads(&log).await[0] {
            EntryPayload::ModelTurn { usage, .. } => {
                assert_eq!(*usage, ModelUsage::unknown("model returned no usage"));
            }
            other => panic!("expected a model turn, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn absent_response_appends_nothing() {
        let dir = tempfile::TempDir::new().unwrap();
        let log = open_log(&dir).await;
        let rail = ExecutionRecordRail::new(Arc::clone(&log), identity());
        let mut ctx = HarnessRailContext::for_event(RailEvent::AfterModelCall);

        rail.after_model_call(&mut ctx).await.unwrap();
        assert!(payloads(&log).await.is_empty(), "no entries expected");
    }

    #[tokio::test]
    async fn outcome_summaries_are_bounded_by_the_policy() {
        let dir = tempfile::TempDir::new().unwrap();
        let log = open_log(&dir).await;
        let policy = ToolRecordPolicy {
            record_read_only: false,
            summarize_limit: 5,
        };
        let rail = ExecutionRecordRail::new(Arc::clone(&log), identity()).with_policy(policy);
        let tool_call = call("call_6", "bash", &json!({"command": "false"}));

        rail.before_tool_call(&mut intent_ctx(tool_call.clone()))
            .await
            .unwrap();
        let result = ToolCallResult::failure(tool_call.clone(), "abcdefghij");
        rail.after_tool_call(&mut completed_ctx(tool_call, result))
            .await
            .unwrap();

        match &payloads(&log).await[1] {
            EntryPayload::ToolCompleted { result_summary, .. } => {
                assert_eq!(result_summary, "abcde…");
            }
            other => panic!("expected a tool outcome, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn calls_without_a_provider_id_still_match_their_outcome() {
        let dir = tempfile::TempDir::new().unwrap();
        let log = open_log(&dir).await;
        let rail = ExecutionRecordRail::new(Arc::clone(&log), identity());
        let tool_call = ToolCall {
            id: None,
            name: "write_file".to_owned(),
            arguments: json!({"path": "b.txt"}).to_string(),
            ..ToolCall::default()
        };

        rail.before_tool_call(&mut intent_ctx(tool_call.clone()))
            .await
            .unwrap();
        let result = ToolCallResult::success(tool_call.clone(), json!({}), "written");
        rail.after_tool_call(&mut completed_ctx(tool_call, result))
            .await
            .unwrap();

        let entries = payloads(&log).await;
        assert_eq!(entries.len(), 2);
        match (&entries[0], &entries[1]) {
            (
                EntryPayload::ToolIntent { tool_call_id, .. },
                EntryPayload::ToolCompleted {
                    tool_call_id: done_id,
                    ..
                },
            ) => {
                assert_eq!(tool_call_id, "write_file:0");
                assert_eq!(done_id, tool_call_id);
            }
            other => panic!("expected an intent and its outcome, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn rail_names_and_priorities_match_the_contract() {
        let dir = tempfile::TempDir::new().unwrap();
        let log = open_log(&dir).await;
        let record = ExecutionRecordRail::new(Arc::clone(&log), identity());
        assert_eq!(record.name(), "persistent_execution_record");
        assert_eq!(record.priority(), 5);
        let gate = Arc::new(FakeGate::admitting());
        let model = ModelCallRail::new(gate, log, identity());
        assert_eq!(model.name(), "persistent_model_call");
        assert_eq!(model.priority(), 10);
    }

    #[test]
    fn deterministic_model_call_ids_are_stable_and_iteration_specific() {
        let first = deterministic_model_call_id(&identity(), 1);
        assert_eq!(first, deterministic_model_call_id(&identity(), 1));
        assert_ne!(first, deterministic_model_call_id(&identity(), 2));
        assert!(first.starts_with("mc_"));
        assert_eq!(first.len(), 3 + 32);
        assert!(first[3..].chars().all(|c| c.is_ascii_hexdigit()));
        assert_ne!(first, deterministic_model_call_id(&other_identity(), 1));
    }

    #[test]
    fn read_only_tool_list_matches_the_python_names() {
        assert_eq!(
            READ_ONLY_TOOLS,
            [
                "read_file",
                "list_files",
                "share_file",
                "read_execution_record",
                "search_group_messages",
                "search_session_history",
                "new_context",
            ]
            .as_slice()
        );
        assert!(READ_ONLY_TOOLS.iter().all(|name| is_read_only_tool(name)));
        for name in [
            "bash",
            "write_file",
            "edit_file",
            "todo_create",
            "memory_search",
        ] {
            assert!(!is_read_only_tool(name), "{name} is not read-only");
        }
    }

    #[test]
    fn policy_defaults_and_usage_mapping() {
        let policy = ToolRecordPolicy::new();
        assert!(!policy.record_read_only);
        assert_eq!(policy.summarize_limit, super::SUMMARY_LIMIT);
        assert_eq!(ToolRecordPolicy::default(), policy);
        let decoded: ToolRecordPolicy = serde_json::from_str("{}").unwrap();
        assert_eq!(decoded, policy);
        assert_eq!(
            model_usage(&Usage::new(0, 3)),
            ModelUsage::known(0, 3),
            "a single non-zero counter is still known"
        );
        assert_eq!(
            model_usage(&Usage::new(0, 0)),
            ModelUsage::unknown("model returned no usage")
        );
    }

    /// A gate that records what the rail asks of it.
    struct FakeGate {
        /// Answer returned by `admit`.
        admission: ModelAdmission,
        /// Error `report` returns; `None` accepts the usage.
        report_error: Option<String>,
        /// Call ids handed to `admit`.
        admitted: parking_lot::Mutex<Vec<String>>,
        /// Usage handed to `report`.
        reported: parking_lot::Mutex<Vec<(String, ModelUsage)>>,
    }

    impl FakeGate {
        /// A gate that admits every call and accepts every receipt.
        fn admitting() -> Self {
            Self {
                admission: ModelAdmission::Admitted {
                    reservation_ref: Some("res_1".to_owned()),
                },
                report_error: None,
                admitted: parking_lot::Mutex::new(Vec::new()),
                reported: parking_lot::Mutex::new(Vec::new()),
            }
        }

        /// A gate that denies every call with `reason`.
        fn denying(reason: &str) -> Self {
            Self {
                admission: ModelAdmission::Denied {
                    reason: reason.to_owned(),
                },
                ..Self::admitting()
            }
        }

        /// A gate whose `report` fails with `error`.
        fn failing_report(error: &str) -> Self {
            Self {
                report_error: Some(error.to_owned()),
                ..Self::admitting()
            }
        }
    }

    #[async_trait]
    impl ModelCallGate for FakeGate {
        async fn admit(
            &self,
            _identity: &ExecutionIdentity,
            call_id: &str,
        ) -> Result<ModelAdmission> {
            self.admitted.lock().push(call_id.to_owned());
            Ok(self.admission.clone())
        }

        async fn report(
            &self,
            _identity: &ExecutionIdentity,
            call_id: &str,
            usage: &ModelUsage,
        ) -> Result<()> {
            self.reported
                .lock()
                .push((call_id.to_owned(), usage.clone()));
            match &self.report_error {
                Some(message) => Err(Error::config(message.clone())),
                None => Ok(()),
            }
        }
    }

    #[tokio::test]
    async fn admitted_model_call_stores_the_id_and_journals_the_turn() {
        let dir = tempfile::TempDir::new().unwrap();
        let log = open_log(&dir).await;
        let gate = Arc::new(FakeGate::admitting());
        let rail = ModelCallRail::new(gate.clone(), Arc::clone(&log), identity());
        let expected = deterministic_model_call_id(&identity(), 2);

        let mut before = HarnessRailContext::for_event(RailEvent::BeforeModelCall);
        before.base.iteration = 2;
        rail.before_model_call(&mut before).await.unwrap();
        assert_eq!(
            before
                .base
                .extra
                .get(MODEL_CALL_ID_EXTRA_KEY)
                .and_then(Value::as_str),
            Some(expected.as_str())
        );
        assert_eq!(*gate.admitted.lock(), vec![expected.clone()]);

        let mut after = HarnessRailContext::for_event(RailEvent::AfterModelCall);
        after.base.iteration = 2;
        after.base.extra = before.base.extra.clone();
        after.base.response = Some(reply("done", 11, 3));
        rail.after_model_call(&mut after).await.unwrap();

        match &payloads(&log).await[0] {
            EntryPayload::ModelTurn {
                model_call_id,
                response,
                usage,
            } => {
                assert_eq!(model_call_id, &expected);
                assert_eq!(response, "done");
                assert_eq!(*usage, ModelUsage::known(11, 3));
            }
            other => panic!("expected a model turn, got {other:?}"),
        }
        let reported = gate.reported.lock();
        assert_eq!(reported.len(), 1);
        assert_eq!(reported[0].0, expected);
        assert_eq!(reported[0].1, ModelUsage::known(11, 3));
    }

    #[tokio::test]
    async fn denied_model_call_returns_the_config_error() {
        let dir = tempfile::TempDir::new().unwrap();
        let log = open_log(&dir).await;
        let gate = Arc::new(FakeGate::denying("budget exhausted"));
        let rail = ModelCallRail::new(gate, Arc::clone(&log), identity());
        let mut ctx = HarnessRailContext::for_event(RailEvent::BeforeModelCall);

        let error = rail.before_model_call(&mut ctx).await.unwrap_err();
        assert!(
            matches!(
                &error,
                Error::Config(message)
                    if message == "persistent model call denied: budget exhausted"
            ),
            "unexpected error: {error:?}"
        );
        assert!(!ctx.base.extra.contains_key(MODEL_CALL_ID_EXTRA_KEY));
        assert!(payloads(&log).await.is_empty(), "no entries expected");
    }

    #[tokio::test]
    async fn report_failure_does_not_fail_the_hook() {
        let dir = tempfile::TempDir::new().unwrap();
        let log = open_log(&dir).await;
        let gate = Arc::new(FakeGate::failing_report("ledger offline"));
        let rail = ModelCallRail::new(gate.clone(), Arc::clone(&log), identity());
        let mut ctx = HarnessRailContext::for_event(RailEvent::AfterModelCall);
        ctx.base.iteration = 1;
        ctx.base.response = Some(reply("hi", 2, 1));

        rail.after_model_call(&mut ctx).await.unwrap();

        let expected = deterministic_model_call_id(&identity(), 1);
        match &payloads(&log).await[0] {
            EntryPayload::ModelTurn { model_call_id, .. } => assert_eq!(model_call_id, &expected),
            other => panic!("expected a model turn, got {other:?}"),
        }
        assert_eq!(gate.reported.lock().len(), 1);
    }

    #[tokio::test]
    async fn model_exception_reports_unknown_usage_and_notes_the_failure() {
        let dir = tempfile::TempDir::new().unwrap();
        let log = open_log(&dir).await;
        let gate = Arc::new(FakeGate::admitting());
        let rail = ModelCallRail::new(gate.clone(), Arc::clone(&log), identity());
        let mut ctx = HarnessRailContext::for_event(RailEvent::OnModelException);
        ctx.base.iteration = 4;

        rail.on_model_exception(&mut ctx).await.unwrap();

        let expected = deterministic_model_call_id(&identity(), 4);
        {
            let reported = gate.reported.lock();
            assert_eq!(reported.len(), 1);
            assert_eq!(reported[0].0, expected);
            assert_eq!(reported[0].1, ModelUsage::unknown("model call failed"));
        }
        match &payloads(&log).await[0] {
            EntryPayload::Note { text } => assert_eq!(text, "model call failed"),
            other => panic!("expected a note, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn model_exception_tolerates_a_failing_gate() {
        let dir = tempfile::TempDir::new().unwrap();
        let log = open_log(&dir).await;
        let gate = Arc::new(FakeGate::failing_report("ledger offline"));
        let rail = ModelCallRail::new(gate, Arc::clone(&log), identity());
        let mut ctx = HarnessRailContext::for_event(RailEvent::OnModelException);

        rail.on_model_exception(&mut ctx).await.unwrap();
        assert_eq!(payloads(&log).await.len(), 1);
    }

    #[tokio::test]
    async fn model_turn_reuses_the_stored_call_id_and_rebuilds_it_otherwise() {
        let dir = tempfile::TempDir::new().unwrap();
        let log = open_log(&dir).await;
        let gate = Arc::new(FakeGate::admitting());
        let rail = ModelCallRail::new(gate, Arc::clone(&log), identity());

        let mut stored = HarnessRailContext::for_event(RailEvent::AfterModelCall);
        stored.base.iteration = 7;
        stored.base.extra.insert(
            MODEL_CALL_ID_EXTRA_KEY.to_string(),
            Value::String("mc_stored".to_owned()),
        );
        stored.base.response = Some(reply("a", 1, 1));
        rail.after_model_call(&mut stored).await.unwrap();

        let mut rebuilt = HarnessRailContext::for_event(RailEvent::AfterModelCall);
        rebuilt.base.iteration = 7;
        rebuilt.base.response = Some(reply("b", 1, 1));
        rail.after_model_call(&mut rebuilt).await.unwrap();

        let expected_rebuilt = deterministic_model_call_id(&identity(), 7);
        let entries = payloads(&log).await;
        assert_eq!(entries.len(), 2);
        let ids: Vec<&str> = entries
            .iter()
            .map(|entry| match entry {
                EntryPayload::ModelTurn { model_call_id, .. } => model_call_id.as_str(),
                other => panic!("expected a model turn, got {other:?}"),
            })
            .collect();
        assert_eq!(ids, ["mc_stored", expected_rebuilt.as_str()]);
    }
}
