//! Human-in-the-loop permission rail: the `ask` of the permission engine.
//!
//! Ports `openjiuwen/harness/rails/security/tool_security_rail.py`'s
//! `PermissionInterruptRail` (v0.1.20) onto the kernel rail seam
//! ([`openjiuwen_agent::rails::AgentRail`]): the rail runs on
//! [`before_tool_call`](AgentRail::before_tool_call), evaluates the tool call
//! through [`PermissionEngine`], and turns the outcome into one of three actions:
//!
//! - **allow** — the hook returns `Ok(())` and the tool runs;
//! - **deny** — [`RailContext::skip_tool_with`] replaces the tool result with a
//!   `[PERMISSION_DENIED] …` message;
//! - **ask** — a [`InterruptRequest`] of kind [`InterruptKind::Permission`] is
//!   recorded on the context and the run pauses. The resume round re-enters the
//!   hook with the user answer in `ctx.extra` under [`RESUME_ANSWERS_KEY`].
//!
//! The hook never returns `Err` for a permission outcome: denials and rejections
//! are tool results, not rail failures, so a rejected call still produces a tool
//! message the model can react to.
//!
//! Answers are [`PermissionConfirmResponse`] payloads (`approved`, `feedback`,
//! `auto_confirm`, `persist_allow`). A "remember" approval is merged into the
//! policy by [`session::merge_allow_rule_into_permissions`], written out through a
//! [`PermissionHost`] hook, and kept in [`SessionPermissionStore`] under the
//! call's conservative session key. Scene hooks, hosted confirmations and
//! permission snapshots also come from the host; the default [`NoopPermissionHost`]
//! asks the built-in interrupt and persists nothing.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_agent::interrupt::{InterruptKind, InterruptRequest};
use openjiuwen_agent::rails::{AgentRail, RailContext};
use openjiuwen_common::error::Result;
use openjiuwen_foundation::ToolCall;
use serde_json::{json, Value};

use crate::permission::engine::PermissionEngine;
use crate::permission::model::{PermissionConfirmResponse, PermissionLevel, PermissionResult};
use crate::permission::presentation::{self, AskCategory};
use crate::permission::session::{
    self, ConfirmationOutcome, NoopPermissionHost, PermissionConfirmationRequest, PermissionHost,
    SceneDecision, SceneHookInput, SessionPermissionStore,
};
use crate::permission::{rules, shell};

/// Python owner string stamped on every interrupt this rail raises.
pub const RAIL_OWNER: &str =
    "openjiuwen.harness.rails.security.tool_security_rail.PermissionInterruptRail";

/// Rails priority (Python `BaseInterruptRail.priority`).
pub const RAIL_PRIORITY: i32 = 90;

/// `ctx.extra` key holding the resume answers: `{ "<tool_call_id>": <answer> }`.
pub const RESUME_ANSWERS_KEY: &str = "_resume_interrupt_answers";

/// `ctx.extra` key holding host auto-confirm decisions (bool or `{ "<key>": true }`).
pub const AUTO_CONFIRM_KEY: &str = "_interrupt_auto_confirm";

/// `ctx.extra` key recording tool-call ids whose "remember" merge already ran.
pub const PERSISTED_APPROVALS_KEY: &str = "_permission_persisted_approvals";

/// Prefix of every tool result the rail substitutes for a denied call.
pub const DENIED_PREFIX: &str = "[PERMISSION_DENIED]";

/// Tool result of a rejected confirmation without user feedback.
pub const REJECTED_MESSAGE: &str = "[PERMISSION_REJECTED] User rejected the request.";

/// Reason used for a rejection when the engine supplied none.
const FALLBACK_DENY_REASON: &str = "Operation not allowed";

/// Reason used when a hosted confirmation failed or answered invalidly.
const FALLBACK_APPROVAL_REASON: &str = "Operation requires approval";

/// `ask_category` metadata value of a risk-finding presentation.
const CATEGORY_FINDING: &str = "finding";

/// Turns the permission engine's `ask` into a confirmation interrupt.
///
/// Python parity: `PermissionInterruptRail`. The rail is consulted for **every**
/// tool call (Python stopped filtering by tool name), and the engine decides what
/// happens. All three collaborators are shared handles: the engine is hot-updated
/// when an approval is remembered, the host supplies product hooks, and the store
/// keeps the session's auto-confirmed keys.
pub struct PermissionInterruptRail {
    /// Rule pipeline the rail evaluates against.
    engine: Arc<PermissionEngine>,
    /// Product hooks (scene hook, hosted confirmation, persistence, snapshot).
    host: Arc<dyn PermissionHost>,
    /// Session-scoped auto-confirm keys.
    store: Arc<SessionPermissionStore>,
}

impl PermissionInterruptRail {
    /// Builds the rail over `engine` with a no-op host and an empty session store.
    #[must_use]
    pub fn new(engine: Arc<PermissionEngine>) -> Self {
        Self {
            engine,
            host: Arc::new(NoopPermissionHost),
            store: Arc::new(SessionPermissionStore::new()),
        }
    }

    /// Replaces the product hooks.
    #[must_use]
    pub fn with_host(mut self, host: Arc<dyn PermissionHost>) -> Self {
        self.host = host;
        self
    }

    /// Replaces the session-scoped auto-confirm store.
    #[must_use]
    pub fn with_store(mut self, store: Arc<SessionPermissionStore>) -> Self {
        self.store = store;
        self
    }

    /// The shared permission engine.
    #[must_use]
    pub fn engine(&self) -> &Arc<PermissionEngine> {
        &self.engine
    }

    /// The shared session permission store.
    #[must_use]
    pub fn store(&self) -> &Arc<SessionPermissionStore> {
        &self.store
    }

    /// Python `_get_auto_confirm_key`: the conservative session key of a call.
    ///
    /// A shell tool gets [`shell::shell_auto_confirm_key`] (`<tool>:<command>`, empty
    /// for a compound or risky command, so one approval never opens every shell
    /// call); every other tool uses its raw name.
    #[must_use]
    pub fn auto_confirm_key(&self, call: &ToolCall) -> String {
        let raw = call.name.as_str();
        let shell_tools = rules::shell_tools_from_config(&self.engine.config());
        if !rules::is_shell_tool(&rules::normalize_tool_name(raw), &shell_tools) {
            return raw.to_owned();
        }
        let args = parse_tool_args(call);
        shell::shell_auto_confirm_key(raw, &rules::command_text(&args))
    }

    /// Python static `parse_confirm_payload`, delegated to
    /// [`PermissionConfirmResponse::parse`].
    #[must_use]
    pub fn parse_confirm_payload(value: &Value) -> Option<PermissionConfirmResponse> {
        PermissionConfirmResponse::parse(value)
    }

    /// Python `_check_with_fresh_config`: re-sync the host snapshot, then evaluate.
    /// A snapshot (disk or session overlay) replaces the policy first, so a rule
    /// written out by an earlier approval wins over the cached one.
    fn fresh_check(&self, tool: &str, args: &Value) -> PermissionResult {
        if let Some(section) = self.host.permissions_snapshot() {
            self.engine.update_permissions(section);
        }
        self.engine.check_permission(tool, args)
    }

    /// Re-evaluates the call and decides a first (unanswered) round.
    ///
    /// `None` means the hook is done: the call was allowed, denied, or
    /// auto-confirmed. `Some(result)` is the `ask` that still needs an answer.
    fn first_round_ask(
        &self,
        ctx: &mut RailContext,
        call: &ToolCall,
        tool: &str,
        args: &Value,
        auto_confirm_key: &str,
    ) -> Option<PermissionResult> {
        let result = self.fresh_check(tool, args);
        if result.is_allowed() {
            tracing::debug!(tool = %tool, rule = ?result.matched_rule, "permission.rail.result decision=allow");
            return None;
        }
        if result.is_denied() {
            tracing::warn!(tool = %tool, rule = ?result.matched_rule, "permission.rail.result decision=deny");
            let message = deny_text(&result);
            ctx.skip_tool_with(message);
            return None;
        }
        if self.is_auto_confirmed(ctx, auto_confirm_key, call) {
            tracing::debug!(tool = %tool, key = %auto_confirm_key, "permission.auto_confirm.hit");
            return None;
        }
        Some(result)
    }

    /// Python `_is_auto_confirmed`: the call was already approved for this session.
    /// True when the store holds the key, or the context flags it globally (`true`)
    /// or for this key / raw tool name under [`AUTO_CONFIRM_KEY`].
    fn is_auto_confirmed(
        &self,
        ctx: &RailContext,
        auto_confirm_key: &str,
        call: &ToolCall,
    ) -> bool {
        if !auto_confirm_key.is_empty() && self.store.is_auto_confirmed(auto_confirm_key) {
            return true;
        }
        let Some(value) = ctx.extra.get(AUTO_CONFIRM_KEY) else {
            return false;
        };
        match value {
            Value::Bool(flag) => *flag,
            Value::Object(map) => [auto_confirm_key, call.name.as_str()]
                .iter()
                .filter(|key| !key.is_empty())
                .any(|key| map.get(*key).and_then(Value::as_bool) == Some(true)),
            _ => false,
        }
    }

    /// Records a fresh confirmation interrupt for `call`.
    ///
    /// Python `interrupt(InterruptRequest(...))` + `_build_message` +
    /// `_build_interrupt_metadata`: the message is the rendered presentation with
    /// its "remember this" hint, and the metadata carries `ask_category`,
    /// `ask_title`, `ask_summary` and `matched_rule`.
    fn raise_interrupt(
        ctx: &mut RailContext,
        call: &ToolCall,
        args: &Value,
        result: &PermissionResult,
        auto_confirm_key: &str,
    ) {
        let raw = call.name.as_str();
        let ask = presentation::build_ask_presentation(result, raw, args);
        let hint = presentation::always_allow_hint(raw, auto_confirm_key, None);
        let message = presentation::render_ask_message(&ask, Some(&hint));
        // Python leaves the request's auto-confirm key empty: the rail stores the
        // session key itself, the kernel must not store this request's key.
        let mut request =
            InterruptRequest::for_tool_call(InterruptKind::Permission, RAIL_OWNER, message, call)
                .with_payload_schema(PermissionConfirmResponse::payload_schema())
                .with_auto_confirm_key("");
        request
            .metadata
            .insert("ask_category".to_owned(), json!(category_str(ask.category)));
        request
            .metadata
            .insert("ask_title".to_owned(), json!(ask.title));
        request
            .metadata
            .insert("ask_summary".to_owned(), json!(ask.summary));
        request.metadata.insert(
            "matched_rule".to_owned(),
            json!(result.matched_rule.clone().unwrap_or_default()),
        );
        tracing::info!(
            tool = %raw,
            rule = ?result.matched_rule,
            category = category_str(ask.category),
            "permission.interrupt.ask"
        );
        ctx.request_interrupt(request);
    }

    /// Applies an answered confirmation (resume round, or a hosted answer).
    ///
    /// Python `resolve_interrupt`'s user-response branch: a rejection becomes the
    /// tool result (`feedback`, or [`REJECTED_MESSAGE`]); an approval re-checks the
    /// call first (a deny rule added while it waited wins); and a "remember" answer
    /// merges its allow rule, writes it through the host and stores the session key
    /// exactly once per paused call.
    async fn apply_confirmed(
        &self,
        ctx: &mut RailContext,
        call: &ToolCall,
        tool: &str,
        args: &Value,
        payload: PermissionConfirmResponse,
        auto_confirm_key: &str,
    ) {
        let raw = call.name.as_str();
        if !payload.approved {
            tracing::info!(tool = %tool, "permission.user.decision decision=deny");
            let message = if payload.feedback.is_empty() {
                REJECTED_MESSAGE.to_owned()
            } else {
                payload.feedback
            };
            ctx.skip_tool_with(message);
            return;
        }

        let current = self.fresh_check(tool, args);
        if current.is_denied() {
            tracing::warn!(
                tool = %tool,
                rule = ?current.matched_rule,
                "permission.rail.result decision=deny_after_approval"
            );
            let message = deny_text(&current);
            ctx.skip_tool_with(message);
            return;
        }

        let tool_call_id = call.id.clone().unwrap_or_default();
        if has_persisted_approval(&ctx.extra, &tool_call_id) {
            // Approved in an earlier resume round of this still-paused call; its
            // rules and auto-confirm were stored then.
            tracing::debug!(
                tool = %tool,
                tool_call_id = %tool_call_id,
                "permission.user.decision confirm_path=carried decision=allow"
            );
            return;
        }

        let permanent = payload.wants_permanent_persist();
        let session_scoped = payload.wants_session_persist();
        if permanent || session_scoped {
            let merged =
                session::merge_allow_rule_into_permissions(&self.engine.config(), tool, args);
            let persisted = if merged.applied() {
                let persisted = if permanent {
                    self.host.persist_allow_rule(&merged.permissions).await
                } else {
                    self.host
                        .persist_session_allow_rule(&merged.permissions)
                        .await
                };
                self.engine.update_permissions(merged.permissions);
                record_persisted_approval(ctx, &tool_call_id);
                persisted
            } else {
                false
            };
            tracing::info!(
                tool = %tool,
                permanent,
                session_scoped,
                persisted,
                "permission.persist.result"
            );
            if session::should_store_auto_confirm(
                true,
                payload.auto_confirm,
                true,
                auto_confirm_key,
                persisted,
                permanent,
            ) {
                self.store.store_auto_confirm(auto_confirm_key);
                mirror_auto_confirm(ctx, raw, auto_confirm_key);
                tracing::info!(key = %auto_confirm_key, "permission.auto_confirm.store");
            }
        }
        tracing::info!(tool = %tool, "permission.user.decision decision=allow");
    }
}

#[allow(clippy::missing_fields_in_debug)]
impl fmt::Debug for PermissionInterruptRail {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PermissionInterruptRail")
            .field("engine_enabled", &self.engine.enabled())
            .field("workspace_root", &self.engine.workspace_root())
            .field("host", &"<dyn PermissionHost>")
            .field("auto_confirm_keys", &self.store.len())
            .finish()
    }
}

#[async_trait]
impl AgentRail for PermissionInterruptRail {
    fn name(&self) -> &'static str {
        "permission_interrupt"
    }

    fn priority(&self) -> i32 {
        RAIL_PRIORITY
    }

    /// Allows, rejects, or pauses the tool call on the context.
    ///
    /// The hook is idempotent with respect to the permission outcome: the same
    /// context can be fired twice (a resume round carried over), and the "remember"
    /// merge runs only once. It never returns `Err` and never panics.
    async fn before_tool_call(&self, ctx: &mut RailContext) -> Result<()> {
        let Some(call) = ctx.tool_call.clone() else {
            return Ok(());
        };
        let raw = call.name.clone();
        let tool = rules::normalize_tool_name(&raw);
        let args = parse_tool_args(&call);
        let tool_call_id = call.id.clone().unwrap_or_default();
        let auto_confirm_key = self.auto_confirm_key(&call);
        let answer = resume_answer(&ctx.extra, &tool_call_id);
        let answer_carried = has_persisted_approval(&ctx.extra, &tool_call_id);
        tracing::debug!(
            tool = %raw,
            normalized = %tool,
            auto_confirm_key = %auto_confirm_key,
            answer = answer.is_some(),
            "permission.rail.before_tool_call"
        );

        // Python runs the scene hook before the policy on both rounds, so a product
        // layer can pre-empt the engine entirely.
        let scene = self
            .host
            .scene_hook(&SceneHookInput {
                tool_name: tool.clone(),
                tool_call_id: tool_call_id.clone(),
                arguments: args.clone(),
                answer_carried,
            })
            .await;
        match scene {
            Some(SceneDecision::Approve) => {
                tracing::info!(tool = %raw, "permission.scene_hook decision=approve");
                return Ok(());
            }
            Some(SceneDecision::Reject(message)) => {
                tracing::info!(tool = %raw, "permission.scene_hook decision=reject");
                ctx.skip_tool_with(message.unwrap_or_else(|| DENIED_PREFIX.to_owned()));
                return Ok(());
            }
            None => {}
        }

        if let Some(value) = answer {
            let Some(payload) = Self::parse_confirm_payload(&value) else {
                // An answer we cannot interpret re-asks; it never rejects.
                let invalid = PermissionResult::new(PermissionLevel::Ask)
                    .with_reason("Invalid confirmation payload");
                Self::raise_interrupt(ctx, &call, &args, &invalid, &auto_confirm_key);
                return Ok(());
            };
            self.apply_confirmed(ctx, &call, &tool, &args, payload, &auto_confirm_key)
                .await;
            return Ok(());
        }

        let Some(result) = self.first_round_ask(ctx, &call, &tool, &args, &auto_confirm_key) else {
            return Ok(());
        };

        let request = PermissionConfirmationRequest {
            tool_name: tool.clone(),
            tool_call_id: tool_call_id.clone(),
            arguments: args.clone(),
            result: result.clone(),
            auto_confirm_key: auto_confirm_key.clone(),
        };
        match self.host.request_confirmation(&request).await {
            ConfirmationOutcome::Response(payload) => {
                self.apply_confirmed(ctx, &call, &tool, &args, payload, &auto_confirm_key)
                    .await;
            }
            ConfirmationOutcome::Failed => {
                let message = hosted_failure(&result, "(Hosted permission request failed)");
                ctx.skip_tool_with(message);
            }
            ConfirmationOutcome::Invalid => {
                let message = hosted_failure(&result, "(Invalid hosted permission response)");
                ctx.skip_tool_with(message);
            }
            ConfirmationOutcome::Interrupt => {
                Self::raise_interrupt(ctx, &call, &args, &result, &auto_confirm_key);
            }
        }
        Ok(())
    }
}

/// Python `parse_tool_args`: malformed text, a scalar or an array all degrade to
/// `{}`, so the engine sees a call without arguments rather than an error.
fn parse_tool_args(call: &ToolCall) -> Value {
    match serde_json::from_str::<Value>(call.arguments.trim()) {
        Ok(Value::Object(map)) => Value::Object(map),
        _ => Value::Object(serde_json::Map::new()),
    }
}

/// The resume answer seeded for `tool_call_id`, when the round carries one.
fn resume_answer(extra: &BTreeMap<String, Value>, tool_call_id: &str) -> Option<Value> {
    if tool_call_id.is_empty() {
        return None;
    }
    extra
        .get(RESUME_ANSWERS_KEY)?
        .as_object()?
        .get(tool_call_id)
        .cloned()
}

/// Whether the "remember" merge of `tool_call_id` already ran.
fn has_persisted_approval(extra: &BTreeMap<String, Value>, tool_call_id: &str) -> bool {
    if tool_call_id.is_empty() {
        return false;
    }
    extra
        .get(PERSISTED_APPROVALS_KEY)
        .and_then(Value::as_array)
        .is_some_and(|ids| ids.iter().any(|id| id.as_str() == Some(tool_call_id)))
}

/// Records `tool_call_id` as merged, deduplicated.
fn record_persisted_approval(ctx: &mut RailContext, tool_call_id: &str) {
    if tool_call_id.is_empty() {
        return;
    }
    if !ctx
        .extra
        .get(PERSISTED_APPROVALS_KEY)
        .is_some_and(Value::is_array)
    {
        ctx.extra
            .insert(PERSISTED_APPROVALS_KEY.to_owned(), json!([]));
    }
    let Some(Value::Array(ids)) = ctx.extra.get_mut(PERSISTED_APPROVALS_KEY) else {
        return;
    };
    if ids.iter().any(|id| id.as_str() == Some(tool_call_id)) {
        return;
    }
    ids.push(Value::String(tool_call_id.to_owned()));
}

/// Mirrors a stored session key into `ctx.extra[AUTO_CONFIRM_KEY]`, keyed by the
/// session key and the raw tool name, so a later round sharing this context sees
/// the decision. A context already carrying a global `true` needs no entry.
fn mirror_auto_confirm(ctx: &mut RailContext, raw_tool_name: &str, auto_confirm_key: &str) {
    let entry = ctx
        .extra
        .entry(AUTO_CONFIRM_KEY.to_owned())
        .or_insert_with(|| json!({}));
    if !entry.is_object() {
        if entry.as_bool() == Some(true) {
            return;
        }
        *entry = json!({});
    }
    let Some(map) = entry.as_object_mut() else {
        return;
    };
    for key in [auto_confirm_key, raw_tool_name] {
        if !key.is_empty() {
            map.insert(key.to_owned(), Value::Bool(true));
        }
    }
}

/// The `[PERMISSION_DENIED] …` tool result of a denied call.
fn deny_text(result: &PermissionResult) -> String {
    format!(
        "{DENIED_PREFIX} {}",
        reason_text(result, FALLBACK_DENY_REASON)
    )
}

/// The `[PERMISSION_DENIED] … (Hosted …)` tool result of a failed host answer.
fn hosted_failure(result: &PermissionResult, suffix: &str) -> String {
    format!(
        "{DENIED_PREFIX} {} {suffix}",
        reason_text(result, FALLBACK_APPROVAL_REASON)
    )
}

/// The trimmed engine reason of `result`, or `fallback` when it has none.
fn reason_text(result: &PermissionResult, fallback: &str) -> String {
    result
        .reason
        .as_deref()
        .map(str::trim)
        .filter(|reason| !reason.is_empty())
        .map_or_else(|| fallback.to_owned(), str::to_owned)
}

/// The wire value of a presentation category (`ask_category` metadata).
fn category_str(category: AskCategory) -> &'static str {
    match category {
        AskCategory::Shell => "shell",
        AskCategory::Path => "path",
        AskCategory::Network => "network",
        AskCategory::Tool => "tool",
        AskCategory::Finding => CATEGORY_FINDING,
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::collections::BTreeMap;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use async_trait::async_trait;
    use openjiuwen_agent::rails::RailEvent;
    use parking_lot::Mutex;

    use super::*;
    use crate::permission::model::PermissionsSection;

    /// Host that records what the rail asks of it.
    #[derive(Default)]
    struct RecordingHost {
        snapshot: Mutex<Option<PermissionsSection>>,
        scene: Mutex<Option<SceneDecision>>,
        confirmation: Mutex<Option<ConfirmationOutcome>>,
        confirm_calls: AtomicUsize,
        persist_allow_calls: AtomicUsize,
        persist_session_calls: AtomicUsize,
        persist_result: bool,
    }

    impl RecordingHost {
        fn new(persist_result: bool) -> Self {
            Self {
                persist_result,
                ..Self::default()
            }
        }
    }

    #[async_trait]
    impl PermissionHost for RecordingHost {
        fn permissions_snapshot(&self) -> Option<PermissionsSection> {
            self.snapshot.lock().clone()
        }

        async fn scene_hook(&self, _input: &SceneHookInput) -> Option<SceneDecision> {
            self.scene.lock().clone()
        }

        async fn request_confirmation(
            &self,
            _request: &PermissionConfirmationRequest,
        ) -> ConfirmationOutcome {
            self.confirm_calls.fetch_add(1, Ordering::SeqCst);
            self.confirmation
                .lock()
                .clone()
                .unwrap_or(ConfirmationOutcome::Interrupt)
        }

        async fn persist_allow_rule(&self, _permissions: &PermissionsSection) -> bool {
            self.persist_allow_calls.fetch_add(1, Ordering::SeqCst);
            self.persist_result
        }

        async fn persist_session_allow_rule(&self, _permissions: &PermissionsSection) -> bool {
            self.persist_session_calls.fetch_add(1, Ordering::SeqCst);
            self.persist_result
        }
    }

    /// A section with `packaged` rules off, the given baselines and `defaults["*"]`.
    fn section(tools: &[(&str, PermissionLevel)], default: PermissionLevel) -> PermissionsSection {
        PermissionsSection {
            package_builtin_rules: false,
            defaults: BTreeMap::from([("*".to_owned(), default)]),
            tools: tools
                .iter()
                .map(|(name, level)| ((*name).to_owned(), *level))
                .collect(),
            ..PermissionsSection::default()
        }
    }

    fn engine_for(section: PermissionsSection) -> Arc<PermissionEngine> {
        Arc::new(PermissionEngine::new(
            crate::permission::engine::PermissionEngineConfig::new(section),
        ))
    }

    fn ask_engine() -> Arc<PermissionEngine> {
        engine_for(section(&[], PermissionLevel::Ask))
    }

    fn bash(command: &str) -> ToolCall {
        ToolCall {
            id: Some("c1".to_owned()),
            name: "bash".to_owned(),
            arguments: format!(r#"{{"command":"{command}"}}"#),
            ..ToolCall::default()
        }
    }

    fn ctx_for(call: ToolCall) -> RailContext {
        let mut ctx = RailContext::new(RailEvent::BeforeToolCall);
        ctx.tool_call = Some(call);
        ctx
    }

    fn resume(ctx: &mut RailContext, value: &Value) {
        ctx.extra
            .insert(RESUME_ANSWERS_KEY.to_owned(), json!({"c1": value}));
    }

    async fn run(rail: &PermissionInterruptRail, ctx: &mut RailContext) {
        rail.before_tool_call(ctx).await.unwrap();
    }

    #[tokio::test]
    async fn allow_lets_the_tool_run() {
        let engine = engine_for(section(
            &[("bash", PermissionLevel::Allow)],
            PermissionLevel::Allow,
        ));
        let rail = PermissionInterruptRail::new(engine);
        let mut ctx = ctx_for(bash("git status"));
        run(&rail, &mut ctx).await;
        assert_eq!(ctx.interrupts.len(), 0);
        assert!(ctx.take_skip_tool().is_none());
    }

    #[tokio::test]
    async fn deny_skips_with_the_engine_reason() {
        let engine = engine_for(section(
            &[("bash", PermissionLevel::Deny)],
            PermissionLevel::Allow,
        ));
        let rail = PermissionInterruptRail::new(engine);
        let mut ctx = ctx_for(bash("git status"));
        run(&rail, &mut ctx).await;
        assert_eq!(ctx.interrupts.len(), 0);
        let skip = ctx.take_skip_tool().unwrap();
        assert!(skip.starts_with("[PERMISSION_DENIED] "), "{skip}");
        assert!(skip.contains("Denied by rule: tools.bash"), "{skip}");
    }

    #[tokio::test]
    async fn ask_raises_one_permission_interrupt() {
        let rail = PermissionInterruptRail::new(ask_engine());
        let mut ctx = ctx_for(bash("git status"));
        run(&rail, &mut ctx).await;
        assert!(ctx.take_skip_tool().is_none());
        assert_eq!(ctx.interrupts.len(), 1);
        let request = &ctx.interrupts[0];
        assert_eq!(request.kind, InterruptKind::Permission);
        assert_eq!(request.owner, RAIL_OWNER);
        assert_eq!(request.auto_confirm_key, "");
        assert_eq!(request.tool_call_id, "c1");
        for key in ["ask_category", "ask_title", "ask_summary", "matched_rule"] {
            assert!(request.metadata.contains_key(key), "missing {key}");
        }
        assert_eq!(request.metadata["ask_category"], json!("shell"));
        assert!(request.payload_schema["properties"]["approved"].is_object());
        assert!(request.message.contains("git status"));
    }

    #[tokio::test]
    async fn missing_tool_call_is_ignored() {
        let rail = PermissionInterruptRail::new(ask_engine());
        let mut ctx = RailContext::new(RailEvent::BeforeToolCall);
        run(&rail, &mut ctx).await;
        assert_eq!(ctx.interrupts.len(), 0);
        assert!(ctx.take_skip_tool().is_none());
    }

    #[tokio::test]
    async fn resume_reject_uses_the_rejected_message() {
        let rail = PermissionInterruptRail::new(ask_engine());
        let mut ctx = ctx_for(bash("git status"));
        resume(&mut ctx, &json!({"approved": false}));
        run(&rail, &mut ctx).await;
        assert_eq!(ctx.interrupts.len(), 0);
        assert_eq!(ctx.take_skip_tool().as_deref(), Some(REJECTED_MESSAGE));
    }

    #[tokio::test]
    async fn resume_reject_with_feedback_uses_feedback() {
        let rail = PermissionInterruptRail::new(ask_engine());
        let mut ctx = ctx_for(bash("git status"));
        resume(&mut ctx, &json!({"approved": false, "feedback": "not now"}));
        run(&rail, &mut ctx).await;
        assert_eq!(ctx.take_skip_tool().as_deref(), Some("not now"));
    }

    #[tokio::test]
    async fn resume_allow_once_runs_and_merges_nothing() {
        let engine = ask_engine();
        let host = Arc::new(RecordingHost::new(true));
        let rail = PermissionInterruptRail::new(Arc::clone(&engine)).with_host(host.clone());
        let mut ctx = ctx_for(bash("git status"));
        resume(&mut ctx, &json!({"approved": true, "auto_confirm": false}));
        run(&rail, &mut ctx).await;
        assert_eq!(ctx.interrupts.len(), 0);
        assert!(ctx.take_skip_tool().is_none());
        assert_eq!(engine.config().approval_overrides.len(), 0);
        assert_eq!(host.persist_allow_calls.load(Ordering::SeqCst), 0);
        assert_eq!(host.persist_session_calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn session_remember_merges_and_stores_auto_confirm() {
        let engine = ask_engine();
        let host = Arc::new(RecordingHost::new(true));
        let rail = PermissionInterruptRail::new(Arc::clone(&engine)).with_host(host.clone());
        let mut ctx = ctx_for(bash("git status"));
        resume(
            &mut ctx,
            &json!({"approved": true, "auto_confirm": true, "persist_allow": false}),
        );
        run(&rail, &mut ctx).await;
        assert!(ctx.take_skip_tool().is_none());
        assert_eq!(engine.config().approval_overrides.len(), 1);
        assert!(rail.store().is_auto_confirmed("bash:git status"));
        assert_eq!(host.persist_session_calls.load(Ordering::SeqCst), 1);
        assert_eq!(host.persist_allow_calls.load(Ordering::SeqCst), 0);
        assert_eq!(
            ctx.extra[AUTO_CONFIRM_KEY]["bash:git status"],
            json!(true),
            "the decision is mirrored onto the shared context"
        );
    }

    #[tokio::test]
    async fn permanent_remember_persists_and_skips_auto_confirm() {
        let engine = ask_engine();
        let host = Arc::new(RecordingHost::new(true));
        let rail = PermissionInterruptRail::new(Arc::clone(&engine)).with_host(host.clone());
        let mut ctx = ctx_for(bash("git status"));
        resume(
            &mut ctx,
            &json!({"approved": true, "auto_confirm": true, "persist_allow": true}),
        );
        run(&rail, &mut ctx).await;
        assert!(ctx.take_skip_tool().is_none());
        assert_eq!(host.persist_allow_calls.load(Ordering::SeqCst), 1);
        assert_eq!(host.persist_session_calls.load(Ordering::SeqCst), 0);
        assert!(!rail.store().is_auto_confirmed("bash:git status"));
    }

    #[tokio::test]
    async fn persist_once_records_a_single_override() {
        let engine = ask_engine();
        let host = Arc::new(RecordingHost::new(true));
        let rail = PermissionInterruptRail::new(Arc::clone(&engine)).with_host(host.clone());
        let mut ctx = ctx_for(bash("git status"));
        resume(
            &mut ctx,
            &json!({"approved": true, "auto_confirm": true, "persist_allow": false}),
        );
        run(&rail, &mut ctx).await;
        run(&rail, &mut ctx).await;
        assert_eq!(engine.config().approval_overrides.len(), 1);
        assert_eq!(host.persist_session_calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn invalid_payload_reinterrupts() {
        let rail = PermissionInterruptRail::new(ask_engine());
        let mut ctx = ctx_for(bash("git status"));
        resume(&mut ctx, &json!({"unexpected": 1}));
        run(&rail, &mut ctx).await;
        assert!(ctx.take_skip_tool().is_none());
        assert_eq!(ctx.interrupts.len(), 1);
        assert_eq!(ctx.interrupts[0].kind, InterruptKind::Permission);
        assert_eq!(ctx.interrupts[0].metadata["matched_rule"], json!(""));
        assert!(ctx.interrupts[0].message.contains("git status"));
    }

    #[tokio::test]
    async fn deny_rule_added_after_approval_wins() {
        let engine = ask_engine();
        let rail = PermissionInterruptRail::new(Arc::clone(&engine));
        let mut ctx = ctx_for(bash("git status"));
        resume(&mut ctx, &json!({"approved": true}));
        engine.update_permissions(section(
            &[("bash", PermissionLevel::Deny)],
            PermissionLevel::Ask,
        ));
        run(&rail, &mut ctx).await;
        let skip = ctx.take_skip_tool().unwrap();
        assert!(skip.starts_with("[PERMISSION_DENIED] "), "{skip}");
        assert!(skip.contains("Denied by rule: tools.bash"), "{skip}");
    }

    #[tokio::test]
    async fn auto_confirm_store_skips_the_interrupt() {
        let engine = ask_engine();
        let store = Arc::new(SessionPermissionStore::new());
        store.store_auto_confirm("bash:git status");
        let rail = PermissionInterruptRail::new(engine).with_store(store);
        let mut ctx = ctx_for(bash("git status"));
        run(&rail, &mut ctx).await;
        assert_eq!(ctx.interrupts.len(), 0);
        assert!(ctx.take_skip_tool().is_none());
    }

    #[tokio::test]
    async fn auto_confirm_context_flag_skips_the_interrupt() {
        let rail = PermissionInterruptRail::new(ask_engine());
        let mut ctx = ctx_for(bash("git status"));
        ctx.extra.insert(
            AUTO_CONFIRM_KEY.to_owned(),
            json!({"bash:git status": true}),
        );
        run(&rail, &mut ctx).await;
        assert_eq!(ctx.interrupts.len(), 0);
    }

    #[tokio::test]
    async fn hosted_confirmation_approval_runs() {
        let host = Arc::new(RecordingHost::new(true));
        *host.confirmation.lock() =
            Some(ConfirmationOutcome::Response(PermissionConfirmResponse {
                approved: true,
                ..PermissionConfirmResponse::default()
            }));
        let rail = PermissionInterruptRail::new(ask_engine()).with_host(host.clone());
        let mut ctx = ctx_for(bash("git status"));
        run(&rail, &mut ctx).await;
        assert_eq!(host.confirm_calls.load(Ordering::SeqCst), 1);
        assert_eq!(ctx.interrupts.len(), 0);
        assert!(ctx.take_skip_tool().is_none());
    }

    #[tokio::test]
    async fn hosted_failure_rejects() {
        let host = Arc::new(RecordingHost::new(true));
        *host.confirmation.lock() = Some(ConfirmationOutcome::Failed);
        let rail = PermissionInterruptRail::new(ask_engine()).with_host(host);
        let mut ctx = ctx_for(bash("git status"));
        run(&rail, &mut ctx).await;
        let skip = ctx.take_skip_tool().unwrap();
        assert!(skip.starts_with("[PERMISSION_DENIED] "), "{skip}");
        assert!(
            skip.contains("(Hosted permission request failed)"),
            "{skip}"
        );
    }

    #[tokio::test]
    async fn hosted_invalid_rejects() {
        let host = Arc::new(RecordingHost::new(true));
        *host.confirmation.lock() = Some(ConfirmationOutcome::Invalid);
        let rail = PermissionInterruptRail::new(ask_engine()).with_host(host);
        let mut ctx = ctx_for(bash("git status"));
        run(&rail, &mut ctx).await;
        let skip = ctx.take_skip_tool().unwrap();
        assert!(
            skip.contains("(Invalid hosted permission response)"),
            "{skip}"
        );
    }

    #[tokio::test]
    async fn scene_hook_approve_overrides_the_engine() {
        let host = Arc::new(RecordingHost::new(true));
        *host.scene.lock() = Some(SceneDecision::Approve);
        let engine = engine_for(section(
            &[("bash", PermissionLevel::Deny)],
            PermissionLevel::Ask,
        ));
        let rail = PermissionInterruptRail::new(engine).with_host(host);
        let mut ctx = ctx_for(bash("git status"));
        run(&rail, &mut ctx).await;
        assert_eq!(ctx.interrupts.len(), 0);
        assert!(ctx.take_skip_tool().is_none());
    }

    #[tokio::test]
    async fn scene_hook_reject_without_a_message_uses_the_prefix() {
        let host = Arc::new(RecordingHost::new(true));
        *host.scene.lock() = Some(SceneDecision::Reject(None));
        let engine = engine_for(section(
            &[("bash", PermissionLevel::Allow)],
            PermissionLevel::Allow,
        ));
        let rail = PermissionInterruptRail::new(engine).with_host(host);
        let mut ctx = ctx_for(bash("git status"));
        run(&rail, &mut ctx).await;
        assert_eq!(ctx.take_skip_tool().as_deref(), Some("[PERMISSION_DENIED]"));
    }

    #[tokio::test]
    async fn auto_confirm_key_shapes() {
        let rail = PermissionInterruptRail::new(ask_engine());
        assert_eq!(
            rail.auto_confirm_key(&bash("git status")),
            "bash:git status"
        );
        assert_eq!(rail.auto_confirm_key(&bash("git status && rm -rf /")), "");
        let call = ToolCall {
            id: Some("c2".to_owned()),
            name: "read_file".to_owned(),
            arguments: r#"{"file_path":"/tmp/a"}"#.to_owned(),
            ..ToolCall::default()
        };
        assert_eq!(rail.auto_confirm_key(&call), "read_file");
        assert_eq!(rail.name(), "permission_interrupt");
        assert_eq!(rail.priority(), RAIL_PRIORITY);
    }
}
