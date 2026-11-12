//! Interrupt (human-in-the-loop) rails: pause a tool call for approval or user
//! input, and consume the answer on resume.
//!
//! Ports the kernel-side behaviour of Python `harness/rails/interrupt/`
//! (`BaseInterruptRail`, `ConfirmInterruptRail`) plus the `ask_user` kind, on
//! this crate's [`AgentRail`] seam. The loop owns state persistence and the
//! interrupt output; a rail only records an [`InterruptRequest`] on the context.
use super::{AgentRail, RailContext};
use crate::abilities::parse_tool_arguments;
use crate::interrupt::{self, InterruptKind, InterruptRequest};
use async_trait::async_trait;
use openjiuwen_common::error::Result;
use openjiuwen_foundation::ToolCall;
use serde_json::Value;
use std::collections::BTreeSet;
use std::fmt;
use std::sync::Arc;
/// `ctx.extra` key holding the resume answers: `{ "<tool_call_id>": <answer> }`.
const RESUME_ANSWERS_KEY: &str = "_resume_interrupt_answers";
/// `ctx.extra` key holding the host's auto-confirm decision: a bool, or an
/// object keyed by tool name with bool values.
const AUTO_CONFIRM_KEY: &str = "_interrupt_auto_confirm";
/// Text returned for an empty ask-user answer.
const EMPTY_ANSWER_TEXT: &str = "User input: (empty)";
/// Message used when neither the call nor the rail supplies a question.
const FALLBACK_ASK_USER_MESSAGE: &str = "Please provide input to continue.";
/// Selects the tool calls an interrupt rail applies to.
#[derive(Clone, Default)]
pub enum ToolFilter {
    /// Every tool call is intercepted.
    #[default]
    All,
    /// Only the named tools are intercepted.
    Names(BTreeSet<String>),
    /// Only the calls the predicate accepts are intercepted.
    Any(Arc<dyn Fn(&ToolCall) -> bool + Send + Sync>),
}
impl ToolFilter {
    /// Intercepts the named tools only.
    #[must_use]
    pub fn names(names: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self::Names(names.into_iter().map(Into::into).collect())
    }
    /// Intercepts the calls `predicate` accepts.
    #[must_use]
    pub fn any(predicate: impl Fn(&ToolCall) -> bool + Send + Sync + 'static) -> Self {
        Self::Any(Arc::new(predicate))
    }
    /// Whether `call` is intercepted by this filter.
    #[must_use]
    pub fn matches(&self, call: &ToolCall) -> bool {
        match self {
            Self::All => true,
            Self::Names(names) => names.contains(&call.name),
            Self::Any(predicate) => predicate(call),
        }
    }
}
impl fmt::Debug for ToolFilter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::All => f.write_str("All"),
            Self::Names(names) => f.debug_tuple("Names").field(names).finish(),
            Self::Any(_) => f.write_str("Any(<predicate>)"),
        }
    }
}
/// Intercepts a matching tool call and asks the user for a decision.
///
/// `before_tool_call` ignores calls the [`ToolFilter`] rejects. For a matching
/// call it consults the answers seeded in `ctx.extra`
/// (`_resume_interrupt_answers`, an object keyed by tool-call id) and the
/// auto-confirm flag (`_interrupt_auto_confirm`, a bool or an object keyed by
/// tool name), then either approves (returns `Ok(())` so the tool runs),
/// replaces the result with the rejection/answer text through
/// [`RailContext::skip_tool_with`], or records
/// [`InterruptRequest::for_tool_call`](crate::interrupt::InterruptRequest::for_tool_call)
/// and pauses. An unanswered call only records the request; a malformed answer
/// re-asks with a fresh id instead of approving.
#[derive(Debug)]
pub struct BaseInterruptRail {
    kind: InterruptKind,
    owner: String,
    message: String,
    filter: ToolFilter,
    payload_schema: Value,
    auto_confirm_key: Option<String>,
}
impl BaseInterruptRail {
    /// Builds a rail of `kind` owned by `owner`, intercepting every tool.
    #[must_use]
    pub fn new(kind: InterruptKind, owner: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            kind,
            owner: owner.into(),
            message: message.into(),
            filter: ToolFilter::All,
            payload_schema: serde_json::json!({}),
            auto_confirm_key: None,
        }
    }
    /// Intercepts only the calls `filter` accepts.
    #[must_use]
    pub fn with_tools(mut self, filter: ToolFilter) -> Self {
        self.filter = filter;
        self
    }
    /// Intercepts only the calls `filter` accepts (alias of
    /// [`BaseInterruptRail::with_tools`]).
    #[must_use]
    pub fn with_filter(self, filter: ToolFilter) -> Self {
        self.with_tools(filter)
    }
    /// Sets the JSON schema of the accepted answer.
    #[must_use]
    pub fn with_payload_schema(mut self, schema: Value) -> Self {
        self.payload_schema = schema;
        self
    }
    /// Sets the session state key the host can use to auto-confirm.
    #[must_use]
    pub fn with_auto_confirm_key(mut self, key: impl Into<String>) -> Self {
        self.auto_confirm_key = Some(key.into());
        self
    }
    /// Returns the interrupt kind raised by this rail.
    #[must_use]
    pub fn kind(&self) -> &InterruptKind {
        &self.kind
    }
    /// Returns the owner string recorded on the request.
    #[must_use]
    pub fn owner(&self) -> &str {
        &self.owner
    }
    /// Returns the question shown to the user.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }
    /// Returns the tool filter.
    #[must_use]
    pub fn filter(&self) -> &ToolFilter {
        &self.filter
    }
    /// Returns the answer schema.
    #[must_use]
    pub fn payload_schema(&self) -> &Value {
        &self.payload_schema
    }
    /// Returns the auto-confirm state key, when configured.
    #[must_use]
    pub fn auto_confirm_key(&self) -> Option<&str> {
        self.auto_confirm_key.as_deref()
    }

    /// Runs the hook with `message` overriding the configured question.
    fn before_tool_call_with(&self, ctx: &mut RailContext, message: Option<String>) {
        let Some(call) = ctx.tool_call.clone() else {
            return;
        };
        if !self.filter.matches(&call) {
            return;
        }
        if let Some(answer) = resume_answer(ctx, &call) {
            match interrupt::parse_approval(&answer) {
                Some(true) => {}
                Some(false) => ctx.skip_tool_with(interrupt::reject_feedback(&answer)),
                None if matches!(self.kind, InterruptKind::AskUser) => {
                    ctx.skip_tool_with(answer_text(&answer));
                }
                None => self.record_request(ctx, &call, message),
            }
            return;
        }
        if auto_confirmed(ctx, &call) {
            return;
        }
        self.record_request(ctx, &call, message);
    }

    /// Records a fresh pause request for `call` on the context.
    fn record_request(&self, ctx: &mut RailContext, call: &ToolCall, message: Option<String>) {
        let message = message.unwrap_or_else(|| self.message.clone());
        let auto_confirm_key = self
            .auto_confirm_key
            .clone()
            .unwrap_or_else(|| call.name.clone());
        let request =
            InterruptRequest::for_tool_call(self.kind.clone(), self.owner.clone(), message, call)
                .with_payload_schema(self.payload_schema.clone())
                .with_auto_confirm_key(auto_confirm_key);
        ctx.request_interrupt(request);
    }
}
#[async_trait]
impl AgentRail for BaseInterruptRail {
    fn name(&self) -> &'static str {
        "interrupt"
    }

    /// Approves, skips, or pauses the intercepted tool call.
    async fn before_tool_call(&self, ctx: &mut RailContext) -> Result<()> {
        self.before_tool_call_with(ctx, None);
        Ok(())
    }
}
/// The `confirm` rail: asks for approval before a matching tool runs.
///
/// A matching call pauses with a [`InterruptKind::Confirm`] request carrying
/// `{approved, feedback, auto_confirm, persist_allow}` as its answer schema and
/// the tool name as its auto-confirm key. Approval lets the tool run, rejection
/// replaces the tool result with the user's feedback, and an auto-confirm flag
/// or a seeded approval skips the pause.
pub struct ConfirmInterruptRail {
    inner: BaseInterruptRail,
}
impl ConfirmInterruptRail {
    /// Builds the rail for every tool, with the canonical confirm schema.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: base_rail(
                InterruptKind::Confirm,
                "openjiuwen.agent.rails.interrupt.ConfirmInterruptRail",
                "Please approve or reject?",
            )
            .with_payload_schema(confirm_schema()),
        }
    }
    /// Builds the rail for the named tools only.
    #[must_use]
    pub fn for_tools(names: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            inner: Self::new().inner.with_tools(ToolFilter::names(names)),
        }
    }
    /// Returns the underlying rail configuration.
    #[must_use]
    pub fn inner(&self) -> &BaseInterruptRail {
        &self.inner
    }
}
impl Default for ConfirmInterruptRail {
    fn default() -> Self {
        Self::new()
    }
}
impl fmt::Debug for ConfirmInterruptRail {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ConfirmInterruptRail")
            .field("kind", self.inner.kind())
            .field("owner", &self.inner.owner())
            .field("message", &self.inner.message())
            .field("filter", self.inner.filter())
            .finish()
    }
}
#[async_trait]
impl AgentRail for ConfirmInterruptRail {
    fn name(&self) -> &'static str {
        "confirm_interrupt"
    }
    /// Pauses matching calls for approval and consumes the answer, as described
    /// on [`BaseInterruptRail`].
    async fn before_tool_call(&self, ctx: &mut RailContext) -> Result<()> {
        self.inner.before_tool_call_with(ctx, None);
        Ok(())
    }
}
/// The `ask_user` rail: the user's answer becomes the tool result.
///
/// A matching call pauses with an [`InterruptKind::AskUser`] request whose
/// message is the call's `question` argument (falling back to the configured
/// message); on resume the answer is returned as the tool result text without
/// invoking the tool.
pub struct AskUserRail {
    inner: BaseInterruptRail,
}
impl AskUserRail {
    /// Builds the rail for every tool.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: base_rail(
                InterruptKind::AskUser,
                "openjiuwen.agent.rails.interrupt.AskUserRail",
                "Please provide your input.",
            ),
        }
    }
    /// Builds the rail for the named tools only.
    #[must_use]
    pub fn for_tools(names: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            inner: Self::new().inner.with_tools(ToolFilter::names(names)),
        }
    }
    /// Returns the underlying rail configuration.
    #[must_use]
    pub fn inner(&self) -> &BaseInterruptRail {
        &self.inner
    }
}
impl Default for AskUserRail {
    fn default() -> Self {
        Self::new()
    }
}
impl fmt::Debug for AskUserRail {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AskUserRail")
            .field("kind", self.inner.kind())
            .field("owner", &self.inner.owner())
            .field("message", &self.inner.message())
            .field("filter", self.inner.filter())
            .finish()
    }
}
#[async_trait]
impl AgentRail for AskUserRail {
    fn name(&self) -> &'static str {
        "ask_user"
    }
    /// Pauses matching calls for user input and returns the answer as the tool
    /// result text.
    async fn before_tool_call(&self, ctx: &mut RailContext) -> Result<()> {
        let message = ctx
            .tool_call
            .as_ref()
            .map(|call| ask_user_message(call, self.inner.message()));
        self.inner.before_tool_call_with(ctx, message);
        Ok(())
    }
}
/// Builds the shared base rail for a concrete interrupt rail.
fn base_rail(kind: InterruptKind, owner: &str, message: &str) -> BaseInterruptRail {
    BaseInterruptRail::new(kind, owner, message)
}
/// Looks up the resume answer seeded for `call` (keyed by tool-call id).
fn resume_answer(ctx: &RailContext, call: &ToolCall) -> Option<Value> {
    let id = call.id.as_deref().filter(|id| !id.is_empty())?;
    ctx.extra
        .get(RESUME_ANSWERS_KEY)?
        .as_object()?
        .get(id)
        .cloned()
}
/// Whether the host asked to auto-confirm `call`.
fn auto_confirmed(ctx: &RailContext, call: &ToolCall) -> bool {
    match ctx.extra.get(AUTO_CONFIRM_KEY) {
        Some(Value::Bool(flag)) => *flag,
        Some(Value::Object(map)) => map
            .get(&call.name)
            .and_then(Value::as_bool)
            .unwrap_or(false),
        _ => false,
    }
}
/// Renders an ask-user answer as the tool result text.
fn answer_text(value: &Value) -> String {
    match value {
        Value::String(text) => non_empty_text(text),
        Value::Object(map) => ["text", "value", "answer", "feedback", "input"]
            .iter()
            .find_map(|key| map.get(*key).and_then(Value::as_str))
            .map_or_else(|| non_empty_text(&value.to_string()), non_empty_text),
        _ => non_empty_text(&value.to_string()),
    }
}
/// Returns `text`, or [`EMPTY_ANSWER_TEXT`] when it is blank.
fn non_empty_text(text: &str) -> String {
    if text.trim().is_empty() {
        EMPTY_ANSWER_TEXT.to_owned()
    } else {
        text.to_owned()
    }
}
/// Returns the question of an ask-user call, falling back to `message`.
fn ask_user_message(call: &ToolCall, message: &str) -> String {
    let question = parse_tool_arguments(&call.arguments)
        .ok()
        .and_then(|args| {
            args.get("question")
                .and_then(Value::as_str)
                .map(str::to_owned)
        })
        .unwrap_or_default();
    if !question.trim().is_empty() {
        return question;
    }
    if message.trim().is_empty() {
        FALLBACK_ASK_USER_MESSAGE.to_owned()
    } else {
        message.to_owned()
    }
}
/// The answer schema of [`ConfirmInterruptRail`].
fn confirm_schema() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "approved": {"type": "boolean"},
            "feedback": {"type": "string"},
            "auto_confirm": {"type": "boolean"},
            "persist_allow": {"type": "boolean"}
        },
        "required": ["approved"]
    })
}
#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use serde_json::json;

    fn call(name: &str) -> ToolCall {
        ToolCall {
            id: Some("call-1".to_owned()),
            name: name.to_owned(),
            arguments: "{}".to_owned(),
            ..ToolCall::default()
        }
    }

    fn call_with(name: &str, arguments: &str) -> ToolCall {
        ToolCall {
            arguments: arguments.to_owned(),
            ..call(name)
        }
    }

    fn ctx_for(call: ToolCall) -> RailContext {
        let mut ctx = RailContext::new(crate::rails::RailEvent::BeforeToolCall);
        ctx.tool_call = Some(call);
        ctx
    }
    #[test]
    fn filter_all_matches_everything() {
        assert!(ToolFilter::All.matches(&call("bash")));
        assert!(ToolFilter::default().matches(&call("anything")));
    }
    #[test]
    fn filter_names_matches_the_listed_tools() {
        let filter = ToolFilter::names(["bash", "edit"]);
        assert!(filter.matches(&call("bash")));
        assert!(filter.matches(&call("edit")));
        assert!(!filter.matches(&call("read")));
    }
    #[test]
    fn filter_any_runs_the_predicate() {
        let filter = ToolFilter::any(|call: &ToolCall| call.name.starts_with("write"));
        assert!(filter.matches(&call("write_file")));
        assert!(!filter.matches(&call("read_file")));
    }
    #[test]
    fn base_rail_builders_and_accessors() {
        let schema = json!({"type": "object"});
        let rail = BaseInterruptRail::new(InterruptKind::Permission, "owner", "run?")
            .with_tools(ToolFilter::names(["bash"]))
            .with_payload_schema(schema.clone())
            .with_auto_confirm_key("__auto__");
        assert_eq!(rail.kind(), &InterruptKind::Permission);
        assert_eq!(rail.owner(), "owner");
        assert_eq!(rail.message(), "run?");
        assert!(rail.filter().matches(&call("bash")));
        assert!(!rail.filter().matches(&call("read")));
        assert_eq!(rail.payload_schema(), &schema);
        assert_eq!(rail.auto_confirm_key(), Some("__auto__"));
        assert_eq!(AgentRail::name(&rail), "interrupt");
    }
    #[test]
    fn base_rail_defaults_to_every_tool_and_no_auto_confirm() {
        let rail = BaseInterruptRail::new(InterruptKind::AskUser, "owner", "which?");
        assert!(rail.filter().matches(&call("any")));
        assert_eq!(rail.payload_schema(), &json!({}));
        assert_eq!(rail.auto_confirm_key(), None);
    }
    #[test]
    fn with_filter_is_with_tools() {
        let rail = BaseInterruptRail::new(InterruptKind::Confirm, "owner", "ok?")
            .with_filter(ToolFilter::names(["bash"]));
        assert!(rail.filter().matches(&call("bash")));
        assert!(!rail.filter().matches(&call("edit")));
    }
    #[test]
    fn confirm_rail_defaults() {
        let rail = ConfirmInterruptRail::new();
        assert_eq!(AgentRail::name(&rail), "confirm_interrupt");
        assert_eq!(rail.inner().kind(), &InterruptKind::Confirm);
        assert_eq!(
            rail.inner().owner(),
            "openjiuwen.agent.rails.interrupt.ConfirmInterruptRail"
        );
        assert_eq!(rail.inner().message(), "Please approve or reject?");
        assert!(rail.inner().payload_schema()["properties"]["approved"].is_object());
        assert!(ConfirmInterruptRail::default()
            .inner()
            .filter()
            .matches(&call("any")));
    }
    #[test]
    fn confirm_rail_for_tools_filters_names() {
        let rail = ConfirmInterruptRail::for_tools(["bash"]);
        assert!(rail.inner().filter().matches(&call("bash")));
        assert!(!rail.inner().filter().matches(&call("edit")));
        assert_eq!(rail.inner().kind(), &InterruptKind::Confirm);
    }
    #[test]
    fn ask_user_rail_defaults() {
        let rail = AskUserRail::new();
        assert_eq!(AgentRail::name(&rail), "ask_user");
        assert_eq!(rail.inner().kind(), &InterruptKind::AskUser);
        assert_eq!(
            rail.inner().owner(),
            "openjiuwen.agent.rails.interrupt.AskUserRail"
        );
        assert_eq!(rail.inner().message(), "Please provide your input.");
        assert_eq!(AskUserRail::default().inner().owner(), rail.inner().owner());
        assert!(AskUserRail::for_tools(["bash"])
            .inner()
            .filter()
            .matches(&call("bash")));
        assert!(!AskUserRail::for_tools(["bash"])
            .inner()
            .filter()
            .matches(&call("edit")));
    }
    #[test]
    fn debug_output_does_not_panic() {
        assert!(format!("{:?}", ToolFilter::All).contains("All"));
        assert!(format!("{:?}", ToolFilter::names(["bash"])).contains("bash"));
        assert!(format!("{:?}", ToolFilter::any(|_: &ToolCall| true)).contains("predicate"));
        assert!(format!("{:?}", ConfirmInterruptRail::new()).contains("ConfirmInterruptRail"));
        assert!(format!("{:?}", AskUserRail::new()).contains("AskUserRail"));
        assert!(format!(
            "{:?}",
            BaseInterruptRail::new(InterruptKind::Confirm, "o", "m")
        )
        .contains("Confirm"));
    }
    #[tokio::test]
    async fn confirm_pauses_with_one_canonical_request() {
        let mut ctx = ctx_for(call_with("bash", "{}"));
        ConfirmInterruptRail::new()
            .before_tool_call(&mut ctx)
            .await
            .unwrap();
        assert!(ctx.is_interrupted());
        assert!(ctx.take_skip_tool().is_none());
        assert_eq!(ctx.interrupts.len(), 1);
        let request = &ctx.interrupts[0];
        assert_eq!(request.tool_call_id, "call-1");
        assert_eq!(request.tool_name, "bash");
        assert_eq!(request.kind, InterruptKind::Confirm);
        assert_eq!(
            request.owner,
            "openjiuwen.agent.rails.interrupt.ConfirmInterruptRail"
        );
        assert_eq!(request.message, "Please approve or reject?");
        assert_eq!(request.auto_confirm_key, "bash");
        assert_eq!(request.payload_schema["required"], json!(["approved"]));
        assert!(request.payload_schema["properties"]["approved"].is_object());
        assert!(request.interrupt_id.starts_with("intr_"));
    }

    #[tokio::test]
    async fn approve_answers_let_the_tool_run() {
        let approved = [
            json!({"call-1": true}),
            json!({"call-1": {"approved": true, "feedback": "ok"}}),
            json!({"call-1": "yes"}),
        ];
        for answers in approved {
            let mut ctx = ctx_for(call("bash"));
            ctx.extra
                .insert(RESUME_ANSWERS_KEY.to_owned(), answers.clone());
            ConfirmInterruptRail::new()
                .before_tool_call(&mut ctx)
                .await
                .unwrap();
            assert!(ctx.interrupts.is_empty(), "answers: {answers}");
            assert!(ctx.take_skip_tool().is_none(), "answers: {answers}");
        }
    }

    #[tokio::test]
    async fn reject_answers_skip_the_tool_with_feedback() {
        let mut ctx = ctx_for(call("bash"));
        ctx.extra.insert(
            RESUME_ANSWERS_KEY.to_owned(),
            json!({"call-1": {"approved": false}}),
        );
        ConfirmInterruptRail::new()
            .before_tool_call(&mut ctx)
            .await
            .unwrap();
        assert!(ctx.interrupts.is_empty(), "no interrupt expected");
        assert_eq!(
            ctx.take_skip_tool().as_deref(),
            Some(crate::interrupt::DEFAULT_REJECT_FEEDBACK)
        );

        let mut ctx = ctx_for(call("bash"));
        ctx.extra.insert(
            RESUME_ANSWERS_KEY.to_owned(),
            json!({"call-1": {"approved": false, "feedback": "no way"}}),
        );
        ConfirmInterruptRail::new()
            .before_tool_call(&mut ctx)
            .await
            .unwrap();
        assert!(ctx.interrupts.is_empty(), "no interrupt expected");
        assert!(ctx.take_skip_tool().unwrap().contains("no way"));
    }

    #[tokio::test]
    async fn ask_user_uses_the_question_argument_and_returns_the_answer() {
        let mut ctx = ctx_for(call_with("ask", r#"{"question":"Which file?"}"#));
        AskUserRail::new().before_tool_call(&mut ctx).await.unwrap();
        assert_eq!(ctx.interrupts.len(), 1);
        assert_eq!(ctx.interrupts[0].message, "Which file?");
        assert_eq!(ctx.interrupts[0].kind, InterruptKind::AskUser);
        assert!(ctx.take_skip_tool().is_none());

        let mut ctx = ctx_for(call_with("ask", r#"{"question":"Which file?"}"#));
        ctx.extra.insert(
            RESUME_ANSWERS_KEY.to_owned(),
            json!({"call-1": "src/main.rs"}),
        );
        AskUserRail::new().before_tool_call(&mut ctx).await.unwrap();
        assert!(ctx.interrupts.is_empty(), "no interrupt expected");
        assert_eq!(ctx.take_skip_tool().as_deref(), Some("src/main.rs"));
    }

    #[tokio::test]
    async fn ask_user_falls_back_when_the_call_has_no_question() {
        let mut ctx = ctx_for(call_with("ask", "{}"));
        AskUserRail::new().before_tool_call(&mut ctx).await.unwrap();
        assert_eq!(ctx.interrupts[0].message, "Please provide your input.");

        let mut ctx = ctx_for(call_with("ask", "{}"));
        AskUserRail::for_tools(["ask"])
            .before_tool_call(&mut ctx)
            .await
            .unwrap();
        assert_eq!(ctx.interrupts.len(), 1);
    }

    #[tokio::test]
    async fn ask_user_answer_objects_render_text() {
        let mut ctx = ctx_for(call_with("ask", r#"{"question":"q"}"#));
        ctx.extra.insert(
            RESUME_ANSWERS_KEY.to_owned(),
            json!({"call-1": {"text": "the answer"}}),
        );
        AskUserRail::new().before_tool_call(&mut ctx).await.unwrap();
        assert_eq!(ctx.take_skip_tool().as_deref(), Some("the answer"));
    }

    #[tokio::test]
    async fn auto_confirm_approves_globally_and_per_tool() {
        let mut ctx = ctx_for(call("bash"));
        ctx.extra.insert(AUTO_CONFIRM_KEY.to_owned(), json!(true));
        ConfirmInterruptRail::new()
            .before_tool_call(&mut ctx)
            .await
            .unwrap();
        assert!(ctx.interrupts.is_empty(), "no interrupt expected");

        let mut ctx = ctx_for(call("bash"));
        ctx.extra
            .insert(AUTO_CONFIRM_KEY.to_owned(), json!({"bash": true}));
        ConfirmInterruptRail::new()
            .before_tool_call(&mut ctx)
            .await
            .unwrap();
        assert!(ctx.interrupts.is_empty(), "no interrupt expected");

        let mut ctx = ctx_for(call("edit"));
        ctx.extra
            .insert(AUTO_CONFIRM_KEY.to_owned(), json!({"bash": true}));
        ConfirmInterruptRail::new()
            .before_tool_call(&mut ctx)
            .await
            .unwrap();
        assert_eq!(ctx.interrupts.len(), 1);
    }

    #[tokio::test]
    async fn filter_rejects_unlisted_tools() {
        let mut ctx = ctx_for(call("edit"));
        ConfirmInterruptRail::for_tools(["bash"])
            .before_tool_call(&mut ctx)
            .await
            .unwrap();
        assert!(ctx.interrupts.is_empty(), "no interrupt expected");
        assert!(ctx.take_skip_tool().is_none());
    }

    #[tokio::test]
    async fn missing_tool_call_is_ignored() {
        let mut ctx = RailContext::new(crate::rails::RailEvent::BeforeToolCall);
        ConfirmInterruptRail::new()
            .before_tool_call(&mut ctx)
            .await
            .unwrap();
        assert!(ctx.interrupts.is_empty(), "no interrupt expected");
    }

    #[tokio::test]
    async fn malformed_answer_re_asks_with_a_fresh_id() {
        let seed = json!({"call-1": {"unexpected": 1}});
        let mut ctx = ctx_for(call("bash"));
        ctx.extra.insert(RESUME_ANSWERS_KEY.to_owned(), seed);
        ConfirmInterruptRail::new()
            .before_tool_call(&mut ctx)
            .await
            .unwrap();
        assert!(ctx.take_skip_tool().is_none());
        assert_eq!(ctx.interrupts.len(), 1);

        let mut fresh = ctx_for(call("bash"));
        ConfirmInterruptRail::new()
            .before_tool_call(&mut fresh)
            .await
            .unwrap();
        assert_eq!(fresh.interrupts.len(), 1);
        assert_ne!(
            ctx.interrupts[0].interrupt_id,
            fresh.interrupts[0].interrupt_id
        );

        // Firing twice on the same context appends another fresh request.
        ConfirmInterruptRail::new()
            .before_tool_call(&mut ctx)
            .await
            .unwrap();
        assert_eq!(ctx.interrupts.len(), 2);
        assert_ne!(
            ctx.interrupts[0].interrupt_id,
            ctx.interrupts[1].interrupt_id
        );
    }

    #[test]
    fn answer_text_never_empty() {
        assert_eq!(answer_text(&json!("hello")), "hello");
        assert_eq!(answer_text(&json!("  ")), EMPTY_ANSWER_TEXT);
        assert_eq!(answer_text(&json!({"value": "v"})), "v");
        assert_eq!(
            answer_text(&json!({"unexpected": 1})),
            r#"{"unexpected":1}"#
        );
        assert_eq!(answer_text(&json!(42)), "42");
    }

    #[test]
    fn confirm_schema_requires_approval() {
        let schema = confirm_schema();
        assert_eq!(schema["required"], json!(["approved"]));
        assert_eq!(schema["type"], "object");
    }
}
