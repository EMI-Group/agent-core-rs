//! Interrupt (human-in-the-loop) rails: pause a tool call for approval or user
//! input, and consume the answer on resume.
//!
//! Ports the kernel-side behaviour of Python `harness/rails/interrupt/`
//! (`BaseInterruptRail`, `ConfirmInterruptRail`) plus the `ask_user` kind, on
//! this crate's [`AgentRail`] seam. The loop owns state persistence and the
//! interrupt output; a rail only records an
//! [`InterruptRequest`](crate::interrupt::InterruptRequest) on the context.
use super::{AgentRail, RailContext};
use crate::interrupt::InterruptKind;
use async_trait::async_trait;
use openjiuwen_common::error::Result;
use openjiuwen_foundation::ToolCall;
use serde_json::Value;
use std::collections::BTreeSet;
use std::fmt;
use std::sync::Arc;
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
/// Intended behaviour (hooked in a later round): `before_tool_call` ignores
/// calls the [`ToolFilter`] rejects; for a matching call it consults the answers
/// seeded in `ctx.extra` (`_resume_interrupt_answers`) and the auto-confirm flag
/// (`_interrupt_auto_confirm`), then either approves (returns `Ok(())` so the
/// tool runs), replaces the result with the rejection/answer text through
/// [`RailContext::skip_tool_with`], or records
/// [`InterruptRequest::for_tool_call`](crate::interrupt::InterruptRequest::for_tool_call)
/// and pauses. An unanswered call only records the request; answers match by
/// tool-call id first, then by interrupt id, and a malformed answer re-asks with
/// a fresh id instead of approving.
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
}
#[async_trait]
impl AgentRail for BaseInterruptRail {
    fn name(&self) -> &'static str {
        "interrupt"
    }
    /// TODO: consult the seeded answers and auto-confirm flag, then approve,
    /// `skip_tool_with`, or record an [`InterruptRequest`](crate::interrupt::InterruptRequest) and
    /// pause.
    async fn before_tool_call(&self, _ctx: &mut RailContext) -> Result<()> {
        Ok(())
    }
}
/// The `confirm` rail: asks for approval before a matching tool runs.
///
/// Intended behaviour (hooked in a later round): a matching call pauses with a
/// [`InterruptKind::Confirm`] request carrying `{approved, feedback,
/// auto_confirm, persist_allow}` as its answer schema. Approval lets the tool
/// run, rejection replaces the tool result with the user's feedback, and an
/// auto-confirm flag or a seeded approval skips the pause. The request carries
/// the tool name as its auto-confirm key.
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
    /// TODO: pause matching calls for approval and consume the answer, as
    /// described on [`BaseInterruptRail`].
    async fn before_tool_call(&self, _ctx: &mut RailContext) -> Result<()> {
        Ok(())
    }
}
/// The `ask_user` rail: the user's answer becomes the tool result.
///
/// Intended behaviour (hooked in a later round): a matching call pauses with an
/// [`InterruptKind::AskUser`] request; on resume the answer is returned as the
/// tool result text without invoking the tool.
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
    /// TODO: pause matching calls for user input and return the answer as the
    /// tool result text.
    async fn before_tool_call(&self, _ctx: &mut RailContext) -> Result<()> {
        Ok(())
    }
}
/// Builds the shared base rail for a concrete interrupt rail.
fn base_rail(kind: InterruptKind, owner: &str, message: &str) -> BaseInterruptRail {
    BaseInterruptRail::new(kind, owner, message)
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
        }
    })
}
#[cfg(test)]
mod tests {
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
    async fn hooks_are_placeholders_that_do_not_interrupt() {
        let mut ctx = RailContext::new(crate::rails::RailEvent::BeforeToolCall);
        ctx.tool_call = Some(call("bash"));
        ConfirmInterruptRail::new()
            .before_tool_call(&mut ctx)
            .await
            .unwrap();
        AskUserRail::new().before_tool_call(&mut ctx).await.unwrap();
        BaseInterruptRail::new(InterruptKind::Permission, "o", "m")
            .before_tool_call(&mut ctx)
            .await
            .unwrap();
        assert!(!ctx.is_interrupted());
    }
}
