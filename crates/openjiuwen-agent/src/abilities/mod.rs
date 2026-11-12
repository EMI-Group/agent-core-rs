//! Tool mounting and execution for agents.
//!
//! Ports the registry half of `openjiuwen/core/single_agent/ability_manager.py`
//! without the legacy `Runner.resource_mgr` lookup: an [`AbilityManager`] owns
//! `Arc<dyn Tool>` instances directly and never resolves tools from a global
//! registry. The runner mounts tools on the manager; the `ReAct` loop only asks
//! it to execute.
//!
//! Execution is rail-free: [`AbilityManager::execute`] runs the calls and turns
//! every failure into a [`ToolCallResult`], while the loop fires the
//! `BeforeToolCall`/`AfterToolCall` rails around it.
pub mod arguments;
pub(crate) mod execute;

pub use arguments::parse_tool_arguments;
use indexmap::IndexMap;
use openjiuwen_foundation::{openai_tools, BaseMessage, Tool, ToolCall, ToolCard};
use serde_json::Value;
use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;
use std::time::Duration;
use tokio_util::sync::CancellationToken;

/// Outcome of [`AbilityManager::add_tool`].
///
/// `reason` is `added_tool`, `refreshed_tool`, or `duplicate_tool`; `added` is
/// `false` only for a rejected duplicate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AddAbilityResult {
    /// Name of the tool the result refers to.
    pub name: String,
    /// Whether the registry stores the tool after the call.
    pub added: bool,
    /// Machine-readable outcome (`added_tool`, `refreshed_tool`, `duplicate_tool`).
    pub reason: String,
}

impl AddAbilityResult {
    /// Builds an `added_tool` result for a newly registered tool.
    #[must_use]
    pub fn added_tool(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            added: true,
            reason: "added_tool".to_owned(),
        }
    }

    /// Builds a `refreshed_tool` result for a re-registered tool with the same
    /// card id.
    #[must_use]
    pub fn refreshed_tool(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            added: true,
            reason: "refreshed_tool".to_owned(),
        }
    }

    /// Builds a `duplicate_tool` result for a rejected same-name tool.
    #[must_use]
    pub fn duplicate_tool(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            added: false,
            reason: "duplicate_tool".to_owned(),
        }
    }
}

/// Per-batch execution settings for [`AbilityManager::execute`].
///
/// The type is runtime-only and never serializes: it carries a cancellation
/// token, which has no wire form.
#[derive(Debug, Clone)]
pub struct ToolExecutionOptions {
    /// Whether parallel-safe calls may run concurrently.
    pub parallel: bool,
    /// Session the calls belong to, when the agent is session-scoped.
    pub session_id: Option<String>,
    /// Deadline passed to each [`openjiuwen_foundation::ToolContext`].
    pub timeout: Option<Duration>,
    /// Cancellation signal shared with the tools.
    pub cancellation: CancellationToken,
    /// Free-form metadata copied into every tool context.
    pub metadata: BTreeMap<String, Value>,
}

impl Default for ToolExecutionOptions {
    /// Parallel execution enabled, with no session, deadline, metadata, or
    /// cancellation.
    fn default() -> Self {
        Self {
            parallel: true,
            session_id: None,
            timeout: None,
            cancellation: CancellationToken::new(),
            metadata: BTreeMap::new(),
        }
    }
}

/// Result of one tool call, including failures.
///
/// Failures are values rather than errors so the loop can always append a tool
/// message the model can react to; [`ToolCallResult::message`] renders it.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolCallResult {
    /// The model call this result answers.
    pub call: ToolCall,
    /// Raw output value on success.
    pub output: Option<Value>,
    /// Text the model reads; the error message on failure.
    pub text: String,
    /// Whether the tool completed successfully.
    pub success: bool,
    /// Error message when the call failed.
    pub error: Option<String>,
}

impl ToolCallResult {
    /// Builds a successful result from the raw `output` and its rendered `text`.
    #[must_use]
    pub fn success(call: ToolCall, output: Value, text: impl Into<String>) -> Self {
        Self {
            call,
            output: Some(output),
            text: text.into(),
            success: true,
            error: None,
        }
    }

    /// Builds a failed result; `error` becomes both the model-visible text and
    /// [`ToolCallResult::error`].
    #[must_use]
    pub fn failure(call: ToolCall, error: impl Into<String>) -> Self {
        let error = error.into();
        Self {
            call,
            output: None,
            text: error.clone(),
            success: false,
            error: Some(error),
        }
    }

    /// Renders the result as the `tool` message appended to the conversation.
    #[must_use]
    pub fn message(&self) -> BaseMessage {
        BaseMessage::tool(self.call.id.clone().unwrap_or_default(), self.text.clone())
            .with_name(self.call.name.clone())
    }
}

/// Ordered registry of the tools mounted on one agent.
///
/// Order is model-visible: [`AbilityManager::cards`] and
/// [`AbilityManager::tool_info`] return tools in registration order (adjusted by
/// [`AbilityManager::reorder`]). The registry is keyed by tool name; a duplicate
/// name whose card id differs is rejected instead of overwriting the existing
/// tool.
pub struct AbilityManager {
    owner_id: String,
    tools: IndexMap<String, Arc<dyn Tool>>,
    revision: u64,
}

impl AbilityManager {
    /// Creates an empty registry owned by `owner_id` (usually the agent id).
    #[must_use]
    pub fn new(owner_id: impl Into<String>) -> Self {
        Self {
            owner_id: owner_id.into(),
            tools: IndexMap::new(),
            revision: 0,
        }
    }

    /// Returns the id of the agent this registry belongs to.
    #[must_use]
    pub fn owner_id(&self) -> &str {
        &self.owner_id
    }

    /// Registers `tool` or refreshes an existing entry with the same name and
    /// card id.
    ///
    /// A same-name tool with a different card id is rejected without mutating
    /// the registry. Both a new registration and a refresh bump
    /// [`AbilityManager::revision`].
    #[must_use]
    pub fn add_tool(&mut self, tool: Arc<dyn Tool>) -> AddAbilityResult {
        let card = tool.card().clone();
        let name = card.name.clone();
        match self.tools.get(&name) {
            Some(existing) if existing.card().id == card.id => {
                self.tools.insert(name.clone(), tool);
                self.revision += 1;
                AddAbilityResult::refreshed_tool(name)
            }
            Some(_) => AddAbilityResult::duplicate_tool(name),
            None => {
                self.tools.insert(name.clone(), tool);
                self.revision += 1;
                AddAbilityResult::added_tool(name)
            }
        }
    }

    /// Removes and returns the tool named `name`, bumping the revision when an
    /// entry was removed.
    #[must_use]
    pub fn remove_tool(&mut self, name: &str) -> Option<Arc<dyn Tool>> {
        let removed = self.tools.shift_remove(name);
        if removed.is_some() {
            self.revision += 1;
        }
        removed
    }

    /// Returns the tool named `name`.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.get(name).cloned()
    }

    /// Returns the cards of all mounted tools in model-visible order.
    #[must_use]
    pub fn cards(&self) -> Vec<ToolCard> {
        self.tools
            .values()
            .map(|tool| tool.card().clone())
            .collect()
    }

    /// Returns the `OpenAI` `tools` projection of [`AbilityManager::cards`].
    ///
    /// An empty registry yields an empty vector.
    #[must_use]
    pub fn tool_info(&self) -> Vec<Value> {
        match openai_tools(&self.cards()) {
            Value::Array(items) => items,
            _ => Vec::new(),
        }
    }

    /// Returns the mutation counter, bumped on every successful change.
    #[must_use]
    pub fn revision(&self) -> u64 {
        self.revision
    }

    /// Returns the number of mounted tools.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Returns `true` when no tools are mounted.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Reorders the registry: `names` first in the given order, then the
    /// remaining tools in their previous order.
    ///
    /// Unknown names are ignored. The revision is bumped only when the
    /// resulting order differs from the current one.
    pub fn reorder(&mut self, names: &[String]) {
        let mut reordered: IndexMap<String, Arc<dyn Tool>> =
            IndexMap::with_capacity(self.tools.len());
        for name in names {
            if let Some(tool) = self.tools.get(name) {
                reordered.insert(name.clone(), Arc::clone(tool));
            }
        }
        for (name, tool) in &self.tools {
            if !reordered.contains_key(name) {
                reordered.insert(name.clone(), Arc::clone(tool));
            }
        }
        let order_unchanged = reordered.keys().eq(self.tools.keys());
        if !order_unchanged {
            self.tools = reordered;
            self.revision += 1;
        }
    }

    /// Executes `calls` in model order, one result per call.
    ///
    /// See `execute_calls` for the failure contract: unknown tools,
    /// malformed arguments, and tool errors are returned as failed results, not
    /// as `Err`.
    pub async fn execute(
        &self,
        calls: &[ToolCall],
        options: &ToolExecutionOptions,
    ) -> Vec<ToolCallResult> {
        execute::execute_calls(&self.tools, calls, options).await
    }
}

impl fmt::Debug for AbilityManager {
    /// Prints the owner, tool names in order, and revision; the tools
    /// themselves are opaque trait objects.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AbilityManager")
            .field("owner_id", &self.owner_id)
            .field("tools", &self.tools.keys().collect::<Vec<_>>())
            .field("revision", &self.revision)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use openjiuwen_common::error::Result;
    use openjiuwen_foundation::ToolContext;
    use serde_json::json;

    struct FakeTool {
        card: ToolCard,
    }

    #[async_trait]
    impl Tool for FakeTool {
        fn card(&self) -> &ToolCard {
            &self.card
        }

        async fn invoke(&self, inputs: Value, _ctx: &ToolContext) -> Result<Value> {
            Ok(inputs)
        }
    }

    fn fake_tool(name: &str, id: &str) -> Arc<dyn Tool> {
        Arc::new(FakeTool {
            card: ToolCard::new(name, "test tool", json!({"type": "object"})).with_id(id),
        })
    }

    fn names(manager: &AbilityManager) -> Vec<String> {
        manager.cards().into_iter().map(|card| card.name).collect()
    }

    #[test]
    fn adding_a_new_tool_reports_added_and_bumps_revision() {
        let mut manager = AbilityManager::new("agent-1");
        let result = manager.add_tool(fake_tool("echo", "id-1"));
        assert_eq!(result, AddAbilityResult::added_tool("echo"));
        assert!(manager.get("echo").is_some());
        assert_eq!(manager.revision(), 1);
        assert_eq!(manager.len(), 1);
        assert!(!manager.is_empty());
        assert_eq!(manager.owner_id(), "agent-1");
    }

    #[test]
    fn re_adding_the_same_card_id_refreshes_and_replaces() {
        let mut manager = AbilityManager::new("agent-1");
        let _ = manager.add_tool(fake_tool("echo", "id-1"));
        let replacement = fake_tool("echo", "id-1");
        let result = manager.add_tool(Arc::clone(&replacement));
        assert_eq!(result, AddAbilityResult::refreshed_tool("echo"));
        assert_eq!(manager.revision(), 2);
        assert!(Arc::ptr_eq(&manager.get("echo").unwrap(), &replacement));
    }

    #[test]
    fn re_adding_a_different_card_id_is_rejected_without_replacement() {
        let mut manager = AbilityManager::new("agent-1");
        let original = fake_tool("echo", "id-1");
        let _ = manager.add_tool(Arc::clone(&original));
        let result = manager.add_tool(fake_tool("echo", "id-2"));
        assert_eq!(result, AddAbilityResult::duplicate_tool("echo"));
        assert_eq!(manager.revision(), 1);
        assert!(Arc::ptr_eq(&manager.get("echo").unwrap(), &original));
    }

    #[test]
    fn removing_unknown_tools_returns_none_without_bumping() {
        let mut manager = AbilityManager::new("agent-1");
        assert!(manager.remove_tool("missing").is_none());
        assert_eq!(manager.revision(), 0);
        let _ = manager.add_tool(fake_tool("echo", "id-1"));
        assert!(manager.remove_tool("echo").is_some());
        assert!(manager.remove_tool("echo").is_none());
        assert_eq!(manager.revision(), 2);
        assert!(manager.is_empty());
    }

    #[test]
    fn reorder_puts_listed_names_first_and_drops_unknown_names() {
        let mut manager = AbilityManager::new("agent-1");
        for (name, id) in [("a", "id-a"), ("b", "id-b"), ("c", "id-c")] {
            let _ = manager.add_tool(fake_tool(name, id));
        }
        let revision = manager.revision();
        manager.reorder(&["c".to_owned(), "a".to_owned(), "missing".to_owned()]);
        assert_eq!(names(&manager), vec!["c", "a", "b"]);
        assert_eq!(manager.revision(), revision + 1);
        manager.reorder(&["c".to_owned(), "a".to_owned(), "b".to_owned()]);
        assert_eq!(manager.revision(), revision + 1);
        assert_eq!(names(&manager), vec!["c", "a", "b"]);
    }

    #[test]
    fn tool_info_is_an_openai_function_array() {
        let mut manager = AbilityManager::new("agent-1");
        assert_eq!(manager.tool_info(), Vec::<Value>::new());
        let _ = manager.add_tool(fake_tool("echo", "id-1"));
        let info = manager.tool_info();
        assert_eq!(info.len(), 1);
        assert_eq!(info[0]["type"], "function");
        assert_eq!(info[0]["function"]["name"], "echo");
    }

    #[test]
    fn debug_prints_owner_names_and_revision() {
        let mut manager = AbilityManager::new("agent-1");
        let _ = manager.add_tool(fake_tool("echo", "id-1"));
        let rendered = format!("{manager:?}");
        assert!(rendered.contains("agent-1"));
        assert!(rendered.contains("echo"));
        assert!(rendered.contains("revision: 1"));
    }

    #[tokio::test]
    async fn execute_returns_one_result_per_call_in_order() {
        let mut manager = AbilityManager::new("agent-1");
        let _ = manager.add_tool(fake_tool("echo", "id-1"));
        let calls = vec![
            ToolCall::new("echo")
                .with_id("call-1")
                .with_arguments(json!({"x": 1})),
            ToolCall::new("missing"),
        ];
        let results = manager
            .execute(&calls, &ToolExecutionOptions::default())
            .await;
        assert_eq!(results.len(), 2);
        assert!(results[0].success);
        assert_eq!(results[0].text, r#"{"x":1}"#);
        assert_eq!(results[0].message().tool_call_id.as_deref(), Some("call-1"));
        assert_eq!(results[0].message().name.as_deref(), Some("echo"));
        assert!(!results[1].success);
        assert_eq!(results[1].error.as_deref(), Some("unknown tool 'missing'"));
    }

    #[tokio::test]
    async fn execute_turns_malformed_arguments_into_a_failure() {
        let mut manager = AbilityManager::new("agent-1");
        let _ = manager.add_tool(fake_tool("echo", "id-1"));
        let calls = vec![ToolCall {
            arguments: "{".to_owned(),
            ..ToolCall::new("echo")
        }];
        let results = manager
            .execute(&calls, &ToolExecutionOptions::default())
            .await;
        assert_eq!(results.len(), 1);
        assert!(!results[0].success);
        assert!(results[0]
            .error
            .as_deref()
            .is_some_and(|error| error.contains("invalid tool arguments")));
    }

    #[tokio::test]
    async fn execute_marks_the_remaining_calls_cancelled() {
        let mut manager = AbilityManager::new("agent-1");
        let _ = manager.add_tool(fake_tool("echo", "id-1"));
        let options = ToolExecutionOptions::default();
        options.cancellation.cancel();
        let calls = vec![ToolCall::new("echo"), ToolCall::new("echo")];
        let results = manager.execute(&calls, &options).await;
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|result| !result.success));
        assert!(results
            .iter()
            .all(|result| result.error.as_deref() == Some("cancelled")));
    }
}
