//! The model-facing team-memory tools of one member.
//!
//! Python parity source: `openjiuwen/agent_teams/memory/toolkit.py`, where `MemberMemoryToolkit`
//! advertises `team_memory_remember`, `team_memory_search`, `team_memory_forget` and
//! `team_memory_stats` to a model. The engine keeps that toolkit in
//! [`openjiuwen_memory::team::MemberMemoryToolkit`] — schemas plus a JSON dispatcher — and this
//! module projects it onto the crate-wide [`Tool`](openjiuwen_foundation::tool::Tool) seam so a
//! member harness can mount the four tools like any other.
//!
//! [`TeamMemoryTool`] owns no logic of its own:
//!
//! - the [`ToolCard`] is built from the toolkit's own [`get_tool_schemas`] entry, so the id, the
//!   name, the description and the input schema are exactly the engine's (nothing is re-worded
//!   here);
//! - [`Tool::invoke`] reports [`ToolContext`] cancellation as [`Error::Cancelled`] and otherwise
//!   delegates to [`MemberMemoryToolkit::handle_tool_call`].
//!
//! Failures are returned as `Err` rather than as the toolkit's `ok: false` envelope: the calling
//! tool layer (harness or agent runtime) owns that envelope, and wrapping it here would nest one
//! protocol inside another.
//!
//! [`get_tool_schemas`]: MemberMemoryToolkit::get_tool_schemas
use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::tool::{Tool, ToolCard, ToolContext};
use openjiuwen_memory::team::MemberMemoryToolkit;
use serde_json::Value;
use tracing::warn;

/// The tracing target of every log line this module emits.
const LOG_TARGET: &str = "openjiuwen_agent_teams::memory";

/// One `MemberMemoryToolkit` tool as a [`Tool`] handle.
pub struct TeamMemoryTool {
    /// The toolkit the tool dispatches into.
    toolkit: Arc<MemberMemoryToolkit>,
    /// The engine's tool name (`team_memory_remember`, ...).
    name: &'static str,
    /// The card built from the engine's schema for `name`.
    card: ToolCard,
}

impl std::fmt::Debug for TeamMemoryTool {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("TeamMemoryTool")
            .field("name", &self.name)
            .field("card_id", &self.card.id)
            .finish_non_exhaustive()
    }
}

impl TeamMemoryTool {
    /// Binds one toolkit tool to `toolkit`.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`] when the toolkit has no schema for `name` (an unknown tool name).
    pub fn new(toolkit: Arc<MemberMemoryToolkit>, name: &'static str) -> Result<Self> {
        let card = card_for(toolkit.as_ref(), name)?;
        Ok(Self {
            toolkit,
            name,
            card,
        })
    }

    /// Every tool of `toolkit`, in the toolkit's own order.
    ///
    /// The toolkit's schema list always covers its tool names, so no tool is dropped; a
    /// theoretical mismatch is logged and skipped rather than aborting the whole set.
    #[must_use]
    pub fn all(toolkit: Arc<MemberMemoryToolkit>) -> Vec<Arc<dyn Tool>> {
        let mut tools: Vec<Arc<dyn Tool>> = Vec::with_capacity(toolkit.tool_names().len());
        for name in toolkit.tool_names() {
            match Self::new(Arc::clone(&toolkit), name) {
                Ok(tool) => tools.push(Arc::new(tool) as Arc<dyn Tool>),
                Err(error) => {
                    warn!(
                        target: LOG_TARGET,
                        tool = name,
                        %error,
                        "skipping a team memory tool without a schema"
                    );
                }
            }
        }
        tools
    }

    /// The engine's tool name behind this handle.
    #[must_use]
    pub fn name(&self) -> &'static str {
        self.name
    }

    /// The toolkit this tool dispatches into.
    #[must_use]
    pub fn toolkit(&self) -> &Arc<MemberMemoryToolkit> {
        &self.toolkit
    }
}

#[async_trait]
impl Tool for TeamMemoryTool {
    fn card(&self) -> &ToolCard {
        &self.card
    }

    async fn invoke(&self, inputs: Value, ctx: &ToolContext) -> Result<Value> {
        ctx.check_cancelled()?;
        self.toolkit.handle_tool_call(self.name, inputs).await
    }
}

/// The four team-memory tools of `toolkit`, ready to mount on a member.
#[must_use]
pub fn team_memory_tools(toolkit: Arc<MemberMemoryToolkit>) -> Vec<Arc<dyn Tool>> {
    TeamMemoryTool::all(toolkit)
}

/// Builds the card of one toolkit tool from the engine's own schema entry.
fn card_for(toolkit: &MemberMemoryToolkit, name: &str) -> Result<ToolCard> {
    let schema = toolkit
        .get_tool_schemas()
        .into_iter()
        .find(|schema| schema.get("name").and_then(Value::as_str) == Some(name))
        .ok_or_else(|| {
            Error::validation(format!(
                "the member memory toolkit has no tool schema named `{name}`"
            ))
        })?;
    let id = required_str(&schema, "id", name)?;
    let description = required_str(&schema, "description", name)?;
    let input_params = schema.get("input_schema").cloned().ok_or_else(|| {
        Error::validation(format!(
            "the team memory tool schema `{name}` has no `input_schema`"
        ))
    })?;
    Ok(ToolCard::new(name, description, input_params).with_id(id))
}

/// Reads one required, non-empty string field of a tool schema.
fn required_str<'a>(schema: &'a Value, field: &str, name: &str) -> Result<&'a str> {
    schema
        .get(field)
        .and_then(Value::as_str)
        .filter(|text| !text.trim().is_empty())
        .ok_or_else(|| {
            Error::validation(format!(
                "the team memory tool schema `{name}` has no `{field}`"
            ))
        })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use openjiuwen_memory::team::{
        MemberMemoryContext, TeamMemory, TeamMemoryFilter, TOOL_FORGET, TOOL_REMEMBER, TOOL_SEARCH,
        TOOL_STATS,
    };
    use serde_json::json;

    use super::*;

    /// A writable toolkit for `dev-1` of `team-1` over a fresh in-memory store.
    fn toolkit() -> Arc<MemberMemoryToolkit> {
        Arc::new(MemberMemoryToolkit::new(
            Arc::new(TeamMemory::in_memory()),
            MemberMemoryContext::new("team-1").with_member("dev-1"),
        ))
    }

    #[tokio::test]
    async fn all_returns_the_four_tools_in_toolkit_order() {
        let toolkit = toolkit();
        let tools = team_memory_tools(Arc::clone(&toolkit));
        let names: Vec<&str> = tools.iter().map(|tool| tool.card().name.as_str()).collect();
        assert_eq!(
            names,
            vec![TOOL_REMEMBER, TOOL_SEARCH, TOOL_FORGET, TOOL_STATS]
        );
        assert_eq!(names, toolkit.tool_names());
    }

    #[tokio::test]
    async fn every_card_carries_the_toolkit_schema() {
        let toolkit = toolkit();
        let schemas = toolkit.get_tool_schemas();
        let tools = TeamMemoryTool::all(Arc::clone(&toolkit));
        assert_eq!(tools.len(), schemas.len());
        for (tool, schema) in tools.iter().zip(schemas.iter()) {
            let card = tool.card();
            assert_eq!(card.name, schema["name"].as_str().unwrap());
            assert_eq!(card.id, schema["id"].as_str().unwrap());
            assert_eq!(card.description, schema["description"].as_str().unwrap());
            assert_eq!(card.input_params, schema["input_schema"]);
            assert!(!card.description.trim().is_empty());
        }
    }

    #[test]
    fn an_unknown_name_is_rejected() {
        let error = TeamMemoryTool::new(toolkit(), "team_memory_nonsense").unwrap_err();
        assert!(matches!(error, Error::Validation(_)), "{error}");
        assert!(
            error.to_string().contains("team_memory_nonsense"),
            "{error}"
        );
    }

    #[tokio::test]
    async fn remembering_stores_a_record() {
        let toolkit = toolkit();
        let tools = team_memory_tools(Arc::clone(&toolkit));
        let remember = tools
            .iter()
            .find(|tool| tool.card().name == TOOL_REMEMBER)
            .expect("the remember tool");
        let output = remember
            .invoke(
                json!({"content": "The release branch freezes on Friday.", "kind": "decision"}),
                &ToolContext::new().with_session("s-1"),
            )
            .await
            .unwrap();
        assert_eq!(output["ok"], true);
        let records = toolkit
            .memory()
            .list(&TeamMemoryFilter::for_team("team-1"))
            .await
            .unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].content, "The release branch freezes on Friday.");
        assert_eq!(records[0].member_id.as_deref(), Some("dev-1"));
    }

    #[tokio::test]
    async fn a_bad_argument_returns_an_error() {
        let tools = team_memory_tools(toolkit());
        let remember = tools
            .iter()
            .find(|tool| tool.card().name == TOOL_REMEMBER)
            .expect("the remember tool");
        let error = remember
            .invoke(json!({}), &ToolContext::new())
            .await
            .unwrap_err();
        assert!(matches!(error, Error::Validation(_)), "{error}");
        let search = tools
            .iter()
            .find(|tool| tool.card().name == TOOL_SEARCH)
            .expect("the search tool");
        let error = search
            .invoke(json!({"query": 7}), &ToolContext::new())
            .await
            .unwrap_err();
        assert!(matches!(error, Error::Validation(_)), "{error}");
    }

    #[tokio::test]
    async fn a_cancelled_context_wins_over_every_argument() {
        let tools = team_memory_tools(toolkit());
        let remember = tools
            .iter()
            .find(|tool| tool.card().name == TOOL_REMEMBER)
            .expect("the remember tool");
        let ctx = ToolContext::new();
        ctx.cancellation.cancel();
        let error = remember
            .invoke(json!({"content": "A decision."}), &ctx)
            .await
            .unwrap_err();
        assert!(matches!(error, Error::Cancelled), "{error}");
    }

    #[tokio::test]
    async fn the_search_and_stats_tools_work() {
        let toolkit = toolkit();
        let tools = team_memory_tools(Arc::clone(&toolkit));
        toolkit
            .handle_tool_call(
                TOOL_REMEMBER,
                json!({"content": "The build runs on the nightly toolchain."}),
            )
            .await
            .unwrap();
        let search = tools
            .iter()
            .find(|tool| tool.card().name == TOOL_SEARCH)
            .expect("the search tool");
        let output = search
            .invoke(json!({"query": "nightly toolchain"}), &ToolContext::new())
            .await
            .unwrap();
        assert_eq!(output["ok"], true);
        assert_eq!(output["result"]["count"], 1);
        let stats = tools
            .iter()
            .find(|tool| tool.card().name == TOOL_STATS)
            .expect("the stats tool");
        let output = stats.invoke(json!({}), &ToolContext::new()).await.unwrap();
        assert_eq!(output["ok"], true);
    }

    #[tokio::test]
    async fn a_read_only_toolkit_refuses_remember() {
        let toolkit = Arc::new(MemberMemoryToolkit::new(
            Arc::new(TeamMemory::in_memory()),
            MemberMemoryContext::new("team-1")
                .with_member("dev-1")
                .read_only(),
        ));
        let tools = team_memory_tools(toolkit);
        let remember = tools
            .iter()
            .find(|tool| tool.card().name == TOOL_REMEMBER)
            .expect("the remember tool");
        let error = remember
            .invoke(json!({"content": "A decision."}), &ToolContext::new())
            .await
            .unwrap_err();
        assert!(matches!(error, Error::Unsupported(_)), "{error}");
    }
}
