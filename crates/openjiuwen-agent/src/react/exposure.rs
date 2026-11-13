//! The model-facing tool projection of one run.
//!
//! The loop sends the model only the tools its [exposure
//! policy](crate::policy::ToolExposurePolicy) and [MCP
//! allowlist](crate::policy::McpToolAllowlist) permit, while the full tool map
//! stays available for execution. This module owns that projection so the
//! runtime snapshot and the loop share one definition of "model-visible".
//!
//! ## MCP convention
//!
//! Tools loaded through [`openjiuwen_foundation::load_mcp_tools`] keep the raw
//! remote tool name as [`ToolCard::name`] and record
//! `metadata["mcp"] = true`, `metadata["server_name"]`, and
//! `metadata["server_id"]`. The allowlist's server scope is the first present of
//! `metadata["server_name"]`, `metadata["server_id"]`, and
//! `metadata["mcp_server"]`; a card with no server scope is never filtered by
//! the allowlist, and neither is any card while the allowlist is empty.
//!
//! Python renames MCP tools to `mcp_<server>_<tool>` for the model, so the
//! allowlist is matched against the card name with an optional leading
//! `mcp_<server>_` prefix stripped, falling back to the raw card name. Both the
//! raw and the prefixed spelling therefore name the same underlying tool.
use crate::policy::{ClientToolRegistry, McpToolAllowlist, ToolExposurePolicy};
use openjiuwen_foundation::{ToolCard, ToolExecution};
use serde_json::Value;

/// Projects `tool_info` and `cards` to the tools the model may see.
///
/// The two input slices are positional (`openai_tools` projects a card per
/// entry); the policy's keep-mask is applied once, and the MCP allowlist then
/// drops denied cards, so the returned schemas and cards stay index-aligned and
/// keep their input order. Cards without a schema entry are dropped by the
/// positional zip, and trailing schemas without a card are ignored.
#[must_use]
pub(crate) fn visible_tools(
    tool_info: &[Value],
    cards: &[ToolCard],
    exposure: &ToolExposurePolicy,
    allowlist: &McpToolAllowlist,
) -> (Vec<Value>, Vec<ToolCard>) {
    let visible_cards = exposure.filter_cards(cards);
    let visible_info = exposure.filter_tool_info(tool_info, cards);
    let mut kept_info = Vec::with_capacity(visible_cards.len());
    let mut kept_cards = Vec::with_capacity(visible_cards.len());
    for (info, card) in visible_info.into_iter().zip(visible_cards) {
        if !mcp_allowed(&card, allowlist) {
            continue;
        }
        kept_info.push(info);
        kept_cards.push(card);
    }
    (kept_info, kept_cards)
}

/// Whether `card`'s MCP server admits the card's tool under `allowlist`.
///
/// A card without a server scope, and every card while the allowlist is empty,
/// is admitted. The allowlist name is the card name without an optional
/// `mcp_<server>_` prefix.
fn mcp_allowed(card: &ToolCard, allowlist: &McpToolAllowlist) -> bool {
    if allowlist.is_empty() {
        return true;
    }
    let Some(server) = mcp_server(card) else {
        return true;
    };
    allowlist.allows(server, allowlist_name(&card.name, server))
}

/// The MCP server scope recorded on `card`, if any.
fn mcp_server(card: &ToolCard) -> Option<&str> {
    ["server_name", "server_id", "mcp_server"]
        .into_iter()
        .find_map(|key| card.metadata.get(key).and_then(Value::as_str))
}

/// The allowlist spelling of `name`: `mcp_<server>_`-prefixed names are
/// stripped back to the underlying tool name, other names pass through.
fn allowlist_name<'a>(name: &'a str, server: &str) -> &'a str {
    let prefix = format!("mcp_{server}_");
    name.strip_prefix(&prefix).unwrap_or(name)
}

/// Whether `name` is executed by the connected client rather than this runtime.
///
/// A tool is a client tool when the registry has a handler for it, or when a
/// mounted card of that name declares
/// [`ToolExecution::Client`](openjiuwen_foundation::ToolExecution).
#[must_use]
pub(crate) fn is_client_tool(
    registry: &ClientToolRegistry,
    cards: &[ToolCard],
    name: &str,
) -> bool {
    registry.is_client_tool(name)
        || cards
            .iter()
            .any(|card| card.name == name && card.execution == ToolExecution::Client)
}

#[cfg(test)]
mod tests {
    use super::*;
    use openjiuwen_foundation::ToolCard;
    use serde_json::json;

    fn card(name: &str) -> ToolCard {
        ToolCard::new(name, "test tool", json!({"type": "object"}))
    }

    fn mcp_card(name: &str, server: &str) -> ToolCard {
        card(name)
            .with_metadata_entry("mcp", json!(true))
            .with_metadata_entry("server_name", json!(server))
    }

    fn info(cards: &[ToolCard]) -> Vec<Value> {
        cards.iter().map(ToolCard::openai_tool).collect()
    }

    #[test]
    fn allowlist_hides_a_card_from_both_projections_and_keeps_alignment() {
        let cards = [card("read"), card("write")];
        let policy = ToolExposurePolicy::allowlist(["read"]);
        let (info, visible) =
            visible_tools(&info(&cards), &cards, &policy, &McpToolAllowlist::new());
        assert_eq!(visible.len(), 1);
        assert_eq!(visible[0].name, "read");
        assert_eq!(info.len(), 1);
        assert_eq!(info[0]["function"]["name"], "read");
    }

    #[test]
    fn deny_hides_a_tool() {
        let cards = [card("read"), card("secret")];
        let policy = ToolExposurePolicy::allow_all().deny("secret");
        let (info, visible) =
            visible_tools(&info(&cards), &cards, &policy, &McpToolAllowlist::new());
        assert_eq!(visible.len(), 1);
        assert_eq!(visible[0].name, "read");
        assert_eq!(info[0]["function"]["name"], "read");
    }

    #[test]
    fn an_unrestricted_allowlist_keeps_an_mcp_card() {
        let cards = [mcp_card("read_file", "files")];
        let (info, visible) = visible_tools(
            &info(&cards),
            &cards,
            &ToolExposurePolicy::allow_all(),
            &McpToolAllowlist::new(),
        );
        assert_eq!(visible.len(), 1);
        assert_eq!(visible[0].name, "read_file");
        assert_eq!(info.len(), 1);
    }

    #[test]
    fn a_restricted_allowlist_hides_a_non_listed_mcp_tool() {
        let cards = [mcp_card("write_file", "files")];
        let mut allowlist = McpToolAllowlist::new();
        allowlist.allow("files", "read_file");
        let (info, visible) = visible_tools(
            &info(&cards),
            &cards,
            &ToolExposurePolicy::allow_all(),
            &allowlist,
        );
        assert_eq!(visible.len(), 0);
        assert_eq!(info.len(), 0);
    }

    #[test]
    fn a_restricted_allowlist_keeps_listed_mcp_tools_in_both_names() {
        let cards = [
            mcp_card("read_file", "files"),
            mcp_card("mcp_files_read_file", "files"),
        ];
        let mut allowlist = McpToolAllowlist::new();
        allowlist.allow("files", "read_file");
        let (info, visible) = visible_tools(
            &info(&cards),
            &cards,
            &ToolExposurePolicy::allow_all(),
            &allowlist,
        );
        assert_eq!(visible.len(), 2);
        assert_eq!(visible[0].name, "read_file");
        assert_eq!(visible[1].name, "mcp_files_read_file");
        assert_eq!(info.len(), 2);
        assert_eq!(info[0]["function"]["name"], "read_file");
        assert_eq!(info[1]["function"]["name"], "mcp_files_read_file");
    }

    #[test]
    fn a_non_mcp_card_is_unaffected_by_the_allowlist() {
        let cards = [card("read"), mcp_card("write_file", "files")];
        let mut allowlist = McpToolAllowlist::new();
        allowlist.allow("files", "read_file");
        let (_, visible) = visible_tools(
            &info(&cards),
            &cards,
            &ToolExposurePolicy::allow_all(),
            &allowlist,
        );
        assert_eq!(visible.len(), 1);
        assert_eq!(visible[0].name, "read");
    }

    #[test]
    fn is_client_tool_detects_the_registry_and_the_card_execution() {
        let registry = ClientToolRegistry::new();
        let cards = [
            card("run_on_client").with_execution(ToolExecution::Client),
            card("server_tool"),
        ];
        // The card execution flag alone makes it a client tool.
        assert!(is_client_tool(&registry, &cards, "run_on_client"));
        assert!(!is_client_tool(&registry, &cards, "server_tool"));

        let registry = ClientToolRegistry::new();
        registry.register("pick", noop_handler());
        let cards = [card("pick")];
        assert!(is_client_tool(&registry, &cards, "pick"));
        assert!(!is_client_tool(&registry, &cards, "missing"));
    }

    /// Minimal handler used to prove the registry path of `is_client_tool`.
    fn noop_handler() -> std::sync::Arc<dyn crate::policy::ClientToolHandler> {
        struct Noop;

        #[async_trait::async_trait]
        impl crate::policy::ClientToolHandler for Noop {
            async fn invoke(
                &self,
                _call: &openjiuwen_foundation::ToolCall,
                answer: &Value,
            ) -> openjiuwen_common::error::Result<Value> {
                Ok(answer.clone())
            }
        }

        std::sync::Arc::new(Noop)
    }
}
