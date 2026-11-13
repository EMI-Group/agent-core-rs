//! Tool exposure policy and MCP tool allowlists.
//!
//! Ports the registration-time half of
//! `openjiuwen/core/single_agent/ability_manager.py`
//! (`set_tool_exposure_policy` / `_apply_tool_exposure_policy`) and
//! `set_mcp_tool_allowlist` onto the shared
//! [`ToolCard`]/[`ToolExposure`] types.
//! The policy never mutates a card: it answers "may the model see this card?"
//! and projects card lists and `OpenAI` tool schemas accordingly.
use openjiuwen_foundation::{ToolCard, ToolExposure};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
/// How the agent decides which registered tools are model-visible.
///
/// Wire names are `allow_all`, `allowlist`, and `deny`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolExposureMode {
    /// Every tool is visible except the names in
    /// [`ToolExposurePolicy::denied`].
    #[default]
    AllowAll,
    /// Only the names in [`ToolExposurePolicy::allowed`] are visible.
    Allowlist,
    /// No tool is visible unless it is named in
    /// [`ToolExposurePolicy::allowed`] as an exception.
    Deny,
}
/// Which tools the model may see, plus optional deferred-tool filtering.
///
/// Precedence: a name in `denied` is always hidden, even when it is also in
/// `allowed`; otherwise [`ToolExposureMode`] decides. `hide_deferred` drops
/// cards whose [`ToolCard::exposure`] is
/// [`ToolExposure::Deferred`] from the model-facing projection, which is the
/// behaviour of Python's progressive tool disclosure when tool search is
/// disabled.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct ToolExposurePolicy {
    /// The visibility mode; [`ToolExposureMode::AllowAll`] by default.
    pub mode: ToolExposureMode,
    /// Names permitted in `allowlist`/`deny` mode (and as `deny` exceptions).
    pub allowed: BTreeSet<String>,
    /// Names always hidden, whatever the mode says.
    pub denied: BTreeSet<String>,
    /// Whether deferred tools are dropped from the model-facing projection.
    pub hide_deferred: bool,
}
impl ToolExposurePolicy {
    /// A policy that exposes every tool.
    #[must_use]
    pub fn allow_all() -> Self {
        Self {
            mode: ToolExposureMode::AllowAll,
            ..Self::default()
        }
    }
    /// A policy that exposes no tool and grants no exceptions.
    #[must_use]
    pub fn deny_all() -> Self {
        Self {
            mode: ToolExposureMode::Deny,
            ..Self::default()
        }
    }
    /// A policy that exposes only `names`.
    #[must_use]
    pub fn allowlist<N>(names: impl IntoIterator<Item = N>) -> Self
    where
        N: Into<String>,
    {
        Self {
            mode: ToolExposureMode::Allowlist,
            allowed: names.into_iter().map(Into::into).collect(),
            ..Self::default()
        }
    }
    /// Adds one allowed name; a consuming builder.
    #[must_use]
    pub fn allow(mut self, name: impl Into<String>) -> Self {
        self.allowed.insert(name.into());
        self
    }
    /// Adds one denied name; a consuming builder. Denial wins over allowance.
    #[must_use]
    pub fn deny(mut self, name: impl Into<String>) -> Self {
        self.denied.insert(name.into());
        self
    }
    /// Sets [`ToolExposurePolicy::hide_deferred`]; a consuming builder.
    #[must_use]
    pub fn with_hide_deferred(mut self, hide_deferred: bool) -> Self {
        self.hide_deferred = hide_deferred;
        self
    }
    /// Returns `true` when the policy itself (ignoring `hide_deferred`) lets
    /// the model see `card`.
    #[must_use]
    pub fn is_visible(&self, card: &ToolCard) -> bool {
        if self.denied.contains(&card.name) {
            return false;
        }
        match self.mode {
            ToolExposureMode::AllowAll => true,
            ToolExposureMode::Allowlist | ToolExposureMode::Deny => {
                self.allowed.contains(&card.name)
            }
        }
    }
    /// The effective exposure of `card` under this policy.
    ///
    /// Hidden cards and policy-visible deferred cards both resolve to
    /// [`ToolExposure::Deferred`]: the shared enum only distinguishes "directly
    /// visible" from "not directly visible". Whether a deferred card is still
    /// kept in the model-facing list is decided by
    /// [`ToolExposurePolicy::hide_deferred`] in
    /// [`ToolExposurePolicy::filter_cards`].
    #[must_use]
    pub fn resolve(&self, card: &ToolCard) -> ToolExposure {
        if self.is_visible(card) && card.exposure == ToolExposure::Direct {
            ToolExposure::Direct
        } else {
            ToolExposure::Deferred
        }
    }
    /// Returns the cards the model may see, preserving their order.
    #[must_use]
    pub fn filter_cards(&self, cards: &[ToolCard]) -> Vec<ToolCard> {
        cards
            .iter()
            .filter(|card| self.keeps(card))
            .cloned()
            .collect()
    }
    /// Returns the `OpenAI` tool schemas whose card is model-visible.
    ///
    /// Schemas are matched to `cards` positionally, as
    /// [`openjiuwen_foundation::openai_tools`] produces them; trailing schemas
    /// without a matching card are dropped.
    #[must_use]
    pub fn filter_tool_info(&self, tool_info: &[Value], cards: &[ToolCard]) -> Vec<Value> {
        tool_info
            .iter()
            .zip(cards)
            .filter(|(_, card)| self.keeps(card))
            .map(|(info, _)| info.clone())
            .collect()
    }
    fn keeps(&self, card: &ToolCard) -> bool {
        self.is_visible(card) && !(self.hide_deferred && card.exposure == ToolExposure::Deferred)
    }
}
/// Sentinel stored by [`McpToolAllowlist::allow_all`] to mean "every tool".
pub const ALL_TOOLS: &str = "*";
/// A per-MCP-server allowlist of underlying MCP tool names.
///
/// Python parity: `set_mcp_tool_allowlist` in
/// `openjiuwen/core/single_agent/ability_manager.py`. Tool names are the
/// underlying MCP names, before the model-facing `mcp_<server>_` prefix.
///
/// Semantics:
/// - an empty allowlist imposes no restriction at all (`allows` is always
///   `true`);
/// - once any entry exists, a server with no entry is denied every tool;
/// - a server with an entry allows exactly the listed names, and an entry
///   containing [`ALL_TOOLS`] allows every tool of that server;
/// - an entry with an empty set allows no tool.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct McpToolAllowlist {
    /// Allowed tool names per MCP server id.
    pub servers: BTreeMap<String, BTreeSet<String>>,
}
impl McpToolAllowlist {
    /// Creates an empty allowlist, which imposes no restriction.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
    /// Allows one tool of `server`.
    pub fn allow(&mut self, server: impl Into<String>, tool: impl Into<String>) {
        self.servers
            .entry(server.into())
            .or_default()
            .insert(tool.into());
    }
    /// Allows every tool of `server` via the [`ALL_TOOLS`] sentinel.
    pub fn allow_all(&mut self, server: impl Into<String>) {
        self.allow(server, ALL_TOOLS);
    }
    /// Returns `true` when no server is restricted.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.servers.is_empty()
    }
    /// Returns `true` when `tool` may be used on `server`.
    #[must_use]
    pub fn allows(&self, server: &str, tool: &str) -> bool {
        if self.servers.is_empty() {
            return true;
        }
        self.servers
            .get(server)
            .is_some_and(|allowed| allowed.contains(tool) || allowed.contains(ALL_TOOLS))
    }
    /// The allowed names recorded for `server`, if the server is restricted.
    #[must_use]
    pub fn tools(&self, server: &str) -> Option<&BTreeSet<String>> {
        self.servers.get(server)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    fn card(name: &str) -> ToolCard {
        ToolCard::new(name, "test tool", json!({}))
    }
    fn deferred(name: &str) -> ToolCard {
        card(name).with_exposure(ToolExposure::Deferred)
    }
    #[test]
    fn allow_all_hides_only_denied_tools() {
        let policy = ToolExposurePolicy::allow_all().deny("secret");
        assert!(policy.is_visible(&card("search")));
        assert!(!policy.is_visible(&card("secret")));
        assert_eq!(policy.resolve(&card("search")), ToolExposure::Direct);
        assert_eq!(policy.resolve(&card("secret")), ToolExposure::Deferred);
    }
    #[test]
    fn allowlist_mode_hides_everything_else() {
        let policy = ToolExposurePolicy::allowlist(["search", "read"]);
        assert!(policy.is_visible(&card("search")));
        assert!(!policy.is_visible(&card("write")));
        assert_eq!(policy.filter_cards(&[card("read"), card("write")]).len(), 1);
    }
    #[test]
    fn denied_wins_over_allowed() {
        let policy = ToolExposurePolicy::allowlist(["search"]).deny("search");
        assert!(!policy.is_visible(&card("search")));
        let policy = ToolExposurePolicy::allow_all()
            .allow("search")
            .deny("search");
        assert!(!policy.is_visible(&card("search")));
    }
    #[test]
    fn deny_mode_allows_only_explicit_exceptions() {
        let policy = ToolExposurePolicy::deny_all().allow("search");
        assert!(policy.is_visible(&card("search")));
        assert!(!policy.is_visible(&card("write")));
        assert_eq!(policy.mode, ToolExposureMode::Deny);
    }
    #[test]
    fn deferred_tools_resolve_to_deferred() {
        let policy = ToolExposurePolicy::allow_all();
        assert_eq!(policy.resolve(&deferred("search")), ToolExposure::Deferred);
        assert_eq!(policy.filter_cards(&[deferred("search")]).len(), 1);
    }
    #[test]
    fn hide_deferred_demotes_deferred_tools() {
        let policy = ToolExposurePolicy::allow_all().with_hide_deferred(true);
        assert_eq!(policy.resolve(&deferred("search")), ToolExposure::Deferred);
        let kept = policy.filter_cards(&[card("read"), deferred("search")]);
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].name, "read");
    }
    #[test]
    fn tool_info_filtering_is_positional() {
        let policy = ToolExposurePolicy::allowlist(["read"]);
        let cards = [card("read"), card("write"), card("search")];
        let info: Vec<Value> = cards.iter().map(ToolCard::openai_tool).collect();
        let filtered = policy.filter_tool_info(&info, &cards);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0]["function"]["name"], "read");
    }
    #[test]
    fn policy_round_trips_and_tolerates_unknown_fields() {
        let policy = ToolExposurePolicy {
            mode: ToolExposureMode::Allowlist,
            allowed: ["read".to_owned()].into_iter().collect(),
            denied: ["write".to_owned()].into_iter().collect(),
            hide_deferred: true,
        };
        let value = serde_json::to_value(&policy).expect("serialize");
        assert_eq!(value["mode"], "allowlist");
        let back: ToolExposurePolicy = serde_json::from_value(value).expect("deserialize");
        assert_eq!(back, policy);
        let tolerant: ToolExposurePolicy =
            serde_json::from_value(json!({"unknown": 1})).expect("defaults");
        assert_eq!(tolerant, ToolExposurePolicy::allow_all());
    }
    #[test]
    fn empty_mcp_allowlist_restricts_nothing() {
        let allowlist = McpToolAllowlist::new();
        assert!(allowlist.is_empty());
        assert!(allowlist.allows("files", "read_file"));
        assert_eq!(allowlist.tools("files"), None);
    }
    #[test]
    fn mcp_allowlist_scopes_to_named_servers() {
        let mut allowlist = McpToolAllowlist::new();
        allowlist.allow("files", "read_file");
        allowlist.allow("files", "list_dir");
        assert!(!allowlist.is_empty());
        assert!(allowlist.allows("files", "read_file"));
        assert!(!allowlist.allows("files", "write_file"));
        assert!(!allowlist.allows("db", "query"));
        assert_eq!(allowlist.tools("files").map(BTreeSet::len), Some(2));
    }
    #[test]
    fn mcp_allow_all_uses_the_wildcard_sentinel() {
        let mut allowlist = McpToolAllowlist::new();
        allowlist.allow_all("files");
        assert!(allowlist.allows("files", "anything"));
        assert!(!allowlist.allows("db", "query"));
        assert!(allowlist
            .tools("files")
            .is_some_and(|t| t.contains(ALL_TOOLS)));
    }
    #[test]
    fn mcp_empty_server_entry_allows_no_tool() {
        let mut allowlist = McpToolAllowlist::new();
        allowlist
            .servers
            .insert("files".to_owned(), BTreeSet::new());
        assert!(!allowlist.allows("files", "read_file"));
    }
}
