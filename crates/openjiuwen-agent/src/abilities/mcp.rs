//! MCP tool abilities: remote MCP tools mounted on an agent.
//!
//! Python parity: the MCP family of
//! `openjiuwen/core/single_agent/ability_manager.py` (`add_mcp_server` and the
//! `mcp_<server>_<tool>` naming it applies). This module keeps the same shape —
//! list a server's tools once, filter them, project the model-facing cards, and
//! mount each one — but expresses it as an ability that composes the
//! foundation's [`McpTool`] rather than reimplementing the transport.
//!
//! # Naming
//!
//! [`McpAbilityOptions::prefix`] is `true` by default and makes every
//! model-visible tool name `<namespace>__<tool>`, where `namespace` is
//! [`McpAbilityOptions::namespace`] when set and the client's
//! [`McpClient::server_name`] otherwise. The `__` separator is a documented
//! deviation: Python prefixes `mcp_<server>_<tool>`, which cannot be parsed
//! unambiguously when the server or tool name contains an underscore.
//! [`McpAbilityOptions::without_prefix`] keeps the remote name verbatim.
//!
//! Renaming only changes the card *name*: the card *id* stays the foundation
//! projection `<server_id>.<server_name>.<tool>`, so a tool keeps its registry
//! identity (and its `refreshed_tool` behaviour on re-mount) across naming
//! options. Cards always carry `remote_name` in `metadata` (the underlying tool
//! name) and, when prefixed, `namespace`, so a caller can recover the remote
//! name without parsing the prefixed one.
//!
//! # Allowlist keys
//!
//! [`McpAbilityOptions::allowlist`] is matched against the MCP **server id**
//! ([`McpClient::server_id`]) and falls back to [`McpClient::server_name`] when
//! the client reports an empty id — the same key the projection puts in the
//! card metadata. Empty allowlists allow everything; once any entry exists, a
//! server without an entry loses all of its tools.
//!
//! # Errors
//!
//! [`load_mcp_abilities`] propagates the [`McpClient::list_tools`] error
//! unchanged (a client failure is not a tool failure).
//!
//! [`McpToolAbility::invoke`] delegates to [`McpTool::invoke`] and maps its
//! failures — server-reported errors and transport errors — to
//! [`Error::Tool`] messages of the form
//! `MCP tool '<card name>' failed: <error>`, because the ability layer reports
//! execution problems as tool errors the model can react to. Cancellation and
//! timeouts pass through unchanged so the caller can distinguish an aborted run
//! or a deadline from a genuine MCP failure.

use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::{
    load_mcp_tools, McpClient, McpTool, McpToolOptions, Tool, ToolCard, ToolContext,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::fmt;
use std::sync::Arc;

use crate::policy::McpToolAllowlist;

/// One remote MCP tool mounted as an [`Tool`] on an agent.
///
/// The ability is a thin composition over [`McpTool`]: the inner tool owns the
/// transport (client, remote name, per-tool options) and this wrapper owns the
/// identity the ability layer reports and the error mapping described in the
/// [module docs](self).
pub struct McpToolAbility {
    inner: McpTool,
    remote_name: String,
}

impl McpToolAbility {
    /// Creates an ability for `tool_name` on `client`, exposed under `card`.
    ///
    /// `card` is the model-facing projection; `tool_name` is the name the MCP
    /// server knows the tool by.
    #[must_use]
    pub fn new(card: ToolCard, client: Arc<dyn McpClient>, tool_name: impl Into<String>) -> Self {
        let remote_name = tool_name.into();
        Self {
            inner: McpTool::new(card, client, remote_name.clone()),
            remote_name,
        }
    }

    /// Returns this ability with the per-tool MCP `options` applied.
    #[must_use]
    pub fn with_options(mut self, options: McpToolOptions) -> Self {
        self.inner = self.inner.with_options(options);
        self
    }

    /// Returns the underlying remote tool name, as the MCP server advertises it.
    #[must_use]
    pub fn remote_name(&self) -> &str {
        &self.remote_name
    }

    /// Returns the name of the MCP server this ability calls.
    #[must_use]
    pub fn server_name(&self) -> &str {
        self.inner.client().server_name()
    }

    /// Returns the inner foundation tool.
    #[must_use]
    pub fn mcp_tool(&self) -> &McpTool {
        &self.inner
    }

    /// Wraps this ability in the `Arc<dyn Tool>` a registry mounts.
    #[must_use]
    pub fn boxed(self) -> Arc<dyn Tool> {
        Arc::new(self)
    }
}

impl fmt::Debug for McpToolAbility {
    /// Prints the card and the remote/server names; the client itself is an
    /// opaque trait object and is never rendered.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("McpToolAbility")
            .field("card", self.inner.card())
            .field("remote_name", &self.remote_name)
            .field("server_name", &self.server_name())
            .finish()
    }
}

#[async_trait]
impl Tool for McpToolAbility {
    fn card(&self) -> &ToolCard {
        self.inner.card()
    }

    /// Calls the remote tool through the composed [`McpTool`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::Cancelled`] and [`Error::Timeout`] unchanged, the inner
    /// tool's [`Error::Validation`] unchanged, and any other failure mapped to
    /// [`Error::Tool`] as `MCP tool '<card name>' failed: <error>`.
    async fn invoke(&self, inputs: Value, ctx: &ToolContext) -> Result<Value> {
        self.inner
            .invoke(inputs, ctx)
            .await
            .map_err(|error| match error {
                Error::Cancelled | Error::Timeout(_) | Error::Validation(_) => error,
                other => Error::tool(format!(
                    "MCP tool '{}' failed: {other}",
                    self.inner.card().name
                )),
            })
    }

    /// Renders the call result with the inner tool's MCP payload rendering.
    fn render_for_llm(&self, output: &Value) -> String {
        self.inner.render_for_llm(output)
    }
}

/// How [`load_mcp_abilities`] names and filters a server's tools.
///
/// The `Default` is prefixing with the client's server name and no filtering;
/// [`McpAbilityOptions::new`] is the same thing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct McpAbilityOptions {
    /// Prefix used by [`McpAbilityOptions::prefix`]; defaults to
    /// [`McpClient::server_name`].
    pub namespace: Option<String>,
    /// Optional per-server allowlist of remote tool names, keyed by server id.
    pub allowlist: Option<McpToolAllowlist>,
    /// Whether to expose `<namespace>__<tool>` instead of the remote name.
    pub prefix: bool,
}

impl Default for McpAbilityOptions {
    /// Namespaced with the server name, with no allowlist.
    fn default() -> Self {
        Self {
            namespace: None,
            allowlist: None,
            prefix: true,
        }
    }
}

impl McpAbilityOptions {
    /// Creates the default options: namespaced by server name, no allowlist.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns these options with an explicit namespace override.
    #[must_use]
    pub fn with_namespace(mut self, namespace: impl Into<String>) -> Self {
        self.namespace = Some(namespace.into());
        self
    }

    /// Returns these options filtered by `allowlist`.
    #[must_use]
    pub fn with_allowlist(mut self, allowlist: McpToolAllowlist) -> Self {
        self.allowlist = Some(allowlist);
        self
    }

    /// Returns these options without a name prefix (remote names verbatim).
    #[must_use]
    pub fn without_prefix(mut self) -> Self {
        self.prefix = false;
        self
    }

    /// Returns these options with prefixed (`true`) or verbatim (`false`) names.
    #[must_use]
    pub fn with_prefix(mut self, prefix: bool) -> Self {
        self.prefix = prefix;
        self
    }
}

/// Loads every tool `client` advertises as a mounted [`Tool`].
///
/// Tools keep the server's order. Each card is the foundation projection of
/// [`load_mcp_tools`] — id `<server_id>.<server_name>.<tool>`, remote
/// description and input schema, metadata `mcp`/`server_name`/`server_id` —
/// extended with `remote_name` and, when prefixed, `namespace`; the card name
/// follows [`McpAbilityOptions::prefix`].
///
/// When [`McpAbilityOptions::allowlist`] is set, a tool is kept only if
/// `allowlist.allows(server, remote_name)` holds, where `server` is
/// [`McpClient::server_id`] and falls back to [`McpClient::server_name`] when
/// the id is empty.
///
/// # Errors
///
/// Returns the [`McpClient::list_tools`] error unchanged when the server cannot
/// be listed.
pub async fn load_mcp_abilities(
    client: Arc<dyn McpClient>,
    options: McpAbilityOptions,
) -> Result<Vec<Arc<dyn Tool>>> {
    let server_id = client.server_id().to_owned();
    let server_name = client.server_name().to_owned();
    let server = if server_id.is_empty() {
        server_name.clone()
    } else {
        server_id
    };
    let namespace = options
        .namespace
        .clone()
        .unwrap_or_else(|| server_name.clone());

    let tools = load_mcp_tools(client.clone(), McpToolOptions::default()).await?;
    let mut abilities: Vec<Arc<dyn Tool>> = Vec::with_capacity(tools.len());
    for tool in tools {
        let remote_name = tool.tool_name().to_owned();
        if let Some(allowlist) = &options.allowlist {
            if !allowlist.allows(&server, &remote_name) {
                continue;
            }
        }
        let mut card = tool.card().clone();
        card.metadata
            .insert("remote_name".to_owned(), json!(remote_name));
        if options.prefix {
            card.name = format!("{namespace}__{remote_name}");
            card.metadata
                .insert("namespace".to_owned(), json!(namespace));
        }
        abilities.push(
            McpToolAbility::new(card, client.clone(), remote_name)
                .with_options(McpToolOptions::default())
                .boxed(),
        );
    }
    Ok(abilities)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use openjiuwen_foundation::{McpRemoteTool, McpToolResult};
    use serde_json::Map;
    use std::sync::Mutex;

    /// A scripted [`McpClient`] recording every call it receives.
    struct FakeClient {
        name: String,
        id: String,
        tools: Vec<McpRemoteTool>,
        calls: Arc<Mutex<Vec<(String, Value)>>>,
        failure: Option<String>,
    }

    impl FakeClient {
        fn new(tools: Vec<McpRemoteTool>) -> Arc<Self> {
            Arc::new(Self {
                name: "files".to_owned(),
                id: "srv-1".to_owned(),
                tools,
                calls: Arc::new(Mutex::new(Vec::new())),
                failure: None,
            })
        }

        fn failing(tools: Vec<McpRemoteTool>, failure: &str) -> Arc<Self> {
            Arc::new(Self {
                name: "files".to_owned(),
                id: "srv-1".to_owned(),
                tools,
                calls: Arc::new(Mutex::new(Vec::new())),
                failure: Some(failure.to_owned()),
            })
        }

        fn with_id(self: &Arc<Self>, id: &str) -> Arc<Self> {
            Arc::new(Self {
                name: self.name.clone(),
                id: id.to_owned(),
                tools: self.tools.clone(),
                calls: Arc::clone(&self.calls),
                failure: self.failure.clone(),
            })
        }

        fn with_name(self: &Arc<Self>, name: &str) -> Arc<Self> {
            Arc::new(Self {
                name: name.to_owned(),
                id: self.id.clone(),
                tools: self.tools.clone(),
                calls: Arc::clone(&self.calls),
                failure: self.failure.clone(),
            })
        }

        fn recorded(&self) -> Vec<(String, Value)> {
            self.calls.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl McpClient for FakeClient {
        fn server_name(&self) -> &str {
            &self.name
        }

        fn server_id(&self) -> &str {
            &self.id
        }

        async fn connect(&self) -> Result<()> {
            Ok(())
        }

        async fn disconnect(&self) -> Result<()> {
            Ok(())
        }

        async fn is_connected(&self) -> bool {
            true
        }

        async fn list_tools(&self) -> Result<Vec<McpRemoteTool>> {
            Ok(self.tools.clone())
        }

        async fn call_tool_with(
            &self,
            tool_name: &str,
            arguments: Value,
            _timeout: Option<std::time::Duration>,
            _include_image_content: bool,
        ) -> Result<McpToolResult> {
            self.calls
                .lock()
                .unwrap()
                .push((tool_name.to_owned(), arguments));
            match &self.failure {
                Some(failure) => Err(Error::tool(failure.clone())),
                None => Ok(McpToolResult::success(Map::from_iter([(
                    "content".to_owned(),
                    json!("ok"),
                )]))),
            }
        }
    }

    fn remote(name: &str) -> McpRemoteTool {
        McpRemoteTool::new(
            name,
            format!("{name} description"),
            json!({"type": "object", "properties": {"path": {"type": "string"}}}),
        )
    }

    fn two_tools() -> Vec<McpRemoteTool> {
        vec![remote("read_file"), remote("list_dir")]
    }

    fn names(abilities: &[Arc<dyn Tool>]) -> Vec<String> {
        abilities
            .iter()
            .map(|tool| tool.card().name.clone())
            .collect()
    }

    #[tokio::test]
    async fn load_without_prefix_keeps_remote_names_and_projection() {
        let client = FakeClient::new(two_tools());
        let abilities = load_mcp_abilities(client, McpAbilityOptions::new().without_prefix())
            .await
            .unwrap();
        assert_eq!(names(&abilities), vec!["read_file", "list_dir"]);
        let card = abilities[0].card();
        assert_eq!(card.id, "srv-1.files.read_file");
        assert_eq!(card.description, "read_file description");
        assert_eq!(
            card.input_params,
            json!({"type": "object", "properties": {"path": {"type": "string"}}})
        );
        assert_eq!(card.metadata["mcp"], json!(true));
        assert_eq!(card.metadata["server_name"], json!("files"));
        assert_eq!(card.metadata["server_id"], json!("srv-1"));
        assert_eq!(card.metadata["remote_name"], json!("read_file"));
        assert!(!card.metadata.contains_key("namespace"));
    }

    #[tokio::test]
    async fn namespace_override_and_server_default_prefix_names() {
        let client = FakeClient::new(two_tools());
        let abilities = load_mcp_abilities(
            client.clone(),
            McpAbilityOptions::new().with_namespace("files"),
        )
        .await
        .unwrap();
        assert_eq!(
            names(&abilities),
            vec!["files__read_file", "files__list_dir"]
        );
        assert_eq!(abilities[0].card().id, "srv-1.files.read_file");

        let abilities = load_mcp_abilities(client, McpAbilityOptions::new())
            .await
            .unwrap();
        assert_eq!(
            names(&abilities),
            vec!["files__read_file", "files__list_dir"]
        );
    }

    #[tokio::test]
    async fn prefix_false_keeps_remote_names() {
        let client = FakeClient::new(two_tools());
        let abilities = load_mcp_abilities(
            client,
            McpAbilityOptions::new()
                .with_namespace("files")
                .with_prefix(false),
        )
        .await
        .unwrap();
        assert_eq!(names(&abilities), vec!["read_file", "list_dir"]);
    }

    #[tokio::test]
    async fn empty_allowlist_keeps_every_tool() {
        let client = FakeClient::new(two_tools());
        let abilities = load_mcp_abilities(
            client,
            McpAbilityOptions::new().with_allowlist(McpToolAllowlist::new()),
        )
        .await
        .unwrap();
        assert_eq!(
            names(&abilities),
            vec!["files__read_file", "files__list_dir"]
        );
    }

    #[tokio::test]
    async fn allowlist_keeps_only_listed_tools() {
        let client = FakeClient::new(two_tools());
        let mut allowlist = McpToolAllowlist::new();
        allowlist.allow("srv-1", "list_dir");
        let abilities =
            load_mcp_abilities(client, McpAbilityOptions::new().with_allowlist(allowlist))
                .await
                .unwrap();
        assert_eq!(names(&abilities), vec!["files__list_dir"]);
    }

    #[tokio::test]
    async fn allow_all_and_unknown_server_entries() {
        let client = FakeClient::new(two_tools());
        let mut allowlist = McpToolAllowlist::new();
        allowlist.allow_all("srv-1");
        let abilities = load_mcp_abilities(
            client.clone(),
            McpAbilityOptions::new().with_allowlist(allowlist),
        )
        .await
        .unwrap();
        assert_eq!(
            names(&abilities),
            vec!["files__read_file", "files__list_dir"]
        );

        let mut allowlist = McpToolAllowlist::new();
        allowlist.allow("other-server", "read_file");
        let abilities =
            load_mcp_abilities(client, McpAbilityOptions::new().with_allowlist(allowlist))
                .await
                .unwrap();
        assert!(abilities.is_empty());
    }

    #[tokio::test]
    async fn allowlist_falls_back_to_the_server_name_without_an_id() {
        let client = FakeClient::new(two_tools()).with_id("");
        let mut allowlist = McpToolAllowlist::new();
        allowlist.allow("files", "read_file");
        let abilities =
            load_mcp_abilities(client, McpAbilityOptions::new().with_allowlist(allowlist))
                .await
                .unwrap();
        assert_eq!(names(&abilities), vec!["files__read_file"]);
    }

    #[tokio::test]
    async fn invoking_forwards_arguments_and_renders_the_content() {
        let client = FakeClient::new(two_tools());
        let abilities =
            load_mcp_abilities(client.clone(), McpAbilityOptions::new().without_prefix())
                .await
                .unwrap();
        let output = abilities[0]
            .invoke(json!({"path": "/x"}), &ToolContext::new())
            .await
            .unwrap();
        assert_eq!(
            client.recorded(),
            vec![("read_file".to_owned(), json!({"path": "/x"}))]
        );
        assert_eq!(output["success"], json!(true));
        assert_eq!(output["data"]["content"], json!("ok"));
        assert_eq!(abilities[0].render_for_llm(&output), "ok");
    }

    #[tokio::test]
    async fn client_failures_become_tool_errors() {
        let client = FakeClient::failing(two_tools(), "server exploded");
        let abilities = load_mcp_abilities(client, McpAbilityOptions::new().without_prefix())
            .await
            .unwrap();
        let error = abilities[0]
            .invoke(json!({"path": "/x"}), &ToolContext::new())
            .await
            .unwrap_err();
        assert_eq!(error.kind(), openjiuwen_common::error::ErrorKind::Tool);
        let message = error.to_string();
        assert!(
            message.contains("MCP tool 'read_file' failed:"),
            "{message}"
        );
        assert!(message.contains("server exploded"), "{message}");
    }

    #[tokio::test]
    async fn cancellation_passes_through_unchanged() {
        let client = FakeClient::new(two_tools());
        let abilities = load_mcp_abilities(client, McpAbilityOptions::new().without_prefix())
            .await
            .unwrap();
        let context = ToolContext::new();
        context.cancellation.cancel();
        let error = abilities[0]
            .invoke(json!({"path": "/x"}), &context)
            .await
            .unwrap_err();
        assert_eq!(error.kind(), openjiuwen_common::error::ErrorKind::Cancelled);
    }

    #[tokio::test]
    async fn card_metadata_describes_the_remote_tool() {
        let client = FakeClient::new(two_tools());
        let abilities = load_mcp_abilities(client, McpAbilityOptions::new().with_namespace("fs"))
            .await
            .unwrap();
        let card = abilities[0].card();
        assert_eq!(card.name, "fs__read_file");
        assert_eq!(card.metadata["mcp"], json!(true));
        assert_eq!(card.metadata["server_name"], json!("files"));
        assert_eq!(card.metadata["server_id"], json!("srv-1"));
        assert_eq!(card.metadata["remote_name"], json!("read_file"));
        assert_eq!(card.metadata["namespace"], json!("fs"));
    }

    #[tokio::test]
    async fn debug_includes_the_card_and_server_names() {
        let client = FakeClient::new(two_tools()).with_name("files");
        let card = ToolCard::new("read_file", "read a file", json!({"type": "object"}));
        let ability = McpToolAbility::new(card, client, "read_file");
        let rendered = format!("{ability:?}");
        assert!(rendered.contains("McpToolAbility"), "{rendered}");
        assert!(rendered.contains("read_file"), "{rendered}");
        assert!(rendered.contains("files"), "{rendered}");
    }

    #[test]
    fn options_builders_set_namespace_allowlist_and_prefix() {
        assert_eq!(McpAbilityOptions::new(), McpAbilityOptions::default());
        assert!(McpAbilityOptions::default().prefix);
        let mut allowlist = McpToolAllowlist::new();
        allowlist.allow("srv-1", "read_file");
        let options = McpAbilityOptions::new()
            .with_namespace("fs")
            .with_allowlist(allowlist.clone())
            .without_prefix();
        assert_eq!(options.namespace.as_deref(), Some("fs"));
        assert_eq!(options.allowlist, Some(allowlist));
        assert!(!options.prefix);
        assert!(options.with_prefix(true).prefix);
    }

    #[test]
    fn ability_accessors_expose_card_remote_and_server_names() {
        let client = FakeClient::new(two_tools());
        let card = ToolCard::new("read_file", "read a file", json!({"type": "object"}));
        let ability = McpToolAbility::new(card, client, "read_file");
        assert_eq!(ability.card().name, "read_file");
        assert_eq!(ability.remote_name(), "read_file");
        assert_eq!(ability.server_name(), "files");
        assert_eq!(ability.mcp_tool().tool_name(), "read_file");
    }
}
