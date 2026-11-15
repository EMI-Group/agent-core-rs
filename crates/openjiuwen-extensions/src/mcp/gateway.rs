//! `ToolGateway` over kernel tools: [`McpToolGateway`] and its projection types.
//!
//! The module docs of [`crate::mcp`] own the projection and outcome tables; this file implements
//! them.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_common::error::{Error, ErrorKind};
use openjiuwen_core::foundation::tool::{
    render_payload_text, render_tool_output, Tool, ToolCard, ToolContext, ToolOutput,
    EMPTY_SUCCESS_TEXT,
};
use openjiuwen_harness_protocol::{
    json_object_from_value, json_object_to_value, HarnessError, HarnessResult, JsonObject,
    JsonValue, McpServerConfig, ToolDefinition, ToolExecutionResult, ToolGateway, ToolInvocation,
    Validate,
};

/// Card metadata key every card projected by `load_mcp_tools` carries, set to `true`.
pub const MCP_SERVED_METADATA_KEY: &str = "mcp";

/// Card metadata key naming the MCP server whose tool a card projects.
pub const MCP_SERVER_NAME_METADATA_KEY: &str = "server_name";

/// Card metadata key carrying the id of the MCP server whose tool a card projects.
pub const MCP_SERVER_ID_METADATA_KEY: &str = "server_id";

/// [`ToolContext::metadata`] key carrying the harness call id of the invocation.
pub const HARNESS_CALL_ID_METADATA_KEY: &str = "harness_call_id";

/// Longest kernel error detail this gateway forwards, in characters.
///
/// Longer details are cut and suffixed with `…`, so a failing transport cannot push an unbounded
/// payload into the harness transcript.
const MAX_ERROR_DETAIL_CHARS: usize = 512;

/// A caller-supplied permission check: `true` means the named tool may be used.
///
/// The gateway consults it in `definitions` (a rejected tool is not advertised) and in `invoke`
/// (a rejected tool is a typed error). It is off by default.
pub type PermissionFilter = Arc<dyn Fn(&str) -> bool + Send + Sync>;

/// One tool this gateway serves: the harness-visible definition plus the card facts the harness
/// `ToolDefinition` cannot carry.
#[derive(Debug, Clone, PartialEq)]
pub struct McpToolEntry {
    /// The definition handed to the harness (`name`, `description`, `input_schema`).
    pub definition: ToolDefinition,
    /// Id of the kernel card the definition was projected from.
    ///
    /// For MCP tools this is `<server_id>.<server_name>.<tool name>`.
    pub card_id: String,
    /// Whether the kernel card allows concurrent calls with its siblings.
    pub parallel_safe: bool,
    /// The kernel card metadata, verbatim: `mcp`, `server_name`, and `server_id` for MCP tools.
    pub metadata: BTreeMap<String, JsonValue>,
}

impl McpToolEntry {
    /// Returns the tool name the harness invokes by.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.definition.name
    }

    /// Returns whether the kernel card marks this tool as MCP-served (`mcp: true`).
    #[must_use]
    pub fn is_mcp_served(&self) -> bool {
        self.metadata
            .get(MCP_SERVED_METADATA_KEY)
            .and_then(JsonValue::as_bool)
            .unwrap_or(false)
    }

    /// Returns the MCP server name the card carries, when it names one.
    #[must_use]
    pub fn server_name(&self) -> Option<&str> {
        self.metadata_string(MCP_SERVER_NAME_METADATA_KEY)
    }

    /// Returns the MCP server id the card carries, when it carries one.
    #[must_use]
    pub fn server_id(&self) -> Option<&str> {
        self.metadata_string(MCP_SERVER_ID_METADATA_KEY)
    }

    /// Reads one non-empty string metadata value.
    fn metadata_string(&self, key: &str) -> Option<&str> {
        self.metadata
            .get(key)?
            .as_str()
            .filter(|value| !value.is_empty())
    }
}

/// A harness [`ToolGateway`] over a fixed set of kernel tools.
///
/// Tools are sorted by name at construction, so [`McpToolGateway::entries`] and
/// [`ToolGateway::definitions`] are deterministic. The gateway holds no transport of its own: a
/// tool's `Tool::invoke` does the I/O, which for an `McpTool` is the MCP client call.
pub struct McpToolGateway {
    tools: Vec<ServedTool>,
    servers: Vec<McpServerConfig>,
    permission: Option<PermissionFilter>,
}

/// A served tool plus the entry projected from its card.
struct ServedTool {
    entry: McpToolEntry,
    tool: Arc<dyn Tool>,
}

impl fmt::Debug for McpToolGateway {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let entries: Vec<&McpToolEntry> = self.tools.iter().map(|served| &served.entry).collect();
        formatter
            .debug_struct("McpToolGateway")
            .field("tools", &entries)
            .field("servers", &self.servers)
            .field("has_permission_filter", &self.permission.is_some())
            .finish()
    }
}

impl McpToolGateway {
    /// Serves `tools`, projecting each card onto a [`McpToolEntry`].
    ///
    /// # Errors
    ///
    /// Returns [`HarnessError::Validation`] when a tool name is empty (after trimming) or when two
    /// tools share a name: the harness resolves tools by name, so an ambiguous name cannot be
    /// served without silently picking one.
    pub fn new(tools: impl IntoIterator<Item = Arc<dyn Tool>>) -> HarnessResult<Self> {
        let mut served = Vec::new();
        let mut names = BTreeSet::new();
        for tool in tools {
            let name = tool.card().name.trim();
            if name.is_empty() {
                return Err(HarnessError::validation(
                    "MCP tool gateway requires a non-empty tool name",
                ));
            }
            if !names.insert(tool.card().name.clone()) {
                return Err(HarnessError::validation(format!(
                    "duplicate MCP tool name `{}`: a gateway serves each name once",
                    tool.card().name
                )));
            }
            served.push(ServedTool {
                entry: project_entry(tool.card()),
                tool,
            });
        }
        served.sort_by(|left, right| left.entry.definition.name.cmp(&right.entry.definition.name));
        Ok(Self {
            tools: served,
            servers: Vec::new(),
            permission: None,
        })
    }

    /// Attaches the [`McpServerConfig`]s describing the servers whose tools this gateway serves.
    ///
    /// The configs are descriptive metadata for the host (which servers back the advertised
    /// tools); the harness still invokes through this gateway instead of connecting itself.
    /// Each config is validated with its own `Validate` rules and the list is sorted by name.
    ///
    /// # Errors
    ///
    /// Returns [`HarnessError::Validation`] for an invalid config (naming the server) or for two
    /// configs sharing a name.
    pub fn with_servers(
        mut self,
        servers: impl IntoIterator<Item = McpServerConfig>,
    ) -> HarnessResult<Self> {
        let mut servers: Vec<McpServerConfig> = servers.into_iter().collect();
        for server in &servers {
            if let Err(error) = server.validate() {
                return Err(HarnessError::validation(format!(
                    "invalid MCP server `{}`: {}",
                    server.name,
                    error.message()
                )));
            }
        }
        servers.sort_by(|left, right| left.name.cmp(&right.name));
        for pair in servers.windows(2) {
            if pair[0].name == pair[1].name {
                return Err(HarnessError::validation(format!(
                    "duplicate MCP server name `{}`: a gateway describes each server once",
                    pair[0].name
                )));
            }
        }
        self.servers = servers;
        Ok(self)
    }

    /// Returns this gateway with `filter` as its permission check.
    ///
    /// A tool the filter rejects is hidden from `definitions` and rejected with a typed
    /// [`HarnessError::State`] if it is invoked anyway.
    #[must_use]
    pub fn with_permission(mut self, filter: PermissionFilter) -> Self {
        self.permission = Some(filter);
        self
    }

    /// Returns every served entry, sorted by tool name, regardless of the permission filter.
    pub fn entries(&self) -> impl Iterator<Item = &McpToolEntry> {
        self.tools.iter().map(|served| &served.entry)
    }

    /// Returns the attached server configs, sorted by server name.
    #[must_use]
    pub fn servers(&self) -> &[McpServerConfig] {
        &self.servers
    }

    /// Returns whether the gateway currently serves no tools.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Returns whether the permission filter (or the absence of one) allows `name`.
    fn is_allowed(&self, name: &str) -> bool {
        self.permission.as_ref().is_none_or(|filter| filter(name))
    }

    /// Returns the served tool names, comma-separated, for an unknown-name error message.
    fn served_names(&self) -> String {
        let names: Vec<&str> = self
            .tools
            .iter()
            .map(|served| served.entry.definition.name.as_str())
            .collect();
        if names.is_empty() {
            "none".to_owned()
        } else {
            names.join(", ")
        }
    }
}

#[async_trait]
impl ToolGateway for McpToolGateway {
    async fn definitions(&self) -> HarnessResult<Vec<ToolDefinition>> {
        Ok(self
            .tools
            .iter()
            .filter(|served| self.is_allowed(&served.entry.definition.name))
            .map(|served| served.entry.definition.clone())
            .collect())
    }

    async fn invoke(&self, invocation: &ToolInvocation) -> HarnessResult<ToolExecutionResult> {
        invocation.validate()?;
        let served = self
            .tools
            .iter()
            .find(|served| served.entry.definition.name == invocation.name)
            .ok_or_else(|| {
                HarnessError::validation(format!(
                    "unknown MCP tool `{}`: this gateway serves [{}]",
                    invocation.name,
                    self.served_names()
                ))
            })?;
        if !self.is_allowed(&invocation.name) {
            return Err(HarnessError::state(format!(
                "MCP tool `{}` is denied by this gateway's permission filter",
                invocation.name
            )));
        }
        let context = ToolContext::new().with_metadata_entry(
            HARNESS_CALL_ID_METADATA_KEY,
            JsonValue::String(invocation.call_id.clone()),
        );
        let arguments = json_object_to_value(&invocation.arguments);
        match served.tool.invoke(arguments, &context).await {
            Ok(value) => Ok(success_result(&invocation.name, value)),
            Err(error) => map_kernel_error(&invocation.name, &error),
        }
    }
}

/// Projects one kernel card onto a [`McpToolEntry`].
fn project_entry(card: &ToolCard) -> McpToolEntry {
    McpToolEntry {
        definition: ToolDefinition {
            name: card.name.clone(),
            description: card.description.clone(),
            input_schema: input_schema(&card.input_params),
        },
        card_id: card.id.clone(),
        parallel_safe: card.parallel_safe,
        metadata: card.metadata.clone(),
    }
}

/// Projects a card's `input_params` onto the harness schema object.
///
/// A card whose schema is not a JSON object projects to `{}`, which accepts any arguments: the
/// harness wire type only carries object schemas.
fn input_schema(input_params: &JsonValue) -> JsonObject {
    json_object_from_value(input_params.clone()).unwrap_or_default()
}

/// Wraps a successful kernel value into a [`ToolExecutionResult`].
///
/// A kernel `ToolOutput` with `success: false` is a tool-level failure even though the kernel
/// returned it as a value, so it becomes an error result instead of a success payload.
fn success_result(tool_name: &str, value: JsonValue) -> ToolExecutionResult {
    match serde_json::from_value::<ToolOutput>(value.clone()) {
        Ok(output) if !output.success => {
            let detail = output
                .error
                .as_ref()
                .filter(|detail| !detail.is_empty())
                .cloned()
                .unwrap_or_else(|| render_tool_output(&output));
            error_result(failure_message(tool_name, &detail))
        }
        Ok(output) => {
            let text = render_tool_output(&output);
            ToolExecutionResult {
                content: success_content(text, output.data.unwrap_or(JsonValue::Null)),
                is_error: false,
            }
        }
        Err(_) => {
            let rendered = render_payload_text(&value);
            let text = if rendered.is_empty() {
                EMPTY_SUCCESS_TEXT.to_owned()
            } else {
                rendered
            };
            ToolExecutionResult {
                content: success_content(text, value),
                is_error: false,
            }
        }
    }
}

/// Maps a kernel invocation error onto the harness channel.
fn map_kernel_error(tool_name: &str, error: &Error) -> HarnessResult<ToolExecutionResult> {
    let kind = error.kind();
    let detail = truncate_detail(error.message().as_ref());
    let message = failure_message(tool_name, &detail);
    match kind {
        ErrorKind::Validation => Err(HarnessError::validation(message)),
        ErrorKind::Unsupported => Err(HarnessError::unsupported_capability(message)),
        ErrorKind::Config => Err(HarnessError::protocol(message)),
        ErrorKind::Cancelled => Err(HarnessError::state(format!(
            "MCP tool `{tool_name}` was cancelled before it produced a result"
        ))),
        _ => Ok(error_result(message)),
    }
}

/// Formats the harness-visible failure message for `tool_name`.
fn failure_message(tool_name: &str, detail: &str) -> String {
    format!("MCP tool `{tool_name}` failed: {detail}")
}

/// Builds the error-result payload: the failure message and nothing else.
fn error_result(message: String) -> ToolExecutionResult {
    let mut object = JsonObject::new();
    object.insert("error".to_owned(), JsonValue::String(message));
    ToolExecutionResult {
        content: json_object_to_value(&object),
        is_error: true,
    }
}

/// Builds the success payload: model-facing text plus the structured value it renders.
fn success_content(text: String, structured: JsonValue) -> JsonValue {
    let mut object = JsonObject::new();
    object.insert("text".to_owned(), JsonValue::String(text));
    object.insert("structured".to_owned(), structured);
    json_object_to_value(&object)
}

/// Truncates a kernel error detail to [`MAX_ERROR_DETAIL_CHARS`] characters.
fn truncate_detail(detail: &str) -> String {
    if detail.chars().count() <= MAX_ERROR_DETAIL_CHARS {
        return detail.to_owned();
    }
    let mut truncated: String = detail.chars().take(MAX_ERROR_DETAIL_CHARS).collect();
    truncated.push('…');
    truncated
}
