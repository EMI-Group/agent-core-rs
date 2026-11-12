//! Per-member blueprint types: the serializable [`DeepAgentSpec`] and its leaf specs.
//!
//! Python parity source: `openjiuwen/harness/schema/deep_agent_spec.py` (`DeepAgentSpec`,
//! `RailSpec`, `BuiltinToolSpec`, `SubAgentSpec`, `WorkspaceSpec`) and
//! `openjiuwen/harness/schema/extension_spec.py`.
//!
//! A [`DeepAgentSpec`] is cold, serializable data: it names a model, a card, a system prompt and
//! the capability lists (tools, MCPs, sub-agents, rails) that a member runtime materializes later.
//! Unknown Python fields are preserved in [`DeepAgentSpec::extra`] so a spec persisted by another
//! release survives a load/save cycle here.
//!
//! [`ToolSpec`] replaces Python's undiscriminated `ToolCard | BuiltinToolSpec` union with one
//! normalized shape; its [`Deserialize`] implementation accepts a bare string, a builtin object
//! (`{"type": …, "params": {…}}`) or a card-shaped object, and always serializes as
//! `{"name", "description", "input_schema"}`.

use std::collections::BTreeMap;

use openjiuwen_agent::AgentCard;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Map, Value};

use crate::constants::DEFAULT_LANGUAGE;

/// Default value of [`DeepAgentSpec::completion_timeout`] in seconds (Python `600.0`).
const DEFAULT_COMPLETION_TIMEOUT: f64 = 600.0;

/// Transport used by [`McpServerSpec`] when the spec does not name one.
const DEFAULT_MCP_TRANSPORT: &str = "stdio";

/// Default workspace root path of [`WorkspaceSpec`].
const DEFAULT_WORKSPACE_ROOT: &str = "./";

/// Returns an empty JSON object, the canonical "no arguments" schema.
fn empty_object() -> Value {
    Value::Object(Map::new())
}

/// A reference to a model by name, optionally carrying provider connection parameters.
///
/// Python parity: `TeamModelConfig` is the full client/request config; the Rust port keeps the
/// addressing subset (`model_name` / `provider`) plus free-form `params` so a spec round-trips
/// without importing the LLM crate's client configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case", default)]
pub struct ModelRef {
    /// Model name used when [`ModelRef::model_name`] is absent.
    pub name: Option<String>,
    /// Model name, preferred over [`ModelRef::name`].
    pub model_name: Option<String>,
    /// Provider key (`openai`, `anthropic`, …), when the spec pins one.
    pub provider: Option<String>,
    /// Extra request parameters forwarded to the provider, unchanged.
    pub params: BTreeMap<String, Value>,
}

impl ModelRef {
    /// Returns the effective model name: [`ModelRef::model_name`], else [`ModelRef::name`].
    #[must_use]
    pub fn resolved_name(&self) -> Option<&str> {
        self.model_name.as_deref().or(self.name.as_deref())
    }
}

/// The serializable blueprint of one `DeepAgent` member.
///
/// Python parity: `DeepAgentSpec`. Fields the Rust runtime does not implement (`vision_model`,
/// `audio_model`, `kv_cache_affinity_config`, …) are not modelled and instead survive in
/// [`DeepAgentSpec::extra`].
///
/// The hand-written [`Default`] mirrors the Python defaults: `enable_task_loop = true`,
/// `completion_timeout = 600.0`, `enable_security_rail = true`, every collection empty and every
/// optional field `None`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct DeepAgentSpec {
    /// Model the member runs on; `None` falls back to the team's model pool or the parent model.
    pub model: Option<ModelRef>,
    /// Agent card (identity, description, input/output schemas) exposed to teammates.
    pub card: Option<AgentCard>,
    /// System prompt handed to the member's agent runtime.
    pub system_prompt: Option<String>,
    /// Tools attached to the member, normalized by [`ToolSpec`].
    pub tools: Vec<ToolSpec>,
    /// MCP servers the member connects to.
    pub mcps: Vec<McpServerSpec>,
    /// Sub-agents the member may delegate to.
    pub subagents: Vec<SubAgentSpec>,
    /// Rails (guardrails, extra capabilities) declared by name.
    pub rails: Vec<RailSpec>,
    /// Whether the member runs the `DeepAgent` task loop; default `true`.
    pub enable_task_loop: bool,
    /// Upper bound on task-loop iterations; `None` means the runtime default.
    pub max_iterations: Option<usize>,
    /// Workspace layout for the member.
    pub workspace: Option<WorkspaceSpec>,
    /// Working directory the member's tools run in.
    pub cwd: Option<String>,
    /// Project root used for path resolution and sandbox restriction.
    pub project_root: Option<String>,
    /// Skills enabled for the member, when the spec narrows discovery.
    pub skills: Option<Vec<String>>,
    /// Language of prompts and rail messages; `None` inherits the team language.
    pub language: Option<String>,
    /// Tool names that require human approval before execution.
    pub approval_required_tools: Vec<String>,
    /// Seconds a member may wait for its task to complete; default `600.0`.
    pub completion_timeout: f64,
    /// Whether the security rail is installed; default `true`.
    pub enable_security_rail: bool,
    /// Prompt pack mode (for example `compact`), when the spec pins one.
    pub prompt_mode: Option<String>,
    /// Python fields this port does not model, preserved verbatim across a JSON round-trip.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

impl Default for DeepAgentSpec {
    fn default() -> Self {
        Self {
            model: None,
            card: None,
            system_prompt: None,
            tools: Vec::new(),
            mcps: Vec::new(),
            subagents: Vec::new(),
            rails: Vec::new(),
            enable_task_loop: true,
            max_iterations: None,
            workspace: None,
            cwd: None,
            project_root: None,
            skills: None,
            language: None,
            approval_required_tools: Vec::new(),
            completion_timeout: DEFAULT_COMPLETION_TIMEOUT,
            enable_security_rail: true,
            prompt_mode: None,
            extra: BTreeMap::new(),
        }
    }
}

impl DeepAgentSpec {
    /// Returns the member's model name, when the spec declares a model.
    #[must_use]
    pub fn model_name(&self) -> Option<&str> {
        self.model.as_ref().and_then(ModelRef::resolved_name)
    }

    /// Sets [`DeepAgentSpec::language`] and returns the updated spec.
    #[must_use]
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }
}

/// A tool attached to a member, normalized from any of the Python spec shapes.
///
/// Accepts, on input:
///
/// - a bare string, which is the tool name (`"bash"`);
/// - a builtin object `{"type": "bash", "params": {…}}`, where the name is `type`, the
///   description comes from `params.description` and the schema from `params.input_schema`, else
///   `params.parameters`, else `{}`;
/// - a card-shaped object, where the name/description come from `name`/`description` and the
///   schema from `input_schema`, else `input_params`, else `{}`.
///
/// Unknown keys are ignored. Output is always the canonical `{"name", "description",
/// "input_schema"}` object.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct ToolSpec {
    /// Name the model calls the tool by.
    pub name: String,
    /// Human-readable description shown to the model.
    pub description: String,
    /// JSON Schema of the accepted arguments; `{}` when the spec declares none.
    pub input_schema: Value,
}

impl Default for ToolSpec {
    fn default() -> Self {
        Self {
            name: String::new(),
            description: String::new(),
            input_schema: empty_object(),
        }
    }
}

/// Normalizes a builtin object `{"type": kind, "params": {…}}` into a [`ToolSpec`].
fn tool_from_builtin(kind: &str, params: Option<&Map<String, Value>>) -> ToolSpec {
    let description = params
        .and_then(|params| params.get("description"))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned();
    let input_schema = params
        .and_then(|params| {
            params
                .get("input_schema")
                .or_else(|| params.get("parameters"))
        })
        .cloned()
        .unwrap_or_else(empty_object);
    ToolSpec {
        name: kind.to_owned(),
        description,
        input_schema,
    }
}

/// Normalizes a card-shaped object into a [`ToolSpec`].
fn tool_from_card(map: &Map<String, Value>) -> ToolSpec {
    let name = map
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned();
    let description = map
        .get("description")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned();
    let input_schema = map
        .get("input_schema")
        .or_else(|| map.get("input_params"))
        .cloned()
        .unwrap_or_else(empty_object);
    ToolSpec {
        name,
        description,
        input_schema,
    }
}

impl<'de> Deserialize<'de> for ToolSpec {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        match value {
            Value::String(name) => Ok(Self {
                name,
                description: String::new(),
                input_schema: empty_object(),
            }),
            Value::Object(map) => {
                // A builtin object is `{"type": <registered name>, "params": {…}}`; a card always
                // carries `name`, so a string `type` without `name` selects the builtin shape.
                let builtin_type = map.get("type").and_then(Value::as_str);
                if map.contains_key("name") {
                    Ok(tool_from_card(&map))
                } else if let Some(kind) = builtin_type {
                    Ok(tool_from_builtin(
                        kind,
                        map.get("params").and_then(Value::as_object),
                    ))
                } else {
                    // Neither shape: tolerate it as an unnamed card so unknown keys never fail.
                    Ok(tool_from_card(&map))
                }
            }
            other => Err(serde::de::Error::custom(format!(
                "tool spec must be a string or an object, got {other}"
            ))),
        }
    }
}

/// A rail reference resolved by name through the rail provider registry.
///
/// Python parity: `RailSpec` (`type` + free-form `params`). The Python key `"type"` is kept on
/// the wire while the Rust field is named `kind`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case", default)]
pub struct RailSpec {
    /// Registered rail type, for example `security` or `skill_use`.
    #[serde(rename = "type")]
    pub kind: String,
    /// Parameters forwarded verbatim to the rail provider.
    pub params: BTreeMap<String, Value>,
}

/// An MCP server a member connects to.
///
/// Python parity: `McpServerConfig`, reduced to what the in-process runtime needs. The transport
/// defaults to `"stdio"`, so a spec that only names a command connects over stdio.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct McpServerSpec {
    /// Server name; the prefix of every generated tool id.
    pub name: String,
    /// Transport kind (`stdio`, `sse`, `streamable_http`); default `"stdio"`.
    pub transport: String,
    /// Command and arguments launched for a stdio server.
    pub command: Vec<String>,
    /// Endpoint URL for HTTP/SSE transports.
    pub url: Option<String>,
    /// Environment variables passed to a stdio server.
    pub env: BTreeMap<String, String>,
}

impl Default for McpServerSpec {
    fn default() -> Self {
        Self {
            name: String::new(),
            transport: DEFAULT_MCP_TRANSPORT.to_owned(),
            command: Vec::new(),
            url: None,
            env: BTreeMap::new(),
        }
    }
}

/// A sub-agent the member may delegate to.
///
/// Python parity: `SubAgentSpec`. Python's unset `language` inherits the parent language; Python's
/// `model`/`rails`/`skills`/`workspace` overrides are not modelled here and are preserved in
/// [`DeepAgentSpec::extra`] on the parent spec.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case", default)]
pub struct SubAgentSpec {
    /// Card describing the sub-agent, when it is addressed individually.
    pub agent_card: Option<AgentCard>,
    /// System prompt of the sub-agent.
    pub system_prompt: String,
    /// Tools of the sub-agent, normalized by [`ToolSpec`].
    pub tools: Vec<ToolSpec>,
    /// MCP servers the sub-agent connects to.
    pub mcps: Vec<McpServerSpec>,
    /// Registered name of the sub-agent, when it is listed in a roster.
    pub name: Option<String>,
}

/// Workspace layout of a member.
///
/// Python parity: `WorkspaceSpec`. Defaults are `"./"` for the root, [`DEFAULT_LANGUAGE`] for the
/// language and `stable_base: false`, so a spec that omits the workspace still allocates one.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct WorkspaceSpec {
    /// Root path of the workspace; default `"./"`.
    pub root_path: String,
    /// Language of generated workspace content; default [`DEFAULT_LANGUAGE`].
    pub language: String,
    /// Whether the workspace base is reused across runs instead of re-created; default `false`.
    pub stable_base: bool,
}

impl Default for WorkspaceSpec {
    fn default() -> Self {
        Self {
            root_path: DEFAULT_WORKSPACE_ROOT.to_owned(),
            language: DEFAULT_LANGUAGE.to_owned(),
            stable_base: false,
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn deep_agent_spec_defaults_match_python() {
        let spec = DeepAgentSpec::default();
        assert!(spec.enable_task_loop);
        assert!(spec.enable_security_rail);
        assert!((spec.completion_timeout - 600.0).abs() < f64::EPSILON);
        assert_eq!(
            serde_json::to_value(&spec).unwrap(),
            json!({
                "model": null,
                "card": null,
                "system_prompt": null,
                "tools": [],
                "mcps": [],
                "subagents": [],
                "rails": [],
                "enable_task_loop": true,
                "max_iterations": null,
                "workspace": null,
                "cwd": null,
                "project_root": null,
                "skills": null,
                "language": null,
                "approval_required_tools": [],
                "completion_timeout": 600.0,
                "enable_security_rail": true,
                "prompt_mode": null,
            })
        );
        // An empty JSON object parses to the same defaults.
        assert_eq!(
            serde_json::from_value::<DeepAgentSpec>(json!({})).unwrap(),
            spec
        );
    }

    #[test]
    fn deep_agent_spec_tolerates_and_preserves_unknown_fields() {
        let spec: DeepAgentSpec = serde_json::from_value(json!({
            "system_prompt": "hi",
            "enable_async_subagent": true,
            "vision_model": {"model": "gpt-4o"},
        }))
        .unwrap();
        assert_eq!(spec.system_prompt.as_deref(), Some("hi"));
        assert_eq!(spec.extra.get("enable_async_subagent"), Some(&json!(true)));
        assert_eq!(
            spec.extra.get("vision_model"),
            Some(&json!({"model": "gpt-4o"}))
        );

        let value = serde_json::to_value(&spec).unwrap();
        assert_eq!(value["enable_async_subagent"], json!(true));
        assert_eq!(value["vision_model"], json!({"model": "gpt-4o"}));
    }

    #[test]
    fn model_ref_resolves_model_name_first() {
        let model = ModelRef {
            name: Some("legacy-name".to_owned()),
            model_name: Some("gpt-4o".to_owned()),
            ..ModelRef::default()
        };
        assert_eq!(model.resolved_name(), Some("gpt-4o"));
        let legacy = ModelRef {
            name: Some("legacy-name".to_owned()),
            ..ModelRef::default()
        };
        assert_eq!(legacy.resolved_name(), Some("legacy-name"));
        assert_eq!(ModelRef::default().resolved_name(), None);

        let spec = DeepAgentSpec {
            model: Some(model),
            ..DeepAgentSpec::default()
        };
        assert_eq!(spec.model_name(), Some("gpt-4o"));
        assert_eq!(DeepAgentSpec::default().model_name(), None);
    }

    #[test]
    fn deep_agent_spec_with_language_sets_member_language() {
        let spec = DeepAgentSpec::default().with_language("en");
        assert_eq!(spec.language.as_deref(), Some("en"));
        assert_eq!(
            serde_json::to_value(&spec).unwrap()["language"],
            json!("en")
        );
    }

    #[test]
    fn tool_spec_accepts_bare_string() {
        let tool: ToolSpec = serde_json::from_value(json!("bash")).unwrap();
        assert_eq!(tool.name, "bash");
        assert_eq!(tool.description, "");
        assert_eq!(tool.input_schema, json!({}));
    }

    #[test]
    fn tool_spec_accepts_builtin_object() {
        let tool: ToolSpec = serde_json::from_value(json!({
            "type": "bash",
            "params": {
                "description": "run bash",
                "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}},
                "unknown_key": 1,
            },
        }))
        .unwrap();
        assert_eq!(tool.name, "bash");
        assert_eq!(tool.description, "run bash");
        assert_eq!(
            tool.input_schema,
            json!({"type": "object", "properties": {"cmd": {"type": "string"}}})
        );

        // `params.input_schema` wins over `params.parameters`.
        let tool: ToolSpec = serde_json::from_value(json!({
            "type": "read_file",
            "params": {"input_schema": {"type": "object", "required": ["path"]}, "parameters": {"x": 1}},
        }))
        .unwrap();
        assert_eq!(
            tool.input_schema,
            json!({"type": "object", "required": ["path"]})
        );
        assert_eq!(tool.description, "");

        // A builtin without params is still just its type.
        let tool: ToolSpec = serde_json::from_value(json!({"type": "todo"})).unwrap();
        assert_eq!(
            tool,
            ToolSpec {
                name: "todo".to_owned(),
                ..ToolSpec::default()
            }
        );
    }

    #[test]
    fn tool_spec_accepts_tool_card_object() {
        let tool: ToolSpec = serde_json::from_value(json!({
            "name": "grep",
            "description": "search",
            "input_params": {"type": "object"},
            "output_params": {"type": "string"},
            "exposure": "direct",
        }))
        .unwrap();
        assert_eq!(tool.name, "grep");
        assert_eq!(tool.description, "search");
        assert_eq!(tool.input_schema, json!({"type": "object"}));

        let tool: ToolSpec = serde_json::from_value(json!({
            "name": "grep",
            "input_schema": {"type": "object", "required": ["pattern"]},
        }))
        .unwrap();
        assert_eq!(
            tool.input_schema,
            json!({"type": "object", "required": ["pattern"]})
        );
        assert_eq!(tool.description, "");

        // A card without a schema gets the empty object, never null.
        let tool: ToolSpec = serde_json::from_value(json!({"name": "grep"})).unwrap();
        assert_eq!(tool.input_schema, json!({}));
    }

    #[test]
    fn tool_spec_rejects_non_object_non_string() {
        assert!(serde_json::from_value::<ToolSpec>(json!(42)).is_err());
        assert!(serde_json::from_value::<ToolSpec>(json!(["bash"])).is_err());
        assert!(serde_json::from_value::<ToolSpec>(json!(null)).is_err());
    }

    #[test]
    fn tool_spec_serializes_canonical_shape() {
        let tool: ToolSpec = serde_json::from_value(json!({
            "type": "bash",
            "params": {"description": "run bash", "input_schema": {"type": "object"}},
        }))
        .unwrap();
        assert_eq!(
            serde_json::to_value(&tool).unwrap(),
            json!({"name": "bash", "description": "run bash", "input_schema": {"type": "object"}})
        );
        let bare: ToolSpec = serde_json::from_value(json!("grep")).unwrap();
        assert_eq!(
            serde_json::to_value(&bare).unwrap(),
            json!({"name": "grep", "description": "", "input_schema": {}})
        );
    }

    #[test]
    fn rail_spec_serializes_type_and_params() {
        let rail = RailSpec {
            kind: "security".to_owned(),
            params: BTreeMap::new(),
        };
        assert_eq!(
            serde_json::to_value(&rail).unwrap(),
            json!({"type": "security", "params": {}})
        );
        let rail: RailSpec =
            serde_json::from_value(json!({"type": "skill_use", "params": {"dir": "x"}})).unwrap();
        assert_eq!(rail.kind, "skill_use");
        assert_eq!(rail.params.get("dir"), Some(&json!("x")));
        // The Rust field name is not accepted on input: `type` is the wire key.
        let rail: RailSpec = serde_json::from_value(json!({"kind": "security"})).unwrap();
        assert_eq!(rail.kind, "");
        assert_eq!(rail.params, BTreeMap::new());
    }

    #[test]
    fn mcp_server_spec_defaults_to_stdio() {
        let spec = McpServerSpec::default();
        assert_eq!(spec.transport, "stdio");
        assert_eq!(spec.command, Vec::<String>::new());
        assert_eq!(spec.url, None);
        assert_eq!(spec.env, BTreeMap::new());

        let spec: McpServerSpec =
            serde_json::from_value(json!({"name": "fs", "command": ["mcp-server-fs"]})).unwrap();
        assert_eq!(spec.transport, "stdio");
        assert_eq!(spec.command, vec!["mcp-server-fs".to_owned()]);
        assert_eq!(
            serde_json::to_value(&spec).unwrap()["transport"],
            json!("stdio")
        );
    }

    #[test]
    fn workspace_spec_defaults() {
        let spec = WorkspaceSpec::default();
        assert_eq!(spec.root_path, "./");
        assert_eq!(spec.language, DEFAULT_LANGUAGE);
        assert!(!spec.stable_base);
        assert_eq!(
            serde_json::to_value(&spec).unwrap(),
            json!({"root_path": "./", "language": DEFAULT_LANGUAGE, "stable_base": false})
        );
        assert_eq!(
            serde_json::from_value::<WorkspaceSpec>(json!({})).unwrap(),
            spec
        );
    }

    #[test]
    fn deep_agent_spec_round_trips_python_json() {
        let raw = json!({
            "model": {"model_name": "gpt-4o", "provider": "openai", "params": {"temperature": 0.2}},
            "card": {"id": "lead0001", "name": "team_leader", "description": "leads the team"},
            "system_prompt": "You are the leader.",
            "tools": [
                {"type": "bash", "params": {"description": "run bash"}},
                {"name": "grep", "description": "search", "input_params": {"type": "object"}},
            ],
            "mcps": [{"name": "fs", "command": ["mcp-server-fs"], "env": {"ROOT": "/tmp"}}],
            "rails": [{"type": "security", "params": {}}],
            "subagents": [
                {"name": "researcher", "system_prompt": "research", "agent_card": {"name": "researcher"}},
            ],
            "enable_task_loop": false,
            "max_iterations": 20,
            "workspace": {"root_path": "/work", "language": "en", "stable_base": true},
            "cwd": "/work",
            "project_root": "/work",
            "skills": ["pdf"],
            "language": "en",
            "approval_required_tools": ["bash"],
            "completion_timeout": 120.0,
            "enable_security_rail": false,
            "prompt_mode": "compact",
            "enable_async_subagent": true,
        });

        let spec: DeepAgentSpec = serde_json::from_value(raw).unwrap();
        assert_eq!(spec.model_name(), Some("gpt-4o"));
        assert_eq!(
            spec.card.as_ref().map(|card| card.name.as_str()),
            Some("team_leader")
        );
        assert_eq!(spec.tools.len(), 2);
        assert_eq!(spec.tools[0].name, "bash");
        assert_eq!(spec.tools[0].description, "run bash");
        assert_eq!(spec.tools[1].name, "grep");
        assert_eq!(spec.tools[1].input_schema, json!({"type": "object"}));
        assert_eq!(spec.mcps[0].transport, "stdio");
        assert_eq!(spec.rails[0].kind, "security");
        assert_eq!(spec.subagents[0].name.as_deref(), Some("researcher"));
        assert!(!spec.enable_task_loop);
        assert!(!spec.enable_security_rail);
        assert_eq!(spec.max_iterations, Some(20));
        assert_eq!(
            spec.workspace.as_ref().map(|ws| ws.root_path.as_str()),
            Some("/work")
        );
        assert_eq!(spec.completion_timeout, 120.0);
        assert_eq!(spec.skills, Some(vec!["pdf".to_owned()]));
        assert_eq!(spec.approval_required_tools, vec!["bash".to_owned()]);
        assert_eq!(spec.prompt_mode.as_deref(), Some("compact"));
        assert_eq!(spec.extra.get("enable_async_subagent"), Some(&json!(true)));

        // Serialize → parse → serialize is stable, and unknown keys survive it.
        let once = serde_json::to_value(&spec).unwrap();
        assert_eq!(
            once["tools"][0],
            json!({"name": "bash", "description": "run bash", "input_schema": {}})
        );
        assert_eq!(once["rails"][0], json!({"type": "security", "params": {}}));
        assert_eq!(once["enable_async_subagent"], json!(true));
        let reparsed: DeepAgentSpec = serde_json::from_value(once.clone()).unwrap();
        assert_eq!(serde_json::to_value(&reparsed).unwrap(), once);
        assert_eq!(reparsed, spec);
    }
}
