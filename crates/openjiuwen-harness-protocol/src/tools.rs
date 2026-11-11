//! Native-tool and MCP service descriptions for external harnesses.
//!
//! Mirrors `openjiuwen.harness_protocol.tools` (v0.1.20): [`ToolDefinition`],
//! [`ToolInvocation`], [`ToolExecutionResult`], [`McpTransport`], [`McpServerConfig`], and the
//! [`ToolGateway`] host service.
//!
//! Wire shape: `snake_case` keys, every key always present (`null` for [`Option::None`], `[]` for
//! empty sequences, `{}` for empty objects), unknown keys tolerated on decode. Python's
//! `MappingProxyType` freezing has no Rust analogue — owned `BTreeMap` values replace it.
//!
//! `McpServerConfig::instance` is the one deliberate structural deviation: an opaque in-process
//! server handle cannot be serialized, so it is skipped on the wire (and in JSON Schema),
//! `Debug` prints `<opaque instance>`, and equality only compares presence.
//!
//! Constructors do not validate implicitly; decode paths and callers that need the Python
//! `__post_init__` invariants call [`Validate::validate`].

use std::any::Any;
use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::errors::{HarnessError, HarnessResult};
use crate::json::{JsonObject, JsonValue};
use crate::validation::{require_non_empty, Validate};

/// Provider-neutral definition of a tool exposed by `OpenJiuwen`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub struct ToolDefinition {
    /// Tool name exposed to the provider.
    pub name: String,
    /// Human-readable description; may be empty.
    pub description: String,
    /// JSON Schema of the tool arguments.
    pub input_schema: JsonObject,
}

impl Validate for ToolDefinition {
    fn validate(&self) -> HarnessResult<()> {
        require_non_empty("tool definition name", &self.name)
    }
}

/// One native tool invocation requested by an external harness.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub struct ToolInvocation {
    /// Provider-assigned identifier of this call.
    pub call_id: String,
    /// Name of the tool the provider wants to run.
    pub name: String,
    /// JSON arguments of the call.
    pub arguments: JsonObject,
}

impl Validate for ToolInvocation {
    fn validate(&self) -> HarnessResult<()> {
        if self.call_id.is_empty() || self.name.is_empty() {
            return Err(HarnessError::validation(
                "tool invocation call_id and name must not be empty",
            ));
        }
        Ok(())
    }
}

/// Provider-neutral result of a native tool invocation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub struct ToolExecutionResult {
    /// Result content in whatever JSON shape the tool produced.
    pub content: JsonValue,
    /// Whether `content` describes a tool failure (Python default `false`).
    #[serde(default)]
    pub is_error: bool,
}

impl Validate for ToolExecutionResult {
    /// Returns `Ok(())`: Python declares no `__post_init__` rules for this type.
    fn validate(&self) -> HarnessResult<()> {
        Ok(())
    }
}

/// Transport used to expose an MCP server to an external harness.
///
/// Wire values are `stdio`, `http`, and `in_process`.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    PartialOrd,
    Ord,
    Serialize,
    Deserialize,
    schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum McpTransport {
    /// The harness spawns the server as a child process.
    Stdio,
    /// The harness talks to the server over HTTP.
    Http,
    /// The server is an SDK object living in this process.
    InProcess,
}

/// Opaque handle to an in-process MCP server instance.
///
/// The handle owns an `Arc<dyn Any + Send + Sync>` so the host can hand a live server object to a
/// harness in the same process. It is not serializable, [`fmt::Debug`] prints exactly
/// `<opaque instance>`, and [`PartialEq`] only records *presence*: two handles always compare
/// equal, which is the documented deviation from Python's identity (`is`) comparison.
#[derive(Clone)]
pub struct McpInstance(Arc<dyn Any + Send + Sync>);

impl McpInstance {
    /// Wraps `instance` in an opaque handle.
    #[must_use]
    pub fn new<T: Any + Send + Sync>(instance: T) -> Self {
        Self(Arc::new(instance))
    }

    /// Returns a reference to the wrapped instance when it is of type `T`.
    #[must_use]
    pub fn downcast_ref<T: Any + Send + Sync>(&self) -> Option<&T> {
        self.0.downcast_ref::<T>()
    }
}

impl fmt::Debug for McpInstance {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("<opaque instance>")
    }
}

impl PartialEq for McpInstance {
    /// Compares presence only: every handle is equal to every other handle.
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

/// Provider-neutral MCP server configuration.
///
/// Exactly one transport-specific target is required: `command` for [`McpTransport::Stdio`], `url`
/// for [`McpTransport::Http`], or `instance` for [`McpTransport::InProcess`]; `env` is only valid
/// for stdio and `headers` only for HTTP.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub struct McpServerConfig {
    /// Server name exposed to the harness.
    pub name: String,
    /// Transport used to reach the server.
    pub transport: McpTransport,
    /// Command and arguments of a stdio server; defaults to an empty list.
    #[serde(default)]
    pub command: Vec<String>,
    /// Endpoint of an HTTP server, when `transport` is [`McpTransport::Http`].
    #[serde(default)]
    pub url: Option<String>,
    /// In-process server handle, when `transport` is [`McpTransport::InProcess`].
    ///
    /// Skipped on the wire and in the JSON Schema: a live server object is not serializable.
    #[serde(skip)]
    #[schemars(skip)]
    pub instance: Option<McpInstance>,
    /// Environment variables passed to a stdio server; defaults to an empty map.
    #[serde(default)]
    pub env: BTreeMap<String, String>,
    /// HTTP headers sent to an HTTP server; defaults to an empty map.
    #[serde(default)]
    pub headers: BTreeMap<String, String>,
}

impl Validate for McpServerConfig {
    fn validate(&self) -> HarnessResult<()> {
        require_non_empty("MCP server name", &self.name)?;
        match self.transport {
            McpTransport::Stdio => {
                if self.command.is_empty() {
                    return Err(HarnessError::validation(
                        "stdio MCP server requires a non-empty command",
                    ));
                }
            }
            McpTransport::Http => {
                if self.url.is_none() {
                    return Err(HarnessError::validation("HTTP MCP server requires url"));
                }
            }
            McpTransport::InProcess => {
                if self.instance.is_none() {
                    return Err(HarnessError::validation(
                        "in-process MCP server requires instance",
                    ));
                }
            }
        }
        if self.transport != McpTransport::Stdio && !self.env.is_empty() {
            return Err(HarnessError::validation(
                "MCP env is only valid for stdio transport",
            ));
        }
        if self.transport != McpTransport::Http && !self.headers.is_empty() {
            return Err(HarnessError::validation(
                "MCP headers are only valid for HTTP transport",
            ));
        }
        let target_count = usize::from(!self.command.is_empty())
            + usize::from(self.url.is_some())
            + usize::from(self.instance.is_some());
        if target_count != 1 {
            return Err(HarnessError::validation(
                "MCP server requires exactly one of command, url, or instance",
            ));
        }
        Ok(())
    }
}

/// Native SDK tool surface supplied by the `OpenJiuwen` host.
#[async_trait]
pub trait ToolGateway: Send + Sync {
    /// Returns the tools visible to this agent.
    ///
    /// # Errors
    ///
    /// Returns [`HarnessError`] when the host cannot enumerate its tool surface.
    async fn definitions(&self) -> HarnessResult<Vec<ToolDefinition>>;

    /// Executes one visible tool under the host's permission policy.
    ///
    /// # Errors
    ///
    /// Returns [`HarnessError`] when the invocation is rejected by policy or the tool fails
    /// outside its own result channel.
    async fn invoke(&self, invocation: &ToolInvocation) -> HarnessResult<ToolExecutionResult>;
}

#[cfg(test)]
mod tests {
    // Tests assert by panicking on failure, which is the intent of a test.
    #![allow(clippy::expect_used, clippy::unwrap_used)]

    use std::collections::BTreeMap;

    use serde_json::json;

    use super::{
        McpInstance, McpServerConfig, McpTransport, ToolDefinition, ToolExecutionResult,
        ToolInvocation,
    };
    use crate::{
        empty_json_object, from_str_validated, json_object_from_value, HarnessErrorKind, Validate,
    };

    fn stdio_config() -> McpServerConfig {
        McpServerConfig {
            name: "files".to_owned(),
            transport: McpTransport::Stdio,
            command: vec![
                "mcp-files".to_owned(),
                "--root".to_owned(),
                "/tmp".to_owned(),
            ],
            url: None,
            instance: None,
            env: BTreeMap::new(),
            headers: BTreeMap::new(),
        }
    }

    fn http_config() -> McpServerConfig {
        McpServerConfig {
            name: "search".to_owned(),
            transport: McpTransport::Http,
            command: Vec::new(),
            url: Some("https://mcp.example.test".to_owned()),
            instance: None,
            env: BTreeMap::new(),
            headers: BTreeMap::new(),
        }
    }

    fn in_process_config() -> McpServerConfig {
        McpServerConfig {
            name: "local".to_owned(),
            transport: McpTransport::InProcess,
            command: Vec::new(),
            url: None,
            instance: Some(McpInstance::new(42_u32)),
            env: BTreeMap::new(),
            headers: BTreeMap::new(),
        }
    }

    #[test]
    fn tool_definition_and_invocation_validation() {
        let definition = ToolDefinition {
            name: "read_file".to_owned(),
            description: String::new(),
            input_schema: empty_json_object(),
        };
        definition
            .validate()
            .expect("an empty description is allowed");

        let mut blank = definition.clone();
        blank.name = String::new();
        assert_eq!(
            blank.validate().expect_err("empty name").message(),
            "tool definition name must not be empty"
        );

        let invocation = ToolInvocation {
            call_id: "call-1".to_owned(),
            name: "read_file".to_owned(),
            arguments: empty_json_object(),
        };
        invocation.validate().expect("valid invocation");

        let mut blank = invocation.clone();
        blank.call_id = String::new();
        assert_eq!(
            blank.validate().expect_err("empty call id").message(),
            "tool invocation call_id and name must not be empty"
        );

        blank.call_id = "call-2".to_owned();
        blank.name = String::new();
        assert_eq!(
            blank.validate().expect_err("empty name").message(),
            "tool invocation call_id and name must not be empty"
        );
    }

    #[test]
    fn tool_execution_result_defaults_and_round_trip() {
        let result: ToolExecutionResult =
            from_str_validated(r#"{"content": {"ok": true}}"#).expect("valid result");
        assert!(!result.is_error);
        result.validate().expect("no invariants");

        let encoded = serde_json::to_value(&result).expect("serializes");
        assert_eq!(encoded, json!({"content": {"ok": true}, "is_error": false}));
        let decoded: ToolExecutionResult = serde_json::from_value(encoded).expect("round trip");
        assert_eq!(decoded, result);
    }

    #[test]
    fn tool_definition_decodes_required_fields() {
        let error = from_str_validated::<ToolDefinition>(r#"{"name": "read_file"}"#)
            .expect_err("missing description and schema");
        assert_eq!(error.kind(), HarnessErrorKind::Protocol);

        let decoded: ToolDefinition = from_str_validated(
            r#"{"name": "read_file", "description": "", "input_schema": {"type": "object"}}"#,
        )
        .expect("valid definition");
        assert_eq!(
            decoded.input_schema,
            json_object_from_value(json!({"type": "object"})).expect("object")
        );
    }

    #[test]
    fn mcp_transport_wire_values() {
        for (transport, wire) in [
            (McpTransport::Stdio, "stdio"),
            (McpTransport::Http, "http"),
            (McpTransport::InProcess, "in_process"),
        ] {
            assert_eq!(
                serde_json::to_value(transport).expect("serializes"),
                json!(wire)
            );
            let decoded: McpTransport = serde_json::from_value(json!(wire)).expect("decodes");
            assert_eq!(decoded, transport);
        }
    }

    #[test]
    fn mcp_config_accepts_one_target_per_transport() {
        stdio_config().validate().expect("stdio with command");
        http_config().validate().expect("http with url");
        in_process_config()
            .validate()
            .expect("in-process with instance");
    }

    #[test]
    fn mcp_config_rejects_missing_targets_in_order() {
        let mut config = stdio_config();
        config.name = String::new();
        assert_eq!(
            config.validate().expect_err("empty name").message(),
            "MCP server name must not be empty"
        );

        let mut config = stdio_config();
        config.command = Vec::new();
        assert_eq!(
            config.validate().expect_err("missing command").message(),
            "stdio MCP server requires a non-empty command"
        );

        let mut config = http_config();
        config.url = None;
        assert_eq!(
            config.validate().expect_err("missing url").message(),
            "HTTP MCP server requires url"
        );

        let mut config = in_process_config();
        config.instance = None;
        assert_eq!(
            config.validate().expect_err("missing instance").message(),
            "in-process MCP server requires instance"
        );
    }

    #[test]
    fn mcp_config_rejects_misplaced_env_and_headers() {
        let mut config = http_config();
        config.env = BTreeMap::from([("TOKEN".to_owned(), "x".to_owned())]);
        assert_eq!(
            config.validate().expect_err("env on http").message(),
            "MCP env is only valid for stdio transport"
        );

        let mut config = in_process_config();
        config.headers = BTreeMap::from([("Authorization".to_owned(), "x".to_owned())]);
        assert_eq!(
            config
                .validate()
                .expect_err("headers on in-process")
                .message(),
            "MCP headers are only valid for HTTP transport"
        );
    }

    #[test]
    fn mcp_config_requires_exactly_one_target() {
        let mut config = stdio_config();
        config.url = Some("https://mcp.example.test".to_owned());
        config.transport = McpTransport::Http;
        assert_eq!(
            config.validate().expect_err("two targets").message(),
            "MCP server requires exactly one of command, url, or instance"
        );

        let mut config = in_process_config();
        config.command = vec!["mcp-files".to_owned()];
        assert_eq!(
            config.validate().expect_err("two targets").message(),
            "MCP server requires exactly one of command, url, or instance"
        );

        // A zero-target config can never reach the count check: every transport rejects its own
        // missing target first (Python behaves identically).
        let no_target = McpServerConfig {
            name: "none".to_owned(),
            transport: McpTransport::Http,
            command: Vec::new(),
            url: None,
            instance: None,
            env: BTreeMap::new(),
            headers: BTreeMap::new(),
        };
        assert_eq!(
            no_target.validate().expect_err("no target").message(),
            "HTTP MCP server requires url"
        );
    }

    #[test]
    fn mcp_instance_downcasts_and_hides_its_debug_repr() {
        let instance = McpInstance::new(7_u32);
        assert_eq!(instance.downcast_ref::<u32>(), Some(&7));
        assert!(instance.downcast_ref::<String>().is_none());
        assert_eq!(format!("{instance:?}"), "<opaque instance>");

        let clone = instance.clone();
        assert_eq!(clone, instance, "equality only compares presence");

        let other = McpInstance::new("text");
        assert_eq!(other, instance);
        assert_eq!(
            format!("{:?}", in_process_config().instance),
            "Some(<opaque instance>)"
        );
    }

    #[test]
    fn mcp_config_skips_instance_on_the_wire() {
        let encoded = serde_json::to_value(in_process_config()).expect("serializes");
        let keys: Vec<&str> = encoded
            .as_object()
            .expect("object")
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(
            keys,
            ["name", "transport", "command", "url", "env", "headers"]
        );
        assert_eq!(encoded["transport"], json!("in_process"));

        let error = from_str_validated::<McpServerConfig>(
            r#"{"name": "local", "transport": "in_process", "instance": null}"#,
        )
        .expect_err("instance is required for in-process transport");
        assert_eq!(error.kind(), HarnessErrorKind::Validation);

        let decoded: McpServerConfig = from_str_validated(
            r#"{"name": "files", "transport": "stdio", "command": ["mcp-files"]}"#,
        )
        .expect("valid stdio config");
        assert_eq!(decoded.command, ["mcp-files"]);
        assert!(decoded.instance.is_none());
        assert_eq!(decoded.env, BTreeMap::new());
        assert_eq!(decoded.headers, BTreeMap::new());

        let schema = serde_json::to_value(schemars::schema_for!(McpServerConfig))
            .expect("schema serializes");
        assert!(schema["properties"].get("instance").is_none());
    }

    #[test]
    fn mcp_config_round_trips_without_the_instance() {
        let config = http_config();
        let encoded = serde_json::to_string(&config).expect("serializes");
        let decoded: McpServerConfig = from_str_validated(&encoded).expect("round trip");
        assert_eq!(decoded, config);

        let mut with_headers = stdio_config();
        with_headers.env = BTreeMap::from([("TOKEN".to_owned(), "secret".to_owned())]);
        with_headers.validate().expect("stdio may carry env");
        let decoded: McpServerConfig =
            from_str_validated(&serde_json::to_string(&with_headers).expect("serializes"))
                .expect("round trip");
        assert_eq!(
            decoded.env,
            BTreeMap::from([("TOKEN".to_owned(), "secret".to_owned())])
        );
        assert_eq!(BTreeMap::<String, String>::new(), decoded.headers);
    }
}
