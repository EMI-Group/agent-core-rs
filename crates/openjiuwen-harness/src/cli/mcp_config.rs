//! Discovery and merge of `mcp.json` server configurations.
//!
//! Python reads only `~/.openjiuwen/mcp.json` with a top-level `mcpServers`
//! map. The Rust CLI also reads `<workspace>/.openjiuwen/mcp.json` and merges
//! the two, with workspace entries overriding home entries by server name
//! (the precedence order comes from [`paths::mcp_config_paths`](crate::cli::paths::mcp_config_paths)).
//!
//! # Documented deviations
//!
//! - The workspace-level configuration file is a Rust addition.
//! - Python silently drops malformed input; this module returns a typed
//!   [`Error::Config`] that names the offending path.
//!
//! # Conversion rules
//!
//! - An entry with no `transport` defaults to [`DEFAULT_TRANSPORT`] (`stdio`).
//! - A `stdio` entry requires `command`; `command`, `args`, `env` and `cwd` are
//!   copied into [`McpServerConfig::params`] as a JSON object, omitting absent
//!   keys (and empty `args`/`env`).
//! - An `sse` or `streamable-http` entry requires `url`, which becomes
//!   [`McpServerConfig::server_path`]; `auth_headers` are copied.
//! - The map key becomes [`McpServerConfig::server_name`], and servers are
//!   emitted sorted by that name.
#![allow(clippy::module_name_repetitions)] // names mirror the phase plan's public API

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use openjiuwen_common::config::OpenJiuwenHome;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_foundation::tool::mcp::{McpClientType, McpServerConfig};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

/// Transport assumed when an entry does not name one.
pub const DEFAULT_TRANSPORT: &str = "stdio";

/// One server entry of an `mcp.json` document.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct McpServerSpec {
    /// Transport name: `stdio`, `sse`, `streamable_http` or `http`.
    #[serde(alias = "client_type")]
    pub transport: String,
    /// Executable of a `stdio` server.
    pub command: Option<String>,
    /// Arguments passed to a `stdio` server.
    pub args: Vec<String>,
    /// Extra environment variables of a `stdio` server.
    pub env: BTreeMap<String, String>,
    /// Working directory of a `stdio` server.
    pub cwd: Option<String>,
    /// Endpoint of an HTTP server.
    #[serde(alias = "server_path")]
    pub url: Option<String>,
    /// Headers sent with every HTTP request.
    #[serde(alias = "authHeaders")]
    pub auth_headers: BTreeMap<String, String>,
}

impl McpServerSpec {
    /// Converts one entry into a foundation server configuration.
    fn to_foundation(&self, name: &str) -> Result<McpServerConfig> {
        let client_type = self.client_type(name)?;
        match client_type {
            McpClientType::Stdio => self.stdio_server(name, client_type),
            McpClientType::Sse | McpClientType::StreamableHttp | McpClientType::OpenApi => {
                self.http_server(name, client_type)
            }
            McpClientType::Playwright => Err(Error::config(format!(
                "mcp server {name}: playwright transport is not supported"
            ))),
        }
    }

    /// Resolves the transport, defaulting to [`DEFAULT_TRANSPORT`].
    fn client_type(&self, name: &str) -> Result<McpClientType> {
        let transport = self.transport.trim();
        let transport = if transport.is_empty() {
            DEFAULT_TRANSPORT
        } else {
            transport
        };
        McpClientType::parse(transport).map_err(|_| {
            Error::config(format!(
                "mcp server {name}: unsupported transport `{transport}`"
            ))
        })
    }

    /// Builds a `stdio` server configuration.
    fn stdio_server(&self, name: &str, client_type: McpClientType) -> Result<McpServerConfig> {
        let command = self
            .command
            .as_deref()
            .map(str::trim)
            .filter(|command| !command.is_empty())
            .ok_or_else(|| {
                Error::config(format!(
                    "mcp server {name}: stdio transport requires `command`"
                ))
            })?;

        let mut params = Map::new();
        params.insert("command".to_owned(), Value::String(command.to_owned()));
        if !self.args.is_empty() {
            params.insert("args".to_owned(), json!(self.args));
        }
        if !self.env.is_empty() {
            params.insert("env".to_owned(), json!(self.env));
        }
        if let Some(cwd) = &self.cwd {
            params.insert("cwd".to_owned(), Value::String(cwd.clone()));
        }

        let mut server = McpServerConfig::stdio(name, command);
        server.client_type = client_type;
        server.params = Value::Object(params);
        Ok(server)
    }

    /// Builds an HTTP (`sse` or `streamable-http`) server configuration.
    fn http_server(&self, name: &str, client_type: McpClientType) -> Result<McpServerConfig> {
        let url = self
            .url
            .as_deref()
            .map(str::trim)
            .filter(|url| !url.is_empty())
            .ok_or_else(|| {
                Error::config(format!(
                    "mcp server {name}: {} transport requires `url`",
                    client_type.as_str()
                ))
            })?;

        let mut server = McpServerConfig::http(name, url);
        server.client_type = client_type;
        server.auth_headers = self.auth_headers.clone();
        Ok(server)
    }
}

/// A whole `mcp.json` document: a map of server name to entry.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct McpConfiguration {
    /// Server entries keyed by server name.
    #[serde(alias = "mcpServers")]
    pub mcp_servers: BTreeMap<String, McpServerSpec>,
}

impl McpConfiguration {
    /// Parses an `mcp.json` document.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] when the text is not a valid configuration
    /// document.
    pub fn parse(text: &str) -> Result<Self> {
        serde_json::from_str(text)
            .map_err(|err| Error::config(format!("invalid mcp configuration: {err}")))
    }

    /// Loads one configuration file, treating a missing file as `None`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] naming the path when the file cannot be read
    /// or does not contain a valid configuration document.
    pub fn load(path: &Path) -> Result<Option<Self>> {
        let text = match std::fs::read_to_string(path) {
            Ok(text) => text,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(err) => {
                return Err(Error::config(format!(
                    "cannot read mcp configuration at {}: {err}",
                    path.display()
                )));
            }
        };
        let config = serde_json::from_str(&text).map_err(|err| {
            Error::config(format!(
                "invalid mcp configuration at {}: {err}",
                path.display()
            ))
        })?;
        Ok(Some(config))
    }

    /// Returns whether the document declares no servers.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.mcp_servers.is_empty()
    }

    /// Converts every entry, sorted by server name, into a foundation config.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] naming the server when an entry has an
    /// unsupported transport, a `stdio` entry lacks `command`, or an HTTP entry
    /// lacks `url`.
    pub fn servers(&self) -> Result<Vec<McpServerConfig>> {
        self.mcp_servers
            .iter()
            .map(|(name, spec)| spec.to_foundation(name))
            .collect()
    }
}

/// Loads and merges every path in order (later paths override earlier ones by
/// server name), then converts the merged entries.
///
/// # Errors
///
/// Returns [`Error::Config`] when a file cannot be read or parsed, or when a
/// merged entry is invalid.
pub fn load_servers(paths: &[PathBuf]) -> Result<Vec<McpServerConfig>> {
    let mut merged: BTreeMap<String, McpServerSpec> = BTreeMap::new();
    for path in paths {
        if let Some(config) = McpConfiguration::load(path)? {
            merged.extend(config.mcp_servers);
        }
    }
    McpConfiguration {
        mcp_servers: merged,
    }
    .servers()
}

/// Loads the CLI's default configuration paths for `home` and `workspace`.
///
/// `OPENJIUWEN_MCP_CONFIG` pins the list to exactly one file; see
/// [`paths::mcp_config_paths`](crate::cli::paths::mcp_config_paths).
///
/// # Errors
///
/// Returns [`Error::Config`] under the same conditions as [`load_servers`].
pub fn load_for(
    home: &OpenJiuwenHome,
    workspace: &Path,
    env: &dyn super::config::EnvSource,
) -> Result<Vec<McpServerConfig>> {
    let paths = super::paths::mcp_config_paths(home, workspace, env);
    load_servers(&paths)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use crate::cli::paths::OPENJIUWEN_MCP_CONFIG;
    use crate::cli::MapEnv;
    use tempfile::TempDir;

    const THREE_SERVERS: &str = r#"{
        "mcpServers": {
            "filesystem": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem"],
                "env": {"ROOT": "/tmp"},
                "cwd": "/tmp/work"
            },
            "events": {
                "transport": "sse",
                "url": "https://example.test/sse",
                "auth_headers": {"Authorization": "Bearer token"}
            },
            "remote": {
                "transport": "streamable-http",
                "url": "https://example.test/mcp"
            }
        }
    }"#;

    fn fixture() -> (TempDir, PathBuf, OpenJiuwenHome) {
        let temp = TempDir::new().unwrap();
        let workspace = temp.path().join("workspace");
        let home_root = temp.path().join("home");
        std::fs::create_dir_all(&workspace).unwrap();
        std::fs::create_dir_all(&home_root).unwrap();
        (temp, workspace, OpenJiuwenHome::from_root(home_root))
    }

    #[test]
    fn three_server_document_converts() {
        let config = McpConfiguration::parse(THREE_SERVERS).unwrap();
        assert!(!config.is_empty());
        let servers = config.servers().unwrap();
        let names: Vec<_> = servers
            .iter()
            .map(|server| server.server_name.as_str())
            .collect();
        assert_eq!(names, vec!["events", "filesystem", "remote"]);

        let stdio = &servers[1];
        assert_eq!(stdio.client_type, McpClientType::Stdio);
        assert_eq!(stdio.server_path, "npx");
        assert_eq!(stdio.params["command"], json!("npx"));
        assert_eq!(
            stdio.params["args"],
            json!(["-y", "@modelcontextprotocol/server-filesystem"])
        );
        assert_eq!(stdio.params["env"], json!({"ROOT": "/tmp"}));
        assert_eq!(stdio.params["cwd"], json!("/tmp/work"));

        let sse = &servers[0];
        assert_eq!(sse.client_type, McpClientType::Sse);
        assert_eq!(sse.server_path, "https://example.test/sse");
        assert_eq!(
            sse.auth_headers.get("Authorization").map(String::as_str),
            Some("Bearer token")
        );

        let http = &servers[2];
        assert_eq!(http.client_type, McpClientType::StreamableHttp);
        assert_eq!(http.server_path, "https://example.test/mcp");
    }

    #[test]
    fn camel_case_and_field_aliases_are_accepted() {
        let text = r#"{
            "mcpServers": {
                "remote": {"client_type": "sse", "server_path": "https://example.test/sse"},
                "local": {"client_type": "stdio", "command": "tool"}
            }
        }"#;
        let servers = McpConfiguration::parse(text).unwrap().servers().unwrap();
        let names: Vec<_> = servers
            .iter()
            .map(|server| server.server_name.as_str())
            .collect();
        assert_eq!(names, vec!["local", "remote"]);
        assert_eq!(servers[0].client_type, McpClientType::Stdio);
        assert_eq!(servers[1].client_type, McpClientType::Sse);
        assert_eq!(servers[1].server_path, "https://example.test/sse");
    }

    #[test]
    fn malformed_json_names_the_path() {
        let (temp, _workspace, _home) = fixture();
        let path = temp.path().join("mcp.json");
        std::fs::write(&path, "{not json").unwrap();

        let err = McpConfiguration::load(&path).unwrap_err();
        assert!(matches!(err, Error::Config(_)));
        let message = err.to_string();
        assert!(message.contains("mcp.json"), "{message}");

        let err = McpConfiguration::parse("{not json").unwrap_err();
        assert!(matches!(err, Error::Config(_)));
    }

    #[test]
    fn missing_file_is_none_and_loads_nothing() {
        let (temp, _workspace, _home) = fixture();
        let path = temp.path().join("absent.json");
        assert!(McpConfiguration::load(&path).unwrap().is_none());
        assert_eq!(
            load_servers(&[path]).unwrap(),
            Vec::<McpServerConfig>::new()
        );
    }

    #[test]
    fn paths_merge_with_workspace_winning() {
        let (temp, _workspace, _home) = fixture();
        let home_path = temp.path().join("home-mcp.json");
        let workspace_path = temp.path().join("workspace-mcp.json");
        std::fs::write(
            &home_path,
            r#"{"mcpServers": {
                "shared": {"command": "home-command"},
                "home-only": {"command": "home-only-command"}
            }}"#,
        )
        .unwrap();
        std::fs::write(
            &workspace_path,
            r#"{"mcpServers": {
                "shared": {"command": "workspace-command"},
                "workspace-only": {"command": "workspace-only-command"}
            }}"#,
        )
        .unwrap();

        let servers = load_servers(&[home_path, workspace_path]).unwrap();
        let names: Vec<_> = servers
            .iter()
            .map(|server| server.server_name.as_str())
            .collect();
        assert_eq!(names, vec!["home-only", "shared", "workspace-only"]);
        let shared = servers
            .iter()
            .find(|server| server.server_name == "shared")
            .unwrap();
        assert_eq!(shared.server_path, "workspace-command");
    }

    #[test]
    fn stdio_without_command_is_a_config_error() {
        let config = McpConfiguration::parse(r#"{"mcpServers": {"broken": {}}}"#).unwrap();
        let err = config.servers().unwrap_err();
        assert!(matches!(err, Error::Config(_)));
        assert!(err.to_string().contains("broken"), "{err}");
    }

    #[test]
    fn http_without_url_is_a_config_error() {
        let config =
            McpConfiguration::parse(r#"{"mcpServers": {"broken": {"transport": "sse"}}}"#).unwrap();
        let err = config.servers().unwrap_err();
        assert!(matches!(err, Error::Config(_)));
        assert!(err.to_string().contains("broken"), "{err}");
    }

    #[test]
    fn unknown_transport_is_a_config_error() {
        let config = McpConfiguration::parse(
            r#"{"mcpServers": {"broken": {"transport": "carrier-pigeon", "command": "x"}}}"#,
        )
        .unwrap();
        let err = config.servers().unwrap_err();
        assert!(matches!(err, Error::Config(_)));
        assert!(err.to_string().contains("carrier-pigeon"), "{err}");
    }

    #[test]
    fn load_for_uses_the_env_override() {
        let (temp, workspace, home) = fixture();
        let path = temp.path().join("pinned.json");
        std::fs::write(
            &path,
            r#"{"mcpServers": {"pinned": {"command": "pinned-command"}}}"#,
        )
        .unwrap();
        let env = MapEnv::from_pairs([(OPENJIUWEN_MCP_CONFIG, path.to_str().unwrap().to_owned())]);

        let servers = load_for(&home, &workspace, &env).unwrap();
        assert_eq!(servers.len(), 1);
        assert_eq!(servers[0].server_name, "pinned");
        assert_eq!(servers[0].server_path, "pinned-command");
    }

    #[test]
    fn load_for_merges_default_paths() {
        let (_temp, workspace, home) = fixture();
        std::fs::write(
            home.join(crate::cli::paths::MCP_CONFIG_FILENAME),
            r#"{"mcpServers": {"shared": {"command": "home-command"}}}"#,
        )
        .unwrap();
        let project_dir = workspace.join(crate::cli::paths::MCP_CONFIG_DIRNAME);
        std::fs::create_dir_all(&project_dir).unwrap();
        std::fs::write(
            project_dir.join(crate::cli::paths::MCP_CONFIG_FILENAME),
            r#"{"mcpServers": {"shared": {"command": "workspace-command"}}}"#,
        )
        .unwrap();

        let servers = load_for(&home, &workspace, &MapEnv::new()).unwrap();
        assert_eq!(servers.len(), 1);
        assert_eq!(servers[0].server_path, "workspace-command");
    }
}
