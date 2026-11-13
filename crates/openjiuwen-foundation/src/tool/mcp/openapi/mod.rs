//! The `OpenAPI`→tools MCP transport.
//!
//! Ports `openjiuwen/core/foundation/tool/mcp/client/openapi_client.py`: one
//! MCP tool per operation of an `OpenAPI` document, where an operation is a
//! path-item entry (`GET`, `POST`, …) with its parameters and JSON request body.
//!
//! The document is loaded lazily on the first operation, from a filesystem path
//! (`.json`, `.yaml`, `.yml`, or sniffed) or from an `http(s)` URL, and is then
//! cached until [`McpClient::disconnect`]. Calls are plain HTTP requests against
//! `servers[0].url` (or the `params.base_url` override); nothing here speaks
//! JSON-RPC, so [`McpClient::list_resources`], [`McpClient::read_resource`],
//! [`McpClient::take_notifications`] and [`McpClient::supports_server_stream`]
//! keep their trait defaults.
//!
//! The Playwright variant of the Python transport stays deferred: it needs a
//! browser runtime that this workspace does not ship, so
//! [`connect_mcp_server`](super::client::connect_mcp_server) returns
//! [`Error::Unsupported`] for it.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::http::HttpClient;
use serde_json::Value;
use tokio::sync::Mutex;

pub(crate) mod call;
pub(crate) mod spec;

pub use spec::{OpenApiOperation, OpenApiParameter, OpenApiSpec};

use super::client::{McpClient, McpRemoteTool, McpServerConfig};
use super::result::McpToolResult;

/// An [`McpClient`] backed by an `OpenAPI` document instead of an MCP server.
///
/// The spec is loaded on [`McpClient::connect`] (called implicitly by every
/// operation) and cached; `disconnect` drops it so the next operation reloads
/// the document.
pub struct OpenApiClient {
    config: McpServerConfig,
    http: HttpClient,
    spec: parking_lot::Mutex<Option<Arc<OpenApiSpec>>>,
    connect_lock: Mutex<()>,
}

impl OpenApiClient {
    /// Validates `config` and builds a client with the default HTTP client.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when `server_path` is empty and
    /// [`Error::Config`] when the default HTTP client cannot be built.
    pub fn new(config: &McpServerConfig) -> Result<Self> {
        let http = HttpClient::with_defaults()?;
        Self::with_http_client(config, http)
    }

    /// Validates `config` and builds a client that performs I/O through `http`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when `server_path` is empty.
    pub fn with_http_client(config: &McpServerConfig, http: HttpClient) -> Result<Self> {
        if config.server_path.trim().is_empty() {
            return Err(Error::validation(
                "MCP openapi transport requires `server_path` (a spec file path or URL)",
            ));
        }
        Ok(Self {
            config: config.clone(),
            http,
            spec: parking_lot::Mutex::new(None),
            connect_lock: Mutex::new(()),
        })
    }

    /// Returns the configuration this client was built from.
    #[must_use]
    pub fn config(&self) -> &McpServerConfig {
        &self.config
    }

    /// Returns the loaded spec, if [`McpClient::connect`] already ran.
    #[must_use]
    pub fn spec(&self) -> Option<Arc<OpenApiSpec>> {
        self.spec.lock().clone()
    }

    /// Returns the loaded operations, if any.
    fn operations(&self) -> Vec<OpenApiOperation> {
        self.spec
            .lock()
            .as_ref()
            .map(|spec| spec.operations().to_vec())
            .unwrap_or_default()
    }

    /// Loads the document once and caches it.
    async fn ensure_spec(&self) -> Result<Arc<OpenApiSpec>> {
        if let Some(spec) = self.spec() {
            return Ok(spec);
        }
        let _guard = self.connect_lock.lock().await;
        if let Some(spec) = self.spec() {
            return Ok(spec);
        }
        let base_override = self.config.params.get("base_url").and_then(Value::as_str);
        let loaded =
            OpenApiSpec::load_with_base(&self.http, &self.config.server_path, base_override)
                .await?;
        tracing::debug!(
            server_name = %self.config.server_name,
            server_id = %self.config.server_id,
            base_url = loaded.base_url(),
            operations = loaded.operations().len(),
            "loaded OpenAPI spec"
        );
        let spec = Arc::new(loaded);
        *self.spec.lock() = Some(Arc::clone(&spec));
        Ok(spec)
    }
}

#[async_trait]
impl McpClient for OpenApiClient {
    fn server_name(&self) -> &str {
        &self.config.server_name
    }

    fn server_id(&self) -> &str {
        &self.config.server_id
    }

    fn include_image_content(&self) -> bool {
        self.config.include_image_content
    }

    async fn connect(&self) -> Result<()> {
        self.ensure_spec().await.map(|_| ())
    }

    async fn disconnect(&self) -> Result<()> {
        let dropped = self.spec.lock().take();
        if dropped.is_some() {
            tracing::debug!(
                server_name = %self.config.server_name,
                "dropped OpenAPI spec"
            );
        }
        Ok(())
    }

    async fn is_connected(&self) -> bool {
        self.spec.lock().is_some()
    }

    async fn list_tools(&self) -> Result<Vec<McpRemoteTool>> {
        self.ensure_spec().await?;
        Ok(self
            .operations()
            .into_iter()
            .map(|operation| {
                McpRemoteTool::new(
                    operation.name,
                    operation.description,
                    operation.input_schema,
                )
            })
            .collect())
    }

    async fn call_tool_with(
        &self,
        tool_name: &str,
        arguments: Value,
        timeout: Option<Duration>,
        include_image_content: bool,
    ) -> Result<McpToolResult> {
        // OpenAPI responses are plain JSON: there are no image blocks to forward.
        let _ = include_image_content;
        let spec = self.ensure_spec().await?;
        let Some(operation) = spec
            .operations()
            .iter()
            .find(|operation| operation.name == tool_name)
        else {
            return Err(Error::mcp(format!("unknown OpenAPI tool `{tool_name}`")));
        };
        let args = arguments.as_object().cloned().unwrap_or_default();
        call::execute(
            &self.http,
            operation,
            spec.base_url(),
            &args,
            &self.config,
            timeout,
        )
        .await
    }
}
