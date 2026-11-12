//! The server configuration, the executor SPI, and the per-turn execution
//! context.

use openjiuwen_common::error::Result;
use serde_json::{Map, Value};

use crate::a2a::types::{self, Message};

/// The default JSON-RPC mount path, trailing slash included.
pub const DEFAULT_RPC_PATH: &str = "/a2a/jsonrpc/";

/// The default agent-card path.
pub const DEFAULT_CARD_PATH: &str = "/.well-known/agent-card.json";

/// The default protocol version advertised on the card (`1.0`, the A2A v1 wire).
pub const DEFAULT_PROTOCOL_VERSION: &str = "1.0";

/// The default bind host (loopback, like Python's `start(host="127.0.0.1")`).
pub const DEFAULT_HOST: &str = "127.0.0.1";

/// The default bind port (Python's `start(port=8000)`).
pub const DEFAULT_PORT: u16 = 8000;

/// The bind address, mount paths, and card a [`super::A2aServer`] publishes.
///
/// Python parity: the `A2AServer(...)` / `start(host, port)` arguments of
/// `a2a_server.py`, collapsed into one value. Defaults match Python:
/// `rpc_path = /a2a/jsonrpc/`, `card_path = /.well-known/agent-card.json`,
/// `protocol_version = 1.0`, `host = 127.0.0.1`, `port = 8000`.
#[derive(Debug, Clone, PartialEq)]
pub struct A2aServerConfig {
    /// The card served at [`Self::card_path`].
    pub agent_card: types::AgentCard,
    /// The JSON-RPC URL to publish on the card, when it differs from the bind
    /// address (Python's `interface_url`). Passed through untouched; the
    /// response body of the card route is the configured card itself.
    pub interface_url: Option<String>,
    /// The path `JSON-RPC` requests are `POST`ed to, mounted literally.
    pub rpc_path: String,
    /// The path the agent card is served at.
    pub card_path: String,
    /// The protocol version this server speaks.
    pub protocol_version: String,
    /// The host to bind (`serve`/`serve_in_background`).
    pub host: String,
    /// The port to bind; `0` picks an ephemeral port at bind time.
    pub port: u16,
}

impl Default for A2aServerConfig {
    fn default() -> Self {
        Self {
            agent_card: types::AgentCard::default(),
            interface_url: None,
            rpc_path: DEFAULT_RPC_PATH.to_owned(),
            card_path: DEFAULT_CARD_PATH.to_owned(),
            protocol_version: DEFAULT_PROTOCOL_VERSION.to_owned(),
            host: DEFAULT_HOST.to_owned(),
            port: DEFAULT_PORT,
        }
    }
}

impl A2aServerConfig {
    /// A configuration for `agent_card` with every other field defaulted.
    #[must_use]
    pub fn new(agent_card: types::AgentCard) -> Self {
        Self {
            agent_card,
            ..Self::default()
        }
    }

    /// Sets the JSON-RPC URL published on the card.
    #[must_use]
    pub fn with_interface_url(mut self, url: impl Into<String>) -> Self {
        self.interface_url = Some(url.into());
        self
    }

    /// Sets the bind host and port.
    #[must_use]
    pub fn with_bind(mut self, host: impl Into<String>, port: u16) -> Self {
        self.host = host.into();
        self.port = port;
        self
    }

    /// Sets the `JSON-RPC` mount path.
    #[must_use]
    pub fn with_rpc_path(mut self, path: impl Into<String>) -> Self {
        self.rpc_path = path.into();
        self
    }
}

/// One turn handed to an [`A2aExecutor`].
///
/// Python parity: the `RequestContext` members `A2AAgentExecutor` reads
/// (`task_id`, `context_id`, `message`) plus the openJiuwen payload metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct A2aExecutionContext {
    /// The task the turn belongs to (from the request, or freshly generated).
    pub task_id: String,
    /// The conversation the turn belongs to (`sessionId` when the request had
    /// one, otherwise freshly generated).
    pub context_id: String,
    /// The request message exactly as received (or synthesized from the payload).
    pub message: Message,
    /// The openJiuwen payload without its reserved `query`/`task_id`/
    /// `sessionId` keys.
    pub metadata: Map<String, Value>,
}

/// The application hook that produces a task.
///
/// Python parity: `A2AAgentExecutor`, minus the streaming/invoke split — an
/// executor publishes through the [`super::EventSink`] and the server decides
/// whether the events are streamed or only stored.
#[async_trait::async_trait]
pub trait A2aExecutor: Send + Sync {
    /// Runs one turn, publishing statuses and artifacts through `sink`.
    ///
    /// The server publishes the `SUBMITTED` snapshot and the `WORKING` status
    /// before calling this, and the terminal status after it returns `Ok` (an
    /// executor may publish its own terminal status first, which suppresses the
    /// server's `COMPLETED`).
    ///
    /// # Errors
    ///
    /// Any error is logged, published as a `FAILED` status, and — for a
    /// non-streaming call — reported as a `-32603` `JSON-RPC` error envelope.
    async fn execute(&self, ctx: A2aExecutionContext, sink: &super::EventSink) -> Result<()>;

    /// Cancels one task.
    ///
    /// The default body returns `Ok(())`; the server publishes the `CANCELED`
    /// status itself, so the observable behavior matches Python's default
    /// "emit Canceled" without forcing every executor to drive the sink.
    ///
    /// # Errors
    ///
    /// Return an error when the cancellation could not be attempted. The server
    /// logs it and publishes `CANCELED` anyway, because cancellation is best
    /// effort (Python's `_cancel_quietly`).
    async fn cancel(&self, _ctx: A2aExecutionContext) -> Result<()> {
        Ok(())
    }
}
