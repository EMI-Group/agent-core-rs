//! `11_a2a_server_client` — an A2A server and an A2A client over loopback.
//!
//! Python counterparts: `examples/a2a/server_openjiuwen_a2a_for_sdk_client.py` (the
//! server half: an agent published over A2A) and
//! `examples/a2a/client_open_a2a_sdk_to_jiuwen_server.py` (the client half: an
//! `a2a-sdk` client calling it). Python runs the two in separate terminals; here
//! both halves live in one process on `127.0.0.1`, so a single run shows the
//! whole request/response round trip.
//!
//! The seam is three values plus one trait:
//!
//! - a card — [`openjiuwen::extensions::a2a::card::AgentCardAdapter::to_a2a`]
//!   converts a kernel `AgentCard` into the A2A card a server advertises;
//! - a [`A2aServerConfig`] — the card plus the bind address; port `0` asks the
//!   kernel for an ephemeral port at bind time;
//! - an [`A2aExecutor`] — the application hook. The server publishes the
//!   `SUBMITTED` snapshot and the `WORKING` status around it, the executor
//!   publishes artifacts and a terminal status through the [`EventSink`], and
//!   the server stores the resulting task;
//! - an [`A2aClient`] — posts JSON-RPC (`SendMessage`) to the server's
//!   `rpc_path` and folds the reply into one `AgentResult`.
//!
//! The lifecycle is [`A2aServerAdapter`] (Python's `A2AServerAdapter`):
//! `start()` binds and serves in the background, `stop()` shuts it down and
//! waits for the serving task. `A2aServer::serve_in_background` is the same
//! thing one layer down and takes a `tokio_util::sync::CancellationToken`; this
//! crate does not depend on `tokio-util`, so the adapter is the handle used here.
//!
//! Everything is loopback and offline — no model, no credential, no DNS — and
//! every await that could block is under a timeout. Requires the `a2a` feature
//! (`--features a2a`), which is how the example is declared in `Cargo.toml`.

#![allow(
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    // The executor body is synchronous on purpose: the trait method is `async`
    // because a real executor awaits its agent, and this echo has nothing to await.
    clippy::unused_async
)]

mod common;

use std::sync::Arc;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use async_trait::async_trait;
use openjiuwen::core::common::error::Result as SdkResult;
use openjiuwen::core::single_agent::AgentCard;
use openjiuwen::extensions::a2a::card::AgentCardAdapter;
use openjiuwen::extensions::a2a::client::{A2aClient, A2aClientConfig};
use openjiuwen::extensions::a2a::server::{
    A2aExecutionContext, A2aExecutor, A2aServer, A2aServerAdapter, A2aServerConfig, EventSink,
    DEFAULT_PROTOCOL_VERSION,
};
use openjiuwen::extensions::a2a::transform::AgentResult;
use openjiuwen::extensions::a2a::types::{Artifact, Message, Part};
use serde_json::json;

use common::banner;

/// Name the server publishes on its A2A card.
const AGENT_NAME: &str = "a2a-echo-agent";

/// Description the server publishes on its A2A card.
const AGENT_DESCRIPTION: &str = "Echoes one query back over the A2A JSON-RPC transport";

/// The transport binding written on the card; the server speaks only this one.
const JSONRPC_BINDING: &str = "JSONRPC";

/// The query the client sends (Python: `inputs={"query": ...}`).
const QUERY: &str = "hello a2a";

/// Bound for every await, so a stuck socket cannot hang the demo.
const STEP_TIMEOUT: Duration = Duration::from_secs(20);

/// The application hook: one query in, one artifact out.
struct EchoExecutor;

/// Joins the text parts of a request message — the only field this executor reads.
fn message_text(message: &Message) -> String {
    message
        .parts
        .iter()
        .filter_map(|part| part.text.as_deref())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Joins the text parts of every artifact in a result, for printing.
fn result_text(result: &AgentResult) -> String {
    result
        .artifacts
        .iter()
        .flat_map(|artifact| artifact.parts.iter())
        .filter_map(|part| part.text.as_deref())
        .collect::<Vec<_>>()
        .join(" ")
}

#[async_trait]
impl A2aExecutor for EchoExecutor {
    /// Answers the turn the server handed over.
    ///
    /// The server has already published `SUBMITTED` and `WORKING`; one terminal
    /// artifact and `sink.complete()` are all this executor has to add.
    async fn execute(&self, ctx: A2aExecutionContext, sink: &EventSink) -> SdkResult<()> {
        let query = message_text(&ctx.message);
        tracing::info!(
            task_id = %ctx.task_id,
            context_id = %ctx.context_id,
            %query,
            "the executor received a query"
        );
        // `last_chunk = true`: the artifact is complete in one chunk, so the sink
        // has nothing left to close afterwards.
        sink.artifact(
            Artifact {
                name: Some(String::from("response")),
                parts: vec![Part::text(format!("echo: {query}"))],
                ..Artifact::default()
            },
            true,
        );
        sink.complete();
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let _mode = common::setup()?;
    banner::title("11 a2a server + client");
    banner::note("offline: the server and the client are both in this process, on loopback");

    // 1. The card. `to_a2a` is Python's `A2AAgentCardAdapter.to_a2a_agent_card`:
    //    the name and description are copied, streaming is advertised, and the
    //    default input/output modes are the A2A text modes. No interface URL is
    //    published because the port is only known once the server has bound;
    //    the client below is pointed at the bound URL directly, which is what a
    //    discovery client would have read off the card.
    let kernel_card = AgentCard::new(AGENT_NAME).with_description(AGENT_DESCRIPTION);
    let card = AgentCardAdapter::to_a2a(
        &kernel_card,
        None,
        JSONRPC_BINDING,
        DEFAULT_PROTOCOL_VERSION,
    );
    banner::step(&format!("card: {} ({})", card.name, card.description));

    // 2. The server. `with_bind` on port 0 asks for an ephemeral loopback port;
    //    the A2A card must advertise no transport the server cannot speak, which
    //    is why `A2aServer::new` fails on a gRPC interface.
    let server = A2aServer::new(
        A2aServerConfig::new(card).with_bind("127.0.0.1", 0),
        Arc::new(EchoExecutor),
    )
    .context("the A2A card advertises a transport this server does not speak")?;
    // The mount path includes its trailing slash, so read it before the config
    // moves into the server.
    let rpc_path = server.config().rpc_path.clone();

    let mut adapter = A2aServerAdapter::new(server);
    tokio::time::timeout(STEP_TIMEOUT, adapter.start())
        .await
        .context("the a2a server did not start in time")?
        .context("the a2a server failed to bind 127.0.0.1")?;
    let addr = adapter
        .local_addr()
        .context("the a2a server reported no bound address")?;
    let rpc_url = format!("http://{addr}{rpc_path}");
    banner::step(&format!("a2a server listening on http://{addr}"));
    banner::note(&format!("json-rpc endpoint: {rpc_url}"));

    // 3. The client. `invoke` posts one `SendMessage` and folds the reply into a
    //    single `AgentResult`; `stream` would yield one frame per SSE event.
    let client = A2aClient::new(A2aClientConfig::new(rpc_url).with_timeout(STEP_TIMEOUT))
        .context("the a2a client requires a non-empty rpc_url")?;
    banner::step(&format!("client invoking with query: {QUERY:?}"));

    let result = tokio::time::timeout(STEP_TIMEOUT, client.invoke(json!({ "query": QUERY })))
        .await
        .context("the a2a client call timed out")?
        .context("the a2a server rejected the call")?;

    banner::note(&format!("status: {}", result.status.as_str()));
    // The task and context ids are minted per run, so they go to the log rather than
    // to stdout: the narrative stays byte-identical between two runs.
    tracing::info!(
        task_id = result.task_id.as_deref().unwrap_or_default(),
        context_id = result.session_id.as_deref().unwrap_or_default(),
        "the server answered with a task and a context"
    );

    let echoed = result_text(&result);
    if echoed.is_empty() {
        bail!("the a2a server answered without any artifact text");
    }
    banner::answer(&echoed);

    // 4. Shutdown: stops the serving task and waits for it.
    banner::step("stopping the a2a server");
    tokio::time::timeout(STEP_TIMEOUT, adapter.stop())
        .await
        .context("the a2a server did not stop in time")?
        .context("the a2a server did not stop cleanly")?;
    banner::note("the a2a server is stopped and its socket is closed");

    Ok(())
}
