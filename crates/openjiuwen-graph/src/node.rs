//! Node abstraction: the [`Node`] trait, its outcome, per-node options, the
//! invocation context, and closure adapters.
//!
//! A node is one unit of work in a superstep. It receives a mutable clone of
//! the [`GraphState`] and a [`GraphContext`] carrying the session, its own id,
//! the superstep number, and the run's cancellation token.

use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use openjiuwen_common::error::Result;
use openjiuwen_session::Session;

use crate::state::GraphState;

/// Per-node execution options.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct NodeOptions {
    /// Per-attempt timeout in milliseconds, unless the run or the graph
    /// overrides it.
    pub timeout_ms: Option<u64>,
    /// How many times an errored attempt is retried (`max_retries + 1`
    /// attempts in total). Halts and cancellations are never retried.
    pub max_retries: u32,
}

impl NodeOptions {
    /// Returns [`NodeOptions::timeout_ms`] as a [`Duration`].
    #[must_use]
    pub fn timeout(&self) -> Option<Duration> {
        self.timeout_ms.map(Duration::from_millis)
    }
}

/// One unit of work in a graph superstep.
///
/// The builder id wins over [`Node::id`]; an empty id from the trait is
/// replaced by the id passed to [`crate::GraphBuilder::node`].
#[async_trait]
pub trait Node: Send + Sync {
    /// The node's own id, or `""` when the builder provides it.
    #[allow(clippy::unnecessary_literal_bound)] // the default is a literal, other ids borrow self
    fn id(&self) -> &str {
        ""
    }

    /// Runs the node against `state`.
    ///
    /// The engine hands every concurrent node its own state clone and merges
    /// the results back in declaration order.
    ///
    /// # Errors
    ///
    /// Returns an [`openjiuwen_common::error::Error`] when the node fails; the
    /// engine adds the node id and superstep as context and retries according
    /// to [`NodeOptions::max_retries`].
    async fn run(&self, state: &mut GraphState, ctx: &GraphContext) -> Result<NodeOutcome>;
}

/// What a node asks the engine to do after it completes.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeOutcome {
    /// Continue along the node's static and conditional edges.
    #[default]
    Continue,
    /// Activate exactly these nodes instead of following routers.
    Route {
        /// Target node ids, in activation order.
        targets: Vec<String>,
    },
    /// End the run after the current superstep.
    Halt {
        /// Value reported in [`crate::HaltInfo`].
        value: Option<Value>,
    },
}

impl NodeOutcome {
    /// Builds a [`NodeOutcome::Route`] from any iterator of node ids.
    #[must_use]
    pub fn route(targets: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self::Route {
            targets: targets.into_iter().map(Into::into).collect(),
        }
    }

    /// Builds a [`NodeOutcome::Halt`] carrying `value`.
    #[must_use]
    pub fn halt(value: Option<Value>) -> Self {
        Self::Halt { value }
    }
}

/// The per-invocation context handed to a [`Node`].
#[derive(Clone)]
pub struct GraphContext {
    session: Arc<dyn Session>,
    graph_id: String,
    node_id: String,
    superstep: u32,
    cancellation: CancellationToken,
    metadata: Arc<BTreeMap<String, Value>>,
}

impl GraphContext {
    /// Builds a context; the engine is the only constructor.
    pub(crate) fn new(
        session: Arc<dyn Session>,
        graph_id: impl Into<String>,
        node_id: impl Into<String>,
        superstep: u32,
        cancellation: CancellationToken,
        metadata: Arc<BTreeMap<String, Value>>,
    ) -> Self {
        Self {
            session,
            graph_id: graph_id.into(),
            node_id: node_id.into(),
            superstep,
            cancellation,
            metadata,
        }
    }

    /// Returns the session the run executes in.
    #[must_use]
    pub fn session(&self) -> &Arc<dyn Session> {
        &self.session
    }

    /// Returns the graph id.
    #[must_use]
    pub fn graph_id(&self) -> &str {
        &self.graph_id
    }

    /// Returns the id of the node being executed.
    #[must_use]
    pub fn node_id(&self) -> &str {
        &self.node_id
    }

    /// Returns the number of the superstep this invocation belongs to.
    #[must_use]
    pub fn superstep(&self) -> u32 {
        self.superstep
    }

    /// Returns the run's cancellation token.
    #[must_use]
    pub fn cancellation(&self) -> &CancellationToken {
        &self.cancellation
    }

    /// Reports whether the run was already cancelled.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.cancellation.is_cancelled()
    }

    /// Waits until the run is cancelled.
    pub async fn cancelled(&self) {
        self.cancellation.cancelled().await;
    }

    /// Returns the graph's metadata, shared as an immutable map.
    #[must_use]
    pub fn metadata(&self) -> &BTreeMap<String, Value> {
        &self.metadata
    }

    /// Returns the metadata as a shared handle, for router contexts.
    #[must_use]
    pub(crate) fn metadata_arc(&self) -> Arc<BTreeMap<String, Value>> {
        Arc::clone(&self.metadata)
    }
}

impl std::fmt::Debug for GraphContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GraphContext")
            .field("graph_id", &self.graph_id)
            .field("node_id", &self.node_id)
            .field("superstep", &self.superstep)
            .field("cancelled", &self.is_cancelled())
            .finish_non_exhaustive()
    }
}

/// The future a [`FnNode`] callback returns: boxed so the engine can prove it
/// is `Send` and run nodes concurrently.
pub type BoxNodeFuture<'a> = Pin<Box<dyn Future<Output = Result<NodeOutcome>> + Send + 'a>>;

/// A [`Node`] backed by an async closure.
///
/// The callback returns a boxed [`Send`] future (Rust 1.85's `AsyncFn` traits
/// cannot prove to a generic wrapper that their returned future is `Send` on
/// stable, so the boxing is explicit and the adapter stays `Send`).
///
/// ```
/// use openjiuwen_common::error::Result;
/// use openjiuwen_graph::{FnNode, GraphContext, GraphState, NodeOutcome};
///
/// let node = FnNode::new("add", |state: &mut GraphState, _ctx: &GraphContext| {
///     Box::pin(async move {
///         let count = state.get("count").and_then(|v| v.as_u64()).unwrap_or(0);
///         state.set("count", serde_json::json!(count + 1));
///         Ok(NodeOutcome::Continue)
///     })
/// });
/// ```
pub struct FnNode<F> {
    id: String,
    f: F,
}

impl<F> FnNode<F>
where
    F: for<'a> Fn(&'a mut GraphState, &'a GraphContext) -> BoxNodeFuture<'a>,
{
    /// Wraps `f` as a node with builder id `id`.
    pub fn new(id: impl Into<String>, f: F) -> Self {
        Self { id: id.into(), f }
    }
}

impl<F> std::fmt::Debug for FnNode<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FnNode")
            .field("id", &self.id)
            .field("f", &"<callback>")
            .finish()
    }
}

#[async_trait]
impl<F> Node for FnNode<F>
where
    F: for<'a> Fn(&'a mut GraphState, &'a GraphContext) -> BoxNodeFuture<'a>
        + Send
        + Sync
        + 'static,
{
    fn id(&self) -> &str {
        &self.id
    }

    async fn run(&self, state: &mut GraphState, ctx: &GraphContext) -> Result<NodeOutcome> {
        (self.f)(state, ctx).await
    }
}

/// A [`Node`] backed by a synchronous closure.
///
/// Sync nodes run inline on the executor task, so they must be cheap and
/// non-blocking; use [`FnNode`] for I/O.
pub struct SyncNode<F> {
    id: String,
    f: F,
}

impl<F> SyncNode<F>
where
    F: Fn(&mut GraphState, &GraphContext) -> Result<NodeOutcome>,
{
    /// Wraps `f` as a node with builder id `id`.
    pub fn new(id: impl Into<String>, f: F) -> Self {
        Self { id: id.into(), f }
    }
}

impl<F> std::fmt::Debug for SyncNode<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SyncNode")
            .field("id", &self.id)
            .field("f", &"<callback>")
            .finish()
    }
}

#[async_trait]
impl<F> Node for SyncNode<F>
where
    F: Fn(&mut GraphState, &GraphContext) -> Result<NodeOutcome> + Send + Sync + 'static,
{
    fn id(&self) -> &str {
        &self.id
    }

    async fn run(&self, state: &mut GraphState, ctx: &GraphContext) -> Result<NodeOutcome> {
        (self.f)(state, ctx)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]
mod tests {
    use super::*;
    use openjiuwen_session::WorkflowSession;
    use serde_json::json;

    fn context() -> GraphContext {
        GraphContext::new(
            Arc::new(WorkflowSession::new("s1")) as Arc<dyn Session>,
            "g",
            "n",
            3,
            CancellationToken::new(),
            Arc::new(BTreeMap::new()),
        )
    }

    #[tokio::test]
    async fn fn_node_runs_with_async_fn() {
        #[allow(clippy::unused_async)]
        async fn increment(state: &mut GraphState, _ctx: &GraphContext) -> Result<NodeOutcome> {
            let count = state.get("count").and_then(Value::as_u64).unwrap_or(0);
            state.set("count", json!(count + 1));
            Ok(NodeOutcome::Continue)
        }
        let node = FnNode::new("increment", |state: &mut GraphState, ctx: &GraphContext| {
            Box::pin(increment(state, ctx))
        });
        assert_eq!(node.id(), "increment");
        let mut state = GraphState::new();
        let outcome = node.run(&mut state, &context()).await.unwrap();
        assert_eq!(outcome, NodeOutcome::Continue);
        assert_eq!(state.get("count"), Some(&json!(1)));
    }

    #[tokio::test]
    async fn fn_node_accepts_async_closures() {
        let node = FnNode::new("route", |state: &mut GraphState, _ctx: &GraphContext| {
            Box::pin(async move {
                let go = state.get("go").and_then(Value::as_bool).unwrap_or(false);
                Ok(NodeOutcome::route(if go { vec!["a"] } else { vec!["b"] }))
            })
        });
        let mut state = GraphState::from_value(json!({"go": true})).unwrap();
        let outcome = node.run(&mut state, &context()).await.unwrap();
        assert_eq!(outcome, NodeOutcome::route(["a"]));
    }

    #[tokio::test]
    async fn sync_node_runs_inline() {
        let node = SyncNode::new("halt", |_state: &mut GraphState, _ctx: &GraphContext| {
            Ok(NodeOutcome::halt(Some(json!(1))))
        });
        let mut state = GraphState::new();
        assert_eq!(
            node.run(&mut state, &context()).await.unwrap(),
            NodeOutcome::Halt {
                value: Some(json!(1))
            }
        );
    }

    #[test]
    fn context_accessors() {
        let ctx = context();
        assert_eq!(ctx.graph_id(), "g");
        assert_eq!(ctx.node_id(), "n");
        assert_eq!(ctx.superstep(), 3);
        assert!(!ctx.is_cancelled());
        assert!(ctx.metadata().is_empty());
        assert_eq!(ctx.session().id(), "s1");
        assert_eq!(ctx.metadata_arc().len(), 0);
        assert!(format!("{ctx:?}").contains("node_id"));
    }

    #[tokio::test]
    async fn context_observes_cancellation() {
        let token = CancellationToken::new();
        let ctx = GraphContext::new(
            Arc::new(WorkflowSession::new("s1")) as Arc<dyn Session>,
            "g",
            "n",
            1,
            token.clone(),
            Arc::new(BTreeMap::new()),
        );
        assert!(!ctx.is_cancelled());
        token.cancel();
        assert!(ctx.is_cancelled());
        ctx.cancelled().await;
    }

    #[test]
    fn node_options_timeout() {
        let options = NodeOptions {
            timeout_ms: Some(250),
            max_retries: 2,
        };
        assert_eq!(options.timeout(), Some(Duration::from_millis(250)));
        assert_eq!(NodeOptions::default().timeout(), None);
    }

    #[test]
    fn outcome_serde_shapes() {
        assert_eq!(
            serde_json::to_value(NodeOutcome::Continue).unwrap(),
            json!("continue")
        );
        assert_eq!(
            serde_json::to_value(NodeOutcome::route(["a", "b"])).unwrap(),
            json!({"route": {"targets": ["a", "b"]}})
        );
        assert_eq!(
            serde_json::to_value(NodeOutcome::halt(Some(json!(1)))).unwrap(),
            json!({"halt": {"value": 1}})
        );
        assert_eq!(
            serde_json::to_value(NodeOutcome::default()).unwrap(),
            json!("continue")
        );
    }
}
