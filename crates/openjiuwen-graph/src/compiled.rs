//! Compiled graph facade: one-shot invocation, full run results, and ordered
//! event streams.

use std::collections::BTreeMap;
use std::sync::Arc;

use futures::stream::BoxStream;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_session::Session;

use crate::event::GraphEvent;
use crate::execution::ExecutionOptions;
use crate::node::{Node, NodeOptions};
use crate::pregel::channel::ChannelManager;
use crate::pregel::dispatch::DispatchTable;
use crate::spec::GraphSpec;
use crate::state::GraphState;
use crate::store::GraphStore;
use crate::visualize::{render_mermaid, MermaidOptions};

/// The result of a full graph run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct GraphRunResult {
    /// The graph id.
    pub graph_id: String,
    /// The final graph state.
    pub state: GraphState,
    /// The number of supersteps executed (including resumed ones).
    pub supersteps: u32,
    /// The nodes that completed, in execution order.
    pub visited: Vec<String>,
    /// The halt that ended the run, when it halted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub halted: Option<HaltInfo>,
    /// Whether the run continued from a stored snapshot.
    pub resumed: bool,
}

/// Where and why a run halted.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct HaltInfo {
    /// The node that returned [`crate::NodeOutcome::Halt`].
    pub node_id: String,
    /// The superstep the halt happened in.
    pub superstep: u32,
    /// The value the node halted with.
    pub value: Option<Value>,
}

/// Everything a compiled graph needs at run time.
pub(crate) struct CompiledInner {
    pub(crate) graph_id: String,
    pub(crate) spec: GraphSpec,
    pub(crate) nodes: BTreeMap<String, Arc<dyn Node>>,
    pub(crate) options: BTreeMap<String, NodeOptions>,
    pub(crate) order: Vec<String>,
    pub(crate) channels: ChannelManager,
    pub(crate) dispatch: DispatchTable,
    pub(crate) store: Option<Arc<dyn GraphStore>>,
    pub(crate) namespace: String,
    pub(crate) metadata: Arc<BTreeMap<String, Value>>,
    pub(crate) max_supersteps: Option<usize>,
    pub(crate) node_timeout_ms: Option<u64>,
    pub(crate) checkpoint_every_superstep: bool,
}

impl CompiledInner {
    /// Returns the compiled runnable of `id`.
    pub(crate) fn node(&self, id: &str) -> Option<&Arc<dyn Node>> {
        self.nodes.get(id)
    }

    /// Returns the spec options of `id`.
    pub(crate) fn node_options(&self, id: &str) -> NodeOptions {
        self.options.get(id).cloned().unwrap_or_default()
    }
}

/// A validated, runnable graph.
///
/// Cloning is cheap (one `Arc`); a compiled graph can be run any number of
/// times, concurrently, with independent state.
#[derive(Clone)]
pub struct CompiledGraph {
    pub(crate) inner: Arc<CompiledInner>,
}

impl CompiledGraph {
    /// Returns the graph id.
    #[must_use]
    pub fn id(&self) -> &str {
        &self.inner.graph_id
    }

    /// Returns the compiled spec.
    #[must_use]
    pub fn spec(&self) -> &GraphSpec {
        &self.inner.spec
    }

    /// Iterates the node ids in declaration order.
    pub fn node_ids(&self) -> impl Iterator<Item = &str> {
        self.inner.order.iter().map(String::as_str)
    }

    /// Returns the entry node ids.
    #[must_use]
    pub fn entry_nodes(&self) -> &[String] {
        &self.inner.spec.entry
    }

    /// Returns the exit node ids.
    #[must_use]
    pub fn exit_nodes(&self) -> Vec<&str> {
        self.inner.spec.exit.iter().map(String::as_str).collect()
    }

    /// Runs the graph once and returns the final state as a JSON object.
    ///
    /// # Errors
    ///
    /// Propagates graph validation, node, router, timeout, and cancellation
    /// failures.
    pub async fn invoke(&self, input: Value, session: Arc<dyn Session>) -> Result<Value> {
        let result = self
            .run(input, session, ExecutionOptions::default())
            .await?;
        Ok(result.state.into_value())
    }

    /// Runs the graph once and returns the full result.
    ///
    /// # Errors
    ///
    /// Propagates graph validation, node, router, timeout, and cancellation
    /// failures.
    pub async fn run(
        &self,
        input: Value,
        session: Arc<dyn Session>,
        options: ExecutionOptions,
    ) -> Result<GraphRunResult> {
        crate::pregel::executor::execute(Arc::clone(&self.inner), input, session, options).await
    }

    /// Streams the run's events with the default execution options.
    pub fn stream(
        &self,
        input: Value,
        session: Arc<dyn Session>,
    ) -> BoxStream<'static, Result<GraphEvent>> {
        self.stream_with(input, session, ExecutionOptions::default())
    }

    /// Streams the run's events.
    ///
    /// The stream yields events in execution order and, when the run fails,
    /// exactly one [`Err`] after the failure's events.
    pub fn stream_with(
        &self,
        input: Value,
        session: Arc<dyn Session>,
        mut options: ExecutionOptions,
    ) -> BoxStream<'static, Result<GraphEvent>> {
        let inner = Arc::clone(&self.inner);
        let (sender, mut receiver) = tokio::sync::mpsc::unbounded_channel();
        options.events = Some(sender);
        let stream = async_stream::stream! {
            let handle = tokio::spawn(async move {
                crate::pregel::executor::execute(inner, input, session, options).await
            });
            while let Some(event) = receiver.recv().await {
                yield Ok(event);
            }
            match handle.await {
                Ok(Ok(_)) => {}
                Ok(Err(error)) => yield Err(error),
                Err(join_error) => yield Err(Error::external(join_error)),
            }
        };
        Box::pin(stream)
    }

    /// Renders the graph as a Mermaid flowchart with default options.
    #[must_use]
    pub fn to_mermaid(&self) -> String {
        self.to_mermaid_with(&MermaidOptions::default())
    }

    /// Renders the graph as a Mermaid flowchart.
    #[must_use]
    pub fn to_mermaid_with(&self, options: &MermaidOptions) -> String {
        render_mermaid(&self.inner.spec, options)
    }
}

impl std::fmt::Debug for CompiledGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompiledGraph")
            .field("graph_id", &self.inner.graph_id)
            .field("nodes", &self.inner.order.len())
            .field("edges", &self.inner.spec.edges.len())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]
mod tests {
    use super::*;
    use crate::builder::GraphBuilder;
    use crate::node::{GraphContext, NodeOutcome, SyncNode};
    use openjiuwen_session::WorkflowSession;
    use serde_json::json;

    fn graph() -> CompiledGraph {
        GraphBuilder::new("g")
            .node(
                "inc",
                SyncNode::new("inc", |state: &mut GraphState, _ctx: &GraphContext| {
                    let count = state.get("count").and_then(Value::as_u64).unwrap_or(0);
                    state.set("count", json!(count + 1));
                    Ok(NodeOutcome::Continue)
                }),
            )
            .compile()
            .unwrap()
    }

    fn session() -> Arc<dyn Session> {
        Arc::new(WorkflowSession::new("s1"))
    }

    #[tokio::test]
    async fn run_and_invoke_report_state() {
        let graph = graph();
        assert_eq!(graph.id(), "g");
        assert_eq!(graph.node_ids().collect::<Vec<_>>(), vec!["inc"]);
        assert_eq!(graph.entry_nodes(), ["inc".to_string()]);
        assert_eq!(graph.exit_nodes(), Vec::<&str>::new());
        assert!(format!("{graph:?}").contains("graph_id"));

        let result = graph
            .run(json!({"count": 1}), session(), ExecutionOptions::default())
            .await
            .unwrap();
        assert_eq!(result.state.get("count"), Some(&json!(2)));
        assert_eq!(result.supersteps, 1);
        assert_eq!(result.visited, vec!["inc".to_string()]);
        assert!(result.halted.is_none());
        assert!(!result.resumed);
        assert_eq!(result.graph_id, "g");

        let value = graph.invoke(json!({"count": 5}), session()).await.unwrap();
        assert_eq!(value, json!({"count": 6}));
    }

    #[tokio::test]
    async fn run_rejects_non_object_input() {
        let error = graph().invoke(json!(7), session()).await.unwrap_err();
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Validation
        );
    }

    #[tokio::test]
    async fn stream_reports_every_event() {
        use futures::StreamExt;
        let events: Vec<GraphEvent> = graph()
            .stream(json!({"count": 0}), session())
            .map(|event| event.unwrap())
            .collect()
            .await;
        assert_eq!(
            events.first().unwrap().kind,
            crate::GraphEventKind::SuperstepStarted
        );
        assert_eq!(
            events.last().unwrap().kind,
            crate::GraphEventKind::GraphCompleted
        );
        assert!(events
            .iter()
            .any(|event| event.kind == crate::GraphEventKind::NodeCompleted));
    }

    #[tokio::test]
    async fn stream_yields_one_error_for_a_failing_run() {
        use futures::StreamExt;
        let graph = GraphBuilder::new("g")
            .node(
                "boom",
                SyncNode::new("boom", |_state: &mut GraphState, _ctx: &GraphContext| {
                    Err(openjiuwen_common::error::Error::validation("boom"))
                }),
            )
            .compile()
            .unwrap();
        let collected: Vec<Result<GraphEvent>> =
            graph.stream(json!({}), session()).collect::<Vec<_>>().await;
        let errors = collected.iter().filter(|item| item.is_err()).count();
        assert_eq!(errors, 1);
        let events: Vec<&GraphEvent> = collected
            .iter()
            .filter_map(|item| item.as_ref().ok())
            .collect();
        assert_eq!(
            events.first().unwrap().kind,
            crate::GraphEventKind::SuperstepStarted
        );
        assert!(events
            .iter()
            .any(|event| event.kind == crate::GraphEventKind::NodeFailed));
        assert_eq!(
            events.last().unwrap().kind,
            crate::GraphEventKind::GraphFailed
        );
    }

    #[test]
    fn mermaid_delegation() {
        let graph = graph();
        assert!(graph.to_mermaid().starts_with("flowchart TB"));
        let options = MermaidOptions {
            title: Some("t".to_string()),
            ..MermaidOptions::default()
        };
        assert!(graph.to_mermaid_with(&options).contains("title: t"));
    }
}
