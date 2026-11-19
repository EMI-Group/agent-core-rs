//! The superstep executor.
//!
//! A run executes whole supersteps: compute the ready nodes in declaration
//! order, consume their channels, run them concurrently, apply their outcomes in
//! declaration order, route, flush the accept-then-activate buffer, and
//! checkpoint. Node state mutations are functional per node and merged in
//! declaration order as a top-level delta against the pre-superstep state, so
//! the outcome of a superstep never depends on which node finished first and a
//! node's top-level key removals propagate to the run state.

use std::sync::Arc;
#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use futures::Future;
use serde_json::json;
use tokio::sync::mpsc::UnboundedSender;
use tokio::task::JoinSet;
use tracing::{debug, warn};

use openjiuwen_common::error::{Error, Result};
use openjiuwen_session::{Session, StreamItem};

use crate::compiled::{CompiledInner, GraphRunResult, HaltInfo};
use crate::event::{error_payload, GraphEvent, GraphEventKind};
use crate::execution::ExecutionOptions;
use crate::node::{GraphContext, Node, NodeOutcome};
use crate::pregel::channel::ChannelManager;
use crate::state::GraphState;
use crate::store::GraphSnapshot;
use crate::DEFAULT_MAX_SUPERSTEPS;
use crate::START_NODE;

/// Delivers events to the optional caller sink and the session trace stream.
#[derive(Clone)]
struct EventSink {
    sender: Option<UnboundedSender<GraphEvent>>,
    session: Arc<dyn Session>,
    write_trace: bool,
}

impl EventSink {
    fn new(
        sender: Option<UnboundedSender<GraphEvent>>,
        session: Arc<dyn Session>,
        write_trace: bool,
    ) -> Self {
        Self {
            sender,
            session,
            write_trace,
        }
    }

    /// Emits one event to every configured consumer.
    async fn emit(&self, event: GraphEvent) {
        if let Some(sender) = &self.sender {
            if sender.send(event.clone()).is_err() {
                debug!(kind = event.kind.as_str(), "graph event receiver dropped");
            }
        }
        if self.write_trace {
            if let Err(error) = self
                .session
                .write_stream(StreamItem::trace("graph_event", event.to_wire()))
                .await
            {
                warn!(%error, "failed to mirror a graph event to the session stream");
            }
        }
    }
}

/// The outcome of one node's attempts within a superstep.
type NodeRunResult = Result<(GraphState, NodeOutcome)>;

/// Counts the boundary snapshots a run builds.
///
/// A test-only seam: production runs carry [`BoundaryProbe::default`], whose
/// [`BoundaryProbe::record`] compiles away, while unit tests pass a counter and
/// prove that a store-less run copies no state for a snapshot no backend could
/// persist, and that a stored run builds exactly one snapshot per boundary.
#[derive(Clone, Default)]
struct BoundaryProbe {
    #[cfg(test)]
    count: Option<Arc<AtomicUsize>>,
}

impl BoundaryProbe {
    /// Counts every snapshot into `count`.
    #[cfg(test)]
    fn counting(count: Arc<AtomicUsize>) -> Self {
        Self { count: Some(count) }
    }

    /// Records one built snapshot.
    fn record(&self) {
        #[cfg(test)]
        if let Some(count) = &self.count {
            count.fetch_add(1, Ordering::SeqCst);
        }
    }
}

/// A node result tagged with its declaration index and id, so outcomes of a
/// concurrent superstep can be applied deterministically.
type OrderedNodeResult = (usize, String, NodeRunResult);

/// Runs `inner` to completion with `input`.
///
/// # Errors
///
/// Propagates snapshot, node, router, timeout, cancellation, and superstep-limit
/// failures. A failed run saves a snapshot at its last completed superstep
/// boundary when a store is configured.
#[allow(clippy::too_many_lines)] // the superstep loop reads best as one sequence
pub(crate) async fn execute(
    inner: Arc<CompiledInner>,
    input: serde_json::Value,
    session: Arc<dyn Session>,
    options: ExecutionOptions,
) -> Result<GraphRunResult> {
    execute_probed(inner, input, session, options, BoundaryProbe::default()).await
}

/// [`execute`] with the boundary-snapshot probe the unit tests count with.
#[allow(clippy::too_many_lines)] // the superstep loop reads best as one sequence
async fn execute_probed(
    inner: Arc<CompiledInner>,
    input: serde_json::Value,
    session: Arc<dyn Session>,
    options: ExecutionOptions,
    probe: BoundaryProbe,
) -> Result<GraphRunResult> {
    let mut state = GraphState::new();
    let mut channels = inner.channels.clone();
    let mut superstep: u32 = 0;
    let mut resumed = false;

    if options.resume {
        if let Some(store) = inner.store.as_ref() {
            let snapshot = store
                .get(session.id(), &inner.namespace)
                .await
                .map_err(|error| {
                    error.with_context(format!(
                        "loading the graph snapshot of session `{}`",
                        session.id()
                    ))
                })?;
            if let Some(snapshot) = snapshot {
                if snapshot.is_resumable() {
                    state = snapshot.state;
                    channels.restore(&snapshot.channels)?;
                    channels.restore_buffer(snapshot.buffer);
                    channels.flush()?;
                    superstep = snapshot.superstep;
                    resumed = true;
                    if !input.is_null() {
                        warn!(
                            session = session.id(),
                            "resuming a graph run: the supplied input is ignored"
                        );
                    }
                }
            }
        } else {
            warn!("resume requested but the graph has no snapshot store");
        }
    }

    if !resumed {
        state = GraphState::from_value(input)?;
        for entry in &inner.spec.entry {
            channels.accept(entry, START_NODE)?;
        }
        channels.flush()?;
    }

    let sink = EventSink::new(
        options.events.clone(),
        Arc::clone(&session),
        options.write_session_trace,
    );
    let max_supersteps = options
        .max_supersteps
        .or(inner.max_supersteps)
        .unwrap_or(DEFAULT_MAX_SUPERSTEPS);
    let mut visited: Vec<String> = Vec::new();
    let mut halted: Option<HaltInfo> = None;

    // The run state is shared with routers through an `Arc`: routing never
    // copies it, and `Arc::make_mut` keeps merges copy-on-write (a merge copies
    // the state only while a router still holds the pre-merge one).
    let mut state = Arc::new(state);

    // The boundary a failed run persists: the initial state before the first
    // superstep, then the last completed superstep's snapshot. It is built only
    // when a store is attached, so a store-less run never deep-copies the state
    // for a snapshot no backend could persist.
    let mut boundary = snapshot_if_stored(&inner, &state, &channels, superstep, &probe);

    loop {
        if options.cancellation.is_cancelled() {
            let error = fail(
                &inner,
                &session,
                boundary.as_ref(),
                superstep,
                Error::cancelled(),
                &sink,
            )
            .await;
            return Err(error);
        }

        let ready = channels.ready_nodes(&inner.order);
        if ready.is_empty() {
            break;
        }
        if superstep as usize >= max_supersteps {
            let error = Error::graph(format!(
                "graph `{}` exceeded the maximum of {max_supersteps} supersteps",
                inner.graph_id
            ));
            let error = fail(&inner, &session, boundary.as_ref(), superstep, error, &sink).await;
            return Err(error);
        }

        superstep += 1;

        sink.emit(GraphEvent::new(
            GraphEventKind::SuperstepStarted,
            None,
            superstep,
            json!({"ready": ready}),
        ))
        .await;
        for id in &ready {
            channels.consume(id)?;
        }
        for id in &ready {
            sink.emit(GraphEvent::new(
                GraphEventKind::NodeStarted,
                Some(id.clone()),
                superstep,
                json!({}),
            ))
            .await;
        }

        let results = run_nodes(&inner, &session, &state, &options, &ready, superstep).await?;

        match apply_outcomes(
            &inner,
            &session,
            &sink,
            &options,
            superstep,
            results,
            &mut state,
            &mut channels,
            &mut visited,
        )
        .await
        {
            Ok(node_halt) => {
                if node_halt.is_some() {
                    halted = node_halt;
                }
            }
            Err(error) => {
                let error =
                    fail(&inner, &session, boundary.as_ref(), superstep, error, &sink).await;
                return Err(error);
            }
        }

        if let Err(error) = channels.flush() {
            let error = fail(&inner, &session, boundary.as_ref(), superstep, error, &sink).await;
            return Err(error);
        }

        sink.emit(GraphEvent::new(
            GraphEventKind::SuperstepCompleted,
            None,
            superstep,
            json!({"superstep": superstep}),
        ))
        .await;

        // One snapshot per durable boundary: it is both this superstep's
        // checkpoint and the boundary a later failure restores, so the state is
        // copied once here instead of once per boundary site.
        if let Some(snapshot) = snapshot_if_stored(&inner, &state, &channels, superstep, &probe) {
            if inner.checkpoint_every_superstep {
                if let Some(store) = inner.store.as_ref() {
                    if let Err(error) = store.save(session.id(), &inner.namespace, &snapshot).await
                    {
                        warn!(%error, "failed to save a graph snapshot after a superstep");
                    }
                }
            }
            boundary = Some(snapshot);
        }

        if let Some(halt) = &halted {
            sink.emit(GraphEvent::new(
                GraphEventKind::GraphHalted,
                Some(halt.node_id.clone()),
                superstep,
                json!({"value": halt.value}),
            ))
            .await;
            break;
        }
    }

    sink.emit(GraphEvent::new(
        GraphEventKind::GraphCompleted,
        None,
        superstep,
        json!({
            "supersteps": superstep,
            "visited": visited,
            "halted": halted.is_some(),
        }),
    ))
    .await;

    Ok(GraphRunResult {
        graph_id: inner.graph_id.clone(),
        state: Arc::try_unwrap(state).unwrap_or_else(|shared| shared.as_ref().clone()),
        supersteps: superstep,
        visited,
        halted,
        resumed,
    })
}

/// Runs every ready node concurrently, returning their outcomes ordered by
/// declaration position.
async fn run_nodes(
    inner: &Arc<CompiledInner>,
    session: &Arc<dyn Session>,
    state: &GraphState,
    options: &ExecutionOptions,
    ready: &[String],
    superstep: u32,
) -> Result<Vec<OrderedNodeResult>> {
    let mut join_set: JoinSet<OrderedNodeResult> = JoinSet::new();
    for (index, id) in ready.iter().enumerate() {
        let node = inner.node(id).cloned().ok_or_else(|| {
            Error::graph(format!("graph channel is ready for unknown node `{id}`"))
        })?;
        let node_options = inner.node_options(id);
        let timeout = options
            .node_timeout_ms
            .or(inner.node_timeout_ms)
            .or(node_options.timeout_ms)
            .map(Duration::from_millis);
        let node_state = state.clone();
        let ctx = GraphContext::new(
            Arc::clone(session),
            inner.graph_id.clone(),
            id.clone(),
            superstep,
            options.cancellation.clone(),
            Arc::clone(&inner.metadata),
        );
        let node_id = id.clone();
        join_set.spawn(async move {
            let result = run_node(
                node,
                &node_id,
                node_state,
                &ctx,
                timeout,
                node_options.max_retries,
            )
            .await;
            (index, node_id, result)
        });
    }

    let mut results = Vec::with_capacity(ready.len());
    while let Some(joined) = join_set.join_next().await {
        let (index, id, result) = joined
            .map_err(|error| Error::external(error).with_context("a graph node task failed"))?;
        results.push((index, id, result));
    }
    results.sort_by_key(|(index, _, _)| *index);
    Ok(results)
}

/// Runs one node with retries, the resolved timeout, and cancellation.
async fn run_node(
    node: Arc<dyn Node>,
    id: &str,
    state: GraphState,
    ctx: &GraphContext,
    timeout: Option<Duration>,
    max_retries: u32,
) -> NodeRunResult {
    let mut last_error: Option<Error> = None;
    // The given state is consumed by the first attempt, so a run without retries
    // never copies it. A retry needs a pristine copy, which is cloned right
    // before an attempt that can still be retried (never after the last one).
    let mut pristine = Some(state);
    for attempt in 0..=max_retries {
        let Some(mut attempt_state) = pristine.take() else {
            break;
        };
        if attempt < max_retries {
            pristine = Some(attempt_state.clone());
        }
        let raced = async {
            tokio::select! {
                biased;
                () = ctx.cancelled() => Err(Error::cancelled()),
                result = node.run(&mut attempt_state, ctx) => result,
            }
        };
        let result = with_timeout(raced, timeout, id).await;
        match result {
            Ok(outcome) => return Ok((attempt_state, outcome)),
            Err(error) if error.is_cancelled() => return Err(error),
            Err(error) => {
                if attempt < max_retries {
                    warn!(
                        node = id,
                        attempt = attempt + 1,
                        error = %error,
                        "graph node failed; retrying"
                    );
                }
                last_error = Some(error);
            }
        }
    }
    Err(last_error.unwrap_or_else(|| Error::graph(format!("node `{id}` failed without an error"))))
}

/// Applies node outcomes in declaration order.
///
/// The run state is the shared [`Arc`] the routers read: `Arc::make_mut` keeps a
/// merge copy-on-write, so a superstep whose routers all finished costs no copy.
/// Each node receives a clone of the pre-superstep state, so its result is
/// merged as a delta relative to that baseline: top-level keys the node changed
/// are deep-merged and top-level keys the node removed are deleted from the run
/// state ([`GraphState::merge_node_delta`]).
#[allow(clippy::too_many_arguments)]
async fn apply_outcomes(
    inner: &Arc<CompiledInner>,
    session: &Arc<dyn Session>,
    sink: &EventSink,
    options: &ExecutionOptions,
    superstep: u32,
    results: Vec<OrderedNodeResult>,
    state: &mut Arc<GraphState>,
    channels: &mut ChannelManager,
    visited: &mut Vec<String>,
) -> Result<Option<HaltInfo>> {
    let base = state.as_ref().clone();
    let mut halted: Option<HaltInfo> = None;
    for (_, id, result) in results {
        let (node_state, outcome) = match result {
            Ok(completed) => completed,
            Err(error) => {
                let error =
                    error.with_context(format!("node `{id}` failed at superstep {superstep}"));
                sink.emit(GraphEvent::new(
                    GraphEventKind::NodeFailed,
                    Some(id),
                    superstep,
                    error_payload(&error),
                ))
                .await;
                return Err(error);
            }
        };

        Arc::make_mut(state).merge_node_delta(&base, node_state);
        visited.push(id.clone());
        sink.emit(GraphEvent::new(
            GraphEventKind::NodeCompleted,
            Some(id.clone()),
            superstep,
            json!({"outcome": outcome}),
        ))
        .await;

        match outcome {
            NodeOutcome::Halt { value } => {
                if halted.is_none() {
                    halted = Some(HaltInfo {
                        node_id: id,
                        superstep,
                        value,
                    });
                }
            }
            other => {
                let ctx = GraphContext::new(
                    Arc::clone(session),
                    inner.graph_id.clone(),
                    id.clone(),
                    superstep,
                    options.cancellation.clone(),
                    Arc::clone(&inner.metadata),
                );
                let targets = inner
                    .dispatch
                    .targets_for(&id, &other, state, &ctx)
                    .await
                    .map_err(|error| {
                        error.with_context(format!("node `{id}` failed at superstep {superstep}"))
                    })?;
                for target in targets {
                    sink.emit(GraphEvent::new(
                        GraphEventKind::EdgeTraversed,
                        Some(id.clone()),
                        superstep,
                        json!({"source": id, "target": target}),
                    ))
                    .await;
                    channels.accept(&target, &id)?;
                }
            }
        }
    }
    Ok(halted)
}

/// Captures the run at a superstep boundary.
fn boundary_snapshot(
    inner: &CompiledInner,
    state: &GraphState,
    channels: &ChannelManager,
    superstep: u32,
) -> GraphSnapshot {
    GraphSnapshot {
        graph_id: inner.graph_id.clone(),
        namespace: inner.namespace.clone(),
        superstep,
        state: state.clone(),
        channels: channels.snapshot(),
        buffer: channels.buffer_snapshot(),
    }
}

/// Captures the run at a superstep boundary, but only when a store is attached.
///
/// A snapshot no backend can persist is pure cost — it deep-copies the whole
/// state and every channel — so a graph compiled without a store never builds
/// one, and a stored graph builds exactly one per boundary.
fn snapshot_if_stored(
    inner: &CompiledInner,
    state: &GraphState,
    channels: &ChannelManager,
    superstep: u32,
    probe: &BoundaryProbe,
) -> Option<GraphSnapshot> {
    inner.store.as_ref()?;
    probe.record();
    Some(boundary_snapshot(inner, state, channels, superstep))
}

/// Saves the boundary snapshot (when possible), emits `GraphFailed`, and returns
/// the original error unchanged.
async fn fail(
    inner: &CompiledInner,
    session: &Arc<dyn Session>,
    boundary: Option<&GraphSnapshot>,
    superstep: u32,
    error: Error,
    sink: &EventSink,
) -> Error {
    if let (Some(store), Some(snapshot)) = (inner.store.as_ref(), boundary) {
        if let Err(save_error) = store.save(session.id(), &inner.namespace, snapshot).await {
            warn!(error = %save_error, "failed to save a graph snapshot after a failed run");
        }
    }
    sink.emit(GraphEvent::new(
        GraphEventKind::GraphFailed,
        None,
        superstep,
        error_payload(&error),
    ))
    .await;
    error
}

/// Resolves a future, mapping an elapsed timeout to [`Error::timeout`].
async fn with_timeout<T>(
    future: impl Future<Output = Result<T>>,
    timeout: Option<Duration>,
    node_id: &str,
) -> Result<T> {
    match timeout {
        Some(limit) => match tokio::time::timeout(limit, future).await {
            Ok(result) => result,
            Err(_) => Err(Error::timeout(format!(
                "node `{node_id}` timed out after {} ms",
                limit.as_millis()
            ))),
        },
        None => future.await,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]
mod tests {
    use super::*;
    use crate::builder::GraphBuilder;
    use crate::node::SyncNode;
    use openjiuwen_session::WorkflowSession;
    use serde_json::Value;

    fn session() -> Arc<dyn Session> {
        Arc::new(WorkflowSession::new("s1"))
    }

    fn counting(
        id: &'static str,
    ) -> SyncNode<impl Fn(&mut GraphState, &GraphContext) -> Result<NodeOutcome>> {
        SyncNode::new(id, move |state: &mut GraphState, _ctx: &GraphContext| {
            let count = state.get("count").and_then(Value::as_u64).unwrap_or(0);
            state.set("count", json!(count + 1));
            Ok(NodeOutcome::Continue)
        })
    }

    #[tokio::test]
    async fn timeout_helper_maps_elapsed_deadlines() {
        let result: Result<()> = with_timeout(
            async {
                tokio::time::sleep(Duration::from_millis(50)).await;
                Ok(())
            },
            Some(Duration::from_millis(1)),
            "slow",
        )
        .await;
        let error = result.unwrap_err();
        assert_eq!(error.kind(), openjiuwen_common::error::ErrorKind::Timeout);
        assert!(error.message().contains("node `slow` timed out"));

        let result: Result<u8> = with_timeout(async { Ok(7) }, None, "fast").await;
        assert_eq!(result.unwrap(), 7);
    }

    #[tokio::test]
    async fn cancellation_before_the_first_superstep_fails_the_run() {
        let graph = GraphBuilder::new("g")
            .node("a", counting("a"))
            .compile()
            .unwrap();
        let options = ExecutionOptions::default();
        options.cancellation.cancel();
        let error = graph.run(json!({}), session(), options).await.unwrap_err();
        assert_eq!(error.kind(), openjiuwen_common::error::ErrorKind::Cancelled);
    }

    #[tokio::test]
    async fn superstep_limit_is_strict() {
        let graph = GraphBuilder::new("g")
            .node("a", counting("a"))
            .edge("a", "a")
            .entry("a")
            .compile()
            .unwrap();
        let error = graph
            .run(
                json!({}),
                session(),
                ExecutionOptions {
                    max_supersteps: Some(3),
                    ..ExecutionOptions::default()
                },
            )
            .await
            .unwrap_err();
        assert!(error
            .message()
            .contains("exceeded the maximum of 3 supersteps"));
    }

    #[tokio::test]
    async fn state_is_merged_in_declaration_order() {
        // Two parallel nodes write the same scalar; the later declaration wins.
        let graph = GraphBuilder::new("g")
            .node(
                "a",
                SyncNode::new("a", |state: &mut GraphState, _ctx: &GraphContext| {
                    state.set("who", json!("a"));
                    Ok(NodeOutcome::Continue)
                }),
            )
            .node(
                "b",
                SyncNode::new("b", |state: &mut GraphState, _ctx: &GraphContext| {
                    state.set("who", json!("b"));
                    Ok(NodeOutcome::Continue)
                }),
            )
            .barrier_edge(["a", "b"], "c")
            .node(
                "c",
                SyncNode::new("c", |_state: &mut GraphState, _ctx: &GraphContext| {
                    Ok(NodeOutcome::Continue)
                }),
            )
            .compile()
            .unwrap();
        let result = graph
            .run(json!({}), session(), ExecutionOptions::default())
            .await
            .unwrap();
        assert_eq!(result.state.get("who"), Some(&json!("b")));
        assert_eq!(
            result.visited,
            vec!["a".to_string(), "b".to_string(), "c".to_string()]
        );
    }
}
