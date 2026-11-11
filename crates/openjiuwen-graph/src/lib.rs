//! Pregel/BSP graph execution engine.
//!
//! Maps to the Python package `openjiuwen/core/graph` in the openJiuwen
//! reference implementation: a graph builder with vertices, edges, barrier
//! fan-in, and conditional routers; a deterministic superstep executor with
//! cancellation, retries, timeouts, checkpoint/resume, streaming events, and
//! Mermaid visualization.
//!
//! Modules:
//!
//! - [`state`] — [`state::GraphState`], the JSON state passed between nodes.
//! - [`node`] — the [`node::Node`] trait, [`node::NodeOutcome`], node options,
//!   and the [`node::GraphContext`] handed to every node invocation.
//! - [`router`] — conditional routing ([`router::Router`], [`router::RouterContext`]).
//! - [`spec`] — the serializable graph description ([`spec::GraphSpec`]).
//! - [`builder`] — [`builder::GraphBuilder`] and [`builder::CompileOptions`].
//! - [`compiled`] — the runnable [`compiled::CompiledGraph`].
//! - [`execution`] — per-run [`execution::ExecutionOptions`].
//! - [`event`] — ordered [`event::GraphEvent`] stream items.
//! - [`store`] — snapshot persistence ([`store::GraphStore`],
//!   [`store::InMemoryGraphStore`], [`store::CheckpointerStore`]).
//! - [`visualize`] — Mermaid rendering of a compiled graph.
//!
//! The executor internals (`pregel`) are crate-private; use the builder and
//! [`compiled::CompiledGraph`] instead.
//!
//! ```
//! use openjiuwen_common::error::Result;
//! use openjiuwen_graph::{GraphBuilder, GraphContext, GraphState, NodeOutcome, SyncNode};
//!
//! # fn main() -> Result<()> {
//! let graph = GraphBuilder::new("example")
//!     .node(
//!         "answer",
//!         SyncNode::new("answer", |state: &mut GraphState, _ctx: &GraphContext| {
//!             state.set("answer", serde_json::json!(42));
//!             Ok(NodeOutcome::Continue)
//!         }),
//!     )
//!     .compile()?;
//! assert_eq!(graph.node_ids().collect::<Vec<_>>(), vec!["answer"]);
//! # Ok(())
//! # }
//! ```
#![forbid(unsafe_code)]

pub mod builder;
pub mod compiled;
pub mod event;
pub mod execution;
pub mod node;
pub mod router;
pub mod spec;
pub mod state;
pub mod store;
pub mod visualize;

pub(crate) mod pregel;

pub use builder::{CompileOptions, GraphBuilder};
pub use compiled::{CompiledGraph, GraphRunResult, HaltInfo};
pub use event::{GraphEvent, GraphEventKind};
pub use execution::ExecutionOptions;
pub use node::{BoxNodeFuture, FnNode, GraphContext, Node, NodeOptions, NodeOutcome, SyncNode};
pub use router::{BoxRouterFuture, FnRouter, Router, RouterContext};
pub use spec::{EdgeKind, EdgeSpec, GraphSpec, NodeSpec};
pub use state::GraphState;
pub use store::{
    ChannelMessage, ChannelSnapshot, CheckpointerStore, GraphSnapshot, GraphStore,
    InMemoryGraphStore,
};
pub use visualize::{render_mermaid, MermaidDirection, MermaidOptions};

/// The default superstep budget of a run (Python `MAX_RECURSIVE_LIMIT`).
pub const DEFAULT_MAX_SUPERSTEPS: usize = 10_000;

/// The reserved routing source that seeds entry nodes (Python `START`).
pub const START_NODE: &str = "__start__";
