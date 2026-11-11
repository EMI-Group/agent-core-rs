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
//! The crate root re-exports the types most consumers need; the exact public
//! surface is documented in this crate's `CONTEXT.md`.
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

/// The default superstep budget of a run (Python `MAX_RECURSIVE_LIMIT`).
pub const DEFAULT_MAX_SUPERSTEPS: usize = 10_000;

/// The reserved routing source that seeds entry nodes (Python `START`).
pub const START_NODE: &str = "__start__";
