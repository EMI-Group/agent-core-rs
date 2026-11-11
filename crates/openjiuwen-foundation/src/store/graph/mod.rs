//! Graph storage.
//!
//! Nodes and edges as plain data (Python's `Entity`/`Relation`/`Episode` collapse
//! into `GraphNode`/`GraphEdge` with typed properties), with upsert, lookup,
//! neighbor traversal, and property filters.

pub mod file;
pub mod memory;
