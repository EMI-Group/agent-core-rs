//! Tests for the rule-based DL reflector and its optional model revision.
//!
//! Every case runs offline on scripted models: no network, no environment reads,
//! and no wall-clock dependence. The per-node-type validation cases build
//! documents from the Python-shaped DL model.
//!
//! The suite is grouped by concern: `graph` (deterministic graph repair),
//! `repair` (reflection vocabulary, debug output, and the optional model
//! revision), `node_rules` (basic and per-node-type field validation), and
//! `branch_rules` (intent and branch conditions). Shared fixtures live in
//! `support`.

mod support;

mod branch_rules;
mod graph;
mod node_rules;
mod repair;
