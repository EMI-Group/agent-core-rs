//! Workflow builder, components, and `` data flow.
//!
//! Maps to the Python package `core.workflow` in the openJiuwen reference implementation.
//! A workflow is a directed graph of components with one start and one end; the builder
//! records component configs and edges and compiling maps them onto `openjiuwen-graph`.
//! Data flow is textual: a component's `inputs_schema` is a JSON template whose pure
//! `""` strings resolve against workflow state before the component runs.

pub mod ability;
pub mod card;
pub mod component;
pub mod components;
pub mod condition;
pub mod context;
pub mod error;
pub mod interrupt;
pub mod output;
pub mod reference;
pub mod registry;

/// Returns the crate name.
#[must_use]
pub fn crate_name() -> &'static str {
    "openjiuwen-workflow"
}
