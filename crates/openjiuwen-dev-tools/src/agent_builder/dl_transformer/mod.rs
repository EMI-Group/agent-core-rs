//! The Python-shaped DL (`SimpleIR`) model and, later, its platform-DSL converter.
//!
//! Ports `openjiuwen.dev_tools.agent_builder.builders.workflow.dl_transformer`:
//! [`model`] holds the wire document model (a bare JSON array of nodes with
//! Python's capitalized `type` values, per-node `parameters`, and branch
//! conditions), which the DL reflector and the future transformer build on.
//!
//! The simplified platform document ([`WorkflowDsl`]) is a separate model and is
//! not produced or consumed here yet.
//!
//! [`WorkflowDsl`]: crate::agent_builder::workflow_dsl::WorkflowDsl

pub mod model;

pub use model::{DlCondition, DlDocument, DlNode, DlNodeType, DlParameters, DlVariable};
