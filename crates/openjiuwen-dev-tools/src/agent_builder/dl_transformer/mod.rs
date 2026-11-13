//! The Python-shaped DL (`SimpleIR`) model and its platform-DSL transformer.
//!
//! Ports `openjiuwen.dev_tools.agent_builder.builders.workflow.dl_transformer`:
//! [`model`] holds the wire document model (a bare JSON array of nodes with
//! Python's capitalized `type` values, per-node `parameters`, and branch
//! conditions), [`platform`] holds the platform workflow document it is
//! transformed into, and [`convert`] implements the transformation itself.
//!
//! [`DlTransformer`] converts a [`DlDocument`] into a [`PlatformWorkflow`] with
//! Python's per-node-type rules: numeric platform `type` ids, `meta.position`
//! stepping by 20, typed input variables (`constant` or `ref` with their
//! `extra.index`), nested `object` output schemas, prompt templates, plugin
//! defaults, branch comparisons, intent lists, and code exception settings.
//! Python's `uuid`-generated ids are replaced by deterministic ids, and an
//! unknown node type is an error instead of a silently skipped node.
//!
//! [`PlatformWorkflow::to_value`] applies Python's null-dropping conversion, so
//! unset optionals leave no key behind, and
//! [`PlatformWorkflow::to_workflow_dsl`] adapts the document to the simplified
//! workflow DSL of [`crate::agent_builder::workflow_dsl`].
//!
//! [`PlatformWorkflow::to_value`]: platform::PlatformWorkflow::to_value
//! [`PlatformWorkflow::to_workflow_dsl`]: platform::PlatformWorkflow::to_workflow_dsl

pub mod convert;
pub mod model;
pub mod platform;

pub use convert::{DlTransformer, PluginResource};
pub use model::{DlCondition, DlDocument, DlNode, DlNodeType, DlParameters, DlVariable};
pub use platform::{PlatformEdge, PlatformNode, PlatformWorkflow};
