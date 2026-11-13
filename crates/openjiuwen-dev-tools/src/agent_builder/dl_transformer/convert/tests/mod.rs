//! Tests for the DL -> platform workflow transformer.
//!
//! Every test is offline and deterministic: the documents are built in memory,
//! no id or timestamp is generated, and the assertions compare whole JSON
//! objects (Python's wire shape) rather than serialized key order.
//!
//! The suite is grouped by concern: `basics` (node types, ids, positions,
//! titles), `branches` and `intents` (branch and intent conversion),
//! `defaults` (plugin/questioner/code/output defaults), `variables` (`_of_`
//! splitting, references, output schemas), and `serialization` (null-dropping,
//! DSL validation, error paths). Shared fixtures live in `support`.

mod support;

mod basics;
mod branches;
mod defaults;
mod intents;
mod serialization;
mod variables;
