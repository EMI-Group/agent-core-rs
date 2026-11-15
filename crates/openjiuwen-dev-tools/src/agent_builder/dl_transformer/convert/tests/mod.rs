//! Tests for the DL -> platform workflow transformer.
//!
//! Every test is offline: the documents are built in memory and no timestamp is
//! read, and the assertions compare whole JSON objects (Python's wire shape)
//! rather than serialized key order.
//! Generated intent ids are the one non-deterministic value, so they are
//! asserted by uuid4 shape and uniqueness instead of a fixed string.
//!
//! The suite is grouped by concern: `basics` (node types, ids, positions,
//! titles), `branches` and `intents` (branch and intent conversion),
//! `defaults` (questioner/code/output defaults), `plugins` (plugin parameter
//! resolution), `variables` (`_of_` splitting, references, output schemas), and
//! `serialization` (null-dropping, DSL validation, error paths).
//! Shared fixtures live in `support`.

mod support;

mod basics;
mod branches;
mod defaults;
mod intents;
mod plugins;
mod serialization;
mod variables;
