//! The model-facing `skill_index` tool: a read-only structured view of one
//! [`SkillFs`](crate::discovery::SkillFs) catalog.
//!
//! Ports Python `openjiuwen.symphony.discovery.directory_toolkit` — the `list`/`search`/
//! `read` execution engine over a virtual category tree, with request validation,
//! pagination cursors, pipelines, and output budgeting.
