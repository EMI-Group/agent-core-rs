//! Agent-facing tools over the capability-tree index.
//!
//! Ports the agent surface of Python `openjiuwen.symphony.agent` (v0.1.20): the browsing and
//! search tool a harness advertises to a model so it can explore the capability tree and read
//! the routing description of one skill, built on the query-time runtime in
//! [`crate::retrieval::search`].
//!
//! [`retrieval_toolkit`] holds that tool: a typed request/response contract, the model-facing
//! card and prompt snapshot, request validation, and deterministic paging.
