//! Agent-facing tools over the capability-tree index.
//!
//! Ports the agent surface of Python `openjiuwen.symphony.agent` (v0.1.20): the browsing and
//! search tool a harness advertises to a model so it can explore the capability tree and read
//! the routing description of one skill, built on the query-time runtime in
//! [`crate::retrieval::search`].
//!
//! [`retrieval_toolkit`](crate::agent::retrieval_toolkit) holds that tool: a typed request/response contract, the model-facing
//! card and prompt snapshot, request validation, and deterministic paging.
//! Everything here is read-only over a loaded index — index building stays in
//! [`crate::retrieval::build`], and the toolkit answers with a typed abstain instead of a
//! fabricated hit when no request can be served.
pub mod retrieval_toolkit;

pub use retrieval_toolkit::{
    render_retrieval_prompt, root_cid, tool_input_schema, RetrievalEntry, RetrievalOperation,
    RetrievalPromptSnapshot, RetrievalRequest, RetrievalResponse, RetrievalToolCard,
    RetrievalToolkit, RetrievalToolkitConfig, BLANK_QUERY_SUGGESTION, NO_RESULTS_SUGGESTION,
    SKILL_RETRIEVAL_TOOL_NAME,
};
