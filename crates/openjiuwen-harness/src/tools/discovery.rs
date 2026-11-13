//! Progressive tool discovery: a BM25 index over tool cards and the `tool_search` tool.
//!
//! Ports `openjiuwen/harness/tools/tool_discovery/{tool_search,bm25}.py`. Deferred tools are
//! held by [`crate::tools::ToolRegistry`] and only advertised to the model after a search hit.
//!
//! # Required public API (registry contract)
//!
//! - `ToolIndex::from_cards(&[ToolCard]) -> ToolIndex`; `search(&self, query: &str, limit:
//!   usize) -> Vec<ToolSearchHit>` with `ToolSearchHit { name, score }` (descending score).
//! - `ToolSearchTool::new(index: Arc<ToolIndex>, language: impl Into<String>) -> Self` and
//!   `ToolSearchTool::card_for(language: &str) -> ToolCard`; model-facing name `tool_search`.
