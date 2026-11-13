//! Web tools: `fetch_webpage` and `web_search`, plus in-crate HTML-to-text extraction.
//!
//! Ports `openjiuwen/harness/tools/web/` (`fetch_webpage.py`, `_http.py`, `_decode.py`,
//! `_quality.py`, `_assessment.py`, `_budget.py`, `free_search.py`, `paid_search.py`).
//!
//! # Required public API (registry contract)
//!
//! - `HttpFetcher` (async trait, `Send + Sync`): `fetch(&self, request: FetchRequest) ->
//!   Result<FetchResponse>`; `FetchRequest` carries the URL and the limits.
//! - `ReqwestFetcher` — the default `HttpFetcher` over `reqwest` with rustls.
//! - `FetchWebpageTool::new(env: ToolEnv, fetcher: Option<Arc<dyn HttpFetcher>>,
//!   language: impl Into<String>) -> Self` and `FetchWebpageTool::card_for(language: &str) ->
//!   ToolCard`; the model-facing name is `fetch_webpage`.
//! - `SearchBackend` (async trait): `search(&self, query: &str, max_results: usize) ->
//!   Result<Vec<SearchResult>>`; `SearchResult { title, url, snippet }`.
//! - `WebSearchTool::new(backend: Option<Arc<dyn SearchBackend>>, language: impl
//!   Into<String>) -> Self` and `WebSearchTool::card_for(language: &str) -> ToolCard`; the
//!   model-facing name is `web_search`.
//! - `html` submodule — HTML-to-text extraction and URL normalisation with unit tests.
