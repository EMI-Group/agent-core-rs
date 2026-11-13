//! Web tools: `fetch_webpage` and `web_search`, plus in-crate HTML-to-text extraction.
//!
//! Ports `openjiuwen/harness/tools/web/` (`fetch_webpage.py`, `_http.py`, `_decode.py`,
//! `_quality.py`, `_assessment.py`, `_budget.py`, `free_search.py`, `paid_search.py`) as the
//! model-facing tool surface plus its injectable seams.
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
//!
//! # Design
//!
//! Both tools follow the crate's tool conventions: a `ToolCard` built from a `schemars` schema,
//! expected failures rendered as `ToolPayload::fail(...)` (never `Err`), `ToolContext::
//! check_cancelled` honoured on entry, `tracing` only and no process-global state.
//!
//! Network access is never reached directly from a tool: `fetch_webpage` goes through
//! [`HttpFetcher`] and `web_search` through [`SearchBackend`], so a host can inject proxies,
//! credentials, caches or fakes. [`ReqwestFetcher`] is the default implementation, built from
//! the workspace-pinned `reqwest` (rustls, no gzip/deflate decoding) with a ten-redirect cap, a
//! per-request timeout and a streaming body cap.
//!
//! `fetch_webpage` consults the permission engine through `ToolEnv::deny_reason`, which folds
//! the network guard pipeline for the fetch-tool aliases, so a `net_guard` deny rule blocks the
//! request before any socket is opened. `web_search` does not: a query is not a URL and the
//! provider is host-supplied.
//!
//! # Deviations from Python
//!
//! - No vendor search SDKs (Tavily/Serper/…), no DuckDuckGo/Bing HTML scrapers and no paid/free
//!   provider selection: a host injects one [`SearchBackend`].
//! - No LLM quality/assessment/budget pass, no challenge/WAF detection, no Jina-reader fallback
//!   and no proxy plumbing beyond what `reqwest` is configured with.
//! - No JavaScript rendering: the extractor sees the bytes the server sent.
//! - Charset support is UTF-8/ASCII, Latin-1/Windows-1252 and UTF-16; other labels decode as
//!   lossy UTF-8 instead of a full codec registry.
//! - The tool payload is the crate's six-key envelope with a JSON `data` object rather than the
//!   Python rendered text block.

mod decode;
pub mod fetch;
pub mod html;
pub mod search;

#[cfg(test)]
mod test_support;
#[cfg(test)]
mod tests;

pub use decode::{
    bom_charset, charset_from_content_type, decode_body, detect_charset, sniff_meta_charset,
    DEFAULT_CHARSET,
};
pub use fetch::{
    FetchRequest, FetchResponse, FetchWebpageArgs, FetchWebpageTool, HttpFetcher, ReqwestFetcher,
    DEFAULT_MAX_BYTES as FETCH_DEFAULT_MAX_BYTES, DEFAULT_MAX_CHARS,
    DEFAULT_TIMEOUT_SECONDS as FETCH_DEFAULT_TIMEOUT_SECONDS, DEFAULT_USER_AGENT,
    MAX_TIMEOUT_SECONDS, MIN_MAX_CHARS, MIN_TIMEOUT_SECONDS, REDIRECT_LIMIT,
    TOOL_NAME as FETCH_WEBPAGE_TOOL_NAME,
};
pub use html::{
    collapse_whitespace, decode_entities, extract, normalize_url, url_scheme, ExtractedHtml,
};
pub use search::{
    parse_search_response, FakeSearchBackend, FakeSearchCall, HttpSearchBackend, SearchBackend,
    SearchResult, WebSearchArgs, WebSearchTool, DEFAULT_MAX_BYTES as SEARCH_DEFAULT_MAX_BYTES,
    DEFAULT_MAX_RESULTS, DEFAULT_TIMEOUT_SECONDS as SEARCH_DEFAULT_TIMEOUT_SECONDS,
    MAX_MAX_RESULTS, MIN_MAX_RESULTS, TOOL_NAME as WEB_SEARCH_TOOL_NAME,
};
