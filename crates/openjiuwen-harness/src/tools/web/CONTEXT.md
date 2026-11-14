# tools/web

## Intent

The web tool surface of the `DeepAgent`: `fetch_webpage` (HTTP GET → charset decode → HTML-to-text) and `web_search` (a query → ranked `{title, url, snippet}` results), plus the in-crate HTML-to-text extractor and URL normaliser both rely on.
Ports `openjiuwen/harness/tools/web/` (`fetch_webpage.py`, `_http.py`, `_decode.py`, `free_search.py`, `paid_search.py`) as the model-facing contract and its injectable seams; provider, quality and budget internals are not transliterated.
`openjiuwen_harness::tools::web` is public (`tools/mod.rs` declares `pub mod web;`); `ToolRegistry` mounts the two tools as the extended builtins `fetch_webpage`/`web_search`, and a host may also construct them directly with their own seams.

## API Surface

### `mod.rs` — flattening surface

- Re-exports every public item of the four submodules, aliasing the clashing constants: `FETCH_WEBPAGE_TOOL_NAME`/`WEB_SEARCH_TOOL_NAME`, `FETCH_DEFAULT_MAX_BYTES`/`SEARCH_DEFAULT_MAX_BYTES`, `FETCH_DEFAULT_TIMEOUT_SECONDS`/`SEARCH_DEFAULT_TIMEOUT_SECONDS`.
- `FETCH_DEFAULT_MAX_CHARS`, `FETCH_DEFAULT_MAX_BYTES` (8 MiB), `FETCH_DEFAULT_TIMEOUT_SECONDS` (45), `MIN_MAX_CHARS` (500), `MIN_TIMEOUT_SECONDS` (5), `MAX_TIMEOUT_SECONDS` (600), `REDIRECT_LIMIT` (10), `DEFAULT_USER_AGENT`.
- `SEARCH_DEFAULT_MAX_BYTES` (1 MB), `SEARCH_DEFAULT_TIMEOUT_SECONDS` (20), `DEFAULT_MAX_RESULTS` (8), `MIN_MAX_RESULTS` (1), `MAX_MAX_RESULTS` (20).

### `fetch.rs` — `fetch_webpage`

- `#[async_trait] pub trait HttpFetcher: Send + Sync { async fn fetch(&self, request: FetchRequest) -> Result<FetchResponse>; }`.
- `FetchRequest { url: String, timeout: Duration, max_bytes: usize, headers: Vec<(String, String)> }`.
- `FetchResponse { status: u16, final_url: String, headers: Vec<(String, String)>, body: Vec<u8> }`, with `header(name) -> Option<&str>` (case-insensitive) and `content_type() -> &str`.
- `ReqwestFetcher::new()`, `with_user_agent(impl Into<String>)`, `from_client(reqwest::Client)`; `Default`; redirect cap 10, no gzip/deflate decoding, streaming body read stopped at `max_bytes`.
- `FetchWebpageTool::new(env: ToolEnv, fetcher: Option<Arc<dyn HttpFetcher>>, language: impl Into<String>)`, `card_for(language: &str) -> ToolCard`, `impl Tool`.
- `FetchWebpageArgs { url, max_chars, max_bytes, timeout_seconds, prompt, caption }` — serde-tolerant, `required = ["url"]`.
- Success payload: `data = {url, final_url, status, content_type, charset, content, truncated}` plus `title` for HTML and `prompt`/`caption` when given.
- Failures are `ToolPayload::fail(...)` (never `Err`): `url cannot be empty`, `unsupported URL scheme '<s>': only http and https are supported`, `invalid URL '<raw>': …`, `http fetcher is not configured`, `failed to fetch webpage: <error>`, `HTTP <status> fetching <url>: <excerpt>` (with `fail_with_data`), `unsupported content type '<media>': fetch_webpage only renders text, HTML and JSON responses`.

### `html.rs` — extraction and URL normalisation

- `ExtractedHtml { title: Option<String>, text: String }`; `extract(html: &str) -> ExtractedHtml`.
- `normalize_url(base: &str, href: &str) -> Option<String>`; `url_scheme(url: &str) -> Option<&str>`; `collapse_whitespace(text: &str) -> String`; `decode_entities(text: &str) -> String`.

### `decode.rs` — charset handling

- `detect_charset(content_type: &str, body: &[u8]) -> String` (BOM → header charset → `<meta charset>` sniff over 4 KiB → `utf-8`).
- `charset_from_content_type`, `sniff_meta_charset`, `decode_body(body, charset) -> String`, `bom_charset`, `DEFAULT_CHARSET`.

### `search.rs` — `web_search`

- `#[async_trait] pub trait SearchBackend: Send + Sync { async fn search(&self, query: &str, max_results: usize) -> Result<Vec<SearchResult>>; }`.
- `SearchResult { title, url, snippet }` (serde, snake_case) with `SearchResult::new`.
- `ENDPOINT_ENV` (`OPENJIUWEN_WEB_SEARCH_ENDPOINT`, blank counts as unset), `CONFIG_GUIDANCE` and `not_configured_reason()` (always starts with the historical `web search backend is not configured` prefix, then names the variable, the `{query}` placeholder, `HttpSearchBackend::new`, `WebSearchTool::with_endpoint` and `ToolRegistry::with_search_backend`).
- `HttpSearchBackend::new(endpoint)`, `with_fetcher(endpoint, Arc<dyn HttpFetcher>)`, `from_env()`, `from_lookup(impl Fn(&str) -> Option<String>)`, `is_configured()`, `with_timeout(Duration)`, `with_max_bytes(usize)`, `endpoint()`; `{query}` template or `?q=…&max_results=…` append. `from_env`/`from_lookup` return `Error::Unsupported` carrying `not_configured_reason()` when the value is missing or blank, and `search` does the same for a blank endpoint without issuing a request.
- `parse_search_response(text) -> Result<Vec<SearchResult>>`; `FakeSearchBackend` (`new`, `with_results`, `failing`, `set_results`, `set_failure`, `calls() -> Vec<FakeSearchCall>`, `last_call()`) and `FakeSearchCall { query, max_results }`.
- `WebSearchTool::new(backend: Option<Arc<dyn SearchBackend>>, language: impl Into<String>)`, `with_endpoint(endpoint, language)` (one-call `HttpSearchBackend` construction), `card_for(language: &str) -> ToolCard`, `impl Tool`; `WebSearchArgs { query, max_results }`.
- Success payload: `data = {"success": true, "results": [{title, url, snippet}]}`.
- Failures: `query cannot be empty`, `not_configured_reason()` rendered bare (no backend, a blank endpoint, or a provider that reports `Error::Unsupported`), and `web search failed: <kind>: <message>` for a configured provider that failed at runtime.

## Constraints

- Both tools implement `openjiuwen_foundation::Tool`; expected failures are `success=false` payloads, `Err` is reserved for cancellation (`ToolContext::check_cancelled`) and malformed argument types.
- No direct network access from a tool: `fetch_webpage` goes through `HttpFetcher`, `web_search` through `SearchBackend`; tests never touch the network beyond loopback.
- `fetch_webpage` consults `ToolEnv::deny_reason("fetch_webpage", args)` before any request, which folds the permission engine and network guard (pipeline C) for the fetch-tool aliases; `web_search` does not (a query is not a URL).
- Card descriptions come from `crate::tools::descriptions::description(name, language)` exactly like `shell.rs`; both rows are populated in English and Chinese.
- No new dependencies: HTML parsing, entity decoding, URL resolution and percent-encoding are in-crate; `reqwest` is the workspace-pinned build (rustls, `json`, `stream`).
- No panics on model input, no `unwrap`/`expect` outside tests, no `unsafe`, `tracing` only (no stdout/stderr), files ≲1000 lines.
- Deterministic tests only: `tempfile` workspaces, loopback `TcpListener` servers, injected fetchers, injected timeouts.

## Design decisions

- **Deviation — seams instead of engines**: Python selects vendors (DuckDuckGo/Bing scrapers, paid SDKs) and passes results through LLM quality/assessment/budget stages; Rust exposes one injectable backend per capability and renders whatever it returns.
- **Deviation — configuration gap is typed**: Python raises when no search provider is available; here the gap is `Error::Unsupported(not_configured_reason())`, reported bare by the tool, while `OPENJIUWEN_WEB_SEARCH_ENDPOINT` (or `HttpSearchBackend::new`/`WebSearchTool::with_endpoint`) opens a working JSON search endpoint without a host implementation.
- **Deviation — no JS rendering / reader proxy**: the extractor sees the served bytes, so a JavaScript shell yields the shell's text; the Jina reader fallback is not ported.
- **Deviation — charset coverage**: UTF-8/ASCII, Latin-1/Windows-1252 and UTF-16 decode exactly; every other label decodes as lossy UTF-8 instead of a full codec registry.
- **Deviation — `max_bytes` argument**: Python has only the `MCP_FETCH_WEBPAGE_MAX_BYTES` environment cap; the tool accepts a per-call `max_bytes` (clamped to `1..=8 MiB`) so oversized-body behaviour is testable offline.
- **Deviation — payload shape**: Python returns a rendered text block / `[ERROR]: …` string; here `data` is a structured JSON object inside the crate's six-key envelope.
- **Deviation — URL normalisation**: in-crate resolution replaces `urllib.parse` (no `url` dependency); user info survives verbatim, `%2e%2e` is not decoded, and a dotted `host:port` prefix is treated as a host rather than a scheme.
- **Body truncation detection**: `FetchResponse` carries no `truncated` flag, so the tool treats a body whose length equals `max_bytes` as capped unless `Content-Length` says the body was complete.
- **JSON passthrough**: `application/json`/`+json` decode to pretty-printed text when the payload parses, otherwise the raw text is passed through.
- **`html::extract` shape**: script/style/noscript/template/svg/canvas/iframe bodies, comments, CDATA and doctypes are dropped; `<title>` is captured separately; block elements emit at most one blank line; `<pre>` keeps raw whitespace; entities decode from a fixed named table plus numeric decimal/hex.

## Test Strategy

- `html.rs` unit tests: script/style/comment removal, first-title capture, block newlines, whitespace collapse, `<pre>` preservation, entity decoding (named, decimal, hex, unknown, numeric zero), malformed/unclosed markup, doctype/CDATA/PI skipping, quoted `>` in attributes, and URL resolution/rejection cases.
- `decode.rs` unit tests: header charset (quoted and bare), meta sniffing (both declaration forms), BOM precedence, lossy UTF-8, Latin-1/Windows-1252, UTF-16 LE/BE and unknown-label fallback.
- `fetch.rs` unit tests: card schema/`required`, limit clamping, URL validation and scheme rejection, HTML/JSON/text rendering, binary rejection, header charset, HTTP and transport failures, missing fetcher, argument errors, permission denial (fetcher not called), content and body truncation, prompt/caption echo, request headers/timeout/byte cap.
- `search.rs` unit tests: card schema, result-count clamping, percent encoding, object/array payload parsing, alternate field names, malformed payloads, request URL building (template and query append), HTTP and JSON failures, payload shape, backend failure, missing backend, fake-backend call recording, `not_configured_reason` content, `from_lookup` trimming plus unset/blank rejection, a blank endpoint issuing no request, and the tool rendering a configuration gap differently from a runtime failure.
- `tests.rs` cross-module suite (74 tests total across the subtree): loopback HTML extraction with title, JSON pretty-printing, header and meta charsets, redirect following, byte-cap truncation, real `reqwest` timeout, connection-refused failure payload, timeout payload, net-guard denial, fake search round-trip, missing backend, the `not_configured_reason` payload naming `OPENJIUWEN_WEB_SEARCH_ENDPOINT`, an endpoint configured through `from_lookup` against a loopback server, `WebSearchTool::with_endpoint` with a blank endpoint, and an HTTP search backend round-trip asserting the exact request line.
- `test_support.rs` supplies the shared `StubFetcher` (canned response/failure, call count, last request), `fetch_response`, `http_response` and a loopback `TestServer` aborted on drop; all helpers are `#[cfg(test)]`.
- Definition of done: `cargo fmt -p openjiuwen-harness`, `cargo test -p openjiuwen-harness --offline`, `cargo test -p openjiuwen-harness --no-default-features --offline`, `cargo clippy -p openjiuwen-harness --all-targets --offline -- -D warnings` clean.

## Deferred Python behavior

- `free_search.py`/`paid_search.py`: DuckDuckGo/Bing HTML scraping, vendor API clients, provider fallback, redirect unwrapping, domestic academic domain switching, relevance assessment and the search budget.
- `fetch_webpage.py`: Jina-reader fallback, proxy configuration, web-research budget, challenge/WAF/login detection, dynamic-shell detection, PDF/Office readers and the `_quality.py` content-status protocol.
- `_assessment.py`/`_budget.py`: LLM-driven result assessment and the shared research budget (host concern behind the seams).

## Routing Table

- `./mod.rs` → crate-internal module declarations, flat re-exports (aliased constants), design/deviations documentation
- `./fetch.rs` → `HttpFetcher`/`ReqwestFetcher`, `FetchRequest`/`FetchResponse`, `fetch_webpage` tool and its payload rendering
- `./html.rs` → `extract`/`ExtractedHtml`, `normalize_url`/`url_scheme`, entity decoding, whitespace collapse
- `./decode.rs` → charset detection (BOM/header/meta) and lossy body decoding
- `./search.rs` → `SearchBackend`/`SearchResult`, `HttpSearchBackend`, `FakeSearchBackend`, `web_search` tool
- `./test_support.rs` → test-only `StubFetcher`, response builders and the loopback `TestServer`
- `./tests.rs` → cross-module offline suite (loopback HTTP, charsets, redirects, truncation, guards, search seams)
