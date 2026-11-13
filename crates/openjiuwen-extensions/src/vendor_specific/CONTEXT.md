# vendor_specific/ — deprecated vendor aliases

## Intent

Python's `extensions/vendor_specific/aliyun_reranker.py` is a deprecated backwards-compatibility shim: `AliyunReranker = DashscopeReranker`, logging a deprecation warning at import.
The reranking implementation lives in the kernel (`openjiuwen_core::retrieval::reranker`); this node keeps the alias shape and ships the real `DashScope` HTTP transport that the alias inherits in Python.
Pick this node only for "AliyunReranker" / `vendor_specific` compatibility requests.

## API Surface

```rust
pub struct AliyunRerankerConfig {
    pub api_key: Option<SecretString>,   // redacted in Debug
    pub api_base: String,                // DashScope-compatible endpoint, default "https://dashscope.aliyuncs.com"
    pub model_name: String,              // default "qwen3-reranker-0.6b"
    pub top_n: Option<usize>,
}
impl Default for AliyunRerankerConfig;

/// Deprecated alias kept for Python parity.
pub struct AliyunReranker { .. }
impl AliyunReranker {
    /// Wraps an already-configured reranker: pure delegation, `config() == None`.
    pub fn new(inner: Arc<dyn Reranker>) -> Self;
    /// Builds the live DashScope HTTP transport (`config() == Some(..)`).
    pub async fn from_config(config: AliyunRerankerConfig) -> Result<Self>;
    pub fn config(&self) -> Option<&AliyunRerankerConfig>;
}
#[async_trait] impl Reranker for AliyunReranker { /* delegates to inner */ }
```

`dashscope.rs` holds the transport internals, all `pub(crate)`: `DashScopeReranker`, `RerankRequest`, `RerankInput`, `RerankParameters`, `RerankResponse`, `RerankOutput`, `RerankResult`, plus the free helpers `endpoint_url`, `wire_top_n`, `map_results`, `status_error`, and `error_detail`.

## Transport Contract

- One `POST` per call to `{normalized api_base}/services/rerank/text-rerank/text-rerank` (Python's `end_point` constant) with `Content-Type: application/json` and `Authorization: Bearer {api_key}`.
- Body: `{"model": <model_name>, "input": {"query": <query>, "documents": [<text>, ...]}, "parameters": {"return_documents": false, "top_n": <n>}}`.
- Response: `{"output": {"results": [{"index": <usize>, "relevance_score": <f64>}, ...]}}`, with a top-level `results` fallback, matching Python's `response_data.get("output", response_data)["results"]`; an `output` that exists without `results` does not fall back.
- `api_base` normalization mirrors Python's `api_base.removesuffix("/").removesuffix(end_point)`: trim whitespace, strip one trailing `/`, strip one trailing `end_point`, then append it back, so bare hosts and full service URLs (with or without a trailing slash) resolve identically and a gateway path prefix survives.
- The credential resolves from `AliyunRerankerConfig::api_key` (a blank value counts as unset), then the `DASHSCOPE_API_KEY` environment variable via `openjiuwen_common::config::env::var`; a missing key is `Error::validation` naming both sources and never echoing a value.
- The HTTP client is built with a fixed 30 s timeout (`DEFAULT_TIMEOUT`); `AliyunRerankerConfig` has no timeout field, so the budget is not configurable.
- Serde: request and response structs are snake_case with unknown fields tolerated; `code` accepts strings or any other JSON value, and `code`/`message`/`request_id` are only used for error reporting.

## Rerank Semantics

- Empty `documents` returns `Ok(vec![])` with no HTTP request.
- Effective limit = the call's `top_n`, else `AliyunRerankerConfig::top_n`, else `documents.len()`.
- `Some(0)` and any limit `>= documents.len()` mean "no truncation"; this differs from kernel `sort_and_truncate`, where `Some(0)` truncates to empty.
- Wire `parameters.top_n` is the effective limit clamped to the document count, rendered as the document count when the effective limit is `0` because DashScope rejects `top_n = 0`.
- Results map onto the input by `index`, preserving `id`/`text`/`metadata` and overwriting only `score` with `relevance_score as f32`; ordering starts in input order, then a stable `f32::total_cmp` descending sort runs, so equal scores keep input order, and a limit strictly between `1` and the count truncates.
- A duplicated `index` keeps the last score (Python's dict assignment); a document absent from the response keeps `0.0` (Python's `dict.fromkeys(doc_ids, 0.0)`).
- An out-of-range `index` and a non-finite `relevance_score` are `Error::transport` naming the index (and the document count for the former); there is no panic and no silent `NaN`.

## Error Mapping

- `401`/`403` -> `Error::validation` naming `AliyunRerankerConfig::api_key`, `DASHSCOPE_API_KEY`, and the status; the body is never quoted because it can echo the key.
- `429`, `5xx`, and any other non-2xx -> `Error::transport` with the status plus `code`/`message`/`request_id` when the body carries them, falling back to a trimmed body snippet bounded to 300 characters.
- Malformed JSON, or a body with no `output.results` and no top-level `results` (`{"output": {}}` included) -> `Error::transport` with the same detail extraction.
- `reqwest` send failures -> `Error::transport`; a timeout names the 30 s budget and the underlying reason, and the API key and document texts are never logged or embedded (payload logging stays at `debug`).
- Client-build failure -> `Error::transport`; a missing credential, an empty `api_base`, an unparseable `api_base`, or a non-`http`/`https` scheme -> `Error::validation` naming the offending field.

## Remaining Differences From Python

- No `instruct` support, no `extra_body`, and no arbitrary `parameters` pass-through: the request body is the fixed schema above.
- No retry/backoff (`max_retries`, `retry_wait`) and no multimodal input; documents are plain strings, and inputs are `Vec<SearchResult>` rather than `str | Document`.
- Python's `DashscopeReranker` reads no environment variable; the `DASHSCOPE_API_KEY` fallback is an intentional addition.
- The timeout is a fixed 30 s; Python's `RerankerConfig.timeout` defaults to 10 s and is configurable.
- Python returns a `dict[str, float]`; this port returns the reranked `SearchResult`s, matching the kernel `Reranker` trait.

## Kernel Trait Contract

- The ported trait is `openjiuwen_retrieval::Reranker` (`crates/openjiuwen-retrieval/src/reranker/mod.rs`), `#[async_trait] pub trait Reranker: Send + Sync` with the single method `async fn rerank(&self, query: &str, documents: Vec<SearchResult>, top_n: Option<usize>) -> Result<Vec<SearchResult>>`.
- `SearchResult { id: String, text: String, score: f32, metadata: serde_json::Map<String, Value> }` is public, `Default`, and `PartialEq` (wire field names are not renamed).
- The trait is unsealed and object-safe, so `Arc<DashScopeReranker>` is stored as `Arc<dyn Reranker>` and `AliyunReranker` remains cheap to clone.

## Constraints

- `reqwest` (async, rustls) is the only HTTP client here; no vendor SDK, and no blocking calls on async tasks.
- Errors: `openjiuwen_common::error::{Error, Result}`; deferred or silent defaults are not acceptable.
- Logging: `tracing` only, target `openjiuwen_extensions::vendor_specific`; the API key and document texts never appear above `debug`.
- Size: `dashscope.rs` is intentionally ~1.1k lines because the transport and its wire/error matrix must live in this node's single write scope; move the `#[cfg(test)]` block into `dashscope/tests.rs` when a wider write scope allows it.

## Test Strategy

- Unit tests beside the code: `dashscope.rs`'s `#[cfg(test)] mod tests` covers the request shape (method, path, bearer token, exact JSON body), index mapping/sorting/tie stability, `top_n` precedence and the `0`/oversized no-truncation cases, missing documents scoring `0.0`, the empty-input short circuit (zero requests), `401`/`403`/`429`/`400`/`500` mapping, malformed and resultless bodies, out-of-range indices, non-finite scores, `endpoint_url` normalization and rejection, `DASHSCOPE_API_KEY` resolution under a process-wide `static Mutex<()>` env guard that restores the variable, serde request/response shapes, and `error_detail` bounding.
- `mod.rs`'s test module keeps the delegation tests (`new` forwards arguments, order, and errors verbatim, `config() == None`) plus the config defaults, `Debug` redaction, and snake_case serde tests, and proves `from_config` builds a working transport against a `wiremock` server.
- All tests are offline: `wiremock` binds `127.0.0.1:0`, no credentials exist, and no test touches the DashScope service.
- Python counterparts named in comments: `tests/unit_tests/core/retrieval/` (`DashscopeReranker`, `StandardReranker._parse_response`) and `tests/unit_tests/extensions/vendor_specific/` when present.

## Routing Table

- `./mod.rs` → `AliyunReranker`, `AliyunRerankerConfig`, the deprecation warning, delegation impl, and its tests
- `./dashscope.rs` → the live `DashScope` transport: wire types, request building, response mapping, credential and endpoint resolution, error mapping, and its tests
