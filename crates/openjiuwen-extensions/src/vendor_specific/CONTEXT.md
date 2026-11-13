# vendor_specific/ — deprecated vendor aliases

## Intent

Python's `extensions/vendor_specific/aliyun_reranker.py` is a deprecated backwards-compatibility shim: `AliyunReranker = DashscopeReranker`, logging a deprecation warning at import.
The real reranker lives in the kernel (`openjiuwen_retrieval::reranker`). Rust keeps only the alias shape: an `AliyunReranker` that adapts an injected `Reranker`, plus the DashScope config surface, whose HTTP transport is deferred (`Error::Unsupported`).
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
    /// Wraps an already-configured reranker (the seam hosts use today).
    pub fn new(inner: Arc<dyn Reranker>) -> Self;
    /// Builds the DashScope HTTP reranker: always `Error::Unsupported` today.
    pub async fn from_config(config: AliyunRerankerConfig) -> Result<Self>;
    pub fn config(&self) -> Option<&AliyunRerankerConfig>;
}
#[async_trait] impl Reranker for AliyunReranker { /* delegates to the injected reranker */ }
```

## Ported Behavior

- The Python alias has no logic of its own; the only behavior is delegation to a DashScope-compatible reranker, so the Rust adapter delegates to `Reranker::rerank` unchanged (order, `top_n` semantics, and errors come from the inner implementation).
- The DashScope HTTP transport (`AliyunReranker::from_config`) is deferred: `Error::Unsupported("AliyunReranker HTTP transport is deferred; wrap a configured reranker with AliyunReranker::new")`.
- Deprecation is expressed in docs and an optional `tracing::warn!` on construction (target `openjiuwen_extensions::vendor_specific`), matching Python's import-time warning without logging on every call.

## Kernel Trait Contract

- The ported trait is `openjiuwen_retrieval::Reranker` (`crates/openjiuwen-retrieval/src/reranker/mod.rs`), `#[async_trait] pub trait Reranker: Send + Sync` with the single method `async fn rerank(&self, query: &str, documents: Vec<SearchResult>, top_n: Option<usize>) -> Result<Vec<SearchResult>>`.
- It is unsealed and object-safe (the kernel stores it as `Option<Arc<dyn Reranker>>` in `knowledge_base`), so a DashScope transport can implement it here using public API only.
- `SearchResult { id: String, text: String, score: f32, metadata: serde_json::Map<String, Value> }` is public, `Default`, and `Deserialize` (wire field names are not renamed), so provider JSON can be mapped straight onto it.
- There is no `RerankRequest`/`RerankResponse` type in the kernel: `Vec<SearchResult>` in/out plus `top_n` is the whole protocol.
- `Reranker::rerank`'s contract is a stable descending `f32::total_cmp` sort with per-call `top_n` winning over any configured limit, `Some(0)` → empty; the kernel's `sort_and_truncate` helper is private, so an implementation here must replicate that ordering itself or delegate to `ScoreReranker`/`LlmReranker`.
- A transport can alternatively implement `openjiuwen_retrieval::LlmScorer` (`async fn score(&self, query: &str, documents: &[String]) -> Result<Vec<f32>>`) and wrap it in `LlmReranker`, which supplies the sorting, truncation, length-mismatch, and non-finite-score validation.
- The kernel also exposes `RerankerConfig { api_key: Option<SecretString>, api_base, model_name (alias "model"), timeout, temperature, top_p }`; `AliyunRerankerConfig` here is the local, smaller shape and does not reference it.

## Constraints

- No vendor SDK or HTTP client code; `reqwest` is not used here.
- Errors: `Error::Unsupported` for the deferred transport; otherwise the inner reranker's error unchanged.
- Files ≤ ~1000 lines.

## Test Strategy

- Unit tests beside `mod.rs`: delegation forwards query/documents/`top_n` to the inner reranker and preserves its ordering and errors; `from_config` returns `ErrorKind::Unsupported`; `AliyunRerankerConfig` defaults and `Debug` redaction of `api_key`.
- Python counterpart named in a comment: `tests/unit_tests/core/retrieval/` (DashScope reranker; this node has no tests of its own in Python).

## Routing Table

- `./mod.rs` → `AliyunReranker`, `AliyunRerankerConfig`, delegation impl, tests
