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

## Constraints

- No vendor SDK or HTTP client code; `reqwest` is not used here.
- Errors: `Error::Unsupported` for the deferred transport; otherwise the inner reranker's error unchanged.
- Files ≤ ~1000 lines.

## Test Strategy

- Unit tests beside `mod.rs`: delegation forwards query/documents/`top_n` to the inner reranker and preserves its ordering and errors; `from_config` returns `ErrorKind::Unsupported`; `AliyunRerankerConfig` defaults and `Debug` redaction of `api_key`.
- Python counterpart named in a comment: `tests/unit_tests/core/retrieval/` (DashScope reranker; this node has no tests of its own in Python).

## Routing Table

- `./mod.rs` → `AliyunReranker`, `AliyunRerankerConfig`, delegation impl, tests
