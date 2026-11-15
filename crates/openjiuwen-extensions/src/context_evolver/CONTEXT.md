# context_evolver/ — memory-evolution schemas, prompts, and the service layer

## Intent

Python's `extensions/context_evolver/` (~66 files, 12.3k lines) mixes dependency-free pieces (schemas, algorithm names, JSON persistence, an in-memory cosine vector store, prompt strings) with LLM/embedding/Milvus-driven algorithm pipelines.
Rust ports the schemas, the prompt/reply layer, the persistence seam, the scoring seam (in process memory or in a kernel vector-store collection), and the whole `TaskMemoryService` contract, with model, persistence, embedding and vector-store access factored out behind traits.
Two services ship: `DeterministicTaskMemoryService` (offline, lexical ranking, extractive summarize) and `LlmTaskMemoryService` (prompt → completion → parse → persist over any injected `MemoryCompletion`).
`UnsupportedTaskMemoryService` stays as the documented deferred default for a configuration that defers the flow.
Pick this node for "memory algorithm names", "AddMemoryRequest", "prompt for ReMe/ACE/ReasoningBank/Cognition", "where is TaskMemoryService", or "how do memories persist".

## API Surface

```rust
pub enum Algo { Ace, ReasoningBank, Reme, Refcon, Divcon, Cognition }
impl Algo {
    pub fn as_str(&self) -> &'static str;      // "ace" | "reasoningbank" | "reme" | "refcon" | "divcon" | "cognition"
    pub fn family(&self) -> AlgoFamily;        // Ace | ReasoningBank | Reme (Refcon/Divcon map to Reme) | Cognition
}
/// Python `TaskMemoryService.normalize_algo_name`: case-insensitive, `rb`/`reasoning_bank` → ReasoningBank,
/// `ref_con`/`div_con` → Refcon/Divcon; unknown names are `Error::Validation`.
pub fn normalize_algo_name(name: &str) -> Result<Algo>;

pub struct AddMemoryRequest { content, query?, when_to_use?, title?, description?, section /* "general" */, label?, attributes }
pub struct RetrieveRequest { pub user_id: String, pub query: String }
pub struct RetrieveResponse { pub status: String, pub memory_string: String, pub retrieved_memory: Value }
pub struct SummarizeRequest { pub user_id: String, pub trajectories: Value }
pub struct SummarizeResponse { pub status: String, pub memories: Value }

#[async_trait] pub trait TaskMemoryService: Send + Sync {
    async fn retrieve(&self, request: RetrieveRequest) -> Result<RetrieveResponse>;
    async fn summarize(&self, request: SummarizeRequest) -> Result<SummarizeResponse>;
    async fn add_memory(&self, user_id: &str, request: AddMemoryRequest) -> Result<()>;
    fn load_memories(&self, user_id: &str) -> Result<Vec<AddMemoryRequest>>;   // synchronous by contract
}
pub struct UnsupportedTaskMemoryService;   // every method -> Error::Unsupported("<algo> pipeline is deferred; inject a TaskMemoryService")
pub struct MemoryVectorStore;              // in-memory cosine index; VectorNode { id, embedding, metadata }, ScoredNode { node, score }
#[async_trait] pub trait VectorIndex: Debug + Send + Sync {
    async fn upsert(&self, nodes: Vec<VectorNode>) -> Result<usize>;
    async fn search(&self, query: &[f32], top_k: usize) -> Result<Vec<ScoredNode>>;
    async fn search_with_metadata(&self, query: &[f32], top_k: usize, metadata: &BTreeMap<String, Value>) -> Result<Vec<ScoredNode>>;
    async fn search_ids(&self, query: &[f32], ids: &[String]) -> Result<Vec<ScoredNode>>;   // exactly the named nodes
    async fn clear(&self) -> Result<()>;
}
pub struct KernelVectorIndex;              // VectorIndex over a kernel VectorStore collection (`id`/`embedding`/`metadata` fields), cosine by default, any `Metric` via `with_metric`
pub enum VectorBackend { InMemory, VectorStore { collection: String } }   // TaskMemoryConfig::vector_backend, serde-tagged on "kind"
pub fn create_task_memory_service_with_vector_store(config, backend, store, vector_store) -> Result<Arc<dyn TaskMemoryService>>;
```

The subtree, one module per concern:

- `./mod.rs` — `Algo`, `AlgoFamily`, `normalize_algo_name`, the request/response cards, `TaskMemoryService`, `UnsupportedTaskMemoryService`, and the re-exports of everything below.
- `./record.rs` — `MemoryDraft`/`MemoryRecord` (the normalized, family-agnostic record), `identity`/`dedup_hash`/`validate`/`with_feedback`/`to_request`/`to_draft`, `render_block`, `generate_memory_id`.
- `./store.rs` — `MemoryStore` (`load`/`replace`/`upsert`/`remove`/`clear`), `InMemoryMemoryStore`, `JsonFileMemoryStore`.
- `./completion.rs` — `MemoryCompletion` seam, `ScriptedCompletion` test double, `OpenAiChatCompletion` HTTP adapter, `ChatCompletionConfig`, tolerant `parse_json_reply`.
- `./prompt/` — per-family prompt builders and reply parsers plus the shared dispatch (`retrieve_prompt`, `summarize_prompt`, `parse_retrieve`, `parse_summarize`, `render_memory_string`).
- `./memory_store.rs` — `VectorNode`, `ScoredNode`, the `VectorIndex` scoring seam, the in-process `MemoryVectorStore`, and the crate-internal unwired-index stub.
- `./kernel_vector_index.rs` — `KernelVectorIndex`: the `VectorIndex` over any kernel `VectorStore` collection.
- `./service/` — `TaskMemoryConfig` (+ `VectorBackend`), `MemoryEmbedding`/`HashingEmbedding`, `ranking`, `create_task_memory_service` / `create_task_memory_service_with_vector_store`, `DeterministicTaskMemoryService`, `LlmTaskMemoryService`.

## Ported Behavior

- `normalize_algo_name` ports Python's table exactly (case-insensitive, `-`/`_`/space tolerated): `ace` → Ace; `rb`, `reasoningbank`, `reasoning_bank` → ReasoningBank; `reme`, `refcon`, `ref_con`, `divcon`, `div_con` → Reme family (returning `Refcon`/`Divcon` for the explicit spellings); `cognition` → Cognition; anything else → `Error::Validation("unknown memory algorithm '<name>'")`.
- `MemoryVectorStore` ports `memory_vector_store.py`: cosine similarity, empty or mismatched vectors score `0.0`, descending order with stable ties, `top_k` clamped, metadata filtering by exact dotted-path equality.
- `VectorIndex` is the scoring seam both backends satisfy and report *raw* cosine similarities, so a service's ordering and blended scores do not depend on the backend; the in-process index orders ties by insertion, the store-backed one by primary key, and every service re-sorts through `finalize_hits` (score descending, id ascending) anyway.
- `KernelVectorIndex` keeps the nodes in any kernel `VectorStore` (`InMemoryVectorStore`, `LocalFileVectorStore`, `Elasticsearch`, or a vendor Milvus adapter registered with the kernel): it creates the collection lazily from the first upsert's embedding dimension, upserts by delete-then-insert (the kernel contract is insert-only), and converts each kernel score back to the raw similarity a service blends with. Dimension mismatches, unknown collections, and blank collection names are typed errors (`Error::Validation`/`Error::NotFound`), never silent.
- `KernelVectorIndex` accepts every kernel `Metric` (`new` is cosine; `with_metric` selects and `metric()` reports it), and the conversion keeps the score monotone in similarity per metric: cosine is `2 * score - 1` (the raw cosine in `[-1, 1]`), dot is `atanh(2 * score - 1)` (the inner product, saturated at `±atanh(1 - 2^-23) ≈ ±8.318` once the kernel's `tanh` compression runs out of `f32` precision), and Euclidean is the kernel score itself, `1 / (1 + distance)`, a bounded similarity in `(0, 1]`.
- The `VectorStore` contract carries no metric, so the index converts whatever score the store reports: the store must be configured to score with the same metric (a vendor backend configured with `Dot`/`Euclidean` does; the kernel's in-process and local-file backends always score cosine through the trait, so search those with `Metric::Cosine`). A mismatched configuration is the caller's, and it is documented on `KernelVectorIndex`.
- `TaskMemoryConfig.vector_backend` selects the index destination (`InMemory` by default); `DeterministicTaskMemoryService::with_vector_store` / `with_vector_index` wire it, and `create_task_memory_service_with_vector_store` does both. A `VectorStore`-selected configuration built without a store fails scoring with `Error::Unsupported` naming the collection and the builder — it never falls back to process memory.
- `MemoryRecord` normalizes Python's per-family classes (`TaskMemory`, `ReasoningBankMemory`, `PersonalMemory`) into one serde-tolerant shape; unknown fields are ignored and `null` equals absent.
- `create_task_memory_service` selects the backend: `Deterministic` (offline), `Llm` (model-driven), `Unsupported` (the documented deferral).
- `DeterministicTaskMemoryService` ranks lexically (`0.7 * lexical + 0.3 * cosine` when an embedder is configured) and summarizes extractively, with no model call.
- `LlmTaskMemoryService` runs prompt → completion → parse → persist; retrieve degrades to the pre-ranked order on an unparseable reply, summarize fails with `Error::Validation` naming the phase.
- Both services answer the same `{"memories": [{"id", "content", "score", "section", …}]}` card and share the score floor, `top_k`, and id generation.
- Not ported (documented): a vendor Milvus *client* (a deployment plugs a Milvus `VectorStore` into `KernelVectorIndex` instead; the seam itself, including dimension-mismatch and unknown-collection errors, is implemented), per-algorithm scoring pipelines beyond cosine, trajectory generation (MaTTS), the offline L2/L3 memory-bank builders, and the Wikipedia tool (blocking `requests`).

## Constraints

- No ML/vendor SDKs: model access goes through `MemoryCompletion`, embeddings through `MemoryEmbedding`, persistence through `MemoryStore`.
- Cosine math is `f32` with total-order sorting (`f32::total_cmp`), never `NaN` in results.
- State lives behind `Arc<RwLock<..>>`/`Arc<Mutex<..>>` and no guard is ever held across `.await`.
- Errors: `Error::Validation` (algorithm names, malformed nodes/replies, blank content, blank collection names, embedding-dimension mismatches), `Error::NotFound` (a store-backed search before the first upsert), `Error::Unsupported` (deferred backend, an unwired store-backed configuration, a vector-store factory handed the model-driven backend); no panics.
- Logging is `tracing` only; memory and trajectory contents never leave a frame above `debug`, and the retrieve fallback logs the error kind only.
- Files ≤ ~1000 lines; `prompt/mod.rs` (dispatch + shared parsers), `completion.rs` (HTTP adapter) and `service/llm.rs` run slightly over because each carries a dense inline test matrix — split them only when a new phase or family is added.

## Test Strategy

- Unit tests beside each module: `normalize_algo_name` (full table), `Algo::family`, the vector store, `KernelVectorIndex` (ranking parity against the in-process index, replace-by-id, dimension/empty-id/unknown-collection errors, dotted-path metadata, `search_ids`, `clear`, the cosine/dot/euclidean score conversions and their bounds, durability across a `LocalFileVectorStore` reopen), record validation/identity/feedback, the store backends (memory and JSON file), prompt parsing per family, ranking, and both services.
- `Service` tests use `ScriptedCompletion` + `InMemoryMemoryStore`, so every flow runs offline and deterministically; the factory tests cover each `MemoryBackend` arm.
- The crate-boundary suite `tests/context_evolver.rs` runs the deterministic file-store round trips (ordering, `top_k`, the on-disk `{"memories": [...]}` shape), the LLM path over a `wiremock` chat-completions mock (including malformed/fenced/hallucinated replies) and over one `ScriptedCompletion` service that both retrieves and summarizes, and the store-backed seam: ranking parity with the in-process index on one dataset, a service rebuilt over the same store, the dimension-mismatch / unknown-collection errors, the unchanged in-process default, and — through a test-local metric-configured store adapter over the kernel engine — each metric's ranking, raw score, `search_ids`/`search_with_metadata` behaviour and its blend inside the deterministic service.
- Python counterparts named in comments: `tests/unit_tests/extensions/context_evolver/` (`test_retrieve_flow.py`, `test_summary_flow.py`, `test_file_connector.py`), `service/task_memory_service.py`, `retrieve/task/*/run.py`, `summary/task/*/update.py`.

## Routing Table

- `./mod.rs` → `Algo`, `normalize_algo_name`, request/response cards, `TaskMemoryService`, `UnsupportedTaskMemoryService`
- `./record.rs` → `MemoryDraft`, `MemoryRecord`, `generate_memory_id`
- `./store.rs` → `MemoryStore`, `InMemoryMemoryStore`, `JsonFileMemoryStore`
- `./completion.rs` → `MemoryCompletion`, `ScriptedCompletion`, `OpenAiChatCompletion`, `parse_json_reply`
- `./prompt/` → `PromptPair`, `Trajectory`, `SummaryDraft`/`SummaryOperation`, `RetrievedHit`, per-family prompts and parsers
- `./memory_store.rs` → `VectorNode`, `ScoredNode`, `VectorIndex`, `MemoryVectorStore`
- `./kernel_vector_index.rs` → `KernelVectorIndex`, the `id`/`embedding`/`metadata` field names, kernel-score inversion
- `./service/` → `TaskMemoryConfig`, `VectorBackend`, ranking, `MemoryEmbedding`, `create_task_memory_service`, `create_task_memory_service_with_vector_store`, `DeterministicTaskMemoryService`, `LlmTaskMemoryService`
