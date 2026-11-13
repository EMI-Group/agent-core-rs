# context_evolver/ — memory-evolution schemas, prompts, and the service layer

## Intent

Python's `extensions/context_evolver/` (~66 files, 12.3k lines) mixes dependency-free pieces (schemas, algorithm names, JSON persistence, an in-memory cosine vector store, prompt strings) with LLM/embedding/Milvus-driven algorithm pipelines.
Rust ports the schemas, the prompt/reply layer, the persistence seam, and the whole `TaskMemoryService` contract, with the model and persistence access factored out behind traits.
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
```

The subtree, one module per concern:

- `./mod.rs` — `Algo`, `AlgoFamily`, `normalize_algo_name`, the request/response cards, `TaskMemoryService`, `UnsupportedTaskMemoryService`, and the re-exports of everything below.
- `./record.rs` — `MemoryDraft`/`MemoryRecord` (the normalized, family-agnostic record), `identity`/`dedup_hash`/`validate`/`with_feedback`/`to_request`/`to_draft`, `render_block`, `generate_memory_id`.
- `./store.rs` — `MemoryStore` (`load`/`replace`/`upsert`/`remove`/`clear`), `InMemoryMemoryStore`, `JsonFileMemoryStore`.
- `./completion.rs` — `MemoryCompletion` seam, `ScriptedCompletion` test double, `OpenAiChatCompletion` HTTP adapter, `ChatCompletionConfig`, tolerant `parse_json_reply`.
- `./prompt/` — per-family prompt builders and reply parsers plus the shared dispatch (`retrieve_prompt`, `summarize_prompt`, `parse_retrieve`, `parse_summarize`, `render_memory_string`).
- `./memory_store.rs` — `VectorNode`, `MemoryVectorStore`, `ScoredNode`.
- `./service/` — `TaskMemoryConfig`, `MemoryEmbedding`/`HashingEmbedding`, `ranking`, `create_task_memory_service`, `DeterministicTaskMemoryService`, `LlmTaskMemoryService`.

## Ported Behavior

- `normalize_algo_name` ports Python's table exactly (case-insensitive, `-`/`_`/space tolerated): `ace` → Ace; `rb`, `reasoningbank`, `reasoning_bank` → ReasoningBank; `reme`, `refcon`, `ref_con`, `divcon`, `div_con` → Reme family (returning `Refcon`/`Divcon` for the explicit spellings); `cognition` → Cognition; anything else → `Error::Validation("unknown memory algorithm '<name>'")`.
- `MemoryVectorStore` ports `memory_vector_store.py`: cosine similarity, empty or mismatched vectors score `0.0`, descending order with stable ties, `top_k` clamped, metadata filtering by exact dotted-path equality.
- `MemoryRecord` normalizes Python's per-family classes (`TaskMemory`, `ReasoningBankMemory`, `PersonalMemory`) into one serde-tolerant shape; unknown fields are ignored and `null` equals absent.
- `create_task_memory_service` selects the backend: `Deterministic` (offline), `Llm` (model-driven), `Unsupported` (the documented deferral).
- `DeterministicTaskMemoryService` ranks lexically (`0.7 * lexical + 0.3 * cosine` when an embedder is configured) and summarizes extractively, with no model call.
- `LlmTaskMemoryService` runs prompt → completion → parse → persist; retrieve degrades to the pre-ranked order on an unparseable reply, summarize fails with `Error::Validation` naming the phase.
- Both services answer the same `{"memories": [{"id", "content", "score", "section", …}]}` card and share the score floor, `top_k`, and id generation.
- Not ported (documented): Milvus persistence, per-algorithm vector-store scoring, trajectory generation (MaTTS), the offline L2/L3 memory-bank builders, and the Wikipedia tool (blocking `requests`).

## Constraints

- No ML/vendor SDKs: model access goes through `MemoryCompletion`, embeddings through `MemoryEmbedding`, persistence through `MemoryStore`.
- Cosine math is `f32` with total-order sorting (`f32::total_cmp`), never `NaN` in results.
- State lives behind `Arc<RwLock<..>>`/`Arc<Mutex<..>>` and no guard is ever held across `.await`.
- Errors: `Error::Validation` (algorithm names, malformed nodes/replies, blank content), `Error::Unsupported` (deferred backend); no panics.
- Logging is `tracing` only; memory and trajectory contents never leave a frame above `debug`, and the retrieve fallback logs the error kind only.
- Files ≤ ~1000 lines; `prompt/mod.rs` (dispatch + shared parsers), `completion.rs` (HTTP adapter) and `service/llm.rs` run slightly over because each carries a dense inline test matrix — split them only when a new phase or family is added.

## Test Strategy

- Unit tests beside each module: `normalize_algo_name` (full table), `Algo::family`, the vector store, record validation/identity/feedback, the store backends (memory and JSON file), prompt parsing per family, ranking, and both services.
- `Service` tests use `ScriptedCompletion` + `InMemoryMemoryStore`, so every flow runs offline and deterministically; the factory tests cover each `MemoryBackend` arm.
- The crate-boundary suite `tests/context_evolver.rs` runs the deterministic file-store round trips (ordering, `top_k`, the on-disk `{"memories": [...]}` shape) and the LLM path over a `wiremock` chat-completions mock, including malformed/fenced/hallucinated replies.
- Python counterparts named in comments: `tests/unit_tests/extensions/context_evolver/` (`test_retrieve_flow.py`, `test_summary_flow.py`, `test_file_connector.py`), `service/task_memory_service.py`, `retrieve/task/*/run.py`, `summary/task/*/update.py`.

## Routing Table

- `./mod.rs` → `Algo`, `normalize_algo_name`, request/response cards, `TaskMemoryService`, `UnsupportedTaskMemoryService`
- `./record.rs` → `MemoryDraft`, `MemoryRecord`, `generate_memory_id`
- `./store.rs` → `MemoryStore`, `InMemoryMemoryStore`, `JsonFileMemoryStore`
- `./completion.rs` → `MemoryCompletion`, `ScriptedCompletion`, `OpenAiChatCompletion`, `parse_json_reply`
- `./prompt/` → `PromptPair`, `Trajectory`, `SummaryDraft`/`SummaryOperation`, `RetrievedHit`, per-family prompts and parsers
- `./memory_store.rs` → `VectorNode`, `MemoryVectorStore`, `ScoredNode`
- `./service/` → `TaskMemoryConfig`, ranking, `MemoryEmbedding`, `create_task_memory_service`, `DeterministicTaskMemoryService`, `LlmTaskMemoryService`
