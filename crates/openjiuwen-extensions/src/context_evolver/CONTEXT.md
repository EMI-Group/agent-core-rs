# context_evolver/ — memory-evolution schemas, vector store, and the service seam

## Intent

Python's `extensions/context_evolver/` (~66 files, 12.3k lines) mixes dependency-free pieces (schemas, algorithm names, JSON persistence, an in-memory cosine vector store, prompt strings) with LLM/embedding/Milvus-driven algorithm pipelines.
Rust ports the dependency-free core and models the rest behind a trait, so a host can plug any `Model`/`Embedding`/persistence backend: the bundled implementation answers `Error::Unsupported` for the LLM-driven flows.
Pick this node for "memory algorithm names", "AddMemoryRequest", "in-memory memory vector store", or "where is TaskMemoryService".

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

pub struct AddMemoryRequest {
    pub content: String,
    pub query: Option<String>,
    pub when_to_use: Option<String>,
    pub title: Option<String>,
    pub description: Option<String>,
    pub section: String,                 // "general"
    pub label: Option<String>,
    pub attributes: BTreeMap<String, Value>,
}

pub struct VectorNode { pub id: String, pub embedding: Vec<f32>, pub metadata: BTreeMap<String, Value> }
pub struct MemoryVectorStore { .. }      // in-memory cosine index
impl MemoryVectorStore {
    pub fn new() -> Self;
    pub async fn upsert(&self, nodes: Vec<VectorNode>) -> Result<usize>;
    pub async fn search(&self, query: &[f32], top_k: usize) -> Result<Vec<ScoredNode>>;
    pub async fn search_with_metadata(&self, query: &[f32], top_k: usize, metadata: &BTreeMap<String, Value>) -> Result<Vec<ScoredNode>>;
    pub async fn len(&self) -> usize;
    pub async fn is_empty(&self) -> bool;
    pub async fn clear(&self);
}
impl Default for MemoryVectorStore;   // same as `new()`
pub struct ScoredNode { pub node: VectorNode, pub score: f32 }   // raw cosine in [-1, 1], higher first; non-finite scores clamp to 0.0

pub struct RetrieveRequest { pub user_id: String, pub query: String }
pub struct RetrieveResponse { pub status: String, pub memory_string: String, pub retrieved_memory: Value }
pub struct SummarizeRequest { pub user_id: String, pub trajectories: Value }
pub struct SummarizeResponse { pub status: String, pub memories: Value }

#[async_trait] pub trait TaskMemoryService: Send + Sync {
    async fn retrieve(&self, request: RetrieveRequest) -> Result<RetrieveResponse>;
    async fn summarize(&self, request: SummarizeRequest) -> Result<SummarizeResponse>;
    async fn add_memory(&self, user_id: &str, request: AddMemoryRequest) -> Result<()>;
    fn load_memories(&self, user_id: &str) -> Result<Vec<AddMemoryRequest>>;
}
pub struct UnsupportedTaskMemoryService { /* algorithm */ }
impl UnsupportedTaskMemoryService { pub fn new(algo: Algo) -> Self; pub fn algo(&self) -> Algo; }
// every method -> Error::Unsupported("context_evolver '<algo>' pipeline is deferred; inject a TaskMemoryService")
```

## Ported Behavior

- `normalize_algo_name` ports Python's table exactly (case-insensitive, `-`/`_`/space tolerated): `ace` → Ace; `rb`, `reasoningbank`, `reasoning_bank` → ReasoningBank; `reme`, `refcon`, `ref_con`, `divcon`, `div_con` → Reme family (returning `Refcon`/`Divcon` for the explicit spellings); `cognition` → Cognition; anything else → `Error::Validation("unknown memory algorithm '<name>'")`.
- `MemoryVectorStore` ports `memory_vector_store.py`: cosine similarity (`dot / (‖a‖·‖b‖)`), empty or mismatched vectors score `0.0` (empty vs empty `1.0`), results sorted descending with stable ties, `top_k` clamped to the index size; metadata filtering matches exact dotted-path equality.
- `AddMemoryRequest` mirrors the Python dataclass defaults (`section = "general"`, everything else `None`/empty).
- Memory serialization is snake_case with `#[serde(default)]` on the struct and `#[serde(default = "default_section")]` on `section`; a missing field, an explicit `null`, and an unknown field are all accepted, and `attributes` maps `null`/absent to an empty `BTreeMap`.
- `attributes` is a `BTreeMap<String, Value>` via a custom `deserialize_attributes` (accepts missing map, explicit `null`, or an object), so the JSON persistence shape `{"memories": [...]}` round-trips regardless of which optional keys a stored memory carries.
- `Algo` and `AlgoFamily` serialize `snake_case` (`reasoning_bank`, `reme`, …) and support `Hash`/`Copy`, so both work as map keys in host code.
- `TaskMemoryService` is public and unsealed: a host implements it in its own crate to supply the LLM/embedding/persistence pipelines, and `UnsupportedTaskMemoryService` is only the deferred default.
- Ported as pure data: `schema/io_schema.py` response cards (`RetrieveResponse`, `SummarizeResponse`), algorithm names, and the JSON persistence shape (`{"memories": [...]}`).
- Not ported (documented, `Error::Unsupported`): LLM-driven retrieve/summarize pipelines, embedding calls, Milvus persistence, trajectory generation (MaTTS), offline L2/L3 memory-bank builders, and the Wikipedia tool (blocking `requests`).

## Constraints

- No ML/vendor SDKs: any model, embedding, or vector persistence access goes through the `TaskMemoryService` trait or the in-memory store.
- Cosine math is `f32` with total-order sorting (`f32::total_cmp`), never `NaN` in results.
- State lives behind `Arc<RwLock<..>>` (never held across `.await`).
- Errors: `Error::Validation` (algorithm names, malformed nodes), `Error::Unsupported` (deferred pipelines); no panics.
- Files ≤ ~1000 lines.

## Test Strategy

- Unit tests beside `mod.rs`: the full `normalize_algo_name` table including unknown → `Validation`; `Algo::family` mapping; `MemoryVectorStore` ordering, `top_k` clamping, empty/mismatched vectors, metadata filter, upsert replacement by id.
- `UnsupportedTaskMemoryService` returns `ErrorKind::Unsupported` from every method and names the algorithm.
- Python counterparts named in comments: `tests/unit_tests/extensions/context_evolver/test_file_connector.py`, `test_retrieve_flow.py` (normalization table only; live flows skipped in Python).

## Routing Table

- `./mod.rs` → `Algo`, `normalize_algo_name`, request/response cards, `TaskMemoryService`, `UnsupportedTaskMemoryService`
- `./memory_store.rs` → `VectorNode`, `MemoryVectorStore`, `ScoredNode`
