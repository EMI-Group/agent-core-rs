# openjiuwen-retrieval

## Intent

Documents, chunking, embeddings, vector stores, rerankers, and knowledge base for the Rust rewrite.
Maps to the Python package `openjiuwen.core.retrieval` (public surface in `openjiuwen/core/retrieval/__init__.py`), with the document/result/config models from `retrieval/common/` and the foundation store ABCs (`Document`, `Embedding`, `Reranker`, `VectorStore`).
Scope is the offline-capable core: text/JSON/Markdown/CSV/HTML/URL loading, character/recursive/token chunking, an OpenAI-compatible embedding client plus a deterministic fake, an in-memory vector store with optional JSON persistence, score/LLM reranking seams, and a knowledge base that runs load → chunk → embed → store → retrieve → rerank.
External vector databases (Milvus, Chroma, PGVector, Elasticsearch), PDF/DOCX/XLSX parsing, triple/graph extraction, multimodal documents, query rewriting, sparse/BM25 and agentic retrievers are deferred behind traits or return `Error::Unsupported` (see `docs/PARITY.md`).
Binary documents need a container layer before they can be parsed, so the crate now ships an in-crate `DEFLATE`/`zlib` decoder and ZIP reader (`src/parsing/`) with no compression dependency; the DOCX/XLSX/PDF-to-`Document` parsers themselves are still deferred.

## API Surface

Module root `openjiuwen_retrieval`; all public items are re-exported from `lib.rs`.

### Data model (`common`)
```rust
pub type Metadata = serde_json::Map<String, serde_json::Value>;
pub struct Document { pub id: String, pub text: String, pub metadata: Metadata }
// JSON keys: id_ (alias id), text, metadata; id defaults to a dashed uuid4.
pub struct TextChunk { pub id: String, pub text: String, pub doc_id: String,
                       pub metadata: Metadata, pub embedding: Option<Vec<f32>> }
// JSON keys: id_, text, doc_id, metadata, embedding (null when absent).
pub struct SearchResult { pub id: String, pub text: String, pub score: f32, pub metadata: Metadata }
pub struct RetrievalResult { pub text: String, pub score: f32, pub metadata: Metadata,
                             pub doc_id: Option<String>, pub chunk_id: Option<String> }
pub struct MultiKBRetrievalResult { pub text: String, pub score: f32, pub raw_score: f32,
                                    pub raw_score_scaled: f32, pub kb_ids: Vec<String>,
                                    pub metadata: Metadata }
```
Constructors: `Document::new(text)` (generated id), `Document::with_id(id, text)`, `TextChunk::from_document(&Document, text, id: Option<String>)` (copies metadata and `doc_id`), `TextChunk::with_embedding`.
`MetadataFilter` (serde from/to JSON): plain-object equality (`{"author":"alice"}`), per-field operators (`{"year":{"$gte":2024}}`), and logical operators (`$and`, `$or`, `$not`), with `Eq/Ne/Gt/Gte/Lt/Lte/In/Nin/Exists/Contains/And/Or/Not` variants and `MetadataFilter::matches(&Metadata) -> bool`.
Config types (snake_case, defaults preserved from Python): `KnowledgeBaseConfig` (`kb_id`, `index_type="hybrid"`, `use_graph=false`, `chunk_size=512`, `chunk_overlap=50`, `use_caption_for_images=false`), `RetrievalConfig` (`top_k=5`, `score_threshold=None`, `use_graph=None`, `agentic=false`, `graph_expansion=false`, `filters=None`), `IndexConfig` (`index_name`, `index_type="hybrid"`, `use_caption_for_images=false`), `VectorStoreConfig` (`store_provider`, `database_name`, `collection_name`, `distance_metric="cosine"`), `EmbeddingConfig` (`model_name`, `base_url`, optional `api_key: SecretString`, `timeout=60s`, `max_retries=3`, `max_batch_size=8`, `headers`, `dimension`), `RerankerConfig` (`api_key`, `api_base`, `model_name` with alias `model`, `timeout=10s`, `temperature=0.95`, `top_p=0.95`).
Constants/defaults: `DEFAULT_TOP_K = 5`, `DEFAULT_CHUNK_SIZE = 512`, `DEFAULT_CHUNK_OVERLAP = 50`, index types `"vector" | "bm25" | "hybrid"`.

### Chunking (`chunking`)
```rust
pub trait Chunker: Send + Sync {
    fn chunk_text(&self, text: &str) -> Result<Vec<String>>;
    fn chunk_documents(&self, documents: &[Document]) -> Result<Vec<TextChunk>>; // default impl
    fn chunk_size(&self) -> usize;
    fn chunk_overlap(&self) -> usize;
}
```
`CharChunker::new(chunk_size, chunk_overlap) -> Result<Self>` — Python `CharChunker`/`CharSplitter` semantics: `step = size - overlap`, overlap clamped to `[0, size-1]`, chunks never exceed `size`, count = `ceil(len/step)`, whitespace preserved. Default 512/50.
`RecursiveCharChunker::new(size, overlap) -> Result<Self>` — separator hierarchy `["\n\n", "\n", " ", ""]`, merges splits up to `size`, carries `overlap` characters across boundaries.
`TokenChunker::new(chunk_size_tokens, overlap_tokens) -> Result<Self>` / `with_token_counter` — token-based approximation; default counter is whitespace tokens (Python uses `tiktoken cl100k_base`; deviation documented).
`ChunkerRegistry::with_defaults()` registers `"char"`, `"recursive"`, `"token"` (and alias `"fixed"`); `register(name, factory)` / `create(name, size, overlap) -> Result<Arc<dyn Chunker>>`.
`chunk_documents` stamps each chunk: new uuid4 id, `metadata = doc.metadata ∪ {chunk_index (per document, 0-based), total_chunks, chunk_id}`; output order is document-major.

### Loaders (`loaders`)
```rust
#[async_trait] pub trait DocumentLoader: Send + Sync {
    fn supports(&self, source: &str) -> bool;
    async fn load(&self, source: &str) -> Result<Vec<Document>>;
}
pub fn html_to_text(html: &str) -> String;
```
Concrete: `TextLoader` (`.txt`), `MarkdownLoader` (`.md`, `.markdown`), `JsonLoader` (`.json`; valid JSON is pretty-printed, malformed JSON keeps raw text), `CsvLoader` (`.csv`, `.tsv`; row docs `"H: v, H2: v2"` with ids `{base}_s{sheet}_r{row}` 1-based including the header, column docs `"Column name: X. Values: a, b"` with ids `{base}_s{sheet}_c{col}` 0-based), `HtmlLoader` (`.html`, `.htm`; tag/script/style stripping, entity decoding, `<title>`/`og:title` extraction), `UrlLoader` (HTTP(S) GET via `openjiuwen_common::http`, HTML→text or plain text), `AutoLoader` (URL → `UrlLoader`, extension → registered loader, stamps `doc_id`, `title`, `file_path`, `file_ext` on every document; missing file → `Error::NotFound`; unknown/PDF/DOCX/XLSX → `Error::Unsupported`), `LoaderRegistry` (lowercase extension map, first registration wins; `register`, `get_for_path`).
`AutoLoader::load_as(source, doc_id: Option<&str>, title: Option<&str>)` mirrors Python `parser.parse(path, doc_id, file_name=...)`: `doc_id` replaces a single-document loader's id verbatim and acts as the id base prefix for multi-document CSV/TSV sources (`{doc_id}_s0_r2`), keeping every generated id distinct.
URL loading sets `metadata["source_url"]` and `source_type="web_page"`; HTML files also set `source_type="web_page"`.
Missing path → `Error::NotFound`; unsupported format → `Error::Unsupported("Unsupported format: ...")` (deviates from Python's silent `[]` for text parsers; documented).

### Embeddings (`embedding`)
```rust
#[async_trait] pub trait Embedding: Send + Sync {
    fn model_name(&self) -> &str;
    fn dimension_hint(&self) -> Option<usize> { None }
    async fn embed_query(&self, text: &str) -> Result<Vec<f32>>;
    async fn embed_documents(&self, texts: &[String], batch_size: Option<usize>) -> Result<Vec<Vec<f32>>>;
    async fn dimension(&self) -> Result<usize> { /* hint or probe with "test" */ }
}
```
`HashEmbedding::new(dimension)` (default 64) — deterministic SHA-256 token hashing with mean pooling and L2 normalisation; used by tests.
`OpenAiEmbedding::new(EmbeddingConfig) -> Result<Self>` — OpenAI-compatible `POST {base_url}/embeddings` (unless the URL already ends in `/embeddings`) with `{model, input[, dimensions]}`; bearer auth when `api_key` is set; batches of `max_batch_size` (default 8) sent in order; parses `data[].embedding` (sorted by `index`), `embedding`, and `embeddings` shapes; empty input/blank item → `Error::Validation`; non-2xx → `Error::Retrieval` with status and a truncated body.

### Vector store (`vector_store`)
```rust
pub struct VectorRecord { pub id: String, pub text: String, pub vector: Vec<f32>,
                          pub metadata: Metadata, pub doc_id: Option<String>, pub chunk_id: Option<String> }
pub struct VectorQuery { pub vector: Vec<f32>, pub top_k: usize /*5*/,
                         pub filter: Option<MetadataFilter>, pub score_threshold: Option<f32> }
#[async_trait] pub trait VectorStore: Send + Sync {
    fn name(&self) -> &str;
    async fn upsert(&self, records: Vec<VectorRecord>) -> Result<usize>;
    async fn get(&self, id: &str) -> Result<Option<VectorRecord>>;
    async fn get_many(&self, ids: &[String]) -> Result<Vec<VectorRecord>>; // default: sequential `get`, skips missing, id order
    async fn list(&self, limit: Option<usize>) -> Result<Vec<VectorRecord>>;
    async fn delete(&self, ids: &[String]) -> Result<usize>;
    async fn delete_by_filter(&self, filter: &MetadataFilter) -> Result<usize>;
    async fn search(&self, query: VectorQuery) -> Result<Vec<SearchResult>>;
    async fn len(&self) -> Result<usize>;
    async fn is_empty(&self) -> Result<bool>;
    async fn clear(&self) -> Result<()>;
}
```
`InMemoryVectorStore::new()`, `::with_collection(name)`, `::open(path)` / `::open_with_collection(path, name)` (load JSON if present, auto-persist after mutations), `persist()`; cosine similarity mapped to `(cos + 1) / 2` (Chroma/Milvus cosine parity), descending score with insertion-order tie-break; records with a different dimension than the query are skipped; metadata filters also resolve the pseudo-fields `doc_id` and `chunk_id`; `search` injects `doc_id`/`chunk_id` into result metadata (Python stores do the same).
Persistence JSON: `{"version":1,"name":"...","records":[...]}` written atomically (temp file + rename).
`VectorStoreRegistry::with_builtin()` registers `"memory"`/`"in_memory"`; `register(provider, factory)`; `create(&VectorStoreConfig) -> Result<Arc<dyn VectorStore>>` returns `Error::Unsupported` for unregistered providers (Milvus/Chroma/PGVector/Elasticsearch seam).

### Rerankers (`reranker`)
```rust
#[async_trait] pub trait Reranker: Send + Sync {
    async fn rerank(&self, query: &str, documents: Vec<SearchResult>, top_n: Option<usize>)
        -> Result<Vec<SearchResult>>;
}
```
`ScoreReranker::new(top_n: Option<usize>)` — deterministic default: descending existing score, optional truncation.
`LlmScorer` trait (`async fn score(&self, query: &str, documents: &[String]) -> Result<Vec<f32>>`) and `LlmReranker::new(Arc<dyn LlmScorer>, top_n)` — adapter seam; a foundation `Model`-backed scorer belongs in an upper crate (`openjiuwen-extensions`) to avoid an upward dependency.

### Knowledge base (`knowledge_base`)
```rust
pub struct SimpleKnowledgeBase { /* config, store, embedder, chunker, loader, reranker */ }
impl SimpleKnowledgeBase {
    pub fn new(config: KnowledgeBaseConfig, store: Arc<dyn VectorStore>,
               embedder: Arc<dyn Embedding>, chunker: Arc<dyn Chunker>) -> Self;
    pub fn from_config(config: KnowledgeBaseConfig, store: Arc<dyn VectorStore>,
                       embedder: Arc<dyn Embedding>) -> Result<Self>; // CharChunker from config sizes
    pub fn with_reranker(self, Arc<dyn Reranker>) -> Self;
    pub fn with_loader(self, Arc<dyn DocumentLoader>) -> Self;
    pub fn id(&self) -> &str;
    pub fn config(&self) -> &KnowledgeBaseConfig;
    pub async fn add_documents(&self, documents: Vec<Document>) -> Result<Vec<String>>;
    pub async fn add_texts(&self, texts: Vec<String>) -> Result<Vec<String>>;
    pub async fn update_documents(&self, documents: Vec<Document>) -> Result<Vec<String>>;
    pub async fn delete_documents(&self, doc_ids: &[String]) -> Result<usize>;
    pub async fn retrieve(&self, query: &str) -> Result<Vec<RetrievalResult>>;
    pub async fn retrieve_with_config(&self, query: &str, config: &RetrievalConfig) -> Result<Vec<RetrievalResult>>;
    pub async fn parse_files(&self, paths: &[String]) -> Result<Vec<Document>>;
    pub async fn parse_urls(&self, urls: &[String]) -> Result<Vec<Document>>;
    pub async fn count(&self) -> Result<usize>;
    pub async fn get_statistics(&self) -> Result<KnowledgeBaseStatistics>;
}
pub struct KnowledgeBaseStatistics { pub kb_id: String, pub index_type: String, pub chunk_count: usize,
                                     pub document_count: usize, pub has_reranker: bool, pub has_loader: bool }
pub async fn retrieve_multi_kb(kbs: &[Arc<SimpleKnowledgeBase>], query: &str, top_k: Option<usize>)
    -> Vec<RetrievalResult>; // dedupe by text, keep max score, descending
pub async fn retrieve_multi_kb_with_source(kbs: &[Arc<SimpleKnowledgeBase>], query: &str, top_k: Option<usize>)
    -> Vec<MultiKBRetrievalResult>; // adds sorted, deduped kb_ids plus raw_score/raw_score_scaled metadata
```
`add_documents` fills empty ids, chunks, embeds all chunk texts in batches, upserts `VectorRecord`s (chunk id, doc_id, chunk metadata), returns document ids in input order.
`retrieve_with_config` embeds the query, searches with `top_k`/`score_threshold`/`filters`, optionally reranks, returns `RetrievalResult`s (`chunk_id` from the hit id, `doc_id` from result metadata) truncated to `top_k`.
`delete_documents` deletes chunks whose `doc_id` is in the list; `update_documents` = delete + add, returning successfully re-added ids.
`parse_files`/`parse_urls` skip failing sources with a `warn` log (Python parity) and return the documents that loaded.
No streaming search: Python exposes none (only `Parser.lazy_parse`, which is non-streaming).

### Binary containers (`parsing`)
```rust
pub const DEFAULT_MAX_OUTPUT_BYTES: usize = 64 * 1024 * 1024;
pub fn inflate(data: &[u8], max_output: usize) -> Result<Vec<u8>>;
pub fn zlib_decompress(data: &[u8], max_output: usize) -> Result<Vec<u8>>;
pub fn adler32(data: &[u8]) -> u32;
pub fn crc32(data: &[u8]) -> u32;

pub struct ZipEntry { pub name: String, pub compression_method: u16, pub flags: u16,
                      pub crc32: u32, pub compressed_size: u32,
                      pub uncompressed_size: u32, pub local_header_offset: u32 }
pub struct ZipArchive { /* archive bytes + entries */ }
impl ZipArchive {
    pub fn from_bytes(data: Vec<u8>) -> Result<ZipArchive>;
    pub async fn open(path: impl AsRef<Path>) -> Result<ZipArchive>;
    pub fn entries(&self) -> &[ZipEntry];
    pub fn entry_names(&self) -> Vec<&str>;
    pub fn entry(&self, name: &str) -> Option<&ZipEntry>;
    pub fn has_entry(&self, name: &str) -> bool;
    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;
    pub fn archive_size(&self) -> usize;
    pub fn read_entry(&self, name: &str) -> Result<Vec<u8>>;
    pub fn read_entry_ignore_case(&self, name: &str) -> Result<Vec<u8>>;
    pub fn read_text(&self, name: &str) -> Result<String>;
}
```
`inflate` decodes raw `DEFLATE` (RFC 1951: stored, fixed and dynamic Huffman, code-length repeat codes, back-references); `zlib_decompress` validates the 2-byte header (CM=8, CINFO<=7, FCHECK, no FDICT) and the trailing `Adler-32`. Malformed streams are `Error::Validation`, short/corrupt payloads are `Error::Retrieval`, preset dictionaries are `Error::Unsupported`.
`ZipArchive` scans backwards for the EOCD record, parses the central directory (stored + `deflate` only, ZIP64/split/encrypted → `Error::Unsupported`), and verifies `CRC-32` and decoded size on extraction. Entry names are byte-oriented: `/` separators, exact case-sensitive lookups, `read_entry_ignore_case` for ASCII-case-insensitive lookups. `open` reads through `tokio::fs`.

### Re-exports (`lib.rs`)
`pub mod common; pub mod chunking; pub mod loaders; pub mod embedding; pub mod vector_store; pub mod reranker; pub mod knowledge_base; pub mod parsing;` plus the public types/traits listed above.

## Constraints

- Depends only on `openjiuwen-common` (shared `Error`/`Result`, `http`, `security::SecretString`, `logging`); `openjiuwen-foundation` is not a dependency, and embeddings use `openjiuwen_common::http` with a local `EmbeddingConfig` (documented deviation).
- `#![forbid(unsafe_code)]`; async via `async_trait`/tokio; no blocking I/O inside async methods (`tokio::fs` for persistence).
- Every public data type: serde snake_case, defaults for missing collection fields, unknown fields tolerated; `Document`/`TextChunk`/results round-trip with Python field names.
- Errors: `openjiuwen_common::error::{Error, Result}`; no panics on user input; invalid sizes/overlaps/inputs → `Error::Validation`; missing files → `Error::NotFound`; unsupported formats/providers → `Error::Unsupported`; store/embedding/network failures → `Error::Retrieval`/`Error::Transport`/`Error::Timeout`.
- Logging: `tracing` (`openjiuwen_common::logging` re-exports); libraries never use `println!`/`eprintln!`.
- Files ≤1000 lines; one responsibility per file; shared helpers live in the lowest module that needs them.
- No new workspace dependencies: use the crates already in the workspace manifest (`regex`, `uuid`, `sha2`, `parking_lot`, `indexmap`, `futures`, `url`, `tempfile`, `wiremock`).
- Tests are offline and deterministic; `openjiuwen-test-support` is not required.
- Crate-level clippy allowances are limited to `module_name_repetitions`, `must_use_candidate`, `missing_errors_doc`, `missing_panics_doc`, and numeric-cast lints; everything else must be clean under `-D warnings`.

## Design Decisions

- The Python `Document.id_` wire name is preserved through `#[serde(rename = "id_", alias = "id")]` while Rust exposes `id`; results use Python's `id`/`doc_id`/`chunk_id` names.
- Python's `Chunker.chunk_documents` is synchronous and returns random uuid4 chunk ids; Rust keeps it synchronous (pure CPU) and stamps the same metadata keys.
- Python `Embedding.dimension` performs a real query because pydantic cannot express async properties; Rust adds `dimension_hint` plus an async default `dimension()` that probes with `"test"`.
- Python has no in-memory vector store and no default reranker; both are new Rust capabilities required by the offline test story. Their score convention follows Chroma/Milvus cosine `(cos+1)/2`.
- `VectorStore::upsert` unifies Python's insert-only (Milvus/Chroma) and upsert (PG) semantics; duplicate ids replace the record, which the knowledge base relies on for re-adds.
- Filters use the store-level JSON shape from Python (`dict` equality and `$` operators); `RetrievalConfig.filters` now actually reaches the store instead of being dropped (documented behavioral fix).
- In-memory persistence is opt-in (`open(path)`), atomic, and human-readable JSON; the store stays the source of truth in memory.
- The LLM reranker takes an `LlmScorer` trait rather than `openjiuwen_foundation::Model` so this crate never depends upward; the adapter lives in `openjiuwen-extensions` when the model layer exists.
- Python's silent `[]` from text/JSON parsers for missing/empty inputs is replaced by explicit `NotFound`/`Validation` errors; `AutoLoader` still treats PDF/DOCX/XLSX as unsupported immediately.
- `parse_files`/`parse_urls` intentionally keep Python's per-source error tolerance (log and skip) because batch ingestion should not fail on one bad file.
- The `DEFLATE`/`zlib` decoder and the ZIP reader are written in-crate because the workspace must stay dependency-free for containers; they return typed errors instead of partial output and cap every allocation at `DEFAULT_MAX_OUTPUT_BYTES`.
- ZIP lookups stay byte-oriented (exact, case-sensitive, `/` separators) because archives are not filesystems; the ASCII-case-insensitive variant is an explicit, documented helper.

## Error Strategy

- Public fallible APIs return `openjiuwen_common::error::Result<T>`.
- Construction validates numeric parameters (`chunk_size > 0`, `0 <= chunk_overlap < chunk_size`, `top_k > 0`, `dimension > 0`) and returns `Error::Validation` with the offending values.
- Loaders: filesystem errors map to `NotFound`/`Transport`; malformed JSON degrades to raw text (Python parity); empty HTML/text content returns `Error::Validation`.
- Embedding/HTTP failures preserve kind (`Timeout`, `Transport`, `Retrieval`) and include the endpoint (never the API key) in messages.
- Store operations never panic on unknown ids (delete returns the count removed; get returns `None`); persistence failures propagate.
- Knowledge base wraps stage failures with `Error::with_context`: `"chunking documents"`, `"embedding chunks"`, `"upserting chunks"`, `"searching vector store"`, `"reranking results"`.

## Test Strategy

- Unit tests beside modules (`#[cfg(test)] mod tests`) and behavior ports cite the Python test file in a comment, e.g. `// port of tests/unit_tests/core/retrieval/indexing/chunker/test_char_chunker.py`.
- Integration tests under `tests/`: `document_json.rs` (JSON field names/round-trip), `chunking.rs` (boundaries/overlap/metadata), `vector_store.rs` (ordering/filtering/upsert/delete/persistence), `loaders.rs` (txt/json/csv/html/url/auto + edge cases), `knowledge_base.rs` (end-to-end with `HashEmbedding`, filters, delete, rerank, statistics), `embedding.rs` (`wiremock`), `parsing_zip.rs` (hand-built ZIP archives: stored/deflated entries, CRC and size failures, unsupported method/encryption/ZIP64, `read_text`, `open`).
- URL and OpenAI embedding tests use `wiremock`; file tests use `tempfile::tempdir`; DEFLATE/ZIP tests use hand-built byte streams and archives; everything runs offline.
- Tests must be deterministic: fake embeddings are hash-based, search ties are insertion-ordered, and no wall-clock or random order is asserted.
- Commands: `cargo fmt -p openjiuwen-retrieval`, `cargo test -p openjiuwen-retrieval`, `cargo clippy -p openjiuwen-retrieval --all-targets -- -D warnings`.

## Notes for Agents

- All eight module trees are implemented: `common`, `chunking`, `loaders`, `embedding`, `vector_store`, `reranker`, `knowledge_base`, `parsing`, re-exported from the crate root.
- `tests/` holds `document_json.rs`, `chunking.rs`, `loaders.rs`, `embedding.rs`, `vector_store.rs`, `knowledge_base.rs`, and `parsing_zip.rs`; unit tests live beside their modules (the DEFLATE suite and its stream writer live in `src/parsing/inflate/tests.rs`).
- Minimal binary-container usage: `ZipArchive::open(path).await?` then `read_text("word/document.xml")?`; raw streams go through `parsing::zlib_decompress(bytes, DEFAULT_MAX_OUTPUT_BYTES)` or `parsing::inflate(...)`.
- DOCX/XLSX/PDF parsers are not implemented yet: they belong on top of `src/parsing/` and stay listed as deferred in `docs/PARITY.md`.
- Minimal offline knowledge base: `SimpleKnowledgeBase::from_config(KnowledgeBaseConfig::new("kb"), Arc::new(InMemoryVectorStore::new()), Arc::new(HashEmbedding::new(64)))?` then `.add_documents(...)`/`.add_texts(...)` and `.retrieve_with_config(query, &RetrievalConfig::default())`; `InMemoryVectorStore::new()` is sync (only `open`/`open_with_collection` are async).
- The in-crate OpenAI client, `HashEmbedding`, and the URL loader use `openjiuwen_common::http`; the crate has no `openjiuwen-foundation` dependency.
- `cargo clippy -p openjiuwen-retrieval --all-targets -- -D warnings` is clean now that the scaffolded foundation dependency is gone; no `--no-deps` workaround is needed.

## Routing Table

- `./src/common/` → `Document`/`TextChunk`, results, configs, `MetadataFilter` (`common/CONTEXT.md`)
- `./src/chunking/` → `Chunker` trait + char/recursive/token chunkers and registry (`chunking/CONTEXT.md`)
- `./src/loaders/` → document loader trait, text/JSON/Markdown/CSV/HTML/URL/auto loaders (`loaders/CONTEXT.md`)
- `./src/embedding/` → `Embedding` trait, OpenAI-compatible client, deterministic hash fake (`embedding/CONTEXT.md`)
- `./src/vector_store/` → `VectorStore` trait, records/queries, in-memory store, JSON persistence, provider registry (`vector_store/CONTEXT.md`)
- `./src/reranker/` → `Reranker` trait, score default, `LlmScorer` seam (`reranker/CONTEXT.md`)
- `./src/knowledge_base/` → `SimpleKnowledgeBase` pipeline, statistics, multi-KB retrieval (`knowledge_base/CONTEXT.md`)
- `./src/parsing/` → in-crate `DEFLATE`/`zlib` decoder and ZIP container reader (`parsing/CONTEXT.md`)
- `./tests/` → cross-module integration tests (`tests/CONTEXT.md`)
- `./Cargo.toml` → crate manifest and dependencies
