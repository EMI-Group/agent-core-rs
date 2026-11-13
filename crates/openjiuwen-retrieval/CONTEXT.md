# openjiuwen-retrieval

## Intent

Documents, chunking, embeddings, vector stores, retrievers, rerankers, query rewriting, triples/graph retrieval, multimodal documents, and knowledge bases for the Rust rewrite.
Maps to the Python package `openjiuwen.core.retrieval` (public surface in `openjiuwen/core/retrieval/__init__.py`), with the document/result/config models from `retrieval/common/` and the foundation store ABCs (`Document`, `Embedding`, `Reranker`, `VectorStore`).
The crate is offline-first: every capability has a deterministic, dependency-free path, and model-backed behaviour is an injected seam.
Implementations: text/JSON/Markdown/CSV/HTML/URL/image/DOCX/XLSX/PDF loading, character/recursive/token chunking, an OpenAI-compatible embedding client plus a deterministic hash fake, an in-memory vector store with JSON persistence, BM25 sparse retrieval, dense/sparse/hybrid/graph/agentic retrievers with RRF fusion, heuristic and model-backed query rewriting, rule-based triple extraction with an in-memory graph store, multimodal document/table representation, pure-Rust DEFLATE/ZIP/XML parsing, and a knowledge base that runs load → chunk → embed → index → retrieve → rerank.
Remaining deferrals are external-service and ML-backed paths: Milvus/Chroma/PGVector/Elasticsearch stores, DashScope embeddings and rerankers, an LLM image captioner and LLM triple extractor (both exist only as traits/seams), multimodal *embedding* models, OCR for scanned PDFs, encrypted PDFs and PDF object streams, and PDF page ordering via the cross-reference table.

## API Surface

Module root `openjiuwen_retrieval`; the commonly used surface is also re-exported from `lib.rs` and a curated `prelude`.

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
Constants/defaults: `DEFAULT_TOP_K = 5`, `DEFAULT_CHUNK_SIZE = 512`, `DEFAULT_CHUNK_OVERLAP = 50`, `DEFAULT_RRF_K = 60`, index types `"vector" | "bm25" | "hybrid"`.

### Chunking (`chunking`)
```rust
pub trait Chunker: Send + Sync {
    fn chunk_text(&self, text: &str) -> Result<Vec<String>>;
    fn chunk_documents(&self, documents: &[Document]) -> Result<Vec<TextChunk>>; // default impl
    fn chunk_size(&self) -> usize;
    fn chunk_overlap(&self) -> usize;
}
```
`CharChunker::new(chunk_size, chunk_overlap) -> Result<Self>` — Python `CharChunker`/`CharSplitter` semantics: `step = size - overlap`, overlap clamped to `[0, size-1]`, chunks never exceed `size`, count = `ceil(len/step)`, whitespace preserved.
`RecursiveCharChunker::new(size, overlap) -> Result<Self>` — separator hierarchy `["\n\n", "\n", " ", ""]`, merges splits up to `size`, carries `overlap` characters across boundaries.
`TokenChunker::new(chunk_size_tokens, overlap_tokens) -> Result<Self>` / `with_token_counter`; the default counter is whitespace tokens (Python uses `tiktoken cl100k_base`; deviation documented).
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
Concrete: `TextLoader` (`.txt`), `MarkdownLoader` (`.md`, `.markdown`), `JsonLoader` (`.json`; valid JSON is pretty-printed, malformed JSON keeps raw text), `CsvLoader` (`.csv`, `.tsv`; row docs and column docs), `HtmlLoader` (`.html`, `.htm`), `UrlLoader` (HTTP(S) GET through `openjiuwen_common::http`), `DocxLoader` (`.docx`, one document: text + `title`/`paragraph_count`/`table_count`), `XlsxLoader` (`.xlsx`, row and column documents from `XlsxWorkbook::to_documents`), `PdfLoader` (`.pdf`, one document: text + `page_count`/`has_text`), `AutoLoader` (URL → `UrlLoader`, extension → registry), `LoaderRegistry` (lowercase extension map, first registration wins).
`LoaderRegistry::with_defaults()` registers text/Markdown/JSON/CSV/TSV/HTML plus `.docx`, `.xlsx`, `.pdf` and every `IMAGE_EXTENSIONS` entry (17 extensions); unknown extensions → `Error::Unsupported("Unsupported format: .ext …")`.
`AutoLoader::load_as(source, doc_id: Option<&str>, title: Option<&str>)` mirrors Python `parser.parse(path, doc_id, file_name=...)`: `doc_id` replaces a single-document loader's id verbatim and acts as the id base prefix for multi-document CSV/TSV/XLSX sources (`{doc_id}_s0_r2`), keeping every generated id distinct.
`AutoLoader` stamps `doc_id`, `title` (file stem), `file_path`, `file_ext` on every document, overwriting a loader-provided title (Python parity).
URL loading sets `metadata["source_url"]` and `source_type="web_page"`; HTML files also set `source_type="web_page"`.
Missing path → `Error::NotFound`; image loading returns one document with `image_path`, `mime_type`, `byte_size`, `modality="image"`, `caption_status`, and captions from the `ImageCaptioner` (offline captioner by default).
A PDF with no extractable text (scanned/image-only) → `Error::Unsupported("PDF has no extractable text: …")`; encrypted PDFs and object streams propagate the parser's `Error::Unsupported`.

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
`OpenAiEmbedding::new(EmbeddingConfig) -> Result<Self>` — OpenAI-compatible `POST {base_url}/embeddings` with `{model, input[, dimensions]}`, bearer auth, batches of `max_batch_size`, tolerant response shapes (`data[].embedding`, `embedding`, `embeddings`); empty input → `Error::Validation`, non-2xx → `Error::Retrieval`.

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
    async fn get_many(&self, ids: &[String]) -> Result<Vec<VectorRecord>>; // default impl
    async fn list(&self, limit: Option<usize>) -> Result<Vec<VectorRecord>>;
    async fn delete(&self, ids: &[String]) -> Result<usize>;
    async fn delete_by_filter(&self, filter: &MetadataFilter) -> Result<usize>;
    async fn search(&self, query: VectorQuery) -> Result<Vec<SearchResult>>;
    async fn len(&self) -> Result<usize>;
    async fn is_empty(&self) -> Result<bool>;
    async fn clear(&self) -> Result<()>;
}
```
`InMemoryVectorStore::new()`, `::with_collection(name)`, `::open(path)` / `::open_with_collection(path, name)` (load JSON if present, auto-persist after mutations), `persist()`; cosine similarity mapped to `(cos + 1) / 2`, descending score with insertion-order tie-break; dimension-mismatched records are skipped; filters resolve `doc_id`/`chunk_id` pseudo-fields; `search` injects them into result metadata.
Persistence JSON: `{"version":1,"name":"…","records":[…]}` written atomically (temp file + rename).
`VectorStoreRegistry::with_builtin()` registers `"memory"`/`"in_memory"`; `create(&VectorStoreConfig)` returns `Error::Unsupported` for Milvus/Chroma/PGVector/Elasticsearch.

### Rerankers (`reranker`)
```rust
#[async_trait] pub trait Reranker: Send + Sync {
    async fn rerank(&self, query: &str, documents: Vec<SearchResult>, top_n: Option<usize>)
        -> Result<Vec<SearchResult>>;
}
```
`ScoreReranker::new(top_n: Option<usize>)` — descending existing score, optional truncation.
`LlmScorer` trait (`async fn score(&self, query: &str, documents: &[String]) -> Result<Vec<f32>>`) and `LlmReranker::new(Arc<dyn LlmScorer>, top_n)` — adapter seam; a `Model`-backed scorer belongs in an upper crate.

### Sparse / BM25 (`sparse`)
```rust
pub struct Bm25Config { pub k1: f32 /*1.2*/, pub b: f32 /*0.75*/ }
pub struct Bm25Query { pub text: String, pub top_k: usize, pub filter: Option<MetadataFilter>, pub score_threshold: Option<f32> }
pub struct Bm25Index { /* per-document term frequencies, lengths, df, N, insertion order */ }
pub struct SparseVector { /* sorted (term, weight) entries */ }
```
`SparseVector`: `new`, `from_weights`, `entries`, `get`, `len`, `is_empty`, `dot`, `l2_norm`, `top_terms(n)` — the crate's sparse-vector representation (Python exposes only a `sparse_vector` *field name*; the stores keep term stats).
`Bm25Index`: `new(Bm25Config)`, `add_document(id, text, metadata)`, `add_documents(&[Document])`, `add_chunks(&[TextChunk])`, `upsert`, `remove`, `contains`, `len`, `is_empty`, `clear`, `stats() -> Bm25Statistics`, `idf(term)`, `search(&Bm25Query) -> Vec<SearchResult>`; stopword builders `with_stopwords`/`without_stopwords` (a small English default list).
Scoring is textbook BM25 (`idf = ln(1 + (N - df + 0.5)/(df + 0.5))`, k1 saturation, b/avgdl length normalisation); raw non-negative scores, score desc then insertion-order tie-break; `filter` ANDs a `MetadataFilter`; documents with no matching term are excluded.
`tokenize(text)` — lowercase Unicode alphanumeric tokens, one token per CJK codepoint, punctuation dropped.

### Retrievers (`retriever`)
```rust
pub enum RetrieveMode { Vector, Sparse, Hybrid /*default*/ } // serde snake_case
#[async_trait] pub trait Retriever: Send + Sync {
    async fn retrieve(&self, query: &str, top_k: usize, score_threshold: Option<f32>,
                      mode: RetrieveMode, filters: Option<&MetadataFilter>) -> Result<Vec<RetrievalResult>>;
    async fn retrieve_search_results(&self, query: &str, top_k: usize, mode: RetrieveMode,
                                     filters: Option<&MetadataFilter>) -> Result<Vec<SearchResult>>;
    async fn batch_retrieve(&self, queries: &[String], top_k: usize, mode: RetrieveMode)
        -> Result<Vec<Vec<RetrievalResult>>>; // sequential default
    async fn close(&self) {}
}
pub fn search_result_to_retrieval(result: SearchResult) -> RetrievalResult;
```
`VectorRetriever::new(store: Arc<dyn VectorStore>, embedder: Arc<dyn Embedding>)` — `Vector` mode only; embeds the query and searches with `top_k`/filter/threshold.
`SparseRetriever::new(index: Arc<Bm25Index>)` — `Sparse` mode only; exposes `index()`.
`HybridRetriever::new(dense, sparse)` / `from_parts(store, embedder, index)`; `with_candidate_factor(usize /*2*/)`, `with_rrf_k(usize /*60*/)` — `Vector` → dense with Python's sparse fallback when dense is empty, `Sparse` → sparse, `Hybrid` → over-fetch `top_k * candidate_factor` per channel, RRF-fuse, truncate; `score_threshold` applies to the dense channel only.
`GraphRetriever::new(chunk_retriever, graph_store)`; `with_max_hops(usize /*2*/)`, `with_expansion_limit(usize /*5*/)`, `with_triple_mode(RetrieveMode /*Sparse*/)` — chunks → entity candidates (query + top chunk texts) → bounded `neighbors` expansion (≤32 triples) → per-triple chunk queries → `rrf_fusion([expanded, chunks])`; empty graph or graph error degrades to the plain chunks with a `tracing::debug`.
`AgenticRetriever::new(retriever, rewriter)`; `with_config(AgenticConfig)`, `with_model(Arc<dyn CompletionModel>)`, `with_extractor(Arc<dyn TripleExtractor>)`; `AgenticConfig { max_iter /*2*/, sufficiency_threshold /*0.6*/, passages_per_iteration /*5*/ }`, `max_iter == 0` → `Error::Validation`.
The agentic loop: retrieve passages → extract triples from the top passages into a `TripleMemory` → deterministic sufficiency check (keyword coverage of accumulated passages) → refine via the rewriter or stop; the model path uses two documented prompts (JSON triples array, `{"sufficient", "next_question"}`) and falls back to heuristics on any model/parse error.
`retrieve_with_trace(…) -> Result<(Vec<RetrievalResult>, AgenticTrace)>` exposes `AgenticTrace { iterations, queries, triples, stopped_reason }` with `AgenticStopReason::{Sufficient, MaxIterations, EmptyRewrite, NoNewResults, ModelFailure}`; the final result is an RRF fusion of every iteration.
`rrf_fusion(results: Vec<Vec<SearchResult>>, k: usize)` (Python parity: keyed by result text, 1-based ranks, keep first-seen object), `rrf_fusion_default`, `deduplicate(results)`.

### Query rewriting and the model seam (`rewrite`, `model`)
```rust
#[async_trait] pub trait CompletionModel: Send + Sync {
    fn name(&self) -> &str;
    async fn complete(&self, prompt: &str) -> Result<String>;
}
pub struct StaticCompletionModel; // scripted replies, last one repeats; call_count()/prompts()
pub fn extract_json_object(text: &str) -> Option<&str>; // first "{"..last "}", ```json fences tolerated

#[async_trait] pub trait QueryRewriter: Send + Sync {
    async fn rewrite(&self, request: &RewriteRequest) -> Result<RewriteOutput>;
    async fn keywords(&self, query: &str) -> Result<Vec<String>>;
    async fn expand(&self, query: &str) -> Result<Vec<String>>;
    async fn rewrite_batch(&self, requests: &[RewriteRequest]) -> Result<Vec<RewriteOutput>>; // default
}
```
`RewriteOutput { before, intention, standalone_query, references, missing, typo: Vec<TypoFix>, gibberish, from_history }` — Python's `rewrite()` schema, all fields serde-defaulted; `TypoFix { original, corrected, reason }`; `RewriteRequest::new(query)`/`with_history`.
`HeuristicQueryRewriter` (`HeuristicRewriteConfig { max_keywords /*8*/, max_queries /*3*/, use_stopwords, stopwords, synonyms, corpus_vocabulary (term→df for IDF), history_window /*4*/ }`) — deterministic: whitespace/filler normalisation, stopword-aware keyword extraction with optional IDF ranking, follow-up resolution from history (short/referential queries are prefixed with the last user turn), synonym `references`, and `expand` → `[query, keyword-only query, synonym-expanded query]` deduped and bounded.
`ModelQueryRewriter` (`ModelRewriteConfig { prompt_template, history_window, fallback: Option<Arc<dyn QueryRewriter>> }`, `DEFAULT_REWRITE_PROMPT`) — explicit `{query}`/`{history}` replacement, JSON extraction, lenient field coercion, blank `standalone_query` clamped to the query, fallback on unparsable output else `Error::Retrieval`.

### Triples and graph (`graph`)
```rust
pub struct Triple { pub subject: String, pub predicate: String, pub object: String, pub metadata: Metadata }
pub struct Entity { pub name: String, pub kind: Option<String>, pub metadata: Metadata }
pub struct TripleQuery { pub subject/predicate/object: Option<String>, pub entity: Option<String>,
                         pub metadata_filter: Option<MetadataFilter>, pub limit: Option<usize> }
#[async_trait] pub trait TripleExtractor: Send + Sync {
    async fn extract(&self, chunks: &[TextChunk]) -> Result<Vec<Triple>>;
}
#[async_trait] pub trait GraphStore: Send + Sync {
    async fn add_triples(&self, triples: Vec<Triple>) -> Result<usize>;
    async fn triples_for_entity(&self, entity: &str) -> Result<Vec<Triple>>;
    async fn neighbors(&self, entity: &str, hops: usize) -> Result<Vec<Triple>>;
    async fn search(&self, query: &TripleQuery) -> Result<Vec<Triple>>;
    async fn all_triples(&self) -> Result<Vec<Triple>>;
    async fn len(&self) -> Result<usize>;
    async fn is_empty(&self) -> Result<bool>;
    async fn clear(&self) -> Result<()>;
    async fn statistics(&self) -> Result<GraphStatistics>;
}
```
`Triple::new/with_metadata/text()/normalized_key()/entities()`; `TripleMemory` (insertion order, dedupe by normalized key, `extend`/`batch_extend`, `triples_str()` = lowercase `"(s p o)"` lines — Python format); `TripleBeam { nodes: Vec<SearchResult>, score }`.
`RuleTripleExtractor` (`RuleExtractionConfig { max_triples_per_chunk /*16*/, min_entity_chars /*2*/, max_entity_chars /*80*/, predicates, include_single_entity /*false*/ }`) — deterministic pattern families: copula `is_a`, `has`/`contains`, `includes` (list splitting for `such as`/`e.g.`), `uses`/`with`, `created`, `acquired`/`owned_by`/`founded_by`, glossary `describes`, opt-in `appears_in`; entity candidates are quoted terms, capitalised multi-word phrases, and CJK runs; metadata `doc_id`, `chunk_id`, rounded `confidence`; `entities_in_text(text) -> Vec<Entity>` is public for graph retrieval.
`InMemoryGraphStore::new()` — entity/predicate adjacency indexes, duplicate-once storage, insertion-ordered reads, bounded `neighbors` BFS; `GraphStatistics { triple_count, entity_count, predicate_count }`.

### Multimodal (`multimodal`)
`Modality { Text, Image, Audio, Video }` (serde snake_case, `Display`, `FromStr`, `mime_prefix`, `from_extension`), `guess_mime_type`.
`ContentPart { modality, data, data_id }`; `MultimodalDocument { id, text, metadata, parts }` with `new`, `with_id`, `add_text`, `add_part`, `add_field(modality, data)` (validates data-URI/URL shape), async `add_file(modality, path)` (tokio read, mime from extension, base64 data URI, random 32-hex `data_id`), `content()` → OpenAI-shaped content array (`text`, `image_url`, `video_url`, `input_audio`, optional `uuid`), `strip()`, `to_document`/`from_document`, `modality_summary`.
`to_data_uri(modality, mime, bytes)` / `parse_data_uri(uri) -> Option<(mime, bytes)>` (base64, workspace `base64` crate).
`ImageCaptioner` trait with `OfflineCaptioner` (deterministic `"[image] <file>"` captions, no model) — the graceful offline degradation path.
`ImageLoader` (`.png .jpg .jpeg .webp .gif .jfif`) implements `DocumentLoader`; `with_captioner`, `with_data_uri(bool)`.
`TableData { headers, rows }`: `new`, `to_markdown`, `from_markdown`, `column`/`column_name`/`column_text`, `row_text`/`row_texts` (Python ExcelParser formats), `to_documents(base_id, sheet_name)` with ids `{base}_s{sheet}_r{row}` (header = row 1) and `to_column_documents` with ids `{base}_s{sheet}_c{col}` (0-based), `to_chunks` (markdown text + `modality="table"` metadata).

### Parsing (`parsing`)
`inflate(data, max_output)` (RFC 1951: stored/fixed/dynamic Huffman, repeat codes, strict validation), `zlib_decompress(data, max_output)` (CM/CINFO/FCHECK validation, preset dictionary → `Unsupported`, Adler-32 verification), `crc32`, `adler32`, `DEFAULT_MAX_OUTPUT_BYTES = 64 MiB` (zip-bomb guard).
`ZipArchive::from_bytes`/async `open`, `entries`/`entry_names`/`entry`/`has_entry`/`read_entry`/`read_text`/`read_entry_ignore_case`/`archive_size`; stored + deflate methods, CRC and size verification, method 12/encryption/ZIP64/split archives → `Error::Unsupported`.
`xml::{parse as parse_xml, XmlDocument, XmlElement, local_name, MAX_DEPTH}` — tolerant tree parser (prolog, comments, CDATA, self-closing tags, attributes, named/decimal/hex entities, prefixed names preserved, depth guard).
`DocxParser::{new, parse_bytes, parse_archive, parse_path}` → `DocxDocument { text, blocks, title, paragraph_count, table_count }` with `DocxBlock::{Paragraph, Heading{level}, Table}`; reads `word/document.xml` (+ `docProps/core.xml` title), maps `Title`/`Subtitle`/`HeadingN` to `#`-prefixed lines and renders `w:tbl` through `TableData::to_markdown`.
`XlsxParser::{new, parse_bytes, parse_archive, parse_path}` → `XlsxWorkbook { sheets: Vec<XlsxSheet { name, index, table }> }` with `sheet()`, `sheet_names()`, `to_documents(base_id)`; resolves `xl/workbook.xml`, `xl/_rels/workbook.xml.rels`, `xl/sharedStrings.xml` (rich-text runs flattened) and worksheet parts (`t="s|str|inlineStr|b|e|d"`, numeric literals, A1 multi-letter columns, gap/width padding, first non-empty row = header).
`PdfParser::{new, parse_bytes, parse_path}` → `PdfDocument { text, pages, page_count, has_text }`; stream scanning in file order, `/FlateDecode` via `zlib_decompress`, text operators `Tj`/`TJ`/`'`/`"`/`Td`/`TD`/`T*`/`Tm`/`BT`/`ET`, PDF literal escapes/octal/continuations, hex strings (UTF-16BE BOM else Latin-1), other filters skipped with `debug`; `/Encrypt` and `/Type /ObjStm` → `Error::Unsupported`; no extractable text → `has_text == false` (no OCR).

### Knowledge base (`knowledge_base`)
```rust
pub struct SimpleKnowledgeBase { /* config, store, embedder, chunker, loader, reranker, sparse index, graph store, rewriter */ }
impl SimpleKnowledgeBase {
    pub fn new(config, store, embedder, chunker) -> Self;
    pub fn from_config(config, store, embedder) -> Result<Self>;
    pub fn with_reranker(self, Arc<dyn Reranker>) -> Self;
    pub fn with_loader(self, Arc<dyn DocumentLoader>) -> Self;
    pub fn with_sparse_index(self, enabled: bool) -> Self;
    pub fn with_rewriter(self, Arc<dyn QueryRewriter>) -> Self;
    pub fn with_agentic(self, enabled: bool) -> Self;
    pub fn with_agentic_iterations(self, iterations: usize) -> Self;
    pub fn with_graph_store(self, Arc<dyn GraphStore>) -> Self;
    pub fn with_graph_hops(self, hops: usize) -> Self;
    pub async fn add_documents / add_texts / update_documents / delete_documents -> Result<Vec<String>|usize>;
    pub async fn retrieve(&self, query: &str) -> Result<Vec<RetrievalResult>>;
    pub async fn retrieve_with_config(&self, query: &str, config: &RetrievalConfig) -> Result<Vec<RetrievalResult>>;
    pub async fn parse_files / parse_urls -> Result<Vec<Document>>;
    pub async fn count / get_statistics;
}
pub struct KnowledgeBaseStatistics { kb_id, index_type, chunk_count, document_count, has_reranker,
                                     has_loader, sparse_index_size, has_graph, has_rewriter, agentic_enabled }
pub async fn retrieve_multi_kb(kbs, query, top_k) -> Vec<RetrievalResult>;
pub async fn retrieve_multi_kb_with_source(kbs, query, top_k) -> Vec<MultiKBRetrievalResult>;
```
`add_documents` fills empty ids, chunks, embeds, upserts `VectorRecord`s, maintains the BM25 index (when enabled) and extracts triples into the graph store (when configured), returning document ids in input order.
`retrieve_with_config` channel selection: `vector` → dense; `bm25` → BM25 (index enabled automatically for that index type); `hybrid` + `with_sparse_index(true)` → dense+BM25 RRF; `hybrid` without a sparse index → dense (documented deviation: Python fuses by default).
`agentic` retrieval wraps the selected channel in `AgenticRetriever` with a corpus-vocabulary `HeuristicQueryRewriter` (offline) or the injected rewriter, bounded by `with_agentic_iterations`.
`use_graph`/`graph_expansion` with a configured graph store routes through `GraphRetriever`; without one they log a `warn` and fall back to the plain channel.
The reranker (when attached) runs after retrieval and before `top_k` truncation; `delete_documents` removes vector and BM25 entries but leaves extracted triples in the graph store (documented).
`retrieve_multi_kb*` dedupe by text, keep max score, sort descending, and attach sorted/deduped `kb_ids` plus `raw_score`/`raw_score_scaled` metadata.

## Constraints

- Depends only on `openjiuwen-common` (shared `Error`/`Result`, `http`, `security::SecretString`, `logging`) plus serde/tokio/async-trait/futures/uuid/regex/sha2/parking_lot/url/indexmap/base64; no `openjiuwen-foundation` dependency (embeddings use `openjiuwen_common::http` with a local `EmbeddingConfig`; documented deviation).
- `#![forbid(unsafe_code)]`; async via `async_trait`/tokio; no blocking I/O inside async methods (`tokio::fs` for persistence and file reading).
- Every public data type: serde snake_case, defaults for missing collection fields, unknown fields tolerated; `Document`/`TextChunk`/results/triples round-trip with Python field names.
- Errors: `openjiuwen_common::error::{Error, Result}`; no panics on user input; invalid sizes/overlaps/inputs/modes → `Error::Validation`; missing files → `Error::NotFound`; unsupported formats/filters/providers/PDF variants → `Error::Unsupported`; store/embedding/network/parse failures → `Error::Retrieval`/`Error::Transport`/`Error::Timeout`.
- Logging: `tracing` (`openjiuwen_common::logging` re-exports); libraries never use `println!`/`eprintln!`.
- Files ≤900-1000 lines; one responsibility per file; shared helpers live in the lowest module that needs them (`parsing/inflate/tests.rs` and `vector_store/memory/tests.rs` keep large test bodies out of the implementation files).
- No new workspace dependencies beyond what `Cargo.toml` already declares; `base64` is a workspace dependency used only by `multimodal`.
- Tests are offline and deterministic; `openjiuwen-test-support` is not required.
- Crate-level clippy allowances are limited to `module_name_repetitions`, `must_use_candidate`, `missing_errors_doc`, `missing_panics_doc`, and numeric-cast lints; everything else must be clean under `-D warnings`.

## Design Decisions

- The Python `Document.id_` wire name is preserved through `#[serde(rename = "id_", alias = "id")]` while Rust exposes `id`; results use Python's `id`/`doc_id`/`chunk_id` names.
- Python has no BM25 index, no in-memory vector store, no graph store, and no default reranker — all are Rust additions required by the offline story; their scoring conventions follow Chroma/Milvus cosine `(cos+1)/2` and textbook BM25, and Python's `rrf_fusion` (text-keyed, k=60) is ported verbatim.
- Python's `SparseRetriever` delegates to a vector database's BM25 function; Rust owns the index (`sparse::Bm25Index`), so sparse retrieval works with no external store, and `SparseVector` provides the term-weight representation the stores never exposed.
- `VectorStore::upsert` unifies Python's insert-only and upsert semantics; duplicate ids replace the record, which the knowledge base relies on for re-adds.
- Filters use the store-level JSON shape from Python and now reach every channel (dense, sparse, hybrid, graph, knowledge base) instead of being dropped — a documented behavioural fix.
- Query rewriting in Python is LLM-only (`standalone_query` + diagnostics); Rust keeps that exact output schema but adds a deterministic `HeuristicQueryRewriter` so rewriting, agentic retrieval, and knowledge-base expansion work offline, with `ModelQueryRewriter` as the model-backed path.
- The Python `AgenticRetriever` requires an LLM for both triple reading and query rewriting; the Rust loop has a deterministic triples+keyword-coverage fallback and only uses a model when one is injected.
- Python's triple extractors are LLM-only; Rust ships a deterministic regex/heuristic `RuleTripleExtractor` (no ML) and keeps the `TripleExtractor` trait as the seam for a future model-backed extractor.
- Python stores triples in a second vector index and bypasses the store abstraction; Rust adds a real `GraphStore` trait + `InMemoryGraphStore` and builds graph retrieval on bounded BFS expansion.
- Python's multimodal document keeps private `_data` tuples; Rust exposes typed `ContentPart`s with the same OpenAI content-array projection, and captions degrade to the deterministic `OfflineCaptioner` because no vision model is bundled.
- DOCX/XLSX/PDF parsing is implemented in-crate (ZIP + DEFLATE + tolerant XML) instead of adding parser crates, keeping the workspace dependency set unchanged; PDF page order is content-stream order (no xref parsing), encrypted PDFs and object streams are typed `Unsupported`, and scanned PDFs yield `has_text == false` instead of OCR.
- `LoaderRegistry::with_defaults()` registers the binary/image loaders, so `AutoLoader` never silently ignores a supported extension; unknown extensions stay `Unsupported`.
- Sparse index defaults on only for `index_type = "bm25"`; `hybrid` fusion is opt-in (`with_sparse_index(true)`) to keep dense results stable, while Python fuses by default.
- `parse_files`/`parse_urls` intentionally keep Python's per-source error tolerance (log and skip) because batch ingestion should not fail on one bad file.

## Error Strategy

- Public fallible APIs return `openjiuwen_common::error::Result<T>`.
- Construction validates numeric parameters (`chunk_size > 0`, `0 <= chunk_overlap < chunk_size`, `top_k > 0`, `dimension > 0`, `k1 >= 0`, `0 <= b <= 1`, `max_iter > 0`, `0 <= sufficiency_threshold <= 1`) and returns `Error::Validation` with the offending values.
- Loaders: filesystem errors map to `NotFound`/`Transport`; malformed JSON degrades to raw text (Python parity); empty HTML/text content returns `Error::Validation`; image-only PDFs return `Error::Unsupported`; DOCX/XLSX container problems propagate typed parse errors.
- Retrievers: mode/channel mismatches → `Error::Validation` naming the supported mode; empty queries → `Error::Validation` (or documented empty result); graph/triple/model failures inside `GraphRetriever`/`AgenticRetriever` degrade instead of failing the call, recorded in `AgenticTrace::stopped_reason`.
- Embedding/HTTP failures preserve kind (`Timeout`, `Transport`, `Retrieval`) and include the endpoint (never the API key) in messages.
- Store operations never panic on unknown ids (delete returns the count removed; get returns `None`); persistence failures propagate.
- Parsing never panics: truncated/invalid DEFLATE, bad ZIP metadata, malformed XML, and broken PDFs return typed errors; output is bounded by `DEFAULT_MAX_OUTPUT_BYTES`.
- Knowledge base wraps stage failures with `Error::with_context`: `"chunking documents"`, `"embedding chunks"`, `"upserting chunks"`, `"searching vector store"`, `"reranking results"`.

## Test Strategy

- Unit tests beside modules (`#[cfg(test)] mod tests`, large suites in `<module>/tests.rs`); behaviour ports cite the Python test file in a comment where applicable.
- Integration tests under `tests/`: `document_json.rs`, `chunking.rs`, `vector_store.rs`, `loaders.rs` (text/JSON/CSV/HTML/URL/image/DOCX/XLSX/PDF through `AutoLoader`), `embedding.rs`, `knowledge_base.rs` (dense pipeline), `knowledge_base_channels.rs` (BM25/hybrid/agentic/graph channels and statistics), `sparse.rs` (BM25 scores, idf/k1/b, filters, RRF), `retrievers.rs` (dense/hybrid/graph/agentic with scripted models), `rewrite.rs` (heuristic + model rewriting, JSON repair, fallbacks), `graph.rs` (extraction pattern families, traversal, queries), `multimodal.rs` (data URIs, content shapes, image loader, tables), `parsing_zip.rs` (byte-built archives, DEFLATE known vectors, CRC/Adler), `parsing_documents.rs` (byte-built DOCX/XLSX + content-stream PDFs).
- Deterministic by construction: hash-based fake embeddings, insertion-ordered ties, rule-based extraction/rewriting, scripted completion models, in-test ZIP/zlib byte builders; no wall-clock or random ordering assertions.
- URL and OpenAI embedding tests use `wiremock`; file tests use `tempfile::tempdir`; everything runs offline.
- Commands: `cargo fmt -p openjiuwen-retrieval`, `cargo test -p openjiuwen-retrieval --offline`, `cargo clippy -p openjiuwen-retrieval --all-targets --offline -- -D warnings`, `cargo doc -p openjiuwen-retrieval --no-deps` (warning-free).

## Notes for Agents

- Every capability documented above is implemented; the remaining deferrals are listed in the Intent section and are external-service/ML paths, all reached through traits or `Error::Unsupported`.
- Minimal offline stack: `SimpleKnowledgeBase::from_config(KnowledgeBaseConfig::new("kb"), Arc::new(InMemoryVectorStore::new()), Arc::new(HashEmbedding::new(64)))?` then `.add_documents(...)`/`.add_texts(...)` and `.retrieve_with_config(query, &RetrievalConfig::default())`; `InMemoryVectorStore::new()` is sync (only `open`/`open_with_collection` are async).
- Offline BM25 path: `let mut index = Bm25Index::new(Bm25Config::default())?; index.add_chunks(&chunks); SparseRetriever::new(Arc::new(index))`.
- Offline hybrid path: `HybridRetriever::from_parts(store, Arc::new(HashEmbedding::new(64)), Arc::new(index))`.
- Offline agentic path: `AgenticRetriever::new(Arc::new(SparseRetriever::new(index)), Arc::new(HeuristicQueryRewriter::default()))` — no model required; add `with_model(Arc::new(StaticCompletionModel::new(replies)))` for the scripted model path.
- Offline graph path: `InMemoryGraphStore::new()` + `RuleTripleExtractor::default()`; `GraphRetriever::new(chunks_retriever, Arc::new(store))`.
- The in-crate OpenAI client, `HashEmbedding`, and the URL loader use `openjiuwen_common::http`; the crate has no `openjiuwen-foundation` dependency.
- Keep `docs/PARITY.md` and `crates/CONTEXT.md` in sync through the parent agent — this crate's deferred list no longer includes BM25, query rewriting, agentic retrievers, graph extraction, multimodal documents, or PDF/DOCX/XLSX parsing.

## Routing Table

- `./src/common/` → `Document`/`TextChunk`, results, configs, `MetadataFilter` (`common/CONTEXT.md`)
- `./src/chunking/` → `Chunker` trait + char/recursive/token chunkers and registry (`chunking/CONTEXT.md`)
- `./src/loaders/` → document loader trait, text/JSON/Markdown/CSV/HTML/URL/DOCX/XLSX/PDF/auto loaders and the registry (`loaders/CONTEXT.md`)
- `./src/embedding/` → `Embedding` trait, OpenAI-compatible client, deterministic hash fake (`embedding/CONTEXT.md`)
- `./src/vector_store/` → `VectorStore` trait, records/queries, in-memory store, JSON persistence, provider registry (`vector_store/CONTEXT.md`)
- `./src/sparse/` → BM25 index, `SparseVector`, tokenizer, stopwords (`sparse/CONTEXT.md`)
- `./src/retriever/` → `Retriever` trait + vector/sparse/hybrid/graph/agentic retrievers and RRF fusion (`retriever/CONTEXT.md`)
- `./src/rewrite/` → `QueryRewriter` trait, heuristic and model-backed rewriters (`rewrite/CONTEXT.md`)
- `./src/model.rs` → `CompletionModel` seam, `StaticCompletionModel`, JSON extraction helper
- `./src/graph/` → `Triple`/`TripleMemory`/`TripleBeam`, rule-based triple extraction, graph store (`graph/CONTEXT.md`)
- `./src/multimodal/` → modalities, `MultimodalDocument`, offline captioner, image loader, table conversion (`multimodal/CONTEXT.md`)
- `./src/parsing/` → DEFLATE/zlib, ZIP containers, tolerant XML, DOCX/XLSX/PDF readers (`parsing/CONTEXT.md`)
- `./src/reranker/` → `Reranker` trait, score default, `LlmScorer` seam (`reranker/CONTEXT.md`)
- `./src/knowledge_base/` → `SimpleKnowledgeBase` pipeline and channels, statistics, multi-KB retrieval (`knowledge_base/CONTEXT.md`)
- `./tests/` → cross-module integration tests (`tests/CONTEXT.md`)
- `./Cargo.toml` → crate manifest and dependencies
