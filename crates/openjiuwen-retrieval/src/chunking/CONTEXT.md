# chunking — document splitting strategies

## Intent

Port Python `retrieval/indexing/processor/chunker/` core behavior: character windows with overlap, recursive character splitting, and token-based approximation, with the shared chunk-stamping contract.
Pure CPU, synchronous, deterministic except for generated chunk ids.

## API Surface

### `mod.rs`
- `pub trait Chunker: Send + Sync` with `chunk_text`, default `chunk_documents`, `chunk_size`, `chunk_overlap`.
- Re-exports `CharChunker`, `RecursiveCharChunker`, `TokenChunker`, `TokenCounter`, `ChunkerRegistry`, default constants.
- `validate_params(size, overlap) -> Result<()>`: `size > 0`, `overlap < size` (overlap 0 allowed); shared by all chunkers.

### `char.rs`
- `CharChunker { chunk_size, chunk_overlap }`; `new` validates, `Default` 512/50.
- Exact Python `CharSplitter` semantics: `overlap = min(overlap, size-1)`, `step = size - overlap`, chunks `[start, min(len, start+size))`, `start += step`; empty input → `[]`; count is `ceil(len/step)`; no trimming.

### `recursive.rs`
- `RecursiveCharChunker { chunk_size, chunk_overlap, separators }`; `new` uses `["\n\n", "\n", " ", ""]`.
- Split on the first separator that yields pieces ≤ size, recurse into pieces larger than size, merge adjacent pieces up to size, then carry a trailing overlap window across merged chunks; final chunks ≤ size where possible.

### `token.rs`
- `pub trait TokenCounter { fn count(&self, text: &str) -> usize; }` with `WhitespaceTokenCounter` default.
- `TokenChunker { chunk_size_tokens, overlap_tokens, counter }`; `new` validates; `with_token_counter`.
- Splits into counter units (whitespace tokens) with a sliding token window (`step = size - overlap`), rejoins with a single space; documents approximate Python `tiktoken cl100k_base` behavior, deviation documented.

### `registry.rs`
- `pub type ChunkerFactory = Arc<dyn Fn(usize, usize) -> Result<Arc<dyn Chunker>> + Send + Sync>`.
- `ChunkerRegistry::with_defaults()` registers `"char"`, `"fixed"` (alias), `"recursive"`, `"token"`; `register(name, factory)` rejects blank/duplicate names; `create(name, size, overlap)` returns `Error::Unsupported` for unknown names.

## Constraints

- `chunk_documents` is document-major: iterate documents in order, chunk each, then stamp.
- Each chunk gets a new uuid4 `id`, `doc_id = document.id`, metadata = document metadata plus `chunk_index` (0-based per document), `total_chunks`, `chunk_id` (same uuid).
- Empty/whitespace-only documents: empty text yields no chunks; whitespace is preserved for `CharChunker` (Python parity), recursive/token chunkers may trim separators at merge boundaries.
- Constructors validate and return `Error::Validation`; never panic.

## Test Strategy

Unit tests beside each file: exact char boundaries/overlap/count, overlap >= size rejection, empty input, recursive separator behavior, token window/overlap and whitespace rejoin, registry duplicate/unknown names, metadata stamping (`chunk_index`, `total_chunks`, `chunk_id`).

## Routing Table

- `./mod.rs` → `Chunker` trait, validation, exports
- `./char.rs` → fixed-size character chunker
- `./recursive.rs` → recursive character chunker
- `./token.rs` → token approximation chunker and counter trait
- `./registry.rs` → named chunker factories
