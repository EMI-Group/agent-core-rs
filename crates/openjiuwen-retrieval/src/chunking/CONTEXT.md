# chunking — document splitting strategies

## Intent

Port Python `retrieval/indexing/processor/chunker/` core behavior: character windows with overlap, recursive character splitting, token-based approximation, the table-unit-aware hybrid chunker, the text-preprocessor family, and the preprocessing chunker facade, all sharing one chunk-stamping contract.
Pure CPU, synchronous, deterministic except for generated chunk ids.

## API Surface

### `mod.rs`

- `pub trait Chunker: Send + Sync` with `chunk_text`, default `chunk_documents`, `chunk_size`, `chunk_overlap`.
- Re-exports `CharChunker`, `RecursiveCharChunker`, `TokenChunker`, `TokenCounter`, `WhitespaceTokenCounter`, `HybridChunker`, `NoSplitPredicate`, `default_no_split`, `TextPreprocessor`, `WhitespaceNormalizer`, `URLEmailRemover`, `SpecialCharacterNormalizer`, `PreprocessingPipeline`, `REDUNDANT_RUN_PATTERN`, `ChunkUnit`, `TextChunker`, `ChunkerRegistry`, `ChunkerFactory`, and the default constants.
- `validate_params(size, overlap) -> Result<()>`: `size > 0`, `overlap < size` (overlap 0 allowed); shared by every window chunker and every registry factory.
- `stamp_chunks(doc, texts)` (crate-private) builds the stamped chunks: one fresh uuid4 `id`, `doc_id`, copied document metadata, and `chunk_index`/`total_chunks`/`chunk_id` overriding document keys.
- `char_windows(text, size, overlap)` (crate-private) is the shared character window slicer.

### `char.rs`

- `CharChunker { chunk_size, chunk_overlap }`; `new` validates, `Default` 512/50.
- Exact Python `CharSplitter` semantics: `overlap = min(overlap, size-1)`, `step = size - overlap`, chunks `[start, min(len, start+size))`, `start += step`; empty input → `[]`; count is `ceil(len/step)`; no trimming.

### `recursive.rs`

- `RecursiveCharChunker { chunk_size, chunk_overlap, separators }`; `new` uses `["\n\n", "\n", " ", ""]`.
- Split on the first separator that yields pieces ≤ size, recurse into pieces larger than size, merge adjacent pieces up to size, then carry a trailing overlap window across merged chunks; final chunks ≤ size where possible.

### `token.rs`

- `pub trait TokenCounter { fn count(&self, text: &str) -> usize; }` with `WhitespaceTokenCounter` default.
- `TokenChunker { chunk_size_tokens, overlap_tokens, counter }`; `new` validates; `with_token_counter`.
- Splits into counter units (whitespace tokens) with a sliding token window (`step = size - overlap`) and rejoins with a single space.

### `hybrid.rs`

- `pub type NoSplitPredicate = Arc<dyn Fn(&Document) -> bool + Send + Sync>`.
- `HybridChunker { inner: Arc<dyn Chunker>, no_split_when: NoSplitPredicate }`; `new(inner)` installs `default_no_split`, `with_predicate(inner, predicate)` installs a caller predicate, and `inner()` / `should_keep_whole(&Document)` expose the wrapped chunker and the decision.
- `default_no_split(&Document) -> bool`: `source_type` metadata equal to `"row"` or `"column"`, the table-unit stamp of the CSV/XLSX loaders and `multimodal::TableData`.
- `chunk_documents` emits one chunk for a predicate match whose trimmed text is non-empty — text `text.trim()`, `chunk_index = 0`, `total_chunks = 1`, `chunk_id` = the chunk id — and delegates every other document (a blank match included) to `inner.chunk_documents` one document at a time, so the inner indices stay document-relative.
- Input document order is preserved; nothing is merged or de-duplicated.
- `chunk_text` always delegates, so the predicate cannot influence a bare text split; `chunk_size`/`chunk_overlap` report the inner values and `Debug` hides the predicate.

### `preprocess.rs`

- `pub trait TextPreprocessor: Send + Sync { fn process(&self, text: &str) -> String; }`; implementations are pure, synchronous, and infallible, and an empty input comes back unchanged.
- `WhitespaceNormalizer::new()`: `\s+` → one space, then trim (Python `re.sub(r"\s+", " ", text).strip()`).
- `URLEmailRemover::new()` (both removals on, empty replacement) with `with_remove_urls`, `with_remove_emails`, `with_replacement`, and the accessors `remove_urls()`/`remove_emails()`/`replacement()`; order is full URLs (`https?://…`, `www.…`), then bare `.com`/`.net`/`.org`/`.cn` domains, then e-mail addresses.
- The replacement text is inserted literally through `regex::NoExpand`, so `$` and `\` are never expansion syntax.
- `SpecialCharacterNormalizer::new()` with `with_chars_to_remove`, `with_chars_to_replace`, `with_replacement(from, to)`, and the accessors `chars_to_remove()`/`chars_to_replace()`; order is strip `[\x00-\x1F\x7F]`, strip redundant symbol runs, apply the replacement pairs in insertion order, then drop every `chars_to_remove` character.
- `PreprocessingPipeline::{new(Vec<Box<dyn TextPreprocessor>>), add_preprocessor, process, len, is_empty}` plus `from_options(normalize_whitespace, remove_url_email)` (whitespace first, remover second, an off flag contributes nothing); `Default` is the identity and `Debug` prints the length only.
- `REDUNDANT_RUN_PATTERN` (`[^\w\s#*_\-|]{2,}`) is the expressible half of Python's lookahead-based redundant-symbol pattern; the private scan in `strip_redundant_symbols` reproduces both lookaheads.
- Lookahead-free equivalence: `#`, `*`, `_`, `-`, `|` sit outside the run class, so `##` headings, `**bold**`, `_italic_`, table pipes, and `---` rules survive untouched.
- The protected prefix (`，。！？；：“”‘’（）【】《》、…·`, plus the ideographs Python lists but which are word characters and never enter a run) at a run start is kept, and the removable suffix is dropped only when at least two characters remain from the first unprotected symbol.
- Exact consequences: `"你好！！世界"` and `"看，。！"` are unchanged, `"你好！！！!!!"` becomes `"你好！！！"`, a run opening on an unprotected symbol is removed whole (`"!!！"` → `""`, `"hello!!!world"` → `"helloworld"`), and a single symbol is kept (`"hello!world"`, `"a~b"`).

### `text_chunker.rs`

- `ChunkUnit { Char, Token }`: `Default = Char`, `as_str`, `Display`, and `FromStr` over Python's trimmed, case-insensitive `chunk_unit` strings, with `Error::Validation` naming any other value.
- `TextChunker { chunk_size, chunk_overlap, unit, pipeline, inner }`; `new(size, overlap, unit, pipeline)`, `from_options(size, overlap, unit, normalize_whitespace, remove_url_email)`, `with_pipeline`, `with_unit`, and the accessors `chunk_size()`/`chunk_overlap()`/`chunk_unit()`/`pipeline()`; `Default` is 512/50/`Char`/empty pipeline.
- Preprocessing happens here and nowhere else: `chunk_text`/`chunk_documents` run the pipeline and then split with the `CharChunker` or `TokenChunker` built from the unit, so the default empty pipeline makes it a plain fixed-size chunker.
- Stamping is the `Chunker` default contract (document-major, per-document `chunk_index`/`total_chunks`/`chunk_id`), and a pipeline that empties a document yields no chunks.

### `registry.rs`

- `pub type ChunkerFactory = Arc<dyn Fn(usize, usize) -> Result<Arc<dyn Chunker>> + Send + Sync>`.
- `ChunkerRegistry::with_defaults()` registers `"char"`, `"fixed"` (alias), `"recursive"`, `"token"`, and `"hybrid"`; `register(name, factory)` rejects blank/duplicate names; `create(name, size, overlap)` returns `Error::Unsupported` for unknown names.
- The `"hybrid"` factory builds a `HybridChunker` over a `CharChunker` with the default no-split predicate and takes the same `(size, overlap)` pair as every other factory.

## Constraints

- `chunk_documents` is document-major: iterate documents in order, chunk each, then stamp.
- Each chunk gets a new uuid4 `id`, `doc_id = document.id`, metadata = document metadata plus `chunk_index` (0-based per document), `total_chunks`, `chunk_id` (same uuid).
- Empty/whitespace-only documents: empty text yields no chunks; `CharChunker` preserves whitespace (Python parity), recursive/token chunkers may trim separators at merge boundaries, `HybridChunker` trims only the single chunk it keeps whole, and `TextChunker` leaves the decision to its pipeline.
- Constructors validate and return `Error::Validation`; no chunker panics on user input.
- Chunker implementations never touch text they were given: only `TextChunker` preprocesses, and preprocessors are infallible and report nothing.
- `lib.rs` re-exports only the original chunking types at the crate root; `HybridChunker`, `NoSplitPredicate`, `default_no_split`, the preprocessors, `PreprocessingPipeline`, `REDUNDANT_RUN_PATTERN`, `ChunkUnit`, and `TextChunker` are reachable as `openjiuwen_retrieval::chunking::…` only.

## Python parity and deviations

- `TextChunker` (Python's `chunking.py`) and the preprocessor family keep Python's option names and preprocessing order; `HybridChunker` keeps Python's `_default_no_split` rule and its one-chunk-per-table-unit output.
- Python's `SpecialCharacterNormalizer` uses regex lookarounds, which the `regex` crate lacks; `REDUNDANT_RUN_PATTERN` plus the scan of `strip_redundant_symbols` reproduce them, and `NoExpand` restores Python's literal replacement semantics.
- The token unit counts with this crate's `WhitespaceTokenCounter` (Python uses `tiktoken cl100k_base`), the approximation every token path in the crate shares.
- The no-split rule is a `NoSplitPredicate` closure rather than Python's overridable method, so callers pass any document predicate by value.

## Test Strategy

Unit tests beside each file: exact char boundaries/overlap/count, overlap >= size rejection, empty input, recursive separator behavior, token window/overlap and whitespace rejoin, hybrid whole/delegation/blank/custom predicate, registry duplicate/unknown names and the `"hybrid"` entry, preprocessor transforms and their lookahead equivalence, pipeline order and `from_options`, `TextChunker` preprocessing for both units plus stamping.
Integration tests in `tests/chunking.rs` cover the same behavior through the public API: hybrid table units and delegation, preprocessor transformations, pipeline ordering and option counts, and `TextChunker` preprocessing and stamping.

## Routing Table

- `./mod.rs` → `Chunker` trait, validation, window slicer, stamping, exports
- `./char.rs` → fixed-size character chunker
- `./recursive.rs` → recursive character chunker
- `./token.rs` → token approximation chunker and counter trait
- `./hybrid.rs` → table-unit-aware hybrid chunker and its no-split predicate
- `./preprocess.rs` → text preprocessors, pipeline, and the redundant-symbol scan
- `./text_chunker.rs` → preprocessing chunker facade and `ChunkUnit`
- `./registry.rs` → named chunker factories
