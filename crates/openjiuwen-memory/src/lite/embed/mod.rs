//! Chunked embedding index and content-hash embedding cache for the local coding-memory store.
//!
//! Ports the deferred half of Python `core/memory/lite/` that keeps a per-chunk vector
//! index beside the Markdown source of truth: `chunk` splits a document into whole-line
//! chunks, `cache` stores vectors by content hash, and `index` ties the two together
//! as an `EmbeddingIndex`.
//!
//! # Chunking
//!
//! Chunking is line-granular, matching Python's `chunk_markdown`: a document is split
//! into runs of whole lines under a configured size cap, so a chunk boundary never falls
//! inside a line and a chunk can always be cited back to a line range in `MEMORY.md`.
//! Chunk identity is the document id plus that line range
//! (`"<document>:<start_line>:<end_line>"`), so re-chunking an unchanged document
//! produces identical ids while a changed body produces different ones.
//!
//! # Embedding cache
//!
//! Embeddings are cached by the pair (model id, content hash): the model id keeps
//! vectors from different embedding models apart, and the content hash makes the entry
//! valid across renames, re-chunking of surrounding text, and process restarts. A cache
//! hit costs no model call, so the index can be rebuilt offline from the Markdown
//! document once its vectors are cached; a changed chunk changes the hash and therefore
//! misses the cache. Query vectors are never cached.
//!
//! # Scoring
//!
//! Similarity is cosine similarity between the query vector and a chunk vector,
//! normalized the same way the crate's record recall normalizes it (`(cosine + 1) / 2`,
//! so the range is `0.0..=1.0`; see `similarity_from_cosine`). The final chunk score is
//! the documented blend of the vector and text signals, with per-group renormalized
//! weights (`blend`):
//!
//! ```text
//! blend = (w_vector * semantic + w_text * keyword) / (w_vector + w_text)
//! ```
//!
//! where `keyword` is the local store's keyword score for the same chunk, clamped to
//! `0.0..=1.0`. Disabling either term renormalizes the other group, both weights zero
//! falls back to the stronger signal, and a chunk with no stored vector scores
//! `similarity = 0.0` rather than being dropped, so the local store can degrade to
//! keyword-only search when no embedding model is configured.
//!
//! # Constraints
//!
//! The index is derived state: it may always be dropped and rebuilt from `MEMORY.md`.
//! No embeddings are stored in the Markdown document, no model is called at open time,
//! and nothing here needs the network: an `EmbeddingCache` works in memory, and every
//! model call goes through the caller-supplied `Embedding`. Semantic search over a store
//! with no embedding model reports a typed `Unsupported` error in the local store rather
//! than a silent keyword fallback.

/// Content-hash keyed embedding-vector cache (in-memory and `SQLite` backends).
pub mod cache;

/// Line-granular Markdown chunking and the reference token estimate.
pub mod chunk;

/// Chunked vector index, cosine similarity, and the semantic/text blend.
pub mod index;

pub use self::cache::{content_hash, CacheKey, EmbeddingCache};
pub use self::chunk::{chunk_markdown, estimate_tokens, ChunkingConfig, MemChunk};
pub use self::index::{
    blend, cosine_similarity, similarity_from_cosine, EmbeddingIndex, EmbeddingStats, IndexedChunk,
    ScoredChunk,
};

/// The tracing target of lite embedding diagnostics.
pub(super) const TARGET: &str = "openjiuwen_memory::lite::embed";
