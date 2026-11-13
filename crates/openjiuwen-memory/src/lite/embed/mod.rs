//! Chunked embedding index and content-hash embedding cache for the local coding-memory store.
//!
//! Ports the deferred half of Python `core/memory/lite/` that keeps a per-chunk vector
//! index beside the Markdown source of truth.
//!
//! # Chunking
//!
//! Chunking is line-granular, matching Python's `chunk_markdown`: a fact body is split
//! into runs of whole lines under a configured size cap, so a chunk boundary never
//! falls inside a line and a chunk can always be cited back to a line range in
//! `MEMORY.md`. Chunk identity is derived from the owning fact name plus the chunk
//! ordinal, so re-chunking an unchanged document produces identical ids and unchanged
//! rows are not rewritten.
//!
//! # Embedding cache
//!
//! Embeddings are cached by the pair (model id, content hash): the model id keeps
//! vectors from different embedding models apart, and the content hash makes the entry
//! valid across renames, re-chunking of surrounding text, and process restarts. A cache
//! hit costs no model call, so the index can be rebuilt offline from the Markdown
//! document; a changed body changes the hash and therefore misses the cache.
//!
//! # Scoring
//!
//! Similarity is cosine similarity between the query vector and a chunk vector,
//! normalized the same way the crate's record recall normalizes it
//! (`(cosine + 1) / 2`, so the range is `0.0..=1.0`). The final chunk score is the
//! documented blend of the vector and text signals, with per-group renormalized
//! weights:
//!
//! ```text
//! semantic = (w_sim * similarity + w_kw * keyword) / (w_sim + w_kw)
//! ```
//!
//! where `keyword` is the local store's keyword score for the same chunk, clamped to
//! `0.0..=1.0`. Disabling either term renormalizes the other group, and a chunk with no
//! stored vector scores `similarity = 0.0` rather than being dropped, so the local store
//! degrades to keyword-only search when no embedding model is configured.
//!
//! # Constraints
//!
//! The index is derived state: it may always be dropped and rebuilt from `MEMORY.md`.
//! No embeddings are stored in the Markdown document, no model is called at open time,
//! and the semantic search mode reports a typed `Unsupported` error when no embedding
//! model is wired in — never a silent keyword fallback.
