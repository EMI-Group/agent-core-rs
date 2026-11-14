//! The local, file-backed coding-memory store: Markdown source of truth plus a derived
//! full-text index, a cross-process lock, and a debounced watcher.
//!
//! Ports the behaviour of Python's `MemoryIndexManager` for one coding-memory root:
//! `MEMORY.md` is the authoritative, human-editable document (see
//! [`MarkdownMemory`](super::markdown::MarkdownMemory)), a derived index database under
//! `.memory/` serves recall, and every mutation is serialized through a cross-process
//! advisory lock so several processes can share one root.
//!
//! # Layout
//!
//! ```text
//! <root>/
//!   MEMORY.md            the source of truth (fact sections + bodies)
//!   .memory/index.db     derived SQLite FTS5 index (feature `sqlite`)
//!   .memory/index.lock   cross-process advisory lock file
//! ```
//!
//! # Consistency model
//!
//! - The Markdown file wins. The index is derived and may always be rebuilt from it
//!   ([`LocalCodingMemory::reindex`](crate::lite::local::LocalCodingMemory::reindex)).
//! - A mutation takes the file lock ([`FileLock`](crate::lite::lock::FileLock)), reloads the Markdown file when its
//!   stamp (modification time plus length) differs from the cached one, applies the
//!   change, writes the document atomically (temporary file plus rename), and updates
//!   the index. A reader applies the same staleness check, so another process's
//!   committed fact becomes visible without an explicit refresh.
//! - Mutations are additionally serialized inside the process by an async mutex; the
//!   file lock alone would also work, but it is a blocking system call and the mutex
//!   keeps concurrent tasks from occupying blocking threads.
//! - `start_watching` additionally reloads (and reindexes) whenever the watcher reports a
//!   `.md` change, so an external editor's edit shows up in `search` without any call
//!   having to observe the file first. [`LocalCodingMemory::reload_count`](crate::lite::local::LocalCodingMemory::reload_count) exposes how
//!   many reloads happened, which is what the watcher test observes.
//!
//! # Search
//!
//! With the `sqlite` feature the store queries the FTS5 index and merges the result with
//! a deterministic in-memory keyword scan; without the feature only the keyword scan
//! runs, so the type exists and works in both builds. The scan is needed in both builds
//! because the FTS `name` column is `UNINDEXED`, so a fact's name alone is not
//! reachable through `MATCH` (the Python implementation has the same property).
//!
//! The keyword score of a fact is the sum over query tokens of the strongest field it
//! matches (name 3, description or tags 2, body 1) divided by three times the token
//! count, capped at `1.0`; it is therefore in `0.0..=1.0` and higher is better. An FTS
//! hit contributes `1.0 - FtsHit::score` (the FTS score itself is higher for weaker
//! matches), and a fact keeps the maximum of its two signals. Only
//! [`SearchOptions::top_k`](crate::record::SearchOptions::top_k),
//! [`SearchOptions::threshold`](crate::record::SearchOptions::threshold),
//! [`SearchOptions::kinds`](crate::record::SearchOptions::kinds), and
//! [`SearchOptions::mode`](crate::record::SearchOptions::mode) are honoured: `Semantic` mode returns
//! [`Error::Unsupported`] because the local
//! store holds no embeddings, and `token_budget` is ignored (there is no context window
//! here).
//!
//! Nothing in this module panics on user input: invalid facts, bad scopes, and missing
//! names are typed errors, and filesystem failures carry the failing path as context.
//!
//! # Modules
//!
//! The store is split by responsibility; the public path is unchanged
//! (`openjiuwen_memory::lite::local::…`).
//!
//! - `layout` — file names, the `.memory/` state directory, the change stamp, and the
//!   atomic document read/write helpers.
//! - `state` — the shared inner state, the handle type, its constructors, and accessors.
//! - `store` — the mutation/read API, locking, staleness checks, and index rebuilds.
//! - `search` — keyword scoring, ranking, and recency.
//! - `watcher` — the debounced watcher lifecycle.

mod layout;
mod search;
mod state;
mod store;
mod watcher;

#[cfg(test)]
mod tests;

pub use layout::{DEFAULT_DEBOUNCE, INDEX_FILE, LOCK_FILE, MEMORY_FILE, STATE_DIR};
pub use state::LocalCodingMemory;

/// Tracing target of the local-store diagnostics.
pub(super) const TARGET: &str = "openjiuwen_memory::lite::local";
