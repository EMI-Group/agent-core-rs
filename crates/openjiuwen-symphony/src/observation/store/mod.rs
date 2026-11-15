//! Durable evidence and immutable observation/merged revision storage.
//!
//! Ports Python `openjiuwen.symphony.observation.store` (v0.1.20). Python keeps the
//! evidence log in `SQLite` behind a `filelock`, and the derived revisions as immutably
//! written JSON documents. This port keeps the same observable contract with no database
//! and no external lock: the evidence log is an append-only `JSONL` file, and every
//! revision document is written atomically (a `<name>.tmp` sibling plus a rename).
//!
//! # On-disk layout
//!
//! ```text
//! <root>/evolution/evidence.jsonl                      # one row per line, append-only
//! <root>/evolution/scopes/<sha256(scope)[:20]>/
//!   current.json                                       # pointer to the active revisions
//!   observations/<observation-revision>/overlay.json   # immutable
//!   merged/<merged-revision>/manifest.json             # immutable
//! ```
//!
//! The scope directory name is the first 20 hex characters of
//! `stable_hash({"graph_scope_id": <scope>})` — the hash is over the JSON *object*, exactly
//! as Python computes it.
//!
//! # Revision ids are content hashes
//!
//! `observation-<sha256(canonical overlay)[:20]>` and
//! `merged-<sha256(schema, scope, static_revision, observation_revision)[:20]>`. Both
//! deliberately exclude `generated_at` (and the merged id also excludes
//! `high_water_sequence`), so re-aggregating identical state reproduces the same ids and a
//! retried publish is idempotent rather than an error: a revision document whose stored
//! content equals the requested content is left untouched, and only a *different* document
//! at the same path is refused.
//!
//! # Concurrency
//!
//! Python serializes appends with `FileLock`. This port serializes them with a
//! [`tokio::sync::Mutex`] that is held across the append's own file I/O, which is the
//! in-process equivalent; a `std::sync` guard is never held across an `.await`. The
//! revision store serializes [`RevisionStore::publish`] the same way, so the overlay, the
//! manifest and the `current.json` pointer can never interleave with a second publisher.
//! The evidence log keeps its parsed rows in memory: the first access reads the whole file
//! (a log is bounded by one row per submitted evidence record), and mutating calls update
//! the index under the same lock.
//!
//! # Module layout
//!
//! The implementation is split by responsibility: `evidence` holds the append-only log,
//! `revision` the immutable overlay, manifest and pointer documents, and `files` the
//! atomic-write, immutability and revision-name helpers both halves share. Every item that
//! used to be public here is re-exported below, so the public paths are unchanged.

mod evidence;
mod files;
mod revision;

pub use evidence::{
    EvidenceAppendInput, EvidenceAppendResult, EvidenceRow, EvidenceStore, EVIDENCE_SCHEMA,
};
pub use revision::{
    CurrentPointer, MergedInput, OverlayInput, RevisionStore, MERGED_SCHEMA, OBSERVATION_SCHEMA,
};
