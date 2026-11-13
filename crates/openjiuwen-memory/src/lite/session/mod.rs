//! JSONL session logs under `sessions/` and their recall index.
//!
//! Ports the deferred session indexing of Python `core/memory/lite/`: each session is one
//! append-only JSONL file under `sessions/`, and a derived index makes those records
//! searchable alongside coding facts.
//!
//! # Format
//!
//! ```text
//! <root>/sessions/<session-id>.jsonl
//! ```
//!
//! The file holds one JSON object per line — the session id, a role or event kind, the
//! text, and a timestamp — so appending a turn never rewrites earlier ones and the file
//! stays readable with ordinary line-oriented tools. Records are written in the crate's
//! snake_case serde shape, tolerant of unknown and missing fields.
//!
//! # Tolerance
//!
//! Decoding is line-by-line and tolerant: a malformed line, a blank line, or a line whose
//! shape does not match the record model is skipped and counted, so one bad append never
//! makes a session unreadable. Skipped lines are reported through the crate's diagnostics
//! target rather than swallowed, and the surviving line numbers are preserved so a hit can
//! still be cited as `sessions/<id>.jsonl#L<n>`.
//!
//! # Recall
//!
//! The derived index scores a record with the same keyword scoring the local store uses for
//! facts — name-like fields weigh most, then description/tags-like fields, then the body —
//! normalized to `0.0..=1.0` and ordered by score descending then by session id and line
//! number ascending. Embeddings are optional: when an embedding model is configured the
//! keyword score is blended with cosine similarity using the documented vector/text blend,
//! and when it is absent the index is keyword-only rather than reporting an error.
//!
//! # Constraints
//!
//! Session logs are append-only and never rewritten or compacted by this module; the index
//! is derived and may be dropped and rebuilt from the JSONL files at any time. Indexing runs
//! only when the caller asks for it — there is no watcher and no background task here.
