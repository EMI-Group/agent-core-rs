//! Per-fact Markdown files and the `MEMORY.md`/`USER.md` rendered index split.
//!
//! Ports the deferred file layout of Python `core/memory/lite/`: instead of keeping every
//! fact inline in one document, each fact owns a Markdown file under `facts/`, and the
//! two top-level documents are rendered views over those files.
//!
//! # Layout
//!
//! ```text
//! <root>/
//!   facts/<name>.md     one fact per file: front matter plus the Markdown body
//!   MEMORY.md           rendered index of the non-user facts
//!   USER.md             rendered index of the user-level facts
//! ```
//!
//! A fact file carries the same interpreted attributes as the inline document form
//! (`description`, `tags`, `path`, `updated_at`), so a fact survives a round trip through
//! either representation. The file name is the fact name, and the fact type decides which
//! rendered index the entry appears in.
//!
//! # Index split
//!
//! `user`-level facts (durable preferences and identity facts about the person) are
//! rendered into `USER.md`; every other fact type (`feedback`, `project`, `reference`) is
//! rendered into `MEMORY.md`. Both indexes are compact, prompt-facing summaries in the
//! existing index shape, capped by the configured line limit, with each entry linking back
//! to its fact file and source path.
//!
//! # Hand edits
//!
//! The rendered indexes may be edited by hand. An edit is re-imported by diffing the
//! rendered entry against the fact file it names: a changed description, tag list, or Fact
//! body is written back to the owning fact file, and a new entry (a line naming a fact that
//! has no file yet) creates one. An entry whose fact file has disappeared is treated as a
//! deletion of that fact. Import is idempotent — re-running it over an unmodified index
//! writes nothing.
//!
//! # Constraints
//!
//! `facts/` is the source of truth and the rendered indexes are derived views of it, so
//! any index can be regenerated at any time. Renders are deterministic (fixed fact-type
//! order, name-ascending inside a type) and end with exactly one newline; writes are atomic
//! (temporary file plus rename) and serialized by the local store's advisory file lock.
