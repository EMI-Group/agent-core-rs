//! The Markdown source of truth of the lite coding memory.
//!
//! A [`MarkdownMemory`](crate::lite::markdown::MarkdownMemory) is one human-editable document holding every coding fact of a
//! scope, mirroring the shape of Python's `MEMORY.md`: a `---` front-matter block, a
//! title, and one section per fact type. Each fact is an `### name` entry carrying a
//! bullet attribute block (`description`, `tags`, `path`, `updated_at`, plus any
//! unknown keys preserved verbatim) followed by the fact body.
//!
//! ```text
//! ---
//! schema: openjiuwen-memory-lite
//! version: 1
//! ---
//!
//! # Memory
//!
//! ## Project
//!
//! ### build
//!
//! - description: How to build the workspace
//! - tags: ["build"]
//! - updated_at: 2026-11-13T07:00:00Z
//!
//! cargo build
//! ```
//!
//! The renderer output is the only shape the parser is required to round-trip, and it
//! does so byte for byte ([`MarkdownMemory::render`](crate::lite::markdown::MarkdownMemory::render) is idempotent under
//! [`MarkdownMemory::parse`](crate::lite::markdown::MarkdownMemory::parse)). Human edits outside that shape never fail the parse:
//! unknown front-matter keys, unknown sections, text inside a known section that is not
//! a fact entry, and non-`key: value` front-matter lines are all preserved verbatim,
//! and the order of fact sections is stable.
//!
//! # Normalization rules
//!
//! - Input is normalized to LF line endings; the renderer always emits LF and exactly
//!   one newline between blocks.
//! - Leading and trailing blank lines of a fact body are dropped, and blank lines
//!   between the `### name` heading, the attribute block, and the body are canonical.
//! - A body line that would otherwise look like a heading is escaped with a backslash
//!   (`\## note`) and unescaped on parse, so fact bodies can contain Markdown headings.
//! - A missing or unparsable `updated_at` attribute decodes to
//!   [`chrono::DateTime::default`] (the Unix epoch) rather than "now", which keeps a
//!   parse deterministic.
//! - `description` and `path` attributes are single-line by construction: embedded
//!   newlines are folded to spaces when rendering.
//! - `tags` are written as a JSON array (a comma-separated value is also accepted when
//!   parsing, so hand-written files stay convenient).
mod document;
mod front_matter;
#[cfg(test)]
mod tests;

pub use document::MarkdownMemory;
pub use front_matter::FrontMatter;
