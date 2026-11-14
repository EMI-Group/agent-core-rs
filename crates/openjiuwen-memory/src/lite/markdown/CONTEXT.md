# lite::markdown

## Intent

The Markdown source of truth of the lite coding memory: one `MEMORY.md` document per scope, with a `---` front-matter block, a title, and one section per coding-fact type.
Parsing is total for free-form input; rendering emits the single canonical shape that parsing round-trips byte for byte.
The directory splits the format in three: `front_matter.rs` owns the `---` block, `document.rs` owns the document model and the fact entries, and `tests.rs` holds the contract tests; the public paths are `crate::lite::markdown::MarkdownMemory` and `crate::lite::markdown::FrontMatter`.

## API Surface

- `crate::lite::markdown::MarkdownMemory` — `new`, `parse`, `render`, `scope`/`with_scope`/`set_scope`, `front_matter`/`with_front_matter`, `title`/`with_title`/`set_title`, `upsert_fact`, `remove`, `fact`, `facts`, `is_empty`.
- `crate::lite::markdown::FrontMatter` — `new`, `get`, `insert`, `remove`, `iter`, `len`, `is_empty`.
- `mod.rs` re-exports both types and holds the module documentation (format, normalization rules).
- `front_matter` is crate-internal: the four schema constants and `parse_lines`/`render` are `pub(super)`, so only code in `lite::markdown` (the document module and the tests) can reach them.

## Constraints

- The document is the source of truth; the SQLite FTS5 index in `lite::fts` is derived and may be rebuilt from it at any time.
- The scope is carried in memory only and is never written to the file; `parse` returns the default scope.
- Parsing never panics and never rejects hand-edited content except an unterminated front-matter block (`Error::Validation`).
- `render` is the canonical shape (`# Memory` title by default, one blank line between blocks, LF, one trailing newline); `with_title`/`set_title` replace only the title line, and an empty title renders no title line.
- Unknown front-matter keys, unknown sections, stray section text, and unknown entry attributes are preserved verbatim.
- No filesystem, network, or clock access: this module is pure string transformation.

## Routing Table

- `./mod.rs` → module documentation and the public re-exports
- `./front_matter.rs` → `FrontMatter`, schema constants, fence parsing/rendering
- `./document.rs` → `MarkdownMemory`, fact entries, blocks, parse/render, title accessors
- `./tests.rs` → the 25 unit tests pinning the render/parse contract and the title accessors
