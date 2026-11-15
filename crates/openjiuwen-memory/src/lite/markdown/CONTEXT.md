# lite::markdown

## Intent

The Markdown source of truth of the lite coding memory: one `MEMORY.md` document per scope, with a `---` front-matter block, a title, and one section per coding-fact type.
Parsing is total for free-form input; rendering emits the single canonical shape that parsing round-trips byte for byte.
A hand edit may repeat an entry, leave a bullet that is not `- key: value` inside an attribute block, or carry a timestamp the crate cannot parse; each is kept in a defined way rather than rejected.
The directory splits the format in three: `front_matter.rs` owns the `---` block, `document.rs` owns the document model and the fact entries, and `tests.rs` holds the contract tests; the public paths are `crate::lite::markdown::MarkdownMemory` and `crate::lite::markdown::FrontMatter`.

## API Surface

- `crate::lite::markdown::MarkdownMemory` — `new`, `parse`, `render`, `scope`/`with_scope`/`set_scope`, `front_matter`/`with_front_matter`, `title`/`with_title`/`set_title`, `upsert_fact`, `remove`, `fact`, `facts`, `is_empty`.
- `crate::lite::markdown::FrontMatter` — `new`, `get`, `insert`, `remove`, `iter`, `len`, `is_empty`.
- `mod.rs` re-exports both types and holds the module documentation (format, normalization rules).
- `front_matter` is crate-internal: the four schema constants and `parse_lines`/`render` are `pub(super)`, so only code in `lite::markdown` (the document module and the tests) can reach them.

## Known Issues

- Duplicate `### name` entries (same section or across sections) are parsed as separate entries: `upsert_fact`/`remove` touch only the first copy, so a write can leave the stale copy first and `fact(name)` then returns it, while both calls still report `true`.
- `unescape_body_line` is not the inverse of `escape_body_line`: a body line that literally starts with `\#` renders unescaped and loses its backslash on the next parse, and every mutation rewrites the whole file.
- `parse_tags` falls back to comma splitting for anything that is not a JSON string array, so `- tags: [one, two]` (or a JSON array with a non-string element) silently yields bracket-fragment tags that are then written back in that form.
- An unparsable `updated_at` decodes to the epoch and is re-rendered as `1970-01-01T00:00:00Z`, destroying the hand-written value on the next write; such a fact also sorts last and is excluded from daily rollups.
- A malformed `- ` bullet with no colon ends the attribute block, so every later attribute silently becomes body text.
- The unterminated-front-matter check only fires when a line contains `:` or parses as an attribute; a colon-less truncated block is accepted and its fence then becomes raw document text.

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
