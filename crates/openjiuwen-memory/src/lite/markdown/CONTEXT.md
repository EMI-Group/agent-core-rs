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

- A fact whose `updated_at` text is unparsable keeps its hand-written text on render, but the value carried in memory is the epoch, so the fact sorts last in `updated_at`-ordered listings and falls outside a `daily_memory` rollup for its real date.
- `upsert_fact` rebuilds the entry from the incoming fact, so a replaced entry loses the stray attribute bullets and the preserved raw `updated_at` text the parse had kept.
- The unterminated-front-matter check only fires when a line of the truncated block contains `:` or parses as an attribute; a colon-less truncated block is accepted and its fence then becomes raw document text.

## Constraints

- The document is the source of truth; the SQLite FTS5 index in `lite::fts` is derived and may be rebuilt from it at any time.
- The scope is carried in memory only and is never written to the file; `parse` returns the default scope.
- Parsing never panics and never rejects hand-edited content except an unterminated front-matter block (`Error::Validation`).
- `render` is the canonical shape (`# Memory` title by default, one blank line between blocks, LF, one trailing newline); `with_title`/`set_title` replace only the title line, and an empty title renders no title line.
- Unknown front-matter keys, unknown sections, stray section text, and unknown entry attributes are preserved verbatim.
- A fact name that appears more than once (in one section or across sections) keeps its last occurrence: every other copy is dropped at parse with a `tracing::warn!`, and `remove`/`upsert_fact` drain every copy of the name, so a read returns the latest text and a delete cannot resurrect a leftover.
- A bullet line in an attribute block that is not `- key: value` is kept verbatim after the recognized attributes (with a warning) instead of ending the block, so the attributes that follow it still parse; a bullet after the blank line that closes the block starts the body.
- `escape_body_line` and `unescape_body_line` are exact inverses: a line that starts a heading, a `###` entry, or a backslash gains one backslash on render, and the parse removes exactly one leading backslash, so escape-then-unescape reproduces every line.
- `tags` accepts a JSON array, a comma-separated list, and a bracketed list without quotes (`[one, two]`); the surrounding brackets never become part of a tag, the fallback emits a warning, and the next render writes the canonical JSON array.
- An unparsable `updated_at` keeps its original text verbatim on render while the fact's `updated_at` value decodes to the epoch, so a hand-written timestamp is not replaced.
- No filesystem, network, or clock access: this module is pure string transformation.

## Routing Table

- `./mod.rs` → module documentation and the public re-exports
- `./front_matter.rs` → `FrontMatter`, schema constants, fence parsing/rendering
- `./document.rs` → `MarkdownMemory`, fact entries, blocks, parse/render, title accessors
- `./tests.rs` → the 34 unit tests pinning the render/parse contract, the duplicate-name, escape-inverse, stray-bullet, bracketed-tag and verbatim-timestamp rules, and the title accessors
