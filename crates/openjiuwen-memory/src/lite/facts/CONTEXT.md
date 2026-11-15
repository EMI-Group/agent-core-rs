# facts

## Intent

Per-fact Markdown files under `facts/` plus the `MEMORY.md`/`USER.md` rendered index split, porting the file layout of Python `openjiuwen.core.memory.lite` (`frontmatter.py`, the per-fact `.md` files, and the summary index) to async Rust.
`facts/` is the source of truth: one Markdown file per coding fact, and the two top-level documents are derived views that may be regenerated at any time.
The module owns the file and document formats only: `render_level`/`parse_level` produce and consume the two index documents as strings, and writing them to disk is the store's job (`crate::lite::local::facts_mode`); scanning, re-import diffing, and index regeneration are caller-driven, so nothing here starts a task, timer, or watcher.

## API Surface

- `FACTS_DIR` (`"facts"`), `MEMORY_FILE` (`"MEMORY.md"`), `USER_FILE` (`"USER.md"`) name the layout.
- `slug::slug(name)` → the readable file-name part: ASCII alphanumerics kept and lowercased, every other run folded to one `-`, edge separators trimmed, capped at 48 characters, `"fact"` when nothing remains.
- `slug::file_name_for(name)` → `<slug>-<hash>.md`, where `hash` is the first eight lowercase hex digits (the high 32 bits) of a 64-bit FNV-1a digest of the exact name.
- `file::FactDocument { front_matter: FrontMatter, body: String }` with `new`, `parse`, `render`, `from_fact`, `to_fact`, `name`, `fact_type`, `description`, `tags`, `path`, `updated_at`.
- `split::IndexLevel { Memory, User }` with `ALL`, `file_name`, `title`, `contains`; `level_of` maps a fact type to its level, and `render_level`/`parse_level` render and parse a level's document.
- `render_level` renders every fact of the level and applies no line cap: the `max_index_lines` truncation exists only in the prompt-facing `crate::lite::index::render_index`.
- `DirStamp { files, total_len, newest_ms }` is the change stamp of one `facts/` directory.
- `FactFiles { root, scope }` is a cloneable handle with `new`, `root`, `scope`, `dir`, `path_for`, `write`, `read`, `read_all`, `read_all_report`, `remove`, `contains`, `stamp`; all I/O methods are `async`.
- `FactFiles::read_all_report` is `read_all` plus the files it could not decode: `(Vec<CodingFact>, Vec<(PathBuf, Error)>)`, sorted facts and every skipped path paired with the parse error that skipped it; a caller that deletes or rewrites fact files uses it, because a file it cannot decode is not evidence that its fact was deleted.

## On-disk layout

```text
<root>/
  facts/<slug>-<hash>.md   one fact per file (front matter plus the Markdown body)
  MEMORY.md                rendered index of the non-user facts
  USER.md                  rendered index of the user-level facts
```

- `FactFiles::dir()` is `<root>/facts`; `write` creates it on demand, and nothing else in the module creates directories.
- `MEMORY.md`/`USER.md` live in the root (not in `facts/`), so `read_all`/`stamp` never see them as facts.
- The file name is derived from the fact name; the `name` field inside the file is authoritative, so renaming a file does not rename a fact and two names that slug identically still get distinct files.

## Front-matter field set and ordering

- `FactDocument::from_fact` writes the canonical keys in this fixed order: `name`, `description`, `type`, `tags`, `path` (only when the fact has one), `updated_at`.
- `type` is the `snake_case` fact-type name (`user`, `feedback`, `project`, `reference`), and an unknown or missing value is `Error::Validation` rather than a silent default.
- `tags` is a compact JSON array (`["build","ci"]`); a comma-separated value (`build, ci`) is accepted on parse, and empty tags are dropped.
- `path` is optional and absent from a fact without one; an empty value decodes as absent.
- `updated_at` is RFC 3339 with nanosecond precision (`SecondsFormat::Nanos`, `Z`), so a fact's timestamp round-trips exactly; a missing or non-RFC-3339 value decodes to `DateTime<Utc>::default()` (the Unix epoch), never "now".
- `description` and `path` are single line by construction (embedded newlines are folded to spaces on render), and `name` is trimmed.
- Unknown `key: value` lines survive `FactDocument::parse` and are re-rendered in place by `FactDocument::render`, but a rewrite through `FactDocument::from_fact` (which `FactFiles::write` uses) emits only the canonical key set, so an unknown key does not survive a fact write.
- A line inside the block that is not a `key: value` pair is skipped on parse (the `FrontMatter` raw tail has no public setter, so such a line is not preserved).
- Only index-level documents, round-tripped through `crate::lite::markdown::MarkdownMemory`, keep unknown attributes across a parse/render cycle.
- Rendering emits LF and exactly one trailing newline; an empty front matter renders no fence block at all (an empty document renders as the empty string), and the body is trimmed of leading/trailing blank lines with inner blank lines and Markdown kept.

## Index and level rules

- `level_of` sends `CodingFactType::User` to `IndexLevel::User` (`USER.md`) and every other type (`feedback`, `project`, `reference`) to `IndexLevel::Memory` (`MEMORY.md`); `IndexLevel::contains` agrees with it, and `IndexLevel::ALL` is `[Memory, User]`.
- `title()` is `Memory`/`User` and `file_name()` is `MEMORY.md`/`USER.md` (the same constants as `MEMORY_FILE`/`USER_FILE`).
- `render_level` keeps only the level's facts, sorts them by fact-type declaration order (`user`, `feedback`, `project`, `reference`) then by name ascending, and renders them through `MarkdownMemory::new().with_title(level.title())` — schema front matter, title, `## <Type>` sections, `### <name>` entries.
- Rendering is byte-deterministic and independent of the input order; an empty slice still renders the front matter plus the level title, so both documents always exist in a stable shape.
- `parse_level` is `MarkdownMemory::parse(text)?.facts()`: entries come back in document order with the default scope and second-precision timestamps.

## Round-trip guarantees

- `FactDocument::from_fact` → `render` → `parse` → `to_fact` reproduces every `CodingFact` field except the scope (a fact file holds one scope, supplied by the caller): name, description, fact type, content, tags, path, and timestamp.
- `parse(render(document))` equals `document` for a document in the shape `render` produces (edge-trimmed body), and `render` is idempotent under `parse`.
- `parse_level(render_level(facts, level))` equals the level's facts sorted by (type order, name), with the default scope and second-precision timestamps.
- A hand-edited index re-imports losslessly when its entries use the canonical attribute set: `parse_level` → `FactFiles::write` per fact → `read_all` → `render_level` reproduces the edited document byte for byte, and an unknown attribute is dropped by the fact-file rewrite (see the unknown-key rule above).

## Atomic writes and tolerance

- `write` validates a non-blank name, creates `facts/`, writes a hidden temporary sibling (`.` + file name + `.` + a per-process counter + `.tmp`, so concurrent writes never share one) and renames it over the target, so a reader sees the previous file or the complete new one and a failed write leaves no fact file behind.
- Temporary files have no `.md` extension, so a concurrent `read_all`/`stamp` ignores them.
- `read` returns `None` for an absent file and a blank name, and rebinds the fact to `FactFiles::scope`; a file that is not decodable returns the parse/decode error to the caller.
- `read_all` reads only regular `*.md` files (case-insensitive extension), skips directories and symlinks (so it never follows a link out of `facts/`), skips a file that does not decode as a coding fact with a `tracing::warn!` naming the path, and returns the facts sorted by name ascending.
- `read_all_report` returns those facts together with every skipped path and its parse error, so a caller can surface how much of the directory it could not read; `read_all` is exactly its first element.
- `remove`/`contains` treat a missing path as absent, a symlink is not a fact file, and a blank name is always absent.
- `stamp` counts only `*.md` regular files, saturates `total_len`, and reports an absent directory as `DirStamp::default()`.
- Filesystem failures are `Error::External` with the failing path as context; validation failures are `Error::Validation`; nothing panics on user input.

## Known Issues

- `FactFiles::remove`/`path_for` address only the canonical `<slug>-<hash>.md`, so a fact file at a non-canonical path (a `name:` edited without renaming the file, or a copied file) is never deleted by name, and its possibly stale copy can win the name-keyed `upsert` in `local::facts_mode::adopt_facts`, silently reverting an edit.
- `FactFiles::write` renders `name` through `single_line`/`trim` (`file.rs:166`) while `path_for` hashes the raw name, so a name with edge whitespace or an embedded newline is stored under a different name than the one that selects its file.
- `FactDocument::to_fact` decodes a present-but-unparsable `updated_at` to the Unix epoch without a diagnostic, which sinks the fact in `updated_at`-ordered listing and excludes it from `daily_memory` rollups (`file.rs:210`).
- `FactDocument::tags`' comma-separated fallback does not strip the brackets of a hand-written JSON-looking list, so `tags: [one, two]` in a *fact file* yields the tags `[one` and `two]` (index documents strip the brackets).
- `split::parse_level` exposes only `MarkdownMemory::facts`, so entries under an unrecognized `## <Type>` heading are dropped with no error and no warning; `local::facts_mode::import_index` then re-renders both index documents from the parsed facts, erasing those entries from disk (`split.rs:106-108`).
- `FactFiles::stamp` folds a `metadata.modified()`/`duration_since(UNIX_EPOCH)` failure into `newest_ms = 0` instead of reporting it, so such a file's changes never move the stamp.

## Routing Table

- `./mod.rs` → module docs, layout constants, `DirStamp`, `FactFiles` (write/read/read_all/read_all_report/remove/contains/stamp), temporary-path and I/O helpers
- `./file.rs` → `FactDocument`, front-matter keys and ordering, parse/render, fact encode/decode, tolerance rules
- `./slug.rs` → `slug`, `file_name_for`, the FNV-1a 64-bit digest and its test vectors
- `./split.rs` → `IndexLevel`, `level_of`, `render_level`, `parse_level`, level sort order

## Test Strategy

- `slug.rs`: 8 unit tests — slug folding/casing/fallbacks, 48-character cap and trimming, determinism, distinct names sharing a slug, path-separator and NUL safety, FNV-1a reference vectors.
- `file.rs`: 20 unit tests — fact round trip, fixed key order, optional `path`/`tags`, single trailing newline, Markdown body preservation, missing/CRLF/hostile front matter, space tolerance, JSON and comma-separated tags, name/type validation errors, optional-field defaults, nanosecond timestamps, accessors.
- `split.rs`: 9 unit tests — level mapping and `contains`, empty-level rendering, split/order/sorting, field-preserving round trip with path and tags, input-order independence, hand-edit tolerance, unterminated block error.
- `mod.rs`: 15 tests over `tempfile::tempdir()` (14 `#[tokio::test]` plus one synchronous path-derivation test) — path derivation, write/read round trip, directory creation with no leftover temporary file, replacement of the same name, absent facts, remove, blank-name rejection, scope rebinding, non-fact Markdown tolerance, symlink escape, stamp tracking, `facts → files → read_all`, the level round trip, and the byte-for-byte hand-edit re-import.
- Gates (all must be green): `cargo fmt -p openjiuwen-memory`, `cargo test -p openjiuwen-memory --offline --lib lite::facts`, the same with `--features sqlite`, `cargo clippy -p openjiuwen-memory --offline --all-targets -- -D warnings` with and without `--features sqlite`, and `cargo doc -p openjiuwen-memory --no-deps` with and without the feature.
- Everything is offline and deterministic: temporary directories, pinned `DateTime::from_timestamp` values, no network, no credentials, no wall-clock assertions.
