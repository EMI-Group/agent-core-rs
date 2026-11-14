# personal_context

## Intent

The offline, deterministic subset of the Python `openjiuwen.harness.personal_context` package plus its rail (`openjiuwen/harness/rails/personal_context.py`).
It keeps portable-path safety, the seven package-owned status codes, per-source metadata with stable `src_<32 hex>` ids, the source-link book, local-files ingestion, BM25F keyword recall over an entry ledger, and a rail that injects a bounded root description into the prompt.
The Python package is ~26k lines dominated by vendor fetchers (Feishu, GitHub, GitCode, Zhihu, Toutiao, RSS, browser bookmarks) and `context_pipeline.py` (10.8k lines); those are not ported.
Every remote or vendor path is behind a source-provider seam and returns a typed `Error::Unsupported` naming the vendor and the reason — silent no-ops are not acceptable.

## API Surface

- Status: `PersonalContextStatus` (`ConfigInvalid` 154000, `StateInvalid` 154001, `FileExecutionError` 154002, `FetchExecutionError` 154003, `PipelineExecutionError` 154004, `PublishExecutionError` 154005, `RuntimeTimeout` 154006) with `code()`, `name()`, `message_template()`, `message(reason)`; `build_error(status, reason)` maps Config→`Error::config`, State→`Error::validation`, the four execution errors→`Error::tool`, RuntimeTimeout→`Error::timeout`.
- Paths: `PORTABLE_FORBIDDEN`, `WINDOWS_DEVICE_NAMES`, `MAX_SEGMENT_CHARS` 80, `MAX_SEGMENT_UTF8_BYTES` 240, `SEMANTIC_NAME_MAX_CHARS` 20, `portable_context_segment_is_safe`, `semantic_context_segment_is_safe`, `validate_service_id`, `service_storage_segment`, `service_storage_segment_is_safe`, `resolve_context_relative_path`.
- Sources: `SourceMetadata` (Python field set: `source_id`, `source_type`, `title`, `locator`, `provider`, `service`, `first_seen`, `last_seen`, `latest_revision`, `latest_hash`), `SourceMetadataStore` (`new`, `root`, `path_for`, `read`, `list`, `upsert`), `normalize_source_locator`, `source_id_for_locator`, `is_source_id`, `display_title`, `observed_at_string`, constants `SOURCE_ID_HEX_CHARS` 32, `MAX_METADATA_BYTES` 64 KiB, `MAX_TITLE_CHARS` 512, `DEFAULT_SOURCE_TITLE`.
- Link book: `SourceLink`, `SourceLinkBook` (`new`, `len`, `is_empty`, `get`, `iter`, `identities`, `register`, `insert`, `merge`, `preview`, `save`, `load`), `resolve_target`, `locator_key`, constants `LINK_TOKEN_PREFIX` `pcs-source-link:`, `IDENTITY_HEX_CHARS` 32, `LINK_BOOK_VERSION` 1.
- Ingest: `LocalDirectoryConfig` (`new`, `with_max_file_bytes`, `with_max_files`, `with_max_text_chars`), `ingest_local_directory`, `IngestedDocument`, `IngestReport`, constants `DEFAULT_MAX_FILE_BYTES` 2 MiB, `DEFAULT_MAX_FILES` 10 000, `DEFAULT_SNIPPET_CHARS` 8 000, `TEXT_EXTENSIONS` (42 entries).
- Search: `search`, `SearchDocument`, `SearchHit`, `snippet`, `tokenize`, `page_heading_fields`, constants `FIELD_WEIGHTS` (title 4.0, path 3.0, headings 2.0, body 1.0), `BM25_K1` 1.2, `BM25_B` 0.75, `MAX_SEARCH_RESULTS` 10, `MAX_SNIPPET_CHARS` 240, `SNIPPET_LEAD_CHARS` 80.
- Providers: `SourceKind` (eight literals, `ALL`, `as_str`, `parse`, `is_deferred`), `FetchRequest`, `FetchedSource`, `SourceProvider` (`kind`, `fetch`, defaulted `is_available`), `LocalFilesProvider`, `UnsupportedSourceProvider`, `SourceProviderRegistry` (`new`, `register`, `get`, `kinds`, `len`, `is_empty`, `with_unsupported_vendors`, `fetch`), `unsupported_message`.
- Store: `PersonalContextStore` (`new`, `home`, `context_root`, `description_path`, `index_path`, `source_root`, `link_book_path`, `metadata_store`, `entries`, `entry`, `search`, `ingest_local`, `link_book`, `save_link_book`, `write_description`), `ContextEntry`, `ContextIndex`, `INDEX_VERSION` 1, `INDEX_FILENAME`, `DESCRIPTION_FILENAME`, `LINK_BOOK_FILENAME`, `SOURCE_ROOT_DIRNAME`, `DEFAULT_SEARCH_LIMIT`.
- Rail: `PersonalContextRail` (`HarnessRail`; `new`, `with_store`, `store`, `config`, `context_root`, `description_path`, `agent_use_enabled`, `refresh_switch`, `search`, `description`), `PersonalContextRailConfig`, `messages_are_contiguous`, `render_content`, constants `PERSONAL_CONTEXT_PRIORITY` 40, `PERSONAL_CONTEXT_SECTION` `personal_context`, `DEFAULT_CONFIG_FILENAME` `personal_context.yaml`, `DEFAULT_LANGUAGE` `cn`, `DEFAULT_MAX_DESCRIPTION_CHARS` 3000, `MAX_CONFIG_BYTES` 4 MiB, `TRUNCATION_NOTICE`.
- The subtree root re-exports all of the above flat; the parent `rails` module and the crate root publish them.

## Constraints

- Async-first only where IO is genuinely async (provider seam, rail hooks); filesystem work is short and synchronous, bounded by explicit byte caps.
- No new crates: digests come from `openjiuwen_sysop::util::sha256_hex` with `crate::rails::sha256::sha256_prefix` for truncation, and this subtree carries its own `<path>.tmp` + rename JSON writer (`atomic.rs`) because `rails::evolution::store` is private to that module.
- Errors are `openjiuwen_common::error::{Error, Result}`; vendor paths are `Error::unsupported`, malformed documents are `Error::validation`, unusable paths are `Error::config`/`Error::security`, a blank-status guard is the `StateInvalid` status.
- No panics on user input, no `unwrap`/`expect` outside tests, `tracing` only (never stdout), secrets and file contents are never logged.
- Path safety is mandatory: absolute paths, `.`/`..`, backslashes, empty segments, non-portable segments, symlink chains and root escapes are rejected before any read or write.
- Clock and home are injected (`observed_at_ms` arguments, `PersonalContextStore::new(home)`); no process-global state beyond the rail's cached switch flag.
- Files ≲1000 production lines; over-long files carry a top-of-file size note and the overrun is inline tests.

## Routing Table

- `./mod.rs` → subtree docs, module declarations, flat re-exports
- `./status.rs` → the seven `${PACKAGE}` status codes, exact Python texts, `build_error` kind mapping
- `./path.rs` → portable/semantic segment checks, service-id validation and storage segments, `resolve_context_relative_path`
- `./atomic.rs` → shared atomic JSON writer/reader and bounded byte reader
- `./source.rs` → locator normalisation, `src_<32 hex>` ids, display titles, RFC3339 stamps, `SourceMetadataStore`
- `./link_book.rs` → `SourceLink` identities, target/locator resolution, label-only previews, persisted book
- `./ingest.rs` → `walkdir` local scan, extension allowlist, byte/file caps, UTF-8 snippets, titles, hashes, report
- `./search.rs` → CJK-aware tokenizer, heading extraction, BM25F ranking, snippet centring
- `./provider.rs` → `SourceKind`, provider trait, `LocalFilesProvider`, `UnsupportedSourceProvider`, registry
- `./store.rs` → `<home>/workspace/context` layout, entry index, dedupe/ingest classification, search facade, link-book and description publication
- `./rail.rs` → agent-use switch, message-contiguity guard, description rendering/truncation, prompt-section publish and clear, `search`

## Design decisions

- Python's per-source Markdown metadata becomes one `<root>/src_<32 hex>.json` per source: field names, `src_<32 hex>` id derivation and first/last-seen semantics are identical, so ids stay wire-compatible; JSON removes the fragile heading/field parser.
- Timestamps are RFC3339 UTC strings derived from an injected `observed_at_ms`; nothing reads the wall clock implicitly.
- The token and heading patterns are resolved through a `OnceLock<Option<Regex>>` cache, so a broken constant is a typed `Error::config` logged once, and `tokenize`/`page_heading_fields` degrade to an empty result instead of panicking.
- The rail's capability replaces Python's `pcs_search` tool registration: `PersonalContextRail::search(query)` holds the same guards (state-invalid when agent use is off, state-invalid for a blank query) while a host decides how to mount a tool over it.
- Link-book identity is `sha256_hex` over the canonical (alphabetically keyed) record JSON truncated to 32 hex, matching Python's `json.dumps(record, sort_keys=True)` so existing books stay readable.
- `resolve_target` returns typed errors where Python returned an empty string for unsupported schemes: an unregistered scheme is a caller bug, not a silent empty target.
- `ingest_local` splits classification between `ingest.rs` (scan counters, `ingested = documents.len()`) and `store.rs` (`unchanged`/`updated` against the ledger), because only the store sees prior state.
- A local file has no provider revision, so `latest_revision` is the content hash; `latest_hash` stays a 64-hex digest.
- `description.md` publication is caller-driven (`write_description`); the deferred Markdown compilation pipeline is what Python's `context_pipeline.py` would have produced.
- The rail clears its section before publishing and after both the model call and the invocation, mirroring Python's temporary runtime attachment, and never aborts a run: a missing, short or unreadable description logs a warning and publishes nothing.
- Rails never import `crate::tools`, `crate::agent` or `crate::subagents`; the rail touches only `openjiuwen_foundation::BaseMessage`, the prompt-section channel and the store.

## Known Issues

- `resolve_context_relative_path` cannot detect non-NFC segments (no Unicode normalisation crate) and rejects only Unicode Cc control characters, not Cf.
- Python's `casefold()` differs from Rust's `to_lowercase()` for a few ligatures, so tokenizer output can differ for those code points.
- `SourceMetadataStore::upsert` writes `latest_hash` verbatim while `read` enforces 64 lowercase hex, so a caller must supply a real digest.
- The heuristic `urljoin`/`html.unescape` ports cover the practical forms only; a full RFC 3986 join is not implemented.

## Deferred Python areas

- Every vendor fetcher and remote transport: `fetch/` (Feishu, GitHub, GitCode, browser bookmarks, Zhihu, Toutiao, RSS) — `UnsupportedSourceProvider` + `SourceProviderRegistry::with_unsupported_vendors`, exact message `` `personal_context`: `<vendor>` source fetcher is not implemented; vendor SDK/remote transport is deferred ``.
- `personal_context.py` (2.8k lines), `context_pipeline.py` (10.8k lines), `agent_support.py`, `file_tools.py`, `models.py`: config contracts, fetch scheduling/cursors, run bookkeeping, deterministic + model-assisted Markdown compilation, atomic publication of generated pages.
- `source_markdown.py` `normalize_source_links` / `rewrite_markdown_prose` and `source_link_book.resolve_source_links` page rewriting: no Markdown parser in-crate; only the book, target resolution and label-only previews are ported.
- The `pcs_search` tool registration (`init`/`_sync_search_tool`/`_owns_search_tool`, ability-manager and prompt-attachment plumbing) and the `PromptAttachmentManager` runtime-attachment channel.
- `context_graph.py` graph/tree/slice building, `read_context_graph_page`, and the `source:<id>` node namespace — only the BM25F search half is ported.
- Vendor credential handling/config (`credentials`, `time_range`, `strategy_profile`, model-client config) and the `PersonalContext` facade lifecycle (`activate_runtime`, `start_collection`, `run_fetch`, `snapshot`, `shutdown`).

## Test Strategy

- Unit tests live beside every module (`#[cfg(test)] mod tests`) with `#![allow(clippy::unwrap_used, clippy::expect_used)]`; everything is offline, deterministic and uses `tempfile` directories plus injected timestamps.
- Status: exact code/name/template assertions for all seven statuses, placeholder substitution, per-family `ErrorKind` mapping, serde snake_case.
- Path: rejection matrix (absolute, `..`, backslash, empty segment, trailing slash, >80 chars, >240 bytes, device names, forbidden/control characters), service-id UTF-16 boundary, legacy vs hashed service segments, symlink chains and root escapes.
- Source: credential/query/fragment stripping, host/scheme normalisation, `[redacted]` cases, id stability across spellings, metadata round-trip, `first_seen` preservation, identity/hash validation, symlinked-root rejection.
- Link book: identity stability and Python parity, register/dedupe, POSIX/http/Windows/`file:`/unsupported-scheme resolution, `locator_key` normalisation, preview, save/load round-trip, missing/malformed/tampered documents, merge precedence.
- Ingest: allowlist and pruned-directory skips, invalid UTF-8, per-file/total caps, POSIX relative paths, id/hash derivation, CJK-boundary truncation, missing/blank configuration errors, warning bounding, symlink skips.
- Search: tokenizer (ASCII, CJK unigrams + bigrams, mixed), heading extraction, BM25F field-weight ordering with hand-computed scores, tie-break by path, limit clamp, snippet centring with both ellipses and CJK.
- Providers: every vendor literal round-trips, every vendor `Unsupported` message asserted verbatim, local fetch (configured directory and request-derived), registry delegation, unregistered kinds.
- Store: ingest classification (new/unchanged/updated with `first_seen` preserved), entry lookup, malformed/unsupported index, blank query, link-book round-trip, description publication, metadata stability across runs.
- Rail: exact wrapper rendering and truncation notice, description read bounds and symlink rejection, switch truth table (bool/string/non-mapping/malformed/missing/oversized/symlinked), section publish/clear isolation, search guards, contiguity matrix, and an ingest → description → section → search recall test.
- Parity references: `tests/unit_tests/harness/rails/test_personal_context_rail.py`, `personal_context/test_path_safety.py`, `test_source_metadata.py`, `test_source_link_book.py`, `test_context_graph.py`.
- Definition of done: `cargo fmt -p openjiuwen-harness`, `cargo test -p openjiuwen-harness --offline --lib rails::personal_context`, the full `cargo test -p openjiuwen-harness`, `cargo clippy -p openjiuwen-harness --all-targets --offline -- -D warnings`, `cargo doc -p openjiuwen-harness --no-deps`.
