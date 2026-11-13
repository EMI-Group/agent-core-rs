# personal_context

## Intent

The portable, offline subset of the Python `openjiuwen.harness.personal_context` package, ported as a harness rail plus its supporting store.
The Python package is ~26k lines and dominated by vendor fetchers (Feishu, GitHub, GitCode, Zhihu, Toutiao, RSS, browser bookmarks) and a 10k-line `context_pipeline.py`; those are not ported.
This subtree implements the parts that can run offline and deterministically: status codes, portable-path safety, source metadata, a persisted source-link book, local-file ingestion, keyword search/recall, and the rail that injects a bounded personal-context prompt section.
Every remote/vendor path stays behind a source-provider trait and returns a typed `Error::Unsupported` naming the vendor and the reason.

## API Surface

Owned by this subtree (re-exported from `src/rails/mod.rs` and the crate root by the integration layer):
- `PersonalContextRail`, `PersonalContextRailConfig`, `PersonalContextStore`, `PersonalContextStatus`.
- `SourceMetadata`, source ids/locator normalisation, `SourceLinkBook`, ingested entries, search hits, and the source-provider seam.

## Constraints

- Async-first on tokio where IO is async; short synchronous filesystem work is acceptable and documented.
- No new crates: digests come from `crate::rails::sha256`, atomic JSON writes follow the `rails/evolution/store.rs` pattern.
- No panics on user input; `openjiuwen_common::error::{Error, Result}` everywhere; `tracing` only, never stdout.
- Path safety is mandatory: reject absolute paths, `..`, backslashes on POSIX paths, symlink chains, oversized files.
- Clock and home directory are injected; no `Utc::now()` outside documented call sites and no process-global state.
- Files ≲1000 lines; split submodules by responsibility and document each one here.

## Routing Table

- `./mod.rs` → subtree docs, declarations, re-exports
- `./status.rs` → PersonalContext status codes and their `Error` mapping
- `./path.rs` → portable path-segment checks and safe root-relative resolution
- `./source.rs` → source metadata, locator normalisation, stable source ids
- `./link_book.rs` → persisted source-link book (JSON, atomic write)
- `./ingest.rs` → local-directory scanning and text snippet extraction
- `./search.rs` → BM25F keyword ranking and snippet rendering
- `./provider.rs` → source-provider seam, vendor `Unsupported` providers, registry
- `./store.rs` → persisted entry index and ingest/search facade
- `./rail.rs` → `PersonalContextRail` prompt-section injection
