# fetch

## Intent

The offline-implementable half of Python's `openjiuwen/harness/personal_context/fetch/` package: real source providers for the kinds whose data a caller can supply without a vendor SDK or an authenticated remote API.
`rss_feed.rs` ports `rss_feed.py`: RSS 2.0, RSS 1.0/RDF and Atom, from a local file, a `file://` path or an `http(s)://` URL.
`bookmarks.rs` ports `browser_bookmarks.py`: a Chrome/Edge `Bookmarks` JSON export and a Netscape `bookmarks.html` export.
`record.rs` is the generic file-backed import for a vendor whose records the caller already exported locally (a JSON array or JSONL/NDJSON lines).
The five authenticated vendor APIs (`github`, `gitcode`, `feishu`, `zhihu_reader`, `toutiao_reader`) are not served here and stay typed `Error::Unsupported` in `../provider.rs`, naming the vendor.
Every provider implements [`../provider.rs`]'s `SourceProvider` and returns `FetchedSource` values, so `../store.rs` dedupes them by `source_id`/`content_hash` exactly like a local-directory scan.
Nothing here logs file contents, URLs with credentials or record payloads.

## API Surface

Shared contract of `./mod.rs`:
- `pub(crate) fn fetched_source(...) -> Result<FetchedSource>` — the one constructor every provider uses: normalises the locator, derives `source_id` (`source_id_for_locator`), folds the title (`display_title`), hashes the indexed text (`sha256_hex`) into `content_hash`/`revision`, and fills `size_bytes`, `relative_path`, `service` and the metadata extras. No provider builds a `FetchedSource` literal.
- The crate-internal seams a provider needs: `../provider.rs` (`SourceProvider`, `FetchRequest`, `FetchedSource`, `SourceKind`), `../source.rs` (`normalize_source_locator`, `source_id_for_locator`, `display_title`), `../path.rs` (`resolve_context_relative_path`), `../atomic.rs` (`read_bounded`), `openjiuwen_sysop::util::sha256_hex`, `openjiuwen_common::http::{HttpClient, HttpClientConfig, HttpRequest}`.

`./feed.rs`:
- `RssFeedProvider` (`SourceProvider` for `SourceKind::RssFeed`; `new(RssFeedConfig)`, `config()`, `Default`), `RssFeedConfig { source, service, root, max_bytes, timeout, max_items, max_text_chars }` with one `with_*` setter per field, `RssFeedConfig::new(source)`.
- `FeedFormat { Rss2, Rdf, Atom }` (`as_str`, serde snake_case) and `FeedEntry { title, link, guid, published, summary, content, format }`.
- `parse_feed(bytes, label) -> Result<ParsedFeed>` (tolerant in-crate scanner; `ParsedFeed { format, title, entries }`), `read_feed_bytes(source, config) -> Result<Vec<u8>>` (local path, `file://` path or `http(s)://` URL under the byte cap and timeout).
- Defaults: `DEFAULT_MAX_FEED_BYTES` 2 MiB, `DEFAULT_FEED_TIMEOUT` 30 s, `DEFAULT_MAX_ITEMS` 20, `DEFAULT_MAX_TEXT_CHARS` 8 000, `FEED_USER_AGENT` `"openjiuwen-personal-context/1.0"`, `FEED_ACCEPT` (`application/rss+xml, application/atom+xml, application/xml, text/xml`).

`./bookmarks.rs`:
- `BrowserBookmarksProvider` (`SourceProvider` for `SourceKind::BrowserBookmarks`; `new(BookmarksConfig)`, `config()`, `Default`), `BookmarksConfig { root, path, service, max_file_bytes, max_items }` with `with_*` setters, `BookmarksConfig::new(path)`.
- `BookmarkFormat { ChromeJson, NetscapeHtml }` (`as_str`), `BookmarkEntry { title, url, folder, added_at }`.
- `parse_bookmarks(bytes, format, label) -> Result<Vec<BookmarkEntry>>` plus `detect_bookmark_format(bytes) -> Option<BookmarkFormat>`; the Chrome parser handles `roots.*.children` trees (`type`/`name`/`url`/`date_added`/`children`), the Netscape parser handles `<DT><H3>…</H3>` folders and `<DT><A HREF=…>…</A>` links with `<DL>` nesting.
- Defaults: `DEFAULT_MAX_BOOKMARKS_BYTES` 8 MiB, `DEFAULT_MAX_BOOKMARKS` 10 000.

`./record.rs`:
- `ExportedRecordsProvider` (`SourceProvider` for the `SourceKind` its config names; `new(RecordImportConfig)`, `config()`, `kind()`), `RecordImportConfig { kind, root, path, service, max_file_bytes, max_records, max_text_chars }` with `with_*` setters, `RecordImportConfig::new(kind, path)`.
- `RecordFormat { Json, Jsonl }` (`as_str`), `ImportedRecord { title, url, text, extra }`, `parse_records(bytes, format, label) -> Result<RecordImportReport>`, `RecordImportReport { records, skipped }`.
- Defaults: `DEFAULT_MAX_RECORDS_BYTES` 8 MiB, `DEFAULT_MAX_RECORDS` 10 000, `DEFAULT_MAX_RECORD_TEXT_CHARS` 8 000.

## Constraints

- No new dependency: feeds and bookmarks are parsed by hand-rolled scanners (`regex` is available but a scanner keeps the tolerant behaviour), JSON through `serde_json`, HTTP through the workspace `reqwest` behind `openjiuwen_common::http::HttpClient`.
- Requests are bounded: every read enforces a byte cap, every HTTP call a timeout, `max_items`/`max_records` bound the result, and a body over the cap is a typed error naming the file or URL.
- Path safety: a configured `root` is canonicalised and a local read must resolve inside it (`resolve_context_relative_path` for relative values, canonical-prefix check for absolute ones; symlink escapes rejected). `RssFeedConfig::root` is optional because the caller names the feed; `RecordImportConfig::root` and `BookmarksConfig::root` default to the file's own parent directory when unset, so a relative path is always anchored.
- Errors are `openjiuwen_common::error::{Error, Result}`: malformed input and unusable paths are `Error::validation` naming the file or URL, IO failures `Error::config`, transport/status failures `Error::transport`, deadlines `Error::timeout`; a malformed record inside a valid document is skipped and counted, never fatal.
- Stable identity: an item's locator is its own link (normalised) or, when a feed entry/bookmark/record has none, `<source locator>#<guid-or-index>`; `source_id` therefore stays identical across re-fetches, and the `content_hash` over the indexed text distinguishes an edited item.
- Degenerate output is fine: a feed with no entries or a bookmarks export with no links returns an empty vector, not an error.
- No panics on user input, no `unwrap`/`expect` outside tests, `tracing` only (never stdout), no secrets or file contents logged, files ≲1000 production lines.
- XML input containing a `DOCTYPE` or `ENTITY` declaration is rejected as a typed validation error (no external-entity expansion, mirroring Python's `_UNSAFE_XML_DECLARATION`).

## Routing Table

- `./mod.rs` → module docs, submodule declarations, flat re-exports, the shared `fetched_source` constructor
- `./feed.rs` → feed byte loading (local/`file://`/HTTP), the tolerant RSS 2.0 / RDF / Atom scanner, entry selection, `RssFeedProvider`
- `./bookmarks.rs` → Chrome JSON and Netscape HTML bookmark parsing, flattening/flatten-dedupe, `BrowserBookmarksProvider`
- `./record.rs` → JSON array and JSONL record mapping, invalid-record accounting, `ExportedRecordsProvider`

## Test Strategy

- Unit tests beside each module (`#[cfg(test)] mod tests`, `#![allow(clippy::unwrap_used, clippy::expect_used)]`), offline and deterministic: `tempfile` roots for files, a loopback `TcpListener` serving one canned response for the HTTP cases, no credentials, no wall-clock assertions.
- `feed.rs`: RSS 2.0 and Atom happy paths from a file and over loopback HTTP, `file://` handling, CDATA/entity decoding, missing optional fields, malformed XML (typed error naming the file), doctype rejection, oversized local file, oversized HTTP body, unreachable URL (unroutable loopback port) and a 404, stable ids across two fetches of one feed, empty feed, item cap.
- `bookmarks.rs`: Chrome JSON with nested folders and two roots, Netscape HTML with nested `<DL>` folders, duplicate URL dedupe, missing `url` nodes, malformed JSON/HTML (typed error), missing file, out-of-root path, byte cap, format detection.
- `record.rs`: JSON array and JSONL happy paths, field aliases (`title`/`name`, `url`/`link`/`href`, `text`/`content`/`description`/`body`), invalid records counted as skipped, malformed JSON document (typed error), missing file, out-of-root path, byte cap, `SourceKind::LocalFiles` rejection.
- Definition of done: `cargo fmt -p openjiuwen-harness`, `cargo test -p openjiuwen-harness --offline --lib rails::personal_context::fetch`, `cargo clippy -p openjiuwen-harness --all-targets --all-features --offline -- -D warnings`.
