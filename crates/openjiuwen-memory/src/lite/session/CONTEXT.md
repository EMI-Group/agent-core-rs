# session

## Intent

JSONL session logs under `<root>/sessions/` plus the keyword recall index over them.
Ports the session indexing of Python `core/memory/lite/manager.py`, which globs `sessions/**/*.jsonl` and treats each file as text, and goes one step further by decoding each line tolerantly into a record.
The module owns append-only writes, recursive discovery, tolerant decoding, chunk extraction, and keyword ranking; it never rewrites or compacts a log.

## API Surface

- `SESSIONS_DIR` — `"sessions"`, the log directory relative to the memory root.
- `SessionStore` (`Debug + Clone`, one path handle, no cached state): `new`, `root`, `dir`, `path_for`, `append`, `files`, `load`, `chunks`, `stamp`.
- `DirStamp { files, total_len, newest_ms }` (`Debug + Clone + Copy + PartialEq + Eq`) — the recursive staleness stamp of the sessions directory.
- `sanitize_session_id(&str) -> Result<String>` — trims, keeps ASCII alphanumerics plus `-`, `_`, `.`, folds every other scalar to `-`, and rejects an empty result with `Error::Validation`; a portable id is returned verbatim and any other id gains `-<8 lowercase hex>` of an FNV-1a digest over the exact id, so distinct ids never share a log.
- `SessionRecord` (`record` module, re-exported): `timestamp: Option<DateTime<Utc>>`, `role: Option<String>`, `content: String`; `new`, `with_role`, `with_timestamp`, `is_empty`, `text`, `parse_line`, `to_json_line`.
- `SessionChunk { path, line, record }`, `SessionHit { path, line, score, similarity, keyword, record }` (`index` module, re-exported).
- `keyword_score(query, text) -> f32` and `rank_chunks(chunks, query, threshold, limit) -> Vec<SessionHit>` (re-exported).

## On-disk layout

```text
<root>/sessions/YYYY-MM-DD/<sanitized session id>.jsonl
```

- The day is supplied by the caller as a `NaiveDate` and rendered `%Y-%m-%d`; the id is sanitized and gets `.jsonl` appended, and a non-portable id carries its discriminator (`café` → `caf-<8 hex>`).
- One JSON object per line: `{"content": ..., "role": ..., "timestamp": "RFC 3339"}` with an absent option omitted, so appending a turn never rewrites earlier lines.
- `to_json_line` escapes through the serializer (embedded newlines and quotes stay on one line) and writes the timestamp in UTC with nanosecond precision, so `parse_line` decodes an equal record.
- Every write stays inside `<root>/sessions/`: no separator survives sanitizing, and the appended `.jsonl` means no sanitized id can be a `.`/`..` component.
- `append` skips whitespace-only records, rejects a record set with no non-empty record, creates the day directory, appends, and flushes.
- A log whose last byte is not a newline (a truncated write or a hand edit) is repaired: `append` writes one newline first with a `tracing::warn!`, so the truncated tail is preserved as its own line and the new records are not glued onto it.
- `files` walks `sessions/**/*.jsonl` recursively and sorts by path; a missing directory is an empty list.
- Decoding is tolerant: `parse_line` returns `None` only for a blank line; a plain-text line, a non-object or truncated JSON payload, and a JSON object without a string content field all become a record whose content is the raw line, with no role and no timestamp.
- Content comes from the first string of `content`, `text`, `message`, `output`; role from `role` or `type`; timestamp from `timestamp`, `ts`, or `time` as an RFC 3339 string or an epoch number (values at or above `1e11` are milliseconds, below are seconds).
- `chunks` yields one `SessionChunk` per non-blank line with the path relative to the store root (`sessions/2026-11-14/abc.jsonl`) and the 1-based line number, in `(path, line)` order.

## Scoring

- `keyword_score` tokenizes the query with `\w+`, lowercases the tokens, keeps at most 10 of them, and counts each token that occurs as a substring of the lowercased text; the score is `matched / total`, `0.0` for a token-less query.
- `rank_chunks` scores each chunk on `SessionRecord::text()` (role and content), sets `score = keyword` and `similarity = 0.0`, keeps `score >= threshold`, orders by score descending then path ascending then line ascending, and truncates to `limit.max(1)`.
- Scoring is keyword-only and needs no embedding model, so the index always works offline and every run produces the same order.

## Constraints

- Errors are `openjiuwen_common::error::Error`: `Validation` for an empty sanitized id or a record set without a non-empty record, `External` carrying the path for filesystem failures.
- No filesystem error is swallowed; `load` of an unreadable file is `External` with the path in the message and source chain.
- Async I/O only (`tokio::fs`), no `unsafe`, no `unwrap`/`expect` outside tests, `tracing` only (target `openjiuwen_memory::lite::session`), every public item documented.
- A tolerated decode emits one `debug` event; no warning is emitted per line, so a plain-text log does not flood the log.
- The index is derived state: it may be dropped and rebuilt from the JSONL files at any time.

## Known Issues

- `parse_line`'s raw-text fallback discards a decodable `role` and `timestamp` whenever the content field is missing or non-string, and nothing marks the record as undecoded, so a corrupted structured line is indistinguishable from a genuine plain-text turn.
- `decode_timestamp` drops an unparsable or out-of-range timestamp silently (`None`, no event); a record without a timestamp is permanently ineligible for a `lite::daily` rollup, which filters on `DailySessionLine::date == Some(date)`.
- `files` matches the extension case-sensitively, so a log named `ABC.JSONL` is invisible to `files`/`chunks`/`stamp` and therefore to recall.
- `stamp` maps a `modified()` or pre-epoch `duration_since` failure to `newest_ms = 0`, so such a file never bumps the staleness stamp.
- `index::TOKENS` degrades to `None` when the static regex fails to compile, which makes `keyword_score` `0.0` for every query (one `warn!`, no error on the public path).
- `SessionStore::relative` falls back to `strip_prefix(..).unwrap_or(path)` and `to_string_lossy`, so a path outside the root or a non-UTF-8 name yields a plausible-looking but wrong citation path.

## Routing Table

- `./mod.rs` → module docs, `SESSIONS_DIR`, `SessionStore`, `sanitize_session_id`, `DirStamp`, line decoding
- `./record.rs` → `SessionRecord` model, tolerant `parse_line`, `to_json_line`, `text`/`is_empty`
- `./index.rs` → `SessionChunk`, `SessionHit`, `keyword_score`, `rank_chunks`, query tokenizer

## Test Strategy

- `record.rs` unit tests: round trip through `to_json_line` (content only, role, timestamp, sub-second precision), content/role field aliases, RFC 3339 strings with offsets and fractions, epoch seconds/millis/fractional decoding, unparsable timestamps, malformed/non-object/wrong-typed/blank/plain-text tolerance, `text`, `is_empty`, serde round trip.
- `index.rs` unit tests: tokenizer case folding and punctuation, the 10-token cap, keyword coverage, substring matching, score/path/line ordering, threshold filtering (including zero-score chunks at a `0.0` threshold), `limit.max(1)`, role text, empty chunk set.
- `mod.rs` unit tests: id sanitizing (slashes, dots, unicode, whitespace, empty → `Validation`), portable ids kept verbatim and folded ids distinct via their discriminator (stable, never equal to another id's fold), traversal containment, `path_for` layout, `append` line-per-record plus cross-call accumulation and escape/newline handling, empty-record skipping and all-empty rejection, repair of a log that ends mid-line, recursive sorted `files`, tolerant `load` of a hand-written mixed file, missing-file `External`, `chunks` path/line numbering, `stamp` file count/length/mtime, two handles on one root.
- Offline and deterministic: `tempfile::tempdir()` roots, real filesystem writes, no network, no credentials, no wall-clock ordering assertions beyond `newest_ms > 0`.
- Gates (all must be green): `cargo fmt -p openjiuwen-memory -- --check`, `cargo test -p openjiuwen-memory --offline --lib lite::session` with and without `--features sqlite`, `cargo clippy -p openjiuwen-memory --offline --all-targets -- -D warnings` with and without the feature, and `cargo doc -p openjiuwen-memory --no-deps` with and without the feature.

## Deferred

- No recency boost and no importance weighting: ranking is keyword coverage only.
- No per-session scoping and no `MemoryScope` filtering; the store is rooted at one memory root.
- No watcher or background session sync in this module (`lite::sync` owns caller-driven triggers).
- No deletion, compaction, rotation, or retention policy for logs.
- No embedding index or cosine similarity in this module: `rank_chunks` sets `SessionHit::similarity` to `0.0` (the store's `search_sessions` layers the optional semantic signal on top through `crate::lite::local::session_mode`).
- Session logs are never copied into `MEMORY.md`/`USER.md`; they are their own recall surface, and `crate::lite::daily` composes their timestamped lines into `daily_memory` rollups.
