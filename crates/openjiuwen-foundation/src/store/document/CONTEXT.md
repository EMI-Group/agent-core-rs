# document — document-storage contract and backends

## Intent

Collections of JSON documents behind one filter/order/page language, with in-memory, local-file, SQLite (`sqlite` feature) and Elasticsearch backends.
Ports Python's `BaseDbStore` role: `mod.rs` holds the trait, the shared query engine, and the contract suite every backend runs.
Shared semantics (upsert replaces by id and auto-creates, a missing collection reads as `None`/empty/`0`, `count`/`delete` ignore paging) live on the `DocumentStore` trait docs in `mod.rs`; `../CONTEXT.md` summarises them one level up.

## API Surface

- `Document { id, fields }` (serde shape `{"id", "fields"}`); `to_value()` is the flat `{"id", ..fields}` view that filters and `order_by` address, so a field named `id` is shadowed (`ID_FIELD`).
- `Query { filter, limit, offset, order_by }` (`limit: None` = unbounded), `SortOrder`, `DocumentStore`.
- `run_query` (filter → stable sort → `skip(offset)` → `take(limit)`) and `matching` (filter only) are `pub(crate)` and are the only implementation of query semantics; `../external/elasticsearch/document.rs` calls both too, so rankings cannot drift across backends.
- `validate_collection` rejects empty, `/`, `\`, NUL, `.` and `..`; every backend calls it before using a name.
- `InMemoryDocumentStore` (`memory.rs`), `LocalFileDocumentStore` (`file.rs`), `SqliteDocumentStore` (`sqlite.rs`, `sqlite` feature); `ElasticsearchDocumentStore` lives in `../external/elasticsearch`.

## Constraints

- Ordering is total and panic-free: `compare_values` never calls `partial_cmp().unwrap()`, `serde_json::Number` cannot hold NaN, and ties keep the backend iteration order (all backends feed id-ascending input), so `sort_by` is stable and cross-backend-consistent.
- `offset`/`limit` are applied with `skip`/`take`; there is no `offset + limit` arithmetic, so no overflow.
- Null and missing values are folded together for ordering (first ascending, last descending); `filter`'s `$eq: null` is the separate, documented case.
- No collection name ever reaches SQL text: `sqlite.rs` keeps two fixed tables and binds every value, and `file.rs` writes one validated `<collection>.json` per collection.
- The file backend writes the whole collection snapshot (temp file + rename) before committing it in memory, so a failed write leaves memory and disk unchanged.

## Known Issues and Gaps

- `sqlite.rs` detects in-memory databases with `url.contains("memory")`: a URL that gives each connection a private temporary database without that word (`sqlite:`, `sqlite://`) keeps the default 10-connection pool, so checkouts see different (mostly table-less) databases, while a file path containing `memory` is needlessly pinned to one connection.
- `sqlite.rs` `query`/`count`/`delete` decode the entire collection (`load_documents` + `fetch_all`) before filtering, so `limit` bounds neither work nor memory, and `delete` then issues one `DELETE` per id.
- `run_query`/`matching` build `Document::to_value()` (a deep clone of every field) for every candidate even when `filter` is `null` and `order_by` is empty, and each backend's `query` clones the matches a second time.
- `compare_rows` re-resolves every `order_by` path through `get_path` on each comparison (`parse_path` allocates a `Vec` plus one `String` per segment), so a sort re-parses its criteria O(n log n) times.
- Numbers compare through `f64` (`compare_values`, `filter::compare`, `filter::equals`), so integers beyond 2^53 compare equal: ordering ties, and `$eq`/`$in`/filtered `delete` can match a different document than the one named.
- `file.rs` and `sqlite.rs` `delete` report the size of the pre-read match set instead of the rows actually removed, so overlapping concurrent deletes over-count; only `memory.rs` is exact (single lock).
- Malformed or typo'd filters (a mistyped `$` key, `$and` that is not an array, `$in` without an array) match nothing by design, so `query`/`count`/`delete` return empty/`0` with no error and a typo'd `delete` filter silently does nothing.
- `LocalFileDocumentStore` rewrites the whole snapshot per mutation: a second instance on the same root silently overwrites the first instance's later writes (only `../CONTEXT.md` states the one-instance assumption, not the module docs).
- `validate_collection` is POSIX-shaped: Windows-invalid characters and over-long names surface as `External` I/O errors in the file backend, and `SqliteDocumentStore::connect` sets `create_if_missing(true)`, so a mistyped path silently creates an empty database.

## Routing Table

- `./mod.rs` → trait, `Document`, `Query`, `run_query`/`matching`, `validate_collection`, unit tests, the shared contract suite (`contract::run`)
- `./memory.rs` → `InMemoryDocumentStore` over one `BTreeMap`
- `./file.rs` → `LocalFileDocumentStore` (per-collection JSON snapshots, atomic rewrite, write lock)
- `./sqlite.rs` → `SqliteDocumentStore` (feature `sqlite`, two fixed tables, Rust-side filtering)
- `../external/elasticsearch/document.rs` → the Elasticsearch backend that also runs `run_query`/`matching`
