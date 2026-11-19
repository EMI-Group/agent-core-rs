# vector — vector storage contracts, shared index, local-file snapshots

## Intent

The `VectorStore` contract (schemas, insert-only adds, normalized similarity search) plus its two reference backends: the in-memory engine and the JSON-snapshot file store.
`VectorIndex` in `memory.rs` is the only implementation of the semantics; `InMemoryVectorStore` and `LocalFileVectorStore` differ only in where the index lives (RAM vs `<root>/vector/<collection>.json`).
Network vector adapters (Chroma/Milvus/Elasticsearch) do not live here — they live in `../external/` and share only the trait and `Metric`.

## API Surface

- `mod.rs`: `VectorDataType`, `FieldSchema`, `CollectionSchema` (+`validate`), `VectorQuery`, `VectorSearchResult`, `Metric` (+`score`), `VectorStore`, `DEFAULT_TOP_K`, `collection_not_found`.
- `memory.rs`: `InMemoryVectorStore` (`new`, `search_with_metric`) plus the `pub(crate)` engine `VectorIndex` / `CollectionData` / `Doc` that the file backend reuses.
- `file.rs`: `LocalFileVectorStore` (`new`, `root`, `search_with_metric`) persisting one snapshot per collection under `<root>/vector/`.

## Constraints

- Metric semantics are frozen by `store/CONTEXT.md` and the contract suite: every score is finite and inside `[0, 1]`, higher is better, ties break by primary key ascending.
- `CollectionSchema::validate` is the single schema gate; both stores validate on create and the file store re-validates every restored snapshot (`VectorIndex::restore_collection`), so `field.dim.unwrap_or(0)` in `search` cannot be reached with a `None` dim.
- `add_docs` validates the whole batch before mutating, rejects duplicate primary keys inside the batch and against stored documents, and applies `default_value`/`auto_id` before insertion.
- All file I/O is `tokio::fs`; no blocking std fs call runs on an async path and no `unwrap`/`expect`/indexing panic is reachable from library data.
- The file backend accepts only `[A-Za-z0-9._-]` collection names (not `.`/`..`), so a name can never escape `<root>/vector/`.

## Design Decisions

- Search materializes each candidate's document JSON through `Doc::to_json` (vectors re-inserted as arrays) so the shared `store::filter` language works on vector and scalar fields alike.
- Result ordering is `score desc, primary key asc` via `f32::total_cmp`, so the order is total and deterministic even though scores are already finite.
- The file store keeps the whole index in memory after `new()` and rewrites the affected collection's snapshot per mutation; two stores on the same root do not observe each other until one is reopened.
- Snapshot load is fail-fast: an unparsable file or an invalid schema aborts `LocalFileVectorStore::new` with the file name in the context, never an empty collection.
- `Metric::Dot` squashes through `tanh`, so it cannot reach `1.0` and saturates for large-magnitude vectors; `Metric::Cosine` maps a zero vector to `0.5` and mismatched/empty dimensions to `0.0`/`1.0` as documented.
- `VectorIndex::to_value`/`from_value` exist for whole-index export and tests; the file backend persists `collection_value` per collection instead.

## Known Issues

- `LocalFileVectorStore` serializes every mutation and its snapshot behind one `tokio::sync::Mutex` (`writes`), so an insert that returned is on disk; `write_snapshot` uses a per-writer unique `.tmp` name, `fsync`s the file, renames atomically and best-effort syncs the directory, cleaning the temporary file up on failure.
- Every file-backend mutation commits in memory first and persists second (`create_collection`, `add_docs`, `delete_docs_by_*`) with no rollback, so a failed write leaves memory and disk divergent and the retry fails with "already exists"/duplicate primary key.
- A failed `fs::rename` in `write_snapshot` returns the error without deleting the temporary file (the document and graph stores do clean it up).
- `VectorIndex::search` deep-clones each candidate's document JSON twice (once for the filter, once into the hit), buffers every matching document before `truncate(top_k)`, and has no cap on `top_k`, so a query against an N-document collection costs O(N) document clones plus O(N log N) sorting regardless of `top_k`.
- `Metric::score` recomputes the query's norm for every candidate, so a cosine search does 2N norm computations where N+1 suffice.
- `search` validates only the query vector's length: a query vector containing `NaN`/`±inf` silently scores `0.0` against every document (all results tie and are ordered by primary key) instead of returning a validation error.

## Routing Table

- `./mod.rs` → data types, schema validation, `Metric`/`normalize_score`, the `VectorStore` trait
- `./memory.rs` → `VectorIndex` engine, `Doc`/`CollectionData`, `prepare_docs`/`prepare_doc`/`parse_vector`/`check_value`, `InMemoryVectorStore`
- `./file.rs` → `LocalFileVectorStore`, snapshot load/write, collection-name validation

## Test Strategy

- Unit tests sit beside each file (`#[cfg(test)] mod tests`); `file.rs` also carries an in-file contract run for both backends.
- The cross-backend contract lives in `../tests/vector.rs` (`assert_vector_contract`: schema validation, lifecycle, insert validation, scores/ties, `top_k`/filters, deletion, missing collections).
- Scores and tie ordering are asserted numerically (`1.0`/`0.5`/`0.0`); no test covers concurrent mutations on one file-store root.
