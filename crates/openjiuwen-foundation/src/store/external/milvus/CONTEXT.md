# milvus — offline test suite and duplicate-primary-key contract of the Milvus backend

## Intent

Hold the offline `wiremock` suite of the Milvus `VectorStore` implementation and record what that implementation guarantees about duplicate primary keys.
The backend itself lives in `../milvus.rs` (v2 REST mapping under `/v2/vectordb/...`); nothing here is a library item.

## API Surface

The behaviour these notes and tests pin down (all of it public API of `../milvus.rs`):
- `MilvusConfig::reject_duplicate_primary_keys: bool` (`true` by default) and `MilvusConfig::with_reject_duplicate_primary_keys(bool)` — whether `add_docs` probes the collection before inserting.
- `MilvusVectorStore::add_docs` — with the probe on, it validates the whole batch locally, then queries `entities/query` with `filter: "<pk> in [...]"`, `outputFields: ["<pk>"]` and `limit: <batch length>` bounded by the batch, and returns `Error::Validation` naming the colliding primary keys when any is already stored; nothing is written in that case.
  The failing error message names the collection, the colliding keys (up to `KEY_EXCERPT = 8`), and the `reject_duplicate_primary_keys` setting that turns the check off.
  With the probe off, `add_docs` is the append the service offers: a duplicate primary key becomes a second row.
  A failing probe fails the insert; it never falls back to appending.
- Everything else (`create_collection`/`describe`/`search`/`delete_*`/`list_collection_names`, score normalization, filter compilation) is unchanged and unrelated to duplicates.

`tests.rs` (private to `milvus`):
- fixtures `schema()`, `auto_schema()`, `describe_data()`, `store()` and the `mount_describe`/`mount_ok` mount helpers.
- insert cases: `insert_sends_the_typed_fields_and_generates_auto_ids`, `insert_rejects_invalid_documents_without_writing`, `insert_falls_back_to_the_document_count`, `insert_probes_the_primary_keys_before_writing` (asserts the probe body), `insert_rejects_primary_keys_the_collection_already_stores` (asserts the validation error and that no insert request is sent), `insert_can_skip_the_duplicate_probe` (asserts no query request when disabled).
- pure unit tests for `compile_filter`, `parse_hits` and `normalize_score` live here too.

## Constraints

- Tests never reach a live Milvus: a `wiremock` loopback server stands in for the v2 API, with `{"code": 0, "data": ...}` envelopes.
- The probe is part of `add_docs`, so any new insert test must mount the `entities/query` response (usually `data: []`) or build the store with `with_reject_duplicate_primary_keys(false)`.
- The `VectorStore` trait is unchanged: the duplicate contract is expressed with the existing `Result<usize>` return and `Error::Validation`, not with a new result type.
- No credentials and no network in the suite; every case is deterministic and bounded.
