# elasticsearch — vector and document stores over the Elasticsearch REST API

## Intent

Two adapters over one shared REST client: `ElasticsearchVectorStore` implements [`VectorStore`](crate::store::vector::VectorStore) on top of a `dense_vector` field, and `ElasticsearchDocumentStore` implements [`DocumentStore`](crate::store::document::DocumentStore) on top of plain index documents.
A store-level collection maps to an index named `{index_prefix}__{collection}` (`index_prefix` defaults to `openjiuwen`), so both adapters share one naming rule and one identity space.
Only the REST transport is used (`openjiuwen_common::http::HttpClient`); no vendor SDK and no dependency beyond what `crates/openjiuwen-foundation/Cargo.toml` already declares (`serde_json`, `base64`, `uuid`).

## API Surface

- `ElasticsearchConfig` — `endpoint: String`, `index_prefix: String` (default `"openjiuwen"`), `username: Option<String>`, `password: Option<SecretString>`, `api_key: Option<SecretString>`, `metric: Metric` (default `Cosine`), `timeout: Duration` (default 30 s); `#[serde(default, rename_all = "snake_case")]` + `Default` + builder setters (`new`, `with_index_prefix`, `with_username`, `with_password`, `with_api_key`, `with_metric`, `with_timeout`).
- `ElasticsearchVectorStore::{new(config), with_client(http, config), config()}` → `VectorStore`; also re-exported as `store::external::elasticsearch::ElasticsearchVectorStore`.
- `ElasticsearchDocumentStore::{new(config), with_client(http, config), config()}` → `DocumentStore`; also re-exported at the module root.
- `query::translate(&Value) -> Option<Value>` — the shared filter → query-DSL compiler; `None` means "not expressible".
- Crate-private `ElasticsearchClient` — holds `HttpClient` + config: `index_name(collection)`, `collection_from_index(name)`, `path(index, suffix)`, `indexes_path()`, `url(path)`, `apply(request)` (timeout, then Basic or ApiKey auth), `send_raw(request)`, `send(request, context)` (non-2xx → `failure`), and `scan(SearchScan, on_page)` — the one `search_after` pager both stores use.
- Crate-private helpers shared by both stores: `index_names_from`, `validate_index_name`, `failure`, `excerpt`, `encode_segment`, `stable_sort(order_by, key_field)`, `search_hits(payload)` → `(Vec<SearchHit>, cursor)`, `SearchScan { index, query, sort, needed, filter }`, `SearchHit { id, source }`, `SEARCH_PAGE_SIZE` (500), `MAX_SCAN_DOCUMENTS` (10 000), `ERROR_BODY_EXCERPT` (200), `LOG_TARGET`.
- The private `mapping` module owns the schema ↔ mapping direction: `similarity_for`, `field_mapping_with`, `properties_for`, `create_index_body`, `schema_from_index`, `schema_from_properties`, `field_from_mapping`, `SUPPORTED_DTYPES`, `SCHEMA_META_KEY`; only `create_index_body` and `schema_from_index` are re-exported for `vector.rs`.
- `field_from_mapping` is exhaustive over `SUPPORTED_DTYPES` (24 Elasticsearch dtypes): the keyword family (`keyword`, `text`, `constant_keyword`, `wildcard` → `Varchar` with `ignore_above` → `max_length`), `ip`/`binary`/`date`/`date_nanos` → `Varchar`, `integer`/`short`/`byte` → `Int32`, `long`/`unsigned_long` → `Int64`, `float`/`half_float`/`scaled_float` → `Float`, `double` → `Double`, `boolean` → `Bool`, `object`/`nested`/`flattened`/`geo_point`/`geo_shape` → `Json`, `dense_vector` → `FloatVector` (with `dims`). Any other dtype — `alias` (it resolves to another field), `percolator`, `some_future_type`, … — is an `Error::Validation` naming the field, the dtype and the supported list; a mapping with no string `type` is the same error.

## Wire contract

Vector adapter:

- `create_collection` → `PUT /{index}` with `{"mappings":{"_meta":{"openjiuwen_schema":"<CollectionSchema JSON>"},"properties":{…}}}`; `FloatVector` → `{"type":"dense_vector","dims":N,"index":true,"similarity":"cosine"|"dot_product"|"l2_norm"}`, `Varchar`→`keyword` (+`ignore_above` from `max_length`), `Int64`→`long`, `Int32`→`integer`, `Float`→`float`, `Double`→`double`, `Bool`→`boolean`, `Json`→`object`; `Array` is rejected with `Error::Validation`; a 400 mentioning `resource_already_exists_exception` becomes `Validation` (the reference backend rejects duplicate collection names).
- `collection_exists` → `HEAD /{index}` (2xx → true, 404 → false); `delete_collection` → `DELETE /{index}` (404 is a no-op).
- `get_schema` → `GET /{index}` → `schema_from_index` (404 → `Ok(None)`): the `_meta.openjiuwen_schema` marker is preferred and restores the exact schema (including `is_primary`/`auto_id`/`dim`/`max_length`/`enable_dynamic_field`); without it the schema is rebuilt from `mappings.properties`, synthesizing a `Varchar` primary-key field named `id` when the mapping has none.
- `list_collection_names` → `GET /{prefix}__*` (200 → object whose keys are index names; 404 → empty), prefix stripped, deduped and sorted.
- `add_docs` → `POST /{index}/_bulk` NDJSON (`Content-Type: application/x-ndjson`) with `{"create":{"_id":<pk>}}` + `_source` line pairs; the whole batch is validated locally first (defaults applied, primary key present / `auto_id` → uuid, batch-unique, every vector field at `dim` finite components, scalar types) so a failed call inserts nothing; item errors `version_conflict_engine_exception` / `document_already_exists` / status 409 → `Validation`, any other item error → `Transport`.
- `search` → `POST /{index}/_search` with `{"knn":{"field":<vector field>,"query_vector":[…],"k":<top_k>,"num_candidates":<max(4*top_k, top_k)>,"filter":{…}},"size":<top_k>,"_source":true}`; the filter is the translated `VectorQuery.filters`; `top_k == 0` returns an empty vector without a request; a non-vector `vector_field` or a query vector of the wrong length is a `Validation` error.
- `search` scores: `fields` is `_source` (primary key back-filled from `_id` when absent) and `score` is `_score` clamped into `[0,1]` (ES reports cosine/dot as a similarity in `[0,1]` and `l2_norm` as `1/(1+d)`); sort by score descending, tie by primary key ascending, truncate to `top_k`.
- `delete_docs_by_ids` → `HEAD /{index}` probe first (missing → `NotFound`), then one `_bulk` of `{"delete":{"_id":…}}`; counts only items whose `result == "deleted"` (`not_found` is ignored).
- `delete_docs_by_filters` → a translatable filter becomes `HEAD /{index}` then `POST /{index}/_delete_by_query` with `{"query":{…}}` (returns `deleted`); an untranslatable filter reads the schema (`GET /{index}`, a missing collection is `NotFound`) and pages `_search` with `search_after` (`{"query":{"match_all":{}},"_source":true,"size":min(SEARCH_PAGE_SIZE, …),"sort":[{"<primary key>":{"order":"asc","missing":"_last","unmapped_type":"keyword"}}]}`, continuing from the previous page's `sort` values), warns once, keeps the ids whose `_source` satisfies `store::filter::matches`, then bulk-deletes them; no `from`/`offset` request is ever sent.
- An untranslatable **search** filter widens the `knn` window to `(4*top_k).max(top_k)` and re-checks hits locally — the fallback only ever sees that bounded candidate window.

Document adapter:

- `create_collection` → `PUT /{index}` with `{"mappings":{"properties":{"id":{"type":"keyword"}}}}`; a 400 `resource_already_exists_exception` is `Ok(())` (creation is idempotent); `list_collections`/`drop_collection` use the same index APIs (404 → empty / no-op).
- `upsert` → `HEAD /{index}`, on 404 `PUT` the permissive mapping, then `POST /{index}/_bulk` with `{"index":{"_id":<document id>}}` + the flat `Document::to_value()` source; returns the document count; any item error → `Transport`.
- `get` → `GET /{index}/_doc/{id}` (2xx → `Document::from_value(_source)`, with `_id` injected when the source lacks `id`; 404 → `None`).
- `query` → `POST /{index}/_search` with `{"query":<translated|match_all>,"_source":true,"size":min(SEARCH_PAGE_SIZE, still-needed),"sort":[…],"search_after":[previous page's `sort` values]}`; `sort` is the caller's `order_by` criteria (`{"field":{"order":"asc"|"desc","missing":"_first"|"_last","unmapped_type":"keyword"}}`) plus the keyword `id` primary key as the stable tiebreaker when no criterion names it. Pages loop until a short page, `offset + limit` documents, or the cap; `offset`/`limit` are applied client-side to the assembled documents. A missing index (404) is an empty result. An untranslatable filter warns once and pages a `match_all` scan (same paging/cap), then delegates to the shared `document::run_query`, so filter/order/page semantics cannot drift.
- `count` → `POST /{index}/_count` (404 → 0; untranslatable filter → capped `match_all` scan + local count).
- `delete` → translatable filter: `_delete_by_query` returning `deleted` (404 → 0); otherwise the capped `match_all` scan + `_bulk` delete fallback counting only `result == "deleted"`.
- Every paged read of either adapter (`query` and the fallbacks of `query`/`count`/`delete`, plus the vector adapter's untranslatable-filter delete scan) runs through the shared `ElasticsearchClient::scan` and fails with `Error::Unsupported` after a `tracing::warn!` naming the cap and the filter as soon as it would need more than `MAX_SCAN_DOCUMENTS` (10 000) documents; it never returns a truncated list. A full page carrying no `sort` values, or a page that repeats documents, is `Error::Validation` because `search_after` cannot continue.

## Constraints

- HTTP through `openjiuwen_common::http::HttpClient` only; NDJSON bulk bodies are built as `String` bytes (each line a compact JSON object) and sent with `Content-Type: application/x-ndjson`.
- Errors: `Validation` for unrepresentable schemas, unrepresentable index dtypes, invalid index/collection names, duplicate ids, bad query vectors and a `search_after` page that cannot be continued; `Unsupported` for a read that would exceed `MAX_SCAN_DOCUMENTS`; `Config` for an empty endpoint or `index_prefix`; `Transport` (via `failure`, naming status + URL + body excerpt) for service failures; `NotFound` for a missing collection on the vector delete paths; never panic, never `unwrap` outside tests.
- Collection names are validated with `document::validate_collection` and the resulting index name with `validate_index_name`: non-empty, lowercase, no `\ / * ? " < > |`, `, # :` or space, no leading `_`, `-` or `+`.
- `tracing` only, no `println!`, no `unsafe`, every public item documented (`cargo clippy --all-targets -- -D warnings`).
- Files stay around ≤1000 lines: `vector.rs` 816, `mapping.rs` 710 (with its `#[cfg(test)]` tree), `mod.rs` 692 (with its test tree), `document.rs` 668, `query.rs` 424; test trees live in `mapping.rs`, `vector/tests.rs` and `document/tests.rs` behind `#[cfg(test)] mod tests;`.
- All tests are offline `wiremock`.

## Design Decisions

- **One shared `ElasticsearchClient`** — config validation, index naming/validation, auth header construction and status→error mapping exist once; both stores are thin method mappings plus their wire bodies.
- **`_meta.openjiuwen_schema` schema marker** — ES mappings cannot express `is_primary`, `auto_id`, `max_length`, `enable_dynamic_field` or `dim` for non-vector fields, so the serialized `CollectionSchema` travels in the index metadata (`create_index_body`) and is preferred by `schema_from_index`; `schema_from_properties` is the best-effort fallback for indexes created outside this adapter.
- **Primary key = `_id`** — a document's primary-key value is the ES `_id`, and `_source` also carries the field under the schema's name, so `VectorSearchResult.fields` and document round-trips stay flat.
- **Never drop a filter** — `query::translate` returns `None` instead of a partial clause; every caller then re-checks locally with `store::filter::matches` over a capped window (every document up to `MAX_SCAN_DOCUMENTS` for the document adapter, `4*top_k` candidates for vector search). Silent over-matching is not acceptable.
- **Exhaustive dtype table, no default type** — every Elasticsearch dtype either maps to a `VectorDataType` or fails with `Validation` naming the field, the dtype and `SUPPORTED_DTYPES`; an unmapped dtype is never collapsed onto `Varchar`/`Json`, so a foreign index cannot silently mis-type a field.
- **`$contains` is not translatable** — ES has no faithful equivalent of the shared substring/array-containment semantics, so it takes the local fallback; a top-level `$not` does translate to `bool.must_not`.
- **Stricter than the in-memory reference, same as Chroma** — an undeclared document field is rejected when `enable_dynamic_field` is false, and a `null` document value is omitted from the stored `_source` rather than stored as a mapping conflict.
- **Idempotency split** — the document contract makes `create_collection`/`drop_collection` idempotent (400 already-exists and 404 become no-ops), while the vector contract rejects a duplicate collection name, so the two adapters translate the same ES status differently on purpose.
- **`search_after` paging with a documented cap** — every read of both adapters pages with `search_after` through the shared `ElasticsearchClient::scan`, sorted by a total order (the document adapter: the caller's criteria plus the `id` keyword; the vector adapter: its schema's primary key), asks for `SEARCH_PAGE_SIZE` (500) documents at a time, and stops at `MAX_SCAN_DOCUMENTS` (10 000, ES's default `max_result_window`). Reaching the cap is an `Error::Unsupported` after one `tracing::warn!` — a truncated `Vec`/id set would be a silent wrong answer, and neither store contract has a "possibly truncated" channel.

## Known Issues

- No `refresh` parameter is ever sent, so a search immediately after `add_docs`/`upsert` observes ES's normal refresh interval — tests never depend on it, but a live caller must not assume read-after-write.
- A collection larger than `MAX_SCAN_DOCUMENTS` cannot be read through `query` (or an untranslatable-filter `count`/`delete`, or the vector adapter's untranslatable-filter delete scan) at all: the call fails with `Unsupported`, so a caller must narrow the filter or pass an explicit `limit` that keeps `offset + limit` under the cap.
- `search_after` is anchored on the keyword `id` field (the vector adapter: its schema's primary-key field), so a document whose `_source` lacks that field (an index created outside this adapter) sorts last and can be skipped once a page is full; `search_after` also needs the server's `sort` values, which a filtered or scripted sort can omit.
- `field_from_mapping` drops mapping options a `FieldSchema` cannot express (`index: false`, `doc_values`, `analyzer`, `scaled_float.scaling_factor`, sub-`properties` of an `object`), and an `alias` mapping is a typed `Validation` instead of being resolved to its target field.
- `schema_from_properties` cannot recover `is_primary` for a non-`id` primary key, `auto_id` or `enable_dynamic_field` from a foreign mapping; such an index round-trips with a synthesized `id` primary key and `enable_dynamic_field = false`.

## Routing Table

- `./mod.rs` → `ElasticsearchConfig`, shared `ElasticsearchClient`, index naming/validation, pagination constants, store re-exports
- `./mapping.rs` → schema ↔ `mappings.properties` conversion both ways, the exhaustive Elasticsearch dtype → `FieldSchema` table (`SUPPORTED_DTYPES`), the `_meta.openjiuwen_schema` marker
- `./query.rs` → filter → Elasticsearch query DSL translation (shared by both adapters), `None` = fall back to a local match
- `./vector.rs` → `ElasticsearchVectorStore` (`VectorStore`), batch validation, knn search, bulk deletes
- `./vector/tests.rs` → offline `wiremock` suite for the vector adapter
- `./document.rs` → `ElasticsearchDocumentStore` (`DocumentStore`), bulk upsert, `search_after` paging and the capped local fallbacks of query/count/delete
- `./document/tests.rs` → offline `wiremock` suite for the document adapter (including a `search_after`-aware responder that serves a paged fake index)
