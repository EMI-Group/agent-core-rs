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
- Crate-private `ElasticsearchClient` — holds `HttpClient` + config: `index_name(collection)`, `collection_from_index(name)`, `path(index, suffix)`, `indexes_path()`, `url(path)`, `apply(request)` (timeout, then Basic or ApiKey auth), `send_raw(request)`, `send(request, context)` (non-2xx → `failure`).
- Crate-private helpers shared by both stores: `similarity_for`, `field_mapping_with`, `properties_for`, `create_index_body`, `schema_from_index`, `schema_from_properties`, `index_names_from`, `validate_index_name`, `failure`, `excerpt`, `encode_segment`, `FILTER_DELETE_PAGE` (500), `DEFAULT_QUERY_LIMIT` (10 000), `ERROR_BODY_EXCERPT` (200).

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
- `delete_docs_by_filters` → existence probe, then a translatable filter becomes `POST /{index}/_delete_by_query` with `{"query":{…}}` (returns `deleted`); an untranslatable filter pages `_search` (ids only, bounded by `FILTER_DELETE_PAGE`/no-fresh-ids guard), keeps the ids whose `_source` satisfies `store::filter::matches`, then bulk-deletes them.
- An untranslatable **search** filter widens the `knn` window to `(4*top_k).max(top_k)` and re-checks hits locally — the fallback only ever sees that bounded candidate window.

Document adapter:

- `create_collection` → `PUT /{index}` with `{"mappings":{"properties":{"id":{"type":"keyword"}}}}`; a 400 `resource_already_exists_exception` is `Ok(())` (creation is idempotent); `list_collections`/`drop_collection` use the same index APIs (404 → empty / no-op).
- `upsert` → `HEAD /{index}`, on 404 `PUT` the permissive mapping, then `POST /{index}/_bulk` with `{"index":{"_id":<document id>}}` + the flat `Document::to_value()` source; returns the document count; any item error → `Transport`.
- `get` → `GET /{index}/_doc/{id}` (2xx → `Document::from_value(_source)`, with `_id` injected when the source lacks `id`; 404 → `None`).
- `query` → `POST /{index}/_search` with `{"query":<translated|match_all>,"sort":[…],"from":offset,"size":limit.unwrap_or(DEFAULT_QUERY_LIMIT),"track_total_hits":true}`; a missing index (404) is an empty result; ordering uses `{"field":{"order":"asc"|"desc","missing":"_first"|"_last","unmapped_type":"keyword"}}` (missing first for `Asc`, last for `Desc`, `sort` omitted when `order_by` is empty); an untranslatable filter pages a bounded `match_all` scan and delegates to the shared `document::run_query`, so filter/order/page semantics cannot drift.
- `count` → `POST /{index}/_count` (404 → 0; untranslatable filter → bounded scan + local count).
- `delete` → translatable filter: `_delete_by_query` returning `deleted` (404 → 0); otherwise the bounded scan + `_bulk` delete fallback counting only `result == "deleted"`.

## Constraints

- HTTP through `openjiuwen_common::http::HttpClient` only; NDJSON bulk bodies are built as `String` bytes (each line a compact JSON object) and sent with `Content-Type: application/x-ndjson`.
- Errors: `Validation` for unrepresentable schemas, invalid index/collection names, duplicate ids and bad query vectors; `Config` for an empty endpoint or `index_prefix`; `Transport` (via `failure`, naming status + URL + body excerpt) for service failures; `NotFound` for a missing collection on the vector delete paths; never panic, never `unwrap` outside tests.
- Collection names are validated with `document::validate_collection` and the resulting index name with `validate_index_name`: non-empty, lowercase, no `\ / * ? " < > |`, `, # :` or space, no leading `_`, `-` or `+`.
- `tracing` only, no `println!`, no `unsafe`, every public item documented (`cargo clippy --all-targets -- -D warnings`).
- Files stay around ≤1000 lines: `mod.rs` (1149 incl. its `#[cfg(test)]` tree), `vector.rs` 816, `document.rs` 559, `query.rs` 424; test trees live in `vector/tests.rs` and `document/tests.rs` behind `#[cfg(test)] mod tests;`.
- All tests are offline `wiremock`.

## Design Decisions

- **One shared `ElasticsearchClient`** — config validation, index naming/validation, auth header construction and status→error mapping exist once; both stores are thin method mappings plus their wire bodies.
- **`_meta.openjiuwen_schema` schema marker** — ES mappings cannot express `is_primary`, `auto_id`, `max_length`, `enable_dynamic_field` or `dim` for non-vector fields, so the serialized `CollectionSchema` travels in the index metadata (`create_index_body`) and is preferred by `schema_from_index`; `schema_from_properties` is the best-effort fallback for indexes created outside this adapter.
- **Primary key = `_id`** — a document's primary-key value is the ES `_id`, and `_source` also carries the field under the schema's name, so `VectorSearchResult.fields` and document round-trips stay flat.
- **Never drop a filter** — `query::translate` returns `None` instead of a partial clause; every caller then re-checks locally with `store::filter::matches` over a bounded window (10 000 documents for the document adapter, `4*top_k` candidates for vector search). Silent over-matching is not acceptable.
- **`$contains` is not translatable** — ES has no faithful equivalent of the shared substring/array-containment semantics, so it takes the local fallback; a top-level `$not` does translate to `bool.must_not`.
- **Stricter than the in-memory reference, same as Chroma** — an undeclared document field is rejected when `enable_dynamic_field` is false, and a `null` document value is omitted from the stored `_source` rather than stored as a mapping conflict.
- **Idempotency split** — the document contract makes `create_collection`/`drop_collection` idempotent (400 already-exists and 404 become no-ops), while the vector contract rejects a duplicate collection name, so the two adapters translate the same ES status differently on purpose.
- **Bounded result window** — when `Query::limit` is `None` the document adapter asks for `DEFAULT_QUERY_LIMIT` (10 000, ES's default `max_result_window`) instead of unbounded paging; the local fallbacks are capped by the same number.

## Known Issues

- No `refresh` parameter is ever sent, so a search immediately after `add_docs`/`upsert` observes ES's normal refresh interval — tests never depend on it, but a live caller must not assume read-after-write.
- The local fallbacks only see their bounded window (10 000 documents / `4*top_k` candidates), so a highly selective untranslatable filter can miss documents outside it; this is documented on the affected methods rather than silently returning wrong results.
- `schema_from_properties` cannot recover `is_primary` for a non-`id` primary key, `auto_id` or `enable_dynamic_field` from a foreign mapping; such an index round-trips with a synthesized `id` primary key and `enable_dynamic_field = false`.

## Routing Table

- `./mod.rs` → `ElasticsearchConfig`, shared `ElasticsearchClient`, index naming and validation, mapping ↔ `CollectionSchema` conversion, store re-exports
- `./query.rs` → filter → Elasticsearch query DSL translation (shared by both adapters), `None` = fall back to a local match
- `./vector.rs` → `ElasticsearchVectorStore` (`VectorStore`), batch validation, knn search, bulk deletes
- `./vector/tests.rs` → offline `wiremock` suite for the vector adapter
- `./document.rs` → `ElasticsearchDocumentStore` (`DocumentStore`), bulk upsert, search/count/delete with local fallbacks
- `./document/tests.rs` → offline `wiremock` suite for the document adapter
