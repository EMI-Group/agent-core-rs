# elasticsearch — vector and document stores over the Elasticsearch REST API

## Intent

Two adapters over one shared REST client: `ElasticsearchVectorStore` implements [`VectorStore`](crate::store::vector::VectorStore) on top of a `dense_vector` field, and `ElasticsearchDocumentStore` implements [`DocumentStore`](crate::store::document::DocumentStore) on top of plain index documents.
A store-level collection maps to an index named `{index_prefix}__{collection}` (`index_prefix` defaults to `openjiuwen`), so both adapters share one naming rule and one identity space.
Only the REST transport is used (`openjiuwen_common::http::HttpClient`); no vendor SDK, no new dependency.

## API Surface

- `ElasticsearchConfig` — `endpoint: String`, `index_prefix: String` (default `"openjiuwen"`), `username: Option<String>`, `password: Option<SecretString>`, `api_key: Option<SecretString>`, `metric: Metric` (default `Cosine`), `timeout: Duration` (default 30 s). `serde::Deserialize` with `#[serde(default)]` + `Default` + builder setters.
- `ElasticsearchVectorStore::{new(config), with_client(client, config)}` → `VectorStore`.
- `ElasticsearchDocumentStore::{new(config), with_client(client, config)}` → `DocumentStore`.
- `snapshot`/`config` accessors if useful.
- Internal (crate-private, shared): a client wrapper holding `HttpClient` + config, building request URLs, applying auth (`Authorization: Basic …` for user/password, `Authorization: ApiKey …` for an API key), mapping non-2xx statuses to `Error::External` with status + URL + body excerpt, and `index_name(collection)` / `collection_from_index(name)`.

## Wire contract

Vector adapter:

- `create_collection` → `PUT /{index}` with `{"mappings":{"properties":{…}}}`; the injected `id` field is `keyword` + primary, `FloatVector` → `{"type":"dense_vector","dims":N,"index":true,"similarity":"cosine"|"dot_product"|"l2_norm"}`, `Varchar`→`keyword`, `Int64`→`long`, `Int32`→`integer`, `Float`→`float`, `Double`→`double`, `Bool`→`boolean`, `Json`→`object`; `Array` is rejected with `Error::Validation`.
- `collection_exists` → `HEAD /{index}` (200 → true, 404 → false).
- `delete_collection` → `DELETE /{index}` (404 is a no-op).
- `get_schema` → `GET /{index}` → mappings → `CollectionSchema` (404 → `Ok(None)`), with the synthesized `id` primary key.
- `list_collection_names` → `GET /{prefix}__*` (200 → object whose keys are index names; 404 → empty), strip the prefix.
- `add_docs` → `POST /{index}/_bulk` NDJSON with `{"create":{"_id":<pk>}}` actions so a duplicate id fails (item error `document_already_exists` → `Error::Validation`); validate the schema and every document first (primary key present or `auto_id` → uuid, vector field length = `dim`, scalar types), exactly like the in-memory reference.
- `search` → `POST /{index}/_search` with `{"knn":{"field":<vector field>,"query_vector":[…],"k":<top_k>,"num_candidates":<k*4>,"filter":{…}},"size":<k>,"_source":true}`; the filter is the translated `VectorQuery.filters`; hits → `VectorSearchResult { score, fields }` where `fields` is `_source` (id included) and `score` is `_score` clamped to `[0,1]` (document why: ES scores cosine/dot as a similarity in `[0,1]` and `l2_norm` as `1/(1+d)`); sort by score descending, tie by primary key ascending, truncate to `top_k`.
- `delete_docs_by_ids` → `POST /{index}/_bulk` with `{"delete":{"_id":…}}` actions; count the items with `result == "deleted"`.
- `delete_docs_by_filters` → translatable filter: `POST /{index}/_delete_by_query` with `{"query":{…}}` and return `deleted`; untranslatable filter: page `_search` hits (bounded), keep the ids whose `_source` satisfies `store::filter::matches`, then bulk-delete them.

Document adapter:

- `create_collection`/`list_collections`/`drop_collection` on the same index APIs.
- `upsert` → `POST /{index}/_bulk` with `{"index":{"_id":<document id>}}` + the flat document (`Document::to_value`); count = number of documents; a missing index is created with a permissive mapping (`{"mappings":{"properties":{"id":{"type":"keyword"}}}}` or dynamic mappings) before the bulk write, so `upsert` auto-creates like the reference.
- `get` → `GET /{index}/_doc/{id}` (200 → `Document::from_value(_source)`, 404 → `None`).
- `query` → `POST /{index}/_search` with `query`, `sort`, `from`, `size`, `track_total_hits`; a missing index (404) is an empty result; `offset` → `from`, `limit` → `size` (documented default when `limit` is `None`, e.g. the 10 000-document result window); ordering uses `{"field":{"order":"asc","missing":"_first","unmapped_type":"keyword"}}` for `Asc` (missing last for `Desc`).
- `count` → `POST /{index}/_count` with `{"query":…}` (404 → 0).
- `delete` → translatable filter: `_delete_by_query` returning `deleted`; otherwise the bounded search-then-bulk-delete fallback (same as the vector store).
- Filter translation lives in `query.rs` and is shared: top-level `$and`/`$or`, plain-object implicit equality, `$eq`/`$ne`/`$gt`/`$gte`/`$lt`/`$lte`/`$in`/`$contains` → `term`/`terms`/`range`/`bool` clauses; `$not`, dotted paths beyond top-level field access, array-valued document fields, and anything else return `None` so the caller falls back to `match_all` plus a local `store::filter::matches` pass.

## Constraints

- HTTP through `openjiuwen_common::http::HttpClient` only; `NDJSON` bulk bodies are built as `String`/`Bytes` (each line a compact JSON object) and sent with `Content-Type: application/x-ndjson`.
- Errors: `Validation` for unrepresentable schemas, invalid index/collection names and duplicate ids; `Config` for unusable configuration; `External` for service failures; never panic, never `unwrap` outside tests.
- Collection names must be valid Elasticsearch index names (lowercase, no `\ / * ? " < > | , # :` and no leading `_`); validate and return `Error::Validation` with the offending name.
- Files stay around ≤1000 lines; test trees live in `<file>/tests.rs` or `tests.rs` next to the module (`#[cfg(test)] mod tests;`).
- All tests offline with `wiremock`.

## Routing Table

- `./mod.rs` → `ElasticsearchConfig`, shared client wrapper, index naming and validation, mapping ↔ `CollectionSchema` conversion, re-exports
- `./query.rs` → filter → Elasticsearch query DSL translation (shared by both adapters) and document↔source helpers
- `./vector.rs` → `ElasticsearchVectorStore` (`VectorStore`)
- `./document.rs` → `ElasticsearchDocumentStore` (`DocumentStore`)
