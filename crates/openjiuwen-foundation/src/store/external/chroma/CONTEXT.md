# chroma — schema metadata and the offline test suite of the Chroma backend

## Intent

Hold the two things the single-file Chroma backend (`../chroma.rs`) keeps out of itself: the collection-metadata encoding of a `CollectionSchema`, and the offline `wiremock` suite that proves the backend's wire behaviour.
`../chroma.rs` owns the REST mapping of `VectorStore` (`/api/v1/collections`, `/get`, `/add`, `/query`, `/delete`); this directory owns how a schema survives Chroma's untyped metadata and how that is tested.

## API Surface

`schema_metadata.rs` (private to `chroma`, `pub(super)` items only):
- `PYTHON_SCHEMA_KEY = "schema"`, `PYTHON_FIELDS_KEY = "fields"`, `LEGACY_SCHEMA_KEY = "openjiuwen:schema"` — the metadata keys the two implementations use; `LEGACY_SCHEMA_KEY` is also the default of `ChromaConfig::schema_metadata_key`.
- `insert(&mut Map<String, Value>, legacy_key: &str, &CollectionSchema) -> Result<()>` — writes the schema three times: under `schema` and `fields` in the Python `CollectionSchema.to_dict()` layout (a JSON string with `{fields: [{name, type, is_primary, auto_id, max_length, dim, default_value}], description, enable_dynamic_field}` and upper-case `VectorDataType` names such as `"FLOAT_VECTOR"`), and under `legacy_key` in this crate's serde layout (`dtype` + snake-case names) for readers written before the Python keys were understood.
- `decode(legacy_key: &str, collection: &str, Option<&Map<String, Value>>) -> Result<CollectionSchema>` — precedence `schema`, then `fields`, then `legacy_key`; every accepted layout is tolerated on every key (Python `type` + upper-case names, this crate's `dtype` + snake-case names, numeric strings for `dim`/`max_length`, a bare field array instead of a schema object).
- `accepted_keys(&str) -> Vec<String>` — the precedence list, used for the error messages.

Read failures:
- none of the accepted keys present → `Error::Unsupported` with the message `collection … was not created through ChromaVectorStore`, naming `"schema"`, `"fields"` and the legacy key.
- a key whose value is not a readable schema (invalid JSON, no `fields`, a field without `name`, an unknown data type) → `Error::Validation` naming the key, the offending field and every accepted key.
- Python-only data types `INT8`/`INT16` widen to `Int64` on read (this crate has no 8/16-bit variant, and Chroma metadata holds JSON scalars); `element_type`/`max_capacity` are ignored; a Python field's `description` is read but not written, matching Python's `FieldSchema.to_dict`.

`tests.rs` — the crate's only Chroma suite (everything runs against a loopback `wiremock` server, no network, no Chroma install):
- fixtures `schema()`, `metadata_for()`, `python_metadata()`, `python_written_schema()`/`python_read_schema()` (the Python `to_dict()` layout as a literal, independent of the writer), `doc()`, `store()`, `mount_collection`/`mount_metadata`.
- `RoundTripChroma` — a `wiremock::Respond` fake that stores the metadata of a created collection and serves it back on `GET`, so create→`get_schema` round trips are exercised end to end.

## Constraints

- Only `openjiuwen_common::error` and `crate::store::vector` types; no HTTP here (the store's `HttpClient` and request builder live in `../chroma.rs`).
- Every error names the metadata keys it accepts; a legitimately foreign collection is `Unsupported`, a corrupt one is `Validation` — never a silent default schema.
- Writes stay additive and backward compatible: neither key set may be dropped while collections written by an older release of this crate remain readable.
- Tests use `wiremock` matchers only, never a live server; new behaviour lands with a test in `tests.rs`, and pure helpers (`compile_where`, `normalize_distance`, `encode_segment`) keep their unit tests there.

## Routing Table

- `../chroma.rs` → `ChromaConfig`, `ChromaVectorStore`, the REST mapping, filter compilation and score normalization
- `./schema_metadata.rs` → Python/legacy collection-metadata encoding and decoding
- `./tests.rs` → the offline wire tests, including the Python-interop cases
