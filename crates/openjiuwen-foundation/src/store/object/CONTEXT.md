# object — buckets of opaque byte objects

## Intent

The object storage seam: `ObjectStore` plus its reference backends.
Ports the abstract half of `openjiuwen/core/foundation/store/object/base_storage_client.py` (`BaseObjectStorageClient`) from openJiuwen v0.1.20; the Python aioboto3/S3 client is not ported (vendor backends are out of scope).

## API Surface

- `ObjectMeta { key, size, updated_at: DateTime<Utc> }` — serde `snake_case`, tolerant of unknown fields.
- `ObjectStore` (async, object-safe, `Send + Sync`): `create_bucket`, `delete_bucket`, `list_buckets`, `put(bucket, key, Bytes) -> ObjectMeta`, `get -> Option<Bytes>`, `delete`, `exists`, `list(bucket, prefix, limit) -> Vec<ObjectMeta>`.
- `InMemoryObjectStore::new()` + `Default`; `LocalObjectStore::new(root) -> Result<Self>` + `root()`.
- Re-exports: `pub use local::LocalObjectStore; pub use memory::InMemoryObjectStore;`.

## Constraints

- Missing bucket on `put`/`get`/`delete`/`exists`/`list` → `Error::not_found`; `delete_bucket` of a missing bucket is a no-op.
- Missing object: `get` → `Ok(None)`, `delete` → no-op, `exists` → `false`.
- `list` output is sorted ascending by key; `prefix` filters keys, `limit` truncates after sorting (`None` = unlimited).
- `put` reports `size` = payload byte length and `updated_at` = write instant.
- Local backend: `<root>/<bucket>/<key>` files, `tokio::fs` only, no blocking std fs on the async path; symlinks are skipped in listings.
- Validation (`Error::validation`): empty keys, absolute keys, backslashes, and `.`/`..`/empty segments; bucket names must be one non-empty segment without separators or NUL.
- No `unsafe`, no `unwrap`/`expect` outside tests; every public item documented; clippy pedantic clean.

## Design Decisions

- The local store keeps no in-memory index: buckets are discovered by scanning the root, so reopening a tree and sharing a root between stores is safe.
- Key validation happens before path construction (never after canonicalization), which keeps the backend independent of filesystem races.
- Tests use `tempfile::TempDir` and `#[tokio::test]`; each backend tests its own behavior plus the shared trait contract (missing bucket, sorted listings, deletion, persistence).
