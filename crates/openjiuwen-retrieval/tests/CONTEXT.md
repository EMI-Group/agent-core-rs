# tests — cross-module integration

## Intent

Integration tests that exercise the public crate API end to end, beyond the unit tests beside each module.
All tests are offline and deterministic.

## API Surface

- `document_json.rs` → `Document`/`TextChunk`/result JSON field names (`id_`, `text`, `metadata`, `doc_id`, `embedding`) and round-trips.
- `chunking.rs` → char boundaries/overlap, recursive and token chunkers, `chunk_documents` metadata stamping.
- `vector_store.rs` → in-memory ordering, top-k, threshold, filters, upsert/delete, dimension mismatch, JSON persistence/reopen, registry.
- `loaders.rs` → text/JSON/CSV/HTML file loaders with `tempfile`, URL loader with `wiremock`, `AutoLoader` dispatch/stamping/errors.
- `knowledge_base.rs` → add/retrieve/filter/rerank/delete/statistics with `HashEmbedding`, multi-KB helpers.
- `embedding.rs` → `OpenAiEmbedding` request/response behavior with `wiremock`.
- `multimodal.rs` → `MultimodalDocument`/`ContentPart` JSON, data-URI round trips, `content()` shapes, `add_file`, `ImageLoader` metadata/errors, Markdown table round trip, row/column documents and table chunks.

## Constraints

- Every test file starts with `#![allow(clippy::unwrap_used, clippy::expect_used)]`.
- Use `tempfile::tempdir` for files and `wiremock::MockServer` for HTTP; never hit the network.
- Assert deterministic properties only (no random ids/order beyond documented guarantees).
- Behavior ports cite the Python test file in a comment where applicable.

## Routing Table

- `./document_json.rs` → data-model serialization
- `./chunking.rs` → chunking behavior
- `./vector_store.rs` → store behavior and persistence
- `./loaders.rs` → loader behavior
- `./knowledge_base.rs` → pipeline end to end
- `./embedding.rs` → OpenAI-compatible embedding client
- `./multimodal.rs` → multimodal documents, image loading, table conversion
