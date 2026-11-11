//! Storage backends: key-value, document, vector, graph, and object.
//!
//! Ports the abstract half of `openjiuwen/core/foundation/store/`: the traits and
//! plain data types every subsystem (memory, retrieval, session, harness) stores
//! through. Vendor backends (Redis, Milvus, Chroma, Elasticsearch, pgvector, S3)
//! are out of scope; the in-memory and local-filesystem implementations are the
//! reference behavior, plus SQLite behind the `sqlite` feature.
//!
//! - [`kv`] — `KvStore` (+ `KvValue`, prefix scans, TTL, pipeline).
//! - [`document`] — `DocumentStore` (collections of JSON documents, filtered queries).
//! - [`vector`] — `VectorStore` (+ `CollectionSchema`, `FieldSchema`, search).
//! - [`graph`] — `GraphStore` (+ `GraphNode`, `GraphEdge`).
//! - [`object`] — `ObjectStore` (buckets and opaque objects).
//! - [`filter`] — the shared equality/operator filter matcher used by every backend.

pub mod document;
pub mod filter;
pub mod graph;
pub mod kv;
pub mod object;
pub mod vector;
