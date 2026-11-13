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
//!
//! The crate-level seams are re-exported here so consumers (memory, retrieval,
//! session, harness) can depend on `openjiuwen_foundation::store::{KvStore, ...}`
//! without importing the module paths.

pub mod document;
pub mod external;
pub mod filter;
pub mod graph;
pub mod kv;
pub mod object;
pub mod vector;

#[cfg(test)]
mod tests;

#[cfg(feature = "sqlite")]
pub use document::SqliteDocumentStore;
pub use document::{Document, DocumentStore, Query, SortOrder};
pub use document::{InMemoryDocumentStore, LocalFileDocumentStore};
pub use filter::{matches, matches_all};
pub use graph::{Direction, GraphEdge, GraphNode, GraphQuery, GraphStore};
pub use graph::{InMemoryGraphStore, LocalFileGraphStore};
#[cfg(feature = "sqlite")]
pub use kv::SqliteKvStore;
pub use kv::{InMemoryKvStore, KvPipeline, KvStore, KvValue, LocalFileKvStore};
pub use object::{InMemoryObjectStore, LocalObjectStore, ObjectMeta, ObjectStore};
pub use vector::{
    CollectionSchema, FieldSchema, InMemoryVectorStore, LocalFileVectorStore, Metric,
    VectorDataType, VectorQuery, VectorSearchResult, VectorStore,
};
