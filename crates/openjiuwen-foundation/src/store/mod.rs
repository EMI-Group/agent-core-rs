//! Storage backends: key-value, document, vector, graph, and object.
//!
//! Ports the abstract half of `openjiuwen/core/foundation/store/`: the traits and
//! plain data types every subsystem (memory, retrieval, session, harness) stores
//! through. The in-memory and local-filesystem implementations are the reference
//! behavior, plus SQLite behind the `sqlite` feature; network-backed adapters
//! (Redis, Chroma, Milvus, Elasticsearch, S3) live in [`external`].
//!
//! - [`kv`] — `KvStore` (+ `KvValue`, prefix scans, TTL, pipeline).
//! - [`document`] — `DocumentStore` (collections of JSON documents, filtered queries).
//! - [`vector`] — `VectorStore` (+ `CollectionSchema`, `FieldSchema`, search).
//! - [`graph`] — `GraphStore` (+ `GraphNode`, `GraphEdge`); in-process only.
//! - [`object`] — `ObjectStore` (buckets and opaque objects).
//! - [`filter`] — the shared equality/operator filter matcher used by every backend.
//! - [`external`] — network-backed adapters for the contracts above
//!   (`ChromaVectorStore`, `MilvusVectorStore`, `ElasticsearchVectorStore`,
//!   `ElasticsearchDocumentStore`, `RedisKvStore`, `S3ObjectStore`) plus the
//!   name-driven [`external::registry`].
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
pub use external::{
    available_backend_names, create_document_store, create_graph_store, create_kv_store,
    create_object_store, create_vector_store, global_registry, register_document_backend,
    register_graph_backend, register_kv_backend, register_object_backend, register_vector_backend,
    BackendKind, ChromaConfig, ChromaVectorStore, Credentials, DocumentStoreFactory,
    ElasticsearchConfig, ElasticsearchDocumentStore, ElasticsearchVectorStore, GraphStoreFactory,
    KvStoreFactory, MilvusConfig, MilvusVectorStore, ObjectStoreFactory, RedisConfig, RedisKvStore,
    S3Config, S3ObjectStore, StoreFamily, StoreRegistry, StoreSpec, VectorStoreFactory,
};
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
