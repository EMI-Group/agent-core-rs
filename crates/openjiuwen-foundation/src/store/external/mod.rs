//! Network-backed ("external") storage backends.
//!
//! The five storage contracts in the parent module are implemented here on top of
//! external services: Chroma, Milvus and Elasticsearch for vectors, Elasticsearch
//! for documents, Redis for key-value, and any S3-compatible service for objects.
//! [`registry`] builds any of them (plus the in-memory, local-file and `SQLite`
//! backends) from a JSON configuration object.
//!
//! Every adapter speaks the service's own wire protocol directly — REST through
//! [`openjiuwen_common::http::HttpClient`],
//! RESP through [`tokio::net::TcpStream`] — because no vendor SDK can be added to
//! an offline build. The in-memory and local-filesystem backends of the parent
//! modules stay the behavioral reference; an adapter matches their semantics
//! wherever the service can express them and documents the difference where it
//! cannot.
//!
//! - [`chroma`] — `ChromaVectorStore` over the Chroma v1 REST API.
//! - [`milvus`] — `MilvusVectorStore` over the Milvus v2 REST API.
//! - [`elasticsearch`] — `ElasticsearchVectorStore` and `ElasticsearchDocumentStore`.
//! - [`redis`] — `RedisKvStore` plus an in-crate RESP2 codec and client.
//! - [`s3`] — `S3ObjectStore` plus an in-crate AWS Signature V4 signer.
//! - [`registry`] — name-driven backend selection from JSON configuration.
pub mod chroma;
pub mod elasticsearch;
pub mod milvus;
pub mod redis;
pub mod registry;
pub mod s3;
pub use chroma::{ChromaConfig, ChromaVectorStore};
pub use elasticsearch::{
    ElasticsearchConfig, ElasticsearchDocumentStore, ElasticsearchVectorStore,
};
pub use milvus::{MilvusConfig, MilvusVectorStore};
pub use redis::{RedisConfig, RedisKvStore};
pub use registry::{
    available_backend_names, create_document_store, create_graph_store, create_kv_store,
    create_object_store, create_vector_store, global_registry, register_document_backend,
    register_graph_backend, register_kv_backend, register_object_backend, register_vector_backend,
    BackendKind, DocumentStoreFactory, GraphStoreFactory, KvStoreFactory, ObjectStoreFactory,
    StoreFamily, StoreRegistry, StoreSpec, VectorStoreFactory,
};
pub use s3::{Credentials, S3Config, S3ObjectStore};
