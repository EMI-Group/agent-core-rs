//! Config-driven storage backend selection.
//!
//! Ports the selection half of the Python `openjiuwen/extensions/store/`
//! package: the memory, local-file, and SQLite backends live in the kernel
//! ([`openjiuwen_core::foundation::store`]) and this module only dispatches to
//! them from a [`StoreBackend`] value. Vendor backends keep their deferred
//! surfaces in child modules: [`redis`] (feature `redis`) for key-value and
//! [`elasticsearch`] for vectors. A deferred backend validates its
//! configuration and then answers
//! [`Error::Unsupported`](openjiuwen_common::error::Error::Unsupported); the
//! Python `GaussDbStore` has no Rust analogue because the kernel's document
//! contract replaces the `SQLAlchemy` engine (see `docs/PARITY.md`).
//!
//! Every factory takes a [`StoreBackend`] so callers can select a backend from
//! configuration (see [`StoreBackend::from_json`]) and hand it to
//! [`build_kv_store`], [`build_document_store`], or [`build_vector_store`].
#![allow(clippy::module_name_repetitions)] // `StoreKind`/`StoreBackend` are the contract's names.
use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::path::PathBuf;
use std::sync::Arc;
pub mod elasticsearch;
#[cfg(feature = "redis")]
pub mod redis;
pub use openjiuwen_core::foundation::store::{
    CollectionSchema, Document, DocumentStore, InMemoryDocumentStore, InMemoryKvStore,
    InMemoryVectorStore, KvStore, KvValue, LocalFileDocumentStore, LocalFileKvStore,
    LocalFileVectorStore, Query, VectorQuery, VectorSearchResult, VectorStore,
};
#[cfg(feature = "sqlite")]
pub use openjiuwen_core::foundation::store::{SqliteDocumentStore, SqliteKvStore};
/// The index-prefix default shared by the configuration decoders and the
/// `Elasticsearch` backend (Python's `ElasticsearchVectorStore` default).
fn default_index_prefix() -> String {
    "agent_vector".to_string()
}
/// The storage backend families this crate can select.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StoreKind {
    /// In-process memory (the behavioral reference).
    Memory,
    /// JSON files below a root directory.
    File,
    /// A SQLite database behind the `sqlite` feature.
    Sqlite,
    /// Redis key-value storage (deferred, feature `redis`).
    Redis,
    /// Elasticsearch vector storage (deferred).
    Elasticsearch,
}
/// A storage backend selected from configuration.
///
/// The JSON shape is internally tagged on `kind` (Python's `type`
/// discriminator is accepted by [`StoreBackend::from_json`] as a fallback), so
/// `{"kind": "memory"}` or
/// `{"kind": "elasticsearch", "url": "http://localhost:9200"}` round-trip
/// through serde.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum StoreBackend {
    /// In-process memory; no configuration.
    Memory,
    /// JSON files below `root`.
    File {
        /// Directory that holds one envelope per key or document.
        root: PathBuf,
    },
    /// A SQLite database identified by `url` (for example `sqlite::memory:`).
    Sqlite {
        /// `SQLx` connection string.
        url: String,
    },
    /// Redis key-value storage; deferred, so only the configuration is kept.
    Redis {
        /// Optional `redis://`, `rediss://`, `redis+cluster://`, or
        /// `rediss+cluster://` URL.
        url: Option<String>,
    },
    /// Elasticsearch vector storage; deferred, so only the configuration is
    /// kept.
    Elasticsearch {
        /// Elasticsearch endpoint, `http://` or `https://`.
        url: String,
        /// Index-name prefix; indices are named `"{index_prefix}__{collection}"`.
        #[serde(default = "default_index_prefix")]
        index_prefix: String,
    },
}
impl StoreBackend {
    /// The backend family of this value.
    #[must_use]
    pub const fn kind(&self) -> StoreKind {
        match self {
            Self::Memory => StoreKind::Memory,
            Self::File { .. } => StoreKind::File,
            Self::Sqlite { .. } => StoreKind::Sqlite,
            Self::Redis { .. } => StoreKind::Redis,
            Self::Elasticsearch { .. } => StoreKind::Elasticsearch,
        }
    }
    /// Decodes a backend from a JSON configuration object.
    ///
    /// The discriminator is `kind`, with `type` accepted as a fallback, and it
    /// is matched case-insensitively. Required fields are `root` for `file`,
    /// `url` for `sqlite` and `elasticsearch` (plus optional `index_prefix`
    /// defaulting to `"agent_vector"`), and optional `url` for `redis`.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::Validation`](openjiuwen_common::error::ErrorKind::Validation)
    /// when `value` is not an object, when the discriminator is missing or is
    /// not a string, when the kind is unknown, or when a required field is
    /// missing, empty, or of the wrong JSON type.
    pub fn from_json(value: &Value) -> Result<Self> {
        let object = value
            .as_object()
            .ok_or_else(|| Error::validation("store backend config must be a JSON object"))?;
        let discriminator = object
            .get("kind")
            .or_else(|| object.get("type"))
            .ok_or_else(|| {
                Error::validation(
                    "store backend config must carry a `kind` (or `type`) discriminator",
                )
            })?;
        let raw_kind = discriminator.as_str().ok_or_else(|| {
            Error::validation("store backend `kind` discriminator must be a string")
        })?;
        match raw_kind.to_ascii_lowercase().as_str() {
            "memory" => Ok(Self::Memory),
            "file" => Ok(Self::File {
                root: PathBuf::from(required_string(object, "file", "root")?),
            }),
            "sqlite" => Ok(Self::Sqlite {
                url: required_string(object, "sqlite", "url")?,
            }),
            "redis" => Ok(Self::Redis {
                url: optional_string(object, "redis", "url")?,
            }),
            "elasticsearch" => Ok(Self::Elasticsearch {
                url: required_string(object, "elasticsearch", "url")?,
                index_prefix: optional_string(object, "elasticsearch", "index_prefix")?
                    .unwrap_or_else(default_index_prefix),
            }),
            other => Err(Error::validation(format!(
                "unknown store backend kind `{other}`; expected one of: memory, file, sqlite, redis, elasticsearch"
            ))),
        }
    }
}
/// Reads a required, non-empty string field of a backend configuration object.
fn required_string(object: &Map<String, Value>, kind: &str, field: &str) -> Result<String> {
    let value = object.get(field).ok_or_else(|| {
        Error::validation(format!(
            "`{kind}` store backend requires a non-empty `{field}` string"
        ))
    })?;
    let text = value.as_str().ok_or_else(|| {
        Error::validation(format!(
            "`{kind}` store backend field `{field}` must be a string"
        ))
    })?;
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Err(Error::validation(format!(
            "`{kind}` store backend field `{field}` must not be empty"
        )));
    }
    Ok(trimmed.to_string())
}
/// Reads an optional string field of a backend configuration object.
///
/// A missing field, `null`, or a blank string all decode as `None`; any other
/// JSON type is a validation error.
fn optional_string(object: &Map<String, Value>, kind: &str, field: &str) -> Result<Option<String>> {
    match object.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(text)) if text.trim().is_empty() => Ok(None),
        Some(Value::String(text)) => Ok(Some(text.trim().to_string())),
        Some(_) => Err(Error::validation(format!(
            "`{kind}` store backend field `{field}` must be a string"
        ))),
    }
}
/// Builds a key-value store from `backend`.
///
/// # Errors
///
/// Returns the kernel's construction error for `file` and `sqlite` backends
/// (an unavailable `sqlite` feature is reported as
/// [`ErrorKind::Unsupported`](openjiuwen_common::error::ErrorKind::Unsupported)),
/// and [`ErrorKind::Unsupported`](openjiuwen_common::error::ErrorKind::Unsupported)
/// for the deferred `redis` backend and for `elasticsearch`, which is not a
/// key-value store.
pub async fn build_kv_store(backend: &StoreBackend) -> Result<Arc<dyn KvStore>> {
    match backend {
        StoreBackend::Memory => Ok(Arc::new(InMemoryKvStore::new())),
        StoreBackend::File { root } => Ok(Arc::new(LocalFileKvStore::new(root).await?)),
        StoreBackend::Sqlite { url } => sqlite_kv_store(url).await,
        StoreBackend::Redis { .. } => Err(Error::unsupported(
            "redis key-value store backend is deferred; use the in-memory, file, or sqlite backend",
        )),
        StoreBackend::Elasticsearch { .. } => Err(Error::unsupported(
            "elasticsearch is not a key-value store backend; use the in-memory, file, or sqlite backend",
        )),
    }
}
/// Builds a document store from `backend`.
///
/// # Errors
///
/// Returns the kernel's construction error for `file` and `sqlite` backends
/// (an unavailable `sqlite` feature is reported as
/// [`ErrorKind::Unsupported`](openjiuwen_common::error::ErrorKind::Unsupported)),
/// and [`ErrorKind::Unsupported`](openjiuwen_common::error::ErrorKind::Unsupported)
/// for the `redis` and `elasticsearch` backends, which have no document store.
pub async fn build_document_store(backend: &StoreBackend) -> Result<Arc<dyn DocumentStore>> {
    match backend {
        StoreBackend::Memory => Ok(Arc::new(InMemoryDocumentStore::new())),
        StoreBackend::File { root } => Ok(Arc::new(LocalFileDocumentStore::new(root).await?)),
        StoreBackend::Sqlite { url } => sqlite_document_store(url).await,
        StoreBackend::Redis { .. } => Err(Error::unsupported(
            "redis has no document store backend; use the in-memory, file, or sqlite backend",
        )),
        StoreBackend::Elasticsearch { .. } => Err(Error::unsupported(
            "elasticsearch has no document store backend; use the in-memory, file, or sqlite backend",
        )),
    }
}
/// Builds a vector store from `backend`.
///
/// # Errors
///
/// Returns the kernel's construction error for the `file` backend and
/// [`ErrorKind::Unsupported`](openjiuwen_common::error::ErrorKind::Unsupported)
/// for `sqlite` (the kernel has no SQLite vector backend), `redis` (deferred),
/// and `elasticsearch` (deferred).
pub async fn build_vector_store(backend: &StoreBackend) -> Result<Arc<dyn VectorStore>> {
    match backend {
        StoreBackend::Memory => Ok(Arc::new(InMemoryVectorStore::new())),
        StoreBackend::File { root } => Ok(Arc::new(LocalFileVectorStore::new(root).await?)),
        StoreBackend::Sqlite { .. } => Err(Error::unsupported(
            "sqlite has no vector store backend; use the in-memory or file vector store",
        )),
        StoreBackend::Redis { .. } => Err(Error::unsupported(
            "redis has no vector store backend; use the in-memory or file vector store",
        )),
        StoreBackend::Elasticsearch { .. } => Err(Error::unsupported(
            "elasticsearch vector store backend is deferred; use the in-memory or file backend",
        )),
    }
}
/// Builds the SQLite key-value store, or reports the missing feature.
#[cfg(feature = "sqlite")]
async fn sqlite_kv_store(url: &str) -> Result<Arc<dyn KvStore>> {
    Ok(Arc::new(SqliteKvStore::connect(url).await?))
}
/// Reports that the `sqlite` feature is off; keeps the factory match arms
/// uniform.
#[cfg(not(feature = "sqlite"))]
#[allow(clippy::unused_async)] // mirrors the async SQLite helper
async fn sqlite_kv_store(_url: &str) -> Result<Arc<dyn KvStore>> {
    Err(Error::unsupported(
        "sqlite key-value store backend is unavailable; enable the `sqlite` feature or use the in-memory or file backend",
    ))
}
/// Builds the SQLite document store, or reports the missing feature.
#[cfg(feature = "sqlite")]
async fn sqlite_document_store(url: &str) -> Result<Arc<dyn DocumentStore>> {
    Ok(Arc::new(SqliteDocumentStore::connect(url).await?))
}
/// Reports that the `sqlite` feature is off; keeps the factory match arms
/// uniform.
#[cfg(not(feature = "sqlite"))]
#[allow(clippy::unused_async)] // mirrors the async SQLite helper
async fn sqlite_document_store(_url: &str) -> Result<Arc<dyn DocumentStore>> {
    Err(Error::unsupported(
        "sqlite document store backend is unavailable; enable the `sqlite` feature or use the in-memory or file backend",
    ))
}
#[cfg(test)]
mod tests {
    // Python counterparts: `tests/unit_tests/extensions/store/test_es_vector_store.py`
    // (Elasticsearch, deferred) and `test_gauss_db.py` (Gauss document store, not
    // ported because the kernel document contract replaces the `SQLAlchemy` engine).
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use openjiuwen_common::error::ErrorKind;
    use openjiuwen_core::foundation::store::{FieldSchema, VectorDataType};
    use serde_json::json;
    #[test]
    fn from_json_reads_every_kind() {
        let memory = StoreBackend::from_json(&json!({"kind": "memory"})).unwrap();
        assert_eq!(memory.kind(), StoreKind::Memory);
        let memory_via_type = StoreBackend::from_json(&json!({"type": "MEMORY"})).unwrap();
        assert_eq!(memory_via_type, StoreBackend::Memory);
        let file = StoreBackend::from_json(&json!({"kind": "file", "root": "/tmp/store"})).unwrap();
        assert_eq!(
            file,
            StoreBackend::File {
                root: PathBuf::from("/tmp/store")
            }
        );
        assert_eq!(file.kind(), StoreKind::File);
        let sqlite =
            StoreBackend::from_json(&json!({"kind": "SQLite", "url": "sqlite::memory:"})).unwrap();
        assert_eq!(
            sqlite,
            StoreBackend::Sqlite {
                url: "sqlite::memory:".to_string()
            }
        );
        assert_eq!(sqlite.kind(), StoreKind::Sqlite);
        let redis_without_url = StoreBackend::from_json(&json!({"kind": "redis"})).unwrap();
        assert_eq!(redis_without_url, StoreBackend::Redis { url: None });
        assert_eq!(redis_without_url.kind(), StoreKind::Redis);
        let redis_with_url =
            StoreBackend::from_json(&json!({"kind": "redis", "url": "redis://localhost:6379"}))
                .unwrap();
        assert_eq!(
            redis_with_url,
            StoreBackend::Redis {
                url: Some("redis://localhost:6379".to_string())
            }
        );
        let es_default = StoreBackend::from_json(
            &json!({"kind": "elasticsearch", "url": "http://localhost:9200"}),
        )
        .unwrap();
        assert_eq!(
            es_default,
            StoreBackend::Elasticsearch {
                url: "http://localhost:9200".to_string(),
                index_prefix: "agent_vector".to_string()
            }
        );
        assert_eq!(es_default.kind(), StoreKind::Elasticsearch);
        let es_explicit = StoreBackend::from_json(
            &json!({"kind": "elasticsearch", "url": "http://localhost:9200", "index_prefix": "custom"}),
        )
        .unwrap();
        assert_eq!(
            es_explicit,
            StoreBackend::Elasticsearch {
                url: "http://localhost:9200".to_string(),
                index_prefix: "custom".to_string()
            }
        );
    }
    #[test]
    fn from_json_rejects_invalid_configs() {
        for value in [
            json!({"kind": "unknown"}),
            json!({}),
            json!({"kind": 5}),
            json!([]),
        ] {
            let error = StoreBackend::from_json(&value).unwrap_err();
            assert_eq!(error.kind(), ErrorKind::Validation, "config: {value}");
        }
        let error = StoreBackend::from_json(&json!({"kind": "unknown"})).unwrap_err();
        let message = error.message().to_ascii_lowercase();
        for kind in ["memory", "file", "sqlite", "redis", "elasticsearch"] {
            assert!(
                message.contains(kind),
                "message must list `{kind}`: {message}"
            );
        }
        let error = StoreBackend::from_json(&json!({"kind": "file"})).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.message().contains("root"));
        let error = StoreBackend::from_json(&json!({"kind": "file", "root": ""})).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        let error = StoreBackend::from_json(&json!({"kind": "sqlite"})).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.message().contains("url"));
        let error = StoreBackend::from_json(&json!({"kind": "redis", "url": 42})).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        let error = StoreBackend::from_json(
            &json!({"kind": "elasticsearch", "url": "http://x", "index_prefix": 7}),
        )
        .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
    }
    #[test]
    fn store_backend_serde_round_trips() {
        let backends = [
            StoreBackend::Memory,
            StoreBackend::File {
                root: PathBuf::from("/tmp/store"),
            },
            StoreBackend::Sqlite {
                url: "sqlite::memory:".to_string(),
            },
            StoreBackend::Redis { url: None },
            StoreBackend::Redis {
                url: Some("redis://localhost:6379".to_string()),
            },
            StoreBackend::Elasticsearch {
                url: "http://localhost:9200".to_string(),
                index_prefix: "custom".to_string(),
            },
        ];
        for backend in &backends {
            let encoded = serde_json::to_value(backend).unwrap();
            assert_eq!(&StoreBackend::from_json(&encoded).unwrap(), backend);
        }
        let decoded: StoreBackend = serde_json::from_value(
            json!({"kind": "elasticsearch", "url": "http://localhost:9200"}),
        )
        .unwrap();
        assert_eq!(
            decoded,
            StoreBackend::Elasticsearch {
                url: "http://localhost:9200".to_string(),
                index_prefix: "agent_vector".to_string()
            }
        );
    }
    #[tokio::test]
    async fn memory_kv_store_round_trips() {
        let store = build_kv_store(&StoreBackend::Memory).await.unwrap();
        store.set("alpha", KvValue::text("one")).await.unwrap();
        assert_eq!(
            store.get("alpha").await.unwrap(),
            Some(KvValue::text("one"))
        );
        assert!(store.exists("alpha").await.unwrap());
        store.set("pre/a", KvValue::text("a")).await.unwrap();
        store.set("pre/b", KvValue::text("b")).await.unwrap();
        let prefixed = store.get_by_prefix("pre/").await.unwrap();
        assert_eq!(prefixed.len(), 2);
        assert_eq!(prefixed.get("pre/a"), Some(&KvValue::text("a")));
        store.delete("alpha").await.unwrap();
        assert!(!store.exists("alpha").await.unwrap());
        assert_eq!(store.get("alpha").await.unwrap(), None);
    }
    #[tokio::test]
    async fn file_kv_store_round_trips_and_persists() {
        let dir = tempfile::TempDir::new().unwrap();
        let backend = StoreBackend::File {
            root: dir.path().to_path_buf(),
        };
        let store = build_kv_store(&backend).await.unwrap();
        store
            .set("persisted", KvValue::text("value"))
            .await
            .unwrap();
        assert_eq!(
            store.get("persisted").await.unwrap(),
            Some(KvValue::text("value"))
        );
        drop(store);
        let reopened = build_kv_store(&backend).await.unwrap();
        assert_eq!(
            reopened.get("persisted").await.unwrap(),
            Some(KvValue::text("value"))
        );
    }
    #[tokio::test]
    async fn memory_document_store_factory_round_trips() {
        let store = build_document_store(&StoreBackend::Memory).await.unwrap();
        store.create_collection("docs").await.unwrap();
        assert_eq!(
            store.list_collections().await.unwrap(),
            vec!["docs".to_string()]
        );
    }
    #[tokio::test]
    async fn memory_vector_store_factory_creates_collections() {
        let store = build_vector_store(&StoreBackend::Memory).await.unwrap();
        let schema = CollectionSchema::new(vec![
            FieldSchema {
                name: "id".to_string(),
                dtype: VectorDataType::Varchar,
                is_primary: true,
                ..FieldSchema::default()
            },
            FieldSchema {
                name: "embedding".to_string(),
                dtype: VectorDataType::FloatVector,
                dim: Some(3),
                ..FieldSchema::default()
            },
        ]);
        store.create_collection("vectors", schema).await.unwrap();
        assert!(store.collection_exists("vectors").await.unwrap());
    }
    #[tokio::test]
    async fn redis_and_elasticsearch_backends_are_unsupported() {
        let redis = StoreBackend::Redis {
            url: Some("redis://localhost:6379".to_string()),
        };
        let elasticsearch = StoreBackend::Elasticsearch {
            url: "http://localhost:9200".to_string(),
            index_prefix: "agent_vector".to_string(),
        };
        for backend in [&redis, &elasticsearch] {
            let error = build_kv_store(backend)
                .await
                .err()
                .expect("deferred key-value backend must fail");
            assert_eq!(error.kind(), ErrorKind::Unsupported);
            assert!(
                error
                    .message()
                    .to_ascii_lowercase()
                    .contains(backend_name(backend)),
                "key-value error must name the backend: {}",
                error.message()
            );
            let error = build_document_store(backend)
                .await
                .err()
                .expect("deferred document backend must fail");
            assert_eq!(error.kind(), ErrorKind::Unsupported);
            assert!(error
                .message()
                .to_ascii_lowercase()
                .contains(backend_name(backend)));
            let error = build_vector_store(backend)
                .await
                .err()
                .expect("deferred vector backend must fail");
            assert_eq!(error.kind(), ErrorKind::Unsupported);
            assert!(error
                .message()
                .to_ascii_lowercase()
                .contains(backend_name(backend)));
        }
        let error = build_vector_store(&elasticsearch)
            .await
            .err()
            .expect("deferred vector backend must fail");
        assert!(error.message().contains("in-memory or file"));
    }
    /// The lowercase name a deferred backend's error message must contain.
    fn backend_name(backend: &StoreBackend) -> &'static str {
        match backend.kind() {
            StoreKind::Redis => "redis",
            StoreKind::Elasticsearch => "elasticsearch",
            _ => "store",
        }
    }
    #[cfg(feature = "sqlite")]
    #[tokio::test]
    async fn sqlite_factories_use_an_in_memory_database() {
        let backend = StoreBackend::Sqlite {
            url: "sqlite::memory:".to_string(),
        };
        let kv = build_kv_store(&backend).await.unwrap();
        kv.set("key", KvValue::text("value")).await.unwrap();
        assert_eq!(kv.get("key").await.unwrap(), Some(KvValue::text("value")));
        let documents = build_document_store(&backend).await.unwrap();
        documents.create_collection("docs").await.unwrap();
        assert_eq!(
            documents.list_collections().await.unwrap(),
            vec!["docs".to_string()]
        );
        let error = build_vector_store(&backend)
            .await
            .err()
            .expect("sqlite has no vector backend");
        assert_eq!(error.kind(), ErrorKind::Unsupported);
    }
    #[cfg(not(feature = "sqlite"))]
    #[tokio::test]
    async fn sqlite_kv_store_requires_the_feature() {
        let backend = StoreBackend::Sqlite {
            url: "sqlite::memory:".to_string(),
        };
        let error = build_kv_store(&backend)
            .await
            .err()
            .expect("sqlite backend without the feature must fail");
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        assert!(error.message().to_ascii_lowercase().contains("sqlite"));
    }
}
