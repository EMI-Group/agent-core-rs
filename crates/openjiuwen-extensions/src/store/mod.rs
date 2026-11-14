//! Config-driven storage backend selection.
//!
//! Ports the selection half of the Python `openjiuwen/extensions/store/`
//! package: the memory, local-file, and SQLite backends live in the kernel
//! ([`openjiuwen_core::foundation::store`]) and this module only dispatches to
//! them from a [`StoreBackend`] value. Vendor backends keep their surfaces in
//! child modules: `redis` (feature `redis`) is a live single-node key-value
//! store over this crate's RESP client, and [`elasticsearch`] is the vector
//! backend. An `elasticsearch` backend also serves the document factory, which
//! builds the kernel's live REST document store
//! ([`ElasticsearchDocumentStore`]); it is not a key-value store. A backend
//! without a surface for the requested factory answers
//! [`Error::Unsupported`]; the Python `GaussDbStore` has no Rust analogue
//! because the kernel's document contract replaces the `SQLAlchemy` engine (see
//! `docs/PARITY.md`).
//!
//! The `redis` backend stores every value as a one-byte kind tag followed by the
//! payload so a [`KvValue`] keeps its `Text`/`Bytes` variant across a round trip;
//! that is an intentional deviation from Python's `RedisStore`, which writes the
//! bare payload. TLS (`rediss://`) and cluster slot routing stay deferred.
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
use url::Url;
pub mod elasticsearch;
#[cfg(feature = "redis")]
pub mod redis;
pub use openjiuwen_core::foundation::store::{
    CollectionSchema, Document, DocumentStore, ElasticsearchConfig, ElasticsearchDocumentStore,
    InMemoryDocumentStore, InMemoryKvStore, InMemoryVectorStore, KvStore, KvValue,
    LocalFileDocumentStore, LocalFileKvStore, LocalFileVectorStore, Metric, Query, SortOrder,
    VectorQuery, VectorSearchResult, VectorStore,
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
    /// Redis key-value storage (live over the in-crate client, feature
    /// `redis`).
    Redis,
    /// Elasticsearch storage; live REST backends for documents and vectors.
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
    /// Redis key-value storage; the `url` configures the in-crate RESP client.
    Redis {
        /// Optional `redis://`, `rediss://`, `redis+cluster://`, or
        /// `rediss+cluster://` URL.
        url: Option<String>,
    },
    /// Elasticsearch storage; the document and vector factories build live
    /// REST stores, while the key-value factory rejects it.
    ///
    /// Credentials may come from the URL userinfo
    /// (`http://user:pass@host:9200`) or from the explicit `username` /
    /// `password` fields; when both are present the explicit fields win, field
    /// by field. An `api_key` is sent as `Authorization: ApiKey <key>`.
    Elasticsearch {
        /// Elasticsearch endpoint, `http://` or `https://`.
        url: String,
        /// Index-name prefix; indices are named `"{index_prefix}__{collection}"`.
        #[serde(default = "default_index_prefix")]
        index_prefix: String,
        /// Optional basic-auth user name, overriding the URL userinfo.
        #[serde(default)]
        username: Option<String>,
        /// Optional basic-auth password, overriding the URL userinfo.
        #[serde(default)]
        password: Option<String>,
        /// Optional API key, sent when no basic-auth pair applies.
        #[serde(default)]
        api_key: Option<String>,
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
    /// defaulting to `"agent_vector"`), and optional `url` for `redis`. The
    /// `elasticsearch` kind additionally accepts the optional `username`,
    /// `password`, and `api_key` credential fields.
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
                username: optional_string(object, "elasticsearch", "username")?,
                password: optional_string(object, "elasticsearch", "password")?,
                api_key: optional_string(object, "elasticsearch", "api_key")?,
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
/// Returns the kernel's construction error for the `file` backend, the client's
/// connection error for a `redis` backend (a missing `url` is a validation
/// error, and an unavailable `sqlite`/`redis` feature is reported as
/// [`ErrorKind::Unsupported`](openjiuwen_common::error::ErrorKind::Unsupported)),
/// and [`ErrorKind::Unsupported`](openjiuwen_common::error::ErrorKind::Unsupported)
/// for `elasticsearch`, which is a document and vector backend, not a key-value
/// store.
pub async fn build_kv_store(backend: &StoreBackend) -> Result<Arc<dyn KvStore>> {
    match backend {
        StoreBackend::Memory => Ok(Arc::new(InMemoryKvStore::new())),
        StoreBackend::File { root } => Ok(Arc::new(LocalFileKvStore::new(root).await?)),
        StoreBackend::Sqlite { url } => sqlite_kv_store(url).await,
        StoreBackend::Redis { url } => redis_kv_store(url.as_deref()).await,
        StoreBackend::Elasticsearch { .. } => Err(Error::unsupported(
            "elasticsearch is not a key-value store backend; use the in-memory, file, or sqlite backend",
        )),
    }
}
/// Builds a document store from `backend`.
///
/// The `elasticsearch` backend builds the kernel's live REST document store
/// ([`ElasticsearchDocumentStore`]); construction validates the URL and
/// performs no I/O, and every request happens inside a [`DocumentStore`]
/// operation.
///
/// # Errors
///
/// Returns the kernel's construction error for `file` and `sqlite` backends
/// (an unavailable `sqlite` feature is reported as
/// [`ErrorKind::Unsupported`](openjiuwen_common::error::ErrorKind::Unsupported)),
/// the `elasticsearch_document_store` configuration errors for the
/// `elasticsearch` backend, and
/// [`ErrorKind::Unsupported`](openjiuwen_common::error::ErrorKind::Unsupported)
/// for the `redis` backend, which has no document store.
pub async fn build_document_store(backend: &StoreBackend) -> Result<Arc<dyn DocumentStore>> {
    match backend {
        StoreBackend::Memory => Ok(Arc::new(InMemoryDocumentStore::new())),
        StoreBackend::File { root } => Ok(Arc::new(LocalFileDocumentStore::new(root).await?)),
        StoreBackend::Sqlite { url } => sqlite_document_store(url).await,
        StoreBackend::Redis { .. } => Err(Error::unsupported(
            "redis has no document store backend; use the in-memory, file, or sqlite backend",
        )),
        StoreBackend::Elasticsearch {
            url,
            index_prefix,
            username,
            password,
            api_key,
        } => Ok(Arc::new(elasticsearch_document_store(
            url,
            index_prefix,
            username.as_deref(),
            password.as_deref(),
            api_key.as_deref(),
        )?)),
    }
}
/// Builds a vector store from `backend`.
///
/// `StoreBackend` carries no vector dimension, so the `elasticsearch` backend
/// uses its own default (`768`).
///
/// # Errors
///
/// Returns the kernel's construction error for the `file` backend and
/// [`ErrorKind::Unsupported`](openjiuwen_common::error::ErrorKind::Unsupported)
/// for `sqlite` (the kernel has no SQLite vector backend) and `redis` (not a
/// vector store), plus the `elasticsearch` backend's own construction error.
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
        StoreBackend::Elasticsearch {
            url, index_prefix, ..
        } => {
            let config = elasticsearch::ElasticsearchVectorStoreConfig {
                url: url.clone(),
                index_prefix: index_prefix.clone(),
                ..elasticsearch::ElasticsearchVectorStoreConfig::default()
            };
            elasticsearch::ElasticsearchVectorStore::connect(config).await
        }
    }
}
/// Builds the live REST document store behind the `elasticsearch` backend.
///
/// Construction validates the configuration (`elasticsearch_document_config`)
/// and performs no I/O, so a factory call stays offline; every request happens
/// inside a [`DocumentStore`] operation.
///
/// # Errors
///
/// Returns [`ErrorKind::Validation`](openjiuwen_common::error::ErrorKind::Validation)
/// for an unusable `url` and the kernel's configuration error for an endpoint it
/// cannot accept.
fn elasticsearch_document_store(
    url: &str,
    index_prefix: &str,
    username: Option<&str>,
    password: Option<&str>,
    api_key: Option<&str>,
) -> Result<ElasticsearchDocumentStore> {
    let config = elasticsearch_document_config(url, index_prefix, username, password, api_key)?;
    ElasticsearchDocumentStore::new(config)
}

/// Translates an `elasticsearch` backend configuration into a kernel
/// [`ElasticsearchConfig`].
///
/// The endpoint handed to the kernel is the parsed URL stripped of userinfo,
/// query, fragment, and trailing slash. The index prefix is forwarded verbatim.
///
/// # Credentials
///
/// Basic-auth credentials come from the URL userinfo
/// (`http://user:pass@host:9200`) unless the explicit `username` / `password`
/// arguments are set, which win field by field: an explicit user name replaces
/// the userinfo user name and an explicit password replaces the userinfo
/// password. A user name without a password (either source) sends an empty
/// password, matching the vector backend; a password without a user name is not
/// sent at all, because HTTP basic authentication needs both. An `api_key` is
/// forwarded as `Authorization: ApiKey <key>`, which the kernel ranks below
/// basic auth.
///
/// # Errors
///
/// Returns [`ErrorKind::Validation`](openjiuwen_common::error::ErrorKind::Validation)
/// when `url` is empty, does not start with `http://` or `https://`, is not a
/// parseable URL, or names no host.
fn elasticsearch_document_config(
    url: &str,
    index_prefix: &str,
    username: Option<&str>,
    password: Option<&str>,
    api_key: Option<&str>,
) -> Result<ElasticsearchConfig> {
    let parsed = elasticsearch_url(url)?;
    let userinfo_username = (!parsed.username().is_empty()).then(|| parsed.username().to_string());
    let userinfo_password = parsed.password().map(str::to_string);
    let username = username.map(str::to_string).or(userinfo_username);
    let password = match (password.map(str::to_string), username.is_some()) {
        (Some(explicit), _) => Some(explicit),
        (None, true) => Some(userinfo_password.unwrap_or_default()),
        (None, false) => None,
    };
    let mut config = ElasticsearchConfig::new(endpoint_without_userinfo(&parsed))
        .with_index_prefix(index_prefix);
    if let Some(username) = username {
        config = config.with_username(username);
    }
    if let Some(password) = password {
        config = config.with_password(password);
    }
    if let Some(api_key) = api_key {
        config = config.with_api_key(api_key);
    }
    Ok(config)
}

/// Parses and validates an `elasticsearch` backend URL.
fn elasticsearch_url(url: &str) -> Result<Url> {
    let url = url.trim();
    if url.is_empty() {
        return Err(Error::validation(
            "elasticsearch backend requires a non-empty `url`",
        ));
    }
    if !(url.starts_with("http://") || url.starts_with("https://")) {
        return Err(Error::validation(format!(
            "elasticsearch url `{url}` must start with http:// or https://"
        )));
    }
    let parsed = Url::parse(url).map_err(|error| {
        Error::validation(format!(
            "elasticsearch url `{url}` is not a valid URL: {error}"
        ))
    })?;
    if parsed.host_str().is_none() {
        return Err(Error::validation(format!(
            "elasticsearch url `{url}` must name a host"
        )));
    }
    Ok(parsed)
}

/// The endpoint without userinfo, query, fragment, or trailing slash.
fn endpoint_without_userinfo(parsed: &Url) -> String {
    let mut stripped = parsed.clone();
    let _ = stripped.set_username("");
    let _ = stripped.set_password(None);
    stripped.set_query(None);
    stripped.set_fragment(None);
    stripped.as_str().trim_end_matches('/').to_string()
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
/// Builds the Redis key-value store, or reports a missing `url`.
#[cfg(feature = "redis")]
async fn redis_kv_store(url: Option<&str>) -> Result<Arc<dyn KvStore>> {
    let url =
        url.ok_or_else(|| Error::validation("redis key-value store backend requires a `url`"))?;
    redis::RedisStore::connect(redis::RedisStoreConfig {
        url: Some(url.to_string()),
        ..redis::RedisStoreConfig::default()
    })
    .await
}
/// Reports that the `redis` feature is off; keeps the factory match arms
/// uniform.
#[cfg(not(feature = "redis"))]
#[allow(clippy::unused_async)] // mirrors the async Redis helper
async fn redis_kv_store(_url: Option<&str>) -> Result<Arc<dyn KvStore>> {
    Err(Error::unsupported(
        "redis key-value store backend is unavailable; enable the `redis` feature or use the in-memory, file, or sqlite backend",
    ))
}
#[cfg(test)]
mod tests {
    // Python counterparts: `tests/unit_tests/extensions/store/test_es_vector_store.py`
    // (Elasticsearch vector store, now live) and `test_gauss_db.py` (Gauss document
    // store, not ported because the kernel document contract replaces the
    // `SQLAlchemy` engine).
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use openjiuwen_common::error::ErrorKind;
    use openjiuwen_common::security::SecretString;
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
                index_prefix: "agent_vector".to_string(),
                username: None,
                password: None,
                api_key: None,
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
                index_prefix: "custom".to_string(),
                username: None,
                password: None,
                api_key: None,
            }
        );
        let es_credentials = StoreBackend::from_json(&json!({
            "kind": "elasticsearch",
            "url": "http://localhost:9200",
            "username": " elastic ",
            "password": "s3cret",
            "api_key": "  api-key-123  ",
        }))
        .unwrap();
        assert_eq!(
            es_credentials,
            StoreBackend::Elasticsearch {
                url: "http://localhost:9200".to_string(),
                index_prefix: "agent_vector".to_string(),
                username: Some("elastic".to_string()),
                password: Some("s3cret".to_string()),
                api_key: Some("api-key-123".to_string()),
            }
        );
        assert_eq!(es_credentials.kind(), StoreKind::Elasticsearch);
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
        for field in ["username", "password", "api_key"] {
            let error = StoreBackend::from_json(&json!({
                "kind": "elasticsearch",
                "url": "http://x",
                field: 7,
            }))
            .unwrap_err();
            assert_eq!(error.kind(), ErrorKind::Validation, "field: {field}");
            assert!(error.message().contains(field), "{}", error.message());
        }
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
                username: Some("elastic".to_string()),
                password: Some("s3cret".to_string()),
                api_key: Some("api-key-123".to_string()),
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
                index_prefix: "agent_vector".to_string(),
                username: None,
                password: None,
                api_key: None,
            },
            "a credential-less config still decodes with the credential fields empty"
        );
        // `index_prefix` and the credential fields all default, so the minimal
        // JSON shape keeps working.
        let minimal: StoreBackend = serde_json::from_value(
            json!({"kind": "elasticsearch", "url": "http://localhost:9200"}),
        )
        .unwrap();
        assert_eq!(minimal, decoded);
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
    /// Python counterpart: `tests/unit_tests/extensions/store/test_es_vector_store.py`.
    ///
    /// Elasticsearch serves documents and vectors but not key values: the
    /// key-value factory reports it and names a backend that does implement
    /// that contract, while the document arm builds the kernel's live REST
    /// document store and the vector arm builds this crate's live REST vector
    /// store. Neither constructor performs I/O, so the assertions stay offline.
    #[tokio::test]
    async fn elasticsearch_factories_split_key_value_document_and_vector() {
        let backend = StoreBackend::Elasticsearch {
            url: "http://localhost:9200".to_string(),
            index_prefix: "agent_vector".to_string(),
            username: None,
            password: None,
            api_key: None,
        };
        let error = build_kv_store(&backend)
            .await
            .err()
            .expect("elasticsearch is not a key-value store");
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        let message = error.message().to_ascii_lowercase();
        assert!(message.contains("elasticsearch"), "{message}");
        assert!(message.contains("in-memory"), "{message}");
        // The document factory builds a live REST store without any I/O, so a
        // loopback endpoint that is never contacted is enough.
        let documents: Arc<dyn DocumentStore> = build_document_store(&backend)
            .await
            .expect("elasticsearch is a live document store backend");
        drop(documents);
        build_vector_store(&backend)
            .await
            .expect("elasticsearch is a live vector store backend");
        // The factory passes `url` and `index_prefix` straight through, so the
        // concrete store built from them names indices the documented way.
        let config = elasticsearch::ElasticsearchVectorStoreConfig {
            url: "http://localhost:9200".to_string(),
            index_prefix: "agent_vector".to_string(),
            ..elasticsearch::ElasticsearchVectorStoreConfig::default()
        };
        let store = elasticsearch::ElasticsearchVectorStore::new(config)
            .expect("a valid elasticsearch config must build without I/O");
        assert_eq!(store.index_name("docs"), "agent_vector__docs");
    }
    /// The document helper forwards the endpoint and the index prefix, and
    /// strips everything a kernel endpoint cannot carry.
    #[test]
    fn elasticsearch_document_config_trims_the_endpoint_and_forwards_the_prefix() {
        let config = elasticsearch_document_config(
            "  http://localhost:9200/base/?pretty=1#top  ",
            "custom",
            None,
            None,
            None,
        )
        .unwrap();
        assert_eq!(config.endpoint, "http://localhost:9200/base");
        assert_eq!(config.index_prefix, "custom");
        assert_eq!(config.username, None);
        assert_eq!(config.password, None);
        assert_eq!(config.api_key, None);
    }
    /// Explicit credential fields win over the URL userinfo, field by field.
    #[test]
    fn elasticsearch_document_config_prefers_explicit_credentials() {
        // Userinfo only: both halves come from the URL.
        let config = elasticsearch_document_config(
            "http://alice:wonderland@127.0.0.1:9200",
            "agent_vector",
            None,
            None,
            None,
        )
        .unwrap();
        assert_eq!(config.endpoint, "http://127.0.0.1:9200");
        assert_eq!(config.username.as_deref(), Some("alice"));
        assert_eq!(
            config.password.as_ref().map(SecretString::expose_secret),
            Some("wonderland")
        );
        // Explicit fields replace the userinfo halves.
        let config = elasticsearch_document_config(
            "http://alice:wonderland@127.0.0.1:9200",
            "agent_vector",
            Some("bob"),
            Some("builder"),
            Some("api-key-123"),
        )
        .unwrap();
        assert_eq!(config.endpoint, "http://127.0.0.1:9200");
        assert_eq!(config.username.as_deref(), Some("bob"));
        assert_eq!(
            config.password.as_ref().map(SecretString::expose_secret),
            Some("builder")
        );
        assert_eq!(
            config.api_key.as_ref().map(SecretString::expose_secret),
            Some("api-key-123")
        );
        // A user name without a password sends an empty password, like the
        // vector backend's URL userinfo handling.
        let config = elasticsearch_document_config(
            "http://alice@127.0.0.1:9200",
            "agent_vector",
            None,
            None,
            None,
        )
        .unwrap();
        assert_eq!(config.username.as_deref(), Some("alice"));
        assert_eq!(
            config.password.as_ref().map(SecretString::expose_secret),
            Some("")
        );
        // A password without a user name is not sent: basic auth needs both.
        let config = elasticsearch_document_config(
            "http://127.0.0.1:9200",
            "agent_vector",
            None,
            Some("orphan"),
            None,
        )
        .unwrap();
        assert_eq!(config.username, None);
        assert_eq!(
            config.password.as_ref().map(SecretString::expose_secret),
            Some("orphan")
        );
        // The store exposes exactly the configuration the helper produced.
        let store = ElasticsearchDocumentStore::new(config)
            .expect("a host-only endpoint is a valid kernel configuration");
        assert_eq!(store.config().endpoint, "http://127.0.0.1:9200");
    }
    /// An unusable backend URL is rejected before any request is sent.
    #[test]
    fn elasticsearch_document_config_validates_the_url() {
        for url in [
            "",
            "   ",
            "localhost:9200",
            "ftp://localhost:9200",
            "http://",
        ] {
            let error =
                elasticsearch_document_config(url, "agent_vector", None, None, None).unwrap_err();
            assert_eq!(error.kind(), ErrorKind::Validation, "url: {url:?}");
            assert!(error.message().contains("url"), "{}", error.message());
        }
        let error =
            elasticsearch_document_config("localhost:9200", "agent_vector", None, None, None)
                .unwrap_err();
        assert!(
            error.message().contains("http://"),
            "the message names the scheme requirement: {}",
            error.message()
        );
    }
    /// The factory forwards the credentials of a configured backend.
    #[tokio::test]
    async fn elasticsearch_document_factory_forwards_the_credentials() {
        let store = elasticsearch_document_store(
            "http://alice:wonderland@127.0.0.1:9200",
            "agent_vector",
            Some("bob"),
            None,
            Some("api-key-123"),
        )
        .expect("the store builds without I/O");
        assert_eq!(store.config().endpoint, "http://127.0.0.1:9200");
        assert_eq!(store.config().username.as_deref(), Some("bob"));
        assert_eq!(
            store
                .config()
                .password
                .as_ref()
                .map(SecretString::expose_secret),
            Some("wonderland"),
            "the explicit user name keeps the userinfo password"
        );
        assert_eq!(
            store
                .config()
                .api_key
                .as_ref()
                .map(SecretString::expose_secret),
            Some("api-key-123")
        );
    }
    #[cfg(feature = "redis")]
    #[tokio::test]
    async fn redis_kv_store_round_trips_against_the_fake_server() {
        let fake = crate::redis::testsupport::FakeRedis::start().await;
        let backend = StoreBackend::Redis {
            url: Some(fake.url()),
        };
        let store = build_kv_store(&backend).await.unwrap();
        store.set("alpha", KvValue::text("one")).await.unwrap();
        assert_eq!(
            store.get("alpha").await.unwrap(),
            Some(KvValue::text("one"))
        );
        assert!(store.exists("alpha").await.unwrap());
        store.set("pre/a", KvValue::text("a")).await.unwrap();
        store
            .set("pre/b", KvValue::bytes(vec![0, 1, 2, 255]))
            .await
            .unwrap();
        let prefixed = store.get_by_prefix("pre/").await.unwrap();
        assert_eq!(prefixed.len(), 2);
        assert_eq!(prefixed.get("pre/a"), Some(&KvValue::text("a")));
        assert_eq!(
            prefixed.get("pre/b"),
            Some(&KvValue::bytes(vec![0, 1, 2, 255]))
        );
        let keys = vec!["alpha".to_string(), "missing".to_string()];
        assert_eq!(
            store.mget(&keys).await.unwrap(),
            vec![Some(KvValue::text("one")), None]
        );
        let mut pipeline = store.pipeline();
        pipeline.set("pipe", KvValue::text("value"));
        pipeline.get("pipe");
        pipeline.exists("pipe");
        assert_eq!(
            pipeline.execute().await.unwrap(),
            vec![
                None,
                Some(KvValue::text("value")),
                Some(KvValue::text("true")),
            ]
        );
        assert_eq!(store.delete_by_prefix("pre/").await.unwrap(), 2);
        store.delete("alpha").await.unwrap();
        assert!(!store.exists("alpha").await.unwrap());
        fake.shutdown().await;
    }
    #[cfg(feature = "redis")]
    #[tokio::test]
    async fn redis_kv_store_requires_a_url() {
        let error = build_kv_store(&StoreBackend::Redis { url: None })
            .await
            .err()
            .expect("a redis backend without a url must fail");
        assert_eq!(error.kind(), ErrorKind::Validation);
        let message = error.message().to_string();
        assert!(message.contains("url"), "{message}");
    }
    #[tokio::test]
    async fn redis_backend_has_no_document_or_vector_store() {
        let backend = StoreBackend::Redis {
            url: Some("redis://localhost:6379".to_string()),
        };
        let error = build_document_store(&backend)
            .await
            .err()
            .expect("redis is not a document store");
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        let message = error.message().to_ascii_lowercase();
        assert!(message.contains("redis"), "{message}");
        let error = build_vector_store(&backend)
            .await
            .err()
            .expect("redis is not a vector store");
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        let message = error.message().to_ascii_lowercase();
        assert!(message.contains("redis"), "{message}");
    }
    #[cfg(not(feature = "redis"))]
    #[tokio::test]
    async fn redis_kv_store_requires_the_feature() {
        let backend = StoreBackend::Redis {
            url: Some("redis://localhost:6379".to_string()),
        };
        let error = build_kv_store(&backend)
            .await
            .err()
            .expect("redis backend without the feature must fail");
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        let message = error.message().to_ascii_lowercase();
        assert!(message.contains("redis"), "{message}");
        assert!(message.contains("`redis` feature"), "{message}");
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
