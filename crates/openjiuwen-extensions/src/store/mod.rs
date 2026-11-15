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
//! bare payload. The Redis adapters accept `rediss://` (TLS) and
//! `redis[s]+cluster://` (cluster redirection) through the crate's own `redis` client.
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
/// `elasticsearch` backend, and the `redis` backend's configuration and
/// connection errors (a missing `url` is a validation error, and a disabled
/// `redis` feature is reported as
/// [`ErrorKind::Unsupported`](openjiuwen_common::error::ErrorKind::Unsupported)).
pub async fn build_document_store(backend: &StoreBackend) -> Result<Arc<dyn DocumentStore>> {
    match backend {
        StoreBackend::Memory => Ok(Arc::new(InMemoryDocumentStore::new())),
        StoreBackend::File { root } => Ok(Arc::new(LocalFileDocumentStore::new(root).await?)),
        StoreBackend::Sqlite { url } => sqlite_document_store(url).await,
        StoreBackend::Redis { url } => redis_document_store(url.as_deref()).await,
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
/// for `sqlite` (the kernel has no SQLite vector backend), the `redis`
/// backend's configuration and connection errors (a missing `url` is a
/// validation error, and a disabled `redis` feature is
/// [`ErrorKind::Unsupported`](openjiuwen_common::error::ErrorKind::Unsupported)),
/// plus the `elasticsearch` backend's own construction error.
pub async fn build_vector_store(backend: &StoreBackend) -> Result<Arc<dyn VectorStore>> {
    match backend {
        StoreBackend::Memory => Ok(Arc::new(InMemoryVectorStore::new())),
        StoreBackend::File { root } => Ok(Arc::new(LocalFileVectorStore::new(root).await?)),
        StoreBackend::Sqlite { .. } => Err(Error::unsupported(
            "sqlite has no vector store backend; use the in-memory or file vector store",
        )),
        StoreBackend::Redis { url } => redis_vector_store(url.as_deref()).await,
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

/// Builds the Redis document store, or reports a missing `url`.
///
/// The backend keeps the kernel [`DocumentStore`] semantics by delegating
/// filters, ordering, and paging to a throwaway in-memory store (see
/// [`redis`] for the key layout and the documented limitations).
#[cfg(feature = "redis")]
async fn redis_document_store(url: Option<&str>) -> Result<Arc<dyn DocumentStore>> {
    let url =
        url.ok_or_else(|| Error::validation("redis document store backend requires a `url`"))?;
    redis::RedisDocumentStore::connect(redis::RedisStoreConfig {
        url: Some(url.to_string()),
        ..redis::RedisStoreConfig::default()
    })
    .await
}

/// Reports that the `redis` feature is off; keeps the factory match arms
/// uniform.
#[cfg(not(feature = "redis"))]
#[allow(clippy::unused_async)] // mirrors the async Redis helper
async fn redis_document_store(_url: Option<&str>) -> Result<Arc<dyn DocumentStore>> {
    Err(Error::unsupported(
        "redis document store backend is unavailable; enable the `redis` feature or use the in-memory, file, or sqlite backend",
    ))
}

/// Builds the Redis vector store, or reports a missing `url`.
///
/// Search ranking, score normalization, and `top_k` truncation are delegated to
/// a throwaway [`InMemoryVectorStore`] with the store's metric, so they cannot
/// drift from the kernel reference (see [`redis`] for the brute-force note).
#[cfg(feature = "redis")]
async fn redis_vector_store(url: Option<&str>) -> Result<Arc<dyn VectorStore>> {
    let url =
        url.ok_or_else(|| Error::validation("redis vector store backend requires a `url`"))?;
    redis::RedisVectorStore::connect(redis::RedisStoreConfig {
        url: Some(url.to_string()),
        ..redis::RedisStoreConfig::default()
    })
    .await
}

/// Reports that the `redis` feature is off; keeps the factory match arms
/// uniform.
#[cfg(not(feature = "redis"))]
#[allow(clippy::unused_async)] // mirrors the async Redis helper
async fn redis_vector_store(_url: Option<&str>) -> Result<Arc<dyn VectorStore>> {
    Err(Error::unsupported(
        "redis vector store backend is unavailable; enable the `redis` feature or use the in-memory or file vector store",
    ))
}
#[cfg(test)]
mod tests;
