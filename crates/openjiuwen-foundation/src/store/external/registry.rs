//! Name-driven backend selection from JSON configuration.
//!
//! Builds any storage backend of the five contracts from a small JSON
//! specification, so a deployment can pick a backend without recompiling:
//!
//! ```
//! use openjiuwen_foundation::store::external::{BackendKind, StoreFamily, StoreSpec, available_backend_names};
//! use serde_json::json;
//!
//! let spec = StoreSpec::parse(&json!({"backend": "es", "endpoint": "http://localhost:9200"}))
//!     .expect("a known backend");
//! assert_eq!(spec.backend, BackendKind::Elasticsearch);
//! assert_eq!(spec.config["endpoint"], json!("http://localhost:9200"));
//!
//! assert_eq!(BackendKind::parse("Local"), Some(BackendKind::File));
//! assert!(available_backend_names(StoreFamily::Vector).contains(&"chroma"));
//! ```
//!
//! A specification is one of `{"backend": "memory", ...}`, `{"type": "memory",
//! ...}`, or the bare string `"memory"`. Everything except the `backend`/`type`
//! member becomes the backend's configuration and is decoded into that backend's
//! `*Config` type, so `{"backend": "chroma", "endpoint": "http://host:8000"}`
//! builds a [`ChromaVectorStore`] and `{"backend": "file", "root": "/var/lib/oj"}`
//! builds a [`LocalFileKvStore`],
//! [`LocalFileDocumentStore`],
//! [`LocalFileVectorStore`],
//! [`LocalObjectStore`], or
//! [`LocalFileGraphStore`], depending on
//! the family the caller asks for.
//!
//! # Plugin seam
//!
//! Extension crates register a factory under a new name with
//! [`register_kv_backend`] (or the document/vector/object/graph equivalent);
//! [`StoreRegistry::create_kv_store`] checks the registered names before the
//! built-ins, so a plugin can also override a built-in name for the process.
//! Registration is process-global and therefore the one documented mutable global
//! of the store subtree: [`StoreRegistry::new`] builds an isolated instance, which
//! is what tests should use.
//!
//! # Deferred backends
//!
//! `pgvector`, `postgres` and `gauss`/`gaussvector` are recognized names for
//! compatibility with the Python package, but their transports need a driver that
//! cannot be added to an offline build; they return
//! [`Error::Unsupported`] instead of
//! degrading silently.

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::sync::Arc;

use once_cell::sync::Lazy;
use openjiuwen_common::error::{Error, Result};
use parking_lot::RwLock;
use serde::de::DeserializeOwned;
use serde_json::Value;

use super::chroma::{ChromaConfig, ChromaVectorStore};
use super::elasticsearch::{
    ElasticsearchConfig, ElasticsearchDocumentStore, ElasticsearchVectorStore,
};
use super::milvus::{MilvusConfig, MilvusVectorStore};
use super::redis::{RedisConfig, RedisKvStore};
use super::s3::{S3Config, S3ObjectStore};
#[cfg(feature = "sqlite")]
use crate::store::document::SqliteDocumentStore;
use crate::store::document::{DocumentStore, InMemoryDocumentStore, LocalFileDocumentStore};
use crate::store::graph::{GraphStore, InMemoryGraphStore, LocalFileGraphStore};
#[cfg(feature = "sqlite")]
use crate::store::kv::SqliteKvStore;
use crate::store::kv::{InMemoryKvStore, KvStore, LocalFileKvStore};
use crate::store::object::{InMemoryObjectStore, LocalObjectStore, ObjectStore};
use crate::store::vector::{InMemoryVectorStore, LocalFileVectorStore, VectorStore};

/// The storage contracts a specification can be built for.
///
/// A backend belongs to one family (Chroma only implements vectors, Redis only
/// key-value), so asking for the wrong family is a configuration error and the
/// message lists the names that family does support.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StoreFamily {
    /// [`KvStore`] — key-value pairs.
    Kv,
    /// [`DocumentStore`] — collections of
    /// JSON documents.
    Document,
    /// [`VectorStore`] — embedding collections.
    Vector,
    /// [`ObjectStore`] — buckets of opaque
    /// objects.
    Object,
    /// [`GraphStore`] — nodes and edges.
    Graph,
}

impl StoreFamily {
    /// The lowercase family name used in diagnostics.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Kv => "kv",
            Self::Document => "document",
            Self::Vector => "vector",
            Self::Object => "object",
            Self::Graph => "graph",
        }
    }

    /// Every built-in (and deferred) backend name accepted for this family,
    /// sorted, including the aliases (`local`, `es`, `gauss`).
    #[must_use]
    pub const fn builtin_names(self) -> &'static [&'static str] {
        match self {
            Self::Kv => &["file", "local", "memory", "redis", "sqlite"],
            Self::Document => &["elasticsearch", "es", "file", "local", "memory", "sqlite"],
            Self::Vector => &[
                "chroma",
                "elasticsearch",
                "es",
                "file",
                "gauss",
                "gaussvector",
                "local",
                "memory",
                "milvus",
                "pgvector",
                "postgres",
            ],
            Self::Object => &["file", "local", "memory", "s3"],
            Self::Graph => &["file", "local", "memory"],
        }
    }
}

impl fmt::Display for StoreFamily {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// A storage backend family name a specification can select.
///
/// Built-ins are `memory`, `file`/`local`, `sqlite` (feature `sqlite`), `redis`,
/// `chroma`, `milvus`, `elasticsearch`/`es` and `s3`; `pgvector`, `postgres` and
/// `gauss`/`gaussvector` are deferred compatibility names that always fail with
/// [`Error::Unsupported`].
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendKind {
    /// In-memory backend of the requested family.
    Memory,
    /// Local-filesystem backend rooted at the configured `root`/`path`.
    File,
    /// SQLite backend (feature `sqlite`).
    Sqlite,
    /// Redis key-value store.
    Redis,
    /// Chroma vector store.
    Chroma,
    /// Milvus vector store.
    Milvus,
    /// Elasticsearch document and vector store.
    Elasticsearch,
    /// S3-compatible object store.
    S3,
    /// Deferred pgvector vector store (no PostgreSQL driver offline).
    Pgvector,
    /// Deferred `GaussDB` vector store (no `GaussDB` driver offline).
    GaussVector,
    /// Deferred PostgreSQL store (no PostgreSQL driver offline).
    Postgres,
}

impl BackendKind {
    /// Resolves a configuration name, case-insensitively.
    ///
    /// Accepts `file`/`local`, `elasticsearch`/`es` and `gauss`/`gaussvector` as
    /// aliases; returns `None` for an unknown name.
    #[must_use]
    pub fn parse(name: &str) -> Option<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "memory" => Some(Self::Memory),
            "file" | "local" => Some(Self::File),
            "sqlite" => Some(Self::Sqlite),
            "redis" => Some(Self::Redis),
            "chroma" => Some(Self::Chroma),
            "milvus" => Some(Self::Milvus),
            "elasticsearch" | "es" => Some(Self::Elasticsearch),
            "s3" => Some(Self::S3),
            "pgvector" => Some(Self::Pgvector),
            "gauss" | "gaussvector" => Some(Self::GaussVector),
            "postgres" => Some(Self::Postgres),
            _ => None,
        }
    }

    /// The canonical lowercase name of this backend.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Memory => "memory",
            Self::File => "file",
            Self::Sqlite => "sqlite",
            Self::Redis => "redis",
            Self::Chroma => "chroma",
            Self::Milvus => "milvus",
            Self::Elasticsearch => "elasticsearch",
            Self::S3 => "s3",
            Self::Pgvector => "pgvector",
            Self::GaussVector => "gaussvector",
            Self::Postgres => "postgres",
        }
    }

    /// Whether this name is recognized but has no offline transport.
    ///
    /// Deferred backends parse and are handled by the registry, but construction
    /// returns [`Error::Unsupported`] rather than a silently degraded store.
    #[must_use]
    pub const fn is_deferred(self) -> bool {
        matches!(self, Self::Pgvector | Self::GaussVector | Self::Postgres)
    }

    /// Whether this backend talks to a network service.
    const fn is_external(self) -> bool {
        matches!(
            self,
            Self::Redis
                | Self::Chroma
                | Self::Milvus
                | Self::Elasticsearch
                | Self::S3
                | Self::Pgvector
                | Self::GaussVector
                | Self::Postgres
        )
    }
}

impl fmt::Display for BackendKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// A parsed backend specification.
///
/// Produced by [`StoreSpec::parse`] or [`StoreSpec::from_map`] and consumed by
/// the [`StoreRegistry`] constructors. `name` keeps the spelling from the input
/// (the alias `es` stays `es`), while `backend` is the resolved kind.
#[derive(Debug, Clone, PartialEq)]
pub struct StoreSpec {
    /// The resolved backend kind.
    pub backend: BackendKind,
    /// The backend name as written in the specification.
    pub name: String,
    /// The backend configuration (a JSON object; empty when absent).
    pub config: Value,
}

impl StoreSpec {
    /// Parses `{"backend": ...}`, `{"type": ...}`, or a bare `"memory"` string.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for a specification without a name, a
    /// non-string or empty name, or an unknown backend name (the message lists
    /// every available name).
    pub fn parse(spec: &Value) -> Result<Self> {
        let (name, config) = raw_spec(spec)?;
        let backend = BackendKind::parse(&name).ok_or_else(|| unknown_backend(&name))?;
        Ok(Self {
            backend,
            name,
            config,
        })
    }

    /// Parses an already-collected map, equivalent to
    /// [`parse`](Self::parse) on the same JSON object.
    ///
    /// # Errors
    ///
    /// As [`parse`](Self::parse).
    pub fn from_map(map: &BTreeMap<String, Value>) -> Result<Self> {
        let object: serde_json::Map<String, Value> = map
            .iter()
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect();
        Self::parse(&Value::Object(object))
    }
}

/// Builds a [`KvStore`] from a specification.
///
/// # Errors
///
/// Returns [`Error::Validation`] for a malformed or unknown specification and for
/// a backend of another family, [`Error::Unsupported`] for a deferred or
/// feature-gated backend, and propagates the backend's own configuration,
/// transport, and timeout errors.
pub async fn create_kv_store(spec: &Value) -> Result<Arc<dyn KvStore>> {
    global_registry().create_kv_store(spec).await
}

/// Builds a [`DocumentStore`] from a specification.
///
/// # Errors
///
/// As [`create_kv_store`].
pub async fn create_document_store(spec: &Value) -> Result<Arc<dyn DocumentStore>> {
    global_registry().create_document_store(spec).await
}

/// Builds a [`VectorStore`] from a specification.
///
/// # Errors
///
/// As [`create_kv_store`].
pub async fn create_vector_store(spec: &Value) -> Result<Arc<dyn VectorStore>> {
    global_registry().create_vector_store(spec).await
}

/// Builds an [`ObjectStore`] from a specification.
///
/// # Errors
///
/// As [`create_kv_store`].
pub async fn create_object_store(spec: &Value) -> Result<Arc<dyn ObjectStore>> {
    global_registry().create_object_store(spec).await
}

/// Builds a [`GraphStore`] from a specification.
///
/// # Errors
///
/// As [`create_kv_store`], except that every external graph backend returns
/// [`Error::Unsupported`]: the `GraphStore` contract has no network adapter.
pub async fn create_graph_store(spec: &Value) -> Result<Arc<dyn GraphStore>> {
    global_registry().create_graph_store(spec).await
}

/// A factory that builds a [`KvStore`] from its JSON configuration.
pub type KvStoreFactory = Arc<dyn Fn(&Value) -> Result<Arc<dyn KvStore>> + Send + Sync>;

/// A factory that builds a [`DocumentStore`] from its JSON configuration.
pub type DocumentStoreFactory = Arc<dyn Fn(&Value) -> Result<Arc<dyn DocumentStore>> + Send + Sync>;

/// A factory that builds a [`VectorStore`] from its JSON configuration.
pub type VectorStoreFactory = Arc<dyn Fn(&Value) -> Result<Arc<dyn VectorStore>> + Send + Sync>;

/// A factory that builds an [`ObjectStore`] from its JSON configuration.
pub type ObjectStoreFactory = Arc<dyn Fn(&Value) -> Result<Arc<dyn ObjectStore>> + Send + Sync>;

/// A factory that builds a [`GraphStore`] from its JSON configuration.
pub type GraphStoreFactory = Arc<dyn Fn(&Value) -> Result<Arc<dyn GraphStore>> + Send + Sync>;

/// Every backend name a family accepts, sorted, aliases included.
///
/// This is the built-in view; [`StoreRegistry::available_names`] additionally
/// lists the names registered as plugins on that instance.
#[must_use]
pub fn available_backend_names(family: StoreFamily) -> Vec<&'static str> {
    family.builtin_names().to_vec()
}

/// The process-wide registry used by the free [`create_kv_store`] family.
///
/// Plugins registered through the free [`register_kv_backend`] family land here,
/// so a factory registered anywhere is visible to every caller of the free
/// constructors.
#[must_use]
// `once_cell` is already a workspace dependency and the module contract names it;
// `std::sync::LazyLock` would be an equivalent but unasked-for substitution, so
// the lint is silenced here rather than at crate level.
#[allow(clippy::non_std_lazy_statics)]
pub fn global_registry() -> &'static StoreRegistry {
    static GLOBAL: Lazy<StoreRegistry> = Lazy::new(StoreRegistry::new);
    Lazy::force(&GLOBAL)
}

/// Registers a plugin [`KvStore`] factory in the process-wide registry.
///
/// Returns the factory it replaced, if any — registering the same name again
/// overrides it, including a built-in name. Use [`StoreRegistry::new`] for an
/// isolated registry instead of this documented process-global state.
pub fn register_kv_backend(
    name: impl Into<String>,
    factory: KvStoreFactory,
) -> Option<KvStoreFactory> {
    global_registry().register_kv_backend(name, factory)
}

/// Registers a plugin [`DocumentStore`] factory in the process-wide registry.
///
/// As [`register_kv_backend`].
pub fn register_document_backend(
    name: impl Into<String>,
    factory: DocumentStoreFactory,
) -> Option<DocumentStoreFactory> {
    global_registry().register_document_backend(name, factory)
}

/// Registers a plugin [`VectorStore`] factory in the process-wide registry.
///
/// As [`register_kv_backend`].
pub fn register_vector_backend(
    name: impl Into<String>,
    factory: VectorStoreFactory,
) -> Option<VectorStoreFactory> {
    global_registry().register_vector_backend(name, factory)
}

/// Registers a plugin [`ObjectStore`] factory in the process-wide registry.
///
/// As [`register_kv_backend`].
pub fn register_object_backend(
    name: impl Into<String>,
    factory: ObjectStoreFactory,
) -> Option<ObjectStoreFactory> {
    global_registry().register_object_backend(name, factory)
}

/// Registers a plugin [`GraphStore`] factory in the process-wide registry.
///
/// As [`register_kv_backend`].
pub fn register_graph_backend(
    name: impl Into<String>,
    factory: GraphStoreFactory,
) -> Option<GraphStoreFactory> {
    global_registry().register_graph_backend(name, factory)
}

/// A set of backend factories: the plugin seam plus every built-in backend.
///
/// The built-ins need no registration; the five maps hold only extension
/// factories, keyed by their normalized (`trim` + lowercase) name. A registered
/// name wins over a built-in of the same spelling. The maps are the one
/// documented mutable global of the store subtree when this registry is
/// [`global_registry`]; [`StoreRegistry::new`] builds an isolated instance.
pub struct StoreRegistry {
    /// Key-value plugins.
    kv: RwLock<HashMap<String, KvStoreFactory>>,
    /// Document plugins.
    document: RwLock<HashMap<String, DocumentStoreFactory>>,
    /// Vector plugins.
    vector: RwLock<HashMap<String, VectorStoreFactory>>,
    /// Object plugins.
    object: RwLock<HashMap<String, ObjectStoreFactory>>,
    /// Graph plugins.
    graph: RwLock<HashMap<String, GraphStoreFactory>>,
}

impl StoreRegistry {
    /// Creates a registry with no plugins.
    #[must_use]
    pub fn new() -> Self {
        Self {
            kv: RwLock::new(HashMap::new()),
            document: RwLock::new(HashMap::new()),
            vector: RwLock::new(HashMap::new()),
            object: RwLock::new(HashMap::new()),
            graph: RwLock::new(HashMap::new()),
        }
    }

    /// Registers a key-value plugin, returning the factory it replaced.
    pub fn register_kv_backend(
        &self,
        name: impl Into<String>,
        factory: KvStoreFactory,
    ) -> Option<KvStoreFactory> {
        register(&self.kv, name, factory)
    }

    /// Registers a document plugin, returning the factory it replaced.
    pub fn register_document_backend(
        &self,
        name: impl Into<String>,
        factory: DocumentStoreFactory,
    ) -> Option<DocumentStoreFactory> {
        register(&self.document, name, factory)
    }

    /// Registers a vector plugin, returning the factory it replaced.
    pub fn register_vector_backend(
        &self,
        name: impl Into<String>,
        factory: VectorStoreFactory,
    ) -> Option<VectorStoreFactory> {
        register(&self.vector, name, factory)
    }

    /// Registers an object plugin, returning the factory it replaced.
    pub fn register_object_backend(
        &self,
        name: impl Into<String>,
        factory: ObjectStoreFactory,
    ) -> Option<ObjectStoreFactory> {
        register(&self.object, name, factory)
    }

    /// Registers a graph plugin, returning the factory it replaced.
    pub fn register_graph_backend(
        &self,
        name: impl Into<String>,
        factory: GraphStoreFactory,
    ) -> Option<GraphStoreFactory> {
        register(&self.graph, name, factory)
    }

    /// Every name this registry accepts for `family`, sorted.
    ///
    /// Includes the built-in names and the plugin names registered on this
    /// instance.
    #[must_use]
    pub fn available_names(&self, family: StoreFamily) -> Vec<String> {
        let mut names: Vec<String> = family
            .builtin_names()
            .iter()
            .map(|name| (*name).to_owned())
            .collect();
        names.extend(self.plugin_names(family));
        names.sort_unstable();
        names.dedup();
        names
    }

    /// Builds a [`KvStore`] from a specification.
    ///
    /// # Errors
    ///
    /// As the free [`create_kv_store`].
    pub async fn create_kv_store(&self, spec: &Value) -> Result<Arc<dyn KvStore>> {
        let (name, config) = raw_spec(spec)?;
        if let Some(factory) = registered(&self.kv, &name) {
            return factory(&config);
        }
        let spec = self.resolve(name, config, StoreFamily::Kv)?;
        match spec.backend {
            BackendKind::Memory => Ok(Arc::new(InMemoryKvStore::new())),
            BackendKind::File => {
                let root = config_root(&spec.config)?;
                Ok(Arc::new(LocalFileKvStore::new(root).await?))
            }
            #[cfg(feature = "sqlite")]
            BackendKind::Sqlite => {
                let url = config_url(&spec.config)?;
                Ok(Arc::new(SqliteKvStore::connect(url).await?))
            }
            #[cfg(not(feature = "sqlite"))]
            BackendKind::Sqlite => Err(Error::unsupported(
                "sqlite backend requires the sqlite feature",
            )),
            BackendKind::Redis => {
                let config = decode_config::<RedisConfig>(&spec)?;
                Ok(Arc::new(RedisKvStore::connect(&config).await?))
            }
            other => Err(self.family_mismatch(other, StoreFamily::Kv)),
        }
    }

    /// Builds a [`DocumentStore`] from a specification.
    ///
    /// # Errors
    ///
    /// As the free [`create_document_store`].
    pub async fn create_document_store(&self, spec: &Value) -> Result<Arc<dyn DocumentStore>> {
        let (name, config) = raw_spec(spec)?;
        if let Some(factory) = registered(&self.document, &name) {
            return factory(&config);
        }
        let spec = self.resolve(name, config, StoreFamily::Document)?;
        match spec.backend {
            BackendKind::Memory => Ok(Arc::new(InMemoryDocumentStore::new())),
            BackendKind::File => {
                let root = config_root(&spec.config)?;
                Ok(Arc::new(LocalFileDocumentStore::new(root).await?))
            }
            #[cfg(feature = "sqlite")]
            BackendKind::Sqlite => {
                let url = config_url(&spec.config)?;
                Ok(Arc::new(SqliteDocumentStore::connect(url).await?))
            }
            #[cfg(not(feature = "sqlite"))]
            BackendKind::Sqlite => Err(Error::unsupported(
                "sqlite backend requires the sqlite feature",
            )),
            BackendKind::Elasticsearch => {
                let config = decode_config::<ElasticsearchConfig>(&spec)?;
                Ok(Arc::new(ElasticsearchDocumentStore::new(config)?))
            }
            other => Err(self.family_mismatch(other, StoreFamily::Document)),
        }
    }

    /// Builds a [`VectorStore`] from a specification.
    ///
    /// # Errors
    ///
    /// As the free [`create_vector_store`].
    pub async fn create_vector_store(&self, spec: &Value) -> Result<Arc<dyn VectorStore>> {
        let (name, config) = raw_spec(spec)?;
        if let Some(factory) = registered(&self.vector, &name) {
            return factory(&config);
        }
        let spec = self.resolve(name, config, StoreFamily::Vector)?;
        match spec.backend {
            BackendKind::Memory => Ok(Arc::new(InMemoryVectorStore::new())),
            BackendKind::File => {
                let root = config_root(&spec.config)?;
                Ok(Arc::new(LocalFileVectorStore::new(root).await?))
            }
            BackendKind::Chroma => {
                let config = decode_config::<ChromaConfig>(&spec)?;
                Ok(Arc::new(ChromaVectorStore::new(config)?))
            }
            BackendKind::Milvus => {
                let config = decode_config::<MilvusConfig>(&spec)?;
                Ok(Arc::new(MilvusVectorStore::new(config)?))
            }
            BackendKind::Elasticsearch => {
                let config = decode_config::<ElasticsearchConfig>(&spec)?;
                Ok(Arc::new(ElasticsearchVectorStore::new(config)?))
            }
            BackendKind::Pgvector | BackendKind::Postgres => Err(Error::unsupported(format!(
                "the \"{}\" backend requires a PostgreSQL driver that is not available offline",
                spec.backend.as_str()
            ))),
            BackendKind::GaussVector => Err(Error::unsupported(
                "the \"gaussvector\" backend requires a GaussDB driver that is not available offline",
            )),
            other => Err(self.family_mismatch(other, StoreFamily::Vector)),
        }
    }

    /// Builds an [`ObjectStore`] from a specification.
    ///
    /// # Errors
    ///
    /// As the free [`create_object_store`].
    pub async fn create_object_store(&self, spec: &Value) -> Result<Arc<dyn ObjectStore>> {
        let (name, config) = raw_spec(spec)?;
        if let Some(factory) = registered(&self.object, &name) {
            return factory(&config);
        }
        let spec = self.resolve(name, config, StoreFamily::Object)?;
        match spec.backend {
            BackendKind::Memory => Ok(Arc::new(InMemoryObjectStore::new())),
            BackendKind::File => {
                let root = config_root(&spec.config)?;
                Ok(Arc::new(LocalObjectStore::new(root).await?))
            }
            BackendKind::S3 => {
                let config = decode_config::<S3Config>(&spec)?;
                Ok(Arc::new(S3ObjectStore::new(config)?))
            }
            other => Err(self.family_mismatch(other, StoreFamily::Object)),
        }
    }

    /// Builds a [`GraphStore`] from a specification.
    ///
    /// Only the in-memory and local-file graph backends exist; every external
    /// name returns [`Error::Unsupported`] because the `GraphStore` contract has
    /// no network adapter.
    ///
    /// # Errors
    ///
    /// As the free [`create_graph_store`].
    pub async fn create_graph_store(&self, spec: &Value) -> Result<Arc<dyn GraphStore>> {
        let (name, config) = raw_spec(spec)?;
        if let Some(factory) = registered(&self.graph, &name) {
            return factory(&config);
        }
        let spec = self.resolve(name, config, StoreFamily::Graph)?;
        match spec.backend {
            BackendKind::Memory => Ok(Arc::new(InMemoryGraphStore::new())),
            BackendKind::File => {
                let root = config_root(&spec.config)?;
                Ok(Arc::new(LocalFileGraphStore::new(root).await?))
            }
            other if other.is_external() => Err(Error::unsupported(format!(
                "backend \"{}\": external graph backends are not implemented; the GraphStore contract has no network adapter",
                other.as_str()
            ))),
            other => Err(self.family_mismatch(other, StoreFamily::Graph)),
        }
    }

    /// Resolves a raw spec for `family`, listing that family's names when the
    /// backend is unknown.
    fn resolve(&self, name: String, config: Value, family: StoreFamily) -> Result<StoreSpec> {
        match BackendKind::parse(&name) {
            Some(backend) => Ok(StoreSpec {
                backend,
                name,
                config,
            }),
            None => Err(Error::validation(format!(
                "unknown storage backend \"{name}\"; available {family} backends are: {}",
                self.available_names(family).join(", ")
            ))),
        }
    }

    /// The error for a backend that belongs to a different family.
    fn family_mismatch(&self, backend: BackendKind, family: StoreFamily) -> Error {
        Error::validation(format!(
            "backend \"{}\" is not a {family} backend; available {family} backends are: {}",
            backend.as_str(),
            self.available_names(family).join(", ")
        ))
    }

    /// The plugin names registered for `family`, sorted.
    fn plugin_names(&self, family: StoreFamily) -> Vec<String> {
        match family {
            StoreFamily::Kv => sorted_keys(&self.kv),
            StoreFamily::Document => sorted_keys(&self.document),
            StoreFamily::Vector => sorted_keys(&self.vector),
            StoreFamily::Object => sorted_keys(&self.object),
            StoreFamily::Graph => sorted_keys(&self.graph),
        }
    }
}

impl Default for StoreRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for StoreRegistry {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("StoreRegistry")
            .field("kv", &sorted_keys(&self.kv))
            .field("document", &sorted_keys(&self.document))
            .field("vector", &sorted_keys(&self.vector))
            .field("object", &sorted_keys(&self.object))
            .field("graph", &sorted_keys(&self.graph))
            .finish()
    }
}

/// Normalizes a plugin name the same way [`BackendKind::parse`] reads one.
fn normalize(name: &str) -> String {
    name.trim().to_ascii_lowercase()
}

/// Inserts a factory under its normalized name, returning the previous one.
fn register<F>(map: &RwLock<HashMap<String, F>>, name: impl Into<String>, factory: F) -> Option<F> {
    map.write().insert(normalize(&name.into()), factory)
}

/// Looks up a plugin factory by normalized name.
fn registered<F: Clone>(map: &RwLock<HashMap<String, F>>, name: &str) -> Option<F> {
    map.read().get(&normalize(name)).cloned()
}

/// The sorted plugin names of one map.
fn sorted_keys<F>(map: &RwLock<HashMap<String, F>>) -> Vec<String> {
    let mut names: Vec<String> = map.read().keys().cloned().collect();
    names.sort_unstable();
    names
}

/// Extracts the backend name and the remaining configuration from `spec`.
fn raw_spec(spec: &Value) -> Result<(String, Value)> {
    match spec {
        Value::String(name) => {
            let name = name.trim();
            if name.is_empty() {
                return Err(missing_backend());
            }
            Ok((name.to_owned(), Value::Object(serde_json::Map::new())))
        }
        Value::Object(object) => {
            let mut name = None;
            for key in ["backend", "type"] {
                if let Some(value) = object.get(key) {
                    let text = value
                        .as_str()
                        .map(str::trim)
                        .filter(|text| !text.is_empty());
                    name = Some(text.ok_or_else(|| invalid_backend_key(key))?);
                    break;
                }
            }
            let name = name.ok_or_else(missing_backend)?;
            let config = object
                .iter()
                .filter(|(key, _)| key.as_str() != "backend" && key.as_str() != "type")
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect();
            Ok((name.to_owned(), Value::Object(config)))
        }
        _ => Err(missing_backend()),
    }
}

/// The error for a spec without a usable backend member.
fn missing_backend() -> Error {
    Error::validation(
        "store spec must name a backend: use {\"backend\": \"memory\", ...}, \
         {\"type\": \"memory\", ...}, or a bare \"memory\" string",
    )
}

/// The error for a `backend`/`type` member that is not a non-empty string.
fn invalid_backend_key(key: &str) -> Error {
    Error::validation(format!(
        "store spec key \"{key}\" must be a non-empty string naming a backend; \
         accepted keys are \"backend\" and \"type\""
    ))
}

/// The error for an unknown backend name, listing every known name.
fn unknown_backend(name: &str) -> Error {
    let mut names: Vec<&str> = Vec::new();
    for family in [
        StoreFamily::Kv,
        StoreFamily::Document,
        StoreFamily::Vector,
        StoreFamily::Object,
        StoreFamily::Graph,
    ] {
        names.extend_from_slice(family.builtin_names());
    }
    names.sort_unstable();
    names.dedup();
    Error::validation(format!(
        "unknown storage backend \"{name}\"; available backends are: {}",
        names.join(", ")
    ))
}

/// Reads the required `root` (or `path`) member of a local-file spec.
fn config_root(config: &Value) -> Result<&str> {
    config
        .get("root")
        .or_else(|| config.get("path"))
        .and_then(Value::as_str)
        .filter(|root| !root.is_empty())
        .ok_or_else(|| {
            Error::validation(
                "the \"file\" backend requires a non-empty \"root\" (or \"path\") \
                 configuration member",
            )
        })
}

/// Reads the required `url` member of a SQLite spec.
#[cfg(feature = "sqlite")]
fn config_url(config: &Value) -> Result<&str> {
    config
        .get("url")
        .and_then(Value::as_str)
        .filter(|url| !url.is_empty())
        .ok_or_else(|| {
            Error::validation(
                "the \"sqlite\" backend requires a non-empty \"url\" configuration member",
            )
        })
}

/// Decodes a spec's configuration into `T`, naming the backend on failure.
fn decode_config<T: DeserializeOwned>(spec: &StoreSpec) -> Result<T> {
    serde_json::from_value(spec.config.clone()).map_err(|error| {
        Error::validation(format!(
            "invalid \"{}\" backend configuration: {error}",
            spec.name
        ))
    })
}

#[cfg(test)]
mod tests;
