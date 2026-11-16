//! Object-store backed graph artifact persistence.
//!
//! The third [`GraphArtifactStore`](crate::artifact::GraphArtifactStore) backend: it keeps
//! the same immutable, numbered revisions and the same `current` pointer as the in-memory
//! and filesystem backends, but stores the documents in an
//! [`ObjectStore`](openjiuwen_foundation::store::ObjectStore) bucket. That makes the graph
//! artifact shareable between processes and hosts (S3-compatible services, Google Cloud
//! Storage, Azure Blob Storage, the local filesystem object store, or any other `ObjectStore`
//! implementation) without changing a single caller.
//!
//! # Transports
//!
//! [`ArtifactTransport`](crate::object_artifact::ArtifactTransport) names the bucket bindings.
//! `S3` and `ObjectStore` are resolved from the crate's own object stores
//! ([`S3ObjectStore`](openjiuwen_foundation::store::external::S3ObjectStore),
//! [`InMemoryObjectStore`](openjiuwen_foundation::store::InMemoryObjectStore),
//! [`LocalObjectStore`](openjiuwen_foundation::store::LocalObjectStore)), `Gcs` builds the
//! Google Cloud Storage client in the `gcs` submodule, which speaks the GCS JSON API directly,
//! and `AzureBlob` builds the Azure Blob Storage client in the `azure` submodule, which speaks
//! the Blob REST API directly.
//!
//! # Credentials
//!
//! The GCS transport resolves its credential in one order, and the first form that is present
//! wins: a non-empty
//! [`GcsConfig::access_token`](crate::object_artifact::GcsConfig::access_token) is sent as a
//! bearer token; otherwise a service-account key file attached by
//! [`GcsObjectStore::from_service_account_file`](crate::object_artifact::GcsObjectStore::from_service_account_file)
//! mints an RS256 JWT assertion and exchanges it at the key file's `token_uri` for a bearer
//! token; otherwise requests are anonymous, which a local emulator and a public read-only
//! bucket accept.
//!
//! # Bucket layout
//!
//! ```text
//! <bucket>/
//!   <prefix>/versions/v1/graph.json   # {"revision": {...}, "graph": <JGF document>}
//!   <prefix>/versions/v2/graph.json
//!   <prefix>/current.json             # the same envelope, plus "graph_id" and "version"
//! ```
//!
//! The object layout mirrors the filesystem backend key for key (`versions/<version>/graph.json`
//! and `current.json`, below the configured prefix), and the documents are the same
//! `{"revision": {...}, "graph": {...}}` envelopes, so the two backends are interchangeable.
//! The prefix may be empty, in which case the documents sit at the bucket root.
//!
//! # Publication order
//!
//! A revision object is written **before** the `current` pointer, so a reader either sees the
//! previous pointer (and the previous, immutable revision it names) or the new one — never a
//! pointer to a graph that was not fully written. Publications inside one store are
//! serialized by an async mutex, which is what keeps two concurrent tasks from claiming the
//! same generated version.
//!
//! One store is the writer of its prefix, like the filesystem backend: concurrent writers
//! pointed at the same bucket and prefix would race on the generated version number, because
//! generating one lists the published revisions.
//!
//! # Errors
//!
//! A missing bucket is tolerated on the read path (`read`, `status`, `versions` answer
//! "nothing published"), because the bucket only has to exist once something is published;
//! [`publish`](crate::artifact::GraphArtifactStore::publish) creates it on first use. Every other backend failure
//! is propagated (as [`Error::External`](openjiuwen_common::error::Error)), and a malformed
//! `current.json` is an error rather than a silent "missing"; malformed *listing* entries
//! (objects whose key is not `versions/<version>/graph.json`) are skipped with a warning,
//! matching the KV-backed stores.

use std::fmt;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use openjiuwen_common::utils::id::now_rfc3339;
use openjiuwen_foundation::store::external::{S3Config, S3ObjectStore};
use openjiuwen_foundation::store::{InMemoryObjectStore, LocalObjectStore, ObjectStore};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::Mutex as AsyncMutex;
use tracing::warn;

use crate::artifact::{GraphArtifactStatus, GraphArtifactStore, GraphRevision};
use crate::error::{validation, Error, ErrorKind, Result};
use crate::graph::CapabilityGraph;

mod azure;
mod gcs;
mod http;

pub use azure::{
    AzureBlobAuth, AzureBlobConfig, AzureBlobStore, API_VERSION, AZURITE_ENDPOINT,
    CLIENT_CREDENTIALS_REASON, DEFAULT_ENDPOINT_SUFFIX,
};
pub use gcs::{GcsConfig, GcsObjectStore, DEFAULT_ENDPOINT};

/// Bucket used by [`ObjectGraphArtifactStore::in_memory`] and
/// [`ObjectGraphArtifactStore::local`], the two constructors that take no bucket name.
pub const DEFAULT_BUCKET: &str = "openjiuwen-artifacts";

/// Key suffix of the current-revision pointer document.
const CURRENT_KEY: &str = "current.json";
/// Key segment holding one object per published revision.
const VERSIONS_DIR: &str = "versions";
/// Object name of one revision's envelope below its version directory.
const GRAPH_FILE: &str = "graph.json";

/// Where an [`ObjectGraphArtifactStore`] keeps its objects.
///
/// The description a caller can serialize and pass around: [`ArtifactTransport::S3`],
/// [`ArtifactTransport::Gcs`] and [`ArtifactTransport::AzureBlob`] carry their connection
/// details, and a caller that already owns an [`ObjectStore`] hands it to
/// [`ObjectGraphArtifactStore::new`], because a live store has no serializable description.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactTransport {
    /// An [`ObjectStore`] the caller already owns.
    ///
    /// It cannot be resolved from a transport descriptor (there is nothing to reconstruct it
    /// from), so [`ObjectGraphArtifactStore::open`] rejects it and points at
    /// [`ObjectGraphArtifactStore::new`].
    ObjectStore,
    /// An S3-compatible service, built into an
    /// [`S3ObjectStore`].
    S3(S3Config),
    /// Google Cloud Storage, built into a [`GcsObjectStore`] over the GCS JSON API.
    Gcs(GcsConfig),
    /// Azure Blob Storage, built into an [`AzureBlobStore`] over the Blob REST API.
    AzureBlob(AzureBlobConfig),
}

impl ArtifactTransport {
    /// The transport name used in diagnostics and in a serialized transport descriptor.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::ObjectStore => "object_store",
            Self::S3(_) => "s3",
            Self::Gcs(_) => "gcs",
            Self::AzureBlob(_) => "azure_blob",
        }
    }
}

/// A [`GraphArtifactStore`] that persists revisions in an object-storage bucket.
///
/// Behaviour matches [`FileGraphArtifactStore`](crate::artifact::FileGraphArtifactStore)
/// exactly — immutable revisions, generated versions that skip pinned identifiers, a
/// `current` pointer that only moves once the revision object is written, and the same
/// validation on version identifiers — while the documents live in a bucket instead of a
/// directory. The layout is documented on the [module](self).
///
/// The backend reads the bucket on every call instead of caching revisions in memory, so a
/// store never serves a revision another writer published after it was constructed.
pub struct ObjectGraphArtifactStore {
    /// The bucket repository shared through the trait object.
    store: Arc<dyn ObjectStore>,
    /// The bucket every object is written to and read from.
    bucket: String,
    /// The `/`-free key prefix; empty means "at the bucket root".
    prefix: String,
    /// Serializes publications, so two tasks cannot claim the same generated version.
    writer: AsyncMutex<()>,
    /// Whether `create_bucket` already succeeded for this store.
    bucket_ready: AtomicBool,
}

impl fmt::Debug for ObjectGraphArtifactStore {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ObjectGraphArtifactStore")
            .field("bucket", &self.bucket)
            .field("prefix", &self.prefix)
            .finish_non_exhaustive()
    }
}

impl ObjectGraphArtifactStore {
    /// Builds a store over `store`, reading and writing `bucket` below `prefix`.
    ///
    /// The prefix is normalized by trimming `/` from both ends; an empty prefix means the
    /// documents sit at the bucket root. A blank bucket (or a whitespace-only prefix) is
    /// reported as a validation error by the first operation that uses it, so construction
    /// cannot fail.
    #[must_use]
    pub fn new(
        store: Arc<dyn ObjectStore>,
        bucket: impl Into<String>,
        prefix: impl Into<String>,
    ) -> Self {
        let prefix = prefix.into();
        Self {
            store,
            bucket: bucket.into().trim().to_string(),
            prefix: prefix.trim_matches('/').to_string(),
            writer: AsyncMutex::new(()),
            bucket_ready: AtomicBool::new(false),
        }
    }

    /// Builds a store over a fresh [`InMemoryObjectStore`] using [`DEFAULT_BUCKET`].
    ///
    /// Nothing survives the store, so this backend is the natural choice for tests.
    #[must_use]
    pub fn in_memory(prefix: impl Into<String>) -> Self {
        Self::new(Arc::new(InMemoryObjectStore::new()), DEFAULT_BUCKET, prefix)
    }

    /// Builds a store over a [`LocalObjectStore`] rooted at `root`, using [`DEFAULT_BUCKET`].
    ///
    /// A reopened store on the same root reports exactly the revisions that were published,
    /// because every revision carries its own metadata envelope.
    ///
    /// # Errors
    ///
    /// Returns [`Error::External`] when `root` cannot be created or used as a directory.
    pub async fn local(root: impl AsRef<Path>, prefix: impl Into<String>) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        let store = LocalObjectStore::new(&root).await.map_err(|error| {
            error.with_context(format!(
                "opening the object store root `{}`",
                root.display()
            ))
        })?;
        Ok(Self::new(Arc::new(store), DEFAULT_BUCKET, prefix))
    }

    /// Builds a store over an [`S3ObjectStore`] described by `config`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] when `config` does not name a usable HTTP(S) endpoint.
    pub fn s3(
        config: S3Config,
        bucket: impl Into<String>,
        prefix: impl Into<String>,
    ) -> Result<Self> {
        let store = S3ObjectStore::new(config)?;
        Ok(Self::new(Arc::new(store), bucket, prefix))
    }

    /// Builds a store over a [`GcsObjectStore`] described by `config`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] when `config` does not name a usable HTTP(S) endpoint.
    pub fn gcs(
        config: GcsConfig,
        bucket: impl Into<String>,
        prefix: impl Into<String>,
    ) -> Result<Self> {
        let store = GcsObjectStore::new(config)?;
        Ok(Self::new(Arc::new(store), bucket, prefix))
    }

    /// Builds a store over an [`AzureBlobStore`] described by `config`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] when the resolved endpoint is not a usable HTTP(S) URL, and
    /// [`Error::Validation`] when the configured credentials are malformed.
    pub fn azure_blob(
        config: AzureBlobConfig,
        bucket: impl Into<String>,
        prefix: impl Into<String>,
    ) -> Result<Self> {
        let store = AzureBlobStore::new(config)?;
        Ok(Self::new(Arc::new(store), bucket, prefix))
    }

    /// Resolves `transport` into a store for `bucket` below `prefix`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for [`ArtifactTransport::ObjectStore`] (there is nothing to
    /// reconstruct a caller-owned store from), and the [`S3ObjectStore`], [`GcsObjectStore`] or
    /// [`AzureBlobStore`] configuration error for a malformed endpoint or credential.
    pub fn open(
        transport: ArtifactTransport,
        bucket: impl Into<String>,
        prefix: impl Into<String>,
    ) -> Result<Self> {
        match transport {
            ArtifactTransport::ObjectStore => Err(validation(
                "transport",
                "an object-store transport carries no connection details; \
                 pass the store to `ObjectGraphArtifactStore::new` instead",
            )),
            ArtifactTransport::S3(config) => Self::s3(config, bucket, prefix),
            ArtifactTransport::Gcs(config) => Self::gcs(config, bucket, prefix),
            ArtifactTransport::AzureBlob(config) => Self::azure_blob(config, bucket, prefix),
        }
    }

    /// The bucket this store reads and writes.
    #[must_use]
    pub fn bucket(&self) -> &str {
        &self.bucket
    }

    /// The normalized key prefix; empty when the documents sit at the bucket root.
    #[must_use]
    pub fn prefix(&self) -> &str {
        &self.prefix
    }

    /// Rejects a blank bucket or a blank prefix.
    ///
    /// The check lives on the operation path (rather than in [`Self::new`]) so construction
    /// stays infallible; the error names the offending field.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] naming `bucket` or `prefix`.
    fn ensure_target(&self) -> Result<()> {
        if self.bucket.trim().is_empty() {
            return Err(validation("bucket", "must not be blank"));
        }
        if self.prefix.trim().is_empty() && !self.prefix.is_empty() {
            return Err(validation("prefix", "must not be blank"));
        }
        Ok(())
    }

    /// Joins `suffix` onto the prefix, or returns `suffix` alone for an empty prefix.
    fn object_key(&self, suffix: &str) -> String {
        if self.prefix.is_empty() {
            suffix.to_string()
        } else {
            format!("{}/{suffix}", self.prefix)
        }
    }

    /// The key of the current-revision pointer document.
    fn current_key(&self) -> String {
        self.object_key(CURRENT_KEY)
    }

    /// The key of one revision's envelope.
    fn version_key(&self, version: &str) -> String {
        self.object_key(&format!("{VERSIONS_DIR}/{version}/{GRAPH_FILE}"))
    }

    /// The listing prefix covering every revision object.
    fn versions_prefix(&self) -> String {
        self.object_key(&format!("{VERSIONS_DIR}/"))
    }

    /// Creates the bucket once, on the first publication.
    ///
    /// `create_bucket` is idempotent for every backend, so a bucket that already exists (a
    /// reopened store) is unaffected.
    ///
    /// # Errors
    ///
    /// Returns the backend's error, with the bucket named, when the bucket cannot be created.
    async fn ensure_bucket(&self) -> Result<()> {
        if self.bucket_ready.load(Ordering::SeqCst) {
            return Ok(());
        }
        self.store
            .create_bucket(&self.bucket)
            .await
            .map_err(|error| {
                error.with_context(format!("creating object bucket `{}`", self.bucket))
            })?;
        self.bucket_ready.store(true, Ordering::SeqCst);
        Ok(())
    }

    /// Reads and decodes one JSON document, or `None` when the object (or the bucket) is absent.
    ///
    /// # Errors
    ///
    /// Returns [`Error::External`] when the payload is not valid JSON or the backend fails.
    async fn get_json(&self, key: &str) -> Result<Option<Value>> {
        let payload = match self.store.get(&self.bucket, key).await {
            Ok(payload) => payload,
            // A missing bucket simply holds nothing yet.
            Err(error) if error.kind() == ErrorKind::NotFound => return Ok(None),
            Err(error) => {
                return Err(error.with_context(format!(
                    "reading object `{key}` from bucket `{}`",
                    self.bucket
                )))
            }
        };
        let Some(bytes) = payload else {
            return Ok(None);
        };
        let value = serde_json::from_slice(&bytes).map_err(|error| {
            Error::external(error).with_context(format!("decoding object `{key}`"))
        })?;
        Ok(Some(value))
    }

    /// Encodes and writes one JSON document.
    ///
    /// # Errors
    ///
    /// Returns [`Error::External`] when the document cannot be encoded or the backend fails.
    async fn put_json(&self, key: &str, value: &Value) -> Result<()> {
        let bytes = serde_json::to_vec_pretty(value).map_err(|error| {
            Error::external(error).with_context(format!("encoding object `{key}`"))
        })?;
        self.store
            .put(&self.bucket, key, Bytes::from(bytes))
            .await
            .map_err(|error| {
                error.with_context(format!(
                    "writing object `{key}` to bucket `{}`",
                    self.bucket
                ))
            })?;
        Ok(())
    }

    /// Lists the versions that have a revision object, ascending and deduplicated.
    ///
    /// Keys that are not `versions/<version>/graph.json` are skipped with a warning: a bucket
    /// may hold unrelated objects below the same prefix, and one of them must not make the
    /// revision index unreadable.
    ///
    /// # Errors
    ///
    /// Returns the backend's error when the listing fails for any reason other than a missing
    /// bucket (which is an empty listing).
    async fn list_versions(&self) -> Result<Vec<String>> {
        let prefix = self.versions_prefix();
        let objects = match self.store.list(&self.bucket, &prefix, None).await {
            Ok(objects) => objects,
            Err(error) if error.kind() == ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => {
                return Err(
                    error.with_context(format!("listing `{prefix}` in bucket `{}`", self.bucket))
                )
            }
        };
        let mut versions = Vec::new();
        for object in objects {
            if let Some(version) = Self::parse_version_key(&object.key, &prefix) {
                versions.push(version);
            } else {
                warn!(
                    key = %object.key,
                    "skipping an object that is not a capability-graph revision"
                );
            }
        }
        versions.sort();
        versions.dedup();
        Ok(versions)
    }

    /// Extracts the version from a `versions/<version>/graph.json` key.
    fn parse_version_key(key: &str, prefix: &str) -> Option<String> {
        let rest = key.strip_prefix(prefix)?;
        let (version, tail) = rest.split_once('/')?;
        if tail != GRAPH_FILE || validate_version(version).is_err() {
            return None;
        }
        Some(version.to_string())
    }

    /// The next free `v{n}`, starting at `published + 1` and skipping pinned identifiers.
    ///
    /// # Errors
    ///
    /// Returns the backend's error when the published revisions cannot be listed.
    async fn next_version(&self) -> Result<String> {
        let published = self.list_versions().await?;
        let mut ordinal = published.len() + 1;
        loop {
            let candidate = format!("v{ordinal}");
            if !published.iter().any(|version| version == &candidate) {
                return Ok(candidate);
            }
            ordinal += 1;
        }
    }

    /// Reads one revision object, or `None` when it was never published.
    ///
    /// The key path is authoritative for the revision identifier, so a document that was
    /// copied to another version keeps the version its key names.
    ///
    /// # Errors
    ///
    /// Returns [`Error::External`] when the document cannot be read or decoded.
    async fn version_document(&self, version: &str) -> Result<Option<StoredRevision>> {
        let key = self.version_key(version);
        let Some(value) = self.get_json(&key).await? else {
            return Ok(None);
        };
        let mut stored: StoredRevision = decode(&value, &key)?;
        stored.revision.version = version.to_string();
        Ok(Some(stored))
    }

    /// Reads the revision the `current` pointer names, or `None` when nothing is published.
    ///
    /// A document holding the complete envelope is served as-is; a pointer-only document is
    /// followed to its revision object. A missing pointer, an empty version, or a pointer
    /// whose revision object is gone all report "nothing published".
    ///
    /// # Errors
    ///
    /// Returns [`Error::External`] when the pointer or the revision it names is malformed.
    async fn current_document(&self) -> Result<Option<StoredRevision>> {
        let key = self.current_key();
        let Some(value) = self.get_json(&key).await? else {
            return Ok(None);
        };
        let pointer: CurrentPointer = decode(&value, &key)?;
        let mut revision = pointer.revision;
        if !pointer.version.is_empty() {
            revision.version = pointer.version;
        }
        if revision.version.is_empty() {
            warn!(
                key = %key,
                "the current capability-graph pointer has no version; treating the store as missing"
            );
            return Ok(None);
        }
        if let Some(graph) = pointer.graph {
            if !graph.is_null() {
                return Ok(Some(StoredRevision { revision, graph }));
            }
        }
        if let Some(stored) = self.version_document(&revision.version).await? {
            Ok(Some(stored))
        } else {
            warn!(
                key = %key,
                version = %revision.version,
                "the current capability-graph revision object is missing"
            );
            Ok(None)
        }
    }

    /// Resolves the revision for `version`, or the current revision when `version` is `None`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::External`] when a document cannot be read or decoded.
    async fn stored(&self, version: Option<&str>) -> Result<Option<StoredRevision>> {
        match version {
            Some(version) => self.version_document(version).await,
            None => self.current_document().await,
        }
    }
}

#[async_trait]
impl GraphArtifactStore for ObjectGraphArtifactStore {
    async fn publish(
        &self,
        graph: &CapabilityGraph,
        version: Option<&str>,
    ) -> Result<GraphRevision> {
        self.ensure_target()?;
        if let Some(version) = version {
            validate_version(version)?;
        }
        // Serialized so two tasks cannot generate the same version or interleave the
        // revision object and the current pointer.
        let _writer = self.writer.lock().await;
        let version = match version {
            Some(version) => version.to_string(),
            None => self.next_version().await?,
        };
        let key = self.version_key(&version);
        if self.get_json(&key).await?.is_some() {
            return Err(validation(
                "version",
                format!("revision `{version}` is already published; revisions are immutable"),
            ));
        }
        let stored = StoredRevision::new(graph, version);
        let document = stored.encode()?;
        let pointer = stored.pointer()?;
        self.ensure_bucket().await?;
        // The revision object first: a reader following the pointer must always find a
        // complete graph behind it.
        self.put_json(&key, &document).await?;
        self.put_json(&self.current_key(), &pointer).await?;
        Ok(stored.revision)
    }

    async fn read(&self, version: Option<&str>) -> Result<Option<CapabilityGraph>> {
        self.ensure_target()?;
        if let Some(version) = version {
            validate_version(version)?;
        }
        self.stored(version)
            .await?
            .map(|stored| stored.graph())
            .transpose()
    }

    async fn status(&self) -> Result<GraphArtifactStatus> {
        self.ensure_target()?;
        Ok(self
            .current_document()
            .await?
            .as_ref()
            .map_or_else(GraphArtifactStatus::missing, |stored| {
                GraphArtifactStatus::ready(&stored.revision)
            }))
    }

    async fn versions(&self) -> Result<Vec<GraphRevision>> {
        self.ensure_target()?;
        let mut revisions = Vec::new();
        for version in self.list_versions().await? {
            if let Some(stored) = self.version_document(&version).await? {
                revisions.push(stored.revision);
            } else {
                warn!(
                    version = %version,
                    "the capability-graph revision object disappeared while listing revisions"
                );
            }
        }
        Ok(revisions)
    }
}

/// One persisted revision: its metadata envelope plus the JGF graph document.
///
/// Identical to the envelope the filesystem backend writes, which is what makes the two
/// backends interchangeable.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
struct StoredRevision {
    /// The revision metadata.
    revision: GraphRevision,
    /// The graph document produced by [`CapabilityGraph::to_value`].
    graph: Value,
}

impl StoredRevision {
    /// Envelopes `graph` as the revision `version`, timestamped now.
    fn new(graph: &CapabilityGraph, version: String) -> Self {
        Self {
            revision: GraphRevision {
                version,
                graph_id: graph.graph_id.clone(),
                created_at: now_rfc3339(),
                node_count: graph.node_count(),
                edge_count: graph.edge_count(),
            },
            graph: graph.to_value(),
        }
    }

    /// Serializes the envelope for persistence.
    ///
    /// # Errors
    ///
    /// Returns [`Error::External`] when the envelope cannot be encoded.
    fn encode(&self) -> Result<Value> {
        serde_json::to_value(self).map_err(|error| {
            Error::external(error).with_context("encoding a capability-graph revision")
        })
    }

    /// Serializes the current-revision pointer: the envelope plus its `graph_id` and `version`.
    ///
    /// Carrying the graph in the pointer keeps it byte-compatible with the filesystem
    /// backend's `current.json`; the two redundant fields make the document legible on its
    /// own, without decoding the nested revision.
    ///
    /// # Errors
    ///
    /// Returns [`Error::External`] when the envelope cannot be encoded.
    fn pointer(&self) -> Result<Value> {
        let mut document = self.encode()?;
        if let Value::Object(map) = &mut document {
            map.insert(
                "graph_id".to_string(),
                Value::String(self.revision.graph_id.clone()),
            );
            map.insert(
                "version".to_string(),
                Value::String(self.revision.version.clone()),
            );
        }
        Ok(document)
    }

    /// Rebuilds the graph from the stored JGF document.
    ///
    /// # Errors
    ///
    /// Returns the graph decoder's error when the document is not a capability graph.
    fn graph(&self) -> Result<CapabilityGraph> {
        CapabilityGraph::from_value(&self.graph)
    }
}

/// The `current.json` document, read tolerantly.
///
/// A document this crate wrote carries both the envelope (including the graph) and the two
/// pointer fields; a pointer-only document is accepted too, and is then followed to the
/// revision object it names.
#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "snake_case", default)]
struct CurrentPointer {
    /// The revision metadata.
    revision: GraphRevision,
    /// The `graph_id` of the current revision (redundant with `revision.graph_id`).
    #[allow(dead_code)]
    graph_id: String,
    /// The version of the current revision (redundant with `revision.version`).
    version: String,
    /// The graph document, when the pointer carries the complete envelope.
    graph: Option<Value>,
}

/// Decodes `value` as `T`, naming `key` in the error context.
///
/// # Errors
///
/// Returns [`Error::External`] when the document does not match `T`.
fn decode<T: DeserializeOwned>(value: &Value, key: &str) -> Result<T> {
    serde_json::from_value(value.clone())
        .map_err(|error| Error::external(error).with_context(format!("decoding object `{key}`")))
}

/// Validates a revision identifier: `[A-Za-z0-9._-]+`, and neither `.` nor `..`.
///
/// The character set keeps a version usable as a single object-key segment, and the `.`/`..`
/// rejection keeps it from escaping the store's prefix.
///
/// # Errors
///
/// Returns [`Error::Validation`] naming the `version` field.
fn validate_version(version: &str) -> Result<()> {
    if version.trim().is_empty() {
        return Err(validation("version", "must not be blank"));
    }
    if version == "." || version == ".." {
        return Err(validation(
            "version",
            format!("`{version}` is not a usable revision name"),
        ));
    }
    let legal = |character: char| character.is_ascii_alphanumeric() || "._-".contains(character);
    if !version.chars().all(legal) {
        return Err(validation(
            "version",
            format!("`{version}` must match [A-Za-z0-9._-]+"),
        ));
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn legal_versions_pass_and_illegal_versions_are_rejected() {
        for version in ["v1", "2024-01-02", "rev_3", "a.b"] {
            assert!(validate_version(version).is_ok(), "version {version:?}");
        }
        for version in ["", "  ", ".", "..", "../evil", "bad/version", "with space"] {
            let error = validate_version(version).expect_err("must reject");
            assert_eq!(error.kind(), ErrorKind::Validation);
            assert!(error.message().contains("version"));
        }
    }

    #[test]
    fn prefixes_are_normalized_and_keys_join_cleanly() {
        let rooted = ObjectGraphArtifactStore::in_memory("/graph/");
        assert_eq!(rooted.prefix(), "graph");
        assert_eq!(rooted.current_key(), "graph/current.json");
        assert_eq!(rooted.version_key("v2"), "graph/versions/v2/graph.json");
        assert_eq!(rooted.versions_prefix(), "graph/versions/");

        let flat = ObjectGraphArtifactStore::in_memory("");
        assert_eq!(flat.prefix(), "");
        assert_eq!(flat.current_key(), "current.json");
        assert_eq!(flat.version_key("v1"), "versions/v1/graph.json");
    }

    #[test]
    fn transport_names_are_stable() {
        assert_eq!(ArtifactTransport::ObjectStore.as_str(), "object_store");
        assert_eq!(ArtifactTransport::Gcs(GcsConfig::default()).as_str(), "gcs");
        assert_eq!(
            ArtifactTransport::AzureBlob(AzureBlobConfig::default()).as_str(),
            "azure_blob"
        );
        assert_eq!(
            ArtifactTransport::S3(S3Config::new("http://localhost:9000")).as_str(),
            "s3"
        );
    }

    #[test]
    fn blank_targets_are_rejected_at_use_time() {
        let store = ObjectGraphArtifactStore::new(Arc::new(InMemoryObjectStore::new()), " ", "p");
        let error = store
            .ensure_target()
            .expect_err("blank bucket must be rejected");
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.message().contains("bucket"));

        let store = ObjectGraphArtifactStore::new(Arc::new(InMemoryObjectStore::new()), "b", "  ");
        let error = store
            .ensure_target()
            .expect_err("a blank prefix must be rejected");
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.message().contains("prefix"));
    }

    #[test]
    fn revision_keys_are_parsed_and_junk_is_skipped() {
        let store = ObjectGraphArtifactStore::in_memory("graph");
        let prefix = store.versions_prefix();
        assert_eq!(
            ObjectGraphArtifactStore::parse_version_key("graph/versions/v1/graph.json", &prefix),
            Some("v1".to_string())
        );
        for key in [
            "graph/versions/v1/other.json",
            "graph/versions/../graph.json",
            "graph/versions/v1",
            "graph/current.json",
            "other/versions/v1/graph.json",
        ] {
            assert_eq!(
                ObjectGraphArtifactStore::parse_version_key(key, &prefix),
                None,
                "key {key:?}"
            );
        }
    }
}
