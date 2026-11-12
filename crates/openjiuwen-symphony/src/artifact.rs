//! Immutable, atomically published capability-graph revisions.
//!
//! Ports Python `openjiuwen.symphony.orchestration.graph.shared.storage`: the capability
//! relationship graph is published as numbered, *immutable* revisions, and a `current`
//! pointer names the revision readers should use. Publishing never rewrites an existing
//! revision, so a reader observes either the previous document or the new one — never a
//! half-written file nor a revision that changed underneath it.
//!
//! Two backends implement [`GraphArtifactStore`]:
//!
//! - [`MemoryGraphArtifactStore`] keeps revisions in process memory. It is the default
//!   for ephemeral runs and tests.
//! - [`FileGraphArtifactStore`] keeps one JSON document per revision below
//!   `root/versions/<version>/graph.json`, plus the `root/current.json` pointer document.
//!   Both are written with the same-directory `*.tmp` write followed by `rename`, so an
//!   interrupted publish leaves the previous revision (and the previous pointer) intact.
//!
//! # On-disk layout
//!
//! ```text
//! <root>/
//!   versions/
//!     v1/graph.json      # {"revision": {...}, "graph": <JGF document>}
//!     v2/graph.json
//!   current.json         # the same envelope, for the current revision
//! ```
//!
//! Each document is a revision envelope: the [`GraphRevision`] metadata sits next to the
//! JGF document produced by [`CapabilityGraph::to_value`]. Keeping the metadata in the
//! document (instead of deriving it from file timestamps) is what makes a reopened store
//! report identical revisions and deterministic ordering.
//!
//! # Concurrency
//!
//! The revision index lives behind a [`std::sync::RwLock`]. The guard is only held for
//! lookups, map mutations, and clones — never across an `.await` — so a publishing task
//! can never block the runtime's reactor. A poisoned lock is recovered with
//! [`std::sync::PoisonError::into_inner`] rather than panicking, because every critical
//! section is a single map operation and the index itself stays coherent.
//!
//! The file backend assumes a **single writer per root** (the runtime is that writer):
//! revisions are cached in memory after construction, so `read`, `status`, and `versions`
//! never race with another process publishing to the same directory. Inside one process
//! `publish` is serialized by an async mutex, so concurrent tasks can neither claim the
//! same version nor interleave the revision document and the `current` pointer.

use std::collections::BTreeMap;
use std::io::ErrorKind as IoErrorKind;
use std::path::{Path, PathBuf};
use std::sync::{PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard};

use async_trait::async_trait;
use openjiuwen_common::utils::id::now_rfc3339;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::fs;
use tokio::sync::Mutex as AsyncMutex;
use tracing::debug;

use crate::error::{validation, Error, Result};
use crate::graph::CapabilityGraph;

/// The `status` reported when no revision has ever been published.
pub const STATUS_MISSING: &str = "missing";
/// The `status` reported when a current revision exists.
pub const STATUS_READY: &str = "ready";

/// Sub-directory holding one directory per published revision.
const VERSIONS_DIR: &str = "versions";
/// File name of one revision's envelope inside its version directory.
const GRAPH_FILE: &str = "graph.json";
/// File name of the current-revision pointer document.
const CURRENT_FILE: &str = "current.json";

/// Metadata of one published graph revision.
///
/// A revision is immutable: [`GraphArtifactStore::publish`] only ever *adds* revisions,
/// and re-publishing an existing [`Self::version`] is rejected.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct GraphRevision {
    /// The revision identifier, unique inside one artifact store (for example `v1`).
    pub version: String,
    /// The `graph_id` of the published graph.
    pub graph_id: String,
    /// RFC 3339 timestamp of the publication; metadata only, never used for ordering.
    pub created_at: String,
    /// The number of nodes in the published graph.
    pub node_count: usize,
    /// The number of edges in the published graph.
    pub edge_count: usize,
}

/// What a [`GraphArtifactStore`] currently holds.
///
/// The status is `"missing"` (see [`STATUS_MISSING`]) until the first revision is
/// published, and `"ready"` (see [`STATUS_READY`]) afterwards.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct GraphArtifactStatus {
    /// [`STATUS_MISSING`] or [`STATUS_READY`].
    pub status: String,
    /// The version of the current revision, when one exists.
    pub current_version: Option<String>,
    /// The `graph_id` of the current revision, when one exists.
    pub graph_id: Option<String>,
    /// The current revision's node count; `0` when missing.
    pub node_count: usize,
    /// The current revision's edge count; `0` when missing.
    pub edge_count: usize,
    /// When the current revision was published, when one exists.
    pub updated_at: Option<String>,
    /// Compact snapshot of the current revision: `graph_id` and `current_version`.
    pub snapshot: BTreeMap<String, Value>,
}

impl GraphArtifactStatus {
    /// Builds the "nothing published yet" status: [`STATUS_MISSING`], no metadata, and
    /// an empty [`Self::snapshot`].
    #[must_use]
    pub fn missing() -> Self {
        Self {
            status: STATUS_MISSING.to_string(),
            current_version: None,
            graph_id: None,
            node_count: 0,
            edge_count: 0,
            updated_at: None,
            snapshot: BTreeMap::new(),
        }
    }

    /// Builds the [`STATUS_READY`] status describing `revision`.
    ///
    /// The snapshot carries the two fields a caller needs to re-read the revision:
    /// `graph_id` and `current_version`.
    #[must_use]
    pub fn ready(revision: &GraphRevision) -> Self {
        let mut snapshot = BTreeMap::new();
        snapshot.insert(
            "graph_id".to_string(),
            Value::String(revision.graph_id.clone()),
        );
        snapshot.insert(
            "current_version".to_string(),
            Value::String(revision.version.clone()),
        );
        Self {
            status: STATUS_READY.to_string(),
            current_version: Some(revision.version.clone()),
            graph_id: Some(revision.graph_id.clone()),
            node_count: revision.node_count,
            edge_count: revision.edge_count,
            updated_at: Some(revision.created_at.clone()),
            snapshot,
        }
    }

    /// Reports whether a current revision exists.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.status == STATUS_READY
    }
}

impl Default for GraphArtifactStatus {
    /// Equivalent to [`GraphArtifactStatus::missing`].
    fn default() -> Self {
        Self::missing()
    }
}

/// Publication and lookup of immutable capability-graph revisions.
///
/// Implementations are shared (`Arc<dyn GraphArtifactStore>`) by the runtime, so `read`,
/// `status`, and `versions` take `&self` and never require exclusive access.
#[async_trait]
pub trait GraphArtifactStore: Send + Sync {
    /// Publishes `graph` as a new revision and returns its metadata.
    ///
    /// `version` pins the revision identifier; `None` asks the store to generate the next
    /// sequential identifier (`v{n}`, starting at `v1`, skipping identifiers that are
    /// already published).
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when `version` is blank, contains characters
    /// outside `[A-Za-z0-9._-]`, is `.` or `..`, or is already published (revisions are
    /// immutable). Returns [`Error::External`] when the backend cannot persist the
    /// revision and [`Error::Config`] when the backend root is unusable. A failed
    /// publication never changes what [`Self::read`] and [`Self::status`] report.
    async fn publish(
        &self,
        graph: &CapabilityGraph,
        version: Option<&str>,
    ) -> Result<GraphRevision>;

    /// Reads a published revision; `None` means "the current revision".
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for a version identifier that could not have been
    /// published (see [`Self::publish`]) and [`Error::External`] when a persisted document
    /// cannot be decoded.
    async fn read(&self, version: Option<&str>) -> Result<Option<CapabilityGraph>>;

    /// Describes the current revision, or reports [`STATUS_MISSING`] when none exists.
    ///
    /// # Errors
    ///
    /// Returns the backend's error when the status cannot be established.
    async fn status(&self) -> Result<GraphArtifactStatus>;

    /// Lists every published revision in ascending version order.
    ///
    /// # Errors
    ///
    /// Returns the backend's error when the revision index cannot be read.
    async fn versions(&self) -> Result<Vec<GraphRevision>>;
}

/// One persisted revision: its metadata envelope plus the JGF graph document.
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

    /// Rebuilds the graph from the stored JGF document.
    ///
    /// # Errors
    ///
    /// Returns [`Error::External`] when the document is not a valid capability graph.
    fn graph(&self) -> Result<CapabilityGraph> {
        CapabilityGraph::from_value(&self.graph)
    }
}

/// The revision index shared by both backends.
#[derive(Debug, Default)]
struct RevisionState {
    /// Revisions keyed by version, iterated in ascending version order.
    revisions: BTreeMap<String, StoredRevision>,
    /// The version readers get for `read(None)`: the most recently published one.
    current: Option<String>,
}

impl RevisionState {
    /// Resolves the version to publish: validates a pinned one, generates a free one
    /// otherwise.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the pinned version is illegal.
    fn resolve_version(&self, requested: Option<&str>) -> Result<String> {
        match requested {
            Some(version) => {
                validate_version(version)?;
                Ok(version.to_string())
            }
            None => Ok(self.next_version()),
        }
    }

    /// The next free `v{n}`, starting at `revisions.len() + 1`.
    fn next_version(&self) -> String {
        let mut ordinal = self.revisions.len() + 1;
        loop {
            let candidate = format!("v{ordinal}");
            if !self.revisions.contains_key(&candidate) {
                return candidate;
            }
            ordinal += 1;
        }
    }

    /// Rejects a version that is already published.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when `version` is already published.
    fn ensure_absent(&self, version: &str) -> Result<()> {
        if self.revisions.contains_key(version) {
            return Err(validation(
                "version",
                format!("revision `{version}` is already published; revisions are immutable"),
            ));
        }
        Ok(())
    }

    /// Envelopes `graph` as a fresh revision that does not collide with an existing one.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for an illegal or already-published version.
    fn prepare(&self, graph: &CapabilityGraph, version: Option<&str>) -> Result<StoredRevision> {
        let version = self.resolve_version(version)?;
        self.ensure_absent(&version)?;
        Ok(StoredRevision::new(graph, version))
    }

    /// Records `stored` and makes it the current revision.
    fn insert(&mut self, stored: StoredRevision) {
        self.current = Some(stored.revision.version.clone());
        self.revisions
            .insert(stored.revision.version.clone(), stored);
    }

    /// The revision for `version`, or the current revision when `version` is `None`.
    fn stored(&self, version: Option<&str>) -> Option<StoredRevision> {
        let key = match version {
            Some(version) => version,
            None => self.current.as_deref()?,
        };
        self.revisions.get(key).cloned()
    }

    /// Describes the current revision, or the missing status when there is none.
    fn status(&self) -> GraphArtifactStatus {
        self.stored(None)
            .as_ref()
            .map_or_else(GraphArtifactStatus::missing, |stored| {
                GraphArtifactStatus::ready(&stored.revision)
            })
    }

    /// Every revision in ascending version order.
    fn versions(&self) -> Vec<GraphRevision> {
        self.revisions
            .values()
            .map(|stored| stored.revision.clone())
            .collect()
    }
}

/// A [`GraphArtifactStore`] that keeps revisions in process memory.
///
/// Nothing survives the process, so this backend is the natural choice for tests and for
/// runs that rebuild the graph on every start. Behaviour is otherwise identical to
/// [`FileGraphArtifactStore`], including immutability and version validation.
#[derive(Debug, Default)]
pub struct MemoryGraphArtifactStore {
    /// The in-memory revision index.
    state: RwLock<RevisionState>,
}

impl MemoryGraphArtifactStore {
    /// Builds an empty store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Locks the index for reading, recovering from a poisoned lock.
    fn read_state(&self) -> RwLockReadGuard<'_, RevisionState> {
        self.state.read().unwrap_or_else(PoisonError::into_inner)
    }

    /// Locks the index for writing, recovering from a poisoned lock.
    fn write_state(&self) -> RwLockWriteGuard<'_, RevisionState> {
        self.state.write().unwrap_or_else(PoisonError::into_inner)
    }
}

#[async_trait]
impl GraphArtifactStore for MemoryGraphArtifactStore {
    async fn publish(
        &self,
        graph: &CapabilityGraph,
        version: Option<&str>,
    ) -> Result<GraphRevision> {
        // The write guard covers the duplicate check and the insert, so two concurrent
        // publishers cannot both claim the same version.
        let mut state = self.write_state();
        let stored = state.prepare(graph, version)?;
        let revision = stored.revision.clone();
        state.insert(stored);
        Ok(revision)
    }

    async fn read(&self, version: Option<&str>) -> Result<Option<CapabilityGraph>> {
        if let Some(version) = version {
            validate_version(version)?;
        }
        let stored = self.read_state().stored(version);
        stored.map(|stored| stored.graph()).transpose()
    }

    async fn status(&self) -> Result<GraphArtifactStatus> {
        Ok(self.read_state().status())
    }

    async fn versions(&self) -> Result<Vec<GraphRevision>> {
        Ok(self.read_state().versions())
    }
}

/// A [`GraphArtifactStore`] that persists revisions below a root directory.
///
/// The layout is documented on the [module](self). Construction creates the root (if
/// needed) and hydrates every published revision plus the `current` pointer, so a store
/// reopened on an existing root reports exactly the revisions that were published.
///
/// A revision document is written before the `current` pointer, so a failed publication
/// may leave an unreferenced version document behind but never advertises a revision that
/// was not fully written; a retry of the same version overwrites that orphan.
#[derive(Debug)]
pub struct FileGraphArtifactStore {
    /// The directory holding `versions/` and `current.json`.
    root: PathBuf,
    /// The in-memory revision index, hydrated on construction.
    state: RwLock<RevisionState>,
    /// Serializes publications, including the disk writes between them.
    writer: AsyncMutex<()>,
}

impl FileGraphArtifactStore {
    /// Opens the store at `root`, creating it when absent and loading existing revisions.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] when `root` cannot be used as a directory and
    /// [`Error::External`] when a persisted revision cannot be read or decoded.
    pub async fn new(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        fs::create_dir_all(&root).await.map_err(|error| {
            Error::config(format!(
                "graph artifact root `{}` is not a usable directory: {error}",
                root.display()
            ))
        })?;
        let state = load_state(&root).await?;
        Ok(Self {
            root,
            state: RwLock::new(state),
            writer: AsyncMutex::new(()),
        })
    }

    /// The root directory this store publishes below.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Locks the index for reading, recovering from a poisoned lock.
    fn read_state(&self) -> RwLockReadGuard<'_, RevisionState> {
        self.state.read().unwrap_or_else(PoisonError::into_inner)
    }

    /// Locks the index for writing, recovering from a poisoned lock.
    fn write_state(&self) -> RwLockWriteGuard<'_, RevisionState> {
        self.state.write().unwrap_or_else(PoisonError::into_inner)
    }
}

#[async_trait]
impl GraphArtifactStore for FileGraphArtifactStore {
    async fn publish(
        &self,
        graph: &CapabilityGraph,
        version: Option<&str>,
    ) -> Result<GraphRevision> {
        // Publications are serialized so two concurrent tasks cannot claim the same
        // version or interleave the revision document and the current pointer.
        let _writer = self.writer.lock().await;
        let stored = self.read_state().prepare(graph, version)?;
        let document = stored.encode()?;
        let version_dir = self.root.join(VERSIONS_DIR).join(&stored.revision.version);
        fs::create_dir_all(&version_dir).await.map_err(|error| {
            path_error("creating the graph revision directory", &version_dir, error)
        })?;
        write_json_atomic(&version_dir.join(GRAPH_FILE), &document).await?;
        write_json_atomic(&self.root.join(CURRENT_FILE), &document).await?;
        let revision = stored.revision.clone();
        self.write_state().insert(stored);
        Ok(revision)
    }

    async fn read(&self, version: Option<&str>) -> Result<Option<CapabilityGraph>> {
        if let Some(version) = version {
            validate_version(version)?;
        }
        let stored = self.read_state().stored(version);
        stored.map(|stored| stored.graph()).transpose()
    }

    async fn status(&self) -> Result<GraphArtifactStatus> {
        Ok(self.read_state().status())
    }

    async fn versions(&self) -> Result<Vec<GraphRevision>> {
        Ok(self.read_state().versions())
    }
}

/// Validates a revision identifier: `[A-Za-z0-9._-]+`, and neither `.` nor `..`.
///
/// The character set is what keeps a version usable as a single directory name, and the
/// `.`/`..` rejection is what keeps it from escaping the store's root.
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

/// Loads every revision plus the current pointer below `root`.
async fn load_state(root: &Path) -> Result<RevisionState> {
    let mut state = RevisionState::default();
    let versions_dir = root.join(VERSIONS_DIR);
    match fs::read_dir(&versions_dir).await {
        Ok(mut entries) => {
            while let Some(entry) = entries.next_entry().await.map_err(|error| {
                path_error("reading the graph revision directory", &versions_dir, error)
            })? {
                let name = entry.file_name();
                let Some(version) = name.to_str() else {
                    debug!("skipping a graph revision directory with a non-UTF-8 name");
                    continue;
                };
                if validate_version(version).is_err() {
                    debug!(version, "skipping a directory that is not a graph revision");
                    continue;
                }
                let path = entry.path().join(GRAPH_FILE);
                if let Some(stored) = load_revision(&path, Some(version)).await? {
                    state.revisions.insert(version.to_string(), stored);
                } else {
                    debug!(
                        version,
                        "graph revision document is missing; skipping the revision"
                    );
                }
            }
        }
        Err(error) if error.kind() == IoErrorKind::NotFound => {}
        Err(error) => {
            return Err(path_error(
                "reading the graph revision directory",
                &versions_dir,
                error,
            ));
        }
    }

    let current_path = root.join(CURRENT_FILE);
    match load_revision(&current_path, None).await? {
        Some(pointer) => {
            let version = pointer.revision.version.clone();
            if version.is_empty() {
                debug!(
                    path = %current_path.display(),
                    "the current graph revision pointer has no version; ignoring it"
                );
            } else {
                // The pointer is authoritative even when its version document is absent:
                // it carries the complete graph, so the revision stays readable.
                state.revisions.entry(version.clone()).or_insert(pointer);
                state.current = Some(version);
            }
        }
        None => {
            // No pointer (an interrupted first publish): fall back to the highest
            // published version so `status` still describes what is on disk.
            state.current = state.revisions.keys().next_back().cloned();
        }
    }
    Ok(state)
}

/// Reads one revision envelope, or `None` when the document does not exist.
///
/// When `version` is given it overrides the version recorded inside the document, so the
/// directory name is the single source of truth for a revision's identity.
async fn load_revision(path: &Path, version: Option<&str>) -> Result<Option<StoredRevision>> {
    let Some(value) = read_json(path).await? else {
        return Ok(None);
    };
    let mut stored: StoredRevision = serde_json::from_value(value).map_err(|error| {
        Error::external(error).with_context(format!("decoding `{}`", path.display()))
    })?;
    if let Some(version) = version {
        stored.revision.version = version.to_string();
    }
    Ok(Some(stored))
}

/// Reads and parses a JSON document, or `None` when it does not exist.
async fn read_json(path: &Path) -> Result<Option<Value>> {
    match fs::read(path).await {
        Ok(bytes) => serde_json::from_slice(&bytes).map(Some).map_err(|error| {
            Error::external(error).with_context(format!("decoding `{}`", path.display()))
        }),
        Err(error) if error.kind() == IoErrorKind::NotFound => Ok(None),
        Err(error) => Err(path_error("reading", path, error)),
    }
}

/// Writes `value` to `path` atomically: `<path>.tmp` in the same directory, then `rename`.
///
/// A reader therefore sees either the previous document or the complete new one. A failed
/// rename removes the temporary file so a retry starts from a clean directory.
async fn write_json_atomic(path: &Path, value: &Value) -> Result<()> {
    let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
        return Err(Error::config(format!(
            "graph artifact path `{}` has no file name",
            path.display()
        )));
    };
    let temporary = path.with_file_name(format!("{file_name}.tmp"));
    let bytes = serde_json::to_vec_pretty(value).map_err(|error| {
        Error::external(error).with_context(format!("encoding `{}`", path.display()))
    })?;
    fs::write(&temporary, &bytes)
        .await
        .map_err(|error| path_error("writing", &temporary, error))?;
    if let Err(error) = fs::rename(&temporary, path).await {
        if let Err(cleanup) = fs::remove_file(&temporary).await {
            debug!(
                path = %temporary.display(),
                error = %cleanup,
                "failed to remove a temporary graph artifact"
            );
        }
        return Err(path_error("committing", path, error));
    }
    Ok(())
}

/// Wraps an I/O failure as [`Error::External`], naming the action and the path.
fn path_error(action: &str, path: &Path, error: std::io::Error) -> Error {
    Error::external(error).with_context(format!("{action} `{}`", path.display()))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn status_default_is_missing() {
        let status = GraphArtifactStatus::default();
        assert_eq!(status, GraphArtifactStatus::missing());
        assert!(!status.is_ready());
        assert!(status.snapshot.is_empty());
    }

    #[test]
    fn legal_versions_pass_and_illegal_versions_are_rejected() {
        for version in ["v1", "2024-01-02", "rev_3", "a.b"] {
            assert!(validate_version(version).is_ok(), "version {version:?}");
        }
        for version in ["", "  ", ".", "..", "../evil", "bad/version", "with space"] {
            let error = validate_version(version).expect_err("must reject");
            assert_eq!(error.kind(), crate::error::ErrorKind::Validation);
            assert!(error.message().contains("version"));
        }
    }

    #[test]
    fn generated_versions_start_at_v1_and_skip_published_ones() {
        let mut state = RevisionState::default();
        assert_eq!(state.next_version(), "v1");
        let graph = CapabilityGraph::new("g");
        state.insert(StoredRevision::new(&graph, "v2".to_string()));
        assert_eq!(
            state.next_version(),
            "v3",
            "v2 is already published, so the first free ordinal wins"
        );
    }

    #[test]
    fn status_is_ready_only_after_a_publication() {
        let mut state = RevisionState::default();
        assert_eq!(state.status(), GraphArtifactStatus::missing());
        let graph = CapabilityGraph::new("g");
        state.insert(StoredRevision::new(&graph, "v1".to_string()));
        let status = state.status();
        assert!(status.is_ready());
        assert_eq!(status.current_version.as_deref(), Some("v1"));
        assert_eq!(status.graph_id.as_deref(), Some("g"));
        assert_eq!(status.snapshot.len(), 2);
    }
}
