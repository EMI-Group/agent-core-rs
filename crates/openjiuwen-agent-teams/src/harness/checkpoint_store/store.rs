//! Durable per-member checkpoint destinations.
//!
//! A [`MemberCheckpointRecord`] pairs the portable
//! [`NativeCheckpoint`](crate::harness::checkpoint::NativeCheckpoint) payload with the optional
//! provider envelope, and a [`MemberCheckpointStore`] persists the latest record per
//! `(session_id, card_id)` identity. Two backends ship: the deterministic in-memory map used by
//! tests and short-lived teams, and a file backend whose layout is one JSON document per identity.
//!
//! Neither backend is a compare-and-set store: a save overwrites the previous record for the same
//! identity and returns a fresh, unique revision string, so the harness-protocol's
//! `expected_storage_revision` is advisory only.
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use async_trait::async_trait;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use tokio::fs;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_harness_protocol::HarnessCheckpoint;

use crate::harness::checkpoint::NativeCheckpoint;

/// File name of the single checkpoint document inside one identity directory.
const CHECKPOINT_FILENAME: &str = "checkpoint.json";

/// Revision prefix of [`InMemoryMemberCheckpointStore`].
const MEMORY_REVISION_PREFIX: &str = "memory";

/// Revision prefix of [`FileMemberCheckpointStore`].
const FILE_REVISION_PREFIX: &str = "file";

/// Identity a checkpoint is stored under: `(session_id, card_id)`.
type CheckpointKey = (String, String);

/// One persisted member checkpoint: the portable payload plus the provider envelope.
///
/// The record carries both because a native harness resumes from the provider envelope
/// ([`protocol`](Self::protocol), an opaque [`HarnessCheckpoint`]) while a team-level restore or an
/// operator inspection reads the portable payload ([`checkpoint`](Self::checkpoint)). `protocol`
/// defaults to `None`, so a record written before envelopes were captured still decodes.
///
/// [`unrecognised_state`](Self::unrecognised_state) reports a degraded projection rather than a
/// failure: the sink always writes a decodable [`HarnessState`](openjiuwen_harness_protocol::HarnessState)
/// into the payload, and names the raw value it could not decode here so a caller can tell a
/// fresh idle member from one restored out of a corrupt or newer-format record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct MemberCheckpointRecord {
    /// Portable member snapshot (session/card identity, state, output).
    pub checkpoint: NativeCheckpoint,
    /// Provider envelope the native harness resumes from, when one was captured.
    #[serde(default)]
    pub protocol: Option<HarnessCheckpoint>,
    /// Raw `state` value of the envelope when it was not a harness state this build knows.
    ///
    /// `None` on every healthy record, including one whose envelope carried no state at all; a
    /// JSON string is kept verbatim, any other JSON value as its compact text. The payload's
    /// [`state`](NativeCheckpoint::state) is [`Idle`](openjiuwen_harness_protocol::HarnessState::Idle)
    /// on that path, so restoring stays tolerant.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unrecognised_state: Option<String>,
}

impl MemberCheckpointRecord {
    /// Wraps a payload without a provider envelope.
    #[must_use]
    pub fn new(checkpoint: NativeCheckpoint) -> Self {
        Self {
            checkpoint,
            protocol: None,
            unrecognised_state: None,
        }
    }

    /// Attaches the provider envelope captured alongside the payload.
    #[must_use]
    pub fn with_protocol(mut self, protocol: HarnessCheckpoint) -> Self {
        self.protocol = Some(protocol);
        self
    }

    /// Records that the envelope's state was unreadable and its raw value, kept verbatim.
    #[must_use]
    pub fn with_unrecognised_state(mut self, raw: impl Into<String>) -> Self {
        self.unrecognised_state = Some(raw.into());
        self
    }

    /// Whether the payload's state is a tolerant projection rather than the stored value.
    #[must_use]
    pub fn has_unrecognised_state(&self) -> bool {
        self.unrecognised_state.is_some()
    }
}

/// Durable per-member checkpoint destination.
///
/// Implementations are object-safe, so the runtime holds an `Arc<dyn MemberCheckpointStore>` and a
/// [`MemberCheckpointSink`](crate::harness::checkpoint_store::MemberCheckpointSink) adapts it to
/// the harness-protocol sink SPI. Storage is last-write-wins per `(session_id, card_id)`.
#[async_trait]
pub trait MemberCheckpointStore: Send + Sync {
    /// Backend name: `"memory"` or `"file"`.
    fn name(&self) -> &'static str;

    /// Persists `record` and returns the storage revision assigned to this write.
    ///
    /// Revisions are non-empty and unique per write; they are not ordered across identities.
    ///
    /// # Errors
    ///
    /// Returns a storage failure (undecodable value, filesystem error) as a
    /// [`openjiuwen_common::error::Error`].
    async fn save(&self, record: MemberCheckpointRecord) -> Result<String>;

    /// Returns the latest record stored for `(session_id, card_id)`, or `None`.
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::ErrorKind::Validation`] when a stored document cannot be
    /// decoded, or a filesystem error for any failure other than a missing document.
    async fn load(&self, session_id: &str, card_id: &str)
        -> Result<Option<MemberCheckpointRecord>>;
}

/// Monotonic per-store revision counter.
///
/// One instance per store; each call returns `<prefix>-<n>` with a strictly increasing `n`, so a
/// revision string is never empty and never repeats within a store.
struct RevisionCounter {
    prefix: &'static str,
    next: AtomicU64,
}

impl RevisionCounter {
    /// Starts at revision 1.
    const fn new(prefix: &'static str) -> Self {
        Self {
            prefix,
            next: AtomicU64::new(0),
        }
    }

    /// Hands out the next revision string.
    fn next_revision(&self) -> String {
        let value = self.next.fetch_add(1, Ordering::Relaxed) + 1;
        format!("{}-{value}", self.prefix)
    }
}

/// Bookkeeping of the in-memory backend: latest record per identity plus the revision history.
#[derive(Default)]
struct MemoryState {
    entries: HashMap<CheckpointKey, MemberCheckpointRecord>,
    revisions: Vec<String>,
}

/// In-memory checkpoint store: deterministic, process-local and lost on shutdown.
///
/// This is the default for tests and for a team that runs one session; the revision history is
/// kept so a test can assert write ordering.
pub struct InMemoryMemberCheckpointStore {
    state: Mutex<MemoryState>,
    revisions: RevisionCounter,
}

impl InMemoryMemberCheckpointStore {
    /// Creates an empty store.
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: Mutex::new(MemoryState::default()),
            revisions: RevisionCounter::new(MEMORY_REVISION_PREFIX),
        }
    }

    /// An `Arc<dyn MemberCheckpointStore>` over a fresh store, for builder injection.
    #[must_use]
    pub fn shared() -> std::sync::Arc<dyn MemberCheckpointStore> {
        std::sync::Arc::new(Self::new())
    }

    /// Number of identities currently stored.
    #[must_use]
    pub fn len(&self) -> usize {
        self.state.lock().entries.len()
    }

    /// Whether no identity is stored yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.state.lock().entries.is_empty()
    }

    /// Test/read accessor for the record stored under this identity.
    #[must_use]
    pub fn get(&self, session_id: &str, card_id: &str) -> Option<MemberCheckpointRecord> {
        let key = (session_id.to_owned(), card_id.to_owned());
        self.state.lock().entries.get(&key).cloned()
    }

    /// All revisions handed out so far, in save order.
    #[must_use]
    pub fn revisions(&self) -> Vec<String> {
        self.state.lock().revisions.clone()
    }
}

impl Default for InMemoryMemberCheckpointStore {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl MemberCheckpointStore for InMemoryMemberCheckpointStore {
    fn name(&self) -> &'static str {
        MEMORY_REVISION_PREFIX
    }

    async fn save(&self, record: MemberCheckpointRecord) -> Result<String> {
        let revision = self.revisions.next_revision();
        let key = (
            record.checkpoint.session_id.clone(),
            record.checkpoint.card_id.clone(),
        );
        let mut state = self.state.lock();
        state.revisions.push(revision.clone());
        state.entries.insert(key, record);
        Ok(revision)
    }

    async fn load(
        &self,
        session_id: &str,
        card_id: &str,
    ) -> Result<Option<MemberCheckpointRecord>> {
        let key = (session_id.to_owned(), card_id.to_owned());
        Ok(self.state.lock().entries.get(&key).cloned())
    }
}

/// Filesystem checkpoint store: `<root>/<session>/<card>/checkpoint.json`.
///
/// Each path component is sanitised (see [`FileMemberCheckpointStore::path_for`]), so an identity
/// can never escape `root`. Writes go to a temporary sibling and are renamed into place, so a
/// reader never observes a half-written document.
pub struct FileMemberCheckpointStore {
    root: PathBuf,
    revisions: RevisionCounter,
}

impl FileMemberCheckpointStore {
    /// Creates a store rooted at `root`; no directory is created until the first save.
    #[must_use]
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            revisions: RevisionCounter::new(FILE_REVISION_PREFIX),
        }
    }

    /// Root directory this store writes under.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Returns `<root>/<sanitised session>/<sanitised card>/checkpoint.json`.
    ///
    /// Sanitisation keeps ASCII alphanumerics, `-` and `_` and replaces every other byte with `_`,
    /// and a component that sanitises to nothing becomes `_`. The rule is fixed rather than
    /// platform-dependent: it makes `..` a plain name (`__`) and removes both separators, so the
    /// returned path always stays under [`root`](Self::root).
    #[must_use]
    pub fn path_for(&self, session_id: &str, card_id: &str) -> PathBuf {
        self.root
            .join(sanitise_component(session_id))
            .join(sanitise_component(card_id))
            .join(CHECKPOINT_FILENAME)
    }
}

#[async_trait]
impl MemberCheckpointStore for FileMemberCheckpointStore {
    fn name(&self) -> &'static str {
        FILE_REVISION_PREFIX
    }

    async fn save(&self, record: MemberCheckpointRecord) -> Result<String> {
        let path = self.path_for(&record.checkpoint.session_id, &record.checkpoint.card_id);
        let revision = self.revisions.next_revision();
        let directory = path
            .parent()
            .map_or_else(|| self.root.clone(), Path::to_path_buf);
        fs::create_dir_all(&directory)
            .await
            .map_err(|error| io_error(&directory, &error))?;
        let bytes = serde_json::to_vec(&record).map_err(|error| {
            Error::validation(format!(
                "cannot encode member checkpoint for {}: {error}",
                path.display()
            ))
        })?;
        let temporary = directory.join(format!("{CHECKPOINT_FILENAME}.tmp-{revision}"));
        fs::write(&temporary, &bytes)
            .await
            .map_err(|error| io_error(&temporary, &error))?;
        if let Err(error) = fs::rename(&temporary, &path).await {
            let _ = fs::remove_file(&temporary).await;
            return Err(io_error(&path, &error));
        }
        Ok(revision)
    }

    async fn load(
        &self,
        session_id: &str,
        card_id: &str,
    ) -> Result<Option<MemberCheckpointRecord>> {
        let path = self.path_for(session_id, card_id);
        let bytes = match fs::read(&path).await {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(io_error(&path, &error)),
        };
        serde_json::from_slice(&bytes).map(Some).map_err(|error| {
            Error::validation(format!(
                "invalid member checkpoint at {}: {error}",
                path.display()
            ))
        })
    }
}

/// Sanitises one path component: ASCII alphanumerics, `-` and `_` survive, every other byte
/// becomes `_`, and an empty result becomes `_`.
fn sanitise_component(component: &str) -> String {
    let mut sanitised = String::with_capacity(component.len());
    for byte in component.bytes() {
        if byte.is_ascii_alphanumeric() || byte == b'-' || byte == b'_' {
            sanitised.push(char::from(byte));
        } else {
            sanitised.push('_');
        }
    }
    if sanitised.is_empty() {
        sanitised.push('_');
    }
    sanitised
}

/// Filesystem failures other than a missing document: validation with the offending path.
fn io_error(path: &Path, error: &std::io::Error) -> Error {
    Error::validation(format!(
        "member checkpoint i/o error at {}: {error}",
        path.display()
    ))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::path::{Path, PathBuf};

    use super::{
        FileMemberCheckpointStore, InMemoryMemberCheckpointStore, MemberCheckpointRecord,
        MemberCheckpointStore,
    };
    use crate::harness::checkpoint::NativeCheckpoint;
    use openjiuwen_common::error::ErrorKind;
    use openjiuwen_harness_protocol::{empty_json_object, HarnessCheckpoint, HarnessState};
    use serde_json::json;

    /// A record carrying a provider envelope, so equality covers both halves.
    fn record(session_id: &str, card_id: &str) -> MemberCheckpointRecord {
        let payload = NativeCheckpoint::new(session_id, card_id)
            .with_cwd("/work/dev-1")
            .with_state(HarnessState::Paused)
            .with_turn_id("turn-3")
            .with_outputs(json!({"answer_parts": ["hi"]}));
        MemberCheckpointRecord::new(payload).with_protocol(HarnessCheckpoint {
            provider: "native".to_owned(),
            schema_version: "1".to_owned(),
            agent_id: "dev-1".to_owned(),
            host_session_id: "team:dev-1".to_owned(),
            checkpoint_id: "cp-3".to_owned(),
            sequence: 3,
            data: empty_json_object(),
            provider_session_id: None,
            revision: None,
        })
    }

    #[tokio::test]
    async fn memory_round_trip_keeps_the_record() {
        let store = InMemoryMemberCheckpointStore::new();
        assert!(store.is_empty());
        assert_eq!(store.name(), "memory");

        let saved = record("team:dev-1", "card-1");
        let revision = store.save(saved.clone()).await.unwrap();
        assert_eq!(revision, "memory-1");
        assert_eq!(
            store.load("team:dev-1", "card-1").await.unwrap(),
            Some(saved)
        );
        assert_eq!(store.len(), 1);
        assert!(!store.is_empty());
        assert_eq!(
            store.get("team:dev-1", "card-1"),
            Some(record("team:dev-1", "card-1"))
        );
        assert_eq!(store.get("team:dev-2", "card-1"), None);
    }

    #[tokio::test]
    async fn memory_newest_write_wins_and_identities_are_independent() {
        let store = InMemoryMemberCheckpointStore::new();
        let first = record("team:dev-1", "card-1");
        let second = MemberCheckpointRecord::new(
            NativeCheckpoint::new("team:dev-1", "card-1").with_state(HarnessState::Idle),
        );
        let other = record("team:dev-2", "card-1");

        let revisions = vec![
            store.save(first).await.unwrap(),
            store.save(second.clone()).await.unwrap(),
            store.save(other.clone()).await.unwrap(),
        ];
        assert_eq!(revisions, ["memory-1", "memory-2", "memory-3"]);
        assert_eq!(store.revisions(), revisions);
        assert!(revisions.iter().all(|revision| !revision.is_empty()));

        assert_eq!(
            store.load("team:dev-1", "card-1").await.unwrap(),
            Some(second)
        );
        assert_eq!(
            store.load("team:dev-2", "card-1").await.unwrap(),
            Some(other)
        );
        assert_eq!(store.len(), 2);
    }

    #[tokio::test]
    async fn memory_unknown_identity_is_none() {
        let store = InMemoryMemberCheckpointStore::new();
        assert_eq!(store.load("missing", "missing").await.unwrap(), None);
        assert_eq!(store.revisions(), Vec::<String>::new());
    }

    #[tokio::test]
    async fn shared_returns_an_object_safe_store() {
        let store = InMemoryMemberCheckpointStore::shared();
        assert_eq!(store.name(), "memory");
        let revision = store
            .save(MemberCheckpointRecord::new(NativeCheckpoint::new("s", "c")))
            .await
            .unwrap();
        assert_eq!(revision, "memory-1");
        assert!(store.load("s", "c").await.unwrap().is_some());
    }

    #[test]
    fn sanitisation_keeps_names_under_the_root() {
        let root = PathBuf::from("/tmp/checkpoints");
        let store = FileMemberCheckpointStore::new(root.clone());
        assert_eq!(store.root(), Path::new("/tmp/checkpoints"));

        let path = store.path_for("team:dev-1", "card.1");
        assert!(path.starts_with(&root));
        assert_eq!(
            path,
            root.join("team_dev-1")
                .join("card_1")
                .join("checkpoint.json")
        );
        let escaped = store.path_for("a/../../b", "c");
        assert!(escaped.starts_with(&root));
        assert_eq!(
            escaped,
            root.join("a_______b").join("c").join("checkpoint.json")
        );
        assert_eq!(sanitiser_probe(""), "_");
        assert_eq!(sanitiser_probe("é"), "__");
    }

    #[tokio::test]
    async fn file_round_trip_keeps_the_record() {
        let directory = tempfile::tempdir().unwrap();
        let store = FileMemberCheckpointStore::new(directory.path());
        let saved = record("team:dev-1", "card-1");

        let revision = store.save(saved.clone()).await.unwrap();
        assert_eq!(revision, "file-1");
        assert!(store.path_for("team:dev-1", "card-1").is_file());
        assert_eq!(
            store.load("team:dev-1", "card-1").await.unwrap(),
            Some(saved)
        );

        let other = record("team:dev-2", "card-1");
        assert_eq!(store.save(other.clone()).await.unwrap(), "file-2");
        assert_eq!(
            store.load("team:dev-2", "card-1").await.unwrap(),
            Some(other.clone())
        );
        assert_eq!(store.save(other).await.unwrap(), "file-3");
        assert_eq!(store.name(), "file");
    }

    #[tokio::test]
    async fn file_missing_document_is_none() {
        let directory = tempfile::tempdir().unwrap();
        let store = FileMemberCheckpointStore::new(directory.path());
        assert_eq!(store.load("absent", "absent").await.unwrap(), None);
    }

    #[tokio::test]
    async fn file_malformed_document_names_the_path() {
        let directory = tempfile::tempdir().unwrap();
        let store = FileMemberCheckpointStore::new(directory.path());
        let path = store.path_for("team:dev-1", "card-1");
        tokio::fs::create_dir_all(path.parent().unwrap())
            .await
            .unwrap();
        tokio::fs::write(&path, b"{not json").await.unwrap();

        let error = store.load("team:dev-1", "card-1").await.unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(
            error.to_string().contains(&path.display().to_string()),
            "message must name the file: {error}"
        );
    }

    #[tokio::test]
    async fn record_without_envelope_round_trips() {
        let directory = tempfile::tempdir().unwrap();
        let store = FileMemberCheckpointStore::new(directory.path());
        let saved = MemberCheckpointRecord::new(NativeCheckpoint::new("team:dev-1", "card-1"));
        assert_eq!(saved.protocol, None);

        store.save(saved.clone()).await.unwrap();
        let loaded = store.load("team:dev-1", "card-1").await.unwrap();
        assert_eq!(loaded, Some(saved.clone()));

        let text = tokio::fs::read_to_string(store.path_for("team:dev-1", "card-1"))
            .await
            .unwrap();
        assert_eq!(text, serde_json::to_string(&saved).unwrap());
    }

    #[tokio::test]
    async fn unrecognised_state_survives_the_file_backend() {
        let directory = tempfile::tempdir().unwrap();
        let store = FileMemberCheckpointStore::new(directory.path());
        let saved = MemberCheckpointRecord::new(NativeCheckpoint::new("team:dev-1", "card-1"))
            .with_unrecognised_state("frozen");
        assert!(saved.has_unrecognised_state());

        store.save(saved.clone()).await.unwrap();

        let path = store.path_for("team:dev-1", "card-1");
        let text = tokio::fs::read_to_string(&path).await.unwrap();
        assert!(
            text.contains("\"unrecognised_state\":\"frozen\""),
            "the degradation is persisted: {text}"
        );
        assert_eq!(
            store.load("team:dev-1", "card-1").await.unwrap(),
            Some(saved)
        );
    }

    #[test]
    fn record_serde_tolerates_unknown_fields() {
        let encoded = serde_json::to_value(record("team:dev-1", "card-1")).unwrap();
        assert_eq!(encoded["checkpoint"]["session_id"], json!("team:dev-1"));
        assert_eq!(encoded["protocol"]["checkpoint_id"], json!("cp-3"));

        let mut tolerant = encoded.clone();
        tolerant["future_knob"] = json!({"any": "thing"});
        tolerant["checkpoint"]["legacy_field"] = json!(1);
        let mut decoded: MemberCheckpointRecord = serde_json::from_value(tolerant).unwrap();
        assert_eq!(decoded, record("team:dev-1", "card-1"));

        decoded.protocol = None;
        let encoded = serde_json::to_value(&decoded).unwrap();
        assert_eq!(encoded["protocol"], serde_json::Value::Null);
        assert_eq!(
            serde_json::from_value::<MemberCheckpointRecord>(json!({
                "checkpoint": {"session_id": "s", "card_id": "c"},
            }))
            .unwrap(),
            MemberCheckpointRecord::new(NativeCheckpoint::new("s", "c"))
        );
    }

    /// Sanitisation is private, so the tests probe it through [`FileMemberCheckpointStore`].
    fn sanitiser_probe(component: &str) -> String {
        let store = FileMemberCheckpointStore::new("/root");
        let path = store.path_for(component, "c");
        path.parent()
            .and_then(Path::parent)
            .and_then(Path::file_name)
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default()
    }
}
