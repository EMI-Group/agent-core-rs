//! Enumerating the persisted executions of a log root.
//!
//! A host that wants to offer "resume one of these executions" has to find the journals first.
//! [`ExecutionCatalog`] is that catalogue: it reads the `<root>/<percent-encoded execution id>/`
//! directories in the layout [`RootLogSource`] defines, and reports one [`ExecutionSummary`] per
//! readable journal. It deliberately shares [`RootLogSource`]'s path rule and journal opening, so
//! the catalogue can never disagree with the resume path about where a journal lives.
//!
//! Listing is best effort per directory: a missing root yields an empty list, and an unreadable
//! directory, a malformed `meta.json` or a journal that cannot be read is logged with
//! `tracing::warn!` and skipped (or, for a journal read failure, reported as unusable), so one
//! damaged execution never hides the healthy ones. The output is sorted deterministically by
//! `created_at` and then `execution_id`.
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};

use super::continuation::{ExecutionLogSource, RootLogSource};
use super::log::{ExecutionLog, LogCapacity, LogMeta, LOG_META_FILE};
use super::util::read_json;
use super::Clock;

/// One persisted execution as [`ExecutionCatalog::list`] reports it.
///
/// The record names the execution (its six identity fields), when its journal was created, how far
/// it has advanced and whether its entries are usable. It carries no entries, so listing a large
/// root stays bounded, and it is serde data so a host can cache it, sort it or show it.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ExecutionSummary {
    /// Unique id of the execution; also names its journal directory.
    pub execution_id: String,
    /// Host namespace that owns the execution.
    pub namespace_id: String,
    /// Agent that runs the execution.
    pub agent_id: String,
    /// Session the execution belongs to.
    pub session_id: String,
    /// Request that started the execution.
    pub request_id: String,
    /// Host correlation reference of the source message or trigger.
    pub source_ref: String,
    /// RFC 3339 creation timestamp read from the journal's `meta.json`.
    pub created_at: String,
    /// Next archive segment number of the journal.
    pub generation: u64,
    /// Number of usable entries the journal holds.
    pub entries: usize,
    /// Sequence number of the newest usable entry (`None` for an empty journal).
    pub last_seq: Option<u64>,
    /// Whether the journal has at least one usable entry.
    pub usable: bool,
}

/// The persisted executions of one log root.
///
/// The layout is [`RootLogSource`]'s: `<root>/<percent-encoded execution id>/meta.json` plus the
/// journal segments. [`ExecutionCatalog::open`] delegates to the same source, so a host can list
/// with the catalogue and resume with the very same path rule.
#[derive(Debug, Clone)]
pub struct ExecutionCatalog {
    /// The source that owns the layout and the clock journals are opened with.
    source: RootLogSource,
}

impl ExecutionCatalog {
    /// Builds a catalogue over `root`, stamping opened journals through the system clock.
    #[must_use]
    pub fn new(root: impl Into<PathBuf>, capacity: LogCapacity) -> Self {
        Self {
            source: RootLogSource::new(root, capacity),
        }
    }

    /// Replaces the clock opened journals are stamped with.
    #[must_use]
    pub fn with_clock(self, clock: Arc<dyn Clock>) -> Self {
        Self {
            source: self.source.with_clock(clock),
        }
    }

    /// Returns the root directory the catalogued journals live under.
    #[must_use]
    pub fn root(&self) -> &Path {
        self.source.root()
    }

    /// Returns the journal directory of `execution_id`, in the layout
    /// [`RootLogSource`] uses.
    #[must_use]
    pub fn path_for(&self, execution_id: &str) -> PathBuf {
        self.source.dir_for(execution_id)
    }

    /// Lists one summary per readable journal, ordered by `created_at` then `execution_id`.
    ///
    /// A missing root yields an empty list. An unreadable directory entry, a directory without a
    /// `meta.json` and a `meta.json` that does not parse are logged with `tracing::warn!` and
    /// skipped. A journal whose entries cannot be read is still reported, with `usable = false`
    /// and no entries, so the host can see the damaged execution instead of it disappearing.
    ///
    /// # Errors
    /// `Error::Session` naming the root when it exists but cannot be read at all.
    pub async fn list(&self) -> Result<Vec<ExecutionSummary>> {
        let root = self.source.root();
        let mut summaries = Vec::new();
        for directory in subdirectories(root).await? {
            let meta_path = directory.join(LOG_META_FILE);
            let meta = match read_json::<LogMeta>(&meta_path).await {
                Ok(Some(meta)) => meta,
                Ok(None) => {
                    tracing::warn!(
                        path = %meta_path.display(),
                        "skipping a persistent-agent log directory without metadata"
                    );
                    continue;
                }
                Err(error) => {
                    tracing::warn!(
                        path = %meta_path.display(),
                        %error,
                        "skipping a persistent-agent log directory with malformed metadata"
                    );
                    continue;
                }
            };
            let execution_id = meta.identity.execution_id.clone();
            let (generation, entries, last_seq, usable) = match self
                .source
                .open_dir(&directory, meta.identity.clone())
                .await
            {
                Ok(log) => match log.read().await {
                    Ok(snapshot) => (
                        log.cursor().await.generation,
                        snapshot.entries.len(),
                        snapshot.entries.last().map(|entry| entry.seq),
                        snapshot.is_usable(),
                    ),
                    Err(error) => {
                        tracing::warn!(
                            execution_id = %execution_id,
                            %error,
                            "persistent-agent journal is not readable; reporting it as unusable"
                        );
                        (meta.generation, 0_usize, None, false)
                    }
                },
                Err(error) => {
                    tracing::warn!(
                        execution_id = %execution_id,
                        %error,
                        "persistent-agent journal cannot be opened; reporting it as unusable"
                    );
                    (meta.generation, 0_usize, None, false)
                }
            };
            summaries.push(ExecutionSummary {
                execution_id,
                namespace_id: meta.identity.namespace_id,
                agent_id: meta.identity.agent_id,
                session_id: meta.identity.session_id,
                request_id: meta.identity.request_id,
                source_ref: meta.identity.source_ref,
                created_at: meta.created_at,
                generation,
                entries,
                last_seq,
                usable,
            });
        }
        summaries.sort_by(|left, right| {
            left.created_at
                .cmp(&right.created_at)
                .then_with(|| left.execution_id.cmp(&right.execution_id))
        });
        Ok(summaries)
    }

    /// Opens the journal of `execution_id`.
    ///
    /// # Errors
    /// `Error::NotFound` when no journal exists for `execution_id`, or any error
    /// [`ExecutionLogSource::open_log`] reports.
    pub async fn open(&self, execution_id: &str) -> Result<ExecutionLog> {
        self.source.open_log(execution_id).await
    }
}

/// Returns the sorted subdirectories of `root`, or an empty list when it is absent.
///
/// A single unreadable entry is logged and skipped; only a root that cannot be opened at all is an
/// error.
async fn subdirectories(root: &Path) -> Result<Vec<PathBuf>> {
    let mut read_dir = match tokio::fs::read_dir(root).await {
        Ok(entries) => entries,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => {
            return Err(Error::session(format!(
                "failed to read persistent-agent log root {}: {error}",
                root.display()
            )));
        }
    };
    let mut found = Vec::new();
    loop {
        let entry = match read_dir.next_entry().await {
            Ok(Some(entry)) => entry,
            Ok(None) => break,
            Err(error) => {
                tracing::warn!(
                    root = %root.display(),
                    %error,
                    "stopping the persistent-agent log scan after an unreadable entry"
                );
                break;
            }
        };
        match entry.file_type().await {
            Ok(kind) if kind.is_dir() => found.push(entry.path()),
            Ok(_) => {}
            Err(error) => {
                tracing::warn!(
                    path = %entry.path().display(),
                    %error,
                    "skipping an unreadable persistent-agent log directory entry"
                );
            }
        }
    }
    found.sort();
    Ok(found)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::sync::Arc;

    use openjiuwen_common::error::Error;
    use serde_json::json;

    use super::{ExecutionCatalog, ExecutionSummary};
    use crate::persistent_agent::entry::{EntryPayload, ModelUsage};
    use crate::persistent_agent::identity::ExecutionIdentity;
    use crate::persistent_agent::log::{ExecutionLog, LogCapacity, LOG_META_FILE};
    use crate::persistent_agent::util::percent_encode_identifier;
    use crate::persistent_agent::ManualClock;

    /// A frozen instant used to give each fixture journal a distinct creation time.
    const FIRST_STAMP: i64 = 1_700_000_000_000;

    fn identity(execution_id: &str) -> ExecutionIdentity {
        ExecutionIdentity::new("ns", "agent", "session", "request", execution_id, "source").unwrap()
    }

    /// Writes a journal under the catalogue layout, stamped at `stamp`, with `payloads` appended.
    async fn seed_log(
        root: &std::path::Path,
        execution_id: &str,
        stamp: i64,
        payloads: &[EntryPayload],
    ) {
        let dir = root.join(percent_encode_identifier(execution_id));
        let log = ExecutionLog::open_with_clock(
            dir,
            identity(execution_id),
            LogCapacity::default(),
            Arc::new(ManualClock::new(stamp)),
        )
        .await
        .unwrap();
        for payload in payloads {
            log.append(payload.clone()).await.unwrap();
        }
    }

    #[tokio::test]
    async fn listing_a_missing_root_is_empty() {
        let dir = tempfile::TempDir::new().unwrap();
        let catalog = ExecutionCatalog::new(dir.path().join("absent"), LogCapacity::default());
        assert_eq!(catalog.list().await.unwrap().len(), 0);
    }

    #[tokio::test]
    async fn summary_reports_identity_counts_and_sequence() {
        let dir = tempfile::TempDir::new().unwrap();
        seed_log(
            dir.path(),
            "exec-1",
            FIRST_STAMP,
            &[
                EntryPayload::execution_started("do work"),
                EntryPayload::model_turn("mc_1", "done", ModelUsage::known(10, 5)),
            ],
        )
        .await;

        let catalog = ExecutionCatalog::new(dir.path(), LogCapacity::default());
        let summaries = catalog.list().await.unwrap();
        assert_eq!(summaries.len(), 1);
        let summary = &summaries[0];
        assert_eq!(summary.execution_id, "exec-1");
        assert_eq!(summary.namespace_id, "ns");
        assert_eq!(summary.agent_id, "agent");
        assert_eq!(summary.session_id, "session");
        assert_eq!(summary.request_id, "request");
        assert_eq!(summary.source_ref, "source");
        assert_eq!(summary.created_at, "2023-11-14T22:13:20.000Z");
        assert_eq!(summary.generation, 1);
        assert_eq!(summary.entries, 2);
        assert_eq!(summary.last_seq, Some(2));
        assert!(summary.usable);
    }

    #[tokio::test]
    async fn listing_skips_corrupt_metadata_and_orders_deterministically() {
        let dir = tempfile::TempDir::new().unwrap();
        // Two healthy journals with distinct creation times, seeded newest-first on disk.
        seed_log(
            dir.path(),
            "zeta",
            FIRST_STAMP + 1_000,
            &[EntryPayload::note("later")],
        )
        .await;
        seed_log(
            dir.path(),
            "alpha",
            FIRST_STAMP,
            &[EntryPayload::note("earlier")],
        )
        .await;
        // Same creation time as `alpha`, so the tie-break is the execution id.
        seed_log(
            dir.path(),
            "beta",
            FIRST_STAMP,
            &[EntryPayload::note("same instant")],
        )
        .await;
        // A directory whose `meta.json` is not JSON, and one without any metadata at all.
        let corrupt = dir.path().join("broken");
        tokio::fs::create_dir_all(&corrupt).await.unwrap();
        tokio::fs::write(corrupt.join(LOG_META_FILE), b"{not json")
            .await
            .unwrap();
        tokio::fs::create_dir_all(dir.path().join("empty"))
            .await
            .unwrap();

        let catalog = ExecutionCatalog::new(dir.path(), LogCapacity::default());
        let summaries = catalog.list().await.unwrap();
        let ids: Vec<&str> = summaries
            .iter()
            .map(|summary| summary.execution_id.as_str())
            .collect();
        assert_eq!(ids, vec!["alpha", "beta", "zeta"]);
    }

    #[tokio::test]
    async fn open_uses_the_root_log_layout() {
        let dir = tempfile::TempDir::new().unwrap();
        seed_log(
            dir.path(),
            "exec/1",
            FIRST_STAMP,
            &[EntryPayload::note("hi")],
        )
        .await;

        let catalog = ExecutionCatalog::new(dir.path(), LogCapacity::default());
        assert_eq!(
            catalog.path_for("exec/1"),
            dir.path().join("exec%2F1"),
            "the path rule is RootLogSource's"
        );
        let log = catalog.open("exec/1").await.unwrap();
        assert_eq!(log.identity().execution_id, "exec/1");
        assert_eq!(log.read().await.unwrap().entries.len(), 1);

        let missing = catalog.open("absent").await.unwrap_err();
        assert!(matches!(missing, Error::NotFound(_)), "got {missing:?}");
    }

    #[test]
    fn summary_round_trips_and_tolerates_unknown_fields() {
        let summary = ExecutionSummary {
            execution_id: "exec-1".to_string(),
            namespace_id: "ns".to_string(),
            agent_id: "agent".to_string(),
            session_id: "session".to_string(),
            request_id: "request".to_string(),
            source_ref: "source".to_string(),
            created_at: "2023-11-14T22:13:20.000Z".to_string(),
            generation: 2,
            entries: 3,
            last_seq: Some(3),
            usable: true,
        };
        let encoded = serde_json::to_value(&summary).unwrap();
        let decoded: ExecutionSummary = serde_json::from_value(encoded).unwrap();
        assert_eq!(summary, decoded);
        let tolerant: ExecutionSummary = serde_json::from_value(json!({
            "execution_id": "exec-1",
            "namespace_id": "ns",
            "agent_id": "agent",
            "session_id": "session",
            "request_id": "request",
            "source_ref": "source",
            "created_at": "2023-11-14T22:13:20.000Z",
            "generation": 2,
            "entries": 3,
            "last_seq": 3,
            "usable": true,
            "future_field": {"nested": true},
        }))
        .unwrap();
        assert_eq!(tolerant, summary);
        let sparse: ExecutionSummary =
            serde_json::from_value(json!({"execution_id": "e"})).unwrap();
        assert_eq!(sparse.execution_id, "e");
        assert_eq!(sparse.entries, 0);
    }
}
