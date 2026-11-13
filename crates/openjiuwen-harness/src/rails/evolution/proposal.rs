//! The bounded, persisted log of self-evolution proposals.
//!
//! A proposal is what an evolution body wants to change next (a skill patch, a
//! prompt tweak, a tool-description rewrite). The log keeps at most `capacity`
//! of them, newest last (unbounded at `capacity == 0`), and can atomically
//! persist them to a configured path so a later run — or an orchestrator in a
//! crate above this one — can read them back.
//!
//! The on-disk document is `{"version": 1, "proposals": [...]}`; a bare JSON
//! array is accepted too, which keeps hand-written fixtures loadable.
use std::fmt;
use std::path::{Path, PathBuf};

use openjiuwen_common::error::{Error, Result};
use parking_lot::Mutex;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::store;

/// Version of the persisted proposal-log document.
const PROPOSAL_LOG_VERSION: u32 = 1;

/// One change a self-evolution body proposes.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(default)]
pub struct EvolutionProposal {
    /// Stable id; [`EvolutionProposal::new`] mints a fresh uuid.
    pub id: String,
    /// Free-form kind, e.g. `skill_patch` or `prompt_tweak`.
    pub kind: String,
    /// One-line human-readable summary of the change.
    pub summary: String,
    /// Clock reading when the proposal was drafted; `0` means "not stamped".
    pub created_at_ms: u64,
}

impl EvolutionProposal {
    /// Builds an unstamped proposal with a fresh uuid.
    #[must_use]
    pub fn new(kind: impl Into<String>, summary: impl Into<String>) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            kind: kind.into(),
            summary: summary.into(),
            created_at_ms: 0,
        }
    }

    /// Stamps the clock reading the proposal was drafted at.
    #[must_use]
    pub fn with_created_at_ms(mut self, created_at_ms: u64) -> Self {
        self.created_at_ms = created_at_ms;
        self
    }
}

/// The persisted shape of a proposal log.
#[derive(Debug, Serialize, Deserialize)]
struct ProposalLogDocument {
    /// Document version.
    version: u32,
    /// The logged proposals, oldest first.
    proposals: Vec<EvolutionProposal>,
}

/// Both accepted on-disk shapes: the versioned document and a bare array.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ProposalLogDocumentRead {
    /// A bare array of proposals.
    List(Vec<EvolutionProposal>),
    /// The versioned document.
    Document(ProposalLogDocument),
}

/// A bounded in-memory proposal log that can persist itself atomically.
///
/// Every accessor takes `&self`, so one log can be shared by the rail and its
/// callers. Persistence is opt-in: without a configured path, [`ProposalLog::save`]
/// and [`ProposalLog::load`] return [`Error::config`].
pub struct ProposalLog {
    /// Maximum number of kept proposals, `0` for unbounded.
    capacity: usize,
    /// Destination of [`ProposalLog::save`], when configured.
    path: Option<PathBuf>,
    /// Oldest first; guarded because every method takes `&self`.
    entries: Mutex<Vec<EvolutionProposal>>,
}

impl ProposalLog {
    /// Creates an in-memory log with no persistence path.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            path: None,
            entries: Mutex::new(Vec::new()),
        }
    }

    /// Configures the path [`ProposalLog::save`] writes and
    /// [`ProposalLog::load`] reads.
    #[must_use]
    pub fn with_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.path = Some(path.into());
        self
    }

    /// The configured persistence path.
    #[must_use]
    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    /// Maximum number of kept proposals, `0` when unbounded.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Number of kept proposals.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.lock().len()
    }

    /// Whether no proposal is kept.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Appends one proposal, evicting the oldest when the log is full.
    pub fn push(&self, proposal: EvolutionProposal) {
        let mut entries = self.entries.lock();
        if self.capacity > 0 && entries.len() >= self.capacity {
            let len = entries.len();
            let excess = len.saturating_add(1).saturating_sub(self.capacity);
            entries.drain(..excess.min(len));
        }
        entries.push(proposal);
    }

    /// The kept proposals, oldest first.
    #[must_use]
    pub fn entries(&self) -> Vec<EvolutionProposal> {
        self.entries.lock().clone()
    }

    /// Drops every proposal; the capacity and path are kept.
    pub fn clear(&self) {
        self.entries.lock().clear();
    }

    /// Atomically writes the kept proposals to the configured path.
    ///
    /// # Errors
    ///
    /// Returns [`Error::config`] when no path is configured or the file cannot
    /// be written.
    pub fn save(&self) -> Result<()> {
        let Some(path) = self.path.as_deref() else {
            return Err(Error::config("proposal log has no configured path"));
        };
        let document = ProposalLogDocument {
            version: PROPOSAL_LOG_VERSION,
            proposals: self.entries(),
        };
        store::write_json_atomic(path, &document, "proposal log")
    }

    /// Reads the proposals stored at the configured path without touching the
    /// in-memory log.
    ///
    /// A missing file is an empty log: the first run of a workspace has none.
    ///
    /// # Errors
    ///
    /// Returns [`Error::config`] when no path is configured or the file cannot
    /// be read, and [`Error::validation`] when the document is malformed.
    pub fn load(&self) -> Result<Vec<EvolutionProposal>> {
        let Some(path) = self.path.as_deref() else {
            return Err(Error::config("proposal log has no configured path"));
        };
        let Some(document) = store::read_json_opt::<ProposalLogDocumentRead>(path, "proposal log")?
        else {
            return Ok(Vec::new());
        };
        Ok(match document {
            ProposalLogDocumentRead::List(proposals) => proposals,
            ProposalLogDocumentRead::Document(document) => document.proposals,
        })
    }
}

impl fmt::Debug for ProposalLog {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ProposalLog")
            .field("capacity", &self.capacity)
            .field("path", &self.path)
            .field("len", &self.len())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use openjiuwen_common::error::ErrorKind;

    use super::*;

    fn proposal(kind: &str, summary: &str) -> EvolutionProposal {
        EvolutionProposal::new(kind, summary)
    }

    #[test]
    fn new_mints_a_uuid_and_leaves_the_stamp_unset() {
        let first = proposal("skill_patch", "tighten the diff prompt");
        let second = proposal("skill_patch", "tighten the diff prompt");
        assert_ne!(first.id, "");
        assert_ne!(first.id, second.id);
        assert_eq!(first.created_at_ms, 0);
        assert_eq!(
            first.with_created_at_ms(42).created_at_ms,
            42,
            "stamping keeps the rest of the proposal"
        );
    }

    #[test]
    fn proposals_keep_their_insertion_order() {
        let log = ProposalLog::new(0);
        log.push(proposal("a", "first"));
        log.push(proposal("b", "second"));

        let summaries: Vec<String> = log
            .entries()
            .into_iter()
            .map(|entry| entry.summary)
            .collect();
        assert_eq!(summaries, ["first", "second"]);
        assert_eq!(log.len(), 2);
    }

    #[test]
    fn bounded_log_drops_the_oldest_proposal() {
        let log = ProposalLog::new(2);
        log.push(proposal("a", "first"));
        log.push(proposal("b", "second"));
        log.push(proposal("c", "third"));

        let summaries: Vec<String> = log
            .entries()
            .into_iter()
            .map(|entry| entry.summary)
            .collect();
        assert_eq!(summaries, ["second", "third"]);
        assert_eq!(log.capacity(), 2);
    }

    #[test]
    fn clear_drops_entries_but_keeps_capacity_and_path() {
        let dir = tempfile::tempdir().unwrap();
        let log = ProposalLog::new(3).with_path(dir.path().join("proposals.json"));
        log.push(proposal("a", "first"));
        log.clear();

        assert_eq!(log.len(), 0);
        assert_eq!(log.capacity(), 3);
        assert!(log.path().is_some());
    }

    #[test]
    fn save_then_load_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("proposals.json");
        let log = ProposalLog::new(0).with_path(&path);
        log.push(proposal("skill_patch", "first").with_created_at_ms(7));
        log.push(proposal("prompt_tweak", "second"));

        log.save().unwrap();
        assert!(!path.with_extension("json.tmp").exists());

        let restored = ProposalLog::new(0).with_path(&path);
        assert_eq!(restored.load().unwrap(), log.entries());
    }

    #[test]
    fn a_bare_array_loads_too() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("proposals.json");
        let document = vec![proposal("skill_patch", "from a fixture")];
        std::fs::write(&path, serde_json::to_string(&document).unwrap()).unwrap();

        let log = ProposalLog::new(0).with_path(&path);
        assert_eq!(log.load().unwrap(), document);
    }

    #[test]
    fn missing_file_loads_as_an_empty_log() {
        let dir = tempfile::tempdir().unwrap();
        let log = ProposalLog::new(0).with_path(dir.path().join("absent.json"));
        assert_eq!(log.load().unwrap().len(), 0);
    }

    #[test]
    fn saving_without_a_path_is_a_config_error() {
        let log = ProposalLog::new(0);
        let error = log.save().unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Config);
        assert!(error.to_string().contains("no configured path"));

        assert_eq!(log.load().unwrap_err().kind(), ErrorKind::Config);
    }

    #[test]
    fn malformed_documents_are_validation_errors() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("proposals.json");
        std::fs::write(&path, "[\"not a proposal\"]").unwrap();

        let log = ProposalLog::new(0).with_path(&path);
        let error = log.load().unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("malformed proposal log file"));
    }

    #[test]
    fn save_creates_missing_parent_directories() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nested").join("proposals.json");
        let log = ProposalLog::new(0).with_path(&path);
        log.push(proposal("a", "first"));

        log.save().unwrap();
        assert!(path.exists());
    }

    #[test]
    fn debug_prints_capacity_path_and_len() {
        let log = ProposalLog::new(5).with_path("/tmp/proposals.json");
        let rendered = format!("{log:?}");
        assert!(rendered.contains("capacity: 5"), "{rendered}");
        assert!(rendered.contains("proposals.json"), "{rendered}");
    }
}
