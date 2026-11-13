//! The per-run evolution snapshot and its atomic file form.
//!
//! [`EvolutionSnapshot`] is what an evolution consumer reads after a run: one
//! `run_id` minted at invocation start, the agent/session it belongs to, when it
//! started and finished, the folded model/tool counters, the per-iteration
//! metrics tail, and the proposals the run drafted.
//!
//! It is written once, at invocation end, through the shared atomic JSON writer
//! (`<path>.tmp` + rename), so a crash never leaves a partial snapshot behind.
use std::path::Path;

use openjiuwen_common::error::{Error, Result};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::metrics::EvolutionMetrics;
use super::proposal::EvolutionProposal;
use super::store;

/// Version of the snapshot document this crate writes.
///
/// Consumers should reject a snapshot whose `version` they do not know rather
/// than guess at missing fields.
pub const EVOLUTION_SNAPSHOT_VERSION: u32 = 1;

/// The serde default of [`EvolutionSnapshot::version`].
const fn snapshot_version() -> u32 {
    EVOLUTION_SNAPSHOT_VERSION
}

/// Everything one harness run contributed to self-evolution.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(default)]
pub struct EvolutionSnapshot {
    /// Document version; [`EVOLUTION_SNAPSHOT_VERSION`] for snapshots written
    /// by this crate.
    #[serde(default = "snapshot_version")]
    pub version: u32,
    /// Fresh uuid of the run, minted when the invocation started.
    pub run_id: String,
    /// Id of the agent the run belongs to.
    pub agent_id: String,
    /// Session id, when the run had a session.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    /// Clock reading when the invocation started, in milliseconds.
    pub started_at_ms: u64,
    /// Clock reading when the invocation finished, in milliseconds.
    pub finished_at_ms: u64,
    /// `finished_at_ms - started_at_ms`, saturating at `0`.
    pub duration_ms: u64,
    /// Number of outer task-loop iterations the run completed, including rows
    /// evicted from [`EvolutionSnapshot::iterations`].
    pub iteration_count: usize,
    /// Model calls of the whole run.
    pub model_calls: u64,
    /// Tool calls of the whole run.
    pub tool_calls: u64,
    /// Tool calls of the whole run that returned a failure.
    pub failed_tool_calls: u64,
    /// Provider stop reason of the run, when one was reported.
    pub stop_reason: Option<String>,
    /// Per-iteration metrics, oldest first (the bounded tail of the run).
    pub iterations: Vec<EvolutionMetrics>,
    /// Proposals the run drafted, oldest first.
    pub proposals: Vec<EvolutionProposal>,
}

impl Default for EvolutionSnapshot {
    fn default() -> Self {
        Self {
            version: EVOLUTION_SNAPSHOT_VERSION,
            run_id: String::new(),
            agent_id: String::new(),
            session_id: None,
            started_at_ms: 0,
            finished_at_ms: 0,
            duration_ms: 0,
            iteration_count: 0,
            model_calls: 0,
            tool_calls: 0,
            failed_tool_calls: 0,
            stop_reason: None,
            iterations: Vec::new(),
            proposals: Vec::new(),
        }
    }
}

impl EvolutionSnapshot {
    /// Atomically writes the snapshot to `path` as pretty JSON.
    ///
    /// # Errors
    ///
    /// Returns [`Error::config`] when the document cannot be encoded or the file
    /// cannot be written.
    pub fn save(&self, path: &Path) -> Result<()> {
        store::write_json_atomic(path, self, "evolution snapshot")
    }

    /// Reads a snapshot written by [`EvolutionSnapshot::save`].
    ///
    /// Unknown fields are tolerated, so a newer writer stays loadable here.
    ///
    /// # Errors
    ///
    /// Returns [`Error::config`] when the file is missing or unreadable, and
    /// [`Error::validation`] when the document is malformed.
    pub fn load(path: &Path) -> Result<Self> {
        store::read_json_opt(path, "evolution snapshot")?.ok_or_else(|| {
            Error::config(format!("evolution snapshot {} is missing", path.display()))
        })
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use openjiuwen_common::error::ErrorKind;

    use super::*;

    fn snapshot() -> EvolutionSnapshot {
        EvolutionSnapshot {
            run_id: "run-1".to_owned(),
            agent_id: "agent-1".to_owned(),
            session_id: Some("session-1".to_owned()),
            started_at_ms: 100,
            finished_at_ms: 175,
            duration_ms: 75,
            iteration_count: 1,
            model_calls: 2,
            tool_calls: 3,
            failed_tool_calls: 1,
            stop_reason: Some("stop".to_owned()),
            iterations: vec![EvolutionMetrics {
                iteration: 0,
                elapsed_ms: 75,
                model_calls: 2,
                ..EvolutionMetrics::default()
            }],
            proposals: vec![EvolutionProposal::new("skill_patch", "first")],
            ..EvolutionSnapshot::default()
        }
    }

    #[test]
    fn default_carries_the_current_version() {
        assert_eq!(
            EvolutionSnapshot::default().version,
            EVOLUTION_SNAPSHOT_VERSION
        );
        assert_eq!(EVOLUTION_SNAPSHOT_VERSION, 1);
    }

    #[test]
    fn save_then_load_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("snapshot.json");
        let written = snapshot();

        written.save(&path).unwrap();
        assert!(!dir.path().join("snapshot.json.tmp").exists());
        assert_eq!(EvolutionSnapshot::load(&path).unwrap(), written);
    }

    #[test]
    fn loading_tolerates_unknown_fields_and_a_missing_version() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("snapshot.json");
        std::fs::write(
            &path,
            r#"{"run_id": "run-9", "agent_id": "a", "future_field": true}"#,
        )
        .unwrap();

        let loaded = EvolutionSnapshot::load(&path).unwrap();
        assert_eq!(loaded.run_id, "run-9");
        assert_eq!(loaded.version, EVOLUTION_SNAPSHOT_VERSION);
        assert_eq!(loaded.agent_id, "a");
    }

    #[test]
    fn a_missing_snapshot_is_a_config_error() {
        let dir = tempfile::tempdir().unwrap();
        let error = EvolutionSnapshot::load(&dir.path().join("absent.json")).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Config);
        assert!(error.to_string().contains("is missing"));
    }

    #[test]
    fn a_malformed_snapshot_is_a_validation_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("snapshot.json");
        std::fs::write(&path, "{\"run_id\": ").unwrap();

        let error = EvolutionSnapshot::load(&path).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("malformed evolution snapshot"));
    }

    #[test]
    fn an_unwritable_destination_is_a_config_error() {
        let dir = tempfile::tempdir().unwrap();
        let blocker = dir.path().join("blocker");
        std::fs::write(&blocker, "not a directory").unwrap();

        let error = snapshot().save(&blocker.join("snapshot.json")).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Config);
        assert!(error.to_string().contains("evolution snapshot"));
    }
}
