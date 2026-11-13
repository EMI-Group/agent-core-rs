//! Serializable run state: the kind and phase vocabulary, the start request, the
//! terminal outcome, and the per-run status and event snapshots.
//!
//! Every type is serde-serializable (`snake_case`, tolerant of unknown fields and,
//! through `#[serde(default)]`, of missing ones), so a host can persist a snapshot
//! or ship an event to a UI unchanged.
//!
//! Python parity: Python's runner had no standalone status type; the fields mirror
//! what `team_runner.py` kept in its active-team records (component label, session
//! id, status) and what the callback `_framework:*` events carried (`run_id`,
//! component, session id).

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// The kind of entry a run is executing.
///
/// The variants mirror the runner's non-streaming entry points
/// ([`crate::Runner::run_agent`], [`crate::Runner::run_workflow`],
/// [`crate::Runner::run_agent_team`]); a streaming call reports the kind of the
/// invocation it streams.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunKind {
    /// A single-agent invocation.
    #[default]
    Agent,
    /// A workflow invocation.
    Workflow,
    /// An agent-team invocation.
    Team,
}

/// The lifecycle phase of a run.
///
/// [`RunPhase::Queued`] is the initial phase a run is created in;
/// [`RunPhase::Running`] is entered just before the invocation starts,
/// [`RunPhase::Paused`] describes a run waiting on a human or an external event,
/// and the three terminal phases are reached through
/// [`RunMonitor::finish`](crate::monitor::RunMonitor::finish).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunPhase {
    /// Created and admitted, not invoked yet.
    #[default]
    Queued,
    /// The invocation is in flight.
    Running,
    /// Suspended (human-in-the-loop or an external wait).
    Paused,
    /// Finished successfully.
    Completed,
    /// Finished with an error.
    Failed,
    /// Finished because it was cancelled.
    Cancelled,
}

impl RunPhase {
    /// Returns `true` for [`Completed`](Self::Completed), [`Failed`](Self::Failed),
    /// and [`Cancelled`](Self::Cancelled): a terminal run accepts no further
    /// transitions and is not listed by
    /// [`RunMonitor::list_active`](crate::monitor::RunMonitor::list_active).
    #[must_use]
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Completed | Self::Failed | Self::Cancelled)
    }
}

/// The request that opens a run: what is being executed and how to label it.
///
/// [`RunMonitor::begin`](crate::monitor::RunMonitor::begin) mints the id;
/// [`RunMonitor::track`](crate::monitor::RunMonitor::track) takes a caller-supplied
/// one (for example the id the run path already uses for tracing).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct RunStart {
    /// Which entry point is executing.
    pub kind: RunKind,
    /// Human-readable component label (the agent/workflow/team card name).
    pub label: String,
    /// Session the run is bound to, when the session was already resolved.
    pub session_id: Option<String>,
    /// Opaque caller detail shown with the initial [`RunPhase::Queued`] event.
    pub detail: Value,
}

impl Default for RunStart {
    /// An unlabelled [`RunKind::Agent`] start with no session and a null detail.
    fn default() -> Self {
        Self {
            kind: RunKind::default(),
            label: String::new(),
            session_id: None,
            detail: Value::Null,
        }
    }
}

impl RunStart {
    /// Builds a start for `kind` labelled `label`, with no session and a null detail.
    #[must_use]
    pub fn new(kind: RunKind, label: impl Into<String>) -> Self {
        Self {
            kind,
            label: label.into(),
            ..Self::default()
        }
    }

    /// Attaches the resolved session id.
    #[must_use]
    pub fn with_session(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }

    /// Attaches the caller detail reported with the initial event.
    #[must_use]
    pub fn with_detail(mut self, detail: Value) -> Self {
        self.detail = detail;
        self
    }
}

/// The terminal result a run finishes with.
///
/// The constructors cover the three terminal phases; [`RunOutcome::with_detail`]
/// attaches the caller detail that is copied into the final
/// [`RunStatus::detail`] and the final
/// [`RunEvent::detail`](crate::monitor::RunEvent::detail).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct RunOutcome {
    /// Terminal phase: [`RunPhase::Completed`], [`RunPhase::Failed`], or
    /// [`RunPhase::Cancelled`]. It is recorded as given.
    pub phase: RunPhase,
    /// The failure message for [`RunPhase::Failed`]; `None` otherwise.
    pub error: Option<String>,
    /// Opaque caller detail for the final event.
    pub detail: Value,
}

impl Default for RunOutcome {
    /// Same as [`RunOutcome::completed`].
    fn default() -> Self {
        Self::completed()
    }
}

impl RunOutcome {
    /// A successful outcome ([`RunPhase::Completed`], no error).
    #[must_use]
    pub fn completed() -> Self {
        Self {
            phase: RunPhase::Completed,
            error: None,
            detail: Value::Null,
        }
    }

    /// A failed outcome ([`RunPhase::Failed`]) carrying `message`.
    #[must_use]
    pub fn failed(message: impl Into<String>) -> Self {
        Self {
            phase: RunPhase::Failed,
            error: Some(message.into()),
            detail: Value::Null,
        }
    }

    /// A cancelled outcome ([`RunPhase::Cancelled`], no error).
    #[must_use]
    pub fn cancelled() -> Self {
        Self {
            phase: RunPhase::Cancelled,
            error: None,
            detail: Value::Null,
        }
    }

    /// Attaches the caller detail for the final event.
    #[must_use]
    pub fn with_detail(mut self, detail: Value) -> Self {
        self.detail = detail;
        self
    }
}

/// The current snapshot of one tracked run.
///
/// A live run is reported by [`RunMonitor::list_active`](crate::monitor::RunMonitor::list_active)
/// and is never evicted; a finished run is retained for the newest
/// `LIVE_RUN_LIMIT` completions and then dropped (see
/// [`RunMonitor`](crate::monitor::RunMonitor)).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct RunStatus {
    /// The run id: minted by [`RunMonitor::begin`](crate::monitor::RunMonitor::begin),
    /// or supplied to [`RunMonitor::track`](crate::monitor::RunMonitor::track).
    pub run_id: String,
    /// Which entry point the run executes.
    pub kind: RunKind,
    /// Human-readable component label (the card name).
    pub label: String,
    /// Current lifecycle phase.
    pub phase: RunPhase,
    /// Session the run is bound to, once resolved.
    pub session_id: Option<String>,
    /// Epoch milliseconds when the run was admitted.
    pub started_at: i64,
    /// Epoch milliseconds of the terminal transition; `None` while live.
    pub finished_at: Option<i64>,
    /// The failure message of a [`RunPhase::Failed`] run; `None` otherwise.
    pub error: Option<String>,
    /// Opaque caller detail of the latest transition.
    pub detail: Value,
}

impl RunStatus {
    /// Returns `true` once the run reached a terminal phase.
    #[must_use]
    pub fn is_terminal(&self) -> bool {
        self.phase.is_terminal()
    }
}

impl Default for RunStatus {
    /// An empty [`RunPhase::Queued`] status with no id and no timestamps.
    fn default() -> Self {
        Self {
            run_id: String::new(),
            kind: RunKind::default(),
            label: String::new(),
            phase: RunPhase::default(),
            session_id: None,
            started_at: 0,
            finished_at: None,
            error: None,
            detail: Value::Null,
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]

    use super::*;
    use serde_json::json;

    fn status() -> RunStatus {
        RunStatus {
            run_id: "run-1".to_string(),
            kind: RunKind::Workflow,
            label: "wf".to_string(),
            phase: RunPhase::Completed,
            session_id: Some("s-1".to_string()),
            started_at: 1_700_000_000_000,
            finished_at: Some(1_700_000_000_500),
            error: None,
            detail: json!({"steps": 3}),
        }
    }

    #[test]
    fn phases_report_terminality() {
        assert!(!RunPhase::Queued.is_terminal());
        assert!(!RunPhase::Running.is_terminal());
        assert!(!RunPhase::Paused.is_terminal());
        assert!(RunPhase::Completed.is_terminal());
        assert!(RunPhase::Failed.is_terminal());
        assert!(RunPhase::Cancelled.is_terminal());
        assert_eq!(RunPhase::default(), RunPhase::Queued);
        assert_eq!(RunKind::default(), RunKind::Agent);
    }

    #[test]
    fn kinds_and_phases_serialize_as_snake_case() {
        assert_eq!(
            serde_json::to_value(RunKind::Agent).unwrap(),
            json!("agent")
        );
        assert_eq!(
            serde_json::to_value(RunKind::Workflow).unwrap(),
            json!("workflow")
        );
        assert_eq!(serde_json::to_value(RunKind::Team).unwrap(), json!("team"));
        assert_eq!(
            serde_json::to_value(RunPhase::Cancelled).unwrap(),
            json!("cancelled")
        );
        assert_eq!(
            serde_json::from_value::<RunPhase>(json!("failed")).unwrap(),
            RunPhase::Failed
        );
    }

    #[test]
    fn status_json_has_the_documented_shape() {
        let value = serde_json::to_value(status()).unwrap();
        assert_eq!(value["run_id"], json!("run-1"));
        assert_eq!(value["kind"], json!("workflow"));
        assert_eq!(value["phase"], json!("completed"));
        assert_eq!(value["session_id"], json!("s-1"));
        assert_eq!(value["started_at"], json!(1_700_000_000_000_i64));
        assert_eq!(value["finished_at"], json!(1_700_000_000_500_i64));
        assert_eq!(value["error"], Value::Null);
        assert_eq!(value["detail"], json!({"steps": 3}));

        let live = RunStatus {
            finished_at: None,
            phase: RunPhase::Running,
            ..status()
        };
        let value = serde_json::to_value(&live).unwrap();
        assert_eq!(value["finished_at"], Value::Null);
        assert_eq!(value["phase"], json!("running"));
    }

    #[test]
    fn status_round_trips_and_tolerates_unknown_fields() {
        let original = status();
        let text = serde_json::to_string(&original).unwrap();
        let back: RunStatus = serde_json::from_str(&text).unwrap();
        assert_eq!(back, original);

        let value = json!({"run_id": "run-2", "phase": "running", "future": true});
        let parsed: RunStatus = serde_json::from_value(value).unwrap();
        assert_eq!(parsed.run_id, "run-2");
        assert_eq!(parsed.phase, RunPhase::Running);
        assert_eq!(parsed.kind, RunKind::Agent);
        assert_eq!(parsed.started_at, 0);
        assert!(parsed.finished_at.is_none());
    }

    #[test]
    fn start_builders_attach_the_session_and_detail() {
        let start = RunStart::new(RunKind::Team, "crew")
            .with_session("s-1")
            .with_detail(json!({"query": "hi"}));
        assert_eq!(start.kind, RunKind::Team);
        assert_eq!(start.label, "crew");
        assert_eq!(start.session_id.as_deref(), Some("s-1"));
        assert_eq!(start.detail, json!({"query": "hi"}));

        let bare = RunStart::default();
        assert_eq!(bare.kind, RunKind::Agent);
        assert!(bare.label.is_empty());
        assert!(bare.session_id.is_none());
        assert_eq!(bare.detail, Value::Null);
    }

    #[test]
    fn outcome_constructors_set_the_terminal_phase_and_error() {
        let completed = RunOutcome::completed();
        assert_eq!(completed.phase, RunPhase::Completed);
        assert!(completed.error.is_none());

        let failed = RunOutcome::failed("boom").with_detail(json!({"step": "llm"}));
        assert_eq!(failed.phase, RunPhase::Failed);
        assert_eq!(failed.error.as_deref(), Some("boom"));
        assert_eq!(failed.detail, json!({"step": "llm"}));

        let cancelled = RunOutcome::cancelled();
        assert_eq!(cancelled.phase, RunPhase::Cancelled);
        assert!(cancelled.error.is_none());
        assert_eq!(RunOutcome::default(), RunOutcome::completed());
    }

    #[test]
    fn outcome_round_trips_through_json() {
        let original = RunOutcome::failed("boom").with_detail(json!({"code": 7}));
        let value = serde_json::to_value(&original).unwrap();
        assert_eq!(value["phase"], json!("failed"));
        assert_eq!(value["error"], json!("boom"));
        let back: RunOutcome = serde_json::from_value(value).unwrap();
        assert_eq!(back, original);
    }
}
