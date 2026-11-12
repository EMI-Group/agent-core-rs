//! Events emitted by an auto-harness session.
//!
//! Ported from the chunk stream Python's `AutoHarnessOrchestrator` yields
//! (`openjiuwen/rsi/harness_rsi/auto_harness/orchestrator.py`, v0.1.20): every
//! chunk is an [`OutputSchema`] carrying a `type` tag and a payload, and the
//! terminal chunk is tagged `harness_session_finished`.
//!
//! Rust models the whole stream as one internally tagged enum: the wire tag key
//! is `type` (not serde's default externally tagged shape), variant names are
//! `snake_case`, and [`AutoHarnessEvent::kind`] returns the same tag without
//! serialising. Callers build a stream from these values (the orchestrator wraps
//! them in `BoxStream<'static, Result<AutoHarnessEvent>>`) and a UI keys its
//! renderer on `kind()` exactly like the Python CLI keys on `chunk.type`.
//!
//! [`OutputSchema`]: https://gitcode.com/openJiuwen/agent-core

use serde::{Deserialize, Serialize};

use crate::harness_rsi::auto_harness::task::{IterationRecord, SessionSummary, StageResult};

/// One event of an auto-harness session.
///
/// Serializes as a JSON object with a `"type"` tag holding the variant name
/// (`session_started`, `iteration_started`, …); the remaining fields are the
/// variant payload. `SessionFinished` uses the Python chunk kind
/// `harness_session_finished` instead of its Rust variant name.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AutoHarnessEvent {
    /// The session accepted a run and starts its iteration loop.
    SessionStarted {
        /// Session identifier.
        session_id: String,
        /// Iteration cap taken from the config.
        max_iterations: u32,
    },
    /// One iteration begins, seeded with the selected parents.
    IterationStarted {
        /// 1-based iteration number.
        iteration: u32,
        /// Ids of the candidates used as mutation parents.
        parents: Vec<String>,
    },
    /// Every candidate of the iteration has been evaluated.
    CandidatesEvaluated {
        /// 1-based iteration number.
        iteration: u32,
        /// Number of candidates the mutator proposed.
        proposed: usize,
        /// Number of candidates that were evaluated.
        evaluated: usize,
        /// Number of candidates that produced a valid report.
        valid: usize,
        /// Best score of the iteration, when anything was graded.
        best_score: Option<f64>,
    },
    /// One iteration is complete and recorded in the session history.
    IterationFinished {
        /// The appended iteration record.
        record: IterationRecord,
    },
    /// The session state was persisted after an iteration.
    CheckpointSaved {
        /// Session identifier.
        session_id: String,
        /// Iteration the persisted state belongs to.
        iteration: u32,
    },
    /// A pipeline stage reported its result.
    StageResult {
        /// Stage name (`plan`, `implement`, `assess`, …).
        stage: String,
        /// Stage outcome.
        result: StageResult,
    },
    /// The session finished; the Python chunk kind is `harness_session_finished`.
    #[serde(rename = "harness_session_finished")]
    SessionFinished {
        /// Final report of the session.
        summary: SessionSummary,
    },
    /// The session was cancelled before it finished.
    SessionCancelled {
        /// Iteration that was in flight (or last completed) at cancellation.
        iteration: u32,
    },
}

impl AutoHarnessEvent {
    /// Returns the serialized `type` tag (the Python chunk kind).
    ///
    /// [`AutoHarnessEvent::SessionFinished`] returns
    /// `"harness_session_finished"`, matching the rename on the variant, so a
    /// caller can route events without serialising them.
    #[must_use]
    pub fn kind(&self) -> &'static str {
        match self {
            Self::SessionStarted { .. } => "session_started",
            Self::IterationStarted { .. } => "iteration_started",
            Self::CandidatesEvaluated { .. } => "candidates_evaluated",
            Self::IterationFinished { .. } => "iteration_finished",
            Self::CheckpointSaved { .. } => "checkpoint_saved",
            Self::StageResult { .. } => "stage_result",
            Self::SessionFinished { .. } => "harness_session_finished",
            Self::SessionCancelled { .. } => "session_cancelled",
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::{json, Value};

    use super::*;
    use crate::evolution::CandidateId;
    use crate::harness_rsi::auto_harness::task::{StageStatus, StopReason};

    fn record() -> IterationRecord {
        IterationRecord {
            iteration: 1,
            parents: vec![CandidateId::new("seed")],
            proposed: 2,
            evaluated: 2,
            valid: 1,
            best_candidate: Some(CandidateId::new("seed/g1/v0")),
            best_score: Some(0.75),
            previous_best_score: None,
            improvement: 0.75,
            accepted: true,
            duration_ms: 12,
            stop: Some(StopReason::TargetScore),
        }
    }

    fn summary() -> SessionSummary {
        SessionSummary {
            session_id: "sess-1".to_owned(),
            iterations: 1,
            best_candidate: Some(CandidateId::new("seed/g1/v0")),
            best_score: Some(0.75),
            evaluated: 2,
            accepted: 1,
            stop_reason: Some(StopReason::TargetScore),
            spent_usd: 0.25,
        }
    }

    fn stage() -> StageResult {
        StageResult {
            status: StageStatus::Failed,
            messages: vec!["implement failed".to_owned()],
            error: "compile error".to_owned(),
            ..StageResult::default()
        }
    }

    fn events() -> Vec<AutoHarnessEvent> {
        vec![
            AutoHarnessEvent::SessionStarted {
                session_id: "sess-1".to_owned(),
                max_iterations: 5,
            },
            AutoHarnessEvent::IterationStarted {
                iteration: 1,
                parents: vec!["seed".to_owned()],
            },
            AutoHarnessEvent::CandidatesEvaluated {
                iteration: 1,
                proposed: 2,
                evaluated: 2,
                valid: 1,
                best_score: Some(0.75),
            },
            AutoHarnessEvent::IterationFinished { record: record() },
            AutoHarnessEvent::CheckpointSaved {
                session_id: "sess-1".to_owned(),
                iteration: 1,
            },
            AutoHarnessEvent::StageResult {
                stage: "implement".to_owned(),
                result: stage(),
            },
            AutoHarnessEvent::SessionFinished { summary: summary() },
            AutoHarnessEvent::SessionCancelled { iteration: 3 },
        ]
    }

    fn tag(value: &Value) -> &str {
        value
            .get("type")
            .and_then(Value::as_str)
            .expect("every event carries a type tag")
    }

    #[test]
    fn kinds_match_serialized_tags() {
        let expected = [
            "session_started",
            "iteration_started",
            "candidates_evaluated",
            "iteration_finished",
            "checkpoint_saved",
            "stage_result",
            "harness_session_finished",
            "session_cancelled",
        ];
        for (event, name) in events().iter().zip(expected) {
            assert_eq!(event.kind(), name);
            let value = serde_json::to_value(event).expect("event serializes");
            assert_eq!(tag(&value), name, "tag of {name}");
        }
    }

    #[test]
    fn session_finished_uses_the_python_chunk_kind() {
        let value = serde_json::to_value(AutoHarnessEvent::SessionFinished { summary: summary() })
            .expect("event serializes");
        assert_eq!(tag(&value), "harness_session_finished");
        assert_eq!(value["summary"]["session_id"], json!("sess-1"));
        assert_eq!(value["summary"]["stop_reason"], json!("target_score"));
    }

    #[test]
    fn events_round_trip_through_json() {
        for event in events() {
            let encoded = serde_json::to_string(&event).expect("event serializes");
            let decoded: AutoHarnessEvent = serde_json::from_str(&encoded).expect("event parses");
            assert_eq!(decoded, event);
        }
    }

    #[test]
    fn payload_fields_keep_python_names() {
        let value = serde_json::to_value(AutoHarnessEvent::IterationStarted {
            iteration: 2,
            parents: vec!["seed".to_owned(), "seed/g1/v0".to_owned()],
        })
        .expect("event serializes");
        assert_eq!(value["iteration"], json!(2));
        assert_eq!(value["parents"], json!(["seed", "seed/g1/v0"]));

        let value = serde_json::to_value(AutoHarnessEvent::CandidatesEvaluated {
            iteration: 2,
            proposed: 3,
            evaluated: 3,
            valid: 2,
            best_score: None,
        })
        .expect("event serializes");
        assert_eq!(value["proposed"], json!(3));
        assert_eq!(value["valid"], json!(2));
        assert_eq!(value["best_score"], Value::Null);

        let value = serde_json::to_value(AutoHarnessEvent::StageResult {
            stage: "implement".to_owned(),
            result: stage(),
        })
        .expect("event serializes");
        assert_eq!(value["stage"], json!("implement"));
        assert_eq!(value["result"]["status"], json!("failed"));
        assert_eq!(value["result"]["error"], json!("compile error"));
    }

    #[test]
    fn iteration_finished_round_trips_the_record() {
        let event = AutoHarnessEvent::IterationFinished { record: record() };
        let value = serde_json::to_value(&event).expect("event serializes");
        assert_eq!(value["record"]["improvement"], json!(0.75));
        assert_eq!(value["record"]["stop"], json!("target_score"));
        let decoded: AutoHarnessEvent = serde_json::from_value(value).expect("event parses");
        assert_eq!(decoded, event);
    }
}
