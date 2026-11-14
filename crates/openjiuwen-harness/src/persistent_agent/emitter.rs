//! Lifecycle events of one persistent execution: numbering, ids, recording and sinks.
//!
//! The [`Emitter`] is the only caller of an [`EventSink`] for one execution, so the order in which
//! events are recorded equals the order of their sequence numbers. Every emitted event carries the
//! execution identity, a 1-based `local_seq`, an injected RFC 3339 timestamp and a stable
//! `event_id` (`evt_` followed by a digest), so a host can deduplicate a retried event by id.
//!
//! Failures are split by kind: a critical event ([`EventType::is_critical`] — tool intent, pause and
//! final) that cannot be recorded fails the emission, while an ordinary event is only diagnosed and
//! still handed back. [`Emitter::emit_ending`] retries the ending once as the *same* event, so a
//! transient sink failure does not duplicate the terminal record.
//!
//! See `CONTEXT.md` in the parent directory for the API contract.
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use super::entry::{FinalStatus, ModelUsage, PauseReason, ToolStatus};
use super::identity::ExecutionIdentity;
use super::Clock;

/// The stable kinds of execution events.
///
/// The variants are ordered as in the Python lifecycle: participation and start first, the
/// terminal frames last, so a caller can compare them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventType {
    /// A participation decision was reached.
    ParticipationDecided,
    /// The execution began with a goal.
    ExecutionStarted,
    /// The execution reported progress.
    ProgressUpdated,
    /// A fragment of the reply stream was produced.
    ReplyDelta,
    /// A model call completed.
    ModelCompleted,
    /// A tool call is about to run.
    ToolIntent,
    /// A tool call reached an outcome.
    ToolCompleted,
    /// The model window was replaced.
    WindowSwitched,
    /// A candidate output was proposed.
    OutputProposed,
    /// The execution paused without a final outcome.
    ExecutionPaused,
    /// The execution reached a terminal outcome.
    ExecutionFinal,
}

impl EventType {
    /// Returns the snake-case spelling of the kind, as it appears in `event_type`.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ParticipationDecided => "participation_decided",
            Self::ExecutionStarted => "execution_started",
            Self::ProgressUpdated => "progress_updated",
            Self::ReplyDelta => "reply_delta",
            Self::ModelCompleted => "model_completed",
            Self::ToolIntent => "tool_intent",
            Self::ToolCompleted => "tool_completed",
            Self::WindowSwitched => "window_switched",
            Self::OutputProposed => "output_proposed",
            Self::ExecutionPaused => "execution_paused",
            Self::ExecutionFinal => "execution_final",
        }
    }

    /// Returns whether failing to record an event of this kind fails the execution.
    ///
    /// Only tool intents and the pause/final frames are critical: they are the records a resumed
    /// execution relies on, so losing one is not recoverable.
    #[must_use]
    pub const fn is_critical(self) -> bool {
        matches!(
            self,
            Self::ToolIntent | Self::ExecutionPaused | Self::ExecutionFinal
        )
    }
}

/// One tool call a pause intercepted before it ran.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct InterceptedCall {
    /// Stable id of the intercepted tool call.
    pub tool_call_id: String,
    /// Name of the tool.
    pub tool_name: String,
    /// Redacted and bounded arguments.
    pub arguments: String,
    /// Whether `arguments` was shortened before it was recorded.
    pub arguments_truncated: bool,
    /// Opaque reference to the confirmation that released the call, when one exists.
    pub approval_ref: Option<String>,
}

/// The body of one execution event, tagged by `event_type`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "event_type", rename_all = "snake_case")]
pub enum PersistentEvent {
    /// The execution began with a goal.
    ExecutionStarted {
        /// Goal the execution works towards.
        goal: String,
        /// Event id of the participation decision that admitted it, for group executions.
        #[serde(default)]
        decision_event_id: Option<String>,
    },
    /// The execution reported progress.
    ProgressUpdated {
        /// Progress text.
        text: String,
    },
    /// A fragment of the reply stream was produced.
    ReplyDelta {
        /// Stream fragment; whitespace is preserved.
        text: String,
    },
    /// A model call completed.
    ModelCompleted {
        /// Stable id of the model call.
        model_call_id: String,
        /// Token usage of the call.
        usage: ModelUsage,
    },
    /// A tool call is about to run.
    ToolIntent {
        /// Opaque id of the intended side effect.
        operation_id: String,
        /// Stable id of the tool call.
        tool_call_id: String,
        /// Name of the tool.
        tool_name: String,
        /// Optional explicit target reference supplied by the model.
        #[serde(default)]
        target_ref: Option<String>,
        /// Digest of the canonical arguments, used to match retries.
        params_digest: String,
        /// Redacted and bounded arguments.
        args_summary: String,
    },
    /// A tool call reached an outcome.
    ToolCompleted {
        /// Stable id of the tool call.
        tool_call_id: String,
        /// Name of the tool.
        tool_name: String,
        /// Outcome of the call.
        status: ToolStatus,
        /// Redacted and bounded result.
        result_summary: String,
        /// Id of the matching intent, for calls that had one.
        #[serde(default)]
        operation_id: Option<String>,
        /// Why the call failed or its outcome is unknown.
        #[serde(default)]
        reason: Option<String>,
    },
    /// The model window was replaced by a new one.
    WindowSwitched {
        /// Reference to the archive of the replaced window.
        archive_ref: String,
        /// Sequence number of the new window.
        window_seq: u64,
    },
    /// The execution paused without a final outcome.
    ExecutionPaused {
        /// Why the execution paused.
        reason: PauseReason,
        /// Human-readable detail, when the reason needs one.
        #[serde(default)]
        detail: Option<String>,
        /// Tool calls the pause intercepted before they ran.
        #[serde(default)]
        intercepted: Vec<InterceptedCall>,
    },
    /// The execution reached a terminal outcome.
    ExecutionFinal {
        /// Terminal outcome.
        status: FinalStatus,
        /// Why the outcome is not a normal completion.
        #[serde(default)]
        reason: Option<String>,
        /// References to the outputs the execution produced.
        #[serde(default)]
        output_refs: Vec<String>,
    },
}

impl PersistentEvent {
    /// Returns the kind of this event.
    #[must_use]
    pub const fn event_type(&self) -> EventType {
        match self {
            Self::ExecutionStarted { .. } => EventType::ExecutionStarted,
            Self::ProgressUpdated { .. } => EventType::ProgressUpdated,
            Self::ReplyDelta { .. } => EventType::ReplyDelta,
            Self::ModelCompleted { .. } => EventType::ModelCompleted,
            Self::ToolIntent { .. } => EventType::ToolIntent,
            Self::ToolCompleted { .. } => EventType::ToolCompleted,
            Self::WindowSwitched { .. } => EventType::WindowSwitched,
            Self::ExecutionPaused { .. } => EventType::ExecutionPaused,
            Self::ExecutionFinal { .. } => EventType::ExecutionFinal,
        }
    }

    /// Returns whether losing this event is fatal to the execution.
    #[must_use]
    pub const fn is_critical(&self) -> bool {
        self.event_type().is_critical()
    }
}

/// One numbered, timestamped event handed to an [`EventSink`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmittedEvent {
    /// Execution the event belongs to.
    pub identity: ExecutionIdentity,
    /// 1-based sequence number within the execution.
    pub local_seq: u64,
    /// RFC 3339 UTC timestamp with millisecond precision.
    pub occurred_at: String,
    /// The event body.
    pub event: PersistentEvent,
    /// Stable id of this event; equals `identity.event_id(local_seq)`.
    pub event_id: String,
}

/// Persists the events of one execution.
///
/// Implementations record in call order; the [`Emitter`] serialises calls, so no sink ever sees two
/// events concurrently for the same execution.
#[async_trait]
pub trait EventSink: Send + Sync {
    /// Records one event.
    ///
    /// # Errors
    /// Any error the backing store reports; the emitter decides whether it is fatal.
    async fn record(&self, event: &EmittedEvent) -> Result<()>;
}

/// An [`EventSink`] that keeps the events in memory.
///
/// Useful for tests and for hosts that only need the events of the current process.
#[derive(Debug, Default)]
pub struct InMemorySink {
    events: Mutex<Vec<EmittedEvent>>,
}

impl InMemorySink {
    /// Creates an empty sink.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns a copy of the recorded events, in recording order.
    #[must_use]
    pub fn events(&self) -> Vec<EmittedEvent> {
        self.events.lock().clone()
    }

    /// Returns how many events were recorded.
    #[must_use]
    pub fn len(&self) -> usize {
        self.events.lock().len()
    }

    /// Returns whether no event was recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.events.lock().is_empty()
    }

    /// Drops every recorded event.
    pub fn clear(&self) {
        self.events.lock().clear();
    }
}

#[async_trait]
impl EventSink for InMemorySink {
    async fn record(&self, event: &EmittedEvent) -> Result<()> {
        self.events.lock().push(event.clone());
        Ok(())
    }
}

/// Numbers, timestamps and records the events of one execution.
pub struct Emitter {
    identity: ExecutionIdentity,
    sink: Arc<dyn EventSink>,
    clock: Arc<dyn Clock>,
    seq: AtomicU64,
    gate: tokio::sync::Mutex<()>,
}

impl fmt::Debug for Emitter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Emitter")
            .field("identity", &self.identity)
            .field("last_seq", &self.last_seq())
            .finish_non_exhaustive()
    }
}

impl Emitter {
    /// Creates an emitter for `identity` that records into `sink` and timestamps with `clock`.
    #[must_use]
    pub fn new(
        identity: ExecutionIdentity,
        sink: Arc<dyn EventSink>,
        clock: Arc<dyn Clock>,
    ) -> Self {
        Self {
            identity,
            sink,
            clock,
            seq: AtomicU64::new(0),
            gate: tokio::sync::Mutex::new(()),
        }
    }

    /// Records one event and returns it numbered.
    ///
    /// The sequence number is allocated before the sink call and is spent even when recording
    /// fails, so a failure leaves a gap but never a duplicate. A critical event that cannot be
    /// recorded fails the call; an ordinary one is logged and still returned.
    ///
    /// # Errors
    /// [`Error::Session`] whose message is
    /// `"critical record of <event_type> failed: <sink error>"` when the sink rejects a critical
    /// event.
    pub async fn emit(&self, event: PersistentEvent) -> Result<EmittedEvent> {
        let _guard = self.gate.lock().await;
        let emitted = self.number(event);
        let event_type = emitted.event.event_type();
        match self.sink.record(&emitted).await {
            Ok(()) => Ok(emitted),
            Err(error) if event_type.is_critical() => Err(Error::session(format!(
                "critical record of {} failed: {error}",
                event_type.as_str()
            ))),
            Err(error) => {
                tracing::warn!(
                    execution_id = %emitted.identity.execution_id,
                    local_seq = emitted.local_seq,
                    event_type = event_type.as_str(),
                    error = %error,
                    "ordinary persistent event record failed",
                );
                Ok(emitted)
            }
        }
    }

    /// Records a pause or final event, retrying it once as the same event.
    ///
    /// Both attempts carry the same sequence number and event id, so a host that received the
    /// first (failed) attempt deduplicates by id. The last error is returned when both attempts
    /// fail.
    ///
    /// # Errors
    /// The sink error of the second attempt.
    pub async fn emit_ending(&self, event: PersistentEvent) -> Result<EmittedEvent> {
        let _guard = self.gate.lock().await;
        let emitted = self.number(event);
        let event_type = emitted.event.event_type();
        if let Err(error) = self.sink.record(&emitted).await {
            tracing::warn!(
                execution_id = %emitted.identity.execution_id,
                local_seq = emitted.local_seq,
                event_type = event_type.as_str(),
                error = %error,
                "ending record failed, retrying once",
            );
            self.sink.record(&emitted).await?;
        }
        Ok(emitted)
    }

    /// Returns the sequence number of the last allocated event (`0` before the first emission).
    #[must_use]
    pub fn last_seq(&self) -> u64 {
        self.seq.load(Ordering::SeqCst)
    }

    /// Returns the identity of the execution this emitter numbers events for.
    #[must_use]
    pub fn identity(&self) -> &ExecutionIdentity {
        &self.identity
    }

    /// Returns the sink the events are recorded into.
    #[must_use]
    pub fn sink(&self) -> Arc<dyn EventSink> {
        Arc::clone(&self.sink)
    }

    /// Numbers and timestamps `event` without recording it.
    fn number(&self, event: PersistentEvent) -> EmittedEvent {
        let local_seq = self.seq.fetch_add(1, Ordering::SeqCst) + 1;
        EmittedEvent {
            identity: self.identity.clone(),
            local_seq,
            occurred_at: self.clock.now_rfc3339(),
            event_id: self.identity.event_id(local_seq),
            event,
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    use async_trait::async_trait;
    use openjiuwen_common::error::{Error, Result};
    use parking_lot::Mutex;

    use super::{
        EmittedEvent, Emitter, EventSink, EventType, InMemorySink, InterceptedCall, PersistentEvent,
    };
    use crate::persistent_agent::entry::{FinalStatus, ModelUsage, PauseReason, ToolStatus};
    use crate::persistent_agent::identity::ExecutionIdentity;
    use crate::persistent_agent::{Clock, ManualClock};

    /// A sink that fails a configurable number of attempts and records the successes.
    #[derive(Debug, Default)]
    struct FlakySink {
        attempts: AtomicUsize,
        failures_left: AtomicUsize,
        recorded: Mutex<Vec<EmittedEvent>>,
    }

    impl FlakySink {
        fn with_failures(failures: usize) -> Self {
            Self {
                attempts: AtomicUsize::new(0),
                failures_left: AtomicUsize::new(failures),
                recorded: Mutex::new(Vec::new()),
            }
        }

        fn attempts(&self) -> usize {
            self.attempts.load(Ordering::SeqCst)
        }

        fn recorded(&self) -> Vec<EmittedEvent> {
            self.recorded.lock().clone()
        }
    }

    #[async_trait]
    impl EventSink for FlakySink {
        async fn record(&self, event: &EmittedEvent) -> Result<()> {
            self.attempts.fetch_add(1, Ordering::SeqCst);
            // The emitter serialises sink calls, so a load/store pair is enough here.
            let failures_left = self.failures_left.load(Ordering::SeqCst);
            if failures_left > 0 {
                self.failures_left
                    .store(failures_left - 1, Ordering::SeqCst);
                return Err(Error::session("sink offline"));
            }
            self.recorded.lock().push(event.clone());
            Ok(())
        }
    }

    fn identity() -> ExecutionIdentity {
        ExecutionIdentity::new("ns", "agent", "session", "request", "execution", "source").unwrap()
    }

    fn emitter(sink: Arc<dyn EventSink>) -> (Emitter, Arc<ManualClock>) {
        let clock = Arc::new(ManualClock::new(1_700_000_000_000));
        let emitter = Emitter::new(identity(), sink, Arc::clone(&clock) as Arc<dyn Clock>);
        (emitter, clock)
    }

    fn progress(text: &str) -> PersistentEvent {
        PersistentEvent::ProgressUpdated {
            text: text.to_owned(),
        }
    }

    #[test]
    fn event_types_use_snake_case_and_flag_the_critical_three() {
        assert_eq!(
            serde_json::to_string(&EventType::ExecutionStarted).unwrap(),
            "\"execution_started\""
        );
        assert_eq!(
            serde_json::to_string(&EventType::ParticipationDecided).unwrap(),
            "\"participation_decided\""
        );
        assert_eq!(EventType::ToolIntent.as_str(), "tool_intent");
        assert_eq!(EventType::ExecutionFinal.as_str(), "execution_final");
        let decoded: EventType = serde_json::from_str("\"window_switched\"").unwrap();
        assert_eq!(decoded, EventType::WindowSwitched);
        assert!(EventType::ToolIntent.is_critical());
        assert!(EventType::ExecutionPaused.is_critical());
        assert!(EventType::ExecutionFinal.is_critical());
        for event_type in [
            EventType::ParticipationDecided,
            EventType::ExecutionStarted,
            EventType::ProgressUpdated,
            EventType::ReplyDelta,
            EventType::ModelCompleted,
            EventType::ToolCompleted,
            EventType::WindowSwitched,
            EventType::OutputProposed,
        ] {
            assert!(
                !event_type.is_critical(),
                "{event_type:?} must not be critical"
            );
        }
        assert!(EventType::ExecutionStarted < EventType::ExecutionFinal);
        assert!(!progress("x").is_critical());
        assert!(PersistentEvent::ExecutionFinal {
            status: FinalStatus::Completed,
            reason: None,
            output_refs: Vec::new(),
        }
        .is_critical());
    }

    #[tokio::test]
    async fn emissions_are_numbered_in_order_with_injected_timestamps() {
        let sink = Arc::new(InMemorySink::new());
        let (emitter, clock) = emitter(Arc::clone(&sink) as Arc<dyn EventSink>);
        assert_eq!(emitter.last_seq(), 0);
        assert_eq!(emitter.identity(), &identity());

        let first = emitter.emit(progress("one")).await.unwrap();
        assert_eq!(first.local_seq, 1);
        assert_eq!(first.occurred_at, "2023-11-14T22:13:20.000Z");
        assert_eq!(first.event_id, identity().event_id(1));
        assert_eq!(first.identity, identity());

        clock.advance(1_500);
        let second = emitter.emit(progress("two")).await.unwrap();
        assert_eq!(second.local_seq, 2);
        assert_eq!(second.occurred_at, "2023-11-14T22:13:21.500Z");
        // Stable across a rebuild: the same sequence number yields the same id.
        assert_eq!(second.event_id, identity().event_id(2));
        assert_ne!(first.event_id, second.event_id);

        let third = emitter.emit(progress("three")).await.unwrap();
        assert_eq!(third.local_seq, 3);
        assert_ne!(third.event_id, first.event_id);
        assert_eq!(emitter.last_seq(), 3);

        let recorded = sink.events();
        assert_eq!(recorded.len(), 3);
        assert_eq!(
            recorded
                .iter()
                .map(|event| event.local_seq)
                .collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
        assert_eq!(recorded[2], third);
    }

    #[tokio::test]
    async fn in_memory_sink_collects_and_clears() {
        let sink = Arc::new(InMemorySink::new());
        assert!(sink.is_empty());
        assert_eq!(sink.len(), 0);
        let (emitter, _clock) = emitter(Arc::clone(&sink) as Arc<dyn EventSink>);
        let first = emitter.emit(progress("hello")).await.unwrap();
        assert_eq!(sink.len(), 1);
        assert!(!sink.is_empty());
        assert_eq!(sink.events(), vec![first]);
        sink.clear();
        assert_eq!(sink.len(), 0);
        assert_eq!(sink.events(), Vec::new());
    }

    #[tokio::test]
    async fn an_ordinary_sink_failure_returns_ok_and_spends_a_sequence_number() {
        let sink = Arc::new(FlakySink::with_failures(1));
        let (emitter, _clock) = emitter(Arc::clone(&sink) as Arc<dyn EventSink>);

        let failed = emitter.emit(progress("lost")).await.unwrap();
        assert_eq!(failed.local_seq, 1);
        assert_eq!(sink.attempts(), 1);
        assert_eq!(sink.recorded(), Vec::new());
        assert_eq!(emitter.last_seq(), 1);

        let second = emitter.emit(progress("kept")).await.unwrap();
        assert_eq!(second.local_seq, 2);
        assert_eq!(emitter.last_seq(), 2);
        let recorded = sink.recorded();
        assert_eq!(recorded.len(), 1);
        assert_eq!(recorded[0].local_seq, 2);
        assert_eq!(recorded[0].event_id, identity().event_id(2));
    }

    #[tokio::test]
    async fn a_critical_sink_failure_reports_the_exact_session_error() {
        let sink = Arc::new(FlakySink::with_failures(2));
        let (emitter, _clock) = emitter(Arc::clone(&sink) as Arc<dyn EventSink>);

        let error = emitter
            .emit(PersistentEvent::ExecutionFinal {
                status: FinalStatus::Failed,
                reason: Some("boom".to_owned()),
                output_refs: Vec::new(),
            })
            .await
            .unwrap_err();
        assert!(matches!(error, Error::Session(_)));
        assert_eq!(
            error.to_string(),
            "session: critical record of execution_final failed: session: sink offline"
        );
        assert_eq!(emitter.last_seq(), 1);
        assert_eq!(sink.attempts(), 1);

        // A refused tool intent is critical as well, and the sequence number is still spent.
        let error = emitter
            .emit(PersistentEvent::ToolIntent {
                operation_id: "op_1".to_owned(),
                tool_call_id: "call_1".to_owned(),
                tool_name: "bash".to_owned(),
                target_ref: None,
                params_digest: "digest".to_owned(),
                args_summary: "{}".to_owned(),
            })
            .await
            .unwrap_err();
        let message = error.to_string();
        assert!(message.contains("critical record of tool_intent failed"));
        assert!(message.contains("sink offline"));
        assert_eq!(emitter.last_seq(), 2);
        assert_eq!(sink.attempts(), 2);
    }

    #[tokio::test]
    async fn emit_ending_retries_a_transient_failure_as_the_same_event() {
        let sink = Arc::new(FlakySink::with_failures(1));
        let (emitter, _clock) = emitter(Arc::clone(&sink) as Arc<dyn EventSink>);

        let ending = emitter
            .emit_ending(PersistentEvent::ExecutionPaused {
                reason: PauseReason::AwaitingConfirmation,
                detail: None,
                intercepted: Vec::new(),
            })
            .await
            .unwrap();
        assert_eq!(ending.local_seq, 1);
        assert_eq!(ending.event_id, identity().event_id(1));
        assert_eq!(sink.attempts(), 2);
        assert_eq!(sink.recorded(), vec![ending.clone()]);
        assert_eq!(emitter.last_seq(), 1);
    }

    #[tokio::test]
    async fn emit_ending_returns_the_last_error_when_the_retry_fails_too() {
        let sink = Arc::new(FlakySink::with_failures(2));
        let (emitter, _clock) = emitter(Arc::clone(&sink) as Arc<dyn EventSink>);

        let error = emitter
            .emit_ending(PersistentEvent::ExecutionFinal {
                status: FinalStatus::Completed,
                reason: None,
                output_refs: vec!["out_1".to_owned()],
            })
            .await
            .unwrap_err();
        assert!(matches!(error, Error::Session(_)));
        assert_eq!(error.to_string(), "session: sink offline");
        assert_eq!(sink.attempts(), 2);
        assert_eq!(sink.recorded(), Vec::new());
        assert_eq!(emitter.last_seq(), 1);
    }

    #[test]
    fn every_event_variant_round_trips() {
        let events = vec![
            PersistentEvent::ExecutionStarted {
                goal: "fix the bug".to_owned(),
                decision_event_id: Some("evt_1".to_owned()),
            },
            progress("half way"),
            PersistentEvent::ReplyDelta {
                text: "  ".to_owned(),
            },
            PersistentEvent::ModelCompleted {
                model_call_id: "mc_1".to_owned(),
                usage: ModelUsage::known(11, 7),
            },
            PersistentEvent::ToolIntent {
                operation_id: "op_1".to_owned(),
                tool_call_id: "call_1".to_owned(),
                tool_name: "bash".to_owned(),
                target_ref: Some("repo".to_owned()),
                params_digest: "digest".to_owned(),
                args_summary: "{\"command\":\"ls\"}".to_owned(),
            },
            PersistentEvent::ToolCompleted {
                tool_call_id: "call_1".to_owned(),
                tool_name: "bash".to_owned(),
                status: ToolStatus::Failed,
                result_summary: "exit 1".to_owned(),
                operation_id: Some("op_1".to_owned()),
                reason: Some("non-zero exit".to_owned()),
            },
            PersistentEvent::WindowSwitched {
                archive_ref: "pa:agent:session:archive:1".to_owned(),
                window_seq: 2,
            },
            PersistentEvent::ExecutionPaused {
                reason: PauseReason::AwaitingConfirmation,
                detail: Some("confirm bash".to_owned()),
                intercepted: vec![InterceptedCall {
                    tool_call_id: "call_1".to_owned(),
                    tool_name: "bash".to_owned(),
                    arguments: "{\"command\":\"rm -rf /\"}".to_owned(),
                    arguments_truncated: false,
                    approval_ref: Some("approval_1".to_owned()),
                }],
            },
            PersistentEvent::ExecutionFinal {
                status: FinalStatus::Cancelled,
                reason: Some("initiator cancelled".to_owned()),
                output_refs: vec!["out_1".to_owned(), "out_2".to_owned()],
            },
        ];
        for event in events {
            let value = serde_json::to_value(&event).unwrap();
            let tag = value["event_type"].as_str().unwrap().to_owned();
            assert_eq!(tag, event.event_type().as_str());
            let decoded: PersistentEvent = serde_json::from_value(value).unwrap();
            assert_eq!(decoded, event);
        }
    }

    #[test]
    fn events_tolerate_missing_optional_fields_and_unknown_ones() {
        let started: PersistentEvent = serde_json::from_value(serde_json::json!({
            "event_type": "execution_started",
            "goal": "do it",
            "future_field": true,
        }))
        .unwrap();
        assert_eq!(
            started,
            PersistentEvent::ExecutionStarted {
                goal: "do it".to_owned(),
                decision_event_id: None,
            }
        );

        let paused: PersistentEvent = serde_json::from_value(serde_json::json!({
            "event_type": "execution_paused",
            "reason": "stopped",
            "intercepted": [{"tool_call_id": "call_1"}],
        }))
        .unwrap();
        let PersistentEvent::ExecutionPaused {
            reason,
            detail,
            intercepted,
        } = paused
        else {
            panic!("expected a paused event");
        };
        assert_eq!(reason, PauseReason::Stopped);
        assert_eq!(detail, None);
        assert_eq!(
            intercepted,
            vec![InterceptedCall {
                tool_call_id: "call_1".to_owned(),
                tool_name: String::new(),
                arguments: String::new(),
                arguments_truncated: false,
                approval_ref: None,
            }]
        );

        let final_event: PersistentEvent = serde_json::from_value(serde_json::json!({
            "event_type": "execution_final",
            "status": "completed",
        }))
        .unwrap();
        assert_eq!(
            final_event,
            PersistentEvent::ExecutionFinal {
                status: FinalStatus::Completed,
                reason: None,
                output_refs: Vec::new(),
            }
        );
    }

    #[test]
    fn emitted_events_round_trip() {
        let emitted = EmittedEvent {
            identity: identity(),
            local_seq: 7,
            occurred_at: "2023-11-14T22:13:20.000Z".to_owned(),
            event: progress("done"),
            event_id: identity().event_id(7),
        };
        let encoded = serde_json::to_string(&emitted).unwrap();
        let decoded: EmittedEvent = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, emitted);
        assert_eq!(emitted.event_id, identity().event_id(emitted.local_seq));
    }
}
