//! Observability seam for team memory: a [`TeamMemoryObserver`] trait plus no-op and `tracing` implementations.
//!
//! This is an *observer* seam, not an exporter. A [`TeamMemoryEvent`] describes one
//! operation ([`TeamMemoryOperation`]) together with the team, member, and
//! [`kind`](crate::team::model::TeamMemoryKind) it touched, how many records it moved, and
//! whether it [`succeeded`](TeamMemoryOutcome::Ok) or [`failed`](TeamMemoryOutcome::Failed);
//! a [`TeamMemoryObserver`] receives that event synchronously and does whatever its owner
//! wants with it — a metric, a span, a log line, or a test recorder. Nothing in this module
//! buffers events, spawns a task, or opens a socket.
//!
//! # Opt-in, and free when absent
//!
//! A [`TeamMemory`](crate::team::TeamMemory) carries `Option<Arc<dyn TeamMemoryObserver>>`
//! that starts as `None`, so the default cost of a team memory operation is one `Option`
//! check and the facade's observable behaviour is unchanged; attaching an observer is
//! purely additive. A caller opts in with
//! [`TeamMemory::with_observer`](crate::team::TeamMemory::with_observer) and attaches either
//! [`TracingObserver`] or its own implementation.
//!
//! # Who attaches what
//!
//! The owner of the facade decides what observability means: the agent-team runtime, the
//! `openjiuwen-team-mcp` server, and the `DeepAgent` harness all build a
//! [`TeamMemory`](crate::team::TeamMemory), so each of them may attach [`TracingObserver`]
//! (one structured `tracing` event per operation at target `openjiuwen_memory::team`) or a
//! custom observer that bridges into its own telemetry. The real OTel/OTLP exporter lives in
//! `openjiuwen-extensions` and is out of scope for this crate: this seam is the point such an
//! adapter would hook into, and the adapter — not this module — owns the transport, sampling,
//! and batching.
//!
//! # Covered operations
//!
//! The facade emits for the operations that change what the team can recall or that spend
//! real work on a query: `Write` ([`TeamMemory::remember`](crate::team::TeamMemory::remember),
//! [`remember_many`](crate::team::TeamMemory::remember_many),
//! [`update`](crate::team::TeamMemory::update)), `Search`, `Forget`, `Clear`, and `Prune`.
//! [`TeamMemoryOperation::Extract`] is reserved for the extractor pipeline. Direct reads and
//! bookkeeping (`get`, `list`, `stats`, `set_expiry`, `to_jsonl`, `from_jsonl`) deliberately
//! emit nothing: they are how a caller inspects or exports the store, and making them
//! observable would let a monitoring observer drive its own traffic.
//!
//! # Failures are reported, never swallowed
//!
//! A failed operation emits exactly one event with [`TeamMemoryOutcome::Failed`] and the
//! error's `Display` text in [`TeamMemoryEvent::error`], and the facade still returns that
//! same typed error to the caller: observation runs beside the result path, not instead of
//! it.
use crate::team::model::TeamMemoryKind;
use serde::{Deserialize, Serialize};
use std::fmt;
use tracing::{debug, warn};
/// The tracing target of every log line this module emits.
const LOG_TARGET: &str = "openjiuwen_memory::team";
/// The team memory operation an event describes.
///
/// The set is deliberately coarse: one variant per facade entry point whose cost or effect
/// is worth reporting. [`TeamMemoryOperation::Extract`] belongs to the extractor pipeline and
/// is never emitted by the facade.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TeamMemoryOperation {
    /// A recall query against the store.
    Search,
    /// A record (or batch of records) written or updated.
    Write,
    /// One record deleted.
    Forget,
    /// Every record matching a filter deleted.
    Clear,
    /// Expired records removed.
    Prune,
    /// Memories extracted from a conversation.
    Extract,
}
impl TeamMemoryOperation {
    /// The `snake_case` string of this operation, used in JSON and as a `tracing` field.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Search => "search",
            Self::Write => "write",
            Self::Forget => "forget",
            Self::Clear => "clear",
            Self::Prune => "prune",
            Self::Extract => "extract",
        }
    }
}
/// How an observed operation ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TeamMemoryOutcome {
    /// The operation completed; see the event's count for how much it touched.
    #[default]
    Ok,
    /// The operation returned an error; the event carries its message.
    Failed,
}
impl TeamMemoryOutcome {
    /// The `snake_case` string of this outcome, used in JSON and as a `tracing` field.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Ok => "ok",
            Self::Failed => "failed",
        }
    }
}
/// One observable team memory operation.
///
/// Build it with [`TeamMemoryEvent::new`], refine it with the `with_*` builders, and mark a
/// failure with [`TeamMemoryEvent::failed`]. The JSON shape is flat `snake_case`, absent
/// optionals are omitted, and unknown fields are tolerated on input.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct TeamMemoryEvent {
    /// The operation that was attempted.
    pub operation: TeamMemoryOperation,
    /// The team the operation ran for.
    pub team_id: String,
    /// The member whose view or record the operation touched, when member-scoped.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub member_id: Option<String>,
    /// The kind of the record involved, when the operation has a single kind.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<TeamMemoryKind>,
    /// How many records the operation touched (`0` for a failed or empty operation).
    #[serde(default)]
    pub count: usize,
    /// Whether the operation succeeded or failed.
    #[serde(default)]
    pub outcome: TeamMemoryOutcome,
    /// The error text of a failed operation; `None` on success.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}
impl TeamMemoryEvent {
    /// A successful, zero-count event for `operation` on `team_id`.
    #[must_use]
    pub fn new(operation: TeamMemoryOperation, team_id: impl Into<String>) -> Self {
        Self {
            operation,
            team_id: team_id.into(),
            member_id: None,
            kind: None,
            count: 0,
            outcome: TeamMemoryOutcome::Ok,
            error: None,
        }
    }
    /// Returns the same event bound to `member_id`.
    #[must_use]
    pub fn with_member(mut self, member_id: impl Into<String>) -> Self {
        self.member_id = Some(member_id.into());
        self
    }
    /// Returns the same event bound to `kind`.
    #[must_use]
    pub fn with_kind(mut self, kind: TeamMemoryKind) -> Self {
        self.kind = Some(kind);
        self
    }
    /// Returns the same event with `count` records recorded.
    #[must_use]
    pub fn with_count(mut self, count: usize) -> Self {
        self.count = count;
        self
    }
    /// Returns the same event marked as failed, carrying `error`'s text.
    #[must_use]
    pub fn failed(mut self, error: impl fmt::Display) -> Self {
        self.outcome = TeamMemoryOutcome::Failed;
        self.error = Some(error.to_string());
        self
    }
    /// `true` when the observed operation failed.
    #[must_use]
    pub fn is_failure(&self) -> bool {
        self.outcome == TeamMemoryOutcome::Failed
    }
}
/// Receives every event a [`TeamMemory`](crate::team::TeamMemory) emits.
///
/// Implementations are synchronous and must not block: the facade calls [`Self::observe`]
/// inline after (or on the error path around) each store call, so an observer that blocks
/// delays the caller. Send work to a channel or a counter if that is what you need.
pub trait TeamMemoryObserver: Send + Sync {
    /// Handles one completed (or failed) operation.
    fn observe(&self, event: &TeamMemoryEvent);
}
/// An observer that discards every event; the explicit form of "no observer".
#[derive(Debug, Default, Clone, Copy)]
pub struct NoopObserver;
impl TeamMemoryObserver for NoopObserver {
    fn observe(&self, _event: &TeamMemoryEvent) {}
}
/// An observer that logs every event as one structured `tracing` event.
///
/// Successes log at `debug!` and [`TeamMemoryOutcome::Failed`] at `warn!`, both at target
/// `openjiuwen_memory::team` with the `operation`, `team_id`, `member_id`, `kind`, `count`,
/// and `outcome` fields (plus `error` on failure). This is the zero-configuration way to make
/// team memory visible without taking on an exporter.
#[derive(Debug, Default, Clone, Copy)]
pub struct TracingObserver;
impl TeamMemoryObserver for TracingObserver {
    fn observe(&self, event: &TeamMemoryEvent) {
        let operation = event.operation.as_str();
        let team_id = event.team_id.as_str();
        let member_id = event.member_id.as_deref().unwrap_or("");
        let kind = event.kind.as_ref().map_or("", TeamMemoryKind::as_str);
        let count = event.count;
        let outcome = event.outcome.as_str();
        match event.outcome {
            TeamMemoryOutcome::Ok => debug!(
                target: LOG_TARGET,
                operation, team_id, member_id, kind, count, outcome, "team memory operation"
            ),
            TeamMemoryOutcome::Failed => warn!(
                target: LOG_TARGET,
                operation,
                team_id,
                member_id,
                kind,
                count,
                outcome,
                error = event.error.as_deref().unwrap_or(""),
                "team memory operation failed"
            ),
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::CallOptions;
    use crate::team::model::{NewTeamMemory, TeamMemoryFilter, TeamMemoryScope, TeamMemoryUpdate};
    use crate::team::store::{TeamMemory, TeamSearchOptions};
    use openjiuwen_common::error::Error;
    use parking_lot::Mutex;
    use serde_json::json;
    use std::sync::Arc;
    use tokio_util::sync::CancellationToken;
    /// An observer that keeps every event it sees.
    #[derive(Debug, Default)]
    struct RecordingObserver {
        events: Mutex<Vec<TeamMemoryEvent>>,
    }
    impl RecordingObserver {
        fn events(&self) -> Vec<TeamMemoryEvent> {
            self.events.lock().clone()
        }
        fn len(&self) -> usize {
            self.events.lock().len()
        }
        fn only(&self) -> TeamMemoryEvent {
            let events = self.events();
            assert_eq!(events.len(), 1, "expected a single event: {events:?}");
            events.into_iter().next().expect("checked len == 1")
        }
    }
    impl TeamMemoryObserver for RecordingObserver {
        fn observe(&self, event: &TeamMemoryEvent) {
            self.events.lock().push(event.clone());
        }
    }
    fn recording_facade() -> (TeamMemory, Arc<RecordingObserver>) {
        let observer = Arc::new(RecordingObserver::default());
        let memory = TeamMemory::in_memory().with_observer(observer.clone());
        (memory, observer)
    }
    fn decision(team: &str, content: &str) -> NewTeamMemory {
        NewTeamMemory::decision(team, content).with_member("alice")
    }
    fn cancelled_call() -> CallOptions {
        let token = CancellationToken::new();
        token.cancel();
        CallOptions::new().with_cancellation(token)
    }
    #[test]
    fn event_builders_fill_defaults_and_overrides() {
        let event = TeamMemoryEvent::new(TeamMemoryOperation::Search, "review-team");
        assert_eq!(event.team_id, "review-team");
        assert_eq!(event.count, 0);
        assert_eq!(event.outcome, TeamMemoryOutcome::Ok);
        assert!(event.member_id.is_none());
        assert!(event.kind.is_none());
        assert!(event.error.is_none());
        assert!(!event.is_failure());
        let event = TeamMemoryEvent::new(TeamMemoryOperation::Write, "review-team")
            .with_member("alice")
            .with_kind(TeamMemoryKind::Decision)
            .with_count(3);
        assert_eq!(event.member_id.as_deref(), Some("alice"));
        assert_eq!(event.kind, Some(TeamMemoryKind::Decision));
        assert_eq!(event.count, 3);
        assert!(!event.is_failure());
    }
    #[test]
    fn failed_sets_the_outcome_and_the_error_text() {
        let event = TeamMemoryEvent::new(TeamMemoryOperation::Write, "review-team")
            .with_member("alice")
            .with_count(2)
            .failed("boom");
        assert_eq!(event.outcome, TeamMemoryOutcome::Failed);
        assert_eq!(event.error.as_deref(), Some("boom"));
        assert!(event.is_failure());
        assert_eq!(event.count, 2);
        let from_error = TeamMemoryEvent::new(TeamMemoryOperation::Search, "review-team")
            .failed(Error::Cancelled);
        assert_eq!(from_error.error.as_deref(), Some("cancelled"));
    }
    #[test]
    fn operation_and_outcome_render_their_snake_case_names() {
        assert_eq!(TeamMemoryOperation::Search.as_str(), "search");
        assert_eq!(TeamMemoryOperation::Write.as_str(), "write");
        assert_eq!(TeamMemoryOperation::Forget.as_str(), "forget");
        assert_eq!(TeamMemoryOperation::Clear.as_str(), "clear");
        assert_eq!(TeamMemoryOperation::Prune.as_str(), "prune");
        assert_eq!(TeamMemoryOperation::Extract.as_str(), "extract");
        assert_eq!(TeamMemoryOutcome::Ok.as_str(), "ok");
        assert_eq!(TeamMemoryOutcome::Failed.as_str(), "failed");
    }
    #[test]
    fn enums_round_trip_as_snake_case_strings() {
        for operation in [
            TeamMemoryOperation::Search,
            TeamMemoryOperation::Write,
            TeamMemoryOperation::Forget,
            TeamMemoryOperation::Clear,
            TeamMemoryOperation::Prune,
            TeamMemoryOperation::Extract,
        ] {
            let value = serde_json::to_value(operation).expect("operation serializes");
            assert_eq!(value, json!(operation.as_str()));
            let parsed: TeamMemoryOperation =
                serde_json::from_value(value).expect("operation deserializes");
            assert_eq!(parsed, operation);
        }
        assert_eq!(
            serde_json::to_value(TeamMemoryOutcome::Ok).expect("outcome serializes"),
            json!("ok")
        );
        assert_eq!(
            serde_json::to_value(TeamMemoryOutcome::Failed).expect("outcome serializes"),
            json!("failed")
        );
        let parsed: TeamMemoryOutcome =
            serde_json::from_value(json!("failed")).expect("outcome deserializes");
        assert_eq!(parsed, TeamMemoryOutcome::Failed);
    }
    #[test]
    fn event_json_carries_the_observable_shape() {
        let event = TeamMemoryEvent::new(TeamMemoryOperation::Write, "review-team")
            .with_kind(TeamMemoryKind::Decision)
            .with_count(2);
        let value = serde_json::to_value(&event).expect("event serializes");
        assert_eq!(value["operation"], json!("write"));
        assert_eq!(value["team_id"], json!("review-team"));
        assert_eq!(value["kind"], json!("decision"));
        assert_eq!(value["count"], json!(2));
        assert_eq!(value["outcome"], json!("ok"));
        assert!(value.get("error").is_none());
        assert!(value.get("member_id").is_none());
        let parsed: TeamMemoryEvent = serde_json::from_value(value).expect("event deserializes");
        assert_eq!(parsed, event);
        let failed = TeamMemoryEvent::new(TeamMemoryOperation::Clear, "review-team")
            .with_member("bob")
            .failed("boom");
        let value = serde_json::to_value(&failed).expect("event serializes");
        assert_eq!(value["member_id"], json!("bob"));
        assert_eq!(value["outcome"], json!("failed"));
        assert_eq!(value["error"], json!("boom"));
    }
    #[test]
    fn facade_and_observers_are_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<TeamMemory>();
        assert_send_sync::<TeamMemoryEvent>();
        assert_send_sync::<NoopObserver>();
        assert_send_sync::<TracingObserver>();
    }
    #[tokio::test]
    async fn write_search_forget_clear_and_prune_are_observed() {
        let (memory, observer) = recording_facade();
        let call = CallOptions::new();
        let first = memory
            .remember(
                decision("review-team", "sqlite for the shared store"),
                &call,
            )
            .await
            .expect("remember succeeds");
        let write = observer.only();
        assert_eq!(write.operation, TeamMemoryOperation::Write);
        assert_eq!(write.team_id, "review-team");
        assert_eq!(write.member_id.as_deref(), Some("alice"));
        assert_eq!(write.kind, Some(TeamMemoryKind::Decision));
        assert_eq!(write.count, 1);
        assert_eq!(write.outcome, TeamMemoryOutcome::Ok);
        memory
            .remember(decision("review-team", "ship on friday"), &call)
            .await
            .expect("remember succeeds");
        let hits = memory
            .search(
                &TeamMemoryScope::member("review-team", "alice"),
                "sqlite",
                &TeamSearchOptions::default(),
            )
            .await
            .expect("search succeeds");
        let search = observer.events().pop().expect("a search event");
        assert_eq!(search.operation, TeamMemoryOperation::Search);
        assert_eq!(search.team_id, "review-team");
        assert_eq!(search.member_id.as_deref(), Some("alice"));
        assert_eq!(search.count, hits.len());
        assert!(search.kind.is_none());
        assert_eq!(search.outcome, TeamMemoryOutcome::Ok);
        let update = TeamMemoryUpdate::default().content("sqlite for the shared store, really");
        memory
            .update("review-team", &first.id, &update, &call)
            .await
            .expect("update succeeds");
        let updated = observer.events().pop().expect("an update event");
        assert_eq!(updated.operation, TeamMemoryOperation::Write);
        assert_eq!(updated.count, 1);
        assert_eq!(updated.outcome, TeamMemoryOutcome::Ok);
        assert!(memory
            .forget("review-team", &first.id)
            .await
            .expect("forget succeeds"));
        let forgot = observer.events().pop().expect("a forget event");
        assert_eq!(forgot.operation, TeamMemoryOperation::Forget);
        assert_eq!(forgot.count, 1);
        assert!(!memory
            .forget("review-team", &first.id)
            .await
            .expect("forget succeeds"));
        let missing = observer.events().pop().expect("a forget event");
        assert_eq!(missing.operation, TeamMemoryOperation::Forget);
        assert_eq!(missing.count, 0);
        let cleared = memory
            .clear(&TeamMemoryFilter::for_team("review-team"))
            .await
            .expect("clear succeeds");
        let clear = observer.events().pop().expect("a clear event");
        assert_eq!(clear.operation, TeamMemoryOperation::Clear);
        assert_eq!(clear.count, cleared);
        let past = chrono::Utc::now() - chrono::Duration::seconds(1);
        memory
            .remember(decision("review-team", "expire me"), &call)
            .await
            .expect("remember succeeds");
        let record = memory
            .list(&TeamMemoryFilter::for_team("review-team"))
            .await
            .expect("list succeeds")
            .remove(0);
        memory
            .set_expiry("review-team", &record.id, Some(past))
            .await
            .expect("set_expiry succeeds");
        let pruned = memory.prune_expired().await.expect("prune succeeds");
        let prune = observer.events().pop().expect("a prune event");
        assert_eq!(prune.operation, TeamMemoryOperation::Prune);
        assert_eq!(prune.count, pruned);
        // get/list/stats/set_expiry/to_jsonl emit nothing.
        assert_eq!(observer.len(), 9);
    }
    #[tokio::test]
    async fn remember_many_emits_one_write_with_the_stored_count() {
        let (memory, observer) = recording_facade();
        let stored = memory
            .remember_many(
                vec![
                    decision("review-team", "first"),
                    decision("review-team", "second"),
                ],
                &CallOptions::new(),
            )
            .await
            .expect("remember_many succeeds");
        assert_eq!(stored.len(), 2);
        let event = observer.only();
        assert_eq!(event.operation, TeamMemoryOperation::Write);
        assert_eq!(event.team_id, "review-team");
        assert_eq!(event.count, 2);
        assert_eq!(event.outcome, TeamMemoryOutcome::Ok);
        assert!(event.kind.is_none());
        assert!(event.member_id.is_none());
    }
    #[tokio::test]
    async fn a_failed_operation_emits_and_still_returns_the_same_error() {
        let (memory, observer) = recording_facade();
        let error = memory
            .remember(decision("review-team", "content"), &cancelled_call())
            .await
            .expect_err("a cancelled call fails");
        assert!(matches!(error, Error::Cancelled));
        let event = observer.only();
        assert_eq!(event.operation, TeamMemoryOperation::Write);
        assert_eq!(event.team_id, "review-team");
        assert_eq!(event.outcome, TeamMemoryOutcome::Failed);
        assert_eq!(event.error.as_deref(), Some("cancelled"));
        assert_eq!(event.count, 0);
        let invalid = NewTeamMemory::decision("review-team", "   ").with_member("alice");
        let error = memory
            .remember(invalid, &CallOptions::new())
            .await
            .expect_err("empty content is rejected");
        assert!(matches!(error, Error::Validation(_)));
        let expected = error.to_string();
        assert!(expected.contains("content"), "{expected}");
        let event = observer.events().pop().expect("a failure event");
        assert!(event.is_failure());
        assert_eq!(event.error.as_deref(), Some(expected.as_str()));
        // A batch that stops part way reports the records it did store.
        let mut bad = decision("review-team", "third");
        bad.content = String::new();
        let error = memory
            .remember_many(
                vec![decision("review-team", "batch first"), bad],
                &CallOptions::new(),
            )
            .await
            .expect_err("a bad input stops the batch");
        assert!(matches!(error, Error::Validation(_)));
        let event = observer.events().pop().expect("a failure event");
        assert_eq!(event.operation, TeamMemoryOperation::Write);
        assert!(event.is_failure());
        assert_eq!(event.count, 1);
    }
    #[tokio::test]
    async fn an_observer_is_optional_and_any_implementation_works() {
        let memory = TeamMemory::in_memory();
        assert!(memory.observer().is_none());
        memory.emit(&TeamMemoryEvent::new(
            TeamMemoryOperation::Search,
            "review-team",
        ));
        let record = memory
            .remember(decision("review-team", "content"), &CallOptions::new())
            .await
            .expect("remember succeeds without an observer");
        assert_eq!(record.team_id, "review-team");
        let memory = TeamMemory::in_memory().with_observer(Arc::new(NoopObserver));
        assert!(memory.observer().is_some());
        memory
            .remember(decision("review-team", "content"), &CallOptions::new())
            .await
            .expect("remember succeeds with a noop observer");
        let memory = TeamMemory::in_memory().with_observer(Arc::new(TracingObserver));
        let error = memory
            .remember(
                NewTeamMemory::decision("review-team", "   "),
                &CallOptions::new(),
            )
            .await
            .expect_err("empty content is rejected");
        assert!(matches!(error, Error::Validation(_)));
        memory.emit(&TeamMemoryEvent::new(
            TeamMemoryOperation::Prune,
            "review-team",
        ));
        memory.emit(
            &TeamMemoryEvent::new(TeamMemoryOperation::Prune, "review-team")
                .failed(Error::Cancelled),
        );
    }
}
