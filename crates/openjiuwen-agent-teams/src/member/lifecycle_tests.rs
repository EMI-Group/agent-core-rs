//! Test suite for the member status machine ([`TeamMember`]).
//!
//! Kept out of `lifecycle.rs` so the implementation stays readable: the module exercises the
//! compare-and-set start path, round-outcome mapping, retire targets and observer forwarding
//! against the in-memory store and a scripted [`MemberRuntime`] fake.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_harness_protocol::HarnessState;
use openjiuwen_session::Session;
use parking_lot::{Mutex, RwLock};
use serde_json::Value;

use crate::member::lifecycle::TeamMember;
use crate::member::runtime::{MemberObserver, MemberRuntime, ObserverEvent, RecordingObserver};
use crate::schema::{
    ExecutionStatus, MemberLifetime, MemberStatus, RoundOutcome, StorageSpec, TeamRole,
};
use crate::storage::{open_store, MemberRecord, TeamStore};

/// Scripted runtime: counts starts, can fail them, records observers, records the phase.
struct FakeMemberRuntime {
    session_id: String,
    state: RwLock<HarnessState>,
    starts: AtomicUsize,
    stopped: AtomicUsize,
    fail_start: bool,
    observers: Mutex<Vec<Arc<dyn MemberObserver>>>,
    team_session_present: RwLock<Option<bool>>,
}

impl FakeMemberRuntime {
    fn new() -> Self {
        Self {
            session_id: "member-session".to_string(),
            state: RwLock::new(HarnessState::Idle),
            starts: AtomicUsize::new(0),
            stopped: AtomicUsize::new(0),
            fail_start: false,
            observers: Mutex::new(Vec::new()),
            team_session_present: RwLock::new(None),
        }
    }

    fn failing() -> Self {
        Self {
            fail_start: true,
            ..Self::new()
        }
    }

    fn start_count(&self) -> usize {
        self.starts.load(Ordering::SeqCst)
    }

    fn stop_count(&self) -> usize {
        self.stopped.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl MemberRuntime for FakeMemberRuntime {
    fn session_id(&self) -> &str {
        &self.session_id
    }

    fn state(&self) -> HarnessState {
        *self.state.read()
    }

    async fn start(&self, team_session: Option<Arc<dyn Session>>) -> Result<()> {
        self.starts.fetch_add(1, Ordering::SeqCst);
        *self.team_session_present.write() = Some(team_session.is_some());
        if self.fail_start {
            return Err(Error::config("harness could not start"));
        }
        *self.state.write() = HarnessState::Running;
        Ok(())
    }

    async fn stop(&self) -> Result<()> {
        self.stopped.fetch_add(1, Ordering::SeqCst);
        *self.state.write() = HarnessState::Terminated;
        Ok(())
    }

    async fn send(&self, _content: String, _immediate: bool) -> Result<()> {
        Ok(())
    }

    async fn abort(&self, _immediate: bool) -> Result<()> {
        Ok(())
    }

    async fn pause(&self) -> Result<()> {
        *self.state.write() = HarnessState::Paused;
        Ok(())
    }

    async fn resume(&self, _query: Option<String>) -> Result<()> {
        *self.state.write() = HarnessState::Running;
        Ok(())
    }

    fn subscribe(&self, observer: Arc<dyn MemberObserver>) {
        self.observers.lock().push(observer);
    }
}

async fn fixture(role: TeamRole) -> (Arc<dyn TeamStore>, TeamMember) {
    let store = open_store(&StorageSpec::in_memory()).unwrap();
    store
        .insert_member(MemberRecord::new("alpha", "dev-1", role))
        .await
        .unwrap();
    let member = TeamMember::new(
        Arc::clone(&store),
        MemberRecord::new("alpha", "dev-1", role),
    );
    (store, member)
}

async fn stored_status(store: &Arc<dyn TeamStore>) -> MemberStatus {
    store
        .member("alpha", "dev-1")
        .await
        .unwrap()
        .expect("member row")
        .status
}

#[tokio::test]
async fn start_transitions_unstarted_to_ready_and_starts_the_runtime_once() {
    let (store, member) = fixture(TeamRole::Teammate).await;
    let runtime = Arc::new(FakeMemberRuntime::new());
    member.attach_runtime(runtime.clone());

    member.start(None).await.unwrap();

    assert_eq!(runtime.start_count(), 1);
    assert_eq!(*runtime.team_session_present.read(), Some(false));
    assert_eq!(member.status(), MemberStatus::Ready);
    assert_eq!(stored_status(&store).await, MemberStatus::Ready);
}

#[tokio::test]
async fn start_failure_rolls_the_row_back_to_unstarted() {
    let (store, member) = fixture(TeamRole::Teammate).await;
    member.attach_runtime(Arc::new(FakeMemberRuntime::failing()));

    let error = member.start(None).await.unwrap_err();

    assert_eq!(error.kind(), openjiuwen_common::error::ErrorKind::Config);
    assert!(error.message().contains("harness could not start"));
    assert_eq!(member.status(), MemberStatus::Unstarted);
    assert_eq!(stored_status(&store).await, MemberStatus::Unstarted);
}

#[tokio::test]
async fn start_without_a_runtime_is_a_config_error() {
    let (_store, member) = fixture(TeamRole::Teammate).await;
    let error = member.start(None).await.unwrap_err();
    assert_eq!(error.kind(), openjiuwen_common::error::ErrorKind::Config);
    assert!(error.message().contains("no runtime attached"));
    assert_eq!(member.status(), MemberStatus::Unstarted);
}

#[tokio::test]
async fn start_on_a_non_unstarted_member_is_rejected() {
    let (store, member) = fixture(TeamRole::Teammate).await;
    let runtime = Arc::new(FakeMemberRuntime::new());
    member.attach_runtime(runtime.clone());
    member.mark(MemberStatus::Starting).await.unwrap();

    let error = member.start(None).await.unwrap_err();

    assert_eq!(
        error.kind(),
        openjiuwen_common::error::ErrorKind::Validation
    );
    assert!(error
        .message()
        .contains("member 'dev-1' is not in unstarted state"));
    assert_eq!(runtime.start_count(), 0);
    assert_eq!(stored_status(&store).await, MemberStatus::Starting);
}

#[tokio::test]
async fn marking_the_current_status_is_an_idempotent_success() {
    let (store, member) = fixture(TeamRole::Teammate).await;
    assert!(member.mark(MemberStatus::Unstarted).await.unwrap());
    assert_eq!(stored_status(&store).await, MemberStatus::Unstarted);
}

#[tokio::test]
async fn an_illegal_transition_returns_false_and_leaves_the_store_untouched() {
    let (store, member) = fixture(TeamRole::Teammate).await;
    assert!(!member.mark(MemberStatus::Dormant).await.unwrap());
    assert_eq!(member.status(), MemberStatus::Unstarted);
    assert_eq!(stored_status(&store).await, MemberStatus::Unstarted);
}

#[tokio::test]
async fn a_lost_compare_and_set_returns_false_and_refreshes_the_cache() {
    let (store, member) = fixture(TeamRole::Teammate).await;
    let peer = TeamMember::new(
        Arc::clone(&store),
        MemberRecord::new("alpha", "dev-1", TeamRole::Teammate),
    );
    // The peer wins the CAS; our handle still believes the row is unstarted.
    assert!(peer.mark(MemberStatus::Starting).await.unwrap());

    assert!(!member.mark(MemberStatus::Starting).await.unwrap());
    assert_eq!(member.status(), MemberStatus::Starting);
}

#[tokio::test]
async fn execution_writes_are_validated_and_persisted() {
    let (store, member) = fixture(TeamRole::Teammate).await;
    member
        .mark_execution(ExecutionStatus::Starting)
        .await
        .unwrap();
    assert_eq!(member.execution(), ExecutionStatus::Starting);
    let row = store.member("alpha", "dev-1").await.unwrap().unwrap();
    assert_eq!(row.execution, ExecutionStatus::Starting);

    let error = member
        .mark_execution(ExecutionStatus::Completed)
        .await
        .unwrap_err();
    assert_eq!(
        error.kind(),
        openjiuwen_common::error::ErrorKind::Validation
    );
    assert_eq!(member.execution(), ExecutionStatus::Starting);
}

#[tokio::test]
async fn note_started_walks_to_running_and_marks_the_member_busy() {
    let (store, member) = fixture(TeamRole::Teammate).await;
    member.mark(MemberStatus::Starting).await.unwrap();
    member.mark(MemberStatus::Ready).await.unwrap();

    member.note_round(RoundOutcome::Started).await.unwrap();

    assert_eq!(member.execution(), ExecutionStatus::Running);
    assert_eq!(member.status(), MemberStatus::Busy);
    let row = store.member("alpha", "dev-1").await.unwrap().unwrap();
    assert_eq!(row.execution, ExecutionStatus::Running);
    assert_eq!(row.status, MemberStatus::Busy);
}

#[tokio::test]
async fn note_finished_walks_to_idle_and_releases_the_member() {
    let (store, member) = fixture(TeamRole::Teammate).await;
    member.mark(MemberStatus::Starting).await.unwrap();
    member.mark(MemberStatus::Ready).await.unwrap();
    member.note_round(RoundOutcome::Started).await.unwrap();

    member.note_round(RoundOutcome::Finished).await.unwrap();

    assert_eq!(member.execution(), ExecutionStatus::Idle);
    assert_eq!(member.status(), MemberStatus::Ready);
    let row = store.member("alpha", "dev-1").await.unwrap().unwrap();
    assert_eq!(row.execution, ExecutionStatus::Idle);
    assert_eq!(row.status, MemberStatus::Ready);
}

#[tokio::test]
async fn note_aborted_walks_the_cancel_chain_to_idle() {
    let (_store, member) = fixture(TeamRole::Teammate).await;
    member.mark(MemberStatus::Starting).await.unwrap();
    member.mark(MemberStatus::Ready).await.unwrap();
    member.note_round(RoundOutcome::Started).await.unwrap();

    member.note_round(RoundOutcome::Aborted).await.unwrap();

    assert_eq!(member.execution(), ExecutionStatus::Idle);
    assert_eq!(member.status(), MemberStatus::Ready);
}

#[tokio::test]
async fn note_failed_walks_failed_to_idle() {
    let (_store, member) = fixture(TeamRole::Teammate).await;
    member.mark(MemberStatus::Starting).await.unwrap();
    member.mark(MemberStatus::Ready).await.unwrap();
    member.note_round(RoundOutcome::Started).await.unwrap();

    member.note_round(RoundOutcome::Failed).await.unwrap();

    assert_eq!(member.execution(), ExecutionStatus::Idle);
    assert_eq!(member.status(), MemberStatus::Ready);
}

#[tokio::test]
async fn repeating_a_round_outcome_is_idempotent() {
    let (_store, member) = fixture(TeamRole::Teammate).await;
    member.mark(MemberStatus::Starting).await.unwrap();
    member.mark(MemberStatus::Ready).await.unwrap();
    member.note_round(RoundOutcome::Started).await.unwrap();
    member.note_round(RoundOutcome::Finished).await.unwrap();

    // A late duplicate callback must not replay an unreachable walk.
    member.note_round(RoundOutcome::Finished).await.unwrap();
    assert_eq!(member.execution(), ExecutionStatus::Idle);
    assert_eq!(member.status(), MemberStatus::Ready);
}

#[tokio::test]
async fn a_duplicate_terminal_note_does_not_revive_a_retired_member() {
    let (_store, member) = fixture(TeamRole::Teammate).await;
    member.mark(MemberStatus::Starting).await.unwrap();
    member.mark(MemberStatus::Ready).await.unwrap();
    member.note_round(RoundOutcome::Started).await.unwrap();
    member.note_round(RoundOutcome::Finished).await.unwrap();
    member.mark(MemberStatus::Stopped).await.unwrap();

    member.note_round(RoundOutcome::Finished).await.unwrap();

    assert_eq!(member.execution(), ExecutionStatus::Idle);
    assert_eq!(member.status(), MemberStatus::Stopped);
}

#[tokio::test]
async fn shutdown_settles_shutdown_lifetime_to_shut_down() {
    let (store, member) = fixture(TeamRole::Teammate).await;
    member.mark(MemberStatus::Starting).await.unwrap();
    member.mark(MemberStatus::Ready).await.unwrap();

    let settled = member
        .shutdown(MemberLifetime::Shutdown, false)
        .await
        .unwrap();

    assert_eq!(settled, MemberStatus::ShutDown);
    assert_eq!(member.status(), MemberStatus::ShutDown);
    assert_eq!(stored_status(&store).await, MemberStatus::ShutDown);
}

#[tokio::test]
async fn shutdown_settles_dormant_lifetime_to_dormant() {
    let (_store, member) = fixture(TeamRole::Teammate).await;
    member.mark(MemberStatus::Starting).await.unwrap();
    member.mark(MemberStatus::Ready).await.unwrap();

    let settled = member
        .shutdown(MemberLifetime::Dormant, false)
        .await
        .unwrap();

    assert_eq!(settled, MemberStatus::Dormant);
    assert_eq!(member.status(), MemberStatus::Dormant);
}

#[tokio::test]
async fn forced_shutdown_wins_over_a_dormant_lifetime() {
    let (_store, member) = fixture(TeamRole::Teammate).await;
    member.mark(MemberStatus::Starting).await.unwrap();
    member.mark(MemberStatus::Ready).await.unwrap();

    let settled = member
        .shutdown(MemberLifetime::Dormant, true)
        .await
        .unwrap();

    assert_eq!(settled, MemberStatus::ShutDown);
}

#[tokio::test]
async fn shutdown_of_an_unstarted_member_settles_directly_to_shut_down() {
    let (_store, member) = fixture(TeamRole::Teammate).await;
    let settled = member
        .shutdown(MemberLifetime::Shutdown, false)
        .await
        .unwrap();
    assert_eq!(settled, MemberStatus::ShutDown);
}

#[tokio::test]
async fn shutdown_of_an_unstarted_member_cannot_reach_dormant() {
    let (store, member) = fixture(TeamRole::Teammate).await;
    let error = member
        .shutdown(MemberLifetime::Dormant, false)
        .await
        .unwrap_err();
    assert_eq!(
        error.kind(),
        openjiuwen_common::error::ErrorKind::Validation
    );
    assert!(error.message().contains("cannot shut down from status"));
    assert_eq!(stored_status(&store).await, MemberStatus::Unstarted);
}

#[tokio::test]
async fn shutdown_is_idempotent_once_settled() {
    let (_store, member) = fixture(TeamRole::Teammate).await;
    member.mark(MemberStatus::Starting).await.unwrap();
    member.mark(MemberStatus::Ready).await.unwrap();
    member
        .shutdown(MemberLifetime::Shutdown, false)
        .await
        .unwrap();

    let settled = member
        .shutdown(MemberLifetime::Shutdown, false)
        .await
        .unwrap();
    assert_eq!(settled, MemberStatus::ShutDown);
}

#[tokio::test]
async fn error_moves_the_member_into_the_error_status() {
    let (_store, member) = fixture(TeamRole::Teammate).await;
    member.mark(MemberStatus::Starting).await.unwrap();
    member.mark(MemberStatus::Ready).await.unwrap();

    member.error("model blew up").await.unwrap();

    assert_eq!(member.status(), MemberStatus::Error);
    // A second report is harmless.
    member.error("model blew up again").await.unwrap();
    assert_eq!(member.status(), MemberStatus::Error);
}

#[test]
fn debug_names_the_identity_and_state() {
    let store = open_store(&StorageSpec::in_memory()).unwrap();
    let member = TeamMember::new(
        store,
        MemberRecord::new("alpha", "dev-1", TeamRole::Teammate),
    );
    let rendered = format!("{member:?}");
    assert!(rendered.contains("dev-1"));
    assert!(rendered.contains("has_runtime: false"));
}

#[tokio::test]
async fn observers_wait_for_the_runtime_and_are_forwarded_on_attach() {
    let (_store, member) = fixture(TeamRole::Teammate).await;
    let observer = Arc::new(RecordingObserver::new());
    member.observe(observer.clone());

    let runtime = Arc::new(FakeMemberRuntime::new());
    member.attach_runtime(runtime.clone());
    assert_eq!(runtime.observers.lock().len(), 1);

    // An observer added after attach goes straight to the runtime.
    let late = Arc::new(RecordingObserver::new());
    member.observe(late.clone());
    assert_eq!(runtime.observers.lock().len(), 2);

    runtime.observers.lock()[0].on_chunk(Value::from(1));
    assert_eq!(observer.take(), vec![ObserverEvent::Chunk(Value::from(1))]);
    assert!(late.is_empty());
}

#[tokio::test]
async fn refresh_reads_a_peer_write_into_the_cache() {
    let (store, member) = fixture(TeamRole::Teammate).await;
    let peer = TeamMember::new(
        Arc::clone(&store),
        MemberRecord::new("alpha", "dev-1", TeamRole::Teammate),
    );
    peer.mark(MemberStatus::Starting).await.unwrap();
    assert_eq!(member.status(), MemberStatus::Unstarted);

    let record = member.refresh().await.unwrap();
    assert_eq!(record.status, MemberStatus::Starting);
    assert_eq!(member.status(), MemberStatus::Starting);
}

#[tokio::test]
async fn getters_expose_identity_and_role() {
    let (_store, member) = fixture(TeamRole::Leader).await;
    assert_eq!(member.team_name(), "alpha");
    assert_eq!(member.member_name(), "dev-1");
    assert_eq!(member.role(), TeamRole::Leader);
    assert!(member.runtime().is_none());
    assert_eq!(member.execution(), ExecutionStatus::Idle);
}

#[tokio::test]
async fn mark_execution_on_a_missing_row_is_not_found() {
    let store = open_store(&StorageSpec::in_memory()).unwrap();
    let member = TeamMember::new(
        store,
        MemberRecord::new("alpha", "ghost", TeamRole::Teammate),
    );
    let error = member
        .mark_execution(ExecutionStatus::Starting)
        .await
        .unwrap_err();
    assert_eq!(error.kind(), openjiuwen_common::error::ErrorKind::NotFound);
}

#[tokio::test]
async fn stopping_the_runtime_is_the_owners_call_not_the_status_machine() {
    // The status machine never stops the runtime; assert the seam stays that way so the
    // runtime manager keeps sole ownership of teardown order.
    let (_store, member) = fixture(TeamRole::Teammate).await;
    let runtime = Arc::new(FakeMemberRuntime::new());
    member.attach_runtime(runtime.clone());
    member.start(None).await.unwrap();
    member
        .shutdown(MemberLifetime::Shutdown, false)
        .await
        .unwrap();
    assert_eq!(runtime.stop_count(), 0);
    assert!(member.runtime().is_some());
}
