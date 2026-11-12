//! Offline, deterministic test suite of the native harness protocol node.
//!
//! Every test drives the real `NativeHarness` over a kernel `DeepAgent` built on
//! [`openjiuwen_test_support::ScriptedModel`] and a temporary workspace: no
//! network, no real provider, no `tokio::time::sleep`. Ordering that matters
//! (a queued follow-up, a pause without an active turn, a finished turn before
//! its cursor is read) is forced with semaphores or checkpoint-sink
//! notifications instead of timing assumptions; cursor reads are wrapped in a
//! five-second timeout purely as a safety net.
//!
//! The suite is split by area: `lifecycle` (start/stop and its errors), `turns`
//! (receipts, lifecycle events, cursors, steering/abort, pause/resume),
//! `checkpoints` (export, sink, restore), `provider` (create mapping and model
//! choice), and `model_control` (discovery and selection).

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use futures::stream::BoxStream;
use futures::StreamExt;
use openjiuwen_agent::AgentCard;
use openjiuwen_common::error::Result;
use openjiuwen_foundation::llm::{ChatResponse, Model, ModelRequestConfig, StreamChunk};
use openjiuwen_foundation::BaseMessage;
use openjiuwen_harness_protocol::{
    json_object_to_value, AbortMode, CheckpointReason, CheckpointSaveReceipt, DeliveryMode,
    HarnessCheckpoint, HarnessCheckpointSink, HarnessContext, HarnessError, HarnessErrorKind,
    HarnessEvent, HarnessEventCursor, HarnessEventPayload, HarnessInput, HarnessModelControl,
    HarnessProtocol, HarnessResult, HarnessState, JsonObject, JsonValue, McpServerConfig,
    McpTransport, ModelSelection, OutputChannel, ResumePolicy, StateChangedEvent, TurnEventKind,
    TurnLifecycleEvent, TurnStatus,
};
use openjiuwen_test_support::ScriptedModel;
use parking_lot::Mutex;
use tempfile::TempDir;
use tokio::sync::Semaphore;

use super::bridge::NativeCheckpointState;
use super::{NativeHarness, NativeHarnessOptions, HARNESS_NAME};
use crate::agent::{DeepAgent, DeepAgentBuilder};
use crate::schema::DeepAgentConfig;
use crate::workspace::Workspace;

mod checkpoints;
mod lifecycle;
mod model_control;
mod provider;
mod turns;

/// Timeout applied to every await that must not hang the suite.
const WAIT: Duration = Duration::from_secs(5);

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// Agents, harness, temporary workspace, and optional checkpoint sink of one test.
struct Fixture {
    /// Kept alive so the temporary workspace outlives the agent.
    _tmp: TempDir,
    /// Agent the harness drives.
    agent: Arc<DeepAgent>,
    /// Harness under test.
    harness: NativeHarness,
    /// Sink attached through the start context, when the test needs one.
    sink: Option<Arc<FakeSink>>,
}

impl Fixture {
    /// Builds the context the harness is started with, attaching the sink.
    fn context(&self) -> HarnessContext {
        match &self.sink {
            Some(sink) => {
                context().with_checkpoint_sink(sink.clone() as Arc<dyn HarnessCheckpointSink>)
            }
            None => context(),
        }
    }

    /// Starts the harness with the fixture context and asserts it succeeded.
    async fn start(&self) {
        self.harness
            .start(self.context())
            .await
            .expect("start succeeds");
    }

    /// Starts the harness with an explicit context and asserts it succeeded.
    async fn start_with(&self, context: HarnessContext) {
        self.harness.start(context).await.expect("start succeeds");
    }

    /// Returns the shared state of the fixture sink.
    fn sink_state(&self) -> Arc<SinkState> {
        self.sink
            .as_ref()
            .expect("the fixture has a checkpoint sink")
            .state()
            .clone()
    }
}

/// Builds a harness over a scripted-model agent in a temporary workspace.
fn fixture_with(model: impl Model + 'static, config: DeepAgentConfig) -> Fixture {
    build_fixture(model, config, None)
}

/// Builds a harness whose start context carries a [`FakeSink`].
fn fixture_with_sink(conflict: bool) -> Fixture {
    let sink = FakeSink::new(conflict);
    build_fixture(ScriptedModel::text("hello"), scripted_config(), Some(sink))
}

/// Shared construction of every fixture.
fn build_fixture(
    model: impl Model + 'static,
    config: DeepAgentConfig,
    sink: Option<Arc<FakeSink>>,
) -> Fixture {
    let tmp = tempfile::tempdir().expect("temporary workspace");
    let agent = DeepAgentBuilder::new(AgentCard::new("deep_agent"))
        .config(config)
        .model(Arc::new(model) as Arc<dyn Model>)
        .workspace(Workspace::new(tmp.path()))
        .build()
        .expect("the deep agent builds");
    let harness = NativeHarness::new(agent.clone(), NativeHarnessOptions::new());
    Fixture {
        _tmp: tmp,
        agent,
        harness,
        sink,
    }
}

/// The default fixture: a plain scripted answer over a temporary workspace.
fn fixture() -> Fixture {
    fixture_with(ScriptedModel::text("hello"), scripted_config())
}

/// A configuration naming the scripted model so `list_models` has an id.
fn scripted_config() -> DeepAgentConfig {
    DeepAgentConfig {
        model_name: Some("scripted-model".to_owned()),
        ..DeepAgentConfig::default()
    }
}

/// The standard host context of the suite.
fn context() -> HarnessContext {
    HarnessContext::new("deep_agent", "agent-1", "host-1", "")
}

// ---------------------------------------------------------------------------
// Cursor and event helpers
// ---------------------------------------------------------------------------

/// Reads the next event of `cursor`, failing fast on timeout or cursor error.
async fn next_event(cursor: &mut Box<dyn HarnessEventCursor>) -> Option<HarnessEvent> {
    tokio::time::timeout(WAIT, cursor.next_event())
        .await
        .expect("timed out waiting for a protocol event")
        .expect("the observation cursor reported an error")
}

/// Reads the error a cursor reports on its next call.
async fn next_error(cursor: &mut Box<dyn HarnessEventCursor>) -> HarnessError {
    tokio::time::timeout(WAIT, cursor.next_event())
        .await
        .expect("timed out waiting for the cursor error")
        .expect_err("expected the cursor to report an error")
}

/// Reads events until `predicate` accepts one, returning everything read.
async fn drain_until(
    cursor: &mut Box<dyn HarnessEventCursor>,
    mut predicate: impl FnMut(&HarnessEvent) -> bool,
) -> Vec<HarnessEvent> {
    let mut events = Vec::new();
    loop {
        let Some(event) = next_event(cursor).await else {
            panic!("the observation cursor ended before the expected event");
        };
        let matched = predicate(&event);
        events.push(event);
        if matched {
            return events;
        }
    }
}

/// Returns the state transition `event` carries, when it is one.
fn state_change(event: &HarnessEvent) -> Option<&StateChangedEvent> {
    match &event.event {
        HarnessEventPayload::StateChanged(payload) => Some(payload),
        _ => None,
    }
}

/// Returns the turn lifecycle payload `event` carries, when it is one.
fn lifecycle(event: &HarnessEvent) -> Option<&TurnLifecycleEvent> {
    match &event.event {
        HarnessEventPayload::TurnLifecycle(payload) => Some(payload),
        _ => None,
    }
}

/// Returns whether `event` is the `kind` lifecycle event of `turn_id`.
fn is_turn_event(event: &HarnessEvent, turn_id: &str, kind: TurnEventKind) -> bool {
    event.turn_id.as_deref() == Some(turn_id)
        && matches!(lifecycle(event), Some(payload) if payload.kind == kind)
}

/// Decodes a checkpoint payload into the provider-private state.
fn decoded(checkpoint: &HarnessCheckpoint) -> NativeCheckpointState {
    serde_json::from_value(json_object_to_value(&checkpoint.data))
        .expect("the checkpoint payload decodes")
}

/// Builds a valid native checkpoint scoped to `agent_id`/`host_session_id`.
///
/// The payload carries a restored turn counter of 4 and plan mode `plan`, so a
/// resumed harness must number its next turn `turn-5`.
fn checkpoint_for(agent_id: &str, host_session_id: &str) -> HarnessCheckpoint {
    let mut state = NativeCheckpointState::new(host_session_id, agent_id, host_session_id);
    state.turn_counter = 4;
    state.plan_mode = "plan".to_owned();
    HarnessCheckpoint {
        provider: HARNESS_NAME.to_owned(),
        schema_version: "1".to_owned(),
        agent_id: agent_id.to_owned(),
        host_session_id: host_session_id.to_owned(),
        checkpoint_id: "checkpoint-restore".to_owned(),
        sequence: 1,
        data: state.to_json(),
        provider_session_id: None,
        revision: None,
    }
}

// ---------------------------------------------------------------------------
// Checkpoint sink fake
// ---------------------------------------------------------------------------

/// Checkpoint sink recording every write and waking waiters after each one.
#[derive(Default)]
struct SinkState {
    /// `(reason, checkpoint)` pairs in write order.
    writes: Mutex<Vec<(CheckpointReason, HarnessCheckpoint)>>,
    /// Wakes a test waiting for a specific reason.
    notify: tokio::sync::Notify,
}

impl SinkState {
    /// Returns the recorded reasons in write order.
    fn reasons(&self) -> Vec<CheckpointReason> {
        self.writes
            .lock()
            .iter()
            .map(|(reason, _)| *reason)
            .collect()
    }

    /// Returns the recorded sequences in write order.
    fn sequences(&self) -> Vec<u64> {
        self.writes
            .lock()
            .iter()
            .map(|(_, checkpoint)| checkpoint.sequence)
            .collect()
    }
}

/// Sink that records writes and optionally fails every one with a conflict.
struct FakeSink {
    /// Shared write log and notification.
    state: Arc<SinkState>,
    /// Whether every write reports [`HarnessError::CheckpointConflict`].
    conflict: bool,
}

impl FakeSink {
    /// Builds a sink that fails every write when `conflict` is set.
    fn new(conflict: bool) -> Arc<Self> {
        Arc::new(Self {
            state: Arc::new(SinkState::default()),
            conflict,
        })
    }

    /// Returns the shared state of this sink.
    fn state(&self) -> &Arc<SinkState> {
        &self.state
    }
}

#[async_trait]
impl HarnessCheckpointSink for FakeSink {
    async fn save(
        &self,
        checkpoint: &HarnessCheckpoint,
        reason: CheckpointReason,
        _expected_storage_revision: Option<&str>,
    ) -> HarnessResult<CheckpointSaveReceipt> {
        self.state.writes.lock().push((reason, checkpoint.clone()));
        self.state.notify.notify_waiters();
        if self.conflict {
            return Err(HarnessError::checkpoint_conflict("stale"));
        }
        Ok(CheckpointSaveReceipt {
            checkpoint_id: checkpoint.checkpoint_id.clone(),
            sequence: checkpoint.sequence,
            storage_revision: format!("rev-{}", checkpoint.sequence),
        })
    }
}

/// Exports the latest checkpoint, waiting until the checkpoint writer is restored.
///
/// [`NativeHarness::export_checkpoint`] briefly sees no checkpoint while
/// [`NativeHarnessInner::emit_checkpoint`](super::inner) awaits the host sink, so
/// a test that observed the sink write must yield until the writer is back.
async fn export_settled(harness: &NativeHarness) -> HarnessCheckpoint {
    let waiting = async {
        loop {
            if let Some(checkpoint) = harness
                .export_checkpoint()
                .await
                .expect("export is infallible")
            {
                return checkpoint;
            }
            tokio::task::yield_now().await;
        }
    };
    tokio::time::timeout(WAIT, waiting)
        .await
        .expect("the checkpoint writer was never restored")
}

/// Waits until the sink recorded a checkpoint published for `reason`.
async fn wait_for_checkpoint(
    state: &Arc<SinkState>,
    reason: CheckpointReason,
) -> HarnessCheckpoint {
    let waiting = async {
        loop {
            let found = state
                .writes
                .lock()
                .iter()
                .find(|(recorded, _)| *recorded == reason)
                .map(|(_, checkpoint)| checkpoint.clone());
            if let Some(checkpoint) = found {
                return checkpoint;
            }
            let notified = state.notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            let raced = state
                .writes
                .lock()
                .iter()
                .any(|(recorded, _)| *recorded == reason);
            if raced {
                continue;
            }
            notified.await;
        }
    };
    tokio::time::timeout(WAIT, waiting)
        .await
        .expect("timed out waiting for the checkpoint")
}

// ---------------------------------------------------------------------------
// Gated model
// ---------------------------------------------------------------------------

/// A scripted model that parks its streamed call until the test releases it.
struct GateModel {
    /// Scripted answers.
    inner: ScriptedModel,
    /// One permit is added as soon as a streamed call is entered.
    entered: Arc<Semaphore>,
    /// The call proceeds after the test adds a permit here.
    release: Arc<Semaphore>,
}

impl GateModel {
    /// Builds a gated model over `inner`, returning the model and both gates.
    fn new(inner: ScriptedModel) -> (Self, Arc<Semaphore>, Arc<Semaphore>) {
        let entered = Arc::new(Semaphore::new(0));
        let release = Arc::new(Semaphore::new(0));
        (
            Self {
                inner,
                entered: entered.clone(),
                release: release.clone(),
            },
            entered,
            release,
        )
    }
}

#[async_trait]
impl Model for GateModel {
    fn provider(&self) -> &str {
        self.inner.provider()
    }

    fn model_name(&self) -> &str {
        self.inner.model_name()
    }

    async fn invoke(
        &self,
        messages: &[BaseMessage],
        request: &ModelRequestConfig,
    ) -> Result<ChatResponse> {
        self.inner.invoke(messages, request).await
    }

    fn stream(
        &self,
        messages: Vec<BaseMessage>,
        request: ModelRequestConfig,
    ) -> BoxStream<'static, Result<StreamChunk>> {
        let inner = self.inner.clone();
        let entered = self.entered.clone();
        let release = self.release.clone();
        async_stream::stream! {
            entered.add_permits(1);
            let _permit = release.acquire().await;
            let mut frames = inner.stream(messages, request);
            while let Some(frame) = frames.next().await {
                yield frame;
            }
        }
        .boxed()
    }
}
