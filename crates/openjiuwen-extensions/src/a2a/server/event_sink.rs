//! The event publisher an executor drives; every emit also updates the store.

use std::collections::BTreeSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use serde::Serialize;
use serde_json::{Map, Value};
use tokio::sync::mpsc;
use tokio::sync::mpsc::error::TrySendError;

use super::store::TaskStore;
use super::{agent_message, fresh_id};
use crate::a2a::types::{
    Artifact, Message, Task, TaskArtifactUpdateEvent, TaskState, TaskStatus, TaskStatusUpdateEvent,
};

/// How many events one streaming subscriber may fall behind before the server
/// drops it.
///
/// The channel is deliberately bounded: an unbounded one would let a client
/// that never reads its `text/event-stream` response make the server buffer the
/// whole event stream in memory. When the buffer is full the publisher is
/// **never** blocked — it logs the overflow at `warn`, closes the subscriber's
/// channel, and the stream reports a terminal `INTERNAL` error frame (see
/// [`super::sse`]). The task itself keeps running and the [`TaskStore`] stays
/// coherent, so `GetTask`/`CancelTask` and a later `SubscribeToTask` still see
/// the finished task.
pub(super) const EVENT_BUFFER_CAPACITY: usize = 256;

/// `tracing` target for event-channel failures.
const LOG_TARGET: &str = "openjiuwen_extensions::a2a::server::event_sink";

/// The subscriber-facing state of one event stream.
#[derive(Debug)]
struct StreamState {
    /// Set once the subscriber was dropped for falling behind.
    overflowed: AtomicBool,
}

/// The receiving half of one [`EventSink`]'s event stream.
///
/// Wraps the bounded channel and reports whether the stream ended because the
/// subscriber overflowed, so [`super::sse`] can finish it with a terminal error
/// frame instead of silently truncating the event sequence.
#[derive(Debug)]
pub(crate) struct EventStream {
    /// The bounded event channel.
    receiver: mpsc::Receiver<Value>,
    /// Shared overflow flag, also observed by the publisher.
    state: Arc<StreamState>,
}

impl EventStream {
    /// Waits for the next event; `None` once the publisher is gone or the
    /// subscriber was dropped.
    pub(crate) async fn recv(&mut self) -> Option<Value> {
        self.receiver.recv().await
    }

    /// Takes the next buffered event without waiting (used by the tests).
    #[cfg(test)]
    pub(crate) fn try_recv(&mut self) -> std::result::Result<Value, mpsc::error::TryRecvError> {
        self.receiver.try_recv()
    }

    /// Whether this stream was terminated because it fell behind.
    pub(crate) fn overflowed(&self) -> bool {
        self.state.overflowed.load(Ordering::Acquire)
    }
}

/// The publisher an [`super::A2aExecutor`] drives while it produces a task.
///
/// Python parity: `a2a.server.tasks.task_updater.TaskUpdater` as used by
/// `A2AAgentExecutor`. Every emit is serialized once, handed to the streaming
/// channel, and folded into the [`TaskStore`] so readers that never see the
/// stream — `GetTask`, `CancelTask`, a non-streaming `SendMessage` — still
/// observe coherent state.
///
/// All methods are infallible: a dropped receiver is ignored (the store update
/// still happens), a subscriber that falls more than [`EVENT_BUFFER_CAPACITY`]
/// events behind is dropped without blocking the publisher, and a terminal
/// method is a no-op once the task has reached a terminal state, matching
/// Python's "already in a terminal state" behavior.
pub struct EventSink {
    task_id: String,
    context_id: String,
    store: Arc<dyn TaskStore>,
    /// The bounded sender; `None` once the subscriber was dropped.
    sender: Mutex<Option<mpsc::Sender<Value>>>,
    open_artifacts: Mutex<BTreeSet<String>>,
    terminal: AtomicBool,
    stream: Arc<StreamState>,
}

impl EventSink {
    /// Creates a sink and the receiver its events are streamed through.
    pub(crate) fn new(
        task_id: String,
        context_id: String,
        store: Arc<dyn TaskStore>,
    ) -> (Self, EventStream) {
        let (sender, receiver) = mpsc::channel(EVENT_BUFFER_CAPACITY);
        let stream = Arc::new(StreamState {
            overflowed: AtomicBool::new(false),
        });
        let receiver_state = Arc::clone(&stream);
        let sink = Self {
            task_id,
            context_id,
            store,
            sender: Mutex::new(Some(sender)),
            open_artifacts: Mutex::new(BTreeSet::new()),
            terminal: AtomicBool::new(false),
            stream,
        };
        (
            sink,
            EventStream {
                receiver,
                state: receiver_state,
            },
        )
    }

    /// The id of the task this sink publishes for.
    #[must_use]
    pub fn task_id(&self) -> &str {
        &self.task_id
    }

    /// The context (conversation) id this sink publishes for.
    #[must_use]
    pub fn context_id(&self) -> &str {
        &self.context_id
    }

    /// Whether a terminal status has been published already.
    #[must_use]
    pub fn is_terminal(&self) -> bool {
        self.terminal.load(Ordering::Acquire)
    }

    /// Whether the streaming subscriber was dropped for falling behind.
    #[must_use]
    pub fn subscriber_overflowed(&self) -> bool {
        self.stream.overflowed.load(Ordering::Acquire)
    }

    /// Publishes the [`Task`] snapshot and stores it (`state` as supplied).
    pub fn task(&self, task: Task) {
        self.emit(&task);
        self.store.put(task);
    }

    /// Publishes a `WORKING` status update and stores it.
    pub fn working(&self, message: Option<Message>) {
        let status = TaskStatus {
            state: TaskState::Working,
            message,
            timestamp: None,
        };
        self.emit(&TaskStatusUpdateEvent {
            task_id: self.task_id.clone(),
            context_id: self.context_id.clone(),
            status: status.clone(),
            metadata: Map::new(),
        });
        self.update_stored(|task| task.status = status);
    }

    /// Publishes one artifact chunk and stores the artifact.
    ///
    /// An empty [`Artifact::artifact_id`] is filled with a fresh id. While
    /// `last_chunk` is `false` the artifact stays open, so a later terminal
    /// method closes it; with `last_chunk` the artifact is complete.
    pub fn artifact(&self, artifact: Artifact, last_chunk: bool) {
        let artifact = with_artifact_id(artifact);
        let artifact_id = artifact.artifact_id.clone();
        self.emit(&TaskArtifactUpdateEvent {
            task_id: self.task_id.clone(),
            context_id: self.context_id.clone(),
            artifact: artifact.clone(),
            append: false,
            last_chunk,
            metadata: Map::new(),
        });
        self.with_open_artifacts(|open| {
            if last_chunk {
                open.remove(&artifact_id);
            } else {
                open.insert(artifact_id.clone());
            }
        });
        self.update_stored(|task| {
            match task
                .artifacts
                .iter_mut()
                .find(|existing| existing.artifact_id == artifact_id)
            {
                Some(existing) => *existing = artifact,
                None => task.artifacts.push(artifact),
            }
        });
    }

    /// Publishes the `COMPLETED` terminal status.
    pub fn complete(&self) {
        self.finish(TaskState::Completed, None);
    }

    /// Publishes the `FAILED` terminal status with `message` as its text.
    pub fn fail(&self, message: impl Into<String>) {
        let status_message = agent_message(message, &self.task_id, &self.context_id);
        self.finish(TaskState::Failed, Some(status_message));
    }

    /// Publishes the `INPUT_REQUIRED` terminal status.
    pub fn require_input(&self, message: Option<Message>) {
        self.finish(TaskState::InputRequired, message);
    }

    /// Publishes the `CANCELED` terminal status.
    pub fn canceled(&self) {
        self.finish(TaskState::Canceled, None);
    }

    /// Publishes a terminal status: closes open artifacts first, then the status.
    fn finish(&self, state: TaskState, message: Option<Message>) {
        if self.terminal.swap(true, Ordering::AcqRel) {
            return;
        }
        self.close_open_artifacts();
        let status = TaskStatus {
            state,
            message,
            timestamp: None,
        };
        self.emit(&TaskStatusUpdateEvent {
            task_id: self.task_id.clone(),
            context_id: self.context_id.clone(),
            status: status.clone(),
            metadata: Map::new(),
        });
        self.update_stored(|task| task.status = status);
    }

    /// Closes every open artifact without replacing its stored contents.
    fn close_open_artifacts(&self) {
        let open = self.with_open_artifacts(std::mem::take);
        for artifact_id in open {
            self.emit(&TaskArtifactUpdateEvent {
                task_id: self.task_id.clone(),
                context_id: self.context_id.clone(),
                artifact: Artifact {
                    artifact_id,
                    ..Artifact::default()
                },
                append: true,
                last_chunk: true,
                metadata: Map::new(),
            });
        }
    }

    /// Serializes `event` and queues it for the stream.
    ///
    /// Never blocks: a full buffer drops the subscriber (and records the
    /// overflow) rather than stalling the executor.
    fn emit<T: Serialize>(&self, event: &T) {
        let value = match serde_json::to_value(event) {
            Ok(value) => value,
            Err(error) => {
                tracing::error!(%error, "failed to serialize an A2A event");
                return;
            }
        };
        let mut sender = self.sender.lock().unwrap_or_else(PoisonError::into_inner);
        let Some(channel) = sender.as_ref() else {
            // The subscriber was dropped on an earlier overflow.
            return;
        };
        match channel.try_send(value) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) => {
                // Drop the subscriber: closing the channel ends its stream, and
                // the sse layer turns the recorded overflow into a terminal
                // error frame. The task itself keeps running.
                sender.take();
                self.stream.overflowed.store(true, Ordering::Release);
                tracing::warn!(
                    target: LOG_TARGET,
                    task_id = self.task_id,
                    capacity = EVENT_BUFFER_CAPACITY,
                    "A2A streaming subscriber fell behind; dropping it without blocking the publisher"
                );
            }
            Err(TrySendError::Closed(_)) => {
                // The receiver was dropped (for example a non-streaming send).
                // The store update still happens.
                tracing::debug!("A2A event receiver dropped; the task store is still updated");
            }
        }
    }

    /// Applies `update` to the stored task, when it exists.
    fn update_stored(&self, update: impl FnOnce(&mut Task)) {
        if let Some(mut task) = self.store.get(&self.task_id) {
            update(&mut task);
            self.store.put(task);
        }
    }

    /// Runs `update` over the open-artifact id set.
    fn with_open_artifacts<R>(&self, update: impl FnOnce(&mut BTreeSet<String>) -> R) -> R {
        let mut open = self
            .open_artifacts
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        update(&mut open)
    }
}

/// Fills an empty artifact id with a fresh one.
fn with_artifact_id(mut artifact: Artifact) -> Artifact {
    if artifact.artifact_id.is_empty() {
        artifact.artifact_id = fresh_id();
    }
    artifact
}
