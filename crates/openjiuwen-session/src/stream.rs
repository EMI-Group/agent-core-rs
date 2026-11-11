//! Stream frames, the bounded delivery hub, and mode-routed writers.
//!
//! Maps to `openjiuwen/core/session/stream/` in the Python reference (v0.1.20):
//! [`StreamMode`] and the `OutputSchema`/`TraceSchema`/`CustomSchema` payload
//! shapes become [`StreamItem`], the per-mode `StreamWriter` classes collapse
//! into [`StreamWriterManager`], and the single-consumer asyncio emitter
//! becomes the bounded broadcast [`StreamHub`].
//!
//! # Delivery contract
//!
//! - A [`StreamHub`] fans one ordered stream out to any number of subscribers;
//!   ordering is guaranteed per subscriber, not across subscribers.
//! - The transport is bounded ([`DEFAULT_STREAM_CAPACITY`]). A subscriber that
//!   falls behind loses the oldest frames: the hub logs a warning and skips
//!   them instead of blocking producers.
//! - Completion is an explicit [`StreamEvent::End`] event. [`StreamHub::close`]
//!   publishes it once, idempotently, and the channel closing for any other
//!   reason is reported as `End` as well so every consumer terminates.
//! - Python's `END_FRAME` sentinel is filtered out of user-visible output there;
//!   here the completion marker is a distinct event and frames matching
//!   [`END_FRAME`] can still be recognised through [`StreamItem::is_end`].
//! - Publishing is synchronous at heart: [`StreamHub::send_now`] and
//!   [`StreamWriterManager::write_now`] never await, so synchronous producers
//!   (trace sinks, plain callbacks) can emit frames. [`StreamHub::send`] and
//!   [`StreamWriterManager::write`] are `async` wrappers that await nothing;
//!   they exist for API symmetry with Python's `await writer.write(...)` and so
//!   future backends can become genuinely asynchronous without a breaking
//!   change.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use async_stream::stream;
use futures::stream::BoxStream;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tokio::sync::broadcast::{self, error::RecvError};

use openjiuwen_common::error::{Error, Result};

use crate::constants::{DEFAULT_STREAM_CAPACITY, END_FRAME};

/// Routing mode of a stream frame.
///
/// Mirrors Python's `BaseStreamMode`: `OUTPUT` carries framework-defined
/// stream data, `TRACE` carries graph trace data, and `CUSTOM` carries
/// runnable-defined data. The `snake_case` wire names are `output`, `trace`,
/// and `custom`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamMode {
    /// Standard stream data defined by the framework.
    #[default]
    Output,
    /// Trace stream data produced by the graph.
    Trace,
    /// Custom stream data defined by the runnable.
    Custom,
}

impl StreamMode {
    /// The `snake_case` wire name of this mode.
    ///
    /// This is the value used in [`StreamItem`] serialization and in log
    /// fields, and it matches the Python mode values exactly.
    #[must_use]
    #[allow(clippy::trivially_copy_pass_by_ref)] // signature fixed by the crate's API surface
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Output => "output",
            Self::Trace => "trace",
            Self::Custom => "custom",
        }
    }
}

/// One framed item published on a session stream.
///
/// Python has three schema classes (`OutputSchema` with `index`,
/// `TraceSchema` without it, and the open-ended `CustomSchema`); Rust uses one
/// struct whose `mode` selects the intended consumer and whose `index` is
/// optional. The wire projection produced by [`StreamItem::to_wire`] drops the
/// routing mode, so the JSON payload keeps Python's field names.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct StreamItem {
    /// Routing mode; never part of the wire payload.
    pub mode: StreamMode,
    /// Frame type, serialized as `"type"` for wire parity (`llm_output`,
    /// `tracer_agent`, `__interaction__`, …).
    #[serde(rename = "type")]
    pub item_type: String,
    /// Position within the frame's logical sequence, when the producer has one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub index: Option<i64>,
    /// Frame payload; must not be [`Value::Null`] when sent through a hub.
    pub payload: Value,
}

impl StreamItem {
    /// Build an output-mode frame.
    ///
    /// `index` is `None` for producers that do not number their frames.
    #[must_use]
    pub fn output(item_type: impl Into<String>, index: Option<i64>, payload: Value) -> Self {
        Self {
            mode: StreamMode::Output,
            item_type: item_type.into(),
            index,
            payload,
        }
    }

    /// Build an indexed output-mode frame.
    ///
    /// Shorthand for [`StreamItem::output`] with `Some(index)`.
    #[must_use]
    pub fn output_at(item_type: impl Into<String>, index: i64, payload: Value) -> Self {
        Self::output(item_type, Some(index), payload)
    }

    /// Build a trace-mode frame.
    ///
    /// Trace frames are unindexed, matching Python's `TraceSchema`.
    #[must_use]
    pub fn trace(item_type: impl Into<String>, payload: Value) -> Self {
        Self {
            mode: StreamMode::Trace,
            item_type: item_type.into(),
            index: None,
            payload,
        }
    }

    /// Build a custom-mode frame with the frame type `"custom"`.
    #[must_use]
    pub fn custom(payload: Value) -> Self {
        Self {
            mode: StreamMode::Custom,
            item_type: "custom".to_owned(),
            index: None,
            payload,
        }
    }

    /// Replace the routing mode, keeping type, index, and payload.
    #[must_use]
    pub fn with_mode(mut self, mode: StreamMode) -> Self {
        self.mode = mode;
        self
    }

    /// Whether this frame is the [`END_FRAME`] completion sentinel.
    #[must_use]
    pub fn is_end(&self) -> bool {
        self.item_type == END_FRAME
    }

    /// The wire projection of this frame.
    ///
    /// The returned object always has `"type"` and `"payload"`, and has
    /// `"index"` only when the frame is indexed. The routing mode is omitted:
    /// Python consumers only ever see `{"type", "index", "payload"}`.
    #[must_use]
    pub fn to_wire(&self) -> Value {
        let mut map = Map::new();
        map.insert("type".to_owned(), Value::String(self.item_type.clone()));
        if let Some(index) = self.index {
            map.insert("index".to_owned(), Value::from(index));
        }
        map.insert("payload".to_owned(), self.payload.clone());
        Value::Object(map)
    }
}

/// Event delivered to a [`StreamHub`] subscriber.
///
/// Serializes with an internal `"event"` tag: `{"event": "item", ...}` for a
/// frame and `{"event": "end"}` for the completion marker.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum StreamEvent {
    /// A framed item.
    Item(StreamItem),
    /// The stream is complete; no further items follow.
    End,
}

/// Bounded, multi-subscriber transport for [`StreamItem`] frames.
///
/// A hub owns a `tokio` broadcast channel plus a closed flag. Producers call
/// [`StreamHub::send_now`] (or the `async` [`StreamHub::send`] wrapper);
/// consumers call [`StreamHub::subscribe`] and receive ordered events. Sends
/// never block: a full per-subscriber buffer makes that subscriber skip its
/// oldest frames with a warning, so a slow consumer cannot stall the workflow.
///
/// A hub without subscribers accepts and drops frames; that is the normal case
/// for workflows that are not being observed.
#[derive(Debug)]
pub struct StreamHub {
    sender: broadcast::Sender<StreamEvent>,
    capacity: usize,
    closed: AtomicBool,
}

impl StreamHub {
    /// Create a hub with the default capacity ([`DEFAULT_STREAM_CAPACITY`]).
    #[must_use]
    pub fn new() -> Arc<Self> {
        Self::with_capacity(DEFAULT_STREAM_CAPACITY)
    }

    /// Create a hub buffering at most `capacity` frames per subscriber.
    ///
    /// A capacity of `0` is meaningless for a bounded channel, so it is raised
    /// to `1` with a debug log rather than panicking.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Arc<Self> {
        let capacity = if capacity == 0 {
            tracing::debug!("stream capacity 0 is raised to 1");
            1
        } else {
            capacity
        };
        let (sender, _) = broadcast::channel(capacity);
        Arc::new(Self {
            sender,
            capacity,
            closed: AtomicBool::new(false),
        })
    }

    /// The per-subscriber buffer capacity of this hub.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Publish a frame to every current subscriber, without awaiting.
    ///
    /// This is the canonical send path. It performs no I/O and never awaits, so
    /// it is safe to call from synchronous code (a trace sink emitting frames
    /// from a plain function, for example) as well as from async tasks. The
    /// underlying transport is `tokio::sync::broadcast::Sender::send`, which is
    /// itself synchronous.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when the payload is [`Value::Null`]
    /// (Python error 111132). Sending to a closed hub, or to a hub without
    /// subscribers, is not an error: the frame is dropped and `Ok(())` is
    /// returned.
    pub fn send_now(&self, item: StreamItem) -> Result<()> {
        if item.payload.is_null() {
            return Err(Error::validation("stream payload is null"));
        }
        if self.is_closed() {
            tracing::debug!(
                item_type = item.item_type.as_str(),
                "dropping stream item: stream hub is closed"
            );
            return Ok(());
        }
        match self.sender.send(StreamEvent::Item(item)) {
            // `tokio::sync::broadcast` reports exactly one failure mode: there
            // were no active receivers. Dropping the frame is the documented
            // behavior for an unobserved stream, not an error.
            Ok(_receivers) => Ok(()),
            Err(broadcast::error::SendError(_)) => Ok(()),
        }
    }

    /// Publish a frame to every current subscriber.
    ///
    /// Async wrapper around [`StreamHub::send_now`]: nothing is awaited
    /// internally, so it completes on the first poll. It is kept `async` for
    /// API symmetry with Python's `await writer.write(...)` and so a future
    /// backend can introduce real await points without a breaking change.
    /// Synchronous producers should call [`StreamHub::send_now`] instead.
    ///
    /// # Errors
    ///
    /// Same as [`StreamHub::send_now`]: [`Error::Validation`] for a `Null`
    /// payload; a closed hub or a hub without subscribers drops the frame and
    /// returns `Ok(())`.
    // The missing `.await` is intentional: the wrapper exists for API symmetry.
    #[allow(clippy::unused_async, clippy::unused_async_trait_impl)]
    pub async fn send(&self, item: StreamItem) -> Result<()> {
        self.send_now(item)
    }

    /// Subscribe to this hub.
    ///
    /// The returned stream yields every event published after the call, in
    /// order, and finishes after [`StreamEvent::End`]. A subscriber created
    /// after [`StreamHub::close`] yields exactly one `End` and finishes; a
    /// subscriber that falls behind loses frames (with a logged warning) but
    /// never terminates the producer.
    #[must_use = "the returned stream does nothing unless polled"]
    pub fn subscribe(&self) -> BoxStream<'static, Result<StreamEvent>> {
        // Subscribe before sampling the flag: a subscriber created in between
        // still observes the `End` broadcast, and one created after close
        // observes the flag and yields its own `End`.
        let mut receiver = self.sender.subscribe();
        let already_closed = self.is_closed();
        Box::pin(stream! {
            if already_closed {
                yield Ok(StreamEvent::End);
                return;
            }
            loop {
                match receiver.recv().await {
                    Ok(StreamEvent::End) | Err(RecvError::Closed) => {
                        yield Ok(StreamEvent::End);
                        return;
                    }
                    Ok(event) => yield Ok(event),
                    Err(RecvError::Lagged(skipped)) => {
                        tracing::warn!(skipped, "stream subscriber lagged");
                    }
                }
            }
        })
    }

    /// Close the hub and publish the completion marker.
    ///
    /// Idempotent: the first call sets the closed flag and broadcasts
    /// [`StreamEvent::End`]; later calls do nothing. Send failures (no
    /// subscribers left) are ignored.
    pub fn close(&self) {
        if self.closed.swap(true, Ordering::AcqRel) {
            return;
        }
        if self.sender.send(StreamEvent::End).is_err() {
            tracing::debug!("dropping stream end marker: no subscribers");
        }
    }

    /// Whether [`StreamHub::close`] has been called.
    #[must_use]
    pub fn is_closed(&self) -> bool {
        self.closed.load(Ordering::Acquire)
    }
}

/// Mode-routed writer over one [`StreamHub`].
///
/// Ports Python's `StreamWriterManager`: it holds the enabled modes and drops
/// frames whose mode is disabled, so a workflow can turn trace traffic off
/// without changing producers. `new` enables all three modes; `with_modes`
/// narrows the set.
#[derive(Debug, Clone)]
pub struct StreamWriterManager {
    hub: Arc<StreamHub>,
    modes: Vec<StreamMode>,
}

impl StreamWriterManager {
    /// Create a manager writing every mode to `hub`.
    #[must_use]
    pub fn new(hub: Arc<StreamHub>) -> Arc<Self> {
        Self::with_modes(
            hub,
            vec![StreamMode::Output, StreamMode::Trace, StreamMode::Custom],
        )
    }

    /// Create a manager enabling only `modes`.
    ///
    /// Duplicates are removed while the caller's order is preserved.
    #[must_use]
    pub fn with_modes(hub: Arc<StreamHub>, modes: Vec<StreamMode>) -> Arc<Self> {
        let mut unique: Vec<StreamMode> = Vec::with_capacity(modes.len());
        for mode in modes {
            if !unique.contains(&mode) {
                unique.push(mode);
            }
        }
        Arc::new(Self { hub, modes: unique })
    }

    /// Whether frames of `mode` are written to the hub.
    #[must_use]
    pub fn enables(&self, mode: StreamMode) -> bool {
        self.modes.contains(&mode)
    }

    /// The enabled modes, in configuration order.
    #[must_use]
    pub fn modes(&self) -> &[StreamMode] {
        &self.modes
    }

    /// The hub this manager writes to.
    #[must_use]
    pub fn hub(&self) -> &Arc<StreamHub> {
        &self.hub
    }

    /// Write `item` to the hub when its mode is enabled, without awaiting.
    ///
    /// This is the canonical write path; see [`StreamHub::send_now`] for why it
    /// is safe to call from synchronous code.
    ///
    /// # Errors
    ///
    /// Propagates [`StreamHub::send_now`] errors (a `Null` payload is
    /// [`Error::Validation`]). Writing a disabled mode is not an error: the
    /// frame is dropped and `Ok(())` is returned.
    pub fn write_now(&self, item: StreamItem) -> Result<()> {
        if !self.enables(item.mode) {
            tracing::debug!(
                mode = item.mode.as_str(),
                "dropping stream item: mode disabled"
            );
            return Ok(());
        }
        self.hub.send_now(item)
    }

    /// Write `item` to the hub when its mode is enabled.
    ///
    /// Async wrapper around [`StreamWriterManager::write_now`]: nothing is
    /// awaited internally, so it completes on the first poll. It is kept
    /// `async` for API symmetry with Python's `await writer.write(...)` and so
    /// a future backend can introduce real await points without a breaking
    /// change. Synchronous producers should call
    /// [`StreamWriterManager::write_now`] instead.
    ///
    /// # Errors
    ///
    /// Same as [`StreamWriterManager::write_now`]: [`Error::Validation`] for a
    /// `Null` payload; a disabled mode, a closed hub, or a hub without
    /// subscribers drops the frame and returns `Ok(())`.
    // The missing `.await` is intentional: the wrapper exists for API symmetry.
    #[allow(clippy::unused_async, clippy::unused_async_trait_impl)]
    pub async fn write(&self, item: StreamItem) -> Result<()> {
        self.write_now(item)
    }

    /// Subscribe to the underlying hub; see [`StreamHub::subscribe`].
    #[must_use = "the returned stream does nothing unless polled"]
    pub fn subscribe(&self) -> BoxStream<'static, Result<StreamEvent>> {
        self.hub.subscribe()
    }

    /// Close the underlying hub; see [`StreamHub::close`].
    pub fn close(&self) {
        self.hub.close();
    }

    /// Whether the underlying hub is closed.
    #[must_use]
    pub fn is_closed(&self) -> bool {
        self.hub.is_closed()
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::sync::Arc;
    use std::time::Duration;

    use futures::stream::BoxStream;
    use futures::StreamExt;
    use serde_json::{json, Value};

    use openjiuwen_common::error::{ErrorKind, Result};

    use super::{StreamEvent, StreamHub, StreamItem, StreamMode, StreamWriterManager};

    /// Upper bound on how long a deterministic in-memory receive may take.
    const RECV_TIMEOUT: Duration = Duration::from_secs(1);

    /// Receive one event, failing the test instead of hanging when none arrives.
    async fn recv(stream: &mut BoxStream<'static, Result<StreamEvent>>) -> StreamEvent {
        tokio::time::timeout(RECV_TIMEOUT, stream.next())
            .await
            .expect("a stream event arrives before the timeout")
            .expect("the stream is still open")
            .expect("the stream yields Ok events")
    }

    #[tokio::test]
    async fn send_now_delivers_from_a_synchronous_context() {
        let hub = StreamHub::new();
        let mut stream = hub.subscribe();

        // Publishing from a plain OS thread proves the synchronous path needs no
        // async context at all.
        let sender = Arc::clone(&hub);
        std::thread::spawn(move || {
            sender
                .send_now(StreamItem::output_at(
                    "llm_output",
                    0,
                    json!({ "delta": "a" }),
                ))
                .expect("the synchronous send path accepts the frame");
        })
        .join()
        .expect("the sending thread finishes");

        match recv(&mut stream).await {
            StreamEvent::Item(item) => {
                assert_eq!(item.mode, StreamMode::Output);
                assert_eq!(item.item_type, "llm_output");
                assert_eq!(item.index, Some(0));
                assert_eq!(item.payload, json!({ "delta": "a" }));
            }
            StreamEvent::End => panic!("the end marker must not precede the frame"),
        }
    }

    #[tokio::test]
    async fn write_now_drops_disabled_modes_and_rejects_null_payloads() {
        let hub = StreamHub::new();
        let manager = StreamWriterManager::with_modes(hub.clone(), vec![StreamMode::Output]);
        let mut stream = manager.subscribe();

        manager
            .write_now(StreamItem::trace("tracer_agent", json!({ "invoke": 1 })))
            .expect("a disabled trace mode is a no-op");
        manager
            .write_now(StreamItem::custom(json!({ "k": "v" })))
            .expect("a disabled custom mode is a no-op");
        manager
            .write_now(StreamItem::output_at(
                "llm_output",
                0,
                json!({ "delta": "hi" }),
            ))
            .expect("the enabled mode is written");
        manager.close();

        // The disabled frames never reach the hub: the output frame is first.
        match recv(&mut stream).await {
            StreamEvent::Item(item) => {
                assert_eq!(item.mode, StreamMode::Output);
                assert_eq!(item.item_type, "llm_output");
                assert_eq!(item.payload, json!({ "delta": "hi" }));
            }
            StreamEvent::End => panic!("the enabled output frame must be delivered"),
        }
        assert_eq!(recv(&mut stream).await, StreamEvent::End);

        let error = hub
            .send_now(StreamItem::output("llm_output", None, Value::Null))
            .expect_err("a null payload is rejected");
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert_eq!(error.to_string(), "validation: stream payload is null");
    }
}
