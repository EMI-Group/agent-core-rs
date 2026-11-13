//! A session-owned trace sink that can be swapped after construction.
//!
//! Python's `BaseSession.set_tracer` lets a product layer replace the tracer of
//! a live session (for example to attach the `OTel` handler family). The Rust
//! port keeps one stable [`TraceSink`] reference per session — [`crate::Session::trace`]
//! hands it out and checkpoints serialize its position — so "replacing the
//! sink" is realized with an indirection instead of a mutable field:
//! [`SwappableTraceSink`] owns an `Arc<dyn TraceSink>` behind an `RwLock`, the
//! session and its tracer both hold the wrapper, and
//! [`SwappableTraceSink::install`] swaps the inner sink.
//!
//! Every accessor clones the current sink out of the lock before using it, so
//! an installed sink that emits (or installs) recursively never deadlocks.
//!
//! ```
//! use std::sync::Arc;
//! use openjiuwen_session::trace_hook::SwappableTraceSink;
//! use openjiuwen_session::tracer::{MemoryTraceSink, TraceSink};
//!
//! let sink = SwappableTraceSink::default();
//! let memory: Arc<MemoryTraceSink> = Arc::new(MemoryTraceSink::new());
//! sink.install(Arc::clone(&memory) as Arc<dyn TraceSink>);
//! assert!(sink.current().position().is_none());
//! ```
use std::sync::Arc;

use parking_lot::RwLock;

use crate::tracer::{NoopTraceSink, TraceFrame, TracePosition, TraceSink};

/// A [`TraceSink`] forwarding to an inner sink that can be replaced at runtime.
///
/// `Default` starts with a [`NoopTraceSink`], so a wrapper created without an
/// explicit sink records nothing until one is installed.
#[derive(Debug)]
pub struct SwappableTraceSink {
    /// The currently installed sink; read-cloned on every operation.
    inner: RwLock<Arc<dyn TraceSink>>,
}

impl Default for SwappableTraceSink {
    fn default() -> Self {
        Self::new(Arc::new(NoopTraceSink::new()))
    }
}

impl SwappableTraceSink {
    /// Wraps `inner` as the initially installed sink.
    #[must_use]
    pub fn new(inner: Arc<dyn TraceSink>) -> Self {
        Self {
            inner: RwLock::new(inner),
        }
    }

    /// Returns a handle to the currently installed sink.
    ///
    /// The lock is released before the handle is used, so the returned sink can
    /// itself install a replacement without blocking on this wrapper.
    #[must_use]
    pub fn current(&self) -> Arc<dyn TraceSink> {
        Arc::clone(&self.inner.read())
    }

    /// Replaces the installed sink for every frame emitted afterwards.
    ///
    /// Spans already started are unaffected in their identity but do route
    /// through this wrapper, so they also observe the new sink.
    pub fn install(&self, sink: Arc<dyn TraceSink>) {
        *self.inner.write() = sink;
    }
}

impl TraceSink for SwappableTraceSink {
    fn emit(&self, frame: TraceFrame) {
        self.current().emit(frame);
    }

    fn position(&self) -> Option<TracePosition> {
        self.current().position()
    }

    fn set_position(&self, position: TracePosition) {
        self.current().set_position(position);
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::sync::Arc;

    use super::SwappableTraceSink;
    use crate::tracer::{
        InvokeType, MemoryTraceSink, NoopTraceSink, TraceFrame, TracePosition, TraceSink, Tracer,
    };
    use serde_json::Value;

    #[test]
    fn default_sink_records_nothing() {
        let wrapper = SwappableTraceSink::default();
        let span = Tracer::with_sink("s1", "trace-0", wrapper.current())
            .start_span("noop", InvokeType::Prompt);
        wrapper.emit(TraceFrame::agent(span.snapshot()));
        span.finish(Value::Null);
        assert!(wrapper.position().is_none());
        assert_eq!(
            format!("{:?}", wrapper.current()),
            format!("{:?}", Arc::new(NoopTraceSink::new()) as Arc<dyn TraceSink>)
        );
    }

    #[test]
    fn install_redirects_frames_and_position() {
        let sink = Arc::new(SwappableTraceSink::default());
        let tracer = Tracer::with_sink("s1", "trace-1", Arc::clone(&sink) as Arc<dyn TraceSink>);

        let span = tracer.start_span("before", InvokeType::Prompt);
        span.finish(Value::Null);
        assert!(sink.position().is_none(), "the noop sink tracks nothing");

        let memory = Arc::new(MemoryTraceSink::new());
        sink.install(Arc::clone(&memory) as Arc<dyn TraceSink>);

        let span = tracer.start_span("after", InvokeType::Prompt);
        span.finish(Value::Null);
        assert_eq!(memory.len(), 1);
        assert_eq!(sink.position().map(|p| p.frame_count), Some(1));
    }

    #[test]
    fn set_position_reaches_the_current_sink() {
        let wrapper = SwappableTraceSink::default();
        let memory = Arc::new(MemoryTraceSink::new());
        wrapper.install(Arc::clone(&memory) as Arc<dyn TraceSink>);
        wrapper.set_position(TracePosition {
            trace_id: "trace-2".to_string(),
            last_invoke_id: None,
            frame_count: 7,
        });
        assert_eq!(memory.position().map(|p| p.frame_count), Some(7));
        assert_eq!(
            wrapper.position().map(|p| p.trace_id),
            Some("trace-2".to_string())
        );
    }

    #[test]
    fn emitting_through_a_recording_sink_is_visible() {
        let wrapper = SwappableTraceSink::default();
        let memory = Arc::new(MemoryTraceSink::new());
        wrapper.install(Arc::clone(&memory) as Arc<dyn TraceSink>);
        let tracer = Tracer::with_sink(
            "s1",
            "trace-3",
            Arc::new(SwappableTraceSink::new(
                Arc::clone(&memory) as Arc<dyn TraceSink>
            )),
        );
        let span = tracer.start_span("x", InvokeType::Prompt);
        let frame = span.snapshot();
        wrapper.emit(TraceFrame::agent(frame));
        assert_eq!(memory.len(), 1);
    }
}
