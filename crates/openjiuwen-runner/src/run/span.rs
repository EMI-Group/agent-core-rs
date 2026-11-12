//! The crate-private [`RunSpan`]: a run's mirror on the session tracer.
//!
//! Python's runner lets the session tracer report one chain span per agent,
//! workflow, or team invocation; the Rust port keeps that behaviour but never
//! lets a tracing problem affect the run. A session without a tracer (the
//! common case for a bare [`openjiuwen_session::Session`]) makes every method a
//! no-op.

use openjiuwen_common::error::Error;
use openjiuwen_session::{InvokeType, SpanHandle, Tracer};
use serde_json::Value;

/// One chain span opened for a run, when the session exposes a tracer.
///
/// Starting records the run inputs; [`RunSpan::finish`] and [`RunSpan::fail`]
/// end the span exactly once (a second call is ignored by
/// [`SpanHandle`]). No method can fail or panic because of the tracer.
pub(crate) struct RunSpan {
    /// The opened span, or `None` when the session has no tracer.
    handle: Option<SpanHandle>,
}

impl RunSpan {
    /// Opens a `Chain` span named `name` on `tracer`, recording `inputs`.
    ///
    /// A missing tracer yields an inert span, so callers never branch.
    pub(crate) fn start(tracer: Option<&Tracer>, name: &str, inputs: Value) -> Self {
        let handle = tracer.map(|tracer| {
            let span = tracer.start_span(name, InvokeType::Chain);
            span.set_inputs(inputs);
            span
        });
        Self { handle }
    }

    /// Finishes the span successfully with `outputs`.
    pub(crate) fn finish(&self, outputs: Value) {
        if let Some(handle) = &self.handle {
            handle.finish(outputs);
        }
    }

    /// Finishes the span as failed with the rendered `error`.
    pub(crate) fn fail(&self, error: &Error) {
        if let Some(handle) = &self.handle {
            handle.fail(error.to_string());
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use openjiuwen_session::MemoryTraceSink;
    use serde_json::json;
    use std::sync::Arc;

    #[test]
    fn without_a_tracer_every_method_is_a_no_op() {
        let span = RunSpan::start(None, "run_agent", json!({"query": "hi"}));
        assert!(span.handle.is_none());
        span.finish(json!({"output": "ok"}));
        span.fail(&Error::validation("boom"));
    }

    #[test]
    fn with_a_tracer_the_span_is_opened_and_finished() {
        let sink = Arc::new(MemoryTraceSink::new());
        let tracer = Tracer::with_sink("session", "trace", sink.clone());
        let span = RunSpan::start(Some(&tracer), "run_agent", json!({"query": "hi"}));
        assert!(span.handle.is_some());
        span.finish(json!({"output": "ok"}));
        assert_eq!(sink.len(), 1);
        // A second terminal transition is ignored by the span handle.
        span.fail(&Error::validation("late"));
    }
}
