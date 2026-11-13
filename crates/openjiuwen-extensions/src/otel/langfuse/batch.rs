//! Ordered ingestion batches.
//!
//! [`IngestionBatch`] is the buffer a caller fills before handing it to
//! [`LangfuseClient::ingest`]; it preserves insertion order (Langfuse correlates
//! observations through the ids inside the events, but a stable order keeps the
//! wire body and the tests readable) and never drops events. Splitting a batch
//! into requests is the client's job, which is why the size limit lives there as
//! `LangfuseClient::with_max_batch_size`.
//!
//! ```
//! use openjiuwen_extensions::otel::langfuse::{
//!     IngestionBatch, IngestionEvent, TraceBody, DEFAULT_MAX_BATCH_SIZE,
//! };
//!
//! let mut batch = IngestionBatch::new();
//! batch.push(IngestionEvent::trace(&TraceBody::new("checkout"))?);
//! assert_eq!(batch.len(), 1);
//! assert_eq!(batch.into_events().len(), 1);
//! assert_eq!(DEFAULT_MAX_BATCH_SIZE, 1000);
//! # Ok::<(), openjiuwen_common::error::Error>(())
//! ```
//!
//! [`LangfuseClient::ingest`]: super::LangfuseClient::ingest

use super::events::IngestionEvent;

/// Number of events Langfuse accepts per ingestion request, and the batch size
/// a fresh [`LangfuseClient`] uses when nothing else is configured.
///
/// The Langfuse REST ingestion API documents 1000 events per request; larger
/// batches are split by the client rather than truncated.
///
/// [`LangfuseClient`]: super::LangfuseClient
pub const DEFAULT_MAX_BATCH_SIZE: usize = 1000;

/// Ordered buffer of [`IngestionEvent`]s.
///
/// ```
/// use openjiuwen_extensions::otel::langfuse::{
///     IngestionBatch, IngestionEvent, SpanBody,
/// };
///
/// let mut batch = IngestionBatch::new();
/// batch.push(IngestionEvent::span(&SpanBody::new("trace-1", "retrieve"))?);
/// assert!(!batch.is_empty());
/// assert_eq!(batch.events()[0].body["name"], "retrieve");
/// # Ok::<(), openjiuwen_common::error::Error>(())
/// ```
#[derive(Debug, Clone, Default, PartialEq)]
pub struct IngestionBatch {
    /// Events in insertion order.
    events: Vec<IngestionEvent>,
}

impl IngestionBatch {
    /// Builds an empty batch.
    #[must_use]
    pub const fn new() -> Self {
        Self { events: Vec::new() }
    }

    /// Builds an empty batch with room for `capacity` events.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            events: Vec::with_capacity(capacity),
        }
    }

    /// Appends one event, keeping the insertion order.
    pub fn push(&mut self, event: IngestionEvent) {
        self.events.push(event);
    }

    /// Appends every event of `events`, keeping their order.
    pub fn extend(&mut self, events: impl IntoIterator<Item = IngestionEvent>) {
        self.events.extend(events);
    }

    /// Reports how many events the batch holds.
    #[must_use]
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Reports whether the batch holds no event.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Borrows the buffered events.
    #[must_use]
    pub fn events(&self) -> &[IngestionEvent] {
        &self.events
    }

    /// Consumes the batch, returning the buffered events in order.
    #[must_use]
    pub fn into_events(self) -> Vec<IngestionEvent> {
        self.events
    }

    /// Reports the request size limit Langfuse documents
    /// ([`DEFAULT_MAX_BATCH_SIZE`], 1000 events per request).
    #[must_use]
    pub const fn max_size_hint() -> usize {
        DEFAULT_MAX_BATCH_SIZE
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::otel::langfuse::{IngestionEventType, TraceBody};

    #[test]
    fn empty_batch_is_empty() {
        let batch = IngestionBatch::new();
        assert!(batch.is_empty());
        assert_eq!(batch.len(), 0);
        assert_eq!(batch.into_events().len(), 0);
    }

    #[test]
    fn push_and_extend_keep_the_insertion_order() {
        let mut batch = IngestionBatch::with_capacity(4);
        batch.push(IngestionEvent::trace(&TraceBody::new("first")).unwrap());
        batch.extend([
            IngestionEvent::trace(&TraceBody::new("second")).unwrap(),
            IngestionEvent::trace(&TraceBody::new("third")).unwrap(),
        ]);
        assert_eq!(batch.len(), 3);
        let names: Vec<&str> = batch
            .events()
            .iter()
            .map(|event| event.body["name"].as_str().unwrap())
            .collect();
        assert_eq!(names, ["first", "second", "third"]);
        assert_eq!(
            batch.events()[0].event_type,
            IngestionEventType::TraceCreate
        );
        assert_eq!(batch.into_events().len(), 3);
    }

    #[test]
    fn default_max_size_is_the_langfuse_request_limit() {
        assert_eq!(IngestionBatch::max_size_hint(), 1000);
        assert_eq!(DEFAULT_MAX_BATCH_SIZE, 1000);
    }
}
