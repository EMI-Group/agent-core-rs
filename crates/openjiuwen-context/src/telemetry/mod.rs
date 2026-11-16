//! Compression telemetry: `context.compression_state` records.
//!
//! Ports `schema/context_state.py`, `context/processor_state_recorder.py` and
//! `context/compression_scope.py`: the compression-state value objects, the
//! recorder that builds and emits them per processor pass, and the
//! request-purpose correlation scope for processor model calls.
//!
//! - [`state`] — the wire contract: [`ContextCompressionState`] and its parts,
//!   the [`CompressionStateStatus`]/[`CompressionStatePhase`] vocabularies, and
//!   the [`CONTEXT_COMPRESSION_STATE_TYPE`] frame type.
//! - [`recorder`] — [`ContextProcessorStateRecorder`], which turns a
//!   [`ContextProcessorStateInput`] into a record, keeps the last
//!   [`DEFAULT_HISTORY_LIMIT`] of them, and fans them out to a
//!   [`CompressionStateSink`] and to the session stream. Every number it
//!   publishes carries a [`TokenProvenance`], so a reader can tell a count from
//!   an estimate.
//! - [`scope`] — [`with_operation`], the task-local binding that makes the model
//!   calls of a compaction pass attributable, plus the request stamping helpers.
//!
//! ```
//! use openjiuwen_context::telemetry::{
//!     CompressionStatePhase, CompressionStateStatus, ContextProcessorStateInput,
//!     ContextProcessorStateRecorder,
//! };
//! use openjiuwen_foundation::prompt::BaseMessage;
//!
//! # async fn example() {
//! let recorder = ContextProcessorStateRecorder::new("session-1", "main");
//! let before = vec![BaseMessage::user("hi"), BaseMessage::assistant("hello")];
//! let started = recorder
//!     .emit(ContextProcessorStateInput::new(
//!         "op-1",
//!         CompressionStateStatus::Started,
//!         CompressionStatePhase::ActiveCompress,
//!         "dialogue",
//!         &before,
//!     ))
//!     .await;
//! assert_eq!(started.operation_id, "op-1");
//! assert_eq!(started.before.messages, 2);
//! assert_eq!(recorder.history().len(), 1);
//! # }
//! ```
pub mod recorder;
pub mod scope;
pub mod state;

pub use recorder::{
    format_timestamp, modified_messages, CompressionStateSink, ContextProcessorStateInput,
    ContextProcessorStateRecorder, MeasuredTokens, RecordingSink, TokenProvenance,
    DEFAULT_HISTORY_LIMIT,
};
pub use scope::{
    apply_operation, current_compression_operation, current_compression_operation_id,
    current_request_purpose, stamp_current_operation, with_operation, with_operation_scope,
    CompressionOperation, COMPACTION_PURPOSE, CONTEXT_OPERATION_ID_KEY, REQUEST_PURPOSE_KEY,
};
pub use state::{
    CompressionStatePhase, CompressionStateStatus, ContextCompressionMetric,
    ContextCompressionModifiedMessage, ContextCompressionSaved, ContextCompressionState,
    ContextCompressionUsage, CONTEXT_COMPRESSION_STATE_TYPE,
};
