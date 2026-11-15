//! Persistent (long-lived, resumable) agent support: execution log, session state, window
//! management, continuation and handoff seams.
//!
//! Ports the self-contained subset of `openjiuwen/harness/persistent_agent/execution/`.
//! The module owns the state that survives one execution — an append-only JSONL journal, the
//! persisted session record, the reconstructed model window and the transfer records a successor
//! or a share bundle consumes — and owns nothing else: no agent loop, no model call, no host
//! queueing or delivery.
//!
//! Every entry point takes its clock and root directory from the caller, so the module is
//! deterministic and offline in tests. The `CONTEXT.md` file in this directory carries the API
//! contract and routing table; the module-level [`Clock`] is the crate's existing injectable
//! clock ([`crate::subagents::Clock`]) re-exported here.
pub mod catalog;
pub mod continuation;
pub mod control;
pub mod emitter;
pub mod entry;
pub mod handoff;
pub mod identity;
pub mod layout;
pub mod log;
pub mod rails;
pub mod recording;
pub mod session;
pub mod share_file;
pub mod util;
pub mod window;

pub use crate::subagents::{rfc3339_from_ms, Clock, ManualClock, SystemClock};

pub use catalog::{ExecutionCatalog, ExecutionSummary};
pub use continuation::{
    continuation_background, resume, Continuation, ExecutionLink, ExecutionLinkSource,
    ExecutionLogSource, FileLinkStore, LogPreview, ResumePlan, RootLogSource, CONTINUATION_RULES,
};
pub use control::{AbortReason, Control, ControlSnapshot};
pub use emitter::{
    EmittedEvent, Emitter, EventSink, EventType, InMemorySink, InterceptedCall, PersistentEvent,
};
pub use entry::{
    args_summary, operation_id, params_digest, redact_arguments, result_summary, EntryKind,
    EntryPayload, ExecutionEntry, FinalStatus, ModelUsage, PauseReason, ToolStatus, UsageStatus,
    DETAIL_LIMIT, ENTRY_SCHEMA_VERSION, SUMMARY_LIMIT,
};
pub use handoff::{
    hand_off, DistributedHandoff, HandoffReceipt, HandoffTarget, HandoffTransfer, HandoffTransport,
    LocalHandoff, RemoteHandoff,
};
pub use identity::{ExecutionIdentity, LinkRelation, SessionBinding, SESSION_KEY_PREFIX};
pub use layout::{PersistentAgentLayout, LINKS_DIR, LOGS_DIR, PERSISTENT_AGENT_NODE, SESSIONS_DIR};
pub use log::{
    ExecutionLog, LogCapacity, LogCursor, LogDigest, LogMeta, LogSnapshot, Replay, LOG_ENCODING,
    LOG_ENTRY_FILE, LOG_META_FILE,
};
pub use rails::{
    deterministic_model_call_id, is_read_only_tool, ExecutionRecordRail, ModelAdmission,
    ModelCallGate, ModelCallRail, ToolRecordPolicy, MODEL_CALL_ID_EXTRA_KEY, READ_ONLY_TOOLS,
};
pub use recording::{
    fresh_id, ActiveExecution, ExecutionRecorder, PersistentRunConfig, DEFAULT_NAMESPACE_ID,
    DEFAULT_SOURCE_REF, SESSION_ENDING_OUTCOME, USAGE_UNKNOWN_REASON,
};
pub use session::{
    patch_stopped_calls, CheckpointKind, ContinuationCursor, FileSessionStore,
    ParticipationDecision, ParticipationState, PersistentSessionRecord, RecordCheckpoint,
    RecordEnding, RecordRollover, RolloverTrigger, SessionStateStore, SessionSummary, WaitingScene,
    HANDOFF_TEXT_PREFIX, PLACEHOLDER_TEXT, RECORD_INVALID_PREFIX, SESSION_RECORD_VERSION,
};
pub use share_file::{
    is_logical_path, valid_caption, ImportedShareBundle, ShareBundle, ShareFile, ShareMode,
    SharedFileEntry, BUNDLE_FILE, CAPTION_MAX_CHARS, FILE_PATH_MAX_CHARS, MAX_SHARED_FILES,
    SHARE_BUNDLE_SCHEMA_VERSION, SHARE_FILE_MAX_BYTES, SHARE_MODES,
};
pub use util::{
    atomic_write, canonical_json, digest_hex, is_valid_identifier, percent_decode_identifier,
    percent_encode_identifier, read_json, sha256_hex_str, truncate_chars, write_json_atomic,
    IDENTIFIER_MAX_CHARS,
};
pub use window::{
    Capacity, CompletionFn, CompletionSummarizer, ElidedEntry, ElisionReason, ExtractiveSummarizer,
    UnsupportedModelSummarizer, WindowAssembler, WindowAssembly, WindowManager, WindowPlan,
    WindowPolicy, WindowSummarizer,
};

#[cfg(test)]
mod tests;
