//! Sessions, state scopes, streaming, and human-in-the-loop interactions.
//!
//! Maps to the Python package `openjiuwen.core.session` in the openJiuwen
//! reference implementation. The crate is the kernel's stable seam for
//! per-run execution context: state lanes, stream frames, and interaction
//! requests/responses, checkpoint persistence, and trace spans and frames.
//!
//! Modules:
//!
//! - [`config`] — [`config::SessionConfig`], built-in defaults and environment overrides.
//! - [`constants`] — environment key names and default ids/frame constants.
//! - [`state`] — state stores ([`state::WorkflowState`], [`state::AgentState`])
//!   and nested JSON path helpers.
//! - [`stream`] — stream frames ([`stream::StreamItem`]) and delivery
//!   ([`stream::StreamHub`], [`stream::StreamWriterManager`]); every publish
//!   reports a [`stream::StreamSendOutcome`], so an open hub with no
//!   subscribers answers `NoSubscribers` instead of silently dropping the frame.
//! - [`interaction`] — human-in-the-loop requests, responses, and the
//!   [`interaction::InteractionBroker`].
//! - [`session`] — the [`session::Session`] contract every graph node,
//!   workflow component, agent, team, and runner receives as `Arc<dyn Session>`:
//!   state scopes, streaming, human-in-the-loop, tracing, and snapshot/restore.
//! - [`sessions`] — concrete sessions ([`sessions::WorkflowSession`],
//!   [`sessions::NodeSession`], [`sessions::AgentSession`],
//!   [`sessions::AgentTeamSession`], [`sessions::SubWorkflowSession`],
//!   [`sessions::RouterSession`]) and their construction factories
//!   ([`sessions::create_workflow_session`],
//!   [`sessions::create_agent_session`],
//!   [`sessions::create_agent_team_session`]).
//! - [`trace_hook`] — [`trace_hook::SwappableTraceSink`], the swappable sink a
//!   session owns so [`session::Session::install_trace_sink`] can reroute
//!   tracing after construction.
//! - [`otel`] — the feature-gated [`otel::install_trace_sink`] compatibility
//!   helper plus [`otel::try_install_trace_sink`], which reports what happened
//!   as a [`otel::TraceSinkInstall`] ([`otel::is_enabled`] lets a caller branch
//!   before calling): without the `otel` feature the sink is *not* installed,
//!   and that is an observable outcome rather than a silent success.
//! - [`checkpoint`] — the [`checkpoint::Checkpointer`] contract, the
//!   [`checkpoint::Checkpoint`] model, the memory, file, SQLite (feature-gated),
//!   and Redis (RESP2, clear text or `rediss://` TLS) backends, and
//!   [`checkpoint::pickle`], the Python-pickle interop codec
//!   ([`checkpoint::pickle::loads`], [`checkpoint::pickle::dumps`]) reached
//!   through [`checkpoint::checkpoint_from_pickle`] and
//!   [`checkpoint::checkpoint_to_pickle`].
//! - [`session_controller`] — scope-isolated, disk-backed session records
//!   ([`session_controller::DiskSessionStore`],
//!   [`session_controller::SessionController`]) in the Python
//!   `sessions.json`/`state.data` layout, plus the session link graph
//!   ([`session_controller::SessionLinks`] and
//!   [`session_controller::LinkKind`]) recording parent and fork relationships.
//! - [`vcs`] — local session version history ([`vcs::SessionHistory`]) with
//!   local refs ([`vcs::Refs`], [`vcs::RefName`], [`vcs::HEAD_REF`],
//!   [`vcs::MAIN_REF`]), `rewind`/`rewind_hard`
//!   ([`vcs::RewindOptions`]), `fork`, and snapshot JSON diffing.
//! - [`tracer`] — trace spans ([`tracer::TraceSpan`]), frames
//!   ([`tracer::TraceFrame`]), sinks ([`tracer::TraceSink`],
//!   [`tracer::MemoryTraceSink`], [`tracer::StreamTraceSink`]), the
//!   [`tracer::Tracer`] span factory, and the [`tracer::TracePosition`]
//!   persisted with checkpoints.
//!
//! JSON stays the native checkpoint format: [`checkpoint::pickle`] is an
//! explicit interop codec for payloads the Python SDK wrote and for handing a
//! checkpoint back to Python, never a second storage format.
//!
//! The types most consumers need are re-exported at the crate root:
//! [`SessionConfig`], the default ids and stream constants ([`DEFAULT_NODE_ID`],
//! [`DEFAULT_STREAM_CAPACITY`], [`DEFAULT_WORKFLOW_ID`], [`END_FRAME`]), the
//! state stores and lane keys ([`WorkflowState`], [`AgentState`], [`StateStore`],
//! [`StateLane`], and the `*_KEY` constants), the interaction vocabulary
//! ([`InteractionBroker`], [`InteractionRequest`], [`InteractionResponse`],
//! [`InteractionKind`], [`InteractionDecision`], [`InteractionOption`],
//! [`InteractionOutput`], [`InteractiveInput`]), the stream frames
//! ([`StreamItem`], [`StreamEvent`], [`StreamMode`], [`StreamSendOutcome`],
//! [`StreamHub`], [`StreamWriterManager`]), checkpointing ([`Checkpointer`],
//! [`Checkpoint`], [`CheckpointMeta`], [`CheckpointNamespace`],
//! [`CheckpointerConfig`], [`CheckpointerKind`], [`MemoryCheckpointer`],
//! [`FileCheckpointer`], [`RedisCheckpointer`], [`RedisOptions`], [`build_key`],
//! [`build_key_with_namespace`], the namespace constants, and the pickle
//! interop helpers [`loads`], [`dumps`], [`BYTES_KEY`],
//! [`checkpoint_from_pickle`], [`checkpoint_to_pickle`]),
//! disk-backed sessions, their link graph, and history ([`DiskSessionStore`],
//! [`SessionController`], [`SessionScope`], [`SessionLinks`], [`SessionLink`],
//! [`LinkKind`], [`LinksIndex`], [`SessionHistory`], [`Version`], [`Refs`],
//! [`RefName`], [`RefEntry`], [`RewindOptions`], [`HEAD_REF`], [`MAIN_REF`],
//! [`MAX_REF_NAME_LEN`]), wrapper sessions ([`SubWorkflowSession`],
//! [`RouterSession`]), the trace-sink hook ([`SwappableTraceSink`],
//! [`install_trace_sink`], [`try_install_trace_sink`], [`TraceSinkInstall`],
//! [`is_enabled`]), and tracing ([`Tracer`], [`TraceSpan`], [`TraceFrame`],
//! [`TraceSink`], [`MemoryTraceSink`], [`StreamTraceSink`], [`NoopTraceSink`],
//! [`TraceSpanKind`], [`TraceFrameType`], [`TracePosition`], [`SpanHandle`],
//! [`InvokeType`], [`SpanStatus`]).
#![forbid(unsafe_code)]

pub mod checkpoint;
pub mod config;
pub mod constants;
pub mod interaction;
pub mod otel;
pub mod session;
pub mod session_controller;
pub mod sessions;
pub mod state;
pub mod stream;
pub mod trace_hook;
pub mod tracer;
pub mod vcs;

#[cfg(feature = "sqlite")]
pub use checkpoint::SqliteCheckpointer;

pub use checkpoint::pickle::{dumps, loads, BYTES_KEY};
pub use checkpoint::{
    build_key, build_key_with_namespace, checkpoint_from_pickle, checkpoint_to_pickle,
    create_checkpointer, Checkpoint, CheckpointMeta, CheckpointNamespace, Checkpointer,
    CheckpointerConfig, CheckpointerKind, FileCheckpointer, MemoryCheckpointer,
    SESSION_NAMESPACE_AGENT, SESSION_NAMESPACE_AGENT_TEAM, SESSION_NAMESPACE_WORKFLOW,
    WORKFLOW_NAMESPACE_GRAPH,
};
pub use config::SessionConfig;

pub use constants::{DEFAULT_NODE_ID, DEFAULT_STREAM_CAPACITY, DEFAULT_WORKFLOW_ID, END_FRAME};

pub use interaction::{
    InteractionBroker, InteractionDecision, InteractionKind, InteractionOption, InteractionOutput,
    InteractionRequest, InteractionResponse, InteractiveInput,
};

pub use session::{Session, SessionMetadata, SessionSnapshot, SessionType};

pub use sessions::{
    create_agent_session, create_agent_team_session, create_workflow_session, AgentSession,
    AgentSessionOptions, AgentTeamSession, AgentTeamSessionOptions, NodeSession, WorkflowSession,
    WorkflowSessionOptions, DEFAULT_TEAM_ID,
};

pub use state::{AgentState, StateLane, StateStore, WorkflowState};
pub use state::{
    AGENT_STATE_KEY, COMP_STATE_KEY, COMP_STATE_UPDATES_KEY, GLOBAL_STATE_KEY,
    GLOBAL_STATE_UPDATES_KEY, IO_STATE_KEY, IO_STATE_UPDATES_KEY, TRACE_STATE_KEY,
    WORKFLOW_STATE_KEY, WORKFLOW_STATE_UPDATES_KEY,
};

pub use stream::{
    StreamEvent, StreamHub, StreamItem, StreamMode, StreamSendOutcome, StreamWriterManager,
};

pub use tracer::{
    InvokeType, MemoryTraceSink, NoopTraceSink, SpanHandle, SpanStatus, StreamTraceSink,
    TraceFrame, TraceFrameType, TracePosition, TraceSink, TraceSpan, TraceSpanKind, Tracer,
};

// Wrapper sessions and the trace-sink hook (added with the `SubWorkflowSession`,
// `RouterSession`, and per-session trace-sink work).
pub use otel::{install_trace_sink, is_enabled, try_install_trace_sink, TraceSinkInstall};
pub use sessions::{RouterSession, SubWorkflowSession};
pub use trace_hook::SwappableTraceSink;

// Disk-backed product sessions (`session_controller`), the parent/fork link
// graph, and local VCS history (versions, refs, rewinds, forks, diffs).
pub use session_controller::{
    DiskSessionStore, LinkKind, LinksIndex, Scope, SessionController, SessionLink, SessionLinks,
    SessionMeta, SessionRecord, SessionScope, SessionScopeKey, Subject,
};
pub use vcs::{
    diff_snapshots, diff_values, ChangeKind, JsonChange, RefEntry, RefName, Refs, RewindOptions,
    SessionHistory, Version, VersionId, VersionMeta, HEAD_REF, MAIN_REF, MAX_REF_NAME_LEN,
};

// Remote checkpointing over Redis.
pub use checkpoint::{RedisCheckpointer, RedisOptions};
