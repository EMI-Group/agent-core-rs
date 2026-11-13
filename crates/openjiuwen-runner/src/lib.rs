//! `Runner`, `ResourceMgr`, callbacks, and the message-queue abstraction — the
//! process-level orchestrator of the openJiuwen kernel.
//!
//! Maps to the Python package `openjiuwen.core.runner` (v0.1.20). The crate owns:
//!
//! - [`Runner`] — configuration, lifecycle, and the six execution entry points
//!   (`run_agent`, `run_workflow`, `run_agent_team` plus streaming variants);
//! - [`ResourceMgr`] — the typed registries for tools, agents, workflows, and
//!   teams, with providers invoked per lookup, plus the tag, model, prompt,
//!   sys-operation, and MCP-server sub-managers ([`TagMgr`], [`ModelMgr`],
//!   [`PromptMgr`], [`SysOperationMgr`], [`McpServerMgr`]);
//! - [`CallbackFramework`] — priority-ordered lifecycle events with explicit
//!   error isolation, the filter family ([`EventFilter`]), ordered handler
//!   groups ([`CallbackChain`]), the handler combinators ([`HandlerBuilder`],
//!   [`transform_io()`](crate::callback::decorator::transform_io),
//!   [`wrap()`](crate::callback::decorator::wrap())), and the guardrail binding
//!   ([`bind_guardrails`]);
//! - [`MessageQueue`] — an in-process, per-topic FIFO queue ([`InMemoryMessageQueue`])
//!   and the trait seam a broker backend (`openjiuwen-extensions`) implements;
//! - [`RunMonitor`] — bounded per-run phase/outcome tracking with a broadcast
//!   event stream ([`RunEvent`]);
//! - [`InteractionHub`] — runner-level human-in-the-loop delivery, interaction
//!   answering, and pause/resume/cancel signals for live runs;
//! - [`TeamPool`] — the in-process team runtime pool with admission, leases,
//!   bounded concurrency, idle eviction, and payload buffering;
//! - the spawn seam ([`AgentSpawner`], [`SpawnSpec`], [`SpawnMessage`]), with
//!   [`SubprocessSpawner`] (a real child over `tokio::process` speaking NDJSON)
//!   and [`RemoteSpawner`]/[`RemoteAgent`] (an HTTP endpoint over `reqwest`)
//!   implemented. The message-queue and A2A transports a remote client can be
//!   asked for ([`RemoteProtocol::Mq`], [`RemoteProtocol::A2a`]) stay typed
//!   `Error::Unsupported`, as does the built-in [`UnsupportedSpawner`] default.
//!
//! Every public type is re-exported flat at the crate root; the module paths
//! (`openjiuwen_runner::callback::CallbackEvent`, `openjiuwen_runner::run::RunOptions`,
//! ...) remain available for hosts that prefer them.
//!
//! Two deliberate omissions keep the flat surface usable: the *free combinator
//! functions* (`on`, `wrap`, `on_wrap`, `emit_before`, `emit_after`,
//! `emit_around`, `transform_io`, `and`, `or`, `not`, `lenient`) stay under
//! `openjiuwen_runner::callback::{decorator, filters}`, because names like `on`,
//! `and`, `or`, and `not` are too generic to occupy the crate root; and
//! crate-internal seams (`RunStatusGuard`, `RunBinding`) are never exported.
//!
//! ```no_run
//! use openjiuwen_runner::{RunOptions, Runner};
//! use serde_json::json;
//!
//! # async fn example() -> openjiuwen_common::error::Result<()> {
//! let runner = Runner::new();
//! runner.start().await?;
//! let output = runner
//!     .run_agent("assistant", json!({"query": "hi"}), RunOptions::default())
//!     .await?;
//! assert!(!output.output.is_empty());
//! runner.stop().await?;
//! # Ok(())
//! # }
//! ```
//!
//! Design boundaries and the ported behaviours are documented in `CONTEXT.md`
//! and in each module directory's `CONTEXT.md`.

pub mod callback;
pub mod config;
pub mod error;
pub mod hitl;
pub mod monitor;
pub mod mq;
pub mod pool;
pub mod resources;
pub mod run;
pub mod runner;
mod runner_spawn;
pub mod spawn;

// ------------------------------------------------------- flat re-exports --

// --- callback dispatch core ---
pub use callback::{
    bind_guardrails, event_for_stage, scan_stage_for, AsyncHandler, CallbackContext, CallbackEvent,
    CallbackFailure, CallbackFramework, CallbackHandler, CallbackId, CallbackInfo,
    CallbackRegistration, CallbackReport, CallbackResult, CallbackVerdict, EventDomain,
    GuardrailHandler, SyncHandler,
};
// --- callback chains ---
pub use callback::{
    CallbackChain, ChainEntry, ChainEntryId, ChainReport, EntryOutcome, EntryRecord,
};
// --- callback filters (combinators `and`/`or`/`not`/`lenient` stay in `callback::filters`) ---
pub use callback::{
    AuthFilter, CircuitBreakerFilter, Clock, ConditionalFilter, ContextModifier, EventFilter,
    FilterChain, FilterDecision, FilterFeedback, FilterInput, FilterRunOutcome, KindFilter,
    LoggingFilter, ManualClock, NameFilter, ParamModifyFilter, PredicateFilter, RateLimitFilter,
    SessionFilter, ValidationFilter,
};
// --- callback handler combinators (free functions stay in `callback::decorator`) ---
pub use callback::{
    BuiltHandler, HandlerBuilder, PayloadTransform, TransformHandler, WrapRegistry, WrappedCall,
};
// --- configuration ---
pub use config::{
    DistributedConfig, MessageQueueConfig, MessageQueueKind, RunnerConfig, TeamPoolConfig,
    CONVERSATION_ID_KEY, DEFAULT_REQUEST_TIMEOUT_SECS, DEFAULT_RUNNER_ID, DEFAULT_SESSION_ID,
    DEFAULT_TEAM_POOL_FRAME_CAPACITY, DEFAULT_TEAM_POOL_IDLE_TIMEOUT_SECS,
    DEFAULT_TEAM_POOL_MAX_RUNS, DEFAULT_TEAM_POOL_PAYLOAD_CAPACITY,
    DEFAULT_TEAM_POOL_REAPER_INTERVAL_SECS, DEFAULT_TEAM_POOL_RETAIN_FINISHED,
    DEFAULT_TEAM_POOL_STOP_TIMEOUT_SECS, DEFAULT_TOPIC_CAPACITY,
};
// --- human-in-the-loop ---
pub use hitl::{
    ActiveRunInfo, ActiveRunSpec, DeliverResult, InteractKind, InteractPayload, InteractionHub,
    RunHandle, RunTarget, INBOX_CAPACITY,
};
// --- monitoring ---
pub use monitor::{
    RunEvent, RunKind, RunMonitor, RunOutcome, RunPhase, RunStart, RunStatus,
    EVENT_CHANNEL_CAPACITY, EVENT_HISTORY_LIMIT, LIVE_RUN_LIMIT,
};
// --- message queue ---
pub use mq::{
    build_message_queue, create_message_queue, DeferredMessageQueue, InMemoryMessageQueue,
    MessageHandler, MessageQueue, QueueMessage, SUCCESS,
};
// --- team runtime pool ---
pub use pool::{
    ActiveTeamInfo, PoolError, PoolFrame, PoolKey, PoolLease, PoolRequest, PoolRun, PoolShutdown,
    TeamPool,
};
// --- registries and their sub-managers ---
pub use resources::{
    DefaultMcpConnector, McpConnector, McpServerInfo, McpServerMgr, McpServerState, ModelCard,
    ModelMgr, ModelProvider, PromptCard, PromptMgr, Provider, RegisterPolicy, Registration,
    Registry, ResourceCard, ResourceKind, ResourceMgr, SysOperationMgr, SysOperationProvider, Tag,
    TagMatchStrategy, TagMgr, TagUpdateStrategy, GLOBAL_TAG,
};
// --- execution ---
pub use run::{
    invoke_agent, invoke_team, invoke_workflow, stream_agent, stream_team, stream_workflow,
    AgentRef, RunOptions, RunServices, TeamRef, WorkflowRef,
};
pub use runner::Runner;
// --- spawn seam and its backends ---
pub use spawn::{
    create_remote_client, frame_bytes, parse_frame_line, read_frame, AgentSpawner, DeferredSpawner,
    ExternalCommandSpawner, InProcessAgent, InProcessFactory, InProcessHandler, InProcessSpawner,
    MonitoredSpawnedAgent, RemoteAgent, RemoteAgentConfig, RemoteClient, RemoteProtocol,
    RemoteSpawner, SpawnAgentKind, SpawnBackend, SpawnHealth, SpawnMessage, SpawnMessageKind,
    SpawnSpec, SpawnedAgent, SpawnedProcess, SpawnerRegistry, SubprocessConfig, SubprocessSpawner,
    UnsupportedSpawner, DEFAULT_MAX_FRAME_BYTES,
};
