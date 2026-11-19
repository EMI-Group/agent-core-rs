//! `DeepAgent` harness of the openJiuwen SDK — a port of the Python `openjiuwen.harness` package.
//!
//! This crate composes the `openjiuwen_core` kernel (agent runtime, sessions, sysop, security,
//! memory, context engine) into the `DeepAgent` task-loop framework with its workspace, permission
//! engine, coding tools, rails, prompts, skills and subagents.
//! It also implements the native provider for the provider-neutral SPI defined by
//! `openjiuwen_harness_protocol`.
//! The module tree and per-module responsibilities are documented in `CONTEXT.md`.
//!
//! # Flat API surface
//!
//! Every primary public item is also re-exported flat from the crate root, mirroring the Python
//! `openjiuwen.harness` package whose submodules are additionally flattened into `__init__`.
//! Module paths remain canonical: `openjiuwen_harness::agent::DeepAgent` and
//! `openjiuwen_harness::DeepAgent` name the same type, so consumers should prefer the module path
//! and treat the flat exports as a convenience spelling.
//!
//! Names that would collide are resolved with documented aliases:
//! - `tools::tool_names` keeps its name; the prompt-side list is exported as
//!   [`tool_description_names`] (alias of `prompts::tool_names`).
//! - `schema::SubAgentConfig` is the single flat spelling; `subagents` re-exports the same type,
//!   so it is listed once.
//! - `rails::SubAgentStatus` (rail bookkeeping) and `subagents::SubagentStatus` (runtime state)
//!   are distinct types and keep their distinct spellings.

#![forbid(unsafe_code)]

pub mod agent;
pub mod cli;
pub mod lsp;
pub mod manifest;
pub mod moa;
pub mod observability;
pub mod permission;
pub mod persistent_agent;
pub mod prompts;
pub mod protocol;
pub mod rails;
pub mod schema;
pub mod skills;
pub mod spec;
pub mod subagents;
pub mod task_loop;
pub mod tools;
pub mod watcher;
pub mod workspace;
pub mod worktree;

// --- agent -----------------------------------------------------------------
pub use agent::{
    apply_deep_agent_parts, clear_state, create_deep_agent, drain_rail_events, load_state,
    resolve_deep_agent_parts, save_state, seed_state, DeepAgent, DeepAgentBuilder, DeepAgentParts,
    DeepAgentSubAgentFactory, RailEventBus, RailEventDrain, DEFAULT_RAIL_EVENT_CAPACITY,
    RAIL_EVENTS_EXTRA_PREFIX, RAIL_EVENT_DRAIN_NAME, RAIL_EVENT_DRAIN_PRIORITY, SESSION_STATE_KEY,
};

// --- schema ----------------------------------------------------------------
pub use schema::{
    AgentMode, DeepAgentConfig, DeepAgentState, DeepLoopEvent, DeepLoopEventType,
    InputDispatchMode, InteractionEvent, InteractionPhase, OnMaxIterations, PlanModeState,
    PromptMode, RoundOutcome, RoundWorkItem, SendInputRequest, StopConditionEvaluator,
    StopConditionState, StopEvaluationContext, StopVerdict, SubAgentConfig,
    TaskLoopNoProgressGuardConfig, TaskPlan, TodoItem, TodoStatus,
};

// --- workspace -------------------------------------------------------------
pub use workspace::{
    is_safe_relative_path, DirectoryBuilder, DirectoryNode, Workspace, WorkspaceNode,
};

// --- permission ------------------------------------------------------------
pub use permission::{
    PermissionConfirmResponse, PermissionEngine, PermissionEngineConfig, PermissionHost,
    PermissionInterruptRail, PermissionLevel, PermissionResult, PermissionsSection,
};

// --- tools -----------------------------------------------------------------
/// Names of the thirty-five tools of [`BuiltinTool::EXTENDED`] (the default twenty plus the
/// fifteen extended ones), in registry order.
pub use tools::extended_tool_names;
/// The goal rail, its tunables, and the `ctx.extra` key under which it queues its events.
///
/// [`GoalRail`] is a [`HarnessRail`] registered like the rails in [`rails`], but it is reachable
/// only as `tools::goal::GoalRail`: [`rails`] must not depend on [`tools`], so the goal rail is
/// exported from the module that owns it instead.
pub use tools::goal::{GoalRail, GoalRailConfig, GOAL_EVENT_EXTRA_KEY};
pub use tools::{
    tool_names, AudioMetadata, AudioMetadataArgs, AudioMetadataTool, BuiltinTool, CronAction,
    CronArgs, CronJob, CronRun, CronRunTrigger, CronSchedule, CronScheduler, CronStatus, CronTool,
    CronUpdate, FakeMcpResourceProvider, FakeMcpResourceSubscriptionProvider, FakeSearchBackend,
    FetchRequest, FetchResponse, FetchWebpageArgs, FetchWebpageTool, GetCurrentGoalArgs,
    GetCurrentGoalTool, GoalAssessment, GoalAssessmentStatus, GoalRecord, GoalStatus, GoalStore,
    HttpFetcher, ImageMetadata, ImageMetadataArgs, ImageMetadataTool, ListMcpResourcesTool,
    McpClientResourceProvider, McpResource, McpResourceContent, McpResourceEvent,
    McpResourceProvider, McpResourceSink, McpResourceSubscriptionArgs,
    McpResourceSubscriptionProvider, McpResourceSubscriptionTool, McpResourceSubscriptions,
    McpSubscriptionInfo, MediaKind, MemoryToolBackend, NotificationOutcome, ReadMcpResourceArgs,
    ReadMcpResourceTool, ReqwestFetcher, SearchBackend, SearchResult, SubmitGoalReportArgs,
    SubmitGoalReportTool, SubscriptionAction, TokenUsage, ToolIndex, ToolRegistry, ToolSearchArgs,
    ToolSearchHit, ToolSearchTool, UserAsker, WebSearchArgs, WebSearchTool, WorkspaceMemoryBackend,
    LIVE_CLIENT_SUBSCRIBE_UNSUPPORTED, LIVE_CLIENT_UNSUBSCRIBE_UNSUPPORTED,
    RESOURCE_LIST_CHANGED_METHOD, RESOURCE_LIST_CHANGED_METHOD_SHORT, RESOURCE_UPDATED_METHOD,
    RESOURCE_UPDATED_METHOD_SHORT,
};

// --- rails -----------------------------------------------------------------
pub use rails::{
    affinity_enabled, build_heartbeat_section, default_heartbeat_clock, default_metrics_clock,
    evolution_adapter, heartbeat_prompt, is_sticky_subagent_type, messages_are_contiguous,
    normalize_source_locator, rail_priority, resolve_context_relative_path, resolve_sub_session_id,
    resolve_subagent_parent_cache_id, scope_sub_session_id, search, source_id_for_locator,
    ContextEntry, EvolutionMetrics, EvolutionProposal, EvolutionRail, EvolutionRailAdapter,
    EvolutionSnapshot, FetchedSource, HarnessRail, HarnessRailAdapter, HarnessRailContext,
    HarnessRailRegistry, HeartbeatClock, HeartbeatEvent, HeartbeatRail, HeartbeatRailConfig,
    HeartbeatReason, IngestReport, IngestedDocument, IterationRecorder, KvCacheAction,
    KvCacheAffinityConfig, KvCacheLifecycleRail, KvCacheLineage, KvCacheProviderRegistry,
    KvCacheSubagent, KvCacheSubagentLifecycle, KvCacheSubagentSession, KvCacheSubagentState,
    LocalDirectoryConfig, LocalFilesProvider, MemoryRail, MetricsClock, MetricsEvolutionRail,
    MetricsEvolutionRailConfig, NoopEvolutionRail, PersonalContextRail, PersonalContextRailConfig,
    PersonalContextStatus, PersonalContextStore, ProposalLog, RuntimeSessionLineage,
    SearchDocument, SearchHit, SecurityRail, SkillsRail, SkillsRailConfig, SourceKind, SourceLink,
    SourceLinkBook, SourceMetadata, SourceMetadataStore, SourceProvider, SourceProviderRegistry,
    SubagentRail, TaskIterationInfo, UnsupportedKvCacheBackend, UnsupportedSourceProvider,
    AFFINITY_PARENT_SESSION_ID_ENV, DEFAULT_HEARTBEAT_EVENT_CAPACITY,
    DEFAULT_HEARTBEAT_INTERVAL_MS, DEFAULT_HEARTBEAT_SESSION_CAPACITY, EVOLUTION_SNAPSHOT_VERSION,
    HEARTBEAT_RUN_KIND, HEARTBEAT_SECTION, HEARTBEAT_SECTION_PRIORITY, RUN_KIND_EXTRA_KEY,
    STICKY_SUBAGENT_TYPE,
};

// --- moa -------------------------------------------------------------------
pub use moa::{
    aggregation_messages, builtin_skill_content, consult, consult_problem, consult_skill,
    consultation_transcript, escape_xml, materialize_builtin_skill, moa_section, rank_proposals,
    rank_with_config, redact, reference_messages, render_reference_proposals, score_progress,
    score_proposal, validate_builtin_skill_name, verifier_messages, AggregationStrategy,
    Aggregator, Consultation, ConsultationHistory, ConsultationOutcome, ConsultationRecord,
    ConsultationSkillMount, JevConfig, LlmVerifier, MoaConfig, MoaMode, MoaRail, MoaSnapshot,
    MoaState, ProgressVerifier, ProposalAttempt, ProposalBatch, ProposalSource, ProposalStore,
    ProposerFailure, RankedProposal, StructuralVerifier, TurnKey, VerifierConfig,
    VerifierSelection, AGGREGATOR_SYSTEM_PROMPT, CONSULT_SKILL_BODY, CONSULT_SKILL_DESCRIPTION,
    CONSULT_SKILL_FILE, CONSULT_SKILL_NAME, CONSULT_TOOL_NAME, DEFAULT_HISTORY_LIMIT,
    DEFAULT_MAX_ADVICE_CHARS, DEFAULT_MAX_CONCURRENCY, DEFAULT_MAX_REFERENCE_CHARS,
    DEFAULT_REFERENCE_TIMEOUT_SECS, MOA_OUTCOME_EXTRA_KEY, MOA_PROMPT_SECTION,
    MOA_PROMPT_SECTION_NAME, MOA_RAIL_NAME, MOA_STATE_EXTRA_KEY, REFERENCE_SYSTEM_PROMPT,
};

// --- lsp -------------------------------------------------------------------
pub use lsp::{
    builtin_server_configs, client_capabilities, parse_call_hierarchy_incoming_calls,
    parse_call_hierarchy_items, parse_call_hierarchy_outgoing_calls, parse_completion_items,
    parse_diagnostics, parse_diagnostics_event, parse_document_diagnostic_report,
    parse_document_symbols, parse_hover, parse_locations, parse_workspace_symbols, path_to_uri,
    symbol_kind_name, uri_to_path, BackoffTimer, CallHierarchyIncomingCall, CallHierarchyItem,
    CallHierarchyOutgoingCall, ChannelTransport, CompletionItem, Diagnostic, DiagnosticCode,
    DiagnosticRegistry, DiagnosticRelatedInformation, DiagnosticReportKind, DiagnosticsChange,
    DiagnosticsConfig, DiagnosticsEvent, DiagnosticsFile, DiscoveryIndex, DocumentDiagnosticReport,
    DocumentSymbol, FrameDecoder, Hover, InstantTimer, Location, LspAction, LspArgs, LspClient,
    LspError, LspServerConfig, LspTool, LspToolConfig, LspTransport, MessageReader, Position,
    Range, ResolvedServer, RestartPolicy, ServerCapabilities, ServerFactory, ServerKey,
    ServerState, ServerStatus, ServerSupervisor, StdioTransport, SymbolInformation, TokioTimer,
    MAX_CONTENT_LENGTH,
};

// --- prompts ---------------------------------------------------------------
/// Alias of `prompts::tool_names` (the prompt-side tool list); the flat [`tool_names`] spelling
/// resolves to the `tools` registry order.
pub use prompts::tool_names as tool_description_names;
pub use prompts::{
    sanitize_path, sanitize_user_content, HarnessPromptBuilder, PromptReport, PromptSection,
    SectionName,
};

// --- subagents -------------------------------------------------------------
/// Retention bound of the persisted sub-agent bucket for `max_subagents` live instances.
pub use subagents::max_persisted_records;
/// Formats epoch milliseconds as an RFC 3339 UTC timestamp (the runtime's stamp format).
pub use subagents::rfc3339_from_ms;
/// Reason stamped on an instance closed by the LRU eviction policy.
pub use subagents::EVICTION_REASON;
/// Key of the persisted sub-agent bucket inside the session state.
pub use subagents::SUBAGENTS_KEY;
pub use subagents::{
    load_bucket_file, save_bucket_file, select_lru_candidate, subagent_tools,
    subagent_tools_with_spawner, subagent_tools_with_transcript,
    subagent_tools_with_transcript_spawner, ActivityEvent, ActivityKind, ActivityStream,
    ActivitySubscription, ActivitySummary, ActivityThrottle, CallbackLifecycleHook, Clock,
    EvictionSchedulerConfig, EvictionSchedulerHandle, EvictionSchedulerStats, LifecycleCallback,
    LifecycleFailure, LifecycleHooks, LifecycleReport, LifecycleStage, LoadReport, LruCandidate,
    ManualClock, PersistedSubAgentBuckets, PersistedSubAgentRecord, ResumeReport,
    RuntimeSubAgentSpawner, SpawnBudget, SubAgentFactory, SubAgentId, SubAgentOutcome,
    SubAgentPreset, SubAgentQuota, SubAgentRegistry, SubAgentRuntime, SubAgentSpawnRequest,
    SubAgentSpec, SubagentActivity, SubagentEviction, SubagentLifecycleEvent,
    SubagentLifecycleHook, SubagentMetadata, SubagentStatus, SubagentTranscriptTool, SweepReport,
    SystemClock, TaskTool, TranscriptBuffer, TranscriptChannel, TranscriptEntry, TranscriptKind,
    TranscriptSubscription, TranscriptSummary, DEFAULT_ACTIVITY_CAPACITY,
    DEFAULT_ACTIVITY_TEXT_MAX_LEN, DEFAULT_ACTIVITY_THROTTLE_MS, DEFAULT_EVICTION_INTERVAL,
    DEFAULT_IDLE_TTL_MS, DEFAULT_TRANSCRIPT_CAPACITY, DEFAULT_TRANSCRIPT_CHANNEL_CAPACITY,
};

// --- skills ----------------------------------------------------------------
pub use skills::{Skill, SkillBudget, SkillLibrary, SkillResourceValidator, SkillRootMount};

// --- spec ------------------------------------------------------------------
pub use spec::{
    resolve, DeepAgentSpec, PluginManifest, PluginRegistry, ResolvedSpec, SpecBuildOptions,
    SpecEntry, SpecRegistry, SpecSource,
};

// --- task loop -------------------------------------------------------------
pub use task_loop::{
    DefaultTaskLoopEventHandler, EventManager, LoopCoordinator, LoopQueues, RoundExecutor,
    StateSink, TaskLoopConfig, TaskLoopController, TaskLoopEventHandler,
};

// --- protocol --------------------------------------------------------------
pub use protocol::{
    native_card, NativeCheckpointState, NativeHarness, NativeHarnessOptions, NativeHarnessProvider,
};

// --- cli -------------------------------------------------------------------
/// Re-export of the `clap` crate that [`cli::Cli`] derives from.
///
/// The parser surface is invisible through the derive alone: a host or an
/// integration test cannot call `Cli::parse_from`/`Cli::try_parse_from`
/// (`clap::Parser`) or build the grammar with `Cli::command()`
/// (`clap::CommandFactory`) without naming the `clap` crate itself, and a
/// different `clap` version would not see those trait impls at all.
/// Re-exporting the exact version this crate was built with lets downstream
/// crates (`openjiuwen-e2e`, the `openjiuwen-cli` binary, tests) assert the CLI
/// grammar, error kinds (`clap::error::ErrorKind`) and process exit codes
/// (`clap::Error::exit_code`) without adding a dependency of their own.
///
/// This is a build-and-test seam, not part of the SDK domain API: production
/// code should hand a parsed [`cli::Cli`] to [`cli::run_cli`] rather than
/// parsing arguments itself.
pub use clap;
pub use cli::{
    run, run_cli, run_once_resumed, run_with_model_resumed, Cli, CliOptions, CliPermissionHost,
    Command, OutputFormat, Repl, ResumedTurn,
};

// --- manifest --------------------------------------------------------------
pub use manifest::{
    builtin_elements, Catalog, ElementKind, HarnessElementDescriptor, InterfaceMethod,
};

// --- observability ---------------------------------------------------------
pub use observability::{NoopTelemetry, Observability, TelemetrySink};

// --- worktree --------------------------------------------------------------
pub use worktree::{
    build_worktree_notice, change_summary, is_ephemeral_slug, is_lock_timeout,
    render_change_summary, slugify, CleanupPolicy, CleanupReport, EnterWorktreeTool,
    ExitWorktreeTool, GitRunner, MemoryWorktreeSessionStore, ProcessGitRunner, SparseConfig,
    SparseMode, WorktreeChangeSummary, WorktreeCreateOptions, WorktreeEvent, WorktreeInfo,
    WorktreeLock, WorktreeLockConfig, WorktreeManager, WorktreeRail, WorktreeRecord,
    WorktreeSessionState, WorktreeTool,
};
// --- watcher ---------------------------------------------------------------
pub use watcher::{
    ChangeKind, DebounceInput, Debouncer, DropStage, DroppedUpdates, FnReloadHandler,
    InstructionReloadHandler, InstructionReloader, McpConfigReloadHandler, McpConfigReloader,
    ReloadFailure, ReloadHandler, ReloadHub, SkillLibraryReloader, SkillsReloadHandler,
    Subscription, WatchEvent, WatchFailure, WatchPlan, WatchTarget, WatchTargetKind, WatchUpdate,
    WatcherConfig, WatcherHandle, WorkspaceWatcher, DEFAULT_DEBOUNCE, DEFAULT_DISPATCH_CAPACITY,
    DEFAULT_EVENT_CAPACITY, DEFAULT_INTAKE_CAPACITY, DEFAULT_MAX_PENDING,
    WATCHER_BACKEND_AVAILABLE,
};
// --- persistent_agent ------------------------------------------------------
pub use persistent_agent::{
    atomic_write, canonical_json, deterministic_model_call_id, digest_hex, fresh_id, hand_off,
    is_logical_path, is_read_only_tool, is_valid_identifier, patch_stopped_calls,
    percent_decode_identifier, percent_encode_identifier, read_json, resume, sha256_hex_str,
    valid_caption, write_json_atomic, AbortReason, ActiveExecution, Capacity, CheckpointKind,
    CompletionFn, CompletionSummarizer, Continuation, ContinuationCursor, Control, ControlSnapshot,
    DistributedHandoff, ElidedEntry, ElisionReason, EmittedEvent, Emitter, EntryKind, EntryPayload,
    EventSink, EventType, ExecutionCatalog, ExecutionEntry, ExecutionIdentity, ExecutionLink,
    ExecutionLinkSource, ExecutionLog, ExecutionLogSource, ExecutionRecordRail, ExecutionRecorder,
    ExecutionSummary, ExtractiveSummarizer, FileLinkStore, FileSessionStore, FinalStatus,
    HandoffReceipt, HandoffTarget, HandoffTransfer, HandoffTransport, ImportedShareBundle,
    InMemorySink, InterceptedCall, LinkRelation, LocalHandoff, LogCapacity, LogCursor, LogDigest,
    LogMeta, LogPreview, LogSnapshot, ModelAdmission, ModelCallGate, ModelCallRail, ModelUsage,
    ParticipationDecision, ParticipationState, PauseReason, PersistentAgentLayout, PersistentEvent,
    PersistentRunConfig, PersistentSessionRecord, RecordCheckpoint, RecordEnding, RecordRollover,
    RemoteHandoff, Replay, ResumePlan, RolloverTrigger, RootLogSource, SessionBinding,
    SessionStateStore, SessionSummary, ShareBundle, ShareFile, ShareMode, SharedFileEntry,
    ToolRecordPolicy, ToolStatus, UnsupportedModelSummarizer, UsageStatus, WaitingScene,
    WindowAssembler, WindowAssembly, WindowManager, WindowPlan, WindowPolicy, WindowSummarizer,
    BUNDLE_FILE, CAPTION_MAX_CHARS, CONTINUATION_RULES, DEFAULT_NAMESPACE_ID, DEFAULT_SOURCE_REF,
    DETAIL_LIMIT, ENTRY_SCHEMA_VERSION, FILE_PATH_MAX_CHARS, HANDOFF_TEXT_PREFIX,
    IDENTIFIER_MAX_CHARS, LINKS_DIR, LOGS_DIR, LOG_ENCODING, LOG_ENTRY_FILE, LOG_META_FILE,
    MAX_SHARED_FILES, MODEL_CALL_ID_EXTRA_KEY, PERSISTENT_AGENT_NODE, PLACEHOLDER_TEXT,
    READ_ONLY_TOOLS, RECORD_INVALID_PREFIX, SESSIONS_DIR, SESSION_ENDING_OUTCOME,
    SESSION_KEY_PREFIX, SESSION_RECORD_VERSION, SHARE_BUNDLE_SCHEMA_VERSION, SHARE_FILE_MAX_BYTES,
    SHARE_MODES, SUMMARY_LIMIT, USAGE_UNKNOWN_REASON,
};

/// Returns the crate name.
#[must_use]
pub fn crate_name() -> &'static str {
    "openjiuwen-harness"
}
