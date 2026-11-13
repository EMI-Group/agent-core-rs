#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Compile-level pinning of the crate-root surface: every documented flat
//! re-export, trait-object coercions, serde coverage, and `Send + Sync`.
//!
//! Nothing here asserts behaviour; the suites in `runner_lifecycle.rs`,
//! `resources.rs`, `callbacks.rs`, `security_binding.rs`, and `message_queue.rs`
//! do that. These tests fail when a re-export is renamed, moved out of the root,
//! or loses a derive the crate promises.

use futures::stream::BoxStream;
use openjiuwen_agent::{Agent, AgentCard, AgentChunk, AgentOutput};
use openjiuwen_common::error::Result;
use openjiuwen_runner::{
    bind_guardrails, create_message_queue, event_for_stage, invoke_agent, invoke_team,
    invoke_workflow, scan_stage_for, stream_agent, stream_team, stream_workflow, AgentRef,
    AgentSpawner, AsyncHandler, CallbackContext, CallbackEvent, CallbackFailure, CallbackFramework,
    CallbackHandler, CallbackId, CallbackInfo, CallbackRegistration, CallbackReport,
    CallbackResult, CallbackVerdict, DistributedConfig, EventDomain, GuardrailHandler,
    InMemoryMessageQueue, MessageHandler, MessageQueue, MessageQueueConfig, MessageQueueKind,
    Provider, QueueMessage, RegisterPolicy, Registration, Registry, ResourceCard, ResourceMgr,
    RunOptions, RunServices, Runner, RunnerConfig, SpawnAgentKind, SpawnMessage, SpawnMessageKind,
    SpawnSpec, SpawnedAgent, SyncHandler, TeamRef, UnsupportedSpawner, WorkflowRef,
    CONVERSATION_ID_KEY, DEFAULT_REQUEST_TIMEOUT_SECS, DEFAULT_RUNNER_ID, DEFAULT_SESSION_ID,
    DEFAULT_TOPIC_CAPACITY, SUCCESS,
};
use openjiuwen_security::{GuardrailOutcome, GuardrailPipeline};
use serde::de::DeserializeOwned;
use serde::Serialize;
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::Duration;

fn assert_send_sync<T: Send + Sync>() {}
fn assert_clone<T: Clone>() {}
fn assert_serde<T: Serialize + DeserializeOwned>() {}
fn assert_resource_card<T: ResourceCard>() {}

fn roundtrip<T>(value: &T)
where
    T: Serialize + DeserializeOwned + PartialEq + std::fmt::Debug,
{
    let text = serde_json::to_string(value).unwrap();
    let back: T = serde_json::from_str(&text).unwrap();
    assert_eq!(&back, value);
}

struct NoopHandler;

#[async_trait::async_trait]
impl CallbackHandler for NoopHandler {
    async fn handle(&self, _ctx: &CallbackContext) -> CallbackResult {
        Ok(CallbackVerdict::Continue)
    }
}

struct EchoHandler;

#[async_trait::async_trait]
impl MessageHandler for EchoHandler {
    async fn handle(&self, message: QueueMessage) -> Result<Value> {
        Ok(message.payload)
    }
}

#[test]
fn public_types_are_send_sync() {
    assert_send_sync::<Runner>();
    assert_send_sync::<RunnerConfig>();
    assert_send_sync::<MessageQueueConfig>();
    assert_send_sync::<MessageQueueKind>();
    assert_send_sync::<DistributedConfig>();
    assert_send_sync::<ResourceMgr>();
    assert_send_sync::<Registry<AgentCard, Arc<dyn Agent>>>();
    assert_send_sync::<Registration>();
    assert_send_sync::<RegisterPolicy>();
    assert_send_sync::<CallbackFramework>();
    assert_send_sync::<CallbackContext>();
    assert_send_sync::<CallbackEvent>();
    assert_send_sync::<CallbackInfo>();
    assert_send_sync::<CallbackReport>();
    assert_send_sync::<CallbackFailure>();
    assert_send_sync::<CallbackVerdict>();
    assert_send_sync::<InMemoryMessageQueue>();
    assert_send_sync::<QueueMessage>();
    assert_send_sync::<SpawnSpec>();
    assert_send_sync::<SpawnMessage>();
    assert_send_sync::<SpawnMessageKind>();
    assert_send_sync::<SpawnAgentKind>();
    assert_send_sync::<RunOptions>();
    assert_send_sync::<RunServices>();
}

#[test]
fn public_types_are_cloneable_where_documented() {
    assert_clone::<RunnerConfig>();
    assert_clone::<MessageQueueConfig>();
    assert_clone::<DistributedConfig>();
    assert_clone::<Registration>();
    assert_clone::<QueueMessage>();
    assert_clone::<SpawnSpec>();
    assert_clone::<SpawnMessage>();
    assert_clone::<CallbackContext>();
    assert_clone::<CallbackEvent>();
    assert_clone::<CallbackId>();
    assert_clone::<RunOptions>();
    assert_clone::<RunServices>();
}

#[test]
fn documented_type_paths_resolve_through_the_flat_root() {
    assert_eq!(
        std::any::type_name::<Runner>(),
        "openjiuwen_runner::runner::Runner"
    );
    assert_eq!(
        std::any::type_name::<RunnerConfig>(),
        "openjiuwen_runner::config::RunnerConfig"
    );
    assert_eq!(
        std::any::type_name::<MessageQueueConfig>(),
        "openjiuwen_runner::config::MessageQueueConfig"
    );
    assert_eq!(
        std::any::type_name::<CallbackEvent>(),
        "openjiuwen_runner::callback::event::CallbackEvent"
    );
    assert_eq!(
        std::any::type_name::<CallbackFramework>(),
        "openjiuwen_runner::callback::framework::CallbackFramework"
    );
    assert_eq!(
        std::any::type_name::<QueueMessage>(),
        "openjiuwen_runner::mq::message::QueueMessage"
    );
    assert_eq!(
        std::any::type_name::<ResourceMgr>(),
        "openjiuwen_runner::resources::ResourceMgr"
    );
    assert_eq!(
        std::any::type_name::<SpawnSpec>(),
        "openjiuwen_runner::spawn::SpawnSpec"
    );
    assert!(std::any::type_name::<RunOptions>().ends_with("RunOptions"));
    assert!(std::any::type_name::<AgentRef>().ends_with("AgentRef"));
    assert!(std::any::type_name::<WorkflowRef>().ends_with("WorkflowRef"));
    assert!(std::any::type_name::<TeamRef>().ends_with("TeamRef"));
}

#[test]
fn data_types_round_trip_through_json() {
    assert_serde::<RunnerConfig>();
    assert_serde::<MessageQueueConfig>();
    assert_serde::<MessageQueueKind>();
    assert_serde::<DistributedConfig>();
    assert_serde::<Registration>();
    assert_serde::<RegisterPolicy>();
    assert_serde::<QueueMessage>();
    assert_serde::<SpawnSpec>();
    assert_serde::<SpawnMessage>();
    assert_serde::<CallbackEvent>();

    roundtrip(&RunnerConfig::default());
    roundtrip(&MessageQueueConfig::in_memory());
    roundtrip(&MessageQueueKind::InMemory);
    roundtrip(&RegisterPolicy::Replace);
    roundtrip(&Registration {
        id: "a1".to_owned(),
        name: "assistant".to_owned(),
        replaced: true,
    });
    roundtrip(&QueueMessage::new(json!({"hello": "world"})));
    roundtrip(&SpawnSpec::class_agent("pkg.agents", "HelperAgent"));
    // `SpawnMessage`'s `timestamp` is an `f64` and `serde_json`'s float parser is
    // not correctly rounded, so the frame is compared field by field with the
    // documented tolerance instead of through the exact `PartialEq` round-trip.
    let frame = SpawnMessage::new(SpawnMessageKind::Input, json!({"q": 1}));
    let text = serde_json::to_string(&frame).unwrap();
    let restored: SpawnMessage = serde_json::from_str(&text).unwrap();
    assert_eq!(restored.kind, frame.kind);
    assert_eq!(restored.payload, frame.payload);
    assert_eq!(restored.message_id, frame.message_id);
    assert!(
        (restored.timestamp - frame.timestamp).abs() < 1e-6,
        "timestamp drifted by more than one part per million: {} vs {}",
        restored.timestamp,
        frame.timestamp
    );
    roundtrip(&CallbackEvent::AgentInvokeOutput);
    roundtrip(&CallbackEvent::Custom("host_event".to_owned()));
}

#[test]
fn constructors_and_builders_are_reachable() {
    let runner = Runner::new();
    assert_eq!(runner.runner_id(), DEFAULT_RUNNER_ID);
    let other = Runner::with_config(RunnerConfig::default());
    assert!(!Arc::ptr_eq(&runner, &other));

    let _ = MessageQueueConfig::in_memory().with_capacity(8);
    let _ = DistributedConfig::default();
    let _ = CONVERSATION_ID_KEY;
    let _ = DEFAULT_SESSION_ID;
    let _ = DEFAULT_TOPIC_CAPACITY;
    let _ = DEFAULT_REQUEST_TIMEOUT_SECS;

    let manager = ResourceMgr::new();
    assert!(manager.is_empty());
    let _: Provider<u32> = Arc::new(|| 7);
    let _: Registry<AgentCard, Arc<dyn Agent>> = Registry::new();
    assert_resource_card::<AgentCard>();

    let framework = CallbackFramework::new();
    let handler: Arc<dyn CallbackHandler> = Arc::new(NoopHandler);
    let registration = CallbackRegistration::new(CallbackEvent::AgentInvokeOutput, handler)
        .with_priority(10)
        .with_namespace("suite")
        .with_tag("smoke")
        .once();
    let id = framework.register(registration);
    assert_eq!(
        framework.handlers(CallbackEvent::AgentInvokeOutput).len(),
        1
    );
    assert_eq!(framework.len(), 1);
    framework.unregister(id);
    assert!(framework.is_empty());
    let _: Arc<dyn CallbackHandler> = Arc::new(SyncHandler::new(|_ctx: &CallbackContext| {
        Ok::<CallbackVerdict, openjiuwen_common::error::Error>(CallbackVerdict::Continue)
    }));
    let _: Arc<dyn CallbackHandler> = Arc::new(AsyncHandler::new(|_ctx: CallbackContext| {
        Box::pin(async {
            Ok::<CallbackVerdict, openjiuwen_common::error::Error>(CallbackVerdict::Continue)
        })
    }));
    let _ = CallbackContext::new(CallbackEvent::AgentInvokeOutput).with_component("a1");
    let _ = CallbackVerdict::Stop;
    let _ = CallbackReport {
        event: CallbackEvent::AgentInvokeOutput,
        fired: 0,
        failures: Vec::new(),
        aborted: false,
    };

    let queue: Arc<dyn MessageQueue> = InMemoryMessageQueue::with_capacity(8);
    assert!(!queue.is_started());
    let _: Arc<dyn MessageHandler> = Arc::new(EchoHandler);
    let _ = QueueMessage::new(json!(1)).with_id("m1");
    assert_eq!(SUCCESS, 0);
    let _ = create_message_queue;

    let _ = SpawnAgentKind::TeamAgent;
    let _ = SpawnSpec::default();
    let _ = SpawnSpec::team_agent(json!({"team": "crew"}));
    let _ = SpawnMessage::new(SpawnMessageKind::Done, Value::Null);
    let spawner: Arc<dyn AgentSpawner> = Arc::new(UnsupportedSpawner);
    let _: Arc<dyn SpawnedAgent>;

    let options = RunOptions::default()
        .with_session_id("s1")
        .with_timeout(Duration::from_secs(1))
        .with_envs("K", Value::from("v"));
    let _ = options.metadata;
    let _: AgentRef = "a1".into();
    let _: WorkflowRef = "w1".into();
    let _: TeamRef = "t1".into();
    let _ = AgentRef::id("a1");
    let services = RunServices::new(
        ResourceMgr::new(),
        CallbackFramework::new(),
        RunnerConfig::default(),
        None,
    );
    let _ = services.with_guardrails(Arc::new(GuardrailPipeline::new()));
    let _ = spawner;
}

#[test]
fn execution_functions_and_bindings_are_exported() {
    let _ = invoke_agent;
    let _ = invoke_workflow;
    let _ = invoke_team;
    let _: fn(RunServices, AgentRef, Value, RunOptions) -> BoxStream<'static, Result<AgentChunk>> =
        stream_agent;
    let _ = stream_workflow;
    let _ = stream_team;
    let _ = bind_guardrails;
    let _ = scan_stage_for;
    let _ = event_for_stage;
    let _: Option<Arc<dyn CallbackHandler>> = Some(Arc::new(GuardrailHandler::new(
        Arc::new(GuardrailPipeline::new()),
        openjiuwen_security::ScanStage::UserInput,
    )));
    let _: Option<Result<GuardrailOutcome>> = None;
    let _ = EventDomain::Agent;
    let _ = CallbackVerdict::Continue;
}

#[test]
fn callback_events_round_trip_through_their_wire_names() {
    assert_ne!(CallbackEvent::ALL.len(), 0);
    for event in CallbackEvent::ALL {
        assert_eq!(
            CallbackEvent::from_name(event.as_str()),
            Some(event.clone()),
            "{}",
            event.as_str()
        );
        assert_eq!(
            event.name(),
            event.as_str().trim_start_matches("_framework:")
        );
    }
}

#[test]
fn runner_debug_is_redacted() {
    let runner = Runner::new();
    let debug = format!("{runner:?}");
    assert!(debug.contains("runner_id"));
    assert!(!debug.contains("api_key"));
}

#[test]
fn agent_output_type_is_reachable_for_facade_results() {
    let _: fn() -> AgentOutput = AgentOutput::default;
    let _: fn() -> AgentCard = AgentCard::default;
}

#[test]
fn spawn_backends_are_flat_on_the_crate_root() {
    use openjiuwen_common::error::ErrorKind;
    use openjiuwen_runner::{
        create_remote_client, frame_bytes, parse_frame_line, read_frame, RemoteAgent,
        RemoteAgentConfig, RemoteClient, RemoteProtocol, RemoteSpawner, SpawnedProcess,
        SubprocessConfig, SubprocessSpawner, DEFAULT_MAX_FRAME_BYTES,
    };

    assert_send_sync::<SubprocessSpawner>();
    assert_send_sync::<SubprocessConfig>();
    assert_send_sync::<SpawnedProcess>();
    assert_send_sync::<RemoteSpawner>();
    assert_send_sync::<RemoteAgent>();
    assert_send_sync::<RemoteAgentConfig>();
    assert_send_sync::<RemoteProtocol>();
    assert_clone::<SubprocessSpawner>();
    assert_clone::<SubprocessConfig>();
    assert_clone::<RemoteSpawner>();
    assert_clone::<RemoteAgentConfig>();
    assert_clone::<RemoteProtocol>();
    assert_serde::<SubprocessConfig>();
    assert_serde::<RemoteAgentConfig>();
    assert_serde::<RemoteProtocol>();

    // Both implemented backends coerce to the `AgentSpawner` seam.
    let spawners: [Arc<dyn AgentSpawner>; 2] = [
        Arc::new(SubprocessSpawner::new(SubprocessConfig::default())),
        Arc::new(RemoteSpawner::new("http://127.0.0.1:1")),
    ];
    assert_eq!(spawners.len(), 2);

    // The remote agent and the client factory both hand back `RemoteClient`.
    let clients: [Arc<dyn RemoteClient>; 2] = [
        Arc::new(RemoteAgent::new(RemoteAgentConfig::new(
            "remote-a1",
            "http://127.0.0.1:1",
        ))),
        RemoteProtocol::Http
            .client(RemoteAgentConfig::new("remote-a2", "http://127.0.0.1:1"))
            .unwrap(),
    ];
    assert_eq!(clients.len(), 2);

    let frame = SpawnMessage::new(SpawnMessageKind::Input, json!({"q": 1}));
    let line = frame_bytes(&frame).unwrap();
    assert_eq!(
        parse_frame_line(&line, DEFAULT_MAX_FRAME_BYTES)
            .unwrap()
            .map(|parsed| parsed.kind),
        Some(SpawnMessageKind::Input)
    );
    let _ = read_frame::<tokio::io::BufReader<tokio::io::Empty>>;
    assert_eq!(DEFAULT_MAX_FRAME_BYTES, 8 * 1024 * 1024);

    // The message-queue and A2A transports stay typed `Unsupported`.
    for protocol in [RemoteProtocol::Mq, RemoteProtocol::A2a] {
        let error = create_remote_client(
            protocol,
            RemoteAgentConfig::new("deferred", "http://127.0.0.1:1"),
        )
        .err()
        .expect("a deferred transport never builds a client");
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        assert_eq!(
            serde_json::to_value(protocol).unwrap(),
            json!(protocol.as_str())
        );
    }
}

#[test]
fn the_spawn_seam_extensions_are_flat_on_the_crate_root() {
    use openjiuwen_runner::{
        DeferredSpawner, ExternalCommandSpawner, InProcessAgent, InProcessFactory,
        InProcessHandler, InProcessSpawner, MonitoredSpawnedAgent, RemoteSpawner, RunMonitor,
        SpawnBackend, SpawnHealth, SpawnerRegistry, SubprocessConfig, SubprocessSpawner,
    };

    /// The admitted-and-wrapped constructor as a plain function pointer.
    type TrackFn =
        fn(Arc<dyn SpawnedAgent>, Arc<RunMonitor>, Option<&str>, Value) -> MonitoredSpawnedAgent;

    assert_send_sync::<SpawnBackend>();
    assert_send_sync::<SpawnHealth>();
    assert_send_sync::<SpawnerRegistry>();
    assert_send_sync::<DeferredSpawner>();
    assert_send_sync::<ExternalCommandSpawner>();
    assert_send_sync::<InProcessSpawner>();
    assert_send_sync::<InProcessAgent>();
    assert_send_sync::<InProcessFactory>();
    assert_send_sync::<MonitoredSpawnedAgent>();
    assert_send_sync::<Arc<dyn InProcessHandler>>();

    assert_clone::<SpawnBackend>();
    assert_clone::<SpawnHealth>();
    assert_clone::<DeferredSpawner>();
    assert_clone::<ExternalCommandSpawner>();
    assert_clone::<InProcessSpawner>();
    assert_clone::<InProcessFactory>();

    assert_serde::<SpawnBackend>();
    assert_serde::<SpawnHealth>();
    roundtrip(&SpawnBackend::ExternalCommand);
    roundtrip(&SpawnHealth::healthy(SpawnBackend::InProcess, None));

    // All four implemented backends coerce to the spawner seam, and every
    // backend name is its `snake_case` wire name.
    let in_process: InProcessFactory = Arc::new(|_spec: &SpawnSpec| {
        Err(openjiuwen_common::error::Error::unsupported(
            "this suite builds no hosted handler",
        ))
    });
    let spawners: [Arc<dyn AgentSpawner>; 4] = [
        Arc::new(SubprocessSpawner::new(SubprocessConfig::default())),
        Arc::new(ExternalCommandSpawner::new("agent")),
        Arc::new(InProcessSpawner::new(in_process)),
        Arc::new(RemoteSpawner::new("http://127.0.0.1:1")),
    ];
    assert_eq!(spawners.len(), 4);
    let deferred: Arc<dyn AgentSpawner> =
        Arc::new(DeferredSpawner::for_backend(SpawnBackend::RemoteA2a));
    assert_eq!(deferred.backend(), SpawnBackend::RemoteA2a);
    let registry = SpawnerRegistry::with_deferred();
    assert_eq!(registry.len(), 2);

    let _: Option<TrackFn> = Some(MonitoredSpawnedAgent::track);
}

#[test]
fn spawn_backend_names_implemented_flags_and_display_are_pinned() {
    use openjiuwen_runner::{DeferredSpawner, SpawnBackend};

    let implemented = [
        SpawnBackend::Subprocess,
        SpawnBackend::ExternalCommand,
        SpawnBackend::InProcess,
        SpawnBackend::RemoteHttp,
    ];
    let deferred = [SpawnBackend::RemoteMq, SpawnBackend::RemoteA2a];
    for backend in implemented {
        assert!(backend.is_implemented(), "{}", backend.as_str());
    }
    for backend in deferred {
        assert!(!backend.is_implemented(), "{}", backend.as_str());
        assert_eq!(
            DeferredSpawner::for_backend(backend).backend_name(),
            backend
        );
    }
    assert!(!SpawnBackend::Unknown.is_implemented());

    let every = [
        SpawnBackend::Unknown,
        SpawnBackend::Subprocess,
        SpawnBackend::ExternalCommand,
        SpawnBackend::InProcess,
        SpawnBackend::RemoteHttp,
        SpawnBackend::RemoteMq,
        SpawnBackend::RemoteA2a,
    ];
    for backend in every {
        assert_eq!(
            serde_json::to_value(backend).unwrap(),
            json!(backend.as_str()),
            "{}",
            backend.as_str()
        );
        assert_eq!(backend.to_string(), backend.as_str());
    }
    assert_eq!(SpawnBackend::default(), SpawnBackend::Unknown);
}

/// A handle implementing only the required seam methods, so the defaulted
/// accessors are reachable through the trait object.
struct SeamChild {
    /// Whether the handle reports itself alive.
    alive: std::sync::atomic::AtomicBool,
}

#[async_trait::async_trait]
#[allow(clippy::unused_async_trait_impl)]
impl SpawnedAgent for SeamChild {
    fn id(&self) -> &'static str {
        "seam-child"
    }

    fn is_alive(&self) -> bool {
        self.alive.load(std::sync::atomic::Ordering::SeqCst)
    }

    fn pid(&self) -> Option<u32> {
        None
    }

    async fn send(&self, _inputs: Value) -> Result<AgentOutput> {
        Ok(AgentOutput::default())
    }

    fn stream(&self, _inputs: Value) -> BoxStream<'static, Result<AgentChunk>> {
        Box::pin(futures::stream::empty())
    }

    async fn shutdown(&self, _timeout: Duration) -> Result<bool> {
        Ok(true)
    }

    async fn force_kill(&self) -> Result<()> {
        Ok(())
    }
}

#[tokio::test]
async fn the_seam_accessors_are_callable_through_the_trait_object() {
    use openjiuwen_common::error::ErrorKind;
    use openjiuwen_runner::{SpawnBackend, SpawnHealth};

    let child: Arc<dyn SpawnedAgent> = Arc::new(SeamChild {
        alive: std::sync::atomic::AtomicBool::new(true),
    });

    assert_eq!(child.backend(), SpawnBackend::Unknown);
    assert!(child.is_alive());
    assert!(child.is_healthy());
    let snapshot: SpawnHealth = child.health_snapshot();
    assert_eq!(snapshot.backend, SpawnBackend::Unknown);
    assert!(snapshot.alive && snapshot.healthy);
    assert_eq!(snapshot.last_error, None);

    assert_eq!(
        child.health().await.unwrap_err().kind(),
        ErrorKind::Unsupported
    );
    assert_eq!(
        child.cancel_task("t-1").await.unwrap_err().kind(),
        ErrorKind::Unsupported
    );
}

#[test]
fn callback_chains_filters_and_handler_builders_are_flat_on_the_crate_root() {
    // `on_and_wrap_stay_under_their_modules` below pins the deliberate scoping of
    // the free combinators; these are the types the crate root does export.
    use openjiuwen_runner::{
        AuthFilter, BuiltHandler, CallbackChain, ChainEntry, ChainEntryId, ChainReport,
        CircuitBreakerFilter, Clock, ConditionalFilter, ContextModifier, EntryOutcome, EntryRecord,
        EventFilter, FilterChain, FilterDecision, FilterFeedback, FilterInput, FilterRunOutcome,
        HandlerBuilder, KindFilter, LoggingFilter, ManualClock, NameFilter, ParamModifyFilter,
        PayloadTransform, PredicateFilter, RateLimitFilter, SessionFilter, TransformHandler,
        ValidationFilter, WrapRegistry, WrappedCall,
    };

    assert_send_sync::<CallbackChain>();
    assert_send_sync::<ChainEntry>();
    assert_send_sync::<ChainEntryId>();
    assert_send_sync::<ChainReport>();
    assert_send_sync::<EntryOutcome>();
    assert_send_sync::<EntryRecord>();
    assert_send_sync::<FilterChain>();
    assert_send_sync::<FilterDecision>();
    assert_send_sync::<FilterInput<'static>>();
    assert_send_sync::<FilterRunOutcome>();
    assert_send_sync::<Clock>();
    assert_send_sync::<ManualClock>();
    assert_send_sync::<HandlerBuilder>();
    assert_send_sync::<BuiltHandler>();
    assert_send_sync::<TransformHandler>();
    assert_send_sync::<WrappedCall>();
    assert_send_sync::<WrapRegistry>();
    assert_send_sync::<NameFilter>();
    assert_send_sync::<KindFilter>();
    assert_send_sync::<SessionFilter>();
    assert_send_sync::<RateLimitFilter>();
    assert_send_sync::<CircuitBreakerFilter>();
    assert_send_sync::<ValidationFilter>();
    assert_send_sync::<LoggingFilter>();
    assert_send_sync::<AuthFilter>();
    assert_send_sync::<ParamModifyFilter>();
    assert_send_sync::<ConditionalFilter>();
    assert_send_sync::<PredicateFilter>();
    assert_clone::<Clock>();
    assert_clone::<ManualClock>();
    assert_clone::<ChainEntryId>();
    assert_clone::<ContextModifier>();
    assert_clone::<PayloadTransform>();

    let _: Option<Arc<dyn EventFilter>> = None;
    let _: Option<&'static dyn FilterFeedback> = None;

    let chain = CallbackChain::new("audit");
    assert!(chain.is_empty());
    assert_eq!(chain.name(), "audit");
    let _ = FilterChain::new().with(NameFilter::new().with_prefix("record"));
    let _ = ManualClock::new().clock();
    let _ = HandlerBuilder::new(CallbackEvent::AgentInvokeOutput).with_priority(1);
}

#[test]
fn free_combinators_stay_under_their_modules() {
    use openjiuwen_runner::callback::decorator::{
        on, on_wrap, transform_io, wrap, PayloadCall, WrapHandler, WrapRegistration,
    };
    use openjiuwen_runner::callback::filters::{
        and, lenient, not, or, LenientFilter, NotFilter, OrFilter,
    };
    use openjiuwen_runner::{
        BuiltHandler, FilterChain, HandlerBuilder, NameFilter, TransformHandler, WrappedCall,
    };

    // The names are too generic for the crate root, so the combinators are
    // reachable only through their modules; the builder and the handler they
    // produce are flat.
    let builder: HandlerBuilder = on(CallbackEvent::AgentInvokeOutput);
    let built: BuiltHandler = builder.sync(|_: &CallbackContext| Ok(CallbackVerdict::Continue));
    assert_eq!(built.event(), &CallbackEvent::AgentInvokeOutput);
    let _: fn(CallbackEvent, PayloadCall) -> WrappedCall = wrap;
    let _: fn(CallbackEvent, Arc<dyn WrapHandler>) -> WrapRegistration = on_wrap;
    let _: TransformHandler = transform_io("suite");
    let name = NameFilter::new().with_prefix("record");
    let _: FilterChain = and(name.clone(), name.clone());
    let _: OrFilter = or(name.clone(), name.clone());
    let _: NotFilter = not(name.clone());
    let _: LenientFilter = lenient(name);
}

#[test]
fn monitor_hitl_and_pool_surfaces_are_flat_on_the_crate_root() {
    use openjiuwen_runner::{
        ActiveRunInfo, ActiveRunSpec, ActiveTeamInfo, DeliverResult, InteractKind, InteractPayload,
        InteractionHub, PoolError, PoolFrame, PoolKey, PoolLease, PoolRequest, PoolRun,
        PoolShutdown, RunEvent, RunHandle, RunKind, RunMonitor, RunOutcome, RunPhase, RunStart,
        RunStatus, RunTarget, TeamPool, DEFAULT_TEAM_POOL_MAX_RUNS, EVENT_CHANNEL_CAPACITY,
        EVENT_HISTORY_LIMIT, INBOX_CAPACITY, LIVE_RUN_LIMIT,
    };

    assert_send_sync::<RunMonitor>();
    assert_send_sync::<RunStatus>();
    assert_send_sync::<RunEvent>();
    assert_send_sync::<RunStart>();
    assert_send_sync::<RunOutcome>();
    assert_send_sync::<RunKind>();
    assert_send_sync::<RunPhase>();
    assert_clone::<RunMonitor>();
    assert_serde::<RunStatus>();
    assert_serde::<RunEvent>();
    assert_serde::<RunStart>();
    assert_serde::<RunOutcome>();

    assert_send_sync::<InteractionHub>();
    assert_send_sync::<RunHandle>();
    assert_send_sync::<ActiveRunSpec>();
    assert_send_sync::<ActiveRunInfo>();
    assert_send_sync::<RunTarget>();
    assert_send_sync::<InteractPayload>();
    assert_send_sync::<InteractKind>();
    assert_send_sync::<DeliverResult>();
    assert_clone::<InteractionHub>();
    assert_clone::<RunHandle>();
    assert_serde::<InteractPayload>();
    assert_serde::<InteractKind>();

    assert_send_sync::<TeamPool>();
    assert_send_sync::<PoolRun>();
    assert_send_sync::<PoolLease>();
    assert_send_sync::<PoolRequest>();
    assert_send_sync::<PoolKey>();
    assert_send_sync::<PoolFrame>();
    assert_send_sync::<PoolError>();
    assert_send_sync::<PoolShutdown>();
    assert_send_sync::<ActiveTeamInfo>();
    assert_clone::<TeamPool>();
    assert_clone::<PoolRun>();
    assert_clone::<PoolKey>();
    assert_serde::<PoolKey>();
    assert_serde::<PoolFrame>();
    assert_serde::<ActiveTeamInfo>();

    let monitor = RunMonitor::new();
    let created: RunStatus = monitor.begin(RunStart::new(RunKind::Agent, "assistant"));
    assert_eq!(created.phase, RunPhase::Queued);
    monitor
        .finish(&created.run_id, RunOutcome::completed())
        .unwrap();
    assert_eq!(monitor.list_active().len(), 0);
    assert_eq!(EVENT_HISTORY_LIMIT, 128);
    assert_eq!(LIVE_RUN_LIMIT, 64);
    assert_eq!(EVENT_CHANNEL_CAPACITY, 256);

    let hub = InteractionHub::new();
    let handle: RunHandle = hub.register(ActiveRunSpec::new("assistant"));
    assert_eq!(hub.len(), 1);
    assert_ne!(handle.run_id(), "");
    assert_eq!(INBOX_CAPACITY, 64);

    let pool = TeamPool::new();
    assert!(pool.is_empty());
    assert_eq!(pool.capacity(), DEFAULT_TEAM_POOL_MAX_RUNS);
}

#[test]
fn resource_managers_are_flat_on_the_crate_root() {
    use openjiuwen_runner::{
        DefaultMcpConnector, McpConnector, McpServerInfo, McpServerMgr, McpServerState, ModelCard,
        ModelMgr, ModelProvider, PromptCard, PromptMgr, ResourceKind, SysOperationMgr,
        SysOperationProvider, Tag, TagMatchStrategy, TagMgr, TagUpdateStrategy, GLOBAL_TAG,
    };

    assert_send_sync::<TagMgr>();
    assert_send_sync::<ModelMgr>();
    assert_send_sync::<PromptMgr>();
    assert_send_sync::<SysOperationMgr>();
    assert_send_sync::<McpServerMgr>();
    assert_send_sync::<ModelProvider>();
    assert_send_sync::<SysOperationProvider>();
    assert_send_sync::<DefaultMcpConnector>();
    assert_send_sync::<Tag>();
    assert_send_sync::<ResourceKind>();
    assert_send_sync::<TagMatchStrategy>();
    assert_send_sync::<TagUpdateStrategy>();
    assert_send_sync::<ModelCard>();
    assert_send_sync::<PromptCard>();
    assert_send_sync::<McpServerInfo>();
    assert_send_sync::<McpServerState>();
    assert_clone::<Tag>();
    assert_clone::<DefaultMcpConnector>();
    assert_serde::<Tag>();
    assert_serde::<ResourceKind>();
    assert_serde::<TagMatchStrategy>();
    assert_serde::<TagUpdateStrategy>();
    assert_serde::<ModelCard>();
    assert_serde::<PromptCard>();
    assert_serde::<McpServerInfo>();
    assert_serde::<McpServerState>();

    let connectors: [Arc<dyn McpConnector>; 1] = [Arc::new(DefaultMcpConnector)];
    assert_eq!(connectors.len(), 1);

    let manager = ResourceMgr::new();
    let tags: &TagMgr = manager.tags();
    assert_eq!(tags.list_tags().len(), 0);
    let _: &ModelMgr = manager.models();
    let _: &PromptMgr = manager.prompts();
    let _: &SysOperationMgr = manager.sysops();
    let _: &McpServerMgr = manager.mcp_servers();
    assert_eq!(GLOBAL_TAG, "__global__");
    assert!(Tag::global().is_global());
    assert_eq!(manager.tags().resources(ResourceKind::Tool).len(), 0);
}
