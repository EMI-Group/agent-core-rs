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
    roundtrip(&SpawnMessage::new(SpawnMessageKind::Input, json!({"q": 1})));
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
        .with_envs(std::collections::BTreeMap::new());
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
    assert!(!CallbackEvent::ALL.is_empty());
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
