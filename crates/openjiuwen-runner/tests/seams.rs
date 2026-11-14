#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Compile-level pinning of the runner's pluggable and distributed seams, plus
//! the typed refusals of every transport this build does not compile.
//!
//! Two jobs: declare each seam with its exact trait-object type, so a renamed,
//! moved, or re-signatured export stops compiling here; and prove that a seam
//! which cannot work answers `Error::Unsupported` naming the missing transport
//! instead of degrading. The seam *behaviours* live in `spawn_routing.rs`,
//! `spawn_seam.rs`, `message_queue.rs`, `runner_lifecycle.rs`, and
//! `security_binding.rs`.
//!
//! Team transports are owned by `openjiuwen-multiagent`; this suite reaches them
//! through that dependency only, because the runner drives teams through the
//! `Team` seam and holds no transport of its own.
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use openjiuwen_agent::Agent;
use openjiuwen_common::error::{ErrorKind, Result};
use openjiuwen_multiagent::{
    build_transport, MessageBusConfig, MessageEnvelope, MessageHandler as BusMessageHandler,
    TransportConfig, TransportKind,
};
use openjiuwen_runner::callback::{ChainEntry, ChainEntryId};
use openjiuwen_runner::resources::{AgentProvider, Provider};
use openjiuwen_runner::{
    build_message_queue, create_message_queue, AgentSpawner, CallbackChain, CallbackContext,
    CallbackHandler, CallbackResult, CallbackVerdict, EventFilter, InMemoryMessageQueue,
    MessageHandler, MessageQueue, MessageQueueConfig, MessageQueueKind, NameFilter, QueueMessage,
    QueueStats, RemoteAgent, RemoteAgentConfig, Runner, RunnerConfig, SpawnBackend, SpawnSpec,
    SpawnedAgent, SpawnerRegistry, SyncHandler, UnsupportedSpawner,
};
use openjiuwen_session::{Checkpointer, MemoryCheckpointer};
use openjiuwen_test_support::eventually;
use serde_json::{json, Value};

/// Bounded wait for a counter a queue worker updates asynchronously.
const BOUND: Duration = Duration::from_secs(5);

/// A callback handler with no opinion, for the trait-object pin.
struct ContinueHandler;

#[async_trait]
impl CallbackHandler for ContinueHandler {
    async fn handle(&self, _ctx: &CallbackContext) -> CallbackResult {
        Ok(CallbackVerdict::Continue)
    }
}

/// A queue handler that echoes its payload.
struct EchoHandler;

#[async_trait]
impl MessageHandler for EchoHandler {
    async fn handle(&self, message: QueueMessage) -> Result<Value> {
        Ok(message.payload)
    }
}

/// The `openjiuwen-multiagent` bus handler `build_transport` takes; the Pulsar
/// path refuses before it is ever reached.
struct NullBusHandler;

#[async_trait]
impl BusMessageHandler for NullBusHandler {
    async fn handle_p2p(&self, _envelope: &MessageEnvelope) -> Result<Value> {
        Ok(Value::Null)
    }

    async fn handle_pubsub(&self, _envelope: &MessageEnvelope, _subscriber: &str) -> Result<()> {
        Ok(())
    }
}

fn assert_send_sync<T: Send + Sync>() {}

/// Pins the resource-provider seam: [`Provider<T>`] is the
/// `Arc<dyn Fn() -> T + Send + Sync>` every registry stores, and
/// [`AgentProvider`] must stay exactly `Provider<Arc<dyn Agent>>`.
#[allow(dead_code)]
fn agent_provider_seam(provider: AgentProvider) -> Provider<Arc<dyn Agent>> {
    let provider: Arc<dyn Fn() -> Arc<dyn Agent> + Send + Sync> = provider;
    provider
}

/// Every pluggable seam, declared with its exact trait-object type.
///
/// This test asserts nothing: it fails to compile when a seam is renamed, moved
/// out of the crate root, or loses its `Send + Sync` bound.
#[test]
fn every_pluggable_seam_coerces_to_its_trait_object() {
    let runner = Runner::new();

    // The message queue seam, and the facade accessor that hands it out.
    let queue: Arc<dyn MessageQueue> = InMemoryMessageQueue::with_capacity(4);
    let installed: Arc<dyn MessageQueue> = runner.message_queue();
    assert!(!Arc::ptr_eq(&queue, &installed), "the runner owns its own");

    // The spawn seam: one spawner, many spawners, and the handles they return.
    let _spawner: Arc<dyn AgentSpawner> = Arc::new(UnsupportedSpawner);
    let _registry: SpawnerRegistry = SpawnerRegistry::with_deferred();
    let spawned: Arc<dyn SpawnedAgent> = Arc::new(RemoteAgent::new(RemoteAgentConfig::new(
        "seam-agent",
        "http://127.0.0.1:1",
    )));
    let _boxed: Box<dyn SpawnedAgent> = Box::new(RemoteAgent::new(RemoteAgentConfig::new(
        "seam-boxed",
        "http://127.0.0.1:1",
    )));
    assert!(
        !spawned.is_alive(),
        "a lazy HTTP handle has no connection yet"
    );

    // The persistence seam: built by `start`, never installed by a setter.
    let _checkpointer: Arc<dyn Checkpointer> = Arc::new(MemoryCheckpointer::new());
    let _configured: Option<Arc<dyn Checkpointer>> = runner.checkpointer();

    // The callback seams: a handler, a filter, a chain entry, and a chain.
    let handler: Arc<dyn CallbackHandler> = Arc::new(ContinueHandler);
    let _boxed_handler: Box<dyn CallbackHandler> =
        Box::new(SyncHandler::new(|_ctx: &CallbackContext| {
            Ok(CallbackVerdict::Continue)
        }));
    let filter: Arc<dyn EventFilter> = Arc::new(NameFilter::new().with_name("seam"));
    let entry: ChainEntry =
        ChainEntry::new("seam", Arc::clone(&handler)).with_filter(Arc::clone(&filter));
    assert_eq!(entry.name(), "seam");
    let chain = CallbackChain::new("seam-chain");
    let _entry_id: ChainEntryId = chain.add_handler(handler);

    // The queue handler seam, and the resource provider seam.
    let _message_handler: Arc<dyn MessageHandler> = Arc::new(EchoHandler);
    let _: Option<AgentProvider> = None;

    // The typed refusals are part of the inventory too.
    assert!(!SpawnBackend::RemoteMq.is_implemented());
    assert!(!SpawnBackend::RemoteA2a.is_implemented());
    assert!(SpawnBackend::Subprocess.is_implemented());
}

/// A deferred queue transport is the empty report, and every operation is the
/// typed refusal naming the transport.
#[tokio::test]
async fn a_deferred_queue_kind_is_a_typed_refusal() {
    for kind in [MessageQueueKind::Pulsar, MessageQueueKind::ZeroMq] {
        let config = MessageQueueConfig {
            kind,
            ..MessageQueueConfig::default()
        };

        // The fallible factory refuses outright…
        let error = create_message_queue(&config)
            .await
            .err()
            .expect("a deferred transport cannot be built");
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        assert!(error.to_string().contains(kind.as_str()), "{error}");

        // …while the infallible one installs the placeholder.
        let queue = build_message_queue(&config);
        let start = queue.start().await.expect_err("start refuses");
        assert_eq!(start.kind(), ErrorKind::Unsupported);
        let text = start.to_string();
        assert!(text.contains(kind.as_str()), "names the backend: {text}");
        assert!(text.contains("in_memory"), "points at the port: {text}");
        let publish = queue
            .publish("seam", QueueMessage::new(json!(1)))
            .await
            .expect_err("publish refuses");
        assert_eq!(publish.kind(), ErrorKind::Unsupported);
        assert!(queue.stats().is_empty(), "a deferred backend does no work");
        assert!(!queue.is_started());
    }
}

/// `SpawnerRegistry::with_deferred` answers the deferred backends by name.
#[tokio::test]
async fn the_deferred_spawn_backends_refuse_by_name() {
    let registry = SpawnerRegistry::with_deferred();
    assert_eq!(
        registry.backends(),
        vec![SpawnBackend::RemoteMq, SpawnBackend::RemoteA2a],
        "both deferred backends are pre-registered"
    );

    let mq = registry
        .spawn_with(
            SpawnBackend::RemoteMq,
            SpawnSpec::class_agent("agent", "Agent"),
        )
        .await
        .err()
        .expect("drunner reply-topic routing is deferred");
    assert_eq!(mq.kind(), ErrorKind::Unsupported);
    let text = mq.to_string();
    assert!(text.contains("Pulsar") && text.contains("ZeroMQ"), "{text}");

    // The same backend reached through a spec hint routes to the same refusal.
    let a2a = registry
        .spawn(SpawnSpec::class_agent("agent", "Agent").with_backend(SpawnBackend::RemoteA2a))
        .await
        .err()
        .expect("the A2A transport lives elsewhere");
    assert_eq!(a2a.kind(), ErrorKind::Unsupported);
    let text = a2a.to_string();
    assert!(text.contains("a2a"), "names the transport: {text}");
    assert!(
        text.contains("openjiuwen-extensions"),
        "points at where it lives: {text}"
    );
}

/// A distributed runner refuses to start rather than running in-process under a
/// distributed name.
#[tokio::test]
async fn a_distributed_runner_refuses_to_start() {
    let runner = Runner::with_config(RunnerConfig::default().with_distributed_mode(true));
    let error = runner
        .start()
        .await
        .expect_err("distributed mode is deferred");
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    let text = error.to_string();
    assert!(text.contains("distributed"), "names the mode: {text}");
    assert!(text.contains("deferred"), "says it is deferred: {text}");
    assert!(!runner.is_started(), "the runner stays stopped");
}

/// The Pulsar team transport is owned by `openjiuwen-multiagent` and refuses at
/// construction; the runner never gains a transport of its own.
#[test]
fn the_pulsar_team_transport_is_typed_unsupported() {
    let config = TransportConfig::new().with_kind(TransportKind::Pulsar);
    let error = build_transport(
        &config,
        "seam-team",
        MessageBusConfig::default(),
        Arc::new(NullBusHandler),
    )
    .expect_err("the Pulsar team transport is not implemented");
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    let text = error.to_string();
    assert!(text.contains("Pulsar"), "names the transport: {text}");
    assert!(text.contains("in_process"), "names what works: {text}");
}

/// `Runner::queue_stats` samples whichever backend is installed: empty before
/// `start`, then the live in-memory counters, and the counters survive `stop`.
#[tokio::test]
async fn queue_stats_samples_the_live_backend() {
    let runner = Runner::new();
    assert!(
        runner.queue_stats().is_empty(),
        "the configured backend is stopped before `start`"
    );

    runner.start().await.expect("start");
    let started = runner.queue_stats();
    assert!(started.started, "the started backend reports it");
    assert_eq!((started.topics, started.handlers), (0, 0));

    // Deliver twice through the queue the runner has installed.
    let queue = runner.message_queue();
    queue
        .add_handler("seam", Arc::new(EchoHandler))
        .await
        .expect("handler");
    for payload in [1, 2] {
        queue
            .publish("seam", QueueMessage::new(json!(payload)))
            .await
            .expect("publish");
    }
    eventually(BOUND, || async { runner.queue_stats().deliveries == 2 })
        .await
        .expect("two deliveries");

    let live = runner.queue_stats();
    assert!(live.started);
    assert_eq!(live.topics, 1);
    assert_eq!(live.handlers, 1);
    assert_eq!(live.dropped(), 0);
    assert_eq!(live.handler_errors, 0);

    runner.stop().await.expect("stop");
    let stopped = runner.queue_stats();
    assert!(!stopped.started, "`stop` leaves the queue stopped");
    assert_eq!(stopped.deliveries, 2, "the totals are monotonic");
    assert!(!stopped.is_empty());
}

/// `QueueStats` is cheap to hold across tasks and ships as `snake_case` JSON that
/// tolerates an unknown counter.
#[test]
fn queue_stats_is_send_sync_and_round_trips_json() {
    assert_send_sync::<QueueStats>();
    assert_send_sync::<MessageQueueKind>();

    let stats = QueueStats {
        started: true,
        topics: 2,
        handlers: 1,
        deliveries: 5,
        dropped_no_worker: 1,
        dropped_no_handler: 2,
        handler_errors: 3,
        rejected_not_started: 4,
    };
    let text = serde_json::to_string(&stats).expect("serialize");
    assert!(
        text.contains("\"dropped_no_handler\":2"),
        "snake_case: {text}"
    );
    let back: QueueStats = serde_json::from_str(&text).expect("deserialize");
    assert_eq!(back, stats);

    let partial: QueueStats = serde_json::from_value(json!({"deliveries": 7, "future_counter": 1}))
        .expect("an unknown field is tolerated");
    assert_eq!(partial.deliveries, 7);
    assert_eq!(partial.dropped(), 0);
    assert!(!partial.started);
}
