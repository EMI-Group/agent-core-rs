//! Agent execution: [`invoke_agent`] and [`stream_agent`].
//!
//! Ports `Runner._prepare_agent`, `Runner._create_agent_session`,
//! `Runner.run_agent`, and `Runner.run_agent_streaming`.
//!
//! The runner owns a session only when the caller did not provide one: an owned
//! [`AgentSession`] is created with the resolved session id and committed by
//! [`AgentSession::post_run`] after the run (on completion or on drop), while a
//! caller-provided session is used as-is and never committed, closed, or ended.

use std::sync::Arc;
use std::time::Instant;

use futures::stream::{BoxStream, StreamExt};
use openjiuwen_agent::{AgentCard, AgentChunk, AgentOutput};
use openjiuwen_common::error::{Error, Result};
use openjiuwen_session::{AgentSession, AgentSessionOptions, Session};
use serde_json::{json, Value};
use uuid::Uuid;

use super::{
    effective_deadline, next_stream_step, record_stream_outcome, resolve_agent, resolve_session_id,
    run_with_limits, stream_frame_output, trace_failure, trace_finish, trace_start, AgentRef,
    RunOptions, RunServices, RunSpan, StreamStep, Teardown, LOG_TARGET,
};
use crate::callback::{CallbackContext, CallbackEvent};

/// Runs one agent invocation to completion.
///
/// A distributed-configured context is refused first:
/// [`RunServices::require_local`] returns `Error::Unsupported` naming the
/// deferred transports before anything else happens.
///
/// The reference is resolved next: an unregistered id is
/// [`openjiuwen_common::error::Error::NotFound`] before any session is created
/// or callback fires. Then `AgentInvokeInput` is fired, the session is resolved
/// (caller-owned, or a fresh [`AgentSession`] announced with
/// `AgentSessionCreated`), the invocation runs under the run deadline and
/// cancellation token, `AgentInvokeOutput` is fired, and an owned session is
/// committed by `post_run`.
///
/// A caller-provided session is never committed, closed, or ended by the
/// runner; an owned session's teardown failure is logged and never replaces a
/// successful output.
///
/// # Errors
///
/// `NotFound` for a missing id, the abort error of an `Abort` verdict on
/// `AgentInvokeInput`/`AgentInvokeOutput`, `Cancelled` when the cancellation
/// token fires, [`crate::error::run_timeout`] when the deadline expires, and
/// whatever the agent itself reports.
#[allow(clippy::needless_pass_by_value)]
pub async fn invoke_agent(
    services: RunServices,
    agent: AgentRef,
    inputs: Value,
    options: RunOptions,
) -> Result<AgentOutput> {
    services.require_local()?;
    let run_id = Uuid::new_v4().to_string();
    let started = Instant::now();

    let agent = resolve_agent(&services, &agent)?;
    let card = agent.card().clone();

    let mut context = CallbackContext::new(CallbackEvent::AgentInvokeInput)
        .with_component(card.id.clone())
        .with_input(inputs.clone());
    if let Some(session) = &options.session {
        context = context.with_session(Arc::clone(session));
    }
    services
        .callbacks
        .trigger(CallbackEvent::AgentInvokeInput, &context)
        .await?;

    let (session, owned) = prepare_session(&services, &card, &options, &inputs).await?;
    // Observation seams: registered only now that every setup step that can fail
    // has succeeded. A session the runner created is attached to the hub watcher;
    // a caller-owned one is not (the runner does not install a watcher on a
    // session it does not own). The binding is declared before the status guard so
    // the guard finalises the run before the hub entry disappears.
    let detail = json!({"component": card.id});
    let binding = crate::hitl::bind_run(
        services.hitl.as_ref(),
        crate::hitl::ActiveRunSpec::for_execution(
            &run_id,
            &card.id,
            None,
            Some(session.id()),
            &options.cancellation,
            detail.clone(),
        ),
        owned.as_ref().map(|_| Arc::clone(&session)),
    );
    let mut status = crate::monitor::RunStatusGuard::begin(
        services.monitor.clone(),
        &run_id,
        crate::monitor::RunStart::new(crate::monitor::RunKind::Agent, card.id.clone())
            .with_session(session.id())
            .with_detail(detail.clone()),
    );
    status.running(detail.clone());
    trace_start("agent", &run_id, &card.id, session.id());
    let span = RunSpan::start(session.tracer(), "run_agent", inputs.clone());

    let deadline = effective_deadline(&services, &options);
    let result = run_with_limits(
        "run_agent",
        agent.invoke(inputs, Some(Arc::clone(&session))),
        &options.cancellation,
        deadline,
    )
    .await;

    let teardown = agent_teardown(owned);
    match result {
        Ok(output) => {
            let value = output.to_value();
            span.finish(value.clone());
            let context = CallbackContext::new(CallbackEvent::AgentInvokeOutput)
                .with_component(card.id.clone())
                .with_output(value)
                .with_session(Arc::clone(&session));
            let callback = services
                .callbacks
                .trigger(CallbackEvent::AgentInvokeOutput, &context)
                .await;
            teardown.finish().await;
            trace_finish("agent", &run_id, &card.id, session.id(), started.elapsed());
            // The callback abort wins, exactly as before: the run reports the
            // abort error, which the status mirrors.
            match &callback {
                Ok(_) => status.ok(detail),
                Err(error) => status.fail(error, detail),
            }
            binding.unregister();
            callback?;
            Ok(output)
        }
        Err(error) => {
            span.fail(&error);
            teardown.finish().await;
            trace_failure(
                "agent",
                &run_id,
                &card.id,
                session.id(),
                started.elapsed(),
                &error,
            );
            status.fail(&error, detail);
            binding.unregister();
            Err(error)
        }
    }
}

/// Runs one agent invocation as a stream of [`AgentChunk`] frames.
///
/// A distributed-configured context refuses as this stream's **single** terminal
/// `Err` frame ([`RunServices::require_local`]): nothing else happens — no
/// reference resolution, no session, no callback, no monitor or hub status.
///
/// The stream is returned immediately, so a resolution or session-setup failure
/// (a missing id, an aborting `AgentSessionCreated`/`AgentStreamInput` handler)
/// is delivered as a one-element stream yielding that error. On success the
/// stream fires `AgentStreamInput` first, fires `AgentStreamOutput` for every
/// frame with a 0-based `index`, and enforces `options.cancellation` between
/// frames plus the deadline as a **total** budget.
///
/// Termination yields exactly one terminal error ({cancelled}, {timeout}, or
/// the inner stream's own failure) and then ends; an owned session is committed
/// after the last frame and also when the consumer drops the stream early.
#[allow(clippy::needless_pass_by_value)]
#[allow(clippy::too_many_lines)] // one linear stream loop; the observation seam is four lines of setup
pub fn stream_agent(
    services: RunServices,
    agent: AgentRef,
    inputs: Value,
    options: RunOptions,
) -> BoxStream<'static, Result<AgentChunk>> {
    let run_id = Uuid::new_v4().to_string();
    async_stream::stream! {
        // Mode enforcement is the stream's first statement: a distributed runner
        // refuses as the single terminal `Err` frame, before a reference is
        // resolved, a session is created, a callback fires, or a status exists.
        if let Err(error) = services.require_local() {
            yield Err(error);
            return;
        }
        let started = Instant::now();
        let agent = match resolve_agent(&services, &agent) {
            Ok(agent) => agent,
            Err(error) => {
                yield Err(error);
                return;
            }
        };
        let card = agent.card().clone();
        let (session, owned) = match prepare_session(&services, &card, &options, &inputs).await {
            Ok(prepared) => prepared,
            Err(error) => {
                yield Err(error);
                return;
            }
        };

        // Captured before the teardown takes ownership of the owned session.
        let attach_session = owned.as_ref().map(|_| Arc::clone(&session));
        let teardown = agent_teardown(owned);
        let context = CallbackContext::new(CallbackEvent::AgentStreamInput)
            .with_component(card.id.clone())
            .with_input(inputs.clone())
            .with_session(Arc::clone(&session));
        if let Err(error) = services
            .callbacks
            .trigger(CallbackEvent::AgentStreamInput, &context)
            .await
        {
            teardown.finish().await;
            yield Err(error);
            return;
        }

        // An aborting `AgentStreamInput` handler is a setup failure, so the run is
        // admitted only after it resolved. Only a runner-created session is
        // attached to the hub watcher.
        let detail = json!({"component": card.id});
        let binding = crate::hitl::bind_run(
            services.hitl.as_ref(),
            crate::hitl::ActiveRunSpec::for_execution(
                &run_id,
                &card.id,
                None,
                Some(session.id()),
                &options.cancellation,
                detail.clone(),
            ),
            attach_session,
        );
        let mut status = crate::monitor::RunStatusGuard::begin(
            services.monitor.clone(),
            &run_id,
            crate::monitor::RunStart::new(crate::monitor::RunKind::Agent, card.id.clone())
                .with_session(session.id())
                .with_detail(detail.clone()),
        );
        status.running(detail.clone());

        trace_start("agent", &run_id, &card.id, session.id());
        let span = RunSpan::start(session.tracer(), "run_agent_stream", inputs.clone());
        let deadline = effective_deadline(&services, &options);
        let deadline_at = deadline.map(|limit| (tokio::time::Instant::now() + limit, limit));
        let mut inner = agent.stream(inputs, Some(Arc::clone(&session)));
        let mut index: i64 = 0;
        let mut failure: Option<Error> = None;

        loop {
            if options.cancellation.is_cancelled() {
                failure = Some(Error::cancelled());
                break;
            }
            match next_stream_step(&mut inner, &options.cancellation, deadline_at).await {
                StreamStep::Cancelled => {
                    failure = Some(Error::cancelled());
                    break;
                }
                StreamStep::TimedOut => {
                    if let Some((_, limit)) = deadline_at {
                        failure = Some(crate::error::run_timeout("run_agent", limit));
                    }
                    break;
                }
                StreamStep::Frame(Some(Ok(chunk))) => {
                    // A run without subscribers must not serialise every frame for
                    // a callback that would return immediately.
                    if let Some(output) = stream_frame_output(
                        &services.callbacks,
                        CallbackEvent::AgentStreamOutput,
                        || chunk.to_value(),
                    ) {
                        let context = CallbackContext::new(CallbackEvent::AgentStreamOutput)
                            .with_component(card.id.clone())
                            .with_index(index)
                            .with_output(output)
                            .with_session(Arc::clone(&session));
                        if let Err(error) = services
                            .callbacks
                            .trigger(CallbackEvent::AgentStreamOutput, &context)
                            .await
                        {
                            failure = Some(error);
                            break;
                        }
                    }
                    index += 1;
                    yield Ok(chunk);
                }
                StreamStep::Frame(Some(Err(error))) => {
                    failure = Some(error);
                    break;
                }
                StreamStep::Frame(None) => break,
            }
        }

        record_stream_outcome(
            &span,
            "agent",
            &run_id,
            &card.id,
            session.id(),
            started,
            failure.as_ref(),
        );
        teardown.finish().await;
        match &failure {
            Some(error) => status.fail(error, detail),
            None => status.ok(detail),
        }
        binding.unregister();
        if let Some(error) = failure {
            yield Err(error);
        }
    }
    .boxed()
}

/// Resolves the run's session: the caller's, or a fresh owned [`AgentSession`].
///
/// Creating a session fires `AgentSessionCreated`; a handler that aborts turns
/// the creation into an error.
async fn prepare_session(
    services: &RunServices,
    card: &AgentCard,
    options: &RunOptions,
    inputs: &Value,
) -> Result<(Arc<dyn Session>, Option<Arc<AgentSession>>)> {
    if let Some(session) = &options.session {
        return Ok((Arc::clone(session), None));
    }
    let owned = Arc::new(AgentSession::with_options(AgentSessionOptions {
        session_id: Some(resolve_session_id(inputs, options, &services.config)),
        agent_id: Some(card.id.clone()),
        agent_name: Some(card.name.clone()).filter(|name| !name.is_empty()),
        envs: options.envs.clone(),
        metadata: options.metadata.clone(),
        checkpointer: options
            .checkpointer
            .clone()
            .or_else(|| services.checkpointer.clone()),
        ..AgentSessionOptions::default()
    }));
    let session: Arc<dyn Session> = Arc::clone(&owned) as Arc<dyn Session>;
    let created = services
        .callbacks
        .trigger(
            CallbackEvent::AgentSessionCreated,
            &CallbackContext::new(CallbackEvent::AgentSessionCreated)
                .with_component(card.id.clone())
                .with_session(Arc::clone(&session)),
        )
        .await;
    if let Err(error) = created {
        if let Err(teardown_error) = owned.post_run().await {
            tracing::warn!(
                target: LOG_TARGET,
                error = %teardown_error,
                "agent session teardown after an aborted setup failed"
            );
        }
        return Err(error);
    }
    Ok((session, Some(owned)))
}

/// Builds the teardown of an owned agent session (`post_run`, or nothing).
fn agent_teardown(session: Option<Arc<AgentSession>>) -> Teardown {
    match session {
        None => Teardown::none(),
        Some(session) => Teardown::new(async move {
            if let Err(error) = session.post_run().await {
                tracing::warn!(
                    target: LOG_TARGET,
                    error = %error,
                    "agent session post_run failed"
                );
            }
        }),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use futures::StreamExt;
    use openjiuwen_agent::{Agent, AgentCard};
    use openjiuwen_session::MemoryCheckpointer;
    use serde_json::json;
    use std::sync::Mutex;
    use std::time::Duration;
    use tokio_util::sync::CancellationToken;

    use crate::callback::{CallbackFramework, CallbackVerdict};
    use crate::config::RunnerConfig;
    use crate::monitor::{RunEvent, RunPhase};
    use crate::resources::ResourceMgr;

    /// An agent whose invoke/stream behaviour the tests script.
    struct FakeAgent {
        card: AgentCard,
        delay: Option<Duration>,
        chunks: Vec<AgentChunk>,
    }

    impl FakeAgent {
        fn new(name: &str) -> Self {
            let mut card = AgentCard::new(name);
            card.id = name.to_owned();
            Self {
                card,
                delay: None,
                chunks: vec![
                    AgentChunk {
                        kind: "llm_output".to_owned(),
                        index: 0,
                        payload: json!("one"),
                    },
                    AgentChunk {
                        kind: "answer".to_owned(),
                        index: 1,
                        payload: json!("two"),
                    },
                ],
            }
        }

        fn slowed(mut self, delay: Duration) -> Self {
            self.delay = Some(delay);
            self
        }
    }

    #[async_trait::async_trait]
    impl Agent for FakeAgent {
        fn card(&self) -> &AgentCard {
            &self.card
        }

        async fn invoke(
            &self,
            inputs: Value,
            _session: Option<Arc<dyn Session>>,
        ) -> Result<AgentOutput> {
            if let Some(delay) = self.delay {
                tokio::time::sleep(delay).await;
            }
            Ok(AgentOutput {
                output: format!("ok:{}", inputs["query"].as_str().unwrap_or_default()),
                ..AgentOutput::default()
            })
        }

        fn stream(
            &self,
            _inputs: Value,
            _session: Option<Arc<dyn Session>>,
        ) -> BoxStream<'static, Result<AgentChunk>> {
            let chunks = self.chunks.clone();
            Box::pin(futures::stream::iter(chunks.into_iter().map(Ok)))
        }
    }

    /// A services bundle with a recording handler for every agent event.
    fn services(events: &Arc<Mutex<Vec<CallbackEvent>>>) -> RunServices {
        let services = RunServices::new(
            ResourceMgr::new(),
            CallbackFramework::new(),
            RunnerConfig::default().with_default_session_id("runner-default"),
            None,
        );
        for event in [
            CallbackEvent::AgentInvokeInput,
            CallbackEvent::AgentInvokeOutput,
            CallbackEvent::AgentStreamInput,
            CallbackEvent::AgentStreamOutput,
            CallbackEvent::AgentSessionCreated,
        ] {
            let events = Arc::clone(events);
            services
                .callbacks
                .register_sync(event, 0, move |ctx: &CallbackContext| {
                    events.lock().unwrap().push(ctx.event.clone());
                    Ok(CallbackVerdict::Continue)
                });
        }
        services
    }

    fn fake(name: &str) -> AgentRef {
        AgentRef::of(Arc::new(FakeAgent::new(name)))
    }

    // ports tests/unit_tests/core/runner/test_runner.py (run_agent callback order)
    #[tokio::test]
    async fn invoke_fires_input_then_output_and_returns_the_answer() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let services = services(&events);
        let output = invoke_agent(
            services,
            fake("fake"),
            json!({"query": "hi", "conversation_id": "conv-9"}),
            RunOptions::default(),
        )
        .await
        .unwrap();
        assert_eq!(output.output, "ok:hi");
        assert_eq!(
            *events.lock().unwrap(),
            vec![
                CallbackEvent::AgentInvokeInput,
                CallbackEvent::AgentSessionCreated,
                CallbackEvent::AgentInvokeOutput,
            ]
        );
    }

    #[tokio::test]
    async fn invoke_uses_the_conversation_id_for_the_owned_session() {
        let session_id = Arc::new(Mutex::new(None));
        let services = RunServices::new(
            ResourceMgr::new(),
            CallbackFramework::new(),
            RunnerConfig::default().with_default_session_id("runner-default"),
            Some(Arc::new(MemoryCheckpointer::new())),
        );
        let observed = Arc::clone(&session_id);
        services.callbacks.register_sync(
            CallbackEvent::AgentSessionCreated,
            0,
            move |ctx: &CallbackContext| {
                *observed.lock().unwrap() = ctx.session_id().map(ToOwned::to_owned);
                Ok(CallbackVerdict::Continue)
            },
        );
        invoke_agent(
            services.clone(),
            fake("fake"),
            json!({"query": "hi", "conversation_id": "conv-9"}),
            RunOptions::default(),
        )
        .await
        .unwrap();
        assert_eq!(session_id.lock().unwrap().as_deref(), Some("conv-9"));
        // Without a conversation id the runner's default applies.
        invoke_agent(
            services,
            fake("fake"),
            json!({"query": "hi"}),
            RunOptions::default(),
        )
        .await
        .unwrap();
        assert_eq!(
            session_id.lock().unwrap().as_deref(),
            Some("runner-default")
        );
    }

    #[tokio::test]
    async fn invoke_reports_an_unregistered_id_before_any_callback() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let services = services(&events);
        let error = invoke_agent(
            services,
            AgentRef::id("missing"),
            json!({"query": "hi"}),
            RunOptions::default(),
        )
        .await
        .unwrap_err();
        assert_eq!(
            error.to_string(),
            "not_found: agent 'missing' is not registered"
        );
        assert!(events.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn a_cancelled_run_reports_cancellation() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let services = services(&events);
        let token = CancellationToken::new();
        token.cancel();
        let error = invoke_agent(
            services,
            AgentRef::of(Arc::new(
                FakeAgent::new("slow").slowed(Duration::from_secs(60)),
            )),
            json!({"query": "hi"}),
            RunOptions::default().with_cancellation(token),
        )
        .await
        .unwrap_err();
        assert!(error.is_cancelled());
        assert_eq!(error.to_string(), "cancelled");
    }

    #[tokio::test]
    async fn a_run_past_its_deadline_reports_a_timeout() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let services = services(&events);
        let error = invoke_agent(
            services,
            AgentRef::of(Arc::new(
                FakeAgent::new("slow").slowed(Duration::from_secs(60)),
            )),
            json!({"query": "hi"}),
            RunOptions::default().with_timeout(Duration::from_millis(5)),
        )
        .await
        .unwrap_err();
        assert_eq!(error.to_string(), "timeout: run_agent timed out after 5ms");
    }

    #[tokio::test]
    async fn streaming_fires_one_output_event_per_frame() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let services = services(&events);
        let frames: Vec<Result<AgentChunk>> = stream_agent(
            services,
            fake("fake"),
            json!({"query": "hi"}),
            RunOptions::default(),
        )
        .collect()
        .await;
        assert_eq!(frames.len(), 2);
        assert!(frames.iter().all(Result::is_ok));
        assert_eq!(
            *events.lock().unwrap(),
            vec![
                CallbackEvent::AgentSessionCreated,
                CallbackEvent::AgentStreamInput,
                CallbackEvent::AgentStreamOutput,
                CallbackEvent::AgentStreamOutput,
            ]
        );
    }

    #[tokio::test]
    async fn streaming_without_subscribers_yields_the_same_frames() {
        // No handler is registered at all: the stream must still deliver every
        // frame unchanged and never project one for the callback hot path.
        let services = RunServices::new(
            ResourceMgr::new(),
            CallbackFramework::new(),
            RunnerConfig::default().with_default_session_id("runner-default"),
            None,
        );
        let frames: Vec<Result<AgentChunk>> = stream_agent(
            services,
            fake("fake"),
            json!({"query": "hi"}),
            RunOptions::default(),
        )
        .collect()
        .await;
        assert_eq!(frames.len(), 2);
        let chunks: Vec<AgentChunk> = frames.into_iter().map(Result::unwrap).collect();
        assert_eq!(chunks[0].kind, "llm_output");
        assert_eq!(chunks[0].payload, json!("one"));
        assert_eq!(chunks[1].kind, "answer");
        assert_eq!(chunks[1].payload, json!("two"));
    }

    #[tokio::test]
    async fn a_stream_subscriber_receives_every_frame_in_order() {
        let observed = Arc::new(Mutex::new(Vec::new()));
        let services = RunServices::new(
            ResourceMgr::new(),
            CallbackFramework::new(),
            RunnerConfig::default().with_default_session_id("runner-default"),
            None,
        );
        let seen = Arc::clone(&observed);
        services.callbacks.register_sync(
            CallbackEvent::AgentStreamOutput,
            0,
            move |ctx: &CallbackContext| {
                seen.lock().unwrap().push((ctx.index, ctx.output.clone()));
                Ok(CallbackVerdict::Continue)
            },
        );
        let frames: Vec<Result<AgentChunk>> = stream_agent(
            services,
            fake("fake"),
            json!({"query": "hi"}),
            RunOptions::default(),
        )
        .collect()
        .await;
        assert_eq!(frames.len(), 2);
        let observed = observed.lock().unwrap().clone();
        assert_eq!(
            observed.iter().map(|(index, _)| *index).collect::<Vec<_>>(),
            vec![Some(0), Some(1)],
            "every frame arrives once, in order, under its 0-based index"
        );
        assert_eq!(observed[0].1.as_ref().unwrap()["payload"], json!("one"));
        assert_eq!(observed[1].1.as_ref().unwrap()["payload"], json!("two"));
    }

    #[tokio::test]
    async fn streaming_reports_a_missing_id_as_its_only_item() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let services = services(&events);
        let frames: Vec<Result<AgentChunk>> = stream_agent(
            services,
            AgentRef::id("missing"),
            json!({}),
            RunOptions::default(),
        )
        .collect()
        .await;
        assert_eq!(frames.len(), 1);
        assert_eq!(
            frames[0].as_ref().unwrap_err().to_string(),
            "not_found: agent 'missing' is not registered"
        );
    }

    // ------------------------------------------------- observation seams --

    /// Drains the transitions a monitor published, in order.
    fn phases(events: &mut tokio::sync::broadcast::Receiver<RunEvent>) -> Vec<RunPhase> {
        std::iter::from_fn(|| events.try_recv().ok())
            .map(|event| event.phase)
            .collect()
    }

    /// Drains the transitions a monitor published, polling instead of sleeping.
    ///
    /// A guarded drop publishes synchronously, so the first pass usually drains
    /// everything; the bounded retry only keeps the drain from racing a mirror that
    /// has not run yet.
    async fn drain(events: &mut tokio::sync::broadcast::Receiver<RunEvent>) -> Vec<RunEvent> {
        let mut seen = Vec::new();
        for _ in 0..100 {
            seen.extend(std::iter::from_fn(|| events.try_recv().ok()));
            if seen.last().is_some_and(|event| event.phase.is_terminal()) {
                break;
            }
            tokio::task::yield_now().await;
        }
        seen
    }

    #[tokio::test]
    async fn invoke_mirrors_the_run_and_unregisters_it() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let monitor = Arc::new(crate::monitor::RunMonitor::new());
        let hub = Arc::new(crate::hitl::InteractionHub::new());
        let mut transitions = monitor.subscribe();
        let services = services(&events)
            .with_monitor(Arc::clone(&monitor))
            .with_hitl(Arc::clone(&hub));

        // The hub entry is live while the run is in flight.
        let live = Arc::new(Mutex::new(Vec::new()));
        let observed = Arc::clone(&live);
        let peek = Arc::clone(&hub);
        services.callbacks.register_sync(
            CallbackEvent::AgentInvokeOutput,
            0,
            move |_ctx: &CallbackContext| {
                *observed.lock().unwrap() = peek.list_active();
                Ok(CallbackVerdict::Continue)
            },
        );

        invoke_agent(
            services,
            fake("fake"),
            json!({"query": "hi"}),
            RunOptions::default(),
        )
        .await
        .unwrap();

        let active = live.lock().unwrap();
        assert_eq!(active.len(), 1, "the run was registered while in flight");
        assert_eq!(active[0].label, "fake");
        assert!(active[0].session_id.is_some());
        assert!(hub.is_empty(), "the run was unregistered on completion");
        assert_eq!(
            phases(&mut transitions),
            vec![RunPhase::Queued, RunPhase::Running, RunPhase::Completed]
        );
        assert_eq!(monitor.list_active(), Vec::new(), "no run stays active");
    }

    #[tokio::test]
    async fn a_cancelled_run_is_mirrored_as_cancelled_and_unregistered() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let monitor = Arc::new(crate::monitor::RunMonitor::new());
        let hub = Arc::new(crate::hitl::InteractionHub::new());
        let mut transitions = monitor.subscribe();
        let services = services(&events)
            .with_monitor(Arc::clone(&monitor))
            .with_hitl(Arc::clone(&hub));
        let token = CancellationToken::new();
        token.cancel();

        let error = invoke_agent(
            services,
            AgentRef::of(Arc::new(
                FakeAgent::new("slow").slowed(Duration::from_secs(60)),
            )),
            json!({"query": "hi"}),
            RunOptions::default().with_cancellation(token),
        )
        .await
        .unwrap_err();

        assert!(error.is_cancelled());
        assert!(hub.is_empty());
        assert_eq!(
            phases(&mut transitions),
            vec![RunPhase::Queued, RunPhase::Running, RunPhase::Cancelled]
        );
    }

    #[tokio::test]
    async fn streaming_mirrors_the_run_and_unregisters_it() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let monitor = Arc::new(crate::monitor::RunMonitor::new());
        let hub = Arc::new(crate::hitl::InteractionHub::new());
        let mut transitions = monitor.subscribe();
        let services = services(&events)
            .with_monitor(Arc::clone(&monitor))
            .with_hitl(Arc::clone(&hub));

        let frames: Vec<Result<AgentChunk>> = stream_agent(
            services,
            fake("fake"),
            json!({"query": "hi"}),
            RunOptions::default(),
        )
        .collect()
        .await;

        assert_eq!(frames.len(), 2);
        assert!(hub.is_empty(), "the stream unregistered its run");
        assert_eq!(
            phases(&mut transitions),
            vec![RunPhase::Queued, RunPhase::Running, RunPhase::Completed]
        );
    }

    #[tokio::test]
    async fn a_stream_dropped_mid_flight_finishes_as_cancelled() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let monitor = Arc::new(crate::monitor::RunMonitor::new());
        let hub = Arc::new(crate::hitl::InteractionHub::new());
        let mut transitions = monitor.subscribe();
        let services = services(&events)
            .with_monitor(Arc::clone(&monitor))
            .with_hitl(Arc::clone(&hub));

        let mut stream = stream_agent(
            services,
            fake("fake"),
            json!({"query": "hi"}),
            RunOptions::default(),
        );
        // One poll yields frame 1 and leaves the generator suspended between
        // frames, so the run is still in flight when the stream goes away.
        assert!(
            matches!(stream.next().await, Some(Ok(_))),
            "the first frame is readable before the drop"
        );
        drop(stream);

        // Dropping the generator drops the status guard before the hub binding
        // (reverse declaration order), so the run is already terminal when
        // `unregister` mirrors its phase; the opposite order would report the run
        // as `Completed` instead of `Cancelled`.
        let published = drain(&mut transitions).await;
        assert_eq!(
            published
                .iter()
                .map(|event| event.phase)
                .collect::<Vec<_>>(),
            vec![RunPhase::Queued, RunPhase::Running, RunPhase::Cancelled],
            "a dropped stream finishes its run as cancelled"
        );
        let last = published.last().expect("a terminal transition");
        assert_eq!(last.detail, json!({"reason": "run dropped"}));
        assert!(hub.is_empty(), "the binding unregistered on drop");
        assert_eq!(monitor.list_active(), Vec::new(), "no run stays active");
    }

    #[tokio::test]
    async fn an_aborting_stream_input_handler_registers_nothing() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let monitor = Arc::new(crate::monitor::RunMonitor::new());
        let hub = Arc::new(crate::hitl::InteractionHub::new());
        let mut transitions = monitor.subscribe();
        let services = services(&events)
            .with_monitor(Arc::clone(&monitor))
            .with_hitl(Arc::clone(&hub));
        services.callbacks.register_sync(
            CallbackEvent::AgentStreamInput,
            0,
            |_ctx: &CallbackContext| Ok(CallbackVerdict::Abort(Error::validation("blocked"))),
        );

        let frames: Vec<Result<AgentChunk>> = stream_agent(
            services,
            fake("fake"),
            json!({"query": "hi"}),
            RunOptions::default(),
        )
        .collect()
        .await;

        assert_eq!(frames.len(), 1);
        assert!(hub.is_empty());
        assert!(
            phases(&mut transitions).is_empty(),
            "a setup failure admits no run"
        );
        assert_eq!(monitor.list_active(), Vec::new(), "no run stays active");
    }
}
