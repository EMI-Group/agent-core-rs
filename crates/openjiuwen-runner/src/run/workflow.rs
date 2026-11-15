//! Workflow execution: [`invoke_workflow`] and [`stream_workflow`].
//!
//! Ports `Runner._prepare_workflow`, `Runner._create_workflow_session`,
//! `Runner.run_workflow`, and `Runner.run_workflow_streaming`.
//!
//! The runner owns a session only when the caller did not provide one: an owned
//! [`WorkflowSession`] is closed with [`Session::close`] after the run (on
//! completion or on drop). Unlike an agent session, it is never committed by the
//! runner — the workflow engine commits the user inputs itself.
//!
//! Cancellation and the deadline are handed to the engine in
//! [`WorkflowRunOptions`]: the workflow owns its own race between the run, the
//! token, and the timeout, so the runner neither wraps nor re-times the call.

use std::sync::Arc;
use std::time::Instant;

use futures::stream::{BoxStream, StreamExt};
use openjiuwen_common::error::{Error, Result};
use openjiuwen_session::{Session, WorkflowSession, WorkflowSessionOptions};
use openjiuwen_workflow::{WorkflowChunk, WorkflowOutput, WorkflowRunOptions};
use serde_json::{json, Value};
use uuid::Uuid;

use super::{
    record_stream_outcome, resolve_session_id, resolve_workflow, to_value, trace_failure,
    trace_finish, trace_start, RunOptions, RunServices, RunSpan, Teardown, WorkflowRef, LOG_TARGET,
};
use crate::callback::{CallbackContext, CallbackEvent};

/// Runs one workflow invocation to completion.
///
/// A distributed-configured context is refused first:
/// [`RunServices::require_local`] returns `Error::Unsupported` naming the
/// deferred transports before anything else happens.
///
/// The reference is resolved next (a missing id is
/// [`openjiuwen_common::error::Error::NotFound`] before any session is created),
/// then `WorkflowInvokeInput` is fired, the session is resolved (caller-owned, or
/// a fresh [`WorkflowSession`]), and the engine runs with the run's cancellation
/// token, timeout, and stream modes. `WorkflowInvokeOutput` is fired on success
/// and `WorkflowError` on failure.
///
/// A caller-provided session is never committed, closed, or ended by the
/// runner; an owned session is closed after the run.
///
/// # Errors
///
/// `NotFound` for a missing id, the abort error of a `WorkflowInvokeInput`
/// handler, and whatever the workflow engine reports (compile, component,
/// cancellation, or deadline failures).
#[allow(clippy::needless_pass_by_value)]
#[allow(clippy::too_many_lines)] // one linear invocation; the observation seam is four lines of setup
pub async fn invoke_workflow(
    services: RunServices,
    workflow: WorkflowRef,
    inputs: Value,
    options: RunOptions,
) -> Result<WorkflowOutput> {
    services.require_local()?;
    let run_id = Uuid::new_v4().to_string();
    let started = Instant::now();

    let workflow = resolve_workflow(&services, &workflow)?;
    let card = workflow.card().clone();
    let (session, owned) = prepare_session(&services, &card.id, &options, &inputs);

    let context = CallbackContext::new(CallbackEvent::WorkflowInvokeInput)
        .with_component(card.id.clone())
        .with_input(inputs.clone())
        .with_session(Arc::clone(&session));
    services
        .callbacks
        .trigger(CallbackEvent::WorkflowInvokeInput, &context)
        .await?;

    // Observation seams: registered only now that every setup step that can fail
    // (resolution, session preparation, an aborting input handler) has succeeded.
    // A session the runner created is attached to the hub watcher; a caller-owned
    // one is not. The binding is declared before the status guard so the guard
    // finalises the run before the hub entry disappears.
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
        crate::monitor::RunStart::new(crate::monitor::RunKind::Workflow, card.id.clone())
            .with_session(session.id())
            .with_detail(detail.clone()),
    );
    status.running(detail.clone());

    trace_start("workflow", &run_id, &card.id, session.id());
    let span = RunSpan::start(session.tracer(), "run_workflow", inputs.clone());
    let teardown = workflow_teardown(owned);
    let run_options = run_options(&options);

    let result = workflow
        .invoke_with(inputs, Arc::clone(&session), run_options)
        .await;

    match result {
        Ok(output) => {
            let value = to_value(&output);
            span.finish(value.clone());
            let context = CallbackContext::new(CallbackEvent::WorkflowInvokeOutput)
                .with_component(card.id.clone())
                .with_output(value)
                .with_session(Arc::clone(&session));
            let callback = services
                .callbacks
                .trigger(CallbackEvent::WorkflowInvokeOutput, &context)
                .await;
            teardown.finish().await;
            trace_finish(
                "workflow",
                &run_id,
                &card.id,
                session.id(),
                started.elapsed(),
            );
            // An aborting output handler wins over the successful output, exactly
            // as before; the status mirrors whichever outcome the run reports.
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
            let context = CallbackContext::new(CallbackEvent::WorkflowError)
                .with_component(card.id.clone())
                .with_error(error.to_string())
                .with_session(Arc::clone(&session));
            // The run already failed: a failing error callback must not replace
            // the engine's error with a callback error.
            if let Err(callback_error) = services
                .callbacks
                .trigger(CallbackEvent::WorkflowError, &context)
                .await
            {
                tracing::warn!(
                    target: LOG_TARGET,
                    error = %callback_error,
                    "workflow error callback aborted; keeping the run error"
                );
            }
            teardown.finish().await;
            trace_failure(
                "workflow",
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

/// Runs one workflow invocation as a stream of [`WorkflowChunk`] frames.
///
/// A distributed-configured context refuses as this stream's **single** terminal
/// `Err` frame ([`RunServices::require_local`]): nothing else happens — no
/// reference resolution, no session, no callback, no monitor or hub status.
///
/// The stream is returned immediately, so a resolution failure is delivered as a
/// one-element stream yielding that error. On success the stream fires
/// `WorkflowStreamInput` first, forwards the engine's frames verbatim, and fires
/// `WorkflowStreamOutput` for every frame with a 0-based `index`. An owned
/// session is closed after the last frame and also when the consumer drops the
/// stream early.
///
/// Cancellation and the deadline are the engine's responsibility (passed through
/// [`WorkflowRunOptions`]), so frames are not re-wrapped or re-timed here.
#[allow(clippy::needless_pass_by_value)]
pub fn stream_workflow(
    services: RunServices,
    workflow: WorkflowRef,
    inputs: Value,
    options: RunOptions,
) -> BoxStream<'static, Result<WorkflowChunk>> {
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
        let workflow = match resolve_workflow(&services, &workflow) {
            Ok(workflow) => workflow,
            Err(error) => {
                yield Err(error);
                return;
            }
        };
        let card = workflow.card().clone();
        let (session, owned) = prepare_session(&services, &card.id, &options, &inputs);

        let context = CallbackContext::new(CallbackEvent::WorkflowStreamInput)
            .with_component(card.id.clone())
            .with_input(inputs.clone())
            .with_session(Arc::clone(&session));
        if let Err(error) = services
            .callbacks
            .trigger(CallbackEvent::WorkflowStreamInput, &context)
            .await
        {
            yield Err(error);
            return;
        }

        // An aborting `WorkflowStreamInput` handler is a setup failure, so the run
        // is admitted only after it resolved. Only a runner-created session is
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
            owned.as_ref().map(|_| Arc::clone(&session)),
        );
        let mut status = crate::monitor::RunStatusGuard::begin(
            services.monitor.clone(),
            &run_id,
            crate::monitor::RunStart::new(crate::monitor::RunKind::Workflow, card.id.clone())
                .with_session(session.id())
                .with_detail(detail.clone()),
        );
        status.running(detail.clone());

        trace_start("workflow", &run_id, &card.id, session.id());
        let span = RunSpan::start(session.tracer(), "run_workflow_stream", inputs.clone());
        let teardown = workflow_teardown(owned);
        let mut inner = workflow.stream_with(inputs, Arc::clone(&session), run_options(&options));
        let mut index: i64 = 0;
        let mut failure: Option<Error> = None;

        while let Some(frame) = inner.next().await {
            match frame {
                Ok(chunk) => {
                    let context = CallbackContext::new(CallbackEvent::WorkflowStreamOutput)
                        .with_component(card.id.clone())
                        .with_index(index)
                        .with_output(chunk.to_wire())
                        .with_session(Arc::clone(&session));
                    if let Err(error) = services
                        .callbacks
                        .trigger(CallbackEvent::WorkflowStreamOutput, &context)
                        .await
                    {
                        failure = Some(error);
                        break;
                    }
                    index += 1;
                    yield Ok(chunk);
                }
                Err(error) => {
                    failure = Some(error);
                    break;
                }
            }
        }

        record_stream_outcome(
            &span,
            "workflow",
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

/// Resolves the run's session: the caller's, or a fresh owned
/// [`WorkflowSession`].
fn prepare_session(
    services: &RunServices,
    workflow_id: &str,
    options: &RunOptions,
    inputs: &Value,
) -> (Arc<dyn Session>, Option<Arc<WorkflowSession>>) {
    if let Some(session) = &options.session {
        return (Arc::clone(session), None);
    }
    let owned = Arc::new(WorkflowSession::with_options(WorkflowSessionOptions {
        session_id: Some(resolve_session_id(inputs, options, &services.config)),
        workflow_id: Some(workflow_id.to_owned()),
        envs: options.envs.clone(),
        metadata: options.metadata.clone(),
        checkpointer: options
            .checkpointer
            .clone()
            .or_else(|| services.checkpointer.clone()),
        ..WorkflowSessionOptions::default()
    }));
    let session: Arc<dyn Session> = Arc::clone(&owned) as Arc<dyn Session>;
    (session, Some(owned))
}

/// Projects the run options onto the engine's [`WorkflowRunOptions`].
fn run_options(options: &RunOptions) -> WorkflowRunOptions {
    WorkflowRunOptions {
        cancellation: options.cancellation.clone(),
        timeout: options.timeout,
        stream_modes: options.stream_modes.clone(),
        ..WorkflowRunOptions::default()
    }
}

/// Builds the teardown of an owned workflow session (`close`, or nothing).
fn workflow_teardown(session: Option<Arc<WorkflowSession>>) -> Teardown {
    match session {
        None => Teardown::none(),
        Some(session) => Teardown::new(async move {
            if let Err(error) = session.close().await {
                tracing::warn!(
                    target: LOG_TARGET,
                    error = %error,
                    "workflow session close failed"
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
    use openjiuwen_workflow::{
        ComponentOptions, EndComponent, StartComponent, Workflow, WorkflowCard,
    };
    use serde_json::json;
    use std::sync::Mutex;

    use crate::callback::{CallbackFramework, CallbackVerdict};
    use crate::config::RunnerConfig;
    use crate::monitor::{RunEvent, RunPhase};
    use crate::resources::ResourceMgr;

    /// A services bundle with a recording handler for every workflow event.
    fn services(events: &Arc<Mutex<Vec<CallbackEvent>>>) -> RunServices {
        let services = RunServices::new(
            ResourceMgr::new(),
            CallbackFramework::new(),
            RunnerConfig::default(),
            None,
        );
        for event in [
            CallbackEvent::WorkflowInvokeInput,
            CallbackEvent::WorkflowInvokeOutput,
            CallbackEvent::WorkflowError,
            CallbackEvent::WorkflowStreamInput,
            CallbackEvent::WorkflowStreamOutput,
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

    /// start → end, both built-in components.
    fn linear_workflow() -> Workflow {
        let mut workflow = Workflow::new(WorkflowCard::new("wf"));
        workflow
            .set_start_comp(
                "start",
                StartComponent::new(),
                ComponentOptions::new().with_inputs_schema(json!({"query": "${inputs.query}"})),
            )
            .set_end_comp(
                "end",
                EndComponent::new(),
                ComponentOptions::new().with_inputs_schema(json!({"answer": "${start.query}"})),
            )
            .add_connection("start", "end");
        workflow
    }

    // ports tests/unit_tests/core/runner/test_runner.py (run_workflow)
    #[tokio::test]
    async fn invoke_fires_input_then_output_and_returns_the_result() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let services = services(&events);
        let output = invoke_workflow(
            services,
            WorkflowRef::of(Arc::new(linear_workflow())),
            json!({"query": "hi"}),
            RunOptions::default(),
        )
        .await
        .unwrap();
        assert_eq!(output.result, json!({"output": {"answer": "hi"}}));
        assert_eq!(
            *events.lock().unwrap(),
            vec![
                CallbackEvent::WorkflowInvokeInput,
                CallbackEvent::WorkflowInvokeOutput,
            ]
        );
    }

    #[tokio::test]
    async fn invoke_reports_an_unregistered_id_before_any_callback() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let services = services(&events);
        let error = invoke_workflow(
            services,
            WorkflowRef::id("missing"),
            json!({}),
            RunOptions::default(),
        )
        .await
        .unwrap_err();
        assert_eq!(
            error.to_string(),
            "not_found: workflow 'missing' is not registered"
        );
        assert!(events.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn a_failing_workflow_fires_the_error_event() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let services = services(&events);
        let error = invoke_workflow(
            services,
            WorkflowRef::of(Arc::new(Workflow::new(WorkflowCard::new("broken")))),
            json!({}),
            RunOptions::default(),
        )
        .await
        .unwrap_err();
        assert!(error.to_string().contains("no start component"));
        assert_eq!(
            *events.lock().unwrap(),
            vec![
                CallbackEvent::WorkflowInvokeInput,
                CallbackEvent::WorkflowError,
            ]
        );
    }

    #[tokio::test]
    async fn streaming_fires_input_and_one_output_event_per_frame() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let services = services(&events);
        let frames: Vec<Result<WorkflowChunk>> = stream_workflow(
            services,
            WorkflowRef::of(Arc::new(linear_workflow())),
            json!({"query": "hi"}),
            RunOptions::default(),
        )
        .collect()
        .await;
        assert!(!frames.is_empty());
        assert!(
            frames.iter().all(Result::is_ok),
            "unexpected error: {frames:?}"
        );
        let events = events.lock().unwrap().clone();
        assert_eq!(events.first(), Some(&CallbackEvent::WorkflowStreamInput));
        assert_eq!(
            events
                .iter()
                .filter(|event| **event == CallbackEvent::WorkflowStreamOutput)
                .count(),
            frames.len()
        );
    }

    #[tokio::test]
    async fn streaming_reports_a_missing_id_as_its_only_item() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let services = services(&events);
        let frames: Vec<Result<WorkflowChunk>> = stream_workflow(
            services,
            WorkflowRef::id("missing"),
            json!({}),
            RunOptions::default(),
        )
        .collect()
        .await;
        assert_eq!(frames.len(), 1);
        assert_eq!(
            frames[0].as_ref().unwrap_err().to_string(),
            "not_found: workflow 'missing' is not registered"
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

        invoke_workflow(
            services,
            WorkflowRef::of(Arc::new(linear_workflow())),
            json!({"query": "hi"}),
            RunOptions::default(),
        )
        .await
        .unwrap();

        assert!(hub.is_empty(), "the run was unregistered on completion");
        assert_eq!(
            phases(&mut transitions),
            vec![RunPhase::Queued, RunPhase::Running, RunPhase::Completed]
        );
        assert_eq!(monitor.list_active(), Vec::new(), "no run stays active");
    }

    #[tokio::test]
    async fn a_failing_run_is_mirrored_as_failed_and_unregistered() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let monitor = Arc::new(crate::monitor::RunMonitor::new());
        let hub = Arc::new(crate::hitl::InteractionHub::new());
        let mut transitions = monitor.subscribe();
        let services = services(&events)
            .with_monitor(Arc::clone(&monitor))
            .with_hitl(Arc::clone(&hub));

        let error = invoke_workflow(
            services,
            WorkflowRef::of(Arc::new(Workflow::new(WorkflowCard::new("broken")))),
            json!({}),
            RunOptions::default(),
        )
        .await
        .unwrap_err();

        assert!(error.to_string().contains("no start component"));
        assert!(hub.is_empty());
        assert_eq!(
            phases(&mut transitions),
            vec![RunPhase::Queued, RunPhase::Running, RunPhase::Failed]
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

        let frames: Vec<Result<WorkflowChunk>> = stream_workflow(
            services,
            WorkflowRef::of(Arc::new(linear_workflow())),
            json!({"query": "hi"}),
            RunOptions::default(),
        )
        .collect()
        .await;

        assert!(!frames.is_empty());
        assert!(hub.is_empty(), "the stream unregistered its run");
        assert_eq!(
            phases(&mut transitions),
            vec![RunPhase::Queued, RunPhase::Running, RunPhase::Completed]
        );
    }

    #[tokio::test]
    async fn a_workflow_stream_dropped_mid_flight_finishes_as_cancelled() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let monitor = Arc::new(crate::monitor::RunMonitor::new());
        let hub = Arc::new(crate::hitl::InteractionHub::new());
        let mut transitions = monitor.subscribe();
        let services = services(&events)
            .with_monitor(Arc::clone(&monitor))
            .with_hitl(Arc::clone(&hub));

        let mut stream = stream_workflow(
            services,
            WorkflowRef::of(Arc::new(linear_workflow())),
            json!({"query": "hi"}),
            RunOptions::default(),
        );
        // One poll yields the first frame and leaves the generator suspended, so
        // the run is still in flight when the stream goes away. The linear
        // workflow may emit a single frame in total; dropping after it exercises
        // the same unfinished-guard path.
        assert!(
            stream.next().await.is_some(),
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
}
