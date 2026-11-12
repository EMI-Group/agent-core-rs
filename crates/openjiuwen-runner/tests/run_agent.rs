//! Agent execution behaviour: reference resolution, session ownership, callback
//! ordering, cancellation, deadlines, and streaming.
//!
//! Drives only the public `openjiuwen_runner::run` surface (`invoke_agent`,
//! `stream_agent`, `AgentRef`, `RunOptions`, `RunServices`) exactly as the
//! `Runner` facade and the harness do.
//! Ports `tests/unit_tests/core/runner/test_runner.py` (`run_agent`,
//! `run_agent_streaming`, `_prepare_agent`, `_create_agent_session`).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::sync::Arc;
use std::time::Duration;

use futures::StreamExt;
use openjiuwen_agent::{Agent, AgentCard, ReActAgent, ResultType};
use openjiuwen_common::error::{Error, ErrorKind};
use openjiuwen_foundation::Model;
use openjiuwen_runner::callback::{CallbackContext, CallbackEvent, CallbackVerdict};
use openjiuwen_runner::run::{invoke_agent, stream_agent, AgentRef, RunOptions};
use openjiuwen_session::{Checkpointer, Session};
use openjiuwen_test_support::{MockCompletion, ScriptedModel};
use parking_lot::Mutex;
use serde_json::json;
use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;

use common::{
    agent_card, agent_events, answer_frames, checkpointed_agent_session, memory_checkpointer,
    record_all, recorded_events, recorder, scripted_agent, services, services_with_checkpointer,
    services_with_default_session, ScriptedAgent,
};

// ---------------------------------------------------------------------------
// Plain runs
// ---------------------------------------------------------------------------

// ports tests/unit_tests/core/runner/test_runner.py::test_run_agent
#[tokio::test]
async fn a_run_returns_the_scripted_answer() {
    let agent = scripted_agent("assistant", "hello from the script");
    let output = invoke_agent(
        services(),
        AgentRef::of(Arc::clone(&agent) as Arc<dyn Agent>),
        json!({"query": "hi"}),
        RunOptions::default(),
    )
    .await
    .unwrap();
    assert_eq!(output.output, "hello from the script");
    assert_eq!(output.result_type, ResultType::Answer);
    assert_eq!(agent.invocations(), 1);
}

// ports tests/unit_tests/core/runner/test_runner.py::test_run_agent_with_agent_id
#[tokio::test]
async fn an_id_reference_is_resolved_through_the_registry() {
    let lookups = Arc::new(Mutex::new(0_usize));
    let services = services();
    let served = Arc::clone(&lookups);
    services
        .resources
        .register_agent(
            agent_card("registered"),
            Arc::new(move || {
                *served.lock() += 1;
                Arc::new(ScriptedAgent::new("registered").with_answer("from the registry"))
                    as Arc<dyn Agent>
            }),
        )
        .unwrap();

    for _ in 0..2 {
        let output = invoke_agent(
            services.clone(),
            AgentRef::id("registered"),
            json!({"query": "hi"}),
            RunOptions::default(),
        )
        .await
        .unwrap();
        assert_eq!(output.output, "from the registry");
    }
    assert_eq!(
        *lookups.lock(),
        2,
        "the provider runs once per lookup (Python parity: a fresh instance per get)"
    );
}

/// The real `ReActAgent` path: the runner drives an agent that owns a model, so
/// session creation and callbacks are proven against the production agent rather
/// than only against the local `ScriptedAgent`.
// ports tests/unit_tests/core/runner/test_runner.py::test_run_agent (real agent)
#[tokio::test]
async fn a_real_react_agent_runs_through_the_runner() {
    let agent = ReActAgent::new(AgentCard::new("react-agent").with_id("react-agent"));
    let model = ScriptedModel::builder()
        .completion(MockCompletion::text("the real answer"))
        .build();
    agent.set_model(Arc::new(model) as Arc<dyn Model>);

    let output = invoke_agent(
        services(),
        AgentRef::from(Arc::new(agent)),
        json!("hi"),
        RunOptions::default(),
    )
    .await
    .unwrap();
    assert_eq!(output.output, "the real answer");
    assert_eq!(output.result_type, ResultType::Answer);
}

// ports tests/unit_tests/core/runner/test_runner.py::test_run_agent_not_found
#[tokio::test]
async fn an_unregistered_id_fails_before_any_callback_or_session() {
    let checkpointer = common::RecordingCheckpointer::new();
    let services = services_with_checkpointer(Arc::clone(&checkpointer) as Arc<dyn Checkpointer>);
    let log = recorder();
    record_all(&services.callbacks, &agent_events(), &log);

    let error = invoke_agent(
        services,
        AgentRef::id("missing"),
        json!({"conversation_id": "conv-missing"}),
        RunOptions::default(),
    )
    .await
    .unwrap_err();

    assert_eq!(error.kind(), ErrorKind::NotFound);
    assert_eq!(
        error.to_string(),
        "not_found: agent 'missing' is not registered"
    );
    assert!(
        recorded_events(&log).is_empty(),
        "a resolution failure fires no callback: {:?}",
        recorded_events(&log)
    );
    assert!(
        checkpointer.calls().is_empty(),
        "no session is created for an unregistered id: {:?}",
        checkpointer.calls()
    );
}

// ---------------------------------------------------------------------------
// Session ownership
// ---------------------------------------------------------------------------

// ports tests/unit_tests/core/runner/test_runner.py::test_run_agent_with_session
#[tokio::test]
async fn a_caller_session_is_used_and_never_committed() {
    let checkpointer = memory_checkpointer();
    let caller = checkpointed_agent_session(
        "caller-session",
        Arc::clone(&checkpointer) as Arc<dyn Checkpointer>,
    );
    let agent = scripted_agent("assistant", "ok");
    let services = services();
    let log = recorder();
    record_all(&services.callbacks, &agent_events(), &log);

    let output = invoke_agent(
        services,
        AgentRef::of(Arc::clone(&agent) as Arc<dyn Agent>),
        json!({"conversation_id": "ignored-for-a-caller-session"}),
        RunOptions::default().with_session(Arc::clone(&caller) as Arc<dyn Session>),
    )
    .await
    .unwrap();

    assert_eq!(output.output, "ok");
    assert_eq!(
        agent.session_ids(),
        vec!["caller-session".to_owned()],
        "the caller's session is the one the agent runs in"
    );
    assert_eq!(
        checkpointer.len("caller-session"),
        0,
        "a caller-provided session is never committed by the runner"
    );
    assert!(
        !recorded_events(&log).contains(&CallbackEvent::AgentSessionCreated),
        "no session is created when the caller provides one: {:?}",
        recorded_events(&log)
    );
}

// ports tests/unit_tests/core/runner/test_runner.py::test_create_agent_session
#[tokio::test]
async fn an_owned_session_takes_the_conversation_id_and_is_committed() {
    let checkpointer = memory_checkpointer();
    let services = services_with_checkpointer(Arc::clone(&checkpointer) as Arc<dyn Checkpointer>);
    let log = recorder();
    record_all(&services.callbacks, &agent_events(), &log);
    let agent = scripted_agent("assistant", "ok");

    invoke_agent(
        services,
        AgentRef::of(Arc::clone(&agent) as Arc<dyn Agent>),
        json!({"conversation_id": "conv-9"}),
        RunOptions::default(),
    )
    .await
    .unwrap();

    assert_eq!(
        agent.session_ids(),
        vec!["conv-9".to_owned()],
        "the owned session is named from `inputs.conversation_id`"
    );
    let created = log
        .lock()
        .iter()
        .filter(|record| record.event == CallbackEvent::AgentSessionCreated)
        .filter_map(|record| record.session_id.clone())
        .collect::<Vec<_>>();
    assert_eq!(created, vec!["conv-9".to_owned()]);
    assert_eq!(
        checkpointer.len("conv-9"),
        1,
        "an owned session is committed by `post_run`"
    );
}

#[tokio::test]
async fn an_owned_session_falls_back_to_the_run_then_the_config_default() {
    let log = recorder();
    let services = services_with_default_session("runner-default");
    record_all(&services.callbacks, &agent_events(), &log);
    let agent = scripted_agent("assistant", "ok");

    invoke_agent(
        services.clone(),
        AgentRef::of(Arc::clone(&agent) as Arc<dyn Agent>),
        json!({"conversation_id": ""}),
        RunOptions::default().with_session_id("caller-session-id"),
    )
    .await
    .unwrap();
    invoke_agent(
        services,
        AgentRef::of(Arc::clone(&agent) as Arc<dyn Agent>),
        json!({}),
        RunOptions::default(),
    )
    .await
    .unwrap();

    let expected = vec!["caller-session-id".to_owned(), "runner-default".to_owned()];
    assert_eq!(
        agent.session_ids(),
        expected,
        "an empty conversation id falls through to the run option, then the config"
    );
    let created = log
        .lock()
        .iter()
        .filter(|record| record.event == CallbackEvent::AgentSessionCreated)
        .filter_map(|record| record.session_id.clone())
        .collect::<Vec<_>>();
    assert_eq!(created, expected);
}

// ---------------------------------------------------------------------------
// Callbacks
// ---------------------------------------------------------------------------

// ports tests/unit_tests/core/runner/callback/test_framework_triggers.py
#[tokio::test]
async fn callbacks_run_by_priority_then_fire_input_before_output() {
    let order: Arc<Mutex<Vec<&'static str>>> = Arc::new(Mutex::new(Vec::new()));
    let services = services();
    let log = recorder();
    record_all(&services.callbacks, &agent_events(), &log);
    for (name, priority) in [("low", 0), ("high", 20)] {
        let order = Arc::clone(&order);
        services.callbacks.register_sync(
            CallbackEvent::AgentInvokeInput,
            priority,
            move |_: &CallbackContext| {
                order.lock().push(name);
                Ok(CallbackVerdict::Continue)
            },
        );
    }

    invoke_agent(
        services,
        AgentRef::of(Arc::new(ScriptedAgent::new("assistant").with_answer("ok")) as Arc<dyn Agent>),
        json!({"conversation_id": "conv-order"}),
        RunOptions::default(),
    )
    .await
    .unwrap();

    assert_eq!(
        *order.lock(),
        vec!["high", "low"],
        "handlers run by descending priority"
    );
    assert_eq!(
        recorded_events(&log),
        vec![
            CallbackEvent::AgentInvokeInput,
            CallbackEvent::AgentSessionCreated,
            CallbackEvent::AgentInvokeOutput,
        ],
        "the session is created between the input and the output event"
    );
    let entries = log.lock().clone();
    assert_eq!(
        entries.last().map(|entry| entry.component.clone()),
        Some("assistant".to_owned()),
        "the output event names the agent card id"
    );
}

// ports tests/unit_tests/core/runner/callback/test_framework_triggers.py (Abort)
#[tokio::test]
async fn an_aborting_input_handler_stops_the_run() {
    let services = services();
    let log = recorder();
    record_all(&services.callbacks, &agent_events(), &log);
    services.callbacks.register_sync(
        CallbackEvent::AgentInvokeInput,
        100,
        |_: &CallbackContext| {
            Ok(CallbackVerdict::Abort(Error::validation(
                "blocked by a handler",
            )))
        },
    );
    let agent = scripted_agent("assistant", "never runs");

    let error = invoke_agent(
        services,
        AgentRef::of(Arc::clone(&agent) as Arc<dyn Agent>),
        json!({"conversation_id": "conv-abort"}),
        RunOptions::default(),
    )
    .await
    .unwrap_err();

    assert_eq!(error.kind(), ErrorKind::Validation);
    assert_eq!(error.to_string(), "validation: blocked by a handler");
    assert_eq!(agent.invocations(), 0, "the agent never ran");
    assert!(
        !recorded_events(&log).contains(&CallbackEvent::AgentSessionCreated),
        "the abort happens before a session is created: {:?}",
        recorded_events(&log)
    );
}

// ---------------------------------------------------------------------------
// Cancellation and deadlines
// ---------------------------------------------------------------------------

// ports tests/unit_tests/core/runner/test_runner.py::test_run_agent_cancelled
#[tokio::test]
async fn a_pre_cancelled_run_reports_cancellation() {
    let token = CancellationToken::new();
    token.cancel();
    let error = invoke_agent(
        services(),
        AgentRef::of(scripted_agent("assistant", "ok") as Arc<dyn Agent>),
        json!({}),
        RunOptions::default().with_cancellation(token),
    )
    .await
    .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Cancelled);
    assert_eq!(error.to_string(), "cancelled");
}

#[tokio::test]
async fn a_cancelled_mid_run_reports_cancellation() {
    let started = Arc::new(Notify::new());
    let gate = Arc::new(Notify::new());
    let agent = Arc::new(
        ScriptedAgent::new("slow")
            .notifying_start(Arc::clone(&started))
            .blocking_on(Arc::clone(&gate)),
    );
    let token = CancellationToken::new();
    let running = tokio::spawn(invoke_agent(
        services(),
        AgentRef::of(Arc::clone(&agent) as Arc<dyn Agent>),
        json!({}),
        RunOptions::default().with_cancellation(token.clone()),
    ));

    started.notified().await;
    assert_eq!(agent.invocations(), 1, "the agent is already running");
    token.cancel();
    let error = running.await.unwrap().unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Cancelled);
}

// ports tests/unit_tests/core/runner/test_runner.py::test_run_agent_timeout
#[tokio::test]
async fn a_run_past_its_deadline_reports_a_timeout() {
    let agent = Arc::new(ScriptedAgent::new("slow").with_delay(Duration::from_secs(60)));
    let error = invoke_agent(
        services(),
        AgentRef::of(agent as Arc<dyn Agent>),
        json!({}),
        RunOptions::default().with_timeout(Duration::from_millis(10)),
    )
    .await
    .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Timeout);
    assert_eq!(error.to_string(), "timeout: run_agent timed out after 10ms");
}

// ---------------------------------------------------------------------------
// Streaming
// ---------------------------------------------------------------------------

// ports tests/unit_tests/core/runner/test_runner.py::test_run_agent_streaming
#[tokio::test]
async fn streaming_preserves_the_frame_order() {
    let agent = Arc::new(ScriptedAgent::new("assistant").with_chunks(answer_frames()));
    let frames = stream_agent(
        services(),
        AgentRef::of(Arc::clone(&agent) as Arc<dyn Agent>),
        json!({"query": "hi"}),
        RunOptions::default(),
    )
    .collect::<Vec<_>>()
    .await;

    let frames = frames
        .into_iter()
        .map(|frame| frame.unwrap())
        .collect::<Vec<_>>();
    assert_eq!(
        frames
            .iter()
            .map(|frame| frame.kind.as_str())
            .collect::<Vec<_>>(),
        vec!["llm_output", "llm_output", "answer"]
    );
    assert_eq!(
        frames.iter().map(|frame| frame.index).collect::<Vec<_>>(),
        vec![0, 1, 2]
    );
    assert_eq!(frames[0].payload["content"], json!("one"));
    assert_eq!(frames[2].payload["output"], json!("one two"));
    assert_eq!(agent.streams(), 1);
}

#[tokio::test]
async fn streaming_fires_one_output_event_per_frame_with_a_zero_based_index() {
    let services = services();
    let log = recorder();
    record_all(&services.callbacks, &agent_events(), &log);
    let agent = Arc::new(ScriptedAgent::new("assistant").with_chunks(answer_frames()));

    let frames = stream_agent(
        services,
        AgentRef::of(Arc::clone(&agent) as Arc<dyn Agent>),
        json!({"conversation_id": "conv-stream"}),
        RunOptions::default(),
    )
    .collect::<Vec<_>>()
    .await;
    assert_eq!(frames.len(), 3);

    let entries = log.lock().clone();
    let stream_outputs = entries
        .iter()
        .filter(|entry| entry.event == CallbackEvent::AgentStreamOutput)
        .collect::<Vec<_>>();
    assert_eq!(stream_outputs.len(), 3);
    assert_eq!(
        stream_outputs
            .iter()
            .map(|entry| entry.index)
            .collect::<Vec<_>>(),
        vec![Some(0), Some(1), Some(2)]
    );
    assert!(
        stream_outputs
            .iter()
            .all(|entry| entry.component == "assistant"),
        "every frame event names the agent card id"
    );
    assert_eq!(
        recorded_events(&log),
        vec![
            CallbackEvent::AgentSessionCreated,
            CallbackEvent::AgentStreamInput,
            CallbackEvent::AgentStreamOutput,
            CallbackEvent::AgentStreamOutput,
            CallbackEvent::AgentStreamOutput,
        ],
        "`AgentStreamInput` precedes exactly one output event per frame"
    );
}

// ports tests/unit_tests/core/runner/test_runner.py::test_run_agent_streaming_cancelled
#[tokio::test]
async fn streaming_a_cancelled_run_yields_exactly_one_terminal_error() {
    let gate = Arc::new(Notify::new());
    let agent = Arc::new(
        ScriptedAgent::new("assistant")
            .with_chunks(answer_frames())
            .gating_stream_on(Arc::clone(&gate)),
    );
    let token = CancellationToken::new();
    let mut stream = stream_agent(
        services(),
        AgentRef::of(Arc::clone(&agent) as Arc<dyn Agent>),
        json!({}),
        RunOptions::default().with_cancellation(token.clone()),
    );

    let first = stream.next().await.unwrap().unwrap();
    assert_eq!(first.index, 0);
    token.cancel();

    let terminal = stream.next().await.unwrap();
    assert_eq!(terminal.unwrap_err().kind(), ErrorKind::Cancelled);
    assert!(
        stream.next().await.is_none(),
        "the stream ends after its single terminal error"
    );
}

#[tokio::test]
async fn streaming_commits_an_owned_session() {
    let checkpointer = memory_checkpointer();
    let services = services_with_checkpointer(Arc::clone(&checkpointer) as Arc<dyn Checkpointer>);
    let agent = Arc::new(ScriptedAgent::new("assistant").with_chunks(answer_frames()));

    let frames = stream_agent(
        services,
        AgentRef::of(Arc::clone(&agent) as Arc<dyn Agent>),
        json!({"conversation_id": "conv-stream-commit"}),
        RunOptions::default(),
    )
    .collect::<Vec<_>>()
    .await;

    assert_eq!(frames.len(), 3);
    assert_eq!(
        checkpointer.len("conv-stream-commit"),
        1,
        "an owned session is committed after the last frame"
    );
}

#[tokio::test]
async fn streaming_reports_an_unregistered_id_as_its_only_item() {
    let frames = stream_agent(
        services(),
        AgentRef::id("missing"),
        json!({}),
        RunOptions::default(),
    )
    .collect::<Vec<_>>()
    .await;

    assert_eq!(frames.len(), 1);
    let error = frames[0].as_ref().unwrap_err();
    assert_eq!(error.kind(), ErrorKind::NotFound);
    assert_eq!(
        error.to_string(),
        "not_found: agent 'missing' is not registered"
    );
}
