//! Callback framework behaviour: vocabulary mapping, dispatch ordering, and
//! error isolation.
//!
//! Drives only the public `openjiuwen_runner::callback` surface.
//! Ports `tests/unit_tests/core/runner/callback/test_framework_triggers.py`
//! (ordering, `once`, enable/disable) and `test_events.py` (name mapping).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::future::BoxFuture;
use openjiuwen_common::error::{Error, ErrorKind};
use openjiuwen_runner::callback::{
    AsyncHandler, CallbackContext, CallbackEvent, CallbackFramework, CallbackHandler,
    CallbackRegistration, CallbackResult, CallbackVerdict, EventDomain, SyncHandler,
};
use serde_json::json;

fn context(event: CallbackEvent) -> CallbackContext {
    CallbackContext::new(event)
}

#[test]
fn every_preset_event_round_trips_through_its_wire_name() {
    // ports tests/unit_tests/core/runner/callback/test_events.py
    assert_eq!(CallbackEvent::ALL.len(), 67);
    for event in CallbackEvent::ALL {
        assert_eq!(event.as_str(), format!("_framework:{}", event.name()));
        assert_eq!(
            CallbackEvent::from_name(event.as_str()),
            Some(event.clone())
        );
        assert_eq!(CallbackEvent::from_name(event.name()), Some(event.clone()));
        assert!(CallbackEvent::from_name(event.as_str()).is_some());
        assert_ne!(event.domain(), EventDomain::Custom, "{event:?}");
    }
}

#[test]
fn wire_names_match_the_python_vocabulary() {
    let cases = [
        (CallbackEvent::AgentStarted, "_framework:agent_started"),
        (
            CallbackEvent::AgentInvokeInput,
            "_framework:agent_invoke_input",
        ),
        (
            CallbackEvent::AgentP2pReceived,
            "_framework:agent_p2p_received",
        ),
        (
            CallbackEvent::WorkflowNodeExecuted,
            "_framework:workflow_node_executed",
        ),
        (CallbackEvent::LlmRoute, "_framework:llm_route"),
        (CallbackEvent::ToolAuth, "_framework:tool_auth"),
        (
            CallbackEvent::ContextCompressionState,
            "_framework:context.compression_state",
        ),
        (
            CallbackEvent::AgentSessionCreated,
            "_framework:agent_session_created",
        ),
        (
            CallbackEvent::RetrievalStarted,
            "_framework:retrieval_started",
        ),
        (CallbackEvent::MemoryDeleted, "_framework:memory_deleted"),
        (CallbackEvent::TaskTimeout, "_framework:task_timeout"),
    ];
    for (event, name) in cases {
        assert_eq!(event.as_str(), name);
        assert_eq!(CallbackEvent::from_name(name), Some(event));
    }
    assert_eq!(CallbackEvent::from_name("_framework:not_real"), None);
    assert_eq!(CallbackEvent::from_name("not_real"), None);
}

#[test]
fn custom_events_are_representable_and_serializable() {
    let event = CallbackEvent::Custom("my_scope:my_event".to_owned());
    assert_eq!(event.name(), "my_scope:my_event");
    assert_eq!(event.as_str(), "my_scope:my_event");
    assert_eq!(event.domain(), EventDomain::Custom);

    let json = serde_json::to_value(CallbackEvent::LlmInvokeInput).unwrap();
    assert_eq!(json, json!("_framework:llm_invoke_input"));
    let decoded: CallbackEvent = serde_json::from_value(json).unwrap();
    assert_eq!(decoded, CallbackEvent::LlmInvokeInput);
    let custom: CallbackEvent = serde_json::from_value(json!("host:event")).unwrap();
    assert_eq!(custom, CallbackEvent::Custom("host:event".to_owned()));
}

#[tokio::test]
async fn handlers_run_by_descending_priority_with_registration_tiebreak() {
    let framework = CallbackFramework::new();
    let order = Arc::new(Mutex::new(Vec::new()));

    for (name, priority) in [
        ("low", 0),
        ("high", 20),
        ("mid_first", 10),
        ("mid_second", 10),
    ] {
        let order = Arc::clone(&order);
        let registration = CallbackRegistration::new(
            CallbackEvent::AgentInvokeInput,
            Arc::new(SyncHandler::new(move |_: &CallbackContext| {
                order.lock().unwrap().push(name);
                Ok(CallbackVerdict::Continue)
            })),
        )
        .with_priority(priority);
        framework.register(registration);
    }

    let report = framework
        .trigger(
            CallbackEvent::AgentInvokeInput,
            &context(CallbackEvent::AgentInvokeInput),
        )
        .await
        .unwrap();
    assert_eq!(report.fired, 4);
    assert!(report.is_clean());
    assert_eq!(
        *order.lock().unwrap(),
        ["high", "mid_first", "mid_second", "low"]
    );
}

#[tokio::test]
async fn sync_and_async_handlers_share_one_dispatch() {
    let framework = CallbackFramework::new();
    let seen = Arc::new(Mutex::new(Vec::new()));

    let sync_seen = Arc::clone(&seen);
    let sync_id = framework.register_sync(
        CallbackEvent::ToolCallStarted,
        0,
        move |ctx: &CallbackContext| {
            sync_seen.lock().unwrap().push(ctx.event.name().to_owned());
            Ok(CallbackVerdict::Continue)
        },
    );

    let async_seen = Arc::clone(&seen);
    let async_id = framework.register_async(CallbackEvent::ToolCallStarted, 0, move |ctx| {
        let async_seen = Arc::clone(&async_seen);
        Box::pin(async move {
            async_seen.lock().unwrap().push(ctx.event.name().to_owned());
            Ok(CallbackVerdict::Continue)
        }) as BoxFuture<'static, CallbackResult>
    });

    assert_ne!(sync_id, async_id);
    let report = framework
        .trigger(
            CallbackEvent::ToolCallStarted,
            &context(CallbackEvent::ToolCallStarted).with_input(json!({"name": "echo"})),
        )
        .await
        .unwrap();
    assert_eq!(report.fired, 2);
    assert_eq!(
        *seen.lock().unwrap(),
        ["tool_call_started", "tool_call_started"]
    );
}

struct RecordingHandler {
    label: &'static str,
    seen: Arc<Mutex<Vec<&'static str>>>,
}

#[async_trait]
impl CallbackHandler for RecordingHandler {
    fn name(&self) -> &str {
        self.label
    }

    async fn handle(&self, ctx: &CallbackContext) -> CallbackResult {
        self.seen.lock().unwrap().push(self.label);
        if ctx.output.is_some() {
            Ok(CallbackVerdict::Continue)
        } else {
            Ok(CallbackVerdict::Stop)
        }
    }
}

#[tokio::test]
async fn a_trait_object_handler_is_a_first_class_registration() {
    let framework = CallbackFramework::new();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let handler: Arc<dyn CallbackHandler> = Arc::new(RecordingHandler {
        label: "custom",
        seen: Arc::clone(&seen),
    });
    framework.register(CallbackRegistration::new(
        CallbackEvent::AgentInvokeOutput,
        handler,
    ));

    let report = framework
        .trigger(
            CallbackEvent::AgentInvokeOutput,
            &context(CallbackEvent::AgentInvokeOutput).with_output(json!("final")),
        )
        .await
        .unwrap();
    assert_eq!(report.fired, 1);
    assert!(!report.aborted);
    assert_eq!(*seen.lock().unwrap(), ["custom"]);
    assert_eq!(
        framework.handlers(CallbackEvent::AgentInvokeOutput)[0].name,
        "custom"
    );
}

#[tokio::test]
async fn a_handler_error_is_recorded_and_dispatch_continues() {
    let framework = CallbackFramework::new();
    let ran = Arc::new(Mutex::new(false));

    framework.register_sync(CallbackEvent::WorkflowStarted, 10, |_: &CallbackContext| {
        Err(Error::validation("handler blew up"))
    });
    let later = Arc::clone(&ran);
    framework.register_sync(
        CallbackEvent::WorkflowStarted,
        0,
        move |_: &CallbackContext| {
            *later.lock().unwrap() = true;
            Ok(CallbackVerdict::Continue)
        },
    );

    let report = framework
        .trigger(
            CallbackEvent::WorkflowStarted,
            &context(CallbackEvent::WorkflowStarted),
        )
        .await
        .unwrap();
    assert_eq!(report.fired, 1);
    assert!(!report.is_clean());
    assert_eq!(report.failures.len(), 1);
    assert_eq!(report.failures[0].handler, "sync_handler");
    assert_eq!(report.failures[0].error.kind(), ErrorKind::Validation);
    assert!(*ran.lock().unwrap());
}

#[tokio::test]
async fn abort_ends_dispatch_and_surfaces_the_error() {
    let framework = CallbackFramework::new();
    let later = Arc::new(Mutex::new(false));

    framework.register_sync(CallbackEvent::AgentInvokeInput, 0, |_: &CallbackContext| {
        Ok(CallbackVerdict::Abort(Error::Security(
            "blocked".to_owned(),
        )))
    });
    let later_clone = Arc::clone(&later);
    framework.register_sync(
        CallbackEvent::AgentInvokeInput,
        0,
        move |_: &CallbackContext| {
            *later_clone.lock().unwrap() = true;
            Ok(CallbackVerdict::Continue)
        },
    );

    let error = framework
        .trigger(
            CallbackEvent::AgentInvokeInput,
            &context(CallbackEvent::AgentInvokeInput),
        )
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Security);
    assert_eq!(error.to_string(), "security: blocked");
    assert!(!*later.lock().unwrap());
}

#[tokio::test]
async fn stop_ends_dispatch_successfully() {
    let framework = CallbackFramework::new();
    let later = Arc::new(Mutex::new(false));

    framework.register_sync(
        CallbackEvent::AgentInvokeInput,
        10,
        |_: &CallbackContext| Ok(CallbackVerdict::Stop),
    );
    let later_clone = Arc::clone(&later);
    framework.register_sync(
        CallbackEvent::AgentInvokeInput,
        0,
        move |_: &CallbackContext| {
            *later_clone.lock().unwrap() = true;
            Ok(CallbackVerdict::Continue)
        },
    );

    let report = framework
        .trigger(
            CallbackEvent::AgentInvokeInput,
            &context(CallbackEvent::AgentInvokeInput),
        )
        .await
        .unwrap();
    assert_eq!(report.fired, 1);
    assert!(report.aborted);
    assert!(report.failures.is_empty());
    assert!(!*later.lock().unwrap());
}

#[tokio::test]
async fn once_handlers_are_removed_after_their_first_successful_run() {
    let framework = CallbackFramework::new();
    let calls = Arc::new(Mutex::new(0));
    let counter = Arc::clone(&calls);
    framework.register(
        CallbackRegistration::new(
            CallbackEvent::SessionCreated,
            Arc::new(SyncHandler::new(move |_: &CallbackContext| {
                *counter.lock().unwrap() += 1;
                Ok(CallbackVerdict::Continue)
            })),
        )
        .once(),
    );

    framework
        .trigger(
            CallbackEvent::SessionCreated,
            &context(CallbackEvent::SessionCreated),
        )
        .await
        .unwrap();
    assert_eq!(framework.len(), 0);

    framework
        .trigger(
            CallbackEvent::SessionCreated,
            &context(CallbackEvent::SessionCreated),
        )
        .await
        .unwrap();
    assert_eq!(*calls.lock().unwrap(), 1);
}

#[tokio::test]
async fn a_failed_once_handler_is_not_consumed() {
    let framework = CallbackFramework::new();
    framework.register(
        CallbackRegistration::new(
            CallbackEvent::SessionCreated,
            Arc::new(SyncHandler::new(|_: &CallbackContext| {
                Err(Error::validation("still failing"))
            })),
        )
        .once(),
    );

    framework
        .trigger(
            CallbackEvent::SessionCreated,
            &context(CallbackEvent::SessionCreated),
        )
        .await
        .unwrap();
    assert_eq!(framework.len(), 1);
}

#[tokio::test]
async fn disable_and_reenable_a_handler() {
    let framework = CallbackFramework::new();
    let id = framework.register_sync(CallbackEvent::MemoryAdded, 0, |_: &CallbackContext| {
        Ok(CallbackVerdict::Continue)
    });

    assert!(framework.set_enabled(id, false));
    let report = framework
        .trigger(
            CallbackEvent::MemoryAdded,
            &context(CallbackEvent::MemoryAdded),
        )
        .await
        .unwrap();
    assert_eq!(report.fired, 0);
    assert!(!framework.handlers(CallbackEvent::MemoryAdded)[0].enabled);

    assert!(framework.set_enabled(id, true));
    let report = framework
        .trigger(
            CallbackEvent::MemoryAdded,
            &context(CallbackEvent::MemoryAdded),
        )
        .await
        .unwrap();
    assert_eq!(report.fired, 1);
}

#[tokio::test]
async fn unregister_by_id_event_namespace_and_tag() {
    let framework = CallbackFramework::new();
    let first = framework.register_sync(CallbackEvent::AgentStarted, 0, |_: &CallbackContext| {
        Ok(CallbackVerdict::Continue)
    });
    framework.register(
        CallbackRegistration::new(
            CallbackEvent::AgentStarted,
            Arc::new(SyncHandler::new(|_: &CallbackContext| {
                Ok(CallbackVerdict::Continue)
            })),
        )
        .with_namespace("observer"),
    );
    framework.register(
        CallbackRegistration::new(
            CallbackEvent::TaskCreated,
            Arc::new(SyncHandler::new(|_: &CallbackContext| {
                Ok(CallbackVerdict::Continue)
            })),
        )
        .with_tag("metrics"),
    );

    assert!(framework.unregister(first));
    assert!(!framework.unregister(first));
    assert_eq!(framework.unregister_namespace("observer"), 1);
    assert_eq!(framework.unregister_tag("metrics"), 1);
    assert!(framework.is_empty());

    framework.register_sync(CallbackEvent::TaskCreated, 0, |_: &CallbackContext| {
        Ok(CallbackVerdict::Continue)
    });
    assert_eq!(framework.unregister_event(CallbackEvent::TaskCreated), 1);
    assert_eq!(framework.unregister_event(CallbackEvent::TaskCreated), 0);
}

#[tokio::test]
async fn dispatch_is_isolated_per_event() {
    let framework = CallbackFramework::new();
    let ran = Arc::new(Mutex::new(false));
    let ran_clone = Arc::clone(&ran);
    framework.register_sync(
        CallbackEvent::ToolInvokeInput,
        0,
        move |_: &CallbackContext| {
            *ran_clone.lock().unwrap() = true;
            Ok(CallbackVerdict::Continue)
        },
    );

    assert!(!framework.has_subscribers(CallbackEvent::LlmInvokeInput));
    assert!(framework.has_subscribers(CallbackEvent::ToolInvokeInput));

    let report = framework
        .trigger(
            CallbackEvent::LlmInvokeInput,
            &context(CallbackEvent::LlmInvokeInput),
        )
        .await
        .unwrap();
    assert_eq!(report.fired, 0);
    assert!(!*ran.lock().unwrap());
}

#[tokio::test]
async fn subscribing_twice_dispatches_twice() {
    let framework = CallbackFramework::new();
    framework.register_sync(CallbackEvent::RetrievalStarted, 0, |_: &CallbackContext| {
        Ok(CallbackVerdict::Continue)
    });
    framework.register_sync(CallbackEvent::RetrievalStarted, 0, |_: &CallbackContext| {
        Ok(CallbackVerdict::Continue)
    });

    let report = framework
        .trigger(
            CallbackEvent::RetrievalStarted,
            &context(CallbackEvent::RetrievalStarted),
        )
        .await
        .unwrap();
    assert_eq!(report.fired, 2);
    assert_eq!(framework.handlers(CallbackEvent::RetrievalStarted).len(), 2);
}

#[tokio::test]
async fn no_subscribers_is_a_clean_no_op() {
    let framework = CallbackFramework::new();
    let report = framework
        .trigger(
            CallbackEvent::LlmStreamOutput,
            &context(CallbackEvent::LlmStreamOutput).with_index(7),
        )
        .await
        .unwrap();
    assert_eq!(report.fired, 0);
    assert!(!report.aborted);
    assert!(report.is_clean());
}

#[tokio::test]
async fn events_and_handlers_expose_registration_state() {
    let framework = CallbackFramework::new();
    assert_eq!(framework.events(), Vec::new());
    assert_eq!(framework.len(), 0);
    assert!(framework.is_empty());

    framework.register(
        CallbackRegistration::new(
            CallbackEvent::AgentStarted,
            Arc::new(
                SyncHandler::new(|_: &CallbackContext| Ok(CallbackVerdict::Continue))
                    .with_name("recorder"),
            ),
        )
        .with_priority(3)
        .with_namespace("audit")
        .with_tag("keep"),
    );
    framework.register_sync(CallbackEvent::ToolCallFinished, 0, |_: &CallbackContext| {
        Ok(CallbackVerdict::Continue)
    });

    assert_eq!(
        framework.events(),
        vec![CallbackEvent::AgentStarted, CallbackEvent::ToolCallFinished]
    );
    assert_eq!(framework.len(), 2);

    let info = &framework.handlers(CallbackEvent::AgentStarted)[0];
    assert_eq!(info.name, "recorder");
    assert_eq!(info.priority, 3);
    assert_eq!(info.namespace, "audit");
    assert!(info.tags.contains("keep"));

    let json = serde_json::to_value(info).unwrap();
    let decoded: openjiuwen_runner::callback::CallbackInfo = serde_json::from_value(json).unwrap();
    assert_eq!(&decoded, info);
}

#[tokio::test]
async fn an_async_handler_can_await_before_continuing() {
    let framework = CallbackFramework::new();
    let completed = Arc::new(Mutex::new(false));
    let flag = Arc::clone(&completed);
    framework.register_async(CallbackEvent::TaskCompleted, 0, move |_ctx| {
        let flag = Arc::clone(&flag);
        Box::pin(async move {
            tokio::task::yield_now().await;
            *flag.lock().unwrap() = true;
            Ok(CallbackVerdict::Continue)
        }) as BoxFuture<'static, CallbackResult>
    });

    framework
        .trigger(
            CallbackEvent::TaskCompleted,
            &context(CallbackEvent::TaskCompleted),
        )
        .await
        .unwrap();
    assert!(*completed.lock().unwrap());
}

#[tokio::test]
async fn the_context_reaches_a_handler_with_its_payload_and_metadata() {
    let framework = CallbackFramework::new();
    let seen = Arc::new(Mutex::new(None));
    let seen_clone = Arc::clone(&seen);
    framework.register_sync(
        CallbackEvent::LlmStreamOutput,
        0,
        move |ctx: &CallbackContext| {
            *seen_clone.lock().unwrap() = Some((
                ctx.component.clone(),
                ctx.index,
                ctx.input.clone(),
                ctx.metadata.clone(),
            ));
            Ok(CallbackVerdict::Continue)
        },
    );

    let ctx = context(CallbackEvent::LlmStreamOutput)
        .with_component("assistant")
        .with_input(json!("chunk"))
        .with_index(2)
        .with_metadata("model", json!("fake"));
    framework
        .trigger(CallbackEvent::LlmStreamOutput, &ctx)
        .await
        .unwrap();

    let seen = seen.lock().unwrap().clone().unwrap();
    assert_eq!(seen.0, "assistant");
    assert_eq!(seen.1, Some(2));
    assert_eq!(seen.2, Some(json!("chunk")));
    assert_eq!(seen.3["model"], json!("fake"));
}

#[tokio::test]
async fn handler_names_default_to_the_adapter_kind() {
    let sync = SyncHandler::new(|_: &CallbackContext| Ok(CallbackVerdict::Continue));
    assert_eq!(sync.name(), "sync_handler");
    let async_handler = AsyncHandler::new(|_: CallbackContext| {
        Box::pin(async move { Ok(CallbackVerdict::Continue) }) as BoxFuture<'static, CallbackResult>
    });
    assert_eq!(async_handler.name(), "async_handler");
}
