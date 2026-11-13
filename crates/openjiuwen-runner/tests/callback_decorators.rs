//! Callback decorators and combinators: the Rust equivalents of Python's
//! callback decorators (`openjiuwen/core/runner/callback/decorator.py`): the
//! [`on`] builder, `emit_before`/`emit_after`/`emit_around`, `transform_io`
//! (as an `EventFilter` and as a `CallbackHandler`), and the `wrap` layer stack
//! with its `WrapRegistry`. Offline and deterministic: no sleeps, no network.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::too_many_lines,
    clippy::unnecessary_literal_bound
)]
use openjiuwen_common::error::{Error, ErrorKind};
use openjiuwen_runner::callback::{
    emit_after, emit_around, emit_before, input_as, on, on_wrap, output_as, payload_call,
    require_input, set_input, set_output, transform_io, wrap, wrap_handler, CallbackChain,
    CallbackContext, CallbackEvent, CallbackFramework, CallbackHandler, CallbackRegistration,
    CallbackResult, CallbackVerdict, NameFilter, NextCall, PredicateFilter, WrapHandler,
    WrapRegistry, WrappedCall,
};
use parking_lot::Mutex;
use serde_json::{json, Value};
use std::collections::BTreeSet;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
/// A shared, ordered record of what the combinators did.
type Log = Arc<Mutex<Vec<String>>>;
fn log() -> Log {
    Arc::new(Mutex::new(Vec::new()))
}
/// The recorded entries as one `|`-joined string.
fn joined(log: &Log) -> String {
    log.lock().join("|")
}
/// Renders an optional payload compactly (`none` when absent).
fn payload(value: Option<&Value>) -> String {
    value.map_or_else(|| "none".to_owned(), ToString::to_string)
}
/// Registers a handler at `priority` that appends `render(ctx)` to `log`.
fn record_with<F>(framework: &CallbackFramework, event: CallbackEvent, log: &Log, render: F)
where
    F: Fn(&CallbackContext) -> String + Send + Sync + 'static,
{
    let log = Arc::clone(log);
    framework.register_sync(event, 0, move |ctx: &CallbackContext| {
        log.lock().push(render(ctx));
        Ok(CallbackVerdict::Continue)
    });
}
/// Records `label:<input>`.
fn record_input(framework: &CallbackFramework, event: CallbackEvent, log: &Log, label: &str) {
    let label = label.to_owned();
    record_with(framework, event, log, move |ctx| {
        format!("{label}:{}", payload(ctx.input.as_ref()))
    });
}
/// Records `label:<output>`.
fn record_output(framework: &CallbackFramework, event: CallbackEvent, log: &Log, label: &str) {
    let label = label.to_owned();
    record_with(framework, event, log, move |ctx| {
        format!("{label}:{}", payload(ctx.output.as_ref()))
    });
}
/// A handler that increments `counter`.
fn counting(
    counter: &Arc<AtomicUsize>,
) -> impl Fn(&CallbackContext) -> CallbackResult + Send + Sync {
    let counter = Arc::clone(counter);
    move |_: &CallbackContext| {
        counter.fetch_add(1, Ordering::Relaxed);
        Ok(CallbackVerdict::Continue)
    }
}
/// A handler that fails with a validation error.
fn failing(message: &'static str) -> impl Fn(&CallbackContext) -> CallbackResult + Send + Sync {
    move |_: &CallbackContext| Err(Error::validation(message))
}
/// A wrap layer that records entering and leaving, forwarding its input.
fn traced_layer(name: &str, log: Log) -> Arc<dyn WrapHandler> {
    let label = name.to_owned();
    wrap_handler(name, move |ctx: CallbackContext, next: NextCall| {
        let log = Arc::clone(&log);
        let label = label.clone();
        Box::pin(async move {
            log.lock().push(format!("{label}:enter"));
            let value = next.call(ctx.input.clone().unwrap_or(Value::Null)).await?;
            log.lock().push(format!("{label}:exit"));
            Ok(value)
        })
    })
}
#[tokio::test]
async fn emit_before_fires_the_auxiliary_event_with_the_source_input() {
    let framework = CallbackFramework::new();
    let seen = log();
    record_with(&framework, CallbackEvent::LlmCallStarted, &seen, |ctx| {
        format!("aux:{}:{}", ctx.event.as_str(), payload(ctx.input.as_ref()))
    });
    record_input(&framework, CallbackEvent::LlmInvokeInput, &seen, "source");
    let emit = emit_before(Arc::clone(&framework), CallbackEvent::LlmCallStarted);
    framework.register(
        CallbackRegistration::new(CallbackEvent::LlmInvokeInput, Arc::new(emit)).with_priority(100),
    );
    let input = json!({"q": "rust"});
    let ctx = CallbackContext::new(CallbackEvent::LlmInvokeInput).with_input(input.clone());
    let report = framework
        .trigger(CallbackEvent::LlmInvokeInput, &ctx)
        .await
        .unwrap();
    assert!(report.is_clean());
    assert_eq!(report.fired, 2);
    let expected = format!(
        "aux:{}:{input}|source:{input}",
        CallbackEvent::LlmCallStarted.as_str()
    );
    assert_eq!(
        joined(&seen),
        expected,
        "the aux event fires first with the source payload"
    );
}
#[tokio::test]
async fn emit_after_sees_the_source_output_and_still_carries_the_input() {
    let framework = CallbackFramework::new();
    let seen = log();
    record_with(
        &framework,
        CallbackEvent::LlmResponseReceived,
        &seen,
        |ctx| {
            assert!(
                ctx.output.is_some(),
                "the source output reaches the aux event"
            );
            format!(
                "aux:input={}:output={}",
                payload(ctx.input.as_ref()),
                payload(ctx.output.as_ref())
            )
        },
    );
    record_output(&framework, CallbackEvent::LlmInvokeOutput, &seen, "source");
    let emit = emit_after(Arc::clone(&framework), CallbackEvent::LlmResponseReceived);
    framework.register(
        CallbackRegistration::new(CallbackEvent::LlmInvokeOutput, Arc::new(emit)).with_priority(50),
    );
    let prompt = json!({"prompt": "hi"});
    let answer = json!({"text": "hello"});
    let ctx = CallbackContext::new(CallbackEvent::LlmInvokeOutput)
        .with_input(prompt.clone())
        .with_output(answer.clone());
    let report = framework
        .trigger(CallbackEvent::LlmInvokeOutput, &ctx)
        .await
        .unwrap();
    assert!(report.is_clean());
    assert_eq!(report.fired, 2);
    assert_eq!(
        joined(&seen),
        format!("aux:input={prompt}:output={answer}|source:{answer}")
    );
}
#[tokio::test]
async fn emit_around_fires_before_then_after_with_the_matching_payloads() {
    let framework = CallbackFramework::new();
    let seen = log();
    record_input(&framework, CallbackEvent::LlmCallStarted, &seen, "before");
    record_output(
        &framework,
        CallbackEvent::LlmResponseReceived,
        &seen,
        "after",
    );
    let around = emit_around(
        Arc::clone(&framework),
        CallbackEvent::LlmCallStarted,
        CallbackEvent::LlmResponseReceived,
    );
    framework.register(
        CallbackRegistration::new(CallbackEvent::LlmInvokeInput, Arc::new(around)).with_priority(9),
    );
    let ctx = CallbackContext::new(CallbackEvent::LlmInvokeInput)
        .with_input(json!("question"))
        .with_output(json!("answer"));
    framework
        .trigger(CallbackEvent::LlmInvokeInput, &ctx)
        .await
        .unwrap();
    assert_eq!(
        joined(&seen),
        "before:\"question\"|after:\"answer\"",
        "the before event fires first, then the after event"
    );
}
#[tokio::test]
async fn an_auxiliary_abort_propagates_out_of_the_emit_handler() {
    let framework = CallbackFramework::new();
    framework.register_sync(CallbackEvent::ToolCallStarted, 0, |_: &CallbackContext| {
        Ok(CallbackVerdict::Abort(Error::security("blocked")))
    });
    let emit = emit_before(Arc::clone(&framework), CallbackEvent::ToolCallStarted);
    let ctx =
        CallbackContext::new(CallbackEvent::ToolInvokeInput).with_input(json!({"tool": "rm"}));
    match emit.handle(&ctx).await.unwrap() {
        CallbackVerdict::Abort(error) => {
            assert_eq!(error.kind(), ErrorKind::Security);
            assert_eq!(error.message(), "blocked", "no extra context is added");
            assert_eq!(error.to_string(), "security: blocked");
        }
        other => panic!("expected an abort, got {other:?}"),
    }
    let downstream = Arc::new(AtomicUsize::new(0));
    framework.register_sync(CallbackEvent::ToolInvokeInput, 0, counting(&downstream));
    framework.register(
        CallbackRegistration::new(CallbackEvent::ToolInvokeInput, Arc::new(emit))
            .with_priority(100),
    );
    let error = framework
        .trigger(CallbackEvent::ToolInvokeInput, &ctx)
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Security);
    assert_eq!(error.to_string(), "security: blocked");
    assert_eq!(downstream.load(Ordering::Relaxed), 0);
}
#[tokio::test]
async fn an_auxiliary_stop_maps_to_a_stop_verdict() {
    let framework = CallbackFramework::new();
    framework.register_sync(CallbackEvent::ToolCallFinished, 0, |_: &CallbackContext| {
        Ok(CallbackVerdict::Stop)
    });
    let emit = emit_before(Arc::clone(&framework), CallbackEvent::ToolCallFinished);
    let ctx = CallbackContext::new(CallbackEvent::ToolInvokeInput);
    assert!(matches!(
        emit.handle(&ctx).await.unwrap(),
        CallbackVerdict::Stop
    ));
    let downstream = Arc::new(AtomicUsize::new(0));
    framework.register_sync(CallbackEvent::ToolInvokeInput, 0, counting(&downstream));
    framework.register(
        CallbackRegistration::new(CallbackEvent::ToolInvokeInput, Arc::new(emit))
            .with_priority(100),
    );
    let report = framework
        .trigger(CallbackEvent::ToolInvokeInput, &ctx)
        .await
        .unwrap();
    assert!(report.aborted, "a stop ends dispatch successfully");
    assert_eq!(report.fired, 1);
    assert_eq!(downstream.load(Ordering::Relaxed), 0);
}
#[tokio::test]
async fn transform_io_apply_rewrites_both_sides_of_the_wrapped_call() {
    let transform = transform_io("shout")
        .with_input(|value| Ok(json!(value.as_str().unwrap_or_default().to_uppercase())))
        .with_output(|value| Ok(json!(format!("{value}!"))));
    assert_eq!(transform.name(), "shout");
    let seen = Arc::new(Mutex::new(None::<Value>));
    let call_seen = Arc::clone(&seen);
    let output = transform
        .apply(json!("hi"), move |input| {
            let call_seen = Arc::clone(&call_seen);
            async move {
                *call_seen.lock() = Some(input.clone());
                Ok(json!(format!(
                    "{}+call",
                    input.as_str().unwrap_or_default()
                )))
            }
        })
        .await
        .unwrap();
    assert_eq!(
        *seen.lock(),
        Some(json!("HI")),
        "the call sees the rewritten input"
    );
    assert_eq!(output, json!("\"HI+call\"!"));
    assert_eq!(transform.last_input(), Some(json!("HI")));
    assert_eq!(transform.last_output(), Some(json!("\"HI+call\"!")));
    let called = Arc::new(AtomicBool::new(false));
    let failing = transform_io("failing").with_input(|_| Err(Error::validation("bad input")));
    let flag = Arc::clone(&called);
    let error = failing
        .apply(json!("hi"), move |input| {
            let flag = Arc::clone(&flag);
            async move {
                flag.store(true, Ordering::Relaxed);
                Ok(input)
            }
        })
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert_eq!(error.message(), "bad input");
    assert!(
        !called.load(Ordering::Relaxed),
        "the wrapped call never ran"
    );
}
#[tokio::test]
async fn transform_io_rewrites_the_working_context_for_later_entries() {
    let event = CallbackEvent::ToolInvokeInput;
    let chain = CallbackChain::new("transform_filter");
    let rewriter = transform_io("upper")
        .with_input(|value| Ok(json!(value.as_str().unwrap_or_default().to_uppercase())));
    let entry = on(event.clone())
        .named("rewriter")
        .when(rewriter)
        .sync(|_: &CallbackContext| Ok(CallbackVerdict::Continue));
    entry.add_to(&chain);
    let observed = Arc::new(Mutex::new(None::<Value>));
    let seen = Arc::clone(&observed);
    on(event.clone())
        .named("observer")
        .sync(move |ctx: &CallbackContext| {
            *seen.lock() = Some(ctx.input.clone().unwrap_or(Value::Null));
            Ok(CallbackVerdict::Continue)
        })
        .add_to(&chain);
    chain
        .trigger(
            event.clone(),
            &CallbackContext::new(event.clone()).with_input(json!("hello")),
        )
        .await
        .unwrap();
    assert_eq!(
        *observed.lock(),
        Some(json!("HELLO")),
        "the later chain entry sees the filter's in-place rewrite"
    );
    // As a plain CallbackHandler it records its own rewrites.
    let recorder = transform_io("recorder")
        .with_input(|value| Ok(json!(value.as_str().unwrap_or_default().to_uppercase())))
        .with_output(|value| Ok(json!(format!("{value}?"))));
    let ctx = CallbackContext::new(event)
        .with_input(json!("hey"))
        .with_output(json!("yo"));
    assert!(matches!(
        recorder.handle(&ctx).await.unwrap(),
        CallbackVerdict::Continue
    ));
    assert_eq!(recorder.last_input(), Some(json!("HEY")));
    assert_eq!(recorder.last_output(), Some(json!("\"yo\"?")));
    assert_eq!(
        ctx.input,
        Some(json!("hey")),
        "the caller's context is intact"
    );
}
#[tokio::test]
async fn wrap_layers_run_outermost_first_and_rewrites_reach_later_layers() {
    let order = log();
    let inner_ran = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&inner_ran);
    let inner = payload_call(move |input: Value| {
        let flag = Arc::clone(&flag);
        async move {
            flag.store(true, Ordering::Relaxed);
            Ok(json!({"call": input}))
        }
    });
    let outer_log = Arc::clone(&order);
    let outer = wrap_handler("outer", move |ctx: CallbackContext, next: NextCall| {
        let log = Arc::clone(&outer_log);
        Box::pin(async move {
            log.lock().push("outer before".to_owned());
            let input = ctx.input.clone().unwrap_or(Value::Null);
            let rewritten = json!(format!("outer({})", input.as_str().unwrap_or_default()));
            let value = next.call(rewritten).await?;
            log.lock().push("outer after".to_owned());
            Ok(value)
        })
    });
    let inner_log = Arc::clone(&order);
    let inner_layer = wrap_handler("inner", move |ctx: CallbackContext, next: NextCall| {
        let log = Arc::clone(&inner_log);
        Box::pin(async move {
            let input = ctx.input.clone().unwrap_or(Value::Null);
            log.lock().push(format!(
                "inner before:{}",
                input.as_str().unwrap_or_default()
            ));
            let value = next.call(input).await?;
            log.lock().push("inner after".to_owned());
            Ok(value)
        })
    });
    let wrapped = wrap(CallbackEvent::ToolInvokeInput, inner)
        .with_handler(outer)
        .with_handler(inner_layer);
    let report = wrapped.call_with_report(json!("x")).await.unwrap();
    assert_eq!(report.value, json!({"call": "outer(x)"}));
    assert!(report.inner_ran);
    assert!(inner_ran.load(Ordering::Relaxed));
    assert_eq!(
        joined(&order),
        "outer before|inner before:outer(x)|inner after|outer after",
        "layers nest outermost-first and the outer rewrite reaches the inner layer"
    );
}
#[tokio::test]
async fn a_wrap_layer_that_never_calls_next_short_circuits_the_chain() {
    let inner_ran = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&inner_ran);
    let inner = payload_call(move |_: Value| {
        let flag = Arc::clone(&flag);
        async move {
            flag.store(true, Ordering::Relaxed);
            Ok(json!("inner"))
        }
    });
    let next_called = Arc::new(AtomicBool::new(true));
    let probe = Arc::clone(&next_called);
    let short = wrap_handler(
        "short_circuit",
        move |_ctx: CallbackContext, next: NextCall| {
            let probe = Arc::clone(&probe);
            Box::pin(async move {
                probe.store(next.called(), Ordering::Relaxed);
                Ok(json!("short"))
            })
        },
    );
    let wrapped = wrap(CallbackEvent::ToolInvokeInput, inner).with_handler(short);
    let report = wrapped.call_with_report(json!("x")).await.unwrap();
    assert_eq!(report.value, json!("short"));
    assert!(!report.inner_ran, "the wrapped call never ran");
    assert!(
        !next_called.load(Ordering::Relaxed),
        "call_next.called() is false"
    );
    assert!(!inner_ran.load(Ordering::Relaxed));
}
#[tokio::test]
async fn wrap_layer_and_wrapped_call_errors_propagate() {
    let failing_layer = wrap_handler("failing", |_ctx: CallbackContext, _next: NextCall| {
        Box::pin(async move { Err::<Value, _>(Error::validation("layer boom")) })
    });
    let wrapped = wrap(
        CallbackEvent::ToolInvokeInput,
        payload_call(|input: Value| async move { Ok(input) }),
    )
    .with_handler(failing_layer);
    let error = wrapped.call(json!("x")).await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert_eq!(error.message(), "layer boom");
    let passthrough = wrap_handler("passthrough", |_ctx: CallbackContext, next: NextCall| {
        Box::pin(async move { next.call(json!(null)).await })
    });
    let wrapped = wrap(
        CallbackEvent::ToolInvokeInput,
        payload_call(|_: Value| async move { Err::<Value, _>(Error::config("inner boom")) }),
    )
    .with_handler(passthrough);
    let error = wrapped.call(json!("x")).await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Config);
    assert_eq!(error.message(), "inner boom");
}
#[tokio::test]
async fn the_wrap_registry_resolves_layers_per_call_and_tracks_changes() {
    let registry = WrapRegistry::new();
    assert!(registry.is_empty());
    let event = CallbackEvent::AgentStarted;
    let order = log();
    let first =
        on_wrap(event.clone(), traced_layer("first", Arc::clone(&order))).register(&registry);
    assert_eq!(registry.len(), 1);
    assert_eq!(registry.events().as_slice(), std::slice::from_ref(&event));
    let call = WrappedCall::from_registry(
        Arc::clone(&registry),
        event.clone(),
        payload_call(|input: Value| async move { Ok(json!({"inner": input})) }),
    );
    assert_eq!(call.call(json!("x")).await.unwrap(), json!({"inner": "x"}));
    // A later-registered, higher-priority layer runs further out on the next call.
    let second = on_wrap(event.clone(), traced_layer("second", Arc::clone(&order)))
        .with_priority(10)
        .register(&registry);
    assert_eq!(registry.len(), 2);
    order.lock().clear();
    let report = call.call_with_report(json!("x")).await.unwrap();
    assert_eq!(report.value, json!({"inner": "x"}));
    assert!(report.inner_ran);
    assert_eq!(
        joined(&order),
        "second:enter|first:enter|first:exit|second:exit"
    );
    // Disabling a layer removes it from the next call.
    assert!(registry.set_enabled(second, false));
    order.lock().clear();
    call.call(json!("x")).await.unwrap();
    assert_eq!(joined(&order), "first:enter|first:exit");
    // Unregistering and per-event removal are reflected in the registry.
    assert!(registry.unregister(second));
    assert_eq!(registry.len(), 1);
    let other = CallbackEvent::WorkflowFinished;
    on_wrap(other.clone(), traced_layer("third", Arc::clone(&order))).register(&registry);
    assert!(registry.events().contains(&other));
    assert_eq!(registry.unregister_event(&other), 1);
    assert_eq!(registry.unregister_event(&other), 0);
    assert!(registry.unregister(first));
    assert!(registry.is_empty());
}
#[tokio::test]
async fn the_builder_reaches_the_registration_and_the_framework_info() {
    let event = CallbackEvent::ToolInvokeInput;
    let order = log();
    let recorder_order = Arc::clone(&order);
    let built = on(event.clone())
        .named("recorder")
        .with_priority(7)
        .in_namespace("audit")
        .with_tag("alpha")
        .with_tag("beta")
        .once()
        .sync(move |_: &CallbackContext| {
            recorder_order.lock().push("recorder".to_owned());
            Ok(CallbackVerdict::Continue)
        });
    assert_eq!(
        (built.name(), built.priority(), built.namespace()),
        ("recorder", 7, "audit")
    );
    assert!(!built.is_fatal());
    let tags = BTreeSet::from(["alpha".to_owned(), "beta".to_owned()]);
    let registration = built.registration();
    assert_eq!(registration.event, event);
    assert_eq!(registration.priority, 7);
    assert_eq!(registration.namespace, "audit");
    assert!(registration.once);
    assert_eq!(registration.tags, tags);
    let framework = CallbackFramework::new();
    let id = built.register(&framework);
    record_with(&framework, event.clone(), &order, |_| "later".to_owned());
    let info = framework.handlers(event.clone());
    assert_eq!(info.len(), 2);
    assert_eq!(info[0].id, id, "higher priority is listed first");
    assert_eq!(info[0].priority, 7);
    assert_eq!(info[0].namespace, "audit");
    assert!(info[0].once && info[0].enabled);
    assert_eq!(info[0].tags, tags);
    // Priority ordering, then the `once` handler disappears after one run.
    let ctx = CallbackContext::new(event.clone());
    let report = framework.trigger(event.clone(), &ctx).await.unwrap();
    assert!(report.is_clean() && report.fired == 2);
    assert_eq!(joined(&order), "recorder|later");
    framework.trigger(event.clone(), &ctx).await.unwrap();
    assert_eq!(joined(&order), "recorder|later|later");
    assert_eq!(framework.handlers(event.clone()).len(), 1);
    assert_eq!(framework.len(), 1);
    // A NameFilter that does not match the inner handler's name skips it.
    let skipped_runs = Arc::new(AtomicUsize::new(0));
    let skipped = on(event.clone())
        .named("skipped")
        .when(NameFilter::new().with_name("not_the_inner_name"))
        .sync(counting(&skipped_runs));
    assert!(matches!(
        skipped.handler().handle(&ctx).await.unwrap(),
        CallbackVerdict::Continue
    ));
    assert_eq!(skipped_runs.load(Ordering::Relaxed), 0);
    // A matching PredicateFilter lets the handler run; a mismatching one skips.
    let matched_runs = Arc::new(AtomicUsize::new(0));
    let matched = on(event.clone())
        .named("matched")
        .when(PredicateFilter::new(
            "has-input",
            |ctx: &CallbackContext| ctx.input.is_some(),
        ))
        .sync(counting(&matched_runs));
    matched
        .handler()
        .handle(&ctx.clone().with_input(json!(1)))
        .await
        .unwrap();
    assert_eq!(matched_runs.load(Ordering::Relaxed), 1);
    matched.handler().handle(&ctx).await.unwrap();
    assert_eq!(matched_runs.load(Ordering::Relaxed), 1);
    // An isolated error stays in the report; `fatal` surfaces it as the trigger error.
    let isolated = on(event.clone()).named("isolated").sync(failing("boom"));
    let framework = CallbackFramework::new();
    isolated.register(&framework);
    let report = framework.trigger(event.clone(), &ctx).await.unwrap();
    assert_eq!(
        (report.failures.len(), report.failures[0].handler.as_str()),
        (1, "isolated")
    );
    assert_eq!(report.failures[0].error.message(), "boom");
    assert!(!report.is_clean());
    let fatal = on(event.clone())
        .named("fatal")
        .fatal()
        .sync(failing("boom"));
    assert!(fatal.is_fatal());
    let framework = CallbackFramework::new();
    fatal.register(&framework);
    let error = framework
        .trigger(event, &ctx)
        .await
        .expect_err("a fatal handler error aborts the dispatch");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert_eq!(error.message(), "boom");
}
#[tokio::test]
async fn builder_handlers_join_an_existing_framework_and_a_chain() {
    let event = CallbackEvent::AgentInvokeOutput;
    let framework = CallbackFramework::new();
    let order = log();
    record_with(&framework, event.clone(), &order, |_| "builder".to_owned());
    record_with(&framework, event.clone(), &order, |_| "plain".to_owned());
    let report = framework
        .trigger(event.clone(), &CallbackContext::new(event.clone()))
        .await
        .unwrap();
    assert!(report.is_clean());
    assert_eq!(report.fired, 2);
    assert_eq!(joined(&order), "builder|plain");
    let chain = CallbackChain::new("audit");
    let built = on(event.clone()).named("chained").sync({
        let chain_order = Arc::clone(&order);
        move |_: &CallbackContext| {
            chain_order.lock().push("chained".to_owned());
            Ok(CallbackVerdict::Continue)
        }
    });
    let _entry = built.add_to(&chain);
    assert_eq!(chain.len(), 1);
    assert_eq!(chain.names(), ["chained"]);
    let report = chain
        .trigger(event.clone(), &CallbackContext::new(event.clone()))
        .await
        .unwrap();
    assert!(report.is_clean());
    assert_eq!(report.fired, 1);
    assert_eq!(joined(&order), "builder|plain|chained");
}
#[tokio::test]
async fn typed_payload_helpers_round_trip_preset_and_custom_events() {
    let mut ctx = CallbackContext::new(CallbackEvent::ToolInvokeInput);
    set_input(&mut ctx, &json!({"q": "rust"})).unwrap();
    set_output(&mut ctx, &vec![1, 2, 3]).unwrap();
    assert_eq!(input_as::<Value>(&ctx).unwrap(), Some(json!({"q": "rust"})));
    assert_eq!(output_as::<Vec<u8>>(&ctx).unwrap(), Some(vec![1, 2, 3]));
    assert_eq!(require_input::<Value>(&ctx).unwrap(), json!({"q": "rust"}));
    assert_eq!(
        input_as::<String>(&ctx).unwrap_err().kind(),
        ErrorKind::Validation
    );
    let empty = CallbackContext::new(CallbackEvent::Custom("host:thing".to_owned()));
    assert_eq!(input_as::<Value>(&empty).unwrap(), None);
    assert_eq!(output_as::<Value>(&empty).unwrap(), None);
    let error = require_input::<Value>(&empty).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert_eq!(
        error.to_string(),
        "validation: callback context carries no input payload"
    );
    // A custom event carries a payload through the same helpers.
    let mut custom = CallbackContext::new(CallbackEvent::Custom("host:thing".to_owned()));
    set_input(&mut custom, &42_u32).unwrap();
    set_output(&mut custom, &"done").unwrap();
    assert_eq!(custom.event.as_str(), "host:thing");
    assert_eq!(input_as::<u32>(&custom).unwrap(), Some(42));
    assert_eq!(
        output_as::<String>(&custom).unwrap(),
        Some("done".to_owned())
    );
    let seen = Arc::new(Mutex::new(None::<u32>));
    let captured = Arc::clone(&seen);
    let built = on(CallbackEvent::Custom("host:thing".to_owned()))
        .named("typed")
        .sync(move |ctx: &CallbackContext| {
            *captured.lock() = Some(require_input::<u32>(ctx)?);
            Ok(CallbackVerdict::Continue)
        });
    built.handler().handle(&custom).await.unwrap();
    assert_eq!(*seen.lock(), Some(42));
}
#[tokio::test]
async fn an_async_builder_sees_the_context_by_value() {
    let seen = Arc::new(Mutex::new(None::<String>));
    let captured = Arc::clone(&seen);
    let built = on(CallbackEvent::AgentInvokeInput)
        .named("async_recorder")
        .async_fn(move |ctx: CallbackContext| {
            let seen = Arc::clone(&captured);
            Box::pin(async move {
                *seen.lock() = Some(format!(
                    "{}:{}",
                    ctx.event.as_str(),
                    ctx.input.unwrap_or(Value::Null)
                ));
                Ok(CallbackVerdict::Continue)
            })
        });
    assert_eq!(built.name(), "async_recorder");
    let framework = CallbackFramework::new();
    built.register(&framework);
    let input = json!({"task": "go"});
    let ctx = CallbackContext::new(CallbackEvent::AgentInvokeInput).with_input(input.clone());
    let report = framework
        .trigger(CallbackEvent::AgentInvokeInput, &ctx)
        .await
        .unwrap();
    assert!(report.is_clean());
    assert_eq!(report.fired, 1);
    assert_eq!(
        *seen.lock(),
        Some(format!(
            "{}:{input}",
            CallbackEvent::AgentInvokeInput.as_str()
        ))
    );
}
