//! Callback chain behaviour: dispatch order, verdicts, filters, nesting, and
//! mutation semantics.
//!
//! Drives only the public `openjiuwen_runner::callback` surface. Ports the parts
//! of `tests/unit_tests/core/runner/callback/test_chain.py` the Rust chain keeps
//! (priority ordering, disabled entries, break/stop), and adds the filter,
//! nesting, and mutation cases the Rust [`CallbackChain`] documents.
//!
//! Everything here is in-process and deterministic: no sleeps, no network, and
//! no LLM. Shared handlers push their label into a `parking_lot`-free
//! `std::sync::Mutex` log so dispatch order is observable.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::sync::{Arc, Mutex};

use openjiuwen_common::error::{Error, ErrorKind};
use openjiuwen_runner::callback::{
    CallbackChain, CallbackContext, CallbackEvent, CallbackFramework, CallbackHandler,
    CallbackRegistration, CallbackVerdict, ChainEntry, EntryOutcome, NameFilter, ParamModifyFilter,
    PredicateFilter, SyncHandler,
};
use serde_json::{json, Value};

/// A [`serde::Serialize`] payload for `trigger_payload`.
#[derive(serde::Serialize)]
struct ToolCall {
    tool: &'static str,
    attempt: u32,
}

type Recorder = Arc<Mutex<Vec<String>>>;

fn recorder() -> Recorder {
    Arc::new(Mutex::new(Vec::new()))
}

fn recorded(log: &Recorder) -> Vec<String> {
    log.lock().unwrap().clone()
}

fn clear(log: &Recorder) {
    log.lock().unwrap().clear();
}

/// A handler that records its label and continues.
fn logging_handler(log: &Recorder, label: &'static str) -> Arc<dyn CallbackHandler> {
    let log = Arc::clone(log);
    Arc::new(SyncHandler::new(move |_: &CallbackContext| {
        log.lock().unwrap().push(label.to_owned());
        Ok(CallbackVerdict::Continue)
    }))
}

/// A handler that records its label and stops the chain.
fn stopping_handler(log: &Recorder, label: &'static str) -> Arc<dyn CallbackHandler> {
    let log = Arc::clone(log);
    Arc::new(SyncHandler::new(move |_: &CallbackContext| {
        log.lock().unwrap().push(label.to_owned());
        Ok(CallbackVerdict::Stop)
    }))
}

/// A handler that records its label and aborts the chain with `ErrorKind::Validation`.
fn aborting_handler(log: &Recorder, label: &'static str) -> Arc<dyn CallbackHandler> {
    let log = Arc::clone(log);
    Arc::new(SyncHandler::new(move |_: &CallbackContext| {
        log.lock().unwrap().push(label.to_owned());
        Ok(CallbackVerdict::Abort(Error::validation(
            "vetoed by handler",
        )))
    }))
}

/// A handler that records nothing and returns an isolated error.
fn failing_handler(message: &'static str) -> Arc<dyn CallbackHandler> {
    Arc::new(SyncHandler::new(move |_: &CallbackContext| {
        Err(Error::validation(message))
    }))
}

fn entry(log: &Recorder, name: &'static str, priority: i32) -> ChainEntry {
    ChainEntry::new(name, logging_handler(log, name)).with_priority(priority)
}

fn context(event: CallbackEvent) -> CallbackContext {
    CallbackContext::new(event)
}

#[tokio::test]
async fn entries_run_by_priority_descending_with_insertion_tiebreak() {
    // ports tests/unit_tests/core/runner/callback/test_chain.py::test_execute_respects_priority_order
    let log = recorder();
    let chain = CallbackChain::new("ordered");
    chain.add(entry(&log, "zero", 0));
    chain.add(entry(&log, "ten", 10));
    chain.add(entry(&log, "five-a", 5));
    chain.add(entry(&log, "five-b", 5));

    let event = CallbackEvent::AgentStarted;
    let report = chain.trigger(event.clone(), &context(event)).await.unwrap();

    assert_eq!(recorded(&log), ["ten", "five-a", "five-b", "zero"]);
    assert_eq!(report.fired, 4);
    assert!(report.is_clean());
    assert_eq!(chain.names(), ["ten", "five-a", "five-b", "zero"]);
}

#[tokio::test]
async fn insertion_order_breaks_ties_and_prepend_jumps_the_queue() {
    let log = recorder();
    let chain = CallbackChain::new("ties");
    chain.add(entry(&log, "first", 0));
    chain.add(entry(&log, "second", 0));
    chain.prepend(entry(&log, "front", 0));
    chain.add(entry(&log, "third", 0));

    assert_eq!(chain.names(), ["front", "first", "second", "third"]);

    let event = CallbackEvent::AgentStarted;
    let report = chain.trigger(event.clone(), &context(event)).await.unwrap();

    assert_eq!(recorded(&log), ["front", "first", "second", "third"]);
    assert_eq!(report.fired, 4);
    assert!(report.is_clean());
}

#[tokio::test]
async fn abort_short_circuits_and_names_the_chain_entry_and_kind() {
    let log = recorder();
    let chain = CallbackChain::new("outer");
    chain.add(entry(&log, "before", 10));
    chain.add(ChainEntry::new("abort", aborting_handler(&log, "abort")).with_priority(5));
    chain.add(entry(&log, "after", 0));

    let event = CallbackEvent::AgentInvokeInput;
    let error = chain
        .trigger(event.clone(), &context(event))
        .await
        .unwrap_err();

    assert_eq!(error.kind(), ErrorKind::Validation);
    assert_eq!(
        error.to_string(),
        "validation: callback chain 'outer' entry 'abort': vetoed by handler"
    );
    assert_eq!(recorded(&log), ["before", "abort"]);
}

#[tokio::test]
async fn stop_ends_the_chain_successfully_and_is_reported_as_stopped() {
    // ports tests/unit_tests/core/runner/callback/test_chain.py::test_break_action_stops_chain
    let log = recorder();
    let chain = CallbackChain::new("halting");
    chain.add(entry(&log, "before", 10));
    chain.add(ChainEntry::new("stop", stopping_handler(&log, "stop")).with_priority(5));
    chain.add(entry(&log, "after", 0));

    let event = CallbackEvent::AgentInvokeInput;
    let detailed = chain
        .trigger_detailed(event.clone(), &mut context(event))
        .await
        .unwrap();

    assert!(detailed.report.aborted);
    assert!(!detailed.report.is_clean());
    assert_eq!(detailed.report.fired, 2);
    assert_eq!(recorded(&log), ["before", "stop"]);
    assert_eq!(detailed.entries.len(), 2);
    assert_eq!(detailed.entries[1].name, "stop");
    assert_eq!(detailed.entries[1].outcome, EntryOutcome::Stopped);
    assert_eq!(detailed.ran().len(), 2);
}

#[tokio::test]
async fn isolated_handler_errors_are_collected_and_dispatch_continues() {
    let log = recorder();
    let chain = CallbackChain::new("isolated");
    chain.add(ChainEntry::new("fail-a", failing_handler("first failure")).with_priority(10));
    chain.add(ChainEntry::new("fail-b", failing_handler("second failure")).with_priority(5));
    chain.add(entry(&log, "ok", 0));

    let event = CallbackEvent::ToolInvokeInput;
    let detailed = chain
        .trigger_detailed(event.clone(), &mut context(event))
        .await
        .unwrap();

    assert_eq!(detailed.report.failures.len(), 2);
    assert_eq!(detailed.report.failures[0].handler, "fail-a");
    assert_eq!(
        detailed.report.failures[0].error.kind(),
        ErrorKind::Validation
    );
    assert!(detailed.report.failures[0]
        .error
        .to_string()
        .contains("first failure"));
    assert_eq!(detailed.report.failures[1].handler, "fail-b");
    assert!(detailed.report.failures[1]
        .error
        .to_string()
        .contains("second failure"));
    assert_eq!(detailed.report.fired, 1);
    assert!(!detailed.report.is_clean());
    assert!(!detailed.report.aborted);
    assert_eq!(recorded(&log), ["ok"]);
    assert_eq!(detailed.entries.len(), 3);
    assert_eq!(detailed.entries[0].outcome, EntryOutcome::Failed);
    assert_eq!(detailed.entries[1].outcome, EntryOutcome::Failed);
    assert_eq!(detailed.entries[2].outcome, EntryOutcome::Ran);
    assert_eq!(detailed.ran().len(), 3);
}

#[tokio::test]
async fn a_fatal_entry_error_aborts_the_chain_and_keeps_the_kind() {
    let log = recorder();
    let chain = CallbackChain::new("fatal-chain");
    chain.add(
        ChainEntry::new("explode", failing_handler("fatal failure"))
            .with_priority(10)
            .fatal(),
    );
    chain.add(entry(&log, "after", 0));

    let event = CallbackEvent::WorkflowError;
    let error = chain
        .trigger(event.clone(), &context(event))
        .await
        .unwrap_err();

    assert_eq!(error.kind(), ErrorKind::Validation);
    let message = error.to_string();
    assert!(message.contains("explode"), "{message}");
    assert!(message.contains("fatal failure"), "{message}");
    assert!(message.contains("fatal-chain"), "{message}");
    assert_eq!(recorded(&log), Vec::<String>::new());
}

#[tokio::test]
async fn filters_skip_entries_and_report_the_filter_and_reason() {
    let log = recorder();
    let chain = CallbackChain::new("filtered");
    chain.add(
        ChainEntry::new("guarded", logging_handler(&log, "guarded"))
            .with_priority(10)
            .when(NameFilter::new().with_name("someone_else")),
    );
    chain.add(entry(&log, "open", 0));

    let event = CallbackEvent::AgentStarted;
    let detailed = chain
        .trigger_detailed(event.clone(), &mut context(event))
        .await
        .unwrap();

    assert_eq!(recorded(&log), ["open"]);
    assert_eq!(detailed.report.fired, 1);
    assert!(detailed.report.is_clean());

    let skipped = detailed.skipped();
    assert_eq!(skipped.len(), 1);
    assert_eq!(skipped[0].name, "guarded");
    match &skipped[0].outcome {
        EntryOutcome::Skipped { filter, reason } => {
            assert_eq!(filter, "name");
            assert!(reason.contains("guarded"), "{reason}");
            assert!(reason.contains("someone_else"), "{reason}");
        }
        other => panic!("expected a skip, got {other:?}"),
    }
}

#[tokio::test]
async fn filter_mutation_is_visible_downstream_and_only_written_back_by_trigger_mut() {
    let log = recorder();
    let chain = CallbackChain::new("mutating");
    chain.add(
        ChainEntry::new("rewrite", logging_handler(&log, "rewrite"))
            .with_priority(10)
            .when(ParamModifyFilter::input(|value: &Value| {
                let query = value.get("q").and_then(Value::as_str).unwrap_or_default();
                Ok(json!({ "q": query.to_uppercase() }))
            })),
    );

    let seen: Arc<Mutex<Vec<Value>>> = Arc::new(Mutex::new(Vec::new()));
    let captured = Arc::clone(&seen);
    chain.add(ChainEntry::new(
        "read",
        Arc::new(SyncHandler::new(move |ctx: &CallbackContext| {
            captured
                .lock()
                .unwrap()
                .push(ctx.input.clone().unwrap_or(Value::Null));
            Ok(CallbackVerdict::Continue)
        })),
    ));

    let event = CallbackEvent::LlmInvokeInput;
    let original = context(event.clone()).with_input(json!({ "q": "rust" }));
    let report = chain.trigger(event.clone(), &original).await.unwrap();

    assert_eq!(report.fired, 2);
    assert_eq!(seen.lock().unwrap().clone(), vec![json!({ "q": "RUST" })]);
    assert_eq!(original.input, Some(json!({ "q": "rust" })));

    let mut mutable = context(event.clone()).with_input(json!({ "q": "rust" }));
    chain.trigger_mut(event, &mut mutable).await.unwrap();

    assert_eq!(mutable.input, Some(json!({ "q": "RUST" })));
    assert_eq!(seen.lock().unwrap().len(), 2);
}

#[tokio::test]
async fn nested_chains_run_their_entries_and_propagate_stop_and_abort() {
    let log = recorder();

    // A nested chain that continues lets the outer entry count as a normal run.
    let inner = Arc::new(CallbackChain::new("inner"));
    inner.add_named("inner-work", logging_handler(&log, "inner-work"));
    let outer = CallbackChain::new("outer");
    outer.add_handler(inner.clone());

    let event = CallbackEvent::AgentStarted;
    let report = outer
        .trigger(event.clone(), &context(event.clone()))
        .await
        .unwrap();
    assert_eq!(recorded(&log), ["inner-work"]);
    assert_eq!(report.fired, 1);
    assert!(report.is_clean());

    // A nested `Stop` ends the outer chain successfully, as a stopped entry.
    clear(&log);
    let halting = Arc::new(CallbackChain::new("nested-stop"));
    halting.add_named("halter", stopping_handler(&log, "halter"));
    let outer = CallbackChain::new("outer-stop");
    outer.add(ChainEntry::new("nested", halting).with_priority(10));
    outer.add(entry(&log, "after", 0));

    let detailed = outer
        .trigger_detailed(event.clone(), &mut context(event.clone()))
        .await
        .unwrap();
    assert!(detailed.report.aborted);
    assert_eq!(detailed.report.fired, 1);
    assert_eq!(recorded(&log), ["halter"]);
    assert_eq!(detailed.entries.len(), 1);
    assert_eq!(detailed.entries[0].outcome, EntryOutcome::Stopped);

    // A nested abort propagates out of the outer `trigger`.
    let aborting = Arc::new(CallbackChain::new("nested-abort"));
    aborting.add_named("bomb", aborting_handler(&log, "bomb"));
    let outer = CallbackChain::new("outer-abort");
    outer.add_handler(aborting.clone());

    let error = outer
        .trigger(event, &context(CallbackEvent::AgentStarted))
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    let message = error.to_string();
    assert!(message.contains("nested-abort"), "{message}");
    assert!(message.contains("bomb"), "{message}");
    assert!(message.contains("vetoed by handler"), "{message}");
}

#[tokio::test]
async fn disabled_entries_never_run_until_reenabled_by_name() {
    // ports tests/unit_tests/core/runner/callback/test_chain.py::test_execute_skips_disabled_callbacks
    let log = recorder();
    let chain = CallbackChain::new("toggled");
    chain.add(ChainEntry::new("muted", logging_handler(&log, "muted")).disabled());
    chain.add(entry(&log, "active", 0));

    let muted_enabled = |chain: &CallbackChain| {
        chain
            .entries()
            .iter()
            .find(|info| info.name == "muted")
            .map(|info| info.enabled)
    };

    let event = CallbackEvent::AgentStarted;
    let report = chain
        .trigger(event.clone(), &context(event.clone()))
        .await
        .unwrap();
    assert_eq!(report.fired, 1);
    assert_eq!(recorded(&log), ["active"]);
    assert_eq!(muted_enabled(&chain), Some(false));

    assert_eq!(chain.set_enabled("muted", true), 1);
    assert_eq!(chain.set_enabled("muted", true), 0);
    assert_eq!(muted_enabled(&chain), Some(true));

    clear(&log);
    let report = chain
        .trigger(event, &context(CallbackEvent::AgentStarted))
        .await
        .unwrap();
    assert_eq!(report.fired, 2);
    assert_eq!(recorded(&log), ["muted", "active"]);
}

#[tokio::test]
async fn removal_handles_duplicates_and_clear_leaves_a_clean_no_op() {
    let log = recorder();
    let chain = CallbackChain::new("removal");
    let first = chain.add_named("dup", logging_handler(&log, "dup-a"));
    chain.add_named("dup", logging_handler(&log, "dup-b"));
    chain.add_named("keeper", logging_handler(&log, "keeper"));

    assert_eq!(chain.len(), 3);
    assert_eq!(chain.names(), ["dup", "dup", "keeper"]);

    assert!(chain.remove_id(first));
    assert!(!chain.remove_id(first));
    assert_eq!(chain.names(), ["dup", "keeper"]);

    chain.add_named("dup", logging_handler(&log, "dup-c"));
    assert_eq!(chain.remove("dup"), 2);
    assert_eq!(chain.remove("dup"), 0);
    assert_eq!(chain.remove("missing"), 0);
    assert_eq!(chain.names(), ["keeper"]);

    chain.clear();
    assert!(chain.is_empty());
    assert_eq!(chain.len(), 0);
    assert_eq!(chain.names(), Vec::<String>::new());

    let event = CallbackEvent::AgentStarted;
    let report = chain.trigger(event.clone(), &context(event)).await.unwrap();
    assert_eq!(report.fired, 0);
    assert!(report.is_clean());
    assert!(!report.aborted);
    assert!(report.failures.is_empty());
}

#[tokio::test]
async fn a_chain_registered_on_the_framework_dispatches_and_surfaces_its_abort() {
    let log = recorder();
    let event = CallbackEvent::AgentInvokeOutput;

    let chain = Arc::new(CallbackChain::new("registered"));
    chain.add_named("chain-work", logging_handler(&log, "chain-work"));
    let framework = CallbackFramework::new();
    let handler: Arc<dyn CallbackHandler> = chain.clone();
    let registration =
        framework.register(CallbackRegistration::new(event.clone(), handler).with_priority(5));
    assert_eq!(registration.as_u64(), 0);

    let report = framework
        .trigger(event.clone(), &context(event.clone()))
        .await
        .unwrap();
    assert_eq!(report.fired, 1);
    assert_eq!(recorded(&log), ["chain-work"]);

    let aborting = Arc::new(CallbackChain::new("aborting"));
    aborting.add_named("bomb", aborting_handler(&log, "bomb"));
    let framework = CallbackFramework::new();
    let handler: Arc<dyn CallbackHandler> = aborting.clone();
    let _registration = framework.register(CallbackRegistration::new(event.clone(), handler));

    let error = framework
        .trigger(event, &context(CallbackEvent::AgentInvokeOutput))
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert_eq!(
        error.to_string(),
        "validation: callback chain 'aborting' entry 'bomb': vetoed by handler"
    );
}

#[tokio::test]
async fn trigger_payload_serializes_into_the_context_and_custom_events_work() {
    let log = recorder();
    let chain = CallbackChain::new("payload");
    let seen: Arc<Mutex<Option<Value>>> = Arc::new(Mutex::new(None));
    let captured = Arc::clone(&seen);
    chain.add(ChainEntry::new(
        "capture",
        Arc::new(SyncHandler::new(move |ctx: &CallbackContext| {
            *captured.lock().unwrap() = ctx.input.clone();
            Ok(CallbackVerdict::Continue)
        })),
    ));

    let payload = ToolCall {
        tool: "search",
        attempt: 2,
    };
    let report = chain
        .trigger_payload(CallbackEvent::ToolInvokeInput, &payload)
        .await
        .unwrap();
    assert_eq!(report.event, CallbackEvent::ToolInvokeInput);
    assert_eq!(report.fired, 1);
    assert_eq!(
        seen.lock().unwrap().clone(),
        Some(json!({ "tool": "search", "attempt": 2 }))
    );

    let custom = CallbackEvent::Custom("host:thing".to_owned());
    let report = chain
        .trigger_payload(custom.clone(), &payload)
        .await
        .unwrap();
    assert_eq!(report.event, custom);
    assert_eq!(report.fired, 1);
    assert_eq!(custom.as_str(), "host:thing");
    assert_eq!(custom.name(), "host:thing");
    assert_eq!(recorded(&log), Vec::<String>::new());
}

#[tokio::test]
async fn entry_metadata_reports_priority_namespace_filters_and_handle_shape() {
    let chain = CallbackChain::new("meta");
    let handler: Arc<dyn CallbackHandler> = Arc::new(
        SyncHandler::new(|_: &CallbackContext| Ok(CallbackVerdict::Continue)).with_name("reporter"),
    );
    let id = chain.add(
        ChainEntry::new("audit", handler)
            .with_priority(7)
            .in_namespace("custom")
            .when(NameFilter::new().with_name("audit"))
            .when(PredicateFilter::new("always", |_ctx: &CallbackContext| {
                true
            })),
    );

    assert_eq!(chain.names(), ["audit"]);
    let entries = chain.entries();
    assert_eq!(entries.len(), 1);
    let info = &entries[0];
    assert_eq!(info.id, id);
    assert_eq!(info.name, "audit");
    assert_eq!(info.namespace, "custom");
    assert_eq!(info.priority, 7);
    assert!(info.enabled);
    assert!(!info.fatal);
    assert_eq!(info.handler, "reporter");
    assert_eq!(info.filters, ["name", "always"]);

    assert_eq!(id.as_u64(), 0);
    assert_eq!(id.to_string(), "chain-entry-0");
    assert!(id.to_string().starts_with("chain-entry-"));
}
