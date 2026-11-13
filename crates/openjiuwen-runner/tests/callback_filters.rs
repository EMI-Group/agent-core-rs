//! Filter behaviour: every built-in filter, the composition operators, the
//! filter run loop, and one end-to-end gate over a `CallbackChain`.
//!
//! Drives only the public `openjiuwen_runner::callback` surface and stays
//! offline and deterministic: time-dependent filters (`RateLimitFilter`,
//! `CircuitBreakerFilter`) are driven by a `ManualClock`, so nothing sleeps.
//! Ports `tests/unit_tests/core/runner/callback/test_filters.py`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use openjiuwen_common::error::{Error, ErrorKind};
use openjiuwen_runner::callback::{
    and, lenient, not, or, record_failure, record_success, run_filters, AuthFilter, CallbackChain,
    CallbackContext, CallbackEvent, CallbackHandler, CallbackVerdict, ChainEntry,
    CircuitBreakerFilter, ClosureFilter, ConditionalFilter, EntryOutcome, EventDomain, EventFilter,
    EventFilterExt, FilterChain, FilterDecision, FilterInput, FilterRunOutcome, KindFilter,
    LoggingFilter, ManualClock, NameFilter, ParamModifyFilter, PredicateFilter, RateLimitFilter,
    SessionFilter, SyncHandler, ValidationFilter,
};
use openjiuwen_session::{AgentSession, Session};
use serde_json::{json, Value};

/// The handler identity every single-filter check uses.
const HANDLER: &str = "handler";
/// The namespace every single-filter check uses.
const NAMESPACE: &str = "default";

/// An empty context for `event`.
fn context(event: CallbackEvent) -> CallbackContext {
    CallbackContext::new(event)
}

/// Runs `filter` against `ctx` for [`HANDLER`] in [`NAMESPACE`].
fn decide<F: EventFilter + ?Sized>(filter: &F, ctx: &mut CallbackContext) -> FilterDecision {
    decide_for(filter, HANDLER, NAMESPACE, ctx)
}

/// Runs `filter` against `ctx` for one handler identity.
fn decide_for<F: EventFilter + ?Sized>(
    filter: &F,
    handler: &str,
    namespace: &str,
    ctx: &mut CallbackContext,
) -> FilterDecision {
    filter
        .allows(FilterInput {
            handler,
            namespace,
            ctx,
        })
        .unwrap()
}

/// Erases `filter` into the shared handle `run_filters` stores.
fn shared(filter: impl EventFilter + 'static) -> Arc<dyn EventFilter> {
    Arc::new(filter)
}

fn expect_allow(decision: &FilterDecision) {
    assert!(decision.is_allow(), "expected an allow, got {decision:?}");
}

fn expect_skip(decision: FilterDecision) -> String {
    match decision {
        FilterDecision::Skip { reason } => reason,
        other => panic!("expected a skip, got {other:?}"),
    }
}

fn expect_reject(decision: FilterDecision) -> Error {
    match decision {
        FilterDecision::Reject(error) => error,
        other => panic!("expected a rejection, got {other:?}"),
    }
}

#[test]
fn name_filter_matches_exact_name_prefix_and_namespace() {
    let mut ctx = context(CallbackEvent::ToolInvokeInput);
    // An unconfigured dimension does not constrain the match.
    expect_allow(&decide(&NameFilter::new(), &mut ctx));
    assert_eq!(NameFilter::new().name(), "name");
    let exact = NameFilter::new().with_name("alpha");
    expect_allow(&decide_for(&exact, "alpha", NAMESPACE, &mut ctx));
    let reason = expect_skip(decide_for(&exact, "beta", NAMESPACE, &mut ctx));
    assert!(reason.contains("beta"), "reason was {reason:?}");
    let prefixed = NameFilter::new().with_prefix("pre");
    expect_allow(&decide_for(&prefixed, "prefix-h", NAMESPACE, &mut ctx));
    let reason = expect_skip(decide_for(&prefixed, "other", NAMESPACE, &mut ctx));
    assert!(reason.contains("other"), "reason was {reason:?}");
    let namespaced = NameFilter::new().with_namespace("audit");
    expect_allow(&decide_for(&namespaced, HANDLER, "audit", &mut ctx));
    let reason = expect_skip(decide_for(&namespaced, HANDLER, "default", &mut ctx));
    assert!(reason.contains("default"), "reason was {reason:?}");
    let named = NameFilter::named(["alpha", "beta"]);
    expect_allow(&decide_for(&named, "alpha", NAMESPACE, &mut ctx));
    expect_allow(&decide_for(&named, "beta", NAMESPACE, &mut ctx));
    let reason = expect_skip(decide_for(&named, "gamma", NAMESPACE, &mut ctx));
    assert!(reason.contains("gamma"), "reason was {reason:?}");
}

#[test]
fn kind_filter_matches_exact_events_and_domains() {
    let mut ctx = context(CallbackEvent::ToolInvokeInput);
    // An empty kind filter matches everything.
    expect_allow(&decide(&KindFilter::new(), &mut ctx));
    assert_eq!(KindFilter::new().name(), "kind");
    let exact = KindFilter::event(CallbackEvent::ToolInvokeInput);
    assert!(exact.events().contains(&CallbackEvent::ToolInvokeInput));
    assert!(exact.domains().is_empty());
    expect_allow(&decide(&exact, &mut ctx));
    let mut llm_ctx = context(CallbackEvent::LlmInvokeInput);
    let reason = expect_skip(decide(&exact, &mut llm_ctx));
    assert!(reason.contains("llm_invoke_input"), "reason was {reason:?}");
    let llm = KindFilter::domain(EventDomain::Llm);
    assert!(llm.domains().contains(&EventDomain::Llm));
    expect_allow(&decide(&llm, &mut llm_ctx));
    let reason = expect_skip(decide(&llm, &mut ctx));
    assert!(
        reason.contains("tool_invoke_input"),
        "reason was {reason:?}"
    );
}

#[test]
fn session_filter_requires_a_matching_session() {
    let absent = context(CallbackEvent::AgentSessionCreated);
    let reason = expect_skip(decide(&SessionFilter::only("s1"), &mut absent.clone()));
    assert!(reason.contains("no session"), "reason was {reason:?}");
    expect_allow(&decide(
        &SessionFilter::only("s1").allow_absent(),
        &mut absent.clone(),
    ));
    let session: Arc<dyn Session> = Arc::new(AgentSession::new("s1"));
    let attached = context(CallbackEvent::AgentSessionCreated).with_session(Arc::clone(&session));
    assert_eq!(attached.session_id(), Some("s1"));
    expect_allow(&decide(&SessionFilter::only("s1"), &mut attached.clone()));
    let reason = expect_skip(decide(&SessionFilter::only("s2"), &mut attached.clone()));
    assert!(
        reason.contains("not in the session set") && reason.contains("s1"),
        "reason was {reason:?}"
    );
    expect_allow(&decide(
        &SessionFilter::any_of(["s0", "s1"]),
        &mut attached.clone(),
    ));
    assert_eq!(SessionFilter::only("s1").name(), "session");
}

#[test]
fn rate_limit_allows_the_budget_then_skips_until_the_window_expires() {
    let clock = ManualClock::new();
    let filter = RateLimitFilter::new(2, Duration::from_secs(1)).with_clock(clock.clock());
    assert_eq!(filter.max_calls(), 2);
    assert_eq!(filter.window(), Duration::from_secs(1));
    assert_eq!(filter.name(), "rate_limit");
    let mut ctx = context(CallbackEvent::ToolInvokeInput);
    expect_allow(&decide(&filter, &mut ctx));
    expect_allow(&decide(&filter, &mut ctx));
    let reason = expect_skip(decide(&filter, &mut ctx));
    assert!(
        reason.contains("rate limit exceeded"),
        "reason was {reason:?}"
    );
    assert_eq!(filter.calls(&ctx, HANDLER), 2);
    // The budget is per handler: another handler has its own window.
    expect_allow(&decide_for(&filter, "other", NAMESPACE, &mut ctx));
    // Exactly one window later the calls are still counted (`now - at > window`
    // is false), so the boundary does not free a slot.
    clock.advance(Duration::from_secs(1));
    expect_skip(decide(&filter, &mut ctx));
    assert_eq!(filter.calls(&ctx, HANDLER), 2);
    // A nanosecond past the window prunes the recorded calls.
    clock.advance(Duration::from_nanos(1));
    expect_allow(&decide(&filter, &mut ctx));
    assert_eq!(filter.calls(&ctx, HANDLER), 1);
}

#[test]
fn circuit_breaker_opens_at_the_threshold_and_closes_after_the_cooldown() {
    let clock = ManualClock::new();
    let breaker = CircuitBreakerFilter::new(2, Duration::from_secs(5)).with_clock(clock.clock());
    assert_eq!(breaker.failure_threshold(), 2);
    assert_eq!(breaker.cooldown(), Duration::from_secs(5));
    assert!(breaker.feedback().is_some());
    assert_eq!(breaker.name(), "circuit_breaker");
    let ctx = context(CallbackEvent::ToolCallError);
    breaker.record_failure(&ctx, HANDLER);
    assert!(!breaker.is_open(&ctx, HANDLER));
    assert_eq!(breaker.failures(&ctx, HANDLER), 1);
    expect_allow(&decide(&breaker, &mut ctx.clone()));
    breaker.record_failure(&ctx, HANDLER);
    assert!(breaker.is_open(&ctx, HANDLER));
    assert_eq!(breaker.failures(&ctx, HANDLER), 2);
    let reason = expect_skip(decide(&breaker, &mut ctx.clone()));
    assert!(
        reason.contains("circuit breaker open"),
        "reason was {reason:?}"
    );
    assert!(reason.contains("5000 ms"), "reason was {reason:?}");
    // The cooldown boundary is strict: exactly the cooldown is still open.
    clock.advance(Duration::from_secs(5));
    expect_skip(decide(&breaker, &mut ctx.clone()));
    clock.advance(Duration::from_millis(1));
    expect_allow(&decide(&breaker, &mut ctx.clone()));
    assert!(!breaker.is_open(&ctx, HANDLER));
    assert_eq!(breaker.failures(&ctx, HANDLER), 0);
    // A success clears a partial failure count.
    breaker.record_failure(&ctx, HANDLER);
    assert_eq!(breaker.failures(&ctx, HANDLER), 1);
    breaker.record_success(&ctx, HANDLER);
    assert_eq!(breaker.failures(&ctx, HANDLER), 0);
    expect_allow(&decide(&breaker, &mut ctx.clone()));
}

#[test]
fn circuit_breaker_is_driven_by_the_feedback_helpers() {
    let clock = ManualClock::new();
    let breaker = CircuitBreakerFilter::new(1, Duration::from_secs(30)).with_clock(clock.clock());
    let filters = vec![shared(breaker.clone())];
    let ctx = context(CallbackEvent::ToolCallError);
    record_failure(&filters, &ctx, HANDLER);
    assert!(breaker.is_open(&ctx, HANDLER));
    record_success(&filters, &ctx, HANDLER);
    assert!(!breaker.is_open(&ctx, HANDLER));
    record_failure(&filters, &ctx, HANDLER);
    let mut working = ctx.clone();
    match run_filters(&filters, HANDLER, NAMESPACE, &mut working).unwrap() {
        FilterRunOutcome::Skip { filter, reason } => {
            assert_eq!(filter, "circuit_breaker");
            assert!(
                reason.contains("circuit breaker open"),
                "reason was {reason:?}"
            );
        }
        other => panic!("expected a skip, got {other:?}"),
    }
}

#[test]
fn validation_filter_rejects_with_a_validation_error() {
    let filter = ValidationFilter::new("payload", |ctx: &CallbackContext| ctx.input.is_some());
    assert_eq!(filter.message(), "argument validation failed for payload");
    assert_eq!(filter.name(), "payload");
    let mut failing = context(CallbackEvent::ToolInvokeInput);
    let error = expect_reject(decide(&filter, &mut failing));
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("payload"), "message was {error}");
    let mut passing = context(CallbackEvent::ToolInvokeInput).with_input(json!({"q": "hi"}));
    expect_allow(&decide(&filter, &mut passing));
    let configured = ValidationFilter::new("payload", |_: &CallbackContext| false)
        .with_message("tool arguments must be an object");
    assert_eq!(configured.message(), "tool arguments must be an object");
    let error = expect_reject(decide(&configured, &mut failing));
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(
        error.to_string().contains("must be an object"),
        "message was {error}"
    );
}

#[test]
fn logging_filter_always_allows_and_composes() {
    assert_eq!(LoggingFilter::new().name(), "logging");
    assert_eq!(LoggingFilter::named("audit").name(), "audit");
    let mut ctx = context(CallbackEvent::ToolInvokeInput);
    expect_allow(&decide(&LoggingFilter::new(), &mut ctx));
    let chain = FilterChain::new()
        .with(LoggingFilter::new())
        .with(NameFilter::new().with_name(HANDLER));
    assert_eq!(chain.len(), 2);
    expect_allow(&decide(&chain, &mut ctx));
    let filters = vec![shared(LoggingFilter::new())];
    match run_filters(&filters, HANDLER, NAMESPACE, &mut ctx).unwrap() {
        FilterRunOutcome::Allow => {}
        other => panic!("expected an allow, got {other:?}"),
    }
}

#[test]
fn auth_filter_authorises_from_metadata() {
    let mut anonymous = context(CallbackEvent::ToolAuth);
    let reason = expect_skip(decide(&AuthFilter::role("admin"), &mut anonymous));
    assert!(reason.contains("guest"), "reason was {reason:?}");
    assert_eq!(AuthFilter::role("admin").name(), "auth");
    let mut admin = context(CallbackEvent::ToolAuth).with_metadata("user_role", json!("admin"));
    expect_allow(&decide(&AuthFilter::role("admin"), &mut admin));
    let mut guest = context(CallbackEvent::ToolAuth).with_metadata("user_role", json!("guest"));
    let error = expect_reject(decide(&AuthFilter::role("admin").rejecting(), &mut guest));
    assert_eq!(error.kind(), ErrorKind::Security);
    let mut untokened = context(CallbackEvent::ToolAuth);
    let reason = expect_skip(decide(&AuthFilter::token("t0k"), &mut untokened));
    assert!(reason.contains("auth_token"), "reason was {reason:?}");
    let mut tokened = context(CallbackEvent::ToolAuth).with_metadata("auth_token", json!("t0k"));
    expect_allow(&decide(&AuthFilter::token("t0k"), &mut tokened));
    let predicate = AuthFilter::predicate("tenant", |ctx: &CallbackContext| {
        ctx.metadata.contains_key("tenant")
    });
    assert_eq!(predicate.name(), "tenant");
    let reason = expect_skip(decide(&predicate, &mut anonymous));
    assert!(reason.contains("tenant"), "reason was {reason:?}");
    let mut tenanted = context(CallbackEvent::ToolAuth).with_metadata("tenant", json!("acme"));
    expect_allow(&decide(&predicate, &mut tenanted));
}

#[test]
fn param_modify_filter_mutations_are_visible_downstream() {
    let seen: Arc<Mutex<Vec<Option<Value>>>> = Arc::new(Mutex::new(Vec::new()));
    let observer = {
        let seen = Arc::clone(&seen);
        move |input: FilterInput<'_>| {
            seen.lock().unwrap().push(input.ctx.input.clone());
            Ok(FilterDecision::Allow)
        }
    };
    let filters = vec![
        shared(ParamModifyFilter::input(|value: &Value| {
            Ok(json!(format!(
                "{}-rewritten",
                value.as_str().unwrap_or_default()
            )))
        })),
        shared(observer),
    ];
    let mut ctx = context(CallbackEvent::ToolInvokeInput).with_input(json!("prompt"));
    match run_filters(&filters, HANDLER, NAMESPACE, &mut ctx).unwrap() {
        FilterRunOutcome::Allow => {}
        other => panic!("expected an allow, got {other:?}"),
    }
    assert_eq!(
        *seen.lock().unwrap(),
        vec![Some(json!("prompt-rewritten"))],
        "the downstream filter must see the rewritten input"
    );
    assert_eq!(ctx.input, Some(json!("prompt-rewritten")));
    // A named modifier may rewrite anything on the context.
    let tagger = ParamModifyFilter::new("tag", |ctx: &mut CallbackContext| {
        ctx.metadata.insert("tagged".to_owned(), json!(true));
        Ok(())
    });
    assert_eq!(tagger.name(), "tag");
    match run_filters(&[shared(tagger)], HANDLER, NAMESPACE, &mut ctx).unwrap() {
        FilterRunOutcome::Allow => {}
        other => panic!("expected an allow, got {other:?}"),
    }
    assert_eq!(ctx.metadata.get("tagged"), Some(&json!(true)));
    // A failing modifier propagates its error (fail-closed).
    let failing = vec![shared(ParamModifyFilter::input(|_: &Value| {
        Err(Error::validation("modifier rejected the payload"))
    }))];
    let error = run_filters(&failing, HANDLER, NAMESPACE, &mut ctx).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(
        error.to_string().contains("modifier rejected the payload"),
        "message was {error}"
    );
    // `lenient(..)` degrades the same failure into a skip.
    let lenient_filters = vec![shared(lenient(ParamModifyFilter::input(|_: &Value| {
        Err(Error::validation("modifier rejected the payload"))
    })))];
    match run_filters(&lenient_filters, HANDLER, NAMESPACE, &mut ctx).unwrap() {
        FilterRunOutcome::Skip { filter, reason } => {
            assert_eq!(filter, "lenient");
            assert!(
                reason.contains("modifier rejected the payload"),
                "reason was {reason:?}"
            );
        }
        other => panic!("expected a skip, got {other:?}"),
    }
}

#[test]
fn conditional_filter_delegates_to_its_inner_filter_only_when_the_condition_holds() {
    let mut ctx = context(CallbackEvent::ToolInvokeInput);
    let never = |_: &CallbackContext| false;
    let skipped = ConditionalFilter::new(never, NameFilter::new());
    assert_eq!(skipped.name(), "conditional(name)");
    let reason = expect_skip(decide(&skipped, &mut ctx));
    assert_eq!(reason, "condition not satisfied");
    let exempt = ConditionalFilter::new(never, NameFilter::new()).allow_on_false();
    expect_allow(&decide(&exempt, &mut ctx));
    // A true condition delegates, including the inner rejection.
    let rejecting = ConditionalFilter::new(
        |_: &CallbackContext| true,
        ValidationFilter::new("payload", |ctx: &CallbackContext| ctx.input.is_some()),
    );
    let error = expect_reject(decide(&rejecting, &mut ctx));
    assert_eq!(error.kind(), ErrorKind::Validation);
    let allowing = ConditionalFilter::new(|_: &CallbackContext| true, NameFilter::new());
    expect_allow(&decide(&allowing, &mut ctx));
}

#[test]
fn composition_stops_at_the_first_skip_and_or_preserves_rejections() {
    let hard = || ValidationFilter::new("hard", |_: &CallbackContext| false);
    let mut ctx = context(CallbackEvent::ToolInvokeInput);
    let allow = NameFilter::new().with_name(HANDLER);
    let skip = NameFilter::new().with_name("somebody_else");
    // AND: the first non-allow wins.
    expect_skip(decide(&and(allow.clone(), skip.clone()), &mut ctx));
    // A skip before a rejection short-circuits the chain.
    let mut chain = FilterChain::new();
    chain.push(skip.clone());
    chain.push(hard());
    assert_eq!(chain.len(), 2);
    assert!(!chain.is_empty());
    expect_skip(decide(&chain, &mut ctx));
    // OR: either side allowing is enough.
    expect_allow(&decide(&or(skip.clone(), allow.clone()), &mut ctx));
    expect_allow(&decide(&or(allow, hard()), &mut ctx));
    // OR propagates a rejection even when the other side only skips.
    let error = expect_reject(decide(&or(hard(), skip.clone()), &mut ctx));
    assert_eq!(error.kind(), ErrorKind::Validation);
    // Both sides skipping yields one combined reason.
    let reason = expect_skip(decide(&or(skip.clone(), skip.clone()), &mut ctx));
    assert!(
        reason.contains("neither filter allowed the handler"),
        "reason was {reason:?}"
    );
    // NOT inverts allow/skip but never a rejection.
    expect_skip(decide(&not(NameFilter::new().with_name(HANDLER)), &mut ctx));
    expect_allow(&decide(&not(skip), &mut ctx));
    let error = expect_reject(decide(&not(hard()), &mut ctx));
    assert_eq!(error.kind(), ErrorKind::Validation);
}

#[test]
fn event_filter_ext_and_arc_composition_are_storable() {
    let mut ctx = context(CallbackEvent::ToolInvokeInput);
    let gated = NameFilter::new()
        .with_name(HANDLER)
        .and(PredicateFilter::new("open", |_: &CallbackContext| true));
    expect_allow(&decide(&gated, &mut ctx));
    expect_allow(&decide(
        &NameFilter::new().with_name("nobody").negate(),
        &mut ctx,
    ));
    expect_allow(&decide(
        &NameFilter::new()
            .with_name("nobody")
            .or(NameFilter::new().with_name(HANDLER)),
        &mut ctx,
    ));
    let failing = ClosureFilter::new("boom", |_: FilterInput<'_>| {
        Err(Error::validation("filter exploded"))
    });
    let reason = expect_skip(decide(&failing.lenient(), &mut ctx));
    assert!(reason.contains("filter exploded"), "reason was {reason:?}");
    let handle: Arc<dyn EventFilter> = NameFilter::new().with_name(HANDLER).into_filter_arc();
    assert_eq!(handle.name(), "name");
    // A shared handle composes like any concrete filter.
    let shared_chain = handle.clone().and(NameFilter::new().with_prefix("hand"));
    expect_allow(&decide(&shared_chain, &mut ctx));
    let chained = FilterChain::new()
        .with_arc(handle)
        .with_arc(NameFilter::new().with_name("not-this").into_filter_arc());
    assert_eq!(
        chained
            .filters()
            .iter()
            .map(EventFilter::name)
            .collect::<Vec<_>>(),
        ["name", "name"]
    );
    expect_skip(decide(&chained, &mut ctx));
}

#[test]
fn run_filters_names_the_first_skipping_filter_and_propagates_errors() {
    let calls = Arc::new(AtomicUsize::new(0));
    let trailing = {
        let calls = Arc::clone(&calls);
        move |_: FilterInput<'_>| {
            calls.fetch_add(1, Ordering::Relaxed);
            Ok(FilterDecision::Allow)
        }
    };
    let filters: Vec<Arc<dyn EventFilter>> = vec![
        shared(LoggingFilter::new()),
        shared(PredicateFilter::new("gate", |_: &CallbackContext| false)),
        shared(trailing),
    ];
    let mut ctx = context(CallbackEvent::ToolInvokeInput);
    match run_filters(&filters, HANDLER, NAMESPACE, &mut ctx).unwrap() {
        FilterRunOutcome::Skip { filter, reason } => {
            assert_eq!(filter, "gate");
            assert_eq!(reason, "gate predicate rejected the handler");
        }
        other => panic!("expected a skip, got {other:?}"),
    }
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    // A filter that cannot decide fails the run (fail-closed).
    let failing = vec![shared(ClosureFilter::new(
        "explode",
        |_: FilterInput<'_>| Err(Error::validation("filter exploded")),
    ))];
    let error = run_filters(&failing, HANDLER, NAMESPACE, &mut ctx).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert_eq!(error.to_string(), "validation: filter exploded");
    // A rejection is an outcome, not an `Err`.
    let rejecting = vec![shared(ValidationFilter::new(
        "payload",
        |_: &CallbackContext| false,
    ))];
    match run_filters(&rejecting, HANDLER, NAMESPACE, &mut ctx).unwrap() {
        FilterRunOutcome::Reject(error) => assert_eq!(error.kind(), ErrorKind::Validation),
        other => panic!("expected a rejection, got {other:?}"),
    }
    // An empty list allows.
    match run_filters(&[], HANDLER, NAMESPACE, &mut ctx).unwrap() {
        FilterRunOutcome::Allow => {}
        other => panic!("expected an allow, got {other:?}"),
    }
}

#[test]
fn closure_filters_use_default_and_explicit_names() {
    let mut ctx = context(CallbackEvent::ToolInvokeInput);
    let bare = |input: FilterInput<'_>| {
        if input.handler == HANDLER {
            Ok(FilterDecision::Allow)
        } else {
            Ok(FilterDecision::skip(format!(
                "unexpected handler '{}'",
                input.handler
            )))
        }
    };
    assert_eq!(bare.name(), "filter");
    expect_allow(&decide(&bare, &mut ctx));
    let reason = expect_skip(decide_for(&bare, "other", NAMESPACE, &mut ctx));
    assert!(reason.contains("other"), "reason was {reason:?}");
    let named = ClosureFilter::new("tenant", |input: FilterInput<'_>| {
        if input.namespace == "tenant-a" {
            Ok(FilterDecision::Allow)
        } else {
            Ok(FilterDecision::skip("wrong namespace"))
        }
    });
    assert_eq!(named.name(), "tenant");
    expect_allow(&decide_for(&named, HANDLER, "tenant-a", &mut ctx));
    match run_filters(&[shared(named)], HANDLER, NAMESPACE, &mut ctx).unwrap() {
        FilterRunOutcome::Skip { filter, reason } => {
            assert_eq!(filter, "tenant");
            assert_eq!(reason, "wrong namespace");
        }
        other => panic!("expected a skip, got {other:?}"),
    }
}

/// A handler that counts its invocations.
fn counting_handler(runs: Arc<AtomicUsize>) -> Arc<dyn CallbackHandler> {
    Arc::new(SyncHandler::new(move |_: &CallbackContext| {
        runs.fetch_add(1, Ordering::Relaxed);
        Ok(CallbackVerdict::Continue)
    }))
}

#[tokio::test]
async fn filters_gate_callback_chain_entries_end_to_end() {
    let runs = Arc::new(AtomicUsize::new(0));
    let chain = CallbackChain::new("filters");
    chain.add(
        ChainEntry::new("allowed", counting_handler(Arc::clone(&runs)))
            .when(PredicateFilter::new("open", |_: &CallbackContext| true)),
    );
    chain.add(
        ChainEntry::new("blocked", counting_handler(Arc::clone(&runs)))
            .when(PredicateFilter::new("closed", |_: &CallbackContext| false)),
    );
    let mut ctx = CallbackContext::new(CallbackEvent::ToolInvokeInput);
    let detailed = chain
        .trigger_detailed(CallbackEvent::ToolInvokeInput, &mut ctx)
        .await
        .unwrap();
    assert_eq!(runs.load(Ordering::Relaxed), 1);
    assert_eq!(detailed.report.fired, 1);
    assert_eq!(detailed.ran().len(), 1);
    assert_eq!(detailed.ran()[0].name, "allowed");
    let skipped = detailed.skipped();
    assert_eq!(skipped.len(), 1);
    assert_eq!(skipped[0].name, "blocked");
    match &skipped[0].outcome {
        EntryOutcome::Skipped { filter, reason } => {
            assert_eq!(filter, "closed");
            assert_eq!(reason, "closed predicate rejected the handler");
        }
        other => panic!("expected a skip, got {other:?}"),
    }
}
