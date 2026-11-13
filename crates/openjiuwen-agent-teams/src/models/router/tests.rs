//! Behavioural tests for the request router.
//!
//! Deterministic and offline: scripted pools, in-memory state, no sleeps, and a bounded `tokio`
//! barrier for the concurrent least-busy scenario.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;

use openjiuwen_common::error::ErrorKind;
use serde_json::json;
use tokio::sync::Barrier;

use super::*;

/// Builds `n` distinct entries named `m0`, `m1`, …
fn entries(n: usize) -> Vec<ModelPoolEntry> {
    (0..n)
        .map(|i| {
            ModelPoolEntry::new(
                format!("m{i}"),
                "openai",
                format!("https://endpoint-{i}.example/v1"),
                format!("sk-{i}"),
            )
        })
        .collect()
}

/// Builds a router over `entries(n)`.
fn router(n: usize, strategy: RouterStrategy) -> ModelRouter {
    ModelRouter::new(ModelPool::new(entries(n)).expect("valid pool"), strategy)
}

#[test]
fn pool_strategy_maps_onto_router_strategy() {
    assert_eq!(
        RouterStrategy::from(ModelPoolStrategy::RoundRobin),
        RouterStrategy::RoundRobin
    );
    assert_eq!(
        RouterStrategy::from(ModelPoolStrategy::ByModelName),
        RouterStrategy::ByName
    );
    assert_eq!(RouterStrategy::default(), RouterStrategy::RoundRobin);
}

#[test]
fn router_wire_names_match_the_documented_strings() {
    let cases = [
        (RouterStrategy::ByName, "by_name"),
        (RouterStrategy::RoundRobin, "round_robin"),
        (RouterStrategy::LeastBusy, "least_busy"),
        (RouterStrategy::RuleBased, "rule_based"),
    ];
    for (strategy, wire) in cases {
        assert_eq!(strategy.as_str(), wire);
        assert_eq!(serde_json::to_value(strategy).unwrap(), json!(wire));
        let parsed: RouterStrategy = serde_json::from_value(json!(wire)).unwrap();
        assert_eq!(parsed, strategy);
    }
}

#[test]
fn route_reason_wire_names_match_the_documented_strings() {
    let cases = [
        (RouteReason::RequestedName, "requested_name"),
        (RouteReason::Rule, "rule"),
        (RouteReason::RoundRobin, "round_robin"),
        (RouteReason::LeastBusy, "least_busy"),
        (RouteReason::FirstEntry, "first_entry"),
    ];
    for (reason, wire) in cases {
        assert_eq!(reason.as_str(), wire);
        assert_eq!(serde_json::to_value(reason).unwrap(), json!(wire));
        let parsed: RouteReason = serde_json::from_value(json!(wire)).unwrap();
        assert_eq!(parsed, reason);
    }
    assert_eq!(RouteReason::default(), RouteReason::FirstEntry);
}

#[test]
fn complexity_is_ordered_and_serializes_snake_case() {
    assert!(Complexity::Low < Complexity::Medium);
    assert!(Complexity::Medium < Complexity::High);
    assert_eq!(
        serde_json::to_value(Complexity::High).unwrap(),
        json!("high")
    );
    let parsed: Complexity = serde_json::from_value(json!("medium")).unwrap();
    assert_eq!(parsed, Complexity::Medium);
}

#[test]
fn request_hint_builders_fill_the_expected_fields() {
    assert_eq!(RequestHint::new(), RequestHint::default());
    let hint = RequestHint::for_role(TeamRole::Leader)
        .with_model_name("m1")
        .with_task_kind("review")
        .with_complexity(Complexity::High)
        .with_needs_tools(true);
    assert_eq!(
        hint,
        RequestHint {
            model_name: Some("m1".to_owned()),
            role: Some(TeamRole::Leader),
            task_kind: Some("review".to_owned()),
            complexity: Some(Complexity::High),
            needs_tools: true,
        }
    );
    assert_eq!(
        hint.with_role(TeamRole::Teammate).role,
        Some(TeamRole::Teammate)
    );
}

#[test]
fn request_hint_tolerates_unknown_json_fields() {
    let hint: RequestHint =
        serde_json::from_value(json!({"model_name": "m0", "future_hint": 7})).unwrap();
    assert_eq!(hint.model_name.as_deref(), Some("m0"));
    assert!(!hint.needs_tools);
}

#[test]
fn default_rule_matcher_matches_everything() {
    let matcher = RuleMatcher::default();
    assert!(matcher.matches(&RequestHint::new()));
    assert!(matcher.matches(&RequestHint::for_role(TeamRole::Teammate).with_needs_tools(true)));
}

#[test]
fn rule_matcher_checks_role_and_task_kind() {
    let matcher = RuleMatcher::default()
        .with_role(TeamRole::Leader)
        .with_task_kind("review");
    assert!(matcher.matches(&RequestHint::for_role(TeamRole::Leader).with_task_kind("review")));
    assert!(!matcher.matches(&RequestHint::for_role(TeamRole::Teammate).with_task_kind("review")));
    assert!(!matcher.matches(&RequestHint::for_role(TeamRole::Leader).with_task_kind("coding")));
    // A missing hint value never satisfies a constraint.
    assert!(!matcher.matches(&RequestHint::new()));
}

#[test]
fn rule_matcher_checks_complexity_exactly_and_at_least() {
    let exact = RuleMatcher::default().with_complexity(Complexity::Medium);
    assert!(exact.matches(&RequestHint::new().with_complexity(Complexity::Medium)));
    assert!(!exact.matches(&RequestHint::new().with_complexity(Complexity::High)));

    let at_least = RuleMatcher::default().with_min_complexity(Complexity::Medium);
    assert!(!at_least.matches(&RequestHint::new().with_complexity(Complexity::Low)));
    assert!(at_least.matches(&RequestHint::new().with_complexity(Complexity::Medium)));
    assert!(at_least.matches(&RequestHint::new().with_complexity(Complexity::High)));
    // No declared complexity cannot prove it clears the threshold.
    assert!(!at_least.matches(&RequestHint::new()));

    // Both constraints must hold together.
    let both = RuleMatcher::default()
        .with_complexity(Complexity::High)
        .with_min_complexity(Complexity::Medium);
    assert!(both.matches(&RequestHint::new().with_complexity(Complexity::High)));
    assert!(!both.matches(&RequestHint::new().with_complexity(Complexity::Medium)));
}

#[test]
fn rule_matcher_checks_needs_tools_in_both_directions() {
    let with_tools = RuleMatcher::default().with_needs_tools(true);
    assert!(with_tools.matches(&RequestHint::new().with_needs_tools(true)));
    assert!(!with_tools.matches(&RequestHint::new()));
    let without_tools = RuleMatcher::default().with_needs_tools(false);
    assert!(without_tools.matches(&RequestHint::new()));
    assert!(!without_tools.matches(&RequestHint::new().with_needs_tools(true)));
}

#[test]
fn routing_rule_reports_its_matcher_and_priority() {
    let rule = RoutingRule::new(
        "leaders-use-m1",
        RuleMatcher::default().with_role(TeamRole::Leader),
        "m1",
    )
    .with_priority(5);
    assert_eq!(rule.name, "leaders-use-m1");
    assert_eq!(rule.priority, 5);
    assert_eq!(rule.target, "m1");
    assert!(rule.matches(&RequestHint::for_role(TeamRole::Leader)));
    assert!(!rule.matches(&RequestHint::for_role(TeamRole::Teammate)));
    assert_eq!(RoutingRule::default().priority, 0);
}

#[test]
fn router_debug_reports_shape_without_credentials() {
    let router = router(2, RouterStrategy::RoundRobin);
    let debug = format!("{router:?}");
    assert!(debug.contains("round_robin"), "{debug}");
    assert!(debug.contains("pool: 2"), "{debug}");
    assert!(!debug.contains("sk-0"), "{debug}");
}

#[test]
fn empty_pool_rejects_every_decision() {
    let router = ModelRouter::new(ModelPool::empty(), RouterStrategy::RoundRobin);
    for hint in [RequestHint::new(), RequestHint::new().with_model_name("m0")] {
        let error = router.decide(&hint).expect_err("empty pool");
        assert_eq!(error.kind(), ErrorKind::Config);
        assert!(error.to_string().contains("model_pool"), "{error}");

        let error = router.acquire(&hint).expect_err("empty pool");
        assert_eq!(error.kind(), ErrorKind::Config);
        assert!(error.to_string().contains("model_pool"), "{error}");
    }
    assert_eq!(router.in_flight(), Vec::<usize>::new());
    assert_eq!(router.total_in_flight(), 0);
}

#[test]
fn for_pool_accepts_an_empty_slice_and_validates_entries() {
    let empty = ModelRouter::for_pool(&[], ModelPoolStrategy::RoundRobin).expect("empty is legal");
    assert!(empty.pool().is_empty());
    assert_eq!(empty.strategy(), RouterStrategy::RoundRobin);

    let mut duplicate = entries(2);
    duplicate[1].model_name = "m0".to_owned();
    let error = ModelRouter::for_pool(&duplicate, ModelPoolStrategy::ByModelName)
        .expect_err("duplicate names");
    assert_eq!(error.kind(), ErrorKind::Validation);

    let by_name =
        ModelRouter::for_pool(&entries(2), ModelPoolStrategy::ByModelName).expect("valid");
    assert_eq!(by_name.strategy(), RouterStrategy::ByName);
    assert_eq!(by_name.pool().len(), 2);
}

#[test]
fn by_name_serves_the_requested_model() {
    let router = router(3, RouterStrategy::ByName);
    let decision = router
        .decide(&RequestHint::new().with_model_name("m2"))
        .expect("non-empty pool");
    assert_eq!(decision.index, 2);
    assert_eq!(decision.entry.model_name, "m2");
    assert_eq!(decision.reason, RouteReason::RequestedName);
    assert_eq!(decision.rule, None);
    assert_eq!(decision.fallback_chain, vec!["m0", "m1"]);
    // A preview reserves nothing.
    assert_eq!(router.total_in_flight(), 0);
}

#[test]
fn by_name_falls_back_to_the_first_entry() {
    let router = router(3, RouterStrategy::ByName);

    let unknown = router
        .decide(&RequestHint::new().with_model_name("m9"))
        .expect("non-empty pool");
    assert_eq!(unknown.index, 0);
    assert_eq!(unknown.entry.model_name, "m0");
    assert_eq!(unknown.reason, RouteReason::FirstEntry);
    assert_eq!(unknown.fallback_chain, vec!["m1", "m2"]);

    let unnamed = router.decide(&RequestHint::new()).expect("non-empty pool");
    assert_eq!(unnamed.index, 0);
    assert_eq!(unnamed.reason, RouteReason::FirstEntry);
    assert_eq!(unnamed.fallback_chain, vec!["m1", "m2"]);
}

#[test]
fn by_name_acquire_reserves_and_releases_the_slot() {
    let router = router(2, RouterStrategy::ByName);
    let hint = RequestHint::new().with_model_name("m1");
    let guard = router.acquire(&hint).expect("pooled");
    assert_eq!(guard.entry().model_name, "m1");
    assert_eq!(guard.decision().reason, RouteReason::RequestedName);
    assert_eq!(router.in_flight(), vec![0, 1]);
    assert_eq!(router.total_in_flight(), 1);
    drop(guard);
    assert_eq!(router.in_flight(), vec![0, 0]);
}

#[test]
fn round_robin_preview_does_not_advance_the_cursor() {
    let router = router(3, RouterStrategy::RoundRobin);
    let hint = RequestHint::new();
    for _ in 0..3 {
        let decision = router.decide(&hint).expect("non-empty pool");
        assert_eq!(decision.index, 0);
        assert_eq!(decision.reason, RouteReason::RoundRobin);
        assert_eq!(decision.fallback_chain, vec!["m1", "m2"]);
    }
    assert_eq!(router.total_in_flight(), 0);
}

#[test]
fn round_robin_acquire_is_fair_over_the_pool() {
    let router = router(3, RouterStrategy::RoundRobin);
    let hint = RequestHint::new();
    let mut picked = Vec::new();
    for _ in 0..9 {
        let guard = router.acquire(&hint).expect("non-empty pool");
        picked.push(guard.decision().index);
        // Dropping frees the slot but must not rewind the rotation.
        drop(guard);
    }
    assert_eq!(picked, vec![0, 1, 2, 0, 1, 2, 0, 1, 2]);
    assert_eq!(router.total_in_flight(), 0);
}

#[test]
fn round_robin_ignores_the_requested_name() {
    let router = router(3, RouterStrategy::RoundRobin);
    let first = router
        .acquire(&RequestHint::new().with_model_name("m2"))
        .expect("non-empty pool");
    assert_eq!(first.decision().index, 0);
    assert_eq!(first.decision().reason, RouteReason::RoundRobin);
    drop(first);
    let second = router
        .acquire(&RequestHint::new().with_model_name("m2"))
        .expect("non-empty pool");
    assert_eq!(second.decision().index, 1);
}

#[test]
fn least_busy_spreads_over_the_pool_and_prefers_the_lowest_index() {
    let router = router(3, RouterStrategy::LeastBusy);
    let hint = RequestHint::new();
    let first = router.acquire(&hint).expect("non-empty pool");
    let second = router.acquire(&hint).expect("non-empty pool");
    let third = router.acquire(&hint).expect("non-empty pool");
    assert_eq!(
        [
            first.decision().index,
            second.decision().index,
            third.decision().index
        ],
        [0, 1, 2]
    );
    assert_eq!(router.in_flight(), vec![1, 1, 1]);

    // All loads are equal again: the tie goes to the lowest pool index.
    let fourth = router.acquire(&hint).expect("non-empty pool");
    assert_eq!(fourth.decision().index, 0);
    assert_eq!(fourth.decision().reason, RouteReason::LeastBusy);
    assert_eq!(router.in_flight(), vec![2, 1, 1]);

    // A preview reports where the next acquisition would go, without changing counts.
    let preview = router.decide(&hint).expect("non-empty pool");
    assert_eq!(preview.index, 1);
    assert_eq!(router.in_flight(), vec![2, 1, 1]);

    drop(fourth);
    assert_eq!(router.in_flight(), vec![1, 1, 1]);
}

#[test]
fn least_busy_fallback_chain_follows_current_load() {
    let router = router(3, RouterStrategy::LeastBusy);
    let hint = RequestHint::new();
    let first = router.acquire(&hint).expect("non-empty pool");
    let second = router.acquire(&hint).expect("non-empty pool");
    // Loads are [1, 1, 0]: the next pick is m2, whose fallback starts with the lighter peers.
    let decision = router.decide(&hint).expect("non-empty pool");
    assert_eq!(decision.index, 2);
    assert_eq!(decision.fallback_chain, vec!["m0", "m1"]);
    drop(first);
    drop(second);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn least_busy_spreads_eight_concurrent_acquisitions() {
    let router = Arc::new(router(3, RouterStrategy::LeastBusy));
    let barrier = Arc::new(Barrier::new(8));
    let mut tasks = Vec::new();
    for _ in 0..8 {
        let router = Arc::clone(&router);
        let barrier = Arc::clone(&barrier);
        tasks.push(tokio::spawn(async move {
            barrier.wait().await;
            let hint = RequestHint::new();
            let guard = router.acquire(&hint).expect("non-empty pool");
            let index = guard.decision().index;
            // Hold every guard until all eight acquisitions have happened, so the
            // distribution reflects simultaneous load rather than sequential reuse.
            barrier.wait().await;
            drop(guard);
            index
        }));
    }
    let mut counts = [0usize; 3];
    for task in tasks {
        counts[task.await.expect("task joined")] += 1;
    }
    assert_eq!(counts, [3, 3, 2]);
    assert_eq!(router.total_in_flight(), 0);
}

#[test]
fn rule_based_applies_the_first_matching_rule() {
    let router = router(3, RouterStrategy::RuleBased).with_rules(vec![
        RoutingRule::new(
            "leaders",
            RuleMatcher::default().with_role(TeamRole::Leader),
            "m2",
        ),
        RoutingRule::new(
            "coders",
            RuleMatcher::default().with_task_kind("coding"),
            "m1",
        )
        .with_priority(1),
    ]);
    let decision = router
        .decide(&RequestHint::for_role(TeamRole::Leader).with_task_kind("coding"))
        .expect("non-empty pool");
    assert_eq!(decision.index, 2);
    assert_eq!(decision.reason, RouteReason::Rule);
    assert_eq!(decision.rule.as_deref(), Some("leaders"));
    // The other matching target follows first, then the remaining pool order.
    assert_eq!(decision.fallback_chain, vec!["m1", "m0"]);

    // A hint the second rule does not match leaves only the pool order for the chain.
    let leader_only = router
        .decide(&RequestHint::for_role(TeamRole::Leader))
        .expect("non-empty pool");
    assert_eq!(leader_only.rule.as_deref(), Some("leaders"));
    assert_eq!(leader_only.fallback_chain, vec!["m0", "m1"]);
}

#[test]
fn rule_based_orders_by_priority_then_name() {
    let router = router(3, RouterStrategy::RuleBased).with_rules(vec![
        RoutingRule::new("b-low", RuleMatcher::default(), "m1").with_priority(9),
        RoutingRule::new("a-high", RuleMatcher::default(), "m2").with_priority(1),
        RoutingRule::new("c-high", RuleMatcher::default(), "m1").with_priority(1),
    ]);
    assert_eq!(
        router
            .rules()
            .iter()
            .map(|rule| rule.name.as_str())
            .collect::<Vec<_>>(),
        vec!["a-high", "c-high", "b-low"]
    );
    let decision = router.decide(&RequestHint::new()).expect("non-empty pool");
    assert_eq!(decision.index, 2);
    assert_eq!(decision.rule.as_deref(), Some("a-high"));
    assert_eq!(decision.fallback_chain, vec!["m1", "m0"]);
}

#[test]
fn rule_target_missing_from_the_pool_skips_to_the_next_rule() {
    let router = router(2, RouterStrategy::RuleBased).with_rules(vec![
        RoutingRule::new("retired-model", RuleMatcher::default(), "m9"),
        RoutingRule::new("fallback-model", RuleMatcher::default(), "m1"),
    ]);
    let decision = router.decide(&RequestHint::new()).expect("non-empty pool");
    assert_eq!(decision.index, 1);
    assert_eq!(decision.rule.as_deref(), Some("fallback-model"));
    assert_eq!(decision.fallback_chain, vec!["m0"]);
}

#[test]
fn rule_based_falls_through_to_by_name_then_the_first_entry() {
    let router = router(3, RouterStrategy::RuleBased).with_rules(vec![RoutingRule::new(
        "coders",
        RuleMatcher::default().with_task_kind("coding"),
        "m1",
    )]);

    // No rule matches, but the hint names a pooled model: by-name semantics apply.
    let named = router
        .decide(&RequestHint::new().with_model_name("m2"))
        .expect("non-empty pool");
    assert_eq!(named.index, 2);
    assert_eq!(named.reason, RouteReason::RequestedName);
    assert_eq!(named.rule, None);
    assert_eq!(named.fallback_chain, vec!["m0", "m1"]);

    // Neither a matching rule nor a pooled name: the first entry.
    let unnamed = router.decide(&RequestHint::new()).expect("non-empty pool");
    assert_eq!(unnamed.index, 0);
    assert_eq!(unnamed.reason, RouteReason::FirstEntry);
    assert_eq!(unnamed.fallback_chain, vec!["m1", "m2"]);
}

#[test]
fn rule_based_without_rules_uses_the_by_name_fallback() {
    let router = router(2, RouterStrategy::RuleBased);
    let decision = router
        .decide(&RequestHint::new().with_model_name("m1"))
        .expect("non-empty pool");
    assert_eq!(decision.index, 1);
    assert_eq!(decision.reason, RouteReason::RequestedName);
    assert_eq!(router.rules().len(), 0);
}

#[test]
fn fallback_chain_is_deduplicated_and_excludes_the_chosen_model() {
    let router = router(2, RouterStrategy::RuleBased).with_rules(vec![
        RoutingRule::new("first", RuleMatcher::default(), "m1"),
        RoutingRule::new("duplicate-target", RuleMatcher::default(), "m1"),
        RoutingRule::new("first-entry", RuleMatcher::default(), "m0"),
    ]);
    let decision = router.decide(&RequestHint::new()).expect("non-empty pool");
    assert_eq!(decision.index, 1);
    assert_eq!(decision.fallback_chain, vec!["m0"]);
    assert!(!decision.fallback_chain.contains(&"m1".to_owned()));
}

#[test]
fn fallback_chain_of_a_single_entry_pool_is_empty() {
    let router = router(1, RouterStrategy::RoundRobin);
    let decision = router.decide(&RequestHint::new()).expect("non-empty pool");
    assert_eq!(decision.index, 0);
    assert_eq!(decision.fallback_chain.len(), 0);
}

#[test]
fn guard_releases_its_slot_when_a_scope_returns_an_error() {
    fn fail_after_acquiring(router: &ModelRouter) -> Result<()> {
        let _guard = router.acquire(&RequestHint::new())?;
        Err(Error::config("caller failed"))
    }

    let router = router(2, RouterStrategy::LeastBusy);
    let error = fail_after_acquiring(&router).expect_err("caller error");
    assert_eq!(error.kind(), ErrorKind::Config);
    assert_eq!(router.in_flight(), vec![0, 0]);
    assert_eq!(router.total_in_flight(), 0);
}

#[test]
fn guard_releases_its_slot_on_panic_unwind() {
    let router = router(1, RouterStrategy::LeastBusy);
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _guard = router.acquire(&RequestHint::new()).expect("non-empty pool");
        assert_eq!(router.total_in_flight(), 1);
        panic!("request panicked");
    }));
    assert!(outcome.is_err());
    assert_eq!(router.total_in_flight(), 0);
}

#[test]
fn decision_round_trips_through_json() {
    let router = router(3, RouterStrategy::RuleBased).with_rules(vec![RoutingRule::new(
        "leaders",
        RuleMatcher::default().with_role(TeamRole::Leader),
        "m2",
    )]);
    let decision = router
        .decide(&RequestHint::for_role(TeamRole::Leader))
        .expect("non-empty pool");
    let value = serde_json::to_value(&decision).unwrap();
    assert_eq!(value["index"], json!(2));
    assert_eq!(value["reason"], json!("rule"));
    assert_eq!(value["rule"], json!("leaders"));
    assert_eq!(value["fallback_chain"], json!(["m0", "m1"]));
    assert_eq!(value["entry"]["model_name"], json!("m2"));
    assert_eq!(
        value["entry"]["api_base_url"],
        json!("https://endpoint-2.example/v1")
    );
    let parsed: RouterDecision = serde_json::from_value(value).unwrap();
    assert_eq!(parsed, decision);

    // Unknown fields and missing optional ones are tolerated.
    let tolerant: RouterDecision =
        serde_json::from_value(json!({"index": 4, "future_field": true})).unwrap();
    assert_eq!(tolerant.index, 4);
    assert_eq!(tolerant.reason, RouteReason::FirstEntry);
    assert!(tolerant.rule.is_none());
}

#[test]
fn routing_rule_round_trips_through_json() {
    let rule = RoutingRule::new(
        "leaders-high",
        RuleMatcher::default()
            .with_role(TeamRole::Leader)
            .with_min_complexity(Complexity::High)
            .with_needs_tools(true),
        "m2",
    )
    .with_priority(3);
    let value = serde_json::to_value(&rule).unwrap();
    assert_eq!(value["matcher"]["min_complexity"], json!("high"));
    assert_eq!(value["matcher"]["needs_tools"], json!(true));
    assert_eq!(value["priority"], json!(3));
    let parsed: RoutingRule = serde_json::from_value(value).unwrap();
    assert_eq!(parsed, rule);
}

#[test]
fn rules_attached_to_a_non_rule_strategy_are_reported_but_unused() {
    let router = router(2, RouterStrategy::RoundRobin).with_rules(vec![RoutingRule::new(
        "coders",
        RuleMatcher::default(),
        "m1",
    )]);
    assert_eq!(router.rules().len(), 1);
    let decision = router.decide(&RequestHint::new()).expect("non-empty pool");
    assert_eq!(decision.index, 0);
    assert_eq!(decision.reason, RouteReason::RoundRobin);
    assert_eq!(decision.rule, None);
}
