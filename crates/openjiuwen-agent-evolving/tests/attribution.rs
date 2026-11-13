//! Integration tests for the `attribution` module.
//!
//! Exercises the dependency graph (validation, traversal, topological order,
//! closure, subgraphs, serde), credit assignment (decay, edge weights, depth
//! cut-off, normalisation, ranking), the deterministic keyword attributor (UTF-8
//! positions, case folding, ordering), the reward-stage round trip, and the
//! diagnostic-signal records. Everything is offline and deterministic: no model
//! calls, no clock, no environment.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::collections::BTreeMap;

use openjiuwen_agent_evolving::attribution::{
    reward_tier, AttributionEdge, AttributionGraph, AttributionNode, CreditAssigner,
    DiagnosticSignal, KeywordAttributor, KeywordRule, RewardStageAdapter, SignalSource,
    SIGNAL_TIERS,
};
use openjiuwen_agent_evolving::reward::{RewardPack, RewardSignal, RewardSummary};
use serde_json::{json, Value};

/// Asserts two floats agree to the tolerance the algorithms promise.
fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-9,
        "expected {expected}, got {actual}"
    );
}

/// A node whose name and kind mirror its id.
fn node(id: &str) -> AttributionNode {
    AttributionNode::new(id, id, "step")
}

/// A weight-1 edge between two node ids.
fn edge(from: &str, to: &str) -> AttributionEdge {
    AttributionEdge::new(from, to)
}

/// A per-node reward map.
fn rewards(pairs: &[(&str, f64)]) -> BTreeMap<String, f64> {
    pairs
        .iter()
        .map(|(node_id, value)| ((*node_id).to_owned(), *value))
        .collect()
}

/// `root <- mid <- leaf`.
fn chain_graph() -> AttributionGraph {
    AttributionGraph::new(
        vec![node("root"), node("mid"), node("leaf")],
        vec![edge("root", "mid"), edge("mid", "leaf")],
    )
    .expect("chain graph is valid")
}

/// `root -> {left, right} -> leaf`.
fn diamond_graph() -> AttributionGraph {
    AttributionGraph::new(
        vec![node("root"), node("left"), node("right"), node("leaf")],
        vec![
            edge("root", "left"),
            edge("left", "leaf"),
            edge("right", "leaf"),
            edge("root", "right"),
        ],
    )
    .expect("diamond graph is valid")
}

/// `root -> {left, right}` (two leaves).
fn two_leaf_graph() -> AttributionGraph {
    AttributionGraph::new(
        vec![node("root"), node("left"), node("right")],
        vec![edge("root", "left"), edge("root", "right")],
    )
    .expect("two-leaf graph is valid")
}

/// A summary carrying just a total.
fn summary(total: f64) -> RewardSummary {
    RewardSummary {
        total,
        ..RewardSummary::zero()
    }
}

// ---------------------------------------------------------------------------
// graph
// ---------------------------------------------------------------------------

#[test]
fn graph_rejects_duplicate_nodes_and_edges() {
    let error = AttributionGraph::new(vec![node("a"), node("a")], Vec::new())
        .expect_err("duplicate node id");
    assert!(error.to_string().contains("`a`"), "{error}");

    let error = AttributionGraph::new(
        vec![node("a"), node("b")],
        vec![edge("a", "b"), edge("a", "b")],
    )
    .expect_err("duplicate edge");
    assert!(error.to_string().contains("a -> b"), "{error}");

    let mut graph = chain_graph();
    assert!(graph.add_node(node("leaf")).is_err());
    assert!(graph.add_edge(edge("root", "mid")).is_err());
}

#[test]
fn graph_rejects_unknown_endpoints_and_bad_weights() {
    let error =
        AttributionGraph::new(vec![node("a")], vec![edge("a", "b")]).expect_err("unknown consumer");
    assert!(error.to_string().contains("`b`"), "{error}");
    let error =
        AttributionGraph::new(vec![node("a")], vec![edge("z", "a")]).expect_err("unknown producer");
    assert!(error.to_string().contains("`z`"), "{error}");

    assert!(edge("a", "b").with_weight(-0.1).validate().is_err());
    assert!(edge("a", "b").with_weight(f64::NAN).validate().is_err());
    assert!(AttributionGraph::new(
        vec![node("a"), node("b")],
        vec![edge("a", "b").with_weight(f64::INFINITY)]
    )
    .is_err());
    assert!(AttributionGraph::new(
        vec![node("a"), node("b")],
        vec![edge("a", "b").with_weight(0.0)]
    )
    .is_ok());

    let mut graph = AttributionGraph::new(vec![node("a")], Vec::new()).unwrap();
    assert!(graph.add_edge(edge("a", "b")).is_err());
}

#[test]
fn graph_relations_preserve_insertion_order() {
    let graph = AttributionGraph::new(
        vec![node("root"), node("left"), node("right"), node("leaf")],
        vec![
            edge("root", "right"),
            edge("root", "left"),
            edge("left", "leaf"),
        ],
    )
    .unwrap();
    assert_eq!(graph.children("root"), ["right", "left"]);
    assert_eq!(graph.parents("leaf"), ["left"]);
    assert_eq!(graph.parents("root").len(), 0);
    assert_eq!(graph.roots(), ["root"]);
    assert_eq!(graph.leaves(), ["right", "leaf"]);
    assert_eq!(
        graph.node_ids().collect::<Vec<_>>(),
        ["root", "left", "right", "leaf"]
    );
    assert_eq!(graph.len(), 4);
    assert_ne!(graph.len(), 0);
    assert!(graph.contains("leaf"));
    assert!(graph.get("left").is_some());
    assert!(graph.get("nope").is_none());
}

#[test]
fn graph_topological_order_prefers_earliest_inserted() {
    assert_eq!(
        diamond_graph().topological_order().unwrap(),
        ["root", "left", "right", "leaf"]
    );
    let graph = AttributionGraph::new(
        vec![node("s"), node("a"), node("b")],
        vec![edge("s", "b"), edge("s", "a")],
    )
    .unwrap();
    assert_eq!(graph.topological_order().unwrap(), ["s", "a", "b"]);
    assert_eq!(
        AttributionGraph::default()
            .topological_order()
            .unwrap()
            .len(),
        0
    );
}

#[test]
fn graph_topological_order_reports_cycle_members() {
    let graph = AttributionGraph::new(
        vec![node("a"), node("b"), node("c")],
        vec![edge("a", "b"), edge("b", "a"), edge("b", "c")],
    )
    .unwrap();
    let message = graph.topological_order().expect_err("cycle").to_string();
    assert!(message.contains("cycle"), "{message}");
    assert!(message.contains("a, b"), "{message}");
    assert!(!message.contains(", c"), "{message}");

    let looped = AttributionGraph::new(vec![node("a")], vec![edge("a", "a")]).unwrap();
    let message = looped
        .topological_order()
        .expect_err("self loop")
        .to_string();
    assert!(message.contains("through nodes: a"), "{message}");
}

#[test]
fn graph_closure_and_ancestors_handle_diamonds_and_cycles() {
    let diamond = diamond_graph();
    assert_eq!(
        diamond.closure(&["root"]).into_iter().collect::<Vec<_>>(),
        ["leaf", "left", "right", "root"]
    );
    assert_eq!(
        diamond.ancestors(&["leaf"]).into_iter().collect::<Vec<_>>(),
        ["leaf", "left", "right", "root"]
    );
    assert_eq!(diamond.closure(&["missing"]).len(), 0);

    let cycle = AttributionGraph::new(
        vec![node("a"), node("b"), node("c")],
        vec![edge("a", "b"), edge("b", "c"), edge("c", "a")],
    )
    .unwrap();
    assert_eq!(cycle.closure(&["a"]).len(), 3);
    assert_eq!(cycle.ancestors(&["a"]).len(), 3);
}

#[test]
fn graph_subgraph_keeps_only_induced_edges() {
    let sub = diamond_graph().subgraph(&["leaf", "left", "unknown"]);
    assert_eq!(sub.node_ids().collect::<Vec<_>>(), ["left", "leaf"]);
    assert_eq!(sub.len(), 2);
    assert_eq!(sub.edges.len(), 1);
    assert_eq!(sub.edges[0].from, "left");
    assert_eq!(sub.edges[0].to, "leaf");
    assert_eq!(sub.topological_order().unwrap(), ["left", "leaf"]);
}

#[test]
fn graph_serde_round_trips_and_defaults() {
    let graph = diamond_graph();
    let value = serde_json::to_value(&graph).unwrap();
    let restored: AttributionGraph = serde_json::from_value(value).unwrap();
    assert_eq!(restored, graph);

    let restored: AttributionGraph = serde_json::from_str(
        r#"{
            "nodes": [{"node_id": "a", "name": "A", "kind": "tool", "extra": 1}],
            "edges": [{"from": "a", "to": "a"}]
        }"#,
    )
    .unwrap();
    assert_close(restored.edges[0].weight, 1.0);
    assert_eq!(restored.nodes[0].metadata.len(), 0);
    assert_eq!(
        serde_json::to_value(&restored.nodes[0]).unwrap()["name"],
        json!("A")
    );
}

// ---------------------------------------------------------------------------
// credit
// ---------------------------------------------------------------------------

#[test]
fn credit_chain_decays_by_half_per_hop() {
    let report = CreditAssigner::new()
        .with_normalise(false)
        .assign(&chain_graph(), &rewards(&[("leaf", 1.0)]))
        .unwrap();
    assert_close(report.credit("leaf"), 1.0);
    assert_close(report.credit("mid"), 0.5);
    assert_close(report.credit("root"), 0.25);
    assert_close(report.total, 1.0);
    assert!(!report.normalised);
    assert_eq!(
        report
            .ranking
            .iter()
            .map(|rank| rank.node_id.as_str())
            .collect::<Vec<_>>(),
        ["leaf", "mid", "root"]
    );
    assert_close(report.ranking[0].share, 1.0);
    assert_close(report.ranking[1].share, 0.5);
    assert_close(report.ranking[2].share, 0.25);
    assert_eq!(report.top(2).len(), 2);
    assert_eq!(report.top(9).len(), 3);
    assert_close(report.credit("absent"), 0.0);
}

#[test]
fn credit_scales_by_edge_weights_and_zero_blocks() {
    let weighted = AttributionGraph::new(
        vec![node("root"), node("mid"), node("leaf")],
        vec![
            edge("root", "mid").with_weight(0.5),
            edge("mid", "leaf").with_weight(2.0),
        ],
    )
    .unwrap();
    let report = CreditAssigner::new()
        .with_normalise(false)
        .assign(&weighted, &rewards(&[("leaf", 1.0)]))
        .unwrap();
    assert_close(report.credit("mid"), 1.0);
    assert_close(report.credit("root"), 0.125);

    let blocked = AttributionGraph::new(
        vec![node("root"), node("leaf")],
        vec![edge("root", "leaf").with_weight(0.0)],
    )
    .unwrap();
    let report = CreditAssigner::new()
        .with_normalise(false)
        .assign(&blocked, &rewards(&[("leaf", 1.0)]))
        .unwrap();
    assert_close(report.credit("leaf"), 1.0);
    assert_close(report.credit("root"), 0.0);
    assert!(!report.per_node.contains_key("root"));
}

#[test]
fn credit_honours_the_depth_cutoff() {
    let depth_one = CreditAssigner::new()
        .with_normalise(false)
        .with_max_depth(1)
        .assign(&chain_graph(), &rewards(&[("leaf", 1.0)]))
        .unwrap();
    assert_close(depth_one.credit("mid"), 0.5);
    assert_close(depth_one.credit("root"), 0.0);

    let only_seeds = CreditAssigner::new()
        .with_normalise(false)
        .with_max_depth(0)
        .assign(&chain_graph(), &rewards(&[("leaf", 1.0)]))
        .unwrap();
    assert_eq!(only_seeds.per_node.len(), 1);
    assert_close(only_seeds.credit("mid"), 0.0);
}

#[test]
fn credit_normalisation_sums_to_the_input_total() {
    let report = CreditAssigner::new()
        .assign(&chain_graph(), &rewards(&[("leaf", 1.0)]))
        .unwrap();
    assert!(report.normalised);
    let sum: f64 = report.per_node.values().sum();
    assert_close(sum, 1.0);
    assert_close(report.credit("leaf"), 4.0 / 7.0);
    assert_close(report.credit("mid"), 2.0 / 7.0);
    assert_close(report.credit("root"), 1.0 / 7.0);

    let zero = CreditAssigner::new()
        .assign(&chain_graph(), &rewards(&[("leaf", 0.0)]))
        .unwrap();
    assert!(!zero.normalised);
    assert_close(zero.credit("leaf"), 0.0);
    assert!(zero.ranking.iter().all(|rank| rank.share.abs() < 1e-9));
}

#[test]
fn credit_ranking_breaks_ties_by_node_id() {
    let graph = AttributionGraph::new(vec![node("b"), node("a")], Vec::new()).unwrap();
    let report = CreditAssigner::new()
        .with_normalise(false)
        .assign(&graph, &rewards(&[("a", 1.0), ("b", 1.0)]))
        .unwrap();
    assert_eq!(
        report
            .ranking
            .iter()
            .map(|rank| rank.node_id.as_str())
            .collect::<Vec<_>>(),
        ["a", "b"]
    );
    assert_close(report.total, 2.0);
    assert_close(report.ranking[0].share, 0.5);
    assert_close(report.ranking[1].share, 0.5);
}

#[test]
fn credit_accumulates_across_rewarded_nodes() {
    let report = CreditAssigner::new()
        .with_normalise(false)
        .assign(&diamond_graph(), &rewards(&[("leaf", 1.0), ("left", 0.5)]))
        .unwrap();
    assert_close(report.credit("leaf"), 1.0);
    assert_close(report.credit("left"), 1.0);
    assert_close(report.credit("right"), 0.5);
    assert_close(report.credit("root"), 0.5);
    assert_close(report.total, 1.5);
    assert_close(report.credit("root") / report.total, 1.0 / 3.0);
    assert_eq!(
        report
            .ranking
            .iter()
            .map(|rank| rank.node_id.as_str())
            .collect::<Vec<_>>(),
        ["leaf", "left", "right", "root"]
    );
}

#[test]
fn credit_validates_rewards_and_configuration() {
    let assigner = CreditAssigner::new();
    let error = assigner
        .assign(&chain_graph(), &rewards(&[("nope", 1.0)]))
        .expect_err("unknown node");
    assert!(error.to_string().contains("nope"), "{error}");
    assert!(assigner
        .assign(&chain_graph(), &rewards(&[("leaf", f64::NAN)]))
        .is_err());
    assert!(assigner
        .assign(&chain_graph(), &rewards(&[("leaf", f64::INFINITY)]))
        .is_err());

    assert!(CreditAssigner::new().with_decay(-0.1).is_err());
    assert!(CreditAssigner::new().with_decay(1.1).is_err());
    assert!(CreditAssigner::new().with_decay(f64::NAN).is_err());
    assert!(CreditAssigner::new().with_decay(0.0).is_ok());

    let configured = CreditAssigner::new()
        .with_decay(0.25)
        .unwrap()
        .with_normalise(false)
        .with_max_depth(3);
    assert_close(configured.decay(), 0.25);
    assert!(!configured.normalise());
    assert_eq!(configured.max_depth(), 3);

    let empty = configured.assign(&chain_graph(), &BTreeMap::new()).unwrap();
    assert_eq!(empty.per_node.len(), 0);
    assert_eq!(empty.ranking.len(), 0);
    assert_close(empty.total, 0.0);
}

// ---------------------------------------------------------------------------
// keyword
// ---------------------------------------------------------------------------

#[test]
fn keyword_positions_are_utf8_byte_offsets() {
    let attributor = KeywordAttributor::new(vec![
        KeywordRule::new(vec!["报错".to_owned()], "tool"),
        KeywordRule::new(vec!["内存".to_owned()], "memory"),
    ]);
    let text = "工具 报错: 内存不足";
    let matches = attributor.attribute(text);
    assert_eq!(matches.len(), 2);
    assert_eq!(matches[0].target, "tool");
    assert_eq!(matches[0].keyword, "报错");
    assert_eq!(matches[0].position, 7);
    assert_eq!(matches[1].target, "memory");
    assert_eq!(matches[1].position, 15);
    assert_eq!(&text[7..13], "报错");
    assert_eq!(&text[15..21], "内存");

    let accented = KeywordAttributor::new(vec![
        KeywordRule::new(vec!["échec".to_owned()], "tool"),
        KeywordRule::new(vec!["command".to_owned()], "script"),
    ]);
    let matches = accented.attribute("Échec du COMMAND");
    assert_eq!(matches.len(), 2);
    assert_eq!(matches[0].position, 0);
    assert_eq!(matches[1].position, 10);
}

#[test]
fn keyword_case_sensitivity_is_per_rule() {
    let text = "Run the COMMAND now";
    let insensitive =
        KeywordAttributor::new(vec![KeywordRule::new(vec!["command".to_owned()], "script")]);
    assert_eq!(insensitive.attribute(text).len(), 1);
    assert_eq!(insensitive.attribute(text)[0].position, 8);

    let sensitive =
        KeywordAttributor::new(vec![
            KeywordRule::new(vec!["command".to_owned()], "script").with_case_sensitive(true)
        ]);
    assert_eq!(sensitive.attribute(text).len(), 0);
    assert_eq!(sensitive.attribute("command")[0].position, 0);
}

#[test]
fn keyword_orders_matches_and_picks_the_best() {
    let attributor = KeywordAttributor::new(vec![
        KeywordRule::new(vec!["error".to_owned()], "tool").with_weight(2.0),
        KeywordRule::new(vec!["error".to_owned(), "memory".to_owned()], "alpha").with_weight(2.0),
        KeywordRule::new(vec!["memory".to_owned()], "memory").with_weight(1.0),
    ]);
    let rows: Vec<(String, String, usize)> = attributor
        .attribute("error memory")
        .into_iter()
        .map(|hit| (hit.target, hit.keyword, hit.position))
        .collect();
    let expected = [
        ("alpha".to_owned(), "error".to_owned(), 0),
        ("tool".to_owned(), "error".to_owned(), 0),
        ("alpha".to_owned(), "memory".to_owned(), 6),
        ("memory".to_owned(), "memory".to_owned(), 6),
    ];
    assert_eq!(rows, expected);

    let best = attributor.best_match("error memory").unwrap();
    assert_eq!(best.target, "alpha");
    assert_eq!(best.position, 0);

    let heavier = KeywordAttributor::new(vec![
        KeywordRule::new(vec!["error".to_owned()], "tool").with_weight(1.0),
        KeywordRule::new(vec!["failure".to_owned()], "tool").with_weight(3.0),
    ]);
    assert_eq!(
        heavier.best_match("error then failure").unwrap().keyword,
        "failure"
    );

    let repeated = KeywordAttributor::new(vec![KeywordRule::new(vec!["aa".to_owned()], "tool")]);
    let positions: Vec<usize> = repeated
        .attribute("aa aa")
        .into_iter()
        .map(|hit| hit.position)
        .collect();
    assert_eq!(positions, [0, 3]);
}

#[test]
fn keyword_default_rules_and_empty_input() {
    let attributor = KeywordAttributor::default_rules();
    let targets: Vec<&str> = attributor
        .rules()
        .iter()
        .map(|rule| rule.target.as_str())
        .collect();
    assert_eq!(
        targets,
        ["tool", "memory", "instruction", "script", "environment"]
    );
    for rule in attributor.rules() {
        let target = rule.target.as_str();
        assert!(rule.validate().is_ok(), "rule {target} is invalid");
        assert_close(rule.weight, 1.0);
        assert!(!rule.case_sensitive);
    }
    let best = attributor
        .best_match("Command 失败 because dependency")
        .unwrap();
    assert_eq!(best.target, "script");
    assert_eq!(best.position, 0);

    let empty = KeywordAttributor::new(Vec::new());
    assert_eq!(empty.attribute("error").len(), 0);
    assert!(empty.best_match("error").is_none());
    assert_eq!(attributor.attribute("").len(), 0);

    let built = KeywordAttributor::new(Vec::new())
        .with_rule(KeywordRule::new(vec!["x".to_owned()], "tool"));
    assert_eq!(built.rules().len(), 1);
    let restored: KeywordRule =
        serde_json::from_str(r#"{"keywords": ["x"], "target": "tool"}"#).unwrap();
    assert_close(restored.weight, 1.0);
    assert!(!restored.case_sensitive);
}

// ---------------------------------------------------------------------------
// reward stage
// ---------------------------------------------------------------------------

#[test]
fn reward_stage_splits_the_total_over_leaves() {
    let summary = RewardPack::default_pack()
        .aggregate(&[RewardSignal::new("correctness", 0.8)])
        .unwrap();
    assert_close(summary.total(), 0.8);

    let adapter = RewardStageAdapter::new(CreditAssigner::new()).with_tier("fair");
    let signals = adapter
        .signals_from_summary(&two_leaf_graph(), &summary)
        .unwrap();
    assert_eq!(signals.len(), 2);
    assert_eq!(signals[0].signal_id, "reward-left");
    assert_eq!(signals[1].signal_id, "reward-right");
    for signal in &signals {
        assert_eq!(signal.source, SignalSource::Reward);
        assert_eq!(signal.tier, "fair");
        assert_close(signal.score.unwrap(), 0.4);
        assert_close(signal.confidence.unwrap(), 1.0);
        let text = signal.text.as_str();
        assert!(text.contains("0.800"), "{text}");
        assert_close(signal.metadata["total"].as_f64().unwrap(), 0.8);
        assert_eq!(signal.metadata["num_nodes"], json!(2));
        assert_eq!(signal.node_ids.len(), 1);
    }
    assert_eq!(adapter.tier(), "fair");
    assert_close(adapter.assigner().decay(), 0.5);
}

#[test]
fn reward_stage_round_trips_and_handles_edge_cases() {
    let adapter = RewardStageAdapter::new(CreditAssigner::new());
    let signals = adapter
        .signals_from_summary(&two_leaf_graph(), &summary(0.9))
        .unwrap();
    let round_tripped = adapter.rewards_from_signals(&signals).unwrap();
    assert_eq!(round_tripped.len(), 2);
    assert_close(round_tripped["left"], 0.45);
    assert_close(round_tripped["right"], 0.45);
    assert_close(round_tripped.values().sum::<f64>(), 0.9);

    let empty = AttributionGraph::default();
    assert_eq!(
        adapter
            .signals_from_summary(&empty, &summary(0.5))
            .unwrap()
            .len(),
        0
    );
    assert_eq!(adapter.rewards_from_signals(&[]).unwrap().len(), 0);
    assert!(adapter
        .signals_from_summary(&two_leaf_graph(), &summary(f64::NAN))
        .is_err());
}

#[test]
fn reward_stage_falls_back_to_every_node_without_leaves() {
    let cycle = AttributionGraph::new(
        vec![node("a"), node("b")],
        vec![edge("a", "b"), edge("b", "a")],
    )
    .unwrap();
    let adapter = RewardStageAdapter::new(CreditAssigner::new());
    let signals = adapter.signals_from_summary(&cycle, &summary(0.4)).unwrap();
    assert_eq!(signals.len(), 2);
    assert_eq!(signals[0].node_ids, ["a".to_owned()]);
    assert_close(signals[0].score.unwrap(), 0.2);
}

#[test]
fn reward_stage_weights_by_confidence_and_skips_unscored() {
    let adapter = RewardStageAdapter::new(CreditAssigner::new());
    let scored = |text: &str, score: f64, confidence: f64| {
        DiagnosticSignal::new(SignalSource::Evaluator, text)
            .unwrap()
            .with_node_ids(vec!["n".to_owned()])
            .with_score(score)
            .with_confidence(confidence)
    };
    let rewards = adapter
        .rewards_from_signals(&[scored("a", 1.0, 0.25), scored("b", 0.0, 0.75)])
        .unwrap();
    assert_close(rewards["n"], 0.25);

    let first = scored("c", 0.3, 0.0);
    let second = scored("d", 0.7, 0.0);
    let zero_confidence = adapter.rewards_from_signals(&[first, second]).unwrap();
    assert_close(zero_confidence["n"], 0.3);

    let unscored = DiagnosticSignal::new(SignalSource::Evaluator, "e")
        .unwrap()
        .with_node_ids(vec!["n".to_owned()]);
    let anchorless = DiagnosticSignal::new(SignalSource::Evaluator, "f")
        .unwrap()
        .with_score(0.9);
    assert!(adapter
        .rewards_from_signals(&[unscored, anchorless])
        .unwrap()
        .is_empty());

    let mut broken = scored("g", 1.0, 1.0);
    broken.score = Some(f64::NAN);
    assert!(adapter.rewards_from_signals(&[broken]).is_err());
}

#[test]
fn reward_stage_end_to_end_credit_reaches_the_root() {
    let graph = chain_graph();
    let adapter = RewardStageAdapter::new(CreditAssigner::new().with_normalise(false));
    let signals = adapter.signals_from_summary(&graph, &summary(0.8)).unwrap();
    let rewards = adapter.rewards_from_signals(&signals).unwrap();
    assert_close(rewards["leaf"], 0.8);

    let report = adapter.assigner().assign(&graph, &rewards).unwrap();
    assert_close(report.credit("leaf"), 0.8);
    assert_close(report.credit("mid"), 0.4);
    assert_close(report.credit("root"), 0.2);
    assert_close(report.total, 0.8);
}

// ---------------------------------------------------------------------------
// diagnostic signals
// ---------------------------------------------------------------------------

#[test]
fn signal_ids_are_content_stable_and_source_dependent() {
    let first = DiagnosticSignal::new(SignalSource::Reward, "score dropped").unwrap();
    let same = DiagnosticSignal::new(SignalSource::Reward, "score dropped").unwrap();
    assert_eq!(first.signal_id, same.signal_id);
    assert_eq!(first.signal_id.len(), "diag-".len() + 16);
    assert!(first.signal_id.starts_with("diag-"));

    let other_source = DiagnosticSignal::new(SignalSource::System, "score dropped").unwrap();
    assert_ne!(first.signal_id, other_source.signal_id);
    let other_text = DiagnosticSignal::new(SignalSource::Reward, "score rose").unwrap();
    assert_ne!(first.signal_id, other_text.signal_id);
    assert!(DiagnosticSignal::new(SignalSource::Reward, "  ").is_err());
}

#[test]
fn signal_validation_and_serde_round_trip() {
    let signal = DiagnosticSignal::new(SignalSource::UserFeedback, "please fix it")
        .unwrap()
        .with_tier("weak")
        .with_node_ids(vec!["step-1".to_owned()])
        .with_score(0.25)
        .with_confidence(0.5)
        .with_message_id("msg-1")
        .with_metadata(BTreeMap::from([("k".to_owned(), Value::from(1))]));
    assert!(signal.validate().is_ok());
    let value = serde_json::to_value(&signal).unwrap();
    assert_eq!(value["source"], "user_feedback");
    let restored: DiagnosticSignal = serde_json::from_value(value).unwrap();
    assert_eq!(restored, signal);

    let mut broken = signal.clone();
    broken.confidence = Some(1.5);
    assert!(broken.validate().is_err());
    broken.confidence = Some(1.0);
    broken.score = Some(f64::INFINITY);
    assert!(broken.validate().is_err());
    broken.score = Some(1.0);
    broken.node_ids = vec![String::new()];
    assert!(broken.validate().is_err());
    broken.node_ids = Vec::new();
    broken.signal_id.clear();
    assert!(broken.validate().is_err());

    let sparse: DiagnosticSignal =
        serde_json::from_str(r#"{"signal_id": "diag-1", "text": "t", "bogus": true}"#).unwrap();
    assert_eq!(sparse.source, SignalSource::Other);
    assert_eq!(sparse.node_ids.len(), 0);
    assert_eq!(sparse.metadata.len(), 0);
}

#[test]
fn signal_sources_tiers_and_display() {
    assert_eq!(SignalSource::UserFeedback.to_string(), "user_feedback");
    assert_eq!(SignalSource::ReviewFeedback.to_string(), "review_feedback");
    assert_eq!(SignalSource::Reward.to_string(), "reward");
    assert_eq!(SignalSource::default(), SignalSource::Other);
    assert_eq!(
        serde_json::to_value(SignalSource::System).unwrap(),
        json!("system")
    );

    assert_eq!(SIGNAL_TIERS, ["perfect", "fair", "weak", "uncertain"]);
    assert_eq!(reward_tier(1.0), "perfect");
    assert_eq!(reward_tier(0.9), "perfect");
    assert_eq!(reward_tier(0.89), "fair");
    assert_eq!(reward_tier(0.75), "fair");
    assert_eq!(reward_tier(0.74), "weak");
    assert_eq!(reward_tier(0.0), "weak");
}
