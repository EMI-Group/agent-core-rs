//! Behaviour tests for the bounded deterministic planner: start selection, expansion,
//! scoring arithmetic, ranking, status, and language helpers.
//!
//! Every test is offline and deterministic. The only non-reproducible plan field is
//! [`OrchestrationPlan::plan_id`], a fresh `uuid_hex` per call (Python parity), so the
//! determinism test compares the whole plan with that one field removed.
#![allow(clippy::unwrap_used, clippy::expect_used)]
#![allow(clippy::float_cmp)]
#![allow(clippy::assert_is_empty)]

use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_symphony::error::{ErrorKind, Result};
use openjiuwen_symphony::graph::{CapabilityEdge, CapabilityGraph, CAN_FEED};
use openjiuwen_symphony::model::{CapabilityDescriptor, CapabilityIo};
use openjiuwen_symphony::orchestration::plan::PlanStatus;
use openjiuwen_symphony::orchestration::planner::{
    normalize_language, plan_title, LexicalSemanticScorer, PlanConfig, PlanMode, PlanRequest,
    Planner, SemanticScorer,
};
use serde_json::json;

/// Asserts that two plan scores match within the formula's floating-point tolerance.
fn assert_score(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-9,
        "score {actual} does not match expected {expected}"
    );
}

/// Builds an input with `required` left unspecified (`None` means required).
fn io(name: &str, io_type: &str) -> CapabilityIo {
    CapabilityIo::new(name, io_type)
}

/// Builds a required input.
fn required_io(name: &str, io_type: &str) -> CapabilityIo {
    CapabilityIo::new(name, io_type).with_required(true)
}

/// Builds an optional input.
fn optional_io(name: &str, io_type: &str) -> CapabilityIo {
    CapabilityIo::new(name, io_type).with_required(false)
}

/// Builds a `tool` capability with the given inputs and outputs.
fn capability(
    id: &str,
    name: &str,
    inputs: Vec<CapabilityIo>,
    outputs: Vec<CapabilityIo>,
) -> CapabilityDescriptor {
    CapabilityDescriptor::new("tool", id, name)
        .with_inputs(inputs)
        .with_outputs(outputs)
}

/// Builds a capability graph from the catalog plus `(source, target, weight)` edges.
fn graph_for(
    capabilities: &[CapabilityDescriptor],
    edges: &[(&str, &str, f64)],
) -> CapabilityGraph {
    let mut graph = CapabilityGraph::from_descriptors("planner-test", capabilities);
    for (source, target, weight) in edges {
        graph
            .add_edge(CapabilityEdge::new(*source, *target, CAN_FEED, *weight))
            .expect("edge endpoints exist");
    }
    graph
}

/// A scorer with fixed per-capability values, so the arithmetic is hand-checkable.
struct FixedScorer {
    /// Qualified id → score; unknown capabilities score `0.0`.
    scores: BTreeMap<String, f64>,
}

impl FixedScorer {
    /// Builds a scorer from `(qualified id, score)` pairs.
    fn new(pairs: &[(&str, f64)]) -> Self {
        Self {
            scores: pairs
                .iter()
                .map(|(id, score)| ((*id).to_string(), *score))
                .collect(),
        }
    }
}

#[async_trait]
impl SemanticScorer for FixedScorer {
    async fn score(
        &self,
        _query: &str,
        capabilities: &[CapabilityDescriptor],
    ) -> Result<BTreeMap<String, f64>> {
        Ok(capabilities
            .iter()
            .map(|capability| {
                let id = capability.qualified_id();
                let score = self.scores.get(&id).copied().unwrap_or(0.0);
                (id, score)
            })
            .collect())
    }
}

/// The chain catalog: `alpha -> beta -> gamma`, where only `alpha` is available.
///
/// `beta` and `gamma` are withdrawn so they are not start capabilities, but the planner
/// may still reach them downstream (availability gates the start set only).
fn chain_catalog() -> Vec<CapabilityDescriptor> {
    vec![
        capability("alpha", "Alpha", Vec::new(), vec![io("value", "string")]),
        capability(
            "beta",
            "Beta",
            vec![required_io("value", "string")],
            vec![io("result", "integer")],
        )
        .with_available(false),
        capability(
            "gamma",
            "Gamma",
            vec![required_io("result", "number")],
            Vec::new(),
        )
        .with_available(false),
    ]
}

/// The companion graph for [`chain_catalog`].
fn chain_graph(capabilities: &[CapabilityDescriptor]) -> CapabilityGraph {
    graph_for(
        capabilities,
        &[
            ("tool:alpha", "tool:beta", 0.9),
            ("tool:beta", "tool:gamma", 0.8),
        ],
    )
}

/// The fixed scorer for [`chain_catalog`]: `alpha` is weak, the downstream nodes strong.
fn chain_planner(config: PlanConfig) -> Planner {
    Planner::with_scorer(
        config,
        Arc::new(FixedScorer::new(&[
            ("tool:alpha", 0.2),
            ("tool:beta", 1.0),
            ("tool:gamma", 1.0),
        ])),
    )
}

#[test]
fn config_and_request_defaults_match_the_contract() {
    assert_eq!(
        PlanConfig::default(),
        PlanConfig {
            mode: PlanMode::Fast,
            top_k: 3,
            max_depth: 4,
            min_edge_confidence: 0.7,
            max_expansions: 128,
            max_plans: 256,
            dynamic_graph_enabled: false,
        }
    );
    assert_eq!(PlanMode::default(), PlanMode::Fast);

    let request = PlanRequest::default();
    assert_eq!(request.query, "");
    assert_eq!(request.language, "cn");
    assert!(request.candidate_ids.is_none());
    assert!(request.disabled_ids.is_empty());
    assert!(request.provided_inputs.is_empty());
    assert!(request.top_k.is_none());
    assert!(request.max_depth.is_none());

    let built = PlanRequest::new("find docs");
    assert_eq!(built.query, "find docs");
    assert_eq!(built.language, "cn");

    assert_eq!(serde_json::to_value(PlanMode::Beam).unwrap(), json!("beam"));
    assert_eq!(serde_json::to_value(PlanMode::Fast).unwrap(), json!("fast"));
}

#[test]
fn normalize_language_covers_the_documented_tags() {
    for chinese in ["zh", "cn", "zh_cn", "zh-cn", "ZH", "  Cn  ", "zh-"] {
        assert_eq!(normalize_language(chinese), "cn", "tag `{chinese}`");
    }
    for english in ["en", "en_us", "en-US", " EN ", "En_Us"] {
        assert_eq!(normalize_language(english), "en", "tag `{english}`");
    }
    for unknown in ["", "fr", "de_de", "english"] {
        assert_eq!(normalize_language(unknown), "cn", "tag `{unknown}`");
    }
}

#[test]
fn plan_title_is_localized() {
    assert_eq!(plan_title("find docs", "cn"), "计划：find docs");
    assert_eq!(plan_title("find docs", "en"), "Plan: find docs");
    assert_eq!(plan_title("find docs", "en-US"), "Plan: find docs");
    assert_eq!(plan_title("find docs", "zh"), "计划：find docs");
}

#[tokio::test]
async fn lexical_scorer_measures_query_term_coverage() {
    let capabilities = vec![
        capability("docs", "Search Documentation", Vec::new(), Vec::new()),
        capability("sums", "Summarize Reports", Vec::new(), Vec::new()),
    ];
    let scorer = LexicalSemanticScorer::new();

    let partial = scorer
        .score("search docs", &capabilities)
        .await
        .expect("lexical scoring is infallible");
    assert_score(partial["tool:docs"], 0.5);
    assert_score(partial["tool:sums"], 0.0);

    let full = scorer
        .score("summarize reports", &capabilities)
        .await
        .unwrap();
    assert_score(full["tool:docs"], 0.0);
    assert_score(full["tool:sums"], 1.0);

    let tokenless = scorer.score("   ", &capabilities).await.unwrap();
    assert_score(tokenless["tool:docs"], 1.0);
    assert_score(tokenless["tool:sums"], 1.0);

    assert_eq!(
        LexicalSemanticScorer::new()
            .score("", &[])
            .await
            .unwrap()
            .len(),
        0
    );
}

#[tokio::test]
async fn blank_query_is_a_validation_error() {
    let planner = Planner::new(PlanConfig::default());
    let error = planner
        .plan(&PlanRequest::new("   "), &CapabilityGraph::new("g"), &[])
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains("query"));
}

#[tokio::test]
async fn empty_catalog_returns_no_plan() {
    let plan = Planner::new(PlanConfig::default())
        .plan(
            &PlanRequest::new("find docs"),
            &CapabilityGraph::new("g"),
            &[],
        )
        .await
        .unwrap();
    assert_eq!(plan.status, PlanStatus::NoPlan);
    assert!(plan.is_empty());
    assert_score(plan.score, 0.0);
    assert!(plan.message.is_none());
    assert_eq!(plan.language, "cn");
    assert_eq!(plan.title, "计划：find docs");
    assert_eq!(plan.plan_id.len(), 32);
}

#[tokio::test]
async fn query_without_positive_score_returns_no_plan() {
    let capabilities = vec![capability(
        "docs",
        "Search Documentation",
        Vec::new(),
        Vec::new(),
    )];
    let graph = graph_for(&capabilities, &[]);
    let plan = Planner::new(PlanConfig::default())
        .plan(&PlanRequest::new("unrelated words"), &graph, &capabilities)
        .await
        .unwrap();
    assert_eq!(plan.status, PlanStatus::NoPlan);
    assert!(plan.is_empty());
}

#[tokio::test]
async fn single_node_plan_is_scored_with_the_formula() {
    let capabilities = vec![capability(
        "solo",
        "Search Documentation",
        Vec::new(),
        vec![io("docs", "array")],
    )];
    let graph = graph_for(&capabilities, &[]);
    let mut request = PlanRequest::new("search");
    request.language = "en".to_string();

    let plan = Planner::new(PlanConfig::default())
        .plan(&request, &graph, &capabilities)
        .await
        .unwrap();

    assert_eq!(plan.status, PlanStatus::Ready);
    assert_eq!(plan.language, "en");
    assert_eq!(plan.title, "Plan: search");
    assert_eq!(plan.steps.len(), 1);
    assert_eq!(plan.steps[0].step_id, "tool:solo");
    assert_eq!(plan.steps[0].capability_type, "tool");
    assert_eq!(plan.steps[0].name, "Search Documentation");
    assert_eq!(plan.steps[0].score, 1.0);
    assert!(plan.steps[0].missing_inputs.is_empty());
    assert!(plan.edges.is_empty());
    assert!(plan.message.is_none());
    // 0.45 * 1.0 + 0.25 * 1.0 (no edges) + 0.20 * 1.0 (no inputs) - 0.03 * 0
    assert_score(plan.score, 0.90);
}

#[tokio::test]
async fn multi_step_chain_is_selected_and_scored() {
    let capabilities = chain_catalog();
    let graph = chain_graph(&capabilities);
    let plan = chain_planner(PlanConfig::default())
        .plan(&PlanRequest::new("chain"), &graph, &capabilities)
        .await
        .unwrap();

    let ids: Vec<&str> = plan
        .steps
        .iter()
        .map(|step| step.step_id.as_str())
        .collect();
    assert_eq!(ids, ["tool:alpha", "tool:beta", "tool:gamma"]);
    assert_eq!(plan.status, PlanStatus::Ready);
    assert!(plan.message.is_none());

    let edges: Vec<(&str, &str)> = plan
        .edges
        .iter()
        .map(|edge| (edge.source.as_str(), edge.target.as_str()))
        .collect();
    assert_eq!(
        edges,
        [("tool:alpha", "tool:beta"), ("tool:beta", "tool:gamma")]
    );

    // Inputs match across the chain, including the integer -> number widening on gamma.
    assert_eq!(plan.steps[1].filled_inputs, ["value"]);
    assert_eq!(plan.steps[2].filled_inputs, ["result"]);
    assert!(plan.steps[2].missing_inputs.is_empty());
    assert_eq!(plan.steps[0].score, 0.2);
    assert_eq!(plan.steps[1].score, 1.0);

    // 0.45 * mean(0.2, 1.0, 1.0)
    // + 0.25 * mean(0.9, 0.8)
    // + 0.20 * 1.0 (both inputs covered)
    // - 0.03 * 2
    let expected = 0.45 * ((0.2 + 1.0 + 1.0) / 3.0) + 0.25 * f64::midpoint(0.9, 0.8) + 0.20 - 0.06;
    assert_score(plan.score, expected);
    assert_score(plan.score, 0.6825);
}

#[tokio::test]
async fn max_depth_truncates_the_greedy_path() {
    let capabilities = chain_catalog();
    let graph = chain_graph(&capabilities);
    let planner = chain_planner(PlanConfig::default());

    let mut shallow = PlanRequest::new("chain");
    shallow.max_depth = Some(1);
    let plan = planner.plan(&shallow, &graph, &capabilities).await.unwrap();
    assert_eq!(plan.steps.len(), 1);
    assert_eq!(plan.steps[0].step_id, "tool:alpha");

    let mut medium = PlanRequest::new("chain");
    medium.max_depth = Some(2);
    let plan = planner.plan(&medium, &graph, &capabilities).await.unwrap();
    let ids: Vec<&str> = plan
        .steps
        .iter()
        .map(|step| step.step_id.as_str())
        .collect();
    assert_eq!(ids, ["tool:alpha", "tool:beta"]);
    // 0.45 * mean(0.2, 1.0) + 0.25 * 0.9 + 0.20 * 1.0 - 0.03 * 1
    assert_score(plan.score, 0.45 * 0.6 + 0.225 + 0.20 - 0.03);
}

#[tokio::test]
async fn disabled_ids_block_starts_and_expansion_targets() {
    let capabilities = chain_catalog();
    let graph = chain_graph(&capabilities);
    let planner = chain_planner(PlanConfig::default());

    let mut request = PlanRequest::new("chain");
    request.disabled_ids = vec!["tool:alpha".to_string()];
    let plan = planner.plan(&request, &graph, &capabilities).await.unwrap();
    assert_eq!(plan.status, PlanStatus::NoPlan);

    // Blocking the intermediate node keeps the plan at the available start.
    let mut request = PlanRequest::new("chain");
    request.disabled_ids = vec!["tool:beta".to_string()];
    let plan = planner.plan(&request, &graph, &capabilities).await.unwrap();
    assert_eq!(plan.steps.len(), 1);
    assert_eq!(plan.steps[0].step_id, "tool:alpha");
}

#[tokio::test]
async fn candidate_ids_prune_the_plan() {
    let capabilities = chain_catalog();
    let graph = chain_graph(&capabilities);
    let planner = chain_planner(PlanConfig::default());

    let mut request = PlanRequest::new("chain");
    request.candidate_ids = Some(vec!["tool:alpha".to_string(), "tool:beta".to_string()]);
    let plan = planner.plan(&request, &graph, &capabilities).await.unwrap();
    let ids: Vec<&str> = plan
        .steps
        .iter()
        .map(|step| step.step_id.as_str())
        .collect();
    assert_eq!(ids, ["tool:alpha", "tool:beta"]);

    // A candidate list of withdrawn capabilities has no viable start.
    let mut request = PlanRequest::new("chain");
    request.candidate_ids = Some(vec!["tool:beta".to_string()]);
    let plan = planner.plan(&request, &graph, &capabilities).await.unwrap();
    assert_eq!(plan.status, PlanStatus::NoPlan);
}

#[tokio::test]
async fn provided_inputs_clear_missing_inputs_and_raise_coverage() {
    let capabilities = vec![capability(
        "solo",
        "Search Documentation",
        vec![required_io("prompt", "string")],
        Vec::new(),
    )];
    let graph = graph_for(&capabilities, &[]);
    let planner = Planner::new(PlanConfig::default());

    let mut request = PlanRequest::new("search");
    request.language = "en".to_string();
    let plan = planner.plan(&request, &graph, &capabilities).await.unwrap();
    assert_eq!(plan.status, PlanStatus::NeedsInput);
    assert_eq!(plan.steps[0].missing_inputs, ["prompt"]);
    assert!(plan.steps[0].filled_inputs.is_empty());
    assert_eq!(plan.message.as_deref(), Some("missing inputs: prompt"));
    // Coverage 0: 0.45 * 1.0 + 0.25 * 1.0 + 0.20 * 0.0
    assert_score(plan.score, 0.70);

    let mut request = PlanRequest::new("search");
    request.provided_inputs = vec!["Prompt".to_string()];
    let plan = planner.plan(&request, &graph, &capabilities).await.unwrap();
    assert_eq!(plan.status, PlanStatus::Ready);
    assert!(plan.steps[0].missing_inputs.is_empty());
    assert_eq!(plan.steps[0].filled_inputs, ["prompt"]);
    assert!(plan.message.is_none());
    assert_score(plan.score, 0.90);
}

#[tokio::test]
async fn needs_input_message_lists_sorted_names_in_the_plan_language() {
    let capabilities = vec![capability(
        "solo",
        "Search Documentation",
        vec![
            required_io("Second", "string"),
            required_io("first", "string"),
            optional_io("third", "string"),
        ],
        Vec::new(),
    )];
    let graph = graph_for(&capabilities, &[]);
    let planner = Planner::new(PlanConfig::default());

    let mut request = PlanRequest::new("search");
    request.language = "en".to_string();
    let plan = planner.plan(&request, &graph, &capabilities).await.unwrap();
    assert_eq!(plan.status, PlanStatus::NeedsInput);
    assert_eq!(plan.steps[0].missing_inputs, ["first", "second"]);
    assert_eq!(
        plan.message.as_deref(),
        Some("missing inputs: first, second")
    );

    let mut request = PlanRequest::new("search");
    request.language = "zh".to_string();
    let plan = planner.plan(&request, &graph, &capabilities).await.unwrap();
    assert_eq!(plan.language, "cn");
    assert_eq!(plan.message.as_deref(), Some("缺少输入：first, second"));
}

#[tokio::test]
async fn custom_scorer_replaces_the_lexical_default() {
    let capabilities = vec![capability(
        "alpha",
        "Something Else",
        Vec::new(),
        Vec::new(),
    )];
    let graph = graph_for(&capabilities, &[]);
    let request = PlanRequest::new("unrelated words");

    let default_plan = Planner::new(PlanConfig::default())
        .plan(&request, &graph, &capabilities)
        .await
        .unwrap();
    assert_eq!(default_plan.status, PlanStatus::NoPlan);

    let config = PlanConfig::default();
    let planner = Planner::with_scorer(
        config.clone(),
        Arc::new(FixedScorer::new(&[("tool:alpha", 0.5)])),
    );
    assert_eq!(planner.config(), &config);
    let plan = planner.plan(&request, &graph, &capabilities).await.unwrap();
    assert_eq!(plan.status, PlanStatus::Ready);
    assert_eq!(plan.steps.len(), 1);
    assert_eq!(plan.steps[0].step_id, "tool:alpha");
    assert_eq!(plan.steps[0].score, 0.5);
    // 0.45 * 0.5 + 0.25 * 1.0 + 0.20 * 1.0
    assert_score(plan.score, 0.675);
}

#[tokio::test]
async fn beam_mode_selects_the_same_best_chain() {
    let capabilities = chain_catalog();
    let graph = chain_graph(&capabilities);
    let config = PlanConfig {
        mode: PlanMode::Beam,
        ..PlanConfig::default()
    };
    let plan = chain_planner(config)
        .plan(&PlanRequest::new("chain"), &graph, &capabilities)
        .await
        .unwrap();
    let ids: Vec<&str> = plan
        .steps
        .iter()
        .map(|step| step.step_id.as_str())
        .collect();
    assert_eq!(ids, ["tool:alpha", "tool:beta", "tool:gamma"]);
    assert_score(plan.score, 0.6825);
}

#[tokio::test]
async fn search_bounds_are_respected() {
    let capabilities = chain_catalog();
    let graph = chain_graph(&capabilities);

    // No expansions: only the single-node plan survives.
    let config = PlanConfig {
        max_expansions: 0,
        ..PlanConfig::default()
    };
    let plan = chain_planner(config)
        .plan(&PlanRequest::new("chain"), &graph, &capabilities)
        .await
        .unwrap();
    assert_eq!(plan.steps.len(), 1);

    // One candidate plan: only the first retained plan survives.
    let config = PlanConfig {
        max_plans: 1,
        ..PlanConfig::default()
    };
    let plan = chain_planner(config)
        .plan(&PlanRequest::new("chain"), &graph, &capabilities)
        .await
        .unwrap();
    assert_eq!(plan.steps.len(), 1);
    assert_eq!(plan.steps[0].step_id, "tool:alpha");
}

#[tokio::test]
async fn planning_is_deterministic_apart_from_the_plan_id() {
    let capabilities = chain_catalog();
    let graph = chain_graph(&capabilities);
    let planner = chain_planner(PlanConfig::default());
    let request = PlanRequest::new("chain");

    let first = planner.plan(&request, &graph, &capabilities).await.unwrap();
    let second = planner.plan(&request, &graph, &capabilities).await.unwrap();

    assert_eq!(first.steps, second.steps);
    assert_eq!(first.edges, second.edges);
    assert_eq!(first.score, second.score);
    assert_eq!(first.plan_id.len(), 32);
    assert_eq!(second.plan_id.len(), 32);

    let mut first_value = first.to_value();
    let mut second_value = second.to_value();
    assert!(first_value
        .as_object_mut()
        .unwrap()
        .remove("plan_id")
        .is_some());
    assert!(second_value
        .as_object_mut()
        .unwrap()
        .remove("plan_id")
        .is_some());
    // The JGF projection embeds the plan id as the graph id, so normalize that too.
    for value in [&mut first_value, &mut second_value] {
        value["planned_graph"]["graph"]["id"] = json!("plan");
    }
    assert_eq!(first_value, second_value);
}
