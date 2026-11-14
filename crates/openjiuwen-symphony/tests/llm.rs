//! Behaviour tests for the LLM-guided stages in [`openjiuwen_symphony::llm`].
//!
//! Every stage is exercised twice: once with a scripted [`StaticCompletionModel`]
//! standing in for a real model, and once with no model at all (the deterministic
//! offline path). Nothing here touches the network, no credentials are read, and the
//! only non-reproducible value — `TreeSearchResult::elapsed_ms` — is never asserted on.
#![allow(clippy::unwrap_used, clippy::expect_used)]
#![allow(clippy::float_cmp)]
#![allow(clippy::similar_names)] // `matcher`/`matches` read better than invented names
#![allow(clippy::assert_is_empty)]
use std::sync::Arc;

use async_trait::async_trait;
use openjiuwen_symphony::evaluation::{EvaluationInput, Metric};
use openjiuwen_symphony::graph::CAN_FEED;
use openjiuwen_symphony::llm::fingerprint::FingerprintExtractor;
use openjiuwen_symphony::llm::judge::{LlmJudgeMetric, DEFAULT_RUBRIC};
use openjiuwen_symphony::llm::ontology::{OntologyConfig, OntologyMatcher, METHOD_CONSENSUS};
use openjiuwen_symphony::llm::scoring::LlmSemanticScorer;
use openjiuwen_symphony::llm::tree::{
    RetrieverItem, RetrieverNode, TreeRetriever, TreeSearchRequest,
};
use openjiuwen_symphony::llm::{CompletionModel, FallbackMode, ModelConfig, StaticCompletionModel};
use openjiuwen_symphony::model::{CapabilityDescriptor, CapabilityIo, SemanticProfile};
use openjiuwen_symphony::orchestration::planner::SemanticScorer;
use openjiuwen_symphony::orchestration::{LexicalSemanticScorer, MatchStrategy, RelationCandidate};
use serde_json::json;

/// Wraps a single-reply scripted model for repeated calls.
fn model(reply: impl Into<String>) -> Arc<StaticCompletionModel> {
    Arc::new(StaticCompletionModel::single(reply))
}

/// Wraps a scripted model replaying `replies` in order.
fn scripted(replies: &[&str]) -> Arc<StaticCompletionModel> {
    Arc::new(StaticCompletionModel::new(
        replies.iter().map(|reply| (*reply).to_string()).collect(),
    ))
}

/// Builds a `tool` capability with the given IO declarations.
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

/// Builds a `can_feed` relation candidate from `source` to `target`.
fn pair(source: &str, target: &str, strategy: MatchStrategy, confidence: f64) -> RelationCandidate {
    RelationCandidate {
        source: source.to_string(),
        target: target.to_string(),
        relation: CAN_FEED.to_string(),
        confidence,
        strategy,
        matched_input: None,
        matched_output: None,
    }
}

/// Builds a `can_feed` relation candidate from `tool:a` to `tool:b`.
fn candidate(strategy: MatchStrategy, confidence: f64) -> RelationCandidate {
    pair("tool:a", "tool:b", strategy, confidence)
}

/// The three-capability catalog used by the ontology tests.
fn ontology_catalog() -> Vec<CapabilityDescriptor> {
    vec![
        capability(
            "a",
            "Extract",
            vec![],
            vec![CapabilityIo::new("text", "string")],
        ),
        capability(
            "b",
            "Summarise",
            vec![CapabilityIo::new("text", "string")],
            vec![CapabilityIo::new("summary", "string")],
        ),
        capability(
            "c",
            "Translate",
            vec![CapabilityIo::new("summary", "string")],
            vec![CapabilityIo::new("translation", "string")],
        ),
    ]
}

/// A tree with one translatable branch and one unrelated root item.
fn translate_tree() -> RetrieverNode {
    RetrieverNode::new("root")
        .with_items(vec![
            RetrieverItem::new("i_root", "root_payload").with_label("weather forecast")
        ])
        .with_children(vec![RetrieverNode::new("c1")
            .with_label("translate")
            .with_description("translate text between languages")
            .with_items(vec![
                RetrieverItem::new("i_child", "child_payload").with_label("translate text")
            ])])
}

// --- tree retrieval -------------------------------------------------------

#[tokio::test]
async fn tree_retrieval_descends_into_the_selected_branch() {
    let model = scripted(&["c1", "i_child"]);
    let retriever = TreeRetriever::new(
        openjiuwen_symphony::llm::TreeRetrieverConfig::default().with_model(model.clone()),
    )
    .with_root(translate_tree());
    let result = retriever
        .search("translate text", &TreeSearchRequest::default())
        .await
        .unwrap();
    assert_eq!(result.payloads, vec!["child_payload".to_string()]);
    assert_eq!(result.method, "llm");
    assert_eq!(model.call_count(), 2);
    let prompts = model.prompts();
    assert!(prompts[0].contains("Query: translate text"));
    assert!(prompts[0].contains("branch c1"));
    assert!(prompts[1].contains("Current branch"));
}

#[tokio::test]
async fn tree_retrieval_zero_abstains() {
    let retriever = TreeRetriever::new(
        openjiuwen_symphony::llm::TreeRetrieverConfig::default().with_model(model("0")),
    )
    .with_root(translate_tree());
    let result = retriever
        .search("translate", &TreeSearchRequest::default())
        .await
        .unwrap();
    assert!(result.payloads.is_empty());
    assert_eq!(result.method, "llm");
}

#[tokio::test]
async fn tree_retrieval_drops_unknown_ids() {
    let retriever = TreeRetriever::new(
        openjiuwen_symphony::llm::TreeRetrieverConfig::default()
            .with_model(model("unknown id\ni_root")),
    )
    .with_root(translate_tree());
    let result = retriever
        .search("translate", &TreeSearchRequest::default())
        .await
        .unwrap();
    assert_eq!(result.payloads, vec!["root_payload".to_string()]);
}

#[tokio::test]
async fn tree_retrieval_honours_top_k() {
    let root = RetrieverNode::new("root").with_items(vec![
        RetrieverItem::new("i1", "p1"),
        RetrieverItem::new("i2", "p2"),
        RetrieverItem::new("i3", "p3"),
    ]);
    let retriever = TreeRetriever::new(
        openjiuwen_symphony::llm::TreeRetrieverConfig::default().with_model(model("i1\ni2\ni3")),
    )
    .with_root(root);
    let result = retriever
        .search("anything", &TreeSearchRequest { top_k: Some(2) })
        .await
        .unwrap();
    assert_eq!(result.payloads, vec!["p1".to_string(), "p2".to_string()]);
}

#[tokio::test]
async fn tree_retrieval_falls_back_to_lexical_ranking_without_a_model() {
    let retriever = TreeRetriever::new(openjiuwen_symphony::llm::TreeRetrieverConfig::default())
        .with_root(translate_tree());
    let result = retriever
        .search("translate", &TreeSearchRequest::default())
        .await
        .unwrap();
    assert_eq!(result.method, "lexical_fallback");
    // The translating branch wins on query-term coverage, and its item comes first.
    assert_eq!(
        result.payloads,
        vec!["child_payload".to_string(), "root_payload".to_string()]
    );
}

#[tokio::test]
async fn tree_retrieval_falls_back_when_the_model_replies_with_nothing() {
    let retriever = TreeRetriever::new(
        openjiuwen_symphony::llm::TreeRetrieverConfig::default().with_model(model("   ")),
    )
    .with_root(translate_tree());
    let result = retriever
        .search("translate", &TreeSearchRequest::default())
        .await
        .unwrap();
    assert_eq!(result.method, "lexical_fallback");
    assert_eq!(
        result.payloads,
        vec!["child_payload".to_string(), "root_payload".to_string()]
    );
}

#[tokio::test]
async fn tree_retrieval_respects_max_depth() {
    let retriever = TreeRetriever::new(
        openjiuwen_symphony::llm::TreeRetrieverConfig::default().with_max_depth(0),
    )
    .with_root(translate_tree());
    let result = retriever
        .search("translate", &TreeSearchRequest::default())
        .await
        .unwrap();
    assert_eq!(result.payloads, vec!["root_payload".to_string()]);
}

#[tokio::test]
async fn tree_retrieval_of_an_empty_root_is_empty() {
    let retriever = TreeRetriever::new(openjiuwen_symphony::llm::TreeRetrieverConfig::default())
        .with_root(RetrieverNode::new("root"));
    let result = retriever
        .search("anything", &TreeSearchRequest::default())
        .await
        .unwrap();
    assert!(result.payloads.is_empty());
    assert_eq!(result.method, "lexical_fallback");
}

// --- ontology matching ----------------------------------------------------

#[tokio::test]
async fn ontology_accepts_a_confident_forward_match() {
    let matcher = OntologyMatcher::new(ontology_catalog()).with_config(
        OntologyConfig::new()
            .with_consensus(false)
            .with_model(model(
                r#"{"matches":[{"id":"c1","direction":"forward","confidence":0.9,"reason":"text flows"}]}"#,
            )),
    );
    let matches = matcher
        .match_candidates(&[candidate(MatchStrategy::TextualCoercion, 0.75)])
        .await
        .unwrap();
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].source_id, "tool:a");
    assert_eq!(matches[0].target_id, "tool:b");
    assert_eq!(matches[0].relation, CAN_FEED);
    assert_eq!(matches[0].confidence, 0.9);
    assert_eq!(matches[0].method, "llm_ontology_match");
    assert!(matches[0].accepted);
    assert_eq!(matches[0].reasons, vec!["text flows".to_string()]);
}

#[tokio::test]
async fn ontology_rejects_a_low_confidence_match() {
    let matcher = OntologyMatcher::new(ontology_catalog()).with_config(
        OntologyConfig::new()
            .with_consensus(false)
            .with_model(model(
                r#"{"matches":[{"id":"c1","direction":"forward","confidence":0.4}]}"#,
            )),
    );
    let matches = matcher
        .match_candidates(&[candidate(MatchStrategy::ExactIo, 1.0)])
        .await
        .unwrap();
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].confidence, 0.4);
    assert!(!matches[0].accepted);
}

#[tokio::test]
async fn ontology_drops_malformed_reply_items_and_clamps_confidence() {
    let reply = r#"{"matches":[
        {"id":"c9","direction":"forward","confidence":0.9},
        {"id":"c1","direction":"sideways","confidence":0.9},
        {"id":"c1","direction":"forward","confidence":"high"},
        {"id":"c1","direction":"forward","confidence":5.0}
    ]}"#;
    let matcher = OntologyMatcher::new(ontology_catalog()).with_config(
        OntologyConfig::new()
            .with_consensus(false)
            .with_model(model(reply)),
    );
    let matches = matcher
        .match_candidates(&[candidate(MatchStrategy::ExactIo, 1.0)])
        .await
        .unwrap();
    assert_eq!(matches.len(), 1, "only the last item is usable");
    assert_eq!(
        matches[0].confidence, 1.0,
        "confidence is clamped into [0, 1]"
    );
}

#[tokio::test]
async fn ontology_reverse_direction_swaps_the_orientation() {
    let matcher = OntologyMatcher::new(ontology_catalog()).with_config(
        OntologyConfig::new()
            .with_consensus(false)
            .with_model(model(
                r#"{"matches":[{"id":"c1","direction":"reverse","confidence":0.8}]}"#,
            )),
    );
    let matches = matcher
        .match_candidates(&[candidate(MatchStrategy::ExactIo, 1.0)])
        .await
        .unwrap();
    assert_eq!(matches[0].source_id, "tool:b");
    assert_eq!(matches[0].target_id, "tool:a");
}

#[tokio::test]
async fn ontology_consensus_keeps_the_intersection_with_the_lower_confidence() {
    // Forward order: c1 = a→b, c2 = b→c. Reversed order swaps the numbering.
    let forward = r#"{"matches":[
        {"id":"c1","direction":"forward","confidence":0.9},
        {"id":"c2","direction":"forward","confidence":0.8}
    ]}"#;
    let reverse = r#"{"matches":[
        {"id":"c1","direction":"forward","confidence":0.85},
        {"id":"c2","direction":"forward","confidence":0.95}
    ]}"#;
    let matcher =
        OntologyMatcher::new(ontology_catalog()).with_model(scripted(&[forward, reverse]));
    let matches = matcher
        .match_candidates(&[
            candidate(MatchStrategy::ExactIo, 1.0),
            pair("tool:b", "tool:c", MatchStrategy::ExactIo, 1.0),
        ])
        .await
        .unwrap();
    assert_eq!(matches.len(), 2);
    assert_eq!(matches[0].source_id, "tool:a");
    assert_eq!(matches[0].target_id, "tool:b");
    assert_eq!(matches[0].confidence, 0.9);
    assert_eq!(matches[0].method, METHOD_CONSENSUS);
    assert_eq!(matches[1].source_id, "tool:b");
    assert_eq!(matches[1].target_id, "tool:c");
    assert_eq!(matches[1].confidence, 0.8);
}

#[tokio::test]
async fn ontology_consensus_drops_matches_only_one_pass_accepted() {
    let forward = r#"{"matches":[
        {"id":"c1","direction":"forward","confidence":0.9},
        {"id":"c2","direction":"forward","confidence":0.9}
    ]}"#;
    let reverse = r#"{"matches":[{"id":"c2","direction":"forward","confidence":0.9}]}"#;
    let matcher =
        OntologyMatcher::new(ontology_catalog()).with_model(scripted(&[forward, reverse]));
    let matches = matcher
        .match_candidates(&[
            candidate(MatchStrategy::ExactIo, 1.0),
            pair("tool:b", "tool:c", MatchStrategy::ExactIo, 1.0),
        ])
        .await
        .unwrap();
    // In the reversed pass c2 is a→b, so only a→b is shared by both passes.
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].source_id, "tool:a");
    assert_eq!(matches[0].target_id, "tool:b");
}

#[tokio::test]
async fn ontology_falls_back_to_the_candidate_rule_without_a_model() {
    let matcher = OntologyMatcher::new(ontology_catalog());
    let matches = matcher
        .match_candidates(&[
            candidate(MatchStrategy::ExactIo, 0.4),
            candidate(MatchStrategy::SemanticOverlap, 0.72),
            candidate(MatchStrategy::TextualCoercion, 0.1),
        ])
        .await
        .unwrap();
    assert_eq!(matches.len(), 3);
    assert!(matches
        .iter()
        .all(|item| item.method == "deterministic_fallback"));
    let confidences: Vec<f64> = matches.iter().map(|item| item.confidence).collect();
    // Declared order survives (every key is the same capability pair).
    assert_eq!(confidences, vec![1.0, 0.72, 0.75]);
    let accepted: Vec<bool> = matches.iter().map(|item| item.accepted).collect();
    assert_eq!(accepted, vec![true, true, true]);
}

#[tokio::test]
async fn ontology_falls_back_when_the_reply_is_unparseable() {
    let matcher = OntologyMatcher::new(ontology_catalog()).with_config(
        OntologyConfig::new()
            .with_consensus(false)
            .with_model(model("no json here")),
    );
    let matches = matcher
        .match_candidates(&[candidate(MatchStrategy::ExactIo, 1.0)])
        .await
        .unwrap();
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].method, "deterministic_fallback");
    assert_eq!(matches[0].confidence, 1.0);
}

#[tokio::test]
async fn ontology_error_fallback_mode_surfaces_the_failure() {
    let matcher = OntologyMatcher::new(ontology_catalog()).with_config(
        OntologyConfig::new()
            .with_consensus(false)
            .with_fallback(FallbackMode::Error)
            .with_model(model("no json here")),
    );
    let error = matcher
        .match_candidates(&[candidate(MatchStrategy::ExactIo, 1.0)])
        .await
        .expect_err("an unusable reply must surface under FallbackMode::Error");
    assert!(error.message().contains("ontology matcher"));
}

#[tokio::test]
async fn ontology_skips_candidates_with_unknown_capabilities() {
    let matcher = OntologyMatcher::new(ontology_catalog());
    let unknown = pair("tool:missing", "tool:b", MatchStrategy::ExactIo, 1.0);
    let matches = matcher.match_candidates(&[unknown]).await.unwrap();
    assert!(matches.is_empty());
}

// --- semantic scoring -----------------------------------------------------

fn scoring_catalog() -> Vec<CapabilityDescriptor> {
    vec![
        CapabilityDescriptor::new("tool", "a", "Extract text")
            .with_description("extracts text from documents"),
        CapabilityDescriptor::new("tool", "b", "Translate")
            .with_description("translates text between languages"),
    ]
}

#[tokio::test]
async fn semantic_scorer_maps_and_clamps_judgements() {
    let reply = r#"{"judgements":[
        {"candidate_id":"tool:a","score":1.7},
        {"candidate_id":"tool:b","score":0.25},
        {"candidate_id":"tool:missing","score":0.9}
    ]}"#;
    let scorer = LlmSemanticScorer::new(ModelConfig::new().with_model(model(reply)));
    let scores = scorer.score("translate", &scoring_catalog()).await.unwrap();
    assert_eq!(scores.get("tool:a"), Some(&1.0));
    assert_eq!(scores.get("tool:b"), Some(&0.25));
    assert_eq!(scores.len(), 2, "unknown candidate ids are dropped");
}

#[tokio::test]
async fn semantic_scorer_counts_a_missing_score_as_zero() {
    let reply = r#"{"judgements":[{"candidate_id":"tool:a"}]}"#;
    let scorer = LlmSemanticScorer::new(ModelConfig::new().with_model(model(reply)));
    let scores = scorer.score("translate", &scoring_catalog()).await.unwrap();
    assert_eq!(scores.get("tool:a"), Some(&0.0));
}

#[tokio::test]
async fn semantic_scorer_falls_back_to_the_lexical_scores() {
    let catalog = scoring_catalog();
    let lexical = LexicalSemanticScorer::new();
    let expected = lexical.score("translate", &catalog).await.unwrap();

    let without_model = LlmSemanticScorer::new(ModelConfig::new());
    let scores = without_model.score("translate", &catalog).await.unwrap();
    assert!(!scores.is_empty());
    assert_eq!(scores, expected);

    let malformed = LlmSemanticScorer::new(ModelConfig::new().with_model(model("not json")));
    let scores = malformed.score("translate", &catalog).await.unwrap();
    assert_eq!(scores, expected);
}

#[tokio::test]
async fn semantic_scorer_surfaces_failures_in_error_mode() {
    let scorer = LlmSemanticScorer::new(
        ModelConfig::new()
            .with_model(model("not json"))
            .with_fallback(FallbackMode::Error),
    );
    let error = scorer
        .score("translate", &scoring_catalog())
        .await
        .expect_err("FallbackMode::Error must surface the model failure");
    assert!(error.message().contains("semantic scorer"));
}

#[tokio::test]
async fn semantic_scorer_is_empty_for_an_empty_catalog() {
    let scorer = LlmSemanticScorer::new(ModelConfig::new().with_model(model("{}")));
    let scores = scorer.score("translate", &[]).await.unwrap();
    assert!(scores.is_empty());
}

// --- fingerprint extraction ----------------------------------------------

fn fingerprint_capability() -> CapabilityDescriptor {
    CapabilityDescriptor::new("tool", "extract", "Extract")
        .with_description("Extracts text")
        .with_classification("document")
        .with_tags(vec!["text".to_string(), "text".to_string()])
        .with_inputs(vec![CapabilityIo::new("path", "string")])
}

#[tokio::test]
async fn fingerprint_extractor_parses_the_model_profile_and_dedupes() {
    let reply = json!({
        "description": "ignored",
        "semantic_profile": {
            "summary": "Turns a document into text",
            "capabilities": ["tool:extract", "tool:extract"],
            "use_cases": ["ocr"],
            "limitations": ["no tables"],
            "keywords": ["document", "text"]
        }
    })
    .to_string();
    let extractor = FingerprintExtractor::new().with_model(model(reply));
    let profile = extractor.extract(&fingerprint_capability()).await.unwrap();
    assert_eq!(profile.summary, "Turns a document into text");
    assert_eq!(profile.capabilities, vec!["tool:extract".to_string()]);
    assert_eq!(profile.use_cases, vec!["ocr".to_string()]);
    assert_eq!(profile.limitations, vec!["no tables".to_string()]);
    assert_eq!(
        profile.keywords,
        vec!["text".to_string(), "document".to_string()]
    );
}

#[tokio::test]
async fn fingerprint_extractor_falls_back_for_a_malformed_reply() {
    let extractor = FingerprintExtractor::new().with_model(model("not json"));
    let profile = extractor.extract(&fingerprint_capability()).await.unwrap();
    assert_eq!(
        profile,
        SemanticProfile {
            summary: "Extracts text".to_string(),
            capabilities: vec!["tool:extract".to_string()],
            use_cases: vec!["text".to_string()],
            limitations: Vec::new(),
            keywords: vec!["text".to_string()],
        }
    );
}

#[tokio::test]
async fn fingerprint_extractor_without_a_model_uses_the_card() {
    let extractor = FingerprintExtractor::new();
    assert!(!extractor.has_model());
    let profile = extractor.extract(&fingerprint_capability()).await.unwrap();
    assert_eq!(profile.summary, "Extracts text");
}

#[tokio::test]
async fn fingerprint_extractor_surfaces_failures_in_error_mode() {
    let extractor = FingerprintExtractor::new()
        .with_model(model("not json"))
        .with_config(
            openjiuwen_symphony::llm::FingerprintConfig::new().with_fallback(FallbackMode::Error),
        );
    assert!(extractor.extract(&fingerprint_capability()).await.is_err());
}

// --- LLM judge ------------------------------------------------------------

fn judged_input() -> EvaluationInput {
    EvaluationInput {
        capability: CapabilityDescriptor::new("tool", "a", "Name")
            .with_description("Does a thing")
            .with_inputs(vec![
                CapabilityIo::new("in", "string").with_description("input")
            ])
            .with_outputs(vec![
                CapabilityIo::new("out", "string").with_description("output")
            ]),
        ..EvaluationInput::default()
    }
}

#[tokio::test]
async fn judge_metric_passes_a_positive_verdict() {
    let metric = LlmJudgeMetric::new("quality", "card quality")
        .with_model(model(r#"{"score":1,"reason":"ok"}"#));
    let score = metric.evaluate(&judged_input()).await.unwrap();
    assert_eq!(score.metric, "quality");
    assert_eq!(score.score, Some(1.0));
    assert!(score.passed);
    assert_eq!(score.detail.as_deref(), Some("ok"));
}

#[tokio::test]
async fn judge_metric_fails_a_negative_verdict() {
    let metric = LlmJudgeMetric::new("quality", "card quality")
        .with_model(model(r#"{"score":0,"reason":"no examples"}"#));
    let score = metric.evaluate(&judged_input()).await.unwrap();
    assert_eq!(score.score, Some(0.0));
    assert!(!score.passed);
    assert_eq!(score.detail.as_deref(), Some("no examples"));
}

#[tokio::test]
async fn judge_metric_repairs_one_invalid_reply() {
    let model = scripted(&["not json", r#"{"score":1,"reason":"fixed"}"#]);
    let metric = LlmJudgeMetric::new("quality", "card quality").with_model(model.clone());
    let score = metric.evaluate(&judged_input()).await.unwrap();
    assert_eq!(score.score, Some(1.0));
    assert_eq!(score.detail.as_deref(), Some("fixed"));
    assert_eq!(model.call_count(), 2);
    assert!(model.prompts()[1].contains("rejected"));
}

#[tokio::test]
async fn judge_metric_falls_back_after_two_invalid_replies() {
    let metric = LlmJudgeMetric::new("quality", "card quality").with_model(model("not json"));
    let score = metric.evaluate(&judged_input()).await.unwrap();
    assert_eq!(score.score, Some(1.0));
    assert!(score.passed);
    let detail = score.detail.unwrap();
    assert!(detail.starts_with("deterministic_fallback:"), "{detail}");
}

#[tokio::test]
async fn judge_metric_surfaces_failures_in_error_mode() {
    let metric = LlmJudgeMetric::new("quality", "card quality")
        .with_model(model("not json"))
        .with_fallback(FallbackMode::Error);
    assert!(metric.evaluate(&judged_input()).await.is_err());
}

#[tokio::test]
async fn judge_metric_without_a_model_uses_the_static_rubric() {
    let metric = LlmJudgeMetric::new("quality", "card quality");
    let score = metric.evaluate(&judged_input()).await.unwrap();
    assert_eq!(score.score, Some(1.0));
    assert!(score.passed);
    assert!(score
        .detail
        .unwrap()
        .starts_with("deterministic_fallback: no model is configured"));
}

// --- serialization --------------------------------------------------------

#[test]
fn llm_data_types_round_trip_through_json() {
    let request: TreeSearchRequest = serde_json::from_str(r#"{"top_k": 3}"#).unwrap();
    assert_eq!(request.top_k, Some(3));
    let unknown: TreeSearchRequest = serde_json::from_str(r#"{"other": true}"#).unwrap();
    assert_eq!(unknown, TreeSearchRequest::default());

    let result = openjiuwen_symphony::llm::TreeSearchResult {
        payloads: vec!["p".to_string()],
        method: "llm".to_string(),
        elapsed_ms: 0,
    };
    let encoded = serde_json::to_value(&result).unwrap();
    assert_eq!(encoded["method"], "llm");
    assert_eq!(
        serde_json::from_value::<openjiuwen_symphony::llm::TreeSearchResult>(encoded).unwrap(),
        result
    );

    let config: openjiuwen_symphony::llm::FingerprintConfig =
        serde_json::from_str(r#"{"fallback": "error"}"#).unwrap();
    assert_eq!(config.fallback, FallbackMode::Error);
    assert_eq!(
        openjiuwen_symphony::llm::FingerprintConfig::default(),
        openjiuwen_symphony::llm::FingerprintConfig::new()
    );

    let matched = openjiuwen_symphony::llm::OntologyMatch::new("tool:a", "tool:b");
    let encoded = serde_json::to_value(&matched).unwrap();
    assert_eq!(encoded["relation"], CAN_FEED);
    assert_eq!(encoded["source_id"], "tool:a");
}

// --- the model seam -------------------------------------------------------

/// A [`CompletionModel`] that always fails, proving the stages treat a model error as
/// a soft failure.
#[derive(Debug)]
struct FailingModel;

#[async_trait]
impl CompletionModel for FailingModel {
    fn name(&self) -> &'static str {
        "failing-model"
    }

    async fn complete(&self, _prompt: &str) -> openjiuwen_symphony::Result<String> {
        Err(openjiuwen_symphony::Error::llm("provider unavailable"))
    }
}

#[tokio::test]
async fn every_stage_falls_back_when_the_model_fails() {
    let failing: Arc<dyn CompletionModel> = Arc::new(FailingModel);
    let catalog = ontology_catalog();

    let matcher = OntologyMatcher::new(catalog.clone()).with_config(
        OntologyConfig::new()
            .with_consensus(false)
            .with_model(failing.clone()),
    );
    let matches = matcher
        .match_candidates(&[candidate(MatchStrategy::ExactIo, 1.0)])
        .await
        .unwrap();
    assert_eq!(matches[0].method, "deterministic_fallback");

    let scorer = LlmSemanticScorer::new(ModelConfig::new().with_model(failing.clone()));
    assert!(!scorer.score("extract", &catalog).await.unwrap().is_empty());

    let extractor = FingerprintExtractor::new().with_model(failing.clone());
    let profile: SemanticProfile = extractor.extract(&fingerprint_capability()).await.unwrap();
    assert_eq!(profile.summary, "Extracts text");

    let retriever = TreeRetriever::new(
        openjiuwen_symphony::llm::TreeRetrieverConfig::default().with_model(failing),
    )
    .with_root(translate_tree());
    let result = retriever
        .search("translate", &TreeSearchRequest::default())
        .await
        .unwrap();
    assert_eq!(result.method, "lexical_fallback");
}

#[tokio::test]
async fn model_names_are_visible_through_the_seams() {
    let model = StaticCompletionModel::single("{}");
    assert_eq!(model.name(), "static-completion-model");
    let metric: Arc<dyn Metric> = Arc::new(LlmJudgeMetric::default());
    assert_eq!(metric.name(), "llm_judge");
    assert_eq!(LlmJudgeMetric::default().rubric(), DEFAULT_RUBRIC);
}
