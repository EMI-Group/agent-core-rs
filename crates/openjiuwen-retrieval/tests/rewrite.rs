#![allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]
//! Integration tests for query rewriting and the shared completion-model seam.
//!
//! Everything runs offline: the heuristic rewriter needs no model at all, and
//! the model rewriter is driven by [`StaticCompletionModel`], which replays a
//! scripted reply. Behavior ports mirror Python's
//! `tests/unit_tests/core/retrieval/query_rewriter`; the heuristic rewriter and
//! the static model are Rust additions needed by the offline test story, so
//! their guarantees (determinism, bounds, fallback) are asserted as documented.
use std::collections::BTreeMap;
use std::sync::Arc;

use openjiuwen_common::error::ErrorKind;
use openjiuwen_retrieval::model::{extract_json_object, CompletionModel, StaticCompletionModel};
use openjiuwen_retrieval::rewrite::{
    normalize_query, tokenize, HeuristicQueryRewriter, HeuristicRewriteConfig, ModelQueryRewriter,
    ModelRewriteConfig, QueryRewriter, RewriteOutput, RewriteRequest, TypoFix,
    DEFAULT_REWRITE_PROMPT,
};

/// Builds a synonym table used by the expansion tests.
fn synonyms() -> BTreeMap<String, Vec<String>> {
    let mut table = BTreeMap::new();
    table.insert("rust".to_owned(), vec!["rustlang".to_owned()]);
    table.insert(
        "async".to_owned(),
        vec!["asynchronous".to_owned(), "concurrent".to_owned()],
    );
    table
}

/// Builds a rewriter with the default policy plus `synonyms()`.
fn rewriter_with_synonyms() -> HeuristicQueryRewriter {
    HeuristicQueryRewriter::new(HeuristicRewriteConfig::default().with_synonyms(synonyms()))
}

/// Builds a model rewriter replaying `reply`.
fn model_rewriter(reply: impl Into<String>) -> ModelQueryRewriter {
    ModelQueryRewriter::with_default_config(Arc::new(StaticCompletionModel::single(reply)))
}

#[tokio::test]
async fn keyword_extraction_drops_stopwords_and_ranks_by_length() {
    let rewriter = HeuristicQueryRewriter::default();
    let keywords = rewriter
        .keywords("What is the difference between Tokio and async-std?")
        .await
        .unwrap();
    assert_eq!(
        keywords,
        vec!["difference", "between", "tokio", "async", "std"]
    );
}

#[tokio::test]
async fn keyword_extraction_uses_idf_when_a_corpus_vocabulary_is_supplied() {
    let vocabulary: BTreeMap<String, usize> = [
        ("tokio".to_owned(), 2),
        ("rust".to_owned(), 40),
        ("python".to_owned(), 35),
    ]
    .into_iter()
    .collect();
    let rewriter = HeuristicQueryRewriter::new(
        HeuristicRewriteConfig::default().with_corpus_vocabulary(vocabulary),
    );

    let keywords = rewriter.keywords("rust python tokio").await.unwrap();
    assert_eq!(keywords, vec!["tokio", "python", "rust"]);

    // The cap still applies after ranking.
    let capped = HeuristicQueryRewriter::new(
        HeuristicRewriteConfig::default()
            .with_corpus_vocabulary(
                [
                    ("tokio".to_owned(), 2),
                    ("rust".to_owned(), 40),
                    ("python".to_owned(), 35),
                ]
                .into_iter()
                .collect(),
            )
            .with_max_keywords(2),
    );
    assert_eq!(
        capped.keywords("rust python tokio").await.unwrap(),
        vec!["tokio", "python"]
    );
}

#[test]
fn whitespace_and_filler_are_normalised_deterministically() {
    assert_eq!(
        normalize_query("  Um,   so   please explain   Tokio  "),
        "explain Tokio"
    );
    assert_eq!(normalize_query("explain tokio, please"), "explain tokio,");
    assert_eq!(normalize_query(""), "");
    assert_eq!(normalize_query("please"), "please");
    assert_eq!(
        tokenize("Tokio's \"runtime\", v1!"),
        vec!["tokio", "s", "runtime", "v1"]
    );
}

#[tokio::test]
async fn rewrite_normalises_but_preserves_the_original_query() {
    let rewriter = HeuristicQueryRewriter::default();
    let output = rewriter
        .rewrite(&RewriteRequest::new("  Please  explain Tokio  "))
        .await
        .unwrap();
    assert_eq!(output.before, "Please  explain Tokio");
    assert_eq!(output.standalone_query, "explain Tokio");
    assert_eq!(output.intention, "please  explain tokio");
    assert!(output.from_history.is_empty());
    assert!(output.missing.is_empty());
    assert!(output.typo.is_empty());
    assert!(output.gibberish.is_empty());
}

#[tokio::test]
async fn follow_up_queries_are_resolved_from_history() {
    let rewriter = HeuristicQueryRewriter::default();
    let request = RewriteRequest::new("what about it").with_history([
        "hello",
        "tell me about the rust book",
        "which chapter covers async",
    ]);
    let output = rewriter.rewrite(&request).await.unwrap();
    assert_eq!(
        output.standalone_query,
        "which chapter covers async what about it"
    );
    assert_eq!(output.from_history, "which chapter covers async");
    assert_eq!(output.before, "what about it");

    // A referential token at the start is enough, even in a longer query.
    let referential = RewriteRequest::new("that runtime sounds interesting")
        .with_history(["tokio has a work stealing scheduler"]);
    let output = rewriter.rewrite(&referential).await.unwrap();
    assert_eq!(
        output.standalone_query,
        "tokio has a work stealing scheduler that runtime sounds interesting"
    );
    assert!(!output.from_history.is_empty());
}

#[tokio::test]
async fn plain_queries_are_rewritten_without_history() {
    let rewriter = HeuristicQueryRewriter::default();
    let request = RewriteRequest::new("explain the tokio scheduler in detail")
        .with_history(["tell me about the rust book"]);
    let output = rewriter.rewrite(&request).await.unwrap();
    assert_eq!(
        output.standalone_query,
        "explain the tokio scheduler in detail"
    );
    assert!(output.from_history.is_empty());

    // A short query with no history is left alone rather than invented from.
    let short = rewriter
        .rewrite(&RewriteRequest::new("tokio"))
        .await
        .unwrap();
    assert_eq!(short.standalone_query, "tokio");
    assert!(short.from_history.is_empty());
}

#[tokio::test]
async fn missing_terms_and_references_track_history_and_synonyms() {
    let rewriter = rewriter_with_synonyms();
    let request =
        RewriteRequest::new("rust async runtime").with_history(["we discussed the rust language"]);
    let output = rewriter.rewrite(&request).await.unwrap();

    assert_eq!(output.missing, vec!["runtime", "async"]);
    assert_eq!(output.references["rust"], serde_json::json!(["rustlang"]));
    assert_eq!(
        output.references["async"],
        serde_json::json!(["asynchronous", "concurrent"])
    );
    assert!(output.references.get("runtime").is_none());
}

#[tokio::test]
async fn expansion_is_deterministic_deduplicated_and_bounded() {
    let rewriter = rewriter_with_synonyms();
    let first = rewriter.expand("rust async runtime").await.unwrap();
    let second = rewriter.expand("rust async runtime").await.unwrap();
    assert_eq!(first, second);
    assert_eq!(first.len(), 3);
    assert_eq!(first[0], "rust async runtime");
    assert_eq!(first[1], "runtime async rust");
    assert_eq!(
        first[2],
        "runtime async asynchronous concurrent rust rustlang"
    );

    // Duplicate variants collapse, and the cap is honoured.
    let capped = HeuristicQueryRewriter::new(HeuristicRewriteConfig::default().with_max_queries(1));
    assert_eq!(
        capped.expand("rust rust tokio").await.unwrap(),
        vec!["rust rust tokio"]
    );
}

#[test]
fn rewrite_output_round_trips_with_every_python_field_name() {
    let mut references = serde_json::Map::new();
    references.insert("it".to_owned(), serde_json::json!(["the rust book"]));
    let output = RewriteOutput {
        before: "what about it".to_owned(),
        intention: "learn the rust book".to_owned(),
        standalone_query: "what about the rust book".to_owned(),
        references,
        missing: vec!["chapter".to_owned()],
        typo: vec![TypoFix::new("rustt", "rust", "spelling")],
        gibberish: vec!["asdf".to_owned()],
        from_history: "tell me about the rust book".to_owned(),
    };

    let text = serde_json::to_string(&output).unwrap();
    for key in [
        "\"before\"",
        "\"intention\"",
        "\"standalone_query\"",
        "\"references\"",
        "\"missing\"",
        "\"typo\"",
        "\"gibberish\"",
        "\"from_history\"",
    ] {
        assert!(text.contains(key), "missing JSON key {key} in {text}");
    }
    let restored: RewriteOutput = serde_json::from_str(&text).unwrap();
    assert_eq!(restored, output);

    // A partial object still deserialises, because every optional field
    // defaults; only `before`/`standalone_query` are required.
    let partial: RewriteOutput =
        serde_json::from_str(r#"{"before": "q", "standalone_query": "q"}"#).unwrap();
    assert_eq!(partial.before, "q");
    assert_eq!(partial.standalone_query, "q");
    assert_eq!(partial.intention, "");
    assert!(partial.references.is_empty());
    assert!(partial.missing.is_empty());
}

#[tokio::test]
async fn model_rewriter_parses_a_well_formed_json_reply() {
    let reply = serde_json::json!({
        "before": "what about it",
        "intention": "learn the rust book",
        "standalone_query": "what about the rust book",
        "references": {"it": ["the rust book"]},
        "missing": ["chapter"],
        "typo": [{"original": "rustt", "corrected": "rust", "reason": "spelling"}],
        "gibberish": [],
        "from_history": "tell me about the rust book"
    })
    .to_string();
    let model = Arc::new(StaticCompletionModel::single(reply));
    let rewriter = ModelQueryRewriter::with_default_config(model.clone());

    let request =
        RewriteRequest::new("what about it").with_history(["hello", "tell me about the rust book"]);
    let output = rewriter.rewrite(&request).await.unwrap();

    assert_eq!(output.before, "what about it");
    assert_eq!(output.intention, "learn the rust book");
    assert_eq!(output.standalone_query, "what about the rust book");
    assert_eq!(output.missing, vec!["chapter"]);
    assert_eq!(output.typo, vec![TypoFix::new("rustt", "rust", "spelling")]);
    assert_eq!(output.from_history, "tell me about the rust book");

    assert_eq!(model.call_count(), 1);
    let prompt = model.last_prompt().unwrap();
    assert!(prompt.contains("user: tell me about the rust book"));
    assert!(prompt.contains("what about it"));
    assert!(DEFAULT_REWRITE_PROMPT.contains("{history}"));
    assert!(DEFAULT_REWRITE_PROMPT.contains("{query}"));
}

#[tokio::test]
async fn model_rewriter_accepts_prose_and_fences_around_the_json() {
    let reply =
        "Sure!\n```json\n{\"before\": \"q\", \"standalone_query\": \"what is rust\"}\n```\nEnjoy.";
    let rewriter = model_rewriter(reply);
    let output = rewriter.rewrite(&RewriteRequest::new("q")).await.unwrap();
    assert_eq!(output.standalone_query, "what is rust");
    assert_eq!(
        extract_json_object(reply).unwrap(),
        "{\"before\": \"q\", \"standalone_query\": \"what is rust\"}"
    );
}

#[tokio::test]
async fn model_rewriter_defaults_missing_fields_and_clamps_a_blank_query() {
    let rewriter = model_rewriter("{\"before\": \"q\", \"standalone_query\": \"   \"}");
    let output = rewriter
        .rewrite(&RewriteRequest::new("  what is rust  "))
        .await
        .unwrap();
    assert_eq!(output.standalone_query, "what is rust");
    assert_eq!(output.intention, "");
    assert!(output.references.is_empty());
    assert!(output.missing.is_empty());
    assert!(output.typo.is_empty());
    assert!(output.gibberish.is_empty());
    assert_eq!(output.from_history, "");
}

#[tokio::test]
async fn malformed_replies_fail_without_a_fallback() {
    let rewriter = model_rewriter("I am sorry, I cannot do that.");
    let error = rewriter
        .rewrite(&RewriteRequest::new("what is rust"))
        .await
        .expect_err("unparsable reply must fail");
    assert_eq!(error.kind(), ErrorKind::Retrieval);
    assert!(
        error.message().contains("no JSON object found"),
        "unexpected message: {}",
        error.message()
    );
    assert!(error.message().contains("I am sorry, I cannot do that."));

    let broken = model_rewriter("{\"standalone_query\": }");
    let error = broken
        .rewrite(&RewriteRequest::new("what is rust"))
        .await
        .expect_err("invalid JSON must fail");
    assert_eq!(error.kind(), ErrorKind::Retrieval);
    assert!(error.message().contains("invalid JSON"));
}

#[tokio::test]
async fn malformed_replies_use_the_configured_fallback() {
    let config =
        ModelRewriteConfig::default().with_fallback(Arc::new(HeuristicQueryRewriter::default()));
    let rewriter = ModelQueryRewriter::new(
        Arc::new(StaticCompletionModel::single("no json here")),
        config,
    );
    let request =
        RewriteRequest::new("what about it").with_history(["tell me about the rust book"]);
    let output = rewriter.rewrite(&request).await.unwrap();
    assert_eq!(
        output.standalone_query,
        "tell me about the rust book what about it"
    );
    assert_eq!(output.from_history, "tell me about the rust book");
}

#[tokio::test]
async fn model_rewriter_keywords_and_expansion_follow_the_rewritten_query() {
    let reply = serde_json::json!({
        "before": "what about it",
        "standalone_query": "the rust book chapter about async runtime"
    })
    .to_string();
    let rewriter = model_rewriter(reply);
    assert_eq!(
        rewriter.keywords("what about it").await.unwrap(),
        vec!["chapter", "runtime", "async", "rust", "book"]
    );
    let variants = rewriter.expand("what about it").await.unwrap();
    assert_eq!(variants[0], "the rust book chapter about async runtime");
    assert_eq!(variants[1], "chapter runtime async rust book");
}

#[tokio::test]
async fn blank_queries_are_rejected_before_any_model_call() {
    let model = Arc::new(StaticCompletionModel::single("{}"));
    let rewriter = ModelQueryRewriter::with_default_config(model.clone());
    let error = rewriter
        .rewrite(&RewriteRequest::new("   "))
        .await
        .expect_err("blank query must fail");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert_eq!(model.call_count(), 0);
    assert!(rewriter.keywords("").await.is_err());
}

#[tokio::test]
async fn batch_rewrite_preserves_request_order() {
    let rewriter = HeuristicQueryRewriter::default();
    let requests = vec![
        RewriteRequest::new("what is rust"),
        RewriteRequest::new("what about it").with_history(["tell me about tokio"]),
        RewriteRequest::new("python async io"),
    ];
    let outputs = rewriter.rewrite_batch(&requests).await.unwrap();
    assert_eq!(outputs.len(), 3);
    assert_eq!(outputs[0].standalone_query, "what is rust");
    assert_eq!(
        outputs[1].standalone_query,
        "tell me about tokio what about it"
    );
    assert_eq!(outputs[2].standalone_query, "python async io");
}

#[tokio::test]
async fn static_model_replays_the_script_then_repeats_the_last_reply() {
    let model = StaticCompletionModel::new(vec!["first".to_owned(), "second".to_owned()]);
    assert_eq!(model.complete("a").await.unwrap(), "first");
    assert_eq!(model.complete("b").await.unwrap(), "second");
    assert_eq!(model.complete("c").await.unwrap(), "second");
    assert_eq!(model.call_count(), 3);
    assert_eq!(model.prompts(), vec!["a", "b", "c"]);
    assert_eq!(model.name(), "static-completion-model");
}
