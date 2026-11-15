//! Integration coverage for the query-time search runtime's foundation layer.
//!
//! Every scenario is offline and deterministic: a hand-built in-memory [`LoadedIndex`] (no
//! filesystem), the retrieval crate's `HashEmbedding` for the embedding path, no clock and no
//! network. The scoring rules under test are documented in
//! [`openjiuwen_symphony::retrieval::search::lexical`].

#![allow(clippy::unwrap_used, clippy::expect_used)]
#![allow(clippy::float_cmp)]

use std::collections::BTreeMap;
use std::sync::Arc;

use openjiuwen_retrieval::HashEmbedding;
use openjiuwen_symphony::retrieval::search::codebooks::CompactCodebook;
use openjiuwen_symphony::retrieval::search::config::{
    DisclosureConfig, EmbeddingScoringConfig, ProgressiveSearchConfig, SelectionMode, DEFAULT_TOP_K,
};
use openjiuwen_symphony::retrieval::search::embedding::{
    blend_scores, cosine_similarity, EmbeddingScorer,
};
use openjiuwen_symphony::retrieval::search::flow::{
    ExposedFragment, SearchCursor, SelectableTarget, SelectionResult, ABSTAIN_TOKEN,
};
use openjiuwen_symphony::retrieval::search::lexical::{
    compile_matcher, tokenize, LexicalDocument, LexicalHit, LexicalIndex, BRANCH_DOCUMENT_PREFIX,
};
use openjiuwen_symphony::retrieval::search::truncation::{
    take_budget, truncate_chars, truncate_chars_with_ellipsis, truncate_sorted_by_score,
    ScoreTruncationConfig, REASON_AUTOCUT, REASON_MIN_SCORE, REASON_RELATIVE_MIN_SCORE,
};
use openjiuwen_symphony::retrieval::types::{
    CatalogRecord, IndexManifest, LoadedIndex, RetrieverChoice, RetrieverItem, RetrieverNode,
    ROOT_NODE_ID,
};

/// A three-record catalog over two branches, plus the tree the records live in.
///
/// ```text
/// ROOT
/// └── Skills
///     ├── Skills.Data   ("Data Skills")
///     │   ├── alpha  Alpha Skill      "handles alpha tasks"
///     │   └── beta   Beta Skill       "consumes alpha feeds"
///     └── Skills.Writing ("Writing Skills")
///         └── gamma  Data Flow Guide  "explains a data flow"
/// ```
fn loaded_index() -> LoadedIndex {
    let alpha = RetrieverNode::new("Skills.Data.alpha", "Alpha Skill")
        .with_description("handles alpha tasks")
        .with_items(vec![RetrieverItem::new("alpha", "Skills.Data.alpha")]);
    let beta = RetrieverNode::new("Skills.Data.beta", "Beta Skill")
        .with_description("consumes alpha feeds")
        .with_items(vec![RetrieverItem::new("beta", "Skills.Data.beta")]);
    let gamma = RetrieverNode::new("Skills.Writing.gamma", "Data Flow Guide")
        .with_description("explains a data flow")
        .with_items(vec![RetrieverItem::new("gamma", "Skills.Writing.gamma")]);
    let data = RetrieverNode::new("Skills.Data", "Data Skills")
        .with_description("everything about data")
        .with_children(vec![alpha, beta]);
    let writing = RetrieverNode::new("Skills.Writing", "Writing Skills")
        .with_description("everything about writing")
        .with_children(vec![gamma]);
    let skills = RetrieverNode::new("Skills", "Skills").with_children(vec![data, writing]);
    let root = RetrieverNode::new(ROOT_NODE_ID, ROOT_NODE_ID).with_children(vec![skills]);

    let records = vec![
        CatalogRecord::new("alpha", "Skills.Data.alpha")
            .with_name("Alpha Skill")
            .with_description("handles alpha tasks")
            .with_retrieval_text("alpha skill handles alpha tasks and alpha feeds"),
        CatalogRecord::new("beta", "Skills.Data.beta")
            .with_name("Beta Skill")
            .with_description("consumes alpha feeds")
            .with_retrieval_text("beta skill consumes alpha feeds"),
        CatalogRecord::new("gamma", "Skills.Writing.gamma")
            .with_name("Data Flow Guide")
            .with_description("explains a data flow")
            .with_retrieval_text("data flow pipelines for a data flow guide"),
    ];
    let choices = vec![
        RetrieverChoice::new("alpha", "Skills.Data.alpha"),
        RetrieverChoice::new("beta", "Skills.Data.beta"),
        RetrieverChoice::new("gamma", "Skills.Writing.gamma"),
    ];
    LoadedIndex {
        root,
        choices,
        records,
        manifest: IndexManifest::new("full", "skill"),
        item_paths: BTreeMap::new(),
    }
}

fn keys(hits: &[LexicalHit]) -> Vec<String> {
    hits.iter().map(|hit| hit.key.clone()).collect()
}

fn documents() -> Vec<LexicalDocument> {
    vec![
        LexicalDocument::new("skills.alpha", "Alpha Skill")
            .with_description("handles alpha tasks")
            .with_category("Skills > Data"),
        LexicalDocument::new("skills.beta", "Beta Skill")
            .with_description("beta beta beta helper")
            .with_body("alpha body text")
            .with_category("Skills > Data"),
        LexicalDocument::new("skills.gamma", "Gamma Skill")
            .with_description("a data flow guide")
            .with_category("Skills > Writing"),
    ]
}

/// 1. `from_loaded_index` indexes the catalog records **and** the tree's branch nodes.
#[test]
fn the_loaded_index_yields_item_and_branch_documents() {
    let index = LexicalIndex::from_loaded_index(&loaded_index());
    // Three catalog records plus four branch nodes (ROOT, Skills, Skills.Data, Skills.Writing).
    assert_eq!(index.len(), 7);
    assert!(!index.is_empty());

    let item = index.document("Skills.Data.alpha").unwrap();
    assert_eq!(item.name, "Alpha Skill");
    assert_eq!(item.description, "handles alpha tasks");
    assert_eq!(item.category, "Skills > Data");
    assert!(item.body.contains("alpha feeds"));

    let branch = index
        .document(&format!("{BRANCH_DOCUMENT_PREFIX}Skills.Data"))
        .unwrap();
    assert_eq!(branch.name, "Data Skills");
    assert_eq!(branch.description, "everything about data");
    assert_eq!(
        branch.category, "Skills",
        "the parent cid path, root excluded"
    );

    let top = index
        .document(&format!("{BRANCH_DOCUMENT_PREFIX}Skills"))
        .unwrap();
    assert_eq!(top.category, "", "a top-level branch has no ancestor cid");

    // A leaf is an item, not a branch document.
    assert!(index
        .document(&format!("{BRANCH_DOCUMENT_PREFIX}Skills.Data.alpha"))
        .is_none());
    assert!(index.document("missing").is_none());
}

/// 2. A name hit outranks a description-only hit, and a branch is rankable through
///    `search_in`.
#[test]
fn field_weighted_ranking_prefers_names_and_can_rank_branches() {
    let index = LexicalIndex::from_loaded_index(&loaded_index());
    let hits = index.search("alpha").unwrap();
    assert_eq!(
        keys(&hits).first().map(String::as_str),
        Some("Skills.Data.alpha"),
        "{:?}",
        keys(&hits)
    );
    let position = |key: &str| keys(&hits).iter().position(|hit| hit == key);
    assert!(
        position("Skills.Data.alpha") < position("Skills.Data.beta"),
        "the identity match must outrank the description-only match: {:?}",
        keys(&hits)
    );

    let branches = index
        .search_in("data", &[&format!("{BRANCH_DOCUMENT_PREFIX}Skills.Data")])
        .unwrap();
    assert_eq!(branches.len(), 1);
    assert_eq!(
        branches[0].key,
        format!("{BRANCH_DOCUMENT_PREFIX}Skills.Data")
    );

    // The same query ranks the branch together with the items when it is not scoped.
    let all = index.search("data").unwrap();
    assert!(keys(&all).contains(&format!("{BRANCH_DOCUMENT_PREFIX}Skills.Data")));
    assert!(keys(&all).contains(&"Skills.Writing.gamma".to_owned()));
}

/// 3. Phrase matching and `|` alternatives behave per the documented rules.
#[test]
fn phrases_and_alternatives_follow_the_documented_rules() {
    let index = LexicalIndex::from_loaded_index(&loaded_index());

    // "data flow" is a phrase hit in the identity fields of the guide, so it wins.
    let hits = index.search("data flow").unwrap();
    assert_eq!(keys(&hits)[0], "Skills.Writing.gamma");

    // A quoted phrase is matched literally.
    let quoted = index.search("\"data flow\"").unwrap();
    assert_eq!(keys(&quoted)[0], "Skills.Writing.gamma");

    // Alternatives: either side matching is enough.
    let either = index.search("writing|alpha").unwrap();
    let either_keys = keys(&either);
    assert!(either_keys.contains(&"Skills.Writing.gamma".to_owned()));
    assert!(either_keys.contains(&"Skills.Data.alpha".to_owned()));
    assert!(either_keys.contains(&format!("{BRANCH_DOCUMENT_PREFIX}Skills.Writing")));

    // Each alternative contributes its own phrase score, so the document matching both
    // alternatives leads the one matching a single alternative.
    let two_alternatives = LexicalIndex::build(vec![
        LexicalDocument::new("k1", "Alpha Writer"),
        LexicalDocument::new("k2", "Alpha Only"),
    ]);
    assert_eq!(
        keys(&two_alternatives.search("alpha|writer").unwrap()),
        vec!["k1".to_owned(), "k2".to_owned()]
    );

    // A substring that only appears in a body still ranks below a name hit.
    let body = index.search("pipelines").unwrap();
    assert_eq!(keys(&body), vec!["Skills.Writing.gamma".to_owned()]);

    // Nothing matches an absent term.
    assert_eq!(index.search("nonexistentterm").unwrap(), vec![]);
}

/// 4. A CJK query returns a bounded, prefix-labelled description snippet.
#[test]
fn the_cjk_snippet_is_labelled_and_bounded() {
    let description = format!("{} 能力树索引构建 {}", "前".repeat(40), "后".repeat(40));
    let index = LexicalIndex::build(vec![
        LexicalDocument::new("skills.tree", "Tree Index").with_description(description.clone())
    ]);
    let snippet = index.match_snippet("skills.tree", &["能力树"], 0, 16);
    assert!(snippet.starts_with("description: "), "{snippet}");
    let body = snippet.trim_start_matches("description: ");
    assert!(body.chars().count() <= 16, "{snippet}");
    assert!(snippet.contains('能'), "{snippet}");
    // The excerpt is a contiguous run of the original text, so no character was split.
    let excerpt: String = body.chars().filter(|character| *character != '…').collect();
    assert!(description.contains(&excerpt), "{snippet}");

    // The description is visible up to `visible_description_chars`, so the same query
    // exposes nothing.
    assert_eq!(index.match_snippet("skills.tree", &["能力树"], 200, 16), "");
    // No query, no snippet.
    assert_eq!(index.match_snippet("skills.tree", &[], 0, 16), "");
}

/// 5. `search_in` scopes the corpus; unknown keys are ignored.
#[test]
fn search_in_only_ranks_the_requested_subset() {
    let index = LexicalIndex::build(documents());
    let scoped = index
        .search_in("alpha", &["skills.beta", "unknown", "skills.beta"])
        .unwrap();
    assert_eq!(keys(&scoped), vec!["skills.beta".to_owned()]);
    assert_eq!(index.search_in("alpha", &["unknown"]).unwrap(), vec![]);
    assert_eq!(
        index.search_in("alpha", &[]).unwrap(),
        index.search("alpha").unwrap(),
        "an empty scope means the whole corpus"
    );
    let scoped = index
        .search_in("skills", &["skills.gamma", "skills.alpha"])
        .unwrap();
    assert!(scoped.len() <= 2);
    assert!(keys(&scoped)
        .iter()
        .all(|key| key == "skills.gamma" || key == "skills.alpha"));
}

/// 6. A blank query is a validation error; an empty index searches to an empty vec.
#[test]
fn blank_queries_are_rejected_and_an_empty_index_is_quiet() {
    let index = LexicalIndex::build(documents());
    for query in ["", "   ", "\n\t"] {
        let error = index.search(query).unwrap_err();
        assert!(error.message().contains("query"), "{}", error.message());
        assert!(index.search_in(query, &["skills.alpha"]).is_err());
    }
    // An over-long expression is rejected by the same bound.
    assert!(index.search(&"a".repeat(1024)).is_err());

    let empty = LexicalIndex::build(Vec::new());
    assert!(empty.is_empty());
    assert_eq!(empty.len(), 0);
    assert_eq!(empty.search("alpha").unwrap(), vec![]);
    assert_eq!(empty.search_in("alpha", &["skills.alpha"]).unwrap(), vec![]);
    assert_eq!(empty.match_snippet("skills.alpha", &["alpha"], 0, 20), "");
}

/// 7. Ranking is deterministic and independent of the input order.
#[test]
fn ranking_is_deterministic_and_order_independent() {
    let forward = LexicalIndex::build(documents());
    let first = forward.search("alpha data").unwrap();
    let second = forward.search("alpha data").unwrap();
    assert_eq!(first, second);
    assert_eq!(
        serde_json::to_string(&first).unwrap(),
        serde_json::to_string(&second).unwrap(),
        "byte-identical serialization"
    );

    let mut reversed = documents();
    reversed.reverse();
    let backward = LexicalIndex::build(reversed);
    assert_eq!(backward.search("alpha data").unwrap(), first);

    // A duplicated key keeps the last document, so the ranking does not depend on it either.
    let duplicated = LexicalIndex::build(vec![
        LexicalDocument::new("k", "First"),
        LexicalDocument::new("k", "Second"),
    ]);
    assert_eq!(duplicated.len(), 1);
    assert_eq!(duplicated.document("k").unwrap().name, "Second");

    // Score ties break on the key.
    let tied = LexicalIndex::build(vec![
        LexicalDocument::new("b", "Same"),
        LexicalDocument::new("a", "Same"),
    ]);
    assert_eq!(keys(&tied.search("same").unwrap()), vec!["a", "b"]);
}

/// 8. The embedding scorer is deterministic and `blend_scores` reproduces either ranking at
///    the extremes.
#[tokio::test]
async fn the_embedding_path_is_deterministic_and_blendable() {
    let embedding: Arc<dyn openjiuwen_retrieval::Embedding> = Arc::new(HashEmbedding::new(16));
    let scorer = EmbeddingScorer::new(embedding);
    let documents = documents();
    assert_eq!(scorer.index(&documents).await.unwrap(), 3);
    assert_eq!(scorer.len(), 3);

    let embedding_scores = scorer.scores("skill").await.unwrap();
    assert_eq!(embedding_scores.len(), 3);
    for score in embedding_scores.values() {
        assert!((-1.0..=1.0).contains(score), "{score}");
        assert!(score.is_finite());
    }
    let again = scorer.scores("skill").await.unwrap();
    assert_eq!(
        embedding_scores, again,
        "the same query embeds to the same scores"
    );

    let embedding_order = scorer.search("skill", 0).await.unwrap();
    assert_eq!(embedding_order.len(), 3);
    assert!(embedding_order
        .windows(2)
        .all(|pair| pair[0].score >= pair[1].score));

    let lexical = LexicalIndex::build(documents);
    let lexical_hits = lexical.search("skill").unwrap();
    assert_eq!(lexical_hits.len(), 3, "every document carries the name hit");

    // weight 0.0 => the lexical ranking (the documented no-embedding path).
    let lexical_only = blend_scores(&lexical_hits, &embedding_scores, 0.0);
    assert_eq!(keys(&lexical_only), keys(&lexical_hits));
    assert!(lexical_only
        .iter()
        .all(|hit| (0.0..=1.0).contains(&hit.score)));
    // weight 1.0 => the embedding ranking.
    let embedding_only = blend_scores(&lexical_hits, &embedding_scores, 1.0);
    assert_eq!(keys(&embedding_only), keys(&embedding_order));

    // A mixed weight keeps every key and stays deterministic.
    let mixed = blend_scores(&lexical_hits, &embedding_scores, 0.5);
    assert_eq!(mixed.len(), lexical_hits.len());
    assert_eq!(blend_scores(&lexical_hits, &embedding_scores, 0.5), mixed);
    // An embedding-only key keeps a score and appears in the union.
    let mut embedding_only = embedding_scores.clone();
    embedding_only.insert("skills.delta".to_owned(), 1.0);
    let union = blend_scores(&lexical_hits, &embedding_only, 1.0);
    assert_eq!(union.len(), 4);
    assert_eq!(union[0].key, "skills.delta");

    // Degenerate cosine inputs never produce NaN.
    assert_eq!(cosine_similarity(&[], &[]), 0.0);
    assert_eq!(cosine_similarity(&[1.0], &[1.0, 0.0]), 0.0);
    assert_eq!(cosine_similarity(&[0.0, 0.0], &[1.0, 0.0]), 0.0);
    assert_eq!(cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]), 1.0);
}

/// 9. Score truncation follows the reference decision order; the budget helpers are bounded
///    and character-safe.
#[test]
fn truncation_decisions_and_budgets_are_hand_computable() {
    // Empty input: nothing kept, no cutoff, no reason.
    let (kept, decision) =
        truncate_sorted_by_score(&[], |score: &f64| *score, &ScoreTruncationConfig::default());
    assert_eq!(kept, Vec::<f64>::new());
    assert_eq!(decision.initial_count, 0);
    assert_eq!(decision.kept_count, 0);
    assert_eq!(decision.cutoff_index, None);
    assert_eq!(decision.reason, "");

    // min_score: 0.3 is the first score below 0.35 => two kept.
    let config = ScoreTruncationConfig {
        min_score: Some(0.35),
        ..ScoreTruncationConfig::default()
    };
    let (kept, decision) = truncate_sorted_by_score(&[0.9, 0.5, 0.3, 0.2], |s| *s, &config);
    assert_eq!(kept, vec![0.9, 0.5]);
    assert_eq!(decision.kept_count, 2);
    assert_eq!(decision.cutoff_index, Some(2));
    assert_eq!(decision.reason, REASON_MIN_SCORE);

    // relative_min_score: half of the best score (1.0) keeps 1.0 and 0.5.
    let config = ScoreTruncationConfig {
        relative_min_score: Some(0.5),
        ..ScoreTruncationConfig::default()
    };
    let (kept, decision) = truncate_sorted_by_score(&[1.0, 0.5, 0.4], |s| *s, &config);
    assert_eq!(kept, vec![1.0, 0.5]);
    assert_eq!(decision.reason, REASON_RELATIVE_MIN_SCORE);

    // autocut: one drop of at least 50% cuts right after it.
    let config = ScoreTruncationConfig {
        autocut_jumps: 1,
        autocut_min_relative_drop: 0.5,
        ..ScoreTruncationConfig::default()
    };
    let (kept, decision) = truncate_sorted_by_score(&[1.0, 0.5, 0.49, 0.48], |s| *s, &config);
    assert_eq!(kept, vec![1.0]);
    assert_eq!(decision.reason, REASON_AUTOCUT);
    assert_eq!(decision.cutoff_index, Some(1));

    // Budgets: character safe, never splitting a multi-byte character.
    assert_eq!(truncate_chars("能力树索引", 3), ("能力树".to_owned(), true));
    assert_eq!(
        truncate_chars("能力树索引", 0),
        ("能力树索引".to_owned(), false)
    );
    assert_eq!(
        truncate_chars_with_ellipsis("能力树索引构建", 5),
        ("能力...".to_owned(), true)
    );
    let items = vec![1, 2, 3, 4];
    assert_eq!(take_budget(items.clone(), 2), (vec![1, 2], true));
    assert_eq!(take_budget(items, 0), (vec![1, 2, 3, 4], false));
}

/// 10. The value objects, the configs and the codebook interoperate as the later waves expect.
#[test]
fn the_foundation_types_compose() {
    let index = LexicalIndex::from_loaded_index(&loaded_index());
    let cursor = SearchCursor::new(&loaded_index().root, 5);
    assert_eq!(cursor.node_id, ROOT_NODE_ID);
    assert_eq!(cursor.branch_path, vec![ROOT_NODE_ID.to_owned()]);
    assert_eq!(cursor.top_k, 5);
    assert_eq!(
        cursor.resolve(&loaded_index().root).unwrap().node_id,
        ROOT_NODE_ID
    );

    let fragment = ExposedFragment::default();
    assert!(fragment.is_empty());
    assert_eq!(fragment.selectable_count(), 0);
    assert!(fragment.resolution("1").is_none());

    assert_eq!(SelectableTarget::default().code(), "");
    assert!(!SelectionResult::default().is_abstain);
    assert_eq!(ABSTAIN_TOKEN, "0");

    let config = ProgressiveSearchConfig::default();
    assert_eq!(config.top_k, 5);
    assert_eq!(config.max_tokens, 96);
    assert_eq!(config.selection_mode, SelectionMode::Generate);
    assert!(config.disclosure.flatten_full_tree_in_prompt);
    assert_eq!(config.embedding, EmbeddingScoringConfig::default());
    assert_eq!(DisclosureConfig::default().exposure_threshold, 12);
    assert_eq!(DEFAULT_TOP_K, 10);
    config.validate().unwrap();

    let codebook = CompactCodebook::new(["AA".to_owned(), "AB".to_owned()]).unwrap();
    assert_eq!(codebook.prefix(2).unwrap(), vec!["AA", "AB"]);
    assert!(codebook.prefix(3).is_err());
    assert!(CompactCodebook::new([ABSTAIN_TOKEN.to_owned()]).is_err());

    // The matcher superset used by the runtime reports its alternatives.
    let matcher = compile_matcher("alpha|beta", true, false).unwrap();
    assert!(matcher.is_literal());
    assert_eq!(matcher.alternatives(), ["alpha", "beta"]);
    assert!(matcher.matches("ALPHA"));
    assert_eq!(tokenize("Alpha-Skill"), vec!["alpha", "skill"]);

    // The index is queryable end to end (the shape the facade will consume).
    assert_ne!(index.search("skill").unwrap(), vec![]);
    assert!(index
        .missing_terms("skill zzznotpresent")
        .contains(&"zzznotpresent".to_owned()));
}
