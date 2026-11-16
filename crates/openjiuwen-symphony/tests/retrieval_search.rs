//! Integration coverage for the query-time search service facade and the layers under it.
//!
//! Every scenario is offline and deterministic: an index directory written with
//! `artifacts::write_index`, hand-built [`RetrieverNode`] trees, the retrieval crate's
//! `HashEmbedding` for the blend path, and `StaticCompletionModel` for scripted replies.
//! No clock, no network, no hash-map iteration order — identical inputs must produce
//! identical ids, summary lines, trace event sequences and candidate rows.
//!
//! # Coverage map
//!
//! | case | subject |
//! | --- | --- |
//! | 01 | index loading, `search`, `search_details` and the summary-line shape |
//! | 02 | corrupt/missing artifacts and an empty index |
//! | 03 | the progressive path with a scripted model |
//! | 04 | the lexical fallback with no model |
//! | 05 | `FallbackMode::Error` without a model |
//! | 06 | an unusable model reply |
//! | 07 | trace event sequence, detail keys and the single-selectable shortcut |
//! | 08 | `prepare()` warming |
//! | 09 | a resumed walk versus an uninterrupted search |
//! | 10 | the `max_steps`/`max_depth`/`max_branch_choices` bounds |
//! | 11 | the flat one-shot retriever and its embedding blend |
//! | 12 | field-weighted ranking and CJK snippets |
//! | 13 | the per-request `top_k` |
//! | 14 | identical reruns |
//! | 15 | public identity (worker ids, never cids) |
//! | 16 | the config translation |
//! | 17 | validation and accessors |
#![allow(clippy::unwrap_used, clippy::expect_used)]
#![allow(clippy::float_cmp, clippy::assert_is_empty)]
#![allow(clippy::too_many_lines)]

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use openjiuwen_retrieval::{Embedding, HashEmbedding};
use openjiuwen_symphony::error::{is_unsupported, ErrorKind};
use openjiuwen_symphony::llm::{FallbackMode, StaticCompletionModel};
use openjiuwen_symphony::retrieval::artifacts::{catalog_content_hash, write_index};
use openjiuwen_symphony::retrieval::search::config::{
    DisclosureConfig, EmbeddingScoringConfig, ProgressiveSearchConfig, SelectionMode, DEFAULT_TOP_K,
};
use openjiuwen_symphony::retrieval::search::engine::{
    FRAGMENT_BUILT_EVENT, FRAGMENT_CONTINUE_EVENT, REDUCE_COMPLETE_EVENT, SEARCH_COMPLETE_EVENT,
    SEARCH_TRUNCATED_EVENT,
};
use openjiuwen_symphony::retrieval::search::flat::{FlatRetriever, METHOD_FLAT_LEXICAL};
use openjiuwen_symphony::retrieval::search::flow::{
    ChatMessage, SelectionProvenance, SelectionResult,
};
use openjiuwen_symphony::retrieval::search::lexical::{LexicalDocument, LexicalIndex};
use openjiuwen_symphony::retrieval::search::progressive::{
    ProgressiveRetriever, METHOD_LEXICAL_FALLBACK, METHOD_PROGRESSIVE,
};
use openjiuwen_symphony::retrieval::search::service::{
    runtime_config_from_config, GenerationConfig, RenderConfig, Retriever, RetrieverConfig,
    SearchRequest, TraversalConfig,
};
use openjiuwen_symphony::retrieval::search::{
    trace_degraded_selections, trace_selection_provenance, FRAGMENT_SELECTED_EVENT, MODE_LEXICAL,
    MODE_LEXICAL_NO_MODEL,
};
use openjiuwen_symphony::retrieval::types::{
    CatalogRecord, IndexManifest, IndexNode, LoadedIndex, RetrieverItem, RetrieverNode,
    CATALOG_FILENAME, MANIFEST_FILENAME, MODE_FULL, ROOT_NODE_ID,
};

/// The leaf `cid` of the alpha writer item.
const ALPHA_WRITER: &str = "Skills.Alpha.Writing.aw-one";
/// The leaf `cid` of the alpha tester item.
const ALPHA_TESTER: &str = "Skills.Alpha.Writing.at-two";
/// The leaf `cid` of the beta compute item.
const BETA_COMPUTE: &str = "Skills.Beta.bc-one";

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// The four-record, two-branch catalog the directory fixtures write.
///
/// ```text
/// Skills.Data      alpha  Alpha Skill      "capability that summarises alpha reports"
///                  beta   Beta Skill       "capability that consumes alpha feeds"
/// Skills.Writing   gamma  Data Flow Guide  "capability that explains a data flow"
///                  delta  Delta Writer     "capability that writes delta prose"
/// ```
fn fixture_records() -> Vec<CatalogRecord> {
    vec![
        CatalogRecord::new("alpha", "Skills.Data.alpha")
            .with_name("Alpha Skill")
            .with_description("capability that summarises alpha reports")
            .with_retrieval_text("alpha skill summarises alpha reports"),
        CatalogRecord::new("beta", "Skills.Data.beta")
            .with_name("Beta Skill")
            .with_description("capability that consumes alpha feeds")
            .with_retrieval_text("beta skill consumes alpha feeds"),
        CatalogRecord::new("gamma", "Skills.Writing.gamma")
            .with_name("Data Flow Guide")
            .with_description("capability that explains a data flow")
            .with_retrieval_text("data flow guide explains a data flow"),
        CatalogRecord::new("delta", "Skills.Writing.delta")
            .with_name("Delta Writer")
            .with_description("capability that writes delta prose")
            .with_retrieval_text("delta writer writes delta prose"),
    ]
}

/// The tree nodes matching [`fixture_records`].
fn fixture_nodes() -> Vec<IndexNode> {
    vec![
        IndexNode::branch("Skills").with_description("every skill"),
        IndexNode::branch("Skills.Data").with_description("data skills"),
        IndexNode::branch("Skills.Writing").with_description("writing skills"),
        IndexNode::leaf("Skills.Data.alpha", "alpha"),
        IndexNode::leaf("Skills.Data.beta", "beta"),
        IndexNode::leaf("Skills.Writing.gamma", "gamma"),
        IndexNode::leaf("Skills.Writing.delta", "delta"),
    ]
}

/// The manifest describing [`fixture_records`].
fn fixture_manifest(records: &[CatalogRecord]) -> IndexManifest {
    IndexManifest {
        mode: MODE_FULL.to_owned(),
        count: records.len(),
        item_paths: records
            .iter()
            .map(|record| format!("store/{}/SKILL.md", record.worker_id))
            .collect(),
        worker_ids: records
            .iter()
            .map(|record| record.worker_id.clone())
            .collect(),
        item_type: "skill".to_owned(),
        content_hash: catalog_content_hash(records),
    }
}

/// Writes the four-record fixture index into `dir`.
async fn write_fixture(dir: &Path) {
    let records = fixture_records();
    write_index(dir, &fixture_nodes(), &records, &fixture_manifest(&records))
        .await
        .unwrap();
}

/// A loaded index over `root` and `records` (the manifest is not consulted by a search).
fn loaded(root: RetrieverNode, records: Vec<CatalogRecord>) -> LoadedIndex {
    LoadedIndex {
        choices: root.flatten_choices(),
        root,
        records,
        manifest: IndexManifest::default(),
        item_paths: BTreeMap::new(),
    }
}

/// The three-level tree the model-path cases walk.
///
/// ```text
/// ROOT
/// ├── Skills.Alpha                      (branch)
/// │   └── Skills.Alpha.Writing          (branch)
/// │       ├── Alpha Writer             ("writes alpha documentation")
/// │       └── Alpha Tester             ("tests alpha behaviour")
/// └── Skills.Beta                       (branch)
///     └── Beta Compute                  ("computes beta totals")
/// ```
fn model_tree() -> RetrieverNode {
    RetrieverNode::new(ROOT_NODE_ID, ROOT_NODE_ID).with_children(vec![
        RetrieverNode::new("Skills.Alpha", "Alpha")
            .with_description("alpha skills")
            .with_children(vec![RetrieverNode::new(
                "Skills.Alpha.Writing",
                "Alpha Writing",
            )
            .with_description("alpha writing skills")
            .with_items(vec![
                RetrieverItem::new(ALPHA_WRITER, ALPHA_WRITER)
                    .with_label("Alpha Writer")
                    .with_description("writes alpha documentation"),
                RetrieverItem::new(ALPHA_TESTER, ALPHA_TESTER)
                    .with_label("Alpha Tester")
                    .with_description("tests alpha behaviour"),
            ])]),
        RetrieverNode::new("Skills.Beta", "Beta")
            .with_description("beta skills")
            .with_items(vec![RetrieverItem::new(BETA_COMPUTE, BETA_COMPUTE)
                .with_label("Beta Compute")
                .with_description("computes beta totals")]),
    ])
}

/// The catalog records [`model_tree`]'s payloads resolve to.
fn model_records() -> Vec<CatalogRecord> {
    vec![
        CatalogRecord::new("alpha-writer", ALPHA_WRITER)
            .with_name("Alpha Writer")
            .with_description("writes alpha documentation"),
        CatalogRecord::new("alpha-tester", ALPHA_TESTER)
            .with_name("Alpha Tester")
            .with_description("tests alpha behaviour"),
        CatalogRecord::new("beta-compute", BETA_COMPUTE)
            .with_name("Beta Compute")
            .with_description("computes beta totals"),
    ]
}

/// A level-by-level render config: one exposure per walk level, numeric compact codes.
fn level_render() -> RenderConfig {
    RenderConfig {
        compact_codes_enabled: true,
        compact_codebook: Vec::new(),
        flatten_tree: false,
        max_exposure_depth: 0,
        exposure_threshold: 100,
    }
}

/// A facade config that walks [`model_tree`] one level at a time.
fn level_config() -> RetrieverConfig {
    RetrieverConfig {
        top_k: 3,
        render: level_render(),
        ..RetrieverConfig::default()
    }
}

/// The facade over [`model_tree`] with [`level_config`].
async fn model_retriever(config: RetrieverConfig) -> Retriever {
    Retriever::from_loaded_index(loaded(model_tree(), model_records()), config)
        .await
        .unwrap()
}

/// An engine config that walks one level at a time.
fn engine_config() -> ProgressiveSearchConfig {
    ProgressiveSearchConfig {
        top_k: 3,
        disclosure: DisclosureConfig {
            max_exposure_depth_per_call: 0,
            exposure_threshold: 100,
            compact_codes_enabled: true,
            ..DisclosureConfig::default()
        },
        ..ProgressiveSearchConfig::default()
    }
}

/// An engine-level retriever over [`model_tree`].
fn engine_retriever(config: ProgressiveSearchConfig) -> ProgressiveRetriever {
    ProgressiveRetriever::new(Arc::new(model_tree()), config).unwrap()
}

/// The item ids of a `ProgressiveSearchResult`.
fn candidate_ids(
    result: &openjiuwen_symphony::retrieval::search::ProgressiveSearchResult,
) -> Vec<&str> {
    result
        .candidates
        .iter()
        .map(|candidate| candidate.item_id.as_str())
        .collect()
}

/// The event types of a trace.
fn event_types(trace: &openjiuwen_symphony::retrieval::types::RetrieverTrace) -> Vec<&str> {
    trace
        .events()
        .iter()
        .map(|event| event.event_type.as_str())
        .collect()
}

// ---------------------------------------------------------------------------
// 01–02: directory loading, results and artifacts
// ---------------------------------------------------------------------------

#[tokio::test]
async fn case_01_loads_an_index_and_publicizes_its_results() {
    let dir = tempfile::tempdir().unwrap();
    write_fixture(dir.path()).await;
    let retriever = Retriever::load(dir.path(), RetrieverConfig::default())
        .await
        .unwrap();
    assert_eq!(retriever.index().records.len(), 4);
    assert_eq!(retriever.config().top_k, DEFAULT_TOP_K);
    assert_eq!(retriever.method(), METHOD_LEXICAL_FALLBACK);

    let ids = retriever.search("alpha").await.unwrap();
    assert_eq!(ids, vec!["alpha", "beta"], "best first, public ids only");

    let details = retriever.search_details("alpha").await.unwrap();
    assert_eq!(details.item_ids, ids);
    assert_eq!(details.method, METHOD_LEXICAL_FALLBACK);
    assert_eq!(details.steps, 1, "a flattened exposure is one visit");
    assert_eq!(details.selected_item_id.as_deref(), Some("alpha"));
    assert_eq!(details.selected_rank, Some(1));
    assert_eq!(
        details.summary_lines,
        vec![
            "1. alpha (source=lexical_fallback)".to_owned(),
            "2. beta (source=lexical_fallback)".to_owned(),
        ]
    );
    assert!(!details.trace_events.is_empty());

    let first = &details.candidates[0];
    assert_eq!(first.rank, 1);
    assert!(first.selected);
    assert_eq!(first.worker_id, "alpha");
    assert_eq!(first.cid, "Skills.Data.alpha");
    assert_eq!(first.name, "Alpha Skill");
    assert_eq!(
        first.description,
        "capability that summarises alpha reports"
    );
    assert_eq!(
        first.branch_path,
        vec!["Skills".to_owned(), "Data".to_owned()],
        "the catalog stores the parent cid's dot-separated segments"
    );
    assert_eq!(first.category, "Skills.Data");
    assert_eq!(
        first.item_id, "Alpha Skill",
        "the tree item id is the record name"
    );
    let second = &details.candidates[1];
    assert_eq!(second.rank, 2);
    assert!(!second.selected);
    assert_eq!(second.worker_id, "beta");
}

#[tokio::test]
async fn case_02_reports_broken_artifacts_and_accepts_an_empty_index() {
    // A directory without `catalog.jsonl` is a NotFound naming that file.
    let missing = tempfile::tempdir().unwrap();
    write_fixture(missing.path()).await;
    std::fs::remove_file(missing.path().join(CATALOG_FILENAME)).unwrap();
    let error = Retriever::load(missing.path(), RetrieverConfig::default())
        .await
        .expect_err("a missing catalog must fail");
    assert_eq!(error.kind(), ErrorKind::NotFound);
    assert_eq!(error.message(), CATALOG_FILENAME);

    // A manifest whose count disagrees with the catalog is a validation error.
    let inconsistent = tempfile::tempdir().unwrap();
    write_fixture(inconsistent.path()).await;
    let manifest_path = inconsistent.path().join(MANIFEST_FILENAME);
    let mut manifest: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&manifest_path).unwrap()).unwrap();
    manifest["count"] = serde_json::json!(5);
    std::fs::write(&manifest_path, serde_json::to_string(&manifest).unwrap()).unwrap();
    let error = Retriever::load(inconsistent.path(), RetrieverConfig::default())
        .await
        .expect_err("an inconsistent manifest must fail");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(
        error.message().starts_with(MANIFEST_FILENAME),
        "{}",
        error.message()
    );

    // An index whose catalog holds no records is a valid index that returns nothing.
    // (`nodes:` without entries is not a YAML sequence, so the tree keeps one branch.)
    let empty = tempfile::tempdir().unwrap();
    write_index(
        empty.path(),
        &[IndexNode::new("Skills.Empty", "branch")],
        &[],
        &IndexManifest {
            mode: MODE_FULL.to_owned(),
            item_type: "skill".to_owned(),
            ..IndexManifest::default()
        },
    )
    .await
    .unwrap();
    let retriever = Retriever::load(empty.path(), RetrieverConfig::default())
        .await
        .unwrap();
    assert!(retriever.search("alpha").await.unwrap().is_empty());
    let details = retriever.search_details("alpha").await.unwrap();
    assert_eq!(details.selected_item_id, None);
    assert_eq!(details.selected_rank, None);
    assert!(details.candidates.is_empty());
    assert!(details.summary_lines.is_empty());
    assert_eq!(
        details
            .trace_events
            .last()
            .map(|event| event.event_type.as_str()),
        Some(SEARCH_COMPLETE_EVENT)
    );
}

// ---------------------------------------------------------------------------
// 03–06: the progressive path, fallback modes and unusable replies
// ---------------------------------------------------------------------------

#[tokio::test]
async fn case_03_a_scripted_model_drives_two_descent_steps() {
    // ROOT exposes Skills.Alpha (1) and Skills.Beta (2); Alpha exposes one branch, which the
    // single-selectable shortcut takes without a model call; Writing exposes both items.
    let model = Arc::new(StaticCompletionModel::new(vec![
        "1".to_owned(),
        "1\n2".to_owned(),
    ]));
    let retriever = model_retriever(level_config())
        .await
        .with_model(model.clone(), "scripted");
    assert_eq!(retriever.method(), METHOD_PROGRESSIVE);

    let details = retriever.search_details("alpha").await.unwrap();
    assert_eq!(details.method, METHOD_PROGRESSIVE);
    assert_eq!(
        details.item_ids,
        vec!["alpha-tester", "alpha-writer"],
        "codes follow the item-id order the exposure assigns"
    );
    assert_eq!(details.steps, 3, "root, Alpha and Writing are each visited");
    assert_eq!(model.call_count(), 2, "the shortcut costs no model call");
    assert_eq!(details.candidates[0].worker_id, "alpha-tester");
    assert_eq!(details.candidates[0].cid, ALPHA_TESTER);
    assert_eq!(
        details.candidates[0].branch_path,
        vec![
            "Skills".to_owned(),
            "Alpha".to_owned(),
            "Writing".to_owned()
        ]
    );
    assert_eq!(details.candidates[0].category, "Skills.Alpha.Writing");
    assert_eq!(details.selected_item_id.as_deref(), Some("alpha-tester"));
    assert_eq!(
        details.summary_lines[0],
        "1. alpha-tester (source=progressive)"
    );
    assert!(event_types(&retriever_probe(&details))
        .contains(&openjiuwen_symphony::retrieval::search::LLM_REQUEST_EVENT));
}

/// The trace of a [`openjiuwen_symphony::retrieval::search::RetrievalResult`], for
/// `event_types`.
fn retriever_probe(
    details: &openjiuwen_symphony::retrieval::search::RetrievalResult,
) -> openjiuwen_symphony::retrieval::types::RetrieverTrace {
    openjiuwen_symphony::retrieval::types::RetrieverTrace {
        events: details.trace_events.clone(),
    }
}

#[tokio::test]
async fn case_04_without_a_model_the_walk_is_the_lexical_fallback() {
    let retriever = model_retriever(level_config()).await;
    assert_eq!(retriever.method(), METHOD_LEXICAL_FALLBACK);
    let details = retriever.search_details("alpha writing").await.unwrap();
    assert_eq!(details.method, METHOD_LEXICAL_FALLBACK);
    assert_eq!(
        details.item_ids,
        vec!["alpha-writer", "alpha-tester"],
        "the pinned lexical order"
    );
    assert_eq!(details.steps, 3);
    assert!(
        !retriever_probe(&details)
            .events()
            .iter()
            .any(|event| event.event_type
                == openjiuwen_symphony::retrieval::search::LLM_REQUEST_EVENT)
    );

    // A second run over the same retriever is identical.
    let again = retriever.search_details("alpha writing").await.unwrap();
    assert_eq!(again.item_ids, details.item_ids);
}

#[tokio::test]
async fn case_05_error_fallback_without_a_model_is_unsupported() {
    let dir = tempfile::tempdir().unwrap();
    write_fixture(dir.path()).await;
    let config = RetrieverConfig {
        generation: GenerationConfig {
            fallback_mode: FallbackMode::Error,
            ..GenerationConfig::default()
        },
        ..RetrieverConfig::default()
    };
    let retriever = Retriever::load(dir.path(), config).await.unwrap();
    let error = retriever
        .search("alpha")
        .await
        .expect_err("generate mode without a model must not select");
    assert!(is_unsupported(&error), "{error}");

    // The same config in lexical mode selects without a model.
    let lexical = RetrieverConfig {
        generation: GenerationConfig {
            mode: SelectionMode::Lexical,
            fallback_mode: FallbackMode::Error,
            ..GenerationConfig::default()
        },
        ..RetrieverConfig::default()
    };
    let retriever = Retriever::load(dir.path(), lexical).await.unwrap();
    assert_eq!(retriever.search("alpha").await.unwrap()[0], "alpha");
}

#[tokio::test]
async fn case_06_an_unusable_model_reply_selects_nothing_and_never_panics() {
    // The root takes both branches, Writing's reply is garbage and Beta's shortcut still
    // yields its item.
    let model = Arc::new(StaticCompletionModel::new(vec![
        "1\n2".to_owned(),
        "garbage !! not a code".to_owned(),
    ]));
    let retriever = model_retriever(level_config())
        .await
        .with_model(model.clone(), "scripted");
    let details = retriever.search_details("alpha").await.unwrap();
    assert_eq!(details.item_ids, vec!["beta-compute"]);
    assert_eq!(details.method, METHOD_PROGRESSIVE);

    // The unusable reply is recorded as a generate selection with no code.
    let failed = details
        .trace_events
        .iter()
        .find(|event| {
            event.event_type == FRAGMENT_SELECTED_EVENT && event.node_id == "Skills.Alpha.Writing"
        })
        .expect("Writing must record a selection");
    assert_eq!(failed.detail["mode"], serde_json::json!("generate"));
    assert_eq!(failed.detail["selected_codes"], serde_json::json!([]));
    assert_eq!(
        failed.detail["raw_output"],
        serde_json::json!("garbage !! not a code")
    );
}

// ---------------------------------------------------------------------------
// 07–10: the trace, warm-up, resumability and the traversal bounds
// ---------------------------------------------------------------------------

/// The `(event_type, node_id)` pairs of a trace.
fn event_pairs(
    trace: &openjiuwen_symphony::retrieval::types::RetrieverTrace,
) -> Vec<(String, String)> {
    trace
        .events()
        .iter()
        .map(|event| (event.event_type.clone(), event.node_id.clone()))
        .collect()
}

/// A tree whose every level has exactly one selectable target.
///
/// ```text
/// ROOT
/// └── Only
///     └── One
/// ```
fn single_chain_tree() -> RetrieverNode {
    RetrieverNode::new(ROOT_NODE_ID, ROOT_NODE_ID).with_children(vec![RetrieverNode::new(
        "Only", "Only",
    )
    .with_children(vec![RetrieverNode::new("One", "One").with_items(vec![
        RetrieverItem::new("one", "one")
            .with_label("Only Item")
            .with_description("the single item"),
    ])])])
}

#[tokio::test]
async fn case_07_the_trace_records_a_depth_first_walk_with_pinned_detail_keys() {
    // The model takes both root branches and both Writing items, so every node is visited.
    let model = Arc::new(StaticCompletionModel::new(vec![
        "1\n2".to_owned(),
        "1\n2".to_owned(),
    ]));
    let retriever = model_retriever(level_config())
        .await
        .with_model(model.clone(), "scripted");
    let details = retriever.search_details("alpha").await.unwrap();
    let trace = retriever_probe(&details);
    let pairs = event_pairs(&trace);
    let expected: Vec<(&str, &str)> = vec![
        (FRAGMENT_BUILT_EVENT, ROOT_NODE_ID),
        (
            openjiuwen_symphony::retrieval::search::LLM_REQUEST_EVENT,
            ROOT_NODE_ID,
        ),
        (
            openjiuwen_symphony::retrieval::search::LLM_RESPONSE_EVENT,
            ROOT_NODE_ID,
        ),
        (FRAGMENT_SELECTED_EVENT, ROOT_NODE_ID),
        (FRAGMENT_CONTINUE_EVENT, ROOT_NODE_ID),
        (FRAGMENT_BUILT_EVENT, "Skills.Alpha"),
        (FRAGMENT_SELECTED_EVENT, "Skills.Alpha"),
        (FRAGMENT_CONTINUE_EVENT, "Skills.Alpha"),
        (FRAGMENT_BUILT_EVENT, "Skills.Alpha.Writing"),
        (
            openjiuwen_symphony::retrieval::search::LLM_REQUEST_EVENT,
            "Skills.Alpha.Writing",
        ),
        (
            openjiuwen_symphony::retrieval::search::LLM_RESPONSE_EVENT,
            "Skills.Alpha.Writing",
        ),
        (FRAGMENT_SELECTED_EVENT, "Skills.Alpha.Writing"),
        (FRAGMENT_CONTINUE_EVENT, "Skills.Alpha.Writing"),
        (REDUCE_COMPLETE_EVENT, "Skills.Alpha.Writing"),
        (REDUCE_COMPLETE_EVENT, "Skills.Alpha"),
        (FRAGMENT_BUILT_EVENT, "Skills.Beta"),
        (FRAGMENT_SELECTED_EVENT, "Skills.Beta"),
        (FRAGMENT_CONTINUE_EVENT, "Skills.Beta"),
        (REDUCE_COMPLETE_EVENT, "Skills.Beta"),
        (REDUCE_COMPLETE_EVENT, ROOT_NODE_ID),
        (SEARCH_COMPLETE_EVENT, ROOT_NODE_ID),
    ];
    let expected: Vec<(String, String)> = expected
        .into_iter()
        .map(|(event, node)| (event.to_owned(), node.to_owned()))
        .collect();
    assert_eq!(pairs, expected);

    // The per-event detail keys, one event of each kind.
    let built = trace
        .events()
        .iter()
        .find(|event| event.event_type == FRAGMENT_BUILT_EVENT)
        .unwrap();
    assert!(built.detail.contains_key("selectable_count"));
    assert!(built.detail.contains_key("rendered_tree"));
    let selected = trace
        .events()
        .iter()
        .find(|event| event.event_type == FRAGMENT_SELECTED_EVENT)
        .unwrap();
    for key in [
        "mode",
        "raw_output",
        "selected_codes",
        "selected_canonical_ids",
    ] {
        assert!(selected.detail.contains_key(key), "missing {key}");
    }
    let continued = trace
        .events()
        .iter()
        .find(|event| event.event_type == FRAGMENT_CONTINUE_EVENT)
        .unwrap();
    for key in [
        "leaf_count",
        "branch_count",
        "selected_terminal_ids",
        "selected_branch_ids",
    ] {
        assert!(continued.detail.contains_key(key), "missing {key}");
    }
    let reduced = trace
        .events()
        .iter()
        .find(|event| event.event_type == REDUCE_COMPLETE_EVENT)
        .unwrap();
    assert!(reduced.detail.contains_key("input_candidates"));
    assert!(reduced.detail.contains_key("output_candidates"));
    let complete = trace.events().last().unwrap();
    assert_eq!(complete.event_type, SEARCH_COMPLETE_EVENT);
    assert!(complete.detail.contains_key("steps"));
    assert!(complete.detail.contains_key("candidates"));

    // Alpha's single selectable branch is taken without asking the model.
    let shortcut = trace
        .events()
        .iter()
        .find(|event| {
            event.event_type == FRAGMENT_SELECTED_EVENT && event.node_id == "Skills.Alpha"
        })
        .unwrap();
    assert_eq!(
        shortcut.detail["mode"],
        serde_json::json!(
            openjiuwen_symphony::retrieval::search::engine::MODE_SINGLE_SELECTABLE_SHORTCUT
        )
    );

    // An all-shortcut walk never prompts the model at all.
    let model = Arc::new(StaticCompletionModel::new(vec!["1".to_owned()]));
    let retriever = Retriever::from_loaded_index(
        loaded(
            single_chain_tree(),
            vec![CatalogRecord::new("one", "one").with_name("Only Item")],
        ),
        level_config(),
    )
    .await
    .unwrap()
    .with_model(model.clone(), "scripted");
    let details = retriever.search_details("item").await.unwrap();
    assert_eq!(details.item_ids, vec!["one"]);
    assert_eq!(model.call_count(), 0, "the shortcut costs no prompt");
    assert!(model.prompts().is_empty());
    let selection_modes: Vec<&serde_json::Value> = details
        .trace_events
        .iter()
        .filter(|event| event.event_type == FRAGMENT_SELECTED_EVENT)
        .map(|event| &event.detail["mode"])
        .collect();
    assert_eq!(
        selection_modes,
        vec![
            &serde_json::json!(
                openjiuwen_symphony::retrieval::search::engine::MODE_SINGLE_SELECTABLE_SHORTCUT
            ),
            &serde_json::json!(
                openjiuwen_symphony::retrieval::search::engine::MODE_SINGLE_SELECTABLE_SHORTCUT
            ),
            &serde_json::json!(
                openjiuwen_symphony::retrieval::search::engine::MODE_SINGLE_SELECTABLE_SHORTCUT
            ),
        ],
        "the unary chain is taken level by level without a prompt"
    );
    assert_eq!(details.method, METHOD_LEXICAL_FALLBACK);
}

#[tokio::test]
async fn case_08_prepare_warms_the_fragments_and_keeps_the_result() {
    let warm_dir = tempfile::tempdir().unwrap();
    write_fixture(warm_dir.path()).await;
    let warm = Retriever::load(warm_dir.path(), RetrieverConfig::default())
        .await
        .unwrap();
    assert_eq!(warm.warmed_fragments(), 0);
    let prepared = warm.prepare().await.unwrap();
    assert!(prepared >= 1, "every reachable node is warmed");
    assert_eq!(warm.warmed_fragments(), prepared);

    let cold_dir = tempfile::tempdir().unwrap();
    write_fixture(cold_dir.path()).await;
    let cold = Retriever::load(cold_dir.path(), RetrieverConfig::default())
        .await
        .unwrap();
    let warm_result = warm.search_details("capability").await.unwrap();
    let cold_result = cold.search_details("capability").await.unwrap();
    assert_eq!(warm_result.item_ids, cold_result.item_ids);
    assert_eq!(warm_result.candidates, cold_result.candidates);
    assert_eq!(
        event_types(&retriever_probe(&warm_result)),
        event_types(&retriever_probe(&cold_result))
    );
}

#[tokio::test]
async fn case_09_a_resumed_walk_matches_an_uninterrupted_search() {
    let model = Arc::new(StaticCompletionModel::new(vec![
        "1\n2".to_owned(),
        "1\n2".to_owned(),
    ]));
    let retriever = engine_retriever(engine_config()).with_model(model.clone(), "scripted");
    let messages = vec![ChatMessage::user("alpha skills")];
    let uninterrupted = retriever.search_messages(&messages, Some(3)).await.unwrap();

    // A fresh walk, stopped after two visits, serialized and resumed.
    let resumable = engine_retriever(engine_config()).with_model(
        Arc::new(StaticCompletionModel::new(vec!["1\n2".to_owned()])),
        "scripted",
    );
    let mut state = resumable.begin(messages.clone(), Some(3)).unwrap();
    let first = resumable.step(&mut state).await.unwrap();
    assert_eq!(
        first.map(|step| step.node_id),
        Some(ROOT_NODE_ID.to_owned())
    );
    let second = resumable.step(&mut state).await.unwrap();
    assert!(second.is_some());
    let encoded = serde_json::to_string(&state).unwrap();
    let mut state: openjiuwen_symphony::retrieval::search::engine::TraversalState =
        serde_json::from_str(&encoded).unwrap();
    while resumable.step(&mut state).await.unwrap().is_some() {}
    let resumed = resumable.finish(state);

    assert_eq!(candidate_ids(&resumed), candidate_ids(&uninterrupted));
    assert_eq!(resumed.candidates, uninterrupted.candidates);
    // The resumed walk closes no `search_complete`; everything before it matches.
    let mut expected = event_types(&uninterrupted.trace);
    expected.pop();
    assert_eq!(event_types(&resumed.trace), expected);
}

#[tokio::test]
async fn case_10_the_traversal_bounds_truncate_deterministically() {
    // A root that carries its own items plus one child branch: the first visit discovers the
    // root's items, and a spent step budget keeps them.
    let tree = RetrieverNode::new(ROOT_NODE_ID, ROOT_NODE_ID)
        .with_items(vec![
            RetrieverItem::new("root.one", "root.one")
                .with_label("Root One")
                .with_description("root skill one"),
            RetrieverItem::new("root.two", "root.two")
                .with_label("Root Two")
                .with_description("root skill two"),
        ])
        .with_children(vec![RetrieverNode::new("Branch", "Branch")
            .with_description("branch skills")
            .with_items(vec![RetrieverItem::new("branch.one", "branch.one")
                .with_label("Branch One")
                .with_description("branch skill one")])]);
    let truncated = ProgressiveRetriever::new(
        Arc::new(tree.clone()),
        ProgressiveSearchConfig {
            max_steps: 1,
            ..engine_config()
        },
    )
    .unwrap();
    let result = truncated.search("skill", Some(5)).await.unwrap();
    assert_eq!(
        candidate_ids(&result),
        vec!["root.one", "root.two"],
        "the discovered leaves survive the truncation"
    );
    assert!(result.trace.event_types().contains(&SEARCH_TRUNCATED_EVENT));
    let truncated_event = result
        .trace
        .events()
        .iter()
        .find(|event| event.event_type == SEARCH_TRUNCATED_EVENT)
        .unwrap();
    assert_eq!(truncated_event.detail["max_steps"], serde_json::json!(1));
    assert_eq!(
        event_types(&result.trace).last().copied(),
        Some(SEARCH_COMPLETE_EVENT)
    );

    // Without the bound the same walk reaches the child branch.
    let unbounded = ProgressiveRetriever::new(Arc::new(tree), engine_config()).unwrap();
    let complete = unbounded.search("skill", Some(5)).await.unwrap();
    assert_eq!(complete.steps, 2);
    assert!(!complete
        .trace
        .event_types()
        .contains(&SEARCH_TRUNCATED_EVENT));
    assert_eq!(
        candidate_ids(&complete),
        vec!["root.one", "branch.one", "root.two"],
        "the root's own leaves round-robin against its child branch"
    );

    // A depth bound stops the walk before the grandchildren.
    let shallow = engine_retriever(ProgressiveSearchConfig {
        max_depth: 1,
        ..engine_config()
    });
    let shallow_result = shallow.search("alpha", Some(3)).await.unwrap();
    assert_eq!(shallow_result.steps, 2, "the root and Skills.Alpha only");
    assert!(shallow_result.candidates.is_empty());

    // A branch-choice bound keeps only the first selected branch.
    let capped = engine_retriever(ProgressiveSearchConfig {
        max_branch_choices: 1,
        ..engine_config()
    });
    let capped_result = capped.search("alpha skills", Some(3)).await.unwrap();
    assert_eq!(capped_result.steps, 3);
    assert_eq!(
        candidate_ids(&capped_result),
        vec![ALPHA_TESTER, ALPHA_WRITER]
    );
    let uncapped = engine_retriever(engine_config());
    let uncapped_result = uncapped.search("alpha skills", Some(3)).await.unwrap();
    assert_eq!(uncapped_result.steps, 4);
    assert_eq!(
        candidate_ids(&uncapped_result),
        vec![BETA_COMPUTE, ALPHA_TESTER, ALPHA_WRITER],
        "the completed child branches round-robin in completion order"
    );
}

// ---------------------------------------------------------------------------
// 11–12: the flat path, field weighting and CJK text
// ---------------------------------------------------------------------------

#[tokio::test]
async fn case_11_the_flat_retriever_ranks_the_whole_catalog_in_one_pass() {
    let dir = tempfile::tempdir().unwrap();
    write_fixture(dir.path()).await;
    let retriever = Retriever::load(dir.path(), RetrieverConfig::default())
        .await
        .unwrap();
    assert_eq!(retriever.flat().method(), METHOD_FLAT_LEXICAL);
    let plain = runtime_config_from_config(Some(&RetrieverConfig {
        top_k: 4,
        ..RetrieverConfig::default()
    }))
    .unwrap();
    let flat = FlatRetriever::new(plain.clone());
    let index = retriever.index();

    let ranked = flat.retrieve(&index.root, "alpha", Some(2)).await.unwrap();
    assert_eq!(ranked.method, METHOD_FLAT_LEXICAL);
    assert_eq!(
        ranked.ranked_keys,
        vec![
            "Skills.Data.alpha".to_owned(),
            "Skills.Data.beta".to_owned()
        ]
    );
    assert_eq!(ranked.candidates.len(), 2, "top_k bounds the list");
    assert_eq!(
        ranked.candidates[0].branch_path,
        vec!["Skills".to_owned(), "Skills.Data".to_owned()],
        "the flat path still reports each payload's branch path"
    );
    assert_eq!(
        ranked.candidates[0].payload, "Skills.Data.alpha",
        "the payload is the ranking identity"
    );

    let unlimited = flat
        .retrieve(&index.root, "capability", Some(0))
        .await
        .unwrap();
    assert_eq!(unlimited.candidates.len(), 4, "0 means unlimited");
    let bounded = flat
        .retrieve(&index.root, "capability", Some(1))
        .await
        .unwrap();
    assert_eq!(bounded.candidates.len(), 1);

    // The blend is deterministic and a zero weight reproduces the lexical order.
    let blended_config = runtime_config_from_config(Some(&RetrieverConfig {
        top_k: 4,
        embedding_weight: 0.5,
        ..RetrieverConfig::default()
    }))
    .unwrap();
    let embedder: Arc<dyn Embedding> = Arc::new(HashEmbedding::new(32));
    let blended = FlatRetriever::new(blended_config).with_embedding(Arc::clone(&embedder));
    assert_eq!(
        blended.method(),
        openjiuwen_symphony::retrieval::search::flat::METHOD_FLAT
    );
    let once = blended
        .retrieve(&index.root, "alpha", Some(0))
        .await
        .unwrap();
    let twice = blended
        .retrieve(&index.root, "alpha", Some(0))
        .await
        .unwrap();
    assert_eq!(once.ranked_keys, twice.ranked_keys);
    assert_eq!(once.candidates, twice.candidates);

    let zero_weight = runtime_config_from_config(Some(&RetrieverConfig {
        top_k: 4,
        embedding_weight: 0.0,
        ..RetrieverConfig::default()
    }))
    .unwrap();
    let lexical_only = FlatRetriever::new(zero_weight).with_embedding(embedder);
    let lexical_keys = lexical_only
        .retrieve(&index.root, "alpha", Some(0))
        .await
        .unwrap()
        .ranked_keys;
    let plain_keys = flat
        .retrieve(&index.root, "alpha", Some(0))
        .await
        .unwrap()
        .ranked_keys;
    assert_eq!(lexical_keys, plain_keys, "weight 0 is the lexical order");
}

/// A two-item catalog whose items match `alpha` through different fields.
///
/// The document name of an item is the `PascalCase` terminal segment of its item id, so the
/// name match is carried by the id and the description match by the description.
fn field_weight_tree() -> RetrieverNode {
    RetrieverNode::new(ROOT_NODE_ID, ROOT_NODE_ID).with_children(vec![RetrieverNode::new(
        "Skills", "Skills",
    )
    .with_items(vec![
        RetrieverItem::new("alpha-zulu", "alpha-zulu")
            .with_label("Zulu Skill")
            .with_description("explains the process"),
        RetrieverItem::new("aardvark", "aardvark")
            .with_label("Aardvark Notes")
            .with_description("reports on the alpha pipeline"),
    ])])
}

/// The catalog records of [`field_weight_tree`], publicized as `zulu` and `aardvark`.
fn field_weight_records() -> Vec<CatalogRecord> {
    vec![
        CatalogRecord::new("zulu", "alpha-zulu")
            .with_name("Zulu Skill")
            .with_description("explains the process"),
        CatalogRecord::new("aardvark", "aardvark")
            .with_name("Aardvark Notes")
            .with_description("reports on the alpha pipeline"),
    ]
}

#[tokio::test]
async fn case_12_the_name_match_outranks_the_description_match_and_cjk_stays_bounded() {
    let retriever = Retriever::from_loaded_index(
        loaded(field_weight_tree(), field_weight_records()),
        RetrieverConfig::default(),
    )
    .await
    .unwrap();
    assert_eq!(
        retriever.search("alpha").await.unwrap(),
        vec!["zulu", "aardvark"],
        "the name match outranks the description-only match even though `aardvark` sorts first"
    );

    // With `alpha` in both descriptions only, the ranking falls back to the identity
    // tie-break, which is the alphabetical order — so the field weight decided the case above.
    let both_descriptions =
        RetrieverNode::new(ROOT_NODE_ID, ROOT_NODE_ID).with_children(vec![RetrieverNode::new(
            "Skills", "Skills",
        )
        .with_items(vec![
            RetrieverItem::new("zeta", "zeta")
                .with_label("Zeta Skill")
                .with_description("reports on the alpha pipeline"),
            RetrieverItem::new("aardvark", "aardvark")
                .with_label("Aardvark Notes")
                .with_description("explains the alpha pipeline"),
        ])]);
    let tied = Retriever::from_loaded_index(
        loaded(
            both_descriptions,
            vec![
                CatalogRecord::new("zeta", "zeta").with_name("Zeta Skill"),
                CatalogRecord::new("aardvark", "aardvark").with_name("Aardvark Notes"),
            ],
        ),
        RetrieverConfig::default(),
    )
    .await
    .unwrap();
    assert_eq!(
        tied.search("alpha").await.unwrap(),
        vec!["aardvark", "zeta"],
        "with no name match the deterministic tie-break decides"
    );

    // A CJK query reaches a CJK catalog and a bounded snippet never splits a character.
    let cjk = Retriever::from_loaded_index(
        loaded(
            RetrieverNode::new(ROOT_NODE_ID, ROOT_NODE_ID).with_children(vec![RetrieverNode::new(
                "能力树",
                "能力树",
            )
            .with_items(vec![RetrieverItem::new("索引", "索引")
                .with_label("能力树索引")
                .with_description("构建并维护能力树索引")])]),
            vec![CatalogRecord::new("ability-tree", "索引")
                .with_name("能力树索引")
                .with_description("构建并维护能力树索引")],
        ),
        RetrieverConfig::default(),
    )
    .await
    .unwrap();
    assert_eq!(cjk.search("能力树").await.unwrap(), vec!["ability-tree"]);

    let index = LexicalIndex::build(vec![LexicalDocument::new("k", "能力树索引")
        .with_description("构建并维护能力树索引，支持渐进式披露与检索")]);
    let snippet = index.match_snippet("k", &["能力树"], 0, 12);
    assert!(!snippet.is_empty());
    let excerpt = snippet
        .trim_start_matches("name: ")
        .trim_start_matches("description: ");
    assert!(excerpt.chars().count() <= 12, "{snippet}");
    assert!(snippet.starts_with("name: ") || snippet.starts_with("description: "));
}

// ---------------------------------------------------------------------------
// 13–17: the request, determinism, public identity, translation and validation
// ---------------------------------------------------------------------------

#[tokio::test]
async fn case_13_the_request_top_k_trims_meaning_unlimited_and_clamps() {
    let dir = tempfile::tempdir().unwrap();
    write_fixture(dir.path()).await;
    let retriever = Retriever::load(
        dir.path(),
        RetrieverConfig {
            top_k: 2,
            ..RetrieverConfig::default()
        },
    )
    .await
    .unwrap();
    assert_eq!(SearchRequest::default().top_k, None);
    let defaulted = retriever.search("capability").await.unwrap();
    assert_eq!(defaulted.len(), 2, "the configured top_k applies");

    let one = retriever
        .search_details_with("capability", &SearchRequest { top_k: Some(1) })
        .await
        .unwrap();
    assert_eq!(one.item_ids.len(), 1);
    assert_eq!(one.candidates.len(), 1);
    assert!(one.candidates[0].selected);
    assert_eq!(one.selected_rank, Some(1));

    let unlimited = retriever
        .search_details_with("capability", &SearchRequest { top_k: Some(0) })
        .await
        .unwrap();
    assert_eq!(unlimited.item_ids.len(), 4, "0 means unlimited");

    let clamped = retriever
        .search_details_with("capability", &SearchRequest { top_k: Some(9) })
        .await
        .unwrap();
    assert_eq!(clamped.item_ids.len(), 2, "an oversized request is clamped");
    let mut ranks: Vec<usize> = clamped.candidates.iter().map(|row| row.rank).collect();
    ranks.dedup();
    assert_eq!(ranks, vec![1, 2]);
    assert_eq!(
        clamped.item_ids,
        clamped
            .candidates
            .iter()
            .map(|row| row.worker_id.clone())
            .collect::<Vec<String>>()
    );
}

#[tokio::test]
async fn case_14_identical_reruns_are_identical() {
    let model = Arc::new(StaticCompletionModel::new(vec!["1\n2".to_owned()]));
    let retriever = model_retriever(level_config())
        .await
        .with_model(model, "scripted");
    let first = retriever.search_details("alpha").await.unwrap();
    let second = retriever.search_details("alpha").await.unwrap();
    assert_eq!(first.item_ids, second.item_ids);
    assert_eq!(first.summary_lines, second.summary_lines);
    assert_eq!(first.candidates, second.candidates);
    assert_eq!(
        event_types(&retriever_probe(&first)),
        event_types(&retriever_probe(&second))
    );
    assert_eq!(first.steps, second.steps);
}

#[tokio::test]
async fn case_15_the_public_ids_are_worker_ids_in_search_order() {
    let retriever = model_retriever(RetrieverConfig {
        top_k: 3,
        render: RenderConfig {
            compact_codes_enabled: true,
            compact_codebook: Vec::new(),
            flatten_tree: true,
            ..level_render()
        },
        ..RetrieverConfig::default()
    })
    .await;
    let details = retriever.search_details("alpha").await.unwrap();
    let ids = retriever.search("alpha").await.unwrap();
    assert_eq!(ids, details.item_ids);
    let cids: Vec<&str> = details
        .candidates
        .iter()
        .map(|row| row.cid.as_str())
        .collect();
    assert!(cids.contains(&ALPHA_WRITER));
    for id in &ids {
        assert!(!cids.contains(&id.as_str()), "{id} must not be a cid");
        assert!(!id.is_empty());
    }
    assert_eq!(
        ids,
        vec!["alpha-tester", "alpha-writer"],
        "only the alpha items match the query, in the pinned order"
    );
    // `selected_item_id` and the summary lines agree with the ranked rows.
    assert_eq!(details.selected_item_id.as_deref(), Some(ids[0].as_str()));
    for (offset, line) in details.summary_lines.iter().enumerate() {
        assert!(
            line.starts_with(&format!("{}. {}", offset + 1, ids[offset])),
            "{line}"
        );
        assert!(line.ends_with("(source=lexical_fallback)"), "{line}");
    }
}

#[test]
fn case_16_the_config_translation_clamps_and_carries() {
    let defaults = runtime_config_from_config(None).unwrap();
    assert_eq!(defaults.top_k, DEFAULT_TOP_K);
    assert_eq!(defaults.max_tokens, 96);
    assert_eq!(defaults.max_branch_choices, 6);
    assert_eq!(defaults.branch_choice_slack, 2);
    assert_eq!(defaults.branch_candidate_slack, 1);
    assert!(defaults.round_robin_branch_reduce);
    assert_eq!(defaults.selection_mode, SelectionMode::Generate);
    assert_eq!(defaults.fallback_mode, FallbackMode::Deterministic);
    assert_eq!(defaults.max_depth, 0);
    assert_eq!(defaults.max_steps, 0);
    assert_eq!(defaults.disclosure.max_exposure_depth_per_call, 99);
    assert_eq!(defaults.disclosure.exposure_threshold, 1_000_000_000);
    assert!(defaults.disclosure.compact_codes_enabled);
    assert!(defaults.disclosure.flatten_full_tree_in_prompt);
    assert_eq!(defaults.disclosure.compact_codebook.len(), 512);
    assert_eq!(defaults.embedding.weight, 0.0);
    // The Rust-only bounds the public config does not expose keep the conservative values.
    assert_eq!(defaults.disclosure.max_chars, 0);
    assert_eq!(defaults.disclosure.max_items_per_branch, 0);
    assert_eq!(defaults.disclosure.max_description_chars, 150);
    assert_eq!(defaults.disclosure.code_width, 0);

    // Every clamp Python's translator performs.
    let mut config = RetrieverConfig {
        top_k: 0,
        embedding_weight: 2.0,
        ..RetrieverConfig::default()
    };
    config.generation.max_tokens = 0;
    config.traversal.max_branch_choices = 0;
    let clamped = runtime_config_from_config(Some(&config)).unwrap();
    assert_eq!(clamped.top_k, 1);
    assert_eq!(clamped.max_tokens, 1);
    assert_eq!(clamped.max_branch_choices, 1);
    assert_eq!(clamped.embedding.weight, 1.0);
    config.embedding_weight = f64::NAN;
    assert_eq!(
        runtime_config_from_config(Some(&config))
            .unwrap()
            .embedding
            .weight,
        0.0
    );

    // The carried-only knobs never reach the runtime, the others translate one to one.
    let mut carried = RetrieverConfig {
        traversal: TraversalConfig {
            max_parallel_branches: 9,
            enable_parallel_branches: false,
            ..TraversalConfig::default()
        },
        ..RetrieverConfig::default()
    };

    carried.generation.trie_constrained_decoding_enabled = true;
    carried.generation.mode = SelectionMode::Lexical;
    carried.generation.fallback_mode = FallbackMode::Error;
    carried.render.flatten_tree = false;
    carried.render.max_exposure_depth = 1;
    carried.render.exposure_threshold = 2;
    carried.render.compact_codes_enabled = false;
    carried.render.compact_codebook = Vec::new();
    let runtime = runtime_config_from_config(Some(&carried)).unwrap();
    assert_eq!(runtime.selection_mode, SelectionMode::Lexical);
    assert_eq!(runtime.fallback_mode, FallbackMode::Error);
    assert!(!runtime.disclosure.flatten_full_tree_in_prompt);
    assert_eq!(runtime.disclosure.max_exposure_depth_per_call, 1);
    assert_eq!(runtime.disclosure.exposure_threshold, 2);
    assert!(!runtime.disclosure.compact_codes_enabled);
    assert!(runtime.disclosure.compact_codebook.is_empty());

    // An unusable code book is rejected while translating.
    carried.render.compact_codebook = vec!["0".to_owned()];
    let error = runtime_config_from_config(Some(&carried)).expect_err("`0` is reserved");
    assert_eq!(error.kind(), ErrorKind::Validation);

    // Both configs round-trip through JSON and tolerate unknown fields.
    let json = serde_json::to_string(&RetrieverConfig::default()).unwrap();
    let parsed: RetrieverConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed, RetrieverConfig::default());
    let tolerant: RetrieverConfig =
        serde_json::from_str(r#"{"top_k": 3, "unknown_field": true}"#).unwrap();
    assert_eq!(tolerant.top_k, 3);
    let request: SearchRequest = serde_json::from_str(r#"{"top_k": 3}"#).unwrap();
    assert_eq!(request.top_k, Some(3));
}

#[tokio::test]
async fn case_17_validation_and_the_accessors() {
    let dir = tempfile::tempdir().unwrap();
    write_fixture(dir.path()).await;
    let retriever = Retriever::load(dir.path(), RetrieverConfig::default())
        .await
        .unwrap();
    for blank in ["", "   ", "\t\n"] {
        let error = retriever
            .search(blank)
            .await
            .expect_err("a blank query fails");
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.message().starts_with("query"), "{}", error.message());
    }

    assert_eq!(retriever.config(), &RetrieverConfig::default());
    assert_eq!(retriever.runtime_config().top_k, DEFAULT_TOP_K);
    assert_eq!(retriever.index().root.node_id, ROOT_NODE_ID);
    assert_eq!(retriever.method(), METHOD_LEXICAL_FALLBACK);
    assert_eq!(retriever.warmed_fragments(), 0);

    let details = retriever.search_details("alpha").await.unwrap();
    let encoded = serde_json::to_string(&details).unwrap();
    let parsed: openjiuwen_symphony::retrieval::search::service::RetrievalResult =
        serde_json::from_str(&encoded).unwrap();
    assert_eq!(parsed.item_ids, details.item_ids);
    assert_eq!(parsed.candidates, details.candidates);
    assert_eq!(parsed.trace_events, details.trace_events);
    // The default render shape is the production one.
    assert_eq!(
        RetrieverConfig::default().traversal,
        TraversalConfig::default()
    );
    assert_eq!(RetrieverConfig::default().render, RenderConfig::default());
    assert_eq!(
        RetrieverConfig::default().generation,
        GenerationConfig::default()
    );
    assert_eq!(
        RenderConfig::default().compact_codebook.len(),
        openjiuwen_symphony::retrieval::search::DEFAULT_COMPACT_CODEBOOK_SIZE
    );
    assert_eq!(
        runtime_config_from_config(Some(&RetrieverConfig::default()))
            .unwrap()
            .embedding,
        EmbeddingScoringConfig::default()
    );
}

// ---------------------------------------------------------------------------
// 18–22: the degraded-selection provenance
// ---------------------------------------------------------------------------

#[tokio::test]
async fn case_18_generate_without_a_model_is_an_observable_degradation() {
    let degraded = model_retriever(level_config()).await;
    assert_eq!(degraded.method(), METHOD_LEXICAL_FALLBACK);
    assert_eq!(
        degraded.selection_provenance(),
        SelectionProvenance::LexicalNoModel
    );

    let lexical = model_retriever(RetrieverConfig {
        generation: GenerationConfig {
            mode: SelectionMode::Lexical,
            ..GenerationConfig::default()
        },
        ..level_config()
    })
    .await;

    let degraded_details = degraded.search_details("alpha writing").await.unwrap();
    let lexical_details = lexical.search_details("alpha writing").await.unwrap();
    assert_eq!(
        degraded_details.item_ids, lexical_details.item_ids,
        "the degradation keeps the lexical ranking"
    );
    assert_eq!(degraded_details.method, METHOD_LEXICAL_FALLBACK);
    assert_eq!(
        degraded_details.selection_provenance,
        SelectionProvenance::LexicalNoModel
    );
    assert!(degraded_details.degraded_selections >= 1);

    let degraded_trace = retriever_probe(&degraded_details);
    let selections: Vec<_> = degraded_trace
        .events()
        .iter()
        .filter(|event| event.event_type == FRAGMENT_SELECTED_EVENT)
        .collect();
    assert!(!selections.is_empty());
    let mut degraded_count = 0_usize;
    for event in &selections {
        match event.detail["mode"].as_str() {
            Some(MODE_LEXICAL_NO_MODEL) => {
                degraded_count += 1;
                assert_eq!(
                    event.detail["provenance"],
                    serde_json::json!("lexical_no_model")
                );
                assert_eq!(event.detail["degraded"], serde_json::json!(true));
            }
            // A node with one selectable target is taken without a selection (never degraded).
            Some("single_selectable_shortcut") => {
                assert_eq!(
                    event.detail["provenance"],
                    serde_json::json!("single_selectable")
                );
                assert_eq!(event.detail["degraded"], serde_json::json!(false));
            }
            other => panic!("unexpected selection mode {other:?}"),
        }
    }
    assert!(degraded_count >= 1, "the model-less selections are marked");
    assert_eq!(trace_degraded_selections(&degraded_trace), degraded_count);
    assert!(
        matches!(
            trace_selection_provenance(&degraded_trace),
            Some(SelectionProvenance::LexicalNoModel | SelectionProvenance::SingleSelectable,)
        ),
        "the walk's last selection is either a degraded selection or the shortcut"
    );

    // The deliberately lexical retriever ranks identically and is not marked degraded.
    assert_eq!(lexical.selection_provenance(), SelectionProvenance::Lexical);
    assert_eq!(
        lexical_details.selection_provenance,
        SelectionProvenance::Lexical
    );
    assert_eq!(lexical_details.degraded_selections, 0);
    for event in lexical_details
        .trace_events
        .iter()
        .filter(|event| event.event_type == FRAGMENT_SELECTED_EVENT)
    {
        let (provenance, mode) = if event.detail["mode"] == serde_json::json!(MODE_LEXICAL) {
            ("lexical", MODE_LEXICAL)
        } else {
            // The single-selectable shortcut is not a selection either.
            ("single_selectable", "single_selectable_shortcut")
        };
        assert_eq!(event.detail["mode"], serde_json::json!(mode));
        assert_eq!(event.detail["provenance"], serde_json::json!(provenance));
        assert_eq!(event.detail["degraded"], serde_json::json!(false));
    }
    assert_eq!(
        trace_degraded_selections(&retriever_probe(&lexical_details)),
        0
    );
}

#[tokio::test]
async fn case_19_error_mode_reports_the_typed_unsupported_with_guidance() {
    let retriever = model_retriever(RetrieverConfig {
        generation: GenerationConfig {
            fallback_mode: FallbackMode::Error,
            ..GenerationConfig::default()
        },
        ..level_config()
    })
    .await;
    let error = retriever
        .search_details("alpha")
        .await
        .expect_err("generate mode without a model must not select");
    assert!(is_unsupported(&error), "{error}");
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    let message = error.message();
    for needle in [
        "needs a model",
        "with_generate_selector",
        "SelectionMode::Lexical",
        "FallbackMode::Deterministic",
    ] {
        assert!(message.contains(needle), "missing {needle:?}: {message}");
    }
}

#[tokio::test]
async fn case_20_the_provenance_fields_round_trip_and_default_for_old_payloads() {
    // An old `SelectionResult` payload carries neither new field.
    let old: SelectionResult =
        serde_json::from_str(r#"{"raw_output":"1","selected_targets":[],"is_abstain":false}"#)
            .unwrap();
    assert_eq!(old.provenance, SelectionProvenance::Lexical);
    assert!(!old.degraded);

    let degraded = SelectionResult::default().with_provenance(SelectionProvenance::LexicalNoModel);
    assert!(degraded.degraded);
    let encoded = serde_json::to_string(&degraded).unwrap();
    let parsed: SelectionResult = serde_json::from_str(&encoded).unwrap();
    assert_eq!(parsed.provenance, SelectionProvenance::LexicalNoModel);
    assert!(parsed.degraded);

    // An old `RetrievalResult` payload parses and reports the additive defaults.
    let legacy: openjiuwen_symphony::retrieval::search::RetrievalResult =
        serde_json::from_str(r#"{"method":"lexical_fallback","item_ids":["alpha-writer"]}"#)
            .unwrap();
    assert_eq!(legacy.selection_provenance, SelectionProvenance::Lexical);
    assert_eq!(legacy.degraded_selections, 0);
    assert!(!legacy.top_k_clamped);

    // A live result round-trips with its provenance.
    let retriever = model_retriever(level_config()).await;
    let details = retriever.search_details("alpha writing").await.unwrap();
    assert_eq!(
        details.selection_provenance,
        SelectionProvenance::LexicalNoModel
    );
    let encoded = serde_json::to_string(&details).unwrap();
    let parsed: openjiuwen_symphony::retrieval::search::RetrievalResult =
        serde_json::from_str(&encoded).unwrap();
    assert_eq!(parsed, details);
}

#[tokio::test]
async fn case_21_the_degraded_walk_renders_byte_identical_disclosures() {
    let retriever = model_retriever(level_config()).await;
    let rendered = |details: &openjiuwen_symphony::retrieval::search::RetrievalResult| {
        details
            .trace_events
            .iter()
            .filter(|event| event.event_type == FRAGMENT_BUILT_EVENT)
            .map(|event| event.detail["rendered_tree"].clone())
            .collect::<Vec<serde_json::Value>>()
    };
    let first = retriever.search_details("alpha writing").await.unwrap();
    let second = retriever.search_details("alpha writing").await.unwrap();
    assert!(!rendered(&first).is_empty());
    assert_eq!(
        rendered(&first),
        rendered(&second),
        "byte-identical disclosures"
    );
    assert_eq!(first.trace_events, second.trace_events);
    assert_eq!(first.selection_provenance, second.selection_provenance);
    assert_eq!(first.degraded_selections, second.degraded_selections);
}

#[tokio::test]
async fn case_22_the_top_k_clamp_is_reported() {
    let dir = tempfile::tempdir().unwrap();
    write_fixture(dir.path()).await;
    let retriever = Retriever::load(
        dir.path(),
        RetrieverConfig {
            top_k: 2,
            ..RetrieverConfig::default()
        },
    )
    .await
    .unwrap();
    let defaulted = retriever.search_details("capability").await.unwrap();
    assert!(
        !defaulted.top_k_clamped,
        "the configured top_k is not a clamp"
    );
    let one = retriever
        .search_details_with("capability", &SearchRequest { top_k: Some(1) })
        .await
        .unwrap();
    assert!(!one.top_k_clamped);
    let unlimited = retriever
        .search_details_with("capability", &SearchRequest { top_k: Some(0) })
        .await
        .unwrap();
    assert!(!unlimited.top_k_clamped, "0 is the unlimited sentinel");
    let oversized = retriever
        .search_details_with("capability", &SearchRequest { top_k: Some(9) })
        .await
        .unwrap();
    assert!(oversized.top_k_clamped);
    assert_eq!(oversized.item_ids.len(), 2);
}
