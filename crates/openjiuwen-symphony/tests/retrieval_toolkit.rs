//! Integration coverage for the agent-facing retrieval toolkit.
//!
//! Everything here is driven through the public crate surface
//! (`openjiuwen_symphony::agent`, `openjiuwen_symphony::retrieval::types`) so the module
//! re-exports and the wire contract are covered, not just the internals.
//!
//! Every scenario is offline and deterministic: a hand-built `LoadedIndex`, the retriever's
//! lexical path (no model is wired), and fixed budgets.
//!
//! # Coverage map
//!
//! | case | subject |
//! | --- | --- |
//! | 01 | `list` at the root and scoped to a branch, with subtree counts |
//! | 02 | `browse` of a branch: scoped children, own items, catalog identity |
//! | 03 | `search` ranking, `method`, `top_k`, and the abstain paths |
//! | 04 | `read` by leaf cid and by worker id: body, SKILL.md line |
//! | 05 | pagination: every row exactly once, cursor resume, cursor rejection |
//! | 06 | clamping (`page_size`/`top_k`) and the typed rejection matrix |
//! | 07 | byte-identical reruns of identical requests |
//! | 08 | the card, the input schema and the prompt snapshot |
//! | 09 | serde round-trips of the request and the response |
#![allow(clippy::unwrap_used, clippy::expect_used)]
#![allow(clippy::too_many_lines, clippy::assert_is_empty)]

use std::collections::BTreeMap;

use openjiuwen_common::card::Card;
use openjiuwen_symphony::agent::retrieval_toolkit::CURSOR_MAX_CHARS;
use openjiuwen_symphony::agent::{
    render_retrieval_prompt, root_cid, tool_input_schema, RetrievalOperation, RetrievalRequest,
    RetrievalResponse, RetrievalToolkit, RetrievalToolkitConfig, BLANK_QUERY_SUGGESTION,
    NO_RESULTS_SUGGESTION, SKILL_RETRIEVAL_TOOL_NAME,
};
use openjiuwen_symphony::error::ErrorKind;
use openjiuwen_symphony::retrieval::types::{
    CatalogRecord, IndexManifest, LoadedIndex, RetrieverItem, RetrieverNode, META_CONTENT,
    ROOT_NODE_ID,
};

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// The `Data` branch, which carries its own items and one child branch.
const DATA: &str = "Skills.Data";
/// The nested `Clean` branch under [`DATA`].
const CLEAN: &str = "Skills.Data.Clean";
/// The leaf cid of the alpha item, attached to [`DATA`].
const ALPHA_LEAF: &str = "Skills.Data.alpha";
/// The leaf cid of the beta item, attached to [`DATA`].
const BETA_LEAF: &str = "Skills.Data.beta";
/// The leaf cid of the only item under [`CLEAN`].
const CLEAN_LEAF: &str = "Skills.Data.Clean.csv-cleaner";
/// An empty branch, so a branch without children or items is covered.
const EMPTY: &str = "Skills.Empty";
/// The `Writing` branch with four own items, so pagination has room.
const WRITING: &str = "Skills.Writing";
/// The leaf cids of the four `Writing` items.
const WRITING_LEAVES: [&str; 4] = [
    "Skills.Writing.data-flow",
    "Skills.Writing.delta-writer",
    "Skills.Writing.echo-writer",
    "Skills.Writing.foxtrot-writer",
];

/// One catalog record with the routing text the lexical scorer reads.
fn record(
    worker_id: &str,
    cid: &str,
    name: &str,
    description: &str,
    retrieval_text: &str,
) -> CatalogRecord {
    CatalogRecord::new(worker_id, cid)
        .with_name(name)
        .with_description(description)
        .with_skill_path(format!("store/{worker_id}/SKILL.md"))
        .with_retrieval_text(retrieval_text)
}

/// The fixture tree:
///
/// ```text
/// ROOT                                 (3 branches)
/// ├── Skills.Data                      (2 own skills, 1 child branch)
/// │   ├── alpha                       -> Skills.Data.alpha
/// │   ├── beta                        -> Skills.Data.beta
/// │   └── Skills.Data.Clean           (1 skill)
/// │       └── csv                     -> Skills.Data.Clean.csv-cleaner
/// ├── Skills.Empty                     (0 skills)
/// └── Skills.Writing                   (4 own skills)
///     ├── data-flow                   -> Skills.Writing.data-flow
///     ├── delta-writer                -> Skills.Writing.delta-writer
///     ├── echo-writer                 -> Skills.Writing.echo-writer
///     └── foxtrot-writer              -> Skills.Writing.foxtrot-writer
/// ```
fn fixture_root() -> RetrieverNode {
    RetrieverNode::new(ROOT_NODE_ID, ROOT_NODE_ID).with_children(vec![
        RetrieverNode::new(DATA, "Data")
            .with_description("data skills")
            .with_items(vec![
                RetrieverItem::new("alpha", ALPHA_LEAF)
                    .with_label("Alpha Skill")
                    .with_description("summarises alpha reports"),
                RetrieverItem::new("beta", BETA_LEAF)
                    .with_label("Beta Skill")
                    .with_description("consumes alpha feeds"),
            ])
            .with_children(vec![RetrieverNode::new(CLEAN, "Clean")
                .with_description("cleaning skills")
                .with_items(vec![RetrieverItem::new("csv", CLEAN_LEAF)
                    .with_label("CSV Cleaner")
                    .with_description("cleans CSV files")])]),
        RetrieverNode::new(EMPTY, "Empty").with_description("empty branch"),
        RetrieverNode::new(WRITING, "Writing")
            .with_description("writing skills")
            .with_items(vec![
                RetrieverItem::new("data-flow", WRITING_LEAVES[0])
                    .with_label("Data Flow Guide")
                    .with_description("explains data flow"),
                RetrieverItem::new("delta-writer", WRITING_LEAVES[1])
                    .with_label("Delta Writer")
                    .with_description("writes delta prose"),
                RetrieverItem::new("echo-writer", WRITING_LEAVES[2])
                    .with_label("Echo Writer")
                    .with_description("writes echo prose"),
                RetrieverItem::new("foxtrot-writer", WRITING_LEAVES[3])
                    .with_label("Foxtrot Writer")
                    .with_description("writes foxtrot prose"),
            ]),
    ])
}

/// The catalog records matching [`fixture_root`].
fn fixture_records() -> Vec<CatalogRecord> {
    let mut alpha = record(
        "alpha-skill",
        ALPHA_LEAF,
        "Alpha Skill",
        "capability that summarises alpha reports",
        "alpha skill summarises alpha reports",
    );
    alpha.metadata.insert(
        META_CONTENT.to_string(),
        "Alpha body\nSecond line".to_string(),
    );
    let mut csv = record(
        "csv-cleaner",
        CLEAN_LEAF,
        "CSV Cleaner",
        "cleans CSV files\nSupports RFC 4180",
        "csv cleaner cleans csv files",
    );
    csv.metadata
        .insert(META_CONTENT.to_string(), "CSV body".to_string());
    vec![
        alpha,
        record(
            "beta-skill",
            BETA_LEAF,
            "Beta Skill",
            "capability that consumes alpha feeds",
            "beta skill consumes alpha feeds",
        ),
        csv,
        record(
            "data-flow-guide",
            WRITING_LEAVES[0],
            "Data Flow Guide",
            "explains data flow",
            "data flow guide explains data flow",
        ),
        record(
            "delta-writer",
            WRITING_LEAVES[1],
            "Delta Writer",
            "writes delta prose",
            "delta writer writes delta prose",
        ),
        record(
            "echo-writer",
            WRITING_LEAVES[2],
            "Echo Writer",
            "writes echo prose",
            "echo writer writes echo prose",
        ),
        record(
            "foxtrot-writer",
            WRITING_LEAVES[3],
            "Foxtrot Writer",
            "writes foxtrot prose",
            "foxtrot writer writes foxtrot prose",
        ),
    ]
}

/// The item-path map matching [`fixture_root`].
fn fixture_item_paths() -> BTreeMap<String, Vec<String>> {
    let mut paths = BTreeMap::new();
    paths.insert(ALPHA_LEAF.to_string(), vec![DATA.to_string()]);
    paths.insert(BETA_LEAF.to_string(), vec![DATA.to_string()]);
    paths.insert(
        CLEAN_LEAF.to_string(),
        vec![DATA.to_string(), CLEAN.to_string()],
    );
    for leaf in WRITING_LEAVES {
        paths.insert(leaf.to_string(), vec![WRITING.to_string()]);
    }
    paths
}

/// The loaded fixture index.
fn fixture_index() -> LoadedIndex {
    let root = fixture_root();
    LoadedIndex {
        choices: root.flatten_choices(),
        root,
        records: fixture_records(),
        manifest: IndexManifest::default(),
        item_paths: fixture_item_paths(),
    }
}

/// A toolkit over [`fixture_index`] for `session-a`.
async fn toolkit() -> RetrievalToolkit {
    RetrievalToolkit::from_loaded_index(fixture_index(), RetrievalToolkitConfig::new("session-a"))
        .await
        .unwrap()
}

/// A toolkit over [`fixture_index`] with `config`.
async fn toolkit_with(config: RetrievalToolkitConfig) -> RetrievalToolkit {
    RetrievalToolkit::from_loaded_index(fixture_index(), config)
        .await
        .unwrap()
}

/// A `list` request for `cid`, or the root when it is `None`.
fn list(cid: Option<&str>) -> RetrievalRequest {
    RetrievalRequest {
        operation: RetrievalOperation::List,
        cid: cid.map(str::to_string),
        ..RetrievalRequest::default()
    }
}

/// A `browse` request for `cid`.
fn browse(cid: &str) -> RetrievalRequest {
    RetrievalRequest {
        operation: RetrievalOperation::Browse,
        cid: Some(cid.to_string()),
        ..RetrievalRequest::default()
    }
}

/// A `search` request for `query`.
fn search(query: &str) -> RetrievalRequest {
    RetrievalRequest {
        operation: RetrievalOperation::Search,
        query: Some(query.to_string()),
        ..RetrievalRequest::default()
    }
}

/// A `read` request for `cid`.
fn read(cid: &str) -> RetrievalRequest {
    RetrievalRequest {
        operation: RetrievalOperation::Read,
        cid: Some(cid.to_string()),
        ..RetrievalRequest::default()
    }
}

/// The entry cids of a response, in render order.
fn entry_cids(response: &RetrievalResponse) -> Vec<&str> {
    response
        .entries
        .iter()
        .map(|entry| entry.cid.as_str())
        .collect()
}

// ---------------------------------------------------------------------------
// 01 - list
// ---------------------------------------------------------------------------

#[tokio::test]
async fn list_at_the_root_returns_sorted_branches_with_subtree_counts() {
    let mut toolkit = toolkit().await;
    let response = toolkit.execute(&list(None)).await.unwrap();

    assert_eq!(response.operation, RetrievalOperation::List);
    assert_eq!(response.cid, ROOT_NODE_ID);
    assert_eq!(response.cid, root_cid());
    assert!(response.path.is_empty());
    assert_eq!(response.total_count, 3);
    assert_eq!(response.shown_count, 3);
    assert_eq!(response.shown_count, response.entries.len());
    assert!(!response.truncated);
    assert!(response.result_complete);
    assert!(!response.abstained);
    assert!(response.next_cursor.is_none());
    assert_eq!(response.effective_top_k, 0);
    assert!(response.page_size > 0);

    assert_eq!(
        entry_cids(&response),
        vec![DATA, EMPTY, WRITING],
        "branches sort by node id"
    );
    for entry in &response.entries {
        assert_eq!(entry.kind, "branch");
        assert!(entry.worker_id.is_empty());
        assert_eq!(entry.rank, 0);
    }
    let data = &response.entries[0];
    assert_eq!(data.label, "Data");
    assert_eq!(data.description, "data skills");
    assert!(data.item_count >= 2, "Data carries at least its own items");
    assert_eq!(data.branch_count, 1);
    let empty = &response.entries[1];
    assert_eq!((empty.item_count, empty.branch_count), (0, 0));
    assert_eq!(response.entries[2].item_count, 4);

    assert!(response.content.contains(DATA));
    assert!(response
        .model_content
        .contains(&format!("operation=list cid={ROOT_NODE_ID}")));
    assert!(response.model_content.contains(&response.content));
    assert_eq!(response.method, toolkit.retriever().method());
}

#[tokio::test]
async fn list_scopes_to_a_branch_and_returns_its_sub_branches() {
    let mut toolkit = toolkit().await;
    let response = toolkit.execute(&list(Some(DATA))).await.unwrap();

    assert_eq!(response.cid, DATA);
    assert_eq!(response.path, vec![DATA.to_string()]);
    assert_eq!(response.total_count, 1, "Data has exactly one sub-branch");
    assert_eq!(entry_cids(&response), vec![CLEAN]);
    assert_eq!(response.entries[0].kind, "branch");
    assert_eq!(response.entries[0].label, "Clean");
    assert_eq!(
        response.entries[0].path,
        vec![DATA.to_string(), CLEAN.to_string()]
    );
    assert_eq!(response.entries[0].item_count, 1);
    assert_eq!(response.entries[0].branch_count, 0);

    // A node's own items are `browse`'s answer, not `list`'s.
    let browsed = toolkit.execute(&browse(DATA)).await.unwrap();
    assert_eq!(entry_cids(&browsed), vec![CLEAN, ALPHA_LEAF, BETA_LEAF]);
}

#[tokio::test]
async fn list_of_an_empty_branch_is_a_complete_empty_answer() {
    let mut toolkit = toolkit().await;
    let response = toolkit.execute(&list(Some(EMPTY))).await.unwrap();

    assert_eq!(response.total_count, 0);
    assert_eq!(response.shown_count, 0);
    assert!(response.entries.is_empty());
    assert!(response.result_complete);
    assert!(!response.abstained, "an empty branch is not an abstain");
    assert!(
        !response.content.is_empty(),
        "the empty message still renders"
    );
}

#[tokio::test]
async fn list_rejects_a_leaf_cid_and_an_unknown_cid() {
    let mut toolkit = toolkit().await;

    let leaf = toolkit.execute(&list(Some(ALPHA_LEAF))).await.unwrap_err();
    assert_eq!(leaf.kind(), ErrorKind::NotFound);
    assert!(
        leaf.to_string().contains("read"),
        "points at operation=read"
    );

    let unknown = toolkit
        .execute(&list(Some("Skills.Nope")))
        .await
        .unwrap_err();
    assert_eq!(unknown.kind(), ErrorKind::NotFound);
}

// ---------------------------------------------------------------------------
// 02 - browse
// ---------------------------------------------------------------------------

#[tokio::test]
async fn browse_returns_the_node_its_scoped_children_and_its_own_items() {
    let mut toolkit = toolkit().await;
    let response = toolkit.execute(&browse(DATA)).await.unwrap();

    assert_eq!(response.operation, RetrievalOperation::Browse);
    assert_eq!(response.cid, DATA);
    assert_eq!(response.path, vec![DATA.to_string()]);
    assert_eq!(response.total_count, 3, "one child branch plus two items");
    assert_eq!(entry_cids(&response), vec![CLEAN, ALPHA_LEAF, BETA_LEAF]);
    assert_eq!(response.entries[0].item_count, 1);
    assert_eq!(
        response.entries[1].description, "capability that summarises alpha reports",
        "an item row carries the catalog record's description"
    );
    assert!(response.content.contains("Data"), "the node title renders");
    assert!(response.model_content.contains("operation=browse"));
}

#[tokio::test]
async fn browse_rejects_an_unknown_cid() {
    let mut toolkit = toolkit().await;
    let error = toolkit.execute(&browse("Skills.Nope")).await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::NotFound);
}

// ---------------------------------------------------------------------------
// 03 - search and abstain
// ---------------------------------------------------------------------------

#[tokio::test]
async fn search_returns_ranked_hits_with_disclosure_ready_fields() {
    let mut toolkit = toolkit().await;
    let response = toolkit.execute(&search("alpha")).await.unwrap();

    assert_eq!(response.operation, RetrievalOperation::Search);
    assert!(!response.abstained);
    assert!(response.abstain_reason.is_none());
    assert!(response.suggestion.is_none());
    assert!(response.total_count >= 1);
    assert_eq!(response.shown_count, response.entries.len());
    assert!(response.page_size > 0, "search pages its hits");
    assert!(response.effective_top_k > 0);
    assert_eq!(response.method, toolkit.retriever().method());

    for (index, entry) in response.entries.iter().enumerate() {
        assert_eq!(entry.kind, "search_hit");
        assert_eq!(entry.rank, index + 1, "ranks are 1-based and dense");
        assert!(!entry.cid.is_empty());
        assert!(!entry.worker_id.is_empty());
        assert!(!entry.label.is_empty());
    }
    assert!(
        response
            .entries
            .iter()
            .any(|entry| entry.worker_id == "alpha-skill"),
        "the alpha record is among the hits: {:?}",
        entry_cids(&response)
    );
    assert!(response.model_content.contains("operation=search"));
}

#[tokio::test]
async fn search_with_no_matches_abstains_without_an_error() {
    let mut toolkit = toolkit().await;
    let response = toolkit
        .execute(&search("zzzz-no-such-capability"))
        .await
        .unwrap();

    assert!(response.abstained);
    assert_eq!(response.abstain_reason.as_deref(), Some("no_results"));
    assert_eq!(response.suggestion.as_deref(), Some(NO_RESULTS_SUGGESTION));
    assert_eq!(response.total_count, 0);
    assert!(response.entries.is_empty());
    assert!(response.result_complete);
    assert!(response.next_cursor.is_none());
}

#[tokio::test]
async fn a_blank_query_abstains_without_an_error() {
    let mut toolkit = toolkit().await;
    let response = toolkit.execute(&search("   ")).await.unwrap();

    assert!(response.abstained);
    assert_eq!(response.abstain_reason.as_deref(), Some("blank_query"));
    assert_eq!(response.suggestion.as_deref(), Some(BLANK_QUERY_SUGGESTION));
    assert_eq!(response.total_count, 0);
    assert!(response.entries.is_empty());
    assert!(response.result_complete);
}

#[tokio::test]
async fn search_honours_top_k() {
    let mut toolkit = toolkit().await;
    let request = RetrievalRequest {
        operation: RetrievalOperation::Search,
        query: Some("writer".to_string()),
        top_k: Some(2),
        ..RetrievalRequest::default()
    };
    let response = toolkit.execute(&request).await.unwrap();

    assert_eq!(response.effective_top_k, 2);
    assert!(response.entries.len() <= 2);
    assert!(response.total_count >= response.entries.len());
}

// ---------------------------------------------------------------------------
// 04 - read
// ---------------------------------------------------------------------------

#[tokio::test]
async fn read_returns_the_routing_body_and_the_skill_path() {
    let mut toolkit = toolkit().await;
    let response = toolkit.execute(&read(ALPHA_LEAF)).await.unwrap();

    assert_eq!(response.operation, RetrievalOperation::Read);
    assert_eq!(response.cid, ALPHA_LEAF);
    assert_eq!(response.total_count, 1);
    assert_eq!(response.entries.len(), 1);
    let entry = &response.entries[0];
    assert_eq!(entry.worker_id, "alpha-skill");
    assert_eq!(entry.label, "Alpha Skill");
    assert_eq!(entry.cid, ALPHA_LEAF);
    assert!(
        response.content.contains("Alpha body"),
        "the META.md body renders: {:?}",
        response.content
    );
    assert!(response.content.contains("Second line"));
    assert!(response.content.contains("store/alpha-skill/SKILL.md"));
    assert_eq!(response.page_size, 0);
    assert!(response.result_complete);
}

#[tokio::test]
async fn read_resolves_a_worker_id_and_rejects_unknown_references() {
    let mut toolkit = toolkit().await;
    let response = toolkit.execute(&read("csv-cleaner")).await.unwrap();
    assert_eq!(response.entries.len(), 1);
    assert_eq!(response.entries[0].cid, CLEAN_LEAF);
    assert!(response.content.contains("CSV body"));

    let unknown = toolkit.execute(&read("no-such-skill")).await.unwrap_err();
    assert_eq!(unknown.kind(), ErrorKind::NotFound);
}

// ---------------------------------------------------------------------------
// 05 - pagination
// ---------------------------------------------------------------------------

#[tokio::test]
async fn pagination_walks_every_row_exactly_once() {
    let mut toolkit = toolkit().await;
    let request = RetrievalRequest {
        operation: RetrievalOperation::Browse,
        cid: Some(WRITING.to_string()),
        page_size: Some(2),
        ..RetrievalRequest::default()
    };

    let first = toolkit.execute(&request).await.unwrap();
    assert_eq!(first.total_count, 4);
    assert_eq!(first.entries.len(), 2);
    assert!(!first.result_complete);
    assert_eq!(first.truncation_reason.as_deref(), Some("page"));
    let cursor = first
        .next_cursor
        .clone()
        .expect("a first page of two mints a cursor");
    assert!(cursor.len() <= CURSOR_MAX_CHARS);
    assert!(
        first.content.contains("[next]"),
        "the footer names the cursor"
    );
    assert!(first.content.contains(&cursor));

    let second = toolkit
        .execute(&RetrievalRequest {
            cursor: Some(cursor.clone()),
            ..request.clone()
        })
        .await
        .unwrap();
    assert_eq!(second.total_count, 4);
    assert_eq!(second.entries.len(), 2);
    assert!(second.result_complete);
    assert!(second.next_cursor.is_none());

    let mut cids: Vec<&str> = entry_cids(&first)
        .into_iter()
        .chain(entry_cids(&second))
        .collect();
    cids.sort_unstable();
    let mut expected = WRITING_LEAVES.to_vec();
    expected.sort_unstable();
    assert_eq!(cids, expected, "no row is repeated or skipped");

    let expired = toolkit
        .execute(&RetrievalRequest {
            cursor: first.next_cursor.clone(),
            ..request
        })
        .await
        .unwrap_err();
    assert_eq!(expired.kind(), ErrorKind::Validation);
    assert!(expired.to_string().contains("cursor"));
}

#[tokio::test]
async fn a_cursor_is_rejected_when_the_request_changed() {
    let mut toolkit = toolkit().await;
    let request = RetrievalRequest {
        operation: RetrievalOperation::Browse,
        cid: Some(WRITING.to_string()),
        page_size: Some(2),
        ..RetrievalRequest::default()
    };
    let first = toolkit.execute(&request).await.unwrap();
    let cursor = first.next_cursor.expect("a cursor was minted");

    let mismatched = toolkit
        .execute(&RetrievalRequest {
            cid: Some(DATA.to_string()),
            cursor: Some(cursor),
            ..request
        })
        .await
        .unwrap_err();
    assert_eq!(mismatched.kind(), ErrorKind::Validation);
    assert!(mismatched.to_string().contains("cursor"));
}

#[tokio::test]
async fn closing_the_toolkit_drops_every_cursor() {
    let mut toolkit = toolkit().await;
    let request = RetrievalRequest {
        operation: RetrievalOperation::Browse,
        cid: Some(WRITING.to_string()),
        page_size: Some(2),
        ..RetrievalRequest::default()
    };
    let first = toolkit.execute(&request).await.unwrap();
    let cursor = first.next_cursor.expect("a cursor was minted");
    assert!(toolkit.cursor_count() > 0);

    toolkit.close();
    assert_eq!(toolkit.cursor_count(), 0);

    let expired = toolkit
        .execute(&RetrievalRequest {
            cursor: Some(cursor),
            ..request
        })
        .await
        .unwrap_err();
    assert_eq!(expired.kind(), ErrorKind::Validation);
}

// ---------------------------------------------------------------------------
// 06 - clamping and the rejection matrix
// ---------------------------------------------------------------------------

#[tokio::test]
async fn an_over_large_page_size_and_top_k_are_clamped_not_rejected() {
    let mut toolkit = toolkit_with(RetrievalToolkitConfig {
        default_page_size: 2,
        max_page_size: 3,
        max_top_k: 2,
        ..RetrievalToolkitConfig::new("session-a")
    })
    .await;

    let listed = toolkit
        .execute(&RetrievalRequest {
            operation: RetrievalOperation::Browse,
            cid: Some(WRITING.to_string()),
            page_size: Some(9_000),
            ..RetrievalRequest::default()
        })
        .await
        .unwrap();
    assert_eq!(listed.page_size, 3, "clamped to max_page_size");
    assert_eq!(listed.entries.len(), 3);
    assert!(listed.next_cursor.is_some(), "the fourth row is still due");

    let searched = toolkit
        .execute(&RetrievalRequest {
            operation: RetrievalOperation::Search,
            query: Some("writer".to_string()),
            top_k: Some(9_000),
            ..RetrievalRequest::default()
        })
        .await
        .unwrap();
    assert_eq!(searched.effective_top_k, 2, "clamped to max_top_k");
}

#[tokio::test]
async fn the_rejection_matrix_names_the_offending_field() {
    let mut toolkit = toolkit().await;

    let cases: Vec<(serde_json::Value, &str)> = vec![
        (serde_json::json!({"operation": "teleport"}), "operation"),
        (serde_json::json!({"operation": 7}), "operation"),
        (serde_json::json!({}), "operation"),
        (serde_json::json!({"operation": "list", "cid": "  "}), "cid"),
        (serde_json::json!({"operation": "list", "cid": 7}), "cid"),
        (
            serde_json::json!({"operation": "search", "cid": "Skills.Data", "query": "alpha"}),
            "cid",
        ),
        (serde_json::json!({"operation": "read"}), "cid"),
        (
            serde_json::json!({"operation": "list", "query": "alpha"}),
            "query",
        ),
        (
            serde_json::json!({"operation": "read", "cid": "Skills.Data.alpha", "page_size": 3}),
            "page_size",
        ),
        (
            serde_json::json!({"operation": "list", "top_k": 3}),
            "top_k",
        ),
        (
            serde_json::json!({"operation": "read", "cid": "Skills.Data.alpha", "cursor": "abc"}),
            "cursor",
        ),
        (
            serde_json::json!({"operation": "list", "cursor": ""}),
            "cursor",
        ),
        (
            serde_json::json!({"operation": "list", "query": "x", "unknown": 1}),
            "arguments",
        ),
        (serde_json::json!(["not", "an", "object"]), "arguments"),
    ];
    for (arguments, field) in cases {
        let error = toolkit.invoke(&arguments).await.unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation, "{arguments}");
        assert!(
            error.to_string().contains(field),
            "expected {field} in {:?}",
            error.to_string()
        );
    }
}

#[tokio::test]
async fn the_configuration_matrix_is_validated() {
    let blank = RetrievalToolkitConfig::new("   ");
    let error = RetrievalToolkit::from_loaded_index(fixture_index(), blank)
        .await
        .err()
        .expect("a blank session scope is rejected");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("session_scope"));

    let big = RetrievalToolkitConfig {
        max_output_chars: 1,
        ..RetrievalToolkitConfig::new("session-a")
    };
    let error = RetrievalToolkit::from_loaded_index(fixture_index(), big)
        .await
        .err()
        .expect("an out-of-range budget is rejected");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("max_output_chars"));

    let page = RetrievalToolkitConfig {
        max_page_size: 0,
        ..RetrievalToolkitConfig::new("session-a")
    };
    let error = RetrievalToolkit::from_loaded_index(fixture_index(), page)
        .await
        .err()
        .expect("a zero page size is rejected");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("max_page_size"));
}

// ---------------------------------------------------------------------------
// 07 - determinism
// ---------------------------------------------------------------------------

#[tokio::test]
async fn repeated_identical_requests_are_byte_identical() {
    let mut toolkit = toolkit().await;
    let requests = [
        list(None),
        list(Some(DATA)),
        browse(DATA),
        search("writer"),
        read(ALPHA_LEAF),
    ];
    for request in requests {
        let first = toolkit.execute(&request).await.unwrap();
        let second = toolkit.execute(&request).await.unwrap();
        assert_eq!(first.content, second.content, "{request:?}");
        assert_eq!(first.model_content, second.model_content, "{request:?}");
        assert_eq!(first.entries, second.entries, "{request:?}");
        assert_eq!(first.next_cursor, second.next_cursor, "{request:?}");
        assert_eq!(first.total_count, second.total_count, "{request:?}");
        assert_eq!(first.shown_count, second.shown_count, "{request:?}");
    }
}

#[tokio::test]
async fn invoke_and_execute_agree() {
    let mut toolkit = toolkit().await;
    let typed = toolkit.execute(&list(Some(DATA))).await.unwrap();
    let invoked = toolkit
        .invoke(&serde_json::json!({"operation": "list", "cid": DATA}))
        .await
        .unwrap();
    assert_eq!(typed.content, invoked.content);
    assert_eq!(typed.model_content, invoked.model_content);
    assert_eq!(typed.entries, invoked.entries);
}

// ---------------------------------------------------------------------------
// 08 - card, schema, prompt
// ---------------------------------------------------------------------------

#[tokio::test]
async fn the_card_and_the_input_schema_describe_the_tool() {
    let toolkit = toolkit().await;
    let card = toolkit.card();

    assert_eq!(card.name, SKILL_RETRIEVAL_TOOL_NAME);
    assert_eq!(card.id, toolkit.tool_id());
    assert!(card.id.starts_with(SKILL_RETRIEVAL_TOOL_NAME));
    assert!(!card.description.is_empty());
    assert!(!card.parallel_safe);
    assert!(!card.stateless, "the toolkit keeps page cursors");
    assert_eq!(card.name(), SKILL_RETRIEVAL_TOOL_NAME);
    assert_eq!(card.id(), toolkit.tool_id());
    assert!(card.description().is_some());

    let schema = tool_input_schema();
    assert_eq!(schema, card.input_schema);
    assert_eq!(schema["type"], "object");
    assert_eq!(schema["required"], serde_json::json!(["operation"]));
    assert_eq!(schema["additionalProperties"], false);
    let properties = schema["properties"].as_object().unwrap();
    for key in ["operation", "cid", "query", "cursor", "page_size", "top_k"] {
        assert!(properties.contains_key(key), "schema lacks {key}");
    }
    let operations = properties["operation"]["enum"].as_array().unwrap();
    assert_eq!(operations.len(), 4);
    for name in ["list", "browse", "search", "read"] {
        assert!(
            operations.iter().any(|value| value == name),
            "schema lacks operation {name}"
        );
    }
}

#[tokio::test]
async fn the_prompt_snapshot_lists_branches_deterministically_and_respects_its_budget() {
    let toolkit = toolkit().await;
    let snapshot = toolkit.prompt_snapshot(4_000);
    assert!(!snapshot.text.is_empty());
    assert_eq!(snapshot.max_chars, 4_000);
    assert!(!snapshot.truncated);
    assert!(snapshot.text.contains(SKILL_RETRIEVAL_TOOL_NAME));
    assert!(snapshot.included.iter().any(|cid| cid == DATA));
    assert!(snapshot.included.iter().any(|cid| cid == WRITING));
    assert!(snapshot.omitted.is_empty());
    assert!(snapshot.text.chars().count() <= 4_000);

    let again = toolkit.prompt_snapshot(4_000);
    assert_eq!(
        snapshot.text, again.text,
        "identical inputs render identically"
    );

    let tiny = render_retrieval_prompt(toolkit.index(), 200);
    assert!(tiny.truncated);
    assert!(
        tiny.text.chars().count() <= 200,
        "the budget still bounds the text"
    );
}

// ---------------------------------------------------------------------------
// 09 - serde
// ---------------------------------------------------------------------------

#[tokio::test]
async fn the_request_and_the_response_round_trip_through_json() {
    let mut toolkit = toolkit().await;
    let request = browse(DATA);
    let json = serde_json::to_string(&request).unwrap();
    let restored: RetrievalRequest = serde_json::from_str(&json).unwrap();
    assert_eq!(restored, request);

    // Unknown fields are tolerated, and an empty document keeps the defaults.
    let tolerant: RetrievalRequest =
        serde_json::from_str("{\"operation\": \"list\", \"extra\": true}").unwrap();
    assert_eq!(tolerant.operation, RetrievalOperation::List);
    let defaulted: RetrievalRequest = serde_json::from_str("{}").unwrap();
    assert_eq!(defaulted, RetrievalRequest::default());

    let response = toolkit.execute(&request).await.unwrap();
    let json = serde_json::to_string(&response).unwrap();
    let restored: RetrievalResponse = serde_json::from_str(&json).unwrap();
    assert_eq!(restored, response);
    assert_eq!(restored.entries, response.entries);
    assert_eq!(restored.abstained, response.abstained);
}
