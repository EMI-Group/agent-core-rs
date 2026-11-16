//! Behaviour tests for the agent retrieval toolkit.
//!
//! Everything here is offline and deterministic: a hand-built [`LoadedIndex`], the
//! retriever's lexical fallback (no model is wired), and fixed budgets. The fixtures cover
//! a nested tree with branch and item rows, an empty branch, a wide tree for output and
//! prompt budget tests, and a catalog with one record per item.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::too_many_lines,
    clippy::assert_is_empty,
    clippy::field_reassign_with_default
)]

use std::collections::BTreeMap;

use serde_json::json;

use super::render::{compact, truncate_chars};
use super::*;
use crate::retrieval::types::{
    CatalogRecord, IndexManifest, LoadedIndex, RetrieverItem, RetrieverNode, META_CONTENT,
    ROOT_NODE_ID,
};

const DATA: &str = "Skills.Data";
const CLEAN: &str = "Skills.Data.Clean";
const ALPHA_LEAF: &str = "Skills.Data.alpha";
const CLEAN_LEAF: &str = "Skills.Data.Clean.csv-cleaner";
const WRITING: &str = "Skills.Writing";
const GAMMA_LEAF: &str = "Skills.Writing.data-flow";
const DELTA_LEAF: &str = "Skills.Writing.delta-writer";
const EMPTY: &str = "Skills.Empty";

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
/// ROOT
/// ├── Skills.Data              (2 skills, 1 branch)
/// │   ├── Items: alpha         -> Skills.Data.alpha
/// │   └── Skills.Data.Clean    (1 skill)
/// │       └── Items: csv       -> Skills.Data.Clean.csv-cleaner
/// ├── Skills.Empty             (0 skills)
/// └── Skills.Writing           (2 skills)
///     └── Items: data-flow, delta-writer
/// ```
fn fixture_root() -> RetrieverNode {
    RetrieverNode::new(ROOT_NODE_ID, ROOT_NODE_ID).with_children(vec![
        RetrieverNode::new(DATA, "Data")
            .with_description("data skills")
            .with_items(vec![RetrieverItem::new("alpha", ALPHA_LEAF)
                .with_label("Alpha Skill")
                .with_description("summarises alpha reports")])
            .with_children(vec![RetrieverNode::new(CLEAN, "Clean")
                .with_description("cleaning skills")
                .with_items(vec![RetrieverItem::new("csv", CLEAN_LEAF)
                    .with_label("CSV Cleaner")
                    .with_description("cleans CSV files")])]),
        RetrieverNode::new(EMPTY, "Empty").with_description("empty branch"),
        RetrieverNode::new(WRITING, "Writing")
            .with_description("writing skills")
            .with_items(vec![
                RetrieverItem::new("data-flow", GAMMA_LEAF)
                    .with_label("Data Flow Guide")
                    .with_description("explains data flow"),
                RetrieverItem::new("delta-writer", DELTA_LEAF)
                    .with_label("Delta Writer")
                    .with_description("writes delta prose"),
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
        csv,
        record(
            "data-flow-guide",
            GAMMA_LEAF,
            "Data Flow Guide",
            "explains data flow",
            "data flow guide explains data flow",
        ),
        record(
            "delta-writer",
            DELTA_LEAF,
            "Delta Writer",
            "writes delta prose",
            "delta writer writes delta prose",
        ),
    ]
}

/// The item-path map matching [`fixture_root`].
fn fixture_item_paths() -> BTreeMap<String, Vec<String>> {
    [
        (ALPHA_LEAF, vec![DATA.to_string()]),
        (CLEAN_LEAF, vec![DATA.to_string(), CLEAN.to_string()]),
        (GAMMA_LEAF, vec![WRITING.to_string()]),
        (DELTA_LEAF, vec![WRITING.to_string()]),
    ]
    .into_iter()
    .map(|(cid, path)| (cid.to_string(), path))
    .collect()
}

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

/// A tree with 40 first-level branches and CJK descriptions, for budget tests.
fn wide_index() -> LoadedIndex {
    let root = RetrieverNode::new(ROOT_NODE_ID, ROOT_NODE_ID).with_children(
        (0..40)
            .map(|index| {
                let cid = format!("Skills.Wide{index:02}");
                RetrieverNode::new(cid.clone(), format!("Wide {index:02}"))
                    .with_description(format!(
                        "能力分支 {index} 处理输入格式、输出格式、验证、评估、测试与执行环境"
                    ))
                    .with_items(vec![RetrieverItem::new(
                        format!("item-{index:02}"),
                        format!("{cid}.item-{index:02}"),
                    )])
            })
            .collect(),
    );
    let records = (0..40)
        .map(|index| {
            let cid = format!("Skills.Wide{index:02}.item-{index:02}");
            record(
                &format!("worker-{index:02}"),
                &cid,
                &format!("Wide Skill {index:02}"),
                &format!("capability {index}"),
                &format!("worker {index} capability"),
            )
        })
        .collect();
    let item_paths = (0..40)
        .map(|index| {
            (
                format!("Skills.Wide{index:02}.item-{index:02}"),
                vec![format!("Skills.Wide{index:02}")],
            )
        })
        .collect();
    LoadedIndex {
        choices: root.flatten_choices(),
        root,
        records,
        manifest: IndexManifest::default(),
        item_paths,
    }
}

async fn toolkit() -> RetrievalToolkit {
    RetrievalToolkit::from_loaded_index(fixture_index(), RetrievalToolkitConfig::new("session-a"))
        .await
        .unwrap()
}

async fn toolkit_with(config: RetrievalToolkitConfig) -> RetrievalToolkit {
    RetrievalToolkit::from_loaded_index(fixture_index(), config)
        .await
        .unwrap()
}

/// A list request for `cid`, or the root when it is `None`.
fn list(cid: Option<&str>) -> RetrievalRequest {
    RetrievalRequest {
        operation: RetrievalOperation::List,
        cid: cid.map(str::to_string),
        ..RetrievalRequest::default()
    }
}

fn browse(cid: &str) -> RetrievalRequest {
    RetrievalRequest {
        operation: RetrievalOperation::Browse,
        cid: Some(cid.to_string()),
        ..RetrievalRequest::default()
    }
}

fn search(query: &str) -> RetrievalRequest {
    RetrievalRequest {
        operation: RetrievalOperation::Search,
        query: Some(query.to_string()),
        ..RetrievalRequest::default()
    }
}

fn read(cid: &str) -> RetrievalRequest {
    RetrievalRequest {
        operation: RetrievalOperation::Read,
        cid: Some(cid.to_string()),
        ..RetrievalRequest::default()
    }
}

// ---------------------------------------------------------------------------
// list
// ---------------------------------------------------------------------------

#[tokio::test]
async fn list_at_the_root_returns_sorted_branches_with_counts() {
    let mut toolkit = toolkit().await;
    let response = toolkit.execute(&list(None)).await.unwrap();
    assert_eq!(response.operation, RetrievalOperation::List);
    assert_eq!(response.cid, ROOT_NODE_ID);
    assert!(response.path.is_empty());
    assert_eq!(response.total_count, 3);
    assert_eq!(response.shown_count, 3);
    assert!(!response.truncated);
    assert!(response.result_complete);
    assert!(!response.abstained);
    assert!(response.next_cursor.is_none());
    assert_eq!(
        response
            .entries
            .iter()
            .map(|entry| entry.cid.as_str())
            .collect::<Vec<_>>(),
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
    assert_eq!(data.path, vec![DATA.to_string()]);
    assert_eq!(data.item_count, 2);
    assert_eq!(data.branch_count, 1);
    let empty = &response.entries[1];
    assert_eq!((empty.item_count, empty.branch_count), (0, 0));
    let writing = &response.entries[2];
    assert_eq!((writing.item_count, writing.branch_count), (2, 0));
    assert_eq!(response.method, toolkit.retriever().method());
    assert_eq!(response.effective_top_k, 0);
    assert_eq!(response.page_size, DEFAULT_PAGE_SIZE);
    assert!(response.content.contains("# Skill Branch List"));
    assert!(response.content.contains("`Skills.Data`"));
    assert!(response
        .model_content
        .starts_with("[skill_retrieval] operation=list cid=ROOT"));
    assert!(response.model_content.contains(&response.content));
}

#[tokio::test]
async fn list_scopes_to_a_nested_or_empty_branch() {
    let mut toolkit = toolkit().await;
    let response = toolkit.execute(&list(Some(DATA))).await.unwrap();
    assert_eq!(response.cid, DATA);
    assert_eq!(response.path, vec![DATA.to_string()]);
    assert_eq!(response.total_count, 1);
    assert_eq!(response.entries[0].cid, CLEAN);
    assert_eq!(
        response.entries[0].path,
        vec![DATA.to_string(), CLEAN.to_string()]
    );
    assert_eq!(response.entries[0].item_count, 1);
    assert_eq!(response.entries[0].branch_count, 0);

    let empty = toolkit.execute(&list(Some(EMPTY))).await.unwrap();
    assert_eq!(empty.total_count, 0);
    assert!(empty.entries.is_empty());
    assert!(empty.content.contains("No sub-branches are available."));
    assert!(empty.result_complete);

    let clean = toolkit.execute(&list(Some(CLEAN))).await.unwrap();
    assert_eq!(clean.total_count, 0);
    assert_eq!(clean.path, vec![DATA.to_string(), CLEAN.to_string()]);
}

#[tokio::test]
async fn list_accepts_the_root_sentinel_and_rejects_a_leaf_cid() {
    let mut toolkit = toolkit().await;
    let explicit = toolkit.execute(&list(Some(ROOT_NODE_ID))).await.unwrap();
    let omitted = toolkit.execute(&list(None)).await.unwrap();
    assert_eq!(explicit.content, omitted.content);
    assert_eq!(explicit.model_content, omitted.model_content);

    let error = toolkit.execute(&list(Some(ALPHA_LEAF))).await.unwrap_err();
    assert_eq!(error.kind(), crate::error::ErrorKind::NotFound);
    assert!(error.message().contains(ALPHA_LEAF));
    assert!(error.message().contains("operation=read"));

    let error = toolkit
        .execute(&list(Some("Skills.Unknown")))
        .await
        .unwrap_err();
    assert_eq!(error.kind(), crate::error::ErrorKind::NotFound);
    assert!(error.message().contains("unknown branch"));
}

// ---------------------------------------------------------------------------
// browse
// ---------------------------------------------------------------------------

#[tokio::test]
async fn browse_returns_branches_then_own_items_with_catalog_identity() {
    let mut toolkit = toolkit().await;
    let response = toolkit.execute(&browse(DATA)).await.unwrap();
    assert_eq!(response.operation, RetrievalOperation::Browse);
    assert_eq!(response.cid, DATA);
    assert_eq!(response.total_count, 2);
    assert_eq!(
        response
            .entries
            .iter()
            .map(|entry| entry.cid.as_str())
            .collect::<Vec<_>>(),
        vec![CLEAN, ALPHA_LEAF],
        "sub-branches come first, then the node's own items"
    );
    let alpha = &response.entries[1];
    assert_eq!(alpha.kind, "item");
    assert_eq!(alpha.worker_id, "alpha-skill");
    assert_eq!(alpha.label, "Alpha Skill");
    assert_eq!(
        alpha.description,
        "capability that summarises alpha reports"
    );
    assert_eq!(alpha.path, vec![DATA.to_string()]);
    assert_eq!(alpha.category, DATA);
    assert_eq!((alpha.item_count, alpha.branch_count), (0, 0));
    assert!(response.content.contains("# Skill Branch Browse"));
    assert!(response
        .content
        .contains("`alpha-skill` (cid: `Skills.Data.alpha`)"));
    assert!(response.content.contains("(1 skills, 0 branches)"));

    let root = toolkit.execute(&browse(ROOT_NODE_ID)).await.unwrap();
    assert_eq!(root.total_count, 3);
    assert!(root.entries.iter().all(|entry| entry.kind == "branch"));

    let empty = toolkit.execute(&browse(EMPTY)).await.unwrap();
    assert_eq!(empty.total_count, 0);
    assert!(empty
        .content
        .contains("No sub-branches or skills are available in this node."));
}

#[tokio::test]
async fn browse_rejects_an_unknown_branch() {
    let mut toolkit = toolkit().await;
    let error = toolkit
        .execute(&browse("Skills.Missing"))
        .await
        .unwrap_err();
    assert_eq!(error.kind(), crate::error::ErrorKind::NotFound);
    let error = toolkit.execute(&browse(ALPHA_LEAF)).await.unwrap_err();
    assert!(error.message().contains("operation=read"));
}

// ---------------------------------------------------------------------------
// search and abstain
// ---------------------------------------------------------------------------

#[tokio::test]
async fn search_returns_ranked_hits_in_retriever_order() {
    let mut toolkit = toolkit().await;
    let response = toolkit.execute(&search("data flow guide")).await.unwrap();
    assert_eq!(response.operation, RetrievalOperation::Search);
    assert!(!response.abstained);
    assert!(response.abstain_reason.is_none());
    assert!(response.total_count >= 1);
    let first = &response.entries[0];
    assert_eq!(first.rank, 1);
    assert_eq!(first.kind, "search_hit");
    assert_eq!(first.cid, GAMMA_LEAF);
    assert_eq!(first.worker_id, "data-flow-guide");
    assert_eq!(first.label, "Data Flow Guide");
    assert!(!first.path.is_empty());
    assert!(!response.method.is_empty());
    assert_eq!(response.method, toolkit.retriever().method());
    assert_eq!(response.effective_top_k, DEFAULT_TOP_K);
    assert!(response.content.starts_with("# Skill Search"));
    assert!(response.content.contains("Query: \"data flow guide\""));
    assert!(response.content.contains("1. `data-flow-guide`"));
    assert!(response
        .model_content
        .starts_with("[skill_retrieval] operation=search"));
    let ranks: Vec<usize> = response.entries.iter().map(|entry| entry.rank).collect();
    assert_eq!(ranks, (1..=ranks.len()).collect::<Vec<_>>());
}

#[tokio::test]
async fn search_with_zero_hits_abstains_instead_of_erroring() {
    let mut toolkit = toolkit().await;
    let response = toolkit
        .execute(&search("zzzz-no-such-skill-zzzz"))
        .await
        .unwrap();
    assert!(response.abstained);
    assert_eq!(response.abstain_reason.as_deref(), Some(ABSTAIN_NO_RESULTS));
    assert!(response.suggestion.is_some());
    assert!(response.entries.is_empty());
    assert_eq!(response.total_count, 0);
    assert!(response.result_complete);
    assert!(!response.truncated);
    assert!(response
        .content
        .contains("No indexed skill matched the query."));
    assert!(response.content.contains("Suggestion:"));
}

#[tokio::test]
async fn search_with_a_blank_or_missing_query_abstains() {
    let mut toolkit = toolkit().await;
    for request in [
        search("   "),
        RetrievalRequest {
            operation: RetrievalOperation::Search,
            ..RetrievalRequest::default()
        },
    ] {
        let response = toolkit.execute(&request).await.unwrap();
        assert!(response.abstained);
        assert_eq!(
            response.abstain_reason.as_deref(),
            Some(ABSTAIN_BLANK_QUERY)
        );
        assert!(response.entries.is_empty());
        assert!(response
            .suggestion
            .as_deref()
            .unwrap()
            .contains("operation=list"));
        assert!(response.content.contains("The search query is empty"));
    }
}

#[tokio::test]
async fn search_clamps_top_k_to_the_configured_maximum() {
    let mut config = RetrievalToolkitConfig::new("session-a");
    config.max_top_k = 2;
    let mut toolkit = toolkit_with(config).await;
    let response = toolkit
        .execute(&RetrievalRequest {
            operation: RetrievalOperation::Search,
            query: Some("data".to_string()),
            top_k: Some(50),
            ..RetrievalRequest::default()
        })
        .await
        .unwrap();
    assert_eq!(response.effective_top_k, 2);
    assert!(response.entries.len() <= 2);
}

// ---------------------------------------------------------------------------
// read
// ---------------------------------------------------------------------------

#[tokio::test]
async fn read_resolves_by_leaf_cid_worker_id_and_item_id() {
    let mut toolkit = toolkit().await;
    let by_cid = toolkit.execute(&read(CLEAN_LEAF)).await.unwrap();
    assert_eq!(by_cid.operation, RetrievalOperation::Read);
    assert_eq!(by_cid.cid, CLEAN_LEAF);
    assert_eq!(by_cid.total_count, 1);
    assert_eq!(by_cid.shown_count, 1);
    assert!(by_cid.result_complete);
    assert_eq!(by_cid.path, vec![DATA.to_string(), CLEAN.to_string()]);
    let entry = &by_cid.entries[0];
    assert_eq!(entry.kind, "item");
    assert_eq!(entry.worker_id, "csv-cleaner");
    assert_eq!(entry.label, "CSV Cleaner");
    assert_eq!(entry.description, "cleans CSV files\nSupports RFC 4180");
    assert!(by_cid.content.contains("# Skill"));
    assert!(by_cid.content.contains("## Item: `CSV Cleaner`"));
    assert!(by_cid.content.contains("- worker_id: `csv-cleaner`"));
    assert!(by_cid
        .content
        .contains("- cid: `Skills.Data.Clean.csv-cleaner`"));
    assert!(by_cid
        .content
        .contains("- SKILL.md: `store/csv-cleaner/SKILL.md`"));
    assert!(by_cid.content.contains("cleans CSV files"));
    assert!(by_cid.content.contains("CSV body"));
    assert_eq!(by_cid.page_size, 0);
    assert_eq!(by_cid.effective_top_k, 0);

    let by_worker = toolkit.execute(&read("csv-cleaner")).await.unwrap();
    assert_eq!(by_worker.content, by_cid.content);
    assert_eq!(by_worker.model_content, by_cid.model_content);
    assert_eq!(by_worker.cid, CLEAN_LEAF);

    let by_item_id = toolkit.execute(&read("csv")).await.unwrap();
    assert_eq!(by_item_id.content, by_cid.content);
}

#[tokio::test]
async fn read_rejects_unknown_items_and_missing_cids() {
    let mut toolkit = toolkit().await;
    let error = toolkit
        .execute(&read("Skills.Nope.nope"))
        .await
        .unwrap_err();
    assert_eq!(error.kind(), crate::error::ErrorKind::NotFound);
    assert!(error.message().contains("unknown item"));

    let error = toolkit
        .execute(&RetrievalRequest {
            operation: RetrievalOperation::Read,
            ..RetrievalRequest::default()
        })
        .await
        .unwrap_err();
    assert_eq!(error.kind(), crate::error::ErrorKind::Validation);
    assert!(error.message().contains("cid"));
    assert!(error.message().contains("required"));
}

#[tokio::test]
async fn read_falls_back_to_tree_items_without_a_catalog_record() {
    let mut index = fixture_index();
    index.records.clear();
    index.item_paths.clear();
    let mut toolkit =
        RetrievalToolkit::from_loaded_index(index, RetrievalToolkitConfig::new("session-a"))
            .await
            .unwrap();
    let response = toolkit.execute(&read("csv")).await.unwrap();
    assert_eq!(response.cid, CLEAN_LEAF);
    assert_eq!(response.entries[0].worker_id, "csv");
    assert_eq!(response.entries[0].label, "CSV Cleaner");
    assert!(response.content.contains("cleans CSV files"));
    assert!(!response.content.contains("SKILL.md"));
}

// ---------------------------------------------------------------------------
// paging and cursors
// ---------------------------------------------------------------------------

#[tokio::test]
async fn list_pages_with_a_cursor_and_resumes_deterministically() {
    let mut config = RetrievalToolkitConfig::new("session-a");
    config.default_page_size = 2;
    let mut toolkit = toolkit_with(config).await;
    let first = toolkit
        .execute(&RetrievalRequest {
            operation: RetrievalOperation::List,
            page_size: Some(2),
            ..RetrievalRequest::default()
        })
        .await
        .unwrap();
    assert_eq!(first.total_count, 3);
    assert_eq!(first.shown_count, 2);
    assert_eq!(first.entries.len(), 2);
    assert!(first.truncated);
    assert_eq!(first.truncation_reason.as_deref(), Some("page"));
    assert!(!first.result_complete);
    let token = first.next_cursor.clone().unwrap();
    assert!(first.content.contains(&format!(
        "[next] {{\"operation\":\"list\",\"cursor\":\"{token}\"}}"
    )));
    assert_eq!(toolkit.cursor_count(), 1);

    let second = toolkit
        .invoke(&json!({"operation": "list", "cursor": token}))
        .await
        .unwrap();
    assert_eq!(second.total_count, 3);
    assert_eq!(second.shown_count, 1);
    assert_eq!(second.entries[0].cid, WRITING);
    assert!(second.result_complete);
    assert!(!second.truncated);
    assert!(second.next_cursor.is_none());
    assert_eq!(toolkit.cursor_count(), 0, "the resumed cursor is consumed");

    let error = toolkit
        .invoke(&json!({"operation": "list", "cursor": token}))
        .await
        .unwrap_err();
    assert!(error.message().contains("unknown or expired"));
}

#[tokio::test]
async fn page_size_zero_selects_the_default_and_large_values_are_clamped() {
    let mut config = RetrievalToolkitConfig::new("session-a");
    config.default_page_size = 2;
    config.max_page_size = 4;
    let mut toolkit = toolkit_with(config).await;
    let zero = toolkit
        .execute(&RetrievalRequest {
            operation: RetrievalOperation::List,
            page_size: Some(0),
            ..RetrievalRequest::default()
        })
        .await
        .unwrap();
    assert_eq!(zero.page_size, 2);
    assert_eq!(zero.shown_count, 2);

    let huge = toolkit
        .execute(&RetrievalRequest {
            operation: RetrievalOperation::List,
            page_size: Some(500),
            ..RetrievalRequest::default()
        })
        .await
        .unwrap();
    assert_eq!(huge.page_size, 4);
    assert_eq!(huge.shown_count, 3);
    assert!(huge.next_cursor.is_none());
}

#[tokio::test]
async fn search_pages_its_hits_and_rejects_a_mismatched_cursor() {
    let mut config = RetrievalToolkitConfig::new("session-a");
    config.default_page_size = 1;
    let mut toolkit = toolkit_with(config).await;
    let first = toolkit
        .execute(&RetrievalRequest {
            operation: RetrievalOperation::Search,
            query: Some("data".to_string()),
            page_size: Some(1),
            ..RetrievalRequest::default()
        })
        .await
        .unwrap();
    let Some(token) = first.next_cursor.clone() else {
        // The fixture must produce at least two hits for "data" for this test to page.
        assert_eq!(first.total_count, 1);
        return;
    };
    assert!(first.total_count > 1);

    for arguments in [
        json!({"operation": "search", "query": "other", "cursor": token}),
        json!({"operation": "list", "cursor": token}),
        json!({"operation": "search", "query": "data", "page_size": 2, "cursor": token}),
    ] {
        let error = toolkit.invoke(&arguments).await.unwrap_err();
        assert_eq!(error.kind(), crate::error::ErrorKind::Validation);
        assert!(error.message().contains("cursor"), "{}", error.message());
    }

    let resumed = toolkit
        .invoke(&json!({"operation": "search", "query": "data", "cursor": token}))
        .await
        .unwrap();
    assert_eq!(resumed.total_count, first.total_count);
    assert_eq!(resumed.entries[0].rank, 2);
}

#[tokio::test]
async fn cursor_rejections_name_the_field() {
    let mut toolkit = toolkit().await;
    let error = toolkit
        .invoke(&json!({"operation": "list", "cursor": "not-a-real-token"}))
        .await
        .unwrap_err();
    assert_eq!(error.kind(), crate::error::ErrorKind::Validation);
    assert!(error.message().contains("cursor:"));
    assert!(error.message().contains("unknown or expired"));

    let error = toolkit
        .invoke(&json!({"operation": "list", "cursor": "   "}))
        .await
        .unwrap_err();
    assert!(error.message().contains("cursor"));

    let too_long = "x".repeat(CURSOR_MAX_CHARS + 1);
    let error = toolkit
        .invoke(&json!({"operation": "list", "cursor": too_long}))
        .await
        .unwrap_err();
    assert!(error.message().contains("cursor"));
}

#[tokio::test]
async fn closing_drops_every_cursor() {
    let mut config = RetrievalToolkitConfig::new("session-a");
    config.default_page_size = 1;
    let mut toolkit = toolkit_with(config).await;
    let first = toolkit
        .execute(&RetrievalRequest {
            operation: RetrievalOperation::List,
            page_size: Some(1),
            ..RetrievalRequest::default()
        })
        .await
        .unwrap();
    assert!(first.next_cursor.is_some());
    assert_eq!(toolkit.cursor_count(), 1);
    toolkit.close();
    assert_eq!(toolkit.cursor_count(), 0);
}

// ---------------------------------------------------------------------------
// output budget
// ---------------------------------------------------------------------------

#[tokio::test]
async fn the_output_budget_shortens_a_wide_page_and_keeps_a_resumable_cursor() {
    let mut config = RetrievalToolkitConfig::new("session-a");
    config.max_output_chars = MIN_OUTPUT_CHARS;
    config.default_page_size = 40;
    config.max_page_size = 40;
    let mut toolkit = RetrievalToolkit::from_loaded_index(wide_index(), config)
        .await
        .unwrap();
    let response = toolkit.execute(&list(None)).await.unwrap();
    assert!(response.truncated);
    assert_eq!(
        response.truncation_reason.as_deref(),
        Some("max_output_chars")
    );
    assert!(response.shown_count >= 1);
    assert!(response.shown_count < response.total_count);
    assert!(!response.result_complete);
    assert!(response.next_cursor.is_some());
    assert!(response.content.chars().count() <= MIN_OUTPUT_CHARS);
    assert!(response.model_content.chars().count() <= MIN_OUTPUT_CHARS);
    assert!(response.content.contains("能力分支"));

    // The page resumes at the first row that was left out.
    let token = response.next_cursor.clone().unwrap();
    let resumed = toolkit
        .invoke(&json!({"operation": "list", "cursor": token}))
        .await
        .unwrap();
    assert!(resumed.entries[0].cid > response.entries.last().unwrap().cid);
}

#[tokio::test]
async fn the_render_helpers_never_split_a_multi_byte_character() {
    let text = "能力树索引构建与检索";
    let (shortened, cut) = truncate_chars(text, 6);
    assert!(cut);
    assert!(shortened.chars().count() <= 6);
    assert!(text.starts_with(shortened.trim_end_matches('.')));
    let compacted = compact("  能力 \n 树  ", 20);
    assert_eq!(compacted, "能力 树");
    assert_eq!(compact("anything", 0), "");
    assert_eq!(truncate_chars("anything", 0), (String::new(), true));
    assert_eq!(truncate_chars("abc", 3), ("abc".to_string(), false));
    assert_eq!(truncate_chars("abcdef", 4).0, "a...");
}

// ---------------------------------------------------------------------------
// determinism
// ---------------------------------------------------------------------------

#[tokio::test]
async fn repeated_calls_are_byte_identical() {
    let mut toolkit = toolkit().await;
    let first = toolkit.invoke(&json!({"operation": "list"})).await.unwrap();
    let second = toolkit.invoke(&json!({"operation": "list"})).await.unwrap();
    assert_eq!(first.content, second.content);
    assert_eq!(first.model_content, second.model_content);
    assert_eq!(first.entries, second.entries);
    assert_eq!(first.next_cursor, second.next_cursor);

    let first = toolkit
        .invoke(&json!({"operation": "search", "query": "data flow"}))
        .await
        .unwrap();
    let second = toolkit
        .invoke(&json!({"operation": "search", "query": "data flow"}))
        .await
        .unwrap();
    assert_eq!(first.content, second.content);
    assert_eq!(first.model_content, second.model_content);

    let first = toolkit
        .invoke(&json!({"operation": "browse", "cid": "Skills.Data"}))
        .await
        .unwrap();
    let second = toolkit
        .invoke(&json!({"operation": "browse", "cid": "Skills.Data"}))
        .await
        .unwrap();
    assert_eq!(first.model_content, second.model_content);
}

// ---------------------------------------------------------------------------
// invoke validation
// ---------------------------------------------------------------------------

#[tokio::test]
async fn invoke_rejects_unknown_keys_and_wrong_types() {
    let mut toolkit = toolkit().await;
    let error = toolkit
        .invoke(&json!({"operation": "list", "limit": 3}))
        .await
        .unwrap_err();
    assert_eq!(error.kind(), crate::error::ErrorKind::Validation);
    assert!(error.message().contains("limit"));
    assert!(error.message().contains("arguments"));

    let error = toolkit
        .invoke(&json!({"cid": "Skills.Data"}))
        .await
        .unwrap_err();
    assert!(error.message().contains("operation"));

    for arguments in [
        json!({"operation": 3}),
        json!({"operation": "list", "cid": 3}),
        json!({"operation": "list", "page_size": "two"}),
        json!({"operation": "list", "page_size": -1}),
        json!({"operation": "list", "top_k": 1.5}),
        json!({"operation": "search", "query": 3}),
        json!({"operation": "list", "cursor": 7}),
        json!([]),
        json!("list"),
    ] {
        let error = toolkit.invoke(&arguments).await.unwrap_err();
        assert_eq!(error.kind(), crate::error::ErrorKind::Validation);
    }
}

#[tokio::test]
async fn invoke_rejects_an_unknown_operation_and_misused_fields() {
    let mut toolkit = toolkit().await;
    let error = toolkit
        .invoke(&json!({"operation": "explore"}))
        .await
        .unwrap_err();
    assert!(error.message().contains("operation"));
    assert!(error.message().contains("explore"));

    let error = toolkit
        .invoke(&json!({"operation": "list", "query": "data"}))
        .await
        .unwrap_err();
    assert!(error.message().contains("query"));

    let error = toolkit
        .invoke(&json!({"operation": "search", "query": "data", "cid": "Skills.Data"}))
        .await
        .unwrap_err();
    assert!(error.message().contains("cid"));

    let error = toolkit
        .invoke(&json!({"operation": "browse", "top_k": 3}))
        .await
        .unwrap_err();
    assert!(error.message().contains("top_k"));

    let error = toolkit
        .invoke(&json!({"operation": "read", "cid": "Skills.Data.alpha", "page_size": 2}))
        .await
        .unwrap_err();
    assert!(error.message().contains("page_size"));

    let error = toolkit
        .invoke(&json!({"operation": "read", "cid": "Skills.Data.alpha", "cursor": "abc"}))
        .await
        .unwrap_err();
    assert!(error.message().contains("cursor"));

    let error = toolkit
        .invoke(&json!({"operation": "browse", "cid": "  "}))
        .await
        .unwrap_err();
    assert!(error.message().contains("cid"));
}

#[tokio::test]
async fn invoke_is_the_typed_path_for_a_root_listing() {
    let mut toolkit = toolkit().await;
    let from_invoke = toolkit.invoke(&json!({"operation": "list"})).await.unwrap();
    let from_execute = toolkit.execute(&list(None)).await.unwrap();
    assert_eq!(from_invoke.content, from_execute.content);
    assert_eq!(from_invoke.entries, from_execute.entries);
}

// ---------------------------------------------------------------------------
// configuration and card
// ---------------------------------------------------------------------------

#[tokio::test]
async fn the_configuration_matrix_is_rejected_by_the_constructor() {
    type Change = Box<dyn Fn(&mut RetrievalToolkitConfig)>;
    let cases: Vec<(&str, Change)> = vec![
        (
            "session_scope",
            Box::new(|config: &mut RetrievalToolkitConfig| {
                config.session_scope = "  ".to_string();
            }),
        ),
        (
            "max_output_chars",
            Box::new(|config: &mut RetrievalToolkitConfig| {
                config.max_output_chars = MIN_OUTPUT_CHARS - 1;
            }),
        ),
        (
            "max_page_size",
            Box::new(|config: &mut RetrievalToolkitConfig| {
                config.max_page_size = 0;
            }),
        ),
        (
            "max_top_k",
            Box::new(|config: &mut RetrievalToolkitConfig| {
                config.max_top_k = 0;
            }),
        ),
        (
            "default_page_size",
            Box::new(|config: &mut RetrievalToolkitConfig| {
                config.default_page_size = 0;
            }),
        ),
    ];
    for (field, change) in cases {
        let mut config = RetrievalToolkitConfig::new("session-a");
        change(&mut config);
        let retriever = crate::retrieval::search::service::Retriever::from_loaded_index(
            fixture_index(),
            crate::retrieval::search::service::RetrieverConfig::default(),
        )
        .await
        .unwrap();
        let error = RetrievalToolkit::new(retriever, config)
            .map(|_| ())
            .unwrap_err();
        assert_eq!(error.kind(), crate::error::ErrorKind::Validation);
        assert!(error.message().contains(field), "{}", error.message());
    }
}

#[tokio::test]
async fn the_card_and_tool_id_are_stable_and_session_scoped() {
    use openjiuwen_common::card::Card;

    let tool = toolkit().await;
    let card = tool.card();
    assert_eq!(card.id, tool.tool_id());
    assert_eq!(card.name, SKILL_RETRIEVAL_TOOL_NAME);
    assert_eq!(card.name, "skill_retrieval");
    assert_eq!(
        card.description,
        crate::agent::retrieval_toolkit::TOOL_DESCRIPTION
    );
    assert!(card.id.starts_with("skill_retrieval__"));
    assert!(!card.parallel_safe);
    assert!(!card.stateless);
    assert_eq!(card.input_schema, tool_input_schema());
    assert_eq!(card.id(), tool.tool_id());
    assert_eq!(card.name(), SKILL_RETRIEVAL_TOOL_NAME);
    assert_eq!(card.description(), Some(card.description.as_str()));

    let same = toolkit().await;
    assert_eq!(same.tool_id(), tool.tool_id());
    let other = RetrievalToolkit::from_loaded_index(
        fixture_index(),
        RetrievalToolkitConfig::new("session-b"),
    )
    .await
    .unwrap();
    assert_ne!(other.tool_id(), tool.tool_id());
    assert_eq!(tool.index(), same.index());
}

// ---------------------------------------------------------------------------
// prompt snapshot
// ---------------------------------------------------------------------------

#[tokio::test]
async fn the_prompt_lists_the_first_level_categories_with_counts() {
    let index = fixture_index();
    let snapshot = render_retrieval_prompt(&index, 20_000);
    assert!(!snapshot.truncated);
    assert_eq!(snapshot.max_chars, 20_000);
    assert_eq!(
        snapshot.included,
        vec![DATA.to_string(), EMPTY.to_string(), WRITING.to_string()]
    );
    assert!(snapshot.omitted.is_empty());
    assert!(snapshot.text.contains("# Agentic Skill Retrieval"));
    assert!(snapshot.text.contains("`skill_retrieval`"));
    assert!(snapshot.text.contains("## First-Level Categories"));
    assert!(snapshot
        .text
        .contains("- `Skills.Data` - data skills (2 skills)"));
    assert!(snapshot
        .text
        .contains("- `Skills.Empty` - empty branch (0 skills)"));
    assert!(snapshot.text.contains("\"operation\":\"search\""));

    let toolkit = toolkit().await;
    assert_eq!(toolkit.prompt_snapshot(20_000), snapshot);
    assert_eq!(render_retrieval_prompt(&index, 20_000), snapshot);
}

#[tokio::test]
async fn the_prompt_is_budget_bounded_and_reports_what_it_omitted() {
    let index = wide_index();
    let unbounded = render_retrieval_prompt(&index, 200_000);
    assert_eq!(unbounded.included.len(), MAX_PROMPT_CATEGORIES_TEST);
    assert_eq!(unbounded.omitted.len(), 40 - MAX_PROMPT_CATEGORIES_TEST);
    assert!(unbounded.text.contains("more categories"));
    assert!(!unbounded.truncated, "the category cap is not a budget cut");

    let budget = unbounded.text.chars().count().saturating_sub(200);
    let bounded = render_retrieval_prompt(&index, budget);
    assert!(bounded.text.chars().count() <= budget);
    assert!(bounded.truncated);
    assert!(!bounded.omitted.is_empty());
    assert!(bounded.included.len() < unbounded.included.len());

    let empty = render_retrieval_prompt(&index, 0);
    assert!(empty.text.is_empty());
    assert!(empty.truncated);
    assert!(empty.included.is_empty());
    assert_eq!(empty.omitted.len(), 40);
}

#[tokio::test]
async fn the_prompt_on_an_empty_tree_says_so() {
    let index = LoadedIndex {
        root: RetrieverNode::new(ROOT_NODE_ID, ROOT_NODE_ID),
        ..LoadedIndex::default()
    };
    let snapshot = render_retrieval_prompt(&index, 20_000);
    assert!(snapshot
        .text
        .contains("No first-level branches are available."));
    assert!(snapshot.included.is_empty());
    assert!(snapshot.omitted.is_empty());
    assert!(!snapshot.truncated);
}

#[tokio::test]
async fn the_prompt_keeps_a_multi_byte_text_within_its_budget() {
    let index = wide_index();
    // A budget below the instruction block cuts the trailing CJK-safe text.
    let budget = 200;
    let snapshot = render_retrieval_prompt(&index, budget);
    assert!(snapshot.text.chars().count() <= budget);
    assert!(snapshot.truncated);
    assert!(snapshot.text.ends_with("..."));
    assert!(snapshot.included.is_empty());
    assert_eq!(snapshot.omitted.len(), 40);
    assert!(snapshot.text.is_char_boundary(snapshot.text.len()));

    // A budget above the instructions keeps a prefix of the category list.
    let head_len = render_retrieval_prompt(&index, 200_000)
        .text
        .chars()
        .count();
    let snapshot = render_retrieval_prompt(&index, head_len / 2);
    assert!(snapshot.text.chars().count() <= head_len / 2);
    assert!(snapshot.truncated);
    assert!(!snapshot.included.is_empty());
    assert!(!snapshot.omitted.is_empty());
}

/// The category cap, mirrored here so the test does not reach into the renderer.
const MAX_PROMPT_CATEGORIES_TEST: usize = 30;

// ---------------------------------------------------------------------------
// ops
// ---------------------------------------------------------------------------

#[test]
fn node_stats_count_the_subtree() {
    let root = fixture_root();
    let root_stats = super::ops::node_stats(&root);
    assert_eq!(root_stats.item_count, 4);
    assert_eq!(root_stats.branch_count, 4, "three children plus Clean");
    let data = root.find(DATA).unwrap();
    assert_eq!(
        super::ops::node_stats(data),
        super::ops::NodeStats {
            item_count: 2,
            branch_count: 1,
        }
    );
    let empty = root.find(EMPTY).unwrap();
    assert_eq!(
        super::ops::node_stats(empty),
        super::ops::NodeStats::default()
    );
}

#[test]
fn search_rows_keep_the_retriever_rank() {
    let result = crate::retrieval::search::service::RetrievalResult {
        method: "lexical_fallback".to_string(),
        candidates: vec![
            crate::retrieval::search::service::CandidateRecord {
                rank: 1,
                item_id: "alpha".to_string(),
                worker_id: "alpha-skill".to_string(),
                cid: ALPHA_LEAF.to_string(),
                name: "Alpha Skill".to_string(),
                description: "summarises alpha reports".to_string(),
                branch_path: vec![DATA.to_string()],
                category: DATA.to_string(),
                selected: true,
            },
            crate::retrieval::search::service::CandidateRecord {
                rank: 2,
                worker_id: String::new(),
                item_id: "csv".to_string(),
                cid: CLEAN_LEAF.to_string(),
                ..crate::retrieval::search::service::CandidateRecord::default()
            },
        ],
        ..crate::retrieval::search::service::RetrievalResult::default()
    };
    let outcome = super::ops::search(&result);
    assert_eq!(outcome.method, "lexical_fallback");
    assert_eq!(outcome.entries[0].worker_id, "alpha-skill");
    assert_eq!(outcome.entries[0].rank, 1);
    assert_eq!(
        outcome.entries[1].worker_id, "csv",
        "item id is the fallback"
    );
    assert_eq!(outcome.entries[1].label, "csv");
}

#[test]
fn the_index_fingerprint_is_available_for_diagnostics() {
    let fingerprint = super::validate::index_fingerprint(&fixture_index());
    assert_eq!(fingerprint.len(), 64);
    assert_ne!(
        fingerprint,
        super::validate::index_fingerprint(&wide_index())
    );
}
