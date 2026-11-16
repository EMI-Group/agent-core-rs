//! Behavior tests for the capability-tree index ARTIFACT half of
//! `openjiuwen_symphony::retrieval`: the three-file artifact set a full build publishes,
//! the atomic publication rules of `artifacts::write_index`, and loading a published index
//! back through `artifacts::load_index`.
//!
//! Everything here goes through the crate's public API only. The fixtures are built with
//! `tempfile::tempdir()` (never the repository working tree), the file names and the pinned
//! format strings are asserted literally, and nothing depends on the wall clock, on the
//! network, or on hash-map iteration order.
//!
//! The build-stage half lives in `tests/retrieval_build.rs`.
#![allow(clippy::unwrap_used, clippy::expect_used)]
#![allow(clippy::float_cmp)]
#![allow(clippy::assert_is_empty)]

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use serde_json::{json, Value};

use openjiuwen_symphony::error::ErrorKind;
use openjiuwen_symphony::retrieval::artifacts::{
    catalog_content_hash, load_index, parse_tree_yaml, render_tree_yaml, write_index,
};
use openjiuwen_symphony::retrieval::build::scanner::ItemSources;
use openjiuwen_symphony::retrieval::build::scanner::SourceItem;
use openjiuwen_symphony::retrieval::build::tree::visualizer::{render_tree_html, write_tree_html};
use openjiuwen_symphony::retrieval::build::workflows::{BuildConfig, IndexBuilder};
use openjiuwen_symphony::retrieval::types::{
    CatalogRecord, IndexManifest, IndexNode, RetrieverItem, CATALOG_FILENAME, DEFAULT_ITEM_TYPE,
    MANIFEST_FILENAME, MODE_FULL, ROOT_NODE_ID, TREE_HTML_FILENAME, TREE_INDEX_FILENAME,
};

/// The nine keys one `catalog.jsonl` line carries.
const CATALOG_KEYS: [&str; 9] = [
    "worker_id",
    "cid",
    "name",
    "description",
    "skill_path",
    "branch_path",
    "category",
    "retrieval_text",
    "metadata",
];

/// The three artifacts a published index consists of, sorted by name.
///
/// The writer emits the tree first, the catalog second and the manifest last; the
/// directory listings below are sorted, so the constant is sorted too.
const ARTIFACTS: [&str; 3] = [CATALOG_FILENAME, MANIFEST_FILENAME, TREE_INDEX_FILENAME];

/// One scanned skill item; the inventory entry is `store/<id>` so a delete can prune it.
fn item(worker_id: &str, name: &str, classification: &str) -> SourceItem {
    SourceItem {
        worker_id: worker_id.to_string(),
        item: RetrieverItem::new(worker_id, worker_id),
        name: name.to_string(),
        description: format!("{name} routes matching requests."),
        content: format!("{name} usage notes."),
        skill_path: format!("store/{worker_id}/SKILL.md"),
        classification: classification.to_string(),
        tags: Vec::new(),
        item_type: "skill".to_string(),
        content_hash: String::new(),
        metadata: BTreeMap::new(),
    }
}

/// Wraps items into one skill source set with a `store/<id>` inventory.
fn sources(items: Vec<SourceItem>) -> ItemSources {
    let mut sources = ItemSources::new("skill");
    for entry in items {
        sources
            .item_paths
            .push(format!("store/{}", entry.worker_id));
        sources.items.push(entry);
    }
    sources.sort();
    sources
}

/// The three-item fixture: two `documents/pdf` skills and one `web` skill.
fn fixture() -> Vec<SourceItem> {
    vec![
        item("PdfFill", "PDF Fill", "documents/pdf"),
        item("DocRead", "Doc Reader", "documents/pdf"),
        item("WebSearch", "Web Search", "web"),
    ]
}

/// The fixture's seven nodes, sorted by `(depth, cid)` — the order the writer emits.
const EXPECTED_CIDS: [&str; 7] = [
    "Skills",
    "Skills.documents",
    "Skills.web",
    "Skills.documents.pdf",
    "Skills.web.websearch",
    "Skills.documents.pdf.docread",
    "Skills.documents.pdf.pdffill",
];

/// The `(cid, type)` pairs of a rendered `tree_index.yaml`, in document order.
fn yaml_nodes(text: &str) -> Vec<(String, String)> {
    let mut lines = text.lines();
    let mut nodes = Vec::new();
    while let Some(line) = lines.next() {
        let Some(raw_cid) = line.strip_prefix("  - cid: ") else {
            continue;
        };
        let cid: String = serde_json::from_str(raw_cid).unwrap();
        let type_line = lines.next().expect("every node entry carries a `type`");
        let raw_type = type_line
            .strip_prefix("    type: ")
            .unwrap_or_else(|| panic!("unexpected node entry line: {type_line:?}"));
        let node_type: String = serde_json::from_str(raw_type).unwrap();
        nodes.push((cid, node_type));
    }
    nodes
}

/// The file names inside `dir`, sorted.
fn directory_entries(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = fs::read_dir(dir)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().to_string())
        .collect();
    names.sort();
    names
}

/// Reads one index artifact.
fn read_artifact(dir: &Path, name: &str) -> String {
    fs::read_to_string(dir.join(name)).unwrap()
}

/// Parses one `catalog.jsonl` document into records, keeping the file order.
fn catalog_records(text: &str) -> Vec<CatalogRecord> {
    text.lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

/// A valid two-leaf index, used by the atomic-publication tests.
fn small_index() -> (Vec<IndexNode>, Vec<CatalogRecord>, IndexManifest) {
    let nodes = vec![
        IndexNode::branch("Skills").with_description("root"),
        IndexNode::leaf("Skills.a", "alpha"),
        IndexNode::leaf("Skills.b", "beta"),
    ];
    let records = vec![
        CatalogRecord::new("alpha", "Skills.a").with_name("Alpha"),
        CatalogRecord::new("beta", "Skills.b").with_name("Beta"),
    ];
    let manifest = IndexManifest {
        mode: MODE_FULL.to_string(),
        count: records.len(),
        item_paths: vec!["store/alpha".to_string(), "store/beta".to_string()],
        worker_ids: records
            .iter()
            .map(|record| record.worker_id.clone())
            .collect(),
        item_type: DEFAULT_ITEM_TYPE.to_string(),
        content_hash: catalog_content_hash(&records),
    };
    (nodes, records, manifest)
}

/// Asserts that `dir` holds no artifact and no stray temporary file.
fn assert_no_artifacts(dir: &Path) {
    for name in ARTIFACTS {
        assert!(
            !dir.join(name).exists(),
            "{name} was written by a rejected publication"
        );
    }
    assert_eq!(
        directory_entries(dir),
        Vec::<String>::new(),
        "a rejected publication leaves the directory empty"
    );
}

// --------------------------------------------------------------------------------------
// 5. The full build artifact set
// --------------------------------------------------------------------------------------

#[tokio::test]
async fn a_full_build_publishes_exactly_the_three_artifacts() {
    let dir = tempfile::tempdir().unwrap();

    let report = IndexBuilder::build(sources(fixture()), dir.path(), BuildConfig::new())
        .await
        .unwrap();

    assert_eq!(report.output_dir, dir.path());
    assert_eq!(report.item_type, "skill");
    assert_eq!(report.node_count, 7, "4 branches plus 3 leaves");
    assert_eq!(report.record_count, 3);
    assert_eq!(
        directory_entries(dir.path()),
        ARTIFACTS
            .iter()
            .map(|name| (*name).to_string())
            .collect::<Vec<String>>(),
        "a full build writes the three artifacts and nothing else"
    );
}

#[tokio::test]
async fn the_tree_index_lists_one_entry_per_node_sorted_by_depth_then_cid() {
    let dir = tempfile::tempdir().unwrap();
    IndexBuilder::build(sources(fixture()), dir.path(), BuildConfig::new())
        .await
        .unwrap();

    let tree = read_artifact(dir.path(), TREE_INDEX_FILENAME);
    assert!(
        tree.starts_with("nodes:\n"),
        "the document starts with `nodes:`"
    );
    assert!(tree.contains("    type: \"leaf\""));
    assert!(tree.contains("    type: \"branch\""));

    let nodes = yaml_nodes(&tree);
    let cids: Vec<&str> = nodes.iter().map(|(cid, _)| cid.as_str()).collect();
    assert_eq!(cids, EXPECTED_CIDS, "nodes are sorted by (depth, cid)");

    let types: Vec<(&str, &str)> = nodes
        .iter()
        .map(|(cid, node_type)| (cid.as_str(), node_type.as_str()))
        .collect();
    assert_eq!(
        types,
        vec![
            ("Skills", "branch"),
            ("Skills.documents", "branch"),
            ("Skills.web", "branch"),
            ("Skills.documents.pdf", "branch"),
            ("Skills.web.websearch", "leaf"),
            ("Skills.documents.pdf.docread", "leaf"),
            ("Skills.documents.pdf.pdffill", "leaf"),
        ]
    );
}

#[tokio::test]
async fn the_catalog_holds_one_nine_keyed_record_per_leaf_sorted_by_cid() {
    let dir = tempfile::tempdir().unwrap();
    IndexBuilder::build(sources(fixture()), dir.path(), BuildConfig::new())
        .await
        .unwrap();

    let text = read_artifact(dir.path(), CATALOG_FILENAME);
    let lines: Vec<&str> = text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .collect();
    assert_eq!(lines.len(), 3, "one record per leaf");

    let keys: BTreeSet<&str> = CATALOG_KEYS.into_iter().collect();
    let mut cids: Vec<String> = Vec::new();
    for line in &lines {
        let value: Value = serde_json::from_str(line).expect("every catalog line is JSON");
        let object = value.as_object().expect("every catalog line is an object");
        let found: BTreeSet<&str> = object.keys().map(String::as_str).collect();
        assert_eq!(
            found, keys,
            "a catalog record carries the nine documented keys"
        );
        assert!(
            object["metadata"].is_object(),
            "the metadata field is an object: {}",
            object["metadata"]
        );
        cids.push(object["cid"].as_str().unwrap().to_string());
    }
    assert_eq!(
        cids,
        vec![
            "Skills.documents.pdf.docread",
            "Skills.documents.pdf.pdffill",
            "Skills.web.websearch"
        ],
        "records are sorted by cid"
    );
    assert_eq!(
        text.lines().count(),
        3,
        "the catalog is line terminated with no blank trailing record"
    );
}

#[tokio::test]
async fn the_manifest_describes_the_published_catalog() {
    let dir = tempfile::tempdir().unwrap();
    IndexBuilder::build(sources(fixture()), dir.path(), BuildConfig::new())
        .await
        .unwrap();

    let records = catalog_records(&read_artifact(dir.path(), CATALOG_FILENAME));
    let manifest: Value =
        serde_json::from_str(&read_artifact(dir.path(), MANIFEST_FILENAME)).unwrap();

    assert_eq!(manifest["mode"], "full");
    assert_eq!(manifest["count"], json!(records.len()));
    assert_eq!(manifest["item_type"], "skill");
    assert_eq!(
        manifest["worker_ids"],
        json!(["DocRead", "PdfFill", "WebSearch"]),
        "worker ids follow the catalog's cid order"
    );
    assert_eq!(
        manifest["item_paths"],
        json!(["store/PdfFill", "store/DocRead", "store/WebSearch"]),
        "the scanned item paths are recorded in first-seen order"
    );
    assert_eq!(
        manifest["content_hash"],
        json!(catalog_content_hash(&records)),
        "the manifest hashes the rendered catalog"
    );
}

#[tokio::test]
async fn two_items_sharing_a_worker_id_get_distinct_leaf_cids() {
    let dir = tempfile::tempdir().unwrap();
    let repeated = vec![
        item("tool", "Tool One", "documents"),
        item("tool", "Tool Two", "documents"),
    ];

    IndexBuilder::build(sources(repeated), dir.path(), BuildConfig::new())
        .await
        .unwrap();

    let tree = read_artifact(dir.path(), TREE_INDEX_FILENAME);
    let leaves: Vec<String> = yaml_nodes(&tree)
        .into_iter()
        .filter(|(_, node_type)| node_type == "leaf")
        .map(|(cid, _)| cid)
        .collect();
    assert_eq!(
        leaves,
        vec![
            "Skills.documents.tool".to_string(),
            "Skills.documents.tool-2".to_string()
        ],
        "a repeated leaf segment is disambiguated with a numeric suffix"
    );
}

// --------------------------------------------------------------------------------------
// 6. Atomic publication
// --------------------------------------------------------------------------------------

#[tokio::test]
async fn a_duplicate_record_cid_is_rejected_before_anything_is_written() {
    let dir = tempfile::tempdir().unwrap();
    let (nodes, _, manifest) = small_index();
    let records = vec![
        CatalogRecord::new("alpha", "Skills.a"),
        CatalogRecord::new("beta", "Skills.a"),
    ];

    let error = write_index(dir.path(), &nodes, &records, &manifest)
        .await
        .unwrap_err();

    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains(CATALOG_FILENAME));
    assert!(error.message().contains("Skills.a"));
    assert_no_artifacts(dir.path());
}

#[tokio::test]
async fn a_duplicate_node_cid_is_rejected_before_anything_is_written() {
    let dir = tempfile::tempdir().unwrap();
    let (_, records, manifest) = small_index();
    let nodes = vec![
        IndexNode::branch("Skills").with_description("root"),
        IndexNode::leaf("Skills.a", "alpha"),
        IndexNode::leaf("Skills.a", "beta"),
    ];

    let error = write_index(dir.path(), &nodes, &records, &manifest)
        .await
        .unwrap_err();

    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains(TREE_INDEX_FILENAME));
    assert!(error.message().contains("duplicate cid"));
    assert!(error.message().contains("Skills.a"));
    assert_no_artifacts(dir.path());
}

#[tokio::test]
async fn a_successful_write_leaves_no_temporary_sibling_behind() {
    let dir = tempfile::tempdir().unwrap();
    let (nodes, records, manifest) = small_index();

    write_index(dir.path(), &nodes, &records, &manifest)
        .await
        .unwrap();

    let entries = directory_entries(dir.path());
    assert_eq!(
        entries,
        ARTIFACTS
            .iter()
            .map(|name| (*name).to_string())
            .collect::<Vec<String>>()
    );
    for name in &entries {
        let extension = Path::new(name)
            .extension()
            .map(|extension| extension.to_string_lossy().to_string());
        assert_ne!(
            extension.as_deref(),
            Some("tmp"),
            "a temporary artifact survived the rename: {name}"
        );
    }
    assert!(read_artifact(dir.path(), TREE_INDEX_FILENAME).starts_with("nodes:\n"));
    let loaded = load_index(dir.path()).await.unwrap();
    assert_eq!(loaded.records.len(), 2);
}

#[tokio::test]
async fn a_rejected_publication_keeps_the_previous_index_intact() {
    let dir = tempfile::tempdir().unwrap();
    let (nodes, records, manifest) = small_index();
    write_index(dir.path(), &nodes, &records, &manifest)
        .await
        .unwrap();
    let before: Vec<String> = ARTIFACTS
        .iter()
        .map(|name| read_artifact(dir.path(), name))
        .collect();

    let broken = vec![
        CatalogRecord::new("alpha", "Skills.a"),
        CatalogRecord::new("beta", "Skills.a"),
    ];
    let error = write_index(dir.path(), &nodes, &broken, &manifest)
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);

    let after: Vec<String> = ARTIFACTS
        .iter()
        .map(|name| read_artifact(dir.path(), name))
        .collect();
    assert_eq!(after, before, "the published index is untouched");
}

#[tokio::test]
async fn a_manifest_count_that_does_not_describe_the_records_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let (nodes, records, mut manifest) = small_index();
    manifest.count = records.len() + 1;

    let error = write_index(dir.path(), &nodes, &records, &manifest)
        .await
        .unwrap_err();

    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains("count"));
    assert_no_artifacts(dir.path());
}

#[tokio::test]
async fn a_blank_manifest_mode_or_item_type_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let (nodes, records, mut manifest) = small_index();
    manifest.mode = "  ".to_string();

    let error = write_index(dir.path(), &nodes, &records, &manifest)
        .await
        .unwrap_err();

    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains("mode"));
    assert_no_artifacts(dir.path());

    let other = tempfile::tempdir().unwrap();
    let (nodes, records, mut manifest) = small_index();
    manifest.item_type.clear();

    let error = write_index(other.path(), &nodes, &records, &manifest)
        .await
        .unwrap_err();

    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains("item_type"));
    assert_no_artifacts(other.path());
}

// --------------------------------------------------------------------------------------
// 8. Loading
// --------------------------------------------------------------------------------------

#[tokio::test]
async fn a_build_round_trips_through_load_index() {
    let dir = tempfile::tempdir().unwrap();
    IndexBuilder::build(sources(fixture()), dir.path(), BuildConfig::new())
        .await
        .unwrap();

    let loaded = load_index(dir.path()).await.unwrap();

    assert_eq!(loaded.root.node_id, ROOT_NODE_ID);
    assert_eq!(loaded.root.label, ROOT_NODE_ID);
    assert_eq!(loaded.choices.len(), 3, "one choice per leaf item");
    let leaf_cids: BTreeSet<String> = loaded
        .records
        .iter()
        .map(|record| record.cid.clone())
        .collect();
    for choice in &loaded.choices {
        assert!(
            leaf_cids.contains(&choice.payload),
            "a choice payload is a leaf cid, got {}",
            choice.payload
        );
    }

    let record_cids: Vec<&str> = loaded
        .records
        .iter()
        .map(|record| record.cid.as_str())
        .collect();
    let mut sorted = record_cids.clone();
    sorted.sort_unstable();
    assert_eq!(record_cids, sorted, "records come back sorted by cid");
    assert_eq!(
        record_cids,
        vec![
            "Skills.documents.pdf.docread",
            "Skills.documents.pdf.pdffill",
            "Skills.web.websearch"
        ]
    );

    assert_eq!(loaded.manifest.mode, MODE_FULL);
    assert_eq!(loaded.manifest.count, 3);
    assert_eq!(loaded.manifest.item_type, "skill");
    assert_eq!(
        loaded.manifest.worker_ids,
        vec!["DocRead", "PdfFill", "WebSearch"]
    );
    assert_eq!(
        loaded.manifest.item_paths,
        vec!["store/PdfFill", "store/DocRead", "store/WebSearch"]
    );

    let pdf: Vec<String> = vec!["Skills", "Skills.documents", "Skills.documents.pdf"]
        .into_iter()
        .map(str::to_string)
        .collect();
    let web: Vec<String> = vec!["Skills", "Skills.web"]
        .into_iter()
        .map(str::to_string)
        .collect();
    let expected: BTreeMap<String, Vec<String>> = BTreeMap::from([
        ("Skills.documents.pdf.docread".to_string(), pdf.clone()),
        ("Skills.documents.pdf.pdffill".to_string(), pdf),
        ("Skills.web.websearch".to_string(), web),
    ]);
    assert_eq!(
        loaded.item_paths, expected,
        "every leaf cid maps to its ancestor cids"
    );
    for cid in &leaf_cids {
        assert!(
            loaded.item_paths.contains_key(cid),
            "{cid} has no item path"
        );
    }
}

#[tokio::test]
async fn a_missing_manifest_is_a_not_found_naming_the_file() {
    let dir = tempfile::tempdir().unwrap();
    IndexBuilder::build(sources(fixture()), dir.path(), BuildConfig::new())
        .await
        .unwrap();
    fs::remove_file(dir.path().join(MANIFEST_FILENAME)).unwrap();

    let error = load_index(dir.path()).await.unwrap_err();

    assert_eq!(error.kind(), ErrorKind::NotFound);
    assert!(error.message().contains(MANIFEST_FILENAME));
}

#[tokio::test]
async fn a_corrupt_catalog_line_is_a_validation_naming_the_file_and_line() {
    let dir = tempfile::tempdir().unwrap();
    IndexBuilder::build(sources(fixture()), dir.path(), BuildConfig::new())
        .await
        .unwrap();
    let mut text = read_artifact(dir.path(), CATALOG_FILENAME);
    text.push_str("{ not json\n");
    fs::write(dir.path().join(CATALOG_FILENAME), text).unwrap();

    let error = load_index(dir.path()).await.unwrap_err();

    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains(CATALOG_FILENAME));
    assert!(error.message().contains("line 4"));
}

#[tokio::test]
async fn a_catalog_cid_that_names_no_leaf_of_the_tree_is_rejected_on_load() {
    let dir = tempfile::tempdir().unwrap();
    IndexBuilder::build(sources(fixture()), dir.path(), BuildConfig::new())
        .await
        .unwrap();

    let text = read_artifact(dir.path(), CATALOG_FILENAME);
    let mut lines: Vec<String> = text.lines().map(str::to_string).collect();
    let mut record: Value = serde_json::from_str(&lines[0]).expect("the line is JSON");
    record["cid"] = json!("Skills.web.doesnotexist");
    lines[0] = record.to_string();
    fs::write(
        dir.path().join(CATALOG_FILENAME),
        format!("{}\n", lines.join("\n")),
    )
    .unwrap();

    // A Python-written manifest carries no content hash, so the leaf check is the only
    // guard left against loading one build's catalog against another build's tree.
    let mut manifest: Value = serde_json::from_str(&read_artifact(dir.path(), MANIFEST_FILENAME))
        .expect("the manifest is JSON");
    manifest["content_hash"] = json!("");
    fs::write(
        dir.path().join(MANIFEST_FILENAME),
        serde_json::to_string_pretty(&manifest).unwrap(),
    )
    .unwrap();

    let error = load_index(dir.path()).await.unwrap_err();

    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains(CATALOG_FILENAME));
    assert!(
        error.message().contains("Skills.web.doesnotexist"),
        "the first offending cid is named: {}",
        error.message()
    );
}

#[tokio::test]
async fn a_tampered_manifest_content_hash_is_a_validation() {
    let dir = tempfile::tempdir().unwrap();
    IndexBuilder::build(sources(fixture()), dir.path(), BuildConfig::new())
        .await
        .unwrap();
    let path = dir.path().join(MANIFEST_FILENAME);
    let mut manifest: Value = serde_json::from_str(&read_artifact(dir.path(), MANIFEST_FILENAME))
        .expect("the manifest is JSON");
    manifest["content_hash"] =
        json!("0000000000000000000000000000000000000000000000000000000000000000");
    fs::write(&path, serde_json::to_string_pretty(&manifest).unwrap()).unwrap();

    let error = load_index(dir.path()).await.unwrap_err();

    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains(MANIFEST_FILENAME));
    assert!(error.message().contains("content_hash"));
}

#[tokio::test]
async fn a_tampered_manifest_count_is_a_validation() {
    let dir = tempfile::tempdir().unwrap();
    IndexBuilder::build(sources(fixture()), dir.path(), BuildConfig::new())
        .await
        .unwrap();
    let path = dir.path().join(MANIFEST_FILENAME);
    let mut manifest: Value = serde_json::from_str(&read_artifact(dir.path(), MANIFEST_FILENAME))
        .expect("the manifest is JSON");
    let count = manifest["count"].as_u64().unwrap();
    manifest["count"] = json!(count + 1);
    fs::write(&path, serde_json::to_string_pretty(&manifest).unwrap()).unwrap();

    let error = load_index(dir.path()).await.unwrap_err();

    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains(MANIFEST_FILENAME));
    assert!(error.message().contains("count"));
}

// --------------------------------------------------------------------------------------
// 9. The empty index
// --------------------------------------------------------------------------------------

/// The SHA-256 of the empty catalog's rendering — the hash an empty manifest records.
const EMPTY_CATALOG_HASH: &str = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

#[tokio::test]
async fn an_empty_item_source_builds_and_round_trips_through_load_index() {
    let dir = tempfile::tempdir().unwrap();

    let report = IndexBuilder::build(ItemSources::new("skill"), dir.path(), BuildConfig::new())
        .await
        .unwrap();

    assert_eq!(report.record_count, 0);
    assert_eq!(report.manifest.count, 0);
    assert_eq!(report.manifest.content_hash, EMPTY_CATALOG_HASH);
    assert_eq!(
        read_artifact(dir.path(), CATALOG_FILENAME),
        "",
        "an empty catalog renders as the empty document"
    );

    let loaded = load_index(dir.path()).await.unwrap();

    assert_eq!(loaded.manifest.count, 0);
    assert_eq!(loaded.manifest.content_hash, EMPTY_CATALOG_HASH);
    assert!(loaded.records.is_empty());
    assert!(
        loaded.choices.is_empty(),
        "an empty index offers no choices"
    );
    assert_eq!(loaded.root.node_id, ROOT_NODE_ID);
    assert!(
        loaded.root.children().is_empty(),
        "the root branch keeps no child of an empty catalog"
    );
    assert!(loaded.item_paths.is_empty());
}

#[tokio::test]
async fn a_zero_node_index_round_trips_through_write_and_load() {
    let dir = tempfile::tempdir().unwrap();
    let manifest = IndexManifest {
        mode: MODE_FULL.to_string(),
        count: 0,
        item_paths: Vec::new(),
        worker_ids: Vec::new(),
        item_type: DEFAULT_ITEM_TYPE.to_string(),
        content_hash: catalog_content_hash(&[]),
    };

    write_index(dir.path(), &[], &[], &manifest).await.unwrap();

    assert_eq!(
        read_artifact(dir.path(), TREE_INDEX_FILENAME),
        "nodes: []\n",
        "a zero-node tree is published as the empty flow sequence"
    );
    assert_eq!(read_artifact(dir.path(), CATALOG_FILENAME), "");

    let loaded = load_index(dir.path()).await.unwrap();

    assert_eq!(loaded.manifest.count, 0);
    assert_eq!(loaded.manifest.content_hash, EMPTY_CATALOG_HASH);
    assert!(loaded.records.is_empty());
    assert!(loaded.root.children().is_empty());
    assert!(loaded.item_paths.is_empty());
}

#[tokio::test]
async fn the_empty_index_manifest_count_and_hash_are_stable_across_builds() {
    let first = tempfile::tempdir().unwrap();
    let second = tempfile::tempdir().unwrap();
    for dir in [first.path(), second.path()] {
        IndexBuilder::build(ItemSources::new("skill"), dir, BuildConfig::new())
            .await
            .unwrap();
    }

    assert_eq!(
        read_artifact(first.path(), MANIFEST_FILENAME),
        read_artifact(second.path(), MANIFEST_FILENAME),
        "an empty index's manifest bytes are build independent"
    );
    let manifest: Value =
        serde_json::from_str(&read_artifact(first.path(), MANIFEST_FILENAME)).unwrap();
    assert_eq!(manifest["count"], json!(0));
    assert_eq!(manifest["worker_ids"], json!([]));
    assert_eq!(manifest["content_hash"], json!(EMPTY_CATALOG_HASH));
    assert_eq!(catalog_content_hash(&[]), EMPTY_CATALOG_HASH);
}

#[tokio::test]
async fn a_published_bare_nodes_header_still_loads_as_an_empty_tree() {
    let dir = tempfile::tempdir().unwrap();
    let manifest = IndexManifest {
        mode: MODE_FULL.to_string(),
        count: 0,
        item_paths: Vec::new(),
        worker_ids: Vec::new(),
        item_type: DEFAULT_ITEM_TYPE.to_string(),
        content_hash: catalog_content_hash(&[]),
    };
    write_index(dir.path(), &[], &[], &manifest).await.unwrap();

    fs::write(dir.path().join(TREE_INDEX_FILENAME), "nodes:\n").unwrap();

    let loaded = load_index(dir.path()).await.unwrap();
    assert!(
        loaded.records.is_empty(),
        "a valueless header is an empty tree"
    );
}

#[test]
fn a_non_empty_tree_index_keeps_its_pinned_layout() {
    let nodes = vec![
        IndexNode::branch("Skills").with_description("All skills"),
        IndexNode::leaf("Skills.a", "alpha"),
    ];

    assert_eq!(
        render_tree_yaml(&nodes),
        concat!(
            "nodes:\n",
            "  - cid: \"Skills\"\n",
            "    type: \"branch\"\n",
            "    description: \"All skills\"\n",
            "  - cid: \"Skills.a\"\n",
            "    type: \"leaf\"\n",
            "    worker_id: \"alpha\"\n",
        ),
        "a non-empty document still opens with a bare `nodes:` header"
    );
    assert_eq!(
        parse_tree_yaml(&render_tree_yaml(&nodes)).unwrap().len(),
        nodes.len()
    );
}

// --------------------------------------------------------------------------------------
// 10. The optional `tree_index.html` visualizer artifact
//
// `tree_index.html` is outside the three-artifact contract: it is written only when
// `BuildOptions::write_html` asks for it, and nothing reads it back. The suites below pin
// that boundary at the artifact level — the default build still publishes exactly three
// entries, the opt-in build adds the fourth next to unchanged artifacts, and the rendered
// bytes are deterministic, escaped and cid-ordered when they come from a real published
// tree and catalog. The renderer's own unit-level rules live beside
// `retrieval::build::tree::visualizer`.
// --------------------------------------------------------------------------------------

/// The three default artifact names, sorted, as owned strings.
fn default_artifacts() -> Vec<String> {
    ARTIFACTS.iter().map(|name| (*name).to_string()).collect()
}

/// The four names an index directory holds when the visualizer is on, sorted.
fn artifacts_with_html() -> Vec<String> {
    let mut names = default_artifacts();
    names.push(TREE_HTML_FILENAME.to_string());
    names.sort();
    names
}

/// `true` when `dir` holds a file whose name ends in `.tmp`.
fn has_temporary_sibling(dir: &Path) -> bool {
    fs::read_dir(dir).unwrap().any(|entry| {
        entry
            .unwrap()
            .file_name()
            .to_string_lossy()
            .ends_with(".tmp")
    })
}

/// Reads the visualizer document of an index directory.
fn read_html(dir: &Path) -> String {
    read_artifact(dir, TREE_HTML_FILENAME)
}

/// The published tree and catalog of `dir`, read back through their own parsers.
fn published_tree(dir: &Path) -> (Vec<IndexNode>, Vec<CatalogRecord>) {
    let nodes = parse_tree_yaml(&read_artifact(dir, TREE_INDEX_FILENAME)).unwrap();
    let records = catalog_records(&read_artifact(dir, CATALOG_FILENAME));
    (nodes, records)
}

/// Two skills whose names and descriptions carry every HTML metacharacter.
fn escaped_fixture() -> Vec<SourceItem> {
    vec![
        item("Ampersand", "PDF & Fill <beta>", "documents/pdf"),
        item("Quoted", "Quote \"double\" and 'single'", "documents/pdf"),
    ]
}

#[tokio::test]
async fn the_visualizer_is_an_opt_in_fourth_artifact() {
    let plain = tempfile::tempdir().unwrap();
    let plain_report = IndexBuilder::build(sources(fixture()), plain.path(), BuildConfig::new())
        .await
        .unwrap();

    assert!(!plain_report.html_written, "the visualizer is opt-in");
    assert_eq!(
        directory_entries(plain.path()),
        default_artifacts(),
        "a default build writes the three artifacts and nothing else"
    );
    assert!(
        !plain.path().join(TREE_HTML_FILENAME).exists(),
        "a default build leaves no stray {TREE_HTML_FILENAME}"
    );
    assert!(!has_temporary_sibling(plain.path()));

    let with_html = tempfile::tempdir().unwrap();
    let report = IndexBuilder::build(
        sources(fixture()),
        with_html.path(),
        BuildConfig::new().with_html(true),
    )
    .await
    .unwrap();

    assert!(report.html_written, "the flag is reported on the build");
    assert_eq!(
        directory_entries(with_html.path()),
        artifacts_with_html(),
        "the opt-in build adds exactly one file"
    );
    assert!(
        !has_temporary_sibling(with_html.path()),
        "the visualizer write leaves no `*.tmp` sibling behind"
    );
    assert_eq!(report.node_count, 7);
    assert_eq!(report.record_count, 3);

    // The extra file is derived data: the loader reads the three artifacts only, so the
    // index contract is unchanged by the flag.
    let loaded = load_index(with_html.path()).await.unwrap();
    let plain_loaded = load_index(plain.path()).await.unwrap();
    assert_eq!(loaded.manifest.count, plain_loaded.manifest.count);
    assert_eq!(
        loaded.manifest.content_hash, plain_loaded.manifest.content_hash,
        "the visualizer does not change the catalog the manifest hashes"
    );
    assert_eq!(loaded.records, plain_loaded.records);
    assert_eq!(loaded.root, plain_loaded.root);
}

#[tokio::test]
async fn the_visualizer_lists_the_published_tree_in_cid_order() {
    let dir = tempfile::tempdir().unwrap();
    IndexBuilder::build(
        sources(fixture()),
        dir.path(),
        BuildConfig::new().with_html(true),
    )
    .await
    .unwrap();

    let html = read_html(dir.path());
    assert!(html.starts_with("<!DOCTYPE html>\n<html lang=\"en\">\n"));
    assert!(html.ends_with("</html>\n"));
    assert!(html.contains("<h1>Skills</h1>"));
    assert!(
        html.contains("<p class=\"summary\">4 branches, 3 items</p>"),
        "the summary counts the published nodes"
    );

    for (cid, name) in [
        ("Skills.documents.pdf.docread", "Doc Reader"),
        ("Skills.documents.pdf.pdffill", "PDF Fill"),
        ("Skills.web.websearch", "Web Search"),
    ] {
        assert!(
            html.contains(&format!(
                "<span class=\"leaf-name\">{name}</span><code class=\"cid\">{cid}</code>"
            )),
            "{cid} is listed with its catalog name"
        );
        assert!(
            html.contains(&format!(
                "<p class=\"description\">{name} routes matching requests.</p>"
            )),
            "{cid} is listed with its catalog description"
        );
    }

    let ordered = [
        "Skills.documents.pdf",
        "Skills.documents.pdf.docread",
        "Skills.documents.pdf.pdffill",
        "Skills.web.websearch",
    ];
    let positions: Vec<usize> = ordered
        .iter()
        .map(|cid| {
            html.find(&format!("<code class=\"cid\">{cid}</code>"))
                .unwrap_or_else(|| panic!("{cid} is not rendered"))
        })
        .collect();
    assert!(
        positions.windows(2).all(|pair| pair[0] < pair[1]),
        "the document reflects the published cid order: {positions:?}"
    );
}

#[tokio::test]
async fn writing_the_visualizer_into_a_published_index_is_byte_identical_and_atomic() {
    let dir = tempfile::tempdir().unwrap();
    IndexBuilder::build(sources(fixture()), dir.path(), BuildConfig::new())
        .await
        .unwrap();
    let (nodes, records) = published_tree(dir.path());
    assert_eq!(nodes.len(), 7, "the published tree is the render input");
    assert_eq!(records.len(), 3);

    let expected = render_tree_html(&nodes, &records, "Skills");
    assert_eq!(
        expected,
        render_tree_html(&nodes, &records, "Skills"),
        "rendering the same input twice is byte identical"
    );

    write_tree_html(dir.path(), &nodes, &records, "Skills")
        .await
        .unwrap();
    assert_eq!(
        read_html(dir.path()),
        expected,
        "the written bytes are the rendered bytes"
    );

    write_tree_html(dir.path(), &nodes, &records, "Skills")
        .await
        .unwrap();
    assert_eq!(
        read_html(dir.path()),
        expected,
        "rewriting over the existing document is idempotent"
    );

    assert_eq!(
        directory_entries(dir.path()),
        artifacts_with_html(),
        "the visualizer sits beside the untouched three artifacts"
    );
    assert!(
        !has_temporary_sibling(dir.path()),
        "an atomic write leaves no `*.tmp` sibling behind"
    );
    let html = read_html(dir.path());
    for record in &records {
        assert!(
            html.contains(&format!("<code class=\"cid\">{}</code>", record.cid)),
            "the published leaf {} is reachable in the document",
            record.cid
        );
    }
}

#[tokio::test]
async fn the_visualizer_escapes_every_published_name_and_description() {
    let dir = tempfile::tempdir().unwrap();
    IndexBuilder::build(
        sources(escaped_fixture()),
        dir.path(),
        BuildConfig::new().with_html(true),
    )
    .await
    .unwrap();

    let html = read_html(dir.path());

    assert!(!html.contains("<beta>"), "no raw tag survives the render");
    assert!(
        html.contains("PDF &amp; Fill &lt;beta&gt;"),
        "the catalog name is escaped"
    );
    assert!(
        html.contains("PDF &amp; Fill &lt;beta&gt; routes matching requests."),
        "the catalog description is escaped"
    );
    assert!(
        html.contains("Quote &quot;double&quot; and &#39;single&#39;"),
        "both quote characters are escaped"
    );
    assert!(
        !html.contains("&amp;amp;"),
        "the escaper never double escapes"
    );
    assert!(
        html.contains("<code class=\"cid\">Skills.documents.pdf.ampersand</code>"),
        "the escaped skill still publishes its slug-derived cid"
    );
}

#[test]
fn rendering_is_independent_of_node_and_record_order() {
    let nodes = vec![
        IndexNode::branch("Skills").with_description("All skills"),
        IndexNode::branch("Skills.web").with_description("Web capabilities."),
        IndexNode::leaf("Skills.web.zeta", "Zeta"),
        IndexNode::leaf("Skills.web.alpha", "Alpha"),
        IndexNode::leaf("Skills.docread", "DocRead"),
    ];
    let records = vec![
        CatalogRecord::new("DocRead", "Skills.docread").with_name("Doc Reader"),
        CatalogRecord::new("Zeta", "Skills.web.zeta").with_name("Zeta"),
        CatalogRecord::new("Alpha", "Skills.web.alpha").with_name("Alpha"),
    ];

    let sorted = render_tree_html(&nodes, &records, "Skills");
    let mut shuffled = nodes.clone();
    shuffled.reverse();
    let mut shuffled_records = records.clone();
    shuffled_records.reverse();

    assert_eq!(
        sorted,
        render_tree_html(&shuffled, &shuffled_records, "Skills"),
        "input order does not reach the document"
    );

    let ordered = [
        "Skills.docread",
        "Skills.web",
        "Skills.web.alpha",
        "Skills.web.zeta",
    ];
    let positions: Vec<usize> = ordered
        .iter()
        .map(|cid| {
            sorted
                .find(&format!("<code class=\"cid\">{cid}</code>"))
                .unwrap_or_else(|| panic!("{cid} is not rendered"))
        })
        .collect();
    assert!(
        positions.windows(2).all(|pair| pair[0] < pair[1]),
        "branches and leaves share one cid order: {positions:?}"
    );
}

#[tokio::test]
async fn an_empty_index_still_renders_a_sane_visualizer_document() {
    let dir = tempfile::tempdir().unwrap();
    let report = IndexBuilder::build(
        ItemSources::new("skill"),
        dir.path(),
        BuildConfig::new().with_html(true),
    )
    .await
    .unwrap();

    assert!(report.html_written);
    assert_eq!(
        report.node_count, 1,
        "an empty build still publishes the bare root branch"
    );
    assert_eq!(report.record_count, 0);
    assert_eq!(
        directory_entries(dir.path()),
        artifacts_with_html(),
        "an empty build publishes the same four names"
    );
    assert!(!has_temporary_sibling(dir.path()));

    let html = read_html(dir.path());
    assert!(html.starts_with("<!DOCTYPE html>\n<html lang=\"en\">\n"));
    assert!(html.ends_with("</html>\n"));
    assert!(html.contains("<h1>Skills</h1>"));
    assert!(
        html.contains("<p class=\"summary\">1 branch, 0 items</p>"),
        "the summary counts the bare root branch"
    );
    assert!(html.contains("<code class=\"cid\">Skills</code>"));
    assert!(
        !html.contains("class=\"leaf"),
        "an empty catalog renders no leaf"
    );
    assert!(
        !html.contains("Unplaced items"),
        "an empty catalog has nothing unplaced"
    );

    // A zero-node input — no root branch at all — still renders a valid document.
    let headless = render_tree_html(&[], &[], "Skills");
    assert!(headless.starts_with("<!DOCTYPE html>\n<html lang=\"en\">\n"));
    assert!(headless.ends_with("</html>\n"));
    assert!(headless.contains("<p class=\"summary\">0 branches, 0 items</p>"));
    assert!(headless.contains("No capabilities are indexed in this tree."));
    assert!(
        !headless.contains("<ul"),
        "an empty tree has no list markup"
    );

    write_tree_html(dir.path(), &[], &[], "Skills")
        .await
        .unwrap();
    assert_eq!(
        read_html(dir.path()),
        headless,
        "a zero-node rewrite publishes the empty document"
    );
    assert_eq!(directory_entries(dir.path()), artifacts_with_html());
}
