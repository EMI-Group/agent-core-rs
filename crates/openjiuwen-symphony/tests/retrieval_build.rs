//! Behavior tests for the capability-tree index BUILD half of
//! `openjiuwen_symphony::retrieval::build`: the item sources, the deterministic grouping,
//! the runtime grouping strategies, the preset override, the schema guard, incremental
//! patches, the optional HTML visualizer, and the typed deferral of the synchronous
//! grouping entry points.
//!
//! Everything here goes through the crate's public API only. The fixtures are built with
//! `tempfile::tempdir()` (never the repository working tree), every assertion is
//! deterministic, and nothing depends on the wall clock, on the network, or on hash-map
//! iteration order.
//!
//! The artifact and loading half lives in `tests/retrieval_artifacts.rs`.
#![allow(clippy::unwrap_used, clippy::expect_used)]
#![allow(clippy::float_cmp)]
#![allow(clippy::assert_is_empty)]

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;
use std::sync::Arc;

use openjiuwen_retrieval::HashEmbedding;
use openjiuwen_symphony::discovery::{SkillInventory, SkillRecord, SkillScanConfig};
use openjiuwen_symphony::error::{is_unsupported, ErrorKind};
use openjiuwen_symphony::llm::{FallbackMode, StaticCompletionModel};
use openjiuwen_symphony::model::CapabilityDescriptor;
use openjiuwen_symphony::retrieval::build::grouping::{
    group_items, group_items_with, GroupingConfig, GroupingOutcome, GroupingProvenance,
    GroupingRuntime, GroupingStrategy,
};
use openjiuwen_symphony::retrieval::build::presets::{PresetNode, TreePreset};
use openjiuwen_symphony::retrieval::build::scanner::{
    ItemSources, SourceItem, DIAGNOSTIC_DUPLICATE_ITEM_ID, DIAGNOSTIC_MALFORMED_ROW,
    DIAGNOSTIC_MISSING_ITEM_ID,
};
use openjiuwen_symphony::retrieval::build::schema::{
    TreeSchema, DIAGNOSTIC_BLANK_LABEL, DIAGNOSTIC_CYCLE, DIAGNOSTIC_DUPLICATE_CID,
    DIAGNOSTIC_EXCESSIVE_DEPTH, DIAGNOSTIC_MISSING_WORKER_ID, DIAGNOSTIC_TOO_MANY_NODES,
};
use openjiuwen_symphony::retrieval::build::workflows::{BuildConfig, IndexBuilder};
use openjiuwen_symphony::retrieval::types::{
    IndexNode, RetrieverItem, CATALOG_FILENAME, MANIFEST_FILENAME, MODE_FULL, TREE_HTML_FILENAME,
    TREE_INDEX_FILENAME,
};

/// One scanned skill item: a name, a description, content, and a `store/<id>` inventory
/// path, so a deletion can prune its inventory entry.
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

/// An item whose tags replace the default classification hint.
fn tagged(mut item: SourceItem, tags: &[&str]) -> SourceItem {
    item.tags = tags.iter().map(|tag| (*tag).to_string()).collect();
    item
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

/// Writes `content` to `root/relative`, creating parent directories.
fn write(root: &Path, relative: &str, content: &str) {
    let path = root.join(relative);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::write(path, content).unwrap();
}

/// A minimal `SKILL.md` with front matter and a body.
fn skill_file(name: &str, description: &str) -> String {
    format!("---\nname: {name}\ndescription: {description}\n---\n\n{description}\n")
}

/// Builds `items` into `dir` and re-opens the published index.
async fn build_and_open(dir: &Path, items: Vec<SourceItem>) -> IndexBuilder {
    IndexBuilder::build(sources(items), dir, BuildConfig::new())
        .await
        .unwrap();
    IndexBuilder::open(dir, BuildConfig::new()).await.unwrap()
}

/// The `cid`s of a builder's nodes, in list order.
fn cids(builder: &IndexBuilder) -> Vec<String> {
    builder
        .nodes()
        .iter()
        .map(|node| node.cid.clone())
        .collect()
}

/// `n` items sharing one classification, keyed `tool0`..`tool{n-1}`.
fn tools(count: usize, classification: &str) -> Vec<SourceItem> {
    (0..count)
        .map(|index| {
            item(
                &format!("tool{index}"),
                &format!("Tool {index}"),
                classification,
            )
        })
        .collect()
}

// --------------------------------------------------------------------------------------
// 1. Item sources
// --------------------------------------------------------------------------------------

#[test]
fn descriptors_become_items_with_the_descriptor_json_as_payload() {
    let descriptor = CapabilityDescriptor::new("skill", "pdf", "PDF Fill")
        .with_description("fills pdf forms")
        .with_classification("documents/pdf")
        .with_tags(["pdf", "docs", "pdf"])
        .with_semantic_content("# Body\nfills forms");

    let sources = ItemSources::from_descriptors([descriptor.clone()]);

    assert_eq!(sources.item_type, "skill");
    assert_eq!(sources.worker_ids(), vec!["pdf"]);
    assert!(sources.diagnostics.is_empty());
    let entry = &sources.items[0];
    assert_eq!(entry.item.item_id, "pdf");
    assert_eq!(
        entry.item.payload,
        serde_json::to_string(&descriptor).unwrap(),
        "the payload is the descriptor JSON"
    );
    assert_eq!(entry.classification, "documents/pdf");
    assert_eq!(
        entry.tags,
        vec!["docs", "pdf"],
        "tags are sorted and unique"
    );
    assert_eq!(entry.content, "# Body\nfills forms");
    assert_eq!(entry.skill_path, "capability://skill/pdf");
    assert_eq!(entry.item_type, "skill");
    assert_eq!(sources.item_paths, vec!["capability://skill/pdf"]);
}

#[test]
fn a_skill_inventory_becomes_one_item_per_record() {
    let inventory = SkillInventory::new(vec![
        SkillRecord::new("beta", "Beta").with_skill_file("beta/SKILL.md"),
        SkillRecord::new("alpha", "Alpha")
            .with_description("adds numbers")
            .with_skill_file("alpha/SKILL.md")
            .with_version("1.2.0")
            .with_author("ada"),
    ]);

    let sources = ItemSources::from_inventory(&inventory);

    assert_eq!(sources.worker_ids(), vec!["alpha", "beta"]);
    let alpha = &sources.items[0];
    assert_eq!(alpha.name, "Alpha");
    assert_eq!(alpha.description, "adds numbers");
    assert_eq!(alpha.skill_path, "alpha/SKILL.md");
    assert_eq!(alpha.item.item_id, "alpha");
    assert_eq!(
        alpha.metadata.get("version").map(String::as_str),
        Some("1.2.0")
    );
    assert_eq!(
        alpha.metadata.get("author").map(String::as_str),
        Some("ada")
    );
    assert!(
        alpha.item.payload.contains("\"worker_id\":\"alpha\""),
        "the payload is the record JSON: {}",
        alpha.item.payload
    );
    assert_eq!(
        sources.item_paths,
        vec!["alpha/SKILL.md", "beta/SKILL.md"],
        "the inventory paths are carried in `worker_id` order"
    );
}

#[test]
fn a_repeated_descriptor_id_is_diagnosed_and_the_first_one_wins() {
    let first = CapabilityDescriptor::new("skill", "pdf", "PDF Fill")
        .with_description("fills pdf forms")
        .with_semantic_content("# Body\nfills forms");
    let second =
        CapabilityDescriptor::new("skill", "pdf", "PDF Reader").with_description("reads pdf forms");

    let sources = ItemSources::from_descriptors([first, second]);

    assert_eq!(sources.worker_ids(), vec!["pdf"]);
    assert_eq!(sources.items.len(), 1, "one leaf, one item");
    assert_eq!(
        sources.items[0].name, "PDF Fill",
        "the first descriptor wins"
    );
    let duplicates: Vec<&str> = sources
        .diagnostics
        .iter()
        .filter(|diagnostic| diagnostic.code == DIAGNOSTIC_DUPLICATE_ITEM_ID)
        .map(|diagnostic| diagnostic.subject.as_str())
        .collect();
    assert_eq!(duplicates, vec!["pdf"]);
    let diagnostic = sources
        .diagnostics
        .iter()
        .find(|diagnostic| diagnostic.code == DIAGNOSTIC_DUPLICATE_ITEM_ID)
        .expect("the repeated id is reported");
    assert!(
        !diagnostic.is_error(),
        "a duplicated id is a warning: the first descriptor is kept"
    );
}

#[test]
fn a_repeated_inventory_record_is_diagnosed_and_the_first_one_wins() {
    let inventory = SkillInventory::new(vec![
        SkillRecord::new("alpha", "Alpha")
            .with_description("adds numbers")
            .with_skill_file("alpha/SKILL.md"),
        SkillRecord::new("alpha", "Alpha Again")
            .with_description("overwrites the first")
            .with_skill_file("beta/SKILL.md"),
    ]);

    let sources = ItemSources::from_inventory(&inventory);

    assert_eq!(sources.worker_ids(), vec!["alpha"]);
    assert_eq!(sources.items.len(), 1);
    assert_eq!(
        sources.items[0].name, "Alpha",
        "the first record in worker_id then skill_file order wins"
    );
    assert_eq!(
        sources.item_paths,
        vec!["alpha/SKILL.md"],
        "the dropped record contributes no inventory entry"
    );
    let duplicates: Vec<&str> = sources
        .diagnostics
        .iter()
        .filter(|diagnostic| diagnostic.code == DIAGNOSTIC_DUPLICATE_ITEM_ID)
        .map(|diagnostic| diagnostic.subject.as_str())
        .collect();
    assert_eq!(duplicates, vec!["alpha"]);
    assert!(
        sources
            .diagnostics
            .iter()
            .filter(|diagnostic| diagnostic.code == DIAGNOSTIC_DUPLICATE_ITEM_ID)
            .all(|diagnostic| !diagnostic.is_error()),
        "a duplicated id is a warning"
    );
}

#[test]
fn jsonl_rows_survive_missing_and_malformed_neighbours() {
    let text = concat!(
        r#"{"id":"pdf","name":"PDF Fill","tags":["docs","pdf"],"classification":"documents/pdf"}"#,
        "\n",
        r#"{"contentExtendParam":{"skillId":"legacy","skillName":"Legacy Skill"}}"#,
        "\n",
        r#"{"name":"no usable id"}"#,
        "\n",
        "not json at all\n",
        r#"{"id":"pdf","name":"Duplicate PDF"}"#,
        "\n",
    );

    let sources = ItemSources::from_jsonl(text);

    assert_eq!(sources.worker_ids(), vec!["legacy", "pdf"]);
    assert_eq!(sources.items.len(), 2, "the good rows survive");

    let legacy = sources
        .items
        .iter()
        .find(|entry| entry.worker_id == "legacy")
        .expect("the legacy contentExtendParam.skillId row is indexed");
    assert_eq!(legacy.name, "Legacy Skill");
    assert_eq!(legacy.skill_path, "jsonl://skill/legacy");

    let pdf = sources
        .items
        .iter()
        .find(|entry| entry.worker_id == "pdf")
        .expect("the valid row is indexed");
    assert_eq!(pdf.name, "PDF Fill", "the first duplicate wins");
    assert_eq!(pdf.tags, vec!["docs", "pdf"]);
    assert_eq!(pdf.classification, "documents/pdf");
    assert_eq!(pdf.skill_path, "jsonl://skill/pdf");
    assert_eq!(
        sources.item_paths,
        vec!["jsonl://skill/pdf", "jsonl://skill/legacy"],
        "the synthetic paths are recorded in first-seen order"
    );

    let codes: Vec<(&str, &str)> = sources
        .diagnostics
        .iter()
        .map(|diagnostic| (diagnostic.code.as_str(), diagnostic.subject.as_str()))
        .collect();
    assert_eq!(
        codes,
        vec![
            (DIAGNOSTIC_MISSING_ITEM_ID, "line:3"),
            (DIAGNOSTIC_MALFORMED_ROW, "line:4"),
            (DIAGNOSTIC_DUPLICATE_ITEM_ID, "pdf"),
        ],
        "diagnostics are ordered by (subject, code) and name the offending row"
    );
    assert!(sources.diagnostics[0].is_error());
    assert!(sources.diagnostics[1].is_error());
    assert!(
        !sources.diagnostics[2].is_error(),
        "a duplicated id is a warning: the first row is kept"
    );
}

#[tokio::test]
async fn a_skill_directory_reports_the_broken_folder_and_keeps_the_good_ones() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    write(root, "alpha/SKILL.md", &skill_file("Alpha", "Adds numbers"));
    write(
        root,
        "beta/SKILL.md",
        &skill_file("Beta", "Translates text"),
    );
    write(
        root,
        "broken/SKILL.md",
        "---\nname: Broken\n\nno terminator\n",
    );

    let sources = ItemSources::from_skill_directory(root, &SkillScanConfig::default())
        .await
        .expect("a broken skill folder does not fail the scan");

    assert_eq!(sources.worker_ids(), vec!["alpha", "beta"]);
    let alpha = &sources.items[0];
    assert_eq!(alpha.name, "Alpha");
    assert_eq!(alpha.content, "Adds numbers", "the body feeds `content`");
    assert!(
        Path::new(&alpha.skill_path).starts_with(root),
        "the relative skill path is joined onto the scan root: {}",
        alpha.skill_path
    );
    assert!(
        sources
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "unterminated_front_matter"
                && diagnostic.subject.contains("broken")),
        "the broken folder is reported, not dropped silently: {:?}",
        sources.diagnostics
    );
    let canonical = fs::canonicalize(root).unwrap();
    assert_eq!(
        sources.item_paths,
        vec![canonical.to_string_lossy().to_string()],
        "the scan root is the single inventory entry"
    );
}

// --------------------------------------------------------------------------------------
// 2. Deterministic grouping
// --------------------------------------------------------------------------------------

#[test]
fn grouping_two_runs_over_the_same_items_are_equal() {
    let items = tools(12, "documents");
    let config = GroupingConfig::default();

    let first = group_items(&items, &config).unwrap();
    let second = group_items(&items, &config).unwrap();

    assert_eq!(first, second);
    assert_eq!(first.len(), items.len());
    let ids: Vec<&str> = first
        .iter()
        .map(|assignment| assignment.worker_id.as_str())
        .collect();
    let mut sorted = ids.clone();
    sorted.sort_unstable();
    assert_eq!(ids, sorted, "assignments are ordered by worker id");
}

#[tokio::test]
async fn two_builds_of_the_same_sources_are_byte_identical() {
    let first = tempfile::tempdir().unwrap();
    let second = tempfile::tempdir().unwrap();
    let mut items = vec![
        item("PdfFill", "PDF Fill", "documents/pdf"),
        tagged(item("Tagged", "Tagged Skill", ""), &["web-search"]),
        item("Reader", "Reader Helper", ""),
    ];
    IndexBuilder::build(sources(items.clone()), first.path(), BuildConfig::new())
        .await
        .unwrap();
    items.reverse();
    IndexBuilder::build(sources(items), second.path(), BuildConfig::new())
        .await
        .unwrap();

    for name in [
        openjiuwen_symphony::retrieval::types::TREE_INDEX_FILENAME,
        openjiuwen_symphony::retrieval::types::CATALOG_FILENAME,
    ] {
        let left = fs::read(first.path().join(name)).unwrap();
        let right = fs::read(second.path().join(name)).unwrap();
        assert_eq!(left, right, "{name} differs between two builds");
    }
}

#[tokio::test]
async fn categories_come_from_classification_tags_then_name_tokens() {
    let dir = tempfile::tempdir().unwrap();
    let builder = build_and_open(
        dir.path(),
        vec![
            item("PdfFill", "PDF Fill", "documents/pdf"),
            tagged(item("Tagged", "Tagged Skill", ""), &["web-search"]),
            item("Reader", "Reader Helper", ""),
        ],
    )
    .await;

    let cids = cids(&builder);
    assert!(
        cids.contains(&"Skills.documents.pdf.pdffill".to_string()),
        "the classification path shapes the cid: {cids:?}"
    );
    assert!(
        cids.contains(&"Skills.web-search.tagged".to_string()),
        "the first tag shapes the cid when the classification is blank: {cids:?}"
    );
    assert!(
        cids.contains(&"Skills.reader.reader".to_string()),
        "the leading name token shapes the cid when there is no tag: {cids:?}"
    );
}

#[test]
fn grouping_bounds_every_branch_to_max_children() {
    let items: Vec<SourceItem> = (0..24)
        .map(|index| {
            item(
                &format!("w{index:02}"),
                &format!("Capability {index}"),
                &format!("group{index:02}"),
            )
        })
        .collect();
    let config = GroupingConfig::default().with_max_children(4);

    let assignments = group_items(&items, &config).unwrap();

    assert_eq!(assignments.len(), 24);
    let mut children: BTreeMap<Vec<String>, BTreeSet<String>> = BTreeMap::new();
    for assignment in &assignments {
        let mut parent: Vec<String> = Vec::new();
        for segment in &assignment.path {
            children
                .entry(parent.clone())
                .or_default()
                .insert(segment.clone());
            parent.push(segment.clone());
        }
    }
    assert!(!children.is_empty(), "every item was placed at the root");
    for (parent, branches) in &children {
        assert!(
            branches.len() <= 4,
            "branch {parent:?} has {} children: {branches:?}",
            branches.len()
        );
    }
    assert!(
        children[&Vec::<String>::new()].contains("other"),
        "the folded classifications collected in `other`"
    );
}

// --------------------------------------------------------------------------------------
// 3. Preset override
// --------------------------------------------------------------------------------------

#[tokio::test]
async fn a_nested_preset_with_blank_child_cids_wins_over_the_classification() {
    let preset = TreePreset::from_yaml(concat!(
        "nodes:\n",
        "  - cid: \"Skills\"\n",
        "    type: \"branch\"\n",
        "    description: \"Capability tree\"\n",
        "    children:\n",
        "      - description: \"Documents\"\n",
        "        children:\n",
        "          - description: \"PDF forms\"\n",
    ))
    .unwrap();
    let derived: Vec<String> = preset
        .flatten()
        .iter()
        .map(|node| node.cid.clone())
        .collect();
    assert_eq!(
        derived,
        vec!["Skills", "Skills.documents", "Skills.documents.pdf-forms"],
        "the blank child cids are derived from their descriptions"
    );
    assert_eq!(
        preset.paths(),
        vec![
            vec!["Skills".to_string()],
            vec!["Skills".to_string(), "documents".to_string()],
            vec![
                "Skills".to_string(),
                "documents".to_string(),
                "pdf-forms".to_string()
            ],
        ]
    );

    let plain = tempfile::tempdir().unwrap();
    let without = build_and_open(
        plain.path(),
        vec![item("PdfFill", "PDF Fill", "documents/pdf")],
    )
    .await;
    assert!(
        cids(&without).contains(&"Skills.documents.pdf.pdffill".to_string()),
        "without a preset the classification shapes the path: {:?}",
        cids(&without)
    );

    let shaped = tempfile::tempdir().unwrap();
    IndexBuilder::build(
        sources(vec![item("PdfFill", "PDF Fill", "documents/pdf")]),
        shaped.path(),
        BuildConfig::new().with_preset(preset),
    )
    .await
    .unwrap();
    let builder = IndexBuilder::open(shaped.path(), BuildConfig::new())
        .await
        .unwrap();
    let leaves: Vec<&IndexNode> = builder
        .nodes()
        .iter()
        .filter(|node| node.is_leaf())
        .collect();
    assert_eq!(leaves.len(), 1);
    assert_eq!(
        leaves[0].cid, "Skills.documents.pdf-forms.pdffill",
        "the leaf sits on the preset path, not on the classification path"
    );
    assert!(cids(&builder).contains(&"Skills.documents.pdf-forms".to_string()));
    assert!(
        !cids(&builder).contains(&"Skills.skills.documents.pdf-forms".to_string()),
        "the preset's leading Skills root is not repeated below the tree root: {:?}",
        cids(&builder)
    );
    assert!(
        !cids(&builder)
            .iter()
            .any(|cid| cid == "Skills.skills" || cid.starts_with("Skills.skills.")),
        "no doubled root segment anywhere in the tree: {:?}",
        cids(&builder)
    );
}

#[tokio::test]
async fn a_duplicate_preset_cid_aborts_the_build_before_writing() {
    let dir = tempfile::tempdir().unwrap();
    let preset = TreePreset {
        nodes: vec![
            PresetNode {
                cid: "Skills".to_string(),
                node_type: "branch".to_string(),
                description: "one".to_string(),
                ..PresetNode::default()
            },
            PresetNode {
                cid: "Skills".to_string(),
                node_type: "branch".to_string(),
                description: "two".to_string(),
                ..PresetNode::default()
            },
        ],
    };

    let error = IndexBuilder::build(
        sources(tools(2, "documents")),
        dir.path(),
        BuildConfig::new().with_preset(preset),
    )
    .await
    .unwrap_err();

    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains(DIAGNOSTIC_DUPLICATE_CID));
    assert!(!dir
        .path()
        .join(openjiuwen_symphony::retrieval::types::TREE_INDEX_FILENAME)
        .exists());
}

// --------------------------------------------------------------------------------------
// 4. Schema validation
// --------------------------------------------------------------------------------------

#[test]
fn schema_reports_a_duplicate_cid_and_names_it() {
    let nodes = vec![
        IndexNode::branch("Skills").with_description("root"),
        IndexNode::leaf("Skills.a", "alpha"),
        IndexNode::leaf("Skills.a", "beta"),
    ];

    let diagnostics = TreeSchema::new().validate(&nodes);
    let duplicates: Vec<&str> = diagnostics
        .iter()
        .filter(|diagnostic| diagnostic.code == DIAGNOSTIC_DUPLICATE_CID)
        .map(|diagnostic| diagnostic.cid.as_str())
        .collect();
    assert_eq!(duplicates, vec!["Skills.a"]);

    let error = TreeSchema::new().validate_or_error(&nodes).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains(DIAGNOSTIC_DUPLICATE_CID));
    assert!(error.message().contains("Skills.a"));
}

#[test]
fn schema_reports_a_cycle_in_a_nested_preset() {
    let preset = TreePreset {
        nodes: vec![PresetNode {
            cid: "Skills".to_string(),
            node_type: "branch".to_string(),
            description: "root".to_string(),
            children: vec![PresetNode {
                cid: "Skills".to_string(),
                node_type: "branch".to_string(),
                description: "child repeating its ancestor".to_string(),
                ..PresetNode::default()
            }],
            ..PresetNode::default()
        }],
    };

    let diagnostics = preset.validate(&TreeSchema::new());
    assert!(
        diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == DIAGNOSTIC_CYCLE),
        "a repeated ancestor cid is a cycle: {diagnostics:?}"
    );

    let error = TreeSchema::new()
        .validate_preset_or_error(&preset)
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains(DIAGNOSTIC_CYCLE));
    assert!(error.message().contains("Skills"));
}

#[test]
fn schema_reports_a_blank_label_and_a_missing_worker_id() {
    let nodes = vec![IndexNode::branch("Skills"), IndexNode::leaf("Skills.a", "")];

    let diagnostics = TreeSchema::new().validate(&nodes);
    let codes: Vec<(&str, &str)> = diagnostics
        .iter()
        .map(|diagnostic| (diagnostic.cid.as_str(), diagnostic.code.as_str()))
        .collect();
    assert_eq!(
        codes,
        vec![
            ("Skills", DIAGNOSTIC_BLANK_LABEL),
            ("Skills.a", DIAGNOSTIC_MISSING_WORKER_ID),
        ]
    );

    let error = TreeSchema::new().validate_or_error(&nodes).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains(DIAGNOSTIC_BLANK_LABEL));
    assert!(error.message().contains("Skills"));
}

#[test]
fn schema_reports_excessive_depth() {
    let nodes = vec![
        IndexNode::branch("Skills").with_description("root"),
        IndexNode::branch("Skills.a").with_description("a"),
        IndexNode::leaf("Skills.a.b", "worker"),
    ];

    let diagnostics = TreeSchema::new().with_max_depth(2).validate(&nodes);
    let depths: Vec<&str> = diagnostics
        .iter()
        .filter(|diagnostic| diagnostic.code == DIAGNOSTIC_EXCESSIVE_DEPTH)
        .map(|diagnostic| diagnostic.cid.as_str())
        .collect();
    assert_eq!(depths, vec!["Skills.a.b"]);

    let error = TreeSchema::new()
        .with_max_depth(2)
        .validate_or_error(&nodes)
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains(DIAGNOSTIC_EXCESSIVE_DEPTH));
    assert!(error.message().contains("Skills.a.b"));
    assert!(TreeSchema::new().validate_or_error(&nodes).is_ok());
}

#[tokio::test]
async fn a_depth_or_size_bound_aborts_the_build_before_writing() {
    let deep = tempfile::tempdir().unwrap();
    let error = IndexBuilder::build(
        sources(tools(2, "documents/pdf")),
        deep.path(),
        BuildConfig::new().with_schema(TreeSchema::new().with_max_depth(2)),
    )
    .await
    .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(
        error.message().contains(DIAGNOSTIC_EXCESSIVE_DEPTH),
        "a tree deeper than `max_depth` must not be published: {}",
        error.message()
    );
    assert_eq!(
        fs::read_dir(deep.path()).unwrap().count(),
        0,
        "an over-deep build writes nothing"
    );

    let crowded = tempfile::tempdir().unwrap();
    let error = IndexBuilder::build(
        sources(tools(3, "documents")),
        crowded.path(),
        BuildConfig::new().with_schema(TreeSchema::new().with_max_nodes(3)),
    )
    .await
    .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(
        error.message().contains(DIAGNOSTIC_TOO_MANY_NODES),
        "a tree larger than `max_nodes` must not be published: {}",
        error.message()
    );
    assert_eq!(
        fs::read_dir(crowded.path()).unwrap().count(),
        0,
        "an over-large build writes nothing"
    );
}

// --------------------------------------------------------------------------------------
// 7. Incremental patches
// --------------------------------------------------------------------------------------

#[tokio::test]
async fn a_failed_add_leaves_the_builder_unchanged() {
    let dir = tempfile::tempdir().unwrap();
    build_and_open(dir.path(), tools(3, "documents")).await;
    let config = BuildConfig::new()
        .with_grouping(GroupingConfig::default().with_strategy(GroupingStrategy::FaissKmeans));
    let mut builder = IndexBuilder::open(dir.path(), config).await.unwrap();
    let nodes = builder.nodes().to_vec();
    let records = builder.records().to_vec();
    let item_paths = builder.item_paths().to_vec();

    // Re-adding an indexed id is the patch that used to drop the item's leaf and record
    // *before* the grouping strategy refused to place it.
    let error = builder
        .add(&sources(vec![item("tool1", "Tool One", "documents/pdf")]))
        .await
        .unwrap_err();

    assert!(is_unsupported(&error), "{error:?} was not Unsupported");
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert_eq!(
        builder.nodes(),
        nodes.as_slice(),
        "a failed add must not drop the node it was about to replace"
    );
    assert_eq!(
        builder.records(),
        records.as_slice(),
        "a failed add must not drop the catalog record it was about to replace"
    );
    assert_eq!(
        builder.item_paths(),
        item_paths.as_slice(),
        "a failed add must not extend the inventory"
    );
}

#[tokio::test]
async fn open_then_add_appends_a_leaf_and_a_record() {
    let dir = tempfile::tempdir().unwrap();
    let mut builder = build_and_open(dir.path(), tools(8, "documents")).await;
    let original: Vec<String> = cids(&builder);
    assert_eq!(builder.records().len(), 8);
    assert_eq!(builder.item_paths().len(), 8);

    let mut added = sources(vec![item("imagegen", "Image Generator", "media/images")]);
    added.item_paths.push("store/imagegen".to_string());
    let patch = builder.add(&added).await.unwrap();

    assert_eq!(patch.added, vec!["imagegen"]);
    assert_eq!(patch.removed, Vec::<String>::new());
    assert_eq!(patch.rebuilt_branches, Vec::<String>::new());
    assert!(
        !patch.escalated,
        "one new record out of eight stays below the 0.25 ratio"
    );
    assert!((patch.change_ratio - 0.125).abs() < 1e-9);

    assert_eq!(
        builder
            .nodes()
            .iter()
            .filter(|node| node.is_leaf() && node.worker_id == "imagegen")
            .count(),
        1
    );
    assert_eq!(builder.records().len(), 9);
    let record = builder
        .records()
        .iter()
        .find(|record| record.worker_id == "imagegen")
        .expect("the new record is in the catalog");
    assert_eq!(record.cid, "Skills.media.images.imagegen");
    assert_eq!(
        record.branch_path,
        vec!["Skills", "media", "images"]
            .into_iter()
            .map(str::to_string)
            .collect::<Vec<String>>()
    );

    let after: BTreeSet<String> = builder
        .nodes()
        .iter()
        .map(|node| node.cid.clone())
        .collect();
    for cid in &original {
        assert!(after.contains(cid), "{cid} vanished from the tree");
    }
    assert!(builder
        .item_paths()
        .iter()
        .any(|path| path == "store/imagegen"));
    assert_eq!(builder.item_paths().len(), 9, "the inventory was extended");
}

#[tokio::test]
async fn adding_an_existing_worker_id_replaces_its_record_and_leaf() {
    let dir = tempfile::tempdir().unwrap();
    let mut builder = build_and_open(dir.path(), tools(8, "documents")).await;
    let previous = builder
        .records()
        .iter()
        .find(|record| record.worker_id == "tool3")
        .expect("tool3 was indexed")
        .cid
        .clone();

    let patch = builder
        .add(&sources(vec![item("tool3", "Tool 3", "media")]))
        .await
        .unwrap();

    assert_eq!(patch.added, vec!["tool3"]);
    assert_eq!(builder.records().len(), 8, "a re-add replaces, not appends");
    assert_eq!(
        builder
            .records()
            .iter()
            .filter(|record| record.worker_id == "tool3")
            .count(),
        1
    );
    let record = builder
        .records()
        .iter()
        .find(|record| record.worker_id == "tool3")
        .unwrap();
    assert_eq!(record.cid, "Skills.media.tool3");
    assert_ne!(record.cid, previous);
    assert_eq!(
        builder
            .nodes()
            .iter()
            .filter(|node| node.is_leaf() && node.worker_id == "tool3")
            .count(),
        1,
        "no stale leaf survives the replacement"
    );
    assert!(builder.nodes().iter().all(|node| node.cid != previous));
}

#[tokio::test]
async fn delete_prunes_the_leaf_its_branch_and_its_inventory_entry() {
    let dir = tempfile::tempdir().unwrap();
    let mut builder = build_and_open(
        dir.path(),
        vec![
            item("tool0", "Tool 0", "documents"),
            item("tool1", "Tool 1", "documents"),
            item("websearch", "Web Search", "web"),
        ],
    )
    .await;
    assert!(cids(&builder).contains(&"Skills.web".to_string()));

    let patch = builder.delete(&["websearch".to_string()]).await.unwrap();

    assert_eq!(patch.removed, vec!["websearch"]);
    assert_eq!(patch.added, Vec::<String>::new());
    assert_eq!(
        patch.rebuilt_branches,
        vec!["Skills.web"],
        "the branch the delete emptied is pruned and reported"
    );
    assert!(builder.nodes().iter().all(|node| node.cid != "Skills.web"));
    assert!(builder
        .nodes()
        .iter()
        .all(|node| node.worker_id != "websearch"));
    assert_eq!(builder.records().len(), 2);
    assert!(builder
        .records()
        .iter()
        .all(|record| record.worker_id != "websearch"));
    assert_eq!(
        builder.item_paths(),
        &["store/tool0".to_string(), "store/tool1".to_string()],
        "the inventory entry of the removed item is gone"
    );

    let noop = builder.delete(&["absent".to_string()]).await.unwrap();
    assert_eq!(noop.removed, Vec::<String>::new());
    assert!(!noop.escalated);
    assert_eq!(noop.change_ratio, 0.0);
    assert_eq!(builder.records().len(), 2);
}

#[tokio::test]
async fn deleting_one_of_three_items_escalates_at_the_default_ratio() {
    let dir = tempfile::tempdir().unwrap();
    let mut builder = build_and_open(
        dir.path(),
        vec![
            item("tool0", "Tool 0", "documents"),
            item("tool1", "Tool 1", "documents"),
            item("tool2", "Tool 2", "web"),
        ],
    )
    .await;

    let patch = builder.delete(&["tool1".to_string()]).await.unwrap();

    assert!(
        patch.escalated,
        "one of three records exceeds the 0.25 ratio"
    );
    assert!((patch.change_ratio - 1.0 / 3.0).abs() < 1e-9);
    assert_eq!(builder.config().escalation_ratio, 0.25);
    assert_eq!(builder.records().len(), 2);
}

#[tokio::test]
async fn rebuild_subtree_re_derives_the_children_and_keeps_record_identity() {
    let dir = tempfile::tempdir().unwrap();
    let mut builder = build_and_open(
        dir.path(),
        vec![
            item("PdfFill", "PDF Fill", "documents"),
            item("DocRead", "Doc Reader", "documents"),
        ],
    )
    .await;
    let before: BTreeMap<String, openjiuwen_symphony::retrieval::types::CatalogRecord> = builder
        .records()
        .iter()
        .map(|record| (record.worker_id.clone(), record.clone()))
        .collect();
    assert_eq!(before.len(), 2);
    for record in before.values() {
        assert_eq!(
            record.branch_path,
            vec!["Skills", "documents"]
                .into_iter()
                .map(str::to_string)
                .collect::<Vec<String>>()
        );
    }

    let patch = builder.rebuild_subtree("Skills.documents").await.unwrap();

    assert_eq!(patch.added, Vec::<String>::new());
    assert_eq!(patch.removed, Vec::<String>::new());
    assert!(!patch.escalated);
    assert_eq!(
        patch.rebuilt_branches,
        vec![
            "Skills.documents.docread".to_string(),
            "Skills.documents.pdffill".to_string()
        ],
        "the children of the branch are re-derived"
    );

    assert_eq!(builder.records().len(), 2);
    for record in builder.records() {
        let old = before
            .get(&record.worker_id)
            .expect("every record keeps its worker id");
        assert_eq!(record.name, old.name);
        assert_eq!(record.description, old.description);
        assert_eq!(record.skill_path, old.skill_path);
        assert_eq!(record.metadata, old.metadata);
        assert!(
            record.cid.starts_with(&format!("{}.", old.cid)),
            "{} did not move below {}",
            record.cid,
            old.cid
        );
        let mut expected_path = old.branch_path.clone();
        expected_path.push(old.cid.rsplit('.').next().unwrap().to_string());
        assert_eq!(record.branch_path, expected_path);
        assert_eq!(record.category, record.branch_path.join("."));
        assert!(
            record.retrieval_text.contains(&record.cid),
            "the retrieval text follows the new cid"
        );
    }

    let snapshot = builder.records().to_vec();
    builder.rebuild_subtree("Skills.documents").await.unwrap();
    assert_eq!(
        builder.records(),
        snapshot.as_slice(),
        "rebuilding the same branch twice reproduces the same records"
    );
}

#[tokio::test]
async fn rebuild_subtree_rejects_an_unknown_branch() {
    let dir = tempfile::tempdir().unwrap();
    let mut builder = build_and_open(dir.path(), tools(3, "documents")).await;

    let error = builder.rebuild_subtree("Skills.absent").await.unwrap_err();

    assert_eq!(error.kind(), ErrorKind::NotFound);
    assert!(error.message().contains("Skills.absent"));
    assert_eq!(
        builder.records().len(),
        3,
        "a failed rebuild changes nothing"
    );
}

// --------------------------------------------------------------------------------------
// 9. Runtime grouping strategies, their synchronous deferral, and the HTML visualizer
// --------------------------------------------------------------------------------------

/// The three artifacts every build publishes, sorted by file name.
const ARTIFACTS: [&str; 3] = [CATALOG_FILENAME, MANIFEST_FILENAME, TREE_INDEX_FILENAME];

/// The three-artifact contract as the sorted file names a build publishes.
fn artifact_names() -> Vec<String> {
    let mut names: Vec<String> = ARTIFACTS.iter().map(|name| (*name).to_string()).collect();
    names.sort();
    names
}

/// The sorted file names `dir` holds.
fn published_names(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = fs::read_dir(dir)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().to_string())
        .collect();
    names.sort();
    names
}

/// The `(worker_id, path)` pairs of one grouping run, in assignment order.
fn grouped_paths(outcome: &GroupingOutcome) -> Vec<(String, Vec<String>)> {
    outcome
        .assignments
        .iter()
        .map(|assignment| (assignment.worker_id.clone(), assignment.path.clone()))
        .collect()
}

/// The distinct branch paths one grouping run produced.
fn distinct_paths(outcome: &GroupingOutcome) -> BTreeSet<Vec<String>> {
    outcome
        .assignments
        .iter()
        .map(|assignment| assignment.path.clone())
        .collect()
}

/// `segments` as one branch path.
fn path_of(segments: &[&str]) -> Vec<String> {
    segments
        .iter()
        .map(|segment| (*segment).to_string())
        .collect()
}

/// Every runtime strategy is implemented, and only the synchronous entry points defer.
///
/// The deferral that remains is the one a synchronous call cannot avoid: it cannot await an
/// embedder or a completion model, so it names `group_items_with` and the crate's deferred
/// scope instead of silently grouping with the deterministic rule.
#[tokio::test]
async fn the_runtime_grouping_strategies_are_supported_and_only_the_sync_entries_defer() {
    let items = tools(3, "documents");
    for strategy in [
        GroupingStrategy::LlmGrouping,
        GroupingStrategy::FaissKmeans,
        GroupingStrategy::Kmeans,
    ] {
        assert!(strategy.is_supported(), "{strategy:?} is implemented");
        assert!(
            strategy.requires_runtime(),
            "{strategy:?} needs a supplied runtime"
        );
    }
    for strategy in [
        GroupingStrategy::Deterministic,
        GroupingStrategy::CategoryPath,
        GroupingStrategy::NameTokens,
    ] {
        assert!(strategy.is_supported(), "{strategy:?} is implemented");
        assert!(
            !strategy.requires_runtime(),
            "{strategy:?} needs no runtime"
        );
        let config = GroupingConfig::default().with_strategy(strategy);
        assert!(group_items(&items, &config).is_ok(), "{strategy:?}");
    }
    for (strategy, reason) in [
        (GroupingStrategy::LlmGrouping, "LLM-guided tree grouping"),
        (GroupingStrategy::FaissKmeans, "faiss k-means tree grouping"),
        (GroupingStrategy::Kmeans, "k-means tree grouping"),
    ] {
        let config = GroupingConfig::default().with_strategy(strategy);
        let error = group_items(&items, &config)
            .expect_err("a synchronous call cannot await a runtime strategy");
        assert!(is_unsupported(&error), "{strategy:?}: {error}");
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        let message = error.message();
        assert!(message.starts_with(reason), "{strategy:?}: {message}");
        assert!(
            message.contains("group_items_with"),
            "{strategy:?} names the async entry point: {message}"
        );
        assert!(
            message.contains("deferred scope"),
            "{strategy:?} points at the deferred scope: {message}"
        );

        let dir = tempfile::tempdir().unwrap();
        let error = IndexBuilder::build(
            sources(tools(3, "documents")),
            dir.path(),
            BuildConfig::new().with_grouping(config),
        )
        .await
        .expect_err("a build without a runtime keeps the typed deferral");
        assert_eq!(error.kind(), ErrorKind::Unsupported, "{strategy:?}");
        assert!(error.message().contains(reason), "{strategy:?}: {error}");
        assert!(
            error.message().contains("group_items_with"),
            "{strategy:?}: {error}"
        );
        assert_eq!(
            published_names(dir.path()),
            Vec::<String>::new(),
            "{strategy:?} published an artifact instead of deferring"
        );
    }
}

/// The embedding-backed strategy places the same items the same way on every run.
#[tokio::test]
async fn kmeans_grouping_through_an_embedding_is_deterministic() {
    let items = vec![
        item("charts", "Chart Builder", "charts"),
        item("mail", "Mail Digest", "mail"),
        item("reader", "PDF Parser", "documents/pdf"),
        item("sheets", "Spreadsheet Reader", "documents/sheets"),
        item("web", "Web Search", "web/search"),
    ];
    let runtime = GroupingRuntime::new().with_embedding(Arc::new(HashEmbedding::new(16)));
    let config = GroupingConfig::default()
        .with_strategy(GroupingStrategy::Kmeans)
        .with_max_children(3)
        .with_max_depth(3)
        .with_clusters(Some(2));
    let first = group_items_with(&items, &config, &runtime).await.unwrap();
    let second = group_items_with(&items, &config, &runtime).await.unwrap();
    assert_eq!(first, second, "two runs over the same inventory differ");
    assert_eq!(first.strategy, GroupingStrategy::Kmeans);
    assert_eq!(first.provenance, GroupingProvenance::Embedding);
    assert_eq!(first.assignments.len(), items.len(), "one path per item");
    let ids: Vec<&str> = first
        .assignments
        .iter()
        .map(|assignment| assignment.worker_id.as_str())
        .collect();
    let mut sorted = ids.clone();
    sorted.sort_unstable();
    assert_eq!(ids, sorted, "the assignments are ordered by worker id");
    for assignment in &first.assignments {
        assert!(
            assignment.path.len() <= 2,
            "{:?} is deeper than max_depth - 1",
            assignment.path
        );
    }
    let mut reversed = items;
    reversed.reverse();
    assert_eq!(
        group_items_with(&reversed, &config, &runtime)
            .await
            .unwrap(),
        first,
        "the caller's item order decided a placement"
    );
}

/// `GroupingConfig::clusters` bounds the branch count of a k-means run.
#[tokio::test]
async fn kmeans_grouping_honours_the_cluster_count_bounds() {
    let items: Vec<SourceItem> = (0..8)
        .map(|index| {
            item(
                &format!("w{index}"),
                &format!("Capability {index}"),
                &format!("group{}", index % 2),
            )
        })
        .collect();
    let runtime = GroupingRuntime::new().with_embedding(Arc::new(HashEmbedding::new(32)));
    // `clusters` clamps into `[1, min(items, max_children)]`, so every case is an upper bound.
    let cases: [(Option<usize>, usize); 5] = [
        (Some(1), 1),
        (Some(2), 2),
        (Some(999), 8),
        (Some(0), 1),
        (None, 1),
    ];
    for (clusters, max_paths) in cases {
        let config = GroupingConfig::default()
            .with_strategy(GroupingStrategy::Kmeans)
            .with_clusters(clusters);
        let outcome = group_items_with(&items, &config, &runtime).await.unwrap();
        let distinct = distinct_paths(&outcome);
        assert!(
            !distinct.is_empty() && distinct.len() <= max_paths,
            "clusters {clusters:?} produced {} paths, want 1..={max_paths}: {distinct:?}",
            distinct.len()
        );
        assert_eq!(outcome.assignments.len(), items.len(), "{clusters:?}");
        assert_eq!(outcome.strategy, GroupingStrategy::Kmeans, "{clusters:?}");
        assert_eq!(
            outcome.provenance,
            GroupingProvenance::Embedding,
            "{clusters:?}"
        );
    }
    let single = GroupingConfig::default()
        .with_strategy(GroupingStrategy::Kmeans)
        .with_clusters(Some(1));
    let outcome = group_items_with(&items, &single, &runtime).await.unwrap();
    assert_eq!(
        distinct_paths(&outcome).len(),
        1,
        "one cluster is one branch"
    );
    let empty = group_items_with(&[], &single, &runtime).await.unwrap();
    assert_eq!(
        empty.assignments,
        Vec::new(),
        "an empty inventory groups to nothing"
    );
    assert_eq!(empty.provenance, GroupingProvenance::Embedding);
}

/// The model-guided strategy places the items the scripted reply names, and only those.
#[tokio::test]
async fn llm_grouping_places_the_items_a_scripted_reply_names() {
    let items = vec![
        item("a", "PDF Parser", "documents/pdf"),
        item("b", "Spreadsheet Reader", "documents/sheets"),
        item("c", "Chart Builder", "charts"),
    ];
    let config = GroupingConfig::default()
        .with_strategy(GroupingStrategy::LlmGrouping)
        .with_max_depth(3);
    let model = Arc::new(StaticCompletionModel::single(
        r#"{"groups":[{"path":["Documents","PDFs","Forms"],"worker_ids":["a","b"]},
            {"path":["Visuals","Charts"],"worker_ids":["c"]}]}"#,
    ));
    let runtime = GroupingRuntime::new().with_model(model.clone());
    let outcome = group_items_with(&items, &config, &runtime).await.unwrap();
    assert_eq!(model.call_count(), 1, "one prompt groups the inventory");
    assert_eq!(outcome.strategy, GroupingStrategy::LlmGrouping);
    assert_eq!(outcome.provenance, GroupingProvenance::Model);
    assert_eq!(
        grouped_paths(&outcome),
        vec![
            ("a".to_string(), path_of(&["documents", "pdfs"])),
            ("b".to_string(), path_of(&["documents", "pdfs"])),
            ("c".to_string(), path_of(&["visuals", "charts"])),
        ],
        "the reply's three segments are cut to max_depth - 1"
    );

    let partial = Arc::new(StaticCompletionModel::single(
        r#"{"groups":[{"path":["Documents"],"worker_ids":["a","b"]}]}"#,
    ));
    let outcome = group_items_with(&items, &config, &GroupingRuntime::new().with_model(partial))
        .await
        .unwrap();
    assert_eq!(outcome.provenance, GroupingProvenance::Model);
    assert_eq!(
        grouped_paths(&outcome),
        vec![
            ("a".to_string(), path_of(&["documents"])),
            ("b".to_string(), path_of(&["documents"])),
        ],
        "an item the reply leaves out gets no placement"
    );
    assert!(
        !outcome
            .assignments
            .iter()
            .any(|assignment| assignment.worker_id == "c"),
        "the reply left c unassigned"
    );
}

/// A failed call or unusable reply either degrades to the rule or surfaces, by config.
#[tokio::test]
async fn llm_grouping_falls_back_or_surfaces_an_unusable_reply() {
    let items = tools(3, "documents");
    let config = GroupingConfig::default().with_strategy(GroupingStrategy::LlmGrouping);
    let unusable = || Arc::new(StaticCompletionModel::single("no json here"));

    let fallback = GroupingRuntime::new().with_model(unusable());
    let outcome = group_items_with(&items, &config, &fallback).await.unwrap();
    assert_eq!(outcome.strategy, GroupingStrategy::Deterministic);
    assert_eq!(
        outcome.provenance,
        GroupingProvenance::DeterministicFallback
    );
    assert_eq!(
        outcome.assignments,
        group_items(&items, &GroupingConfig::default()).unwrap(),
        "the fallback is the deterministic rule"
    );

    let strict = GroupingRuntime::new()
        .with_model(unusable())
        .with_fallback(FallbackMode::Error);
    let error = group_items_with(&items, &config, &strict)
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Llm);
    assert!(error.message().contains("llm_grouping"), "{error}");
    assert!(error.message().contains("no usable JSON object"), "{error}");

    let failed_call = GroupingRuntime::new()
        .with_model(Arc::new(StaticCompletionModel::new(Vec::<String>::new())))
        .with_fallback(FallbackMode::Error);
    let error = group_items_with(&items, &config, &failed_call)
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Llm);
    assert!(error.message().contains("model call failed"), "{error}");

    let missing = GroupingRuntime::new().with_fallback(FallbackMode::Error);
    let error = group_items_with(&items, &config, &missing)
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Config);
    assert!(error.message().contains("with_model"), "{error}");
}

/// A build handed the runtime publishes the same three-artifact index as any other build.
#[tokio::test]
async fn build_with_runtime_publishes_a_normal_index() {
    let dir = tempfile::tempdir().unwrap();
    let items = vec![
        item("charts", "Chart Builder", "charts"),
        item("reader", "PDF Parser", "documents/pdf"),
        item("sheets", "Spreadsheet Reader", "documents/sheets"),
        item("web", "Web Search", "web/search"),
    ];
    let config = BuildConfig::new().with_grouping(
        GroupingConfig::default()
            .with_strategy(GroupingStrategy::Kmeans)
            .with_max_children(2)
            .with_clusters(Some(2)),
    );
    let runtime = GroupingRuntime::new().with_embedding(Arc::new(HashEmbedding::new(16)));
    let report =
        IndexBuilder::build_with_runtime(sources(items), dir.path(), config.clone(), &runtime)
            .await
            .unwrap();
    assert_eq!(report.grouping_provenance, GroupingProvenance::Embedding);
    assert_eq!(report.record_count, 4, "every item keeps a catalog record");
    assert_eq!(report.manifest.count, 4);
    assert!(
        report.node_count > report.record_count,
        "the tree holds branches above its leaves"
    );
    assert_eq!(
        published_names(dir.path()),
        artifact_names(),
        "the runtime build keeps the three-artifact contract"
    );

    let builder = IndexBuilder::open(dir.path(), config).await.unwrap();
    let mut ids: Vec<&str> = builder
        .records()
        .iter()
        .map(|record| record.worker_id.as_str())
        .collect();
    ids.sort_unstable();
    assert_eq!(ids, ["charts", "reader", "sheets", "web"]);
    for record in builder.records() {
        assert_eq!(
            record.branch_path.first().map(String::as_str),
            Some("Skills"),
            "{} sits below the root",
            record.worker_id
        );
    }
    assert!(builder.nodes().iter().any(IndexNode::is_leaf));
    assert_eq!(
        fs::read_to_string(dir.path().join(CATALOG_FILENAME))
            .unwrap()
            .lines()
            .count(),
        4
    );
}

/// The visualizer is one additive, escaped file; the default build keeps three artifacts.
#[tokio::test]
async fn with_html_publishes_the_visualizer_beside_the_three_artifacts() {
    let plain = tempfile::tempdir().unwrap();
    let report = IndexBuilder::build(
        sources(tools(2, "documents")),
        plain.path(),
        BuildConfig::new(),
    )
    .await
    .unwrap();
    assert!(!report.html_written, "the visualizer is opt-in");
    assert_eq!(
        published_names(plain.path()),
        artifact_names(),
        "the default contract is still the three artifacts"
    );

    let dir = tempfile::tempdir().unwrap();
    let items = vec![
        item("PdfFill", "PDF <Fill> & Reader", "documents/pdf"),
        item("Reader", "Doc Reader", "documents"),
    ];
    let report = IndexBuilder::build(
        sources(items),
        dir.path(),
        BuildConfig::new().with_html(true),
    )
    .await
    .unwrap();
    assert!(report.html_written);
    let mut expected = artifact_names();
    expected.push(TREE_HTML_FILENAME.to_string());
    expected.sort();
    assert_eq!(published_names(dir.path()), expected);
    for name in ARTIFACTS {
        assert!(dir.path().join(name).exists(), "{name} was not written");
    }

    let html = fs::read_to_string(dir.path().join(TREE_HTML_FILENAME)).unwrap();
    assert!(html.starts_with("<!DOCTYPE html>\n"), "{html:.80}");
    assert!(html.ends_with("</html>\n"), "the document is closed");
    assert!(
        html.contains("<title>Skills</title>"),
        "the root labels the document"
    );
    assert!(
        html.contains("<h2>Capability tree</h2>"),
        "the tree section is present"
    );
    assert!(
        html.contains("PDF &lt;Fill&gt; &amp; Reader"),
        "the item name is escaped"
    );
    assert!(
        !html.contains("<Fill>"),
        "no raw markup from an item name reaches the document"
    );
    assert!(html.contains("Doc Reader"), "the second item is listed");
}

// --------------------------------------------------------------------------------------
// 10. Atomic publication (the builder path of section 6)
// --------------------------------------------------------------------------------------

#[tokio::test]
async fn a_build_that_cannot_publish_leaves_no_artifact_behind() {
    let dir = tempfile::tempdir().unwrap();
    let config = BuildConfig::new().with_item_type("widget");

    let error = IndexBuilder::build(sources(tools(2, "documents")), dir.path(), config)
        .await
        .unwrap_err();

    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains("item_type"));
    assert_eq!(
        fs::read_dir(dir.path()).unwrap().count(),
        0,
        "a rejected build writes nothing"
    );
}

#[tokio::test]
async fn a_builder_whose_item_type_was_never_validated_cannot_publish() {
    let dir = tempfile::tempdir().unwrap();
    let builder = IndexBuilder::new(BuildConfig::default().with_item_type(""));

    let error = builder.write(dir.path(), MODE_FULL).await.unwrap_err();

    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(
        error.message().contains("item_type"),
        "the write names the field it refuses: {}",
        error.message()
    );
    assert_eq!(
        fs::read_dir(dir.path()).unwrap().count(),
        0,
        "an unloadable index is never published"
    );
    assert_eq!(
        builder.item_type(),
        "skill",
        "the lenient rule fills a blank item type"
    );
}
