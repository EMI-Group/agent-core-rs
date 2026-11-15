//! Catalog-record derivation: one record per indexed leaf.
//!
//! Ports Python `build_catalog_records_from_nodes` from
//! `openjiuwen.symphony.retrieval.build.workflows.artifacts` (openJiuwen SDK v0.1.20), split
//! out of [`workflows`](crate::retrieval::build::workflows) so neither module outgrows its
//! file.
//!
//! A record joins a placed leaf node to the scanned
//! [`SourceItem`] that produced it: the node
//! supplies the position (`cid`, `branch_path`, `category`) and the routing profile, the
//! item supplies the identity text (`name`, `description`, `content`, `skill_path`), and the
//! two are flattened into the `retrieval_text` the search runtime scores. Everything is
//! pure: no clock, no model, no I/O.

use std::collections::BTreeMap;

use tracing::warn;

use crate::retrieval::build::scanner::SourceItem;
use crate::retrieval::types::{
    build_retrieval_text, CatalogRecord, IndexNode, META_CONTENT, META_DONT_SELECT_WHEN,
    META_SELECT_WHEN, META_SOURCE_DESCRIPTION,
};

/// Builds the catalog record of the leaf `node` placed for `item`.
///
/// Mirrors Python's `build_catalog_records_from_nodes` for one node: `name` is the item's
/// name, else the worker id; `description` is the node's description when the node carries
/// a tree profile (a `source_description`, `select_when` or `dont_select_when`) and the
/// item's description otherwise, each falling back to the other; `retrieval_text` is
/// [`build_retrieval_text`] over name, description, content, worker id and `cid`; and
/// `metadata` always carries the four [`META_CONTENT`], [`META_SOURCE_DESCRIPTION`],
/// [`META_SELECT_WHEN`] and [`META_DONT_SELECT_WHEN`] keys.
///
/// `branch_path` and `category` are derived from the node's `cid`, so a re-keyed record
/// only has to pass the new `cid`.
#[must_use]
pub fn catalog_record(node: &IndexNode, item: &SourceItem) -> CatalogRecord {
    let worker_id = node.worker_id.trim();
    let cid = node.cid.clone();
    let name = if item.name.trim().is_empty() {
        worker_id.to_string()
    } else {
        item.name.trim().to_string()
    };
    let node_description = node.description.trim();
    let item_description = item.description.trim();
    let has_tree_profile = !node.source_description.trim().is_empty()
        || !node.select_when.trim().is_empty()
        || !node.dont_select_when.trim().is_empty();
    let description = if has_tree_profile {
        if node_description.is_empty() {
            item_description
        } else {
            node_description
        }
    } else if item_description.is_empty() {
        node_description
    } else {
        item_description
    }
    .to_string();
    let content = item.content.trim();
    let source_description = if node.source_description.trim().is_empty() {
        item_description
    } else {
        node.source_description.trim()
    };
    let metadata: BTreeMap<String, String> = BTreeMap::from([
        (META_CONTENT.to_string(), content.to_string()),
        (
            META_SOURCE_DESCRIPTION.to_string(),
            source_description.to_string(),
        ),
        (
            META_SELECT_WHEN.to_string(),
            node.select_when.trim().to_string(),
        ),
        (
            META_DONT_SELECT_WHEN.to_string(),
            node.dont_select_when.trim().to_string(),
        ),
    ]);
    let retrieval_text = build_retrieval_text(&name, &description, content, worker_id, &cid);
    CatalogRecord::new(worker_id, cid)
        .with_name(name)
        .with_description(description)
        .with_skill_path(item.skill_path.clone())
        .with_retrieval_text(retrieval_text)
        .with_metadata(metadata)
}

/// Derives one catalog record per leaf node that has a matching scanned item.
pub(crate) fn records_for_nodes(nodes: &[IndexNode], items: &[SourceItem]) -> Vec<CatalogRecord> {
    let by_worker: BTreeMap<&str, &SourceItem> = items
        .iter()
        .map(|item| (item.worker_id.as_str(), item))
        .collect();
    let mut records: Vec<CatalogRecord> = Vec::new();
    for node in nodes {
        let worker_id = node.worker_id.trim();
        if !node.is_leaf() || worker_id.is_empty() {
            continue;
        }
        if let Some(item) = by_worker.get(worker_id) {
            records.push(catalog_record(node, item));
        } else {
            warn!(
                worker_id,
                cid = %node.cid,
                "no scanned item carries this leaf's worker id; the catalog drops it"
            );
        }
    }
    sort_records(&mut records);
    records
}

/// Re-keys `record` onto `cid`, keeping its identity text and recomputing the rest.
pub(crate) fn rekey_record(record: &CatalogRecord, cid: &str) -> CatalogRecord {
    let mut rekeyed = CatalogRecord::new(record.worker_id.clone(), cid)
        .with_name(record.name.clone())
        .with_description(record.description.clone())
        .with_skill_path(record.skill_path.clone())
        .with_metadata(record.metadata.clone());
    let retrieval_text = build_retrieval_text(
        &rekeyed.name,
        &rekeyed.description,
        rekeyed.metadata_of(META_CONTENT),
        &rekeyed.worker_id,
        &rekeyed.cid,
    );
    rekeyed.retrieval_text = retrieval_text;
    rekeyed
}

/// Sorts `records` by `cid` (then `worker_id`), the order the catalog renders in.
pub(crate) fn sort_records(records: &mut [CatalogRecord]) {
    records.sort_by(|left, right| {
        left.cid
            .cmp(&right.cid)
            .then_with(|| left.worker_id.cmp(&right.worker_id))
    });
}

/// A sorted copy of `records`.
pub(crate) fn sorted_records(records: &[CatalogRecord]) -> Vec<CatalogRecord> {
    let mut sorted = records.to_vec();
    sort_records(&mut sorted);
    sorted
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn item(worker_id: &str, name: &str) -> SourceItem {
        SourceItem {
            worker_id: worker_id.to_string(),
            name: name.to_string(),
            description: format!("{name} routes matching requests."),
            content: format!("{name} usage notes."),
            skill_path: format!("skills/{worker_id}/SKILL.md"),
            item_type: "skill".to_string(),
            ..SourceItem::default()
        }
    }

    #[test]
    fn catalog_record_derives_its_text_and_metadata() {
        let node = IndexNode::leaf("Skills.docs.pdffill", "PdfFill")
            .with_description("Tree description")
            .with_source_description("Fills PDF forms")
            .with_select_when("Fill a form")
            .with_dont_select_when("Not for scans");
        let record = catalog_record(&node, &item("PdfFill", "PDF Fill"));
        assert_eq!(record.worker_id, "PdfFill");
        assert_eq!(record.cid, "Skills.docs.pdffill");
        assert_eq!(record.name, "PDF Fill");
        assert_eq!(
            record.description, "Tree description",
            "the tree profile's description wins"
        );
        assert_eq!(record.metadata_of(META_CONTENT), "PDF Fill usage notes.");
        assert_eq!(
            record.metadata_of(META_SOURCE_DESCRIPTION),
            "Fills PDF forms"
        );
        assert_eq!(record.metadata_of(META_SELECT_WHEN), "Fill a form");
        assert_eq!(record.metadata_of(META_DONT_SELECT_WHEN), "Not for scans");
        assert_eq!(record.branch_path, vec!["Skills", "docs"]);
        assert_eq!(record.category, "Skills.docs");
        assert_eq!(
            record.retrieval_text,
            "PDF Fill\nTree description\nPDF Fill usage notes.\nPdfFill\nSkills.docs.pdffill"
        );

        let plain =
            IndexNode::leaf("Skills.docs.pdffill", "PdfFill").with_description("Node description");
        let mut nameless = item("PdfFill", "  ");
        nameless.description.clear();
        let record = catalog_record(&plain, &nameless);
        assert_eq!(record.name, "PdfFill", "a blank name falls back to the id");
        assert_eq!(record.description, "Node description");
        assert_eq!(record.skill_path, "skills/PdfFill/SKILL.md");
    }
}
