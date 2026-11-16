//! The four operations: scope resolution, row construction and item lookup.
//!
//! Each operation returns one [`OperationOutcome`] — the typed rows plus the resolved
//! scope — and the toolkit layers paging, rendering and cursors on top. Row ordering is
//! fixed here: branches by `node_id`, items by `item_id` (the tree accessors sort), and
//! search hits keep the retriever's rank order. Nothing reads the clock or a hash map.
use crate::error::{Error, Result};
use crate::retrieval::search::service::RetrievalResult;
use crate::retrieval::types::{
    CatalogRecord, LoadedIndex, RetrieverItem, RetrieverNode, META_CONTENT, ROOT_NODE_ID,
};

/// The outcome of one operation before paging.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct OperationOutcome {
    /// Every row the operation matched, in render order.
    pub entries: Vec<RetrievalEntryRow>,
    /// The resolved scope cid (`ROOT` for the root, the record's leaf cid for a read).
    pub cid: String,
    /// The scope node's label (empty for search and read).
    pub label: String,
    /// The scope node's description (empty for search and read).
    pub description: String,
    /// The scope node's ancestor chain, root excluded.
    pub path: Vec<String>,
    /// The search backend, when a search produced the rows.
    pub method: String,
    /// The full body of a read.
    pub read: Option<ReadItem>,
}

/// One row of an [`OperationOutcome`]; the public response re-exports this shape as
/// [`super::RetrievalEntry`].
pub(crate) type RetrievalEntryRow = super::RetrievalEntry;

/// One item's full routing description, as `read` returns it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct ReadItem {
    /// The typed row describing the item.
    pub entry: RetrievalEntryRow,
    /// The routing body (description plus the catalog's compacted content).
    pub body: String,
    /// The item's source path, when the catalog recorded one.
    pub skill_path: String,
}

/// Item and descendant-branch counts of one node.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct NodeStats {
    /// How many items the subtree carries, the node's own items included.
    pub item_count: usize,
    /// How many descendant branches the node carries (its children and deeper).
    pub branch_count: usize,
}

/// `list`: the direct sub-branches of `cid` (the root when `cid` is `ROOT`).
///
/// # Errors
///
/// [`Error::NotFound`] when `cid` names no branch.
pub(crate) fn list(index: &LoadedIndex, cid: &str) -> Result<OperationOutcome> {
    let (node, path) = resolve_scope(index, cid)?;
    let entries: Vec<RetrievalEntryRow> = node
        .children()
        .into_iter()
        .map(|child| branch_entry(child, &path))
        .collect();
    Ok(OperationOutcome {
        entries,
        cid: resolved_cid(node),
        label: node.label.clone(),
        description: node.description.clone(),
        path,
        method: String::new(),
        read: None,
    })
}

/// `browse`: one node's direct sub-branches followed by its own items.
///
/// # Errors
///
/// [`Error::NotFound`] when `cid` names no branch.
pub(crate) fn browse(index: &LoadedIndex, cid: &str) -> Result<OperationOutcome> {
    let (node, path) = resolve_scope(index, cid)?;
    let mut entries: Vec<RetrievalEntryRow> = Vec::new();
    for child in node.children() {
        entries.push(branch_entry(child, &path));
    }
    for item in node.items() {
        entries.push(item_entry(index, item, &path));
    }
    Ok(OperationOutcome {
        entries,
        cid: resolved_cid(node),
        label: node.label.clone(),
        description: node.description.clone(),
        path,
        method: String::new(),
        read: None,
    })
}

/// `search`: the retriever's ranked candidates as search-hit rows.
pub(crate) fn search(result: &RetrievalResult) -> OperationOutcome {
    let entries = result
        .candidates
        .iter()
        .map(|candidate| {
            let worker_id = non_blank(&candidate.worker_id)
                .or_else(|| non_blank(&candidate.item_id))
                .unwrap_or_default()
                .to_string();
            RetrievalEntryRow {
                kind: "search_hit".to_string(),
                cid: candidate.cid.clone(),
                label: non_blank(&candidate.name).map_or_else(|| worker_id.clone(), str::to_string),
                worker_id,
                description: candidate.description.clone(),
                path: candidate.branch_path.clone(),
                category: candidate.category.clone(),
                rank: candidate.rank,
                ..RetrievalEntryRow::default()
            }
        })
        .collect();
    OperationOutcome {
        entries,
        method: result.method.clone(),
        ..OperationOutcome::default()
    }
}

/// `read`: one item's full routing description, resolved by leaf cid or worker id.
///
/// # Errors
///
/// [`Error::NotFound`] when neither the catalog nor the tree knows `cid`.
pub(crate) fn read(index: &LoadedIndex, cid: &str) -> Result<OperationOutcome> {
    let target = cid.trim();
    let (entry, body, skill_path) = if let Some(record) = find_record(index, target) {
        let entry = entry_from_record(index, record);
        let body = record_body(record);
        (entry, body, record.skill_path.clone())
    } else if let Some((item, path)) = find_item(&index.root, target) {
        let record = index
            .records
            .iter()
            .find(|record| record.cid == item.payload);
        let entry = record.map_or_else(
            || synthetic_entry(item, &path),
            |record| entry_from_record(index, record),
        );
        let body = record.map_or_else(|| item.description.trim().to_string(), record_body);
        let skill_path = record.map_or_else(String::new, |record| record.skill_path.clone());
        (entry, body, skill_path)
    } else if let Some(path) = index.item_paths.get(target) {
        let label = target.rsplit('.').next().unwrap_or(target).to_string();
        let entry = RetrievalEntryRow {
            kind: "item".to_string(),
            cid: target.to_string(),
            label: label.clone(),
            path: path.clone(),
            category: path.join("."),
            ..RetrievalEntryRow::default()
        };
        (entry, label, String::new())
    } else {
        return Err(Error::not_found(format!(
            "unknown item `{target}`: pass a leaf cid or worker id from operation=list, browse or \
             search"
        )));
    };
    let entry = RetrievalEntryRow {
        description: if entry.description.trim().is_empty() {
            body.clone()
        } else {
            entry.description.clone()
        },
        ..entry
    };
    let body = if body.trim().is_empty() {
        entry.label.clone()
    } else {
        body
    };
    Ok(OperationOutcome {
        cid: entry.cid.clone(),
        label: entry.label.clone(),
        description: entry.description.clone(),
        path: entry.path.clone(),
        entries: vec![entry.clone()],
        method: String::new(),
        read: Some(ReadItem {
            entry,
            body,
            skill_path,
        }),
    })
}

/// The item and descendant-branch counts of `node`.
pub(crate) fn node_stats(node: &RetrieverNode) -> NodeStats {
    NodeStats {
        item_count: node.item_count(),
        branch_count: node
            .children()
            .iter()
            .map(|child| 1 + node_stats(child).branch_count)
            .sum(),
    }
}

/// Resolves a branch cid to its node and ancestor chain.
fn resolve_scope<'a>(
    index: &'a LoadedIndex,
    cid: &str,
) -> Result<(&'a RetrieverNode, Vec<String>)> {
    let target = cid.trim();
    if target.is_empty() || target == ROOT_NODE_ID {
        return Ok((&index.root, Vec::new()));
    }
    if let Some(found) = find_node(&index.root, target) {
        return Ok(found);
    }
    if index.records.iter().any(|record| record.cid == target) {
        return Err(Error::not_found(format!(
            "unknown branch `{target}`: it names a leaf item; use operation=read"
        )));
    }
    Err(Error::not_found(format!(
        "unknown branch `{target}`: use operation=list or operation=browse with a branch cid"
    )))
}

/// Finds `target` depth-first in sorted child order, carrying its ancestor chain.
fn find_node<'a>(
    node: &'a RetrieverNode,
    target: &str,
) -> Option<(&'a RetrieverNode, Vec<String>)> {
    fn walk<'a>(
        node: &'a RetrieverNode,
        target: &str,
        path: &mut Vec<String>,
    ) -> Option<(&'a RetrieverNode, Vec<String>)> {
        if node.node_id == target {
            return Some((node, path.clone()));
        }
        for child in node.children() {
            path.push(child.node_id.clone());
            if let Some(found) = walk(child, target, path) {
                return Some(found);
            }
            path.pop();
        }
        None
    }
    walk(node, target, &mut Vec::new())
}

/// Finds an item by payload or item id, carrying its containing node's chain.
fn find_item<'a>(
    node: &'a RetrieverNode,
    target: &str,
) -> Option<(&'a RetrieverItem, Vec<String>)> {
    fn walk<'a>(
        node: &'a RetrieverNode,
        target: &str,
        path: &mut Vec<String>,
    ) -> Option<(&'a RetrieverItem, Vec<String>)> {
        for item in node.items() {
            if item.payload == target || item.item_id == target {
                return Some((item, path.clone()));
            }
        }
        for child in node.children() {
            path.push(child.node_id.clone());
            if let Some(found) = walk(child, target, path) {
                return Some(found);
            }
            path.pop();
        }
        None
    }
    walk(node, target, &mut Vec::new())
}

/// The resolved cid of a node; the synthetic root stays [`ROOT_NODE_ID`].
fn resolved_cid(node: &RetrieverNode) -> String {
    if node.node_id.trim().is_empty() {
        ROOT_NODE_ID.to_string()
    } else {
        node.node_id.clone()
    }
}

/// Builds one branch row beneath `parent_path`.
fn branch_entry(child: &RetrieverNode, parent_path: &[String]) -> RetrievalEntryRow {
    let mut path = parent_path.to_vec();
    path.push(child.node_id.clone());
    let stats = node_stats(child);
    RetrievalEntryRow {
        kind: "branch".to_string(),
        cid: child.node_id.clone(),
        label: non_blank(&child.label).map_or_else(|| child.node_id.clone(), str::to_string),
        worker_id: String::new(),
        description: child.description.clone(),
        path,
        category: String::new(),
        rank: 0,
        item_count: stats.item_count,
        branch_count: stats.branch_count,
    }
}

/// Builds one item row beneath `parent_path`.
fn item_entry(
    index: &LoadedIndex,
    item: &RetrieverItem,
    parent_path: &[String],
) -> RetrievalEntryRow {
    let record = index
        .records
        .iter()
        .find(|record| record.cid == item.payload);
    let worker_id = record
        .and_then(|record| non_blank(&record.worker_id))
        .or_else(|| non_blank(&item.item_id))
        .unwrap_or_default()
        .to_string();
    let label = record
        .and_then(|record| non_blank(&record.name))
        .or_else(|| non_blank(&item.label))
        .or_else(|| non_blank(&item.item_id))
        .map_or_else(|| item.payload.clone(), str::to_string);
    let description = record
        .and_then(|record| non_blank(&record.description))
        .or_else(|| non_blank(&item.description))
        .unwrap_or_default()
        .to_string();
    let path = index
        .item_paths
        .get(&item.payload)
        .cloned()
        .unwrap_or_else(|| parent_path.to_vec());
    let category = record
        .map(|record| record.category.clone())
        .filter(|category| !category.is_empty())
        .unwrap_or_else(|| path.join("."));
    RetrievalEntryRow {
        kind: "item".to_string(),
        cid: item.payload.clone(),
        worker_id,
        label,
        description,
        path,
        category,
        rank: 0,
        item_count: 0,
        branch_count: 0,
    }
}

/// Builds one item row from a catalog record.
fn entry_from_record(index: &LoadedIndex, record: &CatalogRecord) -> RetrievalEntryRow {
    let mut path = index
        .item_paths
        .get(&record.cid)
        .cloned()
        .unwrap_or_else(|| record.branch_path.clone());
    if path.is_empty() {
        path = derived_branch_path(&record.cid);
    }
    let category = if record.category.is_empty() {
        path.join(".")
    } else {
        record.category.clone()
    };
    let label = non_blank(&record.name)
        .or_else(|| non_blank(&record.worker_id))
        .unwrap_or(record.cid.as_str())
        .to_string();
    RetrievalEntryRow {
        kind: "item".to_string(),
        cid: record.cid.clone(),
        worker_id: record.worker_id.clone(),
        label,
        description: record.description.clone(),
        path,
        category,
        rank: 0,
        item_count: 0,
        branch_count: 0,
    }
}

/// Builds one item row from a tree item with no catalog record.
fn synthetic_entry(item: &RetrieverItem, path: &[String]) -> RetrievalEntryRow {
    let label = non_blank(&item.label)
        .or_else(|| non_blank(&item.item_id))
        .unwrap_or(item.payload.as_str())
        .to_string();
    RetrievalEntryRow {
        kind: "item".to_string(),
        cid: item.payload.clone(),
        worker_id: non_blank(&item.item_id).unwrap_or_default().to_string(),
        label,
        description: item.description.trim().to_string(),
        path: path.to_vec(),
        category: path.join("."),
        rank: 0,
        item_count: 0,
        branch_count: 0,
    }
}

/// The record's routing body: its description plus the catalog's compacted content.
fn record_body(record: &CatalogRecord) -> String {
    let mut body = record.description.trim().to_string();
    if let Some(content) = record.metadata.get(META_CONTENT) {
        let content = content.trim();
        if !content.is_empty() && content != body {
            if !body.is_empty() {
                body.push_str("\n\n");
            }
            body.push_str(content);
        }
    }
    body
}

/// Finds a catalog record by leaf cid, then by worker id.
fn find_record<'a>(index: &'a LoadedIndex, target: &str) -> Option<&'a CatalogRecord> {
    index
        .records
        .iter()
        .find(|record| record.cid == target)
        .or_else(|| {
            index
                .records
                .iter()
                .find(|record| record.worker_id == target)
        })
}

/// The ancestor cid chain derived from a leaf cid's segments.
fn derived_branch_path(cid: &str) -> Vec<String> {
    match cid.rfind('.') {
        Some(index) => cid[..index].split('.').map(str::to_string).collect(),
        None => Vec::new(),
    }
}

/// The trimmed value, when it is non-blank.
fn non_blank(value: &str) -> Option<&str> {
    let trimmed = value.trim();
    (!trimmed.is_empty()).then_some(trimmed)
}
