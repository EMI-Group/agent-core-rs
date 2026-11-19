//! The capability-tree index artifact set: `tree_index.yaml`, `catalog.jsonl`,
//! `manifest.json`.
//!
//! Ports the artifact IO of Python `openjiuwen.symphony.retrieval` — `build/io/tree.py`
//! (`write_tree_preset` / `sort_tree_nodes` / `parse_simple_nodes_yaml`),
//! `build/io/catalog.py`, `build/io/manifest.py`, and the tree assembly in
//! `search/artifacts/loading.py` (`build_tree_root`, `_compose_routing_description`).
//!
//! # Layout
//!
//! ```text
//! <index-dir>/
//!   tree_index.yaml   # nodes: [ {cid, type, description?, select_when?, ...} ]
//!   catalog.jsonl     # one compact JSON record per line
//!   manifest.json     # pretty JSON: mode, count, item_paths, worker_ids, item_type, content_hash
//! ```
//!
//! # The empty index
//!
//! A catalog with no records is a supported index, not a failure: a build over an empty
//! item source publishes its root branch with `count: 0`, and a render of zero nodes writes
//! the flow sequence `nodes: []`, which [`parse_tree_yaml`] reads back as an empty node
//! list. The manifest then records `count: 0` and [`catalog_content_hash`]'s digest of the
//! empty catalog, so `write_index` → `load_index` round-trips a zero-item index.
//!
//! # Determinism and atomicity
//!
//! The index is written in three steps — tree, catalog, manifest last — and every step
//! writes a same-directory `<name>.tmp` file followed by a `rename`, so a reader never
//! observes a half-written file. Writing the manifest last means a partially refreshed
//! index still carries the *previous* manifest, and
//! [`load_index`] then rejects the mismatch
//! instead of returning a catalog that does not belong to its tree.
//!
//! Every renderer sorts its input first: nodes by `(depth, cid)` and records by `cid`, so
//! the bytes on disk do not depend on the order a caller happened to build them in.
//!
//! # Deviations from the Python writer
//!
//! - Scalars are quoted with JSON string literals (`serde_json`), which are valid YAML
//!   double-quoted scalars, instead of `json.dumps(..., ensure_ascii=False)`; escapes and
//!   non-ASCII characters come out identically, and the coding is exact rather than
//!   hand escaped.
//! - `worker_id` is emitted whenever it is non-empty. Python trims it first and then
//!   checks, so a whitespace-only worker id is written here and dropped there; keeping the
//!   stored value verbatim is what makes render → parse round-trip exactly.
//! - `manifest.json` carries one extra field, `content_hash` (see
//!   [`IndexManifest`]); the Python build does not
//!   write it, and an empty value means "not recorded", so both shapes load.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::io::ErrorKind as IoErrorKind;
use std::path::Path;

use serde_json::Value;
use sha2::{Digest, Sha256};
use tokio::fs;
use tracing::{debug, warn};

use crate::error::{validation, Error, Result};
use crate::retrieval::build::tree_text::sort_nodes;
use crate::retrieval::types::{
    CatalogRecord, IndexManifest, IndexNode, LoadedIndex, RetrieverItem, RetrieverNode,
    CATALOG_FILENAME, DEFAULT_ITEM_TYPE, MANIFEST_FILENAME, ROOT_NODE_ID, TREE_INDEX_FILENAME,
};

/// The `tree_index.yaml` document of a zero-node index.
///
/// A bare `nodes:` header would be a null `nodes` value, which
/// [`parse_tree_yaml`] rejects, so the empty document spells the
/// sequence out instead.
const EMPTY_TREE_YAML: &str = "nodes: []\n";

/// The maximum number of `.`-separated segments a capability `cid` may have.
///
/// A `cid` is a dotted path, and the tree it describes is walked and dropped recursively, so
/// an unbounded path is an unbounded stack: a hand-edited index (or a payload from an
/// untrusted catalog) with tens of thousands of segments would abort the process with a stack
/// overflow. The load path therefore rejects a deeper `cid` with a typed
/// [`Error::Validation`] instead of trusting it. Sixty-four
/// levels is far beyond any real capability taxonomy (the build schema's own limit is six).
pub const MAX_CID_DEPTH: usize = 64;

/// The number of non-empty `.`-separated segments in `cid`.
#[must_use]
pub fn cid_depth(cid: &str) -> usize {
    cid.split('.')
        .filter(|segment| !segment.trim().is_empty())
        .count()
}

/// Rejects a `cid` deeper than [`MAX_CID_DEPTH`], naming `file` and the segment count.
///
/// The offending path is not echoed: a 10 000-segment id would make the error message itself
/// megabytes long.
fn validate_cid_depth(file: &str, cid: &str, context: impl fmt::Display) -> Result<()> {
    let depth = cid_depth(cid);
    if depth > MAX_CID_DEPTH {
        return Err(validation(
            file,
            format!("{context} has {depth} dot-separated segments; the maximum is {MAX_CID_DEPTH}"),
        ));
    }
    Ok(())
}

/// Writes the three index artifacts into `dir`, creating it when missing.
///
/// The inputs are validated before anything touches the filesystem: a manifest whose
/// `count` does not describe `records`, a blank manifest `mode` or `item_type`, a duplicate
/// node `cid`, a duplicate record `cid`, or a record whose `cid` names no leaf node is an
/// [`Error::Validation`] naming the offending field, so a
/// failed write leaves the previous index untouched.
///
/// An empty node list with an empty record list and a `count` of `0` is a valid index: it
/// renders `nodes: []` and publishes [`catalog_content_hash`]'s digest of the empty catalog.
///
/// Files are written tree first, catalog second and manifest last, each through a
/// same-directory `<name>.tmp` file followed by a `rename`.
///
/// # Errors
///
/// [`Error::Validation`] for an inconsistent manifest or a malformed node or record
/// set, [`Error::Config`] for an unusable target path, and
/// [`Error::External`] for an I/O failure.
pub async fn write_index(
    dir: impl AsRef<Path>,
    nodes: &[IndexNode],
    records: &[CatalogRecord],
    manifest: &IndexManifest,
) -> Result<()> {
    let dir = dir.as_ref();
    validate_manifest(manifest, records)?;
    validate_nodes(nodes)?;
    validate_records(records, nodes)?;
    let tree = render_tree_yaml(nodes);
    let catalog = render_catalog(records);
    let manifest = render_manifest(manifest);
    fs::create_dir_all(dir)
        .await
        .map_err(|error| path_error("creating", dir, error))?;
    write_text_atomic(&dir.join(TREE_INDEX_FILENAME), &tree).await?;
    write_text_atomic(&dir.join(CATALOG_FILENAME), &catalog).await?;
    write_text_atomic(&dir.join(MANIFEST_FILENAME), &manifest).await?;
    debug!(
        directory = %dir.display(),
        nodes = nodes.len(),
        records = records.len(),
        "wrote a capability tree index"
    );
    Ok(())
}

/// Reads and validates one index directory into a [`LoadedIndex`].
///
/// The tree is parsed first, then the catalog, then the manifest. The three are then
/// cross-checked, and the first check that fails is the error:
///
/// - the manifest `count` must equal the number of parsed records;
/// - a non-empty manifest `content_hash` must match the catalog on disk (an empty one
///   means "not recorded" — Python's own builds write no hash — and is not checked);
/// - the manifest `item_type` must not be blank;
/// - every record `cid` must be a `leaf` node of the parsed tree, which is what keeps a
///   catalog from one build out of another build's tree.
///
/// It does *not* check the tree's shape ([`TreeSchema`](crate::retrieval::build::schema::TreeSchema)
/// is a build-time guard, not a load guard), a record's `worker_id` against its leaf's, or
/// the record text (`retrieval_text`, `category`, `branch_path`) against the tree: a catalog
/// this crate wrote is trusted for those.
///
/// Records come back sorted by `cid`, and `item_paths` is derived from the parsed tree —
/// only nodes reachable from the synthetic root contribute.
///
/// # Errors
///
/// [`Error::NotFound`] naming the missing file,
/// [`Error::Validation`] for an unusable document or an
/// inconsistent manifest or catalog, and [`Error::External`] for an
/// I/O failure.
pub async fn load_index(dir: impl AsRef<Path>) -> Result<LoadedIndex> {
    let dir = dir.as_ref();
    let tree_text = read_index_file(dir, TREE_INDEX_FILENAME).await?;
    let catalog_text = read_index_file(dir, CATALOG_FILENAME).await?;
    let manifest_text = read_index_file(dir, MANIFEST_FILENAME).await?;
    let nodes = parse_tree_yaml(&tree_text)?;
    let records = parse_catalog(&catalog_text)?;
    let manifest = parse_manifest(&manifest_text)?;
    if manifest.count != records.len() {
        return Err(validation(
            MANIFEST_FILENAME,
            format!(
                "count {} does not match {} catalog records",
                manifest.count,
                records.len()
            ),
        ));
    }
    if !manifest.content_hash.is_empty() {
        let actual = catalog_content_hash(&records);
        if actual != manifest.content_hash {
            return Err(validation(
                MANIFEST_FILENAME,
                format!(
                    "content_hash {} does not match the catalog's {actual}",
                    manifest.content_hash
                ),
            ));
        }
    }
    if manifest.item_type.trim().is_empty() {
        return Err(validation(MANIFEST_FILENAME, "item_type must not be blank"));
    }
    validate_record_leaves(&nodes, &records)?;
    let root = build_tree_root(&nodes, &records);
    let choices = root.flatten_choices();
    let mut item_paths = BTreeMap::new();
    collect_item_paths(&root, &mut Vec::new(), &mut item_paths);
    debug!(
        directory = %dir.display(),
        nodes = nodes.len(),
        records = records.len(),
        "loaded a capability tree index"
    );
    Ok(LoadedIndex {
        root,
        choices,
        records,
        manifest,
        item_paths,
    })
}

/// Renders the `tree_index.yaml` document for `nodes`.
///
/// Mirrors Python `write_tree_preset`: a `nodes:` header followed by one entry per node
/// sorted by `(depth, cid)`. Each entry emits `cid` and `type` always, then
/// `description`, `select_when`, `dont_select_when`, `source_description` and `worker_id`
/// when non-empty, in that order. Scalars are JSON string literals, which YAML reads back
/// as the same string; the document ends with one newline.
///
/// An empty node list renders as the flow sequence `nodes: []`, so a zero-node index is a
/// document [`parse_tree_yaml`] reads back
/// as an empty node list instead of a null `nodes` value.
///
/// # Panics
///
/// Never: a scalar renders through [`serde_json::Value`]'s infallible `Display`.
#[must_use]
pub fn render_tree_yaml(nodes: &[IndexNode]) -> String {
    let mut sorted: Vec<&IndexNode> = nodes.iter().collect();
    sorted.sort_by(|left, right| {
        (left.depth(), left.cid.as_str()).cmp(&(right.depth(), right.cid.as_str()))
    });
    if sorted.is_empty() {
        return EMPTY_TREE_YAML.to_string();
    }
    let mut lines: Vec<String> = Vec::with_capacity(1 + sorted.len() * 4);
    lines.push("nodes:".to_string());
    for node in sorted {
        lines.push(format!("  - cid: {}", quote_scalar(&node.cid)));
        lines.push(format!("    type: {}", quote_scalar(&node.node_type)));
        for (key, value) in [
            ("description", &node.description),
            ("select_when", &node.select_when),
            ("dont_select_when", &node.dont_select_when),
            ("source_description", &node.source_description),
            ("worker_id", &node.worker_id),
        ] {
            if !value.is_empty() {
                lines.push(format!("    {key}: {}", quote_scalar(value)));
            }
        }
    }
    let mut document = lines.join("\n");
    document.push('\n');
    document
}

/// Parses a `tree_index.yaml` document into nodes sorted by `(depth, cid)`.
///
/// The document is read as YAML, converted to [`serde_json::Value`], and every known key
/// is read tolerantly: an unknown key is ignored, a missing key becomes an empty string,
/// and a scalar of any JSON type is stringified. Both `type` and the legacy `node_type`
/// spell the node type, with `type` winning when it is non-empty. A non-mapping entry is
/// skipped with a warning rather than failing the whole tree.
///
/// A `nodes` value of `[]` — what [`render_tree_yaml`]
/// writes for a zero-node index — and a bare, valueless `nodes:` header both read as an
/// empty node list.
///
/// # Errors
///
/// [`Error::Validation`] naming `tree_index.yaml` when
/// the YAML is unusable, the document has no `nodes` key, or `nodes` is neither a sequence
/// nor empty.
pub fn parse_tree_yaml(text: &str) -> Result<Vec<IndexNode>> {
    let document: serde_yaml_ng::Value = serde_yaml_ng::from_str(text)
        .map_err(|error| validation(TREE_INDEX_FILENAME, format!("invalid YAML: {error}")))?;
    let document = serde_json::to_value(document).map_err(|error| {
        validation(
            TREE_INDEX_FILENAME,
            format!("unusable YAML document: {error}"),
        )
    })?;
    let Some(raw_nodes) = document.get("nodes") else {
        return Err(validation(TREE_INDEX_FILENAME, "missing `nodes` sequence"));
    };
    if raw_nodes.is_null() {
        return Ok(Vec::new());
    }
    let Some(entries) = raw_nodes.as_array() else {
        return Err(validation(TREE_INDEX_FILENAME, "`nodes` is not a sequence"));
    };
    let mut nodes = Vec::with_capacity(entries.len());
    for (index, entry) in entries.iter().enumerate() {
        let Some(fields) = entry.as_object() else {
            warn!(entry = index, "skipping a tree node that is not a mapping");
            continue;
        };
        let node_type = field_string(fields, "type");
        let node_type = if node_type.trim().is_empty() {
            field_string(fields, "node_type")
        } else {
            node_type
        };
        let cid = field_string(fields, "cid").trim().to_string();
        validate_cid_depth(
            TREE_INDEX_FILENAME,
            &cid,
            format!("the tree node at entry {index}"),
        )?;
        nodes.push(IndexNode {
            cid,
            node_type: node_type.trim().to_string(),
            description: field_string(fields, "description"),
            select_when: field_string(fields, "select_when"),
            dont_select_when: field_string(fields, "dont_select_when"),
            source_description: field_string(fields, "source_description"),
            worker_id: field_string(fields, "worker_id").trim().to_string(),
        });
    }
    sort_nodes(&mut nodes);
    Ok(nodes)
}

/// Renders the `catalog.jsonl` document: one compact JSON object per record and line.
///
/// Records are sorted by `cid` (then by `worker_id`), and every line is terminated with a
/// newline; an empty slice renders as the empty string.
///
/// # Panics
///
/// Never: a record that somehow fails to serialize is warned about and skipped, so one bad
/// record cannot lose the whole catalog.
#[must_use]
pub fn render_catalog(records: &[CatalogRecord]) -> String {
    let mut sorted: Vec<&CatalogRecord> = records.iter().collect();
    sorted.sort_by(|left, right| {
        left.cid
            .cmp(&right.cid)
            .then_with(|| left.worker_id.cmp(&right.worker_id))
    });
    let mut document = String::new();
    for record in sorted {
        match serde_json::to_string(record) {
            Ok(line) => {
                document.push_str(&line);
                document.push('\n');
            }
            Err(error) => warn!(
                cid = %record.cid,
                error = %error,
                "skipping a catalog record that does not serialize"
            ),
        }
    }
    document
}

/// Parses a `catalog.jsonl` document into records sorted by `cid`.
///
/// Blank lines are skipped. Every other line must be a JSON object; unknown keys are
/// ignored, the legacy `skill_id` key stands in for `worker_id`, and a `metadata` value of
/// any JSON type is stringified (a nested value as compact JSON). `branch_path` and
/// `category` are taken from the document when present and derived from the `cid`
/// otherwise.
///
/// # Errors
///
/// [`Error::Validation`] naming `catalog.jsonl` and the
/// one-based line number of the first malformed line.
pub fn parse_catalog(text: &str) -> Result<Vec<CatalogRecord>> {
    let mut records = Vec::new();
    for (index, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let number = index + 1;
        let value: Value = serde_json::from_str(line)
            .map_err(|error| validation(CATALOG_FILENAME, format!("line {number}: {error}")))?;
        let Some(fields) = value.as_object() else {
            return Err(validation(
                CATALOG_FILENAME,
                format!("line {number}: expected a JSON object"),
            ));
        };
        let worker_id = field_string(fields, "worker_id");
        let worker_id = if worker_id.is_empty() {
            field_string(fields, "skill_id")
        } else {
            worker_id
        };
        let mut record = CatalogRecord::new(worker_id, field_string(fields, "cid"));
        validate_cid_depth(
            CATALOG_FILENAME,
            &record.cid,
            format!("line {number}'s cid"),
        )?;
        let branch_path = string_list(fields.get("branch_path"));
        if !branch_path.is_empty() {
            record.category = branch_path.join(".");
            record.branch_path = branch_path;
        }
        let category = field_string(fields, "category");
        if !category.is_empty() {
            record.category = category;
        }
        record.name = field_string(fields, "name");
        record.description = field_string(fields, "description");
        record.skill_path = field_string(fields, "skill_path");
        record.retrieval_text = field_string(fields, "retrieval_text");
        record.metadata = metadata_map(fields.get("metadata"));
        records.push(record);
    }
    records.sort_by(|left, right| {
        left.cid
            .cmp(&right.cid)
            .then_with(|| left.worker_id.cmp(&right.worker_id))
    });
    Ok(records)
}

/// Renders the `manifest.json` document as pretty JSON in the struct's field order.
///
/// # Panics
///
/// Never: a manifest that fails to serialize (impossible for this shape) is warned about
/// and renders as `{}`, which [`load_index`]
/// rejects rather than mistaking for a valid manifest.
#[must_use]
pub fn render_manifest(manifest: &IndexManifest) -> String {
    serde_json::to_string_pretty(manifest).unwrap_or_else(|error| {
        warn!(error = %error, "a manifest failed to serialize");
        "{}".to_string()
    })
}

/// Parses a `manifest.json` document.
///
/// Every field is read tolerantly (a missing key keeps its default, and a scalar of any
/// JSON type is stringified). A document without an `item_type` key defaults to
/// [`DEFAULT_ITEM_TYPE`]; an explicitly blank one is preserved so
/// [`load_index`] can reject it.
///
/// # Errors
///
/// [`Error::Validation`] naming `manifest.json` when the
/// document is not a JSON object or `count` is not a representable integer.
pub fn parse_manifest(text: &str) -> Result<IndexManifest> {
    let value: Value = serde_json::from_str(text)
        .map_err(|error| validation(MANIFEST_FILENAME, format!("invalid JSON: {error}")))?;
    let Some(fields) = value.as_object() else {
        return Err(validation(MANIFEST_FILENAME, "expected a JSON object"));
    };
    let count = match fields.get("count") {
        Some(Value::Number(number)) => match number.as_u64() {
            Some(count) => usize::try_from(count).map_err(|error| {
                validation(MANIFEST_FILENAME, format!("count is too large: {error}"))
            })?,
            None => 0,
        },
        _ => 0,
    };
    let item_type = match fields.get("item_type") {
        None | Some(Value::Null) => DEFAULT_ITEM_TYPE.to_string(),
        Some(value) => scalar_string(value),
    };
    Ok(IndexManifest {
        mode: field_string(fields, "mode"),
        count,
        item_paths: string_list(fields.get("item_paths")),
        worker_ids: string_list(fields.get("worker_ids")),
        item_type,
        content_hash: field_string(fields, "content_hash"),
    })
}

/// The SHA-256 hex digest of [`render_catalog`]'s output for `records`.
///
/// This is what [`IndexManifest::content_hash`]
/// records, so a reader can tell a catalog that was rewritten independently of its
/// manifest.
#[must_use]
pub fn catalog_content_hash(records: &[CatalogRecord]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(render_catalog(records).as_bytes());
    format!("{:x}", hasher.finalize())
}

/// Builds the in-memory tree a retrieval run walks.
///
/// Mirrors Python `_build_tree_from_nodes`: the synthetic root is labelled
/// [`ROOT_NODE_ID`]; a leaf node becomes one [`RetrieverItem`] on its parent branch, with
/// `item_id` (and `label`) taken from the matching catalog record's name when non-blank
/// and from the `cid`'s last segment otherwise, `payload` set to the leaf's `cid`, and a
/// description composed from the node (falling back to the record) plus its routing hints.
/// Every other node type is a branch labelled with the `cid`'s last segment.
///
/// A branch with no children and no items is pruned unless it is the root, and a node
/// whose parent `cid` is not itself a branch is re-attached to the root with a warning.
/// Children are sorted by node id and items by item id, so the tree does not depend on
/// the node list's order.
#[must_use]
pub fn build_tree_root(nodes: &[IndexNode], records: &[CatalogRecord]) -> RetrieverNode {
    let mut records_by_payload: BTreeMap<&str, &CatalogRecord> = BTreeMap::new();
    for record in records {
        records_by_payload.insert(record.cid.as_str(), record);
    }
    let mut branch_ids: BTreeSet<&str> = BTreeSet::new();
    for node in nodes {
        let cid = node.cid.trim();
        if !cid.is_empty() && !node.is_leaf() {
            branch_ids.insert(cid);
        }
    }
    let mut assembly = TreeAssembly::default();
    for node in nodes {
        let cid = node.cid.trim();
        if cid.is_empty() {
            warn!("skipping a tree node with a blank cid");
            continue;
        }
        // [`parse_tree_yaml`] rejects a deeper cid, but this entry point is public: skipping
        // an over-deep node is what keeps the recursive assembly (and every later walk) from
        // overflowing the stack.
        if cid_depth(cid) > MAX_CID_DEPTH {
            warn!(
                depth = cid_depth(cid),
                "skipping a tree node deeper than MAX_CID_DEPTH"
            );
            continue;
        }
        let parent = resolve_parent(cid, &branch_ids);
        if node.is_leaf() {
            let record = records_by_payload.get(cid).copied();
            assembly
                .items
                .entry(parent)
                .or_default()
                .push(build_item(cid, node, record));
        } else {
            assembly.specs.insert(cid.to_string(), node);
            assembly
                .children
                .entry(parent)
                .or_default()
                .insert(cid.to_string());
        }
    }
    assembly
        .build("", true)
        .unwrap_or_else(|| RetrieverNode::new(ROOT_NODE_ID, ROOT_NODE_ID))
}

/// Composes a node's routing description from its parts.
///
/// Mirrors Python `_compose_routing_description`: the trimmed description comes first,
/// then `Select when: <…>` unless the description already contains that heading and the
/// hint is non-empty, then `Don't select when: <…>` on the same rule. Parts are joined
/// with `"\n"`.
#[must_use]
pub fn compose_routing_description(
    description: &str,
    select_when: &str,
    dont_select_when: &str,
) -> String {
    let description = description.trim();
    let mut parts: Vec<String> = Vec::new();
    if !description.is_empty() {
        parts.push(description.to_string());
    }
    if !select_when.is_empty() && !description.contains("Select when:") {
        parts.push(format!("Select when: {select_when}"));
    }
    if !dont_select_when.is_empty() && !description.contains("Don't select when:") {
        parts.push(format!("Don't select when: {dont_select_when}"));
    }
    parts.join("\n").trim().to_string()
}

/// The grouping state of one [`build_tree_root`] run.
#[derive(Default)]
struct TreeAssembly<'a> {
    /// Branch nodes by `cid`.
    specs: BTreeMap<String, &'a IndexNode>,
    /// Child branch `cid`s by parent `cid` (`""` is the synthetic root).
    children: BTreeMap<String, BTreeSet<String>>,
    /// Items by parent branch `cid`.
    items: BTreeMap<String, Vec<RetrieverItem>>,
}

impl TreeAssembly<'_> {
    /// Builds the branch at `cid`, dropping an empty non-root branch.
    fn build(&mut self, cid: &str, root: bool) -> Option<RetrieverNode> {
        let child_ids: Vec<String> = self
            .children
            .get(cid)
            .map(|ids| ids.iter().cloned().collect())
            .unwrap_or_default();
        let mut children = Vec::with_capacity(child_ids.len());
        for child in &child_ids {
            if let Some(node) = self.build(child, false) {
                children.push(node);
            }
        }
        let mut items = self.items.remove(cid).unwrap_or_default();
        items.sort_by(|left, right| {
            left.item_id
                .cmp(&right.item_id)
                .then_with(|| left.payload.cmp(&right.payload))
        });
        if !root && children.is_empty() && items.is_empty() {
            return None;
        }
        let (node_id, label, description) = if root {
            (
                ROOT_NODE_ID.to_string(),
                ROOT_NODE_ID.to_string(),
                String::new(),
            )
        } else {
            let description = self.specs.get(cid).map_or_else(String::new, |node| {
                compose_routing_description(
                    &node.description,
                    &node.select_when,
                    &node.dont_select_when,
                )
            });
            let label = last_segment(cid).to_string();
            (cid.to_string(), label, description)
        };
        Some(RetrieverNode {
            node_id,
            label,
            description,
            children,
            items,
        })
    }
}

/// Builds one leaf's [`RetrieverItem`].
fn build_item(cid: &str, node: &IndexNode, record: Option<&CatalogRecord>) -> RetrieverItem {
    let segment = last_segment(cid);
    let name = record.map_or("", |record| record.name.trim());
    let item_id = if name.is_empty() {
        segment.to_string()
    } else {
        name.to_string()
    };
    let label = if name.is_empty() {
        item_id.clone()
    } else {
        name.to_string()
    };
    let description = if node.description.trim().is_empty() {
        record.map_or("", |record| record.description.as_str())
    } else {
        node.description.as_str()
    };
    RetrieverItem {
        item_id,
        payload: cid.to_string(),
        label,
        description: compose_routing_description(
            description,
            &node.select_when,
            &node.dont_select_when,
        ),
    }
}

/// The last `.`-separated segment of `cid`.
fn last_segment(cid: &str) -> &str {
    cid.rsplit('.').next().unwrap_or(cid)
}

/// The branch a node belongs to, re-attaching an orphan to the root with a warning.
fn resolve_parent(cid: &str, branch_ids: &BTreeSet<&str>) -> String {
    let parent = cid.rfind('.').map_or("", |index| &cid[..index]);
    if parent.is_empty() || branch_ids.contains(parent) {
        return parent.to_string();
    }
    warn!(
        cid,
        parent, "a tree node's parent is not a branch; attaching it to the synthetic root"
    );
    String::new()
}

/// Collects each item payload's ancestor `cid` path, the synthetic root excluded.
fn collect_item_paths(
    node: &RetrieverNode,
    ancestors: &mut Vec<String>,
    paths: &mut BTreeMap<String, Vec<String>>,
) {
    let is_root = node.node_id == ROOT_NODE_ID;
    if !is_root {
        ancestors.push(node.node_id.clone());
    }
    for item in node.items() {
        paths.insert(item.payload.clone(), ancestors.clone());
    }
    for child in node.children() {
        collect_item_paths(child, ancestors, paths);
    }
    if !is_root {
        ancestors.pop();
    }
}

/// Validates the manifest of a write: it must describe exactly `records`.
fn validate_manifest(manifest: &IndexManifest, records: &[CatalogRecord]) -> Result<()> {
    if manifest.count != records.len() {
        return Err(validation(
            MANIFEST_FILENAME,
            format!(
                "count {} does not match {} catalog records",
                manifest.count,
                records.len()
            ),
        ));
    }
    if manifest.mode.trim().is_empty() {
        return Err(validation(MANIFEST_FILENAME, "mode must not be blank"));
    }
    if manifest.item_type.trim().is_empty() {
        return Err(validation(MANIFEST_FILENAME, "item_type must not be blank"));
    }
    Ok(())
}

/// Validates the node list of a write: no blank and no duplicate `cid`.
fn validate_nodes(nodes: &[IndexNode]) -> Result<()> {
    let mut seen: BTreeSet<&str> = BTreeSet::new();
    for node in nodes {
        let cid = node.cid.as_str();
        if cid.trim().is_empty() {
            return Err(validation(
                TREE_INDEX_FILENAME,
                "node cid must not be blank",
            ));
        }
        if !seen.insert(cid) {
            return Err(validation(
                TREE_INDEX_FILENAME,
                format!("duplicate cid `{cid}`"),
            ));
        }
        validate_cid_depth(TREE_INDEX_FILENAME, cid, "a node cid")?;
    }
    Ok(())
}

/// Validates the catalog of a write: unique `cid`s, each naming a leaf node.
fn validate_records(records: &[CatalogRecord], nodes: &[IndexNode]) -> Result<()> {
    let mut seen: BTreeSet<&str> = BTreeSet::new();
    for record in records {
        let cid = record.cid.as_str();
        if !seen.insert(cid) {
            return Err(validation(
                CATALOG_FILENAME,
                format!("duplicate cid `{cid}`"),
            ));
        }
    }
    validate_record_leaves(nodes, records)
}

/// Validates that every record `cid` names a `leaf` node of `nodes`.
///
/// Both the writer and the loader call this, so a catalog can never name a `cid` that its
/// own tree does not terminate — the check that keeps a catalog from one build out of
/// another build's tree.
fn validate_record_leaves(nodes: &[IndexNode], records: &[CatalogRecord]) -> Result<()> {
    let leaves: BTreeSet<&str> = nodes
        .iter()
        .filter(|node| node.is_leaf())
        .map(|node| node.cid.as_str())
        .collect();
    for record in records {
        let cid = record.cid.as_str();
        if !leaves.contains(cid) {
            return Err(validation(
                CATALOG_FILENAME,
                format!("cid `{cid}` has no leaf node in {TREE_INDEX_FILENAME}"),
            ));
        }
    }
    Ok(())
}

/// Writes `text` to `path` atomically: `<path>.tmp` in the same directory, then `rename`.
async fn write_text_atomic(path: &Path, text: &str) -> Result<()> {
    let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
        return Err(Error::config(format!(
            "index artifact path `{}` has no file name",
            path.display()
        )));
    };
    let temporary = path.with_file_name(format!("{file_name}.tmp"));
    fs::write(&temporary, text.as_bytes())
        .await
        .map_err(|error| path_error("writing", &temporary, error))?;
    if let Err(error) = fs::rename(&temporary, path).await {
        if let Err(cleanup) = fs::remove_file(&temporary).await {
            debug!(
                path = %temporary.display(),
                error = %cleanup,
                "failed to remove a temporary index artifact"
            );
        }
        return Err(path_error("committing", path, error));
    }
    Ok(())
}

/// Reads one index file, mapping a missing file to [`Error::NotFound`] naming it.
async fn read_index_file(dir: &Path, file_name: &str) -> Result<String> {
    let path = dir.join(file_name);
    match fs::read_to_string(&path).await {
        Ok(text) => Ok(text),
        Err(error) if error.kind() == IoErrorKind::NotFound => Err(Error::not_found(file_name)),
        Err(error) => Err(path_error("reading", &path, error)),
    }
}

/// Wraps an I/O failure as [`Error::External`], naming the action and the path.
fn path_error(action: &str, path: &Path, error: std::io::Error) -> Error {
    Error::external(error).with_context(format!("{action} `{}`", path.display()))
}

/// Renders `value` as a JSON string literal, which YAML reads back as the same string.
fn quote_scalar(value: &str) -> String {
    Value::String(value.to_string()).to_string()
}

/// Reads `key` of `fields` as a string, stringifying any scalar and defaulting to `""`.
fn field_string(fields: &serde_json::Map<String, Value>, key: &str) -> String {
    fields.get(key).map_or(String::new(), scalar_string)
}

/// Stringifies a JSON scalar; `null` and containers become `""`.
fn scalar_string(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        Value::Number(number) => number.to_string(),
        Value::Bool(flag) => flag.to_string(),
        _ => String::new(),
    }
}

/// Reads a JSON array (or a single string) of scalars as strings.
fn string_list(value: Option<&Value>) -> Vec<String> {
    match value {
        Some(Value::Array(entries)) => entries.iter().map(scalar_string).collect(),
        Some(Value::String(text)) => vec![text.clone()],
        _ => Vec::new(),
    }
}

/// Reads a JSON object of metadata values, stringifying each value.
fn metadata_map(value: Option<&Value>) -> BTreeMap<String, String> {
    let Some(Value::Object(entries)) = value else {
        return BTreeMap::new();
    };
    entries
        .iter()
        .map(|(key, value)| (key.clone(), metadata_string(value)))
        .collect()
}

/// Stringifies one metadata value; a nested value becomes compact JSON.
fn metadata_string(value: &Value) -> String {
    match value {
        Value::Null => String::new(),
        Value::String(text) => text.clone(),
        other => other.to_string(),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::retrieval::build::tree_text::cid_depth;
    use crate::retrieval::types::{
        build_retrieval_text, META_SELECT_WHEN, MODE_FULL, NODE_TYPE_LEAF,
    };

    fn sample_nodes() -> Vec<IndexNode> {
        vec![
            IndexNode::leaf("Skills.beta", "beta").with_source_description("from beta"),
            IndexNode::branch("Skills").with_description("All skills"),
            IndexNode::leaf("Skills.alpha", "alpha"),
            IndexNode::branch("Skills.empty"),
        ]
    }

    fn sample_records() -> Vec<CatalogRecord> {
        vec![
            CatalogRecord::new("beta", "Skills.beta")
                .with_name("Beta")
                .with_description("beta skill")
                .with_skill_path("/tmp/beta")
                .with_retrieval_text(build_retrieval_text(
                    "Beta",
                    "beta skill",
                    "body",
                    "beta",
                    "Skills.beta",
                ))
                .with_metadata(BTreeMap::from([(
                    META_SELECT_WHEN.to_string(),
                    "beta tasks".to_string(),
                )])),
            CatalogRecord::new("alpha", "Skills.alpha").with_name("Alpha"),
        ]
    }

    fn manifest_for(records: &[CatalogRecord]) -> IndexManifest {
        IndexManifest {
            mode: MODE_FULL.to_string(),
            count: records.len(),
            item_paths: vec!["/tmp/skills".to_string()],
            worker_ids: records
                .iter()
                .map(|record| record.worker_id.clone())
                .collect(),
            item_type: DEFAULT_ITEM_TYPE.to_string(),
            content_hash: catalog_content_hash(records),
        }
    }

    #[test]
    fn render_tree_yaml_matches_the_python_layout() {
        let nodes = vec![
            IndexNode::branch("Skills").with_description("All skills"),
            IndexNode::leaf("Skills.beta", "beta").with_source_description("from beta"),
            IndexNode::leaf("Skills.alpha", "alpha"),
        ];
        let expected = concat!(
            "nodes:\n",
            "  - cid: \"Skills\"\n",
            "    type: \"branch\"\n",
            "    description: \"All skills\"\n",
            "  - cid: \"Skills.alpha\"\n",
            "    type: \"leaf\"\n",
            "    worker_id: \"alpha\"\n",
            "  - cid: \"Skills.beta\"\n",
            "    type: \"leaf\"\n",
            "    source_description: \"from beta\"\n",
            "    worker_id: \"beta\"\n",
        );
        assert_eq!(render_tree_yaml(&nodes), expected);
    }

    #[test]
    fn render_tree_yaml_escapes_through_json_string_literals() {
        let nodes =
            vec![IndexNode::leaf("Skills.quoted\"leaf", "w").with_description("a: b\n能力")];
        let rendered = render_tree_yaml(&nodes);
        assert!(rendered.contains("cid: \"Skills.quoted\\\"leaf\""));
        assert!(rendered.contains("description: \"a: b\\n能力\""));
        assert!(rendered.ends_with('\n'));
    }

    #[test]
    fn tree_round_trips_through_render_and_parse() {
        let nodes = sample_nodes();
        let parsed = parse_tree_yaml(&render_tree_yaml(&nodes)).unwrap();
        let mut expected = nodes;
        sort_nodes(&mut expected);
        assert_eq!(parsed, expected);
    }

    #[test]
    fn parse_tree_yaml_sorts_and_tolerates_legacy_shapes() {
        let text = concat!(
            "nodes:\n",
            "  - cid: \"Skills.b\"\n",
            "    node_type: \"leaf\"\n",
            "    worker_id: 7\n",
            "  - just a string\n",
            "  - cid: \"Skills\"\n",
            "    type: \"branch\"\n",
            "    unknown_key: 1\n",
            "  - cid: \"Skills.a.b\"\n",
            "    type: \"leaf\"\n",
        );
        let nodes = parse_tree_yaml(text).unwrap();
        let cids: Vec<&str> = nodes.iter().map(|node| node.cid.as_str()).collect();
        assert_eq!(cids, vec!["Skills", "Skills.b", "Skills.a.b"]);
        assert_eq!(nodes[1].node_type, NODE_TYPE_LEAF);
        assert_eq!(nodes[1].worker_id, "7");
        assert!(nodes[0].is_branch());
    }

    #[test]
    fn parse_tree_yaml_rejects_a_document_without_nodes() {
        for text in ["", "nodes: 5", "other: 1"] {
            let error = parse_tree_yaml(text).expect_err("must reject");
            assert_eq!(error.kind(), crate::error::ErrorKind::Validation);
            assert!(error.message().contains(TREE_INDEX_FILENAME));
        }
        let error = parse_tree_yaml("nodes: [").expect_err("must reject");
        assert!(error.message().contains("invalid YAML"));
    }

    #[test]
    fn empty_node_lists_render_and_parse_as_an_empty_sequence() {
        let empty = Vec::<IndexNode>::new();
        assert_eq!(render_tree_yaml(&[]), "nodes: []\n");
        assert_eq!(parse_tree_yaml("nodes: []").unwrap(), empty);
        assert_eq!(
            parse_tree_yaml("nodes:\n").unwrap(),
            empty,
            "a preserved valueless header reads as an empty node list"
        );
        assert_eq!(parse_tree_yaml(&render_tree_yaml(&[])).unwrap(), empty);
    }

    #[test]
    fn catalog_round_trips_and_is_sorted_by_cid() {
        let records = sample_records();
        let rendered = render_catalog(&records);
        assert_eq!(rendered.lines().count(), 2);
        let first: Value = serde_json::from_str(rendered.lines().next().unwrap()).unwrap();
        assert_eq!(first["cid"], "Skills.alpha");
        let parsed = parse_catalog(&rendered).unwrap();
        let sorted: Vec<CatalogRecord> = {
            let mut sorted = records;
            sorted.sort_by(|left, right| left.cid.cmp(&right.cid));
            sorted
        };
        assert_eq!(parsed, sorted);
        assert_eq!(parsed[1].category, "Skills");
        assert_eq!(parsed[1].branch_path, vec!["Skills"]);
        assert_eq!(parsed[1].metadata_of(META_SELECT_WHEN), "beta tasks");
    }

    #[test]
    fn parse_catalog_tolerates_legacy_keys_and_rejects_bad_lines() {
        let text = concat!(
            "\n",
            "{\"skill_id\": \"w\", \"cid\": \"A.b\", \"metadata\": {\"n\": 3, \"nested\": {\"k\": true}}}\n",
            "{\"worker_id\": \"v\", \"cid\": \"A.c\", \"branch_path\": [\"A\"], \"category\": \"Custom\"}\n",
        );
        let records = parse_catalog(text).unwrap();
        assert_eq!(records[0].worker_id, "w");
        assert_eq!(records[0].metadata_of("n"), "3");
        assert_eq!(records[0].metadata_of("nested"), "{\"k\":true}");
        assert_eq!(records[1].worker_id, "v");
        assert_eq!(records[1].category, "Custom");

        let error = parse_catalog("{\"cid\": \"A\"}\nnot json\n").expect_err("must reject");
        assert_eq!(error.kind(), crate::error::ErrorKind::Validation);
        assert!(error.message().contains("catalog.jsonl: line 2"));
    }

    #[test]
    fn manifest_round_trips_with_its_field_order() {
        let records = sample_records();
        let manifest = manifest_for(&records);
        let rendered = render_manifest(&manifest);
        let positions: Vec<usize> = [
            "\"mode\"",
            "\"count\"",
            "\"item_paths\"",
            "\"worker_ids\"",
            "\"item_type\"",
            "\"content_hash\"",
        ]
        .iter()
        .map(|key| rendered.find(key).unwrap())
        .collect();
        let mut sorted_positions = positions.clone();
        sorted_positions.sort_unstable();
        assert_eq!(positions, sorted_positions);
        assert_eq!(parse_manifest(&rendered).unwrap(), manifest);
    }

    #[test]
    fn parse_manifest_defaults_the_item_type_only_when_absent() {
        let absent = parse_manifest("{\"mode\": \"full\", \"count\": 2}").unwrap();
        assert_eq!(absent.item_type, DEFAULT_ITEM_TYPE);
        assert_eq!(absent.count, 2);
        let blank = parse_manifest("{\"item_type\": \"  \"}").unwrap();
        assert_eq!(blank.item_type, "  ");
        let nullable = parse_manifest("{\"item_type\": null}").unwrap();
        assert_eq!(nullable.item_type, DEFAULT_ITEM_TYPE);
        let scalar = parse_manifest("{\"count\": \"3\", \"mode\": 1}").unwrap();
        assert_eq!(scalar.count, 0);
        assert_eq!(scalar.mode, "1");
        assert!(parse_manifest("[]").is_err());
        assert!(parse_manifest("nope").is_err());
    }

    #[test]
    fn catalog_content_hash_tracks_the_rendered_bytes() {
        let records = sample_records();
        let hash = catalog_content_hash(&records);
        assert_eq!(hash.len(), 64);
        assert!(hash
            .chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
        let mut changed = records;
        changed[0].name = "Changed".to_string();
        assert_ne!(catalog_content_hash(&changed), hash);
        assert_eq!(catalog_content_hash(&[]), {
            let mut hasher = Sha256::new();
            hasher.update(b"");
            format!("{:x}", hasher.finalize())
        });
    }

    #[test]
    fn build_tree_root_nests_prunes_and_derives_items() {
        let nodes = sample_nodes();
        let records = sample_records();
        let root = build_tree_root(&nodes, &records);
        assert_eq!(root.node_id, ROOT_NODE_ID);
        assert_eq!(root.label, ROOT_NODE_ID);
        let children = root.children();
        assert_eq!(children.len(), 1, "the empty branch is pruned");
        let skills = children[0];
        assert_eq!(skills.node_id, "Skills");
        assert_eq!(skills.label, "Skills");
        assert_eq!(skills.description, "All skills");
        let items: Vec<&str> = skills
            .items()
            .iter()
            .map(|item| item.item_id.as_str())
            .collect();
        assert_eq!(items, vec!["Alpha", "Beta"]);
        let beta = skills.items()[1];
        assert_eq!(beta.payload, "Skills.beta");
        assert_eq!(beta.label, "Beta");
        assert_eq!(beta.description, "beta skill");
    }

    #[test]
    fn build_tree_root_falls_back_to_the_cid_segment_and_reattaches_orphans() {
        let nodes = vec![
            IndexNode::branch("Skills"),
            IndexNode::leaf("Skills.unnamed", "unnamed").with_select_when("only when x"),
            IndexNode::branch("Missing.child"),
            IndexNode::leaf("Missing.child.item", "w"),
            IndexNode::leaf("Orphan", "o"),
        ];
        let root = build_tree_root(&nodes, &[]);
        let children = root.children();
        let ids: Vec<&str> = children.iter().map(|node| node.node_id.as_str()).collect();
        assert_eq!(
            ids,
            vec!["Missing.child", "Skills"],
            "the orphaned branch re-attaches to the root"
        );
        let skills = children[1];
        let item = skills.items()[0];
        assert_eq!(item.item_id, "unnamed");
        assert_eq!(item.label, "unnamed");
        assert_eq!(item.description, "Select when: only when x");
        let orphan_branch = children[0];
        assert_eq!(orphan_branch.label, "child");
        assert_eq!(orphan_branch.items()[0].item_id, "item");
        let root_items: Vec<&str> = root
            .items()
            .iter()
            .map(|item| item.item_id.as_str())
            .collect();
        assert_eq!(root_items, vec!["Orphan"]);
        assert_eq!(build_tree_root(&[], &[]).children, Vec::new());
    }

    #[test]
    fn compose_routing_description_adds_hints_once() {
        assert_eq!(
            compose_routing_description("  base  ", "pick it", "avoid it"),
            "base\nSelect when: pick it\nDon't select when: avoid it"
        );
        assert_eq!(
            compose_routing_description("base\nSelect when: already", "again", ""),
            "base\nSelect when: already"
        );
        assert_eq!(compose_routing_description("  ", "", ""), "");
        assert_eq!(
            compose_routing_description("", "pick it", ""),
            "Select when: pick it"
        );
    }

    #[tokio::test]
    async fn write_then_load_round_trips_one_index_directory() {
        let directory = tempfile::tempdir().unwrap();
        let nodes = sample_nodes();
        let records = sample_records();
        let manifest = manifest_for(&records);
        write_index(directory.path(), &nodes, &records, &manifest)
            .await
            .unwrap();

        let loaded = load_index(directory.path()).await.unwrap();
        assert_eq!(loaded.manifest, manifest);
        assert_eq!(loaded.records.len(), 2);
        assert_eq!(loaded.root.node_id, ROOT_NODE_ID);
        assert_eq!(
            loaded.choices.len(),
            2,
            "both items are flattened into choices"
        );
        assert_eq!(
            loaded.item_paths.get("Skills.beta"),
            Some(&vec!["Skills".to_string()])
        );
        assert_eq!(
            loaded.item_paths.get("Skills.alpha"),
            Some(&vec!["Skills".to_string()])
        );
        assert!(!directory.path().join("tree_index.yaml.tmp").exists());
    }

    #[tokio::test]
    async fn write_index_rejects_duplicate_and_unmatched_cids() {
        let directory = tempfile::tempdir().unwrap();
        let records = sample_records();
        let manifest = manifest_for(&records);
        let empty_manifest = manifest_for(&[]);
        let duplicated = vec![IndexNode::branch("A"), IndexNode::branch("A")];
        let error = write_index(directory.path(), &duplicated, &[], &empty_manifest)
            .await
            .expect_err("must reject");
        assert!(error.message().contains("duplicate cid `A`"));

        let blank = vec![IndexNode::branch("  ")];
        let error = write_index(directory.path(), &blank, &[], &empty_manifest)
            .await
            .expect_err("must reject");
        assert!(error.message().contains("node cid must not be blank"));

        let nodes = vec![IndexNode::branch("A")];
        let error = write_index(directory.path(), &nodes, &records, &manifest)
            .await
            .expect_err("must reject");
        assert!(error.message().contains("has no leaf node"));

        let duplicate_records = vec![
            CatalogRecord::new("one", "A.b"),
            CatalogRecord::new("two", "A.b"),
        ];
        let leaves = vec![IndexNode::leaf("A.b", "one")];
        let error = write_index(
            directory.path(),
            &leaves,
            &duplicate_records,
            &manifest_for(&duplicate_records),
        )
        .await
        .expect_err("must reject");
        assert!(error.message().contains("duplicate cid `A.b`"));
        assert!(!directory.path().join(TREE_INDEX_FILENAME).exists());
    }

    #[tokio::test]
    async fn load_index_reports_missing_files_and_inconsistent_manifests() {
        let directory = tempfile::tempdir().unwrap();
        let error = load_index(directory.path()).await.expect_err("must fail");
        assert_eq!(error.kind(), crate::error::ErrorKind::NotFound);
        assert_eq!(error.message(), TREE_INDEX_FILENAME);

        let nodes = sample_nodes();
        let records = sample_records();
        fs::write(
            directory.path().join(TREE_INDEX_FILENAME),
            render_tree_yaml(&nodes),
        )
        .await
        .unwrap();
        fs::write(
            directory.path().join(CATALOG_FILENAME),
            render_catalog(&records),
        )
        .await
        .unwrap();
        fs::write(
            directory.path().join(MANIFEST_FILENAME),
            render_manifest(&IndexManifest {
                count: 5,
                ..manifest_for(&records)
            }),
        )
        .await
        .unwrap();
        let error = load_index(directory.path()).await.expect_err("must fail");
        assert!(error.message().contains("count 5 does not match 2"));

        fs::write(
            directory.path().join(MANIFEST_FILENAME),
            render_manifest(&IndexManifest {
                content_hash: "deadbeef".to_string(),
                ..manifest_for(&records)
            }),
        )
        .await
        .unwrap();
        let error = load_index(directory.path()).await.expect_err("must fail");
        assert!(error
            .message()
            .contains("content_hash deadbeef does not match"));

        fs::write(
            directory.path().join(MANIFEST_FILENAME),
            render_manifest(&IndexManifest {
                item_type: " ".to_string(),
                content_hash: String::new(),
                ..manifest_for(&records)
            }),
        )
        .await
        .unwrap();
        let error = load_index(directory.path()).await.expect_err("must fail");
        assert!(error.message().contains("item_type must not be blank"));

        fs::write(
            directory.path().join(MANIFEST_FILENAME),
            render_manifest(&manifest_for(&records)),
        )
        .await
        .unwrap();
        assert!(load_index(directory.path()).await.is_ok());
    }

    #[test]
    fn cid_depth_matches_the_index_node_depth() {
        assert_eq!(cid_depth("A.B.C"), IndexNode::branch("A.B.C").depth());
        assert_eq!(cid_depth(""), 0);
    }
}
