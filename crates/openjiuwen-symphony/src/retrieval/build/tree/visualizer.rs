//! The optional `tree_index.html` visualizer of a built capability tree.
//!
//! Ports Python `openjiuwen.symphony.retrieval.build.tree.visualizer` (v0.1.20): a built
//! node list plus its catalog records are rendered as one self-contained HTML5 document
//! for humans. There is no external stylesheet, no font, no image and no script, so the
//! document is fully readable without `JavaScript`.
//!
//! # An opt-in extra artifact
//!
//! `tree_index.html` is **not** one of the files a build publishes: the artifact contract
//! is the three files written by
//! [`write_index`](crate::retrieval::artifacts::write_index) — `tree_index.yaml`,
//! `catalog.jsonl` and `manifest.json`. The visualizer is opt-in, nothing in a build calls
//! it, and it writes its single extra file next to those three.
//!
//! # Derived data, never read back
//!
//! The document is derived from the node list and the catalog records, and no search path,
//! loader or index reader opens it, so a stale or deleted `tree_index.html` changes no
//! behaviour. It carries no timestamp, no clock reading, no random id and no crate version,
//! so two renders of the same input are byte-identical.
//!
//! # Shape
//!
//! The hierarchy comes from the `cid` prefix rule every other build stage uses: `A.b` sits
//! under `A`, and the top-level segments are the document's roots. Branches and leaves are
//! rendered as nested `<ul>`/`<li>` markup in `cid` order, a branch line shows its label,
//! its `cid`, its recursive leaf count and its description, and a leaf line shows the
//! matching catalog record (or the node's `worker_id` when no record matches) with its
//! `cid` and description. A catalog record whose `cid` names no leaf node is not dropped:
//! it is listed in an `Unplaced items` section after the tree, in `cid` order.
//!
//! Every rendered value passes through one escaper, so `&`, `<`, `>` and both quote
//! characters are entities rather than markup.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use tokio::fs;
use tracing::debug;

use crate::error::{Error, Result};
use crate::retrieval::types::{CatalogRecord, IndexNode, TREE_HTML_FILENAME};

/// The heading a blank `root_label` falls back to.
const DEFAULT_ROOT_LABEL: &str = "Capacity tree";

/// The document's inline stylesheet; the visualizer ships no external asset.
const STYLE: &str = "\
:root { color-scheme: light dark; }
body { font-family: system-ui, -apple-system, \"Segoe UI\", sans-serif; line-height: 1.45; margin: 2rem; }
h1 { font-size: 1.5rem; margin: 0 0 0.25rem; }
h2 { font-size: 1.1rem; margin: 0 0 0.5rem; }
.summary { margin: 0 0 1rem; opacity: 0.75; }
ul.tree, ul.tree ul { border-left: 1px solid rgba(127, 127, 127, 0.35); list-style: none; margin: 0; padding-left: 1rem; }
ul.tree { border-left: none; padding-left: 0; }
li { margin: 0.3rem 0; }
.branch-label { font-weight: 600; }
.leaf-name { font-weight: 500; }
.cid { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.8em; margin-left: 0.4rem; opacity: 0.7; }
.count { font-size: 0.8em; margin-left: 0.4rem; opacity: 0.7; }
.description { font-size: 0.95em; margin: 0.15rem 0 0.25rem; opacity: 0.85; }
.metadata { font-size: 0.85em; margin: 0.15rem 0 0.25rem; opacity: 0.8; }
.metadata dt { display: inline; font-weight: 600; }
.metadata dd { display: inline; margin: 0 0.6rem 0 0.2rem; }
.empty { opacity: 0.7; }";

/// Renders `nodes` and `records` as one self-contained HTML5 document.
///
/// The hierarchy is derived from the `cid` prefix rule, so `A.b.c` is nested under
/// `A.b`; a node whose parent `cid` names no branch node is rendered at the top level
/// with its full `cid`, so nothing disappears. Branches and leaves of one parent are
/// rendered interleaved in `cid` order, with the node type as the tie-breaker, and every
/// collection is a `BTreeMap`/`BTreeSet`, so two renders of the same input are
/// byte-identical.
///
/// A branch line shows the last `cid` segment as its label, the `cid`, the recursive
/// number of leaf items below it and its description when non-blank. A leaf line shows
/// the matching catalog record's name (falling back to the node's `worker_id`, then to
/// the last `cid` segment), the `cid`, the record's description (falling back to the
/// node's) when non-blank and the record's metadata.
///
/// A catalog record whose `cid` names no leaf node is listed in a final `Unplaced items`
/// section in `cid` order, and a leaf node with no catalog record still renders from the
/// node itself. An empty node list renders a valid document with zero counts and an empty
/// tree section.
///
/// A blank `root_label` falls back to `Capacity tree`. Every string value is HTML-escaped.
#[must_use]
pub fn render_tree_html(
    nodes: &[IndexNode],
    records: &[CatalogRecord],
    root_label: &str,
) -> String {
    let label = root_label.trim();
    let label = if label.is_empty() {
        DEFAULT_ROOT_LABEL
    } else {
        label
    };
    let view = TreeView::collect(nodes, records);
    let item_count = view.item_count("");
    let unplaced = view.unplaced_records();
    let mut lines: Vec<String> = Vec::new();
    lines.push("<!DOCTYPE html>".to_string());
    lines.push("<html lang=\"en\">".to_string());
    lines.push("<head>".to_string());
    lines.push("<meta charset=\"utf-8\">".to_string());
    lines.push(
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">".to_string(),
    );
    lines.push(format!("<title>{}</title>", escape_html(label)));
    lines.push("<style>".to_string());
    lines.extend(STYLE.lines().map(str::to_string));
    lines.push("</style>".to_string());
    lines.push("</head>".to_string());
    lines.push("<body>".to_string());
    lines.push("<header>".to_string());
    lines.push(format!("<h1>{}</h1>", escape_html(label)));
    lines.push(format!(
        "<p class=\"summary\">{}, {}</p>",
        branch_count_label(view.branch_count),
        item_count_label(item_count)
    ));
    lines.push("</header>".to_string());
    lines.push("<main>".to_string());
    lines.push("<section id=\"tree\">".to_string());
    lines.push("<h2>Capability tree</h2>".to_string());
    if view.has_children("") {
        lines.push("<ul class=\"tree\">".to_string());
        render_entries(&view, "", &mut lines);
        lines.push("</ul>".to_string());
    } else {
        lines.push("<p class=\"empty\">No capabilities are indexed in this tree.</p>".to_string());
    }
    lines.push("</section>".to_string());
    if !unplaced.is_empty() {
        lines.push("<section id=\"unplaced\">".to_string());
        lines.push(format!(
            "<h2>Unplaced items ({})</h2>",
            item_count_label(unplaced.len())
        ));
        lines.push("<ul class=\"unplaced\">".to_string());
        for record in unplaced {
            lines.push("<li class=\"leaf unplaced\">".to_string());
            lines.push(format!(
                "<span class=\"leaf-name\">{}</span><code class=\"cid\">{}</code>",
                escape_html(&record_name(record)),
                escape_html(&record.cid)
            ));
            if !record.description.trim().is_empty() {
                lines.push(format!(
                    "<p class=\"description\">{}</p>",
                    escape_html(record.description.trim())
                ));
            }
            render_metadata(record, &mut lines);
            lines.push("</li>".to_string());
        }
        lines.push("</ul>".to_string());
        lines.push("</section>".to_string());
    }
    lines.push("</main>".to_string());
    lines.push("</body>".to_string());
    lines.push("</html>".to_string());
    let mut document = lines.join("\n");
    document.push('\n');
    document
}

/// Renders the tree and its metadata into `dir/tree_index.html`, atomically.
///
/// The directory is created when missing, then the document is written to a
/// same-directory `tree_index.html.tmp` sibling and renamed onto the final name, so a
/// reader never observes a half-written file and a successful write leaves no temporary
/// sibling behind. The file is opt-in extra data: it is not part of the three-artifact
/// index contract and nothing reads it back.
///
/// The rendered bytes are exactly [`render_tree_html`]'s output, so writing twice with the
/// same input overwrites the file with identical content.
///
/// # Errors
///
/// [`Error::Config`] when the target path has no usable file name, and
/// [`Error::External`] with the action and path as context when the directory cannot be
/// created, the temporary file cannot be written or the rename fails.
pub async fn write_tree_html(
    dir: impl AsRef<Path>,
    nodes: &[IndexNode],
    records: &[CatalogRecord],
    root_label: &str,
) -> Result<()> {
    let dir = dir.as_ref();
    let document = render_tree_html(nodes, records, root_label);
    fs::create_dir_all(dir)
        .await
        .map_err(|error| path_error("creating", dir, error))?;
    let path = dir.join(TREE_HTML_FILENAME);
    write_atomic(&path, &document).await?;
    debug!(
        path = %path.display(),
        nodes = nodes.len(),
        records = records.len(),
        "wrote a capability tree visualizer document"
    );
    Ok(())
}

/// Appends the `<li>` entries of `parent`'s children, in `cid` order (no `<ul>` wrapper).
fn render_entries(view: &TreeView<'_>, parent: &str, lines: &mut Vec<String>) {
    let Some(nodes) = view.children.get(parent) else {
        return;
    };
    for node in nodes {
        if node.is_leaf() {
            render_leaf(view, node, lines);
        } else {
            render_branch(view, node, lines);
        }
    }
}

/// Appends one branch `<li>`, nested into its own `<ul>` when it has children.
fn render_branch(view: &TreeView<'_>, node: &IndexNode, lines: &mut Vec<String>) {
    lines.push("<li class=\"branch\">".to_string());
    lines.push(format!(
        "<span class=\"branch-label\">{}</span><code class=\"cid\">{}</code><span class=\"count\">{}</span>",
        escape_html(last_segment(&node.cid)),
        escape_html(&node.cid),
        item_count_label(view.item_count(&node.cid))
    ));
    if !node.description.trim().is_empty() {
        lines.push(format!(
            "<p class=\"description\">{}</p>",
            escape_html(node.description.trim())
        ));
    }
    if view.has_children(&node.cid) {
        lines.push("<ul>".to_string());
        render_entries(view, &node.cid, lines);
        lines.push("</ul>".to_string());
    }
    lines.push("</li>".to_string());
}

/// Appends one leaf `<li>`, using its catalog record when one matches its `cid`.
fn render_leaf(view: &TreeView<'_>, node: &IndexNode, lines: &mut Vec<String>) {
    let record = view.records.get(node.cid.as_str()).copied();
    let name = match record {
        Some(record) if !record.name.trim().is_empty() => record.name.trim().to_string(),
        _ if !node.worker_id.trim().is_empty() => node.worker_id.trim().to_string(),
        _ => last_segment(&node.cid).to_string(),
    };
    let description = match record {
        Some(record) if !record.description.trim().is_empty() => record.description.trim(),
        _ => node.description.trim(),
    };
    lines.push("<li class=\"leaf\">".to_string());
    lines.push(format!(
        "<span class=\"leaf-name\">{}</span><code class=\"cid\">{}</code>",
        escape_html(&name),
        escape_html(&node.cid)
    ));
    if !description.is_empty() {
        lines.push(format!(
            "<p class=\"description\">{}</p>",
            escape_html(description)
        ));
    }
    if let Some(record) = record {
        render_metadata(record, lines);
    }
    lines.push("</li>".to_string());
}

/// Appends a record's metadata as a `<dl>`, sorted by key (a `BTreeMap`).
fn render_metadata(record: &CatalogRecord, lines: &mut Vec<String>) {
    if record.metadata.is_empty() {
        return;
    }
    lines.push("<dl class=\"metadata\">".to_string());
    for (key, value) in &record.metadata {
        lines.push(format!(
            "<dt>{}</dt><dd>{}</dd>",
            escape_html(key),
            escape_html(value)
        ));
    }
    lines.push("</dl>".to_string());
}

/// The last `.`-separated segment of `cid` (the whole value when it has no dot).
fn last_segment(cid: &str) -> &str {
    cid.rsplit('.').next().unwrap_or(cid)
}

/// A record's display name, falling back to its worker id and then its `cid`.
fn record_name(record: &CatalogRecord) -> String {
    let name = record.name.trim();
    if !name.is_empty() {
        return name.to_string();
    }
    let worker_id = record.worker_id.trim();
    if !worker_id.is_empty() {
        return worker_id.to_string();
    }
    last_segment(&record.cid).to_string()
}

/// `"1 item"` / `"3 items"`.
fn item_count_label(count: usize) -> String {
    if count == 1 {
        "1 item".to_string()
    } else {
        format!("{count} items")
    }
}

/// `"1 branch"` / `"3 branches"`.
fn branch_count_label(count: usize) -> String {
    if count == 1 {
        "1 branch".to_string()
    } else {
        format!("{count} branches")
    }
}

/// Escapes the five HTML metacharacters, the ampersand first so nothing is double escaped.
fn escape_html(text: &str) -> String {
    let mut escaped = String::with_capacity(text.len());
    for character in text.chars() {
        match character {
            '&' => escaped.push_str("&amp;"),
            '<' => escaped.push_str("&lt;"),
            '>' => escaped.push_str("&gt;"),
            '"' => escaped.push_str("&quot;"),
            '\'' => escaped.push_str("&#39;"),
            other => escaped.push(other),
        }
    }
    escaped
}

/// The tree a visualizer render walks: children by parent `cid` plus the record lookup.
struct TreeView<'a> {
    /// Child nodes by parent `cid`; the empty key holds the top-level nodes.
    children: BTreeMap<&'a str, Vec<&'a IndexNode>>,
    /// Every leaf `cid`, used to find catalog records with no leaf node.
    leaf_cids: BTreeSet<&'a str>,
    /// The catalog records by leaf `cid`; duplicates resolve to the first in `cid` order.
    records: BTreeMap<&'a str, &'a CatalogRecord>,
    /// How many branch nodes the input holds.
    branch_count: usize,
}

impl<'a> TreeView<'a> {
    /// Indexes `nodes` by parent `cid` and `records` by `cid`, both in deterministic order.
    fn collect(nodes: &'a [IndexNode], records: &'a [CatalogRecord]) -> Self {
        let branch_cids: BTreeSet<&str> = nodes
            .iter()
            .filter(|node| node.is_branch())
            .map(|node| node.cid.as_str())
            .collect();
        let leaf_cids: BTreeSet<&str> = nodes
            .iter()
            .filter(|node| node.is_leaf())
            .map(|node| node.cid.as_str())
            .collect();
        let branch_count = nodes.iter().filter(|node| node.is_branch()).count();
        let mut children: BTreeMap<&str, Vec<&IndexNode>> = BTreeMap::new();
        for node in nodes {
            let parent = match node.parent_cid() {
                Some(parent) if branch_cids.contains(parent) => parent,
                _ => "",
            };
            children.entry(parent).or_default().push(node);
        }
        for group in children.values_mut() {
            group.sort_by(|left, right| {
                (left.cid.as_str(), left.node_type.as_str())
                    .cmp(&(right.cid.as_str(), right.node_type.as_str()))
            });
        }
        let mut ordered: Vec<&CatalogRecord> = records.iter().collect();
        ordered.sort_by(|left, right| {
            (left.cid.as_str(), left.worker_id.as_str())
                .cmp(&(right.cid.as_str(), right.worker_id.as_str()))
        });
        let mut by_cid: BTreeMap<&str, &CatalogRecord> = BTreeMap::new();
        for record in ordered {
            by_cid.entry(record.cid.as_str()).or_insert(record);
        }
        Self {
            children,
            leaf_cids,
            records: by_cid,
            branch_count,
        }
    }

    /// Whether the node at `cid` has at least one child.
    fn has_children(&self, cid: &str) -> bool {
        self.children
            .get(cid)
            .is_some_and(|nodes| !nodes.is_empty())
    }

    /// The recursive number of leaf nodes below the node at `parent`.
    fn item_count(&self, parent: &str) -> usize {
        let Some(nodes) = self.children.get(parent) else {
            return 0;
        };
        nodes
            .iter()
            .map(|node| {
                if node.is_leaf() {
                    1
                } else {
                    self.item_count(&node.cid)
                }
            })
            .sum()
    }

    /// Every catalog record whose `cid` names no leaf node, in `(cid, worker_id)` order.
    fn unplaced_records(&self) -> Vec<&'a CatalogRecord> {
        let mut unplaced: Vec<&CatalogRecord> = self
            .records
            .values()
            .copied()
            .filter(|record| !self.leaf_cids.contains(record.cid.as_str()))
            .collect();
        unplaced.sort_by(|left, right| {
            (left.cid.as_str(), left.worker_id.as_str())
                .cmp(&(right.cid.as_str(), right.worker_id.as_str()))
        });
        unplaced
    }
}

/// Writes `text` to `path` atomically: `<path>.tmp` in the same directory, then `rename`.
async fn write_atomic(path: &Path, text: &str) -> Result<()> {
    let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
        return Err(Error::config(format!(
            "visualizer artifact path `{}` has no file name",
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
                "failed to remove a temporary visualizer document"
            );
        }
        return Err(path_error("committing", path, error));
    }
    Ok(())
}

/// Wraps an I/O failure as [`Error::External`], naming the action and the path.
fn path_error(action: &str, path: &Path, error: std::io::Error) -> Error {
    Error::external(error).with_context(format!("{action} `{}`", path.display()))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::{render_tree_html, write_tree_html};
    use crate::retrieval::types::{CatalogRecord, IndexNode, TREE_HTML_FILENAME};

    fn branch(cid: &str, description: &str) -> IndexNode {
        IndexNode::branch(cid).with_description(description)
    }

    fn leaf(cid: &str, worker_id: &str) -> IndexNode {
        IndexNode::leaf(cid, worker_id)
    }

    fn record(cid: &str, worker_id: &str, name: &str) -> CatalogRecord {
        CatalogRecord::new(worker_id, cid).with_name(name)
    }

    fn sample_nodes() -> Vec<IndexNode> {
        vec![
            branch("Skills", ""),
            branch("Skills.documents", "Document capabilities."),
            leaf("Skills.documents.pdf-fill.pdffill", "PdfFill"),
            leaf("Skills.documents.docx-fill", "DocxFill"),
            branch("Skills.tools", "Tool capabilities."),
            leaf("Skills.tools.web-search", "WebSearch"),
        ]
    }

    fn sample_records() -> Vec<CatalogRecord> {
        vec![
            record("Skills.documents.pdf-fill.pdffill", "PdfFill", "PDF fill"),
            record("Skills.documents.docx-fill", "DocxFill", "DOCX fill"),
            record("Skills.tools.web-search", "WebSearch", "Web search"),
        ]
    }

    fn count_of(haystack: &str, needle: &str) -> usize {
        haystack.match_indices(needle).count()
    }

    #[test]
    fn renders_byte_identical_documents() {
        let nodes = sample_nodes();
        let records = sample_records();
        let first = render_tree_html(&nodes, &records, "Skills");
        let second = render_tree_html(&nodes, &records, "Skills");
        assert_eq!(first, second);
        assert!(first.starts_with("<!DOCTYPE html>\n"));
        assert!(first.ends_with("</html>\n"));
        assert!(!first.contains("<script"));
    }

    #[test]
    fn orders_branches_and_leaves_by_cid() {
        let nodes = vec![
            branch("A", ""),
            branch("A.zeta", ""),
            leaf("A.beta", "Beta"),
            branch("A.alpha", ""),
            leaf("A.alpha.two", "Two"),
            leaf("A.alpha.one", "One"),
        ];
        let document = render_tree_html(&nodes, &[], "Root");
        let alpha = document.find("<code class=\"cid\">A.alpha</code>").unwrap();
        let beta = document.find("<code class=\"cid\">A.beta</code>").unwrap();
        let zeta = document.find("<code class=\"cid\">A.zeta</code>").unwrap();
        assert!(alpha < beta, "children must be interleaved by cid");
        assert!(beta < zeta, "branches and leaves share one cid order");
        let one = document
            .find("<code class=\"cid\">A.alpha.one</code>")
            .unwrap();
        let two = document
            .find("<code class=\"cid\">A.alpha.two</code>")
            .unwrap();
        assert!(one < two, "leaves of one branch are in cid order");
    }

    #[test]
    fn escapes_every_html_metacharacter() {
        let mut node = branch("A", "a & b < c > d \"e\" 'f'");
        node.worker_id = "Ampersand & <tag>".to_string();
        let nodes = vec![
            node,
            leaf("A.leaf", "MatchedWorker"),
            leaf("A.orphan", "Worker & <id> \"quoted\" 'single'"),
        ];
        let mut record = record("A.leaf", "MatchedWorker", "Name & <script>");
        record.description = "quotes \" and ' and < and > and &".to_string();
        record
            .metadata
            .insert("key<&>".to_string(), "value\"'&".to_string());
        let document = render_tree_html(&nodes, &[record], "Tree <&> \"root\" 'label'");
        assert!(!document.contains("<script"));
        assert!(document.contains("a &amp; b &lt; c &gt; d &quot;e&quot; &#39;f&#39;"));
        assert!(document.contains("Name &amp; &lt;script&gt;"));
        assert!(document.contains("quotes &quot; and &#39; and &lt; and &gt; and &amp;"));
        assert!(document.contains("<h1>Tree &lt;&amp;&gt; &quot;root&quot; &#39;label&#39;</h1>"));
        assert!(document.contains("<dt>key&lt;&amp;&gt;</dt><dd>value&quot;&#39;&amp;</dd>"));
        assert!(document.contains(
            "<span class=\"leaf-name\">Worker &amp; &lt;id&gt; &quot;quoted&quot; \
             &#39;single&#39;</span><code class=\"cid\">A.orphan</code>"
        ));
        assert!(!document.contains("&amp;amp;"), "no double escaping");
    }

    #[test]
    fn escapes_a_script_injection_attempt() {
        let nodes = vec![
            branch("A", "<script>alert('branch')</script>"),
            leaf("A.leaf", "<script>alert('leaf')</script>"),
        ];
        let mut matched = record(
            "A.leaf",
            "<script>alert('worker')</script>",
            "<script>alert('name')</script>",
        );
        matched.description = "<script>alert('description')</script>".to_string();
        let loose = record(
            "A.missing",
            "<script>alert('worker')</script>",
            "<script>alert('loose')</script>",
        );
        let document =
            render_tree_html(&nodes, &[matched, loose], "<script>alert('title')</script>");
        assert!(!document.contains("<script"), "no raw script tag survives");
        // The branch description, the record name, the record description, the unplaced
        // record name, the title and the heading.
        assert_eq!(count_of(&document, "&lt;script&gt;"), 6);
        assert!(document.contains("&lt;script&gt;alert(&#39;title&#39;)&lt;/script&gt;"));
    }

    #[test]
    fn nests_nodes_by_cid_prefix() {
        let nodes = vec![
            branch("Skills", "Root branch."),
            branch("Skills.documents", "Documents."),
            leaf("Skills.documents.pdf-fill", "PdfFill"),
            leaf("Skills.tools", "Tools"),
        ];
        let document = render_tree_html(&nodes, &[], "Skills");
        let root = document.find("<code class=\"cid\">Skills</code>").unwrap();
        let documents = document
            .find("<code class=\"cid\">Skills.documents</code>")
            .unwrap();
        let pdf_fill = document
            .find("<code class=\"cid\">Skills.documents.pdf-fill</code>")
            .unwrap();
        let tools = document
            .find("<code class=\"cid\">Skills.tools</code>")
            .unwrap();
        assert!(root < documents && documents < pdf_fill);
        assert!(pdf_fill < tools);
        // One `<ul class="tree">` plus one nested `<ul>` for `Skills` and one for the
        // `Skills.documents` branch.
        assert_eq!(count_of(&document, "<ul"), 3);
        assert_eq!(count_of(&document, "</ul>"), 3);
    }

    #[test]
    fn counts_items_per_branch_and_in_the_header() {
        let nodes = vec![
            branch("A", ""),
            leaf("A.one", "One"),
            branch("A.b", ""),
            leaf("A.b.two", "Two"),
            leaf("A.b.three", "Three"),
        ];
        let document = render_tree_html(&nodes, &[], "Counts");
        assert!(document.contains("<p class=\"summary\">2 branches, 3 items</p>"));
        assert!(document.contains(
            "<span class=\"branch-label\">A</span><code class=\"cid\">A</code>\
             <span class=\"count\">3 items</span>"
        ));
        assert!(document.contains(
            "<span class=\"branch-label\">b</span><code class=\"cid\">A.b</code>\
             <span class=\"count\">2 items</span>"
        ));
        let singular = vec![branch("Solo", ""), leaf("Solo.one", "One")];
        let document = render_tree_html(&singular, &[], "Singular");
        assert!(document.contains("<p class=\"summary\">1 branch, 1 item</p>"));
        assert!(document.contains("<span class=\"count\">1 item</span>"));
    }

    #[test]
    fn renders_an_empty_tree() {
        let document = render_tree_html(&[], &[], "");
        assert!(document.starts_with("<!DOCTYPE html>\n<html lang=\"en\">\n"));
        assert!(document.contains("<h1>Capacity tree</h1>"));
        assert!(document.contains("<p class=\"summary\">0 branches, 0 items</p>"));
        assert!(document.contains("<section id=\"tree\">"));
        assert!(document.contains("No capabilities are indexed in this tree."));
        assert!(!document.contains("<ul"));
        assert!(document.contains("<style>"));
    }

    #[test]
    fn renders_a_leaf_without_a_record_and_a_record_without_a_leaf() {
        let nodes = vec![
            branch("A", ""),
            leaf("A.known", "KnownWorker"),
            leaf("A.orphan", "OrphanWorker"),
        ];
        let mut known = record("A.known", "KnownWorker", "Known name");
        known.description = "Known description.".to_string();
        let mut loose = record("A.missing", "LooseWorker", "Loose name");
        loose.description = "Loose description.".to_string();
        let document = render_tree_html(&nodes, &[known, loose], "Partial");
        assert!(document.contains(
            "<span class=\"leaf-name\">Known name</span>\
             <code class=\"cid\">A.known</code>"
        ));
        assert!(document.contains("Known description."));
        assert!(document.contains(
            "<span class=\"leaf-name\">OrphanWorker</span>\
             <code class=\"cid\">A.orphan</code>"
        ));
        assert!(document.contains("<h2>Unplaced items (1 item)</h2>"));
        assert!(document.contains(
            "<span class=\"leaf-name\">Loose name</span>\
             <code class=\"cid\">A.missing</code>"
        ));
        assert!(document.contains("Loose description."));
    }

    #[tokio::test]
    async fn writes_the_document_atomically() {
        let directory = tempfile::tempdir().unwrap();
        let nodes = sample_nodes();
        let records = sample_records();
        let nested = directory.path().join("nested/index");
        write_tree_html(&nested, &nodes, &records, "Skills")
            .await
            .unwrap();
        let path = nested.join(TREE_HTML_FILENAME);
        let first = std::fs::read_to_string(&path).unwrap();
        assert_eq!(first, render_tree_html(&nodes, &records, "Skills"));
        write_tree_html(&nested, &nodes, &records, "Skills")
            .await
            .unwrap();
        let second = std::fs::read_to_string(&path).unwrap();
        assert_eq!(first, second);
        let mut names: Vec<String> = std::fs::read_dir(&nested)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        assert_eq!(names, vec![TREE_HTML_FILENAME.to_string()]);
    }
}
