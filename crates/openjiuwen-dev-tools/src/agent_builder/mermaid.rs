//! Mermaid rendering for the workflow DSL and the typed workflow design.
//!
//! Ports
//! `openjiuwen.dev_tools.agent_builder.builders.workflow.dl_transformer.simpleir_to_mermaid`
//! to the typed graphs of this crate, and keeps Python's exact output grammar.
//! Python walks a node list, derives edges from `next` and `conditions`, and
//! emits one `Node<id>: <description>` rectangle per node; this module renders
//! the [`WorkflowDsl`] and the [`WorkflowDesign`] with the same grammar:
//!
//! ```text
//! graph TD
//!   start[Nodestart: Begin]
//!   decide[Nodedecide: Decide]
//!   yes[Nodeyes: Yes path]
//!   start --> decide
//!   decide -- score > 0.5 --> yes
//! ```
//!
//! Rendering is deterministic and follows document order: the literal
//! `graph TD` header, then one two-space-indented line per node in node-array
//! order, then one line per edge in edge-array order. There is no trailing
//! newline.
//!
//! Node lines are always `  {id}[Node{id}: {label}]` (square brackets, no space
//! after `Node`). The label is the node description, falling back to the node id
//! when the description is blank; the title is not used. Labels escape only
//! backticks and double quotes, both to a single quote, mirroring Python's two
//! chained `str.replace` calls; whitespace and shape delimiters are left
//! untouched exactly as Python leaves them.
//!
//! Edge lines are `  {src} --> {dst}`. A labelled edge is
//! `  {src} -- {label} --> {dst}` only when the source node has more than one
//! outgoing edge *and* the edge condition is non-blank; the label is the escaped
//! condition. A single-successor source never gets an edge label, even when its
//! edge carries a condition.
//!
//! A condition written as a `When condition met[...]` description is reduced to
//! the label it carries by [`when_condition_label`], so the wrapper prefix and
//! its brackets never reach the diagram; every other condition keeps rendering
//! verbatim.
//! Python derives that label with a regex that never produces a usable capture,
//! so the port keeps the intent (the bracketed text, or the remainder after the
//! prefix) instead of reproducing the one-character capture.
//!
//! Ids are sanitized (ASCII alphanumerics and `_` kept, every other character
//! mapped to `_`, empty ids named `node_<index>`, digit-leading ids prefixed with
//! `n_`, collisions resolved with `_2`, `_3`, ... in document order) so that a
//! document with non-ASCII or duplicate ids still renders a parseable diagram.
//! An edge whose endpoint names no node keeps its sanitized form.
//!
//! [`render_design`] feeds the same renderer, so it produces byte-identical
//! output to [`render_mermaid`] over the equivalent
//! [`WorkflowDesign::to_workflow_dsl`] document.

use std::collections::{BTreeMap, BTreeSet};

use super::designer::WorkflowDesign;
use super::workflow_dsl::WorkflowDsl;

mod labels;

use labels::{edge_label, escape_label, resolve_endpoint, sanitize_id, unique_id};

pub use labels::when_condition_label;

/// The Mermaid header line of every rendered diagram.
const HEADER: &str = "graph TD";

/// A node of the intermediate graph both public entry points render.
struct RenderNode<'a> {
    /// The original node id, used for the id map, the label prefix, and the
    /// label fallback when the description is blank.
    id: &'a str,
    /// The node description, used as the label.
    description: &'a str,
}

/// An edge of the intermediate graph both public entry points render.
struct RenderEdge<'a> {
    /// The original id of the node the edge leaves.
    source: &'a str,
    /// The original id of the node the edge enters.
    target: &'a str,
    /// The optional condition rendered as the edge label.
    condition: Option<&'a str>,
}

/// Renders a workflow DSL document as a Mermaid `graph TD` diagram.
///
/// Nodes appear in document order as `  {id}[Node{id}: {description}]`, edges
/// follow as `  {src} --> {dst}`, and an edge is labelled `-- {condition} -->`
/// only when its source has more than one outgoing edge. Ids are sanitized and
/// deduplicated, so rendering the same document twice yields byte-identical
/// output.
#[must_use]
pub fn render_mermaid(dsl: &WorkflowDsl) -> String {
    let nodes: Vec<RenderNode<'_>> = dsl
        .nodes
        .iter()
        .map(|node| RenderNode {
            id: node.id.as_str(),
            description: node.description.as_str(),
        })
        .collect();
    let edges: Vec<RenderEdge<'_>> = dsl
        .edges
        .iter()
        .map(|edge| RenderEdge {
            source: edge.source.as_str(),
            target: edge.target.as_str(),
            condition: edge.condition.as_deref(),
        })
        .collect();

    render(&nodes, &edges)
}

/// Renders a workflow design as a Mermaid `graph TD` diagram.
///
/// Uses the same renderer as [`render_mermaid`], so a design renders
/// byte-identically to the [`WorkflowDsl`] produced by
/// [`WorkflowDesign::to_workflow_dsl`].
#[must_use]
pub fn render_design(design: &WorkflowDesign) -> String {
    let nodes: Vec<RenderNode<'_>> = design
        .nodes
        .iter()
        .map(|node| RenderNode {
            id: node.id.as_str(),
            description: node.description.as_str(),
        })
        .collect();
    let edges: Vec<RenderEdge<'_>> = design
        .edges
        .iter()
        .map(|edge| RenderEdge {
            source: edge.source.as_str(),
            target: edge.target.as_str(),
            condition: edge.condition.as_deref(),
        })
        .collect();

    render(&nodes, &edges)
}

/// Renders the intermediate graph: header, node lines, then edge lines.
fn render(nodes: &[RenderNode<'_>], edges: &[RenderEdge<'_>]) -> String {
    let mut used: BTreeSet<String> = BTreeSet::new();
    let mut rendered_ids: Vec<String> = Vec::with_capacity(nodes.len());
    let mut by_original: BTreeMap<&str, String> = BTreeMap::new();

    for (index, node) in nodes.iter().enumerate() {
        let id = unique_id(&sanitize_id(node.id, index + 1), &mut used);
        by_original.entry(node.id).or_insert_with(|| id.clone());
        rendered_ids.push(id);
    }

    let mut outgoing: BTreeMap<&str, usize> = BTreeMap::new();
    for edge in edges {
        *outgoing.entry(edge.source).or_insert(0) += 1;
    }

    let mut lines: Vec<String> = Vec::with_capacity(nodes.len() + edges.len() + 1);
    lines.push(HEADER.to_string());

    for (node, id) in nodes.iter().zip(&rendered_ids) {
        let label = escape_label(node_label_source(node, id));
        lines.push(format!("  {id}[Node{id}: {label}]"));
    }

    for edge in edges {
        let source = resolve_endpoint(&by_original, edge.source);
        let target = resolve_endpoint(&by_original, edge.target);
        let branching = outgoing.get(edge.source).copied().unwrap_or(0) > 1;
        lines.push(edge_line(&source, &target, edge.condition, branching));
    }

    lines.join("\n")
}

/// Whether `value` holds no non-whitespace character.
fn is_blank(value: &str) -> bool {
    value.trim().is_empty()
}

/// The raw text a node label renders, before escaping.
///
/// The description wins; a blank description falls back to the original node id,
/// or to the rendered id when the original id is empty, so a label is never empty.
fn node_label_source<'a>(node: &RenderNode<'a>, rendered_id: &'a str) -> &'a str {
    if is_blank(node.description) {
        if node.id.is_empty() {
            rendered_id
        } else {
            node.id
        }
    } else {
        node.description
    }
}

/// Formats one edge line.
///
/// The edge is labelled `-- {label} -->` only when `branching` (the source node
/// has more than one outgoing edge) and the label is non-blank; otherwise it
/// renders as `-->`. The label is the extracted `When condition met` label when
/// the condition is such a description, and the escaped whole condition
/// otherwise.
fn edge_line(source: &str, target: &str, condition: Option<&str>, branching: bool) -> String {
    let label = condition.map(edge_label).unwrap_or_default();

    if branching && !is_blank(&label) {
        format!("  {source} -- {label} --> {target}")
    } else {
        format!("  {source} --> {target}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent_builder::designer::{
        DesignEdge, DesignNode, DesignNodeKind, WorkflowDesigner,
    };
    use crate::agent_builder::workflow_dsl::{DslEdge, DslNode};

    /// A DSL node with only the fields rendering reads.
    fn dsl_node(id: &str, node_type: &str, title: &str, description: &str) -> DslNode {
        DslNode {
            id: id.to_string(),
            node_type: node_type.to_string(),
            title: title.to_string(),
            description: description.to_string(),
            ..DslNode::default()
        }
    }

    /// A DSL edge with only the fields rendering reads.
    fn dsl_edge(source: &str, target: &str, condition: Option<&str>) -> DslEdge {
        DslEdge {
            source: source.to_string(),
            target: target.to_string(),
            condition: condition.map(str::to_string),
        }
    }

    /// A workflow document built from the given nodes and edges.
    fn graph(nodes: Vec<DslNode>, edges: Vec<DslEdge>) -> WorkflowDsl {
        WorkflowDsl {
            name: "demo".to_string(),
            description: "a demo workflow".to_string(),
            nodes,
            edges,
        }
    }

    /// A design node with only the fields rendering reads.
    fn design_node(id: &str, kind: DesignNodeKind, description: &str) -> DesignNode {
        DesignNode {
            id: id.to_string(),
            kind,
            title: format!("title of {id}"),
            description: description.to_string(),
            ..DesignNode::default()
        }
    }

    /// A design edge with only the fields rendering reads.
    fn design_edge(source: &str, target: &str, condition: Option<&str>) -> DesignEdge {
        DesignEdge {
            source: source.to_string(),
            target: target.to_string(),
            condition: condition.map(str::to_string),
        }
    }

    #[test]
    fn full_diagram_is_byte_exact() {
        let dsl = graph(
            vec![
                dsl_node("start", "start", "Begin", "Begin"),
                dsl_node("decide", "branch", "Decide", "Decide"),
                dsl_node("yes", "llm", "Yes path", "Yes path"),
                dsl_node("no", "llm", "No path", "No path"),
                dsl_node("end", "end", "Finish", "Finish"),
            ],
            vec![
                dsl_edge("start", "decide", None),
                dsl_edge("decide", "yes", Some("score > 0.5")),
                dsl_edge("decide", "no", Some("otherwise")),
                dsl_edge("yes", "end", None),
                dsl_edge("no", "end", None),
            ],
        );

        assert_eq!(
            render_mermaid(&dsl),
            "\
graph TD
  start[Nodestart: Begin]
  decide[Nodedecide: Decide]
  yes[Nodeyes: Yes path]
  no[Nodeno: No path]
  end[Nodeend: Finish]
  start --> decide
  decide -- score > 0.5 --> yes
  decide -- otherwise --> no
  yes --> end
  no --> end"
        );
    }

    #[test]
    fn single_outgoing_condition_is_not_labelled() {
        let dsl = graph(
            vec![
                dsl_node("a", "start", "A", "A"),
                dsl_node("b", "llm", "B", "B"),
                dsl_node("c", "end", "C", "C"),
            ],
            vec![
                dsl_edge("a", "b", Some("only path")),
                dsl_edge("b", "c", None),
            ],
        );

        assert_eq!(
            render_mermaid(&dsl),
            "\
graph TD
  a[Nodea: A]
  b[Nodeb: B]
  c[Nodec: C]
  a --> b
  b --> c"
        );
    }

    #[test]
    fn branching_source_leaves_conditionless_edges_unlabelled() {
        let dsl = graph(
            vec![
                dsl_node("decide", "branch", "Decide", "Decide"),
                dsl_node("yes", "llm", "Yes path", "Yes path"),
                dsl_node("no", "llm", "No path", "No path"),
            ],
            vec![
                dsl_edge("decide", "yes", Some("yes")),
                dsl_edge("decide", "no", None),
            ],
        );

        assert_eq!(
            render_mermaid(&dsl),
            "\
graph TD
  decide[Nodedecide: Decide]
  yes[Nodeyes: Yes path]
  no[Nodeno: No path]
  decide -- yes --> yes
  decide --> no"
        );
    }

    #[test]
    fn blank_condition_on_a_branching_source_is_unlabelled() {
        let dsl = graph(
            vec![
                dsl_node("a", "branch", "A", "A"),
                dsl_node("b", "end", "B", "B"),
                dsl_node("c", "end", "C", "C"),
            ],
            vec![
                dsl_edge("a", "b", Some("   ")),
                dsl_edge("a", "c", Some("")),
            ],
        );

        assert_eq!(
            render_mermaid(&dsl),
            "\
graph TD
  a[Nodea: A]
  b[Nodeb: B]
  c[Nodec: C]
  a --> b
  a --> c"
        );
    }

    #[test]
    fn node_label_prefers_description_over_title() {
        let dsl = graph(
            vec![
                dsl_node("a", "start", "Title A", "Description A"),
                dsl_node("b", "llm", "Title B", ""),
                dsl_node("c", "end", "", "   "),
            ],
            Vec::new(),
        );

        assert_eq!(
            render_mermaid(&dsl),
            "\
graph TD
  a[Nodea: Description A]
  b[Nodeb: b]
  c[Nodec: c]"
        );
    }

    #[test]
    fn labels_escape_only_quotes_and_backticks() {
        let dsl = graph(
            vec![
                dsl_node("quote", "llm", "", "say \"hi\" and `bye`"),
                dsl_node(
                    "keep",
                    "llm",
                    "",
                    "keep [brackets] {braces} | pipes <angles>",
                ),
            ],
            Vec::new(),
        );

        assert_eq!(
            render_mermaid(&dsl),
            "\
graph TD
  quote[Nodequote: say 'hi' and 'bye']
  keep[Nodekeep: keep [brackets] {braces} | pipes <angles>]"
        );
    }

    #[test]
    fn edge_labels_escape_quotes_and_backticks() {
        let dsl = graph(
            vec![
                dsl_node("a", "branch", "A", "A"),
                dsl_node("b", "end", "B", "B"),
                dsl_node("c", "end", "C", "C"),
            ],
            vec![
                dsl_edge("a", "b", Some("`tick` and \"quote\"")),
                dsl_edge("a", "c", Some("score > 0.5")),
            ],
        );

        assert_eq!(
            render_mermaid(&dsl),
            "\
graph TD
  a[Nodea: A]
  b[Nodeb: B]
  c[Nodec: C]
  a -- 'tick' and 'quote' --> b
  a -- score > 0.5 --> c"
        );
    }

    #[test]
    fn header_is_graph_td_without_a_trailing_newline() {
        let dsl = graph(
            vec![
                dsl_node("a", "start", "A", "A"),
                dsl_node("b", "end", "B", "B"),
            ],
            vec![dsl_edge("a", "b", None)],
        );

        let rendered = render_mermaid(&dsl);
        assert!(rendered.starts_with("graph TD\n"), "{rendered}");
        assert!(!rendered.contains("flowchart"), "{rendered}");
        assert!(!rendered.ends_with('\n'), "{rendered}");
        assert_eq!(rendered.lines().count(), 1 + 2 + 1);
    }

    #[test]
    fn one_line_per_node_then_one_line_per_edge_in_document_order() {
        let dsl = graph(
            vec![
                dsl_node("third", "llm", "", "Third"),
                dsl_node("first", "start", "", "First"),
                dsl_node("second", "end", "", "Second"),
            ],
            vec![
                dsl_edge("third", "first", None),
                dsl_edge("first", "second", None),
            ],
        );

        let rendered = render_mermaid(&dsl);
        assert_eq!(
            rendered.lines().collect::<Vec<_>>(),
            [
                "graph TD",
                "  third[Nodethird: Third]",
                "  first[Nodefirst: First]",
                "  second[Nodesecond: Second]",
                "  third --> first",
                "  first --> second",
            ]
        );
        assert_eq!(rendered, render_mermaid(&dsl));
    }

    #[test]
    fn empty_document_renders_the_header_only() {
        assert_eq!(render_mermaid(&graph(Vec::new(), Vec::new())), "graph TD");
    }

    #[test]
    fn nodes_without_edges_still_render() {
        let dsl = graph(vec![dsl_node("lonely", "llm", "", "Lonely")], Vec::new());
        assert_eq!(
            render_mermaid(&dsl),
            "graph TD\n  lonely[Nodelonely: Lonely]"
        );
    }

    #[test]
    fn ids_are_sanitized_and_deduplicated() {
        let dsl = graph(
            vec![
                dsl_node("my node 1", "llm", "", "Mine"),
                dsl_node("1", "llm", "", "Digit"),
                dsl_node("中文", "llm", "", "Cjk"),
                dsl_node("a b", "llm", "", "First"),
                dsl_node("a-b", "llm", "", "Second"),
                dsl_node("", "llm", "", "Empty"),
            ],
            vec![
                dsl_edge("my node 1", "1", None),
                dsl_edge("a b", "a-b", None),
                dsl_edge("ghost node", "中文", None),
            ],
        );

        assert_eq!(
            render_mermaid(&dsl),
            "\
graph TD
  my_node_1[Nodemy_node_1: Mine]
  n_1[Noden_1: Digit]
  __[Node__: Cjk]
  a_b[Nodea_b: First]
  a_b_2[Nodea_b_2: Second]
  node_6[Nodenode_6: Empty]
  my_node_1 --> n_1
  a_b --> a_b_2
  ghost_node --> __"
        );
    }

    #[test]
    fn sanitized_ids_never_collide() {
        let dsl = graph(
            vec![
                dsl_node("a", "llm", "", "One"),
                dsl_node("a", "llm", "", "Two"),
                dsl_node("a_2", "llm", "", "Three"),
            ],
            Vec::new(),
        );

        assert_eq!(
            render_mermaid(&dsl),
            "\
graph TD
  a[Nodea: One]
  a_2[Nodea_2: Two]
  a_2_2[Nodea_2_2: Three]"
        );
    }

    #[test]
    fn dangling_endpoints_still_render_sanitized() {
        let dsl = graph(
            vec![dsl_node("start", "start", "", "Start")],
            vec![
                dsl_edge("start", "missing node", None),
                dsl_edge("gone", "start", Some("ignored")),
            ],
        );

        assert_eq!(
            render_mermaid(&dsl),
            "\
graph TD
  start[Nodestart: Start]
  start --> missing_node
  gone --> start"
        );
    }

    #[test]
    fn edge_labels_count_raw_source_ids() {
        let dsl = graph(
            vec![
                dsl_node("a", "branch", "", "A"),
                dsl_node("b", "end", "", "B"),
                dsl_node("c", "end", "", "C"),
            ],
            vec![
                dsl_edge("a", "b", Some("left")),
                dsl_edge("a", "c", Some("right")),
            ],
        );

        assert_eq!(
            render_mermaid(&dsl),
            "\
graph TD
  a[Nodea: A]
  b[Nodeb: B]
  c[Nodec: C]
  a -- left --> b
  a -- right --> c"
        );
    }

    #[test]
    fn render_design_matches_the_equivalent_dsl() {
        let design = WorkflowDesign {
            name: "demo".to_string(),
            description: "a demo design".to_string(),
            nodes: vec![
                design_node("start", DesignNodeKind::Start, "Begin"),
                design_node("decide", DesignNodeKind::Branch, "Decide"),
                design_node("yes", DesignNodeKind::Llm, "Yes path"),
                design_node("no", DesignNodeKind::Tool, "No path"),
                design_node("end", DesignNodeKind::End, "Finish"),
            ],
            edges: vec![
                design_edge("start", "decide", None),
                design_edge("decide", "yes", Some("ok")),
                design_edge("decide", "no", Some("fail")),
                design_edge("yes", "end", None),
            ],
        };

        let rendered = render_design(&design);
        assert_eq!(
            rendered,
            "\
graph TD
  start[Nodestart: Begin]
  decide[Nodedecide: Decide]
  yes[Nodeyes: Yes path]
  no[Nodeno: No path]
  end[Nodeend: Finish]
  start --> decide
  decide -- ok --> yes
  decide -- fail --> no
  yes --> end"
        );

        let dsl = design.to_workflow_dsl().expect("the design converts");
        assert_eq!(rendered, render_mermaid(&dsl));
    }

    #[test]
    fn render_design_renders_the_offline_designer_output() {
        let design = WorkflowDesigner::new()
            .design_offline("Collect the order. Then confirm it.")
            .expect("the offline designer accepts a non-blank query");

        let rendered = render_design(&design);
        assert!(rendered.starts_with("graph TD\n"), "{rendered}");
        assert!(!rendered.contains("flowchart"), "{rendered}");
        assert!(rendered.contains("start[Nodestart: "), "{rendered}");
        assert!(rendered.contains("llm_1[Nodellm_1: "), "{rendered}");
        assert!(rendered.contains("end[Nodeend: "), "{rendered}");

        let dsl = design.to_workflow_dsl().expect("the design converts");
        assert_eq!(rendered, render_mermaid(&dsl));
        assert_eq!(rendered, render_design(&design));
    }

    #[test]
    fn when_condition_labels_replace_the_whole_condition() {
        let dsl = graph(
            vec![
                dsl_node("decide", "branch", "", "Decide"),
                dsl_node("yes", "end", "", "Yes path"),
                dsl_node("no", "end", "", "No path"),
            ],
            vec![
                dsl_edge("decide", "yes", Some("When condition met[order valid]")),
                dsl_edge("decide", "no", Some("otherwise")),
            ],
        );

        assert_eq!(
            render_mermaid(&dsl),
            "\
graph TD
  decide[Nodedecide: Decide]
  yes[Nodeyes: Yes path]
  no[Nodeno: No path]
  decide -- order valid --> yes
  decide -- otherwise --> no"
        );
    }

    #[test]
    fn when_conditions_without_brackets_yield_the_remainder() {
        let dsl = graph(
            vec![
                dsl_node("decide", "branch", "", "Decide"),
                dsl_node("yes", "end", "", "Yes path"),
                dsl_node("no", "end", "", "No path"),
            ],
            vec![
                dsl_edge("decide", "yes", Some("When condition met: order valid")),
                dsl_edge("decide", "no", Some("When condition met score > 0.5")),
            ],
        );

        assert_eq!(
            render_mermaid(&dsl),
            "\
graph TD
  decide[Nodedecide: Decide]
  yes[Nodeyes: Yes path]
  no[Nodeno: No path]
  decide -- order valid --> yes
  decide -- score > 0.5 --> no"
        );
    }

    #[test]
    fn when_condition_on_a_single_successor_stays_unlabelled() {
        let dsl = graph(
            vec![
                dsl_node("a", "start", "", "A"),
                dsl_node("b", "end", "", "B"),
            ],
            vec![dsl_edge("a", "b", Some("When condition met[order valid]"))],
        );

        assert_eq!(
            render_mermaid(&dsl),
            "\
graph TD
  a[Nodea: A]
  b[Nodeb: B]
  a --> b"
        );
    }

    #[test]
    fn extracted_when_labels_escape_quotes_and_backticks() {
        let dsl = graph(
            vec![
                dsl_node("decide", "branch", "", "Decide"),
                dsl_node("yes", "end", "", "Yes path"),
                dsl_node("no", "end", "", "No path"),
            ],
            vec![
                dsl_edge(
                    "decide",
                    "yes",
                    Some("When condition met[say \"hi\" and `bye`]"),
                ),
                dsl_edge("decide", "no", Some("otherwise")),
            ],
        );

        assert_eq!(
            render_mermaid(&dsl),
            "\
graph TD
  decide[Nodedecide: Decide]
  yes[Nodeyes: Yes path]
  no[Nodeno: No path]
  decide -- say 'hi' and 'bye' --> yes
  decide -- otherwise --> no"
        );
    }

    #[test]
    fn bare_when_prefix_falls_back_to_the_escaped_condition() {
        let dsl = graph(
            vec![
                dsl_node("decide", "branch", "", "Decide"),
                dsl_node("yes", "end", "", "Yes path"),
                dsl_node("no", "end", "", "No path"),
            ],
            vec![
                dsl_edge("decide", "yes", Some("When condition met")),
                dsl_edge("decide", "no", Some("otherwise")),
            ],
        );

        assert_eq!(
            render_mermaid(&dsl),
            "\
graph TD
  decide[Nodedecide: Decide]
  yes[Nodeyes: Yes path]
  no[Nodeno: No path]
  decide -- When condition met --> yes
  decide -- otherwise --> no"
        );
    }
}
