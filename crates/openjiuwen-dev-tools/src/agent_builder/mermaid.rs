//! Mermaid rendering for the workflow DSL and the typed workflow design.
//!
//! Ports
//! `openjiuwen.dev_tools.agent_builder.builders.workflow.dl_transformer.simpleir_to_mermaid`,
//! adapted to the typed graphs of this crate instead of the Python `SimpleIR`
//! dictionaries. Python walks a node list, derives edges from `next` and
//! `conditions`, and emits one rectangle per node; this module renders the
//! [`WorkflowDsl`] and the [`WorkflowDesign`] as a
//! Mermaid `flowchart` with a shape per node kind, escaped labels, and
//! sanitized ids.
//!
//! Rendering is deterministic and follows document order: the header line, then
//! one two-space-indented line per node, then one line per edge. Node shapes:
//!
//! | kind | shape | syntax |
//! | --- | --- | --- |
//! | start, end | stadium | `([label])` |
//! | llm | rectangle | `[label]` |
//! | branch | rhombus | `{label}` |
//! | tool | hexagon | `{{label}}` |
//! | anything else | rectangle | `[label]` |
//!
//! Edges render as `source --> target` and gain a `-->|condition|` label when
//! the edge carries a non-blank condition.
//!
//! Deviations from Python:
//!
//! - The header is `flowchart TD` (the current Mermaid keyword); Python emits
//!   the legacy `graph TD`. Both parse to the same top-down diagram.
//! - Shapes follow the node kind instead of Python's uniform rectangles with a
//!   `Node<id>:` label prefix.
//! - Labels are escaped and ids sanitized, so quotes, brackets, pipes,
//!   newlines, and non-ASCII ids still produce a parseable diagram.
//! - Edge labels come from the typed condition and use the `-->|label|` form;
//!   Python labels only the outgoing edges of a multi-successor node and uses
//!   `-- label -->`. Edge labels keep `<` and `>` so comparisons such as
//!   `score > 0.5` read naturally, while node labels drop them together with the
//!   shape delimiters.

use std::collections::{BTreeMap, BTreeSet};

use super::designer::{DesignNodeKind, WorkflowDesign};
use super::workflow_dsl::WorkflowDsl;

/// The Mermaid header line of every rendered diagram.
const HEADER: &str = "flowchart TD";

/// The node shapes a rendered diagram can use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NodeShape {
    /// A rounded/stadium node (`([label])`), used for start and end.
    Stadium,
    /// A rectangular node (`[label]`), used for llm and unknown kinds.
    Rectangle,
    /// A rhombus node (`{label}`), used for branches.
    Rhombus,
    /// A hexagonal node (`{{label}}`), used for tools.
    Hexagon,
}

impl NodeShape {
    /// The opening delimiter of the shape.
    const fn open(self) -> &'static str {
        match self {
            Self::Stadium => "([",
            Self::Rectangle => "[",
            Self::Rhombus => "{",
            Self::Hexagon => "{{",
        }
    }

    /// The closing delimiter of the shape.
    const fn close(self) -> &'static str {
        match self {
            Self::Stadium => "])",
            Self::Rectangle => "]",
            Self::Rhombus => "}",
            Self::Hexagon => "}}",
        }
    }
}

/// A node of the intermediate graph both public entry points render.
struct RenderNode<'a> {
    /// The original node id, used for the id map and as the label fallback.
    id: &'a str,
    /// The Mermaid shape the node renders with.
    shape: NodeShape,
    /// The node title, preferred as the label.
    title: &'a str,
    /// The node description, the label fallback after the title.
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

/// Renders a workflow DSL document as a Mermaid `flowchart TD` diagram.
///
/// Nodes appear in document order, each with a shape derived from its `type`
/// (case-insensitively: `start`/`end`, `llm`, `branch`, `tool`, anything else
/// as a rectangle). Edges follow in document order, labelled with their
/// condition when it is non-blank. Ids are sanitized and deduplicated, labels
/// are escaped, so the result is always parseable; rendering the same document
/// twice yields byte-identical output.
#[must_use]
pub fn render_mermaid(dsl: &WorkflowDsl) -> String {
    let nodes: Vec<RenderNode<'_>> = dsl
        .nodes
        .iter()
        .map(|node| RenderNode {
            id: node.id.as_str(),
            shape: dsl_shape(&node.node_type),
            title: node.title.as_str(),
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

/// Renders a workflow design as a Mermaid `flowchart TD` diagram.
///
/// Applies the same shape, escaping, and sanitizing rules as [`render_mermaid`]:
/// [`DesignNodeKind::Start`] and [`DesignNodeKind::End`] render as stadiums,
/// [`DesignNodeKind::Llm`] as rectangles, [`DesignNodeKind::Branch`] as
/// rhombuses, and [`DesignNodeKind::Tool`] as hexagons. A design renders
/// byte-identically to the [`WorkflowDsl`] produced by
/// [`WorkflowDesign::to_workflow_dsl`].
#[must_use]
pub fn render_design(design: &WorkflowDesign) -> String {
    let nodes: Vec<RenderNode<'_>> = design
        .nodes
        .iter()
        .map(|node| RenderNode {
            id: node.id.as_str(),
            shape: design_shape(node.kind),
            title: node.title.as_str(),
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

    let mut lines: Vec<String> = Vec::with_capacity(nodes.len() + edges.len() + 1);
    lines.push(HEADER.to_string());

    for (node, id) in nodes.iter().zip(&rendered_ids) {
        let label = node_label(node, id);
        lines.push(format!(
            "  {id}{}{label}{}",
            node.shape.open(),
            node.shape.close()
        ));
    }

    for edge in edges {
        let source = resolve_endpoint(&by_original, edge.source);
        let target = resolve_endpoint(&by_original, edge.target);
        lines.push(edge_line(&source, &target, edge.condition));
    }

    lines.join("\n")
}

/// Maps a DSL node `type` to its shape.
///
/// Matching is case-insensitive; only `start`, `end`, `llm`, `branch`, and
/// `tool` are recognized, and every other value (including an empty one)
/// renders as a rectangle.
fn dsl_shape(node_type: &str) -> NodeShape {
    match node_type.trim().to_ascii_lowercase().as_str() {
        "start" | "end" => NodeShape::Stadium,
        "branch" => NodeShape::Rhombus,
        "tool" => NodeShape::Hexagon,
        _ => NodeShape::Rectangle,
    }
}

/// Maps a typed design node kind to its shape.
const fn design_shape(kind: DesignNodeKind) -> NodeShape {
    match kind {
        DesignNodeKind::Start | DesignNodeKind::End => NodeShape::Stadium,
        DesignNodeKind::Llm => NodeShape::Rectangle,
        DesignNodeKind::Branch => NodeShape::Rhombus,
        DesignNodeKind::Tool => NodeShape::Hexagon,
    }
}

/// Sanitizes a node id into a Mermaid-safe identifier.
///
/// ASCII alphanumerics and `_` are kept; every other character (CJK, `-`,
/// spaces, punctuation) becomes `_` one by one, without collapsing. An empty
/// result falls back to `node_<index>` (1-based document position), and a result
/// starting with a digit is prefixed with `n_`.
fn sanitize_id(raw: &str, index: usize) -> String {
    let mut sanitized = String::with_capacity(raw.len().max(1));
    for character in raw.chars() {
        if character.is_ascii_alphanumeric() || character == '_' {
            sanitized.push(character);
        } else {
            sanitized.push('_');
        }
    }

    if sanitized.is_empty() {
        return format!("node_{index}");
    }
    if sanitized.starts_with(|character: char| character.is_ascii_digit()) {
        return format!("n_{sanitized}");
    }
    sanitized
}

/// Reserves `base` in `used`, appending `_2`, `_3`, ... until it is unique.
fn unique_id(base: &str, used: &mut BTreeSet<String>) -> String {
    if used.insert(base.to_string()) {
        return base.to_string();
    }

    let mut suffix = 2_u32;
    loop {
        let candidate = format!("{base}_{suffix}");
        if used.insert(candidate.clone()) {
            return candidate;
        }
        suffix += 1;
    }
}

/// Resolves an edge endpoint to the id it is rendered with.
///
/// Known endpoints use their deduplicated rendered id; a dangling endpoint keeps
/// its sanitized form so the edge still points somewhere parseable.
fn resolve_endpoint(ids: &BTreeMap<&str, String>, endpoint: &str) -> String {
    ids.get(endpoint)
        .cloned()
        .unwrap_or_else(|| sanitize_id(endpoint, 0))
}

/// Whether `value` holds no non-whitespace character.
fn is_blank(value: &str) -> bool {
    value.trim().is_empty()
}

/// Picks and escapes the label of a node.
///
/// The title wins, then the description, then the original id (or the rendered
/// `node_<index>` for an empty id). The escaped text falls back to the escaped
/// original id, and finally to the rendered id, so a label is never empty.
fn node_label(node: &RenderNode<'_>, rendered_id: &str) -> String {
    let fallback: &str = if node.id.is_empty() {
        rendered_id
    } else {
        node.id
    };
    let label = if is_blank(node.title) {
        if is_blank(node.description) {
            fallback
        } else {
            node.description
        }
    } else {
        node.title
    };

    let escaped = escape_label(label);
    if escaped.is_empty() {
        let escaped_id = escape_label(fallback);
        if escaped_id.is_empty() {
            rendered_id.to_string()
        } else {
            escaped_id
        }
    } else {
        escaped
    }
}

/// Escapes free text into a label Mermaid can parse inside any node shape.
///
/// `"` and `` ` `` become `'`; `[`, `]`, `{`, `}`, `(`, `)`, `|`, `<`, `>`, and
/// newlines/tabs become spaces. Runs of whitespace collapse to one space and the
/// result is trimmed, so the label never spans lines or closes its shape early.
fn escape_label(raw: &str) -> String {
    escape(raw, true)
}

/// Escapes a condition into an edge label.
///
/// Applies the same rules as [`escape_label`] except that `<` and `>` survive:
/// an edge label is delimited by pipes alone, and comparisons such as
/// `score > 0.5` are the usual condition text.
fn escape_edge_label(raw: &str) -> String {
    escape(raw, false)
}

/// Replaces the characters that would break a label and collapses whitespace.
///
/// `replace_angles` also maps `<` and `>` to spaces; it is set for node labels,
/// whose shape delimiters the engine parses together with the label.
fn escape(raw: &str, replace_angles: bool) -> String {
    let mut replaced = String::with_capacity(raw.len());
    for character in raw.chars() {
        match character {
            '"' | '`' => replaced.push('\''),
            '[' | ']' | '{' | '}' | '(' | ')' | '|' | '\n' | '\r' | '\t' => replaced.push(' '),
            '<' | '>' if replace_angles => replaced.push(' '),
            other => replaced.push(other),
        }
    }

    replaced.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Formats one edge line, labelled when the escaped condition is non-blank.
fn edge_line(source: &str, target: &str, condition: Option<&str>) -> String {
    let label = match condition {
        Some(condition) => escape_edge_label(condition),
        None => String::new(),
    };

    if label.is_empty() {
        format!("  {source} --> {target}")
    } else {
        format!("  {source} -->|{label}| {target}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent_builder::designer::{DesignEdge, DesignNode, WorkflowDesigner};
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
    fn design_node(id: &str, kind: DesignNodeKind, title: &str) -> DesignNode {
        DesignNode {
            id: id.to_string(),
            kind,
            title: title.to_string(),
            ..DesignNode::default()
        }
    }

    #[test]
    fn node_shapes_follow_the_node_kind() {
        let dsl = graph(
            vec![
                dsl_node("start", "start", "Begin", ""),
                dsl_node("generate", "llm", "Generate", ""),
                dsl_node("decide", "branch", "Decide", ""),
                dsl_node("lookup", "tool", "Lookup", ""),
                dsl_node("end", "end", "Finish", ""),
                dsl_node("legacy", "questioner", "Legacy", ""),
            ],
            Vec::new(),
        );

        let rendered = render_mermaid(&dsl);
        let lines: Vec<&str> = rendered.lines().collect();

        assert_eq!(
            lines,
            [
                "flowchart TD",
                "  start([Begin])",
                "  generate[Generate]",
                "  decide{Decide}",
                "  lookup{{Lookup}}",
                "  end([Finish])",
                "  legacy[Legacy]",
            ]
        );
    }

    #[test]
    fn node_type_matching_is_case_insensitive() {
        let dsl = graph(
            vec![
                dsl_node("a", "START", "Upper start", ""),
                dsl_node("b", "Llm", "Mixed llm", ""),
                dsl_node("c", " Branch ", "Padded branch", ""),
                dsl_node("d", "Tool", "Mixed tool", ""),
                dsl_node("e", "End", "Mixed end", ""),
                dsl_node("f", "", "No type", ""),
            ],
            Vec::new(),
        );

        let rendered = render_mermaid(&dsl);
        let lines: Vec<&str> = rendered.lines().collect();

        assert_eq!(lines[1], "  a([Upper start])");
        assert_eq!(lines[2], "  b[Mixed llm]");
        assert_eq!(lines[3], "  c{Padded branch}");
        assert_eq!(lines[4], "  d{{Mixed tool}}");
        assert_eq!(lines[5], "  e([Mixed end])");
        assert_eq!(lines[6], "  f[No type]");
    }

    #[test]
    fn edges_render_conditions_as_labels() {
        let dsl = graph(
            vec![
                dsl_node("start", "start", "Begin", ""),
                dsl_node("decide", "branch", "Decide", ""),
                dsl_node("approve", "llm", "Approve", ""),
                dsl_node("reject", "llm", "Reject", ""),
            ],
            vec![
                dsl_edge("start", "decide", None),
                dsl_edge("decide", "approve", Some("score > 0.5")),
                dsl_edge("decide", "reject", Some("   ")),
            ],
        );

        let rendered = render_mermaid(&dsl);
        let lines: Vec<&str> = rendered.lines().collect();

        assert_eq!(lines[5], "  start --> decide");
        assert_eq!(lines[6], "  decide -->|score > 0.5| approve");
        assert_eq!(lines[7], "  decide --> reject");
    }

    #[test]
    fn edge_conditions_are_escaped() {
        let dsl = graph(
            vec![
                dsl_node("a", "branch", "A", ""),
                dsl_node("b", "llm", "B", ""),
            ],
            vec![dsl_edge("a", "b", Some("\"quoted\" | piped\nnext"))],
        );

        let rendered = render_mermaid(&dsl);
        let lines: Vec<&str> = rendered.lines().collect();

        assert_eq!(lines[3], "  a -->|'quoted' piped next| b");
        assert!(!lines[3].contains('"'));
    }

    #[test]
    fn labels_are_escaped_and_fall_back() {
        let dsl = graph(
            vec![
                dsl_node("first", "llm", "\"quoted\" [brackets] | pipe", ""),
                dsl_node("second", "llm", "", "line one\nline two\tthree"),
                dsl_node("third", "llm", "", ""),
            ],
            Vec::new(),
        );

        let rendered = render_mermaid(&dsl);
        let lines: Vec<&str> = rendered.lines().collect();

        assert_eq!(lines.len(), 4, "every node renders on exactly one line");
        assert_eq!(lines[1], "  first['quoted' brackets pipe]");
        assert_eq!(lines[2], "  second[line one line two three]");
        assert_eq!(lines[3], "  third[third]");
        assert!(!rendered.contains('"'));
        assert!(!rendered.contains('`'));
    }

    #[test]
    fn labels_never_contain_shape_delimiters() {
        for title in [
            "a [b] {c} (d) | e <f>",
            "<<[[]]>>",
            "\"|(){}[]\"",
            "multi\nline\ndescription",
        ] {
            let dsl = graph(vec![dsl_node("node", "node", title, "")], Vec::new());
            let rendered = render_mermaid(&dsl);
            let lines: Vec<&str> = rendered.lines().collect();
            assert_eq!(lines.len(), 2, "one node line expected for {title:?}");

            let label_start = lines[1]
                .find('[')
                .expect("a rectangle opens with a bracket")
                + 1;
            let label = &lines[1][label_start..lines[1].len() - 1];
            assert!(
                label.chars().all(|character| !matches!(
                    character,
                    '[' | ']' | '{' | '}' | '(' | ')' | '|' | '<' | '>' | '\n' | '\r' | '"' | '`'
                )),
                "label leaked a delimiter: {label:?}"
            );
        }
    }

    #[test]
    fn all_delimiter_only_titles_fall_back_to_the_id() {
        let dsl = graph(vec![dsl_node("plain", "llm", "[]{}()|", "")], Vec::new());
        let rendered = render_mermaid(&dsl);
        assert_eq!(rendered, "flowchart TD\n  plain[plain]");
    }

    #[test]
    fn rendering_is_deterministic_and_follows_document_order() {
        let dsl = graph(
            vec![
                dsl_node("alpha", "start", "Alpha", ""),
                dsl_node("beta", "llm", "Beta", ""),
                dsl_node("gamma", "end", "Gamma", ""),
            ],
            vec![
                dsl_edge("alpha", "beta", None),
                dsl_edge("beta", "gamma", Some("done")),
                dsl_edge("alpha", "gamma", None),
            ],
        );

        let expected = [
            "flowchart TD",
            "  alpha([Alpha])",
            "  beta[Beta]",
            "  gamma([Gamma])",
            "  alpha --> beta",
            "  beta -->|done| gamma",
            "  alpha --> gamma",
        ]
        .join("\n");

        assert_eq!(render_mermaid(&dsl), expected);
        assert_eq!(render_mermaid(&dsl), render_mermaid(&dsl));
    }

    #[test]
    fn every_node_and_edge_is_rendered_exactly_once() {
        let dsl = graph(
            vec![
                dsl_node("start", "start", "Begin", ""),
                dsl_node("generate", "llm", "Generate", ""),
                dsl_node("decide", "branch", "Decide", ""),
                dsl_node("end", "end", "Finish", ""),
            ],
            vec![
                dsl_edge("start", "generate", None),
                dsl_edge("generate", "decide", None),
                dsl_edge("decide", "end", Some("ok")),
            ],
        );

        let rendered = render_mermaid(&dsl);
        let lines: Vec<&str> = rendered.lines().collect();

        for definition in [
            "  start([Begin])",
            "  generate[Generate]",
            "  decide{Decide}",
            "  end([Finish])",
        ] {
            assert_eq!(
                lines.iter().filter(|line| **line == definition).count(),
                1,
                "node definition `{definition}` should appear once"
            );
        }
        for connection in ["  start --> generate", "  generate --> decide"] {
            assert_eq!(
                lines.iter().filter(|line| **line == connection).count(),
                1,
                "edge `{connection}` should appear once"
            );
        }
        assert_eq!(
            lines
                .iter()
                .filter(|line| **line == "  decide -->|ok| end")
                .count(),
            1
        );
        assert_eq!(
            lines
                .iter()
                .filter(|line| line.starts_with("  ") && !line.contains(" -->"))
                .count(),
            dsl.nodes.len()
        );
        assert_eq!(
            lines.iter().filter(|line| line.contains(" -->")).count(),
            dsl.edges.len()
        );
    }

    #[test]
    fn node_ids_are_sanitized_and_deduplicated() {
        let dsl = graph(
            vec![
                dsl_node("my node-1", "llm", "Mine", ""),
                dsl_node("1", "llm", "Digit", ""),
                dsl_node("开始", "llm", "Cjk", ""),
                dsl_node("a b", "llm", "First", ""),
                dsl_node("a-b", "llm", "Second", ""),
                dsl_node("", "llm", "Empty", ""),
            ],
            vec![
                dsl_edge("my node-1", "1", None),
                dsl_edge("a b", "a-b", None),
                dsl_edge("ghost node", "end", None),
            ],
        );

        let rendered = render_mermaid(&dsl);
        let node_lines: Vec<&str> = rendered
            .lines()
            .filter(|line| !line.contains(" -->"))
            .collect();
        assert_eq!(
            node_lines,
            [
                "flowchart TD",
                "  my_node_1[Mine]",
                "  n_1[Digit]",
                "  __[Cjk]",
                "  a_b[First]",
                "  a_b_2[Second]",
                "  node_6[Empty]",
            ]
        );

        let edge_lines: Vec<&str> = rendered
            .lines()
            .filter(|line| line.contains(" -->"))
            .collect();
        assert_eq!(
            edge_lines,
            [
                "  my_node_1 --> n_1",
                "  a_b --> a_b_2",
                "  ghost_node --> end",
            ]
        );
    }

    #[test]
    fn sanitized_ids_never_collide() {
        let dsl = graph(
            vec![
                dsl_node("a", "llm", "One", ""),
                dsl_node("a", "llm", "Two", ""),
                dsl_node("a_2", "llm", "Three", ""),
            ],
            Vec::new(),
        );

        let rendered = render_mermaid(&dsl);
        let lines: Vec<&str> = rendered.lines().collect();
        assert_eq!(lines[1], "  a[One]");
        assert_eq!(lines[2], "  a_2[Two]");
        assert_eq!(lines[3], "  a_2_2[Three]");
    }

    #[test]
    fn render_design_shapes_follow_the_design_kind() {
        let design = WorkflowDesign {
            name: "demo".to_string(),
            description: "a demo design".to_string(),
            nodes: vec![
                design_node("start", DesignNodeKind::Start, "Begin"),
                design_node("think", DesignNodeKind::Llm, "Think"),
                design_node("choose", DesignNodeKind::Branch, "Choose"),
                design_node("call", DesignNodeKind::Tool, "Call"),
                design_node("end", DesignNodeKind::End, "Finish"),
            ],
            edges: vec![
                DesignEdge {
                    source: "start".to_string(),
                    target: "think".to_string(),
                    condition: None,
                },
                DesignEdge {
                    source: "choose".to_string(),
                    target: "call".to_string(),
                    condition: Some("ok".to_string()),
                },
            ],
        };

        let rendered = render_design(&design);
        let lines: Vec<&str> = rendered.lines().collect();

        assert_eq!(
            lines,
            [
                "flowchart TD",
                "  start([Begin])",
                "  think[Think]",
                "  choose{Choose}",
                "  call{{Call}}",
                "  end([Finish])",
                "  start --> think",
                "  choose -->|ok| call",
            ]
        );
    }

    #[test]
    fn render_design_matches_the_equivalent_dsl() {
        let design = WorkflowDesigner::new()
            .design_offline("Collect the order. Then confirm it.")
            .expect("the offline designer accepts a non-blank query");

        let rendered = render_design(&design);
        assert!(rendered.starts_with("flowchart TD\n"), "{rendered}");
        assert!(rendered.contains("start"), "{rendered}");
        assert!(rendered.contains("llm_1"), "{rendered}");
        assert!(rendered.contains("end"), "{rendered}");

        let dsl = design.to_workflow_dsl().expect("the design converts");
        assert_eq!(rendered, render_mermaid(&dsl));
    }
}
