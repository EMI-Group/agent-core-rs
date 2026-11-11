//! Mermaid rendering of a graph spec.
//!
//! Python emits Mermaid only (`visualization/Drawable.to_mermaid`); DOT and the
//! network-rendered PNG/SVG variants are intentionally omitted.
//!
//! The output is a `flowchart` with one statement per node and one per edge:
//! entry and exit nodes get round edges, every other node a square box.
//! Conditional edges render as dashed links labeled with their source node id.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::spec::{EdgeKind, GraphSpec, DYNAMIC_TARGET};

/// The flow direction of the rendered flowchart.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MermaidDirection {
    /// Top to bottom (`TB`).
    #[default]
    TopDown,
    /// Left to right (`LR`).
    LeftRight,
}

impl MermaidDirection {
    /// The Mermaid direction keyword.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::TopDown => "TB",
            Self::LeftRight => "LR",
        }
    }
}

/// Options for [`render_mermaid`].
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct MermaidOptions {
    /// Front-matter title; omitted when `None`.
    pub title: Option<String>,
    /// Flow direction; defaults to top-down.
    pub direction: MermaidDirection,
}

/// Renders `spec` as a Mermaid flowchart.
///
/// Nodes are allocated `node_1`, `node_2`, … in declaration order. Nodes that
/// are not part of the spec, and dynamic (`*`) conditional targets, are skipped
/// so the output is always valid Mermaid.
#[must_use]
pub fn render_mermaid(spec: &GraphSpec, options: &MermaidOptions) -> String {
    let mut output = String::new();
    if let Some(title) = &options.title {
        output.push_str("---\n");
        output.push_str("title: ");
        output.push_str(&sanitize_ascii(title, true));
        output.push('\n');
        output.push_str("---\n");
    }
    output.push_str("flowchart ");
    output.push_str(options.direction.as_str());
    output.push('\n');

    let entries: BTreeSet<&str> = spec.entry.iter().map(String::as_str).collect();
    let exits: BTreeSet<&str> = spec.exit.iter().map(String::as_str).collect();
    let mut ids: Vec<(&str, usize)> = Vec::with_capacity(spec.nodes.len());
    for (index, node) in spec.nodes.iter().enumerate() {
        let number = index + 1;
        let label = escape_label(&node.id);
        let is_terminal = entries.contains(node.id.as_str()) || exits.contains(node.id.as_str());
        let shape = if is_terminal {
            format!("node_{number}(\"{label}\")")
        } else {
            format!("node_{number}[\"{label}\"]")
        };
        push_line(&mut output, &shape);
        ids.push((node.id.as_str(), number));
    }

    for edge in &spec.edges {
        let Some(source) = edge.sources.first() else {
            continue;
        };
        if edge.target == DYNAMIC_TARGET {
            continue;
        }
        let Some(&(_, source_number)) = ids.iter().find(|(id, _)| id == source) else {
            continue;
        };
        let Some(&(_, target_number)) = ids.iter().find(|(id, _)| *id == edge.target) else {
            continue;
        };
        match edge.kind {
            EdgeKind::Conditional => {
                let label = sanitize_ascii(source, false);
                push_line(
                    &mut output,
                    &format!("node_{source_number} -. \"{label}\" .-> node_{target_number}"),
                );
            }
            EdgeKind::Static | EdgeKind::Barrier => {
                push_line(
                    &mut output,
                    &format!("node_{source_number} --> node_{target_number}"),
                );
            }
        }
    }
    output
}

/// Appends an indented statement line.
fn push_line(output: &mut String, statement: &str) {
    output.push_str("    ");
    output.push_str(statement);
    output.push('\n');
}

/// Escapes a node label for a double-quoted Mermaid string.
fn escape_label(label: &str) -> String {
    label.replace('\\', "\\\\").replace('"', "#quot;")
}

/// Keeps only characters that are safe inside a Mermaid label or title.
fn sanitize_ascii(label: &str, keep_colon: bool) -> String {
    label
        .chars()
        .map(|character| {
            let allowed = character.is_ascii_alphanumeric()
                || matches!(character, '_' | '-' | ' ' | '.' | '[' | ']' | '(' | ')')
                || (keep_colon && character == ':');
            if allowed {
                character
            } else {
                '_'
            }
        })
        .collect()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]
mod tests {
    use super::*;
    use crate::spec::{EdgeSpec, NodeSpec};
    use std::collections::BTreeMap;

    fn spec() -> GraphSpec {
        GraphSpec {
            graph_id: "g".to_string(),
            entry: vec!["start".to_string()],
            exit: vec!["end".to_string()],
            nodes: ["start", "middle", "end"]
                .into_iter()
                .map(|id| NodeSpec {
                    id: id.to_string(),
                    ..NodeSpec::default()
                })
                .collect(),
            edges: vec![
                EdgeSpec {
                    sources: vec!["start".to_string()],
                    target: "middle".to_string(),
                    kind: EdgeKind::Static,
                },
                EdgeSpec {
                    sources: vec!["middle".to_string()],
                    target: "end".to_string(),
                    kind: EdgeKind::Conditional,
                },
            ],
            metadata: BTreeMap::new(),
        }
    }

    #[test]
    fn renders_header_nodes_and_edges() {
        let rendered = render_mermaid(&spec(), &MermaidOptions::default());
        assert!(rendered.starts_with("flowchart TB\n"));
        assert!(rendered.contains("node_1(\"start\")"));
        assert!(rendered.contains("node_2[\"middle\"]"));
        assert!(rendered.contains("node_3(\"end\")"));
        assert!(rendered.contains("node_1 --> node_2"));
        assert!(rendered.contains("node_2 -. \"middle\" .-> node_3"));
    }

    #[test]
    fn supports_title_and_direction() {
        let options = MermaidOptions {
            title: Some("My graph".to_string()),
            direction: MermaidDirection::LeftRight,
        };
        let rendered = render_mermaid(&spec(), &options);
        assert!(rendered.starts_with("---\ntitle: My graph\n---\nflowchart LR\n"));
        assert_eq!(MermaidDirection::TopDown.as_str(), "TB");
    }

    #[test]
    fn skips_dynamic_and_unknown_edges() {
        let mut graph = spec();
        graph.edges.push(EdgeSpec {
            sources: vec!["middle".to_string()],
            target: DYNAMIC_TARGET.to_string(),
            kind: EdgeKind::Conditional,
        });
        graph.edges.push(EdgeSpec {
            sources: vec!["ghost".to_string()],
            target: "end".to_string(),
            kind: EdgeKind::Static,
        });
        let rendered = render_mermaid(&graph, &MermaidOptions::default());
        assert!(!rendered.contains('*'));
        assert!(!rendered.contains("ghost"));
    }

    #[test]
    fn escapes_labels_and_sanitizes_titles() {
        let mut graph = spec();
        graph.nodes[1].id = "a\"b\\c".to_string();
        let rendered = render_mermaid(&graph, &MermaidOptions::default());
        assert!(rendered.contains("node_2[\"a#quot;b\\\\c\"]"));

        let options = MermaidOptions {
            title: Some("héllo\nworld".to_string()),
            ..MermaidOptions::default()
        };
        let rendered = render_mermaid(&graph, &options);
        assert!(rendered.contains("title: h_llo_world"));
    }

    #[test]
    fn barrier_edges_render_as_static_links() {
        let mut graph = spec();
        graph.edges = vec![EdgeSpec {
            sources: vec!["start".to_string(), "middle".to_string()],
            target: "end".to_string(),
            kind: EdgeKind::Barrier,
        }];
        let rendered = render_mermaid(&graph, &MermaidOptions::default());
        assert!(rendered.contains("node_1 --> node_3"));
    }

    #[test]
    fn options_round_trip() {
        let options = MermaidOptions {
            title: Some("t".to_string()),
            direction: MermaidDirection::LeftRight,
        };
        let decoded: MermaidOptions =
            serde_json::from_str(&serde_json::to_string(&options).unwrap()).unwrap();
        assert_eq!(decoded, options);
        assert_eq!(
            serde_json::to_value(MermaidDirection::TopDown).unwrap(),
            serde_json::json!("top_down")
        );
    }
}
