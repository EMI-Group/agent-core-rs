//! Offline SVG rendering for workflows.
//!
//! Ports the shape language of Python's `Workflow.draw` and `Drawable.to_mermaid_svg`: the
//! start and end components are drawn as round-edged nodes and every other component as a
//! plain node, with one arrow per edge. Rendering happens locally and deterministically
//! instead of asking the mermaid.ink web service to rasterize a Mermaid chart, so the same
//! spec always produces byte-identical SVG.
//!
//! PNG selection is typed rather than silent: Python's `output_format="png"` path sends the
//! chart to the mermaid.ink network service, so [`crate::Workflow::draw`](crate::workflow::Workflow::draw)
//! rejects [`DrawFormat::Png`] with `Error::Unsupported` while Mermaid text and SVG render
//! here.
//!
//! ```
//! use openjiuwen_workflow::card::{WorkflowCard, WorkflowSpec};
//! use openjiuwen_workflow::render::to_svg;
//!
//! let svg = to_svg(&WorkflowSpec::default(), &WorkflowCard::new("empty"), Some("Empty"));
//! assert!(svg.starts_with("<?xml"));
//! assert!(svg.contains("viewBox"));
//! ```

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use serde::{Deserialize, Serialize};

use crate::card::{WorkflowCard, WorkflowSpec};

/// Horizontal gap between the columns of a left-to-right layout, in pixels.
const COL_GAP: u32 = 80;
/// Vertical gap between the rows of a left-to-right layout, in pixels.
const ROW_GAP: u32 = 92;
/// The height of every node box, in pixels.
const NODE_H: u32 = 44;
/// The padding between the canvas edge and the outermost nodes, in pixels.
const MARGIN: u32 = 34;
/// The smallest canvas width, in pixels.
const MIN_W: u32 = 240;
/// The smallest canvas height, in pixels.
const MIN_H: u32 = 120;

/// The document format [`crate::workflow::Workflow::draw`] produces.
///
/// Replaces Python's `output_format` string argument. Mermaid text and SVG are rendered by
/// this crate; [`DrawFormat::Png`] exists so Python's third format is a typed rejection
/// instead of a silent fallback, and `draw` returns `Error::Unsupported` for it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DrawFormat {
    /// A Mermaid `flowchart` document.
    #[default]
    Mermaid,
    /// A standalone SVG document rendered by this crate.
    Svg,
    /// A raster PNG image; rejected by `draw` because it is not rendered offline.
    Png,
}

impl DrawFormat {
    /// The wire name of the format (`"mermaid"`, `"svg"`, or `"png"`).
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Mermaid => "mermaid",
            Self::Svg => "svg",
            Self::Png => "png",
        }
    }
}

/// The primary axis of the layered layout.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayoutDirection {
    /// Layers advance left to right and siblings stack downwards; the required layout.
    #[default]
    LeftRight,
    /// Layers advance top to bottom and siblings stack rightwards.
    TopDown,
}

/// Tunables for [`render_svg`].
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct SvgOptions {
    /// An optional `<title>` for the document; empty or whitespace-only titles are omitted.
    pub title: Option<String>,
    /// The layout direction; [`LayoutDirection::LeftRight`] by default.
    pub direction: LayoutDirection,
    /// Explicit end component ids; when empty the sinks of the spec are used instead.
    pub end_nodes: Vec<String>,
}

/// Renders `spec` as an SVG document, using `title` as its `<title>` when non-empty.
#[must_use]
pub fn to_svg(spec: &WorkflowSpec, card: &WorkflowCard, title: Option<&str>) -> String {
    let options = SvgOptions {
        title: title.map(str::to_owned),
        ..SvgOptions::default()
    };
    render_svg(spec, card, &options)
}

/// Renders `spec` as an SVG document with explicit [`SvgOptions`].
///
/// The output is deterministic: node order, layer order, edge order, and every coordinate
/// derive from the sorted spec maps, with no clocks and no unordered iteration. An empty
/// spec still yields a valid document (an empty canvas with the style block and the edge
/// group), and cycles or disconnected nodes lay out without panicking.
#[must_use]
pub fn render_svg(spec: &WorkflowSpec, card: &WorkflowCard, options: &SvgOptions) -> String {
    let layout = Layout::compute(spec, options);
    let label = card_label(card);
    let mut out: Vec<String> = Vec::with_capacity(64);

    out.push("<?xml version=\"1.0\" encoding=\"UTF-8\"?>".to_owned());
    out.push(format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{w}\" height=\"{h}\" \
         viewBox=\"0 0 {w} {h}\" role=\"img\" aria-label=\"{label}\">",
        w = layout.width,
        h = layout.height,
        label = escape_xml(&label)
    ));

    let title = options
        .title
        .as_deref()
        .map(str::trim)
        .filter(|text| !text.is_empty());
    if let Some(title) = title {
        out.push(format!("  <title>{}</title>", escape_xml(title)));
    }

    push_defs(&mut out);
    push_style(&mut out);
    push_layers(&mut out, &layout);
    push_edges(&mut out, spec, &layout);
    out.push("</svg>".to_owned());

    out.join("\n")
}

/// Renders a minimal standalone SVG document carrying `message`.
///
/// Rendering a workflow never fails; when the definition cannot be compiled, the compiled
/// entry point degrades to this document so a caller still receives valid SVG. `message`
/// is word-wrapped and XML-escaped.
#[must_use]
pub fn error_svg(message: &str) -> String {
    const WIDTH: u32 = 560;
    const BOX_PAD: u32 = 24;

    let lines = wrap_message(message, 72);
    let count = u32::try_from(lines.len()).unwrap_or(1).max(1);
    let height = 40 + count * 18 + BOX_PAD;
    let tspans: Vec<String> = lines
        .iter()
        .enumerate()
        .map(|(index, line)| {
            let dy = if index == 0 { 0 } else { 18 };
            format!(
                "<tspan x=\"{x}\" dy=\"{dy}\">{text}</tspan>",
                x = BOX_PAD + 8,
                text = escape_xml(line)
            )
        })
        .collect();
    let text = tspans.join("");
    let mut out: Vec<String> = Vec::with_capacity(16);
    out.push("<?xml version=\"1.0\" encoding=\"UTF-8\"?>".to_owned());
    out.push(format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{WIDTH}\" height=\"{height}\" \
         viewBox=\"0 0 {WIDTH} {height}\" role=\"img\" aria-label=\"workflow rendering failed\">"
    ));
    out.push("  <title>workflow rendering failed</title>".to_owned());
    push_style(&mut out);
    out.push("  <g class=\"error\">".to_owned());
    out.push(format!(
        "    <rect x=\"8\" y=\"8\" width=\"{w}\" height=\"{h}\" rx=\"10\" ry=\"10\" \
         fill=\"#fff5f5\" stroke=\"#c53030\" stroke-width=\"1.5\"/>",
        w = WIDTH - 16,
        h = height - 16
    ));
    out.push(format!(
        "    <text class=\"node-label\" x=\"{x}\" y=\"30\" text-anchor=\"start\">{text}</text>",
        x = BOX_PAD + 8
    ));
    out.push("  </g>".to_owned());
    out.push("</svg>".to_owned());
    out.join("\n")
}

/// The visual role a node plays, selecting its shape and fill.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Role {
    /// A workflow entry node: round-edged and light blue.
    Start,
    /// A workflow exit node: round-edged and light gray.
    End,
    /// Any other node: a plain rectangle with a white fill.
    Normal,
}

impl Role {
    /// The `class` fragment used on the node group.
    fn class(self) -> &'static str {
        match self {
            Self::Start => "start",
            Self::End => "end",
            Self::Normal => "normal",
        }
    }

    /// The node fill colour.
    fn fill(self) -> &'static str {
        match self {
            Self::Start => "#ebf8ff",
            Self::End => "#edf2f7",
            Self::Normal => "#ffffff",
        }
    }
}

/// A node with its resolved label, geometry, and role.
struct PlacedNode {
    /// The label lines: the id, optionally followed by a distinct display name.
    lines: Vec<String>,
    /// The left edge, in pixels.
    x: u32,
    /// The top edge, in pixels.
    y: u32,
    /// The box width, in pixels.
    w: u32,
    /// The box height, in pixels.
    h: u32,
    /// The visual role.
    role: Role,
}

/// The resolved canvas: placed nodes, their layers, and the document size.
struct Layout {
    /// Every placed node, keyed by id.
    nodes: BTreeMap<String, PlacedNode>,
    /// The node ids of each layer, outermost layer first.
    layers: Vec<Vec<String>>,
    /// The document width, in pixels.
    width: u32,
    /// The document height, in pixels.
    height: u32,
    /// The direction the layers were laid out in.
    direction: LayoutDirection,
}

impl Layout {
    /// Places every node of `spec` on the canvas described by `options`.
    fn compute(spec: &WorkflowSpec, options: &SvgOptions) -> Self {
        let ids = node_ids(spec);
        let layers = compute_layers(spec);
        let starts = start_set(spec, &ids);
        let ends = end_set(spec, options, &ids);

        let mut nodes: BTreeMap<String, PlacedNode> = BTreeMap::new();
        let mut cursor = MARGIN;
        for layer in &layers {
            let widest = layer
                .iter()
                .map(|id| node_width(&label_lines(id, spec)))
                .max()
                .unwrap_or(0);
            for (row, id) in layer.iter().enumerate() {
                let row = u32::try_from(row).unwrap_or(u32::MAX);
                let lines = label_lines(id, spec);
                let w = node_width(&lines);
                let (x, y) = match options.direction {
                    LayoutDirection::LeftRight => (cursor, MARGIN + row.saturating_mul(ROW_GAP)),
                    LayoutDirection::TopDown => (
                        MARGIN + row.saturating_mul(widest.saturating_add(COL_GAP)),
                        cursor,
                    ),
                };
                let role = if starts.contains(id) {
                    Role::Start
                } else if ends.contains(id) {
                    Role::End
                } else {
                    Role::Normal
                };
                nodes.insert(
                    id.clone(),
                    PlacedNode {
                        lines,
                        x,
                        y,
                        w,
                        h: NODE_H,
                        role,
                    },
                );
            }
            cursor = cursor.saturating_add(match options.direction {
                LayoutDirection::LeftRight => widest.saturating_add(COL_GAP),
                LayoutDirection::TopDown => NODE_H.saturating_add(ROW_GAP),
            });
        }

        let width = nodes
            .values()
            .map(|node| node.x.saturating_add(node.w))
            .max()
            .unwrap_or(0)
            .saturating_add(MARGIN)
            .max(MIN_W);
        let height = nodes
            .values()
            .map(|node| node.y.saturating_add(node.h))
            .max()
            .unwrap_or(0)
            .saturating_add(MARGIN)
            .max(MIN_H);

        Self {
            nodes,
            layers,
            width,
            height,
            direction: options.direction,
        }
    }
}

/// Every id the spec mentions: component configs, edge endpoints, and start nodes.
fn node_ids(spec: &WorkflowSpec) -> BTreeSet<String> {
    let mut ids: BTreeSet<String> = spec.comp_configs.keys().cloned().collect();
    for (source, targets) in spec.edges.iter().chain(spec.stream_edges.iter()) {
        ids.insert(source.clone());
        ids.extend(targets.iter().cloned());
    }
    ids.extend(spec.start_nodes.iter().cloned());
    ids
}

/// The merged, sorted adjacency of the batch and streaming edges.
fn adjacency(spec: &WorkflowSpec) -> BTreeMap<String, Vec<String>> {
    let mut merged: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for (source, targets) in spec.edges.iter().chain(spec.stream_edges.iter()) {
        merged
            .entry(source.clone())
            .or_default()
            .extend(targets.iter().cloned());
    }
    merged
        .into_iter()
        .map(|(source, targets)| (source, targets.into_iter().collect()))
        .collect()
}

/// The breadth-first roots: the declared start nodes, else the sources, else the first id.
fn roots(
    spec: &WorkflowSpec,
    ids: &BTreeSet<String>,
    adjacency: &BTreeMap<String, Vec<String>>,
) -> Vec<String> {
    let declared: BTreeSet<String> = spec
        .start_nodes
        .iter()
        .filter(|id| ids.contains(*id))
        .cloned()
        .collect();
    if !declared.is_empty() {
        return declared.into_iter().collect();
    }

    let with_incoming: BTreeSet<&String> = adjacency.values().flatten().collect();
    let sources: Vec<String> = ids
        .iter()
        .filter(|id| !with_incoming.contains(id))
        .cloned()
        .collect();
    if !sources.is_empty() {
        return sources;
    }

    ids.iter().next().cloned().into_iter().collect()
}

/// Groups the node ids into breadth-first layers.
///
/// Cycles terminate because a visited node is never re-entered, and nodes no root reaches
/// (unreachable components, pure cycles) are appended as one trailing sorted layer.
fn compute_layers(spec: &WorkflowSpec) -> Vec<Vec<String>> {
    let ids = node_ids(spec);
    if ids.is_empty() {
        return Vec::new();
    }
    let adjacency = adjacency(spec);
    let mut depth: BTreeMap<String, usize> = BTreeMap::new();
    let mut queue: VecDeque<String> = VecDeque::new();
    for root in roots(spec, &ids, &adjacency) {
        if depth.insert(root.clone(), 0).is_none() {
            queue.push_back(root);
        }
    }
    while let Some(current) = queue.pop_front() {
        let current_depth = depth.get(&current).copied().unwrap_or(0);
        if let Some(neighbours) = adjacency.get(&current) {
            for next in neighbours {
                if !depth.contains_key(next) {
                    depth.insert(next.clone(), current_depth.saturating_add(1));
                    queue.push_back(next.clone());
                }
            }
        }
    }

    let mut grouped: BTreeMap<usize, Vec<String>> = BTreeMap::new();
    for (id, node_depth) in &depth {
        grouped.entry(*node_depth).or_default().push(id.clone());
    }
    let trailing: Vec<String> = ids
        .iter()
        .filter(|id| !depth.contains_key(*id))
        .cloned()
        .collect();
    if !trailing.is_empty() {
        let next_depth = depth
            .values()
            .max()
            .map_or(0, |value| value.saturating_add(1));
        grouped.entry(next_depth).or_default().extend(trailing);
    }
    grouped.into_values().collect()
}

/// The declared start nodes that the spec actually mentions.
fn start_set(spec: &WorkflowSpec, ids: &BTreeSet<String>) -> BTreeSet<String> {
    spec.start_nodes
        .iter()
        .filter(|id| ids.contains(*id))
        .cloned()
        .collect()
}

/// The explicit end nodes, else the sinks, else a component literally named `end`.
fn end_set(spec: &WorkflowSpec, options: &SvgOptions, ids: &BTreeSet<String>) -> BTreeSet<String> {
    let declared: BTreeSet<String> = options
        .end_nodes
        .iter()
        .filter(|id| ids.contains(*id))
        .cloned()
        .collect();
    if !declared.is_empty() {
        return declared;
    }
    let adjacency = adjacency(spec);
    let mut ends: BTreeSet<String> = ids
        .iter()
        .filter(|id| adjacency.get(*id).is_none_or(Vec::is_empty))
        .cloned()
        .collect();
    if spec.comp_configs.contains_key("end") {
        ends.insert("end".to_owned());
    }
    ends
}

/// The label lines of a node: its id, plus a display name when it differs.
fn label_lines(id: &str, spec: &WorkflowSpec) -> Vec<String> {
    let mut lines = vec![id.to_owned()];
    let name = spec
        .comp_configs
        .get(id)
        .and_then(|node| node.name.as_deref());
    match name {
        Some(name) if !name.is_empty() && name != id => lines.push(name.to_owned()),
        _ => {}
    }
    lines
}

/// The box width for `lines`: clamped between 120 and 320 pixels, wider for longer text.
fn node_width(lines: &[String]) -> u32 {
    let chars = lines
        .iter()
        .map(|line| line.chars().count())
        .max()
        .unwrap_or(0);
    let chars = u32::try_from(chars).unwrap_or(u32::MAX);
    let tenths = chars
        .saturating_mul(72)
        .saturating_add(640)
        .clamp(1200, 3200);
    tenths / 10
}

/// The document label: the card name, its id, or `"workflow"` when both are empty.
fn card_label(card: &WorkflowCard) -> String {
    let name = card.name.trim();
    if !name.is_empty() {
        return name.to_owned();
    }
    let id = card.id.trim();
    if id.is_empty() {
        "workflow".to_owned()
    } else {
        id.to_owned()
    }
}

/// Escapes the five XML metacharacters.
fn escape_xml(text: &str) -> String {
    let mut escaped = String::with_capacity(text.len());
    for character in text.chars() {
        match character {
            '&' => escaped.push_str("&amp;"),
            '<' => escaped.push_str("&lt;"),
            '>' => escaped.push_str("&gt;"),
            '"' => escaped.push_str("&quot;"),
            '\'' => escaped.push_str("&apos;"),
            other => escaped.push(other),
        }
    }
    escaped
}

/// Word-wraps `message` into lines of at most `width` characters.
fn wrap_message(message: &str, width: usize) -> Vec<String> {
    let mut lines: Vec<String> = Vec::new();
    let mut current = String::new();
    for word in message.split_whitespace() {
        if current.is_empty() {
            current.push_str(word);
        } else if current.chars().count() + 1 + word.chars().count() <= width {
            current.push(' ');
            current.push_str(word);
        } else {
            lines.push(std::mem::take(&mut current));
            current.push_str(word);
        }
    }
    if !current.is_empty() {
        lines.push(current);
    }
    lines
}

/// Appends the arrow marker definition.
fn push_defs(out: &mut Vec<String>) {
    out.push("  <defs>".to_owned());
    out.push(
        "    <marker id=\"arrow\" viewBox=\"0 0 10 10\" refX=\"9\" refY=\"5\" markerWidth=\"8\" \
         markerHeight=\"8\" orient=\"auto-start-reverse\">"
            .to_owned(),
    );
    out.push("      <path d=\"M 0 0 L 10 5 L 0 10 Z\" fill=\"#4a5568\"/>".to_owned());
    out.push("    </marker>".to_owned());
    out.push("  </defs>".to_owned());
}

/// Appends the document style sheet.
fn push_style(out: &mut Vec<String>) {
    out.push("  <style>".to_owned());
    out.push(
        "    .node-label { font: 12px -apple-system, \"Segoe UI\", Helvetica, Arial, \
         sans-serif; fill: #1a202c; }"
            .to_owned(),
    );
    out.push("    .edge { fill: none; }".to_owned());
    out.push("    .edge.stream { stroke-dasharray: 6 4; }".to_owned());
    out.push("  </style>".to_owned());
}

/// Appends one `<g class="layer">` per layer with its nodes.
fn push_layers(out: &mut Vec<String>, layout: &Layout) {
    for (depth, layer) in layout.layers.iter().enumerate() {
        out.push(format!("  <g class=\"layer\" data-depth=\"{depth}\">"));
        for id in layer {
            if let Some(node) = layout.nodes.get(id) {
                push_node(out, id, node);
            }
        }
        out.push("  </g>".to_owned());
    }
}

/// Appends one `<g class="node">` with its rectangle and label.
fn push_node(out: &mut Vec<String>, id: &str, node: &PlacedNode) {
    out.push(format!(
        "    <g class=\"node {}\" data-id=\"{}\">",
        node.role.class(),
        escape_xml(id)
    ));
    let rounded = if node.role == Role::Normal {
        String::new()
    } else {
        " rx=\"16\" ry=\"16\"".to_owned()
    };
    out.push(format!(
        "      <rect x=\"{x}\" y=\"{y}\" width=\"{w}\" height=\"{h}\"{rounded} \
         fill=\"{fill}\" stroke=\"#4a5568\" stroke-width=\"1.5\"/>",
        x = node.x,
        y = node.y,
        w = node.w,
        h = node.h,
        fill = node.role.fill()
    ));

    let count = u32::try_from(node.lines.len()).unwrap_or(1);
    let center = node.x.saturating_add(node.w / 2);
    let first_baseline = node
        .y
        .saturating_add(node.h / 2)
        .saturating_add(4)
        .saturating_sub(count.saturating_sub(1).saturating_mul(7));
    out.push(format!(
        "      <text class=\"node-label\" x=\"{center}\" y=\"{first_baseline}\" \
         text-anchor=\"middle\">"
    ));
    for (index, line) in node.lines.iter().enumerate() {
        let dy = if index == 0 { 0 } else { 14 };
        out.push(format!(
            "        <tspan x=\"{center}\" dy=\"{dy}\">{}</tspan>",
            escape_xml(line)
        ));
    }
    out.push("      </text>".to_owned());
    out.push("    </g>".to_owned());
}

/// Appends the edge group: batch edges solid, streaming edges dashed.
fn push_edges(out: &mut Vec<String>, spec: &WorkflowSpec, layout: &Layout) {
    out.push("  <g class=\"edges\">".to_owned());
    push_edge_map(out, &spec.edges, layout, false);
    push_edge_map(out, &spec.stream_edges, layout, true);
    out.push("  </g>".to_owned());
}

/// Appends one `<path>` per connection of `edges`.
fn push_edge_map(
    out: &mut Vec<String>,
    edges: &BTreeMap<String, Vec<String>>,
    layout: &Layout,
    stream: bool,
) {
    for (source, targets) in edges {
        let Some(src) = layout.nodes.get(source) else {
            continue;
        };
        for target in targets {
            let Some(dst) = layout.nodes.get(target) else {
                continue;
            };
            let d = edge_path(src, dst, layout.direction);
            let class = if stream { "edge stream" } else { "edge" };
            let style = if stream {
                " stroke=\"#2b6cb0\" stroke-width=\"1.5\" stroke-dasharray=\"6 4\""
            } else {
                " stroke=\"#4a5568\" stroke-width=\"1.5\""
            };
            out.push(format!(
                "    <path class=\"{class}\" data-source=\"{source}\" data-target=\"{target}\" \
                 d=\"{d}\" fill=\"none\"{style} marker-end=\"url(#arrow)\"/>",
                source = escape_xml(source),
                target = escape_xml(target)
            ));
        }
    }
}

/// The cubic Bézier path from the source node to the target node.
///
/// Back edges and self edges keep their control points on the sweep side, so they render as
/// a loop instead of degenerating.
fn edge_path(src: &PlacedNode, dst: &PlacedNode, direction: LayoutDirection) -> String {
    let (x1, y1, x2, y2) = match direction {
        LayoutDirection::LeftRight => (
            i64::from(src.x.saturating_add(src.w)),
            i64::from(src.y.saturating_add(src.h / 2)),
            i64::from(dst.x),
            i64::from(dst.y.saturating_add(dst.h / 2)),
        ),
        LayoutDirection::TopDown => (
            i64::from(src.x.saturating_add(src.w / 2)),
            i64::from(src.y.saturating_add(src.h)),
            i64::from(dst.x.saturating_add(dst.w / 2)),
            i64::from(dst.y),
        ),
    };
    let (c1x, c1y, c2x, c2y) = match direction {
        LayoutDirection::LeftRight => {
            let bend = (x2 - x1).abs() / 2;
            (x1 + bend.max(40), y1, x2 - bend.max(40), y2)
        }
        LayoutDirection::TopDown => {
            let bend = (y2 - y1).abs() / 2;
            (x1, y1 + bend.max(40), x2, y2 - bend.max(40))
        }
    };
    format!("M {x1} {y1} C {c1x} {c1y}, {c2x} {c2y}, {x2} {y2}")
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::card::NodeSpec;

    /// A spec with `comp_ids` nodes, `edges` connections, and `starts` entry ids.
    fn spec_with(comp_ids: &[&str], edges: &[(&str, &str)], starts: &[&str]) -> WorkflowSpec {
        let mut spec = WorkflowSpec {
            comp_configs: comp_ids
                .iter()
                .map(|id| ((*id).to_owned(), NodeSpec::default()))
                .collect(),
            start_nodes: starts.iter().map(|id| (*id).to_owned()).collect(),
            ..WorkflowSpec::default()
        };
        for (source, target) in edges {
            spec.edges
                .entry((*source).to_owned())
                .or_default()
                .push((*target).to_owned());
        }
        spec
    }

    #[test]
    fn linear_chain_layers_follow_the_edges() {
        let spec = spec_with(&["a", "b", "c"], &[("a", "b"), ("b", "c")], &["a"]);
        assert_eq!(compute_layers(&spec), vec![vec!["a"], vec!["b"], vec!["c"]]);
    }

    #[test]
    fn diamond_puts_siblings_in_one_layer() {
        let spec = spec_with(
            &["start", "a", "b", "end"],
            &[("start", "a"), ("start", "b"), ("a", "end"), ("b", "end")],
            &["start"],
        );
        assert_eq!(
            compute_layers(&spec),
            vec![vec!["start"], vec!["a", "b"], vec!["end"]]
        );
    }

    #[test]
    fn cycles_terminate() {
        let spec = spec_with(&["a", "b"], &[("a", "b"), ("b", "a")], &[]);
        assert_eq!(compute_layers(&spec), vec![vec!["a"], vec!["b"]]);
    }

    #[test]
    fn disconnected_nodes_land_in_a_trailing_layer() {
        let spec = spec_with(&["a", "b", "loner"], &[("a", "b")], &["a"]);
        assert_eq!(
            compute_layers(&spec),
            vec![vec!["a"], vec!["b"], vec!["loner"]]
        );
    }

    #[test]
    fn stream_edges_contribute_to_the_layering() {
        let mut spec = spec_with(&["a", "b"], &[], &["a"]);
        spec.stream_edges
            .insert("a".to_owned(), vec!["b".to_owned()]);
        assert_eq!(compute_layers(&spec), vec![vec!["a"], vec!["b"]]);
    }

    #[test]
    fn escaping_covers_every_xml_metacharacter() {
        assert_eq!(
            escape_xml("a & <b> \"c\" 'd'"),
            "a &amp; &lt;b&gt; &quot;c&quot; &apos;d&apos;"
        );
    }

    #[test]
    fn empty_spec_renders_a_valid_document() {
        let svg = to_svg(&WorkflowSpec::default(), &WorkflowCard::new("empty"), None);
        assert!(svg.starts_with("<?xml"));
        assert!(svg.contains("<svg"));
        assert!(svg.contains("viewBox="));
        assert!(svg.contains("<style>"));
        assert!(svg.contains("<g class=\"edges\">"));
        assert!(svg.contains("width=\"240\""));
        assert!(svg.contains("height=\"120\""));
    }

    #[test]
    fn rendering_is_deterministic() {
        let spec = spec_with(&["a", "b"], &[("a", "b")], &["a"]);
        let card = WorkflowCard::new("stable");
        let first = to_svg(&spec, &card, Some("Stable"));
        let second = to_svg(&spec, &card, Some("Stable"));
        assert_eq!(first, second);
    }

    #[test]
    fn start_and_end_nodes_are_rounded() {
        let spec = spec_with(
            &["start", "mid", "end"],
            &[("start", "mid"), ("mid", "end")],
            &["start"],
        );
        let card = WorkflowCard::new("rounded");
        let options = SvgOptions {
            end_nodes: vec!["end".to_owned()],
            ..SvgOptions::default()
        };
        let svg = render_svg(&spec, &card, &options);
        assert_eq!(svg.matches("rx=\"16\"").count(), 2);
        let mid = svg
            .split("<g class=\"node normal\" data-id=\"mid\">")
            .nth(1)
            .expect("the middle node is placed");
        let mid_rect = mid
            .split("<rect")
            .nth(1)
            .expect("the middle node has a rectangle");
        assert!(!mid_rect.contains("rx="));
    }

    #[test]
    fn non_ascii_and_special_ids_are_escaped() {
        let spec = WorkflowSpec {
            comp_configs: [(
                "a&b".to_owned(),
                NodeSpec {
                    name: Some("x<y>".to_owned()),
                    ..NodeSpec::default()
                },
            )]
            .into_iter()
            .collect(),
            edges: [("a&b".to_owned(), vec!["a&b".to_owned()])]
                .into_iter()
                .collect(),
            ..WorkflowSpec::default()
        };
        let svg = to_svg(&spec, &WorkflowCard::new("esc&ape"), None);
        assert!(svg.contains("&amp;"));
        assert!(svg.contains("&lt;y&gt;"));
        assert!(!svg.contains("a&b"));
        assert!(svg.contains("esc&amp;ape"));
    }

    #[test]
    fn a_self_edge_renders_without_panicking() {
        let spec = spec_with(&["a"], &[("a", "a")], &["a"]);
        let svg = to_svg(&spec, &WorkflowCard::new("self"), None);
        assert_eq!(svg.matches("class=\"edge\"").count(), 1);
    }

    #[test]
    fn error_svg_is_a_valid_document() {
        let svg = error_svg("workflow 'wf' could not be compiled: <no start> & no end");
        assert!(svg.starts_with("<?xml"));
        assert!(svg.contains("viewBox="));
        assert!(svg.contains("&lt;no start&gt; &amp; no end"));
        assert!(svg.contains("<title>workflow rendering failed</title>"));
    }

    #[test]
    fn top_down_swaps_the_axes() {
        let spec = spec_with(&["a", "b"], &[("a", "b")], &["a"]);
        let card = WorkflowCard::new("down");
        let left_right = to_svg(&spec, &card, None);
        let top_down = render_svg(
            &spec,
            &card,
            &SvgOptions {
                direction: LayoutDirection::TopDown,
                ..SvgOptions::default()
            },
        );
        assert_ne!(left_right, top_down);
        assert!(top_down.contains("viewBox"));
    }
}
