//! Ports the `Workflow.draw` surface: Mermaid text and offline SVG rendering.
//!
//! Python's `Workflow.draw(title="", output_format="mermaid")` delegates to
//! `Drawable.to_mermaid` / `to_mermaid_svg`, with mermaid.ink rasterizing the chart. These
//! tests cover the Rust stand-ins — [`Workflow::draw`] with a [`DrawFormat`] — and assert
//! the SVG is rendered locally and deterministically. Everything here is offline: no
//! network, no clocks, no hash-ordered iteration.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use async_trait::async_trait;
use openjiuwen_common::error::Result;
use openjiuwen_workflow::card::{NodeSpec, WorkflowCard, WorkflowSpec};
use openjiuwen_workflow::{
    render, Component, ComponentAbility, ComponentCard, ComponentContext, ComponentOptions,
    DrawFormat, EndComponent, StartComponent, SvgOptions, Workflow,
};
use serde_json::Value;

/// A pass-through component standing in for a workflow's middle step.
#[derive(Debug, Clone, Default)]
struct Echo;

#[async_trait]
impl Component for Echo {
    fn card(&self) -> ComponentCard {
        ComponentCard::new("echo", "echo").with_abilities(self.abilities())
    }

    fn abilities(&self) -> Vec<ComponentAbility> {
        vec![ComponentAbility::Invoke]
    }

    fn component_type(&self) -> &'static str {
        "echo"
    }

    async fn invoke(&self, inputs: Value, _ctx: &ComponentContext) -> Result<Value> {
        Ok(inputs)
    }
}

/// A three-node workflow: `start -> middle -> end`.
fn demo() -> Workflow {
    let mut workflow = Workflow::new(WorkflowCard::new("draw-demo"));
    workflow
        .set_start_comp("start", StartComponent::new(), ComponentOptions::new())
        .add_workflow_comp("middle", Echo, ComponentOptions::new())
        .set_end_comp("end", EndComponent::new(), ComponentOptions::new())
        .add_connection("start", "middle")
        .add_connection("middle", "end");
    workflow
}

/// A spec with the given node ids and batch connections.
fn spec_with(comp_ids: &[&str], edges: &[(&str, &str)]) -> WorkflowSpec {
    let mut spec = WorkflowSpec {
        comp_configs: comp_ids
            .iter()
            .map(|id| ((*id).to_owned(), NodeSpec::default()))
            .collect(),
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
fn draw_svg_produces_a_document() {
    let workflow = demo();
    let svg = workflow.draw(DrawFormat::Svg).unwrap();

    assert!(svg.starts_with("<?xml"), "missing XML declaration:\n{svg}");
    assert!(svg.contains("<svg"), "missing root element:\n{svg}");
    assert!(svg.contains("viewBox="), "missing viewBox:\n{svg}");
    assert!(svg.contains("<marker"), "missing arrow marker:\n{svg}");
    for label in ["start", "middle", "end"] {
        assert!(svg.contains(label), "missing label {label}:\n{svg}");
    }
    assert_eq!(
        svg.matches("class=\"edge\"").count(),
        2,
        "two connections render two edges:\n{svg}"
    );

    let compiled = workflow.build().unwrap();
    let compiled_svg = compiled.to_svg();
    assert!(compiled_svg.starts_with("<?xml"));
    assert!(compiled_svg.contains("data-id=\"middle\""));
    assert_eq!(compiled_svg, svg, "draw and to_svg agree");
}

#[test]
fn svg_escapes_labels() {
    let mut spec = WorkflowSpec::default();
    spec.comp_configs.insert(
        "a&b".to_owned(),
        NodeSpec {
            name: Some("x<y>".to_owned()),
            ..NodeSpec::default()
        },
    );
    spec.edges.insert("a&b".to_owned(), vec!["a&b".to_owned()]);

    let svg = render::to_svg(&spec, &WorkflowCard::new("esc&ape"), Some("t<1>"));
    assert!(svg.contains("&amp;"), "ampersand escaped:\n{svg}");
    assert!(svg.contains("&lt;y&gt;"), "angle brackets escaped:\n{svg}");
    assert!(svg.contains("t&lt;1&gt;"), "title escaped:\n{svg}");
    assert!(svg.contains("esc&amp;ape"), "aria label escaped:\n{svg}");
    assert!(!svg.contains("a&b"), "raw id leaked:\n{svg}");
    assert!(!svg.contains("<y>"), "raw name leaked:\n{svg}");

    // All five XML metacharacters, through a user-derived label.
    let card = WorkflowCard::new("&<>\"'");
    let escaped = render::to_svg(&WorkflowSpec::default(), &card, None);
    assert!(
        escaped.contains("&amp;&lt;&gt;&quot;&apos;"),
        "all metacharacters escaped:\n{escaped}"
    );
    assert!(!escaped.contains("\"'"), "raw quotes leaked:\n{escaped}");
}

#[test]
fn start_and_end_are_rounded() {
    let mut spec = spec_with(
        &["start", "mid", "end"],
        &[("start", "mid"), ("mid", "end")],
    );
    spec.start_nodes = vec!["start".to_owned()];

    let svg = render::render_svg(
        &spec,
        &WorkflowCard::new("flow"),
        &SvgOptions {
            end_nodes: vec!["end".to_owned()],
            title: Some("My Flow".to_owned()),
            ..SvgOptions::default()
        },
    );

    assert_eq!(
        svg.matches("rx=\"16\"").count(),
        2,
        "exactly the start and end are rounded:\n{svg}"
    );
    assert!(
        svg.contains("<title>My Flow</title>"),
        "title emitted:\n{svg}"
    );

    let mid_block = svg
        .split("<g class=\"node normal\" data-id=\"mid\">")
        .nth(1)
        .expect("the middle node is placed");
    let mid_rect = mid_block
        .split("<rect")
        .nth(1)
        .expect("the middle node has a rectangle");
    assert!(
        !mid_rect.contains("rx="),
        "the middle node is square:\n{svg}"
    );
}

#[test]
fn draw_is_deterministic() {
    let workflow = demo();
    let first = workflow.draw(DrawFormat::Svg).unwrap();
    let second = workflow.draw(DrawFormat::Svg).unwrap();
    assert_eq!(first, second);
}

#[test]
fn draw_mermaid_matches_to_mermaid() {
    let workflow = demo();
    let drawn = workflow.draw(DrawFormat::Mermaid).unwrap();
    let direct = workflow.to_mermaid().unwrap();
    assert_eq!(drawn, direct);
    assert!(
        drawn.contains("flowchart"),
        "expected a flowchart:\n{drawn}"
    );
}

#[test]
fn empty_single_and_disconnected_render() {
    let empty = render::to_svg(&WorkflowSpec::default(), &WorkflowCard::new("empty"), None);
    assert!(empty.starts_with("<?xml"));
    assert!(empty.contains("<svg"));
    assert!(empty.contains("viewBox="));
    assert!(empty.contains("<style>"));
    assert!(empty.contains("<g class=\"edges\">"));

    let single = WorkflowSpec {
        start_nodes: vec!["solo".to_owned()],
        comp_configs: [("solo".to_owned(), NodeSpec::default())]
            .into_iter()
            .collect(),
        ..WorkflowSpec::default()
    };
    let rendered = render::to_svg(&single, &WorkflowCard::new("single"), None);
    assert!(rendered.contains("solo"));
    assert!(rendered.contains("viewBox="));

    let mut disconnected = spec_with(&["start", "end", "loner"], &[("start", "end")]);
    disconnected.start_nodes = vec!["start".to_owned()];
    let rendered = render::to_svg(&disconnected, &WorkflowCard::new("split"), None);
    assert!(rendered.contains("data-id=\"loner\""), "{rendered}");
    assert!(rendered.contains("data-id=\"end\""));
}
