//! Mermaid rendering.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]

mod common;

use openjiuwen_graph::{FnRouter, GraphBuilder, MermaidDirection, MermaidOptions, RouterContext};

fn sample_graph() -> openjiuwen_graph::CompiledGraph {
    GraphBuilder::new("sample")
        .node("start", common::noop("start"))
        .node("check", common::noop("check"))
        .node("done", common::noop("done"))
        .edge("start", "check")
        .conditional_edges_to(
            "check",
            ["done"],
            FnRouter::new("finish", |_ctx: &RouterContext| {
                Box::pin(async { Ok(vec!["done".to_string()]) })
            }),
        )
        .exit("done")
        .compile()
        .unwrap()
}

#[test]
fn renders_a_top_down_flowchart() {
    let text = sample_graph().to_mermaid();
    assert!(text.starts_with("flowchart TB\n"), "{text}");
    assert!(text.contains("    node_1(\"start\")\n"), "{text}");
    assert!(text.contains("    node_2[\"check\"]\n"), "{text}");
    assert!(text.contains("    node_3(\"done\")\n"), "{text}");
    assert!(text.contains("    node_1 --> node_2\n"), "{text}");
    assert!(
        text.contains("    node_2 -. \"check\" .-> node_3\n"),
        "{text}"
    );
}

#[test]
fn options_add_a_title_and_direction() {
    let text = sample_graph().to_mermaid_with(&MermaidOptions {
        title: Some("My graph: v1".to_string()),
        direction: MermaidDirection::LeftRight,
    });
    assert!(
        text.starts_with("---\ntitle: My graph: v1\n---\nflowchart LR\n"),
        "{text}"
    );
}

#[test]
fn title_is_sanitized() {
    let text = sample_graph().to_mermaid_with(&MermaidOptions {
        title: Some("a\"b`c".to_string()),
        direction: MermaidDirection::TopDown,
    });
    assert!(text.contains("title: a_b_c\n"), "{text}");
    assert!(!text.contains('`'));
}

#[test]
fn labels_escape_quotes_and_backslashes() {
    let graph = GraphBuilder::new("escape")
        .node("a\"b", common::noop("a\"b"))
        .node("c\\d", common::noop("c\\d"))
        .edge("a\"b", "c\\d")
        .compile()
        .unwrap();
    let text = graph.to_mermaid();
    assert!(text.contains("node_1(\"a#quot;b\")"), "{text}");
    assert!(text.contains("node_2[\"c\\\\d\"]"), "{text}");
}

#[test]
fn empty_metadata_graph_renders_no_frontmatter() {
    let text = sample_graph().to_mermaid();
    assert!(!text.contains("---"));
}

#[test]
fn direction_keywords_match_mermaid() {
    assert_eq!(MermaidDirection::TopDown.as_str(), "TB");
    assert_eq!(MermaidDirection::LeftRight.as_str(), "LR");
}

#[test]
fn dynamic_conditional_targets_are_skipped() {
    let graph = GraphBuilder::new("dynamic")
        .node("a", common::noop("a"))
        .node("b", common::noop("b"))
        .conditional_edges(
            "a",
            FnRouter::new("to_b", |_ctx: &RouterContext| {
                Box::pin(async { Ok(vec!["b".to_string()]) })
            }),
        )
        .compile()
        .unwrap();
    let text = graph.to_mermaid();
    assert!(text.contains("node_1(\"a\")"), "{text}");
    assert!(text.contains("node_2(\"b\")"), "{text}");
    assert!(!text.contains('*'), "{text}");
    assert!(!text.contains("-->"), "{text}");
}
