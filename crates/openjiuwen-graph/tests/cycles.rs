//! Bounded cycles and the superstep limit.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]

mod common;

use serde_json::json;

use openjiuwen_graph::{
    CompileOptions, CompiledGraph, ExecutionOptions, FnRouter, GraphBuilder, RouterContext,
};

fn loop_builder(always_loop: bool) -> GraphBuilder {
    let router = FnRouter::new("again", move |ctx: &RouterContext| {
        let count = ctx
            .state()
            .get("count")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);
        Box::pin(async move {
            Ok(if always_loop || count < 3 {
                vec!["loop".to_string()]
            } else {
                vec!["done".to_string()]
            })
        })
    });
    GraphBuilder::new("cycle")
        .node("loop", common::counting("loop"))
        .node("done", common::marker("done"))
        .entry("loop")
        .conditional_edges_to("loop", ["loop", "done"], router)
        .exit("done")
}

#[tokio::test]
async fn bounded_loop_terminates() {
    let result = loop_builder(false)
        .compile()
        .unwrap()
        .run(json!({}), common::session(), ExecutionOptions::default())
        .await
        .unwrap();
    assert_eq!(result.state.get("count"), Some(&json!(3)));
    assert_eq!(result.state.get("ran_done"), Some(&json!(true)));
    assert_eq!(
        result.visited,
        vec![
            "loop".to_string(),
            "loop".to_string(),
            "loop".to_string(),
            "done".to_string()
        ]
    );
    assert_eq!(result.supersteps, 4);
}

#[tokio::test]
async fn unbounded_loop_hits_the_execution_limit() {
    let error = loop_builder(true)
        .compile()
        .unwrap()
        .run(
            json!({}),
            common::session(),
            ExecutionOptions {
                max_supersteps: Some(3),
                ..ExecutionOptions::default()
            },
        )
        .await
        .unwrap_err();
    assert_eq!(error.kind(), openjiuwen_common::error::ErrorKind::Graph);
    assert!(error
        .message()
        .contains("exceeded the maximum of 3 supersteps"));
}

#[tokio::test]
async fn compile_option_sets_the_default_limit() {
    let graph: CompiledGraph = loop_builder(true)
        .compile_with(CompileOptions {
            max_supersteps: Some(2),
            ..CompileOptions::default()
        })
        .unwrap();
    let error = graph
        .run(json!({}), common::session(), ExecutionOptions::default())
        .await
        .unwrap_err();
    assert!(error
        .message()
        .contains("exceeded the maximum of 2 supersteps"));
}
