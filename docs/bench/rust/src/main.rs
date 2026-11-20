//! Harness for the Rust half of `docs/BENCHMARKS.md`.
//!
//! Three subcommands:
//! - `floor`  — process start that links the `openjiuwen` umbrella, touches a
//!   handful of core types and prints `openjiuwen::VERSION`. Its wall time and
//!   peak RSS are measured from outside (`docs/bench/python/measure_proc.py`).
//! - `prompt` — `PromptTemplate::format` throughput, 2000 renders per rep.
//! - `graph`  — 4-node linear graph execution, 200 runs per rep.
//!
//! Every timed workload uses the same sizes as `docs/bench/python/bench.py`:
//! 30 reps, the first 3 discarded as warmup, one timing sample per rep.
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

use serde_json::{json, Value};

use openjiuwen::core::foundation::prompt::{PromptTemplate, TemplateContent};
use openjiuwen::core::graph::{
    ExecutionOptions, GraphBuilder, GraphContext, GraphState, NodeOutcome, SyncNode,
};
use openjiuwen::core::session::{Session, WorkflowSession};

const REPS: usize = 30;
const DISCARD: usize = 3;

const TEMPLATE: &str = "You are {{role}}. Answer the question: {{question}} in at most {{limit}} words.";

/// Length of the rendered text without allocating, so the checksum keeps the
/// render in the measurement without dominating it.
fn content_len(template: &PromptTemplate) -> usize {
    match &template.content {
        TemplateContent::Text(text) => text.len(),
        TemplateContent::Messages(messages) => messages.len(),
    }
}

fn keywords() -> BTreeMap<String, Value> {
    let mut values = BTreeMap::new();
    values.insert("role".to_owned(), json!("a terse assistant"));
    values.insert("question".to_owned(), json!("What is a monad?"));
    values.insert("limit".to_owned(), json!("20"));
    values
}

fn session() -> Arc<dyn Session> {
    Arc::new(WorkflowSession::new("s1"))
}

fn counting(
    id: &'static str,
) -> SyncNode<impl Fn(&mut GraphState, &GraphContext) -> openjiuwen::core::common::error::Result<NodeOutcome>>
{
    SyncNode::new(id, move |state: &mut GraphState, _ctx: &GraphContext| {
        let count = state.get("count").and_then(Value::as_u64).unwrap_or(0);
        state.set("count", json!(count + 1));
        Ok(NodeOutcome::Continue)
    })
}

fn build_graph() -> openjiuwen::core::graph::CompiledGraph {
    // Same logical graph as the Python harness: entry -> a -> b -> exit.
    // Python spells the endpoints as explicit `START`/`END` no-op nodes; the
    // Rust builder models them as the implicit entry and the `exit` node.
    GraphBuilder::new("bench")
        .node("a", counting("a"))
        .node("b", counting("b"))
        .edge("a", "b")
        .exit("b")
        .compile()
        .expect("bench graph compiles")
}

/// Process floor: link the umbrella, construct core types, print the version.
fn floor() {
    let template = PromptTemplate::new(TEMPLATE);
    let rendered = template.format(&keywords());
    let _graph = build_graph();
    let _session = session();
    println!(
        "openjiuwen {} (template {} bytes)",
        openjiuwen::VERSION,
        content_len(&rendered)
    );
}

fn prompt_bench(iterations: usize) -> Vec<f64> {
    let template = PromptTemplate::new(TEMPLATE);
    let values = keywords();
    let mut reps = Vec::with_capacity(REPS);
    let mut checksum = 0usize;
    for _ in 0..REPS {
        let started = Instant::now();
        for _ in 0..iterations {
            checksum += content_len(&template.format(&values));
        }
        reps.push(started.elapsed().as_secs_f64() * 1000.0);
    }
    eprintln!("prompt checksum {checksum}");
    reps
}

async fn graph_bench(iterations: usize) -> Vec<f64> {
    let graph = build_graph();
    let session = session();
    let mut reps = Vec::with_capacity(REPS);
    let mut checksum = 0u64;
    for _ in 0..REPS {
        let started = Instant::now();
        for _ in 0..iterations {
            let result = graph
                .run(json!({}), session.clone(), ExecutionOptions::default())
                .await
                .expect("bench graph runs");
            checksum += result
                .state
                .get("count")
                .and_then(Value::as_u64)
                .unwrap_or(0);
        }
        reps.push(started.elapsed().as_secs_f64() * 1000.0);
    }
    eprintln!("graph checksum {checksum}");
    reps
}

fn report(workload: &str, iterations: usize, reps: &[f64], out: Option<&str>) {
    let retained = &reps[DISCARD.min(reps.len())..];
    let mut sorted = retained.to_vec();
    sorted.sort_by(f64::total_cmp);
    let median = sorted[sorted.len() / 2];
    let p95 = sorted[((sorted.len() - 1) as f64 * 0.95).round() as usize];
    let per_iteration_us = median * 1000.0 / iterations as f64;
    let document = json!({
        "workload": workload,
        "language": "rust",
        "reps_discarded": DISCARD,
        "iterations_per_rep": iterations,
        "reps": reps
            .iter()
            .enumerate()
            .map(|(index, ms)| json!({"rep": index, "ms": ms}))
            .collect::<Vec<_>>(),
        "median_ms": median,
        "p95_ms": p95,
        "per_iteration_us": per_iteration_us,
    });
    let text = serde_json::to_string_pretty(&document).expect("report serializes");
    println!("{text}");
    if let Some(path) = out {
        std::fs::write(path, text).expect("report written");
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let command = args.first().map(String::as_str).unwrap_or("floor");
    let out = args
        .iter()
        .position(|arg| arg == "--out")
        .and_then(|index| args.get(index + 1))
        .map(String::as_str);
    // Batch size per rep. The Python harness uses 2000 renders / 200 graph runs
    // per rep; Rust is ~20x faster per operation, so its default batches are
    // enlarged here to keep every rep above ~10 ms and clear of timer noise.
    let iterations = args
        .iter()
        .position(|arg| arg == "--iterations")
        .and_then(|index| args.get(index + 1))
        .and_then(|value| value.parse::<usize>().ok());

    match command {
        "floor" => floor(),
        "prompt" => {
            let iterations = iterations.unwrap_or(2000);
            report("prompt", iterations, &prompt_bench(iterations), out);
        }
        "graph" => {
            let iterations = iterations.unwrap_or(200);
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("current-thread runtime");
            let reps = runtime.block_on(graph_bench(iterations));
            report("graph", iterations, &reps, out);
        }
        other => {
            eprintln!("unknown subcommand {other}");
            std::process::exit(2);
        }
    }
}
