#![allow(clippy::unwrap_used, clippy::expect_used, clippy::assert_is_empty)]
//! Streaming tests for the implemented `ReAct` loop.
//!
//! Every frame the loop publishes is an `AgentChunk`: model content and
//! reasoning deltas, one usage frame per model call, the `__interaction__`
//! frames of a pause, and one terminal `answer` frame carrying the
//! `AgentOutput`. When a session is supplied those frames are mirrored into the
//! session's stream hub as well. The suite drives a `ScriptedModel` and asserts
//! both routes; it is offline and deterministic, and only the session-mirroring
//! test reads a bounded timeout (500 ms) to drain the hub.
mod common;

use std::sync::Arc;
use std::time::Duration;

use futures::StreamExt;
use openjiuwen_agent::output::{ANSWER, LLM_OUTPUT, LLM_REASONING, LLM_USAGE};
use openjiuwen_agent::{Agent, AgentChunk, ReActAgentConfig};
use openjiuwen_foundation::{Model, Usage};
use openjiuwen_session::{Session, StreamEvent};
use openjiuwen_test_support::{MockCompletion, ScriptedModel};
use serde_json::json;

/// Hands a clone of `model` to the agent while the test keeps the original.
fn shared(model: &ScriptedModel) -> Arc<dyn Model> {
    Arc::new(model.clone())
}

/// Concatenates the `content` of every frame of `kind`.
fn content_of(frames: &[AgentChunk], kind: &str) -> String {
    frames
        .iter()
        .filter(|frame| frame.kind == kind)
        .filter_map(|frame| {
            frame
                .payload
                .get("content")
                .and_then(|value| value.as_str())
        })
        .collect()
}

/// Runs `agent` to completion, returning every frame it published.
async fn collect_frames(
    agent: &openjiuwen_agent::ReActAgent,
    session: Arc<dyn Session>,
) -> Vec<AgentChunk> {
    agent
        .stream(json!("hi"), Some(session))
        .map(|frame| frame.unwrap())
        .collect()
        .await
}

#[tokio::test]
async fn streamed_text_frames_assemble_into_the_final_answer() {
    let model = ScriptedModel::builder().stream_text("hello world").build();
    let agent = common::agent(ReActAgentConfig::new(), shared(&model));

    let frames = collect_frames(&agent, common::session("s1")).await;

    let streamed = content_of(&frames, LLM_OUTPUT);
    let last = frames.last().expect("the stream has a terminal frame");
    assert_eq!(last.kind, ANSWER);
    assert_eq!(last.payload["output"], "hello world");
    assert_eq!(last.payload["result_type"], "answer");
    assert_eq!(streamed, last.payload["output"].as_str().unwrap());
}

#[tokio::test]
async fn reasoning_deltas_become_llm_reasoning_frames() {
    let model = ScriptedModel::builder()
        .completion(MockCompletion::reasoning("think", "answer"))
        .build();
    let agent = common::agent(ReActAgentConfig::new(), shared(&model));

    let frames = collect_frames(&agent, common::session("s1")).await;

    let reasoning = frames
        .iter()
        .filter(|frame| frame.kind == LLM_REASONING)
        .collect::<Vec<_>>();
    assert!(!reasoning.is_empty(), "no reasoning frame was published");
    assert!(content_of(&frames, LLM_REASONING).contains("think"));
    assert_eq!(content_of(&frames, LLM_OUTPUT), "answer");
    assert_eq!(frames.last().unwrap().payload["output"], "answer");
}

#[tokio::test]
async fn one_llm_usage_frame_carries_the_reported_tokens() {
    let model = ScriptedModel::builder()
        .completion(MockCompletion::text("a").with_usage(Usage::new(3, 4)))
        .build();
    let agent = common::agent(ReActAgentConfig::new(), shared(&model));

    let frames = collect_frames(&agent, common::session("s1")).await;

    let usage = frames
        .iter()
        .filter(|frame| frame.kind == LLM_USAGE)
        .collect::<Vec<_>>();
    assert_eq!(usage.len(), 1, "{frames:?}");
    assert_eq!(usage[0].payload["usage_metadata"]["input_tokens"], 3);
    assert_eq!(usage[0].payload["usage_metadata"]["output_tokens"], 4);
}

#[tokio::test]
async fn a_supplied_session_mirrors_the_output_usage_and_answer_frames() {
    let model = ScriptedModel::builder().stream_text("mirrored").build();
    let agent = common::agent(ReActAgentConfig::new(), shared(&model));
    let session = common::session("s2");
    // Subscribe before the run so no frame is produced before the subscriber.
    let mut events = session.subscribe();

    let frames = collect_frames(&agent, Arc::clone(&session)).await;
    assert!(!frames.is_empty(), "the agent stream produced no frames");

    let mut kinds: Vec<String> = Vec::new();
    let deadline = tokio::time::Instant::now() + Duration::from_millis(500);
    loop {
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        if remaining.is_zero() {
            break;
        }
        match tokio::time::timeout(remaining, events.next()).await {
            Ok(Some(Ok(StreamEvent::Item(item)))) => {
                let terminal = item.item_type == ANSWER;
                kinds.push(item.item_type);
                if terminal {
                    break;
                }
            }
            Ok(Some(Ok(StreamEvent::End)) | None) | Err(_) => break,
            Ok(Some(Err(error))) => panic!("session stream failed: {error}"),
        }
    }

    assert!(kinds.iter().any(|kind| kind == LLM_OUTPUT), "{kinds:?}");
    assert!(kinds.iter().any(|kind| kind == LLM_USAGE), "{kinds:?}");
    assert!(kinds.iter().any(|kind| kind == ANSWER), "{kinds:?}");
}
