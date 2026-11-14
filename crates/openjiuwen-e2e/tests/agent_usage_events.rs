#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Usage-event scenario: the `context.usage` / `llm_usage` frames a real agent
//! run publishes, and the provider usage contract those frames feed.
//!
//! This binary is the cross-crate seam between the `ReActAgent` loop, the
//! context engine's usage accounting and the session stream mirror. The agent's
//! own suite pins the frame payloads and the context crate's suite pins the
//! analyzer/aggregator; here the two are driven together the way an operator
//! observes a run:
//!
//! - every streamed turn publishes exactly one `context.usage` frame after the
//!   window is prepared, and the supplied session mirrors the same payload;
//! - a model response that reports tokens additionally publishes one
//!   `llm_usage` frame, after the `context.usage` frame of its iteration and
//!   before the terminal `answer` frame;
//! - the provider usage records behind those frames normalize through
//!   `request_usage_from_metadata` into `RequestKVCacheUsage`, and the
//!   `SessionKVCacheAggregator` rolls them up per aggregation key.
//!
//! Everything is offline and deterministic: the only model is a
//! `ScriptedModel`, token counting is one character per token plus four
//! framing tokens per message, sessions are in-process, and assertions read
//! message/token counters instead of timings, timestamps or generated ids.

mod support;

use std::sync::Arc;

use futures::StreamExt;
use openjiuwen::core::context_engine::usage::{
    request_usage_from_metadata, CacheAggregationKey, SessionKVCacheAggregator, NOT_REPORTED,
    SOURCE_PROVIDER_USAGE, STATUS_NOT_REPORTED, STATUS_PARTIAL_HIT,
};
use openjiuwen::core::context_engine::{ContextConfig, TokenCounterConfig, TokenCounterKind};
use openjiuwen::core::foundation::{BaseMessage, Usage};
use openjiuwen::core::session::{Session, StreamEvent};
use openjiuwen::core::single_agent::output::{ANSWER, CONTEXT_USAGE, LLM_OUTPUT, LLM_USAGE};
use openjiuwen::core::single_agent::{Agent, AgentCard, AgentChunk, ReActAgent, ReActAgentConfig};
use openjiuwen_test_support::{agent_session, MockCompletion, ScriptedModel};
use serde_json::{json, Value};

use support::{collect_bounded, within};

/// The session id every scenario uses unless it needs a second scope.
const SESSION_ID: &str = "usage-events";

/// Deterministic token accounting: one character per token, four framing
/// tokens per message (the context engine's `Characters` counter).
fn character_counting() -> TokenCounterConfig {
    TokenCounterConfig {
        kind: TokenCounterKind::Characters,
        chars_per_token: 1,
    }
}

/// A tiny context configuration with deterministic counting and no processors,
/// so `context.usage` reports exactly the window the loop assembled.
fn usage_config() -> ContextConfig {
    ContextConfig {
        max_context_tokens: Some(160),
        token_counter: character_counting(),
        processors: Vec::new(),
        ..ContextConfig::default()
    }
}

/// A `ReActAgent` whose context engine counts characters and whose system
/// prompt is the fixed three-character text `"ctx"` (seven tokens including
/// framing), which makes every `token_count` assertion exact.
fn usage_agent(model: &ScriptedModel) -> ReActAgent {
    let mut config = ReActAgentConfig::new();
    config.configure_context_engine(usage_config());
    config.prompt_template = vec![BaseMessage::system("ctx")];
    let agent = ReActAgent::new(AgentCard::new("usage-events"));
    // `configure` rebuilds the engine from the configuration, so the template
    // and the character counter are installed before the first turn runs.
    agent.configure(config);
    agent.set_model(Arc::new(model.clone()));
    agent
}

/// Runs one streaming turn and returns every frame it published.
async fn collect_turn(
    agent: &ReActAgent,
    prompt: &str,
    session: &Arc<dyn Session>,
) -> Vec<AgentChunk> {
    collect_bounded(agent.stream(json!(prompt), Some(Arc::clone(session))), 64)
        .await
        .into_iter()
        .map(|frame| frame.unwrap())
        .collect()
}

/// The `context.usage` payloads of `frames`, in arrival order.
fn context_usage_payloads(frames: &[AgentChunk]) -> Vec<&Value> {
    frames
        .iter()
        .filter(|frame| frame.kind == CONTEXT_USAGE)
        .map(|frame| &frame.payload)
        .collect()
}

/// The fixed `context.usage` payload of one fresh invocation: the loop starts
/// at iteration one every time, and `window_size` is null because no window
/// limit was requested.
fn assert_usage_payload(payload: &Value, message_count: usize, token_count: usize) {
    assert_eq!(payload["iteration"], json!(1), "iteration: {payload}");
    assert_eq!(payload["message_count"], json!(message_count));
    assert_eq!(payload["tool_count"], json!(0));
    assert_eq!(payload["token_count"], json!(token_count));
    assert_eq!(payload["overflow"], json!(false));
    assert!(
        payload["window_size"].is_null(),
        "no window limit was requested: {payload}"
    );
    assert_eq!(payload["result_type"], json!("answer"));
}

// Turn one sends the system prompt (3 chars + 4 framing) and the user message
// `hi` (2 + 4): 13 tokens. Turn two adds the observed exchange: `hi` (6),
// `first answer` (12 + 4) and `again` (5 + 4) after the same 7-token system
// message, so the prepared window costs 38 tokens.
#[tokio::test]
async fn each_streamed_turn_emits_one_context_usage_frame_and_mirrors_it() {
    let model = ScriptedModel::builder()
        .stream_text("first answer")
        .stream_text("second answer")
        .build();
    let agent = usage_agent(&model);
    assert_eq!(
        agent.context_engine().config().token_counter.kind,
        TokenCounterKind::Characters
    );
    let session = agent_session(SESSION_ID);
    // Subscribe before the first run so both turns are mirrored.
    let mut events = session.subscribe();

    let first = collect_turn(&agent, "hi", &session).await;
    let first_usage = context_usage_payloads(&first);
    assert_eq!(
        first_usage.len(),
        1,
        "one context.usage frame per turn: {first:?}"
    );
    assert_usage_payload(first_usage[0], 2, 13);
    let first_answer = first.last().expect("the turn streamed a terminal frame");
    assert_eq!(first_answer.kind, ANSWER);
    assert_eq!(first_answer.payload["output"], json!("first answer"));

    let second = collect_turn(&agent, "again", &session).await;
    let second_usage = context_usage_payloads(&second);
    assert_eq!(
        second_usage.len(),
        1,
        "one context.usage frame per turn: {second:?}"
    );
    assert_usage_payload(second_usage[0], 4, 38);
    let second_answer = second.last().expect("the turn streamed a terminal frame");
    assert_eq!(second_answer.kind, ANSWER);
    assert_eq!(second_answer.payload["output"], json!("second answer"));

    // A supplied session is mirrored frame by frame; drain the hub until both
    // turns' terminal frames have arrived.
    let mut mirrored = Vec::new();
    let mut answers = 0;
    within("the session mirror", async {
        while answers < 2 {
            match events.next().await {
                Some(Ok(StreamEvent::Item(item))) if item.item_type == CONTEXT_USAGE => {
                    mirrored.push(item.payload);
                }
                Some(Ok(StreamEvent::Item(item))) if item.item_type == ANSWER => {
                    answers += 1;
                }
                Some(Ok(_)) => {}
                Some(Err(error)) => panic!("the session stream failed: {error}"),
                None => break,
            }
        }
    })
    .await;

    assert_eq!(
        mirrored.len(),
        2,
        "one mirrored usage item per turn: {mirrored:?}"
    );
    assert_eq!(mirrored[0], *first_usage[0]);
    assert_eq!(mirrored[1], *second_usage[0]);
}

// The model's own accounting rides beside the context statistics: the window
// frame is published before the call, the content delta next, the accumulated
// usage at the end of the stream, and the terminal answer last. No frame index
// is asserted because each producer numbers its own frames independently.
#[tokio::test]
async fn a_usage_carrying_model_response_adds_one_llm_usage_frame() {
    let model = ScriptedModel::builder()
        .completion(MockCompletion::text("ok").with_usage(Usage::new(3, 2)))
        .build();
    let agent = usage_agent(&model);
    let session = agent_session("usage-frames");

    let frames = collect_turn(&agent, "hi", &session).await;
    let kinds = frames
        .iter()
        .map(|frame| frame.kind.as_str())
        .collect::<Vec<_>>();
    assert_eq!(
        kinds,
        [CONTEXT_USAGE, LLM_OUTPUT, LLM_USAGE, ANSWER],
        "deterministic frame order for a single text turn: {frames:?}"
    );
    assert_eq!(model.call_count(), 1, "one model call per turn");

    let usage = context_usage_payloads(&frames);
    assert_eq!(usage.len(), 1);
    assert_usage_payload(usage[0], 2, 13);

    let usage_frame = frames
        .iter()
        .find(|frame| frame.kind == LLM_USAGE)
        .expect("the stream reports the model usage");
    assert_eq!(
        usage_frame.payload["usage_metadata"]["input_tokens"],
        json!(3)
    );
    assert_eq!(
        usage_frame.payload["usage_metadata"]["output_tokens"],
        json!(2)
    );
    assert_eq!(
        usage_frame.payload["usage_metadata"]["total_tokens"],
        json!(5)
    );
    assert_eq!(
        usage_frame.payload["usage_metadata"]["cache_tokens"],
        json!(0)
    );
    assert_eq!(usage_frame.payload["result_type"], json!("answer"));

    // The payload is consumable by the provider-usage seam: three prompt
    // tokens and no cache counters means "not reported", never a fabricated
    // miss.
    let normalized = request_usage_from_metadata(&usage_frame.payload["usage_metadata"]);
    assert_eq!(normalized.input_tokens, Some(3));
    assert_eq!(normalized.cache_read_tokens, None);
    assert_eq!(normalized.status, STATUS_NOT_REPORTED);
    assert_eq!(normalized.source, NOT_REPORTED);

    let answer = frames.last().expect("the stream has a terminal frame");
    assert_eq!(answer.kind, ANSWER);
    assert_eq!(answer.payload["output"], json!("ok"));
}

// A model response that carries usage metadata (prompt/completion counters plus
// the legacy `cache_tokens` the seam still understands) normalizes into the
// canonical request contract, and one recorded request rolls up into the
// session cache aggregate under the aggregation key.
#[tokio::test]
async fn provider_usage_metadata_normalizes_the_cache_counters() {
    let mut usage = Usage::new(10, 4);
    usage.cache_tokens = 3;
    let metadata = serde_json::to_value(&usage).expect("usage serializes");
    let request = request_usage_from_metadata(&metadata);

    assert_eq!(request.input_tokens, Some(10));
    // `cache_tokens` is the legacy read counter; the modern field is absent.
    assert_eq!(request.cache_read_tokens, Some(3));
    assert_eq!(request.cache_miss_tokens, Some(7));
    assert!(request.miss_tokens_derived, "the miss was derived");
    assert_eq!(request.cache_write_tokens, None);
    assert_eq!(request.status, STATUS_PARTIAL_HIT);
    assert_eq!(request.source, SOURCE_PROVIDER_USAGE);
    assert!(!request.authoritative);
    assert!(request.invalid_reason.is_none(), "{request:?}");
    let hit_rate = request.hit_rate.expect("a consistent report has a rate");
    assert!((hit_rate - 0.3).abs() < 1e-9, "hit rate: {hit_rate}");

    let key = CacheAggregationKey::new(SESSION_ID, "scripted", "scripted-model");
    let mut aggregator = SessionKVCacheAggregator::new();
    let snapshot = aggregator.record("request-1", &key, &request);

    assert_eq!(snapshot.scope, "session");
    assert_eq!(snapshot.aggregation_method, "token_weighted");
    assert_eq!(snapshot.aggregation_key, key);
    assert_eq!(snapshot.aggregation_key.session_id, SESSION_ID);
    assert_eq!(snapshot.aggregation_key.provider, "scripted");
    assert_eq!(snapshot.aggregation_key.model, "scripted-model");
    assert_eq!(snapshot.aggregation_key.cache_mode, "provider");
    assert_eq!(snapshot.aggregation_key.cache_scope, "session");
    assert_eq!(snapshot.aggregation_key.deployment, "");
    assert_eq!(snapshot.calls_total, 1);
    assert_eq!(snapshot.calls_observed, 1);
    assert_eq!(snapshot.calls_partial, 0);
    assert_eq!(snapshot.calls_not_reported, 0);
    assert_eq!(snapshot.input_tokens_total, 10);
    assert_eq!(snapshot.cache_read_tokens_total, 3);
    assert_eq!(snapshot.cache_miss_tokens_total, 7);
    assert_eq!(snapshot.cache_write_tokens_total, None);
    assert_eq!(snapshot.cache_write_coverage, "none");
    assert_eq!(snapshot.status, STATUS_PARTIAL_HIT);
    assert_eq!(snapshot.data_quality, "derived");
    assert_eq!(aggregator.scope_count(), 1);
    assert!(!aggregator.is_empty());
}

// Two turns in one scope accumulate token-weighted totals, while each request
// keeps only its own counters and a repeated request id is idempotent.
#[tokio::test]
async fn two_turns_accumulate_in_the_session_aggregator() {
    // Turn one: the legacy `cache_tokens` field, miss derived.
    let mut legacy = Usage::new(10, 4);
    legacy.cache_tokens = 3;
    let first = request_usage_from_metadata(&serde_json::to_value(&legacy).unwrap());
    // Turn two: the modern cache counters, complete and consistent.
    let second = request_usage_from_metadata(&json!({
        "input_tokens": 20,
        "cache_read_tokens": 8,
        "cache_miss_tokens": 12,
        "cache_write_tokens": 2,
    }));
    assert_eq!(second.status, STATUS_PARTIAL_HIT);
    assert_eq!(second.cache_miss_tokens, Some(12));
    assert!(!second.miss_tokens_derived);

    let key = CacheAggregationKey::new(SESSION_ID, "scripted", "scripted-model");
    let mut aggregator = SessionKVCacheAggregator::new();
    let after_first = aggregator.record("turn-1", &key, &first);
    assert_eq!(after_first.calls_total, 1);
    assert_eq!(after_first.input_tokens_total, 10);
    assert_eq!(after_first.data_quality, "derived");

    let after_second = aggregator.record("turn-2", &key, &second);
    assert_eq!(after_second.calls_total, 2);
    assert_eq!(after_second.calls_observed, 2);
    assert_eq!(after_second.input_tokens_total, 30);
    assert_eq!(after_second.cache_read_tokens_total, 11);
    assert_eq!(after_second.cache_miss_tokens_total, 19);
    // Only one of the two calls reported a write counter.
    assert_eq!(after_second.cache_write_tokens_total, None);
    assert_eq!(after_second.cache_write_coverage, "partial");
    assert_eq!(after_second.status, STATUS_PARTIAL_HIT);
    assert_eq!(after_second.data_quality, "mixed");
    let weighted = after_second
        .weighted_hit_rate
        .expect("tokens were observed");
    assert!(
        (weighted - 11.0 / 30.0).abs() < 1e-9,
        "weighted hit rate: {weighted}"
    );

    // A per-request report is not cumulative: each one still carries only its
    // own counters.
    assert_eq!(first.input_tokens, Some(10));
    assert_eq!(first.cache_read_tokens, Some(3));
    assert_eq!(second.input_tokens, Some(20));
    assert_eq!(second.cache_read_tokens, Some(8));

    // Recording a request id twice cannot double-count it.
    let repeated = aggregator.record("turn-1", &key, &first);
    assert_eq!(repeated.calls_total, 2);
    assert_eq!(repeated.input_tokens_total, 30);
    assert_eq!(aggregator.scope_count(), 1);
}
