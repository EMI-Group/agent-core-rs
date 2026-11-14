#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//! Engine-level wiring tests for the usage, telemetry and window accessors.
//!
//! The [`ContextEngine`] is the seam callers use, so these tests drive the
//! engine-owned plumbing end to end instead of the individual helpers:
//!
//! - the `context.usage` block [`PreparedContext::usage`] the engine fills when
//!   a [`ContextUsageAnalyzer`] is attached, and its absence otherwise;
//! - the context-window resolution ladder (`max_context_tokens`, policy
//!   override, configured map, built-in table, 200 000-token default) and that
//!   a policy-configured engine still prepares offline;
//! - `usage_report` / `usage_snapshot`, including provider reconciliation and
//!   the `Error::Config` a missing analyzer raises;
//! - the recorder phases of the three engine entry points (`observe`,
//!   `prepare`, `compress`) and the silent uninstrumented engine;
//! - the engine-owned KV-cache rollup (scope isolation, request-id
//!   idempotency, inconsistent calls, blank request ids);
//! - the KV-cache window diffing `detect_window_change` exposes;
//! - the compaction-operation correlation a compression pass stamps onto its
//!   model request and the usage it reports in its compression state.
//!
//! Everything is offline and deterministic: counters are approximate (or the
//! one-character-per-token `characters` counter), models are small fakes, and
//! no test asserts on a wall clock (the access instant is only compared
//! monotonically).
//!
//! [`ContextEngine`]: openjiuwen_context::ContextEngine
//! [`PreparedContext::usage`]: openjiuwen_context::PreparedContext::usage
//! [`ContextUsageAnalyzer`]: openjiuwen_context::ContextUsageAnalyzer
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::stream::BoxStream;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_context::{
    telemetry, CacheAggregationKey, CompressionStatePhase, CompressionStateSink,
    CompressionStateStatus, ContextCategory, ContextConfig, ContextEngine, ContextOptions,
    ContextProcessorStateRecorder, ContextUsageAnalyzer, ContextUsageSnapshot, ContextWindowChange,
    ContextWindowPolicy, ContextWindowSource, PrepareOptions, ProcessorConfig, RecordingSink,
    SummarizationConfig, CONTEXT_USAGE_EVENT_TYPE, DEFAULT_CONTEXT_TOKENS, DEFAULT_SESSION_ID,
    SOURCE_PROVIDER_USAGE,
};
use openjiuwen_foundation::llm::{ChatResponse, Choice, Model, ModelRequestConfig, StreamChunk};
use openjiuwen_foundation::prompt::{BaseMessage, Usage};

/// A model that records the request configuration of every call and answers
/// with `usage`.
struct CapturingModel {
    /// Usage the fake provider reports.
    usage: Usage,
    /// Request configurations of the calls made so far.
    requests: Mutex<Vec<ModelRequestConfig>>,
}

impl CapturingModel {
    /// A model reporting `usage`.
    fn new(usage: Usage) -> Self {
        Self {
            usage,
            requests: Mutex::new(Vec::new()),
        }
    }

    /// The request configurations observed so far, oldest first.
    fn requests(&self) -> Vec<ModelRequestConfig> {
        self.requests.lock().unwrap().clone()
    }
}

#[async_trait]
impl Model for CapturingModel {
    fn provider(&self) -> &'static str {
        "capture"
    }

    fn model_name(&self) -> &'static str {
        "capture-model"
    }

    async fn invoke(
        &self,
        _messages: &[BaseMessage],
        request: &ModelRequestConfig,
    ) -> Result<ChatResponse> {
        self.requests.lock().unwrap().push(request.clone());
        Ok(ChatResponse {
            id: "capture-1".to_owned(),
            model: self.model_name().to_owned(),
            choices: vec![Choice {
                index: 0,
                message: BaseMessage::assistant("condensed summary"),
                finish_reason: None,
            }],
            usage: self.usage.clone(),
        })
    }

    fn stream(
        &self,
        _messages: Vec<BaseMessage>,
        _request: ModelRequestConfig,
    ) -> BoxStream<'static, Result<StreamChunk>> {
        Box::pin(futures::stream::empty())
    }
}

/// A config whose only processor summarises the older round with the model.
fn summarization_config() -> ContextConfig {
    ContextConfig {
        processors: vec![ProcessorConfig::Summarization(SummarizationConfig {
            keep_last_rounds: 1,
            keep_recent_messages: 0,
            min_messages_to_summarize: 2,
            ..SummarizationConfig::default()
        })],
        ..ContextConfig::default()
    }
}

/// A two-round history: the older round is the summarisation candidate.
fn two_round_history() -> Vec<BaseMessage> {
    vec![
        BaseMessage::user("first question"),
        BaseMessage::assistant("first answer"),
        BaseMessage::user("second question"),
        BaseMessage::assistant("second answer"),
    ]
}

/// A recorder collecting its records in `sink`.
fn attach_recorder(
    session_id: &str,
    context_id: &str,
    sink: &Arc<RecordingSink>,
) -> Arc<ContextProcessorStateRecorder> {
    let sink: Arc<dyn CompressionStateSink> = sink.clone();
    Arc::new(ContextProcessorStateRecorder::new(session_id, context_id).with_sink(sink))
}

// ---------------------------------------------------------------------------
// 1. Usage population
// ---------------------------------------------------------------------------

#[tokio::test]
async fn prepare_fills_the_usage_block_only_when_an_analyzer_is_configured() {
    let engine = ContextEngine::new(ContextConfig::default())
        .with_usage_analyzer(Some(Arc::new(ContextUsageAnalyzer::default())));
    let context = engine.create_context("main", None).await.unwrap();
    engine
        .observe(&context, BaseMessage::user("hello, explain the engine"))
        .await
        .unwrap();
    engine
        .observe(
            &context,
            BaseMessage::assistant("the engine prepares windows"),
        )
        .await
        .unwrap();
    let options = PrepareOptions {
        system_messages: vec![BaseMessage::system("you are a careful assistant")],
        ..PrepareOptions::default()
    };
    let prepared = engine.prepare(&context, &options).await.unwrap();

    let usage = prepared.usage.as_ref().expect("the usage block is filled");
    assert_eq!(usage.event_type, CONTEXT_USAGE_EVENT_TYPE);
    // The four categories are always present, even an empty `skills` one.
    for category in ContextCategory::ALL {
        assert!(
            usage.parts.contains_key(category.as_str()),
            "missing category {}",
            category.as_str()
        );
    }
    // The parts are measured over exactly the prepared window, so they sum to
    // its token count and line up with the per-role statistics.
    let part_total: u64 = usage.parts.values().map(|part| part.tokens).sum();
    let window_tokens = u64::try_from(prepared.token_count).unwrap();
    assert_eq!(part_total, window_tokens);
    assert_eq!(usage.context_window.tokens, window_tokens);
    assert_eq!(
        part_total,
        u64::try_from(prepared.stats.total_tokens).unwrap()
    );
    assert_eq!(
        usage.parts[ContextCategory::SystemPrompt.as_str()].tokens,
        u64::try_from(prepared.stats.system_tokens).unwrap()
    );
    assert_eq!(
        usage.parts[ContextCategory::Messages.as_str()].tokens,
        u64::try_from(
            prepared.stats.user_tokens
                + prepared.stats.assistant_tokens
                + prepared.stats.tool_tokens
        )
        .unwrap()
    );
    assert_eq!(usage.parts[ContextCategory::Tools.as_str()].tokens, 0);
    assert_eq!(
        usage.context_stats.as_ref(),
        Some(&prepared.stats),
        "the snapshot carries the statistics it measured"
    );

    // The snapshot round-trips through JSON.
    let value = serde_json::to_value(usage).unwrap();
    let restored: ContextUsageSnapshot = serde_json::from_value(value).unwrap();
    assert_eq!(&restored, usage);

    // Without an analyzer the prepared payload is identical apart from `usage`.
    let plain = ContextEngine::new(ContextConfig::default());
    let plain_context = plain.create_context("main", None).await.unwrap();
    plain
        .observe(
            &plain_context,
            BaseMessage::user("hello, explain the engine"),
        )
        .await
        .unwrap();
    plain
        .observe(
            &plain_context,
            BaseMessage::assistant("the engine prepares windows"),
        )
        .await
        .unwrap();
    let plain_prepared = plain.prepare(&plain_context, &options).await.unwrap();
    assert!(plain_prepared.usage.is_none());
    assert_eq!(plain_prepared.messages, prepared.messages);
    assert_eq!(plain_prepared.token_count, prepared.token_count);
    assert_eq!(plain_prepared.stats, prepared.stats);
}

// ---------------------------------------------------------------------------
// 2. Window resolution ladder
// ---------------------------------------------------------------------------

#[tokio::test]
async fn the_window_resolution_ladder_reports_the_winning_rung() {
    // An explicit max_context_tokens wins even when the policy says otherwise.
    let engine = ContextEngine::new(ContextConfig {
        max_context_tokens: Some(4_096),
        context_window_policy: ContextWindowPolicy::new()
            .with_global_tokens(999)
            .with_model_override(777),
        ..ContextConfig::default()
    });
    let resolved = engine.resolved_window();
    assert_eq!(resolved.tokens, 4_096);
    assert_eq!(resolved.source, ContextWindowSource::Global);

    // The selected-model override wins over the configured map.
    let engine = ContextEngine::new(ContextConfig {
        model_name: Some("gpt-test-model".to_owned()),
        context_window_policy: ContextWindowPolicy::new()
            .with_model_override(24_000)
            .with_model_window("gpt-test-model", 8_000),
        ..ContextConfig::default()
    });
    let resolved = engine.resolved_window();
    assert_eq!(resolved.tokens, 24_000);
    assert_eq!(resolved.source, ContextWindowSource::Override);

    // A configured map entry is matched case-insensitively.
    let engine = ContextEngine::new(ContextConfig {
        model_name: Some("gpt-test-model".to_owned()),
        context_window_policy: ContextWindowPolicy::new()
            .with_model_window("GPT-Test-Model", 8_000),
        ..ContextConfig::default()
    });
    let resolved = engine.resolved_window();
    assert_eq!(resolved.tokens, 8_000);
    assert_eq!(resolved.source, ContextWindowSource::ConfiguredModel);

    // An unknown model falls back to the built-in table, then the default.
    let engine = ContextEngine::new(ContextConfig {
        context_window_policy: ContextWindowPolicy::new().with_model("gpt-4o"),
        ..ContextConfig::default()
    });
    let resolved = engine.resolved_window();
    assert_eq!(resolved.tokens, 128_000);
    assert_eq!(resolved.source, ContextWindowSource::BuiltIn);

    let engine = ContextEngine::new(ContextConfig {
        context_window_policy: ContextWindowPolicy::new().with_model("acme/unknown-model"),
        ..ContextConfig::default()
    });
    let resolved = engine.resolved_window();
    assert_eq!(resolved.tokens, DEFAULT_CONTEXT_TOKENS);
    assert_eq!(resolved.source, ContextWindowSource::Default);
    assert!(resolved.is_default());
}

#[tokio::test]
async fn a_policy_configured_engine_still_prepares_offline() {
    let engine = ContextEngine::new(ContextConfig {
        model_name: Some("gpt-4o".to_owned()),
        context_window_policy: ContextWindowPolicy::new().with_model_window("gpt-4o", 5_000),
        ..ContextConfig::default()
    });
    assert_eq!(engine.resolved_window().tokens, 5_000);

    let context = engine.create_context("main", None).await.unwrap();
    assert_eq!(context.resolved_window().tokens, 5_000);
    engine
        .observe(&context, BaseMessage::user("hello"))
        .await
        .unwrap();
    let prepared = engine
        .prepare(&context, &PrepareOptions::default())
        .await
        .unwrap();
    assert_eq!(prepared.messages.len(), 1);
    assert!(!prepared.overflow);
}

// ---------------------------------------------------------------------------
// 3. usage_report / usage_snapshot
// ---------------------------------------------------------------------------

#[tokio::test]
async fn usage_reports_reconcile_with_the_provider_total_and_need_an_analyzer() {
    let engine = ContextEngine::new(ContextConfig::default())
        .with_usage_analyzer(Some(Arc::new(ContextUsageAnalyzer::default())));
    let context = engine.create_context("main", None).await.unwrap();
    engine
        .observe(&context, BaseMessage::user("explain the engine wiring"))
        .await
        .unwrap();
    engine
        .observe(
            &context,
            BaseMessage::assistant("the engine resolves a window per context"),
        )
        .await
        .unwrap();
    let prepared = engine
        .prepare(&context, &PrepareOptions::default())
        .await
        .unwrap();
    let local_total = u64::try_from(prepared.token_count).unwrap();

    // Without a provider total the report is the frozen local measurement.
    let local_report = engine.usage_report(&context, None).await.unwrap();
    assert_eq!(local_report.context_id, "main");
    assert_eq!(local_report.context_window.tokens, local_total);
    assert!(!local_report.measurement.authoritative_total);
    assert_eq!(local_report.parts.len(), 4);

    // A supplied provider total is authoritative and the parts sum to it.
    let provider_total = local_total + 37;
    let report = engine
        .usage_report(&context, Some(provider_total))
        .await
        .unwrap();
    assert!(report.measurement.authoritative_total);
    assert_eq!(report.context_window.tokens, provider_total);
    assert_eq!(report.context_window.tokens_source, SOURCE_PROVIDER_USAGE);
    assert_eq!(
        report.parts.values().map(|part| part.tokens).sum::<u64>(),
        provider_total
    );
    assert_eq!(report.context_window.reconciliation_delta_tokens, Some(37));

    // The snapshot reconciles the same way and round-trips through JSON.
    let snapshot = engine
        .usage_snapshot(&context, Some(provider_total))
        .await
        .unwrap();
    assert!(snapshot.measurement.authoritative_total);
    assert_eq!(snapshot.context_window.tokens, provider_total);
    assert_eq!(
        snapshot.parts.values().map(|part| part.tokens).sum::<u64>(),
        provider_total
    );
    let json = serde_json::to_value(&snapshot).unwrap();
    let restored: ContextUsageSnapshot = serde_json::from_value(json).unwrap();
    assert_eq!(restored, snapshot);

    // Without an analyzer both entry points refuse with the documented error.
    let plain = ContextEngine::new(ContextConfig::default());
    let plain_context = plain.create_context("main", None).await.unwrap();
    let error = plain.usage_report(&plain_context, None).await.unwrap_err();
    assert!(
        matches!(&error, Error::Config(message) if message.contains("no context usage analyzer")),
        "{error}"
    );
    let error = plain
        .usage_snapshot(&plain_context, None)
        .await
        .unwrap_err();
    assert!(
        matches!(&error, Error::Config(message) if message.contains("no context usage analyzer")),
        "{error}"
    );
}

// ---------------------------------------------------------------------------
// 4. Recorder phase wiring
// ---------------------------------------------------------------------------

#[tokio::test]
async fn engine_entry_points_record_their_phase_and_go_silent_without_a_recorder() {
    let sink = Arc::new(RecordingSink::new());
    let recorder = attach_recorder(DEFAULT_SESSION_ID, "main", &sink);
    let engine = ContextEngine::new(ContextConfig::default()).with_recorder(Some(recorder));
    let context = engine.create_context("main", None).await.unwrap();
    assert!(context.recorder().is_some());

    engine
        .observe(&context, BaseMessage::user("hello"))
        .await
        .unwrap();
    engine
        .observe(&context, BaseMessage::assistant("hi"))
        .await
        .unwrap();
    engine
        .prepare(&context, &PrepareOptions::default())
        .await
        .unwrap();
    let report = engine.compress(&context).await.unwrap();
    assert!(
        !report.is_compressed(),
        "a tiny history compresses to a noop"
    );

    let states = sink.states();
    assert_eq!(states.len(), sink.statuses().len());
    assert!(
        states.iter().all(|state| !state.processor.is_empty()),
        "every record names its processor"
    );
    // Every processor pass is bracketed by a `started` record and its outcome.
    let statuses = sink.statuses();
    assert_eq!(statuses.len() % 2, 0, "{statuses:?}");
    for pair in statuses.chunks(2) {
        assert_eq!(pair[0], CompressionStateStatus::Started);
        assert!(
            matches!(
                pair[1],
                CompressionStateStatus::Completed | CompressionStateStatus::Noop
            ),
            "unexpected outcome {:?}",
            pair[1]
        );
    }
    // The three engine entry points map to the three phases, in call order,
    // and the default pipeline holds two processors: two appends, one prepare
    // and one compression pass.
    let phases: Vec<CompressionStatePhase> = states
        .iter()
        .filter(|state| state.status == CompressionStateStatus::Started)
        .map(|state| state.phase)
        .collect();
    assert_eq!(
        phases,
        vec![
            CompressionStatePhase::AddMessages,
            CompressionStatePhase::AddMessages,
            CompressionStatePhase::AddMessages,
            CompressionStatePhase::AddMessages,
            CompressionStatePhase::GetContextWindow,
            CompressionStatePhase::GetContextWindow,
            CompressionStatePhase::ActiveCompress,
            CompressionStatePhase::ActiveCompress,
        ]
    );

    // The same engine shape without a recorder emits nothing at all.
    let silent_sink = Arc::new(RecordingSink::new());
    let unattached = attach_recorder(DEFAULT_SESSION_ID, "main", &silent_sink);
    let engine = ContextEngine::new(ContextConfig::default());
    let context = engine.create_context("main", None).await.unwrap();
    assert!(context.recorder().is_none());
    engine
        .observe(&context, BaseMessage::user("hello"))
        .await
        .unwrap();
    engine
        .observe(&context, BaseMessage::assistant("hi"))
        .await
        .unwrap();
    engine
        .prepare(&context, &PrepareOptions::default())
        .await
        .unwrap();
    engine.compress(&context).await.unwrap();
    assert!(silent_sink.is_empty());
    assert_eq!(unattached.history().len(), 0);
}

// ---------------------------------------------------------------------------
// 5. KV-cache rollup
// ---------------------------------------------------------------------------

#[tokio::test]
async fn kv_cache_usage_is_scoped_idempotent_and_rejects_blank_request_ids() {
    let engine = ContextEngine::new(ContextConfig::default());
    let scope = CacheAggregationKey::new("session-1", "openai", "gpt-test");
    let valid = engine
        .record_provider_usage(
            scope.clone(),
            "req-1",
            &serde_json::json!({"input_tokens": 100, "cache_read_tokens": 80}),
        )
        .unwrap();
    assert_eq!(valid.calls_total, 1);
    assert_eq!(valid.calls_observed, 1);
    assert!((valid.weighted_hit_rate.unwrap() - 0.8).abs() < 1e-9);

    // An inconsistent report is counted as partial and excluded from the totals.
    let invalid = engine
        .record_provider_usage(
            scope.clone(),
            "req-bad",
            &serde_json::json!({"input_tokens": 10, "cache_read_tokens": 20}),
        )
        .unwrap();
    assert_eq!(invalid.calls_total, 2);
    assert_eq!(invalid.calls_observed, 1);
    assert_eq!(invalid.calls_partial, 1);
    assert_eq!(invalid.input_tokens_total, 100);
    assert_eq!(invalid.cache_read_tokens_total, 80);
    assert!((invalid.weighted_hit_rate.unwrap() - 0.8).abs() < 1e-9);

    // A repeated request id is idempotent: no calls_total bump, no new tokens.
    let repeated = engine
        .record_provider_usage(
            scope.clone(),
            "req-1",
            &serde_json::json!({"input_tokens": 999, "cache_read_tokens": 999}),
        )
        .unwrap();
    assert_eq!(repeated, invalid);
    assert_eq!(repeated.calls_total, 2);
    assert_eq!(repeated.input_tokens_total, 100);

    // A second scope key is independent.
    let other = CacheAggregationKey::new("session-2", "openai", "gpt-test");
    assert!(engine.kv_cache_usage(&other).is_none());
    let other_usage = engine
        .record_provider_usage(
            other.clone(),
            "req-1",
            &serde_json::json!({"input_tokens": 50, "cache_read_tokens": 50}),
        )
        .unwrap();
    assert_eq!(other_usage.calls_total, 1);
    assert_eq!(other_usage.calls_observed, 1);
    let scoped = engine.kv_cache_usage(&scope).expect("the scope has calls");
    assert_eq!(scoped.calls_total, 2);
    assert_eq!(scoped.calls_partial, 1);
    assert_eq!(scoped.aggregation_key, scope);

    // A blank request id would be deduplicated against every other blank id.
    let error = engine
        .record_provider_usage(scope, "  ", &serde_json::json!({}))
        .unwrap_err();
    assert!(
        matches!(&error, Error::Validation(message) if message.contains("request_id")),
        "{error}"
    );
}

// ---------------------------------------------------------------------------
// 6. Window diffing
// ---------------------------------------------------------------------------

#[tokio::test]
async fn window_diffing_reports_unchanged_appended_and_tool_changes() {
    let engine = ContextEngine::new(ContextConfig::default());
    let context = engine.create_context("main", None).await.unwrap();
    engine
        .observe(&context, BaseMessage::user("hello"))
        .await
        .unwrap();

    // Before the second preparation the diff compares the window with itself.
    let first = engine
        .prepare(&context, &PrepareOptions::default())
        .await
        .unwrap();
    assert_eq!(
        engine.detect_window_change(&context),
        ContextWindowChange::Unchanged {
            messages: first.messages.len()
        }
    );
    let first_access = context
        .last_context_window_access()
        .expect("the prepared window was accessed");

    // A user/assistant pair only grows the window: the whole prefix is reusable.
    engine
        .observe(&context, BaseMessage::user("how does prepare work?"))
        .await
        .unwrap();
    engine
        .observe(&context, BaseMessage::assistant("it assembles the window"))
        .await
        .unwrap();
    engine
        .prepare(&context, &PrepareOptions::default())
        .await
        .unwrap();
    let appended = engine.detect_window_change(&context);
    assert_eq!(
        appended,
        ContextWindowChange::Appended {
            prefix_messages: first.messages.len()
        }
    );
    assert_eq!(appended.reusable_prefix_len(), first.messages.len());
    assert!(!appended.invalidates_cache());
    let second_access = context
        .last_context_window_access()
        .expect("the second window was accessed");
    assert!(second_access >= first_access);

    // tool definitions serialize right after the system prompt, so any tool
    // change invalidates the message suffix.
    let options = PrepareOptions {
        tools: vec![serde_json::json!({"name": "read_file"})],
        ..PrepareOptions::default()
    };
    engine.prepare(&context, &options).await.unwrap();
    let tools_changed = engine.detect_window_change(&context);
    assert_eq!(
        tools_changed,
        ContextWindowChange::ToolsChanged {
            requires_suffix_invalidation: true
        }
    );
    assert_eq!(tools_changed.kind(), "tools_changed");
    assert_eq!(tools_changed.reusable_prefix_len(), 0);
    assert!(tools_changed.invalidates_cache());
    let third_access = context
        .last_context_window_access()
        .expect("the third window was accessed");
    assert!(third_access >= second_access);
}

// ---------------------------------------------------------------------------
// 7. Model-call correlation
// ---------------------------------------------------------------------------

#[tokio::test]
async fn a_compression_pass_stamps_the_operation_and_reports_its_usage() {
    // Inside a compaction-operation scope the summariser stamps the request.
    let model = Arc::new(CapturingModel::new(Usage::new(11, 5)));
    let handle: Arc<dyn Model> = model.clone();
    let engine = ContextEngine::new(summarization_config());
    let context = engine
        .create_context_with(
            "main",
            ContextOptions {
                history: two_round_history(),
                model: Some(handle),
                ..ContextOptions::default()
            },
        )
        .await
        .unwrap();
    let report = telemetry::with_operation("op-1", async { engine.compress(&context).await })
        .await
        .unwrap();
    assert!(report.is_compressed(), "the older round was summarised");

    let requests = model.requests();
    assert_eq!(requests.len(), 1);
    assert_eq!(
        requests[0].extra.get("request_purpose"),
        Some(&serde_json::json!("compaction"))
    );
    assert_eq!(
        requests[0].extra.get("context_operation_id"),
        Some(&serde_json::json!("op-1"))
    );
    // The scope is closed once the pass is over.
    assert!(telemetry::current_compression_operation_id().is_none());

    // Outside a scope the request carries no correlation at all.
    let outside = Arc::new(CapturingModel::new(Usage::new(1, 1)));
    let outside_handle: Arc<dyn Model> = outside.clone();
    let engine = ContextEngine::new(summarization_config());
    let context = engine
        .create_context_with(
            "main",
            ContextOptions {
                history: two_round_history(),
                model: Some(outside_handle),
                ..ContextOptions::default()
            },
        )
        .await
        .unwrap();
    assert!(engine.compress(&context).await.unwrap().is_compressed());
    let requests = outside.requests();
    assert_eq!(requests.len(), 1);
    assert!(requests[0].extra.get("request_purpose").is_none());
    assert!(requests[0].extra.get("context_operation_id").is_none());
}

#[tokio::test]
async fn recorded_compression_states_carry_the_usage_the_model_reports() {
    // A provider that reports usage: the terminal record carries it.
    let sink = Arc::new(RecordingSink::new());
    let recorder = attach_recorder(DEFAULT_SESSION_ID, "main", &sink);
    let model: Arc<dyn Model> = Arc::new(CapturingModel::new(Usage::new(11, 5)));
    let engine = ContextEngine::new(summarization_config()).with_recorder(Some(recorder));
    let context = engine
        .create_context_with(
            "main",
            ContextOptions {
                history: two_round_history(),
                model: Some(model),
                ..ContextOptions::default()
            },
        )
        .await
        .unwrap();
    assert!(engine.compress(&context).await.unwrap().is_compressed());

    let states = sink.states();
    let terminal = states.last().expect("the pass emitted a terminal record");
    assert_eq!(terminal.status, CompressionStateStatus::Completed);
    assert_eq!(terminal.phase, CompressionStatePhase::ActiveCompress);
    assert_eq!(terminal.processor, "summarization");
    let usage = terminal
        .compression_usage
        .as_ref()
        .expect("the model call usage is reported");
    assert_eq!(usage.calls, 1);
    assert_eq!(usage.input_tokens, 11);
    assert_eq!(usage.output_tokens, 5);
    assert_eq!(usage.total_tokens, 16);
    assert_eq!(usage.model_name, "capture-model");

    // A provider that reports nothing leaves the usage empty; zeros are never
    // fabricated into a measurement.
    let sink = Arc::new(RecordingSink::new());
    let recorder = attach_recorder(DEFAULT_SESSION_ID, "main", &sink);
    let silent: Arc<dyn Model> = Arc::new(CapturingModel::new(Usage::default()));
    let engine = ContextEngine::new(summarization_config()).with_recorder(Some(recorder));
    let context = engine
        .create_context_with(
            "main",
            ContextOptions {
                history: two_round_history(),
                model: Some(silent),
                ..ContextOptions::default()
            },
        )
        .await
        .unwrap();
    assert!(engine.compress(&context).await.unwrap().is_compressed());
    let terminal = sink.states().last().cloned().expect("a terminal record");
    assert_eq!(terminal.status, CompressionStateStatus::Completed);
    assert!(terminal.compression_usage.is_none());
}
