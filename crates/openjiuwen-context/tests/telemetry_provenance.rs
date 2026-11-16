#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//! Token provenance of the `context.compression_state` telemetry.
//!
//! A record's `tokens` value is only useful if the reader knows whether it is a
//! count or an estimate, so every metric states it: `estimated` says the number
//! came from an approximating strategy and `tokens_source` names that strategy
//! when it can be named.
//! This suite pins the three rungs a recorder can be on —
//! the `ceil(characters / 3)` fallback of a recorder without a counter, a
//! declared approximate counter, and a native tokenizer that counts exactly —
//! plus the engine wiring that makes a record's `before`/`after`/`saved` numbers
//! equal to its per-role `statistic`, the serde tolerance of the new keys in both
//! directions, and the correlation warning of a non-object request `extra`.
//!
//! Everything is offline and deterministic: counters are approximate or a tiny
//! in-memory BPE artifact, models are [`openjiuwen_test_support::ScriptedModel`],
//! the session is an in-process `AgentSession`, and tracing events are captured
//! by a local subscriber instead of a log file.
use std::fmt;
use std::sync::{Arc, Mutex};

use openjiuwen_context::telemetry::recorder::{ContextProcessorStateRecorder, TokenProvenance};
use openjiuwen_context::telemetry::{
    CompressionStatePhase, CompressionStateStatus, ContextCompressionState,
    ContextProcessorStateInput, RecordingSink,
};
use openjiuwen_context::tokenizers::NativeTokenizer;
use openjiuwen_context::{
    count_messages, stamp_current_operation, with_operation, ContextConfig, ContextEngine,
    ContextOptions, PrepareOptions, SelectedTokenizer, TokenCounterConfig, TokenCounterKind,
};
use openjiuwen_foundation::llm::{Model, ModelRequestConfig};
use openjiuwen_foundation::prompt::{ApproxTokenCounter, BaseMessage};
use openjiuwen_session::{AgentSession, Session};
use openjiuwen_test_support::ScriptedModel;
use tracing::field::{Field, Visit};
use tracing::span::{Attributes, Id, Record};
use tracing::{Event, Level, Metadata, Subscriber};

/// The wire name of the recorder's string-length fallback.
const STRING_LENGTH_SOURCE: &str = "string_length_estimate";

/// A tiny BPE artifact: `a`, `b`, `c` merge into `abc`, nothing else is known.
const BPE_FIXTURE: &str = r#"{
    "name": "tiny-bpe",
    "model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "c": 2, "ab": 3, "abc": 4}, "merges": ["a b", "ab c"]},
    "pre_tokenizer": {"type": "Whitespace"}
}"#;

// ---------------------------------------------------------------------------
// Tracing capture
// ---------------------------------------------------------------------------

/// A [`Subscriber`] that keeps the level and rendered message of every event.
#[derive(Clone, Default)]
struct EventCapture {
    events: Arc<Mutex<Vec<(Level, String)>>>,
}

impl EventCapture {
    /// An empty capture.
    fn new() -> Self {
        Self::default()
    }

    /// The rendered messages of every `warn` event, in emission order.
    fn warnings(&self) -> Vec<String> {
        self.events
            .lock()
            .unwrap()
            .iter()
            .filter(|(level, _)| *level == Level::WARN)
            .map(|(_, message)| message.clone())
            .collect()
    }
}

impl Subscriber for EventCapture {
    fn enabled(&self, _metadata: &Metadata<'_>) -> bool {
        true
    }

    fn new_span(&self, _span: &Attributes<'_>) -> Id {
        Id::from_u64(1)
    }

    fn record(&self, _span: &Id, _values: &Record<'_>) {}

    fn record_follows_from(&self, _span: &Id, _follows: &Id) {}

    fn event(&self, event: &Event<'_>) {
        let mut visitor = MessageVisitor(String::new());
        event.record(&mut visitor);
        self.events
            .lock()
            .unwrap()
            .push((*event.metadata().level(), visitor.0));
    }

    fn enter(&self, _span: &Id) {}

    fn exit(&self, _span: &Id) {}
}

/// Renders every field of an event, so a warning's structured fields are
/// asserted alongside its message.
struct MessageVisitor(String);

impl Visit for MessageVisitor {
    fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
        use std::fmt::Write as _;
        let _ = write!(self.0, " {}={value:?}", field.name());
    }
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// Two messages whose counted total differs from `ceil(characters / 3)`.
fn history() -> Vec<BaseMessage> {
    vec![BaseMessage::user("hi"), BaseMessage::assistant("hello")]
}

/// A `started` pass over `before`, with no caller-supplied statistics.
fn started_input(before: &[BaseMessage]) -> ContextProcessorStateInput<'_> {
    ContextProcessorStateInput::new(
        "op-1",
        CompressionStateStatus::Started,
        CompressionStatePhase::GetContextWindow,
        "truncation",
        before,
    )
}

/// A `completed` pass that removed the first message.
fn completed_input<'a>(
    before: &'a [BaseMessage],
    after: &'a [BaseMessage],
) -> ContextProcessorStateInput<'a> {
    ContextProcessorStateInput::new(
        "op-2",
        CompressionStateStatus::Completed,
        CompressionStatePhase::ActiveCompress,
        "truncation",
        before,
    )
    .with_after(after)
}

/// The one-character-per-token counter of the exactness assertions.
fn characters_counter() -> ApproxTokenCounter {
    ApproxTokenCounter::with_chars_per_token(1)
}

// ---------------------------------------------------------------------------
// 1. The fallback rung
// ---------------------------------------------------------------------------

#[test]
fn a_recorder_without_a_counter_flags_the_string_length_estimate_and_warns_once() {
    let capture = EventCapture::new();
    let recorder = ContextProcessorStateRecorder::new("session-1", "main");
    let before = history();
    let after = vec![BaseMessage::assistant("hello")];

    // Three measurements run here (before, after, and the fallback statistic),
    // and the warning is per recorder instance, not per pass.
    let (started, completed) = tracing::subscriber::with_default(capture.clone(), || {
        (
            recorder.build_state(&started_input(&before)),
            recorder.build_state(&completed_input(&before, &after)),
        )
    });
    let third = recorder.build_state(&started_input(&before));

    let estimate_rung = ApproxTokenCounter::with_chars_per_token(3);
    assert!(started.before.estimated);
    assert_eq!(
        started.before.tokens_source.as_deref(),
        Some(STRING_LENGTH_SOURCE)
    );
    assert_eq!(
        started.before.tokens,
        count_messages(&estimate_rung, &before)
    );
    // The statistic is measured on the same rung, so the two totals agree.
    assert_eq!(started.statistic.total_tokens, started.before.tokens);
    assert!(started.statistic.estimated);
    assert!(third.before.estimated);

    let after_metric = completed.after.as_ref().expect("the pass has an after");
    assert!(after_metric.estimated);
    assert_eq!(
        after_metric.tokens_source.as_deref(),
        Some(STRING_LENGTH_SOURCE)
    );
    assert_eq!(after_metric.tokens, count_messages(&estimate_rung, &after));
    assert_eq!(completed.statistic.total_tokens, after_metric.tokens);
    // A saving inherited from two estimates is never published as exact.
    let saved = completed.saved.expect("the pass saved something");
    assert!(saved.estimated);
    assert_eq!(
        saved.tokens,
        i64::try_from(completed.before.tokens).unwrap()
            - i64::try_from(after_metric.tokens).unwrap()
    );

    let warnings = capture.warnings();
    assert_eq!(warnings.len(), 1, "one warning per recorder: {warnings:?}");
    assert!(warnings[0].contains("no token counter"), "{warnings:?}");
    assert!(warnings[0].contains(STRING_LENGTH_SOURCE), "{warnings:?}");
}

// ---------------------------------------------------------------------------
// 2. The declared-counter rungs
// ---------------------------------------------------------------------------

#[test]
fn a_declared_approximate_counter_is_still_reported_as_an_estimate() {
    let recorder = ContextProcessorStateRecorder::new("session-1", "main")
        .with_token_counter(Arc::new(characters_counter()))
        .with_token_provenance(TokenProvenance::estimated(
            "approx_token_counter",
            "estimated by the approximate characters-per-token counter",
        ));
    let before = history();
    let state = recorder.build_state(&started_input(&before));

    assert!(state.before.estimated);
    assert_eq!(
        state.before.tokens_source.as_deref(),
        Some("approx_token_counter")
    );
    assert_eq!(
        state.before.tokens,
        count_messages(&characters_counter(), &before)
    );
    assert_eq!(state.statistic.total_tokens, state.before.tokens);
}

#[test]
fn an_undeclared_counter_is_an_unnamed_estimate_rather_than_a_guess_at_exactness() {
    let recorder = ContextProcessorStateRecorder::new("session-1", "main")
        .with_token_counter(Arc::new(characters_counter()));
    assert!(recorder.token_provenance().is_estimated());
    assert_eq!(recorder.token_provenance().source(), None);

    let before = history();
    let state = recorder.build_state(&started_input(&before));
    assert!(state.before.estimated);
    assert_eq!(state.before.tokens_source, None);
    // An unknown source is omitted from the record instead of written as null.
    let value = serde_json::to_value(&state).unwrap();
    assert!(value["before"].get("tokens_source").is_none());
    assert_eq!(value["before"]["estimated"], true);
}

#[test]
fn a_native_tokenizer_reports_an_exact_count() {
    let native = NativeTokenizer::from_json_str(BPE_FIXTURE).expect("the fixture loads");
    let selected = SelectedTokenizer::from_native(
        Arc::new(native),
        Some("tiny-bpe".to_owned()),
        Some("tiny-bpe".to_owned()),
    );
    let provenance = TokenProvenance::of_selected(&selected);
    assert!(!provenance.is_estimated());
    assert_eq!(provenance.source(), Some("native_tokenizer"));

    let recorder = ContextProcessorStateRecorder::new("session-1", "main")
        .with_token_counter_provenance(Arc::new(selected), provenance);
    let before = history();
    let state = recorder.build_state(&started_input(&before));

    assert!(!state.before.estimated);
    assert_eq!(
        state.before.tokens_source.as_deref(),
        Some("native_tokenizer")
    );
    assert_eq!(
        state.before.tokens,
        count_messages(&selected_counter(), &before)
    );
}

/// The counter behind the native selection, rebuilt for the expected total.
fn selected_counter() -> SelectedTokenizer {
    let native = NativeTokenizer::from_json_str(BPE_FIXTURE).expect("the fixture loads");
    SelectedTokenizer::from_native(
        Arc::new(native),
        Some("tiny-bpe".to_owned()),
        Some("tiny-bpe".to_owned()),
    )
}

// ---------------------------------------------------------------------------
// 3. Engine wiring
// ---------------------------------------------------------------------------

#[tokio::test]
async fn the_engine_installs_its_counter_so_the_record_agrees_with_the_statistic() {
    let sink = Arc::new(RecordingSink::new());
    let recorder = Arc::new(
        ContextProcessorStateRecorder::new("engine-1", "main").with_sink(
            Arc::clone(&sink) as Arc<dyn openjiuwen_context::telemetry::CompressionStateSink>
        ),
    );
    // Nothing is declared before the engine runs.
    assert_eq!(recorder.token_provenance().source(), None);

    let config = ContextConfig {
        token_counter: TokenCounterConfig {
            kind: TokenCounterKind::Characters,
            chars_per_token: 1,
        },
        ..ContextConfig::default()
    };
    let engine = ContextEngine::new(config).with_recorder(Some(Arc::clone(&recorder)));
    let session: Arc<dyn Session> = Arc::new(AgentSession::new("engine-1"));
    let before = history();
    let context = engine
        .create_context_with(
            "main",
            ContextOptions {
                session: Some(session),
                history: before.clone(),
                ..ContextOptions::default()
            },
        )
        .await
        .unwrap();

    // The engine's counter (and its provenance) is installed on the recorder, so
    // a caller-set counter would have to be replaced for this to change.
    assert_eq!(
        recorder.token_provenance().source(),
        Some("characters_token_counter")
    );

    engine
        .prepare(&context, &PrepareOptions::default())
        .await
        .unwrap();

    let states = sink.states();
    assert!(!states.is_empty(), "the engine emitted no records");
    let counted = count_messages(&characters_counter(), &before);
    let estimated_fallback = count_messages(&ApproxTokenCounter::with_chars_per_token(3), &before);
    assert_ne!(
        counted, estimated_fallback,
        "the fixture cannot tell the rungs apart"
    );

    for state in &states {
        assert_eq!(state.before.messages, before.len());
        // The record counts with the engine's counter, not `ceil(chars / 3)`.
        assert_eq!(state.before.tokens, counted);
        assert!(state.before.estimated);
        assert_eq!(
            state.before.tokens_source.as_deref(),
            Some("characters_token_counter")
        );
        match state.after.as_ref() {
            // A terminal record's statistic describes the after history, and the
            // two totals are equal by construction.
            Some(after) => assert_eq!(state.statistic.total_tokens, after.tokens),
            None => assert_eq!(state.statistic.total_tokens, state.before.tokens),
        }
        if let (Some(after), Some(saved)) = (state.after.as_ref(), state.saved.as_ref()) {
            assert_eq!(
                saved.tokens,
                i64::try_from(state.before.tokens).unwrap() - i64::try_from(after.tokens).unwrap()
            );
            assert_eq!(saved.estimated, state.before.estimated || after.estimated);
        }
    }
}

#[tokio::test]
async fn a_caller_set_counter_on_the_recorder_is_never_replaced_by_the_engine() {
    let recorder = Arc::new(
        ContextProcessorStateRecorder::new("engine-2", "main").with_token_counter_provenance(
            Arc::new(characters_counter()),
            TokenProvenance::estimated("caller_counter", "the caller counts characters"),
        ),
    );
    // The engine's own configuration would install `approx_token_counter`.
    let engine =
        ContextEngine::new(ContextConfig::default()).with_recorder(Some(Arc::clone(&recorder)));
    let session: Arc<dyn Session> = Arc::new(AgentSession::new("engine-2"));
    engine
        .create_context_with(
            "main",
            ContextOptions {
                session: Some(session),
                history: history(),
                ..ContextOptions::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(
        recorder.token_provenance().source(),
        Some("caller_counter"),
        "the engine replaced a counter the caller declared"
    );
}

// ---------------------------------------------------------------------------
// 4. The wire contract
// ---------------------------------------------------------------------------

#[test]
fn records_round_trip_with_and_without_the_provenance_keys() {
    let before = history();
    let after = vec![BaseMessage::assistant("hello")];
    let recorder = ContextProcessorStateRecorder::new("session-1", "main");
    let state = recorder.build_state(&completed_input(&before, &after));

    let value = serde_json::to_value(&state).unwrap();
    assert_eq!(value["before"]["estimated"], true);
    assert_eq!(value["before"]["tokens_source"], STRING_LENGTH_SOURCE);
    assert_eq!(value["after"]["estimated"], true);
    assert_eq!(value["saved"]["estimated"], true);
    assert_eq!(
        serde_json::from_value::<ContextCompressionState>(value.clone()).unwrap(),
        state
    );

    // A legacy record of Python's shape carries neither key; it still loads, and
    // its numbers are reported as exact because nothing says otherwise.
    let mut legacy = value;
    for section in ["before", "after", "saved"] {
        let object = legacy[section].as_object_mut().unwrap();
        object.remove("estimated");
        object.remove("tokens_source");
    }
    let loaded: ContextCompressionState = serde_json::from_value(legacy).unwrap();
    assert!(!loaded.before.estimated);
    assert_eq!(loaded.before.tokens_source, None);
    assert!(!loaded.after.unwrap().estimated);
    assert!(!loaded.saved.unwrap().estimated);
    // The per-role statistic is tolerant in the same way.
    let legacy_stats: openjiuwen_context::ContextStats =
        serde_json::from_str(r#"{"total_messages": 2, "total_tokens": 15}"#).unwrap();
    assert!(!legacy_stats.estimated);
    assert!(legacy_stats.with_estimated(true).estimated);
}

// ---------------------------------------------------------------------------
// 5. Correlation warnings
// ---------------------------------------------------------------------------

#[tokio::test]
async fn a_non_object_extra_warns_and_the_model_call_still_runs() {
    let capture = EventCapture::new();
    let model = ScriptedModel::text("summary");
    let extra = serde_json::json!("not-an-object");
    let mut request = ModelRequestConfig::new("fake-model").with_extra(extra.clone());

    {
        let _guard = tracing::subscriber::set_default(capture.clone());
        with_operation("op-1", async {
            stamp_current_operation(&mut request);
            // The request is untouched, so the call below is the one the pass
            // would have made without the correlation.
            model
                .invoke(&[BaseMessage::user("compress this")], &request)
                .await
                .expect("the model call still runs")
        })
        .await;
    }

    assert_eq!(request.extra, extra, "the request extra was rewritten");
    assert_eq!(model.call_count(), 1);
    let warnings = capture.warnings();
    assert_eq!(
        warnings.len(),
        1,
        "one warning per dropped stamp: {warnings:?}"
    );
    assert!(
        warnings[0].contains("neither a JSON object nor null"),
        "{warnings:?}"
    );
    assert!(warnings[0].contains("context_operation_id"), "{warnings:?}");
    assert!(warnings[0].contains("a string"), "{warnings:?}");

    // An object or null extra takes the keys without a warning.
    let quiet = EventCapture::new();
    {
        let _guard = tracing::subscriber::set_default(quiet.clone());
        let mut request = ModelRequestConfig::new("fake-model");
        openjiuwen_context::apply_operation(
            &mut request,
            Some(&openjiuwen_context::CompressionOperation::new("op-2")),
        );
        assert_eq!(
            request.extra[openjiuwen_context::REQUEST_PURPOSE_KEY],
            "compaction"
        );
        assert_eq!(
            request.extra[openjiuwen_context::CONTEXT_OPERATION_ID_KEY],
            "op-2"
        );
    }
    assert_eq!(quiet.warnings(), Vec::<String>::new());
}
