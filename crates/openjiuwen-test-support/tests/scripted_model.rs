//! Trait-level integration tests for the in-process scripted `Model` fake.
//!
//! These mirror `tests/unit_tests/fixtures/mock_llm.py::MockLLMModel` at the
//! `Model` trait boundary: an ordered queue of turns with no prompt matching, a
//! call history recorded at call time (so `call_count`/`call_history` match the
//! Python fake), and a reusable fallback once the script is exhausted.
//!
//! Everything here goes through the public API (`Model`, `ModelExt`, and the
//! `ScriptedModel` helpers an agent or workflow test would use), so the fake is
//! exercised exactly as consumers exercise it. Private helpers stay covered by
//! the unit tests beside `src/scripted_model.rs` and are not re-tested here.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::sync::Arc;
use std::time::Duration;

use futures::stream::BoxStream;
use futures::StreamExt;
use openjiuwen_common::error::{ErrorKind, Result};
use openjiuwen_foundation::llm::{FinishReason, Model, ModelExt, ModelRequestConfig, StreamChunk};
use openjiuwen_foundation::prompt::{BaseMessage, Usage};
use openjiuwen_test_support::{
    text_chunks, CallMode, MockCompletion, ScriptedError, ScriptedModel, ScriptedTurn,
};

/// Drains a model stream so its items can be asserted on.
async fn collect(mut stream: BoxStream<'static, Result<StreamChunk>>) -> Vec<Result<StreamChunk>> {
    let mut items = Vec::new();
    while let Some(item) = stream.next().await {
        items.push(item);
    }
    items
}

/// A request for `model_name` with every other field left at its default.
fn request_for(model_name: &str) -> ModelRequestConfig {
    ModelRequestConfig::new(model_name)
}

#[tokio::test]
async fn arc_dyn_dispatch_serves_turns_in_order() {
    let built = ScriptedModel::builder()
        .respond_text("first")
        .respond_text("second")
        .build();
    let model: Arc<dyn Model> = Arc::new(built);

    assert_eq!(model.provider(), "scripted");
    assert_eq!(model.model_name(), "scripted-model");

    let request = request_for("scripted-model");
    let first = model
        .invoke(&[BaseMessage::user("q1")], &request)
        .await
        .expect("first scripted turn");
    assert_eq!(first.id, "scripted-1");
    assert_eq!(first.message().expect("choice").text(), "first");

    let second = model
        .invoke(&[BaseMessage::user("q2")], &request)
        .await
        .expect("second scripted turn");
    assert_eq!(second.id, "scripted-2");
    assert_eq!(second.message().expect("choice").text(), "second");
}

#[tokio::test]
async fn model_ext_complete_streams_and_records_a_stream_call() {
    let scripted = ScriptedModel::builder().stream_text("streamed").build();
    let model: Arc<dyn Model> = Arc::new(scripted.clone());

    let turn = model
        .complete(
            &[BaseMessage::user("stream please")],
            &request_for("scripted-model"),
        )
        .await
        .expect("completed stream");
    assert_eq!(turn.message.text(), "streamed");
    assert_eq!(turn.finish_reason, Some(FinishReason::Stop));
    assert_eq!(turn.usage, Usage::default());

    assert_eq!(scripted.call_count(), 1);
    let call = scripted.last_call().expect("recorded call");
    assert_eq!(call.mode, CallMode::Stream);
    assert!(call.request.stream, "`complete` forces streaming");
    assert_eq!(call.request.model_name, "scripted-model");
    assert_eq!(call.last_user_text().as_deref(), Some("stream please"));
}

#[tokio::test]
async fn mixed_script_keeps_order_and_folds_chunk_turns() {
    let model = ScriptedModel::builder()
        .respond_text("a")
        .chunks(text_chunks("b"))
        .respond_text("c")
        .build();
    let request = request_for("scripted-model");

    let first = model.invoke(&[], &request).await.expect("response turn");
    assert_eq!(first.id, "scripted-1");
    assert_eq!(first.message().expect("choice").text(), "a");

    let second = model.invoke(&[], &request).await.expect("chunk turn");
    assert_eq!(second.id, "scripted-2");
    assert_eq!(second.choices.len(), 1, "folded chunks yield one choice");
    assert_eq!(second.message().expect("choice").text(), "b");
    assert_eq!(second.finish_reason(), Some(&FinishReason::Stop));

    let third = model.invoke(&[], &request).await.expect("response turn");
    assert_eq!(third.id, "scripted-3");
    assert_eq!(third.message().expect("choice").text(), "c");
    assert_eq!(model.remaining(), 0);

    // The fold also carries usage and a non-default stop reason.
    let fold_model = ScriptedModel::builder()
        .chunks(
            MockCompletion::text("folded")
                .with_usage(Usage::new(4, 2))
                .with_finish_reason(FinishReason::Length)
                .to_chunks(),
        )
        .build();
    let folded = fold_model
        .invoke(&[], &request)
        .await
        .expect("folded chunk turn");
    assert_eq!(folded.message().expect("choice").text(), "folded");
    assert_eq!(folded.usage, Usage::new(4, 2));
    assert_eq!(folded.finish_reason(), Some(&FinishReason::Length));
}

#[tokio::test]
async fn a_failure_turn_propagates_on_both_entry_points() {
    let failure = ScriptedError::transport("connection reset");
    let model = ScriptedModel::builder()
        .fail(failure.clone())
        .fail(failure.clone())
        .build();

    let invoke_error = model
        .invoke(&[], &request_for("scripted-model"))
        .await
        .expect_err("invoke fails");
    assert_eq!(invoke_error.kind(), ErrorKind::Transport);
    assert!(
        invoke_error.message().contains("connection reset"),
        "{invoke_error}"
    );

    let items =
        collect(model.stream(vec![BaseMessage::user("q")], request_for("scripted-model"))).await;
    assert_eq!(items.len(), 1, "a failed stream yields exactly one item");
    let stream_error = items[0].as_ref().expect_err("the only item is an error");
    assert_eq!(stream_error.kind(), ErrorKind::Transport);
    assert!(stream_error.message().contains("connection reset"));
    assert_eq!(model.remaining(), 0, "each call pops exactly one turn");
}

#[tokio::test]
async fn failure_kinds_survive_the_scripted_error_constructors() {
    let cases = [
        (
            ScriptedError::llm("model rejected the request"),
            ErrorKind::Llm,
        ),
        (ScriptedError::config("missing api key"), ErrorKind::Config),
        (
            ScriptedError::timeout("first token timed out"),
            ErrorKind::Timeout,
        ),
    ];

    for (failure, kind) in cases {
        let model = ScriptedModel::new([
            ScriptedTurn::fail_with(failure.clone()),
            ScriptedTurn::fail_with(failure.clone()),
        ]);

        let invoke_error = model
            .invoke(&[], &request_for("scripted-model"))
            .await
            .expect_err("invoke fails");
        assert_eq!(invoke_error.kind(), kind);
        assert!(invoke_error.message().contains(failure.message.as_str()));

        let items = collect(model.stream(Vec::new(), request_for("scripted-model"))).await;
        assert_eq!(items.len(), 1, "{kind:?}: a failed stream yields one item");
        let stream_error = items[0].as_ref().expect_err("stream item is an error");
        assert_eq!(stream_error.kind(), kind);
        assert!(stream_error.message().contains(failure.message.as_str()));
    }
}

#[tokio::test]
async fn calls_record_mode_messages_and_request_in_order() {
    let model = ScriptedModel::builder()
        .respond_text("a")
        .respond_text("b")
        .stream_text("c")
        .build();
    let invocation = request_for("rec-model").with_temperature(0.25);

    model
        .invoke(
            &[
                BaseMessage::system("sys"),
                BaseMessage::user("first question"),
            ],
            &invocation,
        )
        .await
        .expect("first turn");
    model
        .invoke(&[BaseMessage::user("second question")], &invocation)
        .await
        .expect("second turn");
    let items = collect(model.stream(
        vec![BaseMessage::user("third question")],
        invocation.clone(),
    ))
    .await;
    assert!(
        items.iter().all(Result::is_ok),
        "the stream turn is a plain text projection"
    );

    assert_eq!(model.call_count(), 3);
    assert_eq!(model.remaining(), 0);

    let calls = model.calls();
    assert_eq!(
        calls.iter().map(|call| call.mode).collect::<Vec<_>>(),
        vec![CallMode::Invoke, CallMode::Invoke, CallMode::Stream]
    );
    assert_eq!(calls[0].messages.len(), 2);
    assert_eq!(calls[0].last_user_text().as_deref(), Some("first question"));
    assert_eq!(
        calls[1].last_user_text().as_deref(),
        Some("second question")
    );
    assert_eq!(calls[2].last_user_text().as_deref(), Some("third question"));
    assert!(!calls[2].request.stream, "the request is recorded as given");

    let call = model.last_call().expect("recorded call");
    assert_eq!(call.mode, CallMode::Stream);
    assert_eq!(call.request, invocation);
    assert_eq!(call.request.model_name, "rec-model");
}

#[tokio::test]
async fn a_clone_shares_one_script_and_one_history() {
    let model = ScriptedModel::builder()
        .respond_text("from clone")
        .respond_text("from original")
        .build();
    let clone = model.clone();
    assert_eq!(clone.remaining(), 2);

    let from_clone = clone
        .invoke(&[BaseMessage::user("q1")], &request_for("scripted-model"))
        .await
        .expect("clone turn");
    assert_eq!(from_clone.message().expect("choice").text(), "from clone");
    assert_eq!(model.call_count(), 1, "the original sees the clone's call");
    assert_eq!(model.remaining(), 1, "the queue is shared");

    let from_original = model
        .invoke(&[BaseMessage::user("q2")], &request_for("scripted-model"))
        .await
        .expect("original turn");
    assert_eq!(
        from_original.message().expect("choice").text(),
        "from original"
    );
    assert_eq!(clone.call_count(), 2);
    assert_eq!(
        clone.last_call().expect("recorded call").mode,
        CallMode::Invoke
    );
}

#[tokio::test]
async fn an_exhausted_queue_reuses_the_fallback_without_consuming_it() {
    let model = ScriptedModel::builder().respond_text("only turn").build();
    let request = request_for("scripted-model");

    let first = model.invoke(&[], &request).await.expect("scripted turn");
    assert_eq!(first.message().expect("choice").text(), "only turn");
    assert_eq!(model.remaining(), 0);

    for _ in 0..2 {
        let fallback = model.invoke(&[], &request).await.expect("fallback turn");
        assert_eq!(
            fallback.message().expect("choice").text(),
            "Default mock response"
        );
        assert_eq!(model.remaining(), 0, "the fallback is never consumed");
    }
    assert_eq!(model.call_count(), 3);

    model.set_fallback(ScriptedTurn::text("custom"));
    let custom = model.invoke(&[], &request).await.expect("custom fallback");
    assert_eq!(custom.message().expect("choice").text(), "custom");
    assert_eq!(model.remaining(), 0);
}

#[tokio::test]
async fn the_default_fallback_is_the_python_text() {
    let model = ScriptedModel::default();
    assert_eq!(model.remaining(), 0);
    assert_eq!(model.call_count(), 0);

    let response = model
        .invoke(&[], &request_for("scripted-model"))
        .await
        .expect("fallback turn");
    assert_eq!(
        response.message().expect("choice").text(),
        "Default mock response"
    );
    assert_eq!(response.id, "scripted-1");
    assert_eq!(response.model, "scripted-model");
}

#[tokio::test]
async fn wait_for_calls_wakes_on_a_concurrent_call() {
    let model = ScriptedModel::text("hi");
    let background = model.clone();
    let handle = tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(20)).await;
        background
            .invoke(&[BaseMessage::user("q")], &ModelRequestConfig::default())
            .await
    });

    model.wait_for_calls(1).await.expect("the call is recorded");
    assert_eq!(model.call_count(), 1);
    handle.await.expect("task joins").expect("scripted turn");

    let error = model
        .wait_for_calls_timeout(2, Duration::from_millis(20))
        .await
        .expect_err("the second call never happens");
    assert_eq!(error.kind(), ErrorKind::Timeout);

    model
        .wait_for_calls_timeout(1, Duration::from_millis(20))
        .await
        .expect("the first call is already recorded");
}

#[tokio::test]
async fn builder_overrides_echo_and_stamp_the_model_name() {
    let model = ScriptedModel::builder()
        .provider("fake")
        .model_name("fake-model")
        .respond_text("x")
        .build();
    assert_eq!(model.provider(), "fake");
    assert_eq!(model.model_name(), "fake-model");
    assert_eq!(model.remaining(), 1);

    let response = model
        .invoke(&[BaseMessage::user("q")], &request_for("request-model"))
        .await
        .expect("scripted turn");
    assert_eq!(response.id, "scripted-1");
    assert_eq!(
        response.model, "fake-model",
        "an unset response model adopts the configured model name"
    );
    assert_eq!(response.message().expect("choice").text(), "x");
}

#[tokio::test]
async fn text_is_one_turn_and_push_extends_a_built_model() {
    let model = ScriptedModel::text("hi");
    assert_eq!(model.remaining(), 1);
    let first = model
        .invoke(&[], &request_for("scripted-model"))
        .await
        .expect("one turn");
    assert_eq!(first.message().expect("choice").text(), "hi");
    let second = model
        .invoke(&[], &request_for("scripted-model"))
        .await
        .expect("fallback turn");
    assert_eq!(
        second.message().expect("choice").text(),
        "Default mock response"
    );

    let extended = ScriptedModel::default();
    assert_eq!(extended.remaining(), 0);
    extended.push(ScriptedTurn::text("pushed"));
    extended.push_many([ScriptedTurn::text("many-1"), ScriptedTurn::text("many-2")]);
    assert_eq!(extended.remaining(), 3);

    let mut texts = Vec::new();
    for _ in 0..3 {
        let response = extended
            .invoke(&[], &request_for("scripted-model"))
            .await
            .expect("queued turn");
        texts.push(response.message().expect("choice").text().into_owned());
    }
    assert_eq!(texts, vec!["pushed", "many-1", "many-2"]);
    assert_eq!(extended.remaining(), 0);
}

#[test]
fn into_error_keeps_the_documented_kind() {
    let cases = [
        (ScriptedError::llm("llm failed"), ErrorKind::Llm),
        (ScriptedError::config("bad config"), ErrorKind::Config),
        (
            ScriptedError::validation("bad argument"),
            ErrorKind::Validation,
        ),
        (
            ScriptedError::transport("connection reset"),
            ErrorKind::Transport,
        ),
        (
            ScriptedError::timeout("deadline exceeded"),
            ErrorKind::Timeout,
        ),
    ];
    for (failure, kind) in cases {
        let message = failure.message.clone();
        let error = failure.into_error();
        assert_eq!(error.kind(), kind);
        assert_eq!(error.message().as_ref(), message.as_str(), "{kind:?}");
        assert!(!error.is_cancelled(), "{kind:?}");
    }

    let cancelled = ScriptedError::with_kind(ErrorKind::Cancelled, "ignored payload").into_error();
    assert_eq!(cancelled.kind(), ErrorKind::Cancelled);
    assert_eq!(
        cancelled.message(),
        "cancelled",
        "cancellation has no payload"
    );
    assert!(cancelled.is_cancelled());

    let external = ScriptedError::with_kind(ErrorKind::External, "outer failure").into_error();
    assert_eq!(external.kind(), ErrorKind::External);
    assert_eq!(external.message(), "outer failure");
    assert!(
        std::error::Error::source(&external).is_some(),
        "the wrapped error stays reachable as the source"
    );
}
