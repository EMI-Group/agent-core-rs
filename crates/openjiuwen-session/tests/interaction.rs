//! Behavior tests for human-in-the-loop interactions.
//!
//! Ports the observable contract of `openjiuwen/core/session/interaction/`
//! (v0.1.20) plus the Rust-only timeout and cancellation additions: the
//! `__interaction__` frame shape (`InteractionOutput { id, value }`), seeded
//! resume answers that do not re-emit a frame, broker resolution by id, the
//! approval vocabulary on the wire, and the failure modes (`NotFound`,
//! `Timeout`, `Cancelled`).
//!
//! Every wait is bounded by a timeout guard, so a regression fails the test
//! instead of hanging it.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;
use std::time::Duration;

use futures::stream::BoxStream;
use futures::StreamExt;
use openjiuwen_common::error::{Error, ErrorKind, Result};
use openjiuwen_session::interaction::{
    InteractionBroker, InteractionDecision, InteractionKind, InteractionOutput, InteractionRequest,
    InteractionResponse, InteractiveInput, INTERACTION_FRAME_TYPE, INTERACTIVE_INPUT_KEY,
};
use openjiuwen_session::stream::{
    StreamEvent, StreamHub, StreamItem, StreamMode, StreamWriterManager,
};
use serde_json::{json, Value};
use tokio_util::sync::CancellationToken;

/// Upper bound on how long a deterministic in-memory wait may take.
const RECV_TIMEOUT: Duration = Duration::from_secs(1);

/// Build a broker plus a writer over a fresh hub.
fn broker_and_writer() -> (InteractionBroker, Arc<StreamWriterManager>) {
    let hub = StreamHub::new();
    (InteractionBroker::new(), StreamWriterManager::new(hub))
}

/// Receive one event, failing the test instead of hanging when none arrives.
async fn recv(stream: &mut BoxStream<'static, Result<StreamEvent>>) -> StreamEvent {
    tokio::time::timeout(RECV_TIMEOUT, stream.next())
        .await
        .expect("a stream event arrives before the timeout")
        .expect("the stream is still open")
        .expect("the stream yields Ok events")
}

/// Poll the broker until `request_id` is registered as pending.
async fn wait_pending(broker: &InteractionBroker, request_id: &str) {
    for _ in 0..1_000 {
        if broker.pending().iter().any(|id| id == request_id) {
            return;
        }
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    panic!("interaction `{request_id}` never became pending");
}

/// Spawn a request and wait until the broker has registered it.
fn spawn_request(
    broker: &InteractionBroker,
    writer: &Arc<StreamWriterManager>,
    request: InteractionRequest,
) -> (String, tokio::task::JoinHandle<Result<InteractionResponse>>) {
    let request_id = request.id.clone();
    let broker = broker.clone();
    let writer = Arc::clone(writer);
    let handle = tokio::spawn(async move { broker.request(request, &writer).await });
    (request_id, handle)
}

/// Join a spawned request, failing the test when it hangs.
async fn join_request(
    handle: tokio::task::JoinHandle<Result<InteractionResponse>>,
) -> Result<InteractionResponse> {
    tokio::time::timeout(RECV_TIMEOUT, handle)
        .await
        .expect("the request resolves before the timeout")
        .expect("the request task does not panic")
}

#[tokio::test]
async fn seeded_answer_returns_without_emitting_a_frame() {
    let broker = InteractionBroker::new();
    let hub = StreamHub::new();
    let writer = StreamWriterManager::new(Arc::clone(&hub));
    let mut frames = hub.subscribe();

    let request = InteractionRequest::ask_user(json!({"prompt": "which branch?"}));
    let request_id = request.id.clone();
    broker.seed_interrupt(request_id.clone(), json!({"branch": "main"}));

    let response = broker.request(request, &writer).await.unwrap();
    assert_eq!(response.request_id, request_id);
    assert_eq!(response.decision, InteractionDecision::Answer);
    assert_eq!(response.value, json!({"branch": "main"}));
    assert!(!response.is_approved());

    // Python's resume path reads the seeded input instead of emitting the
    // interaction frame, so nothing may reach the stream.
    let next = tokio::time::timeout(Duration::from_millis(50), frames.next()).await;
    assert!(
        next.is_err(),
        "a seeded answer must not publish an interaction frame"
    );
    assert_eq!(broker.pending(), Vec::<String>::new());
}

#[tokio::test]
async fn seed_queues_user_inputs_and_interrupt_answers_fifo() {
    let (broker, writer) = broker_and_writer();

    let mut input = InteractiveInput::new();
    input.update("node_a", json!("first"));
    input.answer("intr_a", json!("seeded-1"));
    broker.seed(input);

    let mut request = InteractionRequest::ask_user(json!({"prompt": "node?"}));
    request.id = "node_a".to_owned();
    let first = broker.request(request.clone(), &writer).await.unwrap();
    assert_eq!(first.value, json!("first"));

    request.id = "intr_a".to_owned();
    let second = broker.request(request.clone(), &writer).await.unwrap();
    assert_eq!(second.value, json!("seeded-1"));

    // One interrupt id may hold several queued answers; they are handed out in
    // FIFO order, one per request.
    broker.seed_interrupt("intr_a", json!("seeded-2"));
    let third = broker.request(request.clone(), &writer).await.unwrap();
    assert_eq!(third.value, json!("seeded-2"));

    // The queue is drained: a fourth request has to wait for a real answer.
    let mut fourth = InteractionRequest::ask_user(json!({"prompt": "again?"}))
        .with_timeout(Duration::from_millis(10));
    fourth.id = "intr_a".to_owned();
    let error = broker.request(fourth, &writer).await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Timeout);
}

#[tokio::test]
async fn resolve_delivers_approve() {
    let (broker, writer) = broker_and_writer();
    let request = InteractionRequest::ask_user(json!({"prompt": "run the tool?"}));
    let (request_id, handle) = spawn_request(&broker, &writer, request);

    wait_pending(&broker, &request_id).await;
    broker
        .resolve(InteractionResponse::approve(request_id.clone()))
        .unwrap();

    let response = join_request(handle).await.unwrap();
    assert_eq!(response.request_id, request_id);
    assert_eq!(response.decision, InteractionDecision::Approve);
    assert_eq!(response.value, Value::Null);
    assert_eq!(response.reason, None);
    assert!(response.is_approved());
    assert_eq!(broker.pending(), Vec::<String>::new());
}

#[tokio::test]
async fn resolve_delivers_approve_always_reject_and_answer() {
    let (broker, writer) = broker_and_writer();

    let (request_id, handle) = spawn_request(
        &broker,
        &writer,
        InteractionRequest::confirm(json!("proceed?")),
    );
    wait_pending(&broker, &request_id).await;
    broker
        .resolve(InteractionResponse::approve_always(request_id.clone()))
        .unwrap();
    let response = join_request(handle).await.unwrap();
    assert_eq!(response.decision, InteractionDecision::ApproveAlways);
    assert!(response.is_approved());

    let (request_id, handle) = spawn_request(
        &broker,
        &writer,
        InteractionRequest::confirm(json!("proceed?")),
    );
    wait_pending(&broker, &request_id).await;
    broker
        .resolve(InteractionResponse::reject(
            request_id.clone(),
            Some("not allowed".to_owned()),
        ))
        .unwrap();
    let response = join_request(handle).await.unwrap();
    assert_eq!(response.decision, InteractionDecision::Reject);
    assert_eq!(response.reason.as_deref(), Some("not allowed"));
    assert!(!response.is_approved());

    let (request_id, handle) = spawn_request(
        &broker,
        &writer,
        InteractionRequest::ask_user(json!("name?")),
    );
    wait_pending(&broker, &request_id).await;
    broker
        .resolve(InteractionResponse::answer(
            request_id.clone(),
            json!({"name": "ada"}),
        ))
        .unwrap();
    let response = join_request(handle).await.unwrap();
    assert_eq!(response.decision, InteractionDecision::Answer);
    assert_eq!(response.value, json!({"name": "ada"}));
    assert!(!response.is_approved());
}

#[tokio::test]
async fn resolve_unknown_id_is_not_found() {
    let broker = InteractionBroker::new();
    let error = broker
        .resolve(InteractionResponse::approve("intr_missing"))
        .unwrap_err();
    assert!(matches!(error, Error::NotFound(_)));
    assert_eq!(error.kind(), ErrorKind::NotFound);
    assert!(error.to_string().contains("interaction `intr_missing`"));
}

#[tokio::test]
async fn timeout_expires_and_clears_pending() {
    let (broker, writer) = broker_and_writer();
    let request =
        InteractionRequest::ask_user(json!("anyone?")).with_timeout(Duration::from_millis(10));
    let request_id = request.id.clone();

    let error = tokio::time::timeout(RECV_TIMEOUT, broker.request(request, &writer))
        .await
        .expect("the request future finishes")
        .unwrap_err();
    assert!(matches!(error, Error::Timeout(_)));
    assert_eq!(error.kind(), ErrorKind::Timeout);
    assert!(error.to_string().contains(&request_id));
    assert_eq!(
        broker.pending(),
        Vec::<String>::new(),
        "a timed-out request leaves no pending entry"
    );
}

#[tokio::test]
async fn broker_default_timeout_applies_when_request_has_none() {
    let (broker, writer) = broker_and_writer();
    broker.set_default_timeout(Some(Duration::from_millis(10)));
    assert_eq!(broker.default_timeout(), Some(Duration::from_millis(10)));

    let request = InteractionRequest::ask_user(json!("anyone?"));
    let error = tokio::time::timeout(RECV_TIMEOUT, broker.request(request, &writer))
        .await
        .expect("the request future finishes")
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Timeout);

    broker.set_default_timeout(None);
    assert_eq!(broker.default_timeout(), None);
}

#[tokio::test]
async fn cancellation_token_cancels_the_wait() {
    let (broker, writer) = broker_and_writer();
    let token = CancellationToken::new();
    let request = InteractionRequest::ask_user(json!("anyone?")).with_cancellation(token.clone());
    let (request_id, handle) = spawn_request(&broker, &writer, request);

    wait_pending(&broker, &request_id).await;
    token.cancel();

    let error = join_request(handle).await.unwrap_err();
    assert!(matches!(error, Error::Cancelled));
    assert_eq!(error.kind(), ErrorKind::Cancelled);
    assert_eq!(broker.pending(), Vec::<String>::new());
}

#[tokio::test]
async fn cancel_all_answers_every_waiter_with_cancel() {
    let (broker, writer) = broker_and_writer();
    let (first_id, first) =
        spawn_request(&broker, &writer, InteractionRequest::confirm(json!("one?")));
    wait_pending(&broker, &first_id).await;
    let (second_id, second) =
        spawn_request(&broker, &writer, InteractionRequest::confirm(json!("two?")));
    wait_pending(&broker, &second_id).await;
    let pending = broker.pending();
    assert_eq!(pending.len(), 2);
    assert!(pending.contains(&first_id));
    assert!(pending.contains(&second_id));

    broker.cancel_all("run aborted");

    for (handle, id) in [(first, first_id), (second, second_id)] {
        let response = join_request(handle).await.unwrap();
        assert_eq!(response.request_id, id);
        assert_eq!(response.decision, InteractionDecision::Cancel);
        assert_eq!(response.reason.as_deref(), Some("run aborted"));
        assert!(!response.is_approved());
    }
    assert_eq!(broker.pending(), Vec::<String>::new());
}

#[tokio::test]
async fn request_emits_the_interaction_frame() {
    let broker = InteractionBroker::new();
    let hub = StreamHub::new();
    let writer = StreamWriterManager::new(Arc::clone(&hub));
    let mut frames = hub.subscribe();

    let request = InteractionRequest::permission("tool.bash", json!({"command": "ls -la"}));
    let (request_id, handle) = spawn_request(&broker, &writer, request);
    wait_pending(&broker, &request_id).await;
    broker
        .resolve(InteractionResponse::approve(request_id.clone()))
        .unwrap();
    join_request(handle).await.unwrap();

    match recv(&mut frames).await {
        StreamEvent::Item(item) => {
            assert_eq!(item.mode, StreamMode::Output);
            assert_eq!(item.item_type, INTERACTION_FRAME_TYPE);
            assert_eq!(item.index, Some(0));
            assert_eq!(
                item.payload,
                json!({"id": "tool.bash", "value": {"command": "ls -la"}})
            );
            // Python's OutputSchema has no mode on the wire.
            let wire = item.to_wire();
            assert_eq!(wire["type"], json!(INTERACTION_FRAME_TYPE));
            assert_eq!(wire["index"], json!(0));
            assert_eq!(
                wire["payload"],
                json!({"id": "tool.bash", "value": {"command": "ls -la"}})
            );
            assert!(wire.get("mode").is_none());
            assert_eq!(wire.as_object().unwrap().len(), 3);
        }
        StreamEvent::End => panic!("the end marker arrived before the frame"),
    }
}

#[tokio::test]
async fn interaction_frames_are_indexed_by_a_broker_wide_counter() {
    let (broker, writer) = broker_and_writer();
    let hub = Arc::clone(writer.hub());
    let mut frames = hub.subscribe();

    for expected in 0..2 {
        let request = InteractionRequest::ask_user(json!({"n": expected}));
        let (request_id, handle) = spawn_request(&broker, &writer, request);
        wait_pending(&broker, &request_id).await;
        broker
            .resolve(InteractionResponse::approve(request_id.clone()))
            .unwrap();
        join_request(handle).await.unwrap();

        match recv(&mut frames).await {
            StreamEvent::Item(item) => assert_eq!(item.index, Some(expected)),
            StreamEvent::End => panic!("the end marker arrived before the frame"),
        }
    }
}

#[test]
fn to_stream_item_falls_back_to_the_request_id() {
    let request = InteractionRequest::ask_user(json!({"prompt": "hi"}));
    let frame = request.to_stream_item(7);
    assert_eq!(
        frame,
        StreamItem::output(
            INTERACTION_FRAME_TYPE,
            Some(7),
            json!({"id": request.id.clone(), "value": {"prompt": "hi"}}),
        )
    );

    let output = InteractionOutput {
        id: request.id.clone(),
        value: json!({"prompt": "hi"}),
    };
    assert_eq!(
        serde_json::to_value(&output).unwrap(),
        json!({"id": request.id, "value": {"prompt": "hi"}})
    );
}

#[test]
fn decisions_use_pythons_approval_vocabulary() {
    assert_eq!(
        serde_json::to_value(InteractionDecision::Approve).unwrap(),
        json!("allow")
    );
    assert_eq!(
        serde_json::to_value(InteractionDecision::ApproveAlways).unwrap(),
        json!("allow_always")
    );
    assert_eq!(
        serde_json::to_value(InteractionDecision::Reject).unwrap(),
        json!("deny")
    );
    assert_eq!(
        serde_json::to_value(InteractionDecision::Cancel).unwrap(),
        json!("cancel")
    );
    assert_eq!(
        serde_json::to_value(InteractionDecision::Answer).unwrap(),
        json!("answer")
    );

    let decoded: InteractionDecision = serde_json::from_value(json!("allow_always")).unwrap();
    assert_eq!(decoded, InteractionDecision::ApproveAlways);
    assert_eq!(InteractionDecision::ApproveAlways.as_str(), "allow_always");
    assert_eq!(InteractionDecision::default(), InteractionDecision::Cancel);
}

#[test]
fn interactive_input_omits_empty_interrupt_answers() {
    let empty = InteractiveInput::new();
    assert!(empty.is_empty());
    let value = serde_json::to_value(&empty).unwrap();
    assert!(value.get("interrupt_answers").is_none());
    assert_eq!(value["user_inputs"], json!({}));
    assert_eq!(value["raw_inputs"], Value::Null);

    let mut filled = InteractiveInput::new();
    filled.update("node_a", json!("answer"));
    filled.answer("intr_a", json!({"ok": true}));
    assert!(!filled.is_empty());
    let value = serde_json::to_value(&filled).unwrap();
    assert_eq!(value["user_inputs"], json!({"node_a": "answer"}));
    assert_eq!(value["interrupt_answers"], json!({"intr_a": {"ok": true}}));

    // Unknown fields are tolerated, empty ids are ignored.
    let decoded: InteractiveInput =
        serde_json::from_str(r#"{"user_inputs": {"n": 1}, "extra": true}"#).unwrap();
    assert_eq!(decoded.user_inputs.get("n"), Some(&json!(1)));
    assert_eq!(
        INTERACTIVE_INPUT_KEY, "__interactive_input__",
        "the state key keeps Python's constant"
    );

    let mut ignored = InteractiveInput::new();
    ignored.update("", json!("dropped"));
    ignored.answer("", json!("dropped"));
    assert!(ignored.is_empty());
}

#[test]
fn request_ids_and_response_helpers() {
    let request = InteractionRequest::new(InteractionKind::Permission, json!({"q": 1}));
    assert!(request.id.starts_with("intr_"));
    assert_eq!(request.kind, InteractionKind::Permission);
    assert_eq!(request.component_id, None);
    assert_eq!(request.timeout_ms, None);
    assert_eq!(request.cancellation, None);
    assert_eq!(request.options, Vec::new());
    assert!(request.metadata.is_empty());

    let permission = InteractionRequest::permission("tool.write", json!({"q": 2}))
        .with_timeout(Duration::from_millis(250))
        .with_metadata("source", json!("test"));
    assert_eq!(permission.kind, InteractionKind::Permission);
    assert_eq!(permission.component_id.as_deref(), Some("tool.write"));
    assert_eq!(permission.timeout_ms, Some(250));
    assert_eq!(permission.metadata.get("source"), Some(&json!("test")));
    assert!(permission.id.starts_with("intr_"), "the minted id is kept");

    let confirm = InteractionRequest::confirm(json!({"q": 3}));
    assert_eq!(confirm.kind, InteractionKind::Confirm);
    let ask = InteractionRequest::ask_user(json!({"q": 4}));
    assert_eq!(ask.kind, InteractionKind::AskUser);
    assert_eq!(InteractionKind::default(), InteractionKind::AskUser);

    // Sub-millisecond durations saturate to zero rather than panicking.
    let tiny = InteractionRequest::ask_user(Value::Null).with_timeout(Duration::from_nanos(1));
    assert_eq!(tiny.timeout_ms, Some(0));

    assert!(InteractionResponse::approve("intr_x").is_approved());
    assert!(InteractionResponse::approve_always("intr_x").is_approved());
    assert!(!InteractionResponse::reject("intr_x", None).is_approved());
    assert!(!InteractionResponse::answer("intr_x", json!(1)).is_approved());
    assert!(!InteractionResponse::cancel("intr_x").is_approved());
    assert_eq!(
        InteractionResponse::default().decision,
        InteractionDecision::Cancel
    );
}

#[test]
fn request_round_trips_through_json() {
    let request = InteractionRequest::permission("tool.bash", json!({"command": "ls"}))
        .with_timeout(Duration::from_millis(500))
        .with_metadata("run", json!("r-1"));
    let encoded = serde_json::to_value(&request).unwrap();
    assert_eq!(encoded["kind"], json!("permission"));
    assert_eq!(encoded["component_id"], json!("tool.bash"));
    assert_eq!(encoded["timeout_ms"], json!(500));
    assert!(
        encoded.get("cancellation").is_none(),
        "runtime state stays out of the wire"
    );

    let decoded: InteractionRequest = serde_json::from_value(encoded).unwrap();
    assert_eq!(decoded, request);
    assert_eq!(decoded.cancellation, None);

    // Unknown fields are tolerated.
    let decoded: InteractionRequest =
        serde_json::from_str(r#"{"id": "intr_1", "kind": "ask_user", "future": 1}"#).unwrap();
    assert_eq!(decoded.kind, InteractionKind::AskUser);
    assert_eq!(decoded.question, Value::Null);
}
