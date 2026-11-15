//! Behavior tests for the session stream hub and writer manager.
//!
//! Ports the observable contract of `openjiuwen/core/session/stream/`
//! (v0.1.20): per-subscriber ordering, completion via the end marker,
//! fire-and-forget sends, mode routing, and the `{"type", "index", "payload"}`
//! wire shape. The transport itself is a bounded broadcast hub, so the tests
//! guard every receive with a short timeout instead of sleeping.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::time::Duration;

use futures::stream::BoxStream;
use futures::StreamExt;
use openjiuwen_common::error::{Error, ErrorKind, Result};
use openjiuwen_session::constants::END_FRAME;
use openjiuwen_session::session::Session;
use openjiuwen_session::sessions::WorkflowSession;
use openjiuwen_session::stream::{
    StreamEvent, StreamHub, StreamItem, StreamMode, StreamSendOutcome, StreamWriterManager,
};
use serde_json::{json, Value};

/// Upper bound on how long a deterministic in-memory receive may take.
const RECV_TIMEOUT: Duration = Duration::from_secs(1);

/// Receive one event, failing the test instead of hanging when none arrives.
async fn recv(stream: &mut BoxStream<'static, Result<StreamEvent>>) -> StreamEvent {
    tokio::time::timeout(RECV_TIMEOUT, stream.next())
        .await
        .expect("a stream event arrives before the timeout")
        .expect("the stream is still open")
        .expect("the stream yields Ok events")
}

/// Send five numbered output frames through `hub`.
async fn send_five(hub: &StreamHub) {
    for index in 0..5 {
        hub.send(StreamItem::output_at(
            "llm_output",
            index,
            json!({ "delta": index.to_string() }),
        ))
        .await
        .unwrap();
    }
}

/// Collect the payloads of the next `count` items, asserting they are output frames.
async fn collect_payloads(
    stream: &mut BoxStream<'static, Result<StreamEvent>>,
    count: i64,
) -> Vec<Value> {
    let mut payloads = Vec::new();
    for index in 0..count {
        match recv(stream).await {
            StreamEvent::Item(item) => {
                assert_eq!(item.mode, StreamMode::Output);
                assert_eq!(item.item_type, "llm_output");
                assert_eq!(item.index, Some(index));
                payloads.push(item.payload);
            }
            StreamEvent::End => panic!("end marker arrived before all payloads"),
        }
    }
    payloads
}

#[tokio::test]
async fn outputs_are_delivered_in_order_then_end() {
    let hub = StreamHub::new();
    let mut stream = hub.subscribe();

    send_five(&hub).await;
    hub.close();

    let payloads = collect_payloads(&mut stream, 5).await;
    assert_eq!(
        payloads,
        (0..5)
            .map(|i| json!({ "delta": i.to_string() }))
            .collect::<Vec<_>>()
    );
    assert_eq!(recv(&mut stream).await, StreamEvent::End);
    assert!(
        stream.next().await.is_none(),
        "the stream finishes after the end marker"
    );
}

#[tokio::test]
async fn two_subscribers_observe_the_same_sequence() {
    let hub = StreamHub::new();
    let mut first = hub.subscribe();
    let mut second = hub.subscribe();

    send_five(&hub).await;
    hub.close();

    let first_payloads = collect_payloads(&mut first, 5).await;
    assert_eq!(recv(&mut first).await, StreamEvent::End);
    assert!(first.next().await.is_none());

    let second_payloads = collect_payloads(&mut second, 5).await;
    assert_eq!(recv(&mut second).await, StreamEvent::End);
    assert!(second.next().await.is_none());

    assert_eq!(first_payloads, second_payloads);
}

#[tokio::test]
async fn null_payload_is_a_validation_error() {
    let hub = StreamHub::new();

    let error = hub
        .send(StreamItem::output("llm_output", None, Value::Null))
        .await
        .expect_err("a null payload is rejected");

    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(matches!(error, Error::Validation(_)));
    assert_eq!(error.to_string(), "validation: stream payload is null");
}

#[tokio::test]
async fn late_subscriber_receives_exactly_one_end() {
    let hub = StreamHub::new();
    hub.close();
    assert!(hub.is_closed());

    let mut stream = hub.subscribe();
    assert_eq!(recv(&mut stream).await, StreamEvent::End);
    assert!(
        stream.next().await.is_none(),
        "a late subscriber does not receive a second end marker"
    );
}

#[tokio::test]
async fn disabled_modes_are_reported_and_dropped() {
    let hub = StreamHub::new();
    let manager = StreamWriterManager::with_modes(hub.clone(), vec![StreamMode::Output]);
    let mut stream = manager.subscribe();

    assert_eq!(
        manager
            .write(StreamItem::trace("tracer_agent", json!({ "invoke": 1 })))
            .await
            .expect("a disabled mode is an outcome, not an error"),
        StreamSendOutcome::ModeDisabled
    );
    assert_eq!(
        manager
            .write(StreamItem::output_at(
                "llm_output",
                0,
                json!({ "delta": "hi" }),
            ))
            .await
            .expect("an enabled mode forwards the hub outcome"),
        StreamSendOutcome::Delivered { subscribers: 1 }
    );
    assert_eq!(
        manager
            .write(StreamItem::custom(json!({ "k": "v" })))
            .await
            .expect("a disabled mode is an outcome, not an error"),
        StreamSendOutcome::ModeDisabled
    );
    manager.close();

    match recv(&mut stream).await {
        StreamEvent::Item(item) => {
            assert_eq!(item.mode, StreamMode::Output);
            assert_eq!(item.item_type, "llm_output");
            assert_eq!(item.index, Some(0));
            assert_eq!(item.payload, json!({ "delta": "hi" }));
        }
        StreamEvent::End => panic!("the enabled output frame must be delivered"),
    }
    assert_eq!(recv(&mut stream).await, StreamEvent::End);
    assert!(stream.next().await.is_none());
}

#[tokio::test]
async fn writer_manager_dedupes_modes_and_preserves_order() {
    let hub = StreamHub::new();
    let manager = StreamWriterManager::with_modes(
        hub,
        vec![StreamMode::Trace, StreamMode::Output, StreamMode::Trace],
    );

    assert_eq!(manager.modes(), &[StreamMode::Trace, StreamMode::Output]);
    assert!(manager.enables(StreamMode::Trace));
    assert!(manager.enables(StreamMode::Output));
    assert!(!manager.enables(StreamMode::Custom));

    let all = StreamWriterManager::new(manager.hub().clone());
    assert_eq!(
        all.modes(),
        &[StreamMode::Output, StreamMode::Trace, StreamMode::Custom]
    );
}

#[tokio::test]
async fn wire_shape_omits_mode_and_optional_index() {
    let item = StreamItem::output("llm_output", Some(0), json!({ "delta": "hi" }));
    let wire = item.to_wire();
    assert_eq!(
        wire,
        json!({ "type": "llm_output", "index": 0, "payload": { "delta": "hi" } })
    );
    assert!(
        wire.get("mode").is_none(),
        "the routing mode is not part of the wire payload"
    );

    // The serde form keeps the routing mode but shares the wire field names.
    let serialized = serde_json::to_value(&item).unwrap();
    assert_eq!(serialized["mode"], json!("output"));
    assert_eq!(serialized["type"], json!("llm_output"));
    assert_eq!(serialized["index"], json!(0));
    assert_eq!(serialized["payload"], json!({ "delta": "hi" }));

    let trace = StreamItem::trace("tracer_agent", json!({}));
    let trace_wire = trace.to_wire();
    assert_eq!(trace_wire, json!({ "type": "tracer_agent", "payload": {} }));
    assert!(trace_wire.get("index").is_none());
    let trace_serialized = serde_json::to_value(&trace).unwrap();
    assert!(trace_serialized.get("index").is_none());

    let custom = StreamItem::custom(json!({ "k": "v" }));
    assert_eq!(custom.mode, StreamMode::Custom);
    assert_eq!(custom.item_type, "custom");
    assert_eq!(
        custom.to_wire(),
        json!({ "type": "custom", "payload": { "k": "v" } })
    );
}

#[tokio::test]
async fn end_sentinel_is_recognised() {
    let sentinel = StreamItem::output(END_FRAME, None, json!({}));
    assert!(sentinel.is_end());
    assert!(!StreamItem::output("llm_output", None, json!({})).is_end());

    let retagged = StreamItem::trace("tracer_agent", json!({})).with_mode(StreamMode::Custom);
    assert_eq!(retagged.mode, StreamMode::Custom);
    assert_eq!(retagged.item_type, "tracer_agent");
}

#[tokio::test]
async fn send_reports_no_subscribers_delivered_and_closed() {
    let hub = StreamHub::new();
    assert_eq!(
        hub.capacity(),
        openjiuwen_session::constants::DEFAULT_STREAM_CAPACITY
    );
    assert_eq!(StreamHub::with_capacity(0).capacity(), 1);
    assert_eq!(StreamHub::with_capacity(16).capacity(), 16);

    // An unobserved stream accepts the frame and says so, instead of silently
    // dropping it.
    let outcome = hub
        .send(StreamItem::output(
            "llm_output",
            Some(0),
            json!({ "delta": "x" }),
        ))
        .await
        .expect("an unobserved stream accepts frames");
    assert_eq!(outcome, StreamSendOutcome::NoSubscribers);
    assert!(!outcome.is_delivered());

    // One subscriber.
    let mut first = hub.subscribe();
    assert_eq!(
        hub.send(StreamItem::output(
            "llm_output",
            Some(1),
            json!({ "delta": "y" }),
        ))
        .await
        .expect("a subscribed hub delivers the frame"),
        StreamSendOutcome::Delivered { subscribers: 1 }
    );
    match recv(&mut first).await {
        StreamEvent::Item(item) => assert_eq!(item.index, Some(1)),
        StreamEvent::End => panic!("the delivered frame must arrive before the end"),
    }

    // Two subscribers: both receive the same frame, and the count says so.
    let mut second = hub.subscribe();
    assert_eq!(
        hub.send(StreamItem::output(
            "llm_output",
            Some(2),
            json!({ "delta": "z" }),
        ))
        .await
        .expect("both subscribers are counted"),
        StreamSendOutcome::Delivered { subscribers: 2 }
    );
    for stream in [&mut first, &mut second] {
        match recv(stream).await {
            StreamEvent::Item(item) => assert_eq!(item.index, Some(2)),
            StreamEvent::End => panic!("the delivered frame must arrive before the end"),
        }
    }

    // After close the frame is not published, which is an outcome.
    hub.close();
    assert!(hub.is_closed());
    assert_eq!(
        hub.send(StreamItem::output(
            "llm_output",
            Some(3),
            json!({ "delta": "w" }),
        ))
        .await
        .expect("sending after close is an outcome, not an error"),
        StreamSendOutcome::Closed
    );
    hub.close();
    assert!(hub.is_closed(), "close is idempotent");

    // Late-subscriber semantics are unchanged: exactly one end marker.
    let mut late = hub.subscribe();
    assert_eq!(recv(&mut late).await, StreamEvent::End);
    assert!(
        late.next().await.is_none(),
        "a late subscriber does not receive a second end marker"
    );
}

/// A session's default `StreamTraceSink` advances its position only for frames
/// a subscriber actually received.
#[tokio::test]
async fn stream_trace_sink_advances_its_position_only_on_delivery() {
    let session = WorkflowSession::new("s-trace-delivery");
    let tracer = session.tracer().expect("a workflow session owns a tracer");

    let span = tracer.start_span("unobserved", openjiuwen_session::InvokeType::Llm);
    span.finish(json!({}));
    assert_eq!(
        session.trace().position(),
        None,
        "a frame no subscriber received must not advance the trace position"
    );

    // A subscriber appears: the next span's frame is delivered and counts.
    let _events = session.subscribe();
    let span = tracer.start_span("observed", openjiuwen_session::InvokeType::Llm);
    span.finish(json!({}));
    let position = session
        .trace()
        .position()
        .expect("the delivered frame yields a trace position");
    assert_eq!(
        position.frame_count, 1,
        "only the delivered frame counts, not the dropped one"
    );
    assert_eq!(position.last_invoke_id, Some(span.id()));
}

#[tokio::test]
async fn stream_event_uses_an_event_tag() {
    assert_eq!(
        serde_json::to_value(StreamEvent::End).unwrap(),
        json!({ "event": "end" })
    );

    let event = StreamEvent::Item(StreamItem::output_at(
        "llm_output",
        2,
        json!({ "delta": "z" }),
    ));
    let value = serde_json::to_value(&event).unwrap();
    assert_eq!(value["event"], json!("item"));
    assert_eq!(value["type"], json!("llm_output"));
    assert_eq!(value["index"], json!(2));
    assert_eq!(serde_json::from_value::<StreamEvent>(value).unwrap(), event);

    let end: StreamEvent = serde_json::from_value(json!({ "event": "end" })).unwrap();
    assert_eq!(end, StreamEvent::End);
}

#[tokio::test]
async fn stream_mode_wire_names() {
    assert_eq!(StreamMode::Output.as_str(), "output");
    assert_eq!(StreamMode::Trace.as_str(), "trace");
    assert_eq!(StreamMode::Custom.as_str(), "custom");
    assert_eq!(
        serde_json::to_value([StreamMode::Output, StreamMode::Trace, StreamMode::Custom]).unwrap(),
        json!(["output", "trace", "custom"])
    );
}
