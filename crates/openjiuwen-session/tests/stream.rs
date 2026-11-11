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
use openjiuwen_session::stream::{
    StreamEvent, StreamHub, StreamItem, StreamMode, StreamWriterManager,
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
async fn disabled_modes_are_dropped() {
    let hub = StreamHub::new();
    let manager = StreamWriterManager::with_modes(hub.clone(), vec![StreamMode::Output]);
    let mut stream = manager.subscribe();

    manager
        .write(StreamItem::trace("tracer_agent", json!({ "invoke": 1 })))
        .await
        .unwrap();
    manager
        .write(StreamItem::output_at(
            "llm_output",
            0,
            json!({ "delta": "hi" }),
        ))
        .await
        .unwrap();
    manager
        .write(StreamItem::custom(json!({ "k": "v" })))
        .await
        .unwrap();
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
async fn send_without_subscribers_and_after_close_is_ok() {
    let hub = StreamHub::new();
    assert_eq!(
        hub.capacity(),
        openjiuwen_session::constants::DEFAULT_STREAM_CAPACITY
    );
    assert_eq!(StreamHub::with_capacity(0).capacity(), 1);
    assert_eq!(StreamHub::with_capacity(16).capacity(), 16);

    hub.send(StreamItem::output(
        "llm_output",
        Some(0),
        json!({ "delta": "x" }),
    ))
    .await
    .expect("an unobserved stream accepts frames");

    hub.close();
    assert!(hub.is_closed());
    hub.send(StreamItem::output(
        "llm_output",
        Some(1),
        json!({ "delta": "y" }),
    ))
    .await
    .expect("sending after close is a no-op");
    hub.close();
    assert!(hub.is_closed(), "close is idempotent");
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
