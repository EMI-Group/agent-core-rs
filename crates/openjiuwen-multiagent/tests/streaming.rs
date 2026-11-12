#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Integration tests for the team stream driver ([`SessionLease`] and
//! [`team_stream`]) and for the team-session identity rules.
//!
//! Every case drives an in-process "strategy run" that writes frames to its own
//! session, so the suite is offline and deterministic: no model, no network, no
//! timers. It pins the parts of the contract strategies rely on — frame order
//! and wire shape, standalone versus caller-owned teardown, a run error surfaced
//! after the frames, and the `conversation_id` seeding of the session id.
mod common;

use std::sync::Arc;

use futures::FutureExt;
use futures::StreamExt;
use openjiuwen_common::error::{Error, ErrorKind, Result};
use openjiuwen_multiagent::{make_team_session, team_stream, RuntimeConfig, TeamCard, TeamRuntime};
use openjiuwen_session::{
    AgentTeamSession, AgentTeamSessionOptions, Session, StreamEvent, StreamItem,
};
use parking_lot::Mutex;
use serde_json::{json, Value};

/// Builds a runtime whose team id does not collide with another test's.
fn runtime(team_id: &str) -> Arc<TeamRuntime> {
    TeamRuntime::new(RuntimeConfig::new().with_team_id(team_id))
}

/// Collects a team stream into a `Vec` of results.
async fn drain(stream: futures::stream::BoxStream<'static, Result<Value>>) -> Vec<Result<Value>> {
    stream.collect().await
}

#[tokio::test]
async fn frames_are_yielded_in_order() {
    let runtime = runtime("stream-order");
    let card = TeamCard::new("test");

    let stream = team_stream(
        runtime,
        card,
        json!({"conversation_id": "conv-1"}),
        None,
        |session| {
            async move {
                session
                    .write_stream(StreamItem::output("message", None, json!({"text": "one"})))
                    .await?;
                session
                    .write_stream(StreamItem::output_at("message", 1, json!({"text": "two"})))
                    .await?;
                Ok(())
            }
            .boxed()
        },
    );

    let frames = drain(stream).await;
    assert_eq!(frames.len(), 2, "every frame is yielded exactly once");

    let first = frames[0].as_ref().unwrap();
    assert_eq!(first["type"], "message");
    assert_eq!(first["payload"]["text"], "one");
    assert!(
        first.get("index").is_none(),
        "an unindexed frame has no `index` key"
    );

    let second = frames[1].as_ref().unwrap();
    assert_eq!(second["type"], "message");
    assert_eq!(second["payload"]["text"], "two");
    assert_eq!(second["index"], 1);
}

#[tokio::test]
async fn standalone_session_is_torn_down() {
    let runtime = runtime("stream-standalone");
    let card = TeamCard::new("test");
    let captured: Arc<Mutex<Option<Arc<AgentTeamSession>>>> = Arc::new(Mutex::new(None));

    let sink = Arc::clone(&captured);
    let stream = team_stream(
        runtime.clone(),
        card,
        json!({"conversation_id": "conv-standalone"}),
        None,
        move |session| {
            let sink = Arc::clone(&sink);
            async move {
                *sink.lock() = Some(Arc::clone(&session));
                assert_eq!(session.id(), "conv-standalone");
                session
                    .write_stream(StreamItem::output("message", None, json!({"text": "hi"})))
                    .await?;
                Ok(())
            }
            .boxed()
        },
    );

    let frames = drain(stream).await;
    assert_eq!(frames.len(), 1);
    assert!(
        frames[0].is_ok(),
        "the run writes one frame: {:?}",
        frames[0]
    );

    let session = captured.lock().clone().unwrap();
    let session_id = session.id().to_owned();
    assert_eq!(session_id, "conv-standalone");
    assert!(
        runtime.team_session(&session_id).is_none(),
        "an owned session is unbound during teardown"
    );

    let mut follow = session.subscribe();
    assert!(
        matches!(follow.next().await, Some(Ok(StreamEvent::End))),
        "the hub is closed, so a late subscriber sees `End` immediately"
    );
    assert!(
        follow.next().await.is_none(),
        "the stream finishes after `End`"
    );
}

#[tokio::test]
async fn caller_owned_session_stays_bound() {
    let runtime = runtime("stream-caller");
    let card = TeamCard::new("test");
    let caller = Arc::new(AgentTeamSession::with_options(AgentTeamSessionOptions {
        session_id: Some("caller-1".to_owned()),
        ..AgentTeamSessionOptions::default()
    }));
    let captured: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));

    let sink = Arc::clone(&captured);
    let stream = team_stream(
        runtime.clone(),
        card,
        json!({"conversation_id": "ignored-1"}),
        Some(Arc::clone(&caller)),
        move |session| {
            let sink = Arc::clone(&sink);
            async move {
                *sink.lock() = Some(session.id().to_owned());
                session
                    .write_stream(StreamItem::output("message", None, json!({"text": "hi"})))
                    .await?;
                Ok(())
            }
            .boxed()
        },
    );

    let frames = drain(stream).await;
    assert_eq!(frames.len(), 1);
    assert!(
        frames[0].is_ok(),
        "the run writes one frame: {:?}",
        frames[0]
    );

    assert_eq!(
        captured.lock().as_deref(),
        Some("caller-1"),
        "the caller's session is used, not a fresh one"
    );
    assert!(
        runtime.team_session("caller-1").is_none(),
        "a caller-owned session is never bound by the lease"
    );

    let mut follow = caller.subscribe();
    assert!(
        matches!(follow.next().await, Some(Ok(StreamEvent::End))),
        "a caller-owned session is still end-streamed on completion"
    );
}

#[tokio::test]
async fn run_error_is_surfaced_after_frames() {
    let runtime = runtime("stream-error");
    let card = TeamCard::new("test");

    let stream = team_stream(
        runtime,
        card,
        json!({"conversation_id": "conv-error"}),
        None,
        |session| {
            async move {
                session
                    .write_stream(StreamItem::output(
                        "message",
                        None,
                        json!({"text": "before"}),
                    ))
                    .await?;
                Err(Error::validation("boom"))
            }
            .boxed()
        },
    );

    let frames = drain(stream).await;
    assert_eq!(frames.len(), 2, "the frame precedes the error");
    let frame = frames[0].as_ref().unwrap();
    assert_eq!(frame["payload"]["text"], "before");

    let error = frames[1].as_ref().unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert_eq!(error.message(), "boom");
}

#[tokio::test]
async fn conversation_id_seeds_the_session_id() {
    let card = TeamCard::new("test");

    let seeded = make_team_session(&card, &json!({"conversation_id": "abc"}));
    assert_eq!(seeded.id(), "abc");
    assert_eq!(seeded.team_id(), card.id);

    let generated = make_team_session(&card, &json!({}));
    let generated_id = generated.id().to_owned();
    assert_ne!(generated_id, "");
    assert_eq!(
        generated_id.matches('-').count(),
        4,
        "a fallback id is a hyphenated uuid: {generated_id}"
    );
    assert!(uuid::Uuid::parse_str(&generated_id).is_ok());

    let empty = make_team_session(&card, &json!({"conversation_id": ""}));
    assert_ne!(empty.id(), "");
    assert_ne!(empty.id(), "abc");
    assert!(uuid::Uuid::parse_str(empty.id()).is_ok());
}
