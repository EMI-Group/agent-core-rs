//! End-to-end tests for the shared JSON-RPC client over a real subprocess.
//!
//! Every case spawns a POSIX shell script written by [`support::FakeCli`] that behaves like a
//! JSON-RPC vendor CLI: it records each stdin line in a transcript and answers by matching the
//! request method, echoing back the id it received. The suite therefore covers the true
//! pipe/flush/EOF path of [`JsonRpcClient`] without a network, a vendor SDK or a wall clock.

#![cfg(unix)]
#![allow(clippy::unwrap_used, clippy::expect_used)] // tests assert by panicking

mod support;

use std::sync::Arc;
use std::time::Duration;

use openjiuwen_harness_protocol::{HarnessErrorKind, HarnessResult, JsonValue};
use openjiuwen_harness_providers::vendor_cli::{ChildProcess, ChildProcessConfig, JsonRpcClient};
use serde_json::json;
use support::FakeCli;

/// Bounded await budget for every client interaction.
const BOUND: Duration = Duration::from_secs(10);

/// A JSON-RPC server: it records every request and answers by method, echoing the received id.
///
/// Notifications (lines without an `id`) are only recorded, and `session.login` also pushes one
/// unsolicited notification after its answer.
const SERVER: &str = r#"while IFS= read -r line; do
  printf '%s\n' "$line" >> "$FAKE_CLI_LOG"
  case "$line" in
    *'"id"'*)
      id=${line#*'"id":'}
      id=${id%%,*}
      case "$line" in
        *'"method":"initialize"'*) printf '{"id":%s,"result":{"ok":true}}\n' "$id" ;;
        *'"method":"session.login"'*)
          printf '{"id":%s,"result":"logged in"}\n' "$id"
          printf '%s\n' '{"method":"session.ready","params":{"id":"s-1"}}' ;;
        *'"method":"do.fail"'*) printf '{"id":%s,"error":{"code":-32601,"message":"method not found"}}\n' "$id" ;;
        *) printf '{"id":%s,"result":{"echo":true}}\n' "$id" ;;
      esac
      ;;
  esac
done"#;

/// Spawns the fake CLI and wraps it in a JSON-RPC client labelled `Fake`.
async fn spawn(fake: &FakeCli) -> Arc<JsonRpcClient> {
    let env = fake.env();
    let child = ChildProcess::spawn(ChildProcessConfig {
        program: fake.path(),
        args: &[],
        cwd: None,
        env: &env,
        inherit_env: true,
        label: "Fake",
        max_line_bytes: 0,
    })
    .await
    .expect("the fake CLI must spawn");
    JsonRpcClient::spawn(Box::new(child), "Fake")
}

/// Awaits `future` with the suite's bounded budget.
async fn bounded<T>(
    future: impl std::future::Future<Output = HarnessResult<T>>,
) -> HarnessResult<T> {
    tokio::time::timeout(BOUND, future)
        .await
        .expect("the fake CLI must answer")
}

/// Awaits the next inbound non-response message with the bounded budget.
async fn next_incoming(
    client: &Arc<JsonRpcClient>,
) -> Option<openjiuwen_harness_protocol::JsonObject> {
    tokio::time::timeout(BOUND, client.next_incoming())
        .await
        .expect("the fake CLI must send a message")
}

#[tokio::test]
async fn requests_round_trip_and_the_transcript_records_them() {
    let fake = FakeCli::script(SERVER);
    let client = spawn(&fake).await;

    let initialized = bounded(client.request("initialize", json!({"client": "test"})))
        .await
        .expect("the server answers initialize");
    assert_eq!(initialized, json!({"ok": true}));

    let echoed = bounded(client.request("ping", json!({"value": 1})))
        .await
        .expect("the server answers ping");
    assert_eq!(echoed, json!({"echo": true}));

    let lines = fake.lines();
    assert_eq!(lines.len(), 2, "unexpected transcript: {lines:?}");
    assert!(
        lines[0].contains(r#""method":"initialize""#),
        "{}",
        lines[0]
    );
    assert!(lines[0].contains(r#""id":1"#), "{}", lines[0]);
    assert!(lines[1].contains(r#""method":"ping""#), "{}", lines[1]);
    assert!(lines[1].contains(r#""id":2"#), "{}", lines[1]);
    client.close().await;
}

#[tokio::test]
async fn an_unsolicited_notification_arrives_through_next_incoming() {
    let fake = FakeCli::script(SERVER);
    let client = spawn(&fake).await;

    let reply = bounded(client.request("session.login", json!({})))
        .await
        .expect("the server answers the login");
    assert_eq!(reply, json!("logged in"));

    let notification = next_incoming(&client)
        .await
        .expect("the unsolicited notification");
    assert_eq!(notification.get("method"), Some(&json!("session.ready")));
    assert_eq!(notification.get("params"), Some(&json!({"id": "s-1"})));
    assert_eq!(notification.get("id"), None, "a notification carries no id");
    client.close().await;
}

#[tokio::test]
async fn a_peer_error_is_reported_with_its_code() {
    let fake = FakeCli::script(SERVER);
    let client = spawn(&fake).await;

    let error = bounded(client.request("do.fail", json!({})))
        .await
        .expect_err("the server rejects do.fail");
    assert_eq!(error.kind(), HarnessErrorKind::Protocol);
    assert_eq!(
        error.message(),
        "Fake JSON-RPC error for 'do.fail': method not found (code -32601)"
    );
    assert!(
        !client.is_closed(),
        "an error answer keeps the client usable"
    );
    client.close().await;
}

#[tokio::test]
async fn notifications_are_written_without_an_id() {
    let fake = FakeCli::script(SERVER);
    let client = spawn(&fake).await;

    bounded(client.notify("session.ping", json!({"at": 1})))
        .await
        .expect("the notification is queued");
    // A request flushed after the notification proves the notification reached the child, because
    // the client writes queued messages in order.
    bounded(client.request("ping", json!({})))
        .await
        .expect("the server answers ping");

    let lines = fake.lines();
    assert_eq!(lines.len(), 2, "unexpected transcript: {lines:?}");
    assert!(
        lines[0].contains(r#""method":"session.ping""#),
        "{}",
        lines[0]
    );
    assert!(!lines[0].contains(r#""id""#), "{}", lines[0]);
    client.close().await;
}

#[tokio::test]
async fn end_of_stream_fails_pending_requests_and_ends_the_stream() {
    // Reads one request, records it and exits without answering.
    const EXITS: &str = "IFS= read -r line\nprintf '%s\\n' \"$line\" >> \"$FAKE_CLI_LOG\"\nexit 0";
    let fake = FakeCli::script(EXITS);
    let client = spawn(&fake).await;

    let error = bounded(client.request("ping", json!({})))
        .await
        .expect_err("the CLI exits before answering");
    assert_eq!(error.kind(), HarnessErrorKind::Protocol);
    assert!(
        error
            .message()
            .starts_with("the Fake CLI exited before answering 'ping'"),
        "unexpected message: {}",
        error.message()
    );
    assert!(client.is_closed());
    assert_eq!(fake.lines().len(), 1);

    assert!(next_incoming(&client).await.is_none(), "the stream ended");
    let late = bounded(client.request("again", json!({})))
        .await
        .expect_err("the client is closed");
    assert_eq!(late.message(), "the Fake CLI closed its JSON-RPC stream");
    client.close().await;
}

#[tokio::test]
async fn a_reply_of_unknown_shape_is_skipped_and_the_client_stays_usable() {
    // The first answer is garbage; the second one answers the request.
    const GARBAGE: &str = "IFS= read -r line\nprintf '%s\\n' \"$line\" >> \"$FAKE_CLI_LOG\"\n\
                           printf '%s\\n' 'not json at all'\n\
                           id=${line#*'\"id\":'}\nid=${id%%,*}\n\
                           printf '{\"id\":%s,\"result\":{\"echo\":true}}\\n' \"$id\"\n\
                           while IFS= read -r line; do :; done";
    let fake = FakeCli::script(GARBAGE);
    let client = spawn(&fake).await;

    let echoed: JsonValue = bounded(client.request("ping", json!({})))
        .await
        .expect("the client survives one unreadable line");
    assert_eq!(echoed, json!({"echo": true}));
    assert!(!client.is_closed());
    client.close().await;
}
