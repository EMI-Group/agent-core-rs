//! Tests for the child-process member runtime in [`super::stdio`].
//!
//! Every test is offline and deterministic: the child is a POSIX `sh` script from
//! [`FakeChild`](super::test_child::FakeChild), every wait is bounded, and no test depends on
//! wall-clock ordering.
//! Polling loops use short sleeps against a deadline instead of fixed long sleeps.

use std::sync::Arc;
use std::time::Duration;

use openjiuwen_common::error::ErrorKind;
use openjiuwen_harness_protocol::HarnessState;
use serde_json::json;
use tokio_util::sync::CancellationToken;

use super::runtime::{MemberRuntime, ObserverEvent, RecordingObserver};
use super::spawn_config::StdioCommand;
use super::stdio::{StdioMemberRuntime, StdioRuntimeOptions, DEFAULT_IO_TIMEOUT};
use super::test_child::FakeChild;
use crate::schema::RoundOutcome;

/// Upper bound for any single await in this suite.
const BOUND: Duration = Duration::from_secs(10);

/// Builds the spawn command that runs `child`.
///
/// The script is passed to `sh` instead of being executed directly: tests run in parallel threads
/// in one process, and a freshly written script executed directly can fail with `ETXTBSY` when a
/// concurrently forking thread inherits its still-open write descriptor.
fn command(child: &FakeChild) -> StdioCommand {
    child.sh_command()
}

/// Polls `condition` until it holds or the bound expires.
async fn wait_until(mut condition: impl FnMut() -> bool) -> bool {
    let deadline = tokio::time::Instant::now() + BOUND;
    loop {
        if condition() {
            return true;
        }
        if tokio::time::Instant::now() >= deadline {
            return false;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

/// Drains `observer` until `done` holds for the events seen so far.
async fn collect_until(
    observer: &RecordingObserver,
    mut done: impl FnMut(&[ObserverEvent]) -> bool,
) -> Vec<ObserverEvent> {
    let deadline = tokio::time::Instant::now() + BOUND;
    let mut events = Vec::new();
    loop {
        events.extend(observer.take());
        if done(&events) {
            return events;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "timed out waiting for member events; saw {events:?}"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

/// Starts a member against `child` and stops it again, so tests clean up promptly.
async fn stop(runtime: &StdioMemberRuntime) {
    tokio::time::timeout(BOUND, runtime.stop())
        .await
        .expect("stop did not return within the bound")
        .expect("stop failed");
}

// 1 ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn spawn_and_start_handshake() {
    let child = FakeChild::serve();
    let runtime = StdioMemberRuntime::spawn(&command(&child), "team-session")
        .await
        .expect("spawn");

    assert_eq!(runtime.session_id(), "team-session");
    assert_eq!(runtime.session_id_value(), "team-session");
    assert!(runtime.is_running());
    assert!(runtime.child_pid().is_some());

    tokio::time::timeout(BOUND, runtime.start(None))
        .await
        .expect("start handshake did not return")
        .expect("start failed");

    assert!(runtime.is_running());
    stop(&runtime).await;
    assert!(!runtime.is_running());
}

// 2 ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn send_round_and_chunk_frames_reach_the_observer_in_order() {
    let child = FakeChild::serve();
    let runtime = StdioMemberRuntime::spawn(&command(&child), "s-2")
        .await
        .expect("spawn");
    let observer = Arc::new(RecordingObserver::new());
    runtime.subscribe(observer.clone());
    tokio::time::timeout(BOUND, runtime.start(None))
        .await
        .expect("start did not return")
        .expect("start failed");

    tokio::time::timeout(BOUND, runtime.send("hi".to_owned(), false))
        .await
        .expect("send did not return")
        .expect("send failed");

    let events = collect_until(&observer, |events| {
        events.contains(&ObserverEvent::Round(RoundOutcome::Finished))
    })
    .await;
    let position = |wanted: &ObserverEvent| events.iter().position(|event| event == wanted);
    let started = position(&ObserverEvent::Round(RoundOutcome::Started)).expect("started frame");
    let chunk = position(&ObserverEvent::Chunk(json!({"delta": "hi"}))).expect("chunk frame");
    let finished = position(&ObserverEvent::Round(RoundOutcome::Finished)).expect("finished frame");
    assert!(started < chunk, "chunk arrived before the round started");
    assert!(chunk < finished, "chunk arrived after the round finished");
    stop(&runtime).await;
}

// 3 ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn pause_and_resume_report_states() {
    let child = FakeChild::serve();
    let runtime = StdioMemberRuntime::spawn(&command(&child), "s-3")
        .await
        .expect("spawn");
    let observer = Arc::new(RecordingObserver::new());
    runtime.subscribe(observer.clone());
    tokio::time::timeout(BOUND, runtime.start(None))
        .await
        .expect("start did not return")
        .expect("start failed");

    tokio::time::timeout(BOUND, runtime.pause())
        .await
        .expect("pause did not return")
        .expect("pause failed");
    collect_until(&observer, |events| {
        events.contains(&ObserverEvent::State(HarnessState::Paused))
    })
    .await;

    tokio::time::timeout(BOUND, runtime.resume(Some("q".to_owned())))
        .await
        .expect("resume did not return")
        .expect("resume failed");
    collect_until(&observer, |events| {
        events.contains(&ObserverEvent::State(HarnessState::Running))
    })
    .await;
    stop(&runtime).await;
}

// 4 ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn abort_reports_an_aborted_round() {
    let child = FakeChild::serve();
    let runtime = StdioMemberRuntime::spawn(&command(&child), "s-4")
        .await
        .expect("spawn");
    let observer = Arc::new(RecordingObserver::new());
    runtime.subscribe(observer.clone());
    tokio::time::timeout(BOUND, runtime.start(None))
        .await
        .expect("start did not return")
        .expect("start failed");

    tokio::time::timeout(BOUND, runtime.abort(false))
        .await
        .expect("abort did not return")
        .expect("abort failed");
    collect_until(&observer, |events| {
        events.contains(&ObserverEvent::Round(RoundOutcome::Aborted))
    })
    .await;
    stop(&runtime).await;
}

// 5 ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn every_observer_receives_the_fan_out_including_a_late_one() {
    let child = FakeChild::serve();
    let runtime = StdioMemberRuntime::spawn(&command(&child), "s-5")
        .await
        .expect("spawn");
    let first = Arc::new(RecordingObserver::new());
    runtime.subscribe(first.clone());
    tokio::time::timeout(BOUND, runtime.start(None))
        .await
        .expect("start did not return")
        .expect("start failed");
    let second = Arc::new(RecordingObserver::new());
    runtime.subscribe(second.clone());

    tokio::time::timeout(BOUND, runtime.send("x".to_owned(), false))
        .await
        .expect("send did not return")
        .expect("send failed");

    for observer in [&first, &second] {
        let events = collect_until(observer, |events| {
            events.contains(&ObserverEvent::Round(RoundOutcome::Finished))
        })
        .await;
        assert!(events.contains(&ObserverEvent::Round(RoundOutcome::Started)));
        assert!(events.contains(&ObserverEvent::Chunk(json!({"delta": "x"}))));
    }
    stop(&runtime).await;
}

// 6 ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn a_refused_start_surfaces_the_child_message() {
    let child = FakeChild::failing_start();
    let runtime = StdioMemberRuntime::spawn(&command(&child), "s-6")
        .await
        .expect("spawn");
    let error = tokio::time::timeout(BOUND, runtime.start(None))
        .await
        .expect("start did not return")
        .expect_err("a refused start must fail");
    assert_eq!(error.kind(), ErrorKind::External);
    assert!(
        error.message().contains("start refused"),
        "message: {}",
        error.message()
    );
}

// 7 ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn a_child_that_exits_immediately_is_terminated() {
    let child = FakeChild::with_body("exit 0");
    let runtime = StdioMemberRuntime::spawn(&command(&child), "s-7")
        .await
        .expect("spawn");
    let error = tokio::time::timeout(BOUND, runtime.start(None))
        .await
        .expect("start did not return")
        .expect_err("start cannot succeed against an exited child");
    assert!(
        matches!(error.kind(), ErrorKind::External | ErrorKind::Transport),
        "kind: {:?}",
        error.kind()
    );
    assert!(
        wait_until(|| runtime.state() == HarnessState::Terminated).await,
        "state: {:?}",
        runtime.state()
    );
    assert!(!runtime.is_running());
    assert_eq!(runtime.exit_code(), Some(0));
}

// 8 ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn garbage_stdout_is_skipped_and_never_kills_the_member() {
    let child = FakeChild::with_body(
        r#"printf 'not json\n'
printf '\n'
printf '{"type":"state","state":"running"}\n'
exec sleep 600"#,
    );
    let runtime = StdioMemberRuntime::spawn(&command(&child), "s-8")
        .await
        .expect("spawn");
    let observer = Arc::new(RecordingObserver::new());
    runtime.subscribe(observer.clone());
    collect_until(&observer, |events| {
        events.contains(&ObserverEvent::State(HarnessState::Running))
    })
    .await;
    assert!(
        runtime.is_running(),
        "garbage must not terminate the member"
    );
}

// 9 ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn stderr_is_captured_into_the_bounded_tail() {
    let child = FakeChild::with_body(
        r#"printf 'boom\n' >&2
while IFS= read -r line; do
  case "$line" in
    *'"type":"start"'*) printf '{"type":"ready","session_id":"child-session"}\n' ;;
  esac
done"#,
    );
    let runtime = StdioMemberRuntime::spawn(&command(&child), "s-9")
        .await
        .expect("spawn");
    tokio::time::timeout(BOUND, runtime.start(None))
        .await
        .expect("start did not return")
        .expect("start failed");
    assert!(
        wait_until(|| runtime.stderr_tail().contains("boom")).await,
        "stderr tail: {:?}",
        runtime.stderr_tail()
    );
    stop(&runtime).await;
}

// 10 --------------------------------------------------------------------------------------------

#[tokio::test]
async fn stop_kills_a_silent_child_and_is_idempotent() {
    let child = FakeChild::silent();
    let runtime = StdioMemberRuntime::spawn(&command(&child), "s-10")
        .await
        .expect("spawn");
    stop(&runtime).await;
    assert!(!runtime.is_running());
    assert_eq!(runtime.state(), HarnessState::Terminated);
    stop(&runtime).await;
}

// 11 --------------------------------------------------------------------------------------------

#[tokio::test]
async fn stop_notifies_observers_of_termination_exactly_once() {
    let child = FakeChild::silent();
    let runtime = StdioMemberRuntime::spawn(&command(&child), "s-11")
        .await
        .expect("spawn");
    let observer = Arc::new(RecordingObserver::new());
    runtime.subscribe(observer.clone());
    stop(&runtime).await;

    let events = observer.take();
    let terminated = events
        .iter()
        .filter(|event| **event == ObserverEvent::State(HarnessState::Terminated))
        .count();
    assert_eq!(terminated, 1, "events: {events:?}");

    stop(&runtime).await;
    assert!(
        observer.take().is_empty(),
        "a second stop must not notify again"
    );
}

// 12 --------------------------------------------------------------------------------------------

#[cfg(unix)]
#[tokio::test]
async fn drop_kills_the_child() {
    let child = FakeChild::silent();
    let runtime = StdioMemberRuntime::spawn(&command(&child), "s-12")
        .await
        .expect("spawn");
    let pid = runtime.child_pid().expect("child pid");
    drop(runtime);

    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    let mut gone = false;
    while tokio::time::Instant::now() < deadline {
        tokio::time::sleep(Duration::from_millis(50)).await;
        let alive = std::process::Command::new("kill")
            .args(["-0", &pid.to_string()])
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .is_ok_and(|status| status.success());
        if !alive {
            gone = true;
            break;
        }
    }
    assert!(gone, "child process {pid} survived the runtime drop");
}

// 13 --------------------------------------------------------------------------------------------

#[tokio::test]
async fn a_cancellation_token_kills_the_child() {
    let child = FakeChild::silent();
    let token = CancellationToken::new();
    let runtime = StdioMemberRuntime::spawn_with(
        &command(&child),
        "s-13",
        StdioRuntimeOptions::new().with_cancellation(token.clone()),
    )
    .await
    .expect("spawn");
    assert!(!runtime.cancellation_token().is_cancelled());

    token.cancel();

    assert!(
        wait_until(|| runtime.state() == HarnessState::Terminated).await,
        "state: {:?}",
        runtime.state()
    );
    assert!(!runtime.is_running());
}

// 14 --------------------------------------------------------------------------------------------

#[tokio::test]
async fn a_missing_program_is_a_config_error() {
    let command = StdioCommand::new("openjiuwen-no-such-member-binary-xyz");
    let error = StdioMemberRuntime::spawn(&command, "s-14")
        .await
        .expect_err("a missing program must fail");
    assert_eq!(error.kind(), ErrorKind::Config);
}

// 15 --------------------------------------------------------------------------------------------

#[tokio::test]
async fn a_silent_child_times_out_the_start_handshake() {
    let child = FakeChild::silent();
    let runtime = StdioMemberRuntime::spawn_with(
        &command(&child),
        "s-15",
        StdioRuntimeOptions::new().with_io_timeout(Duration::from_millis(100)),
    )
    .await
    .expect("spawn");
    let error = tokio::time::timeout(BOUND, runtime.start(None))
        .await
        .expect("start did not return")
        .expect_err("a silent child cannot answer start");
    assert_eq!(error.kind(), ErrorKind::Timeout);
}

// 16 --------------------------------------------------------------------------------------------

#[tokio::test]
async fn session_id_and_exit_code_before_start() {
    let child = FakeChild::serve();
    let runtime = StdioMemberRuntime::spawn(&command(&child), "configured")
        .await
        .expect("spawn");
    assert_eq!(runtime.session_id(), "configured");
    assert_eq!(runtime.exit_code(), None);
    stop(&runtime).await;
}

// 17 --------------------------------------------------------------------------------------------

#[tokio::test]
async fn every_call_fails_fast_after_stop() {
    let child = FakeChild::silent();
    let runtime = StdioMemberRuntime::spawn(&command(&child), "s-17")
        .await
        .expect("spawn");
    stop(&runtime).await;

    let kinds = tokio::time::timeout(BOUND, async {
        let results = [
            runtime.send("x".to_owned(), false).await,
            runtime.pause().await,
            runtime.resume(None).await,
            runtime.abort(true).await,
            runtime.start(None).await,
        ];
        results
            .iter()
            .map(|result| {
                result
                    .as_ref()
                    .expect_err("every call must fail after stop")
                    .kind()
            })
            .collect::<Vec<_>>()
    })
    .await
    .expect("a call hung after stop");

    for kind in kinds {
        assert!(
            matches!(kind, ErrorKind::External | ErrorKind::Transport),
            "kind: {kind:?}"
        );
    }
}

// 18 --------------------------------------------------------------------------------------------

#[test]
fn options_defaults_and_builders() {
    let default = StdioRuntimeOptions::default();
    assert_eq!(default.io_timeout, DEFAULT_IO_TIMEOUT);
    assert!(default.cancellation.is_none());
    assert_eq!(DEFAULT_IO_TIMEOUT, Duration::from_secs(10));

    let options = StdioRuntimeOptions::new()
        .with_io_timeout(Duration::from_millis(250))
        .with_cancellation(CancellationToken::new());
    assert_eq!(options.io_timeout, Duration::from_millis(250));
    assert!(options.cancellation.is_some());
}

#[tokio::test]
async fn the_exposed_cancellation_token_is_usable() {
    let child = FakeChild::serve();
    let runtime = StdioMemberRuntime::spawn(&command(&child), "s-18")
        .await
        .expect("spawn");
    let token = runtime.cancellation_token();
    assert!(!token.is_cancelled());
    assert!(runtime.options().cancellation.is_none());
    token.cancel();
    assert!(token.is_cancelled());
    assert!(wait_until(|| runtime.state() == HarnessState::Terminated).await);
}

// 19 --------------------------------------------------------------------------------------------

#[tokio::test]
async fn a_configurable_io_bound_drives_a_full_round_trip() {
    let child = FakeChild::serve();
    let runtime = StdioMemberRuntime::spawn_with(
        &command(&child),
        "s-19",
        StdioRuntimeOptions::new().with_io_timeout(Duration::from_secs(2)),
    )
    .await
    .expect("spawn");
    assert_eq!(runtime.options().io_timeout, Duration::from_secs(2));
    let observer = Arc::new(RecordingObserver::new());
    runtime.subscribe(observer.clone());

    tokio::time::timeout(BOUND, runtime.start(None))
        .await
        .expect("start did not return")
        .expect("start failed");
    tokio::time::timeout(BOUND, runtime.send("ping".to_owned(), true))
        .await
        .expect("send did not return")
        .expect("send failed");
    collect_until(&observer, |events| {
        events.contains(&ObserverEvent::Round(RoundOutcome::Finished))
    })
    .await;
    stop(&runtime).await;
}

// 20 --------------------------------------------------------------------------------------------

#[tokio::test]
async fn concurrent_sends_are_serialized_without_corrupting_frames() {
    let child = FakeChild::serve();
    let runtime = StdioMemberRuntime::spawn(&command(&child), "s-20")
        .await
        .expect("spawn");
    let observer = Arc::new(RecordingObserver::new());
    runtime.subscribe(observer.clone());
    tokio::time::timeout(BOUND, runtime.start(None))
        .await
        .expect("start did not return")
        .expect("start failed");

    let (first, second) = tokio::join!(
        runtime.send("a".to_owned(), false),
        runtime.send("b".to_owned(), false)
    );
    tokio::time::timeout(BOUND, async {
        first.expect("first send failed");
        second.expect("second send failed");
    })
    .await
    .expect("concurrent sends did not complete");

    let events = collect_until(&observer, |events| {
        let chunks = events
            .iter()
            .filter(|event| matches!(event, ObserverEvent::Chunk(_)))
            .count();
        chunks >= 2
    })
    .await;
    let deltas: Vec<_> = events
        .iter()
        .filter_map(|event| match event {
            ObserverEvent::Chunk(chunk) => chunk.get("delta").cloned(),
            _ => None,
        })
        .collect();
    assert!(deltas.contains(&json!("a")), "deltas: {deltas:?}");
    assert!(deltas.contains(&json!("b")), "deltas: {deltas:?}");
    stop(&runtime).await;
}
