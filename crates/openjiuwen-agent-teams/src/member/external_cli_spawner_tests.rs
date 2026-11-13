//! Tests for [`super::external_cli_spawner`].
//!
//! Every test is offline and deterministic: the "vendor CLI" is a POSIX `sh` script from
//! [`FakeChild`](super::test_child::FakeChild) speaking the member protocol (or writing a marker
//! file), every await is bounded, and polling uses short sleeps against a deadline.
//! Direct script execution is avoided in favour of `sh <script>` because tests run in parallel
//! threads and a freshly written script can fail with `ETXTBSY` when a concurrently forking thread
//! inherits its still-open write descriptor.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use openjiuwen_common::error::ErrorKind;
use serde_json::json;
use tempfile::TempDir;

use super::external_cli_spawner::ExternalCliMemberSpawner;
use super::runtime::{ObserverEvent, RecordingObserver};
use super::spawn::{MemberSpawnRequest, MemberSpawner};
use super::spawn_config::{
    ExternalCliRuntimeConfig, MemberSpawnKind, MemberSpawnOptions, StdioCommand,
};
use super::test_child::FakeChild;
use crate::schema::RoundOutcome;

/// Upper bound for any single await in this suite.
const BOUND: Duration = Duration::from_secs(10);

/// Poll interval used by the bounded waits in this suite.
const POLL: Duration = Duration::from_millis(20);

/// Builds a launch configuration that runs `child` through `sh`.
fn external_config(child: &FakeChild) -> ExternalCliRuntimeConfig {
    ExternalCliRuntimeConfig {
        command: child.sh_command(),
        ..ExternalCliRuntimeConfig::default()
    }
}

/// Builds a spawner that launches `child`.
fn spawner_for(child: &FakeChild) -> ExternalCliMemberSpawner {
    ExternalCliMemberSpawner::new(external_config(child))
}

/// Builds a request for `member_name` in team `alpha`.
fn request(member: &str, session: &str) -> MemberSpawnRequest {
    MemberSpawnRequest::new("alpha", member, session, "/tmp/ws")
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
        tokio::time::sleep(POLL).await;
    }
}

/// Polls `path` until it holds a non-empty string, then returns it.
async fn read_marker(path: &Path) -> String {
    let deadline = tokio::time::Instant::now() + BOUND;
    loop {
        if let Ok(text) = std::fs::read_to_string(path) {
            if !text.is_empty() {
                return text;
            }
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "marker {} was not written within the bound",
            path.display()
        );
        tokio::time::sleep(POLL).await;
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
        tokio::time::sleep(POLL).await;
    }
}

/// Reports whether this platform exposes a readable `/proc` process table.
fn process_table_is_readable() -> bool {
    Path::new("/proc/self").exists()
}

/// Reports whether `pid` still names a live process (Linux `/proc` only).
fn process_is_alive(pid: u32) -> bool {
    Path::new(&format!("/proc/{pid}")).exists()
}

/// Writes `body` as a child script that records a marker and then idles until killed.
fn marker_child(marker: &Path, body: &str) -> FakeChild {
    FakeChild::with_body(&format!(
        "{body} > '{marker}'\nexec sleep 600",
        marker = marker.display()
    ))
}

// 1 ---------------------------------------------------------------------------------------------

#[test]
fn config_round_trips_the_diagnostic_labels() {
    let config = ExternalCliRuntimeConfig {
        command: StdioCommand::new("claude").with_arg("--print"),
        cli_agent: Some("claude".to_owned()),
        provider: Some("anthropic".to_owned()),
        model: Some("claude-sonnet-4".to_owned()),
    };
    let spawner = ExternalCliMemberSpawner::new(config.clone());

    assert_eq!(spawner.config(), &config);
    assert_eq!(spawner.config().cli_agent.as_deref(), Some("claude"));
    assert_eq!(spawner.config().provider.as_deref(), Some("anthropic"));
    assert_eq!(spawner.config().model.as_deref(), Some("claude-sonnet-4"));
    assert_eq!(spawner.live_members(), Vec::<String>::new());
    assert!(!spawner.is_running("dev-1"));
    assert_eq!(spawner.child_pid("dev-1"), None);
}

// 2 ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn spawn_and_start_a_protocol_speaking_child() {
    let child = FakeChild::serve();
    let spawner = spawner_for(&child);

    let runtime = tokio::time::timeout(BOUND, spawner.spawn(request("dev-1", "sess-1")))
        .await
        .expect("spawn did not return")
        .expect("spawn failed");

    assert_eq!(runtime.session_id(), "sess-1");
    assert_eq!(spawner.live_members(), vec!["dev-1".to_owned()]);
    assert!(spawner.is_running("dev-1"));
    assert!(spawner.child_pid("dev-1").is_some());

    tokio::time::timeout(BOUND, runtime.start(None))
        .await
        .expect("start did not return")
        .expect("start failed");
    assert!(spawner.is_running("dev-1"));

    tokio::time::timeout(BOUND, spawner.despawn("dev-1"))
        .await
        .expect("despawn did not return")
        .expect("despawn failed");
    assert!(!spawner.is_running("dev-1"));
    assert_eq!(spawner.live_members(), Vec::<String>::new());
}

// 3 ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn a_send_delivers_a_chunk_to_the_observer() {
    let child = FakeChild::serve();
    let spawner = spawner_for(&child);
    let runtime = tokio::time::timeout(BOUND, spawner.spawn(request("dev-1", "sess-1")))
        .await
        .expect("spawn did not return")
        .expect("spawn failed");
    let observer = Arc::new(RecordingObserver::new());
    runtime.subscribe(observer.clone());

    tokio::time::timeout(BOUND, runtime.start(None))
        .await
        .expect("start did not return")
        .expect("start failed");
    tokio::time::timeout(BOUND, runtime.send("hello".to_owned(), false))
        .await
        .expect("send did not return")
        .expect("send failed");

    let events = collect_until(&observer, |events| {
        events.contains(&ObserverEvent::Chunk(json!({"delta": "hello"})))
    })
    .await;
    assert!(events.contains(&ObserverEvent::Round(RoundOutcome::Started)));
    assert!(events.contains(&ObserverEvent::Round(RoundOutcome::Finished)));

    tokio::time::timeout(BOUND, spawner.despawn("dev-1"))
        .await
        .expect("despawn did not return")
        .expect("despawn failed");
}

// 4 ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn a_missing_program_is_a_config_error() {
    let spawner = ExternalCliMemberSpawner::new(ExternalCliRuntimeConfig {
        command: StdioCommand::new("openjiuwen-missing-external-cli"),
        ..ExternalCliRuntimeConfig::default()
    });

    let Err(error) = tokio::time::timeout(BOUND, spawner.spawn(request("dev-1", "sess-1")))
        .await
        .expect("spawn did not return")
    else {
        panic!("a missing program must not spawn a child");
    };

    assert_eq!(error.kind(), ErrorKind::Config);
    assert!(
        error
            .to_string()
            .contains("openjiuwen-missing-external-cli"),
        "the error must name the missing program, got: {error}"
    );
    assert_eq!(spawner.live_members(), Vec::<String>::new());
}

// 5 ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn a_per_request_config_overrides_the_spawner_config() {
    let child = FakeChild::serve();
    let spawner = ExternalCliMemberSpawner::new(ExternalCliRuntimeConfig {
        command: StdioCommand::new("openjiuwen-missing-external-cli"),
        ..ExternalCliRuntimeConfig::default()
    });
    let mut request = request("dev-1", "sess-1");
    request.options = MemberSpawnOptions::from_kind(MemberSpawnKind::ExternalCli)
        .with_external_cli(external_config(&child));

    let runtime = tokio::time::timeout(BOUND, spawner.spawn(request))
        .await
        .expect("spawn did not return")
        .expect("the per-request external-CLI config must win");
    assert_eq!(runtime.session_id(), "sess-1");

    tokio::time::timeout(BOUND, spawner.despawn("dev-1"))
        .await
        .expect("despawn did not return")
        .expect("despawn failed");
}

// 6 ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn a_request_of_another_kind_ignores_the_external_cli_options() {
    // Documented rule: the per-request external-CLI config applies only when the request names
    // `MemberSpawnKind::ExternalCli`; a `Process` request falls back to the spawner's own config.
    let child = FakeChild::serve();
    let spawner = ExternalCliMemberSpawner::new(ExternalCliRuntimeConfig {
        command: StdioCommand::new("openjiuwen-missing-external-cli"),
        ..ExternalCliRuntimeConfig::default()
    });
    let mut request = request("dev-1", "sess-1");
    request.options = MemberSpawnOptions::from_kind(MemberSpawnKind::Process)
        .with_external_cli(external_config(&child));

    let Err(error) = tokio::time::timeout(BOUND, spawner.spawn(request))
        .await
        .expect("spawn did not return")
    else {
        panic!("a non-external-CLI request must not use the external-CLI options");
    };

    assert_eq!(error.kind(), ErrorKind::Config);
    assert!(error
        .to_string()
        .contains("openjiuwen-missing-external-cli"));
    assert_eq!(spawner.live_members(), Vec::<String>::new());
}

// 7 ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn identity_env_reaches_the_child() {
    let dir = TempDir::new().expect("temp dir for the marker");
    let marker = dir.path().join("identity.txt");
    let child = marker_child(
        &marker,
        r#"printf '%s|%s' "$OPENJIUWEN_MEMBER_NAME" "$OPENJIUWEN_SESSION_ID""#,
    );
    let spawner = spawner_for(&child);

    let runtime = tokio::time::timeout(BOUND, spawner.spawn(request("dev-1", "sess-1")))
        .await
        .expect("spawn did not return")
        .expect("spawn failed");

    assert_eq!(read_marker(&marker).await, "dev-1|sess-1");

    tokio::time::timeout(BOUND, spawner.despawn("dev-1"))
        .await
        .expect("despawn did not return")
        .expect("despawn failed");
    drop(runtime);
}

// 8 ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn extra_config_env_reaches_the_child() {
    let dir = TempDir::new().expect("temp dir for the marker");
    let marker = dir.path().join("token.txt");
    let child = marker_child(&marker, r#"printf '%s' "$TEAM_TEST_TOKEN""#);
    let spawner = ExternalCliMemberSpawner::new(ExternalCliRuntimeConfig {
        command: StdioCommand::new("sh")
            .with_arg(child.program())
            .with_env("TEAM_TEST_TOKEN", "s3cr3t"),
        ..ExternalCliRuntimeConfig::default()
    });

    let runtime = tokio::time::timeout(BOUND, spawner.spawn(request("dev-1", "sess-1")))
        .await
        .expect("spawn did not return")
        .expect("spawn failed");

    assert_eq!(read_marker(&marker).await, "s3cr3t");

    tokio::time::timeout(BOUND, spawner.despawn("dev-1"))
        .await
        .expect("despawn did not return")
        .expect("despawn failed");
    drop(runtime);
}

// 9 ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn diagnostic_labels_add_no_argv_element_and_no_environment_variable() {
    let dir = TempDir::new().expect("temp dir for the marker");
    let marker = dir.path().join("argv.txt");
    let child = marker_child(
        &marker,
        concat!(
            r#"printf '%s|%s|%s|%s' "$#""#,
            r#" "${OPENJIUWEN_CLI_AGENT:-unset}" "${OPENJIUWEN_CLI_PROVIDER:-unset}""#,
            r#" "${OPENJIUWEN_CLI_MODEL:-unset}""#,
        ),
    );
    let spawner = ExternalCliMemberSpawner::new(ExternalCliRuntimeConfig {
        command: StdioCommand::new("sh")
            .with_arg(child.program())
            .with_arg("--protocol"),
        cli_agent: Some("claude".to_owned()),
        provider: Some("anthropic".to_owned()),
        model: Some("claude-sonnet-4".to_owned()),
    });

    let runtime = tokio::time::timeout(BOUND, spawner.spawn(request("dev-1", "sess-1")))
        .await
        .expect("spawn did not return")
        .expect("spawn failed");

    // Exactly the one configured argument reaches the child, and none of the labels is turned into
    // an environment variable.
    assert_eq!(read_marker(&marker).await, "1|unset|unset|unset");

    tokio::time::timeout(BOUND, spawner.despawn("dev-1"))
        .await
        .expect("despawn did not return")
        .expect("despawn failed");
    drop(runtime);
}

// 10 --------------------------------------------------------------------------------------------

#[tokio::test]
async fn despawn_stops_the_child_and_is_idempotent() {
    let child = FakeChild::serve();
    let spawner = spawner_for(&child);
    let runtime = tokio::time::timeout(BOUND, spawner.spawn(request("dev-1", "sess-1")))
        .await
        .expect("spawn did not return")
        .expect("spawn failed");
    let pid = spawner
        .child_pid("dev-1")
        .expect("a spawned child has a pid");
    assert!(spawner.is_running("dev-1"));

    tokio::time::timeout(BOUND, spawner.despawn("dev-1"))
        .await
        .expect("despawn did not return")
        .expect("despawn failed");
    assert!(!spawner.is_running("dev-1"));
    assert_eq!(spawner.child_pid("dev-1"), None);
    assert_eq!(spawner.live_members(), Vec::<String>::new());

    // A second despawn for the same (now unknown) member is a no-op, not an error.
    tokio::time::timeout(BOUND, spawner.despawn("dev-1"))
        .await
        .expect("despawn did not return")
        .expect("despawn of an unknown member must not fail");
    tokio::time::timeout(BOUND, spawner.despawn("never-spawned"))
        .await
        .expect("despawn did not return")
        .expect("despawn of a never-spawned member must not fail");

    if process_table_is_readable() {
        assert!(
            wait_until(|| !process_is_alive(pid)).await,
            "child {pid} is still alive after despawn"
        );
    }
    drop(runtime);
}

// 11 --------------------------------------------------------------------------------------------

#[tokio::test]
async fn respawn_replaces_the_live_runtime() {
    let child = FakeChild::serve();
    let spawner = spawner_for(&child);
    let first = tokio::time::timeout(BOUND, spawner.spawn(request("dev-1", "sess-1")))
        .await
        .expect("spawn did not return")
        .expect("spawn failed");
    let first_pid = spawner.child_pid("dev-1").expect("first pid");
    assert!(spawner.is_running("dev-1"));
    // Drop the test's handle so the spawner's registry entry is the last owner: replacing it then
    // drops the old runtime and kills its child.
    drop(first);

    let second = tokio::time::timeout(BOUND, spawner.spawn(request("dev-1", "sess-1")))
        .await
        .expect("re-spawn did not return")
        .expect("re-spawn failed");
    let second_pid = spawner.child_pid("dev-1").expect("second pid");

    assert_ne!(first_pid, second_pid);
    assert_eq!(spawner.live_members(), vec!["dev-1".to_owned()]);
    assert!(spawner.is_running("dev-1"));
    if process_table_is_readable() {
        assert!(
            wait_until(|| !process_is_alive(first_pid)).await,
            "the replaced child {first_pid} is still alive"
        );
    }

    tokio::time::timeout(BOUND, spawner.despawn("dev-1"))
        .await
        .expect("despawn did not return")
        .expect("despawn failed");
    drop(second);
}

// 12 --------------------------------------------------------------------------------------------

#[tokio::test]
async fn live_members_are_listed_sorted() {
    let child = FakeChild::serve();
    let spawner = spawner_for(&child);
    let zeta = tokio::time::timeout(BOUND, spawner.spawn(request("zeta", "sess-z")))
        .await
        .expect("spawn did not return")
        .expect("spawn failed");
    let alpha = tokio::time::timeout(BOUND, spawner.spawn(request("alpha", "sess-a")))
        .await
        .expect("spawn did not return")
        .expect("spawn failed");

    assert_eq!(
        spawner.live_members(),
        vec!["alpha".to_owned(), "zeta".to_owned()]
    );

    for member in ["zeta", "alpha"] {
        tokio::time::timeout(BOUND, spawner.despawn(member))
            .await
            .expect("despawn did not return")
            .expect("despawn failed");
    }
    assert_eq!(spawner.live_members(), Vec::<String>::new());
    drop((zeta, alpha));
}

// 13 --------------------------------------------------------------------------------------------

#[tokio::test]
async fn the_spawner_is_object_safe_behind_arc() {
    let child = FakeChild::serve();
    let concrete = Arc::new(spawner_for(&child));
    let spawner: Arc<dyn MemberSpawner> = Arc::clone(&concrete) as Arc<dyn MemberSpawner>;

    let runtime = tokio::time::timeout(BOUND, spawner.spawn(request("dev-1", "sess-1")))
        .await
        .expect("spawn did not return")
        .expect("spawn failed");
    assert!(concrete.is_running("dev-1"));
    tokio::time::timeout(BOUND, runtime.start(None))
        .await
        .expect("start did not return")
        .expect("start failed");

    tokio::time::timeout(BOUND, spawner.despawn("dev-1"))
        .await
        .expect("despawn did not return")
        .expect("despawn failed");
    assert!(!concrete.is_running("dev-1"));
}

// 14 --------------------------------------------------------------------------------------------

#[tokio::test]
async fn a_diagnostic_label_does_not_change_the_session_id() {
    // The session id comes from the request only; labels never substitute for it.
    let child = FakeChild::serve();
    let spawner = ExternalCliMemberSpawner::new(ExternalCliRuntimeConfig {
        command: StdioCommand::new("sh").with_arg(child.program()),
        cli_agent: Some("codex".to_owned()),
        provider: Some("openai".to_owned()),
        model: Some("gpt-5".to_owned()),
    });

    let runtime = tokio::time::timeout(
        BOUND,
        spawner.spawn(request("dev-1", "session-from-request")),
    )
    .await
    .expect("spawn did not return")
    .expect("spawn failed");
    assert_eq!(runtime.session_id(), "session-from-request");

    tokio::time::timeout(BOUND, spawner.despawn("dev-1"))
        .await
        .expect("despawn did not return")
        .expect("despawn failed");
    drop(runtime);
}
