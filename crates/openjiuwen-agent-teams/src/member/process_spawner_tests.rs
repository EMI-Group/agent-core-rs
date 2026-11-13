//! Tests for the OS-process member spawner in [`super::process_spawner`].
//!
//! Every test is offline and deterministic: the child is a POSIX `sh` script from
//! [`FakeChild`](super::test_child::FakeChild), every await is bounded, and every poll uses a short
//! sleep against a deadline instead of a fixed wait.
//! The script is run through `sh <path>` rather than executed directly, because a freshly written
//! script executed directly can fail with `ETXTBSY` while another test thread forks.
//! Spawns go through
//! [`spawn_with_retry`](super::test_child::spawn_with_retry) as a belt-and-braces measure, and use
//! the more forgiving [`SPAWN_BOUND`]; the retry is test-side only.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use openjiuwen_common::error::ErrorKind;
use tempfile::TempDir;

use super::process_spawner::ProcessMemberSpawner;
use super::runtime::MemberRuntime;
use super::spawn::{MemberSpawnRequest, MemberSpawner, MEMBER_NAME_ENV};
use super::spawn_config::{MemberSpawnKind, MemberSpawnOptions, ProcessMemberConfig, StdioCommand};
use super::test_child::{spawn_with_retry, FakeChild};

/// Upper bound for any single protocol await in this suite.
const BOUND: Duration = Duration::from_secs(10);

/// Upper bound for one spawn attempt.
///
/// Generous on purpose: a launch can be delayed while the other test threads fork and exec at the
/// same time. It is still bounded, and only a failing run pays it.
const SPAWN_BOUND: Duration = Duration::from_secs(30);

/// The program name no test host has installed, used for the configuration-error cases.
const MISSING_PROGRAM: &str = "openjiuwen-no-such-team-member-xyz";

/// The command that runs `child` through `sh`.
fn sh_command(child: &FakeChild) -> StdioCommand {
    child.sh_command()
}

/// A launch configuration running `child`.
fn process_config(child: &FakeChild) -> ProcessMemberConfig {
    ProcessMemberConfig {
        command: sh_command(child),
    }
}

/// A spawner whose default launch command runs `child`.
fn spawner_for(child: &FakeChild) -> ProcessMemberSpawner {
    ProcessMemberSpawner::new(process_config(child))
}

/// A request that selects the OS-process mode with no per-request configuration.
fn process_request(member: &str) -> MemberSpawnRequest {
    MemberSpawnRequest::new("alpha", member, "sess-1", "/tmp/ws")
        .with_options(MemberSpawnOptions::from_kind(MemberSpawnKind::Process))
}

/// Spawns `request`, bounded.
async fn spawn_ok(
    spawner: &ProcessMemberSpawner,
    request: MemberSpawnRequest,
) -> Arc<dyn MemberRuntime> {
    spawn_with_retry(|| async {
        tokio::time::timeout(SPAWN_BOUND, spawner.spawn(request.clone()))
            .await
            .expect("spawn did not return within the bound")
    })
    .await
    .expect("spawn failed")
}

/// Starts `runtime`, bounded.
async fn start_ok(runtime: &Arc<dyn MemberRuntime>) {
    tokio::time::timeout(BOUND, runtime.start(None))
        .await
        .expect("start did not return within the bound")
        .expect("start failed");
}

/// Despawns `member`, bounded.
async fn despawn_ok(spawner: &ProcessMemberSpawner, member: &str) {
    tokio::time::timeout(BOUND, spawner.despawn(member))
        .await
        .expect("despawn did not return within the bound")
        .expect("despawn failed");
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

/// Reads a file as trimmed text, treating a missing or half-written file as empty.
fn read_trimmed(path: &Path) -> String {
    std::fs::read_to_string(path)
        .unwrap_or_default()
        .trim()
        .to_string()
}

/// Waits until the operating system no longer lists `pid`.
#[cfg(target_os = "linux")]
async fn wait_process_gone(pid: u32) -> bool {
    wait_until(|| !Path::new(&format!("/proc/{pid}")).exists()).await
}

// 1 ---------------------------------------------------------------------------------------------
#[test]
fn default_config_is_team_member_serve() {
    let spawner = ProcessMemberSpawner::default();
    assert_eq!(
        spawner.config().command.to_argv(),
        vec!["team-member".to_string(), "serve".to_string()]
    );
    assert_eq!(spawner.live_members(), Vec::<String>::new());
}

// 2 ---------------------------------------------------------------------------------------------
#[test]
fn config_returns_the_configured_command() {
    let child = FakeChild::serve();
    let command = sh_command(&child).with_env("EXTRA_KNOB", "on");
    let spawner = ProcessMemberSpawner::new(ProcessMemberConfig {
        command: command.clone(),
    });
    assert_eq!(spawner.config().command, command);
    assert_eq!(
        spawner.config().command.env().get("EXTRA_KNOB"),
        Some(&"on".to_string())
    );
}

// 3 ---------------------------------------------------------------------------------------------
#[tokio::test]
async fn spawn_registers_a_live_child_and_starts_it() {
    let child = FakeChild::serve();
    let spawner = spawner_for(&child);

    let runtime = spawn_ok(&spawner, process_request("dev-1")).await;

    assert_eq!(runtime.session_id(), "sess-1");
    assert!(spawner.is_running("dev-1"));
    assert_eq!(spawner.live_members(), vec!["dev-1".to_string()]);
    let pid = spawner
        .child_pid("dev-1")
        .expect("the child must report a pid");
    assert!(pid > 0);

    start_ok(&runtime).await;
    assert_eq!(spawner.child_pid("dev-1"), Some(pid));

    despawn_ok(&spawner, "dev-1").await;
}

// 4 ---------------------------------------------------------------------------------------------
#[tokio::test]
async fn despawn_stops_the_child_and_empties_the_registry() {
    let child = FakeChild::serve();
    let spawner = spawner_for(&child);
    let runtime = spawn_ok(&spawner, process_request("dev-1")).await;
    start_ok(&runtime).await;
    let pid = spawner.child_pid("dev-1").expect("pid");

    despawn_ok(&spawner, "dev-1").await;

    assert!(!spawner.is_running("dev-1"));
    assert_eq!(spawner.live_members(), Vec::<String>::new());
    assert!(spawner.child_pid("dev-1").is_none());
    #[cfg(target_os = "linux")]
    assert!(
        wait_process_gone(pid).await,
        "the child process {pid} must be reaped after despawn"
    );
}

// 5 ---------------------------------------------------------------------------------------------
#[tokio::test]
async fn despawn_of_an_unknown_member_is_ok() {
    let child = FakeChild::serve();
    let spawner = spawner_for(&child);
    despawn_ok(&spawner, "ghost").await;
    assert_eq!(spawner.live_members(), Vec::<String>::new());
}

// 6 ---------------------------------------------------------------------------------------------
#[tokio::test]
async fn despawn_after_the_child_exited_is_ok() {
    let child = FakeChild::with_body("exit 0");
    let spawner = spawner_for(&child);
    let _runtime = spawn_ok(&spawner, process_request("dev-1")).await;

    despawn_ok(&spawner, "dev-1").await;

    assert!(!spawner.is_running("dev-1"));
    assert_eq!(spawner.live_members(), Vec::<String>::new());
}

// 7 ---------------------------------------------------------------------------------------------
#[tokio::test]
async fn a_missing_program_is_a_config_error() {
    let spawner = ProcessMemberSpawner::new(ProcessMemberConfig {
        command: StdioCommand::new(MISSING_PROGRAM),
    });

    let error = tokio::time::timeout(BOUND, spawner.spawn(process_request("dev-1")))
        .await
        .expect("spawn did not return within the bound");
    let Err(error) = error else {
        panic!("a missing program must fail the spawn");
    };

    assert_eq!(error.kind(), ErrorKind::Config);
    assert!(
        error.message().contains(MISSING_PROGRAM),
        "the configuration error must name the program, got: {error}"
    );
    assert_eq!(spawner.live_members(), Vec::<String>::new());
}

// 8 ---------------------------------------------------------------------------------------------
#[tokio::test]
async fn a_process_request_overrides_the_spawner_config() {
    let child = FakeChild::serve();
    let spawner = ProcessMemberSpawner::new(ProcessMemberConfig {
        command: StdioCommand::new(MISSING_PROGRAM),
    });
    let request = process_request("dev-1").with_options(
        MemberSpawnOptions::from_kind(MemberSpawnKind::Process).with_process(ProcessMemberConfig {
            command: sh_command(&child),
        }),
    );

    let runtime = spawn_ok(&spawner, request).await;

    assert!(spawner.is_running("dev-1"));
    start_ok(&runtime).await;
    despawn_ok(&spawner, "dev-1").await;
}

// 9 ---------------------------------------------------------------------------------------------
#[tokio::test]
async fn a_process_request_without_a_config_uses_the_spawner_config() {
    let child = FakeChild::serve();
    let spawner = spawner_for(&child);
    let request = process_request("dev-1");
    assert!(request.options.process.is_none());

    let runtime = spawn_ok(&spawner, request).await;
    start_ok(&runtime).await;
    despawn_ok(&spawner, "dev-1").await;
}

// 10 --------------------------------------------------------------------------------------------
#[tokio::test]
async fn an_in_process_request_still_uses_the_spawner_config() {
    let child = FakeChild::serve();
    let spawner = spawner_for(&child);
    // Dispatch is the caller's job: a spawner handed a request spawns a process whatever mode the
    // request names, and the in-process default falls through to the spawner's own command.
    let request = MemberSpawnRequest::new("alpha", "dev-1", "sess-1", "/tmp/ws");
    assert!(request.options.is_in_process());

    let runtime = spawn_ok(&spawner, request).await;
    assert!(spawner.is_running("dev-1"));
    start_ok(&runtime).await;
    despawn_ok(&spawner, "dev-1").await;
}

// 11 --------------------------------------------------------------------------------------------
#[tokio::test]
async fn the_identity_env_reaches_the_child() {
    let dir = TempDir::new().expect("temp dir for the identity dump");
    let out = dir.path().join("identity.txt");
    let child = FakeChild::with_body(&format!(
        "printf '%s|%s|%s|%s\\n' \"$OPENJIUWEN_TEAM_NAME\" \"$OPENJIUWEN_MEMBER_NAME\" \
         \"$OPENJIUWEN_SESSION_ID\" \"$OPENJIUWEN_MEMBER_WORKSPACE\" > '{}'",
        out.display()
    ));
    let spawner = spawner_for(&child);
    let request = MemberSpawnRequest::new("alpha", "dev-1", "sess-1", "/tmp/ws")
        .with_options(MemberSpawnOptions::from_kind(MemberSpawnKind::Process));

    let _runtime = spawn_ok(&spawner, request).await;

    assert!(
        wait_until(|| read_trimmed(&out) == "alpha|dev-1|sess-1|/tmp/ws").await,
        "the child must see the identity environment, got: {:?}",
        read_trimmed(&out)
    );
    despawn_ok(&spawner, "dev-1").await;
}

// 12 --------------------------------------------------------------------------------------------
#[tokio::test]
async fn an_explicit_env_wins_over_the_request_value() {
    let dir = TempDir::new().expect("temp dir for the member name dump");
    let out = dir.path().join("member-name.txt");
    let child = FakeChild::with_body(&format!(
        "printf '%s\\n' \"$OPENJIUWEN_MEMBER_NAME\" > '{}'",
        out.display()
    ));
    let command = sh_command(&child).with_env(MEMBER_NAME_ENV, "explicit");
    let spawner = ProcessMemberSpawner::new(ProcessMemberConfig { command });

    let _runtime = spawn_ok(&spawner, process_request("dev-1")).await;

    assert!(
        wait_until(|| read_trimmed(&out) == "explicit").await,
        "a pre-set environment value must survive the identity layer, got: {:?}",
        read_trimmed(&out)
    );
    despawn_ok(&spawner, "dev-1").await;
}

// 13 --------------------------------------------------------------------------------------------
#[tokio::test]
async fn respawning_a_member_replaces_the_registry_entry() {
    let child = FakeChild::serve();
    let spawner = spawner_for(&child);

    let first = spawn_ok(&spawner, process_request("dev-1")).await;
    let first_pid = spawner
        .child_pid("dev-1")
        .expect("the first child must report a pid");
    start_ok(&first).await;

    let second = spawn_ok(&spawner, process_request("dev-1")).await;
    let second_pid = spawner
        .child_pid("dev-1")
        .expect("the replacement child must report a pid");

    assert_ne!(first_pid, second_pid, "a re-spawn must start a new child");
    assert_eq!(spawner.live_members(), vec!["dev-1".to_string()]);
    assert!(spawner.is_running("dev-1"));
    #[cfg(target_os = "linux")]
    assert!(
        wait_process_gone(first_pid).await,
        "the replaced child {first_pid} must be stopped by the re-spawn"
    );

    start_ok(&second).await;
    despawn_ok(&spawner, "dev-1").await;
}

// 14 --------------------------------------------------------------------------------------------
#[tokio::test]
async fn a_failed_launch_registers_nothing_and_keeps_a_live_runtime() {
    let child = FakeChild::serve();
    let spawner = spawner_for(&child);
    let runtime = spawn_ok(&spawner, process_request("dev-1")).await;
    start_ok(&runtime).await;
    let pid = spawner.child_pid("dev-1").expect("pid");

    let broken = process_request("dev-1").with_options(
        MemberSpawnOptions::from_kind(MemberSpawnKind::Process).with_process(ProcessMemberConfig {
            command: StdioCommand::new(MISSING_PROGRAM),
        }),
    );
    let error = tokio::time::timeout(BOUND, spawner.spawn(broken))
        .await
        .expect("spawn did not return within the bound");
    let Err(error) = error else {
        panic!("a missing program must fail the spawn");
    };
    assert_eq!(error.kind(), ErrorKind::Config);
    assert_eq!(
        spawner.live_members(),
        vec!["dev-1".to_string()],
        "a failed launch must leave the live runtime alone"
    );
    assert_eq!(spawner.child_pid("dev-1"), Some(pid));

    let fresh = ProcessMemberSpawner::new(ProcessMemberConfig {
        command: StdioCommand::new(MISSING_PROGRAM),
    });
    assert!(fresh.spawn(process_request("dev-2")).await.is_err());
    assert_eq!(fresh.live_members(), Vec::<String>::new());

    despawn_ok(&spawner, "dev-1").await;
}

// 15 --------------------------------------------------------------------------------------------
#[tokio::test]
async fn child_pid_is_none_for_an_unknown_member() {
    let child = FakeChild::serve();
    let spawner = spawner_for(&child);
    assert!(spawner.child_pid("ghost").is_none());
    assert!(!spawner.is_running("ghost"));
    assert_eq!(spawner.live_members(), Vec::<String>::new());
}

// 16 --------------------------------------------------------------------------------------------
#[tokio::test]
async fn two_members_spawn_independently_and_are_listed_sorted() {
    let child = FakeChild::serve();
    let spawner = spawner_for(&child);

    let second = spawn_ok(&spawner, process_request("dev-2")).await;
    let first = spawn_ok(&spawner, process_request("dev-1")).await;

    assert_eq!(
        spawner.live_members(),
        vec!["dev-1".to_string(), "dev-2".to_string()]
    );
    let first_pid = spawner.child_pid("dev-1").expect("pid for dev-1");
    let second_pid = spawner.child_pid("dev-2").expect("pid for dev-2");
    assert_ne!(first_pid, second_pid);

    start_ok(&first).await;
    start_ok(&second).await;

    despawn_ok(&spawner, "dev-1").await;
    assert_eq!(spawner.live_members(), vec!["dev-2".to_string()]);
    despawn_ok(&spawner, "dev-2").await;
    assert_eq!(spawner.live_members(), Vec::<String>::new());
}

// 17 --------------------------------------------------------------------------------------------
#[tokio::test]
async fn the_spawner_is_object_safe_and_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<ProcessMemberSpawner>();

    let child = FakeChild::serve();
    let spawner: Arc<dyn MemberSpawner> = Arc::new(spawner_for(&child));
    let runtime = tokio::time::timeout(BOUND, spawner.spawn(process_request("dev-1")))
        .await
        .expect("spawn did not return within the bound")
        .expect("spawn failed");
    start_ok(&runtime).await;
    tokio::time::timeout(BOUND, spawner.despawn("dev-1"))
        .await
        .expect("despawn did not return within the bound")
        .expect("despawn failed");
}
