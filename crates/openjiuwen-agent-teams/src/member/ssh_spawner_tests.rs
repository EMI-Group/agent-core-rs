//! Tests for the SSH-transport member spawner in [`super::ssh_spawner`].
//!
//! Everything here is offline and deterministic.
//! The pure cases assert the exact `ssh` argv the spawner renders; the process cases point the
//! `program` at a [`FakeChild`](super::test_child::FakeChild) script that ignores the `ssh`
//! arguments and behaves as the remote member, so the JSON-lines protocol is exercised without any
//! network.
//! Every await is bounded by [`BOUND`] and polling uses short sleeps against a deadline.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use openjiuwen_common::error::ErrorKind;

use super::runtime::{MemberObserver, MemberRuntime, ObserverEvent, RecordingObserver};
use super::spawn::{MemberSpawnRequest, MemberSpawner};
use super::spawn_config::{MemberSpawnKind, MemberSpawnOptions, SshTransportSpec};
use super::ssh_spawner::SshMemberSpawner;
use super::test_child::FakeChild;
use crate::schema::RoundOutcome;

/// Upper bound for any single await in this suite.
const BOUND: Duration = Duration::from_secs(10);

/// The smallest spec that passes validation: default `ssh` program, port 22, a 15 s connect
/// timeout and `exec` enabled.
fn minimal_spec() -> SshTransportSpec {
    SshTransportSpec {
        host: "build-box".to_string(),
        remote_command: "team-member serve".to_string(),
        ..SshTransportSpec::default()
    }
}

/// A request that fills every identity field.
fn request() -> MemberSpawnRequest {
    MemberSpawnRequest::new("alpha", "dev-1", "sess-1", "/tmp/member-ws")
}

/// The identity prologue plus the rendering rules for the four filled identity fields, as the
/// single quoted remote word `ssh` would receive.
fn filled_remote_word(member_command: &str) -> String {
    format!(
        "'export OPENJIUWEN_TEAM_NAME='\\''alpha'\\''; \
         export OPENJIUWEN_MEMBER_NAME='\\''dev-1'\\''; \
         export OPENJIUWEN_SESSION_ID='\\''sess-1'\\''; \
         export OPENJIUWEN_MEMBER_WORKSPACE='\\''/tmp/member-ws'\\''; {member_command}'"
    )
}

/// A request asking for a specific SSH endpoint through its options.
fn ssh_request(host: &str, remote_command: &str) -> MemberSpawnRequest {
    let spec = SshTransportSpec {
        host: host.to_string(),
        remote_command: remote_command.to_string(),
        ..SshTransportSpec::default()
    };
    request().with_options(MemberSpawnOptions::from_kind(MemberSpawnKind::Ssh).with_ssh(spec))
}

/// Spawns `request` with a bound and returns the member runtime.
async fn spawn_member(
    spawner: &SshMemberSpawner,
    request: MemberSpawnRequest,
) -> Arc<dyn MemberRuntime> {
    tokio::time::timeout(BOUND, spawner.spawn(request))
        .await
        .expect("spawn did not return within the bound")
        .expect("spawn failed")
}

/// Despawns `member_name` with a bound.
async fn despawn(spawner: &SshMemberSpawner, member_name: &str) {
    tokio::time::timeout(BOUND, spawner.despawn(member_name))
        .await
        .expect("despawn did not return within the bound")
        .expect("despawn failed");
}

/// Drains `observer` until `done` holds for the events seen so far, bounded by [`BOUND`].
async fn collect_until(
    observer: &RecordingObserver,
    mut done: impl FnMut(&[ObserverEvent]) -> bool,
) -> Vec<ObserverEvent> {
    tokio::time::timeout(BOUND, async {
        let mut events = Vec::new();
        loop {
            events.extend(observer.take());
            if done(&events) {
                return events;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .expect("timed out waiting for member events")
}

// --- pure argv cases -------------------------------------------------------------------------

#[test]
fn minimal_argv_carries_the_identity_prologue_and_exec_before_the_member_command() {
    let spawner = SshMemberSpawner::new(minimal_spec());
    assert_eq!(
        spawner.argv_for(&request()).expect("argv"),
        vec![
            "ssh".to_string(),
            "-p".to_string(),
            "22".to_string(),
            "-o".to_string(),
            "ConnectTimeout=15".to_string(),
            "--".to_string(),
            "build-box".to_string(),
            filled_remote_word("exec team-member serve"),
        ]
    );
}

#[test]
fn a_full_spec_renders_every_flag_in_order_and_omits_exec() {
    let mut options = BTreeMap::new();
    options.insert("Compression".to_string(), "yes".to_string());
    options.insert("ServerAliveInterval".to_string(), "30".to_string());
    let spec = SshTransportSpec {
        program: "ssh".to_string(),
        host: "build-box".to_string(),
        port: 2222,
        username: Some("builder".to_string()),
        key_file: Some(PathBuf::from("/home/builder/.ssh/id_ed25519")),
        password: None,
        agent: false,
        known_hosts: Some(PathBuf::from("/home/builder/.ssh/known_hosts")),
        disable_host_key_check: true,
        connect_timeout_s: None,
        use_exec: false,
        options,
        remote_command: "team-member serve".to_string(),
    };
    let spawner = SshMemberSpawner::new(spec);
    assert_eq!(
        spawner.argv_for(&request()).expect("argv"),
        vec![
            "ssh".to_string(),
            "-p".to_string(),
            "2222".to_string(),
            "-i".to_string(),
            "/home/builder/.ssh/id_ed25519".to_string(),
            "-o".to_string(),
            "UserKnownHostsFile=/home/builder/.ssh/known_hosts".to_string(),
            "-o".to_string(),
            "StrictHostKeyChecking=no".to_string(),
            "-o".to_string(),
            "Compression=yes".to_string(),
            "-o".to_string(),
            "ServerAliveInterval=30".to_string(),
            "--".to_string(),
            "builder@build-box".to_string(),
            filled_remote_word("team-member serve"),
        ]
    );
}

#[test]
fn an_empty_host_is_a_config_error() {
    let spec = SshTransportSpec {
        host: String::new(),
        remote_command: "team-member serve".to_string(),
        ..SshTransportSpec::default()
    };
    let error = SshMemberSpawner::new(spec)
        .argv_for(&request())
        .expect_err("empty host must not render an argv");
    assert_eq!(error.kind(), ErrorKind::Config);
}

#[test]
fn port_zero_is_a_validation_error() {
    let spec = SshTransportSpec {
        port: 0,
        ..minimal_spec()
    };
    let error = SshMemberSpawner::new(spec)
        .argv_for(&request())
        .expect_err("port zero must not render an argv");
    assert_eq!(error.kind(), ErrorKind::Validation);
}

#[test]
fn an_empty_remote_command_is_a_config_error_even_with_identity_fields() {
    // The prologue alone would make the rendered word non-empty, so the raw command is validated
    // before the prologue is prepended.
    let spec = SshTransportSpec {
        host: "build-box".to_string(),
        remote_command: String::new(),
        ..SshTransportSpec::default()
    };
    let error = SshMemberSpawner::new(spec)
        .command_for(&request())
        .expect_err("an empty remote command must not become a prologue-only command");
    assert_eq!(error.kind(), ErrorKind::Config);
}

#[test]
fn empty_request_values_are_skipped_from_the_prologue() {
    let spawner = SshMemberSpawner::new(minimal_spec());
    let request = MemberSpawnRequest::new("", "dev-1", "", "");
    assert_eq!(
        spawner.remote_command_for(&request),
        "export OPENJIUWEN_MEMBER_NAME='dev-1'; team-member serve"
    );
    assert_eq!(
        spawner
            .argv_for(&request)
            .expect("argv")
            .last()
            .map(String::as_str),
        Some("'export OPENJIUWEN_MEMBER_NAME='\\''dev-1'\\''; exec team-member serve'")
    );
}

#[test]
fn a_request_level_ssh_spec_overrides_the_spawner_spec() {
    let spawner = SshMemberSpawner::new(minimal_spec());
    let request = ssh_request("other-box", "team-member serve");
    let argv = spawner.argv_for(&request).expect("argv");
    assert!(argv.contains(&"other-box".to_string()), "{argv:?}");
    assert!(!argv.contains(&"build-box".to_string()), "{argv:?}");
    // The spawner's own spec stays untouched by the override.
    assert_eq!(spawner.spec().host, "build-box");
}

#[test]
fn a_process_kind_request_does_not_pick_up_ssh_options() {
    let spawner = SshMemberSpawner::new(minimal_spec());
    let request = request().with_options(
        MemberSpawnOptions::from_kind(MemberSpawnKind::Process).with_ssh(SshTransportSpec {
            host: "other-box".to_string(),
            remote_command: "team-member serve".to_string(),
            ..SshTransportSpec::default()
        }),
    );
    let argv = spawner.argv_for(&request).expect("argv");
    assert!(argv.contains(&"build-box".to_string()), "{argv:?}");
    assert!(!argv.contains(&"other-box".to_string()), "{argv:?}");
}

#[test]
fn password_and_agent_add_no_argv_element() {
    let plain = SshMemberSpawner::new(minimal_spec());
    let decorated = SshMemberSpawner::new(SshTransportSpec {
        password: Some("hunter2".to_string()),
        agent: true,
        ..minimal_spec()
    });
    let plain_argv = plain.argv_for(&request()).expect("argv");
    let decorated_argv = decorated.argv_for(&request()).expect("argv");
    assert_eq!(plain_argv, decorated_argv);
    assert!(
        !decorated_argv.iter().any(|arg| arg.contains("hunter2")),
        "the password must never reach the argv: {decorated_argv:?}"
    );
}

#[test]
fn single_quotes_in_the_remote_command_are_escaped() {
    let spawner = SshMemberSpawner::new(SshTransportSpec {
        host: "build-box".to_string(),
        remote_command: "team-member serve --note 'hi'".to_string(),
        ..SshTransportSpec::default()
    });
    let request = MemberSpawnRequest::new("", "", "", "");
    let argv = spawner.argv_for(&request).expect("argv");
    assert_eq!(
        argv.last().map(String::as_str),
        Some("'exec team-member serve --note '\\''hi'\\'''")
    );
}

#[test]
fn remote_command_for_returns_the_raw_prologue_and_command() {
    let spawner = SshMemberSpawner::new(minimal_spec());
    // `exec` and the surrounding quotes are rendering rules of `command_for`, not part of the raw
    // remote command.
    assert_eq!(
        spawner.remote_command_for(&request()),
        "export OPENJIUWEN_TEAM_NAME='alpha'; \
         export OPENJIUWEN_MEMBER_NAME='dev-1'; \
         export OPENJIUWEN_SESSION_ID='sess-1'; \
         export OPENJIUWEN_MEMBER_WORKSPACE='/tmp/member-ws'; \
         team-member serve"
    );
}

// --- process cases ---------------------------------------------------------------------------

#[tokio::test]
async fn a_spawned_member_speaks_the_protocol_over_the_ssh_child() {
    let child = FakeChild::serve();
    let spawner = SshMemberSpawner::new(SshTransportSpec {
        program: child.program(),
        ..minimal_spec()
    });
    let runtime = spawn_member(
        &spawner,
        MemberSpawnRequest::new("alpha", "dev-1", "ssh-session", "/tmp/member-ws"),
    )
    .await;
    assert_eq!(runtime.session_id(), "ssh-session");
    assert!(spawner.is_running("dev-1"));
    assert!(spawner.child_pid("dev-1").is_some());
    assert_eq!(spawner.live_members(), vec!["dev-1".to_string()]);

    let observer = Arc::new(RecordingObserver::new());
    runtime.subscribe(Arc::clone(&observer) as Arc<dyn MemberObserver>);
    tokio::time::timeout(BOUND, runtime.start(None))
        .await
        .expect("start did not return within the bound")
        .expect("start failed");
    tokio::time::timeout(BOUND, runtime.send("hi".to_owned(), false))
        .await
        .expect("send did not return within the bound")
        .expect("send failed");

    let events = collect_until(&observer, |events| {
        events.contains(&ObserverEvent::Round(RoundOutcome::Finished))
    })
    .await;
    assert!(
        events.contains(&ObserverEvent::Chunk(serde_json::json!({"delta": "hi"}))),
        "the sent content must come back as a chunk: {events:?}"
    );

    despawn(&spawner, "dev-1").await;
    assert!(!spawner.is_running("dev-1"));
}

#[tokio::test]
async fn a_missing_ssh_binary_is_a_config_error() {
    let spawner = SshMemberSpawner::new(SshTransportSpec {
        program: "openjiuwen-no-such-ssh-xyz".to_string(),
        ..minimal_spec()
    });
    let Err(error) = tokio::time::timeout(BOUND, spawner.spawn(request()))
        .await
        .expect("spawn did not return within the bound")
    else {
        panic!("a missing ssh program must not spawn a member");
    };
    assert_eq!(error.kind(), ErrorKind::Config);
    assert_eq!(spawner.live_members(), Vec::<String>::new());
}

#[tokio::test]
async fn despawn_stops_the_child_and_is_idempotent() {
    let child = FakeChild::serve();
    let spawner = SshMemberSpawner::new(SshTransportSpec {
        program: child.program(),
        ..minimal_spec()
    });
    let _runtime = spawn_member(&spawner, request()).await;
    assert!(spawner.is_running("dev-1"));

    despawn(&spawner, "dev-1").await;
    assert!(!spawner.is_running("dev-1"));
    assert_eq!(spawner.live_members(), Vec::<String>::new());
    assert_eq!(spawner.child_pid("dev-1"), None);

    // A second despawn (or a never-spawned member) is not an error.
    despawn(&spawner, "dev-1").await;
    despawn(&spawner, "never-spawned").await;
}

#[tokio::test]
async fn a_re_spawn_replaces_the_live_member() {
    let child = FakeChild::serve();
    let spawner = SshMemberSpawner::new(SshTransportSpec {
        program: child.program(),
        ..minimal_spec()
    });
    let _first = spawn_member(&spawner, request()).await;
    let _second = spawn_member(&spawner, request()).await;

    assert_eq!(spawner.live_members(), vec!["dev-1".to_string()]);
    assert!(spawner.is_running("dev-1"));
    assert!(spawner.child_pid("dev-1").is_some());

    despawn(&spawner, "dev-1").await;
    assert_eq!(spawner.live_members(), Vec::<String>::new());
}

#[tokio::test]
async fn live_members_are_sorted_by_name() {
    let child = FakeChild::serve();
    let spawner = SshMemberSpawner::new(SshTransportSpec {
        program: child.program(),
        ..minimal_spec()
    });
    let zeta = MemberSpawnRequest::new("alpha", "zeta", "sess-1", "/tmp/zeta");
    let alpha = MemberSpawnRequest::new("alpha", "alpha", "sess-1", "/tmp/alpha");
    let _zeta = spawn_member(&spawner, zeta).await;
    let _alpha = spawn_member(&spawner, alpha).await;

    assert_eq!(
        spawner.live_members(),
        vec!["alpha".to_string(), "zeta".to_string()]
    );
    despawn(&spawner, "zeta").await;
    despawn(&spawner, "alpha").await;
    assert_eq!(spawner.live_members(), Vec::<String>::new());
}
