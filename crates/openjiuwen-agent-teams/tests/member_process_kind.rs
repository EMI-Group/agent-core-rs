//! The child-process member path, driven end to end against the real `team-member` binary.
//!
//! Python parity source: the OS-process branch of `openjiuwen/agent_teams/spawn/` plus the
//! `serve` entry point of the `openjiuwen-team-member` skill CLI. The in-crate `src/member/stdio*`
//! and `src/skill/serve*` unit suites pin each end over injected pipes; this suite pins the
//! *process* boundary between them — the part no in-process test can see: which stream a verdict
//! lands on, which exit code a shell observes, and whether the runtime really launches a child
//! when the blueprint selects the `process` spawn kind.
//!
//! 1. **`serve` is a real subcommand with a CLI contract.** `serve --help` is rendered by the
//!    scope-independent parser *before* a join descriptor is loaded, so it reaches **stdout** and
//!    exits **0**; `serve --definitely-not-a-flag` is a CLI-side error: clap's diagnostic on
//!    **stderr**, exit **2**. Neither needs `OPENJIUWEN_TEAM_JOIN` (its identity is the
//!    environment, not a descriptor).
//! 2. **The frames on the wire are the documented ones.** Driving `serve` directly as a child:
//!    an unparsable line is answered with exactly one `error` frame (the child keeps serving), a
//!    blank line is not a frame at all, and `{"type":"stop"}` ends the loop with exit **0**.
//! 3. **The `process` spawn kind launches `team-member serve`.** A blueprint whose
//!    `metadata["spawn"]` names `{"kind":"process","process":{"command":{…}}}` makes
//!    `spawn_member` run the configured program, so the teammate's roster row reaches `ready`
//!    through the child's `ready` handshake instead of the injected harness factory (which is
//!    never called for that member). `MemberStatus::Ready` is only written after
//!    `MemberRuntime::start` returns `Ok(())`, and for a child that means a `ready` frame arrived:
//!    a child that never answers leaves the row rolled back to `unstarted`.
//! 4. **A refusing child surfaces as a typed error, not a silently-ready member.** When the child
//!    cannot resolve a model (its `MODEL_*` environment is blanked through the launch command), it
//!    answers `start` with an `error` frame and exits: the spawn fails with an external-style
//!    error naming the refusing member process and carrying the child's own reason, and the row
//!    rolls back to `unstarted`.
//!    The `MODEL_*` values are *explicitly blank* on the command rather than merely absent, so the
//!    case stays deterministic even when the developer's shell exports a provider.
//!
//! Model credentials come from `openjiuwen_test_support`'s loopback `MockLlmServer`
//! (`MODEL_PROVIDER`/`MODEL_NAME`/`API_KEY`/`API_BASE` are exported to this process, and the child
//! inherits them): the native factory only *builds* the provider client during the handshake, so
//! the child answers `ready` without any request — asserted, so the process path is proven
//! offline. The server is still the honest endpoint, and if a future handshake does call it, the
//! call is served rather than hanging.
//!
//! Everything stays offline and bounded: the child cases wait through `support::BOUND` (or a
//! documented short observation window), every spawned process is reaped — explicitly on the happy
//! path, by `kill_on_drop` when a test fails — and no case is `#[ignore]`d.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![allow(clippy::missing_panics_doc)]
mod support;

use std::process::{Command, Output};
use std::time::Duration;

use openjiuwen_agent_teams::member::protocol::MemberResponse;
use openjiuwen_agent_teams::member::{MemberSpawnKind, MemberSpawner};
use openjiuwen_agent_teams::runtime::spec_meta::SPAWN_METADATA_KEY;
use openjiuwen_agent_teams::schema::{MemberLifetime, MemberStatus};
use openjiuwen_agent_teams::storage::TeamStore;
use openjiuwen_common::error::ErrorKind;
use openjiuwen_test_support::{MockLlmServer, TEST_API_KEY};
use serde_json::{json, Value};
use support::{sample_spec, wait_until, Fixture, BOUND};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader, Lines};
use tokio::process::{Child, ChildStdin, ChildStdout};
use tokio::time::timeout;

/// Program path of the binary under test, handed to the runtime as the spawn command.
const TEAM_MEMBER: &str = env!("CARGO_BIN_EXE_team-member");

/// Window in which the child must stay silent about a blank line.
///
/// A single short observation window (the `tests/CONTEXT.md` allowance): the child has already
/// been observed answering a real frame, so silence over this window is what "not a frame" means.
const SILENCE_WINDOW: Duration = Duration::from_millis(250);

/// Bound on the team-side waits, which cover a freshly launched child's own start-up.
///
/// Larger than [`BOUND`] because the child builds a native harness before it answers `ready`;
/// it is still a deadline, not a sleep, and the case finishes in well under a second.
const SPAWN_BOUND: Duration = Duration::from_secs(20);

/// Runs `team-member` with `args` and no join descriptor in the environment.
fn run_team_member(args: &[&str]) -> Output {
    Command::new(TEAM_MEMBER)
        .args(args)
        .env_remove("OPENJIUWEN_TEAM_JOIN")
        .output()
        .expect("the team-member binary should run")
}

/// Returns the exit code of `output`, panicking when the child was killed by a signal.
fn code_of(output: &Output) -> i32 {
    output
        .status
        .code()
        .expect("the child should exit normally, not be killed by a signal")
}

/// Renders a child's streams for a failure message.
fn describe(output: &Output) -> String {
    format!(
        "exit {}, stdout {:?}, stderr {:?}",
        code_of(output),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    )
}

/// Pins that `serve --help` renders the serve usage on stdout and exits `0`.
///
/// `serve` is dispatched before the descriptor stage, so no `OPENJIUWEN_TEAM_JOIN` exists and
/// stderr stays empty: help is a successful invocation, not a diagnostic.
#[test]
fn serve_help_renders_on_stdout_and_exits_zero() {
    let output = run_team_member(&["serve", "--help"]);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_eq!(code_of(&output), 0, "{}", describe(&output));
    assert!(
        output.stderr.is_empty(),
        "help belongs on stdout: {}",
        describe(&output)
    );
    // The scope-independent `serve` tree: its own usage line, the serve about text and the
    // descriptor flags it accepts (but never needs).
    assert!(
        stdout.contains("Usage: team-member serve [OPTIONS]"),
        "{}",
        describe(&output)
    );
    assert!(
        stdout.contains("Serve one member over stdin/stdout until stopped."),
        "{}",
        describe(&output)
    );
    assert!(
        stdout.contains("--descriptor-json"),
        "{}",
        describe(&output)
    );
}

/// Pins that an unknown `serve` flag is a CLI-side error: stderr, exit `2`, clean stdout.
#[test]
fn serve_rejects_an_unknown_flag_on_stderr() {
    let output = run_team_member(&["serve", "--definitely-not-a-flag"]);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(code_of(&output), 2, "{}", describe(&output));
    assert!(
        stderr.contains("--definitely-not-a-flag"),
        "the diagnostic must name the offending flag: {}",
        describe(&output)
    );
    assert!(
        output.stdout.is_empty(),
        "nothing may reach the protocol channel: {}",
        describe(&output)
    );
}

/// A live `team-member serve` child plus its frame channel, killed and reaped on every path.
///
/// The `Drop` guard is what keeps a failing assertion from leaking a process: `kill_on_drop` also
/// covers the `Child` tokio owns, and `start_kill` makes the intent explicit.
struct ServeChild {
    /// The child under test.
    child: Child,
    /// Its stdin: the leader-to-member frame channel.
    stdin: ChildStdin,
    /// Its stdout: the member-to-leader frame channel, one line per frame.
    lines: Lines<BufReader<ChildStdout>>,
    /// Its stderr, drained after the exit so a pipe buffer can never block the child.
    stderr: tokio::process::ChildStderr,
}

impl ServeChild {
    /// Launches `team-member serve` with piped stdio and no join descriptor.
    fn spawn() -> Self {
        let mut child = tokio::process::Command::new(TEAM_MEMBER)
            .arg("serve")
            .env_remove("OPENJIUWEN_TEAM_JOIN")
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .expect("the team-member serve child should spawn");
        let stdin = child.stdin.take().expect("the child stdin");
        let stdout = child.stdout.take().expect("the child stdout");
        let stderr = child.stderr.take().expect("the child stderr");
        Self {
            child,
            stdin,
            lines: BufReader::new(stdout).lines(),
            stderr,
        }
    }

    /// Writes one raw line to the child's stdin and flushes it.
    async fn write_line(&mut self, line: &str) {
        self.stdin
            .write_all(format!("{line}\n").as_bytes())
            .await
            .expect("writing to the serve child should succeed");
        self.stdin
            .flush()
            .await
            .expect("flushing the serve child's stdin should succeed");
    }

    /// Reads exactly one frame, panicking when none arrives within [`BOUND`].
    async fn next_frame(&mut self) -> MemberResponse {
        let line = timeout(BOUND, self.lines.next_line())
            .await
            .expect("the serve child did not answer within the bound")
            .expect("reading the serve child's stdout should succeed")
            .expect("the serve child closed stdout without answering");
        MemberResponse::parse_line(&line).unwrap_or_else(|error| {
            panic!("the serve child wrote a non-frame line {line:?}: {error}")
        })
    }

    /// Asserts that no frame arrives for [`SILENCE_WINDOW`].
    async fn expect_no_frame(&mut self, what: &str) {
        if let Ok(read) = timeout(SILENCE_WINDOW, self.lines.next_line()).await {
            let line = read.expect("reading the serve child's stdout should succeed");
            panic!("{what} must not produce a frame, but the child wrote {line:?}");
        }
    }

    /// Waits for the child to exit within [`BOUND`] and returns its exit code.
    ///
    /// The remaining stdout lines and the stderr diagnostics are collected too, so a wrong exit
    /// code is reported with everything the child said.
    async fn finish(mut self) -> (i32, Vec<String>, String) {
        let status = timeout(BOUND, self.child.wait())
            .await
            .expect("the serve child did not exit within the bound")
            .expect("waiting for the serve child should succeed");
        let mut trailing: Vec<String> = Vec::new();
        while let Ok(Some(line)) = self.lines.next_line().await {
            let line = line.trim();
            if !line.is_empty() {
                trailing.push(line.to_owned());
            }
        }
        let mut stderr = String::new();
        self.stderr
            .read_to_string(&mut stderr)
            .await
            .expect("draining the serve child's stderr should succeed");
        let code = status
            .code()
            .expect("the serve child should exit normally, not be killed by a signal");
        (code, trailing, stderr)
    }
}

impl Drop for ServeChild {
    /// Kills a child a failing assertion would otherwise leave running.
    ///
    /// The exit status is not awaited here (`Drop` cannot block safely); `kill_on_drop` hands the
    /// reaping to tokio's orphan reaper, so the process is still collected.
    fn drop(&mut self) {
        let _ = self.child.start_kill();
    }
}

/// Pins the serve loop's wire contract: one `error` frame per unparsable line, nothing for a blank
/// line, and exit `0` on `stop`.
#[tokio::test]
async fn serve_answers_garbage_keeps_serving_and_stops_on_ask() {
    let mut child = ServeChild::spawn();

    // An unparsable line is answered, and answered exactly once.
    child.write_line("this is not json").await;
    let frame = child.next_frame().await;
    assert!(
        matches!(frame, MemberResponse::Error { .. }),
        "an unparsable line must be answered with an error frame, got {frame:?}"
    );
    let message = frame.error_message().expect("the error frame message");
    assert!(
        message.contains("member request frame"),
        "the error must name the protocol frame: {message}"
    );
    assert!(
        message.contains("this is not json"),
        "the error must echo the offending line: {message}"
    );

    // A blank line is not a frame at all.
    child.write_line("").await;
    child.expect_no_frame("a blank line").await;

    // The child is still serving: a second bad line is answered the same way.
    child.write_line("neither is this").await;
    let frame = child.next_frame().await;
    assert!(
        matches!(frame, MemberResponse::Error { .. }),
        "the child must keep serving after a blank line, got {frame:?}"
    );
    assert!(
        frame
            .error_message()
            .is_some_and(|message| message.contains("neither is this")),
        "the second error must echo its own line: {frame:?}"
    );

    // `stop` ends the loop with exit 0, and the member was never started, so no further frame.
    child.write_line("{\"type\":\"stop\"}").await;
    let (code, trailing, stderr) = child.finish().await;
    assert_eq!(
        code, 0,
        "stop must exit 0; trailing stdout {trailing:?}, stderr {stderr:?}"
    );
    assert!(
        trailing.is_empty(),
        "stop must not write a further frame: {trailing:?}"
    );
    assert!(
        !stderr.contains("\"type\""),
        "stderr is diagnostics only, never protocol: {stderr:?}"
    );
}

/// Points a blueprint at the OS-process backend, launched with `command`.
///
/// The shape is the documented `metadata["spawn"]` directive: a bare `MemberSpawnOptions` object
/// whose `process.command` is the child's launch line.
fn with_process_spawn(spec: &mut openjiuwen_agent_teams::schema::TeamAgentSpec, command: &Value) {
    spec.metadata.insert(
        SPAWN_METADATA_KEY.to_owned(),
        json!({"kind": "process", "process": {"command": command}}),
    );
}

/// Exports the model environment the child inherits, restoring the previous values on drop.
///
/// The child resolves its model through `NativeMemberFactory::from_env`, so the credentials have
/// to reach this process's environment; the guard restores every variable afterwards so a sibling
/// case (or a developer's shell) never observes the fake endpoint.
struct ModelEnv {
    /// Previous values, `None` when the variable was unset.
    previous: Vec<(String, Option<String>)>,
}

impl ModelEnv {
    /// Sets `pairs` and remembers what it replaced.
    fn set(pairs: &[(&str, &str)]) -> Self {
        let mut previous = Vec::with_capacity(pairs.len());
        for (key, value) in pairs {
            previous.push(((*key).to_owned(), std::env::var(key).ok()));
            std::env::set_var(key, value);
        }
        Self { previous }
    }
}

impl Drop for ModelEnv {
    /// Restores every variable this guard touched.
    fn drop(&mut self) {
        for (key, value) in &self.previous {
            match value {
                Some(value) => std::env::set_var(key, value),
                None => std::env::remove_var(key),
            }
        }
    }
}

/// Pins that the `process` spawn kind launches the member binary and settles on the child's
/// `ready` handshake — with the in-process factory never involved.
#[tokio::test]
async fn process_spawn_kind_launches_the_member_binary() {
    // The offline endpoint the child would talk to; the handshake only builds the client.
    let mock = MockLlmServer::start().await;
    let _env = ModelEnv::set(&[
        ("MODEL_PROVIDER", "openai"),
        ("MODEL_NAME", "mock-model"),
        ("API_KEY", TEST_API_KEY),
        ("API_BASE", mock.base_url()),
    ]);

    let fixture = Fixture::new();
    let mut spec = sample_spec();
    with_process_spawn(
        &mut spec,
        &json!({"program": TEAM_MEMBER, "args": ["serve"]}),
    );
    let agent = fixture.activate(&spec, "session-a").await;

    timeout(SPAWN_BOUND, agent.spawn_member("dev-1"))
        .await
        .expect("the process spawn did not finish within the bound")
        .expect("the process member did not spawn; the child should have answered ready");

    // `ready` is only written after the runtime's start handshake succeeded, and for a child that
    // handshake is the `ready` frame: the row is the observable proof the child answered.
    wait_until("dev-1 to be ready", || {
        agent
            .backend()
            .member("dev-1")
            .is_some_and(|member| member.status() == MemberStatus::Ready)
    })
    .await;

    assert_eq!(
        agent.spawner().spawn_kind("dev-1"),
        Some(MemberSpawnKind::Process),
        "the team must materialize dev-1 through the process spawner"
    );
    assert!(
        fixture.factory.harness("dev-1").is_none(),
        "a process member must never reach the in-process harness factory"
    );
    let stored = fixture
        .store
        .member("alpha", "dev-1")
        .await
        .expect("reading the roster row should succeed")
        .expect("the dev-1 row is persisted");
    assert_eq!(stored.status, MemberStatus::Ready, "the store agrees");
    let runtime = agent
        .backend()
        .member("dev-1")
        .and_then(|member| member.runtime())
        .expect("a started process member owns a runtime");
    assert_eq!(
        runtime.session_id(),
        agent.session_id(),
        "the child runs under the team session"
    );
    // The handshake only builds the provider client: the child answered `ready` without a single
    // model request, which is what keeps the process path offline and deterministic.
    assert_eq!(
        mock.request_count(),
        0,
        "the ready handshake must not call the provider"
    );

    // Shutdown writes the `stop` frame and reaps the child, so no process outlives the case.
    let settled = timeout(
        SPAWN_BOUND,
        agent.shutdown_member("dev-1", MemberLifetime::Shutdown, true),
    )
    .await
    .expect("the process member did not shut down within the bound")
    .expect("shutting the process member down failed");
    assert_eq!(settled, MemberStatus::ShutDown);
    assert_eq!(
        agent.spawner().spawn_kind("dev-1"),
        None,
        "shutdown must despawn the child backend"
    );
    mock.shutdown().await;
}

/// Pins the failure shape of a refusing child: a typed error, a rolled-back row, and no
/// in-process fallback.
///
/// The launch command blanks the model environment, so the child cannot resolve a model however
/// the developer's shell is configured. It answers `start` with an `error` frame and exits, which
/// the runtime maps onto a refusal instead of pretending the member is ready.
#[tokio::test]
async fn a_refusing_child_fails_the_spawn_with_a_typed_error() {
    let fixture = Fixture::new();
    let mut spec = sample_spec();
    with_process_spawn(
        &mut spec,
        &json!({
            "program": TEAM_MEMBER,
            "args": ["serve"],
            "env": {
                "MODEL_PROVIDER": "",
                "MODEL_NAME": "",
                "API_KEY": "",
                "API_BASE": "",
            },
        }),
    );
    let agent = fixture.activate(&spec, "session-a").await;

    let error = timeout(SPAWN_BOUND, agent.spawn_member("dev-1"))
        .await
        .expect("the refusing spawn did not finish within the bound")
        .expect_err("a child that cannot resolve a model must fail the spawn");
    assert_eq!(
        error.kind(),
        ErrorKind::External,
        "unexpected error: {error}"
    );
    let message = error.to_string();
    assert!(
        message.contains("member process refused to start"),
        "the error must name the refusing process: {message}"
    );
    assert!(
        message.contains("model provider"),
        "the error must carry the child's own reason: {message}"
    );
    // A failed start rolls the row back, so the roster never claims a ready member.
    assert_eq!(
        agent
            .backend()
            .member("dev-1")
            .expect("the roster row exists")
            .status(),
        MemberStatus::Unstarted
    );
    assert!(
        fixture.factory.harness("dev-1").is_none(),
        "a process member must never fall back to the in-process factory"
    );
    // The child exited on its own; despawn still owns the registry entry, so reap it explicitly.
    agent
        .spawner()
        .despawn("dev-1")
        .await
        .expect("despawning the failed child should succeed");
}
