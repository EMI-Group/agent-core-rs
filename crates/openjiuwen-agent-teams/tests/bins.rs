//! The crate's executable entry points, driven as real child processes.
//!
//! Python parity source: the `openjiuwen-team-member` skill CLI and the `openjiuwen-team-mcp`
//! server entry points (`openjiuwen/agent_teams/skill/cli.py`, `…/mcp/server.py`). This suite pins
//! the *process* contract of the two binaries in `src/bin/` — the part no in-process unit test can
//! see: which stream each verdict lands on and which exit code a shell observes.
//!
//! 1. **`--help` is success on stdout.** Help is answered by the scope parser *before* the join
//!    descriptor is loaded, so `--help` (before or after the subcommand) renders clap's help on
//!    **stdout** and exits **0** even with no descriptor — Python argparse parity.
//!    `serve --help` is the same contract one stage earlier: the scope-independent serve stage
//!    answers it before the descriptor scan, so it needs no descriptor either.
//!    Every other CLI-side problem (bad flag, missing argument, malformed or absent descriptor)
//!    still goes through `fail_cli`: diagnostic on **stderr**, exit **2**.
//! 2. **The descriptor stage runs first for real commands.** No arguments at all reports the
//!    missing `OPENJIUWEN_TEAM_JOIN` descriptor, not a usage line; an unknown-to-this-port `--team`
//!    flag also dies in the descriptor stage (exit `2`).
//! 3. **A team this process does not host exits `1`** with
//!    `no live team '<team>' in this process` on stderr — the documented "unresolved" contract,
//!    distinct from a CLI-side error (`2`).
//! 4. **`openjiuwen-team-mcp` without a descriptor exits `1`** before any stdio serving with a
//!    stderr diagnostic: the entry point installs a stderr log subscriber, so the
//!    `tracing::error!` from the descriptor load is rendered while stdout stays a clean wire.
//!    (`openjiuwen_common::logging` defaults to stdout, which would corrupt the protocol channel,
//!    so the binary forces `LogWriter::Stderr`.)
//!
//! Everything is offline and deterministic: no network, no model, and the only waiting is a short
//! bounded poll on the MCP child. Every child runs with `OPENJIUWEN_TEAM_JOIN` removed so an
//! ambient value in the developer's shell cannot change a result. Because each case is a separate
//! process, the CLI cases start with an empty process-global runtime registry: nothing is hosted,
//! so no registry state is read or written by this suite.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![allow(clippy::missing_panics_doc)]
use std::process::{Command, Output};

/// A member-scope join descriptor with every optional field left at its default.
const MEMBER_DESCRIPTOR: &str = r#"{"team_name":"alpha","member_name":"dev-1","scope":"member"}"#;

/// Runs the `team-member` binary with `args` and no `OPENJIUWEN_TEAM_JOIN`, returning its output.
fn run_team_member(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_team-member"))
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

/// Pins that `--help` renders on stdout and exits `0` with no join descriptor at all.
///
/// Help is answered by the scope parser before the descriptor is loaded (Python argparse parity), so
/// the child never reports the missing `OPENJIUWEN_TEAM_JOIN` and stderr stays empty.
#[test]
fn help_exits_zero_on_stdout() {
    let output = run_team_member(&["--help"]);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_eq!(code_of(&output), 0, "{}", describe(&output));
    assert!(
        output.stderr.is_empty(),
        "help belongs on stdout: {}",
        describe(&output)
    );
    // clap's rendered help: the usage line names the program, the tree lists `inbox`.
    assert!(
        stdout.contains("Usage: team-member"),
        "{}",
        describe(&output)
    );
    assert!(stdout.contains("inbox"), "{}", describe(&output));
}

/// Pins that the scope-independent `serve` stage answers `--help` on stdout and exits `0`.
///
/// `serve` is dispatched *before* the join descriptor is scanned (it takes its identity from the
/// environment, not from a descriptor), so its help needs no `OPENJIUWEN_TEAM_JOIN` either. The
/// rendered usage is the serve tree's own, which is why it names the program *and* the command.
#[test]
fn serve_help_exits_zero_on_stdout() {
    let output = run_team_member(&["serve", "--help"]);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_eq!(code_of(&output), 0, "{}", describe(&output));
    assert!(
        output.stderr.is_empty(),
        "serve help belongs on stdout: {}",
        describe(&output)
    );
    assert!(
        stdout.contains("Usage: team-member serve"),
        "{}",
        describe(&output)
    );
    // `serve` accepts the connection flags for launcher parity, so its usage lists them.
    assert!(
        stdout.contains("--descriptor-file"),
        "{}",
        describe(&output)
    );
}

/// Pins that a bare invocation reports the missing descriptor, not a usage line.
#[test]
fn no_arguments_reports_the_missing_descriptor() {
    let output = run_team_member(&[]);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(code_of(&output), 2, "{}", describe(&output));
    assert!(!stderr.trim().is_empty(), "{}", describe(&output));
    // The connection stage runs first, so the diagnostic is the descriptor error.
    assert!(
        stderr.contains("OPENJIUWEN_TEAM_JOIN"),
        "{}",
        describe(&output)
    );
}

/// Pins that an unknown connection flag is a CLI-side error (exit `2`), not a live-team failure.
///
/// `--team` is not a flag in this port; the descriptor scan ignores it, the descriptor then fails to
/// load, and the command tree is never reached.
#[test]
fn unknown_connection_flag_is_a_cli_side_error() {
    let output = run_team_member(&["--team", "alpha", "member", "inbox"]);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(code_of(&output), 2, "{}", describe(&output));
    assert!(!stderr.trim().is_empty(), "{}", describe(&output));
}

/// Pins the "no live team" contract: a well-formed descriptor for a team this process does not
/// host exits `1` with the resolver's message on stderr.
///
/// The member-scope tree takes the command word directly (`inbox`, not `member inbox`): this port
/// has no `member <name> …` wrapper subcommand, so `inbox` is the invocation that reaches the
/// resolver without a parse error.
#[test]
fn unhosted_team_exits_one() {
    let output = run_team_member(&["--descriptor-json", MEMBER_DESCRIPTOR, "inbox"]);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(code_of(&output), 1, "{}", describe(&output));
    assert!(
        stderr.contains("no live team 'alpha' in this process"),
        "{}",
        describe(&output)
    );
}

/// Pins that the MCP entry point fails fast without a join descriptor: exit `1`, clean stdout, a
/// stderr diagnostic.
///
/// The descriptor load fails before any stdio serving, so no MCP frame is written. The failure goes
/// through `tracing::error!` and the entry point installs a **stderr** subscriber, so the diagnostic
/// is rendered on stderr while stdout stays a clean protocol channel.
#[cfg(feature = "mcp")]
#[test]
fn mcp_without_descriptor_exits_one_with_a_diagnostic() {
    use std::process::Stdio;
    use std::time::{Duration, Instant};

    /// How long the child may take to fail on the descriptor load.
    const DEADLINE: Duration = Duration::from_secs(60);
    /// Poll step of the bounded wait.
    const STEP: Duration = Duration::from_millis(25);

    let mut child = Command::new(env!("CARGO_BIN_EXE_openjiuwen-team-mcp"))
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .env_remove("OPENJIUWEN_TEAM_JOIN")
        .spawn()
        .expect("the openjiuwen-team-mcp binary should spawn");

    let started = Instant::now();
    loop {
        match child
            .try_wait()
            .expect("polling the MCP child should succeed")
        {
            Some(_) => break,
            None if started.elapsed() >= DEADLINE => {
                let _ = child.kill();
                panic!(
                    "openjiuwen-team-mcp did not exit within {DEADLINE:?} without a join descriptor"
                );
            }
            None => std::thread::sleep(STEP),
        }
    }

    let output = child
        .wait_with_output()
        .expect("collecting the MCP child output should succeed");
    assert_eq!(code_of(&output), 1, "{}", describe(&output));
    assert!(
        output.stdout.is_empty(),
        "no MCP frame is served without a descriptor: {}",
        describe(&output)
    );
    assert!(
        !String::from_utf8_lossy(&output.stderr).trim().is_empty(),
        "the stderr subscriber should render the tracing diagnostic: {}",
        describe(&output)
    );
}
