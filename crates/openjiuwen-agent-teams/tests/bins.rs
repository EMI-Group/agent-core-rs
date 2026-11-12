//! The crate's executable entry points, driven as real child processes.
//!
//! Python parity source: the `openjiuwen-team-member` skill CLI and the `openjiuwen-team-mcp`
//! server entry points (`openjiuwen/agent_teams/skill/cli.py`, `…/mcp/server.py`). This suite pins
//! the *process* contract of the two binaries in `src/bin/` — the part no in-process unit test can
//! see: which stream each verdict lands on and which exit code a shell observes.
//!
//! 1. **`--help` is a CLI-side error.** The descriptor is resolved *before* the clap subcommand
//!    parse, and every failure in that stage goes through `fail_cli`; clap's `DisplayHelp` is
//!    mapped onto `Error::validation`, so rendered help lands on **stderr** with exit **2**. (A
//!    library change that treats `--help` as success on stdout is out of this binary's scope — this
//!    suite pins what the port actually does.)
//! 2. **The descriptor stage runs first.** No arguments at all reports the missing
//!    `OPENJIUWEN_TEAM_JOIN` descriptor, not a usage line; an unknown-to-this-port `--team` flag
//!    also dies in the descriptor stage (exit `2`).
//! 3. **A team this process does not host exits `1`** with
//!    `no live team '<team>' in this process` on stderr — the documented "unresolved" contract,
//!    distinct from a CLI-side error (`2`).
//! 4. **`openjiuwen-team-mcp` without a descriptor exits `1`** before any stdio serving and writes
//!    nothing to stderr: the failure is reported through `tracing::error!` and no subscriber is
//!    installed by that thin entry point.
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

/// Pins that `--help` renders on stderr and exits `2`, not `0` on stdout.
///
/// The library loads the descriptor *before* parsing the command tree, and maps clap's
/// `DisplayHelp` onto `Error::validation`, so help travels the CLI-side error path. The descriptor
/// must be supplied for the child to reach the parse stage at all.
#[test]
fn help_is_reported_as_a_cli_side_error() {
    let output = run_team_member(&["--descriptor-json", MEMBER_DESCRIPTOR, "--help"]);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(code_of(&output), 2, "{}", describe(&output));
    assert!(
        output.stdout.is_empty(),
        "help belongs on stderr in this port: {}",
        describe(&output)
    );
    // clap's rendered member-scope help: the usage line names the program, the tree lists `inbox`.
    assert!(stderr.contains("team-member"), "{}", describe(&output));
    assert!(stderr.contains("inbox"), "{}", describe(&output));
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

/// Pins that the MCP entry point fails fast without a join descriptor: exit `1`, nothing on stderr.
///
/// The descriptor load fails before any stdio serving, so no MCP frame is written either. The
/// failure goes through `tracing::error!` and this thin entry point installs no subscriber, so
/// stderr stays empty — that is the pinned behaviour, not an oversight.
#[cfg(feature = "mcp")]
#[test]
fn mcp_without_descriptor_exits_one_silently() {
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
        output.stderr.is_empty(),
        "no subscriber is installed, so the tracing error is not rendered: {}",
        describe(&output)
    );
}
