//! Process-level smoke tests for the `openjiuwen` binary.
//!
//! These pin the wiring between `src/main.rs` and `openjiuwen_harness::cli` as
//! an operating system sees it: the exit code, and which stream each verdict
//! lands on. `--help`, `--version` and parse errors are answered by clap before
//! any command runs, so every case here is offline — no model, no network, no
//! terminal.
//!
//! Covered:
//!
//! 1. `--version` prints the binary's name and version on **stdout** and exits
//!    `0` (clap's `version` attribute, backed by the workspace version);
//! 2. `--help` lists both subcommands (`run`, `chat`) on **stdout**, exits `0`,
//!    and the subcommand help is reachable (`run --help` lists its output
//!    format flag);
//! 3. an unknown subcommand is a usage error: nonzero exit (clap uses `2`) with
//!    the diagnostic on **stderr** and nothing on stdout.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::process::{Command, Output};

/// Runs the compiled binary with `args`, returning its captured output.
fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_openjiuwen"))
        .args(args)
        .output()
        .expect("the openjiuwen binary should run")
}

/// Returns the child's exit code, panicking when it was killed by a signal.
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

/// Returns whether some line of `text` starts with `word` (a clap help row).
fn has_row(text: &str, word: &str) -> bool {
    text.lines()
        .any(|line| line.split_whitespace().next() == Some(word))
}

#[test]
fn version_flag_reports_the_crate_version() {
    let output = run(&["--version"]);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_eq!(code_of(&output), 0, "{}", describe(&output));
    assert!(
        stdout.contains(env!("CARGO_PKG_VERSION")),
        "expected version {} in {stdout:?}",
        env!("CARGO_PKG_VERSION")
    );
    assert!(
        stdout.contains("openjiuwen"),
        "expected the binary name in {stdout:?}"
    );
}

#[test]
fn help_lists_run_and_chat() {
    let output = run(&["--help"]);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_eq!(code_of(&output), 0, "{}", describe(&output));
    assert!(has_row(&stdout, "run"), "no `run` command in {stdout:?}");
    assert!(has_row(&stdout, "chat"), "no `chat` command in {stdout:?}");
}

#[test]
fn run_help_lists_the_output_format_flag() {
    let output = run(&["run", "--help"]);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_eq!(code_of(&output), 0, "{}", describe(&output));
    assert!(
        stdout.contains("--output-format"),
        "expected `--output-format` in {stdout:?}"
    );
}

#[test]
fn unknown_subcommand_is_a_usage_error() {
    let output = run(&["definitely-not-a-command"]);
    assert_ne!(code_of(&output), 0, "{}", describe(&output));
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("unrecognized subcommand"),
        "{}",
        describe(&output)
    );
    assert!(
        output.stdout.is_empty(),
        "usage errors must not write to stdout: {}",
        describe(&output)
    );
}
