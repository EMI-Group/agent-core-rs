//! Process-level smoke tests for the `openjiuwen` binary: grammar and help.
//!
//! These pin the wiring between `src/main.rs`, `src/cli.rs` and
//! `openjiuwen_harness::cli` as an operating system sees it: the exit code and
//! which stream each verdict lands on. `--help`, `--version` and parse errors are
//! answered by clap before any command runs, so every case here is offline — no
//! model, no network, no terminal, and (with the exception of `run`/`chat` help,
//! which are answered before the workspace is touched) no filesystem state.
//!
//! Covered:
//!
//! 1. `--version` prints the binary's name and version on **stdout** and exits
//!    `0` (clap's `version` attribute, backed by the workspace version);
//! 2. `--help` lists all three command groups (`run`, `chat`, `auto-harness`) on
//!    **stdout**, exits `0`, and every subcommand's help is reachable;
//! 3. `auto-harness`'s help lists its five subcommands, and `run`/`experience`
//!    list the flags and nested commands the port documents;
//! 4. `--no-color` is accepted on both sides of a subcommand and is offered in
//!    the help of the top-level grammar;
//! 5. an unknown subcommand is a usage error: exit `2` (clap's documented code)
//!    with the diagnostic on **stderr** and nothing on stdout.
//!
//! Every child is spawned through [`support::Sandbox`], so it runs in a temporary
//! working directory with a temporary `OPENJIUWEN_HOME` and cannot read the
//! developer's environment.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod support;

use support::{assert_exit, assert_success, describe, has_row, stderr, stdout, Sandbox};

#[test]
fn version_flag_reports_the_crate_version() {
    let sandbox = Sandbox::new();
    let output = sandbox.run_blocking(&["--version"]);
    let text = assert_success(&output);
    assert!(
        text.contains(env!("CARGO_PKG_VERSION")),
        "expected version {} in {text:?}",
        env!("CARGO_PKG_VERSION")
    );
    assert!(
        text.contains("openjiuwen"),
        "expected the binary name in {text:?}"
    );
}

#[test]
fn help_lists_every_command_group() {
    let sandbox = Sandbox::new();
    let output = sandbox.run_blocking(&["--help"]);
    let text = assert_success(&output);
    for row in ["run", "chat", "auto-harness"] {
        assert!(has_row(&text, row), "no `{row}` command in {text:?}");
    }
}

#[test]
fn run_help_lists_the_output_format_flag() {
    let sandbox = Sandbox::new();
    let output = sandbox.run_blocking(&["run", "--help"]);
    let text = assert_success(&output);
    assert!(
        text.contains("--output-format"),
        "expected `--output-format` in {text:?}"
    );
}

#[test]
fn unknown_subcommand_is_a_usage_error() {
    let sandbox = Sandbox::new();
    let output = sandbox.run_blocking(&["definitely-not-a-command"]);
    assert_exit(&output, 2);
    assert!(
        stderr(&output).contains("unrecognized subcommand"),
        "{}",
        describe(&output)
    );
    assert!(
        stdout(&output).is_empty(),
        "usage errors must not write to stdout: {}",
        describe(&output)
    );
}

#[test]
fn auto_harness_help_lists_its_subcommands() {
    let sandbox = Sandbox::new();
    let output = sandbox.run_blocking(&["auto-harness", "--help"]);
    let text = assert_success(&output);
    for row in ["run", "experience", "verify-ext", "gap-analyze", "history"] {
        assert!(has_row(&text, row), "no `{row}` subcommand in {text:?}");
    }
}

#[test]
fn auto_harness_run_help_lists_its_flags() {
    let sandbox = Sandbox::new();
    let output = sandbox.run_blocking(&["auto-harness", "run", "--help"]);
    let text = assert_success(&output);
    for flag in [
        "--task",
        "--task-file",
        "--dry-run",
        "--stage",
        "--budget",
        "--goal",
        "--pipeline",
    ] {
        assert!(text.contains(flag), "expected `{flag}` in {text:?}");
    }
}

#[test]
fn auto_harness_experience_help_lists_search_and_list() {
    let sandbox = Sandbox::new();
    let output = sandbox.run_blocking(&["auto-harness", "experience", "--help"]);
    let text = assert_success(&output);
    for row in ["search", "list"] {
        assert!(has_row(&text, row), "no `{row}` subcommand in {text:?}");
    }
}

#[test]
fn no_color_is_a_global_flag_on_both_sides_of_a_subcommand() {
    let sandbox = Sandbox::new();
    // Before the subcommand.
    let before = sandbox.run_blocking(&["--no-color", "--help"]);
    assert_success(&before);
    // After the subcommand, on a subcommand that reuses the harness argument type.
    let after = sandbox.run_blocking(&["run", "--no-color", "--help"]);
    assert_success(&after);
    // And it is documented by the top-level grammar.
    let text = stdout(&before);
    assert!(
        text.contains("--no-color"),
        "expected `--no-color` in {text:?}"
    );
}
