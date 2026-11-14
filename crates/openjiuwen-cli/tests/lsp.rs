//! Process-level tests for `openjiuwen lsp`.
//!
//! The binary is the only place where the command's contract is observable as an
//! operating system sees it: the exit code and the stream each verdict lands on.
//! Every case here is offline and no language server is required — the help and
//! grammar cases are answered by clap, the missing-file case is rejected by the
//! command before a server is even selected, and the failing-program case uses
//! `/bin/false` as an explicitly configured "server".
//!
//! Covered:
//!
//! 1. `lsp --help` exits `0` and documents the actions (`hover`, `definition`,
//!    `references`, `diagnostics`, `incoming-calls`, `outgoing-calls`) and the
//!    flags a caller needs (`--file`, `--line`, `--column`,
//!    `--server-command`, `--json`);
//! 2. an unknown action is a clap usage error: exit `2`, diagnostic on stderr,
//!    nothing on stdout;
//! 3. a `--file` that does not exist fails fast with `error: not_found:` naming
//!    the resolved path — no server is launched for a typo;
//! 4. a language-server program that cannot serve (`/bin/false`, or a program
//!    that does not exist) is reported as a typed `error: tool:` diagnostic
//!    (exit `1`) with empty stdout;
//! 5. the missing-file verdict is an error shape, not a document: `--json` still
//!    prints nothing on stdout.
//!
//! Every child is spawned through [`support::Sandbox`], so it runs in a temporary
//! working directory with a temporary `OPENJIUWEN_HOME` and a temporary `-w`
//! workspace, and every invocation is bounded instead of hanging the suite.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod support;

use support::{assert_exit, assert_success, describe, stderr, stdout, Sandbox};

/// Builds `-w <workspace> lsp <tail…>` for `sandbox`.
fn lsp_args(sandbox: &Sandbox, tail: &[&str]) -> Vec<String> {
    let [flag, workspace] = sandbox.workspace_flag();
    let mut args = vec![flag, workspace, "lsp".to_owned()];
    args.extend(tail.iter().map(|arg| (*arg).to_owned()));
    args
}

#[test]
fn help_lists_the_actions_and_flags() {
    let sandbox = Sandbox::new();
    let output = sandbox.run_blocking(&["lsp", "--help"]);
    let text = assert_success(&output);
    for action in [
        "hover",
        "definition",
        "references",
        "diagnostics",
        "incoming-calls",
        "outgoing-calls",
    ] {
        assert!(
            text.contains(action),
            "expected the `{action}` action in {text:?}"
        );
    }
    for flag in ["--file", "--line", "--column", "--server-command", "--json"] {
        assert!(text.contains(flag), "expected `{flag}` in {text:?}");
    }
}

#[test]
fn unknown_action_is_a_usage_error() {
    let sandbox = Sandbox::new();
    let output = sandbox.run_blocking(&["lsp", "definitely-not-an-action"]);
    assert_exit(&output, 2);
    let error = stderr(&output);
    assert!(
        error.contains("definitely-not-an-action"),
        "the diagnostic should name the rejected action: {}",
        describe(&output)
    );
    assert!(
        stdout(&output).is_empty(),
        "usage errors must not write to stdout: {}",
        describe(&output)
    );
}

#[test]
fn missing_file_is_not_found() {
    let sandbox = Sandbox::new();
    let output = sandbox.run_blocking(&lsp_args(
        &sandbox,
        &["diagnostics", "--file", "missing.py"],
    ));
    assert_exit(&output, 1);
    let error = stderr(&output);
    assert!(
        error.contains("error: not_found:"),
        "expected a `not_found` diagnostic: {}",
        describe(&output)
    );
    assert!(
        error.contains("missing.py"),
        "the diagnostic should name the missing file: {}",
        describe(&output)
    );
    assert!(
        stdout(&output).is_empty(),
        "a missing file must not produce a result on stdout: {}",
        describe(&output)
    );
}

#[test]
fn missing_file_with_json_keeps_stdout_empty() {
    let sandbox = Sandbox::new();
    let output = sandbox.run_blocking(&lsp_args(
        &sandbox,
        &["diagnostics", "--file", "missing.py", "--json"],
    ));
    assert_exit(&output, 1);
    assert!(
        stdout(&output).is_empty(),
        "an error is not a JSON document: {}",
        describe(&output)
    );
    assert!(
        stderr(&output).contains("error: not_found:"),
        "expected a `not_found` diagnostic: {}",
        describe(&output)
    );
}

#[cfg(unix)]
#[test]
fn a_failing_server_program_exits_one() {
    if !std::path::Path::new("/bin/false").exists() {
        eprintln!("skipping: /bin/false is not available on this host");
        return;
    }
    let sandbox = Sandbox::new();
    sandbox.write("workspace/main.py", "x = 1\n");
    let output = sandbox.run_blocking(&lsp_args(
        &sandbox,
        &[
            "diagnostics",
            "--file",
            "main.py",
            "--server-command",
            "/bin/false",
        ],
    ));
    assert_exit(&output, 1);
    let error = stderr(&output);
    assert!(
        error.contains("error: tool:"),
        "expected a typed tool diagnostic: {}",
        describe(&output)
    );
    assert!(
        stdout(&output).is_empty(),
        "a failed action must not produce a result on stdout: {}",
        describe(&output)
    );
}

#[test]
fn an_unspawnable_server_program_exits_one() {
    // Portable companion to the `/bin/false` case: this program cannot exist,
    // so the server launch fails deterministically on every host.
    let sandbox = Sandbox::new();
    sandbox.write("workspace/main.py", "x = 1\n");
    let output = sandbox.run_blocking(&lsp_args(
        &sandbox,
        &[
            "symbols",
            "--file",
            "main.py",
            "--server-command",
            "openjiuwen-no-such-language-server",
        ],
    ));
    assert_exit(&output, 1);
    let error = stderr(&output);
    assert!(
        error.contains("error: tool:"),
        "expected a typed tool diagnostic: {}",
        describe(&output)
    );
    assert!(
        stdout(&output).is_empty(),
        "a failed action must not produce a result on stdout: {}",
        describe(&output)
    );
}
