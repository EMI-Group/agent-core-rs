//! Offline integration suite for the CI gate runner.
//!
//! Python parity: `openjiuwen/rsi/harness_rsi/auto_harness/infra/ci_gate_runner.py`
//! (v0.1.20) — `_load_gates`, `_normalize_command`, `_get_diff_line_ranges`,
//! `_filter_*_by_line_ranges`, `_sanitize_failure_output`, and `run`.
//!
//! Every test is scripted: gate commands run through [`FakeCommandRunner`], so
//! no shell, no `make`, and no repository are involved.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::sync::Arc;
use std::time::Duration;

use openjiuwen_common::error::ErrorKind;
use openjiuwen_rsi::runtime::ci::{
    changed_line_ranges, default_gates_yaml, extract_commits, filter_findings, gate_outcome,
    normalize_command, parse_gates, sanitize_failure_output, shell_command, CiGateRunner,
    CiGateSpec, GateKind, FAIL_MARKER, OUTPUT_LIMIT,
};
use openjiuwen_rsi::runtime::command::{
    CommandOutcome, CommandRunner, FakeCommandRunner, FakeReply,
};

/// The shell a gate runs under, mirroring [`shell_command`].
fn shell() -> &'static str {
    if cfg!(windows) {
        "cmd.exe"
    } else {
        "sh"
    }
}

/// Builds a gate spec with an explicit kind and scope.
fn gate(name: &str, kind: GateKind, command: &str, commits: u32) -> CiGateSpec {
    CiGateSpec {
        name: name.to_owned(),
        kind,
        command: command.to_owned(),
        commits,
    }
}

#[test]
fn parse_gates_reads_entries_and_infers_kind() {
    let yaml = r#"
gates:
  - name: lint
    command: "make check COMMITS=1"
  - name: type-check
    kind: type-check
    command: "python -m mypy ."
  - name: custom
    kind: something-else
    command: "echo hi"
    commits: 3
"#;
    let gates = parse_gates(yaml);
    assert_eq!(gates.len(), 3);
    assert_eq!(gates[0].name, "lint");
    assert_eq!(gates[0].kind, GateKind::Lint);
    assert_eq!(gates[0].commits, 1);
    assert_eq!(gates[1].kind, GateKind::TypeCheck);
    assert_eq!(gates[1].commits, 0);
    assert_eq!(gates[2].kind, GateKind::Command);
    assert_eq!(gates[2].commits, 3);
}

#[test]
fn parse_gates_accepts_python_ci_gates_key() {
    let gates = parse_gates(default_gates_yaml());
    assert_eq!(gates.len(), 3);
    assert_eq!(
        gates
            .iter()
            .map(|gate| gate.name.as_str())
            .collect::<Vec<_>>(),
        vec!["lint", "type-check", "test"]
    );
    assert_eq!(gates[2].kind, GateKind::Command);
}

#[test]
fn parse_gates_fails_closed_on_bad_input() {
    assert_eq!(parse_gates("not: [valid: yaml").len(), 0);
    assert_eq!(parse_gates("{}").len(), 0);
    assert_eq!(parse_gates("gates: []").len(), 0);
    // An entry without a command cannot be run and must not become a gate.
    assert_eq!(parse_gates("gates:\n  - name: lint\n").len(), 0);
    assert_eq!(parse_gates("gates:\n  - command: echo hi\n").len(), 0);
}

#[test]
fn parse_gates_derives_commits_from_the_command() {
    assert_eq!(extract_commits("make check COMMITS=3"), 3);
    assert_eq!(extract_commits("make check COMMITS="), 0);
    assert_eq!(extract_commits("pytest -q"), 0);
    let gates = parse_gates("gates:\n  - name: lint\n    command: make check COMMITS=2\n");
    assert_eq!(gates[0].commits, 2);
}

#[test]
fn normalize_command_rewrites_only_the_known_shapes() {
    let python = Some("/w/.venv/bin/python");
    assert_eq!(
        normalize_command("python -m pytest tests", python, true),
        "/w/.venv/bin/python -m pytest tests"
    );
    assert_eq!(
        normalize_command("python -m pytest tests", None, true),
        "python -m pytest tests"
    );
    // An unquoted space ends the assignment, exactly like `shlex.split`.
    assert_eq!(
        normalize_command("make test TESTFLAGS=-q tests/x.py", python, false),
        "/w/.venv/bin/python -m pytest -q"
    );
    assert_eq!(
        normalize_command(
            "make test TESTFLAGS=\"-q tests/x.py -k foo\"",
            python,
            false
        ),
        "/w/.venv/bin/python -m pytest -q tests/x.py -k foo"
    );
    // make is available, so the command is left alone.
    assert_eq!(
        normalize_command("make test TESTFLAGS=-q", python, true),
        "make test TESTFLAGS=-q"
    );
    // Unknown make targets and unrelated commands are never rewritten.
    assert_eq!(
        normalize_command("make check COMMITS=1", python, false),
        "make check COMMITS=1"
    );
    assert_eq!(
        normalize_command("  cargo test  ", python, false),
        "  cargo test  "
    );
    // A python path with spaces is quoted.
    assert_eq!(
        normalize_command("python -m pytest", Some("/tmp/a b/python"), false),
        "'/tmp/a b/python' -m pytest"
    );
}

#[test]
fn changed_line_ranges_merges_hunks_across_files() {
    let diff = "diff --git a/src/a.py b/src/a.py\n\
                --- a/src/a.py\n\
                +++ b/src/a.py\n\
                @@ -1,2 +1,3 @@\n\
                +one\n\
                @@ -10 +11,2 @@\n\
                +two\n\
                --- a/src/b.py\n\
                +++ b/src/b.py\n\
                @@ -0,0 +1,2 @@\n\
                +three\n";
    let ranges = changed_line_ranges(diff);
    assert_eq!(
        ranges.get("src/a.py"),
        Some(&vec![(1, 3), (11, 12)]),
        "a missing count means one line"
    );
    assert_eq!(ranges.get("src/b.py"), Some(&vec![(1, 2)]));
    assert_eq!(ranges.len(), 2);
}

#[test]
fn changed_line_ranges_handles_deletions_and_binary_files() {
    let diff = "--- a/src/gone.py\n\
                +++ /dev/null\n\
                @@ -1,4 +0,0 @@\n\
                -a\n\
                --- a/src/mod.py\n\
                +++ b/src/mod.py\n\
                @@ -5,2 +5,0 @@\n\
                -b\n\
                --- a/logo.png\n\
                +++ b/logo.png\n\
                Binary files a/logo.png and b/logo.png differ\n";
    let ranges = changed_line_ranges(diff);
    assert!(!ranges.contains_key("src/gone.py"));
    assert_eq!(ranges.get("src/mod.py"), Some(&Vec::new()));
    assert_eq!(ranges.get("logo.png"), Some(&Vec::new()));
}

#[test]
fn changed_line_ranges_falls_back_to_the_whole_file() {
    let diff = "--- a/src/x.py\n+++ b/src/x.py\n@@ this is not a hunk @@\n";
    let ranges = changed_line_ranges(diff);
    assert_eq!(ranges.get("src/x.py"), Some(&vec![(1, 1_000_000)]));
    // A hunk without a preceding file header is ignored.
    assert!(changed_line_ranges("@@ -1 +1 @@\n").is_empty());
}

#[test]
fn filter_findings_keeps_only_changed_lines() {
    let ranges = BTreeMap::from([
        ("src/a.py".to_owned(), vec![(2, 3)]),
        ("src/b.py".to_owned(), vec![(10, 10)]),
    ]);
    let output = "src/a.py:2:1: E1 changed\n\
                  src/a.py:9:1: E2 unchanged\n\
                  src/a.py:3: error: changed too\n\
                  src/b.py:10: warning: changed\n\
                  src/c.py:1: dropped file\n\
                  not a finding at all\n\
                  ./src/a.py:2:1: normalised";
    let filtered = filter_findings(output, &ranges);
    assert_eq!(
        filtered,
        "src/a.py:2:1: E1 changed\n\
         src/a.py:3: error: changed too\n\
         src/b.py:10: warning: changed\n\
         ./src/a.py:2:1: normalised"
    );
    assert_eq!(filter_findings("", &ranges), "");
    assert_eq!(filter_findings("src/a.py:2: x", &BTreeMap::new()), "");
}

#[test]
fn sanitize_failure_output_keeps_head_and_failure_section() {
    assert_eq!(sanitize_failure_output("short", OUTPUT_LIMIT), "short");
    assert_eq!(sanitize_failure_output("  \n ", OUTPUT_LIMIT), "");

    let mut text = String::new();
    for index in 0..2000 {
        writeln!(text, "collection line {index}").unwrap();
    }
    text.push_str("==================== FAILURES ====================\n");
    for index in 0..200 {
        writeln!(text, "traceback line {index}").unwrap();
    }
    let sanitized = sanitize_failure_output(&text, OUTPUT_LIMIT);
    assert!(sanitized.chars().count() <= OUTPUT_LIMIT);
    assert!(sanitized.contains("collection line 0"), "head is kept");
    assert!(
        sanitized.contains("FAILURES"),
        "the failure section is kept"
    );
    assert!(sanitized.contains("traceback line 0"));
    assert!(sanitized.contains("…[truncated]…"));

    // Without a failure section the tail is kept.
    let long = "a".repeat(500);
    let sanitized = sanitize_failure_output(&long, 100);
    assert!(sanitized.chars().count() <= 100);
    assert!(sanitized.ends_with('a'));
    assert_eq!(sanitize_failure_output(&long, 0), "");
}

#[test]
fn gate_outcome_applies_the_pass_rule_and_the_marker() {
    let spec = gate("test", GateKind::Command, "pytest", 0);
    let passed = gate_outcome(
        "test",
        &spec,
        &CommandOutcome::success_with("1 passed"),
        None,
    );
    assert!(passed.passed);
    assert_eq!(passed.name, "test");
    assert_eq!(passed.output, "1 passed");

    let failed = gate_outcome(
        "test",
        &spec,
        &CommandOutcome::failure_with(1, "1 failed"),
        None,
    );
    assert!(!failed.passed);
    assert_eq!(failed.output, "1 failed");

    let marked = gate_outcome(
        "test",
        &spec,
        &CommandOutcome::success_with(format!("oops {FAIL_MARKER}")),
        None,
    );
    assert!(!marked.passed, "the marker fails an otherwise clean exit");
    assert!(!marked.output.contains(FAIL_MARKER));
    assert_eq!(marked.output, "oops");
}

#[test]
fn gate_outcome_scopes_lint_findings_to_changed_lines() {
    let spec = gate("lint", GateKind::Lint, "ruff check .", 1);
    let ranges = BTreeMap::from([("src/a.py".to_owned(), vec![(2, 2)])]);
    let outcome = CommandOutcome::success_with("src/a.py:2: E1\nsrc/a.py:9: E2");
    let scoped = gate_outcome("lint", &spec, &outcome, Some(&ranges));
    assert_eq!(scoped.output, "src/a.py:2: E1");
    // A command gate never filters, even when a scope is supplied.
    let command = gate("test", GateKind::Command, "pytest", 1);
    let unfiltered = gate_outcome("test", &command, &outcome, Some(&ranges));
    assert_eq!(unfiltered.output, outcome.stdout);
}

#[test]
fn shell_command_uses_the_platform_shell_and_gate_env() {
    let spec = shell_command("pytest -q");
    assert_eq!(spec.program, shell());
    let expected_args: Vec<String> = if cfg!(windows) {
        vec!["/c".to_owned(), "pytest -q".to_owned()]
    } else {
        vec!["-c".to_owned(), "pytest -q".to_owned()]
    };
    assert_eq!(spec.args, expected_args);
    assert_eq!(spec.env.get("CI").map(String::as_str), Some("1"));
    assert!(spec.env_remove.iter().any(|key| key == "VIRTUAL_ENV"));
    assert!(spec.merge_stderr);
}

#[tokio::test]
async fn run_all_passes_when_every_gate_passes() {
    let runner = Arc::new(
        FakeCommandRunner::new()
            .reply(
                FakeReply::outcome(shell(), CommandOutcome::success_with("clean"))
                    .with_args(["ruff"]),
            )
            .reply(
                FakeReply::outcome(shell(), CommandOutcome::success_with("1 passed"))
                    .with_args(["pytest"]),
            ),
    );
    let calls = Arc::clone(&runner);
    let gate_runner = CiGateRunner::new(runner, "/ws").with_gates(vec![
        gate("lint", GateKind::Lint, "ruff check .", 0),
        gate("test", GateKind::Command, "pytest -q", 0),
    ]);
    let report = gate_runner.run("all").await.unwrap();
    assert!(report.passed);
    assert_eq!(report.gates.len(), 2);
    assert_eq!(report.errors, "");
    assert_eq!(calls.call_count(), 2);
    // Gate commands run in the workspace, with the gate environment.
    let spec = &calls.calls()[0];
    assert_eq!(spec.cwd.as_deref(), Some(std::path::Path::new("/ws")));
    assert_eq!(spec.env.get("CI").map(String::as_str), Some("1"));
}

#[tokio::test]
async fn run_selects_by_name_and_reports_failures() {
    let runner = Arc::new(
        FakeCommandRunner::new()
            .reply(
                FakeReply::outcome(shell(), CommandOutcome::failure_with(1, "1 failed"))
                    .with_args(["pytest"]),
            )
            .reply(
                FakeReply::outcome(shell(), CommandOutcome::success_with("clean"))
                    .with_args(["ruff"]),
            ),
    );
    let calls = Arc::clone(&runner);
    let gate_runner = CiGateRunner::new(runner, "/ws").with_gates(vec![
        gate("lint", GateKind::Lint, "ruff check .", 0),
        gate("test", GateKind::Command, "pytest -q", 0),
    ]);
    let report = gate_runner.run("test").await.unwrap();
    assert!(!report.passed);
    assert_eq!(report.gates.len(), 1);
    assert!(report.errors.contains("[test]"));
    assert!(report.errors.contains("1 failed"));
    assert_eq!(calls.call_count(), 1);

    // `check` is Python's alias for the lint gate.
    let report = gate_runner.run("check").await.unwrap();
    assert!(report.passed);
    assert_eq!(report.gates[0].name, "lint");
}

#[tokio::test]
async fn run_fails_a_gate_that_prints_the_marker() {
    let runner = Arc::new(FakeCommandRunner::new().reply(FakeReply::outcome(
        shell(),
        CommandOutcome::success_with(format!("sub-command failed\n{FAIL_MARKER}")),
    )));
    let gate_runner = CiGateRunner::new(runner, "/ws").with_gates(vec![gate(
        "test",
        GateKind::Command,
        "pytest",
        0,
    )]);
    let report = gate_runner.run("all").await.unwrap();
    assert!(!report.passed);
    assert!(!report.gates[0].output.contains(FAIL_MARKER));
}

#[tokio::test]
async fn run_rejects_an_unknown_action_and_empty_configuration() {
    let runner = Arc::new(FakeCommandRunner::new());
    let gate_runner = CiGateRunner::new(runner, "/ws").with_gates(vec![gate(
        "test",
        GateKind::Command,
        "pytest",
        0,
    )]);
    let error = gate_runner.run("nope").await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("nope"));

    let empty = CiGateRunner::new(Arc::new(FakeCommandRunner::new()), "/ws").with_gates(Vec::new());
    let report = empty.run("all").await.unwrap();
    assert!(!report.passed, "an empty gate list must not pass vacuously");
    assert!(report.errors.contains("no CI gates configured"));
}

#[tokio::test]
async fn run_normalises_make_and_injects_the_python_interpreter() {
    let runner = Arc::new(FakeCommandRunner::new().fallback(CommandOutcome::success_with("ok")));
    let calls = Arc::clone(&runner);
    let gate_runner = CiGateRunner::new(runner, "/ws")
        .with_python("/ws/.venv/bin/python")
        .with_timeout(Duration::from_secs(30))
        .with_gates(vec![gate(
            "test",
            GateKind::Command,
            "make test TESTFLAGS=-q",
            0,
        )]);
    let report = gate_runner.run("all").await.unwrap();
    assert!(report.passed);
    let spec = &calls.calls()[0];
    if which::which("make").is_ok() {
        assert!(spec.args[1].starts_with("make test"), "make is available");
    } else {
        assert!(spec.args[1].starts_with("/ws/.venv/bin/python -m pytest -q"));
    }
    assert_eq!(
        spec.env.get("AUTO_HARNESS_PYTHON").map(String::as_str),
        Some("/ws/.venv/bin/python")
    );
    assert_eq!(spec.timeout, Some(Duration::from_secs(30)));
}

#[tokio::test]
async fn lint_gate_filters_output_with_the_scripted_diff() {
    let diff = "--- a/src/a.py\n+++ b/src/a.py\n@@ -2,0 +2 @@\n+x = 1\n";
    let runner = Arc::new(
        FakeCommandRunner::new()
            .reply(
                FakeReply::outcome("git", CommandOutcome::success_with(diff))
                    .with_args(["diff", "-U0", "HEAD~1.."]),
            )
            .reply(
                FakeReply::outcome(
                    shell(),
                    CommandOutcome::success_with(
                        "src/a.py:2: E1 changed\nsrc/a.py:9: E2 unchanged",
                    ),
                )
                .with_args(["ruff"]),
            ),
    );
    let gate_runner = CiGateRunner::new(runner, "/ws").with_gates(vec![gate(
        "lint",
        GateKind::Lint,
        "ruff check .",
        1,
    )]);
    let report = gate_runner.run("all").await.unwrap();
    assert!(report.passed);
    assert_eq!(report.gates[0].output, "src/a.py:2: E1 changed");
}

#[tokio::test]
async fn lint_gate_falls_back_to_unfiltered_output_when_the_diff_fails() {
    let runner = Arc::new(
        FakeCommandRunner::new()
            .reply(FakeReply::outcome(
                "git",
                CommandOutcome::failure_with(128, "fatal: bad revision"),
            ))
            .reply(
                FakeReply::outcome(shell(), CommandOutcome::success_with("src/a.py:99: E2"))
                    .with_args(["ruff"]),
            ),
    );
    let gate_runner = CiGateRunner::new(runner, "/ws").with_gates(vec![gate(
        "lint",
        GateKind::Lint,
        "ruff check .",
        1,
    )]);
    let report = gate_runner.run("all").await.unwrap();
    assert_eq!(report.gates[0].output, "src/a.py:99: E2");
}

#[test]
fn gate_selectors_expose_the_configuration() {
    let runner: Arc<dyn CommandRunner> = Arc::new(FakeCommandRunner::new());
    let gate_runner = CiGateRunner::new(runner, "/ws")
        .with_python("python3")
        .with_timeout(Duration::from_secs(5))
        .with_gates(vec![gate("lint", GateKind::Lint, "ruff", 0)]);
    assert_eq!(gate_runner.gates().len(), 1);
    assert_eq!(gate_runner.python(), Some("python3"));
    assert_eq!(gate_runner.timeout(), Some(Duration::from_secs(5)));
    assert_eq!(gate_runner.workspace(), std::path::Path::new("/ws"));
}
