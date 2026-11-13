//! Offline integration suite for the Python toolchain, pytest report parsing,
//! and subprocess program evaluation.
//!
//! Python parity: `harness_rsi/auto_harness/stages/verify.py` (`_check_pip_available`,
//! `_bootstrap_pip`, `_run_pip_install`, `_run_pytest_file`) and
//! `artifact_rsi/program_opt/{execution,reply_format}.py` (stage → run → read the
//! reply back).
//!
//! Every command runs through a scripted [`CommandRunner`]; pytest report files
//! are real files under `tempfile` directories, and nothing here needs a Python
//! interpreter, a network, or a shell.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use openjiuwen_common::error::{ErrorKind, Result};
use openjiuwen_rsi::artifact_rsi::program_opt::ReplyFormat;
use openjiuwen_rsi::runtime::command::{
    CommandOutcome, CommandRunner, CommandSpec, FakeCommandRunner, FakeReply,
};
use openjiuwen_rsi::runtime::pytest::{
    parse_json_report, parse_junit_xml, parse_report, PytestReport, TestFailure,
};
use openjiuwen_rsi::runtime::python::{
    discover_python, pytest_args, PythonToolchain, ReportFormat,
};
use openjiuwen_rsi::runtime::subprocess::{
    parse_files_reply, parse_tagged_reply, SubprocessConfig, SubprocessEvaluator,
};

/// A runner that records calls and can write one reply file into the cwd.
struct ReportRunner {
    outcome: CommandOutcome,
    /// `(file name, content)` written into the command's working directory.
    reply: Option<(String, String)>,
    calls: Mutex<Vec<CommandSpec>>,
}

impl ReportRunner {
    fn new(outcome: CommandOutcome) -> Self {
        Self {
            outcome,
            reply: None,
            calls: Mutex::new(Vec::new()),
        }
    }

    fn with_reply(mut self, name: &str, content: &str) -> Self {
        self.reply = Some((name.to_owned(), content.to_owned()));
        self
    }

    fn calls(&self) -> Vec<CommandSpec> {
        self.calls.lock().unwrap().clone()
    }

    /// Returns the value of the first argument starting with `prefix`.
    fn argument(&self, prefix: &str) -> Option<String> {
        self.calls()
            .iter()
            .flat_map(|spec| spec.args.iter())
            .find_map(|arg| arg.strip_prefix(prefix).map(str::to_owned))
    }
}

#[async_trait]
impl CommandRunner for ReportRunner {
    #[allow(clippy::unnecessary_literal_bound)] // the trait contract is `&str`
    fn name(&self) -> &str {
        "report"
    }

    async fn run(&self, spec: CommandSpec) -> Result<CommandOutcome> {
        self.calls.lock().unwrap().push(spec.clone());
        if let (Some(cwd), Some((name, content))) = (spec.cwd.as_ref(), self.reply.as_ref()) {
            std::fs::write(cwd.join(name), content).unwrap();
            // The pytest report path is absolute, so also honour it directly.
            for arg in &spec.args {
                for prefix in ["--json-report-file=", "--junitxml="] {
                    if let Some(path) = arg.strip_prefix(prefix) {
                        if let Some(parent) = Path::new(path).parent() {
                            let _ = std::fs::create_dir_all(parent);
                        }
                        std::fs::write(path, content).unwrap();
                    }
                }
            }
        }
        Ok(self.outcome.clone())
    }
}

/// A runner that writes a reply file into the command's working directory.
struct ProgramRunner {
    outcome: CommandOutcome,
    reply: Option<(String, String)>,
    calls: Mutex<Vec<CommandSpec>>,
}

impl ProgramRunner {
    fn new(outcome: CommandOutcome) -> Self {
        Self {
            outcome,
            reply: None,
            calls: Mutex::new(Vec::new()),
        }
    }

    fn with_reply(mut self, name: &str, content: &str) -> Self {
        self.reply = Some((name.to_owned(), content.to_owned()));
        self
    }

    fn calls(&self) -> Vec<CommandSpec> {
        self.calls.lock().unwrap().clone()
    }
}

#[async_trait]
impl CommandRunner for ProgramRunner {
    #[allow(clippy::unnecessary_literal_bound)] // the trait contract is `&str`
    fn name(&self) -> &str {
        "program"
    }

    async fn run(&self, spec: CommandSpec) -> Result<CommandOutcome> {
        self.calls.lock().unwrap().push(spec.clone());
        if let (Some(cwd), Some((name, content))) = (spec.cwd.as_ref(), self.reply.as_ref()) {
            std::fs::write(cwd.join(name), content).unwrap();
        }
        Ok(self.outcome.clone())
    }
}

/// The venv interpreter path the toolchain computes on this platform.
fn expected_venv_python(venv: &Path) -> PathBuf {
    if cfg!(windows) {
        venv.join("Scripts").join("python.exe")
    } else {
        venv.join("bin").join("python")
    }
}

const JSON_REPORT: &str = r#"{
  "summary": {"total": 4, "passed": 2, "failed": 1, "error": 1, "skipped": 0},
  "tests": [
    {"nodeid": "tests/test_a.py::test_ok", "outcome": "passed"},
    {"nodeid": "tests/test_a.py::test_ok2", "outcome": "passed"},
    {"nodeid": "tests/test_a.py::test_bad", "outcome": "failed",
     "call": {"longrepr": "assert 1 == 2\nsecond line"}},
    {"nodeid": "tests/test_a.py::test_err", "outcome": "error",
     "setup": {"longrepr": "fixture exploded"}}
  ]
}"#;

#[test]
fn json_report_counters_and_failures() {
    let report = parse_json_report(JSON_REPORT).unwrap();
    assert_eq!(
        report,
        PytestReport {
            total: 4,
            passed: 2,
            failed: 1,
            errors: 1,
            skipped: 0,
            failures: vec![
                TestFailure {
                    name: "tests/test_a.py::test_bad".to_owned(),
                    message: "assert 1 == 2".to_owned(),
                },
                TestFailure {
                    name: "tests/test_a.py::test_err".to_owned(),
                    message: "fixture exploded".to_owned(),
                },
            ],
        }
    );
}

#[test]
fn json_report_derives_missing_counters_from_the_test_list() {
    let text = r#"{"tests": [
        {"nodeid": "a", "outcome": "passed"},
        {"nodeid": "b", "outcome": "failed", "call": {"crash": {"message": "boom"}}},
        {"nodeid": "c", "outcome": "skipped"},
        {"nodeid": "d", "outcome": "xfailed"}
    ]}"#;
    let report = parse_json_report(text).unwrap();
    assert_eq!(report.total, 4);
    assert_eq!(report.failed, 1);
    assert_eq!(report.errors, 0);
    assert_eq!(report.skipped, 2);
    assert_eq!(report.passed, 1, "total - failed - errors - skipped");
    assert_eq!(report.failures[0].message, "boom");
}

#[test]
fn json_report_is_strict_about_its_root() {
    let error = parse_json_report("not json").unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    let error = parse_json_report("[]").unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    // An empty object is a valid but empty report.
    let empty = parse_json_report("{}").unwrap();
    assert_eq!(empty.total, 0);
}

#[test]
fn junit_report_reads_attributes_any_order_and_self_closing_cases() {
    let xml = r#"<?xml version="1.0" encoding="utf-8"?>
<testsuites>
  <!-- a comment with <testsuite tests="99"> inside -->
  <testsuite failures="1" tests="3" skipped="1" errors="0" name="pytest">
    <testcase name="test_ok" classname="tests.test_a" time="0.1"/>
    <testcase classname="tests.test_a" name="test_bad">
      <failure message="assert &quot;x&quot; &lt; 2 &amp;&amp; ok">long traceback
second line</failure>
    </testcase>
    <testcase classname="tests.test_a" name="test_skip">
      <skipped message="no reason"/>
    </testcase>
  </testsuite>
</testsuites>"#;
    let report = parse_junit_xml(xml).unwrap();
    assert_eq!(report.total, 3);
    assert_eq!(report.failed, 1);
    assert_eq!(report.errors, 0);
    assert_eq!(report.skipped, 1);
    assert_eq!(report.passed, 1);
    assert_eq!(
        report.failures,
        vec![TestFailure {
            name: "tests.test_a::test_bad".to_owned(),
            message: "assert \"x\" < 2 && ok".to_owned(),
        }]
    );
}

#[test]
fn junit_report_derives_counters_and_reads_failure_bodies() {
    let xml = r#"<testsuite>
    <testcase classname="c" name="a"/>
    <testcase classname="c" name="b"><error>setup failed: no module</error></testcase>
    <testcase classname="c" name="c"><failure type="AssertionError"/></testcase>
</testsuite>"#;
    let report = parse_junit_xml(xml).unwrap();
    assert_eq!(report.total, 3);
    assert_eq!(report.errors, 1);
    assert_eq!(report.failed, 1);
    assert_eq!(report.passed, 1);
    assert_eq!(
        report.failures,
        vec![
            TestFailure {
                name: "c::b".to_owned(),
                message: "setup failed: no module".to_owned(),
            },
            TestFailure {
                name: "c::c".to_owned(),
                message: String::new(),
            },
        ]
    );
}

#[test]
fn junit_report_rejects_a_wrong_root() {
    let error = parse_junit_xml("<html><body/></html>").unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.to_string().contains("testsuite"));
    assert_eq!(
        parse_junit_xml("").unwrap_err().kind(),
        ErrorKind::Validation
    );
}

#[test]
fn report_dispatch_follows_the_format() {
    assert_eq!(
        parse_report(JSON_REPORT, ReportFormat::Json).unwrap().total,
        4
    );
    let xml = r#"<testsuite tests="1"><testcase classname="c" name="a"/></testsuite>"#;
    assert_eq!(parse_report(xml, ReportFormat::JUnit).unwrap().passed, 1);
    assert_eq!(
        parse_report(JSON_REPORT, ReportFormat::JUnit)
            .unwrap_err()
            .kind(),
        ErrorKind::Validation
    );
}

#[test]
fn pytest_args_select_the_plugin_flags() {
    assert_eq!(
        pytest_args(ReportFormat::Json, Path::new("/tmp/r.json")),
        vec![
            "--json-report".to_owned(),
            "--json-report-file=/tmp/r.json".to_owned()
        ]
    );
    assert_eq!(
        pytest_args(ReportFormat::JUnit, Path::new("/tmp/r.xml")),
        vec!["--junitxml=/tmp/r.xml".to_owned()]
    );
}

#[test]
fn discover_python_is_host_dependent_but_never_panics() {
    let found = discover_python();
    if let Some(path) = found {
        assert_ne!(path.as_str(), "");
    }
}

#[tokio::test]
async fn ensure_venv_creates_once_and_detects_an_existing_environment() {
    let dir = tempfile::tempdir().unwrap();
    let runner = Arc::new(FakeCommandRunner::new().reply(FakeReply::outcome(
        "python3",
        CommandOutcome::success_with("created"),
    )));
    let calls = Arc::clone(&runner);
    let mut toolchain = PythonToolchain::new(runner, dir.path()).with_python("python3");

    let setup = toolchain.ensure_venv().await.unwrap();
    let venv = dir.path().join(".venv");
    assert!(setup.created);
    assert_eq!(setup.venv_dir, venv);
    assert_eq!(setup.venv_python, expected_venv_python(&venv));
    assert_eq!(setup.python, "python3");
    assert_eq!(calls.call_count(), 1);
    let spec = &calls.calls()[0];
    assert_eq!(spec.program, "python3");
    assert_eq!(spec.args, vec!["-m", "venv", venv.to_str().unwrap()]);
    assert_eq!(spec.cwd.as_deref(), Some(dir.path()));

    // A second call sees the marker file and does not run python again.
    std::fs::create_dir_all(&venv).unwrap();
    std::fs::write(venv.join("pyvenv.cfg"), "home = /usr\n").unwrap();
    let again = toolchain.ensure_venv().await.unwrap();
    assert!(!again.created);
    assert_eq!(calls.call_count(), 1);
}

#[tokio::test]
async fn ensure_venv_reports_a_failing_creation() {
    let dir = tempfile::tempdir().unwrap();
    let runner = Arc::new(FakeCommandRunner::new().reply(FakeReply::outcome(
        "python3",
        CommandOutcome::failure_with(1, "No module named venv"),
    )));
    let mut toolchain = PythonToolchain::new(runner, dir.path()).with_python("python3");
    let error = toolchain.ensure_venv().await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Tool);
    assert!(error.to_string().contains("python -m venv"));
}

#[tokio::test]
async fn pip_version_and_install_report_their_outcomes() {
    let dir = tempfile::tempdir().unwrap();
    let runner = Arc::new(
        FakeCommandRunner::new()
            .reply(
                FakeReply::outcome("python3", CommandOutcome::success_with("pip 24.0 from /x"))
                    .with_args(["pip", "--version"]),
            )
            .reply(
                FakeReply::outcome(
                    "python3",
                    CommandOutcome::failure_with(1, "ERROR: no matching distribution"),
                )
                .with_args(["install", "missing"]),
            )
            .reply(
                FakeReply::outcome(
                    "python3",
                    CommandOutcome::success_with("installed requests"),
                )
                .with_args(["install", "requests"]),
            ),
    );
    let calls = Arc::clone(&runner);
    let toolchain = PythonToolchain::new(runner, dir.path()).with_python("python3");

    assert_eq!(toolchain.pip_version().await.unwrap(), "pip 24.0 from /x");

    let failed = toolchain
        .pip_install(&["missing".to_owned()])
        .await
        .unwrap();
    assert!(!failed.success);
    assert_eq!(
        failed.output,
        "pip_install_failed: ERROR: no matching distribution"
    );

    let success = toolchain
        .pip_install(&["requests".to_owned()])
        .await
        .unwrap();
    assert!(success.success);
    assert_eq!(success.output, "installed requests");
    assert_eq!(
        calls.calls()[1].args,
        vec!["-m", "pip", "install", "missing"]
    );
}

#[tokio::test]
async fn pip_install_truncates_failure_text_to_500_characters() {
    let dir = tempfile::tempdir().unwrap();
    let noise = "e".repeat(600);
    let runner = Arc::new(FakeCommandRunner::new().reply(FakeReply::outcome(
        "python3",
        CommandOutcome::failure_with(1, noise),
    )));
    let toolchain = PythonToolchain::new(runner, dir.path()).with_python("python3");
    let outcome = toolchain.pip_install(&["x".to_owned()]).await.unwrap();
    assert!(!outcome.success);
    assert!(outcome.output.starts_with("pip_install_failed: "));
    assert_eq!(
        outcome
            .output
            .trim_start_matches("pip_install_failed: ")
            .chars()
            .count(),
        500
    );
}

#[tokio::test]
async fn pip_install_requirements_file_uses_dash_r() {
    let dir = tempfile::tempdir().unwrap();
    let runner = Arc::new(FakeCommandRunner::new().fallback(CommandOutcome::success_with("ok")));
    let calls = Arc::clone(&runner);
    let toolchain = PythonToolchain::new(runner, dir.path()).with_python("python3");
    let outcome = toolchain
        .pip_install_requirements_file(Path::new("/ws/requirements.txt"))
        .await
        .unwrap();
    assert!(outcome.success);
    assert_eq!(
        calls.calls()[0].args,
        vec!["-m", "pip", "install", "-r", "/ws/requirements.txt"]
    );
}

#[tokio::test]
async fn ensure_pip_bootstraps_only_when_the_probe_fails() {
    let dir = tempfile::tempdir().unwrap();
    let runner = Arc::new(
        FakeCommandRunner::new()
            .reply(
                FakeReply::outcome(
                    "python3",
                    CommandOutcome::failure_with(1, "No module named pip"),
                )
                .with_args(["pip", "--version"]),
            )
            .reply(
                FakeReply::outcome("python3", CommandOutcome::success_with("installed"))
                    .with_args(["ensurepip"]),
            ),
    );
    let calls = Arc::clone(&runner);
    let toolchain = PythonToolchain::new(runner, dir.path()).with_python("python3");
    toolchain.ensure_pip().await.unwrap();
    assert_eq!(calls.call_count(), 2);
    assert_eq!(calls.calls()[1].args, vec!["-m", "ensurepip", "--upgrade"]);

    let broken = Arc::new(
        FakeCommandRunner::new().fallback(CommandOutcome::failure_with(1, "no pip, no ensurepip")),
    );
    let toolchain = PythonToolchain::new(broken, dir.path()).with_python("python3");
    let error = toolchain.ensure_pip().await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Tool);
}

#[tokio::test]
async fn run_pytest_parses_the_json_report_and_cleans_up() {
    let dir = tempfile::tempdir().unwrap();
    let runner = Arc::new(
        ReportRunner::new(CommandOutcome::success_with("2 passed, 1 failed"))
            .with_reply("report.json", JSON_REPORT),
    );
    let calls = Arc::clone(&runner);
    let venv = dir.path().join(".venv");
    let toolchain = PythonToolchain::new(runner, dir.path())
        .with_python("python3")
        .with_venv(&venv);
    let outcome = toolchain
        .run_pytest(Path::new("tests"), &["-k".to_owned(), "smoke".to_owned()])
        .await
        .unwrap();
    assert!(outcome.passed);
    assert_eq!(outcome.exit_code, Some(0));
    assert_eq!(outcome.format, Some(ReportFormat::Json));
    let report = outcome.report.unwrap();
    assert_eq!(report.total, 4);
    assert_eq!(report.failed, 1);
    assert_eq!(report.failures.len(), 2);

    let spec = &calls.calls()[0];
    assert_eq!(spec.program, "python3");
    assert_eq!(
        spec.args[..6],
        ["-m", "pytest", "tests", "-q", "-o", "addopts="]
    );
    assert_eq!(spec.args[6], "--json-report");
    assert!(spec.args[7].starts_with("--json-report-file="));
    assert_eq!(spec.args[8..], ["-k", "smoke"]);
    assert_eq!(spec.env.get("CI").map(String::as_str), Some("1"));
    assert_eq!(
        spec.env.get("VIRTUAL_ENV").map(String::as_str),
        Some(venv.to_str().unwrap())
    );
    assert_eq!(spec.cwd.as_deref(), Some(dir.path()));
    assert!(spec.merge_stderr);

    let report_path = calls.argument("--json-report-file=").unwrap();
    assert!(
        !Path::new(&report_path).exists(),
        "the temporary report directory is removed"
    );
}

#[tokio::test]
async fn run_pytest_supports_junit_and_failed_exits() {
    let dir = tempfile::tempdir().unwrap();
    let xml = r#"<testsuite tests="2" failures="1"><testcase classname="c" name="a"/><testcase classname="c" name="b"><failure message="boom"/></testcase></testsuite>"#;
    let runner = Arc::new(
        ReportRunner::new(CommandOutcome::failure_with(1, "1 failed, 1 passed"))
            .with_reply("report.xml", xml),
    );
    let calls = Arc::clone(&runner);
    let toolchain = PythonToolchain::new(runner, dir.path())
        .with_python("python3")
        .with_report_format(ReportFormat::JUnit);
    let outcome = toolchain.run_pytest(Path::new("tests"), &[]).await.unwrap();
    assert!(!outcome.passed);
    assert_eq!(outcome.exit_code, Some(1));
    assert_eq!(outcome.format, Some(ReportFormat::JUnit));
    let report = outcome.report.unwrap();
    assert_eq!(report.total, 2);
    assert_eq!(report.failed, 1);
    assert_eq!(report.passed, 1);
    assert!(calls
        .calls()
        .iter()
        .flat_map(|spec| spec.args.iter())
        .any(|arg| arg.starts_with("--junitxml=")));
}

#[tokio::test]
async fn run_pytest_without_a_report_returns_raw_output() {
    let dir = tempfile::tempdir().unwrap();
    let runner = Arc::new(ReportRunner::new(CommandOutcome::success_with(
        "no report plugin installed",
    )));
    let toolchain = PythonToolchain::new(runner, dir.path()).with_python("python3");
    let outcome = toolchain.run_pytest(Path::new("tests"), &[]).await.unwrap();
    assert!(outcome.passed);
    assert!(outcome.report.is_none());
    assert_eq!(outcome.output, "no report plugin installed");
}

#[tokio::test]
async fn subprocess_evaluator_parses_a_tagged_reply() {
    let root = tempfile::tempdir().unwrap();
    let runner = Arc::new(ProgramRunner::new(CommandOutcome::success_with(
        "noise\n```json\n{\"answer\": 42}\n```\n",
    )));
    let calls = Arc::clone(&runner);
    let evaluator = SubprocessEvaluator::new(runner).with_config(SubprocessConfig {
        workdir_root: root.path().to_path_buf(),
        reply_format: ReplyFormat::Tagged,
        ..SubprocessConfig::default()
    });
    let result = evaluator.evaluate("print('hi')").await.unwrap();
    assert!(result.success, "{:?}", result.error);
    assert_eq!(result.response, serde_json::json!({"answer": 42}));
    assert!(result.error.is_none());
    assert!(!result.workdir.exists(), "the workdir is removed");

    let spec = &calls.calls()[0];
    assert_eq!(spec.program, "python3");
    assert_eq!(spec.args, vec!["candidate.py"]);
    assert_eq!(spec.cwd.as_deref(), Some(result.workdir.as_path()));
    let staged = std::fs::read_to_string(result.workdir.join("candidate.py"));
    assert!(staged.is_err(), "the workdir is gone after the run");
}

#[tokio::test]
async fn subprocess_evaluator_parses_a_files_reply() {
    let root = tempfile::tempdir().unwrap();
    let runner = Arc::new(
        ProgramRunner::new(CommandOutcome::success_with("done"))
            .with_reply("output.json", r#"{"score": 0.5}"#),
    );
    let evaluator = SubprocessEvaluator::new(runner).with_config(SubprocessConfig {
        workdir_root: root.path().to_path_buf(),
        ..SubprocessConfig::default()
    });
    let result = evaluator.evaluate("print('hi')").await.unwrap();
    assert!(result.success, "{:?}", result.error);
    assert_eq!(result.response, serde_json::json!({"score": 0.5}));
    assert!(!result.workdir.exists());
}

#[tokio::test]
async fn subprocess_evaluator_reports_missing_and_invalid_replies() {
    let root = tempfile::tempdir().unwrap();
    let missing = SubprocessEvaluator::new(Arc::new(ProgramRunner::new(
        CommandOutcome::success_with("done"),
    )))
    .with_config(SubprocessConfig {
        workdir_root: root.path().to_path_buf(),
        ..SubprocessConfig::default()
    });
    let result = missing.evaluate("print('hi')").await.unwrap();
    assert!(!result.success);
    assert_eq!(result.response, serde_json::Value::Null);
    let error = result.error.unwrap();
    assert!(
        error.contains("neither output.json nor response.json"),
        "{error}"
    );

    let invalid = SubprocessEvaluator::new(Arc::new(
        ProgramRunner::new(CommandOutcome::success_with("done"))
            .with_reply("output.json", "{not json"),
    ))
    .with_config(SubprocessConfig {
        workdir_root: root.path().to_path_buf(),
        ..SubprocessConfig::default()
    });
    let result = invalid.evaluate("print('hi')").await.unwrap();
    assert!(!result.success);
    assert!(result.error.as_deref().unwrap().contains("not valid JSON"));

    let unparseable = SubprocessEvaluator::new(Arc::new(ProgramRunner::new(
        CommandOutcome::success_with("no json here"),
    )))
    .with_config(SubprocessConfig {
        workdir_root: root.path().to_path_buf(),
        reply_format: ReplyFormat::Tagged,
        ..SubprocessConfig::default()
    });
    let result = unparseable.evaluate("print('hi')").await.unwrap();
    assert!(!result.success);
    assert!(result
        .error
        .as_deref()
        .unwrap()
        .contains("no JSON reply found"));
}

#[tokio::test]
async fn subprocess_evaluator_reports_a_non_zero_exit() {
    let root = tempfile::tempdir().unwrap();
    let evaluator = SubprocessEvaluator::new(Arc::new(ProgramRunner::new(
        CommandOutcome::failure_with(2, "```json\n{\"answer\": 1}\n```"),
    )))
    .with_config(SubprocessConfig {
        workdir_root: root.path().to_path_buf(),
        reply_format: ReplyFormat::Tagged,
        ..SubprocessConfig::default()
    });
    let result = evaluator.evaluate("raise SystemExit(2)").await.unwrap();
    assert!(!result.success);
    assert_eq!(result.response, serde_json::json!({"answer": 1}));
    let error = result.error.unwrap();
    assert!(error.contains("exited with code 2"), "{error}");
    assert!(error.contains("answer"));
}

#[tokio::test]
async fn subprocess_evaluator_can_keep_the_workdir() {
    let root = tempfile::tempdir().unwrap();
    let runner = Arc::new(
        ProgramRunner::new(CommandOutcome::success_with("done"))
            .with_reply("response.json", "{\"ok\": true}"),
    );
    let calls = Arc::clone(&runner);
    let evaluator = SubprocessEvaluator::new(runner)
        .with_config(SubprocessConfig {
            workdir_root: root.path().to_path_buf(),
            timeout: None,
            env: [("TOKEN".to_owned(), "abc".to_owned())].into(),
            ..SubprocessConfig::default()
        })
        .keep_workdir();
    let result = evaluator.evaluate("print('hi')").await.unwrap();
    assert!(result.success);
    assert!(result.workdir.join("candidate.py").exists());
    assert!(result.workdir.join("response.json").exists());
    assert_eq!(
        calls.calls()[0].env.get("TOKEN").map(String::as_str),
        Some("abc")
    );
    assert_eq!(calls.calls()[0].timeout, None);
    std::fs::remove_dir_all(&result.workdir).unwrap();
}

#[tokio::test]
async fn parse_files_reply_prefers_output_json() {
    let dir = tempfile::tempdir().unwrap();
    assert_eq!(
        parse_files_reply(dir.path()).await.unwrap_err().kind(),
        ErrorKind::Validation
    );
    std::fs::write(dir.path().join("response.json"), r#"{"r": 1}"#).unwrap();
    assert_eq!(
        parse_files_reply(dir.path()).await.unwrap(),
        serde_json::json!({"r": 1})
    );
    std::fs::write(dir.path().join("output.json"), r#"{"o": 2}"#).unwrap();
    assert_eq!(
        parse_files_reply(dir.path()).await.unwrap(),
        serde_json::json!({"o": 2})
    );
}

#[test]
fn parse_tagged_reply_prefers_the_last_fenced_block() {
    let stdout = "log\n```json\n{\"a\": 1}\n```\nmore\n<result>{\"b\": 2}</result>\n";
    assert_eq!(
        parse_tagged_reply(stdout),
        Some(serde_json::json!({"a": 1}))
    );
    assert_eq!(
        parse_tagged_reply("<RESULT>{\"c\": 3}</RESULT>"),
        Some(serde_json::json!({"c": 3}))
    );
    assert_eq!(parse_tagged_reply("  "), None);
}
