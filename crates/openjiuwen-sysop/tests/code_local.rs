//! Ports `tests/unit_tests/core/sys_operation/local/test_code_operation.py`.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;
use std::time::{Duration, Instant};

use openjiuwen_sysop::code::local_code;
use openjiuwen_sysop::code::types::{CodeExecParams, Language};
use openjiuwen_sysop::config::LocalWorkConfig;
use openjiuwen_sysop::operation::CodeOperation;
use openjiuwen_sysop::status::SysOpStatus;

/// Python's success message for a completed run, whatever its exit status.
const SUCCESS_MESSAGE: &str = "Code executed successfully";

/// `true` when a Python 3 interpreter exists; snippet tests skip otherwise.
fn require_python() -> bool {
    if which::which("python3").is_ok() {
        true
    } else {
        eprintln!("skipping code test: `python3` is not on PATH");
        false
    }
}

/// Builds a local code backend whose workspace is `dir`.
fn backend(dir: &tempfile::TempDir) -> Arc<dyn CodeOperation> {
    let workspace = dir.path().to_string_lossy().into_owned();
    local_code(LocalWorkConfig::unrestricted(workspace)).unwrap()
}

#[tokio::test]
async fn python_snippet_runs_and_captures_stdout() {
    if !require_python() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let result = backend(&dir)
        .execute_code(CodeExecParams::new("print(\"hi\")"))
        .await;
    assert_eq!(
        result.code,
        SysOpStatus::Success.code(),
        "{}",
        result.message
    );
    assert_eq!(result.message, SUCCESS_MESSAGE);
    let output = result.data.expect("a successful run carries output");
    assert_eq!(output.exit_code, 0);
    assert_eq!(output.stdout.trim(), "hi");
    assert_eq!(output.language, "python");
    assert!(!output.timed_out);
}

#[tokio::test]
async fn non_zero_exit_is_not_an_error() {
    if !require_python() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let result = backend(&dir)
        .execute_code(CodeExecParams::new("import sys; sys.exit(3)"))
        .await;
    // Python parity: the run succeeded and simply reported the non-zero status.
    assert_eq!(
        result.code,
        SysOpStatus::Success.code(),
        "{}",
        result.message
    );
    assert_eq!(result.message, SUCCESS_MESSAGE);
    let output = result.data.expect("a completed run carries output");
    assert_eq!(output.exit_code, 3);
    assert!(!output.timed_out);
}

#[tokio::test]
async fn stderr_is_captured() {
    if !require_python() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let result = backend(&dir)
        .execute_code(CodeExecParams::new(
            "import sys\nprint(\"boom\", file=sys.stderr)\nsys.exit(1)",
        ))
        .await;
    assert_eq!(
        result.code,
        SysOpStatus::Success.code(),
        "{}",
        result.message
    );
    assert_eq!(result.message, SUCCESS_MESSAGE);
    let output = result.data.expect("output is captured on a non-zero exit");
    assert_eq!(output.exit_code, 1);
    assert!(output.stderr.contains("boom"), "stderr: {}", output.stderr);
}

#[tokio::test]
async fn empty_code_is_rejected() {
    // Validation runs before the runtime lookup, so no interpreter is required here.
    let dir = tempfile::tempdir().unwrap();
    let backend = backend(&dir);
    for code in ["", "   ", " \n\t "] {
        let result = backend.execute_code(CodeExecParams::new(code)).await;
        assert_eq!(
            result.code,
            SysOpStatus::CodeExecutionError.code(),
            "code {code:?}"
        );
        assert!(
            result.message.contains("code can not be empty"),
            "code {code:?}: {}",
            result.message
        );
        // Python's non-stream path carries no payload for a validation failure.
        assert!(result.data.is_none(), "code {code:?}");
    }
}

#[tokio::test]
async fn force_file_strategy_runs_and_returns_output() {
    if !require_python() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let result = backend(&dir)
        .execute_code(CodeExecParams::new("print(\"file-mode\")").with_force_file(true))
        .await;
    assert_eq!(
        result.code,
        SysOpStatus::Success.code(),
        "{}",
        result.message
    );
    assert_eq!(result.message, SUCCESS_MESSAGE);
    let output = result.data.expect("a successful run carries output");
    assert_eq!(output.exit_code, 0);
    assert_eq!(output.stdout.trim(), "file-mode");
}

#[tokio::test]
async fn timeout_returns_199005_with_partial_output() {
    if !require_python() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let code = "import time\nprint(\"start\", flush=True)\ntime.sleep(5)";
    let started = Instant::now();
    let result = backend(&dir)
        .execute_code(CodeExecParams::new(code).with_timeout_secs(1))
        .await;
    let elapsed = started.elapsed();
    assert_eq!(
        result.code,
        SysOpStatus::CodeExecutionError.code(),
        "{}",
        result.message
    );
    assert!(
        result.message.to_lowercase().contains("timeout"),
        "{}",
        result.message
    );
    let output = result
        .data
        .expect("a timeout keeps the partially captured output");
    assert!(output.timed_out, "timed_out must be set");
    assert_eq!(output.exit_code, -1);
    assert!(output.stdout.contains("start"), "stdout: {}", output.stdout);
    assert!(elapsed < Duration::from_secs(4), "elapsed {elapsed:?}");
}

#[tokio::test]
async fn language_metadata_is_reported() {
    let python = CodeExecParams::new("print(1)");
    assert_eq!(python.language, Language::Python);
    assert_eq!(python.language.as_str(), "python");

    let javascript = CodeExecParams::javascript("console.log(1)");
    assert_eq!(javascript.language, Language::Javascript);
    assert_eq!(javascript.language.as_str(), "javascript");
}

#[tokio::test]
async fn javascript_snippet_runs_when_node_is_available() {
    if which::which("node").is_err() {
        eprintln!("skipping JavaScript code test: `node` is not on PATH");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let result = backend(&dir)
        .execute_code(CodeExecParams::javascript("console.log(1)"))
        .await;
    assert_eq!(
        result.code,
        SysOpStatus::Success.code(),
        "{}",
        result.message
    );
    assert_eq!(result.message, SUCCESS_MESSAGE);
    let output = result.data.expect("a successful run carries output");
    assert_eq!(output.exit_code, 0);
    assert!(output.stdout.contains('1'), "stdout: {}", output.stdout);
    assert_eq!(output.language, "javascript");
}

#[tokio::test]
async fn code_with_multiline_output_preserves_newlines() {
    if !require_python() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let result = backend(&dir)
        .execute_code(CodeExecParams::new("print(\"a\")\nprint(\"b\")"))
        .await;
    assert_eq!(
        result.code,
        SysOpStatus::Success.code(),
        "{}",
        result.message
    );
    let output = result.data.expect("a successful run carries output");
    assert_eq!(output.exit_code, 0);
    assert_eq!(output.stdout, "a\nb\n");
}

#[tokio::test]
async fn successful_run_echoes_code_and_reports_no_truncation() {
    if !require_python() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let code = "print(\"echo\")";
    let result = backend(&dir).execute_code(CodeExecParams::new(code)).await;
    assert_eq!(
        result.code,
        SysOpStatus::Success.code(),
        "{}",
        result.message
    );
    let output = result.data.expect("a successful run carries output");
    assert_eq!(output.code_content, code);
    assert_eq!(output.language, "python");
    assert!(!output.truncated, "a short run must not be truncated");
    assert!(!output.timed_out);
    assert_eq!(output.stdout.trim(), "echo");
}
