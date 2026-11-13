//! Offline smoke test: run every runnable example once.
//!
//! `cargo test -p openjiuwen-examples` compiles the example targets declared in
//! `Cargo.toml`, so their executables land in `target/<profile>/examples`, the
//! sibling of the `target/<profile>/deps` directory holding this test binary.
//! The test runs each executable sequentially with a scrubbed environment and a
//! temporary working directory, then asserts a zero exit status and non-empty
//! stdout. The example list is hard-coded: renaming or dropping an example fails
//! here instead of silently shrinking coverage.
//!
//! `11_a2a_server_client` is the only example behind `required-features =
//! ["a2a"]`, so a default build has no executable for it. That example is
//! recorded as skipped (and only then), while every other example must be
//! present, run, and pass.

use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{bail, ensure, Context, Result};

/// Examples that must build and pass with default features.
const REQUIRED_EXAMPLES: &[&str] = &[
    "01_hello_agent",
    "02_react_with_tools",
    "03_sessions_and_resume",
    "04_workflow_pipeline",
    "05_context_and_memory",
    "06_retrieval",
    "07_mcp_tool",
    "08_deep_agent",
    "09_harness_provider",
    "10_agent_team",
    "12_self_improvement",
    "13_context_processors",
    "14_workflow_components",
    "15_retrieval_quality",
    "16_memory_lite",
    "17_team_persistence",
    "18_team_attach",
    "19_symphony_skills",
    "20_prompt_tuning",
];

/// The one example that only exists when the `a2a` feature is enabled.
const FEATURE_GATED_EXAMPLE: &str = "11_a2a_server_client";

/// Variables that would make an example reach a real provider, MCP server, or
/// verbosity setting. They are removed so the child always runs offline.
const SCRUBBED_VARIABLES: &[&str] = &[
    "OPENJIUWEN_LIVE",
    "MODEL_NAME",
    "MODEL_PROVIDER",
    "API_KEY",
    "API_BASE",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENJIUWEN_API_KEY",
    "OPENJIUWEN_MCP_SERVER_CMD",
    "OPENJIUWEN_LOG_LEVEL",
    "RUST_LOG",
];

/// Variable that keeps example diagnostics on stderr but quiet.
const LOG_LEVEL_VARIABLE: &str = "OPENJIUWEN_LOG_LEVEL";
/// Suffix cargo gives an executable on this platform.
const EXE_SUFFIX: &str = if cfg!(windows) { ".exe" } else { "" };
/// Upper bound for one example; every example runs in well under a second.
const TIMEOUT: Duration = Duration::from_secs(90);
/// Poll interval while waiting for an example to exit.
const POLL_INTERVAL: Duration = Duration::from_millis(50);
/// How much trailing stderr to quote when an example fails.
const STDERR_TAIL_LINES: usize = 40;

/// The `target/<profile>/examples` directory beside this test binary.
fn executables_root() -> Result<PathBuf> {
    let exe = std::env::current_exe().context("cannot locate the running test executable")?;
    let profile_dir = exe
        .parent()
        .and_then(Path::parent)
        .context("the test executable is not inside a cargo profile directory")?;
    Ok(profile_dir.join("examples"))
}

/// The executable cargo builds for example `name`.
fn executable_path(root: &Path, name: &str) -> PathBuf {
    root.join(format!("{name}{EXE_SUFFIX}"))
}

/// Runs one example and returns the number of bytes it wrote to stdout.
fn run_executable(root: &Path, name: &str, workdir: &Path) -> Result<u64> {
    let exe = executable_path(root, name);
    let stdout_path = workdir.join(format!("{name}.stdout"));
    let stderr_path = workdir.join(format!("{name}.stderr"));
    let stdout = File::create(&stdout_path).with_context(|| {
        format!(
            "cannot capture the stdout of `{name}` at {}",
            stdout_path.display()
        )
    })?;
    let stderr = File::create(&stderr_path).with_context(|| {
        format!(
            "cannot capture the stderr of `{name}` at {}",
            stderr_path.display()
        )
    })?;

    let mut command = Command::new(&exe);
    command
        .current_dir(workdir)
        .stdin(Stdio::null())
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(stderr));
    for variable in SCRUBBED_VARIABLES {
        command.env_remove(variable);
    }
    command.env(LOG_LEVEL_VARIABLE, "error");

    let mut child = command
        .spawn()
        .with_context(|| format!("cannot start example `{name}` from {}", exe.display()))?;
    let status = wait_for_exit(&mut child, name, &stderr_path)?;
    let tail = stderr_tail(&stderr_path);
    if !status.success() {
        bail!(
            "example `{name}` failed with {}; stderr tail:\n{tail}",
            describe_status(status)
        );
    }

    let stdout_len = fs::metadata(&stdout_path)
        .with_context(|| format!("cannot inspect the captured stdout of `{name}`"))?
        .len();
    ensure!(
        stdout_len > 0,
        "example `{name}` exited 0 but wrote nothing to stdout; stderr tail:\n{tail}"
    );
    Ok(stdout_len)
}

/// Waits for `child` to exit, killing and reporting it after [`TIMEOUT`].
fn wait_for_exit(child: &mut Child, name: &str, stderr_path: &Path) -> Result<ExitStatus> {
    let started = Instant::now();
    loop {
        if let Some(status) = child
            .try_wait()
            .with_context(|| format!("cannot poll example `{name}`"))?
        {
            return Ok(status);
        }
        if started.elapsed() >= TIMEOUT {
            let _ = child.kill();
            let _ = child.wait();
            bail!(
                "example `{name}` did not exit within {} s and was killed; stderr tail:\n{}",
                TIMEOUT.as_secs(),
                stderr_tail(stderr_path)
            );
        }
        thread::sleep(POLL_INTERVAL);
    }
}

/// Describes an exit status without relying on its platform-specific display.
fn describe_status(status: ExitStatus) -> String {
    status.code().map_or_else(
        || "termination by a signal".to_owned(),
        |code| format!("exit code {code}"),
    )
}

/// The last [`STDERR_TAIL_LINES`] lines of a captured stderr file.
fn stderr_tail(path: &Path) -> String {
    match fs::read_to_string(path) {
        Ok(text) if !text.trim().is_empty() => {
            let skip = text.lines().count().saturating_sub(STDERR_TAIL_LINES);
            let tail = text.lines().skip(skip).collect::<Vec<_>>().join("\n");
            if tail.is_empty() {
                "<stderr held only blank lines>".to_owned()
            } else {
                tail
            }
        }
        Ok(_) => "<stderr was empty>".to_owned(),
        Err(error) => format!("<stderr unreadable: {error}>"),
    }
}

#[test]
fn every_example_runs_offline() -> Result<()> {
    let root = executables_root()?;
    if !root.is_dir() {
        bail!(
            "no example executables in {}: `cargo test -p openjiuwen-examples --offline` builds \
             them, or run `cargo build -p openjiuwen-examples --examples` first",
            root.display()
        );
    }
    let workdir =
        tempfile::tempdir().context("cannot create a temporary working directory for the runs")?;
    let mut executed = 0_usize;
    let mut skipped = 0_usize;

    for &name in REQUIRED_EXAMPLES {
        let path = executable_path(&root, name);
        ensure!(
            path.is_file(),
            "example `{name}` has no executable at {}: `tests/examples_offline.rs` and the \
             `[[example]]` entries in `Cargo.toml` disagree",
            path.display()
        );
        let stdout_len = run_executable(&root, name, workdir.path())?;
        println!("  ok   {name} ({stdout_len} bytes on stdout)");
        executed += 1;
    }

    let gated = executable_path(&root, FEATURE_GATED_EXAMPLE);
    if gated.is_file() {
        let stdout_len = run_executable(&root, FEATURE_GATED_EXAMPLE, workdir.path())?;
        println!("  ok   {FEATURE_GATED_EXAMPLE} ({stdout_len} bytes on stdout)");
        executed += 1;
    } else {
        println!(
            "  skip {FEATURE_GATED_EXAMPLE}: not built without the `a2a` feature \
             (`cargo test -p openjiuwen-examples --features a2a`)"
        );
        skipped += 1;
    }

    println!("example smoke: {executed} executed, {skipped} skipped");
    ensure!(
        executed >= REQUIRED_EXAMPLES.len(),
        "only {executed} examples ran; expected at least {} default-feature examples",
        REQUIRED_EXAMPLES.len()
    );
    Ok(())
}
