//! Process-level suite of the `watch` command.
//!
//! `watch` is the terminal end of the harness workspace watcher
//! (`openjiuwen_harness::watcher`), so its contract is pinned here through real
//! child processes:
//!
//! 1. **The plan** — `watch --list` renders the resolved targets of a workspace
//!    (instruction files, MCP configuration, skill root) and the paths it had to
//!    skip, in both the text and the `--json` form.
//! 2. **The empty plan** — a workspace and home that do not exist produce no
//!    target, and starting the watcher is a typed `config` error (exit `1`,
//!    empty stdout) instead of a silent, permanently idle process.
//! 3. **The live stream** — a change to a watched instruction file after the
//!    readiness line produces exactly one JSON `event` on stdout, and
//!    `--max-events 1` then shuts the watcher down and exits `0`.
//!
//! Every case gets its own [`support::Sandbox`] (temp cwd, temp
//! `OPENJIUWEN_HOME`, temp `-w` workspace) and builds no model.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod support;

use std::time::Duration;

use tokio::io::{AsyncRead, AsyncReadExt};

use support::{assert_exit, assert_success, non_empty_lines, parse_json, stderr, Sandbox};

/// How long to wait for the readiness line before failing the case.
const READY_BUDGET: Duration = Duration::from_secs(15);
/// How long to wait for the child to exit after the change was written.
const EXIT_BUDGET: Duration = Duration::from_secs(45);

/// Runs one `watch` invocation scoped to the sandbox's workspace.
fn run_watch(sandbox: &Sandbox, tail: &[&str]) -> std::process::Output {
    let mut args: Vec<String> = sandbox.workspace_flag().into();
    args.push("watch".to_owned());
    args.extend(tail.iter().map(|arg| (*arg).to_owned()));
    sandbox.run_blocking(&args)
}

/// Creates the files `--list` has to discover: workspace and home instruction
/// files, the workspace MCP configuration and the workspace skill root.
fn seed_watchable_tree(sandbox: &Sandbox) {
    sandbox.write("workspace/OPENJIUWEN.md", "# workspace memory\n");
    sandbox.write("home/OPENJIUWEN.md", "# home memory\n");
    sandbox.write("workspace/.openjiuwen/mcp.json", "{\"mcpServers\": {}}\n");
    std::fs::create_dir_all(sandbox.workspace().join("skills"))
        .expect("creating the workspace skill root");
}

/// Reads `reader` until its lossy text contains `needle` or it closes.
///
/// Bounded by [`READY_BUDGET`], so a watcher that never starts fails with the
/// bytes seen so far instead of hanging the suite.
async fn read_until(reader: &mut (impl AsyncRead + Unpin), needle: &str) -> String {
    let mut seen: Vec<u8> = Vec::new();
    let mut chunk = [0u8; 512];
    let outcome = tokio::time::timeout(READY_BUDGET, async {
        loop {
            match reader.read(&mut chunk).await {
                Ok(0) => break,
                Ok(read) => {
                    seen.extend_from_slice(&chunk[..read]);
                    if String::from_utf8_lossy(&seen).contains(needle) {
                        break;
                    }
                }
                Err(err) => panic!("reading the child stream: {err}"),
            }
        }
    })
    .await;
    let text = String::from_utf8_lossy(&seen).into_owned();
    assert!(
        outcome.is_ok(),
        "timed out after {READY_BUDGET:?} waiting for {needle:?}; saw {text:?}"
    );
    assert!(
        text.contains(needle),
        "expected {needle:?} before the stream closed; saw {text:?}"
    );
    text
}

#[test]
fn list_prints_the_instruction_mcp_and_skill_targets() {
    let sandbox = Sandbox::new();
    seed_watchable_tree(&sandbox);

    let text = assert_success(&run_watch(&sandbox, &["--list"]));
    let workspace = sandbox.workspace().display().to_string();
    let home = sandbox.home().display().to_string();

    let instruction = format!("instruction_file\t{workspace}/OPENJIUWEN.md");
    let home_instruction = format!("instruction_file\t{home}/OPENJIUWEN.md");
    let mcp = format!("mcp_config\t{workspace}/.openjiuwen/mcp.json");
    let skills = format!("skills\t{workspace}/skills (recursive)");

    for expected in [&instruction, &home_instruction, &mcp, &skills] {
        assert!(
            text.contains(expected.as_str()),
            "missing target line {expected:?} in {text:?}"
        );
    }
    // The plan is deterministic: a skipped root is recorded, never silently
    // dropped.
    assert!(
        text.lines().any(|line| line.starts_with("skipped\t")),
        "the missing roots are recorded as skipped: {text:?}"
    );
}

#[test]
fn list_json_renders_targets_and_skipped_paths() {
    let sandbox = Sandbox::new();
    seed_watchable_tree(&sandbox);

    let value = parse_json(&assert_success(&run_watch(&sandbox, &["--list", "--json"])));
    let targets = value["targets"]
        .as_array()
        .unwrap_or_else(|| panic!("targets is an array: {value}"));
    assert!(!targets.is_empty(), "the seeded tree has targets: {value}");

    let workspace = sandbox.workspace().display().to_string();
    for target in targets {
        assert!(target["kind"].is_string(), "kind is a string: {target}");
        assert!(target["path"].is_string(), "path is a string: {target}");
        assert!(
            target.get("label").is_some(),
            "label is always present: {target}"
        );
        assert!(
            target["recursive"].is_boolean(),
            "recursive is a boolean: {target}"
        );
    }
    assert!(
        targets.iter().any(|target| {
            target["kind"] == "instruction_file"
                && target["path"] == format!("{workspace}/OPENJIUWEN.md")
        }),
        "the workspace instruction file is a target: {value}"
    );
    assert!(
        targets
            .iter()
            .any(|target| { target["kind"] == "skills" && target["recursive"] == true }),
        "the skill root is a recursive target: {value}"
    );

    let skipped = value["skipped"]
        .as_array()
        .unwrap_or_else(|| panic!("skipped is an array: {value}"));
    assert!(
        !skipped.is_empty(),
        "the missing agent/plugin roots are recorded: {value}"
    );
    for path in skipped {
        let path = path
            .as_str()
            .unwrap_or_else(|| panic!("a skipped path is a string: {value}"));
        assert!(
            std::path::Path::new(path).is_absolute(),
            "skipped paths are absolute: {path:?}"
        );
    }
}

#[tokio::test]
async fn an_empty_plan_is_a_typed_config_error() {
    let sandbox = Sandbox::new();
    // Both roots are left missing, so every candidate target's watch path is
    // absent and `existing_only` keeps nothing.
    let missing_home = sandbox.path("missing-home");
    let missing_workspace = sandbox.path("missing-workspace");
    let args = [
        "-w".to_owned(),
        missing_workspace.display().to_string(),
        "watch".to_owned(),
    ];
    let output = sandbox
        .run_with_env(&args, &[("OPENJIUWEN_HOME", &missing_home)])
        .await;

    let stdout = assert_exit(&output, 1);
    assert!(
        stdout.is_empty(),
        "a rejected plan writes no stdout: {stdout:?}"
    );
    let err = stderr(&output);
    assert!(
        err.contains("error: config:"),
        "an empty plan is a config error: {err:?}"
    );
    assert!(
        err.contains("at least one watch target"),
        "the diagnostic names the missing targets: {err:?}"
    );
}

#[tokio::test]
async fn a_change_is_reported_as_one_json_event_and_max_events_exits() {
    let sandbox = Sandbox::new();
    // Makes the plan non-empty before the watcher starts; the watched root is
    // the workspace itself (the instruction-file targets).
    sandbox.write("workspace/OPENJIUWEN.md", "# workspace memory\n");

    let mut args: Vec<String> = sandbox.workspace_flag().into();
    for arg in [
        "watch",
        "--json",
        "--max-events",
        "1",
        "--debounce-ms",
        "100",
    ] {
        args.push(arg.to_owned());
    }
    let mut child = sandbox.spawn(&args);
    let mut stdout = child.stdout.take().expect("piped stdout");
    let mut stderr = child.stderr.take().expect("piped stderr");

    let ready = read_until(&mut stderr, "target(s)").await;
    assert!(
        ready.contains("watching "),
        "the readiness line reports the target count: {ready:?}"
    );

    // A *new* instruction document after the watcher registered its roots: the
    // file target may not exist at start-up, which is exactly what makes this
    // observable.
    std::fs::write(sandbox.workspace().join("IDENTITY.md"), "# identity\n")
        .expect("writing the watched instruction file");

    let mut report = String::new();
    let read = tokio::time::timeout(EXIT_BUDGET, stdout.read_to_string(&mut report)).await;
    assert!(
        read.is_ok(),
        "the watcher did not exit within {EXIT_BUDGET:?}: {report:?}"
    );
    let mut err_tail = String::new();
    let _ = tokio::time::timeout(EXIT_BUDGET, stderr.read_to_string(&mut err_tail)).await;
    let status = tokio::time::timeout(EXIT_BUDGET, child.wait())
        .await
        .expect("the child exits within the budget")
        .expect("waiting for the child");
    assert!(
        status.success(),
        "max-events exits 0; report {report:?}, stderr {err_tail:?}"
    );

    let lines = non_empty_lines(&report);
    assert_eq!(
        lines.len(),
        1,
        "exactly one update is rendered before the watcher shuts down: {report:?}"
    );
    let update = parse_json(lines[0]);
    assert_eq!(update["type"], "event", "the update is an event: {update}");
    assert_eq!(
        update["kind"], "instruction_file",
        "the created identity document is an instruction file: {update}"
    );
    assert!(
        update["path"]
            .as_str()
            .is_some_and(|path| path.ends_with("IDENTITY.md")),
        "the event names the changed path: {update}"
    );
    assert!(
        ["created", "modified"].contains(&update["change"].as_str().unwrap_or("")),
        "a written file is created or modified: {update}"
    );
}
