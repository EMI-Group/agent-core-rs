//! Process-level suite of the `auto-harness` command group.
//!
//! `auto-harness` is the one command group the binary dispatches itself (the
//! harness grammar has no words for it), so its contract is pinned here through
//! real child processes:
//!
//! 1. **Task resolution** — `--task`, `--task-file` (JSON array *and* single
//!    object, unknown keys tolerated), `--task` winning over `--task-file`, the
//!    empty list, and the validation error naming the file when an entry has no
//!    `topic`.
//! 2. **The dry run** — stdout carries the task list as JSON, stderr carries the
//!    `[dry-run] task list ->` note (stdout stays machine-readable), and
//!    `<workspace>/auto_harness/runs/tasks.json` holds the same document.
//! 3. **The deferred surfaces** — a non-dry `run` and `verify-ext` exit `1` with
//!    a typed `unsupported: …` diagnostic, `--stage` is named in the message, and
//!    `--pipeline bogus` is a clap usage error (exit `2`, empty stdout).
//! 4. **The experience store** — recency ordering, `--limit`, the `--type` filter
//!    applied *after* the limit, keyword search over `topic`/`summary`/`details`
//!    (never `outcome`), tokens shorter than two characters matching nothing,
//!    malformed lines skipped, `--json` arrays, and the documented empty-store
//!    messages.
//! 5. **`history`, `gap-analyze`** — the `list` line format and JSON form, and the
//!    placeholder gap analysis with `gaps: []`.
//!
//! Everything is hermetic: each case gets its own [`support::Sandbox`] (temp cwd,
//! temp `OPENJIUWEN_HOME`, temp `-w` workspace), no model is built, and every
//! child is bounded by [`support::CHILD_WAIT`].
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod support;

use std::process::Output;

use serde_json::{json, Value};
use support::{
    assert_exit, assert_success, describe, non_empty_lines, parse_json, stderr, stdout, Sandbox,
};

/// The seeded experience store: three types, ascending timestamps, and one
/// malformed line that every read must skip.
///
/// `OUTCOME-TOKEN` only appears in the `outcome` field, which search never reads.
const SEEDED_STORE: &str = concat!(
    r#"{"type":"insight","topic":"alpha parser","summary":"faster parser","details":"cache","outcome":"OUTCOME-TOKEN","timestamp":1000.0}"#,
    "\n",
    r#"{"type":"failure","topic":"beta builder","summary":"build broke","timestamp":2000.0}"#,
    "\n",
    r#"{"type":"optimization","topic":"gamma runner","summary":"pr merged","timestamp":3000.0}"#,
    "\n",
    "{not json}\n",
    r#"{"type":"insight","topic":"delta","summary":"newest","timestamp":4000.0}"#,
    "\n",
);

/// Runs one `auto-harness` invocation scoped to the sandbox's workspace.
fn run_auto_harness(sandbox: &Sandbox, tail: &[&str]) -> Output {
    let mut args: Vec<String> = sandbox.workspace_flag().into();
    args.extend(tail.iter().map(|arg| (*arg).to_owned()));
    sandbox.run_blocking(&args)
}

/// Writes [`SEEDED_STORE`] into the sandbox's experience store.
fn seed_store(sandbox: &Sandbox) {
    sandbox.write_auto_harness("experience/experiences.jsonl", SEEDED_STORE);
}

/// Returns the topic of every task in a dry-run document.
fn topics(value: &Value) -> Vec<String> {
    value
        .as_array()
        .unwrap_or_else(|| panic!("expected an array, got {value}"))
        .iter()
        .map(|task| {
            task["topic"]
                .as_str()
                .unwrap_or_else(|| panic!("every task has a topic: {task}"))
                .to_owned()
        })
        .collect()
}

// --- task resolution and the dry run --------------------------------------

#[test]
fn dry_run_prints_and_writes_the_single_task() {
    let sandbox = Sandbox::new();
    let output = run_auto_harness(
        &sandbox,
        &[
            "auto-harness",
            "run",
            "--task",
            "fix the parser",
            "--dry-run",
        ],
    );
    let text = assert_success(&output);
    let value = parse_json(&text);
    let tasks = value.as_array().expect("the dry run prints an array");
    assert_eq!(tasks.len(), 1, "{value}");
    let task = &tasks[0];
    for key in ["topic", "description", "files"] {
        assert!(task.get(key).is_some(), "missing `{key}`: {task}");
    }
    assert_eq!(task["topic"], "fix the parser");
    assert_eq!(task["description"], "");
    assert_eq!(task["files"], json!([]));

    let path = sandbox.auto_harness_path("runs/tasks.json");
    let written = std::fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("the dry run writes {}: {err}", path.display()));
    assert_eq!(
        parse_json(&written),
        value,
        "the runs file holds the echoed task list"
    );
    assert!(
        stderr(&output).contains("[dry-run] task list ->"),
        "the note belongs on stderr: {}",
        describe(&output)
    );
}

#[test]
fn task_file_accepts_an_array_and_a_single_object() {
    let sandbox = Sandbox::new();
    sandbox.write(
        "tasks.json",
        concat!(
            r#"[{"topic":"first","description":"first task","files":["src/a.rs"],"unknown":1},"#,
            r#"{"topic":"second"}]"#,
        ),
    );
    let array_path = sandbox.path("tasks.json").display().to_string();
    let output = run_auto_harness(
        &sandbox,
        &[
            "auto-harness",
            "run",
            "--task-file",
            &array_path,
            "--dry-run",
        ],
    );
    let value = parse_json(&assert_success(&output));
    let tasks = value.as_array().expect("an array of tasks");
    assert_eq!(tasks.len(), 2, "{value}");
    assert_eq!(tasks[0]["topic"], "first");
    assert_eq!(tasks[0]["description"], "first task");
    assert_eq!(tasks[0]["files"], json!(["src/a.rs"]));
    assert_eq!(tasks[1]["topic"], "second");
    assert_eq!(tasks[1]["description"], "");
    assert_eq!(tasks[1]["files"], json!([]));

    sandbox.write("single.json", r#"{"topic":"only"}"#);
    let single_path = sandbox.path("single.json").display().to_string();
    let output = run_auto_harness(
        &sandbox,
        &[
            "auto-harness",
            "run",
            "--task-file",
            &single_path,
            "--dry-run",
        ],
    );
    let value = parse_json(&assert_success(&output));
    assert_eq!(topics(&value), vec!["only".to_owned()]);
}

#[test]
fn task_file_entry_without_topic_fails_and_names_the_file() {
    let sandbox = Sandbox::new();
    let path = sandbox.write("bad.json", r#"[{"description":"no topic"}]"#);
    let shown = path.display().to_string();
    let output = run_auto_harness(
        &sandbox,
        &["auto-harness", "run", "--task-file", &shown, "--dry-run"],
    );
    assert_exit(&output, 1);
    let diagnostic = stderr(&output);
    assert!(
        diagnostic.contains("error:"),
        "a runtime failure is reported as `error:`: {}",
        describe(&output)
    );
    assert!(
        diagnostic.contains(&shown),
        "the message names the task file: {diagnostic}"
    );
    assert!(
        stdout(&output).is_empty(),
        "a rejected task file prints no list: {}",
        describe(&output)
    );
    assert!(
        !sandbox.auto_harness_path("runs/tasks.json").exists(),
        "a rejected task file writes no runs document"
    );
}

#[test]
fn task_flag_wins_over_the_task_file() {
    let sandbox = Sandbox::new();
    let path = sandbox.write("tasks.json", r#"[{"topic":"from file"}]"#);
    let shown = path.display().to_string();
    let output = run_auto_harness(
        &sandbox,
        &[
            "auto-harness",
            "run",
            "--task",
            "from flag",
            "--task-file",
            &shown,
            "--dry-run",
        ],
    );
    let value = parse_json(&assert_success(&output));
    assert_eq!(topics(&value), vec!["from flag".to_owned()]);
}

#[test]
fn dry_run_without_tasks_prints_an_empty_array() {
    let sandbox = Sandbox::new();
    let text = assert_success(&run_auto_harness(
        &sandbox,
        &["auto-harness", "run", "--dry-run"],
    ));
    assert_eq!(text.trim(), "[]");
}

// --- deferred surfaces ----------------------------------------------------

#[test]
fn non_dry_run_reports_the_deferred_evaluation_runner() {
    let sandbox = Sandbox::new();
    let output = run_auto_harness(&sandbox, &["auto-harness", "run", "--task", "x"]);
    // Exactly `1`: the failure is a runtime error, not a usage error.
    assert_exit(&output, 1);
    let diagnostic = stderr(&output);
    assert!(
        diagnostic.contains("unsupported"),
        "the deferral is typed: {diagnostic}"
    );
    assert!(
        diagnostic.contains("harness evaluation runner"),
        "the message names the missing runner: {diagnostic}"
    );
}

#[test]
fn stage_flag_is_named_in_the_deferred_error() {
    let sandbox = Sandbox::new();
    let output = run_auto_harness(
        &sandbox,
        &["auto-harness", "run", "--task", "x", "--stage", "verify"],
    );
    assert_exit(&output, 1);
    assert!(
        stderr(&output).contains("stage `verify`"),
        "the message names the requested stage: {}",
        describe(&output)
    );
}

#[test]
fn unknown_pipeline_is_a_usage_error() {
    let sandbox = Sandbox::new();
    let output = sandbox.run_blocking(&["auto-harness", "run", "--pipeline", "bogus"]);
    assert_exit(&output, 2);
    assert!(
        stdout(&output).is_empty(),
        "a usage error prints nothing on stdout: {}",
        describe(&output)
    );
    let diagnostic = stderr(&output);
    assert!(
        diagnostic.contains("--pipeline"),
        "the diagnostic names the flag: {diagnostic}"
    );
    assert!(
        diagnostic.contains("bogus"),
        "the diagnostic quotes the rejected value: {diagnostic}"
    );
}

#[test]
fn verify_ext_reports_the_deferred_runtime_extension_loader() {
    let sandbox = Sandbox::new();
    let extension = sandbox.path("extension");
    std::fs::create_dir_all(&extension).expect("an extension directory");
    let output = run_auto_harness(
        &sandbox,
        &[
            "auto-harness",
            "verify-ext",
            "--ext-path",
            &extension.display().to_string(),
        ],
    );
    assert_exit(&output, 1);
    let diagnostic = stderr(&output);
    assert!(
        diagnostic.contains("unsupported"),
        "the deferral is typed: {diagnostic}"
    );
    assert!(
        diagnostic.contains("runtime extension loader"),
        "the message names the missing loader: {diagnostic}"
    );
}

#[test]
fn verify_ext_does_not_validate_the_path() {
    let sandbox = Sandbox::new();
    // A path that does not exist still fails in the deferred loader, not in a
    // path check: the loader is the first thing the command touches.
    let output = run_auto_harness(
        &sandbox,
        &[
            "auto-harness",
            "verify-ext",
            "--ext-path",
            "/definitely/not/a/path",
        ],
    );
    assert_exit(&output, 1);
    assert!(
        stderr(&output).contains("runtime extension loader"),
        "{}",
        describe(&output)
    );
}

// --- experience store -----------------------------------------------------

#[test]
fn experience_list_orders_newest_first_and_skips_the_malformed_line() {
    let sandbox = Sandbox::new();
    seed_store(&sandbox);
    let text = assert_success(&run_auto_harness(
        &sandbox,
        &["auto-harness", "experience", "list"],
    ));
    let lines = non_empty_lines(&text);
    let expected = [
        "[insight] delta: newest",
        "[optimization] gamma runner: pr merged",
        "[failure] beta builder: build broke",
        "[insight] alpha parser: faster parser",
    ];
    assert_eq!(lines, expected, "stdout was {text:?}");
}

#[test]
fn experience_list_honours_the_limit_and_filters_after_it() {
    let sandbox = Sandbox::new();
    seed_store(&sandbox);
    let limited = assert_success(&run_auto_harness(
        &sandbox,
        &["auto-harness", "experience", "list", "--limit", "2"],
    ));
    assert_eq!(
        non_empty_lines(&limited),
        [
            "[insight] delta: newest",
            "[optimization] gamma runner: pr merged"
        ]
    );

    // Python truncates to the limit first and filters by type second, so a window
    // of two records holding no failure comes back empty.
    let empty = assert_success(&run_auto_harness(
        &sandbox,
        &[
            "auto-harness",
            "experience",
            "list",
            "--limit",
            "2",
            "--type",
            "failure",
        ],
    ));
    assert_eq!(empty.trim(), "no experience records");

    let widened = assert_success(&run_auto_harness(
        &sandbox,
        &[
            "auto-harness",
            "experience",
            "list",
            "--limit",
            "4",
            "--type",
            "failure",
        ],
    ));
    assert_eq!(
        non_empty_lines(&widened),
        ["[failure] beta builder: build broke"]
    );
}

#[test]
fn experience_search_matches_topic_summary_and_details_only() {
    let sandbox = Sandbox::new();
    seed_store(&sandbox);
    let cases = [
        ("parser", "[insight] alpha parser: faster parser"), // topic
        ("faster", "[insight] alpha parser: faster parser"), // summary
        ("cache", "[insight] alpha parser: faster parser"),  // details
    ];
    for (query, expected) in cases {
        let text = assert_success(&run_auto_harness(
            &sandbox,
            &["auto-harness", "experience", "search", query],
        ));
        assert_eq!(
            non_empty_lines(&text),
            [expected],
            "query {query:?} matched {text:?}"
        );
    }
    // `outcome` is never searched.
    let text = assert_success(&run_auto_harness(
        &sandbox,
        &["auto-harness", "experience", "search", "OUTCOME-TOKEN"],
    ));
    assert_eq!(text.trim(), "no matching experiences");
}

#[test]
fn experience_search_ignores_tokens_shorter_than_two_characters() {
    let sandbox = Sandbox::new();
    seed_store(&sandbox);
    let text = assert_success(&run_auto_harness(
        &sandbox,
        &["auto-harness", "experience", "search", "a"],
    ));
    assert_eq!(
        text.trim(),
        "no matching experiences",
        "a one-character token is not a keyword"
    );
    // A short token next to a usable one is simply dropped.
    let text = assert_success(&run_auto_harness(
        &sandbox,
        &["auto-harness", "experience", "search", "a builder"],
    ));
    assert_eq!(
        non_empty_lines(&text),
        ["[failure] beta builder: build broke"]
    );
}

#[test]
fn experience_json_flag_prints_an_array() {
    let sandbox = Sandbox::new();
    seed_store(&sandbox);
    // The flag is global to the group, so both positions are accepted.
    let before = parse_json(&assert_success(&run_auto_harness(
        &sandbox,
        &["auto-harness", "--json", "experience", "list"],
    )));
    let rows = before.as_array().expect("a JSON array of experiences");
    assert_eq!(rows.len(), 4, "{before}");
    assert_eq!(rows[0]["topic"], "delta");
    assert_eq!(rows[0]["type"], "insight");
    assert_eq!(rows[0]["timestamp"], 4000.0);

    let after = parse_json(&assert_success(&run_auto_harness(
        &sandbox,
        &[
            "auto-harness",
            "experience",
            "list",
            "--json",
            "--limit",
            "1",
        ],
    )));
    assert_eq!(after.as_array().expect("an array").len(), 1, "{after}");
    assert_eq!(after[0]["topic"], "delta");
}

#[test]
fn empty_store_prints_the_documented_messages() {
    let sandbox = Sandbox::new();
    let listed = assert_success(&run_auto_harness(
        &sandbox,
        &["auto-harness", "experience", "list"],
    ));
    assert_eq!(listed.trim(), "no experience records");
    let searched = assert_success(&run_auto_harness(
        &sandbox,
        &["auto-harness", "experience", "search", "alpha"],
    ));
    assert_eq!(searched.trim(), "no matching experiences");
    assert!(
        sandbox.auto_harness_path("experience").is_dir(),
        "reading a fresh store creates the documented directory"
    );
}

// --- history and gap analysis ---------------------------------------------

#[test]
fn history_uses_the_list_line_format_and_honours_the_limit() {
    let sandbox = Sandbox::new();
    seed_store(&sandbox);
    let text = assert_success(&run_auto_harness(
        &sandbox,
        &["auto-harness", "history", "--limit", "1"],
    ));
    assert_eq!(non_empty_lines(&text), ["[insight] delta: newest"]);
}

#[test]
fn history_json_is_a_global_flag_before_the_subcommand() {
    let sandbox = Sandbox::new();
    seed_store(&sandbox);
    let value = parse_json(&assert_success(&run_auto_harness(
        &sandbox,
        &["auto-harness", "--json", "history"],
    )));
    let rows = value.as_array().expect("a JSON array of experiences");
    assert_eq!(rows.len(), 4, "the default history limit is 20: {value}");
    assert_eq!(rows[0]["topic"], "delta");
}

#[test]
fn gap_analyze_prints_the_placeholder_line() {
    let sandbox = Sandbox::new();
    let text = assert_success(&run_auto_harness(
        &sandbox,
        &["auto-harness", "gap-analyze"],
    ));
    assert!(
        text.contains("phase 1 placeholder"),
        "the placeholder text is printed: {text:?}"
    );
}

#[test]
fn gap_analyze_json_is_the_placeholder_object() {
    let sandbox = Sandbox::new();
    let value = parse_json(&assert_success(&run_auto_harness(
        &sandbox,
        &["auto-harness", "gap-analyze", "--json"],
    )));
    assert_eq!(value["gaps"], json!([]), "{value}");
    assert_eq!(value["placeholder"], json!(true), "{value}");
    assert!(
        value["note"]
            .as_str()
            .is_some_and(|note| note.contains("phase 1 placeholder")),
        "the note explains the placeholder: {value}"
    );
}
