//! Process-level suite of the `cron` command group.
//!
//! `cron` is a host-side view of the same scheduler the agent's `cron` tool
//! uses, so its contract is pinned here through real child processes:
//!
//! 1. **The store** — a fresh workspace reports `no cron jobs`, `create` writes
//!    `<workspace>/cron/jobs.json`, and `list --json` reports the persisted job.
//! 2. **Mutations** — `toggle` flips the flag both ways, `cancel` removes the
//!    job, and `run` records a manual execution whose receipt `runs` shows.
//! 3. **Reads** — `preview --count` returns exactly that many strictly
//!    increasing RFC 3339 occurrences, and `status` reports the counts.
//! 4. **Failures** — an unknown id is a typed `not_found` error (exit `1`,
//!    empty stdout), an unparsable expression a `validation` error naming the
//!    cron expression.
//!
//! Every case gets its own [`support::Sandbox`] (temp cwd, temp
//! `OPENJIUWEN_HOME`, temp `-w` workspace), builds no model, and is bounded by
//! [`support::CHILD_WAIT`].
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod support;

use std::process::Output;

use serde_json::Value;
use support::{assert_exit, assert_success, non_empty_lines, parse_json, stderr, Sandbox};

/// Runs one `cron` invocation scoped to the sandbox's workspace.
fn run_cron(sandbox: &Sandbox, tail: &[&str]) -> Output {
    let mut args: Vec<String> = sandbox.workspace_flag().into();
    args.push("cron".to_owned());
    args.extend(tail.iter().map(|arg| (*arg).to_owned()));
    sandbox.run_blocking(&args)
}

/// Creates a job and returns its id, read from the `--json` document.
///
/// The id is never parsed out of the text form: the JSON object is the
/// contract, so a change to the text layout cannot silently break the suite.
fn create_job(sandbox: &Sandbox, expression: &str, prompt: &str) -> String {
    let output = run_cron(
        sandbox,
        &["create", expression, "--prompt", prompt, "--json"],
    );
    let job = parse_json(&assert_success(&output));
    job["id"]
        .as_str()
        .unwrap_or_else(|| panic!("every created job has an id: {job}"))
        .to_owned()
}

/// Returns the elements of a JSON array.
fn array(value: &Value) -> &Vec<Value> {
    value
        .as_array()
        .unwrap_or_else(|| panic!("expected a JSON array, got {value}"))
}

/// Returns the string elements of a JSON array.
fn strings(value: &Value) -> Vec<&str> {
    array(value)
        .iter()
        .map(|item| {
            item.as_str()
                .unwrap_or_else(|| panic!("expected a JSON string, got {item}"))
        })
        .collect()
}

#[test]
fn list_on_a_fresh_workspace_reports_no_jobs() {
    let sandbox = Sandbox::new();
    let output = run_cron(&sandbox, &["list"]);
    assert_eq!(assert_success(&output).trim(), "no cron jobs");
}

#[test]
fn create_persists_the_job_and_list_reports_it() {
    let sandbox = Sandbox::new();
    let output = run_cron(
        &sandbox,
        &[
            "create",
            "*/5 * * * *",
            "--prompt",
            "ping",
            "--description",
            "every five minutes",
            "--json",
        ],
    );
    let created = parse_json(&assert_success(&output));
    let id = created["id"].as_str().expect("every created job has an id");
    assert_eq!(created["cron"], "*/5 * * * *");
    assert_eq!(created["prompt"], "ping");
    assert_eq!(created["description"], "every five minutes");
    assert_eq!(created["enabled"], true);

    let jobs_file = sandbox.workspace().join("cron").join("jobs.json");
    assert!(
        jobs_file.is_file(),
        "creating a job writes {}",
        jobs_file.display()
    );

    let listed = parse_json(&assert_success(&run_cron(&sandbox, &["list", "--json"])));
    let jobs = array(&listed);
    assert_eq!(jobs.len(), 1);
    assert_eq!(jobs[0]["id"].as_str(), Some(id));
    assert_eq!(jobs[0]["cron"], "*/5 * * * *");
    assert_eq!(jobs[0]["prompt"], "ping");
    assert_eq!(jobs[0]["enabled"], true);

    // The text form of `get` is the list row for that one job.
    let text = assert_success(&run_cron(&sandbox, &["get", id]));
    assert!(
        text.starts_with(&format!("{id}\t*/5 * * * *\tnext ")),
        "get prints the job row: {text:?}"
    );
    assert!(text.trim_end().ends_with("\tenabled"), "{text:?}");
}

#[test]
fn toggle_flips_the_enabled_flag() {
    let sandbox = Sandbox::new();
    let id = create_job(&sandbox, "0 * * * *", "hourly");

    let disabled = parse_json(&assert_success(&run_cron(
        &sandbox,
        &["toggle", &id, "--json"],
    )));
    assert_eq!(disabled["enabled"], false);

    let enabled = parse_json(&assert_success(&run_cron(
        &sandbox,
        &["toggle", &id, "--json"],
    )));
    assert_eq!(enabled["enabled"], true);

    let text = assert_success(&run_cron(&sandbox, &["toggle", &id]));
    assert!(
        text.starts_with("disabled "),
        "text mode leads with the new state: {text:?}"
    );
}

#[test]
fn preview_returns_the_requested_occurrences() {
    let sandbox = Sandbox::new();
    let id = create_job(&sandbox, "*/5 * * * *", "ping");

    let preview = parse_json(&assert_success(&run_cron(
        &sandbox,
        &["preview", &id, "--count", "3", "--json"],
    )));
    let stamps = strings(&preview);
    assert_eq!(stamps.len(), 3, "three occurrences were requested");
    for stamp in &stamps {
        chrono::DateTime::parse_from_rfc3339(stamp)
            .unwrap_or_else(|err| panic!("{stamp:?} is not RFC 3339: {err}"));
    }
    for pair in stamps.windows(2) {
        assert!(pair[0] < pair[1], "occurrences increase: {pair:?}");
    }

    let text = assert_success(&run_cron(&sandbox, &["preview", &id, "--count", "3"]));
    assert_eq!(
        non_empty_lines(&text).len(),
        3,
        "text mode prints one per line"
    );
}

#[test]
fn run_records_a_manual_execution() {
    let sandbox = Sandbox::new();
    let id = create_job(&sandbox, "0 9 * * *", "standup");

    let text = assert_success(&run_cron(&sandbox, &["run", &id]));
    assert!(
        text.starts_with(&format!("ran {id}; next run at ")),
        "run reports the advanced occurrence: {text:?}"
    );

    let runs = parse_json(&assert_success(&run_cron(&sandbox, &["runs", "--json"])));
    let runs = array(&runs);
    assert_eq!(runs.len(), 1);
    assert_eq!(runs[0]["trigger"], "manual");
    assert_eq!(runs[0]["job_id"].as_str(), Some(id.as_str()));

    let text = assert_success(&run_cron(&sandbox, &["runs"]));
    assert!(text.contains("\tmanual\toccurrence "), "{text:?}");
}

#[test]
fn unknown_jobs_are_not_found_errors() {
    let sandbox = Sandbox::new();
    for tail in [
        vec!["get", "missing"],
        vec!["toggle", "missing"],
        vec!["preview", "missing"],
        vec!["run", "missing"],
        vec!["cancel", "missing"],
    ] {
        let output = run_cron(&sandbox, &tail);
        let out = assert_exit(&output, 1);
        assert!(out.is_empty(), "{tail:?} must not write to stdout: {out:?}");
        let err = stderr(&output);
        assert!(
            err.contains("error: not_found:"),
            "{tail:?} reports the error kind: {err:?}"
        );
        assert!(
            err.contains("cron job not found: missing"),
            "{tail:?} names the missing id: {err:?}"
        );
    }
}

#[test]
fn create_rejects_an_invalid_expression() {
    let sandbox = Sandbox::new();
    let output = run_cron(&sandbox, &["create", "not a cron", "--prompt", "x"]);
    let out = assert_exit(&output, 1);
    assert!(
        out.is_empty(),
        "a rejected expression writes no stdout: {out:?}"
    );
    let err = stderr(&output);
    assert!(
        err.contains("error: validation:"),
        "the parse failure is a validation error: {err:?}"
    );
    assert!(
        err.contains("cron expression"),
        "the diagnostic names the cron expression: {err:?}"
    );
    assert!(
        !sandbox.workspace().join("cron").join("jobs.json").exists(),
        "a rejected expression stores nothing"
    );
}

#[test]
fn status_reports_counts_and_the_next_run() {
    let sandbox = Sandbox::new();
    let empty = parse_json(&assert_success(&run_cron(&sandbox, &["status", "--json"])));
    assert_eq!(empty["total"], 0);
    assert_eq!(empty["enabled"], 0);
    assert_eq!(empty["runs"], 0);

    let id = create_job(&sandbox, "*/5 * * * *", "ping");
    assert_success(&run_cron(&sandbox, &["run", &id]));

    let status = parse_json(&assert_success(&run_cron(&sandbox, &["status", "--json"])));
    assert_eq!(status["total"], 1);
    assert_eq!(status["enabled"], 1);
    assert_eq!(status["disabled"], 0);
    assert_eq!(status["due"], 0);
    assert_eq!(status["runs"], 1);

    let text = assert_success(&run_cron(&sandbox, &["status"]));
    let lines = non_empty_lines(&text);
    assert_eq!(
        lines.first().copied(),
        Some("1 cron job(s): 1 enabled, 0 disabled, 0 due; 1 run record(s)")
    );
    assert!(
        lines.iter().any(|line| line.starts_with("next run at ")),
        "the pending occurrence is reported: {text:?}"
    );
}

#[test]
fn cancel_removes_the_job() {
    let sandbox = Sandbox::new();
    let id = create_job(&sandbox, "@daily", "digest");

    let text = assert_success(&run_cron(&sandbox, &["cancel", &id]));
    assert_eq!(text.trim(), format!("cancelled {id}"));

    let listed = parse_json(&assert_success(&run_cron(&sandbox, &["list", "--json"])));
    assert!(array(&listed).is_empty(), "the job is gone: {listed}");
}
