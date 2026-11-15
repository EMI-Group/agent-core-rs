//! Offline, process-level `sessions` scenarios of the `openjiuwen` binary.
//!
//! The catalogue command is driven through a real child process against a
//! temporary `OPENJIUWEN_HOME` and a temporary `-w` workspace
//! ([`support::Sandbox`]), so the three stores it merges are the ones a user
//! really has:
//!
//! - **cli** — hand-written `<home>/sessions/<id>.json` records, in the
//!   documented shape the CLI itself writes;
//! - **execution** — a `persistent_agent/logs` tree, here deliberately damaged
//!   (a `meta.json` that is not JSON, a directory without one) to prove a broken
//!   journal is skipped instead of failing the listing;
//! - **state** — the `persistent_agent/sessions` tree, which this suite leaves
//!   empty: the persistent-agent seeding cases live in `tests/resume.rs`.
//!
//! Covered: the empty catalogue, newest-first listing with its kind filter and
//! limit, `show` in both output modes, the skip-a-corrupt-journal rule, the
//! `not_found` of an unknown id, `prune` (dry run, `--all`, `--older-than`), the
//! `validation` error of a selector-less prune, the clap conflict of `--all` with
//! `--older-than`, and the three help pages.
//!
//! Nothing here builds a model, binds a socket or touches the network: the suite
//! is pure filesystem plus argv.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod support;

use chrono::{SecondsFormat, Utc};
use serde_json::json;
use support::{assert_exit, assert_success, has_row, non_empty_lines, parse_json, stderr, Sandbox};

/// The `-w <workspace>` plus `sessions` prefix every invocation carries.
///
/// `-w` is placed before the subcommand on purpose: clap treats it as global, and
/// this is the spelling the README documents.
fn args(sandbox: &Sandbox, tail: &[&str]) -> Vec<String> {
    let mut args = vec![
        "-w".to_owned(),
        sandbox.workspace().display().to_string(),
        "sessions".to_owned(),
    ];
    args.extend(tail.iter().map(|arg| (*arg).to_owned()));
    args
}

/// Returns `now` as the RFC 3339 timestamp a freshly written record carries.
fn now() -> String {
    Utc::now().to_rfc3339_opts(SecondsFormat::Secs, true)
}

/// Writes a CLI session record for `id` with the documented on-disk shape.
///
/// The path is the one [`openjiuwen_harness::cli::SessionStore`] reads
/// (`<home>/sessions/<id>.json`), so the suite does not depend on the CLI to
/// create the fixture it then inspects.
fn write_record(sandbox: &Sandbox, id: &str, created_at: &str, messages: &[(&str, &str)]) {
    let messages: Vec<serde_json::Value> = messages
        .iter()
        .map(|(role, content)| json!({"role": role, "content": content, "timestamp": created_at}))
        .collect();
    let record = json!({
        "session_id": id,
        "model": "mock-model",
        "created_at": created_at,
        "messages": messages,
    });
    sandbox.write(
        format!("home/sessions/{id}.json"),
        serde_json::to_string_pretty(&record).expect("a JSON record"),
    );
}

/// Returns the four tab-separated columns of one `list` row.
fn columns(line: &str) -> Vec<&str> {
    line.split('\t').collect()
}

#[test]
fn an_empty_catalogue_prints_no_sessions() {
    let sandbox = Sandbox::new();

    let text = assert_success(&sandbox.run_blocking(&args(&sandbox, &["list"])));
    assert_eq!(text.trim(), "no sessions");

    let text = assert_success(&sandbox.run_blocking(&args(&sandbox, &["list", "--json"])));
    assert_eq!(parse_json(&text), json!([]));
}

#[test]
fn hand_written_records_are_listed_newest_first() {
    let sandbox = Sandbox::new();
    write_record(
        &sandbox,
        "cli-aaaa1111",
        "2024-01-02T03:04:05Z",
        &[("user", "first question"), ("assistant", "first answer")],
    );
    write_record(
        &sandbox,
        "cli-bbbb2222",
        "2024-06-07T08:09:10Z",
        &[("user", "later question"), ("assistant", "later answer")],
    );

    let text = assert_success(&sandbox.run_blocking(&args(&sandbox, &["list"])));
    let lines = non_empty_lines(&text);
    assert_eq!(lines.len(), 2, "both records are listed: {text:?}");
    let newest = columns(lines[0]);
    assert_eq!(newest[0], "cli-bbbb2222", "newest first: {text:?}");
    assert_eq!(newest[1], "2024-06-07T08:09:10Z");
    assert_eq!(newest[2], "cli");
    assert_eq!(newest[3], "2 messages, model mock-model");
    let oldest = columns(lines[1]);
    assert_eq!(oldest[0], "cli-aaaa1111");
    assert_eq!(oldest[1], "2024-01-02T03:04:05Z");
    assert_eq!(oldest[2], "cli");

    let value = parse_json(&assert_success(
        &sandbox.run_blocking(&args(&sandbox, &["list", "--json"])),
    ));
    let rows = value.as_array().expect("a JSON array of rows");
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0]["id"], "cli-bbbb2222");
    assert_eq!(rows[0]["kind"], "cli");
    assert_eq!(rows[0]["created_at"], "2024-06-07T08:09:10Z");
    assert_eq!(rows[0]["detail"], "2 messages, model mock-model");
    assert_eq!(rows[1]["id"], "cli-aaaa1111");

    let limited = parse_json(&assert_success(
        &sandbox.run_blocking(&args(&sandbox, &["list", "--limit", "1", "--json"])),
    ));
    let rows = limited.as_array().expect("a JSON array of rows");
    assert_eq!(
        rows.len(),
        1,
        "the limit truncates after ordering: {limited}"
    );
    assert_eq!(rows[0]["id"], "cli-bbbb2222");

    let executions =
        assert_success(&sandbox.run_blocking(&args(&sandbox, &["list", "--kind", "execution"])));
    assert_eq!(
        executions.trim(),
        "no sessions",
        "the kind filter runs before the limit and there are no journals"
    );
}

#[test]
fn showing_a_cli_record_prints_its_transcript() {
    let sandbox = Sandbox::new();
    write_record(
        &sandbox,
        "cli-aaaa1111",
        "2024-01-02T03:04:05Z",
        &[("user", "first question"), ("assistant", "first answer")],
    );

    let text = assert_success(&sandbox.run_blocking(&args(&sandbox, &["show", "cli-aaaa1111"])));
    assert!(text.contains("session cli-aaaa1111\n"), "got {text:?}");
    assert!(text.contains("kind cli\n"), "got {text:?}");
    assert!(
        text.contains("created 2024-01-02T03:04:05Z\n"),
        "got {text:?}"
    );
    assert!(text.contains("messages 2\n"), "got {text:?}");
    assert!(
        text.lines().any(|line| line == "user: first question"),
        "the transcript is printed one message per line: {text:?}"
    );
    assert!(
        text.lines().any(|line| line == "assistant: first answer"),
        "the transcript is printed one message per line: {text:?}"
    );
    assert!(
        !text.contains("plan execution"),
        "a CLI record has no journal plan: {text:?}"
    );

    let value = parse_json(&assert_success(
        &sandbox.run_blocking(&args(&sandbox, &["show", "cli-aaaa1111", "--json"])),
    ));
    assert_eq!(value["id"], "cli-aaaa1111");
    assert_eq!(value["kind"], "cli");
    assert_eq!(value["created_at"], "2024-01-02T03:04:05Z");
    assert_eq!(value["detail"], "2 messages, model mock-model");
    assert!(value["plan"].is_null(), "no plan for a CLI record: {value}");
    assert_eq!(
        value["record"]["session_id"], "cli-aaaa1111",
        "the JSON document carries the record behind the resume: {value}"
    );
    let messages = value["messages"]
        .as_array()
        .expect("a JSON array of messages");
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0]["role"], "user");
    assert_eq!(messages[0]["content"], "first question");
    assert_eq!(messages[1]["role"], "assistant");
    assert_eq!(messages[1]["content"], "first answer");
}

#[test]
fn a_corrupt_journal_is_skipped_instead_of_failing_the_listing() {
    let sandbox = Sandbox::new();
    sandbox.write(
        "workspace/persistent_agent/logs/broken/meta.json",
        "{not json",
    );
    std::fs::create_dir_all(sandbox.path("workspace/persistent_agent/logs/metadata-less"))
        .expect("a journal directory without metadata");
    write_record(
        &sandbox,
        "cli-aaaa1111",
        "2024-01-02T03:04:05Z",
        &[("user", "first question")],
    );

    let text = assert_success(&sandbox.run_blocking(&args(&sandbox, &["list", "--json"])));
    let value = parse_json(&text);
    let rows = value.as_array().expect("a JSON array of rows");
    assert_eq!(rows.len(), 1, "only the healthy record is listed: {text:?}");
    assert_eq!(rows[0]["id"], "cli-aaaa1111");
    for row in rows {
        assert_ne!(row["id"], "broken");
        assert_ne!(row["id"], "metadata-less");
    }
}

#[test]
fn an_unknown_id_is_a_typed_not_found() {
    let sandbox = Sandbox::new();
    let output = sandbox.run_blocking(&args(&sandbox, &["show", "does-not-exist"]));
    let text = assert_exit(&output, 1);
    assert!(text.is_empty(), "nothing reaches stdout: {text:?}");
    let note = stderr(&output);
    assert!(
        note.contains("error: not_found: unknown session: does-not-exist"),
        "got {note:?}"
    );
}

#[test]
fn prune_reports_the_selection_then_deletes_it() {
    let sandbox = Sandbox::new();
    write_record(
        &sandbox,
        "cli-aaaa1111",
        "2024-01-02T03:04:05Z",
        &[("user", "first question")],
    );
    write_record(
        &sandbox,
        "cli-bbbb2222",
        "2024-06-07T08:09:10Z",
        &[("user", "later question")],
    );

    let text =
        assert_success(&sandbox.run_blocking(&args(&sandbox, &["prune", "--all", "--dry-run"])));
    let lines = non_empty_lines(&text);
    assert_eq!(
        lines,
        vec![
            "would prune cli\tcli-bbbb2222",
            "would prune cli\tcli-aaaa1111"
        ],
        "a dry run reports the selection newest first"
    );

    let listed = parse_json(&assert_success(
        &sandbox.run_blocking(&args(&sandbox, &["list", "--json"])),
    ));
    assert_eq!(
        listed.as_array().expect("rows").len(),
        2,
        "a dry run deletes nothing: {listed}"
    );

    let text = assert_success(&sandbox.run_blocking(&args(&sandbox, &["prune", "--all"])));
    let lines = non_empty_lines(&text);
    assert_eq!(
        lines,
        vec!["pruned cli\tcli-bbbb2222", "pruned cli\tcli-aaaa1111"],
        "the real prune reports what it deleted"
    );

    let text = assert_success(&sandbox.run_blocking(&args(&sandbox, &["list"])));
    assert_eq!(text.trim(), "no sessions", "both records are gone");

    let value = parse_json(&assert_success(
        &sandbox.run_blocking(&args(&sandbox, &["prune", "--all", "--json"])),
    ));
    assert_eq!(value["dry_run"], false);
    assert_eq!(value["pruned"], 0);
    assert_eq!(value["entries"], json!([]));
}

#[test]
fn prune_needs_a_selector() {
    let sandbox = Sandbox::new();
    write_record(&sandbox, "cli-aaaa1111", &now(), &[("user", "just now")]);

    let text =
        assert_success(&sandbox.run_blocking(&args(&sandbox, &["prune", "--older-than", "7"])));
    assert_eq!(
        text.trim(),
        "nothing to prune",
        "a fresh record is younger than the age filter"
    );

    let text = assert_success(&sandbox.run_blocking(&args(&sandbox, &["list"])));
    assert!(
        text.contains("cli-aaaa1111"),
        "the age filter deleted nothing: {text:?}"
    );

    let output = sandbox.run_blocking(&args(&sandbox, &["prune"]));
    let text = assert_exit(&output, 1);
    assert!(text.is_empty(), "nothing reaches stdout: {text:?}");
    let note = stderr(&output);
    assert!(
        note.contains("error: validation: prune needs --older-than <DAYS> or --all"),
        "got {note:?}"
    );

    let output = sandbox.run_blocking(&args(&sandbox, &["prune", "--all", "--older-than", "1"]));
    let text = assert_exit(&output, 2);
    assert!(
        text.is_empty(),
        "a clap conflict writes nothing to stdout: {text:?}"
    );
    let note = stderr(&output);
    assert!(
        note.contains("--older-than") && note.contains("--all"),
        "the usage error names both selectors: {note:?}"
    );
}

#[test]
fn sessions_help_documents_the_commands() {
    let sandbox = Sandbox::new();

    let text = assert_success(&sandbox.run_blocking(&args(&sandbox, &["--help"])));
    for row in ["list", "show", "prune"] {
        assert!(has_row(&text, row), "no `{row}` subcommand in {text:?}");
    }
    for flag in ["--data-dir", "--json"] {
        assert!(text.contains(flag), "expected `{flag}` in {text:?}");
    }

    let text = assert_success(&sandbox.run_blocking(&args(&sandbox, &["list", "--help"])));
    for flag in ["--kind", "--limit"] {
        assert!(text.contains(flag), "expected `{flag}` in {text:?}");
    }

    let text = assert_success(&sandbox.run_blocking(&args(&sandbox, &["prune", "--help"])));
    for flag in ["--older-than", "--all", "--dry-run"] {
        assert!(text.contains(flag), "expected `{flag}` in {text:?}");
    }

    let text = assert_success(&sandbox.run_blocking(&args(&sandbox, &["show", "--help"])));
    assert!(
        text.contains("<ID>") || text.to_lowercase().contains("session"),
        "the show help documents its id argument: {text:?}"
    );
}
