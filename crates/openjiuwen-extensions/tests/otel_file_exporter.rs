//! Collector-free OTLP-JSON file exporter: crate-boundary behaviour.
//!
//! Python counterpart: `openjiuwen/extensions/observability/file_exporter.py`
//! (v0.1.20) plus the `traces_dir`/`file_retention_days` fields of
//! `observability/config.py`. Every test writes into a `tempfile::TempDir`, so
//! no test touches the workspace or the default `./traces` directory, and none
//! needs a collector, a socket or a clock rollover: the retention window is
//! simulated by setting a file's mtime instead of waiting.

#![cfg(feature = "otel")]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use futures::executor::block_on;
use opentelemetry::trace::{Span as _, Tracer as _};
use opentelemetry::KeyValue;
use opentelemetry_sdk::trace::SpanExporter;
use serde_json::{json, Value};

use openjiuwen_extensions::observability::file_exporter::TraceFileExporter;
use openjiuwen_extensions::otel::{init_otel_tracer, ExporterType, OtelTracerConfig};

/// One retention day, mirroring Python's `_SECONDS_PER_DAY`.
const DAY: u64 = 86_400;

/// Python's `_CLEANUP_INTERVAL`: cleanup runs on the 64th export.
const CLEANUP_INTERVAL: usize = 64;

/// A file-exporter configuration rooted at `dir`.
fn config(dir: &Path, retention_days: u32) -> OtelTracerConfig {
    OtelTracerConfig {
        traces_dir: dir.to_string_lossy().into_owned(),
        file_retention_days: retention_days,
        ..OtelTracerConfig::default()
    }
    .with_exporter_type(ExporterType::File)
}

/// Every `traces-*.jsonl` file under `dir`, sorted.
fn trace_files(dir: &Path) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = fs::read_dir(dir)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("traces-") && name.ends_with(".jsonl"))
        })
        .collect();
    files.sort();
    files
}

/// Parses a JSONL trace file, dropping the trailing empty line.
fn read_lines(path: &Path) -> Vec<Value> {
    let text = fs::read_to_string(path).unwrap();
    text.lines()
        .filter(|line| !line.is_empty())
        .map(|line| {
            serde_json::from_str(line).unwrap_or_else(|error| {
                panic!("line is not valid JSON ({error}): {line}");
            })
        })
        .collect()
}

/// The single span of a standalone single-span request line.
fn single_span(line: &Value) -> &Value {
    let spans = line["resourceSpans"][0]["scopeSpans"][0]["spans"]
        .as_array()
        .unwrap_or_else(|| panic!("a line carries `spans`: {line}"));
    assert_eq!(
        spans.len(),
        1,
        "every line must be a standalone single-span request: {line}"
    );
    &spans[0]
}

/// Asserts that `id` is a lowercase hex string of `len` characters.
fn assert_hex_id(id: &Value, len: usize) {
    let text = id
        .as_str()
        .unwrap_or_else(|| panic!("an id must be a string: {id}"));
    assert_eq!(text.len(), len, "unexpected id length: {text}");
    assert!(
        text.chars()
            .all(|digit| digit.is_ascii_digit() || ('a'..='f').contains(&digit)),
        "ids are lowercase hex: {text}"
    );
}

/// Sets a file's mtime without an extra dependency.
fn set_modified(path: &Path, time: SystemTime) {
    let file = fs::File::options().write(true).open(path).unwrap();
    file.set_modified(time).unwrap();
}

/// Creates `path` with an mtime `age` seconds in the past.
fn write_aged(path: &Path, contents: &[u8], age: Duration) -> PathBuf {
    fs::write(path, contents).unwrap();
    set_modified(path, SystemTime::now() - age);
    path.to_path_buf()
}

#[test]
fn a_batch_writes_one_single_span_otlp_json_line_per_span() {
    // file_exporter.py: every ended span becomes one line of
    // `traces-<YYYY-MM-DD>.jsonl`, each a standalone single-span
    // ExportTraceServiceRequest with hex ids.
    let dir = tempfile::tempdir().unwrap();
    let handle = init_otel_tracer(config(dir.path(), 0)).expect("the file exporter initializes");
    let tracer = handle
        .tracer()
        .expect("the file exporter is a live provider");

    tracer
        .span_builder("file.first")
        .with_attributes([KeyValue::new("custom.attr", "value-1")])
        .start(&tracer)
        .end();
    tracer.span_builder("file.second").start(&tracer).end();
    handle.force_flush();

    let files = trace_files(dir.path());
    assert_eq!(files.len(), 1, "one file per day: {files:?}");
    let expected_name = format!("traces-{}.jsonl", chrono::Local::now().format("%Y-%m-%d"));
    assert_eq!(
        files[0].file_name().unwrap().to_str().unwrap(),
        expected_name,
        "the daily file name carries the local date"
    );

    let lines = read_lines(&files[0]);
    assert!(lines.len() >= 2, "one line per span, got {lines:?}");

    let mut names = Vec::new();
    for line in &lines {
        let span = single_span(line);
        assert_hex_id(&span["traceId"], 32);
        assert_hex_id(&span["spanId"], 16);
        if !span["parentSpanId"].is_null() {
            assert_hex_id(&span["parentSpanId"], 16);
        }
        names.push(span["name"].as_str().unwrap().to_owned());

        // The provider handed its resource to the exporter, so the line is a
        // complete OTLP request rather than a bare span.
        let resource_attributes = line["resourceSpans"][0]["resource"]["attributes"]
            .as_array()
            .expect("the resource carries attributes");
        assert!(
            resource_attributes
                .contains(&json!({"key": "service.name", "value": {"stringValue": "openjiuwen"}})),
            "unexpected resource: {resource_attributes:?}"
        );
    }
    assert!(names.contains(&"file.first".to_owned()), "names: {names:?}");
    assert!(
        names.contains(&"file.second".to_owned()),
        "names: {names:?}"
    );
}

#[test]
fn a_span_attribute_survives_the_json_line() {
    // The line comes from the same encoder the OTLP/HTTP exporter uses, so an
    // attribute must round-trip unchanged.
    let dir = tempfile::tempdir().unwrap();
    let handle = init_otel_tracer(config(dir.path(), 0)).expect("the file exporter initializes");
    let tracer = handle
        .tracer()
        .expect("the file exporter is a live provider");
    tracer
        .span_builder("file.attributes")
        .with_attributes([
            KeyValue::new("custom.string", "value-1"),
            KeyValue::new("custom.count", 42_i64),
            KeyValue::new("custom.flag", true),
        ])
        .start(&tracer)
        .end();
    handle.force_flush();

    let files = trace_files(dir.path());
    assert_eq!(files.len(), 1);
    let lines = read_lines(&files[0]);
    let span = lines
        .iter()
        .map(single_span)
        .find(|span| span["name"] == json!("file.attributes"))
        .expect("the span is exported");
    let attributes = span["attributes"].as_array().expect("attributes");
    for expected in [
        json!({"key": "custom.string", "value": {"stringValue": "value-1"}}),
        json!({"key": "custom.count", "value": {"intValue": "42"}}),
        json!({"key": "custom.flag", "value": {"boolValue": true}}),
    ] {
        assert!(
            attributes.contains(&expected),
            "missing {expected}: {attributes:?}"
        );
    }
}

#[test]
fn a_second_export_appends_to_the_same_daily_file() {
    // file_exporter.py: appends accumulate in one file per calendar day; the
    // exporter never rotates per export.
    let dir = tempfile::tempdir().unwrap();
    let handle = init_otel_tracer(config(dir.path(), 0)).expect("the file exporter initializes");
    let tracer = handle
        .tracer()
        .expect("the file exporter is a live provider");

    tracer.span_builder("file.append.one").start(&tracer).end();
    handle.force_flush();
    let files = trace_files(dir.path());
    assert_eq!(files.len(), 1);
    let first = read_lines(&files[0]).len();
    assert!(first >= 1, "the first export wrote a line");

    tracer.span_builder("file.append.two").start(&tracer).end();
    handle.force_flush();
    let files = trace_files(dir.path());
    assert_eq!(files.len(), 1, "one file per day, not per export");
    let lines = read_lines(&files[0]);
    assert!(
        lines.len() > first,
        "the second export appends: {first} then {}",
        lines.len()
    );
    let names: Vec<&str> = lines
        .iter()
        .map(|line| single_span(line)["name"].as_str().unwrap())
        .collect();
    assert!(names.contains(&"file.append.one"), "names: {names:?}");
    assert!(names.contains(&"file.append.two"), "names: {names:?}");
}

#[test]
fn retention_prunes_expired_trace_files_on_the_cleanup_interval() {
    // file_exporter.py: `_maybe_cleanup` sweeps every `_CLEANUP_INTERVAL`
    // exports, deleting only regular `*.jsonl` files whose mtime predates
    // `retention_days * 86400`; nothing else is touched.
    let dir = tempfile::tempdir().unwrap();
    let old = write_aged(
        &dir.path().join("traces-2000-01-01.jsonl"),
        b"{}\n",
        Duration::from_secs(8 * DAY),
    );
    let notes = write_aged(
        &dir.path().join("notes.txt"),
        b"keep me\n",
        Duration::from_secs(365 * DAY),
    );
    let exporter = TraceFileExporter::new(dir.path(), 7);
    let today = exporter.traces_file();
    fs::write(&today, b"{}\n").unwrap();

    // Empty batches write nothing but still count as export cycles.
    for _ in 0..CLEANUP_INTERVAL - 1 {
        block_on(exporter.export(Vec::new())).unwrap();
    }
    assert!(
        old.exists(),
        "cleanup waits for the {CLEANUP_INTERVAL}th export"
    );

    block_on(exporter.export(Vec::new())).unwrap();
    assert!(!old.exists(), "an expired trace file is pruned");
    assert!(notes.exists(), "a non-`.jsonl` file is never pruned");
    assert!(
        today.exists(),
        "a fresh trace file inside the window survives"
    );
}

#[test]
fn retention_zero_never_prunes() {
    // file_exporter.py: `retention_days <= 0` disables the sweep entirely.
    let dir = tempfile::tempdir().unwrap();
    let old = write_aged(
        &dir.path().join("traces-2000-01-01.jsonl"),
        b"{}\n",
        Duration::from_secs(365 * DAY),
    );
    let exporter = TraceFileExporter::new(dir.path(), 0);

    for _ in 0..CLEANUP_INTERVAL {
        block_on(exporter.export(Vec::new())).unwrap();
    }
    assert!(old.exists(), "retention_days = 0 keeps every trace file");
}

#[test]
fn a_missing_traces_directory_is_not_an_export_error() {
    // file_exporter.py: `_cleanup_old_files` ignores FileNotFoundError, and a
    // failed export never fails the caller.
    let dir = tempfile::tempdir().unwrap();
    let missing = dir.path().join("never-created");
    let exporter = TraceFileExporter::new(&missing, 7);
    // Remove the directory `new` created, then keep exporting: cleanup must not
    // raise, and the append failure must stay a logged warning.
    fs::remove_dir(&missing).unwrap();

    for _ in 0..CLEANUP_INTERVAL {
        block_on(exporter.export(Vec::new())).unwrap();
    }
    assert!(
        !path_exists(&missing),
        "the export never writes without a directory"
    );
}

/// Reports whether `path` exists, without panicking on an I/O error.
fn path_exists(path: &Path) -> bool {
    path.try_exists().unwrap_or(false)
}
