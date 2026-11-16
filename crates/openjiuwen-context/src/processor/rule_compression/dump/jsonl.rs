//! The JSONL dump writer and the reader that replays both dump formats.
//!
//! The writer appends one compact JSON object per line to
//! `{session_id or "session"}-{context_id or "context"}.jsonl` and flushes, so a
//! conversation accumulates in one file per session and context and a crash
//! keeps every completed line.
//!
//! The reader accepts both formats: a file that parses as one JSON object is a
//! single pretty-printed dump, and anything else is read line by line with blank
//! lines skipped. A malformed record is
//! [`Error::Validation`](openjiuwen_common::error::Error) naming the
//! source and the one-based line number — nothing is silently dropped, and a
//! pretty-printed file whose object is broken is reported at its first line.
use std::io::Write as _;
use std::path::{Path, PathBuf};

use openjiuwen_common::error::{Error, Result};

use super::{create_dir, dump_payload, jsonl_file_name, DumpRecord, RuleCompressionDumpRecord};

/// Appends `record` as one JSONL line in `dir`, returning the file path.
///
/// The directory is created when missing and the file is opened for append, so
/// concurrent writers cannot truncate each other's history.
pub fn append_jsonl(record: &DumpRecord<'_>, dir: &Path) -> Result<PathBuf> {
    create_dir(dir)?;
    let path = dir.join(jsonl_file_name(record.session_id, record.context_id));
    let mut line = serde_json::to_string(&dump_payload(record))
        .map_err(|error| Error::sys_op(format!("serialize dump line: {error}")))?;
    line.push('\n');
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .map_err(|error| Error::sys_op(format!("open dump {}: {error}", path.display())))?;
    file.write_all(line.as_bytes())
        .map_err(|error| Error::sys_op(format!("append dump {}: {error}", path.display())))?;
    file.flush()
        .map_err(|error| Error::sys_op(format!("flush dump {}: {error}", path.display())))?;
    Ok(path)
}

/// Reads one dump file, in either format, oldest record first.
pub fn read_dump_file(path: &Path) -> Result<Vec<RuleCompressionDumpRecord>> {
    let text = std::fs::read_to_string(path)
        .map_err(|error| Error::sys_op(format!("read dump {}: {error}", path.display())))?;
    read_dump_text(&text, &path.display().to_string())
}

/// Reads every dump file of `dir`, sorted by file name so replay is deterministic.
///
/// Only `.json` and `.jsonl` files are read; other entries are ignored.
pub fn read_dump_dir(dir: &Path) -> Result<Vec<RuleCompressionDumpRecord>> {
    let entries = std::fs::read_dir(dir)
        .map_err(|error| Error::sys_op(format!("read dump dir {}: {error}", dir.display())))?;
    let mut files: Vec<PathBuf> = entries
        .filter_map(std::result::Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.is_file() && is_dump_file(path))
        .collect();
    files.sort();
    let mut records = Vec::new();
    for file in files {
        records.extend(read_dump_file(&file)?);
    }
    Ok(records)
}

/// Reads dump records out of `text`, attributing failures to `source`.
///
/// A `source` is a file path used verbatim in the error message, so a caller
/// replaying an in-memory dump names it something meaningful.
pub fn read_dump_text(text: &str, source: &str) -> Result<Vec<RuleCompressionDumpRecord>> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Ok(Vec::new());
    }
    if let Ok(record) = serde_json::from_str::<RuleCompressionDumpRecord>(trimmed) {
        return Ok(vec![with_path(record, source)]);
    }
    let mut records = Vec::new();
    for (index, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let record = serde_json::from_str::<RuleCompressionDumpRecord>(line).map_err(|error| {
            Error::validation(format!(
                "malformed rule-compression dump record at {source}:{}: {error}",
                index + 1
            ))
        })?;
        records.push(with_path(record, source));
    }
    Ok(records)
}

/// Sets the record's `path` to the source it was read from.
fn with_path(mut record: RuleCompressionDumpRecord, source: &str) -> RuleCompressionDumpRecord {
    record.path = PathBuf::from(source);
    record
}

/// Whether `path` carries one of the two dump extensions.
fn is_dump_file(path: &Path) -> bool {
    path.extension()
        .and_then(std::ffi::OsStr::to_str)
        .is_some_and(|extension| {
            extension.eq_ignore_ascii_case("json") || extension.eq_ignore_ascii_case("jsonl")
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn record<'a>(original: &'a str, compressed: &'a str) -> DumpRecord<'a> {
        DumpRecord {
            timestamp: 1.5,
            session_id: "session-1",
            context_id: "main",
            tool_call_id: "call_1",
            content_type: "LOG",
            pass_name: "add",
            lossy: true,
            original,
            compressed,
            details: None,
        }
    }

    #[test]
    fn jsonl_appends_one_line_per_application_and_reads_back() {
        let dir = tempfile::tempdir().expect("temp dir");
        let first = append_jsonl(&record("before", "after"), dir.path()).expect("first");
        let second = append_jsonl(&record("more", "less"), dir.path()).expect("second");
        assert_eq!(first, second);
        assert_eq!(
            first.file_name().and_then(|n| n.to_str()),
            Some("session-1-main.jsonl")
        );
        let text = std::fs::read_to_string(&first).expect("read");
        assert_eq!(text.lines().count(), 2);
        let records = read_dump_file(&first).expect("reader");
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].original_content, "before");
        assert_eq!(records[1].compressed_content, "less");
        assert!(records.iter().all(|record| record.lossy));
        assert_eq!(records[0].path, first);
    }

    #[test]
    fn a_pretty_printed_dump_reads_back_as_one_record() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = super::super::write_dump(&record("before", "after"), dir.path()).expect("dump");
        let records = read_dump_dir(dir.path()).expect("reader");
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].rule_compression_type, "LOG");
        assert_eq!(records[0].original_chars, 6);
        assert_eq!(records[0].path, path);
    }

    #[test]
    fn blank_lines_are_skipped_and_malformed_lines_are_reported() {
        let mut text = serde_json::to_string(&json_record()).expect("json");
        text.push_str("\n\n{not json}\n");
        let error = read_dump_text(&text, "memory").expect_err("malformed");
        assert!(matches!(error, Error::Validation(_)));
        assert!(error.to_string().contains("memory:3"));

        let mut good = serde_json::to_string(&json_record()).expect("json");
        good.push_str("\n\n");
        let records = read_dump_text(&good, "memory").expect("reader");
        assert_eq!(records.len(), 1);
        assert_eq!(
            read_dump_text("  \n\n", "memory").expect("empty"),
            Vec::new()
        );
    }

    fn json_record() -> Value {
        serde_json::json!({
            "timestamp": 2.0,
            "session_id": "s",
            "context_id": "c",
            "tool_call_id": "call_2",
            "rule_compression_type": "PLAIN_TEXT",
            "rule_compression_pass": "ttl",
            "original_chars": 4,
            "compressed_chars": 2,
            "lossy": true,
            "details": Value::Null,
            "original_content": "abcd",
            "compressed_content": "ab",
        })
    }
}
