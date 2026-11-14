//! Debug dumps of one rule-compression application.
//!
//! Ports `RuleCompressionPipeline._dump_result`/`_dump_dir`. A dump is one JSON
//! file named `{epoch_ms}_{pass}_{type}_{tool_call_id}.json`, written under the
//! configured directory or `OPENJIUWEN_RULE_COMPRESSION_DUMP_DIR`, holding the
//! original and compressed content, their character counts, and the compressor's
//! `details`. Dumping is a debugging aid: the pipeline logs a warning and keeps
//! the compressed message when a write fails.
//!
//! The Python reference falls back to `<workspace>/context/<session>_context/
//! rule_compression_logs` when no directory is configured; the Rust pipeline has
//! no `ModelContext` workspace handle, so an unconfigured pipeline dumps nowhere.
use std::path::{Path, PathBuf};

use openjiuwen_common::error::{Error, Result};
use serde_json::{json, Value};

/// Environment variable that names the dump directory.
pub const RULE_COMPRESSION_DUMP_DIR_ENV: &str = "OPENJIUWEN_RULE_COMPRESSION_DUMP_DIR";

/// Placeholder used when a dump field has no value.
const UNKNOWN: &str = "unknown";

/// Longest filename component kept from one sanitized part.
const MAX_FILENAME_PART_CHARS: usize = 80;

/// One dumped compression.
#[derive(Debug, Clone)]
pub struct DumpRecord<'a> {
    /// Wall-clock seconds of the compression.
    pub timestamp: f64,
    /// Session the compressed message belongs to, `"unknown_session"` when unknown.
    pub session_id: &'a str,
    /// Context the compressed message belongs to, `"unknown_context"` when unknown.
    pub context_id: &'a str,
    /// Tool call answered by the compressed message.
    pub tool_call_id: &'a str,
    /// Detected content type (`JSON_ARRAY`, `GIT_DIFF`, …).
    pub content_type: &'a str,
    /// Pipeline pass (`add` or `ttl`).
    pub pass_name: &'a str,
    /// Content before compression.
    pub original: &'a str,
    /// Content after compression.
    pub compressed: &'a str,
    /// The compressor's details payload.
    pub details: Option<&'a Value>,
}

/// Sanitizes one filename component the way Python's `_safe_filename_part` does.
#[must_use]
pub fn safe_filename_part(value: &str) -> String {
    let mut sanitized = String::with_capacity(value.len());
    for character in value.chars() {
        if character.is_ascii_alphanumeric() || matches!(character, '_' | '.' | '-') {
            sanitized.push(character);
        } else {
            sanitized.push('-');
        }
    }
    let trimmed = sanitized.trim_matches('-');
    let limited: String = trimmed.chars().take(MAX_FILENAME_PART_CHARS).collect();
    if limited.is_empty() {
        UNKNOWN.to_owned()
    } else {
        limited
    }
}

/// The dump file name for one compression.
#[must_use]
pub fn dump_file_name(
    timestamp: f64,
    pass_name: &str,
    content_type: &str,
    tool_call_id: &str,
) -> String {
    let epoch_ms = (timestamp * 1000.0) as i64;
    let parts = [
        epoch_ms.to_string(),
        safe_filename_part(pass_name),
        safe_filename_part(content_type),
        safe_filename_part(tool_call_id),
    ];
    format!("{}.json", parts.join("_"))
}

/// The directory dumps go to: the configured value, else the environment.
#[must_use]
pub fn resolve_dump_dir(configured: Option<&str>) -> Option<PathBuf> {
    if let Some(directory) = configured.map(str::trim).filter(|value| !value.is_empty()) {
        return Some(PathBuf::from(directory));
    }
    let from_env = std::env::var(RULE_COMPRESSION_DUMP_DIR_ENV).ok()?;
    let trimmed = from_env.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(PathBuf::from(trimmed))
    }
}

/// Writes `record` into `dir`, returning the file path.
///
/// The directory is created when missing; an existing dump is overwritten, which
/// only happens when two compressions of the same tool call land in the same
/// millisecond.
pub fn write_dump(record: &DumpRecord<'_>, dir: &Path) -> Result<PathBuf> {
    std::fs::create_dir_all(dir)
        .map_err(|error| Error::SysOp(format!("create dump dir {}: {error}", dir.display())))?;
    let path = dir.join(dump_file_name(
        record.timestamp,
        record.pass_name,
        record.content_type,
        record.tool_call_id,
    ));
    let payload = json!({
        "timestamp": record.timestamp,
        "session_id": record.session_id,
        "context_id": record.context_id,
        "tool_call_id": record.tool_call_id,
        "rule_compression_type": record.content_type,
        "rule_compression_pass": record.pass_name,
        "original_chars": record.original.chars().count(),
        "compressed_chars": record.compressed.chars().count(),
        "details": record.details.cloned().unwrap_or(Value::Null),
        "original_content": record.original,
        "compressed_content": record.compressed,
    });
    let serialized = serde_json::to_vec_pretty(&payload)
        .map_err(|error| Error::SysOp(format!("serialize dump payload: {error}")))?;
    std::fs::write(&path, serialized)
        .map_err(|error| Error::SysOp(format!("write dump {}: {error}", path.display())))?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filename_parts_are_sanitized() {
        assert_eq!(safe_filename_part("call/abc:1"), "call-abc-1");
        assert_eq!(safe_filename_part("--"), UNKNOWN);
        assert_eq!(
            dump_file_name(1.5, "add", "JSON_ARRAY", "call_1"),
            "1500_add_JSON_ARRAY_call_1.json"
        );
    }

    #[test]
    fn dumps_round_trip_through_json() {
        let dir = tempfile::tempdir().expect("temp dir");
        let record = DumpRecord {
            timestamp: 1.5,
            session_id: "session",
            context_id: "context",
            tool_call_id: "call_1",
            content_type: "LOG",
            pass_name: "add",
            original: "before",
            compressed: "after",
            details: None,
        };
        let path = write_dump(&record, dir.path()).expect("dump");
        let text = std::fs::read_to_string(path).expect("read");
        let parsed: Value = serde_json::from_str(&text).expect("json");
        assert_eq!(parsed["original_chars"], 6);
        assert_eq!(parsed["compressed_chars"], 5);
        assert_eq!(parsed["rule_compression_type"], "LOG");
    }
}
