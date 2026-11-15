//! Debug dumps of one rule-compression application, and the reader that replays them.
//!
//! Ports `RuleCompressionPipeline._dump_result`/`_dump_dir`/`_expand_dump_dir_template`.
//! Two formats are written, both carrying the same record:
//!
//! - [`RuleCompressionDumpFormat::Json`] (the default) writes one pretty-printed
//!   object per application, named `{epoch_ms}_{pass}_{type}_{tool_call_id}.json`,
//! - [`RuleCompressionDumpFormat::Jsonl`] appends one compact object per line to
//!   `{session_id or "session"}-{context_id or "context"}.jsonl`, so a whole
//!   conversation accumulates in a single file per session and context.
//!
//! Both forms are read back by [`read_dump_file`]/[`read_dump_dir`] (see
//! [`jsonl`](self) for the reader), so a dump is a durable record rather than
//! write-only forensics. A malformed record is reported as
//! [`Error::Validation`](openjiuwen_common::error::Error) naming the
//! file and the one-based line number; lines are never silently dropped.
//!
//! The directory is the configured value, else `OPENJIUWEN_RULE_COMPRESSION_DUMP_DIR`,
//! else the relative workspace default [`DEFAULT_DUMP_DIR`]. Every candidate
//! expands the `{session_id}`, `{context_id}`, `{date}` and `{timestamp}`
//! templates (the last two are the UTC calendar date as `YYYYMMDD`), and the
//! session/context values are sanitized exactly like [`safe_filename_part`].
//!
//! Dumping is a debugging aid: the pipeline warns once per directory, naming the
//! path and the error, and keeps the compressed message when a write fails.
use std::path::{Path, PathBuf};

use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::offload::OffloadScope;

mod jsonl;

pub use jsonl::{append_jsonl, read_dump_dir, read_dump_file, read_dump_text};

/// Environment variable that names the dump directory.
pub const RULE_COMPRESSION_DUMP_DIR_ENV: &str = "OPENJIUWEN_RULE_COMPRESSION_DUMP_DIR";

/// Relative directory dumps go to when nothing is configured.
///
/// The Python reference falls back into the `ModelContext` workspace
/// (`<workspace>/context/<session>_context/rule_compression_logs`); the Rust
/// pipeline has no workspace handle, so it uses this documented, git-ignorable
/// path under the process working directory instead.
pub const DEFAULT_DUMP_DIR: &str = ".openjiuwen/rule_compression_dumps";

/// Session id used when the scope carries none.
pub const UNKNOWN_SESSION_ID: &str = "unknown_session";

/// Context id used when the scope carries none.
pub const UNKNOWN_CONTEXT_ID: &str = "unknown_context";

/// Placeholder used when a dump filename part has no value.
const UNKNOWN: &str = "unknown";

/// Longest filename component kept from one sanitized part.
const MAX_FILENAME_PART_CHARS: usize = 80;

/// Seconds in one day, used by the `{date}`/`{timestamp}` template expansion.
const SECONDS_PER_DAY: f64 = 86_400.0;

/// How a dump is written.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuleCompressionDumpFormat {
    /// One pretty-printed JSON object per application, in its own file.
    #[default]
    Json,
    /// One compact JSON object per line, appended to one file per session/context.
    Jsonl,
}

impl RuleCompressionDumpFormat {
    /// The wire spelling (`json` / `jsonl`).
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Json => "json",
            Self::Jsonl => "jsonl",
        }
    }

    /// Parses a wire spelling, rejecting anything else with [`Error::Config`].
    ///
    /// `serde` already refuses an unknown variant when a configuration is
    /// deserialized; this is the typed-error path for callers that hold the
    /// spelling as a string.
    pub fn parse(value: &str) -> Result<Self> {
        match value.trim() {
            "json" => Ok(Self::Json),
            "jsonl" => Ok(Self::Jsonl),
            other => Err(Error::config(format!(
                "unknown rule-compression dump format {other:?}; expected \"json\" or \"jsonl\""
            ))),
        }
    }
}

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
    /// Whether the accepted candidate dropped information.
    pub lossy: bool,
    /// Content before compression.
    pub original: &'a str,
    /// Content after compression.
    pub compressed: &'a str,
    /// The compressor's details payload.
    pub details: Option<&'a Value>,
}

/// One dumped compression as read back from disk.
///
/// The shape matches what [`write_dump`] and [`append_jsonl`] write, plus the
/// `path` of the file the record came from.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct RuleCompressionDumpRecord {
    /// Wall-clock seconds of the compression.
    pub timestamp: f64,
    /// Session the compressed message belongs to.
    pub session_id: String,
    /// Context the compressed message belongs to.
    pub context_id: String,
    /// Tool call answered by the compressed message.
    pub tool_call_id: String,
    /// Detected content type (`JSON_ARRAY`, `GIT_DIFF`, …).
    pub rule_compression_type: String,
    /// Pipeline pass (`add` or `ttl`).
    pub rule_compression_pass: String,
    /// Characters before compression.
    pub original_chars: usize,
    /// Characters after compression.
    pub compressed_chars: usize,
    /// Whether the accepted candidate dropped information.
    pub lossy: bool,
    /// The compressor's details payload, [`Value::Null`] when it reported none.
    pub details: Value,
    /// Content before compression.
    pub original_content: String,
    /// Content after compression.
    pub compressed_content: String,
    /// File the record was read from.
    pub path: PathBuf,
}

/// Values a dump directory template can interpolate.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct DumpTemplateVars {
    /// Owning session, empty when unknown.
    pub session_id: String,
    /// Owning context, empty when unknown.
    pub context_id: String,
    /// Wall-clock seconds the `{date}`/`{timestamp}` templates expand from.
    pub timestamp: f64,
}

impl DumpTemplateVars {
    /// Template values for a session and context at `timestamp`.
    #[must_use]
    pub fn new(
        session_id: impl Into<String>,
        context_id: impl Into<String>,
        timestamp: f64,
    ) -> Self {
        Self {
            session_id: session_id.into(),
            context_id: context_id.into(),
            timestamp,
        }
    }

    /// Template values for an offload scope at `timestamp`.
    #[must_use]
    pub fn from_scope(scope: &OffloadScope, timestamp: f64) -> Self {
        Self::new(
            scope.session_id.clone(),
            scope.context_id.clone(),
            timestamp,
        )
    }

    /// The session id, or [`UNKNOWN_SESSION_ID`] when the scope carries none.
    #[must_use]
    pub fn session_or_unknown(&self) -> &str {
        or_unknown(&self.session_id, UNKNOWN_SESSION_ID)
    }

    /// The context id, or [`UNKNOWN_CONTEXT_ID`] when the scope carries none.
    #[must_use]
    pub fn context_or_unknown(&self) -> &str {
        or_unknown(&self.context_id, UNKNOWN_CONTEXT_ID)
    }
}

/// `value` when it is non-empty after trimming, `fallback` otherwise.
fn or_unknown<'a>(value: &'a str, fallback: &'a str) -> &'a str {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        fallback
    } else {
        trimmed
    }
}

/// Sanitizes one filename component the way Python's `_safe_filename_part` does.
#[must_use]
pub fn safe_filename_part(value: &str) -> String {
    safe_filename_part_or(value, UNKNOWN)
}

/// Sanitizes `value`, falling back to `fallback` when nothing usable remains.
#[must_use]
pub fn safe_filename_part_or(value: &str, fallback: &str) -> String {
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
        fallback.to_owned()
    } else {
        limited
    }
}

/// The dump file name for one compression in the JSON format.
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

/// The JSONL dump file name of one session/context conversation.
///
/// The parts fall back to `"session"`/`"context"` rather than `"unknown"`, so a
/// scopeless pipeline still accumulates into one inspectable file.
#[must_use]
pub fn jsonl_file_name(session_id: &str, context_id: &str) -> String {
    format!(
        "{}-{}.jsonl",
        safe_filename_part_or(or_unknown(session_id, ""), "session"),
        safe_filename_part_or(or_unknown(context_id, ""), "context"),
    )
}

/// Expands the `{session_id}`, `{context_id}`, `{date}` and `{timestamp}`
/// templates of `template`.
///
/// The date templates are replaced first, so a session id that happens to
/// contain the literal `{date}` cannot re-trigger expansion.
#[must_use]
pub fn expand_dump_dir_template(template: &str, vars: &DumpTemplateVars) -> String {
    let date = utc_date_stamp(vars.timestamp);
    let expanded = template
        .replace("{date}", &date)
        .replace("{timestamp}", &date)
        .replace(
            "{session_id}",
            &safe_filename_part(vars.session_or_unknown()),
        )
        .replace(
            "{context_id}",
            &safe_filename_part(vars.context_or_unknown()),
        );
    expanded
}

/// The UTC calendar date of `timestamp` (epoch seconds) as `YYYYMMDD`.
///
/// A clock before the epoch, or a non-finite value, maps to `19700101`.
#[must_use]
pub fn utc_date_stamp(timestamp: f64) -> String {
    let seconds = if timestamp.is_finite() {
        timestamp.max(0.0)
    } else {
        0.0
    };
    let days = (seconds / SECONDS_PER_DAY).floor() as i64;
    let (year, month, day) = civil_from_days(days);
    format!("{year:04}{month:02}{day:02}")
}

/// Converts days since 1970-01-01 to a proleptic Gregorian `(year, month, day)`.
///
/// This is Howard Hinnant's `civil_from_days`, so the crate needs no calendar
/// dependency.
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let shifted = days + 719_468;
    let era = shifted.div_euclid(146_097);
    let day_of_era = shifted.rem_euclid(146_097);
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_index = (5 * day_of_year + 2) / 153;
    let day = (day_of_year - (153 * month_index + 2) / 5 + 1) as u32;
    let month = if month_index < 10 {
        month_index + 3
    } else {
        month_index - 9
    } as u32;
    (if month <= 2 { year + 1 } else { year }, month, day)
}

/// The directory dumps go to.
///
/// Resolution order: the configured value, the environment variable, then
/// [`DEFAULT_DUMP_DIR`]. Every candidate is template-expanded, so an absolute
/// `dir` and the relative default behave identically; the last one always
/// applies, so the result is always a directory to write into.
#[must_use]
pub fn resolve_dump_dir(configured: Option<&str>, vars: &DumpTemplateVars) -> PathBuf {
    if let Some(template) = configured.map(str::trim).filter(|value| !value.is_empty()) {
        return PathBuf::from(expand_dump_dir_template(template, vars));
    }
    let from_env = std::env::var(RULE_COMPRESSION_DUMP_DIR_ENV).ok();
    if let Some(template) = from_env
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        return PathBuf::from(expand_dump_dir_template(template, vars));
    }
    PathBuf::from(expand_dump_dir_template(DEFAULT_DUMP_DIR, vars))
}

/// The JSON payload of one dump, shared by both formats.
fn dump_payload(record: &DumpRecord<'_>) -> Value {
    json!({
        "timestamp": record.timestamp,
        "session_id": record.session_id,
        "context_id": record.context_id,
        "tool_call_id": record.tool_call_id,
        "rule_compression_type": record.content_type,
        "rule_compression_pass": record.pass_name,
        "original_chars": record.original.chars().count(),
        "compressed_chars": record.compressed.chars().count(),
        "lossy": record.lossy,
        "details": record.details.cloned().unwrap_or(Value::Null),
        "original_content": record.original,
        "compressed_content": record.compressed,
    })
}

/// Writes `record` into `dir` in the format of `format`, returning the file path.
///
/// The directory is created when missing; in the JSON format an existing dump is
/// overwritten, which only happens when two compressions of the same tool call
/// land in the same millisecond.
pub fn write_record(
    record: &DumpRecord<'_>,
    dir: &Path,
    format: RuleCompressionDumpFormat,
) -> Result<PathBuf> {
    match format {
        RuleCompressionDumpFormat::Json => write_dump(record, dir),
        RuleCompressionDumpFormat::Jsonl => append_jsonl(record, dir),
    }
}

/// Writes `record` as one pretty-printed JSON file in `dir`, returning its path.
pub fn write_dump(record: &DumpRecord<'_>, dir: &Path) -> Result<PathBuf> {
    create_dir(dir)?;
    let path = dir.join(dump_file_name(
        record.timestamp,
        record.pass_name,
        record.content_type,
        record.tool_call_id,
    ));
    let serialized = serde_json::to_vec_pretty(&dump_payload(record))
        .map_err(|error| Error::sys_op(format!("serialize dump payload: {error}")))?;
    std::fs::write(&path, serialized)
        .map_err(|error| Error::sys_op(format!("write dump {}: {error}", path.display())))?;
    Ok(path)
}

/// Creates `dir`, attributing a failure to the directory that could not be made.
fn create_dir(dir: &Path) -> Result<()> {
    std::fs::create_dir_all(dir)
        .map_err(|error| Error::sys_op(format!("create dump dir {}: {error}", dir.display())))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vars() -> DumpTemplateVars {
        // 2024-02-29T12:00:00Z
        DumpTemplateVars::new("session/one", "ctx:one", 1_709_208_000.0)
    }

    #[test]
    fn filename_parts_are_sanitized() {
        assert_eq!(safe_filename_part("call/abc:1"), "call-abc-1");
        assert_eq!(safe_filename_part("--"), UNKNOWN);
        assert_eq!(
            dump_file_name(1.5, "add", "JSON_ARRAY", "call_1"),
            "1500_add_JSON_ARRAY_call_1.json"
        );
        assert_eq!(jsonl_file_name("", ""), "session-context.jsonl");
        assert_eq!(jsonl_file_name("s 1", "c/2"), "s-1-c-2.jsonl");
    }

    #[test]
    fn templates_expand_ids_and_the_utc_date() {
        let expanded = expand_dump_dir_template("/tmp/{date}/{session_id}/{context_id}", &vars());
        assert_eq!(expanded, "/tmp/20240229/session-one/ctx-one");
        assert_eq!(
            expand_dump_dir_template("/tmp/{timestamp}", &vars()),
            "/tmp/20240229"
        );
        let unknown = DumpTemplateVars::new("", "", 0.0);
        assert_eq!(
            expand_dump_dir_template("/tmp/{session_id}/{context_id}", &unknown),
            "/tmp/unknown_session/unknown_context"
        );
    }

    #[test]
    fn dates_convert_from_epoch_seconds() {
        assert_eq!(utc_date_stamp(0.0), "19700101");
        assert_eq!(utc_date_stamp(1_709_208_000.0), "20240229");
        assert_eq!(utc_date_stamp(f64::NAN), "19700101");
        assert_eq!(utc_date_stamp(-10.0), "19700101");
    }

    #[test]
    fn the_configured_directory_wins_and_the_default_always_applies() {
        let vars = vars();
        assert_eq!(
            resolve_dump_dir(Some("/tmp/dumps/{session_id}"), &vars),
            PathBuf::from("/tmp/dumps/session-one")
        );
        let resolved = resolve_dump_dir(Some("   "), &vars);
        assert!(!resolved.as_os_str().is_empty());
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
            lossy: true,
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
        assert_eq!(parsed["lossy"], true);
    }

    #[test]
    fn the_format_spellings_are_parsed_strictly() {
        assert_eq!(
            RuleCompressionDumpFormat::parse("jsonl").expect("jsonl"),
            RuleCompressionDumpFormat::Jsonl
        );
        let error = RuleCompressionDumpFormat::parse("yaml").expect_err("unknown format");
        assert!(matches!(error, Error::Config(_)));
        assert_eq!(RuleCompressionDumpFormat::default().as_str(), "json");
    }
}
