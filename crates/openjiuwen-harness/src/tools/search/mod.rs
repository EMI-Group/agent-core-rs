//! Port of `openjiuwen/harness/tools/filesystem.py`: the `glob`, `list_files` and `grep` tools.
//!
//! `glob` delegates to [`FsOperation::search_files`] and `list_files` to
//! [`FsOperation::list_files`] / [`FsOperation::list_directories`], so both reuse the
//! sys-operation sandbox and keep the Python payload shapes. `grep` runs an in-process
//! scanner (`walkdir` + `regex`) that emits ripgrep-shaped output, so it needs neither the
//! `rg` binary nor a shell.
//!
//! Deviations from Python, all deliberate:
//!
//! - `grep` never shells out: `--type` filters and multiline search report the Python
//!   "requires ripgrep (rg)" failures, and the Windows `Select-String` fallback is not
//!   ported.
//! - `grep` skips a file whose first [`BINARY_SNIFF_BYTES`] contain a NUL byte; ripgrep's
//!   `--max-columns` preview and colours are not reproduced.
//! - `glob` de-duplicates by absolute path and relativizes the returned names against the
//!   resolved base; the Python symlink normalization of `os.path.relpath` is replaced by a
//!   canonicalized-base retry.

// Python parity: the grep helpers report expected failures as `ToolPayload`, and the public
// signature is part of the tool contract. Boxing the error would change every caller, so the
// size/performance heuristic does not apply here.
#![allow(clippy::result_large_err)]

mod glob;
mod grep;
mod list;
#[cfg(test)]
mod tests;

use std::path::Path;

use serde::{Deserialize, Deserializer};
use serde_json::Value;

pub use self::glob::{GlobArgs, GlobTool};
pub use self::grep::{GrepArgs, GrepOutputMode, GrepTool};
pub use self::list::{ListDirArgs, ListDirTool};

/// Python message for a document search without a pattern.
const MISSING_PATTERN: &str = "pattern is required";

/// Python message for a directory search without a filesystem backend.
const MISSING_FS: &str = "filesystem operation is not configured";

// ---------------------------------------------------------------------------------------
// helpers shared by more than one search tool
// ---------------------------------------------------------------------------------------

/// Reads the strings of a `name` array in a payload `data` object.
fn string_list(data: &Value, name: &str) -> Vec<String> {
    data.get(name)
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default()
}

/// Renders `path` relative to `root` with `/` separators, falling back to the absolute path.
fn relative_name(root: &Path, path: &Path) -> String {
    let relative = path
        .strip_prefix(root)
        .ok()
        .map(Path::to_path_buf)
        .or_else(|| {
            std::fs::canonicalize(root)
                .ok()
                .and_then(|root| path.strip_prefix(&root).ok().map(Path::to_path_buf))
        });
    relative.map_or_else(
        || path.to_string_lossy().replace('\\', "/"),
        |relative| relative.to_string_lossy().replace('\\', "/"),
    )
}

/// Deserializes a flag leniently; a missing, `null` or unrecognized value is `false`.
fn deserialize_flag<'de, D>(deserializer: D) -> std::result::Result<bool, D::Error>
where
    D: Deserializer<'de>,
{
    let value = Option::<Value>::deserialize(deserializer)?;
    Ok(value.as_ref().and_then(json_flag).unwrap_or(false))
}

/// Reads a boolean out of the Python truthy spellings (`true`, `1`, `"yes"`, `"on"`).
fn json_flag(value: &Value) -> Option<bool> {
    match value {
        Value::Bool(flag) => Some(*flag),
        Value::Number(number) => Some(number.as_f64().is_some_and(|number| number != 0.0)),
        Value::String(text) => match text.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Some(true),
            "0" | "false" | "no" | "off" => Some(false),
            _ => None,
        },
        _ => None,
    }
}
