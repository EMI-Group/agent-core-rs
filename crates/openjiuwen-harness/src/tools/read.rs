//! Port of `openjiuwen/harness/tools/filesystem.py`: the `read_file` tool.
//!
//! The tool reads a window of a text file, renders it with `cat -n` style line numbers and
//! records the observed lines in the read-before-write registry so `write_file`/`edit_file`
//! can refuse to clobber content the model has never seen.

// The module stays private until `ToolRegistry::build` wires the tool set; keeping the
// intended public surface (`pub`) documents the eventual export and avoids a churn diff.
#![allow(unreachable_pub)]

use std::sync::Arc;

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use openjiuwen_common::error::Result;
use openjiuwen_foundation::tool::{Tool, ToolCard, ToolContext};
use openjiuwen_sysop::fs::types::{FileContent, LineRange, ReadFileParams, ReadMode};
use openjiuwen_sysop::operation::FsOperation;

use super::descriptions;
use super::read_state::{file_fingerprint, ReadCoverage, ReadStateRegistry};
use super::support::{parse_args, schema_for, tool_card, ToolEnv, ToolPayload};

/// Maximum number of lines a single read returns (`ReadFileTool.MAX_LINES_TO_READ`).
const MAX_LINES_TO_READ: usize = 2000;

/// Maximum file size, in bytes, accepted by a read without an explicit `limit`.
const MAX_SIZE_BYTES: u64 = 256 * 1024;

/// The warning returned when a whole-file read finds no content.
const EMPTY_FILE_WARNING: &str = "Warning: the file exists but the contents are empty.";

/// Arguments of the `read_file` tool.
///
/// Python parity: the `file_path`/`offset`/`limit`/`pages`/`caption` parameters of
/// `ReadFileTool.invoke`. `pages` and `caption` are accepted and ignored because the PDF and
/// image readers are deferred.
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct ReadFileArgs {
    /// The absolute path to the file to read.
    pub file_path: String,
    /// The line number to start reading from. Only provide if the file is too large to read at once or you already know the needed location.
    #[serde(default)]
    #[schemars(with = "u64")]
    pub offset: Option<u64>,
    /// The number of lines to read. Only provide if the file is too large to read at once or only a specific range is needed.
    #[serde(default)]
    #[schemars(with = "u64")]
    pub limit: Option<u64>,
    /// PDF-only page range, e.g. '1-5', '3', '10-20'. Maximum 20 pages per request
    #[serde(default)]
    #[schemars(with = "String")]
    pub pages: Option<String>,
    /// Image caption request; accepted and ignored (image reads are deferred).
    #[serde(default)]
    #[schemars(with = "String")]
    pub caption: Option<String>,
}

/// Reads a text file, one window at a time, with line numbers.
pub struct ReadFileTool {
    /// Path resolution, permission gate and description language.
    env: ToolEnv,
    /// The filesystem backend, when one is configured.
    fs: Option<Arc<dyn FsOperation>>,
    /// Read-before-write bookkeeping shared with the other filesystem tools.
    state: Arc<ReadStateRegistry>,
    /// The model-facing card built from [`ToolEnv::language`].
    card: ToolCard,
}

impl ReadFileTool {
    /// Builds the tool over `env`, reading through `fs` and recording into `state`.
    #[must_use]
    pub fn new(
        env: ToolEnv,
        fs: Option<Arc<dyn FsOperation>>,
        state: Arc<ReadStateRegistry>,
    ) -> Self {
        let card = Self::card_for(env.language());
        Self {
            env,
            fs,
            state,
            card,
        }
    }

    /// Builds the model-facing card in `language`.
    ///
    /// Unknown languages fall back to the Python default (`cn`) inside
    /// [`descriptions::description`].
    #[must_use]
    pub fn card_for(language: &str) -> ToolCard {
        tool_card(
            "read_file",
            descriptions::description("read_file", language),
            schema_for::<ReadFileArgs>(),
            true,
        )
    }

    /// Renders `text` with 1-based, six-column line numbers (`cat -n` / Python `_cat_n`).
    ///
    /// A trailing newline does not add a phantom line, and the empty string stays empty.
    #[must_use]
    pub fn number_lines(text: &str) -> String {
        if text.is_empty() {
            return String::new();
        }
        text.lines()
            .enumerate()
            .map(|(index, line)| format!("{:>6}\t{line}", index + 1))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Reads the requested window and returns the tool payload.
    async fn read(&self, inputs: &Value) -> ToolPayload {
        // Python reads `file_path` from the raw arguments and reports its absence with the
        // plain validation text rather than a schema error, so check it before parsing.
        if inputs.is_object()
            && inputs
                .get("file_path")
                .and_then(Value::as_str)
                .is_none_or(str::is_empty)
        {
            return ToolPayload::fail("file_path is required");
        }

        let args: ReadFileArgs = match parse_args(inputs) {
            Ok(args) => args,
            Err(payload) => return payload,
        };
        if args.file_path.is_empty() {
            return ToolPayload::fail("file_path is required");
        }

        let resolved = match self
            .env
            .resolve_path("read_file", "file_path", &args.file_path, inputs)
            .await
        {
            Ok(path) => path,
            Err(payload) => return payload,
        };

        let Some(fs) = self.fs.as_ref() else {
            return ToolPayload::fail("filesystem operation is not configured");
        };

        let user_supplied_limit = args.limit.is_some();
        let offset = usize::try_from(args.offset.unwrap_or(0)).unwrap_or(usize::MAX);
        let max_lines = u64::try_from(MAX_LINES_TO_READ).unwrap_or(u64::MAX);
        let limit = usize::try_from(args.limit.unwrap_or(max_lines).clamp(1, max_lines))
            .unwrap_or(MAX_LINES_TO_READ)
            .clamp(1, MAX_LINES_TO_READ);

        let params = ReadFileParams {
            path: resolved.to_string_lossy().into_owned(),
            mode: ReadMode::Text,
            line_range: Some(LineRange {
                start: offset.saturating_add(1),
                end: offset.saturating_add(limit),
            }),
            ..ReadFileParams::default()
        };

        let result = fs.read_file(params).await;
        if result.is_failure() {
            return ToolPayload::fail(result.message);
        }
        let Some(data) = result.data else {
            return ToolPayload::fail("read_file returned no data from the filesystem operation");
        };

        let total_bytes = data.total_bytes;
        let raw_text = match data.content {
            FileContent::Text(text) => text,
            FileContent::Bytes(bytes) => String::from_utf8_lossy(&bytes).into_owned(),
        };

        if !user_supplied_limit && total_bytes > MAX_SIZE_BYTES {
            return ToolPayload::fail(oversized_message(total_bytes));
        }

        let (displayed, line_count) = if raw_text.trim().is_empty() {
            let content = if offset == 0 {
                EMPTY_FILE_WARNING.to_owned()
            } else {
                format!(
                    "Error: The file exists but no content was returned at offset {offset}. Use a smaller offset or read from the beginning."
                )
            };
            (content, 0)
        } else {
            let displayed = Self::number_lines(&raw_text);
            let count = displayed.lines().count();
            (displayed, count)
        };

        let payload = ToolPayload::ok(json!({
            "content": displayed,
            "file_path": resolved.to_string_lossy(),
            "line_count": line_count,
        }));

        // A metadata failure only skips registration: the read itself already succeeded.
        match tokio::fs::metadata(&resolved).await {
            Ok(metadata) => {
                let coverage = (line_count > 0).then(|| ReadCoverage {
                    start: offset.saturating_add(1),
                    end: offset.saturating_add(line_count),
                });
                self.state.record_read(
                    &resolved,
                    file_fingerprint(&metadata),
                    coverage,
                    Some(raw_text),
                );
            }
            Err(error) => {
                tracing::debug!(
                    path = %resolved.display(),
                    %error,
                    "skipping read-state registration: metadata unavailable"
                );
            }
        }

        payload
    }
}

/// Builds the Python "file too large" message for a `total_bytes` file.
#[allow(clippy::cast_precision_loss)]
fn oversized_message(total_bytes: u64) -> String {
    format!(
        "Error: File content ({:.1}KB) exceeds maximum allowed size ({}KB). Use grep for specific content or read a smaller offset/limit range.",
        total_bytes as f64 / 1024.0,
        MAX_SIZE_BYTES / 1024
    )
}

#[async_trait]
impl Tool for ReadFileTool {
    fn card(&self) -> &ToolCard {
        &self.card
    }

    async fn invoke(&self, inputs: Value, ctx: &ToolContext) -> Result<Value> {
        ctx.check_cancelled()?;
        Ok(self.read(&inputs).await.to_json())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::sync::Arc;

    use serde_json::{json, Value};
    use tempfile::TempDir;

    use super::*;
    use crate::tools::read_state::WriteGate;
    use crate::workspace::Workspace;
    use openjiuwen_sysop::{LocalWorkConfig, SysOperation};

    /// Builds a tool over a real local backend rooted at a fresh temp directory.
    fn setup() -> (ReadFileTool, Arc<ReadStateRegistry>, TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let operation = SysOperation::local(LocalWorkConfig::unrestricted(
            dir.path().to_string_lossy().to_string(),
        ))
        .unwrap();
        let fs = operation.fs();
        let state = ReadStateRegistry::new();
        let tool = ReadFileTool::new(
            ToolEnv::new(Workspace::new(dir.path())),
            fs,
            Arc::clone(&state),
        );
        (tool, state, dir)
    }

    /// Invokes the tool with a fresh context and unwraps the payload envelope.
    async fn invoke(tool: &ReadFileTool, inputs: Value) -> Value {
        tool.invoke(inputs, &ToolContext::new()).await.unwrap()
    }

    #[test]
    fn card_exposes_python_arguments() {
        let card = ReadFileTool::card_for("en");
        assert_eq!(card.name, "read_file");
        assert!(card.parallel_safe);
        assert_eq!(card.input_params["required"], json!(["file_path"]));

        let properties = card.input_params["properties"].as_object().unwrap();
        let mut keys: Vec<&str> = properties.keys().map(String::as_str).collect();
        keys.sort_unstable();
        assert_eq!(
            keys,
            ["caption", "file_path", "limit", "offset", "pages"],
            "unexpected property set: {keys:?}"
        );
        assert_eq!(properties["offset"]["type"], json!("integer"));
        assert_eq!(properties["limit"]["type"], json!("integer"));
    }

    #[test]
    fn number_lines_formats_six_wide_numbers_and_tabs() {
        assert_eq!(
            ReadFileTool::number_lines("alpha\nbeta"),
            "     1\talpha\n     2\tbeta"
        );
        assert_eq!(ReadFileTool::number_lines(""), "");
    }

    #[tokio::test]
    async fn reads_a_five_line_file_with_line_numbers() {
        let (tool, _state, dir) = setup();
        std::fs::write(
            dir.path().join("notes.txt"),
            "one\ntwo\nthree\nfour\nfive\n",
        )
        .unwrap();

        let out = invoke(&tool, json!({"file_path": "notes.txt"})).await;

        assert_eq!(out["success"], json!(true));
        assert_eq!(
            out["data"]["content"],
            json!("     1\tone\n     2\ttwo\n     3\tthree\n     4\tfour\n     5\tfive")
        );
        assert_eq!(out["data"]["line_count"], json!(5));
        let path = out["data"]["file_path"].as_str().unwrap();
        assert!(path.ends_with("notes.txt"), "unexpected path: {path}");
        assert_eq!(out["error"], Value::Null);
    }

    #[tokio::test]
    async fn offset_and_limit_read_a_window_and_register_partial_coverage() {
        let (tool, state, dir) = setup();
        let full_content = "one\ntwo\nthree\nfour\nfive\n";
        let path = dir.path().join("notes.txt");
        std::fs::write(&path, full_content).unwrap();

        let out = invoke(
            &tool,
            json!({"file_path": "notes.txt", "offset": 2, "limit": 2}),
        )
        .await;

        assert_eq!(out["success"], json!(true));
        assert_eq!(out["data"]["content"], json!("     1\tthree\n     2\tfour"));
        assert_eq!(out["data"]["line_count"], json!(2));

        let fingerprint = file_fingerprint(&std::fs::metadata(&path).unwrap());
        assert_eq!(
            state.write_gate(&path, fingerprint, full_content),
            WriteGate::Partial {
                lines_read: 2,
                total_lines: 0,
                resume_offset: 0,
            }
        );
    }

    #[tokio::test]
    async fn missing_file_path_is_a_failure_payload() {
        let (tool, _state, _dir) = setup();

        for inputs in [json!({}), json!({"file_path": ""})] {
            let out = invoke(&tool, inputs).await;
            assert_eq!(out["success"], json!(false));
            assert_eq!(out["error"], json!("file_path is required"));
        }
    }

    #[tokio::test]
    async fn nonexistent_file_returns_failure_without_panicking() {
        let (tool, _state, _dir) = setup();

        let out = invoke(&tool, json!({"file_path": "missing.txt"})).await;

        assert_eq!(out["success"], json!(false));
        let error = out["error"].as_str().unwrap();
        assert!(!error.is_empty(), "expected a failure message");
    }

    #[tokio::test]
    async fn oversized_file_requires_an_explicit_limit() {
        let (tool, _state, dir) = setup();
        std::fs::write(dir.path().join("big.txt"), "x".repeat(300 * 1024)).unwrap();

        let out = invoke(&tool, json!({"file_path": "big.txt"})).await;
        assert_eq!(out["success"], json!(false));
        let error = out["error"].as_str().unwrap();
        assert!(
            error.contains("exceeds maximum allowed size (256KB)"),
            "unexpected error: {error}"
        );

        let limited = invoke(&tool, json!({"file_path": "big.txt", "limit": 10})).await;
        assert_eq!(limited["success"], json!(true));
    }

    #[tokio::test]
    async fn empty_file_returns_the_python_warning() {
        let (tool, _state, dir) = setup();
        std::fs::write(dir.path().join("empty.txt"), "").unwrap();

        let out = invoke(&tool, json!({"file_path": "empty.txt"})).await;

        assert_eq!(out["success"], json!(true));
        assert_eq!(
            out["data"]["content"],
            json!("Warning: the file exists but the contents are empty.")
        );
        assert_eq!(out["data"]["line_count"], json!(0));
    }

    #[tokio::test]
    async fn offset_past_end_of_file_returns_python_message() {
        let (tool, _state, dir) = setup();
        std::fs::write(dir.path().join("short.txt"), "a\nb\nc\n").unwrap();

        let out = invoke(&tool, json!({"file_path": "short.txt", "offset": 99})).await;

        assert_eq!(out["success"], json!(true));
        assert_eq!(
            out["data"]["content"],
            json!("Error: The file exists but no content was returned at offset 99. Use a smaller offset or read from the beginning.")
        );
        assert_eq!(out["data"]["line_count"], json!(0));
    }

    #[tokio::test]
    async fn full_read_is_visible_through_the_write_gate() {
        let (tool, state, dir) = setup();
        let full_content = "one\ntwo\nthree\nfour\nfive\n";
        let path = dir.path().join("notes.txt");
        std::fs::write(&path, full_content).unwrap();

        let out = invoke(&tool, json!({"file_path": "notes.txt"})).await;
        assert_eq!(out["success"], json!(true));

        let fingerprint = file_fingerprint(&std::fs::metadata(&path).unwrap());
        assert_eq!(
            state.write_gate(&path, fingerprint, full_content),
            WriteGate::Ok
        );
    }

    #[tokio::test]
    async fn missing_filesystem_backend_is_a_failure_payload() {
        let dir = tempfile::tempdir().unwrap();
        let tool = ReadFileTool::new(
            ToolEnv::new(Workspace::new(dir.path())),
            None,
            ReadStateRegistry::new(),
        );

        let out = invoke(&tool, json!({"file_path": "a.txt"})).await;

        assert_eq!(out["success"], json!(false));
        assert_eq!(
            out["error"],
            json!("filesystem operation is not configured")
        );
    }
}
