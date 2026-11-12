//! Port of `openjiuwen/harness/tools/filesystem.py`: the `write_file` and `edit_file` tools.
//!
//! `write_file` replaces a whole file, so it refuses to run until the model has read every
//! line of the existing content. `edit_file` replaces one exact string and writes the result
//! back with a compare-and-swap on the bytes it read, so a concurrent change is never silently
//! clobbered. Both tools record the content they wrote, which lets a later write or edit skip
//! the read-before-write round trip.
//!
//! Deviation from Python: the `.agent_history` operation log, the `.ipynb` guard, quote
//! tolerance, XML desanitization, trailing-whitespace stripping and the scripted "similar
//! paths" hint are not ported, and `original_file` is always `null` in the write payload.

// The module stays private until `ToolRegistry::build` wires the tool set; keeping the
// intended public surface (`pub`) documents the eventual export and avoids a churn diff.
#![allow(unreachable_pub)]

use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use openjiuwen_common::error::Result;
use openjiuwen_foundation::tool::{
    render_payload_text, render_tool_output, Tool, ToolCard, ToolContext, ToolOutput,
};
use openjiuwen_sysop::fs::types::{FileContent, WriteFileParams};
use openjiuwen_sysop::operation::FsOperation;
use openjiuwen_sysop::util::sha256_hex;
use openjiuwen_sysop::SysOpStatus;

use super::descriptions;
use super::read_state::{file_fingerprint, EditGate, ReadStateRegistry, WriteGate};
use super::support::{parse_args, schema_for, tool_card, ToolEnv, ToolPayload};

/// Largest file `edit_file` will modify, in bytes (`EditFileTool.MAX_FILE_SIZE`).
const MAX_FILE_SIZE: u64 = 1024 * 1024 * 1024;

/// Python's refusal when a create would overwrite content the model never saw.
const CREATE_CONFLICT: &str = "Cannot create new file - file already exists.";

/// Python's refusal of a no-op edit.
const IDENTICAL_STRINGS: &str =
    "old_string and new_string are identical; no changes would be made.";

/// Message returned by both tools when no filesystem backend was wired in.
const NO_BACKEND: &str = "filesystem operation is not configured";

/// Arguments of the `write_file` tool.
///
/// Python parity: the `file_path` and `content` parameters of `WriteFileTool.invoke`.
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct WriteFileArgs {
    /// Absolute path of the file to write
    pub file_path: String,
    /// Content to write
    pub content: String,
}

/// Overwrites a whole file after checking that the model has read the old content.
pub struct WriteFileTool {
    /// Path resolution, permission gate and description language.
    env: ToolEnv,
    /// The filesystem backend, when one is configured.
    fs: Option<Arc<dyn FsOperation>>,
    /// Read-before-write bookkeeping shared with the other filesystem tools.
    state: Arc<ReadStateRegistry>,
    /// The model-facing card built from [`ToolEnv::language`].
    card: ToolCard,
}

impl WriteFileTool {
    /// Builds the tool over `env`, writing through `fs` and recording into `state`.
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
            "write_file",
            descriptions::description("write_file", language),
            schema_for::<WriteFileArgs>(),
            false,
        )
    }

    /// Validates the arguments, applies the read gate and overwrites the file.
    async fn write(&self, inputs: &Value) -> ToolPayload {
        // Python reports every argument problem with plain validation text rather than a
        // schema error, so the raw JSON is checked before serde sees it.
        if !inputs.is_object() {
            return ToolPayload::fail("tool arguments must be an object");
        }
        if inputs
            .get("file_path")
            .and_then(Value::as_str)
            .is_none_or(str::is_empty)
        {
            return ToolPayload::fail("file_path is required");
        }
        match inputs.get("content") {
            Some(Value::String(_)) => {}
            None | Some(Value::Null) => return ToolPayload::fail("content is required"),
            Some(_) => return ToolPayload::fail("content must be a string"),
        }
        let args: WriteFileArgs = match parse_args(inputs) {
            Ok(args) => args,
            Err(payload) => return payload,
        };

        let resolved = match self
            .env
            .resolve_path("write_file", "file_path", &args.file_path, inputs)
            .await
        {
            Ok(path) => path,
            Err(payload) => return payload,
        };
        let Some(fs) = self.fs.as_ref() else {
            return ToolPayload::fail(NO_BACKEND);
        };
        let display = resolved.to_string_lossy().into_owned();

        let existing = match tokio::fs::metadata(&resolved).await {
            Ok(metadata) if metadata.is_dir() => {
                return ToolPayload::fail(format!("Target path is a directory: {display}"));
            }
            Ok(metadata) => Some(metadata),
            Err(_) => None,
        };
        if let Some(metadata) = existing.as_ref() {
            let fingerprint = file_fingerprint(metadata);
            let current_text = match tokio::fs::read(&resolved).await {
                Ok(bytes) => String::from_utf8_lossy(&bytes).into_owned(),
                Err(error) => return ToolPayload::fail(error.to_string()),
            };
            let gate = self.state.write_gate(&resolved, fingerprint, &current_text);
            if gate == WriteGate::Stale {
                self.state.invalidate(&resolved);
            }
            if let Some(payload) = write_gate_failure(&gate, &display) {
                return payload;
            }
        }

        let created = existing.is_none();
        let bytes_written = args.content.len();
        // A written file is "observed" in its LF form, matching what `read_file` records.
        let recorded = args.content.replace("\r\n", "\n");
        let params = WriteFileParams {
            path: display.clone(),
            content: FileContent::Text(args.content),
            prepend_newline: false,
            append_newline: false,
            append: false,
            create_if_not_exist: true,
            permissions: None,
            ..WriteFileParams::default()
        };
        let result = fs.write_file(params).await;
        if result.is_failure() {
            return ToolPayload::fail(result.message);
        }

        record_written_file(&self.state, &resolved, recorded).await;
        ToolPayload::ok(json!({
            "file_path": display,
            "bytes_written": bytes_written,
            "type": if created { "create" } else { "update" },
            "created": created,
            "original_file": Value::Null,
        }))
    }
}

#[async_trait]
impl Tool for WriteFileTool {
    fn card(&self) -> &ToolCard {
        &self.card
    }

    async fn invoke(&self, inputs: Value, ctx: &ToolContext) -> Result<Value> {
        ctx.check_cancelled()?;
        Ok(self.write(&inputs).await.to_json())
    }

    fn render_for_llm(&self, output: &Value) -> String {
        let Some(data) = success_data(output) else {
            return render_default(output);
        };
        let action = if data.get("created").and_then(Value::as_bool) == Some(true) {
            "Created"
        } else {
            "Updated"
        };
        format!(
            "{action} {} ({} bytes written).",
            data.get("file_path")
                .and_then(Value::as_str)
                .unwrap_or_default(),
            data.get("bytes_written")
                .and_then(Value::as_u64)
                .unwrap_or_default()
        )
    }
}

/// Arguments of the `edit_file` tool.
///
/// Python parity: the `file_path`/`old_string`/`new_string`/`replace_all` parameters of
/// `EditFileTool.invoke`. `old_string` defaults to `""` so an omitted value selects the
/// create branch of the Python tool.
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct EditFileArgs {
    /// The absolute path to the file to edit
    pub file_path: String,
    /// The text to replace (empty string creates a new file or writes to an empty file).
    /// Must match exactly once unless `replace_all=true` or more context is provided
    #[serde(default)]
    pub old_string: String,
    /// The replacement text, must differ from `old_string`
    pub new_string: String,
    /// Replace all occurrences of `old_string` in the file, default false
    #[serde(default)]
    pub replace_all: bool,
}

/// Replaces one exact string in a file, creating the file when `old_string` is empty.
pub struct EditFileTool {
    /// Path resolution, permission gate and description language.
    env: ToolEnv,
    /// The filesystem backend, when one is configured.
    fs: Option<Arc<dyn FsOperation>>,
    /// Read-before-write bookkeeping shared with the other filesystem tools.
    state: Arc<ReadStateRegistry>,
    /// The model-facing card built from [`ToolEnv::language`].
    card: ToolCard,
}

impl EditFileTool {
    /// Builds the tool over `env`, writing through `fs` and recording into `state`.
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
        let mut input_params = schema_for::<EditFileArgs>();
        install_required_fields(&mut input_params);
        tool_card(
            "edit_file",
            descriptions::description("edit_file", language),
            input_params,
            false,
        )
    }

    /// Validates the arguments, takes the per-path lock and dispatches to a branch.
    async fn edit(&self, inputs: &Value) -> ToolPayload {
        // Python reports every argument problem with plain validation text rather than a
        // schema error, so the raw JSON is checked before serde sees it.
        if !inputs.is_object() {
            return ToolPayload::fail("tool arguments must be an object");
        }
        if inputs
            .get("file_path")
            .and_then(Value::as_str)
            .is_none_or(str::is_empty)
        {
            return ToolPayload::fail("file_path is required");
        }
        match inputs.get("new_string") {
            Some(Value::String(_)) => {}
            None | Some(Value::Null) => return ToolPayload::fail("new_string is required"),
            Some(_) => return ToolPayload::fail("new_string must be a string"),
        }
        match inputs.get("old_string") {
            None | Some(Value::Null | Value::String(_)) => {}
            Some(_) => return ToolPayload::fail("old_string must be a string"),
        }
        // Python reads `old_string` with a `""` default; serde's `#[serde(default)]` covers the
        // absent case, so only an explicit `null` has to be normalized first.
        let mut normalized = inputs.clone();
        if normalized.get("old_string").is_some_and(Value::is_null) {
            if let Value::Object(object) = &mut normalized {
                object.insert("old_string".to_owned(), Value::String(String::new()));
            }
        }
        let args: EditFileArgs = match parse_args(&normalized) {
            Ok(args) => args,
            Err(payload) => return payload,
        };

        // The no-op check runs before path resolution and before the create branch, as in
        // Python, so `old_string == new_string == ""` never creates a file.
        if args.old_string == args.new_string {
            return ToolPayload::fail(IDENTICAL_STRINGS);
        }

        let resolved = match self
            .env
            .resolve_path("edit_file", "file_path", &args.file_path, inputs)
            .await
        {
            Ok(path) => path,
            Err(payload) => return payload,
        };
        let Some(fs) = self.fs.as_ref() else {
            return ToolPayload::fail(NO_BACKEND);
        };
        let display = resolved.to_string_lossy().into_owned();
        // Python serializes the whole read-modify-write transaction per resolved path.
        let _guard = self.state.lock(&resolved).await;

        if args.old_string.is_empty() {
            return self
                .create_file(&resolved, &display, &args.new_string, fs.as_ref())
                .await;
        }
        self.replace(&resolved, &display, &args, fs.as_ref()).await
    }

    /// Python's new-file branch: `old_string` is empty, so `new_string` becomes the content.
    async fn create_file(
        &self,
        path: &Path,
        display: &str,
        new_string: &str,
        fs: &dyn FsOperation,
    ) -> ToolPayload {
        if tokio::fs::metadata(path).await.is_ok() {
            let raw = match tokio::fs::read(path).await {
                Ok(raw) => raw,
                Err(error) => return ToolPayload::fail(error.to_string()),
            };
            // A blank existing file may be replaced; anything else is a conflict.
            if !decode_text(&raw).trim().is_empty() {
                return ToolPayload::fail(CREATE_CONFLICT);
            }
        }

        let params = WriteFileParams {
            path: display.to_owned(),
            content: FileContent::Text(new_string.to_owned()),
            prepend_newline: false,
            create_if_not_exist: true,
            permissions: None,
            ..WriteFileParams::default()
        };
        let result = fs.write_file(params).await;
        if result.is_failure() {
            return ToolPayload::fail(format!("Create failed: {}", result.message));
        }

        let recorded = new_string.replace("\r\n", "\n");
        record_written_file(&self.state, path, recorded).await;
        ToolPayload::ok(json!({
            "file_path": display,
            "replacements": 0,
            "created": true,
        }))
    }

    /// Python's replacement branch: one exact, unique string is swapped under a CAS write.
    async fn replace(
        &self,
        path: &Path,
        display: &str,
        args: &EditFileArgs,
        fs: &dyn FsOperation,
    ) -> ToolPayload {
        let metadata = match tokio::fs::metadata(path).await {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return ToolPayload::fail(format!("Error: File not found: '{display}'."));
            }
            Err(error) => return ToolPayload::fail(error.to_string()),
        };
        if metadata.len() > MAX_FILE_SIZE {
            return ToolPayload::fail(format!(
                "Error: File is too large ({} GiB). Maximum allowed size is 1 GiB.",
                metadata.len() / MAX_FILE_SIZE
            ));
        }

        let fingerprint = file_fingerprint(&metadata);
        let raw = match tokio::fs::read(path).await {
            Ok(raw) => raw,
            Err(error) => return ToolPayload::fail(error.to_string()),
        };
        let eol = if raw.windows(2).any(|window| window == b"\r\n") {
            "\r\n"
        } else {
            "\n"
        };
        let current_lf = decode_text(&raw).replace("\r\n", "\n");

        let gate = self.state.edit_gate(path, fingerprint, &current_lf);
        if gate == EditGate::Stale {
            self.state.invalidate(path);
        }
        if let Some(payload) = edit_gate_failure(&gate, display) {
            return payload;
        }

        // Both sides are normalized to LF for matching; the file's own EOL is restored on write.
        let old_lf = args.old_string.replace("\r\n", "\n");
        let new_lf = args.new_string.replace("\r\n", "\n");
        let count = current_lf.matches(old_lf.as_str()).count();
        if count == 0 {
            return ToolPayload::fail(format!(
                "Error: old_string not found in '{display}'. Read the file and retry with the exact current text."
            ));
        }
        if count > 1 && !args.replace_all {
            return ToolPayload::fail(format!(
                "Error: old_string matches {count} times in '{display}'. Provide more surrounding context to make it unique, or set replace_all=true only if replacing every occurrence is intended."
            ));
        }
        let (edited_lf, replacements) = if args.replace_all {
            (current_lf.replace(old_lf.as_str(), new_lf.as_str()), count)
        } else {
            (current_lf.replacen(old_lf.as_str(), new_lf.as_str(), 1), 1)
        };
        let new_text = if eol == "\r\n" {
            edited_lf.replace('\n', "\r\n")
        } else {
            edited_lf.clone()
        };

        let params = WriteFileParams {
            path: display.to_owned(),
            content: FileContent::Text(new_text),
            prepend_newline: false,
            create_if_not_exist: true,
            permissions: None,
            expected_content_sha256: Some(sha256_hex(&raw)),
            ..WriteFileParams::default()
        };
        let result = fs.write_file(params).await;
        if result.is_failure() {
            if result.code == SysOpStatus::FileVersionConflict.code() {
                return ToolPayload::fail(
                    "File changed while preparing the edit. Re-read it before editing again.",
                );
            }
            return ToolPayload::fail(format!("Write failed: {}", result.message));
        }

        record_written_file(&self.state, path, edited_lf).await;
        ToolPayload::ok(json!({
            "file_path": display,
            "replacements": replacements,
        }))
    }
}

#[async_trait]
impl Tool for EditFileTool {
    fn card(&self) -> &ToolCard {
        &self.card
    }

    async fn invoke(&self, inputs: Value, ctx: &ToolContext) -> Result<Value> {
        ctx.check_cancelled()?;
        Ok(self.edit(&inputs).await.to_json())
    }

    fn render_for_llm(&self, output: &Value) -> String {
        let Some(data) = success_data(output) else {
            return render_default(output);
        };
        let file_path = data
            .get("file_path")
            .and_then(Value::as_str)
            .unwrap_or_default();
        // Creating a file through an empty `old_string` reads as a creation, not an edit.
        if data.get("created").and_then(Value::as_bool) == Some(true) {
            return format!("Created {file_path}.");
        }
        let replacements = data
            .get("replacements")
            .and_then(Value::as_u64)
            .unwrap_or_default();
        format!("Edited {file_path} ({replacements} replacement(s)).")
    }
}

/// Writes the Python `required` set into an [`EditFileArgs`] schema.
///
/// `old_string` and `replace_all` carry serde defaults so a partial call still parses, which
/// makes schemars drop them from `required`; the Python card nevertheless requires
/// `old_string`, so the list is installed verbatim.
fn install_required_fields(schema: &mut Value) {
    let Some(object) = schema.as_object_mut() else {
        return;
    };
    object.insert(
        "required".to_owned(),
        json!(["file_path", "old_string", "new_string"]),
    );
}

/// Maps a [`WriteGate`] verdict onto the Python failure payload, or `None` to proceed.
fn write_gate_failure(gate: &WriteGate, display: &str) -> Option<ToolPayload> {
    match gate {
        WriteGate::Ok => None,
        WriteGate::NotRead => Some(ToolPayload::fail(format!(
            "Error: File has not been read yet: {display}. Read it before attempting to write it."
        ))),
        WriteGate::Stale => Some(ToolPayload::fail(format!(
            "Error: File has been modified since read: {display}. Read it again before attempting to write it."
        ))),
        WriteGate::Partial {
            lines_read,
            total_lines,
            resume_offset,
        } => Some(ToolPayload::fail(format!(
            "Error: File has not been fully read yet. You have read {lines_read} of {total_lines} lines. Continue with read_file offset={resume_offset} until the whole file is covered, then write. A full-file overwrite cannot use partial reads — it would lose the unread content."
        ))),
    }
}

/// Maps an [`EditGate`] verdict onto the Python failure payload, or `None` to proceed.
fn edit_gate_failure(gate: &EditGate, display: &str) -> Option<ToolPayload> {
    match gate {
        EditGate::Ok => None,
        EditGate::NotRead => Some(ToolPayload::fail(format!(
            "Error: File must be read before editing: {display}. Read it before attempting to edit it."
        ))),
        EditGate::Stale => Some(ToolPayload::fail(format!(
            "Error: File has been modified since read: {display}. Read it again before attempting to edit it."
        ))),
    }
}

/// Decodes file bytes the way Python's `_detect_encoding` does.
///
/// A UTF-16 LE byte-order mark selects UTF-16 LE; everything else decodes as UTF-8 with
/// replacement characters, so a binary file never becomes a tool failure.
fn decode_text(raw: &[u8]) -> String {
    if raw.starts_with(&[0xff, 0xfe]) {
        let units: Vec<u16> = raw[2..]
            .chunks_exact(2)
            .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
            .collect();
        return String::from_utf16_lossy(&units);
    }
    String::from_utf8_lossy(raw).into_owned()
}

/// Records `content` as the observed content of a file the tool just wrote.
///
/// A metadata failure only skips the registration (logged at debug level): the write itself
/// already succeeded, so the tool must not report a failure.
async fn record_written_file(state: &ReadStateRegistry, path: &Path, content: String) {
    match tokio::fs::metadata(path).await {
        Ok(metadata) => state.record_written(path, file_fingerprint(&metadata), content),
        Err(error) => tracing::debug!(
            path = %path.display(),
            %error,
            "skipping write-state registration: metadata unavailable"
        ),
    }
}

/// Returns the `data` of a successful payload, or `None` when the call failed.
fn success_data(output: &Value) -> Option<&Value> {
    if output.get("success").and_then(Value::as_bool) != Some(true) {
        return None;
    }
    output.get("data")
}

/// Reproduces the [`Tool::render_for_llm`] default without recursing through `self`.
fn render_default(output: &Value) -> String {
    match serde_json::from_value::<ToolOutput>(output.clone()) {
        Ok(parsed) => render_tool_output(&parsed),
        Err(_) => render_payload_text(output),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::sync::Arc;

    use serde_json::{json, Value};
    use tempfile::TempDir;

    use super::*;
    use crate::tools::read_state::ReadCoverage;
    use crate::workspace::Workspace;
    use openjiuwen_sysop::{LocalWorkConfig, SysOperation};

    /// Both tools plus their shared read-state registry, over one temp workspace.
    struct Fixture {
        write: WriteFileTool,
        edit: EditFileTool,
        state: Arc<ReadStateRegistry>,
        dir: TempDir,
    }

    /// Builds both tools over a real local backend rooted at a fresh temp directory.
    fn setup() -> Fixture {
        let dir = tempfile::tempdir().unwrap();
        let operation = SysOperation::local(LocalWorkConfig::unrestricted(
            dir.path().to_string_lossy().to_string(),
        ))
        .unwrap();
        let fs = operation.fs();
        let state = ReadStateRegistry::new();
        let env = ToolEnv::new(Workspace::new(dir.path()));
        let write = WriteFileTool::new(env.clone(), fs.clone(), Arc::clone(&state));
        let edit = EditFileTool::new(env, fs, Arc::clone(&state));
        Fixture {
            write,
            edit,
            state,
            dir,
        }
    }

    /// Invokes `tool` with a fresh context and unwraps the payload envelope.
    async fn call(tool: &dyn Tool, inputs: Value) -> Value {
        tool.invoke(inputs, &ToolContext::new()).await.unwrap()
    }

    /// `edit_file` arguments for `name`.
    fn edit_args(name: &str, old_string: &str, new_string: &str) -> Value {
        json!({"file_path": name, "old_string": old_string, "new_string": new_string})
    }

    /// The absolute path `name` resolves to inside `dir`, as the tools report it.
    fn path_of(dir: &TempDir, name: &str) -> String {
        let root = std::fs::canonicalize(dir.path()).unwrap();
        root.join(name).to_string_lossy().into_owned()
    }

    /// The sorted property names of `card`'s input schema.
    fn property_keys(card: &ToolCard) -> Vec<&str> {
        let properties = card.input_params["properties"].as_object().unwrap();
        let mut keys: Vec<&str> = properties.keys().map(String::as_str).collect();
        keys.sort_unstable();
        keys
    }

    /// Asserts the payload failed with exactly `message`.
    fn assert_error(out: &Value, message: &str) {
        assert_eq!(out["success"], json!(false), "unexpected payload: {out}");
        assert_eq!(out["error"], json!(message));
    }

    /// Registers a read of `path` covering the 1-based inclusive lines `span`.
    fn register_read(state: &ReadStateRegistry, path: &Path, content: &str, span: (usize, usize)) {
        let (start, end) = span;
        let metadata = std::fs::metadata(path).unwrap();
        let coverage = Some(ReadCoverage { start, end });
        state.record_read(
            path,
            file_fingerprint(&metadata),
            coverage,
            Some(content.to_owned()),
        );
    }

    /// Registers a completed whole-file read of `path` holding `content`.
    fn register_full_read(state: &ReadStateRegistry, path: &Path, content: &str) {
        register_read(state, path, content, (1, content.lines().count().max(1)));
    }

    #[test]
    fn cards_match_python_arguments() {
        let write = WriteFileTool::card_for("en");
        assert_eq!(write.name, "write_file");
        assert!(!write.parallel_safe);
        assert_eq!(
            write.input_params["required"],
            json!(["file_path", "content"])
        );
        assert_eq!(property_keys(&write), ["content", "file_path"]);

        let edit = EditFileTool::card_for("en");
        assert_eq!(edit.name, "edit_file");
        assert!(!edit.parallel_safe);
        let required = json!(["file_path", "old_string", "new_string"]);
        assert_eq!(edit.input_params["required"], required);
        let keys = ["file_path", "new_string", "old_string", "replace_all"];
        assert_eq!(property_keys(&edit), keys);
        assert_eq!(
            edit.input_params["properties"]["replace_all"]["type"],
            json!("boolean")
        );
    }

    #[tokio::test]
    async fn write_creates_a_new_file() {
        let fx = setup();
        let out = call(
            &fx.write,
            json!({"file_path": "new.txt", "content": "hi\n"}),
        )
        .await;

        assert_eq!(out["data"]["type"], json!("create"));
        assert_eq!(out["data"]["created"], json!(true));
        assert_eq!(out["data"]["bytes_written"], json!(3));
        assert_eq!(out["data"]["original_file"], Value::Null);
        assert_eq!(out["data"]["file_path"], json!(path_of(&fx.dir, "new.txt")));
        assert_eq!(
            std::fs::read_to_string(fx.dir.path().join("new.txt")).unwrap(),
            "hi\n"
        );
        let rendered = fx.write.render_for_llm(&out);
        assert!(rendered.starts_with("Created "), "unexpected: {rendered}");
        assert!(
            rendered.ends_with(" bytes written)."),
            "unexpected: {rendered}"
        );
    }

    #[tokio::test]
    async fn write_requires_a_complete_fresh_read() {
        let fx = setup();
        let path = fx.dir.path().join("a.txt");
        std::fs::write(&path, "one\ntwo\nthree\n").unwrap();
        let args = |content: &str| json!({"file_path": "a.txt", "content": content});
        let resolved = path_of(&fx.dir, "a.txt");
        let not_read = format!(
            "Error: File has not been read yet: {resolved}. Read it before attempting to write it."
        );

        // Never read.
        assert_error(&call(&fx.write, args("new\n")).await, &not_read);

        // Only a window was read.
        let metadata = std::fs::metadata(&path).unwrap();
        let window = Some(ReadCoverage { start: 2, end: 3 });
        fx.state
            .record_read(&path, file_fingerprint(&metadata), window, None);
        let out = call(&fx.write, args("new\n")).await;
        let error = out["error"].as_str().unwrap();
        assert!(
            error.starts_with("Error: File has not been fully read yet."),
            "{error}"
        );
        assert!(error.contains("Continue with read_file offset="), "{error}");

        // Read in full, then changed on disk.
        register_full_read(&fx.state, &path, "one\ntwo\nthree\n");
        std::fs::write(&path, "one\ntwo\nthree\nfour\n").unwrap();
        let stale = format!("Error: File has been modified since read: {resolved}. Read it again before attempting to write it.");
        assert_error(&call(&fx.write, args("new\n")).await, &stale);
        let current = std::fs::read_to_string(&path).unwrap();
        let fingerprint = file_fingerprint(&std::fs::metadata(&path).unwrap());
        let gate = fx.state.write_gate(&path, fingerprint, &current);
        assert_eq!(gate, WriteGate::NotRead);
    }

    #[tokio::test]
    async fn write_updates_a_fully_read_file() {
        let fx = setup();
        let path = fx.dir.path().join("a.txt");
        std::fs::write(&path, "one\ntwo\n").unwrap();
        register_full_read(&fx.state, &path, "one\ntwo\n");

        let out = call(&fx.write, json!({"file_path": "a.txt", "content": "x\n"})).await;

        assert_eq!(out["data"]["type"], json!("update"));
        assert_eq!(out["data"]["created"], json!(false));
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "x\n");
        assert!(fx.write.render_for_llm(&out).starts_with("Updated "));
    }

    #[tokio::test]
    async fn write_reports_directory_targets_and_bad_arguments() {
        let fx = setup();
        std::fs::create_dir(fx.dir.path().join("adir")).unwrap();

        let out = call(&fx.write, json!({"file_path": "adir", "content": "x"})).await;
        let error = out["error"].as_str().unwrap();
        assert!(error.starts_with("Target path is a directory: "), "{error}");

        let cases = [
            (json!({"file_path": "a.txt"}), "content is required"),
            (
                json!({"file_path": "a.txt", "content": Value::Null}),
                "content is required",
            ),
            (
                json!({"file_path": "a.txt", "content": 5}),
                "content must be a string",
            ),
            (json!({"content": "x"}), "file_path is required"),
            (json!([]), "tool arguments must be an object"),
        ];
        for (inputs, message) in cases {
            assert_error(&call(&fx.write, inputs).await, message);
        }
    }

    #[tokio::test]
    async fn edit_replaces_a_unique_string() {
        let fx = setup();
        let path = fx.dir.path().join("a.txt");
        std::fs::write(&path, "alpha\nbeta\n").unwrap();
        register_full_read(&fx.state, &path, "alpha\nbeta\n");

        let out = call(&fx.edit, edit_args("a.txt", "beta", "BETA")).await;

        assert_eq!(out["data"]["replacements"], json!(1));
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "alpha\nBETA\n");
        let expected = format!("Edited {} (1 replacement(s)).", path_of(&fx.dir, "a.txt"));
        assert_eq!(fx.edit.render_for_llm(&out), expected);
    }

    #[tokio::test]
    async fn edit_validates_match_uniqueness() {
        let fx = setup();
        let path = fx.dir.path().join("a.txt");
        std::fs::write(&path, "x x\n").unwrap();
        register_full_read(&fx.state, &path, "x x\n");

        let repeated = format!(
            "Error: old_string matches 2 times in '{}'. Provide more surrounding context to make it unique, or set replace_all=true only if replacing every occurrence is intended.",
            path_of(&fx.dir, "a.txt")
        );
        assert_error(
            &call(&fx.edit, edit_args("a.txt", "x", "y")).await,
            &repeated,
        );

        let all = json!({"file_path": "a.txt", "old_string": "x", "new_string": "y", "replace_all": true});
        let out = call(&fx.edit, all).await;
        assert_eq!(out["data"]["replacements"], json!(2));
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "y y\n");
    }

    #[tokio::test]
    async fn edit_rejects_noop_and_unmatched_edits() {
        let fx = setup();

        let noop = call(&fx.edit, edit_args("missing.txt", "s", "s")).await;
        assert_error(&noop, IDENTICAL_STRINGS);

        let path = fx.dir.path().join("a.txt");
        std::fs::write(&path, "alpha\n").unwrap();
        register_full_read(&fx.state, &path, "alpha\n");
        let missing = format!(
            "Error: old_string not found in '{}'. Read the file and retry with the exact current text.",
            path_of(&fx.dir, "a.txt")
        );
        let out = call(&fx.edit, edit_args("a.txt", "gamma", "delta")).await;
        assert_error(&out, &missing);
    }

    #[tokio::test]
    async fn edit_requires_a_prior_read_and_an_existing_target() {
        let fx = setup();
        std::fs::write(fx.dir.path().join("a.txt"), "alpha\n").unwrap();

        let not_read = format!(
            "Error: File must be read before editing: {}. Read it before attempting to edit it.",
            path_of(&fx.dir, "a.txt")
        );
        let out = call(&fx.edit, edit_args("a.txt", "alpha", "beta")).await;
        assert_error(&out, &not_read);

        let absent = format!(
            "Error: File not found: '{}'.",
            path_of(&fx.dir, "ghost.txt")
        );
        let out = call(&fx.edit, edit_args("ghost.txt", "a", "b")).await;
        assert_error(&out, &absent);
    }

    #[tokio::test]
    async fn edit_creates_missing_files_and_refuses_existing_content() {
        let fx = setup();

        let out = call(&fx.edit, edit_args("fresh.txt", "", "hi\n")).await;
        assert_eq!(out["data"]["created"], json!(true));
        assert_eq!(out["data"]["replacements"], json!(0));
        assert_eq!(
            std::fs::read_to_string(fx.dir.path().join("fresh.txt")).unwrap(),
            "hi\n"
        );
        let created = format!("Created {}.", path_of(&fx.dir, "fresh.txt"));
        assert_eq!(fx.edit.render_for_llm(&out), created);

        // An omitted or explicitly null `old_string` selects the same branch.
        for name in ["b.txt", "c.txt"] {
            let mut inputs = json!({"file_path": name, "new_string": "hi\n"});
            if name == "c.txt" {
                inputs["old_string"] = Value::Null;
            }
            let out = call(&fx.edit, inputs).await;
            assert_eq!(out["data"]["created"], json!(true));
        }

        std::fs::write(fx.dir.path().join("taken.txt"), "already here\n").unwrap();
        let out = call(&fx.edit, edit_args("taken.txt", "", "hi\n")).await;
        assert_error(&out, CREATE_CONFLICT);
    }

    #[tokio::test]
    async fn edit_preserves_crlf_line_endings() {
        let fx = setup();
        let path = fx.dir.path().join("a.txt");
        std::fs::write(&path, "a\r\nb\r\n").unwrap();
        register_full_read(&fx.state, &path, "a\nb\n");

        let out = call(&fx.edit, edit_args("a.txt", "b", "B")).await;

        assert_eq!(out["data"]["replacements"], json!(1));
        assert_eq!(std::fs::read(&path).unwrap(), b"a\r\nB\r\n");
    }

    #[tokio::test]
    async fn edit_validates_its_arguments() {
        let fx = setup();
        let cases = [
            (json!({"file_path": "a.txt"}), "new_string is required"),
            (
                json!({"file_path": "a.txt", "new_string": Value::Null}),
                "new_string is required",
            ),
            (
                json!({"file_path": "a.txt", "new_string": 5}),
                "new_string must be a string",
            ),
            (
                json!({"file_path": "a.txt", "old_string": 5, "new_string": "b"}),
                "old_string must be a string",
            ),
            (json!({"new_string": "b"}), "file_path is required"),
        ];
        for (inputs, message) in cases {
            assert_error(&call(&fx.edit, inputs).await, message);
        }
    }

    #[tokio::test]
    async fn missing_filesystem_backend_is_a_failure_payload() {
        let dir = tempfile::tempdir().unwrap();
        let env = ToolEnv::new(Workspace::new(dir.path()));
        let write = WriteFileTool::new(env.clone(), None, ReadStateRegistry::new());
        let edit = EditFileTool::new(env, None, ReadStateRegistry::new());

        let inputs = json!({"file_path": "a.txt", "content": "x"});
        assert_error(&call(&write, inputs).await, NO_BACKEND);
        let inputs = edit_args("a.txt", "a", "b");
        assert_error(&call(&edit, inputs).await, NO_BACKEND);
    }
}
