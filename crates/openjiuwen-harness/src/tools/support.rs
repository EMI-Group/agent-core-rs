//! Shared tool helpers: the payload envelope, field rendering, truncation, schema generation, tool cards and path resolution.
//!
//! This is the lowest layer of the model-facing tool set: it performs no IO beyond the
//! workspace path checks and never talks to a backend. Every tool in
//! [`crate::tools`] returns a [`ToolPayload`] and resolves model-supplied paths through
//! [`ToolEnv`], so the payload shape and the path rules stay identical across tools.
//!
//! Python parity: the `ToolOutput` envelope, `render_fields` and the truncation helpers of
//! `openjiuwen/harness/tools/`, plus the schema/card construction shared by every tool.

// Python parity: tool helpers return the `ToolPayload` envelope by value, and the public
// signatures are part of the tool contract. Boxing the error would change every caller, so
// the size/performance heuristic does not apply here.
#![allow(clippy::result_large_err)]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use openjiuwen_foundation::tool::ToolCard;
use openjiuwen_sysop::workspace::{canonicalize_containing, clean_path, is_within};

use crate::permission::{PermissionEngine, PermissionLevel};
use crate::workspace::Workspace;

/// Longest accepted model-supplied path, in characters.
const MAX_PATH_CHARS: usize = 4096;

/// Longest accepted single path component, in characters.
const MAX_COMPONENT_CHARS: usize = 255;

/// The Python `ToolOutput` envelope, as JSON.
///
/// Python parity: `openjiuwen/harness/tools/base.py::ToolOutput`. Every optional field
/// serializes as `null` when unset (no `skip_serializing_if`), so consumers always see all
/// six keys.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ToolPayload {
    /// Whether the tool call succeeded.
    pub success: bool,
    /// Tool-specific result data; `None` on failure unless explicitly attached.
    pub data: Option<Value>,
    /// Human-readable failure message; `None` on success.
    pub error: Option<String>,
    /// Content the model should see even when the full payload is trimmed.
    pub extracted_content: Option<String>,
    /// Whether `extracted_content` is meant to be shown only once.
    pub include_extracted_content_only_once: bool,
    /// Long-term-memory text attached by the memory tools.
    pub long_term_memory: Option<String>,
}

impl ToolPayload {
    /// Builds a success payload carrying `data`.
    #[must_use]
    pub fn ok(data: impl Into<Value>) -> Self {
        Self {
            success: true,
            data: Some(data.into()),
            ..Self::default()
        }
    }

    /// Builds a success payload carrying an empty JSON object, like a no-op tool.
    #[must_use]
    pub fn ok_empty() -> Self {
        Self::ok(json!({}))
    }

    /// Builds a failure payload with `message` and no data.
    #[must_use]
    pub fn fail(message: impl Into<String>) -> Self {
        Self {
            success: false,
            data: None,
            error: Some(message.into()),
            ..Self::default()
        }
    }

    /// Builds a failure payload with `message` and the diagnostic `data`.
    #[must_use]
    pub fn fail_with_data(message: impl Into<String>, data: impl Into<Value>) -> Self {
        Self {
            success: false,
            data: Some(data.into()),
            error: Some(message.into()),
            ..Self::default()
        }
    }

    /// Attaches `text` as the payload's `extracted_content`.
    #[must_use]
    pub fn with_extracted_content(mut self, text: impl Into<String>) -> Self {
        self.extracted_content = Some(text.into());
        self
    }

    /// Renders the envelope as a JSON object.
    ///
    /// All six keys are always present; unset optional fields render as `null`.
    #[must_use]
    pub fn to_json(&self) -> Value {
        json!({
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "extracted_content": self.extracted_content,
            "include_extracted_content_only_once": self.include_extracted_content_only_once,
            "long_term_memory": self.long_term_memory,
        })
    }
}

/// Renders `key: text` pairs joined by `\n`.
///
/// Ports the Python `render_fields`: a pair whose value is `null` or the empty string is
/// skipped, a JSON string renders verbatim (no quotes) and any other value renders as
/// compact JSON. Input order is preserved.
#[must_use]
pub fn render_fields(fields: &[(&str, Value)]) -> String {
    let mut rendered = String::new();
    for (key, value) in fields {
        let text = match value {
            Value::Null => continue,
            Value::String(text) if text.is_empty() => continue,
            Value::String(text) => text.clone(),
            other => compact_json(other),
        };
        if !rendered.is_empty() {
            rendered.push('\n');
        }
        rendered.push_str(key);
        rendered.push_str(": ");
        rendered.push_str(&text);
    }
    rendered
}

/// Serializes `value` as compact JSON, falling back to `null` instead of panicking.
#[must_use]
pub fn compact_json(value: &Value) -> String {
    match serde_json::to_string(value) {
        Ok(rendered) => rendered,
        Err(error) => {
            tracing::debug!(
                target: "openjiuwen_harness::tools::support",
                %error,
                "failed to render JSON value"
            );
            "null".to_owned()
        }
    }
}

/// Result of [`truncate_chars`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Truncated {
    /// The rendered text, with [`truncation_marker`] appended when truncated.
    pub text: String,
    /// Whether anything was dropped.
    pub truncated: bool,
    /// Number of characters dropped from the input.
    pub omitted_chars: usize,
}

/// The marker appended to a truncated string.
#[must_use]
pub fn truncation_marker(omitted: usize) -> String {
    format!("\n... [output truncated: {omitted} characters omitted] ...")
}

/// Truncates `text` to at most `max_chars` characters.
///
/// `max_chars == 0` means unlimited. Characters, not bytes, are counted, so a multi-byte
/// UTF-8 character is never split; the tail is replaced by [`truncation_marker`].
#[must_use]
pub fn truncate_chars(text: &str, max_chars: usize) -> Truncated {
    let total = text.chars().count();
    if max_chars == 0 || total <= max_chars {
        return Truncated {
            text: text.to_owned(),
            truncated: false,
            omitted_chars: 0,
        };
    }
    let omitted = total - max_chars;
    let prefix: String = text.chars().take(max_chars).collect();
    Truncated {
        text: format!("{prefix}{}", truncation_marker(omitted)),
        truncated: true,
        omitted_chars: omitted,
    }
}

/// Derives the JSON Schema of a `JsonSchema` type in the Python tool shape.
///
/// The generated schema is stripped of the top-level `$schema`, `title`, `definitions` and
/// `$defs` keys, and is normalized to an object schema with `type: "object"` and
/// `properties`; `required` is kept only when present and non-empty.
#[must_use]
pub fn schema_for<T: JsonSchema>() -> Value {
    let schema = schemars::SchemaGenerator::default().into_root_schema_for::<T>();
    let mut value = match serde_json::to_value(&schema) {
        Ok(value) => value,
        Err(error) => {
            tracing::debug!(
                target: "openjiuwen_harness::tools::support",
                %error,
                "failed to serialize generated tool schema"
            );
            json!({})
        }
    };
    let Value::Object(object) = &mut value else {
        return json!({"type": "object", "properties": {}});
    };
    for key in ["$schema", "title", "definitions", "$defs"] {
        object.remove(key);
    }
    object
        .entry("type")
        .or_insert_with(|| Value::String("object".to_owned()));
    object
        .entry("properties")
        .or_insert_with(|| Value::Object(serde_json::Map::new()));
    match object.get("required") {
        Some(Value::Array(required)) if required.is_empty() => {
            object.remove("required");
        }
        Some(_) | None => {}
    }
    value
}

/// Builds a model-facing tool card.
///
/// The card keeps the [`ToolCard`] defaults: a fresh hex uuid, `exposure = Direct`,
/// `execution = Server` and `idempotent = false`.
#[must_use]
pub fn tool_card(
    name: &str,
    description: &str,
    input_params: Value,
    parallel_safe: bool,
) -> ToolCard {
    ToolCard::new(name, description, input_params).with_parallel_safe(parallel_safe)
}

/// Whether `language` selects English prompt text; only the exact string `"en"` does.
#[must_use]
pub fn is_english(language: &str) -> bool {
    language == "en"
}

/// The workspace plus the optional permission gate and language a tool resolves against.
///
/// Path resolution is sandboxed: relative paths go through [`Workspace::resolve`], absolute
/// paths are accepted only when they point inside the workspace, and a `deny` permission
/// decision short-circuits before any path validation.
#[derive(Clone)]
pub struct ToolEnv {
    /// The workspace model-supplied paths are resolved against.
    workspace: Workspace,
    /// Optional permission engine; every check fails open when absent.
    permission: Option<Arc<PermissionEngine>>,
    /// Description language; defaults to `"cn"` like the Python tools.
    language: String,
    /// Optional access boundary every resolved path must stay inside.
    ///
    /// Python parity: the `project_root` layer of
    /// `openjiuwen/core/sys_operation/cwd.py`, which the filesystem and shell guards use as
    /// one of their allowed roots. `None` keeps the workspace root as the only boundary.
    access_root: Option<PathBuf>,
}

impl ToolEnv {
    /// Builds an environment over `workspace` with the default language `"cn"`.
    #[must_use]
    pub fn new(workspace: Workspace) -> Self {
        Self {
            workspace,
            permission: None,
            language: "cn".to_owned(),
            access_root: None,
        }
    }

    /// Narrows path resolution to `root`, or widens it back to the workspace with `None`.
    ///
    /// A configured root is an access boundary, not a resolution root: relative paths still
    /// resolve against the workspace, and a resolved path outside `root` is rejected as an
    /// access denial.
    #[must_use]
    pub fn with_access_root(mut self, root: Option<PathBuf>) -> Self {
        self.access_root = root;
        self
    }

    /// The access boundary of this environment, when one is configured.
    #[must_use]
    pub fn access_root(&self) -> Option<&Path> {
        self.access_root.as_deref()
    }

    /// Wires the permission engine used by [`Self::deny_reason`].
    #[must_use]
    pub fn with_permission(mut self, engine: Arc<PermissionEngine>) -> Self {
        self.permission = Some(engine);
        self
    }

    /// Sets the description language.
    #[must_use]
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = language.into();
        self
    }

    /// The workspace model-supplied paths are resolved against.
    #[must_use]
    pub fn workspace(&self) -> &Workspace {
        &self.workspace
    }

    /// The description language.
    #[must_use]
    pub fn language(&self) -> &str {
        &self.language
    }

    /// The permission engine consulted before a path is touched, when one is wired.
    ///
    /// Tools that mutate state outside the filesystem backends (such as `worktree`) forward the
    /// engine instead of consulting [`Self::deny_reason`] themselves.
    #[must_use]
    pub fn permission(&self) -> Option<&Arc<PermissionEngine>> {
        self.permission.as_ref()
    }

    /// Whether the description language is English.
    #[must_use]
    pub fn is_english(&self) -> bool {
        is_english(&self.language)
    }

    /// Returns the reason a `deny` decision rejects the call, or `None` when it may proceed.
    ///
    /// `ask` and `allow` decisions do not block: only the interrupt rail can ask a human, so
    /// the tool itself must not treat `ask` as a failure.
    #[must_use]
    pub fn deny_reason(&self, tool: &str, args: &Value) -> Option<String> {
        let engine = self.permission.as_ref()?;
        let result = engine.check_permission(tool, args);
        if result.permission == PermissionLevel::Deny {
            Some(
                result
                    .reason
                    .clone()
                    .unwrap_or_else(|| format!("Permission denied: {tool}")),
            )
        } else {
            None
        }
    }

    /// Resolves a model-supplied file path inside the workspace.
    ///
    /// Check order: the permission gate, an empty path, a NUL byte, the total length, single
    /// component length, then sandboxed resolution. Every failure is a
    /// [`ToolPayload::fail`].
    ///
    /// # Errors
    ///
    /// Returns a failure payload when the call is denied, `raw` is empty, contains a NUL
    /// byte, exceeds the length limits, points outside the workspace, or is rejected by
    /// [`Workspace::resolve`].
    pub async fn resolve_path(
        &self,
        tool: &str,
        field: &str,
        raw: &str,
        args: &Value,
    ) -> Result<PathBuf, ToolPayload> {
        self.resolve_checked(tool, field, raw, args, false).await
    }

    /// Resolves a model-supplied directory path inside the workspace.
    ///
    /// Like [`Self::resolve_path`], except an empty `raw` means the workspace root.
    ///
    /// # Errors
    ///
    /// Same conditions as [`Self::resolve_path`].
    pub async fn resolve_dir(
        &self,
        tool: &str,
        field: &str,
        raw: &str,
        args: &Value,
    ) -> Result<PathBuf, ToolPayload> {
        self.resolve_checked(tool, field, raw, args, true).await
    }

    /// Shared implementation of [`Self::resolve_path`] and [`Self::resolve_dir`].
    async fn resolve_checked(
        &self,
        tool: &str,
        field: &str,
        raw: &str,
        args: &Value,
        empty_means_root: bool,
    ) -> Result<PathBuf, ToolPayload> {
        if let Some(reason) = self.deny_reason(tool, args) {
            return Err(ToolPayload::fail(reason));
        }
        if raw.is_empty() {
            return if empty_means_root {
                Ok(self.workspace.root_path.clone())
            } else {
                Err(ToolPayload::fail(format!("{field} is required")))
            };
        }
        if raw.contains('\0') {
            return Err(ToolPayload::fail(format!(
                "Invalid path '{raw}': path contains a NUL byte."
            )));
        }
        let length = raw.chars().count();
        if length > MAX_PATH_CHARS {
            return Err(ToolPayload::fail(format!(
                "file_path is too long ({length} chars; max {MAX_PATH_CHARS}). Pass a plain path in `file_path` and the file body in `content`."
            )));
        }
        if has_overlong_component(raw) {
            return Err(ToolPayload::fail(
                "file_path has a component longer than 255 characters. Pass a plain path in `file_path` and the file body in `content` — do not serialize the whole tool-call JSON into `file_path`."
                    .to_owned(),
            ));
        }
        let relative = self.relative_for_resolution(raw).await?;
        let resolved = self
            .workspace
            .resolve(&relative)
            .await
            .map_err(|error| ToolPayload::fail(format!("Invalid path '{raw}': {error}")))?;
        self.check_access_root(raw, &resolved).await
    }

    /// Rejects `resolved` when it falls outside the configured access boundary.
    ///
    /// The boundary is the project root (Python's `project_root` CWD layer). It only ever
    /// narrows what the workspace already allows, so an environment without a boundary
    /// keeps the workspace root as the single restriction.
    async fn check_access_root(&self, raw: &str, resolved: &Path) -> Result<PathBuf, ToolPayload> {
        let Some(root) = self.access_root.as_deref() else {
            return Ok(resolved.to_path_buf());
        };
        let boundary = canonicalize_containing(root).await;
        if is_within(&canonicalize_containing(resolved).await, &boundary) {
            return Ok(resolved.to_path_buf());
        }
        Err(ToolPayload::fail(format!(
            "Access denied: '{raw}' is outside the project root."
        )))
    }

    /// Turns `raw` into a path [`Workspace::resolve`] accepts.
    ///
    /// Relative paths pass through unchanged; absolute paths must lexically normalize to a
    /// location inside the canonicalized workspace root, and are then reduced to the relative
    /// remainder.
    async fn relative_for_resolution(&self, raw: &str) -> Result<String, ToolPayload> {
        let path = Path::new(raw);
        if !path.is_absolute() {
            return Ok(raw.to_owned());
        }
        let root = tokio::fs::canonicalize(&self.workspace.root_path)
            .await
            .unwrap_or_else(|_| self.workspace.root_path.clone());
        let normalized = clean_path(path);
        match normalized.strip_prefix(&root) {
            Ok(relative) => Ok(relative.to_string_lossy().into_owned()),
            Err(_) => Err(ToolPayload::fail(format!(
                "Access denied: '{raw}' is outside the workspace."
            ))),
        }
    }
}

/// Whether any `/`- or `\`-separated component of `raw` exceeds [`MAX_COMPONENT_CHARS`].
fn has_overlong_component(raw: &str) -> bool {
    raw.split(['/', '\\'])
        .any(|component| component.chars().count() > MAX_COMPONENT_CHARS)
}

/// Deserializes model-supplied tool arguments into `T`.
///
/// Python tools receive a JSON object; a non-object payload is a protocol violation and is
/// reported before serde sees it.
///
/// # Errors
///
/// Returns a failure payload when `inputs` is not a JSON object or does not match `T`.
pub fn parse_args<T: DeserializeOwned>(inputs: &Value) -> Result<T, ToolPayload> {
    if !inputs.is_object() {
        return Err(ToolPayload::fail("tool arguments must be an object"));
    }
    serde_json::from_value(inputs.clone())
        .map_err(|error| ToolPayload::fail(format!("invalid tool arguments: {error}")))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::collections::BTreeMap;

    use schemars::JsonSchema;
    use serde::{Deserialize, Serialize};
    use serde_json::{json, Value};

    use super::*;
    use crate::permission::{
        PermissionEngine, PermissionEngineConfig, PermissionLevel, PermissionsSection,
    };

    /// Arguments used by the schema and parsing tests.
    #[derive(Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
    struct TestArgs {
        /// Path argument.
        file_path: String,
        /// Numeric argument.
        count: u32,
    }

    /// A type whose fields are all optional, so the schema has no `required` list.
    #[derive(Debug, JsonSchema)]
    #[allow(dead_code)]
    struct OptionalArgs {
        /// Optional value.
        note: Option<String>,
    }

    /// A workspace rooted at a fresh temporary directory.
    fn workspace(dir: &tempfile::TempDir) -> Workspace {
        Workspace::new(dir.path())
    }

    /// A tool env with no permission engine.
    fn env(dir: &tempfile::TempDir) -> ToolEnv {
        ToolEnv::new(workspace(dir))
    }

    /// An engine over `tools` and `defaults["*"]`, with packaged floors disabled.
    fn engine(
        tools: &[(&str, PermissionLevel)],
        default: PermissionLevel,
    ) -> Arc<PermissionEngine> {
        let section = PermissionsSection {
            package_builtin_rules: false,
            defaults: BTreeMap::from([("*".to_owned(), default)]),
            tools: tools
                .iter()
                .map(|(name, level)| ((*name).to_owned(), *level))
                .collect(),
            ..PermissionsSection::default()
        };
        Arc::new(PermissionEngine::new(PermissionEngineConfig::new(section)))
    }

    #[test]
    fn payload_builders_and_json_shape() {
        let ok = ToolPayload::ok(json!({"value": 1}));
        assert!(ok.success);
        assert_eq!(ok.data, Some(json!({"value": 1})));
        assert_eq!(ok.error, None);

        let empty = ToolPayload::ok_empty();
        assert_eq!(empty.data, Some(json!({})));

        let fail = ToolPayload::fail("boom");
        assert!(!fail.success);
        assert_eq!(fail.error.as_deref(), Some("boom"));
        assert_eq!(fail.data, None);

        let with_data = ToolPayload::fail_with_data("boom", json!({"detail": 2}));
        assert!(!with_data.success);
        assert_eq!(with_data.data, Some(json!({"detail": 2})));

        let extracted = ToolPayload::ok_empty().with_extracted_content("shown once");
        assert_eq!(extracted.extracted_content.as_deref(), Some("shown once"));
        assert!(!extracted.include_extracted_content_only_once);

        let json = extracted.to_json();
        let object = json.as_object().unwrap();
        assert_eq!(object.len(), 6);
        for key in [
            "success",
            "data",
            "error",
            "extracted_content",
            "include_extracted_content_only_once",
            "long_term_memory",
        ] {
            assert!(object.contains_key(key), "missing key {key}");
        }
        assert_eq!(object.get("error"), Some(&Value::Null));
        assert_eq!(object.get("long_term_memory"), Some(&Value::Null));
        assert_eq!(
            object.get("include_extracted_content_only_once"),
            Some(&Value::Bool(false))
        );
    }

    #[test]
    fn render_fields_skips_nulls_and_empty_strings() {
        let fields = [
            ("ignored_null", Value::Null),
            ("ignored_empty", json!("")),
            ("text", json!("verbatim")),
            ("number", json!(3)),
            ("object", json!({"x": 1})),
            ("array", json!([1, "two"])),
        ];
        assert_eq!(
            render_fields(&fields),
            "text: verbatim\nnumber: 3\nobject: {\"x\":1}\narray: [1,\"two\"]"
        );
        assert_eq!(render_fields(&[]), "");
    }

    #[test]
    fn truncate_chars_handles_limits_and_multibyte_text() {
        let unlimited = truncate_chars("anything", 0);
        assert!(!unlimited.truncated);
        assert_eq!(unlimited.text, "anything");
        assert_eq!(unlimited.omitted_chars, 0);

        let exact = truncate_chars("abc", 3);
        assert_eq!(exact.text, "abc");
        assert!(!exact.truncated);

        let cut = truncate_chars("abcdef", 3);
        assert_eq!(
            cut.text,
            "abc\n... [output truncated: 3 characters omitted] ..."
        );
        assert!(cut.truncated);
        assert_eq!(cut.omitted_chars, 3);

        let multibyte = truncate_chars("汉字abc", 2);
        assert_eq!(
            multibyte.text,
            "汉字\n... [output truncated: 3 characters omitted] ..."
        );
        assert_eq!(multibyte.omitted_chars, 3);
        assert!(multibyte.text.starts_with("汉字"));
    }

    #[test]
    fn schema_for_strips_dialect_keys_and_keeps_properties() {
        let schema = schema_for::<TestArgs>();
        let object = schema.as_object().unwrap();
        assert_eq!(object.get("type"), Some(&json!("object")));
        let properties = object.get("properties").unwrap().as_object().unwrap();
        assert!(properties.contains_key("file_path"));
        assert!(properties.contains_key("count"));
        assert!(object.get("required").is_some());
        for key in ["$schema", "title", "definitions", "$defs"] {
            assert!(!object.contains_key(key), "unexpected key {key}");
        }

        let optional = schema_for::<OptionalArgs>();
        assert!(!optional.as_object().unwrap().contains_key("required"));
        assert_eq!(
            optional.get("type"),
            Some(&json!("object")),
            "optional schema must still describe an object"
        );
    }

    #[test]
    fn tool_card_keeps_python_defaults() {
        let card = tool_card(
            "read_file",
            "Reads a file",
            json!({"type": "object"}),
            false,
        );
        assert_eq!(card.name, "read_file");
        assert_eq!(card.description, "Reads a file");
        assert_eq!(card.input_params, json!({"type": "object"}));
        assert!(!card.parallel_safe);
        assert!(!card.idempotent);
        assert_ne!(card.id, "");
    }

    #[test]
    fn language_helpers_default_to_chinese() {
        assert!(is_english("en"));
        assert!(!is_english("cn"));
        assert!(!is_english("EN"));

        let dir = tempfile::tempdir().unwrap();
        let default = env(&dir);
        assert_eq!(default.language(), "cn");
        assert!(!default.is_english());

        let english = default.with_language("en");
        assert_eq!(english.language(), "en");
        assert!(english.is_english());
    }

    #[tokio::test]
    async fn relative_path_resolves_inside_workspace() {
        let dir = tempfile::tempdir().unwrap();
        let env = env(&dir);
        let resolved = env
            .resolve_path("read_file", "file_path", "notes.md", &json!({}))
            .await
            .unwrap();
        let root = tokio::fs::canonicalize(dir.path()).await.unwrap();
        assert_eq!(resolved, root.join("notes.md"));
    }

    #[tokio::test]
    async fn traversal_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let env = env(&dir);
        let payload = env
            .resolve_path("read_file", "file_path", "../escape.txt", &json!({}))
            .await
            .unwrap_err();
        assert!(!payload.success);
        let message = payload.error.unwrap();
        assert!(
            message.starts_with("Invalid path '../escape.txt':"),
            "unexpected message: {message}"
        );
    }

    #[tokio::test]
    async fn absolute_path_inside_root_is_accepted() {
        let dir = tempfile::tempdir().unwrap();
        let env = env(&dir);
        let root = tokio::fs::canonicalize(dir.path()).await.unwrap();
        let raw = root.join("inside.txt").to_string_lossy().into_owned();
        let resolved = env
            .resolve_path("write_file", "file_path", &raw, &json!({}))
            .await
            .unwrap();
        assert_eq!(resolved, root.join("inside.txt"));
    }

    #[tokio::test]
    async fn absolute_path_outside_root_is_denied() {
        let dir = tempfile::tempdir().unwrap();
        let other = tempfile::tempdir().unwrap();
        let env = env(&dir);
        let raw = other
            .path()
            .join("outside.txt")
            .to_string_lossy()
            .into_owned();
        let payload = env
            .resolve_path("read_file", "file_path", &raw, &json!({}))
            .await
            .unwrap_err();
        assert_eq!(
            payload,
            ToolPayload::fail(format!("Access denied: '{raw}' is outside the workspace."))
        );
    }

    #[tokio::test]
    async fn path_validation_messages_match_python() {
        let dir = tempfile::tempdir().unwrap();
        let env = env(&dir);

        let empty = env
            .resolve_path("read_file", "file_path", "", &json!({}))
            .await
            .unwrap_err();
        assert_eq!(empty.error.as_deref(), Some("file_path is required"));

        let empty_other_field = env.resolve_dir("glob", "path", "", &json!({})).await;
        assert_eq!(empty_other_field, Ok(dir.path().to_path_buf()));

        let nul = env
            .resolve_path("read_file", "file_path", "a\0b", &json!({}))
            .await
            .unwrap_err();
        assert_eq!(
            nul.error.as_deref(),
            Some("Invalid path 'a\0b': path contains a NUL byte.")
        );

        let too_long = "a".repeat(MAX_PATH_CHARS + 1);
        let long = env
            .resolve_path("write_file", "file_path", &too_long, &json!({}))
            .await
            .unwrap_err();
        let expected_long = format!(
            "file_path is too long ({} chars; max {MAX_PATH_CHARS}). Pass a plain path in `file_path` and the file body in `content`.",
            MAX_PATH_CHARS + 1
        );
        assert_eq!(long.error.as_deref(), Some(expected_long.as_str()));

        let long_component = "a".repeat(300);
        let component = env
            .resolve_path("write_file", "file_path", &long_component, &json!({}))
            .await
            .unwrap_err();
        assert_eq!(
            component.error.as_deref(),
            Some("file_path has a component longer than 255 characters. Pass a plain path in `file_path` and the file body in `content` — do not serialize the whole tool-call JSON into `file_path`.")
        );
    }

    #[test]
    fn deny_reason_is_none_without_an_engine() {
        let dir = tempfile::tempdir().unwrap();
        let env = env(&dir);
        assert_eq!(env.deny_reason("write_file", &json!({})), None);
    }

    #[tokio::test]
    async fn deny_decision_short_circuits_resolution() {
        let dir = tempfile::tempdir().unwrap();
        let env = ToolEnv::new(workspace(&dir)).with_permission(engine(
            &[("write_file", PermissionLevel::Deny)],
            PermissionLevel::Allow,
        ));
        let args = json!({"file_path": "a.txt"});
        assert_eq!(
            env.deny_reason("write_file", &args),
            Some("Denied by rule: tools.write_file".to_owned())
        );

        let denied = env
            .resolve_path("write_file", "file_path", "a.txt", &args)
            .await
            .unwrap_err();
        assert_eq!(
            denied.error.as_deref(),
            Some("Denied by rule: tools.write_file")
        );

        // The permission gate runs before any path validation, so even an empty path is
        // reported as the denial.
        let empty = env
            .resolve_path("write_file", "file_path", "", &args)
            .await
            .unwrap_err();
        assert_eq!(
            empty.error.as_deref(),
            Some("Denied by rule: tools.write_file")
        );
    }

    #[test]
    fn allow_and_ask_do_not_block() {
        let dir = tempfile::tempdir().unwrap();
        let allowed = ToolEnv::new(workspace(&dir)).with_permission(engine(
            &[("write_file", PermissionLevel::Allow)],
            PermissionLevel::Allow,
        ));
        assert_eq!(allowed.deny_reason("write_file", &json!({})), None);

        let asked = ToolEnv::new(workspace(&dir)).with_permission(engine(
            &[("write_file", PermissionLevel::Ask)],
            PermissionLevel::Allow,
        ));
        assert_eq!(asked.deny_reason("write_file", &json!({})), None);
    }

    #[test]
    fn parse_args_reports_protocol_violations() {
        let non_object = parse_args::<TestArgs>(&json!([])).unwrap_err();
        assert_eq!(
            non_object.error.as_deref(),
            Some("tool arguments must be an object")
        );

        let parsed = parse_args::<TestArgs>(&json!({"file_path": "a.txt", "count": 2})).unwrap();
        assert_eq!(
            parsed,
            TestArgs {
                file_path: "a.txt".to_owned(),
                count: 2,
            }
        );

        let wrong_type = parse_args::<TestArgs>(&json!({"file_path": 5, "count": 2})).unwrap_err();
        let message = wrong_type.error.unwrap();
        assert!(
            message.starts_with("invalid tool arguments: "),
            "unexpected message: {message}"
        );
    }
}
