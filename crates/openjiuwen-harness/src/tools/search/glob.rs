//! The `glob` tool: finds files by name pattern through the sys-operation sandbox.
//!
//! Python parity: `openjiuwen/harness/tools/filesystem.py::GlobTool`.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use openjiuwen_common::error::Result;
use openjiuwen_foundation::tool::{
    render_payload_text, render_tool_output, Tool, ToolCard, ToolContext, ToolOutput,
};
use openjiuwen_sysop::fs::types::SearchFilesParams;
use openjiuwen_sysop::operation::FsOperation;

use super::super::descriptions::description;
use super::super::support::{parse_args, schema_for, tool_card, ToolEnv, ToolPayload};
use super::{relative_name, string_list, MISSING_FS, MISSING_PATTERN};

/// Maximum number of files `glob` returns (`GlobTool.DEFAULT_MAX_RESULTS`).
const MAX_GLOB_RESULTS: usize = 100;

/// Arguments of the `glob` tool.
///
/// Python parity: the `pattern`/`path` parameters of `GlobTool.invoke`. A missing or empty
/// `path` searches the workspace root.
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct GlobArgs {
    /// The glob pattern to match file names against, e.g. `**/*.rs` or `src/**/*.{rs,toml}`.
    pub pattern: String,
    /// Directory to search in; defaults to the workspace root.
    #[serde(default)]
    #[schemars(with = "String")]
    pub path: Option<String>,
}

/// The `glob` tool: finds files by name pattern through a [`FsOperation`] backend.
pub struct GlobTool {
    /// Path resolution, permission gate and description language.
    env: ToolEnv,
    /// The filesystem backend, when one is configured.
    fs: Option<Arc<dyn FsOperation>>,
    /// The model-facing card built from [`ToolEnv::language`].
    card: ToolCard,
}

impl GlobTool {
    /// Builds the tool over `env`, searching through `fs`.
    #[must_use]
    pub fn new(env: ToolEnv, fs: Option<Arc<dyn FsOperation>>) -> Self {
        let card = Self::card_for(env.language());
        Self { env, fs, card }
    }

    /// Builds the model-facing card in `language`.
    #[must_use]
    pub fn card_for(language: &str) -> ToolCard {
        tool_card(
            "glob",
            description("glob", language),
            schema_for::<GlobArgs>(),
            true,
        )
    }

    /// Expands shell-style brace alternatives, one nesting level deep.
    ///
    /// `*.{rs,toml}` becomes `["*.rs", "*.toml"]`. A pattern without a brace pair, whose
    /// brace body still contains braces, or with fewer than two alternatives is returned
    /// unchanged as a single element. Every expanded alternative is expanded again, so
    /// `{a,b}/{c,d}` yields four patterns. Alternatives are trimmed.
    #[must_use]
    pub fn expand_braces(pattern: &str) -> Vec<String> {
        let Some((prefix, body, suffix)) = split_brace_group(pattern) else {
            return vec![pattern.to_owned()];
        };
        let alternatives: Vec<&str> = body
            .split(',')
            .map(str::trim)
            .filter(|alternative| !alternative.is_empty())
            .collect();
        if alternatives.len() < 2 {
            return vec![pattern.to_owned()];
        }
        let mut expanded = Vec::new();
        for alternative in alternatives {
            expanded.extend(Self::expand_braces(&format!(
                "{prefix}{alternative}{suffix}"
            )));
        }
        expanded
    }

    /// Runs the search and returns the payload.
    async fn execute(&self, inputs: &Value) -> ToolPayload {
        // Python reads `pattern` from the raw arguments and reports its absence with the
        // plain validation text rather than a schema error, so check it before parsing.
        if inputs.is_object()
            && inputs
                .get("pattern")
                .and_then(Value::as_str)
                .is_none_or(str::is_empty)
        {
            return ToolPayload::fail(MISSING_PATTERN);
        }
        let args: GlobArgs = match parse_args(inputs) {
            Ok(args) => args,
            Err(payload) => return payload,
        };
        if args.pattern.is_empty() {
            return ToolPayload::fail(MISSING_PATTERN);
        }

        let base = match self
            .env
            .resolve_dir("glob", "path", args.path.as_deref().unwrap_or(""), inputs)
            .await
        {
            Ok(base) => base,
            Err(payload) => return payload,
        };
        let Some(fs) = self.fs.as_ref() else {
            return ToolPayload::fail(MISSING_FS);
        };

        let started = Instant::now();
        let base_string = base.to_string_lossy().into_owned();
        let mut matching_files: Vec<String> = Vec::new();
        for pattern in Self::expand_braces(&args.pattern) {
            let result = fs
                .search_files(SearchFilesParams::new(base_string.clone(), pattern))
                .await;
            if result.is_failure() {
                return ToolPayload::fail(result.message);
            }
            let Some(data) = result.data else {
                continue;
            };
            for item in data.matching_files {
                if !matching_files.iter().any(|known| known == &item.path) {
                    matching_files.push(item.path);
                }
            }
        }

        let truncated = matching_files.len() > MAX_GLOB_RESULTS;
        matching_files.truncate(MAX_GLOB_RESULTS);
        let filenames: Vec<String> = matching_files
            .iter()
            .map(|path| relative_name(&base, &PathBuf::from(path)))
            .collect();
        let duration_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
        ToolPayload::ok(json!({
            "durationMs": duration_ms,
            "numFiles": matching_files.len(),
            "filenames": filenames,
            "matching_files": matching_files,
            "truncated": truncated,
            "count": matching_files.len(),
        }))
    }
}

#[async_trait]
impl Tool for GlobTool {
    fn card(&self) -> &ToolCard {
        &self.card
    }

    async fn invoke(&self, inputs: Value, ctx: &ToolContext) -> Result<Value> {
        ctx.check_cancelled()?;
        Ok(self.execute(&inputs).await.to_json())
    }

    fn render_for_llm(&self, output: &Value) -> String {
        let Ok(parsed) = serde_json::from_value::<ToolOutput>(output.clone()) else {
            return render_payload_text(output);
        };
        let Some(data) = parsed.data.as_ref().filter(|data| data.is_object()) else {
            return render_tool_output(&parsed);
        };
        if !parsed.success {
            return render_tool_output(&parsed);
        }
        let files = string_list(data, "matching_files");
        if files.is_empty() {
            return "No files found.".to_owned();
        }
        let mut text = files.join("\n");
        if data
            .get("truncated")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            text = format!(
                "{text}\n(Results truncated to the first {} files; use a more specific path or pattern.)",
                files.len()
            );
        }
        text
    }
}

/// Splits `pattern` around its first brace group.
///
/// Returns `(prefix, body, suffix)`, or `None` when there is no complete brace pair or the
/// body itself contains braces (a nested group, which is not expanded).
fn split_brace_group(pattern: &str) -> Option<(&str, &str, &str)> {
    let open = pattern.find('{')?;
    let close = pattern[open + 1..].find('}')? + open + 1;
    let body = &pattern[open + 1..close];
    if body.contains(['{', '}']) {
        return None;
    }
    Some((&pattern[..open], body, &pattern[close + 1..]))
}
