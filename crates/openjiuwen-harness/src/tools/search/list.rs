//! The `list_files` tool: lists one directory level through the sys-operation sandbox.
//!
//! Python parity: `openjiuwen/harness/tools/filesystem.py::ListDirTool`.

use std::sync::Arc;

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use openjiuwen_common::error::Result;
use openjiuwen_foundation::tool::{
    render_payload_text, render_tool_output, Tool, ToolCard, ToolContext, ToolOutput,
};
use openjiuwen_sysop::fs::types::{ListDirsParams, ListFilesData, ListFilesParams};
use openjiuwen_sysop::operation::FsOperation;

use super::super::descriptions::description;
use super::super::support::{parse_args, schema_for, tool_card, ToolEnv, ToolPayload};
use super::{deserialize_flag, string_list, MISSING_FS};

/// Arguments of the `list_files` tool.
///
/// Python parity: the `path`/`show_hidden` parameters of `ListDirTool.invoke`. A missing or
/// empty `path` lists the workspace root, and hidden entries are hidden by default.
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct ListDirArgs {
    /// Directory to list; defaults to the workspace root.
    #[serde(default)]
    #[schemars(with = "String")]
    pub path: Option<String>,
    /// Include entries whose name starts with `.`.
    #[serde(default, deserialize_with = "deserialize_flag")]
    #[schemars(with = "bool")]
    pub show_hidden: bool,
}

/// The `list_files` tool: lists one directory level through a [`FsOperation`] backend.
pub struct ListDirTool {
    /// Path resolution, permission gate and description language.
    env: ToolEnv,
    /// The filesystem backend, when one is configured.
    fs: Option<Arc<dyn FsOperation>>,
    /// The model-facing card built from [`ToolEnv::language`].
    card: ToolCard,
}

impl ListDirTool {
    /// Builds the tool over `env`, listing through `fs`.
    #[must_use]
    pub fn new(env: ToolEnv, fs: Option<Arc<dyn FsOperation>>) -> Self {
        let card = Self::card_for(env.language());
        Self { env, fs, card }
    }

    /// Builds the model-facing card in `language`.
    #[must_use]
    pub fn card_for(language: &str) -> ToolCard {
        tool_card(
            "list_files",
            description("list_files", language),
            schema_for::<ListDirArgs>(),
            true,
        )
    }

    /// Lists one directory level and returns the payload.
    async fn execute(&self, inputs: &Value) -> ToolPayload {
        let args: ListDirArgs = match parse_args(inputs) {
            Ok(args) => args,
            Err(payload) => return payload,
        };
        let base = match self
            .env
            .resolve_dir(
                "list_files",
                "path",
                args.path.as_deref().unwrap_or(""),
                inputs,
            )
            .await
        {
            Ok(base) => base,
            Err(payload) => return payload,
        };
        let Some(fs) = self.fs.as_ref() else {
            return ToolPayload::fail(MISSING_FS);
        };
        let base_string = base.to_string_lossy().into_owned();

        let files = fs
            .list_files(ListFilesParams::new(base_string.clone()))
            .await;
        if files.is_failure() {
            return ToolPayload::fail(format!("Failed to list files: {}", files.message));
        }
        let dirs = fs.list_directories(ListDirsParams::new(base_string)).await;
        if dirs.is_failure() {
            return ToolPayload::fail(format!("Failed to list directories: {}", dirs.message));
        }

        let mut files = entry_names(files.data, args.show_hidden);
        let mut dirs = entry_names(dirs.data, args.show_hidden);
        files.sort_unstable();
        dirs.sort_unstable();
        ToolPayload::ok(json!({"files": files, "dirs": dirs}))
    }
}

#[async_trait]
impl Tool for ListDirTool {
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
        let mut entries: Vec<String> = string_list(data, "dirs")
            .into_iter()
            .map(|name| format!("{name}/"))
            .collect();
        entries.extend(string_list(data, "files"));
        if entries.is_empty() {
            "No entries found.".to_owned()
        } else {
            entries.join("\n")
        }
    }
}

/// Maps a listing to entry names, dropping dotted entries unless `show_hidden`.
fn entry_names(data: Option<ListFilesData>, show_hidden: bool) -> Vec<String> {
    let Some(data) = data else {
        return Vec::new();
    };
    data.list_items
        .into_iter()
        .map(|item| item.name)
        .filter(|name| show_hidden || !name.starts_with('.'))
        .collect()
}
