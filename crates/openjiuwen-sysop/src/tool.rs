//! Tool specifications generated from operation parameter types.
//!
//! Python parity: `SysOperationToolAdapter.extract_tools` and `BaseOperation`
//! (`core/sys_operation/tool_adapter.py`, `base.py`) render one tool per registered
//! operation, with the id `"<card>.<kind>.<method>"` and a JSON Schema built from the
//! Python signature. Rust builds the schema from the parameter struct via `schemars`, so
//! the schema and the deserializer can never disagree.
//!
//! Tool ids and names are Python-compatible; the local filesystem surface contributes
//! exactly ten tools, so an agent sees the same tool set it sees in Python. Streaming
//! variants and the Rust-only extension operations are not rendered as tools.
//!
//! ```
//! use openjiuwen_sysop::tool::fs_tool_specs;
//!
//! let specs = fs_tool_specs("sys_op");
//! assert_eq!(specs.len(), 10);
//! assert_eq!(specs[0].tool_id, "sys_op.fs.read_file");
//! assert!(specs[0].input_schema.get("properties").is_some());
//! ```

use schemars::schema_for;
use serde::Serialize;
use serde_json::Value;

use crate::{
    code::types::CodeExecParams,
    fs::types::{
        DownloadFileParams, ListDirsParams, ListFilesParams, ReadFileParams, SearchFilesParams,
        UploadFileParams, WriteFileParams,
    },
    mode::OperationKind,
    shell::types::ShellExecParams,
};

/// A tool rendered from one operation method.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct OperationToolSpec {
    /// Tool id: `"<card>.<kind>.<method>"`.
    pub tool_id: String,
    /// Tool name as seen by an agent (`"read_file"`).
    pub name: String,
    /// Human-readable description.
    pub description: String,
    /// Which operation surface the tool belongs to.
    pub kind: OperationKind,
    /// JSON Schema of the tool input, generated from the parameter struct.
    pub input_schema: Value,
}

impl OperationToolSpec {
    /// Builds a spec for `method` of `kind`, deriving the schema from `P`.
    #[must_use]
    pub fn new<P: schemars::JsonSchema>(
        card_id: &str,
        kind: OperationKind,
        method: &str,
        description: &str,
    ) -> Self {
        Self {
            tool_id: crate::card::generate_tool_id(card_id, kind, method),
            name: method.to_string(),
            description: description.to_string(),
            kind,
            input_schema: schema_value::<P>(),
        }
    }
}

/// Renders the JSON Schema of `P` as a JSON value.
///
/// A schema that cannot be serialized degrades to the empty object schema (the tool stays
/// callable, unvalidated) instead of panicking; the degradation is logged with the type name
/// so an unusable tool schema is not silent.
#[must_use]
pub fn schema_value<P: schemars::JsonSchema>() -> Value {
    let schema = schema_for!(P);
    serde_json::to_value(schema).unwrap_or_else(|error| {
        tracing::warn!(
            schema_type = std::any::type_name::<P>(),
            %error,
            "the tool parameter schema could not be serialized; using an empty object schema"
        );
        Value::Object(serde_json::Map::new())
    })
}

/// Tool specs for the ten filesystem operations.
#[must_use]
pub fn fs_tool_specs(card_id: &str) -> Vec<OperationToolSpec> {
    vec![
        OperationToolSpec::new::<ReadFileParams>(
            card_id,
            OperationKind::Fs,
            "read_file",
            "Read a file as text or bytes, optionally sliced by head, tail, or line range.",
        ),
        OperationToolSpec::new::<WriteFileParams>(
            card_id,
            OperationKind::Fs,
            "write_file",
            "Write or append content to a file, optionally guarded by a content hash.",
        ),
        OperationToolSpec::new::<UploadFileParams>(
            card_id,
            OperationKind::Fs,
            "upload_file",
            "Copy a host file into the workspace.",
        ),
        OperationToolSpec::new::<DownloadFileParams>(
            card_id,
            OperationKind::Fs,
            "download_file",
            "Copy a workspace file to the host.",
        ),
        OperationToolSpec::new::<ListFilesParams>(
            card_id,
            OperationKind::Fs,
            "list_files",
            "List files in a directory, optionally recursive and filtered by suffix.",
        ),
        OperationToolSpec::new::<ListDirsParams>(
            card_id,
            OperationKind::Fs,
            "list_directories",
            "List directories under a directory, optionally recursive.",
        ),
        OperationToolSpec::new::<SearchFilesParams>(
            card_id,
            OperationKind::Fs,
            "search_files",
            "Search for files matching a glob pattern below a directory.",
        ),
    ]
    .into_iter()
    .chain([
        OperationToolSpec::new::<ReadFileParams>(
            card_id,
            OperationKind::Fs,
            "read_file_stream",
            "Stream a file as text lines or byte chunks.",
        ),
        OperationToolSpec::new::<UploadFileParams>(
            card_id,
            OperationKind::Fs,
            "upload_file_stream",
            "Stream a host-to-workspace copy.",
        ),
        OperationToolSpec::new::<DownloadFileParams>(
            card_id,
            OperationKind::Fs,
            "download_file_stream",
            "Stream a workspace-to-host copy.",
        ),
    ])
    .collect()
}

/// Tool specs for the shell operations that return a complete result.
#[must_use]
pub fn shell_tool_specs(card_id: &str) -> Vec<OperationToolSpec> {
    vec![
        OperationToolSpec::new::<ShellExecParams>(
            card_id,
            OperationKind::Shell,
            "execute_cmd",
            "Run a shell command in the workspace and return exit status, stdout, and stderr.",
        ),
        OperationToolSpec::new::<ShellExecParams>(
            card_id,
            OperationKind::Shell,
            "execute_cmd_background",
            "Start a shell command without waiting and return its process id.",
        ),
    ]
}

/// Tool specs for the code operations that return a complete result.
#[must_use]
pub fn code_tool_specs(card_id: &str) -> Vec<OperationToolSpec> {
    vec![OperationToolSpec::new::<CodeExecParams>(
        card_id,
        OperationKind::Code,
        "execute_code",
        "Run a Python or JavaScript snippet in a subprocess and return its output.",
    )]
}

/// Every tool spec of a card, in filesystem/shell/code order.
#[must_use]
pub fn all_tool_specs(card_id: &str) -> Vec<OperationToolSpec> {
    let mut specs = fs_tool_specs(card_id);
    specs.extend(shell_tool_specs(card_id));
    specs.extend(code_tool_specs(card_id));
    specs
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn filesystem_surface_has_ten_tools_with_prefixed_ids() {
        let specs = fs_tool_specs("sys_op");
        assert_eq!(specs.len(), 10);
        assert_eq!(specs[0].tool_id, "sys_op.fs.read_file");
        assert_eq!(specs[0].name, "read_file");
        assert!(specs
            .iter()
            .all(|spec| spec.tool_id.starts_with("sys_op.fs.")));
        let names: Vec<&str> = specs.iter().map(|spec| spec.name.as_str()).collect();
        for expected in [
            "read_file",
            "write_file",
            "upload_file",
            "download_file",
            "list_files",
            "list_directories",
            "search_files",
            "read_file_stream",
            "upload_file_stream",
            "download_file_stream",
        ] {
            assert!(names.contains(&expected), "missing tool {expected}");
        }
    }

    #[test]
    fn schemas_describe_the_parameters() {
        let specs = fs_tool_specs("c");
        let read = &specs[0];
        let properties = read.input_schema["properties"].as_object().unwrap();
        assert!(properties.contains_key("path"));
        assert!(properties.contains_key("mode"));
        assert!(properties.contains_key("line_range"));
    }

    #[test]
    fn all_specs_cover_every_kind() {
        let specs = all_tool_specs("c");
        assert_eq!(specs.len(), 13);
        assert_eq!(
            specs.iter().filter(|s| s.kind == OperationKind::Fs).count(),
            10
        );
        assert_eq!(
            specs
                .iter()
                .filter(|s| s.kind == OperationKind::Shell)
                .count(),
            2
        );
        assert_eq!(
            specs
                .iter()
                .filter(|s| s.kind == OperationKind::Code)
                .count(),
            1
        );
    }
}
