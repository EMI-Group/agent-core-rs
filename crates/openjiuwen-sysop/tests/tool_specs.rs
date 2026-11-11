//! Ports tests/unit_tests/core/sys_operation/test_tool_adapter.py (generated tool specs).

#![allow(clippy::unwrap_used, clippy::expect_used)]
#![allow(clippy::doc_markdown)] // The required module doc names a Python test path verbatim.

use openjiuwen_sysop::{
    mode::{OperationKind, OperationMode},
    tool::OperationToolSpec,
    LocalWorkConfig, SysOperation,
};
use serde_json::{json, Value};
use tempfile::TempDir;

/// Every tool name rendered for a local card, in creation order.
const EXPECTED_NAMES: [&str; 13] = [
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
    "execute_cmd",
    "execute_cmd_background",
    "execute_code",
];

/// Builds a local service whose workspace is a fresh temporary directory.
fn local_service() -> (TempDir, SysOperation) {
    let dir = TempDir::new().unwrap();
    let workspace = dir.path().to_string_lossy().into_owned();
    let op = SysOperation::local(LocalWorkConfig::unrestricted(workspace)).unwrap();
    (dir, op)
}

/// The spec whose tool name is `name`.
fn spec_named<'a>(specs: &'a [OperationToolSpec], name: &str) -> &'a OperationToolSpec {
    specs
        .iter()
        .find(|spec| spec.name == name)
        .unwrap_or_else(|| panic!("missing tool spec {name}"))
}

/// The `properties` object of a spec's generated input schema.
fn schema_properties(spec: &OperationToolSpec) -> &serde_json::Map<String, Value> {
    spec.input_schema
        .get("properties")
        .and_then(Value::as_object)
        .unwrap_or_else(|| panic!("tool {} has no schema properties", spec.name))
}

/// Tool ids of `specs`, in order.
fn tool_ids(specs: &[OperationToolSpec]) -> Vec<String> {
    specs.iter().map(|spec| spec.tool_id.clone()).collect()
}

#[test]
fn exposes_thirteen_tool_specs() {
    let (_dir, op) = local_service();
    assert_eq!(
        op.tool_specs().len(),
        13,
        "a local card renders ten filesystem, two shell, and one code tool"
    );
}

#[test]
fn kind_partition_is_ten_fs_two_shell_one_code() {
    let (_dir, op) = local_service();
    assert_eq!(op.tool_specs_for(OperationKind::Fs).len(), 10);
    assert_eq!(op.tool_specs_for(OperationKind::Shell).len(), 2);
    assert_eq!(op.tool_specs_for(OperationKind::Code).len(), 1);

    let mut concatenated = tool_ids(&op.tool_specs_for(OperationKind::Fs));
    concatenated.extend(tool_ids(&op.tool_specs_for(OperationKind::Shell)));
    concatenated.extend(tool_ids(&op.tool_specs_for(OperationKind::Code)));
    assert_eq!(
        tool_ids(&op.tool_specs()),
        concatenated,
        "tool_specs() lists filesystem, then shell, then code tools"
    );
}

#[test]
fn tool_ids_have_the_card_kind_method_shape() {
    let (_dir, op) = local_service();
    let specs = op.tool_specs();
    for spec in &specs {
        let segments: Vec<&str> = spec.tool_id.split('.').collect();
        assert_eq!(
            segments.len(),
            3,
            "tool id {} must have three dot-separated segments",
            spec.tool_id
        );
        assert_eq!(segments[0], "sys_op", "card id segment in {}", spec.tool_id);
        assert_eq!(
            segments[1],
            spec.kind.as_str(),
            "kind segment in {}",
            spec.tool_id
        );
        assert_eq!(segments[2], spec.name, "method segment in {}", spec.tool_id);
    }
    assert_eq!(specs[0].tool_id, "sys_op.fs.read_file");
}

#[test]
fn tool_names_are_unique() {
    let (_dir, op) = local_service();
    let mut names: Vec<String> = op
        .tool_specs()
        .iter()
        .map(|spec| spec.name.clone())
        .collect();
    names.sort();
    names.dedup();
    assert_eq!(names.len(), 13, "tool names must be unique: {names:?}");
    for expected in EXPECTED_NAMES {
        assert!(
            names.iter().any(|name| name == expected),
            "missing tool name {expected}"
        );
    }
}

#[test]
fn every_input_schema_is_a_json_object() {
    let (_dir, op) = local_service();
    for spec in op.tool_specs() {
        assert!(
            spec.input_schema.is_object(),
            "schema of {} is not a JSON object",
            spec.name
        );
        assert_eq!(
            spec.input_schema["type"],
            json!("object"),
            "schema of {} must declare type = object",
            spec.name
        );
    }
}

#[test]
fn schemas_expose_expected_parameter_properties() {
    let (_dir, op) = local_service();
    let specs = op.tool_specs();

    let read = schema_properties(spec_named(&specs, "read_file"));
    for key in [
        "path",
        "mode",
        "head",
        "tail",
        "line_range",
        "encoding",
        "chunk_size",
    ] {
        assert!(read.contains_key(key), "read_file schema lacks {key}");
    }

    let write = schema_properties(spec_named(&specs, "write_file"));
    for key in [
        "path",
        "content",
        "prepend_newline",
        "append",
        "create_if_not_exist",
        "expected_content_sha256",
    ] {
        assert!(write.contains_key(key), "write_file schema lacks {key}");
    }

    let shell = schema_properties(spec_named(&specs, "execute_cmd"));
    for key in ["command", "cwd", "timeout_secs"] {
        assert!(shell.contains_key(key), "execute_cmd schema lacks {key}");
    }
}

#[test]
fn kind_filtering_matches_kind_field() {
    let (_dir, op) = local_service();
    let mut covered: Vec<String> = Vec::new();
    for kind in OperationKind::all() {
        for spec in op.tool_specs_for(kind) {
            assert_eq!(
                spec.kind, kind,
                "tool {} was filtered for {kind}",
                spec.tool_id
            );
            covered.push(spec.tool_id);
        }
    }
    covered.sort();
    let mut all = tool_ids(&op.tool_specs());
    all.sort();
    assert_eq!(
        covered, all,
        "the three filtered lists cover every tool once"
    );
    assert_eq!(covered.len(), 13);
}

#[test]
fn descriptions_are_non_empty() {
    let (_dir, op) = local_service();
    for spec in op.tool_specs() {
        assert!(
            !spec.description.trim().is_empty(),
            "tool {} has an empty description",
            spec.name
        );
    }
}

#[test]
fn local_service_reports_local_mode_and_no_isolation_key() {
    let (_dir, op) = local_service();
    assert_eq!(op.mode(), OperationMode::Local);
    assert_eq!(op.card().id, "sys_op");
    assert!(op.isolation_key_template().is_none());
    assert!(
        op.fs().is_some(),
        "local mode provides a filesystem backend"
    );
    assert!(op.shell().is_some(), "local mode provides a shell backend");
    assert!(op.code().is_some(), "local mode provides a code backend");
}
