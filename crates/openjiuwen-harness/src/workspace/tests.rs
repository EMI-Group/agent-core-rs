//! Cross-file unit tests for the workspace module: schema, builder, links, resolution, and
//! serde. Every test is offline and deterministic and uses a temporary directory.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::too_many_lines,
    clippy::float_cmp
)]

use std::path::{Path, PathBuf};

use openjiuwen_common::error::ErrorKind;
use openjiuwen_sysop::{LocalWorkConfig, SysOperation};
use openjiuwen_test_support::TempWorkspace;

use super::*;

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

/// Creates a temp workspace fixture.
fn temp_workspace() -> TempWorkspace {
    TempWorkspace::new().expect("temp workspace")
}

/// Creates a local sys-operation rooted at `root`.
fn local_operation(root: &Path) -> SysOperation {
    SysOperation::local(LocalWorkConfig::unrestricted(
        root.to_string_lossy().to_string(),
    ))
    .expect("local sys operation")
}

/// Builds a `DirectoryBuilder` over a local backend rooted at `root`.
fn builder(root: &Path) -> DirectoryBuilder {
    DirectoryBuilder::new(local_operation(root), root)
}

/// Reads a workspace file as UTF-8.
fn read(root: &Path, relative: &str) -> String {
    std::fs::read_to_string(root.join(relative))
        .unwrap_or_else(|error| panic!("read {relative}: {error}"))
}

/// The cn top-level node names, in schema order.
fn cn_names() -> Vec<&'static str> {
    vec![
        "AGENT.md",
        "SOUL.md",
        "HEARTBEAT.md",
        "IDENTITY.md",
        "USER.md",
        "memory",
        "coding_memory",
        "todo",
        "messages",
        "skills",
        "agents",
        "context",
    ]
}

/// The en top-level node names, in schema order.
fn en_names() -> Vec<&'static str> {
    vec![
        "AGENT.md",
        "SOUL.md",
        "HEARTBEAT.md",
        "IDENTITY.md",
        "USER.md",
        "memory",
        "todo",
        "messages",
        "skills",
        "agents",
        "context",
    ]
}

// ---------------------------------------------------------------------------
// schema
// ---------------------------------------------------------------------------

#[test]
fn cn_schema_has_twelve_top_level_nodes_in_order() {
    let schema = get_workspace_schema("cn");
    let names: Vec<&str> = schema.iter().map(|node| node.name.as_str()).collect();
    assert_eq!(names, cn_names());
    assert_eq!(schema.len(), 12);
    assert!(names.contains(&"coding_memory"));
    assert_eq!(schema[0].description.as_deref(), Some("基础配置和能力"));
    assert!(schema[0].is_file());
    assert_eq!(schema[0].path.as_deref(), Some("AGENT.md"));
}

#[test]
fn en_schema_has_eleven_nodes_and_no_coding_memory() {
    let schema = get_workspace_schema("en");
    let names: Vec<&str> = schema.iter().map(|node| node.name.as_str()).collect();
    assert_eq!(names, en_names());
    assert_eq!(schema.len(), 11);
    assert!(!names.contains(&"coding_memory"));

    let workspace = Workspace::for_language("/tmp/en-workspace", "en");
    assert!(workspace
        .get_node_path_for(WorkspaceNode::CodingMemory)
        .is_none());
    assert!(workspace.get_node_path_for(WorkspaceNode::Memory).is_some());
}

#[test]
fn cn_and_en_descriptions_differ() {
    let cn = get_workspace_schema("cn");
    let en = get_workspace_schema("en");
    assert_eq!(cn[0].description.as_deref(), Some("基础配置和能力"));
    let en_description = en[0].description.as_deref().unwrap();
    assert!(
        en_description.starts_with("Basic"),
        "unexpected description: {en_description}"
    );
    assert_ne!(cn[0].description, en[0].description);
    assert_eq!(cn[5].description.as_deref(), Some("记忆核心模块"));
    assert_eq!(en[5].description.as_deref(), Some("Memory core module"));
}

#[test]
fn packaged_content_is_bilingual_and_required_literals_are_present() {
    let cn_agent = default_content("cn", "AGENT.md");
    assert!(cn_agent.starts_with("# 智能体"));
    assert!(cn_agent.contains("首次运行"));
    assert!(cn_agent.contains("心跳"));

    let en_agent = default_content("en", "AGENT.md");
    assert!(en_agent.starts_with("# AGENT"));
    assert!(en_agent.contains("This folder is home."));
    assert!(en_agent.contains("## First Run"));

    let cn_soul = default_content("cn", "SOUL.md");
    assert!(cn_soul.contains("灵魂"));
    assert!(cn_soul.contains("## 身份定位"));

    let en_soul = default_content("en", "SOUL.md");
    assert!(en_soul.contains("SOUL"));
    assert!(en_soul.contains("genuinely helpful"));
    assert!(en_soul.contains("Have opinions."));
    assert!(!en_soul.contains("灵魂"));

    assert!(default_content("cn", "memory/MEMORY.md").contains("### 应该保存在这里的内容"));
    assert!(default_content("en", "memory/MEMORY.md").contains("### What should be saved here"));
    assert!(default_content("cn", "context/session_memory.md").contains("# 会话标题"));
    assert!(default_content("en", "context/session_memory.md").contains("# Session Title"));
    assert!(default_content("cn", "IDENTITY.md").contains("**名字：**"));
    assert!(default_content("en", "IDENTITY.md").contains("**Name:**"));
    assert!(default_content("cn", "USER.md").contains("# 用户"));
    assert!(default_content("en", "USER.md").contains("# USER"));
}

#[test]
fn schema_trees_are_independent_per_call() {
    let mut first = get_workspace_schema("cn");
    first[0].name = "MUTATED".to_string();
    first[5].children.clear();
    let second = get_workspace_schema("cn");
    assert_eq!(second[0].name, "AGENT.md");
    assert_eq!(second[5].children.len(), 2);
    assert_ne!(first, second);
}

#[test]
fn unknown_language_falls_back_to_cn_and_missing_resource_is_empty() {
    let unknown = get_workspace_schema("fr");
    let names: Vec<&str> = unknown.iter().map(|node| node.name.as_str()).collect();
    assert_eq!(names, cn_names());
    assert_eq!(get_workspace_schema("EN"), get_workspace_schema("cn"));
    assert_eq!(default_content("fr", "AGENT.md"), "");
    assert_eq!(default_content("cn", "does/not/exist.md"), "");
    assert_eq!(default_content("en", "coding_memory/MEMORY.md"), "");
}

#[test]
fn every_seeded_file_content_matches_the_packaged_resource() {
    let schema = get_workspace_schema("cn");
    let agent = schema.iter().find(|node| node.name == "AGENT.md").unwrap();
    assert_eq!(agent.default_content, default_content("cn", "AGENT.md"));
    let memory = schema.iter().find(|node| node.name == "memory").unwrap();
    let memory_md = memory
        .children
        .iter()
        .find(|node| node.name == "MEMORY.md")
        .unwrap();
    assert_eq!(
        memory_md.default_content,
        default_content("cn", "memory/MEMORY.md")
    );
    let coding = schema
        .iter()
        .find(|node| node.name == "coding_memory")
        .unwrap();
    assert_eq!(coding.children[0].default_content, "");
    let context = schema.iter().find(|node| node.name == "context").unwrap();
    assert_eq!(
        context.children[0].default_content,
        default_content("cn", "context/session_memory.md")
    );
    // Directory nodes never carry seeded content.
    for node in &schema {
        if !node.is_file() {
            assert!(node.default_content.is_empty(), "{}", node.name);
        }
    }
}

// ---------------------------------------------------------------------------
// WorkspaceNode and DirectoryNode
// ---------------------------------------------------------------------------

#[test]
fn workspace_node_strings_round_trip() {
    assert_eq!(WorkspaceNode::ALL.len(), 15);
    assert_eq!(WorkspaceNode::AgentMd.as_str(), "AGENT.md");
    assert_eq!(WorkspaceNode::CodingMemory.as_str(), "coding_memory");
    assert_eq!(WorkspaceNode::TeamLinks.as_str(), ".team");
    assert_eq!(WorkspaceNode::WorktreeLinks.as_str(), ".worktree");
    for node in WorkspaceNode::ALL {
        let text = node.as_str();
        assert_eq!(node.to_string(), text);
        assert_eq!(node.to_string().parse::<WorkspaceNode>().unwrap(), node);
        let borrowed: &'static str = node.into();
        assert_eq!(borrowed, text);
        assert_eq!(serde_json::to_value(node).unwrap(), serde_json::json!(text));
    }
    let error = "not-a-node".parse::<WorkspaceNode>().unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains("unknown workspace node"));
}

#[test]
fn directory_node_constructors_and_helpers() {
    let file = DirectoryNode::file("AGENT.md", "AGENT.md", "content")
        .with_description("base")
        .with_children(vec![DirectoryNode::dir("child", "child")]);
    assert!(file.is_file());
    assert_eq!(file.content(), "content");
    assert_eq!(file.description.as_deref(), Some("base"));
    assert_eq!(file.children.len(), 1);
    assert!(!file.children[0].is_file());
    assert_eq!(file.resolved_path(), "AGENT.md");

    let fallback = DirectoryNode {
        name: "memory".to_string(),
        ..DirectoryNode::default()
    };
    assert_eq!(fallback.resolved_path(), "memory");
    let empty_path = DirectoryNode::dir("memory", "");
    assert_eq!(empty_path.resolved_path(), "memory");
}

#[test]
fn directory_node_validation_reports_python_messages() {
    let empty = DirectoryNode::default();
    let error = empty.validate().unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert_eq!(
        error.message(),
        "Directory `name` must be a non-empty string."
    );

    let slash = DirectoryNode::dir("bad/name", "bad/name");
    let error = slash.validate().unwrap_err();
    assert!(error.message().contains("path separators"));
    assert!(error.message().contains("bad/name"));

    let backslash = DirectoryNode::dir("bad\\name", "bad\\name");
    assert!(backslash.validate().is_err());

    let nested = DirectoryNode::dir("memory", "memory").with_children(vec![empty_named("")]);
    let error = nested.validate().unwrap_err();
    assert_eq!(
        error.message(),
        "Directory `name` must be a non-empty string."
    );
}

/// Builds a node with an explicit (possibly invalid) name.
fn empty_named(name: &str) -> DirectoryNode {
    DirectoryNode {
        name: name.to_string(),
        ..DirectoryNode::default()
    }
}

// ---------------------------------------------------------------------------
// Workspace construction and lookup
// ---------------------------------------------------------------------------

#[test]
fn default_workspace_uses_cn_schema() {
    let workspace = Workspace::default();
    assert_eq!(workspace.root_path, PathBuf::from("./"));
    assert_eq!(workspace.language, "cn");
    assert_eq!(workspace.language(), "cn");
    assert!(!workspace.is_english());
    assert_eq!(workspace.directories, get_workspace_schema("cn"));
}

#[test]
fn for_language_selects_the_matching_schema() {
    let workspace = Workspace::for_language("/tmp/agent", "en");
    assert!(workspace.is_english());
    assert_eq!(workspace.language(), "en");
    assert_eq!(workspace.directories.len(), 11);
    assert!(workspace.validate().is_ok());

    let unknown = Workspace::for_language("/tmp/agent", "zh");
    assert_eq!(unknown.directories, get_workspace_schema("cn"));
    assert_eq!(unknown.language(), "zh");
}

#[test]
fn with_language_swaps_untouched_defaults() {
    let workspace = Workspace::new("/tmp/agent");
    let english = workspace.with_language("en");
    assert_eq!(english.language, "en");
    assert_eq!(english.directories, get_workspace_schema("en"));
    let chinese = english.with_language("cn");
    assert_eq!(chinese.directories, get_workspace_schema("cn"));
}

#[test]
fn with_language_preserves_custom_directories() {
    let mut workspace = Workspace::new("/tmp/agent");
    workspace
        .set_directory(DirectoryNode::dir("custom", "custom").with_description("keep me"))
        .unwrap();
    let english = workspace.with_language("en");
    assert_eq!(english.language, "en");
    // A customized schema is preserved, not reconciled: every cn default (including
    // `coding_memory`) stays and the custom node survives.
    assert_eq!(english.directories.len(), 13);
    let custom = english
        .directories
        .iter()
        .find(|node| node.name == "custom")
        .unwrap();
    assert_eq!(custom.description.as_deref(), Some("keep me"));
    assert!(english
        .directories
        .iter()
        .any(|node| node.name == "coding_memory"));
}

#[test]
fn ensure_default_schema_is_idempotent() {
    let mut workspace = Workspace::new("/tmp/agent");
    workspace.directories.clear();
    assert_eq!(workspace.ensure_default_schema(), 12);
    assert_eq!(workspace.ensure_default_schema(), 0);
    assert_eq!(workspace.directories.len(), 12);
}

#[test]
fn ensure_default_schema_appends_only_missing_nodes_and_caller_wins() {
    let mut workspace = Workspace::new("/tmp/agent");
    let custom_memory = DirectoryNode::dir("memory", "memory").with_description("custom memory");
    workspace.directories.retain(|node| node.name != "memory");
    workspace.directories.push(custom_memory);
    workspace.directories.retain(|node| node.name != "todo");

    assert_eq!(workspace.ensure_default_schema(), 1);
    assert_eq!(workspace.directories.last().unwrap().name, "todo");
    let memory = workspace
        .directories
        .iter()
        .find(|node| node.name == "memory")
        .unwrap();
    assert_eq!(memory.description.as_deref(), Some("custom memory"));
    assert!(memory.children.is_empty(), "children are never back-filled");
}

#[test]
fn with_directories_validates_and_supplements() {
    let custom = DirectoryNode::dir("custom", "custom");
    let workspace = Workspace::new("/tmp/agent")
        .with_directories(vec![custom])
        .unwrap();
    assert_eq!(workspace.directories.len(), 13);
    assert_eq!(workspace.directories[0].name, "custom");
    assert_eq!(workspace.directories.last().unwrap().name, "context");

    let invalid = DirectoryNode::dir("bad/name", "bad/name");
    let error = Workspace::new("/tmp/agent")
        .with_directories(vec![invalid])
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);

    let untouched = Workspace::for_language("/tmp/agent", "en")
        .with_directories(Vec::new())
        .unwrap();
    assert_eq!(untouched.directories.len(), 11);
}

#[test]
fn get_directory_searches_nested_nodes_and_falls_back() {
    let workspace = Workspace::new("/tmp/agent");
    assert_eq!(
        workspace.get_directory("AGENT.md"),
        Some("AGENT.md".to_string())
    );
    assert_eq!(
        workspace.get_directory("MEMORY.md"),
        Some("MEMORY.md".to_string())
    );
    assert_eq!(
        workspace.get_directory("daily_memory"),
        Some("daily_memory".to_string())
    );
    assert_eq!(
        workspace.get_directory_for(WorkspaceNode::Memory),
        Some("memory".to_string())
    );
    assert_eq!(workspace.get_directory("missing"), None);

    let mut customized = Workspace::new("/tmp/agent");
    customized
        .set_directory(DirectoryNode::dir("memory", "memory"))
        .unwrap();
    // The replaced node has no children, so the nested name comes from the defaults.
    assert_eq!(
        customized.get_directory("MEMORY.md"),
        Some("MEMORY.md".to_string())
    );

    let mut pathed = Workspace::new("/tmp/agent");
    pathed
        .set_directory(DirectoryNode {
            name: "alias".to_string(),
            path: Some("actual".to_string()),
            ..DirectoryNode::default()
        })
        .unwrap();
    assert_eq!(pathed.get_directory("alias"), Some("actual".to_string()));
}

#[test]
fn get_node_path_is_top_level_only() {
    let workspace = Workspace::new("/tmp/agent");
    assert_eq!(
        workspace.get_node_path("memory"),
        Some(PathBuf::from("/tmp/agent/memory"))
    );
    assert_eq!(
        workspace.get_node_path_for(WorkspaceNode::AgentMd),
        Some(PathBuf::from("/tmp/agent/AGENT.md"))
    );
    assert_eq!(
        workspace.get_node_path_for(WorkspaceNode::Skills),
        Some(PathBuf::from("/tmp/agent/skills"))
    );
    assert!(workspace.get_node_path("MEMORY.md").is_none());
    assert!(workspace.get_node_path("daily_memory").is_none());
    assert!(workspace
        .get_node_path_for(WorkspaceNode::MemoryMd)
        .is_none());
    assert!(workspace.get_node_path("missing").is_none());
}

#[test]
fn set_directory_upserts_and_validates() {
    let mut workspace = Workspace::new("/tmp/agent");
    workspace
        .set_directory(DirectoryNode::dir("AGENT.md", "AGENT.md").with_description("replaced"))
        .unwrap();
    assert_eq!(workspace.directories.len(), 12);
    let agent = workspace
        .directories
        .iter()
        .find(|node| node.name == "AGENT.md")
        .unwrap();
    assert_eq!(agent.description.as_deref(), Some("replaced"));

    workspace
        .set_directory(DirectoryNode::dir("custom", "custom"))
        .unwrap();
    assert_eq!(workspace.directories.len(), 13);
    assert_eq!(workspace.directories.last().unwrap().name, "custom");

    let error = workspace
        .set_directory(DirectoryNode::dir("", ""))
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    let error = workspace
        .set_directory(DirectoryNode::dir("bad/name", "bad/name"))
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert_eq!(workspace.directories.len(), 13);
}

#[test]
fn set_directories_replaces_and_validates_atomically() {
    let mut workspace = Workspace::new("/tmp/agent");
    workspace
        .set_directories(vec![DirectoryNode::dir("only", "only")])
        .unwrap();
    assert_eq!(workspace.directories.len(), 13);
    assert_eq!(workspace.directories[0].name, "only");

    let before = workspace.directories.clone();
    let error = workspace
        .set_directories(vec![
            DirectoryNode::dir("fine", "fine"),
            DirectoryNode::dir("bad name/", "bad name/"),
        ])
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert_eq!(workspace.directories, before);
}

#[test]
fn validate_is_recursive() {
    let mut workspace = Workspace::new("/tmp/agent");
    workspace.directories[5].children.push(empty_named(""));
    let error = workspace.validate().unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert_eq!(
        error.message(),
        "Directory `name` must be a non-empty string."
    );
}

// ---------------------------------------------------------------------------
// serde
// ---------------------------------------------------------------------------

#[test]
fn directory_node_json_omits_absent_optional_keys() {
    let minimal = DirectoryNode {
        name: "skills".to_string(),
        ..DirectoryNode::default()
    };
    assert_eq!(
        serde_json::to_value(&minimal).unwrap(),
        serde_json::json!({"name": "skills"})
    );

    let file = DirectoryNode::file("AGENT.md", "AGENT.md", "body").with_description("base");
    assert_eq!(
        serde_json::to_value(&file).unwrap(),
        serde_json::json!({
            "name": "AGENT.md",
            "path": "AGENT.md",
            "description": "base",
            "is_file": true,
            "default_content": "body",
        })
    );

    let parsed: DirectoryNode =
        serde_json::from_str(r#"{"name":"x","unknown_field":1,"children":[]}"#).unwrap();
    assert_eq!(parsed, minimal_named("x"));
}

/// Builds a node with only a name set.
fn minimal_named(name: &str) -> DirectoryNode {
    DirectoryNode {
        name: name.to_string(),
        ..DirectoryNode::default()
    }
}

#[test]
fn workspace_json_round_trips() {
    let mut workspace = Workspace::for_language("/tmp/agent", "en");
    workspace
        .set_directory(
            DirectoryNode::file("NOTES.md", "notes/NOTES.md", "hello")
                .with_description("custom note"),
        )
        .unwrap();
    let json = serde_json::to_string(&workspace).unwrap();
    let mut parsed: Workspace = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed, workspace);
    assert_eq!(parsed.ensure_default_schema(), 0);
}

#[test]
fn workspace_json_supplements_cn_defaults() {
    let mut workspace: Workspace = serde_json::from_str(r#"{"root_path":"/tmp/x"}"#).unwrap();
    assert_eq!(workspace.root_path, PathBuf::from("/tmp/x"));
    assert_eq!(workspace.language, "cn");
    assert_eq!(workspace.directories, get_workspace_schema("cn"));
    assert_eq!(workspace.ensure_default_schema(), 0);
}

#[test]
fn workspace_json_accepts_null_and_empty_fields() {
    let workspace: Workspace =
        serde_json::from_str(r#"{"root_path":null,"directories":[],"language":null}"#).unwrap();
    assert_eq!(workspace.root_path, PathBuf::from("./"));
    assert_eq!(workspace.language, "cn");
    assert_eq!(workspace.directories.len(), 12);

    let english: Workspace = serde_json::from_str(r#"{"language":"en","directories":[]}"#).unwrap();
    assert_eq!(english.directories, get_workspace_schema("en"));

    let empty: Workspace = serde_json::from_str("{}").unwrap();
    assert_eq!(empty, Workspace::default());
}

#[test]
fn workspace_json_rejects_invalid_directory_nodes() {
    let error = serde_json::from_str::<Workspace>(r#"{"directories":[{"name":"bad/name"}]}"#)
        .expect_err("invalid node");
    assert!(error.to_string().contains("path separators"), "{error}");

    let error = serde_json::from_str::<Workspace>(r#"{"directories":[{"name":""}]}"#)
        .expect_err("empty name");
    assert!(error.to_string().contains("non-empty string"), "{error}");
}

#[test]
fn json_schemas_are_generated_for_the_public_types() {
    let workspace = serde_json::to_value(schemars::schema_for!(Workspace)).unwrap();
    assert!(workspace.is_object());
    assert!(workspace.to_string().contains("root_path"));
    let node = serde_json::to_value(schemars::schema_for!(DirectoryNode)).unwrap();
    assert!(node.to_string().contains("default_content"));
    let link = serde_json::to_value(schemars::schema_for!(WorkspaceLink)).unwrap();
    assert!(link.to_string().contains("target"));
    let kinds = serde_json::to_value(schemars::schema_for!(LinkKind)).unwrap();
    assert!(kinds.to_string().contains("symlink"));
}

// ---------------------------------------------------------------------------
// DirectoryBuilder
// ---------------------------------------------------------------------------

#[tokio::test]
async fn build_seeds_the_full_cn_schema() {
    let temp = temp_workspace();
    let root = temp.workspace();
    let workspace = Workspace::for_language(&root, "cn");
    builder(&root)
        .build(&workspace.directories)
        .await
        .expect("build");

    assert!(read(&root, "AGENT.md").starts_with("# 智能体"));
    assert!(read(&root, "AGENT.md").contains("首次运行"));
    assert!(read(&root, "USER.md").contains("# 用户"));
    assert!(read(&root, "memory/MEMORY.md").contains("### 应该保存在这里的内容"));
    assert!(root.join("memory/daily_memory/.workspace").is_file());
    assert!(read(&root, "context/session_memory.md").contains("# 会话标题"));
    assert!(root.join("skills/.workspace").is_file());
    assert!(root.join("todo/.workspace").is_file());
    assert!(root.join("messages/.workspace").is_file());
    assert!(root.join("agents/.workspace").is_file());
    assert!(root.join("memory/.workspace").is_file());
    assert!(root.join("coding_memory/.workspace").is_file());
    // No leading newline: the packaged content is written verbatim.
    assert!(!read(&root, "AGENT.md").starts_with('\n'));
}

#[tokio::test]
async fn build_never_overwrites_existing_files() {
    let temp = temp_workspace();
    let root = temp.workspace();
    temp.write("workspace/AGENT.md", "hand written").unwrap();
    temp.write("workspace/memory/MEMORY.md", "custom memory")
        .unwrap();
    let workspace = Workspace::new(&root);
    builder(&root)
        .build(&workspace.directories)
        .await
        .expect("build");

    assert_eq!(read(&root, "AGENT.md"), "hand written");
    assert_eq!(read(&root, "memory/MEMORY.md"), "custom memory");
}

#[tokio::test]
async fn build_writes_zero_byte_content_for_empty_defaults() {
    let temp = temp_workspace();
    let root = temp.workspace();
    let workspace = Workspace::new(&root);
    builder(&root)
        .build(&workspace.directories)
        .await
        .expect("build");

    let metadata = std::fs::metadata(root.join("coding_memory/MEMORY.md")).unwrap();
    assert_eq!(metadata.len(), 0, "empty content is a zero-byte file");
    let marker = std::fs::metadata(root.join("skills/.workspace")).unwrap();
    assert_eq!(marker.len(), 0);
}

#[tokio::test]
async fn build_is_idempotent() {
    let temp = temp_workspace();
    let root = temp.workspace();
    let workspace = Workspace::for_language(&root, "en");
    let builder = builder(&root);
    builder
        .build(&workspace.directories)
        .await
        .expect("first build");
    builder
        .build(&workspace.directories)
        .await
        .expect("second build");
    assert!(read(&root, "AGENT.md").contains("This folder is home."));
    assert!(root.join("skills/.workspace").is_file());
}

#[tokio::test]
async fn build_materializes_nested_custom_directories() {
    let temp = temp_workspace();
    let root = temp.workspace();
    let directories = vec![DirectoryNode::dir("project", "project").with_children(vec![
        DirectoryNode::dir("src", "src").with_children(vec![DirectoryNode::file(
            "main.rs",
            "main.rs",
            "fn main() {}\n",
        )]),
    ])];
    builder(&root).build(&directories).await.expect("build");
    assert!(root.join("project/.workspace").is_file());
    assert!(root.join("project/src/.workspace").is_file());
    assert_eq!(read(&root, "project/src/main.rs"), "fn main() {}\n");
}

#[tokio::test]
async fn build_rejects_unsafe_paths_before_io() {
    let temp = temp_workspace();
    let root = temp.workspace();
    let escaping = vec![DirectoryNode::file("escape.txt", "../escape.txt", "nope")];
    let error = builder(&root)
        .build(&escaping)
        .await
        .expect_err("unsafe path");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(error.message().contains("Unsafe path detected"), "{error}");
    assert!(!temp.exists("escape.txt"));
    assert!(!root.join("escape.txt").exists());

    let nested = vec![DirectoryNode::dir("project", "project")
        .with_children(vec![DirectoryNode::dir("bad", "../../escape")
            .with_children(vec![DirectoryNode::file("x.txt", "x.txt", "x")])])];
    let error = builder(&root).build(&nested).await.expect_err("nested");
    assert_eq!(error.kind(), ErrorKind::Validation);
    assert!(!temp.path("escape").exists());
    assert!(!root.join("project/x.txt").exists());
}

#[test]
fn is_safe_relative_path_table() {
    for unsafe_path in [
        "",
        "/abs",
        "\\abs",
        "C:\\x",
        "C:x",
        "c:/x",
        "\\\\srv\\share",
        "..",
        "a/../../b",
        "x/..",
        ".",
        "a/\0/b",
    ] {
        assert!(
            !is_safe_relative_path(unsafe_path),
            "expected unsafe: {unsafe_path:?}"
        );
    }
    for safe_path in ["a/b", "./a/b", "a\\b", ".hidden", "a/../b", "a/b.", "x"] {
        assert!(
            is_safe_relative_path(safe_path),
            "expected safe: {safe_path:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// links
// ---------------------------------------------------------------------------

#[test]
fn link_team_creates_a_listed_link() {
    let temp = temp_workspace();
    let root = temp.workspace();
    let target = temp.create_dir("teams/alpha").unwrap();
    let workspace = Workspace::new(&root);

    let link = workspace.link_team("team_abc", &target).expect("link team");
    assert_eq!(link, root.join(".team/team_abc"));
    assert!(link.exists());

    #[cfg(unix)]
    {
        let metadata = std::fs::symlink_metadata(&link).unwrap();
        assert!(metadata.file_type().is_symlink());
    }
    let resolved = std::fs::canonicalize(&link).unwrap();
    assert_eq!(resolved, std::fs::canonicalize(&target).unwrap());

    let links = workspace.list_team_links();
    assert_eq!(links.len(), 1);
    assert_eq!(links[0].name, "team_abc");
    assert!(matches!(
        links[0].kind,
        LinkKind::Symlink | LinkKind::Directory
    ));
    assert_eq!(
        std::fs::canonicalize(&links[0].target).unwrap(),
        std::fs::canonicalize(&target).unwrap()
    );
}

#[test]
fn relinking_is_idempotent() {
    let temp = temp_workspace();
    let root = temp.workspace();
    let target = temp.create_dir("teams/alpha").unwrap();
    let workspace = Workspace::new(&root);
    let first = workspace.link_team("team_abc", &target).unwrap();
    let second = workspace.link_team("team_abc", &target).unwrap();
    assert_eq!(first, second);
    assert_eq!(workspace.list_team_links().len(), 1);
}

#[test]
fn list_team_links_is_sorted_by_name() {
    let temp = temp_workspace();
    let root = temp.workspace();
    let workspace = Workspace::new(&root);
    for name in ["gamma", "alpha", "beta"] {
        let target = temp.create_dir(format!("teams/{name}")).unwrap();
        workspace.link_team(name, &target).unwrap();
    }
    let names: Vec<String> = workspace
        .list_team_links()
        .into_iter()
        .map(|link| link.name)
        .collect();
    assert_eq!(names, vec!["alpha", "beta", "gamma"]);
}

#[test]
fn unlink_team_removes_only_the_link() {
    let temp = temp_workspace();
    let root = temp.workspace();
    let target = temp.create_dir("teams/alpha").unwrap();
    temp.write("teams/alpha/AGENT.md", "team content").unwrap();
    let workspace = Workspace::new(&root);
    workspace.link_team("team_abc", &target).unwrap();

    assert!(workspace.unlink_team("team_abc").unwrap());
    assert!(!root.join(".team/team_abc").exists());
    assert!(target.exists(), "the link target is untouched");
    assert_eq!(read(&target, "AGENT.md"), "team content");
    assert_eq!(workspace.list_team_links(), Vec::new());
    assert!(!workspace.unlink_team("team_abc").unwrap());
}

#[test]
fn worktree_links_mirror_team_links() {
    let temp = temp_workspace();
    let root = temp.workspace();
    let target = temp.create_dir("worktrees/feature").unwrap();
    let workspace = Workspace::new(&root);

    let link = workspace.link_worktree("feature", &target).unwrap();
    assert_eq!(link, root.join(".worktree/feature"));
    let links = workspace.list_worktree_links();
    assert_eq!(links.len(), 1);
    assert_eq!(links[0].name, "feature");
    assert_eq!(workspace.list_team_links(), Vec::new());
    assert!(workspace.unlink_worktree("feature").unwrap());
    assert_eq!(workspace.list_worktree_links(), Vec::new());
}

#[test]
fn foreign_files_are_neither_listed_nor_deleted() {
    let temp = temp_workspace();
    let root = temp.workspace();
    std::fs::create_dir_all(root.join(".team")).unwrap();
    temp.write("workspace/.team/notes.txt", "keep").unwrap();
    let workspace = Workspace::new(&root);

    assert_eq!(workspace.list_team_links(), Vec::new());
    assert!(!workspace.unlink_team("notes.txt").unwrap());
    assert_eq!(read(&root, ".team/notes.txt"), "keep");

    std::fs::create_dir_all(root.join(".team/plain-dir")).unwrap();
    assert_eq!(
        workspace.list_team_links(),
        Vec::new(),
        "no marker, no link"
    );
}

#[test]
fn unsafe_link_names_are_rejected() {
    let temp = temp_workspace();
    let root = temp.workspace();
    let target = temp.create_dir("teams/alpha").unwrap();
    let workspace = Workspace::new(&root);

    for name in ["../escape", "a/b", "a\\b", "", ".", ".."] {
        let error = workspace.link_team(name, &target).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation, "name {name:?}");
        assert!(error.message().contains("invalid link name"), "{error}");
    }
    assert!(!temp.path("escape").exists());
    assert!(workspace.unlink_team("../escape").is_err());
}

#[test]
fn missing_link_target_is_not_found() {
    let temp = temp_workspace();
    let root = temp.workspace();
    let workspace = Workspace::new(&root);
    let error = workspace
        .link_team("team_abc", temp.path("teams/missing"))
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::NotFound);
}

#[cfg(unix)]
#[test]
fn dangling_links_are_listed_from_the_read_link_fallback() {
    let temp = temp_workspace();
    let root = temp.workspace();
    std::fs::create_dir_all(root.join(".team")).unwrap();
    std::os::unix::fs::symlink("missing-target", root.join(".team/dangling")).unwrap();
    let workspace = Workspace::new(&root);

    let links = workspace.list_team_links();
    assert_eq!(links.len(), 1);
    assert_eq!(links[0].name, "dangling");
    assert_eq!(links[0].kind, LinkKind::Symlink);
    assert_eq!(links[0].target, root.join(".team/missing-target"));
}

#[test]
fn copy_fallback_links_are_listed_via_the_marker() {
    let temp = temp_workspace();
    let root = temp.workspace();
    let target = temp.create_dir("teams/alpha").unwrap();
    temp.write("workspace/.team/copied/content.txt", "copied")
        .unwrap();
    temp.write(
        "workspace/.team/copied/.workspace-link",
        format!("{}\n", target.display()),
    )
    .unwrap();
    let workspace = Workspace::new(&root);

    let links = workspace.list_team_links();
    assert_eq!(links.len(), 1);
    assert_eq!(links[0].name, "copied");
    assert_eq!(links[0].kind, LinkKind::Directory);
    assert_eq!(links[0].target, target);
    assert_eq!(read(&root, ".team/copied/content.txt"), "copied");

    assert!(workspace.unlink_team("copied").unwrap());
    assert!(!root.join(".team/copied").exists());
}

// ---------------------------------------------------------------------------
// resolution
// ---------------------------------------------------------------------------

#[tokio::test]
async fn resolve_keeps_paths_inside_the_workspace() {
    let temp = temp_workspace();
    let root = temp.workspace();
    let workspace = Workspace::new(&root);
    let canonical_root = std::fs::canonicalize(&root).unwrap();

    let resolved = workspace.resolve("memory/MEMORY.md").await.unwrap();
    assert!(resolved.starts_with(&canonical_root));
    assert!(resolved.ends_with("MEMORY.md"));

    let dotted = workspace.resolve("./AGENT.md").await.unwrap();
    assert_eq!(dotted, canonical_root.join("AGENT.md"));
}

#[tokio::test]
async fn resolve_rejects_unsafe_paths() {
    let temp = temp_workspace();
    let root = temp.workspace();
    let workspace = Workspace::new(&root);
    for path in ["../x", "/abs", "C:/x", "", "a/../../b"] {
        let error = workspace.resolve(path).await.unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation, "path {path:?}");
        assert!(error.message().contains("unsafe workspace path"), "{error}");
    }
}

#[tokio::test]
async fn resolve_existing_requires_the_path_to_exist() {
    let temp = temp_workspace();
    let root = temp.workspace();
    temp.write("workspace/memory/MEMORY.md", "body").unwrap();
    let workspace = Workspace::new(&root);

    let resolved = workspace
        .resolve_existing("memory/MEMORY.md")
        .await
        .unwrap();
    assert_eq!(
        resolved,
        std::fs::canonicalize(root.join("memory/MEMORY.md")).unwrap()
    );

    let error = workspace
        .resolve_existing("memory/missing.md")
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::NotFound);
    assert!(error.message().contains("memory/missing.md"));
}

#[cfg(unix)]
#[tokio::test]
async fn resolve_rejects_symlink_escapes() {
    let temp = temp_workspace();
    let root = temp.workspace();
    let outside = temp.create_dir("outside").unwrap();
    temp.write("outside/secret.txt", "secret").unwrap();
    std::os::unix::fs::symlink(&outside, root.join("link")).unwrap();
    let workspace = Workspace::new(&root);

    let error = workspace.resolve("link/secret.txt").await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Security);
    assert!(
        error.message().contains("escapes workspace root"),
        "{error}"
    );
    assert!(workspace.resolve("link").await.is_err());
}
