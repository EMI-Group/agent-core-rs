//! The default bilingual workspace schema and its packaged content.
//!
//! Python keeps the schema topology in `workspace/directory_builder.py` and the seed
//! markdown in `harness/prompts/workspace_content/{cn,en}/**`. Here the topology lives in
//! [`get_workspace_schema`] and the markdown is compiled in through [`default_content`]
//! (`include_str!`), so a schema is always paired with its own resources and an unknown
//! language or resource name yields an empty string instead of an error.

use crate::workspace::node::DirectoryNode;

/// Returns a fresh default schema tree for `language`.
///
/// Only the exact string `"en"` selects the English schema; every other value (including
/// `"EN"`, `"zh"`, and the empty string) selects the Chinese schema, matching Python's
/// `language == "en"` test. A new tree is built on every call, so a caller may mutate the
/// result without affecting any other workspace.
#[must_use]
pub fn get_workspace_schema(language: &str) -> Vec<DirectoryNode> {
    if language == "en" {
        english_schema()
    } else {
        chinese_schema()
    }
}

/// Returns the packaged seed content for `relative_path` under `language`.
///
/// An unknown language or path returns `""` (never an error, and never a cross-language
/// fallback), which is what lets a schema node legitimately seed an empty file.
#[must_use]
pub fn default_content(language: &str, relative_path: &str) -> &'static str {
    match (language, relative_path) {
        ("cn", "AGENT.md") => include_str!("resources/cn/AGENT.md"),
        ("cn", "SOUL.md") => include_str!("resources/cn/SOUL.md"),
        ("cn", "HEARTBEAT.md") => include_str!("resources/cn/HEARTBEAT.md"),
        ("cn", "IDENTITY.md") => include_str!("resources/cn/IDENTITY.md"),
        ("cn", "USER.md") => include_str!("resources/cn/USER.md"),
        ("cn", "memory/MEMORY.md") => include_str!("resources/cn/memory/MEMORY.md"),
        ("cn", "context/session_memory.md") => {
            include_str!("resources/cn/context/session_memory.md")
        }
        ("en", "AGENT.md") => include_str!("resources/en/AGENT.md"),
        ("en", "SOUL.md") => include_str!("resources/en/SOUL.md"),
        ("en", "HEARTBEAT.md") => include_str!("resources/en/HEARTBEAT.md"),
        ("en", "IDENTITY.md") => include_str!("resources/en/IDENTITY.md"),
        ("en", "USER.md") => include_str!("resources/en/USER.md"),
        ("en", "memory/MEMORY.md") => include_str!("resources/en/memory/MEMORY.md"),
        ("en", "context/session_memory.md") => {
            include_str!("resources/en/context/session_memory.md")
        }
        _ => "",
    }
}

/// The Chinese schema: twelve top-level nodes, including `coding_memory`.
fn chinese_schema() -> Vec<DirectoryNode> {
    vec![
        seed_file("AGENT.md", "基础配置和能力", "cn/AGENT.md"),
        seed_file("SOUL.md", "人格、性格和价值观", "cn/SOUL.md"),
        seed_file("HEARTBEAT.md", "心跳日志和状态记录", "cn/HEARTBEAT.md"),
        seed_file("IDENTITY.md", "身份凭证和权限", "cn/IDENTITY.md"),
        seed_file("USER.md", "用户数据目录", "cn/USER.md"),
        DirectoryNode::dir("memory", "memory")
            .with_description("记忆核心模块")
            .with_children(vec![
                seed_file("MEMORY.md", "长期记忆索引和摘要", "cn/memory/MEMORY.md"),
                DirectoryNode::dir("daily_memory", "daily_memory")
                    .with_description("每日结构化记忆"),
            ]),
        DirectoryNode::dir("coding_memory", "coding_memory")
            .with_description("Coding Agent 记忆模块")
            .with_children(vec![DirectoryNode::file("MEMORY.md", "MEMORY.md", "")
                .with_description("Coding 记忆索引")]),
        DirectoryNode::dir("todo", "todo").with_description("待办事项目录"),
        DirectoryNode::dir("messages", "messages").with_description("消息历史目录"),
        DirectoryNode::dir("skills", "skills").with_description("技能库目录"),
        DirectoryNode::dir("agents", "agents").with_description("子智能体嵌套目录"),
        DirectoryNode::dir("context", "context")
            .with_description("上下文offload以及session memory目录")
            .with_children(vec![seed_file(
                "session_memory.md",
                "session memory模版",
                "cn/context/session_memory.md",
            )]),
    ]
}

/// The English schema: eleven top-level nodes (there is no `coding_memory`).
fn english_schema() -> Vec<DirectoryNode> {
    vec![
        seed_file(
            "AGENT.md",
            "Basic agent configuration and capabilities",
            "en/AGENT.md",
        ),
        seed_file(
            "SOUL.md",
            "Agent personality, character, values, and behavioral guidelines",
            "en/SOUL.md",
        ),
        seed_file(
            "HEARTBEAT.md",
            "Heartbeat log / status recording",
            "en/HEARTBEAT.md",
        ),
        seed_file(
            "IDENTITY.md",
            "Identity credentials, unique identifier, and permission information",
            "en/IDENTITY.md",
        ),
        seed_file("USER.md", "User data directory", "en/USER.md"),
        DirectoryNode::dir("memory", "memory")
            .with_description("Memory core module")
            .with_children(vec![
                seed_file(
                    "MEMORY.md",
                    "Memory overview, index, and important memory summaries",
                    "en/memory/MEMORY.md",
                ),
                DirectoryNode::dir("daily_memory", "daily_memory")
                    .with_description("Daily structured memory"),
            ]),
        DirectoryNode::dir("todo", "todo").with_description("Todo items"),
        DirectoryNode::dir("messages", "messages").with_description("Message history module"),
        DirectoryNode::dir("skills", "skills").with_description("Skills library directory"),
        DirectoryNode::dir("agents", "agents").with_description("Sub-agent nesting directory"),
        DirectoryNode::dir("context", "context")
            .with_description("Context offload and session memory file")
            .with_children(vec![seed_file(
                "session_memory.md",
                "session memory template",
                "en/context/session_memory.md",
            )]),
    ]
}

/// Builds a file node whose `name` and `path` are the same literal name.
fn seed_file(name: &str, description: &str, resource: &str) -> DirectoryNode {
    let language = if resource.starts_with("en/") {
        "en"
    } else {
        "cn"
    };
    DirectoryNode::file(name, name, default_content(language, &resource[3..]))
        .with_description(description)
}
