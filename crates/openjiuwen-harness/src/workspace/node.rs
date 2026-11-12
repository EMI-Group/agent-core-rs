//! Named workspace nodes and directory-schema entries.
//!
//! [`WorkspaceNode`] is the fixed vocabulary of well-known names inside a workspace
//! (Python `WorkspaceNode`), and [`DirectoryNode`] is one node of the workspace schema
//! tree. Python models a directory node as `Dict[str, Any]`; Rust uses a typed struct and
//! reproduces Python's validation messages through
//! [`Error::validation`](openjiuwen_common::error::Error::validation).

use std::fmt;
use std::str::FromStr;

use openjiuwen_common::error::{Error, Result};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// A well-known name inside a workspace.
///
/// The string values are the Python enum values (`AGENT.md`, `memory`, `coding_memory`,
/// `.team`, …), so a serialized node can be compared with a persisted Python workspace.
/// Use [`WorkspaceNode::ALL`] to iterate the vocabulary in Python declaration order.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, JsonSchema,
)]
pub enum WorkspaceNode {
    /// `AGENT.md` — the agent's base configuration and capabilities.
    #[serde(rename = "AGENT.md")]
    AgentMd,
    /// `SOUL.md` — personality, character, and values.
    #[serde(rename = "SOUL.md")]
    SoulMd,
    /// `HEARTBEAT.md` — heartbeat log and periodic task checklist.
    #[serde(rename = "HEARTBEAT.md")]
    HeartbeatMd,
    /// `IDENTITY.md` — identity credentials and permissions.
    #[serde(rename = "IDENTITY.md")]
    IdentityMd,
    /// `USER.md` — user data and preferences.
    #[serde(rename = "USER.md")]
    UserMd,
    /// `memory` — the memory core module.
    #[serde(rename = "memory")]
    Memory,
    /// `coding_memory` — the coding-agent memory module (Chinese schema only).
    #[serde(rename = "coding_memory")]
    CodingMemory,
    /// `todo` — todo items.
    #[serde(rename = "todo")]
    Todo,
    /// `messages` — message history.
    #[serde(rename = "messages")]
    Messages,
    /// `skills` — the skills library.
    #[serde(rename = "skills")]
    Skills,
    /// `agents` — nested sub-agent configurations.
    #[serde(rename = "agents")]
    Agents,
    /// `MEMORY.md` — the long-term memory index.
    #[serde(rename = "MEMORY.md")]
    MemoryMd,
    /// `daily_memory` — daily structured memory.
    #[serde(rename = "daily_memory")]
    DailyMemory,
    /// `.team` — links to team workspaces.
    #[serde(rename = ".team")]
    TeamLinks,
    /// `.worktree` — links to git worktrees.
    #[serde(rename = ".worktree")]
    WorktreeLinks,
}

impl WorkspaceNode {
    /// Every node, in Python declaration order.
    pub const ALL: [Self; 15] = [
        Self::AgentMd,
        Self::SoulMd,
        Self::HeartbeatMd,
        Self::IdentityMd,
        Self::UserMd,
        Self::Memory,
        Self::CodingMemory,
        Self::Todo,
        Self::Messages,
        Self::Skills,
        Self::Agents,
        Self::MemoryMd,
        Self::DailyMemory,
        Self::TeamLinks,
        Self::WorktreeLinks,
    ];

    /// The Python string value of this node.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::AgentMd => "AGENT.md",
            Self::SoulMd => "SOUL.md",
            Self::HeartbeatMd => "HEARTBEAT.md",
            Self::IdentityMd => "IDENTITY.md",
            Self::UserMd => "USER.md",
            Self::Memory => "memory",
            Self::CodingMemory => "coding_memory",
            Self::Todo => "todo",
            Self::Messages => "messages",
            Self::Skills => "skills",
            Self::Agents => "agents",
            Self::MemoryMd => "MEMORY.md",
            Self::DailyMemory => "daily_memory",
            Self::TeamLinks => ".team",
            Self::WorktreeLinks => ".worktree",
        }
    }
}

impl fmt::Display for WorkspaceNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for WorkspaceNode {
    type Err = Error;

    fn from_str(input: &str) -> Result<Self> {
        Self::ALL
            .into_iter()
            .find(|node| node.as_str() == input)
            .ok_or_else(|| Error::validation(format!("unknown workspace node: {input}")))
    }
}

impl From<WorkspaceNode> for &'static str {
    fn from(node: WorkspaceNode) -> Self {
        node.as_str()
    }
}

/// One node of the workspace schema tree (Python `DirectoryNode` dictionary).
///
/// A node is either a file (`is_file`, seeded with `default_content` when absent) or a
/// directory (identified on disk by a `.workspace` marker) and may carry children. Every
/// optional key is omitted from the JSON when absent, so the serialized shape matches the
/// Python dictionaries that are persisted with a session.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct DirectoryNode {
    /// Node name; also the on-disk name when [`Self::path`] is absent.
    pub name: String,
    /// Path relative to the parent directory; falls back to [`Self::name`] when absent.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    /// Human-readable description shown in the workspace documentation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Child nodes, in schema order.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub children: Vec<DirectoryNode>,
    /// Whether this node is a file rather than a directory.
    #[serde(skip_serializing_if = "is_false")]
    pub is_file: bool,
    /// Content seeded into a file node when it does not exist yet.
    #[serde(skip_serializing_if = "String::is_empty")]
    pub default_content: String,
}

/// `serde` predicate that omits `false` booleans.
#[allow(clippy::trivially_copy_pass_by_ref)] // serde's `skip_serializing_if` requires `&T`.
fn is_false(value: &bool) -> bool {
    !*value
}

impl DirectoryNode {
    /// Builds a file node with `name`, `path`, and seeded `content`.
    #[must_use]
    pub fn file(
        name: impl Into<String>,
        path: impl Into<String>,
        content: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            path: Some(path.into()),
            description: None,
            children: Vec::new(),
            is_file: true,
            default_content: content.into(),
        }
    }

    /// Builds a directory node with `name` and `path`.
    #[must_use]
    pub fn dir(name: impl Into<String>, path: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            path: Some(path.into()),
            description: None,
            children: Vec::new(),
            is_file: false,
            default_content: String::new(),
        }
    }

    /// Returns this node with a description.
    #[must_use]
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Returns this node with `children` appended.
    #[must_use]
    pub fn with_children(mut self, children: Vec<Self>) -> Self {
        self.children = children;
        self
    }

    /// The on-disk path of this node: [`Self::path`], or [`Self::name`] when absent/empty.
    #[must_use]
    pub fn resolved_path(&self) -> &str {
        match self.path.as_deref() {
            Some(path) if !path.is_empty() => path,
            _ => &self.name,
        }
    }

    /// Whether this node is a file.
    #[must_use]
    pub const fn is_file(&self) -> bool {
        self.is_file
    }

    /// The content seeded into this node when it is a file.
    #[must_use]
    pub fn content(&self) -> &str {
        &self.default_content
    }

    /// Validates this node and, recursively, its children.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::Validation`](openjiuwen_common::error::ErrorKind::Validation)
    /// carrying Python's message when a name is empty or contains a path separator.
    /// Path *safety* (absolute paths, `..`) is checked by
    /// [`is_safe_relative_path`](crate::workspace::is_safe_relative_path), not here.
    pub fn validate(&self) -> Result<()> {
        if self.name.is_empty() {
            return Err(Error::validation(
                "Directory `name` must be a non-empty string.",
            ));
        }
        if self.name.contains('/') || self.name.contains('\\') {
            return Err(Error::validation(format!(
                "Directory `name` must not contain path separators: {:?}",
                self.name
            )));
        }
        for child in &self.children {
            child.validate()?;
        }
        Ok(())
    }
}
