//! The workspace itself: construction, schema normalization, lookup, and mutation.
//!
//! Python's `Workspace` dataclass runs `__post_init__` to fill in defaults and validate the
//! directory tree; the Rust type reproduces that in [`Workspace::new`],
//! [`Workspace::with_directories`], and a hand-written [`Deserialize`] implementation so a
//! persisted workspace round-trips with the same normalization.
//!
//! Lookup quirks that callers rely on (both intentionally Python-compatible):
//!
//! - [`Workspace::get_directory`] searches the whole tree (nested names included) and falls
//!   back to the language default schema on a miss.
//! - [`Workspace::get_node_path`] is **top-level only** and never falls back.

use std::path::PathBuf;

use openjiuwen_common::error::Result;
use schemars::JsonSchema;
use serde::{de, Deserialize, Deserializer, Serialize};

use crate::workspace::node::{DirectoryNode, WorkspaceNode};
use crate::workspace::schema::get_workspace_schema;

/// The on-disk workspace: a root path plus the directory schema and its language.
///
/// The schema is data only; materializing it is
/// [`DirectoryBuilder`](crate::workspace::DirectoryBuilder)'s job
/// ([`DirectoryBuilder`](crate::workspace::DirectoryBuilder)), and every model-supplied
/// path is resolved through [`Workspace::resolve`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, JsonSchema)]
pub struct Workspace {
    /// Absolute or relative root directory of the workspace.
    #[schemars(with = "String")]
    pub root_path: PathBuf,
    /// Top-level schema nodes, in order.
    pub directories: Vec<DirectoryNode>,
    /// Schema language; only the exact string `"en"` selects the English schema.
    pub language: String,
}

impl Default for Workspace {
    fn default() -> Self {
        Self::new("./")
    }
}

impl Workspace {
    /// Builds a workspace rooted at `root_path` with the Chinese default schema.
    pub fn new(root_path: impl Into<PathBuf>) -> Self {
        Self {
            root_path: root_path.into(),
            directories: get_workspace_schema("cn"),
            language: "cn".to_string(),
        }
    }

    /// Builds a workspace rooted at `root_path` with the schema for `language`.
    pub fn for_language(root_path: impl Into<PathBuf>, language: impl Into<String>) -> Self {
        let language = language.into();
        Self {
            root_path: root_path.into(),
            directories: get_workspace_schema(&language),
            language,
        }
    }

    /// Returns this workspace with a different language.
    ///
    /// Untouched default nodes are swapped for the new language's defaults; a customized
    /// schema is preserved and supplemented with any missing default nodes.
    #[must_use]
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        let untouched = self.directories == get_workspace_schema(&self.language);
        self.language = language.into();
        if untouched {
            self.directories = get_workspace_schema(&self.language);
        } else {
            self.ensure_default_schema();
        }
        self
    }

    /// Returns this workspace with `directories` as its schema.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::Validation`](openjiuwen_common::error::ErrorKind::Validation)
    /// when a node is invalid. An empty list keeps the language defaults.
    pub fn with_directories(mut self, directories: Vec<DirectoryNode>) -> Result<Self> {
        if directories.is_empty() {
            return Ok(self);
        }
        for node in &directories {
            node.validate()?;
        }
        self.directories = directories;
        self.ensure_default_schema();
        Ok(self)
    }

    /// Validates every top-level node and, recursively, its children.
    ///
    /// # Errors
    ///
    /// Returns the first
    /// [`ErrorKind::Validation`](openjiuwen_common::error::ErrorKind::Validation) failure
    /// reported by [`DirectoryNode::validate`](crate::workspace::DirectoryNode::validate).
    pub fn validate(&self) -> Result<()> {
        for node in &self.directories {
            node.validate()?;
        }
        Ok(())
    }

    /// Appends any default top-level node the schema is missing.
    ///
    /// Nodes already present (matched by `name`) win, defaults cannot be opted out, order is
    /// preserved, and children are never back-filled. Returns how many nodes were appended.
    pub fn ensure_default_schema(&mut self) -> usize {
        let defaults = get_workspace_schema(&self.language);
        let mut added = 0;
        for default in defaults {
            if self
                .directories
                .iter()
                .any(|existing| existing.name == default.name)
            {
                continue;
            }
            tracing::info!(node = %default.name, language = %self.language, "supplemented default workspace node");
            self.directories.push(default);
            added += 1;
        }
        added
    }

    /// Whether the schema language is English.
    #[must_use]
    pub fn is_english(&self) -> bool {
        self.language == "en"
    }

    /// The schema language.
    #[must_use]
    pub fn language(&self) -> &str {
        &self.language
    }

    /// Returns the path of the first node named `name`, searching the whole tree.
    ///
    /// On a miss this falls back to the default schema for the current language, which is
    /// what keeps a nested lookup working after a caller replaced a default directory with
    /// a childless node. The returned value is the node's `path` (or its `name`).
    #[must_use]
    pub fn get_directory(&self, name: &str) -> Option<String> {
        if let Some(node) = find_node(&self.directories, name) {
            return Some(node.resolved_path().to_string());
        }
        let defaults = get_workspace_schema(&self.language);
        find_node(&defaults, name).map(|node| node.resolved_path().to_string())
    }

    /// Convenience wrapper over [`Self::get_directory`] for a well-known node.
    #[must_use]
    pub fn get_directory_for(&self, node: WorkspaceNode) -> Option<String> {
        self.get_directory(node.as_str())
    }

    /// Returns the path of a **top-level** node named `name`, without default fallback.
    ///
    /// Nested names (`MEMORY.md`, `daily_memory`, `session_memory.md`) return `None`, and so
    /// does a node whose path is unsafe: joining an absolute or `..`-escaping node path would
    /// redirect the caller's write outside the workspace.
    #[must_use]
    pub fn get_node_path(&self, name: &str) -> Option<PathBuf> {
        self.directories
            .iter()
            .find(|node| {
                node.name == name && crate::workspace::is_safe_relative_path(node.resolved_path())
            })
            .map(|node| self.root_path.join(node.resolved_path()))
    }

    /// Convenience wrapper over [`Self::get_node_path`] for a well-known node.
    #[must_use]
    pub fn get_node_path_for(&self, node: WorkspaceNode) -> Option<PathBuf> {
        self.get_node_path(node.as_str())
    }

    /// Validates `node` and upserts it into the top-level schema by name.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::Validation`](openjiuwen_common::error::ErrorKind::Validation) when
    /// the node is invalid.
    pub fn set_directory(&mut self, node: DirectoryNode) -> Result<()> {
        node.validate()?;
        match self
            .directories
            .iter_mut()
            .find(|existing| existing.name == node.name)
        {
            Some(existing) => *existing = node,
            None => self.directories.push(node),
        }
        Ok(())
    }

    /// Validates `nodes`, replaces the top-level schema with them, and re-supplements
    /// defaults.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::Validation`](openjiuwen_common::error::ErrorKind::Validation) when
    /// any node is invalid; nothing is changed in that case.
    pub fn set_directories(
        &mut self,
        nodes: impl IntoIterator<Item = DirectoryNode>,
    ) -> Result<()> {
        let nodes: Vec<DirectoryNode> = nodes.into_iter().collect();
        for node in &nodes {
            node.validate()?;
        }
        self.directories = nodes;
        self.ensure_default_schema();
        Ok(())
    }

    /// Returns the default schema for `language` (Python `get_default_directory`).
    #[must_use]
    pub fn get_default_directory(language: &str) -> Vec<DirectoryNode> {
        get_workspace_schema(language)
    }
}

/// Finds the first node named `name` in `nodes`, depth-first and including children.
fn find_node<'a>(nodes: &'a [DirectoryNode], name: &str) -> Option<&'a DirectoryNode> {
    for node in nodes {
        if node.name == name {
            return Some(node);
        }
        if let Some(found) = find_node(&node.children, name) {
            return Some(found);
        }
    }
    None
}

/// The raw JSON shape of a [`Workspace`]; every field is optional on input.
#[derive(Deserialize, Default)]
#[serde(default)]
struct WorkspaceParts {
    root_path: Option<PathBuf>,
    directories: Vec<DirectoryNode>,
    language: Option<String>,
}

impl<'de> Deserialize<'de> for Workspace {
    /// Deserializes with Python `__post_init__` semantics: missing `root_path` becomes
    /// `./`, missing/empty `directories` becomes the language defaults, missing `language`
    /// becomes `cn`, then the tree is validated and supplemented with default nodes.
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let parts = WorkspaceParts::deserialize(deserializer)?;
        let language = parts.language.unwrap_or_else(|| "cn".to_string());
        let mut workspace = Self {
            root_path: parts.root_path.unwrap_or_else(|| PathBuf::from("./")),
            directories: parts.directories,
            language,
        };
        if workspace.directories.is_empty() {
            workspace.directories = get_workspace_schema(&workspace.language);
        } else {
            workspace.validate().map_err(de::Error::custom)?;
        }
        workspace.ensure_default_schema();
        Ok(workspace)
    }
}
