//! The agent's workspace: named directory nodes, the default bilingual schema, safe path
//! resolution, team/worktree links, and materialization of the schema on disk.
//!
//! This module is the security boundary for every model-supplied path in the harness: a
//! tool must resolve a path through [`Workspace::resolve`] (or check it with
//! [`is_safe_relative_path`]) before touching the filesystem, and [`Workspace::get_node_path`]
//! is the only way the harness itself names well-known nodes.
//!
//! ```no_run
//! use openjiuwen_harness::workspace::{DirectoryBuilder, Workspace};
//! use openjiuwen_sysop::{LocalWorkConfig, SysOperation};
//!
//! # async fn example() -> openjiuwen_common::error::Result<()> {
//! let existing = std::path::Path::new(".");
//! let workspace = Workspace::for_language(existing, "en");
//! let operation = SysOperation::local(LocalWorkConfig::unrestricted("."))?;
//! let builder = DirectoryBuilder::new(operation, existing);
//! builder.build(&workspace.directories).await?;
//! assert_eq!(workspace.get_node_path("skills"), Some(existing.join("skills")));
//! # Ok(())
//! # }
//! ```
//!
//! Module layout:
//!
//! - [`node`] — [`WorkspaceNode`] and [`DirectoryNode`]
//! - [`schema`] — the cn/en default schema and packaged seed content
//! - [`workspace`] — the [`Workspace`] type: construction, normalization, lookup, mutation
//! - [`resolve`] — sandboxed path resolution (impl block on [`Workspace`])
//! - [`links`] — `.team`/`.worktree` link management
//! - [`directory_builder`] — [`DirectoryBuilder`] and [`is_safe_relative_path`]

mod directory_builder;
mod links;
mod node;
mod resolve;
mod schema;
#[allow(clippy::module_inception)]
// Python keeps `workspace/workspace.py`; the file name is part of the port.
mod workspace;

pub use directory_builder::{is_safe_relative_path, DirectoryBuilder, WORKSPACE_MARKER};
pub use links::{LinkKind, WorkspaceLink, LINK_MARKER, TEAM_LINKS_DIR, WORKTREE_LINKS_DIR};
pub use node::{DirectoryNode, WorkspaceNode};
pub use schema::{default_content, get_workspace_schema};
pub use workspace::Workspace;

#[cfg(test)]
mod tests;
