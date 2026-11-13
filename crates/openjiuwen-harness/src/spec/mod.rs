//! File-based `DeepAgentSpec` agent templates, plugin manifests and the bundled
//! `resources/` catalog.
//!
//! This module ports the declarative half of the Python
//! `openjiuwen.harness.resources` package and the `DeepAgentSpec` parts of
//! `factory.py`/`deep_agent.py` as ordinary offline data handling:
//!
//! - [`DeepAgentSpec`] — a serde template of one agent (identity, prompt
//!   fragments, model hints, tools, skills, sub-agents, rails, workspace hints
//!   and metadata).
//! - [`SpecRegistry`] — discovery of templates in a workspace, a set of home
//!   directories, and the embedded built-in catalog.
//! - [`PluginManifest`]/[`PluginRegistry`]/[`ResolvedPlugins`] — plugin
//!   manifests under `<workspace>/.openjiuwen/plugins/` and their aggregate
//!   resolution.
//! - [`ResolvedSpec`]/[`resolve`]/[`build_agent`] — validation and materializing
//!   a template into the harness parts, plus a helper that hands those parts to
//!   the existing `DeepAgentBuilder`/`create_deep_agent` surface.
//! - [`builtin_specs`]/[`BUILTIN_RULES_YAML`] — the embedded built-in catalog
//!   the registry falls back to.
//!
//! # Safety boundary
//!
//! Templates and plugin manifests are *data only*. Loading one reads files,
//! parses YAML/JSON/Markdown and validates names against fixed tables; it never
//! loads a dynamic library, spawns a process, imports Python, or executes code
//! from a template. A tool is selected by its model-facing name through
//! [`BuiltinTool::from_name`](crate::tools::BuiltinTool::from_name); unknown
//! names are rejected with [`Error::validation`](openjiuwen_common::error::Error::validation).
//!
//! # Wire form
//!
//! Every type is `snake_case`, tolerant of unknown fields (`deny_unknown_fields`
//! is never used) and carries documented defaults, so templates written by a
//! newer Python release still load as long as the known fields fit.
// The public names (`DeepAgentSpec`, `SpecRegistry`, `SubAgentSpec`, ...) mirror
// the Python contract and intentionally repeat the module name.
#![allow(clippy::module_name_repetitions)]

mod plugin;
mod registry;
mod resolve;
mod resources;
#[allow(clippy::module_inception)]
mod spec;
#[cfg(test)]
mod tests;

pub use plugin::{
    load_plugins, PluginManifest, PluginRegistry, ResolvedPlugin, ResolvedPlugins,
    PLUGIN_MANIFEST_FILENAME,
};
pub use registry::{SpecEntry, SpecRegistry, SpecSource, AGENTS_DIR};
pub use resolve::{
    build_agent, resolve, resolve_with_source, ResolvedSkill, ResolvedSpec, SpecBuildOptions,
};
pub use resources::{builtin_specs, BUILTIN_RULES_YAML};
pub use spec::{
    DeepAgentSpec, ModelSpec, PromptFragment, SkillSpec, SpecFormat, SubAgentSpec, WorkspaceHints,
    KNOWN_RAILS, SUPPORTED_SPEC_EXTENSIONS,
};

use openjiuwen_common::error::Result;

use crate::workspace::Workspace;

impl SpecRegistry {
    /// Resolves the discovered template called `name`.
    ///
    /// Relative skill roots are resolved against the template's own directory
    /// (or the workspace root for a built-in template).
    ///
    /// # Errors
    ///
    /// Returns [`openjiuwen_common::error::Error::not_found`] when no template
    /// has that name, and propagates the validation errors of [`resolve`].
    pub fn resolve(&self, name: &str, workspace: &Workspace) -> Result<ResolvedSpec> {
        let entry = self.entry(name).ok_or_else(|| {
            openjiuwen_common::error::Error::not_found(format!(
                "agent spec '{name}' was not discovered (available: {})",
                self.names().join(", ")
            ))
        })?;
        resolve_with_source(&entry.spec, workspace)
    }
}
