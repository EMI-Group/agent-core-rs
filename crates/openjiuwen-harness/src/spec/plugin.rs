//! Plugin manifests and their resolution into an aggregate of harness pieces.
//!
//! A plugin is a directory under `<workspace>/.openjiuwen/plugins/<name>/` with a
//! `plugin.yaml` manifest:
//!
//! ```yaml
//! name: acme
//! version: "0.1.0"
//! description: Acme analysis helpers
//! tools: [read_file, web_search]
//! rails: [security]
//! skills: [skills]
//! spec: agent.yaml
//! metadata:
//!   owner: platform-team
//! ```
//!
//! Paths (`skills`, `spec`) are package-relative: every plugin resource stays
//! inside its own directory, so a manifest can never point the loader at the
//! host filesystem.
//!
//! # No execution
//!
//! Loading a plugin reads and validates files only. There is deliberately no
//! dynamic-library loading, no process spawn, no Python import, no network
//! access, and no evaluation of expressions or templates. A tool is selected by
//! its model-facing name through
//! [`BuiltinTool::from_name`](crate::tools::BuiltinTool::from_name); an unknown
//! name is a typed validation error, never a silent no-op.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use openjiuwen_common::error::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::tools::BuiltinTool;
use crate::workspace::{is_safe_relative_path, Workspace};

use super::spec::{normalize_rail, DeepAgentSpec};

/// File name of a plugin manifest inside its plugin directory.
pub const PLUGIN_MANIFEST_FILENAME: &str = "plugin.yaml";

/// Directory (relative to a workspace root) holding plugin packages.
const PLUGINS_DIR: &str = ".openjiuwen/plugins";

/// The typed contents of a `plugin.yaml` manifest.
///
/// Unknown fields are ignored and every field defaults, so a manifest written by
/// a newer release still loads as long as its known declarations fit.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct PluginManifest {
    /// Plugin name; falls back to the package directory name when omitted.
    pub name: String,
    /// Optional package version.
    pub version: Option<String>,
    /// Human-readable description.
    pub description: String,
    /// Model-facing tool names the plugin adds to its host.
    pub tools: Vec<String>,
    /// Rail names the plugin mounts.
    pub rails: Vec<String>,
    /// Skill-root directories, relative to the plugin package.
    pub skills: Vec<String>,
    /// Optional agent-template file, relative to the plugin package.
    pub spec: Option<String>,
    /// Free-form discovery metadata.
    pub metadata: BTreeMap<String, Value>,
    /// Path the manifest was loaded from; absent for in-memory manifests.
    #[serde(skip)]
    pub source_path: Option<PathBuf>,
}

impl PluginManifest {
    /// Loads and validates a manifest from `path`.
    ///
    /// YAML (`.yaml`, `.yml`) and JSON (`.json`) are accepted. An omitted `name`
    /// is derived from the package directory. Only the manifest shape is
    /// validated here; tool, rail, skill and spec resolution happens in
    /// [`PluginRegistry::load`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::sys_op`] when the file cannot be read and
    /// [`Error::validation`] for an unsupported extension or malformed document.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let extension = path
            .extension()
            .and_then(std::ffi::OsStr::to_str)
            .map(str::to_ascii_lowercase)
            .unwrap_or_default();
        let text = std::fs::read_to_string(path).map_err(|error| {
            Error::sys_op(format!(
                "failed to read plugin manifest {}: {error}",
                path.display()
            ))
        })?;
        let mut manifest: Self = match extension.as_str() {
            "yaml" | "yml" => serde_yaml_ng::from_str(&text).map_err(|error| {
                Error::validation(format!(
                    "failed to parse YAML plugin manifest {}: {error}",
                    path.display()
                ))
            })?,
            "json" => serde_json::from_str(&text).map_err(|error| {
                Error::validation(format!(
                    "failed to parse JSON plugin manifest {}: {error}",
                    path.display()
                ))
            })?,
            other => {
                return Err(Error::validation(format!(
                    "unsupported plugin manifest extension '{other}' for {}; supported: yaml, yml, json",
                    path.display()
                )))
            }
        };
        if manifest.name.trim().is_empty() {
            if let Some(directory) = path.parent().and_then(Path::file_name) {
                manifest.name = directory.to_string_lossy().into_owned();
            }
        }
        manifest.source_path = Some(path.to_path_buf());
        manifest.validate()?;
        Ok(manifest)
    }

    /// Validates the manifest's own fields.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`] for an empty name or an empty version.
    pub fn validate(&self) -> Result<()> {
        if self.name.trim().is_empty() {
            return Err(Error::validation(format!(
                "plugin manifest {} is missing a non-empty 'name'",
                self.label()
            )));
        }
        if self
            .version
            .as_deref()
            .is_some_and(|version| version.trim().is_empty())
        {
            return Err(Error::validation(format!(
                "plugin '{}' in {} has an empty 'version'",
                self.name,
                self.label()
            )));
        }
        Ok(())
    }

    /// Returns the manifest location used in diagnostics.
    #[must_use]
    pub fn label(&self) -> String {
        self.source_path.as_ref().map_or_else(
            || format!("plugin '{}' (in-memory)", self.name),
            |path| format!("plugin '{}' ({})", self.name, path.display()),
        )
    }
}

/// One fully resolved plugin package.
#[derive(Debug, Clone)]
pub struct ResolvedPlugin {
    /// The validated manifest.
    pub manifest: PluginManifest,
    /// Absolute path of the plugin directory.
    pub directory: PathBuf,
    /// Tools declared by the manifest, in declaration order.
    pub tools: Vec<BuiltinTool>,
    /// Canonical rail names declared by the manifest.
    pub rails: Vec<String>,
    /// Absolute skill roots, in declaration order.
    pub skill_roots: Vec<PathBuf>,
    /// The optional agent template, already parsed.
    pub spec: Option<DeepAgentSpec>,
}

/// The aggregate of every loaded plugin.
///
/// The aggregate is read-only data: it does not mount anything. A host chooses
/// which tool instances, rails, skill roots and templates to hand to a
/// `DeepAgent`.
#[derive(Debug, Clone, Default)]
pub struct ResolvedPlugins {
    /// Resolved plugins in manifest-name order.
    pub plugins: Vec<ResolvedPlugin>,
}

impl ResolvedPlugins {
    /// Returns every distinct declared tool, in first-declaration order.
    #[must_use]
    pub fn tools(&self) -> Vec<BuiltinTool> {
        let mut tools = Vec::new();
        for plugin in &self.plugins {
            for tool in &plugin.tools {
                if !tools.contains(tool) {
                    tools.push(*tool);
                }
            }
        }
        tools
    }

    /// Returns every distinct rail name, in first-declaration order.
    #[must_use]
    pub fn rails(&self) -> Vec<String> {
        let mut rails = Vec::new();
        for plugin in &self.plugins {
            for rail in &plugin.rails {
                if !rails.contains(rail) {
                    rails.push(rail.clone());
                }
            }
        }
        rails
    }

    /// Returns every distinct skill root, in first-declaration order.
    #[must_use]
    pub fn skill_roots(&self) -> Vec<PathBuf> {
        let mut roots: Vec<PathBuf> = Vec::new();
        for plugin in &self.plugins {
            for root in &plugin.skill_roots {
                if !roots.contains(root) {
                    roots.push(root.clone());
                }
            }
        }
        roots
    }

    /// Returns the agent templates declared by the plugins, in load order.
    #[must_use]
    pub fn specs(&self) -> Vec<&DeepAgentSpec> {
        self.plugins
            .iter()
            .filter_map(|plugin| plugin.spec.as_ref())
            .collect()
    }

    /// Returns the number of loaded plugins.
    #[must_use]
    pub fn len(&self) -> usize {
        self.plugins.len()
    }

    /// Returns `true` when no plugin was loaded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.plugins.is_empty()
    }
}

/// A validated set of plugin packages discovered in one workspace.
#[derive(Debug, Clone, Default)]
pub struct PluginRegistry {
    /// Resolved plugins in manifest-name order.
    plugins: Vec<ResolvedPlugin>,
}

impl PluginRegistry {
    /// Loads every plugin under `<workspace>/.openjiuwen/plugins/`.
    ///
    /// A missing plugins directory yields an empty registry. Directories without
    /// a `plugin.yaml` are skipped (they are not plugin packages); a malformed
    /// manifest, a duplicate plugin name, an unknown tool/rail or a missing
    /// package-relative resource is an error.
    ///
    /// # Errors
    ///
    /// Returns [`Error::sys_op`] when a directory or file cannot be read, and
    /// [`Error::validation`] for a malformed manifest, a duplicate name, an
    /// unknown tool or rail, an unsafe path, or a model/spec mismatch.
    pub fn load(workspace: &Workspace) -> Result<Self> {
        let root = workspace.root_path.join(PLUGINS_DIR);
        let read = match std::fs::read_dir(&root) {
            Ok(read) => read,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Ok(Self::default())
            }
            Err(error) => {
                return Err(Error::sys_op(format!(
                    "failed to read plugin directory {}: {error}",
                    root.display()
                )))
            }
        };
        let mut directories: Vec<PathBuf> = read
            .filter_map(std::result::Result::ok)
            .map(|entry| entry.path())
            .filter(|path| path.is_dir())
            .collect();
        directories.sort();
        let mut registry = Self::default();
        for directory in directories {
            let manifest_path = directory.join(PLUGIN_MANIFEST_FILENAME);
            if !manifest_path.is_file() {
                tracing::debug!(
                    directory = %directory.display(),
                    "skipping a plugin directory without a manifest"
                );
                continue;
            }
            let manifest = PluginManifest::from_path(&manifest_path)?;
            if let Some(existing) = registry.get(&manifest.name) {
                return Err(Error::validation(format!(
                    "duplicate plugin name '{}': '{}' conflicts with '{}'",
                    manifest.name,
                    existing.manifest.label(),
                    manifest.label()
                )));
            }
            registry.plugins.push(resolve_plugin(manifest, &directory)?);
        }
        registry.plugins.sort_by(|left, right| {
            left.manifest
                .name
                .cmp(&right.manifest.name)
                .then_with(|| left.directory.cmp(&right.directory))
        });
        Ok(registry)
    }

    /// Returns the aggregate of the loaded plugins.
    #[must_use]
    pub fn resolve(&self) -> ResolvedPlugins {
        ResolvedPlugins {
            plugins: self.plugins.clone(),
        }
    }

    /// Returns the resolved plugin called `name`.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&ResolvedPlugin> {
        self.plugins
            .iter()
            .find(|plugin| plugin.manifest.name == name)
    }

    /// Returns every resolved plugin in manifest-name order.
    #[must_use]
    pub fn plugins(&self) -> &[ResolvedPlugin] {
        &self.plugins
    }

    /// Returns the number of loaded plugins.
    #[must_use]
    pub fn len(&self) -> usize {
        self.plugins.len()
    }

    /// Returns `true` when no plugin was loaded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.plugins.is_empty()
    }
}

/// Loads and aggregates every plugin of `workspace`.
///
/// # Errors
///
/// Propagates [`PluginRegistry::load`].
pub fn load_plugins(workspace: &Workspace) -> Result<ResolvedPlugins> {
    Ok(PluginRegistry::load(workspace)?.resolve())
}

/// Validates and resolves one manifest against its package directory.
fn resolve_plugin(manifest: PluginManifest, directory: &Path) -> Result<ResolvedPlugin> {
    let label = manifest.label();
    let mut tools = Vec::new();
    for name in &manifest.tools {
        let tool = BuiltinTool::from_name(name).ok_or_else(|| {
            Error::validation(format!(
                "unknown tool '{name}' declared by {label}; known tools: {}",
                crate::tools::extended_tool_names().join(", ")
            ))
        })?;
        if !tools.contains(&tool) {
            tools.push(tool);
        }
    }
    let mut rails = Vec::new();
    for name in &manifest.rails {
        let canonical = normalize_rail(name).map_err(|error| {
            Error::validation(format!("{} declared by {label}", error.message()))
        })?;
        if !rails.iter().any(|known| known == canonical) {
            rails.push(canonical.to_string());
        }
    }
    let mut skill_roots = Vec::new();
    for root in &manifest.skills {
        if !is_safe_relative_path(root) {
            return Err(Error::validation(format!(
                "unsafe skill root '{root}' declared by {label}; plugin resources must be package-relative"
            )));
        }
        let path = directory.join(root);
        if !path.is_dir() {
            return Err(Error::validation(format!(
                "skill root '{}' declared by {label} is not a directory",
                path.display()
            )));
        }
        if !skill_roots.contains(&path) {
            skill_roots.push(path);
        }
    }
    let spec = match &manifest.spec {
        Some(declared) => {
            if !is_safe_relative_path(declared) {
                return Err(Error::validation(format!(
                    "unsafe agent-template path '{declared}' declared by {label}; plugin resources must be package-relative"
                )));
            }
            let path = directory.join(declared);
            if !path.is_file() {
                return Err(Error::validation(format!(
                    "agent template '{}' declared by {label} does not exist",
                    path.display()
                )));
            }
            Some(DeepAgentSpec::from_path(&path)?)
        }
        None => None,
    };
    Ok(ResolvedPlugin {
        manifest,
        directory: directory.to_path_buf(),
        tools,
        rails,
        skill_roots,
        spec,
    })
}
