//! Discovery of agent templates across the workspace, home directories and the
//! embedded built-in catalog.
//!
//! # Locations and precedence
//!
//! [`SpecRegistry::discover`] scans, in this order:
//!
//! 1. `<workspace>/.openjiuwen/agents/` — workspace-owned templates;
//! 2. each `<home>/.openjiuwen/agents/` directory, in the order supplied;
//! 3. the embedded [`builtin_specs`] catalog.
//!
//! Within one directory the files are sorted by path, so discovery is
//! deterministic. A duplicate name is a typed [`Error::validation`] that names
//! both the winning (first-discovered, higher-precedence) and the losing path.
//! Built-in templates are defaults: a same-named file template shadows the
//! built-in instead of failing, which is the only precedence rule that can
//! shadow silently.

use std::path::{Path, PathBuf};

use openjiuwen_common::error::{Error, Result};

use crate::workspace::Workspace;

use super::resources::builtin_specs;
use super::spec::{DeepAgentSpec, SpecFormat};

/// Directory (relative to a workspace or home root) holding agent templates.
pub const AGENTS_DIR: &str = ".openjiuwen/agents";

/// Where a discovered template came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SpecSource {
    /// `<workspace>/.openjiuwen/agents/` — highest precedence.
    Workspace,
    /// `<home>/.openjiuwen/agents/`.
    Home,
    /// The embedded built-in resources — lowest precedence.
    Builtin,
}

impl SpecSource {
    /// Returns the stable label used in diagnostics and [`SpecEntry`] reports.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Workspace => "workspace",
            Self::Home => "home",
            Self::Builtin => "builtin",
        }
    }
}

/// One discovered template and its origin.
#[derive(Debug, Clone)]
pub struct SpecEntry {
    /// Template name, unique inside the registry.
    pub name: String,
    /// Source file; absent for a built-in template.
    pub path: Option<PathBuf>,
    /// Origin of the template.
    pub source: SpecSource,
    /// The parsed template.
    pub spec: DeepAgentSpec,
}

impl SpecEntry {
    /// Returns a human-readable source label for diagnostics.
    #[must_use]
    pub fn label(&self) -> String {
        self.path.as_ref().map_or_else(
            || format!("built-in resources ({})", self.source.as_str()),
            |path| path.display().to_string(),
        )
    }
}

/// A deterministic set of discovered agent templates.
#[derive(Debug, Clone, Default)]
pub struct SpecRegistry {
    /// Entries in discovery order: workspace, home, then built-in.
    entries: Vec<SpecEntry>,
}

impl SpecRegistry {
    /// Discovers templates from `workspace`, every `home_dirs` entry, and the
    /// embedded built-in catalog.
    ///
    /// A missing `.openjiuwen/agents` directory is not an error; unreadable
    /// directories and malformed files are. Built-in templates are appended in
    /// catalog order and shadowed by a file template of the same name.
    ///
    /// # Errors
    ///
    /// Returns [`Error::sys_op`] when a directory or file cannot be read, and
    /// [`Error::validation`] when a template cannot be parsed, has an empty
    /// name, or duplicates an already-discovered file template.
    pub fn discover(workspace: &Workspace, home_dirs: &[PathBuf]) -> Result<Self> {
        let mut registry = Self::default();
        registry.discover_dir(&workspace.root_path.join(AGENTS_DIR), SpecSource::Workspace)?;
        for home in home_dirs {
            registry.discover_dir(&home.join(AGENTS_DIR), SpecSource::Home)?;
        }
        for spec in builtin_specs()? {
            if registry.entry(&spec.name).is_some() {
                tracing::debug!(
                    name = %spec.name,
                    "built-in agent template is shadowed by a discovered file"
                );
                continue;
            }
            registry.insert(SpecEntry {
                name: spec.name.clone(),
                path: None,
                source: SpecSource::Builtin,
                spec,
            })?;
        }
        Ok(registry)
    }

    /// Builds a registry from in-memory templates, preserving order.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`] when a template is invalid or a name
    /// repeats.
    pub fn from_specs(specs: Vec<DeepAgentSpec>) -> Result<Self> {
        let mut registry = Self::default();
        for spec in specs {
            spec.validate()?;
            registry.insert(SpecEntry {
                name: spec.name.clone(),
                path: spec.source_path.clone(),
                source: if spec.source_path.is_some() {
                    SpecSource::Workspace
                } else {
                    SpecSource::Builtin
                },
                spec,
            })?;
        }
        Ok(registry)
    }

    /// Returns the template called `name`.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&DeepAgentSpec> {
        self.entry(name).map(|entry| &entry.spec)
    }

    /// Returns the discovered entry called `name`.
    #[must_use]
    pub fn entry(&self, name: &str) -> Option<&SpecEntry> {
        self.entries.iter().find(|entry| entry.name == name)
    }

    /// Returns every entry in discovery order.
    #[must_use]
    pub fn entries(&self) -> &[SpecEntry] {
        &self.entries
    }

    /// Returns the template names in discovery order.
    #[must_use]
    pub fn names(&self) -> Vec<&str> {
        self.entries
            .iter()
            .map(|entry| entry.name.as_str())
            .collect()
    }

    /// Returns the number of discovered templates.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` when no template was discovered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Scans one directory for supported template files.
    fn discover_dir(&mut self, directory: &Path, source: SpecSource) -> Result<()> {
        let read = match std::fs::read_dir(directory) {
            Ok(read) => read,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(error) => {
                return Err(Error::sys_op(format!(
                    "failed to read agent spec directory {}: {error}",
                    directory.display()
                )))
            }
        };
        let mut paths: Vec<PathBuf> = read
            .filter_map(std::result::Result::ok)
            .map(|entry| entry.path())
            .filter(|path| path.is_file() && SpecFormat::from_path(path).is_ok())
            .collect();
        paths.sort();
        for path in paths {
            let spec = DeepAgentSpec::from_path(&path)?;
            if spec.name.trim().is_empty() {
                return Err(Error::validation(format!(
                    "agent spec {} has no name and its file stem is not a valid name",
                    path.display()
                )));
            }
            tracing::debug!(
                name = %spec.name,
                path = %path.display(),
                source = source.as_str(),
                "discovered agent template"
            );
            self.insert(SpecEntry {
                name: spec.name.clone(),
                path: Some(path),
                source,
                spec,
            })?;
        }
        Ok(())
    }

    /// Inserts an entry, rejecting a duplicate name with both paths.
    ///
    /// Discovering the same directory twice (a workspace root repeated in the
    /// home list) is idempotent: the identical path is skipped instead of being
    /// reported as a duplicate.
    fn insert(&mut self, entry: SpecEntry) -> Result<()> {
        if let Some(existing) = self.entries.iter().find(|known| known.name == entry.name) {
            if entry.path.is_some() && existing.path == entry.path {
                return Ok(());
            }
            return Err(Error::validation(format!(
                "duplicate agent spec name '{}': '{}' conflicts with '{}' (precedence workspace > home > built-in; the first-discovered entry wins)",
                entry.name,
                existing.label(),
                entry.label()
            )));
        }
        self.entries.push(entry);
        Ok(())
    }
}
