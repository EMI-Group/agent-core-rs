//! Permanent "always allow" rules, persisted as YAML under the CLI home.
//!
//! When a user answers `a` (always) at a permission prompt, the CLI remembers the
//! decision so later sessions do not ask again. This module is that persistence
//! layer and nothing more: it reads and writes a
//! [`PermissionsSection`] as YAML at
//! `<home>/permissions.yaml` (see
//! [`allow_rules_path`](super::paths::allow_rules_path)) and folds a stored
//! section onto the CLI's built-in defaults.
//!
//! The file is a plain `permissions:` document, so it can be hand-edited with the
//! same keys the agent YAML uses (`enabled`, `tools`, `rules`,
//! `approval_overrides`, the guards). A corrupt file is never fatal: [`AllowRules::load`]
//! logs a warning and behaves as if no rules were stored, because a broken rules
//! file must not lock the user out of the CLI.
//!
//! Writes are atomic (temporary sibling file plus rename), so a crash during a
//! write never leaves a half-written rules file behind.

use std::io::ErrorKind;
use std::path::{Path, PathBuf};

use openjiuwen_common::config::OpenJiuwenHome;
use openjiuwen_common::error::{Error, Result};

use crate::permission::PermissionsSection;

use super::paths;

/// Permanent allow rules stored in one YAML file.
///
/// The handle holds only the file location; every operation touches the disk on
/// demand, so constructing one is free and side-effect free.
#[derive(Debug, Clone)]
pub struct AllowRules {
    /// Location of the YAML rules file.
    path: PathBuf,
}

impl AllowRules {
    /// Builds a handle over an explicit path.
    #[must_use]
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    /// Builds a handle over `<home>/permissions.yaml`.
    #[must_use]
    pub fn for_home(home: &OpenJiuwenHome) -> Self {
        Self::new(paths::allow_rules_path(home))
    }

    /// Returns the rules file location.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Loads the persisted section, or `None` when the file is absent.
    ///
    /// A malformed document is logged with `tracing::warn!` and reported as
    /// `Ok(None)`: a corrupt rules file must never block the CLI, and treating it
    /// as "no rules" keeps every tool at its fail-closed default.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] naming the path when the file exists but cannot
    /// be read (permissions, a directory occupying the path, invalid UTF-8).
    pub fn load(&self) -> Result<Option<PermissionsSection>> {
        let text = match std::fs::read_to_string(&self.path) {
            Ok(text) => text,
            Err(error) if error.kind() == ErrorKind::NotFound => return Ok(None),
            Err(error) => {
                return Err(Error::config(format!(
                    "failed to read allow-rule file {}: {error}",
                    self.path.display()
                )))
            }
        };
        match serde_yaml_ng::from_str(&text) {
            Ok(section) => Ok(Some(section)),
            Err(error) => {
                tracing::warn!(
                    path = %self.path.display(),
                    %error,
                    "ignoring malformed allow-rule file and using the built-in defaults"
                );
                Ok(None)
            }
        }
    }

    /// Writes `section` as YAML (atomic temporary file plus rename).
    ///
    /// Missing parent directories are created first, and the document is written
    /// to `<path>.tmp` before being renamed over `<path>`, so a crash never
    /// exposes a partial file.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] naming the path when the parent directory cannot
    /// be created, the document cannot be encoded, or the write or rename fails.
    pub fn persist(&self, section: &PermissionsSection) -> Result<()> {
        let text = Self::yaml(section)?;
        if let Some(dir) = self.path.parent() {
            std::fs::create_dir_all(dir).map_err(|error| {
                Error::config(format!(
                    "failed to create allow-rule directory {}: {error}",
                    dir.display()
                ))
            })?;
        }
        let temp = temp_path(&self.path);
        std::fs::write(&temp, text).map_err(|error| {
            Error::config(format!(
                "failed to write allow-rule file {}: {error}",
                temp.display()
            ))
        })?;
        std::fs::rename(&temp, &self.path).map_err(|error| {
            Error::config(format!(
                "failed to replace allow-rule file {}: {error}",
                self.path.display()
            ))
        })
    }

    /// Folds the persisted rules on top of the CLI defaults.
    ///
    /// Precedence, in the order the merge applies it:
    ///
    /// - `tools`: every stored entry overrides the same tool in `base.tools`,
    ///   while tools only `base` knows about are kept untouched;
    /// - `rules` and `approval_overrides`: the stored entries are appended after
    ///   the base ones, so a stored rule never shadows a base rule;
    /// - `defaults`: stored entries override `base` per key;
    /// - `enabled`, `package_builtin_rules`, `permission_mode`, `categories`,
    ///   `file_guard`, `net_guard` and `shell_guard`: taken from the stored
    ///   document, which is the persisted superset [`AllowRules::persist`] writes.
    ///
    /// When no rules file exists the base section is returned unchanged.
    ///
    /// # Errors
    ///
    /// See [`AllowRules::load`].
    pub fn load_merged(&self, base: &PermissionsSection) -> Result<PermissionsSection> {
        let Some(stored) = self.load()? else {
            return Ok(base.clone());
        };
        let mut merged = base.clone();
        for (tool, level) in &stored.tools {
            merged.tools.insert(tool.clone(), *level);
        }
        merged.rules.extend(stored.rules.iter().cloned());
        merged
            .approval_overrides
            .extend(stored.approval_overrides.iter().cloned());
        for (key, level) in &stored.defaults {
            merged.defaults.insert(key.clone(), *level);
        }
        merged.enabled = stored.enabled;
        merged.package_builtin_rules = stored.package_builtin_rules;
        merged.permission_mode = stored.permission_mode;
        merged.categories = stored.categories;
        merged.file_guard = stored.file_guard;
        merged.net_guard = stored.net_guard;
        merged.shell_guard = stored.shell_guard;
        Ok(merged)
    }

    /// Serializes `section` to the YAML document this module writes.
    ///
    /// Exposed so callers and tests can assert the persisted shape without
    /// touching the disk.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] when the section cannot be encoded as YAML.
    pub fn yaml(section: &PermissionsSection) -> Result<String> {
        serde_yaml_ng::to_string(section).map_err(|error| {
            Error::config(format!("failed to encode allow rules as YAML: {error}"))
        })
    }
}

/// Returns the sibling temporary path used for the atomic rewrite.
fn temp_path(path: &Path) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(".tmp");
    PathBuf::from(name)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use tempfile::TempDir;

    use super::*;
    use crate::permission::{PermissionLevel, PermissionRule};

    /// Returns a rule used by the merge assertions.
    fn rule(id: &str) -> PermissionRule {
        PermissionRule {
            id: id.to_owned(),
            tools: vec!["shell".to_owned()],
            pattern: "git *".to_owned(),
            ..PermissionRule::default()
        }
    }

    /// Builds a handle over a fresh temporary directory.
    fn handle() -> (TempDir, AllowRules) {
        let temp = TempDir::new().unwrap();
        let rules = AllowRules::new(temp.path().join("permissions.yaml"));
        (temp, rules)
    }

    /// Builds the section the round-trip tests persist.
    fn section() -> PermissionsSection {
        let mut section = PermissionsSection::default();
        section
            .tools
            .insert("write_file".to_owned(), PermissionLevel::Allow);
        section.rules.push(rule("stored-rule"));
        section.permission_mode = Some("strict".to_owned());
        section
    }

    #[test]
    fn persist_then_load_round_trips_a_section() {
        let (_temp, rules) = handle();
        let section = section();
        rules.persist(&section).unwrap();

        let loaded = rules.load().unwrap().unwrap();
        assert_eq!(loaded, section);
        assert_eq!(
            loaded.tools.get("write_file"),
            Some(&PermissionLevel::Allow)
        );
        assert_eq!(loaded.permission_mode.as_deref(), Some("strict"));
    }

    #[test]
    fn yaml_document_uses_the_python_tool_scalar_form() {
        let text = AllowRules::yaml(&section()).unwrap();
        assert!(
            text.contains("write_file: allow"),
            "unexpected YAML document:\n{text}"
        );
        let parsed: PermissionsSection = serde_yaml_ng::from_str(&text).unwrap();
        assert_eq!(parsed, section());
    }

    #[test]
    fn missing_file_loads_as_none() {
        let (_temp, rules) = handle();
        assert!(rules.load().unwrap().is_none());
    }

    #[test]
    fn malformed_yaml_loads_as_none_without_panicking() {
        let (temp, rules) = handle();
        std::fs::write(temp.path().join("permissions.yaml"), "tools: [unclosed").unwrap();
        assert!(rules.load().unwrap().is_none());
    }

    #[test]
    fn load_merged_lets_stored_tools_override_the_base() {
        let (_temp, rules) = handle();
        rules.persist(&section()).unwrap();

        let mut base = PermissionsSection::default();
        base.tools
            .insert("write_file".to_owned(), PermissionLevel::Ask);
        base.tools
            .insert("read_file".to_owned(), PermissionLevel::Ask);
        base.rules.push(rule("base-rule"));

        let merged = rules.load_merged(&base).unwrap();
        assert_eq!(
            merged.tools.get("write_file"),
            Some(&PermissionLevel::Allow),
            "the stored allow must win over the base ask"
        );
        assert_eq!(
            merged.tools.get("read_file"),
            Some(&PermissionLevel::Ask),
            "an unrelated base tool must be untouched"
        );
        let ids: Vec<&str> = merged.rules.iter().map(|rule| rule.id.as_str()).collect();
        assert_eq!(ids, vec!["base-rule", "stored-rule"]);
        assert_eq!(merged.permission_mode.as_deref(), Some("strict"));
    }

    #[test]
    fn load_merged_without_a_file_returns_the_base_unchanged() {
        let (_temp, rules) = handle();
        let mut base = PermissionsSection::default();
        base.tools
            .insert("write_file".to_owned(), PermissionLevel::Ask);
        assert_eq!(rules.load_merged(&base).unwrap(), base);
    }

    #[test]
    fn successful_persist_leaves_no_temporary_file_behind() {
        let (temp, rules) = handle();
        rules.persist(&section()).unwrap();
        assert!(temp.path().join("permissions.yaml").exists());
        assert!(!temp.path().join("permissions.yaml.tmp").exists());
    }

    #[test]
    fn persist_creates_missing_parent_directories() {
        let temp = TempDir::new().unwrap();
        let rules = AllowRules::new(temp.path().join("nested/dir/permissions.yaml"));
        rules.persist(&section()).unwrap();
        assert_eq!(rules.load().unwrap().unwrap(), section());
    }

    #[test]
    fn for_home_targets_permissions_yaml_under_the_home_root() {
        let home = OpenJiuwenHome::from_root("cli-home");
        let rules = AllowRules::for_home(&home);
        assert_eq!(rules.path(), Path::new("cli-home").join("permissions.yaml"));
    }
}
