//! Sparse-checkout configuration: parsed paths, mode, and validation.
//!
//! Port of the sparse-checkout flow in
//! `openjiuwen/harness/tools/worktree/{git,backend,models}.py`: Python keeps the
//! configured paths in `WorktreeConfig.sparse_paths`, creates the worktree with
//! `git worktree add --no-checkout`, then runs
//! `git sparse-checkout set --cone -- <paths>` followed by `git checkout HEAD`.
//! A persisted [`super::models::WorktreeRecord`] only remembers whether sparse
//! checkout was used; this port keeps the paths themselves so a re-entered
//! worktree can re-apply the same selection.
//!
//! # Modes
//!
//! [`SparseMode::Cone`] is the default and maps to git's cone semantics: a path
//! names a directory whose contents are included wholesale. Git requires cone
//! paths to be directories, which is also the only shape the worktree helper
//! produces. [`SparseMode::Patterns`] maps to `--no-cone` and passes the entries
//! through as git patterns (`*.rs`, `docs/`), so the caller owns the semantics.
//!
//! # Paths
//!
//! Sparse paths are repository-relative, never absolute, and never escape the
//! worktree. [`validate_sparse_path`] refuses a hostile entry instead of
//! rewriting it; [`SparseConfig::normalized`] then only cleans up separators and
//! whitespace, and collapses duplicates. Validation always runs before
//! normalisation, so a path such as `../etc` can never be "fixed" into
//! `etc`.
use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use openjiuwen_common::error::{Error, Result};

/// Maximum number of paths or patterns one sparse checkout may select.
pub const MAX_SPARSE_PATHS: usize = 256;

/// Sparse-checkout selection mode.
///
/// `Cone` is git's default `sparse-checkout` mode (directories); `Patterns` is
/// `--no-cone`, where every entry is a git path pattern.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SparseMode {
    /// Directory-oriented cone mode (git's default).
    #[default]
    Cone,
    /// Pattern mode (`--no-cone`).
    Patterns,
}

impl SparseMode {
    /// The `snake_case` wire name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cone => "cone",
            Self::Patterns => "patterns",
        }
    }

    /// The `git sparse-checkout set` flag for this mode.
    #[must_use]
    pub const fn git_flag(self) -> &'static str {
        match self {
            Self::Cone => "--cone",
            Self::Patterns => "--no-cone",
        }
    }
}

/// One sparse-checkout selection: the paths and the mode they are read in.
///
/// Both fields default, so `{}` deserialises to an empty cone selection and
/// persisted records written before sparse checkout existed stay loadable.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct SparseConfig {
    /// Repository-relative paths (cone mode) or git patterns (pattern mode).
    pub paths: Vec<String>,
    /// How `paths` are interpreted.
    pub mode: SparseMode,
}

impl SparseConfig {
    /// Builds and validates a selection of `paths` in `mode`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`] for empty, over-long or control-character
    /// input and [`Error::security`] for absolute paths, traversal segments,
    /// NUL bytes, `~` and drive prefixes.
    pub fn new(
        paths: impl IntoIterator<Item = impl Into<String>>,
        mode: SparseMode,
    ) -> Result<Self> {
        let config = Self {
            paths: paths.into_iter().map(Into::into).collect(),
            mode,
        };
        config.validate()?;
        Ok(config)
    }

    /// Builds and validates a cone-mode selection.
    ///
    /// # Errors
    ///
    /// See [`SparseConfig::new`].
    pub fn cone(paths: impl IntoIterator<Item = impl Into<String>>) -> Result<Self> {
        Self::new(paths, SparseMode::Cone)
    }

    /// Builds and validates a pattern-mode (`--no-cone`) selection.
    ///
    /// # Errors
    ///
    /// See [`SparseConfig::new`].
    pub fn patterns(paths: impl IntoIterator<Item = impl Into<String>>) -> Result<Self> {
        Self::new(paths, SparseMode::Patterns)
    }

    /// Whether no path is selected.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.paths.is_empty()
    }

    /// Validates the path count and every path.
    ///
    /// # Errors
    ///
    /// Returns [`Error::validation`] when there are more than
    /// [`MAX_SPARSE_PATHS`] paths or a path is empty / contains control
    /// characters, and [`Error::security`] for a path that is absolute or
    /// traverses upwards. Duplicates are allowed here and collapsed by
    /// [`SparseConfig::normalized`].
    pub fn validate(&self) -> Result<()> {
        if self.paths.len() > MAX_SPARSE_PATHS {
            return Err(Error::validation(format!(
                "sparse checkout accepts at most {MAX_SPARSE_PATHS} paths, got {}",
                self.paths.len()
            )));
        }
        for path in &self.paths {
            validate_sparse_path(path)?;
        }
        Ok(())
    }

    /// The cleaned-up selection: trimmed, `/`-separated, no trailing slash,
    /// duplicates collapsed in first-seen order.
    ///
    /// Entries that become empty after trimming are dropped. Callers must have
    /// validated the config first — this method never repairs a hostile path.
    #[must_use]
    pub fn normalized(&self) -> Self {
        let mut seen = HashSet::new();
        let mut paths = Vec::with_capacity(self.paths.len());
        for raw in &self.paths {
            let mut path = raw.trim().replace('\\', "/");
            while path.len() > 1 && path.ends_with('/') {
                path.pop();
            }
            if path.is_empty() {
                continue;
            }
            if seen.insert(path.clone()) {
                paths.push(path);
            }
        }
        Self {
            paths,
            mode: self.mode,
        }
    }
}

/// Validates one repository-relative sparse path or pattern.
///
/// # Errors
///
/// - [`Error::security`] for absolute paths (`/`, `\`, `~`, `C:`), any `.` or
///   `..` segment, and embedded NUL bytes;
/// - [`Error::validation`] for empty or whitespace-only input and for control
///   characters.
pub fn validate_sparse_path(path: &str) -> Result<()> {
    let trimmed = path.trim();
    if trimmed.is_empty() {
        return Err(Error::validation(
            "sparse checkout path must not be empty".to_owned(),
        ));
    }
    if path.contains('\0') {
        return Err(Error::security(
            "sparse checkout path must not contain NUL bytes".to_owned(),
        ));
    }
    if path.chars().any(char::is_control) {
        return Err(Error::validation(
            "sparse checkout path must not contain control characters".to_owned(),
        ));
    }
    if trimmed.starts_with('/') || trimmed.starts_with('\\') {
        return Err(Error::security(format!(
            "sparse checkout path must be repository-relative, got absolute path `{trimmed}`"
        )));
    }
    if trimmed.starts_with('~') {
        return Err(Error::security(format!(
            "sparse checkout path must not reference a home directory: `{trimmed}`"
        )));
    }
    let mut chars = trimmed.chars();
    if matches!((chars.next(), chars.next()), (Some(first), Some(':')) if first.is_ascii_alphabetic())
    {
        return Err(Error::security(format!(
            "sparse checkout path must not carry a drive prefix: `{trimmed}`"
        )));
    }
    for segment in trimmed.split(['/', '\\']) {
        if segment == "." || segment == ".." {
            return Err(Error::security(format!(
                "sparse checkout path must not contain `{segment}` segments: `{trimmed}`"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use openjiuwen_common::error::ErrorKind;

    #[test]
    fn hostile_paths_are_refused_with_security_errors() {
        for path in [
            "/etc/passwd",
            "\\windows",
            "~/.ssh",
            "C:/repo",
            "C:\\repo",
            "../outside",
            "src/../../etc",
            "src/..",
            "./src",
            "src/\0name",
        ] {
            let error = validate_sparse_path(path).expect_err(path);
            assert_eq!(error.kind(), ErrorKind::Security, "{path}");
        }
    }

    #[test]
    fn empty_and_control_paths_are_validation_errors() {
        for path in ["", "   ", "src/\u{7}name", "a\tb"] {
            let error = validate_sparse_path(path).expect_err(path);
            assert_eq!(error.kind(), ErrorKind::Validation, "{path}");
        }
    }

    #[test]
    fn ordinary_paths_and_patterns_pass() {
        for path in ["src", "src/lib", "docs/", "*.rs", "src/**/mod.rs"] {
            validate_sparse_path(path).expect(path);
        }
    }

    #[test]
    fn config_validation_counts_paths_and_duplicates_are_allowed() {
        let duplicated = SparseConfig::cone(["src", "src", "docs"]).expect("config");
        assert_eq!(duplicated.paths.len(), 3);
        assert_eq!(duplicated.normalized().paths, ["src", "docs"]);

        let too_many: Vec<String> = (0..=MAX_SPARSE_PATHS).map(|i| format!("p{i}")).collect();
        let error = SparseConfig::cone(too_many).expect_err("too many");
        assert_eq!(error.kind(), ErrorKind::Validation);
    }

    #[test]
    fn normalization_trims_and_collapses_separators() {
        let config = SparseConfig {
            paths: vec![
                " src/ ".to_owned(),
                "src".to_owned(),
                "lib\\mod".to_owned(),
                "src//".to_owned(),
                "   ".to_owned(),
                "docs".to_owned(),
            ],
            mode: SparseMode::Patterns,
        };
        let normalized = config.normalized();
        assert_eq!(normalized.paths, ["src", "lib/mod", "docs"]);
        assert_eq!(normalized.mode, SparseMode::Patterns);
    }

    #[test]
    fn config_round_trips_and_tolerates_missing_fields() {
        let config = SparseConfig::patterns(["*.rs", "src/"]).expect("config");
        let json = serde_json::to_string(&config).expect("serialize");
        assert!(json.contains(r#""mode":"patterns""#));
        assert_eq!(
            serde_json::from_str::<SparseConfig>(&json).expect("deserialize"),
            config
        );

        let empty: SparseConfig = serde_json::from_str("{}").expect("deserialize");
        assert_eq!(empty.paths.len(), 0);
        assert!(!SparseConfig::cone(["src"]).expect("config").is_empty());
        assert_eq!(empty.mode, SparseMode::Cone);
        assert_eq!(empty.mode.git_flag(), "--cone");
        assert_eq!(SparseMode::Patterns.git_flag(), "--no-cone");
        assert_eq!(SparseMode::Patterns.as_str(), "patterns");
    }
}
