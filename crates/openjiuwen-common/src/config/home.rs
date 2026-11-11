//! `OpenJiuwenHome`: the root directory for `openJiuwen` runtime state.
//!
//! Python resolves its home directory with small ad-hoc helpers (see
//! `openjiuwen/core/common/logging/utils.py`); Rust keeps one type so every
//! subsystem derives logs, sessions, and caches from the same root.
//!
//! Resolution never panics and never creates anything: [`OpenJiuwenHome::resolve`]
//! only computes a path, and [`OpenJiuwenHome::ensure`] is the single place that
//! touches the filesystem.

use std::fmt;
use std::path::{Path, PathBuf};

use crate::config::env;
use crate::error::{Error, Result};

/// Name of the home directory used when `OPENJIUWEN_HOME` is not set.
pub const DEFAULT_DIR_NAME: &str = ".openjiuwen";

/// Environment variable holding the home directory overrides.
const HOME_ENV: &str = "HOME";

/// Windows equivalent of [`HOME_ENV`].
const USERPROFILE_ENV: &str = "USERPROFILE";

/// The `openJiuwen` home directory.
///
/// The value is a plain path: no I/O happens until [`OpenJiuwenHome::ensure`] is
/// called, so constructing a home is cheap and side-effect free.
///
/// ```
/// use openjiuwen_common::config::OpenJiuwenHome;
///
/// let home = OpenJiuwenHome::from_root("/var/lib/openjiuwen");
/// assert_eq!(home.logs_dir().to_str(), Some("/var/lib/openjiuwen/logs"));
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
#[must_use]
pub struct OpenJiuwenHome {
    root: PathBuf,
}

impl OpenJiuwenHome {
    /// Resolves the home directory from the environment.
    ///
    /// Precedence:
    /// 1. `OPENJIUWEN_HOME`, when set to a non-blank value;
    /// 2. `$HOME/.openjiuwen` (`%USERPROFILE%\.openjiuwen` on Windows);
    /// 3. `./.openjiuwen`, a relative fallback used when neither is available.
    ///
    /// The fallback keeps the type total: a container without `HOME` still gets
    /// a usable, if unusual, path instead of a panic.
    pub fn resolve() -> Self {
        if let Some(root) = env::openjiuwen_home() {
            return Self::from_root(root);
        }
        if let Some(home) = std::env::var_os(HOME_ENV).or_else(|| std::env::var_os(USERPROFILE_ENV))
        {
            return Self::from_root(PathBuf::from(home).join(DEFAULT_DIR_NAME));
        }
        Self::from_root(DEFAULT_DIR_NAME)
    }

    /// Builds a home rooted at `path` without touching the environment.
    pub fn from_root(path: impl Into<PathBuf>) -> Self {
        Self { root: path.into() }
    }

    /// Returns the root directory.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Joins a relative path onto the root.
    #[must_use]
    pub fn join(&self, rel: impl AsRef<Path>) -> PathBuf {
        self.root.join(rel)
    }

    /// Returns the log directory (`<root>/logs`).
    #[must_use]
    pub fn logs_dir(&self) -> PathBuf {
        self.join("logs")
    }

    /// Returns the session-state directory (`<root>/sessions`).
    #[must_use]
    pub fn sessions_dir(&self) -> PathBuf {
        self.join("sessions")
    }

    /// Creates the root directory, including missing parents.
    ///
    /// Idempotent: an existing root is success. Nothing is created until this is
    /// called, which keeps read-only commands (for example `openjiuwen --help`)
    /// from writing to disk.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] with the offending path when the directory
    /// cannot be created (permissions, a file occupying the path, a read-only
    /// filesystem).
    pub fn ensure(&self) -> Result<()> {
        std::fs::create_dir_all(&self.root).map_err(|err| {
            Error::config(format!(
                "failed to create openJiuwen home at {}: {err}",
                self.root.display()
            ))
        })
    }
}

impl Default for OpenJiuwenHome {
    fn default() -> Self {
        Self::resolve()
    }
}

impl AsRef<Path> for OpenJiuwenHome {
    fn as_ref(&self) -> &Path {
        self.root()
    }
}

impl fmt::Display for OpenJiuwenHome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.root.display(), f)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::config::test_support::EnvGuard;
    use crate::error::ErrorKind;

    // Python openjiuwen/core/common/logging/utils.py (path resolution helpers).

    #[test]
    fn resolve_honors_openjiuwen_home_over_home() {
        let env = EnvGuard::new(&[env::OPENJIUWEN_HOME, HOME_ENV, USERPROFILE_ENV]);
        env.set(HOME_ENV, "ignored-home");
        env.set(USERPROFILE_ENV, "ignored-profile");
        env.set(env::OPENJIUWEN_HOME, "custom-home");

        assert_eq!(
            OpenJiuwenHome::resolve(),
            OpenJiuwenHome::from_root("custom-home")
        );
        assert_eq!(
            OpenJiuwenHome::default(),
            OpenJiuwenHome::from_root("custom-home")
        );
    }

    #[test]
    fn resolve_falls_back_to_home_directory() {
        let env = EnvGuard::new(&[env::OPENJIUWEN_HOME, HOME_ENV, USERPROFILE_ENV]);
        env.set(HOME_ENV, "home-dir");

        assert_eq!(
            OpenJiuwenHome::resolve(),
            OpenJiuwenHome::from_root(PathBuf::from("home-dir").join(DEFAULT_DIR_NAME))
        );
    }

    #[test]
    fn resolve_falls_back_to_relative_directory() {
        let _guard = EnvGuard::new(&[env::OPENJIUWEN_HOME, HOME_ENV, USERPROFILE_ENV]);

        assert_eq!(
            OpenJiuwenHome::resolve(),
            OpenJiuwenHome::from_root(DEFAULT_DIR_NAME)
        );
        assert_eq!(OpenJiuwenHome::resolve().root(), Path::new(".openjiuwen"));
    }

    #[test]
    fn from_root_join_and_derived_directories() {
        let home = OpenJiuwenHome::from_root("state-root");

        assert_eq!(home.root(), Path::new("state-root"));
        assert_eq!(
            home.join("sessions/a.json"),
            PathBuf::from("state-root/sessions/a.json")
        );
        assert_eq!(home.logs_dir(), PathBuf::from("state-root/logs"));
        assert_eq!(home.sessions_dir(), PathBuf::from("state-root/sessions"));
        assert_eq!(home.as_ref(), Path::new("state-root"));
        assert_eq!(home.to_string(), "state-root");
        assert_eq!(home, OpenJiuwenHome::from_root(PathBuf::from("state-root")));
    }

    #[test]
    fn ensure_creates_the_root_and_is_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let home = OpenJiuwenHome::from_root(dir.path().join("nested/state"));
        assert!(!home.root().exists());

        home.ensure().unwrap();
        assert!(home.root().is_dir());
        home.ensure().unwrap();
        assert!(home.root().is_dir());

        assert!(home.logs_dir().starts_with(dir.path()));
    }

    #[test]
    fn ensure_reports_the_path_when_creation_fails() {
        let dir = tempfile::tempdir().unwrap();
        let blocking_file = dir.path().join("blocking");
        std::fs::write(&blocking_file, b"not a directory").unwrap();

        let home = OpenJiuwenHome::from_root(blocking_file.join("nested"));
        let err = home.ensure().unwrap_err();

        assert_eq!(err.kind(), ErrorKind::Config);
        assert!(err.to_string().contains("openJiuwen home"));
    }
}
