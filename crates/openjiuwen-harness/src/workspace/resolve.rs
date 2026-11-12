//! Sandboxed path resolution for model-supplied workspace paths.
//!
//! [`Workspace::resolve`] is the single gate every harness tool goes through before it
//! touches the filesystem: the path must be lexically relative and safe, and the resolved
//! path must stay inside the workspace root even after symlinks are followed. Both checks
//! reuse [`openjiuwen_sysop::workspace`] so the harness and the sysop backends agree on
//! what "inside the workspace" means.

use std::path::PathBuf;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_sysop::workspace::{canonicalize_containing, is_within};

use crate::workspace::directory_builder::is_safe_relative_path;
use crate::workspace::workspace::Workspace;

impl Workspace {
    /// Resolves a model-supplied relative path to a path inside the workspace.
    ///
    /// The path does not need to exist (write targets are resolved the same way): the
    /// longest existing prefix is canonicalized so a symlink pointing outside the
    /// workspace is rejected.
    ///
    /// # Errors
    ///
    /// - [`ErrorKind::Validation`] when the path is unsafe (empty, absolute, drive-letter,
    ///   UNC, or `..`-escaping).
    /// - [`ErrorKind::Security`] when the resolved path escapes the workspace root.
    pub async fn resolve(&self, path: &str) -> Result<PathBuf> {
        if !is_safe_relative_path(path) {
            return Err(Error::validation(format!("unsafe workspace path: {path}")));
        }
        let root = canonicalize_containing(&self.root_path).await;
        let joined = self.root_path.join(path);
        let resolved = canonicalize_containing(&joined).await;
        if !is_within(&resolved, &root) {
            return Err(Error::security(format!(
                "path escapes workspace root: {path}"
            )));
        }
        Ok(resolved)
    }

    /// Resolves a path like [`Self::resolve`] and requires it to exist.
    ///
    /// # Errors
    ///
    /// Propagates [`Self::resolve`] failures and returns [`ErrorKind::NotFound`] when the
    /// resolved path cannot be stat'ed.
    pub async fn resolve_existing(&self, path: &str) -> Result<PathBuf> {
        let resolved = self.resolve(path).await?;
        tokio::fs::metadata(&resolved)
            .await
            .map_err(|error| Error::not_found(format!("{path}: {error}")))?;
        Ok(resolved)
    }
}
