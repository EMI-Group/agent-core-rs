//! Resolution of the vendor CLI executable.
//!
//! A vendor harness needs an executable before it can start, and the failure has to be actionable:
//! the message names the binary, the provider and the fix. An explicitly configured path
//! (`cli_path` for Claude Code and Codex, `dsh_bin` for DSH) wins over `PATH`, and a bad explicit
//! path is an error instead of a silent fallback to whatever `PATH` happens to contain — running a
//! different binary than the operator pinned is worse than not running at all.

use std::path::{Path, PathBuf};

use openjiuwen_harness_protocol::{HarnessError, HarnessResult};

/// How a vendor CLI executable should be located.
pub struct BinarySpec<'a> {
    /// Executable name looked up on `PATH`, for example `claude`.
    pub name: &'a str,
    /// Explicitly configured path (`cli_path` / `dsh_bin`), which wins over `PATH`.
    pub explicit: Option<&'a Path>,
    /// Human-readable harness label used in messages, for example `Claude Code`.
    pub label: &'a str,
    /// Installation hint appended to every failure message.
    pub install_hint: &'a str,
}

/// Resolves the executable described by `spec`.
///
/// An explicit path is validated as-is (it must exist, be a regular file — symlinks are followed —
/// and be executable on Unix); otherwise the binary is looked up on `PATH` with [`which::which`].
///
/// # Errors
///
/// Returns [`HarnessError::Protocol`] naming the binary and the fix when the executable is missing
/// or the explicit path is not executable. It never reports
/// [`HarnessError::UnsupportedCapability`]: a missing CLI is an operator configuration problem, not
/// an unimplemented feature.
pub fn resolve_binary(spec: &BinarySpec<'_>) -> HarnessResult<PathBuf> {
    if let Some(path) = spec.explicit {
        return validate_explicit(spec, path);
    }
    match which::which(spec.name) {
        Ok(path) => Ok(path),
        Err(_) => Err(HarnessError::protocol(not_found_message(spec))),
    }
}

/// Returns whether the executable described by `spec` resolves.
///
/// This is a convenience wrapper over [`resolve_binary`] that discards the reason.
#[must_use]
pub fn is_available(spec: &BinarySpec<'_>) -> bool {
    resolve_binary(spec).is_ok()
}

/// Message used when the binary is absent from `PATH`.
fn not_found_message(spec: &BinarySpec<'_>) -> String {
    format!(
        "{} CLI binary '{}' was not found on PATH: {}; set `cli_path` to an absolute path or add \
         '{}' to PATH",
        spec.label, spec.name, spec.install_hint, spec.name
    )
}

/// Message used when an explicitly configured path cannot be executed.
fn not_executable_message(spec: &BinarySpec<'_>, path: &Path) -> String {
    format!(
        "{} CLI binary '{}' is not executable: {}",
        spec.label,
        path.display(),
        spec.install_hint
    )
}

/// Validates an explicitly configured executable path.
fn validate_explicit(spec: &BinarySpec<'_>, path: &Path) -> HarnessResult<PathBuf> {
    // `metadata` follows symlinks, so a shim pointing at the real installation is accepted.
    let metadata = std::fs::metadata(path)
        .map_err(|_| HarnessError::protocol(not_executable_message(spec, path)))?;
    if !metadata.is_file() || !has_execute_permission(&metadata) {
        return Err(HarnessError::protocol(not_executable_message(spec, path)));
    }
    Ok(path.to_path_buf())
}

/// Returns whether `metadata` carries an execute bit (always true where the mode is unavailable).
#[cfg(unix)]
fn has_execute_permission(metadata: &std::fs::Metadata) -> bool {
    use std::os::unix::fs::PermissionsExt;

    metadata.permissions().mode() & 0o111 != 0
}

/// Returns whether `metadata` carries an execute bit (no mode bits exist off Unix).
#[cfg(not(unix))]
fn has_execute_permission(_metadata: &std::fs::Metadata) -> bool {
    true
}

#[cfg(test)]
mod tests {
    // Tests assert by panicking on failure, which is the intent of a test.
    #![allow(clippy::expect_used, clippy::unwrap_used)]

    use std::path::PathBuf;

    use openjiuwen_harness_protocol::HarnessErrorKind;

    use super::{resolve_binary, BinarySpec};

    /// Spec with the given explicit path.
    fn spec(explicit: Option<&std::path::Path>) -> BinarySpec<'_> {
        BinarySpec {
            name: "openjiuwen-nonexistent-test-cli",
            explicit,
            label: "Test Harness",
            install_hint: "install the test harness",
        }
    }

    /// Writes a file with `mode` into `dir` and returns its path.
    #[cfg(unix)]
    fn write_file(dir: &tempfile::TempDir, name: &str, mode: u32) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;

        let path = dir.path().join(name);
        std::fs::write(&path, "#!/bin/sh\nexit 0\n").expect("write the fake binary");
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(mode))
            .expect("chmod the fake binary");
        path
    }

    #[test]
    fn missing_binary_names_the_binary_and_path() {
        let error = resolve_binary(&spec(None)).expect_err("the name does not exist");
        assert_eq!(error.kind(), HarnessErrorKind::Protocol);
        assert!(error.message().contains("openjiuwen-nonexistent-test-cli"));
        assert!(error.message().contains("not found on PATH"));
        assert!(error.message().contains("install the test harness"));
        assert!(error.message().contains("PATH"));
    }

    #[test]
    fn available_builtin_resolves() {
        // `sh` is guaranteed on every supported test host; an explicit path is not involved.
        let resolved = resolve_binary(&BinarySpec {
            name: "sh",
            explicit: None,
            label: "POSIX shell",
            install_hint: "a POSIX shell is required",
        })
        .expect("sh is on PATH");
        assert!(resolved.is_absolute() || resolved.exists());
    }

    #[cfg(unix)]
    #[test]
    fn explicit_executable_path_wins_over_path() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = write_file(&dir, "fake-cli", 0o755);
        let resolved = resolve_binary(&spec(Some(&path))).expect("an executable file");
        assert_eq!(resolved, path);
    }

    #[cfg(unix)]
    #[test]
    fn explicit_non_executable_path_is_rejected() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = write_file(&dir, "fake-cli", 0o644);
        let error = resolve_binary(&spec(Some(&path))).expect_err("no execute bit");
        assert_eq!(error.kind(), HarnessErrorKind::Protocol);
        assert!(error.message().contains(&path.display().to_string()));
        assert!(error.message().contains("is not executable"));
        assert!(error.message().contains("install the test harness"));
    }

    #[test]
    fn explicit_missing_path_is_rejected() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("absent");
        let error = resolve_binary(&spec(Some(&path))).expect_err("the path does not exist");
        assert!(error.message().contains("is not executable"));
    }

    #[test]
    fn is_available_reflects_resolution() {
        assert!(!super::is_available(&spec(None)));
    }
}
