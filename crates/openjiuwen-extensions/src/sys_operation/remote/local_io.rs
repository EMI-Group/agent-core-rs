//! Host-side file transfer helpers for the remote sandbox providers.
//!
//! Python parity: the `download_file`/`upload_file` paths of the providers in
//! `openjiuwen/extensions/sys_operation/sandbox/providers/`, which read and
//! write the caller's local filesystem on one side and the container on the
//! other. Only the host side lives here; the container side is the provider's
//! transport.
//!
//! Everything is `tokio::fs`, so no blocking call runs on an async task.

use std::io::ErrorKind;

/// Why a host-side write failed.
#[derive(Debug)]
pub(crate) enum HostIoError {
    /// The target exists and `overwrite` was `false`.
    AlreadyExists,
    /// Any other filesystem failure, already formatted for the caller.
    Io(String),
}

/// Reads a host file into memory.
///
/// # Errors
///
/// Returns the formatted failure message when the file cannot be read.
pub(crate) async fn read_host_file(path: &str) -> std::result::Result<Vec<u8>, String> {
    tokio::fs::read(path)
        .await
        .map_err(|error| format!("failed to read host file '{path}': {error}"))
}

/// Writes `data` to the host path `path`.
///
/// Parent directories are created when `create_parent_dirs` is set; an existing
/// target is reported as [`HostIoError::AlreadyExists`] unless `overwrite` is
/// set.
///
/// # Errors
///
/// Returns [`HostIoError::AlreadyExists`] for an existing target and
/// [`HostIoError::Io`] for every filesystem failure.
pub(crate) async fn write_host_file(
    path: &str,
    data: &[u8],
    create_parent_dirs: bool,
    overwrite: bool,
) -> std::result::Result<(), HostIoError> {
    let target = std::path::Path::new(path);
    if !overwrite {
        match tokio::fs::metadata(target).await {
            Ok(_) => return Err(HostIoError::AlreadyExists),
            Err(error) if error.kind() == ErrorKind::NotFound => {}
            Err(error) => {
                return Err(HostIoError::Io(format!(
                    "failed to inspect host file '{path}': {error}"
                )))
            }
        }
    }
    if create_parent_dirs {
        if let Some(parent) = target.parent() {
            if !parent.as_os_str().is_empty() {
                tokio::fs::create_dir_all(parent).await.map_err(|error| {
                    HostIoError::Io(format!(
                        "failed to create host directory '{}': {error}",
                        parent.display()
                    ))
                })?;
            }
        }
    }
    tokio::fs::write(target, data)
        .await
        .map_err(|error| HostIoError::Io(format!("failed to write host file '{path}': {error}")))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn write_then_read_round_trips() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("payload.bin");
        let payload = vec![0_u8, 1, 2, 254, 255];
        write_host_file(path.to_str().unwrap(), &payload, false, false)
            .await
            .unwrap();
        assert_eq!(
            read_host_file(path.to_str().unwrap()).await.unwrap(),
            payload
        );
    }

    #[tokio::test]
    async fn an_existing_target_is_reported_unless_overwrite_is_set() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("existing.txt");
        write_host_file(path.to_str().unwrap(), b"one", false, false)
            .await
            .unwrap();

        let error = write_host_file(path.to_str().unwrap(), b"two", false, false)
            .await
            .unwrap_err();
        assert!(matches!(error, HostIoError::AlreadyExists));
        assert_eq!(
            read_host_file(path.to_str().unwrap()).await.unwrap(),
            b"one"
        );

        write_host_file(path.to_str().unwrap(), b"two", false, true)
            .await
            .unwrap();
        assert_eq!(
            read_host_file(path.to_str().unwrap()).await.unwrap(),
            b"two"
        );
    }

    #[tokio::test]
    async fn parent_directories_are_created_on_request() {
        let dir = tempdir().unwrap();
        let nested = dir.path().join("a/b/c/file.txt");
        write_host_file(nested.to_str().unwrap(), b"hi", true, false)
            .await
            .unwrap();
        assert_eq!(
            read_host_file(nested.to_str().unwrap()).await.unwrap(),
            b"hi"
        );
    }

    #[tokio::test]
    async fn a_missing_parent_without_the_flag_is_an_io_error() {
        let dir = tempdir().unwrap();
        let nested = dir.path().join("missing/file.txt");
        let error = write_host_file(nested.to_str().unwrap(), b"hi", false, false)
            .await
            .unwrap_err();
        match error {
            HostIoError::Io(message) => {
                assert!(message.contains("failed to write host file"), "{message}");
            }
            HostIoError::AlreadyExists => panic!("expected an io error"),
        }
    }

    #[tokio::test]
    async fn reading_a_missing_file_reports_the_path() {
        let dir = tempdir().unwrap();
        let missing = dir.path().join("nope.txt");
        let message = read_host_file(missing.to_str().unwrap()).await.unwrap_err();
        assert!(message.contains("failed to read host file"));
        assert!(message.contains("nope.txt"));
    }
}
