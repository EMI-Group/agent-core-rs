//! Atomic JSON file helpers shared by the proposal log and the snapshot writer.
//!
//! Both files are small and rewritten whole, so they are written to a sibling
//! `<path>.tmp` first and renamed over the destination: a crash never exposes a
//! partial document. `label` names the document in every error message, which
//! keeps the two call sites from duplicating the format strings.
use std::path::{Path, PathBuf};

use openjiuwen_common::error::{Error, Result};
use serde::de::DeserializeOwned;
use serde::Serialize;

/// Returns the sibling temporary path used for an atomic write.
pub(super) fn temp_path(path: &Path) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(".tmp");
    PathBuf::from(name)
}

/// Writes `value` to `path` as pretty JSON plus a trailing newline.
///
/// The document goes to `<path>.tmp` first and is renamed over `path`, so a
/// crash never exposes a partial file. Every failure is an
/// [`Error::config`]: the caller handed us a path we cannot use.
pub(super) fn write_json_atomic<T>(path: &Path, value: &T, label: &str) -> Result<()>
where
    T: Serialize + ?Sized,
{
    if let Some(dir) = path.parent() {
        if !dir.as_os_str().is_empty() {
            std::fs::create_dir_all(dir).map_err(|error| {
                Error::config(format!(
                    "failed to create {label} directory {}: {error}",
                    dir.display()
                ))
            })?;
        }
    }
    let mut json = serde_json::to_string_pretty(value).map_err(|error| {
        Error::config(format!(
            "failed to encode {label} file {}: {error}",
            path.display()
        ))
    })?;
    json.push('\n');
    let temp = temp_path(path);
    std::fs::write(&temp, json).map_err(|error| {
        Error::config(format!(
            "failed to write {label} file {}: {error}",
            temp.display()
        ))
    })?;
    std::fs::rename(&temp, path).map_err(|error| {
        Error::config(format!(
            "failed to replace {label} file {}: {error}",
            path.display()
        ))
    })
}

/// Reads and decodes `path`, returning `None` when the file does not exist.
///
/// A malformed document is an [`Error::validation`]; any other IO failure is an
/// [`Error::config`].
pub(super) fn read_json_opt<T>(path: &Path, label: &str) -> Result<Option<T>>
where
    T: DeserializeOwned,
{
    match std::fs::read(path) {
        Ok(bytes) => serde_json::from_slice(&bytes).map(Some).map_err(|error| {
            Error::validation(format!(
                "malformed {label} file {}: {error}",
                path.display()
            ))
        }),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(Error::config(format!(
            "failed to read {label} file {}: {error}",
            path.display()
        ))),
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use openjiuwen_common::error::ErrorKind;
    use serde_json::json;

    use super::*;

    #[test]
    fn write_then_read_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nested").join("doc.json");
        write_json_atomic(&path, &json!({"a": 1}), "test document").unwrap();

        let decoded: Option<serde_json::Value> = read_json_opt(&path, "test document").unwrap();
        assert_eq!(decoded, Some(json!({"a": 1})));
        assert!(!dir.path().join("nested").join("doc.json.tmp").exists());
    }

    #[test]
    fn missing_file_reads_as_none() {
        let dir = tempfile::tempdir().unwrap();
        let decoded: Option<serde_json::Value> =
            read_json_opt(&dir.path().join("absent.json"), "test document").unwrap();
        assert!(decoded.is_none());
    }

    #[test]
    fn malformed_document_is_a_validation_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("doc.json");
        std::fs::write(&path, "{\"a\":").unwrap();

        let error = read_json_opt::<serde_json::Value>(&path, "test document").unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("malformed test document file"));
    }

    #[test]
    fn unwritable_destination_is_a_config_error() {
        let dir = tempfile::tempdir().unwrap();
        let blocker = dir.path().join("blocker");
        std::fs::write(&blocker, "not a directory").unwrap();

        let error =
            write_json_atomic(&blocker.join("doc.json"), &json!({}), "test document").unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Config);
        assert!(error.to_string().contains("test document"));
    }

    #[test]
    fn temp_path_appends_the_suffix() {
        assert_eq!(
            temp_path(Path::new("/tmp/evolution.json")),
            PathBuf::from("/tmp/evolution.json.tmp")
        );
    }
}
