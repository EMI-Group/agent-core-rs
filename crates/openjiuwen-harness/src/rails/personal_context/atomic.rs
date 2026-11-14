//! Atomic JSON file helpers shared by the metadata store, the link book and the
//! entry index.
//!
//! Every document here is small and rewritten whole, so it is written to a
//! sibling `<path>.tmp` first and renamed over the destination: a crash never
//! exposes a partial file. `label` names the document in every error message so
//! call sites stay free of duplicated format strings.
//!
//! `crate::rails::evolution::store` owns the same pattern, but it is private to
//! `evolution`; this module is the PersonalContext-side copy.
use std::path::{Path, PathBuf};

use openjiuwen_common::error::{Error, Result};
use serde::de::DeserializeOwned;
use serde::Serialize;

/// Returns the sibling temporary path used for an atomic write.
pub(crate) fn temp_path(path: &Path) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(".tmp");
    PathBuf::from(name)
}

/// Writes `value` to `path` as pretty JSON plus a trailing newline.
///
/// The document goes to `<path>.tmp` first and is renamed over `path`, so a
/// crash never exposes a partial file. Every failure is an [`Error::config`]:
/// the caller handed us a path we cannot use.
pub(crate) fn write_json_atomic<T>(path: &Path, value: &T, label: &str) -> Result<()>
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
pub(crate) fn read_json_opt<T>(path: &Path, label: &str) -> Result<Option<T>>
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

/// Reads at most `limit` bytes of `path`.
///
/// A missing file yields `None`, so callers can distinguish "absent" from
/// "too large"; exceeding `limit` is an [`Error::validation`]. Used for the
/// bounded read of the agent-use switch and other small control documents.
pub(crate) fn read_bounded(path: &Path, limit: usize, label: &str) -> Result<Option<Vec<u8>>> {
    let metadata = match std::fs::metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(Error::config(format!(
                "failed to read {label} {}: {error}",
                path.display()
            )))
        }
    };
    if !metadata.is_file() {
        return Err(Error::validation(format!(
            "{label} {} is not a regular file",
            path.display()
        )));
    }
    if metadata.len() > limit as u64 {
        return Err(Error::validation(format!(
            "{label} {} exceeds the {limit}-byte limit",
            path.display()
        )));
    }
    std::fs::read(path).map(Some).map_err(|error| {
        Error::config(format!(
            "failed to read {label} {}: {error}",
            path.display()
        ))
    })
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
        assert!(read_bounded(&dir.path().join("absent.yaml"), 16, "switch")
            .unwrap()
            .is_none());
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
    fn bounded_read_enforces_the_limit() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("switch.yaml");
        std::fs::write(&path, b"agent_use_enabled: true\n").unwrap();

        assert_eq!(
            read_bounded(&path, 1024, "switch").unwrap().unwrap().len(),
            24
        );
        let error = read_bounded(&path, 4, "switch").unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
    }

    #[test]
    fn bounded_read_rejects_directories() {
        let dir = tempfile::tempdir().unwrap();
        let error = read_bounded(dir.path(), 1024, "switch").unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
    }

    #[test]
    fn temp_path_appends_the_suffix() {
        assert_eq!(
            temp_path(Path::new("/tmp/personal_context.json")),
            PathBuf::from("/tmp/personal_context.json.tmp")
        );
    }
}
