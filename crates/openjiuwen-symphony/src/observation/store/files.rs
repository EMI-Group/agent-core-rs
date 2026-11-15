//! Private filesystem helpers shared by the evidence log and the revision store.
//!
//! Both halves of the observation store write JSON documents below one artifact root, so the
//! atomic write, the immutable-revision rule, the revision-name validation and the path error
//! mapping live here instead of being duplicated. Nothing in this module is public: the
//! stores expose their own contract.

use std::io::ErrorKind as IoErrorKind;
use std::path::Path;

use serde_json::{Map, Value};
use tokio::fs;
use tracing::debug;

use crate::error::{validation, Error, Result};

/// The artifact sub-directory holding the log and every derived revision.
pub(super) const EVOLUTION_DIR: &str = "evolution";

/// The `generated_at` JSON key, which never participates in a revision id.
pub(super) const GENERATED_AT_FIELD: &str = "generated_at";

/// Rejects a revision name that could escape its directory.
///
/// Caller-supplied revisions are only ever used as one path segment, so anything beyond
/// `[A-Za-z0-9._-]` (and the `.`/`..` entries) is a validation error naming the field.
pub(super) fn validate_revision(field: &str, revision: &str) -> Result<()> {
    let legal = !revision.is_empty()
        && revision != "."
        && revision != ".."
        && revision.chars().all(|character| {
            character.is_ascii_alphanumeric() || matches!(character, '.' | '_' | '-')
        });
    if legal {
        Ok(())
    } else {
        Err(validation(
            field,
            format!("`{revision}` is not a legal Symphony revision name"),
        ))
    }
}

/// Reads a JSON object document, or [`None`] when the path does not exist.
pub(super) async fn read_object(path: &Path) -> Result<Option<Map<String, Value>>> {
    let bytes = match fs::read(path).await {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == IoErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(path_error("reading", path, error)),
    };
    let value: Value = serde_json::from_slice(&bytes).map_err(|error| {
        Error::external(error).with_context(format!("decoding `{}`", path.display()))
    })?;
    match value {
        Value::Object(document) => Ok(Some(document)),
        _ => Err(Error::validation(format!(
            "Symphony revision payload must be an object: {}",
            path.display()
        ))),
    }
}

/// Writes an immutable revision document, reusing an identical existing one.
///
/// `generated_at` is metadata, so it is dropped from both sides before the comparison: a
/// republish of the same state succeeds without touching the file, while a *different*
/// document at the same content-derived path is refused.
pub(super) async fn write_immutable_json(path: &Path, document: &Value) -> Result<()> {
    if let Some(existing) = read_object(path).await? {
        if strip_generated_at(&Value::Object(existing)) != strip_generated_at(document) {
            return Err(Error::validation(format!(
                "immutable Symphony revision already exists with different content: {}",
                path.display()
            )));
        }
        return Ok(());
    }
    write_json_atomic(path, document).await
}

/// Returns `value` without its `generated_at` entry.
///
/// `generated_at` is metadata, so two publications of the same state compare equal even
/// though their timestamps differ.
fn strip_generated_at(value: &Value) -> Value {
    let mut value = value.clone();
    if let Value::Object(document) = &mut value {
        document.remove(GENERATED_AT_FIELD);
    }
    value
}

/// Writes `value` to `path` atomically: a `<name>.tmp` sibling, then a rename.
///
/// A reader therefore observes either the previous document or the complete new one. A
/// failed rename removes the temporary file so a retry starts from a clean directory.
pub(super) async fn write_json_atomic(path: &Path, value: &Value) -> Result<()> {
    let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
        return Err(Error::config(format!(
            "Symphony revision path `{}` has no file name",
            path.display()
        )));
    };
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .await
            .map_err(|error| path_error("creating", parent, error))?;
    }
    let temporary = path.with_file_name(format!("{file_name}.tmp"));
    let bytes = serde_json::to_vec_pretty(value).map_err(|error| {
        Error::external(error).with_context(format!("encoding `{}`", path.display()))
    })?;
    fs::write(&temporary, &bytes)
        .await
        .map_err(|error| path_error("writing", &temporary, error))?;
    if let Err(error) = fs::rename(&temporary, path).await {
        if let Err(cleanup) = fs::remove_file(&temporary).await {
            debug!(
                path = %temporary.display(),
                error = %cleanup,
                "failed to remove a temporary Symphony revision"
            );
        }
        return Err(path_error("committing", path, error));
    }
    Ok(())
}

/// Wraps an I/O failure as [`Error::External`], naming the action and the path.
pub(super) fn path_error(action: &str, path: &Path, error: std::io::Error) -> Error {
    Error::external(error).with_context(format!("{action} `{}`", path.display()))
}

/// Builds the error for a pointer that names a document which is not on disk.
pub(super) fn missing_document_error(path: &Path) -> Error {
    let error = std::io::Error::new(
        IoErrorKind::NotFound,
        "Symphony revision document is missing",
    );
    Error::external(error).with_context(format!("reading `{}`", path.display()))
}
