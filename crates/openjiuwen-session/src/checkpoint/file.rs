//! The filesystem checkpoint backend.
//!
//! Ports the storage role of `openjiuwen/core/session/checkpointer/persistence.py`
//! (v0.1.20) to JSON files instead of Python pickles: each checkpoint is
//! `<root>/<session_id>/<checkpoint_id>.json`, written atomically through a
//! temporary file and a rename. All I/O uses `tokio::fs`, so the backend is safe
//! to call from async tasks.
use std::path::{Path, PathBuf};

use async_trait::async_trait;
use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::utils::id::{now_rfc3339, uuid_hex};
use tokio::fs;

use super::{
    validate_checkpoint_id, validate_session_id, Checkpoint, CheckpointMeta, Checkpointer,
};

/// A checkpointer that stores one JSON document per checkpoint on disk.
///
/// Clones share the same root directory. Reads and writes are crash-safe: a
/// checkpoint is written to a unique temporary file in the session directory and
/// renamed into place, so a failed write never leaves a partial document under
/// the final name.
#[derive(Debug, Clone)]
pub struct FileCheckpointer {
    root: PathBuf,
}

impl FileCheckpointer {
    /// Creates a backend rooted at `root`; the directory is created on first
    /// save.
    #[must_use]
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// Returns the directory holding `session_id`'s checkpoints.
    fn session_dir(&self, session_id: &str) -> PathBuf {
        self.root.join(session_id)
    }

    /// Returns the JSON document path for one checkpoint.
    fn checkpoint_path(&self, session_id: &str, checkpoint_id: &str) -> PathBuf {
        self.session_dir(session_id)
            .join(format!("{checkpoint_id}.json"))
    }

    /// Lists the JSON documents of a session in directory order.
    ///
    /// A missing session directory yields an empty list; entries that are not
    /// regular `*.json` files are ignored.
    async fn json_documents(&self, session_id: &str) -> Result<Vec<PathBuf>> {
        let directory = self.session_dir(session_id);
        let mut entries = match fs::read_dir(&directory).await {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => {
                return Err(io_error("read checkpoint directory", &directory, &error));
            }
        };
        let mut documents = Vec::new();
        while let Some(entry) = entries
            .next_entry()
            .await
            .map_err(|error| io_error("read checkpoint directory", &directory, &error))?
        {
            let path = entry.path();
            if path.extension().and_then(std::ffi::OsStr::to_str) != Some("json") {
                continue;
            }
            let file_type = entry
                .file_type()
                .await
                .map_err(|error| io_error("inspect checkpoint file", &path, &error))?;
            if file_type.is_file() {
                documents.push(path);
            }
        }
        Ok(documents)
    }
}

#[async_trait]
impl Checkpointer for FileCheckpointer {
    async fn save(&self, mut checkpoint: Checkpoint) -> Result<String> {
        validate_session_id(&checkpoint.session_id)?;
        if checkpoint.checkpoint_id.is_empty() {
            checkpoint.checkpoint_id = uuid_hex();
        } else {
            validate_checkpoint_id(&checkpoint.checkpoint_id)?;
        }
        if checkpoint.created_at.is_empty() {
            checkpoint.created_at = now_rfc3339();
        }
        let checkpoint_id = checkpoint.checkpoint_id.clone();
        let directory = self.session_dir(&checkpoint.session_id);
        let path = self.checkpoint_path(&checkpoint.session_id, &checkpoint_id);
        fs::create_dir_all(&directory)
            .await
            .map_err(|error| io_error("create checkpoint directory", &directory, &error))?;
        let bytes = serde_json::to_vec_pretty(&checkpoint).map_err(|error| {
            Error::session(format!(
                "failed to serialize checkpoint {checkpoint_id}: {error}"
            ))
        })?;
        let temporary = directory.join(format!(".{}.tmp", uuid_hex()));
        if let Err(error) = fs::write(&temporary, &bytes).await {
            remove_quietly(&temporary).await;
            return Err(io_error("write checkpoint file", &temporary, &error));
        }
        if let Err(error) = fs::rename(&temporary, &path).await {
            remove_quietly(&temporary).await;
            return Err(io_error("finalize checkpoint file", &path, &error));
        }
        Ok(checkpoint_id)
    }

    async fn load(&self, session_id: &str, checkpoint_id: &str) -> Result<Option<Checkpoint>> {
        validate_session_id(session_id)?;
        validate_checkpoint_id(checkpoint_id)?;
        let path = self.checkpoint_path(session_id, checkpoint_id);
        let bytes = match fs::read(&path).await {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(io_error("read checkpoint file", &path, &error)),
        };
        let checkpoint = serde_json::from_slice(&bytes).map_err(|error| {
            Error::session(format!(
                "corrupt checkpoint file {}: {error}",
                path.display()
            ))
        })?;
        Ok(Some(checkpoint))
    }

    async fn list(&self, session_id: &str) -> Result<Vec<CheckpointMeta>> {
        validate_session_id(session_id)?;
        let mut metas = Vec::new();
        for path in self.json_documents(session_id).await? {
            let bytes = fs::read(&path)
                .await
                .map_err(|error| io_error("read checkpoint file", &path, &error))?;
            let checkpoint: Checkpoint = serde_json::from_slice(&bytes).map_err(|error| {
                Error::session(format!(
                    "corrupt checkpoint file {}: {error}",
                    path.display()
                ))
            })?;
            metas.push(CheckpointMeta::from(&checkpoint));
        }
        metas.sort_by(|left, right| {
            (&left.created_at, &left.checkpoint_id).cmp(&(&right.created_at, &right.checkpoint_id))
        });
        Ok(metas)
    }

    async fn delete(&self, session_id: &str, checkpoint_id: &str) -> Result<()> {
        validate_session_id(session_id)?;
        validate_checkpoint_id(checkpoint_id)?;
        let path = self.checkpoint_path(session_id, checkpoint_id);
        match fs::remove_file(&path).await {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(io_error("delete checkpoint file", &path, &error)),
        }
    }

    async fn release(&self, session_id: &str) -> Result<()> {
        validate_session_id(session_id)?;
        let directory = self.session_dir(session_id);
        match fs::remove_dir_all(&directory).await {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(io_error("delete checkpoint directory", &directory, &error)),
        }
    }

    async fn session_exists(&self, session_id: &str) -> Result<bool> {
        validate_session_id(session_id)?;
        Ok(!self.json_documents(session_id).await?.is_empty())
    }
}

/// Maps an I/O failure to a [`openjiuwen_common::error::ErrorKind::Session`]
/// error naming the operation and path.
fn io_error(action: &str, path: &Path, error: &std::io::Error) -> Error {
    Error::session(format!("failed to {action} {}: {error}", path.display()))
}

/// Best-effort removal of a temporary file after a failed write or rename.
async fn remove_quietly(path: &Path) {
    let _ = fs::remove_file(path).await;
}
