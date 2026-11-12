//! Per-session todo persistence: the `todo.json` layout, safe session keys and the locks.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde_json::Value;
use tokio::sync::{Mutex, OwnedMutexGuard};

use openjiuwen_common::error::Result;
use openjiuwen_foundation::tool::ToolContext;

use crate::schema::TodoItem;
use crate::workspace::{Workspace, WorkspaceNode};

/// Persisted file name inside each session directory.
const TODO_FILE: &str = "todo.json";
/// Session key used when the invocation has no usable session id.
const DEFAULT_SESSION: &str = "default";

/// Whether `session` is safe to use as a single path component.
fn is_safe_session(session: &str) -> bool {
    !session.is_empty() && session != ".." && !session.contains(['/', '\\', '\0'])
}

/// Maps a raw session id to a safe single path component.
fn normalize_session(session: &str) -> &str {
    if is_safe_session(session) {
        session
    } else {
        DEFAULT_SESSION
    }
}

/// Returns the session key of `ctx`, falling back to `default` for an absent or unsafe id.
#[must_use]
pub fn session_key(ctx: &ToolContext) -> String {
    ctx.session_id.as_deref().map_or_else(
        || DEFAULT_SESSION.to_owned(),
        |session| normalize_session(session).to_owned(),
    )
}

/// Per-session todo persistence with one async lock per session.
///
/// The store owns no process-global state: each registry (or test) builds its own instance,
/// and only the files under [`TodoStore::dir`] are shared.
pub struct TodoStore {
    dir: PathBuf,
    locks: Mutex<HashMap<String, Arc<Mutex<()>>>>,
}

impl TodoStore {
    /// Builds a store rooted at `dir`.
    #[must_use]
    pub fn new(dir: impl Into<PathBuf>) -> Arc<Self> {
        Arc::new(Self {
            dir: dir.into(),
            locks: Mutex::new(HashMap::new()),
        })
    }

    /// Builds a store over the workspace `todo` node, falling back to `<root_path>/todo`.
    #[must_use]
    pub fn for_workspace(workspace: &Workspace) -> Arc<Self> {
        let dir = workspace
            .get_node_path_for(WorkspaceNode::Todo)
            .unwrap_or_else(|| workspace.root_path.join(WorkspaceNode::Todo.as_str()));
        Self::new(dir)
    }

    /// Returns the root directory holding the per-session files.
    #[must_use]
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Returns `<dir>/<session>/todo.json`; an unsafe session id is mapped to `default`, so no
    /// caller can escape [`Self::dir`].
    #[must_use]
    pub fn path_for(&self, session: &str) -> PathBuf {
        self.dir().join(normalize_session(session)).join(TODO_FILE)
    }

    /// Loads the todo items of `session`; a missing file yields an empty list.
    ///
    /// # Errors
    ///
    /// Returns an error when the file exists but cannot be read, is not a JSON array, or holds
    /// an item that does not deserialize as a [`TodoItem`].
    pub async fn load(&self, session: &str) -> Result<Vec<TodoItem>> {
        let path = self.path_for(session);
        let bytes = match tokio::fs::read(&path).await {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => return Err(error.into()),
        };
        let items: Vec<Value> = serde_json::from_slice(&bytes)?;
        items.iter().map(TodoItem::from_dict).collect()
    }

    /// Replaces the persisted todo list of `session` through an atomic rename.
    ///
    /// # Errors
    ///
    /// Returns an error when the session directory cannot be created, the items cannot be
    /// serialized, or the temporary file cannot be written or renamed.
    pub async fn save(&self, session: &str, items: &[TodoItem]) -> Result<()> {
        let path = self.path_for(session);
        let dir = path
            .parent()
            .map_or_else(|| self.dir.clone(), Path::to_path_buf);
        tokio::fs::create_dir_all(&dir).await?;
        let payload = Value::Array(items.iter().map(TodoItem::to_dict).collect());
        let rendered = serde_json::to_string_pretty(&payload)?;
        let temporary = dir.join(format!("{TODO_FILE}.tmp-{}", uuid::Uuid::new_v4()));
        tokio::fs::write(&temporary, rendered.as_bytes()).await?;
        if let Err(error) = tokio::fs::rename(&temporary, &path).await {
            // Windows cannot replace the target in place; remove it and retry once.
            tracing::debug!(%error, path = %path.display(), "retrying todo rename");
            let _ = tokio::fs::remove_file(&path).await;
            if let Err(error) = tokio::fs::rename(&temporary, &path).await {
                let _ = tokio::fs::remove_file(&temporary).await;
                return Err(error.into());
            }
        }
        Ok(())
    }

    /// Acquires the operation lock of `session`; hold the guard for a whole read-modify-write
    /// span. The lock map is never held across an `.await`.
    pub async fn lock(&self, session: &str) -> OwnedMutexGuard<()> {
        let mutex = {
            let mut locks = self.locks.lock().await;
            Arc::clone(
                locks
                    .entry(normalize_session(session).to_owned())
                    .or_insert_with(|| Arc::new(Mutex::new(()))),
            )
        };
        mutex.lock_owned().await
    }
}
