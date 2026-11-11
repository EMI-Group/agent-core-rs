//! The in-process checkpoint backend.
//!
//! Ports the storage role of `openjiuwen/core/session/checkpointer/inmemory.py`
//! (v0.1.20) without its workflow/agent lifecycle hooks: checkpoints are kept in
//! a lock-guarded map keyed by session id, then checkpoint id.
use std::collections::BTreeMap;
use std::sync::{Arc, OnceLock};

use async_trait::async_trait;
use openjiuwen_common::error::Result;
use openjiuwen_common::utils::id::{now_rfc3339, uuid_hex};
use parking_lot::RwLock;

use super::{
    validate_checkpoint_id, validate_session_id, Checkpoint, CheckpointMeta, Checkpointer,
};

/// The process-wide instance returned by [`MemoryCheckpointer::shared`].
static SHARED: OnceLock<MemoryCheckpointer> = OnceLock::new();

/// An in-process checkpointer backed by a lock-guarded map.
///
/// Clones share the same storage, so a single instance can be held by several
/// sessions and facades. It is intended for tests and single-process runs;
/// checkpoints do not survive a restart.
#[derive(Debug, Clone, Default)]
pub struct MemoryCheckpointer {
    sessions: Arc<RwLock<BTreeMap<String, BTreeMap<String, Checkpoint>>>>,
}

impl MemoryCheckpointer {
    /// Creates an empty store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the process-wide shared store, creating it on first use.
    ///
    /// The singleton exists for parity with Python's default checkpointer and
    /// is the only documented global in this crate; prefer [`Self::new`] in
    /// tests to keep runs isolated.
    #[must_use]
    pub fn shared() -> Self {
        SHARED.get_or_init(Self::new).clone()
    }

    /// Returns the number of checkpoints stored for `session_id`.
    #[must_use]
    pub fn len(&self, session_id: &str) -> usize {
        self.sessions
            .read()
            .get(session_id)
            .map_or(0, BTreeMap::len)
    }

    /// Returns whether the store holds no checkpoints for any session.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.sessions.read().values().all(BTreeMap::is_empty)
    }

    /// Removes every checkpoint of every session.
    pub fn clear(&self) {
        self.sessions.write().clear();
    }
}

#[async_trait]
impl Checkpointer for MemoryCheckpointer {
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
        self.sessions
            .write()
            .entry(checkpoint.session_id.clone())
            .or_default()
            .insert(checkpoint_id.clone(), checkpoint);
        Ok(checkpoint_id)
    }

    async fn load(&self, session_id: &str, checkpoint_id: &str) -> Result<Option<Checkpoint>> {
        validate_session_id(session_id)?;
        validate_checkpoint_id(checkpoint_id)?;
        let checkpoint = self
            .sessions
            .read()
            .get(session_id)
            .and_then(|checkpoints| checkpoints.get(checkpoint_id))
            .cloned();
        Ok(checkpoint)
    }

    async fn list(&self, session_id: &str) -> Result<Vec<CheckpointMeta>> {
        validate_session_id(session_id)?;
        let mut metas: Vec<CheckpointMeta> = self
            .sessions
            .read()
            .get(session_id)
            .map(|checkpoints| checkpoints.values().map(CheckpointMeta::from).collect())
            .unwrap_or_default();
        metas.sort_by(|left, right| {
            (&left.created_at, &left.checkpoint_id).cmp(&(&right.created_at, &right.checkpoint_id))
        });
        Ok(metas)
    }

    async fn delete(&self, session_id: &str, checkpoint_id: &str) -> Result<()> {
        validate_session_id(session_id)?;
        validate_checkpoint_id(checkpoint_id)?;
        let mut sessions = self.sessions.write();
        if let Some(checkpoints) = sessions.get_mut(session_id) {
            checkpoints.remove(checkpoint_id);
            if checkpoints.is_empty() {
                sessions.remove(session_id);
            }
        }
        Ok(())
    }

    async fn release(&self, session_id: &str) -> Result<()> {
        validate_session_id(session_id)?;
        self.sessions.write().remove(session_id);
        Ok(())
    }

    async fn session_exists(&self, session_id: &str) -> Result<bool> {
        validate_session_id(session_id)?;
        Ok(self
            .sessions
            .read()
            .get(session_id)
            .is_some_and(|checkpoints| !checkpoints.is_empty()))
    }
}
