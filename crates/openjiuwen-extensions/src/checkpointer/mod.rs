//! Checkpointer backend selection for the openJiuwen Rust SDK.
//!
//! Ports the provider split of `openjiuwen/extensions/checkpointer/` (v0.1.20):
//! the in-memory, file, and sqlite backends live in the kernel
//! (`openjiuwen_core::session::checkpoint`), while this module re-exports that
//! surface and dispatches [`build_checkpointer`] to the kernel factory.
//!
//! The Redis backend lives in the [`redis`] submodule behind the `redis`
//! feature. It is API-shaped but deferred: every operation returns
//! [`openjiuwen_common::error::ErrorKind::Unsupported`], and pickled payloads
//! are a deliberate non-goal (see `docs/PARITY.md`).
//!
//! ```
//! use openjiuwen_extensions::checkpointer::{build_checkpointer, CheckpointerConfig,
//!     CheckpointerKind};
//!
//! # async fn example() -> openjiuwen_common::error::Result<()> {
//! let config = CheckpointerConfig {
//!     kind: CheckpointerKind::InMemory,
//!     ..CheckpointerConfig::default()
//! };
//! let checkpointer = build_checkpointer(&config).await?;
//! assert!(!checkpointer.session_exists("empty-session").await?);
//! # Ok(())
//! # }
//! ```
use std::sync::Arc;

use openjiuwen_common::error::Result;

#[cfg(feature = "sqlite")]
pub use openjiuwen_core::session::checkpoint::SqliteCheckpointer;
pub use openjiuwen_core::session::checkpoint::{
    Checkpoint, CheckpointMeta, CheckpointNamespace, Checkpointer, CheckpointerConfig,
    CheckpointerKind, FileCheckpointer, MemoryCheckpointer,
};

#[cfg(feature = "redis")]
pub mod redis;

/// Builds the checkpointer selected by `config.kind`.
///
/// Delegates to [`openjiuwen_core::session::checkpoint::create_checkpointer`]:
/// `InMemory`, `File`, and (with the `sqlite` feature) `Sqlite` are built by the
/// kernel; this module adds no backend of its own outside the deferred Redis
/// surface in [`redis`].
///
/// # Errors
///
/// Returns [`openjiuwen_common::error::ErrorKind::Validation`] when `File` or
/// `Sqlite` is requested without a `path`, and
/// [`openjiuwen_common::error::ErrorKind::Unsupported`] when `Sqlite` is
/// requested but the kernel crate was built without its `sqlite` feature.
pub async fn build_checkpointer(config: &CheckpointerConfig) -> Result<Arc<dyn Checkpointer>> {
    openjiuwen_core::session::checkpoint::create_checkpointer(config).await
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use openjiuwen_common::error::ErrorKind;
    use serde_json::json;

    // port of tests/unit_tests/extensions/checkpointer/test_redis_checkpointer_provider.py
    // (backend selection only; Redis lives in `redis`)

    #[tokio::test]
    async fn in_memory_backend_round_trips_a_checkpoint() {
        let config = CheckpointerConfig {
            kind: CheckpointerKind::InMemory,
            ..CheckpointerConfig::default()
        };
        let checkpointer = build_checkpointer(&config).await.unwrap();

        let checkpoint_id = checkpointer
            .save(
                Checkpoint::new("session-1", CheckpointNamespace::Agent, "agent-1")
                    .with_global_state(json!({"status": "running"})),
            )
            .await
            .unwrap();
        let loaded = checkpointer
            .load("session-1", &checkpoint_id)
            .await
            .unwrap()
            .expect("the checkpoint was just saved");
        assert_eq!(loaded.session_id, "session-1");
        assert_eq!(loaded.entity_id, "agent-1");
        assert_eq!(loaded.global_state, json!({"status": "running"}));
    }

    #[tokio::test]
    async fn file_backend_without_a_path_is_a_validation_error() {
        let config = CheckpointerConfig {
            kind: CheckpointerKind::File,
            ..CheckpointerConfig::default()
        };
        let error = build_checkpointer(&config).await.unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(
            error.to_string().contains("path"),
            "unexpected message: {error}"
        );
    }

    #[tokio::test]
    async fn file_backend_uses_the_configured_directory() {
        let directory = tempfile::tempdir().unwrap();
        let config = CheckpointerConfig {
            kind: CheckpointerKind::File,
            path: Some(directory.path().to_path_buf()),
            ..CheckpointerConfig::default()
        };
        let checkpointer = build_checkpointer(&config).await.unwrap();
        let checkpoint_id = checkpointer
            .save(
                Checkpoint::new("session-2", CheckpointNamespace::Workflow, "flow-1")
                    .with_component_state(json!({"done": true})),
            )
            .await
            .unwrap();
        assert_eq!(
            checkpointer.list("session-2").await.unwrap()[0].checkpoint_id,
            checkpoint_id
        );
    }

    #[cfg(feature = "sqlite")]
    #[tokio::test]
    async fn sqlite_backend_uses_the_configured_database_file() {
        let directory = tempfile::tempdir().unwrap();
        let config = CheckpointerConfig {
            kind: CheckpointerKind::Sqlite,
            path: Some(directory.path().join("checkpoints.db")),
            ..CheckpointerConfig::default()
        };
        let checkpointer = build_checkpointer(&config).await.unwrap();
        let checkpoint_id = checkpointer
            .save(Checkpoint::new(
                "session-3",
                CheckpointNamespace::AgentTeam,
                "team-1",
            ))
            .await
            .unwrap();
        assert!(checkpointer
            .load("session-3", &checkpoint_id)
            .await
            .unwrap()
            .is_some());
    }
}
