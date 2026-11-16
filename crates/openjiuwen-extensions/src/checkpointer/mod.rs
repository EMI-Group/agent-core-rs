//! Checkpointer backend selection for the openJiuwen Rust SDK.
//!
//! Ports the provider split of `openjiuwen/extensions/checkpointer/` (v0.1.20):
//! the in-memory, file, and sqlite backends live in the kernel
//! (`openjiuwen_core::session::checkpoint`), while this module re-exports that
//! surface and dispatches [`build_checkpointer`] to the kernel factory.
//!
//! The Redis backend lives in the `redis` submodule behind the `redis` feature.
//! It is a live implementation over this crate's in-tree `RESP` client, which
//! handles `rediss://` (TLS) and cluster redirection. Two Redis backends are
//! therefore reachable: [`build_checkpointer`] forwards `CheckpointerKind::Redis`
//! to the kernel factory (the kernel's own Redis checkpointer, with the URL read
//! from `extra["url"]` or `path`), while this crate's backend over the in-tree
//! client is constructed directly with
//! `checkpointer::redis::RedisCheckpointer::new` (lazy connect) or
//! `connect`/`create` (eager connect).
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
/// `InMemory`, `File`, `Redis`, and (with the `sqlite` feature) `Sqlite` are
/// built by the kernel. The Redis backend under `redis` is this crate's own
/// implementation over the in-tree `RESP` client and is reached through
/// `RedisCheckpointer::connect`/`create` instead, because the kernel factory
/// builds the kernel's Redis checkpointer for `CheckpointerKind::Redis`.
///
/// # Errors
///
/// Returns [`openjiuwen_common::error::ErrorKind::Validation`] when `File` or
/// `Sqlite` is requested without a `path`, or `Redis` without a URL (in
/// `extra["url"]` or `path`), and
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
