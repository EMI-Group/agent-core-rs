//! The per-container bookkeeping entry of the sandbox lifecycle.
//!
//! Python parity: `SandboxRecord` (`sandbox/gateway/sandbox_store.py`).
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::sandbox::config::SandboxEndpoint;
use crate::sandbox::launcher::SandboxStatus;

/// The bookkeeping entry of one sandbox container.
///
/// Python parity: `SandboxRecord` (`sandbox/gateway/sandbox_store.py`). Python keys the
/// record by isolation key inside the store; this struct carries the key too, so a record
/// is meaningful on its own. Python's `container_config_hash` and `metadata` are not
/// ported (they served the provider cache, which this rewrite does not keep).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", default)]
pub struct SandboxRecord {
    /// Isolation key the record is stored under.
    pub isolation_key: String,
    /// Runtime-assigned sandbox id, when the launcher reports one.
    pub sandbox_id: Option<String>,
    /// Endpoint handed to the providers.
    pub endpoint: SandboxEndpoint,
    /// Current lifecycle state.
    pub status: SandboxStatus,
    /// Launcher type that created the container.
    pub launcher_type: String,
    /// Creation time, in seconds since the Unix epoch.
    pub created_epoch_secs: u64,
    /// Last time the endpoint was handed out, in seconds since the Unix epoch.
    pub last_used_epoch_secs: u64,
}
