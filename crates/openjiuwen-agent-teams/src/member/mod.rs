//! The member seam: the runtime a member's brain implements, the status machine over the roster
//! row, the leader-side activity projection, and the spawn seams (in-process task handle plus the
//! three child-process spawners).
//!
//! Python parity sources: `openjiuwen/agent_teams/agent/member.py`, `agent/member_runtime.py`,
//! `agent/member_activity.py`, `spawn/inprocess_spawn.py`, `spawn/inprocess_handle.py`,
//! `spawn/external_cli_spawn.py`, `external/cli_agent/spawn.py` and `schema/ssh_transport.py`.
//!
//! Module map:
//!
//! | File | Contents |
//! | --- | --- |
//! | [`runtime`] | [`MemberRuntime`], [`MemberObserver`], [`RecordingObserver`], [`ObserverSet`] |
//! | [`lifecycle`] | [`TeamMember`]: status machine, start/rollback, round notes, retire |
//! | [`activity`] | [`MemberActivityRegistry`], [`IdleDebounce`], [`DEFAULT_IDLE_DEBOUNCE`] |
//! | [`spawn`] | [`MemberSpawner`], [`MemberSpawnRequest`], [`InProcessSpawnHandle`], [`SpawnRegistry`] |
//! | [`spawn_config`] | [`MemberSpawnKind`], [`MemberSpawnOptions`], [`StdioCommand`], the per-kind launch configs |
//! | [`protocol`] | [`MemberRequest`] / [`MemberResponse`]: the JSON-lines child-process protocol |
//! | [`stdio`] | [`StdioMemberRuntime`]: drives any child process over that protocol |
//! | [`process_spawner`] | [`ProcessMemberSpawner`]: `team-member` child processes |
//! | [`external_cli_spawner`] | [`ExternalCliMemberSpawner`]: third-party CLI child processes |
//! | [`ssh_spawner`] | [`SshMemberSpawner`]: the same protocol over an `ssh` pipe |
//!
//! The schema types this module's own signatures expose ([`RoundOutcome`], [`MemberStatus`],
//! [`ExecutionStatus`], [`TeamRole`], [`MemberLifetime`]) are re-exported here so callers can drive
//! the member seam through this module alone; `crate::schema` remains their owner.
//!
//! The split follows Python's roles rather than its files: a runtime is the seam the coordination
//! layer drives, a `TeamMember` is the persisted row plus its transition rules, the activity
//! registry is the leader's projection of the whole roster, and the spawn handle owns the task
//! that keeps a teammate alive. The default member is still in-process; the three child-process
//! spawners are opt-in through [`MemberSpawnOptions`] and share one JSON-lines member protocol.
//!
//! ```
//! use std::sync::Arc;
//! use openjiuwen_agent_teams::member::TeamMember;
//! use openjiuwen_agent_teams::schema::{MemberStatus, StorageSpec, TeamRole};
//! use openjiuwen_agent_teams::storage::{open_store, MemberRecord};
//!
//! # async fn demo() -> openjiuwen_common::error::Result<()> {
//! let store = open_store(&StorageSpec::in_memory())?;
//! store
//!     .insert_member(MemberRecord::new("alpha", "dev-1", TeamRole::Teammate))
//!     .await?;
//! let member = TeamMember::new(store, MemberRecord::new("alpha", "dev-1", TeamRole::Teammate));
//! member.mark(MemberStatus::Starting).await?;
//! member.mark(MemberStatus::Ready).await?;
//! member.note_round(openjiuwen_agent_teams::schema::RoundOutcome::Started).await?;
//! assert_eq!(member.status(), MemberStatus::Busy);
//! assert_eq!(member.execution(), openjiuwen_agent_teams::schema::ExecutionStatus::Running);
//! # Ok(())
//! # }
//! ```

pub mod activity;
pub mod external_cli_spawner;
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod external_cli_spawner_tests;
pub mod lifecycle;
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod lifecycle_tests;
pub mod process_spawner;
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod process_spawner_tests;
pub mod protocol;
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod protocol_tests;
pub mod runtime;
pub mod spawn;
pub mod spawn_config;
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod spawn_config_tests;
pub mod ssh_spawner;
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod ssh_spawner_tests;
pub mod stdio;
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod stdio_tests;
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, dead_code)]
mod test_child;

pub use activity::{IdleDebounce, MemberActivityRegistry, DEFAULT_IDLE_DEBOUNCE};
pub use external_cli_spawner::ExternalCliMemberSpawner;
pub use lifecycle::TeamMember;
pub use process_spawner::ProcessMemberSpawner;
pub use protocol::{MemberRequest, MemberResponse};
pub use runtime::{MemberObserver, MemberRuntime, ObserverEvent, ObserverSet, RecordingObserver};
pub use spawn::{
    InProcessSpawnHandle, MemberSpawnRequest, MemberSpawner, SpawnRegistry, MEMBER_NAME_ENV,
    MEMBER_WORKSPACE_ENV, SESSION_ID_ENV, TEAM_NAME_ENV,
};
pub use spawn_config::{
    ExternalCliRuntimeConfig, MemberSpawnKind, MemberSpawnOptions, ProcessMemberConfig,
    SshTransportSpec, StdioCommand,
};
pub use ssh_spawner::SshMemberSpawner;
pub use stdio::{StdioMemberRuntime, StdioRuntimeOptions, DEFAULT_IO_TIMEOUT};

// Re-exports of the `schema` types this module's signatures use — the schema module owns them.
pub use crate::schema::{ExecutionStatus, MemberLifetime, MemberStatus, RoundOutcome, TeamRole};
