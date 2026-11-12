//! The member seam: the runtime a member's brain implements, the status machine over the roster
//! row, the leader-side activity projection and the in-process spawn handle.
//!
//! Python parity sources: `openjiuwen/agent_teams/agent/member.py`, `agent/member_runtime.py`,
//! `agent/member_activity.py` and `spawn/inprocess_spawn.py` / `spawn/inprocess_handle.py`.
//!
//! Module map:
//!
//! | File | Contents |
//! | --- | --- |
//! | [`runtime`] | [`MemberRuntime`], [`MemberObserver`], [`RecordingObserver`], [`ObserverSet`] |
//! | [`lifecycle`] | [`TeamMember`]: status machine, start/rollback, round notes, retire |
//! | [`activity`] | [`MemberActivityRegistry`], [`IdleDebounce`], [`DEFAULT_IDLE_DEBOUNCE`] |
//! | [`spawn`] | [`MemberSpawner`], [`MemberSpawnRequest`], [`InProcessSpawnHandle`], [`SpawnRegistry`] |
//!
//! The schema types this module's own signatures expose ([`RoundOutcome`], [`MemberStatus`],
//! [`ExecutionStatus`], [`TeamRole`], [`MemberLifetime`]) are re-exported here so callers can drive
//! the member seam through this module alone; `crate::schema` remains their owner.
//!
//! The split follows Python's roles rather than its files: a runtime is the seam the coordination
//! layer drives, a `TeamMember` is the persisted row plus its transition rules, the activity
//! registry is the leader's projection of the whole roster, and the spawn handle owns the task
//! that keeps a teammate alive. Everything here is in-process; OS-process and external-CLI members
//! are deferred and surface through [`MemberSpawner`] as
//! [`openjiuwen_common::error::Error::Unsupported`].
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
pub mod lifecycle;
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod lifecycle_tests;
pub mod runtime;
pub mod spawn;

pub use activity::{IdleDebounce, MemberActivityRegistry, DEFAULT_IDLE_DEBOUNCE};
pub use lifecycle::TeamMember;
pub use runtime::{MemberObserver, MemberRuntime, ObserverEvent, ObserverSet, RecordingObserver};
pub use spawn::{InProcessSpawnHandle, MemberSpawnRequest, MemberSpawner, SpawnRegistry};

// Re-exports of the `schema` types this module's signatures use — the schema module owns them.
pub use crate::schema::{ExecutionStatus, MemberLifetime, MemberStatus, RoundOutcome, TeamRole};
