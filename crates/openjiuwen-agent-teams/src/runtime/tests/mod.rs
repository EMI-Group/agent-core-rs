//! Crossed-seam integration suite for the team runtime.
//!
//! The per-file unit tests beside `team_agent`, `leader`, `manager` and `backend` pin the
//! behaviour of each file in isolation; this suite crosses the seams instead — a
//! [`TeamRuntimeManager`](crate::runtime::manager::TeamRuntimeManager) that activates a blueprint,
//! the `openjiuwen_agent::Agent` / `openjiuwen_multiagent::Team` adapters over it, the leader run
//! loop, the organisation backend, the in-process transport and the shared store.
//!
//! Everything is offline and deterministic: the harnesses come from
//! [`ScriptedFactory`](crate::runtime::team_agent::test_support::ScriptedFactory), the workspace
//! roots from
//! [`RecordingWorkspaceProvider`](crate::runtime::team_agent::test_support::RecordingWorkspaceProvider)
//! and the store is an `InMemoryTeamStore`. Waits are bounded by
//! [`support::BOUND`] (two seconds); nothing sleeps for longer than a single 150 ms observation
//! window.
//!
//! | Suite | Crossed seams |
//! | --- | --- |
//! | [`end_to_end`] | manager → adapters → leader loop → store, dispatch, global registry |
//! | [`members`] | manager-built team → spawner → harness factory → workspace provider → task board |
//! | [`mail`] | `TeamAgent` mail → backend rows → in-process bus → leader steering |

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod end_to_end;
mod mail;
mod members;
mod support;
