//! Config, state, task/todo, stop conditions, interaction, loop events, and agent mode schemas.
//!
//! Pure data vocabulary of the harness: serde types, pure helpers and
//! Python-compatible JSON round-trips. No IO, no live handles, no harness-module
//! imports beyond the leaf references `Workspace`, `PermissionsSection` and
//! `PromptMode` (defined in [`mode`], re-exported by `crate::prompts`).
//! See `CONTEXT.md` for the full public contract.

pub mod config;
pub mod interaction;
pub mod loop_event;
pub mod mode;
pub mod state;
pub mod stop_condition;
pub mod task;
