//! Team-scoped memory records, persistence, and projection.
//!
//! [`model`] holds the plain `serde` data of team memory: scopes, kinds, visibility,
//! records, create/update inputs, filters, and result types.
//! [`store`] is the persistence seam — the [`store::TeamMemoryStore`] trait, its in-memory
//! implementation, the [`store::TeamMemory`] facade, and the shared search ranking.
//! A [`store::TeamMemoryStore`] is the single source of truth for team memory: the facade
//! never caches, and deleted rows are gone from every view.
//! The `SQLite` backend in `team::sqlite` is an opt-in feature (`sqlite`): it implements
//! the same trait over a database file, so a facade can be made durable without changing
//! any caller.
//! [`projection`] converts a team record to and from the unified
//! [`crate::record::MemoryRecord`] so team memory can flow through the same recall and
//! prompt paths as long-term memory — a view derived from the store, never a second copy
//! of it.
pub mod extractor;
pub mod hooks;
pub mod markdown;
pub mod model;
pub mod observe;
pub mod projection;
pub mod prompts;
#[cfg(feature = "sqlite")]
pub mod sqlite;
pub mod store;
pub mod toolkit;

pub use model::*;
pub use projection::*;
#[cfg(feature = "sqlite")]
pub use sqlite::*;
pub use store::*;
