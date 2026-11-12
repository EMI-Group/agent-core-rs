//! Team-scoped memory records, persistence, and projection.
//!
//! [`model`] holds the plain `serde` data of team memory: scopes, kinds, visibility,
//! records, create/update inputs, filters, and result types.
//! [`store`] is the persistence seam — the [`store::TeamMemoryStore`] trait, its in-memory
//! implementation, the [`store::TeamMemory`] facade, and the shared search ranking.
//! A [`store::TeamMemoryStore`] is the single source of truth for team memory: the facade
//! never caches, and deleted rows are gone from every view.
//! [`projection`] converts a team record to and from the unified
//! [`crate::record::MemoryRecord`] so team memory can flow through the same recall and
//! prompt paths as long-term memory — a view derived from the store, never a second copy
//! of it.
pub mod model;
pub mod projection;
pub mod store;
pub use model::*;
pub use projection::*;
pub use store::*;
