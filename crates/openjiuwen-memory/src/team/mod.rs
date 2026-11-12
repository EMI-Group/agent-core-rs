//! Team-scoped memory records and persistence.
//!
//! [`model`] holds the plain `serde` data of team memory: scopes, kinds, visibility,
//! records, create/update inputs, filters, and result types.
pub mod model;
pub use model::*;
