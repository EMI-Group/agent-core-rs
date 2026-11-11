//! Cross-backend storage contract tests.
//!
//! Every suite in this module asserts one store contract from the crate's public
//! seam (`crate::store::{...}`) against every backend that implements it, so the
//! in-memory reference, the local-filesystem backend, and (behind the `sqlite`
//! feature) the `SQLite` backend cannot drift apart.
//!
//! Each backend test instantiates its store and calls the matching
//! `assert_*_contract` helper exactly once; backend-specific behavior (file
//! persistence, name validation, `SQLite` prefix scans) lives in its own test.

mod document;
mod graph_object;
mod kv;
#[cfg(feature = "sqlite")]
mod sqlite;
mod vector;
