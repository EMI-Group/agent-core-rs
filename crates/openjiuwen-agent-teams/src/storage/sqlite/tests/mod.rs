//! SQLite store suite: backend parity, persistence, races, migrations and `open_store` dispatch.
//!
//! Everything runs offline and deterministically: file databases live in `tempfile::tempdir()`,
//! no test sleeps, ignores or depends on wall-clock ordering (only on stamps strictly growing).
//!
//! `contract.rs` holds the backend-independent sequence and `parity.rs` runs it over both
//! `InMemoryTeamStore` and `SqliteTeamStore`; the remaining files cover what only a real database
//! can show — reopen persistence, concurrent compare-and-set, the schema-version guard and the
//! `StorageSpec` dispatch.

#![allow(clippy::unwrap_used, clippy::expect_used)]

mod contract;
mod migration;
mod open_store;
mod parity;
mod persistence;
mod races;
mod support;
