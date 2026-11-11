//! Key-value storage.
//!
//! The trait mirrors the Python `BaseKVStore` contract: string keys, text-or-bytes
//! values, prefix scans, `mget` in input order, batch delete counts, a
//! create-only `exclusive_set` with optional TTL, and a lightweight pipeline.

pub mod file;
pub mod memory;
#[cfg(feature = "sqlite")]
pub mod sqlite;
