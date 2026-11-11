//! Document storage.
//!
//! Collections of JSON documents with a small filter language (equality plus
//! `$eq/$ne/$gt/$gte/$lt/$lte/$in/$contains`), ordering, pagination, and counts.
//! The Python `BaseDbStore` returns a SQLAlchemy engine; that is not portable, so
//! this crate exposes a document-shaped contract instead.

pub mod file;
pub mod memory;
#[cfg(feature = "sqlite")]
pub mod sqlite;
