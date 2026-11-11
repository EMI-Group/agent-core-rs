//! Object storage.
//!
//! Buckets of opaque byte objects with prefix listing and metadata. The local
//! implementation maps buckets to directories under a root path.

pub mod local;
pub mod memory;
