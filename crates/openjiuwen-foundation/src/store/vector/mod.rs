//! Vector storage.
//!
//! Collection schemas (`CollectionSchema`/`FieldSchema`), insert-only document
//! adds, and similarity search whose scores are always normalized to
//! "higher is better". The in-memory index is the reference implementation; the
//! local-filesystem store persists the same index as JSON.

pub mod file;
pub mod memory;
