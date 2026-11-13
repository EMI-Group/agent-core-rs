//! Local session version history and JSON diffing.
//!
//! Ports the local, JSON-only subset of `openjiuwen/core/session/vcs/`
//! (v0.1.20), built on top of [`crate::session_controller::DiskSessionStore`]:
//! sequential [`Version`] documents under `<session_dir>/history/versions/`,
//! [`SessionHistory`] commit/list/load/latest/diff/restore/delete/clear, and
//! pure snapshot diffing ([`diff_values`], [`diff_snapshots`]).
//!
//! Deferred from the Python package: git and other external backends, the
//! event/WAL (`e<N>`) references, `rewind`/`fork`, codecs, and context-engine
//! adapters. The local history is self-contained and offline.
//!
//! ```
//! use openjiuwen_session::vcs::diff_values;
//! use serde_json::json;
//!
//! let changes = diff_values(&json!({"a": 1}), &json!({"a": 2}));
//! assert_eq!(changes[0].path, "a");
//! ```
pub mod diff;
pub mod history;
pub mod version;

pub use diff::{diff_snapshots, diff_values, ChangeKind, JsonChange};
pub use history::SessionHistory;
pub use version::{Version, VersionId, VersionMeta};
