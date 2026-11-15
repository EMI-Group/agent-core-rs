//! Local session version history, refs, rewinds, forks and JSON diffing.
//!
//! Ports the local, JSON-only subset of `openjiuwen/core/session/vcs/`
//! (v0.1.20), built on top of [`crate::session_controller::DiskSessionStore`]:
//! sequential [`Version`] documents under `<session_dir>/history/versions/`,
//! the local refs document ([`Refs`]) under `<session_dir>/history/refs.json`,
//! [`SessionHistory`] commit/list/load/latest/diff/restore/delete/clear,
//! ref management (`set_ref`/`delete_ref`/`resolve_ref`/`list_refs`/`head`),
//! `rewind`/`rewind_hard` ([`RewindOptions`]) and `fork`, plus pure snapshot
//! diffing ([`diff_values`], [`diff_snapshots`]).
//!
//! Refs are local metadata over the version files — `HEAD` is the mutable tip
//! (advanced by every commit, reset by a rewind) and `main` tracks the latest
//! commit — not git objects: names are flat, and no git or external backend is
//! involved.
//!
//! Deferred from the Python package: git and other external backends, the
//! event/WAL (`e<N>`) references, codecs, and context-engine adapters.
//! The local history, its refs, rewinds and forks are self-contained and
//! offline.
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
pub mod refs;
pub mod version;

pub use diff::{diff_snapshots, diff_values, ChangeKind, JsonChange};
pub use history::{RewindOptions, SessionHistory};
pub use refs::{RefEntry, RefName, Refs, HEAD_REF, MAIN_REF, MAX_REF_NAME_LEN};
pub use version::{Version, VersionId, VersionMeta};
